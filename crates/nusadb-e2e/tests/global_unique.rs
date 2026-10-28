//! Unique keys of a partitioned table that leave out a partitioning column, over the wire: no
//! duplicate commits across partitions with concurrent writers into different partitions at every
//! isolation level, a `REPEATABLE READ` transaction that began before a partition was created and
//! filled, `ATTACH PARTITION` racing writers, `COPY`, a restart, deferrable keys, and a stress loop
//! of writers that insert, move and swap keys across partitions.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "integration test harness asserts via unwrap/panic/indexing over linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, TxnStatus, handle_client};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn next(conn: &mut Conn) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Connect and consume the handshake up to the first `ReadyForQuery`.
async fn open(engine: &Arc<dyn StorageEngine>) -> Conn {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 3,
            user: "nusadb-root".to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    while !matches!(next(&mut conn).await, BackendMessage::ReadyForQuery(_)) {}
    conn
}

/// Render one backend message: `C:<tag>`, `E:<sqlstate>:<message>`, `D:<fields>`, `Z:<I|T|E>`.
fn render(msg: &BackendMessage) -> String {
    match msg {
        BackendMessage::CommandComplete { tag } => format!("C:{tag}"),
        BackendMessage::Error { code, message } => format!("E:{code}:{message}"),
        BackendMessage::DataRow { values } => {
            let fields: Vec<String> = values
                .iter()
                .map(|v| {
                    v.as_ref().map_or_else(
                        || "NULL".to_owned(),
                        |b| String::from_utf8_lossy(b).into_owned(),
                    )
                })
                .collect();
            format!("D:{}", fields.join(","))
        },
        BackendMessage::ReadyForQuery(status) => match status {
            TxnStatus::Idle => "Z:I".to_owned(),
            TxnStatus::InTransaction => "Z:T".to_owned(),
            TxnStatus::Failed => "Z:E".to_owned(),
        },
        other => format!("{other:?}"),
    }
}

async fn until_ready(conn: &mut Conn) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(render(&msg));
        if done {
            return out;
        }
    }
}

async fn q(conn: &mut Conn, sql: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    until_ready(conn).await
}

#[allow(dead_code, reason = "shared helper set")]
/// Run `sql` through the extended protocol (Parse, Bind, Execute, Sync) without parameters.
async fn extended(conn: &mut Conn, sql: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Parse {
            name: String::new(),
            sql: sql.to_owned(),
            param_types: Vec::new(),
        },
    )
    .await;
    send(
        conn,
        FrontendMessage::Bind {
            portal: String::new(),
            statement: String::new(),
            params: Vec::new(),
            result_formats: Vec::new(),
        },
    )
    .await;
    send(
        conn,
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 0,
        },
    )
    .await;
    send(conn, FrontendMessage::Sync).await;
    until_ready(conn).await
}

/// `COPY <table> FROM STDIN` with `data`.
async fn copy_in(conn: &mut Conn, sql: &str, data: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    loop {
        let msg = next(conn).await;
        if matches!(msg, BackendMessage::CopyInResponse { .. }) {
            break;
        }
        assert!(!matches!(msg, BackendMessage::Error { .. }), "{msg:?}");
    }
    send(
        conn,
        FrontendMessage::CopyData {
            data: data.as_bytes().to_vec(),
        },
    )
    .await;
    send(conn, FrontendMessage::CopyDone).await;
    until_ready(conn).await
}

/// The SQLSTATE of the first error in a response, if any.
fn error_code(out: &[String]) -> Option<String> {
    out.iter()
        .find_map(|m| m.strip_prefix("E:"))
        .map(|rest| rest.chars().take(5).collect())
}

/// The final transaction status of a response.
#[allow(dead_code, reason = "shared helper set")]
fn status(out: &[String]) -> &str {
    out.last().map_or("", String::as_str)
}

/// Run `sql` and require it to succeed.
async fn ok(conn: &mut Conn, sql: &str) {
    let out = q(conn, sql).await;
    assert_eq!(error_code(&out), None, "{sql}: {out:?}");
}

/// Run `sql` and require it to fail with `code`.
async fn fails(conn: &mut Conn, sql: &str, code: &str) -> Vec<String> {
    let out = q(conn, sql).await;
    assert_eq!(error_code(&out).as_deref(), Some(code), "{sql}: {out:?}");
    out
}

/// Run `sql` and return its single value.
async fn scalar(conn: &mut Conn, sql: &str) -> String {
    let out = q(conn, sql).await;
    out.iter()
        .find_map(|m| m.strip_prefix("D:"))
        .unwrap_or_else(|| panic!("{sql}: no row in {out:?}"))
        .to_owned()
}

fn memory_engine() -> Arc<dyn StorageEngine> {
    Arc::new(BtreeEngine::new())
}

const LEVELS: [&str; 3] = ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"];

/// A table with a key that leaves out the partition key, over four partitions.
async fn schema(c: &mut Conn) {
    let out = q(
        c,
        "CREATE TABLE ev (id int NOT NULL, d int NOT NULL, v text, CONSTRAINT ev_pk PRIMARY KEY (id)) \
         PARTITION BY RANGE (d)",
    )
    .await;
    assert_eq!(error_code(&out), None, "{out:?}");
    // Creating the key says what it costs.
    assert!(
        out.iter().any(|m| m.contains("probes every partition")),
        "no notice: {out:?}"
    );
    for (i, lo) in [0, 100, 200, 300].iter().enumerate() {
        ok(
            c,
            &format!(
                "CREATE TABLE ev_{i} PARTITION OF ev FOR VALUES FROM ({lo}) TO ({})",
                lo + 100
            ),
        )
        .await;
    }
}

async fn duplicates(c: &mut Conn) -> String {
    scalar(
        c,
        "SELECT count(*) FROM (SELECT id FROM ev GROUP BY id HAVING count(*) > 1) d",
    )
    .await
}

#[tokio::test]
async fn writers_into_different_partitions_never_both_commit_a_key() {
    for level in LEVELS {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        schema(&mut a).await;

        // Both insert key 5, into different partitions, both through the parent and directly.
        for (sa, sb) in [
            (
                "INSERT INTO ev VALUES (5, 10, 'a')",
                "INSERT INTO ev VALUES (5, 110, 'b')",
            ),
            (
                "INSERT INTO ev_2 VALUES (6, 210, 'a')",
                "INSERT INTO ev_3 VALUES (6, 310, 'b')",
            ),
        ] {
            ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
            ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
            assert_eq!(scalar(&mut b, "SELECT count(*) FROM ev").await.len(), 1);
            ok(&mut a, sa).await;
            let wb = q(&mut b, sb).await;
            assert_eq!(error_code(&wb).as_deref(), Some("40001"), "{level}: {wb:?}");
            ok(&mut b, "ROLLBACK").await;
            ok(&mut a, "COMMIT").await;
        }
        // After A committed, B sees the key, whatever its snapshot.
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        fails(&mut b, "INSERT INTO ev VALUES (5, 250, 'late')", "23505").await;
        ok(&mut b, "ROLLBACK").await;

        // A moves its key to another partition while B tries to take it in a third.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "UPDATE ev SET d = 150 WHERE id = 5").await;
        let wb = q(&mut b, "INSERT INTO ev VALUES (5, 350, 'x')").await;
        assert!(error_code(&wb).is_some(), "{level}: {wb:?}");
        ok(&mut a, "COMMIT").await;

        // A swaps two keys across partitions in one statement while B inserts one of them.
        ok(&mut a, "INSERT INTO ev VALUES (7, 20, 's'), (8, 220, 't')").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(
            &mut a,
            "UPDATE ev SET id = CASE id WHEN 7 THEN 8 ELSE 7 END WHERE id IN (7, 8)",
        )
        .await;
        let wb = q(&mut b, "INSERT INTO ev VALUES (7, 320, 'intruder')").await;
        assert!(error_code(&wb).is_some(), "{level}: {wb:?}");
        ok(&mut a, "COMMIT").await;
        assert_eq!(
            scalar(
                &mut a,
                "SELECT string_agg(id || v, ',' ORDER BY id) FROM ev WHERE id IN (7, 8)"
            )
            .await,
            "7t,8s",
            "{level}"
        );
        assert_eq!(duplicates(&mut a).await, "0", "{level}");
    }
}

#[tokio::test]
async fn a_repeatable_read_transaction_sees_a_partition_created_after_it_began() {
    for level in ["REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        schema(&mut a).await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(scalar(&mut a, "SELECT count(*) FROM ev").await, "0");
        // B creates and fills a partition A's snapshot does not know.
        ok(
            &mut b,
            "CREATE TABLE ev_4 PARTITION OF ev FOR VALUES FROM (400) TO (500)",
        )
        .await;
        ok(&mut b, "INSERT INTO ev VALUES (1, 450, 'b')").await;
        // And attaches a populated table.
        ok(
            &mut b,
            "CREATE TABLE ev_5 (id int NOT NULL, d int NOT NULL, v text)",
        )
        .await;
        ok(&mut b, "INSERT INTO ev_5 VALUES (2, 550, 'b')").await;
        ok(
            &mut b,
            "ALTER TABLE ev ATTACH PARTITION ev_5 FOR VALUES FROM (500) TO (600)",
        )
        .await;
        fails(&mut a, "INSERT INTO ev VALUES (1, 10, 'a')", "23505").await;
        ok(&mut a, "ROLLBACK").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        fails(&mut a, "INSERT INTO ev_0 VALUES (2, 10, 'a')", "23505").await;
        ok(&mut a, "ROLLBACK").await;
        assert_eq!(duplicates(&mut a).await, "0", "{level}");
    }
}

#[tokio::test]
async fn attach_partition_races_writers() {
    let engine = memory_engine();
    let mut a = open(&engine).await;
    let mut b = open(&engine).await;
    schema(&mut a).await;
    ok(
        &mut b,
        "CREATE TABLE x (id int NOT NULL, d int NOT NULL, v text)",
    )
    .await;
    ok(&mut b, "INSERT INTO x VALUES (9, 450, 'x')").await;

    // A writes key 9 and has not committed: the attach of a table holding 9 is refused.
    ok(&mut a, "BEGIN").await;
    ok(&mut a, "INSERT INTO ev VALUES (9, 10, 'a')").await;
    fails(
        &mut b,
        "ALTER TABLE ev ATTACH PARTITION x FOR VALUES FROM (400) TO (500)",
        "40001",
    )
    .await;
    ok(&mut a, "COMMIT").await;
    // Once A committed, the attach sees the duplicate.
    fails(
        &mut b,
        "ALTER TABLE ev ATTACH PARTITION x FOR VALUES FROM (400) TO (500)",
        "23505",
    )
    .await;

    // The attach holds its keys until it ends: a writer of one of them is refused meanwhile.
    ok(&mut b, "UPDATE x SET id = 10").await;
    ok(&mut b, "BEGIN").await;
    ok(
        &mut b,
        "ALTER TABLE ev ATTACH PARTITION x FOR VALUES FROM (400) TO (500)",
    )
    .await;
    let wa = q(&mut a, "INSERT INTO ev VALUES (10, 20, 'a')").await;
    assert_eq!(error_code(&wa).as_deref(), Some("40001"), "{wa:?}");
    ok(&mut b, "COMMIT").await;
    fails(&mut a, "INSERT INTO ev VALUES (10, 20, 'a')", "23505").await;
    assert_eq!(duplicates(&mut a).await, "0");
}

#[tokio::test]
async fn copy_and_extended_query_are_checked() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    schema(&mut c).await;
    ok(&mut c, "INSERT INTO ev VALUES (1, 10, 'a')").await;
    let out = copy_in(&mut c, "COPY ev FROM STDIN", "2\t110\tb\n1\t210\tdup\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM ev").await, "1");
    let out = copy_in(&mut c, "COPY ev FROM STDIN", "2\t110\tb\n3\t210\tc\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    let out = copy_in(&mut c, "COPY ev_3 FROM STDIN", "3\t310\tdup\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    // CHAR(n) keys: trailing blanks are not significant, in a load as anywhere.
    ok(
        &mut c,
        "CREATE TABLE ck (k char(3), d int NOT NULL, UNIQUE (k)) PARTITION BY LIST (d)",
    )
    .await;
    ok(
        &mut c,
        "CREATE TABLE ck_1 PARTITION OF ck FOR VALUES IN (1)",
    )
    .await;
    ok(
        &mut c,
        "CREATE TABLE ck_2 PARTITION OF ck FOR VALUES IN (2)",
    )
    .await;
    let out = copy_in(&mut c, "COPY ck FROM STDIN", "ab \t1\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    let out = copy_in(&mut c, "COPY ck FROM STDIN", "ab \t2\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    let out = copy_in(&mut c, "COPY ck FROM STDIN", "ab\t2\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    let out = copy_in(&mut c, "COPY ck_2 FROM STDIN", "ab  \t2\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM ck").await, "1");
    let out = extended(&mut c, "INSERT INTO ev VALUES (2, 320, 'dup')").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM ev").await, "3");
}

#[tokio::test]
async fn the_key_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        schema(&mut c).await;
        ok(
            &mut c,
            "CREATE TABLE dk (k int NOT NULL, d int NOT NULL, CONSTRAINT dk_k UNIQUE (k) \
             DEFERRABLE INITIALLY DEFERRED) PARTITION BY RANGE (d)",
        )
        .await;
        ok(
            &mut c,
            "CREATE TABLE dk_0 PARTITION OF dk FOR VALUES FROM (0) TO (100)",
        )
        .await;
        ok(
            &mut c,
            "CREATE TABLE dk_1 PARTITION OF dk FOR VALUES FROM (100) TO (200)",
        )
        .await;
        ok(&mut c, "INSERT INTO ev VALUES (1, 10, 'a'), (2, 110, 'b')").await;
        ok(&mut c, "INSERT INTO dk VALUES (1, 10), (2, 110)").await;
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    fails(&mut c, "INSERT INTO ev VALUES (1, 210, 'dup')", "23505").await;
    fails(&mut c, "INSERT INTO ev_3 VALUES (2, 310, 'dup')", "23505").await;
    // The deferrable key: a swap across partitions in two statements commits ...
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "UPDATE dk SET k = 2 WHERE d = 10").await;
    ok(&mut c, "UPDATE dk SET k = 1 WHERE d = 110").await;
    ok(&mut c, "COMMIT").await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(k || ':' || d, ',' ORDER BY k) FROM dk"
        )
        .await,
        "1:110,2:10"
    );
    // ... a duplicate left at COMMIT does not, and IMMEDIATE checks it at once.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO dk VALUES (1, 20)").await;
    fails(&mut c, "COMMIT", "23505").await;
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO dk VALUES (2, 120)").await;
    fails(&mut c, "SET CONSTRAINTS dk_k IMMEDIATE", "23505").await;
    ok(&mut c, "ROLLBACK").await;
    // Two connections writing one deferred key into two partitions: never both.
    let mut b = open(&engine).await;
    ok(&mut c, "BEGIN").await;
    ok(&mut b, "BEGIN").await;
    ok(&mut c, "INSERT INTO dk VALUES (5, 30)").await;
    let wb = q(&mut b, "INSERT INTO dk VALUES (5, 130)").await;
    assert_eq!(error_code(&wb).as_deref(), Some("40001"), "{wb:?}");
    ok(&mut b, "ROLLBACK").await;
    ok(&mut c, "COMMIT").await;
    assert_eq!(
        scalar(&mut c, "SELECT count(*) FROM dk WHERE k = 5").await,
        "1"
    );
}

/// Many connections insert, move, swap and delete a few keys across the partitions at every
/// isolation level. Every refusal is 40001 or 23505, and the committed state never holds a key
/// twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_stress_loop_never_commits_a_duplicate_across_partitions() {
    let engine = memory_engine();
    let mut setup = open(&engine).await;
    schema(&mut setup).await;
    for id in 0..8 {
        ok(
            &mut setup,
            &format!("INSERT INTO ev VALUES ({id}, {}, 'seed')", id * 50),
        )
        .await;
    }
    let mut tasks = Vec::new();
    for worker in 0..12_u32 {
        let engine = Arc::clone(&engine);
        tasks.push(tokio::spawn(async move {
            let mut c = open(&engine).await;
            let mut committed = 0_u32;
            let mut state = 0x9e37_79b9_u32.wrapping_mul(worker + 1);
            let mut rand = move |n: u32| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state % n
            };
            for i in 0..200_u32 {
                let level = LEVELS[(worker as usize + i as usize) % LEVELS.len()];
                let x = rand(12);
                let y = rand(12);
                let d = rand(400);
                let steps: Vec<String> = match i % 6 {
                    0 => vec![format!("INSERT INTO ev VALUES ({x}, {d}, 'w{worker}')")],
                    1 => vec![format!("UPDATE ev SET d = {d} WHERE id = {x}")],
                    2 => vec![format!(
                        "UPDATE ev SET id = CASE id WHEN {x} THEN {y} ELSE {x} END WHERE id IN ({x}, {y})"
                    )],
                    3 => vec![format!("DELETE FROM ev WHERE id = {x}")],
                    4 => vec![format!(
                        "INSERT INTO ev_{} VALUES ({x}, {}, 'direct')",
                        d / 100,
                        d
                    )],
                    _ => vec![
                        format!("UPDATE ev SET id = id + 100 WHERE id = {x}"),
                        format!("UPDATE ev SET id = {x} WHERE id = {y}"),
                        format!("UPDATE ev SET id = {y} WHERE id = {x} + 100"),
                    ],
                };
                let begin = q(&mut c, &format!("BEGIN ISOLATION LEVEL {level}")).await;
                assert_eq!(error_code(&begin), None, "{begin:?}");
                let mut failed = false;
                for sql in &steps {
                    let out = q(&mut c, sql).await;
                    if let Some(code) = error_code(&out) {
                        assert!(
                            matches!(code.as_str(), "40001" | "23505"),
                            "{sql}: {out:?}"
                        );
                        failed = true;
                        break;
                    }
                }
                if failed {
                    q(&mut c, "ROLLBACK").await;
                    continue;
                }
                let commit = q(&mut c, "COMMIT").await;
                match error_code(&commit).as_deref() {
                    None => committed += 1,
                    Some("40001" | "23505") => {},
                    Some(_) => panic!("{commit:?}"),
                }
            }
            committed
        }));
    }
    let mut committed = 0;
    for t in tasks {
        committed += t.await.unwrap();
    }
    assert!(committed > 0, "no writer ever committed");
    assert_eq!(duplicates(&mut setup).await, "0");
    // Every row sits in the partition its bound names.
    assert_eq!(
        scalar(
            &mut setup,
            "SELECT (SELECT count(*) FROM ev_0 WHERE d >= 100) + (SELECT count(*) FROM ev_1 WHERE d < 100 OR d >= 200) \
             + (SELECT count(*) FROM ev_2 WHERE d < 200 OR d >= 300) + (SELECT count(*) FROM ev_3 WHERE d < 300)"
        )
        .await,
        "0"
    );
}
