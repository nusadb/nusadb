//! Exclusion constraints over the wire: the error code and message, COPY, upsert, restart
//! persistence, isolation from uncommitted DDL, and concurrent writers at every isolation level
//! (two connections, then a stress loop of many).

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
            minor: 0,
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

/// The SQLSTATE of the first error in a response, if any.
fn error_code(out: &[String]) -> Option<String> {
    out.iter()
        .find_map(|m| m.strip_prefix("E:"))
        .map(|rest| rest.chars().take(5).collect())
}

/// Run `sql` and require it to succeed.
async fn ok(conn: &mut Conn, sql: &str) {
    let out = q(conn, sql).await;
    assert_eq!(error_code(&out), None, "{sql}: {out:?}");
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

const HIST: &str = "CREATE TABLE hist (hist_id uuid PRIMARY KEY, k int NOT NULL, \
    valid_from timestamptz NOT NULL, valid_to timestamptz, CONSTRAINT hist_no_overlap \
    EXCLUDE USING gist (k WITH =, tstzrange(valid_from, valid_to) WITH &&))";

#[tokio::test]
async fn a_violation_is_23p01_and_names_the_constraint_and_both_keys() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, HIST).await;
    ok(
        &mut c,
        "INSERT INTO hist VALUES (gen_random_uuid(), 1, '2024-01-01 00:00:00+00', NULL)",
    )
    .await;
    let out = q(
        &mut c,
        "INSERT INTO hist VALUES (gen_random_uuid(), 1, '2024-03-01 00:00:00+00', \
         '2024-04-01 00:00:00+00')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    let message = out.iter().find(|m| m.starts_with("E:")).unwrap();
    assert!(message.contains("\"hist_no_overlap\""), "{message}");
    assert!(
        message.contains("(k, tstzrange(valid_from, valid_to))=(1, [\"2024-03-01"),
        "{message}"
    );
    assert!(message.contains("existing key"), "{message}");
    assert!(message.contains("=(1, [\"2024-01-01"), "{message}");
}

#[tokio::test]
async fn copy_and_upsert_are_checked() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(
        &mut c,
        "CREATE TABLE s (id int PRIMARY KEY, k int, r int4range, EXCLUDE USING gist (k WITH =, r WITH &&))",
    )
    .await;

    // COPY FROM STDIN: a batch whose rows overlap each other is refused as a whole.
    let copy = async |c: &mut Conn, data: &str| -> Vec<String> {
        send(
            c,
            FrontendMessage::Query {
                sql: "COPY s FROM STDIN".to_owned(),
            },
        )
        .await;
        loop {
            let msg = next(c).await;
            if matches!(msg, BackendMessage::CopyInResponse { .. }) {
                break;
            }
            assert!(!matches!(msg, BackendMessage::Error { .. }), "{msg:?}");
        }
        send(
            c,
            FrontendMessage::CopyData {
                data: data.as_bytes().to_vec(),
            },
        )
        .await;
        send(c, FrontendMessage::CopyDone).await;
        until_ready(c).await
    };
    let out = copy(&mut c, "1\t1\t[1,10)\n2\t1\t[5,6)\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    let out = copy(&mut c, "1\t1\t[1,10)\n2\t1\t[10,20)\n3\t2\t[1,10)\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    // A later COPY conflicting with a committed row.
    let out = copy(&mut c, "4\t2\t[9,11)\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM s").await, "3");

    // Upsert: the DO UPDATE arm moving a range onto another row's is refused; the insert arm is
    // checked too; DO NOTHING skips the conflicting row.
    let out = q(
        &mut c,
        "INSERT INTO s VALUES (2, 1, '[15,16)') ON CONFLICT (id) DO UPDATE SET r = '[5,12)'",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    ok(
        &mut c,
        "INSERT INTO s VALUES (2, 1, '[15,16)') ON CONFLICT (id) DO UPDATE SET r = '[12,30)'",
    )
    .await;
    let out = q(
        &mut c,
        "INSERT INTO s VALUES (5, 1, '[29,31)') ON CONFLICT (id) DO UPDATE SET r = EXCLUDED.r",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    ok(
        &mut c,
        "INSERT INTO s VALUES (5, 1, '[29,31)'), (6, 1, '[40,41)') ON CONFLICT DO NOTHING",
    )
    .await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(id::text, ',' ORDER BY id) FROM s"
        )
        .await,
        "1,2,3,6"
    );
    // MERGE is checked as well.
    let out = q(
        &mut c,
        "MERGE INTO s USING (SELECT 3 AS id) src ON s.id = src.id \
         WHEN MATCHED THEN UPDATE SET k = 1",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
}

#[tokio::test]
async fn the_constraint_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        ok(&mut c, HIST).await;
        ok(
            &mut c,
            "INSERT INTO hist VALUES (gen_random_uuid(), 7, '2024-01-01 00:00:00+00', \
             '2024-02-01 00:00:00+00')",
        )
        .await;
        ok(
            &mut c,
            "CREATE TABLE plain (r daterange, CONSTRAINT plain_x EXCLUDE USING gist (r WITH &&))",
        )
        .await;
        ok(
            &mut c,
            "INSERT INTO plain VALUES ('[2024-01-01,2024-02-01)')",
        )
        .await;
        // Close the connection and wait for its server task to let go of the engine.
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    let out = q(
        &mut c,
        "INSERT INTO hist VALUES (gen_random_uuid(), 7, '2024-01-15 00:00:00+00', NULL)",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    let out = q(
        &mut c,
        "INSERT INTO plain VALUES ('[2024-01-31,2024-03-01)')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    ok(
        &mut c,
        "INSERT INTO plain VALUES ('[2024-02-01,2024-03-01)')",
    )
    .await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(constraint_name || ':' || constraint_type, ',' ORDER BY constraint_name) \
             FROM information_schema.table_constraints WHERE table_name IN ('hist', 'plain')"
        )
        .await,
        "hist_no_overlap:EXCLUDE,hist_pkey:PRIMARY KEY,plain_x:EXCLUDE"
    );
    ok(&mut c, "ALTER TABLE hist DROP CONSTRAINT hist_no_overlap").await;
    ok(
        &mut c,
        "INSERT INTO hist VALUES (gen_random_uuid(), 7, '2024-01-15 00:00:00+00', NULL)",
    )
    .await;
}

#[tokio::test]
async fn an_uncommitted_constraint_does_not_reject_other_sessions() {
    let engine = memory_engine();
    let mut a = open(&engine).await;
    let mut b = open(&engine).await;
    ok(&mut a, "CREATE TABLE u (r int4range)").await;
    ok(&mut a, "INSERT INTO u VALUES ('[1,10)')").await;
    ok(&mut a, "BEGIN").await;
    ok(
        &mut a,
        "ALTER TABLE u ADD CONSTRAINT u_x EXCLUDE USING gist (r WITH &&)",
    )
    .await;
    // Another session never sees the uncommitted constraint: its write either waits out the DDL
    // (refused with the retryable 40001) or proceeds, but it is never an exclusion violation.
    let out = q(&mut b, "INSERT INTO u VALUES ('[5,6)')").await;
    assert_ne!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
    let listed = scalar(
        &mut b,
        "SELECT count(*) FROM information_schema.table_constraints WHERE constraint_name = 'u_x'",
    )
    .await;
    assert_eq!(listed, "0");
    ok(&mut a, "ROLLBACK").await;
    ok(&mut b, "INSERT INTO u VALUES ('[5,6)')").await;
    // Created inside a transaction and committed, it then applies to everyone.
    ok(&mut a, "BEGIN").await;
    ok(
        &mut a,
        "CREATE TABLE v (r int4range, CONSTRAINT v_x EXCLUDE USING gist (r WITH &&))",
    )
    .await;
    ok(&mut a, "COMMIT").await;
    ok(&mut b, "INSERT INTO v VALUES ('[1,5)')").await;
    let out = q(&mut a, "INSERT INTO v VALUES ('[4,6)')").await;
    assert_eq!(error_code(&out).as_deref(), Some("23P01"), "{out:?}");
}

/// Two connections insert overlapping intervals for one key inside open transactions: exactly
/// one commits, at every isolation level.
#[tokio::test]
async fn two_concurrent_overlapping_writers_never_both_commit() {
    for level in ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        ok(
            &mut a,
            "CREATE TABLE c (id int PRIMARY KEY, k int, r int4range, \
             EXCLUDE USING gist (k WITH =, r WITH &&))",
        )
        .await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        // Both snapshots are taken before either write.
        assert_eq!(scalar(&mut a, "SELECT count(*) FROM c").await, "0");
        assert_eq!(scalar(&mut b, "SELECT count(*) FROM c").await, "0");
        ok(&mut a, "INSERT INTO c VALUES (1, 1, '[1,10)')").await;
        let out_b = q(&mut b, "INSERT INTO c VALUES (2, 1, '[5,15)')").await;
        let commit_a = q(&mut a, "COMMIT").await;
        let commit_b = q(&mut b, "COMMIT").await;
        let b_failed = error_code(&out_b).is_some() || error_code(&commit_b).is_some();
        assert!(
            matches!(
                error_code(&out_b).as_deref(),
                Some("40001" | "23P01") | None
            ),
            "{level}: {out_b:?}"
        );
        assert_eq!(error_code(&commit_a), None, "{level}: {commit_a:?}");
        assert!(b_failed, "{level}: both overlapping writers committed");
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM c").await,
            "1",
            "{level}"
        );

        // A writer whose snapshot predates another's commit still sees the committed row.
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(scalar(&mut b, "SELECT count(*) FROM c").await, "1");
        ok(&mut a, "INSERT INTO c VALUES (3, 1, '[20,30)')").await;
        let out = q(&mut b, "INSERT INTO c VALUES (4, 1, '[25,26)')").await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("23P01"),
            "{level}: {out:?}"
        );
        ok(&mut b, "ROLLBACK").await;

        // An UPDATE moving a range against a concurrent insert.
        ok(&mut a, "BEGIN").await;
        ok(&mut a, "UPDATE c SET r = '[40,50)' WHERE id = 3").await;
        let out = q(&mut b, "INSERT INTO c VALUES (5, 1, '[45,46)')").await;
        assert!(error_code(&out).is_some(), "{level}: {out:?}");
        ok(&mut a, "COMMIT").await;
        let out = q(&mut b, "INSERT INTO c VALUES (5, 1, '[45,46)')").await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("23P01"),
            "{level}: {out:?}"
        );
        ok(&mut b, "INSERT INTO c VALUES (5, 1, '[20,30)')").await;
    }
}

/// Many connections race overlapping inserts (and range moves) for a few keys. Every refusal is
/// 40001 or 23P01, and the committed table never holds two overlapping rows for one key.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_stress_loop_of_concurrent_writers_keeps_the_constraint() {
    let engine = memory_engine();
    let mut setup = open(&engine).await;
    ok(
        &mut setup,
        "CREATE TABLE st (id serial PRIMARY KEY, k int NOT NULL, r int4range NOT NULL, \
         EXCLUDE USING gist (k WITH =, r WITH &&))",
    )
    .await;
    ok(
        &mut setup,
        "CREATE TABLE sn (id serial PRIMARY KEY, r int4range NOT NULL, \
         EXCLUDE USING gist (r WITH &&))",
    )
    .await;
    let levels = ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"];
    let mut tasks = Vec::new();
    for worker in 0..12_u32 {
        let engine = Arc::clone(&engine);
        tasks.push(tokio::spawn(async move {
            let mut c = open(&engine).await;
            let mut committed = 0_u32;
            // A small deterministic generator per worker.
            let mut state = 0x9e37_79b9_u32.wrapping_mul(worker + 1);
            let mut rand = move |n: u32| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state % n
            };
            for i in 0..60_u32 {
                let level = levels[(worker as usize + i as usize) % levels.len()];
                let k = rand(3);
                let lo = rand(200);
                let hi = lo + 1 + rand(10);
                let table_sql = if i % 4 == 3 {
                    format!("INSERT INTO sn (r) VALUES ('[{lo},{hi})')")
                } else if i % 5 == 4 {
                    format!(
                        "UPDATE st SET r = '[{lo},{hi})' WHERE id = (SELECT min(id) FROM st WHERE k = {k})"
                    )
                } else {
                    format!("INSERT INTO st (k, r) VALUES ({k}, '[{lo},{hi})')")
                };
                let begin = q(&mut c, &format!("BEGIN ISOLATION LEVEL {level}")).await;
                assert_eq!(error_code(&begin), None, "{begin:?}");
                let out = q(&mut c, &table_sql).await;
                let code = error_code(&out);
                if code.is_some() {
                    assert!(
                        matches!(code.as_deref(), Some("40001" | "23P01")),
                        "{table_sql}: {out:?}"
                    );
                    q(&mut c, "ROLLBACK").await;
                    continue;
                }
                let commit = q(&mut c, "COMMIT").await;
                match error_code(&commit).as_deref() {
                    None => committed += 1,
                    Some("40001" | "23P01") => {},
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
    assert_eq!(
        scalar(
            &mut setup,
            "SELECT count(*) FROM st a JOIN st b ON a.id < b.id AND a.k = b.k AND a.r && b.r"
        )
        .await,
        "0"
    );
    assert_eq!(
        scalar(
            &mut setup,
            "SELECT count(*) FROM sn a JOIN sn b ON a.id < b.id AND a.r && b.r"
        )
        .await,
        "0"
    );
}

/// Only the exclusion constraint's own marker name is reserved. Views that store a rewritten
/// `BETWEEN SYMMETRIC` (re-parsed on every use), tables and functions with other `__nusadb_` names
/// keep working, and a forged marker CHECK is still refused.
#[tokio::test]
async fn only_the_exclusion_marker_name_is_reserved() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, "CREATE TABLE sym (x int)").await;
    ok(&mut c, "INSERT INTO sym VALUES (1), (5), (9)").await;
    ok(
        &mut c,
        "CREATE VIEW sym_v AS SELECT x FROM sym WHERE x BETWEEN SYMMETRIC 6 AND 2",
    )
    .await;
    assert_eq!(scalar(&mut c, "SELECT x FROM sym_v").await, "5");
    ok(
        &mut c,
        "CREATE VIEW sym_vv AS SELECT x FROM sym_v WHERE x BETWEEN SYMMETRIC 10 AND 0",
    )
    .await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM sym_vv").await, "1");
    ok(
        &mut c,
        "CREATE MATERIALIZED VIEW sym_mv AS SELECT x FROM sym WHERE x BETWEEN SYMMETRIC 10 AND 4",
    )
    .await;
    ok(&mut c, "INSERT INTO sym VALUES (7)").await;
    ok(&mut c, "REFRESH MATERIALIZED VIEW sym_mv").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM sym_mv").await, "3");

    // An incrementally maintained view re-parses its stored text on every base-table write, and an
    // auto-updatable view on every write through it.
    ok(
        &mut c,
        "CREATE MATERIALIZED VIEW sym_ivm WITH (incremental = true) AS \
         SELECT x FROM sym WHERE x BETWEEN SYMMETRIC 10 AND 4",
    )
    .await;
    ok(&mut c, "INSERT INTO sym VALUES (8)").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM sym_ivm").await, "4");
    ok(&mut c, "INSERT INTO sym_v VALUES (3)").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM sym_v").await, "2");

    ok(&mut c, "CREATE TABLE __nusadb_x (a int, b int)").await;
    ok(&mut c, "INSERT INTO __nusadb_x (a, b) VALUES (1, 2)").await;
    assert_eq!(scalar(&mut c, "SELECT a + b FROM __nusadb_x").await, "3");
    ok(
        &mut c,
        "CREATE FUNCTION __nusadb_f(a int) RETURNS int AS 'SELECT a + 1' LANGUAGE SQL",
    )
    .await;
    assert_eq!(scalar(&mut c, "SELECT __nusadb_f(1)").await, "2");

    for forged in [
        "CREATE TABLE fake (r int4range, CHECK (__nusadb_exclude('gist', '', r, '&&')))",
        "ALTER TABLE sym ADD CONSTRAINT f CHECK (__nusadb_exclude('gist', '', x, '='))",
        "CREATE FUNCTION __nusadb_exclude(a int) RETURNS int AS 'SELECT a' LANGUAGE SQL",
    ] {
        let out = q(&mut c, forged).await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("42939"),
            "{forged}: {out:?}"
        );
    }
}

/// A stored `CHECK` that only looks like the exclusion marker (written by an older build, or a
/// synthetic type check) is enforced as an ordinary predicate, never as an exclusion constraint and
/// never as an internal error. The catalog entries are written through the engine, as no SQL can
/// write them any more.
#[tokio::test]
async fn a_stored_check_that_only_looks_like_the_marker_is_an_ordinary_check() {
    use nusadb_core::engine::IsolationLevel;
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, "CREATE TABLE old (a int, r int4range)").await;
    let table = engine.lookup_table("old").unwrap().unwrap();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    // Not one top-level marker call: a predicate over a user function of that name.
    engine
        .add_check_constraint(txn, table.id, "old_cmp", b"__nusadb_exclude(a) = 1")
        .unwrap();
    // A valid stored form, but under a synthetic type-check name: a type check is a predicate.
    engine
        .add_check_constraint(
            txn,
            table.id,
            "nusadb_typeck_r_0",
            b"__nusadb_exclude('gist', '', r, '&&')",
        )
        .unwrap();
    engine.commit(txn).unwrap();

    let out = q(&mut c, "INSERT INTO old VALUES (1, '[1,5)')").await;
    let code = error_code(&out);
    assert_ne!(code.as_deref(), Some("XX000"), "{out:?}");
    assert_ne!(code.as_deref(), Some("23P01"), "{out:?}");
    assert_eq!(code.as_deref(), Some("42883"), "{out:?}");
    assert_eq!(
        scalar(
            &mut c,
            "SELECT constraint_type FROM information_schema.table_constraints \
             WHERE constraint_name = 'old_cmp'"
        )
        .await,
        "CHECK"
    );

    // A malformed marker call alone (wrong arguments) is an ordinary CHECK too.
    ok(&mut c, "CREATE TABLE old2 (a int)").await;
    let table = engine.lookup_table("old2").unwrap().unwrap();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine
        .add_check_constraint(txn, table.id, "old2_c", b"__nusadb_exclude(a)")
        .unwrap();
    engine.commit(txn).unwrap();
    let out = q(&mut c, "INSERT INTO old2 VALUES (1)").await;
    assert_eq!(error_code(&out).as_deref(), Some("42883"), "{out:?}");
    // Dropping such a constraint works and leaves the table writable.
    ok(&mut c, "ALTER TABLE old2 DROP CONSTRAINT old2_c").await;
    ok(&mut c, "INSERT INTO old2 VALUES (1)").await;
}

/// `ALTER TABLE ... ADD COLUMN` with a type that does not exist is `42704`, as `CREATE TABLE` and
/// `ALTER COLUMN ... TYPE` are; an existing domain still works.
#[tokio::test]
async fn add_column_with_an_unknown_type_is_42704() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, "CREATE TABLE ac (a int)").await;
    let out = q(&mut c, "ALTER TABLE ac ADD COLUMN b no_such_type").await;
    assert_eq!(error_code(&out).as_deref(), Some("42704"), "{out:?}");
    ok(&mut c, "CREATE DOMAIN posint AS int CHECK (VALUE > 0)").await;
    ok(&mut c, "ALTER TABLE ac ADD COLUMN c posint").await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT count(*) FROM information_schema.columns WHERE table_name = 'ac'"
        )
        .await,
        "2"
    );
}

/// A data dir written by an older build can hold a user function named `__nusadb_exclude` and a
/// `CHECK` whose text is exactly an (untagged) exclusion stored form. Both stay what they were: the
/// function is callable from queries, views and procedures and can be dropped (with its argument
/// list), and the `CHECK` is an ordinary predicate calling it, never an exclusion constraint, also
/// when `LIKE` copies it. The legacy rows are written the way an older build left them: the
/// function's catalog row renamed by a superuser, the `CHECK` through the engine.
#[tokio::test]
async fn legacy_marker_named_function_and_check_keep_their_meaning() {
    use nusadb_core::engine::IsolationLevel;
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(
        &mut c,
        "CREATE FUNCTION legacy_tmp(m text, i text, r int4range, o text) RETURNS boolean \
         AS 'SELECT true' LANGUAGE SQL",
    )
    .await;
    ok(
        &mut c,
        "UPDATE nusadb_functions SET name = '__nusadb_exclude' WHERE name = 'legacy_tmp'",
    )
    .await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT __nusadb_exclude('gist', '', int4range(1, 2), '&&')"
        )
        .await,
        "true"
    );
    ok(&mut c, "CREATE TABLE oc (r int4range)").await;
    let table = engine.lookup_table("oc").unwrap().unwrap();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine
        .add_check_constraint(
            txn,
            table.id,
            "oc_check1",
            b"__nusadb_exclude('gist', '', r, '&&')",
        )
        .unwrap();
    engine.commit(txn).unwrap();

    // An ordinary CHECK. A CHECK predicate resolves built-in functions only, so a write fails
    // with 42883, as it did on the build that wrote it; never 23P01, and never an internal error.
    let out = q(&mut c, "INSERT INTO oc VALUES ('[1,5)')").await;
    assert_eq!(error_code(&out).as_deref(), Some("42883"), "{out:?}");
    let kinds = "SELECT string_agg(table_name || ':' || constraint_name || ':' || constraint_type, \
                 ',' ORDER BY table_name) FROM information_schema.table_constraints \
                 WHERE table_name IN ('oc', 'oc_like')";
    assert_eq!(scalar(&mut c, kinds).await, "oc:oc_check1:CHECK");
    ok(&mut c, "CREATE TABLE oc_like (LIKE oc INCLUDING ALL)").await;
    assert_eq!(
        scalar(&mut c, kinds).await,
        "oc:oc_check1:CHECK,oc_like:oc_check1:CHECK"
    );
    assert_eq!(
        scalar(
            &mut c,
            "SELECT count(*) FROM information_schema.check_constraints \
             WHERE constraint_name = 'oc_check1'"
        )
        .await,
        "2"
    );
    // No index was made for it, and dropping it frees the table, overlapping rows and all.
    assert_eq!(
        scalar(
            &mut c,
            "SELECT count(*) FROM information_schema.statistics WHERE table_name LIKE 'oc%'"
        )
        .await,
        "0"
    );
    ok(&mut c, "ALTER TABLE oc DROP CONSTRAINT oc_check1").await;
    ok(
        &mut c,
        "INSERT INTO oc VALUES ('[1,5)'), ('[3,8)'), ('[2,4)')",
    )
    .await;

    // Views and procedures may still call it.
    ok(
        &mut c,
        "CREATE VIEW ov AS SELECT __nusadb_exclude('gist', '', r, '&&') AS x FROM oc",
    )
    .await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM ov WHERE x").await, "3");
    ok(
        &mut c,
        "CREATE PROCEDURE op() AS $$ BEGIN INSERT INTO oc SELECT r FROM oc WHERE \
         __nusadb_exclude('gist', '', r, '&&') LIMIT 1; END $$",
    )
    .await;
    ok(&mut c, "CALL op()").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM oc").await, "4");

    // A stored routine, DO block or trigger body creating a CHECK that calls it creates an
    // ordinary CHECK, as the older build did: never an exclusion constraint, and never 42939.
    ok(
        &mut c,
        "CREATE PROCEDURE mk() AS $$ BEGIN CREATE TABLE lp1 (r int4range, \
         CHECK (__nusadb_exclude('gist', '', r, '&&'))); END $$",
    )
    .await;
    ok(&mut c, "CALL mk()").await;
    ok(
        &mut c,
        "DO $$ BEGIN ALTER TABLE lp1 ADD CONSTRAINT lp1_b CHECK (__nusadb_exclude('gist', '', r, \
         '&&')); END $$",
    )
    .await;
    ok(&mut c, "CREATE TABLE lp_src (a int)").await;
    ok(
        &mut c,
        "CREATE FUNCTION mk_tf() RETURNS trigger LANGUAGE nusascript AS $$ BEGIN CREATE TABLE lp2 \
         (r int4range, CHECK (__nusadb_exclude('gist', '', r, '&&'))); RETURN NEW; END $$",
    )
    .await;
    ok(
        &mut c,
        "CREATE TRIGGER mk_tg AFTER INSERT ON lp_src FOR EACH ROW EXECUTE FUNCTION mk_tf()",
    )
    .await;
    ok(&mut c, "INSERT INTO lp_src VALUES (1)").await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(table_name || ':' || constraint_type, ',' ORDER BY constraint_name) \
             FROM information_schema.table_constraints WHERE table_name IN ('lp1', 'lp2')"
        )
        .await,
        "lp1:CHECK,lp1:CHECK,lp2:CHECK"
    );
    // The same CHECK written directly by a client is refused (42939).
    let out = q(
        &mut c,
        "ALTER TABLE lp1 ADD CHECK (__nusadb_exclude('gist', '', r, '&&'))",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("42939"), "{out:?}");

    // Dropping it works, with or without its argument list.
    ok(&mut c, "DROP VIEW ov").await;
    ok(&mut c, "DROP PROCEDURE op").await;
    ok(
        &mut c,
        "DROP FUNCTION __nusadb_exclude(text, text, int4range, text)",
    )
    .await;
    let out = q(
        &mut c,
        "SELECT __nusadb_exclude('gist', '', int4range(1, 2), '&&')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("42883"), "{out:?}");
    // New definitions of the name are still refused.
    let out = q(
        &mut c,
        "CREATE FUNCTION __nusadb_exclude(a int) RETURNS int AS 'SELECT a' LANGUAGE SQL",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("42939"), "{out:?}");
}

/// A column-level `EXCLUDE` is refused with a message that does not leak the rewrite's marker.
#[tokio::test]
async fn a_column_level_exclude_is_refused_clearly() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, "CREATE TABLE ct (a int)").await;
    for sql in [
        "CREATE TABLE l1 (r int4range EXCLUDE USING gist (r WITH &&))",
        "ALTER TABLE ct ADD COLUMN r int4range EXCLUDE USING gist (r WITH &&)",
        "ALTER TABLE ct ALTER COLUMN a SET DEFAULT EXCLUDE USING gist (a WITH =)",
    ] {
        let out = q(&mut c, sql).await;
        assert_eq!(error_code(&out).as_deref(), Some("0A000"), "{sql}: {out:?}");
        let message = out.iter().find(|m| m.starts_with("E:")).unwrap();
        assert!(
            message.contains("EXCLUDE is a table constraint"),
            "{message}"
        );
        assert!(!message.contains("__nusadb"), "{message}");
    }
}
