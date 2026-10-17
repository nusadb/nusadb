//! Deferrable constraints over the wire: the SQLSTATE a deferred violation is reported with at
//! COMMIT, on every path that ends a transaction (simple and extended query, COPY, auto-commit,
//! PREPARE TRANSACTION, procedures with EXCEPTION blocks), persistence across a restart, and
//! concurrent writers (two connections at every isolation level, then a stress loop of many).

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

const SCHEMA: [&str; 4] = [
    "CREATE TABLE t (id int CONSTRAINT t_pk PRIMARY KEY DEFERRABLE INITIALLY DEFERRED, v text)",
    "CREATE TABLE parent (id int PRIMARY KEY)",
    "CREATE TABLE child (id int PRIMARY KEY, pid int CONSTRAINT child_fk REFERENCES parent (id) \
     DEFERRABLE INITIALLY DEFERRED)",
    "CREATE TABLE booking (room int, during int4range, CONSTRAINT no_overlap EXCLUDE USING gist \
     (room WITH =, during WITH &&) DEFERRABLE INITIALLY DEFERRED)",
];

async fn schema(c: &mut Conn) {
    for sql in SCHEMA {
        ok(c, sql).await;
    }
}

#[tokio::test]
async fn a_deferred_violation_fails_commit_with_its_code_and_rolls_back() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    schema(&mut c).await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'a')").await;

    // UNIQUE / PRIMARY KEY: 23505, naming the constraint.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO t VALUES (2, 'b')").await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    let out = fails(&mut c, "COMMIT", "23505").await;
    assert!(out.iter().any(|m| m.contains("\"t_pk\"")), "{out:?}");
    // The transaction is over and nothing of it committed.
    assert_eq!(status(&out), "Z:I");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");

    // FOREIGN KEY: 23503.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO child VALUES (1, 42)").await;
    let out = fails(&mut c, "COMMIT", "23503").await;
    assert!(out.iter().any(|m| m.contains("\"child_fk\"")), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM child").await, "0");

    // EXCLUDE: 23P01.
    ok(&mut c, "BEGIN").await;
    ok(
        &mut c,
        "INSERT INTO booking VALUES (1, '[1,5)'), (1, '[3,9)')",
    )
    .await;
    let out = fails(&mut c, "COMMIT", "23P01").await;
    assert!(out.iter().any(|m| m.contains("\"no_overlap\"")), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM booking").await, "0");

    // `END` commits the same way.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    fails(&mut c, "END", "23505").await;

    // Auto-commit: the statement's own transaction checks at its end.
    fails(&mut c, "INSERT INTO t VALUES (1, 'dup')", "23505").await;
    fails(&mut c, "INSERT INTO child VALUES (1, 42)", "23503").await;
    fails(
        &mut c,
        "INSERT INTO booking VALUES (1, '[1,5)'), (1, '[3,9)')",
        "23P01",
    )
    .await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");

    // A multi-statement simple query: the implicit transaction checks when it ends.
    let out = q(
        &mut c,
        "INSERT INTO t VALUES (5, 'x'); INSERT INTO t VALUES (5, 'y')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");
    // ... and a swap there commits.
    ok(
        &mut c,
        "INSERT INTO t VALUES (2, 'b'); UPDATE t SET id = 3 WHERE id = 1; \
         UPDATE t SET id = 1 WHERE id = 2; UPDATE t SET id = 2 WHERE id = 3",
    )
    .await;
    assert_eq!(
        scalar(&mut c, "SELECT string_agg(id || v, ',' ORDER BY id) FROM t").await,
        "1b,2a"
    );
}

#[tokio::test]
async fn set_constraints_over_the_wire() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    schema(&mut c).await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'a')").await;

    // Outside a block: a warning, nothing else.
    let out = q(&mut c, "SET CONSTRAINTS ALL IMMEDIATE").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    assert!(out.iter().any(|m| m == "C:SET CONSTRAINTS"), "{out:?}");
    assert!(
        out.iter().any(|m| m.contains("transaction blocks")),
        "{out:?}"
    );

    // IMMEDIATE checks what is pending: the failure aborts the block until ROLLBACK TO.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "SAVEPOINT s").await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    let out = fails(&mut c, "SET CONSTRAINTS t_pk IMMEDIATE", "23505").await;
    assert_eq!(status(&out), "Z:E");
    ok(&mut c, "ROLLBACK TO SAVEPOINT s").await;
    ok(&mut c, "SET CONSTRAINTS t_pk IMMEDIATE").await;
    fails(&mut c, "INSERT INTO t VALUES (1, 'dup')", "23505").await;
    ok(&mut c, "ROLLBACK").await;

    ok(&mut c, "BEGIN").await;
    fails(&mut c, "SET CONSTRAINTS missing DEFERRED", "42704").await;
    ok(&mut c, "ROLLBACK").await;
    ok(&mut c, "BEGIN").await;
    fails(&mut c, "SET CONSTRAINTS parent_pkey DEFERRED", "55000").await;
    ok(&mut c, "ROLLBACK").await;

    // A schema-qualified name.
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "SET CONSTRAINTS public.t_pk IMMEDIATE").await;
    fails(&mut c, "INSERT INTO t VALUES (1, 'dup')", "23505").await;
    ok(&mut c, "ROLLBACK").await;

    // The extended protocol, inside a block and in auto-commit.
    let out = extended(&mut c, "BEGIN").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    let out = extended(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    let out = extended(&mut c, "COMMIT").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    let out = extended(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");
}

#[tokio::test]
async fn copy_is_checked_when_its_transaction_ends() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    schema(&mut c).await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'a')").await;
    // Auto-commit COPY with a duplicate of a committed key.
    let out = copy_in(&mut c, "COPY t FROM STDIN", "2\tb\n1\tdup\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");
    // Inside a block: the duplicate is accepted for now, and fixed before COMMIT.
    ok(&mut c, "BEGIN").await;
    let out = copy_in(&mut c, "COPY t FROM STDIN", "2\tb\n1\tdup\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    ok(&mut c, "UPDATE t SET id = 3 WHERE v = 'dup'").await;
    ok(&mut c, "COMMIT").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "3");
    // Children loaded before their parents.
    ok(&mut c, "BEGIN").await;
    let out = copy_in(&mut c, "COPY child FROM STDIN", "1\t10\n2\t20\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    ok(&mut c, "INSERT INTO parent VALUES (10)").await;
    fails(&mut c, "COMMIT", "23503").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM child").await, "0");
}

#[tokio::test]
async fn prepare_transaction_runs_the_checks_and_a_restart_owes_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        schema(&mut c).await;
        ok(&mut c, "INSERT INTO t VALUES (1, 'a'), (2, 'b')").await;

        // A violation at PREPARE: refused, rolled back, nothing prepared.
        ok(&mut c, "BEGIN").await;
        ok(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
        let out = fails(&mut c, "PREPARE TRANSACTION 'bad'", "23505").await;
        assert_eq!(status(&out), "Z:I");
        assert_eq!(
            scalar(&mut c, "SELECT count(*) FROM nusadb_prepared_xacts").await,
            "0"
        );

        // A valid swap is prepared, survives a restart, and commits from another connection.
        ok(&mut c, "BEGIN").await;
        ok(&mut c, "UPDATE t SET id = 3 WHERE id = 1").await;
        ok(&mut c, "UPDATE t SET id = 1 WHERE id = 2").await;
        ok(&mut c, "UPDATE t SET id = 2 WHERE id = 3").await;
        ok(&mut c, "PREPARE TRANSACTION 'swap'").await;
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    ok(&mut c, "COMMIT PREPARED 'swap'").await;
    assert_eq!(
        scalar(&mut c, "SELECT string_agg(id || v, ',' ORDER BY id) FROM t").await,
        "1b,2a"
    );
    // The timing survived the restart.
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(constraint_name || ':' || is_deferrable || ':' || initially_deferred, \
             ',' ORDER BY constraint_name) FROM information_schema.table_constraints \
             WHERE constraint_name IN ('t_pk', 'child_fk', 'no_overlap', 'parent_pkey')"
        )
        .await,
        "child_fk:YES:YES,no_overlap:YES:YES,parent_pkey:NO:NO,t_pk:YES:YES"
    );
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'dup')").await;
    fails(&mut c, "COMMIT", "23505").await;
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "INSERT INTO child VALUES (1, 7)").await;
    ok(&mut c, "INSERT INTO parent VALUES (7)").await;
    ok(&mut c, "COMMIT").await;
}

#[tokio::test]
async fn procedures_and_exception_blocks() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    schema(&mut c).await;
    ok(&mut c, "INSERT INTO t VALUES (1, 'a')").await;
    ok(&mut c, "CREATE TABLE log (msg text)").await;
    // The duplicate a body queued is undone with the body when its handler runs.
    ok(
        &mut c,
        "CREATE PROCEDURE undone() AS $$
         BEGIN
           INSERT INTO t VALUES (1, 'dup');
           RAISE 'abandon';
         EXCEPTION WHEN OTHERS THEN
           INSERT INTO log VALUES ('handled');
         END
         $$",
    )
    .await;
    // A nested call queues its duplicate for the caller's commit.
    ok(
        &mut c,
        "CREATE PROCEDURE inner_dup() AS $$ INSERT INTO t VALUES (1, 'dup') $$",
    )
    .await;
    ok(
        &mut c,
        "CREATE PROCEDURE outer_dup() AS $$ CALL inner_dup() $$",
    )
    .await;
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "CALL undone()").await;
    ok(&mut c, "COMMIT").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM log").await, "1");
    ok(&mut c, "BEGIN").await;
    ok(&mut c, "CALL outer_dup()").await;
    fails(&mut c, "COMMIT", "23505").await;
    fails(&mut c, "CALL outer_dup()", "23505").await;
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "1");
}

/// Two connections write the same deferred key in open transactions, at every isolation level:
/// never do both commit. The second writer fails on the key lock the first holds (40001) when it
/// writes, or at its COMMIT.
#[tokio::test]
async fn two_writers_of_one_deferred_key_never_both_commit() {
    for level in ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        schema(&mut a).await;
        ok(&mut a, "INSERT INTO t VALUES (1, 'a'), (2, 'b')").await;
        ok(&mut a, "INSERT INTO parent VALUES (10)").await;

        // Both insert key 5.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(scalar(&mut b, "SELECT count(*) FROM t").await, "2");
        ok(&mut a, "INSERT INTO t VALUES (5, 'a5')").await;
        let wb = q(&mut b, "INSERT INTO t VALUES (5, 'b5')").await;
        let ca = q(&mut a, "COMMIT").await;
        let cb = q(
            &mut b,
            if error_code(&wb).is_some() {
                "ROLLBACK"
            } else {
                "COMMIT"
            },
        )
        .await;
        assert_eq!(error_code(&ca), None, "{level}: {ca:?}");
        assert!(
            matches!(error_code(&wb).as_deref(), Some("40001")) || error_code(&cb).is_some(),
            "{level}: both writers of key 5 committed: {wb:?} {cb:?}"
        );
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM t WHERE id = 5").await,
            "1",
            "{level}"
        );

        // A swaps 1 and 2 while B inserts 1: B is refused, A's swap commits.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "UPDATE t SET id = 2 WHERE v = 'a'").await;
        ok(&mut a, "UPDATE t SET id = 1 WHERE v = 'b'").await;
        let wb = q(&mut b, "INSERT INTO t VALUES (1, 'intruder')").await;
        assert!(error_code(&wb).is_some(), "{level}: {wb:?}");
        ok(&mut a, "COMMIT").await;
        assert_eq!(
            scalar(
                &mut a,
                "SELECT string_agg(id || v, ',' ORDER BY id) FROM t WHERE id < 3"
            )
            .await,
            "1b,2a",
            "{level}"
        );

        // A child of a missing parent while another connection adds the parent: the parent write
        // waits out the child's shared lock (40001); the child's COMMIT then finds no parent.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "INSERT INTO child VALUES (1, 20)").await;
        let wb = q(&mut b, "INSERT INTO parent VALUES (20)").await;
        assert_eq!(error_code(&wb).as_deref(), Some("40001"), "{level}: {wb:?}");
        fails(&mut a, "COMMIT", "23503").await;

        // A deletes a referenced parent (deferred NO ACTION); B adds a child of it: B is refused on
        // A's exclusive key lock, and A's COMMIT finds the existing child still referencing it.
        ok(&mut a, "INSERT INTO child VALUES (2, 10)").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "DELETE FROM parent WHERE id = 10").await;
        let wb = q(&mut b, "INSERT INTO child VALUES (3, 10)").await;
        assert_eq!(error_code(&wb).as_deref(), Some("40001"), "{level}: {wb:?}");
        fails(&mut a, "COMMIT", "23503").await;
        assert_eq!(scalar(&mut a, "SELECT count(*) FROM parent").await, "1");

        // B committed a duplicate before A's deferred write: A's COMMIT sees it.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(scalar(&mut a, "SELECT count(*) FROM t").await, "3");
        ok(&mut b, "INSERT INTO t VALUES (7, 'b7')").await;
        let wa = q(&mut a, "INSERT INTO t VALUES (7, 'a7')").await;
        if error_code(&wa).is_none() {
            fails(&mut a, "COMMIT", "23505").await;
        } else {
            ok(&mut a, "ROLLBACK").await;
        }
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM t WHERE id = 7").await,
            "1",
            "{level}"
        );

        // Two exclusion writers of overlapping ranges.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "INSERT INTO booking VALUES (1, '[1,10)')").await;
        let wb = q(&mut b, "INSERT INTO booking VALUES (1, '[5,15)')").await;
        ok(&mut a, "COMMIT").await;
        let cb = q(
            &mut b,
            if error_code(&wb).is_some() {
                "ROLLBACK"
            } else {
                "COMMIT"
            },
        )
        .await;
        assert!(
            error_code(&wb).is_some() || error_code(&cb).is_some(),
            "{level}: both overlapping bookings committed"
        );
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM booking").await,
            "1",
            "{level}"
        );
    }
}

/// Many connections race swaps, inserts and deletes over a few deferred keys, parents and children.
/// Every refusal is 40001 or a constraint code, and the committed state never holds a duplicate
/// key, a child without its parent, or two overlapping bookings.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_stress_loop_keeps_every_deferred_constraint() {
    let engine = memory_engine();
    let mut setup = open(&engine).await;
    ok(
        &mut setup,
        "CREATE TABLE k (slot int NOT NULL, owner int NOT NULL, \
         CONSTRAINT k_slot UNIQUE (slot) DEFERRABLE INITIALLY DEFERRED)",
    )
    .await;
    ok(&mut setup, "CREATE TABLE p (id int PRIMARY KEY)").await;
    ok(
        &mut setup,
        "CREATE TABLE c (id serial PRIMARY KEY, pid int NOT NULL CONSTRAINT c_fk REFERENCES p (id) \
         DEFERRABLE INITIALLY DEFERRED)",
    )
    .await;
    ok(
        &mut setup,
        "CREATE TABLE b (id serial PRIMARY KEY, room int NOT NULL, during int4range NOT NULL, \
         CONSTRAINT b_x EXCLUDE USING gist (room WITH =, during WITH &&) DEFERRABLE INITIALLY DEFERRED)",
    )
    .await;
    for slot in 0..6 {
        ok(
            &mut setup,
            &format!("INSERT INTO k VALUES ({slot}, {slot})"),
        )
        .await;
    }
    for id in 0..4 {
        ok(&mut setup, &format!("INSERT INTO p VALUES ({id})")).await;
    }
    let levels = ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"];
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
            for i in 0..50_u32 {
                let level = levels[(worker as usize + i as usize) % levels.len()];
                let x = rand(6);
                let y = rand(6);
                let steps: Vec<String> = match i % 5 {
                    // Swap two slots through a temporary duplicate; now and then leave it.
                    0 | 1 => {
                        let mut s = vec![format!(
                            "UPDATE k SET slot = {x}, owner = owner + 1000 WHERE slot = {y}"
                        )];
                        if rand(5) != 0 {
                            s.push(format!(
                                "UPDATE k SET slot = {y} WHERE slot = {x} AND owner < 1000"
                            ));
                        }
                        s.push("UPDATE k SET owner = owner - 1000 WHERE owner >= 1000".to_owned());
                        s
                    },
                    // A child, then (maybe) its parent.
                    2 => {
                        let pid = 4 + rand(4);
                        let mut s = vec![format!("INSERT INTO c (pid) VALUES ({pid})")];
                        if rand(2) == 0 {
                            s.push(format!("INSERT INTO p VALUES ({pid})"));
                        }
                        s
                    },
                    // Remove a parent, maybe re-create it.
                    3 => {
                        let pid = rand(8);
                        let mut s = vec![format!("DELETE FROM p WHERE id = {pid}")];
                        if rand(2) == 0 {
                            s.push(format!("INSERT INTO p VALUES ({pid})"));
                        }
                        s
                    },
                    // Bookings, moved through an overlap.
                    _ => {
                        let room = rand(2);
                        let lo = rand(50);
                        vec![
                            format!(
                                "INSERT INTO b (room, during) VALUES ({room}, '[{lo},{})')",
                                lo + 1 + rand(5)
                            ),
                            format!(
                                "UPDATE b SET during = '[{lo},{})' WHERE id = (SELECT min(id) FROM b WHERE room = {room})",
                                lo + 2
                            ),
                        ]
                    },
                };
                let begin = q(&mut c, &format!("BEGIN ISOLATION LEVEL {level}")).await;
                assert_eq!(error_code(&begin), None, "{begin:?}");
                let mut failed = false;
                for sql in &steps {
                    let out = q(&mut c, sql).await;
                    if let Some(code) = error_code(&out) {
                        assert!(
                            matches!(code.as_str(), "40001" | "23505" | "23503" | "23P01"),
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
                    Some("40001" | "23505" | "23503" | "23P01") => {},
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
            "SELECT count(*) FROM (SELECT slot FROM k GROUP BY slot HAVING count(*) > 1) d"
        )
        .await,
        "0"
    );
    assert_eq!(scalar(&mut setup, "SELECT count(*) FROM k").await, "6");
    assert_eq!(
        scalar(
            &mut setup,
            "SELECT count(*) FROM c WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.id = c.pid)"
        )
        .await,
        "0"
    );
    assert_eq!(
        scalar(
            &mut setup,
            "SELECT count(*) FROM b x JOIN b y ON x.id < y.id AND x.room = y.room AND x.during && y.during"
        )
        .await,
        "0"
    );
}
