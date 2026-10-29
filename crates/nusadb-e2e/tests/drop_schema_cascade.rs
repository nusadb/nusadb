//! `DROP SCHEMA ... CASCADE` across the schema boundary, over the wire: foreign keys in both
//! directions survive a restart in their dropped state, a concurrent writer on the referenced table
//! refuses the drop rather than racing it, a temporary table's foreign key goes with the session,
//! and a data directory an earlier build left with dangling entries is repaired at open.
//!
//! The fixture `dangling_schema_drop.wal` is the log of a data directory an earlier build wrote:
//!
//! - `par (id PK)` holding 1 and 2, and `s.ch (id PK, pid REFERENCES par ON DELETE CASCADE)`;
//! - `s.p2 (id PK)` and `oc (id, pid REFERENCES s.p2)` outside the schema;
//! - the view `s.v` and the sequence `s.sq`;
//! - then `DROP SCHEMA s CASCADE`, which left both foreign keys, the dropped tables' constraints
//!   and indexes, the view and the sequence behind.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic/indexing over linear protocol scripts"
)]

use std::path::Path;
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
async fn open(
    engine: &Arc<dyn StorageEngine>,
) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
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
    (conn, handle)
}

/// Render one backend message compactly: `C:<tag>`, `E:<sqlstate>`, `D:<fields>`, `Z:<I|T|E>`.
fn render(msg: &BackendMessage) -> String {
    match msg {
        BackendMessage::CommandComplete { tag } => format!("C:{tag}"),
        BackendMessage::Error { code, .. } => format!("E:{code}"),
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

/// Send a simple query; its data rows and error codes, without the framing.
async fn q(conn: &mut Conn, sql: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        let rendered = render(&msg);
        if rendered.starts_with("D:") || rendered.starts_with("E:") {
            out.push(rendered);
        }
        if done {
            return out;
        }
    }
}

/// Run each statement, asserting it succeeds without rows.
async fn ok(conn: &mut Conn, statements: &[&str]) {
    for sql in statements {
        assert!(q(conn, sql).await.is_empty(), "{sql}");
    }
}

fn engine_at(wal: &Path) -> Arc<dyn StorageEngine> {
    Arc::new(BtreeEngine::open(wal).expect("open the engine"))
}

#[tokio::test]
async fn foreign_keys_across_the_boundary_stay_dropped_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    {
        let engine = engine_at(&wal);
        let (mut c, handle) = open(&engine).await;
        ok(
            &mut c,
            &[
                "CREATE TABLE par (id INT PRIMARY KEY)",
                "INSERT INTO par VALUES (1), (2)",
                "CREATE SCHEMA s",
                "CREATE TABLE s.ch (id INT PRIMARY KEY, pid INT REFERENCES par ON DELETE CASCADE)",
                "INSERT INTO s.ch VALUES (1, 1)",
                "CREATE TABLE s.p (id INT PRIMARY KEY)",
                "INSERT INTO s.p VALUES (1)",
                "CREATE TABLE oc (id INT, pid INT REFERENCES s.p)",
                "INSERT INTO oc VALUES (1, 1)",
                "CREATE VIEW s.v AS SELECT id FROM s.ch",
                "CREATE SEQUENCE s.q",
            ],
        )
        .await;
        assert_eq!(q(&mut c, "DROP SCHEMA s").await, ["E:2BP01"]);
        ok(&mut c, &["DROP SCHEMA s CASCADE"]).await;
        drop(c);
        handle.await.unwrap().unwrap();
    }
    let engine = engine_at(&wal);
    let (mut c, _) = open(&engine).await;
    ok(
        &mut c,
        &[
            "DELETE FROM par WHERE id = 1",
            "INSERT INTO oc VALUES (2, 99)",
            "CREATE SCHEMA s",
            "CREATE TABLE s.ch (id INT PRIMARY KEY, pid INT REFERENCES par ON DELETE CASCADE)",
            "CREATE TABLE s.p (id INT PRIMARY KEY)",
            "CREATE VIEW s.v AS SELECT 7 AS id",
            "CREATE SEQUENCE s.q",
            "INSERT INTO s.ch VALUES (2, 2)",
            "DELETE FROM par WHERE id = 2",
        ],
    )
    .await;
    assert_eq!(q(&mut c, "SELECT COUNT(*) FROM s.ch").await, ["D:0"]);
    assert_eq!(q(&mut c, "SELECT id FROM s.v").await, ["D:7"]);
    assert_eq!(q(&mut c, "SELECT nextval('s.q')").await, ["D:1"]);
}

#[tokio::test]
async fn a_writer_on_the_referenced_table_refuses_the_drop_and_rollback_restores_it() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine).await;
    let (mut b, _) = open(&engine).await;
    ok(
        &mut a,
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "INSERT INTO par VALUES (1), (2)",
            "CREATE SCHEMA s",
            "CREATE TABLE s.ch (id INT PRIMARY KEY, pid INT REFERENCES par ON DELETE CASCADE)",
            "INSERT INTO s.ch VALUES (1, 1)",
        ],
    )
    .await;
    // A running writer holds the referenced table: the drop of the foreign key into it does not
    // wait, it refuses with a serialization failure, and nothing is dropped.
    ok(&mut a, &["BEGIN", "INSERT INTO par VALUES (3)"]).await;
    assert_eq!(q(&mut b, "DROP SCHEMA s CASCADE").await, ["E:40001"]);
    ok(&mut a, &["COMMIT"]).await;
    assert_eq!(q(&mut b, "SELECT COUNT(*) FROM s.ch").await, ["D:1"]);

    // A rolled-back drop restores the tables and the foreign key, which still cascades.
    ok(&mut b, &["BEGIN", "DROP SCHEMA s CASCADE", "ROLLBACK"]).await;
    ok(&mut a, &["DELETE FROM par WHERE id = 1"]).await;
    assert_eq!(q(&mut a, "SELECT COUNT(*) FROM s.ch").await, ["D:0"]);

    // Until it commits, the other session does not see the tables as gone: reading one, or
    // deleting from the table its foreign key references, is refused rather than racing the drop.
    ok(&mut b, &["BEGIN", "DROP SCHEMA s CASCADE"]).await;
    assert_eq!(q(&mut a, "SELECT COUNT(*) FROM s.ch").await, ["E:40001"]);
    assert_eq!(q(&mut a, "DELETE FROM par WHERE id = 2").await, ["E:40001"]);
    ok(&mut b, &["COMMIT"]).await;
    ok(&mut a, &["DELETE FROM par WHERE id = 2"]).await;
    assert_eq!(q(&mut a, "SELECT COUNT(*) FROM par").await, ["D:1"]);
}

#[tokio::test]
async fn a_temporary_tables_foreign_key_goes_with_the_session() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine).await;
    ok(
        &mut a,
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "INSERT INTO par VALUES (1), (2)",
        ],
    )
    .await;
    let (mut t, handle) = open(&engine).await;
    ok(
        &mut t,
        &[
            "CREATE TEMP TABLE tch (id INT, pid INT REFERENCES par ON DELETE CASCADE)",
            "INSERT INTO tch VALUES (1, 1)",
        ],
    )
    .await;
    drop(t);
    handle.await.unwrap().unwrap();
    ok(&mut a, &["DELETE FROM par WHERE id = 1"]).await;
    assert_eq!(q(&mut a, "SELECT id FROM par").await, ["D:2"]);
}

/// A copy of the fixture as the log of a fresh data directory under `dir`.
fn legacy_log(dir: &Path) -> std::path::PathBuf {
    let wal = dir.join("btree.wal");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dangling_schema_drop.wal"),
        &wal,
    )
    .expect("copy the fixture");
    wal
}

#[tokio::test]
async fn a_directory_with_dangling_entries_is_repaired_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let wal = legacy_log(dir.path());
    let engine = engine_at(&wal);
    // The view and sequence of the dropped schema are reported, never dropped behind the user's
    // back (a view of the default schema may be named "s.v" on purpose).
    let orphans: Vec<(&str, String)> = nusadb_sql::orphaned_schema_objects(&*engine)
        .unwrap()
        .into_iter()
        .map(|o| (o.kind, o.name))
        .collect();
    assert_eq!(
        orphans,
        [("VIEW", "s.v".to_owned()), ("SEQUENCE", "s.sq".to_owned())]
    );
    let (mut c, handle) = open(&engine).await;
    ok(
        &mut c,
        &[
            // The foreign key into the dropped s.p2 no longer constrains oc ...
            "INSERT INTO oc VALUES (2, 99)",
            // ... and the one from the dropped s.ch no longer blocks deletes from par.
            "DELETE FROM par WHERE id = 1",
            // The reported objects drop with the statement the report names ...
            "DROP VIEW s.v",
            "DROP SEQUENCE s.sq",
            // ... and every name the dropped schema held is free again.
            "CREATE SCHEMA s",
            "CREATE TABLE s.ch (id INT PRIMARY KEY, pid INT REFERENCES par ON DELETE CASCADE)",
            "CREATE TABLE s.p2 (id INT PRIMARY KEY)",
            "CREATE VIEW s.v AS SELECT 1 AS x",
            "CREATE SEQUENCE s.sq",
            "INSERT INTO s.ch VALUES (2, 2)",
            "DELETE FROM par WHERE id = 2",
        ],
    )
    .await;
    assert_eq!(q(&mut c, "SELECT COUNT(*) FROM s.ch").await, ["D:0"]);
    assert_eq!(
        q(&mut c, "SELECT id, pid FROM oc ORDER BY id").await,
        ["D:1,1", "D:2,99"]
    );
    assert!(
        nusadb_sql::orphaned_schema_objects(&*engine)
            .unwrap()
            .is_empty()
    );
    drop(c);
    handle.await.unwrap().unwrap();
    drop(engine);
    // The engine's removal of the dangling keys, constraints and indexes was logged: it holds
    // after a further restart, with nothing left to remove.
    let engine = engine_at(&wal);
    let (mut c, _) = open(&engine).await;
    ok(
        &mut c,
        &["INSERT INTO oc VALUES (3, 1000)", "DELETE FROM par"],
    )
    .await;
    assert_eq!(q(&mut c, "SELECT COUNT(*) FROM s.ch").await, ["D:0"]);
}

/// A dangling foreign key a running server still holds (one it has not reopened since) does not
/// refuse the statements on its live end: the engine below is built in that state directly.
#[tokio::test]
async fn a_dangling_foreign_key_is_tolerated_until_the_next_open() {
    use nusadb_core::{FkAction, ForeignKeyDef, IsolationLevel};
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, _) = open(&engine).await;
    ok(
        &mut c,
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "INSERT INTO par VALUES (1), (2)",
            "CREATE TABLE gone (id INT, pid INT)",
            "CREATE TABLE oc (id INT, pid INT)",
            "CREATE TABLE gp (id INT PRIMARY KEY)",
        ],
    )
    .await;
    let par = engine.lookup_table("par").unwrap().unwrap().id;
    let gone = engine.lookup_table("gone").unwrap().unwrap().id;
    let oc = engine.lookup_table("oc").unwrap().unwrap().id;
    let gp = engine.lookup_table("gp").unwrap().unwrap().id;
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    for (name, child, parent) in [("gone_fk", gone, par), ("oc_fk", oc, gp)] {
        engine
            .add_foreign_key(
                txn,
                &ForeignKeyDef {
                    name: name.to_owned(),
                    child_table: child,
                    child_columns: vec!["pid".to_owned()],
                    parent_table: parent,
                    parent_columns: Vec::new(),
                    on_delete: FkAction::Cascade,
                    on_update: FkAction::NoAction,
                },
            )
            .unwrap();
    }
    // Dropped behind the SQL layer's back, the way an earlier build's DROP SCHEMA did.
    engine.drop_table(txn, gone).unwrap();
    engine.drop_table(txn, gp).unwrap();
    engine.commit(txn).unwrap();
    ok(
        &mut c,
        &[
            "DELETE FROM par WHERE id = 1",
            "UPDATE par SET id = 3 WHERE id = 2",
            "INSERT INTO oc VALUES (1, 42)",
        ],
    )
    .await;
}

/// Every foreign key the engine holds names two tables that exist.
fn assert_no_dangling_foreign_key(engine: &Arc<dyn StorageEngine>) {
    let txn = engine
        .begin(nusadb_core::IsolationLevel::default())
        .unwrap();
    for (schema, name) in engine.list_tables_qualified_as_of(txn).unwrap() {
        let Some(table) = engine.lookup_table_as_of_in(txn, &schema, &name).unwrap() else {
            continue;
        };
        for fk in engine.list_foreign_keys_as_of(txn, table.id).unwrap() {
            for end in [fk.child_table, fk.parent_table] {
                assert!(
                    engine.lookup_table_by_id_as_of(txn, end).unwrap().is_some(),
                    "foreign key {} names a dropped table",
                    fk.name
                );
            }
        }
    }
    engine.rollback(txn).unwrap();
}

/// A table another transaction creates in the schema and has not committed refuses the drop with
/// a serialization failure instead of being dropped without its foreign keys; once it commits, a
/// retry drops it with everything else.
#[tokio::test]
async fn an_uncommitted_table_in_the_schema_refuses_the_drop() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine).await;
    let (mut b, _) = open(&engine).await;
    ok(
        &mut a,
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "CREATE SCHEMA s",
            "CREATE TABLE s.t1 (id INT, pid INT REFERENCES par)",
        ],
    )
    .await;
    ok(&mut b, &["BEGIN", "CREATE TABLE s.late (id INT)"]).await;
    assert_eq!(q(&mut a, "DROP SCHEMA s CASCADE").await, ["E:40001"]);
    assert_eq!(
        q(
            &mut a,
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 's'"
        )
        .await,
        ["D:1"]
    );
    ok(&mut b, &["COMMIT"]).await;
    ok(
        &mut a,
        &[
            "DROP SCHEMA s CASCADE",
            "INSERT INTO par VALUES (1)",
            "DELETE FROM par",
        ],
    )
    .await;
    assert_no_dangling_foreign_key(&engine);
}

/// Tables committed into the schema while `DROP SCHEMA ... CASCADE` runs are either dropped with
/// their foreign keys or make the drop fail; none is dropped bare, so no foreign key is left
/// naming a dropped table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tables_created_during_the_drop_are_never_dropped_bare() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine).await;
    ok(
        &mut a,
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "INSERT INTO par VALUES (1), (2)",
            "CREATE SCHEMA s",
        ],
    )
    .await;
    for i in 0..400 {
        let sql = format!("CREATE TABLE s.t{i} (id INT PRIMARY KEY, pid INT REFERENCES par)");
        ok(&mut a, &[sql.as_str()]).await;
    }
    let (mut creator, _) = open(&engine).await;
    let creating = tokio::spawn(async move {
        let mut created = 0;
        for k in 0..300 {
            let sql = format!("CREATE TABLE s.late{k} (id INT, pid INT REFERENCES par)");
            if q(&mut creator, &sql).await.is_empty() {
                created += 1;
            }
        }
        created
    });
    let (mut dropper, _) = open(&engine).await;
    let dropped = loop {
        match q(&mut dropper, "DROP SCHEMA s CASCADE").await.as_slice() {
            [] => break true,
            [e] if e == "E:40001" => {},
            other => panic!("{other:?}"),
        }
        if creating.is_finished() {
            break false;
        }
    };
    let created = creating.await.unwrap();
    assert!(created > 0, "no table was created while the drop ran");
    if !dropped {
        ok(&mut a, &["DROP SCHEMA s CASCADE"]).await;
    }
    assert_no_dangling_foreign_key(&engine);
    // The referenced table carries no foreign key of a dropped table.
    let par = engine.lookup_table("par").unwrap().unwrap();
    assert!(engine.list_foreign_keys(par.id).unwrap().is_empty());
    ok(&mut a, &["DELETE FROM par WHERE id = 1"]).await;
}

/// A default-schema view or sequence named with a dot is recorded as the default schema's, so the
/// check at open never reports it as left behind by a dropped schema, while one of a real schema
/// that is gone still would be.
#[tokio::test]
async fn a_dotted_default_schema_name_is_not_reported_at_open() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, _) = open(&engine).await;
    ok(
        &mut c,
        &[
            "CREATE VIEW public.\"gone.v\" AS SELECT 1 AS x",
            "CREATE SEQUENCE public.\"gone.q\"",
            "CREATE SCHEMA z",
            "CREATE VIEW z.v AS SELECT 2 AS x",
            "DROP SCHEMA z CASCADE",
        ],
    )
    .await;
    assert!(
        nusadb_sql::orphaned_schema_objects(&*engine)
            .unwrap()
            .is_empty()
    );
    assert_eq!(q(&mut c, "SELECT x FROM \"gone.v\"").await, ["D:1"]);
}
