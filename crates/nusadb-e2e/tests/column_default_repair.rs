//! The one-time repair of column defaults that builds before schema-qualified default keys filed
//! under a bare table name. The fixture is the log of a data directory such a build wrote:
//!
//! - `app.e` gained `flag INT DEFAULT 99` by `ADD COLUMN` while `public.e` has no `flag`, so the
//!   default can only be `app.e`'s;
//! - `app.f` gained `flag INT DEFAULT 7` by `ADD COLUMN` while `public.f` has a `flag` of its own,
//!   so the entry may be either table's;
//! - `x.g` and `y.g` both gained `v` with a default, so the entry may be either one's;
//! - `public.h` and `app.h` declared `c` with their own defaults, filed correctly.

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
use nusadb_sql::{AmbiguousDefault, repair_misfiled_column_defaults};
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

/// Connect as `user` and consume the handshake up to the first `ReadyForQuery`.
async fn open(
    engine: &Arc<dyn StorageEngine>,
    user: &str,
) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: user.to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    while !matches!(next(&mut conn).await, BackendMessage::ReadyForQuery(_)) {}
    (conn, handle)
}

/// Render one backend message compactly: `C:<tag>`, `E:<sqlstate>`, `D:<fields>`, `T` for a row
/// description, `Z:<I|T|E>`, or the message's debug form.
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
        BackendMessage::RowDescription { .. } | BackendMessage::RowDescriptionTyped { .. } => {
            "T".to_owned()
        },
        BackendMessage::ReadyForQuery(status) => match status {
            TxnStatus::Idle => "Z:I".to_owned(),
            TxnStatus::InTransaction => "Z:T".to_owned(),
            TxnStatus::Failed => "Z:E".to_owned(),
        },
        other => format!("{other:?}"),
    }
}

/// Every message up to and including `ReadyForQuery`, rendered.
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

/// Send a simple query and return its rendered response.
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

/// The data rows (and any error code) of a response, without the framing messages.
fn data(out: &[String]) -> Vec<String> {
    out.iter()
        .filter(|m| m.starts_with("D:") || m.starts_with("E:"))
        .cloned()
        .collect()
}

/// A copy of the fixture as the log of a fresh data directory under `dir`.
fn legacy_log(dir: &Path) -> std::path::PathBuf {
    let wal = dir.join("btree.wal");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/misfiled_column_defaults.wal"),
        &wal,
    )
    .expect("copy the fixture");
    wal
}

/// The defaults `information_schema.columns` reports, as `schema.table.column=default`.
async fn reported_defaults(conn: &mut Conn) -> Vec<String> {
    data(
        &q(
            conn,
            "SELECT table_schema || '.' || table_name || '.' || column_name || '=' || column_default \
             FROM information_schema.columns WHERE column_default IS NOT NULL ORDER BY 1",
        )
        .await,
    )
}

#[tokio::test]
async fn a_legacy_directory_is_repaired_once_and_stays_repaired() {
    let dir = tempfile::tempdir().expect("tempdir");
    let wal = legacy_log(dir.path());
    {
        let engine = BtreeEngine::open(&wal).expect("open the legacy log");
        let report = repair_misfiled_column_defaults(&engine).expect("repair");
        assert!(report.ran);
        assert_eq!(
            report.moved,
            [("app".to_owned(), "e".to_owned(), "flag".to_owned())]
        );
        let mut ambiguous = report.ambiguous;
        ambiguous.sort_by(|l, r| l.table.cmp(&r.table));
        assert_eq!(
            ambiguous,
            [
                AmbiguousDefault {
                    table: "f".to_owned(),
                    column: "flag".to_owned(),
                    default_sql: "7".to_owned(),
                    schemas: vec!["public".to_owned(), "app".to_owned()],
                },
                AmbiguousDefault {
                    table: "g".to_owned(),
                    column: "v".to_owned(),
                    default_sql: "2".to_owned(),
                    schemas: vec!["x".to_owned(), "y".to_owned()],
                },
            ]
        );
    }

    // Reopen from the log alone: the repair is durable, and it does not run twice.
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&wal).expect("reopen"));
    let again = repair_misfiled_column_defaults(&*engine).expect("second repair");
    assert!(!again.ran, "{again:?}");
    assert!(again.moved.is_empty() && again.ambiguous.is_empty());

    let (mut c, _) = open(&engine, "nusadb-root").await;
    for sql in [
        "INSERT INTO app.e (id) VALUES (2)",
        "INSERT INTO f (id) VALUES (2)",
        "INSERT INTO app.f (id) VALUES (2)",
        "INSERT INTO h (id) VALUES (2)",
        "INSERT INTO app.h (id) VALUES (2)",
    ] {
        assert!(data(&q(&mut c, sql).await).is_empty(), "{sql}");
    }
    // app.e now gets its default; the ambiguous entry still serves public.f, untouched.
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM app.e WHERE id = 2").await),
        ["D:99"]
    );
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM f WHERE id = 2").await),
        ["D:7"]
    );
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM app.f WHERE id = 2").await),
        ["D:NULL"]
    );
    assert_eq!(
        data(&q(&mut c, "SELECT c FROM h WHERE id = 2").await),
        ["D:5"]
    );
    assert_eq!(
        data(&q(&mut c, "SELECT c FROM app.h WHERE id = 2").await),
        ["D:6"]
    );
    assert_eq!(
        reported_defaults(&mut c).await,
        [
            "D:app.e.flag=99",
            "D:app.h.c=6",
            "D:public.f.flag=7",
            "D:public.h.c=5",
        ]
    );

    // The moved entry is gone from the bare name: a `flag` column added to public.e later does
    // not pick up app.e's default.
    for sql in [
        "ALTER TABLE e ADD COLUMN flag INT",
        "INSERT INTO e (id) VALUES (5)",
    ] {
        assert!(data(&q(&mut c, sql).await).is_empty(), "{sql}");
    }
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM e WHERE id = 5").await),
        ["D:NULL"]
    );

    // The documented manual fix for an ambiguous entry: set the default on the owner, drop it
    // from public.f.
    for sql in [
        "ALTER TABLE app.f ALTER COLUMN flag SET DEFAULT 7",
        "ALTER TABLE public.f ALTER COLUMN flag DROP DEFAULT",
        "INSERT INTO app.f (id) VALUES (3)",
        "INSERT INTO f (id) VALUES (3)",
    ] {
        assert!(data(&q(&mut c, sql).await).is_empty(), "{sql}");
    }
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM app.f WHERE id = 3").await),
        ["D:7"]
    );
    assert_eq!(
        data(&q(&mut c, "SELECT flag FROM f WHERE id = 3").await),
        ["D:NULL"]
    );
}

#[tokio::test]
async fn a_catalog_this_build_creates_needs_no_repair() {
    let dir = tempfile::tempdir().expect("tempdir");
    let wal = dir.path().join("btree.wal");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&wal).expect("open"));
        // No catalog yet: nothing to do.
        assert!(!repair_misfiled_column_defaults(&*engine).unwrap().ran);
        let (mut c, handle) = open(&engine, "nusadb-root").await;
        for sql in [
            "CREATE SCHEMA app",
            "CREATE TABLE e (id INT, flag INT DEFAULT 1)",
            "CREATE TABLE app.e (id INT)",
            "ALTER TABLE app.e ADD COLUMN flag INT DEFAULT 2",
        ] {
            assert!(data(&q(&mut c, sql).await).is_empty(), "{sql}");
        }
        // Close the connection so the server task lets go of the engine.
        drop(c);
        let _ = handle.await;
    }
    let engine = BtreeEngine::open(&wal).expect("reopen");
    let report = repair_misfiled_column_defaults(&engine).unwrap();
    assert!(
        !report.ran,
        "public.e's own default is not reported as ambiguous: {report:?}"
    );
}
