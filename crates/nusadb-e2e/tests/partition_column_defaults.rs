//! Column defaults, sequences and column types on partitioned tables over the wire: `COPY` through
//! the parent and into a partition, the statements a client sends most often, and what a partition
//! keeps across a restart.

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
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, handle_client};
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

/// Render one backend message: `C:<tag>`, `E:<sqlstate>:<message>`, `D:<fields>`.
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
        other => format!("{other:?}"),
    }
}

async fn until_ready(conn: &mut Conn) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        if matches!(msg, BackendMessage::ReadyForQuery(_)) {
            return out;
        }
        out.push(render(&msg));
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

/// Run `sql` and require it to fail with `code`.
async fn fails(conn: &mut Conn, sql: &str, code: &str) {
    let out = q(conn, sql).await;
    assert_eq!(error_code(&out).as_deref(), Some(code), "{sql}: {out:?}");
}

/// The data rows of `sql`.
async fn rows(conn: &mut Conn, sql: &str) -> Vec<String> {
    q(conn, sql)
        .await
        .into_iter()
        .filter_map(|m| m.strip_prefix("D:").map(str::to_owned))
        .collect()
}

/// Run a `COPY ... FROM STDIN` with `data` as its payload; returns the response after the data.
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

/// A partitioned table with a default, a sequence, an enum column and a generated column, and
/// two partitions.
async fn schema(c: &mut Conn) {
    ok(c, "CREATE TYPE mood AS ENUM ('sad', 'happy', 'ok')").await;
    ok(
        c,
        "CREATE TABLE pc (id INT, d INT NOT NULL, x INT DEFAULT 7, s SERIAL, \
         m mood DEFAULT 'ok', g INT GENERATED ALWAYS AS (id * 10) STORED) PARTITION BY LIST (d)",
    )
    .await;
    ok(c, "CREATE TABLE pc1 PARTITION OF pc FOR VALUES IN (1)").await;
    ok(c, "CREATE TABLE pc2 PARTITION OF pc FOR VALUES IN (2)").await;
}

#[tokio::test]
async fn copy_fills_defaults_through_the_parent_and_into_a_partition() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = open(&engine).await;
    schema(&mut c).await;
    let out = copy_in(
        &mut c,
        "COPY pc (id, d, m) FROM STDIN",
        "1\t1\thappy\n2\t2\tsad\n",
    )
    .await;
    assert_eq!(error_code(&out), None, "{out:?}");
    let out = copy_in(&mut c, "COPY pc1 (id, d) FROM STDIN", "3\t1\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    // A row outside the partition, an unknown label, and a key no partition takes.
    let out = copy_in(&mut c, "COPY pc1 (id, d) FROM STDIN", "4\t2\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23514"), "{out:?}");
    let out = copy_in(&mut c, "COPY pc (id, d, m) FROM STDIN", "5\t1\tangry\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("22P02"), "{out:?}");
    let out = copy_in(&mut c, "COPY pc (id, d) FROM STDIN", "6\t3\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23514"), "{out:?}");
    assert_eq!(
        rows(&mut c, "SELECT id, d, x, s, m, g FROM pc ORDER BY id").await,
        ["1,1,7,1,happy,10", "2,2,7,2,sad,20", "3,1,7,3,ok,30"]
    );
}

#[tokio::test]
async fn the_reported_statements_take_defaults_and_enum_labels() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = open(&engine).await;
    ok(
        &mut c,
        "CREATE TABLE pd (id int, d int NOT NULL, x int DEFAULT 7, s serial) PARTITION BY LIST (d)",
    )
    .await;
    ok(&mut c, "CREATE TABLE pd1 PARTITION OF pd FOR VALUES IN (1)").await;
    ok(&mut c, "INSERT INTO pd (id, d) VALUES (1, 1)").await;
    ok(&mut c, "INSERT INTO pd1 (id, d) VALUES (2, 1)").await;
    assert_eq!(
        rows(&mut c, "SELECT id, d, x, s FROM pd ORDER BY id").await,
        ["1,1,7,1", "2,1,7,2"]
    );
    ok(&mut c, "CREATE TYPE mood AS ENUM ('sad','happy')").await;
    ok(
        &mut c,
        "CREATE TABLE pe (id int, d int NOT NULL, m mood) PARTITION BY LIST (d)",
    )
    .await;
    ok(&mut c, "CREATE TABLE pe1 PARTITION OF pe FOR VALUES IN (1)").await;
    ok(&mut c, "INSERT INTO pe VALUES (1, 1, 'happy')").await;
    ok(&mut c, "INSERT INTO pe1 VALUES (2, 1, 'happy')").await;
    ok(&mut c, "INSERT INTO pe VALUES (3, 1, 'happy'::mood)").await;
    fails(&mut c, "INSERT INTO pe VALUES (4, 1, 'angry')", "22P02").await;
    fails(&mut c, "INSERT INTO pe1 VALUES (4, 1, 'angry')", "22P02").await;
    assert_eq!(
        rows(&mut c, "SELECT id, m FROM pe ORDER BY id").await,
        ["1,happy", "2,happy", "3,happy"]
    );
    // UPDATE ... SET col = DEFAULT, through the parent and on the partition.
    ok(&mut c, "UPDATE pd SET x = 0").await;
    ok(
        &mut c,
        "UPDATE pd SET x = DEFAULT, s = DEFAULT WHERE id = 1",
    )
    .await;
    ok(&mut c, "UPDATE pd1 SET x = DEFAULT WHERE id = 2").await;
    assert_eq!(
        rows(&mut c, "SELECT id, x, s FROM pd ORDER BY id").await,
        ["1,7,3", "2,7,2"]
    );
}

#[tokio::test]
async fn a_partition_keeps_its_defaults_and_types_across_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        schema(&mut c).await;
        ok(&mut c, "CREATE DOMAIN posint AS INT CHECK (VALUE > 0)").await;
        ok(
            &mut c,
            "CREATE TABLE pp (id SERIAL, d INT NOT NULL, p posint, v VARCHAR(2)) \
             PARTITION BY RANGE (d)",
        )
        .await;
        ok(
            &mut c,
            "CREATE TABLE pp_lo PARTITION OF pp FOR VALUES FROM (0) TO (10)",
        )
        .await;
        ok(&mut c, "INSERT INTO pc (id, d) VALUES (1, 1)").await;
        ok(&mut c, "INSERT INTO pc2 (id, d, m) VALUES (2, 2, 'sad')").await;
        ok(&mut c, "INSERT INTO pp (d, p) VALUES (1, 1)").await;
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    // The partitions still fill the parent's defaults from the one sequence, resolve labels,
    // compute the generated column, and enforce the domain and the length.
    ok(&mut c, "INSERT INTO pc1 (id, d, m) VALUES (3, 1, 'happy')").await;
    ok(&mut c, "INSERT INTO pc (id, d) VALUES (4, 2)").await;
    fails(
        &mut c,
        "INSERT INTO pc2 (id, d, m) VALUES (5, 2, 'angry')",
        "22P02",
    )
    .await;
    assert_eq!(
        rows(&mut c, "SELECT id, d, x, s, m, g FROM pc ORDER BY id").await,
        [
            "1,1,7,1,ok,10",
            "2,2,7,2,sad,20",
            "3,1,7,3,happy,30",
            "4,2,7,4,ok,40"
        ]
    );
    ok(&mut c, "INSERT INTO pp_lo (d, p) VALUES (2, 1)").await;
    fails(&mut c, "INSERT INTO pp_lo (d, p) VALUES (3, 0)", "23514").await;
    fails(&mut c, "INSERT INTO pp (d, v) VALUES (3, 'abc')", "22001").await;
    assert_eq!(
        rows(&mut c, "SELECT id, d, p FROM pp ORDER BY id").await,
        ["1,1,1", "2,2,1"]
    );
}
