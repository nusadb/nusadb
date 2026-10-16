//! `COMMENT ON` between connections: two writers of one comment never leave two rows, a comment
//! and the drop of its object never both commit, only an owner (or a superuser) may comment, and
//! the comments view hides other sessions' temporary objects and tables the reader cannot see.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
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

const COMMENTS: &str = "SELECT object_type, object_name, description \
                        FROM information_schema.nusadb_comments ORDER BY 1, 2";

async fn engine_with_table() -> (Arc<dyn StorageEngine>, Conn, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine, "nusadb-root").await;
    let (b, _) = open(&engine, "nusadb-root").await;
    q(&mut a, "CREATE TABLE t (id INT PRIMARY KEY, c INT)").await;
    (engine, a, b)
}

#[tokio::test]
async fn two_writers_of_one_comment_never_leave_two_rows() {
    for level in [
        "",
        " ISOLATION LEVEL REPEATABLE READ",
        " ISOLATION LEVEL SERIALIZABLE",
    ] {
        let (_engine, mut a, mut b) = engine_with_table().await;
        q(&mut a, &format!("BEGIN{level}")).await;
        q(&mut b, &format!("BEGIN{level}")).await;
        assert_eq!(
            data(&q(&mut a, "COMMENT ON TABLE t IS 'A'").await),
            Vec::<String>::new()
        );
        assert_eq!(
            data(&q(&mut b, "COMMENT ON TABLE t IS 'B'").await),
            ["E:40001"]
        );
        q(&mut a, "COMMIT").await;
        q(&mut b, "ROLLBACK").await;
        assert_eq!(data(&q(&mut a, COMMENTS).await), ["D:TABLE,t,A"], "{level}");
    }
}

#[tokio::test]
async fn a_writer_after_a_commit_replaces_the_row() {
    let (_engine, mut a, mut b) = engine_with_table().await;
    q(&mut b, "BEGIN ISOLATION LEVEL REPEATABLE READ").await;
    q(&mut b, "SELECT 1").await;
    q(&mut a, "COMMENT ON TABLE t IS 'A'").await;
    // B's snapshot predates A's comment; it must not add a second row next to it.
    let out = data(&q(&mut b, "COMMENT ON TABLE t IS 'B'").await);
    q(&mut b, "COMMIT").await;
    let rows = data(&q(&mut a, COMMENTS).await);
    assert_eq!(rows.len(), 1, "{out:?} {rows:?}");
}

#[tokio::test]
async fn a_comment_and_the_drop_of_its_object_never_both_commit() {
    // Drop first, comment second.
    let (_engine, mut a, mut b) = engine_with_table().await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "DROP TABLE t").await;
    assert_eq!(
        data(&q(&mut b, "COMMENT ON TABLE t IS 'x'").await),
        ["E:40001"]
    );
    q(&mut a, "COMMIT").await;
    q(&mut a, "CREATE TABLE t (id INT)").await;
    assert!(data(&q(&mut a, COMMENTS).await).is_empty());

    // Comment first, drop second; the same for a column and for a function.
    let (_engine, mut a, mut b) = engine_with_table().await;
    q(
        &mut a,
        "CREATE FUNCTION f(x INT) RETURNS INT AS 'SELECT x' LANGUAGE SQL",
    )
    .await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "COMMENT ON TABLE t IS 'x'").await;
    q(&mut a, "COMMENT ON COLUMN t.c IS 'y'").await;
    q(&mut a, "COMMENT ON FUNCTION f IS 'z'").await;
    assert_eq!(data(&q(&mut b, "DROP TABLE t").await), ["E:40001"]);
    assert_eq!(
        data(&q(&mut b, "ALTER TABLE t DROP COLUMN c").await),
        ["E:40001"]
    );
    assert_eq!(data(&q(&mut b, "DROP FUNCTION f").await), ["E:40001"]);
    q(&mut a, "COMMIT").await;
    assert_eq!(
        data(&q(&mut a, COMMENTS).await),
        ["D:COLUMN,c,y", "D:FUNCTION,f,z", "D:TABLE,t,x"]
    );
}

#[tokio::test]
async fn only_an_owner_or_a_superuser_may_comment() {
    let (engine, mut root, _) = engine_with_table().await;
    for sql in [
        "CREATE ROLE alice LOGIN",
        "GRANT SELECT ON t TO alice",
        "CREATE SCHEMA s",
        "CREATE SEQUENCE sq",
        "CREATE FUNCTION f(x INT) RETURNS INT AS 'SELECT x' LANGUAGE SQL",
        "COMMENT ON TABLE t IS 'root'",
    ] {
        q(&mut root, sql).await;
    }
    let (mut alice, _) = open(&engine, "alice").await;
    for sql in [
        "COMMENT ON TABLE t IS 'alice'",
        "COMMENT ON COLUMN t.c IS 'alice'",
        "COMMENT ON CONSTRAINT t_pkey ON t IS 'alice'",
        "COMMENT ON SCHEMA s IS 'alice'",
        "COMMENT ON SCHEMA public IS 'alice'",
        "COMMENT ON SEQUENCE sq IS 'alice'",
        "COMMENT ON FUNCTION f IS 'alice'",
        "COMMENT ON TABLE t IS NULL",
    ] {
        assert_eq!(data(&q(&mut alice, sql).await), ["E:42501"], "{sql}");
    }
    // Alice holds SELECT on t, so she sees its comment; a table of her own she may comment on.
    assert_eq!(data(&q(&mut alice, COMMENTS).await), ["D:TABLE,t,root"]);
    q(&mut alice, "CREATE TABLE mine (id INT)").await;
    assert_eq!(
        data(&q(&mut alice, "COMMENT ON TABLE mine IS 'hers'").await),
        Vec::<String>::new()
    );
    // The superuser may comment on anything, `public` included.
    assert_eq!(
        data(&q(&mut root, "COMMENT ON SCHEMA public IS 'default'").await),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn the_view_hides_what_the_reader_cannot_see() {
    let (engine, mut root, mut other) = engine_with_table().await;
    q(&mut root, "CREATE ROLE bob LOGIN").await;
    q(&mut root, "COMMENT ON TABLE t IS 'private'").await;
    q(&mut root, "CREATE TEMP TABLE tt (id INT)").await;
    q(&mut root, "COMMENT ON TABLE tt IS 'temp'").await;
    assert_eq!(
        data(&q(&mut root, COMMENTS).await),
        ["D:TABLE,t,private", "D:TABLE,tt,temp"]
    );
    // Another session (a superuser too) does not see the first one's temporary objects.
    assert_eq!(data(&q(&mut other, COMMENTS).await), ["D:TABLE,t,private"]);
    // A role with no privilege on `t` sees no comment on it.
    let (mut bob, _) = open(&engine, "bob").await;
    assert!(data(&q(&mut bob, COMMENTS).await).is_empty());
}

#[tokio::test]
async fn an_empty_comment_removes_it() {
    let (_engine, mut a, _) = engine_with_table().await;
    q(&mut a, "COMMENT ON TABLE t IS 'x'").await;
    q(&mut a, "COMMENT ON TABLE t IS ''").await;
    assert!(data(&q(&mut a, COMMENTS).await).is_empty());
}
