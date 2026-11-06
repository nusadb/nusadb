//! Column grants over a connection. The wire path answers column checks from one read of the grants
//! per statement, so these tests pin that the read lasts no longer than the statement: a grant or a
//! revoke reaches the next statement of the same transaction, and from another connection too.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic over linear protocol scripts"
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

/// Connect as `user` and consume the handshake up to the first `ReadyForQuery`.
async fn open(engine: &Arc<dyn StorageEngine>, user: &str) -> Conn {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
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
    conn
}

/// What a statement came back with: the SQLSTATE of its error, or the fields of its rows.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Refused(String),
    Rows(Vec<Vec<Option<String>>>),
}

async fn run(conn: &mut Conn, sql: &str) -> Outcome {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut rows = Vec::new();
    let mut refused = None;
    loop {
        match next(conn).await {
            BackendMessage::DataRow { values } => rows.push(
                values
                    .into_iter()
                    .map(|v| v.map(|bytes| String::from_utf8(bytes).unwrap()))
                    .collect(),
            ),
            BackendMessage::Error { code, .. } => refused = Some(code),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    refused.map_or(Outcome::Rows(rows), Outcome::Refused)
}

/// Run `sql` and expect it not to be refused.
async fn ok(conn: &mut Conn, sql: &str) -> Vec<Vec<Option<String>>> {
    match run(conn, sql).await {
        Outcome::Rows(rows) => rows,
        Outcome::Refused(code) => panic!("`{sql}` was refused with {code}"),
    }
}

/// Run `sql` and expect a permission denial.
async fn denied(conn: &mut Conn, sql: &str) {
    assert_eq!(
        run(conn, sql).await,
        Outcome::Refused("42501".to_owned()),
        "{sql}"
    );
}

/// An engine with `t (id, v, s)`, one row, and a role that may read `id` and `s` and write `v`.
async fn fixture() -> (Arc<dyn StorageEngine>, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    for sql in [
        "CREATE TABLE t (id INT, v INT, s INT)",
        "INSERT INTO t VALUES (1, 2, 3)",
        "CREATE ROLE clerk LOGIN",
        "GRANT SELECT (id, s), UPDATE (v) ON t TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    (engine, root)
}

#[tokio::test]
async fn a_revoke_from_another_connection_reaches_the_next_statement_of_an_open_transaction() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    let read = "UPDATE t SET v = s";

    ok(&mut clerk, "BEGIN").await;
    ok(&mut clerk, read).await;
    ok(&mut clerk, read).await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    denied(&mut clerk, read).await;
    ok(&mut clerk, "ROLLBACK").await;

    denied(&mut clerk, read).await;
    ok(&mut root, "GRANT SELECT (s) ON t TO clerk").await;
    ok(&mut clerk, read).await;
}

#[tokio::test]
async fn a_revoke_reaches_the_next_statement_of_its_own_transaction() {
    let (_engine, mut root) = fixture().await;
    let read = "UPDATE t SET v = s";

    ok(&mut root, "BEGIN").await;
    ok(&mut root, "SET ROLE clerk").await;
    ok(&mut root, read).await;
    ok(&mut root, "RESET ROLE").await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    ok(&mut root, "SET ROLE clerk").await;
    denied(&mut root, read).await;
}

#[tokio::test]
async fn a_wildcard_returning_through_a_parent_is_the_parents_columns_over_the_wire() {
    let (engine, mut root) = fixture().await;
    for sql in [
        "CREATE TABLE ip (id INT, v INT)",
        "CREATE TABLE ic (extra TEXT) INHERITS (ip)",
        "INSERT INTO ip VALUES (1, 10)",
        "INSERT INTO ic VALUES (2, 20, 'secret')",
        "GRANT SELECT (id, v), UPDATE (v), DELETE ON ip TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    let field = |v: &str| Some(v.to_owned());
    assert_eq!(
        ok(&mut clerk, "UPDATE ip AS t SET v = v + 1 RETURNING t.*").await,
        vec![vec![field("1"), field("11")], vec![field("2"), field("21")]]
    );
    assert_eq!(
        ok(&mut clerk, "DELETE FROM ip WHERE id = 2 RETURNING *").await,
        vec![vec![field("2"), field("21")]]
    );
}
