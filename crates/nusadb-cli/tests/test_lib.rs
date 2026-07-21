//! Tests for the nusa-cli client library (`src/lib.rs`): result rendering (pure) and a full
//! query session driven against a real `nusadb-server` over TCP.

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_cli::{
    OutputFormat, collect_result, format_result, handshake, render_data_row, run_query,
    split_statements,
};
use nusadb_core::StorageEngine;
use nusadb_wire::{AuthStore, Connection, ServerConfig, serve, serve_with_shutdown};
use tokio::net::{TcpListener, TcpStream};

#[test]
fn render_row_formats_values_and_null() {
    let row = vec![Some(b"1".to_vec()), None, Some(b"alice".to_vec())];
    assert_eq!(render_data_row(&row), "1 | NULL | alice");
}

#[tokio::test]
async fn cli_runs_sql_against_a_real_server() {
    // Boot a server on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));

    // Drive a session through the CLI client library.
    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "u", "nusadb", None).await.unwrap();

    assert_eq!(
        run_query(&mut conn, "CREATE TABLE t (id INT NOT NULL)")
            .await
            .unwrap(),
        vec!["CREATE TABLE".to_owned()]
    );
    assert_eq!(
        run_query(&mut conn, "INSERT INTO t VALUES (5)")
            .await
            .unwrap(),
        vec!["INSERT 1".to_owned()]
    );
    // SELECT renders: header line, one row, then the command tag.
    assert_eq!(
        run_query(&mut conn, "SELECT id FROM t").await.unwrap(),
        vec!["id".to_owned(), "5".to_owned(), "SELECT 1".to_owned(),]
    );
    // A bad statement renders a single error line and leaves the session usable.
    let err = run_query(&mut conn, "SELECT id FROM ghost").await.unwrap();
    assert_eq!(err.len(), 1);
    assert!(
        err[0].starts_with("ERROR"),
        "expected error line, got {:?}",
        err[0]
    );

    server.abort();
}

/// The CLI completes a SCRAM-SHA-256 handshake against a server started with `--auth-user`, then
/// runs queries; a wrong password is rejected (client).
#[tokio::test]
async fn cli_authenticates_with_scram_against_an_auth_server() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        auth: Some(Arc::new(
            AuthStore::from_passwords([("alice", "s3cret")]).unwrap(),
        )),
        ..Default::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        config,
        std::future::pending::<()>(),
    ));

    // Correct password → authenticated session that can run a query.
    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "alice", "nusa", Some("s3cret"))
        .await
        .unwrap();
    assert_eq!(
        run_query(&mut conn, "CREATE TABLE t (id INT NOT NULL)")
            .await
            .unwrap(),
        vec!["CREATE TABLE".to_owned()]
    );

    // Wrong password → the handshake fails (the server rejects the proof).
    let mut bad = Connection::new(TcpStream::connect(addr).await.unwrap());
    assert!(
        handshake(&mut bad, "alice", "nusa", Some("wrong"))
            .await
            .is_err(),
        "a wrong password must fail the handshake"
    );

    // No password supplied at all → a clear error rather than a hang.
    let mut none = Connection::new(TcpStream::connect(addr).await.unwrap());
    assert!(
        handshake(&mut none, "alice", "nusadb", None).await.is_err(),
        "a server that requires auth must fail fast when no password is given"
    );

    server.abort();
}

#[tokio::test]
async fn batch_collects_structured_results_and_formats_them() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));

    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "u", "nusadb", None).await.unwrap();

    // A `--command`-style batch: split on `;`, run each, collect structured results.
    let batch = "CREATE TABLE t (id INT NOT NULL, name TEXT); \
                 INSERT INTO t VALUES (5, 'alice'); \
                 SELECT id, name FROM t";
    let stmts = split_statements(batch);
    assert_eq!(stmts.len(), 3);

    let mut last = None;
    for stmt in &stmts {
        last = Some(collect_result(&mut conn, stmt).await.unwrap());
    }
    let select = last.unwrap();
    assert_eq!(select.columns, vec!["id".to_owned(), "name".to_owned()]);
    assert_eq!(select.rows.len(), 1);
    assert_eq!(select.tag.as_deref(), Some("SELECT 1"));
    assert!(select.error.is_none());

    // The structured result renders correctly through the user-facing formats.
    assert_eq!(
        format_result(&select, OutputFormat::Csv),
        vec!["id,name".to_owned(), "5,alice".to_owned()]
    );
    assert_eq!(
        format_result(&select, OutputFormat::Json),
        vec![r#"[{"id":"5","name":"alice"}]"#.to_owned()]
    );

    server.abort();
}
