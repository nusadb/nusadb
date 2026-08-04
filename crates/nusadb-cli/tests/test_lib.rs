//! Tests for the nusa-cli client library (`src/lib.rs`): result rendering (pure) and a full
//! query session driven against a real `nusadb-server` over TCP.

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_cli::{
    CopyIo, OutputFormat, collect_result, collect_result_with_copy, format_result, handshake,
    render_data_row, run_query, split_statements,
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

/// `COPY … FROM STDIN` loads the bytes it is handed, and `COPY … TO STDOUT` writes the rows back.
/// Before the client understood the COPY sub-protocol it ignored the server's `CopyInResponse` and
/// waited for a `ReadyForQuery` that could not come while the server waited for data — the session
/// hung until it was killed.
#[tokio::test]
async fn copy_streams_data_in_and_out() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));
    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "u", "nusadb", None).await.unwrap();
    collect_result(&mut conn, "CREATE TABLE t (id INT NOT NULL, s TEXT)")
        .await
        .unwrap();

    // Server text format: tab-delimited, `\N` for NULL.
    let mut source: &[u8] = b"1\talice\n2\t\\N\n3\tbob\n";
    let loaded = collect_result_with_copy(
        &mut conn,
        "COPY t FROM STDIN",
        CopyIo {
            source: Some(&mut source),
            sink: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(loaded.error, None, "COPY FROM STDIN reported an error");
    assert_eq!(loaded.tag.as_deref(), Some("COPY 3"));
    // The flag a batch reads to know the single input stream is spent. Without it a second COPY in
    // the same batch silently loads nothing.
    assert!(loaded.copied_in, "the load consumed the source stream");

    // The rows are really there, NULL included.
    let rows = collect_result(&mut conn, "SELECT id, s FROM t ORDER BY id")
        .await
        .unwrap();
    assert_eq!(
        rows.rows,
        vec![
            vec![Some(b"1".to_vec()), Some(b"alice".to_vec())],
            vec![Some(b"2".to_vec()), None],
            vec![Some(b"3".to_vec()), Some(b"bob".to_vec())],
        ]
    );

    // Export round-trips: what comes back out is what went in.
    let mut sink: Vec<u8> = Vec::new();
    let exported = collect_result_with_copy(
        &mut conn,
        "COPY t TO STDOUT",
        CopyIo {
            source: None,
            sink: Some(&mut sink),
        },
    )
    .await
    .unwrap();
    assert_eq!(exported.error, None);
    assert_eq!(exported.tag.as_deref(), Some("COPY 3"));
    // The flag that keeps the command tag off the data stream. Without it `COPY 3` is appended to
    // the exported rows and the file no longer loads back.
    assert!(exported.copied_out, "the export wrote to the sink");
    assert_eq!(
        String::from_utf8(sink).unwrap(),
        "1\talice\n2\t\\N\n3\tbob\n"
    );

    server.abort();
}

/// A `COPY` with no stream to serve it is refused and the session stays usable. The failure mode
/// this replaces was a client that waited forever, which no timeout in the CLI would have caught.
#[tokio::test]
async fn copy_without_a_stream_is_refused_not_hung() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));
    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "u", "nusadb", None).await.unwrap();
    collect_result(&mut conn, "CREATE TABLE t (id INT NOT NULL)")
        .await
        .unwrap();

    // `collect_result` attaches no streams. Bound the wait so a regression reports rather than
    // wedging the suite.
    let refused = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        collect_result(&mut conn, "COPY t FROM STDIN"),
    )
    .await
    .expect("COPY without input must not hang")
    .unwrap();
    let err = refused.error.expect("expected a refusal");
    assert!(
        err.contains("needs input"),
        "wanted the needs-input refusal, got `{err}`"
    );

    // The connection is still in protocol sync afterwards.
    let after = collect_result(&mut conn, "SELECT count(*) FROM t")
        .await
        .unwrap();
    assert_eq!(after.rows, vec![vec![Some(b"0".to_vec())]]);

    server.abort();
}

/// `COPY … TO STDOUT` with nowhere to write is refused, and the rows the server already sent are
/// discarded so the session stays in protocol sync. That direction never reads from the client, so
/// the refusal cannot be a reply — it has to be a local error.
#[tokio::test]
async fn copy_out_without_a_sink_is_refused_and_drains() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));
    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
    handshake(&mut conn, "u", "nusadb", None).await.unwrap();
    collect_result(&mut conn, "CREATE TABLE t (id INT NOT NULL)")
        .await
        .unwrap();
    collect_result(&mut conn, "INSERT INTO t VALUES (1), (2)")
        .await
        .unwrap();

    let refused = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        collect_result(&mut conn, "COPY t TO STDOUT"),
    )
    .await
    .expect("COPY TO STDOUT without a sink must not hang")
    .unwrap();
    let err = refused.error.expect("expected a refusal");
    assert!(
        err.contains("had no sink"),
        "wanted the no-sink refusal, got `{err}`"
    );
    assert!(!refused.copied_out, "nothing was written anywhere");

    // Still in sync: the discarded rows did not desynchronise the frame stream.
    let after = collect_result(&mut conn, "SELECT count(*) FROM t")
        .await
        .unwrap();
    assert_eq!(after.rows, vec![vec![Some(b"2".to_vec())]]);

    server.abort();
}
