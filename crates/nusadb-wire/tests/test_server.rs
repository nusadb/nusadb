//! Tests for `server` (`src/server.rs`) — the connection state machine, driven over an
//! in-memory duplex stream against a real `BtreeEngine`.

#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::panic,
    clippy::items_after_statements,
    reason = "integration test harness asserts via unwrap/panic; linear protocol scripts with \
              per-test local helper fns"
)]

use std::sync::Arc;
use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    AuthStore, BackendMessage, Connection, DescribeTarget, FrontendMessage, TxnStatus,
    handle_client, handle_client_with,
};
use tokio::io::{AsyncRead, AsyncWrite};

async fn next<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

async fn query<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>, sql: &str) {
    conn.write_frame(
        &FrontendMessage::Query {
            sql: sql.to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
}

fn cc(tag: &str) -> BackendMessage {
    BackendMessage::CommandComplete {
        tag: tag.to_owned(),
    }
}

/// Consume the post-auth handshake chatter — `BackendKeyData` and the startup `ParameterStatus`
/// reports — up to and including the initial `ReadyForQuery(Idle)`.
async fn consume_until_ready<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    loop {
        match next(conn).await {
            BackendMessage::ReadyForQuery(status) => {
                assert_eq!(status, TxnStatus::Idle);
                return;
            },
            BackendMessage::BackendKeyData { .. } | BackendMessage::ParameterStatus { .. } => {},
            other => panic!("unexpected pre-ready handshake message: {other:?}"),
        }
    }
}

/// Send Startup and consume `AuthOk` + `BackendKeyData` + `ParameterStatus`* + the initial
/// `ReadyForQuery`.
async fn start_session<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(conn).await, BackendMessage::AuthOk);
    consume_until_ready(conn).await;
}

/// A unique-key point lookup runs on the reactor-inline path — the fires counter
/// advances across the round trip (anti-vacuous-gate: the pin fails if the gate stops
/// admitting it) and the returned row is exact. A candidate WITHOUT a unique index (range
/// predicate / non-indexed column) still answers correctly through the pool fallback.
#[tokio::test]
async fn point_get_runs_inline_and_fallback_stays_correct() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let _handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;

    query(&mut conn, "CREATE TABLE pg3 (id INT PRIMARY KEY, v TEXT)").await;
    assert_eq!(next(&mut conn).await, cc("CREATE TABLE"));
    consume_until_ready(&mut conn).await;
    query(
        &mut conn,
        "INSERT INTO pg3 VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .await;
    assert_eq!(next(&mut conn).await, cc("INSERT 3"));
    consume_until_ready(&mut conn).await;

    // The unique-key lookup: exactly one row, and the inline counter must advance (tests share
    // the process, so assert a monotonic increase, not an exact delta).
    let before = nusadb_wire::server::inline_point_get_count();
    query(&mut conn, "SELECT v FROM pg3 WHERE id = 2").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["v".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"b".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    consume_until_ready(&mut conn).await;
    assert!(
        nusadb_wire::server::inline_point_get_count() > before,
        "the unique-key lookup must run on the inline point-get path"
    );

    // Repeat inside an explicit transaction — the inline path serves the open-txn state too.
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    consume_until_ready_in_txn(&mut conn).await;
    let before_txn = nusadb_wire::server::inline_point_get_count();
    query(&mut conn, "SELECT v FROM pg3 WHERE id = 3").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["v".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"c".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    consume_until_ready_in_txn(&mut conn).await;
    assert!(nusadb_wire::server::inline_point_get_count() > before_txn);
    query(&mut conn, "COMMIT").await;
    assert_eq!(next(&mut conn).await, cc("COMMIT"));
    consume_until_ready(&mut conn).await;

    // The TRUE PUNT path (audit): an equality conjunct on a NON-indexed column IS a
    // syntactic candidate, but it plans to a SeqScan — the plan-shape gate refuses INSIDE the
    // inline attempt and re-dispatches the cloned statement to the pool. The result must be
    // exact and the fires counter must NOT advance (the extended-query tests never touch the
    // simple-query inline gate, so the counter is quiet across this window).
    let before_punt = nusadb_wire::server::inline_point_get_count();
    query(&mut conn, "SELECT id FROM pg3 WHERE v = 'a'").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    consume_until_ready(&mut conn).await;
    assert_eq!(
        nusadb_wire::server::inline_point_get_count(),
        before_punt,
        "a punted candidate must not count as an inline run"
    );

    // A range predicate is not even a candidate (no equality conjunct) — it skips the inline
    // branch entirely and answers from the pool: two rows in key order.
    query(&mut conn, "SELECT v FROM pg3 WHERE id > 1 ORDER BY id").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["v".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"b".to_vec())]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"c".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 2"));
    consume_until_ready(&mut conn).await;

    // A miss on the inline path is an empty, well-formed result — not an error.
    query(&mut conn, "SELECT v FROM pg3 WHERE id = 99").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["v".to_owned()]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    consume_until_ready(&mut conn).await;
}

/// Like [`consume_until_ready`] but inside a transaction block (`ReadyForQuery(InTransaction)`).
async fn consume_until_ready_in_txn<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    loop {
        match next(conn).await {
            BackendMessage::ReadyForQuery(status) => {
                assert_eq!(status, TxnStatus::InTransaction);
                return;
            },
            BackendMessage::BackendKeyData { .. } | BackendMessage::ParameterStatus { .. } => {},
            other => panic!("unexpected pre-ready message: {other:?}"),
        }
    }
}

/// The startup handshake reports run-time parameters (`ParameterStatus`) — at least `server_version`
/// (non-empty) — between authentication and the initial `ReadyForQuery`, so a client can configure
/// itself without a round-trip query.
#[tokio::test]
async fn startup_handshake_reports_parameter_status() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let _handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);

    // Collect every pre-ready frame; one must be ParameterStatus { server_version, <non-empty> }.
    let mut server_version = None;
    loop {
        match next(&mut conn).await {
            BackendMessage::ReadyForQuery(_) => break,
            BackendMessage::ParameterStatus { name, value } if name == "server_version" => {
                server_version = Some(value);
            },
            BackendMessage::ParameterStatus { .. } | BackendMessage::BackendKeyData { .. } => {},
            other => panic!("unexpected pre-ready handshake message: {other:?}"),
        }
    }
    assert!(
        server_version.is_some_and(|v| !v.is_empty()),
        "handshake must report a non-empty server_version"
    );
}

/// Explicit transactions over the wire: ROLLBACK discards, COMMIT persists, the transaction status
/// byte tracks `I`/`T`/`E`, and a failed statement aborts the transaction until ROLLBACK.
#[tokio::test]
async fn explicit_transactions_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;

    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // BEGIN moves the session into a transaction (status `T`).
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    // An INSERT inside the transaction keeps status `T`.
    query(&mut conn, "INSERT INTO t VALUES (1)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    // ROLLBACK discards the insert and returns to `I`.
    query(&mut conn, "ROLLBACK").await;
    assert_eq!(next(&mut conn).await, cc("ROLLBACK"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "SELECT id FROM t").await;
    // No RowDescription rows — the rolled-back insert is gone (SELECT 0).
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // BEGIN; INSERT; COMMIT persists the row.
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "INSERT INTO t VALUES (2)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "COMMIT").await;
    assert_eq!(next(&mut conn).await, cc("COMMIT"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "SELECT id FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"2".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // A failed statement inside a transaction aborts it (status `E`); intervening statements are
    // rejected until ROLLBACK.
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "SELECT * FROM ghost").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Failed)
    );
    // A statement in the aborted transaction is rejected.
    query(&mut conn, "INSERT INTO t VALUES (3)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Failed)
    );
    // ROLLBACK recovers.
    query(&mut conn, "ROLLBACK").await;
    assert_eq!(next(&mut conn).await, cc("ROLLBACK"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // The committed row 2 is still the only one (3 was never inserted).
    query(&mut conn, "SELECT id FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"2".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Savepoints over the wire (A-UR.03): `SAVEPOINT` / `RELEASE` / `ROLLBACK TO SAVEPOINT` work inside a
/// transaction; `ROLLBACK TO SAVEPOINT` undoes only the work after the savepoint and recovers a
/// failed transaction; a savepoint outside a transaction block is rejected.
#[tokio::test]
async fn savepoints_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;

    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // A savepoint outside a transaction block is rejected.
    query(&mut conn, "SAVEPOINT sp").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // BEGIN; INSERT(1); SAVEPOINT sp1; INSERT(2); ROLLBACK TO sp1 undoes only the second insert.
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "INSERT INTO t VALUES (1)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "SAVEPOINT sp1").await;
    assert_eq!(next(&mut conn).await, cc("SAVEPOINT"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "INSERT INTO t VALUES (2)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );
    query(&mut conn, "ROLLBACK TO SAVEPOINT sp1").await;
    assert_eq!(next(&mut conn).await, cc("ROLLBACK"));
    // Still inside the transaction after rolling back to the savepoint.
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    // A failed statement aborts the transaction, but ROLLBACK TO SAVEPOINT recovers it.
    query(&mut conn, "SELECT * FROM ghost").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Failed)
    );
    query(&mut conn, "ROLLBACK TO SAVEPOINT sp1").await;
    assert_eq!(next(&mut conn).await, cc("ROLLBACK"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    query(&mut conn, "COMMIT").await;
    assert_eq!(next(&mut conn).await, cc("COMMIT"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Only row 1 survived: row 2 was undone by ROLLBACK TO SAVEPOINT.
    query(&mut conn, "SELECT id FROM t ORDER BY id").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    drop(conn);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn full_session_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));

    let mut conn = Connection::new(client);

    // Startup handshake.
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // DDL + DML, each acknowledged by CommandComplete then ReadyForQuery.
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "INSERT INTO t VALUES (1, 'alice')").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // SELECT returns RowDescription + one DataRow + CommandComplete.
    query(&mut conn, "SELECT id, name FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned(), "name".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec()), Some(b"alice".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // A bad statement comes back as an Error, and the session stays usable.
    query(&mut conn, "SELECT * FROM ghost").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `SET` / `SHOW` / `RESET` work over the wire against a per-connection GUC store, and
/// `current_setting` reflects an earlier `SET` — closing the QA gap where `SHOW <guc>` over the wire
/// was rejected with "session-control requires a Session".
#[tokio::test]
async fn set_show_and_reset_session_variable_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;

    // SET records the value; the tag is the standard `SET`.
    query(&mut conn, "SET search_path = 'myschema'").await;
    assert_eq!(next(&mut conn).await, cc("SET"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // SHOW reads it back as a one-row result (this is the statement that used to be rejected).
    query(&mut conn, "SHOW search_path").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["search_path".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"myschema".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SHOW"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // `current_setting` observes the same SET (session state persists across statements).
    query(&mut conn, "SELECT current_setting('search_path')").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::RowDescription { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"myschema".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // RESET clears it; SHOW then reports the empty string (no built-in default for search_path).
    query(&mut conn, "RESET search_path").await;
    assert_eq!(next(&mut conn).await, cc("SET"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    query(&mut conn, "SHOW search_path").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["search_path".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SHOW"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // A well-known read-only GUC still reports its honest built-in default over the wire.
    query(&mut conn, "SHOW server_version").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["server_version".to_owned()]
        }
    );
    match next(&mut conn).await {
        BackendMessage::DataRow { values } => {
            assert_eq!(values.len(), 1);
            assert!(values[0].as_ref().is_some_and(|v| !v.is_empty()));
        },
        other => panic!("expected a DataRow for SHOW server_version, got {other:?}"),
    }
    assert_eq!(next(&mut conn).await, cc("SHOW"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Drive the extended-query protocol: Parse → Bind → Describe → Execute → Sync, plus Close.
#[tokio::test]
async fn extended_query_prepared_statement() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    // Startup, then seed a table via a simple query.
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    query(&mut conn, "INSERT INTO t VALUES (1), (2), (3)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Parse → Bind → Describe(portal) → Execute → Sync.
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "SELECT id FROM t ORDER BY id".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);

    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params: vec![],
            result_formats: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);

    conn.write_frame(
        &FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "p".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );

    // Execute with max_rows = 2 → two rows then PortalSuspended.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 2,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec())]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"2".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, BackendMessage::PortalSuspended);

    // Execute again → the last row then CommandComplete.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"3".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));

    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Close the portal.
    conn.write_frame(
        &FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "p".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::CloseComplete);

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// G6: `Describe(Portal)` must report row metadata WITHOUT executing the statement — a mutating
/// statement must not take effect until `Execute`.
#[tokio::test]
async fn describe_portal_does_not_execute_the_statement() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Parse + Bind an INSERT, then Describe its portal.
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "INSERT INTO t VALUES (1)".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);
    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params: vec![],
            result_formats: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);
    // Describe(Portal) on an INSERT → NoData (no result columns), and crucially no row is inserted.
    conn.write_frame(
        &FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "p".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::NoData);
    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // The table must still be empty — Describe did not run the INSERT.
    query(&mut conn, "SELECT id FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Execute actually performs the insert.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Now the row is visible.
    query(&mut conn, "SELECT id FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// G7: Describe(Statement) reports the real number of `$n` placeholders in the prepared statement,
/// not a hard-coded 0, so a driver that introspects parameter count binds correctly.
#[tokio::test]
async fn describe_statement_reports_parameter_count() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // Parse a statement with two placeholders, then Describe the statement.
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "SELECT * FROM t WHERE a = $1 AND b = $2".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);

    conn.write_frame(
        &FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "s".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ParameterDescription { count: 2 }
    );
    assert_eq!(next(&mut conn).await, BackendMessage::NoData);

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Extended query with a `$1` parameter: Parse a parameterized statement, Bind a value,
/// and confirm Execute returns the matching row.
#[tokio::test]
async fn extended_query_with_parameter() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;
    query(
        &mut conn,
        "INSERT INTO t VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')",
    )
    .await;
    assert_eq!(next(&mut conn).await, cc("INSERT 3"));
    let _ = next(&mut conn).await;

    // Parse a parameterized query, bind id = 2, and execute.
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "SELECT name FROM t WHERE id = $1".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);

    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params: vec![Some(b"2".to_vec())],
            result_formats: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);

    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"bob".to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));

    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A `Bind` requesting the binary result format makes `Execute` emit binary `DataRow`
/// fields — an INT comes back as its 8-byte big-endian encoding, not decimal text.
#[tokio::test]
async fn binary_result_format_emits_binary_data_rows() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;
    query(&mut conn, "INSERT INTO t VALUES (1)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 1"));
    let _ = next(&mut conn).await;

    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "SELECT id FROM t".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);

    // Request the binary format for the single result column.
    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params: vec![],
            result_formats: vec![1],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);

    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    // `1` in binary = i64 big-endian, not the text "1".
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(1_i64.to_be_bytes().to_vec())]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));

    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `COPY t FROM STDIN` drives the COPY sub-protocol: the server replies `CopyInResponse`, gathers
/// the client's `CopyData`/`CopyDone` stream (reassembled across chunks), bulk-loads it, and reports
/// `COPY n`. A `\N` field becomes SQL NULL.
#[tokio::test]
async fn copy_from_stdin_bulk_loads_rows() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;

    query(&mut conn, "COPY t (id, name) FROM STDIN").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyInResponse { columns: 2 }
    );

    // Two chunks, split mid-line, to exercise reassembly. Row 3's name is `\N` (NULL).
    conn.write_frame(
        &FrontendMessage::CopyData {
            data: b"1\talice\n2\t".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(
        &FrontendMessage::CopyData {
            data: b"bob\n3\t\\N\n".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::CopyDone.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(next(&mut conn).await, cc("COPY 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "SELECT id, name FROM t ORDER BY id").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned(), "name".to_owned()]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec()), Some(b"alice".to_vec())]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"2".to_vec()), Some(b"bob".to_vec())]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"3".to_vec()), None]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `COPY FROM` with a bad row aborts the whole load (all-or-nothing) and the session recovers.
#[tokio::test]
async fn copy_from_stdin_bad_row_aborts_the_load() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;

    query(&mut conn, "COPY t FROM STDIN").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyInResponse { columns: 0 }
    );
    // `oops` is not an integer → the load fails.
    conn.write_frame(
        &FrontendMessage::CopyData {
            data: b"1\noops\n".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::CopyDone.encode().unwrap())
        .await
        .unwrap();
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Nothing was committed (all-or-nothing).
    query(&mut conn, "SELECT id FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `COPY t TO STDOUT` streams the table's rows back as `CopyOutResponse` + `CopyData` + `CopyDone`,
/// then `COPY n`. A NULL renders as the `\N` marker.
#[tokio::test]
async fn copy_to_stdout_streams_rows() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;
    query(&mut conn, "INSERT INTO t VALUES (1, 'alice'), (2, NULL)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;

    query(&mut conn, "COPY t (id, name) TO STDOUT").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyOutResponse { columns: 2 }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyData {
            data: b"1\talice\n2\t\\N\n".to_vec()
        }
    );
    assert_eq!(next(&mut conn).await, BackendMessage::CopyDone);
    assert_eq!(next(&mut conn).await, cc("COPY 2"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `COPY FROM STDIN` aborts once the cumulative `CopyData` exceeds the configured byte cap
/// instead of buffering without bound. The session stays in protocol sync and nothing is loaded.
#[tokio::test]
async fn copy_from_stdin_aborts_when_over_byte_cap() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    // A tiny 8-byte cap so a small payload trips the guard.
    let handle = tokio::spawn(handle_client_with(
        server,
        engine,
        None,
        None,
        None,
        None,
        None,
        Some(Duration::from_mins(1)),
        Some(8),
    ));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await; // ReadyForQuery
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;

    query(&mut conn, "COPY t FROM STDIN").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyInResponse { columns: 0 }
    );
    // First chunk (6 bytes) is under the cap; the second pushes the cumulative total over 8.
    conn.write_frame(
        &FrontendMessage::CopyData {
            data: b"1\n2\n3\n".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(
        &FrontendMessage::CopyData {
            data: b"4\n5\n6\n7\n".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::CopyDone.encode().unwrap())
        .await
        .unwrap();
    // The load is rejected with an error, and the session recovers to ReadyForQuery.
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // Nothing was committed — the over-cap load loaded zero rows.
    query(&mut conn, "SELECT id FROM t").await;
    let _ = next(&mut conn).await; // RowDescription
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    let _ = next(&mut conn).await; // ReadyForQuery

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A connection that opens but never finishes the handshake is dropped after `handshake_timeout`,
/// freeing its slot, even though no idle timeout is configured (slowloris defence).
#[tokio::test]
async fn handshake_timeout_drops_a_stalled_connection() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    // No idle timeout, but a short handshake deadline that always applies.
    let handle = tokio::spawn(handle_client_with(
        server,
        engine,
        None,
        None,
        None,
        None,
        None,
        Some(Duration::from_millis(50)),
        None,
    ));
    // The client connects and then stalls — it never sends a Startup frame.
    let mut conn = Connection::new(client);

    // The server must close the connection on its own once the handshake deadline elapses, well
    // within this generous bound. A read on the client side then sees end-of-stream.
    let closed = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("server did not drop the stalled connection within the timeout");
    closed.unwrap().unwrap();
    assert!(
        conn.read_frame().await.unwrap().is_none(),
        "server should have closed the connection after the handshake timeout"
    );
}

/// Drive the client side of a SCRAM-SHA-256 handshake up to `client-final`, returning the
/// connection positioned to read the server's response. `password` is what the client proves.
async fn scram_handshake<S>(conn: &mut Connection<S>, user: &str, password: &str)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use nusadb_wire::auth::scram;

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: user.to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(conn).await,
        BackendMessage::AuthSasl {
            mechanisms: vec!["SCRAM-SHA-256".to_owned()],
        },
    );

    let client_nonce = scram::generate_nonce().unwrap();
    let client_first_bare = format!("n={user},r={client_nonce}");
    let client_first = format!("n,,{client_first_bare}");
    conn.write_frame(
        &FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_owned(),
            data: client_first.into_bytes(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();

    let BackendMessage::AuthSaslContinue { data } = next(conn).await else {
        panic!("expected AuthSaslContinue");
    };
    let server_first_msg = String::from_utf8(data).unwrap();
    let server_first = scram::ServerFirst::parse(&server_first_msg).unwrap();
    let client_final = scram::client_final_message(
        password,
        "n,,",
        &client_first_bare,
        &server_first_msg,
        &server_first,
    )
    .unwrap();
    conn.write_frame(
        &FrontendMessage::SaslResponse {
            data: client_final.into_bytes(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
}

/// A correct password completes the SCRAM handshake and the session then works normally.
#[tokio::test]
async fn scram_authentication_succeeds_with_correct_password() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let auth = Arc::new(AuthStore::from_passwords([("alice", "s3cret")]).unwrap());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client_with(
        server,
        engine,
        None,
        None,
        None,
        Some(auth),
        None,
        None,
        None,
    ));
    let mut conn = Connection::new(client);

    scram_handshake(&mut conn, "alice", "s3cret").await;

    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::AuthSaslFinal { .. }
    ));
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // The authenticated session runs queries normally.
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    let _ = next(&mut conn).await; // CommandComplete
    let _ = next(&mut conn).await; // ReadyForQuery

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A wrong password fails the handshake with an error and never reaches the query loop.
#[tokio::test]
async fn scram_authentication_fails_with_wrong_password() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let auth = Arc::new(AuthStore::from_passwords([("alice", "s3cret")]).unwrap());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client_with(
        server,
        engine,
        None,
        None,
        None,
        Some(auth),
        None,
        None,
        None,
    ));
    let mut conn = Connection::new(client);

    scram_handshake(&mut conn, "alice", "wrong-password").await;

    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// The server hands each session a `BackendKeyData`; a `CancelRequest` with that key on a fresh
/// connection is handled cleanly. With no statement in flight the cancel is a no-op, so the session
/// survives (the per-statement token is reset before each statement).
#[tokio::test]
async fn cancel_request_is_handled_and_an_idle_session_survives() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    let BackendMessage::BackendKeyData { pid, secret } = next(&mut conn).await else {
        panic!("expected BackendKeyData after AuthOk");
    };
    assert_ne!(pid, 0);
    // Skip the startup ParameterStatus reports up to the initial ReadyForQuery.
    consume_until_ready(&mut conn).await;

    // A separate connection sends a cancel request for this session's key.
    let (cancel_client, cancel_server) = tokio::io::duplex(1024);
    let cancel_handle = tokio::spawn(handle_client(cancel_server, Arc::clone(&engine)));
    let mut cancel_conn = Connection::new(cancel_client);
    cancel_conn
        .write_frame(
            &FrontendMessage::CancelRequest { pid, secret }
                .encode()
                .unwrap(),
        )
        .await
        .unwrap();
    // The cancel protocol is fire-and-disconnect (no read), so flush before dropping.
    cancel_conn.flush_now().await.unwrap();
    drop(cancel_conn);
    cancel_handle.await.unwrap().unwrap(); // the cancel connection closed cleanly

    // The original session is unharmed by a between-statement cancel.
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// An error inside an extended-query pipeline is reported, and the server then skips messages
/// until the next Sync (skip-until-Sync semantics).
#[tokio::test]
async fn extended_query_error_skips_until_sync() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // Bind to a statement that was never parsed → Error, then the pipeline is poisoned until Sync.
    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "ghost".to_owned(),
            params: vec![],
            result_formats: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));

    // An Execute after the error is ignored (no reply) — only Sync gets a response.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn rejects_missing_startup() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(4096);
    let handle = tokio::spawn(handle_client(server, engine));

    let mut conn = Connection::new(client);
    // Send a Query before Startup → server replies Error and closes.
    query(&mut conn, "SELECT 1").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));
    handle.await.unwrap().unwrap();
}

/// `Describe(Statement)` on a prepared statement that declares more `$n` placeholders than the
/// wire's u16 `ParameterDescription` count can carry must fail cleanly — not silently saturate the
/// reported count to `u16::MAX`, which would under-report the real parameter count.
#[tokio::test]
async fn describe_statement_with_too_many_params_fails_rather_than_saturating() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // `$65536` → parameter_count = 65536, one past u16::MAX (the wire count field's capacity).
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "big".to_owned(),
            sql: "SELECT $65536".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);

    conn.write_frame(
        &FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "big".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    // A clean error, NOT a ParameterDescription { count: u16::MAX } that under-reports.
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Row-level security is enforced by the **connection's authenticated user** over the wire,
/// not just in unit tests: a non-superuser connection is filtered by policy, the bootstrap
/// superuser bypasses RLS. This is the live-enforcement path we flagged as a pre-publish blocker.
#[tokio::test]
async fn row_level_security_is_enforced_per_connection_user() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());

    // Open a trust-on-startup connection declaring `user`, completing the handshake.
    async fn connect(
        engine: &Arc<dyn StorageEngine>,
        user: &str,
    ) -> (
        Connection<tokio::io::DuplexStream>,
        tokio::task::JoinHandle<std::io::Result<()>>,
    ) {
        let (client, server) = tokio::io::duplex(64 * 1024);
        let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
        let mut conn = Connection::new(client);
        conn.write_frame(
            &FrontendMessage::Startup {
                major: 1,
                minor: 0,
                user: user.to_owned(),
                database: "d".to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
        consume_until_ready(&mut conn).await;
        (conn, handle)
    }

    // Send `sql`, drain to ReadyForQuery, asserting no error.
    async fn run_ok(conn: &mut Connection<tokio::io::DuplexStream>, sql: &str) {
        query(conn, sql).await;
        loop {
            match next(conn).await {
                BackendMessage::Error { message, .. } => {
                    panic!("unexpected error for `{sql}`: {message}")
                },
                BackendMessage::ReadyForQuery(_) => break,
                _ => {},
            }
        }
    }

    // Send a single-column `SELECT id`, returning the id cells (raw bytes) in row order.
    async fn select_ids(conn: &mut Connection<tokio::io::DuplexStream>, sql: &str) -> Vec<Vec<u8>> {
        query(conn, sql).await;
        let mut ids = Vec::new();
        loop {
            match next(conn).await {
                BackendMessage::DataRow { values } => {
                    ids.push(values[0].clone().expect("non-null id"));
                },
                BackendMessage::Error { message, .. } => {
                    panic!("unexpected error for `{sql}`: {message}")
                },
                BackendMessage::ReadyForQuery(_) => break,
                _ => {},
            }
        }
        ids
    }

    async fn terminate(
        mut conn: Connection<tokio::io::DuplexStream>,
        handle: tokio::task::JoinHandle<std::io::Result<()>>,
    ) {
        conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
            .await
            .unwrap();
        drop(conn);
        handle.await.unwrap().unwrap();
    }

    // As the bootstrap superuser: create the table, enable RLS, and add an owner-scoped policy.
    let (mut su, su_handle) = connect(&engine, "nusa-root").await;
    run_ok(&mut su, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)").await;
    run_ok(
        &mut su,
        "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob'), (3, 'alice')",
    )
    .await;
    run_ok(&mut su, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY").await;
    run_ok(
        &mut su,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    )
    .await;
    // The superuser bypasses RLS: it sees all three rows.
    assert_eq!(
        select_ids(&mut su, "SELECT id FROM doc ORDER BY id")
            .await
            .len(),
        3
    );
    terminate(su, su_handle).await;

    // A non-superuser connection is filtered by the policy: alice sees only her two rows, and the
    // predicate's CURRENT_USER resolves to *this connection's* user.
    let (mut alice, alice_handle) = connect(&engine, "alice").await;
    assert_eq!(
        select_ids(&mut alice, "SELECT id FROM doc ORDER BY id").await,
        vec![b"1".to_vec(), b"3".to_vec()]
    );
    // COPY does not pass through the RLS-aware analyzer, so it fails closed for a non-superuser on
    // an RLS table — no wire path escapes enforcement.
    query(&mut alice, "COPY doc TO STDOUT").await;
    let mut copy_refused = false;
    loop {
        match next(&mut alice).await {
            BackendMessage::Error { message, .. } => {
                assert!(
                    message.contains("row-level security"),
                    "COPY refusal should cite row-level security, got: {message}"
                );
                copy_refused = true;
            },
            BackendMessage::ReadyForQuery(_) => break,
            other => panic!("unexpected message during COPY refusal: {other:?}"),
        }
    }
    assert!(
        copy_refused,
        "COPY must be refused for a non-superuser on an RLS table"
    );
    terminate(alice, alice_handle).await;

    // A different non-superuser the policy does not name sees nothing (default-deny).
    let (mut carol, carol_handle) = connect(&engine, "carol").await;
    assert!(
        select_ids(&mut carol, "SELECT id FROM doc")
            .await
            .is_empty()
    );
    terminate(carol, carol_handle).await;
}

/// A multi-row `SELECT` streams every row to the socket in order (Phase 2): `RowDescription`,
/// then one `DataRow` per row, then `CommandComplete` with the right count — the streamed frame
/// sequence is exactly what the buffered path produced.
#[tokio::test]
async fn large_select_streams_every_row_in_order() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;

    // Drain a statement to ReadyForQuery, asserting no error.
    async fn ok(conn: &mut Connection<tokio::io::DuplexStream>, sql: &str) {
        query(conn, sql).await;
        loop {
            match next(conn).await {
                BackendMessage::Error { message, .. } => panic!("unexpected error: {message}"),
                BackendMessage::ReadyForQuery(_) => break,
                _ => {},
            }
        }
    }

    const N: i64 = 60;
    ok(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    let tuples = (0..N)
        .map(|i| format!("({i})"))
        .collect::<Vec<_>>()
        .join(", ");
    ok(&mut conn, &format!("INSERT INTO t VALUES {tuples}")).await;

    // RowDescription first.
    query(&mut conn, "SELECT id FROM t ORDER BY id").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );
    // Then exactly N DataRows, in ascending id order.
    for i in 0..N {
        match next(&mut conn).await {
            BackendMessage::DataRow { values } => {
                assert_eq!(values, vec![Some(i.to_string().into_bytes())], "row {i}");
            },
            other => panic!("expected DataRow {i}, got {other:?}"),
        }
    }
    // Then CommandComplete with the full count, then ReadyForQuery.
    assert_eq!(next(&mut conn).await, cc("SELECT 60"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Send Startup with a chosen protocol `minor` and consume the post-auth setup.
async fn start_session_minor<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    minor: u16,
) {
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(conn).await, BackendMessage::AuthOk);
    consume_until_ready(conn).await;
}

/// A `minor >= 1` connection receives the typed `RowDescriptionTyped` (with per-column
/// type tags); a `minor = 0` connection receives the classic names-only `RowDescription`
/// (byte-identical to 1.0). Same query, two connections, two shapes.
#[tokio::test]
async fn typed_row_description_negotiated_by_minor() {
    use nusadb_core::ColumnType;
    use nusadb_wire::column_type_tag;

    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    // minor = 1 → typed metadata.
    start_session_minor(&mut conn, 1).await;
    query(
        &mut conn,
        "CREATE TABLE t (id INT NOT NULL, name TEXT, ok BOOL)",
    )
    .await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    query(&mut conn, "SELECT id, name, ok FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescriptionTyped {
            columns: vec![
                ("id".to_owned(), column_type_tag(ColumnType::Int)),
                ("name".to_owned(), column_type_tag(ColumnType::Text)),
                ("ok".to_owned(), column_type_tag(ColumnType::Bool)),
            ],
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    drop(conn);
    handle.await.unwrap().unwrap();

    // minor = 0 → classic untyped RowDescription (regression: 1.0 byte-identical).
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session_minor(&mut conn, 0).await;
    query(&mut conn, "SELECT id, name, ok FROM t").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned(), "name".to_owned(), "ok".to_owned()],
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `INSERT/UPDATE ... RETURNING` runs buffered then replays its row set into the wire sink, but it
/// must still advertise the projection's real per-column types — like a streamed SELECT — rather than
/// letting every column default to text. A strict-typed driver relies on `RETURNING id` being an
/// integer, not a string (QA RETURNING-type finding).
#[tokio::test]
async fn returning_row_description_is_typed() {
    use nusadb_core::ColumnType;
    use nusadb_wire::column_type_tag;

    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    start_session_minor(&mut conn, 1).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // The non-text `id` column must be typed `int`, and the computed `id + 1` too — not text.
    query(
        &mut conn,
        "INSERT INTO t VALUES (7, 'z') RETURNING id, name, id + 1",
    )
    .await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescriptionTyped {
            columns: vec![
                ("id".to_owned(), column_type_tag(ColumnType::Int)),
                ("name".to_owned(), column_type_tag(ColumnType::Text)),
                ("?column?".to_owned(), column_type_tag(ColumnType::Int)),
            ],
        }
    );
    // Drain the data row, command tag, and ready-for-query.
    while !matches!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    ) {}
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Protocol 1.2: an `ARRAY` column's type tag carries its element type for a `minor >= 2` connection
/// (so a client decodes the elements at their real type), while a `minor = 1` connection keeps the
/// plain `0x0F` ARRAY tag — backward compatible (array-elem-text finding).
#[tokio::test]
async fn array_column_tag_carries_element_type_at_minor_2() {
    use nusadb_core::{ArrayElem, ColumnType};
    use nusadb_wire::{column_type_tag, column_type_tag_v2};

    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());

    // minor 2 → the array column's tag carries its INT element type (0x82).
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session_minor(&mut conn, 2).await;
    query(&mut conn, "SELECT ARRAY[1,2,3] AS a").await;
    match next(&mut conn).await {
        BackendMessage::RowDescriptionTyped { columns } => {
            assert_eq!(columns.len(), 1);
            assert_eq!(
                columns[0].1,
                column_type_tag_v2(ColumnType::Array(ArrayElem::Int))
            );
            assert_eq!(columns[0].1, 0x82);
        },
        other => panic!("expected RowDescriptionTyped, got {other:?}"),
    }
    while !matches!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    ) {}
    drop(conn);
    handle.await.unwrap().unwrap();

    // minor 1 → the array column keeps the plain ARRAY tag (0x0F), unchanged from 1.1.
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session_minor(&mut conn, 1).await;
    query(&mut conn, "SELECT ARRAY[1,2,3] AS a").await;
    match next(&mut conn).await {
        BackendMessage::RowDescriptionTyped { columns } => {
            assert_eq!(
                columns[0].1,
                column_type_tag(ColumnType::Array(ArrayElem::Int))
            );
            assert_eq!(columns[0].1, 0x0F);
        },
        other => panic!("expected RowDescriptionTyped, got {other:?}"),
    }
    while !matches!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    ) {}
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// One logical response leaves the server as **one** OS-level write (frame coalescing).
///
/// The pre-coalescing server wrote+flushed every backend frame individually, so a simple query
/// produced >=4 TCP segments (`RowDescription` / `DataRow` / `CommandComplete` / `ReadyForQuery`)
/// and the client's delayed-ACK put a ~40ms floor under every query. This pins the fix: the whole
/// startup burst is one write, and each simple-query response is one more.
#[tokio::test]
async fn query_response_is_coalesced_into_a_single_write() {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    /// Delegating stream wrapper that counts the `poll_write` calls that reach the transport.
    struct WriteCounting<S> {
        inner: S,
        writes: Arc<AtomicUsize>,
    }
    impl<S: AsyncRead + Unpin> AsyncRead for WriteCounting<S> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_read(cx, buf)
        }
    }
    impl<S: AsyncWrite + Unpin> AsyncWrite for WriteCounting<S> {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            let poll = Pin::new(&mut self.inner).poll_write(cx, buf);
            if matches!(poll, Poll::Ready(Ok(_))) {
                self.writes.fetch_add(1, Ordering::SeqCst);
            }
            poll
        }
        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(cx)
        }
        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_shutdown(cx)
        }
    }

    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let writes = Arc::new(AtomicUsize::new(0));
    let counted = WriteCounting {
        inner: server,
        writes: Arc::clone(&writes),
    };
    let handle = tokio::spawn(handle_client(counted, Arc::clone(&engine)));
    let mut conn = Connection::new(client);

    // Startup: AuthOk + BackendKeyData + ParameterStatus* + ReadyForQuery arrive as ONE write.
    // (The bytes are only readable after the write was counted, so the assert cannot race.)
    start_session(&mut conn).await;
    assert_eq!(
        writes.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the whole startup burst must be coalesced into a single write"
    );

    // A simple query: RowDescription + DataRow + CommandComplete + ReadyForQuery — one more write.
    query(&mut conn, "SELECT 1").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::RowDescription { .. }
    ));
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::DataRow { .. }
    ));
    assert_eq!(next(&mut conn).await, cc("SELECT 1"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    assert_eq!(
        writes.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "a simple-query response must be coalesced into a single write"
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    conn.flush_now().await.unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// P-ISOLATION: the isolation level is honored over the wire, requested any of the three ways.
///
/// Observable behavior, not just a command tag: a REPEATABLE READ transaction keeps its snapshot
/// while a concurrent connection commits a row — under the old code every wire transaction ran
/// READ COMMITTED regardless, so the second read would have seen the new row.
#[tokio::test]
async fn transaction_isolation_is_settable_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client_a, server_a) = tokio::io::duplex(64 * 1024);
    let handle_a = tokio::spawn(handle_client(server_a, Arc::clone(&engine)));
    let mut a = Connection::new(client_a);
    start_session(&mut a).await;
    let (client_b, server_b) = tokio::io::duplex(64 * 1024);
    let handle_b = tokio::spawn(handle_client(server_b, Arc::clone(&engine)));
    let mut b = Connection::new(client_b);
    start_session(&mut b).await;

    /// Read a one-row/one-column SELECT result off `conn` and return the cell text.
    async fn one_cell<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) -> String {
        let mut cell = None;
        loop {
            match next(conn).await {
                BackendMessage::DataRow { values } => {
                    cell = Some(String::from_utf8(values[0].clone().unwrap()).unwrap());
                },
                BackendMessage::ReadyForQuery(_) => return cell.expect("a data row"),
                BackendMessage::RowDescription { .. }
                | BackendMessage::RowDescriptionTyped { .. }
                | BackendMessage::CommandComplete { .. } => {},
                other => panic!("unexpected message: {other:?}"),
            }
        }
    }
    /// Run a statement expecting only a `CommandComplete` tag.
    async fn ok<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>, sql: &str) {
        query(conn, sql).await;
        assert!(
            matches!(next(conn).await, BackendMessage::CommandComplete { .. }),
            "expected success for {sql}"
        );
        loop {
            if matches!(next(conn).await, BackendMessage::ReadyForQuery(_)) {
                break;
            }
        }
    }

    ok(&mut a, "CREATE TABLE iso (id INT NOT NULL)").await;
    ok(&mut a, "INSERT INTO iso VALUES (1)").await;

    // Three ways to request REPEATABLE READ; each must hold its snapshot across B's commit.
    let setups: [&[&str]; 3] = [
        &["BEGIN ISOLATION LEVEL REPEATABLE READ"],
        &["BEGIN", "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"],
        &[
            "SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ",
            "BEGIN",
        ],
    ];
    for (i, setup) in setups.iter().enumerate() {
        for stmt in *setup {
            ok(&mut a, stmt).await;
        }
        query(&mut a, "SELECT count(*) FROM iso").await;
        let before = one_cell(&mut a).await;
        ok(&mut b, "INSERT INTO iso VALUES (99)").await; // autocommit on the other connection
        query(&mut a, "SELECT count(*) FROM iso").await;
        let during = one_cell(&mut a).await;
        assert_eq!(
            before, during,
            "case {i}: a REPEATABLE READ snapshot must not see the concurrent commit"
        );
        ok(&mut a, "COMMIT").await;
        query(&mut a, "SELECT count(*) FROM iso").await;
        let after = one_cell(&mut a).await;
        assert_ne!(
            before, after,
            "case {i}: after COMMIT the new snapshot sees the committed row"
        );
        // Reset the session default so case 1/2 are not accidentally green via a leftover GUC.
        ok(&mut a, "RESET default_transaction_isolation").await;
    }

    // SHOW transaction_isolation reports the level actually enforced: the
    // active transaction's level inside a block, the session default outside one.
    ok(&mut a, "BEGIN ISOLATION LEVEL SERIALIZABLE").await;
    query(&mut a, "SHOW transaction_isolation").await;
    assert_eq!(one_cell(&mut a).await, "serializable");
    query(&mut a, "SELECT current_setting('transaction_isolation')").await;
    assert_eq!(one_cell(&mut a).await, "serializable");
    ok(&mut a, "COMMIT").await;
    query(&mut a, "SHOW transaction_isolation").await;
    assert_eq!(
        one_cell(&mut a).await,
        "read committed",
        "outside a block the session default is reported again"
    );
    ok(
        &mut a,
        "SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ",
    )
    .await;
    query(&mut a, "SHOW transaction_isolation").await;
    assert_eq!(
        one_cell(&mut a).await,
        "repeatable read",
        "in autocommit the session default is the effective level"
    );
    ok(&mut a, "RESET default_transaction_isolation").await;

    // SET TRANSACTION after a query in the block is refused (the reference engine: 25001) and aborts the block.
    ok(&mut a, "BEGIN").await;
    query(&mut a, "SELECT count(*) FROM iso").await;
    let _ = one_cell(&mut a).await;
    query(&mut a, "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE").await;
    assert!(matches!(next(&mut a).await, BackendMessage::Error { .. }));
    assert_eq!(
        next(&mut a).await,
        BackendMessage::ReadyForQuery(TxnStatus::Failed)
    );
    ok(&mut a, "ROLLBACK").await;

    // READ ONLY is refused loudly rather than silently opening a writable transaction.
    query(&mut a, "BEGIN READ ONLY").await;
    assert!(matches!(next(&mut a).await, BackendMessage::Error { .. }));
    assert_eq!(
        next(&mut a).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // An invalid default_transaction_isolation is rejected at SET (not silently at BEGIN).
    query(&mut a, "SET default_transaction_isolation = 'sirializable'").await;
    assert!(matches!(next(&mut a).await, BackendMessage::Error { .. }));
    assert_eq!(
        next(&mut a).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    for (mut conn, handle) in [(a, handle_a), (b, handle_b)] {
        conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
            .await
            .unwrap();
        conn.flush_now().await.unwrap();
        drop(conn);
        handle.await.unwrap().unwrap();
    }
}

/// `COPY FROM` into declared integer/float aliases: a BIGINT / SMALLINT / REAL
/// column bulk-loads exactly like its base type — the field is parsed by the column's *storage*
/// type, so an alias no longer falls to the Text path and dies at encode ("expected Int, found
/// Text") while the same INSERT worked.
#[tokio::test]
async fn copy_from_stdin_loads_bigint_and_alias_columns() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;
    query(
        &mut conn,
        "CREATE TABLE big (id BIGINT NOT NULL, small SMALLINT, r REAL)",
    )
    .await;
    let _ = next(&mut conn).await;
    let _ = next(&mut conn).await;

    query(&mut conn, "COPY big (id, small, r) FROM STDIN").await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CopyInResponse { columns: 3 }
    );
    conn.write_frame(
        &FrontendMessage::CopyData {
            // An id beyond i32 proves the value really lands as a 64-bit integer.
            data: b"5000000000\t7\t1.5\n2\t\\N\t\\N\n".to_vec(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::CopyDone.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(next(&mut conn).await, cc("COPY 2"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "SELECT id, small, r FROM big ORDER BY id").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::RowDescription { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"2".to_vec()), None, None]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![
                Some(b"5000000000".to_vec()),
                Some(b"7".to_vec()),
                Some(b"1.5".to_vec())
            ]
        }
    );
    assert_eq!(next(&mut conn).await, cc("SELECT 2"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    conn.flush_now().await.unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// `SET statement_timeout` must actually arm the per-statement cancel timer — it used to be echoed
/// by `SHOW` but silently ignored (the timer read only the server's `--statement-timeout`):
/// an unparseable value is rejected at SET time (22023); an armed 1ms timeout cancels a deep
/// recursion with the standard `57014` (`query_canceled`); `0` opts the session back out.
#[tokio::test]
async fn session_statement_timeout_cancels_a_long_statement() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let _handle = tokio::spawn(handle_client(server, Arc::clone(&engine)));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;

    // An unparseable value is rejected loudly at SET time, never stored-then-ignored.
    query(&mut conn, "SET statement_timeout = 'banana'").await;
    match next(&mut conn).await {
        BackendMessage::Error { code, message } => {
            assert_eq!(code, "22023", "expected invalid_parameter_value: {message}");
        },
        other => panic!("expected a 22023 error, got {other:?}"),
    }
    consume_until_ready(&mut conn).await;

    // Arm a 1ms session timeout, then run a recursion that takes far longer: the timer must trip
    // the cancel token and the statement must fail with query_canceled — not run to completion.
    query(&mut conn, "SET statement_timeout = '1ms'").await;
    assert_eq!(next(&mut conn).await, cc("SET"));
    consume_until_ready(&mut conn).await;
    query(
        &mut conn,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 2000000) \
         SELECT count(*) FROM r",
    )
    .await;
    loop {
        match next(&mut conn).await {
            BackendMessage::Error { code, message } => {
                assert_eq!(
                    code, "57014",
                    "expected query_canceled, got {code}: {message}"
                );
                break;
            },
            // The streaming path may announce the row shape before the cancellation lands.
            BackendMessage::RowDescription { .. } => {},
            BackendMessage::DataRow { .. } => {
                panic!("the statement completed despite the 1ms session statement_timeout")
            },
            other => panic!("unexpected message: {other:?}"),
        }
    }
    consume_until_ready(&mut conn).await;

    // `0` opts the session out of any timeout again: a small statement completes normally.
    query(&mut conn, "SET statement_timeout = '0'").await;
    assert_eq!(next(&mut conn).await, cc("SET"));
    consume_until_ready(&mut conn).await;
    query(
        &mut conn,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 100) \
         SELECT count(*) FROM r",
    )
    .await;
    let mut saw_row = false;
    loop {
        match next(&mut conn).await {
            BackendMessage::RowDescription { .. } => {},
            BackendMessage::DataRow { values } => {
                assert_eq!(values, vec![Some(b"100".to_vec())]);
                saw_row = true;
            },
            BackendMessage::CommandComplete { .. } => break,
            other => panic!("unexpected message: {other:?}"),
        }
    }
    assert!(saw_row, "the recursion result row never arrived");
    consume_until_ready(&mut conn).await;
}

// ---------------------------------------------------------------------------
// A fetch-all `Execute` (max_rows = 0) over a text-format portal streams its
// rows straight to the socket instead of materializing the whole result set first. These tests
// pin the observable protocol behaviour: rows arrive, `RowDescription` is NOT repeated on
// `Execute`, and every edge (empty result, re-`Execute`, parameters, binary fallback, error,
// transaction) matches the buffered path it replaces.
// ---------------------------------------------------------------------------

/// Start a server, finish startup, and seed `t(id INT)` with rows 1..=3. Returns the client
/// connection and the server task handle (await it after sending `Terminate`).
async fn seeded_three_rows() -> (
    Connection<tokio::io::DuplexStream>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, engine));
    let mut conn = Connection::new(client);
    conn.write_frame(
        &FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    consume_until_ready(&mut conn).await;
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    query(&mut conn, "INSERT INTO t VALUES (1), (2), (3)").await;
    assert_eq!(next(&mut conn).await, cc("INSERT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    (conn, handle)
}

/// Send `Parse`/`Bind` for a text-result portal named `p` over statement `s`, expecting the
/// `ParseComplete`/`BindComplete` acknowledgements.
async fn parse_bind_text(
    conn: &mut Connection<tokio::io::DuplexStream>,
    sql: &str,
    params: Vec<Option<Vec<u8>>>,
) {
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: sql.to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(conn).await, BackendMessage::ParseComplete);
    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params,
            result_formats: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(conn).await, BackendMessage::BindComplete);
}

/// Send `Execute(portal = "p", max_rows)` and `Sync`.
async fn execute_sync(conn: &mut Connection<tokio::io::DuplexStream>, max_rows: u32) {
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
}

/// Core case: `Describe` sends `RowDescription`; a fetch-all `Execute` streams the `DataRow`s and
/// its `CommandComplete` — and must NOT repeat `RowDescription` (that is `Describe`'s job).
#[tokio::test]
async fn stream_portal_fetch_all_does_not_repeat_row_description() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(&mut conn, "SELECT id FROM t ORDER BY id", vec![]).await;

    conn.write_frame(
        &FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "p".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::RowDescription {
            columns: vec!["id".to_owned()]
        }
    );

    execute_sync(&mut conn, 0).await;
    for expect in [b"1".to_vec(), b"2".to_vec(), b"3".to_vec()] {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(expect)]
            }
        );
    }
    // No second `RowDescription` — the very next frame is `CommandComplete`, then `ReadyForQuery`.
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A fetch-all `Execute` with no preceding `Describe` still streams rows (and, per protocol,
/// still emits no `RowDescription` — the client chose not to ask for it).
#[tokio::test]
async fn stream_portal_fetch_all_without_describe_still_streams() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(&mut conn, "SELECT id FROM t ORDER BY id", vec![]).await;
    execute_sync(&mut conn, 0).await;
    for expect in [b"1".to_vec(), b"2".to_vec(), b"3".to_vec()] {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(expect)]
            }
        );
    }
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Empty result set: the stream emits `CommandComplete` (with a 0 count) and no `DataRow`s.
#[tokio::test]
async fn stream_portal_empty_result_sends_only_command_complete() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(&mut conn, "SELECT id FROM t WHERE id > 100", vec![]).await;
    execute_sync(&mut conn, 0).await;
    assert_eq!(next(&mut conn).await, cc("SELECT 0"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Re-`Execute` a fully-streamed portal before `Sync`: it is a no-op (no rows, no second
/// `CommandComplete`) — matching the buffered path's drained-portal behaviour.
#[tokio::test]
async fn stream_portal_re_execute_after_drain_is_a_noop() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(&mut conn, "SELECT id FROM t ORDER BY id", vec![]).await;

    // First `Execute` streams all three rows + `CommandComplete`.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    for expect in [b"1".to_vec(), b"2".to_vec(), b"3".to_vec()] {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(expect)]
            }
        );
    }
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));

    // Second `Execute` (no re-`Bind`) → drained portal → nothing emitted. `Sync` then replies.
    execute_sync(&mut conn, 0).await;
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A parameterised fetch-all stream applies the bound `$1` and returns the filtered rows.
#[tokio::test]
async fn stream_portal_with_parameter_filters_rows() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(
        &mut conn,
        "SELECT id FROM t WHERE id > $1 ORDER BY id",
        vec![Some(b"1".to_vec())],
    )
    .await;
    execute_sync(&mut conn, 0).await;
    for expect in [b"2".to_vec(), b"3".to_vec()] {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(expect)]
            }
        );
    }
    assert_eq!(next(&mut conn).await, cc("SELECT 2"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A binary result format cannot use the text stream: the portal falls back to the buffered
/// materialize-then-drain path and returns big-endian `int4` `DataRow`s.
#[tokio::test]
async fn stream_portal_binary_format_uses_buffered_path() {
    let (mut conn, handle) = seeded_three_rows().await;
    conn.write_frame(
        &FrontendMessage::Parse {
            name: "s".to_owned(),
            sql: "SELECT id FROM t ORDER BY id".to_owned(),
            param_types: vec![],
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);
    conn.write_frame(
        &FrontendMessage::Bind {
            portal: "p".to_owned(),
            statement: "s".to_owned(),
            params: vec![],
            result_formats: vec![1], // binary
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);

    execute_sync(&mut conn, 0).await;
    for n in 1_i64..=3 {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(n.to_be_bytes().to_vec())]
            }
        );
    }
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A fetch-all stream whose statement fails to plan reports `ErrorResponse`, skips frames until
/// `Sync`, and `Sync` then restores `ReadyForQuery` — matching the buffered error path.
#[tokio::test]
async fn stream_portal_error_recovers_at_sync() {
    let (mut conn, handle) = seeded_three_rows().await;
    parse_bind_text(&mut conn, "SELECT missing_col FROM t", vec![]).await;

    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { .. }
    ));

    // A stray `Execute` before `Sync` is skipped (failed state); `Sync` clears it.
    conn.write_frame(
        &FrontendMessage::Execute {
            portal: "p".to_owned(),
            max_rows: 0,
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    conn.write_frame(&FrontendMessage::Sync.encode().unwrap())
        .await
        .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// Inside an explicit transaction the fetch-all stream still works and `ReadyForQuery` reports the
/// in-transaction status, so the streamed path threads transaction state correctly.
#[tokio::test]
async fn stream_portal_fetch_all_inside_transaction() {
    let (mut conn, handle) = seeded_three_rows().await;
    query(&mut conn, "BEGIN").await;
    assert_eq!(next(&mut conn).await, cc("BEGIN"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    parse_bind_text(&mut conn, "SELECT id FROM t ORDER BY id", vec![]).await;
    execute_sync(&mut conn, 0).await;
    for expect in [b"1".to_vec(), b"2".to_vec(), b"3".to_vec()] {
        assert_eq!(
            next(&mut conn).await,
            BackendMessage::DataRow {
                values: vec![Some(expect)]
            }
        );
    }
    assert_eq!(next(&mut conn).await, cc("SELECT 3"));
    // Still inside the transaction after the streamed `Execute`.
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::InTransaction)
    );

    query(&mut conn, "COMMIT").await;
    assert_eq!(next(&mut conn).await, cc("COMMIT"));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );
    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);
    handle.await.unwrap().unwrap();
}
