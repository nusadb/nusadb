//! Concurrent auto-commit writers to one row, over the wire.
//!
//! This is the shape that made a hot row unusable: several connections updating the same row with no
//! explicit transaction, each losing statement coming back as SQLSTATE `40001` for the application to
//! handle. A single auto-commit statement commits nothing and shows the application no intermediate
//! result, so the server retries it itself.
//!
//! The test lives at the wire level on purpose. The retry exists at several auto-commit sites (see
//! `nusadb_sql::retry`), and only this one is the path a real client takes — so it is the only one
//! that proves the behaviour a client actually gets.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, TxnStatus, handle_client};
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

async fn consume_until_ready<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    loop {
        if let BackendMessage::ReadyForQuery(status) = next(conn).await {
            assert_eq!(status, TxnStatus::Idle);
            return;
        }
    }
}

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

/// Open a connection to `engine` and complete its handshake.
async fn connect(engine: &Arc<dyn StorageEngine>) -> Connection<tokio::io::DuplexStream> {
    let (client, server) = tokio::io::duplex(64 * 1024);
    drop(tokio::spawn(handle_client(server, Arc::clone(engine))));
    let mut conn = Connection::new(client);
    start_session(&mut conn).await;
    conn
}

/// Run `sql` and return its `CommandComplete` tag, or the error's SQLSTATE.
async fn run<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    sql: &str,
) -> Result<String, String> {
    query(conn, sql).await;
    let mut outcome = None;
    loop {
        match next(conn).await {
            BackendMessage::CommandComplete { tag } => outcome = Some(Ok(tag)),
            BackendMessage::Error { code, .. } => outcome = Some(Err(code)),
            BackendMessage::ReadyForQuery(_) => {
                return outcome.unwrap_or_else(|| panic!("no outcome for: {sql}"));
            },
            _ => {},
        }
    }
}

/// Several connections incrementing the same row with no explicit transaction. Every statement must
/// succeed, and the total must be exact — a retry that dropped an attempt or applied one twice shows
/// up here as a count that is not the number of increments issued.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_to_one_row_never_surface_a_conflict() {
    const WRITERS: usize = 8;
    const WRITES_EACH: usize = 25;

    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut setup = connect(&engine).await;
    assert_eq!(
        run(&mut setup, "CREATE TABLE ctr (id INT PRIMARY KEY, v INT)").await,
        Ok("CREATE TABLE".to_owned())
    );
    assert_eq!(
        run(&mut setup, "INSERT INTO ctr VALUES (1, 0)").await,
        Ok("INSERT 1".to_owned())
    );

    let writers: Vec<_> = (0..WRITERS)
        .map(|_| {
            let engine = Arc::clone(&engine);
            tokio::spawn(async move {
                let mut conn = connect(&engine).await;
                let mut errors = Vec::new();
                for _ in 0..WRITES_EACH {
                    if let Err(code) = run(&mut conn, "UPDATE ctr SET v = v + 1 WHERE id = 1").await
                    {
                        errors.push(code);
                    }
                }
                errors
            })
        })
        .collect();

    let mut errors = Vec::new();
    for writer in writers {
        errors.extend(writer.await.unwrap());
    }
    assert!(
        errors.is_empty(),
        "every auto-commit write should have been retried to completion, got: {errors:?}"
    );

    // The count is the correctness assertion: exactly one application per statement issued.
    query(&mut setup, "SELECT v FROM ctr WHERE id = 1").await;
    let mut value = None;
    loop {
        match next(&mut setup).await {
            BackendMessage::DataRow { values } => {
                value = values[0].clone();
            },
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    assert_eq!(
        value
            .as_deref()
            .map(|v| String::from_utf8_lossy(v).to_string()),
        Some((WRITERS * WRITES_EACH).to_string()),
        "each increment must be applied exactly once"
    );
}

/// The retry belongs to auto-commit alone. Inside `BEGIN…COMMIT` a conflict is reported, because an
/// earlier statement in the same transaction may already have returned a result the application acted
/// on — only the application can decide the whole transaction is safe to replay.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_conflict_inside_an_explicit_transaction_still_reaches_the_client() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut setup = connect(&engine).await;
    assert_eq!(
        run(
            &mut setup,
            "CREATE TABLE acct (id INT PRIMARY KEY, bal INT)"
        )
        .await,
        Ok("CREATE TABLE".to_owned())
    );
    assert_eq!(
        run(&mut setup, "INSERT INTO acct VALUES (1, 100)").await,
        Ok("INSERT 1".to_owned())
    );

    // A transaction that writes the row and stays open: with no-wait locks, anything else touching
    // that row loses immediately and keeps losing, so no timing decides this test.
    let mut blocker = connect(&engine).await;
    assert_eq!(run(&mut blocker, "BEGIN").await, Ok("BEGIN".to_owned()));
    query(&mut blocker, "UPDATE acct SET bal = 1 WHERE id = 1").await;
    loop {
        if matches!(next(&mut blocker).await, BackendMessage::ReadyForQuery(_)) {
            break;
        }
    }

    let mut victim = connect(&engine).await;
    assert_eq!(run(&mut victim, "BEGIN").await, Ok("BEGIN".to_owned()));
    query(&mut victim, "UPDATE acct SET bal = 2 WHERE id = 1").await;
    let mut code = None;
    loop {
        match next(&mut victim).await {
            BackendMessage::Error { code: c, .. } => code = Some(c),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    assert_eq!(
        code.as_deref(),
        Some("40001"),
        "a conflict inside an explicit transaction belongs to the application"
    );
}

/// `SET max_autocommit_retries = 0` switches the retry off, and the first conflict is reported —
/// the escape hatch for a deployment that would rather see conflicts than wait through them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_retry_can_be_switched_off_per_session() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut setup = connect(&engine).await;
    assert_eq!(
        run(&mut setup, "CREATE TABLE k (id INT PRIMARY KEY, v INT)").await,
        Ok("CREATE TABLE".to_owned())
    );
    assert_eq!(
        run(&mut setup, "INSERT INTO k VALUES (1, 0)").await,
        Ok("INSERT 1".to_owned())
    );

    let mut blocker = connect(&engine).await;
    assert_eq!(run(&mut blocker, "BEGIN").await, Ok("BEGIN".to_owned()));
    assert_eq!(
        run(&mut blocker, "UPDATE k SET v = 1 WHERE id = 1").await,
        Ok("UPDATE 1".to_owned())
    );

    let mut victim = connect(&engine).await;
    assert_eq!(
        run(&mut victim, "SET max_autocommit_retries = 0").await,
        Ok("SET".to_owned())
    );
    assert_eq!(
        run(&mut victim, "UPDATE k SET v = 2 WHERE id = 1").await,
        Err("40001".to_owned()),
        "with the retry off the first conflict is the answer"
    );
}
