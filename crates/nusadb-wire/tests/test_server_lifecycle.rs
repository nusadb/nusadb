//! Server lifecycle: idle-connection timeout and graceful shutdown / drain,
//! driven end-to-end over real TCP against a `BtreeEngine`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test harness asserts via unwrap/expect"
)]

use std::future::pending;
use std::sync::Arc;
use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, FrontendMessage, Metrics, PROTOCOL_VERSION, ServerConfig,
    serve_with_shutdown,
};
use tokio::net::{TcpListener, TcpStream};

/// Send the Startup message (without waiting for a reply).
async fn send_startup(client: &mut Connection<TcpStream>) {
    client
        .write_frame(
            &FrontendMessage::Startup {
                major: PROTOCOL_VERSION.0,
                minor: PROTOCOL_VERSION.1,
                user: "u".to_owned(),
                database: "nusadb".to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
}

/// Perform the Startup handshake and read through to the first `ReadyForQuery`.
async fn startup(client: &mut Connection<TcpStream>) {
    send_startup(client).await;
    read_until_ready(client).await;
}

/// Send a simple query and read through to its `ReadyForQuery`.
async fn query(client: &mut Connection<TcpStream>, sql: &str) {
    client
        .write_frame(
            &FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    read_until_ready(client).await;
}

/// Drain backend messages until a `ReadyForQuery` is seen.
async fn read_until_ready(client: &mut Connection<TcpStream>) {
    loop {
        let frame = client
            .read_frame()
            .await
            .unwrap()
            .expect("server closed before ReadyForQuery");
        if matches!(
            BackendMessage::decode(&frame).unwrap(),
            BackendMessage::ReadyForQuery(_)
        ) {
            return;
        }
    }
}

#[tokio::test]
async fn idle_connection_is_closed_after_the_idle_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        idle_timeout: Some(Duration::from_millis(100)),
        drain_timeout: None,
        ..ServerConfig::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        config,
        pending::<()>(),
    ));

    let mut client = Connection::new(TcpStream::connect(addr).await.unwrap());
    startup(&mut client).await;

    // Stay silent — the server must close the connection within a couple of idle windows. A clean
    // close surfaces as `Ok(None)` (EOF) on the next read.
    let closed = tokio::time::timeout(Duration::from_secs(2), client.read_frame()).await;
    assert!(
        matches!(closed, Ok(Ok(None))),
        "server should close an idle connection, got {closed:?}"
    );
    server.abort();
}

#[tokio::test]
async fn graceful_shutdown_closes_connections_and_returns() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        ServerConfig::default(),
        async move {
            let _ = shutdown_rx.await;
        },
    ));

    // An established session that successfully runs a query (proving it is live before shutdown).
    let mut client = Connection::new(TcpStream::connect(addr).await.unwrap());
    startup(&mut client).await;
    client
        .write_frame(
            &FrontendMessage::Query {
                sql: "SELECT 1".to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    read_until_ready(&mut client).await;

    // Trigger graceful shutdown: serve stops accepting, signals the connection, and drains it.
    shutdown_tx.send(()).unwrap();

    // The now-idle connection is closed by the drain (clean EOF).
    let closed = tokio::time::timeout(Duration::from_secs(2), client.read_frame()).await;
    assert!(
        matches!(closed, Ok(Ok(None))),
        "drain should close the idle connection, got {closed:?}"
    );

    // serve_with_shutdown returns Ok once every connection has drained.
    let joined = tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("serve should return promptly after drain")
        .expect("serve task panicked");
    assert!(joined.is_ok(), "serve returned an error: {joined:?}");
}

#[tokio::test]
async fn connection_limit_queues_until_a_slot_frees() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        max_connections: Some(1),
        ..ServerConfig::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        config,
        pending::<()>(),
    ));

    // The first client takes the only slot.
    let mut first = Connection::new(TcpStream::connect(addr).await.unwrap());
    startup(&mut first).await;

    // The second connects but is queued behind the limit — its Startup gets no reply yet.
    let mut second = Connection::new(TcpStream::connect(addr).await.unwrap());
    send_startup(&mut second).await;
    let queued = tokio::time::timeout(Duration::from_millis(300), second.read_frame()).await;
    assert!(
        queued.is_err(),
        "second connection should be queued behind max_connections=1, got {queued:?}"
    );

    // Free the slot by dropping the first connection; the second is then served.
    drop(first);
    tokio::time::timeout(Duration::from_secs(2), read_until_ready(&mut second))
        .await
        .expect("second connection should be served once the slot frees");

    server.abort();
}

#[tokio::test]
async fn metrics_count_connections_and_queries() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let metrics = Arc::new(Metrics::new());
    let config = ServerConfig {
        metrics: Some(Arc::clone(&metrics)),
        ..ServerConfig::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        config,
        pending::<()>(),
    ));

    let mut client = Connection::new(TcpStream::connect(addr).await.unwrap());
    startup(&mut client).await;
    query(&mut client, "SELECT 1").await; // succeeds
    query(&mut client, "SELECT * FROM ghost").await; // unknown table → error

    // Counters are bumped before each ReadyForQuery, so they are settled once both queries return.
    assert_eq!(metrics.connections_total(), 1);
    assert_eq!(metrics.queries_total(), 2);
    assert_eq!(metrics.query_errors_total(), 1);
    assert!(
        metrics
            .render_prometheus()
            .contains("nusadb_queries_total 2")
    );

    server.abort();
}

#[tokio::test]
async fn connection_limit_fast_rejects_with_53300_when_configured() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        max_connections: Some(1),
        reject_excess_connections: true,
        ..ServerConfig::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        config,
        pending::<()>(),
    ));

    // The first client takes the only slot.
    let mut first = Connection::new(TcpStream::connect(addr).await.unwrap());
    startup(&mut first).await;

    // The second is refused immediately with 53300 (P-CONNCAP fast-reject) instead of queueing.
    let mut second = Connection::new(TcpStream::connect(addr).await.unwrap());
    let refused = tokio::time::timeout(Duration::from_secs(2), second.read_frame())
        .await
        .expect("the reject must arrive promptly, not queue")
        .unwrap()
        .expect("an error frame, not a bare close");
    match BackendMessage::decode(&refused).unwrap() {
        BackendMessage::Error { code, message } => {
            assert_eq!(code, "53300");
            assert!(
                message.contains("too many clients"),
                "message should say so: {message}"
            );
        },
        other => panic!("expected a 53300 error, got {other:?}"),
    }

    // Once the slot frees, a fresh connection is served normally again.
    drop(first);
    // The freed permit returns when the first connection's task finishes; retry briefly.
    let mut served = false;
    for _ in 0..20 {
        let mut third = Connection::new(TcpStream::connect(addr).await.unwrap());
        send_startup(&mut third).await;
        third.flush_now().await.unwrap();
        if let Ok(Ok(Some(frame))) =
            tokio::time::timeout(Duration::from_millis(200), third.read_frame()).await
            && !matches!(
                BackendMessage::decode(&frame).unwrap(),
                BackendMessage::Error { .. }
            )
        {
            served = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(served, "a connection after the slot freed must be served");

    server.abort();
}
