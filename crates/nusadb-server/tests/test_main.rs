//! End-to-end integration test for `nusadb-server`: boot the wire server on a real (ephemeral)
//! TCP port and drive a full client session against it. Kept out of `main.rs` so the binary's
//! production code stays free of test-only socket/runtime machinery.

// Integration tests are their own crate, so the `allow-*-in-tests` clippy carve-outs (which only
// cover `#[cfg(test)]` modules) don't apply here.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, TxnStatus, serve};
use tokio::net::{TcpListener, TcpStream};

async fn next(conn: &mut Connection<TcpStream>) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Consume the post-auth handshake chatter — `BackendKeyData` and the startup `ParameterStatus`
/// reports — up to and including the initial `ReadyForQuery(Idle)`.
async fn consume_until_ready(conn: &mut Connection<TcpStream>) {
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

async fn query(conn: &mut Connection<TcpStream>, sql: &str) {
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

/// Boot the server on an ephemeral port and drive a full session over real TCP.
#[tokio::test]
async fn serves_clients_over_tcp() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let server = tokio::spawn(serve(listener, engine));

    let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());

    // Handshake.
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

    // CREATE + INSERT.
    query(&mut conn, "CREATE TABLE t (id INT NOT NULL)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    query(&mut conn, "INSERT INTO t VALUES (7)").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { tag } if tag == "INSERT 1"
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    // SELECT returns the row over the wire.
    query(&mut conn, "SELECT id FROM t").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::RowDescription { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"7".to_vec())]
        }
    );
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
    server.abort();
}
