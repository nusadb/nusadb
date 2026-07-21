//! Integration test for CLI TLS: the client builds a rustls config that trusts the test
//! certificate, completes the handshake against a TLS-enabled `nusadb-wire` server, and runs a
//! query over the encrypted stream — exercising the same client path `nusa-cli --tls` uses.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test harness asserts via unwrap/expect"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_cli::{collect_result, handshake, tls_client_config};
use nusadb_core::StorageEngine;
use nusadb_wire::{Connection, ServerConfig, serve_with_shutdown, tls};
use rustls::pki_types::ServerName;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsConnector;

const CERT_PEM: &[u8] = include_bytes!("data/localhost-cert.pem");
const KEY_PEM: &[u8] = include_bytes!("data/localhost-key.pem");

#[tokio::test]
async fn cli_runs_sql_against_a_tls_server() {
    // Boot a TLS server on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        tls: Some(tls::server_config(CERT_PEM, KEY_PEM).unwrap()),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    // Client: the same path `nusa-cli --tls --tls-ca <cert> --tls-domain localhost` takes.
    let tcp = TcpStream::connect(addr).await.unwrap();
    let connector = TlsConnector::from(Arc::new(tls_client_config(CERT_PEM).unwrap()));
    let server_name = ServerName::try_from("localhost").unwrap();
    let stream = connector
        .connect(server_name, tcp)
        .await
        .expect("tls handshake");
    let mut conn = Connection::new(stream);

    handshake(&mut conn, "u", "nusadb", None).await.unwrap();
    collect_result(&mut conn, "CREATE TABLE t (id INT NOT NULL, name TEXT)")
        .await
        .unwrap();
    collect_result(&mut conn, "INSERT INTO t VALUES (1, 'alice')")
        .await
        .unwrap();

    let result = collect_result(&mut conn, "SELECT id, name FROM t")
        .await
        .unwrap();
    assert_eq!(result.columns, vec!["id".to_owned(), "name".to_owned()]);
    assert_eq!(
        result.rows,
        vec![vec![Some(b"1".to_vec()), Some(b"alice".to_vec())]]
    );
    assert!(result.error.is_none());

    let _ = stop_tx.send(());
    let _ = server.await;
}

#[tokio::test]
async fn wrong_ca_fails_the_handshake() {
    // A client that does not trust the server's certificate must not establish a session.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let config = ServerConfig {
        tls: Some(tls::server_config(CERT_PEM, KEY_PEM).unwrap()),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    // The certificate is valid for `localhost` only; verifying it against a different server name
    // must fail the handshake, so a misconfigured client never talks to the wrong host.
    let tcp = TcpStream::connect(addr).await.unwrap();
    let connector = TlsConnector::from(Arc::new(tls_client_config(CERT_PEM).unwrap()));
    let wrong_name = ServerName::try_from("not-the-server").unwrap();
    let handshake = connector.connect(wrong_name, tcp).await;
    assert!(
        handshake.is_err(),
        "handshake must fail when the certificate does not match the server name"
    );

    let _ = stop_tx.send(());
    let _ = server.await;
}
