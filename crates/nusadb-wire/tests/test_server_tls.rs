//! Integration test for TLS: a real rustls client completes the handshake against the
//! TLS-configured server and runs a full session over the encrypted stream.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "integration test harness asserts via unwrap/expect/panic; one linear session script"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, FrontendMessage, ServerConfig, TxnStatus, serve_with_shutdown, tls,
};
use rustls::pki_types::pem::PemObject as _;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio::net::{TcpListener, TcpStream};

const CERT_PEM: &[u8] = include_bytes!("data/localhost-cert.pem");
const KEY_PEM: &[u8] = include_bytes!("data/localhost-key.pem");

// Mutual-TLS material: a CA, a CA-signed server cert/key, and a CA-signed client cert/key.
const CA_PEM: &[u8] = include_bytes!("data/ca-cert.pem");
const MTLS_SERVER_CERT: &[u8] = include_bytes!("data/mtls-server-cert.pem");
const MTLS_SERVER_KEY: &[u8] = include_bytes!("data/mtls-server-key.pem");
const MTLS_CLIENT_CERT: &[u8] = include_bytes!("data/mtls-client-cert.pem");
const MTLS_CLIENT_KEY: &[u8] = include_bytes!("data/mtls-client-key.pem");

/// A client rustls config that trusts the test's self-signed certificate.
fn client_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    for cert in CertificateDer::pem_slice_iter(CERT_PEM) {
        roots.add(cert.unwrap()).unwrap();
    }
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

/// A client config that trusts the mTLS CA. `with_cert` presents the CA-signed client certificate
/// (the path the server accepts); otherwise it offers no client certificate (rejected by mTLS).
fn mtls_client_config(with_cert: bool) -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    for ca in CertificateDer::pem_slice_iter(CA_PEM) {
        roots.add(ca.unwrap()).unwrap();
    }
    let builder = rustls::ClientConfig::builder().with_root_certificates(roots);
    if with_cert {
        let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(MTLS_CLIENT_CERT)
            .map(Result::unwrap)
            .collect();
        let key = PrivateKeyDer::from_pem_slice(MTLS_CLIENT_KEY).unwrap();
        builder.with_client_auth_cert(certs, key).unwrap()
    } else {
        builder.with_no_client_auth()
    }
}

async fn next<S>(conn: &mut Connection<S>) -> BackendMessage
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Consume the post-auth handshake chatter — `BackendKeyData` and the startup `ParameterStatus`
/// reports — up to and including the initial `ReadyForQuery(Idle)`.
async fn consume_until_ready<S>(conn: &mut Connection<S>)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
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

#[tokio::test]
async fn full_session_over_tls() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let tls_config = tls::server_config(CERT_PEM, KEY_PEM).expect("server tls config");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let config = ServerConfig {
        tls: Some(tls_config),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    // --- Client: TCP connect, then TLS handshake to "localhost" ---
    let tcp = TcpStream::connect(addr).await.unwrap();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config()));
    let server_name = ServerName::try_from("localhost").unwrap();
    let tls_stream = connector
        .connect(server_name, tcp)
        .await
        .expect("tls handshake");

    let mut conn = Connection::new(tls_stream);

    // Startup handshake over the encrypted stream.
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

    // A round-trip query proves the encrypted channel carries real SQL.
    conn.write_frame(
        &FrontendMessage::Query {
            sql: "CREATE TABLE t (id INT NOT NULL)".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CommandComplete { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(
        &FrontendMessage::Query {
            sql: "INSERT INTO t VALUES (42)".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CommandComplete {
            tag: "INSERT 1".to_owned()
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(
        &FrontendMessage::Query {
            sql: "SELECT id FROM t".to_owned(),
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
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"42".to_vec())]
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::CommandComplete {
            tag: "SELECT 1".to_owned()
        }
    );
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::ReadyForQuery(TxnStatus::Idle)
    );

    conn.write_frame(&FrontendMessage::Terminate.encode().unwrap())
        .await
        .unwrap();
    drop(conn);

    let _ = stop_tx.send(());
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn plaintext_client_is_rejected_on_a_tls_listener() {
    // A client that skips the TLS handshake and sends a raw Nusa frame must not get a usable
    // session — the server is doing a TLS handshake and will fail to parse the plaintext.
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let tls_config = tls::server_config(CERT_PEM, KEY_PEM).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = ServerConfig {
        tls: Some(tls_config),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    let tcp = TcpStream::connect(addr).await.unwrap();
    let mut conn = Connection::new(tcp);
    // Send a plaintext Startup frame; the TLS server reads it as a (bad) ClientHello, fails the
    // handshake, and drops the connection — so the client never receives an `AuthOk`.
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

    let got = tokio::time::timeout(std::time::Duration::from_secs(3), conn.read_frame()).await;
    let got_authok = matches!(
        &got,
        Ok(Ok(Some(frame))) if BackendMessage::decode(frame).is_ok_and(|m| m == BackendMessage::AuthOk)
    );
    assert!(
        !got_authok,
        "a plaintext client must not establish a session on a TLS listener (got {got:?})"
    );

    let _ = stop_tx.send(());
    let _ = server.await;
}

#[tokio::test]
async fn mutual_tls_session_with_a_client_certificate() {
    // mTLS: the server requires a CA-signed client certificate. A client that presents one
    // completes the handshake and runs a normal session over the mutually-authenticated stream.
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let tls_config = tls::server_config_mtls(MTLS_SERVER_CERT, MTLS_SERVER_KEY, CA_PEM)
        .expect("mtls server config");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = ServerConfig {
        tls: Some(tls_config),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    let tcp = TcpStream::connect(addr).await.unwrap();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(mtls_client_config(true)));
    let server_name = ServerName::try_from("localhost").unwrap();
    let tls_stream = connector
        .connect(server_name, tcp)
        .await
        .expect("mtls handshake with client cert");
    let mut conn = Connection::new(tls_stream);

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

    conn.write_frame(
        &FrontendMessage::Query {
            sql: "SELECT 1".to_owned(),
        }
        .encode()
        .unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::RowDescription { .. }
    ));
    assert_eq!(
        next(&mut conn).await,
        BackendMessage::DataRow {
            values: vec![Some(b"1".to_vec())]
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
    drop(conn);
    let _ = stop_tx.send(());
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn mtls_rejects_a_client_without_a_certificate() {
    // A client that trusts the CA but presents no certificate must not get a session on an mTLS
    // listener — the server demands a client certificate and aborts the handshake.
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let tls_config = tls::server_config_mtls(MTLS_SERVER_CERT, MTLS_SERVER_KEY, CA_PEM).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = ServerConfig {
        tls: Some(tls_config),
        ..Default::default()
    };
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(listener, engine, config, async move {
        let _ = stop_rx.await;
    }));

    let tcp = TcpStream::connect(addr).await.unwrap();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(mtls_client_config(false)));
    let server_name = ServerName::try_from("localhost").unwrap();
    // The server rejects the certless client during the handshake; even if the local handshake
    // future resolves, no usable session (AuthOk) is ever established.
    let established = match connector.connect(server_name, tcp).await {
        Ok(tls_stream) => {
            let mut conn = Connection::new(tls_stream);
            if conn
                .write_frame(
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
                .is_err()
            {
                false
            } else {
                let got =
                    tokio::time::timeout(std::time::Duration::from_secs(3), conn.read_frame())
                        .await;
                matches!(
                    &got,
                    Ok(Ok(Some(frame)))
                        if BackendMessage::decode(frame)
                            .is_ok_and(|m| m == BackendMessage::AuthOk)
                )
            }
        },
        Err(_) => false,
    };
    assert!(
        !established,
        "a client without a certificate must not establish an mTLS session"
    );

    let _ = stop_tx.send(());
    let _ = server.await;
}
