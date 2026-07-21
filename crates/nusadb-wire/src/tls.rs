//! TLS for the Nusa Wire Protocol — rustls over TCP.
//!
//! The engine speaks its own framing (`[type][len][payload]`) with no in-band
//! SSL-request negotiation, so it uses **implicit TLS**: when the server is started with
//! a [`rustls::ServerConfig`] the accept loop wraps every accepted
//! connection in a rustls server session before any Nusa frame is read, and the
//! per-connection handler runs unchanged over the encrypted stream (it is
//! generic over [`tokio::io::AsyncRead`] + `AsyncWrite`). There is no
//! plaintext fallback on a TLS listener — a client that cannot complete the
//! handshake is dropped, so a TLS endpoint never leaks an unencrypted exchange.
//!
//! [`server_config`] builds the rustls config from a PEM certificate chain and
//! private key; key material is parsed with [`rustls::pki_types`] (no extra
//! dependency). The crypto backend is whatever the workspace's rustls enables
//! (aws-lc-rs today). [`server_config_mtls`] additionally **requires** every
//! client to present a certificate chaining to a configured CA (mutual TLS,
//! Follow-up): the transport is mutually authenticated before any frame is
//! read. Mapping a verified client certificate's subject (CN/SAN) onto a SQL
//! username is a separate refinement — SCRAM-SHA-256 ([`crate::auth`]) remains
//! the user-identity mechanism; mTLS adds a transport-level peer requirement.
//! The SSL-request negotiation handshake is out of scope here.

use std::sync::Arc;

use rustls::RootCertStore;
use rustls::ServerConfig;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;

use crate::error::WireError;

/// Build a rustls [`ServerConfig`] from a PEM certificate chain and private key.
///
/// The certificate is served to clients; the private key signs the handshake.
/// No client certificate is required (server-only TLS).
///
/// # Errors
/// [`WireError::Tls`] if the PEM cannot be parsed, holds no certificate, or the
/// certificate/key pair is rejected by rustls.
pub fn server_config(cert_pem: &[u8], key_pem: &[u8]) -> Result<Arc<ServerConfig>, WireError> {
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(cert_pem)
        .collect::<Result<_, _>>()
        .map_err(|e| WireError::Tls(format!("certificate PEM: {e}")))?;
    if certs.is_empty() {
        return Err(WireError::Tls(
            "certificate PEM contained no certificates".to_owned(),
        ));
    }
    let key = PrivateKeyDer::from_pem_slice(key_pem)
        .map_err(|e| WireError::Tls(format!("private key PEM: {e}")))?;
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| WireError::Tls(format!("certificate/key rejected: {e}")))?;
    Ok(Arc::new(config))
}

/// Build a [`ServerConfig`] by reading the certificate and key from PEM files.
///
/// # Errors
/// [`WireError::Tls`] if a file cannot be read or [`server_config`] rejects its
/// contents.
pub fn server_config_from_files(
    cert_path: &std::path::Path,
    key_path: &std::path::Path,
) -> Result<Arc<ServerConfig>, WireError> {
    let cert = std::fs::read(cert_path)
        .map_err(|e| WireError::Tls(format!("reading {}: {e}", cert_path.display())))?;
    let key = std::fs::read(key_path)
        .map_err(|e| WireError::Tls(format!("reading {}: {e}", key_path.display())))?;
    server_config(&cert, &key)
}

/// Build a mutual-TLS [`ServerConfig`] (follow-up).
///
/// The server presents `cert_pem`/`key_pem` **and** requires every client to present a certificate
/// chaining to a trust anchor in `client_ca_pem`. A client with no certificate, or one not signed by
/// the CA, fails the handshake and is dropped before any Nusa frame is exchanged — so an mTLS
/// endpoint never serves an unauthenticated peer.
///
/// This authenticates the *transport peer*. Resolving the verified certificate's subject to a SQL
/// username is a separate step; user authentication still flows through SCRAM ([`crate::auth`]).
///
/// # Errors
/// [`WireError::Tls`] if either PEM cannot be parsed, the server cert/key pair is rejected, the CA
/// PEM holds no certificate, or the client verifier cannot be built.
pub fn server_config_mtls(
    cert_pem: &[u8],
    key_pem: &[u8],
    client_ca_pem: &[u8],
) -> Result<Arc<ServerConfig>, WireError> {
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(cert_pem)
        .collect::<Result<_, _>>()
        .map_err(|e| WireError::Tls(format!("certificate PEM: {e}")))?;
    if certs.is_empty() {
        return Err(WireError::Tls(
            "certificate PEM contained no certificates".to_owned(),
        ));
    }
    let key = PrivateKeyDer::from_pem_slice(key_pem)
        .map_err(|e| WireError::Tls(format!("private key PEM: {e}")))?;

    let mut roots = RootCertStore::empty();
    for ca in CertificateDer::pem_slice_iter(client_ca_pem) {
        let ca = ca.map_err(|e| WireError::Tls(format!("client CA PEM: {e}")))?;
        roots
            .add(ca)
            .map_err(|e| WireError::Tls(format!("client CA rejected: {e}")))?;
    }
    if roots.is_empty() {
        return Err(WireError::Tls(
            "client CA PEM contained no certificates".to_owned(),
        ));
    }
    let verifier = WebPkiClientVerifier::builder(Arc::new(roots))
        .build()
        .map_err(|e| WireError::Tls(format!("client verifier: {e}")))?;

    let config = ServerConfig::builder()
        .with_client_cert_verifier(verifier)
        .with_single_cert(certs, key)
        .map_err(|e| WireError::Tls(format!("certificate/key rejected: {e}")))?;
    Ok(Arc::new(config))
}

/// Build a mutual-TLS [`ServerConfig`] by reading the server certificate, key, and client CA from
/// PEM files (see [`server_config_mtls`]).
///
/// # Errors
/// [`WireError::Tls`] if a file cannot be read or [`server_config_mtls`] rejects its contents.
pub fn server_config_mtls_from_files(
    cert_path: &std::path::Path,
    key_path: &std::path::Path,
    client_ca_path: &std::path::Path,
) -> Result<Arc<ServerConfig>, WireError> {
    let cert = std::fs::read(cert_path)
        .map_err(|e| WireError::Tls(format!("reading {}: {e}", cert_path.display())))?;
    let key = std::fs::read(key_path)
        .map_err(|e| WireError::Tls(format!("reading {}: {e}", key_path.display())))?;
    let ca = std::fs::read(client_ca_path)
        .map_err(|e| WireError::Tls(format!("reading {}: {e}", client_ca_path.display())))?;
    server_config_mtls(&cert, &key, &ca)
}

#[cfg(test)]
mod tests {
    use super::{server_config, server_config_mtls};
    use crate::error::WireError;

    // A self-signed P-256 certificate (CN=localhost, SAN DNS:localhost) and its
    // PKCS#8 key, valid for ~100 years — shared with the integration TLS test.
    const CERT_PEM: &[u8] = include_bytes!("../tests/data/localhost-cert.pem");
    const KEY_PEM: &[u8] = include_bytes!("../tests/data/localhost-key.pem");

    // A CA + CA-signed server cert/key for mutual TLS. The matching client
    // cert/key live alongside them and drive the integration handshake test.
    const CA_PEM: &[u8] = include_bytes!("../tests/data/ca-cert.pem");
    const MTLS_SERVER_CERT: &[u8] = include_bytes!("../tests/data/mtls-server-cert.pem");
    const MTLS_SERVER_KEY: &[u8] = include_bytes!("../tests/data/mtls-server-key.pem");

    #[test]
    fn builds_a_config_from_valid_pem() {
        assert!(server_config(CERT_PEM, KEY_PEM).is_ok());
    }

    #[test]
    fn empty_certificate_pem_is_rejected() {
        let err = server_config(b"", KEY_PEM).unwrap_err();
        assert!(matches!(err, WireError::Tls(_)));
    }

    #[test]
    fn garbage_key_is_rejected() {
        assert!(
            server_config(
                CERT_PEM,
                b"-----BEGIN PRIVATE KEY-----\nbad\n-----END PRIVATE KEY-----\n"
            )
            .is_err()
        );
    }

    #[test]
    fn mtls_config_builds_with_a_client_ca() {
        assert!(server_config_mtls(MTLS_SERVER_CERT, MTLS_SERVER_KEY, CA_PEM).is_ok());
    }

    #[test]
    fn mtls_config_rejects_an_empty_client_ca() {
        let err = server_config_mtls(MTLS_SERVER_CERT, MTLS_SERVER_KEY, b"").unwrap_err();
        assert!(matches!(err, WireError::Tls(_)));
    }
}
