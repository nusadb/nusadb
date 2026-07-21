//! The connection transport: plaintext TCP or an implicit TLS 1.3 session over TCP
//! (`docs/wire-protocol.md` §5.1), plus the client-side TLS configuration helper.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;

use crate::error::{Error, Result};

/// A client connection's byte stream: plaintext, or a rustls client session (both over TCP).
///
/// The two concrete inner types are wrapped in one `enum` (rather than a boxed `dyn` trait object)
/// so the [`AsyncRead`]/[`AsyncWrite`] impls stay allocation-free and statically dispatched. Both
/// inner streams are `Unpin`, so the `enum` is `Unpin` and the impls can project through
/// `get_mut()`.
#[derive(Debug)]
pub(crate) enum Transport {
    /// Plaintext TCP (a dev/local server with no TLS).
    Plain(TcpStream),
    /// An implicit TLS 1.3 client session over TCP.
    Tls(Box<TlsStream<TcpStream>>),
}

impl AsyncRead for Transport {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_read(cx, buf),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for Transport {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_write(cx, buf),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_flush(cx),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(s) => Pin::new(s).poll_shutdown(cx),
            Self::Tls(s) => Pin::new(s.as_mut()).poll_shutdown(cx),
        }
    }
}

/// Build a [`rustls::ClientConfig`] trusting only the certificate(s) in `ca_pem` as roots.
///
/// There is no system-trust-store fallback: the caller decides exactly what to trust, matching how
/// `nusadb-server --tls-cert` ships a self-signed or private-CA certificate.
///
/// # Errors
/// [`Error::Tls`] if the PEM holds no certificate, one fails to parse, or rustls rejects it as a
/// trust anchor.
pub fn tls_client_config(ca_pem: &[u8]) -> Result<Arc<rustls::ClientConfig>> {
    use rustls::pki_types::CertificateDer;
    use rustls::pki_types::pem::PemObject as _;

    let mut roots = rustls::RootCertStore::empty();
    let mut added = 0usize;
    for cert in CertificateDer::pem_slice_iter(ca_pem) {
        let cert = cert.map_err(|e| Error::Tls(e.to_string()))?;
        roots.add(cert).map_err(|e| Error::Tls(e.to_string()))?;
        added += 1;
    }
    if added == 0 {
        return Err(Error::Tls("CA PEM contained no certificates".to_owned()));
    }
    Ok(Arc::new(
        rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth(),
    ))
}
