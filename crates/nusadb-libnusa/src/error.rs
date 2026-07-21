//! Client error type.

use std::fmt;

/// An error from a libnusa operation.
#[derive(Debug)]
pub enum Error {
    /// A transport (socket / TLS) I/O failure.
    Io(std::io::Error),
    /// A protocol violation: an unexpected message, a malformed frame, or a server reply that
    /// breaks the state machine in `docs/wire-protocol.md`.
    Protocol(String),
    /// The server rejected a statement: a 5-character SQLSTATE plus its message (wire `Error`, §14).
    Server {
        /// 5-character SQLSTATE code.
        code: String,
        /// Human-readable server message.
        message: String,
    },
    /// Authentication failed (wrong password, an unverifiable server signature, or no password when
    /// the server demanded one).
    Auth(String),
    /// A TLS-configuration failure (e.g. an unparsable CA bundle).
    Tls(String),
    /// A field could not be decoded into the requested Rust type.
    Decode(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Protocol(m) => write!(f, "protocol error: {m}"),
            Self::Server { code, message } => write!(f, "server error {code}: {message}"),
            Self::Auth(m) => write!(f, "authentication failed: {m}"),
            Self::Tls(m) => write!(f, "tls error: {m}"),
            Self::Decode(m) => write!(f, "decode error: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<nusadb_wire::WireError> for Error {
    fn from(e: nusadb_wire::WireError) -> Self {
        Self::Protocol(e.to_string())
    }
}

/// A libnusa `Result`.
pub type Result<T> = std::result::Result<T, Error>;
