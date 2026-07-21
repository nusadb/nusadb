//! SCRAM-SHA-256 authentication (RFC 5802 / RFC 7677).
//!
//! The handshake is built on `ring` 0.17 primitives (PBKDF2 / HMAC / SHA-256 / CSPRNG) plus a
//! small in-crate base64 codec — deliberately *not* the `scram` crate, which pins an
//! unmaintained, vulnerable `ring` 0.16 (RUSTSEC-2025-0009 / -0010).
//!
//! [`scram`] is the message layer (parse `client-first` / `client-final`, build `server-first`,
//! derive credentials, constant-time proof verify). [`AuthStore`] is the config-based credential
//! store, and the server's SASL handshake wires them together.

mod base64;
pub mod scram;

pub use scram::{ChannelBinding, ClientFirst, ScramError, ServerFirst, generate_nonce};

use std::collections::HashMap;

use scram::{StoredCredentials, derive_credentials, generate_salt};

/// The default PBKDF2 iteration count for newly-derived credentials (RFC 7677 recommends ≥ 4096).
pub const DEFAULT_ITERATIONS: u32 = 4096;

/// A config-based SCRAM credential store: a map of username → derived credentials.
///
/// Built once at server start from plaintext passwords (each gets a fresh random salt); the
/// plaintext is never retained. A catalog-backed store (`__nusadb_users`) is a later evolution —
/// this struct is the seam both share.
#[derive(Debug, Clone)]
pub struct AuthStore {
    users: HashMap<String, StoredCredentials>,
}

impl AuthStore {
    /// Build a store by deriving SCRAM credentials for each `(username, password)` at
    /// [`DEFAULT_ITERATIONS`].
    ///
    /// # Errors
    /// [`ScramError`] if the CSPRNG fails or a derivation does.
    pub fn from_passwords<I, U, P>(users: I) -> Result<Self, ScramError>
    where
        I: IntoIterator<Item = (U, P)>,
        U: Into<String>,
        P: AsRef<str>,
    {
        let mut map = HashMap::new();
        for (user, password) in users {
            let salt = generate_salt()?;
            let creds = derive_credentials(password.as_ref(), salt, DEFAULT_ITERATIONS)?;
            map.insert(user.into(), creds);
        }
        Ok(Self { users: map })
    }

    /// The stored credentials for `user`, or `None` if no such user is configured.
    #[must_use]
    pub(crate) fn lookup(&self, user: &str) -> Option<&StoredCredentials> {
        self.users.get(user)
    }
}
