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

use std::borrow::Cow;
use std::collections::HashMap;

use scram::{StoredCredentials, derive_credentials, derive_mock_credentials, generate_salt};

/// The default PBKDF2 iteration count for newly-derived credentials (RFC 7677 recommends ≥ 4096).
pub const DEFAULT_ITERATIONS: u32 = 4096;

/// A config-based SCRAM credential store: a map of username → derived credentials.
///
/// Built once at server start from plaintext passwords (each gets a fresh random salt); the
/// plaintext is never retained. A catalog-backed store (`__nusadb_users`) is a later evolution —
/// this struct is the seam both share.
#[derive(Clone)]
pub struct AuthStore {
    users: HashMap<String, StoredCredentials>,
    /// Keys the stand-in credentials handed out for names that are not in `users` — see
    /// [`credentials_for`](Self::credentials_for). Random per store, never leaves the process.
    ///
    /// Being per-store, it is re-rolled when the server restarts, so a stand-in salt changes across
    /// a restart while a real user's does not. Someone who can probe the same name before and after
    /// one could tell the two apart on that alone. Closing that needs a secret that outlives the
    /// process — deriving it from the store's own contents would survive a restart of an unchanged
    /// configuration, and is the obvious next step if this matters.
    mock_secret: Vec<u8>,
}

/// Prints no key material: the derived credentials and the stand-in secret are the whole point of
/// the store, and a stray `{:?}` in a log should not spill them.
impl std::fmt::Debug for AuthStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthStore")
            .field("users", &self.users.len())
            .finish_non_exhaustive()
    }
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
        Ok(Self {
            users: map,
            mock_secret: generate_salt()?,
        })
    }

    /// Credentials to authenticate `user` against — the real ones, or a stand-in when the name is
    /// unknown, so the exchange looks the same either way and only the proof check decides.
    ///
    /// This is deliberately the only way to reach a user's credentials. A lookup that could return
    /// "no such user" invites answering on the spot, and that answer names the absent user through
    /// the shape of the exchange no matter how carefully its text is worded: a real user gets a
    /// `server-first` reply and a made-up one does not. Here the caller cannot tell, so it has no
    /// choice but to run the exchange and let the proof fail.
    ///
    /// The stand-in advertises [`DEFAULT_ITERATIONS`], matching what every configured user is
    /// created with. A store that ever holds users at differing counts must pass the count in use
    /// here instead, or the advertised number distinguishes them again.
    #[must_use]
    pub(crate) fn credentials_for(&self, user: &str) -> Cow<'_, StoredCredentials> {
        // The stand-in is derived whether or not it is used, so both answers cost the same work.
        // Deriving it only on the miss left the unknown path measurably slower — a far weaker
        // signal than the one this closes, but a real one, and a microsecond on a per-connection
        // path is not worth keeping it.
        let stand_in = derive_mock_credentials(&self.mock_secret, user, DEFAULT_ITERATIONS);
        self.users
            .get(user)
            .map_or(Cow::Owned(stand_in), Cow::Borrowed)
    }
}
