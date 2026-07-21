//! Out-of-band cancel-key registry.
//!
//! Each connection registers its [`CancelToken`] under a unique backend key `(pid, secret)` and
//! sends the key to the client (`BackendKeyData`). To cancel an in-flight statement, a client opens
//! a *fresh* connection and sends a `CancelRequest` with that key; the server looks the token up
//! here and trips it, so the running statement aborts at its next cooperative check point.
//!
//! The registry is process-global (one server per process is the norm; keys are unique across
//! servers via a shared counter). The `secret` is a CSPRNG value so an attacker cannot cancel
//! another connection's statement without having observed its `BackendKeyData`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use nusadb_sql::cancel::CancelToken;
use ring::rand::{SecureRandom, SystemRandom};

static REGISTRY: OnceLock<Mutex<HashMap<u32, (u32, CancelToken)>>> = OnceLock::new();
static NEXT_PID: AtomicU32 = AtomicU32::new(1);

/// Access the registry map, recovering from a poisoned lock (a panic elsewhere must not wedge
/// cancellation for every other connection).
fn with_registry<R>(f: impl FnOnce(&mut HashMap<u32, (u32, CancelToken)>) -> R) -> R {
    let mutex = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut guard)
}

/// A connection's backend cancellation key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKey {
    /// Unique connection id.
    pub pid: u32,
    /// Secret proving the requester observed the original `BackendKeyData`.
    pub secret: u32,
}

/// Register `token` for a fresh connection, returning its key and an RAII [`Registration`] that
/// deregisters the key when dropped (at the end of the connection).
#[must_use]
pub fn register(token: CancelToken) -> (BackendKey, Registration) {
    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    let secret = random_u32();
    with_registry(|map| map.insert(pid, (secret, token)));
    (BackendKey { pid, secret }, Registration { pid })
}

/// Trip the cancel token of the connection identified by `(pid, secret)`, iff the secret matches.
/// Returns whether a matching connection was found and cancelled.
pub fn cancel(pid: u32, secret: u32) -> bool {
    with_registry(|map| {
        if let Some((stored_secret, token)) = map.get(&pid)
            && *stored_secret == secret
        {
            token.store(true, Ordering::Relaxed);
            return true;
        }
        false
    })
}

/// Deregisters a connection's cancel key on drop.
#[derive(Debug)]
pub struct Registration {
    pid: u32,
}

impl Drop for Registration {
    fn drop(&mut self) {
        with_registry(|map| map.remove(&self.pid));
    }
}

/// A CSPRNG `u32` for the cancel secret. A fill failure (astronomically unlikely) falls back to a
/// counter-derived value so registration never fails.
fn random_u32() -> u32 {
    let mut buf = [0u8; 4];
    if SystemRandom::new().fill(&mut buf).is_ok() {
        u32::from_be_bytes(buf)
    } else {
        NEXT_PID.load(Ordering::Relaxed).wrapping_mul(2_654_435_761)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn cancel_trips_the_registered_token_only_with_the_right_secret() {
        let token = Arc::new(AtomicBool::new(false));
        let (key, _registration) = register(Arc::clone(&token));

        // Wrong secret: no effect.
        assert!(!cancel(key.pid, key.secret.wrapping_add(1)));
        assert!(!token.load(Ordering::Relaxed));

        // Right key: the token is tripped.
        assert!(cancel(key.pid, key.secret));
        assert!(token.load(Ordering::Relaxed));
    }

    #[test]
    fn an_unknown_key_cancels_nothing() {
        assert!(!cancel(u32::MAX, 0));
    }

    #[test]
    fn dropping_the_registration_deregisters_the_key() {
        let token = Arc::new(AtomicBool::new(false));
        let key = {
            let (key, _registration) = register(Arc::clone(&token));
            assert!(cancel(key.pid, key.secret)); // present while registered
            token.store(false, Ordering::Relaxed);
            key
        };
        // After the guard drops, the key is gone.
        assert!(!cancel(key.pid, key.secret));
    }
}
