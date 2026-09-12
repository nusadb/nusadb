//! Session-scoped advisory locks.
//!
//! An advisory lock is a cooperative, application-defined lock keyed by an integer the application
//! chooses; the engine attaches no meaning to the key. This registry is process-global — every
//! session on the same server sees the same lock table — and each lock is held by a *session*,
//! identified here by its unique temporary-schema name (the one stable per-session token both the
//! embedded and wire paths already carry). A session may take the same lock more than once
//! (re-entrant); an equal number of unlocks releases it. Every lock a session holds is released
//! when the session ends.
//!
//! Only the **non-blocking, exclusive, session-scoped** operations are built: acquiring is
//! try-only, consistent with the engine's no-wait locking elsewhere (a blocking wait would park a
//! worker thread against that design). Blocking, shared-mode, and transaction-scoped variants are
//! refused at analysis with a clear message rather than half-implemented.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

/// `lock key -> { session token -> re-entrant hold count }`. A key with no holders is removed, so an
/// empty inner map never lingers.
static ADVISORY_LOCKS: LazyLock<Mutex<HashMap<i64, HashMap<String, u64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn locks() -> std::sync::MutexGuard<'static, HashMap<i64, HashMap<String, u64>>> {
    // A panic while the lock was held cannot corrupt the plain map, so recover the guard rather than
    // propagate the poison (a poisoned advisory table must not wedge every later session).
    ADVISORY_LOCKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Try to take advisory lock `key` for `session` without waiting. Succeeds (returns `true`) when the
/// lock is free or already held by this same session (re-entrant, bumping its count); returns
/// `false` when another session holds it.
pub(super) fn try_lock(key: i64, session: &str) -> bool {
    let mut map = locks();
    let held_by_other = map
        .get(&key)
        .is_some_and(|holders| holders.keys().any(|s| s != session));
    if held_by_other {
        return false;
    }
    *map.entry(key)
        .or_default()
        .entry(session.to_owned())
        .or_insert(0) += 1;
    true
}

/// Release one hold of advisory lock `key` by `session`. Returns `true` if this session held the
/// lock (and one hold was released), `false` if it did not hold it.
pub(super) fn unlock(key: i64, session: &str) -> bool {
    let mut map = locks();
    let Some(holders) = map.get_mut(&key) else {
        return false;
    };
    let Some(count) = holders.get_mut(session) else {
        return false;
    };
    *count -= 1;
    if *count == 0 {
        holders.remove(session);
        if holders.is_empty() {
            map.remove(&key);
        }
    }
    true
}

/// Release every advisory lock held by `session` (all keys, regardless of re-entrant count).
pub(super) fn unlock_all(session: &str) {
    let mut map = locks();
    map.retain(|_, holders| {
        holders.remove(session);
        !holders.is_empty()
    });
}

#[cfg(test)]
mod tests {
    use super::{try_lock, unlock, unlock_all};

    #[test]
    fn exclusive_across_sessions_and_reentrant_within_one() {
        let (a, b) = ("nusadb_temp_1", "nusadb_temp_2");
        // A distinct key per test keeps the process-global table from colliding across tests.
        let k = 0x5AD_0001;
        assert!(try_lock(k, a)); // free -> A takes it
        assert!(!try_lock(k, b)); // B is refused
        assert!(try_lock(k, a)); // A re-enters (count 2)
        assert!(!try_lock(k, b)); // still refused
        assert!(unlock(k, a)); // count 1 -> still held by A
        assert!(!try_lock(k, b)); // still refused
        assert!(unlock(k, a)); // count 0 -> released
        assert!(try_lock(k, b)); // now B can take it
        assert!(!unlock(k, a)); // A never held the current hold
        unlock_all(b);
        assert!(try_lock(k, a)); // fully free again after B's release
        unlock_all(a);
    }

    #[test]
    fn unlock_all_drops_every_key() {
        let s = "nusadb_temp_99";
        let (k1, k2) = (0x5AD_0002, 0x5AD_0003);
        assert!(try_lock(k1, s));
        assert!(try_lock(k2, s));
        assert!(try_lock(k1, s)); // re-entrant on k1
        unlock_all(s);
        // Both keys are free for another session now.
        let other = "nusadb_temp_98";
        assert!(try_lock(k1, other));
        assert!(try_lock(k2, other));
        unlock_all(other);
    }
}
