//! When a serialization conflict may be retried by the server, and how long to wait first.
//!
//! Concurrency control is optimistic and locks do not wait, so two sessions writing the same row
//! make one of them fail with SQLSTATE `40001` rather than queue. For a **single statement in
//! auto-commit** that failure need not reach the client: nothing was committed, the statement has no
//! intermediate results the application has seen, and running it again is exactly what a correct
//! client retry loop would do. Inside `BEGIN…COMMIT` the same retry would be wrong — a statement
//! there may follow others whose results the application already acted on, so replaying it silently
//! changes what those results meant, and only the application knows whether the whole transaction
//! can be replayed.
//!
//! This module holds the policy rather than the loops, because there is more than one auto-commit
//! site and a client must not be able to tell which one served it. Keeping the classification, the
//! budget and the back-off here is what makes that true by construction, rather than by several
//! implementations agreeing today and drifting tomorrow.
//!
//! The sites, and why each ends where it does:
//!
//! - The embedded [`Session`](crate::Session), for a directly submitted statement and for `EXECUTE`
//!   of a prepared one. Both materialize their result before returning, so a re-attempt is invisible.
//! - The wire server's buffered path — likewise materialized, likewise invisible.
//! - The wire server's streaming path, which additionally may only re-attempt while the rows it has
//!   produced can still be recalled. Once a batch has been sent it cannot be unsaid, and sending it
//!   twice would corrupt the result, so a large `SELECT` that conflicts part-way through reports the
//!   conflict instead.
//! - A reactor-inline statement does *not* retry in place: the back-off sleeps, and sleeping on the
//!   reactor would stall every connection sharing that thread. It hands the statement back to the
//!   pool, which retries it there.
//!
//! One path deliberately has no retry at all: [`Session::stream_select`](crate::Session), which
//! writes into a caller-supplied sink it cannot recall rows from.

use crate::error::Error;

/// Session setting bounding the extra attempts: `SET max_autocommit_retries = N`.
pub const MAX_AUTOCOMMIT_RETRIES: &str = "max_autocommit_retries";

/// Extra attempts when the setting is unset.
///
/// From measurement, not taste: with eight sessions writing the *same* row in a loop — the worst
/// case a single-row counter can present — the deepest attempt actually observed was in the low
/// teens, so this leaves several times that as headroom, while the back-off below keeps the common
/// one-or-two-retry case in the sub-millisecond range.
const DEFAULT_RETRIES: u32 = 50;

/// Largest budget a session may ask for. That there *is* a ceiling is part of the contract: a
/// statement nobody can win must eventually give up and say so, rather than occupying its connection
/// indefinitely while looking to the application like a slow query.
const MAX_RETRIES: u32 = 100;

/// Parse a `max_autocommit_retries` setting value, clamped to the ceiling above.
///
/// `None` for anything that is not a non-negative integer, so `SET max_autocommit_retries` can
/// reject the value loudly rather than storing a string the retry would then quietly ignore — the
/// same treatment `work_mem` and `statement_timeout` get.
#[must_use]
pub fn parse_max_autocommit_retries(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok().map(|n| n.min(MAX_RETRIES))
}

/// The retry budget from a session's raw `max_autocommit_retries` value.
///
/// A value that does not parse falls back to the default rather than to zero: `SET` rejects such a
/// value up front, so reaching here means it arrived some other way, and a budget of zero would
/// switch the protection off — the one interpretation an unreadable value must not be given.
#[must_use]
pub fn budget(setting: Option<&str>) -> u32 {
    setting.map_or(DEFAULT_RETRIES, |v| {
        parse_max_autocommit_retries(v).unwrap_or(DEFAULT_RETRIES)
    })
}

/// Whether re-running the statement could plausibly succeed: the two failures optimistic
/// concurrency control produces under contention, and nothing else.
///
/// Deliberately narrow. A constraint violation, a type error or an I/O fault would fail the same way
/// again, so retrying them would turn one error into several while the client waits. Only the codes
/// the standard marks retryable qualify — `40001` (serialization failure) and `40P01` (deadlock
/// detected) — read through [`Error::sqlstate`], so this agrees with what the client is told by
/// construction rather than by a second, parallel classification.
#[must_use]
pub fn is_retryable_conflict(err: &Error) -> bool {
    matches!(err.sqlstate(), "40001" | "40P01")
}

/// Pause before attempt number `attempt` (1 for the first retry): exponential, capped, and jittered.
///
/// The cap keeps a heavily contended statement from turning a bounded attempt budget into an
/// unbounded wait. The jitter matters more than it looks: without it, sessions that collided once are
/// released in lockstep and collide again on the same tick, which preserves the contention instead of
/// resolving it.
///
/// The jitter deliberately does **not** come from the statement RNG. That generator is the one
/// `RANDOM()` reads and `SETSEED` pins, so drawing from it would make a seeded session's random
/// sequence depend on whether some unrelated statement happened to hit a conflict — silently breaking
/// the reproducibility `SETSEED` exists to provide.
///
/// The sleep is a real one rather than a [`Clock`](nusadb_core::Clock) port call, which would
/// otherwise be a gap in deterministic simulation. It is not: a deterministic run is single-threaded
/// and so produces no concurrent conflict, and a statement that never conflicts never gets here.
pub fn back_off(attempt: u32) {
    std::thread::sleep(std::time::Duration::from_micros(back_off_micros(attempt)));
}

/// How long [`back_off`] sleeps for `attempt`, split out so the schedule can be asserted without a
/// test that measures elapsed time.
fn back_off_micros(attempt: u32) -> u64 {
    /// First back-off — short enough to be invisible when a conflict clears on the next attempt.
    const BASE_MICROS: u64 = 200;
    /// Ceiling for a single wait. Even a full budget of capped waits stays in the sub-second range,
    /// which is what keeps the worst case a slow statement rather than a hung one.
    const CAP_MICROS: u64 = 5_000;
    /// Decorrelates the waits of sessions that collided with each other: every read hands out a
    /// distinct ticket, so two sessions backing off from the same conflict draw different jitter.
    static JITTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    // `1 << 5` × BASE already exceeds the cap, so clamping the shift keeps it from overflowing on a
    // deep attempt without changing the schedule.
    let ceiling = BASE_MICROS
        .saturating_mul(1_u64 << attempt.min(5))
        .min(CAP_MICROS);
    // SplitMix64 over the ticket, so consecutive tickets give unrelated waits rather than a ramp.
    let mut z = JITTER
        .fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Full jitter — a wait anywhere in [0, ceiling), which spreads colliding sessions better than
    // jittering narrowly around a fixed centre.
    z % ceiling
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_or_unusable_setting_keeps_the_protection_on() {
        assert_eq!(budget(None), DEFAULT_RETRIES);
        // A typo must not read as "switch it off" — that would silently undo the feature.
        assert_eq!(budget(Some("banana")), DEFAULT_RETRIES);
        assert_eq!(budget(Some("")), DEFAULT_RETRIES);
        assert_eq!(budget(Some("-1")), DEFAULT_RETRIES);
    }

    #[test]
    fn an_explicit_budget_is_honoured_up_to_the_ceiling() {
        assert_eq!(budget(Some("0")), 0, "zero must switch the retry off");
        assert_eq!(budget(Some("7")), 7);
        assert_eq!(budget(Some(" 7 ")), 7, "surrounding space is not a typo");
        assert_eq!(budget(Some("100000")), MAX_RETRIES, "clamped, not accepted");
    }

    #[test]
    fn only_the_standard_retryable_codes_qualify() {
        let conflict = Error::Core(nusadb_core::Error::SerializationConflict {
            txn: nusadb_core::TxnId(1),
        });
        assert_eq!(conflict.sqlstate(), "40001");
        assert!(is_retryable_conflict(&conflict));

        // Everything a re-run cannot change stays a single, immediate error.
        for err in [
            Error::DivisionByZero,
            Error::IntegerOutOfRange,
            Error::Cancelled,
            Error::Empty,
        ] {
            assert!(
                !is_retryable_conflict(&err),
                "{} ({}) must not be retried",
                err,
                err.sqlstate()
            );
        }
    }

    #[test]
    fn the_back_off_grows_then_holds_at_the_cap() {
        // Every wait stays inside the ceiling for its attempt, and the ceiling stops growing rather
        // than overflowing once the shift is clamped.
        for attempt in 1_u32..=64 {
            let ceiling = 200_u64.saturating_mul(1 << attempt.min(5)).min(5_000);
            let waited = back_off_micros(attempt);
            assert!(
                waited < ceiling,
                "attempt {attempt}: waited {waited}µs, ceiling {ceiling}µs"
            );
        }
        // A whole default budget of capped waits is bounded well under a second, so an unwinnable
        // statement is slow, never hung.
        assert!(u64::from(DEFAULT_RETRIES) * 5_000 < 1_000_000);
    }

    #[test]
    fn two_back_offs_at_the_same_attempt_differ() {
        // Sessions released in lockstep collide again; the jitter is what prevents it. Sampling a
        // deep attempt gives a wide enough ceiling that equal draws would be a real signal.
        let draws: Vec<u64> = (0..8).map(|_| back_off_micros(5)).collect();
        assert!(
            draws.iter().any(|d| *d != draws[0]),
            "back-off is not jittering: {draws:?}"
        );
    }
}
