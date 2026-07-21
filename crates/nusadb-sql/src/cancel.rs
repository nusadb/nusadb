//! Cooperative statement cancellation.
//!
//! The executor is synchronous and runs one statement per thread (the wire server drives it on a
//! blocking pool thread). Rather than thread a cancel handle through every operator, the running
//! statement publishes its [`CancelToken`] in a thread-local, and the hot loops call [`check`] at
//! their boundaries. Check points today: the universal `scan_table` row loop (every
//! `SELECT`/`UPDATE`/`DELETE` materializes through it); the nested-loop join, polled per outer row
//! AND amortized (per ~1024) across the inner loop so a small-outer × large-inner cross join is
//! interruptible; the sort key-evaluation loop; and window partitioning / per-partition processing.
//! Still unpolled (a follow-up): the pure-`Vec` set-op multiset helpers and `dedupe_rows`
//! (`INTERSECT`/`EXCEPT`/`UNION` dedup), and the comparison sort's `sort_by` phase itself. When the
//! token is tripped — by a statement timeout or an out-of-band cancel request — the next `check`
//! returns [`Error::Cancelled`], the statement unwinds via `?`, and its transaction rolls back.
//!
//! Cancellation is *cooperative*: a statement is interrupted at the next check point (e.g. between
//! row batches of a scan), not pre-emptively mid-instruction.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::Error;

/// Parse a `statement_timeout` setting value into a [`Duration`].
///
/// Follows the conventional time-GUC form: a bare integer is **milliseconds**; an integer with a
/// `us` / `ms` / `s` / `min` / `h` / `d` suffix (case-insensitive, optional whitespace) is scaled
/// accordingly. `Duration::ZERO` (from `0`) means "no timeout". Returns `None` for anything else
/// (empty, negative, fractional, unknown unit, overflow) so `SET statement_timeout` can reject the
/// value loudly instead of storing a string the timer would then silently ignore.
#[must_use]
pub fn parse_statement_timeout(value: &str) -> Option<Duration> {
    let v = value.trim();
    let split = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
    let (digits, unit) = v.split_at(split);
    if digits.is_empty() {
        return None;
    }
    let n: u64 = digits.parse().ok()?;
    // Scale in microseconds — the finest unit accepted — so every branch shares one overflow check.
    let unit_micros: u64 = match unit.trim().to_ascii_lowercase().as_str() {
        "us" => 1,
        // A bare integer is in milliseconds (the conventional time-GUC unit).
        "" | "ms" => 1_000,
        "s" => 1_000_000,
        "min" => 60_000_000,
        "h" => 3_600_000_000,
        "d" => 86_400_000_000,
        _ => return None,
    };
    n.checked_mul(unit_micros).map(Duration::from_micros)
}

/// A shared cancellation flag for one running statement. The executor reads it; a timer or
/// a cancel-request handler sets it.
pub type CancelToken = Arc<AtomicBool>;

thread_local! {
    /// The cancel token of the statement currently executing on this thread, if any.
    static CURRENT: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
}

/// Install `token` as the current thread's cancel token for the lifetime of the returned guard.
///
/// The previous token is restored on drop. Wrap a statement's execution in this so a panic or early
/// return cannot leak the token onto the next statement that reuses the thread.
#[must_use]
pub fn scope(token: CancelToken) -> CancelGuard {
    let previous = CURRENT.with(|c| c.borrow_mut().replace(token));
    CancelGuard { previous }
}

/// Restores the previous cancel token when dropped (see [`scope`]).
#[derive(Debug)]
pub struct CancelGuard {
    previous: Option<CancelToken>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}

/// Return [`Error::Cancelled`] if the current statement's cancel token has been tripped, else `Ok`.
/// Called at executor check points (scan/loop boundaries). A no-op when no token is installed.
///
/// # Errors
/// [`Error::Cancelled`] when the current statement has been cancelled.
#[inline]
pub fn check() -> Result<(), Error> {
    CURRENT.with(|c| {
        if c.borrow()
            .as_ref()
            .is_some_and(|t| t.load(Ordering::Relaxed))
        {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_is_a_no_op_without_a_token() {
        assert!(check().is_ok());
    }

    #[test]
    fn a_tripped_token_makes_check_fail_until_the_guard_drops() {
        let token = Arc::new(AtomicBool::new(false));
        {
            let _guard = scope(Arc::clone(&token));
            assert!(check().is_ok());
            token.store(true, Ordering::Relaxed);
            assert!(matches!(check(), Err(Error::Cancelled)));
        }
        // Once the guard drops, the token is no longer consulted on this thread.
        assert!(check().is_ok());
    }

    #[test]
    fn parse_statement_timeout_follows_time_guc_units() {
        for (input, expect_micros) in [
            ("100", Some(100_000)), // bare integer = milliseconds
            ("100ms", Some(100_000)),
            ("500us", Some(500)),
            ("2s", Some(2_000_000)),
            ("1min", Some(60_000_000)),
            ("1h", Some(3_600_000_000)),
            ("1d", Some(86_400_000_000)),
            (" 100 MS ", Some(100_000)), // unit is case-insensitive, whitespace tolerated
            ("0", Some(0)),              // no timeout
            ("banana", None),
            ("100xs", None),
            ("-1", None),
            ("1.5s", None), // fractional values are rejected, not rounded
            ("", None),
            ("999999999999999999d", None), // overflow
        ] {
            assert_eq!(
                parse_statement_timeout(input),
                expect_micros.map(Duration::from_micros),
                "parse_statement_timeout({input:?})"
            );
        }
    }

    #[test]
    fn nested_scopes_restore_the_outer_token() {
        let outer = Arc::new(AtomicBool::new(true));
        let inner = Arc::new(AtomicBool::new(false));
        let _outer_guard = scope(Arc::clone(&outer));
        assert!(matches!(check(), Err(Error::Cancelled)));
        {
            let _inner_guard = scope(Arc::clone(&inner));
            assert!(check().is_ok()); // inner token not tripped
        }
        // Outer token is restored after the inner guard drops.
        assert!(matches!(check(), Err(Error::Cancelled)));
    }
}
