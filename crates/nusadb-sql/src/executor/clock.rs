//! Wall clock for the niladic time functions.
//!
//! SQL fixes three different instants:
//!
//! - the **transaction** instant, read by `NOW()`, `TRANSACTION_TIMESTAMP()`, `CURRENT_TIMESTAMP`,
//!   `CURRENT_DATE`, `CURRENT_TIME` and `LOCALTIMESTAMP`. It is the time the transaction began and
//!   stays the same for every statement of that transaction;
//! - the **statement** instant, read by `STATEMENT_TIMESTAMP()`. It is the time the client's
//!   statement began, and every row and every nested statement it runs (a trigger, a procedure or
//!   `DO` body, a SQL function) observes it unchanged;
//! - the real time, read by `CLOCK_TIMESTAMP()` ([`wall_clock_micros`]), which moves on every call.
//!
//! In autocommit the transaction holds exactly one statement, so the first two coincide.
//!
//! Both pinned instants live in thread-locals. `nusadb-sql` is single-threaded per statement (the
//! wire layer dispatches each query on its own blocking task), so a thread-local is the natural
//! carrier and costs nothing on the hot path. A blocking task can run on a different pool thread for
//! each statement of one transaction, so the transaction's start cannot be remembered here: the
//! owner of the transaction (the wire connection or the embedded [`Session`](super::Session))
//! records it at `BEGIN` and hands it in with every statement through [`pin_transaction_start`].
//! The hand-off names the transaction, and a statement running under any other transaction ignores
//! it, so a value left behind on a pooled thread can never leak into another connection.

use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

use nusadb_core::TxnId;

/// Microseconds in a calendar day — the divisor splitting an epoch-micros instant into a
/// days-since-epoch date and a micros-since-midnight time.
pub(super) const MICROS_PER_DAY: i64 = 86_400_000_000;

thread_local! {
    /// The statement instant in microseconds since the Unix epoch. `None` before any statement
    /// has run on this thread.
    static STATEMENT_NOW_MICROS: Cell<Option<i64>> = const { Cell::new(None) };

    /// The transaction instant in microseconds since the Unix epoch. `None` before any statement
    /// has run on this thread.
    static TRANSACTION_NOW_MICROS: Cell<Option<i64>> = const { Cell::new(None) };

    /// The start time of an open transaction block, as handed in by the transaction's owner for
    /// the statement about to run: `(transaction id, start micros)`.
    static PINNED_TXN_START: Cell<Option<(u64, i64)>> = const { Cell::new(None) };

    /// How many [`StatementClock`] guards are alive on this thread. Only the outermost one pins
    /// the instants, so a nested statement (trigger, procedure body, `Batch` child) keeps the
    /// client statement's clock.
    static CLOCK_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Record when the transaction block of the next statement began (`None` in autocommit).
///
/// Read once by the next outermost statement to start, and only when that statement runs under
/// the same transaction id; cleared when that statement ends.
pub fn pin_transaction_start(start: Option<(TxnId, i64)>) {
    PINNED_TXN_START.with(|cell| cell.set(start.map(|(txn, micros)| (txn.0, micros))));
}

/// Keeps the statement clock pinned while it is alive. Created by [`enter_statement`].
#[must_use = "the statement clock is only held while the guard is alive"]
pub(super) struct StatementClock {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for StatementClock {
    fn drop(&mut self) {
        let left = CLOCK_DEPTH.with(|depth| {
            let left = depth.get().saturating_sub(1);
            depth.set(left);
            left
        });
        // The hand-off served this statement; the next one must bring its own.
        if left == 0 {
            pin_transaction_start(None);
        }
    }
}

/// Start a statement under transaction `txn`. The outermost call on this thread reads the wall
/// clock as the statement instant and takes the transaction instant from [`pin_transaction_start`]
/// when it names `txn`, falling back to the statement instant (autocommit). A nested call (a
/// trigger, a procedure body, a `Batch` child) changes nothing, so the whole client statement shares
/// one clock.
pub(super) fn enter_statement(txn: TxnId) -> StatementClock {
    let depth = CLOCK_DEPTH.with(|depth| {
        let current = depth.get();
        depth.set(current.saturating_add(1));
        current
    });
    if depth == 0 {
        let statement = wall_clock_micros();
        let transaction = PINNED_TXN_START
            .with(Cell::get)
            .filter(|(id, _)| *id == txn.0)
            .map_or(statement, |(_, started)| started);
        STATEMENT_NOW_MICROS.with(|cell| cell.set(Some(statement)));
        TRANSACTION_NOW_MICROS.with(|cell| cell.set(Some(transaction)));
    }
    StatementClock {
        _not_send: std::marker::PhantomData,
    }
}

/// Read a pinned instant. If nothing has been pinned on this thread yet (a unit test that evaluates
/// an expression without running a statement), capture the wall clock now and memoize it in both
/// cells so repeated reads in the same context stay consistent.
fn pinned(cell: &'static std::thread::LocalKey<Cell<Option<i64>>>) -> i64 {
    cell.with(Cell::get).unwrap_or_else(|| {
        let micros = wall_clock_micros();
        for slot in [&STATEMENT_NOW_MICROS, &TRANSACTION_NOW_MICROS] {
            slot.with(|c| {
                if c.get().is_none() {
                    c.set(Some(micros));
                }
            });
        }
        micros
    })
}

/// The statement instant (`STATEMENT_TIMESTAMP()`), microseconds since the Unix epoch.
pub(super) fn statement_now_micros() -> i64 {
    pinned(&STATEMENT_NOW_MICROS)
}

/// The transaction instant (`NOW()` and the `CURRENT_*` / `LOCAL*` forms), microseconds since the
/// Unix epoch.
pub(super) fn transaction_now_micros() -> i64 {
    pinned(&TRANSACTION_NOW_MICROS)
}

/// Microseconds since the epoch at midnight of `days` (days since the epoch). `None` on overflow —
/// only reachable for dates near the i64 micros limit (year ~294247), never for realistic data.
pub(super) fn day_start_micros(days: i32) -> Option<i64> {
    i64::from(days).checked_mul(MICROS_PER_DAY)
}

/// The wall clock as microseconds since the Unix epoch, read now (`CLOCK_TIMESTAMP()`).
///
/// A clock reading before the epoch clamps to `0`; a reading beyond `i64::MAX` micros (year
/// ~294247) saturates. Neither can panic.
pub fn wall_clock_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transaction_instant_comes_from_the_pin_only_for_the_named_transaction() {
        pin_transaction_start(Some((TxnId(41), 1_000)));
        {
            let _clock = enter_statement(TxnId(41));
            assert_eq!(transaction_now_micros(), 1_000);
            assert!(statement_now_micros() > 1_000);
        }
        // Another transaction (a pooled thread reused by another connection) ignores the pin.
        pin_transaction_start(Some((TxnId(41), 1_000)));
        {
            let _clock = enter_statement(TxnId(42));
            assert_eq!(transaction_now_micros(), statement_now_micros());
        }
        pin_transaction_start(None);
        {
            let _clock = enter_statement(TxnId(41));
            assert_eq!(transaction_now_micros(), statement_now_micros());
        }
    }

    #[test]
    fn the_pin_serves_one_statement() {
        pin_transaction_start(Some((TxnId(9), 500)));
        {
            let _clock = enter_statement(TxnId(9));
            assert_eq!(transaction_now_micros(), 500);
        }
        let _clock = enter_statement(TxnId(9));
        assert_eq!(transaction_now_micros(), statement_now_micros());
    }

    #[test]
    fn a_nested_statement_keeps_the_outer_clock() {
        pin_transaction_start(None);
        let _outer = enter_statement(TxnId(7));
        let statement = statement_now_micros();
        std::thread::sleep(std::time::Duration::from_millis(2));
        {
            let _inner = enter_statement(TxnId(7));
            assert_eq!(statement_now_micros(), statement);
        }
        assert_eq!(statement_now_micros(), statement);
    }
}
