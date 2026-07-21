//! Statement wall clock for the niladic time functions (`NOW`, `CURRENT_TIMESTAMP`,
//! `CURRENT_DATE`, `CURRENT_TIME`).
//!
//! SQL requires these to be **statement-stable**: every occurrence in one statement must observe
//! the same instant, regardless of how many rows are scanned. The instant is captured once, at the
//! top of the executor's [`dispatch`](super::dispatch), and pinned in a thread-local that the
//! evaluator reads for each row. (`nusadb-sql` is single-threaded per statement — the wire layer
//! dispatches each query on its own blocking task — so a thread-local is the natural carrier and
//! costs nothing on the hot path.)

use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

/// Microseconds in a calendar day — the divisor splitting an epoch-micros instant into a
/// days-since-epoch date and a micros-since-midnight time.
pub(super) const MICROS_PER_DAY: i64 = 86_400_000_000;

thread_local! {
    /// The statement instant in microseconds since the Unix epoch, pinned by
    /// [`set_statement_now`]. `None` before any statement has run on this thread.
    static STATEMENT_NOW_MICROS: Cell<Option<i64>> = const { Cell::new(None) };
}

/// Pin the wall clock as the current statement's instant. Called once per top-level statement so
/// all of its time functions agree.
pub(super) fn set_statement_now() {
    STATEMENT_NOW_MICROS.with(|cell| cell.set(Some(wall_clock_micros())));
}

/// The pinned statement instant (microseconds since the Unix epoch). If nothing has been pinned on
/// this thread yet — e.g. a unit test that evaluates an expression without going through
/// [`dispatch`](super::dispatch) — capture the wall clock now and memoize it so repeated reads in
/// the same context stay consistent.
pub(super) fn statement_now_micros() -> i64 {
    STATEMENT_NOW_MICROS.with(|cell| {
        cell.get().unwrap_or_else(|| {
            let micros = wall_clock_micros();
            cell.set(Some(micros));
            micros
        })
    })
}

/// Days since the Unix epoch (proleptic Gregorian) for the pinned statement instant. Floored, so a
/// pre-epoch instant lands on the calendar day that contains it.
pub(super) fn statement_today() -> i32 {
    let days = statement_now_micros().div_euclid(MICROS_PER_DAY);
    i32::try_from(days).unwrap_or(if days < 0 { i32::MIN } else { i32::MAX })
}

/// Microseconds since midnight for the pinned statement instant, always in `[0, MICROS_PER_DAY)`.
pub(super) fn statement_time_of_day() -> i64 {
    statement_now_micros().rem_euclid(MICROS_PER_DAY)
}

/// Microseconds since the epoch at midnight of `days` (days since the epoch). `None` on overflow —
/// only reachable for dates near the i64 micros limit (year ~294247), never for realistic data.
pub(super) fn day_start_micros(days: i32) -> Option<i64> {
    i64::from(days).checked_mul(MICROS_PER_DAY)
}

/// The wall clock as microseconds since the Unix epoch. A clock reading before the epoch clamps to
/// `0`; a reading beyond `i64::MAX` micros (year ~294247) saturates — neither can panic.
fn wall_clock_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX)
        })
}
