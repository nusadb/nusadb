//! Thread-local "skip these base rows" set for `FOR UPDATE ... SKIP LOCKED` (the job-queue
//! pattern: workers claim rows without blocking on each other).
//!
//! [`LockRows`](super::PhysicalOperator::LockRows) populates the set with the tids whose row lock
//! another transaction holds, then executes its inner pipeline under the returned guard; every
//! base-scan path consults [`skipped`] so a skipped row never reaches the output — and a `LIMIT`
//! above the scan therefore fills up from lockable rows, like the reference engine. A thread-local is safe here for
//! the same reason as [`recursive::working_set`](super::recursive): a statement executes on one
//! blocking-pool thread end to end.
//!
//! Known scope: the guard covers every scan of the target table while the pipeline runs, so a
//! (rare) subquery in the SELECT list that re-reads the same table also skips those rows; the
//! analyzer already keeps the lockable shape simple (single table, subquery-free WHERE).

use std::cell::RefCell;
use std::collections::HashSet;

use nusadb_core::{TableId, Tid};

thread_local! {
    static SKIP: RefCell<Option<(TableId, HashSet<Tid>)>> = const { RefCell::new(None) };
}

/// RAII guard: clears the skip set when the `LockRows` execution ends.
pub(super) struct SkipGuard;

impl Drop for SkipGuard {
    fn drop(&mut self) {
        SKIP.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Install the skip set for `table` for the lifetime of the returned guard.
pub(super) fn scope(table: TableId, tids: HashSet<Tid>) -> SkipGuard {
    SKIP.with(|slot| *slot.borrow_mut() = Some((table, tids)));
    SkipGuard
}

/// Whether `tid` of `table` is currently being skipped (`SKIP LOCKED`).
pub(super) fn skipped(table: TableId, tid: Tid) -> bool {
    SKIP.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|(t, set)| *t == table && set.contains(&tid))
    })
}
