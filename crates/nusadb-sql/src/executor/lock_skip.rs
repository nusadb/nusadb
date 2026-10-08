//! Thread-local per-row overrides for `SELECT ... FOR UPDATE` / `FOR SHARE`.
//!
//! [`LockRows`](super::PhysicalOperator::LockRows) records, per base row, whether its pipeline must
//! hide it (another transaction holds its lock under `SKIP LOCKED`, it lies past the lock cap, or
//! it changed after the snapshot and no longer matches) or read its newer version (it changed and
//! still matches), then executes the pipeline under the returned guard. Every base-scan path
//! consults [`resolve`], so a hidden row never reaches the output, a `LIMIT` above the scan fills
//! up from the rows actually locked, and a locked row reads as its newest version. A thread-local
//! is safe here for the same reason as [`recursive::working_set`](super::recursive): a statement
//! executes on one blocking-pool thread end to end.
//!
//! The overrides are kept per table: a `FOR UPDATE` over an inheritance parent nests one
//! `LockRows` per table, and each must see its own decisions and leave the others' in place.
//!
//! Known scope: a guard covers every scan of its table while the pipeline runs, so a (rare)
//! subquery in the SELECT list that re-reads the same table sees the same overrides; the analyzer
//! already keeps the lockable shape simple (subquery-free WHERE).

use std::cell::RefCell;
use std::collections::HashMap;

use nusadb_core::{SharedTuple, TableId, Tid};

/// Per row of a locked table: `None` hides it, a tuple replaces it.
type Overrides = HashMap<Tid, Option<SharedTuple>>;

thread_local! {
    static OVERRIDES: RefCell<HashMap<TableId, Overrides>> = RefCell::new(HashMap::new());
}

/// RAII guard: when the `LockRows` execution ends, restores what its table's overrides were before
/// it (nothing, unless an enclosing `LockRows` locked the same table).
pub(super) struct OverrideGuard {
    table: TableId,
    previous: Option<Overrides>,
}

impl Drop for OverrideGuard {
    fn drop(&mut self) {
        OVERRIDES.with(|slot| {
            let mut slot = slot.borrow_mut();
            match self.previous.take() {
                Some(previous) => {
                    slot.insert(self.table, previous);
                },
                None => {
                    slot.remove(&self.table);
                },
            }
        });
    }
}

/// Install `overrides` for `table` for the lifetime of the returned guard: a tid mapped to `None`
/// is hidden from every scan of `table`, one mapped to a tuple reads as that tuple.
pub(super) fn scope(table: TableId, overrides: Overrides) -> OverrideGuard {
    let previous = OVERRIDES.with(|slot| slot.borrow_mut().insert(table, overrides));
    OverrideGuard { table, previous }
}

/// How a scan sees a row while a `LockRows` guard is active.
pub(super) enum Seen {
    /// As the scan read it.
    Keep,
    /// Not at all: another transaction holds it locked (`SKIP LOCKED`), it lies past the lock
    /// cap, or it changed after the snapshot and no longer matches.
    Hide,
    /// As this newer version: it changed after the snapshot and still matches.
    Replace(SharedTuple),
}

/// How a scan of `table` sees `tid` (always [`Seen::Keep`] with no guard active).
pub(super) fn resolve(table: TableId, tid: Tid) -> Seen {
    OVERRIDES.with(
        |slot| match slot.borrow().get(&table).and_then(|o| o.get(&tid)) {
            Some(None) => Seen::Hide,
            Some(Some(tuple)) => Seen::Replace(SharedTuple::clone(tuple)),
            None => Seen::Keep,
        },
    )
}

/// Whether `tid` of `table` is hidden from scans (see [`resolve`]).
pub(super) fn hidden(table: TableId, tid: Tid) -> bool {
    matches!(resolve(table, tid), Seen::Hide)
}
