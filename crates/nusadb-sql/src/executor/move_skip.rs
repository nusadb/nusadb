//! Statement-scoped registry of rows INSERTED by partition row movement, so the same `UPDATE`
//! statement's later per-partition branches do not re-match a row that already moved into their
//! partition (the reference engine likewise visits each row once). Installed by the parent-level
//! `UPDATE` driver around its branch loop; the per-branch matching scan consults it. A
//! thread-local is safe for the same reason as [`lock_skip`](super::lock_skip): a statement
//! executes on one blocking-pool thread end to end.

use std::cell::RefCell;
use std::collections::HashSet;

use nusadb_core::{TableId, Tid};

thread_local! {
    static MOVED: RefCell<Option<HashSet<(u64, Tid)>>> = const { RefCell::new(None) };
}

/// RAII guard: clears the registry when the statement's `UPDATE` driver ends.
pub(super) struct MoveGuard;

impl Drop for MoveGuard {
    fn drop(&mut self) {
        MOVED.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Install an empty registry for the driving `UPDATE` statement.
pub(super) fn scope() -> MoveGuard {
    MOVED.with(|slot| *slot.borrow_mut() = Some(HashSet::new()));
    MoveGuard
}

/// Record a row the movement just inserted into `table`.
pub(super) fn record(table: TableId, tid: Tid) {
    MOVED.with(|slot| {
        if let Some(set) = slot.borrow_mut().as_mut() {
            set.insert((table.0, tid));
        }
    });
}

/// Whether `tid` of `table` was inserted by this statement's own row movement.
pub(super) fn moved_here(table: TableId, tid: Tid) -> bool {
    MOVED.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|set| set.contains(&(table.0, tid)))
    })
}
