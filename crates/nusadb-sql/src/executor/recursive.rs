//! Recursive CTE (`WITH RECURSIVE`) execution.
//!
//! A recursive CTE is evaluated to a fixpoint: run the *base* term, then repeatedly run the
//! *recursive* term — which references the CTE itself — over the rows produced by the previous
//! round, accumulating results until a round adds nothing new. The CTE is exposed to both the
//! recursive term and the outer query as a synthetic table (a [`TableId`] in the reserved high
//! range); a scan of that table reads the current working set from the thread-local registry below
//! rather than the storage engine.

use std::cell::RefCell;
use std::collections::HashMap;

use nusadb_core::TableId;

use super::Row;

thread_local! {
    /// The rows currently bound to each recursive-CTE synthetic table on this thread.
    static WORKING_SETS: RefCell<HashMap<u64, Vec<Row>>> = RefCell::new(HashMap::new());
}

/// The rows currently bound to the recursive-CTE table `id`, or `None` if `id` is not a bound
/// recursive CTE (an ordinary table). The scan path consults this before the engine.
#[must_use]
pub(super) fn working_set(id: TableId) -> Option<Vec<Row>> {
    WORKING_SETS.with(|sets| sets.borrow().get(&id.0).cloned())
}

/// Bind `rows` to the recursive-CTE table `id` for the lifetime of the returned guard, restoring
/// the previous binding on drop (so nested recursive CTEs and re-entrant scans stay correct).
#[must_use]
pub(super) fn bind(id: TableId, rows: Vec<Row>) -> WorkingSetGuard {
    let previous = WORKING_SETS.with(|sets| sets.borrow_mut().insert(id.0, rows));
    WorkingSetGuard { id: id.0, previous }
}

/// Restores the previous working-set binding for a recursive-CTE table on drop.
#[derive(Debug)]
pub(super) struct WorkingSetGuard {
    id: u64,
    previous: Option<Vec<Row>>,
}

impl Drop for WorkingSetGuard {
    fn drop(&mut self) {
        WORKING_SETS.with(|sets| {
            let mut map = sets.borrow_mut();
            match self.previous.take() {
                Some(rows) => map.insert(self.id, rows),
                None => map.remove(&self.id),
            };
        });
    }
}
