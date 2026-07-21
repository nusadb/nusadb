//! Per-node actual-row instrumentation for `EXPLAIN ANALYZE`.
//!
//! While a [`Session`] is live, both execution surfaces record how many rows each physical
//! operator produced — [`super::ops::execute_op`] as each node materializes, and the streaming
//! sources through their counting wrappers as they drain — keyed by the operator node's address,
//! which is stable because the instrumented run executes the *same* plan tree that `EXPLAIN`
//! later formats. Off (the default), the only cost on the hot path is one thread-local check per
//! operator, never per row.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::planner::PhysicalOperator;

thread_local! {
    /// The live collection map (operator address → rows produced), or `None` when off.
    static ACTUAL_ROWS: RefCell<Option<HashMap<usize, u64>>> = const { RefCell::new(None) };
}

/// RAII instrumentation window: collection starts at construction and always stops on drop, so an
/// erroring `EXPLAIN ANALYZE` cannot leak an enabled collector into later statements.
pub(super) struct Session;

impl Session {
    /// Start collecting (clearing any stale map defensively).
    pub(super) fn begin() -> Self {
        ACTUAL_ROWS.with(|c| *c.borrow_mut() = Some(HashMap::new()));
        Self
    }

    /// End the session and return the collected counts: operator address → total rows produced.
    #[allow(
        clippy::unused_self,
        reason = "consuming self is the point: taking the counts ends the RAII session, so a \
                  stale collector cannot outlive its EXPLAIN ANALYZE"
    )]
    pub(super) fn take(self) -> HashMap<usize, u64> {
        ACTUAL_ROWS
            .with(|c| c.borrow_mut().take())
            .unwrap_or_default()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        ACTUAL_ROWS.with(|c| *c.borrow_mut() = None);
    }
}

/// Whether a session is live — checked once per operator, never per row.
pub(super) fn enabled() -> bool {
    ACTUAL_ROWS.with(|c| c.borrow().is_some())
}

/// The instrumentation key of an operator node: its address in the executed plan tree.
pub(super) fn key(op: &PhysicalOperator) -> usize {
    std::ptr::from_ref(op) as usize
}

/// Add `rows` to the operator's running total — a node re-executed in a loop accumulates, so the
/// reported figure is the loop total.
pub(super) fn record(key: usize, rows: u64) {
    ACTUAL_ROWS.with(|c| {
        if let Some(map) = c.borrow_mut().as_mut() {
            *map.entry(key).or_insert(0) += rows;
        }
    });
}
