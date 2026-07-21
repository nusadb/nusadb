//! In-memory budget tracking for spilling operators.
//!
//! A spilling operator (sort run-generation, hash-join partitioning, …) accumulates rows in memory
//! and asks the budget, row by row, whether the next one still fits. When it does not, the operator
//! flushes what it holds to a spill run and resets. Row sizes reuse the same estimator `work_mem`
//! uses ([`row_bytes`](super::super::ops::row_bytes)), so the two budgets agree.

use crate::ast;
use crate::executor::ops::row_bytes;

/// Tracks the bytes an operator currently holds in memory against a fixed limit.
pub(in crate::executor) struct MemBudget {
    used: usize,
    limit: usize,
}

impl MemBudget {
    /// A budget that admits up to `limit` bytes. A `limit` of `0` admits everything (spill
    /// effectively disabled), matching the `work_mem == 0` convention.
    pub(in crate::executor) const fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }

    /// Account for `row` and report whether it still fits within the limit.
    ///
    /// Returns `false` when adding the row would exceed the limit **and** the budget is not empty —
    /// the caller should flush the current in-memory batch, [`reset`](Self::reset), then call again.
    /// A single row larger than the whole limit is always admitted into an empty budget, so an
    /// operator can never deadlock on a row it can never make fit.
    pub(in crate::executor) fn admit(&mut self, row: &[ast::Value]) -> bool {
        let bytes = row_bytes(row);
        if self.limit != 0 && self.used != 0 && self.used.saturating_add(bytes) > self.limit {
            return false;
        }
        self.used = self.used.saturating_add(bytes);
        true
    }

    /// Forget the accumulated bytes (call after flushing a batch to disk).
    pub(in crate::executor) const fn reset(&mut self) {
        self.used = 0;
    }

    /// Bytes currently accounted for.
    pub(in crate::executor) const fn used(&self) -> usize {
        self.used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(int_cols: usize) -> Vec<ast::Value> {
        (0..int_cols)
            .map(|i| ast::Value::Int(i64::try_from(i).unwrap_or_default()))
            .collect()
    }

    #[test]
    fn admits_until_the_limit_then_refuses() {
        let one = row_bytes(&row(1));
        // Room for exactly two such rows.
        let mut budget = MemBudget::new(one * 2);
        assert!(budget.admit(&row(1)), "first row fits");
        assert!(budget.admit(&row(1)), "second row fits exactly");
        assert!(!budget.admit(&row(1)), "third row overflows");
        assert_eq!(budget.used(), one * 2, "the refused row is not accounted");
    }

    #[test]
    fn a_single_oversized_row_is_admitted_into_an_empty_budget() {
        let mut budget = MemBudget::new(1); // smaller than any row
        assert!(
            budget.admit(&row(4)),
            "an empty budget must admit one row even if it alone exceeds the limit"
        );
        assert!(!budget.admit(&row(1)), "but then the next row overflows");
    }

    #[test]
    fn reset_clears_the_accounting() {
        let mut budget = MemBudget::new(row_bytes(&row(1)) * 2);
        assert!(budget.admit(&row(1)));
        budget.reset();
        assert_eq!(budget.used(), 0);
        assert!(budget.admit(&row(1)));
        assert!(budget.admit(&row(1)));
    }

    #[test]
    fn zero_limit_admits_everything() {
        let mut budget = MemBudget::new(0);
        for _ in 0..1000 {
            assert!(budget.admit(&row(8)), "limit 0 never refuses");
        }
    }
}
