//! [`Sort`]: order all of a child's rows by one or more sort keys.
//!
//! Unlike the streaming operators ([`SeqScan`](super::SeqScan)/[`Filter`](super::Filter)/
//! [`Project`](super::Project)/[`Limit`](super::Limit)), sorting is **buffering**: the
//! first [`next_batch`](super::Operator::next_batch) call drains the whole child, sorts the
//! materialized rows, and hands them back [`BATCH_SIZE`](crate::BATCH_SIZE) at a time.
//!
//! Each [`OrderByKey`] is evaluated once per row (decorate-sort-undecorate) and rows are ordered
//! by the keys left-to-right with the shared null-aware comparator
//! ([`compare_order_key`](crate::executor::eval)), which applies `DESC` and explicit
//! `NULLS FIRST`/`LAST`. The sort is **stable** (Rust's `slice::sort_by` is a timsort
//! variant), so equal-key rows keep their input order. With the default null placement, NULLs sort
//! last for `ASC` and first for `DESC`.

use std::sync::Arc;

use crate::ast;
use crate::batch::convert::{batch_to_rows, rows_to_batch};
use crate::batch::{RecordBatch, Schema};
use crate::error::Error;
use crate::executor::eval::{compare_order_key, eval};
use crate::planner::OrderByKey;
use crate::vectorized::Operator;

/// A multi-key sort over a child [`Operator`]'s batch stream.
#[derive(Debug)]
pub struct Sort {
    child: Box<dyn Operator>,
    keys: Vec<OrderByKey>,
    schema: Arc<Schema>,
    /// Limit-aware top-N: when `Some(m)`, only the first `m` rows in
    /// sort order are emitted downstream, selected via a bounded partial selection instead of a
    /// full sort — result-identical to the full sort's first `m` rows.
    top_n: Option<usize>,
    /// `None` until the child is drained + sorted; then the sorted rows, emitted in chunks.
    sorted: Option<std::vec::IntoIter<crate::Row>>,
}

impl Sort {
    /// Build a sort of `child` ordered by `keys` (applied left-to-right). The output schema
    /// equals the input schema — sorting reorders rows, never columns. `top_n` bounds the output to
    /// the first `m` rows in sort order.
    #[must_use]
    pub fn new(child: Box<dyn Operator>, keys: Vec<OrderByKey>, top_n: Option<usize>) -> Self {
        let schema = Arc::clone(child.schema());
        Self {
            child,
            keys,
            schema,
            top_n,
            sorted: None,
        }
    }

    /// Compare two decorated rows by the sort keys, breaking exact ties by arrival index
    /// (ascending) so the order is total and equals the stable full sort — the tie-break the
    /// partial selection needs to reproduce the full sort's first `m` rows exactly.
    fn cmp_decorated(
        &self,
        a: &(Vec<ast::Value>, usize, crate::Row),
        b: &(Vec<ast::Value>, usize, crate::Row),
    ) -> std::cmp::Ordering {
        for (idx, (av, bv)) in a.0.iter().zip(&b.0).enumerate() {
            let (ascending, nulls) = self
                .keys
                .get(idx)
                .map_or((true, ast::NullOrdering::Default), |k| {
                    (k.ascending, k.nulls)
                });
            let ord = compare_order_key(av, bv, ascending, nulls);
            if !ord.is_eq() {
                return ord;
            }
        }
        a.1.cmp(&b.1)
    }

    /// Drain the child into one buffer and sort it in place. With a `top_n` cap set and smaller
    /// than the row count, selects the `m` smallest rows (O(N) partition + O(m log m)) instead of a
    /// full O(N log N) sort — the result is the full sort's first `m` rows, bit-identical.
    fn buffer_and_sort(&mut self) -> Result<Vec<crate::Row>, Error> {
        let mut rows: Vec<crate::Row> = Vec::new();
        while let Some(batch) = self.child.next_batch()? {
            rows.append(&mut batch_to_rows(&batch));
        }
        // Pre-evaluate each key per row so the comparator itself is infallible; carry the arrival
        // index as the stable tie-break (matches the timsort's equal-key input order).
        let mut decorated: Vec<(Vec<ast::Value>, usize, crate::Row)> =
            Vec::with_capacity(rows.len());
        for (idx, row) in rows.into_iter().enumerate() {
            let key_values = self
                .keys
                .iter()
                .map(|k| eval(&k.expr, &row))
                .collect::<Result<Vec<_>, _>>()?;
            decorated.push((key_values, idx, row));
        }
        match self.top_n {
            // Bounded selection: partition so the `m` smallest are first, then sort just those.
            Some(m) if m < decorated.len() => {
                decorated.select_nth_unstable_by(m, |a, b| self.cmp_decorated(a, b));
                decorated.truncate(m);
                decorated.sort_unstable_by(|a, b| self.cmp_decorated(a, b));
            },
            // No cap, or the cap covers every row: an ordinary full sort. The arrival-index
            // tie-break makes `sort_unstable_by` produce exactly the stable sort's order.
            _ => decorated.sort_unstable_by(|a, b| self.cmp_decorated(a, b)),
        }
        Ok(decorated.into_iter().map(|(_, _, row)| row).collect())
    }
}

impl Operator for Sort {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.sorted.is_none() {
            let sorted = self.buffer_and_sort()?;
            self.sorted = Some(sorted.into_iter());
        }
        // Just ensured `Some`; the `else` is unreachable but keeps the code panic-free.
        let Some(iter) = &mut self.sorted else {
            return Ok(None);
        };
        let chunk: Vec<crate::Row> = iter.by_ref().take(crate::BATCH_SIZE).collect();
        if chunk.is_empty() {
            return Ok(None);
        }
        rows_to_batch(&self.schema, chunk).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::Sort;
    use crate::Field;
    use crate::ast;
    use crate::batch::{Int64Array, Schema, StringArray};
    use crate::executor::row;
    use crate::planner::{OrderByKey, TypedExpr, TypedExprKind};
    use crate::vectorized::{Operator, SeqScan};
    use nusadb_core::engine::{SharedTuple, Tid, TupleScan};
    use nusadb_core::{ColumnType, PageId, Result as CoreResult, SlotIdx};
    use std::sync::Arc;

    struct VecScan {
        tuples: Vec<SharedTuple>,
        pos: usize,
    }

    impl TupleScan for VecScan {
        fn try_next(&mut self) -> CoreResult<Option<(Tid, SharedTuple)>> {
            let item = self.tuples.get(self.pos).map(|t| {
                (
                    Tid {
                        page: PageId(0),
                        slot: SlotIdx(0),
                    },
                    Arc::clone(t),
                )
            });
            if item.is_some() {
                self.pos += 1;
            }
            Ok(item)
        }
    }

    /// A `SeqScan` over `(k INT, tag TEXT)` rows.
    fn scan(rows: &[(Option<i64>, &str)]) -> SeqScan {
        let types = [ColumnType::Int, ColumnType::Text];
        let tuples = rows
            .iter()
            .map(|&(k, tag)| {
                let key = k.map_or(ast::Value::Null, ast::Value::Int);
                let row = [key, ast::Value::Text(tag.to_owned())];
                SharedTuple::from(row::encode(&row, &types).unwrap().as_slice())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", ColumnType::Int, true),
            Field::new("tag", ColumnType::Text, true),
        ]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    fn key(index: usize, ty: ColumnType, ascending: bool) -> OrderByKey {
        OrderByKey {
            expr: TypedExpr {
                kind: TypedExprKind::Column(index),
                ty,
            },
            ascending,
            nulls: crate::ast::NullOrdering::Default,
        }
    }

    /// Collect (k, tag) pairs in emitted order.
    fn rows_of(mut op: Sort) -> Vec<(Option<i64>, String)> {
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            let ks = batch
                .column(0)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let tags = batch
                .column(1)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                out.push((ks.get(i), tags.get(i).unwrap().to_owned()));
            }
        }
        out
    }

    #[test]
    fn sorts_ascending_by_one_key() {
        let op = Sort::new(
            Box::new(scan(&[(Some(3), "c"), (Some(1), "a"), (Some(2), "b")])),
            vec![key(0, ColumnType::Int, true)],
            None,
        );
        let ks: Vec<_> = rows_of(op).into_iter().map(|(k, _)| k).collect();
        assert_eq!(ks, vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn sorts_descending() {
        let op = Sort::new(
            Box::new(scan(&[(Some(1), "a"), (Some(3), "c"), (Some(2), "b")])),
            vec![key(0, ColumnType::Int, false)],
            None,
        );
        let ks: Vec<_> = rows_of(op).into_iter().map(|(k, _)| k).collect();
        assert_eq!(ks, vec![Some(3), Some(2), Some(1)]);
    }

    #[test]
    fn nulls_sort_last_ascending() {
        let op = Sort::new(
            Box::new(scan(&[(None, "x"), (Some(2), "b"), (Some(1), "a")])),
            vec![key(0, ColumnType::Int, true)],
            None,
        );
        let ks: Vec<_> = rows_of(op).into_iter().map(|(k, _)| k).collect();
        assert_eq!(ks, vec![Some(1), Some(2), None]);
    }

    #[test]
    fn second_key_breaks_ties_and_sort_is_stable() {
        // Primary key all equal (k=1); secondary ascending on tag orders the ties.
        let op = Sort::new(
            Box::new(scan(&[(Some(1), "b"), (Some(1), "a"), (Some(1), "c")])),
            vec![
                key(0, ColumnType::Int, true),
                key(1, ColumnType::Text, true),
            ],
            None,
        );
        let tags: Vec<_> = rows_of(op).into_iter().map(|(_, t)| t).collect();
        assert_eq!(tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn empty_input_yields_no_batch() {
        let op = Sort::new(
            Box::new(scan(&[])),
            vec![key(0, ColumnType::Int, true)],
            None,
        );
        assert!(rows_of(op).is_empty());
    }

    #[test]
    fn sorts_across_batch_boundary() {
        // 1.5 batches of descending input → fully sorted ascending, re-chunked at BATCH_SIZE.
        let total = crate::BATCH_SIZE + crate::BATCH_SIZE / 2;
        let input: Vec<(Option<i64>, &str)> = (0..total)
            .rev()
            .map(|i| (Some(i64::try_from(i).unwrap()), "x"))
            .collect();
        let op = Sort::new(
            Box::new(scan(&input)),
            vec![key(0, ColumnType::Int, true)],
            None,
        );
        let ks: Vec<_> = rows_of(op).into_iter().filter_map(|(k, _)| k).collect();
        assert_eq!(ks.len(), total);
        assert_eq!(ks.first(), Some(&0));
        assert_eq!(ks.last(), Some(&i64::try_from(total - 1).unwrap()));
        assert!(ks.windows(2).all(|w| w[0] <= w[1]));
    }

    /// The bounded top-N selection returns exactly the first `m` rows of
    /// the full sort — same rows, same order, including tie stability (equal `k` rows keep input
    /// order, observed via `tag`). Proven differentially against the uncapped sort of the same
    /// input for every `m`.
    #[test]
    fn top_n_selects_first_m_rows_stably() {
        // Heavy key duplication so ties (and their stable ordering) are exercised; `tag` is unique
        // per row so the retained tie order is observable.
        let input: Vec<(Option<i64>, String)> = (0..50)
            .map(|i| (Some(i64::from(i % 5)), format!("t{i:02}")))
            .collect();
        let input_ref: Vec<(Option<i64>, &str)> =
            input.iter().map(|(k, t)| (*k, t.as_str())).collect();
        let keys = || vec![key(0, ColumnType::Int, true)];
        let full = rows_of(Sort::new(Box::new(scan(&input_ref)), keys(), None));
        for m in [0_usize, 1, 7, 20, 50, 80] {
            let bounded = rows_of(Sort::new(Box::new(scan(&input_ref)), keys(), Some(m)));
            let expected: Vec<_> = full.iter().take(m).cloned().collect();
            assert_eq!(
                bounded, expected,
                "top-{m} must equal the full sort's first {m}"
            );
        }
    }
}
