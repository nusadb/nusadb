//! [`GroupedAggregate`] (A-PERF.AGG6 / F2c): vectorized hash `GROUP BY` over a child batch
//! stream — the largest gap the F2 design closed, since grouped aggregation previously forced the
//! whole plan back onto the row path.
//!
//! The operator covers bare-column group keys and columnar-foldable calls (the
//! [`columnar_call_shape`] contract: no `FILTER`/second argument/`ORDER BY`, not
//! `GROUPING`/`ARRAY_AGG`/two-argument statistics). Keys and argument values are read straight
//! off the batch columns via [`value_at`] — the same per-element conversion `batch_to_rows` uses
//! — and fold through the row path's own machinery: [`GroupIndex`] (one hash/equality contract
//! for both group-by paths, per the F2c "coordinate with the row-path hash group-by" rule),
//! [`fold_value`] / [`fold_count_star`], and [`finalize_aggregate`]. Same values, same code, same
//! row order ⇒ the output multiset **and** the first-seen emission order are bit-identical to
//! `run_group_aggregate_streamed` on every CPU.
//!
//! Spill parity: when spill-to-disk is configured the row path's bounded-memory **sort-based**
//! group-by is authoritative, so the routing in [`super::try_build`] falls back before
//! this operator is built; like the row path's streamed hash group-by, it holds O(groups) state.

use std::sync::Arc;

use crate::Field;
use crate::ast;
use crate::batch::convert::{rows_to_batch, value_at};
use crate::batch::{RecordBatch, Schema};
use crate::error::Error;
use crate::executor::agg::{GroupIndex, finalize_aggregate, fold_count_star, fold_value};
use crate::executor::row::Row;
use crate::planner::{AggregateCall, TypedExpr, TypedExprKind};
use crate::vectorized::Operator;
use crate::vectorized::aggregate::{ColumnarShape, columnar_call_shape};

/// A hash `GROUP BY` aggregate over a child [`Operator`]'s batch stream: one output row per
/// distinct key tuple (key columns first, one column per call after), groups emitted in
/// first-seen order.
#[derive(Debug)]
pub struct GroupedAggregate {
    child: Box<dyn Operator>,
    /// The batch column each group key reads (keys are bare columns by construction).
    key_indices: Vec<usize>,
    calls: Vec<AggregateCall>,
    /// How each call folds: `COUNT(*)` or a single-value fold over its argument column.
    arg_shapes: Vec<ColumnarShape>,
    schema: Arc<Schema>,
    /// Finalized output rows, produced on the first pull; drained in `BATCH_SIZE` chunks.
    out: Option<std::vec::IntoIter<Row>>,
}

impl GroupedAggregate {
    /// Build the operator, or `None` when the plan shape is outside the vectorized grouped fold
    /// (a computed key, or a call [`columnar_call_shape`] rejects) — the caller then falls back
    /// to the row path.
    #[must_use]
    pub(super) fn try_new(
        child: Box<dyn Operator>,
        group_keys: &[TypedExpr],
        calls: Vec<AggregateCall>,
    ) -> Option<Self> {
        let key_indices = group_keys
            .iter()
            .map(|k| match k.kind {
                TypedExprKind::Column(idx) => Some(idx),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let arg_shapes = calls
            .iter()
            .map(columnar_call_shape)
            .collect::<Option<Vec<_>>>()?;
        // Output layout mirrors the row path's group-aggregate rows: key values, then one
        // finalized value per call.
        let fields = group_keys
            .iter()
            .enumerate()
            .map(|(i, k)| Field::new(format!("k{i}"), k.ty, true))
            .chain(
                calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Field::new(format!("agg{i}"), c.result_ty, true)),
            )
            .collect();
        Some(Self {
            child,
            key_indices,
            calls,
            arg_shapes,
            schema: Arc::new(Schema::new(fields)),
            out: None,
        })
    }

    /// Drain the child stream into per-group accumulators and finalize every group, in first-seen
    /// order — the columnar counterpart of `run_group_aggregate_streamed`'s loop.
    fn fold_child(&mut self) -> Result<Vec<Row>, Error> {
        let mut groups = GroupIndex::new();
        // Only a `STRING_AGG … ORDER BY`'s sort keys would read the row, and `columnar_call_shape`
        // excludes `ORDER BY`, so an empty row satisfies `fold_value`'s contract.
        let empty_row: Row = Vec::new();
        while let Some(batch) = self.child.next_batch()? {
            // Resolve the key/argument columns once per batch. A missing column mirrors the row
            // path's malformed-tuple error for an out-of-range column reference.
            let key_cols = self
                .key_indices
                .iter()
                .map(|&c| batch.column(c).ok_or(Error::MalformedTuple { offset: c }))
                .collect::<Result<Vec<_>, _>>()?;
            let arg_cols = self
                .arg_shapes
                .iter()
                .map(|shape| match shape {
                    ColumnarShape::CountStar => Ok(None),
                    ColumnarShape::Column(c) => batch
                        .column(*c)
                        .map(Some)
                        .ok_or(Error::MalformedTuple { offset: *c }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            for i in 0..batch.num_rows() {
                let key: Vec<ast::Value> = key_cols
                    .iter()
                    .map(|col| value_at(col.as_ref(), i))
                    .collect();
                let at = groups.find_or_create(key, self.calls.len());
                let Some(accs) = groups.accs_at(at) else {
                    continue;
                };
                for ((acc, call), col) in accs.iter_mut().zip(&self.calls).zip(&arg_cols) {
                    match col {
                        // COUNT(*): every row counts, NULLs included.
                        None => fold_count_star(acc),
                        Some(col) => {
                            fold_value(acc, call, value_at(col.as_ref(), i), &empty_row)?;
                        },
                    }
                }
            }
        }
        let states = groups.into_states();
        let mut out = Vec::with_capacity(states.len());
        for (key, accs) in states {
            let mut row = key;
            for (acc, call) in accs.into_iter().zip(&self.calls) {
                row.push(finalize_aggregate(acc, call)?);
            }
            out.push(row);
        }
        Ok(out)
    }
}

impl Operator for GroupedAggregate {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.out.is_none() {
            self.out = Some(self.fold_child()?.into_iter());
        }
        let Some(iter) = self.out.as_mut() else {
            return Ok(None);
        };
        let rows: Vec<Row> = iter.by_ref().take(crate::BATCH_SIZE).collect();
        if rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(rows_to_batch(&self.schema, rows)?))
    }
}

#[cfg(test)]
mod tests {
    use super::GroupedAggregate;
    use crate::ast;
    use crate::executor::row;
    use crate::planner::AggregateCall;
    use crate::vectorized::{Operator, SeqScan};
    use crate::{Field, Schema};
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

    /// A `SeqScan` over two nullable columns `(k, v)` of the given types.
    fn two_col_scan(rows: &[(ast::Value, ast::Value)], types: [ColumnType; 2]) -> SeqScan {
        let tuples = rows
            .iter()
            .map(|(k, v)| {
                SharedTuple::from(
                    row::encode(&[k.clone(), v.clone()], &types)
                        .unwrap()
                        .as_slice(),
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", types[0], true),
            Field::new("v", types[1], true),
        ]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    /// An `AggregateCall` whose argument is column 1 (`v`), or argument-less for `COUNT(*)`.
    fn call(
        func: ast::AggregateFunc,
        arg_ty: Option<ColumnType>,
        result_ty: ColumnType,
    ) -> AggregateCall {
        AggregateCall {
            func,
            arg: arg_ty.map(|ty| crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(1),
                ty,
            }),
            result_ty,
            distinct: false,
            fraction: None,
            ordered_set_descending: false,
            filter: None,
            separator: None,
            arg2: None,
            order_by: Vec::new(),
            grouping_args: Vec::new(),
        }
    }

    /// The group key: column 0 (`k`).
    fn key(ty: ColumnType) -> crate::planner::TypedExpr {
        crate::planner::TypedExpr {
            kind: crate::planner::TypedExprKind::Column(0),
            ty,
        }
    }

    fn run(
        scan: SeqScan,
        keys: &[crate::planner::TypedExpr],
        calls: Vec<AggregateCall>,
    ) -> Vec<row::Row> {
        let mut op =
            GroupedAggregate::try_new(Box::new(scan), keys, calls).expect("supported shape");
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            out.extend(crate::batch::convert::batch_to_rows(&batch));
        }
        out
    }

    /// Groups fold across rows in first-seen order; NULL keys form one group; COUNT(*) counts
    /// NULL values, the value folds skip them; DISTINCT dedups within the group.
    #[test]
    fn groups_fold_first_seen_with_null_keys_and_distinct() {
        use ast::AggregateFunc::{Count, Max, Sum};
        use ast::Value as V;
        let rows = vec![
            (V::Int(2), V::Int(10)),
            (V::Int(1), V::Int(5)),
            (V::Null, V::Int(7)),
            (V::Int(2), V::Int(10)), // duplicate value for the DISTINCT call
            (V::Int(2), V::Null),    // NULL value: counted by COUNT(*), skipped by SUM/MAX
            (V::Null, V::Int(3)),
        ];
        let calls = vec![
            call(Count, None, ColumnType::Int), // COUNT(*)
            call(Sum, Some(ColumnType::Int), ColumnType::Int),
            call(Max, Some(ColumnType::Int), ColumnType::Int),
            AggregateCall {
                distinct: true,
                ..call(Count, Some(ColumnType::Int), ColumnType::Int)
            },
        ];
        let got = run(
            two_col_scan(&rows, [ColumnType::Int, ColumnType::Int]),
            &[key(ColumnType::Int)],
            calls,
        );
        assert_eq!(
            got,
            vec![
                // First-seen order: 2, 1, NULL.
                vec![V::Int(2), V::Int(3), V::Int(20), V::Int(10), V::Int(1)],
                vec![V::Int(1), V::Int(1), V::Int(5), V::Int(5), V::Int(1)],
                vec![V::Null, V::Int(2), V::Int(10), V::Int(7), V::Int(2)],
            ]
        );
    }

    /// Group state carries across batch boundaries (> `BATCH_SIZE` input rows), and the exact-i128
    /// integer SUM contract holds through the grouped fold.
    #[test]
    fn groups_merge_across_batches() {
        use ast::AggregateFunc::{Count, Sum};
        use ast::Value as V;
        let n = i64::try_from(crate::BATCH_SIZE).unwrap() * 2 + 500;
        let rows: Vec<(V, V)> = (0..n).map(|i| (V::Int(i % 5), V::Int(i))).collect();
        let got = run(
            two_col_scan(&rows, [ColumnType::Int, ColumnType::Int]),
            &[key(ColumnType::Int)],
            vec![
                call(Count, None, ColumnType::Int),
                call(Sum, Some(ColumnType::Int), ColumnType::Int),
            ],
        );
        assert_eq!(got.len(), 5);
        // First-seen order is 0,1,2,3,4; per-group count and exact sum.
        for (g, row) in got.iter().enumerate() {
            let g = i64::try_from(g).unwrap();
            let count = (n - g + 4) / 5;
            let sum: i64 = (0..n).filter(|i| i % 5 == g).sum();
            assert_eq!(
                row,
                &vec![V::Int(g), V::Int(count), V::Int(sum)],
                "group {g}"
            );
        }
    }

    /// More groups than `BATCH_SIZE` drain over multiple output batches, preserving first-seen
    /// order end to end.
    #[test]
    fn many_groups_chunk_into_multiple_output_batches() {
        use ast::AggregateFunc::Count;
        use ast::Value as V;
        let n = i64::try_from(crate::BATCH_SIZE).unwrap() + 300;
        let rows: Vec<(V, V)> = (0..n).map(|i| (V::Int(i), V::Int(i))).collect();
        let scan = two_col_scan(&rows, [ColumnType::Int, ColumnType::Int]);
        let mut op = GroupedAggregate::try_new(
            Box::new(scan),
            &[key(ColumnType::Int)],
            vec![call(Count, None, ColumnType::Int)],
        )
        .expect("supported shape");
        let first = op.next_batch().unwrap().expect("first chunk");
        assert_eq!(first.num_rows(), crate::BATCH_SIZE);
        let second = op.next_batch().unwrap().expect("second chunk");
        assert_eq!(second.num_rows(), 300);
        assert!(op.next_batch().unwrap().is_none());
        // First-seen order spans the chunk boundary.
        let rows0 = crate::batch::convert::batch_to_rows(&first);
        assert_eq!(rows0[0], vec![V::Int(0), V::Int(1)]);
        let rows1 = crate::batch::convert::batch_to_rows(&second);
        assert_eq!(
            rows1[0],
            vec![V::Int(i64::try_from(crate::BATCH_SIZE).unwrap()), V::Int(1)]
        );
    }

    /// GROUP BY over an empty input yields no rows (unlike the no-GROUP-BY scalar aggregate).
    #[test]
    fn empty_input_yields_no_groups() {
        use ast::AggregateFunc::Count;
        let got = run(
            two_col_scan(&[], [ColumnType::Int, ColumnType::Int]),
            &[key(ColumnType::Int)],
            vec![call(Count, None, ColumnType::Int)],
        );
        assert!(got.is_empty());
    }

    /// Shapes the columnar grouped fold cannot fold identically are refused at build time (the
    /// planner then keeps the row path): a computed key, a `FILTER`ed call, `ARRAY_AGG`.
    #[test]
    fn unsupported_shapes_are_refused() {
        use ast::AggregateFunc::{ArrayAgg, Count};
        let computed_key = crate::planner::TypedExpr {
            kind: crate::planner::TypedExprKind::Literal(ast::Value::Int(1)),
            ty: ColumnType::Int,
        };
        let scan = || two_col_scan(&[], [ColumnType::Int, ColumnType::Int]);
        assert!(
            GroupedAggregate::try_new(
                Box::new(scan()),
                &[computed_key],
                vec![call(Count, None, ColumnType::Int)]
            )
            .is_none()
        );
        let filtered = AggregateCall {
            filter: Some(key(ColumnType::Bool)),
            ..call(Count, None, ColumnType::Int)
        };
        assert!(
            GroupedAggregate::try_new(Box::new(scan()), &[key(ColumnType::Int)], vec![filtered])
                .is_none()
        );
        let array_agg = call(
            ArrayAgg,
            Some(ColumnType::Int),
            ColumnType::Array(nusadb_core::engine::ArrayElem::Int),
        );
        assert!(
            GroupedAggregate::try_new(Box::new(scan()), &[key(ColumnType::Int)], vec![array_agg])
                .is_none()
        );
    }
}
