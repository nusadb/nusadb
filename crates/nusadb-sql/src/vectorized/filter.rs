//! [`Filter`]: drop rows of its child's batches that fail a predicate.
//!
//! It pulls a batch from its child, evaluates the `WHERE`-style predicate against each
//! row, and re-materializes only the rows where the predicate is **`TRUE`** (SQL's
//! three-valued logic: `FALSE` and `NULL` both drop the row). The output schema equals the
//! input schema — `Filter` removes rows, never columns.
//!
//! A recognized `column <cmp> literal` predicate over an INT/FLOAT column takes a
//! column-at-a-time SIMD fast path ([`simd`](crate::vectorized::simd)/),
//! including null-containing columns (the mask is cleared at null rows). Every other
//! predicate is evaluated with the shared row evaluator
//! ([`eval`](crate::executor::eval)) over rows read back from the batch
//! (`batch_to_rows`). Empty result batches are skipped: each
//! [`next_batch`](super::Operator::next_batch) returns the next batch that has at least one
//! surviving row, so a parent never sees a zero-row batch mid-stream.

use std::sync::Arc;

use crate::ast;
use crate::batch::convert::{batch_to_rows, filter_batch};
use crate::batch::{Array, Float64Array, Int64Array, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::eval::eval;
use crate::planner::{TypedExpr, TypedExprKind};
use crate::vectorized::Operator;
use crate::vectorized::simd::{self, CmpOp};

/// A recognized `column <cmp> literal` predicate the SIMD kernels can evaluate column-at-a-time.
#[derive(Debug, Clone, Copy)]
enum ColCmp {
    /// `int_column <cmp> int_literal`.
    Int(usize, CmpOp, i64),
    /// `float_column <cmp> float_literal`.
    Float(usize, CmpOp, f64),
}

/// A predicate over batches, shared by the [`Filter`] operator and the parallel grouped
/// aggregate's worker-side filtering — one masking source of truth, so the two paths
/// cannot disagree on which rows survive.
#[derive(Debug)]
pub(super) struct BatchPredicate {
    predicate: TypedExpr,
    /// Pre-recognized SIMD fast path for a `column <cmp> literal` predicate over an INT or
    /// FLOAT column. `None` (or a null-containing column at run time) falls back to the row evaluator.
    fast: Option<ColCmp>,
}

impl BatchPredicate {
    pub(super) fn new(predicate: TypedExpr) -> Self {
        let fast = recognize_col_cmp(&predicate);
        Self { predicate, fast }
    }

    /// The keep-mask for `batch`: `true` where the predicate evaluates `TRUE` (SQL 3VL: `FALSE`
    /// and `NULL` both drop). A recognized `column <cmp> literal` takes the SIMD path; everything
    /// else evaluates row by row with the shared row evaluator. Indexes align with the batch's
    /// rows, so a caller can carry per-row context (the parallel fold's global positions) across
    /// the filtering.
    pub(super) fn mask(&self, batch: &RecordBatch) -> Result<Vec<bool>, Error> {
        if let Some(mask) = self.simd_mask(batch) {
            return Ok(mask);
        }
        let mut mask = Vec::with_capacity(batch.num_rows());
        for row in batch_to_rows(batch) {
            mask.push(matches!(
                eval(&self.predicate, &row)?,
                ast::Value::Bool(true)
            ));
        }
        Ok(mask)
    }

    /// The per-row selection mask from the SIMD fast path, or `None` to fall back to the row
    /// evaluator (no recognized predicate, or a type/ordinal mismatch).
    ///
    /// The kernel compares the dense `values()` slice (null slots hold a placeholder), then
    /// [`drop_nulls`] clears the mask at null rows: `col <cmp> literal` is NULL where `col` is
    /// NULL, and NULL is not TRUE, so a null row is dropped — matching the row evaluator's 3VL.
    fn simd_mask(&self, batch: &RecordBatch) -> Option<Vec<bool>> {
        match self.fast? {
            ColCmp::Int(idx, op, scalar) => {
                let col = batch.column(idx)?.as_any().downcast_ref::<Int64Array>()?;
                let mut mask = simd::filter_i64(col.values(), op, scalar);
                drop_nulls(col, &mut mask);
                Some(mask)
            },
            ColCmp::Float(idx, op, scalar) => {
                let col = batch.column(idx)?.as_any().downcast_ref::<Float64Array>()?;
                let mut mask = simd::filter_f64(col.values(), op, scalar);
                drop_nulls(col, &mut mask);
                Some(mask)
            },
        }
    }
}

/// A predicate filter over a child [`Operator`]'s batch stream.
#[derive(Debug)]
pub struct Filter {
    child: Box<dyn Operator>,
    predicate: BatchPredicate,
    schema: Arc<Schema>,
}

impl Filter {
    /// Build a filter that keeps rows of `child` for which `predicate` evaluates to `TRUE`.
    ///
    /// `predicate`'s [`Column`](crate::planner::TypedExprKind::Column) ordinals index into
    /// `child`'s schema, which is also the filter's output schema.
    #[must_use]
    pub fn new(child: Box<dyn Operator>, predicate: TypedExpr) -> Self {
        let schema = Arc::clone(child.schema());
        Self {
            child,
            predicate: BatchPredicate::new(predicate),
            schema,
        }
    }

    /// Apply the predicate to `batch`, returning the surviving rows as a new batch
    /// ([`BatchPredicate::mask`] + [`filter_batch`]).
    fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch, Error> {
        let mask = self.predicate.mask(batch)?;
        filter_batch(batch, &mask)
    }
}

/// Clear `mask` bits at the column's null rows, so a null compares as "not TRUE" and is dropped.
/// A null-free column (the common case) early-returns after one check, skipping the per-row loop.
fn drop_nulls(col: &dyn Array, mask: &mut [bool]) {
    if col.null_count() == 0 {
        return;
    }
    for (i, keep) in mask.iter_mut().enumerate() {
        if col.is_null(i) {
            *keep = false;
        }
    }
}

/// Recognize `column <cmp> literal` (or the symmetric `literal <cmp> column`) over an INT or FLOAT
/// column, so a SIMD kernel can evaluate the predicate a vector lane at a time.
fn recognize_col_cmp(predicate: &TypedExpr) -> Option<ColCmp> {
    use TypedExprKind::{Column, Literal};
    let TypedExprKind::Binary { left, op, right } = &predicate.kind else {
        return None;
    };
    let cmp = CmpOp::from_binary_op(*op)?;
    match (&left.kind, &right.kind) {
        (Column(i), Literal(ast::Value::Int(v))) => Some(ColCmp::Int(*i, cmp, *v)),
        (Literal(ast::Value::Int(v)), Column(i)) => Some(ColCmp::Int(*i, cmp.swapped(), *v)),
        (Column(i), Literal(ast::Value::Float(v))) => Some(ColCmp::Float(*i, cmp, *v)),
        (Literal(ast::Value::Float(v)), Column(i)) => Some(ColCmp::Float(*i, cmp.swapped(), *v)),
        // A NUMERIC literal (a plain decimal now types as NUMERIC) drives the same FLOAT
        // kernel as a Float literal — keeping the SIMD fast path live for `WHERE f > 0.5`.
        (Column(i), Literal(ast::Value::Numeric(v))) => Some(ColCmp::Float(*i, cmp, v.to_f64())),
        (Literal(ast::Value::Numeric(v)), Column(i)) => {
            Some(ColCmp::Float(*i, cmp.swapped(), v.to_f64()))
        },
        _ => None,
    }
}

impl Operator for Filter {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        loop {
            let Some(batch) = self.child.next_batch()? else {
                return Ok(None);
            };
            let filtered = self.apply(&batch)?;
            if filtered.num_rows() > 0 {
                return Ok(Some(filtered));
            }
            // Whole batch filtered out — pull the next one rather than emit an empty batch.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Filter;
    use crate::Field;
    use crate::ast::{self, BinaryOp};
    use crate::batch::{Array, Float64Array, Int64Array, RecordBatch, Schema};
    use crate::executor::row;
    use crate::planner::{TypedExpr, TypedExprKind};
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

    /// A `SeqScan` over `id INT` rows `0..n`.
    fn id_scan(ids: &[i64]) -> SeqScan {
        let types = [ColumnType::Int];
        let tuples = ids
            .iter()
            .map(|&i| {
                SharedTuple::from(
                    row::encode(&[ast::Value::Int(i)], &types)
                        .unwrap()
                        .as_slice(),
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("id", ColumnType::Int, true)]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    /// A `SeqScan` over a nullable `id INT` column (`None` becomes SQL NULL).
    fn nullable_id_scan(ids: &[Option<i64>]) -> SeqScan {
        let types = [ColumnType::Int];
        let tuples = ids
            .iter()
            .map(|&i| {
                let value = i.map_or(ast::Value::Null, ast::Value::Int);
                SharedTuple::from(row::encode(&[value], &types).unwrap().as_slice())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("id", ColumnType::Int, true)]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    /// Predicate `id > threshold`.
    fn id_gt(threshold: i64) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(TypedExpr {
                    kind: TypedExprKind::Column(0),
                    ty: ColumnType::Int,
                }),
                op: BinaryOp::Gt,
                right: Box::new(TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Int(threshold)),
                    ty: ColumnType::Int,
                }),
            },
            ty: ColumnType::Bool,
        }
    }

    fn ids_of(batches: &[RecordBatch]) -> Vec<i64> {
        let mut out = Vec::new();
        for batch in batches {
            let col = batch
                .column(0)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..col.len() {
                out.push(col.get(i).unwrap());
            }
        }
        out
    }

    /// A `SeqScan` over a single `price FLOAT` column.
    fn price_scan(prices: &[f64]) -> SeqScan {
        let types = [ColumnType::Float];
        let tuples = prices
            .iter()
            .map(|&p| {
                SharedTuple::from(
                    row::encode(&[ast::Value::Float(p)], &types)
                        .unwrap()
                        .as_slice(),
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "price",
            ColumnType::Float,
            true,
        )]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    /// Predicate `price < threshold`.
    fn price_lt(threshold: f64) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(TypedExpr {
                    kind: TypedExprKind::Column(0),
                    ty: ColumnType::Float,
                }),
                op: BinaryOp::Lt,
                right: Box::new(TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Float(threshold)),
                    ty: ColumnType::Float,
                }),
            },
            ty: ColumnType::Bool,
        }
    }

    fn prices_of(batches: &[RecordBatch]) -> Vec<f64> {
        let mut out = Vec::new();
        for batch in batches {
            let col = batch
                .column(0)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..col.len() {
                out.push(col.get(i).unwrap());
            }
        }
        out
    }

    fn drain(mut op: Filter) -> Vec<RecordBatch> {
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            out.push(batch);
        }
        out
    }

    #[test]
    fn keeps_only_matching_rows() {
        let op = Filter::new(Box::new(id_scan(&[1, 2, 3, 4, 5])), id_gt(3));
        assert_eq!(op.schema().len(), 1);
        let batches = drain(op);
        assert_eq!(ids_of(&batches), vec![4, 5]);
    }

    #[test]
    fn float_column_uses_simd_fast_path() {
        // Exercises the FLOAT SIMD kernel via the recognized `price < literal` predicate.
        let op = Filter::new(Box::new(price_scan(&[1.5, 2.5, 3.5, 4.5])), price_lt(3.0));
        let batches = drain(op);
        assert_eq!(prices_of(&batches), vec![1.5, 2.5]);
    }

    #[test]
    fn nullable_column_uses_simd_and_drops_nulls() {
        // The SIMD fast path now handles a null-containing column: rows where `id` is NULL make
        // `id > 2` evaluate to NULL (not TRUE) and are dropped, matching the row evaluator's 3VL.
        let op = Filter::new(
            Box::new(nullable_id_scan(&[Some(1), None, Some(3), None, Some(5)])),
            id_gt(2),
        );
        let batches = drain(op);
        assert_eq!(ids_of(&batches), vec![3, 5]);
    }

    #[test]
    fn empty_when_nothing_matches() {
        let op = Filter::new(Box::new(id_scan(&[1, 2, 3])), id_gt(100));
        assert!(drain(op).is_empty());
    }

    #[test]
    fn null_predicate_drops_row() {
        // `id > NULL` is NULL for every row → all dropped (NULL is not TRUE).
        let predicate = TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(TypedExpr {
                    kind: TypedExprKind::Column(0),
                    ty: ColumnType::Int,
                }),
                op: BinaryOp::Gt,
                right: Box::new(TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Null),
                    ty: ColumnType::Int,
                }),
            },
            ty: ColumnType::Bool,
        };
        let op = Filter::new(Box::new(id_scan(&[1, 2, 3])), predicate);
        assert!(drain(op).is_empty());
    }

    #[test]
    fn skips_fully_filtered_batch_then_finds_match() {
        // Span two batches: first BATCH_SIZE rows are 0, last 3 rows are 9 → only the
        // tail survives `id > 5`, exercising the empty-batch skip in `next_batch`.
        let mut ids = vec![0_i64; crate::BATCH_SIZE];
        ids.extend([9, 9, 9]);
        let op = Filter::new(Box::new(id_scan(&ids)), id_gt(5));
        let batches = drain(op);
        assert_eq!(ids_of(&batches), vec![9, 9, 9]);
    }
}
