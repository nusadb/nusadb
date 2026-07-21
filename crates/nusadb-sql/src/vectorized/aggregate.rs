//! [`ScalarAggregate`]: fold a child's batch stream into one result row, one
//! column per [`AggregateCall`], with **no `GROUP BY`**.
//!
//! `MIN`/`MAX` over a bare `INT` or integer-backed temporal column and `SUM`/`AVG` over a bare
//! integer column (no `DISTINCT`/`FILTER`) are reduced column-at-a-time by the SIMD kernels
//! ([`simd`](crate::vectorized::simd)); `COUNT` is a null-skipping tally. Any other call whose
//! argument is a bare column (`FLOAT`/`NUMERIC` SUM/AVG/MIN/MAX, `DISTINCT`, ordered-set, …)
//! folds **directly over the column** through the row path's own accumulate/finalize code
//! (A-PERF.AGG5b) — sequential, so float rounding is bit-identical — without materializing a
//! `Vec<Row>`. Only a call that truly needs rows (computed argument, `FILTER`, second argument,
//! `ORDER BY`, `GROUPING`, `ARRAY_AGG`) falls back to the shared row evaluator
//! ([`fold_aggregates`](crate::executor::agg::fold_aggregates)) over materialized rows.
//!
//! The result is therefore **identical to the row path on every CPU**, AVX2 or not: only the
//! bit-exact, order-independent reductions (`COUNT`, integer/temporal `MIN`/`MAX`, exact-`i128`
//! integer `SUM`/`AVG`) take the SIMD path. `FLOAT` `SUM` and `FLOAT` `MIN`/`MAX` deliberately stay
//! scalar — a SIMD `f64` SUM reorders the (non-associative) adds and `_mm256_min_pd` mishandles
//! `NaN`, either of which would make a query's result depend on the host's instruction set.
//! Bit-exact batch=row results are a correctness/determinism invariant for the engine
//! (the design, 2026-06-14).

use std::sync::Arc;

use crate::Field;
use crate::ast;
use crate::batch::convert::{batch_to_rows, rows_to_batch, value_at};
use crate::batch::{
    Array, DateKind, Int64Array, RecordBatch, Schema, TemporalArray, TemporalKind, TimeKind,
    TimeTzKind, TimestampKind, TimestampTzKind,
};
use crate::error::Error;
use crate::executor::agg::{
    Acc, finalize_aggregate, fold_aggregates, fold_value, is_integer, numeric_overflow,
};
use crate::executor::row::Row;
use crate::planner::{AggregateCall, TypedExprKind};
use crate::vectorized::{Operator, simd};
use nusadb_core::ColumnType;

/// A no-`GROUP BY` aggregate over a child [`Operator`]'s batch stream, emitting exactly one row.
#[derive(Debug)]
pub struct ScalarAggregate {
    child: Box<dyn Operator>,
    calls: Vec<AggregateCall>,
    schema: Arc<Schema>,
    /// Set once the single result row has been emitted; the next pull returns end-of-stream.
    done: bool,
}

impl ScalarAggregate {
    /// Build a scalar aggregate over `child` computing each of `calls`.
    #[must_use]
    pub fn new(child: Box<dyn Operator>, calls: Vec<AggregateCall>) -> Self {
        let fields = calls
            .iter()
            .enumerate()
            .map(|(i, c)| Field::new(format!("agg{i}"), c.result_ty, true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        Self {
            child,
            calls,
            schema,
            done: false,
        }
    }
}

/// Non-null `i64` values of column `idx` across `batches`, or `None` if it is not an `Int64Array`.
fn int_values(batches: &[RecordBatch], idx: usize) -> Option<Vec<i64>> {
    let mut out = Vec::new();
    for batch in batches {
        let col = batch.column(idx)?.as_any().downcast_ref::<Int64Array>()?;
        if col.null_count() == 0 {
            out.extend_from_slice(col.values());
        } else {
            for i in 0..col.len() {
                if let Some(v) = col.get(i) {
                    out.push(v);
                }
            }
        }
    }
    Some(out)
}

/// The non-null values of column `idx` as `i64`s when the column is one of the integer-backed
/// temporal types, plus the constructor back to the column's value variant (A-PERF.AGG5c / F2d).
/// `Date` widens from its `i32` day count (order-preserving, lossless); `Time`/`Timestamp`/
/// `TimestampTz` are microsecond counts; `TimeTz`'s packed form orders exactly like the timetz
/// comparison, so an `i64` MIN/MAX picks the correct element. `None` for any other column type.
/// The widened values of an integer-backed temporal column plus the constructor back to its
/// value variant (see [`temporal_values`]).
type TemporalColumn = (Vec<i64>, fn(i64) -> ast::Value);

fn temporal_values(batches: &[RecordBatch], idx: usize, ty: ColumnType) -> Option<TemporalColumn> {
    fn gather<K: TemporalKind>(
        batches: &[RecordBatch],
        idx: usize,
        widen: fn(K::Native) -> i64,
    ) -> Option<Vec<i64>> {
        let mut out = Vec::new();
        for batch in batches {
            let col = batch
                .column(idx)?
                .as_any()
                .downcast_ref::<TemporalArray<K>>()?;
            for i in 0..col.len() {
                if let Some(v) = col.get(i) {
                    out.push(widen(v));
                }
            }
        }
        Some(out)
    }
    // The widened Date round-trips: every value came from an `i32`, so the narrowing always
    // succeeds (`map_or` keeps the no-panic contract anyway).
    match ty.physical() {
        ColumnType::Date => Some((gather::<DateKind>(batches, idx, i64::from)?, |v| {
            i32::try_from(v).map_or(ast::Value::Null, ast::Value::Date)
        })),
        ColumnType::Time => Some((gather::<TimeKind>(batches, idx, |v| v)?, ast::Value::Time)),
        ColumnType::TimeTz => Some((
            gather::<TimeTzKind>(batches, idx, |v| v)?,
            ast::Value::TimeTz,
        )),
        ColumnType::Timestamp => Some((
            gather::<TimestampKind>(batches, idx, |v| v)?,
            ast::Value::Timestamp,
        )),
        ColumnType::TimestampTz => Some((
            gather::<TimestampTzKind>(batches, idx, |v| v)?,
            ast::Value::TimestampTz,
        )),
        _ => None,
    }
}

/// The count of non-null values of column `idx` across `batches` (for `COUNT(col)`).
fn nonnull_count(batches: &[RecordBatch], idx: usize) -> Option<i64> {
    let mut total: i64 = 0;
    for batch in batches {
        let col = batch.column(idx)?;
        let n = batch.num_rows().saturating_sub(col.null_count());
        total = total.saturating_add(i64::try_from(n).unwrap_or(i64::MAX));
    }
    Some(total)
}

/// Compute one aggregate call via the SIMD kernels, or `None` if the call is not SIMD-eligible (the
/// caller then folds it on the row path). A returned `Some(Ok(_))` is the finished value; a
/// `Some(Err(_))` is a real failure (integer `SUM` overflow) that the row path would raise too.
fn simd_aggregate(
    call: &AggregateCall,
    batches: &[RecordBatch],
) -> Option<Result<ast::Value, Error>> {
    use ast::AggregateFunc as F;
    // DISTINCT / FILTER / ordered-set fraction are not handled here.
    if call.distinct || call.filter.is_some() || call.fraction.is_some() {
        return None;
    }
    // COUNT(*) — every row, NULLs included.
    if matches!(call.func, F::Count) && call.arg.is_none() {
        let n: usize = batches.iter().map(RecordBatch::num_rows).sum();
        return Some(Ok(ast::Value::Int(i64::try_from(n).unwrap_or(i64::MAX))));
    }
    // Every other eligible call's argument is a bare column reference.
    let arg = call.arg.as_ref()?;
    let TypedExprKind::Column(idx) = arg.kind else {
        return None;
    };
    match call.func {
        F::Count => Some(Ok(ast::Value::Int(nonnull_count(batches, idx)?))),
        // MIN/MAX over an INT or integer-backed temporal column reduce bit-exactly via SIMD
        // (order-independent, no overflow; A-PERF.AGG5c widened the temporal family onto the same
        // i64 kernels). A FLOAT/NUMERIC column returns `None` → the row path, whose total-order
        // compare handles NaN exactly and which the SIMD float min/max would not match.
        F::Min | F::Max => {
            let (vals, wrap): TemporalColumn = match int_values(batches, idx) {
                Some(vals) => (vals, ast::Value::Int),
                None => temporal_values(batches, idx, arg.ty)?,
            };
            let m = if matches!(call.func, F::Min) {
                simd::min_i64(&vals)
            } else {
                simd::max_i64(&vals)
            };
            Some(Ok(m.map_or(ast::Value::Null, wrap)))
        },
        // SUM over an integer column with an integer result (A-PERF.AGG5a / F2a): integer addition
        // is associative, so the SIMD block reduction equals the row path's sequential `i128`
        // accumulator bit-for-bit, and the finalize is the row path's exact contract — `i64` on
        // success, the same overflow error otherwise. A NUMERIC-typed result (decimal
        // accumulator) or non-integer argument falls through to the row path.
        F::Sum if is_integer(arg.ty) && is_integer(call.result_ty) => {
            let vals = int_values(batches, idx)?;
            if vals.is_empty() {
                return Some(Ok(ast::Value::Null)); // no non-NULL input → SQL NULL
            }
            Some(
                i64::try_from(simd::sum_i128(&vals))
                    .map(ast::Value::Int)
                    .map_err(|_| numeric_overflow()),
            )
        },
        // AVG over an integer column with the NUMERIC result the analyzer assigns it: the exact
        // `i128` sum divides by the non-NULL count through the same Decimal division as the row
        // path's finalize (G22 — exact past 2^53). Any other typing falls through to the row path.
        F::Avg if is_integer(arg.ty) && matches!(call.result_ty, ColumnType::Numeric { .. }) => {
            let vals = int_values(batches, idx)?;
            if vals.is_empty() {
                return Some(Ok(ast::Value::Null)); // no non-NULL input → SQL NULL
            }
            let sum = crate::numeric::Decimal::from_i128(simd::sum_i128(&vals));
            let count = crate::numeric::Decimal::from_i64(i64::try_from(vals.len()).ok()?);
            Some(
                sum.checked_div(&count)
                    .map(ast::Value::Numeric)
                    .ok_or_else(numeric_overflow),
            )
        },
        // Everything else (float SUM/AVG/MIN/MAX, NUMERIC arguments, …) stays on the exact
        // scalar/row path: SUM(FLOAT) keeps its sequential rounding, so results are identical on
        // every CPU (the batch=row determinism invariant).
        _ => None,
    }
}

/// A columnar-foldable call shape (see [`columnar_call_shape`]).
#[derive(Clone, Copy, Debug)]
pub(super) enum ColumnarShape {
    /// Argument-less `COUNT(*)`: every row counts
    /// ([`crate::executor::agg::fold_count_star`]).
    CountStar,
    /// A single-value [`fold_value`] over this column.
    Column(usize),
}

/// How `call` can fold columnarly, or `None` when it truly needs row evaluation — a computed
/// argument, `FILTER` (row predicate), a second argument, `ORDER BY` sort keys, `GROUPING` (folds
/// no values), or `ARRAY_AGG` (keeps NULLs; folds outside the single-value path). Shared by the
/// scalar [`columnar_fold`] and the grouped operator
/// ([`super::GroupedAggregate`], A-PERF.AGG6) so both admit exactly the same call shapes.
pub(super) fn columnar_call_shape(call: &AggregateCall) -> Option<ColumnarShape> {
    use ast::AggregateFunc as F;
    if call.filter.is_some() || call.arg2.is_some() || !call.order_by.is_empty() {
        return None;
    }
    if matches!(call.func, F::Grouping | F::ArrayAgg) || call.func.is_two_arg() {
        return None;
    }
    let Some(arg) = call.arg.as_ref() else {
        return matches!(call.func, F::Count).then_some(ColumnarShape::CountStar);
    };
    match arg.kind {
        TypedExprKind::Column(idx) => Some(ColumnarShape::Column(idx)),
        _ => None,
    }
}

/// Fold one non-SIMD-eligible call **directly over its argument column** (A-PERF.AGG5b / F2b), or
/// `None` when [`columnar_call_shape`] says the call needs row evaluation. The column's values
/// feed [`fold_value`] + [`finalize_aggregate`] — the very code the row path runs, in the same
/// row order — so the result is bit-identical to `batch_to_rows` + `fold_aggregates` (float SUM's
/// sequential rounding included) without ever allocating the `Vec<Row>`. (An argument-less
/// `COUNT(*)` never reaches here — the SIMD tally covers it — so only the column shape folds.)
fn columnar_fold(
    call: &AggregateCall,
    batches: &[RecordBatch],
) -> Option<Result<ast::Value, Error>> {
    let ColumnarShape::Column(idx) = columnar_call_shape(call)? else {
        return None;
    };
    // Only STRING_AGG's ORDER BY sort keys read the row, and ORDER BY is excluded above, so an
    // empty row satisfies `fold_value`'s contract.
    let empty_row: Row = Vec::new();
    let mut acc = Acc::default();
    for batch in batches {
        let col = batch.column(idx)?;
        for i in 0..batch.num_rows() {
            if let Err(e) = fold_value(&mut acc, call, value_at(col.as_ref(), i), &empty_row) {
                return Some(Err(e));
            }
        }
    }
    Some(finalize_aggregate(acc, call))
}

impl Operator for ScalarAggregate {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Buffer the child's output columnarly (kept for the SIMD + columnar-fold paths); rows are
        // materialized only for a call that truly needs row evaluation (computed argument, FILTER,
        // second argument, ORDER BY, GROUPING, ARRAY_AGG — see `columnar_fold`).
        let mut batches = Vec::new();
        while let Some(batch) = self.child.next_batch()? {
            batches.push(batch);
        }
        let mut rows: Option<Vec<Row>> = None;

        let mut result: Vec<ast::Value> = Vec::with_capacity(self.calls.len());
        for call in &self.calls {
            let value = if let Some(value) = simd_aggregate(call, &batches) {
                value?
            } else if let Some(value) = columnar_fold(call, &batches) {
                // Not SIMD-eligible but a bare-column argument: fold the column directly through
                // the row path's own accumulate/finalize code (A-PERF.AGG5b) — bit-identical,
                // no `Vec<Row>`.
                value?
            } else {
                // Fall back to the row evaluator for a call that needs real rows.
                let rows = rows.get_or_insert_with(|| {
                    batches.iter().flat_map(batch_to_rows).collect::<Vec<_>>()
                });
                let folded = fold_aggregates(std::slice::from_ref(call), rows.iter())?;
                folded.into_iter().next().ok_or_else(|| {
                    Error::Unsupported("internal: empty scalar-aggregate result".to_owned())
                })?
            };
            result.push(value);
        }
        Ok(Some(rows_to_batch(&self.schema, vec![result])?))
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarAggregate;
    use crate::ast;
    use crate::batch::{Float64Array, Int64Array};
    use crate::executor::row;
    use crate::numeric::Decimal;
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

    /// A `SeqScan` over a single nullable `INT` column.
    fn int_scan(values: &[Option<i64>]) -> SeqScan {
        let types = [ColumnType::Int];
        let tuples = values
            .iter()
            .map(|&v| {
                let value = v.map_or(ast::Value::Null, ast::Value::Int);
                SharedTuple::from(row::encode(&[value], &types).unwrap().as_slice())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("v", ColumnType::Int, true)]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    fn sum_call() -> AggregateCall {
        AggregateCall {
            func: ast::AggregateFunc::Sum,
            arg: Some(crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
                ty: ColumnType::Int,
            }),
            result_ty: ColumnType::Int,
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

    fn call(func: ast::AggregateFunc, result_ty: ColumnType) -> AggregateCall {
        AggregateCall {
            func,
            arg: Some(crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
                ty: ColumnType::Int,
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

    /// A `SeqScan` over a single nullable column of `ty`, from already-typed values.
    fn typed_scan(values: &[ast::Value], ty: ColumnType) -> SeqScan {
        let types = [ty];
        let tuples = values
            .iter()
            .map(|v| {
                SharedTuple::from(
                    row::encode(std::slice::from_ref(v), &types)
                        .unwrap()
                        .as_slice(),
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("v", ty, true)]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    /// A MIN/MAX call whose argument column is `ty`.
    fn minmax_call(func: ast::AggregateFunc, ty: ColumnType) -> AggregateCall {
        AggregateCall {
            func,
            arg: Some(crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
                ty,
            }),
            result_ty: ty,
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

    /// A-PERF.AGG5c / F2d: MIN/MAX over the integer-backed temporal columns takes the SIMD i64
    /// kernels and matches the row-path fold exactly (the scalar oracle), NULLs skipped, empty →
    /// NULL. TIMETZ exercises the packed representation, whose i64 order is the timetz order.
    #[test]
    fn temporal_min_max_matches_the_row_path_oracle() {
        use ast::Value as V;
        let cases: Vec<(ColumnType, Vec<V>)> = vec![
            (
                ColumnType::Date,
                vec![V::Date(19_000), V::Null, V::Date(-3), V::Date(7)],
            ),
            (
                ColumnType::Timestamp,
                vec![V::Timestamp(5), V::Timestamp(-9), V::Null],
            ),
            (
                ColumnType::TimestampTz,
                vec![V::TimestampTz(100), V::TimestampTz(2)],
            ),
            (ColumnType::Time, vec![V::Time(3_600), V::Time(60)]),
            (
                ColumnType::TimeTz,
                vec![
                    V::TimeTz(crate::temporal::parse_timetz("13:45:30+07").unwrap()),
                    V::TimeTz(crate::temporal::parse_timetz("06:45:30+00").unwrap()),
                    V::Null,
                ],
            ),
            (ColumnType::Date, vec![V::Null, V::Null]), // all-NULL → NULL
        ];
        for (ty, values) in cases {
            for func in [ast::AggregateFunc::Min, ast::AggregateFunc::Max] {
                let got = run(typed_scan(&values, ty), vec![minmax_call(func, ty)]);
                let rows: Vec<Vec<ast::Value>> = values.iter().map(|v| vec![v.clone()]).collect();
                let expected =
                    crate::executor::agg::fold_aggregates(&[minmax_call(func, ty)], rows.iter())
                        .unwrap();
                assert_eq!(got, expected, "{func:?} over {ty:?} {values:?}");
            }
        }
    }

    /// A-PERF.AGG5a / F2a: SUM (integer result) and AVG (NUMERIC result) over an integer column
    /// take the exact-i128 SIMD kernel and match the row-path fold bit-for-bit — including values
    /// past 2^53, where an f64-accumulated sum would silently round (G22).
    #[test]
    fn sum_avg_int_matches_row_path_oracle() {
        use ast::AggregateFunc::{Avg, Sum};
        use ast::Value as V;
        let numeric = ColumnType::Numeric {
            precision: 0,
            scale: 0,
        };
        let cases: Vec<Vec<Option<i64>>> = vec![
            vec![Some(3), Some(1), None, Some(5), Some(2)],
            // Exactness past 2^53: 2^53 + 1 + 1 — an f64 accumulator would lose the +2.
            vec![Some(9_007_199_254_740_992), Some(1), Some(1)],
            vec![Some(i64::MAX), Some(i64::MIN), Some(-7)],
            vec![Some(-1); 7],
            vec![None, None], // all-NULL → NULL
            vec![],           // empty → NULL
        ];
        for values in cases {
            for call in [call(Sum, ColumnType::Int), call(Avg, numeric)] {
                let got = run(int_scan(&values), vec![call.clone()]);
                let rows: Vec<row::Row> = values
                    .iter()
                    .map(|&v| vec![v.map_or(V::Null, V::Int)])
                    .collect();
                let expected =
                    crate::executor::agg::fold_aggregates(std::slice::from_ref(&call), rows.iter())
                        .unwrap();
                assert_eq!(got, expected, "{:?} over {values:?}", call.func);
            }
        }
    }

    /// A-PERF.AGG5b / F2b: every bare-column call the SIMD kernels do not cover folds directly
    /// over the column and must equal the row-path fold exactly — float adversaries (NaN, ±0.0,
    /// ±inf: sequential rounding + total-order MIN/MAX), exact NUMERIC past 2^53, DISTINCT dedup,
    /// TEXT MIN/MAX, and an ordered-set percentile. Results compare via `Debug` (Value's
    /// `PartialEq` says `NaN != NaN`, but both sides must *produce* the same NaN).
    #[test]
    fn columnar_fold_matches_the_row_path_oracle() {
        use ast::AggregateFunc as F;
        use ast::Value as V;
        let numeric = ColumnType::Numeric {
            precision: 0,
            scale: 0,
        };
        let floats = vec![
            V::Float(1.5),
            V::Float(f64::NAN),
            V::Float(-0.0),
            V::Float(0.0),
            V::Float(f64::INFINITY),
            V::Float(f64::NEG_INFINITY),
            V::Null,
            V::Float(2.5e-10),
        ];
        let numerics = vec![
            V::Numeric(Decimal::parse("9007199254740993").unwrap()),
            V::Numeric(Decimal::parse("7").unwrap()),
            V::Numeric(Decimal::parse("-2").unwrap()),
            V::Null,
        ];
        let texts = vec![
            V::Text("b".to_owned()),
            V::Text("a".to_owned()),
            V::Null,
            V::Text("c".to_owned()),
        ];
        let distinct_floats = vec![
            V::Float(1.0),
            V::Float(1.0),
            V::Float(f64::NAN),
            V::Float(f64::NAN),
            V::Float(-0.0),
            V::Float(0.0),
            V::Null,
        ];
        let mut cases: Vec<(ColumnType, Vec<V>, AggregateCall)> = Vec::new();
        for func in [F::Sum, F::Avg, F::Min, F::Max] {
            cases.push((ColumnType::Float, floats.clone(), {
                minmax_call(func, ColumnType::Float)
            }));
            cases.push((numeric, numerics.clone(), minmax_call(func, numeric)));
        }
        for func in [F::Min, F::Max] {
            cases.push((ColumnType::Text, texts.clone(), {
                minmax_call(func, ColumnType::Text)
            }));
        }
        // COUNT(DISTINCT float): all NaN are one value, -0.0 and 0.0 are one value.
        cases.push((
            ColumnType::Float,
            distinct_floats,
            AggregateCall {
                distinct: true,
                result_ty: ColumnType::Int,
                ..minmax_call(F::Count, ColumnType::Float)
            },
        ));
        // Ordered-set percentile over a float column (fraction, not FILTER, so columnar-eligible).
        cases.push((
            ColumnType::Float,
            vec![V::Float(3.0), V::Float(1.0), V::Float(2.0)],
            AggregateCall {
                fraction: Some(0.5),
                ..minmax_call(F::PercentileCont, ColumnType::Float)
            },
        ));
        // Empty and all-NULL inputs → NULL.
        cases.push((
            ColumnType::Float,
            vec![V::Null, V::Null],
            minmax_call(F::Sum, ColumnType::Float),
        ));
        cases.push((
            ColumnType::Float,
            vec![],
            minmax_call(F::Avg, ColumnType::Float),
        ));
        for (ty, values, call) in cases {
            let got = run(typed_scan(&values, ty), vec![call.clone()]);
            let rows: Vec<row::Row> = values.iter().map(|v| vec![v.clone()]).collect();
            let expected =
                crate::executor::agg::fold_aggregates(std::slice::from_ref(&call), rows.iter())
                    .unwrap();
            assert_eq!(
                format!("{got:?}"),
                format!("{expected:?}"),
                "{:?} (distinct={}) over {ty:?} {values:?}",
                call.func,
                call.distinct
            );
        }
    }

    /// A-PERF.AGG5a: an integer SUM that overflows `i64` fails on the SIMD path with **the same
    /// error** the row path raises — never a wrapped or truncated value.
    #[test]
    fn sum_int_overflow_errors_like_the_row_path() {
        let values = [Some(i64::MAX), Some(1)];
        let mut op = ScalarAggregate::new(
            Box::new(int_scan(&values)),
            vec![call(ast::AggregateFunc::Sum, ColumnType::Int)],
        );
        let got = op.next_batch().expect_err("i64::MAX + 1 must overflow");
        let rows = [vec![ast::Value::Int(i64::MAX)], vec![ast::Value::Int(1)]];
        let expected = crate::executor::agg::fold_aggregates(
            &[call(ast::AggregateFunc::Sum, ColumnType::Int)],
            rows.iter(),
        )
        .expect_err("row path must overflow identically");
        assert_eq!(got.to_string(), expected.to_string());
    }

    fn run(scan: SeqScan, calls: Vec<AggregateCall>) -> Vec<ast::Value> {
        let mut op = ScalarAggregate::new(Box::new(scan), calls);
        let batch = op.next_batch().unwrap().expect("one result batch");
        assert!(op.next_batch().unwrap().is_none(), "only one row expected");
        // Read each output column's single value back.
        (0..batch.num_columns())
            .map(|c| {
                let col = batch.column(c).unwrap();
                if col.is_null(0) {
                    ast::Value::Null
                } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                    ast::Value::Int(a.get(0).unwrap())
                } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                    ast::Value::Float(a.get(0).unwrap())
                } else if let Some(a) = col.as_any().downcast_ref::<crate::batch::DecimalArray>() {
                    ast::Value::Numeric(a.get(0).unwrap())
                } else if let Some(a) = col.as_any().downcast_ref::<crate::batch::StringArray>() {
                    ast::Value::Text(a.get(0).unwrap().to_owned())
                } else if let Some(a) = col
                    .as_any()
                    .downcast_ref::<crate::batch::TemporalArray<crate::batch::DateKind>>()
                {
                    ast::Value::Date(a.get(0).unwrap())
                } else if let Some(a) = col
                    .as_any()
                    .downcast_ref::<crate::batch::TemporalArray<crate::batch::TimeKind>>()
                {
                    ast::Value::Time(a.get(0).unwrap())
                } else if let Some(a) = col
                    .as_any()
                    .downcast_ref::<crate::batch::TemporalArray<crate::batch::TimeTzKind>>()
                {
                    ast::Value::TimeTz(a.get(0).unwrap())
                } else if let Some(a) = col
                    .as_any()
                    .downcast_ref::<crate::batch::TemporalArray<crate::batch::TimestampKind>>()
                {
                    ast::Value::Timestamp(a.get(0).unwrap())
                } else if let Some(a) = col
                    .as_any()
                    .downcast_ref::<crate::batch::TemporalArray<crate::batch::TimestampTzKind>>()
                {
                    ast::Value::TimestampTz(a.get(0).unwrap())
                } else {
                    panic!("unexpected column type")
                }
            })
            .collect()
    }

    #[test]
    fn sum_min_max_count_over_int_column() {
        use ast::AggregateFunc::{Count, Max, Min, Sum};
        let scan = int_scan(&[Some(3), Some(1), None, Some(5), Some(2)]);
        let calls = vec![
            call(Sum, ColumnType::Int),
            call(Min, ColumnType::Int),
            call(Max, ColumnType::Int),
            call(Count, ColumnType::Int),
        ];
        let out = run(scan, calls);
        assert_eq!(
            out,
            vec![
                ast::Value::Int(11), // 3+1+5+2 (NULL skipped)
                ast::Value::Int(1),
                ast::Value::Int(5),
                ast::Value::Int(4), // four non-null
            ]
        );
    }

    #[test]
    fn aggregates_over_empty_input_are_null_except_count() {
        use ast::AggregateFunc::{Count, Min, Sum};
        let out = run(
            int_scan(&[]),
            vec![
                call(Sum, ColumnType::Int),
                call(Min, ColumnType::Int),
                call(Count, ColumnType::Int),
            ],
        );
        assert_eq!(
            out,
            vec![ast::Value::Null, ast::Value::Null, ast::Value::Int(0)]
        );
    }

    #[test]
    fn matches_row_path_fold_for_sum() {
        // The SIMD SUM(INT) equals the row evaluator's fold over the same rows.
        let n = i64::try_from(crate::BATCH_SIZE).unwrap() + 7;
        let values: Vec<Option<i64>> = (0..n)
            .map(|i| if i % 5 == 0 { None } else { Some(i) })
            .collect();
        let simd_out = run(int_scan(&values), vec![sum_call()]);
        let rows: Vec<row::Row> = values
            .iter()
            .map(|&v| vec![v.map_or(ast::Value::Null, ast::Value::Int)])
            .collect();
        let expected = crate::executor::agg::fold_aggregates(&[sum_call()], rows.iter()).unwrap();
        assert_eq!(simd_out, expected);
    }
}
