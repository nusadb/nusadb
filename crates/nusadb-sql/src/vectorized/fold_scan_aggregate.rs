//! [`FoldScanAggregate`]: a scalar aggregate fused into the scan.
//!
//! For a `ScalarAggregate` whose calls are only integer `COUNT`/`SUM`/`MIN`/`MAX` over bare columns of
//! a single table scan, this decodes each tuple's needed columns and accumulates directly into the
//! per-call [`Acc`] — skipping the `Int64Array` column build (the row->column transpose) the batch path
//! pays. A scalar aggregate is a streaming sink, not a pipeline breaker, so the columnar intermediate
//! between scan and fold is unnecessary materialization (measured: the build is ~3× the per-field
//! decode-walk on a plain integer scan).
//!
//! **Correctness = the row path's, by reuse.** Each value folds through the very [`Acc`] methods the
//! grouped SIMD path uses (`fold_int_sum_partial`/`fold_min_int`/`fold_max_int`/`add_count`), which are
//! order-free/associative, and the result is produced by the shared [`finalize_aggregate`]. Folding
//! one value at a time therefore yields the same accumulator — hence the same value and the same
//! overflow error — as reducing the whole column. Any call outside the supported shape (`DISTINCT`,
//! `FILTER`, a computed argument, a non-integer `SUM`/`MIN`/`MAX`, `AVG`, temporal `MIN`/`MAX`, …) makes
//! the whole operator decline, and the planner keeps the columnar aggregate.

use std::sync::Arc;

use nusadb_core::engine::{TableSchema, TupleScan};
use nusadb_core::{ColumnType, StorageEngine, TxnId};

use crate::ast;
use crate::batch::convert::rows_to_batch;
use crate::batch::{Field, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::agg::{Acc, finalize_aggregate, is_integer};
use crate::executor::row;
use crate::planner::{AggregateCall, TypedExprKind};
use crate::vectorized::Operator;

/// One eligible call's fold. The `usize` is the **source** column ordinal (the tuple's own layout),
/// already mapped back through any projection-pushdown narrowing.
#[derive(Debug, Clone, Copy)]
enum ScanFold {
    /// `COUNT(*)`: every row.
    CountStar,
    /// `COUNT(col)`: rows where the column is non-null (any column type — presence only).
    CountCol(usize),
    /// `SUM(int_col)` with an integer result.
    SumInt(usize),
    /// `MIN(int_col)`.
    MinInt(usize),
    /// `MAX(int_col)`.
    MaxInt(usize),
}

impl ScanFold {
    /// The source column this fold reads, or `None` for `COUNT(*)`.
    const fn column(self) -> Option<usize> {
        match self {
            Self::CountStar => None,
            Self::CountCol(c) | Self::SumInt(c) | Self::MinInt(c) | Self::MaxInt(c) => Some(c),
        }
    }

    /// Whether this fold needs the column's **value** decoded (`SUM`/`MIN`/`MAX`), as opposed to only
    /// its presence (`COUNT(col)`).
    const fn needs_value(self) -> bool {
        matches!(self, Self::SumInt(_) | Self::MinInt(_) | Self::MaxInt(_))
    }
}

/// Classify one call into a [`ScanFold`], mapping its argument's scan-output ordinal back to the
/// source column, or `None` when the call is outside the fused-scan shape.
fn classify(call: &AggregateCall, columns: &[usize], source_width: usize) -> Option<ScanFold> {
    use ast::AggregateFunc as F;
    if call.distinct
        || call.filter.is_some()
        || call.fraction.is_some()
        || call.arg2.is_some()
        || !call.order_by.is_empty()
    {
        return None;
    }
    // A row-value COUNT carries no `arg` either, but it is not COUNT(*): its fields are evaluated
    // per row, so folding it as a bare tally would swallow a field's error (`count((a, 1/0))`
    // raises on the row path) and would bake in the "a row value is never NULL" rule this fold
    // knows nothing about.
    if !call.row_args.is_empty() {
        return None;
    }
    // `COUNT(*)` — every row, no argument.
    if matches!(call.func, F::Count) && call.arg.is_none() {
        return Some(ScanFold::CountStar);
    }
    let arg = call.arg.as_ref()?;
    let TypedExprKind::Column(out) = arg.kind else {
        return None;
    };
    // Map the scan-output ordinal to the source column: identity for a full-width scan, else the
    // narrowed scan's kept ordinals.
    let source = if columns.is_empty() {
        (out < source_width).then_some(out)?
    } else {
        *columns.get(out)?
    };
    match call.func {
        F::Count => Some(ScanFold::CountCol(source)),
        // SUM over an integer column: the exact i128 fold serves both an integer result and a
        // NUMERIC result — SUM(BIGINT) promotes to NUMERIC, and `finalize_aggregate` reads the
        // same i128 accumulator for either. Keeping it on this raw-tuple fold avoids materializing
        // a column for the (integer) argument.
        F::Sum
            if is_integer(arg.ty)
                && (is_integer(call.result_ty)
                    || matches!(call.result_ty, ColumnType::Numeric { .. })) =>
        {
            Some(ScanFold::SumInt(source))
        },
        F::Min if is_integer(arg.ty) => Some(ScanFold::MinInt(source)),
        F::Max if is_integer(arg.ty) => Some(ScanFold::MaxInt(source)),
        _ => None,
    }
}

/// A scalar aggregate folded directly over a table's tuple scan.
pub struct FoldScanAggregate {
    /// The table's tuple scan (owns its cursor).
    scan: Box<dyn TupleScan>,
    /// Every source column's type, in tuple layout order — drives the field walk's skip of unneeded
    /// columns.
    source_types: Vec<ColumnType>,
    /// One fold per call, in call order.
    folds: Vec<ScanFold>,
    /// `col_needed[c]` — any fold reads source column `c` (else the field is skipped without decoding).
    col_needed: Vec<bool>,
    /// `col_decode_value[c]` — a `SUM`/`MIN`/`MAX` fold reads source column `c`'s integer value (else
    /// only its presence is needed, so it is not decoded as an `i64`).
    col_decode_value: Vec<bool>,
    /// The calls, kept for [`finalize_aggregate`].
    calls: Vec<AggregateCall>,
    /// Output schema: one nullable field per call.
    schema: Arc<Schema>,
    /// Set once the single result row has been produced.
    done: bool,
}

impl FoldScanAggregate {
    /// Build a fused scalar aggregate over an already-open `scan` whose tuples are encoded under
    /// `source_types`, computing `calls`. `columns` is the scan's projection-pushdown kept ordinals
    /// (empty = full width). Returns `None` when any call is outside the supported shape.
    fn new(
        scan: Box<dyn TupleScan>,
        source_types: Vec<ColumnType>,
        columns: &[usize],
        calls: &[AggregateCall],
    ) -> Option<Self> {
        let source_width = source_types.len();
        let mut folds = Vec::with_capacity(calls.len());
        for call in calls {
            folds.push(classify(call, columns, source_width)?);
        }
        let mut col_needed = vec![false; source_width];
        let mut col_decode_value = vec![false; source_width];
        for fold in &folds {
            // `c` comes from `classify`, which mapped it into `0..source_width`.
            if let Some(c) = fold.column()
                && let (Some(needed), Some(decode)) =
                    (col_needed.get_mut(c), col_decode_value.get_mut(c))
            {
                *needed = true;
                *decode = *decode || fold.needs_value();
            }
        }
        let schema = Arc::new(Schema::new(
            calls
                .iter()
                .enumerate()
                .map(|(i, c)| Field::new(format!("agg{i}"), c.result_ty, true))
                .collect(),
        ));
        Some(Self {
            scan,
            source_types,
            folds,
            col_needed,
            col_decode_value,
            calls: calls.to_vec(),
            schema,
            done: false,
        })
    }

    /// Open a fused scalar aggregate over `table` within `txn`, or `None` when `calls` is outside the
    /// supported shape (the caller then builds the columnar aggregate).
    ///
    /// # Errors
    ///
    /// Propagates any error from opening the storage [`scan`](StorageEngine::scan).
    pub fn try_open(
        engine: &dyn StorageEngine,
        txn: TxnId,
        table: &TableSchema,
        columns: &[usize],
        calls: &[AggregateCall],
    ) -> Result<Option<Self>, Error> {
        let source_types: Vec<ColumnType> = table.columns.iter().map(|c| c.ty).collect();
        // Classify against the source layout first, so an ineligible shape does not open a scan.
        let source_width = source_types.len();
        if calls
            .iter()
            .any(|call| classify(call, columns, source_width).is_none())
        {
            return Ok(None);
        }
        let scan = engine.scan(txn, table.id)?;
        Ok(Self::new(scan, source_types, columns, calls))
    }

    /// Fold the entire scan into the per-call accumulators, decoding only needed columns.
    #[allow(
        clippy::indexing_slicing,
        reason = "col_needed/col_decode_value are sized to source_types.len(); the field walk enumerates that same range, so c is always in bounds"
    )]
    fn fold_all(&mut self) -> Result<Vec<Acc>, Error> {
        let mut accs: Vec<Acc> = (0..self.calls.len()).map(|_| Acc::default()).collect();
        let mut seen: u32 = 0;
        while let Some((_, tuple)) = self.scan.try_next()? {
            // Cooperative cancellation, amortized over the scan.
            seen = seen.wrapping_add(1);
            if seen.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            // COUNT(*) counts the row before any column is looked at.
            for (i, fold) in self.folds.iter().enumerate() {
                if matches!(fold, ScanFold::CountStar)
                    && let Some(acc) = accs.get_mut(i)
                {
                    acc.add_count(1);
                }
            }
            let bytes: &[u8] = &tuple;
            let mut pos = 0usize;
            for (c, &ty) in self.source_types.iter().enumerate() {
                let (present, payload) = row::field_tag(bytes, pos)?;
                if !self.col_needed[c] {
                    // Unneeded column: advance past it without materializing anything.
                    pos = if present {
                        row::skip_value(bytes, payload, ty)?
                    } else {
                        payload
                    };
                    continue;
                }
                if !present {
                    // A NULL never contributes to SUM/MIN/MAX/COUNT(col); the tag is already consumed.
                    pos = payload;
                    continue;
                }
                if self.col_decode_value[c] {
                    // Integer column read once; every SUM/MIN/MAX/COUNT(col) fold on it uses the value.
                    let (v, next) = row::read_i64_field(bytes, payload)?;
                    for (i, fold) in self.folds.iter().enumerate() {
                        let Some(acc) = accs.get_mut(i) else { continue };
                        match *fold {
                            ScanFold::SumInt(cc) if cc == c => {
                                acc.fold_int_sum_partial(i128::from(v));
                            },
                            ScanFold::MinInt(cc) if cc == c => acc.fold_min_int(v),
                            ScanFold::MaxInt(cc) if cc == c => acc.fold_max_int(v),
                            ScanFold::CountCol(cc) if cc == c => acc.add_count(1),
                            _ => {},
                        }
                    }
                    pos = next;
                } else {
                    // Only COUNT(col) reads this column — presence is enough, so skip its value.
                    for (i, fold) in self.folds.iter().enumerate() {
                        if let ScanFold::CountCol(cc) = *fold
                            && cc == c
                            && let Some(acc) = accs.get_mut(i)
                        {
                            acc.add_count(1);
                        }
                    }
                    pos = row::skip_value(bytes, payload, ty)?;
                }
            }
        }
        Ok(accs)
    }
}

impl std::fmt::Debug for FoldScanAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoldScanAggregate")
            .field("folds", &self.folds)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl Operator for FoldScanAggregate {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let accs = self.fold_all()?;
        let result: Vec<ast::Value> = accs
            .into_iter()
            .zip(&self.calls)
            .map(|(acc, call)| finalize_aggregate(acc, call))
            .collect::<Result<_, _>>()?;
        rows_to_batch(&self.schema, vec![result]).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::{FoldScanAggregate, ScanFold, classify};
    use crate::ast;
    use crate::executor::agg::fold_aggregates;
    use crate::executor::row::{self, Row};
    use crate::planner::{AggregateCall, TypedExpr, TypedExprKind};
    use crate::vectorized::Operator;
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

    fn scan(rows: &[Vec<ast::Value>], types: &[ColumnType]) -> Box<dyn TupleScan> {
        let tuples = rows
            .iter()
            .map(|r| SharedTuple::from(row::encode(r, types).unwrap().as_slice()))
            .collect();
        Box::new(VecScan { tuples, pos: 0 })
    }

    fn call(
        func: ast::AggregateFunc,
        arg: Option<usize>,
        arg_ty: ColumnType,
        result_ty: ColumnType,
    ) -> AggregateCall {
        AggregateCall {
            func,
            arg: arg.map(|i| TypedExpr {
                kind: TypedExprKind::Column(i),
                ty: arg_ty,
            }),
            result_ty,
            distinct: false,
            fraction: None,
            ordered_set_descending: false,
            filter: None,
            separator: None,
            arg2: None,
            order_by: Vec::new(),
            row_args: Vec::new(),
            grouping_args: Vec::new(),
        }
    }

    /// The fused fold must equal the row path's `fold_aggregates` over the same rows, for the whole
    /// integer COUNT/SUM/MIN/MAX surface, over NULL-heavy and adversarial (i64 extremes) input.
    #[test]
    fn matches_row_path_over_int_aggregates() {
        use ast::AggregateFunc::{Count, Max, Min, Sum};
        let int = ColumnType::Int;
        let rows: Vec<Vec<ast::Value>> = vec![
            vec![ast::Value::Int(3), ast::Value::Int(100)],
            vec![ast::Value::Int(1), ast::Value::Null],
            vec![ast::Value::Null, ast::Value::Int(-7)],
            vec![ast::Value::Int(5), ast::Value::Int(2)],
            vec![ast::Value::Int(i64::MAX), ast::Value::Int(i64::MIN)],
            vec![ast::Value::Int(-3), ast::Value::Int(2)],
        ];
        let types = [int, int];
        // COUNT(*), COUNT(a), SUM(b), MIN(a), MAX(a), MIN(b), MAX(b) — SUM(a) would overflow i64, and
        // both paths must error identically, so it is exercised separately below. COUNT is `Int` here,
        // matching the analyzer (`Count` → `ColumnType::Int`), so the built column's type matches.
        let calls = vec![
            call(Count, None, int, int),
            call(Count, Some(0), int, int),
            call(Sum, Some(1), int, int),
            call(Min, Some(0), int, int),
            call(Max, Some(0), int, int),
            call(Min, Some(1), int, int),
            call(Max, Some(1), int, int),
        ];
        let mut op = FoldScanAggregate::new(scan(&rows, &types), types.to_vec(), &[], &calls)
            .expect("eligible shape");
        let batch = op.next_batch().unwrap().expect("one result batch");
        let got: Row = crate::batch::convert::batch_to_rows(&batch).pop().unwrap();
        let want = fold_aggregates(&calls, rows.iter()).unwrap();
        assert_eq!(got, want);
        assert!(op.next_batch().unwrap().is_none());
    }

    /// An integer `SUM` that overflows `i64` must fail with the same error the row path raises.
    #[test]
    fn sum_overflow_errors_like_the_row_path() {
        use ast::AggregateFunc::Sum;
        let int = ColumnType::Int;
        let rows = vec![vec![ast::Value::Int(i64::MAX)], vec![ast::Value::Int(1)]];
        let types = [int];
        let calls = vec![call(Sum, Some(0), int, int)];
        let mut op =
            FoldScanAggregate::new(scan(&rows, &types), types.to_vec(), &[], &calls).unwrap();
        let got = op.next_batch().expect_err("i64::MAX + 1 overflows");
        let want = fold_aggregates(&calls, rows.iter()).expect_err("row path overflows too");
        assert_eq!(got.to_string(), want.to_string());
    }

    /// A narrowed scan maps the call's output ordinal back to the source column: here the scan keeps
    /// only source column 2 (as output ordinal 0), so `SUM(col0)` sums the third stored column.
    #[test]
    fn narrowed_scan_maps_output_ordinal_to_source() {
        use ast::AggregateFunc::Sum;
        let int = ColumnType::Int;
        // Three stored columns; the aggregate reads only the third (source ordinal 2).
        let rows: Vec<Vec<ast::Value>> = vec![
            vec![ast::Value::Int(9), ast::Value::Int(9), ast::Value::Int(10)],
            vec![ast::Value::Int(9), ast::Value::Int(9), ast::Value::Int(20)],
            vec![ast::Value::Int(9), ast::Value::Int(9), ast::Value::Null],
        ];
        let types = [int, int, int];
        // The narrowed scan output has one column (ordinal 0) = source ordinal 2.
        let calls = vec![call(Sum, Some(0), int, int)];
        let mut op =
            FoldScanAggregate::new(scan(&rows, &types), types.to_vec(), &[2], &calls).unwrap();
        let batch = op.next_batch().unwrap().unwrap();
        let got = crate::batch::convert::batch_to_rows(&batch).pop().unwrap();
        assert_eq!(got, vec![ast::Value::Int(30)]);
    }

    /// A non-integer `SUM` (and other unsupported shapes) makes the operator decline, so the planner
    /// keeps the columnar aggregate.
    #[test]
    fn declines_unsupported_shape() {
        use ast::AggregateFunc::{Avg, Sum};
        let float = ColumnType::Float;
        let int = ColumnType::Int;
        // SUM(FLOAT): not an integer sum.
        assert!(
            FoldScanAggregate::new(
                scan(&[], &[float]),
                vec![float],
                &[],
                &[call(Sum, Some(0), float, float)],
            )
            .is_none()
        );
        // AVG is not folded here.
        assert!(
            FoldScanAggregate::new(
                scan(&[], &[int]),
                vec![int],
                &[],
                &[call(
                    Avg,
                    Some(0),
                    int,
                    ColumnType::Numeric {
                        precision: 0,
                        scale: 0
                    },
                )],
            )
            .is_none()
        );
    }

    /// A row-value `COUNT` looks like `COUNT(*)` (no `arg`) but is not one: the fused fold has no
    /// way to evaluate its fields, so it must decline rather than tally rows — which would drop a
    /// field's evaluation error and silently disagree with the row path.
    #[test]
    fn declines_a_row_value_count() {
        use ast::AggregateFunc::Count;
        let int = ColumnType::Int;
        let mut composite = call(Count, None, int, int);
        composite.row_args = vec![
            TypedExpr {
                kind: TypedExprKind::Column(0),
                ty: int,
            },
            TypedExpr {
                kind: TypedExprKind::Column(1),
                ty: int,
            },
        ];
        assert!(classify(&composite, &[], 2).is_none());
        // The same call without the fields is a real COUNT(*), still claimed.
        assert!(matches!(
            classify(&call(Count, None, int, int), &[], 2),
            Some(ScanFold::CountStar)
        ));
    }
}
