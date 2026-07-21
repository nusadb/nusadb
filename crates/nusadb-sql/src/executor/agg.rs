//! Aggregation operators: `GroupBy` / `GROUPING SETS` and scalar aggregates,
//! plus the per-call accumulator and its finalization.
//!
//! Hoisted verbatim out of `executor/ops.rs` (ADR 007 §4.6 deviation cleanup).
//! Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

thread_local! {
    /// How many statistics-routed direct hash folds ran under a spill budget — pins assert on
    /// it so a silently re-routed plan can never make an equivalence test vacuous.
    static STATS_HASH_AGG_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many statistics-routed direct hash folds have run under a spill budget on this thread.
#[must_use]
pub fn stats_hash_agg_count() -> u64 {
    STATS_HASH_AGG_RUNS.with(std::cell::Cell::get)
}

/// Note one statistics-routed direct hash fold (see [`stats_hash_agg_count`]).
pub(super) fn note_stats_hash_agg() {
    STATS_HASH_AGG_RUNS.with(|c| c.set(c.get() + 1));
}

/// Spilling group-by: sort the input by the group keys via the external merge sort (which
/// spills to disk under the work-memory budget), then fold *adjacent* groups in one streaming pass —
/// so only one group's rows plus the merge heads are held in memory, not the whole input or every
/// group at once. Result is the same multiset as [`run_group_aggregate_streamed`]: rows sharing a group key
/// are adjacent after sorting (`group_keys_equal` ⇒ they compare equal ⇒ they sort together).
///
/// # Errors
/// Propagates streaming, spill-file, sort, and aggregate-evaluation errors.
pub(super) fn sort_based_group_aggregate(
    input: &PhysicalOperator,
    group_keys: &[TypedExpr],
    calls: &[AggregateCall],
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    // Sort by every group key (ascending, default NULL placement — only adjacency matters here).
    let order: Vec<crate::planner::OrderByKey> = group_keys
        .iter()
        .map(|expr| crate::planner::OrderByKey {
            expr: expr.clone(),
            ascending: true,
            nulls: ast::NullOrdering::Default,
        })
        .collect();
    let mut sorted = super::spill_sort::sorted_input(input, &order, config, engine, txn)?;

    let mut out = Vec::new();
    let mut group: Vec<Row> = Vec::new();
    let mut current: Option<Vec<ast::Value>> = None;
    while let Some(row) = sorted.try_next()? {
        let key = group_keys
            .iter()
            .map(|k| eval_arg(k, &row))
            .collect::<Result<Vec<_>, _>>()?;
        let same_group = current
            .as_ref()
            .is_some_and(|prev| group_keys_equal(prev, &key));
        if !same_group {
            // Group boundary: emit the just-finished group (if any), then start a new one.
            if let Some(prev) = current.take() {
                let mut out_row = prev;
                out_row.extend(fold_aggregates(calls, &group)?);
                out.push(out_row);
                group.clear();
            }
            current = Some(key);
        }
        group.push(row);
    }
    if let Some(prev) = current {
        let mut out_row = prev;
        out_row.extend(fold_aggregates(calls, &group)?);
        out.push(out_row);
    }
    Ok(out)
}

/// Multi-grouping-set aggregation over a **pulled row stream** (/ the
/// residual): one pass over the input, folding every row into per-set per-group accumulators —
/// memory O(Σ groups per set) instead of O(input), so a `ROLLUP`/`CUBE` over millions of rows no
/// longer materializes the scan. Within a set, groups are emitted in first-seen order and each
/// group's rows fold in input order; the emitted row keeps the full
/// `[group keys ++ aggregate results]` width, with `NULL` in every key column the set does not
/// group by — exactly the materializing version's output. A linear group search (mirroring
/// [`run_group_aggregate_streamed`]) keeps each set order-stable; a hash group-by is a shared
/// perf follow-up.
///
/// # Errors
/// Propagates streaming, key-evaluation, and aggregate-evaluation errors.
pub(super) fn run_grouping_sets_aggregate_streamed(
    source: &mut dyn super::stream::RowSource,
    group_keys: &[TypedExpr],
    grouping_sets: &[Vec<usize>],
    calls: &[AggregateCall],
) -> Result<Vec<Row>, Error> {
    // Per set: the key expressions it groups by (a subset of `group_keys`) and its group states.
    let active: Vec<Vec<&TypedExpr>> = grouping_sets
        .iter()
        .map(|set| set.iter().filter_map(|&i| group_keys.get(i)).collect())
        .collect();
    let mut per_set: Vec<GroupIndex> = grouping_sets.iter().map(|_| GroupIndex::new()).collect();
    while let Some(row) = source.try_next()? {
        for (exprs, groups) in active.iter().zip(&mut per_set) {
            let key = exprs
                .iter()
                .map(|expr| eval_arg(expr, &row))
                .collect::<Result<Vec<_>, _>>()?;
            let at = groups.find_or_create(key, calls.len());
            if let Some(accs) = groups.accs_at(at) {
                accumulate_row(accs, calls, &row)?;
            }
        }
    }
    let mut out = Vec::new();
    for (set, groups) in grouping_sets.iter().zip(per_set) {
        let mut groups = groups.states;
        // A set with no rows still yields the grand-total row when it groups by
        // nothing (e.g. the `()` set of ROLLUP/CUBE), matching scalar aggregation.
        if groups.is_empty() && set.is_empty() {
            groups.push((Vec::new(), vec![Acc::default(); calls.len()]));
        }
        for (active_key, accs) in groups {
            // Lay the active key values back into their full-width slots; absent
            // columns stay `NULL`.
            let mut out_row = vec![ast::Value::Null; group_keys.len()];
            for (&slot, value) in set.iter().zip(active_key) {
                if let Some(cell) = out_row.get_mut(slot) {
                    *cell = value;
                }
            }
            for (acc, call) in accs.into_iter().zip(calls) {
                out_row.push(finalize_aggregate(acc, call)?);
            }
            // GROUPING(...): its value is a function of *this* set, not the folded rows, so
            // overwrite each GROUPING slot with the bitmask for the keys it names. Aggregate slot `ci`
            // lives at `group_keys.len() + ci` in the `[keys ++ aggregates]` row.
            for (ci, call) in calls.iter().enumerate() {
                if matches!(call.func, ast::AggregateFunc::Grouping)
                    && let Some(cell) = out_row.get_mut(group_keys.len() + ci)
                {
                    *cell = ast::Value::Int(grouping_mask(&call.grouping_args, set));
                }
            }
            out.push(out_row);
        }
    }
    Ok(out)
}

/// Compute the `GROUPING(arg, ...)` bitmask for one super-aggregate row. `grouping_args` are
/// the `group_keys` indices the call names (leftmost = most-significant bit); `active` are the key
/// indices this grouping set still groups by. A bit is `1` when its key was *grouped away* (not in
/// `active`), `0` when present — so `GROUPING(a, b)` over the set `{a}` yields `0b01` = `1`.
fn grouping_mask(grouping_args: &[usize], active: &[usize]) -> i64 {
    let n = grouping_args.len();
    let mut mask = 0i64;
    for (pos, key_idx) in grouping_args.iter().enumerate() {
        if !active.contains(key_idx) {
            // Leftmost argument is the highest bit: shift by (n - 1 - pos).
            mask |= 1i64 << (n - 1 - pos);
        }
    }
    mask
}

/// Fold one group of rows into a single result row, one value per aggregate
/// call. Used for both the whole input (scalar aggregate) and per-group
/// (`GROUP BY`). NULL handling follows SQL: `COUNT(*)` counts every row, every
/// other aggregate skips NULL arguments; an empty group gives `COUNT = 0` and
/// `NULL` for everything else.
#[allow(
    clippy::cast_precision_loss,
    reason = "AVG divides an i64 count into the f64 sum — the count→f64 widening is inherently lossy and intended"
)]
/// One accumulator per aggregate call, walked in lockstep with `calls` via `zip`.
#[derive(Default, Clone)]
pub(crate) struct Acc {
    count: i64,
    sum: f64,
    // Exact running sum for integer SUM: `SUM(Int)` must not lose precision through
    // f64 (wrong result past 2^53). `sum` (f64) is kept for AVG, which is fractional anyway.
    int_sum: i128,
    // Exact running sum for NUMERIC SUM/AVG; `None` until a numeric value is seen.
    dec_sum: Option<crate::numeric::Decimal>,
    min: Option<ast::Value>,
    max: Option<ast::Value>,
    any_seen: bool,
    // For `DISTINCT` aggregates: the non-`NULL` argument values already folded into this
    // accumulator, bucketed by [`distinct_hash`] so a duplicate is found in O(1) amortized instead of
    // an O(n) scan of every prior value (the old `Vec` made `COUNT(DISTINCT)` O(n²) — A-perf). The
    // hash only has to satisfy "compare-equal ⇒ same bucket"; correctness is still decided by
    // [`eval::compare`] within the bucket, so a hash collision costs a comparison, never correctness.
    // Empty (and unused) for non-DISTINCT calls.
    distinct_seen: std::collections::HashMap<u64, Vec<ast::Value>>,
    // For ordered-set aggregates (PERCENTILE_CONT/DISC, MODE): every non-`NULL` ordering
    // value in the group, sorted at finalization. Empty (and unused) for other aggregates.
    ordered_values: Vec<ast::Value>,
    // For ARRAY_AGG: every collected value in input order, NULLs included (unlike the other
    // aggregates, which skip NULL). Empty (and unused) for other aggregates.
    array_items: Vec<ast::Value>,
    // For ARRAY_AGG / STRING_AGG with `ORDER BY`: the evaluated sort-key tuple for each
    // value in `array_items`, in the same order. Empty (and unused) when the call has no `ORDER BY`.
    agg_sort_keys: Vec<Vec<ast::Value>>,
    // For STDDEV/VARIANCE: the running sum of squares (`sum` holds the running sum, `count` the n).
    sum_sq: f64,
    // For BOOL_AND/BOOL_OR: the running boolean fold; `None` until the first non-NULL bool is seen.
    bool_fold: Option<bool>,
    // For BIT_AND/BIT_OR: the running integer bit fold; `None` until the first non-NULL int.
    bit_fold: Option<i64>,
    // For the two-argument statistical aggregates CORR/COVAR_POP/COVAR_SAMP: the moments of
    // the second value (`x`) and the cross term. `sum`/`sum_sq` hold the first value (`y`)'s sum and
    // sum of squares, and `count` the number of non-NULL `(y, x)` pairs.
    sum_x: f64,
    sum_x2: f64,
    sum_xy: f64,
}

/// Finalize one `REGR_*` linear-regression aggregate from the `(y, x)` pair moments. All but
/// `REGR_COUNT` are `NULL` for an empty group; the slope/intercept/R² are additionally `NULL` when
/// `Sxx` is `0` (a vertical fit), and `REGR_R2` is `1` when `Syy` is `0`.
#[allow(
    clippy::cast_precision_loss,
    reason = "the pair count -> f64 widening is intended for the fractional statistics"
)]
fn regr_finalize(func: ast::AggregateFunc, acc: &Acc) -> ast::Value {
    use ast::AggregateFunc as F;
    // REGR_COUNT is the pair count and is always defined (0 for an empty group).
    if func == F::RegrCount {
        return ast::Value::Int(acc.count);
    }
    if acc.count < 1 {
        return ast::Value::Null;
    }
    let n = acc.count as f64;
    let avgx = acc.sum_x / n;
    let avgy = acc.sum / n;
    // Σ(x−avgx)², Σ(y−avgy)², Σ(x−avgx)(y−avgy); the squared terms are clamped against float
    // cancellation, the cross term may legitimately be negative.
    let sxx = (acc.sum_x2 - acc.sum_x * acc.sum_x / n).max(0.0);
    let syy = (acc.sum_sq - acc.sum * acc.sum / n).max(0.0);
    let sxy = acc.sum_xy - acc.sum_x * acc.sum / n;
    match func {
        F::RegrAvgx => ast::Value::Float(avgx),
        F::RegrAvgy => ast::Value::Float(avgy),
        F::RegrSxx => ast::Value::Float(sxx),
        F::RegrSyy => ast::Value::Float(syy),
        F::RegrSxy => ast::Value::Float(sxy),
        F::RegrSlope if sxx != 0.0 => ast::Value::Float(sxy / sxx),
        F::RegrIntercept if sxx != 0.0 => {
            let slope = sxy / sxx;
            ast::Value::Float(slope.mul_add(-avgx, avgy))
        },
        F::RegrR2 if sxx != 0.0 => {
            if syy == 0.0 {
                ast::Value::Float(1.0)
            } else {
                ast::Value::Float(sxy.powi(2) / (sxx * syy))
            }
        },
        // RegrSlope/RegrIntercept/RegrR2 with Sxx == 0 (undefined fit), or any non-REGR function.
        _ => ast::Value::Null,
    }
}

/// Whether `ty` is one of the integer widths (`SMALLINT`/`INT`/`BIGINT`). Because an expression now
/// carries its integer width, a `BIGINT`/`SMALLINT` aggregate argument must take the
/// same exact-i128 path as a plain `INT` rather than falling through to float accumulation.
pub(crate) const fn is_integer(ty: ColumnType) -> bool {
    matches!(
        ty,
        ColumnType::SmallInt | ColumnType::Int | ColumnType::BigInt
    )
}

/// Produce the final value of one aggregate from its accumulator.
#[allow(
    clippy::cast_precision_loss,
    reason = "final integer-sum / count -> f64 to form a fractional average"
)]
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-aggregate finalization; length tracks the aggregate set"
)]
pub(crate) fn finalize_aggregate(acc: Acc, call: &AggregateCall) -> Result<ast::Value, Error> {
    use crate::ast::AggregateFunc as F;
    Ok(match call.func {
        F::Count => ast::Value::Int(acc.count),
        // GROUPING(...): the `0` placeholder ("nothing grouped away"). The grouping-sets path
        // overwrites this per super-aggregate row via `grouping_mask`; any other path (plain GROUP BY
        // never emits a GROUPING call) keeps `0`.
        F::Grouping => ast::Value::Int(0),
        F::Sum => {
            if !acc.any_seen {
                ast::Value::Null
            } else if matches!(call.result_ty, ColumnType::Numeric { .. }) {
                // SUM(NUMERIC) is exact.
                ast::Value::Numeric(acc.dec_sum.ok_or_else(numeric_overflow)?)
            } else if is_integer(call.result_ty) {
                // SUM over any integer width returns an integer from the exact i128 accumulator;
                // overflowing i64 is an error rather than a silent f64-truncated wrong answer.
                ast::Value::Int(i64::try_from(acc.int_sum).map_err(|_| numeric_overflow())?)
            } else {
                ast::Value::Float(acc.sum)
            }
        },
        F::Avg => {
            if acc.count == 0 {
                ast::Value::Null
            } else if matches!(call.result_ty, ColumnType::Numeric { .. }) {
                // AVG over an exact type (Int / NUMERIC) is exact NUMERIC — `AVG(int)` divides the
                // exact i128 sum, never an f64-accumulated one, so it stays exact past 2^53 and
                // matches the NUMERIC division precision (Temuan-4). The sum comes from the i128
                // accumulator for an integer argument or the Decimal accumulator for a NUMERIC one.
                let sum = if call.arg.as_ref().is_some_and(|a| is_integer(a.ty)) {
                    crate::numeric::Decimal::from_i128(acc.int_sum)
                } else {
                    acc.dec_sum.ok_or_else(numeric_overflow)?
                };
                let avg = sum
                    .checked_div(&crate::numeric::Decimal::from_i64(acc.count))
                    .ok_or_else(numeric_overflow)?;
                ast::Value::Numeric(avg)
            } else {
                // AVG(FLOAT) stays FLOAT.
                ast::Value::Float(acc.sum / acc.count as f64)
            }
        },
        F::Min => acc.min.unwrap_or(ast::Value::Null),
        F::Max => acc.max.unwrap_or(ast::Value::Null),
        F::PercentileCont | F::PercentileDisc | F::Mode => finalize_ordered_set(
            call.func,
            call.fraction,
            call.ordered_set_descending,
            acc.ordered_values,
        ),
        // ARRAY_AGG: the collected values as an array; an empty group (nothing seen) → NULL.
        // An `ORDER BY` clause reorders the values before they form the array.
        F::ArrayAgg => {
            if acc.any_seen {
                let items = if call.order_by.is_empty() {
                    acc.array_items
                } else {
                    sort_agg_values(acc.array_items, acc.agg_sort_keys, &call.order_by)
                };
                ast::Value::Array(items)
            } else {
                ast::Value::Null
            }
        },
        // STRING_AGG: the collected text values joined by the separator; NULL for an empty group.
        // An `ORDER BY` clause reorders the values before they are joined.
        F::StringAgg => {
            if acc.any_seen {
                let items = if call.order_by.is_empty() {
                    acc.array_items
                } else {
                    sort_agg_values(acc.array_items, acc.agg_sort_keys, &call.order_by)
                };
                let sep = call.separator.as_deref().unwrap_or("");
                let joined = items
                    .iter()
                    .map(|v| match v {
                        ast::Value::Text(s) => s.clone(),
                        other => crate::display::value_text(other),
                    })
                    .collect::<Vec<_>>()
                    .join(sep);
                ast::Value::Text(joined)
            } else {
                ast::Value::Null
            }
        },
        // BOOL_AND / BOOL_OR: the folded boolean, or NULL for an empty / all-NULL group.
        F::BoolAnd | F::BoolOr => acc.bool_fold.map_or(ast::Value::Null, ast::Value::Bool),
        // BIT_AND / BIT_OR / BIT_XOR: the folded integer, or NULL for an empty / all-NULL group.
        F::BitAnd | F::BitOr | F::BitXor => acc.bit_fold.map_or(ast::Value::Null, ast::Value::Int),
        // COVAR_POP / COVAR_SAMP / CORR over the (y, x) pairs. `sum`/`sum_sq` are Σy/Σy²,
        // `sum_x`/`sum_x2` are Σx/Σx², `sum_xy` is Σxy, `count` is the pair count `n`.
        F::CovarPop | F::CovarSamp => {
            let min = if call.func == F::CovarPop { 1 } else { 2 };
            if acc.count < min {
                ast::Value::Null
            } else {
                let n = acc.count as f64;
                let divisor = if call.func == F::CovarPop { n } else { n - 1.0 };
                ast::Value::Float((acc.sum_xy - acc.sum_x * acc.sum / n) / divisor)
            }
        },
        F::Corr => {
            if acc.count < 1 {
                ast::Value::Null
            } else {
                let n = acc.count as f64;
                let cov = n.mul_add(acc.sum_xy, -(acc.sum_x * acc.sum));
                let var_x = n.mul_add(acc.sum_x2, -(acc.sum_x * acc.sum_x)).max(0.0);
                let var_y = n.mul_add(acc.sum_sq, -(acc.sum * acc.sum)).max(0.0);
                let denom = (var_x * var_y).sqrt();
                // Either input having zero variance leaves the correlation undefined (NULL).
                if denom == 0.0 {
                    ast::Value::Null
                } else {
                    ast::Value::Float(cov / denom)
                }
            }
        },
        // The REGR_* linear-regression family over the (y, x) pairs; each is a closed-form
        // function of the same moment sums.
        F::RegrCount
        | F::RegrAvgx
        | F::RegrAvgy
        | F::RegrSxx
        | F::RegrSyy
        | F::RegrSxy
        | F::RegrSlope
        | F::RegrIntercept
        | F::RegrR2 => regr_finalize(call.func, &acc),
        // STDDEV / VARIANCE: the sample statistic; NULL for fewer than two values. A tiny negative
        // variance from float cancellation is clamped to 0.
        F::Stddev | F::Variance => {
            if acc.count < 2 {
                ast::Value::Null
            } else {
                let n = acc.count as f64;
                let variance = ((acc.sum_sq - acc.sum * acc.sum / n) / (n - 1.0)).max(0.0);
                ast::Value::Float(if call.func == F::Variance {
                    variance
                } else {
                    variance.sqrt()
                })
            }
        },
        // STDDEV_POP / VAR_POP: the population statistic (divisor `n`); NULL only for an empty group.
        F::StddevPop | F::VarPop => {
            if acc.count < 1 {
                ast::Value::Null
            } else {
                let n = acc.count as f64;
                let variance = ((acc.sum_sq - acc.sum * acc.sum / n) / n).max(0.0);
                ast::Value::Float(if call.func == F::VarPop {
                    variance
                } else {
                    variance.sqrt()
                })
            }
        },
    })
}

/// Compute an ordered-set aggregate (`PERCENTILE_CONT`/`PERCENTILE_DISC`, `MODE`) over the
/// group's collected ordering values. An empty group yields `NULL`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "percentile rank math is f64; the group size and ranks are small non-negative counts"
)]
fn finalize_ordered_set(
    func: ast::AggregateFunc,
    fraction: Option<f64>,
    descending: bool,
    mut values: Vec<ast::Value>,
) -> ast::Value {
    use crate::ast::AggregateFunc as F;
    if values.is_empty() {
        return ast::Value::Null;
    }
    values.sort_by(eval::compare);
    // A `DESC` WITHIN GROUP ordering reverses the set; the percentile/mode formulas below then apply
    // the fraction from the top (e.g. `percentile_cont(0.25) ... DESC` = the value 25% from the top).
    if descending {
        values.reverse();
    }
    let n = values.len();
    match func {
        // Continuous percentile: linear interpolation between the two values bracketing the rank
        // `f·(n−1)` (0-based). Numeric input only (guaranteed by the analyzer) → FLOAT.
        F::PercentileCont => {
            let f = fraction.unwrap_or(0.0);
            let rank = f * (n - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            let weight = rank - lo as f64;
            let lo_v = values.get(lo).map_or(0.0, value_as_f64);
            let hi_v = values.get(hi).map_or(0.0, value_as_f64);
            ast::Value::Float((hi_v - lo_v).mul_add(weight, lo_v))
        },
        // Discrete percentile: the first value whose 1-based position reaches `f·n`.
        F::PercentileDisc => {
            let f = fraction.unwrap_or(0.0);
            // `ceil(f·n)` clamped to `[1, n]`; for f=0 this is the first value.
            let pos = (f * n as f64).ceil() as usize;
            let idx = pos.clamp(1, n) - 1;
            values.get(idx).cloned().unwrap_or(ast::Value::Null)
        },
        // Mode: the most frequent value; ties resolve to the one that sorts first (smallest). Equal
        // values are adjacent after sorting, so a single run-length pass suffices.
        F::Mode => {
            let (mut best_start, mut best_len) = (0usize, 0usize);
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n
                    && values
                        .get(i)
                        .zip(values.get(j))
                        .is_some_and(|(a, b)| eval::compare(a, b) == std::cmp::Ordering::Equal)
                {
                    j += 1;
                }
                if j - i > best_len {
                    best_len = j - i;
                    best_start = i;
                }
                i = j;
            }
            values.get(best_start).cloned().unwrap_or(ast::Value::Null)
        },
        _ => ast::Value::Null,
    }
}

/// Record the `ORDER BY` sort-key tuple for the value just collected into `acc.array_items`, keeping
/// the two vectors parallel. A no-op when the call has no `ORDER BY`.
fn push_agg_sort_keys(acc: &mut Acc, call: &AggregateCall, row: &Row) -> Result<(), Error> {
    if call.order_by.is_empty() {
        return Ok(());
    }
    let mut keys = Vec::with_capacity(call.order_by.len());
    for key in &call.order_by {
        keys.push(eval_arg(&key.expr, row)?);
    }
    acc.agg_sort_keys.push(keys);
    Ok(())
}

/// Reorder the collected `ARRAY_AGG` / `STRING_AGG` values by their recorded `ORDER BY` keys.
/// `keys` is parallel to `items`; the sort is stable, so equal keys keep input order.
fn sort_agg_values(
    items: Vec<ast::Value>,
    keys: Vec<Vec<ast::Value>>,
    order_by: &[OrderByKey],
) -> Vec<ast::Value> {
    let mut paired: Vec<(Vec<ast::Value>, ast::Value)> = keys.into_iter().zip(items).collect();
    paired.sort_by(|(ka, _), (kb, _)| {
        // The key tuples and `order_by` are the same length (one key per ORDER BY item), so zipping
        // all three keeps the comparison in lock-step without indexing.
        for ((a, b), key) in ka.iter().zip(kb).zip(order_by) {
            let ord = eval::compare_order_key(a, b, key.ascending, key.nulls);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    paired.into_iter().map(|(_, value)| value).collect()
}

/// Fold a set of rows into one aggregate result row. Takes any single-pass iterator of `&Row` so
/// callers (group-by buckets, grouping sets, window frames) can fold over **borrowed** rows —
/// indices into the input or a sub-slice — without cloning them into a fresh `Vec`.
pub(crate) fn fold_aggregates<'a, I>(calls: &[AggregateCall], input_rows: I) -> Result<Row, Error>
where
    I: IntoIterator<Item = &'a Row>,
{
    let mut accs = vec![Acc::default(); calls.len()];
    for row in input_rows {
        accumulate_row(&mut accs, calls, row)?;
    }
    let mut out = Vec::with_capacity(calls.len());
    for (acc, call) in accs.into_iter().zip(calls) {
        out.push(finalize_aggregate(acc, call)?);
    }
    Ok(out)
}

/// Whether `call` is a plain `COUNT(*)` — the count-every-row aggregate with no argument, no
/// `DISTINCT`, and no `FILTER`. Such a call needs only the row count, not any row's contents.
const fn is_plain_count_star(call: &AggregateCall) -> bool {
    matches!(call.func, ast::AggregateFunc::Count)
        && call.arg.is_none()
        && !call.distinct
        && call.filter.is_none()
}

/// `COUNT(*)` fast-path: when a scalar aggregate is one or more plain `COUNT(*)`s directly over a
/// base-table `SeqScan`, the answer is just the number of visible rows — [`count_table`] counts them
/// without decoding any row, instead of the fold materializing every row's full width only to
/// discard it. Returns the single output row, or `None` when the shape does not qualify (a `FILTER`,
/// `DISTINCT`, an argument, an intervening `Filter`/projection, or any non-`COUNT(*)` call) so the
/// caller folds the streamed input as usual. Byte-identical to the fold: each `COUNT(*)` slot is
/// `Int(count)` (see [`finalize_aggregate`]).
pub(super) fn scalar_count_star_fast(
    input: &PhysicalOperator,
    calls: &[AggregateCall],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<Row>, Error> {
    let PhysicalOperator::SeqScan { table, .. } = input else {
        return Ok(None);
    };
    if calls.is_empty() || !calls.iter().all(is_plain_count_star) {
        return Ok(None);
    }
    let n = super::scan::count_table(table, engine, txn)?;
    // EXPLAIN ANALYZE per-node actuals: the fast-path never streams the scan through its counting
    // source, so record the scan node's row count here (the aggregate's own one row is recorded by
    // the caller's node wrapper). Keyed by the scan node's address in the executed plan tree.
    if super::instrument::enabled() {
        super::instrument::record(super::instrument::key(input), n as u64);
    }
    let count = i64::try_from(n).map_err(|_| numeric_overflow())?;
    Ok(Some(vec![ast::Value::Int(count); calls.len()]))
}

/// Fold a **pulled row stream** into one scalar-aggregate output row without materializing the
/// input: `count`/`sum`/`min`/`max`/`avg` hold O(1) accumulator state
/// (DISTINCT / ordered-set / `array_agg` hold what they inherently must), where the materializing
/// path held the entire scan and ran out of memory on a full-table `count(*)` past ~1M rows.
/// Yields exactly what
/// [`fold_aggregates`] over the same rows would.
///
/// # Errors
/// Propagates streaming and aggregate-evaluation errors.
pub(super) fn fold_aggregates_streamed(
    calls: &[AggregateCall],
    source: &mut dyn super::stream::RowSource,
) -> Result<Row, Error> {
    let mut accs = vec![Acc::default(); calls.len()];
    while let Some(row) = source.try_next()? {
        accumulate_row(&mut accs, calls, &row)?;
    }
    let mut out = Vec::with_capacity(calls.len());
    for (acc, call) in accs.into_iter().zip(calls) {
        out.push(finalize_aggregate(acc, call)?);
    }
    Ok(out)
}

/// The aggregate functions whose value over a sliding frame is maintainable in O(1) per frame-edge
/// move — reproducing [`finalize_aggregate`] over the frame EXACTLY. See [`sliding_window_aggregate`].
enum SlideKind {
    CountStar, // count(*): the frame width (NULLs included)
    CountExpr, // count(expr): non-NULL values
    SumInt,    // SUM over an integer result: exact i128 add/subtract
    SumDec,    // SUM over a NUMERIC result: exact Decimal add/subtract
    Avg,       // AVG over an exact (integer/NUMERIC) result: exact sum / count
    Min,       // MIN via a monotonic (increasing) deque
    Max,       // MAX via a monotonic (decreasing) deque
}

/// Evaluate an explicit-frame window aggregate in O(n) total with a sliding accumulator, when `call`
/// is one whose add/remove (or monotonic-deque) form reproduces [`finalize_aggregate`] over the
/// frame EXACTLY. Returns `Ok(false)` for any other aggregate, so the caller folds each frame from
/// scratch. This is the Leis'15 removable-aggregate optimisation for `ROWS` frames: the previous
/// per-row re-fold was O(n·w) in the frame width `w`, which a swinging `ROWS BETWEEN x PRECEDING AND
/// y FOLLOWING` (or `MIN`/`MAX`) drove toward O(n²) as `w` grew (
/// swinging-frame case).
///
/// `frame_at(k)` is the inclusive `[lo, hi]` frame for row `k`, or `None` when the frame is empty.
/// The frames MUST be monotonic — both edges non-decreasing in `k`, and any empty frames confined to
/// the ends — which holds for `ROWS` frames (the caller restricts to those). `value_at(pos)`
/// evaluates the aggregate argument at ordered position `pos`; `emit(k, value)` receives each result.
///
/// SUM/AVG are handled only for exact integer/`NUMERIC` results — a FLOAT running total would drift
/// under add/subtract and diverge from the from-scratch fold, so those fall back.
///
/// # Errors
/// Propagates argument-evaluation and numeric-overflow errors.
#[allow(
    clippy::too_many_lines,
    reason = "one add-arm and one remove-arm per supported aggregate kind; the length tracks that \
              fixed set and splitting it would only scatter the tightly-coupled slide state"
)]
pub(super) fn sliding_window_aggregate(
    call: &AggregateCall,
    n: usize,
    frame_at: impl Fn(usize) -> Option<(usize, usize)>,
    value_at: impl Fn(usize) -> Result<ast::Value, Error>,
    mut emit: impl FnMut(usize, ast::Value) -> Result<(), Error>,
) -> Result<bool, Error> {
    use crate::ast::AggregateFunc as F;
    let kind = match call.func {
        F::Count if call.arg.is_none() => SlideKind::CountStar,
        F::Count => SlideKind::CountExpr,
        F::Min => SlideKind::Min,
        F::Max => SlideKind::Max,
        F::Sum if is_integer(call.result_ty) => SlideKind::SumInt,
        F::Sum if matches!(call.result_ty, ColumnType::Numeric { .. }) => SlideKind::SumDec,
        F::Avg if matches!(call.result_ty, ColumnType::Numeric { .. }) => SlideKind::Avg,
        _ => return Ok(false),
    };

    let mut count: i64 = 0; // non-NULL value count (SUM/AVG `any_seen` + AVG divisor + count(expr))
    let mut int_sum: i128 = 0;
    let mut dec_sum = crate::numeric::Decimal::ZERO;
    // Monotonic deque of (pos, value) for MIN/MAX. Positions increase front-to-back, so a position
    // leaving the frame is only ever the front.
    let mut deque: std::collections::VecDeque<(usize, ast::Value)> =
        std::collections::VecDeque::new();
    let mut cur_lo = 0usize; // live frame is the half-open [cur_lo, cur_hi)
    let mut cur_hi = 0usize;

    for k in 0..n {
        let Some((lo, hi)) = frame_at(k) else {
            // Empty frame (only ever at the ends for a ROWS frame): the empty-aggregate value.
            emit(k, finalize_aggregate(Acc::default(), call)?)?;
            continue;
        };
        let hi_ex = hi + 1;
        // Advance the high edge: add each row entering the frame.
        while cur_hi < hi_ex {
            let value = value_at(cur_hi)?;
            let is_null = matches!(value, ast::Value::Null);
            match kind {
                SlideKind::CountStar => {},
                SlideKind::CountExpr => {
                    if !is_null {
                        count += 1;
                    }
                },
                SlideKind::SumInt => {
                    if let ast::Value::Int(i) = value {
                        int_sum = int_sum.wrapping_add(i128::from(i));
                        count += 1;
                    }
                },
                SlideKind::SumDec | SlideKind::Avg => {
                    if let Some(d) = value_as_decimal(&value) {
                        dec_sum = dec_sum.checked_add(&d).ok_or_else(numeric_overflow)?;
                        int_sum = int_sum.wrapping_add(match value {
                            ast::Value::Int(i) => i128::from(i),
                            _ => 0,
                        });
                        count += 1;
                    }
                },
                SlideKind::Min => {
                    if !is_null {
                        while deque.back().is_some_and(|(_, v)| {
                            eval::compare(v, &value) == std::cmp::Ordering::Greater
                        }) {
                            deque.pop_back();
                        }
                        deque.push_back((cur_hi, value));
                    }
                },
                SlideKind::Max => {
                    if !is_null {
                        while deque.back().is_some_and(|(_, v)| {
                            eval::compare(v, &value) == std::cmp::Ordering::Less
                        }) {
                            deque.pop_back();
                        }
                        deque.push_back((cur_hi, value));
                    }
                },
            }
            cur_hi += 1;
        }
        // Advance the low edge: remove each row leaving the frame.
        while cur_lo < lo {
            match kind {
                SlideKind::CountStar => {},
                SlideKind::CountExpr => {
                    if !matches!(value_at(cur_lo)?, ast::Value::Null) {
                        count -= 1;
                    }
                },
                SlideKind::SumInt => {
                    if let ast::Value::Int(i) = value_at(cur_lo)? {
                        int_sum = int_sum.wrapping_sub(i128::from(i));
                        count -= 1;
                    }
                },
                SlideKind::SumDec | SlideKind::Avg => {
                    let value = value_at(cur_lo)?;
                    if let Some(d) = value_as_decimal(&value) {
                        dec_sum = dec_sum.checked_sub(&d).ok_or_else(numeric_overflow)?;
                        int_sum = int_sum.wrapping_sub(match value {
                            ast::Value::Int(i) => i128::from(i),
                            _ => 0,
                        });
                        count -= 1;
                    }
                },
                SlideKind::Min | SlideKind::Max => {
                    if deque.front().is_some_and(|(pos, _)| *pos == cur_lo) {
                        deque.pop_front();
                    }
                },
            }
            cur_lo += 1;
        }
        let value = if matches!(kind, SlideKind::CountStar) {
            ast::Value::Int(i64::try_from(cur_hi - cur_lo).unwrap_or(i64::MAX))
        } else {
            let front = deque.front().map(|(_, v)| v.clone());
            let (min, max) = match kind {
                SlideKind::Min => (front, None),
                SlideKind::Max => (None, front),
                _ => (None, None),
            };
            let acc = Acc {
                count,
                any_seen: count > 0,
                int_sum,
                dec_sum: Some(dec_sum),
                min,
                max,
                ..Acc::default()
            };
            finalize_aggregate(acc, call)?
        };
        emit(k, value)?;
    }
    Ok(true)
}

/// Hash a group-key tuple compatibly with [`group_keys_equal`] (A-PERF.AGG1): keys that compare
/// equal must land in the same bucket, so the equality probe stays authoritative (a collision
/// costs one comparison, never correctness). Each element delegates to [`distinct_hash`], which
/// was built for exactly this invariant and already handles the traps a naive `f64`-bits hash
/// reintroduces: `-0.0`/`+0.0` normalize together, every NaN shares one bucket (this codebase's
/// float compare treats all NaN as equal), and `Int`/`Numeric` hash the **exact** trimmed decimal
/// — two equal high-precision values whose `mantissa as f64` rounds differently across scales
/// still land together.
///
/// Scope note (same contract `DISTINCT` relies on): a `Float` never buckets with `Int`/`Numeric`.
/// That pair cannot reach a group key — a single key expression yields type-homogeneous values
/// (mixed numeric CASE/COALESCE branches are widened by the analyzer and **coerced at eval time**
/// via `coerce_numeric_to`), so cross-family equality only exists for value pairs no evaluated
/// key repertoire contains.
/// A fixed-seed [`ahash::AHasher`] for the hot bucketing hashes (`GROUP BY` keys, `DISTINCT`
/// aggregate values). ahash is ~3-5× faster than the standard `DefaultHasher` (SipHash-1-3) for the
/// short keys these paths hash per row — the residual cost QA measured on `GROUP BY`.
///
/// Fixed seeds keep it **deterministic**, which the bucketing invariant requires: two values that
/// [`eval::compare`] calls equal canonicalize to the same bytes here and so must hash to the same
/// value (`distinct_hash_agrees_with_compare_equality` pins this). The hash only ever *buckets* —
/// membership is always decided by `eval::compare` / [`group_keys_equal`] within the bucket (a
/// collision costs one comparison, never a wrong answer) — so swapping the non-cryptographic hash
/// function cannot change any result. (Not a hash-flooding regression either: the `DefaultHasher` it
/// replaces is likewise fixed-seed, and the outer `HashMap` index still re-hashes with its own
/// randomized state.)
fn fast_hasher() -> ahash::AHasher {
    use std::hash::BuildHasher as _;
    ahash::RandomState::with_seeds(
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
        0x1656_67b1_9e37_79f9,
        0x2545_f491_4f6c_dd1d,
    )
    .build_hasher()
}

pub(super) fn group_key_hash(key: &[ast::Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = fast_hasher();
    for value in key {
        distinct_hash(value).hash(&mut hasher);
    }
    hasher.finish()
}

/// Evaluate an aggregate argument / group key, fast-pathing the ubiquitous plain column reference
/// (`sum(col)`, `GROUP BY col`) past the expression interpreter (A-PERF.AGG2). Mirrors
/// [`eval::eval`]'s `Column` arm exactly, including the malformed-tuple error.
fn eval_arg(expr: &TypedExpr, row: &Row) -> Result<ast::Value, Error> {
    if let crate::planner::TypedExprKind::Column(index) = expr.kind {
        return row
            .get(index)
            .cloned()
            .ok_or(Error::MalformedTuple { offset: index });
    }
    eval::eval(expr, row)
}

/// First-seen-ordered group states with a hash index over the keys (A-PERF.AGG1): find-or-create
/// is O(1) amortized instead of a linear scan per row (O(rows × groups)). Emission order stays
/// the `states` insertion order. Shared with the vectorized `GroupedAggregate` (A-PERF.AGG6) so
/// both group-by paths probe **one** hash/equality contract — the F2c "coordinate with the
/// row-path hash group-by, don't duplicate it" requirement.
pub(crate) struct GroupIndex {
    states: Vec<(Vec<ast::Value>, Vec<Acc>)>,
    index: HashMap<u64, Vec<usize>>,
}

impl GroupIndex {
    pub(crate) fn new() -> Self {
        Self {
            states: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// The position of `key`'s group, creating it (with `calls_len` fresh accumulators) on first
    /// sight. `key` is consumed only when the group is new.
    pub(crate) fn find_or_create(&mut self, key: Vec<ast::Value>, calls_len: usize) -> usize {
        let bucket = self.index.entry(group_key_hash(&key)).or_default();
        let found = bucket.iter().copied().find(|&at| {
            self.states
                .get(at)
                .is_some_and(|(k, _)| group_keys_equal(k, &key))
        });
        found.unwrap_or_else(|| {
            let at = self.states.len();
            self.states.push((key, vec![Acc::default(); calls_len]));
            bucket.push(at);
            at
        })
    }

    pub(crate) fn accs_at(&mut self, at: usize) -> Option<&mut Vec<Acc>> {
        self.states.get_mut(at).map(|(_, accs)| accs)
    }

    /// Consume the index into its `(key, accumulators)` states, in first-seen order.
    pub(crate) fn into_states(self) -> Vec<(Vec<ast::Value>, Vec<Acc>)> {
        self.states
    }
}

/// Whether values of `ty` are safe under parallel partial aggregation's merge: two values
/// that [`eval::compare`] calls equal must be **byte-identical**, because a first-seen tie
/// between workers is broken by merge order — a type where equal values can differ in
/// representation (NUMERIC `1.0` vs `1.00`, FLOAT `-0.0` vs `0.0`) would leak that order into
/// the output bytes. Applies to group-key columns and MIN/MAX arguments alike.
pub(crate) const fn parallel_safe_ty(ty: ColumnType) -> bool {
    matches!(
        ty.physical(),
        ColumnType::Bool
            | ColumnType::Int
            | ColumnType::Text
            | ColumnType::Bytes
            | ColumnType::Date
            | ColumnType::Time
            | ColumnType::TimeTz
            | ColumnType::Timestamp
            | ColumnType::TimestampTz
            | ColumnType::Uuid
    )
}

/// Whether `call`'s accumulator can be split into parallel partials and merged back
/// bit-identically to the sequential fold. Requires an associative, merge-order-free
/// fold: `COUNT` is a plain tally; integer/NUMERIC `SUM` totals are exact (`i128` / decimal) —
/// a FLOAT-typed `SUM`/`AVG` reads the non-associative `f64` running total, so it stays
/// sequential; `MIN`/`MAX` qualify only over [`parallel_safe_ty`] arguments. Everything else
/// (DISTINCT, FILTER, ordered-set, statistics, collectors) keeps the sequential path.
pub(crate) fn call_is_parallel_mergeable(call: &AggregateCall) -> bool {
    use crate::ast::AggregateFunc as F;
    if call.distinct
        || call.filter.is_some()
        || call.arg2.is_some()
        || !call.order_by.is_empty()
        || call.fraction.is_some()
        || call.separator.is_some()
    {
        return false;
    }
    match call.func {
        F::Count => true,
        F::Sum => matches!(
            call.result_ty.physical(),
            ColumnType::Int | ColumnType::Numeric { .. }
        ),
        F::Min | F::Max => call
            .arg
            .as_ref()
            .is_some_and(|arg| parallel_safe_ty(arg.ty)),
        _ => false,
    }
}

/// Fold a parallel partial accumulator into `into` — the merge step of the parallel grouped
/// aggregate. Only sound for calls [`call_is_parallel_mergeable`] admits: the merged
/// state is then identical to folding both partials' input rows through [`fold_value`]
/// sequentially, in any interleaving.
///
/// # Errors
/// A call outside the mergeable set is refused loudly (an internal routing bug, never data).
pub(crate) fn merge_acc(into: &mut Acc, from: Acc, call: &AggregateCall) -> Result<(), Error> {
    use crate::ast::AggregateFunc as F;
    match call.func {
        F::Count => into.count += from.count,
        F::Sum => {
            let into_empty = into.count == 0;
            // Mirror the fold: every accumulator advances so finalize reads a consistent total
            // whichever one the result type selects. The `f64` total is merged for completeness
            // but never read here (the mergeable gate excludes FLOAT-typed SUM).
            into.sum += from.sum;
            into.int_sum = into.int_sum.wrapping_add(from.int_sum);
            into.dec_sum = match (into.dec_sum.take(), from.dec_sum) {
                (Some(a), Some(b)) => a.checked_add(&b),
                (a, None) if from.count == 0 => a,
                (None, b) if into_empty => b,
                // One side folded values but lost its exact decimal total (overflow) — the
                // merged total is lost exactly as the sequential fold would have lost it.
                _ => None,
            };
            into.count += from.count;
        },
        F::Min => {
            if let Some(value) = from.min
                && into
                    .min
                    .as_ref()
                    .is_none_or(|cur| eval::compare(&value, cur) == std::cmp::Ordering::Less)
            {
                into.min = Some(value);
            }
        },
        F::Max => {
            if let Some(value) = from.max
                && into
                    .max
                    .as_ref()
                    .is_none_or(|cur| eval::compare(&value, cur) == std::cmp::Ordering::Greater)
            {
                into.max = Some(value);
            }
        },
        _ => {
            return Err(Error::Unsupported(
                "internal: aggregate is not parallel-mergeable".to_owned(),
            ));
        },
    }
    into.any_seen |= from.any_seen;
    Ok(())
}

/// Fold one row into a `COUNT(*)` accumulator — the argument-less arm of [`accumulate_row`],
/// exposed for the vectorized grouped fold (A-PERF.AGG6), which has no row to pass. Identical to
/// that arm: every row counts, NULLs included.
pub(crate) const fn fold_count_star(acc: &mut Acc) {
    acc.count += 1;
    acc.any_seen = true;
}

/// Incremental group-by over a **pulled row stream**: per-group
/// accumulators — O(groups) memory instead of O(input). Groups are emitted in first-seen order
/// and each group's rows fold in input order, matching the materializing group-by it
/// replaces. (The spilling sort-based group-by remains the
/// bounded-memory path for huge group counts)
///
/// # Errors
/// Propagates streaming, key-evaluation, and aggregate-evaluation errors.
pub(super) fn run_group_aggregate_streamed(
    source: &mut dyn super::stream::RowSource,
    group_keys: &[TypedExpr],
    calls: &[AggregateCall],
) -> Result<Vec<Row>, Error> {
    let mut groups = GroupIndex::new();
    while let Some(row) = source.try_next()? {
        let key = group_keys
            .iter()
            .map(|k| eval_arg(k, &row))
            .collect::<Result<Vec<_>, _>>()?;
        let at = groups.find_or_create(key, calls.len());
        if let Some(accs) = groups.accs_at(at) {
            accumulate_row(accs, calls, &row)?;
        }
    }
    let mut out = Vec::with_capacity(groups.states.len());
    for (key, accs) in groups.states {
        let mut out_row = key;
        for (acc, call) in accs.into_iter().zip(calls) {
            out_row.push(finalize_aggregate(acc, call)?);
        }
        out.push(out_row);
    }
    Ok(out)
}

/// Fold **one** input row into the per-call accumulators — the single-row step of
/// [`fold_aggregates`], factored out so a streaming caller can pull rows from a source and fold
/// them one at a time instead of materializing the whole input.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-aggregate accumulate dispatch; length tracks the aggregate set"
)]
pub(super) fn accumulate_row(
    accs: &mut [Acc],
    calls: &[AggregateCall],
    row: &Row,
) -> Result<(), Error> {
    use crate::ast::AggregateFunc as F;

    {
        for (acc, call) in accs.iter_mut().zip(calls) {
            // FILTER (WHERE pred): a row contributes to this aggregate only when its
            // predicate is TRUE (a FALSE or NULL result skips the row for this call alone).
            if let Some(filter) = &call.filter
                && !matches!(eval_arg(filter, row)?, ast::Value::Bool(true))
            {
                continue;
            }
            match call.func {
                F::Count if call.arg.is_none() => {
                    acc.count += 1;
                    acc.any_seen = true;
                },
                // GROUPING(...) folds no row values: its result is a function of the current
                // grouping set, written by `run_grouping_sets_aggregate_streamed` after this fold. Finalizing
                // yields the `0` placeholder ("nothing grouped away") for any other path.
                F::Grouping => {},
                // ARRAY_AGG collects every value in input order, NULLs included — so it must
                // run before the NULL-skip below. DISTINCT drops duplicates ("not distinct" equality,
                // so a single NULL is kept).
                F::ArrayAgg => {
                    let arg = call.arg.as_ref().ok_or_else(|| {
                        Error::Unsupported("internal: array_agg requires an argument".to_owned())
                    })?;
                    let value = eval_arg(arg, row)?;
                    if call.distinct {
                        // Reuse the hash-bucketed distinct index (O(1) amortized) instead of scanning
                        // every value collected so far — the linear `array_items.iter().any(...)` made
                        // ARRAY_AGG(DISTINCT) O(k²) per group. NULLs dedup too (a single NULL is
                        // kept); `distinct_hash` buckets every NULL together and `eval::compare` stays
                        // the authoritative tie-break, so a collision costs a comparison, not
                        // correctness.
                        let bucket = acc.distinct_seen.entry(distinct_hash(&value)).or_default();
                        let dup = bucket.iter().any(|seen| match (seen, &value) {
                            (ast::Value::Null, ast::Value::Null) => true,
                            (ast::Value::Null, _) | (_, ast::Value::Null) => false,
                            (a, b) => eval::compare(a, b) == std::cmp::Ordering::Equal,
                        });
                        if dup {
                            continue;
                        }
                        bucket.push(value.clone());
                    }
                    acc.array_items.push(value);
                    push_agg_sort_keys(acc, call, row)?;
                    acc.any_seen = true;
                },
                // The two-argument statistical aggregates (CORR/COVAR_*/REGR_*) fold over
                // (y, x) pairs; a pair contributes only when BOTH values are non-NULL, so this runs
                // before the single-value NULL-skip below.
                f if f.is_two_arg() => {
                    let (arg_y, arg_x) = (call.arg.as_ref(), call.arg2.as_ref());
                    let (Some(arg_y), Some(arg_x)) = (arg_y, arg_x) else {
                        return Err(Error::Unsupported(format!(
                            "internal: aggregate {:?} requires two arguments",
                            call.func,
                        )));
                    };
                    let (y, x) = (eval_arg(arg_y, row)?, eval_arg(arg_x, row)?);
                    if matches!(y, ast::Value::Null) || matches!(x, ast::Value::Null) {
                        continue;
                    }
                    let (yf, xf) = (value_as_f64(&y), value_as_f64(&x));
                    acc.sum += yf;
                    acc.sum_sq += yf * yf;
                    acc.sum_x += xf;
                    acc.sum_x2 += xf * xf;
                    acc.sum_xy += xf * yf;
                    acc.count += 1;
                    acc.any_seen = true;
                },
                _ => {
                    let arg = call.arg.as_ref().ok_or_else(|| {
                        Error::Unsupported(format!(
                            "internal: aggregate {:?} requires an argument",
                            call.func,
                        ))
                    })?;
                    let value = eval_arg(arg, row)?;
                    fold_value(acc, call, value, row)?;
                },
            }
        }
    }
    Ok(())
}

/// Fold one already-evaluated argument `value` into `acc` — the value-level step of
/// [`accumulate_row`]'s single-argument arm, factored out so the vectorized columnar fold
/// (A-PERF.AGG5b) can feed values straight off a column array in row order without materializing
/// rows, and still run **this exact code**. `row` is consulted only by a
/// `STRING_AGG … ORDER BY`'s sort keys (a no-op when the call has no `ORDER BY`, which is the only
/// shape the columnar caller sends).
///
/// Not for `GROUPING` (folds no values), `ARRAY_AGG` (keeps NULLs — folds *before* the NULL-skip
/// here), or the two-argument statistics (pair-wise NULL skip) — those fold in
/// [`accumulate_row`]'s outer match and hit an `unreachable!` arm below.
///
/// # Errors
/// Propagates sort-key evaluation errors.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-aggregate accumulate dispatch; length tracks the aggregate set"
)]
pub(crate) fn fold_value(
    acc: &mut Acc,
    call: &AggregateCall,
    value: ast::Value,
    row: &Row,
) -> Result<(), Error> {
    use crate::ast::AggregateFunc as F;
    if matches!(value, ast::Value::Null) {
        return Ok(());
    }
    // DISTINCT: fold each distinct argument value only once per group.
    // NULLs are already skipped above (SQL counts distinct non-NULL values).
    if call.distinct {
        let bucket = acc.distinct_seen.entry(distinct_hash(&value)).or_default();
        if bucket
            .iter()
            .any(|seen| eval::compare(seen, &value) == std::cmp::Ordering::Equal)
        {
            return Ok(());
        }
        bucket.push(value.clone());
    }
    acc.any_seen = true;
    match call.func {
        F::Count => acc.count += 1,
        // GROUPING folds no row values; handled in accumulate_row's outer match, never here.
        F::Grouping => unreachable!("GROUPING is folded in the outer match arm"),
        F::Sum | F::Avg => {
            // Maintain every accumulator so `finalize_aggregate` reads a consistent
            // total whatever the value mix is — mixed-numeric CASE/COALESCE branches
            // can now feed one SUM/AVG, e.g. `SUM(CASE WHEN c THEN 0.5 ELSE 1
            // END)`. `sum` is the f64 total a FLOAT-typed result reads.
            acc.sum += value_as_f64(&value);
            // Exact i128 total for an INT-typed result (G22).
            if let ast::Value::Int(i) = &value {
                acc.int_sum = acc.int_sum.wrapping_add(i128::from(*i));
            }
            // Exact decimal total for a NUMERIC-typed result. A NUMERIC-typed
            // aggregate never receives a FLOAT value (FLOAT dominates to a FLOAT
            // result), so skipping FLOAT keeps the decimal total exact whenever it
            // is the one finalize reads.
            if let Some(d) = value_as_decimal(&value) {
                let acc_dec = acc.dec_sum.unwrap_or(crate::numeric::Decimal::ZERO);
                acc.dec_sum = acc_dec.checked_add(&d);
            }
            acc.count += 1;
        },
        F::Min => {
            if acc
                .min
                .as_ref()
                .is_none_or(|cur| eval::compare(&value, cur) == std::cmp::Ordering::Less)
            {
                acc.min = Some(value);
            }
        },
        F::Max => {
            if acc
                .max
                .as_ref()
                .is_none_or(|cur| eval::compare(&value, cur) == std::cmp::Ordering::Greater)
            {
                acc.max = Some(value);
            }
        },
        // STDDEV/VARIANCE (sample) and STDDEV_POP/VAR_POP (population) accumulate the
        // running sum + sum of squares + count; the statistic is computed at
        // finalization.
        F::Stddev | F::Variance | F::StddevPop | F::VarPop => {
            let f = value_as_f64(&value);
            acc.sum += f;
            acc.sum_sq += f * f;
            acc.count += 1;
        },
        // STRING_AGG collects the non-NULL text values in input order; joined with
        // the separator at finalization (reuses `array_items` as the value store).
        F::StringAgg => {
            acc.array_items.push(value);
            push_agg_sort_keys(acc, call, row)?;
        },
        // BOOL_AND/BOOL_OR fold the non-NULL booleans with AND / OR respectively.
        F::BoolAnd => {
            if let ast::Value::Bool(b) = value {
                acc.bool_fold = Some(acc.bool_fold.unwrap_or(true) && b);
            }
        },
        F::BoolOr => {
            if let ast::Value::Bool(b) = value {
                acc.bool_fold = Some(acc.bool_fold.unwrap_or(false) || b);
            }
        },
        // BIT_AND/BIT_OR/BIT_XOR fold the non-NULL integers bitwise. The identity is
        // all-ones for AND, 0 for OR/XOR (B-fn).
        F::BitAnd => {
            if let ast::Value::Int(i) = value {
                acc.bit_fold = Some(acc.bit_fold.unwrap_or(!0) & i);
            }
        },
        F::BitOr => {
            if let ast::Value::Int(i) = value {
                acc.bit_fold = Some(acc.bit_fold.unwrap_or(0) | i);
            }
        },
        F::BitXor => {
            if let ast::Value::Int(i) = value {
                acc.bit_fold = Some(acc.bit_fold.unwrap_or(0) ^ i);
            }
        },
        // Ordered-set aggregates collect every non-NULL ordering value; the
        // percentile/mode is computed from the sorted set at finalization.
        F::PercentileCont | F::PercentileDisc | F::Mode => {
            acc.ordered_values.push(value);
        },
        // ARRAY_AGG is handled in accumulate_row's outer match (it keeps NULLs, so it runs
        // before the NULL-skip above) and never reaches this non-NULL value path.
        F::ArrayAgg => unreachable!("array_agg is folded in the outer match arm"),
        // The two-argument statistical aggregates (CORR/COVAR_*/REGR_*) fold over
        // (y, x) pairs in accumulate_row's outer match (the pair is skipped when either
        // side is NULL) and never reach this single-value path.
        F::Corr
        | F::CovarPop
        | F::CovarSamp
        | F::RegrCount
        | F::RegrAvgx
        | F::RegrAvgy
        | F::RegrSxx
        | F::RegrSyy
        | F::RegrSxy
        | F::RegrSlope
        | F::RegrIntercept
        | F::RegrR2 => {
            unreachable!("two-argument statistical aggregates fold in the outer arm")
        },
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "numeric coercion for SUM/AVG accumulation"
)]
pub(super) fn value_as_f64(v: &ast::Value) -> f64 {
    match v {
        ast::Value::Int(i) => *i as f64,
        ast::Value::Float(f) => *f,
        ast::Value::Numeric(d) => d.to_f64(),
        _ => 0.0,
    }
}

/// A bucket hash for `DISTINCT` dedup that is consistent with [`eval::compare`]: two values that
/// compare *equal* hash to the same bucket (so they meet and dedup). It need NOT avoid collisions
/// between unequal values — the bucket scan re-checks with `compare`, so a collision only costs a
/// comparison. The subtle cases mirror `compare`:
///
/// - **Int and Numeric compare *exactly*** (scale-independent: `1` == `1.0` == `1.00`), so both hash
///   on the *exact* trailing-zero-stripped decimal `(mantissa, scale)` — NOT a lossy `f64`, which
///   would split equal high-precision values whose `mantissa as f64` rounds differently across scales.
/// - **Float** hashes on its normalized bits (`-0.0` → `0.0`); two floats compare equal iff their
///   bits match (excluding `±0`). **`NaN`** gets its own bucket: `compare`'s total order makes every
///   `NaN` equal to every other `NaN` and distinct from every number, so an isolated bucket keeps the
///   NaNs together and apart from numbers — matching the standard "all NaN are one value".
/// - Every other type hashes on a per-variant tag plus its exact `compare` key (raw `i64`/bytes/
///   string, `INTERVAL`'s canonical estimate, recursive element hashes for arrays, total-order bits
///   for vectors). Distinct variants never compare equal, so distinct tags are safe.
///
/// Residual: `compare` is *intransitive* across the Float↔exact boundary (it compares a Float to a
/// Numeric via lossy `to_f64`), so a single aggregate that mixes Float with Int/Numeric values that
/// round-equal — only reachable via a mixed-type `CASE`/`COALESCE` argument — may keep them in
/// separate buckets. Every *single-type* `DISTINCT` aggregate (the standard case; a column has one
/// type) is exact.
fn distinct_hash(v: &ast::Value) -> u64 {
    use std::hash::{Hash, Hasher};

    use ast::Value as V;
    let mut h = fast_hasher();
    // Int and Numeric share one tag and key on the exact canonical decimal so `1`, `1.0`, `1.00`
    // (and high-precision scale variants) land together.
    let exact = |d: crate::numeric::Decimal, h: &mut ahash::AHasher| {
        let d = d.trim_scale();
        1u8.hash(h);
        d.mantissa.hash(h);
        d.scale.hash(h);
    };
    match v {
        V::Int(i) => exact(crate::numeric::Decimal::from_i64(*i), &mut h),
        V::Numeric(d) => exact(*d, &mut h),
        V::Float(f) => {
            if f.is_nan() {
                2u8.hash(&mut h);
            } else {
                3u8.hash(&mut h);
                let norm = if *f == 0.0 { 0.0 } else { *f };
                norm.to_bits().hash(&mut h);
            }
        },
        V::Null => 0u8.hash(&mut h),
        V::Bool(b) => (4u8, b).hash(&mut h),
        V::Text(s) => (5u8, s).hash(&mut h),
        V::Json(s) => (6u8, s).hash(&mut h),
        V::Date(d) => (7u8, d).hash(&mut h),
        V::Time(t) => (8u8, t).hash(&mut h),
        V::TimeTz(t) => (9u8, t).hash(&mut h),
        V::Timestamp(t) => (10u8, t).hash(&mut h),
        V::TimestampTz(t) => (11u8, t).hash(&mut h),
        V::Uuid(u) => (12u8, u).hash(&mut h),
        V::Bytes(b) => (13u8, b).hash(&mut h),
        // INTERVAL compares by its canonical estimate (so `1 day` == `24:00:00`); hash that.
        V::Interval(iv) => (14u8, iv.estimate_micros()).hash(&mut h),
        // Recurse so element-wise-equal arrays land together; mix in the length (compare's tiebreak).
        V::Array(items) => {
            15u8.hash(&mut h);
            items.len().hash(&mut h);
            for item in items {
                distinct_hash(item).hash(&mut h);
            }
        },
        // Vectors order by `f32::total_cmp`, where equal ⇔ identical bits — so the raw bits are safe.
        V::Vector(vec) => {
            16u8.hash(&mut h);
            vec.len().hash(&mut h);
            for x in vec {
                x.to_bits().hash(&mut h);
            }
        },
    }
    h.finish()
}

/// The value as an exact [`Decimal`] for the NUMERIC running total, or `None` for a FLOAT (a
/// NUMERIC-typed aggregate never receives one, so it is skipped rather than lossily rounded).
const fn value_as_decimal(v: &ast::Value) -> Option<crate::numeric::Decimal> {
    match v {
        ast::Value::Int(i) => Some(crate::numeric::Decimal::from_i64(*i)),
        ast::Value::Numeric(d) => Some(*d),
        _ => None,
    }
}

/// Error for a NUMERIC SUM/AVG that overflowed the `i128` mantissa. Also raised by the
/// vectorized `SUM(INT)` path (A-PERF.AGG5a), which must fail identically to the row path.
pub(crate) fn numeric_overflow() -> Error {
    Error::Unsupported("numeric aggregate overflow".to_owned())
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::distinct_hash;
    use super::{
        Acc, call_is_parallel_mergeable, finalize_aggregate, fold_aggregates, fold_value,
        merge_acc, sliding_window_aggregate,
    };
    use crate::ast::Value as V;
    use crate::executor::eval;
    use crate::interval::Interval;
    use crate::numeric::Decimal;
    use crate::planner::AggregateCall;
    use nusadb_core::ColumnType;

    /// The hash-index invariant the streamed group-by relies on (A-PERF.AGG1): any two keys
    /// [`crate::executor::ops::group_keys_equal`] calls equal must land in the same hash bucket.
    /// Includes the adversarial rows a naive `f64`-bits hash gets wrong (the audit-caught set):
    /// `-0.0`/`+0.0`, differing NaN payloads, and equal high-precision `Int`/`Numeric` values
    /// whose `mantissa as f64` rounds differently across scales. `Float`-vs-`Int`/`Numeric` pairs
    /// are deliberately absent: a single evaluated key expression never mixes those families
    /// (see `group_key_hash`'s scope note).
    #[test]
    fn group_key_hash_agrees_with_group_key_equality() {
        let month = V::Interval(Interval {
            months: 1,
            days: 0,
            micros: 0,
        });
        let thirty_days = V::Interval(Interval {
            months: 0,
            days: 30,
            micros: 0,
        });
        // A NaN with a non-canonical payload: equal to f64::NAN under this codebase's float
        // compare (all NaN are one group), but a different bit pattern.
        let odd_nan = f64::from_bits(f64::NAN.to_bits() | 1);
        let equal_pairs: Vec<(Vec<V>, Vec<V>)> = vec![
            (vec![V::Float(-0.0)], vec![V::Float(0.0)]),
            (vec![V::Float(f64::NAN)], vec![V::Float(odd_nan)]),
            (
                vec![V::Int(9_007_199_254_740_993)],
                vec![V::Numeric(
                    Decimal::parse("9007199254740993").expect("decimal"),
                )],
            ),
            (
                vec![V::Numeric(
                    Decimal::parse("9007199254740993").expect("decimal"),
                )],
                vec![V::Numeric(
                    Decimal::parse("9007199254740993.00").expect("decimal"),
                )],
            ),
            (
                vec![V::Int(7)],
                vec![V::Numeric(Decimal::parse("7.00").expect("decimal"))],
            ),
            (
                vec![V::Null, V::Int(3)],
                vec![V::Null, V::Numeric(Decimal::parse("3").expect("decimal"))],
            ),
            (vec![month], vec![thirty_days]),
            (
                vec![V::Array(vec![V::Int(1), V::Null])],
                vec![V::Array(vec![
                    V::Numeric(Decimal::parse("1.00").expect("decimal")),
                    V::Null,
                ])],
            ),
        ];
        for (a, b) in equal_pairs {
            assert!(
                crate::executor::ops::group_keys_equal(&a, &b),
                "precondition: {a:?} and {b:?} must be one group"
            );
            assert_eq!(
                super::group_key_hash(&a),
                super::group_key_hash(&b),
                "equal keys must hash equal: {a:?} vs {b:?}"
            );
        }
    }

    /// The bucket-hash invariant the DISTINCT dedup relies on: within a single value type — the
    /// standard case, since a column has one type — any two values that `compare` reports as *equal*
    /// must hash to the same bucket (otherwise they would never meet and would be miscounted as
    /// distinct). The bucket scan handles the reverse direction, so only this implication must hold.
    #[test]
    fn distinct_hash_agrees_with_compare_equality() {
        let num = |s: &str| V::Numeric(Decimal::parse(s).expect("decimal"));
        let iv = |s: &str| V::Interval(Interval::parse(s).expect("interval"));
        // Same-type pairs (plus the exact Int/Numeric family) that `compare` treats as equal — each
        // must share a bucket.
        let equal_pairs = [
            (V::Int(1), num("1.00")),  // int vs numeric — both exact, scale-independent
            (num("1.5"), num("1.50")), // numeric scale variants
            // The high-precision case a lossy-f64 key would split (the mantissa rounds differently
            // across scales past 2^53): exactly equal, so they must stay in one bucket.
            (num("123456789012345678"), num("123456789012345678.00")),
            (V::Float(0.0), V::Float(-0.0)), // signed zero (same Float type)
            (iv("1 day"), iv("24:00:00")),   // interval canonical estimate
            (V::Float(f64::NAN), V::Float(-f64::NAN)), // all NaN share one bucket
        ];
        for (a, b) in &equal_pairs {
            assert_eq!(
                eval::compare(a, b),
                Ordering::Equal,
                "test premise: {a:?} and {b:?} should compare equal"
            );
            assert_eq!(
                distinct_hash(a),
                distinct_hash(b),
                "compare-equal values {a:?} and {b:?} must share a bucket"
            );
        }
        // A NaN must NOT share a bucket with a real number (compare's NaN quirk would otherwise let
        // it swallow them); they stay distinct because the buckets differ.
        assert_ne!(
            distinct_hash(&V::Float(f64::NAN)),
            distinct_hash(&V::Float(5.0))
        );
        // Distinct ordinary values land in distinct buckets (a smoke check, not a correctness need).
        assert_ne!(distinct_hash(&V::Int(1)), distinct_hash(&V::Int(2)));
        assert_ne!(
            distinct_hash(&V::Text("a".into())),
            distinct_hash(&V::Text("b".into()))
        );
    }

    /// An [`AggregateCall`] over an argument of `arg_ty` (or argument-less `COUNT(*)`).
    /// The O(n) sliding-window aggregator must reproduce the from-scratch frame fold BYTE-FOR-BYTE
    /// for every supported aggregate and every ROWS frame shape — including swinging frames
    /// (`x PRECEDING … y FOLLOWING`), following-only frames (empty at the end, a jump-started low
    /// edge), preceding-only frames, and unbounded edges — over data with NULLs and negatives.
    #[test]
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::type_complexity,
        clippy::needless_range_loop,
        reason = "test-only index/offset arithmetic over a small fixed-size fixture"
    )]
    fn sliding_window_aggregate_matches_the_from_scratch_fold() {
        use crate::ast::AggregateFunc as F;
        let bigint = ColumnType::BigInt;
        let numeric = ColumnType::Numeric {
            precision: 30,
            scale: 4,
        };
        let n = 40usize;

        // Deterministic pseudo-random columns (no RNG) with NULLs and negatives.
        let int_rows: Vec<Vec<V>> = (0..n as i64)
            .map(|i| {
                vec![if i % 7 == 3 {
                    V::Null
                } else {
                    V::Int(((i * 37 + 11) % 50) - 25)
                }]
            })
            .collect();
        let num_rows: Vec<Vec<V>> = (0..n as i64)
            .map(|i| {
                vec![if i % 9 == 4 {
                    V::Null
                } else {
                    V::Numeric(
                        Decimal::parse(&format!("{}.{}", (i * 13) % 20, i % 10)).expect("dec"),
                    )
                }]
            })
            .collect();

        // [lo, hi] for a frame given as signed offsets from k, with frame_bounds-style clamp + empty.
        let frame = |start_off: i64, end_off: i64, k: usize| -> Option<(usize, usize)> {
            let ki = k as i64;
            let last = n as i64 - 1;
            let lo = (ki + start_off).max(0);
            let hi = (ki + end_off).min(last);
            if lo > hi {
                None
            } else {
                Some((lo as usize, hi as usize))
            }
        };
        // Swinging, preceding-only, following-only (jump start + empty tail), and unbounded edges.
        let specs: [(i64, i64); 7] = [
            (-2, 1),
            (-3, 0),
            (0, 2),
            (2, 5),
            (-5, -2),
            (-1_000_000, 0),
            (0, 1_000_000),
        ];

        let scenarios: Vec<(F, Option<ColumnType>, ColumnType, &Vec<Vec<V>>)> = vec![
            (F::Count, None, bigint, &int_rows),         // count(*)
            (F::Count, Some(bigint), bigint, &int_rows), // count(expr)
            (F::Sum, Some(bigint), bigint, &int_rows),
            (F::Min, Some(bigint), bigint, &int_rows),
            (F::Max, Some(bigint), bigint, &int_rows),
            (F::Avg, Some(bigint), numeric, &int_rows), // avg(int) -> exact NUMERIC
            (F::Sum, Some(numeric), numeric, &num_rows),
        ];

        for (func, arg_ty, result_ty, rows) in scenarios {
            let c = call(func, arg_ty, result_ty);
            for (so, eo) in specs {
                let mut got = vec![V::Null; n];
                let handled = sliding_window_aggregate(
                    &c,
                    n,
                    |k| frame(so, eo, k),
                    |pos| Ok(rows[pos][0].clone()),
                    |k, v| {
                        got[k] = v;
                        Ok(())
                    },
                )
                .expect("slide ok");
                assert!(handled, "{func:?} must be handled by the sliding path");
                for k in 0..n {
                    let expect = match frame(so, eo, k) {
                        Some((lo, hi)) => {
                            fold_aggregates(std::slice::from_ref(&c), rows[lo..=hi].iter())
                                .expect("fold")
                                .into_iter()
                                .next()
                                .expect("one value")
                        },
                        None => finalize_aggregate(Acc::default(), &c).expect("empty"),
                    };
                    assert_eq!(
                        got[k], expect,
                        "{func:?} frame ({so},{eo}) at k={k} must match the re-fold"
                    );
                }
            }
        }
    }

    fn call(
        func: crate::ast::AggregateFunc,
        arg_ty: Option<ColumnType>,
        result_ty: ColumnType,
    ) -> AggregateCall {
        AggregateCall {
            func,
            arg: arg_ty.map(|ty| crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
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

    /// The parallel-mergeable gate admits exactly the associative, merge-order-free
    /// folds: COUNT, INT/NUMERIC-typed SUM, MIN/MAX over byte-deterministic argument types —
    /// and refuses FLOAT SUM/AVG (non-associative f64), NUMERIC MIN/MAX (equal values can
    /// differ in spelling), DISTINCT, and FILTER.
    #[test]
    fn parallel_mergeable_gate_admits_only_order_free_folds() {
        use crate::ast::AggregateFunc as F;
        let numeric = ColumnType::Numeric {
            precision: 10,
            scale: 2,
        };
        assert!(call_is_parallel_mergeable(&call(
            F::Count,
            None,
            ColumnType::Int
        )));
        assert!(call_is_parallel_mergeable(&call(
            F::Count,
            Some(ColumnType::Float),
            ColumnType::Int
        )));
        assert!(call_is_parallel_mergeable(&call(
            F::Sum,
            Some(ColumnType::Int),
            ColumnType::BigInt
        )));
        assert!(call_is_parallel_mergeable(&call(
            F::Sum,
            Some(numeric),
            numeric
        )));
        assert!(call_is_parallel_mergeable(&call(
            F::Min,
            Some(ColumnType::Text),
            ColumnType::Text
        )));
        assert!(call_is_parallel_mergeable(&call(
            F::Max,
            Some(ColumnType::Timestamp),
            ColumnType::Timestamp
        )));
        // Refusals.
        assert!(!call_is_parallel_mergeable(&call(
            F::Sum,
            Some(ColumnType::Float),
            ColumnType::Float
        )));
        assert!(!call_is_parallel_mergeable(&call(
            F::Avg,
            Some(ColumnType::Int),
            ColumnType::Float
        )));
        assert!(!call_is_parallel_mergeable(&call(
            F::Min,
            Some(numeric),
            numeric
        )));
        assert!(!call_is_parallel_mergeable(&call(
            F::Min,
            Some(ColumnType::Float),
            ColumnType::Float
        )));
        assert!(!call_is_parallel_mergeable(&AggregateCall {
            distinct: true,
            ..call(F::Count, Some(ColumnType::Int), ColumnType::Int)
        }));
        assert!(!call_is_parallel_mergeable(&AggregateCall {
            filter: Some(crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(1),
                ty: ColumnType::Bool,
            }),
            ..call(F::Count, None, ColumnType::Int)
        }));
        assert!(!call_is_parallel_mergeable(&call(
            F::ArrayAgg,
            Some(ColumnType::Int),
            ColumnType::Array(nusadb_core::engine::ArrayElem::Int)
        )));
    }

    /// [`merge_acc`] over split partials finalizes exactly like the sequential fold over the
    /// whole value list — for every admitted call kind, every split point (including the empty
    /// prefix/suffix), NULLs included.
    #[test]
    fn merge_acc_matches_sequential_fold_at_every_split() {
        use crate::ast::AggregateFunc as F;
        let numeric = ColumnType::Numeric {
            precision: 12,
            scale: 3,
        };
        let dec = |s: &str| V::Numeric(Decimal::parse(s).unwrap());
        let cases: Vec<(AggregateCall, Vec<V>)> = vec![
            (
                call(F::Count, Some(ColumnType::Int), ColumnType::Int),
                vec![V::Int(1), V::Null, V::Int(3), V::Int(3), V::Null],
            ),
            (
                call(F::Sum, Some(ColumnType::Int), ColumnType::Int),
                vec![V::Int(5), V::Int(-7), V::Null, V::Int(i64::MAX), V::Int(1)],
            ),
            (
                call(F::Sum, Some(numeric), numeric),
                vec![dec("1.5"), V::Null, dec("-0.25"), dec("100.125")],
            ),
            (
                call(F::Min, Some(ColumnType::Int), ColumnType::Int),
                vec![V::Int(4), V::Null, V::Int(-9), V::Int(-9), V::Int(7)],
            ),
            (
                call(F::Max, Some(ColumnType::Text), ColumnType::Text),
                vec![
                    V::Text("b".into()),
                    V::Null,
                    V::Text("zz".into()),
                    V::Text("a".into()),
                ],
            ),
        ];
        for (c, values) in cases {
            let mut whole = Acc::default();
            for v in &values {
                fold_value(&mut whole, &c, v.clone(), &Vec::new()).unwrap();
            }
            let expected = finalize_aggregate(whole, &c).unwrap();
            for split in 0..=values.len() {
                let (a, b) = values.split_at(split);
                let mut left = Acc::default();
                for v in a {
                    fold_value(&mut left, &c, v.clone(), &Vec::new()).unwrap();
                }
                let mut right = Acc::default();
                for v in b {
                    fold_value(&mut right, &c, v.clone(), &Vec::new()).unwrap();
                }
                merge_acc(&mut left, right, &c).unwrap();
                let got = finalize_aggregate(left, &c).unwrap();
                assert_eq!(got, expected, "{:?} split at {split}", c.func);
            }
        }
        // COUNT(*): the argument-less tally merges as a plain sum.
        let star = call(F::Count, None, ColumnType::Int);
        let mut whole = Acc::default();
        for _ in 0..7 {
            super::fold_count_star(&mut whole);
        }
        let expected = finalize_aggregate(whole, &star).unwrap();
        let (mut left, mut right) = (Acc::default(), Acc::default());
        for _ in 0..3 {
            super::fold_count_star(&mut left);
        }
        for _ in 0..4 {
            super::fold_count_star(&mut right);
        }
        merge_acc(&mut left, right, &star).unwrap();
        assert_eq!(finalize_aggregate(left, &star).unwrap(), expected);
        // A non-mergeable call is refused loudly, never silently mis-merged.
        let avg = call(F::Avg, Some(ColumnType::Int), ColumnType::Float);
        assert!(merge_acc(&mut Acc::default(), Acc::default(), &avg).is_err());
    }
}
