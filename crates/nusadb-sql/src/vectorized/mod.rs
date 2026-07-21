//! Vectorized execution operators — the column-at-a-time counterpart to the
//! row-at-a-time [`crate::executor`].
//!
//! Each operator is a pull-based ("Volcano") node over [`RecordBatch`] streams: a parent
//! repeatedly calls [`Operator::next_batch`] on its child until it returns `Ok(None)`.
//! Batches carry up to [`BATCH_SIZE`](crate::BATCH_SIZE) rows, so per-call dispatch is
//! amortized over the whole batch rather than paid per row.
//!
//! The leaf is [`SeqScan`], which turns a table scan into a batch stream via the
//! [`crate::batch::RecordBatchScan`] adapter; [`Filter`] drops rows that
//! fail a predicate; [`Project`] rewrites the columns by evaluating expressions.
//! [`Limit`] applies `OFFSET`/`LIMIT`; [`Sort`] orders rows by sort keys.
//! Later tasks add joins and aggregation on top of the same [`Operator`] contract.

mod aggregate;
mod filter;
mod group_aggregate;
mod limit;
mod parallel;
mod project;
mod seq_scan;
mod simd;
mod sort;

pub use aggregate::ScalarAggregate;
pub use filter::Filter;
pub use group_aggregate::GroupedAggregate;
pub use limit::Limit;
pub use parallel::{ParallelGroupedAggregate, fold_count, parallel_scope};
pub use project::Project;
pub use seq_scan::SeqScan;
pub use sort::Sort;

use std::cell::Cell;
use std::sync::Arc;

use nusadb_core::{StorageEngine, TxnId};

use crate::batch::{RecordBatch, Schema};
use crate::error::Error;
use crate::executor::row::Row;
use crate::planner::{PhysicalOperator, TypedExpr, TypedExprKind};

thread_local! {
    /// Whether the SELECT executor routes a supported plan through the vectorized (batch) path
    /// instead of the row-at-a-time path. Opt-in, default off (wiring): the batch operators
    /// still evaluate per row (SIMD kernels are a follow-up), so enabling it trades the row path's
    /// directness for the batch path without a speedup yet — kept off until those kernels land.
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Whether vectorized SELECT execution is currently enabled on this thread.
#[must_use]
pub fn is_enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Enable (or disable) the vectorized SELECT path for the lifetime of the returned guard, restoring
/// the previous setting on drop. The default is off.
#[must_use]
pub fn scope(enabled: bool) -> EnabledGuard {
    let previous = ENABLED.with(|c| c.replace(enabled));
    EnabledGuard { previous }
}

/// Restores the previous [`is_enabled`] setting on drop.
#[derive(Debug)]
pub struct EnabledGuard {
    previous: bool,
}

impl Drop for EnabledGuard {
    fn drop(&mut self) {
        ENABLED.with(|c| c.set(self.previous));
    }
}

/// Execute `op` through the vectorized path when its shape is fully supported, returning the result
/// rows. `Ok(None)` means the plan is not vectorizable (the caller falls back to the row path); the
/// produced rows are identical to that path — only the execution strategy differs.
///
/// # Errors
/// Propagates any scan/evaluation error from running the operator tree.
pub(crate) fn execute(
    op: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
    est_scan_rows: Option<u64>,
) -> Result<Option<Vec<Row>>, Error> {
    let Some(mut root) = try_build(op, engine, txn, est_scan_rows)? else {
        return Ok(None);
    };
    let mut rows = Vec::new();
    while let Some(batch) = root.next_batch()? {
        rows.extend(crate::batch::convert::batch_to_rows(&batch));
    }
    Ok(Some(rows))
}

/// Translate a physical SELECT subtree into a vectorized operator tree, or `Ok(None)` if any node or
/// expression is outside what the vectorized operators support (then the caller uses the row path).
/// Supported: `SeqScan`, `Filter`, `Project`, `Limit`, `Sort`, `ScalarAggregate`, and
/// `GroupAggregate` over subquery-free expressions.
fn try_build(
    op: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
    est_scan_rows: Option<u64>,
) -> Result<Option<Box<dyn Operator>>, Error> {
    let built: Box<dyn Operator> = match op {
        // A projection-pushdown-narrowed scan yields rows whose layout the row path's
        // rewritten ordinals expect; the columnar scan decodes the full table width, so fall back
        // to the row path rather than apply the projection here.
        PhysicalOperator::SeqScan { table, columns } => {
            if !columns.is_empty() {
                return Ok(None);
            }
            Box::new(SeqScan::open(engine, txn, table)?)
        },
        PhysicalOperator::Filter { input, predicate } => {
            if !expr_is_vectorizable(predicate) {
                return Ok(None);
            }
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            Box::new(Filter::new(child, predicate.clone()))
        },
        PhysicalOperator::Project { input, columns } => {
            // Directly above an aggregate node, a bare `AggregateRef(i)` reads output ordinal
            // `i` of the synthesized `[keys.., aggs..]` row — the row evaluator's two arms are
            // identical — so it rewrites to `Column(i)` and vectorizes. Without this the
            // projection every aggregate plan carries kept ALL grouped/scalar aggregates off
            // the vectorized path. A nested `AggregateRef` (e.g. `COUNT(*) + 1`) still falls
            // back below.
            let columns = resolve_projection_agg_refs(columns, input);
            if !columns.iter().all(|c| expr_is_vectorizable(&c.expr)) {
                return Ok(None);
            }
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            Box::new(Project::new(child, columns))
        },
        PhysicalOperator::Sort {
            input,
            keys,
            limit_ties,
            top_n,
        } => {
            // The `WITH TIES` tie trim is only implemented on the row path, so
            // fall back to it rather than vectorizing a sort that would drop the trailing peers.
            if limit_ties.is_some() {
                return Ok(None);
            }
            let keys = resolve_sort_agg_refs(keys, input);
            if !keys.iter().all(|k| expr_is_vectorizable(&k.expr)) {
                return Ok(None);
            }
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            // Limit-aware top-N: select the first `m` rows via a bounded
            // partial selection instead of a full sort — result-identical to the full sort's first
            // `m` rows. `top_n` is capped in the planner, so `m` fits `usize`.
            let top_n = top_n.and_then(|m| usize::try_from(m).ok());
            Box::new(Sort::new(child, keys, top_n))
        },
        PhysicalOperator::Limit {
            input,
            count,
            offset,
        } => {
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            let limit = (*count != u64::MAX).then(|| usize::try_from(*count).unwrap_or(usize::MAX));
            let offset = usize::try_from(*offset).unwrap_or(usize::MAX);
            Box::new(Limit::new(child, offset, limit))
        },
        // Scalar aggregate (no GROUP BY): SIMD-reduces eligible COUNT/SUM/MIN/MAX over a plain
        // column and folds the rest on the row path. Every call's argument and
        // FILTER must be subquery-free so that row-path fallback can evaluate them.
        PhysicalOperator::ScalarAggregate { input, calls } => {
            // A scalar aggregate over a (filtered) table scan folds on parallel
            // workers under the same gates as the grouped form — one merged group, one output
            // row (empty input included). A gate refusal falls through to the sequential path.
            if let Some(par) =
                try_parallel_aggregate(input, &[], calls, true, engine, txn, est_scan_rows)?
            {
                return Ok(Some(par));
            }
            let exprs_ok = calls.iter().all(|c| {
                c.arg.as_ref().is_none_or(expr_is_vectorizable)
                    && c.filter.as_ref().is_none_or(expr_is_vectorizable)
            });
            if !exprs_ok {
                return Ok(None);
            }
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            Box::new(ScalarAggregate::new(child, calls.clone()))
        },
        // Grouped aggregate (GROUP BY, A-PERF.AGG6 / F2c): the vectorized hash group-by covers
        // bare-column keys + columnar-foldable calls, sharing the row path's GroupIndex + fold
        // machinery so the output multiset AND first-seen emission order are identical. When
        // spill-to-disk is configured, the row path's bounded-memory sort-based group-by is
        // authoritative → fall back rather than hold O(groups) state past the budget.
        PhysicalOperator::GroupAggregate {
            input,
            group_keys,
            calls,
        } => {
            // A grouped aggregate directly over a table scan (pushdown-narrowed or not)
            // with a large-enough estimate folds on parallel workers — bit-identical output, so
            // a gate refusal just falls through. Tried BEFORE the spill bail: under a spill
            // budget `try_new` engages only when the ANALYZE statistics bound the group count
            // (its hash state then provably stays tiny), so the bounded-memory contract holds;
            // every other spill case still defers to the sort-based row path.
            if let Some(par) =
                try_parallel_aggregate(input, group_keys, calls, false, engine, txn, est_scan_rows)?
            {
                return Ok(Some(par));
            }
            if crate::executor::spill_is_configured() {
                return Ok(None);
            }
            let Some(child) = try_build(input, engine, txn, est_scan_rows)? else {
                return Ok(None);
            };
            let Some(grouped) = GroupedAggregate::try_new(child, group_keys, calls.clone()) else {
                return Ok(None);
            };
            Box::new(grouped)
        },
        // Everything else (IndexScan, OneRow, joins, grouping-sets aggregation, window, DISTINCT,
        // set-returning, recursive CTE) has no vectorized operator yet → fall back to the row path.
        _ => return Ok(None),
    };
    Ok(Some(built))
}

/// The R5 parallel attempt shared by the grouped and scalar aggregate arms: engage
/// [`ParallelGroupedAggregate`] when the aggregate sits over a (filtered) table scan and every
/// gate passes, else `None` (the caller builds the sequential operator).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the aggregate plan shapes' own field set"
)]
fn try_parallel_aggregate(
    input: &PhysicalOperator,
    group_keys: &[TypedExpr],
    calls: &[crate::planner::AggregateCall],
    scalar: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
    est_scan_rows: Option<u64>,
) -> Result<Option<Box<dyn Operator>>, Error> {
    let Some((table, columns, predicate)) = scan_with_filter(input) else {
        return Ok(None);
    };
    let Some(par) = ParallelGroupedAggregate::try_new(
        engine,
        txn,
        table,
        columns,
        predicate,
        group_keys,
        calls.to_vec(),
        scalar,
        est_scan_rows,
    )?
    else {
        return Ok(None);
    };
    let boxed: Box<dyn Operator> = Box::new(par);
    Ok(Some(boxed))
}

/// Destructure an aggregate's input as a table scan with an optional pushed-down filter:
/// `SeqScan` or `Filter { SeqScan }` — the two shapes the parallel aggregate can fold on
/// workers. `None` for anything else.
fn scan_with_filter(
    op: &PhysicalOperator,
) -> Option<(
    &nusadb_core::engine::TableSchema,
    &[usize],
    Option<&TypedExpr>,
)> {
    match op {
        PhysicalOperator::SeqScan { table, columns } => Some((table, columns, None)),
        PhysicalOperator::Filter { input, predicate } => match &**input {
            PhysicalOperator::SeqScan { table, columns } => Some((table, columns, Some(predicate))),
            _ => None,
        },
        _ => None,
    }
}

/// Rewrite each bare `AggregateRef(slot)` projection over a post-aggregation row to
/// `Column(slot)` — the row evaluator's two arms are identical there (see
/// [`yields_aggregate_row`]). Anything else (including a nested ref) is kept, and the
/// vectorizability check after decides its fate.
fn resolve_projection_agg_refs(
    columns: &[crate::planner::Projection],
    input: &PhysicalOperator,
) -> Vec<crate::planner::Projection> {
    let over_aggregate = yields_aggregate_row(input);
    columns
        .iter()
        .map(|c| match c.expr.kind {
            TypedExprKind::AggregateRef(slot) if over_aggregate => crate::planner::Projection {
                expr: TypedExpr {
                    kind: TypedExprKind::Column(slot),
                    ty: c.expr.ty,
                },
                name: c.name.clone(),
            },
            _ => c.clone(),
        })
        .collect()
}

/// The ORDER-BY-key counterpart of [`resolve_projection_agg_refs`]: above an aggregate, sort
/// keys reference the synthesized row via bare `AggregateRef`s — rewrite them to columns.
fn resolve_sort_agg_refs(
    keys: &[crate::planner::OrderByKey],
    input: &PhysicalOperator,
) -> Vec<crate::planner::OrderByKey> {
    let over_aggregate = yields_aggregate_row(input);
    keys.iter()
        .map(|k| match k.expr.kind {
            TypedExprKind::AggregateRef(slot) if over_aggregate => crate::planner::OrderByKey {
                expr: TypedExpr {
                    kind: TypedExprKind::Column(slot),
                    ty: k.expr.ty,
                },
                ..k.clone()
            },
            _ => k.clone(),
        })
        .collect()
}

/// Whether `op` yields the synthesized post-aggregation row — the aggregate node itself, or a
/// layout-preserving operator (`Sort`/`Limit`) over one. Against that row a bare
/// `AggregateRef(i)` reads output ordinal `i`, exactly the row evaluator's `Column(i)` arm, so
/// the two kinds rewrite freely (see the `Project`/`Sort` arms above).
fn yields_aggregate_row(op: &PhysicalOperator) -> bool {
    match op {
        PhysicalOperator::GroupAggregate { .. } | PhysicalOperator::ScalarAggregate { .. } => true,
        PhysicalOperator::Sort { input, .. } | PhysicalOperator::Limit { input, .. } => {
            yields_aggregate_row(input)
        },
        _ => false,
    }
}

/// Whether `expr` can be evaluated by the vectorized operators, which reuse the row evaluator and so
/// cannot resolve subqueries (those must run on the row path that pre-resolves them). Exhaustive, so
/// a new expression kind forces an explicit decision here.
fn expr_is_vectorizable(expr: &TypedExpr) -> bool {
    use TypedExprKind as K;
    match &expr.kind {
        // Subqueries, set-returning calls, outer-column / aggregate references are resolved or
        // expanded by the row path before evaluation — the vectorized evaluator cannot handle them.
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. }
        | K::QuantifiedArray { .. }
        | K::SetReturning { .. }
        | K::OuterColumn { .. }
        // An array slice yields an array; it has no vectorized kernel, so it uses the row path.
        | K::ArraySlice { .. }
        | K::AggregateRef(_) => false,
        K::Literal(_) | K::Column(_) => true,
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            expr_is_vectorizable(left) && expr_is_vectorizable(right)
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => expr_is_vectorizable(inner),
        K::InList { expr, list, .. } => {
            expr_is_vectorizable(expr) && list.iter().all(expr_is_vectorizable)
        },
        K::Between {
            expr, low, high, ..
        } => expr_is_vectorizable(expr) && expr_is_vectorizable(low) && expr_is_vectorizable(high),
        K::Like { expr, pattern, .. }
        | K::RegexMatch { expr, pattern, .. }
        | K::SimilarTo { expr, pattern, .. } => {
            expr_is_vectorizable(expr) && expr_is_vectorizable(pattern)
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(expr_is_vectorizable)
                && branches
                    .iter()
                    .all(|b| expr_is_vectorizable(&b.when) && expr_is_vectorizable(&b.then))
                && default.as_deref().is_none_or(expr_is_vectorizable)
        },
        // A sequence built-in is side-effecting / session-stateful and must be resolved by the row
        // path (`resolve_sequence_calls`, which enforces the single-evaluation rule); it never
        // vectorizes, so a projection carrying one falls back to the row path.
        K::ScalarFunction { func, .. } if func.is_sequence() => false,
        // A scalar UDF is vectorizable iff its arguments are: the vectorized evaluator reuses
        // the row evaluator, which invokes the registered function per row.
        K::Coalesce(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args) => args.iter().all(expr_is_vectorizable),
        K::Crypto { value, key, .. } => expr_is_vectorizable(value) && expr_is_vectorizable(key),
        K::Subscript { base, index } => expr_is_vectorizable(base) && expr_is_vectorizable(index),
    }
}

/// A pull-based vectorized execution node.
///
/// Drive an operator by calling [`Operator::next_batch`] until it yields `Ok(None)`.
/// Every batch it produces matches [`Operator::schema`], and once the stream ends (or
/// errors) the operator is exhausted.
///
/// [`Debug`] is a supertrait so operator trees (which hold child operators as
/// `Box<dyn Operator>`) are printable.
pub trait Operator: std::fmt::Debug {
    /// The schema shared by every [`RecordBatch`] this operator produces.
    fn schema(&self) -> &Arc<Schema>;

    /// Produce the next batch, or `Ok(None)` at end of stream.
    ///
    /// # Errors
    ///
    /// Propagates any decode/storage/evaluation error encountered while producing the batch.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error>;
}
