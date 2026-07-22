//! Cost estimation: predicate selectivity and operator cardinality from
//! ANALYZE statistics (technique from the design
//! `Research_Result/nusadb_cardinality_estimation_sketches.md` §2).
//!
//! Given the per-column [`ColumnStats`] that the `stats` module produced (NDV, MCV
//! list, equi-depth histogram, null count) plus the table's authoritative row
//! count, this module estimates:
//!
//! - **equality** `col = v` — the MCV frequency when `v` is a heavy hitter, else
//!   the residual `(1 − Σmcv)/NDV_rest` spread over the non-MCV distinct values;
//! - **range** `col < v` (and `<=`/`>`/`>=`) — interpolation over the equi-depth
//!   histogram boundaries;
//! - boolean combinations (`AND` = product, `OR` = inclusion–exclusion, `NOT` =
//!   complement, `IN` = Σ equality, `BETWEEN` = range difference, `IS NULL`).
//!
//! Estimates are **deterministic** (they only read the deterministic stats), so
//! a plan's estimated cardinality never changes between runs for the same data
//! — the plan-stability guarantee requires. The estimator is consumed
//! today by `EXPLAIN` (it annotates each operator with its estimated row count);
//! cost-based plan *selection* (index-vs-seq, join order) plugs into the same
//! numbers once those alternative plans exist.

#![allow(
    clippy::cast_precision_loss,
    reason = "estimation math crosses integer<->float by design; inputs are bounded by row counts"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "inclusion-exclusion a+b-a*b is clearer than a fused mul_add here"
)]

use nusadb_core::{ColumnStats, ColumnType, TableSchema, TableStats};

use super::row;
use super::stats::value_cmp;
use crate::ast;
use crate::planner::{PhysicalOperator, SetOpTree, TypedExpr, TypedExprKind};

/// Fallback equality selectivity when stats are missing (a common default).
const DEFAULT_EQ: f64 = 0.005;
/// Fallback range selectivity when the histogram is unusable.
const DEFAULT_RANGE: f64 = 1.0 / 3.0;
/// Fallback selectivity for predicates the estimator does not model.
const DEFAULT_SELECTIVITY: f64 = 0.25;

/// A single table's decoded statistics, the context selectivity is estimated in.
///
/// Column references in a single-table predicate are ordinals into this table's
/// schema; `column` maps an ordinal to its stats by name.
pub(crate) struct ScanStats<'a> {
    schema: &'a TableSchema,
    stats: &'a TableStats,
}

impl<'a> ScanStats<'a> {
    pub(crate) const fn new(schema: &'a TableSchema, stats: &'a TableStats) -> Self {
        Self { schema, stats }
    }

    /// Authoritative live row count at analyze time.
    pub(crate) const fn row_count(&self) -> f64 {
        self.stats.row_count as f64
    }

    /// Stats + type for the column at ordinal `index`, if analyzed.
    fn column(&self, index: usize) -> Option<(&ColumnStats, ColumnType)> {
        let def = self.schema.columns.get(index)?;
        let cs = self.stats.columns.iter().find(|c| c.column == def.name)?;
        Some((cs, def.ty))
    }
}

/// Relative per-tuple costs. A *random* tuple fetch (an index lookup resolving a row by
/// `tid`) is several times pricier than a *sequential* read, and per-tuple CPU work (evaluating a
/// predicate / projecting / hashing) is cheap next to I/O. The absolute units do not matter — only
/// the ratios, which decide between alternative plans.
const SEQ_TUPLE_COST: f64 = 1.0;
const RANDOM_TUPLE_COST: f64 = 4.0;
const CPU_TUPLE_COST: f64 = 0.1;

/// Whether a cost-based planner should prefer an index scan over a sequential scan for a single
/// table of `rows` rows when the index bound selects a `selectivity` fraction of them.
///
/// A sequential scan reads every row once (`rows · SEQ`); an index scan random-fetches only the
/// matched rows (`rows · selectivity · RANDOM`). The index wins once it touches few enough rows —
/// the classic "use the index only when selective" rule, here derived from the cost ratio rather
/// than a hard-coded threshold.
#[must_use]
pub(crate) fn prefers_index_scan(rows: f64, selectivity: f64) -> bool {
    let seq_cost = rows * SEQ_TUPLE_COST;
    let index_cost = rows * selectivity.clamp(0.0, 1.0) * RANDOM_TUPLE_COST;
    index_cost < seq_cost
}

/// Estimated number of rows operator `op` produces. Single-table plans use
/// `ctx` for selectivity; without stats (or for multi-table plans, where
/// `ctx` is `None`) it falls back to default selectivities and passthrough
/// cardinalities.
pub(crate) fn estimate_rows(op: &PhysicalOperator, ctx: Option<&ScanStats>) -> f64 {
    match op {
        // An index scan only returns rows that exist, so the table row count is a safe upper bound;
        // a bound-aware estimate is a refinement once histograms drive index selection.
        PhysicalOperator::SeqScan { table, .. } | PhysicalOperator::IndexScan { table, .. } => ctx
            .filter(|c| c.schema.id == table.id)
            .map_or(0.0, ScanStats::row_count),
        PhysicalOperator::Filter { input, predicate } => {
            let inrows = estimate_rows(input, ctx);
            let sel = ctx.map_or(DEFAULT_SELECTIVITY, |c| selectivity(predicate, c));
            inrows * sel
        },
        PhysicalOperator::Limit {
            input,
            count,
            offset,
        } => {
            #[allow(
                clippy::cast_precision_loss,
                reason = "LIMIT/OFFSET counts beyond 2^52 are not realistic"
            )]
            (estimate_rows(input, ctx) - *offset as f64)
                .max(0.0)
                .min(*count as f64)
        },
        PhysicalOperator::OneRow
        | PhysicalOperator::InfoSchemaScan { .. }
        | PhysicalOperator::ScalarAggregate { .. } => 1.0,
        // A `(VALUES ...)` source emits exactly its literal rows.
        PhysicalOperator::Values { rows } => rows.len() as f64,
        // A `(SELECT ... UNION ...)` source: upper-bound its cardinality by summing the branches.
        PhysicalOperator::SetOperation(set_op) => estimate_set_op_rows(&set_op.tree, ctx),
        PhysicalOperator::GroupAggregate {
            input, group_keys, ..
        } => estimate_groups(input, group_keys, ctx),
        // One pass per grouping set; coarsely scale the full-key group estimate by the set count.
        PhysicalOperator::GroupingSetsAggregate {
            input,
            group_keys,
            grouping_sets,
            ..
        } => estimate_groups(input, group_keys, ctx) * grouping_sets.len().max(1) as f64,
        // A set-returning projection expands each row to ≥0 rows; with no SRF cardinality estimate
        // available, pass the input estimate through as a coarse lower bound.
        PhysicalOperator::Sort { input, .. }
        | PhysicalOperator::Project { input, .. }
        | PhysicalOperator::ProjectSet { input, .. }
        | PhysicalOperator::Distinct { input }
        | PhysicalOperator::DistinctOn { input, .. }
        // `LockRows` (FOR UPDATE/SHARE) returns its input unchanged.
        | PhysicalOperator::LockRows { input, .. }
        // Window appends columns; the row count is unchanged.
        | PhysicalOperator::Window { input, .. } => estimate_rows(input, ctx),
        // Multi-table shapes: a coarse product / sum keeps EXPLAIN populated even
        // though `ctx` (single-table) cannot resolve their concatenated ordinals.
        PhysicalOperator::NestedLoopJoin { left, right, .. }
        | PhysicalOperator::HashJoin { left, right, .. }
        | PhysicalOperator::LateralJoin { left, right, .. } => {
            estimate_rows(left, ctx).max(estimate_rows(right, ctx))
        },
        // The result cardinality of a recursive query is its body's; the synthetic CTE tables have
        // no single-table `ctx`, so this falls back to the default passthrough.
        PhysicalOperator::WithRecursive { body, .. }
        | PhysicalOperator::WithModifying { body, .. } => estimate_rows(body, ctx),
        // A k-NN search returns at most `k` rows.
        #[allow(
            clippy::cast_precision_loss,
            reason = "a LIMIT k beyond 2^52 is not realistic"
        )]
        PhysicalOperator::VectorKnn { k, table, .. } => {
            let table_rows = ctx
                .filter(|c| c.schema.id == table.id)
                .map_or(f64::INFINITY, ScanStats::row_count);
            (*k as f64).min(table_rows)
        },
    }
}

/// Upper-bound the row count of a set-operation tree by summing its leaves (the `UNION ALL` bound;
/// `INTERSECT`/`EXCEPT`/distinct `UNION` only shrink the result). Used to size a set-op FROM source.
fn estimate_set_op_rows(tree: &SetOpTree<PhysicalOperator>, ctx: Option<&ScanStats>) -> f64 {
    match tree {
        SetOpTree::Leaf(op) => estimate_rows(op, ctx),
        SetOpTree::Node { left, right, .. } => {
            estimate_set_op_rows(left, ctx) + estimate_set_op_rows(right, ctx)
        },
    }
}

/// Estimated total cost of executing `op` (and its inputs), in the relative units above.
///
/// I/O dominates: a `SeqScan` pays one sequential read per row; an `IndexScan` random-fetches its
/// matched rows; `Sort` adds an `n·log n` comparison term; a `NestedLoopJoin` is the quadratic
/// `left·right` shape a `HashJoin` avoids. Every operator adds cheap per-tuple CPU. EXPLAIN reports
/// this, and the planner compares it to choose between alternative plans.
pub(crate) fn estimate_cost(op: &PhysicalOperator, ctx: Option<&ScanStats>) -> f64 {
    match op {
        // A `(VALUES ...)` / `(SELECT ... UNION ...)` source scans its rows, costed like a sequential
        // scan over the produced cardinality.
        PhysicalOperator::SeqScan { .. }
        | PhysicalOperator::Values { .. }
        | PhysicalOperator::SetOperation(_) => estimate_rows(op, ctx) * SEQ_TUPLE_COST,
        // Random-fetch each matched row; `estimate_rows` upper-bounds the matched set.
        PhysicalOperator::IndexScan { .. } => estimate_rows(op, ctx) * RANDOM_TUPLE_COST,
        PhysicalOperator::OneRow | PhysicalOperator::InfoSchemaScan { .. } => 0.0,
        // Every linear single-input operator costs its input plus cheap per-tuple CPU.
        PhysicalOperator::Filter { input, .. }
        | PhysicalOperator::Project { input, .. }
        | PhysicalOperator::ProjectSet { input, .. }
        | PhysicalOperator::Distinct { input }
        | PhysicalOperator::DistinctOn { input, .. }
        | PhysicalOperator::Window { input, .. }
        | PhysicalOperator::LockRows { input, .. }
        | PhysicalOperator::ScalarAggregate { input, .. }
        | PhysicalOperator::GroupAggregate { input, .. }
        | PhysicalOperator::GroupingSetsAggregate { input, .. }
        | PhysicalOperator::Limit { input, .. } => {
            estimate_cost(input, ctx) + estimate_rows(input, ctx) * CPU_TUPLE_COST
        },
        PhysicalOperator::Sort { input, .. } => {
            let n = estimate_rows(input, ctx).max(1.0);
            estimate_cost(input, ctx) + n * n.log2() * CPU_TUPLE_COST
        },
        // Hash join builds + probes linearly; nested-loop pays the row product.
        PhysicalOperator::HashJoin { left, right, .. } => {
            estimate_cost(left, ctx)
                + estimate_cost(right, ctx)
                + (estimate_rows(left, ctx) + estimate_rows(right, ctx)) * CPU_TUPLE_COST
        },
        PhysicalOperator::NestedLoopJoin { left, right, .. } => {
            estimate_cost(left, ctx)
                + estimate_cost(right, ctx)
                + estimate_rows(left, ctx) * estimate_rows(right, ctx) * CPU_TUPLE_COST
        },
        // A lateral join re-executes the right side once per left row, so it pays the right's cost
        // per left row plus the row product (like a nested loop whose inner is rebuilt each pass).
        PhysicalOperator::LateralJoin { left, right, .. } => {
            let left_rows = estimate_rows(left, ctx);
            estimate_cost(left, ctx)
                + left_rows * estimate_cost(right, ctx)
                + left_rows * estimate_rows(right, ctx) * CPU_TUPLE_COST
        },
        PhysicalOperator::WithRecursive { ctes, body } => {
            let cte_cost: f64 = ctes
                .iter()
                .map(|c| estimate_cost(&c.base, ctx) + estimate_cost(&c.recursive, ctx))
                .sum();
            cte_cost + estimate_cost(body, ctx)
        },
        // The data-modifying statements run separately; approximate the query cost by the body's.
        PhysicalOperator::WithModifying { body, .. } => estimate_cost(body, ctx),
        // A k-NN search scans the table (to build/serve the index) and returns `k` rows; cost
        // is dominated by the scan, approximated by the table row count when stats are available.
        PhysicalOperator::VectorKnn { table, .. } => {
            ctx.filter(|c| c.schema.id == table.id)
                .map_or(0.0, ScanStats::row_count)
                * SEQ_TUPLE_COST
        },
    }
}

/// Estimate the number of groups a `GROUP BY` produces: the product of the
/// group-key columns' NDVs (capped at the input rows), or the input rows when a
/// key is not a plain analyzed column.
fn estimate_groups(
    input: &PhysicalOperator,
    group_keys: &[TypedExpr],
    ctx: Option<&ScanStats>,
) -> f64 {
    let inrows = estimate_rows(input, ctx);
    let Some(ctx) = ctx else { return inrows };
    let mut groups = 1.0_f64;
    for key in group_keys {
        let ndv = match &key.kind {
            #[allow(
                clippy::cast_precision_loss,
                reason = "NDV beyond 2^52 is not a realistic estimation input"
            )]
            TypedExprKind::Column(index) => ctx
                .column(*index)
                .map_or(inrows, |(cs, _)| cs.distinct_count.max(1) as f64),
            _ => return inrows,
        };
        groups *= ndv;
    }
    groups.min(inrows).max(1.0)
}

/// Estimated selectivity (fraction of rows kept) of a boolean predicate, in
/// `[0, 1]`.
pub(crate) fn selectivity(pred: &TypedExpr, ctx: &ScanStats) -> f64 {
    match &pred.kind {
        TypedExprKind::Literal(ast::Value::Bool(true)) => 1.0,
        TypedExprKind::Literal(ast::Value::Bool(false)) => 0.0,
        TypedExprKind::Binary { left, op, right } => binary_selectivity(left, *op, right, ctx),
        TypedExprKind::Unary {
            op: ast::UnaryOp::Not,
            expr,
        } => 1.0 - selectivity(expr, ctx),
        TypedExprKind::IsNull { expr, negated } => null_selectivity(expr, *negated, ctx),
        TypedExprKind::InList {
            expr,
            list,
            negated,
        } => in_list_selectivity(expr, list, *negated, ctx),
        TypedExprKind::Between {
            expr,
            low,
            high,
            negated,
        } => between_selectivity(expr, low, high, *negated, ctx),
        _ => DEFAULT_SELECTIVITY,
    }
}

fn binary_selectivity(
    left: &TypedExpr,
    op: ast::BinaryOp,
    right: &TypedExpr,
    ctx: &ScanStats,
) -> f64 {
    use ast::BinaryOp::{And, Eq, Gt, GtEq, Lt, LtEq, NotEq, Or};
    match op {
        And => selectivity(left, ctx) * selectivity(right, ctx),
        Or => {
            let a = selectivity(left, ctx);
            let b = selectivity(right, ctx);
            a + b - a * b
        },
        Eq | NotEq | Lt | LtEq | Gt | GtEq => {
            let Some((index, value, flipped)) = column_and_literal(left, right) else {
                return DEFAULT_SELECTIVITY;
            };
            let Some((cs, ty)) = ctx.column(index) else {
                return comparison_default(op);
            };
            comparison_selectivity(cs, ty, normalize_op(op, flipped), value, ctx.row_count())
        },
        _ => DEFAULT_SELECTIVITY,
    }
}

/// Selectivity of `col <op> literal` once the column/literal have been resolved.
fn comparison_selectivity(
    cs: &ColumnStats,
    ty: ColumnType,
    op: ast::BinaryOp,
    value: &ast::Value,
    rows: f64,
) -> f64 {
    use ast::BinaryOp::{Eq, Gt, GtEq, Lt, LtEq, NotEq};
    match op {
        Eq => equality_selectivity(cs, ty, value, rows),
        NotEq => 1.0 - equality_selectivity(cs, ty, value, rows),
        Lt | LtEq => frac_below(cs, ty, value).unwrap_or(DEFAULT_RANGE),
        Gt | GtEq => 1.0 - frac_below(cs, ty, value).unwrap_or(1.0 - DEFAULT_RANGE),
        _ => DEFAULT_SELECTIVITY,
    }
}

/// `col = value`: exact MCV frequency when present, else the residual mass
/// `(rows − Σmcv) / rows` divided evenly over the non-MCV distinct values.
fn equality_selectivity(cs: &ColumnStats, ty: ColumnType, value: &ast::Value, rows: f64) -> f64 {
    if rows <= 0.0 {
        return 0.0;
    }
    let mut mcv_total = 0.0_f64;
    for (bytes, freq) in &cs.most_common {
        #[allow(
            clippy::cast_precision_loss,
            reason = "frequencies are bounded by the row count"
        )]
        let f = *freq as f64;
        mcv_total += f;
        if decode_one(bytes, ty).is_some_and(|v| value_cmp(&v, value).is_eq()) {
            return (f / rows).clamp(0.0, 1.0);
        }
    }
    let ndv_rest = cs
        .distinct_count
        .saturating_sub(cs.most_common.len() as u64)
        .max(1);
    #[allow(
        clippy::cast_precision_loss,
        reason = "NDV beyond 2^52 is not a realistic estimation input"
    )]
    let ndv_rest = ndv_rest as f64;
    let nonmcv_rows = (rows - mcv_total).max(0.0);
    (nonmcv_rows / rows / ndv_rest).clamp(0.0, 1.0)
}

/// Fraction of rows whose value is below `value`, interpolated over the
/// equi-depth histogram. `None` when there is no usable histogram.
fn frac_below(cs: &ColumnStats, ty: ColumnType, value: &ast::Value) -> Option<f64> {
    if cs.histogram.len() < 2 {
        return None;
    }
    let bounds: Vec<ast::Value> = cs
        .histogram
        .iter()
        .filter_map(|b| decode_one(b, ty))
        .collect();
    if bounds.len() < 2 {
        return None;
    }
    let buckets = (bounds.len() - 1) as f64;
    let first = bounds.first()?;
    let last = bounds.last()?;
    if value_cmp(value, first).is_le() {
        return Some(0.0);
    }
    if value_cmp(value, last).is_ge() {
        return Some(1.0);
    }
    for i in 0..bounds.len() - 1 {
        let lo = bounds.get(i)?;
        let hi = bounds.get(i + 1)?;
        if value_cmp(value, hi).is_lt() {
            let within = interpolate(lo, hi, value);
            #[allow(
                clippy::cast_precision_loss,
                reason = "bucket index is small (<= histogram bucket count)"
            )]
            let base = i as f64;
            return Some(((base + within) / buckets).clamp(0.0, 1.0));
        }
    }
    Some(1.0)
}

/// Position of `value` within bucket `[lo, hi)` as a fraction in `[0, 1)`. Pure
/// numeric interpolation for `Int`/`Float`; non-numeric values step at the
/// bucket's lower edge (fraction `0`).
fn interpolate(lo: &ast::Value, hi: &ast::Value, value: &ast::Value) -> f64 {
    let (Some(l), Some(h), Some(v)) = (numeric(lo), numeric(hi), numeric(value)) else {
        return 0.0;
    };
    if h <= l {
        return 0.0;
    }
    ((v - l) / (h - l)).clamp(0.0, 1.0)
}

fn numeric(v: &ast::Value) -> Option<f64> {
    match v {
        #[allow(
            clippy::cast_precision_loss,
            reason = "estimation interpolation tolerates the i64->f64 rounding"
        )]
        ast::Value::Int(i) => Some(*i as f64),
        ast::Value::Float(f) => Some(*f),
        ast::Value::Bool(b) => Some(f64::from(*b)),
        _ => None,
    }
}

fn null_selectivity(expr: &TypedExpr, negated: bool, ctx: &ScanStats) -> f64 {
    let rows = ctx.row_count();
    if rows <= 0.0 {
        return 0.0;
    }
    let frac = match &expr.kind {
        #[allow(
            clippy::cast_precision_loss,
            reason = "null counts are bounded by the row count"
        )]
        TypedExprKind::Column(index) => ctx
            .column(*index)
            .map_or(0.0, |(cs, _)| (cs.null_count as f64 / rows).clamp(0.0, 1.0)),
        _ => return DEFAULT_SELECTIVITY,
    };
    if negated { 1.0 - frac } else { frac }
}

fn in_list_selectivity(
    expr: &TypedExpr,
    list: &[TypedExpr],
    negated: bool,
    ctx: &ScanStats,
) -> f64 {
    let Some(index) = column_ordinal(expr) else {
        return DEFAULT_SELECTIVITY;
    };
    let Some((cs, ty)) = ctx.column(index) else {
        return DEFAULT_SELECTIVITY;
    };
    let rows = ctx.row_count();
    let mut sel = 0.0_f64;
    for item in list {
        if let TypedExprKind::Literal(v) = &item.kind {
            sel += equality_selectivity(cs, ty, v, rows);
        }
    }
    sel = sel.clamp(0.0, 1.0);
    if negated { 1.0 - sel } else { sel }
}

fn between_selectivity(
    expr: &TypedExpr,
    low: &TypedExpr,
    high: &TypedExpr,
    negated: bool,
    ctx: &ScanStats,
) -> f64 {
    let sel = match (column_ordinal(expr), &low.kind, &high.kind) {
        (Some(index), TypedExprKind::Literal(lo), TypedExprKind::Literal(hi)) => {
            match ctx.column(index) {
                Some((cs, ty)) => {
                    let below_hi = frac_below(cs, ty, hi).unwrap_or(1.0 - DEFAULT_RANGE);
                    let below_lo = frac_below(cs, ty, lo).unwrap_or(DEFAULT_RANGE);
                    (below_hi - below_lo).clamp(0.0, 1.0)
                },
                None => DEFAULT_RANGE,
            }
        },
        _ => DEFAULT_RANGE,
    };
    if negated { 1.0 - sel } else { sel }
}

/// Destructure `col <op> literal` (or the flipped `literal <op> col`). Returns
/// the column ordinal, the literal value, and whether the operands were flipped.
const fn column_and_literal<'a>(
    left: &'a TypedExpr,
    right: &'a TypedExpr,
) -> Option<(usize, &'a ast::Value, bool)> {
    match (&left.kind, &right.kind) {
        (TypedExprKind::Column(index), TypedExprKind::Literal(v)) => Some((*index, v, false)),
        (TypedExprKind::Literal(v), TypedExprKind::Column(index)) => Some((*index, v, true)),
        _ => None,
    }
}

const fn column_ordinal(expr: &TypedExpr) -> Option<usize> {
    match expr.kind {
        TypedExprKind::Column(index) => Some(index),
        _ => None,
    }
}

/// Flip a comparison operator when the column was on the right (`5 < col` is
/// `col > 5`).
const fn normalize_op(op: ast::BinaryOp, flipped: bool) -> ast::BinaryOp {
    use ast::BinaryOp::{Gt, GtEq, Lt, LtEq};
    if !flipped {
        return op;
    }
    match op {
        Lt => Gt,
        LtEq => GtEq,
        Gt => Lt,
        GtEq => LtEq,
        other => other,
    }
}

const fn comparison_default(op: ast::BinaryOp) -> f64 {
    use ast::BinaryOp::{Eq, NotEq};
    match op {
        Eq => DEFAULT_EQ,
        NotEq => 1.0 - DEFAULT_EQ,
        _ => DEFAULT_RANGE,
    }
}

/// Decode a single opaque stat value (min/max/MCV/histogram entry) back to a
/// [`ast::Value`].
fn decode_one(bytes: &[u8], ty: ColumnType) -> Option<ast::Value> {
    row::decode(bytes, &[ty]).ok()?.into_iter().next()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::unnecessary_box_returns,
    clippy::cast_possible_wrap,
    reason = "test helpers: exact determinism check + Box<TypedExpr> builders + small-int casts"
)]
mod tests {
    use super::{ScanStats, estimate_rows, selectivity};
    use crate::ast;
    use crate::executor::stats::column_stats;
    use crate::planner::{PhysicalOperator, TypedExpr, TypedExprKind};
    use nusadb_core::{ColumnDef, ColumnType, TableId, TableSchema, TableStats};

    fn schema() -> TableSchema {
        TableSchema {
            schema: "public".to_owned(),
            id: TableId(1),
            name: "t".to_owned(),
            columns: vec![
                ColumnDef {
                    name: "id".to_owned(),
                    ty: ColumnType::Int,
                    nullable: false,
                },
                ColumnDef {
                    name: "cat".to_owned(),
                    ty: ColumnType::Int,
                    nullable: true,
                },
            ],
        }
    }

    /// Build `TableStats` by analyzing the given id/cat columns.
    fn stats_for(ids: &[i64], cats: &[Option<i64>]) -> TableStats {
        let id_vals: Vec<ast::Value> = ids.iter().map(|&i| ast::Value::Int(i)).collect();
        let cat_vals: Vec<ast::Value> = cats
            .iter()
            .map(|c| c.map_or(ast::Value::Null, ast::Value::Int))
            .collect();
        TableStats {
            row_count: ids.len() as u64,
            page_count: 0,
            columns: vec![
                column_stats("id", &id_vals, ColumnType::Int).unwrap(),
                column_stats("cat", &cat_vals, ColumnType::Int).unwrap(),
            ],
        }
    }

    fn col(index: usize) -> Box<TypedExpr> {
        Box::new(TypedExpr {
            kind: TypedExprKind::Column(index),
            ty: ColumnType::Int,
        })
    }

    fn lit(i: i64) -> Box<TypedExpr> {
        Box::new(TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Int(i)),
            ty: ColumnType::Int,
        })
    }

    fn binary(left: Box<TypedExpr>, op: ast::BinaryOp, right: Box<TypedExpr>) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary { left, op, right },
            ty: ColumnType::Bool,
        }
    }

    /// q-error = max(est/act, act/est); 1.0 is perfect.
    fn q_error(est: f64, act: f64) -> f64 {
        if est <= 0.0 || act <= 0.0 {
            return f64::INFINITY;
        }
        (est / act).max(act / est)
    }

    #[test]
    fn equality_on_unique_column_estimates_one_row() {
        let s = schema();
        let st = stats_for(&[1, 2, 3, 4, 5, 6, 7, 8], &[None; 8]);
        let ctx = ScanStats::new(&s, &st);
        let pred = binary(col(0), ast::BinaryOp::Eq, lit(3));
        // 8 distinct ids → selectivity ~ 1/8; estimated rows ~ 1, actual = 1.
        let est = 8.0 * selectivity(&pred, &ctx);
        assert!(q_error(est, 1.0) < 1.5, "est={est}");
    }

    #[test]
    fn equality_on_skewed_column_uses_mcv() {
        let s = schema();
        // cat = 7 appears 100x; ten other values once each.
        let mut cats: Vec<Option<i64>> = vec![Some(7); 100];
        cats.extend((0..10).map(Some));
        let ids: Vec<i64> = (0..cats.len() as i64).collect();
        let st = stats_for(&ids, &cats);
        let ctx = ScanStats::new(&s, &st);
        let n = cats.len() as f64;
        // Heavy hitter: estimate ~100 rows.
        let hot = binary(col(1), ast::BinaryOp::Eq, lit(7));
        let est_hot = n * selectivity(&hot, &ctx);
        assert!(q_error(est_hot, 100.0) < 1.2, "hot est={est_hot}");
        // Cold value: estimate ~1 row.
        let cold = binary(col(1), ast::BinaryOp::Eq, lit(3));
        let est_cold = n * selectivity(&cold, &ctx);
        assert!(q_error(est_cold, 1.0) < 3.0, "cold est={est_cold}");
    }

    #[test]
    fn range_uses_histogram() {
        let s = schema();
        let ids: Vec<i64> = (0..1000).collect();
        let st = stats_for(&ids, &[None; 1000]);
        let ctx = ScanStats::new(&s, &st);
        // id < 250 → ~25% of 1000 = 250 rows actual.
        let pred = binary(col(0), ast::BinaryOp::Lt, lit(250));
        let est = 1000.0 * selectivity(&pred, &ctx);
        assert!(q_error(est, 250.0) < 1.3, "est={est}");
    }

    #[test]
    fn range_flipped_operands() {
        let s = schema();
        let ids: Vec<i64> = (0..1000).collect();
        let st = stats_for(&ids, &[None; 1000]);
        let ctx = ScanStats::new(&s, &st);
        // 750 < id  ≡  id > 750  → ~25% → 250 rows.
        let pred = binary(lit(750), ast::BinaryOp::Lt, col(0));
        let est = 1000.0 * selectivity(&pred, &ctx);
        assert!(q_error(est, 250.0) < 1.4, "est={est}");
    }

    #[test]
    fn and_is_product_or_is_inclusion_exclusion() {
        let s = schema();
        let ids: Vec<i64> = (0..100).collect();
        let st = stats_for(&ids, &[None; 100]);
        let ctx = ScanStats::new(&s, &st);
        let a = binary(col(0), ast::BinaryOp::Lt, lit(50)); // ~0.5
        let b = binary(col(0), ast::BinaryOp::Lt, lit(25)); // ~0.25
        let and = binary(Box::new(a.clone()), ast::BinaryOp::And, Box::new(b.clone()));
        let or = binary(Box::new(a), ast::BinaryOp::Or, Box::new(b));
        let sa = 0.5;
        let sb = 0.25;
        assert!((selectivity(&and, &ctx) - sa * sb).abs() < 0.05);
        assert!((selectivity(&or, &ctx) - (sa + sb - sa * sb)).abs() < 0.05);
    }

    #[test]
    fn is_null_uses_null_fraction() {
        let s = schema();
        // 30 of 100 cat values are NULL.
        let cats: Vec<Option<i64>> = (0..100)
            .map(|i| if i < 30 { None } else { Some(i) })
            .collect();
        let ids: Vec<i64> = (0..100).collect();
        let st = stats_for(&ids, &cats);
        let ctx = ScanStats::new(&s, &st);
        let is_null = TypedExpr {
            kind: TypedExprKind::IsNull {
                expr: col(1),
                negated: false,
            },
            ty: ColumnType::Bool,
        };
        assert!((selectivity(&is_null, &ctx) - 0.30).abs() < 0.01);
    }

    #[test]
    fn estimate_rows_walks_filter_and_limit() {
        let s = schema();
        let ids: Vec<i64> = (0..1000).collect();
        let st = stats_for(&ids, &[None; 1000]);
        let ctx = ScanStats::new(&s, &st);
        let scan = PhysicalOperator::SeqScan {
            table: s.clone(),
            columns: Vec::new(),
        };
        let filtered = PhysicalOperator::Filter {
            input: Box::new(scan),
            predicate: binary(col(0), ast::BinaryOp::Lt, lit(100)), // ~10%
        };
        let limited = PhysicalOperator::Limit {
            input: Box::new(filtered),
            count: 25,
            offset: 0,
        };
        // ~100 rows after filter, capped to 25 by LIMIT.
        assert!((estimate_rows(&limited, Some(&ctx)) - 25.0).abs() < 0.001);
    }

    #[test]
    fn estimate_is_deterministic() {
        let s = schema();
        let ids: Vec<i64> = (0..500).collect();
        let st = stats_for(&ids, &[None; 500]);
        let ctx = ScanStats::new(&s, &st);
        let pred = binary(col(0), ast::BinaryOp::Lt, lit(123));
        assert_eq!(selectivity(&pred, &ctx), selectivity(&pred, &ctx));
    }

    #[test]
    fn prefers_index_scan_only_when_selective() {
        // Very selective bound (0.1% of rows) → the index wins.
        assert!(super::prefers_index_scan(10_000.0, 0.001));
        // A bound that keeps most rows → a sequential scan is cheaper.
        assert!(!super::prefers_index_scan(10_000.0, 0.9));
        // The crossover sits below the random/sequential cost ratio (1/4).
        assert!(super::prefers_index_scan(1000.0, 0.2));
        assert!(!super::prefers_index_scan(1000.0, 0.3));
    }

    #[test]
    fn estimate_cost_grows_with_work() {
        let s = schema();
        let ids: Vec<i64> = (0..1000).collect();
        let st = stats_for(&ids, &[None; 1000]);
        let ctx = ScanStats::new(&s, &st);
        let scan = PhysicalOperator::SeqScan {
            table: s.clone(),
            columns: Vec::new(),
        };
        let scan_cost = super::estimate_cost(&scan, Some(&ctx));
        // A sequential scan costs about one unit per row.
        assert!((scan_cost - 1000.0).abs() < 1.0, "scan_cost={scan_cost}");
        // Adding a Filter on top only increases the cost (it never decreases).
        let filtered = PhysicalOperator::Filter {
            input: Box::new(scan),
            predicate: binary(col(0), ast::BinaryOp::Lt, lit(100)),
        };
        assert!(super::estimate_cost(&filtered, Some(&ctx)) > scan_cost);
    }

    #[test]
    fn missing_stats_fall_back_to_defaults() {
        let s = schema();
        let st = TableStats {
            row_count: 100,
            page_count: 0,
            columns: Vec::new(), // no analyzed columns
        };
        let ctx = ScanStats::new(&s, &st);
        let pred = binary(col(0), ast::BinaryOp::Eq, lit(3));
        let sel = selectivity(&pred, &ctx);
        assert!(sel > 0.0 && sel < 1.0);
    }
}
