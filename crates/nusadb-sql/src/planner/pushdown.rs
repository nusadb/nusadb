//! Projection pushdown / column pruning: narrow a single-table scan to
//! only the columns its pipeline actually reads, so the executor builds rows
//! holding just those columns instead of the full table width.
//!
//! # Why only the simple single-table shape
//!
//! A `Column(ord)` ordinal is meaningful only relative to the row layout its
//! operator consumes. That layout is the base table's column space *only* while
//! the pipeline neither joins (which concatenates two tables' columns into one
//! row, so an ordinal may mean column *k* of the right table, not the left) nor
//! aggregates / windows (which redefine the row as group-keys-plus-results or
//! append window outputs). The moment any of those appears, the same ordinal
//! refers to a different column — so blindly pruning a scan underneath them and
//! rewriting ordinals would silently corrupt results.
//!
//! Rather than track every column space, this pass runs only when the whole
//! plan is the shape where the scan's table layout reaches every reference
//! unchanged: a single [`PhysicalOperator::SeqScan`] leaf with nothing above it
//! but row-preserving, table-ordinal-space operators (`Filter`, `Sort`,
//! `DistinctOn`, `Project`/`ProjectSet`, `Distinct`, `Limit`). Joins,
//! aggregation, windows, `IndexScan`, `LockRows` (which re-scans the *full* base
//! row to lock it), recursive CTEs, etc. disqualify the plan and it is left
//! untouched — a safe, honest no-op for those.
//!
//! In the qualifying shape, every `Column` reference indexes the base table, so
//! the pass: (1) collects the referenced ordinals, (2) if a strict non-empty
//! subset is used, builds the old→new ordinal map, (3) rewrites every reference
//! to the narrowed layout, and (4) records the kept ordinals on the `SeqScan`.
//! A correlated subquery could reference the scan's row via `OuterColumn` (whose
//! ordinals this pass does not rewrite), so any subquery anywhere in the scope
//! disqualifies the plan too.

use super::plan_types::{PhysicalOperator, TypedExpr, TypedExprKind};
use std::collections::{BTreeSet, HashMap};

/// Apply projection pushdown to `op` in place when it is the simple single-table
/// shape (or an aggregate over one); otherwise leave it unchanged.
pub(super) fn pushdown_projection(op: &mut PhysicalOperator) {
    pushdown_simple(op);
    pushdown_under_aggregate(op);
}

/// The original single-table pass: the scan's table layout reaches every reference unchanged.
fn pushdown_simple(op: &mut PhysicalOperator) {
    // (1) Confirm the qualifying shape and learn the base table's column count.
    let Some(table_len) = simple_scan_len(op) else {
        return;
    };
    // An already-narrowed scan means the pass ran on this subtree before (an inlined view body is
    // pushed down when the view's own plan was built): its references are in the *narrowed* space,
    // so re-collecting them and overwriting the kept list would corrupt the layout. Idempotence
    // guard: never run twice.
    if scan_is_narrowed(op) {
        return;
    }
    // A `Project`/`ProjectSet` redefines the column space, so references *above* it are no longer in
    // the table's ordinal space. A plain single-table SELECT has exactly one (its select list); a
    // second one means an inlined CTE / view / derived table sits below, whose output space the
    // upper projection indexes — mixing spaces would corrupt the remap. Bail in that case.
    if project_count(op) > 1 {
        return;
    }
    // (2) Collect the table columns referenced anywhere above the scan.
    let mut needed = BTreeSet::new();
    let mut has_subquery = false;
    collect_op(op, &mut needed, &mut has_subquery);
    // A correlated subquery may read this scan's row through `OuterColumn`, whose
    // ordinals are not rewritten here — bail rather than risk a stale reference.
    if has_subquery {
        return;
    }
    // Every reference must land inside the base table (guaranteed by the shape,
    // checked defensively: an out-of-range ordinal would not be remapped).
    if needed.iter().any(|&c| c >= table_len) {
        return;
    }
    // (3) Prune only when a strict, non-empty subset is read — a full or empty
    // set means there is nothing to gain (and an empty set is the `SELECT 1`
    // case, left on the full path for simplicity).
    let kept: Vec<usize> = needed.into_iter().collect();
    if kept.is_empty() || kept.len() == table_len {
        return;
    }
    // (4) old table ordinal → its index in the narrowed row.
    let remap: HashMap<usize, usize> = kept
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    remap_op(op, &remap);
    set_scan_columns(op, kept);
}

/// Narrow the scan under an aggregate: a `ScalarAggregate` /
/// `GroupAggregate` / `GroupingSetsAggregate` redefines its *output* space (so the whole-plan pass
/// above bails), but its **input** subtree is still in the base table's ordinal space, and the
/// aggregate's own per-row expressions (call args, `FILTER` predicates, in-aggregate `ORDER BY`,
/// group keys) index that same input space. So: collect the ordinals those expressions and the
/// input chain reference, narrow the scan, and remap them together — everything *above* the
/// aggregate indexes its output layout (group keys ++ results), which does not depend on how the
/// input rows are laid out, and is left untouched.
///
/// The payoff is largest exactly where QA measured the scan being slow at scale: a full-table
/// `count(*)` referenced no column at all yet decoded every column of every row; it now decodes
/// only the table's first column (the row codec must still walk to *some* column, so ordinal 0 is
/// the cheapest), and `sum(one_col)` decodes one column instead of the full width.
fn pushdown_under_aggregate(op: &mut PhysicalOperator) {
    use PhysicalOperator as O;
    let Some(agg) = find_aggregate_mut(op) else {
        return;
    };
    // The aggregate's input must be the clean single-table chain, with no Project/ProjectSet
    // below (an inlined view/CTE/derived table starts a fresh column space).
    let (input, exprs_len) = match agg {
        O::ScalarAggregate { input, .. }
        | O::GroupAggregate { input, .. }
        | O::GroupingSetsAggregate { input, .. } => {
            let Some(table_len) = simple_scan_len(input) else {
                return;
            };
            if project_count(input) > 0 {
                return;
            }
            // Idempotence guard (see `pushdown_simple`): an inlined view body over an aggregate
            // arrives already narrowed — its expressions index the narrowed space, so running
            // again would remap them a second time and overwrite the kept list with narrowed
            // ordinals (grouping by the wrong columns).
            if scan_is_narrowed(input) {
                return;
            }
            (&**input, table_len)
        },
        _ => return,
    };
    let table_len = exprs_len;
    // Collect from the input chain plus the aggregate's own input-space expressions.
    let mut needed = BTreeSet::new();
    let mut has_subquery = false;
    collect_op(input, &mut needed, &mut has_subquery);
    for_each_aggregate_expr(agg, &mut |expr| {
        collect_expr(expr, &mut needed, &mut has_subquery);
    });
    if has_subquery || needed.iter().any(|&c| c >= table_len) {
        return;
    }
    // `count(*)` references nothing: decode only the first column (the codec must decode at least
    // one; ordinal 0 stops the tuple walk immediately). Otherwise prune to the referenced subset.
    let kept: Vec<usize> = if needed.is_empty() {
        vec![0]
    } else {
        needed.into_iter().collect()
    };
    if kept.len() >= table_len {
        return;
    }
    let remap: HashMap<usize, usize> = kept
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    let (O::ScalarAggregate { input, .. }
    | O::GroupAggregate { input, .. }
    | O::GroupingSetsAggregate { input, .. }) = agg
    else {
        return;
    };
    remap_op(input, &remap);
    for_each_aggregate_expr(agg, &mut |expr| remap_expr_cb(expr, &remap));
    let (O::ScalarAggregate { input, .. }
    | O::GroupAggregate { input, .. }
    | O::GroupingSetsAggregate { input, .. }) = agg
    else {
        return;
    };
    set_scan_columns(input, kept);
}

/// Descend from the plan root to the first aggregate operator, passing only through operators
/// whose expressions index the *current* (post-aggregate) space — they are unaffected by
/// narrowing the aggregate's input. Anything else (joins, window, a second aggregate, scans)
/// disqualifies the plan.
fn find_aggregate_mut(op: &mut PhysicalOperator) -> Option<&mut PhysicalOperator> {
    use PhysicalOperator as O;
    match op {
        O::ScalarAggregate { .. } | O::GroupAggregate { .. } | O::GroupingSetsAggregate { .. } => {
            Some(op)
        },
        O::Filter { input, .. }
        | O::Sort { input, .. }
        | O::Project { input, .. }
        | O::ProjectSet { input, .. }
        | O::Distinct { input }
        | O::DistinctOn { input, .. }
        | O::Limit { input, .. } => find_aggregate_mut(input),
        _ => None,
    }
}

/// Visit every input-space expression an aggregate operator owns: each call's argument(s) and
/// `FILTER`, the in-aggregate `ORDER BY` keys, and the group keys. (`grouping_args` are indices
/// into `group_keys`, not table ordinals — nothing to visit.)
fn for_each_aggregate_expr(agg: &mut PhysicalOperator, f: &mut dyn FnMut(&mut TypedExpr)) {
    use PhysicalOperator as O;
    let (calls, group_keys) = match agg {
        O::ScalarAggregate { calls, .. } => (calls, None),
        O::GroupAggregate {
            calls, group_keys, ..
        }
        | O::GroupingSetsAggregate {
            calls, group_keys, ..
        } => (calls, Some(group_keys)),
        _ => return,
    };
    for call in calls {
        if let Some(arg) = &mut call.arg {
            f(arg);
        }
        if let Some(arg2) = &mut call.arg2 {
            f(arg2);
        }
        if let Some(filter) = &mut call.filter {
            f(filter);
        }
        for key in &mut call.order_by {
            f(&mut key.expr);
        }
    }
    if let Some(keys) = group_keys {
        for key in keys {
            f(key);
        }
    }
}

/// `remap_expr` with the collect-callback shape (immutable collect vs mutable remap share
/// [`for_each_aggregate_expr`] via a mutable visitor).
fn remap_expr_cb(expr: &mut TypedExpr, map: &HashMap<usize, usize>) {
    remap_expr(expr, map);
}

/// Whether the chain's `SeqScan` already carries a narrowed column list (a prior run of this
/// pass — e.g. an inlined view body): the idempotence guard for both passes.
fn scan_is_narrowed(op: &PhysicalOperator) -> bool {
    use PhysicalOperator as O;
    match op {
        O::SeqScan { columns, .. } => !columns.is_empty(),
        O::Filter { input, .. }
        | O::Sort { input, .. }
        | O::Project { input, .. }
        | O::ProjectSet { input, .. }
        | O::Distinct { input }
        | O::DistinctOn { input, .. }
        | O::Limit { input, .. } => scan_is_narrowed(input),
        _ => false,
    }
}

/// `Some(table.columns.len())` when `op` is one `SeqScan` reached only through
/// row-preserving, table-ordinal-space operators; `None` otherwise.
fn simple_scan_len(op: &PhysicalOperator) -> Option<usize> {
    use PhysicalOperator as O;
    match op {
        O::SeqScan { table, .. } => Some(table.columns.len()),
        O::Filter { input, .. }
        | O::Sort { input, .. }
        | O::Project { input, .. }
        | O::ProjectSet { input, .. }
        | O::Distinct { input }
        | O::DistinctOn { input, .. }
        | O::Limit { input, .. } => simple_scan_len(input),
        // Anything else (joins, aggregation, window, IndexScan, InfoSchemaScan,
        // VectorKnn, OneRow, LockRows, recursive CTE) redefines or duplicates the
        // column space — not a single clean table layout.
        _ => None,
    }
}

/// Count the `Project`/`ProjectSet` operators in a simple-shape chain (each one starts a fresh
/// column space). Only called on a chain `simple_scan_len` already accepted.
fn project_count(op: &PhysicalOperator) -> usize {
    use PhysicalOperator as O;
    match op {
        O::Project { input, .. } | O::ProjectSet { input, .. } => 1 + project_count(input),
        O::Filter { input, .. }
        | O::Sort { input, .. }
        | O::Distinct { input }
        | O::DistinctOn { input, .. }
        | O::Limit { input, .. } => project_count(input),
        _ => 0,
    }
}

// ── Collect referenced table columns ──────────────────────────────────────

fn collect_op(op: &PhysicalOperator, cols: &mut BTreeSet<usize>, has_sq: &mut bool) {
    use PhysicalOperator as O;
    match op {
        O::Filter { input, predicate } => {
            collect_expr(predicate, cols, has_sq);
            collect_op(input, cols, has_sq);
        },
        O::Sort { input, keys, .. } => {
            for k in keys {
                collect_expr(&k.expr, cols, has_sq);
            }
            collect_op(input, cols, has_sq);
        },
        O::Project { input, columns } | O::ProjectSet { input, columns, .. } => {
            for p in columns {
                collect_expr(&p.expr, cols, has_sq);
            }
            collect_op(input, cols, has_sq);
        },
        O::DistinctOn { input, keys } => {
            for k in keys {
                collect_expr(k, cols, has_sq);
            }
            collect_op(input, cols, has_sq);
        },
        O::Distinct { input } | O::Limit { input, .. } => collect_op(input, cols, has_sq),
        // Unreachable for a shape `simple_scan_len` accepted; treat as opaque.
        _ => {},
    }
}

fn collect_expr(expr: &TypedExpr, cols: &mut BTreeSet<usize>, has_sq: &mut bool) {
    use TypedExprKind as K;
    match &expr.kind {
        K::Column(ord) => {
            cols.insert(*ord);
        },
        // Subqueries (and the outer-row references they may carry) take this plan
        // out of the prunable shape — flag and stop descending into them.
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. } => *has_sq = true,
        K::Literal(_) | K::OuterColumn { .. } | K::AggregateRef(_) => {},
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            collect_expr(left, cols, has_sq);
            collect_expr(right, cols, has_sq);
        },
        K::QuantifiedArray { expr, array, .. } => {
            collect_expr(expr, cols, has_sq);
            collect_expr(array, cols, has_sq);
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => collect_expr(inner, cols, has_sq),
        K::InList {
            expr: inner, list, ..
        } => {
            collect_expr(inner, cols, has_sq);
            for item in list {
                collect_expr(item, cols, has_sq);
            }
        },
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            collect_expr(inner, cols, has_sq);
            collect_expr(low, cols, has_sq);
            collect_expr(high, cols, has_sq);
        },
        K::Like {
            expr: inner,
            pattern,
            ..
        }
        | K::RegexMatch {
            expr: inner,
            pattern,
            ..
        }
        | K::SimilarTo {
            expr: inner,
            pattern,
            ..
        } => {
            collect_expr(inner, cols, has_sq);
            collect_expr(pattern, cols, has_sq);
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(o) = operand {
                collect_expr(o, cols, has_sq);
            }
            for b in branches {
                collect_expr(&b.when, cols, has_sq);
                collect_expr(&b.then, cols, has_sq);
            }
            if let Some(d) = default {
                collect_expr(d, cols, has_sq);
            }
        },
        K::Coalesce(args)
        | K::ArrayLiteral(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::SetReturning { args, .. } => {
            for a in args {
                collect_expr(a, cols, has_sq);
            }
        },
        K::Crypto { value, key, .. } => {
            collect_expr(value, cols, has_sq);
            collect_expr(key, cols, has_sq);
        },
        K::Subscript { base, index } => {
            collect_expr(base, cols, has_sq);
            collect_expr(index, cols, has_sq);
        },
        K::ArraySlice { base, lower, upper } => {
            collect_expr(base, cols, has_sq);
            for bound in [lower, upper].into_iter().flatten() {
                collect_expr(bound, cols, has_sq);
            }
        },
    }
}

// ── Rewrite references onto the narrowed layout ───────────────────────────

fn remap_op(op: &mut PhysicalOperator, map: &HashMap<usize, usize>) {
    use PhysicalOperator as O;
    match op {
        O::Filter { input, predicate } => {
            remap_expr(predicate, map);
            remap_op(input, map);
        },
        O::Sort { input, keys, .. } => {
            for k in keys {
                remap_expr(&mut k.expr, map);
            }
            remap_op(input, map);
        },
        O::Project { input, columns } | O::ProjectSet { input, columns, .. } => {
            for p in columns {
                remap_expr(&mut p.expr, map);
            }
            remap_op(input, map);
        },
        O::DistinctOn { input, keys } => {
            for k in keys {
                remap_expr(k, map);
            }
            remap_op(input, map);
        },
        O::Distinct { input } | O::Limit { input, .. } => remap_op(input, map),
        _ => {},
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per TypedExprKind variant; length tracks the expression grammar"
)]
fn remap_expr(expr: &mut TypedExpr, map: &HashMap<usize, usize>) {
    use TypedExprKind as K;
    match &mut expr.kind {
        K::Column(ord) => {
            // Every collected ordinal is in the map; leave anything else (which
            // cannot occur in the qualifying shape) untouched.
            if let Some(&new) = map.get(ord) {
                *ord = new;
            }
        },
        // Subquery-free shape only (the pass bailed otherwise), so these never
        // appear here; keep the match exhaustive without descending into plans.
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. }
        | K::Literal(_)
        | K::OuterColumn { .. }
        | K::AggregateRef(_) => {},
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            remap_expr(left, map);
            remap_expr(right, map);
        },
        K::QuantifiedArray { expr, array, .. } => {
            remap_expr(expr, map);
            remap_expr(array, map);
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => remap_expr(inner, map),
        K::InList {
            expr: inner, list, ..
        } => {
            remap_expr(inner, map);
            for item in list {
                remap_expr(item, map);
            }
        },
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            remap_expr(inner, map);
            remap_expr(low, map);
            remap_expr(high, map);
        },
        K::Like {
            expr: inner,
            pattern,
            ..
        }
        | K::RegexMatch {
            expr: inner,
            pattern,
            ..
        }
        | K::SimilarTo {
            expr: inner,
            pattern,
            ..
        } => {
            remap_expr(inner, map);
            remap_expr(pattern, map);
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(o) = operand {
                remap_expr(o, map);
            }
            for b in branches {
                remap_expr(&mut b.when, map);
                remap_expr(&mut b.then, map);
            }
            if let Some(d) = default {
                remap_expr(d, map);
            }
        },
        K::Coalesce(args)
        | K::ArrayLiteral(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::SetReturning { args, .. } => {
            for a in args {
                remap_expr(a, map);
            }
        },
        K::Crypto { value, key, .. } => {
            remap_expr(value, map);
            remap_expr(key, map);
        },
        K::Subscript { base, index } => {
            remap_expr(base, map);
            remap_expr(index, map);
        },
        K::ArraySlice { base, lower, upper } => {
            remap_expr(base, map);
            for bound in [lower, upper].into_iter().flatten() {
                remap_expr(bound, map);
            }
        },
    }
}

/// Record the kept ordinals on the single `SeqScan` at the bottom of the chain.
fn set_scan_columns(op: &mut PhysicalOperator, kept: Vec<usize>) {
    use PhysicalOperator as O;
    match op {
        O::SeqScan { columns, .. } => *columns = kept,
        O::Filter { input, .. }
        | O::Sort { input, .. }
        | O::Project { input, .. }
        | O::ProjectSet { input, .. }
        | O::Distinct { input }
        | O::DistinctOn { input, .. }
        | O::Limit { input, .. } => set_scan_columns(input, kept),
        _ => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast;
    use crate::planner::plan_types::{Projection, TypedExpr, TypedExprKind};
    use nusadb_core::engine::{ColumnDef, TableSchema};
    use nusadb_core::{ColumnType, TableId};

    /// A three-column table `t(a, b, c)`.
    fn table_t() -> TableSchema {
        let col = |name: &str| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Int,
            nullable: true,
        };
        TableSchema {
            schema: "public".to_owned(),
            id: TableId(1),
            name: "t".to_owned(),
            columns: vec![col("a"), col("b"), col("c")],
        }
    }

    fn column(ord: usize) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Column(ord),
            ty: ColumnType::Int,
        }
    }

    fn scan() -> PhysicalOperator {
        PhysicalOperator::SeqScan {
            table: table_t(),
            columns: Vec::new(),
        }
    }

    fn proj(input: PhysicalOperator, exprs: Vec<TypedExpr>) -> PhysicalOperator {
        PhysicalOperator::Project {
            input: Box::new(input),
            columns: exprs
                .into_iter()
                .map(|expr| Projection {
                    expr,
                    name: "x".to_owned(),
                })
                .collect(),
        }
    }

    /// Pull the `columns` recorded on the (single) `SeqScan` of a chain.
    fn scan_columns(op: &PhysicalOperator) -> &[usize] {
        use PhysicalOperator as O;
        match op {
            O::SeqScan { columns, .. } => columns,
            O::Filter { input, .. }
            | O::Sort { input, .. }
            | O::Project { input, .. }
            | O::Distinct { input }
            | O::Limit { input, .. } => scan_columns(input),
            other => panic!("no scan under {other:?}"),
        }
    }

    /// An [`AggregateCall`] with only `func`/`arg` set (the fields this pass visits).
    fn agg_call(
        func: ast::AggregateFunc,
        arg: Option<TypedExpr>,
    ) -> crate::planner::plan_types::AggregateCall {
        crate::planner::plan_types::AggregateCall {
            func,
            arg,
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

    #[test]
    fn count_star_scan_narrows_to_the_first_column() {
        // SELECT count(*) FROM t — no column referenced: decode only ordinal 0,
        // not the full three-column width.
        let mut op = proj(
            PhysicalOperator::ScalarAggregate {
                input: Box::new(scan()),
                calls: vec![agg_call(ast::AggregateFunc::Count, None)],
            },
            vec![column(0)], // references the aggregate's OUTPUT slot 0, untouched by the pass
        );
        pushdown_projection(&mut op);
        let PhysicalOperator::Project { input, columns } = &op else {
            panic!("expected Project over the aggregate");
        };
        assert_eq!(
            columns[0].expr.kind,
            TypedExprKind::Column(0),
            "output-space reference must not be remapped"
        );
        let PhysicalOperator::ScalarAggregate { input, .. } = &**input else {
            panic!("expected ScalarAggregate");
        };
        assert_eq!(scan_columns(input), &[0]);
    }

    #[test]
    fn group_aggregate_scan_narrows_and_remaps_input_space() {
        // SELECT b, sum(c) FROM t GROUP BY b — reads {b, c} = {1, 2}; remap 1→0, 2→1 in the
        // aggregate's own expressions (group key + call arg).
        let mut op = PhysicalOperator::GroupAggregate {
            input: Box::new(scan()),
            group_keys: vec![column(1)],
            calls: vec![agg_call(ast::AggregateFunc::Sum, Some(column(2)))],
        };
        pushdown_projection(&mut op);
        let PhysicalOperator::GroupAggregate {
            input,
            group_keys,
            calls,
        } = &op
        else {
            panic!("expected GroupAggregate");
        };
        assert_eq!(scan_columns(input), &[1, 2]);
        assert_eq!(group_keys[0].kind, TypedExprKind::Column(0));
        assert_eq!(
            calls[0].arg.as_ref().map(|a| &a.kind),
            Some(&TypedExprKind::Column(1))
        );
    }

    #[test]
    fn pushdown_is_idempotent_over_an_already_narrowed_aggregate() {
        // An inlined view body arrives already pushed down; a second run (the outer query's
        // lowering) must be a no-op — re-collecting narrowed-space ordinals used to overwrite the
        // kept list and group by the wrong columns (the p11_views/plain regression).
        let mut op = PhysicalOperator::GroupAggregate {
            input: Box::new(scan()),
            group_keys: vec![column(1)],
            calls: vec![agg_call(ast::AggregateFunc::Sum, Some(column(2)))],
        };
        pushdown_projection(&mut op);
        let first = format!("{op:?}");
        pushdown_projection(&mut op);
        assert_eq!(format!("{op:?}"), first, "second run must change nothing");
    }

    #[test]
    fn aggregate_over_full_width_reference_stays_untouched() {
        // Every column referenced → nothing to prune.
        let mut op = PhysicalOperator::GroupAggregate {
            input: Box::new(scan()),
            group_keys: vec![column(0), column(1)],
            calls: vec![agg_call(ast::AggregateFunc::Sum, Some(column(2)))],
        };
        pushdown_projection(&mut op);
        let PhysicalOperator::GroupAggregate { input, .. } = &op else {
            panic!("expected GroupAggregate");
        };
        let empty: &[usize] = &[];
        assert_eq!(scan_columns(input), empty);
    }

    #[test]
    fn prunes_and_remaps_a_simple_projection() {
        // SELECT c FROM t  →  only column 2 is read; it remaps to ordinal 0 of the narrowed row.
        let mut op = proj(scan(), vec![column(2)]);
        pushdown_projection(&mut op);
        assert_eq!(scan_columns(&op), &[2]);
        let PhysicalOperator::Project { columns, .. } = &op else {
            panic!("expected Project");
        };
        assert_eq!(columns[0].expr.kind, TypedExprKind::Column(0));
    }

    #[test]
    fn keeps_filter_and_projection_columns_and_remaps_both() {
        // SELECT c FROM t WHERE a > 0  →  reads {a, c} = {0, 2}; remap 0→0, 2→1.
        let filter = PhysicalOperator::Filter {
            input: Box::new(scan()),
            predicate: TypedExpr {
                kind: TypedExprKind::Binary {
                    left: Box::new(column(0)),
                    op: ast::BinaryOp::Gt,
                    right: Box::new(TypedExpr {
                        kind: TypedExprKind::Literal(ast::Value::Int(0)),
                        ty: ColumnType::Int,
                    }),
                },
                ty: ColumnType::Bool,
            },
        };
        let mut op = proj(filter, vec![column(2)]);
        pushdown_projection(&mut op);
        assert_eq!(scan_columns(&op), &[0, 2]);
        // Projection's `c` (was 2) → 1; filter's `a` (was 0) → 0.
        let PhysicalOperator::Project { columns, input } = &op else {
            panic!("expected Project");
        };
        assert_eq!(columns[0].expr.kind, TypedExprKind::Column(1));
        let PhysicalOperator::Filter { predicate, .. } = input.as_ref() else {
            panic!("expected Filter");
        };
        let TypedExprKind::Binary { left, .. } = &predicate.kind else {
            panic!("expected Binary");
        };
        assert_eq!(left.kind, TypedExprKind::Column(0));
    }

    #[test]
    fn no_prune_when_all_columns_referenced() {
        // SELECT a, b, c FROM t  →  full width, nothing to prune.
        let mut op = proj(scan(), vec![column(0), column(1), column(2)]);
        pushdown_projection(&mut op);
        assert!(scan_columns(&op).is_empty());
    }

    #[test]
    fn no_prune_under_a_join() {
        // A join concatenates column spaces, so the pass must leave the scan untouched even though
        // the projection names a single ordinal.
        let join = PhysicalOperator::NestedLoopJoin {
            left: Box::new(scan()),
            right: Box::new(scan()),
            predicate: TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Bool(true)),
                ty: ColumnType::Bool,
            },
            kind: ast::JoinKind::Inner,
            left_width: 3,
            right_width: 3,
            coalesce_pairs: Vec::new(),
        };
        let mut op = proj(join, vec![column(0)]);
        pushdown_projection(&mut op);
        // Both scans keep full width (empty = identity).
        let PhysicalOperator::Project { input, .. } = &op else {
            panic!("expected Project");
        };
        let PhysicalOperator::NestedLoopJoin { left, right, .. } = input.as_ref() else {
            panic!("expected join");
        };
        assert!(scan_columns(left).is_empty());
        assert!(scan_columns(right).is_empty());
    }

    #[test]
    fn no_prune_with_two_projections() {
        // An inlined CTE / view / derived table adds a second Project, so the outer projection's
        // ordinals index the inner projection's output — a different space. Pruning must bail, or
        // the remap would corrupt cross-boundary references.
        let inner = proj(scan(), vec![column(0), column(2)]);
        let mut op = proj(inner, vec![column(0)]);
        pushdown_projection(&mut op);
        assert!(scan_columns(&op).is_empty());
    }
}
