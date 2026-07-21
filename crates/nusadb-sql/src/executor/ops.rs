//! Query operator execution: SELECT pipeline, set operations, the `execute_op` dispatcher,
//! aggregation, window, and join operators.
//!
//! Split verbatim out of `executor/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

// === SELECT ===============================================================

pub(super) fn run_select(
    op: &PhysicalOperator,
    est_scan_rows: Option<u64>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let columns = output_columns(op);
    // Route to the vectorized batch path when it is force-enabled (the opt-in test/override flag) or
    // when the plan-time scanned-row estimate is large enough to amortize the batch overhead (
    // selective routing, the design recommendation: ≥ VECTORIZED_MIN_ROWS). Otherwise use the row-at-a-time
    // path. Both produce identical rows; `vectorized::execute` itself falls back to the row path for
    // any plan shape it does not support, so an over-eager route is at worst a wasted try. The
    // estimate is computed at plan time, so this adds no run-time stats fetch on the default path.
    let use_batch =
        crate::vectorized::is_enabled() || est_scan_rows.is_some_and(meets_vectorize_threshold);
    let rows = if use_batch {
        match crate::vectorized::execute(op, engine, txn, est_scan_rows)? {
            // The batch path materializes its own result, so enforce the work-memory budget on it
            // too (the row path checks inside `execute_op`).
            Some(rows) => {
                enforce_work_mem(&rows)?;
                rows
            },
            None => execute_op(op, engine, txn)?,
        }
    } else {
        execute_op(op, engine, txn)?
    };
    Ok(ExecutionResult::Rows { columns, rows })
}

/// The statistics-estimated in-memory hash-aggregation state for grouping `input` by
/// `group_keys`: the key columns' NDV product (plus a NULL group per nullable key) times a
/// conservative per-group state size. `None` when the input is not a single analyzed table or
/// any key is not a bare column — the caller then uses the estimate-free sort-based fold.
fn estimated_group_state_bytes(
    input: &PhysicalOperator,
    group_keys: &[TypedExpr],
    engine: &dyn StorageEngine,
) -> Option<u64> {
    /// Key `Vec<Value>` + accumulator vector + hash-index slot, deliberately generous.
    const EST_GROUP_STATE_BYTES: u64 = 360;
    // The ordinals in `group_keys` index `input`'s row layout, which must be exactly the
    // scan's — descend only layout-preserving operators to reach it (audit catch: a
    // Filter-wrapped narrowed scan otherwise read the WRONG column's NDV, and a low-NDV
    // misread would route an over-budget aggregate to the unbounded hash arm). Anything
    // layout-changing refuses: the sort fold needs no estimate.
    let (table, scan_columns) = layout_preserved_scan(input)?;
    let stats = engine.table_stats(table.id).ok()??;
    let mut groups: u64 = 1;
    for key in group_keys {
        let crate::planner::TypedExprKind::Column(ordinal) = key.kind else {
            return None;
        };
        // A pushdown-narrowed scan presents its kept columns; map back to source ordinals.
        let source = if scan_columns.is_empty() {
            ordinal
        } else {
            *scan_columns.get(ordinal)?
        };
        let column = table.columns.get(source)?;
        let cs = stats.columns.iter().find(|c| c.column == column.name)?;
        let ndv = cs
            .distinct_count
            .max(1)
            .saturating_add(u64::from(cs.null_count > 0));
        groups = groups.saturating_mul(ndv);
    }
    Some(groups.saturating_mul(EST_GROUP_STATE_BYTES))
}

/// The table scan whose row layout `op` presents **unchanged**: descends the layout-preserving
/// operators (`Filter`/`Sort`/`Limit`/`Distinct`) only — a `Project` (or anything else that
/// reshapes the row) returns `None`, because ordinals above it no longer index the scan.
fn layout_preserved_scan(
    op: &PhysicalOperator,
) -> Option<(&nusadb_core::engine::TableSchema, &[usize])> {
    match op {
        PhysicalOperator::SeqScan { table, columns } => Some((table, columns)),
        PhysicalOperator::Filter { input, .. }
        | PhysicalOperator::Sort { input, .. }
        | PhysicalOperator::Limit { input, .. }
        | PhysicalOperator::Distinct { input, .. } => layout_preserved_scan(input),
        _ => None,
    }
}

// === Set operations ===============================================

/// Execute a `UNION`/`INTERSECT`/`EXCEPT` plan: evaluate the operand tree, combine per operator,
/// then apply the combined `ORDER BY` and `LIMIT`.
pub(super) fn run_set_operation(
    plan: &PhysicalSetOp,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let mut rows = if let Some(cfg) = super::spill::spill_config() {
        // Spill-to-disk: sorted-merge the set-op tree over disk-backed runs so the operands
        // and the de-dup set never co-reside in memory.
        super::spill_setop::eval_set_tree_spilling(
            &plan.tree,
            plan.columns.len(),
            &cfg,
            engine,
            txn,
        )?
    } else {
        eval_set_tree(&plan.tree, engine, txn)?
    };
    if !plan.order_by.is_empty() {
        sort_rows(&mut rows, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        rows.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
    }
    Ok(ExecutionResult::Rows {
        columns: plan.columns.clone(),
        rows,
    })
}

/// Recursively evaluate a set-operation tree into its result rows.
pub(super) fn eval_set_tree(
    tree: &SetOpTree<PhysicalOperator>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    match tree {
        SetOpTree::Leaf(op) => execute_op(op, engine, txn),
        SetOpTree::Node {
            op,
            all,
            left,
            right,
        } => {
            let left = eval_set_tree(left, engine, txn)?;
            let right = eval_set_tree(right, engine, txn)?;
            Ok(combine_set(*op, *all, left, right))
        },
    }
}

/// Combine two row sets per the set operator. Row equality is SQL "not distinct" (`NULL` = `NULL`),
/// via [`group_keys_equal`]. The non-`ALL` forms de-duplicate; the `ALL` forms keep multiset counts.
pub(super) fn combine_set(op: ast::SetOp, all: bool, left: Vec<Row>, right: Vec<Row>) -> Vec<Row> {
    match op {
        ast::SetOp::Union => {
            let mut out = left;
            out.extend(right);
            if all { out } else { dedupe_rows(out) }
        },
        ast::SetOp::Intersect if all => consume_matching(left, &right, true),
        ast::SetOp::Intersect => filter_by_membership(dedupe_rows(left), &right, true),
        ast::SetOp::Except if all => consume_matching(left, &right, false),
        ast::SetOp::Except => filter_by_membership(dedupe_rows(left), &right, false),
    }
}

/// Multiset helper for `INTERSECT ALL` / `EXCEPT ALL`: walk `left` in order against the remaining
/// occurrence counts of `right`. For each left row that matches an unconsumed right occurrence,
/// consume it; `keep_matched` selects which side is emitted — `true` keeps the matched rows
/// (`INTERSECT ALL` → `min(count)` per value), `false` keeps the unmatched ones
/// (`EXCEPT ALL` → `max(0, left − right)` per value).
pub(super) fn consume_matching(left: Vec<Row>, right: &[Row], keep_matched: bool) -> Vec<Row> {
    // Fast path: when every value is a *hash-safe* type — one whose canonical key bytes compare
    // exactly like `group_keys_equal` (Int/Bool/Text/Date/Time/Timestamp(Tz)/Uuid, and NULL via its
    // distinct `0x00` tag) — count the right side in a `HashMap` and probe per left row → O(left +
    // right). `Float`/`NUMERIC` are excluded: equal values can encode to different bytes, so hashing
    // would split them. Output order (left order) is preserved in both paths.
    if left
        .iter()
        .chain(right)
        .all(|row| row.iter().all(is_hash_safe_value))
    {
        return consume_matching_hashed(left, right, keep_matched);
    }
    // Fallback: linear `group_keys_equal` probe — correct for every type (O(left × right)).
    let mut counts: Vec<(Row, usize)> = Vec::new();
    for row in right {
        match counts.iter_mut().find(|(k, _)| group_keys_equal(k, row)) {
            Some((_, n)) => *n += 1,
            None => counts.push((row.clone(), 1)),
        }
    }
    let mut out = Vec::new();
    for row in left {
        let matched = match counts.iter_mut().find(|(k, _)| group_keys_equal(k, &row)) {
            Some((_, n)) if *n > 0 => {
                *n -= 1;
                true
            },
            _ => false,
        };
        if matched == keep_matched {
            out.push(row);
        }
    }
    out
}

/// `consume_matching` fast path for rows of hash-safe types: count right occurrences by canonical
/// key bytes, then consume per left row — O(left + right), left order preserved.
fn consume_matching_hashed(left: Vec<Row>, right: &[Row], keep_matched: bool) -> Vec<Row> {
    let mut counts: std::collections::HashMap<Vec<u8>, usize> = std::collections::HashMap::new();
    for row in right {
        if let Ok(key) = index_key::encode_index_key(row) {
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut out = Vec::new();
    for row in left {
        let matched =
            index_key::encode_index_key(&row).is_ok_and(|key| match counts.get_mut(&key) {
                Some(n) if *n > 0 => {
                    *n -= 1;
                    true
                },
                _ => false,
            });
        if matched == keep_matched {
            out.push(row);
        }
    }
    out
}

/// Whether a value's canonical index-key encoding compares byte-for-byte exactly like
/// `group_keys_equal` (so it is safe to use as a hash key for set-operation counting, or to probe a
/// backing index for a uniqueness check). Excludes `Float`/`NUMERIC` (equal values can differ
/// byte-wise) and the types the encoder rejects.
pub(super) const fn is_hash_safe_value(value: &ast::Value) -> bool {
    matches!(
        value,
        ast::Value::Null
            | ast::Value::Bool(_)
            | ast::Value::Int(_)
            | ast::Value::Text(_)
            | ast::Value::Date(_)
            | ast::Value::Time(_)
            | ast::Value::Timestamp(_)
            | ast::Value::TimestampTz(_)
            | ast::Value::Uuid(_)
    )
}

/// Pre-resolve every uncorrelated subquery in `expr` to a subquery-free node,
/// running each subquery's plan once against the same engine/txn snapshot. After this returns,
/// [`eval::eval`] sees only ordinary expressions:
///
/// * scalar `(SELECT ...)` → [`TypedExprKind::Literal`] (NULL for zero rows; error for >1 row),
/// * `[NOT] EXISTS` → `Literal(Bool)` (row presence XOR `negated`),
/// * `expr [NOT] IN (SELECT ...)` → [`TypedExprKind::InList`] of literals (so the existing
///   `InList` evaluator supplies the SQL NULL-membership semantics).
///
/// Nesting is handled by [`execute_op`]'s own recursion: a sub-plan executed here re-enters
/// `execute_op`, whose operator arms pre-resolve their own predicates in turn.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-expression-kind descent; length tracks the expression grammar"
)]
pub(super) fn resolve_subqueries(
    expr: &mut TypedExpr,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    use crate::planner::TypedExprKind as K;
    // Descend into children first (an `InSubquery` probe is descended when the node is rewritten).
    match &mut expr.kind {
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            resolve_subqueries(left, engine, txn)?;
            resolve_subqueries(right, engine, txn)?;
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => resolve_subqueries(inner, engine, txn)?,
        K::InList {
            expr: inner, list, ..
        } => {
            resolve_subqueries(inner, engine, txn)?;
            for item in list {
                resolve_subqueries(item, engine, txn)?;
            }
        },
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            resolve_subqueries(inner, engine, txn)?;
            resolve_subqueries(low, engine, txn)?;
            resolve_subqueries(high, engine, txn)?;
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
            resolve_subqueries(inner, engine, txn)?;
            resolve_subqueries(pattern, engine, txn)?;
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(operand) = operand {
                resolve_subqueries(operand, engine, txn)?;
            }
            for branch in branches {
                resolve_subqueries(&mut branch.when, engine, txn)?;
                resolve_subqueries(&mut branch.then, engine, txn)?;
            }
            if let Some(default) = default {
                resolve_subqueries(default, engine, txn)?;
            }
        },
        K::Coalesce(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args)
        | K::SetReturning { args, .. } => {
            for arg in args {
                resolve_subqueries(arg, engine, txn)?;
            }
        },
        K::Crypto { value, key, .. } => {
            resolve_subqueries(value, engine, txn)?;
            resolve_subqueries(key, engine, txn)?;
        },
        K::QuantifiedArray { expr, array, .. } => {
            resolve_subqueries(expr, engine, txn)?;
            resolve_subqueries(array, engine, txn)?;
        },
        K::Subscript { base, index } => {
            resolve_subqueries(base, engine, txn)?;
            resolve_subqueries(index, engine, txn)?;
        },
        K::ArraySlice { base, lower, upper } => {
            resolve_subqueries(base, engine, txn)?;
            for bound in [lower, upper].into_iter().flatten() {
                resolve_subqueries(bound, engine, txn)?;
            }
        },
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. } => {},
        K::Literal(_) | K::Column(_) | K::OuterColumn { .. } | K::AggregateRef(_) => {
            return Ok(());
        },
    }
    // Rewrite this node if it is itself a subquery (`expr.ty` is preserved across the swap).
    if matches!(
        expr.kind,
        K::ScalarSubquery(_)
            | K::Exists { .. }
            | K::InSubquery { .. }
            | K::QuantifiedSubquery { .. }
    ) {
        // A correlated subquery cannot be resolved before an outer row is bound; leave it
        // in place during pre-resolution so the per-row pass resolves it.
        if DEFER_CORRELATED.with(std::cell::Cell::get) && subquery_is_correlated(&expr.kind) {
            return Ok(());
        }
        let owned = std::mem::replace(&mut expr.kind, K::Literal(ast::Value::Null));
        expr.kind = match owned {
            K::ScalarSubquery(plan) => {
                let rows = execute_op(&crate::planner::plan_select(*plan), engine, txn)?;
                K::Literal(scalar_subquery_value(rows)?)
            },
            K::Exists { plan, negated } => {
                let present =
                    !execute_op(&crate::planner::plan_select(*plan), engine, txn)?.is_empty();
                K::Literal(ast::Value::Bool(present ^ negated))
            },
            K::InSubquery {
                expr: mut probe,
                plan,
                negated,
            } => {
                resolve_subqueries(&mut probe, engine, txn)?;
                let elem_ty = probe.ty;
                let rows = execute_op(&crate::planner::plan_select(*plan), engine, txn)?;
                let list = rows
                    .into_iter()
                    .map(|mut row| TypedExpr {
                        kind: K::Literal(if row.is_empty() {
                            ast::Value::Null
                        } else {
                            row.swap_remove(0)
                        }),
                        ty: elem_ty,
                    })
                    .collect();
                K::InList {
                    expr: probe,
                    list,
                    negated,
                }
            },
            K::QuantifiedSubquery {
                expr: mut probe,
                op,
                all,
                plan,
            } => {
                resolve_subqueries(&mut probe, engine, txn)?;
                let elem_ty = probe.ty;
                let rows = execute_op(&crate::planner::plan_select(*plan), engine, txn)?;
                // `x op ANY(rows)` is `(x op r0) OR (x op r1) ...`; `ALL` uses `AND`. Reusing the
                // binary OR/AND eval gives correct three-valued logic for free, and an empty subquery
                // collapses to the identity (ANY -> FALSE, ALL -> TRUE).
                let combine = if all {
                    ast::BinaryOp::And
                } else {
                    ast::BinaryOp::Or
                };
                let mut chain: Option<TypedExpr> = None;
                for mut row in rows {
                    let value = if row.is_empty() {
                        ast::Value::Null
                    } else {
                        row.swap_remove(0)
                    };
                    let term = TypedExpr {
                        kind: K::Binary {
                            left: probe.clone(),
                            op,
                            right: Box::new(TypedExpr {
                                kind: K::Literal(value),
                                ty: elem_ty,
                            }),
                        },
                        ty: ColumnType::Bool,
                    };
                    chain = Some(match chain {
                        None => term,
                        Some(acc) => TypedExpr {
                            kind: K::Binary {
                                left: Box::new(acc),
                                op: combine,
                                right: Box::new(term),
                            },
                            ty: ColumnType::Bool,
                        },
                    });
                }
                chain.map_or(K::Literal(ast::Value::Bool(all)), |c| c.kind)
            },
            _ => unreachable!("guarded by the matches! above"),
        };
    }
    Ok(())
}

/// The single value a scalar subquery contributes: NULL for an empty result, the lone value of a
/// one-row result, else a (run-time) cardinality error.
fn scalar_subquery_value(rows: Vec<Row>) -> Result<ast::Value, Error> {
    match rows.len() {
        0 => Ok(ast::Value::Null),
        1 => Ok(rows
            .into_iter()
            .next()
            .and_then(|mut row| (!row.is_empty()).then(|| row.swap_remove(0)))
            .unwrap_or(ast::Value::Null)),
        _ => Err(Error::Unsupported(
            "scalar subquery returned more than one row".to_owned(),
        )),
    }
}

/// Whether `expr` contains any (scalar/EXISTS/IN) subquery node, i.e. whether [`resolve_subqueries`]
/// would change it. Lets [`resolved_expr`] skip the clone for the common subquery-free predicate.
pub(crate) fn contains_subquery(expr: &TypedExpr) -> bool {
    use crate::planner::TypedExprKind as K;
    match &expr.kind {
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. } => true,
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            contains_subquery(left) || contains_subquery(right)
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => contains_subquery(inner),
        K::InList {
            expr: inner, list, ..
        } => contains_subquery(inner) || list.iter().any(contains_subquery),
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => contains_subquery(inner) || contains_subquery(low) || contains_subquery(high),
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
        } => contains_subquery(inner) || contains_subquery(pattern),
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_some_and(contains_subquery)
                || branches
                    .iter()
                    .any(|b| contains_subquery(&b.when) || contains_subquery(&b.then))
                || default.as_deref().is_some_and(contains_subquery)
        },
        K::Coalesce(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args)
        | K::SetReturning { args, .. } => args.iter().any(contains_subquery),
        K::Crypto { value, key, .. } => contains_subquery(value) || contains_subquery(key),
        K::QuantifiedArray { expr, array, .. } => {
            contains_subquery(expr) || contains_subquery(array)
        },
        K::Subscript { base, index } => contains_subquery(base) || contains_subquery(index),
        K::ArraySlice { base, lower, upper } => {
            contains_subquery(base)
                || lower.as_deref().is_some_and(contains_subquery)
                || upper.as_deref().is_some_and(contains_subquery)
        },
        K::Literal(_) | K::Column(_) | K::OuterColumn { .. } | K::AggregateRef(_) => false,
    }
}

/// Whether `expr` contains a sequence built-in (`nextval`/`currval`/`setval`) anywhere in its own
/// scope. Lets a caller skip the clone + [`resolve_sequence_calls`] for the common sequence-free
/// case. A sequence call inside a *subquery* is that subquery's own concern, so descent stops at
/// subquery boundaries (mirroring [`contains_subquery`]).
pub(crate) fn contains_sequence_call(expr: &TypedExpr) -> bool {
    use crate::planner::TypedExprKind as K;
    match &expr.kind {
        K::ScalarFunction { func, args } => {
            func.is_sequence() || args.iter().any(contains_sequence_call)
        },
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            contains_sequence_call(left) || contains_sequence_call(right)
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => contains_sequence_call(inner),
        K::InList {
            expr: inner, list, ..
        } => contains_sequence_call(inner) || list.iter().any(contains_sequence_call),
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            contains_sequence_call(inner)
                || contains_sequence_call(low)
                || contains_sequence_call(high)
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
        } => contains_sequence_call(inner) || contains_sequence_call(pattern),
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_some_and(contains_sequence_call)
                || branches
                    .iter()
                    .any(|b| contains_sequence_call(&b.when) || contains_sequence_call(&b.then))
                || default.as_deref().is_some_and(contains_sequence_call)
        },
        K::Coalesce(args)
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args)
        | K::SetReturning { args, .. } => args.iter().any(contains_sequence_call),
        K::Crypto { value, key, .. } => {
            contains_sequence_call(value) || contains_sequence_call(key)
        },
        K::QuantifiedArray { expr, array, .. } => {
            contains_sequence_call(expr) || contains_sequence_call(array)
        },
        K::Subscript { base, index } => {
            contains_sequence_call(base) || contains_sequence_call(index)
        },
        K::ArraySlice { base, lower, upper } => {
            contains_sequence_call(base)
                || lower.as_deref().is_some_and(contains_sequence_call)
                || upper.as_deref().is_some_and(contains_sequence_call)
        },
        // A sequence call inside a subquery is resolved when that subquery runs, not here; leaf
        // nodes carry no function call.
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. }
        | K::Literal(_)
        | K::Column(_)
        | K::OuterColumn { .. }
        | K::AggregateRef(_) => false,
    }
}

/// Resolve every sequence built-in (`nextval`/`currval`/`setval`) in `expr` to an `INT` literal by
/// calling the engine, in place. The caller MUST only invoke this where `expr` is evaluated
/// **exactly once** — a `SELECT` with no `FROM` (a single `OneRow` input) or a `VALUES` tuple — so
/// an advancing call (`nextval`/`setval`) advances the sequence exactly once. In any per-row context
/// (a scan projection, `WHERE`, `UPDATE`) the call is left unresolved and the pure per-row evaluator
/// rejects it loudly ([`eval::eval`]), rather than being resolved once and broadcast to every row
/// (which would under-advance the sequence).
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-node-kind traversal mirroring resolve_subqueries; splitting it \
              would scatter the tree walk"
)]
pub(super) fn resolve_sequence_calls(
    expr: &mut TypedExpr,
    engine: &dyn StorageEngine,
) -> Result<(), Error> {
    use crate::planner::TypedExprKind as K;
    // Descend into children first (post-order): a nested sequence call resolves before its parent.
    match &mut expr.kind {
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            resolve_sequence_calls(left, engine)?;
            resolve_sequence_calls(right, engine)?;
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => resolve_sequence_calls(inner, engine)?,
        K::InList {
            expr: inner, list, ..
        } => {
            resolve_sequence_calls(inner, engine)?;
            for item in list {
                resolve_sequence_calls(item, engine)?;
            }
        },
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            resolve_sequence_calls(inner, engine)?;
            resolve_sequence_calls(low, engine)?;
            resolve_sequence_calls(high, engine)?;
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
            resolve_sequence_calls(inner, engine)?;
            resolve_sequence_calls(pattern, engine)?;
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(operand) = operand {
                resolve_sequence_calls(operand, engine)?;
            }
            for branch in branches {
                resolve_sequence_calls(&mut branch.when, engine)?;
                resolve_sequence_calls(&mut branch.then, engine)?;
            }
            if let Some(default) = default {
                resolve_sequence_calls(default, engine)?;
            }
        },
        K::Coalesce(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args)
        | K::SetReturning { args, .. } => {
            for arg in args {
                resolve_sequence_calls(arg, engine)?;
            }
        },
        K::Crypto { value, key, .. } => {
            resolve_sequence_calls(value, engine)?;
            resolve_sequence_calls(key, engine)?;
        },
        K::QuantifiedArray { expr, array, .. } => {
            resolve_sequence_calls(expr, engine)?;
            resolve_sequence_calls(array, engine)?;
        },
        K::Subscript { base, index } => {
            resolve_sequence_calls(base, engine)?;
            resolve_sequence_calls(index, engine)?;
        },
        K::ArraySlice { base, lower, upper } => {
            resolve_sequence_calls(base, engine)?;
            for bound in [lower, upper].into_iter().flatten() {
                resolve_sequence_calls(bound, engine)?;
            }
        },
        // Subqueries carry their own scope; their sequence calls resolve when the subquery executes.
        K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. }
        | K::Literal(_)
        | K::Column(_)
        | K::OuterColumn { .. }
        | K::AggregateRef(_) => {},
    }
    // Resolve this node if it is itself a sequence built-in.
    if let K::ScalarFunction { func, args } = &expr.kind
        && func.is_sequence()
    {
        let value = eval_sequence_call(*func, args, engine)?;
        expr.kind = K::Literal(ast::Value::Int(value));
    }
    Ok(())
}

/// Evaluate one sequence built-in against the engine, returning its `INT` result. The first
/// argument is the sequence name (text); `setval` additionally takes the target value and an
/// optional `is_called`. Arguments are evaluated against the empty row — a sequence call is only
/// resolved in a no-column context.
fn eval_sequence_call(
    func: ast::ScalarFunc,
    args: &[TypedExpr],
    engine: &dyn StorageEngine,
) -> Result<i64, Error> {
    let empty: Row = Vec::new();
    let name = match args.first() {
        Some(arg) => match eval::eval(arg, &empty)? {
            ast::Value::Text(s) => s,
            ast::Value::Null => {
                return Err(Error::Unsupported(format!(
                    "{}() sequence name must not be NULL",
                    func.name()
                )));
            },
            _ => {
                return Err(Error::Unsupported(format!(
                    "{}() sequence name must be text",
                    func.name()
                )));
            },
        },
        None => {
            return Err(Error::Unsupported(format!(
                "{}() requires a sequence name",
                func.name()
            )));
        },
    };
    let id = engine
        .lookup_sequence(&name)?
        .ok_or_else(|| Error::Unsupported(format!("sequence \"{name}\" does not exist")))?;
    match func {
        ast::ScalarFunc::SequenceNext => Ok(engine.sequence_next(id)?),
        ast::ScalarFunc::SequenceCurrent => Ok(engine.sequence_current(id)?),
        ast::ScalarFunc::SequenceSet => {
            let value = match args.get(1).map(|a| eval::eval(a, &empty)).transpose()? {
                Some(ast::Value::Int(v)) => v,
                Some(ast::Value::Null) => {
                    return Err(Error::Unsupported(
                        "setval() target value must not be NULL".to_owned(),
                    ));
                },
                _ => {
                    return Err(Error::Unsupported(
                        "setval() requires an integer target value".to_owned(),
                    ));
                },
            };
            // The optional third argument `is_called`: only the default `true` is expressible via
            // the treaty (`sequence_set` makes the next `nextval` return `value + increment`). The
            // `false` form (next `nextval` returns `value` itself) needs the sequence increment,
            // which the treaty does not expose, so it is rejected rather than silently misapplied.
            if let Some(arg) = args.get(2) {
                match eval::eval(arg, &empty)? {
                    ast::Value::Bool(true) => {},
                    ast::Value::Bool(false) => {
                        return Err(Error::Unsupported(
                            "setval(sequence, value, false) — the is_called = false form — is not \
                             supported"
                                .to_owned(),
                        ));
                    },
                    ast::Value::Null => {
                        return Err(Error::Unsupported(
                            "setval() is_called must not be NULL".to_owned(),
                        ));
                    },
                    _ => {
                        return Err(Error::Unsupported(
                            "setval() is_called must be a boolean".to_owned(),
                        ));
                    },
                }
            }
            engine.sequence_set(id, value)?;
            Ok(value)
        },
        _ => unreachable!("eval_sequence_call is only called for sequence built-ins"),
    }
}

std::thread_local! {
    /// True while [`resolve_subqueries`] is pre-resolving a predicate *before* the per-row loop:
    /// a correlated subquery cannot be resolved yet (no outer row is bound), so it is left
    /// in place to be resolved per row. False during the per-row pass, when every remaining
    /// subquery is resolved against the now-bound outer row.
    static DEFER_CORRELATED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set [`DEFER_CORRELATED`] to `value` for the lifetime of the returned guard, restoring the prior
/// value on drop (so nested sub-plan resolution stays correct). A caller that pre-resolves subqueries
/// without a per-row pass (e.g. the UPDATE/DELETE paths) sets `true` so a correlated subquery is left
/// in place and rejected at eval, rather than resolved against an unbound row to a wrong NULL.
#[must_use]
pub(super) fn defer_correlated(value: bool) -> DeferCorrelatedGuard {
    let previous = DEFER_CORRELATED.with(|c| c.replace(value));
    DeferCorrelatedGuard { previous }
}

pub(super) struct DeferCorrelatedGuard {
    previous: bool,
}

impl Drop for DeferCorrelatedGuard {
    fn drop(&mut self) {
        DEFER_CORRELATED.with(|c| c.set(self.previous));
    }
}

/// Whether the subquery node `kind` is correlated — i.e. its plan references a column from an
/// enclosing query (an [`TypedExprKind::OuterColumn`]). Such a node must be resolved per outer row,
/// not once up front.
fn subquery_is_correlated(kind: &crate::planner::TypedExprKind) -> bool {
    use crate::planner::TypedExprKind as K;
    match kind {
        K::ScalarSubquery(plan)
        | K::Exists { plan, .. }
        | K::InSubquery { plan, .. }
        | K::QuantifiedSubquery { plan, .. } => plan_has_outer_column(plan),
        _ => false,
    }
}

/// Whether `expr`'s tree (descending into nested subquery plans) contains any
/// [`TypedExprKind::OuterColumn`] — i.e. a correlated reference to an enclosing query.
fn expr_has_outer_column(expr: &TypedExpr) -> bool {
    use crate::planner::TypedExprKind as K;
    match &expr.kind {
        K::OuterColumn { .. } => true,
        K::Literal(_) | K::Column(_) | K::AggregateRef(_) => false,
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            expr_has_outer_column(left) || expr_has_outer_column(right)
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. }
        | K::Cast(inner, _) => expr_has_outer_column(inner),
        K::InList {
            expr: inner, list, ..
        } => expr_has_outer_column(inner) || list.iter().any(expr_has_outer_column),
        K::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            expr_has_outer_column(inner)
                || expr_has_outer_column(low)
                || expr_has_outer_column(high)
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
        } => expr_has_outer_column(inner) || expr_has_outer_column(pattern),
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_some_and(expr_has_outer_column)
                || branches
                    .iter()
                    .any(|b| expr_has_outer_column(&b.when) || expr_has_outer_column(&b.then))
                || default.as_deref().is_some_and(expr_has_outer_column)
        },
        K::Coalesce(args)
        | K::ScalarFunction { args, .. }
        | K::ScalarUdf { args, .. }
        | K::ArrayLiteral(args)
        | K::SetReturning { args, .. } => args.iter().any(expr_has_outer_column),
        K::Crypto { value, key, .. } => expr_has_outer_column(value) || expr_has_outer_column(key),
        K::QuantifiedArray { expr, array, .. } => {
            expr_has_outer_column(expr) || expr_has_outer_column(array)
        },
        K::Subscript { base, index } => expr_has_outer_column(base) || expr_has_outer_column(index),
        K::ArraySlice { base, lower, upper } => {
            expr_has_outer_column(base)
                || lower.as_deref().is_some_and(expr_has_outer_column)
                || upper.as_deref().is_some_and(expr_has_outer_column)
        },
        K::ScalarSubquery(plan) | K::Exists { plan, .. } => plan_has_outer_column(plan),
        K::InSubquery {
            expr: probe, plan, ..
        }
        | K::QuantifiedSubquery {
            expr: probe, plan, ..
        } => expr_has_outer_column(probe) || plan_has_outer_column(plan),
    }
}

/// Whether `plan`'s expressions reference any enclosing-query column. Used to decide
/// whether a subquery must be re-run per outer row. Over-approximates safely: a subquery that is
/// only *internally* correlated (to its own nested subquery) is also flagged, which merely re-runs
/// it per outer row — never wrong, just not the minimal amount of work.
fn plan_has_outer_column(plan: &crate::planner::SelectPlan) -> bool {
    let exprs = plan
        .projection
        .iter()
        .map(|p| &p.expr)
        .chain(plan.filter.iter())
        .chain(plan.having.iter())
        .chain(plan.group_keys.iter())
        .chain(plan.distinct_on.iter())
        .chain(plan.order_by.iter().map(|k| &k.expr))
        .chain(plan.joins.iter().map(|j| &j.on))
        .chain(plan.aggregates.iter().filter_map(|a| a.arg.as_ref()))
        .chain(plan.aggregates.iter().filter_map(|a| a.filter.as_ref()))
        .chain(plan.windows.iter().flat_map(|w| w.args.iter()))
        .chain(plan.windows.iter().flat_map(|w| w.partition.iter()))
        .chain(
            plan.windows
                .iter()
                .flat_map(|w| w.order.iter().map(|k| &k.expr)),
        );
    exprs.into_iter().any(expr_has_outer_column)
        || plan.from_cte.as_deref().is_some_and(plan_has_outer_column)
        || plan
            .joins
            .iter()
            .any(|j| j.input_cte.as_deref().is_some_and(plan_has_outer_column))
        // A subquery body cannot itself carry a `WITH` today (the parser rejects it), so a subquery
        // plan never has `recursive_ctes` in practice — but cover them so the walker stays complete
        // (and correct) if that surface ever opens up.
        || plan
            .recursive_ctes
            .iter()
            .any(|cte| plan_has_outer_column(&cte.base) || plan_has_outer_column(&cte.recursive))
}

/// Pre-resolve `expr`'s uncorrelated subqueries against `engine`/`txn`, returning an expression
/// whose only remaining subqueries (if any) are correlated — to be resolved per outer row
/// by the caller. Borrows `expr` untouched when it has no subquery to resolve (the common case) and
/// only clones when a rewrite is actually needed.
pub(super) fn resolved_expr<'a>(
    expr: &'a TypedExpr,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Cow<'a, TypedExpr>, Error> {
    if !contains_subquery(expr) {
        return Ok(Cow::Borrowed(expr));
    }
    let mut owned = expr.clone();
    let _defer = defer_correlated(true);
    resolve_subqueries(&mut owned, engine, txn)?;
    Ok(Cow::Owned(owned))
}

/// Evaluate `expr` for one outer `row` when it carries a correlated subquery: bind the row
/// as the enclosing scope, resolve the remaining (correlated) subqueries against it, then evaluate.
pub(super) fn eval_correlated(
    expr: &TypedExpr,
    row: &Row,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ast::Value, Error> {
    let _outer = eval::bind_outer_row(row.clone());
    let mut resolved = expr.clone();
    // The outer row is bound, so resolve every remaining subquery (none are deferred now).
    let _defer = defer_correlated(false);
    resolve_subqueries(&mut resolved, engine, txn)?;
    eval::eval(&resolved, row)
}

/// Per-query work-memory budget in bytes, or `0` for unlimited (the default).
///
/// The process-wide **default** cap on how many bytes any one query may materialize in a single
/// executor stage. When non-zero, [`execute_op`] returns [`Error::OutOfMemory`] for a stage whose
/// row set exceeds it, so a runaway sort / aggregate / join over a huge input fails honestly
/// instead of OOM-killing the server. Set via [`set_work_mem`]; the server exposes it as
/// `--work-mem`. A session `SET work_mem = '…'` overrides it per statement — budget checks go
/// through `effective_work_mem`, which consults the pinned session context first.
static WORK_MEM: AtomicUsize = AtomicUsize::new(0);

/// Set the process-default per-query work-memory budget in bytes; `0` (the default) disables it.
pub fn set_work_mem(bytes: usize) {
    WORK_MEM.store(bytes, Ordering::Relaxed);
}

/// The process-default per-query work-memory budget in bytes, or `0` if unlimited.
///
/// Budget enforcement uses the statement-effective resolution (`effective_work_mem`) instead, so
/// a session `SET work_mem` is honoured.
#[must_use]
pub fn work_mem() -> usize {
    WORK_MEM.load(Ordering::Relaxed)
}

/// Parse a `work_mem` setting value into bytes.
///
/// Follows the conventional memory-GUC form: a bare integer is **kilobytes**; an integer with a
/// `kB` / `MB` / `GB` / `TB` suffix (case-insensitive, optional whitespace) is scaled accordingly.
/// `0` means unlimited, matching `--work-mem 0`. Returns `None` for anything else (empty,
/// negative, fractional, unknown unit, overflow) so `SET work_mem` can reject the value loudly
/// instead of storing a string the budget check would then ignore.
#[must_use]
pub fn parse_work_mem(value: &str) -> Option<usize> {
    let v = value.trim();
    let split = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
    let (digits, unit) = v.split_at(split);
    if digits.is_empty() {
        return None;
    }
    let n: usize = digits.parse().ok()?;
    let scale: usize = match unit.trim().to_ascii_lowercase().as_str() {
        // A bare integer is in kilobytes (the conventional memory-GUC unit).
        "" | "kb" => 1024,
        "mb" => 1024 * 1024,
        "gb" => 1024 * 1024 * 1024,
        "tb" => 1024 * 1024 * 1024 * 1024,
        _ => return None,
    };
    n.checked_mul(scale)
}

/// The statement-effective work-memory budget in bytes, or `0` if unlimited.
///
/// A session `SET work_mem` (pinned into the statement's [session context](super::session_ctx) by
/// every execution path before it runs the statement) overrides the process default — this is what
/// makes the budget tunable per session (the guards used to read only the process default, so a
/// session `SET work_mem` was silently ignored). A pinned value that fails to parse falls back to
/// the process default; `SET`-time validation rejects such values, so this is defense in depth,
/// not a policy.
pub(super) fn effective_work_mem() -> usize {
    super::session_ctx::setting("work_mem")
        .and_then(|v| parse_work_mem(&v))
        .unwrap_or_else(work_mem)
}

/// Estimated heap + inline bytes of one value: the enum's own size plus the heap a
/// variable-length variant owns (text/JSON string bytes, array elements recursively).
fn value_bytes(v: &ast::Value) -> usize {
    std::mem::size_of::<ast::Value>()
        + match v {
            ast::Value::Text(s) | ast::Value::Json(s) => s.len(),
            ast::Value::Array(items) => items.iter().map(value_bytes).sum(),
            _ => 0,
        }
}

/// Estimated bytes one row occupies. Shared with the spill `MemBudget` so its per-row budget agrees
/// with `work_mem`'s whole-stage estimate.
pub(super) fn row_bytes(row: &[ast::Value]) -> usize {
    row.iter().map(value_bytes).sum()
}

/// Estimated bytes a materialized row set occupies.
fn rows_bytes(rows: &[Row]) -> usize {
    rows.iter().map(|r| row_bytes(r)).sum()
}

/// Enforce the work-memory `budget` (0 = unset) against the running byte size of a limit-aware
/// top-N's retained rows. Loud, like every other stage's budget check.
fn enforce_top_n_budget(budget: usize, retained_bytes: usize) -> Result<(), Error> {
    if budget != 0 && retained_bytes > budget {
        return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
            "query work_mem of {budget} bytes exceeded ({retained_bytes} bytes retained by a \
             top-N ORDER BY ... LIMIT); use a smaller LIMIT/OFFSET or raise work_mem \
             (SET work_mem / --work-mem)"
        ))));
    }
    Ok(())
}

/// Enforce the work-memory budget against a materialized stage. A no-op when the budget is
/// unset (the default). Reads the statement-effective budget so `SET work_mem` is honoured.
pub(super) fn enforce_work_mem(rows: &[Row]) -> Result<(), Error> {
    let budget = effective_work_mem();
    if budget != 0 {
        let used = rows_bytes(rows);
        if used > budget {
            return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                "query work_mem of {budget} bytes exceeded ({used} bytes materialized in one \
                 stage); add a more selective WHERE/LIMIT or raise work_mem (SET work_mem / \
                 --work-mem)"
            ))));
        }
    }
    Ok(())
}

/// Execute one physical operator into a fully-materialized row set, enforcing the per-query
/// work-memory budget on the result.
///
/// The row-path executor materializes each stage in full (it is not yet streaming), so the budget
/// is checked **after** each stage materializes — a coarse cap that bounds any one query to roughly
/// `work_mem` per stage and stops a runaway query from compounding across stages. Failing *before*
/// the allocation needs the streaming / spill-to-disk operators; until then
/// this is the honest-error backstop.
pub(super) fn execute_op(
    op: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let rows = execute_op_inner(op, engine, txn)?;
    enforce_work_mem(&rows)?;
    // EXPLAIN ANALYZE per-node actuals: one thread-local check per operator, never per
    // row. The streaming sources record through their own counting wrappers instead.
    if super::instrument::enabled() {
        super::instrument::record(super::instrument::key(op), rows.len() as u64);
    }
    Ok(rows)
}

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-operator dispatch; length tracks the operator set, not complexity"
)]
fn execute_op_inner(
    op: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    match op {
        PhysicalOperator::SeqScan { table, columns } => {
            super::scan::scan_rows_projected(table, columns, engine, txn)
        },
        PhysicalOperator::IndexScan {
            table,
            index,
            lo,
            hi,
            ..
        } => index_scan_rows(table, index, lo, hi, engine, txn),
        PhysicalOperator::OneRow => Ok(vec![Vec::new()]),
        PhysicalOperator::Values { rows } => {
            // A `(VALUES ...) AS x` source: each row's cells reference no columns, so they evaluate
            // against an empty input row.
            let empty: Row = Vec::new();
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|e| eval::eval(e, &empty))
                        .collect::<Result<Row, _>>()
                })
                .collect()
        },
        PhysicalOperator::SetOperation(set_op) => {
            // A `(SELECT ... UNION ...) AS x` source: run it through the same path as a top-level set
            // operation (which yields a materialized row set).
            match run_set_operation(set_op, engine, txn)? {
                ExecutionResult::Rows { rows, .. } => Ok(rows),
                _ => Err(Error::Unsupported(
                    "internal: a derived set operation did not produce a row set".to_owned(),
                )),
            }
        },
        PhysicalOperator::LockRows {
            input,
            table,
            predicate,
            mode,
            skip_locked,
        } => {
            // `FOR UPDATE` / `FOR SHARE`: take a row lock on every base row that satisfies the
            // predicate, then return the pipeline's rows unchanged. The analyzer guarantees a
            // single-table shape and a subquery-free predicate, so the WHERE evaluates against the
            // scanned base row directly. The lock is held until the transaction ends; the lock
            // manager is no-wait, so a concurrent writer of a locked row aborts with a serialization
            // conflict rather than blocking (the lost-update escape hatch).
            //
            // `SKIP LOCKED` (the job-queue pattern): a matched row whose lock another transaction
            // holds is collected instead of aborting, and the pipeline then runs under a scan
            // guard that hides exactly those rows — so workers claim disjoint rows without
            // blocking, and a LIMIT fills up from lockable rows, like the reference engine.
            let mut lock_held_elsewhere: HashSet<Tid> = HashSet::new();
            for (tid, row) in super::scan::scan_table(table, engine, txn)? {
                let matched = match predicate {
                    Some(pred) => matches!(eval::eval(pred, &row)?, ast::Value::Bool(true)),
                    None => true,
                };
                if !matched {
                    continue;
                }
                match engine.lock_row(txn, table.id, tid, *mode) {
                    Ok(()) => {},
                    Err(nusadb_core::Error::SerializationConflict { .. }) if *skip_locked => {
                        lock_held_elsewhere.insert(tid);
                    },
                    Err(e) => return Err(e.into()),
                }
            }
            let _guard = (!lock_held_elsewhere.is_empty())
                .then(|| super::lock_skip::scope(table.id, lock_held_elsewhere));
            execute_op(input, engine, txn)
        },
        PhysicalOperator::Filter { input, predicate } => {
            let rows = execute_op(input, engine, txn)?;
            // WHERE and HAVING both lower to a Filter; pre-resolve any uncorrelated subquery in the
            // predicate once before scanning rows. Any subquery left after that is
            // correlated and is resolved per row against the bound outer row.
            let predicate = resolved_expr(predicate, engine, txn)?;
            let correlated = contains_subquery(&predicate);
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                let verdict = if correlated {
                    eval_correlated(&predicate, &row, engine, txn)?
                } else {
                    eval::eval(&predicate, &row)?
                };
                if matches!(verdict, ast::Value::Bool(true)) {
                    out.push(row);
                }
            }
            Ok(out)
        },
        PhysicalOperator::Sort {
            input,
            keys,
            limit_ties,
            top_n,
        } => {
            // A subquery in an ORDER BY key must be pre-resolved per row, which the
            // spill sort's streaming comparator does not do — so route to the in-memory path then.
            let has_subquery = keys.iter().any(|k| contains_subquery(&k.expr));
            // Limit-aware top-N: the enclosing LIMIT needs only the first
            // `m` rows in sort order, so select them with a bounded streaming pass — O(N log m),
            // retaining `m` rows — instead of the full O(N log N) sort. Every input row is still
            // pulled (so a SERIALIZABLE scan's read set is unchanged), only the sort is avoided.
            // Skipped when an ORDER BY key holds a subquery (needs per-row resolution the streaming
            // comparator cannot do) — the full sort below handles that.
            if let Some(m) = top_n
                && !has_subquery
            {
                return top_n_rows(input, keys, *m, engine, txn);
            }
            if let Some(cap) = limit_ties {
                // FETCH FIRST n ROWS WITH TIES: the tie trim re-evaluates the keys
                // on the boundary rows, so it runs over the in-memory sort (not the streaming spill
                // path). A subquery in an ORDER BY key would need per-row resolution here — reject it
                // loudly rather than mis-evaluate.
                if has_subquery {
                    return Err(Error::Unsupported(
                        "FETCH FIRST ... WITH TIES with a subquery in ORDER BY is not yet supported"
                            .to_owned(),
                    ));
                }
                let mut rows = execute_op(input, engine, txn)?;
                sort_rows(&mut rows, keys)?;
                return apply_ties_limit(rows, keys, cap);
            }
            if !has_subquery && let Some(cfg) = super::spill::spill_config() {
                // With spill-to-disk enabled, bound the sort's working memory via an external merge
                // sort; otherwise materialize the input and sort it in memory.
                return super::spill_sort::external_sort(input, keys, &cfg, engine, txn);
            }
            let mut rows = execute_op(input, engine, txn)?;
            if has_subquery {
                sort_rows_with_subqueries(&mut rows, keys, engine, txn)?;
            } else {
                sort_rows(&mut rows, keys)?;
            }
            Ok(rows)
        },
        PhysicalOperator::Project { input, columns } => {
            let rows = execute_op(input, engine, txn)?;
            // A scalar subquery in the SELECT list is pre-resolved once here; a correlated
            // one is left for per-row resolution against the bound outer row.
            let mut columns = columns
                .iter()
                .map(|p| resolved_expr(&p.expr, engine, txn))
                .collect::<Result<Vec<_>, _>>()?;
            // Sequence built-ins (nextval/currval/setval) are resolved here, where the input row
            // count is known. They are sound only when the projection is evaluated exactly once — a
            // single input row, e.g. a no-FROM SELECT's `OneRow`. Over more than one row the calls
            // are left unresolved so the per-row evaluator rejects them loudly (resolving once would
            // under-advance the sequence); over zero rows nothing is evaluated, so nothing is
            // resolved and no spurious advance happens.
            if rows.len() == 1 && columns.iter().any(|c| contains_sequence_call(c)) {
                for col in &mut columns {
                    resolve_sequence_calls(col.to_mut(), engine)?;
                }
            }
            let any_correlated = columns.iter().any(|c| contains_subquery(c));
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                let projected: Row = columns
                    .iter()
                    .map(|expr| {
                        if any_correlated && contains_subquery(expr) {
                            eval_correlated(expr, &row, engine, txn)
                        } else {
                            eval::eval(expr, &row)
                        }
                    })
                    .collect::<Result<Row, _>>()?;
                out.push(projected);
            }
            Ok(out)
        },
        PhysicalOperator::ProjectSet {
            input,
            columns,
            ordinality,
        } => {
            let rows = execute_op(input, engine, txn)?;
            // Pre-resolve uncorrelated subqueries in every column once; the set-returning
            // column is evaluated per input row to produce its element list. A *correlated*
            // subquery in a ProjectSet column is not supported and surfaces `Unsupported` at eval
            // (v1: correlated subqueries are not combined with set-returning projections).
            let resolved = columns
                .iter()
                .map(|p| resolved_expr(&p.expr, engine, txn))
                .collect::<Result<Vec<_>, _>>()?;
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                // Evaluate the scalar columns once per input row; the SRF column expands to a list.
                // The analyzer guarantees exactly one SRF column, so `srf` is set exactly once.
                let mut scalars: Row = Vec::with_capacity(resolved.len());
                let mut srf: Option<(usize, Vec<ast::Value>)> = None;
                for (i, expr) in resolved.iter().enumerate() {
                    if let crate::planner::TypedExprKind::SetReturning { func, args } = &expr.kind {
                        srf = Some((i, eval_set_returning(*func, args, &row)?));
                        scalars.push(ast::Value::Null); // placeholder, set per produced element
                    } else {
                        scalars.push(eval::eval(expr, &row)?);
                    }
                }
                // One output row per produced element; an empty/NULL set emits no row for this input.
                // `WITH ORDINALITY` appends a 1-based counter of the produced elements.
                if let Some((pos, elements)) = srf {
                    for (idx, element) in elements.into_iter().enumerate() {
                        let mut output = scalars.clone();
                        if let Some(slot) = output.get_mut(pos) {
                            *slot = element;
                        }
                        if *ordinality {
                            let n = i64::try_from(idx).map_or(i64::MAX, |i| i.saturating_add(1));
                            output.push(ast::Value::Int(n));
                        }
                        out.push(output);
                    }
                }
            }
            Ok(out)
        },
        PhysicalOperator::Limit {
            input,
            count,
            offset,
        } => {
            // Pull the child through the streaming cursor and stop once the window is filled
            // (`LIMIT 5` over a 1M-row join used to materialize the whole
            // join output first and blow the work-mem budget). `stream_op` yields exactly this
            // arm's old rows in the same order — truly-streaming children (scan/filter/project/
            // hash-join/…) now stop early, blocking ones still materialize inside it unchanged.
            let mut child = super::stream::stream_op(input, engine, txn)?;
            let skip = usize::try_from(*offset).unwrap_or(usize::MAX);
            let n = usize::try_from(*count).unwrap_or(usize::MAX);
            let mut out = Vec::new();
            let mut skipped = 0_usize;
            while out.len() < n {
                let Some(row) = child.try_next()? else {
                    break;
                };
                if skipped < skip {
                    skipped += 1;
                    continue;
                }
                out.push(row);
            }
            Ok(out)
        },
        PhysicalOperator::Distinct { input } => {
            if let Some(cfg) = super::spill::spill_config() {
                // Spill-to-disk: sort on all columns, emit one row per adjacent-equal run.
                return sort_based_distinct(input, &cfg, engine, txn);
            }
            let rows = execute_op(input, engine, txn)?;
            Ok(dedupe_rows(rows))
        },
        PhysicalOperator::DistinctOn { input, keys } => {
            // Keep the first input row per distinct key tuple; "first" follows the ORDER BY
            // the planner placed beneath this. NULL is not distinct from NULL (group_keys_equal).
            let rows = execute_op(input, engine, txn)?;
            let mut seen: Vec<Vec<ast::Value>> = Vec::new();
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                let key: Vec<ast::Value> = keys
                    .iter()
                    .map(|k| eval::eval(k, &row))
                    .collect::<Result<_, _>>()?;
                if !seen.iter().any(|s| group_keys_equal(s, &key)) {
                    seen.push(key);
                    out.push(row);
                }
            }
            Ok(out)
        },
        PhysicalOperator::ScalarAggregate { input, calls } => {
            // No GROUP BY: one global group folds the whole input into one row. The input is
            // pulled through the streaming cursor rather than materialized:
            // the accumulators are what the aggregate needs, not the rows — a full-table
            // `count(*)` used to hold the entire scan and OOM past ~1M rows.
            // `COUNT(*)` over a plain scan skips the fold (and the per-row decode) entirely.
            if let Some(out_row) = super::agg::scalar_count_star_fast(input, calls, engine, txn)? {
                return Ok(vec![out_row]);
            }
            let mut child = super::stream::stream_op(input, engine, txn)?;
            let out_row = super::agg::fold_aggregates_streamed(calls, child.as_mut())?;
            Ok(vec![out_row])
        },
        PhysicalOperator::GroupAggregate {
            input,
            group_keys,
            calls,
        } => {
            // With spill-to-disk enabled, the group-state memory must stay bounded. When the
            // ANALYZE statistics bound the group state within half the budget, the DIRECT hash
            // fold serves — zero disk, measured 2.1x over the sort-based fold at 10k groups.
            // Anything larger (or unestimated) keeps the sort-based fold
            // Whose memory bound needs no estimate. A grace-partitioned hash fold was
            // built and MEASURED SLOWER than the sort fold at 200k groups (two-pass spill of
            // every input row), so it was deliberately not shipped — a hybrid fold-then-spill
            // design is the recorded follow-up if many-group speed is still wanted.
            if let Some(cfg) = super::spill::spill_config() {
                let budget = u64::try_from(cfg.threshold_bytes).unwrap_or(u64::MAX);
                match estimated_group_state_bytes(input, group_keys, engine) {
                    Some(est) if est <= budget / 2 => super::agg::note_stats_hash_agg(),
                    _ => {
                        return sort_based_group_aggregate(
                            input, group_keys, calls, &cfg, engine, txn,
                        );
                    },
                }
            }
            let mut child = super::stream::stream_op(input, engine, txn)?;
            super::agg::run_group_aggregate_streamed(child.as_mut(), group_keys, calls)
        },
        PhysicalOperator::GroupingSetsAggregate {
            input,
            group_keys,
            grouping_sets,
            calls,
        } => {
            // One streamed pass folding into per-set per-group accumulators — O(sum of groups)
            // memory instead of O(input) (the residual).
            let mut child = super::stream::stream_op(input, engine, txn)?;
            super::agg::run_grouping_sets_aggregate_streamed(
                child.as_mut(),
                group_keys,
                grouping_sets,
                calls,
            )
        },
        PhysicalOperator::Window {
            input,
            windows,
            top_n,
        } => {
            let input_rows = match top_n {
                // Limit-aware ranking window: the planner proved every
                // window is ranking-only over a single partition sharing one order, and the outer
                // LIMIT wants only the first `m` rows in that order — so compute over just the `m`
                // smallest rows (bounded memory, no full materialization). A ranking value at
                // position `k` depends only on rows at positions `≤ k`, so ranking over the first
                // `m` rows is identical to the full computation for those rows.
                Some(m) => {
                    let order = windows.first().map_or(&[][..], |w| &w.order);
                    top_n_rows(input, order, *m, engine, txn)?
                },
                None => execute_op(input, engine, txn)?,
            };
            run_window(input_rows, windows)
        },
        PhysicalOperator::NestedLoopJoin {
            left,
            right,
            predicate,
            kind,
            left_width,
            right_width,
            coalesce_pairs,
        } => {
            let left_rows = execute_op(left, engine, txn)?;
            let right_rows = execute_op(right, engine, txn)?;
            // A subquery in the JOIN ON predicate is uncorrelated, so resolve it once up front.
            let predicate = resolved_expr(predicate, engine, txn)?;
            let rows = run_nested_loop_join(
                &left_rows,
                &right_rows,
                &predicate,
                *kind,
                *left_width,
                *right_width,
            )?;
            Ok(merge_join_using_columns(rows, coalesce_pairs))
        },
        PhysicalOperator::HashJoin {
            left,
            right,
            keys,
            residual,
            kind,
            left_width,
            right_width,
            coalesce_pairs,
        } => {
            // The equi-keys are plain column refs; only the residual can carry a subquery.
            let residual = residual
                .as_ref()
                .map(|r| resolved_expr(r, engine, txn))
                .transpose()?;
            // With spill-to-disk enabled, an equi-join (any of Inner/Left/Right/Full — the only kinds
            // a HashJoin carries; Cross lowers to a nested loop) bounds its build side via a grace
            // hash join. The no-spill case keeps the materializing in-memory path.
            let rows = if let Some(cfg) = super::spill::spill_config() {
                grace_join(
                    left,
                    right,
                    keys,
                    residual.as_deref(),
                    *kind,
                    *left_width,
                    *right_width,
                    &cfg,
                    engine,
                    txn,
                )?
            } else {
                let left_rows = execute_op(left, engine, txn)?;
                let right_rows = execute_op(right, engine, txn)?;
                // Materialized path: with both sides in hand the build-side choice is
                // free and exact — build on the LEFT when it is decisively smaller (INNER with
                // no USING/NATURAL merge only; everything else keeps the build-right default).
                if matches!(kind, ast::JoinKind::Inner)
                    && coalesce_pairs.is_empty()
                    && left_rows.len() * 4 <= right_rows.len()
                {
                    super::join::run_hash_join_left_build(
                        &left_rows,
                        &right_rows,
                        keys,
                        residual.as_deref(),
                        *left_width,
                    )?
                } else {
                    run_hash_join(
                        &left_rows,
                        &right_rows,
                        keys,
                        residual.as_deref(),
                        *kind,
                        *left_width,
                        *right_width,
                    )?
                }
            };
            Ok(merge_join_using_columns(rows, coalesce_pairs))
        },
        PhysicalOperator::LateralJoin {
            left,
            right,
            predicate,
            kind,
            right_width,
        } => {
            let left_rows = execute_op(left, engine, txn)?;
            // The predicate spans `[left ++ right]` and never references the outer scope, so any
            // subquery in it is uncorrelated — resolve it once up front (as nested-loop join does).
            let predicate = resolved_expr(predicate, engine, txn)?;
            run_lateral_join(
                &left_rows,
                right,
                &predicate,
                *kind,
                *right_width,
                engine,
                txn,
            )
        },
        PhysicalOperator::InfoSchemaScan { view } => run_info_schema(*view, engine, txn),
        PhysicalOperator::VectorKnn {
            table,
            column_ordinal,
            query,
            k,
            filter,
        } => run_vector_knn(
            table,
            *column_ordinal,
            query,
            *k,
            filter.as_ref(),
            engine,
            txn,
        ),
        PhysicalOperator::WithRecursive { ctes, body } => {
            run_recursive_cte(ctes, body, engine, txn)
        },
        PhysicalOperator::WithModifying { ctes, body } => {
            run_modifying_ctes(ctes, body, engine, txn)
        },
    }
}

/// Execute a query's data-modifying CTEs: run each statement once (performing its
/// writes), bind its `RETURNING` rows to the CTE's synthetic table, then run the body — which reads
/// those rows as a relation. The bindings drop after the body produces its rows. The analyzer
/// forbids the modified table from being read elsewhere, so the body never observes the writes
/// directly (only the materialized RETURNING rows), keeping the result snapshot-faithful.
fn run_modifying_ctes(
    ctes: &[crate::planner::PhysicalModifyingCte],
    body: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    use crate::planner::PhysicalPlan;
    let mut guards = Vec::with_capacity(ctes.len());
    for cte in ctes {
        let result = match &cte.plan {
            PhysicalPlan::Insert(p) => super::dml::run_insert(p, engine, txn)?,
            PhysicalPlan::Update(p) => super::dml::run_update(p, engine, txn)?,
            PhysicalPlan::Delete(p) => super::dml::run_delete(p, engine, txn)?,
            _ => {
                return Err(Error::Unsupported(
                    "a data-modifying CTE must be INSERT/UPDATE/DELETE".to_owned(),
                ));
            },
        };
        // The analyzer requires a RETURNING clause, so the statement yields a row set; bind it.
        let rows = match result {
            ExecutionResult::Rows { rows, .. } => rows,
            _ => Vec::new(),
        };
        guards.push(super::recursive::bind(cte.id, rows));
    }
    // The bindings must stay live for the whole body, then unbind — drop the guards after it runs.
    let result = execute_op(body, engine, txn);
    drop(guards);
    result
}

/// `LATERAL` dependent join: for each left row, bind it as the enclosing scope and
/// re-execute the `right` subquery (whose correlated `OuterColumn` references then read that row),
/// emitting `[left ++ right]` for each pair satisfying `predicate`. A `Left` join NULL-pads an
/// unmatched left row; `Inner`/`Cross` drop it. Unlike a materialized join, `right` is re-run per
/// left row, so its result set can differ row to row.
fn run_lateral_join(
    left_rows: &[Row],
    right: &PhysicalOperator,
    predicate: &TypedExpr,
    kind: ast::JoinKind,
    right_width: usize,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let keep_unmatched_left = matches!(kind, ast::JoinKind::Left | ast::JoinKind::Full);
    let mut out = Vec::new();
    for left_row in left_rows {
        // Cooperative cancellation: break at a left-row boundary rather than running every
        // re-execution of the right side to completion.
        crate::cancel::check()?;
        // Bind the left row so the right subquery's `OuterColumn` references resolve to it, then
        // re-execute the right side for just this row. The guard pops the binding before the next.
        let right_rows = {
            let _outer = eval::bind_outer_row(left_row.clone());
            execute_op(right, engine, txn)?
        };
        let mut left_matched = false;
        for right_row in &right_rows {
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if matches!(eval::eval(predicate, &joined)?, ast::Value::Bool(true)) {
                out.push(joined);
                left_matched = true;
            }
        }
        if keep_unmatched_left && !left_matched {
            let mut joined = left_row.clone();
            joined.extend(std::iter::repeat_n(ast::Value::Null, right_width));
            out.push(joined);
        }
    }
    Ok(out)
}

/// Fixed seed for HNSW level assignment, so a rebuilt vector index is reproducible.
const HNSW_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Minimum search beam width: a wider beam than `k` trades a little latency for higher recall.
const HNSW_EF_SEARCH_MIN: usize = 64;

/// Session variable controlling the HNSW search beam width: `SET hnsw_ef_search = N`.
const EF_SEARCH_SETTING: &str = "hnsw_ef_search";

/// The HNSW search beam width for a `k`-NN query. Honors the `hnsw_ef_search` session
/// setting when set to a positive integer (a larger beam trades latency for recall), otherwise uses
/// the default floor. Always at least `k` — the search needs to consider at least `k` candidates.
fn resolve_ef_search(k: usize) -> usize {
    super::session_ctx::setting(EF_SEARCH_SETTING)
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map_or_else(|| k.max(HNSW_EF_SEARCH_MIN), |hint| hint.max(k))
}

/// One cached, in-memory HNSW vector index plus the rows it was built from. `signature` is a
/// fold of the table's visible tids at build time — the staleness signal: an `INSERT`/`DELETE`/
/// `UPDATE` changes the visible tid set (an MVCC update supersedes a row with a new tid), so a
/// changed signature triggers a rebuild and the cache can never serve stale row contents. Only rows
/// with a non-NULL vector are inserted into the graph; `id_to_row` maps each internal HNSW id back to
/// its index in `rows`.
struct CachedVectorIndex {
    signature: u64,
    index: crate::hnsw::HnswIndex,
    rows: Vec<Row>,
    id_to_row: Vec<usize>,
    /// The tid of each row in `rows`, parallel to it. Used to diff the cached set against the live
    /// table so a pure-append change can be applied incrementally instead of rebuilt.
    tids: Vec<Tid>,
}

/// Cache key for a built HNSW index: the engine instance, the table, and the column ordinal.
/// The engine's address disambiguates distinct engine instances that share a thread (e.g.
/// two independent test engines): they reuse table ids and tid layouts, so without it their
/// signatures could collide and one engine's cache could be served for another's query.
type HnswCacheKey = (usize, nusadb_core::TableId, usize);

/// One per-key cache slot: the built index for a given `(engine, table, column)`, or `None` before
/// its first build. Each slot has its own lock, so a (re)build of one index never blocks queries on
/// another — only the brief outer-map lookup is shared.
type HnswSlot = std::sync::Arc<std::sync::RwLock<Option<CachedVectorIndex>>>;

/// Process-wide cache of built HNSW indexes, shared across threads so the wire server's
/// blocking-task pool builds each index once rather than once per thread. The outer lock only maps a
/// key to its slot (held briefly); the per-slot lock guards the index itself — reads (the warm path)
/// take the slot's shared read lock and run concurrently, and only a (re)build of *that* slot takes
/// its exclusive write lock. The graph is still not persisted to disk — a fresh process rebuilds on
/// first use (page-backed persistence is the separate milestone).
static HNSW_CACHE: std::sync::LazyLock<std::sync::RwLock<HashMap<HnswCacheKey, HnswSlot>>> =
    std::sync::LazyLock::new(|| std::sync::RwLock::new(HashMap::new()));

/// The cache slot for `key`, creating an empty one if absent. The outer map lock is released before
/// returning, so the caller locks only the per-key slot — never both at once (no lock-order cycle).
/// Poisoning is recovered as elsewhere (a poisoned lock still holds a valid map/slot).
fn hnsw_slot(key: HnswCacheKey) -> HnswSlot {
    {
        let map = HNSW_CACHE
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(slot) = map.get(&key) {
            return std::sync::Arc::clone(slot);
        }
    }
    // Absent — insert under the write lock (another thread may have created it while we waited).
    let mut map = HNSW_CACHE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::sync::Arc::clone(
        map.entry(key)
            .or_insert_with(|| std::sync::Arc::new(std::sync::RwLock::new(None))),
    )
}

/// A stable per-process identifier for an engine instance — its data-pointer address. Used only to
/// key the per-thread index cache; never dereferenced.
fn engine_identity(engine: &dyn StorageEngine) -> usize {
    std::ptr::from_ref(engine).cast::<()>().addr()
}

/// A cheap staleness signature for `table`: a fold of every visible tid, its tuple bytes, and the
/// count. Hashing the bytes (not just the tid) keeps the signature engine-agnostic: an engine may
/// keep a row's tid stable across an `UPDATE` (the clustered btree does; an append-only engine allocates
/// a fresh tid per version), so tids alone cannot witness an in-place content change. Any
/// `INSERT`/`DELETE`/`UPDATE` visible to `txn` therefore changes the signature, a cached vector
/// index built from a different signature is rebuilt before use, and the cache never serves stale
/// row contents. One hashing pass over the table remains far cheaper than the HNSW rebuild it
/// guards.
fn scan_signature(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<u64, Error> {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut scan = engine.scan(txn, table.id)?;
    let mut count: u64 = 0;
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        tid.hash(&mut hasher);
        tuple[..].hash(&mut hasher);
        count += 1;
    }
    count.hash(&mut hasher);
    Ok(hasher.finish())
}

/// Build an HNSW index over `table`'s `column_ordinal` `VECTOR(n)` column from a fresh scan.
/// Rows whose vector is `NULL` (or not a vector) are kept in `rows` but not inserted, so a search
/// never returns them — their internal id is simply absent from the graph.
fn build_vector_index(
    table: &TableSchema,
    column_ordinal: usize,
    dim: usize,
    signature: u64,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<CachedVectorIndex, Error> {
    let scanned = super::scan::scan_table(table, engine, txn)?;
    let mut index = crate::hnsw::HnswIndex::new(
        dim,
        crate::hnsw::Metric::Cosine,
        crate::hnsw::HnswParams::default(),
        HNSW_SEED,
    );
    // Insert only rows with a matching-dimension vector. The HNSW assigns sequential ids 0,1,2,…, so
    // `id_to_row[id]` recovers the row index (a NULL / wrong-dim vector is left out of the graph and
    // can never be returned — matching the exact path, which skips it too).
    let mut rows = Vec::with_capacity(scanned.len());
    let mut tids = Vec::with_capacity(scanned.len());
    let mut id_to_row = Vec::new();
    for (tid, row) in scanned {
        if let Some(ast::Value::Vector(v)) = row.get(column_ordinal)
            && v.len() == dim
        {
            index.insert(v.clone())?;
            id_to_row.push(rows.len());
        }
        tids.push(tid);
        rows.push(row);
    }
    Ok(CachedVectorIndex {
        signature,
        index,
        rows,
        id_to_row,
        tids,
    })
}

/// Bring a cached vector index up to date for the current snapshot. Diffs the live rows
/// against the cache: if rows were only **added** (a pure append, the common vector-workload case),
/// the new rows are inserted into the existing graph — avoiding an O(n log n) rebuild. Any
/// **removal** (a `DELETE`, or the superseded version of an `UPDATE`) means the graph would need a
/// node removed, which the HNSW does not support, so it falls back to a full rebuild — and so does
/// a **changed row under a kept tid**: an engine may keep a row's address stable across an `UPDATE`
/// (the clustered btree does), so kept tids are re-decoded and compared to the cached rows. Only
/// runs when the table signature already changed, so the compare pass costs nothing on the warm
/// path. Correct either way: the returned index reflects exactly the rows visible to `txn`.
fn maintain_vector_index(
    cached: CachedVectorIndex,
    table: &TableSchema,
    column_ordinal: usize,
    dim: usize,
    signature: u64,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<CachedVectorIndex, Error> {
    let cached_rows: std::collections::HashMap<Tid, usize> = cached
        .tids
        .iter()
        .enumerate()
        .map(|(i, &tid)| (tid, i))
        .collect();
    let schema = column_types(table);
    // Decode the rows whose tid is new since the build; note every live tid to detect removals,
    // and re-decode kept tids to detect an in-place content change.
    let mut live: std::collections::HashSet<Tid> = std::collections::HashSet::new();
    let mut added: Vec<(Tid, Row)> = Vec::new();
    let mut scan = engine.scan(txn, table.id)?;
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        live.insert(tid);
        match cached_rows.get(&tid) {
            None => added.push((tid, row::decode(&tuple, &schema)?)),
            Some(&i) => {
                if cached.rows.get(i) != Some(&row::decode(&tuple, &schema)?) {
                    // A kept tid's content changed (an in-place UPDATE): the graph holds the old
                    // vector and cannot replace a node → rebuild.
                    return build_vector_index(table, column_ordinal, dim, signature, engine, txn);
                }
            },
        }
    }
    // A cached tid no longer visible means a removal happened → rebuild (HNSW has no node delete).
    if cached.tids.iter().any(|t| !live.contains(t)) {
        return build_vector_index(table, column_ordinal, dim, signature, engine, txn);
    }
    // Pure append: extend the existing graph with the new rows.
    let CachedVectorIndex {
        mut index,
        mut rows,
        mut id_to_row,
        mut tids,
        ..
    } = cached;
    for (tid, row) in added {
        if let Some(ast::Value::Vector(v)) = row.get(column_ordinal)
            && v.len() == dim
        {
            index.insert(v.clone())?;
            id_to_row.push(rows.len());
        }
        tids.push(tid);
        rows.push(row);
    }
    Ok(CachedVectorIndex {
        signature,
        index,
        rows,
        id_to_row,
        tids,
    })
}

/// Exact k-NN fallback when no HNSW index is declared: scan, compute cosine distance to `query` for
/// each row, and return the `k` nearest in distance order — identical to `Sort(<=>) + Limit`.
/// Candidate over-fetch factor for a filtered k-NN search: the HNSW search returns `k ×`
/// this many candidates so that, after a `WHERE` filter removes some, enough survive to fill `k`.
/// A more selective filter is handled by the exact fallback when even the over-fetch falls short.
const FILTER_OVERFETCH: usize = 8;

/// Whether `row` passes the (optional) `WHERE` filter — `true` when there is none, else the
/// predicate must evaluate to exactly `TRUE` (SQL three-valued logic: `NULL`/unknown excludes).
fn row_passes(filter: Option<&TypedExpr>, row: &Row) -> Result<bool, Error> {
    match filter {
        None => Ok(true),
        Some(pred) => Ok(matches!(eval::eval(pred, row)?, ast::Value::Bool(true))),
    }
}

fn exact_vector_knn(
    table: &TableSchema,
    column_ordinal: usize,
    query: &[f32],
    k: usize,
    filter: Option<&TypedExpr>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let rows = super::scan::scan_rows(table, engine, txn)?;
    let mut scored: Vec<(f64, Row)> = Vec::new();
    for row in rows {
        if let Some(ast::Value::Vector(v)) = row.get(column_ordinal)
            && let Some(dist) = crate::vector::cosine_distance(query, v)
            && row_passes(filter, &row)?
        {
            scored.push((dist, row));
        }
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(k);
    Ok(scored.into_iter().map(|(_, row)| row).collect())
}

/// Search a cached HNSW index for the `k` nearest rows that pass the optional filter. The
/// search over-fetches `want` candidates (≥ `k`) so the post-filter can still fill `k`; it maps each
/// hit's graph id back to its row and keeps the first `k` that pass. Caller holds the cache lock.
fn search_cached(
    cached: &CachedVectorIndex,
    query_vec: &[f32],
    want: usize,
    ef: usize,
    k: usize,
    filter: Option<&TypedExpr>,
) -> Result<Vec<Row>, Error> {
    let hits = cached.index.search(query_vec, want, ef)?;
    let mut out = Vec::with_capacity(k.min(hits.len()));
    for (id, _) in hits {
        let Some(&row_idx) = cached.id_to_row.get(id as usize) else {
            continue;
        };
        let Some(row) = cached.rows.get(row_idx) else {
            continue;
        };
        if row_passes(filter, row)? {
            out.push(row.clone());
            if out.len() == k {
                break;
            }
        }
    }
    Ok(out)
}

/// Execute a [`PhysicalOperator::VectorKnn`]: return the `k` rows of `table` nearest to the
/// query vector under cosine distance that also pass the optional `WHERE` filter. Uses the declared
/// HNSW index (approximate, cached) when one exists, otherwise an exact scan. With a filter the
/// index search over-fetches and post-filters, falling back to an exact filtered scan if too few
/// candidates survive — so a selective filter never under-returns. Either way the result is the `k`
/// nearest matching rows in ascending-distance order.
#[allow(
    clippy::significant_drop_tightening,
    reason = "the cache lock guard must stay held while `search_cached` borrows the cached index"
)]
fn run_vector_knn(
    table: &TableSchema,
    column_ordinal: usize,
    query: &TypedExpr,
    k: u64,
    filter: Option<&TypedExpr>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    // The query vector is a constant (the planner only routes a column-free query expr): evaluate it
    // against an empty row. A NULL (or otherwise non-vector) query matches nothing.
    let ast::Value::Vector(query_vec) = eval::eval(query, &Vec::new())? else {
        return Ok(Vec::new());
    };
    let k = usize::try_from(k).unwrap_or(usize::MAX);
    if k == 0 {
        return Ok(Vec::new());
    }
    // A `WHERE` filter over a single table carries only uncorrelated subqueries (no enclosing query),
    // so resolve them once up front; the resolved predicate is then a plain per-row eval.
    let resolved_filter = filter.map(|f| resolved_expr(f, engine, txn)).transpose()?;
    let filter = resolved_filter.as_deref();
    // No HNSW index on this column → exact scan (still correct, just O(n)).
    let Some(entry) = super::vector_index_for_column(engine, txn, &table.name, column_ordinal)?
    else {
        return exact_vector_knn(table, column_ordinal, &query_vec, k, filter, engine, txn);
    };
    // A query whose dimension does not match the index falls back to exact (which simply finds no
    // comparable rows); the analyzer already enforces matching dimensions for `<=>`, so this is a
    // defensive guard rather than an expected path.
    if entry.dim != query_vec.len() {
        return exact_vector_knn(table, column_ordinal, &query_vec, k, filter, engine, txn);
    }
    // Over-fetch candidates when filtering so enough survive the post-filter to fill `k`.
    let want = if filter.is_some() {
        k.saturating_mul(FILTER_OVERFETCH)
    } else {
        k
    };
    let ef = resolve_ef_search(want);
    let key = (engine_identity(engine), table.id, column_ordinal);
    let current = scan_signature(table, engine, txn)?;
    let slot = hnsw_slot(key);
    // Warm path: the slot's shared read lock lets concurrent queries search a fresh index in parallel,
    // and a build of another key's slot does not block this one.
    let knn = {
        let guard = slot
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match guard.as_ref() {
            Some(cached) if cached.signature == current => {
                Some(search_cached(cached, &query_vec, want, ef, k, filter)?)
            },
            _ => None,
        }
    };
    // Cold/stale: take this slot's exclusive write lock to (re)build, then search the built index.
    let knn = if let Some(rows) = knn {
        rows
    } else {
        let mut guard = slot
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Re-check under the write lock — another thread may have built it while we waited.
        let stale = guard.as_ref().is_none_or(|c| c.signature != current);
        if stale {
            // Reuse the cached graph for a pure append (incremental insert); otherwise rebuild.
            let built = match guard.take() {
                Some(cached) => maintain_vector_index(
                    cached,
                    table,
                    column_ordinal,
                    entry.dim,
                    current,
                    engine,
                    txn,
                )?,
                None => build_vector_index(table, column_ordinal, entry.dim, current, engine, txn)?,
            };
            *guard = Some(built);
        }
        let cached = guard.as_ref().ok_or_else(|| {
            Error::Unsupported("vector index cache missing after build".to_owned())
        })?;
        search_cached(cached, &query_vec, want, ef, k, filter)?
    };
    // A selective filter can starve the over-fetched candidate set; an exact filtered scan then
    // guarantees the true k nearest matching rows rather than under-returning.
    if filter.is_some() && knn.len() < k {
        return exact_vector_knn(table, column_ordinal, &query_vec, k, filter, engine, txn);
    }
    Ok(knn)
}

/// Evaluate a set-returning function for one input row into its element list. `UNNEST(arr)`
/// yields the array's elements in order; a `NULL` (or non-array, defensively) array yields nothing.
fn eval_set_returning(
    func: ast::SetReturningFunc,
    args: &[TypedExpr],
    row: &Row,
) -> Result<Vec<ast::Value>, Error> {
    use ast::SetReturningFunc as Srf;
    // The first argument is the value being expanded; arity was fixed by the analyzer.
    let first = args
        .first()
        .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
    match func {
        // UNNEST(arr) → each element; a NULL/non-array yields no rows.
        Srf::Unnest => Ok(match first {
            ast::Value::Array(items) => items,
            _ => Vec::new(),
        }),
        // JSON_ARRAY_ELEMENTS(json) → each element as JSON; a NULL/non-array yields no rows.
        Srf::JsonArrayElements => Ok(match first {
            ast::Value::Json(doc) => crate::json::array_elements(&doc)
                .unwrap_or_default()
                .into_iter()
                .map(ast::Value::Json)
                .collect(),
            _ => Vec::new(),
        }),
        // JSONB_OBJECT_KEYS(json) → each top-level key as TEXT; a NULL/non-object yields no rows.
        Srf::JsonObjectKeys => Ok(match first {
            ast::Value::Json(doc) => crate::json::object_keys(&doc)
                .unwrap_or_default()
                .into_iter()
                .map(ast::Value::Text)
                .collect(),
            _ => Vec::new(),
        }),
        // JSONB_ARRAY_ELEMENTS_TEXT(json) → each element as TEXT (a JSON null becomes SQL NULL); a
        // NULL/non-array document yields no rows.
        Srf::JsonArrayElementsText => Ok(match first {
            ast::Value::Json(doc) => crate::json::array_elements_text(&doc)
                .unwrap_or_default()
                .into_iter()
                .map(|e| e.map_or(ast::Value::Null, ast::Value::Text))
                .collect(),
            _ => Vec::new(),
        }),
        // STRING_TO_TABLE(s, sep) → one TEXT row per split piece; a NULL argument yields no rows.
        Srf::StringToTable => {
            let sep = args
                .get(1)
                .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
            let (ast::Value::Text(s), ast::Value::Text(sep)) = (first, sep) else {
                return Ok(Vec::new());
            };
            Ok(eval::split_on_literal(&s, &sep)
                .into_iter()
                .map(ast::Value::Text)
                .collect())
        },
        // REGEXP_SPLIT_TO_TABLE(s, pattern [, flags]) → one TEXT row per split piece; a NULL argument
        // yields no rows.
        Srf::RegexpSplitToTable => {
            let pattern = args
                .get(1)
                .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
            let flags = match args.get(2) {
                Some(a) => eval::eval(a, row)?,
                None => ast::Value::Text(String::new()),
            };
            let (ast::Value::Text(s), ast::Value::Text(pat), ast::Value::Text(flags)) =
                (first, pattern, flags)
            else {
                return Ok(Vec::new());
            };
            Ok(eval::regexp_split_pieces(&s, &pat, &flags)?
                .into_iter()
                .map(ast::Value::Text)
                .collect())
        },
        // REGEXP_MATCHES(s, pattern [, flags]) → one TEXT[] row per match; the `g` flag returns every
        // match, else only the first. A NULL argument yields no rows.
        Srf::RegexpMatches => eval_regexp_matches(first, args, row),
        // JSONB_PATH_QUERY(json, path) → each match as JSON. A NULL document/path yields no rows; an
        // unparseable path (outside the supported subset) is a runtime error.
        Srf::JsonPathQuery => {
            let path = args
                .get(1)
                .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
            let (ast::Value::Json(doc), ast::Value::Text(path)) = (first, path) else {
                return Ok(Vec::new());
            };
            let matches = crate::json::path_query(&doc, &path).ok_or_else(|| {
                Error::Unsupported(format!(
                    "jsonb_path_query: unsupported or invalid jsonpath `{path}`"
                ))
            })?;
            Ok(matches.into_iter().map(ast::Value::Json).collect())
        },
        // GENERATE_SERIES(start, stop [, step]) → one INT per value in the series. A NULL argument
        // yields no rows.
        Srf::GenerateSeries => {
            let stop = args
                .get(1)
                .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
            let step = match args.get(2) {
                Some(a) => eval::eval(a, row)?,
                None => ast::Value::Int(1),
            };
            // The temporal form: a DATE/TIMESTAMP[TZ] range stepped by an INTERVAL → one timestamp per
            // row. A `DATE` bound is taken at midnight; a TIMESTAMPTZ range keeps
            // the timezone-aware element type.
            if let (Some(start_us), Some(stop_us), ast::Value::Interval(step)) =
                (temporal_micros(&first), temporal_micros(&stop), &step)
            {
                let tzaware = matches!(first, ast::Value::TimestampTz(_));
                return generate_series_temporal(start_us, stop_us, step, tzaware);
            }
            let (ast::Value::Int(start), ast::Value::Int(stop), ast::Value::Int(step)) =
                (first, stop, step)
            else {
                return Ok(Vec::new());
            };
            generate_series(start, stop, step)
        },
    }
}

/// Evaluate `REGEXP_MATCHES(s, pattern [, flags])` into one `TEXT[]` row per match (`first` is the
/// already-evaluated source argument). A NULL argument yields no rows.
fn eval_regexp_matches(
    first: ast::Value,
    args: &[TypedExpr],
    row: &Row,
) -> Result<Vec<ast::Value>, Error> {
    let pattern = args
        .get(1)
        .map_or(Ok(ast::Value::Null), |a| eval::eval(a, row))?;
    let flags = match args.get(2) {
        Some(a) => eval::eval(a, row)?,
        None => ast::Value::Text(String::new()),
    };
    let (ast::Value::Text(s), ast::Value::Text(pat), ast::Value::Text(flags)) =
        (first, pattern, flags)
    else {
        return Ok(Vec::new());
    };
    eval::regexp_all_matches(&s, &pat, &flags)
}

/// `GENERATE_SERIES(start, stop, step)` — the inclusive integer series from `start` to `stop`.
/// A zero step is an error; the row count is capped to guard against a runaway materialization, and
/// the walk stops cleanly if the next value would overflow `i64`.
fn generate_series(start: i64, stop: i64, step: i64) -> Result<Vec<ast::Value>, Error> {
    const MAX_ROWS: usize = 10_000_000;
    if step == 0 {
        return Err(Error::Unsupported(
            "generate_series: step must not be zero".to_owned(),
        ));
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        let in_range = if step > 0 { cur <= stop } else { cur >= stop };
        if !in_range {
            break;
        }
        if out.len() >= MAX_ROWS {
            return Err(Error::Unsupported(format!(
                "generate_series: the series exceeds the {MAX_ROWS}-row limit"
            )));
        }
        out.push(ast::Value::Int(cur));
        match cur.checked_add(step) {
            Some(next) => cur = next,
            None => break,
        }
    }
    Ok(out)
}

/// Microseconds since the epoch for a temporal `generate_series` bound (a `DATE` is taken at
/// midnight), or `None` for a non-temporal value.
fn temporal_micros(v: &ast::Value) -> Option<i64> {
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    match v {
        ast::Value::Timestamp(t) | ast::Value::TimestampTz(t) => Some(*t),
        ast::Value::Date(d) => i64::from(*d).checked_mul(MICROS_PER_DAY),
        _ => None,
    }
}

/// `GENERATE_SERIES(start, stop, interval step)` over a temporal range → one timestamp per row.
/// The step direction is the sign of one applied interval (a month-bearing step is
/// non-uniform, so the direction is taken from its effect on `start`); a step that does not advance is
/// an error. Row count is capped like the integer series, and the walk stops if a step stops advancing
/// (saturating overflow).
fn generate_series_temporal(
    start: i64,
    stop: i64,
    step: &crate::interval::Interval,
    tzaware: bool,
) -> Result<Vec<ast::Value>, Error> {
    const MAX_ROWS: usize = 10_000_000;
    let next_of =
        |t: i64| crate::temporal::add_interval_to_micros(t, step.months, step.days, step.micros);
    if next_of(start) == start {
        return Err(Error::Unsupported(
            "generate_series: step must not be zero".to_owned(),
        ));
    }
    let forward = next_of(start) > start;
    let wrap = |t: i64| {
        if tzaware {
            ast::Value::TimestampTz(t)
        } else {
            ast::Value::Timestamp(t)
        }
    };
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        let in_range = if forward { cur <= stop } else { cur >= stop };
        if !in_range {
            break;
        }
        if out.len() >= MAX_ROWS {
            return Err(Error::Unsupported(format!(
                "generate_series: the series exceeds the {MAX_ROWS}-row limit"
            )));
        }
        out.push(wrap(cur));
        let next = next_of(cur);
        if (forward && next <= cur) || (!forward && next >= cur) {
            break;
        }
        cur = next;
    }
    Ok(out)
}

/// Produce the metadata rows for an `information_schema` view. Each view is a synthetic
/// table whose rows come from engine introspection (`list_tables`, `lookup_table`, view definitions)
/// rather than from storage. The schema of the returned rows matches the view's column definitions
/// in [`InfoSchemaView::table_schema`](crate::planner::InfoSchemaView::table_schema).
///
/// # Errors
/// Propagates engine errors from table listing and schema lookup.
pub(super) fn run_info_schema(
    view: crate::planner::InfoSchemaView,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    use crate::planner::InfoSchemaView as V;
    match view {
        V::Tables => info_schema_tables(engine, txn),
        V::Columns => info_schema_columns(engine, txn),
        V::Schemata => info_schema_schemata(engine),
        V::Views => info_schema_views(engine, txn),
        V::TableConstraints => info_schema_table_constraints(engine, txn),
        V::KeyColumnUsage => info_schema_key_column_usage(engine, txn),
        V::Statistics => info_schema_statistics(engine, txn),
    }
}

/// Map a [`ConstraintKind`](nusadb_core::engine::ConstraintKind) to the SQL-standard
/// `table_constraints.constraint_type` text.
const fn constraint_type_name(kind: nusadb_core::engine::ConstraintKind) -> &'static str {
    use nusadb_core::engine::ConstraintKind as K;
    match kind {
        K::PrimaryKey => "PRIMARY KEY",
        K::Unique => "UNIQUE",
        K::ForeignKey => "FOREIGN KEY",
        K::Check => "CHECK",
    }
}

/// `information_schema.table_constraints`: one row per PK/UNIQUE/FK/CHECK constraint of every user
/// table, sourced from the engine's catalog.
fn info_schema_table_constraints(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    for name in &engine.list_tables_as_of(txn)? {
        if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
            continue;
        }
        let Some(schema) = engine.lookup_table(name)? else {
            continue;
        };
        for c in engine.list_constraints(schema.id)? {
            // Synthetic type-bound checks (a VARCHAR(n) length or narrow-int range) are an
            // implementation detail of the declared type, not a user constraint — enforced on every
            // write but hidden from introspection.
            if c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX) {
                continue;
            }
            rows.push(vec![
                ast::Value::Text("nusadb".to_owned()),
                ast::Value::Text("public".to_owned()),
                ast::Value::Text(c.name),
                ast::Value::Text("nusadb".to_owned()),
                ast::Value::Text("public".to_owned()),
                ast::Value::Text(name.clone()),
                ast::Value::Text(constraint_type_name(c.kind).to_owned()),
            ]);
        }
    }
    Ok(rows)
}

/// `information_schema.key_column_usage`: one row per (key constraint, key column) with the column's
/// 1-based position in the key. `CHECK` constraints have no key columns, so they do not
/// appear. Backs JDBC `getPrimaryKeys`.
fn info_schema_key_column_usage(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    for name in &engine.list_tables_as_of(txn)? {
        if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
            continue;
        }
        let Some(schema) = engine.lookup_table(name)? else {
            continue;
        };
        for c in engine.list_constraints(schema.id)? {
            for (pos, col) in c.columns.iter().enumerate() {
                rows.push(vec![
                    ast::Value::Text("nusadb".to_owned()),
                    ast::Value::Text("public".to_owned()),
                    ast::Value::Text(c.name.clone()),
                    ast::Value::Text("nusadb".to_owned()),
                    ast::Value::Text("public".to_owned()),
                    ast::Value::Text(name.clone()),
                    ast::Value::Text(col.clone()),
                    ast::Value::Int(i64::try_from(pos + 1).unwrap_or(0)),
                ]);
            }
        }
    }
    Ok(rows)
}

/// `information_schema.statistics`: one row per (index, key column) with the column's 1-based
/// position — an index-metadata view. `non_unique` is 0 for a unique index, 1 otherwise.
/// Backs JDBC `getIndexInfo`.
fn info_schema_statistics(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    for name in &engine.list_tables_as_of(txn)? {
        if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
            continue;
        }
        let Some(schema) = engine.lookup_table(name)? else {
            continue;
        };
        for idx in engine.list_indexes(schema.id)? {
            let non_unique = i64::from(!idx.unique);
            for (pos, col) in idx.columns.iter().enumerate() {
                rows.push(vec![
                    ast::Value::Text("nusadb".to_owned()),
                    ast::Value::Text("public".to_owned()),
                    ast::Value::Text(name.clone()),
                    ast::Value::Int(non_unique),
                    ast::Value::Text(idx.name.clone()),
                    ast::Value::Int(i64::try_from(pos + 1).unwrap_or(0)),
                    ast::Value::Text(col.clone()),
                ]);
            }
        }
    }
    Ok(rows)
}

/// `information_schema.tables`: one row per user table or view.
fn info_schema_tables(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let names = engine.list_tables_as_of(txn)?;
    let mut rows = Vec::with_capacity(names.len());
    for name in &names {
        // Skip system tables (nusadb_* prefix) from the engine's list.
        if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
            continue;
        }
        rows.push(vec![
            ast::Value::Text("nusadb".to_owned()),
            ast::Value::Text("public".to_owned()),
            ast::Value::Text(name.clone()),
            ast::Value::Text("BASE TABLE".to_owned()),
        ]);
    }
    // Views: non-materialized views are stored in `nusadb_views`; materialized views have a backing
    // table (listed above) but also appear in `nusadb_matviews`. We can't distinguish them here
    // without a catalog scan, so omit a dedicated views entry — the engine's list_tables already
    // includes materialized-view backing tables as BASE TABLE, which is correct for now.
    Ok(rows)
}

/// `information_schema.columns`: one row per column of every user table.
fn info_schema_columns(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let names = engine.list_tables_as_of(txn)?;
    let mut rows = Vec::new();
    for name in &names {
        if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
            continue;
        }
        let Some(schema) = engine.lookup_table(name)? else {
            continue;
        };
        // The column DEFAULT expressions live in the `nusadb_column_defaults` catalog, keyed by
        // (table, column) — not in the schema's ColumnDef.
        let defaults = crate::executor::coldefault::load_defaults(name, engine, txn)?;
        for (pos, col) in schema.columns.iter().enumerate() {
            // Standard `information_schema.columns.data_type` name (e.g. `integer`, `character
            // varying`) so driver/ORM reflection maps the type correctly — distinct from the
            // short `SHOW COLUMNS` spelling. The reflection columns are NULL when not applicable.
            let nullable_int =
                |v: Option<u32>| v.map_or(ast::Value::Null, |n| ast::Value::Int(i64::from(n)));
            // `column_default` is the stored DEFAULT SQL, or NULL when the column has none. A SERIAL /
            // IDENTITY column's default is stored as a sentinel; render it as `nextval('<seq>')`.
            let column_default = defaults.iter().find(|(c, _)| *c == col.name).map_or(
                ast::Value::Null,
                |(_, sql)| {
                    crate::executor::coldefault::serial_sequence(sql).map_or_else(
                        || ast::Value::Text(sql.clone()),
                        |seq| ast::Value::Text(format!("nextval('{seq}')")),
                    )
                },
            );
            rows.push(vec![
                ast::Value::Text("nusadb".to_owned()),
                ast::Value::Text("public".to_owned()),
                ast::Value::Text(name.clone()),
                ast::Value::Text(col.name.clone()),
                ast::Value::Int(i64::try_from(pos + 1).unwrap_or(0)),
                ast::Value::Text(info_schema_data_type(col.ty).to_owned()),
                ast::Value::Text(if col.nullable { "YES" } else { "NO" }.to_owned()),
                nullable_int(char_max_length(col.ty)),
                nullable_int(numeric_precision(col.ty)),
                nullable_int(numeric_scale(col.ty)),
                column_default,
            ]);
        }
    }
    Ok(rows)
}

/// The standard `information_schema.columns.data_type` name for a column type. These are the
/// SQL-standard / common spellings drivers and ORMs expect (`integer`, not `int`), distinct from the
/// short names `SHOW COLUMNS` renders. `SMALLINT`/`BIGINT` round-trip to their own names (K4/K6
/// integer fidelity); `INT` and the unnamed narrow widths (`TINYINT`/`MEDIUMINT`) report `integer`.
pub(crate) const fn info_schema_data_type(ty: ColumnType) -> &'static str {
    match ty {
        ColumnType::Bool => "boolean",
        ColumnType::Int => "integer",
        ColumnType::SmallInt => "smallint",
        ColumnType::BigInt => "bigint",
        ColumnType::Float => "double precision",
        ColumnType::Real => "real",
        ColumnType::Text => "text",
        ColumnType::VarChar(_) => "character varying",
        ColumnType::Char(_) => "character",
        ColumnType::Bytes => "bytea",
        ColumnType::Timestamp => "timestamp without time zone",
        ColumnType::TimestampTz => "timestamp with time zone",
        ColumnType::Date => "date",
        ColumnType::Time => "time without time zone",
        ColumnType::TimeTz => "time with time zone",
        ColumnType::Uuid => "uuid",
        ColumnType::Numeric { .. } => "numeric",
        ColumnType::Json => "json",
        ColumnType::Jsonb => "jsonb",
        ColumnType::Interval => "interval",
        ColumnType::Array(_) => "ARRAY",
        ColumnType::Vector(_) => "vector",
    }
}

/// `character_maximum_length` for a `VARCHAR(n)` / `CHAR(n)` column; `None` for every other type.
const fn char_max_length(ty: ColumnType) -> Option<u32> {
    match ty {
        ColumnType::VarChar(n) | ColumnType::Char(n) => Some(n),
        _ => None,
    }
}

/// `numeric_precision` for a `NUMERIC(p, s)` column (`None` when unconstrained or non-numeric).
const fn numeric_precision(ty: ColumnType) -> Option<u32> {
    match ty {
        ColumnType::Numeric { precision, .. } if precision > 0 => Some(precision as u32),
        _ => None,
    }
}

/// `numeric_scale` for a `NUMERIC(p, s)` column (`None` when unconstrained or non-numeric).
const fn numeric_scale(ty: ColumnType) -> Option<u32> {
    match ty {
        ColumnType::Numeric { precision, scale } if precision > 0 => Some(scale as u32),
        _ => None,
    }
}

/// `information_schema.schemata`: one row per schema — the implicit `public` namespace, the synthetic
/// `information_schema` catalog namespace, plus every `CREATE SCHEMA` namespace. Emitted sorted
/// by name for deterministic output.
fn info_schema_schemata(engine: &dyn StorageEngine) -> Result<Vec<Row>, Error> {
    // `public` and the synthetic `information_schema` always exist; created namespaces add to them.
    let mut names: Vec<String> = vec![
        nusadb_core::PUBLIC_SCHEMA.to_owned(),
        "information_schema".to_owned(),
    ];
    names.extend(engine.list_schemas()?.into_iter().map(|(_, n)| n));
    names.sort();
    Ok(names
        .iter()
        .map(|schema| {
            vec![
                ast::Value::Text("nusadb".to_owned()),
                ast::Value::Text(schema.clone()),
                ast::Value::Text(crate::BOOTSTRAP_SUPERUSER.to_owned()),
            ]
        })
        .collect())
}

/// `information_schema.views`: one row per non-materialized view with its defining SQL. Materialized
/// views are intentionally excluded (per the SQL standard they are not `information_schema.views`;
/// they surface as `BASE TABLE`s in `information_schema.tables` via their backing table).
fn info_schema_views(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let mut rows = Vec::new();
    // Non-materialized views are stored in the `nusadb_views` catalog table; scan it directly.
    if let Some(cat) = engine.lookup_table(super::VIEW_CATALOG)? {
        let schema = [nusadb_core::ColumnType::Text, nusadb_core::ColumnType::Text];
        let mut scan = engine.scan(txn, cat.id)?;
        while let Some((_, bytes)) = scan.try_next()? {
            let row = super::row::decode(&bytes, &schema)?;
            if let [ast::Value::Text(n), ast::Value::Text(def)] = row.as_slice() {
                rows.push(vec![
                    ast::Value::Text("nusadb".to_owned()),
                    ast::Value::Text("public".to_owned()),
                    ast::Value::Text(n.clone()),
                    ast::Value::Text(def.clone()),
                ]);
            }
        }
    }
    Ok(rows)
}

/// A last-resort backstop on recursive-CTE rounds, for the case where the work-memory budget is
/// unlimited (`work_mem = 0`) AND no statement timeout is set — without it a genuinely
/// non-terminating recursion producing tiny rows could loop forever. It is deliberately far above
/// any legitimate depth (a correct 20k-deep chain took ~20k rounds), so it never rejects a valid
/// recursion; the real bounds are the `work_mem` memory cap below and the per-round cancel/timeout
/// check. (Was `10_000`, which false-rejected valid deep recursion as "non-terminating".)
const MAX_RECURSIVE_ITERATIONS: usize = 10_000_000;

/// Execute a `WITH RECURSIVE` body: materialize every CTE to a fixpoint, bind each result to
/// its synthetic table, then run `body` over those bindings. The bindings are dropped (restoring any
/// previous ones) once the body has produced its rows.
fn run_recursive_cte(
    ctes: &[PhysicalRecursiveCte],
    body: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    // Materialize each CTE fully before binding any, so the body sees complete results.
    let mut materialized: Vec<(nusadb_core::TableId, Vec<Row>)> = Vec::with_capacity(ctes.len());
    for cte in ctes {
        materialized.push((cte.id, materialize_recursive_cte(cte, engine, txn)?));
    }
    // Bind every CTE's result for the body; the guards restore the prior bindings on drop.
    let _guards: Vec<_> = materialized
        .into_iter()
        .map(|(id, rows)| super::recursive::bind(id, rows))
        .collect();
    execute_op(body, engine, txn)
}

/// Evaluate one recursive CTE to its fixpoint. Run the base term, then repeatedly run the
/// recursive term over the rows produced by the previous round (the "working set"), accumulating
/// until a round adds nothing new. `UNION ALL` keeps every produced row; `UNION` keeps only rows not
/// already in the result (and dedupes the base).
fn materialize_recursive_cte(
    cte: &PhysicalRecursiveCte,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let base = execute_op(&cte.base, engine, txn)?;
    let (mut accumulated, mut working) = if cte.union_all {
        (base.clone(), base)
    } else {
        let deduped = dedupe_rows(base);
        (deduped.clone(), deduped)
    };
    // For UNION (distinct), a hash index over `accumulated` answers "already seen?" in O(1)
    // amortized instead of scanning the whole accumulated result per produced row — the linear
    // `accumulated.iter().any(...)` was O(n²) and could dominate a deep recursion.
    // `group_keys_equal` stays the authoritative tie-break within a hash bucket. UNION ALL keeps
    // every row and needs no membership test, so the index is built only for the distinct path.
    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    if !cte.union_all {
        for (idx, row) in accumulated.iter().enumerate() {
            seen.entry(super::agg::group_key_hash(row))
                .or_default()
                .push(idx);
        }
    }
    // Bound the recursion by memory, not an arbitrary round count: track the accumulated result's
    // byte size incrementally (O(1) per new row) and stop loudly if it exceeds `work_mem`, exactly
    // like every other materializing stage. This lets an arbitrarily deep-but-finite recursion run
    // (its rows fit in the budget) while a runaway that keeps producing new rows is caught before it
    // exhausts memory — and it is configurable via `SET work_mem` / `--work-mem`
    // (statement-effective: the session value must win here, not only the process default).
    let budget = effective_work_mem();
    let mut accumulated_bytes: usize = accumulated.iter().map(|r| row_bytes(r)).sum();
    let mut iterations = 0;
    while !working.is_empty() {
        // A long or non-terminating recursion is abortable by a statement timeout / cancel request.
        crate::cancel::check()?;
        iterations += 1;
        if iterations > MAX_RECURSIVE_ITERATIONS {
            return Err(Error::Unsupported(format!(
                "recursive CTE exceeded the {MAX_RECURSIVE_ITERATIONS}-round safety backstop \
                 (likely a non-terminating recursion under an unlimited work_mem and no statement \
                 timeout) — add a termination condition, set a statement timeout, or a work_mem cap"
            )));
        }
        // Bind the working set so the recursive term's self-reference scans exactly those rows.
        let produced = {
            let _guard = super::recursive::bind(cte.id, working.clone());
            execute_op(&cte.recursive, engine, txn)?
        };
        if cte.union_all {
            if produced.is_empty() {
                break;
            }
            accumulated_bytes += produced.iter().map(|r| row_bytes(r)).sum::<usize>();
            accumulated.extend(produced.iter().cloned());
            working = produced;
        } else {
            // Keep only genuinely new rows (not already in the accumulated result); they become the
            // next working set so the recursion makes progress toward the fixpoint.
            let mut new_rows = Vec::new();
            for row in dedupe_rows(produced) {
                let h = super::agg::group_key_hash(&row);
                let already = seen.get(&h).is_some_and(|bucket| {
                    bucket.iter().any(|&i| {
                        accumulated
                            .get(i)
                            .is_some_and(|s| group_keys_equal(s, &row))
                    })
                });
                if !already {
                    accumulated_bytes += row_bytes(&row);
                    seen.entry(h).or_default().push(accumulated.len());
                    accumulated.push(row.clone());
                    new_rows.push(row);
                }
            }
            if new_rows.is_empty() {
                break;
            }
            working = new_rows;
        }
        if budget != 0 && accumulated_bytes > budget {
            return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                "recursive CTE exceeded work_mem of {budget} bytes ({accumulated_bytes} bytes \
                 accumulated) — add a termination condition, a more selective query, or raise \
                 work_mem (SET work_mem / --work-mem)"
            ))));
        }
    }
    Ok(accumulated)
}

/// Window-function evaluation: compute one column per [`WindowExpr`]
/// and append it to every row, preserving input order. Each window is evaluated
/// independently over its own `PARTITION BY` / `ORDER BY`. Linear partition
/// search and prefix re-folding keep it simple (a single-pass design is a perf
/// follow-up, mirroring the other row-based operators).
pub(super) fn run_window(input_rows: Vec<Row>, windows: &[WindowExpr]) -> Result<Vec<Row>, Error> {
    // One result column per window, each holding a value per input row in input order.
    let mut columns: Vec<Vec<ast::Value>> = Vec::with_capacity(windows.len());
    for window in windows {
        columns.push(compute_window(&input_rows, window)?);
    }
    let mut out = Vec::with_capacity(input_rows.len());
    for (i, mut row) in input_rows.into_iter().enumerate() {
        for column in &columns {
            row.push(column.get(i).cloned().unwrap_or(ast::Value::Null));
        }
        out.push(row);
    }
    Ok(out)
}

/// Compute one window function's value for every input row (returned in input
/// order). Rows are bucketed by `PARTITION BY`; within a partition they are
/// ordered by the window `ORDER BY` and the function is applied with the default
/// frame — the whole partition for an unordered aggregate, or running through the
/// current peer group (RANGE … CURRENT ROW) for an ordered one.
pub(super) fn compute_window(rows: &[Row], window: &WindowExpr) -> Result<Vec<ast::Value>, Error> {
    use crate::ast::WindowFunc as W;

    let mut result = vec![ast::Value::Null; rows.len()];

    // Bucket original row indices by partition key (first-seen order, NULL-not-distinct). A hash
    // index maps each partition-key hash to the partition indices sharing it, so a row finds its
    // partition in O(1) amortized instead of scanning every partition key seen so far — the linear
    // `partition_keys.iter().position(...)` was O(n × P) and dominated `PARTITION BY` over many
    // partitions (614x vs the reference at 10k partitions). `group_keys_equal` stays the
    // authoritative tie-break within a hash bucket, so a collision costs one comparison, never
    // correctness, and first-seen partition order (hence output order) is unchanged.
    let mut partition_keys: Vec<Vec<ast::Value>> = Vec::new();
    let mut partitions: Vec<Vec<usize>> = Vec::new();
    let mut index: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        // Amortized cooperative-cancel poll: a window over a large input evaluates its partition
        // keys, per-partition sort, and frame values entirely on-CPU, so poll the deadline here (and
        // per bucket below) rather than ignoring it until the whole window finishes.
        if i.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        let key = window
            .partition
            .iter()
            .map(|e| eval::eval(e, row))
            .collect::<Result<Vec<_>, _>>()?;
        let bucket = index.entry(super::agg::group_key_hash(&key)).or_default();
        if let Some(&p) = bucket.iter().find(|&&p| {
            partition_keys
                .get(p)
                .is_some_and(|k| group_keys_equal(k, &key))
        }) {
            if let Some(rows_in) = partitions.get_mut(p) {
                rows_in.push(i);
            }
        } else {
            bucket.push(partition_keys.len());
            partition_keys.push(key);
            partitions.push(vec![i]);
        }
    }

    for bucket in &partitions {
        crate::cancel::check()?;
        // Order the partition's rows by the window ORDER BY (stable; identity when unordered).
        let mut ordered: Vec<(Vec<ast::Value>, usize)> = Vec::with_capacity(bucket.len());
        for &i in bucket {
            let Some(row) = rows.get(i) else { continue };
            let key = window
                .order
                .iter()
                .map(|k| eval::eval(&k.expr, row))
                .collect::<Result<Vec<_>, _>>()?;
            ordered.push((key, i));
        }
        ordered.sort_by(|a, b| {
            for (idx, (av, bv)) in a.0.iter().zip(&b.0).enumerate() {
                let (ascending, nulls) = window
                    .order
                    .get(idx)
                    .map_or((true, ast::NullOrdering::Default), |k| {
                        (k.ascending, k.nulls)
                    });
                let ord = eval::compare_order_key(av, bv, ascending, nulls);
                if !ord.is_eq() {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        match &window.func {
            W::RowNumber | W::Rank | W::DenseRank => {
                assign_ranking(&ordered, &window.func, &mut result);
            },
            W::Aggregate(func) => {
                let call = AggregateCall {
                    func: *func,
                    arg: window.args.first().cloned(),
                    result_ty: window.result_ty,
                    // Window aggregates do not carry DISTINCT, FILTER, an ordered-set
                    // fraction, or the two-argument statistical forms — those are
                    // grouped-aggregate clauses.
                    distinct: false,
                    fraction: None,
                    ordered_set_descending: false,
                    filter: None,
                    separator: None,
                    arg2: None,
                    order_by: Vec::new(),
                    grouping_args: Vec::new(),
                };
                assign_window_aggregate(&ordered, rows, &call, window, &mut result)?;
            },
            W::Lag | W::Lead | W::FirstValue | W::LastValue | W::NthValue => {
                assign_navigation(&ordered, rows, window, &mut result)?;
            },
            W::Ntile | W::CumeDist | W::PercentRank => {
                assign_distribution(&ordered, rows, window, &mut result)?;
            },
        }
    }
    Ok(result)
}

/// Assign ranking values (`ROW_NUMBER`/`RANK`/`DENSE_RANK`) over one ordered
/// partition. `ordered` is `(order-key values, original row index)` in window
/// order; ties (equal order keys) are peers.
pub(super) fn assign_ranking(
    ordered: &[(Vec<ast::Value>, usize)],
    func: &ast::WindowFunc,
    result: &mut [ast::Value],
) {
    use crate::ast::WindowFunc as W;
    let mut rank = 0usize;
    let mut dense = 0usize;
    let mut prev: Option<&Vec<ast::Value>> = None;
    for (position, (key, orig)) in ordered.iter().enumerate() {
        if prev.is_none_or(|p| !group_keys_equal(p, key)) {
            dense += 1;
            rank = position + 1;
        }
        let value = if matches!(func, W::RowNumber) {
            position + 1
        } else if matches!(func, W::Rank) {
            rank
        } else {
            dense
        };
        if let Some(cell) = result.get_mut(*orig) {
            *cell = ast::Value::Int(i64::try_from(value).unwrap_or(i64::MAX));
        }
        prev = Some(key);
    }
}

/// The inclusive ordered-partition index range `[start, end]` of the peer group (equal `ORDER BY`
/// keys) containing position `k`.
fn peer_group(ordered: &[(Vec<ast::Value>, usize)], k: usize) -> (usize, usize) {
    let key = ordered.get(k).map(|(key, _)| key);
    let mut start = k;
    while start > 0
        && ordered
            .get(start - 1)
            .map(|(key, _)| key)
            .zip(key)
            .is_some_and(|(a, b)| group_keys_equal(a, b))
    {
        start -= 1;
    }
    let mut end = k;
    while ordered
        .get(end + 1)
        .map(|(key, _)| key)
        .zip(key)
        .is_some_and(|(a, b)| group_keys_equal(a, b))
    {
        end += 1;
    }
    (start, end)
}

/// Walk `n` peer groups from the current peer group `[cur_lo, cur_hi]` for a `GROUPS` frame bound:
/// `forward` follows (later groups), else precedes (earlier groups). Returns the resulting group's
/// `lo` for a start bound or `hi` for an end bound, as a partition index. Walking past the partition
/// edge clamps to the first/last group (the reference engine: an offset beyond the partition is the partition edge).
fn group_step(
    ordered: &[(Vec<ast::Value>, usize)],
    cur_lo: usize,
    cur_hi: usize,
    n: u64,
    forward: bool,
    at_start: bool,
) -> i64 {
    let (mut lo, mut hi) = (cur_lo, cur_hi);
    let len = ordered.len();
    for _ in 0..n {
        if forward {
            if hi + 1 >= len {
                break; // already at the last group
            }
            (lo, hi) = peer_group(ordered, hi + 1);
        } else {
            if lo == 0 {
                break; // already at the first group
            }
            (lo, hi) = peer_group(ordered, lo - 1);
        }
    }
    i64::try_from(if at_start { lo } else { hi }).unwrap_or(i64::MAX)
}

/// The i64 comparison key of a `RANGE` ordering value, or `None` for `NULL` (outside any value range)
/// or an unsupported type. An integer is itself; a `DATE` is its midnight micros; a `TIMESTAMP[TZ]`
/// is its micros — so an integer or temporal ordering both compare as i64 (the analyzer restricts a
/// `RANGE` value offset to those column types).
fn range_key(v: &ast::Value) -> Option<i64> {
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    match v {
        ast::Value::Int(i) => Some(*i),
        ast::Value::Date(d) => i64::from(*d).checked_mul(MICROS_PER_DAY),
        ast::Value::Timestamp(t) | ast::Value::TimestampTz(t) => Some(*t),
        _ => None,
    }
}

/// The boundary i64 key for a `RANGE` bound: the current key minus the offset for `PRECEDING`, plus it
/// for `FOLLOWING`. An integer offset shifts an integer key; an `INTERVAL` offset shifts a temporal
/// (micros) key. `None` on i64 overflow.
fn range_boundary(cur_key: i64, off: &ast::Value, preceding: bool, ascending: bool) -> Option<i64> {
    // The boundary subtracts the offset for a preceding bound under ASC ordering, and for a following
    // bound under DESC ordering (where preceding rows have *larger* keys). Otherwise it adds.
    let subtract = ascending == preceding;
    match off {
        ast::Value::Int(n) => {
            if subtract {
                cur_key.checked_sub(*n)
            } else {
                cur_key.checked_add(*n)
            }
        },
        ast::Value::Interval(iv) => {
            let iv = if subtract { iv.checked_neg()? } else { *iv };
            Some(crate::temporal::add_interval_to_micros(
                cur_key, iv.months, iv.days, iv.micros,
            ))
        },
        _ => None,
    }
}

/// The partition index of a `RANGE` frame bound at `boundary`. The partition is in window-sort order,
/// so keys are non-decreasing for `ASC` and non-increasing for `DESC`. A start bound returns the first
/// row that has reached the boundary, an end bound the last row still within it; the comparison flips
/// with the sort direction (ASC start: `key >= boundary`; DESC start: `key <= boundary`). A row with a
/// `NULL`/non-key value is skipped (outside any value range). Returns `len` (start, none reached) or
/// `-1` (end, none within) for an empty result.
fn range_scan(
    ordered: &[(Vec<ast::Value>, usize)],
    boundary: i64,
    at_start: bool,
    ascending: bool,
) -> i64 {
    let key_at = |j: usize| {
        ordered
            .get(j)
            .and_then(|(keys, _)| keys.first())
            .and_then(range_key)
    };
    let len = ordered.len();
    // A row is "reached"/"within" when its key is on the inside of the boundary for the sort order.
    let in_frame = |key: i64| {
        if at_start == ascending {
            key >= boundary
        } else {
            key <= boundary
        }
    };
    if at_start {
        for j in 0..len {
            if key_at(j).is_some_and(in_frame) {
                return i64::try_from(j).unwrap_or(i64::MAX);
            }
        }
        i64::try_from(len).unwrap_or(i64::MAX)
    } else {
        for j in (0..len).rev() {
            if key_at(j).is_some_and(in_frame) {
                return i64::try_from(j).unwrap_or(i64::MAX);
            }
        }
        -1
    }
}

/// Resolve a window frame to the inclusive `[lo, hi]` ordered-partition index range for the row at
/// position `k`, or `None` if the frame is empty / out of the partition. `ROWS` frames use physical
/// offsets from `k`; `RANGE`/`GROUPS` (peer-based) resolve `CURRENT ROW` to the current peer group;
/// a `GROUPS` offset counts peer groups via [`group_step`]; a `RANGE` value offset ranges over the
/// ordering value via [`range_scan`].
fn frame_bounds(
    frame: &WindowFrame,
    k: usize,
    ordered: &[(Vec<ast::Value>, usize)],
) -> Option<(usize, usize)> {
    let len = ordered.len();
    if len == 0 {
        return None;
    }
    let (peer_lo, peer_hi) = if frame.peer_based {
        peer_group(ordered, k)
    } else {
        (k, k)
    };
    // A `RANGE` value frame ranges over the single (integer/temporal) ordering column, ASC or DESC.
    // A NULL current ordering value frames only its peers (the reference engine), since the offset arithmetic is undefined.
    let range = matches!(
        frame.start,
        FrameBound::RangePreceding(_) | FrameBound::RangeFollowing(_)
    ) || matches!(
        frame.end,
        FrameBound::RangePreceding(_) | FrameBound::RangeFollowing(_)
    );
    let cur_key = ordered
        .get(k)
        .and_then(|(keys, _)| keys.first())
        .and_then(range_key);
    if range && cur_key.is_none() {
        return Some((peer_lo, peer_hi));
    }
    let ki = i64::try_from(k).unwrap_or(i64::MAX);
    let len = i64::try_from(len).unwrap_or(i64::MAX);
    let off = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
    let pos = |bound: &FrameBound, at_start: bool| -> i64 {
        match bound {
            FrameBound::UnboundedPreceding => 0,
            FrameBound::UnboundedFollowing => len - 1,
            FrameBound::CurrentRow if frame.peer_based => {
                i64::try_from(if at_start { peer_lo } else { peer_hi }).unwrap_or(i64::MAX)
            },
            FrameBound::CurrentRow => ki,
            // A peer-based frame with an integer offset is `GROUPS`: the offset counts peer groups,
            // not rows. A `ROWS` offset is a physical row count.
            FrameBound::Preceding(n) if frame.peer_based => {
                group_step(ordered, peer_lo, peer_hi, *n, false, at_start)
            },
            FrameBound::Following(n) if frame.peer_based => {
                group_step(ordered, peer_lo, peer_hi, *n, true, at_start)
            },
            FrameBound::Preceding(n) => ki - off(*n),
            FrameBound::Following(n) => ki + off(*n),
            // A `RANGE` bound: the boundary is the current value minus/plus the offset; the frame's
            // start is the first row whose value reaches it, the end the last row still within it
            // (rows with a NULL value are outside any value range).
            FrameBound::RangePreceding(o) | FrameBound::RangeFollowing(o) => {
                let preceding = matches!(bound, FrameBound::RangePreceding(_));
                let ascending = !frame.range_descending;
                // `cur_key` is `Some` here (a NULL current value returned early above).
                let Some(boundary) =
                    cur_key.and_then(|cur| range_boundary(cur, o, preceding, ascending))
                else {
                    // Offset arithmetic overflowed: the bound is the partition edge on that side.
                    return if at_start { 0 } else { len - 1 };
                };
                range_scan(ordered, boundary, at_start, ascending)
            },
        }
    };
    let lo = pos(&frame.start, true).max(0);
    let hi = pos(&frame.end, false).min(len - 1);
    if lo > hi {
        None
    } else {
        Some((
            usize::try_from(lo).unwrap_or(0),
            usize::try_from(hi).unwrap_or(0),
        ))
    }
}

/// Assign an aggregate window's values over one ordered partition. With an explicit frame
/// each row folds over its frame (physical for `ROWS`, peer-aware for `RANGE`/`GROUPS`); otherwise
/// the default frame applies — the whole partition when unordered, or running through each peer
/// group when ordered.
pub(super) fn assign_window_aggregate(
    ordered: &[(Vec<ast::Value>, usize)],
    rows: &[Row],
    call: &AggregateCall,
    window: &WindowExpr,
    result: &mut [ast::Value],
) -> Result<(), Error> {
    // Fold one window frame (borrowed rows, no clone) into this call's single value.
    fn fold_one<'a>(
        call: &AggregateCall,
        frame: impl IntoIterator<Item = &'a Row>,
    ) -> Result<ast::Value, Error> {
        Ok(fold_aggregates(std::slice::from_ref(call), frame)?
            .into_iter()
            .next()
            .unwrap_or(ast::Value::Null))
    }

    // Explicit frame: fold over each row's frame. A `ROWS` frame (`!peer_based`) has monotonic
    // edges, so an O(n) sliding accumulator handles the supported aggregates (COUNT/SUM/AVG/MIN/MAX)
    // — closing the swinging-frame O(n·w) re-fold the Leis'15 note warns about. Anything else
    // (`RANGE`/`GROUPS`, or ARRAY_AGG/STDDEV/float-SUM/…) falls back to the per-row re-fold below.
    if let Some(frame) = &window.frame {
        let len = ordered.len();
        if !frame.peer_based {
            let handled = super::agg::sliding_window_aggregate(
                call,
                len,
                |k| frame_bounds(frame, k, ordered),
                |pos| match (
                    call.arg.as_ref(),
                    ordered.get(pos).and_then(|(_, i)| rows.get(*i)),
                ) {
                    (Some(arg), Some(row)) => eval::eval(arg, row),
                    _ => Ok(ast::Value::Null), // count(*) (no arg) never reads this
                },
                |k, value| {
                    set_window_cell(ordered, result, k, value);
                    Ok(())
                },
            )?;
            if handled {
                return Ok(());
            }
        }
        for k in 0..len {
            // Fallback — collect frame-row *references* (pointers, not row clones) and re-fold.
            let frame_rows: Vec<&Row> = match frame_bounds(frame, k, ordered) {
                Some((lo, hi)) => ordered
                    .get(lo..=hi)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|(_, i)| rows.get(*i))
                    .collect(),
                None => Vec::new(),
            };
            let value = fold_one(call, frame_rows.iter().copied())?;
            set_window_cell(ordered, result, k, value);
        }
        return Ok(());
    }

    if window.order.is_empty() {
        // Whole-partition aggregate: one value shared by every row.
        let frame: Vec<&Row> = ordered.iter().filter_map(|(_, i)| rows.get(*i)).collect();
        let value = fold_one(call, frame.iter().copied())?;
        for (_, orig) in ordered {
            if let Some(cell) = result.get_mut(*orig) {
                *cell = value.clone();
            }
        }
        return Ok(());
    }

    // Running aggregate (default RANGE … CURRENT ROW): the frame only GROWS as we walk the ordered
    // partition, so keep ONE persistent accumulator and fold each new peer row into it exactly once
    // (O(n) total), emitting each peer group's running value by finalizing a clone. The previous
    // code re-folded the whole growing frame at every peer-group boundary — folds of size 1, 2, …, n
    // = O(n²) — which hung on running totals / cumulative sums over non-trivial tables.
    // Finalizing a clone matches `fold_aggregates`, which is
    // itself accumulate-then-finalize, so the per-group value is identical to the old whole-frame
    // fold. (`array_agg`/`string_agg` windows stay O(n²) in their inherent output size, as before.)
    let mut acc = super::agg::Acc::default();
    let mut peer_origs: Vec<usize> = Vec::new();
    let mut prev: Option<Vec<ast::Value>> = None;
    for (key, orig) in ordered {
        if prev.as_ref().is_some_and(|p| !group_keys_equal(p, key)) && !peer_origs.is_empty() {
            let value = super::agg::finalize_aggregate(acc.clone(), call)?;
            for o in std::mem::take(&mut peer_origs) {
                if let Some(cell) = result.get_mut(o) {
                    *cell = value.clone();
                }
            }
        }
        if let Some(row) = rows.get(*orig) {
            super::agg::accumulate_row(
                std::slice::from_mut(&mut acc),
                std::slice::from_ref(call),
                row,
            )?;
        }
        peer_origs.push(*orig);
        prev = Some(key.clone());
    }
    if !peer_origs.is_empty() {
        let value = super::agg::finalize_aggregate(acc, call)?;
        for o in peer_origs {
            if let Some(cell) = result.get_mut(o) {
                *cell = value.clone();
            }
        }
    }
    Ok(())
}

/// Write `value` into `result` for the row at ordered position `pos`.
fn set_window_cell(
    ordered: &[(Vec<ast::Value>, usize)],
    result: &mut [ast::Value],
    pos: usize,
    value: ast::Value,
) {
    if let Some((_, orig)) = ordered.get(pos)
        && let Some(cell) = result.get_mut(*orig)
    {
        *cell = value;
    }
}

/// Evaluate a constant integer window argument (a `LAG`/`LEAD` offset, an `NTH_VALUE`/`NTILE`
/// count) at ordered position `k`; `None` for NULL or a non-integer.
fn eval_window_int(
    expr: &TypedExpr,
    ordered: &[(Vec<ast::Value>, usize)],
    rows: &[Row],
    k: usize,
) -> Result<Option<i64>, Error> {
    let Some(row) = ordered.get(k).and_then(|(_, o)| rows.get(*o)) else {
        return Ok(None);
    };
    match eval::eval(expr, row)? {
        ast::Value::Int(n) => Ok(Some(n)),
        _ => Ok(None),
    }
}

/// Assign navigation-window values (`LAG`/`LEAD`/`FIRST_VALUE`/`LAST_VALUE`/`NTH_VALUE`) over one
/// ordered partition. `FIRST_VALUE`/`LAST_VALUE`/`NTH_VALUE` read within the frame: the explicit
/// frame if given, else the **default frame** `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
/// ROW` — through the current row's peer group when ordered, the whole partition when not
/// (reading the whole partition under an ORDER BY silently turned
/// `LAST_VALUE` into "last of partition"). `LAG`/`LEAD` step by an offset (default 1),
/// independent of the frame, falling back to the optional default argument when out of range.
fn assign_navigation(
    ordered: &[(Vec<ast::Value>, usize)],
    rows: &[Row],
    window: &WindowExpr,
    result: &mut [ast::Value],
) -> Result<(), Error> {
    use crate::ast::WindowFunc as W;

    let len = ordered.len();
    let Some(value_expr) = window.args.first() else {
        return Ok(());
    };
    for k in 0..len {
        // The frame `FIRST_VALUE`/`LAST_VALUE`/`NTH_VALUE` read from: the explicit frame for
        // this row, else the default frame — partition start through the current row's peer
        // group (with no ORDER BY every row is a peer, so that is the whole partition).
        let frame_range: Option<(usize, usize)> = match &window.frame {
            Some(frame) => frame_bounds(frame, k, ordered),
            None if window.order.is_empty() => len.checked_sub(1).map(|hi| (0, hi)),
            None => Some((0, peer_group(ordered, k).1)),
        };
        // The position within `ordered` to read `value_expr` from, if any.
        let target: Option<usize> = match &window.func {
            W::FirstValue => frame_range.map(|(lo, _)| lo),
            W::LastValue => frame_range.map(|(_, hi)| hi),
            W::NthValue => match window.args.get(1) {
                Some(n_expr) => match (eval_window_int(n_expr, ordered, rows, k)?, frame_range) {
                    (Some(n), Some((lo, hi))) if n >= 1 => usize::try_from(n - 1)
                        .ok()
                        .and_then(|i| lo.checked_add(i))
                        .filter(|&t| t <= hi),
                    _ => None,
                },
                None => None,
            },
            W::Lag | W::Lead => {
                let offset = match window.args.get(1) {
                    Some(e) => eval_window_int(e, ordered, rows, k)?.unwrap_or(1),
                    None => 1,
                };
                let delta = if matches!(window.func, W::Lag) {
                    offset.checked_neg().unwrap_or(0)
                } else {
                    offset
                };
                i64::try_from(k)
                    .ok()
                    .and_then(|cur| cur.checked_add(delta))
                    .and_then(|p| usize::try_from(p).ok())
                    .filter(|&t| t < len)
            },
            _ => None,
        };

        let value = match target
            .and_then(|t| ordered.get(t))
            .and_then(|(_, o)| rows.get(*o))
        {
            Some(row) => eval::eval(value_expr, row)?,
            // Out of frame: LAG/LEAD use the default argument (evaluated at the current row), else NULL.
            None => match window
                .args
                .get(2)
                .zip(ordered.get(k).and_then(|(_, o)| rows.get(*o)))
            {
                Some((default, row)) => eval::eval(default, row)?,
                None => ast::Value::Null,
            },
        };
        set_window_cell(ordered, result, k, value);
    }
    Ok(())
}

/// Assign distribution-window values (`NTILE`/`CUME_DIST`/`PERCENT_RANK`) over one ordered
/// partition. `CUME_DIST`/`PERCENT_RANK` are peer-aware (rows with equal `ORDER BY` keys share a
/// value). `NTILE(n)` splits the partition into `n` near-equal buckets (the first `len % n` get one
/// extra row).
#[allow(
    clippy::cast_precision_loss,
    reason = "row counts widen to f64 for the [0,1] distribution ratios — inherent and intended"
)]
fn assign_distribution(
    ordered: &[(Vec<ast::Value>, usize)],
    rows: &[Row],
    window: &WindowExpr,
    result: &mut [ast::Value],
) -> Result<(), Error> {
    use crate::ast::WindowFunc as W;

    let len = ordered.len();
    if len == 0 {
        return Ok(());
    }
    match &window.func {
        W::Ntile => {
            let raw = match window.args.first() {
                Some(e) => eval_window_int(e, ordered, rows, 0)?.unwrap_or(0),
                None => 0,
            };
            if raw < 1 {
                return Err(Error::Unsupported(
                    "NTILE requires a positive bucket count".to_owned(),
                ));
            }
            let n = usize::try_from(raw).unwrap_or(usize::MAX);
            let base = len / n;
            let rem = len % n;
            let big = rem * (base + 1);
            for k in 0..len {
                let bucket = if base == 0 || k < big {
                    k / (base + 1)
                } else {
                    rem + (k - big) / base
                };
                let value = ast::Value::Int(i64::try_from(bucket + 1).unwrap_or(i64::MAX));
                set_window_cell(ordered, result, k, value);
            }
        },
        W::CumeDist | W::PercentRank => {
            let mut start = 0;
            while start < len {
                let mut end = start;
                while end + 1 < len
                    && ordered
                        .get(end + 1)
                        .zip(ordered.get(start))
                        .is_some_and(|(a, b)| group_keys_equal(&a.0, &b.0))
                {
                    end += 1;
                }
                let value = if matches!(window.func, W::CumeDist) {
                    ast::Value::Float((end + 1) as f64 / len as f64)
                } else if len > 1 {
                    ast::Value::Float(start as f64 / (len - 1) as f64)
                } else {
                    ast::Value::Float(0.0)
                };
                for pos in start..=end {
                    set_window_cell(ordered, result, pos, value.clone());
                }
                start = end + 1;
            }
        },
        _ => {},
    }
    Ok(())
}

/// Remove duplicate rows for `SELECT DISTINCT`, keeping the first occurrence of each distinct row
/// and preserving input order. Two rows are duplicates when every column is "not distinct" (`NULL`
/// is not distinct from `NULL`), reusing [`group_keys_equal`].
pub(super) fn dedupe_rows(rows: Vec<Row>) -> Vec<Row> {
    // Fast path: when every value is hash-safe — its canonical key bytes compare exactly
    // like `group_keys_equal` (Int/Bool/Text/temporal/Uuid + NULL) — dedup via a `HashSet` of those
    // bytes, O(n), first-seen order kept. `Float`/`NUMERIC` are excluded (equal values can encode to
    // different bytes), so they take the linear fallback. Output is identical in both paths.
    if rows.iter().all(|row| row.iter().all(is_hash_safe_value)) {
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            // Hash-safe rows always encode; the defensive `Err` branch keeps correctness regardless.
            match index_key::encode_index_key(&row) {
                Ok(key) => {
                    if seen.insert(key) {
                        out.push(row);
                    }
                },
                Err(_) => {
                    if !out.iter().any(|e| group_keys_equal(e, &row)) {
                        out.push(row);
                    }
                },
            }
        }
        return out;
    }
    let mut out: Vec<Row> = Vec::new();
    for row in rows {
        if !out.iter().any(|existing| group_keys_equal(existing, &row)) {
            out.push(row);
        }
    }
    out
}

/// Filter `left` by membership in `right` under SQL "not distinct" equality: `keep_present == true`
/// keeps rows present in `right` (`INTERSECT`), `false` keeps rows absent from it (`EXCEPT`). When
/// every value is hash-safe this builds a `HashSet` of `right`'s key bytes and probes per left row —
/// O(left + right); otherwise it falls back to a linear scan. `left` order is preserved.
pub(super) fn filter_by_membership(left: Vec<Row>, right: &[Row], keep_present: bool) -> Vec<Row> {
    if left
        .iter()
        .chain(right)
        .all(|row| row.iter().all(is_hash_safe_value))
    {
        let set: std::collections::HashSet<Vec<u8>> = right
            .iter()
            .filter_map(|r| index_key::encode_index_key(r).ok())
            .collect();
        return left
            .into_iter()
            .filter(|row| {
                let present = index_key::encode_index_key(row).is_ok_and(|key| set.contains(&key));
                present == keep_present
            })
            .collect();
    }
    left.into_iter()
        .filter(|row| right.iter().any(|x| group_keys_equal(x, row)) == keep_present)
        .collect()
}

/// SQL grouping equality: two key tuples are in the same group when each
/// component is "not distinct" — `NULL` groups with `NULL`, otherwise values
/// must compare equal.
pub(super) fn group_keys_equal(a: &[ast::Value], b: &[ast::Value]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (ast::Value::Null, ast::Value::Null) => true,
            (ast::Value::Null, _) | (_, ast::Value::Null) => false,
            _ => eval::compare(x, y) == std::cmp::Ordering::Equal,
        })
}

/// Spilling `DISTINCT`: bound working memory to ~`threshold_bytes` by sorting the input on
/// **all** output columns (external merge sort, [`spill_sort::sorted_input`]) and emitting the first
/// row of each adjacent-equal run. Duplicates land next to each other after the sort, so a single
/// trailing `prev` row suffices to deduplicate — never the whole input at once.
///
/// The sort keys are synthesized `Column(i)` references for `i in 0..width`. Their `ty` is a
/// sort-only placeholder: [`eval`](crate::executor::eval::eval) resolves a `Column` ref by indexing
/// the row and [`compare_order_key`](crate::executor::eval::compare_order_key) compares the resulting
/// `Value` — neither reads the expression's declared type. Adjacent-run equality reuses
/// [`group_keys_equal`] (`NULL` is not distinct from `NULL`), matching the in-memory `dedupe_rows`
/// predicate, so the result is the same multiset (order is unspecified for `DISTINCT`).
///
/// # Errors
/// Propagates streaming, spill-file I/O, and key-evaluation errors.
fn sort_based_distinct(
    input: &PhysicalOperator,
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    use crate::planner::{OrderByKey, TypedExpr, TypedExprKind};

    let width = output_columns(input).len();
    let keys: Vec<OrderByKey> = (0..width)
        .map(|i| OrderByKey {
            expr: TypedExpr {
                kind: TypedExprKind::Column(i),
                // Sort-only placeholder: a `Column` ref is resolved by row index, never by `ty`.
                ty: nusadb_core::ColumnType::Int,
            },
            ascending: true,
            nulls: ast::NullOrdering::Default,
        })
        .collect();

    let mut sorted = super::spill_sort::sorted_input(input, &keys, config, engine, txn)?;
    let mut out: Vec<Row> = Vec::new();
    while let Some(row) = sorted.try_next()? {
        // Adjacent-run dedup correctness depends on the sort covering *every* output column: if
        // `width` under-counts the real row width, two true duplicates could be interleaved by a
        // distinct row that ties on the sorted prefix and so escape de-duplication. The planner
        // always seats `Distinct` directly above `Project`/`ProjectSet`, so `width` equals the row
        // width — assert it so a future plan shape that breaks the invariant fails loudly in tests
        // rather than silently returning duplicates.
        debug_assert_eq!(
            row.len(),
            width,
            "spilling DISTINCT sorts on all {width} columns, but a row has {} — the sort key would \
             miss a column and adjacent-run dedup could keep duplicates",
            row.len()
        );
        // The previously emitted distinct row is `out.last()`; comparing against it (instead of a
        // cloned `prev`) drops one row clone per distinct output row — meaningful for a large
        // DISTINCT result under a tight memory budget.
        if out.last().is_some_and(|prev| group_keys_equal(prev, &row)) {
            continue; // duplicate of the previously emitted row
        }
        out.push(row);
    }
    Ok(out)
}

/// Walk the pipeline to find the `Project` that names the output columns.
/// `Project` is always present (the planner unconditionally wraps the source).
pub(super) fn output_columns(op: &PhysicalOperator) -> Vec<String> {
    match op {
        PhysicalOperator::Project { columns, .. }
        | PhysicalOperator::ProjectSet { columns, .. } => {
            columns.iter().map(|c| c.name.clone()).collect()
        },
        PhysicalOperator::Limit { input, .. }
        | PhysicalOperator::Distinct { input }
        // `LockRows` (FOR UPDATE/SHARE) yields its input's rows unchanged.
        | PhysicalOperator::LockRows { input, .. } => output_columns(input),
        // The output columns of a `WITH RECURSIVE` query are its body's.
        PhysicalOperator::WithRecursive { body, .. }
        | PhysicalOperator::WithModifying { body, .. } => output_columns(body),
        _ => Vec::new(),
    }
}

/// The output column **types** of an operator, parallel to [`output_columns`]. Walks the
/// same operators in the same order, so the two line up element-for-element. Each projected column's
/// type is its already-resolved [`TypedExpr::ty`](crate::planner::TypedExpr).
pub(super) fn output_column_types(op: &PhysicalOperator) -> Vec<ColumnType> {
    match op {
        PhysicalOperator::Project { columns, .. }
        | PhysicalOperator::ProjectSet { columns, .. } => {
            columns.iter().map(|c| c.expr.ty).collect()
        },
        PhysicalOperator::Limit { input, .. }
        | PhysicalOperator::Distinct { input }
        | PhysicalOperator::LockRows { input, .. } => output_column_types(input),
        PhysicalOperator::WithRecursive { body, .. }
        | PhysicalOperator::WithModifying { body, .. } => output_column_types(body),
        _ => Vec::new(),
    }
}

/// Apply a `FETCH FIRST n ROWS WITH TIES` cap to already-sorted `rows`: skip
/// `cap.offset` rows, keep `cap.count` rows, then extend the result with every following row that
/// ties the last kept row on the ORDER BY `keys`. Two rows tie when every key compares equal (the
/// same peer relation the sort uses, so the kept boundary never splits a tie group at the end).
fn apply_ties_limit(
    rows: Vec<Row>,
    keys: &[OrderByKey],
    cap: &crate::planner::TiesLimit,
) -> Result<Vec<Row>, Error> {
    let start = usize::try_from(cap.offset).unwrap_or(usize::MAX);
    let count = usize::try_from(cap.count).unwrap_or(usize::MAX);
    if count == 0 || start >= rows.len() {
        return Ok(Vec::new());
    }
    // The last kept row (within `count` of `start`) defines the tie set extended past it.
    let kept = count.min(rows.len() - start);
    // `kept >= 1` here (count >= 1 and start < len), so this index is in bounds.
    let Some(last_row) = rows.get(start + kept - 1) else {
        return Ok(rows);
    };
    let last_keys = eval_sort_keys(keys, last_row)?;
    let mut extra = 0;
    for cand in rows.iter().skip(start + kept) {
        let cand_keys = eval_sort_keys(keys, cand)?;
        let ties = keys
            .iter()
            .zip(&last_keys)
            .zip(&cand_keys)
            .all(|((k, a), b)| {
                eval::compare_order_key(a, b, k.ascending, k.nulls) == std::cmp::Ordering::Equal
            });
        if ties {
            extra += 1;
        } else {
            break;
        }
    }
    Ok(rows.into_iter().skip(start).take(kept + extra).collect())
}

/// Evaluate every ORDER BY key against `row` into the comparable value tuple used by the tie test.
fn eval_sort_keys(keys: &[OrderByKey], row: &Row) -> Result<Vec<ast::Value>, Error> {
    keys.iter().map(|k| eval::eval(&k.expr, row)).collect()
}

pub(super) fn sort_rows(rows: &mut Vec<Row>, keys: &[OrderByKey]) -> Result<(), Error> {
    // Pre-evaluate sort keys per row so the sort comparator is infallible.
    let mut decorated: Vec<(Vec<ast::Value>, Row)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.drain(..).enumerate() {
        // Amortized cooperative-cancel poll so a statement timeout interrupts a large sort's key
        // evaluation rather than running it to completion.
        if i.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        let evaluated: Vec<ast::Value> = keys
            .iter()
            .map(|k| eval::eval(&k.expr, &row))
            .collect::<Result<Vec<_>, _>>()?;
        decorated.push((evaluated, row));
    }
    finish_sort(decorated, keys, rows);
    Ok(())
}

/// Like [`sort_rows`] but an ORDER BY key may contain a subquery: each key's
/// uncorrelated subqueries are pre-resolved once, and a correlated one is evaluated per row against
/// the bound outer row. `engine`/`txn` supply the subquery execution context.
fn sort_rows_with_subqueries(
    rows: &mut Vec<Row>,
    keys: &[OrderByKey],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    // Pre-resolve uncorrelated subqueries in every key once; a correlated one stays for per-row eval.
    let resolved: Vec<OrderByKey> = {
        let _defer = defer_correlated(true);
        keys.iter()
            .map(|k| {
                let mut expr = k.expr.clone();
                resolve_subqueries(&mut expr, engine, txn)?;
                Ok::<_, Error>(OrderByKey {
                    expr,
                    ascending: k.ascending,
                    nulls: k.nulls,
                })
            })
            .collect::<Result<_, _>>()?
    };
    let any_correlated = resolved.iter().any(|k| contains_subquery(&k.expr));
    let mut decorated: Vec<(Vec<ast::Value>, Row)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.drain(..).enumerate() {
        if i.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        let evaluated: Vec<ast::Value> = resolved
            .iter()
            .map(|k| {
                if any_correlated && contains_subquery(&k.expr) {
                    eval_correlated(&k.expr, &row, engine, txn)
                } else {
                    eval::eval(&k.expr, &row)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        decorated.push((evaluated, row));
    }
    finish_sort(decorated, &resolved, rows);
    Ok(())
}

/// Sort the decorated `(sort-key values, row)` pairs by `keys`' direction/null-ordering and push the
/// rows back into `rows` in order. The comparator is infallible (keys are pre-evaluated).
fn finish_sort(
    mut decorated: Vec<(Vec<ast::Value>, Row)>,
    keys: &[OrderByKey],
    rows: &mut Vec<Row>,
) {
    decorated.sort_by(|a, b| {
        for (idx, (av, bv)) in a.0.iter().zip(&b.0).enumerate() {
            let (ascending, nulls) = keys
                .get(idx)
                .map_or((true, ast::NullOrdering::Default), |k| {
                    (k.ascending, k.nulls)
                });
            let ord = eval::compare_order_key(av, bv, ascending, nulls);
            if !ord.is_eq() {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    for (_, row) in decorated {
        rows.push(row);
    }
}

/// Compare two decorated rows by `keys`' direction/null-ordering, breaking exact ties by arrival
/// sequence (ascending) so the order is total and matches the stable [`finish_sort`] exactly — the
/// same total order the top-N selection maintains.
fn cmp_top_n(
    a_keys: &[ast::Value],
    a_seq: u64,
    b_keys: &[ast::Value],
    b_seq: u64,
    keys: &[OrderByKey],
) -> std::cmp::Ordering {
    for (idx, (av, bv)) in a_keys.iter().zip(b_keys).enumerate() {
        let (ascending, nulls) = keys
            .get(idx)
            .map_or((true, ast::NullOrdering::Default), |k| {
                (k.ascending, k.nulls)
            });
        let ord = eval::compare_order_key(av, bv, ascending, nulls);
        if !ord.is_eq() {
            return ord;
        }
    }
    a_seq.cmp(&b_seq)
}

/// One row retained by the bounded top-N pass: its pre-evaluated sort keys, its arrival sequence
/// (the stable tie-break), and the row itself. `Ord` is the sort's total order (keys then seq), so
/// a max-heap of these keeps the current largest — the eviction candidate — at its peek.
struct TopNEntry {
    keys: Vec<ast::Value>,
    seq: u64,
    row: Row,
    order: std::rc::Rc<Vec<OrderByKey>>,
}

impl PartialEq for TopNEntry {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}
impl Eq for TopNEntry {}
impl PartialOrd for TopNEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopNEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_top_n(&self.keys, self.seq, &other.keys, other.seq, &self.order)
    }
}

/// Select the first `m` rows of `input` in `keys` order without a full sort
/// A streaming pass keeps a bounded max-heap of the `m` smallest rows
/// by the sort's total order (keys, then arrival sequence for stable ties), so the result is
/// **identical** to the full sort's first `m` rows — O(N log m) time, `m` rows retained. Every
/// input row is pulled (a `SERIALIZABLE` scan therefore reads and records the same rows a full
/// `SeqScan` + `Sort` would), only the sort itself is avoided. The caller's `Limit`/`OFFSET`
/// operator still applies unchanged above.
fn top_n_rows(
    input: &PhysicalOperator,
    keys: &[OrderByKey],
    m: u64,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let m = usize::try_from(m).unwrap_or(usize::MAX);
    if m == 0 {
        // The enclosing LIMIT needs no rows, but the scan must still run so its reads are
        // recorded (isolation parity with the full-sort path); drain and drop.
        let mut src = super::stream::stream_op(input, engine, txn)?;
        while src.try_next()?.is_some() {}
        return Ok(Vec::new());
    }
    let order = std::rc::Rc::new(keys.to_vec());
    // Grow the heap to the ACTUAL retained count (≤ min(m, N)); do NOT pre-allocate the `m` cap,
    // which would waste memory for a large `LIMIT` over a small table.
    let mut heap: std::collections::BinaryHeap<TopNEntry> = std::collections::BinaryHeap::new();
    // The work-memory budget bounds the retained rows by BYTES, not just the row count — a wide
    // (TEXT/JSON/VECTOR) `LIMIT n` could otherwise hold `n` large rows unchecked. Tracked
    // incrementally so a violation errors before the memory is over-committed.
    let budget = effective_work_mem();
    let mut retained_bytes: usize = 0;
    let mut src = super::stream::stream_op(input, engine, txn)?;
    let mut seq: u64 = 0;
    while let Some(row) = src.try_next()? {
        let key_values = eval_sort_keys(keys, &row)?;
        let row_size = row_bytes(&row);
        if heap.len() < m {
            retained_bytes += row_size;
            enforce_top_n_budget(budget, retained_bytes)?;
            heap.push(TopNEntry {
                keys: key_values,
                seq,
                row,
                order: std::rc::Rc::clone(&order),
            });
        } else if let Some(max) = heap.peek()
            && cmp_top_n(&key_values, seq, &max.keys, max.seq, keys) == std::cmp::Ordering::Less
        {
            // This row sorts before the current largest kept row — evict that and keep this one.
            if let Some(evicted) = heap.pop() {
                retained_bytes = retained_bytes.saturating_sub(row_bytes(&evicted.row));
            }
            retained_bytes += row_size;
            enforce_top_n_budget(budget, retained_bytes)?;
            heap.push(TopNEntry {
                keys: key_values,
                seq,
                row,
                order: std::rc::Rc::clone(&order),
            });
        }
        seq = seq.saturating_add(1);
    }
    // The heap holds the m smallest rows in no particular emission order; sort just those into the
    // final ascending order (O(m log m)) and hand back the rows.
    let mut kept = heap.into_vec();
    kept.sort_unstable_by(|a, b| cmp_top_n(&a.keys, a.seq, &b.keys, b.seq, keys));
    Ok(kept.into_iter().map(|e| e.row).collect())
}
