//! Constant folding: evaluate constant sub-expressions at plan time and short-circuit
//! boolean operators that have a literal operand, so the executor does not redo the work per row.
//!
//! Behavior-preserving by construction. A node folds to a literal only when it is built entirely
//! from literals via pure, deterministic operators — never a column/subquery/aggregate reference and
//! never a function call (which may be volatile, e.g. `random()`/`now()`, or key-dependent like
//! `encrypt()`) — AND its evaluation succeeds. An expression that would error (e.g. `1/0`) is left
//! intact so the error still surfaces at runtime in its original context (a dead `CASE` branch's
//! error never fires, because the whole `CASE` folds via short-circuit evaluation instead).

use super::plan_types::{
    AggregateCall, SelectPlan, TypedCaseBranch, TypedExpr, TypedExprKind, WindowExpr,
};
use crate::ast;
use crate::executor::eval;
use nusadb_core::ColumnType;

/// Fold every constant sub-expression of a `SELECT` plan's clauses to a literal. Sub-plans
/// (CTEs, set-op branches, derived tables, subqueries) are folded when the planner recurses into
/// them, so this only walks `select`'s own clause expressions.
pub(super) fn fold_select(mut select: SelectPlan) -> SelectPlan {
    select.filter = select.filter.map(fold_expr).filter(|f| !is_true_literal(f));
    select.having = select.having.map(fold_expr).filter(|h| !is_true_literal(h));
    for proj in &mut select.projection {
        fold_in_place(&mut proj.expr);
    }
    for key in &mut select.order_by {
        fold_in_place(&mut key.expr);
    }
    for key in &mut select.group_keys {
        fold_in_place(key);
    }
    for key in &mut select.distinct_on {
        fold_in_place(key);
    }
    for join in &mut select.joins {
        fold_in_place(&mut join.on);
    }
    for agg in &mut select.aggregates {
        fold_aggregate(agg);
    }
    for window in &mut select.windows {
        fold_window(window);
    }
    select
}

fn fold_aggregate(agg: &mut AggregateCall) {
    if let Some(arg) = agg.arg.take() {
        agg.arg = Some(fold_expr(arg));
    }
    if let Some(filter) = agg.filter.take() {
        agg.filter = Some(fold_expr(filter));
    }
}

fn fold_window(window: &mut WindowExpr) {
    for arg in &mut window.args {
        fold_in_place(arg);
    }
    for key in &mut window.partition {
        fold_in_place(key);
    }
    for key in &mut window.order {
        fold_in_place(&mut key.expr);
    }
}

/// Fold `expr` in place (a small helper for the many `&mut TypedExpr` clause slots above).
fn fold_in_place(expr: &mut TypedExpr) {
    let folded = fold_expr(std::mem::replace(
        expr,
        TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Null),
            ty: ColumnType::Bool,
        },
    ));
    *expr = folded;
}

/// Fold every constant sub-expression of `expr` to a literal, bottom-up.
fn fold_expr(expr: TypedExpr) -> TypedExpr {
    let ty = expr.ty;
    let node = TypedExpr {
        kind: fold_children(expr.kind),
        ty,
    };
    // Whole-node constant fold: only a pure-constant node, and only if evaluation succeeds (an
    // erroring expression stays intact so its error fires at runtime in the right context).
    if is_foldable_constant(&node)
        && let Ok(value) = eval::eval(&node, &Vec::new())
    {
        return TypedExpr {
            kind: TypedExprKind::Literal(value),
            ty,
        };
    }
    // Boolean short-circuit for `AND`/`OR` with one literal-bool operand — the common
    // generated-SQL `WHERE 1=1 AND <pred>` shape after `1=1` folds to TRUE.
    simplify_bool(node)
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per TypedExprKind variant; length tracks the expression grammar"
)]
fn fold_children(kind: TypedExprKind) -> TypedExprKind {
    use TypedExprKind as K;
    match kind {
        // Leaves and references carry no foldable child.
        K::Literal(_)
        | K::Column(_)
        | K::OuterColumn { .. }
        | K::AggregateRef(_)
        | K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. } => kind,
        K::Binary { left, op, right } => K::Binary {
            left: fold_box(left),
            op,
            right: fold_box(right),
        },
        K::QuantifiedArray {
            expr,
            op,
            all,
            array,
        } => K::QuantifiedArray {
            expr: fold_box(expr),
            op,
            all,
            array: fold_box(array),
        },
        K::Unary { op, expr } => K::Unary {
            op,
            expr: fold_box(expr),
        },
        K::IsNull { expr, negated } => K::IsNull {
            expr: fold_box(expr),
            negated,
        },
        K::IsDistinctFrom {
            left,
            right,
            negated,
        } => K::IsDistinctFrom {
            left: fold_box(left),
            right: fold_box(right),
            negated,
        },
        K::IsBool {
            expr,
            truth,
            negated,
        } => K::IsBool {
            expr: fold_box(expr),
            truth,
            negated,
        },
        K::InList {
            expr,
            list,
            negated,
        } => K::InList {
            expr: fold_box(expr),
            list: fold_vec(list),
            negated,
        },
        K::Between {
            expr,
            low,
            high,
            negated,
        } => K::Between {
            expr: fold_box(expr),
            low: fold_box(low),
            high: fold_box(high),
            negated,
        },
        K::Like {
            expr,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => K::Like {
            expr: fold_box(expr),
            pattern: fold_box(pattern),
            negated,
            escape,
            case_insensitive,
        },
        K::RegexMatch {
            expr,
            pattern,
            case_sensitive,
            negated,
        } => K::RegexMatch {
            expr: fold_box(expr),
            pattern: fold_box(pattern),
            case_sensitive,
            negated,
        },
        K::SimilarTo {
            expr,
            pattern,
            negated,
        } => K::SimilarTo {
            expr: fold_box(expr),
            pattern: fold_box(pattern),
            negated,
        },
        K::Case {
            operand,
            branches,
            default,
        } => K::Case {
            operand: operand.map(fold_box),
            branches: branches
                .into_iter()
                .map(|b| TypedCaseBranch {
                    when: fold_expr(b.when),
                    then: fold_expr(b.then),
                })
                .collect(),
            default: default.map(fold_box),
        },
        K::Coalesce(items) => K::Coalesce(fold_vec(items)),
        K::ArrayLiteral(items) => K::ArrayLiteral(fold_vec(items)),
        K::Subscript { base, index } => K::Subscript {
            base: fold_box(base),
            index: fold_box(index),
        },
        K::ArraySlice { base, lower, upper } => K::ArraySlice {
            base: fold_box(base),
            lower: lower.map(fold_box),
            upper: upper.map(fold_box),
        },
        K::Cast(inner, try_cast) => K::Cast(fold_box(inner), try_cast),
        // Function/crypto calls: fold their arguments, but never fold the call itself — a built-in
        // may be volatile (`random()`, `now()`) or key-dependent (`encrypt()`).
        K::Crypto { op, value, key } => K::Crypto {
            op,
            value: fold_box(value),
            key: fold_box(key),
        },
        K::ScalarFunction { func, args } => K::ScalarFunction {
            func,
            args: fold_vec(args),
        },
        // A UDF's arguments fold, but the call itself is never constant-evaluated (a UDF may be
        // non-deterministic), so the node is kept.
        K::ScalarUdf {
            name,
            args,
            arg_types,
        } => K::ScalarUdf {
            name,
            args: fold_vec(args),
            arg_types,
        },
        K::SetReturning { func, args } => K::SetReturning {
            func,
            args: fold_vec(args),
        },
    }
}

#[allow(
    clippy::boxed_local,
    reason = "mirrors the boxed AST child fields; consumes the caller's Box directly"
)]
fn fold_box(expr: Box<TypedExpr>) -> Box<TypedExpr> {
    Box::new(fold_expr(*expr))
}

fn fold_vec(exprs: Vec<TypedExpr>) -> Vec<TypedExpr> {
    exprs.into_iter().map(fold_expr).collect()
}

/// Whether `expr` is built entirely from literals via pure, deterministic operators — safe to
/// evaluate at plan time. Excludes every column/subquery/aggregate reference and every function call
/// (possibly volatile or key-dependent), so evaluation can never read a row, run a subquery, or
/// precompute a non-deterministic value.
fn is_foldable_constant(expr: &TypedExpr) -> bool {
    use TypedExprKind as K;
    match &expr.kind {
        K::Literal(_) => true,
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            is_foldable_constant(left) && is_foldable_constant(right)
        },
        K::Unary { expr, .. }
        | K::IsNull { expr, .. }
        | K::IsBool { expr, .. }
        | K::Cast(expr, _) => is_foldable_constant(expr),
        K::InList { expr, list, .. } => {
            is_foldable_constant(expr) && list.iter().all(is_foldable_constant)
        },
        K::Between {
            expr, low, high, ..
        } => is_foldable_constant(expr) && is_foldable_constant(low) && is_foldable_constant(high),
        K::Like { expr, pattern, .. }
        | K::RegexMatch { expr, pattern, .. }
        | K::SimilarTo { expr, pattern, .. } => {
            is_foldable_constant(expr) && is_foldable_constant(pattern)
        },
        K::Subscript { base, index } => is_foldable_constant(base) && is_foldable_constant(index),
        K::ArraySlice { base, lower, upper } => {
            is_foldable_constant(base)
                && lower.as_deref().is_none_or(is_foldable_constant)
                && upper.as_deref().is_none_or(is_foldable_constant)
        },
        K::Coalesce(items) | K::ArrayLiteral(items) => items.iter().all(is_foldable_constant),
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(is_foldable_constant)
                && branches
                    .iter()
                    .all(|b| is_foldable_constant(&b.when) && is_foldable_constant(&b.then))
                && default.as_deref().is_none_or(is_foldable_constant)
        },
        // Column / OuterColumn / AggregateRef / Crypto / ScalarFunction / SetReturning / subqueries.
        _ => false,
    }
}

/// Short-circuit `AND`/`OR` when one operand is a literal boolean. Valid in SQL three-valued logic:
/// `false AND x = false`, `true AND x = x`, `true OR x = true`, `false OR x = x` (each holds even
/// when `x` is `NULL`).
fn simplify_bool(node: TypedExpr) -> TypedExpr {
    let TypedExpr { kind, ty } = node;
    let TypedExprKind::Binary { left, op, right } = kind else {
        return TypedExpr { kind, ty };
    };
    let (lhs, rhs) = (as_bool(&left), as_bool(&right));
    match op {
        ast::BinaryOp::And => {
            if lhs == Some(false) || rhs == Some(false) {
                return bool_literal(false, ty);
            }
            if lhs == Some(true) {
                return *right;
            }
            if rhs == Some(true) {
                return *left;
            }
        },
        ast::BinaryOp::Or => {
            if lhs == Some(true) || rhs == Some(true) {
                return bool_literal(true, ty);
            }
            if lhs == Some(false) {
                return *right;
            }
            if rhs == Some(false) {
                return *left;
            }
        },
        _ => {},
    }
    TypedExpr {
        kind: TypedExprKind::Binary { left, op, right },
        ty,
    }
}

/// The boolean value of `expr` if it is a literal `TRUE`/`FALSE`; `None` otherwise (including `NULL`).
const fn as_bool(expr: &TypedExpr) -> Option<bool> {
    match &expr.kind {
        TypedExprKind::Literal(ast::Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// Whether `expr` is the literal `TRUE` (a `WHERE`/`HAVING` that folded to this is dropped entirely).
fn is_true_literal(expr: &TypedExpr) -> bool {
    as_bool(expr) == Some(true)
}

const fn bool_literal(value: bool, ty: ColumnType) -> TypedExpr {
    TypedExpr {
        kind: TypedExprKind::Literal(ast::Value::Bool(value)),
        ty,
    }
}
