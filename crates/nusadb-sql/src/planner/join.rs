//! Join planning: equi-key extraction and hash-vs-nested-loop selection.
//!
//! Split verbatim out of `planner/mod.rs` (ADR 007). Resolves siblings via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

/// Build the physical operator for one join. Splits the `ON` predicate into
/// `AND`-conjuncts, peels off usable equi-keys (one side referencing only left
/// columns, the other only right columns, both the same hashable type), and
/// emits a [`PhysicalOperator::HashJoin`] when at least one key is found —
/// otherwise a [`PhysicalOperator::NestedLoopJoin`] over the full predicate.
pub(super) fn build_join(
    left: PhysicalOperator,
    right: PhysicalOperator,
    on: TypedExpr,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
    coalesce_pairs: Vec<(usize, usize)>,
) -> PhysicalOperator {
    let mut conjuncts = Vec::new();
    split_conjuncts(on, &mut conjuncts);

    let mut keys = Vec::new();
    let mut residual_conjuncts = Vec::new();
    for conjunct in conjuncts {
        match classify_conjunct(conjunct, left_width) {
            Ok(key) => keys.push(key),
            Err(other) => residual_conjuncts.push(other),
        }
    }

    // ON-clause predicate pushdown. For an INNER join, `ON` is equivalent to
    // `WHERE`, so a residual conjunct referencing only one input can be evaluated before the join
    // rather than on every emitted pair. A selective single-side predicate — e.g. `o1.id < 1000`
    // in a self-join `o1 ⋈ o2 ON o1.customer_id = o2.customer_id AND o1.id < o2.id AND o1.id < 1000`
    // — then shrinks that input from O(n) to O(selective) rows before the hash join, so the join's
    // output collapses from O(n × fan-out) intermediate pairs to a small set (QA measured the
    // un-pushed form growing supra-linearly: 118×→210× the reference engine at 3M→8M rows). Outer joins keep such
    // predicates in the residual: a predicate on an outer join's preserved side must not drop its
    // rows, so only the unambiguously-safe INNER case is pushed. A subquery-bearing conjunct is
    // never pushed (it may correlate to the joined row, and the right-side shift below assumes no
    // nested scope).
    let mut left = left;
    let mut right = right;
    if matches!(kind, ast::JoinKind::Inner) {
        let mut keep = Vec::new();
        let mut left_push = Vec::new();
        let mut right_push = Vec::new();
        for conjunct in residual_conjuncts {
            if crate::executor::ops::contains_subquery(&conjunct) {
                keep.push(conjunct);
                continue;
            }
            match column_side(&conjunct, left_width) {
                Side::Left => left_push.push(conjunct),
                Side::Right => right_push.push(conjunct),
                Side::NeitherOrBoth => keep.push(conjunct),
            }
        }
        if let Some(predicate) = rebuild_and(left_push) {
            left = PhysicalOperator::Filter {
                input: Box::new(left),
                predicate,
            };
        }
        if let Some(mut predicate) = rebuild_and(right_push) {
            // A right-only conjunct references columns `[left_width, left_width + right_width)` in
            // the joined-row space; the right input's own Filter reads columns `[0, right_width)`,
            // so shift every column ordinal down by `left_width`.
            remap_columns(&mut predicate, left_width);
            right = PhysicalOperator::Filter {
                input: Box::new(right),
                predicate,
            };
        }
        residual_conjuncts = keep;
    }

    if keys.is_empty() {
        // No hashable equi-key: reassemble the full predicate for a nested loop.
        // A real join always has at least one conjunct; the `true` fallback is
        // only a defensive no-op for an (impossible) empty predicate.
        let predicate = rebuild_and(residual_conjuncts).unwrap_or(TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Bool(true)),
            ty: ColumnType::Bool,
        });
        return PhysicalOperator::NestedLoopJoin {
            left: Box::new(left),
            right: Box::new(right),
            predicate,
            kind,
            left_width,
            right_width,
            coalesce_pairs,
        };
    }
    PhysicalOperator::HashJoin {
        left: Box::new(left),
        right: Box::new(right),
        keys,
        residual: rebuild_and(residual_conjuncts),
        kind,
        left_width,
        right_width,
        coalesce_pairs,
    }
}

/// Flatten a predicate into its top-level `AND`-conjuncts. `AND` is associative
/// and commutative under three-valued logic, so the order does not matter.
pub(super) fn split_conjuncts(expr: TypedExpr, out: &mut Vec<TypedExpr>) {
    match expr.kind {
        TypedExprKind::Binary {
            left,
            op: ast::BinaryOp::And,
            right,
        } => {
            split_conjuncts(*left, out);
            split_conjuncts(*right, out);
        },
        other => out.push(TypedExpr {
            kind: other,
            ty: expr.ty,
        }),
    }
}

/// Re-`AND` a list of conjuncts back into one boolean expression, or `None` if
/// the list is empty.
pub(super) fn rebuild_and(conjuncts: Vec<TypedExpr>) -> Option<TypedExpr> {
    let mut iter = conjuncts.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, next| TypedExpr {
        kind: TypedExprKind::Binary {
            left: Box::new(acc),
            op: ast::BinaryOp::And,
            right: Box::new(next),
        },
        ty: ColumnType::Bool,
    }))
}

/// Try to turn one conjunct into a [`HashKey`]. Succeeds only for an equality
/// whose two sides reference, respectively, only left columns and only right
/// columns (in either textual order) and share a hashable type. Otherwise the
/// conjunct is returned unchanged as residual.
pub(super) fn classify_conjunct(
    conjunct: TypedExpr,
    left_width: usize,
) -> Result<HashKey, TypedExpr> {
    let TypedExprKind::Binary {
        left,
        op: ast::BinaryOp::Eq,
        right,
    } = conjunct.kind
    else {
        return Err(conjunct);
    };
    // Compare physical types: VARCHAR/CHAR key TEXT-typed values, and the declared integer widths
    // key the same runtime i64 — the equality the executor hashes is on the runtime values.
    let same_hashable_type =
        left.ty.physical() == right.ty.physical() && is_hashable_key_type(left.ty.physical());
    let key = match (
        column_side(&left, left_width),
        column_side(&right, left_width),
    ) {
        (Side::Left, Side::Right) if same_hashable_type => HashKey {
            left: *left,
            right: *right,
        },
        (Side::Right, Side::Left) if same_hashable_type => HashKey {
            left: *right,
            right: *left,
        },
        _ => {
            // Not a usable equi-key — hand it back as residual.
            return Err(TypedExpr {
                kind: TypedExprKind::Binary {
                    left,
                    op: ast::BinaryOp::Eq,
                    right,
                },
                ty: conjunct.ty,
            });
        },
    };
    Ok(key)
}

/// Only types whose value-equality the executor can hash **compare-compatibly** (equal values ⇒
/// equal key atoms) are eligible as hash keys. widened this from the
/// original `Int`/`Bool`/`Text`: `BIGINT` is the standard ID type, so almost every real-world
/// equi-join was silently falling to the O(n²) nested loop (QA: 5k×5k = ~6s, 300k×300k = hung).
/// The integer widths share one runtime `i64`; the temporals are exact integer counts (TIMETZ's
/// packed form compares exactly); UUID/BYTEA are exact bytes; NUMERIC canonicalizes through the
/// trimmed exact decimal (`1.0` = `1.00`). `Float` stays excluded (NaN / `-0.0` hashing hazards),
/// as do `Interval` (mixed-unit equality: `1 mon` = `30 days`), `Json`, and the containers. Such
/// joins fall back to nested-loop, which evaluates equality through the standard evaluator.
/// Called with [`ColumnType::physical`] types.
pub(super) const fn is_hashable_key_type(ty: ColumnType) -> bool {
    matches!(
        ty,
        ColumnType::Int
            | ColumnType::SmallInt
            | ColumnType::BigInt
            | ColumnType::Bool
            | ColumnType::Text
            | ColumnType::Date
            | ColumnType::Time
            | ColumnType::TimeTz
            | ColumnType::Timestamp
            | ColumnType::TimestampTz
            | ColumnType::Uuid
            | ColumnType::Numeric { .. }
            | ColumnType::Bytes
    )
}

/// Which input's columns an expression references.
pub(super) enum Side {
    /// References only left-input columns (or one specific side cleanly).
    Left,
    /// References only right-input columns.
    Right,
    /// References no columns (constant) or columns from both sides — not a
    /// usable join key.
    NeitherOrBoth,
}

/// Classify which join input `expr`'s column references belong to. A constant
/// (no columns) or a mix of both sides is [`Side::NeitherOrBoth`].
pub(super) fn column_side(expr: &TypedExpr, left_width: usize) -> Side {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    if cols.is_empty() {
        Side::NeitherOrBoth
    } else if cols.iter().all(|&c| c < left_width) {
        Side::Left
    } else if cols.iter().all(|&c| c >= left_width) {
        Side::Right
    } else {
        Side::NeitherOrBoth
    }
}

/// Shift every `Column` ordinal in `expr` down by `shift` — used to rebase a right-only join
/// conjunct from the joined-row coordinate space into the right input's own `[0, right_width)`
/// space when pushing it below the join. Only reached for a conjunct with no subquery (guarded at
/// the call site), so the subquery/`OuterColumn` arms are inert; they leave any nested scope
/// untouched, which is correct even if one ever slips through.
pub(super) fn remap_columns(expr: &mut TypedExpr, shift: usize) {
    match &mut expr.kind {
        TypedExprKind::Column(index) => *index = index.saturating_sub(shift),
        TypedExprKind::Literal(_)
        | TypedExprKind::OuterColumn { .. }
        | TypedExprKind::AggregateRef(_)
        | TypedExprKind::ScalarSubquery(_)
        | TypedExprKind::Exists { .. } => {},
        TypedExprKind::Binary { left, right, .. }
        | TypedExprKind::IsDistinctFrom { left, right, .. } => {
            remap_columns(left, shift);
            remap_columns(right, shift);
        },
        TypedExprKind::Unary { expr: inner, .. }
        | TypedExprKind::IsNull { expr: inner, .. }
        | TypedExprKind::IsBool { expr: inner, .. }
        | TypedExprKind::Cast(inner, _)
        | TypedExprKind::InSubquery { expr: inner, .. }
        | TypedExprKind::QuantifiedSubquery { expr: inner, .. } => remap_columns(inner, shift),
        TypedExprKind::QuantifiedArray { expr, array, .. } => {
            remap_columns(expr, shift);
            remap_columns(array, shift);
        },
        TypedExprKind::InList {
            expr: inner, list, ..
        } => {
            remap_columns(inner, shift);
            for item in list {
                remap_columns(item, shift);
            }
        },
        TypedExprKind::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            remap_columns(inner, shift);
            remap_columns(low, shift);
            remap_columns(high, shift);
        },
        TypedExprKind::Like {
            expr: inner,
            pattern,
            ..
        }
        | TypedExprKind::RegexMatch {
            expr: inner,
            pattern,
            ..
        }
        | TypedExprKind::SimilarTo {
            expr: inner,
            pattern,
            ..
        } => {
            remap_columns(inner, shift);
            remap_columns(pattern, shift);
        },
        TypedExprKind::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(op) = operand {
                remap_columns(op, shift);
            }
            for branch in branches {
                remap_columns(&mut branch.when, shift);
                remap_columns(&mut branch.then, shift);
            }
            if let Some(def) = default {
                remap_columns(def, shift);
            }
        },
        TypedExprKind::Coalesce(args)
        | TypedExprKind::ScalarFunction { args, .. }
        | TypedExprKind::ScalarUdf { args, .. }
        | TypedExprKind::ArrayLiteral(args)
        | TypedExprKind::SetReturning { args, .. } => {
            for arg in args {
                remap_columns(arg, shift);
            }
        },
        TypedExprKind::Crypto { value, key, .. } => {
            remap_columns(value, shift);
            remap_columns(key, shift);
        },
        TypedExprKind::Subscript { base, index } => {
            remap_columns(base, shift);
            remap_columns(index, shift);
        },
        TypedExprKind::ArraySlice { base, lower, upper } => {
            remap_columns(base, shift);
            for bound in [lower, upper].into_iter().flatten() {
                remap_columns(bound, shift);
            }
        },
    }
}

/// Gather every `Column` ordinal referenced anywhere in `expr`.
pub(super) fn collect_columns(expr: &TypedExpr, out: &mut Vec<usize>) {
    match &expr.kind {
        TypedExprKind::Column(index) => out.push(*index),
        // An `OuterColumn` indexes an enclosing query's row, not this join's input, so it is not a
        // column of either side. Literals and scalar/EXISTS subquery bodies reference none either.
        TypedExprKind::Literal(_)
        | TypedExprKind::OuterColumn { .. }
        | TypedExprKind::AggregateRef(_)
        | TypedExprKind::ScalarSubquery(_)
        | TypedExprKind::Exists { .. } => {},
        TypedExprKind::Binary { left, right, .. }
        | TypedExprKind::IsDistinctFrom { left, right, .. } => {
            collect_columns(left, out);
            collect_columns(right, out);
        },
        TypedExprKind::Unary { expr: inner, .. }
        | TypedExprKind::IsNull { expr: inner, .. }
        | TypedExprKind::IsBool { expr: inner, .. }
        | TypedExprKind::Cast(inner, _)
        // Only the probe of an IN / quantified subquery can reference an outer (join-input) column.
        | TypedExprKind::InSubquery { expr: inner, .. }
        | TypedExprKind::QuantifiedSubquery { expr: inner, .. } => collect_columns(inner, out),
        TypedExprKind::QuantifiedArray { expr, array, .. } => {
            collect_columns(expr, out);
            collect_columns(array, out);
        },
        TypedExprKind::InList {
            expr: inner, list, ..
        } => {
            collect_columns(inner, out);
            for item in list {
                collect_columns(item, out);
            }
        },
        TypedExprKind::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            collect_columns(inner, out);
            collect_columns(low, out);
            collect_columns(high, out);
        },
        TypedExprKind::Like {
            expr: inner,
            pattern,
            ..
        }
        | TypedExprKind::RegexMatch {
            expr: inner,
            pattern,
            ..
        }
        | TypedExprKind::SimilarTo {
            expr: inner,
            pattern,
            ..
        } => {
            collect_columns(inner, out);
            collect_columns(pattern, out);
        },
        TypedExprKind::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(op) = operand {
                collect_columns(op, out);
            }
            for branch in branches {
                collect_columns(&branch.when, out);
                collect_columns(&branch.then, out);
            }
            if let Some(def) = default {
                collect_columns(def, out);
            }
        },
        TypedExprKind::Coalesce(args)
        | TypedExprKind::ScalarFunction { args, .. }
        | TypedExprKind::ScalarUdf { args, .. }
        | TypedExprKind::ArrayLiteral(args)
        // A set-returning function only appears in a projection, never a join predicate, but cover
        // its arguments here for completeness.
        | TypedExprKind::SetReturning { args, .. } => {
            for arg in args {
                collect_columns(arg, out);
            }
        },
        TypedExprKind::Crypto { value, key, .. } => {
            collect_columns(value, out);
            collect_columns(key, out);
        },
        TypedExprKind::Subscript { base, index } => {
            collect_columns(base, out);
            collect_columns(index, out);
        },
        TypedExprKind::ArraySlice { base, lower, upper } => {
            collect_columns(base, out);
            for bound in [lower, upper].into_iter().flatten() {
                collect_columns(bound, out);
            }
        },
    }
}
