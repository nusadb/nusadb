//! Predicate implication for partial indexes.
//!
//! A partial index holds only the rows its predicate is true for, so a scan through it may answer a
//! query only when every row the query can return is among them: the query's `WHERE` must *imply*
//! the index predicate. [`implies`] proves that from the two conjunct lists, and answers `false`
//! whenever it cannot, so an index is never used on a guess.
//!
//! Each index conjunct must follow from one query conjunct (conjuncts are not combined). The rules:
//!
//! - an identical conjunct;
//! - `e = c`, with the constant on either side, against another constant comparison on the same
//!   `e`: `e = 5` gives `e = 5`, `e >= 3`, `e > 4`, `e IN (5, 6)`; `e > 10` gives `e > 5` and
//!   `e >= 10`; an `IN` list gives what every one of its values gives;
//! - a bare boolean column `b` is `b = true`, and `NOT b` is `b = false`;
//! - `e IS NOT NULL` follows from any comparison with `e` on one side (whatever the other side is),
//!   and from `IN` or `BETWEEN` on `e`, as each is NULL (not true) when `e` is NULL;
//! - an index conjunct `a OR b` follows from whatever gives `a` or gives `b`, and a query conjunct
//!   `a OR b` gives what both `a` and `b` give.
//!
//! Constants are compared with the executor's own comparison, and only between constants of one
//! type whose order is total and exact (booleans, integers, `NUMERIC`, text, dates, times,
//! timestamps without time zone, UUIDs), on an expression of that same type. Anything else is
//! proven only by an identical conjunct.

use std::borrow::Cow;
use std::cmp::Ordering;

use nusadb_core::ColumnType;

use super::plan_types::{TypedExpr, TypedExprKind};
use crate::ast;

/// How deep [`expr_implies`] follows nested `AND`/`OR` before giving up (answering "not proven").
const MAX_DEPTH: usize = 8;

/// Whether the conjuncts `query` (the `WHERE` a scan answers) imply every conjunct of `index`
/// (a partial index's predicate). `false` unless proven.
pub(super) fn implies(query: &[&TypedExpr], index: &[TypedExpr]) -> bool {
    index
        .iter()
        .all(|needed| query.iter().any(|given| expr_implies(given, needed, 0)))
}

/// Whether `given` (true for a row) proves `needed` true for that row.
fn expr_implies(given: &TypedExpr, needed: &TypedExpr, depth: usize) -> bool {
    if given == needed {
        return true;
    }
    if depth >= MAX_DEPTH {
        return false;
    }
    let next = depth + 1;
    // `needed` is `a OR b`: either side proven is enough.
    if let Some(parts) = branches(needed, ast::BinaryOp::Or)
        && parts.iter().any(|p| expr_implies(given, p, next))
    {
        return true;
    }
    // `needed` is `a AND b` (inside an OR): both sides must be proven.
    if let Some(parts) = branches(needed, ast::BinaryOp::And)
        && parts.iter().all(|p| expr_implies(given, p, next))
    {
        return true;
    }
    // `given` is `a OR b`: whichever side holds, it must prove `needed`.
    if let Some(parts) = branches(given, ast::BinaryOp::Or) {
        return parts.iter().all(|g| expr_implies(g, needed, next));
    }
    // `given` is `a AND b` (inside an OR): one side proving it is enough.
    if let Some(parts) = branches(given, ast::BinaryOp::And) {
        return parts.iter().any(|g| expr_implies(g, needed, next));
    }
    if let TypedExprKind::IsNull {
        expr: operand,
        negated: true,
    } = &needed.kind
        && strict_comparison_on(given, operand)
    {
        return true;
    }
    match (Atom::of(given), Atom::of(needed)) {
        (Some(g), Some(n)) => atom_implies(&g, &n),
        _ => false,
    }
}

/// Whether `given` is a comparison with `operand` as one side, whatever the other: the executor's
/// comparison is NULL when either side is, so such a conjunct holds only for a non-NULL `operand`.
fn strict_comparison_on(given: &TypedExpr, operand: &TypedExpr) -> bool {
    matches!(
        &given.kind,
        TypedExprKind::Binary { left, op, right }
            if is_comparison(*op) && (**left == *operand || **right == *operand)
    )
}

/// The operands of a chain of `op` (`AND` or `OR`), flattened; `None` when `expr` is not one.
fn branches(expr: &TypedExpr, op: ast::BinaryOp) -> Option<Vec<&TypedExpr>> {
    fn walk<'a>(expr: &'a TypedExpr, op: ast::BinaryOp, out: &mut Vec<&'a TypedExpr>) {
        match &expr.kind {
            TypedExprKind::Binary { left, op: o, right } if *o == op => {
                walk(left, op, out);
                walk(right, op, out);
            },
            _ => out.push(expr),
        }
    }
    match &expr.kind {
        TypedExprKind::Binary { op: o, .. } if *o == op => {
            let mut out = Vec::new();
            walk(expr, op, &mut out);
            Some(out)
        },
        _ => None,
    }
}

/// A constant of a type whose order the implication rules may reason with.
#[derive(Clone)]
struct Constant<'a> {
    value: Cow<'a, ast::Value>,
    ty: ColumnType,
}

impl Constant<'_> {
    /// The order of `self` against `other` under the executor's comparison, or `None` when the two
    /// are not of one exactly ordered type.
    fn order(&self, other: &Self) -> Option<Ordering> {
        let (a, b) = (self.value.as_ref(), other.value.as_ref());
        (self.ty == other.ty && same_ordered_variant(a, b))
            .then(|| crate::executor::eval::compare(a, b))
    }
}

/// Whether `a` and `b` are the same value variant, one with a total, exact order under
/// [`crate::executor::eval::compare`]. Floating-point values are left out (their order against
/// other numbers goes through `f64`), as are the time-zone-dependent and composite types.
const fn same_ordered_variant(a: &ast::Value, b: &ast::Value) -> bool {
    use ast::Value as V;
    matches!(
        (a, b),
        (V::Bool(_), V::Bool(_))
            | (V::Int(_), V::Int(_))
            | (V::Numeric(_), V::Numeric(_))
            | (V::Text(_), V::Text(_))
            | (V::Date(_), V::Date(_))
            | (V::Time(_), V::Time(_))
            | (V::Timestamp(_), V::Timestamp(_))
            | (V::Uuid(_), V::Uuid(_))
    )
}

/// One conjunct in the normal form the rules reason about.
enum Atom<'a> {
    /// `operand <op> constant`, the operand on the left. `op` is a comparison.
    Compare {
        operand: &'a TypedExpr,
        op: ast::BinaryOp,
        constant: Constant<'a>,
    },
    /// `operand IN (constants)`, without its `NULL` elements (which never make it true).
    In {
        operand: &'a TypedExpr,
        constants: Vec<Constant<'a>>,
    },
    /// `operand BETWEEN low AND high` (only its strictness in `operand` is used).
    Between { operand: &'a TypedExpr },
    /// `operand IS NOT NULL`.
    NotNull { operand: &'a TypedExpr },
}

impl<'a> Atom<'a> {
    fn of(expr: &'a TypedExpr) -> Option<Self> {
        match &expr.kind {
            TypedExprKind::Binary { left, op, right } => {
                let (operand, op, literal) = match (&left.kind, &right.kind) {
                    (_, TypedExprKind::Literal(_)) => (&**left, *op, &**right),
                    (TypedExprKind::Literal(_), _) => (&**right, mirror(*op)?, &**left),
                    _ => return None,
                };
                if !is_comparison(op) {
                    return None;
                }
                let TypedExprKind::Literal(value) = &literal.kind else {
                    return None;
                };
                // A comparison with NULL is never true; nothing is concluded from it.
                if matches!(value, ast::Value::Null) || is_constant(operand) {
                    return None;
                }
                Some(Self::Compare {
                    operand,
                    op,
                    constant: Constant {
                        value: Cow::Borrowed(value),
                        ty: literal.ty,
                    },
                })
            },
            TypedExprKind::InList {
                expr: operand,
                list,
                negated: false,
            } if !is_constant(operand) => {
                let mut constants = Vec::with_capacity(list.len());
                for item in list {
                    match &item.kind {
                        TypedExprKind::Literal(ast::Value::Null) => {},
                        TypedExprKind::Literal(value) => constants.push(Constant {
                            value: Cow::Borrowed(value),
                            ty: item.ty,
                        }),
                        _ => return None,
                    }
                }
                Some(Self::In { operand, constants })
            },
            TypedExprKind::Between {
                expr: operand,
                negated: false,
                ..
            } if !is_constant(operand) => Some(Self::Between { operand }),
            TypedExprKind::IsNull {
                expr: operand,
                negated: true,
            } => Some(Self::NotNull { operand }),
            // A boolean column on its own is `col = true`; `NOT col` is `col = false`.
            TypedExprKind::Column(_) if expr.ty == ColumnType::Bool => Some(Self::Compare {
                operand: expr,
                op: ast::BinaryOp::Eq,
                constant: Constant {
                    value: Cow::Owned(ast::Value::Bool(true)),
                    ty: ColumnType::Bool,
                },
            }),
            TypedExprKind::Unary {
                op: ast::UnaryOp::Not,
                expr: operand,
            } if operand.ty == ColumnType::Bool
                && matches!(operand.kind, TypedExprKind::Column(_)) =>
            {
                Some(Self::Compare {
                    operand,
                    op: ast::BinaryOp::Eq,
                    constant: Constant {
                        value: Cow::Owned(ast::Value::Bool(false)),
                        ty: ColumnType::Bool,
                    },
                })
            },
            _ => None,
        }
    }

    /// The expression the atom constrains.
    const fn operand(&self) -> &'a TypedExpr {
        match self {
            Self::Compare { operand, .. }
            | Self::In { operand, .. }
            | Self::Between { operand }
            | Self::NotNull { operand } => operand,
        }
    }
}

/// Whether `expr` reads nothing from the row (a literal, possibly cast or negated).
fn is_constant(expr: &TypedExpr) -> bool {
    match &expr.kind {
        TypedExprKind::Literal(_) => true,
        TypedExprKind::Cast(inner, _) | TypedExprKind::Unary { expr: inner, .. } => {
            is_constant(inner)
        },
        _ => false,
    }
}

const fn is_comparison(op: ast::BinaryOp) -> bool {
    matches!(
        op,
        ast::BinaryOp::Eq
            | ast::BinaryOp::NotEq
            | ast::BinaryOp::Lt
            | ast::BinaryOp::LtEq
            | ast::BinaryOp::Gt
            | ast::BinaryOp::GtEq
    )
}

/// `c <op> e` as `e <mirrored> c`.
const fn mirror(op: ast::BinaryOp) -> Option<ast::BinaryOp> {
    Some(match op {
        ast::BinaryOp::Eq => ast::BinaryOp::Eq,
        ast::BinaryOp::NotEq => ast::BinaryOp::NotEq,
        ast::BinaryOp::Lt => ast::BinaryOp::Gt,
        ast::BinaryOp::LtEq => ast::BinaryOp::GtEq,
        ast::BinaryOp::Gt => ast::BinaryOp::Lt,
        ast::BinaryOp::GtEq => ast::BinaryOp::LtEq,
        _ => return None,
    })
}

/// Whether atom `given` proves atom `needed` (both over the same operand).
fn atom_implies(given: &Atom<'_>, needed: &Atom<'_>) -> bool {
    if given.operand() != needed.operand() {
        return false;
    }
    match needed {
        // Every comparison, `IN` and `BETWEEN` is NULL when its operand is, so it holds only for a
        // non-NULL operand.
        Atom::NotNull { .. } => match given {
            Atom::In { constants, .. } => !constants.is_empty(),
            Atom::Compare { .. } | Atom::Between { .. } | Atom::NotNull { .. } => true,
        },
        Atom::Compare {
            op: needed_op,
            constant: needed_constant,
            ..
        } => given_values(given).is_some_and(|values| {
            !values.is_empty()
                && values.iter().all(|(op, constant)| {
                    compare_implies(*op, constant, *needed_op, needed_constant, needed)
                })
        }),
        Atom::In {
            constants: allowed, ..
        } => given_values(given).is_some_and(|values| {
            !values.is_empty()
                && values.iter().all(|(op, constant)| {
                    *op == ast::BinaryOp::Eq
                        && allowed
                            .iter()
                            .any(|a| constant.order(a) == Some(Ordering::Equal))
                })
        }),
        Atom::Between { .. } => false,
    }
}

/// `given` as alternatives `operand <op> constant`, one of which holds: a comparison is one, an `IN`
/// list is an equality per value. `None` for an atom that is neither.
fn given_values<'g, 'a>(given: &'g Atom<'a>) -> Option<Vec<(ast::BinaryOp, &'g Constant<'a>)>> {
    match given {
        Atom::Compare { op, constant, .. } => Some(vec![(*op, constant)]),
        Atom::In { constants, .. } => {
            Some(constants.iter().map(|c| (ast::BinaryOp::Eq, c)).collect())
        },
        Atom::Between { .. } | Atom::NotNull { .. } => None,
    }
}

/// Whether `e <given_op> b` proves `e <needed_op> a`, on an operand of the constants' type.
fn compare_implies(
    given_op: ast::BinaryOp,
    b: &Constant<'_>,
    needed_op: ast::BinaryOp,
    a: &Constant<'_>,
    needed: &Atom<'_>,
) -> bool {
    use ast::BinaryOp::{Eq, Gt, GtEq, Lt, LtEq, NotEq};
    // The operand must have the constants' own type, so it is compared exactly as they are.
    let operand_ty = needed.operand().ty.physical();
    if crate::executor::row::runtime_type_of(&a.value).physical() != operand_ty {
        return false;
    }
    let Some(order) = b.order(a) else {
        return false;
    };
    // `e <given_op> b` proves `e <needed_op> a` exactly when `b` stands to `a` as below: an
    // equality proves what its constant satisfies, a lower bound proves a lower bound at or below
    // it, an upper bound an upper bound at or above it, and a bound proves `<> a` for an `a`
    // outside it.
    match (given_op, needed_op) {
        (Eq, Eq) | (NotEq, NotEq) => order == Ordering::Equal,
        (Eq, NotEq) => order != Ordering::Equal,
        (Eq | GtEq, Gt) | (GtEq, NotEq) => order == Ordering::Greater,
        (Eq | GtEq, GtEq) | (Gt, Gt | GtEq | NotEq) => order != Ordering::Less,
        (Eq | LtEq, Lt) | (LtEq, NotEq) => order == Ordering::Less,
        (Eq | LtEq, LtEq) | (Lt, Lt | LtEq | NotEq) => order != Ordering::Greater,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::implies;
    use crate::ast;
    use crate::planner::plan_types::{TypedExpr, TypedExprKind};
    use nusadb_core::ColumnType;

    fn col(ordinal: usize, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Column(ordinal),
            ty,
        }
    }

    fn lit(value: ast::Value, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Literal(value),
            ty,
        }
    }

    fn int(v: i64) -> TypedExpr {
        lit(ast::Value::Int(v), ColumnType::Int)
    }

    fn text(v: &str) -> TypedExpr {
        lit(ast::Value::Text(v.to_owned()), ColumnType::Text)
    }

    fn bin(left: TypedExpr, op: ast::BinaryOp, right: TypedExpr) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            ty: ColumnType::Bool,
        }
    }

    fn not_null(e: TypedExpr, negated: bool) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::IsNull {
                expr: Box::new(e),
                negated,
            },
            ty: ColumnType::Bool,
        }
    }

    fn in_list(e: TypedExpr, list: Vec<TypedExpr>) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::InList {
                expr: Box::new(e),
                list,
                negated: false,
            },
            ty: ColumnType::Bool,
        }
    }

    fn proves(query: &[TypedExpr], index: &[TypedExpr]) -> bool {
        let refs: Vec<&TypedExpr> = query.iter().collect();
        implies(&refs, index)
    }

    use ast::BinaryOp::{Eq, Gt, GtEq, Lt, LtEq, NotEq, Or};

    fn k() -> TypedExpr {
        col(0, ColumnType::Int)
    }

    fn s() -> TypedExpr {
        col(1, ColumnType::Text)
    }

    #[test]
    fn identical_and_mirrored_equalities_prove() {
        let pred = bin(s(), Eq, text("active"));
        assert!(proves(
            std::slice::from_ref(&pred),
            std::slice::from_ref(&pred)
        ));
        assert!(proves(
            &[bin(text("active"), Eq, s())],
            std::slice::from_ref(&pred)
        ));
        assert!(!proves(
            &[bin(s(), Eq, text("idle"))],
            std::slice::from_ref(&pred)
        ));
        assert!(!proves(&[], &[pred]));
    }

    #[test]
    fn strict_comparisons_prove_not_null_but_is_null_does_not() {
        let pred = not_null(s(), true);
        assert!(proves(
            &[bin(s(), Eq, text("a"))],
            std::slice::from_ref(&pred)
        ));
        assert!(proves(
            &[bin(s(), NotEq, text("a"))],
            std::slice::from_ref(&pred)
        ));
        assert!(proves(
            &[in_list(s(), vec![text("a")])],
            std::slice::from_ref(&pred)
        ));
        assert!(!proves(
            &[not_null(s(), false)],
            std::slice::from_ref(&pred)
        ));
        // Whatever the other side of the comparison, it is NULL when `s` is.
        assert!(proves(
            &[bin(s(), LtEq, col(3, ColumnType::Text))],
            std::slice::from_ref(&pred)
        ));
        assert!(!proves(
            &[bin(s(), ast::BinaryOp::Plus, text("x"))],
            std::slice::from_ref(&pred)
        ));
        // `IN (NULL)` is never true, but nothing is concluded from it.
        let only_null = in_list(s(), vec![lit(ast::Value::Null, ColumnType::Text)]);
        assert!(!proves(&[only_null], std::slice::from_ref(&pred)));
        // Another column proves nothing about this one.
        assert!(!proves(&[bin(k(), Eq, int(1))], &[pred]));
        // A comparison with NULL proves nothing.
        assert!(!proves(
            &[bin(s(), Eq, lit(ast::Value::Null, ColumnType::Text))],
            &[bin(s(), Eq, text("a"))]
        ));
    }

    #[test]
    fn ranges_prove_wider_ranges_only() {
        assert!(proves(&[bin(k(), Gt, int(10))], &[bin(k(), Gt, int(5))]));
        assert!(proves(&[bin(k(), Gt, int(5))], &[bin(k(), Gt, int(5))]));
        assert!(proves(&[bin(k(), Gt, int(5))], &[bin(k(), GtEq, int(5))]));
        assert!(!proves(&[bin(k(), GtEq, int(5))], &[bin(k(), Gt, int(5))]));
        assert!(!proves(&[bin(k(), Gt, int(4))], &[bin(k(), Gt, int(5))]));
        assert!(proves(&[bin(k(), Eq, int(6))], &[bin(k(), Gt, int(5))]));
        assert!(!proves(&[bin(k(), Eq, int(5))], &[bin(k(), Gt, int(5))]));
        assert!(proves(&[bin(int(3), Gt, k())], &[bin(k(), LtEq, int(3))]));
        assert!(proves(&[bin(k(), Lt, int(3))], &[bin(k(), NotEq, int(3))]));
        assert!(!proves(
            &[bin(k(), LtEq, int(3))],
            &[bin(k(), NotEq, int(3))]
        ));
        assert!(!proves(&[bin(k(), Lt, int(3))], &[bin(k(), Gt, int(0))]));
        // An integer constant against a text operand is not reasoned with.
        assert!(!proves(&[bin(s(), Gt, int(10))], &[bin(s(), Gt, int(5))]));
    }

    #[test]
    fn in_lists_prove_by_every_value() {
        let pred = in_list(s(), vec![text("a"), text("b")]);
        assert!(proves(
            &[bin(s(), Eq, text("a"))],
            std::slice::from_ref(&pred)
        ));
        assert!(proves(
            &[in_list(s(), vec![text("b")])],
            std::slice::from_ref(&pred)
        ));
        assert!(!proves(
            &[in_list(s(), vec![text("b"), text("c")])],
            std::slice::from_ref(&pred)
        ));
        assert!(proves(
            &[in_list(k(), vec![int(6), int(7)])],
            &[bin(k(), Gt, int(5))]
        ));
        assert!(!proves(
            &[in_list(k(), vec![int(5), int(7)])],
            &[bin(k(), Gt, int(5))]
        ));
    }

    #[test]
    fn boolean_columns_and_their_negation() {
        let flag = col(2, ColumnType::Bool);
        let not_flag = TypedExpr {
            kind: TypedExprKind::Unary {
                op: ast::UnaryOp::Not,
                expr: Box::new(flag.clone()),
            },
            ty: ColumnType::Bool,
        };
        let t = lit(ast::Value::Bool(true), ColumnType::Bool);
        let f = lit(ast::Value::Bool(false), ColumnType::Bool);
        assert!(proves(
            &[bin(flag.clone(), Eq, t.clone())],
            std::slice::from_ref(&flag)
        ));
        assert!(proves(
            std::slice::from_ref(&flag),
            &[bin(flag.clone(), Eq, t)]
        ));
        assert!(proves(
            &[bin(flag.clone(), Eq, f)],
            std::slice::from_ref(&not_flag)
        ));
        assert!(!proves(
            std::slice::from_ref(&flag),
            std::slice::from_ref(&not_flag)
        ));
        assert!(!proves(&[not_flag], &[flag]));
    }

    #[test]
    fn disjunctions() {
        let a = bin(s(), Eq, text("a"));
        let b = bin(s(), Eq, text("b"));
        let either = bin(a.clone(), Or, b);
        assert!(proves(
            std::slice::from_ref(&a),
            std::slice::from_ref(&either)
        ));
        assert!(proves(
            std::slice::from_ref(&either),
            std::slice::from_ref(&either)
        ));
        assert!(!proves(
            std::slice::from_ref(&either),
            std::slice::from_ref(&a)
        ));
        assert!(proves(
            std::slice::from_ref(&either),
            &[in_list(s(), vec![text("a"), text("b")])]
        ));
        assert!(proves(&[either], &[not_null(s(), true)]));
    }

    #[test]
    fn every_index_conjunct_must_be_proven() {
        let index = [bin(s(), Eq, text("a")), not_null(k(), true)];
        assert!(proves(
            &[bin(k(), LtEq, int(9)), bin(s(), Eq, text("a"))],
            &index
        ));
        assert!(!proves(&[bin(s(), Eq, text("a"))], &index));
    }
}
