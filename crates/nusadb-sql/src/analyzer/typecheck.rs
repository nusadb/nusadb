//! Type rules: assignability, unification, and the per-operator type checks.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === Type rules ===========================================================

pub(super) fn check_assignable(column: &ColumnDef, value: &TypedExpr) -> Result<(), Error> {
    if assignable(column.ty, value.ty) {
        Ok(())
    } else {
        Err(Error::TypeMismatch {
            context: format!("assignment to column `{}`", column.name),
            expected: column.ty,
            found: value.ty,
        })
    }
}

/// Whether a `source`-typed value may be stored into a `target`-typed column.
/// Identical types always assign; an integer additionally widens into a float; and a text value
/// coerces into a temporal / UUID column (parsed at encode time, implicit unknown-literal rule).
pub(super) fn assignable(target: ColumnType, source: ColumnType) -> bool {
    // `VARCHAR(n)`/`CHAR(n)` are TEXT for assignability — the declared length is enforced by the
    // desugared `CHECK`, not by the type rules. Normalize both sides so a text value assigns into a
    // character column (and a character column assigns wherever TEXT would).
    let target = target.physical();
    let source = source.physical();
    target == source
        // An exact INT or NUMERIC literal/value widens into a FLOAT column (the float's inexactness
        // is accepted at the call site). NUMERIC→FLOAT matters since a plain decimal literal now
        // types as NUMERIC, so `INSERT INTO t(float_col) VALUES (0.5)` must still assign.
        || (target == ColumnType::Float
            && matches!(source, ColumnType::Int | ColumnType::Numeric { .. }))
        || (source == ColumnType::Text && is_temporal_or_uuid(target))
        // TIMESTAMP ↔ TIMESTAMPTZ: an assignment cast in either direction, so the ubiquitous
        // `DEFAULT CURRENT_TIMESTAMP` (a `timestamptz`) fills a plain `timestamp` column. The two
        // differ only by the session time zone, which is fixed at UTC here, so the instant is
        // preserved exactly; the conversion happens in `row::adopt_column_type`.
        || matches!(
            (target, source),
            (ColumnType::Timestamp, ColumnType::TimestampTz)
                | (ColumnType::TimestampTz, ColumnType::Timestamp)
        )
        // NUMERIC accepts Int / Float / Text / any-scale Numeric, rescaled at encode time.
        || (matches!(target, ColumnType::Numeric { .. })
            && matches!(
                source,
                ColumnType::Int | ColumnType::Float | ColumnType::Text | ColumnType::Numeric { .. }
            ))
        // JSON accepts a text value (parsed + canonicalized at encode time).
        || (target == ColumnType::Json && source == ColumnType::Text)
        // The full-text types accept a text literal (parsed + canonicalized at encode time).
        || (matches!(target, ColumnType::Tsvector | ColumnType::Tsquery)
            && source == ColumnType::Text)
        // ARRAY accepts a `{...}` text literal (parsed at encode time).
        || (matches!(target, ColumnType::Array(_)) && source == ColumnType::Text)
        // INTERVAL accepts a text literal (parsed at encode time).
        || (target == ColumnType::Interval && source == ColumnType::Text)
        // VECTOR accepts a `[..]` text literal (parsed + dimension-checked at encode time).
        || (matches!(target, ColumnType::Vector(_)) && source == ColumnType::Text)
        // BYTEA accepts a `\x<hex>` text literal (parsed at encode time).
        || (target == ColumnType::Bytes && source == ColumnType::Text)
}

pub(super) const fn is_numeric(ty: ColumnType) -> bool {
    matches!(
        ty,
        ColumnType::SmallInt
            | ColumnType::Int
            | ColumnType::BigInt
            | ColumnType::Float
            | ColumnType::Numeric { .. }
    )
}

/// The type an expression carries for a value of declared type `ty`. The integer widths
/// (`SMALLINT`/`INT`/`BIGINT`) are preserved so their range is enforced in arithmetic and casts;
/// every other declared width (`VARCHAR(n)`/`CHAR(n)`/`REAL`/`JSONB`) collapses to
/// its physical type, which carries no runtime distinction. Use this wherever a column's or cast's
/// declared type becomes an expression type.
pub(super) const fn expr_type(ty: ColumnType) -> ColumnType {
    match ty {
        ColumnType::SmallInt | ColumnType::Int | ColumnType::BigInt => ty,
        other => other.physical(),
    }
}

/// The wider of two integer types under the reference engine's promotion rule (`SMALLINT` < `INT` < `BIGINT`): mixed
/// integer arithmetic takes the wider operand's type, so its overflow bound is the wider one.
pub(super) const fn wider_int(a: ColumnType, b: ColumnType) -> ColumnType {
    use ColumnType::{BigInt, Int, SmallInt};
    if matches!(a, BigInt) || matches!(b, BigInt) {
        BigInt
    } else if matches!(a, Int) || matches!(b, Int) {
        Int
    } else {
        SmallInt
    }
}

/// Types whose value is parsed from a `TEXT` *literal* by the "unknown literal" coercion rule — the
/// temporal types, `UUID`, `MACADDR`, the network types, the bit strings, the ranges, and the
/// geometric types: everything with a canonical text form but no literal syntax of its own. A text
/// literal compared or assigned
/// against one of these is
/// re-typed as that type (parsed at evaluation, loud-rejecting a bad string), so e.g.
/// `WHERE mac = '08:00:2b:01:02:03'` and `INSERT ... VALUES ('08:00:2b:01:02:03')` work.
pub(super) const fn is_temporal_or_uuid(ty: ColumnType) -> bool {
    matches!(
        ty,
        ColumnType::Date
            | ColumnType::Time
            | ColumnType::TimeTz
            | ColumnType::Timestamp
            | ColumnType::TimestampTz
            | ColumnType::Uuid
            | ColumnType::Macaddr
            | ColumnType::Macaddr8
            | ColumnType::Inet
            | ColumnType::Cidr
            | ColumnType::Bit(_)
            | ColumnType::VarBit(_)
            | ColumnType::Range(_)
            | ColumnType::Geometry(_)
    )
}

pub(super) const fn is_bare_null(expr: &ast::Expr) -> bool {
    matches!(expr, ast::Expr::Literal(ast::Value::Null))
}

pub(super) const fn is_null_literal(typed: &TypedExpr) -> bool {
    matches!(typed.kind, TypedExprKind::Literal(ast::Value::Null))
}
