//! Expression converters: literals/intervals, operators, CASE, functions, aggregates, window calls.
//!
//! Split verbatim out of `parser/mod.rs` (ADR 007); see that module for the
//! anti-corruption-layer contract. Cross-submodule converters resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

thread_local! {
    /// The `WINDOW w AS (...)` definitions of the SELECT currently being converted, so an
    /// `OVER w` reference resolves to the right spec. Saved/restored per SELECT so a
    /// subquery's WINDOW clause never leaks into an enclosing one.
    static NAMED_WINDOWS: std::cell::RefCell<std::collections::HashMap<String, sql::WindowSpec>> =
        std::cell::RefCell::new(std::collections::HashMap::new());

    /// Synthetic named windows minted by the window-exclude preprocessor. sqlparser does not
    /// parse a frame `EXCLUDE` clause, so the preprocessor rewrites each `OVER ( spec EXCLUDE mode )`
    /// into `OVER __nusadb_excl_win_<n>` and records `<name> -> (spec, mode)` here. Keyed by the
    /// unique synthetic name, this associates the captured exclusion with the exact window function
    /// call sqlparser produces — no positional or source-order guessing. Populated per statement by
    /// [`super::install_exclude_windows`] and consumed (removed) in [`convert_window_function`].
    static EXCLUDE_WINDOWS: std::cell::RefCell<
        std::collections::HashMap<String, (sql::WindowSpec, ast::WindowExclude)>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());

    /// `IS JSON` predicate modes minted by the `strip_is_json` preprocessor. sqlparser (0.62,
    /// the latest mainline) cannot parse the postfix `IS JSON` operator, so the preprocessor
    /// rewrites each `<expr> IS [NOT] JSON [type] [uniq]` into
    /// `<expr> IS DISTINCT FROM __nusadb_isjson_<n>` — an IS-predicate of the SAME precedence, so
    /// sqlparser binds the identical left operand — and records `<sentinel> -> (negated, item_type,
    /// unique_keys)` here. Keyed by the unique per-occurrence sentinel, this associates each captured
    /// mode with the exact predicate sqlparser produces, even when one statement carries several
    /// `IS JSON` predicates. Populated per statement by [`super::install_is_json`] and consumed
    /// (removed) in [`convert_is_distinct`].
    static IS_JSON_MODES: std::cell::RefCell<
        std::collections::HashMap<String, (bool, ast::JsonItemType, bool)>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Replace the exclude-window map with `windows`, returning the previous contents (so the caller
/// can restore or clear). Used by the parser preprocessor to seed the synthetic named windows for
/// the statement about to be converted.
pub(super) fn install_exclude_windows(
    windows: std::collections::HashMap<String, (sql::WindowSpec, ast::WindowExclude)>,
) {
    EXCLUDE_WINDOWS.with(|c| *c.borrow_mut() = windows);
}

/// Seed the `IS JSON` predicate-mode map for the statement about to be converted (clearing any
/// leftover from a prior statement). Consumed per sentinel in [`convert_is_distinct`].
pub(super) fn install_is_json(
    modes: std::collections::HashMap<String, (bool, ast::JsonItemType, bool)>,
) {
    IS_JSON_MODES.with(|c| *c.borrow_mut() = modes);
}

/// Install `windows` as the current named-window scope for the lifetime of the returned guard,
/// restoring the previous scope on drop (so nested SELECT conversion stays correct).
#[must_use]
pub(super) fn enter_named_windows(
    windows: std::collections::HashMap<String, sql::WindowSpec>,
) -> NamedWindowsGuard {
    let previous = NAMED_WINDOWS.with(|c| c.replace(windows));
    NamedWindowsGuard { previous }
}

/// RAII guard restoring the previous named-window scope on drop (see [`enter_named_windows`]).
pub(super) struct NamedWindowsGuard {
    previous: std::collections::HashMap<String, sql::WindowSpec>,
}

impl Drop for NamedWindowsGuard {
    fn drop(&mut self) {
        NAMED_WINDOWS.with(|c| *c.borrow_mut() = std::mem::take(&mut self.previous));
    }
}

// === Expressions ==========================================================

/// Convert a typed string literal (`DATE '…'`, `TIME '…'`, `TIMESTAMP '…'`, `TIMESTAMP WITH TIME
/// ZONE '…'`) into the corresponding [`ast::Value`]. A type without a dedicated literal form
/// (e.g. `TEXT '…'`) falls back to a plain text literal; a malformed value is rejected.
pub(super) fn convert_typed_literal(
    data_type: &sql::DataType,
    value: String,
) -> Result<ast::Expr, Error> {
    let ty = convert_data_type(data_type)?;
    let parsed = match ty {
        ColumnType::Date => crate::temporal::parse_date(&value).map(ast::Value::Date),
        ColumnType::Time => crate::temporal::parse_time(&value).map(ast::Value::Time),
        ColumnType::Timestamp => {
            crate::temporal::parse_timestamp(&value).map(ast::Value::Timestamp)
        },
        ColumnType::TimestampTz => {
            crate::temporal::parse_timestamptz(&value).map(ast::Value::TimestampTz)
        },
        ColumnType::TimeTz => crate::temporal::parse_timetz(&value).map(ast::Value::TimeTz),
        ColumnType::Uuid => crate::temporal::parse_uuid(&value).map(ast::Value::Uuid),
        ColumnType::Macaddr => crate::macaddr::parse(&value).map(ast::Value::Macaddr),
        ColumnType::Inet => crate::inet::parse_inet(&value).map(ast::Value::Inet),
        ColumnType::Cidr => crate::inet::parse_cidr(&value).map(ast::Value::Inet),
        ColumnType::Bit(n) => {
            crate::bit::parse(&value).map(|b| ast::Value::Bit(crate::bit::fit(&b, n as usize)))
        },
        ColumnType::VarBit(max) => crate::bit::parse(&value)
            .map(|b| ast::Value::Bit(crate::bit::truncate(&b, max.map(|n| n as usize)))),
        ColumnType::Range(kind) => {
            crate::range::parse(&value, kind).map(|r| ast::Value::Range(Box::new(r)))
        },
        // `NUMERIC '…'` / `DECIMAL '…'` parse exactly.
        ColumnType::Numeric { .. } => {
            crate::numeric::Decimal::parse(&value).map(ast::Value::Numeric)
        },
        // `JSON '…'` parses + canonicalizes.
        ColumnType::Json => crate::json::canonicalize(&value).map(ast::Value::Json),
        // Other typed strings (e.g. `TEXT '…'`) carry no special parsing — treat as text.
        _ => return Ok(ast::Expr::Literal(ast::Value::Text(value))),
    };
    parsed
        .map(ast::Expr::Literal)
        .ok_or(Error::InvalidValue { ty, value })
}

/// Convert an `INTERVAL '…'` literal. The value is a string (`'1 day'`) — possibly a bare
/// number combined with a trailing field (`INTERVAL '1' DAY`).
pub(super) fn convert_interval(interval: sql::Interval) -> Result<ast::Expr, Error> {
    let raw = match *interval.value {
        sql::Expr::Value(sql::ValueWithSpan {
            value:
                sql::Value::SingleQuotedString(s)
                | sql::Value::DoubleQuotedString(s)
                | sql::Value::EscapedStringLiteral(s)
                | sql::Value::UnicodeStringLiteral(s),
            ..
        }) => s,
        sql::Expr::Value(sql::ValueWithSpan {
            value: sql::Value::Number(n, _),
            ..
        }) => n,
        other => return unsupported(&format!("INTERVAL value `{other}`")),
    };
    let text = match interval.leading_field {
        Some(field) => format!("{raw} {}", interval_field_unit(&field)),
        None => raw,
    };
    crate::interval::Interval::parse(&text)
        .map(ast::Value::Interval)
        .map(ast::Expr::Literal)
        .ok_or(Error::InvalidValue {
            ty: ColumnType::Interval,
            value: text,
        })
}

/// The interval unit word for a sqlparser `DateTimeField` leading field.
pub(super) const fn interval_field_unit(field: &sql::DateTimeField) -> &'static str {
    use sql::DateTimeField as F;
    match field {
        F::Year => "year",
        F::Month => "month",
        F::Week(_) => "week",
        F::Day => "day",
        F::Hour => "hour",
        F::Minute => "minute",
        _ => "second",
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per sqlparser expression variant; a flat dispatch reads better than splitting it"
)]
pub(super) fn convert_expr(expr: sql::Expr) -> Result<ast::Expr, Error> {
    match expr {
        // A surviving `IS JSON` sentinel reaching here means the predicate was chained directly
        // before another postfix/ternary operator (`x IS JSON IS NULL`, `x IS JSON BETWEEN a AND b`)
        // whose node shape the spine re-association does not descend through, so the synthetic name
        // leaked. Reject it clearly — never surface the internal name as an "unknown column" — and
        // point at the parenthesised form, which parses fine: `(x IS JSON) IS NULL`.
        sql::Expr::Identifier(ident)
            if ident.quote_style.is_none() && ident.value.starts_with(super::IS_JSON_PREFIX) =>
        {
            unsupported(
                "IS JSON written directly before another operator (IS/BETWEEN/IN/LIKE …) is not \
                 supported; parenthesise it, e.g. `(x IS JSON) IS NULL`",
            )
        },
        sql::Expr::Identifier(ident) => Ok(ast::Expr::Column(fold_ident(&ident))),
        sql::Expr::Value(sql::ValueWithSpan {
            value: sql::Value::Placeholder(p),
            ..
        }) => convert_placeholder(&p),
        sql::Expr::Value(value) => convert_value(value.value).map(ast::Expr::Literal),
        // Typed string literals: `DATE '…'`, `TIME '…'`, `TIMESTAMP '…'`,
        // `TIMESTAMP WITH TIME ZONE '…'`. The ODBC `{d '…'}` spelling (newly parsed by
        // sqlparser 0.62) is out of surface.
        sql::Expr::TypedString(ts) => {
            if ts.uses_odbc_syntax {
                return unsupported("ODBC-syntax typed literal `{d/t/ts '...'}`");
            }
            let value = match ts.value.value {
                sql::Value::SingleQuotedString(s)
                | sql::Value::DoubleQuotedString(s)
                | sql::Value::EscapedStringLiteral(s)
                | sql::Value::UnicodeStringLiteral(s) => s,
                other => return unsupported(&format!("typed literal value `{other}`")),
            };
            convert_typed_literal(&ts.data_type, value)
        },
        // `INTERVAL '1 day'` / `INTERVAL '1' DAY`.
        sql::Expr::Interval(interval) => convert_interval(interval),
        sql::Expr::Nested(inner) => convert_expr(*inner),
        // `<expr> COLLATE "C"/"POSIX"` is a no-op — NusaDB already sorts text by byte value — so it
        // reduces to the inner expression; a locale collation is rejected loudly (D-COLLATE).
        sql::Expr::Collate { expr, collation } => {
            require_byte_order_collation(&collation)?;
            convert_expr(*expr)
        },
        sql::Expr::IsNull(inner) => Ok(ast::Expr::IsNull {
            expr: Box::new(convert_expr(*inner)?),
            negated: false,
        }),
        sql::Expr::IsNotNull(inner) => Ok(ast::Expr::IsNull {
            expr: Box::new(convert_expr(*inner)?),
            negated: true,
        }),
        // `IS [NOT] DISTINCT FROM`.
        sql::Expr::IsDistinctFrom(left, right) => convert_is_distinct(*left, *right, false),
        sql::Expr::IsNotDistinctFrom(left, right) => convert_is_distinct(*left, *right, true),
        // `IS [NOT] {TRUE|FALSE|UNKNOWN}`.
        sql::Expr::IsTrue(inner) => convert_is_bool(*inner, ast::TruthValue::True, false),
        sql::Expr::IsNotTrue(inner) => convert_is_bool(*inner, ast::TruthValue::True, true),
        sql::Expr::IsFalse(inner) => convert_is_bool(*inner, ast::TruthValue::False, false),
        sql::Expr::IsNotFalse(inner) => convert_is_bool(*inner, ast::TruthValue::False, true),
        sql::Expr::IsUnknown(inner) => convert_is_bool(*inner, ast::TruthValue::Unknown, false),
        sql::Expr::IsNotUnknown(inner) => convert_is_bool(*inner, ast::TruthValue::Unknown, true),
        sql::Expr::BinaryOp { left, op, right } => {
            use sql::BinaryOperator as B;
            // `(s1, e1) OVERLAPS (s2, e2)` — both sides must be two-element row expressions. The
            // parser lowers this to a dedicated `Overlaps` node so the analyzer/executor can apply
            // the period-overlap three-valued logic directly.
            if op == B::Overlaps {
                let [s1, e1] = overlaps_pair(*left)?;
                let [s2, e2] = overlaps_pair(*right)?;
                return Ok(ast::Expr::Overlaps {
                    s1: Box::new(convert_expr(s1)?),
                    e1: Box::new(convert_expr(e1)?),
                    s2: Box::new(convert_expr(s2)?),
                    e2: Box::new(convert_expr(e2)?),
                });
            }
            // POSIX regex match operators `~`/`~*`/`!~`/`!~*` become a dedicated
            // `RegexMatch` node rather than an ordinary binary op.
            let regex = match &op {
                B::PGRegexMatch => Some((true, false)),
                B::PGRegexIMatch => Some((false, false)),
                B::PGRegexNotMatch => Some((true, true)),
                B::PGRegexNotIMatch => Some((false, true)),
                _ => None,
            };
            if let Some((case_sensitive, negated)) = regex {
                Ok(ast::Expr::RegexMatch {
                    expr: Box::new(convert_expr(*left)?),
                    pattern: Box::new(convert_expr(*right)?),
                    case_sensitive,
                    negated,
                })
            } else if op == B::BitwiseXor {
                // `^`. The generic tokenizer maps the caret to `BitwiseXor`, but the
                // source text was `^`, which in the reference engine is **exponentiation** (left-associative; `#` is
                // the reference engine's XOR, handled by the dialect hook) — so lower it to the existing `power()`
                // built-in, sharing its typing and evaluation exactly.
                Ok(ast::Expr::ScalarFunction {
                    func: ast::ScalarFunc::Power,
                    args: vec![convert_expr(*left)?, convert_expr(*right)?],
                })
            } else {
                Ok(ast::Expr::Binary {
                    left: Box::new(convert_expr(*left)?),
                    op: convert_binary_op(op)?,
                    right: Box::new(convert_expr(*right)?),
                })
            }
        },
        // A minus in front of a numeric literal belongs to the literal, not to an operation on it.
        // Read apart, the most negative integer overflows: its magnitude is one past the largest
        // positive one, so it would fall through to an exact decimal and then refuse to assign to a
        // BIGINT column.
        sql::Expr::UnaryOp {
            op: sql::UnaryOperator::Minus,
            expr,
        } if matches!(
            expr.as_ref(),
            sql::Expr::Value(sql::ValueWithSpan {
                value: sql::Value::Number(..),
                ..
            })
        ) =>
        {
            let sql::Expr::Value(sql::ValueWithSpan {
                value: sql::Value::Number(n, _),
                ..
            }) = *expr
            else {
                unreachable!("guarded by the match above")
            };
            convert_number(&format!("-{n}")).map(ast::Expr::Literal)
        },
        sql::Expr::UnaryOp { op, expr } => Ok(ast::Expr::Unary {
            op: convert_unary_op(op)?,
            expr: Box::new(convert_expr(*expr)?),
        }),
        sql::Expr::CompoundIdentifier(parts) => match parts.as_slice() {
            [table, column] => Ok(ast::Expr::QualifiedColumn {
                table: fold_ident(table),
                column: fold_ident(column),
            }),
            // `public.table.column` resolves to `table.column`: `public` is the default (and
            // only) schema namespace. Any other schema qualifier is rejected (no silent collapse).
            [schema, table, column] if fold_ident(schema) == PUBLIC_SCHEMA => {
                Ok(ast::Expr::QualifiedColumn {
                    table: fold_ident(table),
                    column: fold_ident(column),
                })
            },
            _ => unsupported(
                "qualified name with more than two parts (only `public.table.column` is recognised)",
            ),
        },
        sql::Expr::InList {
            expr,
            list,
            negated,
        } => {
            let inner = convert_expr(*expr)?;
            let items = list
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Expr::InList {
                expr: Box::new(inner),
                list: items,
                negated,
            })
        },
        // `<value> AT TIME ZONE <zone>` desugars to a scalar function so it reuses the analyzer's
        // per-argument typing and the executor's dispatch.
        sql::Expr::AtTimeZone {
            timestamp,
            time_zone,
        } => Ok(ast::Expr::ScalarFunction {
            func: ast::ScalarFunc::AtTimeZone,
            args: vec![convert_expr(*timestamp)?, convert_expr(*time_zone)?],
        }),
        sql::Expr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(ast::Expr::Between {
            expr: Box::new(convert_expr(*expr)?),
            low: Box::new(convert_expr(*low)?),
            high: Box::new(convert_expr(*high)?),
            negated,
        }),
        sql::Expr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if any {
                return unsupported("LIKE ANY (...)");
            }
            Ok(ast::Expr::Like {
                expr: Box::new(convert_expr(*expr)?),
                pattern: Box::new(convert_expr(*pattern)?),
                negated,
                escape: like_escape_char(escape_char)?,
                case_insensitive: false,
            })
        },
        // ILIKE is case-insensitive LIKE. The matcher folds case per character (rather than
        // lower-casing both sides up front), so a `_` still matches one source character even when a
        // letter's lowercase form has a different length, and an alphabetic `ESCAPE` keeps its meaning
        // (deep-gate #12).
        sql::Expr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if any {
                return unsupported("ILIKE ANY (...)");
            }
            Ok(ast::Expr::Like {
                expr: Box::new(convert_expr(*expr)?),
                pattern: Box::new(convert_expr(*pattern)?),
                negated,
                escape: like_escape_char(escape_char)?,
                case_insensitive: true,
            })
        },
        sql::Expr::SimilarTo {
            negated,
            expr,
            pattern,
            escape_char,
        } => {
            if escape_char.is_some() {
                return unsupported("SIMILAR TO ... ESCAPE clause");
            }
            Ok(ast::Expr::SimilarTo {
                expr: Box::new(convert_expr(*expr)?),
                pattern: Box::new(convert_expr(*pattern)?),
                negated,
            })
        },
        sql::Expr::Tuple(items) => {
            let items = items
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Expr::Row(items))
        },
        // `ARRAY[a, b, c]` / `[a, b, c]` array constructor.
        sql::Expr::Array(arr) => {
            let elems = arr
                .elem
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Expr::ArrayLiteral(elems))
        },
        // `base[index]` array element access and `base[i:j]` array slice (B-fn). sqlparser
        // 0.62 models chained subscripts as one access chain (`a[1][2]` = root + two links); fold
        // the chain left-to-right. A `.field` composite access is out of surface.
        sql::Expr::CompoundFieldAccess { root, access_chain } => {
            let mut base = convert_expr(*root)?;
            for access in access_chain {
                base = match access {
                    sql::AccessExpr::Subscript(sql::Subscript::Index { index }) => {
                        ast::Expr::Subscript {
                            base: Box::new(base),
                            index: Box::new(convert_expr(index)?),
                        }
                    },
                    sql::AccessExpr::Subscript(sql::Subscript::Slice {
                        lower_bound,
                        upper_bound,
                        stride,
                    }) => {
                        if stride.is_some() {
                            return unsupported("array slice with a stride `a[i:j:k]`");
                        }
                        ast::Expr::ArraySlice {
                            base: Box::new(base),
                            lower: lower_bound
                                .map(|e| convert_expr(e).map(Box::new))
                                .transpose()?,
                            upper: upper_bound
                                .map(|e| convert_expr(e).map(Box::new))
                                .transpose()?,
                        }
                    },
                    sql::AccessExpr::Dot(field) => {
                        // `(expr).field` — composite field access. The field must be a plain
                        // identifier (`(x).*` and other dotted forms are out of surface).
                        let sql::Expr::Identifier(ident) = field else {
                            return unsupported(&format!("composite field access `.{field}`"));
                        };
                        ast::Expr::FieldAccess {
                            base: Box::new(base),
                            field: fold_ident(&ident),
                        }
                    },
                };
            }
            Ok(base)
        },
        sql::Expr::Case {
            case_token: _,
            end_token: _,
            operand,
            conditions,
            else_result,
        } => convert_case(operand, conditions, else_result),
        // Subqueries. Each subquery is converted through `convert_select`, so a
        // set-operation or otherwise out-of-surface subquery body is rejected there.
        sql::Expr::Subquery(query) => {
            Ok(ast::Expr::ScalarSubquery(Box::new(convert_select(*query)?)))
        },
        sql::Expr::Exists { subquery, negated } => Ok(ast::Expr::Exists {
            negated,
            subquery: Box::new(convert_select(*subquery)?),
        }),
        sql::Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => Ok(ast::Expr::InSubquery {
            expr: Box::new(convert_expr(*expr)?),
            negated,
            subquery: Box::new(convert_select(*subquery)?),
        }),
        sql::Expr::Function(function) => convert_function_call(function),
        // `SUBSTRING(s FROM a FOR b)` / `SUBSTRING(s, a, b)` — both reach here.
        sql::Expr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => convert_substring(*expr, substring_from, substring_for),
        // `TRIM([BOTH|LEADING|TRAILING] [chars] FROM s)` / `TRIM(s, chars)`.
        sql::Expr::Trim {
            expr,
            trim_where,
            trim_what,
            trim_characters,
        } => convert_trim(*expr, trim_where, trim_what, trim_characters),
        // `POSITION(sub IN s)` — 1-based index of `sub` in `s`.
        sql::Expr::Position { expr, r#in } => Ok(ast::Expr::ScalarFunction {
            func: ast::ScalarFunc::Position,
            args: vec![convert_expr(*expr)?, convert_expr(*r#in)?],
        }),
        // `OVERLAY(s PLACING r FROM start [FOR len])` — replace a slice of `s` with `r`.
        sql::Expr::Overlay {
            expr,
            overlay_what,
            overlay_from,
            overlay_for,
        } => {
            let mut args = vec![
                convert_expr(*expr)?,
                convert_expr(*overlay_what)?,
                convert_expr(*overlay_from)?,
            ];
            if let Some(for_len) = overlay_for {
                args.push(convert_expr(*for_len)?);
            }
            Ok(ast::Expr::ScalarFunction {
                func: ast::ScalarFunc::Overlay,
                args,
            })
        },
        // `CEIL(x)` / `FLOOR(x)` — sqlparser models these as dedicated nodes (the `x TO field` and
        // `x, scale` forms are not supported).
        sql::Expr::Ceil { expr, field } => convert_ceil_floor(*expr, &field, ast::ScalarFunc::Ceil),
        sql::Expr::Floor { expr, field } => {
            convert_ceil_floor(*expr, &field, ast::ScalarFunc::Floor)
        },
        // `EXTRACT(field FROM source)`. The field keyword is normalised to a lowercase
        // text-literal first argument; the analyzer validates it against the supported set.
        sql::Expr::Extract { field, expr, .. } => Ok(ast::Expr::ScalarFunction {
            func: ast::ScalarFunc::Extract,
            args: vec![
                ast::Expr::Literal(ast::Value::Text(field.to_string().to_lowercase())),
                convert_expr(*expr)?,
            ],
        }),
        sql::Expr::Cast {
            kind,
            expr,
            data_type,
            ..
        } => {
            // `TRY_CAST`/`SAFE_CAST` yield NULL on a failed conversion; `CAST(x AS t)` and the
            // postfix `x::t` error instead. All four share one node, distinguished by `try_cast`.
            let try_cast = matches!(kind, sql::CastKind::TryCast | sql::CastKind::SafeCast);
            // `x::varchar(n)` / `x::char(n)` truncates to `n` characters (standard explicit-cast
            // semantics) — desugar to `substring(cast(x as text), 1, n)`. For `CHAR(n)`/`CHARACTER(n)`
            // (bpchar), trailing blanks are insignificant, so the result is additionally
            // right-trimmed of spaces — its canonical form — making `length()` and equality match the
            // blank-padded semantics (`length('ab '::char(5))` = 2; `'x  '::char(5) = 'x'::char(5)`).
            // A plain `VARCHAR`/`CHAR`/`TEXT` without a length is left as an ordinary text cast.
            if let Some((n, blank_padded)) = char_cast_limit(&data_type) {
                let inner_text = ast::Expr::Cast {
                    expr: Box::new(convert_expr(*expr)?),
                    target: ColumnType::Text,
                    try_cast,
                };
                let truncated = ast::Expr::ScalarFunction {
                    func: ast::ScalarFunc::Substring,
                    args: vec![
                        inner_text,
                        ast::Expr::Literal(ast::Value::Int(1)),
                        ast::Expr::Literal(ast::Value::Int(i64::from(n))),
                    ],
                };
                return Ok(if blank_padded {
                    ast::Expr::ScalarFunction {
                        func: ast::ScalarFunc::RTrim,
                        args: vec![
                            truncated,
                            ast::Expr::Literal(ast::Value::Text(" ".to_owned())),
                        ],
                    }
                } else {
                    truncated
                });
            }
            // A cast target that is not a built-in type but a bare user-defined type name
            // (e.g. a composite type `x::addr`) defers to the analyzer via `CastNamed`; a genuinely
            // unknown type still surfaces the original "unsupported type" error.
            match convert_data_type(&data_type) {
                Ok(target) => Ok(ast::Expr::Cast {
                    expr: Box::new(convert_expr(*expr)?),
                    target,
                    try_cast,
                }),
                Err(e) => match deferred_udt_name(&data_type) {
                    Some(type_name) => Ok(ast::Expr::CastNamed {
                        expr: Box::new(convert_expr(*expr)?),
                        type_name,
                        try_cast,
                    }),
                    None => Err(e),
                },
            }
        },
        // Quantified comparison `x <op> ANY(...)` / `x <op> ALL(...)` over an `ARRAY[...]`
        // literal. Desugared into an OR / AND chain of element comparisons so it reuses the existing
        // binary-op type-checking and three-valued logic. (sqlparser only parses a subquery operand
        // with double parens; the runtime-array and subquery forms remain a follow-up.)
        sql::Expr::AnyOp {
            left,
            compare_op,
            right,
            // Whether the query spelled `SOME` instead of `ANY` — synonyms, same semantics.
            is_some: _,
        } => desugar_quantified(*left, compare_op, *right, ast::BinaryOp::Or),
        sql::Expr::AllOp {
            left,
            compare_op,
            right,
        } => desugar_quantified(*left, compare_op, *right, ast::BinaryOp::And),
        other => unsupported(&format!("expression `{other}`")),
    }
}

/// Desugar `left <op> ANY/ALL(ARRAY[e0, e1, ...])` into `(left op e0) <combine> (left op e1) ...`,
/// where `combine` is `OR` for `ANY` and `AND` for `ALL`. An empty array yields the identity:
/// `ANY` → `FALSE`, `ALL` → `TRUE`. Only an `ARRAY[...]` literal operand is supported.
fn desugar_quantified(
    left: sql::Expr,
    compare_op: sql::BinaryOperator,
    right: sql::Expr,
    combine: ast::BinaryOp,
) -> Result<ast::Expr, Error> {
    // Subquery operand: `x = ANY((SELECT ...))` is `x IN (...)` and `x <> ALL((SELECT ...))`
    // is `x NOT IN (...)` — both map onto the existing IN-subquery (semi/anti-join) machinery. Other
    // operator/quantifier combinations need a general quantified-subquery plan (a follow-up).
    if let sql::Expr::Subquery(query) = right {
        // `= ANY` / `<> ALL` map onto the optimized IN / NOT IN (semi/anti-join) path.
        match (combine, &compare_op) {
            (ast::BinaryOp::Or, sql::BinaryOperator::Eq) => {
                return Ok(ast::Expr::InSubquery {
                    expr: Box::new(convert_expr(left)?),
                    negated: false,
                    subquery: Box::new(convert_select(*query)?),
                });
            },
            (ast::BinaryOp::And, sql::BinaryOperator::NotEq) => {
                return Ok(ast::Expr::InSubquery {
                    expr: Box::new(convert_expr(left)?),
                    negated: true,
                    subquery: Box::new(convert_select(*query)?),
                });
            },
            _ => {},
        }
        // Every other operator/quantifier becomes a general quantified comparison.
        return Ok(ast::Expr::QuantifiedComparison {
            expr: Box::new(convert_expr(left)?),
            op: convert_binary_op(compare_op)?,
            all: combine == ast::BinaryOp::And,
            subquery: Box::new(convert_select(*query)?),
        });
    }
    let arr = match right {
        sql::Expr::Array(arr) => arr,
        // A runtime array operand (a column / expression, not an `ARRAY[...]` literal) cannot be
        // desugared to a static comparison chain — the executor iterates the array value at run time.
        other => {
            return Ok(ast::Expr::QuantifiedArray {
                expr: Box::new(convert_expr(left)?),
                op: convert_binary_op(compare_op)?,
                all: combine == ast::BinaryOp::And,
                array: Box::new(convert_expr(other)?),
            });
        },
    };
    let op = convert_binary_op(compare_op)?;
    let left = convert_expr(left)?;
    let mut chain: Option<ast::Expr> = None;
    for elem in arr.elem {
        let term = ast::Expr::Binary {
            left: Box::new(left.clone()),
            op,
            right: Box::new(convert_expr(elem)?),
        };
        chain = Some(match chain {
            None => term,
            Some(acc) => ast::Expr::Binary {
                left: Box::new(acc),
                op: combine,
                right: Box::new(term),
            },
        });
    }
    // Empty array: ANY over nothing is FALSE, ALL over nothing is TRUE.
    Ok(
        chain
            .unwrap_or_else(|| ast::Expr::Literal(ast::Value::Bool(combine == ast::BinaryOp::And))),
    )
}

pub(super) fn convert_case(
    operand: Option<Box<sql::Expr>>,
    conditions: Vec<sql::CaseWhen>,
    else_result: Option<Box<sql::Expr>>,
) -> Result<ast::Expr, Error> {
    if conditions.is_empty() {
        return unsupported("CASE with no WHEN branches");
    }
    let operand = match operand {
        Some(e) => Some(Box::new(convert_expr(*e)?)),
        None => None,
    };
    let mut branches = Vec::with_capacity(conditions.len());
    for when in conditions {
        branches.push(ast::CaseBranch {
            when: convert_expr(when.condition)?,
            then: convert_expr(when.result)?,
        });
    }
    let default = match else_result {
        Some(e) => Some(Box::new(convert_expr(*e)?)),
        None => None,
    };
    Ok(ast::Expr::Case {
        operand,
        branches,
        default,
    })
}

/// Map a folded function name to its [`ast::WindowFunc`], if it is a recognised window
/// function. Returns `None` for unknown names (caller turns that into `Unsupported`).
pub(super) fn window_func_by_name(name: &str) -> Option<ast::WindowFunc> {
    use ast::AggregateFunc::{Avg, Count, Max, Min, Sum};
    use ast::WindowFunc::{
        CumeDist, DenseRank, FirstValue, Lag, LastValue, Lead, NthValue, Ntile, PercentRank, Rank,
        RowNumber,
    };
    match name {
        "row_number" => Some(RowNumber),
        "rank" => Some(Rank),
        "dense_rank" => Some(DenseRank),
        "ntile" => Some(Ntile),
        "cume_dist" => Some(CumeDist),
        "percent_rank" => Some(PercentRank),
        "lag" => Some(Lag),
        "lead" => Some(Lead),
        "first_value" => Some(FirstValue),
        "last_value" => Some(LastValue),
        "nth_value" => Some(NthValue),
        "count" => Some(ast::WindowFunc::Aggregate(Count)),
        "sum" => Some(ast::WindowFunc::Aggregate(Sum)),
        "avg" => Some(ast::WindowFunc::Aggregate(Avg)),
        "min" => Some(ast::WindowFunc::Aggregate(Min)),
        "max" => Some(ast::WindowFunc::Aggregate(Max)),
        _ => None,
    }
}

/// Convert a window function call (has an `OVER` clause) into [`ast::Expr::WindowFunction`].
/// Frame bounds (`ROWS`/`RANGE`/`GROUPS`) are not yet modelled — they are rejected
/// with [`Error::Unsupported`] pointing at
pub(super) fn convert_window_function(function: sql::Function) -> Result<ast::Expr, Error> {
    // Rejected one clause at a time, not as one arm: `FILTER` is supported on an aggregate used
    // `OVER` a window, while these three are not supported at all. Lumping them together would make
    // the surface unreadable from the error message alone.
    if function.null_treatment.is_some() {
        return unsupported("IGNORE NULLS / RESPECT NULLS in window function");
    }
    if !function.within_group.is_empty() {
        return unsupported("WITHIN GROUP in window function");
    }
    if !matches!(function.parameters, sql::FunctionArguments::None) {
        return unsupported("parameterized aggregate in window function");
    }
    // The window spec, plus a frame `EXCLUDE` mode captured by the preprocessor (re-attached to the
    // frame below). `exclude_mode` is `None` when this window had no `EXCLUDE`; `Some(mode)` demands
    // a frame (rejected loudly if absent).
    let (spec, exclude_mode) = resolve_over_spec(function.over)?;
    // window_frame is handled below; no early rejection here.
    if spec.window_name.is_some() {
        return unsupported("named window base in OVER clause is not yet supported");
    }
    let name_ident = function
        .name
        .0
        .last()
        .ok_or_else(|| Error::InvalidStatement("empty window function name".to_owned()))?;
    let name = fold_part(name_ident)?;
    let func = window_func_by_name(&name)
        .ok_or_else(|| Error::FunctionArgs(format!("no window function named `{name}`")))?;
    let args = match function.args {
        sql::FunctionArguments::None => Vec::new(),
        sql::FunctionArguments::List(list) => {
            if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
                return unsupported("DISTINCT/ALL/clauses inside window function");
            }
            let mut out = Vec::with_capacity(list.args.len());
            for arg in list.args {
                match arg {
                    // `COUNT(*)` in a window context: treat `*` as no argument (same as plain
                    // `COUNT(*)`), so `Aggregate(Count)` with empty args is consistent.
                    sql::FunctionArg::Unnamed(
                        sql::FunctionArgExpr::Wildcard | sql::FunctionArgExpr::QualifiedWildcard(_),
                    ) => {},
                    sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => {
                        out.push(convert_expr(e)?);
                    },
                    _ => return unsupported("unsupported window function argument form"),
                }
            }
            out
        },
        sql::FunctionArguments::Subquery(_) => {
            return unsupported("subquery in window function");
        },
    };
    let partition = spec
        .partition_by
        .into_iter()
        .map(convert_expr)
        .collect::<Result<Vec<_>, _>>()?;
    let order = spec
        .order_by
        .into_iter()
        .map(|obe| {
            if obe.with_fill.is_some() {
                return unsupported("ORDER BY ... WITH FILL in window function");
            }
            let nulls = match obe.options.nulls_first {
                None => ast::NullOrdering::Default,
                Some(true) => ast::NullOrdering::First,
                Some(false) => ast::NullOrdering::Last,
            };
            Ok(ast::OrderByItem {
                expr: convert_expr(obe.expr)?,
                ascending: obe.options.asc.unwrap_or(true),
                nulls,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut frame = spec.window_frame.map(convert_window_frame).transpose()?;
    if let Some(mode) = exclude_mode {
        attach_frame_exclude(frame.as_mut(), mode, &func)?;
    }
    // `FILTER (WHERE pred)` restricts what the aggregate reads. The ranking, navigation and
    // distribution functions aggregate nothing, so a filter on one of them has no meaning to give
    // — reject rather than accept-and-ignore, which would answer a question nobody asked.
    let filter = match function.filter {
        None => None,
        Some(pred) => {
            if !matches!(func, ast::WindowFunc::Aggregate(_)) {
                return unsupported("FILTER on a non-aggregate window function");
            }
            Some(Box::new(convert_expr(*pred)?))
        },
    };
    Ok(ast::Expr::WindowFunction(Box::new(ast::WindowFunction {
        func,
        args,
        partition,
        order,
        frame,
        filter,
    })))
}

/// Resolve an `OVER (...)` / `OVER w` clause to its window spec plus any captured frame `EXCLUDE`
/// mode. A synthetic `__nusadb_excl_win_<n>` reference minted by the parser's window-exclude
/// preprocessor resolves to the stored spec and carries the exclusion mode; an ordinary named
/// window resolves against the enclosing SELECT's `WINDOW` clause; an inline spec carries no mode.
fn resolve_over_spec(
    over: Option<sql::WindowType>,
) -> Result<(sql::WindowSpec, Option<ast::WindowExclude>), Error> {
    match over {
        Some(sql::WindowType::WindowSpec(s)) => Ok((s, None)),
        Some(sql::WindowType::NamedWindow(name)) => {
            let key = fold_ident(&name);
            if let Some((spec, mode)) = EXCLUDE_WINDOWS.with(|c| c.borrow_mut().remove(&key)) {
                Ok((spec, Some(mode)))
            } else {
                let spec = NAMED_WINDOWS
                    .with(|c| c.borrow().get(&key).cloned())
                    .ok_or_else(|| {
                        Error::InvalidStatement(format!(
                            "window `{key}` is not defined in a WINDOW clause"
                        ))
                    })?;
                Ok((spec, None))
            }
        },
        None => unreachable!("convert_window_function called without OVER clause"),
    }
}

/// Convert a sqlparser [`sql::WindowFrame`] into an [`ast::WindowFrame`].
pub(super) fn convert_window_frame(f: sql::WindowFrame) -> Result<ast::WindowFrame, Error> {
    let units = match f.units {
        sql::WindowFrameUnits::Rows => ast::WindowFrameUnits::Rows,
        sql::WindowFrameUnits::Range => ast::WindowFrameUnits::Range,
        sql::WindowFrameUnits::Groups => ast::WindowFrameUnits::Groups,
    };
    let start = convert_window_frame_bound(f.start_bound)?;
    let end = f.end_bound.map(convert_window_frame_bound).transpose()?;
    // `EXCLUDE` is not modelled by sqlparser; it is stripped and captured by the parser's
    // window-exclude preprocessor and re-attached in `convert_window_function`. Default here.
    Ok(ast::WindowFrame {
        units,
        start,
        end,
        exclude: ast::WindowExclude::NoOthers,
    })
}

/// Attach a frame `EXCLUDE` mode (captured by the parser's window-exclude preprocessor) to `frame`.
///
/// `EXCLUDE` requires a frame clause (the reference engine treats `OVER (... EXCLUDE ...)` without a
/// frame as a syntax error), so a captured mode with no frame is rejected loudly rather than
/// silently dropped (which would compute the wrong aggregate). A real exclusion (anything but
/// `NO OTHERS`) only has a defined meaning for a frame-reading aggregate — the ranking / navigation
/// / distribution functions are rejected rather than silently ignoring the clause.
fn attach_frame_exclude(
    frame: Option<&mut ast::WindowFrame>,
    mode: ast::WindowExclude,
    func: &ast::WindowFunc,
) -> Result<(), Error> {
    if mode != ast::WindowExclude::NoOthers && !matches!(func, ast::WindowFunc::Aggregate(_)) {
        return unsupported("EXCLUDE on a non-aggregate window function");
    }
    match frame {
        Some(f) => {
            f.exclude = mode;
            Ok(())
        },
        None => unsupported("EXCLUDE requires a window frame clause (ROWS/RANGE/GROUPS ...)"),
    }
}

/// Convert a single window-frame boundary.
pub(super) fn convert_window_frame_bound(
    bound: sql::WindowFrameBound,
) -> Result<ast::WindowFrameBound, Error> {
    match bound {
        sql::WindowFrameBound::Preceding(None) => Ok(ast::WindowFrameBound::UnboundedPreceding),
        sql::WindowFrameBound::Preceding(Some(expr)) => Ok(ast::WindowFrameBound::Preceding(
            Box::new(convert_expr(*expr)?),
        )),
        sql::WindowFrameBound::CurrentRow => Ok(ast::WindowFrameBound::CurrentRow),
        sql::WindowFrameBound::Following(None) => Ok(ast::WindowFrameBound::UnboundedFollowing),
        sql::WindowFrameBound::Following(Some(expr)) => Ok(ast::WindowFrameBound::Following(
            Box::new(convert_expr(*expr)?),
        )),
    }
}

/// Convert an ordered-set aggregate `func(args) WITHIN GROUP (ORDER BY ...)` into
/// [`ast::Expr::WithinGroup`], e.g. `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`.
///
/// `OVER` is already routed away by the caller. `FILTER`, `NULLS`, and parametric clauses
/// combined with `WITHIN GROUP` are rejected. The direct args (the percentile fraction) and the
/// `WITHIN GROUP` sort keys are both captured.
fn convert_within_group(function: sql::Function) -> Result<ast::Expr, Error> {
    if function.filter.is_some()
        || function.null_treatment.is_some()
        || !matches!(function.parameters, sql::FunctionArguments::None)
    {
        return unsupported("WITHIN GROUP combined with FILTER / NULLS / parametric clause");
    }
    let name_ident = function
        .name
        .0
        .last()
        .ok_or_else(|| Error::InvalidStatement("empty function name".to_owned()))?;
    let func = fold_part(name_ident)?;
    let args = match function.args {
        sql::FunctionArguments::None => Vec::new(),
        sql::FunctionArguments::List(list) => {
            if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
                return unsupported("WITHIN GROUP function with DISTINCT/ALL/clauses");
            }
            let mut out = Vec::with_capacity(list.args.len());
            for arg in list.args {
                match arg {
                    sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => {
                        out.push(convert_expr(e)?);
                    },
                    _ => return unsupported("unsupported WITHIN GROUP argument form"),
                }
            }
            out
        },
        sql::FunctionArguments::Subquery(_) => {
            return unsupported("subquery in WITHIN GROUP function");
        },
    };
    let order_by = function
        .within_group
        .into_iter()
        .map(|obe| {
            if obe.with_fill.is_some() {
                return unsupported("ORDER BY ... WITH FILL in WITHIN GROUP");
            }
            let nulls = match obe.options.nulls_first {
                None => ast::NullOrdering::Default,
                Some(true) => ast::NullOrdering::First,
                Some(false) => ast::NullOrdering::Last,
            };
            Ok(ast::OrderByItem {
                expr: convert_expr(obe.expr)?,
                ascending: obe.options.asc.unwrap_or(true),
                nulls,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ast::Expr::WithinGroup(Box::new(ast::WithinGroup {
        func,
        args,
        order_by,
    })))
}

/// Lower a function call. NusaDB recognizes a curated set of names: the five
/// aggregates (`count`/`sum`/`avg`/`min`/`max`) and `coalesce`; everything
/// else is rejected.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-name dispatch over the recognised built-ins; grows with the surface"
)]
pub(super) fn convert_function_call(function: sql::Function) -> Result<ast::Expr, Error> {
    // Window functions (OVER clause) are handled separately.
    if function.over.is_some() {
        return convert_window_function(function);
    }
    // `func(args) WITHIN GROUP (ORDER BY ...)` — ordered-set aggregate.
    if !function.within_group.is_empty() {
        return convert_within_group(function);
    }
    if function.null_treatment.is_some()
        || !matches!(function.parameters, sql::FunctionArguments::None)
    {
        return unsupported("function with NULLS / parametric clause");
    }
    let name_ident = function
        .name
        .0
        .last()
        .ok_or_else(|| Error::InvalidStatement("empty function name".to_owned()))?;
    let name = fold_part(name_ident)?;

    let mut arg_list = match function.args {
        sql::FunctionArguments::List(list) => list,
        // The niladic clock built-ins (`CURRENT_TIMESTAMP`, `CURRENT_DATE`, `CURRENT_TIME`)
        // and session-user built-ins (`CURRENT_USER`, `SESSION_USER`, `USER`) arrive in their
        // bare keyword form with no argument list; map those here. Any other argument-less call is
        // unsupported.
        sql::FunctionArguments::None => {
            return scalar_func_by_name(&name)
                .filter(|f| f.is_clock() || f.is_session_user())
                .map_or_else(
                    || unsupported("function call with non-list arguments"),
                    |func| {
                        Ok(ast::Expr::ScalarFunction {
                            func,
                            args: Vec::new(),
                        })
                    },
                );
        },
        sql::FunctionArguments::Subquery(_) => {
            return unsupported("function call with non-list arguments");
        },
    };
    // An aggregate may carry an `ORDER BY` clause (`array_agg(x ORDER BY y)`); any other argument
    // clause, and `ORDER BY` on a non-aggregate, is unsupported. The clause is peeled off here so the
    // generic "no clauses" gate below still rejects it for scalar functions.
    let is_aggregate = aggregate_func(&name).is_some();
    let mut agg_order_by = Vec::new();
    if is_aggregate {
        let mut clauses = std::mem::take(&mut arg_list.clauses);
        clauses.retain(|c| match c {
            sql::FunctionArgumentClause::OrderBy(items) => {
                agg_order_by.clone_from(items);
                false
            },
            _ => true,
        });
        if !clauses.is_empty() {
            return unsupported("aggregate argument clause other than ORDER BY");
        }
    }
    if !arg_list.clauses.is_empty() {
        return unsupported("function with argument clauses");
    }

    // Aggregates take a single argument that may be `*` (COUNT only), so they
    // are handled before the scalar-argument loop below which rejects `*`.
    // FILTER (WHERE pred) and DISTINCT are only valid on aggregates.
    if let Some(func) = aggregate_func(&name) {
        let distinct = match arg_list.duplicate_treatment {
            None => false,
            Some(sql::DuplicateTreatment::Distinct) => true,
            Some(sql::DuplicateTreatment::All) => {
                return unsupported("COUNT(ALL ...) — use COUNT(expr) for non-distinct counting");
            },
        };
        let filter = function
            .filter
            .map(|f| convert_expr(*f).map(Box::new))
            .transpose()?;
        let order_by = convert_order_by_items(agg_order_by)?;
        return convert_aggregate(func, arg_list.args, distinct, filter, order_by);
    }
    if function.filter.is_some() {
        return unsupported("FILTER (WHERE ...) is only valid on aggregate functions");
    }
    if arg_list.duplicate_treatment.is_some() {
        return unsupported("DISTINCT / ALL is only valid on aggregate functions");
    }

    let mut args = Vec::with_capacity(arg_list.args.len());
    for arg in arg_list.args {
        let expr = match arg {
            sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => e,
            sql::FunctionArg::Named { .. } | sql::FunctionArg::ExprNamed { .. } => {
                return unsupported("named function argument");
            },
            sql::FunctionArg::Unnamed(
                sql::FunctionArgExpr::Wildcard
                | sql::FunctionArgExpr::QualifiedWildcard(_)
                | sql::FunctionArgExpr::WildcardWithOptions(_),
            ) => return unsupported("`*` is not valid in this scalar function call"),
        };
        args.push(convert_expr(expr)?);
    }
    match name.as_str() {
        "coalesce" => {
            if args.is_empty() {
                return unsupported("COALESCE with no arguments");
            }
            Ok(ast::Expr::Coalesce(args))
        },
        // NVL(a, b) / IFNULL(a, b) are the two-argument aliases of COALESCE.
        "nvl" | "ifnull" => {
            if args.len() != 2 {
                return unsupported("NVL / IFNULL takes exactly two arguments");
            }
            Ok(ast::Expr::Coalesce(args))
        },
        "encrypt" | "decrypt" => convert_crypto_call(&name, args),
        // `ROW(a, b, ...)` keyword form of the row-value constructor; the bare `(a, b)`
        // tuple form is handled directly by the `sql::Expr::Tuple` arm in `convert_expr`.
        "row" => Ok(ast::Expr::Row(args)),
        // Set-returning functions: one input row → N output rows. The analyzer enforces
        // that they appear only at the top of a SELECT-list item.
        "unnest" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::Unnest,
            args,
        }),
        "json_array_elements" | "jsonb_array_elements" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonArrayElements,
            args,
        }),
        "jsonb_path_query" | "json_path_query" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonPathQuery,
            args,
        }),
        "generate_series" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::GenerateSeries,
            args,
        }),
        "jsonb_object_keys" | "json_object_keys" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonObjectKeys,
            args,
        }),
        "regexp_split_to_table" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::RegexpSplitToTable,
            args,
        }),
        // `regexp_matches` is set-returning (one row per match with the `g` flag), unlike the scalar
        // `regexp_match` which returns only the first match's groups as a `TEXT[]`.
        "regexp_matches" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::RegexpMatches,
            args,
        }),
        "string_to_table" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::StringToTable,
            args,
        }),
        "jsonb_each" | "json_each" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonEach,
            args,
        }),
        "jsonb_each_text" | "json_each_text" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonEachText,
            args,
        }),
        "jsonb_array_elements_text" | "json_array_elements_text" => Ok(ast::Expr::SetReturning {
            func: ast::SetReturningFunc::JsonArrayElementsText,
            args,
        }),
        // Scalar string built-ins reachable through the ordinary call syntax. Arity and
        // argument types are validated by the analyzer. An unrecognised name is kept as a generic
        // `FunctionCall` for the analyzer to resolve against the user-defined-function registry
        // — and to reject as an unknown function if no UDF is registered.
        _ => match scalar_func_by_name(&name) {
            Some(func) => Ok(ast::Expr::ScalarFunction { func, args }),
            None => Ok(ast::Expr::FunctionCall { name, args }),
        },
    }
}

/// Resolve the `LIKE`/`ILIKE` escape character: the explicit `ESCAPE 'c'` if given, else the
/// default backslash. `None` means no escape character (an explicit `ESCAPE ''`); a multi-character
/// escape is rejected.
fn like_escape_char(escape_char: Option<sql::ValueWithSpan>) -> Result<Option<char>, Error> {
    // With no `ESCAPE` clause the default escape character is backslash, so `\%` and `\_` match a
    // literal `%`/`_` (SQL-standard).
    let Some(v) = escape_char else {
        return Ok(Some('\\'));
    };
    let s = match v.value {
        sql::Value::SingleQuotedString(s)
        | sql::Value::DoubleQuotedString(s)
        | sql::Value::EscapedStringLiteral(s)
        | sql::Value::UnicodeStringLiteral(s) => s,
        other => return unsupported(&format!("LIKE ESCAPE value `{other}`")),
    };
    let mut chars = s.chars();
    let ch = chars.next();
    if ch.is_some() && chars.next().is_some() {
        return unsupported("LIKE ESCAPE must be a single character");
    }
    // An explicit empty `ESCAPE ''` disables the escape character entirely.
    Ok(ch)
}

/// Map a folded function name to its [`ast::ScalarFunc`] when it is a recognised scalar built-in
/// reachable through the ordinary `f(args)` call syntax. Returns `None` for unknown names so
/// the caller rejects them as `Unsupported`. Special syntactic forms (`SUBSTRING ... FROM ... FOR`,
/// `TRIM ... FROM`, `POSITION ... IN`) are handled by their dedicated `convert_expr` arms instead.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-builtin name table; splitting it would scatter the mapping"
)]
fn scalar_func_by_name(name: &str) -> Option<ast::ScalarFunc> {
    use ast::ScalarFunc as F;
    Some(match name {
        "length" | "char_length" | "character_length" => F::Length,
        "octet_length" => F::OctetLength,
        "bit_length" => F::BitLength,
        // `GROUPING(...)` (a.k.a. `GROUPING_ID`) — super-aggregate indicator, resolved by the
        // analyzer against the query's grouping sets.
        "grouping" | "grouping_id" => F::Grouping,
        "upper" => F::Upper,
        "lower" => F::Lower,
        "substr" | "substring" => F::Substring,
        "replace" => F::Replace,
        "lpad" => F::Lpad,
        "rpad" => F::Rpad,
        "ltrim" => F::LTrim,
        "rtrim" => F::RTrim,
        "btrim" | "trim" => F::BTrim,
        "concat" => F::Concat,
        "concat_ws" => F::ConcatWs,
        "left" => F::Left,
        "right" => F::Right,
        "split_part" => F::SplitPart,
        "reverse" => F::Reverse,
        "starts_with" => F::StartsWith,
        "ascii" => F::Ascii,
        "chr" => F::Chr,
        "initcap" => F::Initcap,
        "repeat" => F::Repeat,
        "strpos" => F::Strpos,
        "translate" => F::Translate,
        "regexp_replace" => F::RegexpReplace,
        // `regexp_match` (scalar, first match's groups as `TEXT[]`); the set-returning `regexp_matches`
        // is handled above as a `SetReturningFunc`.
        "regexp_match" => F::RegexpMatch,
        "regexp_like" => F::RegexpLike,
        "regexp_count" => F::RegexpCount,
        "regexp_instr" => F::RegexpInstr,
        "regexp_substr" => F::RegexpSubstr,
        "regexp_split_to_array" => F::RegexpSplitToArray,
        // Niladic clock built-ins, reachable through the parenthesised call form `NOW()` /
        // `CURRENT_TIMESTAMP()` (the bare keyword form is mapped in `convert_function_call`).
        "now" => F::Now,
        "statement_timestamp" => F::StatementTimestamp,
        "localtimestamp" => F::LocalTimestamp,
        "current_timestamp" => F::CurrentTimestamp,
        "current_date" => F::CurrentDate,
        "current_time" => F::CurrentTime,
        // Session-user built-ins. The bare keyword forms `CURRENT_USER` / `SESSION_USER` /
        // `USER` map here via `convert_function_call`; `USER` is a synonym for `CURRENT_USER`.
        "current_user" | "user" => F::CurrentUser,
        "session_user" => F::SessionUser,
        // `current_setting(name)` reads a session `SET` variable.
        "current_setting" => F::CurrentSetting,
        // Date/time functions reachable through the ordinary call form. `EXTRACT` has its
        // own special syntax (`EXTRACT(field FROM source)`) handled in `convert_expr`.
        "date_trunc" => F::DateTrunc,
        // `date_part(field, source)` is the function-call spelling of `EXTRACT(field FROM source)`;
        // the field arrives as a string-literal argument that the analyzer lowercases and validates.
        "date_part" => F::Extract,
        "age" => F::Age,
        "to_char" => F::ToChar,
        "to_date" => F::ToDate,
        "to_timestamp" => F::ToTimestamp,
        "to_number" => F::ToNumber,
        "make_date" => F::MakeDate,
        "make_time" => F::MakeTime,
        "make_timestamp" => F::MakeTimestamp,
        "make_timestamptz" => F::MakeTimestamptz,
        "make_interval" => F::MakeInterval,
        "justify_days" => F::JustifyDays,
        "justify_hours" => F::JustifyHours,
        "justify_interval" => F::JustifyInterval,
        "scale" => F::Scale,
        "min_scale" => F::MinScale,
        "trim_scale" => F::TrimScale,
        "isfinite" => F::IsFinite,
        "encode" => F::Encode,
        "decode" => F::Decode,
        "convert_to" => F::ConvertTo,
        "convert_from" => F::ConvertFrom,
        "date_bin" => F::DateBin,
        // Math.
        "abs" => F::Abs,
        "round" => F::Round,
        "ceil" | "ceiling" => F::Ceil,
        "floor" => F::Floor,
        "sign" => F::Sign,
        "mod" => F::Mod,
        "power" | "pow" => F::Power,
        "sqrt" => F::Sqrt,
        "ln" => F::Ln,
        "log" => F::Log,
        "log10" => F::Log10,
        "exp" => F::Exp,
        "sin" => F::Sin,
        "cos" => F::Cos,
        "tan" => F::Tan,
        "asin" => F::Asin,
        "acos" => F::Acos,
        "atan" => F::Atan,
        "atan2" => F::Atan2,
        "cot" => F::Cot,
        "cbrt" => F::Cbrt,
        "sinh" => F::Sinh,
        "cosh" => F::Cosh,
        "tanh" => F::Tanh,
        "asinh" => F::Asinh,
        "acosh" => F::Acosh,
        "atanh" => F::Atanh,
        "gcd" => F::Gcd,
        "lcm" => F::Lcm,
        "div" => F::Div,
        "factorial" => F::Factorial,
        "bit_count" => F::BitCount,
        "to_hex" => F::ToHex,
        "width_bucket" => F::WidthBucket,
        "num_nonnulls" => F::NumNonNulls,
        "num_nulls" => F::NumNulls,
        "sha256" => F::Sha256,
        "sha512" => F::Sha512,
        "md5" => F::Md5,
        "quote_literal" => F::QuoteLiteral,
        "quote_nullable" => F::QuoteNullable,
        "quote_ident" => F::QuoteIdent,
        "format" => F::Format,
        "degrees" => F::Degrees,
        "radians" => F::Radians,
        "pi" => F::Pi,
        "trunc" => F::Trunc,
        // Random.
        "random" => F::Random,
        "setseed" => F::Setseed,
        // Conditional.
        "nullif" => F::Nullif,
        "greatest" => F::Greatest,
        "least" => F::Least,
        "cardinality" => F::Cardinality,
        "array_length" => F::ArrayLength,
        "array_fill" => F::ArrayFill,
        "array_lower" => F::ArrayLower,
        "array_upper" => F::ArrayUpper,
        "array_dims" => F::ArrayDims,
        "array_to_string" => F::ArrayToString,
        "string_to_array" => F::StringToArray,
        "array_append" => F::ArrayAppend,
        "array_prepend" => F::ArrayPrepend,
        "array_cat" => F::ArrayCat,
        "array_position" => F::ArrayPosition,
        "array_remove" => F::ArrayRemove,
        "array_replace" => F::ArrayReplace,
        "array_positions" => F::ArrayPositions,
        "array_ndims" => F::ArrayNdims,
        // Vector distance functions.
        "l2_distance" => F::L2Distance,
        "cosine_distance" => F::CosineDistance,
        "inner_product" => F::InnerProduct,
        "l1_distance" => F::L1Distance,
        "vector_dims" => F::VectorDims,
        "vector_norm" => F::VectorNorm,
        "host" => F::InetHost,
        "masklen" => F::InetMasklen,
        "family" => F::InetFamily,
        "network" => F::InetNetwork,
        "broadcast" => F::InetBroadcast,
        "set_masklen" => F::InetSetMasklen,
        "get_bit" => F::BitGetBit,
        "set_bit" => F::BitSetBit,
        // Range accessors. `lower`/`upper` are not here: they share their spelling with the
        // text-folding functions, so the analyzer resolves them by argument type.
        "isempty" => F::RangeIsEmpty,
        "int4range" => F::Int4Range,
        "int8range" => F::Int8Range,
        "numrange" => F::NumRange,
        "daterange" => F::DateRange,
        "tsrange" => F::TsRange,
        "tstzrange" => F::TsTzRange,
        "lower_inc" => F::RangeLowerInc,
        "upper_inc" => F::RangeUpperInc,
        "lower_inf" => F::RangeLowerInf,
        "upper_inf" => F::RangeUpperInf,
        // UUID generation: both the modern built-in name and the alternative spelling.
        "gen_random_uuid" | "uuid_generate_v4" => F::UuidGenerateV4,
        // System functions.
        "version" => F::Version,
        // `nusadb_typeof(expr)` — NusaDB's static type-introspection builtin; folded to the static
        // type name at analysis.
        "nusadb_typeof" => F::NusadbTypeof,
        "current_database" => F::CurrentDatabase,
        "current_schema" => F::CurrentSchema,
        // JSON inspection + construction scalars.
        "json_typeof" | "jsonb_typeof" => F::JsonTypeof,
        "json_array_length" | "jsonb_array_length" => F::JsonArrayLength,
        "to_json" | "to_jsonb" => F::ToJson,
        "row_to_json" => F::RowToJson,
        "json_build_object" | "jsonb_build_object" => F::JsonBuildObject,
        "json_object" | "jsonb_object" => F::JsonObject,
        "json_build_array" | "jsonb_build_array" => F::JsonBuildArray,
        "jsonb_set" | "json_set" => F::JsonbSet,
        "jsonb_strip_nulls" | "json_strip_nulls" => F::JsonbStripNulls,
        "jsonb_pretty" | "json_pretty" => F::JsonbPretty,
        "jsonb_path_exists" | "json_path_exists" => F::JsonbPathExists,
        "jsonb_insert" | "json_insert" => F::JsonbInsert,
        "jsonb_path_query_first" | "json_path_query_first" => F::JsonbPathQueryFirst,
        "jsonb_path_query_array" | "json_path_query_array" => F::JsonbPathQueryArray,
        "jsonb_exists" | "json_exists" => F::JsonbExists,
        "to_tsvector" => F::ToTsvector,
        "to_tsquery" => F::ToTsquery,
        "plainto_tsquery" => F::PlaintoTsquery,
        "ts_rank" => F::TsRank,
        "ts_rank_cd" => F::TsRankCd,
        "rrf_score" => F::RrfScore,
        _ => return None,
    })
}

/// Lower a `CEIL(x)` / `FLOOR(x)` node into the matching [`ast::ScalarFunc`]. The
/// `CEIL(x TO field)` datetime form and the `CEIL(x, scale)` form are rejected — only the plain
/// single-argument rounding is supported.
fn convert_ceil_floor(
    expr: sql::Expr,
    field: &sql::CeilFloorKind,
    func: ast::ScalarFunc,
) -> Result<ast::Expr, Error> {
    match field {
        sql::CeilFloorKind::DateTimeField(sql::DateTimeField::NoDateTime) => {
            Ok(ast::Expr::ScalarFunction {
                func,
                args: vec![convert_expr(expr)?],
            })
        },
        sql::CeilFloorKind::DateTimeField(_) => unsupported("CEIL/FLOOR ... TO <field>"),
        sql::CeilFloorKind::Scale(_) => unsupported("CEIL/FLOOR with a scale argument"),
    }
}

/// The declared length `n` of a sized character cast target plus whether it is blank-padded —
/// `VARCHAR(n)` / `CHAR(n)` and their spelling variants — or `None` for an unsized character type
/// (`TEXT`, `VARCHAR`/`CHAR` without a length, or `VARCHAR(MAX)`). An explicit cast truncates to `n`
/// characters; for `CHAR(n)` (bpchar) the second tuple field is `true`, so the caller additionally
/// strips trailing blanks (they are insignificant), while `VARCHAR(n)` keeps them.
fn char_cast_limit(ty: &sql::DataType) -> Option<(u32, bool)> {
    use sql::DataType as D;
    // `blank_padded` is `true` for `CHAR(n)`/`CHARACTER(n)` (bpchar — trailing blanks are
    // insignificant) and `false` for the `VARCHAR(n)` family.
    let (len, blank_padded) = match ty {
        D::Char(len) | D::Character(len) => (len.as_ref()?, true),
        D::Varchar(len) | D::CharVarying(len) | D::CharacterVarying(len) | D::Nvarchar(len) => {
            (len.as_ref()?, false)
        },
        _ => return None,
    };
    match len {
        sql::CharacterLength::IntegerLength { length, .. } => {
            Some((u32::try_from(*length).ok()?, blank_padded))
        },
        sql::CharacterLength::Max => None,
    }
}

/// Lower a `SUBSTRING(s FROM start [FOR length])` / `SUBSTRING(s, start [, length])` node into a
/// positional [`ast::ScalarFunc::Substring`] call. A missing start (`SUBSTRING(s FOR n)`)
/// defaults to position 1, per the standard.
fn convert_substring(
    expr: sql::Expr,
    substring_from: Option<Box<sql::Expr>>,
    substring_for: Option<Box<sql::Expr>>,
) -> Result<ast::Expr, Error> {
    let start = match substring_from {
        Some(start) => convert_expr(*start)?,
        None => ast::Expr::Literal(ast::Value::Int(1)),
    };
    let mut args = vec![convert_expr(expr)?, start];
    if let Some(length) = substring_for {
        args.push(convert_expr(*length)?);
    }
    Ok(ast::Expr::ScalarFunction {
        func: ast::ScalarFunc::Substring,
        args,
    })
}

/// Lower a `TRIM([BOTH|LEADING|TRAILING] [what] FROM s)` / `TRIM(s, chars)` node into the matching
/// directional [`ast::ScalarFunc`] (`BTrim`/`LTrim`/`RTrim`) with an optional trim-set argument.
/// Only a single trim-character argument is supported.
fn convert_trim(
    expr: sql::Expr,
    trim_where: Option<sql::TrimWhereField>,
    trim_what: Option<Box<sql::Expr>>,
    trim_characters: Option<Vec<sql::Expr>>,
) -> Result<ast::Expr, Error> {
    let func = match trim_where {
        None | Some(sql::TrimWhereField::Both) => ast::ScalarFunc::BTrim,
        Some(sql::TrimWhereField::Leading) => ast::ScalarFunc::LTrim,
        Some(sql::TrimWhereField::Trailing) => ast::ScalarFunc::RTrim,
    };
    let mut args = vec![convert_expr(expr)?];
    // The trim set arrives either as `what FROM s` (`trim_what`) or the `TRIM(s, chars)` comma form
    // (`trim_characters`); the two are mutually exclusive in practice. Reject more than one.
    match (trim_what, trim_characters) {
        (Some(_), Some(_)) => return unsupported("TRIM with both FROM and comma trim characters"),
        (Some(what), None) => args.push(convert_expr(*what)?),
        (None, Some(mut chars)) => match chars.len() {
            0 => {},
            1 => args.push(convert_expr(chars.remove(0))?),
            _ => return unsupported("TRIM with more than one trim character"),
        },
        (None, None) => {},
    }
    Ok(ast::Expr::ScalarFunction { func, args })
}

/// Lower an `encrypt(value, key)` / `decrypt(value, key)` call; both
/// take exactly two scalar arguments.
pub(super) fn convert_crypto_call(
    name: &str,
    mut args: Vec<ast::Expr>,
) -> Result<ast::Expr, Error> {
    if args.len() != 2 {
        return unsupported(&format!("{name}() takes exactly (value, key)"));
    }
    let key = Box::new(args.remove(1));
    let value = Box::new(args.remove(0));
    Ok(if name == "encrypt" {
        ast::Expr::Encrypt { value, key }
    } else {
        ast::Expr::Decrypt { value, key }
    })
}

/// Map a folded function name to its [`ast::AggregateFunc`], if it names one.
pub(super) fn aggregate_func(name: &str) -> Option<ast::AggregateFunc> {
    use ast::AggregateFunc::{Avg, Count, Max, Min, Sum};
    match name {
        "count" => Some(Count),
        "sum" => Some(Sum),
        "avg" => Some(Avg),
        "min" => Some(Min),
        "max" => Some(Max),
        "array_agg" => Some(ast::AggregateFunc::ArrayAgg),
        // JSON_AGG and JSONB_AGG both collect into a JSON array (our JSON type is binary-backed).
        "jsonb_agg" | "json_agg" => Some(ast::AggregateFunc::JsonAgg),
        // JSON_OBJECT_AGG / JSONB_OBJECT_AGG aggregate (key, value) pairs into a JSON object.
        "json_object_agg" | "jsonb_object_agg" => Some(ast::AggregateFunc::JsonObjectAgg),
        // EVERY is the SQL-standard alias for BOOL_AND.
        "bool_and" | "every" => Some(ast::AggregateFunc::BoolAnd),
        "bool_or" => Some(ast::AggregateFunc::BoolOr),
        "stddev" | "stddev_samp" => Some(ast::AggregateFunc::Stddev),
        "variance" | "var_samp" => Some(ast::AggregateFunc::Variance),
        "stddev_pop" => Some(ast::AggregateFunc::StddevPop),
        "var_pop" => Some(ast::AggregateFunc::VarPop),
        "bit_and" => Some(ast::AggregateFunc::BitAnd),
        "bit_or" => Some(ast::AggregateFunc::BitOr),
        "bit_xor" => Some(ast::AggregateFunc::BitXor),
        "corr" => Some(ast::AggregateFunc::Corr),
        "covar_pop" => Some(ast::AggregateFunc::CovarPop),
        "covar_samp" => Some(ast::AggregateFunc::CovarSamp),
        "regr_count" => Some(ast::AggregateFunc::RegrCount),
        "regr_avgx" => Some(ast::AggregateFunc::RegrAvgx),
        "regr_avgy" => Some(ast::AggregateFunc::RegrAvgy),
        "regr_sxx" => Some(ast::AggregateFunc::RegrSxx),
        "regr_syy" => Some(ast::AggregateFunc::RegrSyy),
        "regr_sxy" => Some(ast::AggregateFunc::RegrSxy),
        "regr_slope" => Some(ast::AggregateFunc::RegrSlope),
        "regr_intercept" => Some(ast::AggregateFunc::RegrIntercept),
        "regr_r2" => Some(ast::AggregateFunc::RegrR2),
        // `GROUP_CONCAT` is an alternate spelling of `STRING_AGG` (with a default `,` separator).
        "string_agg" | "group_concat" => Some(ast::AggregateFunc::StringAgg),
        _ => None,
    }
}

/// Lower an aggregate call's arguments. `COUNT(*)` becomes a no-argument
/// `Count`; every other form takes exactly one scalar argument expression.
/// Convert a sqlparser `ORDER BY` list (from a window spec or an aggregate's argument clause) into
/// [`ast::OrderByItem`]s. `WITH FILL` is rejected; a missing `ASC`/`DESC` defaults to `ASC`.
pub(super) fn convert_order_by_items(
    items: Vec<sql::OrderByExpr>,
) -> Result<Vec<ast::OrderByItem>, Error> {
    items
        .into_iter()
        .map(|obe| {
            if obe.with_fill.is_some() {
                return unsupported("ORDER BY ... WITH FILL");
            }
            let nulls = match obe.options.nulls_first {
                None => ast::NullOrdering::Default,
                Some(true) => ast::NullOrdering::First,
                Some(false) => ast::NullOrdering::Last,
            };
            Ok(ast::OrderByItem {
                expr: convert_expr(obe.expr)?,
                ascending: obe.options.asc.unwrap_or(true),
                nulls,
            })
        })
        .collect()
}

pub(super) fn convert_aggregate(
    func: ast::AggregateFunc,
    args: Vec<sql::FunctionArg>,
    distinct: bool,
    filter: Option<Box<ast::Expr>>,
    order_by: Vec<ast::OrderByItem>,
) -> Result<ast::Expr, Error> {
    // `ORDER BY` inside an aggregate only changes the result of the order-sensitive collectors.
    if !order_by.is_empty()
        && !matches!(
            func,
            ast::AggregateFunc::ArrayAgg
                | ast::AggregateFunc::JsonAgg
                | ast::AggregateFunc::StringAgg
        )
    {
        return unsupported(
            "ORDER BY inside an aggregate is only supported for ARRAY_AGG / JSONB_AGG / STRING_AGG",
        );
    }
    let is_wildcard = matches!(
        args.first(),
        Some(sql::FunctionArg::Unnamed(
            sql::FunctionArgExpr::Wildcard | sql::FunctionArgExpr::QualifiedWildcard(_)
        ))
    );

    // COUNT(*) — exactly one wildcard, no inner expression.
    if matches!(func, ast::AggregateFunc::Count) && args.len() == 1 && is_wildcard {
        return Ok(ast::Expr::Aggregate {
            func,
            arg: None,
            distinct,
            filter,
            separator: None,
            arg2: None,
            order_by: Vec::new(),
        });
    }

    // JSON_OBJECT_AGG(key, value) takes two per-row arguments (key, value) — the key becomes an
    // object key (text), the value any JSON value. Unlike the statistical two-arg aggregates the value
    // is not numeric-constrained, so it has its own block (like STRING_AGG). DISTINCT / ORDER BY are
    // not supported for the object form.
    if matches!(func, ast::AggregateFunc::JsonObjectAgg) {
        if distinct {
            return unsupported("JSON_OBJECT_AGG does not support DISTINCT");
        }
        if !order_by.is_empty() {
            return unsupported("JSON_OBJECT_AGG does not support ORDER BY");
        }
        let [key, value] = unnamed_two_args(args, "JSON_OBJECT_AGG")?;
        return Ok(ast::Expr::Aggregate {
            func,
            arg: Some(Box::new(convert_expr(key)?)),
            distinct,
            filter,
            separator: None,
            arg2: Some(Box::new(convert_expr(value)?)),
            order_by: Vec::new(),
        });
    }

    // STRING_AGG / GROUP_CONCAT take the value plus an optional separator; a missing separator
    // defaults to ',' (the GROUP_CONCAT default).
    if matches!(func, ast::AggregateFunc::StringAgg) {
        let (value, sep) = unnamed_one_or_two_args(args, "STRING_AGG")?;
        let separator = match sep {
            Some(sep) => Box::new(convert_expr(sep)?),
            None => Box::new(ast::Expr::Literal(ast::Value::Text(",".to_owned()))),
        };
        return Ok(ast::Expr::Aggregate {
            func,
            arg: Some(Box::new(convert_expr(value)?)),
            distinct,
            filter,
            separator: Some(separator),
            arg2: None,
            order_by,
        });
    }

    // The two-argument statistical aggregates (CORR / COVAR_* / REGR_*) take two plain-expression
    // arguments (y, x).
    if func.is_two_arg() {
        let [y, x] = unnamed_two_args(args, &format!("{func:?}"))?;
        return Ok(ast::Expr::Aggregate {
            func,
            arg: Some(Box::new(convert_expr(y)?)),
            distinct,
            filter,
            separator: None,
            arg2: Some(Box::new(convert_expr(x)?)),
            order_by: Vec::new(),
        });
    }

    if args.len() != 1 {
        return unsupported(&format!("{func:?} takes exactly one argument"));
    }
    let arg = match args.into_iter().next() {
        Some(sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e))) => e,
        Some(sql::FunctionArg::Unnamed(
            sql::FunctionArgExpr::Wildcard
            | sql::FunctionArgExpr::QualifiedWildcard(_)
            | sql::FunctionArgExpr::WildcardWithOptions(_),
        )) => return unsupported("`*` is only valid as the argument to COUNT"),
        Some(sql::FunctionArg::Named { .. } | sql::FunctionArg::ExprNamed { .. }) => {
            return unsupported("named function argument");
        },
        None => return unsupported("aggregate requires an argument"),
    };
    Ok(ast::Expr::Aggregate {
        func,
        arg: Some(Box::new(convert_expr(arg)?)),
        distinct,
        filter,
        separator: None,
        arg2: None,
        order_by,
    })
}

/// Extract one required plus one optional unnamed plain-expression argument (for `STRING_AGG` /
/// `GROUP_CONCAT`, whose separator may be omitted).
fn unnamed_one_or_two_args(
    args: Vec<sql::FunctionArg>,
    name: &str,
) -> Result<(sql::Expr, Option<sql::Expr>), Error> {
    if args.is_empty() || args.len() > 2 {
        return unsupported(&format!("{name} takes one or two arguments"));
    }
    let mut exprs = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => exprs.push(e),
            _ => return unsupported(&format!("{name} arguments must be plain expressions")),
        }
    }
    let mut it = exprs.into_iter();
    let Some(value) = it.next() else {
        return unsupported(&format!("{name} requires at least one argument"));
    };
    Ok((value, it.next()))
}

/// Extract exactly two unnamed plain-expression arguments (for `STRING_AGG` and the two-argument
/// statistical aggregates), rejecting `*`, named, or the wrong count.
fn unnamed_two_args(args: Vec<sql::FunctionArg>, name: &str) -> Result<[sql::Expr; 2], Error> {
    if args.len() != 2 {
        return unsupported(&format!("{name} takes exactly two arguments"));
    }
    let mut out = Vec::with_capacity(2);
    for arg in args {
        match arg {
            sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => out.push(e),
            _ => return unsupported(&format!("{name} arguments must be plain expressions")),
        }
    }
    out.try_into()
        .map_err(|_| Error::FunctionArgs(format!("{name} takes exactly two arguments")))
}

/// Lower a positional placeholder `$1`, `$2`, … into a zero-based
/// [`ast::Expr::Parameter`]. Only the `$N` form is supported.
pub(super) fn convert_placeholder(p: &str) -> Result<ast::Expr, Error> {
    let index = p
        .strip_prefix('$')
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            Error::InvalidStatement(format!("unsupported placeholder `{p}` (use $1, $2, …)"))
        })?;
    Ok(ast::Expr::Parameter(index - 1))
}

pub(super) fn convert_value(value: sql::Value) -> Result<ast::Value, Error> {
    match value {
        sql::Value::Null => Ok(ast::Value::Null),
        sql::Value::Boolean(b) => Ok(ast::Value::Bool(b)),
        sql::Value::Number(n, _) => convert_number(&n),
        sql::Value::SingleQuotedString(s)
        | sql::Value::DoubleQuotedString(s)
        // E'…' / U&'…' arrive already unescaped from the tokenizer.
        | sql::Value::EscapedStringLiteral(s)
        | sql::Value::UnicodeStringLiteral(s) => Ok(ast::Value::Text(s)),
        other => unsupported(&format!("literal `{other}`")),
    }
}

pub(super) fn convert_number(n: &str) -> Result<ast::Value, Error> {
    if let Ok(int) = n.parse::<i64>() {
        return Ok(ast::Value::Int(int));
    }
    // A plain decimal literal (`0.1`, `123.45`, `.5`) — or an integer too large for `i64` — is
    // *exact* NUMERIC, not `f64`: parsing it as `f64` here would silently round it before it ever
    // reached a NUMERIC column or exact arithmetic (`INSERT INTO t(numeric_col) VALUES
    // (0.45)` must not pre-round). Forms `Decimal::parse` does not model — an exponent (`1e10`), or
    // more fractional digits than `MAX_SCALE` — fall back to `f64`.
    if let Some(dec) = crate::numeric::Decimal::parse(n) {
        return Ok(ast::Value::Numeric(dec));
    }
    if let Ok(float) = n.parse::<f64>() {
        return Ok(ast::Value::Float(float));
    }
    unsupported(&format!("numeric literal `{n}`"))
}

/// Build an [`ast::Expr::IsDistinctFrom`] from a sqlparser `IS [NOT] DISTINCT FROM`.
///
/// The `strip_is_json` preprocessor rewrites each postfix `<expr> IS JSON ...` (which sqlparser
/// cannot parse) into `<expr> IS DISTINCT FROM __nusadb_isjson_<n>`. sqlparser binds the identical
/// **left** operand (an IS-predicate has the same precedence as `IS JSON`), but it parses the
/// **right** operand of `IS DISTINCT FROM` with *full* precedence, so any infix operator that
/// follows the predicate is swallowed into that right operand — `x IS JSON AND y` arrives as
/// `x IS DISTINCT FROM (__nusadb_isjson_1 AND y)`. The sentinel is therefore always the leftmost
/// leaf of the right subtree; when it is present, splice the [`ast::Expr::IsJson`] over the
/// (untouched) left operand in the sentinel's place and re-associate the rest of the subtree —
/// recovering `(x IS JSON) AND y`. Any other `IS DISTINCT FROM` is converted normally.
pub(super) fn convert_is_distinct(
    left: sql::Expr,
    right: sql::Expr,
    negated: bool,
) -> Result<ast::Expr, Error> {
    // `IS NOT DISTINCT FROM <sentinel>` never occurs — the preprocessor only ever emits the plain
    // `IS DISTINCT FROM` form — so a sentinel only needs recognising on the non-negated path.
    if !negated && spine_bottoms_at_is_json_sentinel(&right) {
        let operand = Box::new(convert_expr(left)?);
        return build_is_json_spine(right, operand);
    }
    Ok(ast::Expr::IsDistinctFrom {
        left: Box::new(convert_expr(left)?),
        right: Box::new(convert_expr(right)?),
        negated,
    })
}

/// Whether the left spine of `expr` (descending through binary operators) bottoms out at an
/// unquoted `IS JSON` sentinel identifier — the signal that `expr` is the greedily-parsed right
/// operand of a rewritten `IS JSON` predicate rather than a genuine `IS DISTINCT FROM` operand.
fn spine_bottoms_at_is_json_sentinel(expr: &sql::Expr) -> bool {
    let mut cur = expr;
    loop {
        match cur {
            sql::Expr::Identifier(ident) => {
                return ident.quote_style.is_none()
                    && IS_JSON_MODES.with(|c| c.borrow().contains_key(&ident.value));
            },
            sql::Expr::BinaryOp { left, .. } => cur = left,
            _ => return false,
        }
    }
}

/// Rebuild the greedily-parsed right operand `expr`, replacing its leftmost-leaf `IS JSON` sentinel
/// with the [`ast::Expr::IsJson`] predicate over `operand` and re-associating the binary spine above
/// it. Only reached after [`spine_bottoms_at_is_json_sentinel`] confirms a sentinel is present.
fn build_is_json_spine(expr: sql::Expr, operand: Box<ast::Expr>) -> Result<ast::Expr, Error> {
    match expr {
        sql::Expr::Identifier(ident) if ident.quote_style.is_none() => {
            match IS_JSON_MODES.with(|c| c.borrow_mut().remove(&ident.value)) {
                Some((negated, item_type, unique_keys)) => Ok(ast::Expr::IsJson {
                    operand,
                    negated,
                    item_type,
                    unique_keys,
                }),
                // Unreachable: the spine check just confirmed this identifier is a live sentinel.
                None => unsupported("internal: IS JSON sentinel vanished before conversion"),
            }
        },
        sql::Expr::BinaryOp { left, op, right } => Ok(ast::Expr::Binary {
            left: Box::new(build_is_json_spine(*left, operand)?),
            op: convert_binary_op(op)?,
            right: Box::new(convert_expr(*right)?),
        }),
        // Unreachable given the spine check, which only descends through binary operators.
        other => unsupported(&format!(
            "unexpected expression after IS JSON predicate: `{other}`"
        )),
    }
}

/// Build an [`ast::Expr::IsBool`] from a sqlparser `IS [NOT] {TRUE|FALSE|UNKNOWN}`.
pub(super) fn convert_is_bool(
    inner: sql::Expr,
    truth: ast::TruthValue,
    negated: bool,
) -> Result<ast::Expr, Error> {
    Ok(ast::Expr::IsBool {
        expr: Box::new(convert_expr(inner)?),
        truth,
        negated,
    })
}

/// One side of an `OVERLAPS` predicate: `(start, end)`. `sqlparser` parses each side as an
/// `Expr::Tuple` of exactly two elements; any other shape (a single parenthesised value arrives as
/// `Expr::Nested`, a longer row as a wider tuple) is rejected loudly.
fn overlaps_pair(side: sql::Expr) -> Result<[sql::Expr; 2], Error> {
    let unsupported =
        || Error::InvalidStatement("OVERLAPS requires two 2-element row expressions".to_owned());
    match side {
        sql::Expr::Tuple(elems) => <[sql::Expr; 2]>::try_from(elems).map_err(|_| unsupported()),
        _ => Err(unsupported()),
    }
}

pub(super) fn convert_binary_op(op: sql::BinaryOperator) -> Result<ast::BinaryOp, Error> {
    use sql::BinaryOperator as B;
    let mapped = match op {
        B::Eq => ast::BinaryOp::Eq,
        B::NotEq => ast::BinaryOp::NotEq,
        B::Lt => ast::BinaryOp::Lt,
        B::LtEq => ast::BinaryOp::LtEq,
        B::Gt => ast::BinaryOp::Gt,
        B::GtEq => ast::BinaryOp::GtEq,
        B::And => ast::BinaryOp::And,
        B::Or => ast::BinaryOp::Or,
        B::Plus => ast::BinaryOp::Plus,
        B::Minus => ast::BinaryOp::Minus,
        B::Multiply => ast::BinaryOp::Multiply,
        B::Divide => ast::BinaryOp::Divide,
        B::Modulo => ast::BinaryOp::Modulo,
        // Integer bitwise AND / OR / XOR. `#` (the reference engine's XOR) is parsed by the
        // dialect hook into `PGBitwiseXor`; the caret (`BitwiseXor` under the generic tokenizer)
        // is the reference engine's exponentiation and is lowered to `power()` before this table is consulted.
        B::BitwiseAnd => ast::BinaryOp::BitAnd,
        B::BitwiseOr => ast::BinaryOp::BitOr,
        B::PGBitwiseXor => ast::BinaryOp::BitXor,
        // Integer bit-shifts `<<` / `>>` (B-fn).
        B::PGBitwiseShiftLeft => ast::BinaryOp::ShiftLeft,
        B::PGBitwiseShiftRight => ast::BinaryOp::ShiftRight,
        // Array overlap `&&` — whether two arrays share any element (B-fn).
        B::PGOverlap => ast::BinaryOp::ArrayOverlap,
        // String concatenation `||`.
        B::StringConcat => ast::BinaryOp::Concat,
        // JSON operators.
        B::Arrow => ast::BinaryOp::JsonGet,
        B::LongArrow => ast::BinaryOp::JsonGetText,
        B::AtArrow => ast::BinaryOp::JsonContains,
        B::ArrowAt => ast::BinaryOp::JsonContainedBy,
        // JSON path operators.
        B::HashArrow => ast::BinaryOp::JsonGetPath,
        B::HashLongArrow => ast::BinaryOp::JsonGetPathText,
        // Vector cosine distance `<=>` (sqlparser tokenizes it as `Spaceship`).
        B::Spaceship => ast::BinaryOp::VectorDistance,
        // The remaining vector distance operators, and the JSON key-existence operators, arrive as
        // `Custom` from the dialect's infix hook, which fuses their multi-token forms
        // (`<` `->` / `<` `#>` / `<` `+` `>`, and `?` `|` / `?` `&`).
        B::Custom(symbol) => match symbol.as_str() {
            "<->" => ast::BinaryOp::VectorL2Distance,
            "<#>" => ast::BinaryOp::VectorNegInnerProduct,
            "<+>" => ast::BinaryOp::VectorL1Distance,
            "?" => ast::BinaryOp::JsonExists,
            "?|" => ast::BinaryOp::JsonExistsAny,
            "?&" => ast::BinaryOp::JsonExistsAll,
            other => return unsupported(&format!("binary operator `{other}`")),
        },
        // Full-text search match `@@` (F1).
        B::AtAt => ast::BinaryOp::TsMatch,
        other => return unsupported(&format!("binary operator `{other}`")),
    };
    Ok(mapped)
}

pub(super) fn convert_unary_op(op: sql::UnaryOperator) -> Result<ast::UnaryOp, Error> {
    match op {
        sql::UnaryOperator::Not => Ok(ast::UnaryOp::Not),
        sql::UnaryOperator::Minus => Ok(ast::UnaryOp::Negate),
        sql::UnaryOperator::Plus => Ok(ast::UnaryOp::Plus),
        other => unsupported(&format!("unary operator `{other}`")),
    }
}
