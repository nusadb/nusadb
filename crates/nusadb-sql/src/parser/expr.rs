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
                        return unsupported(&format!("composite field access `.{field}`"));
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
            Ok(ast::Expr::Cast {
                expr: Box::new(convert_expr(*expr)?),
                target: convert_data_type(&data_type)?,
                try_cast,
            })
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
    if function.filter.is_some()
        || function.null_treatment.is_some()
        || !function.within_group.is_empty()
        || !matches!(function.parameters, sql::FunctionArguments::None)
    {
        return unsupported("FILTER / NULLS / WITHIN GROUP in window function");
    }
    let spec = match function.over {
        Some(sql::WindowType::WindowSpec(s)) => s,
        // `OVER w` — resolve `w` to a `WINDOW w AS (...)` definition of the enclosing SELECT.
        Some(sql::WindowType::NamedWindow(name)) => {
            let key = fold_ident(&name);
            NAMED_WINDOWS
                .with(|c| c.borrow().get(&key).cloned())
                .ok_or_else(|| {
                    Error::Unsupported(format!("window `{key}` is not defined in a WINDOW clause"))
                })?
        },
        None => unreachable!("convert_window_function called without OVER clause"),
    };
    // window_frame is handled below; no early rejection here.
    if spec.window_name.is_some() {
        return unsupported("named window base in OVER clause is not yet supported");
    }
    let name_ident = function
        .name
        .0
        .last()
        .ok_or_else(|| Error::Unsupported("empty window function name".to_owned()))?;
    let name = fold_part(name_ident)?;
    let func = window_func_by_name(&name)
        .ok_or_else(|| Error::Unsupported(format!("window function `{name}` is not recognised")))?;
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
    let frame = spec.window_frame.map(convert_window_frame).transpose()?;
    Ok(ast::Expr::WindowFunction(Box::new(ast::WindowFunction {
        func,
        args,
        partition,
        order,
        frame,
    })))
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
    Ok(ast::WindowFrame { units, start, end })
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
        .ok_or_else(|| Error::Unsupported("empty function name".to_owned()))?;
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
        .ok_or_else(|| Error::Unsupported("empty function name".to_owned()))?;
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

/// Extract the single `LIKE`/`ILIKE` `ESCAPE 'c'` character from sqlparser's string model, rejecting
/// a multi-character or empty escape.
fn like_escape_char(escape_char: Option<sql::ValueWithSpan>) -> Result<Option<char>, Error> {
    escape_char
        .map(|v| {
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
            ch.ok_or_else(|| {
                Error::Unsupported("LIKE ESCAPE must be a non-empty character".to_owned())
            })
        })
        .transpose()
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
        // UUID generation: both the modern built-in name and the alternative spelling.
        "gen_random_uuid" | "uuid_generate_v4" => F::UuidGenerateV4,
        // System functions.
        "version" => F::Version,
        // `nusa_typeof(expr)` — NusaDB's spelling of the `pg_typeof` idiom; folded to the static
        // type name at analysis.
        "nusa_typeof" => F::NusaTypeof,
        "current_database" => F::CurrentDatabase,
        "current_schema" => F::CurrentSchema,
        // JSON inspection + construction scalars.
        "json_typeof" | "jsonb_typeof" => F::JsonTypeof,
        "json_array_length" | "jsonb_array_length" => F::JsonArrayLength,
        "to_json" | "to_jsonb" => F::ToJson,
        "row_to_json" => F::RowToJson,
        "json_build_object" | "jsonb_build_object" => F::JsonBuildObject,
        "json_build_array" | "jsonb_build_array" => F::JsonBuildArray,
        "jsonb_set" | "json_set" => F::JsonbSet,
        "jsonb_strip_nulls" | "json_strip_nulls" => F::JsonbStripNulls,
        "jsonb_pretty" | "json_pretty" => F::JsonbPretty,
        "jsonb_path_exists" | "json_path_exists" => F::JsonbPathExists,
        "jsonb_insert" | "json_insert" => F::JsonbInsert,
        "jsonb_path_query_first" | "json_path_query_first" => F::JsonbPathQueryFirst,
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
        .map_err(|_| Error::Unsupported(format!("{name} takes exactly two arguments")))
}

/// Lower a positional placeholder `$1`, `$2`, … into a zero-based
/// [`ast::Expr::Parameter`]. Only the `$N` form is supported.
pub(super) fn convert_placeholder(p: &str) -> Result<ast::Expr, Error> {
    let index = p
        .strip_prefix('$')
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            Error::Unsupported(format!("unsupported placeholder `{p}` (use $1, $2, …)"))
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
pub(super) fn convert_is_distinct(
    left: sql::Expr,
    right: sql::Expr,
    negated: bool,
) -> Result<ast::Expr, Error> {
    Ok(ast::Expr::IsDistinctFrom {
        left: Box::new(convert_expr(left)?),
        right: Box::new(convert_expr(right)?),
        negated,
    })
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
        other => unsupported(&format!("unary operator `{other}`")),
    }
}
