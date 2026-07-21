//! Expression type-checking: predicates, `analyze_expr` and its per-kind helpers.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === Predicates ===========================================================

/// Analyze an optional `WHERE` predicate; the result must be boolean-typed.
///
/// The boolean expectation is also passed as the NULL type hint, so a literal
/// `WHERE NULL` is accepted (it types as boolean `NULL`).
pub(super) fn analyze_predicate(
    predicate: Option<ast::Expr>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<Option<TypedExpr>, Error> {
    let Some(expr) = predicate else {
        return Ok(None);
    };
    let typed = analyze_expr(&expr, scope, catalog, Some(ColumnType::Bool))?;
    if typed.ty != ColumnType::Bool {
        return Err(Error::TypeMismatch {
            context: "WHERE clause".to_owned(),
            expected: ColumnType::Bool,
            found: typed.ty,
        });
    }
    Ok(Some(typed))
}

// === Expression type-checking =============================================

pub(super) fn analyze_expr(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    hint: Option<ColumnType>,
) -> Result<TypedExpr, Error> {
    analyze_expr_agg(expr, scope, catalog, hint, None)
}

/// Like [`analyze_expr`], but with an optional aggregate sink. When
/// `aggregates` is `Some`, an aggregate call anywhere in the expression is
/// registered into the sink and replaced by a [`TypedExprKind::AggregateRef`];
/// when `None` (a `WHERE` clause, an aggregate's own argument, ...) an
/// aggregate is rejected. All type-checking lives here, so projection and
/// non-projection contexts agree on what types compose.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-expression-kind dispatch; length scales with the expression grammar"
)]
pub(super) fn analyze_expr_agg(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    hint: Option<ColumnType>,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // Coerce against the *physical* type: a `VARCHAR(n)`/`CHAR(n)` column hint behaves as `TEXT`,
    // so literal/value coercion never has to special-case the declared character types.
    let hint = hint.map(ColumnType::physical);
    match expr {
        ast::Expr::Literal(value) => {
            let ty = match value {
                ast::Value::Null => return analyze_null(hint),
                ast::Value::Bool(_) => ColumnType::Bool,
                // The reference engine types an integer literal by magnitude: `INT` (int4) when it fits, else `BIGINT`
                // (int8). This keeps `2147483647 + 1` an int4 overflow while a literal that needs 64
                // bits stays a bigint and so is not falsely bounded at int4.
                ast::Value::Int(i) => {
                    if i32::try_from(*i).is_ok() {
                        ColumnType::Int
                    } else {
                        ColumnType::BigInt
                    }
                },
                ast::Value::Float(_) => ColumnType::Float,
                ast::Value::Text(_) => ColumnType::Text,
                ast::Value::Date(_) => ColumnType::Date,
                ast::Value::Time(_) => ColumnType::Time,
                ast::Value::Timestamp(_) => ColumnType::Timestamp,
                ast::Value::TimestampTz(_) => ColumnType::TimestampTz,
                ast::Value::TimeTz(_) => ColumnType::TimeTz,
                ast::Value::Uuid(_) => ColumnType::Uuid,
                // A numeric literal's declared precision/scale is unconstrained.
                ast::Value::Numeric(_) => ColumnType::Numeric {
                    precision: 0,
                    scale: 0,
                },
                ast::Value::Json(_) => ColumnType::Json,
                ast::Value::Interval(_) => ColumnType::Interval,
                ast::Value::Array(items) => {
                    // Reject a heterogeneous array literal (`{1,'a'}`) here rather than
                    // inferring from the first element and failing later at encode.
                    ColumnType::Array(crate::executor::row::array_elem_checked(items)?)
                },
                // A vector literal types as its own dimension; the parser produces vectors
                // via a text cast, so this arm is reached only for a synthesized literal.
                #[allow(clippy::cast_possible_truncation, reason = "vector dim fits u32")]
                ast::Value::Vector(v) => ColumnType::Vector(v.len() as u32),
                ast::Value::Bytes(_) => ColumnType::Bytes,
            };
            Ok(TypedExpr {
                kind: TypedExprKind::Literal(value.clone()),
                ty,
            })
        },
        ast::Expr::Column(name) => {
            let (kind, ty) = resolve_scoped_or_outer(scope, None, name)?;
            Ok(TypedExpr { kind, ty })
        },
        // A `$n` placeholder must be replaced by `bind_parameters` before analysis (extended
        // query protocol); one reaching here was never bound.
        ast::Expr::Parameter(n) => Err(Error::Unsupported(format!(
            "parameter ${} was not bound (use a prepared statement with Bind)",
            n + 1
        ))),
        ast::Expr::QualifiedColumn { table, column } => {
            let (kind, ty) = resolve_scoped_or_outer(scope, Some(table), column)?;
            Ok(TypedExpr { kind, ty })
        },
        ast::Expr::Binary { left, op, right } => {
            analyze_binary(left, *op, right, scope, catalog, aggregates)
        },
        ast::Expr::Unary { op, expr } => analyze_unary(*op, expr, scope, catalog, aggregates),
        ast::Expr::IsNull { expr, negated } => {
            // `IS [NOT] NULL` only inspects nullness, so the operand's type is irrelevant — a bare
            // `NULL` operand (`NULL IS NULL`) takes a default `TEXT` type rather than being rejected
            // as untyped. A typed operand keeps its own type (the hint is a fallback only).
            let operand =
                analyze_expr_agg(expr, scope, catalog, Some(ColumnType::Text), aggregates)?;
            Ok(TypedExpr {
                kind: TypedExprKind::IsNull {
                    expr: Box::new(operand),
                    negated: *negated,
                },
                ty: ColumnType::Bool,
            })
        },
        ast::Expr::IsDistinctFrom {
            left,
            right,
            negated,
        } => analyze_is_distinct_from(left, right, *negated, scope, catalog, aggregates),
        ast::Expr::IsBool {
            expr,
            truth,
            negated,
        } => {
            let operand =
                analyze_expr_agg(expr, scope, catalog, Some(ColumnType::Bool), aggregates)?;
            if operand.ty != ColumnType::Bool {
                return Err(Error::TypeMismatch {
                    context: "IS TRUE/FALSE/UNKNOWN".to_owned(),
                    expected: ColumnType::Bool,
                    found: operand.ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::IsBool {
                    expr: Box::new(operand),
                    truth: *truth,
                    negated: *negated,
                },
                ty: ColumnType::Bool,
            })
        },
        ast::Expr::InList {
            expr,
            list,
            negated,
        } => analyze_in_list(expr, list, *negated, scope, catalog, aggregates),
        ast::Expr::Between {
            expr,
            low,
            high,
            negated,
        } => analyze_between(expr, low, high, *negated, scope, catalog, aggregates),
        ast::Expr::Like {
            expr,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => analyze_like(
            expr,
            pattern,
            *negated,
            *escape,
            *case_insensitive,
            scope,
            catalog,
            aggregates,
        ),
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => analyze_case(
            operand.as_deref(),
            branches,
            default.as_deref(),
            scope,
            catalog,
            aggregates,
        ),
        ast::Expr::Coalesce(args) => analyze_coalesce(args, scope, catalog, aggregates),
        ast::Expr::Cast {
            expr,
            target,
            try_cast,
        } => analyze_cast(expr, *target, *try_cast, scope, catalog, aggregates),
        ast::Expr::Encrypt { value, key } => {
            analyze_crypto(CryptoOp::Encrypt, value, key, scope, catalog, aggregates)
        },
        ast::Expr::Decrypt { value, key } => {
            analyze_crypto(CryptoOp::Decrypt, value, key, scope, catalog, aggregates)
        },
        ast::Expr::ScalarFunction { func, args } => {
            analyze_scalar_function(*func, args, scope, catalog, aggregates)
        },
        ast::Expr::FunctionCall { name, args } => {
            analyze_udf_call(name, args, scope, catalog, aggregates)
        },
        // A set-returning function reaching the general expression path is misplaced — it is only
        // valid at the top of a SELECT-list item, where `analyze_projection` handles it.
        ast::Expr::SetReturning { func, .. } => Err(Error::Unsupported(format!(
            "set-returning function {}() may only appear at the top level of the SELECT list",
            func.name()
        ))),
        // Subqueries. The body is analyzed against its own scope first; a column
        // that misses it falls back to the enclosing scope pushed here, producing an `OuterColumn`
        // (a correlated subquery). An uncorrelated body simply never references the outer
        // scope. The executor pre-resolves uncorrelated subqueries once; correlated ones run per
        // outer row.
        ast::Expr::ScalarSubquery(subquery) => {
            let plan = {
                let _outer = push_outer_scope(scope);
                analyze_select((**subquery).clone(), catalog)?
            };
            let ty = single_subquery_column(&plan, "scalar subquery")?;
            Ok(TypedExpr {
                kind: TypedExprKind::ScalarSubquery(Box::new(plan)),
                ty,
            })
        },
        ast::Expr::Exists { negated, subquery } => {
            // EXISTS only tests row presence, so the projection arity is irrelevant.
            let plan = {
                let _outer = push_outer_scope(scope);
                analyze_select((**subquery).clone(), catalog)?
            };
            Ok(TypedExpr {
                kind: TypedExprKind::Exists {
                    plan: Box::new(plan),
                    negated: *negated,
                },
                ty: ColumnType::Bool,
            })
        },
        ast::Expr::InSubquery {
            expr,
            negated,
            subquery,
        } => {
            let probe = analyze_expr_agg(expr, scope, catalog, None, aggregates)?;
            let plan = {
                let _outer = push_outer_scope(scope);
                analyze_select((**subquery).clone(), catalog)?
            };
            let elem_ty = single_subquery_column(&plan, "IN (subquery)")?;
            if probe.ty != elem_ty && !is_null_literal(&probe) {
                return Err(Error::TypeMismatch {
                    context: "IN (subquery)".to_owned(),
                    expected: probe.ty,
                    found: elem_ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::InSubquery {
                    expr: Box::new(probe),
                    plan: Box::new(plan),
                    negated: *negated,
                },
                ty: ColumnType::Bool,
            })
        },
        ast::Expr::QuantifiedComparison {
            expr,
            op,
            all,
            subquery,
        } => {
            let probe = analyze_expr_agg(expr, scope, catalog, None, aggregates)?;
            let plan = {
                let _outer = push_outer_scope(scope);
                analyze_select((**subquery).clone(), catalog)?
            };
            let elem_ty = single_subquery_column(&plan, "quantified subquery")?;
            if probe.ty != elem_ty && !is_null_literal(&probe) {
                return Err(Error::TypeMismatch {
                    context: "quantified comparison (subquery)".to_owned(),
                    expected: probe.ty,
                    found: elem_ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::QuantifiedSubquery {
                    expr: Box::new(probe),
                    op: *op,
                    all: *all,
                    plan: Box::new(plan),
                },
                ty: ColumnType::Bool,
            })
        },
        ast::Expr::QuantifiedArray {
            expr,
            op,
            all,
            array,
        } => {
            let mut aggregates = aggregates;
            let probe = analyze_expr_agg(expr, scope, catalog, None, aggregates.as_deref_mut())?;
            let array_typed = analyze_expr_agg(array, scope, catalog, None, aggregates)?;
            // A bound array parameter (`id = ANY($1)`) arrives as a bare TEXT literal — a driver sends
            // the array as its `{...}` text form, which our binding types as TEXT. Coerce it to an
            // array of the probe's element type, exactly as an explicit `$1::int[]` would: the executor
            // parses the text at evaluation and an unparseable literal still loud-rejects (never a
            // silent wrong row). Only a bare TEXT literal is coerced; a genuinely non-array operand
            // (or a probe type that cannot be an array element) still falls through to the mismatch.
            let array_typed = match nusadb_core::engine::ArrayElem::from_column_type(probe.ty) {
                Some(elem)
                    if matches!(
                        &array_typed.kind,
                        TypedExprKind::Literal(ast::Value::Text(_))
                    ) =>
                {
                    TypedExpr {
                        kind: TypedExprKind::Cast(Box::new(array_typed), false),
                        ty: ColumnType::Array(elem),
                    }
                },
                _ => array_typed,
            };
            let ColumnType::Array(elem) = array_typed.ty else {
                return Err(Error::TypeMismatch {
                    context: "ANY/ALL right operand".to_owned(),
                    expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
                    found: array_typed.ty,
                });
            };
            let elem_ty = elem.column_type();
            if probe.ty != elem_ty && !is_null_literal(&probe) {
                return Err(Error::TypeMismatch {
                    context: "ANY/ALL comparison against an array element".to_owned(),
                    expected: probe.ty,
                    found: elem_ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::QuantifiedArray {
                    expr: Box::new(probe),
                    op: *op,
                    all: *all,
                    array: Box::new(array_typed),
                },
                ty: ColumnType::Bool,
            })
        },
        // SIMILAR TO: SQL-standard regex match; both operands Text, result Bool.
        ast::Expr::SimilarTo {
            expr,
            pattern,
            negated,
        } => analyze_similar_to(expr, pattern, *negated, scope, catalog, aggregates),
        // Regex match `~`/`~*`/`!~`/`!~*`.
        ast::Expr::RegexMatch {
            expr,
            pattern,
            case_sensitive,
            negated,
        } => analyze_regex_match(
            expr,
            pattern,
            *case_sensitive,
            *negated,
            scope,
            catalog,
            aggregates,
        ),
        // Array constructor / subscript.
        ast::Expr::ArrayLiteral(elems) => {
            analyze_array_literal(elems, hint, scope, catalog, aggregates)
        },
        ast::Expr::Subscript { base, index } => {
            analyze_subscript(base, index, scope, catalog, aggregates)
        },
        ast::Expr::ArraySlice { base, lower, upper } => analyze_array_slice(
            base,
            lower.as_deref(),
            upper.as_deref(),
            scope,
            catalog,
            aggregates,
        ),
        // Ordered-set aggregate WITHIN GROUP.
        ast::Expr::WithinGroup(wg) => analyze_within_group(wg, scope, catalog, aggregates),
        // ROW(...) is parsed but row-value comparison/evaluation is not yet wired.
        ast::Expr::Row(_) => Err(Error::Unsupported(
            "ROW(...) constructor is parsed but the executor path is not yet implemented"
                .to_owned(),
        )),
        // A window function is only valid where the SELECT pipeline supplies a window stage
        // (the projection path lifts it before expression analysis); anywhere else there is
        // no execution path for it, so reject it here.
        ast::Expr::WindowFunction(_) => Err(Error::Unsupported(
            "window functions (OVER) are not supported in this position".to_owned(),
        )),
        // An aggregate is only valid where a sink is supplied (a projection,
        // `HAVING`, or `ORDER BY`). Anywhere else — a `WHERE` clause, or inside
        // another aggregate's own argument — there is no sink, so it is rejected.
        ast::Expr::Aggregate {
            func,
            arg,
            distinct,
            filter,
            separator,
            arg2,
            order_by,
        } => match aggregates {
            Some(sink) => {
                // COUNT(DISTINCT *) is meaningless — DISTINCT needs a concrete argument to dedupe.
                if *distinct && arg.is_none() {
                    return Err(Error::Unsupported(
                        "DISTINCT requires an argument (COUNT(DISTINCT *) is not valid)".to_owned(),
                    ));
                }
                // The two-argument statistical aggregates (CORR/COVAR_*/REGR_*) take a second
                // per-row numeric argument; DISTINCT is not meaningful over a pair.
                let two_arg = func.is_two_arg();
                if *distinct && two_arg {
                    return Err(Error::Unsupported(
                        "DISTINCT is not supported for two-argument statistical aggregates"
                            .to_owned(),
                    ));
                }
                let typed_arg2 = match (arg2, two_arg) {
                    (Some(a2), true) => {
                        let typed = analyze_expr(a2, scope, catalog, None)?;
                        if !is_numeric(typed.ty) {
                            return Err(Error::TypeMismatch {
                                context: format!("{func:?} requires numeric arguments"),
                                expected: ColumnType::Float,
                                found: typed.ty,
                            });
                        }
                        Some(typed)
                    },
                    (None, true) => {
                        return Err(Error::Unsupported(format!(
                            "{func:?} requires two arguments"
                        )));
                    },
                    (Some(_), false) => {
                        return Err(Error::Unsupported(format!(
                            "{func:?} takes a single argument"
                        )));
                    },
                    (None, false) => None,
                };
                // STRING_AGG's separator must be a constant string, resolved here to a plain value
                // the executor reads (it is not per-row state).
                let separator = match separator {
                    None => None,
                    Some(sep) => {
                        let typed = analyze_expr(sep, scope, catalog, Some(ColumnType::Text))?;
                        match typed.kind {
                            TypedExprKind::Literal(ast::Value::Text(s)) => Some(s),
                            _ => {
                                return Err(Error::Unsupported(
                                    "STRING_AGG separator must be a constant string".to_owned(),
                                ));
                            },
                        }
                    },
                };
                // FILTER (WHERE pred): the predicate is resolved against the pre-aggregation
                // scope (it sees input columns, not aggregates) and must be boolean.
                let typed_filter = match filter {
                    Some(pred) => {
                        let typed = analyze_expr(pred, scope, catalog, Some(ColumnType::Bool))?;
                        if typed.ty != ColumnType::Bool {
                            return Err(Error::TypeMismatch {
                                context: "aggregate FILTER (WHERE ...)".to_owned(),
                                expected: ColumnType::Bool,
                                found: typed.ty,
                            });
                        }
                        Some(typed)
                    },
                    None => None,
                };
                let (typed_arg, result_ty) =
                    analyze_aggregate(*func, arg.as_deref(), scope, catalog)?;
                // `ORDER BY` keys reference source rows (the pre-aggregation scope), not aggregates,
                // so they are resolved against `scope` with no aggregate sink.
                let mut order_keys = Vec::with_capacity(order_by.len());
                for item in order_by {
                    order_keys.push(OrderByKey {
                        expr: analyze_expr(&item.expr, scope, catalog, None)?,
                        ascending: item.ascending,
                        nulls: item.nulls,
                    });
                }
                let idx = sink.len();
                sink.push(AggregateCall {
                    func: *func,
                    arg: typed_arg,
                    result_ty,
                    distinct: *distinct,
                    fraction: None,
                    ordered_set_descending: false,
                    filter: typed_filter,
                    separator,
                    arg2: typed_arg2,
                    order_by: order_keys,
                    grouping_args: Vec::new(),
                });
                Ok(TypedExpr {
                    kind: TypedExprKind::AggregateRef(idx),
                    ty: result_ty,
                })
            },
            None => Err(Error::Unsupported(
                "aggregate functions are only allowed in a SELECT projection, HAVING, or ORDER BY"
                    .to_owned(),
            )),
        },
    }
}

pub(super) fn analyze_coalesce(
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    if args.is_empty() {
        return Err(Error::Unsupported("COALESCE with no arguments".to_owned()));
    }
    // Resolve the non-NULL arguments to a common result type first. A bare `NULL` literal carries no
    // type of its own, so it is deferred and typed from that result — this lets a leading `NULL`
    // infer from a later argument (e.g. `COALESCE(NULL, 7)`), which left-to-right typing cannot.
    let mut slots: Vec<Option<TypedExpr>> = Vec::with_capacity(args.len());
    let mut result_ty: Option<ColumnType> = None;
    let mut deferred: Vec<usize> = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if matches!(arg, ast::Expr::Literal(ast::Value::Null)) {
            deferred.push(i);
            slots.push(None);
            continue;
        }
        let typed = analyze_expr_agg(arg, scope, catalog, result_ty, aggregates.as_deref_mut())?;
        result_ty = Some(unify_result_ty(result_ty, typed.ty, "COALESCE")?);
        slots.push(Some(typed));
    }
    // Every argument NULL → an untyped NULL, which materializes as TEXT (the reference engine's unknown -> text), like
    // an all-NULL CASE — `COALESCE(NULL, NULL)` is NULL, not an "ambiguous type" error.
    let resolved = result_ty.unwrap_or(ColumnType::Text);
    for i in deferred {
        if let Some(slot) = slots.get_mut(i) {
            *slot = Some(analyze_null(Some(resolved))?);
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Coalesce(slots.into_iter().flatten().collect()),
        ty: resolved,
    })
}

/// Analyze `encrypt(value, key)` / `decrypt(value, key)`: both operands must be
/// `Text`, and the call returns `Text` (hex ciphertext or recovered plaintext).
pub(super) fn analyze_crypto(
    op: CryptoOp,
    value: &ast::Expr,
    key: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = match op {
        CryptoOp::Encrypt => "encrypt",
        CryptoOp::Decrypt => "decrypt",
    };
    let value = analyze_expr_agg(
        value,
        scope,
        catalog,
        Some(ColumnType::Text),
        aggregates.as_deref_mut(),
    )?;
    let key = analyze_expr_agg(key, scope, catalog, Some(ColumnType::Text), aggregates)?;
    for (arg, what) in [(&value, "value"), (&key, "key")] {
        if arg.ty != ColumnType::Text {
            return Err(Error::TypeMismatch {
                context: format!("{name}() {what}"),
                expected: ColumnType::Text,
                found: arg.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Crypto {
            op,
            value: Box::new(value),
            key: Box::new(key),
        },
        ty: ColumnType::Text,
    })
}

/// The argument/result contract of a scalar built-in.
#[derive(Clone, Copy)]
enum ScalarSig {
    /// Fixed arity: `required` argument types plus optional trailing types, and a result type.
    /// (The variadic CONCAT family lives in `analyze_text_polymorphic`, outside this table.)
    Fixed(&'static [ColumnType], &'static [ColumnType], ColumnType),
}

/// Analyze a call to a registered scalar user-defined function. The name is resolved against
/// the UDF registry; if no UDF is registered, the function is unknown. Each argument is analyzed with
/// the declared parameter type as its hint (so a bare `NULL` types from context) and checked to be
/// assignable to it; the result type is the UDF's declared return type.
fn analyze_udf_call(
    name: &str,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // Sequence built-ins are recognized by name (they are not `sqlparser` keywords, so they arrive
    // as generic function calls) before the UDF/SQL-function lookup below.
    if let Some(func) = sequence_func_by_name(name) {
        return analyze_sequence_function(func, name, args, scope, catalog, aggregates);
    }
    let Some((arg_types, return_type)) = crate::udf::scalar_udf_signature(name) else {
        // Not a Rust UDF — try a SQL scalar function, inlined in place of the call.
        if let Some(func) = catalog.lookup_function(name)? {
            return analyze_sql_function(name, &func, args, scope, catalog, aggregates);
        }
        return Err(Error::UnknownFunction(name.to_owned()));
    };
    if args.len() != arg_types.len() {
        return Err(Error::ArityMismatch {
            context: format!("function `{name}`"),
            expected: arg_types.len(),
            found: args.len(),
        });
    }
    let mut typed = Vec::with_capacity(args.len());
    for (arg, &want) in args.iter().zip(&arg_types) {
        let expr = analyze_expr_agg(arg, scope, catalog, Some(want), aggregates.as_deref_mut())?;
        if !assignable(want, expr.ty) {
            return Err(Error::TypeMismatch {
                context: format!("argument to function `{name}`"),
                expected: want,
                found: expr.ty,
            });
        }
        typed.push(expr);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarUdf {
            name: name.to_owned(),
            args: typed,
            arg_types,
        },
        ty: return_type,
    })
}

/// Map a (case-insensitive) function name to its sequence built-in, or `None` if it is not one.
const fn sequence_func_by_name(name: &str) -> Option<ast::ScalarFunc> {
    if name.eq_ignore_ascii_case("nextval") {
        Some(ast::ScalarFunc::SequenceNext)
    } else if name.eq_ignore_ascii_case("currval") {
        Some(ast::ScalarFunc::SequenceCurrent)
    } else if name.eq_ignore_ascii_case("setval") {
        Some(ast::ScalarFunc::SequenceSet)
    } else {
        None
    }
}

/// Analyze a sequence built-in call (`nextval`/`currval`/`setval`). The first argument is the
/// sequence name (text); `setval` additionally takes a `bigint` target value and an optional
/// `bool` `is_called`. The result type is `INT` (`BIGINT`). Argument count/type are validated here;
/// the actual advance/read against the engine happens at execution time (a sequence call is a
/// [`TypedExprKind::ScalarFunction`], resolved to a literal only where it is evaluated exactly once).
fn analyze_sequence_function(
    func: ast::ScalarFunc,
    name: &str,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // (min, max) argument counts and, positionally, the expected type of each argument.
    let (min, max, want): (usize, usize, &[ColumnType]) = match func {
        ast::ScalarFunc::SequenceNext | ast::ScalarFunc::SequenceCurrent => {
            (1, 1, &[ColumnType::Text])
        },
        ast::ScalarFunc::SequenceSet => {
            (2, 3, &[ColumnType::Text, ColumnType::Int, ColumnType::Bool])
        },
        _ => unreachable!("caller passes only sequence built-ins"),
    };
    if args.len() < min || args.len() > max {
        return Err(Error::ArityMismatch {
            context: format!("function `{name}`"),
            expected: max,
            found: args.len(),
        });
    }
    let mut typed = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let hint = want.get(i).copied();
        let expr = analyze_expr_agg(arg, scope, catalog, hint, aggregates.as_deref_mut())?;
        if let Some(want_ty) = hint
            && !assignable(want_ty, expr.ty)
        {
            return Err(Error::TypeMismatch {
                context: format!("argument {} to function `{name}`", i + 1),
                expected: want_ty,
                found: expr.ty,
            });
        }
        typed.push(expr);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction { func, args: typed },
        ty: ColumnType::Int,
    })
}

/// Maximum SQL-function inlining depth, so a (mutually) recursive function aborts rather than
/// inlining forever at analysis time.
const MAX_FN_INLINE_DEPTH: usize = 32;

thread_local! {
    /// Current SQL-function inlining depth on this thread.
    static FN_INLINE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Inline a SQL scalar function: substitute the call's argument expressions for the body's
/// `$1..$n` and analyze the resulting expression in place of the call, against the caller's scope.
/// Bounded by a recursion-depth guard so a recursive function definition cannot loop forever.
fn analyze_sql_function(
    name: &str,
    func: &crate::analyzer::FunctionDef,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    if args.len() != func.param_count {
        return Err(Error::ArityMismatch {
            context: format!("function `{name}`"),
            expected: func.param_count,
            found: args.len(),
        });
    }
    let depth = FN_INLINE_DEPTH.with(|d| {
        let n = d.get();
        d.set(n + 1);
        n
    });
    let result = (|| {
        if depth >= MAX_FN_INLINE_DEPTH {
            return Err(Error::Unsupported(format!(
                "function `{name}` inlining exceeded the recursion limit"
            )));
        }
        // The body parsed + validated to `SELECT <expr>` at creation; extract its expression.
        let ast::Statement::Select(select) = crate::parse(&func.body)? else {
            return Err(Error::Unsupported(format!(
                "function `{name}` body is not a SELECT"
            )));
        };
        let Some(ast::SelectItem::Expr { expr, .. }) = select.projection.into_iter().next() else {
            return Err(Error::Unsupported(format!(
                "function `{name}` body has no scalar expression"
            )));
        };
        let mut inlined = expr;
        crate::params::substitute_param_exprs(&mut inlined, args, &func.param_names);
        // The result type is the inlined body's type; the declared RETURNS is not re-checked here.
        analyze_expr_agg(&inlined, scope, catalog, None, aggregates)
    })();
    FN_INLINE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Analyze a scalar built-in [`ScalarFunc`] call: validate arity, type-check each
/// argument, and resolve the result type. An argument typed as a bare `NULL` literal is accepted in
/// any position. Most functions are NULL-strict at run time; the variadic `CONCAT`/`CONCAT_WS`
/// instead skip `NULL` arguments (handled in the executor).
/// An unconstrained `NUMERIC` (no declared precision/scale) for use as a signature parameter or
/// result type; it accepts any Int/Float/Numeric argument under [`assignable`].
const NUMERIC_ANY: ColumnType = ColumnType::Numeric {
    precision: 0,
    scale: 0,
};

#[allow(
    clippy::too_many_lines,
    reason = "flat dispatch + exhaustive signature table over the scalar-function set"
)]
pub(super) fn analyze_scalar_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ColumnType::{Int, Text};
    use ast::ScalarFunc as F;
    // GROUPING(key, ...) — super-aggregate indicator. Its arguments must be GROUP BY key
    // expressions, but the grouping sets are not in scope here; so we only type-check the arguments
    // against the source scope (they reference source columns, not aggregates) and carry them through
    // as a `ScalarFunction { Grouping, .. }` node. `rebase_onto_aggregation` later matches each
    // argument against the resolved `group_keys` and rewrites this node into the runtime bitmask
    // reference (or a constant `0` for a plain `GROUP BY`). Result is always `INT`.
    // NUSA_TYPEOF(expr) — the static SQL type name of the argument (NusaDB's `pg_typeof`). The type
    // is known here, so fold the call to a constant TEXT literal; the executor never sees this node.
    if matches!(func, F::NusaTypeof) {
        let [arg] = args else {
            return Err(Error::ArityMismatch {
                context: "function `nusa_typeof`".to_owned(),
                expected: 1,
                found: args.len(),
            });
        };
        let typed = analyze_expr_agg(arg, scope, catalog, None, aggregates.as_deref_mut())?;
        let name = crate::executor::ops::info_schema_data_type(typed.ty).to_owned();
        return Ok(TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Text(name)),
            ty: Text,
        });
    }
    if matches!(func, F::Grouping) {
        if args.is_empty() {
            return Err(Error::Unsupported(
                "GROUPING requires at least one argument".to_owned(),
            ));
        }
        let typed_args = args
            .iter()
            .map(|arg| analyze_expr(arg, scope, catalog, None))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction {
                func: F::Grouping,
                args: typed_args,
            },
            ty: Int,
        });
    }
    // The date/time functions take a temporal argument (one of four column types) and a result type
    // that can depend on it — neither expressible with the fixed `ScalarSig` table below — so they
    // are validated directly.
    if matches!(
        func,
        F::Extract | F::DateTrunc | F::Age | F::ToChar | F::ToDate | F::ToTimestamp | F::AtTimeZone
    ) {
        return analyze_temporal_function(func, args, scope, catalog, aggregates);
    }
    // The math functions are numeric-polymorphic (argument type Int/Float/Numeric, result either the
    // unified numeric type or always Float) — not expressible with the fixed table.
    if matches!(
        func,
        F::Abs
            | F::Round
            | F::Ceil
            | F::Floor
            | F::Sign
            | F::Mod
            | F::Power
            | F::Sqrt
            | F::Ln
            | F::Log
            | F::Exp
            | F::Sin
            | F::Cos
            | F::Tan
            | F::Asin
            | F::Acos
            | F::Atan
            | F::Atan2
            | F::Cot
            | F::Cbrt
            | F::Sinh
            | F::Cosh
            | F::Tanh
            | F::Asinh
            | F::Acosh
            | F::Atanh
            | F::Degrees
            | F::Radians
            | F::Trunc
    ) {
        return analyze_numeric_function(func, args, scope, catalog, aggregates);
    }
    // The conditional functions unify their arguments to a common type (any comparable type, not just
    // numeric) and skip NULLs at run time — handled like COALESCE, not via the fixed table.
    if matches!(func, F::Nullif | F::Greatest | F::Least) {
        return analyze_conditional_function(func, args, scope, catalog, aggregates);
    }
    // ARRAY_LENGTH(arr, dim) / ARRAY_TO_STRING(arr, sep) take an array of any element type — the
    // element type is polymorphic, so they are not expressible with the fixed table.
    if matches!(
        func,
        F::ArrayLength | F::ArrayLower | F::ArrayUpper | F::ArrayToString
    ) {
        return analyze_array_function(func, args, scope, catalog, aggregates);
    }
    // ARRAY_APPEND/PREPEND/CAT/POSITION/REMOVE take a polymorphic array — not table-shaped.
    if matches!(
        func,
        F::ArrayAppend
            | F::ArrayPrepend
            | F::ArrayCat
            | F::ArrayPosition
            | F::ArrayPositions
            | F::ArrayRemove
    ) {
        return analyze_array_mutate(func, args, scope, catalog, aggregates);
    }
    // ARRAY_REPLACE(arr, from, to) takes a polymorphic array plus two element-typed values (B-fn).
    if matches!(func, F::ArrayReplace) {
        return analyze_array_replace(func, args, scope, catalog, aggregates);
    }
    // CARDINALITY / ARRAY_NDIMS (→ INT) and ARRAY_DIMS (→ TEXT) take one array of any element type —
    // not expressible with the fixed table since the element type is polymorphic.
    if matches!(func, F::Cardinality | F::ArrayDims | F::ArrayNdims) {
        let name = func.name();
        let [arg_expr] = args else {
            return Err(Error::Unsupported(format!(
                "{name}() expects 1 argument, got {}",
                args.len()
            )));
        };
        let arg = analyze_expr_agg(arg_expr, scope, catalog, None, aggregates.as_deref_mut())?;
        if !matches!(arg.ty, ColumnType::Array(_)) && !is_null_literal(&arg) {
            return Err(Error::Unsupported(format!(
                "{name}() expects an array argument, got {:?}",
                arg.ty
            )));
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction {
                func,
                args: vec![arg],
            },
            ty: if func == F::ArrayDims { Text } else { Int },
        });
    }
    // ISFINITE(value) accepts a NUMERIC or temporal value of any of several types, so it is not
    // expressible with the fixed table; it always yields BOOL (B-fn).
    if matches!(func, F::IsFinite) {
        let name = func.name();
        let [arg_expr] = args else {
            return Err(Error::Unsupported(format!(
                "{name}() expects 1 argument, got {}",
                args.len()
            )));
        };
        let arg = analyze_expr_agg(arg_expr, scope, catalog, None, aggregates)?;
        let ok = matches!(
            arg.ty,
            ColumnType::Date
                | ColumnType::Timestamp
                | ColumnType::TimestampTz
                | ColumnType::Interval
                | ColumnType::Float
                | ColumnType::Numeric { .. }
        );
        if !ok && !is_null_literal(&arg) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument"),
                expected: ColumnType::Timestamp,
                found: arg.ty,
            });
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction {
                func,
                args: vec![arg],
            },
            ty: ColumnType::Bool,
        });
    }
    // NUM_NONNULLS / NUM_NULLS count their arguments by NULL-ness; the arguments may be any type and
    // they never propagate NULL, so they are not expressible with the fixed table.
    if matches!(func, F::NumNonNulls | F::NumNulls) {
        if args.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}() expects at least 1 argument",
                func.name()
            )));
        }
        let mut typed = Vec::with_capacity(args.len());
        for arg in args {
            // Only NULL-ness matters, so a bare `NULL` argument needs no real type — give it a
            // placeholder hint (`INT`) so it resolves instead of erroring as an ambiguous NULL.
            let hint = matches!(arg, ast::Expr::Literal(ast::Value::Null)).then_some(Int);
            typed.push(analyze_expr_agg(
                arg,
                scope,
                catalog,
                hint,
                aggregates.as_deref_mut(),
            )?);
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction { func, args: typed },
            ty: Int,
        });
    }
    // FORMAT(fmt, ...) takes a TEXT format string plus arguments of any type substituted into its
    // `%s`/`%I`/`%L` specifiers; the arguments are not table-shaped and NULL is not propagated, so it
    // is handled directly (B-fn).
    if matches!(func, F::Format) {
        let name = func.name();
        let Some((fmt_expr, rest)) = args.split_first() else {
            return Err(Error::Unsupported(format!(
                "{name}() expects at least 1 argument (the format string)"
            )));
        };
        let fmt = analyze_expr_agg(
            fmt_expr,
            scope,
            catalog,
            Some(Text),
            aggregates.as_deref_mut(),
        )?;
        if !matches!(fmt.ty, Text) && !is_null_literal(&fmt) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() format string"),
                expected: Text,
                found: fmt.ty,
            });
        }
        let mut typed = vec![fmt];
        for arg in rest {
            // Each substituted argument keeps its natural type; a bare NULL needs a placeholder hint
            // (the specifier decides how NULL renders), so type it as TEXT to resolve.
            let hint = matches!(arg, ast::Expr::Literal(ast::Value::Null)).then_some(Text);
            typed.push(analyze_expr_agg(
                arg,
                scope,
                catalog,
                hint,
                aggregates.as_deref_mut(),
            )?);
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction { func, args: typed },
            ty: Text,
        });
    }
    // The vector distance functions take two same-dimension VECTORs and return FLOAT — the dimension
    // is part of the type, so this is not expressible with the fixed table.
    if matches!(func, F::L2Distance | F::CosineDistance | F::InnerProduct) {
        return analyze_vector_function(func, args, scope, catalog, aggregates);
    }
    // TO_JSON / JSON_BUILD_OBJECT / JSON_BUILD_ARRAY take arguments of any type and return JSON — not
    // expressible with the fixed-type table.
    if matches!(func, F::ToJson | F::JsonBuildObject | F::JsonBuildArray) {
        return analyze_json_construct(func, args, scope, catalog, aggregates);
    }
    // ROW_TO_JSON expands a ROW(...) constructor into a JSON object; its single argument is a row,
    // not a scalar, so it is not expressible with the fixed-type table.
    if func == F::RowToJson {
        return analyze_row_to_json(args, scope, catalog, aggregates);
    }
    // LENGTH/OCTET_LENGTH/BIT_LENGTH are Text-or-BYTEA polymorphic (over BYTEA
    // they count octets, BIT_LENGTH 8x), and CONCAT/CONCAT_WS accept any textout-able scalar
    // — neither is expressible in the fixed table below.
    if matches!(
        func,
        F::Length | F::OctetLength | F::BitLength | F::Concat | F::ConcatWs
    ) {
        return analyze_text_polymorphic(func, func.name(), args, scope, catalog, aggregates);
    }
    // SUBSTRING is overloaded on its second argument's TYPE — `substring(s FROM 2)` is
    // positional while `substring(s FROM 'o.b')` is the POSIX-regex form —
    // which the fixed table cannot express.
    if func == F::Substring {
        return analyze_substring(args, scope, catalog, aggregates);
    }
    let sig = match func {
        // GROUPING(...) is resolved by the early `matches!(func, F::Grouping)` branch above (it has no
        // fixed scalar signature), so it never reaches this table.
        F::Grouping => unreachable!("GROUPING is handled before the scalar signature table"),
        // NUSA_TYPEOF is folded to a TEXT literal by the early `matches!(func, F::NusaTypeof)` branch
        // above, so it never reaches this table either.
        F::NusaTypeof => unreachable!("NUSA_TYPEOF is folded before the scalar signature table"),
        // The sequence built-ins arrive as generic function calls (`FunctionCall`) and are analyzed
        // by `analyze_sequence_function` before this typed-builtin table, so they never reach here.
        F::SequenceNext | F::SequenceCurrent | F::SequenceSet => {
            unreachable!("sequence built-ins are analyzed before the scalar signature table")
        },
        // ASCII takes one TEXT argument and returns INT (the LENGTH family is intercepted by
        // `analyze_text_polymorphic` above — Text-or-BYTEA).
        F::Ascii => ScalarSig::Fixed(&[Text], &[], Int),
        // GCD(a, b) / LCM(a, b) / DIV(a, b) take two INT arguments and return INT.
        F::Gcd | F::Lcm | F::Div => ScalarSig::Fixed(&[Int, Int], &[], Int),
        // FACTORIAL(n) takes one INT and returns INT.
        // FACTORIAL(n) and BIT_COUNT(n) both take one INT and return INT.
        F::Factorial | F::BitCount => ScalarSig::Fixed(&[Int], &[], Int),
        // WIDTH_BUCKET(operand, low, high, count) → INT histogram bucket. The numeric
        // operand/bounds accept INT/NUMERIC (they widen to FLOAT); the bucket count is an INT.
        F::WidthBucket => ScalarSig::Fixed(
            &[ColumnType::Float, ColumnType::Float, ColumnType::Float, Int],
            &[],
            Int,
        ),
        // STARTS_WITH(s, prefix) takes two TEXT arguments and returns BOOL.
        F::StartsWith => ScalarSig::Fixed(&[Text, Text], &[], ColumnType::Bool),
        // STRING_TO_ARRAY(s, sep) splits TEXT on TEXT into TEXT[].
        F::StringToArray => ScalarSig::Fixed(
            &[Text, Text],
            &[],
            ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
        ),
        // UPPER/LOWER/REVERSE/INITCAP, the hash digests SHA256/SHA512/MD5, the quoting helpers
        // QUOTE_LITERAL/QUOTE_IDENT, and `current_setting(name)` take one TEXT argument → TEXT.
        F::Upper
        | F::Lower
        | F::Reverse
        | F::Initcap
        | F::Sha256
        | F::Sha512
        | F::Md5
        | F::QuoteLiteral
        | F::QuoteIdent
        | F::CurrentSetting => ScalarSig::Fixed(&[Text], &[], Text),
        // CHR(n) maps an INT code point to a one-character TEXT; TO_HEX(n) renders an INT as a
        // lowercase hexadecimal TEXT string.
        F::Chr | F::ToHex => ScalarSig::Fixed(&[Int], &[], Text),
        // JSON inspection: JSON_TYPEOF(json) → TEXT; JSONB_PRETTY(json) → TEXT.
        F::JsonTypeof | F::JsonbPretty => ScalarSig::Fixed(&[ColumnType::Json], &[], Text),
        F::JsonArrayLength => ScalarSig::Fixed(&[ColumnType::Json], &[], Int),
        // JSONB_STRIP_NULLS(json) → JSON.
        F::JsonbStripNulls => ScalarSig::Fixed(&[ColumnType::Json], &[], ColumnType::Json),
        // JSONB_PATH_EXISTS(json, path) → BOOL.
        F::JsonbPathExists | F::JsonbExists => {
            ScalarSig::Fixed(&[ColumnType::Json, Text], &[], ColumnType::Bool)
        },
        // Full-text search (F1): TO_TSVECTOR/TO_TSQUERY/PLAINTO_TSQUERY([config,] text) → the
        // canonical tsvector/tsquery text form. The optional leading argument is the configuration;
        // with one argument the default configuration applies (rejected at evaluation until a
        // non-`simple` configuration exists).
        F::ToTsvector | F::ToTsquery | F::PlaintoTsquery => {
            ScalarSig::Fixed(&[Text], &[Text], Text)
        },
        // TS_RANK / TS_RANK_CD(tsvector, tsquery [, normalization INT]) → the relevance score as a
        // REAL. tsvector/tsquery are carried as TEXT; the optional third argument is the
        // normalization bit-mask.
        F::TsRank | F::TsRankCd => ScalarSig::Fixed(&[Text, Text], &[Int], ColumnType::Real),
        // RRF_SCORE(rank [, k]) → the Reciprocal Rank Fusion contribution 1/(k + rank) as FLOAT,
        // k defaulting to 60.
        F::RrfScore => ScalarSig::Fixed(&[Int], &[Int], ColumnType::Float),
        // JSONB_PATH_QUERY_FIRST(json, path) → JSON (the first match, or NULL).
        F::JsonbPathQueryFirst => {
            ScalarSig::Fixed(&[ColumnType::Json, Text], &[], ColumnType::Json)
        },
        // JSONB_SET(target, path TEXT[], new_value [, create_missing BOOL]) → JSON and
        // JSONB_INSERT(target, path TEXT[], new_value [, insert_after BOOL]) → JSON share the
        // same argument shape.
        F::JsonbSet | F::JsonbInsert => ScalarSig::Fixed(
            &[
                ColumnType::Json,
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
                ColumnType::Json,
            ],
            &[ColumnType::Bool],
            ColumnType::Json,
        ),
        // SUBSTRING is intercepted by `analyze_substring` above (its positional-vs-regex form
        // is dispatched on the second argument's type) and never reaches this table.
        F::Substring => {
            unreachable!("SUBSTRING is handled before the signature table")
        },
        F::Replace | F::Translate => ScalarSig::Fixed(&[Text, Text, Text], &[], Text),
        F::SplitPart => ScalarSig::Fixed(&[Text, Text, Int], &[], Text),
        // OVERLAY(s PLACING r FROM start [FOR len]) → TEXT, with an optional FOR length.
        F::Overlay => ScalarSig::Fixed(&[Text, Text, Int], &[Int], Text),
        // POSITION(sub IN s) and STRPOS(s, sub) both take two TEXT arguments → INT.
        F::Position | F::Strpos => ScalarSig::Fixed(&[Text, Text], &[], Int),
        // TO_NUMBER(text, format) → NUMERIC (B-fn).
        F::ToNumber => ScalarSig::Fixed(&[Text, Text], &[], NUMERIC_ANY),
        F::Lpad | F::Rpad => ScalarSig::Fixed(&[Text, Int], &[Text], Text),
        F::LTrim | F::RTrim | F::BTrim => ScalarSig::Fixed(&[Text], &[Text], Text),
        // LEFT(s, n) / RIGHT(s, n) / REPEAT(s, n) all take (TEXT, INT) → TEXT.
        F::Left | F::Right | F::Repeat => ScalarSig::Fixed(&[Text, Int], &[], Text),
        // REGEXP_REPLACE(s, pat, repl [, flags]); REGEXP_MATCH(s, pat [, flags]) → TEXT[].
        F::RegexpReplace => ScalarSig::Fixed(&[Text, Text, Text], &[Text], Text),
        // REGEXP_MATCH and REGEXP_SPLIT_TO_ARRAY(s, pattern [, flags]) both return TEXT[].
        F::RegexpMatch | F::RegexpSplitToArray => ScalarSig::Fixed(
            &[Text, Text],
            &[Text],
            ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
        ),
        // REGEXP_LIKE/COUNT/INSTR/SUBSTR(s, pattern [, flags]) — (TEXT, TEXT) + optional flags, with
        // a BOOL / INT / INT / TEXT result respectively.
        F::RegexpLike => ScalarSig::Fixed(&[Text, Text], &[Text], ColumnType::Bool),
        F::RegexpCount | F::RegexpInstr => ScalarSig::Fixed(&[Text, Text], &[Text], Int),
        F::RegexpSubstr => ScalarSig::Fixed(&[Text, Text], &[Text], Text),
        // CONCAT needs ≥1 value; CONCAT_WS needs at least its separator.
        // CONCAT/CONCAT_WS and the LENGTH family are intercepted by
        // `analyze_text_polymorphic` above and never reach
        // this table.
        F::Concat | F::ConcatWs | F::Length | F::OctetLength | F::BitLength => {
            unreachable!("text-polymorphic functions are handled before the signature table")
        },
        // Niladic clock built-ins resolved from the statement's wall clock.
        F::Now | F::CurrentTimestamp => ScalarSig::Fixed(&[], &[], ColumnType::TimestampTz),
        F::CurrentDate => ScalarSig::Fixed(&[], &[], ColumnType::Date),
        F::CurrentTime => ScalarSig::Fixed(&[], &[], ColumnType::Time),
        // Niladic session-user / system built-ins → TEXT. `current_setting` is grouped
        // above with the string functions.
        F::CurrentUser | F::SessionUser | F::Version | F::CurrentDatabase | F::CurrentSchema => {
            ScalarSig::Fixed(&[], &[], Text)
        },
        // PI() → FLOAT (niladic constant); RANDOM() → FLOAT (niladic, volatile).
        F::Pi | F::Random => ScalarSig::Fixed(&[], &[], ColumnType::Float),
        // SETSEED(x: FLOAT) → BOOL.
        // UUID_GENERATE_V4() → UUID (niladic, volatile).
        F::UuidGenerateV4 => ScalarSig::Fixed(&[], &[], ColumnType::Uuid),
        F::Setseed => ScalarSig::Fixed(&[ColumnType::Float], &[], ColumnType::Bool),
        // MAKE_DATE/MAKE_TIME/MAKE_TIMESTAMP build a temporal value from integer fields.
        F::MakeDate => ScalarSig::Fixed(&[Int, Int, Int], &[], ColumnType::Date),
        // MAKE_TIME(hour, min, sec) — seconds is FLOAT so fractional seconds are accepted.
        F::MakeTime => ScalarSig::Fixed(&[Int, Int, ColumnType::Float], &[], ColumnType::Time),
        F::MakeTimestamp => {
            ScalarSig::Fixed(&[Int, Int, Int, Int, Int, Int], &[], ColumnType::Timestamp)
        },
        // MAKE_INTERVAL(years, months, weeks, days, hours, mins, secs) — every field is optional and
        // positional, defaulting to 0; the seconds field is FLOAT.
        F::MakeInterval => ScalarSig::Fixed(
            &[],
            &[Int, Int, Int, Int, Int, Int, ColumnType::Float],
            ColumnType::Interval,
        ),
        // JUSTIFY_DAYS / JUSTIFY_HOURS / JUSTIFY_INTERVAL(interval) → interval (B-fn).
        F::JustifyDays | F::JustifyHours | F::JustifyInterval => {
            ScalarSig::Fixed(&[ColumnType::Interval], &[], ColumnType::Interval)
        },
        // SCALE / MIN_SCALE(numeric) → int; TRIM_SCALE(numeric) → numeric (B-fn). The unconstrained
        // `NUMERIC` param (`precision: 0`) accepts any Int/Float/Numeric argument.
        F::Scale | F::MinScale => ScalarSig::Fixed(&[NUMERIC_ANY], &[], Int),
        F::TrimScale => ScalarSig::Fixed(&[NUMERIC_ANY], &[], NUMERIC_ANY),
        // ENCODE(bytea, format) → text; DECODE(text, format) → bytea (B-fn).
        F::Encode => ScalarSig::Fixed(&[ColumnType::Bytes, Text], &[], Text),
        F::Decode => ScalarSig::Fixed(&[Text, Text], &[], ColumnType::Bytes),
        // DATE_BIN(stride INTERVAL, source TIMESTAMP, origin TIMESTAMP) → TIMESTAMP.
        F::DateBin => ScalarSig::Fixed(
            &[
                ColumnType::Interval,
                ColumnType::Timestamp,
                ColumnType::Timestamp,
            ],
            &[],
            ColumnType::Timestamp,
        ),
        // Handled above by `analyze_temporal_function` / `analyze_numeric_function` (their argument
        // and result types are not fixed-table shaped).
        F::Extract
        | F::DateTrunc
        | F::Age
        | F::AtTimeZone
        | F::ToChar
        | F::ToDate
        | F::ToTimestamp
        | F::Abs
        | F::Round
        | F::Ceil
        | F::Floor
        | F::Sign
        | F::Mod
        | F::Power
        | F::Sqrt
        | F::Ln
        | F::Log
        | F::Exp
        | F::Sin
        | F::Cos
        | F::Tan
        | F::Asin
        | F::Acos
        | F::Atan
        | F::Atan2
        | F::Cot
        | F::Cbrt
        | F::Sinh
        | F::Cosh
        | F::Tanh
        | F::Asinh
        | F::Acosh
        | F::Atanh
        | F::Degrees
        | F::Radians
        | F::Trunc
        | F::Nullif
        | F::Greatest
        | F::Least
        | F::Cardinality
        | F::ArrayDims
        | F::ArrayLength
        | F::ArrayLower
        | F::ArrayUpper
        | F::ArrayToString
        | F::ArrayAppend
        | F::ArrayPrepend
        | F::ArrayCat
        | F::ArrayPosition
        | F::ArrayRemove
        | F::ArrayReplace
        | F::ArrayPositions
        | F::ArrayNdims
        | F::L2Distance
        | F::CosineDistance
        | F::InnerProduct
        | F::ToJson
        | F::RowToJson
        | F::JsonBuildObject
        | F::JsonBuildArray
        | F::NumNonNulls
        | F::NumNulls
        | F::IsFinite
        | F::Format => {
            unreachable!(
                "temporal/numeric/conditional/array/vector/json-construct/null-count functions are \
                 dispatched before the ScalarSig table"
            )
        },
    };
    let name = func.name();
    // Per-argument types: indexed for `Fixed`, uniformly `Text` for `Variadic`.
    let ScalarSig::Fixed(required, optional, result) = sig;
    let (min, max) = (required.len(), required.len() + optional.len());
    if args.len() < min || args.len() > max {
        let arity = match (min, max) {
            (lo, hi) if lo == hi => lo.to_string(),
            (lo, usize::MAX) => format!("at least {lo}"),
            (lo, hi) => format!("{lo}..={hi}"),
        };
        return Err(Error::Unsupported(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    let expected_at = |i: usize| {
        required
            .get(i)
            .or_else(|| optional.get(i - required.len()))
            .copied()
            .unwrap_or(Text)
    };
    let mut typed_args = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let expected = expected_at(i);
        let typed = analyze_expr_agg(
            arg,
            scope,
            catalog,
            Some(expected),
            aggregates.as_deref_mut(),
        )?;
        // A FLOAT parameter also accepts an INT or NUMERIC argument (coerced to f64 at eval) — the
        // same widening `assignable` allows, and needed since a plain decimal literal now types as
        // NUMERIC, e.g. `SETSEED(0.5)`. An unconstrained NUMERIC parameter likewise accepts
        // any NUMERIC (regardless of declared precision/scale) or an INT, e.g. `SCALE(12.34)`.
        let coercible = (expected == ColumnType::Float
            && matches!(typed.ty, ColumnType::Int | ColumnType::Numeric { .. }))
            || (matches!(expected, ColumnType::Numeric { .. })
                && matches!(typed.ty, ColumnType::Int | ColumnType::Numeric { .. }));
        if typed.ty != expected && !coercible && !is_null_literal(&typed) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument {}", i + 1),
                expected,
                found: typed.ty,
            });
        }
        typed_args.push(typed);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: result,
    })
}

/// Analyze `TO_JSON(value)` / `JSON_BUILD_OBJECT(k1, v1, ...)` / `JSON_BUILD_ARRAY(v1, ...)`.
/// Arguments are of any type (each kept at its natural type; the executor serializes to JSON);
/// `JSON_BUILD_OBJECT` requires an even argument count, `JSON_BUILD_ARRAY` accepts any. Result is
/// `JSON`.
fn analyze_json_construct(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    if func == F::ToJson && args.len() != 1 {
        return Err(Error::Unsupported(format!(
            "{name}() expects 1 argument, got {}",
            args.len()
        )));
    }
    if func == F::JsonBuildObject && !args.len().is_multiple_of(2) {
        return Err(Error::Unsupported(format!(
            "{name}() requires an even number of arguments (key, value pairs), got {}",
            args.len()
        )));
    }
    let mut typed = Vec::with_capacity(args.len());
    for arg in args {
        typed.push(analyze_expr_agg(
            arg,
            scope,
            catalog,
            None,
            aggregates.as_deref_mut(),
        )?);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction { func, args: typed },
        ty: ColumnType::Json,
    })
}

/// Analyze `ROW_TO_JSON(...)`. Two forms are supported, both lowered to an interleaved
/// `key, value, key, value, …` argument list that the executor walks in order:
///
/// - `row_to_json(row(a, b))` / `row_to_json((a, b))` — a `ROW(...)` constructor, serialized with
///   positional field names `f1`, `f2`, ….
/// - `row_to_json(t)` — a bare table or alias in scope, expanded to every one of its columns in
///   order, keyed by the real column name (the primary use).
///
/// Result is `JSON`. Any other argument (a scalar, a non-relation name) is rejected with a clear
/// message rather than silently mis-serialized.
fn analyze_row_to_json(
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let [arg] = args else {
        return Err(Error::Unsupported(format!(
            "row_to_json() expects 1 argument, got {}",
            args.len()
        )));
    };
    let text_key = |name: String| TypedExpr {
        kind: TypedExprKind::Literal(ast::Value::Text(name)),
        ty: ColumnType::Text,
    };
    let fields = match arg {
        // ROW(...) / (a, b, …): positional field names f1, f2, ….
        ast::Expr::Row(items) => {
            let mut typed = Vec::with_capacity(items.len() * 2);
            for (i, item) in items.iter().enumerate() {
                typed.push(text_key(format!("f{}", i + 1)));
                typed.push(analyze_expr_agg(
                    item,
                    scope,
                    catalog,
                    None,
                    aggregates.as_deref_mut(),
                )?);
            }
            typed
        },
        // A bare relation name: expand to every column of that table/alias, keyed by column name.
        ast::Expr::Column(name)
            if scope
                .iter()
                .any(|c| &c.qualifier == name && !c.qualified_only) =>
        {
            let mut typed = Vec::new();
            for (index, col) in scope.iter().enumerate() {
                if &col.qualifier != name || col.qualified_only {
                    continue;
                }
                typed.push(text_key(col.def.name.clone()));
                typed.push(TypedExpr {
                    kind: TypedExprKind::Column(index),
                    ty: col.def.ty.physical(),
                });
            }
            typed
        },
        _ => {
            return Err(Error::Unsupported(
                "row_to_json() expects a ROW(...) constructor or a table/alias in the FROM clause \
                 (e.g. row_to_json(row(a, b)) or row_to_json(t))"
                    .to_owned(),
            ));
        },
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func: ast::ScalarFunc::RowToJson,
            args: fields,
        },
        ty: ColumnType::Json,
    })
}

/// Analyze `ARRAY_LENGTH(arr, dim)` / `ARRAY_TO_STRING(arr, sep)`. The first argument is an
/// array of any element type; the second is an `INT` dimension (`ARRAY_LENGTH`, result `INT`) or a
/// `TEXT` separator (`ARRAY_TO_STRING`, result `TEXT`).
fn analyze_array_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let [arr_expr, second_expr] = args else {
        return Err(Error::Unsupported(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    };
    let arr = analyze_expr_agg(arr_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    if !matches!(arr.ty, ColumnType::Array(_)) && !is_null_literal(&arr) {
        return Err(Error::Unsupported(format!(
            "{name}() expects an array first argument, got {:?}",
            arr.ty
        )));
    }
    let (second_ty, result) = if func == F::ArrayToString {
        (ColumnType::Text, ColumnType::Text)
    } else {
        // ARRAY_LENGTH / ARRAY_LOWER / ARRAY_UPPER take a dimension INT and return INT.
        (ColumnType::Int, ColumnType::Int)
    };
    let second = analyze_expr_agg(second_expr, scope, catalog, Some(second_ty), aggregates)?;
    if second.ty != second_ty && !is_null_literal(&second) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() second argument"),
            expected: second_ty,
            found: second.ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![arr, second],
        },
        ty: result,
    })
}

/// Analyze `ARRAY_APPEND(arr, elem)` / `ARRAY_PREPEND(elem, arr)` / `ARRAY_CAT(a, b)`: the
/// result keeps the array's element type, and an appended/prepended element (or the second array's
/// element type) must be assignable to it.
fn analyze_array_mutate(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let [a_expr, b_expr] = args else {
        return Err(Error::Unsupported(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    };
    // Analyze the array operand first (so its element type can hint a bare-NULL element), then the
    // other operand. `args` is kept in the original positional order for the executor.
    let (array_expr, other_expr, array_is_first, elem_label) = match func {
        F::ArrayAppend => (a_expr, b_expr, true, "appended element"),
        F::ArrayPrepend => (b_expr, a_expr, false, "prepended element"),
        F::ArrayCat => (a_expr, b_expr, true, "second array"),
        F::ArrayPosition | F::ArrayPositions | F::ArrayRemove => (a_expr, b_expr, true, "element"),
        _ => unreachable!("non-array-mutate function routed to analyze_array_mutate"),
    };
    let array = analyze_expr_agg(array_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let ColumnType::Array(array_elem) = array.ty else {
        return Err(Error::Unsupported(format!(
            "{name}() expects an array argument, got {:?}",
            array.ty
        )));
    };
    // CAT's other operand is the same array type; APPEND/PREPEND's is the scalar element type.
    let expected = if func == F::ArrayCat {
        array.ty
    } else {
        array_elem.column_type()
    };
    let other = analyze_expr_agg(other_expr, scope, catalog, Some(expected), aggregates)?;
    if other.ty != expected && !is_null_literal(&other) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() {elem_label}"),
            expected,
            found: other.ty,
        });
    }
    // Restore positional order: `array_is_first` is true except for PREPEND (element, array).
    let typed_args = if array_is_first {
        vec![array, other]
    } else {
        vec![other, array]
    };
    // ARRAY_POSITION returns the 1-based index as INT; ARRAY_POSITIONS an INT[] of all indexes; the
    // others return the (transformed) array.
    let result_ty = match func {
        F::ArrayPosition => ColumnType::Int,
        F::ArrayPositions => ColumnType::Array(nusadb_core::engine::ArrayElem::Int),
        _ => ColumnType::Array(array_elem),
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: result_ty,
    })
}

/// Analyze `ARRAY_REPLACE(arr, from, to)`: a polymorphic array plus two values of its element type;
/// the result keeps `arr`'s array type (B-fn).
fn analyze_array_replace(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    let [arr_expr, from_expr, to_expr] = args else {
        return Err(Error::Unsupported(format!(
            "{name}() expects 3 arguments, got {}",
            args.len()
        )));
    };
    let array = analyze_expr_agg(arr_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let ColumnType::Array(array_elem) = array.ty else {
        return Err(Error::Unsupported(format!(
            "{name}() expects an array argument, got {:?}",
            array.ty
        )));
    };
    let elem_ty = array_elem.column_type();
    let mut typed = vec![array];
    for (expr, label) in [(from_expr, "from"), (to_expr, "to")] {
        let v = analyze_expr_agg(
            expr,
            scope,
            catalog,
            Some(elem_ty),
            aggregates.as_deref_mut(),
        )?;
        if v.ty != elem_ty && !is_null_literal(&v) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() {label} element"),
                expected: elem_ty,
                found: v.ty,
            });
        }
        typed.push(v);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction { func, args: typed },
        ty: ColumnType::Array(array_elem),
    })
}

/// Unify two numeric types for a type-preserving math result: `FLOAT` dominates (its inexactness is
/// contagious), then `NUMERIC` (exact) over `INT`, else `INT` — mirroring `check_arithmetic`.
fn widen_numeric(a: ColumnType, b: ColumnType) -> ColumnType {
    use ColumnType::{Float, Numeric};
    if a == Float || b == Float {
        Float
    } else if matches!(a, Numeric { .. }) || matches!(b, Numeric { .. }) {
        Numeric {
            precision: 0,
            scale: 0,
        }
    } else {
        // Unifying two integer branches (CASE/UNION) takes the wider width.
        wider_int(a, b)
    }
}

/// Analyze a numeric math built-in. Every argument must be numeric (`INT`/`FLOAT`/
/// `NUMERIC`) or a bare `NULL` (typed `FLOAT`, so e.g. `ABS(NULL)` is a `FLOAT` `NULL` rather than
/// ambiguous). Type-preserving functions (`ABS`/`CEIL`/`FLOOR`/`SIGN`/`ROUND`/`MOD`) return the
/// unified numeric type; the power/transcendental/trig functions always return `FLOAT`.
fn analyze_numeric_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ColumnType::{Float, Int};
    use ast::ScalarFunc as F;
    // (min arity, max arity, result is always FLOAT).
    let (min, max, force_float) = match func {
        F::Abs | F::Ceil | F::Floor | F::Sign => (1, 1, false),
        F::Round | F::Trunc => (1, 2, false),
        F::Mod => (2, 2, false),
        F::Power | F::Atan2 => (2, 2, true),
        F::Log => (1, 2, true),
        F::Sqrt
        | F::Ln
        | F::Exp
        | F::Sin
        | F::Cos
        | F::Tan
        | F::Asin
        | F::Acos
        | F::Atan
        | F::Cot
        | F::Cbrt
        | F::Sinh
        | F::Cosh
        | F::Tanh
        | F::Asinh
        | F::Acosh
        | F::Atanh
        | F::Degrees
        | F::Radians => (1, 1, true),
        _ => unreachable!("non-numeric function routed to analyze_numeric_function"),
    };
    let name = func.name();
    if args.len() < min || args.len() > max {
        let arity = if min == max {
            min.to_string()
        } else {
            format!("{min}..={max}")
        };
        return Err(Error::Unsupported(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    let mut typed_args = Vec::with_capacity(args.len());
    let mut unified = Int;
    for (i, arg) in args.iter().enumerate() {
        let typed = analyze_expr_agg(arg, scope, catalog, Some(Float), aggregates.as_deref_mut())?;
        if !is_numeric(typed.ty) && !is_null_literal(&typed) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument {}", i + 1),
                expected: Float,
                found: typed.ty,
            });
        }
        unified = widen_numeric(unified, typed.ty);
        typed_args.push(typed);
    }
    // ROUND's / TRUNC's optional second argument is an integer count of decimal places.
    if matches!(func, F::Round | F::Trunc)
        && let Some(d) = typed_args.get(1)
        && d.ty != Int
        && !is_null_literal(d)
    {
        return Err(Error::TypeMismatch {
            context: format!("{name}() decimal places"),
            expected: Int,
            found: d.ty,
        });
    }
    let result = if force_float { Float } else { unified };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: result,
    })
}

/// Analyze a conditional built-in (`NULLIF`, `GREATEST`, `LEAST`). Arguments unify to a
/// common type (like `COALESCE`): the running type is threaded as the NULL hint so a bare `NULL`
/// adopts its siblings' type, and the result is that unified type. `NULLIF` takes exactly two
/// arguments; `GREATEST`/`LEAST` take one or more.
fn analyze_conditional_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let (min, max) = if func == F::Nullif {
        (2, 2)
    } else {
        (1, usize::MAX)
    };
    if args.len() < min || args.len() > max {
        let arity = if min == max {
            min.to_string()
        } else {
            format!("at least {min}")
        };
        return Err(Error::Unsupported(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    // Resolve the non-NULL arguments to a common type first and defer the bare `NULL` literals,
    // typing them from that result afterwards — exactly like COALESCE. This lets a leading NULL
    // infer from a later argument (`GREATEST(NULL, 5)` is INT) and makes an all-NULL call a plain
    // untyped NULL (→ TEXT, the standard unknown -> text rule) instead of an ambiguous-type error.
    let mut slots: Vec<Option<TypedExpr>> = Vec::with_capacity(args.len());
    let mut result_ty: Option<ColumnType> = None;
    let mut deferred: Vec<usize> = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if matches!(arg, ast::Expr::Literal(ast::Value::Null)) {
            deferred.push(i);
            slots.push(None);
            continue;
        }
        let typed = analyze_expr_agg(arg, scope, catalog, result_ty, aggregates.as_deref_mut())?;
        result_ty = Some(unify_result_ty(result_ty, typed.ty, name)?);
        slots.push(Some(typed));
    }
    let resolved = result_ty.unwrap_or(ColumnType::Text);
    for i in deferred {
        if let Some(slot) = slots.get_mut(i) {
            *slot = Some(analyze_null(Some(resolved))?);
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: slots.into_iter().flatten().collect(),
        },
        ty: resolved,
    })
}

/// Analyze a vector distance function `l2_distance` / `cosine_distance` / `inner_product`:
/// exactly two `VECTOR`s of the same dimension, returning `FLOAT`. The dimension is part of the type,
/// so this cannot use the fixed-signature table (mirrors the `<=>` operator's [`check_vector_distance`]).
/// A `NULL` argument is allowed (the call evaluates to `NULL`).
fn analyze_vector_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    let [a_expr, b_expr] = args else {
        return Err(Error::Unsupported(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    };
    let a = analyze_expr_agg(a_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let b = analyze_expr_agg(b_expr, scope, catalog, None, aggregates)?;
    // Both must be same-dimension VECTORs; a bare NULL is allowed (typed from its sibling, else Bool).
    let ok = matches!((a.ty, b.ty), (ColumnType::Vector(x), ColumnType::Vector(y)) if x == y)
        || is_null_literal(&a)
        || is_null_literal(&b);
    if !ok {
        return Err(Error::TypeMismatch {
            context: format!("{name}() arguments"),
            expected: a.ty,
            found: b.ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![a, b],
        },
        ty: ColumnType::Float,
    })
}

/// Analyze an `ARRAY[a, b, ...]` constructor. Elements unify to one common scalar type (the
/// running type is threaded as the NULL hint, like `COALESCE`), which must be a valid array element
/// type; the result is the `ColumnType::Array` of that element. An empty `ARRAY[]` has no inferable
/// element type and is rejected.
fn analyze_array_literal(
    elems: &[ast::Expr],
    hint: Option<ColumnType>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let mut typed = Vec::with_capacity(elems.len());
    let mut elem_ty: Option<ColumnType> = None;
    for elem in elems {
        let t = analyze_expr_agg(elem, scope, catalog, elem_ty, aggregates.as_deref_mut())?;
        elem_ty = Some(unify_result_ty(elem_ty, t.ty, "ARRAY element")?);
        typed.push(t);
    }
    // An empty `ARRAY[]` has no element to infer from, so it takes the element type from an enclosing
    // type hint (e.g. `CAST(ARRAY[] AS INT[])` or assignment to an array column); without one its
    // element type is genuinely unknowable and is rejected.
    let elem_col_ty = elem_ty
        .or_else(|| match hint {
            Some(ColumnType::Array(elem)) => Some(elem.column_type()),
            _ => None,
        })
        .ok_or_else(|| {
            Error::Unsupported(
                "empty ARRAY[] has no inferable element type — add an explicit cast".to_owned(),
            )
        })?;
    // Map the unified element type to a storable array element. NUMERIC is now a supported element
    // type (exact decimals — `ARRAY[1, 2.0]` is `NUMERIC[]`), so only the genuinely
    // non-element types (nested arrays, JSON, BYTES, …) are rejected here.
    let elem = nusadb_core::engine::ArrayElem::from_column_type(elem_col_ty).ok_or_else(|| {
        Error::Unsupported(format!(
            "ARRAY of {elem_col_ty:?} elements is not supported"
        ))
    })?;
    Ok(TypedExpr {
        kind: TypedExprKind::ArrayLiteral(typed),
        ty: ColumnType::Array(elem),
    })
}

/// Analyze a `base[index]` subscript: `base` must be array-typed and `index` must be `Int`.
/// The result is the array's element type (`NULL` at run time for an out-of-range or `NULL` index).
fn analyze_subscript(
    base: &ast::Expr,
    index: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let base_t = analyze_expr_agg(base, scope, catalog, None, aggregates.as_deref_mut())?;
    let ColumnType::Array(elem) = base_t.ty else {
        return Err(Error::TypeMismatch {
            context: "array subscript base".to_owned(),
            expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Int),
            found: base_t.ty,
        });
    };
    let index_t = analyze_expr_agg(index, scope, catalog, Some(ColumnType::Int), aggregates)?;
    if index_t.ty != ColumnType::Int && !is_null_literal(&index_t) {
        return Err(Error::TypeMismatch {
            context: "array subscript index".to_owned(),
            expected: ColumnType::Int,
            found: index_t.ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Subscript {
            base: Box::new(base_t),
            index: Box::new(index_t),
        },
        ty: elem.column_type(),
    })
}

/// Analyze a `base[lower:upper]` array slice (B-fn): `base` must be array-typed and each present
/// bound must be `Int`. The result is the *array* type itself (a slice of an array is an array). Each
/// bound is optional (`a[2:]`, `a[:3]`, `a[:]`).
fn analyze_array_slice(
    base: &ast::Expr,
    lower: Option<&ast::Expr>,
    upper: Option<&ast::Expr>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let base_t = analyze_expr_agg(base, scope, catalog, None, aggregates.as_deref_mut())?;
    if !matches!(base_t.ty, ColumnType::Array(_)) {
        return Err(Error::TypeMismatch {
            context: "array slice base".to_owned(),
            expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Int),
            found: base_t.ty,
        });
    }
    let array_ty = base_t.ty;
    let mut bound = |expr: Option<&ast::Expr>| -> Result<Option<Box<TypedExpr>>, Error> {
        let Some(expr) = expr else { return Ok(None) };
        let t = analyze_expr_agg(
            expr,
            scope,
            catalog,
            Some(ColumnType::Int),
            aggregates.as_deref_mut(),
        )?;
        if t.ty != ColumnType::Int && !is_null_literal(&t) {
            return Err(Error::TypeMismatch {
                context: "array slice bound".to_owned(),
                expected: ColumnType::Int,
                found: t.ty,
            });
        }
        Ok(Some(Box::new(t)))
    };
    let lower_t = bound(lower)?;
    let upper_t = bound(upper)?;
    Ok(TypedExpr {
        kind: TypedExprKind::ArraySlice {
            base: Box::new(base_t),
            lower: lower_t,
            upper: upper_t,
        },
        ty: array_ty,
    })
}

/// Analyze an ordered-set aggregate `f(args) WITHIN GROUP (ORDER BY key)` — `PERCENTILE_CONT`,
/// `PERCENTILE_DISC`, or `MODE`. The single `ORDER BY` key becomes the aggregate's `arg`;
/// a percentile's fraction is a constant in `[0, 1]`. The `ORDER BY` must be a single ascending key
/// without a `NULLS` clause (v1). Registered into the aggregate sink like an ordinary aggregate.
fn analyze_within_group(
    wg: &ast::WithinGroup,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::AggregateFunc as F;
    let Some(sink) = aggregates else {
        return Err(Error::Unsupported(
            "ordered-set aggregates are only allowed in a SELECT projection, HAVING, or ORDER BY"
                .to_owned(),
        ));
    };
    let func = match wg.func.as_str() {
        "percentile_cont" => F::PercentileCont,
        "percentile_disc" => F::PercentileDisc,
        "mode" => F::Mode,
        other => {
            return Err(Error::Unsupported(format!(
                "ordered-set aggregate `{other}` is not supported (PERCENTILE_CONT / \
                 PERCENTILE_DISC / MODE only)"
            )));
        },
    };
    // WITHIN GROUP takes exactly one ORDER BY key. `DESC` reverses the ordered set; a `NULLS
    // FIRST/LAST` clause is accepted but has no effect (NULL ordering values are excluded from the
    // set, so their placement cannot change the percentile/mode).
    let [order_item] = wg.order_by.as_slice() else {
        return Err(Error::Unsupported(
            "WITHIN GROUP requires exactly one ORDER BY expression".to_owned(),
        ));
    };
    let ordered_set_descending = !order_item.ascending;
    // The ordered value references source rows, not aggregates (no nested aggregate sink).
    let order_value = analyze_expr(&order_item.expr, scope, catalog, None)?;
    let (fraction, result_ty) = match func {
        F::Mode => {
            if !wg.args.is_empty() {
                return Err(Error::Unsupported(
                    "MODE() takes no direct arguments".to_owned(),
                ));
            }
            (None, order_value.ty)
        },
        F::PercentileCont | F::PercentileDisc => {
            let [fraction_expr] = wg.args.as_slice() else {
                return Err(Error::Unsupported(
                    "PERCENTILE_CONT / PERCENTILE_DISC take exactly one fraction argument"
                        .to_owned(),
                ));
            };
            // Per-percentile result type (shared by the scalar and array-of-fractions forms).
            let elem_ty = if func == F::PercentileCont {
                // Continuous interpolation requires a numeric ordering value and yields FLOAT.
                if !matches!(
                    order_value.ty,
                    ColumnType::Int | ColumnType::Float | ColumnType::Numeric { .. }
                ) {
                    return Err(Error::TypeMismatch {
                        context: "PERCENTILE_CONT ordering value".to_owned(),
                        expected: ColumnType::Float,
                        found: order_value.ty,
                    });
                }
                ColumnType::Float
            } else {
                // Discrete percentile returns an actual element of the ordered set.
                order_value.ty
            };
            // Array-of-fractions form: `PERCENTILE_CONT(ARRAY[f1, f2, ...]) WITHIN GROUP (...)` returns
            // an array with one percentile per fraction. Desugared in a helper into one scalar
            // percentile aggregate per fraction, wrapped in an array constructor over their refs.
            if let ast::Expr::ArrayLiteral(items) = fraction_expr {
                return analyze_percentile_array(
                    func,
                    items,
                    &order_value,
                    elem_ty,
                    ordered_set_descending,
                    sink,
                );
            }
            let fraction = const_fraction(fraction_expr)?;
            (Some(fraction), elem_ty)
        },
        _ => unreachable!("guarded by the func match above"),
    };
    let idx = sink.len();
    sink.push(AggregateCall {
        func,
        arg: Some(order_value),
        result_ty,
        distinct: false,
        fraction,
        ordered_set_descending,
        filter: None,
        separator: None,
        arg2: None,
        order_by: Vec::new(),
        grouping_args: Vec::new(),
    });
    Ok(TypedExpr {
        kind: TypedExprKind::AggregateRef(idx),
        ty: result_ty,
    })
}

/// Desugar the array-of-fractions percentile form `PERCENTILE_CONT/DISC(ARRAY[f1, f2, ...]) WITHIN
/// GROUP (ORDER BY x)` into one scalar percentile aggregate per fraction, returning an array
/// constructor over their refs (result element type `elem_ty`, already validated for the func). This
/// reuses the scalar percentile execution path with no new aggregate state or executor arm — the
/// `ArrayLiteral` evaluator collects each resolved `AggregateRef` into the result array, in order.
fn analyze_percentile_array(
    func: ast::AggregateFunc,
    fraction_items: &[ast::Expr],
    order_value: &TypedExpr,
    elem_ty: ColumnType,
    ordered_set_descending: bool,
    sink: &mut Vec<AggregateCall>,
) -> Result<TypedExpr, Error> {
    let Some(array_elem) = nusadb_core::engine::ArrayElem::from_column_type(elem_ty) else {
        return Err(Error::Unsupported(
            "PERCENTILE_DISC over this ordering type does not support the array-of-fractions form"
                .to_owned(),
        ));
    };
    let mut refs = Vec::with_capacity(fraction_items.len());
    for item in fraction_items {
        let fraction = const_fraction(item)?;
        let idx = sink.len();
        sink.push(AggregateCall {
            func,
            arg: Some(order_value.clone()),
            result_ty: elem_ty,
            distinct: false,
            fraction: Some(fraction),
            ordered_set_descending,
            filter: None,
            separator: None,
            arg2: None,
            order_by: Vec::new(),
            grouping_args: Vec::new(),
        });
        refs.push(TypedExpr {
            kind: TypedExprKind::AggregateRef(idx),
            ty: elem_ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ArrayLiteral(refs),
        ty: ColumnType::Array(array_elem),
    })
}

/// A constant percentile fraction: a numeric literal in `[0, 1]`.
fn const_fraction(expr: &ast::Expr) -> Result<f64, Error> {
    let fraction = match expr {
        ast::Expr::Literal(ast::Value::Float(f)) => *f,
        #[allow(
            clippy::cast_precision_loss,
            reason = "a 0/1 integer fraction converts exactly; larger values are rejected below"
        )]
        ast::Expr::Literal(ast::Value::Int(i)) => *i as f64,
        ast::Expr::Literal(ast::Value::Numeric(d)) => d.to_f64(),
        _ => {
            return Err(Error::Unsupported(
                "the PERCENTILE fraction must be a constant numeric literal".to_owned(),
            ));
        },
    };
    if !(0.0..=1.0).contains(&fraction) {
        return Err(Error::InvalidValue {
            ty: ColumnType::Float,
            value: format!("percentile fraction {fraction} is outside [0, 1]"),
        });
    }
    Ok(fraction)
}

/// Analyze a date/time built-in (`EXTRACT`, `DATE_TRUNC`, `AGE`, `AT TIME ZONE`) whose
/// argument and result types depend on the temporal source, which the fixed [`ScalarSig`] table
/// cannot express. The field keyword is carried as a typed lowercase `Text` literal so the executor
/// can read it.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-temporal-function dispatch; length tracks the function set"
)]
fn analyze_temporal_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ColumnType::{Date, Float, Interval, Text, Time, Timestamp, TimestampTz};
    use ast::ScalarFunc as F;
    let name = func.name();
    let is_temporal = |ty| matches!(ty, Date | Time | Timestamp | TimestampTz);
    // Rebuild a field-carrying call: the field name as a typed `Text` literal, then the source.
    let field_call = |src: TypedExpr, result, field| TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![
                TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Text(field)),
                    ty: ColumnType::Text,
                },
                src,
            ],
        },
        ty: result,
    };
    match func {
        F::Extract | F::DateTrunc => {
            let (field_expr, source_expr) = expect_two_args(args, name)?;
            let field = expect_field_literal(field_expr, name)?;
            let valid_field = match func {
                F::Extract => is_extract_field(&field),
                _ => is_trunc_field(&field),
            };
            if !valid_field {
                return Err(Error::Unsupported(format!(
                    "{name}() field `{field}` is not supported"
                )));
            }
            let source =
                analyze_expr_agg(source_expr, scope, catalog, None, aggregates.as_deref_mut())?;
            // EXTRACT accepts any temporal source; DATE_TRUNC truncates a timestamp.
            let ok_source = match func {
                // EXTRACT also reads the fields of an INTERVAL (e.g. `epoch`, `day`, `hour`).
                F::Extract => is_temporal(source.ty) || source.ty == Interval,
                _ => matches!(source.ty, Timestamp | TimestampTz),
            };
            if !ok_source {
                return Err(Error::TypeMismatch {
                    context: format!("{name}() source"),
                    expected: Timestamp,
                    found: source.ty,
                });
            }
            // EXTRACT → FLOAT; DATE_TRUNC preserves the source's temporal type.
            let result = if func == F::Extract { Float } else { source.ty };
            Ok(field_call(source, result, field))
        },
        F::AtTimeZone => {
            // `<value> AT TIME ZONE <zone>`: value is TIMESTAMP or TIMESTAMPTZ; the zone is a text
            // name/offset (`'UTC'`, `'+05:00'`) or an INTERVAL fixed offset (`INTERVAL '5 hours'`).
            // The result flips the time-zone-awareness — TIMESTAMP → TIMESTAMPTZ, TIMESTAMPTZ → TIMESTAMP.
            let [value, zone] = args else {
                return Err(Error::Unsupported(
                    "AT TIME ZONE expects a value and a zone".to_owned(),
                ));
            };
            let value = analyze_expr_agg(value, scope, catalog, None, aggregates.as_deref_mut())?;
            let result_ty = match value.ty {
                Timestamp => TimestampTz,
                TimestampTz => Timestamp,
                _ if is_null_literal(&value) => TimestampTz,
                other => {
                    return Err(Error::TypeMismatch {
                        context: "AT TIME ZONE value".to_owned(),
                        expected: Timestamp,
                        found: other,
                    });
                },
            };
            let zone = analyze_expr_agg(zone, scope, catalog, Some(Text), aggregates)?;
            if !matches!(zone.ty, Text | ColumnType::Interval) && !is_null_literal(&zone) {
                return Err(Error::TypeMismatch {
                    context: "AT TIME ZONE zone".to_owned(),
                    expected: Text,
                    found: zone.ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::ScalarFunction {
                    func,
                    args: vec![value, zone],
                },
                ty: result_ty,
            })
        },
        F::Age => {
            if args.is_empty() || args.len() > 2 {
                return Err(Error::Unsupported(format!(
                    "{name}() expects 1 or 2 argument(s), got {}",
                    args.len()
                )));
            }
            let mut typed = Vec::with_capacity(args.len());
            for arg in args {
                let t = analyze_expr_agg(arg, scope, catalog, None, aggregates.as_deref_mut())?;
                if !matches!(t.ty, Date | Timestamp | TimestampTz) && !is_null_literal(&t) {
                    return Err(Error::TypeMismatch {
                        context: format!("{name}() argument"),
                        expected: Timestamp,
                        found: t.ty,
                    });
                }
                typed.push(t);
            }
            Ok(TypedExpr {
                kind: TypedExprKind::ScalarFunction { func, args: typed },
                ty: Interval,
            })
        },
        // `to_timestamp(epoch)` (a single numeric argument) reads UNIX epoch seconds → TIMESTAMPTZ;
        // the two-argument `to_timestamp(text, format)` keeps the text-parsing path.
        F::ToTimestamp if matches!(args, [_]) => {
            let [epoch] = args else {
                return Err(Error::Unsupported(format!("{name}() expects 1 argument")));
            };
            let arg = analyze_expr_agg(epoch, scope, catalog, None, aggregates.as_deref_mut())?;
            if !matches!(arg.ty, Float | ColumnType::Int | ColumnType::Numeric { .. })
                && !is_null_literal(&arg)
            {
                return Err(Error::TypeMismatch {
                    context: format!("{name}() epoch"),
                    expected: Float,
                    found: arg.ty,
                });
            }
            Ok(TypedExpr {
                kind: TypedExprKind::ScalarFunction {
                    func,
                    args: vec![arg],
                },
                ty: TimestampTz,
            })
        },
        F::ToChar | F::ToDate | F::ToTimestamp => {
            analyze_format_function(func, args, scope, catalog, aggregates)
        },
        _ => unreachable!("analyze_temporal_function dispatch is exhaustive over temporal funcs"),
    }
}

/// Analyze `TO_CHAR` / `TO_DATE` / `TO_TIMESTAMP`: `(value, format)` where `format` is text;
/// `TO_CHAR`'s value is temporal (→ `Text`), the parsers' value is text (→ `Date` / `Timestamp`).
fn analyze_format_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ColumnType::{Date, Text, Timestamp};
    use ast::ScalarFunc as F;
    let name = func.name();
    let (value_expr, format_expr) = expect_two_args(args, name)?;
    let value = analyze_expr_agg(value_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let value_ok = if func == F::ToChar {
        // TO_CHAR formats either a temporal value or a number (B-fn).
        matches!(
            value.ty,
            Date | ColumnType::Time | Timestamp | ColumnType::TimestampTz
        ) || is_numeric(value.ty)
            || is_null_literal(&value)
    } else {
        matches!(value.ty, Text) || is_null_literal(&value)
    };
    if !value_ok {
        return Err(Error::TypeMismatch {
            context: format!("{name}() first argument"),
            expected: if func == F::ToChar { Timestamp } else { Text },
            found: value.ty,
        });
    }
    let format = analyze_expr_agg(format_expr, scope, catalog, None, aggregates)?;
    if !matches!(format.ty, Text) && !is_null_literal(&format) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() format"),
            expected: Text,
            found: format.ty,
        });
    }
    let result = match func {
        F::ToChar => Text,
        F::ToDate => Date,
        _ => Timestamp,
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![value, format],
        },
        ty: result,
    })
}

/// Read exactly two positional arguments, erroring with an arity message otherwise.
fn expect_two_args<'a>(
    args: &'a [ast::Expr],
    name: &str,
) -> Result<(&'a ast::Expr, &'a ast::Expr), Error> {
    match args {
        [a, b] => Ok((a, b)),
        _ => Err(Error::Unsupported(format!(
            "{name}() expects 2 argument(s), got {}",
            args.len()
        ))),
    }
}

/// Read a lowercase text-literal field name, erroring if the argument is not a string literal.
fn expect_field_literal(expr: &ast::Expr, name: &str) -> Result<String, Error> {
    match expr {
        ast::Expr::Literal(ast::Value::Text(s)) => Ok(s.to_lowercase()),
        _ => Err(Error::Unsupported(format!(
            "{name}() field must be a string literal"
        ))),
    }
}

/// Field names supported by `EXTRACT`.
fn is_extract_field(field: &str) -> bool {
    matches!(
        field,
        "year"
            | "month"
            | "day"
            | "hour"
            | "minute"
            | "second"
            | "dow"
            | "isodow"
            | "doy"
            | "quarter"
            | "epoch"
            | "week"
    )
}

/// Precisions supported by `DATE_TRUNC`.
fn is_trunc_field(field: &str) -> bool {
    matches!(
        field,
        "microsecond"
            | "microseconds"
            | "second"
            | "minute"
            | "hour"
            | "day"
            | "week"
            | "month"
            | "quarter"
            | "year"
    )
}

pub(super) fn analyze_cast(
    expr: &ast::Expr,
    target: ColumnType,
    try_cast: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // No `Some(target)` hint — the inner expression keeps its natural type
    // and the executor handles the conversion. A bare `NULL` casts cleanly
    // because the target type itself supplies the hint.
    let inner = analyze_expr_agg(expr, scope, catalog, Some(target), aggregates)?;
    Ok(TypedExpr {
        kind: TypedExprKind::Cast(Box::new(inner), try_cast),
        // The integer width is kept (`SMALLINT`/`INT`/`BIGINT`) so a narrowing cast enforces its
        // range at evaluation, matching the reference engine (`9999999999::int` errors) and the storage-side int range
        // check. Every other declared width collapses to its physical type.
        ty: super::expr_type(target),
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "flat WHEN/THEN/ELSE typing pass with deferred bare-NULL branch resolution"
)]
pub(super) fn analyze_case(
    operand: Option<&ast::Expr>,
    branches: &[ast::CaseBranch],
    default: Option<&ast::Expr>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // For the simple form, every `when` must be comparable to the operand.
    // For the searched form, every `when` must be boolean.
    let operand_typed = match operand {
        // A bare untyped `NULL` operand (`CASE NULL WHEN … END`) has no type of its own. The reference
        // engine types it from the WHEN values it is compared against (they must be comparable), so
        // peek the first WHEN that types concretely for the hint — analyzed WITHOUT the aggregate sink
        // so a peek cannot double-collect an aggregate, and swallowing a peek error is safe because the
        // loop below re-analyzes each WHEN authoritatively and surfaces any real error. Every WHEN also
        // untyped-NULL falls back to TEXT (the unknown -> text rule), so `CASE NULL WHEN NULL THEN 1
        // ELSE 2 END` evaluates rather than raising "cannot infer the type of NULL".
        Some(expr) if is_bare_null(expr) => {
            let hint = branches
                .iter()
                .find_map(|b| analyze_expr_agg(&b.when, scope, catalog, None, None).ok())
                .map_or(ColumnType::Text, |t| t.ty);
            Some(analyze_expr_agg(
                expr,
                scope,
                catalog,
                Some(hint),
                aggregates.as_deref_mut(),
            )?)
        },
        Some(expr) => Some(analyze_expr_agg(
            expr,
            scope,
            catalog,
            None,
            aggregates.as_deref_mut(),
        )?),
        None => None,
    };

    // A `THEN`/`ELSE` that is a bare `NULL` has no type of its own; like the reference engine it takes the type unified
    // from the other branches, so `CASE WHEN c THEN NULL ELSE <typed> END` is valid rather than an
    // "ambiguous NULL" error. Such branches are deferred here and typed once the
    // result type is known. If *every* result is NULL the CASE is an untyped NULL, which materializes
    // as TEXT — the reference engine's unknown -> text fallback.
    let mut typed_branches: Vec<TypedCaseBranch> = Vec::with_capacity(branches.len());
    let mut null_then_slots: Vec<usize> = Vec::new();
    let mut result_ty: Option<ColumnType> = None;

    for branch in branches {
        let when_typed = if let Some(op) = &operand_typed {
            let w = analyze_expr_agg(
                &branch.when,
                scope,
                catalog,
                Some(op.ty),
                aggregates.as_deref_mut(),
            )?;
            if !comparable(op.ty, w.ty) {
                return Err(Error::TypeMismatch {
                    context: "CASE WHEN value".to_owned(),
                    expected: op.ty,
                    found: w.ty,
                });
            }
            w
        } else {
            let w = analyze_expr_agg(
                &branch.when,
                scope,
                catalog,
                Some(ColumnType::Bool),
                aggregates.as_deref_mut(),
            )?;
            if w.ty != ColumnType::Bool {
                return Err(Error::TypeMismatch {
                    context: "CASE WHEN predicate".to_owned(),
                    expected: ColumnType::Bool,
                    found: w.ty,
                });
            }
            w
        };
        // A bare `NULL` THEN is deferred (its slot recorded) and typed in the resolution pass below;
        // the pushed placeholder is overwritten there. The `WHEN` predicate is still analyzed above so
        // its validation and aggregate collection are unaffected.
        if matches!(branch.then, ast::Expr::Literal(ast::Value::Null)) {
            null_then_slots.push(typed_branches.len());
            typed_branches.push(TypedCaseBranch {
                when: when_typed,
                then: TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Null),
                    ty: ColumnType::Text,
                },
            });
        } else {
            let then_typed = analyze_expr_agg(
                &branch.then,
                scope,
                catalog,
                result_ty,
                aggregates.as_deref_mut(),
            )?;
            result_ty = Some(unify_result_ty(result_ty, then_typed.ty, "CASE THEN")?);
            typed_branches.push(TypedCaseBranch {
                when: when_typed,
                then: then_typed,
            });
        }
    }

    let mut null_default = false;
    let default_typed = match default {
        // A bare `NULL` ELSE is deferred just like a NULL THEN.
        Some(ast::Expr::Literal(ast::Value::Null)) => {
            null_default = true;
            None
        },
        Some(expr) => {
            let d = analyze_expr_agg(expr, scope, catalog, result_ty, aggregates)?;
            result_ty = Some(unify_result_ty(result_ty, d.ty, "CASE ELSE")?);
            Some(Box::new(d))
        },
        None => None,
    };

    // Unify across every typed branch; with no typed branch at all (every result is NULL) the reference engine yields an
    // untyped NULL, materialized here as TEXT.
    let resolved_ty = result_ty.unwrap_or(ColumnType::Text);
    let make_null = || TypedExpr {
        kind: TypedExprKind::Literal(ast::Value::Null),
        ty: resolved_ty,
    };
    for slot in null_then_slots {
        if let Some(branch) = typed_branches.get_mut(slot) {
            branch.then = make_null();
        }
    }
    let default_typed = if null_default {
        Some(Box::new(make_null()))
    } else {
        default_typed
    };

    Ok(TypedExpr {
        kind: TypedExprKind::Case {
            operand: operand_typed.map(Box::new),
            branches: typed_branches,
            default: default_typed,
        },
        ty: resolved_ty,
    })
}

/// Pick the result type that unifies `seen` (already-decided) with `next`
/// (new branch / default). NusaDB requires identical types across CASE
/// results; mixed numeric (Int/Float) is the only widening allowed.
pub(super) fn unify_result_ty(
    seen: Option<ColumnType>,
    next: ColumnType,
    context: &str,
) -> Result<ColumnType, Error> {
    match seen {
        None => Ok(next),
        Some(prev) if prev == next => Ok(prev),
        // Mixed numeric branches widen by the same rule as arithmetic: FLOAT dominates, then NUMERIC
        // over INT. NUMERIC participates here because a plain decimal literal now types as NUMERIC
        // So e.g. `CASE … THEN 0.5 ELSE 1.0::float END` and `SELECT 0.5 UNION SELECT 1`
        // must still unify rather than raise a spurious TypeMismatch.
        Some(prev) if is_numeric(prev) && is_numeric(next) => Ok(widen_numeric(prev, next)),
        Some(prev) => Err(Error::TypeMismatch {
            context: context.to_owned(),
            expected: prev,
            found: next,
        }),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the LIKE node's fields plus the analysis scope/catalog/aggregate sink"
)]
pub(super) fn analyze_like(
    expr: &ast::Expr,
    pattern: &ast::Expr,
    negated: bool,
    escape: Option<char>,
    case_insensitive: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let expr_typed = analyze_expr_agg(
        expr,
        scope,
        catalog,
        Some(ColumnType::Text),
        aggregates.as_deref_mut(),
    )?;
    let pattern_typed =
        analyze_expr_agg(pattern, scope, catalog, Some(ColumnType::Text), aggregates)?;
    for (operand, label) in [
        (&expr_typed, "LIKE subject"),
        (&pattern_typed, "LIKE pattern"),
    ] {
        if operand.ty != ColumnType::Text {
            return Err(Error::TypeMismatch {
                context: label.to_owned(),
                expected: ColumnType::Text,
                found: operand.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Like {
            expr: Box::new(expr_typed),
            pattern: Box::new(pattern_typed),
            negated,
            escape,
            case_insensitive,
        },
        ty: ColumnType::Bool,
    })
}

/// Analyze a regex-match operator `~`/`~*`/`!~`/`!~*`: both operands must be `TEXT`, the
/// result is `BOOL`. The pattern is compiled (and validated) per row by the executor.
pub(super) fn analyze_regex_match(
    expr: &ast::Expr,
    pattern: &ast::Expr,
    case_sensitive: bool,
    negated: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let expr_typed = analyze_expr_agg(
        expr,
        scope,
        catalog,
        Some(ColumnType::Text),
        aggregates.as_deref_mut(),
    )?;
    let pattern_typed =
        analyze_expr_agg(pattern, scope, catalog, Some(ColumnType::Text), aggregates)?;
    for (operand, label) in [
        (&expr_typed, "regex-match subject"),
        (&pattern_typed, "regex-match pattern"),
    ] {
        if operand.ty != ColumnType::Text {
            return Err(Error::TypeMismatch {
                context: label.to_owned(),
                expected: ColumnType::Text,
                found: operand.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::RegexMatch {
            expr: Box::new(expr_typed),
            pattern: Box::new(pattern_typed),
            case_sensitive,
            negated,
        },
        ty: ColumnType::Bool,
    })
}

/// Analyze `expr [NOT] SIMILAR TO pattern`: both operands must be `TEXT`, the result is
/// `BOOL`. The SQL `SIMILAR TO` pattern is translated to a POSIX regex (anchored) by the executor.
pub(super) fn analyze_similar_to(
    expr: &ast::Expr,
    pattern: &ast::Expr,
    negated: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let expr_typed = analyze_expr_agg(
        expr,
        scope,
        catalog,
        Some(ColumnType::Text),
        aggregates.as_deref_mut(),
    )?;
    let pattern_typed =
        analyze_expr_agg(pattern, scope, catalog, Some(ColumnType::Text), aggregates)?;
    for (operand, label) in [
        (&expr_typed, "SIMILAR TO subject"),
        (&pattern_typed, "SIMILAR TO pattern"),
    ] {
        if operand.ty != ColumnType::Text {
            return Err(Error::TypeMismatch {
                context: label.to_owned(),
                expected: ColumnType::Text,
                found: operand.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::SimilarTo {
            expr: Box::new(expr_typed),
            pattern: Box::new(pattern_typed),
            negated,
        },
        ty: ColumnType::Bool,
    })
}

pub(super) fn analyze_in_list(
    expr: &ast::Expr,
    list: &[ast::Expr],
    negated: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // An untyped bare-`NULL` probe takes its type from a LITERAL first list item
    // (`NULL IN (1, 2)` is NULL, three-valued) — mirroring how bare-NULL
    // list ITEMS already type from the probe's hint below. Restricted to a literal so the
    // peek analysis cannot double-collect an aggregate (literals carry none); a non-literal
    // first item keeps the untyped-NULL rejection.
    let probe_hint = match (expr, list.first()) {
        (ast::Expr::Literal(ast::Value::Null), Some(first @ ast::Expr::Literal(_))) => {
            Some(analyze_expr_agg(first, scope, catalog, None, None)?.ty)
        },
        _ => None,
    };
    let expr_typed = analyze_expr_agg(expr, scope, catalog, probe_hint, aggregates.as_deref_mut())?;
    let mut typed_list = Vec::with_capacity(list.len());
    for item in list {
        let item_typed = analyze_expr_agg(
            item,
            scope,
            catalog,
            Some(expr_typed.ty),
            aggregates.as_deref_mut(),
        )?;
        // A bare string literal in the list adopts the probe's temporal / UUID type, so
        // `col IN ($1, $2)` (date bounds bound as text) type-checks like the explicit `::date` form.
        let item_typed = coerce_unknown_literal(item_typed, expr_typed.ty);
        if !comparable(expr_typed.ty, item_typed.ty) {
            return Err(Error::TypeMismatch {
                context: "IN list".to_owned(),
                expected: expr_typed.ty,
                found: item_typed.ty,
            });
        }
        typed_list.push(item_typed);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::InList {
            expr: Box::new(expr_typed),
            list: typed_list,
            negated,
        },
        ty: ColumnType::Bool,
    })
}

pub(super) fn analyze_between(
    expr: &ast::Expr,
    low: &ast::Expr,
    high: &ast::Expr,
    negated: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let expr_typed = analyze_expr_agg(expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let low_typed = analyze_expr_agg(
        low,
        scope,
        catalog,
        Some(expr_typed.ty),
        aggregates.as_deref_mut(),
    )?;
    let high_typed = analyze_expr_agg(high, scope, catalog, Some(expr_typed.ty), aggregates)?;
    // Unknown-literal coercion, both directions: `col BETWEEN $1 AND $2` (temporal `col`, text-bound
    // bounds) and the rarer `'2026-01-01' BETWEEN d1 AND d2` (text-literal probe) each type-check like
    // the explicit `::date` form. A no-op unless the anchor is temporal / UUID and the peer is a bare
    // `TEXT` literal.
    let expr_typed = coerce_unknown_literal(expr_typed, low_typed.ty);
    let low_typed = coerce_unknown_literal(low_typed, expr_typed.ty);
    let high_typed = coerce_unknown_literal(high_typed, expr_typed.ty);
    for (operand, label) in [(&low_typed, "low"), (&high_typed, "high")] {
        if !comparable(expr_typed.ty, operand.ty) {
            return Err(Error::TypeMismatch {
                context: format!("BETWEEN {label}"),
                expected: expr_typed.ty,
                found: operand.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Between {
            expr: Box::new(expr_typed),
            low: Box::new(low_typed),
            high: Box::new(high_typed),
            negated,
        },
        ty: ColumnType::Bool,
    })
}

pub(super) fn comparable(a: ColumnType, b: ColumnType) -> bool {
    // Mirror `check_comparison` (the rule for `=`/`<`/`>` …) exactly: two values compare when they
    // share a type, or both are numeric (the executor's `compare` orders every Int/Float/Numeric
    // pair). Keeping this in lockstep means BETWEEN / IN / simple-CASE accept precisely what a plain
    // comparison accepts. Previously the temporal types (DATE/TIME/TIMESTAMPTZ/TIMETZ/INTERVAL),
    // UUID, JSON and arrays type-checked under `<`/`=` yet were spuriously rejected by
    // BETWEEN/IN/CASE — even though the executor already orders all of them. NUMERIC of
    // differing precision/scale still compares via the numeric arm even when `a != b`.
    a == b || (is_numeric(a) && is_numeric(b))
}

/// Implicit unknown-literal coercion for a comparison-shaped operand pair.
///
/// The reference engine treats a bare string literal as an *unknown* type that adopts the type of
/// whatever it is compared against. Our literal typing pins a string literal to `TEXT`, so a
/// comparison against a temporal / `UUID` operand — `WHERE d >= '2026-01-01'`, or the identical
/// query with a bound `$1` (a driver sends a date/time/uuid parameter as text over the extended
/// protocol) — would raise a spurious `TypeMismatch: expected Date, found Text`. When `anchor` is a
/// temporal / `UUID` type and `operand` is a bare `TEXT` literal, re-type it as a cast to `anchor`,
/// producing the exact same typed expression an explicit `'…'::date` would: the executor parses the
/// text at evaluation, and an unparseable string still loud-rejects (never a silent wrong row).
///
/// Only a `TEXT` *literal* is coerced — a genuinely `TEXT`-typed column or expression versus a
/// temporal column stays a real type error, matching the reference engine (only string literals are
/// "unknown"). A non-temporal `anchor` (or a non-literal operand) is returned unchanged.
fn coerce_unknown_literal(operand: TypedExpr, anchor: ColumnType) -> TypedExpr {
    if is_temporal_or_uuid(anchor)
        && matches!(&operand.kind, TypedExprKind::Literal(ast::Value::Text(_)))
    {
        TypedExpr {
            kind: TypedExprKind::Cast(Box::new(operand), false),
            ty: anchor,
        }
    } else {
        operand
    }
}

pub(super) fn analyze_null(hint: Option<ColumnType>) -> Result<TypedExpr, Error> {
    hint.map_or_else(
        || {
            Err(Error::AmbiguousNull {
                context: "a position with no type context".to_owned(),
            })
        },
        |ty| {
            Ok(TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Null),
                ty,
            })
        },
    )
}

/// Desugar a row comparison `(a, b, …) OP (c, d, …)` into an equivalent scalar boolean expression.
/// `=`/`<>` fold element-wise with `AND`/`OR`; the ordering operators are
/// lexicographic — field `i` is `l[i] <strict> r[i] OR (l[i] = r[i] AND <rest>)`, with the last field
/// using the full operator. Both rows must have the same non-zero length.
fn desugar_row_comparison(
    left: &[ast::Expr],
    op: ast::BinaryOp,
    right: &[ast::Expr],
) -> Result<ast::Expr, Error> {
    use ast::BinaryOp as B;
    if left.len() != right.len() {
        return Err(Error::Unsupported(
            "a row comparison requires both rows to have the same number of fields".to_owned(),
        ));
    }
    let bin = |a: ast::Expr, o: B, b: ast::Expr| ast::Expr::Binary {
        left: Box::new(a),
        op: o,
        right: Box::new(b),
    };
    let pairs: Vec<(&ast::Expr, &ast::Expr)> = left.iter().zip(right).collect();
    let combined = match op {
        B::Eq => pairs
            .iter()
            .map(|(a, b)| bin((*a).clone(), B::Eq, (*b).clone()))
            .reduce(|acc, e| bin(acc, B::And, e)),
        B::NotEq => pairs
            .iter()
            .map(|(a, b)| bin((*a).clone(), B::NotEq, (*b).clone()))
            .reduce(|acc, e| bin(acc, B::Or, e)),
        B::Lt | B::LtEq | B::Gt | B::GtEq => {
            let strict = if matches!(op, B::Lt | B::LtEq) {
                B::Lt
            } else {
                B::Gt
            };
            pairs.iter().rev().fold(None, |rest, (a, b)| {
                Some(rest.map_or_else(
                    || bin((*a).clone(), op, (*b).clone()),
                    |rest| {
                        bin(
                            bin((*a).clone(), strict, (*b).clone()),
                            B::Or,
                            bin(bin((*a).clone(), B::Eq, (*b).clone()), B::And, rest),
                        )
                    },
                ))
            })
        },
        _ => {
            return Err(Error::Unsupported(
                "a row (…) is only valid with a comparison operator".to_owned(),
            ));
        },
    };
    combined
        .ok_or_else(|| Error::Unsupported("a row comparison requires a non-empty row".to_owned()))
}

pub(super) fn analyze_binary(
    left: &ast::Expr,
    op: ast::BinaryOp,
    right: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // A row comparison `(a, b) OP (c, d)` / `ROW(a, b) OP ROW(c, d)` desugars to a
    // scalar boolean expression before type-checking, so it inherits the ordinary comparison rules and
    // 3-valued NULL logic (e.g. `(1, NULL) < (1, 2)` is NULL, matching the reference engine).
    if matches!(
        op,
        ast::BinaryOp::Eq
            | ast::BinaryOp::NotEq
            | ast::BinaryOp::Lt
            | ast::BinaryOp::LtEq
            | ast::BinaryOp::Gt
            | ast::BinaryOp::GtEq
    ) && let (ast::Expr::Row(l), ast::Expr::Row(r)) = (left, right)
    {
        let desugared = desugar_row_comparison(l, op, r)?;
        return analyze_expr_agg(
            &desugared,
            scope,
            catalog,
            Some(ColumnType::Bool),
            aggregates,
        );
    }
    // When BOTH operands are a bare `NULL`, neither can be typed from a sibling. Most operators are
    // genuinely ambiguous then — `NULL + NULL` has no unique operator to resolve — but comparison,
    // logical and concatenation operators resolve two unknowns to a default type and evaluate to
    // `NULL`, so type both NULLs with that default rather than failing (`NULL = NULL` is `NULL`, not
    // an error). [`both_null_binary_hint`] returns `None` for the ambiguous operators.
    let (mut left_typed, mut right_typed) = match (is_bare_null(left), is_bare_null(right)) {
        (true, true) => {
            let hint = both_null_binary_hint(op).ok_or_else(|| Error::AmbiguousNull {
                context: "both operands of a binary operator".to_owned(),
            })?;
            let left_typed =
                analyze_expr_agg(left, scope, catalog, Some(hint), aggregates.as_deref_mut())?;
            let right_typed = analyze_expr_agg(right, scope, catalog, Some(hint), aggregates)?;
            (left_typed, right_typed)
        },
        _ => analyze_operands(left, right, scope, catalog, aggregates)?,
    };
    // Unknown-literal coercion: on a comparison, a bare string literal adopts the other operand's
    // temporal / UUID type, so a parameterized date filter (`WHERE d >= $1`, bound as text) type-checks
    // exactly like the explicit `$1::date` form. A no-op for every non-comparison operator and for
    // operands that are not a bare `TEXT` literal.
    if matches!(
        op,
        ast::BinaryOp::Eq
            | ast::BinaryOp::NotEq
            | ast::BinaryOp::Lt
            | ast::BinaryOp::LtEq
            | ast::BinaryOp::Gt
            | ast::BinaryOp::GtEq
    ) {
        right_typed = coerce_unknown_literal(right_typed, left_typed.ty);
        left_typed = coerce_unknown_literal(left_typed, right_typed.ty);
    }
    let ty = check_binary(op, left_typed.ty, right_typed.ty)?;
    Ok(TypedExpr {
        kind: TypedExprKind::Binary {
            left: Box::new(left_typed),
            op,
            right: Box::new(right_typed),
        },
        ty,
    })
}

/// The default operand type for a binary operator whose operands are *both* a bare `NULL`, or `None`
/// when the operator leaves two unknowns genuinely ambiguous (arithmetic, bitwise, JSON, vector —
/// no unique operator resolves there). Comparisons and `||` resolve as `TEXT`; `AND`/`OR` as `BOOL`.
/// The call still evaluates to `NULL`; this only picks a type so analysis does not reject it.
const fn both_null_binary_hint(op: ast::BinaryOp) -> Option<ColumnType> {
    use ast::BinaryOp as Op;
    match op {
        Op::Eq | Op::NotEq | Op::Lt | Op::LtEq | Op::Gt | Op::GtEq | Op::Concat => {
            Some(ColumnType::Text)
        },
        Op::And | Op::Or => Some(ColumnType::Bool),
        _ => None,
    }
}

/// Analyze `left IS [NOT] DISTINCT FROM right`: the operands must be comparable (the same
/// rule as `=`), and the result is always `BOOL`. A bare `NULL` operand is typed from its sibling.
pub(super) fn analyze_is_distinct_from(
    left: &ast::Expr,
    right: &ast::Expr,
    negated: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // `NULL IS [NOT] DISTINCT FROM NULL` is well-defined — two NULLs are never distinct — unlike a
    // bare-NULL `=`, which `analyze_operands` rejects as ambiguous. Type both NULLs as `BOOL` (the
    // operand type is irrelevant to the all-NULL outcome) so the predicate is accepted.
    let (left_typed, right_typed) = if is_bare_null(left) && is_bare_null(right) {
        let left_typed = analyze_expr_agg(
            left,
            scope,
            catalog,
            Some(ColumnType::Bool),
            aggregates.as_deref_mut(),
        )?;
        let right_typed =
            analyze_expr_agg(right, scope, catalog, Some(ColumnType::Bool), aggregates)?;
        (left_typed, right_typed)
    } else {
        analyze_operands(left, right, scope, catalog, aggregates)?
    };
    // A bare string literal adopts the sibling's temporal / UUID type (same unknown-literal rule as
    // `=`), so `d IS DISTINCT FROM $1` (date bound as text) type-checks like `$1::date`.
    let right_typed = coerce_unknown_literal(right_typed, left_typed.ty);
    let left_typed = coerce_unknown_literal(left_typed, right_typed.ty);
    // Validate comparability (reuses the `=` type rule); the result type is always BOOL.
    check_comparison(left_typed.ty, right_typed.ty)?;
    Ok(TypedExpr {
        kind: TypedExprKind::IsDistinctFrom {
            left: Box::new(left_typed),
            right: Box::new(right_typed),
            negated,
        },
        ty: ColumnType::Bool,
    })
}

/// Analyze both operands of a binary operator, typing a bare `NULL` operand
/// from its concretely-typed sibling.
pub(super) fn analyze_operands(
    left: &ast::Expr,
    right: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<(TypedExpr, TypedExpr), Error> {
    match (is_bare_null(left), is_bare_null(right)) {
        (true, true) => Err(Error::AmbiguousNull {
            context: "both operands of a binary operator".to_owned(),
        }),
        (true, false) => {
            let right_typed =
                analyze_expr_agg(right, scope, catalog, None, aggregates.as_deref_mut())?;
            let left_typed =
                analyze_expr_agg(left, scope, catalog, Some(right_typed.ty), aggregates)?;
            Ok((left_typed, right_typed))
        },
        (false, true) => {
            let left_typed =
                analyze_expr_agg(left, scope, catalog, None, aggregates.as_deref_mut())?;
            let right_typed =
                analyze_expr_agg(right, scope, catalog, Some(left_typed.ty), aggregates)?;
            Ok((left_typed, right_typed))
        },
        (false, false) => {
            let left_typed =
                analyze_expr_agg(left, scope, catalog, None, aggregates.as_deref_mut())?;
            let right_typed = analyze_expr_agg(right, scope, catalog, None, aggregates)?;
            Ok((left_typed, right_typed))
        },
    }
}

pub(super) fn check_binary(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use ast::BinaryOp as Op;
    match op {
        Op::Eq | Op::NotEq | Op::Lt | Op::LtEq | Op::Gt | Op::GtEq => check_comparison(left, right),
        Op::And | Op::Or => check_logical(left, right),
        Op::Plus | Op::Minus | Op::Multiply | Op::Divide | Op::Modulo => {
            // INTERVAL / temporal arithmetic takes priority over numeric.
            check_interval_arith(op, left, right).map_or_else(|| check_arithmetic(left, right), Ok)
        },
        Op::BitAnd | Op::BitOr | Op::BitXor | Op::ShiftLeft | Op::ShiftRight => {
            check_bitwise(op, left, right)
        },
        Op::ArrayOverlap => check_array_overlap(left, right),
        Op::Concat => check_concat(left, right),
        // `@>` / `<@` are containment over JSON *and* arrays, so they get their own checker.
        Op::JsonContains | Op::JsonContainedBy => check_containment(op, left, right),
        Op::JsonGet | Op::JsonGetText | Op::JsonGetPath | Op::JsonGetPathText => {
            check_json(op, left, right)
        },
        Op::VectorDistance => check_vector_distance(left, right),
        Op::TsMatch => check_ts_match(left, right),
    }
}

/// Type rule for `@@` (F1): both operands are the text forms of a `tsvector`/`tsquery` (either
/// order, like the reference engine), so both must be `TEXT`; the result is the `BOOL` match.
pub(super) fn check_ts_match(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    if matches!(
        left,
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
    ) && matches!(
        right,
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
    ) {
        Ok(ColumnType::Bool)
    } else {
        Err(Error::TypeMismatch {
            context: "`@@` text-search match".to_owned(),
            expected: ColumnType::Text,
            found: if matches!(
                left,
                ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
            ) {
                right
            } else {
                left
            },
        })
    }
}

/// Type rule for `<=>`: both operands must be `VECTOR`s of the same dimension; the result
/// is the `FLOAT` distance. A bare `NULL` operand is already typed from its sibling earlier.
pub(super) fn check_vector_distance(
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    match (left, right) {
        (ColumnType::Vector(a), ColumnType::Vector(b)) if a == b => Ok(ColumnType::Float),
        _ => Err(Error::TypeMismatch {
            context: "`<=>` vector distance".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for `||`: `TEXT || TEXT → TEXT`; array concatenation `T[] || T[] → T[]`
/// and element append/prepend `T[] || T` / `T || T[] → T[]` (the scalar must be the array's element
/// type). A bare `NULL` operand is already typed from its sibling by [`analyze_operands`].
/// Analyze the text-polymorphic functions outside the fixed signature table:
/// `LENGTH`/`OCTET_LENGTH`/`BIT_LENGTH` take one TEXT **or** BYTEA argument and
/// return INT; `CONCAT`/`CONCAT_WS` accept any [`textout_scalar`] argument —
/// NULLs are skipped at evaluation — with `CONCAT_WS`'s first argument (the separator) required
/// to be TEXT. Mirrors the fixed table's arity message and NULL-literal tolerance.
fn analyze_text_polymorphic(
    func: ast::ScalarFunc,
    name: &str,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let length_family = matches!(func, F::Length | F::OctetLength | F::BitLength);
    let (min, max) = match func {
        F::Length | F::OctetLength | F::BitLength => (1, 1),
        F::Concat => (1, usize::MAX),
        _ => (2, usize::MAX), // CONCAT_WS: separator + at least one value
    };
    if args.len() < min || args.len() > max {
        let arity = if min == max {
            min.to_string()
        } else {
            format!("at least {min}")
        };
        return Err(Error::Unsupported(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    let mut typed_args = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let typed = analyze_expr_agg(
            arg,
            scope,
            catalog,
            Some(ColumnType::Text),
            aggregates.as_deref_mut(),
        )?;
        let ok = if length_family {
            matches!(typed.ty.physical(), ColumnType::Text | ColumnType::Bytes)
        } else if matches!(func, F::ConcatWs) && i == 0 {
            typed.ty.physical() == ColumnType::Text
        } else {
            textout_scalar(typed.ty)
        };
        if !ok && !is_null_literal(&typed) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument {}", i + 1),
                expected: ColumnType::Text,
                found: typed.ty,
            });
        }
        typed_args.push(typed);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: if length_family {
            ColumnType::Int
        } else {
            ColumnType::Text
        },
    })
}

/// Analyze `SUBSTRING`, overloaded on its second argument's type:
/// `substring(s, start [, len])` / `substring(s FROM start [FOR len])` is the positional form
/// (TEXT, INT [, INT]) → TEXT, while `substring(s FROM 'pattern')` with a TEXT second argument
/// is the POSIX-regex form → the first capture group of the first match (whole match when the
/// pattern has no groups), `NULL` when there is no match. The three-argument all-TEXT form
/// (SQL-standard `SIMILAR TO` escape syntax) is rejected loudly. `substr()` shares the lowering,
/// so it accepts the regex form too.
fn analyze_substring(
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let Some(([source_arg, second_arg], rest)) = args.split_first_chunk() else {
        return Err(Error::Unsupported(format!(
            "substring() expects 2..=3 argument(s), got {}",
            args.len()
        )));
    };
    if rest.len() > 1 {
        return Err(Error::Unsupported(format!(
            "substring() expects 2..=3 argument(s), got {}",
            args.len()
        )));
    }
    let mut typed_args = Vec::with_capacity(args.len());
    let source = analyze_expr_agg(
        source_arg,
        scope,
        catalog,
        Some(ColumnType::Text),
        aggregates.as_deref_mut(),
    )?;
    if source.ty != ColumnType::Text && !is_null_literal(&source) {
        return Err(Error::TypeMismatch {
            context: "substring() argument 1".to_owned(),
            expected: ColumnType::Text,
            found: source.ty,
        });
    }
    typed_args.push(source);
    // The INT hint keeps a bare `NULL` start typing as the positional form (as the fixed table
    // did); a TEXT-typed expression still types TEXT and selects the regex form.
    let second = analyze_expr_agg(
        second_arg,
        scope,
        catalog,
        Some(ColumnType::Int),
        aggregates.as_deref_mut(),
    )?;
    let regex_form = second.ty == ColumnType::Text;
    if regex_form && args.len() == 3 {
        return Err(Error::Unsupported(
            "substring(s FROM pattern FOR escape) (SIMILAR TO regex form) is not supported; use the POSIX form substring(s FROM 'pattern')"
                .to_owned(),
        ));
    }
    if !regex_form && second.ty != ColumnType::Int && !is_null_literal(&second) {
        return Err(Error::TypeMismatch {
            context: "substring() argument 2".to_owned(),
            expected: ColumnType::Int,
            found: second.ty,
        });
    }
    typed_args.push(second);
    if let Some(len) = rest.first() {
        let len = analyze_expr_agg(len, scope, catalog, Some(ColumnType::Int), aggregates)?;
        if len.ty != ColumnType::Int && !is_null_literal(&len) {
            return Err(Error::TypeMismatch {
                context: "substring() argument 3".to_owned(),
                expected: ColumnType::Int,
                found: len.ty,
            });
        }
        typed_args.push(len);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func: ast::ScalarFunc::Substring,
            args: typed_args,
        },
        ty: ColumnType::Text,
    })
}

/// Whether `ty` has a text output rendering for `||`/`CONCAT` coercion:
/// every scalar the cast-to-text path renders (booleans render `t`/`f` via the output
/// function). BYTEA, JSON, arrays, and vectors are deliberately excluded — each has (or
/// reserves) its own concatenation semantics.
pub(super) const fn textout_scalar(ty: ColumnType) -> bool {
    matches!(
        ty.physical(),
        ColumnType::Text
            | ColumnType::Bool
            | ColumnType::Int
            | ColumnType::Float
            | ColumnType::Numeric { .. }
            | ColumnType::Date
            | ColumnType::Time
            | ColumnType::TimeTz
            | ColumnType::Timestamp
            | ColumnType::TimestampTz
            | ColumnType::Uuid
            | ColumnType::Interval
    )
}

pub(super) fn check_concat(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    use ColumnType::Array;
    match (left, right) {
        // Text concatenation (the original `||`).
        (ColumnType::Text, ColumnType::Text) => Ok(ColumnType::Text),
        // BYTEA concatenation: `bytea || bytea → bytea`.
        (ColumnType::Bytes, ColumnType::Bytes) => Ok(ColumnType::Bytes),
        // Array concatenation: same element type on both sides.
        (Array(a), Array(b)) if a == b => Ok(Array(a)),
        // Append/prepend an element to an array: the scalar must be the array's element type.
        (Array(a), elem) if a.column_type() == elem => Ok(Array(a)),
        (elem, Array(b)) if b.column_type() == elem => Ok(Array(b)),
        // One TEXT side coerces the other scalar to its text output (the
        // reference's text-any concatenation): `'x' || 5` = `'x5'`. BYTEA keeps its own `||`,
        // JSON stays reserved (a future json-concat operator), arrays follow the rules above.
        (ColumnType::Text, other) if textout_scalar(other) => Ok(ColumnType::Text),
        (other, ColumnType::Text) if textout_scalar(other) => Ok(ColumnType::Text),
        _ => Err(Error::TypeMismatch {
            context: "`||` concatenation".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

pub(super) fn check_comparison(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    if left == right || (is_numeric(left) && is_numeric(right)) {
        Ok(ColumnType::Bool)
    } else {
        Err(Error::TypeMismatch {
            context: "comparison".to_owned(),
            expected: left,
            found: right,
        })
    }
}

/// Type rule for INTERVAL / temporal arithmetic, or `None` if `op`/operands are not such a
/// case (caller falls back to numeric). `+`: `interval+interval→interval`, `T+interval→T` /
/// `interval+T→T` for a temporal `T` (DATE promotes to TIMESTAMP). `-`: `interval-interval→interval`,
/// `T-interval→T`.
pub(super) fn check_interval_arith(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Option<ColumnType> {
    use ColumnType::{Date, Int, Interval, Time, Timestamp, TimestampTz};
    let temporal_result = |t: ColumnType| if t == Date { Timestamp } else { t };
    match op {
        ast::BinaryOp::Plus => match (left, right) {
            (Interval, Interval) => Some(Interval),
            (Interval, t @ (Timestamp | TimestampTz | Date))
            | (t @ (Timestamp | TimestampTz | Date), Interval) => Some(temporal_result(t)),
            // `time + interval` wraps within the 24-hour clock.
            (Time, Interval) | (Interval, Time) => Some(Time),
            // `date + integer` adds whole days and yields a DATE (commutative).
            (Date, Int) | (Int, Date) => Some(Date),
            _ => None,
        },
        ast::BinaryOp::Minus => match (left, right) {
            // `interval - interval` and `timestamp - timestamp` (same kind) both yield an INTERVAL.
            (Interval, Interval) | (Timestamp, Timestamp) | (TimestampTz, TimestampTz) => {
                Some(Interval)
            },
            (t @ (Timestamp | TimestampTz | Date), Interval) => Some(temporal_result(t)),
            // `time - interval` wraps like the plus; `time - time` is the elapsed INTERVAL.
            (Time, Interval) => Some(Time),
            (Time, Time) => Some(Interval),
            // `date - integer` subtracts whole days → DATE; `date - date` is the day count → INTEGER.
            (Date, Int) => Some(Date),
            (Date, Date) => Some(Int),
            _ => None,
        },
        // `interval * integer` scales each component (commutative) → INTERVAL.
        ast::BinaryOp::Multiply => match (left, right) {
            (Interval, Int) | (Int, Interval) => Some(Interval),
            _ => None,
        },
        _ => None,
    }
}

/// Type-check the containment operators `@>` (contains) and `<@` (contained-by), which apply to
/// both `JSON` and arrays (standard), always yielding `BOOL`:
/// - **Arrays:** both operands are arrays of the *same* element type, regardless of direction.
/// - **JSON:** the *container* side is `JSON` and the *contained* side is `JSON` (or `TEXT` parsed
///   as JSON at eval time). For `@>` the container is the left operand; for `<@` it is the right.
pub(super) fn check_containment(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use ColumnType::{Array, Json, Text};
    // Array containment: same element type on both sides (direction does not matter).
    if let (Array(a), Array(b)) = (left, right) {
        if a == b {
            return Ok(ColumnType::Bool);
        }
        return Err(Error::TypeMismatch {
            context: "array `@>`/`<@` containment".to_owned(),
            expected: left,
            found: right,
        });
    }
    // JSON containment: the container is JSON; the contained side may be JSON or text.
    let (container, contained) = if op == ast::BinaryOp::JsonContains {
        (left, right)
    } else {
        (right, left)
    };
    if container == Json && matches!(contained, Json | Text) {
        return Ok(ColumnType::Bool);
    }
    Err(Error::TypeMismatch {
        context: "`@>`/`<@` containment (JSON or array)".to_owned(),
        expected: Json,
        found: if container == Json {
            contained
        } else {
            container
        },
    })
}

/// Type-check a JSON navigation operator: the left operand must be `JSON`. `->`/`->>` take a
/// text key or integer index; `->` yields `JSON`, `->>` yields `TEXT`. (`@>`/`<@` are handled by
/// [`check_containment`].)
pub(super) fn check_json(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    if left != ColumnType::Json {
        return Err(Error::TypeMismatch {
            context: "JSON operator".to_owned(),
            expected: ColumnType::Json,
            found: left,
        });
    }
    match op {
        ast::BinaryOp::JsonGet | ast::BinaryOp::JsonGetText => {
            if !matches!(right, ColumnType::Text | ColumnType::Int) {
                return Err(Error::TypeMismatch {
                    context: "JSON `->`/`->>` key".to_owned(),
                    expected: ColumnType::Text,
                    found: right,
                });
            }
            if op == ast::BinaryOp::JsonGet {
                Ok(ColumnType::Json)
            } else {
                Ok(ColumnType::Text)
            }
        },
        // `#>` / `#>>` take a `text[]` path; `#>` yields `JSON`, `#>>` yields `TEXT`. A bare
        // text value like `'{a,b}'` is accepted and coerced to `text[]` at eval time (SQL-standard).
        ast::BinaryOp::JsonGetPath | ast::BinaryOp::JsonGetPathText => {
            if !matches!(
                right,
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text) | ColumnType::Text
            ) {
                return Err(Error::TypeMismatch {
                    context: "JSON `#>`/`#>>` path".to_owned(),
                    expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
                    found: right,
                });
            }
            if op == ast::BinaryOp::JsonGetPath {
                Ok(ColumnType::Json)
            } else {
                Ok(ColumnType::Text)
            }
        },
        _ => Err(Error::Unsupported(
            "non-JSON operator in check_json".to_owned(),
        )),
    }
}

pub(super) fn check_logical(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    for ty in [left, right] {
        if ty != ColumnType::Bool {
            return Err(Error::TypeMismatch {
                context: "logical operator (AND/OR)".to_owned(),
                expected: ColumnType::Bool,
                found: ty,
            });
        }
    }
    Ok(ColumnType::Bool)
}

pub(super) fn check_arithmetic(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    for ty in [left, right] {
        if !is_numeric(ty) {
            return Err(Error::TypeMismatch {
                context: "arithmetic operator".to_owned(),
                expected: ColumnType::Int,
                found: ty,
            });
        }
    }
    // Float dominates (its inexactness is contagious); otherwise NUMERIC dominates Int (exact);
    // else plain integer arithmetic.
    if left == ColumnType::Float || right == ColumnType::Float {
        Ok(ColumnType::Float)
    } else if matches!(left, ColumnType::Numeric { .. })
        || matches!(right, ColumnType::Numeric { .. })
    {
        Ok(ColumnType::Numeric {
            precision: 0,
            scale: 0,
        })
    } else {
        // Integer arithmetic takes the wider operand's width, so the result's overflow bound is the
        // wider one (`int4 + int8 → int8`), matching the reference engine.
        Ok(wider_int(left, right))
    }
}

/// Type rule for the integer bitwise operators `&` / `|`: both operands must be `INT` and the
/// result is `INT`. Unlike arithmetic, there is no `FLOAT`/`NUMERIC` widening — bit operations are
/// defined only on integers.
pub(super) fn check_bitwise(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let symbol = match op {
        ast::BinaryOp::BitAnd => "&",
        ast::BinaryOp::BitXor => "#",
        ast::BinaryOp::BitOr => "|",
        ast::BinaryOp::ShiftLeft => "<<",
        _ => ">>",
    };
    for ty in [left, right] {
        if ty != ColumnType::Int {
            return Err(Error::TypeMismatch {
                context: format!("bitwise operator `{symbol}`"),
                expected: ColumnType::Int,
                found: ty,
            });
        }
    }
    Ok(ColumnType::Int)
}

/// Type rule for array overlap `&&`: both operands must be arrays of the *same* element type; the
/// result is `BOOL` (whether they share any element). A bare `NULL` operand is typed from its sibling
/// earlier (B-fn).
pub(super) fn check_array_overlap(
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    match (left, right) {
        (ColumnType::Array(a), ColumnType::Array(b)) if a == b => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: "array overlap operator `&&`".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

pub(super) fn analyze_unary(
    op: ast::UnaryOp,
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // `NOT` hints BOOL so a bare `NULL` operand types from context (
    // `NOT NULL` is NULL, three-valued) instead of rejecting as untypeable.
    let hint = match op {
        ast::UnaryOp::Not => Some(ColumnType::Bool),
        ast::UnaryOp::Negate => None,
    };
    let operand = analyze_expr_agg(expr, scope, catalog, hint, aggregates)?;
    let ty = match op {
        ast::UnaryOp::Not if operand.ty == ColumnType::Bool => ColumnType::Bool,
        ast::UnaryOp::Not => {
            return Err(Error::TypeMismatch {
                context: "NOT operator".to_owned(),
                expected: ColumnType::Bool,
                found: operand.ty,
            });
        },
        ast::UnaryOp::Negate if is_numeric(operand.ty) => operand.ty,
        ast::UnaryOp::Negate => {
            return Err(Error::TypeMismatch {
                context: "negation".to_owned(),
                expected: ColumnType::Int,
                found: operand.ty,
            });
        },
    };
    Ok(TypedExpr {
        kind: TypedExprKind::Unary {
            op,
            expr: Box::new(operand),
        },
        ty,
    })
}

/// The element type of a subquery that must yield exactly one column (scalar
/// and `IN` subqueries/). A different arity is a static error rather
/// than a run-time surprise.
fn single_subquery_column(plan: &SelectPlan, context: &str) -> Result<ColumnType, Error> {
    match plan.projection.as_slice() {
        [only] => Ok(only.expr.ty),
        _ => Err(Error::Unsupported(format!(
            "{context} must return exactly one column"
        ))),
    }
}
