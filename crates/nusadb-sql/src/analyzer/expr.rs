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
                ast::Value::Macaddr(_) => ColumnType::Macaddr,
                ast::Value::Macaddr8(_) => ColumnType::Macaddr8,
                ast::Value::Geometry(g) => ColumnType::Geometry(g.kind()),
                ast::Value::Tsvector(_) => ColumnType::Tsvector,
                ast::Value::Tsquery(_) => ColumnType::Tsquery,
                ast::Value::Xml(_) => ColumnType::Xml,
                ast::Value::Enum { .. } => ColumnType::Enum,
                ast::Value::Inet(a) => a.column_type(),
                ast::Value::Bit(b) => crate::bit::column_type(b),
                ast::Value::Range(r) => ColumnType::Range(r.kind),
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
        ast::Expr::Parameter(n) => Err(Error::UndefinedParameter(format!(
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
        ast::Expr::IsJson {
            operand,
            negated,
            item_type,
            unique_keys,
        } => {
            // The predicate validates the operand's *text*, so the operand must be textual or
            // JSON/JSONB (a bare string literal is fine — a `NULL` literal defaults to `TEXT`). The
            // result is a nullable `BOOLEAN` (a `NULL` operand yields `NULL` at evaluation).
            let operand =
                analyze_expr_agg(operand, scope, catalog, Some(ColumnType::Text), aggregates)?;
            match operand.ty.physical() {
                ColumnType::Text | ColumnType::Json => {},
                other => {
                    return Err(Error::TypeMismatch {
                        context: "IS JSON".to_owned(),
                        expected: ColumnType::Text,
                        found: other,
                    });
                },
            }
            Ok(TypedExpr {
                kind: TypedExprKind::IsJson {
                    operand: Box::new(operand),
                    negated: *negated,
                    item_type: *item_type,
                    unique_keys: *unique_keys,
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
            symmetric,
        } => analyze_between(
            expr, low, high, *negated, *symmetric, scope, catalog, aggregates,
        ),
        ast::Expr::Overlaps { s1, e1, s2, e2 } => {
            analyze_overlaps(s1, e1, s2, e2, scope, catalog, aggregates)
        },
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
        // `(expr).field` — composite field access.
        ast::Expr::FieldAccess { base, field } => {
            analyze_field_access(base, field, scope, catalog, aggregates)
        },
        // `expr::T` where `T` is a user-defined (composite) type name.
        ast::Expr::CastNamed {
            expr,
            type_name,
            try_cast,
        } => analyze_cast_named(expr, type_name, *try_cast, scope, catalog, aggregates),
        // `ROW(a, b, ...)` — an anonymous composite value. Each field types itself and the value is
        // carried in the canonical `(f1,f2,…)` text form, exactly like `ROW(...)::T` with the field
        // types inferred instead of declared. (`ROW(...)::T` itself is handled in
        // `analyze_cast_named`, which validates against the named type's declared fields.)
        ast::Expr::Row(items) => analyze_row_constructor(items, scope, catalog, aggregates),
        // A window function is only valid where the SELECT pipeline supplies a window stage
        // (the projection path lifts it before expression analysis); anywhere else there is
        // no execution path for it, so reject it here.
        ast::Expr::WindowFunction(_) => Err(Error::InvalidStatement(
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
                    return Err(Error::FunctionArgs(
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
                // JSON_OBJECT_AGG(key, value) also takes a second per-row argument, but the value may
                // be any type (serialized to JSON), so it is not numeric-constrained like the
                // statistical two-arg aggregates.
                let json_obj = matches!(func, ast::AggregateFunc::JsonObjectAgg);
                let typed_arg2 = if json_obj {
                    match arg2 {
                        Some(a2) => Some(analyze_expr(a2, scope, catalog, None)?),
                        None => {
                            return Err(Error::FunctionArgs(
                                "json_object_agg requires two arguments (key, value)".to_owned(),
                            ));
                        },
                    }
                } else {
                    match (arg2, two_arg) {
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
                            return Err(Error::FunctionArgs(format!(
                                "{func:?} requires two arguments"
                            )));
                        },
                        (Some(_), false) => {
                            return Err(Error::FunctionArgs(format!(
                                "{func:?} takes a single argument"
                            )));
                        },
                        (None, false) => None,
                    }
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
                                return Err(Error::InvalidStatement(
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
                // A row value as the argument — `COUNT((a, b))`, `COUNT(DISTINCT ROW(a, b))` —
                // folds its fields as one composite. Only COUNT has that path: its result is a
                // tally, so it needs no composite type, no composite storage, and no composite
                // comparison. Every other aggregate keeps rejecting a row value.
                let row_args = match (*func, arg.as_deref()) {
                    (ast::AggregateFunc::Count, Some(ast::Expr::Row(items))) => {
                        if items.is_empty() {
                            return Err(Error::InvalidStatement(
                                "COUNT() over an empty row value has nothing to count".to_owned(),
                            ));
                        }
                        items
                            .iter()
                            .map(|item| analyze_expr(item, scope, catalog, None))
                            .collect::<Result<Vec<_>, _>>()?
                    },
                    // A row value under any other aggregate would need a composite *value* to sum,
                    // order, or collect — the reference engine rejects `max(record)` too. Refuse it
                    // rather than let the anonymous constructor's text form aggregate as if it were
                    // an ordinary string.
                    (_, Some(ast::Expr::Row(_))) => {
                        return Err(Error::Unsupported(format!(
                            "a row value cannot be aggregated by {}() (only COUNT tallies a row \
                             value)",
                            func.name()
                        )));
                    },
                    _ => Vec::new(),
                };
                let (typed_arg, result_ty) = if row_args.is_empty() {
                    // A plain aggregate's argument may not itself be an aggregate (no sink), so a
                    // nested aggregate without a window is rejected.
                    analyze_aggregate(*func, arg.as_deref(), scope, catalog, None)?
                } else {
                    (None, ColumnType::Int)
                };
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
                    hypothetical_args: Vec::new(),
                    ordered_set_keys: Vec::new(),
                    filter: typed_filter,
                    separator,
                    arg2: typed_arg2,
                    order_by: order_keys,
                    row_args,
                    grouping_args: Vec::new(),
                });
                Ok(TypedExpr {
                    kind: TypedExprKind::AggregateRef(idx),
                    ty: result_ty,
                })
            },
            None => Err(Error::InvalidGrouping(
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
        return Err(Error::FunctionArgs("COALESCE with no arguments".to_owned()));
    }
    // Two different enum types have no common type — refuse before unifying (42846).
    reject_mixed_enum_operands("COALESCE", args.iter(), scope, catalog)?;
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
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // A NusaScript function is run by the interpreter per row, not inlined like a SQL function. Type
    // its argument expressions, then emit a `NusaCall` the executor invokes through the per-statement
    // execution context; the result type is the declared return type.
    if matches!(func.language, crate::ast::FunctionLanguage::NusaScript) {
        if args.len() != func.param_count {
            return Err(Error::ArityMismatch {
                context: format!("function `{name}`"),
                expected: func.param_count,
                found: args.len(),
            });
        }
        let mut typed_args = Vec::with_capacity(args.len());
        for arg in args {
            typed_args.push(analyze_expr_agg(
                arg,
                scope,
                catalog,
                None,
                aggregates.as_deref_mut(),
            )?);
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::NusaCall {
                args: typed_args,
                def: Box::new(crate::planner::NusaCallDef {
                    name: name.to_owned(),
                    param_names: func.param_names.clone(),
                    body: func.body.clone(),
                    return_type: func.return_type,
                }),
            },
            ty: func.return_type,
        });
    }
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
            return Err(Error::LimitExceeded(format!(
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
    use nusadb_core::engine::GeomKind;
    // GROUPING(key, ...) — super-aggregate indicator. Its arguments must be GROUP BY key
    // expressions, but the grouping sets are not in scope here; so we only type-check the arguments
    // against the source scope (they reference source columns, not aggregates) and carry them through
    // as a `ScalarFunction { Grouping, .. }` node. `rebase_onto_aggregation` later matches each
    // argument against the resolved `group_keys` and rewrites this node into the runtime bitmask
    // reference (or a constant `0` for a plain `GROUP BY`). Result is always `INT`.
    // NUSADB_TYPEOF(expr) — the static SQL type name of the argument (NusaDB's type-introspection
    // builtin). The type is known here, so fold the call to a constant TEXT literal; the executor
    // never sees this node.
    if matches!(func, F::NusadbTypeof) {
        let [arg] = args else {
            return Err(Error::ArityMismatch {
                context: "function `nusadb_typeof`".to_owned(),
                expected: 1,
                found: args.len(),
            });
        };
        // A bare, undecorated `NULL` has no determined type — report `unknown` (the reference
        // engine's pseudo-type for a typeless NULL) rather than failing to type-check the argument.
        // A typed NULL such as `NULL::int` is a cast, not a bare NULL, so it still reports `integer`.
        if is_bare_null(arg) {
            return Ok(TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Text("unknown".to_owned())),
                ty: Text,
            });
        }
        let typed = analyze_expr_agg(arg, scope, catalog, None, aggregates.as_deref_mut())?;
        let name = crate::executor::ops::info_schema_data_type(typed.ty).to_owned();
        return Ok(TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Text(name)),
            ty: Text,
        });
    }
    if matches!(func, F::Grouping) {
        if args.is_empty() {
            return Err(Error::FunctionArgs(
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
        F::Extract
            | F::DatePart
            | F::DateTrunc
            | F::Age
            | F::ToChar
            | F::ToDate
            | F::ToTimestamp
            | F::AtTimeZone
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
            | F::Log10
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
    // AREA(g) / CENTER(g) are polymorphic over the geometric kind — a `box` or a `circle` — so the
    // single argument's kind is validated directly, not via the fixed (single-kind) table.
    if matches!(func, F::GeomArea | F::GeomCenter) {
        return analyze_geom_measure(func, args, scope, catalog, aggregates);
    }
    // NPOINTS(g) is polymorphic over the vertex-carrying kinds — a `path` or a `polygon` — so the
    // single argument's kind is validated directly, not via the fixed (single-kind) table.
    if matches!(func, F::GeomNpoints) {
        return analyze_geom_npoints(func, args, scope, catalog, aggregates);
    }
    // ARRAY_LENGTH(arr, dim) / ARRAY_TO_STRING(arr, sep) take an array of any element type — the
    // element type is polymorphic, so they are not expressible with the fixed table.
    if matches!(
        func,
        F::ArrayLength | F::ArrayLower | F::ArrayUpper | F::ArrayToString | F::TrimArray
    ) {
        return analyze_array_function(func, args, scope, catalog, aggregates);
    }
    // ARRAY_FILL(value, dims) builds an array whose element type is the first argument's, so its
    // result type is polymorphic — not expressible with the fixed table.
    if matches!(func, F::ArrayFill) {
        return analyze_array_fill(args, scope, catalog, aggregates);
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
    // JSONB_EXTRACT_PATH[_TEXT](json, VARIADIC path text) is variadic — not table-shaped.
    if matches!(func, F::JsonExtractPath | F::JsonExtractPathText) {
        return analyze_json_extract_path(func, args, scope, catalog, aggregates);
    }
    // XMLCONCAT(VARIADIC xml) is variadic — not table-shaped.
    if matches!(func, F::XmlConcat) {
        return analyze_xmlconcat(args, scope, catalog, aggregates);
    }
    // CARDINALITY / ARRAY_NDIMS (→ INT) and ARRAY_DIMS (→ TEXT) take one array of any element type —
    // not expressible with the fixed table since the element type is polymorphic.
    if matches!(func, F::Cardinality | F::ArrayDims | F::ArrayNdims) {
        let name = func.name();
        let [arg_expr] = args else {
            return Err(Error::FunctionArgs(format!(
                "{name}() expects 1 argument, got {}",
                args.len()
            )));
        };
        let arg = analyze_expr_agg(arg_expr, scope, catalog, None, aggregates.as_deref_mut())?;
        if !matches!(arg.ty, ColumnType::Array(_)) && !is_null_literal(&arg) {
            return Err(Error::FunctionArgs(format!(
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
            return Err(Error::FunctionArgs(format!(
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
            return Err(Error::FunctionArgs(format!(
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
            return Err(Error::FunctionArgs(format!(
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
    if matches!(
        func,
        F::L2Distance | F::CosineDistance | F::InnerProduct | F::L1Distance
    ) {
        return analyze_vector_function(func, args, scope, catalog, aggregates);
    }
    // The unary vector functions take one VECTOR: VECTOR_DIMS → INT, VECTOR_NORM → FLOAT.
    if matches!(func, F::VectorDims | F::VectorNorm) {
        return analyze_vector_unary(func, args, scope, catalog, aggregates);
    }
    // The INET/CIDR functions accept either network type (not expressible in the fixed table, which
    // needs an exact argument type) and their result type varies (TEXT/INT/CIDR/INET/passthrough).
    if matches!(
        func,
        F::InetHost
            | F::InetMasklen
            | F::InetFamily
            | F::InetNetwork
            | F::InetBroadcast
            | F::InetSetMasklen
            | F::InetNetmask
            | F::InetHostmask
            | F::InetAbbrev
            | F::InetMerge
            | F::InetSameFamily
    ) {
        return analyze_inet_function(func, args, scope, catalog, aggregates);
    }
    // The range accessors take any range kind and their result type follows the element kind, so
    // they are not expressible in the fixed table either.
    if matches!(
        func,
        F::RangeLower
            | F::RangeUpper
            | F::RangeIsEmpty
            | F::RangeLowerInc
            | F::RangeUpperInc
            | F::RangeLowerInf
            | F::RangeUpperInf
    ) {
        return analyze_range_function(func, args, scope, catalog, aggregates);
    }
    // The range constructors take two element-typed bounds plus an optional bound-flags string, and
    // their result type is the range they name — neither fits the fixed table.
    if let Some(kind) = func.range_kind() {
        return analyze_range_constructor(func, kind, args, scope, catalog, aggregates);
    }
    // `range_merge` takes two ranges and returns a range of their (shared) kind — a result type that
    // follows the argument, so it is not expressible in the fixed table either.
    if matches!(func, F::RangeMerge) {
        return analyze_range_merge(func, args, scope, catalog, aggregates);
    }
    // `lower`/`upper` are overloaded: over a range they yield a bound, over text they fold case.
    // The argument decides, so it is analyzed once here and handed to whichever form wins —
    // re-analyzing to pick would double the work per call and square it for every nested one.
    if matches!(func, F::Lower | F::Upper)
        && let [arg] = args
    {
        let a = analyze_expr_agg(arg, scope, catalog, Some(Text), aggregates.as_deref_mut())?;
        if let ColumnType::Range(kind) = a.ty {
            let func = if func == F::Lower {
                F::RangeLower
            } else {
                F::RangeUpper
            };
            return Ok(range_accessor(func, a, kind));
        }
        // Not a range, so this is the text form. Applying its `ScalarSig::Fixed(&[Text], &[], Text)`
        // entry here keeps the already-analyzed argument instead of sending it back through the
        // table below; the arity and type errors read the same either way.
        if a.ty != Text && !is_null_literal(&a) {
            return Err(Error::TypeMismatch {
                context: format!("{}() argument 1", func.name()),
                expected: Text,
                found: a.ty,
            });
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::ScalarFunction {
                func,
                args: vec![a],
            },
            ty: Text,
        });
    }
    // GET_BIT/SET_BIT take a bit string (any width) and integer positions — not expressible in the
    // fixed table, and SET_BIT's result type is the input's own bit type.
    if matches!(func, F::BitGetBit | F::BitSetBit) {
        return analyze_bit_function(func, args, scope, catalog, aggregates);
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
        // NUSADB_TYPEOF is folded to a TEXT literal by the early `matches!(func, F::NusadbTypeof)` branch
        // above, so it never reaches this table either.
        F::NusadbTypeof => {
            unreachable!("NUSADB_TYPEOF is folded before the scalar signature table")
        },
        // The sequence built-ins arrive as generic function calls (`FunctionCall`) and are analyzed
        // by `analyze_sequence_function` before this typed-builtin table, so they never reach here.
        F::SequenceNext | F::SequenceCurrent | F::SequenceSet => {
            unreachable!("sequence built-ins are analyzed before the scalar signature table")
        },
        // ASCII takes one TEXT argument and returns INT (the LENGTH family is intercepted by
        // `analyze_text_polymorphic` above — Text-or-BYTEA).
        F::Ascii => ScalarSig::Fixed(&[Text], &[], Int),
        // MACADDR8_SET7BIT(macaddr8) → macaddr8 (sets the locally-administered bit of the first byte).
        F::Macaddr8Set7bit => ScalarSig::Fixed(&[ColumnType::Macaddr8], &[], ColumnType::Macaddr8),
        // Geometric constructors and box accessors.
        // POINT(x, y) → point (two FLOAT coordinates).
        F::PointCtor => ScalarSig::Fixed(
            &[ColumnType::Float, ColumnType::Float],
            &[],
            ColumnType::Geometry(GeomKind::Point),
        ),
        // BOX(p1, p2) → box (two point corners).
        F::BoxCtor => ScalarSig::Fixed(
            &[
                ColumnType::Geometry(GeomKind::Point),
                ColumnType::Geometry(GeomKind::Point),
            ],
            &[],
            ColumnType::Geometry(GeomKind::Box),
        ),
        // HEIGHT/WIDTH(box) → FLOAT. (AREA/CENTER are polymorphic over box|circle and handled before
        // this table.)
        F::GeomHeight | F::GeomWidth => ScalarSig::Fixed(
            &[ColumnType::Geometry(GeomKind::Box)],
            &[],
            ColumnType::Float,
        ),
        // RADIUS/DIAMETER(circle) → FLOAT.
        F::GeomRadius | F::GeomDiameter => ScalarSig::Fixed(
            &[ColumnType::Geometry(GeomKind::Circle)],
            &[],
            ColumnType::Float,
        ),
        // NPOINTS is polymorphic over path|polygon and analyzed before this table.
        F::GeomNpoints => {
            unreachable!("NPOINTS is analyzed before the scalar signature table")
        },
        // ISOPEN/ISCLOSED(path) → BOOL.
        F::GeomIsOpen | F::GeomIsClosed => ScalarSig::Fixed(
            &[ColumnType::Geometry(GeomKind::Path)],
            &[],
            ColumnType::Bool,
        ),
        // AREA/CENTER are polymorphic over box|circle and analyzed before this table.
        F::GeomArea | F::GeomCenter => {
            unreachable!("AREA/CENTER are analyzed before the scalar signature table")
        },
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
        // PARSE_IDENT(text [, strict bool]) → TEXT[].
        F::ParseIdent => ScalarSig::Fixed(
            &[Text],
            &[ColumnType::Bool],
            ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
        ),
        // GET_BYTE(bytea, n) → INT; SET_BYTE(bytea, n, v) → BYTEA.
        F::GetByte => ScalarSig::Fixed(&[ColumnType::Bytes, Int], &[], Int),
        F::SetByte => ScalarSig::Fixed(&[ColumnType::Bytes, Int, Int], &[], ColumnType::Bytes),
        // UPPER/LOWER/REVERSE/INITCAP, the MD5 fingerprint, the quoting helpers
        // QUOTE_LITERAL/QUOTE_IDENT, and `current_setting(name)` take one TEXT argument → TEXT.
        F::Upper
        | F::Lower
        | F::Reverse
        | F::Initcap
        | F::Md5
        | F::QuoteLiteral
        | F::QuoteNullable
        | F::QuoteIdent
        | F::CurrentSetting => ScalarSig::Fixed(&[Text], &[], Text),
        // The SHA-2 digests take one BYTEA argument → BYTEA (a bare string literal is coerced to
        // bytea by the unknown-literal rule), so `encode(sha256(x), 'hex')` round-trips.
        F::Sha224 | F::Sha256 | F::Sha384 | F::Sha512 => {
            ScalarSig::Fixed(&[ColumnType::Bytes], &[], ColumnType::Bytes)
        },
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
        // JSONB_PATH_MATCH(json, path) → BOOL (or NULL).
        F::JsonbPathMatch => ScalarSig::Fixed(&[ColumnType::Json, Text], &[], ColumnType::Bool),
        // Full-text search (F1): TO_TSVECTOR/TO_TSQUERY/PLAINTO_TSQUERY([config,] text) → the
        // canonical tsvector/tsquery text form. The optional leading argument is the configuration;
        // with one argument the default configuration applies (rejected at evaluation until a
        // non-`simple` configuration exists).
        F::ToTsvector => ScalarSig::Fixed(&[Text], &[Text], ColumnType::Tsvector),
        F::ToTsquery | F::PlaintoTsquery => ScalarSig::Fixed(&[Text], &[Text], ColumnType::Tsquery),
        // TS_RANK / TS_RANK_CD(tsvector, tsquery [, normalization INT]) → the relevance score as a
        // REAL. The optional third argument is the normalization bit-mask.
        F::TsRank | F::TsRankCd => ScalarSig::Fixed(
            &[ColumnType::Tsvector, ColumnType::Tsquery],
            &[Int],
            ColumnType::Real,
        ),
        // NUMNODE(tsquery) → INT; STRIP(tsvector) → tsvector; SETWEIGHT(tsvector, weight text) →
        // tsvector. The LENGTH family (which gains a tsvector arm) is handled by
        // `analyze_text_polymorphic` above.
        F::Numnode => ScalarSig::Fixed(&[ColumnType::Tsquery], &[], Int),
        F::Strip => ScalarSig::Fixed(&[ColumnType::Tsvector], &[], ColumnType::Tsvector),
        F::Setweight => ScalarSig::Fixed(&[ColumnType::Tsvector, Text], &[], ColumnType::Tsvector),
        // RRF_SCORE(rank [, k]) → the Reciprocal Rank Fusion contribution 1/(k + rank) as FLOAT,
        // k defaulting to 60.
        F::RrfScore => ScalarSig::Fixed(&[Int], &[Int], ColumnType::Float),
        // JSONB_PATH_QUERY_FIRST(json, path) → JSON (the first match, or NULL).
        F::JsonbPathQueryFirst | F::JsonbPathQueryArray => {
            ScalarSig::Fixed(&[ColumnType::Json, Text], &[], ColumnType::Json)
        },
        // JSON_OBJECT(pairs text[]) or JSON_OBJECT(keys text[], values text[]) → JSON object.
        F::JsonObject => ScalarSig::Fixed(
            &[ColumnType::Array(nusadb_core::engine::ArrayElem::Text)],
            &[ColumnType::Array(nusadb_core::engine::ArrayElem::Text)],
            ColumnType::Json,
        ),
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
        // JSONB_SET_LAX adds an optional null_value_treatment TEXT after the create_missing flag.
        F::JsonbSetLax => ScalarSig::Fixed(
            &[
                ColumnType::Json,
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
                ColumnType::Json,
            ],
            &[ColumnType::Bool, ColumnType::Text],
            ColumnType::Json,
        ),
        // XML_IS_WELL_FORMED[_DOCUMENT|_CONTENT](text) → BOOL.
        F::XmlIsWellFormed | F::XmlIsWellFormedDocument | F::XmlIsWellFormedContent => {
            ScalarSig::Fixed(&[Text], &[], ColumnType::Bool)
        },
        // XMLCOMMENT(text) → XML.
        F::XmlComment => ScalarSig::Fixed(&[Text], &[], ColumnType::Xml),
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
        // regexp_count(string, pattern [, start int [, flags text]]) — arg 3 is the 1-based start
        // position (integer), not flags.
        F::RegexpCount => ScalarSig::Fixed(&[Text, Text], &[Int, Text], Int),
        // regexp_instr(string, pattern [, start [, N [, endoption [, flags [, subexpr]]]]]) — the
        // 3rd/4th args are integer position/occurrence, not flags (flags are 6th).
        F::RegexpInstr => ScalarSig::Fixed(&[Text, Text], &[Int, Int, Int, Text, Int], Int),
        // regexp_substr(string, pattern [, start int [, N int [, flags text [, subexpr int]]]]).
        // The 3rd argument is the 1-based start position (an integer), NOT flags — flags come 5th.
        F::RegexpSubstr => ScalarSig::Fixed(&[Text, Text], &[Int, Int, Text, Int], Text),
        // CONCAT needs ≥1 value; CONCAT_WS needs at least its separator.
        // CONCAT/CONCAT_WS and the LENGTH family are intercepted by
        // `analyze_text_polymorphic` above and never reach
        // this table.
        F::Concat | F::ConcatWs | F::Length | F::OctetLength | F::BitLength => {
            unreachable!("text-polymorphic functions are handled before the signature table")
        },
        // Niladic clock built-ins resolved from the statement's wall clock.
        F::Now | F::CurrentTimestamp | F::StatementTimestamp => {
            ScalarSig::Fixed(&[], &[], ColumnType::TimestampTz)
        },
        F::CurrentDate => ScalarSig::Fixed(&[], &[], ColumnType::Date),
        F::CurrentTime => ScalarSig::Fixed(&[], &[], ColumnType::Time),
        F::LocalTimestamp => ScalarSig::Fixed(&[], &[], ColumnType::Timestamp),
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
        F::MakeTimestamptz => ScalarSig::Fixed(
            &[Int, Int, Int, Int, Int, Int],
            &[],
            ColumnType::TimestampTz,
        ),
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
        // ENCODE(bytea, format)→text and CONVERT_FROM(bytea, encoding)→text share the (Bytes, Text)→
        // Text shape; DECODE(text, format)→bytea and CONVERT_TO(text, encoding)→bytea share
        // (Text, Text)→Bytes. The executor distinguishes them by function.
        F::Encode | F::ConvertFrom => ScalarSig::Fixed(&[ColumnType::Bytes, Text], &[], Text),
        F::Decode | F::ConvertTo => ScalarSig::Fixed(&[Text, Text], &[], ColumnType::Bytes),
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
        | F::DatePart
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
        | F::Log10
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
        | F::ArrayFill
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
        | F::TrimArray
        | F::JsonExtractPath
        | F::JsonExtractPathText
        | F::XmlConcat
        | F::ArrayNdims
        | F::L2Distance
        | F::CosineDistance
        | F::InnerProduct
        | F::L1Distance
        | F::VectorDims
        | F::VectorNorm
        | F::InetHost
        | F::InetMasklen
        | F::InetFamily
        | F::InetNetwork
        | F::InetBroadcast
        | F::InetSetMasklen
        | F::InetNetmask
        | F::InetHostmask
        | F::InetAbbrev
        | F::InetMerge
        | F::InetSameFamily
        | F::BitGetBit
        | F::BitSetBit
        | F::RangeLower
        | F::RangeUpper
        | F::RangeIsEmpty
        | F::RangeLowerInc
        | F::RangeUpperInc
        | F::RangeLowerInf
        | F::RangeUpperInf
        | F::Int4Range
        | F::Int8Range
        | F::NumRange
        | F::DateRange
        | F::TsRange
        | F::TsTzRange
        | F::RangeMerge
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
        return Err(Error::FunctionArgs(format!(
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
        // A bare string literal for a JSON / array / temporal / … parameter is coerced to that type
        // (the unknown-literal rule), so `jsonb_set('{...}', '{a}', '9')` type-checks like its
        // explicit-cast form. A no-op for a TEXT parameter or a non-literal argument.
        let typed = coerce_text_literal_to(typed, expected);
        // A FLOAT parameter also accepts an INT or NUMERIC argument (coerced to f64 at eval) — the
        // same widening `assignable` allows, and needed since a plain decimal literal now types as
        // NUMERIC, e.g. `SETSEED(0.5)`. An unconstrained NUMERIC parameter likewise accepts
        // any NUMERIC (regardless of declared precision/scale) or an INT, e.g. `SCALE(12.34)`.
        let coercible = (expected == ColumnType::Float
            && matches!(typed.ty, ColumnType::Int | ColumnType::Numeric { .. }))
            || (matches!(expected, ColumnType::Numeric { .. })
                && matches!(typed.ty, ColumnType::Int | ColumnType::Numeric { .. }))
            // TO_HEX accepts any integer width (int2/int4/int8): the evaluator masks int2/int4 to
            // 32 bits and renders int8 with all 64, mirroring the reference engine's two overloads.
            || (func == ast::ScalarFunc::ToHex
                && matches!(
                    typed.ty,
                    ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt
                ));
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
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 1 argument, got {}",
            args.len()
        )));
    }
    if func == F::JsonBuildObject && !args.len().is_multiple_of(2) {
        return Err(Error::FunctionArgs(format!(
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
        return Err(Error::FunctionArgs(format!(
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
            return Err(Error::FunctionArgs(
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
/// `ARRAY_FILL(value, dims)` — a one-dimensional array of `value` repeated `dims[1]` times. The
/// element type is `value`'s type (so the result is polymorphic), and `dims` is a 1-D `INT[]` (NusaDB
/// arrays are one-dimensional). A bare untyped NULL `value` has no array element type and is rejected,
/// matching the reference engine's "could not determine polymorphic type".
fn analyze_array_fill(
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let [value_expr, dims_expr] = args else {
        return Err(Error::FunctionArgs(format!(
            "array_fill() expects 2 arguments, got {}",
            args.len()
        )));
    };
    let value = analyze_expr_agg(value_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let Some(array_elem) = nusadb_core::engine::ArrayElem::from_column_type(value.ty) else {
        return Err(Error::Unsupported(format!(
            "array_fill() does not support an element of type {:?}",
            value.ty
        )));
    };
    let dims_ty = ColumnType::Array(nusadb_core::engine::ArrayElem::Int);
    let dims = analyze_expr_agg(dims_expr, scope, catalog, Some(dims_ty), aggregates)?;
    if dims.ty != dims_ty && !is_null_literal(&dims) {
        return Err(Error::TypeMismatch {
            context: "array_fill() dimensions".to_owned(),
            expected: dims_ty,
            found: dims.ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func: ast::ScalarFunc::ArrayFill,
            args: vec![value, dims],
        },
        ty: ColumnType::Array(array_elem),
    })
}

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
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    };
    let arr = analyze_expr_agg(arr_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    if !matches!(arr.ty, ColumnType::Array(_)) && !is_null_literal(&arr) {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects an array first argument, got {:?}",
            arr.ty
        )));
    }
    let (second_ty, result) = if func == F::ArrayToString {
        (ColumnType::Text, ColumnType::Text)
    } else if func == F::TrimArray {
        // TRIM_ARRAY(arr, n) removes the last `n` elements, keeping the array's own type.
        (ColumnType::Int, arr.ty)
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
        return Err(Error::FunctionArgs(format!(
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
        return Err(Error::FunctionArgs(format!(
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

/// Analyze `XMLCONCAT(VARIADIC xml)`: one or more `XML` arguments, yielding `XML`. A bare `NULL`
/// argument is allowed (skipped at evaluation).
fn analyze_xmlconcat(
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    if args.is_empty() {
        return Err(Error::FunctionArgs(
            "xmlconcat() expects at least one argument".to_owned(),
        ));
    }
    let mut analyzed = Vec::with_capacity(args.len());
    for arg in args {
        let value = analyze_expr_agg(
            arg,
            scope,
            catalog,
            Some(ColumnType::Xml),
            aggregates.as_deref_mut(),
        )?;
        if value.ty != ColumnType::Xml && !is_null_literal(&value) {
            return Err(Error::TypeMismatch {
                context: "xmlconcat() argument".to_owned(),
                expected: ColumnType::Xml,
                found: value.ty,
            });
        }
        analyzed.push(value);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func: ast::ScalarFunc::XmlConcat,
            args: analyzed,
        },
        ty: ColumnType::Xml,
    })
}

/// Analyze `JSONB_EXTRACT_PATH[_TEXT](json, VARIADIC path text)`: a JSON (or text) document followed
/// by one or more text path elements. The `_TEXT` form yields `TEXT`, the plain form yields `JSON`.
fn analyze_json_extract_path(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let text_like = |ty: ColumnType| {
        matches!(
            ty,
            ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
        )
    };
    let Some((json_expr, path_exprs)) = args.split_first() else {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects at least 2 arguments, got 0"
        )));
    };
    if path_exprs.is_empty() {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects at least one path element"
        )));
    }
    let json = analyze_expr_agg(
        json_expr,
        scope,
        catalog,
        Some(ColumnType::Json),
        aggregates.as_deref_mut(),
    )?;
    if json.ty != ColumnType::Json && !text_like(json.ty) && !is_null_literal(&json) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() first argument"),
            expected: ColumnType::Json,
            found: json.ty,
        });
    }
    let mut analyzed = vec![json];
    for path_expr in path_exprs {
        let elem = analyze_expr_agg(
            path_expr,
            scope,
            catalog,
            Some(ColumnType::Text),
            aggregates.as_deref_mut(),
        )?;
        if !text_like(elem.ty) && !is_null_literal(&elem) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() path element"),
                expected: ColumnType::Text,
                found: elem.ty,
            });
        }
        analyzed.push(elem);
    }
    let ty = if func == F::JsonExtractPathText {
        ColumnType::Text
    } else {
        ColumnType::Json
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: analyzed,
        },
        ty,
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
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 3 arguments, got {}",
            args.len()
        )));
    };
    let array = analyze_expr_agg(arr_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let ColumnType::Array(array_elem) = array.ty else {
        return Err(Error::FunctionArgs(format!(
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

/// Analyze the single-argument `TRUNC`. On a `MACADDR8` argument it is the MACADDR8 overload (zero
/// the trailing bytes, returning `MACADDR8`); otherwise it is the numeric single-argument form
/// (truncate toward zero, preserving the unified numeric type). The argument is analyzed once so an
/// aggregate inside it is recorded a single time.
fn analyze_trunc_unary(
    arg_expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ColumnType::{Float, Int};
    let arg = analyze_expr_agg(arg_expr, scope, catalog, Some(Float), aggregates)?;
    let ty = if arg.ty == ColumnType::Macaddr8 {
        ColumnType::Macaddr8
    } else if arg.ty == ColumnType::Macaddr {
        ColumnType::Macaddr
    } else if is_numeric(arg.ty) || is_null_literal(&arg) {
        widen_numeric(Int, arg.ty)
    } else {
        return Err(Error::TypeMismatch {
            context: "trunc() argument 1".to_owned(),
            expected: Float,
            found: arg.ty,
        });
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func: ast::ScalarFunc::Trunc,
            args: vec![arg],
        },
        ty,
    })
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
    // TRUNC has a MACADDR8 overload (`trunc(macaddr8)` zeros the last five bytes) that dispatches on
    // the single argument's type, so it is analyzed on its own path rather than the numeric table.
    if func == F::Trunc
        && let [arg_expr] = args
    {
        return analyze_trunc_unary(arg_expr, scope, catalog, aggregates);
    }
    // (min arity, max arity, result is always FLOAT). LN/LOG/LOG10 are not forced to float: they are
    // polymorphic (numeric/int argument → exact NUMERIC, float → double precision — see `is_log`),
    // so they share the non-float arms (`LN`/`LOG10` with the one-arg group, `LOG` with the 1..=2 one).
    let (min, max, force_float) = match func {
        F::Abs | F::Ceil | F::Floor | F::Sign | F::Ln | F::Log10 => (1, 1, false),
        F::Round | F::Trunc | F::Log => (1, 2, false),
        F::Mod => (2, 2, false),
        F::Power | F::Atan2 => (2, 2, true),
        F::Sqrt
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
    // LN/LOG/LOG10 keep a numeric argument numeric (no float hint) and return NUMERIC unless the
    // argument is a float, in which case they return double precision.
    let is_log = matches!(func, F::Ln | F::Log | F::Log10);
    let name = func.name();
    if args.len() < min || args.len() > max {
        let arity = if min == max {
            min.to_string()
        } else {
            format!("{min}..={max}")
        };
        return Err(Error::FunctionArgs(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    let mut typed_args = Vec::with_capacity(args.len());
    let mut unified = Int;
    for (i, arg) in args.iter().enumerate() {
        // The log family keeps a numeric argument numeric (and types a bare NULL as NUMERIC); the
        // others prefer float for a bare literal.
        let hint = if is_log {
            Some(NUMERIC_ANY)
        } else {
            Some(Float)
        };
        let typed = analyze_expr_agg(arg, scope, catalog, hint, aggregates.as_deref_mut())?;
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
    let result = if is_log {
        // NUMERIC (unconstrained) for an int/numeric argument; double precision for a float.
        if unified == Float { Float } else { NUMERIC_ANY }
    } else if force_float {
        Float
    } else {
        unified
    };
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
/// Analyze `AREA(g)` / `CENTER(g)`, each taking a single geometric argument that is a `box` or a
/// `circle`. `AREA` yields the `FLOAT` area; `CENTER` yields the `point` center. A non-geometry
/// argument, or a `point` (which has no area or distinct center), is a loud type error.
fn analyze_geom_measure(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use nusadb_core::engine::GeomKind;
    let name = func.name();
    let [arg] = args else {
        return Err(Error::ArityMismatch {
            context: format!("function `{name}`"),
            expected: 1,
            found: args.len(),
        });
    };
    let typed = analyze_expr_agg(arg, scope, catalog, None, aggregates)?;
    if !matches!(
        typed.ty,
        ColumnType::Geometry(GeomKind::Box | GeomKind::Circle)
    ) {
        return Err(Error::TypeMismatch {
            context: format!("argument to function `{name}`"),
            expected: ColumnType::Geometry(GeomKind::Box),
            found: typed.ty,
        });
    }
    let ty = if func == ast::ScalarFunc::GeomCenter {
        ColumnType::Geometry(GeomKind::Point)
    } else {
        ColumnType::Float
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![typed],
        },
        ty,
    })
}

/// Analyze `NPOINTS(g)`, taking a single geometric argument that carries a vertex list — a `path` or
/// a `polygon` — and yielding the `INT` vertex count. A non-geometry argument, or a geometric kind
/// with no vertex list (`point`/`box`/`circle`/`lseg`/`line`), is a loud type error.
fn analyze_geom_npoints(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use nusadb_core::engine::GeomKind;
    let name = func.name();
    let [arg] = args else {
        return Err(Error::ArityMismatch {
            context: format!("function `{name}`"),
            expected: 1,
            found: args.len(),
        });
    };
    let typed = analyze_expr_agg(arg, scope, catalog, None, aggregates)?;
    if !matches!(
        typed.ty,
        ColumnType::Geometry(GeomKind::Path | GeomKind::Polygon)
    ) {
        return Err(Error::TypeMismatch {
            context: format!("argument to function `{name}`"),
            expected: ColumnType::Geometry(GeomKind::Path),
            found: typed.ty,
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![typed],
        },
        ty: ColumnType::Int,
    })
}

fn analyze_conditional_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    // Two different enum types have no common type — refuse before unifying (42846). The reference
    // engine spells the function upper-case in this message.
    reject_mixed_enum_operands(&name.to_uppercase(), args.iter(), scope, catalog)?;
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
        return Err(Error::FunctionArgs(format!(
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
        return Err(Error::FunctionArgs(format!(
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

/// Analyze a unary vector function: one `VECTOR` argument. `VECTOR_DIMS` returns `INT` (the
/// dimension count), `VECTOR_NORM` returns `FLOAT` (the Euclidean norm). A bare `NULL` is allowed.
/// Analyze an `INET`/`CIDR` scalar function. The first argument must be a network type; `SET_MASKLEN`
/// takes a second `INT`. The result type is per-function (`HOST`→`TEXT`, `MASKLEN`/`FAMILY`→`INT`,
/// `NETWORK`→`CIDR`, `BROADCAST`→`INET`, `SET_MASKLEN`→the input's own type).
/// Analyze `GET_BIT(bits, n)` → `INT` and `SET_BIT(bits, n, v)` → the input's own type. The first
/// argument is a bit string or a `BYTEA` (`SET_BIT` returns the same type it was given); the
/// remaining arguments are `INT` positions/values.
fn analyze_bit_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let want = if func == F::BitSetBit { 3 } else { 2 };
    if args.len() != want {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects {want} argument(s), got {}",
            args.len()
        )));
    }
    let mut typed = Vec::with_capacity(want);
    for (i, arg) in args.iter().enumerate() {
        let t = analyze_expr_agg(arg, scope, catalog, None, aggregates.as_deref_mut())?;
        let ok = if i == 0 {
            is_bit_type(t.ty) || matches!(t.ty, ColumnType::Bytes)
        } else {
            matches!(t.ty, ColumnType::Int)
        };
        if !ok && !is_null_literal(&t) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument {}", i + 1),
                expected: if i == 0 {
                    ColumnType::VarBit(None)
                } else {
                    ColumnType::Int
                },
                found: t.ty,
            });
        }
        typed.push(t);
    }
    let ty = if func == F::BitGetBit {
        ColumnType::Int
    } else {
        typed.first().map_or(ColumnType::VarBit(None), |a| a.ty)
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction { func, args: typed },
        ty,
    })
}

/// Analyze a range accessor: one argument of any range kind. `LOWER`/`UPPER` yield the element
/// type of that kind; the rest yield `BOOL`. A bare string literal argument is coerced to the range
/// type only when the call already fixes the kind, which it never does — so a literal must be cast
/// (`'[1,10)'::int4range`), keeping `lower('[1,10)')` from silently meaning the text function.
fn analyze_range_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    let [a_expr] = args else {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 1 argument, got {}",
            args.len()
        )));
    };
    // A bare NULL carries no element kind, so `LOWER`/`UPPER` would have no result type to report.
    // Catch it here — analyzing it first would raise the generic no-type-context error instead.
    if is_bare_null(a_expr) {
        return Err(Error::AmbiguousNull {
            context: format!("{name}() range argument"),
        });
    }
    let a = analyze_expr_agg(a_expr, scope, catalog, None, aggregates)?;
    let ColumnType::Range(kind) = a.ty else {
        return Err(Error::TypeMismatch {
            context: format!("{name}() argument"),
            expected: ColumnType::Range(nusadb_core::engine::RangeKind::Int),
            found: a.ty,
        });
    };
    Ok(range_accessor(func, a, kind))
}

/// Analyze a range constructor: two bounds of the element type, plus an optional `TEXT` bound-flags
/// argument (`'[)'`, `'(]'`, `'[]'`, `'()'`). A `NULL` bound is *unbounded on that side* rather than
/// a `NULL` result, so a bare `NULL` argument is accepted and types from the element hint.
fn analyze_range_constructor(
    func: ast::ScalarFunc,
    kind: nusadb_core::engine::RangeKind,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    if args.len() < 2 || args.len() > 3 {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 2..=3 argument(s), got {}",
            args.len()
        )));
    }
    let elem = kind.element_type();
    let mut typed_args = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().take(2).enumerate() {
        let a = analyze_expr_agg(arg, scope, catalog, Some(elem), aggregates.as_deref_mut())?;
        // A bare string literal takes the element type, as it would for any other parameter.
        let a = coerce_text_literal_to(a, elem);
        if !is_range_element(a.ty, kind) && !is_null_literal(&a) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() argument {}", i + 1),
                expected: elem,
                found: a.ty,
            });
        }
        typed_args.push(a);
    }
    if let Some(flags_expr) = args.get(2) {
        let f = analyze_expr_agg(
            flags_expr,
            scope,
            catalog,
            Some(ColumnType::Text),
            aggregates,
        )?;
        if f.ty != ColumnType::Text && !is_null_literal(&f) {
            return Err(Error::TypeMismatch {
                context: format!("{name}() bound flags"),
                expected: ColumnType::Text,
                found: f.ty,
            });
        }
        typed_args.push(f);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: ColumnType::Range(kind),
    })
}

/// Analyze `RANGE_MERGE(range, range)`: two ranges of the same element kind, yielding that range
/// type. There is no element form, so both arguments must already be ranges — a bare string literal
/// carries no kind and needs an explicit cast (`'[1,5)'::int4range`), exactly as the range
/// strict-order operators require.
fn analyze_range_merge(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    let [a_expr, b_expr] = args else {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    };
    let a = analyze_expr_agg(a_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    let b = analyze_expr_agg(b_expr, scope, catalog, None, aggregates)?;
    let (ColumnType::Range(ka), ColumnType::Range(kb)) = (a.ty, b.ty) else {
        // Point the error at whichever operand is not a range.
        let (found, expected) = if is_range_type(a.ty) {
            (b.ty, a.ty)
        } else {
            (a.ty, ColumnType::Range(nusadb_core::engine::RangeKind::Int))
        };
        return Err(Error::TypeMismatch {
            context: format!("{name}() arguments"),
            expected,
            found,
        });
    };
    if ka != kb {
        return Err(Error::TypeMismatch {
            context: format!("{name}() arguments"),
            expected: ColumnType::Range(ka),
            found: ColumnType::Range(kb),
        });
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![a, b],
        },
        ty: ColumnType::Range(ka),
    })
}

/// Build the typed call for a range accessor whose argument is already analyzed to `kind`.
/// `LOWER`/`UPPER` yield the element type; the predicates yield `BOOL`.
fn range_accessor(
    func: ast::ScalarFunc,
    arg: TypedExpr,
    kind: nusadb_core::engine::RangeKind,
) -> TypedExpr {
    use ast::ScalarFunc as F;
    let ty = if matches!(func, F::RangeLower | F::RangeUpper) {
        kind.element_type()
    } else {
        ColumnType::Bool
    };
    TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![arg],
        },
        ty,
    }
}

fn analyze_inet_function(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::ScalarFunc as F;
    let name = func.name();
    let want = if matches!(func, F::InetSetMasklen | F::InetMerge | F::InetSameFamily) {
        2
    } else {
        1
    };
    if args.len() != want {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects {want} argument(s), got {}",
            args.len()
        )));
    }
    // The arity check above already guarantees these are present, so reaching either is a bug in
    // this function rather than anything the caller wrote — which is why they do not report `42883`
    // and blame the call.
    let a_expr = args.first().ok_or_else(|| {
        Error::Internal(format!(
            "{name}() passed the arity check with no first argument"
        ))
    })?;
    let a = analyze_expr_agg(a_expr, scope, catalog, None, aggregates.as_deref_mut())?;
    if !is_inet_type(a.ty) && !is_null_literal(&a) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() argument"),
            expected: ColumnType::Inet,
            found: a.ty,
        });
    }
    let mut typed_args = vec![a.clone()];
    let ty = match func {
        F::InetHost | F::InetAbbrev => ColumnType::Text,
        F::InetMasklen | F::InetFamily => ColumnType::Int,
        F::InetNetwork => ColumnType::Cidr,
        F::InetBroadcast | F::InetNetmask | F::InetHostmask => ColumnType::Inet,
        // INET_MERGE(inet, inet) → CIDR and INET_SAME_FAMILY(inet, inet) → BOOL both take a second
        // INET/CIDR argument; validate and record it here.
        F::InetMerge | F::InetSameFamily => {
            let b_expr = args.get(1).ok_or_else(|| {
                Error::Internal(format!(
                    "{name}() passed the arity check with no second argument"
                ))
            })?;
            let b = analyze_expr_agg(b_expr, scope, catalog, None, aggregates)?;
            if !is_inet_type(b.ty) && !is_null_literal(&b) {
                return Err(Error::TypeMismatch {
                    context: format!("{name}() second argument"),
                    expected: ColumnType::Inet,
                    found: b.ty,
                });
            }
            typed_args.push(b);
            if func == F::InetMerge {
                ColumnType::Cidr
            } else {
                ColumnType::Bool
            }
        },
        F::InetSetMasklen => {
            let n_expr = args.get(1).ok_or_else(|| {
                Error::Internal(format!(
                    "{name}() passed the arity check with no mask-length argument"
                ))
            })?;
            let n = analyze_expr_agg(n_expr, scope, catalog, None, aggregates)?;
            if !matches!(n.ty, ColumnType::Int) && !is_null_literal(&n) {
                return Err(Error::TypeMismatch {
                    context: format!("{name}() mask length"),
                    expected: ColumnType::Int,
                    found: n.ty,
                });
            }
            typed_args.push(n);
            a.ty // SET_MASKLEN keeps the input's INET/CIDR type.
        },
        _ => unreachable!("analyze_inet_function only handles the INET/CIDR functions"),
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty,
    })
}

fn analyze_vector_unary(
    func: ast::ScalarFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let name = func.name();
    let [a_expr] = args else {
        return Err(Error::FunctionArgs(format!(
            "{name}() expects 1 argument, got {}",
            args.len()
        )));
    };
    let a = analyze_expr_agg(a_expr, scope, catalog, None, aggregates)?;
    if !matches!(a.ty, ColumnType::Vector(_)) && !is_null_literal(&a) {
        return Err(Error::TypeMismatch {
            context: format!("{name}() argument"),
            expected: ColumnType::Vector(0),
            found: a.ty,
        });
    }
    let ty = if matches!(func, ast::ScalarFunc::VectorDims) {
        ColumnType::Int
    } else {
        ColumnType::Float
    };
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: vec![a],
        },
        ty,
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
        .ok_or_else(|| Error::AmbiguousNull {
            context: "an empty ARRAY[] — add an explicit cast, e.g. ARRAY[]::INT[]".to_owned(),
        })?;
    // Map the unified element type to a storable array element. NUMERIC is a supported element type
    // (exact decimals — `ARRAY[1, 2.0]` is `NUMERIC[]`). A **nested** array element makes this a
    // multidimensional array: the type stays the scalar element's array
    // (`ARRAY[[1,2],[3,4]]` is `integer[]`, not `integer[][]`) — the extra dimension lives in the
    // value — so one array level is peeled off here. The rectangular constraint (every sub-array the
    // same length) is enforced at run time by the array constructor's evaluator. Other non-element
    // types (JSON, BYTES, …) are still rejected.
    let elem = match elem_col_ty {
        ColumnType::Array(inner) => inner,
        scalar => nusadb_core::engine::ArrayElem::from_column_type(scalar).ok_or_else(|| {
            Error::Unsupported(format!("ARRAY of {scalar:?} elements is not supported"))
        })?,
    };
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
    // The result element type. A first subscript on an array yields its element type. A *chained*
    // subscript (`a[i][j]`) indexes one dimension deeper into a multidimensional array: `a[i]` was
    // already typed as the scalar element (dimensionality is a value property, not part of the
    // type), and the further `[j]` still yields that scalar. Only a subscript base may chain, so a
    // bare scalar (`(1)[2]`) is still rejected.
    let elem_ty = match base_t.ty {
        ColumnType::Array(elem) => elem.column_type(),
        // A `point` is subscriptable: `p[0]` is its X and `p[1]` its Y (both `FLOAT`, 0-based).
        ColumnType::Geometry(nusadb_core::engine::GeomKind::Point) => ColumnType::Float,
        scalar if matches!(base, ast::Expr::Subscript { .. }) => scalar,
        _ => {
            return Err(Error::TypeMismatch {
                context: "array subscript base".to_owned(),
                expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Int),
                found: base_t.ty,
            });
        },
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
        ty: elem_ty,
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

/// Analyze an ordered-set aggregate `f(args) WITHIN GROUP (ORDER BY key)`. The percentile / `MODE`
/// forms (`PERCENTILE_CONT`, `PERCENTILE_DISC`, `MODE`) take a single `ORDER BY` key that becomes
/// the aggregate's `arg`; a percentile's fraction is a constant in `[0, 1]`. The hypothetical-set
/// forms (`RANK` / `DENSE_RANK` / `PERCENT_RANK` / `CUME_DIST`) take N keys and dispatch to
/// [`analyze_hypothetical_set`]. Registered into the aggregate sink like an ordinary aggregate.
fn analyze_within_group(
    wg: &ast::WithinGroup,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    use ast::AggregateFunc as F;
    let Some(sink) = aggregates else {
        return Err(Error::InvalidGrouping(
            "ordered-set aggregates are only allowed in a SELECT projection, HAVING, or ORDER BY"
                .to_owned(),
        ));
    };
    let func = match wg.func.as_str() {
        "percentile_cont" => F::PercentileCont,
        "percentile_disc" => F::PercentileDisc,
        "mode" => F::Mode,
        "rank" => F::Rank,
        "dense_rank" => F::DenseRank,
        "percent_rank" => F::PercentRank,
        "cume_dist" => F::CumeDist,
        other => {
            return Err(Error::FunctionArgs(format!(
                "ordered-set aggregate `{other}` is not supported (PERCENTILE_CONT / \
                 PERCENTILE_DISC / MODE / RANK / DENSE_RANK / PERCENT_RANK / CUME_DIST only)"
            )));
        },
    };
    // Hypothetical-set aggregates take N ORDER BY keys (one per direct argument) and compare the
    // argument tuple lexicographically against each row's key tuple; they have their own analysis
    // path below. The percentile / MODE ordered-set aggregates take exactly one ORDER BY key.
    if matches!(func, F::Rank | F::DenseRank | F::PercentRank | F::CumeDist) {
        return analyze_hypothetical_set(func, wg, scope, catalog, sink);
    }
    // WITHIN GROUP takes exactly one ORDER BY key. `DESC` reverses the ordered set; a `NULLS
    // FIRST/LAST` clause is accepted but has no effect (NULL ordering values are excluded from the
    // set, so their placement cannot change the percentile/mode).
    let [order_item] = wg.order_by.as_slice() else {
        return Err(Error::InvalidStatement(
            "WITHIN GROUP requires exactly one ORDER BY expression".to_owned(),
        ));
    };
    let ordered_set_descending = !order_item.ascending;
    // The ordered value references source rows, not aggregates (no nested aggregate sink).
    let order_value = analyze_expr(&order_item.expr, scope, catalog, None)?;
    let (fraction, result_ty) = match func {
        F::Mode => {
            if !wg.args.is_empty() {
                return Err(Error::FunctionArgs(
                    "MODE() takes no direct arguments".to_owned(),
                ));
            }
            (None, order_value.ty)
        },
        F::PercentileCont | F::PercentileDisc => {
            let [fraction_expr] = wg.args.as_slice() else {
                return Err(Error::FunctionArgs(
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
        hypothetical_args: Vec::new(),
        ordered_set_keys: Vec::new(),
        filter: None,
        separator: None,
        arg2: None,
        order_by: Vec::new(),
        row_args: Vec::new(),
        grouping_args: Vec::new(),
    });
    Ok(TypedExpr {
        kind: TypedExprKind::AggregateRef(idx),
        ty: result_ty,
    })
}

/// Analyze a hypothetical-set aggregate (`RANK` / `DENSE_RANK` / `PERCENT_RANK` / `CUME_DIST`) in
/// its multi-key `f(a1, …, aN) WITHIN GROUP (ORDER BY k1, …, kN)` form, registering it into the
/// aggregate `sink`. The direct arguments form a hypothetical tuple compared lexicographically
/// against each row's `ORDER BY` key tuple, every key honoring its own `ASC`/`DESC`.
///
/// The argument count must equal the key count (else a loud error). Each key expression resolves
/// against the source `scope`; each argument is a per-group constant, so it resolves against an
/// empty scope — a column reference is a loud error rather than a per-row value — and its type must
/// be comparable with the corresponding key under the same rule `=`/`<`/`>` use (cross-type numeric
/// included). `RANK` / `DENSE_RANK` yield a `BIGINT` position; `PERCENT_RANK` / `CUME_DIST` a
/// `FLOAT` ratio.
fn analyze_hypothetical_set(
    func: ast::AggregateFunc,
    wg: &ast::WithinGroup,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    sink: &mut Vec<AggregateCall>,
) -> Result<TypedExpr, Error> {
    use ast::AggregateFunc as F;
    if wg.order_by.is_empty() {
        return Err(Error::InvalidStatement(
            "WITHIN GROUP requires at least one ORDER BY expression".to_owned(),
        ));
    }
    if wg.args.len() != wg.order_by.len() {
        return Err(Error::FunctionArgs(
            "the number of hypothetical direct arguments must match the number of ordering columns"
                .to_owned(),
        ));
    }
    let mut ordered_set_keys = Vec::with_capacity(wg.order_by.len());
    let mut hypothetical_args = Vec::with_capacity(wg.args.len());
    for (order_item, arg_expr) in wg.order_by.iter().zip(&wg.args) {
        // The key references source rows, not aggregates (no nested aggregate sink). NULL keys are
        // kept and ordered by their NULLS placement (the default is NULLS LAST for ASC, NULLS FIRST
        // for DESC), matching the ordered-set NULL semantics.
        let key = analyze_expr(&order_item.expr, scope, catalog, None)?;
        let arg = analyze_expr(arg_expr, &[], catalog, None)?;
        check_comparison(key.ty, arg.ty)?;
        let descending = !order_item.ascending;
        let nulls_first = match order_item.nulls {
            ast::NullOrdering::First => true,
            ast::NullOrdering::Last => false,
            // Default: NULLS LAST for ASC, NULLS FIRST for DESC.
            ast::NullOrdering::Default => descending,
        };
        ordered_set_keys.push(OrderedSetKey {
            expr: key,
            descending,
            nulls_first,
        });
        hypothetical_args.push(arg);
    }
    let result_ty = if matches!(func, F::Rank | F::DenseRank) {
        ColumnType::Int
    } else {
        ColumnType::Float
    };
    let idx = sink.len();
    sink.push(AggregateCall {
        func,
        arg: None,
        result_ty,
        distinct: false,
        fraction: None,
        ordered_set_descending: false,
        hypothetical_args,
        ordered_set_keys,
        filter: None,
        separator: None,
        arg2: None,
        order_by: Vec::new(),
        row_args: Vec::new(),
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
            hypothetical_args: Vec::new(),
            ordered_set_keys: Vec::new(),
            filter: None,
            separator: None,
            arg2: None,
            order_by: Vec::new(),
            row_args: Vec::new(),
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
            return Err(Error::InvalidStatement(
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
    use ColumnType::{Date, Float, Interval, Text, Time, TimeTz, Timestamp, TimestampTz};
    use ast::ScalarFunc as F;
    let name = func.name();
    let is_temporal = |ty| matches!(ty, Date | Time | TimeTz | Timestamp | TimestampTz);
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
        F::Extract | F::DatePart | F::DateTrunc => {
            let (field_expr, source_expr) = expect_two_args(args, name)?;
            let field = expect_field_literal(field_expr, name)?;
            let valid_field = match func {
                F::Extract | F::DatePart => is_extract_field(&field),
                _ => is_trunc_field(&field),
            };
            if !valid_field {
                return Err(Error::Unsupported(format!(
                    "{name}() field `{field}` is not supported"
                )));
            }
            let source =
                analyze_expr_agg(source_expr, scope, catalog, None, aggregates.as_deref_mut())?;
            // DATE_TRUNC has no DATE form, so a DATE source widens to midnight and the call is
            // TIMESTAMPTZ-typed. The convention being followed: where a conversion is needed and
            // both the naive and the time-zone-aware form would serve, the time-zone-aware one
            // wins, being the preferred type of the date/time category. `DATE_TRUNC('month',
            // d::timestamp)` is how a caller asks for a naive result instead. Note this
            // deliberately differs from `DATE + INTERVAL`, which widens to the naive TIMESTAMP —
            // there no conversion is needed, so the preference never comes up.
            //
            // Widening to midnight *UTC* is only right while DATE_TRUNC of a TIMESTAMPTZ also
            // truncates in UTC: the two cancel. Should a session time zone ever drive conversion,
            // both halves have to become session-relative together, or a month bucket taken from
            // a DATE column reads as the previous month west of UTC.
            let source = if func == F::DateTrunc && source.ty == Date {
                TypedExpr {
                    kind: TypedExprKind::Cast(Box::new(source), false),
                    ty: TimestampTz,
                }
            } else {
                source
            };
            // EXTRACT accepts any temporal source; DATE_TRUNC truncates a timestamp.
            let ok_source = match func {
                // EXTRACT/DATE_PART also read the fields of an INTERVAL (e.g. `epoch`, `day`, `hour`).
                F::Extract | F::DatePart => is_temporal(source.ty) || source.ty == Interval,
                _ => matches!(source.ty, Timestamp | TimestampTz),
            };
            if !ok_source {
                return Err(Error::TypeMismatch {
                    context: format!("{name}() source"),
                    expected: Timestamp,
                    found: source.ty,
                });
            }
            // EXTRACT → exact NUMERIC; DATE_PART → double precision; DATE_TRUNC preserves the source's
            // temporal type.
            let result = match func {
                F::Extract => NUMERIC_ANY,
                F::DatePart => Float,
                _ => source.ty,
            };
            Ok(field_call(source, result, field))
        },
        F::AtTimeZone => {
            // `<value> AT TIME ZONE <zone>`: value is TIMESTAMP or TIMESTAMPTZ; the zone is a text
            // name/offset (`'UTC'`, `'+05:00'`) or an INTERVAL fixed offset (`INTERVAL '5 hours'`).
            // The result flips the time-zone-awareness — TIMESTAMP → TIMESTAMPTZ, TIMESTAMPTZ → TIMESTAMP.
            let [value, zone] = args else {
                return Err(Error::FunctionArgs(
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
                return Err(Error::FunctionArgs(format!(
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
                return Err(Error::FunctionArgs(format!("{name}() expects 1 argument")));
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
        // TO_CHAR formats a temporal value, an interval, or a number (B-fn).
        matches!(
            value.ty,
            Date | ColumnType::Time | Timestamp | ColumnType::TimestampTz | ColumnType::Interval
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
        // `to_timestamp(text, format)` yields a TIMESTAMPTZ (the parsed instant in the session zone,
        // fixed at UTC here), matching the reference engine — not a zoneless TIMESTAMP.
        _ => ColumnType::TimestampTz,
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
        _ => Err(Error::FunctionArgs(format!(
            "{name}() expects 2 argument(s), got {}",
            args.len()
        ))),
    }
}

/// Read a lowercase text-literal field name, erroring if the argument is not a string literal.
fn expect_field_literal(expr: &ast::Expr, name: &str) -> Result<String, Error> {
    match expr {
        ast::Expr::Literal(ast::Value::Text(s)) => Ok(s.to_lowercase()),
        _ => Err(Error::InvalidStatement(format!(
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
            | "decade"
            | "century"
            | "millennium"
            | "isoyear"
            | "microseconds"
            | "milliseconds"
            | "julian"
            | "timezone"
            | "timezone_hour"
            | "timezone_minute"
    )
}

/// Precisions supported by `DATE_TRUNC`.
fn is_trunc_field(field: &str) -> bool {
    matches!(
        field,
        "microsecond"
            | "microseconds"
            | "millisecond"
            | "milliseconds"
            | "second"
            | "minute"
            | "hour"
            | "day"
            | "week"
            | "month"
            | "quarter"
            | "year"
            | "decade"
            | "century"
            | "millennium"
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

/// The declared fields of the composite type an expression denotes, or `None` if the expression is
/// not a value of a statically known composite type. Only a cast to a composite type (`x::T`) and a
/// composite base-table column are recognised — a bare `f(...)` returning composite, a nested field,
/// or any other operand has no statically known composite type (out of first-cut scope), so it
/// returns `None` and the caller rejects it loudly rather than mis-typing it.
fn composite_type_of(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<Option<Vec<(String, ColumnType)>>, Error> {
    let type_name = match expr {
        ast::Expr::CastNamed { type_name, .. } => Some(type_name.clone()),
        ast::Expr::Column(name) => super::scoped_composite_type(scope, None, name),
        ast::Expr::QualifiedColumn { table, column } => {
            super::scoped_composite_type(scope, Some(table), column)
        },
        _ => None,
    };
    type_name.map_or(Ok(None), |name| catalog.lookup_composite(&name))
}

/// The user-defined enum type NAME of `expr` when it is statically enum-typed: an enum base-table
/// column, or a `x::enum` cast. `None` otherwise. Used by comparison to coerce a text literal to the
/// sibling's enum type and to reject comparing two different enum types.
fn enum_type_of(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<Option<String>, Error> {
    let name = match expr {
        ast::Expr::Column(name) => super::scoped_enum_type(scope, None, name),
        ast::Expr::QualifiedColumn { table, column } => {
            super::scoped_enum_type(scope, Some(table), column)
        },
        // A `::T` cast is enum-typed only when `T` is a declared enum type.
        ast::Expr::CastNamed { type_name, .. } if catalog.enum_labels(type_name)?.is_some() => {
            Some(type_name.clone())
        },
        // MIN/MAX return one of their input values unchanged, so they carry the argument's enum
        // type — `min(m) = min(p)` across two enum types must be caught like `m = p` is.
        ast::Expr::Aggregate {
            func: ast::AggregateFunc::Min | ast::AggregateFunc::Max,
            arg: Some(arg),
            ..
        } => enum_type_of(arg, scope, catalog)?,
        // CASE / COALESCE / GREATEST / LEAST / NULLIF return one of their value operands; their own
        // analysis rejects mixing two different enum types (42846), so the first enum-typed operand
        // is representative of the whole expression.
        ast::Expr::Case {
            branches, default, ..
        } => {
            let mut found = None;
            for branch in branches {
                if let Some(t) = enum_type_of(&branch.then, scope, catalog)? {
                    found = Some(t);
                    break;
                }
            }
            if found.is_none()
                && let Some(default) = default
            {
                found = enum_type_of(default, scope, catalog)?;
            }
            found
        },
        ast::Expr::Coalesce(items) => {
            let mut found = None;
            for item in items {
                if let Some(t) = enum_type_of(item, scope, catalog)? {
                    found = Some(t);
                    break;
                }
            }
            found
        },
        ast::Expr::ScalarFunction {
            func: ast::ScalarFunc::Greatest | ast::ScalarFunc::Least | ast::ScalarFunc::Nullif,
            args,
        } => {
            let mut found = None;
            for arg in args {
                if let Some(t) = enum_type_of(arg, scope, catalog)? {
                    found = Some(t);
                    break;
                }
            }
            found
        },
        _ => None,
    };
    Ok(name)
}

/// The CASE/COALESCE/GREATEST/LEAST/NULLIF mixed-enum guard: if two of `exprs` are statically typed
/// as *different* enum types, refuse with `42846` — the reference engine has no conversion between
/// two enum types, and letting them unify silently would compare unrelated ordinals.
fn reject_mixed_enum_operands<'a>(
    what: &str,
    exprs: impl Iterator<Item = &'a ast::Expr>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<(), Error> {
    let mut first: Option<String> = None;
    for expr in exprs {
        if let Some(t) = enum_type_of(expr, scope, catalog)? {
            match &first {
                None => first = Some(t),
                Some(f) if *f != t => {
                    return Err(Error::Coded {
                        message: format!("{what} could not convert type {t} to {f}"),
                        sqlstate: "42846",
                    });
                },
                Some(_) => {},
            }
        }
    }
    Ok(())
}

/// If `operand` is a bare text literal, resolve it to the value of enum type `enum_type` (an unknown
/// label is `22P02`); otherwise return it unchanged. Coerces the literal in `enum_col = 'label'`.
fn coerce_text_literal_to_enum(
    operand: TypedExpr,
    enum_type: &str,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    let TypedExprKind::Literal(ast::Value::Text(label)) = &operand.kind else {
        return Ok(operand);
    };
    let labels = catalog.enum_labels(enum_type)?.unwrap_or_default();
    labels.iter().position(|l| l == label).map_or_else(
        || {
            Err(Error::Coded {
                message: format!("invalid input value for enum {enum_type}: \"{label}\""),
                sqlstate: "22P02",
            })
        },
        |ordinal| {
            Ok(TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Enum {
                    ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
                    label: label.clone(),
                }),
                ty: ColumnType::Enum,
            })
        },
    )
}

/// The SQL spelling of a comparison operator, for the "operator does not exist" message when two
/// different enum types are compared.
const fn comparison_op_symbol(op: ast::BinaryOp) -> &'static str {
    match op {
        ast::BinaryOp::Eq => "=",
        ast::BinaryOp::NotEq => "<>",
        ast::BinaryOp::Lt => "<",
        ast::BinaryOp::LtEq => "<=",
        ast::BinaryOp::Gt => ">",
        _ => ">=",
    }
}

/// `(expr).field` — extract one field of a composite value. The operand's composite type is resolved
/// statically ([`composite_type_of`]); the field is looked up by name (a miss is a loud error), and
/// the executor parses the operand's canonical text form and returns that field.
/// `ROW(a, b, ...)` with no target type: analyze each field with no hint (its own type wins), allow
/// a bare `NULL` field (it formats as the empty canonical field regardless of type), and build the
/// same [`CompositeExpr::Construct`] the typed `ROW(...)::T` path uses — so evaluation, formatting,
/// and quoting are shared. The value's enclosing type is `Text` (the canonical `(f1,f2,…)` form).
fn analyze_row_constructor(
    items: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let mut fields = Vec::with_capacity(items.len());
    let mut field_types = Vec::with_capacity(items.len());
    for item in items {
        // A bare `NULL` field has no type context of its own; type it as text — the canonical form
        // writes a NULL field as empty either way.
        let field = if matches!(item, ast::Expr::Literal(ast::Value::Null)) {
            TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Null),
                ty: ColumnType::Text,
            }
        } else {
            analyze_expr_agg(item, scope, catalog, None, aggregates.as_deref_mut())?
        };
        field_types.push(field.ty);
        fields.push(field);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Composite(Box::new(CompositeExpr::Construct {
            fields,
            field_types,
        })),
        ty: ColumnType::Text,
    })
}

fn analyze_field_access(
    base: &ast::Expr,
    field: &str,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // An anonymous record's fields are named `f1..fN`, as the reference engine names them:
    // `(ROW(1,'a')).f1` reads the first field. Any other name — or an out-of-range ordinal — is
    // refused with the reference engine's own error (`42703`).
    if let ast::Expr::Row(items) = base {
        let base_typed = analyze_row_constructor(items, scope, catalog, aggregates)?;
        let TypedExprKind::Composite(construct) = &base_typed.kind else {
            return Err(Error::Internal(
                "a ROW(...) base did not analyze to a composite construct".to_owned(),
            ));
        };
        let CompositeExpr::Construct { field_types, .. } = construct.as_ref() else {
            return Err(Error::Internal(
                "a ROW(...) base did not analyze to a composite construct".to_owned(),
            ));
        };
        let field_types = field_types.clone();
        let index = field
            .strip_prefix('f')
            .and_then(|digits| digits.parse::<usize>().ok())
            .filter(|&n| n >= 1 && n <= field_types.len())
            .map(|n| n - 1)
            .ok_or_else(|| Error::Coded {
                message: format!("could not identify column \"{field}\" in record data type"),
                sqlstate: "42703",
            })?;
        let field_ty = field_types
            .get(index)
            .copied()
            .ok_or_else(|| Error::Internal("record field index out of range".to_owned()))?;
        return Ok(TypedExpr {
            kind: TypedExprKind::Composite(Box::new(CompositeExpr::Field {
                base: Box::new(base_typed),
                field_types,
                index,
            })),
            ty: super::expr_type(field_ty),
        });
    }
    let Some(fields) = composite_type_of(base, scope, catalog)? else {
        return Err(Error::InvalidStatement(format!(
            "field access `.{field}` requires an operand of a known composite type (a composite \
             column or a cast to a composite type)"
        )));
    };
    let index = fields
        .iter()
        .position(|(name, _)| name == field)
        .ok_or_else(|| {
            Error::InvalidStatement(format!("composite type has no field named {field:?}"))
        })?;
    let field_ty = fields
        .get(index)
        .map(|(_, ty)| *ty)
        .ok_or_else(|| Error::Internal("composite field index out of range".to_owned()))?;
    let field_types: Vec<ColumnType> = fields.iter().map(|(_, ty)| *ty).collect();
    // The operand evaluates to the canonical text form (a composite column is stored as `TEXT`; a
    // composite cast produces the text form), so analyze it normally with no hint.
    let base_typed = analyze_expr_agg(base, scope, catalog, None, aggregates)?;
    Ok(TypedExpr {
        kind: TypedExprKind::Composite(Box::new(CompositeExpr::Field {
            base: Box::new(base_typed),
            field_types,
            index,
        })),
        ty: super::expr_type(field_ty),
    })
}

/// `expr::T` where `T` is a user-defined type name. Only a composite type is in surface; an enum /
/// domain / unknown name is rejected loudly (they have no composite cast path). `ROW(...)::T` builds
/// the composite from its field expressions; any other operand is treated as a text value parsed
/// against `T`'s field types.
fn analyze_cast_named(
    expr: &ast::Expr,
    type_name: &str,
    _try_cast: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    // A cast to a user-defined enum type. A constant text literal resolves to its declaration-order
    // ordinal now (an unknown label is `22P02`); `NULL::enum` is a typed null. A non-constant operand
    // is not yet supported (a runtime text→enum cast would need the label set at evaluation time).
    if let Some(labels) = catalog.enum_labels(type_name)? {
        return match expr {
            ast::Expr::Literal(ast::Value::Null) => Ok(TypedExpr {
                kind: TypedExprKind::Literal(ast::Value::Null),
                ty: ColumnType::Enum,
            }),
            ast::Expr::Literal(ast::Value::Text(label)) => {
                labels.iter().position(|l| l == label).map_or_else(
                    || {
                        Err(Error::Coded {
                            message: format!(
                                "invalid input value for enum {type_name}: \"{label}\""
                            ),
                            sqlstate: "22P02",
                        })
                    },
                    |ordinal| {
                        Ok(TypedExpr {
                            kind: TypedExprKind::Literal(ast::Value::Enum {
                                ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
                                label: label.clone(),
                            }),
                            ty: ColumnType::Enum,
                        })
                    },
                )
            },
            _ => Err(Error::Unsupported(format!(
                "a cast to enum type \"{type_name}\" is supported only for a text literal"
            ))),
        };
    }
    let Some(fields) = catalog.lookup_composite(type_name)? else {
        return Err(Error::ObjectNotFound(format!(
            "type \"{type_name}\" does not exist or is not a composite type"
        )));
    };
    let field_types: Vec<ColumnType> = fields.iter().map(|(_, ty)| *ty).collect();
    if let ast::Expr::Row(items) = expr {
        if items.len() != field_types.len() {
            return Err(Error::InvalidStatement(format!(
                "composite type \"{type_name}\" has {} fields but the ROW value supplies {}",
                field_types.len(),
                items.len()
            )));
        }
        let mut typed = Vec::with_capacity(items.len());
        for (item, want) in items.iter().zip(&field_types) {
            let field =
                analyze_expr_agg(item, scope, catalog, Some(*want), aggregates.as_deref_mut())?;
            // A `NULL` field is always allowed; otherwise it must be assignable to the declared type
            // (the executor coerces it to that type before formatting the canonical text form).
            if !is_null_literal(&field) && !assignable(*want, field.ty) {
                return Err(Error::TypeMismatch {
                    context: format!("field of composite type \"{type_name}\""),
                    expected: *want,
                    found: field.ty,
                });
            }
            typed.push(field);
        }
        return Ok(TypedExpr {
            kind: TypedExprKind::Composite(Box::new(CompositeExpr::Construct {
                fields: typed,
                field_types,
            })),
            ty: ColumnType::Text,
        });
    }
    // A non-ROW operand is a text value spelling the composite's canonical form (e.g. `'(a,b)'::T`).
    let inner = analyze_expr_agg(expr, scope, catalog, Some(ColumnType::Text), aggregates)?;
    if inner.ty.physical() != ColumnType::Text {
        return Err(Error::InvalidStatement(format!(
            "a cast to composite type \"{type_name}\" requires a ROW(...) value or a text value"
        )));
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Composite(Box::new(CompositeExpr::Cast {
            expr: Box::new(inner),
            field_types,
        })),
        ty: ColumnType::Text,
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
    // Two different enum types among the result branches have no common type — refuse before
    // unifying (42846), or the branches would silently compare/return unrelated ordinals.
    reject_mixed_enum_operands(
        "CASE/WHEN",
        branches.iter().map(|b| &b.then).chain(default),
        scope,
        catalog,
    )?;
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
    // A row-constructor probe `(a, b) IN ((1, 1), (2, 1))` desugars to a chain of row comparisons,
    // reusing the same field-wise 3-valued logic as `(a, b) = (1, 1)`. `IN` is an `OR` of row
    // equalities; `NOT IN` is (by De Morgan) an `AND` of row inequalities — both correct under NULLs.
    // Every list item must itself be a row of the same width, or it is a loud arity error.
    if let ast::Expr::Row(l) = expr {
        let item_op = if negated {
            ast::BinaryOp::NotEq
        } else {
            ast::BinaryOp::Eq
        };
        let mut comparisons = Vec::with_capacity(list.len());
        for item in list {
            let ast::Expr::Row(r) = item else {
                return Err(Error::InvalidStatement(
                    "a row on the left of IN requires every list item to be a row of the same width"
                        .to_owned(),
                ));
            };
            comparisons.push(desugar_row_comparison(l, item_op, r)?);
        }
        let combine = if negated {
            ast::BinaryOp::And
        } else {
            ast::BinaryOp::Or
        };
        let folded = comparisons.into_iter().reduce(|acc, e| ast::Expr::Binary {
            left: Box::new(acc),
            op: combine,
            right: Box::new(e),
        });
        // An empty list makes `IN` unconditionally false and `NOT IN` unconditionally true.
        let desugared = folded.unwrap_or(ast::Expr::Literal(ast::Value::Bool(negated)));
        return analyze_expr_agg(
            &desugared,
            scope,
            catalog,
            Some(ColumnType::Bool),
            aggregates,
        );
    }
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
    // When the probe is an enum, a bare text list item adopts that enum type (like `= 'label'`), so
    // `m IN ('low', 'high')` needs no per-item cast; an unknown label is a loud error.
    let probe_enum = enum_type_of(expr, scope, catalog)?;
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
        let item_typed = match probe_enum.as_deref() {
            Some(enum_type) => coerce_text_literal_to_enum(item_typed, enum_type, catalog)?,
            None => item_typed,
        };
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

#[allow(
    clippy::too_many_arguments,
    reason = "mirrors analyze_expr_agg's threaded context"
)]
pub(super) fn analyze_between(
    expr: &ast::Expr,
    low: &ast::Expr,
    high: &ast::Expr,
    negated: bool,
    symmetric: bool,
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
            symmetric,
        },
        ty: ColumnType::Bool,
    })
}

/// Analyze `(s1, e1) OVERLAPS (s2, e2)`. Both starts must be temporal
/// (`Date`/`Time`/`Timestamp`/`TimestampTz`); each end must be the same temporal type as its start
/// or an `INTERVAL` (the interval form, whose real end is `start + interval`). The result is a
/// nullable `Bool`.
pub(super) fn analyze_overlaps(
    s1: &ast::Expr,
    e1: &ast::Expr,
    s2: &ast::Expr,
    e2: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<TypedExpr, Error> {
    let is_temporal = |ty: ColumnType| {
        matches!(
            ty,
            ColumnType::Date | ColumnType::Time | ColumnType::Timestamp | ColumnType::TimestampTz
        )
    };
    let is_null_lit = |e: &ast::Expr| matches!(e, ast::Expr::Literal(ast::Value::Null));
    let raw = [s1, e1, s2, e2];
    // Pass 1: analyze every non-NULL endpoint and discover the shared temporal type (the anchor)
    // that types any bare NULL endpoint. An end may be an INTERVAL, which is not a valid anchor.
    let mut typed: [Option<TypedExpr>; 4] = [None, None, None, None];
    let mut anchor: Option<ColumnType> = None;
    for (slot, e) in typed.iter_mut().zip(raw) {
        if is_null_lit(e) {
            continue;
        }
        let t = analyze_expr_agg(e, scope, catalog, anchor, aggregates.as_deref_mut())?;
        if anchor.is_none() && is_temporal(t.ty) {
            anchor = Some(t.ty);
        }
        *slot = Some(t);
    }
    let Some(anchor) = anchor else {
        return Err(Error::InvalidStatement(
            "OVERLAPS requires at least one temporal endpoint to determine the period type"
                .to_owned(),
        ));
    };
    // Pass 2: type the deferred bare-NULL endpoints against the anchor.
    for (slot, e) in typed.iter_mut().zip(raw) {
        if is_null_lit(e) {
            *slot = Some(analyze_null(Some(anchor))?);
        }
    }
    // Every slot is now populated; recover the four operands without panicking.
    let assembled: Vec<TypedExpr> = typed.into_iter().flatten().collect();
    let Ok([s1_typed, e1_typed, s2_typed, e2_typed]) = <[TypedExpr; 4]>::try_from(assembled) else {
        return Err(Error::InvalidStatement(
            "OVERLAPS requires two 2-element row expressions".to_owned(),
        ));
    };
    // A bare temporal string literal adopts the anchor type (mirrors BETWEEN/IN).
    let s1_typed = coerce_unknown_literal(s1_typed, anchor);
    let e1_typed = coerce_unknown_literal(e1_typed, anchor);
    let s2_typed = coerce_unknown_literal(s2_typed, anchor);
    let e2_typed = coerce_unknown_literal(e2_typed, anchor);
    for (start, label) in [(&s1_typed, "first"), (&s2_typed, "second")] {
        if !is_temporal(start.ty) {
            return Err(Error::TypeMismatch {
                context: format!("OVERLAPS {label} period start"),
                expected: ColumnType::Timestamp,
                found: start.ty,
            });
        }
    }
    // Each end must be the anchor temporal type or an INTERVAL (interval form).
    for (end, label) in [(&e1_typed, "first"), (&e2_typed, "second")] {
        if end.ty != ColumnType::Interval && !comparable(anchor, end.ty) {
            return Err(Error::TypeMismatch {
                context: format!("OVERLAPS {label} period end"),
                expected: anchor,
                found: end.ty,
            });
        }
    }
    Ok(TypedExpr {
        kind: TypedExprKind::Overlaps {
            s1: Box::new(s1_typed),
            e1: Box::new(e1_typed),
            s2: Box::new(s2_typed),
            e2: Box::new(e2_typed),
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
/// comparison against a non-text operand — `WHERE id = '1'`, `WHERE d >= '2026-01-01'`, or the
/// identical query with a bound `$1` (a driver sends the parameter as text over the extended
/// protocol) — would raise a spurious `TypeMismatch`. When `operand` is a bare `TEXT` literal and
/// `anchor` is a type that accepts a text value, re-type it as a cast to `anchor`, producing the
/// exact same typed expression an explicit `'…'::type` would: the executor parses the text at
/// evaluation, and an unparseable string still loud-rejects (never a silent wrong row) — so
/// `id = 'abc'` fails with the `invalid_text_representation` error, not a wrong match.
///
/// Only a `TEXT` *literal* is coerced — a genuinely `TEXT`-typed column or expression versus a
/// non-text operand stays a real type error, matching the reference engine (only string literals are
/// "unknown"). A non-coercible `anchor` (or a non-literal operand) is returned unchanged. Shared with
/// the `INSERT`/`VALUES` path, where the `anchor` is the target column's type.
pub(super) fn coerce_unknown_literal(operand: TypedExpr, anchor: ColumnType) -> TypedExpr {
    if !matches!(&operand.kind, TypedExprKind::Literal(ast::Value::Text(_))) {
        return operand;
    }
    // A temporal / `UUID` / `VECTOR` anchor keeps its exact type (`'[1,0,0]'` is the same literal form
    // an `INSERT` accepts for a `VECTOR` column). A numeric or boolean anchor coerces to its physical
    // type, which always has a text-parsing cast — so `int` / `smallint` / `numeric` / `real` / `bool`
    // all accept an unknown literal, matching how the reference engine resolves it.
    let target = if is_temporal_or_uuid(anchor) || matches!(anchor, ColumnType::Vector(_)) {
        Some(anchor)
    } else if matches!(
        anchor.physical(),
        ColumnType::Int | ColumnType::Float | ColumnType::Bool | ColumnType::Numeric { .. }
    ) {
        Some(anchor.physical())
    } else {
        None
    };
    match target {
        Some(ty) => TypedExpr {
            kind: TypedExprKind::Cast(Box::new(operand), false),
            ty,
        },
        None => operand,
    }
}

/// For `INSERT`/`VALUES`: coerce a bare string literal into an integer / float / boolean / numeric
/// column.
///
/// Those types accept a text value only as an unknown literal, and for them a cast is exactly the
/// assignment — they carry no length or padding, unlike a `BIT(n)` / `CHAR(n)` cast — so
/// `INSERT INTO t(int_col) VALUES ('123')` stores `123` and `VALUES ('xyz')` loud-rejects at
/// evaluation. NUMERIC is included so a `'NaN'` (or any string) literal lands as a real `Numeric`
/// in the row, not a stray `Text`: the tuple codec parses it either way, but a text-typed row value
/// would encode a *text* index key that never matches the `Numeric` key a query builds. Every other
/// column type keeps its own assignment path: a `BIT`/temporal/range/… column already accepts a text
/// literal through [`assignable`] with its own length/parse rules, which a cast here would wrongly
/// relax. A non-literal operand or any other target is unchanged.
pub(super) fn coerce_insert_literal(typed: TypedExpr, column: ColumnType) -> TypedExpr {
    if matches!(&typed.kind, TypedExprKind::Literal(ast::Value::Text(_)))
        && matches!(
            column.physical(),
            ColumnType::Int | ColumnType::Float | ColumnType::Bool | ColumnType::Numeric { .. }
        )
    {
        TypedExpr {
            kind: TypedExprKind::Cast(Box::new(typed), false),
            ty: column.physical(),
        }
    } else {
        typed
    }
}

/// Coerce a bare `TEXT` literal argument to a function's declared parameter type when the reference
/// engine's implicit unknown-literal rule allows it (`assignable(expected, TEXT)`: `JSON`, arrays,
/// temporal/UUID, interval, vector, bytea, numeric). A string literal is "unknown"-typed, so
/// `jsonb_object_keys('{...}')` / `jsonb_set('{...}', '{a}', '9')` must type-check like the explicit
/// `'...'::json` / `'{a}'::text[]` forms rather than raising `TypeMismatch: expected Json, found
/// Text`. The re-typed cast parses the literal at evaluation and a bad value still loud-rejects.
///
/// Only a bare `TEXT` *literal* is coerced — a genuinely `TEXT`-typed column or expression stays a
/// real mismatch (the reference engine treats only string literals as unknown). A `TEXT`-expecting
/// parameter, or a type that does not accept a text value, returns `typed` unchanged.
pub(super) fn coerce_text_literal_to(typed: TypedExpr, expected: ColumnType) -> TypedExpr {
    let want = expected.physical();
    if want != ColumnType::Text
        && matches!(&typed.kind, TypedExprKind::Literal(ast::Value::Text(_)))
        && assignable(want, ColumnType::Text)
    {
        TypedExpr {
            kind: TypedExprKind::Cast(Box::new(typed), false),
            ty: super::expr_type(expected),
        }
    } else {
        typed
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
        return Err(Error::InvalidStatement(
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
            return Err(Error::InvalidStatement(
                "a row (…) is only valid with a comparison operator".to_owned(),
            ));
        },
    };
    combined.ok_or_else(|| {
        Error::InvalidStatement("a row comparison requires a non-empty row".to_owned())
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "flat operator-coercion pass; length tracks the operand-coercion special cases"
)]
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
    // A comparison between two composite operands of the same type orders them field-by-field via
    // `CompositeVal::compare` (two-valued, NULL-last) — distinct from the three-valued `ROW(…)`
    // comparison above. A composite operand paired with a non-composite one is rejected loudly
    // rather than silently compared as text.
    if matches!(
        op,
        ast::BinaryOp::Eq
            | ast::BinaryOp::NotEq
            | ast::BinaryOp::Lt
            | ast::BinaryOp::LtEq
            | ast::BinaryOp::Gt
            | ast::BinaryOp::GtEq
    ) {
        let left_fields = composite_type_of(left, scope, catalog)?;
        let right_fields = composite_type_of(right, scope, catalog)?;
        match (left_fields, right_fields) {
            (Some(lf), Some(rf)) => {
                let lt: Vec<ColumnType> = lf.iter().map(|(_, ty)| *ty).collect();
                let rt: Vec<ColumnType> = rf.iter().map(|(_, ty)| *ty).collect();
                if lt != rt {
                    return Err(Error::InvalidStatement(
                        "cannot compare values of different composite types".to_owned(),
                    ));
                }
                let left_typed =
                    analyze_expr_agg(left, scope, catalog, None, aggregates.as_deref_mut())?;
                let right_typed = analyze_expr_agg(right, scope, catalog, None, aggregates)?;
                return Ok(TypedExpr {
                    kind: TypedExprKind::Composite(Box::new(CompositeExpr::Compare {
                        left: Box::new(left_typed),
                        right: Box::new(right_typed),
                        op,
                        field_types: lt,
                    })),
                    ty: ColumnType::Bool,
                });
            },
            (Some(_), None) | (None, Some(_)) => {
                return Err(Error::InvalidStatement(
                    "cannot compare a composite value to a non-composite value".to_owned(),
                ));
            },
            (None, None) => {},
        }
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
    // type (numeric, boolean, temporal, …), so `WHERE id = '1'` or a parameterized date filter
    // (`WHERE d >= $1`, bound as text) type-checks exactly like the explicit `'1'::int` / `$1::date`
    // form. A no-op for every non-comparison operator and for operands that are not a bare `TEXT`
    // literal.
    if matches!(
        op,
        ast::BinaryOp::Eq
            | ast::BinaryOp::NotEq
            | ast::BinaryOp::Lt
            | ast::BinaryOp::LtEq
            | ast::BinaryOp::Gt
            | ast::BinaryOp::GtEq
    ) {
        // Enum: a bare text literal beside an enum operand (column or `::enum` cast) adopts that enum
        // type, so `m = 'high'` resolves `'high'` to its ordinal without an explicit cast. Comparing
        // two DIFFERENT enum types has no operator (`42883`), matching the reference engine.
        let left_enum = enum_type_of(left, scope, catalog)?;
        let right_enum = enum_type_of(right, scope, catalog)?;
        if let (Some(le), Some(re)) = (&left_enum, &right_enum)
            && le != re
        {
            return Err(Error::Coded {
                message: format!(
                    "operator does not exist: {le} {} {re}",
                    comparison_op_symbol(op)
                ),
                sqlstate: "42883",
            });
        }
        if let Some(enum_type) = left_enum.as_deref().or(right_enum.as_deref()) {
            left_typed = coerce_text_literal_to_enum(left_typed, enum_type, catalog)?;
            right_typed = coerce_text_literal_to_enum(right_typed, enum_type, catalog)?;
        }
        right_typed = coerce_unknown_literal(right_typed, left_typed.ty);
        left_typed = coerce_unknown_literal(left_typed, right_typed.ty);
        // A DATE compared to a TIMESTAMP / TIMESTAMPTZ widens to that type (the date read at
        // midnight), so `d = TIMESTAMP '…'` type-checks instead of raising a mismatch — the same
        // implicit widening `DATE + INTERVAL` already relies on.
        if left_typed.ty == ColumnType::Date
            && matches!(
                right_typed.ty,
                ColumnType::Timestamp | ColumnType::TimestampTz
            )
        {
            left_typed = TypedExpr {
                kind: TypedExprKind::Cast(Box::new(left_typed), false),
                ty: right_typed.ty,
            };
        } else if right_typed.ty == ColumnType::Date
            && matches!(
                left_typed.ty,
                ColumnType::Timestamp | ColumnType::TimestampTz
            )
        {
            right_typed = TypedExpr {
                kind: TypedExprKind::Cast(Box::new(right_typed), false),
                ty: left_typed.ty,
            };
        }
    }
    // The vector distance operators have only a `VECTOR <op> VECTOR` form, so a bare string literal
    // next to a `VECTOR` operand is unambiguous: coerce it to that operand's vector type, exactly
    // as `INSERT` coerces the same `'[…]'` literal into a `VECTOR` column. Without this the operator
    // demands an explicit `::VECTOR(n)` the column-insert never asks for.
    if matches!(
        op,
        ast::BinaryOp::VectorDistance
            | ast::BinaryOp::VectorL2Distance
            | ast::BinaryOp::VectorNegInnerProduct
            | ast::BinaryOp::VectorL1Distance
    ) {
        if matches!(left_typed.ty, ColumnType::Vector(_)) {
            right_typed = coerce_unknown_literal(right_typed, left_typed.ty);
        }
        if matches!(right_typed.ty, ColumnType::Vector(_)) {
            left_typed = coerce_unknown_literal(left_typed, right_typed.ty);
        }
    }
    // `&&` (overlap) and the boundary operators `-|-` / `&<` / `&>` over ranges have only a
    // range/range form, so a bare string literal next to a range operand is unambiguous and coerces
    // to that range's kind. (`@>`/`<@` do not: a literal there could be the range or one element, so
    // it stays a mismatch until cast.)
    if matches!(
        op,
        ast::BinaryOp::ArrayOverlap
            | ast::BinaryOp::RangeAdjacent
            | ast::BinaryOp::RangeNotExtendRight
            | ast::BinaryOp::RangeNotExtendLeft
    ) {
        if is_range_type(left_typed.ty) {
            right_typed = coerce_unknown_literal(right_typed, left_typed.ty);
        }
        if is_range_type(right_typed.ty) {
            left_typed = coerce_unknown_literal(left_typed, right_typed.ty);
        }
    }
    // The JSON key operators take a key, not a second document, so a bare `NULL` on their right
    // must not be typed from the JSON document on their left ([`analyze_operands`] types it from the
    // sibling by default). Re-type it as the key type the operator actually wants, so `j - NULL` and
    // `j ? NULL` evaluate to NULL instead of failing to type-check.
    if let Some(key_ty) = json_key_operand_type(op, left_typed.ty)
        && is_null_literal(&right_typed)
    {
        right_typed.ty = key_ty;
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

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-operator dispatch; splitting it would scatter the operator table"
)]
pub(super) fn check_binary(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use ast::BinaryOp as Op;
    use nusadb_core::engine::GeomKind;
    match op {
        Op::Eq | Op::NotEq | Op::Lt | Op::LtEq | Op::Gt | Op::GtEq => check_comparison(left, right),
        Op::And | Op::Or => check_logical(left, right),
        // JSON `-` deletes a key / an array index / a set of keys, yielding the trimmed document.
        // Checked before the numeric rule (JSON is not numeric).
        Op::Minus if left == ColumnType::Json => check_json_delete(right),
        // Geometric point arithmetic: `p1 + p2` / `p1 - p2` (vector add/sub) and `p1 * p2` / `p1 / p2`
        // (complex multiply/divide) each yield a point. Checked before the numeric rule (a point is
        // not numeric).
        Op::Plus | Op::Minus | Op::Multiply | Op::Divide
            if left == ColumnType::Geometry(GeomKind::Point)
                && right == ColumnType::Geometry(GeomKind::Point) =>
        {
            Ok(ColumnType::Geometry(GeomKind::Point))
        },
        // Geometric path concatenation: `path + path` appends the second path's vertices to the
        // first, yielding a `path` (NULL at evaluation when either operand is a closed path). Checked
        // before the numeric rule (a path is not numeric).
        Op::Plus
            if left == ColumnType::Geometry(GeomKind::Path)
                && right == ColumnType::Geometry(GeomKind::Path) =>
        {
            Ok(ColumnType::Geometry(GeomKind::Path))
        },
        // INET arithmetic: `inet ± int` / `int + inet` (offset the address, → INET) and
        // `inet - inet` (address difference, → BIGINT). Checked before the numeric rule (a network
        // address is not numeric); the helper rejects the undefined forms (`int - inet`,
        // `inet + inet`, `inet ± numeric`) loudly.
        Op::Plus | Op::Minus if is_inet_type(left) || is_inet_type(right) => {
            check_inet_arithmetic(op, left, right)
        },
        // Range set operators: `range + range` (union), `range * range` (intersection), and
        // `range - range` (difference) over two ranges of the same element kind, each yielding that
        // range type. Checked before the numeric rule (a range is not a number).
        Op::Plus | Op::Multiply | Op::Minus if is_range_type(left) || is_range_type(right) => {
            check_range_setop(op, left, right)
        },
        Op::Plus | Op::Multiply | Op::Divide | Op::Modulo | Op::Minus => {
            // Element-wise vector arithmetic (`+`/`-`/`*` over two same-dimension vectors) is checked
            // before the numeric rule (a vector operand is not numeric).
            if let (ColumnType::Vector(x), ColumnType::Vector(y)) = (left, right) {
                return check_vector_arithmetic(op, x, y);
            }
            // INTERVAL / temporal arithmetic takes priority over numeric.
            check_interval_arith(op, left, right).map_or_else(|| check_arithmetic(left, right), Ok)
        },
        // INET/CIDR reuse `<<` / `>>` (contained-by / contains) and `&&` (overlaps) as network
        // predicates, yielding `BOOL`, when both operands are network addresses.
        Op::ShiftLeft | Op::ShiftRight | Op::ArrayOverlap
            if is_inet_type(left) && is_inet_type(right) =>
        {
            Ok(ColumnType::Bool)
        },
        // Ranges reuse `<<` / `>>` as strict-order predicates (`range << range` strictly-left-of,
        // `range >> range` strictly-right-of), yielding `BOOL`. Checked before the bit/integer shift
        // rules below (a range is neither a bit string nor an integer).
        Op::ShiftLeft | Op::ShiftRight if is_range_type(left) || is_range_type(right) => {
            check_range_strict(op, left, right)
        },
        // MACADDR8: `&`/`|` combine two eight-byte addresses byte-wise, yielding a MACADDR8.
        Op::BitAnd | Op::BitOr if left == ColumnType::Macaddr8 && right == ColumnType::Macaddr8 => {
            Ok(ColumnType::Macaddr8)
        },
        Op::BitAnd | Op::BitOr if left == ColumnType::Macaddr && right == ColumnType::Macaddr => {
            Ok(ColumnType::Macaddr)
        },
        // Geometry: `#` (BitXor) is the `lseg # lseg` intersection, yielding a (nullable) `point`.
        // Checked before the bit/integer XOR below (a geometry operand is neither a bit string nor an
        // integer).
        Op::BitXor if is_geometry_type(left) || is_geometry_type(right) => {
            check_geom_intersection(left, right)
        },
        // BIT strings: `&`/`|`/`#` combine two (equal-length) bit strings, `<<`/`>>` shift by an
        // `INT`, all yielding a bit string; `||` concatenates into a variable-length result.
        Op::BitAnd | Op::BitOr | Op::BitXor if is_bit_type(left) && is_bit_type(right) => Ok(left),
        Op::ShiftLeft | Op::ShiftRight if is_bit_type(left) && matches!(right, ColumnType::Int) => {
            Ok(left)
        },
        // Full-text `tsvector || tsvector` concatenates into a `tsvector`. Checked before the
        // bit/array/string `||` arms below.
        Op::Concat if left == ColumnType::Tsvector && right == ColumnType::Tsvector => {
            Ok(ColumnType::Tsvector)
        },
        // Full-text `tsquery || tsquery` (OR) and `tsquery && tsquery` (AND) both yield a `tsquery`
        // (the executor distinguishes the two). Checked before the bit/geometry/range/array arms.
        Op::Concat | Op::ArrayOverlap
            if left == ColumnType::Tsquery && right == ColumnType::Tsquery =>
        {
            Ok(ColumnType::Tsquery)
        },
        Op::Concat if is_bit_type(left) && is_bit_type(right) => Ok(ColumnType::VarBit(None)),
        Op::BitAnd | Op::BitOr | Op::BitXor | Op::ShiftLeft | Op::ShiftRight => {
            check_bitwise(op, left, right)
        },
        // Geometry: `box && box` overlap and `box @> point` / `point <@ box` containment, each `BOOL`.
        // Checked before the range/array rules (a geometry operand is neither).
        Op::ArrayOverlap if is_geometry_type(left) || is_geometry_type(right) => {
            check_geom_overlap(left, right)
        },
        Op::JsonContains | Op::JsonContainedBy
            if is_geometry_type(left) || is_geometry_type(right) =>
        {
            check_geom_containment(op, left, right)
        },
        // Ranges: `&&` overlap and `@>`/`<@` containment, of a range or of a single element.
        Op::ArrayOverlap if is_range_type(left) || is_range_type(right) => {
            check_range_overlap(left, right)
        },
        Op::JsonContains | Op::JsonContainedBy if is_range_type(left) || is_range_type(right) => {
            check_range_containment(op, left, right)
        },
        Op::ArrayOverlap => check_array_overlap(left, right),
        Op::Concat => check_concat(left, right),
        // `@>` / `<@` are containment over JSON *and* arrays, so they get their own checker.
        Op::JsonContains | Op::JsonContainedBy => check_containment(op, left, right),
        Op::JsonGet | Op::JsonGetText | Op::JsonGetPath | Op::JsonGetPathText => {
            check_json(op, left, right)
        },
        // JSON `#-` deletes the element at a `text[]` path, yielding the trimmed document.
        Op::JsonDeletePath => check_json_delete_path(left, right),
        Op::JsonExists | Op::JsonExistsAny | Op::JsonExistsAll => {
            check_json_exists(op, left, right)
        },
        Op::VectorDistance => check_vector_distance("<=>", left, right),
        // `<->` is the geometric distance (`point <-> point`, `box <-> box`) when both operands are
        // geometry, else the vector L2 distance.
        Op::VectorL2Distance if is_geometry_type(left) || is_geometry_type(right) => {
            check_geom_distance(left, right)
        },
        Op::VectorL2Distance => check_vector_distance("<->", left, right),
        Op::VectorNegInnerProduct => check_vector_distance("<#>", left, right),
        Op::VectorL1Distance => check_vector_distance("<+>", left, right),
        Op::TsMatch => check_ts_match(left, right),
        // `@?` — a JSON document on the left, a jsonpath text on the right, yielding `BOOL`.
        Op::JsonPathExists => check_json_path_exists(left, right),
        // `~=` geometric same-as — both operands the same geometric kind, yielding `BOOL`.
        Op::GeomSameAs => check_geom_same_as(left, right),
        // `?||` / `?-|` / `?#` geometric predicates — `lseg`↔`lseg` or `line`↔`line`, yielding `BOOL`.
        Op::GeomParallel | Op::GeomPerpendicular | Op::GeomIntersects => {
            check_geom_predicate(op, left, right)
        },
        // INET/CIDR subnet-or-equal `<<=` / supernet-or-equal `>>=` — both operands are network
        // addresses, yielding `BOOL`.
        Op::InetSubnetEq | Op::InetSupernetEq => check_inet_subnet(op, left, right),
        // Range boundary predicates `-|-` (adjacent), `&<` (does not extend right), `&>` (does not
        // extend left) — both operands ranges of the same element kind, yielding `BOOL`.
        Op::RangeAdjacent | Op::RangeNotExtendRight | Op::RangeNotExtendLeft => {
            check_range_bound_predicate(op, left, right)
        },
    }
}

/// Type rule for INET arithmetic: `inet + int` / `int + inet` / `inet - int` yield an `INET` (the
/// address offset by the integer), and `inet - inet` yields a `BIGINT` (the signed address
/// difference). Every other operand combination involving a network address (`int - inet`,
/// `inet + inet`, `inet ± numeric`) is undefined and rejected.
fn check_inet_arithmetic(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use ast::BinaryOp as Op;
    const fn is_int(ty: ColumnType) -> bool {
        matches!(
            ty,
            ColumnType::SmallInt | ColumnType::Int | ColumnType::BigInt
        )
    }
    // `inet + int` and the commutative `int + inet`, plus `inet - int`, offset the address → INET.
    let addr_offset = (matches!(op, Op::Plus)
        && ((is_inet_type(left) && is_int(right)) || (is_int(left) && is_inet_type(right))))
        || (matches!(op, Op::Minus) && is_inet_type(left) && is_int(right));
    // `inet - inet` is the signed address difference → BIGINT.
    let addr_diff = matches!(op, Op::Minus) && is_inet_type(left) && is_inet_type(right);
    if addr_offset {
        Ok(ColumnType::Inet)
    } else if addr_diff {
        Ok(ColumnType::BigInt)
    } else {
        Err(Error::TypeMismatch {
            context: "network address arithmetic (`inet ± int`, `inet - inet`)".to_owned(),
            expected: left,
            found: right,
        })
    }
}

/// Type rule for the INET subnet-or-equal `<<=` / supernet-or-equal `>>=` predicates: both operands
/// must be network addresses (`INET` or `CIDR`); the result is `BOOL`.
fn check_inet_subnet(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    if is_inet_type(left) && is_inet_type(right) {
        Ok(ColumnType::Bool)
    } else {
        let symbol = if matches!(op, ast::BinaryOp::InetSubnetEq) {
            "<<="
        } else {
            ">>="
        };
        Err(Error::TypeMismatch {
            context: format!("`{symbol}` network subnet predicate"),
            expected: left,
            found: right,
        })
    }
}

/// Whether a column type is a geometric type.
const fn is_geometry_type(ty: ColumnType) -> bool {
    matches!(ty, ColumnType::Geometry(_))
}

/// Type rule for the geometric distance `<->`: `point <-> point`, `box <-> box`, `circle <-> circle`,
/// `lseg <-> lseg`, `line <-> line` (same-kind), and the mixed `circle <-> point` / `lseg <-> point`
/// / `line <-> point` (either order); each yields the `FLOAT` distance.
fn check_geom_distance(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    use nusadb_core::engine::GeomKind::{Circle, Line, Lseg, Point};
    match (left, right) {
        (ColumnType::Geometry(a), ColumnType::Geometry(b))
            if a == b
                || matches!(
                    (a, b),
                    (Circle | Lseg | Line, Point) | (Point, Circle | Lseg | Line)
                ) =>
        {
            Ok(ColumnType::Float)
        },
        _ => Err(Error::TypeMismatch {
            context: "`<->` geometric distance".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for `#` geometric intersection: `lseg # lseg` and `line # line`, yielding the (nullable)
/// `point` where the two operands cross — `NULL` at evaluation when they do not cross at a single
/// point (collinear / non-intersecting segments, or parallel lines).
fn check_geom_intersection(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    use nusadb_core::engine::GeomKind::{Line, Lseg, Point};
    match (left, right) {
        (ColumnType::Geometry(Lseg), ColumnType::Geometry(Lseg))
        | (ColumnType::Geometry(Line), ColumnType::Geometry(Line)) => {
            Ok(ColumnType::Geometry(Point))
        },
        _ => Err(Error::TypeMismatch {
            context: "`#` geometric intersection".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for the geometric predicates `?||` (parallel), `?-|` (perpendicular) and `?#`
/// (intersects): `lseg ? lseg` and `line ? line` (same kind, only `lseg`/`line`); each yields the
/// `BOOL` predicate. Mirrors [`check_geom_intersection`], which shares the `lseg`/`line` domain.
fn check_geom_predicate(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use nusadb_core::engine::GeomKind::{Line, Lseg};
    let context = match op {
        ast::BinaryOp::GeomParallel => "`?||` geometric parallel",
        ast::BinaryOp::GeomPerpendicular => "`?-|` geometric perpendicular",
        _ => "`?#` geometric intersects",
    };
    match (left, right) {
        (ColumnType::Geometry(Lseg), ColumnType::Geometry(Lseg))
        | (ColumnType::Geometry(Line), ColumnType::Geometry(Line)) => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: context.to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for `~=`: both operands are the same geometric kind; the result is `BOOL`.
fn check_geom_same_as(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    match (left, right) {
        (ColumnType::Geometry(a), ColumnType::Geometry(b)) if a == b => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: "`~=` geometric same-as".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for geometric overlap `&&`: `box && box` or `circle && circle`; the result is `BOOL`.
fn check_geom_overlap(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    use nusadb_core::engine::GeomKind::{Box, Circle, Polygon};
    match (left, right) {
        (ColumnType::Geometry(Box), ColumnType::Geometry(Box))
        | (ColumnType::Geometry(Circle), ColumnType::Geometry(Circle))
        | (ColumnType::Geometry(Polygon), ColumnType::Geometry(Polygon)) => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: "`&&` geometric overlap".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for geometric containment: `@>` (contains) and `<@` (contained by), for `box @> point`,
/// `circle @> point`, `circle @> circle`, `polygon @> point`, and `polygon @> polygon`; each yields
/// `BOOL`.
fn check_geom_containment(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    use nusadb_core::engine::GeomKind::{Box, Circle, Point, Polygon};
    let (container, element) = match op {
        ast::BinaryOp::JsonContains => (left, right),
        _ => (right, left),
    };
    if matches!(
        (container, element),
        (
            ColumnType::Geometry(Box | Circle | Polygon),
            ColumnType::Geometry(Point)
        ) | (ColumnType::Geometry(Circle), ColumnType::Geometry(Circle))
            | (ColumnType::Geometry(Polygon), ColumnType::Geometry(Polygon))
    ) {
        Ok(ColumnType::Bool)
    } else {
        Err(Error::TypeMismatch {
            context: "geometric containment (`@>` / `<@`)".to_owned(),
            expected: container,
            found: element,
        })
    }
}

/// `json @? jsonpath` — a JSON (or text) document on the left, a jsonpath text on the right,
/// yielding `BOOL`.
pub(super) fn check_json_path_exists(
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let text_like = |t: ColumnType| {
        matches!(
            t,
            ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
        )
    };
    if (left == ColumnType::Json || text_like(left)) && text_like(right) {
        Ok(ColumnType::Bool)
    } else {
        Err(Error::TypeMismatch {
            context: "`@?` json path exists".to_owned(),
            expected: ColumnType::Json,
            found: if left == ColumnType::Json || text_like(left) {
                right
            } else {
                left
            },
        })
    }
}

/// Type rule for `@@` (F1): both operands are the text forms of a `tsvector`/`tsquery` (either
/// order, like the reference engine), so both must be `TEXT`; the result is the `BOOL` match.
pub(super) fn check_ts_match(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    // `@@` also overloads as a jsonpath predicate check: a JSON document on the left, a jsonpath text
    // on the right, yielding the `BOOL` (or NULL) predicate result — like `jsonb_path_match`.
    if left == ColumnType::Json
        && matches!(
            right,
            ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_)
        )
    {
        return Ok(ColumnType::Bool);
    }
    // `tsvector @@ tsquery`, either operand order, is the native form. A text operand is also
    // accepted (parsed at evaluation), keeping `to_tsvector(...) @@ '…'` and column-of-text usage.
    let is_ts_operand = |ty: ColumnType| {
        matches!(
            ty,
            ColumnType::Tsvector
                | ColumnType::Tsquery
                | ColumnType::Text
                | ColumnType::VarChar(_)
                | ColumnType::Char(_)
        )
    };
    if is_ts_operand(left) && is_ts_operand(right) {
        Ok(ColumnType::Bool)
    } else {
        Err(Error::TypeMismatch {
            context: "`@@` text-search match".to_owned(),
            expected: ColumnType::Tsvector,
            found: if is_ts_operand(left) { right } else { left },
        })
    }
}

/// Type rule for the vector distance operators `<=>` / `<->` / `<#>` / `<+>`: both operands must
/// be `VECTOR`s of the same dimension; the result is the `FLOAT` distance. A bare `NULL` operand is
/// already typed from its sibling earlier.
pub(super) fn check_vector_distance(
    operator: &str,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    match (left, right) {
        (ColumnType::Vector(a), ColumnType::Vector(b)) if a == b => Ok(ColumnType::Float),
        _ => Err(Error::TypeMismatch {
            context: format!("`{operator}` vector distance"),
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
#[allow(
    clippy::too_many_lines,
    reason = "one flat pass over the text-polymorphic family plus the CHAR(n) octet-length rewrite"
)]
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
        return Err(Error::FunctionArgs(format!(
            "{name}() expects {arity} argument(s), got {}",
            args.len()
        )));
    }
    // `LENGTH(lseg)` and `LENGTH(path)` are a FLOAT (the segment's / path's Euclidean length), unlike
    // the INT text/BYTEA/BIT lengths — the sole arg's type decides the result.
    let lseg_ty = ColumnType::Geometry(nusadb_core::engine::GeomKind::Lseg);
    let path_ty = ColumnType::Geometry(nusadb_core::engine::GeomKind::Path);
    let is_geom_length_arg = |ty: ColumnType| ty == lseg_ty || ty == path_ty;
    let mut length_of_geometry = false;
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
                || is_bit_type(typed.ty)
                || (func == F::Length && is_geom_length_arg(typed.ty))
                // `length(tsvector)` counts distinct lexemes (an INT, not the char count).
                || (func == F::Length && typed.ty == ColumnType::Tsvector)
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
        length_of_geometry = func == F::Length && is_geom_length_arg(typed.ty);
        typed_args.push(typed);
    }
    // `CHAR(n)` is blank-padded to `n` characters in the reference engine, and `octet_length` counts
    // that padding — NusaDB stores the value blank-trimmed, so add the `(n - char_length)` pad bytes
    // (each a single-byte space) onto the octet length of the stored value. `length` (char count) and
    // `bit_length` do not count the padding, so they need no adjustment and are left as-is. The
    // `CHAR(n)` width is gone from the arg's type (it normalizes to `TEXT`), so read it from the raw
    // column reference in scope; only a bare `CHAR(n)` column has this blank-padded width.
    let char_pad_len = match args.first() {
        Some(ast::Expr::Column(name)) => super::scoped_char_len(scope, None, name),
        Some(ast::Expr::QualifiedColumn { table, column }) => {
            super::scoped_char_len(scope, Some(table), column)
        },
        _ => None,
    };
    if func == F::OctetLength
        && let Some(n) = char_pad_len
        && let Some(arg) = typed_args.first().cloned()
    {
        let int_lit = |v: i64| TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Int(v)),
            ty: ColumnType::Int,
        };
        let call = |f: F, a: TypedExpr| TypedExpr {
            kind: TypedExprKind::ScalarFunction {
                func: f,
                args: vec![a],
            },
            ty: ColumnType::Int,
        };
        let bin = |op: ast::BinaryOp, l: TypedExpr, r: TypedExpr| TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(l),
                op,
                right: Box::new(r),
            },
            ty: ColumnType::Int,
        };
        // octet_length(arg) + (n - char_length(arg))
        let pad = bin(
            ast::BinaryOp::Minus,
            int_lit(i64::from(n)),
            call(F::Length, arg.clone()),
        );
        return Ok(bin(ast::BinaryOp::Plus, call(F::OctetLength, arg), pad));
    }
    Ok(TypedExpr {
        kind: TypedExprKind::ScalarFunction {
            func,
            args: typed_args,
        },
        ty: if length_of_geometry {
            ColumnType::Float
        } else if length_family {
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
        return Err(Error::FunctionArgs(format!(
            "substring() expects 2..=3 argument(s), got {}",
            args.len()
        )));
    };
    if rest.len() > 1 {
        return Err(Error::FunctionArgs(format!(
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
        // JSON concatenation: objects merge, arrays concatenate, anything else pairs up into an
        // array (see `json::concat`). As with `@>`, the other side may be a bare text value, parsed
        // as JSON at evaluation. Checked before the text-coercion arms below, which would otherwise
        // render the JSON side as text and quietly produce a string.
        (ColumnType::Json, ColumnType::Json | ColumnType::Text)
        | (ColumnType::Text, ColumnType::Json) => Ok(ColumnType::Json),
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
        // `interval * number` scales each component (commutative) → INTERVAL. An INT factor is exact;
        // a FLOAT or NUMERIC factor scales fractionally (`INTERVAL '1 month' * 1.5`).
        ast::BinaryOp::Multiply => {
            let numeric =
                |t: ColumnType| matches!(t, Int | ColumnType::Float | ColumnType::Numeric { .. });
            match (left, right) {
                (Interval, t) | (t, Interval) if numeric(t) => Some(Interval),
                _ => None,
            }
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

/// The key type a JSON key operator (`-`, `?`, `?|`, `?&`) wants on its right, or `None` when `op`
/// is not one of them (or its left operand is not a JSON document, so `-` is ordinary arithmetic).
/// Used to type a bare `NULL` key, which would otherwise inherit the document's own type.
const fn json_key_operand_type(op: ast::BinaryOp, left: ColumnType) -> Option<ColumnType> {
    match op {
        ast::BinaryOp::Minus | ast::BinaryOp::JsonExists if matches!(left, ColumnType::Json) => {
            Some(ColumnType::Text)
        },
        ast::BinaryOp::JsonExistsAny | ast::BinaryOp::JsonExistsAll
            if matches!(left, ColumnType::Json) =>
        {
            Some(ColumnType::Array(nusadb_core::engine::ArrayElem::Text))
        },
        _ => None,
    }
}

/// Type rule for JSON `-`: the left operand is `JSON` (checked by the caller) and the right is the
/// key to remove — `TEXT` (an object member / array string element), `INT` (an array index), or
/// `TEXT[]` (several keys at once). The result is the trimmed `JSON` document.
pub(super) fn check_json_delete(right: ColumnType) -> Result<ColumnType, Error> {
    if matches!(
        right,
        ColumnType::Text
            | ColumnType::Int
            | ColumnType::Array(nusadb_core::engine::ArrayElem::Text)
    ) {
        Ok(ColumnType::Json)
    } else {
        Err(Error::TypeMismatch {
            context: "JSON `-` key (TEXT, INT or TEXT[])".to_owned(),
            expected: ColumnType::Text,
            found: right,
        })
    }
}

/// Type rule for JSON `#-`: the left operand is `JSON` (a `JSONB` value shares the same physical
/// type) and the right is the `text[]` path to the element to remove. A bare text value like
/// `'{a,b}'` is accepted where `text[]` is wanted and parsed at evaluation, the same leniency `#>`
/// gives its path. The result is the trimmed `JSON` document.
pub(super) fn check_json_delete_path(
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    if !matches!(left, ColumnType::Json | ColumnType::Jsonb) {
        return Err(Error::TypeMismatch {
            context: "JSON `#-` document".to_owned(),
            expected: ColumnType::Json,
            found: left,
        });
    }
    if !matches!(
        right,
        ColumnType::Array(nusadb_core::engine::ArrayElem::Text) | ColumnType::Text
    ) {
        return Err(Error::TypeMismatch {
            context: "JSON `#-` path".to_owned(),
            expected: ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
            found: right,
        });
    }
    Ok(ColumnType::Json)
}

/// Type rule for the JSON key-existence operators: the left operand is `JSON`, the right a single
/// `TEXT` key for `?` or a `TEXT[]` key list for `?|` / `?&`; the result is `BOOL`. A bare text
/// value is accepted where `text[]` is wanted and parsed as `{a,b}` at evaluation, the same
/// leniency `#>` gives its path.
pub(super) fn check_json_exists(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let text_array = ColumnType::Array(nusadb_core::engine::ArrayElem::Text);
    let (wanted, ok) = if op == ast::BinaryOp::JsonExists {
        (ColumnType::Text, right == ColumnType::Text)
    } else {
        (
            text_array,
            matches!(right, ColumnType::Text) || right == text_array,
        )
    };
    if left != ColumnType::Json {
        return Err(Error::TypeMismatch {
            context: "JSON `?`/`?|`/`?&` document".to_owned(),
            expected: ColumnType::Json,
            found: left,
        });
    }
    if !ok {
        return Err(Error::TypeMismatch {
            context: "JSON `?`/`?|`/`?&` key".to_owned(),
            expected: wanted,
            found: right,
        });
    }
    Ok(ColumnType::Bool)
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
        _ => Err(Error::Internal(
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

/// Type rule for element-wise vector arithmetic. `+`, `-`, and `*` combine two vectors of the same
/// dimension into a vector of that dimension; `/` and `%` are not defined on vectors. A dimension
/// mismatch is a loud error, like the vector distance functions.
fn check_vector_arithmetic(op: ast::BinaryOp, x: u32, y: u32) -> Result<ColumnType, Error> {
    use ast::BinaryOp as Op;
    if !matches!(op, Op::Plus | Op::Minus | Op::Multiply) {
        return Err(Error::Unsupported(
            "vector arithmetic supports only `+`, `-`, and `*`".to_owned(),
        ));
    }
    if x != y {
        return Err(Error::TypeMismatch {
            context: "vector arithmetic (dimensions differ)".to_owned(),
            expected: ColumnType::Vector(x),
            found: ColumnType::Vector(y),
        });
    }
    Ok(ColumnType::Vector(x))
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

/// Whether `ty` is a network-address type (`INET` or `CIDR`).
pub(super) const fn is_inet_type(ty: ColumnType) -> bool {
    matches!(ty, ColumnType::Inet | ColumnType::Cidr)
}

/// Whether `ty` is a bit-string type (`BIT` or `BIT VARYING`).
pub(super) const fn is_bit_type(ty: ColumnType) -> bool {
    matches!(ty, ColumnType::Bit(_) | ColumnType::VarBit(_))
}

/// Whether `ty` is a range type.
pub(super) const fn is_range_type(ty: ColumnType) -> bool {
    matches!(ty, ColumnType::Range(_))
}

/// Whether a value of `ty` can be the element of a range of `kind` — the element type itself, plus
/// the widening a bound comparison already tolerates (an `INT` against a numeric range, and any
/// declared precision/scale of `NUMERIC`).
///
/// The comparison is on the *physical* type, so `SMALLINT`/`BIGINT` count as the integer element
/// they are stored as — `int8range` is a spelling of the same integer kind, and rejecting a
/// `BIGINT` against it would be a hole in the surface, not a safety check.
const fn is_range_element(ty: ColumnType, kind: nusadb_core::engine::RangeKind) -> bool {
    use nusadb_core::engine::RangeKind;
    let ty = ty.physical();
    match kind {
        RangeKind::Int => matches!(ty, ColumnType::Int),
        RangeKind::Num => matches!(ty, ColumnType::Int | ColumnType::Numeric { .. }),
        RangeKind::Date => matches!(ty, ColumnType::Date),
        RangeKind::Ts => matches!(ty, ColumnType::Timestamp),
        RangeKind::TsTz => matches!(ty, ColumnType::TimestampTz),
    }
}

/// Type rule for range `&&` (overlap): both operands are ranges of the same element kind, and the
/// result is `BOOL`. Unlike containment there is no element form — two ranges or nothing.
fn check_range_overlap(left: ColumnType, right: ColumnType) -> Result<ColumnType, Error> {
    match (left, right) {
        (ColumnType::Range(a), ColumnType::Range(b)) if a == b => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: "range overlap `&&`".to_owned(),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for range `@>` / `<@` (containment): the container is a range, and the contained side
/// is either a range of the same element kind or a single element of it. The result is `BOOL`.
///
/// A bare `TEXT` operand is *not* accepted: `r @> '5'` could mean either form, so it is rejected
/// rather than guessed — the cast (`r @> '5'::int` / `r @> '[1,5)'::int4range`) says which.
fn check_range_containment(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let (container, contained) = if op == ast::BinaryOp::JsonContains {
        (left, right)
    } else {
        (right, left)
    };
    let ColumnType::Range(kind) = container else {
        return Err(Error::TypeMismatch {
            context: "range `@>`/`<@` container".to_owned(),
            expected: contained,
            found: container,
        });
    };
    if contained == container || is_range_element(contained, kind) {
        return Ok(ColumnType::Bool);
    }
    Err(Error::TypeMismatch {
        context: "range `@>`/`<@` contained operand (a range of the same kind, or one element)"
            .to_owned(),
        expected: kind.element_type(),
        found: contained,
    })
}

/// Type rule for the range set operators `+` (union), `*` (intersection), and `-` (difference): both
/// operands are ranges of the same element kind, and the result is that range type. A mismatched kind
/// (or a non-range operand) is a loud type error.
fn check_range_setop(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let symbol = match op {
        ast::BinaryOp::Plus => "+",
        ast::BinaryOp::Multiply => "*",
        _ => "-",
    };
    match (left, right) {
        (ColumnType::Range(a), ColumnType::Range(b)) if a == b => Ok(ColumnType::Range(a)),
        _ => Err(Error::TypeMismatch {
            context: format!("range set operator `{symbol}`"),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for the range strict-order predicates `<<` (strictly left of) and `>>` (strictly right
/// of): both operands are ranges of the same element kind, and the result is `BOOL`.
fn check_range_strict(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let symbol = if matches!(op, ast::BinaryOp::ShiftLeft) {
        "<<"
    } else {
        ">>"
    };
    match (left, right) {
        (ColumnType::Range(a), ColumnType::Range(b)) if a == b => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: format!("range strict-order operator `{symbol}`"),
            expected: left,
            found: right,
        }),
    }
}

/// Type rule for the range boundary predicates `-|-` (adjacent), `&<` (does not extend to the right
/// of), and `&>` (does not extend to the left of): both operands are ranges of the same element kind,
/// and the result is `BOOL`.
fn check_range_bound_predicate(
    op: ast::BinaryOp,
    left: ColumnType,
    right: ColumnType,
) -> Result<ColumnType, Error> {
    let symbol = match op {
        ast::BinaryOp::RangeAdjacent => "-|-",
        ast::BinaryOp::RangeNotExtendRight => "&<",
        _ => "&>",
    };
    match (left, right) {
        (ColumnType::Range(a), ColumnType::Range(b)) if a == b => Ok(ColumnType::Bool),
        _ => Err(Error::TypeMismatch {
            context: format!("range boundary operator `{symbol}`"),
            expected: left,
            found: right,
        }),
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
        // `!!` hints tsquery so a bare literal/`NULL` operand types from context.
        ast::UnaryOp::TsqueryNot => Some(ColumnType::Tsquery),
        ast::UnaryOp::Negate
        | ast::UnaryOp::Plus
        | ast::UnaryOp::BitNot
        | ast::UnaryOp::GeomCenter => None,
    };
    let operand = analyze_expr_agg(expr, scope, catalog, hint, aggregates)?;
    // A bare string literal under `!!` is coerced to tsquery (the unknown-literal rule).
    let operand = if op == ast::UnaryOp::TsqueryNot {
        coerce_text_literal_to(operand, ColumnType::Tsquery)
    } else {
        operand
    };
    let ty = match op {
        ast::UnaryOp::Not if operand.ty == ColumnType::Bool => ColumnType::Bool,
        ast::UnaryOp::Not => {
            return Err(Error::TypeMismatch {
                context: "NOT operator".to_owned(),
                expected: ColumnType::Bool,
                found: operand.ty,
            });
        },
        ast::UnaryOp::Negate | ast::UnaryOp::Plus if is_numeric(operand.ty) => operand.ty,
        ast::UnaryOp::Negate => {
            return Err(Error::TypeMismatch {
                context: "negation".to_owned(),
                expected: ColumnType::Int,
                found: operand.ty,
            });
        },
        ast::UnaryOp::Plus => {
            return Err(Error::TypeMismatch {
                context: "unary plus".to_owned(),
                expected: ColumnType::Int,
                found: operand.ty,
            });
        },
        // `~` complements an integer, a MACADDR, a MACADDR8, or a bit string bit-for-bit, preserving
        // the operand type (a bit string keeps its length).
        ast::UnaryOp::BitNot
            if matches!(
                operand.ty,
                ColumnType::Int | ColumnType::Macaddr | ColumnType::Macaddr8 | ColumnType::Bit(_)
            ) =>
        {
            operand.ty
        },
        ast::UnaryOp::BitNot => {
            return Err(Error::TypeMismatch {
                context: "bitwise complement".to_owned(),
                expected: ColumnType::Int,
                found: operand.ty,
            });
        },
        // `@@` — the center point of a box, circle, lseg, or polygon, yielding a `point`.
        ast::UnaryOp::GeomCenter
            if matches!(
                operand.ty,
                ColumnType::Geometry(
                    nusadb_core::engine::GeomKind::Box
                        | nusadb_core::engine::GeomKind::Circle
                        | nusadb_core::engine::GeomKind::Lseg
                        | nusadb_core::engine::GeomKind::Polygon
                )
            ) =>
        {
            ColumnType::Geometry(nusadb_core::engine::GeomKind::Point)
        },
        ast::UnaryOp::GeomCenter => {
            return Err(Error::TypeMismatch {
                context: "geometric center `@@`".to_owned(),
                expected: ColumnType::Geometry(nusadb_core::engine::GeomKind::Box),
                found: operand.ty,
            });
        },
        // `!!` — negate a tsquery, yielding a tsquery.
        ast::UnaryOp::TsqueryNot if operand.ty == ColumnType::Tsquery => ColumnType::Tsquery,
        ast::UnaryOp::TsqueryNot => {
            return Err(Error::TypeMismatch {
                context: "tsquery negation `!!`".to_owned(),
                expected: ColumnType::Tsquery,
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
        _ => Err(Error::InvalidStatement(format!(
            "{context} must return exactly one column"
        ))),
    }
}
