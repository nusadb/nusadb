//! Window-function resolution.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === Window functions ==========================================

/// Reserved, un-typeable name of the `i`-th synthetic window-result column. The
/// leading control char cannot occur in a parsed identifier, so it never
/// collides with a user column.
pub(super) fn window_col_name(i: usize) -> String {
    format!("\u{1}window#{i}")
}

/// Default output name for an un-aliased window function (the function's name).
pub(super) fn window_default_name(func: &ast::WindowFunc) -> String {
    use ast::{AggregateFunc as A, WindowFunc as W};
    match func {
        W::RowNumber => "row_number",
        W::Rank => "rank",
        W::DenseRank => "dense_rank",
        W::Ntile => "ntile",
        W::CumeDist => "cume_dist",
        W::PercentRank => "percent_rank",
        W::Lag => "lag",
        W::Lead => "lead",
        W::FirstValue => "first_value",
        W::LastValue => "last_value",
        W::NthValue => "nth_value",
        W::Aggregate(A::Count) => "count",
        W::Aggregate(A::Sum) => "sum",
        W::Aggregate(A::Avg) => "avg",
        W::Aggregate(A::Min) => "min",
        W::Aggregate(A::Max) => "max",
        // Ordered-set aggregates are not window functions, but the enum permits the pairing;
        // name them defensively (they never reach here via the parser's OVER path).
        W::Aggregate(A::PercentileCont) => "percentile_cont",
        W::Aggregate(A::PercentileDisc) => "percentile_disc",
        W::Aggregate(A::Mode) => "mode",
        // ARRAY_AGG and the statistical aggregates are not wired as window functions; named
        // defensively (the parser's OVER path never produces them).
        W::Aggregate(A::ArrayAgg) => "array_agg",
        W::Aggregate(A::JsonAgg) => "jsonb_agg",
        W::Aggregate(A::BoolAnd) => "bool_and",
        W::Aggregate(A::BoolOr) => "bool_or",
        W::Aggregate(A::Stddev) => "stddev",
        W::Aggregate(A::Variance) => "variance",
        W::Aggregate(A::StddevPop) => "stddev_pop",
        W::Aggregate(A::VarPop) => "var_pop",
        W::Aggregate(A::BitAnd) => "bit_and",
        W::Aggregate(A::BitOr) => "bit_or",
        W::Aggregate(A::BitXor) => "bit_xor",
        W::Aggregate(A::Corr) => "corr",
        W::Aggregate(A::CovarPop) => "covar_pop",
        W::Aggregate(A::CovarSamp) => "covar_samp",
        W::Aggregate(A::RegrCount) => "regr_count",
        W::Aggregate(A::RegrAvgx) => "regr_avgx",
        W::Aggregate(A::RegrAvgy) => "regr_avgy",
        W::Aggregate(A::RegrSxx) => "regr_sxx",
        W::Aggregate(A::RegrSyy) => "regr_syy",
        W::Aggregate(A::RegrSxy) => "regr_sxy",
        W::Aggregate(A::RegrSlope) => "regr_slope",
        W::Aggregate(A::RegrIntercept) => "regr_intercept",
        W::Aggregate(A::RegrR2) => "regr_r2",
        W::Aggregate(A::StringAgg) => "string_agg",
        // GROUPING is a super-aggregate indicator, never a window function; named defensively (the
        // parser's OVER path never produces it — it is a scalar built-in, not an aggregate spelling).
        W::Aggregate(A::Grouping) => "grouping",
    }
    .to_owned()
}

/// Extract window functions from a projection list. Each window call — at the *top* of a
/// `SELECT` item or nested inside a larger expression (`ROUND(AVG(x) OVER w, 2)`,
/// `CAST(AVG(x) OVER w AS INT)`) — is resolved against `scope`, pushed to `windows`, and rewritten
/// in place to a reference to its synthetic appended column, so the surrounding expression then
/// type-checks normally against the window-extended projection scope.
pub(super) fn rewrite_window_items(
    items: Vec<ast::SelectItem>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    windows: &mut Vec<WindowExpr>,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<Vec<ast::SelectItem>, Error> {
    let mut rewritten = Vec::with_capacity(items.len());
    for item in items {
        match item {
            // A bare top-level window keeps its nice default name (`avg`, `rank`, …).
            ast::SelectItem::Expr {
                expr: ast::Expr::WindowFunction(wf),
                alias,
            } => {
                let name = alias.unwrap_or_else(|| window_default_name(&wf.func));
                windows.push(resolve_window(
                    *wf,
                    scope,
                    catalog,
                    aggregates.as_deref_mut(),
                )?);
                rewritten.push(ast::SelectItem::Expr {
                    expr: ast::Expr::Column(window_col_name(windows.len() - 1)),
                    alias: Some(name),
                });
            },
            // Any other projection expression may CONTAIN window calls nested inside it; extract them
            // in place (its name is resolved later by the projection, like any composite expression).
            ast::SelectItem::Expr { mut expr, alias } => {
                extract_windows(
                    &mut expr,
                    scope,
                    catalog,
                    windows,
                    aggregates.as_deref_mut(),
                )?;
                rewritten.push(ast::SelectItem::Expr { expr, alias });
            },
            other => rewritten.push(other),
        }
    }
    Ok(rewritten)
}

/// Recursively replace every `WindowFunction` node within `expr` by a reference to its synthetic
/// window column (resolving + appending the [`WindowExpr`] to `windows`). Subqueries and aggregate
/// arguments are *not* descended — a window inside a subquery is that block's own concern, and a
/// window inside an aggregate argument is invalid SQL (rejected elsewhere).
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-expression-variant recursive walker; length tracks the AST"
)]
fn extract_windows(
    expr: &mut ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    windows: &mut Vec<WindowExpr>,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<(), Error> {
    use ast::Expr as E;
    if matches!(expr, E::WindowFunction(_)) {
        // Take the window function out and replace this node with its synthetic column reference.
        let taken = std::mem::replace(expr, E::Literal(ast::Value::Null));
        let E::WindowFunction(wf) = taken else {
            unreachable!("guarded by the matches! above")
        };
        windows.push(resolve_window(
            *wf,
            scope,
            catalog,
            aggregates.as_deref_mut(),
        )?);
        *expr = E::Column(window_col_name(windows.len() - 1));
        return Ok(());
    }
    // Borrow a child mutably and recurse, threading the aggregate sink so a grouping aggregate inside
    // a nested window's PARTITION/ORDER/args is extracted into it.
    let mut rec = |e: &mut ast::Expr, aggs: Option<&mut Vec<AggregateCall>>| {
        extract_windows(e, scope, catalog, windows, aggs)
    };
    match expr {
        E::Binary { left, right, .. } | E::IsDistinctFrom { left, right, .. } => {
            rec(left, aggregates.as_deref_mut())?;
            rec(right, aggregates.as_deref_mut())?;
        },
        E::Unary { expr: inner, .. }
        | E::IsNull { expr: inner, .. }
        | E::IsBool { expr: inner, .. }
        | E::Cast { expr: inner, .. } => rec(inner, aggregates.as_deref_mut())?,
        E::InList {
            expr: inner, list, ..
        } => {
            rec(inner, aggregates.as_deref_mut())?;
            for e in list {
                rec(e, aggregates.as_deref_mut())?;
            }
        },
        E::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            rec(inner, aggregates.as_deref_mut())?;
            rec(low, aggregates.as_deref_mut())?;
            rec(high, aggregates.as_deref_mut())?;
        },
        E::Like {
            expr: inner,
            pattern,
            ..
        }
        | E::SimilarTo {
            expr: inner,
            pattern,
            ..
        }
        | E::RegexMatch {
            expr: inner,
            pattern,
            ..
        } => {
            rec(inner, aggregates.as_deref_mut())?;
            rec(pattern, aggregates.as_deref_mut())?;
        },
        E::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(op) = operand {
                rec(op, aggregates.as_deref_mut())?;
            }
            for b in branches {
                rec(&mut b.when, aggregates.as_deref_mut())?;
                rec(&mut b.then, aggregates.as_deref_mut())?;
            }
            if let Some(d) = default {
                rec(d, aggregates.as_deref_mut())?;
            }
        },
        E::Encrypt { value, key } | E::Decrypt { value, key } => {
            rec(value, aggregates.as_deref_mut())?;
            rec(key, aggregates.as_deref_mut())?;
        },
        E::Subscript { base, index } => {
            rec(base, aggregates.as_deref_mut())?;
            rec(index, aggregates.as_deref_mut())?;
        },
        E::ArraySlice { base, lower, upper } => {
            rec(base, aggregates.as_deref_mut())?;
            if let Some(l) = lower {
                rec(l, aggregates.as_deref_mut())?;
            }
            if let Some(u) = upper {
                rec(u, aggregates.as_deref_mut())?;
            }
        },
        E::Coalesce(items)
        | E::Row(items)
        | E::ArrayLiteral(items)
        | E::ScalarFunction { args: items, .. }
        | E::SetReturning { args: items, .. }
        | E::FunctionCall { args: items, .. } => {
            for e in items {
                rec(e, aggregates.as_deref_mut())?;
            }
        },
        // Leaves, separate query blocks (subqueries), aggregate calls (a window there is invalid),
        // and the already-handled `WindowFunction` — nothing to descend into.
        _ => {},
    }
    Ok(())
}

/// Type-check one window-function call into a [`WindowExpr`]: ranking,
/// aggregate-over-window, navigation (`LAG`/`LEAD`/`FIRST_VALUE`/`LAST_VALUE`/
/// `NTH_VALUE`), and distribution (`NTILE`/`CUME_DIST`/`PERCENT_RANK`), with an
/// optional explicit frame: `ROWS` with offsets, or peer-based
/// `RANGE`/`GROUPS` with `UNBOUNDED`/`CURRENT ROW` bounds.
#[allow(
    clippy::too_many_lines,
    reason = "one linear branch per window-function shape sharing the same frame/argument validation; splitting would scatter that shared logic"
)]
pub(super) fn resolve_window(
    wf: ast::WindowFunction,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    mut aggregates: Option<&mut Vec<AggregateCall>>,
) -> Result<WindowExpr, Error> {
    use ast::WindowFunc as W;

    // A `GROUPS` frame counts peer groups, which are only defined by an ordering: without `ORDER BY`
    // there are no groups to count (matches the SQL standard / common-dialect rule).
    if matches!(
        wf.frame.as_ref().map(|f| &f.units),
        Some(ast::WindowFrameUnits::Groups)
    ) && wf.order.is_empty()
    {
        return Err(Error::Unsupported(
            "a GROUPS window frame requires an ORDER BY clause".to_owned(),
        ));
    }
    // `PARTITION BY` / `ORDER BY` keys may reference a grouping aggregate when the query also
    // aggregates (`rank() OVER (ORDER BY sum(x)) … GROUP BY …`): analyze them with the shared
    // aggregate sink so such an aggregate is extracted into it (and rebased onto the post-aggregation
    // row later). Without a sink (a window with no aggregation), an aggregate here is rejected as
    // before.
    let partition = wf
        .partition
        .iter()
        .map(|e| analyze_expr_agg(e, scope, catalog, None, aggregates.as_deref_mut()))
        .collect::<Result<Vec<_>, _>>()?;
    let order = wf
        .order
        .iter()
        .map(|item| {
            Ok(OrderByKey {
                expr: analyze_expr_agg(
                    &item.expr,
                    scope,
                    catalog,
                    None,
                    aggregates.as_deref_mut(),
                )?,
                ascending: item.ascending,
                nulls: item.nulls,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // The frame is resolved after the ORDER BY so a `RANGE <v> PRECEDING/FOLLOWING` offset can be
    // validated against the single ordering column it ranges over.
    let frame = wf
        .frame
        .as_ref()
        .map(|f| resolve_frame(f, &order))
        .transpose()?;
    // Resolve every argument expression against the source scope. `LAG`/`LEAD`'s 3rd (default)
    // argument fills the value column, so it is typed against the value expression's type — a bare
    // `NULL` default (`lag(x, 1, NULL)`) takes the value's type rather than failing as an ambiguous
    // NULL, and a literal of an assignable type is coerced downstream.
    let typed_args = match (&wf.func, wf.args.as_slice()) {
        (W::Lag | W::Lead, [value_expr, offset_expr, default_expr]) => {
            let value =
                analyze_expr_agg(value_expr, scope, catalog, None, aggregates.as_deref_mut())?;
            let offset =
                analyze_expr_agg(offset_expr, scope, catalog, None, aggregates.as_deref_mut())?;
            let default = analyze_expr_agg(
                default_expr,
                scope,
                catalog,
                Some(value.ty),
                aggregates.as_deref_mut(),
            )?;
            vec![value, offset, default]
        },
        _ => wf
            .args
            .iter()
            .map(|e| analyze_expr_agg(e, scope, catalog, None, aggregates.as_deref_mut()))
            .collect::<Result<Vec<_>, _>>()?,
    };

    let (args, result_ty) = match &wf.func {
        W::RowNumber | W::Rank | W::DenseRank => {
            if !typed_args.is_empty() {
                return Err(Error::Unsupported(
                    "ranking window functions take no arguments".to_owned(),
                ));
            }
            (Vec::new(), ColumnType::Int)
        },
        // CUME_DIST / PERCENT_RANK: no arguments, relative position in [0, 1].
        W::CumeDist | W::PercentRank => {
            if !typed_args.is_empty() {
                return Err(Error::Unsupported(
                    "CUME_DIST / PERCENT_RANK take no arguments".to_owned(),
                ));
            }
            (Vec::new(), ColumnType::Float)
        },
        // NTILE(n): one integer bucket-count argument, integer bucket number out.
        W::Ntile => {
            let n = single_int_arg(&typed_args, "NTILE")?;
            (vec![n], ColumnType::Int)
        },
        // FIRST_VALUE/LAST_VALUE(expr): result type is the argument's type.
        W::FirstValue | W::LastValue => {
            let expr = single_value_arg(&typed_args, &wf.func)?;
            let ty = expr.ty;
            (vec![expr], ty)
        },
        // LAG/LEAD(expr [, offset [, default]]): result type is the first argument's type.
        W::Lag | W::Lead => resolve_navigation_offset(typed_args, &wf.func)?,
        // NTH_VALUE(expr, n): result type is the first argument's type.
        W::NthValue => {
            let [expr, n] = <[TypedExpr; 2]>::try_from(typed_args).map_err(|_| {
                Error::Unsupported("NTH_VALUE takes exactly two arguments (expr, n)".to_owned())
            })?;
            require_int(&n, "NTH_VALUE position")?;
            let ty = expr.ty;
            (vec![expr, n], ty)
        },
        W::Aggregate(func) => {
            let (arg, ty) = analyze_aggregate(*func, wf.args.first(), scope, catalog)?;
            (arg.into_iter().collect(), ty)
        },
    };
    Ok(WindowExpr {
        func: wf.func,
        args,
        partition,
        order,
        frame,
        result_ty,
    })
}

/// Resolve an explicit window frame. `ROWS` frames use a physical row offset and `GROUPS`
/// frames a peer-group offset — both accept a non-negative integer-literal `n PRECEDING`/`FOLLOWING`.
/// `RANGE` is peer-based and (v1) allows only the `UNBOUNDED`/`CURRENT ROW` bounds — a `RANGE` value
/// offset is not yet supported. The shorthand `<unit> <bound>` (no `BETWEEN`) ends at `CURRENT ROW`.
fn resolve_frame(f: &ast::WindowFrame, order: &[OrderByKey]) -> Result<WindowFrame, Error> {
    use ast::WindowFrameBound as B;
    let peer_based = !matches!(f.units, ast::WindowFrameUnits::Rows);
    let range = matches!(f.units, ast::WindowFrameUnits::Range);
    // A `RANGE` value offset ranges over the single ordering column; resolve its type (and reject the
    // sub-cases the executor does not handle) so each bound can validate its offset against it. A
    // `GROUPS` offset is a peer-group count and a `ROWS` offset a physical row count, both integers.
    let is_offset = |b: &B| matches!(b, B::Preceding(_) | B::Following(_));
    let (range_order_ty, range_descending) =
        if range && (is_offset(&f.start) || f.end.as_ref().is_some_and(is_offset)) {
            match order {
                // A single ASC or DESC ordering column. The executor reverses the value-boundary
                // direction for DESC (preceding rows have larger keys), so both are supported.
                [key] => (Some(key.expr.ty), !key.ascending),
                _ => {
                    return Err(Error::Unsupported(
                        "a RANGE frame with a value offset requires exactly one ORDER BY column"
                            .to_owned(),
                    ));
                },
            }
        } else {
            (None, false)
        };
    let start = resolve_frame_bound(&f.start, range, range_order_ty)?;
    let end = match &f.end {
        Some(bound) => resolve_frame_bound(bound, range, range_order_ty)?,
        None => FrameBound::CurrentRow,
    };
    Ok(WindowFrame {
        start,
        end,
        peer_based,
        range_descending,
    })
}

/// Resolve one frame bound. A `ROWS`/`GROUPS` `<n> PRECEDING`/`FOLLOWING` offset is a non-negative
/// integer literal (a row or peer-group count); a `RANGE` offset is a non-negative value of the
/// ordering column's type (a numeric column → a numeric offset, a temporal column → an interval).
fn resolve_frame_bound(
    bound: &ast::WindowFrameBound,
    range: bool,
    range_order_ty: Option<ColumnType>,
) -> Result<FrameBound, Error> {
    use ast::WindowFrameBound as B;
    Ok(match bound {
        B::UnboundedPreceding => FrameBound::UnboundedPreceding,
        B::CurrentRow => FrameBound::CurrentRow,
        B::UnboundedFollowing => FrameBound::UnboundedFollowing,
        B::Preceding(offset) if range => {
            FrameBound::RangePreceding(const_range_offset(offset, range_order_ty)?)
        },
        B::Following(offset) if range => {
            FrameBound::RangeFollowing(const_range_offset(offset, range_order_ty)?)
        },
        B::Preceding(offset) => FrameBound::Preceding(const_frame_offset(offset)?),
        B::Following(offset) => FrameBound::Following(const_frame_offset(offset)?),
    })
}

/// A `RANGE` frame value offset, added to / subtracted from the current ordering value at execution.
/// v1 supports an **integer** ordering column with a non-negative integer offset, and a temporal
/// (`DATE`/`TIMESTAMP[TZ]`) ordering column with a non-negative `INTERVAL` offset — both reduce to an
/// i64 comparison key. A float/numeric ordering column is not yet supported (loud-reject, never a
/// silent-wrong frame).
fn const_range_offset(
    expr: &ast::Expr,
    order_ty: Option<ColumnType>,
) -> Result<crate::ast::Value, Error> {
    use crate::ast::Value as V;
    use ColumnType as T;
    let order_ty = order_ty.ok_or_else(|| {
        Error::Unsupported("a RANGE frame value offset requires an ORDER BY column".to_owned())
    })?;
    let int_ordered = matches!(order_ty, T::SmallInt | T::Int | T::BigInt);
    let temporal = matches!(order_ty, T::Date | T::Timestamp | T::TimestampTz);
    if !int_ordered && !temporal {
        return Err(Error::Unsupported(format!(
            "a RANGE frame value offset over a {order_ty:?} ordering is not yet supported \
             (only an integer or date/time ordering column)"
        )));
    }
    let ast::Expr::Literal(value) = expr else {
        return Err(Error::Unsupported(
            "a RANGE frame offset must be a literal value".to_owned(),
        ));
    };
    let ok = match value {
        V::Int(n) if int_ordered => *n >= 0,
        V::Interval(iv) if temporal => iv.months >= 0 && iv.days >= 0 && iv.micros >= 0,
        _ => false,
    };
    if ok {
        Ok(value.clone())
    } else {
        Err(Error::Unsupported(format!(
            "a RANGE frame offset over a {order_ty:?} ordering must be a non-negative {} literal",
            if temporal { "INTERVAL" } else { "integer" }
        )))
    }
}

/// A window-frame offset: a non-negative integer literal.
fn const_frame_offset(expr: &ast::Expr) -> Result<u64, Error> {
    match expr {
        ast::Expr::Literal(ast::Value::Int(n)) if *n >= 0 => Ok(u64::try_from(*n).unwrap_or(0)),
        _ => Err(Error::Unsupported(
            "window frame offset must be a non-negative integer literal".to_owned(),
        )),
    }
}

/// A window function taking exactly one integer argument (`NTILE`).
fn single_int_arg(args: &[TypedExpr], func: &str) -> Result<TypedExpr, Error> {
    let [arg] = args else {
        return Err(Error::Unsupported(format!(
            "{func} takes exactly one argument"
        )));
    };
    require_int(arg, func)?;
    Ok(arg.clone())
}

/// A window function taking exactly one value argument of any type (`FIRST_VALUE`/`LAST_VALUE`).
fn single_value_arg(args: &[TypedExpr], func: &ast::WindowFunc) -> Result<TypedExpr, Error> {
    let [arg] = args else {
        return Err(Error::Unsupported(format!(
            "{func:?} takes exactly one argument"
        )));
    };
    Ok(arg.clone())
}

/// Resolve `LAG`/`LEAD(expr [, offset [, default]])`: 1-3 args, result typed as `expr`.
fn resolve_navigation_offset(
    mut args: Vec<TypedExpr>,
    func: &ast::WindowFunc,
) -> Result<(Vec<TypedExpr>, ColumnType), Error> {
    if args.is_empty() || args.len() > 3 {
        return Err(Error::Unsupported(format!(
            "{func:?} takes 1 to 3 arguments (expr [, offset [, default]])"
        )));
    }
    if let Some(offset) = args.get(1) {
        require_int(offset, "LAG/LEAD offset")?;
    }
    // The result is the value's type, but the reference engine's LAG/LEAD return type drops the NUMERIC typmod (an
    // unconstrained NUMERIC). This matters for a coerced default: rendering follows each value's own
    // scale, so `lag(numeric(10,2)_col, 1, 0)` shows the default as `0`, not the column-scaled `0.00`,
    // while the actual values still render with their stored scale.
    let result_ty = match args.first().map_or(ColumnType::Int, |e| e.ty) {
        ColumnType::Numeric { .. } => ColumnType::Numeric {
            precision: 0,
            scale: 0,
        },
        other => other,
    };
    // The default fills the value column when the offset lands outside the partition, so it must be
    // the value's type. A literal NULL already fits any type; a matching type passes through; an
    // assignable one (e.g. an INT default into a NUMERIC/FLOAT value, as the reference engine coerces) is wrapped in a
    // cast so it evaluates to the value type; anything else is a loud type error.
    if args.len() == 3 {
        let default = args.remove(2);
        let coerced = if default.ty == result_ty
            || matches!(default.kind, TypedExprKind::Literal(ast::Value::Null))
        {
            default
        } else if super::assignable(result_ty, default.ty) {
            TypedExpr {
                // The cast's target type is carried in `ty`; `false` = a plain (non-try) cast.
                kind: TypedExprKind::Cast(Box::new(default), false),
                ty: result_ty,
            }
        } else {
            return Err(Error::TypeMismatch {
                context: "LAG/LEAD default".to_owned(),
                expected: result_ty,
                found: default.ty,
            });
        };
        args.push(coerced);
    }
    Ok((args, result_ty))
}

/// Require an `Int`-typed argument (offsets / positions / bucket counts).
fn require_int(expr: &TypedExpr, what: &str) -> Result<(), Error> {
    if expr.ty == ColumnType::Int {
        Ok(())
    } else {
        Err(Error::TypeMismatch {
            context: what.to_owned(),
            expected: ColumnType::Int,
            found: expr.ty,
        })
    }
}
