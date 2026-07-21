//! Query-parameter binding for the extended-query protocol.
//!
//! A prepared statement carries positional placeholders `$1`, `$2`, … (parsed
//! into [`ast::Expr::Parameter`]). [`bind_parameters`] decodes the client's
//! wire-format parameter values and substitutes each placeholder with the
//! corresponding literal *before* analysis, so the rest of the pipeline
//! (analyzer, planner, executor) sees an ordinary parameterless statement.
//!
//! Values arrive in text format (the same bytes a `DataRow` carries). Their type
//! is inferred from the text — integer, then float, then boolean, else text —
//! which covers the common driver cases. A value whose text is numeric but whose
//! target column is `TEXT` would be mis-inferred; binding such a value should use
//! an explicit `CAST`. (Precise type-directed binding via declared parameter
//! types is a follow-up.)

use crate::ast;
use crate::error::Error;

/// Substitute every `$n` placeholder in `stmt` with its bound value from
/// `params` (in wire text format; `None` = SQL `NULL`).
///
/// # Errors
/// [`Error::Unsupported`] if a referenced parameter has no bound value, or a
/// value is not valid UTF-8.
pub fn bind_parameters(
    mut stmt: ast::Statement,
    params: &[Option<Vec<u8>>],
) -> Result<ast::Statement, Error> {
    let decoded = params
        .iter()
        .map(|p| decode_param(p.as_deref()))
        .collect::<Result<Vec<_>, _>>()?;
    substitute_stmt(&mut stmt, &decoded)?;
    Ok(stmt)
}

/// Substitute the positional parameters (`$1`..`$n`) in `stmt` with `values` (SQL-level `EXECUTE`,
/// ). Unlike [`bind_parameters`] (which decodes wire bytes), the values are already evaluated.
///
/// # Errors
/// Propagates substitution errors (e.g. a placeholder index outside `values`).
pub fn substitute_values(
    mut stmt: ast::Statement,
    values: &[ast::Value],
) -> Result<ast::Statement, Error> {
    substitute_stmt(&mut stmt, values)?;
    Ok(stmt)
}

/// Decode one wire-format parameter into a literal value (type inferred from the
/// text). `None` is SQL `NULL`.
fn decode_param(raw: Option<&[u8]>) -> Result<ast::Value, Error> {
    let Some(bytes) = raw else {
        return Ok(ast::Value::Null);
    };
    let text = std::str::from_utf8(bytes)
        .map_err(|_| Error::Unsupported("parameter value is not valid UTF-8".to_owned()))?;
    if let Ok(i) = text.parse::<i64>() {
        return Ok(ast::Value::Int(i));
    }
    if let Ok(f) = text.parse::<f64>() {
        return Ok(ast::Value::Float(f));
    }
    Ok(match text {
        "true" | "t" | "TRUE" => ast::Value::Bool(true),
        "false" | "f" | "FALSE" => ast::Value::Bool(false),
        other => ast::Value::Text(other.to_owned()),
    })
}

/// The number of positional parameters (`$1`..`$n`) a statement references — the highest
/// placeholder index + 1, or 0 if it has none.
///
/// Used by the extended-query `Describe(Statement)` path to report an accurate
/// `ParameterDescription` count instead of hard-coding 0 (G7). Mirrors the expression coverage
/// of [`bind_parameters`] (it walks the same nodes that binding substitutes).
#[must_use]
pub fn parameter_count(stmt: &ast::Statement) -> usize {
    let mut max = 0usize; // exclusive upper bound: highest (placeholder index + 1) seen
    count_stmt(stmt, &mut max);
    max
}

fn count_stmt(stmt: &ast::Statement, max: &mut usize) {
    match stmt {
        ast::Statement::Select(select) => count_select(select, max),
        ast::Statement::SetOperation(set) => count_set_body(&set.body, max),
        ast::Statement::Insert(insert) => {
            if let ast::InsertSource::Values(rows) = &insert.source {
                for row in rows {
                    // A `None` cell is an explicit `DEFAULT` — it carries no placeholder.
                    for expr in row.iter().flatten() {
                        count_expr(expr, max);
                    }
                }
            }
            count_returning(&insert.returning, max);
        },
        ast::Statement::Update(update) => {
            for assignment in &update.assignments {
                count_expr(&assignment.value, max);
            }
            count_from(update.from.as_ref(), max);
            count_opt(update.filter.as_ref(), max);
            count_returning(&update.returning, max);
        },
        ast::Statement::Delete(delete) => {
            count_from(delete.using.as_ref(), max);
            count_opt(delete.filter.as_ref(), max);
            count_returning(&delete.returning, max);
        },
        ast::Statement::Merge(merge) => count_merge(merge, max),
        ast::Statement::Explain(inner, _) => count_stmt(inner, max),
        // A nested `CALL`'s arguments may reference the enclosing procedure's `$n`.
        ast::Statement::Call(call) => {
            for arg in &call.args {
                count_expr(arg, max);
            }
        },
        _ => {},
    }
}

fn count_merge(merge: &ast::Merge, max: &mut usize) {
    count_expr(&merge.on, max);
    for when in &merge.whens {
        match when {
            ast::MergeWhen::Matched { pred, action } => {
                count_opt(pred.as_ref(), max);
                if let ast::MatchedAction::Update { assignments } = action {
                    for assignment in assignments {
                        count_expr(&assignment.value, max);
                    }
                }
            },
            ast::MergeWhen::NotMatched { pred, insert } => {
                count_opt(pred.as_ref(), max);
                for value in &insert.values {
                    count_expr(value, max);
                }
            },
        }
    }
}

fn count_set_body(body: &ast::SelectBody, max: &mut usize) {
    match body {
        ast::SelectBody::Select(select) => count_select(select, max),
        ast::SelectBody::SetOp { left, right, .. } => {
            count_set_body(left, max);
            count_set_body(right, max);
        },
    }
}

fn count_select(select: &ast::Select, max: &mut usize) {
    for cte in &select.with {
        match &cte.body {
            ast::CteBody::Query(q) => count_set_body(q, max),
            ast::CteBody::Modifying(stmt) => count_stmt(stmt, max),
        }
    }
    if let Some(ast::Distinct::On(exprs)) = &select.distinct {
        for expr in exprs {
            count_expr(expr, max);
        }
    }
    for item in &select.projection {
        if let ast::SelectItem::Expr { expr, .. } = item {
            count_expr(expr, max);
        }
    }
    count_from(select.from.as_ref(), max);
    count_opt(select.filter.as_ref(), max);
    count_group_by(&select.group_by, max);
    count_opt(select.having.as_ref(), max);
    for order in &select.order_by {
        count_expr(&order.expr, max);
    }
}

fn count_table_ref(table: &ast::TableRef, max: &mut usize) {
    if let Some(subquery) = &table.subquery {
        count_select(subquery, max);
    }
    if let Some(values) = &table.values {
        for cell in values.iter().flatten() {
            count_expr(cell, max);
        }
    }
    if let Some(set_op) = &table.set_op {
        count_set_body(&set_op.body, max);
    }
}

fn count_from(from: Option<&ast::FromClause>, max: &mut usize) {
    if let Some(from) = from {
        count_table_ref(&from.base, max);
        for join in &from.joins {
            count_table_ref(&join.table, max);
            if let ast::JoinCondition::On(expr) = &join.condition {
                count_expr(expr, max);
            }
        }
    }
}

fn count_group_by(group_by: &ast::GroupBy, max: &mut usize) {
    match group_by {
        ast::GroupBy::Expressions(keys) => {
            for key in keys {
                count_expr(key, max);
            }
        },
        ast::GroupBy::Rollup(sets)
        | ast::GroupBy::Cube(sets)
        | ast::GroupBy::GroupingSets(sets) => {
            for group in sets {
                for expr in group {
                    count_expr(expr, max);
                }
            }
        },
    }
}

fn count_returning(items: &[ast::SelectItem], max: &mut usize) {
    for item in items {
        if let ast::SelectItem::Expr { expr, .. } = item {
            count_expr(expr, max);
        }
    }
}

fn count_opt(expr: Option<&ast::Expr>, max: &mut usize) {
    if let Some(e) = expr {
        count_expr(e, max);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-expression-variant walker; length tracks the AST"
)]
fn count_expr(expr: &ast::Expr, max: &mut usize) {
    match expr {
        ast::Expr::Parameter(n) => *max = (*max).max(n + 1),
        ast::Expr::Literal(_) | ast::Expr::Column(_) | ast::Expr::QualifiedColumn { .. } => {},
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            count_expr(left, max);
            count_expr(right, max);
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsBool { expr, .. }
        | ast::Expr::Cast { expr, .. } => count_expr(expr, max),
        ast::Expr::InList { expr, list, .. } => {
            count_expr(expr, max);
            for item in list {
                count_expr(item, max);
            }
        },
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            count_expr(expr, max);
            count_expr(low, max);
            count_expr(high, max);
        },
        ast::Expr::Like { expr, pattern, .. }
        | ast::Expr::SimilarTo { expr, pattern, .. }
        | ast::Expr::RegexMatch { expr, pattern, .. } => {
            count_expr(expr, max);
            count_expr(pattern, max);
        },
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => {
            count_opt(operand.as_deref(), max);
            for branch in branches {
                count_expr(&branch.when, max);
                count_expr(&branch.then, max);
            }
            count_opt(default.as_deref(), max);
        },
        ast::Expr::Coalesce(args)
        | ast::Expr::ScalarFunction { args, .. }
        | ast::Expr::FunctionCall { args, .. }
        | ast::Expr::SetReturning { args, .. } => {
            for arg in args {
                count_expr(arg, max);
            }
        },
        ast::Expr::Aggregate { arg, filter, .. } => {
            count_opt(arg.as_deref(), max);
            count_opt(filter.as_deref(), max);
        },
        ast::Expr::Encrypt { value, key } | ast::Expr::Decrypt { value, key } => {
            count_expr(value, max);
            count_expr(key, max);
        },
        ast::Expr::ScalarSubquery(select)
        | ast::Expr::Exists {
            subquery: select, ..
        } => {
            count_select(select, max);
        },
        ast::Expr::InSubquery { expr, subquery, .. }
        | ast::Expr::QuantifiedComparison { expr, subquery, .. } => {
            count_expr(expr, max);
            count_select(subquery, max);
        },
        ast::Expr::QuantifiedArray { expr, array, .. } => {
            count_expr(expr, max);
            count_expr(array, max);
        },
        ast::Expr::Row(items) | ast::Expr::ArrayLiteral(items) => {
            for item in items {
                count_expr(item, max);
            }
        },
        ast::Expr::Subscript { base, index } => {
            count_expr(base, max);
            count_expr(index, max);
        },
        ast::Expr::ArraySlice { base, lower, upper } => {
            count_expr(base, max);
            for bound in [lower, upper].into_iter().flatten() {
                count_expr(bound, max);
            }
        },
        ast::Expr::WindowFunction(wf) => {
            for arg in &wf.args {
                count_expr(arg, max);
            }
            for p in &wf.partition {
                count_expr(p, max);
            }
            for o in &wf.order {
                count_expr(&o.expr, max);
            }
            count_window_frame(wf.frame.as_ref(), max);
        },
        ast::Expr::WithinGroup(wg) => {
            for arg in &wg.args {
                count_expr(arg, max);
            }
            for o in &wg.order_by {
                count_expr(&o.expr, max);
            }
        },
    }
}

fn count_window_frame(frame: Option<&ast::WindowFrame>, max: &mut usize) {
    let Some(frame) = frame else { return };
    count_frame_bound(&frame.start, max);
    if let Some(end) = &frame.end {
        count_frame_bound(end, max);
    }
}

fn count_frame_bound(bound: &ast::WindowFrameBound, max: &mut usize) {
    match bound {
        ast::WindowFrameBound::Preceding(e) | ast::WindowFrameBound::Following(e) => {
            count_expr(e, max);
        },
        ast::WindowFrameBound::UnboundedPreceding
        | ast::WindowFrameBound::CurrentRow
        | ast::WindowFrameBound::UnboundedFollowing => {},
    }
}

/// Replace each call-parameter reference in `expr` with the call argument expression — SQL-function
/// inlining. A parameter is referenced either **positionally** as `$n` (→ `args[n-1]`,
/// parsed to [`ast::Expr::Parameter`]) **or by name** — the declared parameter name as an unqualified
/// identifier (→ `args[i]` for the `i`-th name in `param_names`). Both forms are bound here, so
/// `CREATE FUNCTION f(x INT) AS 'SELECT x+1'` is callable as `f(5)`, exactly like the
/// `$1` form. A *qualified* identifier (`t.x`) is never a parameter, so it is left as a column.
///
/// Unlike [`substitute_values`] (which substitutes literal values), this splices whole argument
/// expressions, so a function call composes like a macro. Placeholders nested inside a subquery are
/// left untouched (a function body should keep its parameters at the top level).
#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive arm per Expr variant; mirrors substitute_expr"
)]
pub(crate) fn substitute_param_exprs(
    expr: &mut ast::Expr,
    args: &[ast::Expr],
    param_names: &[String],
) {
    match expr {
        ast::Expr::Parameter(n) => {
            if let Some(arg) = args.get(*n) {
                *expr = arg.clone();
            }
        },
        // An unqualified identifier that names a declared parameter binds to the matching argument
        // (named-parameter call). Names were folded to lowercase at both `CREATE FUNCTION` and parse
        // time, so a plain match is correct. A non-parameter identifier stays a column.
        ast::Expr::Column(name) => {
            if let Some(idx) = param_names.iter().position(|p| p == name)
                && let Some(arg) = args.get(idx)
            {
                *expr = arg.clone();
            }
        },
        // Leaves carry no placeholder; a bare subquery body keeps its own placeholders (not
        // substituted in v1).
        ast::Expr::Literal(_)
        | ast::Expr::QualifiedColumn { .. }
        | ast::Expr::ScalarSubquery(_)
        | ast::Expr::Exists { .. } => {},
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            substitute_param_exprs(left, args, param_names);
            substitute_param_exprs(right, args, param_names);
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsBool { expr, .. }
        | ast::Expr::Cast { expr, .. } => substitute_param_exprs(expr, args, param_names),
        ast::Expr::InList { expr, list, .. } => {
            substitute_param_exprs(expr, args, param_names);
            for item in list {
                substitute_param_exprs(item, args, param_names);
            }
        },
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            substitute_param_exprs(expr, args, param_names);
            substitute_param_exprs(low, args, param_names);
            substitute_param_exprs(high, args, param_names);
        },
        ast::Expr::Like { expr, pattern, .. }
        | ast::Expr::SimilarTo { expr, pattern, .. }
        | ast::Expr::RegexMatch { expr, pattern, .. } => {
            substitute_param_exprs(expr, args, param_names);
            substitute_param_exprs(pattern, args, param_names);
        },
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(operand) = operand {
                substitute_param_exprs(operand, args, param_names);
            }
            for branch in branches {
                substitute_param_exprs(&mut branch.when, args, param_names);
                substitute_param_exprs(&mut branch.then, args, param_names);
            }
            if let Some(default) = default {
                substitute_param_exprs(default, args, param_names);
            }
        },
        ast::Expr::Coalesce(items)
        | ast::Expr::ScalarFunction { args: items, .. }
        | ast::Expr::FunctionCall { args: items, .. }
        | ast::Expr::SetReturning { args: items, .. }
        | ast::Expr::Row(items)
        | ast::Expr::ArrayLiteral(items) => {
            for item in items {
                substitute_param_exprs(item, args, param_names);
            }
        },
        ast::Expr::Aggregate { arg, filter, .. } => {
            if let Some(arg) = arg {
                substitute_param_exprs(arg, args, param_names);
            }
            if let Some(filter) = filter {
                substitute_param_exprs(filter, args, param_names);
            }
        },
        ast::Expr::Encrypt { value, key } | ast::Expr::Decrypt { value, key } => {
            substitute_param_exprs(value, args, param_names);
            substitute_param_exprs(key, args, param_names);
        },
        ast::Expr::Subscript { base, index } => {
            substitute_param_exprs(base, args, param_names);
            substitute_param_exprs(index, args, param_names);
        },
        ast::Expr::ArraySlice { base, lower, upper } => {
            substitute_param_exprs(base, args, param_names);
            for bound in [lower, upper].into_iter().flatten() {
                substitute_param_exprs(bound, args, param_names);
            }
        },
        ast::Expr::WindowFunction(wf) => {
            for arg in &mut wf.args {
                substitute_param_exprs(arg, args, param_names);
            }
            for partition in &mut wf.partition {
                substitute_param_exprs(partition, args, param_names);
            }
            for order in &mut wf.order {
                substitute_param_exprs(&mut order.expr, args, param_names);
            }
        },
        ast::Expr::WithinGroup(wg) => {
            for arg in &mut wg.args {
                substitute_param_exprs(arg, args, param_names);
            }
            for order in &mut wg.order_by {
                substitute_param_exprs(&mut order.expr, args, param_names);
            }
        },
        // Probe of a subquery: substitute the probe, but not inside the subquery body.
        ast::Expr::InSubquery { expr, .. } | ast::Expr::QuantifiedComparison { expr, .. } => {
            substitute_param_exprs(expr, args, param_names);
        },
        ast::Expr::QuantifiedArray { expr, array, .. } => {
            substitute_param_exprs(expr, args, param_names);
            substitute_param_exprs(array, args, param_names);
        },
    }
}

fn substitute_stmt(stmt: &mut ast::Statement, params: &[ast::Value]) -> Result<(), Error> {
    match stmt {
        ast::Statement::Select(select) => substitute_select(select, params),
        ast::Statement::SetOperation(set) => substitute_set_body(&mut set.body, params),
        ast::Statement::Insert(insert) => {
            match &mut insert.source {
                ast::InsertSource::Values(rows) => {
                    for row in rows.iter_mut() {
                        // A `None` cell is an explicit `DEFAULT` — no placeholder to substitute.
                        for expr in row.iter_mut().flatten() {
                            substitute_expr(expr, params)?;
                        }
                    }
                },
                ast::InsertSource::Select(_) | ast::InsertSource::DefaultValues => {},
            }
            substitute_returning(&mut insert.returning, params)
        },
        ast::Statement::Update(update) => {
            for assignment in &mut update.assignments {
                substitute_expr(&mut assignment.value, params)?;
            }
            substitute_from(update.from.as_mut(), params)?;
            substitute_opt(update.filter.as_mut(), params)?;
            substitute_returning(&mut update.returning, params)
        },
        ast::Statement::Delete(delete) => {
            substitute_from(delete.using.as_mut(), params)?;
            substitute_opt(delete.filter.as_mut(), params)?;
            substitute_returning(&mut delete.returning, params)
        },
        ast::Statement::Merge(merge) => substitute_merge(merge, params),
        ast::Statement::Explain(inner, _) => substitute_stmt(inner, params),
        // A nested `CALL`'s arguments may reference the enclosing procedure's `$n`.
        ast::Statement::Call(call) => {
            for arg in &mut call.args {
                substitute_expr(arg, params)?;
            }
            Ok(())
        },
        // Statements without value expressions cannot carry parameters.
        _ => Ok(()),
    }
}

fn substitute_merge(merge: &mut ast::Merge, params: &[ast::Value]) -> Result<(), Error> {
    substitute_expr(&mut merge.on, params)?;
    for when in &mut merge.whens {
        match when {
            ast::MergeWhen::Matched { pred, action } => {
                substitute_opt(pred.as_mut(), params)?;
                if let ast::MatchedAction::Update { assignments } = action {
                    for assignment in assignments {
                        substitute_expr(&mut assignment.value, params)?;
                    }
                }
            },
            ast::MergeWhen::NotMatched { pred, insert } => {
                substitute_opt(pred.as_mut(), params)?;
                for value in &mut insert.values {
                    substitute_expr(value, params)?;
                }
            },
        }
    }
    Ok(())
}

fn substitute_set_body(body: &mut ast::SelectBody, params: &[ast::Value]) -> Result<(), Error> {
    match body {
        ast::SelectBody::Select(select) => substitute_select(select, params),
        ast::SelectBody::SetOp { left, right, .. } => {
            substitute_set_body(left, params)?;
            substitute_set_body(right, params)
        },
    }
}

fn substitute_select(select: &mut ast::Select, params: &[ast::Value]) -> Result<(), Error> {
    for cte in &mut select.with {
        match &mut cte.body {
            ast::CteBody::Query(q) => substitute_set_body(q, params)?,
            ast::CteBody::Modifying(stmt) => substitute_stmt(stmt, params)?,
        }
    }
    if let Some(ast::Distinct::On(exprs)) = &mut select.distinct {
        for expr in exprs {
            substitute_expr(expr, params)?;
        }
    }
    for item in &mut select.projection {
        if let ast::SelectItem::Expr { expr, .. } = item {
            substitute_expr(expr, params)?;
        }
    }
    substitute_from(select.from.as_mut(), params)?;
    substitute_opt(select.filter.as_mut(), params)?;
    substitute_group_by(&mut select.group_by, params)?;
    substitute_opt(select.having.as_mut(), params)?;
    for order in &mut select.order_by {
        substitute_expr(&mut order.expr, params)?;
    }
    Ok(())
}

fn substitute_table_ref(table: &mut ast::TableRef, params: &[ast::Value]) -> Result<(), Error> {
    if let Some(subquery) = &mut table.subquery {
        substitute_select(subquery, params)?;
    }
    if let Some(values) = &mut table.values {
        for cell in values.iter_mut().flatten() {
            substitute_expr(cell, params)?;
        }
    }
    if let Some(set_op) = &mut table.set_op {
        substitute_set_body(&mut set_op.body, params)?;
    }
    Ok(())
}

fn substitute_from(from: Option<&mut ast::FromClause>, params: &[ast::Value]) -> Result<(), Error> {
    if let Some(from) = from {
        substitute_table_ref(&mut from.base, params)?;
        for join in &mut from.joins {
            substitute_table_ref(&mut join.table, params)?;
            if let ast::JoinCondition::On(expr) = &mut join.condition {
                substitute_expr(expr, params)?;
            }
        }
    }
    Ok(())
}

fn substitute_returning(items: &mut [ast::SelectItem], params: &[ast::Value]) -> Result<(), Error> {
    for item in items {
        if let ast::SelectItem::Expr { expr, .. } = item {
            substitute_expr(expr, params)?;
        }
    }
    Ok(())
}

fn substitute_group_by(group_by: &mut ast::GroupBy, params: &[ast::Value]) -> Result<(), Error> {
    match group_by {
        ast::GroupBy::Expressions(keys) => {
            for key in keys {
                substitute_expr(key, params)?;
            }
            Ok(())
        },
        ast::GroupBy::Rollup(sets)
        | ast::GroupBy::Cube(sets)
        | ast::GroupBy::GroupingSets(sets) => {
            for group in sets {
                for expr in group {
                    substitute_expr(expr, params)?;
                }
            }
            Ok(())
        },
    }
}

fn substitute_opt(expr: Option<&mut ast::Expr>, params: &[ast::Value]) -> Result<(), Error> {
    expr.map_or(Ok(()), |e| substitute_expr(e, params))
}

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-expression-variant walker; length tracks the AST"
)]
fn substitute_expr(expr: &mut ast::Expr, params: &[ast::Value]) -> Result<(), Error> {
    match expr {
        ast::Expr::Parameter(n) => {
            let value = params.get(*n).cloned().ok_or_else(|| {
                Error::Unsupported(format!("parameter ${} was not bound", *n + 1))
            })?;
            *expr = ast::Expr::Literal(value);
            Ok(())
        },
        ast::Expr::Literal(_) | ast::Expr::Column(_) | ast::Expr::QualifiedColumn { .. } => Ok(()),
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            substitute_expr(left, params)?;
            substitute_expr(right, params)
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsBool { expr, .. }
        | ast::Expr::Cast { expr, .. } => substitute_expr(expr, params),
        ast::Expr::InList { expr, list, .. } => {
            substitute_expr(expr, params)?;
            for item in list {
                substitute_expr(item, params)?;
            }
            Ok(())
        },
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            substitute_expr(expr, params)?;
            substitute_expr(low, params)?;
            substitute_expr(high, params)
        },
        ast::Expr::Like { expr, pattern, .. }
        | ast::Expr::SimilarTo { expr, pattern, .. }
        | ast::Expr::RegexMatch { expr, pattern, .. } => {
            substitute_expr(expr, params)?;
            substitute_expr(pattern, params)
        },
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => {
            substitute_opt(operand.as_deref_mut(), params)?;
            for branch in branches {
                substitute_expr(&mut branch.when, params)?;
                substitute_expr(&mut branch.then, params)?;
            }
            substitute_opt(default.as_deref_mut(), params)
        },
        ast::Expr::Coalesce(args)
        | ast::Expr::ScalarFunction { args, .. }
        | ast::Expr::FunctionCall { args, .. }
        | ast::Expr::SetReturning { args, .. } => {
            for arg in args {
                substitute_expr(arg, params)?;
            }
            Ok(())
        },
        ast::Expr::Aggregate { arg, filter, .. } => {
            substitute_opt(arg.as_deref_mut(), params)?;
            substitute_opt(filter.as_deref_mut(), params)
        },
        ast::Expr::Encrypt { value, key } | ast::Expr::Decrypt { value, key } => {
            substitute_expr(value, params)?;
            substitute_expr(key, params)
        },
        ast::Expr::ScalarSubquery(select)
        | ast::Expr::Exists {
            subquery: select, ..
        } => substitute_select(select, params),
        ast::Expr::InSubquery { expr, subquery, .. }
        | ast::Expr::QuantifiedComparison { expr, subquery, .. } => {
            substitute_expr(expr, params)?;
            substitute_select(subquery, params)
        },
        ast::Expr::QuantifiedArray { expr, array, .. } => {
            substitute_expr(expr, params)?;
            substitute_expr(array, params)
        },
        ast::Expr::Row(items) | ast::Expr::ArrayLiteral(items) => {
            for item in items {
                substitute_expr(item, params)?;
            }
            Ok(())
        },
        ast::Expr::Subscript { base, index } => {
            substitute_expr(base, params)?;
            substitute_expr(index, params)
        },
        ast::Expr::ArraySlice { base, lower, upper } => {
            substitute_expr(base, params)?;
            for bound in [lower, upper].into_iter().flatten() {
                substitute_expr(bound, params)?;
            }
            Ok(())
        },
        ast::Expr::WindowFunction(wf) => {
            for arg in &mut wf.args {
                substitute_expr(arg, params)?;
            }
            for p in &mut wf.partition {
                substitute_expr(p, params)?;
            }
            for o in &mut wf.order {
                substitute_expr(&mut o.expr, params)?;
            }
            substitute_window_frame(wf.frame.as_mut(), params)?;
            Ok(())
        },
        ast::Expr::WithinGroup(wg) => substitute_within_group(wg, params),
    }
}

fn substitute_within_group(wg: &mut ast::WithinGroup, params: &[ast::Value]) -> Result<(), Error> {
    for arg in &mut wg.args {
        substitute_expr(arg, params)?;
    }
    for o in &mut wg.order_by {
        substitute_expr(&mut o.expr, params)?;
    }
    Ok(())
}

fn substitute_window_frame(
    frame: Option<&mut ast::WindowFrame>,
    params: &[ast::Value],
) -> Result<(), Error> {
    let Some(frame) = frame else { return Ok(()) };
    substitute_frame_bound(&mut frame.start, params)?;
    if let Some(end) = &mut frame.end {
        substitute_frame_bound(end, params)?;
    }
    Ok(())
}

fn substitute_frame_bound(
    bound: &mut ast::WindowFrameBound,
    params: &[ast::Value],
) -> Result<(), Error> {
    match bound {
        ast::WindowFrameBound::Preceding(e) | ast::WindowFrameBound::Following(e) => {
            substitute_expr(e, params)
        },
        ast::WindowFrameBound::UnboundedPreceding
        | ast::WindowFrameBound::CurrentRow
        | ast::WindowFrameBound::UnboundedFollowing => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{bind_parameters, parameter_count};
    use crate::parser::parse;

    #[test]
    fn parameter_count_reports_highest_placeholder_plus_one() {
        let count = |sql: &str| parameter_count(&parse(sql).unwrap());
        assert_eq!(count("SELECT * FROM t"), 0);
        assert_eq!(count("SELECT * FROM t WHERE id = $1"), 1);
        assert_eq!(count("SELECT * FROM t WHERE id = $1 AND name = $2"), 2);
        // The highest index wins, even when reused or referenced out of order.
        assert_eq!(count("SELECT * FROM t WHERE a = $2 OR b = $2"), 2);
        assert_eq!(count("INSERT INTO t VALUES ($1, $3)"), 3);
        // Placeholders nested in expressions are counted too.
        assert_eq!(count("UPDATE t SET v = $1 WHERE id = COALESCE($2, $3)"), 3);
    }

    /// Bind params into `sql`, then re-render via the analyzer-agnostic `Debug` to confirm the
    /// placeholders became literals.
    fn bound(sql: &str, params: &[Option<Vec<u8>>]) -> String {
        let stmt = parse(sql).unwrap();
        format!("{:?}", bind_parameters(stmt, params).unwrap())
    }

    #[test]
    fn binds_int_text_and_null_in_where_and_values() {
        let out = bound(
            "SELECT * FROM t WHERE id = $1 AND name = $2",
            &[Some(b"42".to_vec()), Some(b"alice".to_vec())],
        );
        assert!(out.contains("Int(42)"), "{out}");
        assert!(out.contains("Text(\"alice\")"), "{out}");
        assert!(!out.contains("Parameter"), "{out}");

        let ins = bound(
            "INSERT INTO t VALUES ($1, $2)",
            &[Some(b"7".to_vec()), None],
        );
        assert!(ins.contains("Int(7)") && ins.contains("Null"), "{ins}");
    }

    #[test]
    fn float_and_bool_inference() {
        let out = bound(
            "SELECT * FROM t WHERE score = $1 AND active = $2",
            &[Some(b"3.5".to_vec()), Some(b"true".to_vec())],
        );
        assert!(out.contains("Float(3.5)"), "{out}");
        assert!(out.contains("Bool(true)"), "{out}");
    }

    #[test]
    fn unbound_parameter_is_rejected() {
        let stmt = parse("SELECT * FROM t WHERE id = $2").unwrap();
        assert!(bind_parameters(stmt, &[Some(b"1".to_vec())]).is_err());
    }

    #[test]
    fn no_parameters_is_a_passthrough() {
        let out = bound("SELECT id FROM t WHERE id = 1", &[]);
        assert!(
            out.contains("Int(1)") && !out.contains("Parameter"),
            "{out}"
        );
    }
}
