//! NusaScript interpreter — runs a parsed procedure block.
//!
//! Tree-walks the [`ScriptStmt`] block parsed by [`crate::parser::parse_script`]. A variable
//! environment maps declared names to current values; `IF`/`WHILE` drive control flow; embedded SQL
//! runs re-entrantly in the caller's transaction (via the executor's `dispatch`). Before an embedded
//! statement or an expression runs, the procedure's `$1..$n` parameters and the in-scope variables are
//! substituted with their literal values, so the analyzer/executor see an ordinary parameterless,
//! variable-free statement.
//!
//! `WHILE` and `FOR` are each bounded by an iteration cap so a runaway loop aborts rather than hangs.
#![allow(clippy::wildcard_imports)]

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use super::*;
use crate::parser::{ScriptBlock, ScriptStmt};

/// Maximum `WHILE` iterations before a loop is aborted as non-terminating.
const MAX_LOOP_ITERS: u64 = 1_000_000;

thread_local! {
    /// Monotonic counter for unique `EXCEPTION`-block savepoint names on this thread.
    static SAVEPOINT_SEQ: Cell<u64> = const { Cell::new(0) };
}

/// A unique savepoint name for an `EXCEPTION` block.
fn next_savepoint() -> String {
    SAVEPOINT_SEQ.with(|seq| {
        let n = seq.get();
        seq.set(n.wrapping_add(1));
        format!("__nusa_exc_{n}")
    })
}

/// Control-flow outcome of executing a statement or block.
enum Flow {
    /// Continue with the next statement.
    Normal,
    /// `RETURN` was hit — stop the enclosing procedure.
    Return,
}

/// The variable environment: declared name → current value.
pub(super) type Env = HashMap<String, ast::Value>;

/// Run a parsed NusaScript block with the call's `$n` arguments, returning the final
/// variable environment (so the caller can read back `OUT` parameter values).
pub(super) fn run_block(
    block: &ScriptBlock,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Env, Error> {
    let mut env = Env::new();
    exec_block(block, &mut env, params, engine, txn)?;
    Ok(env)
}

/// Execute a block, honoring its `EXCEPTION WHEN OTHERS THEN` handler: if the body errors and
/// a handler is present, roll the body's writes back to a savepoint and run the handler instead.
/// `Error::Cancelled` (a statement timeout / cancel) is never caught — it always propagates.
fn exec_block(
    block: &ScriptBlock,
    env: &mut Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Flow, Error> {
    let Some(handler) = &block.handler else {
        return exec_stmts(&block.body, env, params, engine, txn);
    };
    let savepoint = next_savepoint();
    engine.savepoint(txn, &savepoint)?;
    match exec_stmts(&block.body, env, params, engine, txn) {
        Ok(flow) => {
            engine.release_savepoint(txn, &savepoint)?;
            Ok(flow)
        },
        Err(Error::Cancelled) => Err(Error::Cancelled),
        Err(_caught) => {
            // Undo the body's partial writes, then run the handler in their place.
            engine.rollback_to(txn, &savepoint)?;
            exec_stmts(handler, env, params, engine, txn)
        },
    }
}

fn exec_stmts(
    stmts: &[ScriptStmt],
    env: &mut Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Flow, Error> {
    for stmt in stmts {
        if matches!(exec_one(stmt, env, params, engine, txn)?, Flow::Return) {
            return Ok(Flow::Return);
        }
    }
    Ok(Flow::Normal)
}

fn exec_one(
    stmt: &ScriptStmt,
    env: &mut Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Flow, Error> {
    match stmt {
        ScriptStmt::Declare { name, default } => {
            let value = match default {
                Some(expr) => eval_value(expr, env, params, engine, txn)?,
                None => ast::Value::Null,
            };
            env.insert(name.clone(), value);
            Ok(Flow::Normal)
        },
        ScriptStmt::Assign { name, value } => {
            let value = eval_value(value, env, params, engine, txn)?;
            env.insert(name.clone(), value);
            Ok(Flow::Normal)
        },
        ScriptStmt::If { arms, els } => {
            for (cond, body) in arms {
                if eval_bool(cond, env, params, engine, txn)? {
                    return exec_stmts(body, env, params, engine, txn);
                }
            }
            els.as_ref().map_or(Ok(Flow::Normal), |body| {
                exec_stmts(body, env, params, engine, txn)
            })
        },
        ScriptStmt::While { cond, body } => {
            let mut iterations = 0u64;
            while eval_bool(cond, env, params, engine, txn)? {
                iterations += 1;
                if iterations > MAX_LOOP_ITERS {
                    return Err(Error::Unsupported(
                        "NusaScript WHILE loop exceeded the iteration limit".to_owned(),
                    ));
                }
                if matches!(exec_stmts(body, env, params, engine, txn)?, Flow::Return) {
                    return Ok(Flow::Return);
                }
            }
            Ok(Flow::Normal)
        },
        ScriptStmt::For {
            var,
            low,
            high,
            body,
        } => {
            let lo = eval_int(low, env, params, engine, txn)?;
            let hi = eval_int(high, env, params, engine, txn)?;
            // Bound the loop like WHILE: `FOR i IN 1 TO <huge>` must not hang the connection.
            let mut iterations = 0u64;
            for i in lo..=hi {
                iterations += 1;
                if iterations > MAX_LOOP_ITERS {
                    return Err(Error::Unsupported(
                        "NusaScript FOR loop exceeded the iteration limit".to_owned(),
                    ));
                }
                env.insert(var.clone(), ast::Value::Int(i));
                if matches!(exec_stmts(body, env, params, engine, txn)?, Flow::Return) {
                    return Ok(Flow::Return);
                }
            }
            Ok(Flow::Normal)
        },
        ScriptStmt::Raise(expr) => {
            let value = eval_value(expr, env, params, engine, txn)?;
            Err(Error::Raised(message_text(&value)))
        },
        ScriptStmt::Return => Ok(Flow::Return),
        ScriptStmt::Block(block) => exec_block(block, env, params, engine, txn),
        ScriptStmt::Sql(sql) => {
            let bound = bind((**sql).clone(), env, params, engine, txn)?;
            let logical = crate::analyze(bound, &ExecCatalog { engine, txn })?;
            super::dispatch(crate::plan(logical), engine, txn)?;
            Ok(Flow::Normal)
        },
    }
}

/// Evaluate a scalar expression with the current variables + parameters bound, by running
/// `SELECT (expr)` and reading the single result cell.
fn eval_value(
    expr: &ast::Expr,
    env: &Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ast::Value, Error> {
    let select = wrap_select(expr.clone());
    let bound = bind(select, env, params, engine, txn)?;
    let logical = crate::analyze(bound, &ExecCatalog { engine, txn })?;
    match super::dispatch(crate::plan(logical), engine, txn)? {
        ExecutionResult::Rows { rows, .. } => Ok(rows
            .into_iter()
            .next()
            .and_then(|row| row.into_iter().next())
            .unwrap_or(ast::Value::Null)),
        _ => Ok(ast::Value::Null),
    }
}

/// Evaluate an expression expected to be an integer (a `FOR` loop bound).
fn eval_int(
    expr: &ast::Expr,
    env: &Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<i64, Error> {
    match eval_value(expr, env, params, engine, txn)? {
        ast::Value::Int(n) => Ok(n),
        other => Err(Error::Unsupported(format!(
            "FOR loop bound must be an integer, got {other:?}"
        ))),
    }
}

/// Evaluate a boolean condition: `TRUE` fires, `FALSE`/`NULL` do not (SQL three-valued logic).
fn eval_bool(
    expr: &ast::Expr,
    env: &Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    Ok(matches!(
        eval_value(expr, env, params, engine, txn)?,
        ast::Value::Bool(true)
    ))
}

/// Wrap a scalar expression in `SELECT (expr)` so it can be analyzed + evaluated.
fn wrap_select(expr: ast::Expr) -> ast::Statement {
    ast::Statement::Select(ast::Select {
        with: Vec::new(),
        distinct: None,
        projection: vec![ast::SelectItem::Expr { expr, alias: None }],
        from: None,
        filter: None,
        group_by: ast::GroupBy::Expressions(Vec::new()),
        having: None,
        order_by: Vec::new(),
        limit: None,
        limit_with_ties: false,
        offset: None,
        lock: None,
    })
}

/// Render a value as the text of a `RAISE` message.
fn message_text(value: &ast::Value) -> String {
    match value {
        ast::Value::Text(s) => s.clone(),
        ast::Value::Null => "NULL".to_owned(),
        ast::Value::Bool(b) => b.to_string(),
        ast::Value::Int(n) => n.to_string(),
        ast::Value::Float(f) => f.to_string(),
        other => format!("{other:?}"),
    }
}

/// Bind a statement's `$n` parameters and in-scope variables to their literal values, so the rest of
/// the pipeline sees an ordinary parameterless, variable-free statement.
fn bind(
    stmt: ast::Statement,
    env: &Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ast::Statement, Error> {
    let mut stmt = crate::params::substitute_values(stmt, params)?;
    let ctx = BindCtx { env, engine, txn };
    bind_vars_stmt(&mut stmt, &ctx, &HashSet::new())?;
    Ok(stmt)
}

// === Variable substitution (bare `Column(name)` → literal) ================
//
// A bare `Column(name)` is replaced by its variable value only when `name` is NOT a column of a table
// in scope at that point: a real column reference shadows a like-named variable (matching SQL scoping),
// so e.g. `DECLARE n INT DEFAULT 99; INSERT INTO t SELECT n FROM t` inserts column `t.n`, not 99 (
// shadow-corruption fix). The in-scope column names are threaded down as `shadowed`, extended with each
// `FROM`/target table's columns as the walk descends into the scope that table introduces. The INSERT
// target is *not* in scope for its `VALUES`/`SELECT` source, so `INSERT INTO t(id) VALUES (id)` still
// binds the variable `id`.
//
// `shadowed` resolves *base table / view* columns (`lookup_table_as_of`). A CTE's or derived table's
// own output columns are not enumerated, so a bare reference to such an output column that also names
// a variable is still substituted — a narrow residual of the original defect (the common base-table
// case is closed). A follow-up should thread CTE/derived output names in too.

/// Context for the bind walk: the variable environment plus the engine/transaction used to resolve a
/// scope's table columns.
struct BindCtx<'a> {
    env: &'a Env,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
}

/// The column names of `name` (empty if the table/view is unknown), used to shadow like-named variables.
fn table_columns(name: &str, ctx: &BindCtx) -> Result<HashSet<String>, Error> {
    Ok(ctx
        .engine
        .lookup_table_as_of(ctx.txn, name)?
        .map(|schema| schema.columns.into_iter().map(|c| c.name).collect())
        .unwrap_or_default())
}

/// The union of column names of every *named* table in `from` (a derived-table subquery has its own
/// inner scope, resolved when the walk recurses into it, so it contributes none here).
fn from_columns(from: Option<&ast::FromClause>, ctx: &BindCtx) -> Result<HashSet<String>, Error> {
    let mut cols = HashSet::new();
    if let Some(from) = from {
        // A derived table — `(SELECT ...)`, `(VALUES ...)`, or `(SELECT ... UNION ...)` — has its own
        // inner scope, so it contributes no outer column names here (only a *named* table does).
        if from.base.subquery.is_none() && from.base.values.is_none() && from.base.set_op.is_none()
        {
            cols.extend(table_columns(&from.base.name, ctx)?);
        }
        for join in &from.joins {
            if join.table.subquery.is_none()
                && join.table.values.is_none()
                && join.table.set_op.is_none()
            {
                cols.extend(table_columns(&join.table.name, ctx)?);
            }
        }
    }
    Ok(cols)
}

fn bind_vars_stmt(
    stmt: &mut ast::Statement,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    match stmt {
        ast::Statement::Select(select) => bind_vars_select(select, ctx, shadowed)?,
        ast::Statement::SetOperation(set) => bind_vars_set_body(&mut set.body, ctx, shadowed)?,
        ast::Statement::Insert(insert) => {
            // The row source cannot reference the INSERT target's columns, so it binds under the
            // caller's scope (`VALUES (id)` stays the variable `id` even if the target has column id).
            match &mut insert.source {
                ast::InsertSource::Values(rows) => {
                    for row in rows.iter_mut() {
                        // A `None` cell is an explicit `DEFAULT` — no expression to bind.
                        for expr in row.iter_mut().flatten() {
                            bind_vars_expr(expr, ctx, shadowed)?;
                        }
                    }
                },
                ast::InsertSource::Select(select) => bind_vars_select(select, ctx, shadowed)?,
                ast::InsertSource::DefaultValues => {},
            }
            // RETURNING projects the inserted row, so the target's columns are in scope for it.
            if !insert.returning.is_empty() {
                let scope = with_table(shadowed, &insert.table, ctx)?;
                bind_vars_items(&mut insert.returning, ctx, &scope)?;
            }
        },
        ast::Statement::Update(update) => {
            // The target table and any FROM source are in scope for SET/WHERE/RETURNING.
            let mut scope = with_table(shadowed, &update.table, ctx)?;
            scope.extend(from_columns(update.from.as_ref(), ctx)?);
            for assignment in &mut update.assignments {
                bind_vars_expr(&mut assignment.value, ctx, &scope)?;
            }
            bind_vars_from(update.from.as_mut(), ctx, &scope)?;
            bind_vars_opt(update.filter.as_mut(), ctx, &scope)?;
            bind_vars_items(&mut update.returning, ctx, &scope)?;
        },
        ast::Statement::Delete(delete) => {
            let mut scope = with_table(shadowed, &delete.table, ctx)?;
            scope.extend(from_columns(delete.using.as_ref(), ctx)?);
            bind_vars_from(delete.using.as_mut(), ctx, &scope)?;
            bind_vars_opt(delete.filter.as_mut(), ctx, &scope)?;
            bind_vars_items(&mut delete.returning, ctx, &scope)?;
        },
        ast::Statement::Call(call) => {
            for arg in &mut call.args {
                bind_vars_expr(arg, ctx, shadowed)?;
            }
        },
        // Other statement kinds never appear in a NusaScript body.
        _ => {},
    }
    Ok(())
}

/// `shadowed` extended with the columns of table `name` (the scope a target/FROM table introduces).
fn with_table(
    shadowed: &HashSet<String>,
    name: &str,
    ctx: &BindCtx,
) -> Result<HashSet<String>, Error> {
    let mut scope = shadowed.clone();
    scope.extend(table_columns(name, ctx)?);
    Ok(scope)
}

fn bind_vars_set_body(
    body: &mut ast::SelectBody,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    match body {
        ast::SelectBody::Select(select) => bind_vars_select(select, ctx, shadowed)?,
        ast::SelectBody::SetOp { left, right, .. } => {
            bind_vars_set_body(left, ctx, shadowed)?;
            bind_vars_set_body(right, ctx, shadowed)?;
        },
    }
    Ok(())
}

fn bind_vars_select(
    select: &mut ast::Select,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    // A CTE body cannot see this query's FROM, so it binds under the inherited scope.
    for cte in &mut select.with {
        match &mut cte.body {
            ast::CteBody::Query(q) => bind_vars_set_body(q, ctx, shadowed)?,
            ast::CteBody::Modifying(stmt) => bind_vars_stmt(stmt, ctx, shadowed)?,
        }
    }
    // This SELECT's FROM tables shadow like-named variables for its own clause expressions.
    let mut scope = shadowed.clone();
    scope.extend(from_columns(select.from.as_ref(), ctx)?);
    if let Some(ast::Distinct::On(exprs)) = &mut select.distinct {
        for expr in exprs {
            bind_vars_expr(expr, ctx, &scope)?;
        }
    }
    for item in &mut select.projection {
        if let ast::SelectItem::Expr { expr, .. } = item {
            bind_vars_expr(expr, ctx, &scope)?;
        }
    }
    bind_vars_from(select.from.as_mut(), ctx, &scope)?;
    bind_vars_opt(select.filter.as_mut(), ctx, &scope)?;
    bind_vars_group_by(&mut select.group_by, ctx, &scope)?;
    bind_vars_opt(select.having.as_mut(), ctx, &scope)?;
    for order in &mut select.order_by {
        bind_vars_expr(&mut order.expr, ctx, &scope)?;
    }
    Ok(())
}

fn bind_vars_from(
    from: Option<&mut ast::FromClause>,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    if let Some(from) = from {
        bind_vars_table_ref(&mut from.base, ctx, shadowed)?;
        for join in &mut from.joins {
            bind_vars_table_ref(&mut join.table, ctx, shadowed)?;
            if let ast::JoinCondition::On(expr) = &mut join.condition {
                bind_vars_expr(expr, ctx, shadowed)?;
            }
        }
    }
    Ok(())
}

/// Bind NusaScript variables inside a FROM item: a derived-table subquery, or the cell expressions
/// of a `(VALUES ...)` derived table.
fn bind_vars_table_ref(
    table: &mut ast::TableRef,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    if let Some(subquery) = &mut table.subquery {
        bind_vars_select(subquery, ctx, shadowed)?;
    }
    if let Some(values) = &mut table.values {
        for cell in values.iter_mut().flatten() {
            bind_vars_expr(cell, ctx, shadowed)?;
        }
    }
    if let Some(set_op) = &mut table.set_op {
        bind_vars_set_body(&mut set_op.body, ctx, shadowed)?;
    }
    Ok(())
}

fn bind_vars_items(
    items: &mut [ast::SelectItem],
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    for item in items {
        if let ast::SelectItem::Expr { expr, .. } = item {
            bind_vars_expr(expr, ctx, shadowed)?;
        }
    }
    Ok(())
}

fn bind_vars_group_by(
    group_by: &mut ast::GroupBy,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    match group_by {
        ast::GroupBy::Expressions(keys) => {
            for key in keys {
                bind_vars_expr(key, ctx, shadowed)?;
            }
        },
        ast::GroupBy::Rollup(sets)
        | ast::GroupBy::Cube(sets)
        | ast::GroupBy::GroupingSets(sets) => {
            for group in sets {
                for expr in group {
                    bind_vars_expr(expr, ctx, shadowed)?;
                }
            }
        },
    }
    Ok(())
}

fn bind_vars_opt(
    expr: Option<&mut ast::Expr>,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    if let Some(expr) = expr {
        bind_vars_expr(expr, ctx, shadowed)?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive arm per Expr variant; mirrors crate::params substitution"
)]
fn bind_vars_expr(
    expr: &mut ast::Expr,
    ctx: &BindCtx,
    shadowed: &HashSet<String>,
) -> Result<(), Error> {
    match expr {
        ast::Expr::Column(name) => {
            // A bare identifier that names a table column in scope is a column reference, not a
            // variable — leave it for the analyzer to resolve (the column shadows the variable).
            if !shadowed.contains(name)
                && let Some(value) = ctx.env.get(name)
            {
                *expr = ast::Expr::Literal(value.clone());
            }
        },
        ast::Expr::Literal(_) | ast::Expr::QualifiedColumn { .. } | ast::Expr::Parameter(_) => {},
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            bind_vars_expr(left, ctx, shadowed)?;
            bind_vars_expr(right, ctx, shadowed)?;
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsBool { expr, .. }
        | ast::Expr::Cast { expr, .. } => bind_vars_expr(expr, ctx, shadowed)?,
        ast::Expr::InList { expr, list, .. } => {
            bind_vars_expr(expr, ctx, shadowed)?;
            for item in list {
                bind_vars_expr(item, ctx, shadowed)?;
            }
        },
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            bind_vars_expr(expr, ctx, shadowed)?;
            bind_vars_expr(low, ctx, shadowed)?;
            bind_vars_expr(high, ctx, shadowed)?;
        },
        ast::Expr::Like { expr, pattern, .. }
        | ast::Expr::SimilarTo { expr, pattern, .. }
        | ast::Expr::RegexMatch { expr, pattern, .. } => {
            bind_vars_expr(expr, ctx, shadowed)?;
            bind_vars_expr(pattern, ctx, shadowed)?;
        },
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => {
            bind_vars_opt(operand.as_deref_mut(), ctx, shadowed)?;
            for branch in branches {
                bind_vars_expr(&mut branch.when, ctx, shadowed)?;
                bind_vars_expr(&mut branch.then, ctx, shadowed)?;
            }
            bind_vars_opt(default.as_deref_mut(), ctx, shadowed)?;
        },
        ast::Expr::Coalesce(args)
        | ast::Expr::ScalarFunction { args, .. }
        | ast::Expr::FunctionCall { args, .. }
        | ast::Expr::SetReturning { args, .. } => {
            for arg in args {
                bind_vars_expr(arg, ctx, shadowed)?;
            }
        },
        ast::Expr::Aggregate { arg, filter, .. } => {
            bind_vars_opt(arg.as_deref_mut(), ctx, shadowed)?;
            bind_vars_opt(filter.as_deref_mut(), ctx, shadowed)?;
        },
        ast::Expr::Encrypt { value, key } | ast::Expr::Decrypt { value, key } => {
            bind_vars_expr(value, ctx, shadowed)?;
            bind_vars_expr(key, ctx, shadowed)?;
        },
        ast::Expr::ScalarSubquery(select)
        | ast::Expr::Exists {
            subquery: select, ..
        } => bind_vars_select(select, ctx, shadowed)?,
        ast::Expr::InSubquery { expr, subquery, .. }
        | ast::Expr::QuantifiedComparison { expr, subquery, .. } => {
            bind_vars_expr(expr, ctx, shadowed)?;
            bind_vars_select(subquery, ctx, shadowed)?;
        },
        ast::Expr::QuantifiedArray { expr, array, .. } => {
            bind_vars_expr(expr, ctx, shadowed)?;
            bind_vars_expr(array, ctx, shadowed)?;
        },
        ast::Expr::Row(items) | ast::Expr::ArrayLiteral(items) => {
            for item in items {
                bind_vars_expr(item, ctx, shadowed)?;
            }
        },
        ast::Expr::Subscript { base, index } => {
            bind_vars_expr(base, ctx, shadowed)?;
            bind_vars_expr(index, ctx, shadowed)?;
        },
        ast::Expr::ArraySlice { base, lower, upper } => {
            bind_vars_expr(base, ctx, shadowed)?;
            for bound in [lower, upper].into_iter().flatten() {
                bind_vars_expr(bound, ctx, shadowed)?;
            }
        },
        ast::Expr::WindowFunction(wf) => {
            for arg in &mut wf.args {
                bind_vars_expr(arg, ctx, shadowed)?;
            }
            for partition in &mut wf.partition {
                bind_vars_expr(partition, ctx, shadowed)?;
            }
            for order in &mut wf.order {
                bind_vars_expr(&mut order.expr, ctx, shadowed)?;
            }
        },
        ast::Expr::WithinGroup(wg) => {
            for arg in &mut wg.args {
                bind_vars_expr(arg, ctx, shadowed)?;
            }
            for order in &mut wg.order_by {
                bind_vars_expr(&mut order.expr, ctx, shadowed)?;
            }
        },
    }
    Ok(())
}
