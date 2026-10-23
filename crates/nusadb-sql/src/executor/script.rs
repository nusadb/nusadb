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
use crate::parser::{RaiseLevel, ScriptBlock, ScriptStmt, TriggerReturn};

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
        format!("__nusadb_exc_{n}")
    })
}

/// Control-flow outcome of executing a statement or block.
enum Flow {
    /// Continue with the next statement.
    Normal,
    /// `RETURN` was hit — stop the enclosing routine, carrying the returned value (`None` for a bare
    /// `RETURN`, as a procedure uses; `Some` for a function's `RETURN expr`).
    Return(Option<ast::Value>),
    /// A trigger function's `RETURN NEW` / `OLD` / `NULL`: stop, naming the row to proceed with.
    ReturnRow(TriggerReturn),
}

impl Flow {
    const fn is_normal(&self) -> bool {
        matches!(self, Self::Normal)
    }
}

/// The variable environment: declared name → current value, plus the column type of each variable
/// that stands for a table column (a trigger function's `NEW.col` fields).
///
/// The type matters only while the value is `NULL`. A non-null value is bound as a literal, and a
/// `NULL` of a typed variable is bound the way [`column_null`] binds it for that column type, so it
/// is never stricter than the column's non-null values (see [`Env::bound`]).
#[derive(Debug, Default)]
pub(super) struct Env {
    values: HashMap<String, ast::Value>,
    types: HashMap<String, ColumnType>,
}

impl Env {
    /// An empty environment.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// The current value of variable `name`.
    pub(super) fn get(&self, name: &str) -> Option<&ast::Value> {
        self.values.get(name)
    }

    /// Set variable `name` to `value`.
    pub(super) fn insert(&mut self, name: String, value: ast::Value) {
        self.values.insert(name, value);
    }

    /// Set variable `name`, which stands for a column of type `ty`, to `value`.
    pub(super) fn insert_typed(&mut self, name: String, value: ast::Value, ty: ColumnType) {
        // A column whose NULL stays bare (see `column_null`) binds like an untyped variable, so its
        // type need not be kept.
        if null_type(ty).is_some() {
            self.types.insert(name.clone(), ty);
        }
        self.values.insert(name, value);
    }

    /// The expression variable `name` is bound to, or `None` if no such variable is in scope: its
    /// value as a literal, except that a `NULL` in a typed variable is bound by [`column_null`].
    fn bound(&self, name: &str) -> Option<ast::Expr> {
        let value = self.values.get(name)?;
        Some(match (value, self.types.get(name)) {
            (ast::Value::Null, Some(&ty)) => column_null(ty),
            (value, _) => ast::Expr::Literal(value.clone()),
        })
    }
}

impl<const N: usize> From<[(String, ast::Value); N]> for Env {
    fn from(values: [(String, ast::Value); N]) -> Self {
        Self {
            values: HashMap::from(values),
            types: HashMap::new(),
        }
    }
}

/// The expression a `NULL` in a column of type `ty` is bound as. The rule is that a `NULL` must never
/// type-check more strictly than the column's non-null values, which are bound as literals.
///
/// A bare `NULL` has no type, so it cannot take part in an operator chosen by operand type
/// (`NULL + INTERVAL '1 day'`, `NULL - NULL`). For a column whose non-null value is always bound as a
/// literal of one fixed type, the `NULL` is bound as `CAST(NULL AS t)` with that same type `t`
/// ([`null_type`]). Every other column keeps a bare `NULL`, as before.
pub(super) fn column_null(ty: ColumnType) -> ast::Expr {
    let null = ast::Expr::Literal(ast::Value::Null);
    match null_type(ty) {
        Some(target) => ast::Expr::Cast {
            expr: Box::new(null),
            target,
            try_cast: false,
        },
        None => null,
    }
}

/// The type a `NULL` in a column of type `ty` is cast to, or `None` to bind it as a bare `NULL`.
///
/// A type is listed only when its non-null value is always bound as a literal of that type, from a
/// stored row and from the value an `INSERT` computed alike, and the cast `NULL` is accepted wherever
/// that literal is. The cast is to the type the literal itself has: an integer column (`SMALLINT`,
/// `INT`, `BIGINT`) gives `INT`, not its declared width; `REAL` gives `FLOAT` and `JSONB` gives
/// `JSON`; a `NUMERIC(p, s)` gives an unconstrained `NUMERIC`.
///
/// The rest stay bare. A `TEXT`/`VARCHAR`/`CHAR` value (and a composite, stored as text) is a string
/// literal that takes the type it meets (`'123'` into a `BIGINT` column, `= 1`). So, in an `INSERT`,
/// is the value of a type written as a string (`INET`, `CIDR`, ranges, arrays, `POINT`, `VECTOR`,
/// `TSVECTOR`, `TSQUERY`, `XML`, `BYTEA`, `BIT`, `BIT VARYING`), since `NEW` there carries the value before it is converted
/// to the column's type. A typed `NULL` for those would be stricter than their values.
const fn null_type(ty: ColumnType) -> Option<ColumnType> {
    match ty {
        ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => Some(ColumnType::Int),
        ColumnType::Float | ColumnType::Real => Some(ColumnType::Float),
        ColumnType::Json | ColumnType::Jsonb => Some(ColumnType::Json),
        ColumnType::Numeric { .. } => Some(ColumnType::Numeric {
            precision: 0,
            scale: 0,
        }),
        ColumnType::Bool
        | ColumnType::Date
        | ColumnType::Time
        | ColumnType::TimeTz
        | ColumnType::Timestamp
        | ColumnType::TimestampTz
        | ColumnType::Interval
        | ColumnType::Uuid
        | ColumnType::Enum
        | ColumnType::Macaddr
        | ColumnType::Macaddr8 => Some(ty),
        ColumnType::Text
        | ColumnType::VarChar(_)
        | ColumnType::Char(_)
        | ColumnType::Bytes
        | ColumnType::Bit(_)
        | ColumnType::VarBit(_)
        | ColumnType::Inet
        | ColumnType::Cidr
        | ColumnType::Range(_)
        | ColumnType::Array(_)
        | ColumnType::Vector(_)
        | ColumnType::Geometry(_)
        | ColumnType::Tsvector
        | ColumnType::Tsquery
        | ColumnType::Xml => None,
    }
}

/// Run a parsed NusaScript block with the call's `$n` arguments, returning the final
/// variable environment (so the caller can read back `OUT` parameter values).
pub(super) fn run_block(
    block: &ScriptBlock,
    params: &[ast::Value],
    param_names: &[String],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Env, Error> {
    let mut env = Env::new();
    // Seed the parameters by name so the body may reference them as `n` as well as positionally as
    // `$1`..`$n` (the same as a function body). A caller with no named parameters passes an empty
    // slice, leaving only the positional binding.
    for (name, value) in param_names.iter().zip(params) {
        env.insert(name.clone(), value.clone());
    }
    exec_block(block, &mut env, params, engine, txn)?;
    Ok(env)
}

/// Run a trigger function's block over a seeded environment (the trigger variables and the new
/// row's fields), returning the final environment and the row its `RETURN` named: `None` when the
/// body ended, or stopped at a bare `RETURN`, without naming one.
pub(super) fn run_trigger_block(
    block: &ScriptBlock,
    mut env: Env,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(Env, Option<TriggerReturn>), Error> {
    let returned = match exec_block(block, &mut env, &[], engine, txn)? {
        Flow::ReturnRow(row) => Some(row),
        Flow::Normal | Flow::Return(_) => None,
    };
    Ok((env, returned))
}

/// Run a parsed NusaScript block as a *function* body: execute it and yield the value of the
/// `RETURN expr` that stopped it, coerced to the declared return type. Reaching the end of the body —
/// or a bare `RETURN` with no value — without returning a value is an error: a function must return
/// one. (A procedure uses [`run_block`], which ignores the return value.)
pub(super) fn run_function_block(
    block: &ScriptBlock,
    params: &[ast::Value],
    param_names: &[String],
    engine: &dyn StorageEngine,
    txn: TxnId,
    return_ty: ColumnType,
) -> Result<ast::Value, Error> {
    let mut env = Env::new();
    // Seed the parameters by name so the body may reference them as `n` as well as positionally as
    // `$1`..`$n` (the positional form is bound from `params` by the embedded-SQL path).
    for (name, value) in param_names.iter().zip(params) {
        env.insert(name.clone(), value.clone());
    }
    match exec_block(block, &mut env, params, engine, txn)? {
        Flow::Return(Some(value)) => super::eval::cast_value(value, return_ty),
        Flow::Return(None) | Flow::ReturnRow(_) | Flow::Normal => Err(Error::Coded {
            message: "control reached end of function without RETURN".to_owned(),
            sqlstate: "2F005",
        }),
    }
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
    super::deferred::savepoint(engine, txn, &savepoint)?;
    match exec_stmts(&block.body, env, params, engine, txn) {
        Ok(flow) => {
            super::deferred::release_savepoint(engine, txn, &savepoint)?;
            Ok(flow)
        },
        Err(Error::Cancelled) => Err(Error::Cancelled),
        Err(_caught) => {
            // Undo the body's partial writes, then run the handler in their place.
            super::deferred::rollback_to(engine, txn, &savepoint)?;
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
        let flow = exec_one(stmt, env, params, engine, txn)?;
        if !flow.is_normal() {
            return Ok(flow);
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
                    return Err(Error::LimitExceeded(
                        "NusaScript WHILE loop exceeded the iteration limit".to_owned(),
                    ));
                }
                let flow = exec_stmts(body, env, params, engine, txn)?;
                if !flow.is_normal() {
                    return Ok(flow);
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
                    return Err(Error::LimitExceeded(
                        "NusaScript FOR loop exceeded the iteration limit".to_owned(),
                    ));
                }
                env.insert(var.clone(), ast::Value::Int(i));
                let flow = exec_stmts(body, env, params, engine, txn)?;
                if !flow.is_normal() {
                    return Ok(flow);
                }
            }
            Ok(Flow::Normal)
        },
        ScriptStmt::Perform(expr) => {
            // Evaluated for its side effects (nextval(), a function call); the value is discarded.
            eval_value(expr, env, params, engine, txn)?;
            Ok(Flow::Normal)
        },
        ScriptStmt::Raise(expr) => {
            let value = eval_value(expr, env, params, engine, txn)?;
            Err(Error::Raised(message_text(&value)))
        },
        ScriptStmt::RaiseMessage {
            level,
            format,
            args,
        } => raise_message(*level, format, args, env, params, engine, txn),
        ScriptStmt::Return(expr) => {
            let value = match expr {
                Some(expr) => Some(eval_value(expr, env, params, engine, txn)?),
                None => None,
            };
            Ok(Flow::Return(value))
        },
        ScriptStmt::ReturnRow(row) => Ok(Flow::ReturnRow(*row)),
        ScriptStmt::Block(block) => exec_block(block, env, params, engine, txn),
        ScriptStmt::Sql(sql) => {
            let bound = bind((**sql).clone(), env, params, engine, txn)?;
            let logical = crate::analyze(bound, &ExecCatalog::new(engine, txn))?;
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
    let logical = crate::analyze(bound, &ExecCatalog::new(engine, txn))?;
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
        other => Err(Error::InvalidStatement(format!(
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

/// `RAISE level 'format', args`: an error for `EXCEPTION`, nothing for `LOG`/`DEBUG`, a notice
/// for the rest.
fn raise_message(
    level: RaiseLevel,
    format: &str,
    args: &[ast::Expr],
    env: &Env,
    params: &[ast::Value],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Flow, Error> {
    let values = args
        .iter()
        .map(|arg| eval_value(arg, env, params, engine, txn))
        .collect::<Result<Vec<_>, Error>>()?;
    let message = format_raise(format, &values)?;
    let (severity, code) = match level {
        RaiseLevel::Exception => return Err(Error::Raised(message)),
        // Server-side levels: nothing reaches the client.
        RaiseLevel::Debug | RaiseLevel::Log => return Ok(Flow::Normal),
        RaiseLevel::Info => (crate::notice::Severity::Info, "00000"),
        RaiseLevel::Notice => (crate::notice::Severity::Notice, "00000"),
        RaiseLevel::Warning => (crate::notice::Severity::Warning, "01000"),
    };
    crate::notice::raise(severity, code, message);
    Ok(Flow::Normal)
}

/// Fill a `RAISE` format: each `%` takes the next value (`NULL` shows as `<NULL>`), `%%` is a
/// literal `%`. The number of `%` must match the number of values.
fn format_raise(format: &str, values: &[ast::Value]) -> Result<String, Error> {
    let mut out = String::with_capacity(format.len());
    let mut next = values.iter();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
        } else if chars.peek() == Some(&'%') {
            chars.next();
            out.push('%');
        } else {
            let value = next.next().ok_or_else(|| {
                Error::InvalidStatement("too few parameters specified for RAISE".to_owned())
            })?;
            match value {
                ast::Value::Null => out.push_str("<NULL>"),
                other => out.push_str(&message_text(other)),
            }
        }
    }
    if next.next().is_some() {
        return Err(Error::InvalidStatement(
            "too many parameters specified for RAISE".to_owned(),
        ));
    }
    Ok(out)
}

/// Render a value as the text of a `RAISE` message.
fn message_text(value: &ast::Value) -> String {
    match value {
        ast::Value::Text(s) => s.clone(),
        ast::Value::Null => "NULL".to_owned(),
        ast::Value::Bool(b) => b.to_string(),
        ast::Value::Int(n) => n.to_string(),
        // Canonical float rendering (exponent form + `Infinity`/`NaN`), same as every output path.
        ast::Value::Float(f) => crate::display::format_float(*f),
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
                && let Some(bound) = ctx.env.bound(name)
            {
                *expr = bound;
            }
        },
        ast::Expr::Literal(_) | ast::Expr::QualifiedColumn { .. } | ast::Expr::Parameter(_) => {},
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            bind_vars_expr(left, ctx, shadowed)?;
            bind_vars_expr(right, ctx, shadowed)?;
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsJson { operand: expr, .. }
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
        ast::Expr::Overlaps { s1, e1, s2, e2 } => {
            bind_vars_expr(s1, ctx, shadowed)?;
            bind_vars_expr(e1, ctx, shadowed)?;
            bind_vars_expr(s2, ctx, shadowed)?;
            bind_vars_expr(e2, ctx, shadowed)?;
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
        ast::Expr::FieldAccess { base: inner, .. }
        | ast::Expr::AllFields(inner)
        | ast::Expr::CastNamed { expr: inner, .. } => {
            bind_vars_expr(inner, ctx, shadowed)?;
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
            if let Some(filter) = wf.filter.as_mut() {
                bind_vars_expr(filter, ctx, shadowed)?;
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
