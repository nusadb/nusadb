//! Stored procedures + `CALL`.
//!
//! A procedure is a named block of one or more `;`-separated SQL data statements, persisted in an
//! engine-scoped `nusadb_procedures` catalog (view/policy/trigger system-table pattern — no storage
//! spine change). Statements reference the call arguments positionally as `$1`..`$n`, reusing the
//! prepared-statement parameter machinery ([`crate::params`]); `CALL` binds the arguments and runs
//! each statement in sequence, re-entrantly, in the caller's transaction. A thread-local depth guard
//! bounds (possibly mutual) recursive calls.
//!
//! A NusaScript (`BEGIN … END`) body may also reference the `IN` parameters by name (the names are
//! stored in the catalog and seeded into the interpreter's variable environment), so `FOR i IN lo TO
//! hi` works as well as the positional `$1`..`$n`. A plain linear-SQL body keeps positional binding.
#![allow(clippy::wildcard_imports)]

use std::cell::Cell;

use super::*;
use crate::planner::{CallPlan, CreateProcedurePlan, DropProcedurePlan};

/// Engine-scoped system catalog of procedure definitions:
/// `(name, in_param_count, out_params, param_names, body)` text columns. `out_params` and
/// `param_names` are comma-separated lists of `OUT` / `IN` parameter names. Rows written before
/// `param_names` existed have four columns and load with no `IN` names (positional `$n` only).
// `pub(super)` so the rename guard can scan for bodies that name a column.
pub(super) const PROCEDURE_CATALOG: &str = "nusadb_procedures";

/// The current five-text-column schema of [`PROCEDURE_CATALOG`]:
/// `(name, in_param_count, out_params, param_names, body)`.
const PROCEDURE_CATALOG_SCHEMA: [ColumnType; 5] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// The legacy four-column schema (`name, in_param_count, out_params, body`) of rows written before
/// the `param_names` column was added — still readable, with no `IN` parameter names.
const PROCEDURE_CATALOG_SCHEMA_V4: [ColumnType; 4] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// Maximum nesting depth for cascading `CALL`s.
const MAX_CALL_DEPTH: usize = 64;

thread_local! {
    /// Current `CALL` nesting depth on this thread, used to bound recursion.
    static CALL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard incrementing the call depth on entry and decrementing on drop; refuses past the limit.
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Result<Self, Error> {
        CALL_DEPTH.with(|depth| {
            let current = depth.get();
            if current >= MAX_CALL_DEPTH {
                return Err(Error::ProcedureRecursionLimit {
                    limit: MAX_CALL_DEPTH,
                });
            }
            depth.set(current + 1);
            Ok(Self)
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        CALL_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// `CREATE [OR REPLACE] PROCEDURE ...`: persist the definition. Without `OR REPLACE`, a
/// same-named procedure is an error.
pub(super) fn run_create_procedure(
    plan: &CreateProcedurePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if !plan.or_replace && procedure_exists(engine, txn, &plan.name)? {
        return Err(Error::ProcedureExists {
            name: plan.name.clone(),
        });
    }
    let cat = ensure_procedure_catalog(engine, txn)?;
    delete_procedure_row(engine, txn, &plan.name)?;
    let row = [
        ast::Value::Text(plan.name.clone()),
        ast::Value::Text(plan.param_count.to_string()),
        ast::Value::Text(plan.out_params.join(",")),
        ast::Value::Text(super::function::encode_param_names(&plan.param_names)),
        ast::Value::Text(plan.body.clone()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &PROCEDURE_CATALOG_SCHEMA)?)?;
    Ok(ExecutionResult::ProcedureCreated)
}

/// `DROP PROCEDURE [IF EXISTS] name`.
pub(super) fn run_drop_procedure(
    plan: &DropProcedurePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let removed = delete_procedure_row(engine, txn, &plan.name)?;
    if !removed && !plan.if_exists {
        return Err(Error::ProcedureNotFound {
            name: plan.name.clone(),
        });
    }
    Ok(ExecutionResult::ProcedureDropped)
}

/// `CALL name(args)`: bind the arguments to the body's `$1..$n` and run each statement in
/// sequence in the caller's transaction, behind the recursion guard.
pub(super) fn run_call(
    plan: &CallPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let _guard = DepthGuard::enter()?;
    let Some(proc) = load_procedure(engine, txn, &plan.name)? else {
        return Err(Error::ProcedureNotFound {
            name: plan.name.clone(),
        });
    };
    if plan.args.len() != proc.param_count {
        return Err(Error::ProcedureArgCount {
            name: plan.name.clone(),
            expected: proc.param_count,
            found: plan.args.len(),
        });
    }
    if crate::parser::is_script(&proc.body) {
        // A NusaScript `BEGIN ... END` body: run the interpreter, then read back the OUT
        // parameters' final values from the variable environment.
        let block = crate::parser::parse_script(&proc.body)?;
        let env = super::script::run_block(&block, &plan.args, &proc.param_names, engine, txn)?;
        let values: Vec<ast::Value> = proc
            .out_params
            .iter()
            .map(|name| env.get(name).cloned().unwrap_or(ast::Value::Null))
            .collect();
        Ok(call_result(proc.out_params, values))
    } else {
        // A plain sequence of SQL statements: bind `$n` and run each in order. A linear body has no
        // variables, so OUT parameters come back NULL.
        for stmt in crate::parser::parse_statements(&proc.body)? {
            let bound = crate::params::substitute_values(stmt, &plan.args)?;
            let logical = crate::analyze(bound, &ExecCatalog::new(engine, txn))?;
            super::dispatch(crate::plan(logical), engine, txn)?;
        }
        let values = vec![ast::Value::Null; proc.out_params.len()];
        Ok(call_result(proc.out_params, values))
    }
}

/// Run an anonymous `DO` block: the same body grammar as a procedure, executed once with no
/// parameters. A NusaScript `BEGIN ... END` body runs through the interpreter; a plain statement
/// sequence runs each statement in order. Produces no result rows.
pub(super) fn run_do(
    body: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let _guard = DepthGuard::enter()?;
    if crate::parser::is_script(body) {
        let block = crate::parser::parse_script(body)?;
        super::script::run_block(&block, &[], &[], engine, txn)?;
    } else {
        for stmt in crate::parser::parse_statements(body)? {
            let logical = crate::analyze(stmt, &ExecCatalog::new(engine, txn))?;
            super::dispatch(crate::plan(logical), engine, txn)?;
        }
    }
    Ok(ExecutionResult::ProcedureCalled)
}

/// The result of a `CALL`: a one-row result of the `OUT` parameters, or `ProcedureCalled` when there
/// are none.
fn call_result(out_params: Vec<String>, values: Vec<ast::Value>) -> ExecutionResult {
    if out_params.is_empty() {
        ExecutionResult::ProcedureCalled
    } else {
        ExecutionResult::Rows {
            columns: out_params,
            rows: vec![values],
            command: RowsCommand::Select,
        }
    }
}

/// Look up the procedure catalog, creating it (lazily) if it does not exist yet.
fn ensure_procedure_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, PROCEDURE_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["name", "param_count", "out_params", "param_names", "body"]
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: PROCEDURE_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// A decoded procedure-catalog row.
struct DecodedProc {
    name: String,
    param_count: usize,
    out_params: Vec<String>,
    param_names: Vec<String>,
    body: String,
}

/// Decode one procedure-catalog row, accepting the current five-column shape
/// `(name, in_param_count, out_params, param_names, body)` and the legacy four-column shape written
/// before `param_names` existed (which loads with no `IN` names, positional `$n` only).
fn decode_procedure_row(bytes: &[u8]) -> Result<DecodedProc, Error> {
    if let Ok(row) = row::decode(bytes, &PROCEDURE_CATALOG_SCHEMA)
        && let [
            ast::Value::Text(name),
            ast::Value::Text(count),
            ast::Value::Text(outs),
            ast::Value::Text(names),
            ast::Value::Text(body),
        ] = row.as_slice()
    {
        return Ok(DecodedProc {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            out_params: super::function::decode_param_names(outs),
            param_names: super::function::decode_param_names(names),
            body: body.clone(),
        });
    }
    let row = row::decode(bytes, &PROCEDURE_CATALOG_SCHEMA_V4)?;
    if let [
        ast::Value::Text(name),
        ast::Value::Text(count),
        ast::Value::Text(outs),
        ast::Value::Text(body),
    ] = row.as_slice()
    {
        return Ok(DecodedProc {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            out_params: super::function::decode_param_names(outs),
            param_names: Vec::new(),
            body: body.clone(),
        });
    }
    Err(Error::Internal(
        "nusadb_procedures: unrecognised catalog row shape".to_owned(),
    ))
}

/// Every procedure's `(name, body)` — for `information_schema.routines`. Empty when the catalog
/// does not exist yet.
pub(super) fn all_procedures(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(String, String)>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PROCEDURE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let proc = decode_procedure_row(&bytes)?;
        out.push((proc.name, proc.body));
    }
    Ok(out)
}

/// Whether a procedure named `name` exists.
fn procedure_exists(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<bool, Error> {
    Ok(load_procedure(engine, txn, name)?.is_some())
}

/// Fetch the named procedure's decoded definition, or `None`.
fn load_procedure(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<DecodedProc>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PROCEDURE_CATALOG)? else {
        return Ok(None);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let proc = decode_procedure_row(&bytes)?;
        if proc.name == name {
            return Ok(Some(proc));
        }
    }
    Ok(None)
}

/// Remove the named procedure's row, returning whether one was deleted.
fn delete_procedure_row(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PROCEDURE_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        if decode_procedure_row(&bytes)?.name == name {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}
