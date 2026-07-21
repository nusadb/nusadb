//! Stored procedures + `CALL`.
//!
//! A procedure is a named block of one or more `;`-separated SQL data statements, persisted in an
//! engine-scoped `nusadb_procedures` catalog (view/policy/trigger system-table pattern — no storage
//! spine change). Statements reference the call arguments positionally as `$1`..`$n`, reusing the
//! prepared-statement parameter machinery ([`crate::params`]); `CALL` binds the arguments and runs
//! each statement in sequence, re-entrantly, in the caller's transaction. A thread-local depth guard
//! bounds (possibly mutual) recursive calls.
//!
//! Named parameters, `OUT` parameters, control flow (NusaScript), and `CREATE FUNCTION` are honest
//! follow-ups; this is the linear-SQL-body core.
#![allow(clippy::wildcard_imports)]

use std::cell::Cell;

use super::*;
use crate::planner::{CallPlan, CreateProcedurePlan, DropProcedurePlan};

/// Engine-scoped system catalog of procedure definitions: `(name, in_param_count, out_params, body)`
/// text columns. `out_params` is a comma-separated list of `OUT` parameter names.
const PROCEDURE_CATALOG: &str = "nusadb_procedures";

/// The four-text-column schema of [`PROCEDURE_CATALOG`].
const PROCEDURE_CATALOG_SCHEMA: [ColumnType; 4] = [
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
    let Some((param_count, out_params, body)) = load_procedure(engine, txn, &plan.name)? else {
        return Err(Error::ProcedureNotFound {
            name: plan.name.clone(),
        });
    };
    if plan.args.len() != param_count {
        return Err(Error::ProcedureArgCount {
            name: plan.name.clone(),
            expected: param_count,
            found: plan.args.len(),
        });
    }
    if crate::parser::is_script(&body) {
        // A NusaScript `BEGIN ... END` body: run the interpreter, then read back the OUT
        // parameters' final values from the variable environment.
        let block = crate::parser::parse_script(&body)?;
        let env = super::script::run_block(&block, &plan.args, engine, txn)?;
        let values: Vec<ast::Value> = out_params
            .iter()
            .map(|name| env.get(name).cloned().unwrap_or(ast::Value::Null))
            .collect();
        Ok(call_result(out_params, values))
    } else {
        // A plain sequence of SQL statements: bind `$n` and run each in order. A linear body has no
        // variables, so OUT parameters come back NULL.
        for stmt in crate::parser::parse_statements(&body)? {
            let bound = crate::params::substitute_values(stmt, &plan.args)?;
            let logical = crate::analyze(bound, &ExecCatalog { engine, txn })?;
            super::dispatch(crate::plan(logical), engine, txn)?;
        }
        let values = vec![ast::Value::Null; out_params.len()];
        Ok(call_result(out_params, values))
    }
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
    let columns = ["name", "param_count", "out_params", "body"]
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

/// Whether a procedure named `name` exists.
fn procedure_exists(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<bool, Error> {
    Ok(load_procedure(engine, txn, name)?.is_some())
}

/// Fetch `(in_param_count, out_param_names, body)` for the named procedure, or `None`.
fn load_procedure(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<(usize, Vec<String>, String)>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PROCEDURE_CATALOG)? else {
        return Ok(None);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PROCEDURE_CATALOG_SCHEMA)?;
        if let [
            ast::Value::Text(n),
            ast::Value::Text(count),
            ast::Value::Text(outs),
            ast::Value::Text(body),
        ] = row.as_slice()
            && n == name
        {
            let param_count = count.parse::<usize>().unwrap_or(0);
            let out_params = if outs.is_empty() {
                Vec::new()
            } else {
                outs.split(',').map(str::to_owned).collect()
            };
            return Ok(Some((param_count, out_params, body.clone())));
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
        let row = row::decode(&bytes, &PROCEDURE_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == name) {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}
