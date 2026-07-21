//! SQL scalar functions — `CREATE`/`DROP FUNCTION` persistence + lookup.
//!
//! A SQL function is a named `SELECT <expr>` body, persisted in an engine-scoped `nusadb_functions`
//! catalog (`(name, param_count, param_names, body)`), mirroring the view/procedure system-table
//! pattern — no storage-spine change. The analyzer inlines a call to it (substituting the call's
//! arguments for the body's `$1..$n` **or** the declared parameter names), so a SQL function composes
//! exactly like a built-in. The declared `RETURNS` type is accepted but not stored/enforced; the
//! result type is the inlined body's type.
//!
//! The `param_names` column was added for named-parameter calls. Rows written before it
//! existed have only three columns; [`decode_function_row`] decodes both shapes (an old row yields no
//! names — positional-only), so the change needs no migration.

#![allow(clippy::wildcard_imports)]

use super::*;
use crate::FunctionDef;
use crate::planner::{CreateFunctionPlan, DropFunctionPlan};

/// Engine-scoped system catalog of SQL functions: `(name, param_count, param_names, body)` text
/// columns.
const FUNCTION_CATALOG: &str = "nusadb_functions";

/// The current four-text-column schema of [`FUNCTION_CATALOG`] (added `param_names`).
const FUNCTION_CATALOG_SCHEMA: [ColumnType; 4] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// The legacy three-column schema (`name, param_count, body`) of rows written before `param_names`
/// existed — decoded as a fallback so the catalog needs no migration.
const FUNCTION_CATALOG_SCHEMA_LEGACY: [ColumnType; 3] =
    [ColumnType::Text, ColumnType::Text, ColumnType::Text];

/// Join parameter names for the `param_names` catalog column. Names are lowercase-folded identifiers,
/// so a comma never appears inside one; an empty list stores as the empty string.
fn encode_param_names(names: &[String]) -> String {
    names.join(",")
}

/// Split the `param_names` catalog column back into names. The empty string is no names (a
/// zero-parameter function, or a legacy row).
fn decode_param_names(text: &str) -> Vec<String> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.split(',').map(str::to_owned).collect()
    }
}

/// Decode one catalog row to `(name, param_count, param_names, body)`, accepting both the current
/// four-column shape and the legacy three-column shape (no `param_names`) so old rows still load.
fn decode_function_row(bytes: &[u8]) -> Result<(String, usize, Vec<String>, String), Error> {
    if let Ok(row) = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA)
        && let [
            ast::Value::Text(name),
            ast::Value::Text(count),
            ast::Value::Text(names),
            ast::Value::Text(body),
        ] = row.as_slice()
    {
        return Ok((
            name.clone(),
            count.parse::<usize>().unwrap_or(0),
            decode_param_names(names),
            body.clone(),
        ));
    }
    // Legacy three-column row: `(name, param_count, body)` with no parameter names.
    let row = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA_LEGACY)?;
    if let [
        ast::Value::Text(name),
        ast::Value::Text(count),
        ast::Value::Text(body),
    ] = row.as_slice()
    {
        return Ok((
            name.clone(),
            count.parse::<usize>().unwrap_or(0),
            Vec::new(),
            body.clone(),
        ));
    }
    Err(Error::MalformedTuple { offset: 0 })
}

/// `CREATE [OR REPLACE] FUNCTION ...`: persist the definition. Without `OR REPLACE`, a
/// same-named function is an error.
pub(super) fn run_create_function(
    plan: &CreateFunctionPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if !plan.or_replace && lookup_function_definition(engine, txn, &plan.name)?.is_some() {
        return Err(Error::FunctionExists {
            name: plan.name.clone(),
        });
    }
    let cat = ensure_function_catalog(engine, txn)?;
    delete_function_row(engine, txn, &plan.name)?;
    let row = [
        ast::Value::Text(plan.name.clone()),
        ast::Value::Text(plan.param_count.to_string()),
        ast::Value::Text(encode_param_names(&plan.param_names)),
        ast::Value::Text(plan.body.clone()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &FUNCTION_CATALOG_SCHEMA)?)?;
    Ok(ExecutionResult::FunctionCreated)
}

/// `DROP FUNCTION [IF EXISTS] name`.
pub(super) fn run_drop_function(
    plan: &DropFunctionPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let removed = delete_function_row(engine, txn, &plan.name)?;
    if !removed && !plan.if_exists {
        return Err(Error::FunctionNotFound {
            name: plan.name.clone(),
        });
    }
    Ok(ExecutionResult::FunctionDropped)
}

/// The definition of SQL function `name` under `txn`'s snapshot, for the analyzer to inline.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn lookup_function_definition(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<FunctionDef>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, FUNCTION_CATALOG)? else {
        return Ok(None);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let (n, param_count, param_names, body) = decode_function_row(&bytes)?;
        if n == name {
            return Ok(Some(FunctionDef {
                param_count,
                param_names,
                body,
            }));
        }
    }
    Ok(None)
}

/// Look up the function catalog, creating it (lazily) if it does not exist yet.
fn ensure_function_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, FUNCTION_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["name", "param_count", "param_names", "body"]
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: FUNCTION_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Remove the named function's row, returning whether one was deleted.
fn delete_function_row(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, FUNCTION_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let (n, ..) = decode_function_row(&bytes)?;
        if n == name {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}
