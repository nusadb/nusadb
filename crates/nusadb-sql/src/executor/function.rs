//! SQL scalar functions — `CREATE`/`DROP FUNCTION` persistence + lookup.
//!
//! A SQL function is a named `SELECT <expr>` body, persisted in an engine-scoped `nusadb_functions`
//! catalog (`(name, param_count, param_names, body, language)`), mirroring the view/procedure
//! system-table pattern — no storage-spine change. The analyzer inlines a call to a SQL function
//! (substituting the call's arguments for the body's `$1..$n` **or** the declared parameter names), so
//! it composes exactly like a built-in; the declared `RETURNS` type is accepted but not
//! stored/enforced, and the result type is the inlined body's type. A NusaScript function stores its
//! `BEGIN … END` body under `language = "nusascript"` (running such a function from a query is a
//! following increment).
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
// `pub(super)` so the rename guard can scan for bodies that name a column.
pub(super) const FUNCTION_CATALOG: &str = "nusadb_functions";

/// The current five-text-column schema of [`FUNCTION_CATALOG`]: `(name, param_count, param_names,
/// body, language)` — `language` was appended after `body`.
const FUNCTION_CATALOG_SCHEMA: [ColumnType; 5] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// The four-column schema (`name, param_count, param_names, body`) of rows written before `language`
/// was tracked — decoded as a fallback (the language is taken to be `SQL`) so the catalog needs no
/// migration.
const FUNCTION_CATALOG_SCHEMA_V4: [ColumnType; 4] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// The legacy three-column schema (`name, param_count, body`) of rows written before `param_names`
/// existed — decoded as a fallback so the catalog needs no migration.
const FUNCTION_CATALOG_SCHEMA_LEGACY: [ColumnType; 3] =
    [ColumnType::Text, ColumnType::Text, ColumnType::Text];

/// Encode a function's implementation language for the catalog's `language` column.
const fn encode_language(language: ast::FunctionLanguage) -> &'static str {
    match language {
        ast::FunctionLanguage::Sql => "sql",
        ast::FunctionLanguage::NusaScript => "nusascript",
    }
}

/// Decode the catalog's `language` column. Any value other than `nusascript` (including the empty
/// string of a fallback-decoded older row) is `SQL`.
const fn decode_language(text: &str) -> ast::FunctionLanguage {
    if text.eq_ignore_ascii_case("nusascript") {
        ast::FunctionLanguage::NusaScript
    } else {
        ast::FunctionLanguage::Sql
    }
}

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

/// Decode one catalog row to `(name, param_count, param_names, language, body)`, accepting the current
/// five-column shape and the two older shapes (four columns without `language`, three without
/// `param_names`) so rows written by any prior version still load — an older row is `SQL`.
fn decode_function_row(
    bytes: &[u8],
) -> Result<(String, usize, Vec<String>, ast::FunctionLanguage, String), Error> {
    // Current five-column row: `(name, param_count, param_names, body, language)`.
    if let Ok(row) = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA)
        && let [
            ast::Value::Text(name),
            ast::Value::Text(count),
            ast::Value::Text(names),
            ast::Value::Text(body),
            ast::Value::Text(language),
        ] = row.as_slice()
    {
        return Ok((
            name.clone(),
            count.parse::<usize>().unwrap_or(0),
            decode_param_names(names),
            decode_language(language),
            body.clone(),
        ));
    }
    // Four-column row (no `language`): `(name, param_count, param_names, body)` — an `SQL` function.
    if let Ok(row) = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA_V4)
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
            ast::FunctionLanguage::Sql,
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
            ast::FunctionLanguage::Sql,
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
        ast::Value::Text(encode_language(plan.language).to_owned()),
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
        let (n, param_count, param_names, language, body) = decode_function_row(&bytes)?;
        if n == name {
            return Ok(Some(FunctionDef {
                param_count,
                param_names,
                language,
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
    let columns = ["name", "param_count", "param_names", "body", "language"]
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
