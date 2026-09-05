//! SQL scalar functions — `CREATE`/`DROP FUNCTION` persistence + lookup.
//!
//! A function is persisted in an engine-scoped `nusadb_functions` catalog (`(name, param_count,
//! param_names, body, language, return_type)`), mirroring the view/procedure system-table pattern — no
//! storage-spine change. A SQL function's body is a `SELECT <expr>` that the analyzer inlines at the
//! call site (substituting the call's arguments for the body's `$1..$n` **or** the declared parameter
//! names), so it composes exactly like a built-in; its result type is the inlined body's type. A
//! NusaScript function's `BEGIN … END` body (under `language = "nusascript"`) is run by the
//! interpreter, which yields the value of its `RETURN expr` coerced to the stored `return_type`.
//!
//! The catalog schema grew over time — `param_names` (named-parameter calls), then `language`
//! (SQL vs NusaScript), then `return_type` — so the current row has six text columns.
//! [`decode_function_row`] decodes the current shape and every older shape (five without
//! `return_type`, four without `language`, three without `param_names`), each older column defaulting
//! sensibly (no names → positional-only, no language → `SQL`, no return type → `TEXT`), so the change
//! needs no migration.

#![allow(clippy::wildcard_imports)]

use super::*;
use crate::FunctionDef;
use crate::planner::{CreateFunctionPlan, DropFunctionPlan};

/// Engine-scoped system catalog of functions: `(name, param_count, param_names, body, language,
/// return_type)` text columns.
// `pub(super)` so the rename guard can scan for bodies that name a column.
pub(super) const FUNCTION_CATALOG: &str = "nusadb_functions";

/// The current six-text-column schema of [`FUNCTION_CATALOG`]: `(name, param_count, param_names,
/// body, language, return_type)` — `return_type` was appended after `language`.
const FUNCTION_CATALOG_SCHEMA: [ColumnType; 6] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// The five-column schema (`…, body, language`) of rows written before `return_type` was tracked —
/// decoded as a fallback (the return type defaults to `TEXT`) so the catalog needs no migration.
const FUNCTION_CATALOG_SCHEMA_V5: [ColumnType; 5] = [
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

/// A decoded function-catalog row.
pub(super) struct DecodedFunction {
    pub(super) name: String,
    param_count: usize,
    param_names: Vec<String>,
    pub(super) language: ast::FunctionLanguage,
    pub(super) return_type: ColumnType,
    pub(super) body: String,
}

/// Every function-catalog row, decoded — for `information_schema.routines`. Empty when the catalog
/// does not exist yet.
pub(super) fn all_functions(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<DecodedFunction>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, FUNCTION_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        out.push(decode_function_row(&bytes)?);
    }
    Ok(out)
}

/// Parse a stored return-type string back to a [`ColumnType`], defaulting to `TEXT` for an absent
/// (fallback-decoded older row) or unrecognisable value — a return type written by this engine via
/// [`super::ddl::type_name`] always parses, so the default only guards older rows and corruption.
fn decode_return_type(text: &str) -> ColumnType {
    crate::parser::parse_column_type(text).unwrap_or(ColumnType::Text)
}

/// Decode one catalog row, accepting the current six-column shape and the three older shapes (five
/// without `return_type`, four without `language`, three without `param_names`) so rows written by any
/// prior version still load. An older row has no return type (defaults to `TEXT`) and, before the
/// `language` column, is an `SQL` function.
fn decode_function_row(bytes: &[u8]) -> Result<DecodedFunction, Error> {
    // Current six-column row: `(name, param_count, param_names, body, language, return_type)`.
    if let Ok(row) = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA)
        && let [
            ast::Value::Text(name),
            ast::Value::Text(count),
            ast::Value::Text(names),
            ast::Value::Text(body),
            ast::Value::Text(language),
            ast::Value::Text(return_type),
        ] = row.as_slice()
    {
        return Ok(DecodedFunction {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            param_names: decode_param_names(names),
            language: decode_language(language),
            return_type: decode_return_type(return_type),
            body: body.clone(),
        });
    }
    // Five-column row (no `return_type`): `(name, param_count, param_names, body, language)`.
    if let Ok(row) = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA_V5)
        && let [
            ast::Value::Text(name),
            ast::Value::Text(count),
            ast::Value::Text(names),
            ast::Value::Text(body),
            ast::Value::Text(language),
        ] = row.as_slice()
    {
        return Ok(DecodedFunction {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            param_names: decode_param_names(names),
            language: decode_language(language),
            return_type: ColumnType::Text,
            body: body.clone(),
        });
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
        return Ok(DecodedFunction {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            param_names: decode_param_names(names),
            language: ast::FunctionLanguage::Sql,
            return_type: ColumnType::Text,
            body: body.clone(),
        });
    }
    // Legacy three-column row: `(name, param_count, body)` with no parameter names.
    let row = row::decode(bytes, &FUNCTION_CATALOG_SCHEMA_LEGACY)?;
    if let [
        ast::Value::Text(name),
        ast::Value::Text(count),
        ast::Value::Text(body),
    ] = row.as_slice()
    {
        return Ok(DecodedFunction {
            name: name.clone(),
            param_count: count.parse::<usize>().unwrap_or(0),
            param_names: Vec::new(),
            language: ast::FunctionLanguage::Sql,
            return_type: ColumnType::Text,
            body: body.clone(),
        });
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
        ast::Value::Text(super::ddl::type_name(plan.return_type)),
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
        let f = decode_function_row(&bytes)?;
        if f.name == name {
            return Ok(Some(FunctionDef {
                param_count: f.param_count,
                param_names: f.param_names,
                language: f.language,
                return_type: f.return_type,
                body: f.body,
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
    let columns = [
        "name",
        "param_count",
        "param_names",
        "body",
        "language",
        "return_type",
    ]
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
        if decode_function_row(&bytes)?.name == name {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> ast::Value {
        ast::Value::Text(s.to_owned())
    }

    /// A row written by any prior catalog shape still decodes, with the columns that shape lacked
    /// taking their documented defaults. This pins the on-disk backward-compat contract so a future
    /// schema change cannot silently break loading old function definitions.
    #[test]
    fn decode_function_row_accepts_every_historical_shape() {
        // Current six-column shape: all fields present.
        let six = row::encode(
            &[
                text("f6"),
                text("2"),
                text("a,b"),
                text("SELECT $1 + $2"),
                text("nusascript"),
                text("BIGINT"),
            ],
            &FUNCTION_CATALOG_SCHEMA,
        )
        .expect("encode six-column row");
        let d = decode_function_row(&six).expect("decode six-column row");
        assert_eq!(d.name, "f6");
        assert_eq!(d.param_count, 2);
        assert_eq!(d.param_names, vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(d.language, ast::FunctionLanguage::NusaScript);
        assert_eq!(d.return_type, ColumnType::BigInt);
        assert_eq!(d.body, "SELECT $1 + $2");

        // Five-column shape (no `return_type`): defaults to TEXT, language preserved.
        let five = row::encode(
            &[
                text("f5"),
                text("1"),
                text("x"),
                text("SELECT $1"),
                text("nusascript"),
            ],
            &FUNCTION_CATALOG_SCHEMA_V5,
        )
        .expect("encode five-column row");
        let d = decode_function_row(&five).expect("decode five-column row");
        assert_eq!(d.name, "f5");
        assert_eq!(d.language, ast::FunctionLanguage::NusaScript);
        assert_eq!(d.return_type, ColumnType::Text);

        // Four-column shape (no `language`): an SQL function, return type TEXT.
        let four = row::encode(
            &[text("f4"), text("1"), text("x"), text("SELECT $1")],
            &FUNCTION_CATALOG_SCHEMA_V4,
        )
        .expect("encode four-column row");
        let d = decode_function_row(&four).expect("decode four-column row");
        assert_eq!(d.name, "f4");
        assert_eq!(d.param_names, vec!["x".to_owned()]);
        assert_eq!(d.language, ast::FunctionLanguage::Sql);
        assert_eq!(d.return_type, ColumnType::Text);

        // Legacy three-column shape (no `param_names`): positional-only, SQL, TEXT.
        let three = row::encode(
            &[text("f3"), text("1"), text("SELECT $1")],
            &FUNCTION_CATALOG_SCHEMA_LEGACY,
        )
        .expect("encode three-column row");
        let d = decode_function_row(&three).expect("decode three-column row");
        assert_eq!(d.name, "f3");
        assert_eq!(d.param_count, 1);
        assert!(d.param_names.is_empty());
        assert_eq!(d.language, ast::FunctionLanguage::Sql);
        assert_eq!(d.return_type, ColumnType::Text);
        assert_eq!(d.body, "SELECT $1");
    }
}
