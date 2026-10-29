//! The schema a view or a standalone sequence belongs to.
//!
//! Views and sequences are filed by one text key: the bare name in the default schema and
//! `schema.name` in any other. A default-schema object whose own name holds a dot
//! (`CREATE VIEW "a.b"`) therefore has the same key as an object `b` in a schema `a`, and the key
//! alone cannot say which one it is. This engine-scoped catalog, `nusadb_object_schemas`, records
//! the schema of every view and sequence whose key holds a dot, one row per object:
//! `(kind, key, schema)`. A key without a dot needs no row: it is always the default schema's.
//!
//! A dotted key with no row was filed by a release that predates the catalog. Its schema is
//! unknown, and every caller treats it as belonging to no schema it could drop: `DROP SCHEMA`
//! neither counts nor drops it.
//!
//! The catalog sits in the reserved `nusadb_*` namespace. A table of its name without its shape is
//! never read or written: it attributes nothing, and recording a dotted name is refused while it
//! is there. The catalog is created on first use; two transactions creating it at once conflict
//! (`40001`) and the second can retry.
#![allow(clippy::wildcard_imports)]

use std::collections::HashMap;

use super::*;

/// Engine-scoped catalog of the schemas of dotted view and sequence keys.
const CATALOG: &str = "nusadb_object_schemas";

/// The catalog's column names, in order; all `TEXT NOT NULL`.
const CATALOG_COLUMNS: [&str; 3] = ["kind", "name", "schema"];

/// `(kind, name, schema)`, all text.
const CATALOG_SCHEMA: [ColumnType; 3] = [ColumnType::Text; 3];

/// The kind of a view row.
pub(super) const VIEW: &str = "view";

/// The kind of a sequence row.
pub(super) const SEQUENCE: &str = "sequence";

/// What the catalog's name resolves to.
enum Found {
    Missing,
    Catalog(nusadb_core::TableId),
    Foreign,
}

fn find(engine: &dyn StorageEngine, txn: TxnId) -> Result<Found, Error> {
    Ok(match engine.lookup_table_as_of(txn, CATALOG)? {
        None => Found::Missing,
        Some(t)
            if t.columns.len() == CATALOG_COLUMNS.len()
                && t.columns
                    .iter()
                    .zip(CATALOG_COLUMNS)
                    .all(|(c, name)| c.name == name && c.ty == ColumnType::Text && !c.nullable) =>
        {
            Found::Catalog(t.id)
        },
        Some(_) => Found::Foreign,
    })
}

/// Every row: `(address, kind, key, schema)`.
fn rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(nusadb_core::Tid, String, String, String)>, Error> {
    let Found::Catalog(catalog) = find(engine, txn)? else {
        return Ok(Vec::new());
    };
    let mut scan = engine.scan(txn, catalog)?;
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        if let [
            ast::Value::Text(kind),
            ast::Value::Text(name),
            ast::Value::Text(schema),
        ] = row::decode(&bytes, &CATALOG_SCHEMA)?.as_slice()
        {
            out.push((tid, kind.clone(), name.clone(), schema.clone()));
        }
    }
    Ok(out)
}

/// The schema of every attributed `(kind, key)`.
pub(super) fn load(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<HashMap<(String, String), String>, Error> {
    Ok(rows(engine, txn)?
        .into_iter()
        .map(|(_, kind, name, schema)| ((kind, name), schema))
        .collect())
}

/// The schema `key` of `kind` belongs to: the default schema for a key without a dot, the
/// recorded one for a dotted key, and `None` for a dotted key with no row (filed by an earlier
/// release, so unknown).
pub(super) fn schema_of<'a>(
    attributed: &'a HashMap<(String, String), String>,
    kind: &str,
    key: &str,
) -> Option<&'a str> {
    if !key.contains('.') {
        return Some(nusadb_core::PUBLIC_SCHEMA);
    }
    attributed
        .get(&(kind.to_owned(), key.to_owned()))
        .map(String::as_str)
}

/// Record that the `kind` object filed as `key` was created in `schema`. A key without a dot
/// needs no row; any earlier row for the key is replaced.
///
/// # Errors
/// Storage errors; `42P07` when a table that is not the catalog holds its name.
pub(super) fn record(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: &str,
    key: &str,
    schema: &str,
) -> Result<(), Error> {
    forget(engine, txn, kind, key)?;
    if !key.contains('.') {
        return Ok(());
    }
    let catalog = match find(engine, txn)? {
        Found::Catalog(id) => id,
        Found::Foreign => {
            return Err(Error::Coded {
                message: format!(
                    "cannot record the schema of {kind} \"{key}\": a table named \"{CATALOG}\" \
                     exists and is not the catalog this engine keeps under that name; rename or \
                     drop it"
                ),
                sqlstate: "42P07",
            });
        },
        Found::Missing => {
            let columns = CATALOG_COLUMNS
                .into_iter()
                .map(|name| ColumnDef {
                    name: name.to_owned(),
                    ty: ColumnType::Text,
                    nullable: false,
                })
                .collect();
            engine.create_table(
                txn,
                &TableDef {
                    schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
                    name: CATALOG.to_owned(),
                    columns,
                },
            )?
        },
    };
    let row = [
        ast::Value::Text(kind.to_owned()),
        ast::Value::Text(key.to_owned()),
        ast::Value::Text(schema.to_owned()),
    ];
    engine.insert(txn, catalog, &row::encode(&row, &CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Forget the row of the `kind` object filed as `key`, when it is dropped.
///
/// # Errors
/// Storage errors.
pub(super) fn forget(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: &str,
    key: &str,
) -> Result<(), Error> {
    if !key.contains('.') {
        return Ok(());
    }
    let Found::Catalog(catalog) = find(engine, txn)? else {
        return Ok(());
    };
    for (tid, k, name, _) in rows(engine, txn)? {
        if k == kind && name == key {
            engine.delete(txn, catalog, tid)?;
        }
    }
    Ok(())
}
