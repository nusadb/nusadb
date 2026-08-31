//! Range table partitioning — metadata catalog + `INSERT` routing.
//!
//! `CREATE TABLE m (...) PARTITION BY RANGE (col)` records a *parent* row here; `CREATE TABLE p
//! PARTITION OF m FOR VALUES FROM (lo) TO (hi)` records a *partition* row (and a child→parent edge in
//! the inheritance catalog, so a query on the parent expands over its partitions exactly as an
//! inheritance parent does). The parent table holds no rows of its own: an `INSERT` into it is routed
//! to the partition whose `[lo, hi)` bound contains the key. Mirrors the trigger/function catalog
//! pattern — no storage-spine change.
#![allow(clippy::wildcard_imports)]

use std::cmp::Ordering;

use super::*;

/// Engine-scoped system catalog of partition metadata.
pub(super) const PARTITION_CATALOG: &str = "nusadb_partitions";

/// Five-text-column schema: `(role, table, aux, lo, hi)`.
/// - a parent row is `("parent", <parent table>, <key column>, "", "")`;
/// - a partition row is `("part", <partition table>, <parent table>, <lo hex>, <hi hex>)`, where the
///   bounds are hex of the key value encoded at the key column's type.
const PARTITION_CATALOG_SCHEMA: [ColumnType; 5] = [ColumnType::Text; 5];

/// A resolved partition of a parent: its table name and `[lo, hi)` key bound (decoded to values).
pub(super) struct PartitionEntry {
    pub table: String,
    pub lo: ast::Value,
    pub hi: ast::Value,
}

/// Record a `PARTITION BY RANGE (key)` parent.
pub(super) fn record_parent(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_column: &str,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let row = [
        text("parent"),
        text(parent),
        text(key_column),
        text(""),
        text(""),
    ];
    engine.insert(txn, cat, &row::encode(&row, &PARTITION_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Record a partition of `parent` with a `[lo, hi)` bound already coerced to the key type.
pub(super) fn record_partition(
    engine: &dyn StorageEngine,
    txn: TxnId,
    partition: &str,
    parent: &str,
    key_ty: ColumnType,
    lo: &ast::Value,
    hi: &ast::Value,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let row = [
        text("part"),
        text(partition),
        text(parent),
        text(&encode_bound(lo, key_ty)?),
        text(&encode_bound(hi, key_ty)?),
    ];
    engine.insert(txn, cat, &row::encode(&row, &PARTITION_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Whether any partition metadata exists at all — the cheap gate that lets a database with no
/// partitioning skip the per-`INSERT` catalog probes. One catalog lookup + first-row read.
pub(super) fn has_any(engine: &dyn StorageEngine, txn: TxnId) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? else {
        return Ok(false);
    };
    Ok(engine.scan(txn, cat.id)?.try_next()?.is_some())
}

/// The partition key column of `table` if it is a partitioned parent, else `None`.
pub(super) fn parent_key_column(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<Option<String>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "parent" && field(&row, 1) == table {
            return Ok(Some(field(&row, 2).to_owned()));
        }
    }
    Ok(None)
}

/// The partitioned parent of `table` if it is a partition, else `None` (no bound decode needed).
pub(super) fn partition_parent(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<Option<String>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 1) == table {
            return Ok(Some(field(&row, 2).to_owned()));
        }
    }
    Ok(None)
}

/// The `(parent, lo, hi)` bound of `table` if it is a partition, decoded at `key_ty`.
pub(super) fn partition_bound(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    key_ty: ColumnType,
) -> Result<Option<(String, ast::Value, ast::Value)>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 1) == table {
            let lo = decode_bound(field(&row, 3), key_ty)?;
            let hi = decode_bound(field(&row, 4), key_ty)?;
            return Ok(Some((field(&row, 2).to_owned(), lo, hi)));
        }
    }
    Ok(None)
}

/// Every partition of `parent`, decoded at `key_ty`, sorted by lower bound.
pub(super) fn partitions_of(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_ty: ColumnType,
) -> Result<Vec<PartitionEntry>, Error> {
    let mut out = Vec::new();
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 2) == parent {
            out.push(PartitionEntry {
                table: field(&row, 1).to_owned(),
                lo: decode_bound(field(&row, 3), key_ty)?,
                hi: decode_bound(field(&row, 4), key_ty)?,
            });
        }
    }
    out.sort_by(|a, b| super::eval::compare(&a.lo, &b.lo));
    Ok(out)
}

/// Remove every partition-catalog row naming `table` (as parent or partition), on DROP.
pub(super) fn remove_for(engine: &dyn StorageEngine, txn: TxnId, table: &str) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PARTITION_CATALOG_SCHEMA)?;
        if field(&row, 1) == table || (field(&row, 0) == "part" && field(&row, 2) == table) {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// Whether `key` falls in `[lo, hi)` — the range-partition containment test.
pub(super) fn contains(key: &ast::Value, lo: &ast::Value, hi: &ast::Value) -> bool {
    super::eval::compare(key, lo) != Ordering::Less
        && super::eval::compare(key, hi) == Ordering::Less
}

// === helpers ==============================================================

fn text(s: &str) -> ast::Value {
    ast::Value::Text(s.to_owned())
}

fn field(row: &[ast::Value], i: usize) -> &str {
    match row.get(i) {
        Some(ast::Value::Text(s)) => s,
        _ => "",
    }
}

/// All partition-catalog rows (empty when the catalog does not exist yet).
fn rows(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Vec<ast::Value>>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        out.push(row::decode(&bytes, &PARTITION_CATALOG_SCHEMA)?);
    }
    Ok(out)
}

/// Encode a bound value (already the key type) as hex of its one-column row encoding, so it survives
/// in a text catalog column and decodes back to the exact typed value.
fn encode_bound(value: &ast::Value, key_ty: ColumnType) -> Result<String, Error> {
    let bytes = row::encode(std::slice::from_ref(value), &[key_ty])?;
    Ok(super::crypto::to_hex(&bytes))
}

fn decode_bound(hex: &str, key_ty: ColumnType) -> Result<ast::Value, Error> {
    let bytes = super::crypto::from_hex(hex).ok_or(Error::MalformedTuple { offset: 0 })?;
    let mut decoded = row::decode(&bytes, &[key_ty])?;
    Ok(decoded.pop().unwrap_or(ast::Value::Null))
}

/// Look up the partition catalog, creating it lazily if absent.
fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["role", "table", "aux", "lo", "hi"]
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: PARTITION_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}
