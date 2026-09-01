//! Table partitioning — metadata catalog + `INSERT` routing (RANGE, LIST, HASH).
//!
//! `CREATE TABLE m (...) PARTITION BY {RANGE|LIST|HASH} (col)` records a *parent* row here;
//! `CREATE TABLE p PARTITION OF m FOR VALUES ...` records a *partition* row (and a child→parent edge
//! in the inheritance catalog, so a query on the parent expands over its partitions exactly as an
//! inheritance parent does). The parent table holds no rows of its own: an `INSERT` into it is routed
//! to the partition whose bound accepts the key — a `[lo, hi)` range, an `IN (...)` value set, or
//! `hash(key) mod modulus = remainder`. Mirrors the trigger/function catalog pattern — no
//! storage-spine change.
#![allow(clippy::wildcard_imports)]

use std::cmp::Ordering;

use super::*;

/// Engine-scoped system catalog of partition metadata.
pub(super) const PARTITION_CATALOG: &str = "nusadb_partitions";

/// Five-text-column schema: `(role, table, aux, kind, payload)`.
/// - a parent row is `("parent", <parent table>, <key column>, <strategy>, "")`;
/// - a partition row is `("part", <partition table>, <parent table>, <kind>, <payload>)`, where
///   `kind` is `range`/`list`/`hash` and `payload` encodes the bound (see [`encode_payload`]).
const PARTITION_CATALOG_SCHEMA: [ColumnType; 5] = [ColumnType::Text; 5];

/// A partition's bound (values already coerced to the key column's type).
#[derive(Clone)]
pub(super) enum PartitionBound {
    /// `[lo, hi)` — a key `k` belongs when `lo <= k < hi`.
    Range { lo: ast::Value, hi: ast::Value },
    /// An explicit value set — a key belongs when it equals one of them.
    List(Vec<ast::Value>),
    /// `hash(key) mod modulus = remainder`.
    Hash { modulus: u64, remainder: u64 },
}

/// A resolved partition of a parent: its table name and bound.
pub(super) struct PartitionEntry {
    pub table: String,
    pub bound: PartitionBound,
}

/// The strategy string stored for a parent (`range`/`list`/`hash`).
pub(super) const fn strategy_str(s: ast::PartitionStrategy) -> &'static str {
    match s {
        ast::PartitionStrategy::Range => "range",
        ast::PartitionStrategy::List => "list",
        ast::PartitionStrategy::Hash => "hash",
    }
}

/// Record a `PARTITION BY <strategy> (key)` parent.
pub(super) fn record_parent(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_column: &str,
    strategy: ast::PartitionStrategy,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let row = [
        text("parent"),
        text(parent),
        text(key_column),
        text(strategy_str(strategy)),
        text(""),
    ];
    engine.insert(txn, cat, &row::encode(&row, &PARTITION_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Record a partition of `parent` with a bound already coerced to the key type.
pub(super) fn record_partition(
    engine: &dyn StorageEngine,
    txn: TxnId,
    partition: &str,
    parent: &str,
    key_ty: ColumnType,
    bound: &PartitionBound,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let (kind, payload) = encode_payload(bound, key_ty)?;
    let row = [
        text("part"),
        text(partition),
        text(parent),
        text(kind),
        text(&payload),
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

/// The strategy string of `table` if it is a partitioned parent, else `None`.
pub(super) fn parent_strategy(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<Option<String>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "parent" && field(&row, 1) == table {
            return Ok(Some(field(&row, 3).to_owned()));
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

/// The `(parent, bound)` of `table` if it is a partition, decoded at `key_ty`.
pub(super) fn partition_bound(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    key_ty: ColumnType,
) -> Result<Option<(String, PartitionBound)>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 1) == table {
            let bound = decode_payload(field(&row, 3), field(&row, 4), key_ty)?;
            return Ok(Some((field(&row, 2).to_owned(), bound)));
        }
    }
    Ok(None)
}

/// Every partition of `parent`, decoded at `key_ty`.
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
                bound: decode_payload(field(&row, 3), field(&row, 4), key_ty)?,
            });
        }
    }
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

/// Whether `key` belongs in `bound`. `NULL` belongs only to a `LIST` partition that explicitly holds
/// `NULL`; a range/hash partition never accepts `NULL` (matching the reference engine).
pub(super) fn accepts(key: &ast::Value, bound: &PartitionBound, key_ty: ColumnType) -> bool {
    match bound {
        PartitionBound::Range { lo, hi } => {
            !matches!(key, ast::Value::Null)
                && super::eval::compare(key, lo) != Ordering::Less
                && super::eval::compare(key, hi) == Ordering::Less
        },
        PartitionBound::List(values) => values
            .iter()
            .any(|v| super::eval::compare(key, v) == Ordering::Equal),
        PartitionBound::Hash { modulus, remainder } => {
            !matches!(key, ast::Value::Null)
                && *modulus != 0
                && value_hash(key, key_ty) % modulus == *remainder
        },
    }
}

/// A deterministic 64-bit hash of a key value at its column type, for HASH partition routing. Uses
/// FNV-1a over the value's canonical row encoding — self-contained and stable within an engine build
/// (the exact partition a key lands in is engine-specific, as it is in the reference engine).
pub(super) fn value_hash(value: &ast::Value, key_ty: ColumnType) -> u64 {
    let bytes = row::encode(std::slice::from_ref(value), &[key_ty]).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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

/// Encode a bound to `(kind, payload)`. Range/list values are hex of their one-column row encoding
/// (so they survive in a text column and decode back to the exact typed value), pipe-separated; hash
/// stores `modulus|remainder` as decimals.
fn encode_payload(
    bound: &PartitionBound,
    key_ty: ColumnType,
) -> Result<(&'static str, String), Error> {
    Ok(match bound {
        PartitionBound::Range { lo, hi } => (
            "range",
            format!("{}|{}", hex_value(lo, key_ty)?, hex_value(hi, key_ty)?),
        ),
        PartitionBound::List(values) => {
            let parts = values
                .iter()
                .map(|v| hex_value(v, key_ty))
                .collect::<Result<Vec<_>, _>>()?;
            ("list", parts.join("|"))
        },
        PartitionBound::Hash { modulus, remainder } => ("hash", format!("{modulus}|{remainder}")),
    })
}

/// Decode a `(kind, payload)` back to a bound at `key_ty`.
fn decode_payload(kind: &str, payload: &str, key_ty: ColumnType) -> Result<PartitionBound, Error> {
    let bad = || Error::MalformedTuple { offset: 0 };
    match kind {
        "range" => {
            let (lo, hi) = payload.split_once('|').ok_or_else(bad)?;
            Ok(PartitionBound::Range {
                lo: unhex_value(lo, key_ty)?,
                hi: unhex_value(hi, key_ty)?,
            })
        },
        "list" => {
            let values = payload
                .split('|')
                .map(|h| unhex_value(h, key_ty))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(PartitionBound::List(values))
        },
        "hash" => {
            let (m, r) = payload.split_once('|').ok_or_else(bad)?;
            Ok(PartitionBound::Hash {
                modulus: m.parse().map_err(|_| bad())?,
                remainder: r.parse().map_err(|_| bad())?,
            })
        },
        _ => Err(bad()),
    }
}

/// Hex of a single value's row encoding at `key_ty` (round-trips to the exact typed value).
fn hex_value(value: &ast::Value, key_ty: ColumnType) -> Result<String, Error> {
    let bytes = row::encode(std::slice::from_ref(value), &[key_ty])?;
    Ok(super::crypto::to_hex(&bytes))
}

fn unhex_value(hex: &str, key_ty: ColumnType) -> Result<ast::Value, Error> {
    let bytes = super::crypto::from_hex(hex).ok_or(Error::MalformedTuple { offset: 0 })?;
    let mut decoded = row::decode(&bytes, &[key_ty])?;
    Ok(decoded.pop().unwrap_or(ast::Value::Null))
}

/// Look up the partition catalog, creating it lazily if absent.
fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["role", "table", "aux", "kind", "payload"]
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
