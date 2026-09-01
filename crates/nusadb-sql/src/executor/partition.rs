//! Table partitioning — metadata catalog + `INSERT` routing (RANGE, LIST, HASH, DEFAULT).
//!
//! `CREATE TABLE m (...) PARTITION BY {RANGE|LIST|HASH} (col, ...)` records a *parent* row here;
//! `CREATE TABLE p PARTITION OF m FOR VALUES ...` records a *partition* row (and a child→parent edge
//! in the inheritance catalog, so a query on the parent expands over its partitions exactly as an
//! inheritance parent does). The parent table holds no rows of its own: an `INSERT` into it is routed
//! to the partition whose bound accepts the key tuple — a `[lo, hi)` range (compared lexicographically
//! for a multi-column key), an `IN (...)` value set, or `hash(key) mod modulus = remainder` — or to
//! the `DEFAULT` catch-all when no other bound matches. `RANGE`/`HASH` allow a multi-column key;
//! `LIST` is single-column. Mirrors the trigger/function catalog pattern — no storage-spine change.
#![allow(clippy::wildcard_imports)]

use std::cmp::Ordering;

use super::*;

/// Engine-scoped system catalog of partition metadata.
pub(super) const PARTITION_CATALOG: &str = "nusadb_partitions";

/// Separator joining a parent's key column names in the catalog's `aux` field. A unit-separator
/// control character never appears in an identifier, so it round-trips names unambiguously.
const KEY_SEP: char = '\u{1f}';

/// Five-text-column schema: `(role, table, aux, kind, payload)`.
/// - a parent row is `("parent", <parent table>, <key columns joined by KEY_SEP>, <strategy>, "")`;
/// - a partition row is `("part", <partition table>, <parent table>, <kind>, <payload>)`, where
///   `kind` is `range`/`list`/`hash`/`default` and `payload` encodes the bound (see [`encode_payload`]).
const PARTITION_CATALOG_SCHEMA: [ColumnType; 5] = [ColumnType::Text; 5];

/// A partition's bound (values already coerced to the key column types).
#[derive(Clone)]
pub(super) enum PartitionBound {
    /// `[lo, hi)` — a key tuple `k` belongs when `lo <= k < hi` (lexicographic tuple order).
    Range {
        lo: Vec<ast::Value>,
        hi: Vec<ast::Value>,
    },
    /// An explicit value set (single-column) — a key belongs when it equals one of them.
    List(Vec<ast::Value>),
    /// `hash(key) mod modulus = remainder`.
    Hash { modulus: u64, remainder: u64 },
    /// The catch-all partition: holds every row matching no sibling's bound. Never matches via
    /// [`accepts`] — routing falls back to it only when no other partition accepts the key.
    Default,
}

/// Whether `bound` is the catch-all [`PartitionBound::Default`].
pub(super) const fn is_default(bound: &PartitionBound) -> bool {
    matches!(bound, PartitionBound::Default)
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

/// Record a `PARTITION BY <strategy> (key, ...)` parent.
pub(super) fn record_parent(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_columns: &[String],
    strategy: ast::PartitionStrategy,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let row = [
        text("parent"),
        text(parent),
        text(&key_columns.join(&KEY_SEP.to_string())),
        text(strategy_str(strategy)),
        text(""),
    ];
    engine.insert(txn, cat, &row::encode(&row, &PARTITION_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Record a partition of `parent` with a bound already coerced to the key column types.
pub(super) fn record_partition(
    engine: &dyn StorageEngine,
    txn: TxnId,
    partition: &str,
    parent: &str,
    key_tys: &[ColumnType],
    bound: &PartitionBound,
) -> Result<(), Error> {
    let cat = ensure_catalog(engine, txn)?;
    let (kind, payload) = encode_payload(bound, key_tys)?;
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

/// The partition key columns of `table` if it is a partitioned parent, else `None` (always non-empty
/// when `Some`).
pub(super) fn parent_key_columns(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<Option<Vec<String>>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "parent" && field(&row, 1) == table {
            return Ok(Some(
                field(&row, 2).split(KEY_SEP).map(str::to_owned).collect(),
            ));
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

/// The `(parent, bound)` of `table` if it is a partition, decoded at `key_tys`.
pub(super) fn partition_bound(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    key_tys: &[ColumnType],
) -> Result<Option<(String, PartitionBound)>, Error> {
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 1) == table {
            let bound = decode_payload(field(&row, 3), field(&row, 4), key_tys)?;
            return Ok(Some((field(&row, 2).to_owned(), bound)));
        }
    }
    Ok(None)
}

/// Every partition of `parent`, decoded at `key_tys`.
pub(super) fn partitions_of(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_tys: &[ColumnType],
) -> Result<Vec<PartitionEntry>, Error> {
    let mut out = Vec::new();
    for row in rows(engine, txn)? {
        if field(&row, 0) == "part" && field(&row, 2) == parent {
            out.push(PartitionEntry {
                table: field(&row, 1).to_owned(),
                bound: decode_payload(field(&row, 3), field(&row, 4), key_tys)?,
            });
        }
    }
    Ok(out)
}

/// Remove only the partition-catalog row for `partition` (its `("part", partition, …)` row), leaving
/// any parent row and every other partition untouched — for `DETACH PARTITION`, which severs one
/// partition without dropping its table.
pub(super) fn remove_partition_entry(
    engine: &dyn StorageEngine,
    txn: TxnId,
    partition: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PARTITION_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PARTITION_CATALOG_SCHEMA)?;
        if field(&row, 0) == "part" && field(&row, 1) == partition {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
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

/// Whether the key tuple `key` belongs in `bound`. A `NULL` in any key element is matched by no
/// range/hash bound (such a row routes to the catch-all); a `LIST` set may hold an explicit `NULL`.
pub(super) fn accepts(key: &[ast::Value], bound: &PartitionBound, key_tys: &[ColumnType]) -> bool {
    match bound {
        PartitionBound::Range { lo, hi } => {
            !key.iter().any(|v| matches!(v, ast::Value::Null))
                && compare_tuple(key, lo) != Ordering::Less
                && compare_tuple(key, hi) == Ordering::Less
        },
        PartitionBound::List(values) => {
            // LIST is single-column: compare the sole key element to each listed value.
            key.first().is_some_and(|k| {
                values
                    .iter()
                    .any(|v| super::eval::compare(k, v) == Ordering::Equal)
            })
        },
        PartitionBound::Hash { modulus, remainder } => {
            !key.iter().any(|v| matches!(v, ast::Value::Null))
                && *modulus != 0
                && value_hash(key, key_tys) % modulus == *remainder
        },
        // The catch-all never accepts a key directly; routing falls back to it explicitly.
        PartitionBound::Default => false,
    }
}

/// Lexicographic comparison of two key tuples (element-by-element, first difference wins).
pub(super) fn compare_tuple(a: &[ast::Value], b: &[ast::Value]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        match super::eval::compare(x, y) {
            Ordering::Equal => {},
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

/// The direct partitions of `parent` whose bound provably contains no key satisfying `constraints`
/// — the partitions a query with those key predicates need not scan. Each `constraint` is tagged with
/// the key column it constrains. A partition is dropped only when the constraints provably leave it
/// empty, so the result never changes which rows a query returns — only how many partitions it scans.
pub(super) fn prune(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
    key_tys: &[ColumnType],
    constraints: &[crate::PruneConstraint],
) -> Result<Vec<String>, Error> {
    let mut dropped = Vec::new();
    for p in partitions_of(engine, txn, parent, key_tys)? {
        if partition_excluded(&p.bound, constraints, key_tys) {
            dropped.push(p.table);
        }
    }
    Ok(dropped)
}

/// Whether `bound` provably contains no key satisfying `constraints`. Conservative — returns `false`
/// (keep the partition) whenever no static proof applies.
///
/// Leading-column (`key_index == 0`) constraints drive range/list/hash pruning: for a single-column
/// key a range bound is `[lo, hi)` (half-open, exact); for a multi-column key the leading value of
/// every row lies in the closed interval `[lo[0], hi[0]]` (a row with leading `= hi[0]` is possible
/// when the trailing bound allows), so the leading test uses that closed interval — a proven superset.
///
/// A *non-leading* (`key_index == 1`) constraint can prune a range partition only when the partition's
/// leading value is pinned (`lo[0] == hi[0]`) **and** the query pins the leading column to that same
/// value with `=`: the partition then reduces to a range on the second column — `[lo[1], hi[1])` when
/// the second column is the last key column (exact), else the closed superset `[lo[1], hi[1]]`.
fn partition_excluded(
    bound: &PartitionBound,
    constraints: &[crate::PruneConstraint],
    key_tys: &[ColumnType],
) -> bool {
    let Some(&lead_ty) = key_tys.first() else {
        return false;
    };
    let multi = key_tys.len() > 1;
    // Cast a constraint's constant to a key column's type; `None` if it cannot coerce or is NULL.
    let cast = |value: &ast::Value, ty: ColumnType| -> Option<ast::Value> {
        match super::eval::cast_value(value.clone(), ty) {
            Ok(v) if !matches!(v, ast::Value::Null) => Some(v),
            _ => None,
        }
    };
    match bound {
        PartitionBound::Range { lo, hi } => {
            let (Some(lo0), Some(hi0)) = (lo.first(), hi.first()) else {
                return false;
            };
            // Leading-column constraints.
            for c in constraints.iter().filter(|c| c.key_index == 0) {
                if let Some(v) = cast(&c.value, lead_ty) {
                    let excluded = if multi {
                        leading_range_excludes(lo0, hi0, c.op, &v)
                    } else {
                        range_excludes(lo0, hi0, c.op, &v)
                    };
                    if excluded {
                        return true;
                    }
                }
            }
            // Second-column constraints, only when the leading value is pinned and the query pins it.
            let leading_pinned = super::eval::compare(lo0, hi0) == Ordering::Equal;
            let query_pins_leading = constraints.iter().any(|c| {
                c.key_index == 0
                    && matches!(c.op, crate::PruneOp::Eq)
                    && cast(&c.value, lead_ty)
                        .is_some_and(|v| super::eval::compare(&v, lo0) == Ordering::Equal)
            });
            if multi && leading_pinned && query_pins_leading {
                // The second column is exactly bounded (half-open) only when it is the last key
                // column; with further columns it is a closed superset.
                let second_is_last = key_tys.len() == 2;
                if let (Some(&second_ty), Some(lo1), Some(hi1)) =
                    (key_tys.get(1), lo.get(1), hi.get(1))
                {
                    for c in constraints.iter().filter(|c| c.key_index == 1) {
                        if let Some(v) = cast(&c.value, second_ty) {
                            let excluded = if second_is_last {
                                range_excludes(lo1, hi1, c.op, &v)
                            } else {
                                leading_range_excludes(lo1, hi1, c.op, &v)
                            };
                            if excluded {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        },
        // A value set (single-column) is excluded when no member satisfies some leading constraint.
        PartitionBound::List(values) => constraints.iter().filter(|c| c.key_index == 0).any(|c| {
            cast(&c.value, lead_ty).is_some_and(|v| !values.iter().any(|x| op_holds(x, c.op, &v)))
        }),
        // A single-column hash partition can be excluded by a leading equality (route the constant,
        // keep the one that would hold it). A multi-column hash needs every key column.
        PartitionBound::Hash { .. } => {
            !multi
                && constraints.iter().filter(|c| c.key_index == 0).any(|c| {
                    matches!(c.op, crate::PruneOp::Eq)
                        && cast(&c.value, lead_ty)
                            .is_some_and(|v| !accepts(std::slice::from_ref(&v), bound, key_tys))
                })
        },
        // The catch-all can hold any key no sibling covers, so it can never be pruned.
        PartitionBound::Default => false,
    }
}

/// Whether `x <op> v` holds (both already the key type).
fn op_holds(x: &ast::Value, op: crate::PruneOp, v: &ast::Value) -> bool {
    let c = super::eval::compare(x, v);
    match op {
        crate::PruneOp::Eq => c == Ordering::Equal,
        crate::PruneOp::Lt => c == Ordering::Less,
        crate::PruneOp::LtEq => c != Ordering::Greater,
        crate::PruneOp::Gt => c == Ordering::Greater,
        crate::PruneOp::GtEq => c != Ordering::Less,
    }
}

/// Whether the range `[lo, hi)` provably contains no key satisfying `key <op> v`. See the truth table
/// in the partition-pruning tests for each case.
fn range_excludes(lo: &ast::Value, hi: &ast::Value, op: crate::PruneOp, v: &ast::Value) -> bool {
    use super::eval::compare;
    match op {
        // `= v`: excluded unless `v` falls in `[lo, hi)`.
        crate::PruneOp::Eq => {
            !(compare(v, lo) != Ordering::Less && compare(v, hi) == Ordering::Less)
        },
        // `< v`: every key is `>= lo`, so none is `< v` once `lo >= v`.
        crate::PruneOp::Lt => compare(lo, v) != Ordering::Less,
        // `<= v`: none is `<= v` once `lo > v`.
        crate::PruneOp::LtEq => compare(lo, v) == Ordering::Greater,
        // `> v` / `>= v`: every key is `< hi`, so none is `> v` (nor `>= v`) once `v >= hi`.
        crate::PruneOp::Gt | crate::PruneOp::GtEq => compare(v, hi) != Ordering::Less,
    }
}

/// Whether a multi-column range partition provably contains no key whose *leading* column satisfies
/// `key <op> v`, given the leading column lies in the **closed** interval `[lo0, hi0]` (both bounds
/// possible — see [`excludes`]). A proven superset, so this only ever keeps extra partitions.
fn leading_range_excludes(
    lo0: &ast::Value,
    hi0: &ast::Value,
    op: crate::PruneOp,
    v: &ast::Value,
) -> bool {
    use super::eval::compare;
    match op {
        // `= v`: excluded unless `v` falls in `[lo0, hi0]`.
        crate::PruneOp::Eq => {
            !(compare(v, lo0) != Ordering::Less && compare(v, hi0) != Ordering::Greater)
        },
        // `< v`: every leading value is `>= lo0`, so none is `< v` once `lo0 >= v`.
        crate::PruneOp::Lt => compare(lo0, v) != Ordering::Less,
        // `<= v`: none is `<= v` once `lo0 > v`.
        crate::PruneOp::LtEq => compare(lo0, v) == Ordering::Greater,
        // `> v`: every leading value is `<= hi0`, so none is `> v` once `hi0 <= v`.
        crate::PruneOp::Gt => compare(hi0, v) != Ordering::Greater,
        // `>= v`: none is `>= v` once `hi0 < v`.
        crate::PruneOp::GtEq => compare(hi0, v) == Ordering::Less,
    }
}

/// A deterministic 64-bit hash of a key tuple at its column types, for HASH partition routing. Uses
/// FNV-1a over the tuple's canonical row encoding — self-contained and stable within an engine build
/// (the exact partition a key lands in is engine-specific, as it is in the reference engine).
pub(super) fn value_hash(key: &[ast::Value], key_tys: &[ColumnType]) -> u64 {
    let bytes = row::encode(key, key_tys).unwrap_or_default();
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

/// Encode a bound to `(kind, payload)`. A range payload is `hex(lo tuple)|hex(hi tuple)` (each side's
/// values row-encoded together, so they survive in a text column and decode back to the exact typed
/// tuple); a list payload is the pipe-separated hex of each single-column value; a hash payload is
/// `modulus|remainder`.
fn encode_payload(
    bound: &PartitionBound,
    key_tys: &[ColumnType],
) -> Result<(&'static str, String), Error> {
    Ok(match bound {
        PartitionBound::Range { lo, hi } => (
            "range",
            format!("{}|{}", hex_tuple(lo, key_tys)?, hex_tuple(hi, key_tys)?),
        ),
        PartitionBound::List(values) => {
            let one = key_tys.first().copied().unwrap_or(ColumnType::Text);
            let parts = values
                .iter()
                .map(|v| hex_tuple(std::slice::from_ref(v), &[one]))
                .collect::<Result<Vec<_>, _>>()?;
            ("list", parts.join("|"))
        },
        PartitionBound::Hash { modulus, remainder } => ("hash", format!("{modulus}|{remainder}")),
        PartitionBound::Default => ("default", String::new()),
    })
}

/// Decode a `(kind, payload)` back to a bound at `key_tys`.
fn decode_payload(
    kind: &str,
    payload: &str,
    key_tys: &[ColumnType],
) -> Result<PartitionBound, Error> {
    let bad = || Error::MalformedTuple { offset: 0 };
    match kind {
        "range" => {
            let (lo, hi) = payload.split_once('|').ok_or_else(bad)?;
            Ok(PartitionBound::Range {
                lo: unhex_tuple(lo, key_tys)?,
                hi: unhex_tuple(hi, key_tys)?,
            })
        },
        "list" => {
            let one = key_tys.first().copied().unwrap_or(ColumnType::Text);
            let values = payload
                .split('|')
                .map(|h| Ok(unhex_tuple(h, &[one])?.pop().unwrap_or(ast::Value::Null)))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(PartitionBound::List(values))
        },
        "hash" => {
            let (m, r) = payload.split_once('|').ok_or_else(bad)?;
            Ok(PartitionBound::Hash {
                modulus: m.parse().map_err(|_| bad())?,
                remainder: r.parse().map_err(|_| bad())?,
            })
        },
        "default" => Ok(PartitionBound::Default),
        _ => Err(bad()),
    }
}

/// Hex of a value tuple's row encoding at `key_tys` (round-trips to the exact typed tuple).
fn hex_tuple(values: &[ast::Value], key_tys: &[ColumnType]) -> Result<String, Error> {
    let bytes = row::encode(values, key_tys)?;
    Ok(super::crypto::to_hex(&bytes))
}

fn unhex_tuple(hex: &str, key_tys: &[ColumnType]) -> Result<Vec<ast::Value>, Error> {
    let bytes = super::crypto::from_hex(hex).ok_or(Error::MalformedTuple { offset: 0 })?;
    row::decode(&bytes, key_tys)
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
