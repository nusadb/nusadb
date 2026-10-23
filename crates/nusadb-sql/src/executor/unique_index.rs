//! `CREATE UNIQUE INDEX ... [NULLS [NOT] DISTINCT] [WHERE pred]`: the SQL side of a unique index
//! that is not the backing index of a `PRIMARY KEY` / `UNIQUE` constraint.
//!
//! # Enforcement
//!
//! Such an index is enforced by the engine itself: every entry goes through `index_insert`, whose
//! uniqueness check and entry write are one atomic step under the index's write latch, against the
//! newest version of every row holding the key (committed or not). Every write path (INSERT, COPY,
//! INSERT ... SELECT, UPDATE, MERGE, upsert, foreign-key cascades, the CREATE INDEX backfill)
//! maintains the index through that call, so a duplicate is refused with `23505` on all of them
//! and two concurrent writers of one key can never both commit, whatever their isolation level:
//! the second one's insert finds the first one's row and fails. A partial index holds only the
//! rows its predicate accepts, so only those rows take part.
//!
//! The engine compares key bytes, and the key encoding gives `NULL` one fixed byte, so two `NULL`
//! keys are equal to it. That is `NULLS NOT DISTINCT`. For the default `NULLS DISTINCT`, an entry
//! whose key holds a `NULL` gets its row's address appended (see [`null_suffix`]), which makes it
//! differ from every other row's entry, so it never conflicts. Scans are unaffected: no equality
//! bound ever holds a `NULL`, and the suffix keeps the entry inside every range its key prefix
//! falls in.
//!
//! # Storage
//!
//! The index catalog has no field for the setting, so `NULLS NOT DISTINCT` is carried in the
//! stored predicate behind a tag (`parser::tag_index_predicate`); [`split_predicate`] reads it back.
//!
//! # `ON CONFLICT`
//!
//! A unique index can be the arbiter of an upsert. `ON CONFLICT (cols)` infers a full unique index
//! over exactly those columns; `ON CONFLICT (cols) WHERE pred` also accepts a partial one whose
//! predicate's conjuncts all appear among `pred`'s ([`infer_arbiter`]). `ON CONFLICT DO NOTHING`
//! skips a row that would collide on any unique index ([`keep_non_conflicting`]).

use super::*;
use crate::planner::TypedExpr;

/// The partial-index predicate of `def` (`None` for a full index) and whether the index is
/// `NULLS NOT DISTINCT`.
pub(super) fn split_predicate(def: &nusadb_core::IndexDef) -> (Option<&str>, bool) {
    crate::parser::untag_index_predicate(def.predicate.as_deref())
}

/// The bytes appended to a `NULLS DISTINCT` unique index entry whose key holds a `NULL`: the row's
/// address, so the entry is unique to its row. A key with no `NULL` is stored as is.
pub(super) fn null_suffix(tid: Tid) -> [u8; 11] {
    let mut out = [0u8; 11];
    // A marker byte above every field tag (`0x00` NULL, `0x01` value), then the address.
    out[0] = 0xFF;
    out[1..9].copy_from_slice(&tid.page.0.to_be_bytes());
    out[9..].copy_from_slice(&tid.slot.0.to_be_bytes());
    out
}

/// The ids of `table`'s constraint-backing indexes. Those are checked by the constraint machinery,
/// not by the engine, and keep their plain key encoding.
pub(super) fn backing_index_ids(
    table: &TableSchema,
    engine: &dyn StorageEngine,
) -> Result<HashSet<nusadb_core::IndexId>, Error> {
    Ok(engine
        .list_constraints(table.id)?
        .into_iter()
        .filter_map(|c| c.index)
        .collect())
}

/// What the write path needs to check one entry of a unique index that backs no constraint.
pub(super) struct Guard {
    /// The index name, for the error.
    pub(super) name: String,
    /// The indexed table (the key locks are taken on it).
    pub(super) table: nusadb_core::TableId,
    /// The table's column types, to decode the rows a probe returns.
    pub(super) types: Vec<ColumnType>,
    /// Whether two `NULL` keys collide.
    pub(super) nulls_not_distinct: bool,
}

/// The [`Guard`] of index `def` (id `id`) on `table`, or `None` when it is not unique or backs a
/// constraint (whose checks live in the constraint machinery).
pub(super) fn guard_for(
    def: &nusadb_core::IndexDef,
    id: nusadb_core::IndexId,
    table: &TableSchema,
    backing: &HashSet<nusadb_core::IndexId>,
) -> Option<Guard> {
    (def.unique && !backing.contains(&id)).then(|| Guard {
        name: def.name.clone(),
        table: table.id,
        types: column_types(table),
        nulls_not_distinct: split_predicate(def).1,
    })
}

/// The no-wait key-lock id for `key` of `index`: every writer of the key takes it exclusively and
/// holds it to the end of its transaction, so two transactions writing one key collide at once.
pub(super) fn lock_hash(index: nusadb_core::IndexId, key: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    "unique index".hash(&mut hasher);
    index.0.hash(&mut hasher);
    key.hash(&mut hasher);
    hasher.finish()
}

/// The duplicate-key error of unique index `name`, in the engine's wording.
pub(super) fn duplicate(name: &str) -> Error {
    nusadb_core::Error::ConstraintViolation(format!("duplicate key violates unique index {name}"))
        .into()
}

/// One unique index usable as an `ON CONFLICT` arbiter: plain key columns, optionally partial.
pub(super) struct ArbiterKey {
    /// The key column ordinals, in index order.
    pub(super) ordinals: Vec<usize>,
    /// Whether two `NULL` keys collide.
    pub(super) nulls_not_distinct: bool,
    /// The partial-index predicate; a row it does not accept holds no key.
    pub(super) predicate: Option<TypedExpr>,
}

impl ArbiterKey {
    /// An arbiter over a key constraint's columns (full, `NULL` keys never collide).
    pub(super) const fn columns(ordinals: Vec<usize>) -> Self {
        Self {
            ordinals,
            nulls_not_distinct: false,
            predicate: None,
        }
    }

    /// `row`'s key under this arbiter, or `None` when the row takes no part in it.
    pub(super) fn key(&self, row: &Row) -> Result<Option<Vec<ast::Value>>, Error> {
        if let Some(predicate) = &self.predicate
            && !matches!(eval::eval(predicate, row)?, ast::Value::Bool(true))
        {
            return Ok(None);
        }
        Ok(dml::unique_key(
            row,
            &self.ordinals,
            self.nulls_not_distinct,
        ))
    }
}

/// `table`'s unique indexes over plain columns that back no constraint, with their definitions.
fn user_unique_indexes(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(nusadb_core::IndexDef, nusadb_core::IndexId, ArbiterKey)>, Error> {
    let defs: Vec<_> = engine
        .list_indexes_as_of(txn, table.id)?
        .into_iter()
        .filter(|d| d.unique && d.key_exprs.is_empty())
        .collect();
    if defs.is_empty() {
        return Ok(Vec::new());
    }
    let backing = backing_index_ids(table, engine)?;
    let mut out = Vec::new();
    for def in defs {
        let Some(id) = engine.lookup_index(&def.name)? else {
            continue;
        };
        if backing.contains(&id) {
            continue;
        }
        let (predicate_sql, nulls_not_distinct) = split_predicate(&def);
        let predicate = match predicate_sql {
            Some(sql) => {
                match crate::analyzer::analyze_check_predicate(sql, table, &dml::EmptyCatalog) {
                    Ok(typed) => Some(typed),
                    // An index whose predicate no longer analyzes is not maintained either.
                    Err(_) => continue,
                }
            },
            None => None,
        };
        let Ok(ordinals) = dml::constraint_ordinals(table, &def.columns) else {
            continue;
        };
        let key = ArbiterKey {
            ordinals,
            nulls_not_distinct,
            predicate,
        };
        out.push((def, id, key));
    }
    Ok(out)
}

/// The unique index an `ON CONFLICT (cols) [WHERE pred]` target names: one over exactly `wanted`
/// (the target's column ordinals, in any order) whose predicate, if it has one, is implied by
/// `inference` (each of its conjuncts is one of `inference`'s). `None` when no index qualifies.
pub(super) fn infer_arbiter(
    table: &TableSchema,
    wanted: &[usize],
    inference: Option<&str>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<ArbiterKey>, Error> {
    let mut wanted = wanted.to_vec();
    wanted.sort_unstable();
    let given = inference
        .map(crate::parser::predicate_conjuncts)
        .transpose()?
        .unwrap_or_default();
    // A full index wins over a partial one, as it covers every row.
    let mut partial = None;
    for (def, _, key) in user_unique_indexes(table, engine, txn)? {
        let mut got = key.ordinals.clone();
        got.sort_unstable();
        if got != wanted {
            continue;
        }
        match split_predicate(&def).0 {
            None => return Ok(Some(key)),
            Some(sql) if partial.is_none() => {
                let needed = crate::parser::predicate_conjuncts(sql)?;
                if needed.iter().all(|c| given.contains(c)) {
                    partial = Some(key);
                }
            },
            Some(_) => {},
        }
    }
    Ok(partial)
}

/// `ON CONFLICT DO NOTHING`: drop the rows of `rows` that would collide on a unique index (with a
/// row already in the table, or with an earlier kept row), keeping the first of a colliding set.
/// Each key is probed through its index, reading the latest committed rows plus this transaction's
/// own, the state the write's own check sees.
pub(super) fn keep_non_conflicting(
    table: &TableSchema,
    rows: Vec<Row>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let indexes = user_unique_indexes(table, engine, txn)?;
    if indexes.is_empty() || rows.is_empty() {
        return Ok(rows);
    }
    let types = column_types(table);
    // Per index, the encoded keys of the rows kept so far.
    let mut kept_keys: Vec<HashSet<Vec<u8>>> = vec![HashSet::new(); indexes.len()];
    let mut kept = Vec::with_capacity(rows.len());
    'rows: for row in rows {
        let mut keys = Vec::with_capacity(indexes.len());
        for ((_, id, index), taken) in indexes.iter().zip(&kept_keys) {
            let Some(key) = index.key(&row)? else {
                keys.push(None);
                continue;
            };
            let encoded = index_key::encode_index_key(&key)?;
            if taken.contains(&encoded)
                || held_by_a_row(*id, index, &encoded, &key, &types, engine, txn)?
            {
                continue 'rows;
            }
            keys.push(Some(encoded));
        }
        for (key, taken) in keys.into_iter().zip(&mut kept_keys) {
            if let Some(key) = key {
                taken.insert(key);
            }
        }
        kept.push(row);
    }
    Ok(kept)
}

/// Whether a live row already holds `key` (encoded `encoded`) in unique index `id`.
fn held_by_a_row(
    id: nusadb_core::IndexId,
    index: &ArbiterKey,
    encoded: &[u8],
    key: &[ast::Value],
    types: &[ColumnType],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    let mut scan = engine.index_scan_committed(
        txn,
        id,
        std::ops::Bound::Included(encoded.to_vec()),
        std::ops::Bound::Included(encoded.to_vec()),
    )?;
    while let Some((_, tuple)) = scan.try_next()? {
        // The entry may belong to an older version of the row: count it only when the version the
        // probe sees still holds the key.
        let row = row::decode(&tuple, types)?;
        if index
            .key(&row)?
            .is_some_and(|k| dml::unique_key_eq(&k, key))
        {
            return Ok(true);
        }
    }
    Ok(false)
}
