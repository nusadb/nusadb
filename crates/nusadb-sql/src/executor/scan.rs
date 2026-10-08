//! Scan helpers: materialize a table's visible rows for the executor.
//!
//! Split verbatim out of `executor/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === Scan helpers =========================================================

pub(super) fn scan_table(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, Row)>, Error> {
    let schema = column_types(table);
    let mut scan = engine.scan(txn, table.id)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        // Cooperative cancellation: a statement timeout / cancel request aborts a long scan
        // at the next row boundary rather than running to completion.
        crate::cancel::check()?;
        // Under a `FOR UPDATE` / `FOR SHARE` guard (no-op otherwise): a row the lock step left
        // out (held elsewhere under SKIP LOCKED, past the cap, changed and no longer matching) is
        // invisible to this pipeline, and a changed row that still matches reads as its newest
        // version.
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        out.push((tid, row::decode(&tuple, &schema)?));
    }
    Ok(out)
}

/// A DML statement's target rows: those a point lookup found, or a table scan read one row at a
/// time with exactly [`scan_table`]'s visibility, so a statement over a large table does not hold
/// every row it passes over. A scan never reads rows its own transaction writes after it opened.
pub(super) enum TargetRows {
    Found(std::vec::IntoIter<(Tid, Row)>),
    Scan {
        scan: Box<dyn nusadb_core::engine::TupleScan>,
        table: nusadb_core::TableId,
        schema: Vec<ColumnType>,
    },
}

impl TargetRows {
    /// Every visible row of `table`, streamed.
    pub(super) fn scan(
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<Self, Error> {
        Ok(Self::Scan {
            scan: engine.scan(txn, table.id)?,
            table: table.id,
            schema: column_types(table),
        })
    }

    /// The next target row.
    pub(super) fn try_next(&mut self) -> Result<Option<(Tid, Row)>, Error> {
        match self {
            Self::Found(rows) => Ok(rows.next()),
            Self::Scan {
                scan,
                table,
                schema,
            } => {
                while let Some((tid, tuple)) = scan.try_next()? {
                    crate::cancel::check()?;
                    let tuple = match super::lock_skip::resolve(*table, tid) {
                        super::lock_skip::Seen::Hide => continue,
                        super::lock_skip::Seen::Replace(newer) => newer,
                        super::lock_skip::Seen::Keep => tuple,
                    };
                    return Ok(Some((tid, row::decode(&tuple, schema)?)));
                }
                Ok(None)
            },
        }
    }
}

/// Count the visible rows of `table` **without decoding any row bytes** — the `COUNT(*)` fast-path.
/// Same visibility as [`scan_table`] (the engine applies MVCC per tuple, plus `SKIP LOCKED` and the
/// recursive-CTE working set), so the count is exactly what folding over the decoded rows would
/// yield, but it skips the `O(rows × columns)` row materialization that a bare `COUNT(*)` discards.
pub(super) fn count_table(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<usize, Error> {
    // A recursive CTE exposes its working set as a synthetic table (see [`scan_rows`]).
    if let Some(rows) = super::recursive::working_set(table.id) {
        crate::cancel::check()?;
        return Ok(rows.len());
    }
    let mut scan = engine.scan(txn, table.id)?;
    let mut count = 0usize;
    while let Some((tid, _tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        // A row a `LockRows` guard hides is invisible here; a replaced one still counts once.
        if super::lock_skip::hidden(table.id, tid) {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

/// Like [`scan_table`], but with *latest-committed* visibility (plus this txn's own writes) for a
/// uniqueness check that must not miss a row another transaction committed after a frozen REPEATABLE
/// READ / SERIALIZABLE snapshot. Keeps the `Tid` so a caller can exclude the rows it is itself
/// rewriting.
pub(super) fn scan_table_committed(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, Row)>, Error> {
    let schema = column_types(table);
    let mut scan = engine.scan_committed(txn, table.id)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        out.push((tid, row::decode(&tuple, &schema)?));
    }
    Ok(out)
}

/// Materialize a table's rows for a uniqueness / `PRIMARY KEY` constraint check: unlike
/// [`scan_rows`], this reads the *latest committed* state (plus this txn's own writes) rather than the
/// txn's frozen snapshot, so a row another transaction committed after a REPEATABLE READ / SERIALIZABLE
/// txn began is still seen and a duplicate key is rejected.
pub(super) fn scan_rows_committed(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let schema = column_types(table);
    let mut scan = engine.scan_committed(txn, table.id)?;
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        out.push(row::decode(&tuple, &schema)?);
    }
    Ok(out)
}

pub(super) fn scan_rows(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    // A recursive CTE exposes its working set as a synthetic table; a scan of it reads those
    // in-memory rows from the registry instead of the storage engine.
    if let Some(rows) = super::recursive::working_set(table.id) {
        crate::cancel::check()?;
        return Ok(rows);
    }
    Ok(scan_table(table, engine, txn)?
        .into_iter()
        .map(|(_, row)| row)
        .collect())
}

/// Materialize the visible rows of `table`, keeping only the projected `columns`. An empty
/// `columns` is the identity — the full row, exactly as [`scan_rows`]. A non-empty list is the
/// ascending source ordinals the projection-pushdown pass narrowed the scan to, and each row holds
/// just those columns in that order.
pub(super) fn scan_rows_projected(
    table: &TableSchema,
    columns: &[usize],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    if columns.is_empty() {
        return scan_rows(table, engine, txn);
    }
    // A recursive CTE's working set lives in memory as full-width rows; project it directly. (The
    // pushdown pass never narrows a recursive-CTE scan, so this is a defensive path.)
    if let Some(rows) = super::recursive::working_set(table.id) {
        crate::cancel::check()?;
        return Ok(rows
            .into_iter()
            .map(|r| columns.iter().filter_map(|&i| r.get(i).cloned()).collect())
            .collect());
    }
    let schema = column_types(table);
    let mut scan = engine.scan(txn, table.id)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        out.push(row::decode_projected(&tuple, &schema, columns)?);
    }
    Ok(out)
}

/// [`index_scan_rows`] as a stream: rows are decoded as the engine's index cursor yields them,
/// with the same `SKIP LOCKED` filtering and row cap.
///
/// # Errors
/// [`Error::IndexNotFound`] for an unknown index; propagates key-encoding and storage errors.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the IndexScan operator's own field set, like index_scan_rows"
)]
pub(super) fn index_scan_source(
    table: &TableSchema,
    index: &str,
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
    direction: nusadb_core::engine::ScanDirection,
    limit: Option<usize>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<IndexScanSource, Error> {
    let id = engine
        .lookup_index(index)?
        .ok_or_else(|| Error::IndexNotFound {
            name: index.to_owned(),
        })?;
    let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
    let scan = engine.index_scan_directed_limited(txn, id, lo_key, hi_key, direction, limit)?;
    Ok(IndexScanSource {
        scan,
        table: table.id,
        schema: column_types(table),
        remaining: limit,
    })
}

/// The rows of an index range, decoded one at a time (see [`index_scan_source`]).
pub(super) struct IndexScanSource {
    scan: Box<dyn nusadb_core::TupleScan>,
    table: nusadb_core::TableId,
    schema: Vec<ColumnType>,
    remaining: Option<usize>,
}

impl super::stream::RowSource for IndexScanSource {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        while let Some((tid, tuple)) = self.scan.try_next()? {
            crate::cancel::check()?;
            // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
            let tuple = match super::lock_skip::resolve(self.table, tid) {
                super::lock_skip::Seen::Hide => continue,
                super::lock_skip::Seen::Replace(newer) => newer,
                super::lock_skip::Seen::Keep => tuple,
            };
            if let Some(remaining) = self.remaining.as_mut() {
                *remaining -= 1;
            }
            return Ok(Some(row::decode(&tuple, &self.schema)?));
        }
        Ok(None)
    }
}

/// Materialize the visible rows of `table` whose `index` key falls in `[lo, hi]`, in ascending key
/// order. The bound *values* are encoded into the index's order-preserving key
/// bytes; the engine maps each in-range entry to a row and applies MVCC visibility.
///
/// Safe under every isolation level: the index is MVCC-aware — an entry is kept until VACUUM
/// reclaims its row version, and the engine's `index_scan` filters each entry by per-tid visibility
/// against the transaction's snapshot — so a frozen REPEATABLE READ / SERIALIZABLE reader still finds
/// the row versions visible to it (and only those). The sequential-scan fallback is gone.
#[allow(
    clippy::too_many_arguments,
    reason = "index identity (table, index), key range (lo, hi), scan direction + row cap, and the (engine, txn) handle — each is an independent input to one ordered scan"
)]
pub(super) fn index_scan_rows(
    table: &TableSchema,
    index: &str,
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
    direction: nusadb_core::engine::ScanDirection,
    limit: Option<usize>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let id = engine
        .lookup_index(index)?
        .ok_or_else(|| Error::IndexNotFound {
            name: index.to_owned(),
        })?;
    let schema = column_types(table);
    // Pass the row cap to the engine so it stops materializing after `limit` visible rows in key
    // order — the O(range) → O(limit) win for `ORDER BY … LIMIT`. The executor-side break below is a
    // backstop for an engine whose limited scan falls back to the full directed scan. The planner
    // disqualifies this capped ordered scan whenever `SKIP LOCKED` is in force (see
    // `try_ordered_index_scan`), precisely because the engine caps on *visible* rows with no notion
    // of locks — so a locked row skipped below could otherwise make the count fall short of `limit`.
    // With that path excluded the skip set is never populated for this scan (a plain `FOR UPDATE`
    // does not skip), so the `skipped` check below never fires and the cap is exact.
    let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
    let mut scan = engine.index_scan_directed_limited(txn, id, lo_key, hi_key, direction, limit)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        out.push(row::decode(&tuple, &schema)?);
        if let Some(cap) = limit
            && out.len() >= cap
        {
            break;
        }
    }
    Ok(out)
}

/// Like [`index_scan_rows`], but keeps each row's `Tid` — for an UPDATE/DELETE that finds its target
/// rows through an index (`WHERE pk = const`) instead of a full [`scan_table`], then updates/deletes
/// by tid. Same snapshot visibility, MVCC filtering, and `SKIP LOCKED` handling as [`scan_table`];
/// the rows come back in ascending key order.
pub(super) fn index_scan_table(
    table: &TableSchema,
    index: &str,
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, Row)>, Error> {
    let id = engine
        .lookup_index(index)?
        .ok_or_else(|| Error::IndexNotFound {
            name: index.to_owned(),
        })?;
    let schema = column_types(table);
    let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
    let mut scan = engine.index_scan(txn, id, lo_key, hi_key)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        out.push((tid, row::decode(&tuple, &schema)?));
    }
    Ok(out)
}

/// One side of an encoded index-key range.
type KeyBound = std::ops::Bound<Vec<u8>>;

/// Encode a scan's key bounds into the order-preserving index-key bytes the engine compares.
///
/// A bound may name only a prefix of a `key_columns`-column key (`a = 1` on an index over
/// `(a, b)`), and then it covers every key that starts with it. Each encoded field starts with a
/// tag byte below `PAST_PREFIX`, so every key extending `prefix` sorts before
/// `prefix ++ PAST_PREFIX`: an inclusive upper bound becomes "below that", and an exclusive lower
/// bound "from there". A bound naming the whole key, which nothing extends, stays as it is, so an
/// equality on the whole key remains the engine's point read.
fn encode_key_range(
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
) -> Result<(KeyBound, KeyBound), Error> {
    use std::ops::Bound;
    const PAST_PREFIX: u8 = 0x02;
    let encode = |values: &[ast::Value], past: bool| -> Result<Vec<u8>, Error> {
        let mut key = index_key::encode_index_key(values)?;
        if past {
            key.push(PAST_PREFIX);
        }
        Ok(key)
    };
    let lo = match lo {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(values) => Bound::Included(encode(values, false)?),
        Bound::Excluded(values) if values.len() < key_columns => {
            Bound::Included(encode(values, true)?)
        },
        Bound::Excluded(values) => Bound::Excluded(encode(values, false)?),
    };
    let hi = match hi {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(values) if values.len() < key_columns => {
            Bound::Excluded(encode(values, true)?)
        },
        Bound::Included(values) => Bound::Included(encode(values, false)?),
        Bound::Excluded(values) => Bound::Excluded(encode(values, false)?),
    };
    Ok((lo, hi))
}

#[cfg(test)]
mod key_range_tests {
    use std::ops::Bound;

    use super::encode_key_range;
    use crate::ast::Value;

    fn key(values: &[Value]) -> Vec<u8> {
        super::index_key::encode_index_key(values).unwrap()
    }

    #[test]
    fn a_whole_key_equality_stays_a_point_read() {
        let k = vec![Value::Int(1), Value::Int(2)];
        let (lo, hi) =
            encode_key_range(&Bound::Included(k.clone()), &Bound::Included(k.clone()), 2).unwrap();
        assert_eq!(
            (lo, hi),
            (Bound::Included(key(&k)), Bound::Included(key(&k)))
        );
        let one = vec![Value::Int(7)];
        let (lo, hi) = encode_key_range(
            &Bound::Excluded(one.clone()),
            &Bound::Excluded(one.clone()),
            1,
        )
        .unwrap();
        assert_eq!(
            (lo, hi),
            (Bound::Excluded(key(&one)), Bound::Excluded(key(&one)))
        );
    }

    #[test]
    fn a_prefix_covers_every_key_that_starts_with_it() {
        let p = vec![Value::Int(1)];
        let mut past = key(&p);
        past.push(0x02);
        let (lo, hi) =
            encode_key_range(&Bound::Included(p.clone()), &Bound::Included(p.clone()), 2).unwrap();
        assert_eq!(
            (lo, hi),
            (Bound::Included(key(&p)), Bound::Excluded(past.clone()))
        );
        let (lo, _) = encode_key_range(&Bound::Excluded(p.clone()), &Bound::Unbounded, 2).unwrap();
        assert_eq!(lo, Bound::Included(past));
        // Every extension of the prefix, NULL or not, sorts inside the range.
        for second in [Value::Null, Value::Int(i64::MIN), Value::Int(i64::MAX)] {
            let full = key(&[Value::Int(1), second]);
            let mut upper = key(&p);
            upper.push(0x02);
            assert!(full >= key(&p) && full < upper);
        }
    }
}
