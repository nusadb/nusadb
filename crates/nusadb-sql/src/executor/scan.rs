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
        // `FOR UPDATE ... SKIP LOCKED`: a row another transaction holds locked is invisible to
        // this pipeline (no-op unless a LockRows guard is active).
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        out.push((tid, row::decode(&tuple, &schema)?));
    }
    Ok(out)
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
        // `FOR UPDATE ... SKIP LOCKED`: a row another transaction holds locked is invisible here.
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

/// Like [`scan_table`], but with *latest-committed* visibility (plus this txn's own writes) for a
/// uniqueness check that must not miss a row another transaction committed after a frozen REPEATABLE
/// READ / SERIALIZABLE snapshot (A-QA1b). Keeps the `Tid` so a caller can exclude the rows it is itself
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

/// Materialize a table's rows for a uniqueness / `PRIMARY KEY` constraint check (A-QA1b): unlike
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
        // `FOR UPDATE ... SKIP LOCKED` (see `scan_table`).
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        out.push(row::decode_projected(&tuple, &schema, columns)?);
    }
    Ok(out)
}

/// Materialize the visible rows of `table` whose `index` key falls in `[lo, hi]`, in ascending key
/// order. The bound *values* are encoded into the index's order-preserving key
/// bytes; the engine maps each in-range entry to a row and applies MVCC visibility.
///
/// Safe under every isolation level: the index is MVCC-aware — an entry is kept until VACUUM
/// reclaims its row version, and the engine's `index_scan` filters each entry by per-tid visibility
/// against the transaction's snapshot — so a frozen REPEATABLE READ / SERIALIZABLE reader still finds
/// the row versions visible to it (and only those). The sequential-scan fallback is gone.
pub(super) fn index_scan_rows(
    table: &TableSchema,
    index: &str,
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let id = engine
        .lookup_index(index)?
        .ok_or_else(|| Error::IndexNotFound {
            name: index.to_owned(),
        })?;
    let schema = column_types(table);
    let mut scan = engine.index_scan(txn, id, encode_key_bound(lo)?, encode_key_bound(hi)?)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        // `FOR UPDATE ... SKIP LOCKED` (see `scan_table`).
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        out.push(row::decode(&tuple, &schema)?);
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
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, Row)>, Error> {
    let id = engine
        .lookup_index(index)?
        .ok_or_else(|| Error::IndexNotFound {
            name: index.to_owned(),
        })?;
    let schema = column_types(table);
    let mut scan = engine.index_scan(txn, id, encode_key_bound(lo)?, encode_key_bound(hi)?)?;
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        // `FOR UPDATE ... SKIP LOCKED` (see `scan_table`).
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        out.push((tid, row::decode(&tuple, &schema)?));
    }
    Ok(out)
}

/// Encode a key-bound's prefix values into the order-preserving index-key bytes the engine compares.
fn encode_key_bound(
    bound: &std::ops::Bound<Vec<ast::Value>>,
) -> Result<std::ops::Bound<Vec<u8>>, Error> {
    use std::ops::Bound;
    Ok(match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(values) => Bound::Included(index_key::encode_index_key(values)?),
        Bound::Excluded(values) => Bound::Excluded(index_key::encode_index_key(values)?),
    })
}
