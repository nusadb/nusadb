//! Scan helpers: materialize a table's visible rows for the executor.
//!
//! Split verbatim out of `executor/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::global_unique::AFTER_BLANK;
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
    let (scan, check) = if let Some(legacy) = legacy_index_scan(
        table,
        index,
        id,
        lo,
        hi,
        key_columns,
        direction,
        limit,
        engine,
        txn,
    )? {
        legacy
    } else {
        let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
        let scan = engine.index_scan_directed_limited(txn, id, lo_key, hi_key, direction, limit)?;
        (scan, None)
    };
    Ok(IndexScanSource {
        scan,
        check,
        table: table.id,
        schema: column_types(table),
        remaining: limit,
    })
}

/// The rows of an index range, decoded one at a time (see [`index_scan_source`]).
pub(super) struct IndexScanSource {
    scan: Box<dyn nusadb_core::TupleScan>,
    /// For a legacy index, the bounds each row must still meet.
    check: Option<KeyCheck>,
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
            let row = row::decode(&tuple, &self.schema)?;
            if self.check.as_ref().is_some_and(|c| !c.keeps(&row)) {
                continue;
            }
            if let Some(remaining) = self.remaining.as_mut() {
                *remaining = remaining.saturating_sub(1);
            }
            return Ok(Some(row));
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
    let (mut scan, check) = if let Some(legacy) = legacy_index_scan(
        table,
        index,
        id,
        lo,
        hi,
        key_columns,
        direction,
        limit,
        engine,
        txn,
    )? {
        legacy
    } else {
        let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
        let scan: Box<dyn nusadb_core::TupleScan> =
            engine.index_scan_directed_limited(txn, id, lo_key, hi_key, direction, limit)?;
        (scan, None)
    };
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        let row = row::decode(&tuple, &schema)?;
        if check.as_ref().is_some_and(|c| !c.keeps(&row)) {
            continue;
        }
        out.push(row);
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
    let (mut scan, check) = if let Some(legacy) = legacy_index_scan(
        table,
        index,
        id,
        lo,
        hi,
        key_columns,
        nusadb_core::engine::ScanDirection::Forward,
        None,
        engine,
        txn,
    )? {
        legacy
    } else {
        let (lo_key, hi_key) = encode_key_range(lo, hi, key_columns)?;
        let scan: Box<dyn nusadb_core::TupleScan> = engine.index_scan(txn, id, lo_key, hi_key)?;
        (scan, None)
    };
    let mut out = Vec::new();
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        // Under a `FOR UPDATE` / `FOR SHARE` guard: hidden or replaced (see `scan_table`).
        let tuple = match super::lock_skip::resolve(table.id, tid) {
            super::lock_skip::Seen::Hide => continue,
            super::lock_skip::Seen::Replace(newer) => newer,
            super::lock_skip::Seen::Keep => tuple,
        };
        let row = row::decode(&tuple, &schema)?;
        if check.as_ref().is_some_and(|c| !c.keeps(&row)) {
            continue;
        }
        out.push((tid, row));
    }
    Ok(out)
}

/// One side of an encoded index-key range.
type KeyBound = std::ops::Bound<Vec<u8>>;

/// Entries already read, served as a scan.
struct EntryList(std::vec::IntoIter<(Tid, nusadb_core::SharedTuple)>);

impl nusadb_core::TupleScan for EntryList {
    fn try_next(&mut self) -> nusadb_core::Result<Option<(Tid, nusadb_core::SharedTuple)>> {
        Ok(self.0.next())
    }
}

/// Several scans read one after the other.
struct Chained(std::vec::IntoIter<Box<dyn nusadb_core::TupleScan>>);

impl nusadb_core::TupleScan for Chained {
    fn try_next(&mut self) -> nusadb_core::Result<Option<(Tid, nusadb_core::SharedTuple)>> {
        while let Some(scan) = self.0.as_mut_slice().first_mut() {
            if let Some(entry) = scan.try_next()? {
                return Ok(Some(entry));
            }
            self.0.next();
        }
        Ok(None)
    }
}

/// The bounds a row read through a legacy index must meet, checked on its stored (canonical)
/// key: the index ranges read for them are wider (see [`legacy_index_scan`]).
pub(super) struct KeyCheck {
    lo: std::ops::Bound<Vec<ast::Value>>,
    hi: std::ops::Bound<Vec<ast::Value>>,
    ordinals: Vec<usize>,
}

impl KeyCheck {
    /// The key of `row` under the index.
    fn key_of(&self, row: &[ast::Value]) -> Vec<ast::Value> {
        self.ordinals
            .iter()
            .map(|&o| row.get(o).cloned().unwrap_or(ast::Value::Null))
            .collect()
    }

    /// Whether `key` lies within the bounds (a bound naming a key prefix is compared on it).
    fn holds(&self, key: &[ast::Value]) -> bool {
        use std::cmp::Ordering;
        use std::ops::Bound;
        let cmp = |bound: &[ast::Value]| {
            let n = bound.len().min(key.len());
            dml::unique_key_cmp(key.get(..n).unwrap_or(key), bound.get(..n).unwrap_or(bound))
        };
        let above_lo = match &self.lo {
            Bound::Unbounded => true,
            Bound::Included(v) => cmp(v) != Ordering::Less,
            Bound::Excluded(v) => cmp(v) == Ordering::Greater,
        };
        let below_hi = match &self.hi {
            Bound::Unbounded => true,
            Bound::Included(v) => cmp(v) != Ordering::Greater,
            Bound::Excluded(v) => cmp(v) == Ordering::Less,
        };
        above_lo && below_hi
    }

    /// Whether the decoded `row` belongs to the scan.
    pub(super) fn keeps(&self, row: &[ast::Value]) -> bool {
        self.holds(&self.key_of(row))
    }
}

/// A scan through a legacy index: its entries, and the check each row read from them still needs
/// (`None` when the scan applied it already).
type LegacyScan = (Box<dyn nusadb_core::TupleScan>, Option<KeyCheck>);

/// A scan through an index an earlier release built over a `CHAR(n)` column
/// (`executor::char_canon`), or `None` for any other index.
///
/// Such an index may hold a value with the trailing blanks the statement wrote, while its row reads
/// back without them, so the entries of a key do not all sit where the key encodes. An equality
/// probes every spelling of the key (`global_unique::probe_ranges`: at most 256 exact spellings,
/// then a range over the rest of the key, each range disjoint from the others); a range scan starts
/// at its lower bound as written and ends past every spelling of its upper bound (see
/// [`spelling_ceiling`]). Each row has one live entry for the version a reader sees, so no row is
/// read twice.
///
/// A predicate scan (no `limit`) streams the ranges as the engine yields them, in the order of the
/// spellings, which is not quite key order; the caller drops the rows the returned [`KeyCheck`]
/// refuses. Only the ordered scan of an `ORDER BY ... LIMIT` carries a `limit`, and it is answered
/// by [`capped_legacy_scan`] in key order, holding no more than `limit` rows.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the IndexScan operator's own field set, like index_scan_rows"
)]
fn legacy_index_scan(
    table: &TableSchema,
    index: &str,
    id: nusadb_core::IndexId,
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
    direction: nusadb_core::engine::ScanDirection,
    limit: Option<usize>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<LegacyScan>, Error> {
    // The common case, answered from the statement's cached legacy set without the catalog.
    if !super::global_unique::canon_state(engine, txn)?.is_legacy(id) {
        return Ok(None);
    }
    let Some(def) = engine
        .list_indexes_as_of(txn, table.id)?
        .into_iter()
        .find(|d| d.name == index)
    else {
        return Ok(None);
    };
    if !def.key_exprs.is_empty()
        || !super::char_canon::needs_legacy_probe(engine, txn, id, &def, table)?
    {
        return Ok(None);
    }
    let ordinals = dml::constraint_ordinals(table, &def.columns)?;
    let types = dml::ordinal_types(table, &ordinals);
    let ranges = legacy_ranges(lo, hi, key_columns, &types)?;
    let check = KeyCheck {
        lo: lo.clone(),
        hi: hi.clone(),
        ordinals,
    };
    if limit.is_none() && direction == nusadb_core::engine::ScanDirection::Forward {
        let scans = ranges
            .into_iter()
            .map(|(from, to)| engine.index_scan(txn, id, from, to))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some((Box::new(Chained(scans.into_iter())), Some(check))));
    }
    let whole_key = matches!(
        (lo, hi),
        (std::ops::Bound::Included(a), std::ops::Bound::Included(b))
            if a.len() == key_columns && dml::unique_key_cmp(a, b) == std::cmp::Ordering::Equal
    );
    let rows = capped_legacy_scan(
        table,
        id,
        &ranges,
        &check,
        &types,
        direction,
        limit.unwrap_or(usize::MAX),
        whole_key,
        engine,
        txn,
    )?;
    Ok(Some((Box::new(EntryList(rows.into_iter())), None)))
}

/// One row kept by [`capped_legacy_scan`], ordered by key in the scan's direction. Rows with equal
/// keys have no order among them under `ORDER BY`, so any of them may be kept.
struct Kept {
    key: Vec<ast::Value>,
    tid: Tid,
    tuple: nusadb_core::SharedTuple,
    backward: bool,
}

impl Kept {
    fn order(&self, other: &Self) -> std::cmp::Ordering {
        let forward = dml::unique_key_cmp(&self.key, &other.key);
        if self.backward {
            forward.reverse()
        } else {
            forward
        }
    }
}

impl PartialEq for Kept {
    fn eq(&self, other: &Self) -> bool {
        self.order(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Kept {}
impl PartialOrd for Kept {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Kept {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order(other)
    }
}

#[cfg(test)]
thread_local! {
    /// Entries [`capped_legacy_scan`] read on this thread, for the tests that bound its work.
    pub(super) static CAPPED_ENTRIES_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The rows a capped read of a legacy index needs, in key order for its direction: the bounded
/// heap of the first `cap` and what it takes to know when to stop.
///
/// Where an entry can sit: every `CHAR(n)` value of an entry carries 0 or more of the blanks its
/// row no longer has, so an entry of key `K` lies at or above `K` encoded bare and at or below `K`
/// with every `CHAR(n)` value padded to full width (a bound computed without the padding where a
/// column is wide, see [`spelling_ceiling`]). When only the last key column is a `CHAR(n)`
/// one (the common shape: a single `CHAR(n)` column, or one after other columns), more holds:
/// every entry of a key below `W` sits below `W` encoded bare, provided `W` holds no character
/// below the blank (`'a\t'` is above `'a'`, yet `'a  '` sits above `'a\t'`).
struct Capped<'a> {
    engine: &'a dyn StorageEngine,
    txn: TxnId,
    id: nusadb_core::IndexId,
    check: &'a KeyCheck,
    types: &'a [ColumnType],
    schema: Vec<ColumnType>,
    backward: bool,
    cap: usize,
    /// Whether no key column but the last is a `CHAR(n)` one.
    tail_char: bool,
    /// Whether the read is an equality on the whole key: every row kept holds that one key, so
    /// the read ends once `cap` rows are kept.
    whole_key: bool,
    kept: std::collections::BinaryHeap<Kept>,
    /// The tids in `kept`: a row read again is not kept twice.
    in_heap: HashSet<Tid>,
    /// The encoded bound [`Capped::past_the_kept`] compares with, for the worst kept key as it
    /// stands; cleared whenever the heap changes.
    bound: Option<Vec<u8>>,
}

/// What reading one row tells a [`Capped`] read.
enum Flow {
    /// Read on.
    More,
    /// No later entry of this range can hold a key that would be kept.
    Stop,
    /// A backward read is inside a run of the worst kept key: the rest of the range holds that
    /// key and keys below it, except keys extending it with blanks and a character below the
    /// blank, which sit between its spellings.
    Run,
}

/// The first chunk a capped read asks the engine for, and the most it ever asks for at once.
const CHUNK_FIRST_MAX: usize = 1024;

impl Capped<'_> {
    fn full(&self) -> bool {
        self.kept.len() >= self.cap
    }

    /// Read one entry (its row's key `key`): keep its row if it is among the first `cap`, and say
    /// whether the range can end here (`stops`: whether the entries arrive in the read's
    /// direction).
    fn offer(
        &mut self,
        tid: Tid,
        tuple: nusadb_core::SharedTuple,
        key: Vec<ast::Value>,
        stops: bool,
    ) -> Result<Flow, Error> {
        crate::cancel::check()?;
        #[cfg(test)]
        CAPPED_ENTRIES_READ.with(|n| n.set(n.get() + 1));
        if self.whole_key && self.full() {
            return Ok(Flow::Stop);
        }
        if stops
            && self.full()
            && let Some(flow) = self.past_the_kept(&key)?
        {
            return Ok(flow);
        }
        if self.in_heap.contains(&tid) || !self.check.holds(&key) {
            return Ok(Flow::More);
        }
        let candidate = Kept {
            key,
            tid,
            tuple,
            backward: self.backward,
        };
        if self.full() {
            // Only a row strictly before the worst kept one takes its place.
            if self.kept.peek().is_none_or(|worst| candidate >= *worst) {
                return Ok(Flow::More);
            }
            if let Some(out) = self.kept.pop() {
                self.in_heap.remove(&out.tid);
            }
        }
        self.in_heap.insert(tid);
        self.kept.push(candidate);
        self.bound = None;
        Ok(Flow::More)
    }

    /// With the heap full, what a row of key `key` read in the scan's direction says about the
    /// rest of its range (see the type docs), or `None` to read on.
    fn past_the_kept(&mut self, key: &[ast::Value]) -> Result<Option<Flow>, Error> {
        let Some(worst) = self.kept.peek() else {
            return Ok(None);
        };
        let worst = &worst.key;
        let low = has_low_char(worst, self.types);
        if self.backward {
            if self.tail_char && !low {
                return Ok(match dml::unique_key_cmp(key, worst) {
                    std::cmp::Ordering::Less => Some(Flow::Stop),
                    std::cmp::Ordering::Equal => Some(Flow::Run),
                    std::cmp::Ordering::Greater => None,
                });
            }
            // Every later entry sits at or below this one, so at or below the ceiling of its
            // key's spellings, and holds a key at or below its own entry.
            let below = spelling_ceiling(key, self.types, true)?;
            if self.bound.is_none() {
                self.bound = Some(index_key::encode_index_key(worst)?);
            }
            return Ok(self
                .bound
                .as_ref()
                .is_some_and(|w| below <= *w)
                .then_some(Flow::Stop));
        }
        if low {
            return Ok(None);
        }
        // Every entry of a key before `worst` sits below `bound`; every later entry sits at or
        // above this row's key encoded bare.
        if self.bound.is_none() {
            self.bound = Some(spelling_ceiling(worst, self.types, false)?);
        }
        let here = index_key::encode_index_key(key)?;
        Ok(self
            .bound
            .as_ref()
            .is_some_and(|b| here >= *b)
            .then_some(Flow::Stop))
    }

    /// Read `from..to` in the scan's direction, a bounded chunk at a time. `runs`: whether a run of
    /// the worst kept key may be stepped over (see [`Flow::Run`]).
    fn range(&mut self, mut from: KeyBound, mut to: KeyBound, runs: bool) -> Result<(), Error> {
        use nusadb_core::engine::ScanDirection;
        let direction = if self.backward {
            ScanDirection::Backward
        } else {
            ScanDirection::Forward
        };
        let ceiling = self
            .cap
            .saturating_mul(2)
            .saturating_add(CHUNK_FIRST_MAX)
            .max(4096);
        let mut chunk = self.cap.saturating_mul(2).clamp(32, CHUNK_FIRST_MAX);
        loop {
            crate::cancel::check()?;
            let mut scan = self.engine.index_scan_directed_limited(
                self.txn,
                self.id,
                from.clone(),
                to.clone(),
                direction,
                Some(chunk),
            )?;
            let mut read = 0usize;
            let mut last: Option<nusadb_core::SharedTuple> = None;
            while let Some((tid, tuple)) = scan.try_next()? {
                read += 1;
                let key = self.check.key_of(&row::decode(&tuple, &self.schema)?);
                last = Some(std::sync::Arc::clone(&tuple));
                match self.offer(tid, tuple, key, true)? {
                    Flow::Stop => return Ok(()),
                    Flow::Run if runs => {
                        drop(scan);
                        return self.between_spellings(&from, &to);
                    },
                    Flow::More | Flow::Run => {},
                }
            }
            drop(scan);
            let Some(last) = last else {
                return Ok(());
            };
            if read < chunk {
                return Ok(());
            }
            let last = self.check.key_of(&row::decode(&last, &self.schema)?);
            // Resume at the far end of where the last row's entry may sit, never past where this
            // chunk started; the rows read again there are kept once.
            let moved = if self.backward {
                let resume = spelling_ceiling(&last, self.types, true)?;
                let next = tighter_upper(to.clone(), resume);
                let moved = next != to;
                to = next;
                moved
            } else {
                let resume = index_key::encode_index_key(&last)?;
                let next = tighter_lower(from.clone(), resume);
                let moved = next != from;
                from = next;
                moved
            };
            if !moved {
                if chunk >= ceiling {
                    // A run of one key longer than any chunk, which no rule above can step over:
                    // read the rest once, as a stream.
                    return self.stream(from, to);
                }
                chunk = chunk.saturating_mul(2).min(ceiling);
            }
        }
    }

    /// The rest of `from..to` read once, forward (the order a range streams in), with every row
    /// offered; a forward read still stops early.
    fn stream(&mut self, from: KeyBound, to: KeyBound) -> Result<(), Error> {
        let mut scan = self.engine.index_scan(self.txn, self.id, from, to)?;
        while let Some((tid, tuple)) = scan.try_next()? {
            let key = self.check.key_of(&row::decode(&tuple, &self.schema)?);
            if matches!(self.offer(tid, tuple, key, !self.backward)?, Flow::Stop) {
                return Ok(());
            }
        }
        Ok(())
    }

    /// After [`Flow::Run`]: the keys still wanted sit between the spellings of the worst kept key
    /// `W`, in the gaps `(W + m blanks, W + m+1 blanks)` of its last column (they hold exactly the
    /// keys that extend it with blanks and then a character below the blank, and none of the
    /// spellings of `W`). Each gap is built when it is read, within `from..to`. Past
    /// [`super::global_unique::MAX_SPELLINGS`] gaps, the rest, up to the value followed by the
    /// character after the blank, is read as one range, `W`'s own wider spellings included.
    fn between_spellings(&mut self, from: &KeyBound, to: &KeyBound) -> Result<(), Error> {
        let Some(worst) = self.kept.peek().map(|k| k.key.clone()) else {
            return Ok(());
        };
        let Some(ColumnType::Char(width)) = self.types.get(worst.len().saturating_sub(1)) else {
            return Ok(());
        };
        let Some(ast::Value::Text(text)) = worst.last() else {
            return Ok(());
        };
        let slack = usize::try_from(*width)
            .unwrap_or(usize::MAX)
            .saturating_sub(text.chars().count());
        let within = |lo: KeyBound, hi: KeyBound| {
            use std::ops::Bound;
            let lo = match (from, lo) {
                (Bound::Included(f) | Bound::Excluded(f), Bound::Excluded(g)) if *f > g => {
                    from.clone()
                },
                (_, lo) => lo,
            };
            let hi = match (to, hi) {
                (Bound::Included(t) | Bound::Excluded(t), Bound::Excluded(g)) if *t < g => {
                    to.clone()
                },
                (_, hi) => hi,
            };
            (lo, hi)
        };
        let mut spelled = worst.clone();
        let mut lower = index_key::encode_index_key(&spelled)?;
        for _ in 0..slack.min(super::global_unique::MAX_SPELLINGS) {
            crate::cancel::check()?;
            if let Some(ast::Value::Text(t)) = spelled.last_mut() {
                t.push(' ');
            }
            let upper = index_key::encode_index_key(&spelled)?;
            let (lo, hi) = within(
                std::ops::Bound::Excluded(std::mem::replace(&mut lower, upper.clone())),
                std::ops::Bound::Excluded(upper),
            );
            self.range(lo, hi, false)?;
        }
        if slack > super::global_unique::MAX_SPELLINGS {
            crate::cancel::check()?;
            if let Some(ast::Value::Text(t)) = spelled.last_mut() {
                t.truncate(t.trim_end_matches(' ').len());
                t.push(AFTER_BLANK);
            }
            let (lo, hi) = within(
                std::ops::Bound::Excluded(lower),
                std::ops::Bound::Excluded(index_key::encode_index_key(&spelled)?),
            );
            self.range(lo, hi, false)?;
        }
        Ok(())
    }
}

/// An encoded bound at or above every spelling of `key` (columns of `types`; with `pad_last`
/// false, the last column is taken as it is). Each `CHAR(n)` value is padded to its width while
/// that adds at most [`super::global_unique::MAX_SPELLINGS`] blanks; the first that would take
/// more is followed by [`AFTER_BLANK`] instead, and the columns after it are left out. So the bound
/// costs at most that many blanks per column, however wide the column.
///
/// With `pad_last` false and `key` holding no character below the blank, it is also above every
/// entry of a key ordered before `key`: a smaller value pads (or is followed) to below the padded
/// (or followed) value, and the last column's smaller values sit below it bare.
fn spelling_ceiling(
    key: &[ast::Value],
    types: &[ColumnType],
    pad_last: bool,
) -> Result<Vec<u8>, Error> {
    let last = key.len().saturating_sub(1);
    let mut out = Vec::with_capacity(key.len());
    for (i, value) in key.iter().enumerate() {
        match (types.get(i), value) {
            (Some(ColumnType::Char(width)), ast::Value::Text(text)) if i < last || pad_last => {
                let slack = usize::try_from(*width)
                    .unwrap_or(usize::MAX)
                    .saturating_sub(text.chars().count());
                let mut text = text.clone();
                if slack > super::global_unique::MAX_SPELLINGS {
                    text.push(AFTER_BLANK);
                    out.push(ast::Value::Text(text));
                    break;
                }
                text.push_str(&" ".repeat(slack));
                out.push(ast::Value::Text(text));
            },
            _ => out.push(value.clone()),
        }
    }
    index_key::encode_index_key(&out)
}

/// Whether a `CHAR(n)` value of `key` holds a character below the blank.
fn has_low_char(key: &[ast::Value], types: &[ColumnType]) -> bool {
    key.iter().zip(types).any(|(value, ty)| {
        matches!((ty, value), (ColumnType::Char(_), ast::Value::Text(text)) if text.chars().any(|c| c < ' '))
    })
}

/// The first `cap` rows, in key order for `direction`, of the legacy index `id` over `ranges`
/// (rows the `check` refuses left out). `whole_key`: the bounds are an equality on the whole key.
///
/// What the read holds is one chunk plus `cap` rows. How far it reads is bounded by where the
/// entries of the rows it keeps can sit (see [`Capped`]), and rows tied with the last one kept are
/// not read past. Two shapes, in either direction, have no such bound and are read through once,
/// as a stream (time proportional to the range, memory still `cap` rows): a run of one key longer
/// than a chunk at the
/// cut of a key with a `CHAR(n)` column before its last, and a cut at a key holding a character
/// below the blank.
#[allow(
    clippy::too_many_arguments,
    reason = "the index, its ranges and their check, and the scan's order, cap, shape and handle"
)]
fn capped_legacy_scan(
    table: &TableSchema,
    id: nusadb_core::IndexId,
    ranges: &[(KeyBound, KeyBound)],
    check: &KeyCheck,
    types: &[ColumnType],
    direction: nusadb_core::engine::ScanDirection,
    cap: usize,
    whole_key: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, nusadb_core::SharedTuple)>, Error> {
    if cap == 0 {
        return Ok(Vec::new());
    }
    let tail_char = types
        .iter()
        .take(types.len().saturating_sub(1))
        .all(|t| !matches!(t, ColumnType::Char(_)));
    let mut read = Capped {
        engine,
        txn,
        id,
        check,
        types,
        schema: column_types(table),
        // Equal keys come in no order, so a whole-key equality reads forward.
        backward: !whole_key && direction == nusadb_core::engine::ScanDirection::Backward,
        cap,
        tail_char,
        whole_key,
        kept: std::collections::BinaryHeap::new(),
        in_heap: HashSet::new(),
        bound: None,
    };
    let single = ranges.len() == 1;
    for (from, to) in ranges {
        if read.whole_key && read.full() {
            break;
        }
        read.range(from.clone(), to.clone(), single && tail_char)?;
    }
    Ok(read
        .kept
        .into_sorted_vec()
        .into_iter()
        .map(|k| (k.tid, k.tuple))
        .collect())
}

/// The tighter of upper bound `to` and `Included(resume)`.
fn tighter_upper(to: KeyBound, resume: Vec<u8>) -> KeyBound {
    use std::ops::Bound;
    match to {
        Bound::Included(end) if end <= resume => Bound::Included(end),
        Bound::Excluded(end) if end <= resume => Bound::Excluded(end),
        _ => Bound::Included(resume),
    }
}

/// The tighter of lower bound `from` and `Included(resume)`.
fn tighter_lower(from: KeyBound, resume: Vec<u8>) -> KeyBound {
    use std::ops::Bound;
    match from {
        Bound::Included(start) if start >= resume => Bound::Included(start),
        Bound::Excluded(start) if start >= resume => Bound::Excluded(start),
        _ => Bound::Included(resume),
    }
}

/// The index ranges [`legacy_index_scan`] reads for the bounds `lo` / `hi` on a legacy index
/// over `key_columns` columns of `types`: every spelling of an equality, else the range from the
/// lower bound as written to past every spelling of the upper bound (below an excluded upper bound
/// whose only `CHAR(n)` column is its last, the bound itself).
fn legacy_ranges(
    lo: &std::ops::Bound<Vec<ast::Value>>,
    hi: &std::ops::Bound<Vec<ast::Value>>,
    key_columns: usize,
    types: &[ColumnType],
) -> Result<Vec<(KeyBound, KeyBound)>, Error> {
    use std::ops::Bound;
    const PAST_PREFIX: u8 = 0x02;
    Ok(match (lo, hi) {
        (Bound::Included(a), Bound::Included(b))
            if a.len() == b.len() && dml::unique_key_cmp(a, b) == std::cmp::Ordering::Equal =>
        {
            let prefix = a.len() < key_columns;
            super::global_unique::probe_ranges(a, types.get(..a.len()).unwrap_or(types))?
                .into_iter()
                .map(|(from, to)| match (prefix, to) {
                    (true, Bound::Included(mut end)) => {
                        end.push(PAST_PREFIX);
                        (from, Bound::Excluded(end))
                    },
                    (_, to) => (from, to),
                })
                .collect()
        },
        _ => {
            let from = match lo {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(v) | Bound::Excluded(v) => {
                    Bound::Included(index_key::encode_index_key(v)?)
                },
            };
            let to = match hi {
                Bound::Unbounded => Bound::Unbounded,
                // Past the ceiling of its spellings (see `spelling_ceiling`): every spelling of a
                // key up to it sorts below. A value holding a character below the blank would not
                // (a smaller key can pad to above it), so such a bound reads to the end instead.
                // Below a value whose only `CHAR(n)` column is its last, and which does not end
                // in a blank, every spelling of a smaller key sits below the value itself, bare
                // (`'a'` is below `'a '`, yet its spelling `'a  '` sits above it).
                Bound::Excluded(v)
                    if !has_low_char(v, types)
                        && !matches!(v.last(), Some(ast::Value::Text(t)) if t.ends_with(' '))
                        && types
                            .iter()
                            .take(v.len().saturating_sub(1))
                            .all(|t| !matches!(t, ColumnType::Char(_))) =>
                {
                    Bound::Excluded(index_key::encode_index_key(v)?)
                },
                Bound::Included(v) | Bound::Excluded(v) => {
                    if has_low_char(v, types) {
                        Bound::Unbounded
                    } else {
                        let mut end = spelling_ceiling(v, types, true)?;
                        end.push(PAST_PREFIX);
                        Bound::Excluded(end)
                    }
                },
            };
            vec![(from, to)]
        },
    })
}

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
