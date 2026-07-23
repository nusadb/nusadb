//! The treaty implementation: tables as clustered B-link trees, **MVCC with undo versions and
//! read views**.
//!
//! Every leaf value carries a [`RowMeta`] header (`xmin`, `xmax`, undo pointer); superseded
//! versions live in the engine's undo arena; readers resolve visibility through a [`ReadView`]
//! (see [`crate::mvcc`]). Snapshot discipline matches the predecessor engine's (removed 2026-07-09; its semantics were kept so the engine swap changed no observable behavior): `READ COMMITTED` /
//! `READ UNCOMMITTED` take a **fresh view at every read** (statement-level), `REPEATABLE READ` /
//! `SERIALIZABLE` pin the view taken at `BEGIN`. `REPEATABLE READ` is snapshot isolation (write
//! skew permitted — snapshot isolation's documented contract). `SERIALIZABLE` adds a **row-level read-write
//! antidependency check**: a transaction records the rows it
//! reads and, at commit, aborts (40001) if any was modified by a concurrent transaction that
//! committed after its snapshot — preventing write-skew over existing rows (the Hermitage `G2`
//! anomaly). Predicate/phantom antidependencies over not-yet-existing rows are the further
//! further SSI refinement (predicate-level read tracking); row-level SSI is the shipped contract. Write-write
//! conflicts are **no-wait**
//! and **first-updater-wins at every isolation level** (the OCC discipline):
//! writing over a row whose newest version was written by a concurrent (still-active) transaction,
//! **or by any transaction this one's `BEGIN` snapshot cannot see** (i.e. committed after it
//! began), raises `SerializationConflict` (SQLSTATE 40001) instead of blocking or silently
//! last-writer-wins. Reads stay per-level (`READ COMMITTED` sees the latest committed value); only
//! write admission consults the begin snapshot, so a `v = v + 1` computed from a now-stale read
//! aborts-and-retries rather than losing the concurrent update.
//!
//! `ROLLBACK` / `ROLLBACK TO SAVEPOINT` restore the exact previous **encoded** leaf entries
//! (header included), so an aborted transaction leaves no version behind — which is precisely
//! what lets [`ReadView::sees`] equate "ended and present" with "committed".
//!
//! Remaining phase limitations (each owned later): not durable, no secondary-index treaty
//! methods, the undo arena and deleted rows are reclaimed by purge, and a tuple must
//! fit one leaf after the [`mvcc::META`] header.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Seek;
use std::ops::Bound;
use std::path::Path;
#[cfg(feature = "dst-fault")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use nusadb_core::engine::{
    AlterOp, IndexDef, IndexKind, IsolationLevel, RowLockMode, SequenceDef, SharedTuple, TableDef,
    TableLockMode, TableStats, Tid, TupleScan,
};
use nusadb_core::{
    Constraint, ConstraintKind, Error, FkAction, ForeignKeyDef, IndexId, PageStore, Result,
    SchemaId, SequenceId, SlotIdx, TableId, TableSchema, TxnId,
};
use nusadb_wal::{WalRecord, WalWriter};

use crate::mvcc::{self, ReadView, RowMeta, UndoVersion};
use crate::node;
use crate::store::MemPageStore;
use crate::tree::ClusteredTree;
use crate::wal::LoggedOp;

/// The largest user tuple the engine accepts: one leaf entry minus the MVCC header.
const MAX_USER_TUPLE: usize = node::MAX_TUPLE - mvcc::META;

/// Bytes charged per written row *on top of* its logical tuple length, so the per-transaction write
/// ceiling reflects the row's real retained footprint rather than only its logical bytes. Each write
/// keeps, until commit: the stored version's MVCC header ([`mvcc::META`], 24 B), a page slot entry
/// plus page fragmentation, and an undo-log record for rollback. For a narrow row this fixed cost
/// dominates — charging only `tuple.len()` under-counts the true footprint several-fold, letting
/// millions of tiny rows accumulate before the ceiling trips (that was the residual `COPY`/narrow-row
/// OOM path after streaming fixed the parse side). Charging a deliberately conservative over-estimate
/// makes the ceiling abort early (safe) rather than late (OOM); it is a safety bound, not an exact
/// accountant.
#[allow(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is required so the crate's #[cfg(test)] modules — siblings of this private \
              `engine` module, not descendants — can reference the charge overhead to compute test \
              ceilings; the lint misfires because the enclosing module is private"
)]
pub(crate) const PER_ROW_WRITE_OVERHEAD: u64 = mvcc::META as u64 + 40;

/// How many rows one incremental [`BtreeEngine::purge`] batch reclaims before releasing the table
/// writer latch and reclamation gate, so concurrent writers interleave instead of stalling for a
/// whole pass. Bounds the per-batch latch hold to a few milliseconds at the measured
/// ~2 microseconds-per-version reclaim cost, while keeping the per-batch re-descend overhead
/// negligible against the reclaim work.
#[allow(
    clippy::redundant_pub_crate,
    reason = "pub(crate) so the crate's multi-batch purge tests can size a table past one batch"
)]
pub(crate) const PURGE_ROW_BATCH: usize = 4096;

/// The durable-log handle: the framing writer plus a second handle to the same file for the
/// commit-point fsync (the writer owns its handle exclusively). The fsync handle is shared behind an
/// `Arc` so a committer can take it out to fsync outside the writer lock with a cheap reference-count
/// bump, rather than duplicating the file descriptor (a syscall pair) on every commit.
struct Wal {
    writer: WalWriter<File>,
    sync: Arc<File>,
}

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal").finish_non_exhaustive()
    }
}

/// The clustered B-link/B+tree engine (MVCC read views + durable redo WAL; sharded latching).
///
/// [`BtreeEngine::new`] is in-memory (tests, scratch work); [`BtreeEngine::open`] is durable —
/// every committed transaction survives `kill -9` (see [`crate::wal`]). See the crate docs for
/// the design shape and the per-phase limitations.
///
/// # Latching discipline (the global `Mutex<State>` is gone)
///
/// State is sharded into independently locked domains; an operation takes only the domains it
/// touches, each for the shortest span that preserves its invariant. To make deadlock impossible
/// every path acquires **nested** locks in this fixed rank order (taking a later-rank lock while
/// holding an earlier one is allowed; the reverse is never done — sequential acquire-release of
/// any ranks is always fine):
///
/// 1. `commit_gate` — commits only
/// 2. `catalog` (`RwLock`) — DML/readers hold `read` for the whole call (schema stability),
///    DDL holds `write` (drains every in-flight operation)
/// 3. `TableState::write` (per table) — tree writes; never two tables at once
/// 4. `IndexState::data` (per index, `RwLock`) — never two indexes at once
/// 5. `dropped` — the purge queue of dropped trees
/// 6. `txns` — transaction + lock manager (O(1) critical sections)
/// 7. `seqs` — sequences
/// 8. `reclaim` (`RwLock`) — the undo arena, doubling as the **reclamation gate**: every
///    chain-walking reader holds `read` across its walk; purge holds `write` while freeing
///    arena slots or deallocating dropped trees' pages, so a stale leaf pointer can never chase
///    a recycled slot and a latch-free scan can never touch a freed page
/// 9. `wal` — appends; **every logged operation appends while still holding the latch of the
///    object it mutated** (table/index/catalog/sequence), so per-object log order equals apply
///    order and replay converges
///
/// Readers never latch trees: B-link descent (right-link chase, Lehman–Yao publish order) keeps
/// a concurrent split structurally safe, page reads/writes are atomic in the store, and MVCC
/// stamps hide uncommitted versions. A transaction leaves `active` only **after** its outcome is
/// fully applied (commit: after the group fsync; abort: after the undo completes) — a
/// [`ReadView`] equates "ended and present" with committed, so the order is load-bearing.
#[derive(Debug, Default)]
pub struct BtreeEngine {
    store: MemPageStore,
    /// Rank 2: tables, indexes, constraints, namespaces, stats — the schema. DML holds `read`
    /// (fully parallel), DDL holds `write`.
    catalog: RwLock<Catalog>,
    /// Rank 5: trees of committed-dropped tables awaiting page reclamation by purge. An
    /// entry is removed on rollback (the drop was undone) or once purge frees the pages.
    dropped: Mutex<Vec<DroppedPages>>,
    /// Rank 6: the transaction + lock manager.
    txns: Mutex<TxnDomain>,
    /// Rank 7: sequences. **Non-transactional** counters: every advance is fsynced to
    /// the log before the value escapes, and rollback never rewinds one (gap semantics).
    seqs: Mutex<SeqDomain>,
    /// Rank 8: the undo arena behind the reclamation gate (see the struct docs).
    reclaim: RwLock<UndoDomain>,
    /// Rank 1: makes [`SERIALIZABLE` antidependency check → commit-marker append → `staged`
    /// insert] atomic across committers. Without it two symmetric write-skew transactions could
    /// each pass the check before either stages — the check must observe every earlier
    /// committer as staged or committed.
    commit_gate: Mutex<()>,
    /// `None` = in-memory engine; `Some` = durable, logging to the WAL file (rank 9).
    wal: Option<Mutex<Wal>>,
    /// Coalesces concurrent committers' `fsync`s into shared ones: the durability
    /// point runs OUTSIDE every engine latch, so one `fsync` serves every commit staged while
    /// it was in flight — the fix for durable write throughput shrinking as workers grow.
    group: nusadb_wal::GroupCommit,
    /// Monotonic committed-data-change counter: bumped by every commit that wrote, so
    /// the SQL result cache can validate a cached result cheaply. Never persisted — recovery
    /// restarting it at zero is fine because the cache is empty then too.
    data_version: AtomicU64,
    /// Optional ceiling (bytes) on one transaction's uncommitted row writes. `None` (the default)
    /// imposes no limit and leaves behavior unchanged; `Some(limit)` makes a transaction whose row
    /// writes would exceed `limit` fail loudly with [`Error::OutOfMemory`] and abort — so a single
    /// oversized transaction (e.g. a multi-million-row bulk load into the in-memory page store)
    /// cannot grow until the OS OOM-kills the whole server, taking every client down with it. Set
    /// once at construction via [`BtreeEngine::with_max_txn_write_bytes`].
    max_txn_write_bytes: Option<u64>,
    /// Optional ceiling (bytes) on the in-memory page store's total resident footprint. `None` (the
    /// default) imposes no limit and leaves behavior unchanged; `Some(limit)` makes a row `insert`
    /// that would grow the store past `limit` fail loudly with [`Error::OutOfMemory`] and abort the
    /// transaction — so a bulk load bigger than RAM (e.g. a multi-million-row COPY streamed as many
    /// committed batches, each under the per-transaction ceiling but accumulating resident) is
    /// rejected gracefully instead of growing until the OS OOM-kills the whole server. Unlike the
    /// per-transaction ceiling this bounds *committed-resident* data across the whole store. Only
    /// `insert` (the monotonic-growth path) is gated, so `DELETE`/`TRUNCATE` stay available to free
    /// space at the ceiling. Set once at construction via
    /// [`BtreeEngine::with_max_total_resident_bytes`].
    max_total_resident_bytes: Option<u64>,
    /// DST fault point (compiled only under the `dst-fault` feature — never in production
    /// builds): when armed, the next group-leader fsync reports failure AFTER the buffer
    /// reached the file, modeling the fsyncgate shape (the kernel had the bytes, `fsync`
    /// said no, the record can still hit disk) that black-box fault injection cannot time —
    /// a device-level error always breaks the *append* first, taking the recoverable path.
    #[cfg(feature = "dst-fault")]
    dst_fail_next_fsync: AtomicBool,
    /// DST fault point (compiled only under `dst-fault`): when armed, the next WAL append reports
    /// ENOSPC (`StorageFull`) WITHOUT writing the record — modeling a disk-full write syscall that
    /// fails *before* acknowledging. This is the categorically different shape from
    /// `dst_fail_next_fsync` (there the bytes reached the file; here nothing does), and it drives
    /// the commit/abort disk-full paths: a failed commit-marker append must leave nothing staged
    /// and roll the transaction back cleanly, never stranding it in `active`. One-shot: the flag
    /// clears when it fires, so recovery and later commits append normally.
    #[cfg(feature = "dst-fault")]
    dst_fail_next_wal_append: AtomicBool,
}

/// The schema domain (rank 2): everything DDL-shaped. The `RwLock` around it is the
/// schema-stability latch — a DML call holds `read` for its whole span, so a table or index it
/// resolved cannot be dropped from under it; DDL takes `write` and thereby drains every
/// in-flight operation.
#[derive(Debug, Default)]
struct Catalog {
    tables: HashMap<u64, TableState>,
    by_name: HashMap<(String, String), u64>,
    next_table_id: u64,
    /// Secondary indexes by id: sorted entries `key bytes → row-ids`, payload = row-id.
    indexes: HashMap<u64, IndexState>,
    idx_by_name: HashMap<String, u64>,
    next_index_id: u64,
    /// `PRIMARY KEY` / `UNIQUE` constraints per table (the catalog family): each is backed
    /// by a unique index whose byte-level check is exempted (the SQL layer's scan-based
    /// checks own the constraint semantics).
    constraints: HashMap<u64, Vec<UniqueState>>,
    /// `CHECK` constraints per table: name + opaque predicate bytes (the SQL layer evaluates).
    checks: HashMap<u64, Vec<CheckState>>,
    /// `FOREIGN KEY`s by (globally unique) constraint name.
    foreign_keys: HashMap<String, FkState>,
    /// `ANALYZE` statistics per table (opaque per-column bytes; the engine never decodes them).
    stats: HashMap<u64, TableStats>,
    /// SQL schemas (namespaces) by id → name. Rollback-aware DDL.
    namespaces: HashMap<u64, String>,
    ns_by_name: HashMap<String, u64>,
    next_namespace_id: u64,
}

/// The transaction + lock manager (rank 6). Every critical section over it is O(1)-ish (map
/// lookups, a push) — never a tree walk.
#[derive(Debug)]
struct TxnDomain {
    txns: HashMap<u64, TxnState>,
    /// Transactions begun and not yet ended — the raw material of every read view.
    active: HashSet<u64>,
    next_txn_id: u64,
    /// The no-wait lock table (`LOCK TABLE` · `FOR UPDATE/SHARE` · uniqueness
    /// keys): row, key and table locks in distinct namespaces, held until the owning transaction
    /// ends. A conflict aborts (40001) immediately — never waits — so there is no deadlock to
    /// detect (the same no-wait discipline as write admission).
    locks: HashMap<LockId, LockHolders>,
    /// Transactions whose commit marker is appended but whose group `fsync` has not returned.
    /// Still in `active` too, so no view sees their writes and no writer overtakes them —
    /// but a `SERIALIZABLE` antidependency check must count them as committed: their commit
    /// record is already ordered ahead of any later committer's in the log.
    staged: HashSet<u64>,
    /// Per-table write versions, STAGED instant (SSI narrowing): bumped when a transaction
    /// that wrote the table stages its commit marker — the instant it starts counting as
    /// committed for the antidependency check. Compared at a reader's COMMIT.
    table_write_versions_staged: HashMap<u64, u64>,
    /// Per-table write versions, FINISHED instant: bumped when the transaction's writes become
    /// visible to new readers (`finish_commit`). Snapshotted at a `SERIALIZABLE` reader's
    /// BEGIN. The skip fires only when `staged_now == finished_at_begin` — a writer anywhere in
    /// the staged-but-unfinished window (whose writes the reader cannot see but whose commit
    /// already outranks the reader's) makes the two differ, forcing full validation. Audit
    /// catch: a single stage-time map let a reader that began during a writer's fsync inherit
    /// the bump into its baseline while not seeing the rows — hiding a write-skew abort.
    table_write_versions_finished: HashMap<u64, u64>,
}

impl UndoOp {
    /// The table whose ROW STATE this op mutated (a stamp the antidependency check can see), or
    /// `None` for catalog/index-entry ops — an index-entry move always rides a row op on the
    /// same table, and pure DDL leaves no row stamps to conflict on.
    const fn row_table(&self) -> Option<u64> {
        match self {
            Self::Inserted { table, .. }
            | Self::Updated { table, .. }
            | Self::Deleted { table, .. } => Some(*table),
            _ => None,
        }
    }
}

impl TxnDomain {
    /// The tables whose ROW state `txn` mutated (per its undo), or empty for a reader.
    fn touched_tables(&self, txn: u64) -> std::collections::HashSet<u64> {
        self.txns
            .get(&txn)
            .map(|state| state.undo.iter().filter_map(UndoOp::row_table).collect())
            .unwrap_or_default()
    }

    /// Bump the STAGED write version of every table `txn` wrote — at the instant its commit
    /// marker is appended (it starts counting as committed for the antidependency check).
    fn bump_staged_versions(&mut self, txn: u64) {
        for table in self.touched_tables(txn) {
            *self.table_write_versions_staged.entry(table).or_insert(0) += 1;
        }
    }

    /// Bump the FINISHED write version of every table `txn` wrote — at the instant its writes
    /// become visible to new readers. Must be called while `txn`'s state (and its undo) is
    /// still present.
    fn bump_finished_versions(&mut self, txn: u64) {
        for table in self.touched_tables(txn) {
            *self.table_write_versions_finished.entry(table).or_insert(0) += 1;
        }
    }
}

impl Default for TxnDomain {
    fn default() -> Self {
        Self {
            txns: HashMap::new(),
            active: HashSet::new(),
            // Transaction id 0 is reserved: `mvcc::NO_XMAX` (= 0) marks a live version, so a
            // real transaction may never stamp an xmax of 0.
            next_txn_id: 1,
            locks: HashMap::new(),
            staged: HashSet::new(),
            table_write_versions_staged: HashMap::new(),
            table_write_versions_finished: HashMap::new(),
        }
    }
}

/// The sequence domain (rank 7): its own latch so a `nextval` burst's fsyncs never stall row
/// writers.
#[derive(Debug, Default)]
struct SeqDomain {
    sequences: HashMap<u64, SequenceState>,
    seq_by_name: HashMap<String, u64>,
    next_sequence_id: u64,
}

/// The undo arena (rank 8): superseded row versions addressed by [`RowMeta::undo`]; `None` =
/// slot freed by purge, freed indices recycled through `free`. The `RwLock` around it is
/// the reclamation gate (see [`BtreeEngine`]'s latching docs).
#[derive(Debug, Default)]
struct UndoDomain {
    arena: Vec<Option<UndoVersion>>,
    free: Vec<u64>,
    /// Slots orphaned by an aborted `UPDATE` — the parked version its rollback disconnected
    /// from every chain — queued as `(slot, aborting txn)` for purge to free once the abort is
    /// SETTLED. Freeing eagerly at undo time would race a chain-walking reader that read the
    /// not-yet-rolled-back leaf and still needs the parked slot to reach its visible version;
    /// once no view concurrent with the aborting transaction remains, no such walk can exist.
    orphans: Vec<(u64, u64)>,
}

/// A borrowed catalog guard for the undo path: row/index undo needs only `read` (per-object
/// latches do the real exclusion), DDL undo needs `write`. The caller picks the guard by
/// inspecting the ops ([`BtreeEngine::undo_needs_catalog_write`]), so `get_mut` failing is a
/// logic error surfaced loudly, never a panic.
enum CatalogRef<'a> {
    Read(&'a Catalog),
    Write(&'a mut Catalog),
}

impl CatalogRef<'_> {
    const fn get(&self) -> &Catalog {
        match self {
            Self::Read(c) => c,
            Self::Write(c) => c,
        }
    }

    fn get_mut(&mut self) -> Result<&mut Catalog> {
        match self {
            Self::Write(c) => Ok(c),
            Self::Read(_) => Err(Error::Io(std::io::Error::other(
                "nusadb-btree: DDL undo reached under a catalog read guard (internal bug)",
            ))),
        }
    }
}

/// Identity of one lockable item. The variants are distinct namespaces, so a table lock, a row
/// lock and a uniqueness-key lock on the same table never alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LockId {
    /// A specific row (`SELECT ... FOR UPDATE/SHARE`).
    Row { table: u64, page: u64, slot: u16 },
    /// A logical unique-key value — serializes concurrent `UNIQUE`/`PRIMARY KEY` writers of the
    /// same key so a snapshot-based uniqueness scan cannot admit two duplicates.
    Key { table: u64, hash: u64 },
    /// The whole table (`LOCK TABLE`). Shared acquisition doubles as the intention lock
    /// under row/key locks and row writes, so `ACCESS EXCLUSIVE` sees all concurrent activity.
    Table { table: u64 },
}

/// Which transactions hold one lock: holder id → whether the hold is exclusive. Shared holders
/// coexist; an exclusive holder is alone.
#[derive(Debug, Default)]
struct LockHolders {
    holders: HashMap<u64, bool>,
}

#[derive(Debug)]
struct TableState {
    schema: TableSchema,
    /// The current root page (rank-free: an atomic, not a latch). It moves only when the root
    /// splits, and an old root stays a valid B-link entry point (its right links cover every
    /// key at or beyond its high key), so a reader loading a just-stale root still lands on the
    /// right leaf — readers enter the tree without any latch.
    root: AtomicU64,
    /// `O(1)` approximate live-row count for plan-time routing (see
    /// [`StorageEngine::approx_row_count`](nusadb_core::StorageEngine::approx_row_count)). Starts
    /// [`UNINIT`](TableState::APPROX_UNINIT); the first read fills it from an `O(n)` [`row_count`]
    /// walk (post-restart the in-memory counter is 0 but the tree may hold rows), and each commit
    /// then maintains it by the transaction's net `inserted − deleted`. A routing hint only, never a
    /// correctness input, so a slightly stale value is fine.
    approx_rows: AtomicU64,
    /// Absolute write churn since this table's stats were last refreshed by `ANALYZE`: the count of
    /// row operations (`inserts + updates + deletes`) each commit applies, reset to `0` when
    /// [`analyze_table`](StorageEngine::analyze_table) stores fresh stats. Unlike
    /// [`approx_rows`](TableState::approx_rows) this is *absolute*, not net — a table churned heavily
    /// but kept the same size still needs re-analysing, so an update and an insert+delete both count.
    /// Consumed by auto-analyze to decide when the planner's histogram/MCV statistics have gone stale
    /// (D-AUTO-ANALYZE). A hint only — a slightly stale value never affects correctness.
    churn_since_analyze: AtomicU64,
    /// Rank 3 — the per-table writer latch: tree mutations, row-id minting, and the WAL append
    /// of each row op run under it, so same-table writes (and their log records) are totally
    /// ordered while different tables proceed in parallel.
    write: Mutex<TableWrite>,
    /// The current schema version — bumped by every `ALTER TABLE`.
    schema_version: u32,
    /// Every schema version this table has had, so a row written under an older one stays
    /// resolvable (the SQL layer eagerly rewrites rows on ALTER, so in practice only the current
    /// version carries live rows; the history still backs `schema_for_version`).
    schema_history: HashMap<u32, TableSchema>,
}

impl TableState {
    /// [`approx_rows`](TableState::approx_rows) sentinel: not yet initialized (no row-count walk has
    /// run). `u64::MAX` is unreachable as a real live-row count, so it is an unambiguous marker.
    const APPROX_UNINIT: u64 = u64::MAX;

    /// The tree's current root as a [`nusadb_core::PageId`].
    fn root_id(&self) -> nusadb_core::PageId {
        nusadb_core::PageId(self.root.load(Ordering::Acquire))
    }

    /// Publish a (possibly moved) root after a tree mutation.
    fn set_root(&self, root: nusadb_core::PageId) {
        self.root.store(root.0, Ordering::Release);
    }

    /// The raw approximate-count word, or [`APPROX_UNINIT`](Self::APPROX_UNINIT) if never filled.
    /// Relaxed: an approximate routing hint needs no ordering.
    fn approx_rows_raw(&self) -> u64 {
        self.approx_rows.load(Ordering::Relaxed)
    }

    /// Fill the counter with `counted` (an `O(n)` walk's result) **only if still uninitialized** — so
    /// a delta that a commit applied between the walk and here is never clobbered by the stale count.
    /// A concurrent initializer computes the same count, so the loser's CAS simply no-ops.
    fn init_approx_rows(&self, counted: u64) {
        let _ = self.approx_rows.compare_exchange(
            Self::APPROX_UNINIT,
            counted,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Apply a committed transaction's net row change (`inserted − deleted`) to the approximate
    /// count — but only once it has been initialized (an uninitialized counter stays `UNINIT` so the
    /// first read still does a full walk that already reflects every committed write). Saturating, so
    /// the estimate never wraps past `0` if concurrent deltas race.
    fn add_approx_delta(&self, delta: i64) {
        if delta == 0 {
            return;
        }
        let _ = self
            .approx_rows
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != Self::APPROX_UNINIT).then(|| current.saturating_add_signed(delta))
            });
    }

    /// The write churn accumulated since the last `ANALYZE`. Relaxed: a staleness hint needs no
    /// ordering.
    fn churn_raw(&self) -> u64 {
        self.churn_since_analyze.load(Ordering::Relaxed)
    }

    /// Add a committed transaction's absolute row-operation count to the churn tally. Saturating, so
    /// a pathological counter never wraps.
    fn add_churn(&self, ops: u64) {
        if ops == 0 {
            return;
        }
        self.churn_since_analyze.fetch_add(ops, Ordering::Relaxed);
    }

    /// Reset the churn tally — the statistics are now fresh (called when `ANALYZE` stores them).
    fn reset_churn(&self) {
        self.churn_since_analyze.store(0, Ordering::Relaxed);
    }
}

/// The mutable per-table write state guarded by [`TableState::write`].
#[derive(Debug, Default)]
struct TableWrite {
    next_row_id: u64,
}

/// A dropped table's tree, queued for purge: pages are only reclaimed once the dropping
/// transaction is settled (committed and visible to every view).
#[derive(Debug)]
struct DroppedPages {
    txn: u64,
    root: nusadb_core::PageId,
}

/// A sequence: its definition and the last value handed out (`None` before the first
/// `nextval`).
#[derive(Debug, Clone)]
struct SequenceState {
    def: SequenceDef,
    current: Option<i64>,
}

/// Advance `seq` one step — `start` first, then `current + increment`, wrapping to the opposite
/// bound when cycling, erring when exhausted (the sequence contract carried over unchanged from the predecessor engine).
fn advance_sequence(seq: &mut SequenceState) -> Result<i64> {
    let d = &seq.def;
    let next = match seq.current {
        None => d.start,
        Some(cur) => match cur.checked_add(d.increment) {
            Some(v) if d.increment >= 0 && v <= d.max_value => v,
            Some(v) if d.increment < 0 && v >= d.min_value => v,
            _ if d.cycle => {
                if d.increment >= 0 {
                    d.min_value
                } else {
                    d.max_value
                }
            },
            _ => return Err(sequence_error("sequence reached its limit")),
        },
    };
    seq.current = Some(next);
    Ok(next)
}

fn sequence_error(msg: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        msg.to_owned(),
    ))
}

fn sequence_not_found(id: SequenceId) -> Error {
    sequence_error(&format!("sequence {} not found", id.0))
}

/// Apply one `ALTER TABLE` action to `schema` in place (validation rules carried over unchanged from the predecessor engine).
fn apply_alter(schema: &mut TableSchema, op: &AlterOp) -> Result<()> {
    match op {
        AlterOp::AddColumn(col) => {
            if schema.columns.iter().any(|c| c.name == col.name) {
                return Err(alter_error(&format!("column {} already exists", col.name)));
            }
            schema.columns.push(col.clone());
        },
        AlterOp::DropColumn { name } => {
            let before = schema.columns.len();
            schema.columns.retain(|c| &c.name != name);
            if schema.columns.len() == before {
                return Err(alter_error(&format!("column {name} not found")));
            }
        },
        AlterOp::RenameColumn { from, to } => {
            if from != to && schema.columns.iter().any(|c| &c.name == to) {
                return Err(alter_error(&format!("column {to} already exists")));
            }
            let col = schema
                .columns
                .iter_mut()
                .find(|c| &c.name == from)
                .ok_or_else(|| alter_error(&format!("column {from} not found")))?;
            col.name.clone_from(to);
        },
        AlterOp::RenameTable { name } => {
            schema.name.clone_from(name);
        },
        AlterOp::AlterColumnType { column, ty } => {
            let col = schema
                .columns
                .iter_mut()
                .find(|c| &c.name == column)
                .ok_or_else(|| alter_error(&format!("column {column} not found")))?;
            col.ty = *ty;
        },
        AlterOp::SetNotNull { column } | AlterOp::DropNotNull { column } => {
            let col = schema
                .columns
                .iter_mut()
                .find(|c| &c.name == column)
                .ok_or_else(|| alter_error(&format!("column {column} not found")))?;
            col.nullable = matches!(op, AlterOp::DropNotNull { .. });
        },
    }
    Ok(())
}

fn alter_error(msg: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("nusadb-btree: {msg}"),
    ))
}

fn schema_error(msg: &str) -> Error {
    // Same shape as `alter_error` (both are InvalidInput DDL errors) — kept as a distinct name
    // for call-site readability.
    alter_error(msg)
}

fn schema_not_found(id: SchemaId) -> Error {
    schema_error(&format!("schema id {} not found", id.0))
}

/// A declared `PRIMARY KEY` / `UNIQUE` constraint: the catalog record beside its backing index.
#[derive(Debug, Clone)]
struct UniqueState {
    name: String,
    columns: Vec<String>,
    primary: bool,
    index: u64,
}

/// A declared `CHECK` constraint: name + the SQL layer's opaque predicate bytes.
#[derive(Debug, Clone)]
struct CheckState {
    name: String,
    expr: Vec<u8>,
}

/// A declared `FOREIGN KEY`: child/parent linkage plus the two backing indexes.
#[derive(Debug, Clone)]
struct FkState {
    name: String,
    child_table: u64,
    child_columns: Vec<String>,
    parent_table: u64,
    /// The parent's PK/UNIQUE backing index the FK resolves referenced keys against.
    parent_index: u64,
    /// The child-side (non-unique) index the SQL layer maintains on child writes.
    child_index: u64,
    on_delete: FkAction,
    on_update: FkAction,
}

/// A secondary index: the catalog definition plus its sorted entries. The map is ordered
/// by the opaque key bytes, so a range scan walks it in ascending key order; each key maps to
/// the row-ids carrying it (non-unique indexes may hold several). Each entry carries its own
/// MVCC stamps (`xmin`/`xmax`): the SQL layer never deletes an entry when an `UPDATE` moves a
/// row to a new key (it only inserts the new one), and the row keeps its address across
/// versions, so the base row alone cannot tell a reader which KEY its visible version carries —
/// the entry stamps do. `index_insert` under a new key dead-stamps the row's previous alive
/// entry in the same index (one alive key per row per index), and `index_scan` filters entries
/// by the caller's view before resolving the base row (2-hop, ADR 008 §D2).
#[derive(Debug)]
struct IndexState {
    def: IndexDef,
    /// Whether every live row has its entry (the coverage promise). True from birth: the
    /// creating statement backfills in the same transaction, and every later write maintains
    /// the entries. Never flips after creation, so it lives outside the data latch.
    complete: bool,
    /// Rank 4 — the per-index latch: scans hold `read`, and every entry mutation (including its
    /// uniqueness pre-check and WAL append, which must be atomic with the apply) holds `write`.
    data: RwLock<IndexData>,
}

/// One index's entries, guarded by [`IndexState::data`].
#[derive(Debug, Default)]
struct IndexData {
    entries: BTreeMap<Vec<u8>, BTreeMap<u64, Vec<EntryMeta>>>,
    /// The one alive (not dead-stamped) key per row — the reverse map `index_insert` consults to
    /// stamp a row's previous key in O(1).
    alive: HashMap<u64, Vec<u8>>,
}

/// One visibility range of an index entry: the transaction that created it and (if dead-stamped)
/// the one that superseded it — `mvcc::NO_XMAX` while alive. A `(key, row)` slot holds a small
/// vec of these (almost always one): a row whose key moves away and later moves **back** earns a
/// second disjoint range, so a snapshot pinned before the first move and one taken after the
/// second each find the range their version of the row belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EntryMeta {
    xmin: u64,
    xmax: u64,
}

/// What one [`IndexState::apply_insert`] actually did — recorded by the caller so the inverse is
/// exact.
enum AppliedInsert {
    /// The row's alive entry already carries this key (a same-key `UPDATE` re-insert: the old and
    /// new row versions share the key, and the existing entry — whose `xmin` every reader of
    /// either version already sees — serves both). Nothing changed; nothing to undo
    /// (overwriting the committed entry here, as the first
    /// draft did, loses its `xmin` for concurrent readers and its identity for rollback).
    Noop,
    /// A new alive range was pushed; `stamped` is the old key whose alive range was dead-stamped
    /// (a key move), if any.
    Inserted { stamped: Option<Vec<u8>> },
}

impl IndexData {
    /// Whether the entry with ranges `metas` is visible under `view`: any range visible.
    fn entry_visible(metas: &[EntryMeta], view: &ReadView) -> bool {
        metas
            .iter()
            .any(|m| view.sees(m.xmin) && (m.xmax == mvcc::NO_XMAX || !view.sees(m.xmax)))
    }

    /// Apply one entry insert by `txn`. A same-key re-insert over the row's alive entry is a
    /// no-op; a key move dead-stamps the old key's alive range and pushes a fresh range under the
    /// new key (appending, never overwriting, so a re-used key keeps its older ranges for pinned
    /// snapshots). Shared by the live path and WAL replay, so recovery re-derives the same state.
    fn apply_insert(&mut self, key: &[u8], row_id: u64, txn: u64) -> AppliedInsert {
        let stamped = match self.alive.get(&row_id) {
            Some(old_key) if old_key.as_slice() == key => return AppliedInsert::Noop,
            Some(old_key) => {
                let old_key = old_key.clone();
                if let Some(meta) = self
                    .entries
                    .get_mut(&old_key)
                    .and_then(|rows| rows.get_mut(&row_id))
                    .and_then(|metas| metas.iter_mut().rfind(|m| m.xmax == mvcc::NO_XMAX))
                {
                    meta.xmax = txn;
                }
                Some(old_key)
            },
            None => None,
        };
        self.entries
            .entry(key.to_vec())
            .or_default()
            .entry(row_id)
            .or_default()
            .push(EntryMeta {
                xmin: txn,
                xmax: mvcc::NO_XMAX,
            });
        self.alive.insert(row_id, key.to_vec());
        AppliedInsert::Inserted { stamped }
    }

    /// Remove the alive range `txn` pushed — the exact inverse of an
    /// [`AppliedInsert::Inserted`] — and clear the reverse map if it pointed here.
    fn remove_inserted(&mut self, key: &[u8], row_id: u64, txn: u64) {
        if let Some(rows) = self.entries.get_mut(key) {
            if let Some(metas) = rows.get_mut(&row_id) {
                if let Some(pos) = metas
                    .iter()
                    .rposition(|m| m.xmin == txn && m.xmax == mvcc::NO_XMAX)
                {
                    metas.remove(pos);
                }
                if metas.is_empty() {
                    rows.remove(&row_id);
                }
            }
            if rows.is_empty() {
                self.entries.remove(key);
            }
        }
        if self.alive.get(&row_id).is_some_and(|k| k.as_slice() == key) {
            self.alive.remove(&row_id);
        }
    }

    /// Apply one physical entry removal (the raw `index_delete` treaty call): drop the row's
    /// alive range under `key`, and clear the reverse map if it pointed here. Returns the removed
    /// range's stamps, if one existed.
    fn apply_delete(&mut self, key: &[u8], row_id: u64) -> Option<EntryMeta> {
        let removed = self.entries.get_mut(key).and_then(|rows| {
            let metas = rows.get_mut(&row_id)?;
            let pos = metas.iter().rposition(|m| m.xmax == mvcc::NO_XMAX)?;
            let meta = metas.remove(pos);
            if metas.is_empty() {
                rows.remove(&row_id);
            }
            Some(meta)
        });
        if self.entries.get(key).is_some_and(BTreeMap::is_empty) {
            self.entries.remove(key);
        }
        if removed.is_some() && self.alive.get(&row_id).is_some_and(|k| k.as_slice() == key) {
            self.alive.remove(&row_id);
        }
        removed
    }

    /// Apply one dead-stamp reversal: revive the range `txn` stamped (the inverse of the stamp
    /// [`apply_insert`] placed) and point the reverse map back at it.
    fn apply_unstamp(&mut self, key: &[u8], row_id: u64, txn: u64) {
        if let Some(meta) = self
            .entries
            .get_mut(key)
            .and_then(|rows| rows.get_mut(&row_id))
            .and_then(|metas| metas.iter_mut().rfind(|m| m.xmax == txn))
        {
            meta.xmax = mvcc::NO_XMAX;
            self.alive.insert(row_id, key.to_vec());
        }
    }
}

#[derive(Debug)]
struct TxnState {
    undo: Vec<UndoOp>,
    savepoints: Vec<(String, usize)>,
    level: IsolationLevel,
    /// The transaction's current visibility snapshot (`view_for`). Fixed at `BEGIN` for
    /// `REPEATABLE READ`/`SERIALIZABLE`; refreshed at each statement start for `READ COMMITTED`/
    /// `READ UNCOMMITTED` (`begin_statement`), so every read within one statement sees a consistent
    /// view while a later statement sees intervening commits.
    pinned: ReadView,
    /// The `(table, row_id)` rows this transaction has read — tracked **only under
    /// `SERIALIZABLE`** to detect a read-write antidependency:
    /// at commit, if any row it read was modified (or deleted) by a concurrent transaction that
    /// committed after its snapshot, it aborts (40001). This turns snapshot isolation into
    /// row-level serializability — it prevents write-skew over existing rows, the anomaly the
    /// Hermitage `G2` case exercises. (Predicate/phantom antidependencies over rows that did not
    /// yet exist are the further, predicate-level SSI refinement; snapshot isolation already hides such
    /// rows from a frozen `SERIALIZABLE` reader.) Empty for every other level: `REPEATABLE READ`
    /// is snapshot isolation, which permits write-skew by design.
    reads: HashSet<(u64, u64)>,
    /// Every lock this transaction holds, released when it ends (commit, rollback, or abort).
    locks: Vec<LockId>,
    /// The per-table write versions observed at `begin` — tracked **only under `SERIALIZABLE`**
    /// (empty otherwise): the antidependency check skips every read of a table whose version has
    /// not moved (SSI narrowing).
    write_versions_at_begin: HashMap<u64, u64>,
    /// Bytes of uncommitted row data this transaction has written — the running total the optional
    /// [`BtreeEngine::with_max_txn_write_bytes`] ceiling is checked against. Always maintained; only
    /// consulted when a limit is configured. Discarded when the transaction ends (commit/abort).
    write_bytes: u64,
}

/// The inverse of one applied write, replayed newest-first on rollback. Row ops carry the whole
/// previous **encoded** leaf value (header + tuple), so replay restores the version chain
/// exactly.
#[derive(Debug)]
enum UndoOp {
    Inserted {
        table: u64,
        row_id: u64,
    },
    Updated {
        table: u64,
        row_id: u64,
        old: Vec<u8>,
        /// The arena slot this update parked the superseded version in. Undo restores `old`
        /// (whose own chain pointer predates the update), disconnecting the slot from every
        /// chain — it is queued as an orphan for purge to free once the abort settles.
        undo_idx: u64,
    },
    Deleted {
        table: u64,
        row_id: u64,
        old: Vec<u8>,
    },
    CreatedTable {
        table: u64,
    },
    DroppedTable {
        table: u64,
        state: TableState,
    },
    CreatedIndex {
        index: u64,
    },
    DroppedIndex {
        index: u64,
        state: IndexState,
    },
    IndexInserted {
        index: u64,
        key: Vec<u8>,
        row_id: u64,
        /// The row's previous alive key this insert dead-stamped (an `UPDATE` moved the row to a
        /// new key); revived on undo.
        stamped: Option<Vec<u8>>,
    },
    IndexDeleted {
        index: u64,
        key: Vec<u8>,
        row_id: u64,
        /// The removed entry's stamps, restored exactly on undo.
        meta: EntryMeta,
    },
    AddedConstraint {
        table: u64,
        name: String,
    },
    DroppedConstraint {
        table: u64,
        state: UniqueState,
    },
    AddedCheck {
        table: u64,
        name: String,
    },
    DroppedCheck {
        table: u64,
        state: CheckState,
    },
    AddedForeignKey {
        name: String,
        child_table: u64,
    },
    DroppedForeignKey {
        state: FkState,
    },
    AnalyzedTable {
        table: u64,
        previous: Option<Box<TableStats>>,
    },
    CreatedSequence {
        id: u64,
        name: String,
    },
    AlteredSchema {
        table: u64,
        previous: Box<TableSchema>,
        previous_version: u32,
        new_version: u32,
    },
    CreatedSchema {
        id: u64,
        name: String,
    },
    DroppedSchema {
        id: u64,
        name: String,
    },
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each sharded guard IS the critical section of its domain: dropping it earlier \
              than its last use would race the very invariant it guards (see the latching \
              discipline on the struct docs)"
)]
impl BtreeEngine {
    /// A new empty in-memory engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-transaction uncommitted-write-memory ceiling (bytes), returning the engine.
    /// `None` (the default) means unlimited. With `Some(limit)`, a transaction whose accumulated
    /// row writes would exceed `limit` is rejected with [`Error::OutOfMemory`] and aborts, so one
    /// oversized transaction fails loudly instead of exhausting process memory. Intended to be
    /// called once, right after [`new`](Self::new) / [`open`](Self::open), before the engine is
    /// shared.
    #[must_use]
    pub const fn with_max_txn_write_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_txn_write_bytes = limit;
        self
    }

    /// Set the global resident-memory ceiling (bytes) on the in-memory page store, returning the
    /// engine. `None` (the default) means unlimited. With `Some(limit)`, a row `insert` that would
    /// grow the store's total resident page memory past `limit` is rejected with
    /// [`Error::OutOfMemory`] and aborts its transaction, so a bulk load larger than RAM degrades to
    /// a loud error instead of an OS OOM-kill of the whole server. Complements
    /// [`with_max_txn_write_bytes`](Self::with_max_txn_write_bytes): that bounds one in-flight
    /// transaction, this bounds committed-resident data across the store. Intended to be called once,
    /// right after [`new`](Self::new) / [`open`](Self::open), before the engine is shared — so
    /// recovery (which does not go through `insert`) always completes unbounded.
    #[must_use]
    pub const fn with_max_total_resident_bytes(mut self, limit: Option<u64>) -> Self {
        self.max_total_resident_bytes = limit;
        self
    }

    /// Current resident footprint (bytes) of the in-memory page store — the metric the global
    /// resident-memory ceiling ([`with_max_total_resident_bytes`](Self::with_max_total_resident_bytes))
    /// bounds. Observability for monitoring and tests.
    ///
    /// # Errors
    /// Fails only on a poisoned store lock.
    pub fn resident_bytes(&self) -> Result<u64> {
        self.store.resident_bytes()
    }

    /// Open (or create) a **durable** engine over the WAL file at `path`.
    ///
    /// Recovery replays the durable log prefix in two passes — pass 1 collects the committed
    /// transaction set, pass 2 re-applies committed operations in log order — then truncates
    /// any torn tail (so later appends are never stranded behind garbage) and resumes the
    /// writer past the last durable LSN.
    ///
    /// # Errors
    /// Propagates file I/O errors and reports an undecodable foreign record loudly.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let engine = Self::new();
        let mut last_good: u64 = 0;
        let mut last_lsn: u64 = 0;
        let mut records: Vec<WalRecord> = Vec::new();
        // Recovery must distinguish a torn *tail* (a crash mid-append — safe to truncate to the last
        // good record) from a *hole in the middle* of the log (bit-rot / a bad sector). Since the WAL
        // is the sole durable copy of the database (no checkpoint, volatile pages), truncating at a
        // mid-log hole would silently DROP every committed transaction past it AND destroy the
        // still-intact log evidence. The whole log is read into
        // memory and `recover_prefix` scans it with byte-level resync: on any corruption it looks for
        // a valid record *after* it (proof of a mid-log hole → refuse to open, file untouched), and
        // only truncates a torn/garbage tail with no valid record following. A CRC that now covers
        // the header (lsn + len) makes a zeroed bad sector or a corrupted length fail validation
        // instead of masquerading as a valid record and desyncing the scan.
        match std::fs::read(path) {
            Ok(buf) => match nusadb_wal::recover_prefix(&buf) {
                Ok(prefix) => {
                    for (_lsn, record) in prefix.records {
                        records.push(record);
                    }
                    last_lsn = prefix.last_lsn;
                    last_good = prefix.good_bytes;
                },
                Err(hole) => {
                    let after = hole.next_valid_at.map_or_else(
                        || {
                            "no valid record prefix at all (an incompatible/older WAL format, or \
                             corruption from the first byte)"
                                .to_owned()
                        },
                        |at| format!("with valid records after it (next at byte {at})"),
                    );
                    return Err(nusadb_core::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "nusadb-btree: WAL corruption in the MIDDLE of the log at byte {} of {}, \
                             {after} — refusing to open (truncating here would silently lose every \
                             committed transaction past the corruption). Restore the WAL from a \
                             backup or repair it before reopening.",
                            hole.at,
                            path.display()
                        ),
                    )));
                },
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => return Err(e.into()),
        }
        engine.replay(&records)?;
        // Truncate the torn tail (if any) BEFORE appending: records written after garbage would
        // be unreachable to every future recovery.
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        file.set_len(last_good)?;
        file.sync_all()?;
        // Duplicate the file descriptor once here, at open, and share it behind an `Arc`; each
        // commit then clones the `Arc` (a reference-count bump) instead of the descriptor.
        let sync = Arc::new(file.try_clone()?);
        let mut writer_file = file;
        writer_file.seek(std::io::SeekFrom::End(0))?;
        let writer = WalWriter::resume(writer_file, nusadb_core::Lsn(last_lsn + 1));
        let mut engine = engine;
        engine.wal = Some(Mutex::new(Wal { writer, sync }));
        Ok(engine)
    }

    /// DST-only (`dst-fault` feature): arm the fault point so the NEXT group-leader fsync
    /// reports failure after its flush reached the file — the fsyncgate shape (the record
    /// can still hit disk although durability was reported failed). One-shot: the flag
    /// clears when it fires, so recovery and later commits sync normally.
    #[cfg(feature = "dst-fault")]
    pub fn dst_fail_next_fsync(&self) {
        self.dst_fail_next_fsync.store(true, Ordering::SeqCst);
    }

    /// DST-only (`dst-fault` feature): arm the fault point so the NEXT WAL append fails with an
    /// ENOSPC-shaped error *before* writing — nothing reaches the log. Use it to exercise the
    /// disk-full commit path: the commit-marker append fails, so the commit must abort the
    /// transaction cleanly (no partial durable state, locks released) and surface the error.
    /// One-shot: the flag clears when it fires, so a later retry commits normally.
    #[cfg(feature = "dst-fault")]
    pub fn dst_fail_next_wal_append(&self) {
        self.dst_fail_next_wal_append.store(true, Ordering::SeqCst);
    }

    /// Re-apply the committed operations of a recovered log, in log order. Post-recovery there
    /// are no live snapshots, so versions collapse: every replayed row is a fresh single
    /// version stamped with its original (committed) transaction id, and a committed delete
    /// simply removes the entry — semantically identical to the pre-crash visible state.
    fn replay(&self, records: &[WalRecord]) -> Result<()> {
        // Abort always wins: a transaction counts as committed iff it has a `CommitTxn` marker AND
        // no `AbortTxn` marker, regardless of the order the two appear in the log. That corner
        // exists — a commit whose fsync failed leaves the transaction active with its marker possibly
        // flushed; if the caller then rolls back, the abort marker is the truth and replay must not
        // resurrect the transaction. Tracking the two sets separately (rather than last-marker-wins)
        // makes the rule robust to any ordering, not just the monotonic one today's writer produces.
        let mut committed: HashSet<u64> = HashSet::new();
        let mut aborted: HashSet<u64> = HashSet::new();
        for record in records {
            match record {
                WalRecord::CommitTxn { txn } => {
                    committed.insert(txn.0);
                },
                WalRecord::AbortTxn { txn } => {
                    aborted.insert(txn.0);
                },
                _ => {},
            }
        }
        committed.retain(|txn| !aborted.contains(txn));
        // Recovery is single-threaded (the engine is not yet shared), so the domain guards are
        // taken once up front, in rank order.
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
        let mut max_txn: u64 = 0;
        for record in records {
            match record {
                WalRecord::CommitTxn { txn } | WalRecord::AbortTxn { txn } => {
                    max_txn = max_txn.max(txn.0);
                },
                WalRecord::Put { .. } => {
                    let Some(op) = LoggedOp::from_record(record) else {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "nusadb-btree: foreign or corrupt record in the engine WAL",
                        )));
                    };
                    max_txn = max_txn.max(op.txn());
                    // Non-transactional records (sequence family) apply unconditionally — a
                    // counter advance is durable the moment it was fsynced, commit or not.
                    if op.is_non_transactional() || committed.contains(&op.txn()) {
                        Self::replay_op(&mut cat, &mut seqs, &self.store, &op)?;
                    }
                },
                _ => {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "nusadb-btree: unexpected record shape in the engine WAL",
                    )));
                },
            }
        }
        drop(seqs);
        drop(cat);
        let mut txns = self.txns.lock().map_err(|_| poisoned())?;
        txns.next_txn_id = txns.next_txn_id.max(max_txn + 1);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-op replay dispatcher; splitting it would only scatter                   the recovery semantics"
    )]
    fn replay_op(
        cat: &mut Catalog,
        seqs: &mut SeqDomain,
        store: &MemPageStore,
        op: &LoggedOp,
    ) -> Result<()> {
        match op {
            LoggedOp::CreateTable { txn: _, table, def } => {
                let tree = ClusteredTree::create(store)?;
                let schema = TableSchema {
                    id: TableId(*table),
                    schema: def.schema.clone(),
                    name: def.name.clone(),
                    columns: def.columns.clone(),
                };
                cat.by_name
                    .insert((def.schema.clone(), def.name.clone()), *table);
                cat.tables.insert(
                    *table,
                    TableState {
                        schema: schema.clone(),
                        root: AtomicU64::new(tree.root().0),
                        approx_rows: AtomicU64::new(TableState::APPROX_UNINIT),
                        churn_since_analyze: AtomicU64::new(0),
                        write: Mutex::new(TableWrite::default()),
                        schema_version: 0,
                        schema_history: std::iter::once((0, schema)).collect(),
                    },
                );
                cat.next_table_id = cat.next_table_id.max(*table + 1);
            },
            LoggedOp::DropTable { txn: _, table } => {
                if let Some(state) = cat.tables.remove(table) {
                    let root = state.root_id();
                    cat.by_name
                        .remove(&(state.schema.schema.clone(), state.schema.name));
                    // Mid-recovery there are no live views: the rebuilt tree can be freed at
                    // once instead of queueing for purge.
                    let tree = ClusteredTree::open(store, root);
                    for page in tree.pages()? {
                        store.deallocate_page(page)?;
                    }
                }
            },
            LoggedOp::Insert {
                txn,
                table,
                row_id,
                tuple,
            } => {
                if let Some(t) = cat.tables.get_mut(table) {
                    let value = mvcc::encode_row(RowMeta::fresh(*txn), tuple);
                    let mut tree = ClusteredTree::open(store, t.root_id());
                    tree.insert(*row_id, &value)?;
                    t.set_root(tree.root());
                    let w = t.write.get_mut().map_err(|_| poisoned())?;
                    w.next_row_id = w.next_row_id.max(*row_id + 1);
                }
            },
            LoggedOp::Update {
                txn,
                table,
                row_id,
                tuple,
            } => {
                if let Some(t) = cat.tables.get_mut(table) {
                    let value = mvcc::encode_row(RowMeta::fresh(*txn), tuple);
                    let mut tree = ClusteredTree::open(store, t.root_id());
                    // Upsert: a savepoint-compensation Update may follow a logged Delete.
                    if tree.get(*row_id)?.is_some() {
                        tree.update(*row_id, &value)?;
                    } else {
                        tree.insert(*row_id, &value)?;
                        let w = t.write.get_mut().map_err(|_| poisoned())?;
                        w.next_row_id = w.next_row_id.max(*row_id + 1);
                    }
                    t.set_root(tree.root());
                }
            },
            LoggedOp::Delete {
                txn: _,
                table,
                row_id,
            } => {
                if let Some(t) = cat.tables.get_mut(table) {
                    let tree = ClusteredTree::open(store, t.root_id());
                    // Tolerant: a compensation Delete may target an already-absent row.
                    let _ = tree.delete(*row_id)?;
                    t.set_root(tree.root());
                }
            },
            LoggedOp::CreateIndex { txn: _, index, def } => {
                cat.idx_by_name.insert(def.name.clone(), *index);
                cat.indexes.insert(
                    *index,
                    IndexState {
                        def: def.clone(),
                        complete: true,
                        data: RwLock::new(IndexData::default()),
                    },
                );
                cat.next_index_id = cat.next_index_id.max(*index + 1);
            },
            LoggedOp::DropIndex { txn: _, index } => {
                if let Some(state) = cat.indexes.remove(index) {
                    cat.idx_by_name.remove(&state.def.name);
                }
            },
            LoggedOp::IndexInsert {
                txn,
                index,
                row_id,
                key,
            } => {
                // The shared apply path re-derives the same dead-stamp the live insert placed,
                // so recovery converges without stamps ever entering the log.
                if let Some(idx) = cat.indexes.get_mut(index) {
                    idx.data
                        .get_mut()
                        .map_err(|_| poisoned())?
                        .apply_insert(key, *row_id, *txn);
                }
            },
            LoggedOp::IndexDelete {
                txn: _,
                index,
                row_id,
                key,
            } => {
                // Tolerant: a compensation delete may target an already-absent entry.
                if let Some(idx) = cat.indexes.get_mut(index) {
                    idx.data
                        .get_mut()
                        .map_err(|_| poisoned())?
                        .apply_delete(key, *row_id);
                }
            },
            LoggedOp::IndexUnstamp {
                txn,
                index,
                row_id,
                key,
            } => {
                if let Some(idx) = cat.indexes.get_mut(index) {
                    idx.data
                        .get_mut()
                        .map_err(|_| poisoned())?
                        .apply_unstamp(key, *row_id, *txn);
                }
            },
            LoggedOp::AddUnique {
                txn: _,
                table,
                index,
                name,
                columns,
                primary,
            } => {
                cat.constraints
                    .entry(*table)
                    .or_default()
                    .push(UniqueState {
                        name: name.clone(),
                        columns: columns.clone(),
                        primary: *primary,
                        index: *index,
                    });
            },
            LoggedOp::AddCheck {
                txn: _,
                table,
                name,
                expr,
            } => {
                cat.checks.entry(*table).or_default().push(CheckState {
                    name: name.clone(),
                    expr: expr.clone(),
                });
            },
            LoggedOp::AddFk {
                txn: _,
                name,
                child_table,
                child_columns,
                parent_table,
                parent_index,
                child_index,
                on_delete,
                on_update,
            } => {
                cat.foreign_keys.insert(
                    name.clone(),
                    FkState {
                        name: name.clone(),
                        child_table: *child_table,
                        child_columns: child_columns.clone(),
                        parent_table: *parent_table,
                        parent_index: *parent_index,
                        child_index: *child_index,
                        on_delete: *on_delete,
                        on_update: *on_update,
                    },
                );
            },
            LoggedOp::DropConstraint {
                txn: _,
                table,
                name,
            } => {
                // Whichever kind carries the name; the backing index's own DropIndex record
                // follows separately in the log. Tolerant of an already-absent name.
                if let Some(list) = cat.checks.get_mut(table) {
                    list.retain(|c| &c.name != name);
                }
                if let Some(list) = cat.constraints.get_mut(table) {
                    list.retain(|c| &c.name != name);
                }
                if cat
                    .foreign_keys
                    .get(name)
                    .is_some_and(|fk| fk.child_table == *table)
                {
                    cat.foreign_keys.remove(name);
                }
            },
            LoggedOp::SetStats {
                txn: _,
                table,
                stats,
            } => {
                cat.stats.insert(*table, stats.clone());
            },
            LoggedOp::ClearStats { txn: _, table } => {
                cat.stats.remove(table);
            },
            LoggedOp::SeqCreate { id, def } => {
                seqs.seq_by_name.insert(def.name.clone(), *id);
                seqs.sequences.insert(
                    *id,
                    SequenceState {
                        def: def.clone(),
                        current: None,
                    },
                );
                seqs.next_sequence_id = seqs.next_sequence_id.max(*id + 1);
            },
            LoggedOp::SeqDrop { id } => {
                if let Some(seq) = seqs.sequences.remove(id) {
                    seqs.seq_by_name.remove(&seq.def.name);
                }
            },
            LoggedOp::SeqSet { id, value } => {
                if let Some(seq) = seqs.sequences.get_mut(id) {
                    seq.current = Some(*value);
                }
            },
            LoggedOp::AlterSchema {
                txn: _,
                table,
                version,
                def,
            } => {
                if let Some(t) = cat.tables.get_mut(table) {
                    let previous_name = t.schema.name.clone();
                    let previous_schema = t.schema.schema.clone();
                    let new_schema = TableSchema {
                        id: TableId(*table),
                        schema: def.schema.clone(),
                        name: def.name.clone(),
                        columns: def.columns.clone(),
                    };
                    // Reflect a rename in the by-name index.
                    if previous_name != def.name || previous_schema != def.schema {
                        cat.by_name.remove(&(previous_schema, previous_name));
                        cat.by_name
                            .insert((def.schema.clone(), def.name.clone()), *table);
                    }
                    if let Some(t) = cat.tables.get_mut(table) {
                        t.schema = new_schema.clone();
                        t.schema_version = *version;
                        t.schema_history.insert(*version, new_schema);
                        // A savepoint-compensation record reverts to a LOWER version; drop the
                        // now-orphaned higher-version entries so `schema_history` matches the
                        // live (undo_ops) path exactly (versions are monotonic forward, so this
                        // only ever prunes on a revert, never a legitimate forward alter).
                        t.schema_history.retain(|&v, _| v <= *version);
                    }
                }
            },
            LoggedOp::SchemaCreate { txn: _, id, name } => {
                cat.ns_by_name.insert(name.clone(), *id);
                cat.namespaces.insert(*id, name.clone());
                cat.next_namespace_id = cat.next_namespace_id.max(*id + 1);
            },
            LoggedOp::SchemaDrop { txn: _, id, name } => {
                cat.namespaces.remove(id);
                cat.ns_by_name.remove(name);
            },
        }
        Ok(())
    }

    /// Append `record` to the durable log (no fsync — the commit marker's fsync is the
    /// durability point). A no-op for the in-memory engine.
    ///
    /// Called after the in-memory apply, whose undo entry is already pushed — so if the append
    /// fails, the mutating call errors out with the transaction effectively abort-only: a
    /// `rollback` reverts the applied change and memory/log converge again.
    fn log(&self, record: &WalRecord) -> Result<()> {
        if let Some(wal) = &self.wal {
            let mut wal = wal.lock().map_err(|_| poisoned())?;
            wal.writer.append(record)?;
        }
        Ok(())
    }

    /// End a committed transaction in memory: drop its state, release its locks, leave the
    /// active set (its writes become visible), and bump the data-change version iff it wrote
    /// (a read-only commit leaves the SQL result cache valid). On the durable path this
    /// runs only AFTER the group fsync returned — the durability point precedes visibility.
    ///
    /// Returns the committed transaction's per-table [`CommitDeltas`] (net row change + write churn)
    /// for the caller to fold into the `O(1)` approximate row counters and the auto-analyze churn
    /// tally via [`apply_commit_deltas`] — which the caller does **after** releasing the `txns` lock,
    /// since that update takes the (lower-rank) catalog guard.
    fn finish_commit(t: &mut TxnDomain, txn: TxnId, data_version: &AtomicU64) -> CommitDeltas {
        let mut deltas = CommitDeltas {
            net: HashMap::new(),
            churn: HashMap::new(),
        };
        if let Some(state) = t.txns.remove(&txn.0) {
            t.release_locks(txn.0, &state.locks);
            if !state.undo.is_empty() {
                data_version.fetch_add(1, Ordering::SeqCst);
                deltas = commit_deltas(&state.undo);
            }
        }
        t.active.remove(&txn.0);
        deltas
    }

    /// Fold a committed transaction's per-table deltas (from [`finish_commit`]) into the `O(1)`
    /// approximate row counters and the auto-analyze churn tallies. Called with the `txns` lock
    /// released; it takes the catalog read guard (a lower lock rank) and only nudges already-live
    /// counters.
    fn apply_commit_deltas(&self, deltas: &CommitDeltas) -> Result<()> {
        if deltas.net.is_empty() && deltas.churn.is_empty() {
            return Ok(());
        }
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        for (&table, &delta) in &deltas.net {
            if let Some(state) = cat.tables.get(&table) {
                state.add_approx_delta(delta);
            }
        }
        for (&table, &ops) in &deltas.churn {
            if let Some(state) = cat.tables.get(&table) {
                state.add_churn(ops);
            }
        }
        Ok(())
    }

    /// Append `record` and fsync immediately — the durability point of a **non-transactional**
    /// op (the sequence family): the record must be durable before its effect can escape (a
    /// `nextval` value handed to a client must never repeat after a crash). A no-op for the
    /// in-memory engine.
    fn log_durable(&self, record: &WalRecord) -> Result<()> {
        self.log(record)?;
        self.sync_log()
    }

    /// The group leader's flush: write the buffer to the file under the wal lock, note the tail
    /// LSN it covers, then `fsync` on a cloned handle WITHOUT the lock — so stagers keep
    /// appending (and queueing behind the next leader) while the fsync runs. Returns the highest
    /// LSN made durable.
    #[cfg_attr(
        not(feature = "dst-fault"),
        expect(
            clippy::unused_self,
            reason = "`self` carries the dst-fault injection flag; without the feature the \
                      receiver is unused but the signature must not flip-flop on a cfg"
        )
    )]
    fn flush_and_sync(&self, wal: &Mutex<Wal>) -> std::io::Result<u64> {
        let (tail, sync) = {
            let mut wal = wal
                .lock()
                .map_err(|_| std::io::Error::other("engine WAL lock poisoned"))?;
            wal.writer
                .flush()
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            (
                wal.writer.next_lsn().0.saturating_sub(1),
                Arc::clone(&wal.sync),
            )
        };
        // DST fault point: fail AFTER the flush above so the record is in the file (and can
        // therefore "resurrect" on restart) while the durability report says failure — the
        // exact fsyncgate shape the commit fail-stop exists for.
        #[cfg(feature = "dst-fault")]
        if self.dst_fail_next_fsync.swap(false, Ordering::SeqCst) {
            return Err(std::io::Error::other(
                "dst-fault: injected commit-fsync failure",
            ));
        }
        sync.sync_data()?;
        Ok(tail)
    }

    /// Make everything appended so far durable — through the group coordinator, so a burst of
    /// non-transactional durability points (`nextval` under load) shares fsyncs with commits.
    fn sync_log(&self) -> Result<()> {
        let Some(wal) = &self.wal else {
            return Ok(());
        };
        // Reading the tail AFTER our caller's append means `seq` covers it; a concurrent later
        // append only raises the bar (over-waiting is harmless).
        let seq = {
            let wal = wal.lock().map_err(|_| poisoned())?;
            wal.writer.next_lsn().0.saturating_sub(1)
        };
        self.group.commit(seq, || self.flush_and_sync(wal))?;
        Ok(())
    }

    /// Log the compensation operations for a partial rollback: the logical inverses of the
    /// undone writes, so replay converges to the post-rollback state even though the earlier
    /// op records stay in the log.
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-undo-op inverse table; splitting it would scatter the                   compensation semantics"
    )]
    fn log_compensations(&self, txn: TxnId, undone: &[UndoOp]) -> Result<()> {
        if self.wal.is_none() {
            return Ok(());
        }
        // Newest-first replay order mirrors the in-memory undo.
        for op in undone.iter().rev() {
            let comps: Vec<LoggedOp> = match op {
                UndoOp::Inserted { table, row_id } => vec![LoggedOp::Delete {
                    txn: txn.0,
                    table: *table,
                    row_id: *row_id,
                }],
                UndoOp::Updated {
                    table, row_id, old, ..
                }
                | UndoOp::Deleted { table, row_id, old } => {
                    let (meta, tuple) =
                        mvcc::decode_row(old).ok_or_else(|| corrupt_row(*row_id))?;
                    if meta.xmax == mvcc::NO_XMAX {
                        vec![LoggedOp::Update {
                            txn: txn.0,
                            table: *table,
                            row_id: *row_id,
                            tuple: tuple.to_vec(),
                        }]
                    } else {
                        // The restored state was already a deleted row.
                        vec![LoggedOp::Delete {
                            txn: txn.0,
                            table: *table,
                            row_id: *row_id,
                        }]
                    }
                },
                // DDL compensation: recreate/drop mirrors of the undone catalog ops.
                UndoOp::CreatedTable { table } => vec![LoggedOp::DropTable {
                    txn: txn.0,
                    table: *table,
                }],
                UndoOp::DroppedTable { table, state } => {
                    // Replay's DropTable discards the rows, so the compensating CreateTable
                    // alone would resurrect the table EMPTY — every live row must be re-logged
                    // too (rows only chain-visible to old snapshots are skipped: post-recovery
                    // there are no old snapshots).
                    let mut comps = vec![LoggedOp::CreateTable {
                        txn: txn.0,
                        table: *table,
                        def: TableDef {
                            schema: state.schema.schema.clone(),
                            name: state.schema.name.clone(),
                            columns: state.schema.columns.clone(),
                        },
                    }];
                    let tree = ClusteredTree::open(&self.store, state.root_id());
                    for (row_id, value) in tree.scan()? {
                        let (meta, tuple) =
                            mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
                        if meta.xmax == mvcc::NO_XMAX {
                            comps.push(LoggedOp::Insert {
                                txn: txn.0,
                                table: *table,
                                row_id,
                                tuple: tuple.to_vec(),
                            });
                        }
                    }
                    comps
                },
                UndoOp::CreatedIndex { index } => vec![LoggedOp::DropIndex {
                    txn: txn.0,
                    index: *index,
                }],
                UndoOp::DroppedIndex { index, state } => {
                    // Same shape as DroppedTable: the definition alone replays empty, so every
                    // entry is re-logged with it. Dead entries go first and each row's alive
                    // entry last, so replaying the shared insert path leaves the same entry
                    // alive (stamp identities differ, but every stamp is committed by the time
                    // these compensations apply, so visibility is identical). A dead-only chain
                    // (reachable only through the raw `index_delete` treaty call) restores with
                    // its newest dead entry alive — the base-row hop still filters invisible
                    // rows, so the residue matches the entry's pre-drop reachability.
                    let mut comps = vec![LoggedOp::CreateIndex {
                        txn: txn.0,
                        index: *index,
                        def: state.def.clone(),
                    }];
                    let mut alive_last = Vec::new();
                    let data = state.data.read().map_err(|_| poisoned())?;
                    for (key, rows) in &data.entries {
                        for (&row_id, metas) in rows {
                            // One record per (key, row): post-recovery there are no pinned
                            // pre-drop snapshots, so only the alive range matters — replaying
                            // the insert re-derives an equivalent single range.
                            if metas.iter().any(|m| m.xmax == mvcc::NO_XMAX) {
                                alive_last.push((key, row_id));
                            } else {
                                comps.push(LoggedOp::IndexInsert {
                                    txn: txn.0,
                                    index: *index,
                                    row_id,
                                    key: key.clone(),
                                });
                            }
                        }
                    }
                    for (key, row_id) in alive_last {
                        comps.push(LoggedOp::IndexInsert {
                            txn: txn.0,
                            index: *index,
                            row_id,
                            key: key.clone(),
                        });
                    }
                    comps
                },
                UndoOp::IndexInserted {
                    index,
                    key,
                    row_id,
                    stamped,
                } => {
                    let mut comps = vec![LoggedOp::IndexDelete {
                        txn: txn.0,
                        index: *index,
                        row_id: *row_id,
                        key: key.clone(),
                    }];
                    // The insert dead-stamped the row's previous alive entry; replay must revive
                    // it exactly like the in-memory undo does.
                    if let Some(old_key) = stamped {
                        comps.push(LoggedOp::IndexUnstamp {
                            txn: txn.0,
                            index: *index,
                            row_id: *row_id,
                            key: old_key.clone(),
                        });
                    }
                    comps
                },
                UndoOp::IndexDeleted {
                    index,
                    key,
                    row_id,
                    meta: _,
                } => vec![LoggedOp::IndexInsert {
                    txn: txn.0,
                    index: *index,
                    row_id: *row_id,
                    key: key.clone(),
                }],
                UndoOp::AddedConstraint { table, name } | UndoOp::AddedCheck { table, name } => {
                    vec![LoggedOp::DropConstraint {
                        txn: txn.0,
                        table: *table,
                        name: name.clone(),
                    }]
                },
                UndoOp::DroppedConstraint { table, state } => vec![LoggedOp::AddUnique {
                    txn: txn.0,
                    table: *table,
                    index: state.index,
                    name: state.name.clone(),
                    columns: state.columns.clone(),
                    primary: state.primary,
                }],
                UndoOp::DroppedCheck { table, state } => vec![LoggedOp::AddCheck {
                    txn: txn.0,
                    table: *table,
                    name: state.name.clone(),
                    expr: state.expr.clone(),
                }],
                UndoOp::AddedForeignKey { name, child_table } => {
                    vec![LoggedOp::DropConstraint {
                        txn: txn.0,
                        table: *child_table,
                        name: name.clone(),
                    }]
                },
                UndoOp::DroppedForeignKey { state } => vec![LoggedOp::AddFk {
                    txn: txn.0,
                    name: state.name.clone(),
                    child_table: state.child_table,
                    child_columns: state.child_columns.clone(),
                    parent_table: state.parent_table,
                    parent_index: state.parent_index,
                    child_index: state.child_index,
                    on_delete: state.on_delete,
                    on_update: state.on_update,
                }],
                UndoOp::AnalyzedTable { table, previous } => previous.as_ref().map_or_else(
                    || {
                        vec![LoggedOp::ClearStats {
                            txn: txn.0,
                            table: *table,
                        }]
                    },
                    |prev| {
                        vec![LoggedOp::SetStats {
                            txn: txn.0,
                            table: *table,
                            stats: (**prev).clone(),
                        }]
                    },
                ),
                // A SeqCreate record replays unconditionally, so the undone create must be
                // neutralized in the log too.
                UndoOp::CreatedSequence { id, .. } => vec![LoggedOp::SeqDrop { id: *id }],
                // Re-establish the pre-alter schema (the earlier AlterSchema record stays in the
                // log; this reverts it on replay).
                UndoOp::AlteredSchema {
                    table,
                    previous,
                    previous_version,
                    ..
                } => vec![LoggedOp::AlterSchema {
                    txn: txn.0,
                    table: *table,
                    version: *previous_version,
                    def: TableDef {
                        schema: previous.schema.clone(),
                        name: previous.name.clone(),
                        columns: previous.columns.clone(),
                    },
                }],
                UndoOp::CreatedSchema { id, name } => vec![LoggedOp::SchemaDrop {
                    txn: txn.0,
                    id: *id,
                    name: name.clone(),
                }],
                UndoOp::DroppedSchema { id, name } => vec![LoggedOp::SchemaCreate {
                    txn: txn.0,
                    id: *id,
                    name: name.clone(),
                }],
            };
            for comp in comps {
                self.log(&comp.to_record())?;
            }
        }
        // A neutralizing SeqDrop must be as durable as the SeqCreate it erases (both replay
        // unconditionally): fsync when the undone tail contained one.
        if undone
            .iter()
            .any(|op| matches!(op, UndoOp::CreatedSequence { .. }))
        {
            self.sync_log()?;
        }
        Ok(())
    }
}

/// Encode a row-id as the treaty's [`Tid`]: the id *is* the address (ADR 008 §D1 · stable across
/// splits/merges by construction; nothing in the treaty makes a `Tid` physical).
const fn tid_of(row_id: u64) -> Tid {
    Tid {
        page: nusadb_core::PageId(row_id),
        slot: SlotIdx(0),
    }
}

/// The row-id a treaty [`Tid`] addresses (inverse of [`tid_of`]).
const fn row_id_of(tid: Tid) -> u64 {
    tid.page.0
}

const fn unknown_txn(txn: TxnId) -> Error {
    Error::UnknownTransaction { txn }
}

fn table_not_found(table: TableId) -> Error {
    Error::TableNotFound {
        name: format!("table id {}", table.0),
    }
}

/// A Tid that addresses no live row (deleted or never existed) — loud, the same contract the predecessor engine's
/// missing-version error.
fn tuple_not_found(tid: Tid) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("nusadb-btree: no row at tid {tid:?}"),
    ))
}

fn constraint_not_found(table: TableId, name: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "nusadb-btree: no constraint named {name} on table {}",
            table.0
        ),
    ))
}

fn fk_not_found(name: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("nusadb-btree: no foreign key named {name}"),
    ))
}

fn index_not_found(index: IndexId) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("nusadb-btree: no index with id {}", index.0),
    ))
}

/// Borrow a `Bound<Vec<u8>>` as a `Bound<&[u8]>` for `BTreeMap::range`.
const fn as_slice_bound(b: &Bound<Vec<u8>>) -> Bound<&[u8]> {
    match b {
        Bound::Included(v) => Bound::Included(v.as_slice()),
        Bound::Excluded(v) => Bound::Excluded(v.as_slice()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// The engine-level bound: a user tuple must leave room for the MVCC header in one leaf.
fn tuple_too_large(len: usize) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "nusadb-btree: tuple of {len} bytes exceeds the single-leaf capacity of \
             {MAX_USER_TUPLE} bytes (overflow pages are a later phase)"
        ),
    ))
}

/// The per-table maintenance a committed transaction implies, read once from its undo log.
struct CommitDeltas {
    /// Net live-row change (`inserted − deleted`) → the `O(1)` approximate row counters
    /// ([`TableState::add_approx_delta`]). An update or an insert+delete of the same row nets to zero.
    net: HashMap<u64, i64>,
    /// Absolute write churn (`inserts + updates + deletes`) → the auto-analyze staleness tally
    /// ([`TableState::add_churn`]). Every row op counts, including updates and both halves of an
    /// insert+delete, because heavy churn ages statistics even when the row count is unchanged.
    churn: HashMap<u64, u64>,
}

/// Compute a committed transaction's per-table [`CommitDeltas`] from its undo log in one pass: an
/// `Inserted` is `+1` net and `+1` churn, a `Deleted` is `−1` net and `+1` churn, an `Updated` is
/// `0` net and `+1` churn, and every other op (index, DDL, sequence) touches neither.
fn commit_deltas(undo: &[UndoOp]) -> CommitDeltas {
    let mut net: HashMap<u64, i64> = HashMap::new();
    let mut churn: HashMap<u64, u64> = HashMap::new();
    for op in undo {
        match op {
            UndoOp::Inserted { table, .. } => {
                *net.entry(*table).or_default() += 1;
                *churn.entry(*table).or_default() += 1;
            },
            UndoOp::Deleted { table, .. } => {
                *net.entry(*table).or_default() -= 1;
                *churn.entry(*table).or_default() += 1;
            },
            UndoOp::Updated { table, .. } => *churn.entry(*table).or_default() += 1,
            _ => {},
        }
    }
    CommitDeltas { net, churn }
}

/// The loud error a transaction hits when its uncommitted row writes exceed the configured
/// per-transaction memory ceiling — so it aborts instead of exhausting process memory.
fn txn_memory_exceeded(limit: u64, attempted: u64) -> Error {
    Error::OutOfMemory(format!(
        "transaction exceeded its write-memory limit of {limit} bytes (needed {attempted}); \
         split it into smaller transactions or raise the limit"
    ))
}

/// The loud error an `insert` hits when the in-memory page store has grown to the configured global
/// resident-memory ceiling — so the write aborts gracefully instead of growing until the OS
/// OOM-kills the server. `DELETE`/`TRUNCATE` stay available to free space.
fn resident_memory_exceeded(limit: u64, resident: u64) -> Error {
    Error::OutOfMemory(format!(
        "the in-memory store reached its resident-memory limit of {limit} bytes ({resident} bytes \
         resident); free rows (DELETE/TRUNCATE), raise the limit, or use a larger host"
    ))
}

/// A leaf value that does not carry a decodable MVCC header — corruption-class, loud.
fn corrupt_row(row_id: u64) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("nusadb-btree: row {row_id} has no decodable version header"),
    ))
}

fn poisoned() -> Error {
    Error::Io(std::io::Error::other(
        "nusadb-btree: engine state lock poisoned by a previous panic",
    ))
}

/// Append the commit marker for `txn`, honoring the DST WAL-append fault point. In production
/// (no `dst-fault`) this is a plain `writer.append`; under `dst-fault`, an armed one-shot fault
/// makes it report ENOSPC (`StorageFull`) *before* writing anything, so a test can drive the
/// disk-full commit/abort path without a real full disk. Kept a free function so it never carries
/// an unused `self` when the fault point is compiled out.
fn append_commit_marker(
    engine: &BtreeEngine,
    writer: &mut WalWriter<File>,
    txn: TxnId,
) -> Result<nusadb_core::Lsn> {
    #[cfg(feature = "dst-fault")]
    if engine
        .dst_fail_next_wal_append
        .swap(false, Ordering::SeqCst)
    {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::StorageFull,
            "dst-fault: injected ENOSPC on WAL append",
        )));
    }
    #[cfg(not(feature = "dst-fault"))]
    let _ = engine; // the fault point is compiled out in production builds
    writer.append(&WalRecord::CommitTxn { txn })
}

impl TxnDomain {
    /// A view taken NOW for `own`: other active transactions invisible, later ids invisible.
    fn fresh_view(&self, own: u64) -> ReadView {
        let mut active = self.active.clone();
        active.remove(&own);
        ReadView {
            own,
            active,
            horizon: self.next_txn_id,
        }
    }

    /// The view `txn` reads under — its `pinned` **statement snapshot**.
    ///
    /// Every level reads under one snapshot, so all reads WITHIN a single statement observe a
    /// consistent view: previously `READ COMMITTED`/`READ UNCOMMITTED`
    /// took a *fresh* view on every engine call, so a statement touching two tables (a join, a
    /// self-join, two scalar subqueries) could read each under a different snapshot and, with a
    /// concurrent transfer committing between the two reads, see money created from nothing. The
    /// snapshot is fixed at `BEGIN` for `REPEATABLE READ`/`SERIALIZABLE`; for `READ COMMITTED`/`READ
    /// UNCOMMITTED` it is refreshed at each statement start ([`BtreeEngine::begin_statement`]), so a
    /// later statement still sees transactions that committed in between (standard RC).
    fn view_for(&self, txn: u64) -> Result<ReadView> {
        let state = self.txns.get(&txn).ok_or_else(|| unknown_txn(TxnId(txn)))?;
        Ok(state.pinned.clone())
    }

    /// No-wait write admission for `txn` against a row whose newest version is `meta`:
    ///
    /// - newest version written by a concurrent (active) other transaction → conflict (40001);
    /// - **first-updater-wins (all isolation levels):** a newest version whose creator this
    ///   transaction's `BEGIN` snapshot cannot see was committed *after* this transaction began,
    ///   so writing over it would lose the concurrent update — conflict (40001), not last-writer-
    ///   wins. This is the lost-update guard the OCC engine relies on:
    ///   under `READ COMMITTED` a statement re-reads the latest committed value, but the value it
    ///   *wrote back* was computed from a read that may predate a now-committed concurrent write,
    ///   so the write itself must abort-and-retry (the no-wait OCC discipline) — the
    ///   caller retries and recomputes against the committed value. Reads stay fresh per level
    ///   (see [`State::view_for`]); only write admission consults the begin snapshot.
    /// - newest version already deleted: by an active other → conflict; by a transaction the begin
    ///   snapshot cannot see → conflict (first-updater-wins on the delete); otherwise (a delete
    ///   already visible at `BEGIN`, or this transaction's own) the row is gone → not-found.
    fn admit_write(&self, txn: u64, meta: RowMeta, tid: Tid) -> Result<()> {
        let other_active = |id: u64| id != txn && self.active.contains(&id);
        // The begin snapshot: a version whose creator it cannot see was committed after this
        // transaction began. Absent only for an unknown txn (rejected earlier), so default to a
        // conflict-free view.
        let unseen = |stamp: u64| {
            self.txns
                .get(&txn)
                .is_some_and(|state| !state.pinned.sees(stamp))
        };
        if other_active(meta.xmin) {
            return Err(Error::SerializationConflict { txn: TxnId(txn) });
        }
        if meta.xmax != mvcc::NO_XMAX {
            if other_active(meta.xmax) || unseen(meta.xmax) {
                return Err(Error::SerializationConflict { txn: TxnId(txn) });
            }
            return Err(tuple_not_found(tid));
        }
        if unseen(meta.xmin) {
            return Err(Error::SerializationConflict { txn: TxnId(txn) });
        }
        Ok(())
    }

    /// Acquire `id` for `txn` in the requested mode, no-wait: a conflict is an immediate
    /// [`Error::SerializationConflict`] (the caller retries), never a block — so no deadlock can
    /// form. Re-entrant (a lock already held at or above the requested strength is a no-op); a
    /// shared → exclusive upgrade succeeds only for a sole holder.
    fn acquire_lock(&mut self, txn: u64, id: LockId, exclusive: bool) -> Result<()> {
        // Every caller guards the transaction's existence, but a lock granted to an unknown
        // transaction could never be released — refuse defensively rather than leak.
        if !self.txns.contains_key(&txn) {
            return Err(unknown_txn(TxnId(txn)));
        }
        let entry = self.locks.entry(id).or_default();
        if let Some(&held_exclusive) = entry.holders.get(&txn) {
            if held_exclusive || !exclusive {
                return Ok(());
            }
            if entry.holders.len() == 1 {
                entry.holders.insert(txn, true);
                return Ok(());
            }
            return Err(Error::SerializationConflict { txn: TxnId(txn) });
        }
        let compatible = if exclusive {
            entry.holders.is_empty()
        } else {
            entry.holders.values().all(|&e| !e)
        };
        if !compatible {
            return Err(Error::SerializationConflict { txn: TxnId(txn) });
        }
        entry.holders.insert(txn, exclusive);
        if let Some(t) = self.txns.get_mut(&txn) {
            t.locks.push(id);
        }
        Ok(())
    }

    /// Release every lock in `held` for `txn` — called once when the transaction ends.
    fn release_locks(&mut self, txn: u64, held: &[LockId]) {
        for id in held {
            if let Some(entry) = self.locks.get_mut(id) {
                entry.holders.remove(&txn);
                if entry.holders.is_empty() {
                    self.locks.remove(id);
                }
            }
        }
    }

    /// The shared table-intention lock every row write and row/key lock takes first, so a
    /// concurrent `LOCK TABLE ACCESS EXCLUSIVE` genuinely excludes all table activity
    /// (multi-granularity the lock-table contract carried over from the predecessor engine).
    fn lock_table_intention(&mut self, txn: u64, table: u64) -> Result<()> {
        self.acquire_lock(txn, LockId::Table { table }, false)
    }
}

/// The scan the treaty hands back: the rows visible under the caller's read view, materialized
/// at open in row-id order (a stable snapshot for the scan's lifetime).
struct VecScan {
    rows: std::vec::IntoIter<(Tid, SharedTuple)>,
}

impl TupleScan for VecScan {
    fn try_next(&mut self) -> Result<Option<(Tid, SharedTuple)>> {
        Ok(self.rows.next())
    }
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each sharded guard IS the critical section of its domain: dropping it earlier \
              than its last use would race the very invariant it guards (see the latching \
              discipline on the struct docs)"
)]
impl nusadb_core::StorageEngine for BtreeEngine {
    fn begin_statement(&self, txn: TxnId) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        // Refresh the statement snapshot for READ COMMITTED / READ UNCOMMITTED so this statement's
        // reads see a fresh, consistent view. REPEATABLE READ / SERIALIZABLE
        // keep the BEGIN-pinned snapshot. An unknown/ended txn is tolerated (a no-op), matching the
        // engine's other txn-id-lenient calls — a spurious refresh cannot break correctness.
        let refresh = matches!(
            t.txns.get(&txn.0).map(|s| s.level),
            Some(IsolationLevel::ReadCommitted | IsolationLevel::ReadUncommitted)
        );
        if refresh {
            let view = t.fresh_view(txn.0);
            if let Some(state) = t.txns.get_mut(&txn.0) {
                state.pinned = view;
            }
        }
        Ok(())
    }

    fn begin(&self, level: IsolationLevel) -> Result<TxnId> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        let id = t.next_txn_id;
        t.next_txn_id += 1;
        t.active.insert(id);
        // Snapshot the FINISHED-instant versions (only when the check will consult them —
        // audit perf note: never clone on the non-SERIALIZABLE fast path).
        let write_versions = if matches!(level, IsolationLevel::Serializable) {
            t.table_write_versions_finished.clone()
        } else {
            HashMap::new()
        };
        let pinned = t.fresh_view(id);
        t.txns.insert(
            id,
            TxnState {
                undo: Vec::new(),
                savepoints: Vec::new(),
                level,
                pinned,
                reads: HashSet::new(),
                locks: Vec::new(),
                // The FINISHED-instant versions at begin (SSI narrowing) — empty for
                // levels that never validate reads.
                write_versions_at_begin: write_versions,
                write_bytes: 0,
            },
        );
        Ok(TxnId(id))
    }

    fn txn_isolation(&self, txn: TxnId) -> Option<IsolationLevel> {
        let t = self.txns.lock().ok()?;
        t.txns.get(&txn.0).map(|t| t.level)
    }

    fn data_version(&self) -> Option<u64> {
        Some(self.data_version.load(Ordering::SeqCst))
    }

    fn commit(&self, txn: TxnId) -> Result<()> {
        // The commit gate makes [SSI check → marker append → staged insert] one atomic step
        // across committers: the check must observe every earlier committer as staged or
        // committed, or two symmetric write-skew transactions could each pass their check
        // before either stages.
        let gate = self.commit_gate.lock().map_err(|_| poisoned())?;
        {
            let t = self.txns.lock().map_err(|_| poisoned())?;
            if !t.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
        }
        // SERIALIZABLE read-write antidependency check: if a row
        // this transaction read was modified by a concurrent transaction that has since committed,
        // the schedule is not serializable — abort it (the caller retries), undoing its writes
        // exactly like a rollback. Done BEFORE the durability point so an aborted transaction
        // leaves no commit marker.
        let conflict = match self.serializable_read_conflict(txn.0) {
            Ok(conflict) => conflict,
            Err(e) => {
                // Defense in depth: the SSI check itself failed
                // (e.g. a page-store I/O error). The transaction is intact and un-staged, so roll
                // it back ourselves before surfacing the error — a forgotten caller rollback must
                // not strand it in `active` with its locks held.
                let state = {
                    let mut t = self.txns.lock().map_err(|_| poisoned())?;
                    t.txns.remove(&txn.0).ok_or_else(|| unknown_txn(txn))?
                };
                drop(gate);
                self.abort(txn, state);
                return Err(e);
            },
        };
        if conflict {
            // Abort exactly like ROLLBACK (same neutralization of non-transactional side effects),
            // then surface the conflict — consistent with SSI's abort-at-commit discipline.
            let state = {
                let mut t = self.txns.lock().map_err(|_| poisoned())?;
                t.txns.remove(&txn.0).ok_or_else(|| unknown_txn(txn))?
                // The transaction stays in `active` until the undo completes (see `abort`).
            };
            drop(gate);
            self.abort(txn, state);
            return Err(Error::SerializationConflict { txn });
        }
        // Durability point (group commit): STAGE the commit under the gate — append the
        // marker (fixing this commit's log order) while the transaction stays in `active`, so no
        // view sees its writes yet — then run the fsync OUTSIDE every latch through the group
        // coordinator, where one fsync serves every commit staged while it was in flight. If the
        // fsync fails, the transaction stays active (the caller may retry or roll back; on
        // replay a later abort marker overrides the possibly-flushed commit marker).
        let seq = match &self.wal {
            // In-memory engine: no durability point; the transaction ends right here (under the
            // gate, so a SERIALIZABLE checker never observes a marker-less in-between).
            None => {
                let deltas = {
                    let mut t = self.txns.lock().map_err(|_| poisoned())?;
                    // In-memory, stage and finish coincide: bump both instants under one lock hold.
                    t.bump_staged_versions(txn.0);
                    t.bump_finished_versions(txn.0);
                    Self::finish_commit(&mut t, txn, &self.data_version)
                };
                // Fold the net row change into the approximate counters with `txns` released.
                self.apply_commit_deltas(&deltas)?;
                return Ok(());
            },
            Some(wal) => {
                // Read-only fast path: a transaction that wrote nothing has
                // no txn-scoped WAL records — no marker to order, nothing to replay — so its
                // commit needs no durability point. Every wire round-trip runs in an implicit
                // transaction, so a plain SELECT otherwise paid a full group-commit fsync
                // (measured ~5ms/query floor on Linux loopback, ~100x the reference round-
                // trip). `undo` captures every txn-scoped write (rollback correctness already
                // depends on that): rows, DDL, index/constraint ops, ANALYZE — while
                // `nextval` is non-transactional and already durable at op time
                // (`log_durable`), so skipping its enclosing commit marker loses nothing.
                {
                    let mut t = self.txns.lock().map_err(|_| poisoned())?;
                    if t.txns.get(&txn.0).is_some_and(|s| s.undo.is_empty()) {
                        // A write-free commit changes no row count — the delta map is empty.
                        let _ = Self::finish_commit(&mut t, txn, &self.data_version);
                        return Ok(());
                    }
                }
                let lsn = {
                    let mut wal = wal.lock().map_err(|_| poisoned())?;
                    // The commit-marker append honors the DST ENOSPC fault point
                    // (`append_commit_marker`); a disk-full failure drives the abort path below.
                    match append_commit_marker(self, &mut wal.writer, txn) {
                        Ok(lsn) => lsn,
                        Err(e) => {
                            drop(wal);
                            // Defense in depth: the commit marker
                            // could not be appended (e.g. ENOSPC). Nothing is staged yet and the
                            // transaction is intact, so roll it back ourselves before surfacing the
                            // error — never leave it stranded in `active` with its locks held.
                            let state = {
                                let mut t = self.txns.lock().map_err(|_| poisoned())?;
                                t.txns.remove(&txn.0).ok_or_else(|| unknown_txn(txn))?
                            };
                            drop(gate);
                            self.abort(txn, state);
                            return Err(e);
                        },
                    }
                };
                let mut t = self.txns.lock().map_err(|_| poisoned())?;
                t.bump_staged_versions(txn.0);
                t.staged.insert(txn.0);
                lsn.0
            },
        };
        drop(gate);
        let flushed = self.group.commit(seq, || {
            let Some(wal) = &self.wal else {
                // Unreachable: `seq` only exists on the durable path.
                return Err(std::io::Error::other("group commit without a WAL"));
            };
            self.flush_and_sync(wal)
        });
        let Ok(mut t) = self.txns.lock() else {
            // The commit is already durable (the fsync above succeeded), so returning an error would
            // lie — the transaction WILL resurrect as committed on restart
            // A poisoned lock means a prior panic left engine state
            // undefined; the only sound response is to stop, letting recovery replay the durable
            // commit honestly. `process::abort`, not `panic!` (a panic in the server's
            // `spawn_blocking` task is caught by the runtime and would keep serving).
            eprintln!(
                "nusadb-btree: FATAL — txns lock poisoned after a durable commit; aborting so \
                 recovery replays the committed transaction on restart"
            );
            std::process::abort();
        };
        t.staged.remove(&txn.0);
        if let Err(e) = flushed {
            // A WAL commit durability failure (write or fsync) is UNRECOVERABLE and MUST stop the
            // process. A failed `fsync` may have left the `CommitTxn`
            // record in the OS page cache while the kernel marked the page clean (the 2018
            // "fsyncgate" hazard) — so the transaction could still reach disk and RESURRECT as
            // committed after a restart, even though returning an error here would report failure
            // to the client. Retrying the fsync is unsound (a second fsync may report success while
            // the data is already lost), and the antidote `AbortTxn` marker is likewise not
            // durably guaranteed. So, taking the standard post-fsyncgate durability stance, do NOT
            // return and keep serving possibly-lost / possibly-resurrecting data — abort the
            // process so recovery replays the durable prefix honestly on restart. `process::abort`
            // (not
            // `panic!`) is deliberate: a panic inside the server's `spawn_blocking` task is caught
            // by the runtime and the process would keep serving.
            drop(t);
            eprintln!(
                "nusadb-btree: FATAL — WAL commit fsync failed ({e}); aborting to preserve \
                 durability (an fsync failure is unrecoverable; the database recovers its durable \
                 prefix on restart)"
            );
            std::process::abort();
        }
        // The writes become visible to new readers here — bump the FINISHED instant while the
        // state (and its undo) is still present.
        t.bump_finished_versions(txn.0);
        let deltas = Self::finish_commit(&mut t, txn, &self.data_version);
        drop(t); // release `txns` before the approximate-counter update takes the catalog guard
        self.apply_commit_deltas(&deltas)?;
        Ok(())
    }

    fn rollback(&self, txn: TxnId) -> Result<()> {
        let state = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            t.txns.remove(&txn.0).ok_or_else(|| unknown_txn(txn))?
            // Still in `active`: the transaction leaves it only after the undo completes (see
            // `abort` — a ReadView equates "ended and present" with committed).
        };
        self.abort(txn, state);
        Ok(())
    }

    fn savepoint(&self, txn: TxnId, name: &str) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        let txn_state = t.txns.get_mut(&txn.0).ok_or_else(|| unknown_txn(txn))?;
        let mark = txn_state.undo.len();
        // A same-named savepoint replaces the older one (SQL semantics).
        txn_state.savepoints.retain(|(n, _)| n != name);
        txn_state.savepoints.push((name.to_owned(), mark));
        Ok(())
    }

    fn rollback_to(&self, txn: TxnId, name: &str) -> Result<()> {
        let tail = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            let txn_state = t.txns.get_mut(&txn.0).ok_or_else(|| unknown_txn(txn))?;
            let Some(pos) = txn_state.savepoints.iter().rposition(|(n, _)| n == name) else {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("savepoint {name} does not exist"),
                )));
            };
            let mark = txn_state.savepoints.get(pos).map_or(0, |(_, m)| *m);
            // Keep the savepoint itself (SQL: ROLLBACK TO leaves it re-usable); drop later ones.
            txn_state.savepoints.truncate(pos + 1);
            txn_state.undo.split_off(mark)
        };
        // Compensations are appended and the memory undo applied under one catalog guard, so
        // replay's view of catalog-shaped inverses can never interleave with a concurrent DDL.
        self.rollback_tail(txn, tail, true)
    }

    fn release_savepoint(&self, txn: TxnId, name: &str) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        let txn_state = t.txns.get_mut(&txn.0).ok_or_else(|| unknown_txn(txn))?;
        let Some(pos) = txn_state.savepoints.iter().rposition(|(n, _)| n == name) else {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("savepoint {name} does not exist"),
            )));
        };
        // Forget the marker and everything after it; all writes stay (RELEASE, not ROLLBACK TO),
        // so a later ROLLBACK TO this name fails.
        txn_state.savepoints.truncate(pos);
        Ok(())
    }

    fn lock_row(&self, txn: TxnId, table: TableId, tid: Tid, mode: RowLockMode) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        if !t.txns.contains_key(&txn.0) {
            return Err(unknown_txn(txn));
        }
        // The shared table intention first, so a concurrent `LOCK TABLE ACCESS EXCLUSIVE`
        // conflicts with this row lock — then the row lock itself. No-wait: a conflict is a
        // `SerializationConflict`, not a block (the no-wait discipline).
        t.lock_table_intention(txn.0, table.0)?;
        t.acquire_lock(
            txn.0,
            LockId::Row {
                table: table.0,
                page: tid.page.0,
                slot: tid.slot.0,
            },
            matches!(mode, RowLockMode::Exclusive),
        )
    }

    fn lock_key(&self, txn: TxnId, table: TableId, key_hash: u64, mode: RowLockMode) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        if !t.txns.contains_key(&txn.0) {
            return Err(unknown_txn(txn));
        }
        // Serializes concurrent writers of the same UNIQUE/PRIMARY KEY value: the second
        // same-key writer aborts at lock time, before its uniqueness scan, closing the snapshot
        // race that would otherwise admit a duplicate under any isolation level.
        t.lock_table_intention(txn.0, table.0)?;
        t.acquire_lock(
            txn.0,
            LockId::Key {
                table: table.0,
                hash: key_hash,
            },
            matches!(mode, RowLockMode::Exclusive),
        )
    }

    fn lock_table(&self, txn: TxnId, table: TableId, mode: TableLockMode) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        if !t.txns.contains_key(&txn.0) {
            return Err(unknown_txn(txn));
        }
        // `ACCESS SHARE` coexists with row/key activity (all shared holds); `ACCESS EXCLUSIVE`
        // requires sole ownership, so it conflicts with every concurrent intention (row writes,
        // row/key locks) on the table.
        t.acquire_lock(
            txn.0,
            LockId::Table { table: table.0 },
            matches!(mode, TableLockMode::AccessExclusive),
        )
    }

    fn create_table(&self, txn: TxnId, def: &TableDef) -> Result<TableId> {
        let tree = ClusteredTree::create(&self.store)?;
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            self.store.deallocate_page(tree.root())?;
            return Err(unknown_txn(txn));
        }
        // A non-public schema must have been created (`CREATE SCHEMA`) first — creating
        // into a missing namespace is rejected loudly, not silently landed in `public`.
        if def.schema != nusadb_core::PUBLIC_SCHEMA && !cat.ns_by_name.contains_key(&def.schema) {
            self.store.deallocate_page(tree.root())?;
            return Err(schema_error(&format!(
                "schema \"{}\" does not exist",
                def.schema
            )));
        }
        let key = (def.schema.clone(), def.name.clone());
        if cat.by_name.contains_key(&key) {
            self.store.deallocate_page(tree.root())?;
            return Err(Error::TableExists {
                name: def.name.clone(),
            });
        }
        let id = cat.next_table_id;
        cat.next_table_id += 1;
        let schema = TableSchema {
            id: TableId(id),
            schema: def.schema.clone(),
            name: def.name.clone(),
            columns: def.columns.clone(),
        };
        cat.tables.insert(
            id,
            TableState {
                schema: schema.clone(),
                root: AtomicU64::new(tree.root().0),
                approx_rows: AtomicU64::new(TableState::APPROX_UNINIT),
                churn_since_analyze: AtomicU64::new(0),
                write: Mutex::new(TableWrite::default()),
                schema_version: 0,
                schema_history: std::iter::once((0, schema)).collect(),
            },
        );
        cat.by_name.insert(key, id);
        self.push_undo(txn.0, UndoOp::CreatedTable { table: id })?;
        // Logged under the catalog write guard: DDL log order equals catalog apply order.
        self.log(
            &LoggedOp::CreateTable {
                txn: txn.0,
                table: id,
                def: def.clone(),
            }
            .to_record(),
        )?;
        Ok(TableId(id))
    }

    fn drop_table(&self, txn: TxnId, table: TableId) -> Result<()> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let state = cat
            .tables
            .remove(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        cat.by_name
            .remove(&(state.schema.schema.clone(), state.schema.name.clone()));
        // Queue the tree for page reclamation; purge frees it once this txn settles, and the
        // rollback path (or a compensated savepoint rollback) removes the entry again.
        self.dropped
            .lock()
            .map_err(|_| poisoned())?
            .push(DroppedPages {
                txn: txn.0,
                root: state.root_id(),
            });
        self.push_undo(
            txn.0,
            UndoOp::DroppedTable {
                table: table.0,
                state,
            },
        )?;
        self.log(
            &LoggedOp::DropTable {
                txn: txn.0,
                table: table.0,
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>> {
        self.lookup_table_in(nusadb_core::PUBLIC_SCHEMA, name)
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        // The btree catalog is not versioned (creates/drops apply to the maps eagerly, undone on
        // rollback), so the map's contents already mirror `lookup_table`'s visibility. Sorted for
        // deterministic output.
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let mut names: Vec<String> = cat.tables.values().map(|t| t.schema.name.clone()).collect();
        names.sort();
        Ok(names)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .by_name
            .get(&(schema.to_owned(), name.to_owned()))
            .and_then(|id| cat.tables.get(id))
            .map(|t| t.schema.clone()))
    }

    fn lookup_table_as_of(&self, txn: TxnId, name: &str) -> Result<Option<TableSchema>> {
        let _ = txn;
        self.lookup_table(name)
    }

    fn lookup_table_as_of_in(
        &self,
        txn: TxnId,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>> {
        let _ = txn;
        self.lookup_table_in(schema, name)
    }

    fn alter_table(&self, txn: TxnId, table: TableId, op: &AlterOp) -> Result<()> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let (previous, previous_version) = {
            let t = cat
                .tables
                .get(&table.0)
                .ok_or_else(|| table_not_found(table))?;
            (t.schema.clone(), t.schema_version)
        };
        // Compute + validate the new schema (errors here leave state untouched).
        let mut new_schema = previous.clone();
        apply_alter(&mut new_schema, op)?;
        // A rename must not collide with another table in the same namespace.
        if (new_schema.name != previous.name || new_schema.schema != previous.schema)
            && cat
                .by_name
                .contains_key(&(new_schema.schema.clone(), new_schema.name.clone()))
        {
            return Err(alter_error(&format!(
                "table {} already exists",
                new_schema.name
            )));
        }
        // Apply the (possible) rename to the by-name index.
        if previous.name != new_schema.name || previous.schema != new_schema.schema {
            cat.by_name
                .remove(&(previous.schema.clone(), previous.name.clone()));
            cat.by_name.insert(
                (new_schema.schema.clone(), new_schema.name.clone()),
                table.0,
            );
        }
        // Advance the schema version; the old version stays in the history.
        let new_version = previous_version
            .checked_add(1)
            .ok_or_else(|| alter_error("schema version overflow"))?;
        if let Some(t) = cat.tables.get_mut(&table.0) {
            t.schema = new_schema.clone();
            t.schema_version = new_version;
            t.schema_history.insert(new_version, new_schema.clone());
        }
        self.push_undo(
            txn.0,
            UndoOp::AlteredSchema {
                table: table.0,
                previous: Box::new(previous),
                previous_version,
                new_version,
            },
        )?;
        self.log(
            &LoggedOp::AlterSchema {
                txn: txn.0,
                table: table.0,
                version: new_version,
                def: TableDef {
                    schema: new_schema.schema,
                    name: new_schema.name,
                    columns: new_schema.columns,
                },
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn create_schema(&self, txn: TxnId, name: &str) -> Result<SchemaId> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        if cat.ns_by_name.contains_key(name) {
            return Err(schema_error(&format!("schema {name} already exists")));
        }
        let id = cat.next_namespace_id;
        cat.next_namespace_id += 1;
        cat.ns_by_name.insert(name.to_owned(), id);
        cat.namespaces.insert(id, name.to_owned());
        self.push_undo(
            txn.0,
            UndoOp::CreatedSchema {
                id,
                name: name.to_owned(),
            },
        )?;
        self.log(
            &LoggedOp::SchemaCreate {
                txn: txn.0,
                id,
                name: name.to_owned(),
            }
            .to_record(),
        )?;
        Ok(SchemaId(id))
    }

    fn drop_schema(&self, txn: TxnId, id: SchemaId, cascade: bool) -> Result<()> {
        // Collect the member tables first (releasing the guard) so RESTRICT can reject before
        // any mutation and CASCADE can drop them through the normal `drop_table` path.
        let (name, members) = {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            if !self.txn_exists(txn.0)? {
                return Err(unknown_txn(txn));
            }
            let Some(name) = cat.namespaces.get(&id.0).cloned() else {
                return Err(schema_not_found(id));
            };
            let members: Vec<TableId> = cat
                .tables
                .values()
                .filter(|t| t.schema.schema == name)
                .map(|t| t.schema.id)
                .collect();
            (name, members)
        };
        if !members.is_empty() && !cascade {
            return Err(Error::DependentObjectsExist(format!(
                "schema \"{name}\" is not empty (use CASCADE to drop its {} table(s))",
                members.len()
            )));
        }
        // CASCADE: drop each member in the same transaction (re-latches internally), so the whole
        // DROP SCHEMA commits or rolls back atomically with the namespace removal.
        for table in members {
            self.drop_table(txn, table)?;
        }
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        let Some(name) = cat.namespaces.remove(&id.0) else {
            return Err(schema_not_found(id));
        };
        cat.ns_by_name.remove(&name);
        self.log(
            &LoggedOp::SchemaDrop {
                txn: txn.0,
                id: id.0,
                name: name.clone(),
            }
            .to_record(),
        )?;
        self.push_undo(txn.0, UndoOp::DroppedSchema { id: id.0, name })?;
        Ok(())
    }

    fn lookup_schema(&self, name: &str) -> Result<Option<SchemaId>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.ns_by_name.get(name).copied().map(SchemaId))
    }

    fn list_schemas(&self) -> Result<Vec<(SchemaId, String)>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .namespaces
            .iter()
            .map(|(&id, name)| (SchemaId(id), name.clone()))
            .collect())
    }

    fn insert(&self, txn: TxnId, table: TableId, tuple: &[u8]) -> Result<Tid> {
        if tuple.len() > MAX_USER_TUPLE {
            return Err(tuple_too_large(tuple.len()));
        }
        // Bound the global resident footprint before growing it: once the in-memory page store has
        // reached the configured ceiling, refuse a new row rather than letting committed data
        // accumulate until the OS OOM-kills the server (a no-op when no ceiling is set). This bounds
        // the streamed-bulk-load case the per-transaction ceiling misses — many small committed
        // batches, each under the per-transaction limit but accumulating resident. Only `insert` is
        // gated, so `DELETE`/`TRUNCATE` stay available to free space at the ceiling.
        self.check_resident_memory()?;
        // Bound this transaction's uncommitted write memory before mutating anything, so an
        // oversized bulk load aborts loudly rather than OOM-killing the server (no-op when no
        // limit is configured). Charge the real retained footprint (logical bytes + fixed per-row
        // overhead), not just `tuple.len()`, so a flood of narrow rows is bounded too.
        self.charge_txn_memory(txn.0, tuple.len() as u64 + PER_ROW_WRITE_OVERHEAD)?;
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        {
            // A row write holds the shared table intention, so `LOCK TABLE ACCESS EXCLUSIVE`
            // genuinely excludes concurrent writers.
            let mut txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.lock_table_intention(txn.0, table.0)?;
        }
        // The per-table writer latch spans mint → tree write → undo push → WAL append, so
        // same-table row ops are totally ordered and the log mirrors that order.
        let mut w = t.write.lock().map_err(|_| poisoned())?;
        let row_id = w.next_row_id;
        w.next_row_id += 1;
        let value = mvcc::encode_row(RowMeta::fresh(txn.0), tuple);
        let mut tree = ClusteredTree::open(&self.store, t.root_id());
        tree.insert(row_id, &value)?;
        t.set_root(tree.root());
        self.push_undo(
            txn.0,
            UndoOp::Inserted {
                table: table.0,
                row_id,
            },
        )?;
        self.log(
            &LoggedOp::Insert {
                txn: txn.0,
                table: table.0,
                row_id,
                tuple: tuple.to_vec(),
            }
            .to_record(),
        )?;
        Ok(tid_of(row_id))
    }

    fn update(&self, txn: TxnId, table: TableId, tid: Tid, tuple: &[u8]) -> Result<Tid> {
        if tuple.len() > MAX_USER_TUPLE {
            return Err(tuple_too_large(tuple.len()));
        }
        // Charge the new version's real footprint against the per-transaction ceiling (see `insert`).
        self.charge_txn_memory(txn.0, tuple.len() as u64 + PER_ROW_WRITE_OVERHEAD)?;
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        {
            // A row write holds the shared table intention.
            let mut txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.lock_table_intention(txn.0, table.0)?;
        }
        let row_id = row_id_of(tid);
        // Under the table latch: read the newest version, admit, park, install, log — one
        // atomic same-table step (two admitted writers over one row are impossible). The tree
        // opens AFTER the latch: only latch holders move the root, so it cannot go stale here.
        let _w = t.write.lock().map_err(|_| poisoned())?;
        let mut tree = ClusteredTree::open(&self.store, t.root_id());
        let old_value = tree.get(row_id)?.ok_or_else(|| tuple_not_found(tid))?;
        let (old_meta, old_tuple) =
            mvcc::decode_row(&old_value).ok_or_else(|| corrupt_row(row_id))?;
        // Admission consults the txn domain AFTER the newest version was read under the table
        // latch: its writer, if concurrent, is still active or already ended — either way the
        // point queries see it (never the reverse race).
        self.txns
            .lock()
            .map_err(|_| poisoned())?
            .admit_write(txn.0, old_meta, tid)?;
        // Park the superseded version BEFORE installing the new one, so a chain-walking reader
        // that sees the new leaf always finds the parked version. A freed slot is reused before
        // the arena grows.
        let parked = UndoVersion {
            meta: old_meta,
            tuple: old_tuple.to_vec(),
        };
        let undo_idx = {
            let mut undo = self.reclaim.write().map_err(|_| poisoned())?;
            if let Some(i) = undo.free.pop() {
                if let Some(slot) = undo.arena.get_mut(usize::try_from(i).unwrap_or(usize::MAX)) {
                    *slot = Some(parked);
                }
                i
            } else {
                undo.arena.push(Some(parked));
                u64::try_from(undo.arena.len().saturating_sub(1)).unwrap_or(mvcc::NO_UNDO)
            }
        };
        let new_value = mvcc::encode_row(
            RowMeta {
                xmin: txn.0,
                xmax: mvcc::NO_XMAX,
                undo: undo_idx,
            },
            tuple,
        );
        tree.update(row_id, &new_value)?;
        t.set_root(tree.root());
        self.push_undo(
            txn.0,
            UndoOp::Updated {
                table: table.0,
                row_id,
                old: old_value,
                undo_idx,
            },
        )?;
        self.log(
            &LoggedOp::Update {
                txn: txn.0,
                table: table.0,
                row_id,
                tuple: tuple.to_vec(),
            }
            .to_record(),
        )?;
        // The row keeps its address (its row-id) across versions.
        Ok(tid)
    }

    fn delete(&self, txn: TxnId, table: TableId, tid: Tid) -> Result<()> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        {
            // A row write holds the shared table intention.
            let mut txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.lock_table_intention(txn.0, table.0)?;
        }
        let row_id = row_id_of(tid);
        // The tree opens AFTER the latch: only latch holders move the root (see `update`).
        let _w = t.write.lock().map_err(|_| poisoned())?;
        let mut tree = ClusteredTree::open(&self.store, t.root_id());
        let old_value = tree.get(row_id)?.ok_or_else(|| tuple_not_found(tid))?;
        let (meta, old_tuple) = mvcc::decode_row(&old_value).ok_or_else(|| corrupt_row(row_id))?;
        // Charge the old row retained in the undo log against the per-transaction ceiling before
        // mutating anything — a mass `DELETE` in one transaction grows the undo log by one old-row
        // copy per row, so it is bounded like `insert`/`update` (no-op when no limit is configured).
        // `old_value` already includes the MVCC header; add the fixed per-row overhead for parity
        // with `insert`/`update`.
        self.charge_txn_memory(txn.0, old_value.len() as u64 + PER_ROW_WRITE_OVERHEAD)?;
        self.txns
            .lock()
            .map_err(|_| poisoned())?
            .admit_write(txn.0, meta, tid)?;
        // Delete stamps xmax in place: old snapshots keep seeing the row, the deleter (once
        // committed) hides it from newer views. Purge reclaims the entry.
        let new_value = mvcc::encode_row(
            RowMeta {
                xmin: meta.xmin,
                xmax: txn.0,
                undo: meta.undo,
            },
            old_tuple,
        );
        tree.update(row_id, &new_value)?;
        t.set_root(tree.root());
        self.push_undo(
            txn.0,
            UndoOp::Deleted {
                table: table.0,
                row_id,
                old: old_value,
            },
        )?;
        self.log(
            &LoggedOp::Delete {
                txn: txn.0,
                table: table.0,
                row_id,
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn scan(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        let (view, serializable) = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            let level = txns
                .txns
                .get(&txn.0)
                .map(|t| t.level)
                .ok_or_else(|| unknown_txn(txn))?;
            (
                txns.view_for(txn.0)?,
                matches!(level, IsolationLevel::Serializable),
            )
        };
        // Latch-free tree walk under the reclamation gate: B-link keeps a concurrent split
        // structurally safe, MVCC stamps hide uncommitted versions, and holding `read` on the
        // gate keeps every undo slot this walk can reach pinned (purge holds `write` to free).
        // Visitor walk (single-copy): each visible tuple is copied exactly once, from the
        // leaf's page buffer straight into its `Arc` — no per-row `Vec` in between.
        let mut rows: Vec<(Tid, SharedTuple)> = Vec::new();
        let mut read_ids: Vec<u64> = Vec::new();
        {
            let undo = self.reclaim.read().map_err(|_| poisoned())?;
            let tree = ClusteredTree::open(&self.store, t.root_id());
            tree.scan_with(|row_id, value| {
                let (meta, tuple) = mvcc::decode_row(value).ok_or_else(|| corrupt_row(row_id))?;
                if let Some(visible) = mvcc::visible_tuple(meta, tuple, &undo.arena, &view) {
                    rows.push((tid_of(row_id), SharedTuple::from(visible)));
                    if serializable {
                        read_ids.push(row_id);
                    }
                }
                Ok(())
            })?;
        }
        // Record the read set for a SERIALIZABLE transaction so a later concurrent write to one of
        // these rows aborts this one at commit.
        if !read_ids.is_empty()
            && let Some(state) = self
                .txns
                .lock()
                .map_err(|_| poisoned())?
                .txns
                .get_mut(&txn.0)
        {
            state
                .reads
                .extend(read_ids.into_iter().map(|row_id| (table.0, row_id)));
        }
        Ok(Box::new(VecScan {
            rows: rows.into_iter(),
        }))
    }

    fn scan_committed(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        // A uniqueness/constraint check must see the LATEST committed state plus this
        // transaction's own writes — never its frozen snapshot: under REPEATABLE READ /
        // SERIALIZABLE a snapshot scan would miss a row a concurrent transaction committed after
        // this one began, letting a duplicate key commit. A fresh view is exactly that state.
        // No SERIALIZABLE read tracking: this is a system constraint scan, not a user
        // observation (the discipline carried over from the predecessor engine).
        let view = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.fresh_view(txn.0)
        };
        let undo = self.reclaim.read().map_err(|_| poisoned())?;
        let tree = ClusteredTree::open(&self.store, t.root_id());
        // Visitor walk (single-copy), same as `scan`.
        let mut rows: Vec<(Tid, SharedTuple)> = Vec::new();
        tree.scan_with(|row_id, value| {
            let (meta, tuple) = mvcc::decode_row(value).ok_or_else(|| corrupt_row(row_id))?;
            if let Some(visible) = mvcc::visible_tuple(meta, tuple, &undo.arena, &view) {
                rows.push((tid_of(row_id), SharedTuple::from(visible)));
            }
            Ok(())
        })?;
        Ok(Box::new(VecScan {
            rows: rows.into_iter(),
        }))
    }

    fn vacuum(&self) -> Result<usize> {
        // `VACUUM`'s btree equivalent is a purge pass: report the reclaimed version count
        // (superseded chain versions freed plus dead rows physically removed).
        let stats = self.purge()?;
        Ok(stats.versions_reclaimed + stats.rows_removed)
    }

    fn create_sequence(&self, txn: TxnId, def: &SequenceDef) -> Result<SequenceId> {
        // Rank order: the txn check (rank 6) precedes the sequence latch (rank 7).
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let id = {
            let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
            if seqs.seq_by_name.contains_key(&def.name) {
                return Err(sequence_error(&format!(
                    "sequence {} already exists",
                    def.name
                )));
            }
            let id = seqs.next_sequence_id;
            seqs.next_sequence_id += 1;
            seqs.seq_by_name.insert(def.name.clone(), id);
            seqs.sequences.insert(
                id,
                SequenceState {
                    def: def.clone(),
                    current: None,
                },
            );
            // Durable immediately (non-transactional create), under the sequence latch so the
            // log order of sequence records matches the apply order; RAM is rolled back if the
            // append fails, so memory never runs ahead of the log.
            if let Err(e) = self.log_durable(
                &LoggedOp::SeqCreate {
                    id,
                    def: def.clone(),
                }
                .to_record(),
            ) {
                seqs.sequences.remove(&id);
                seqs.seq_by_name.remove(&def.name);
                return Err(e);
            }
            id
        };
        // The sequence OBJECT still rolls back with its transaction (a rolled-back
        // `CREATE TABLE ... SERIAL` leaves no phantom sequence): the undo drops it from memory
        // and the rollback path appends the neutralizing SeqDrop.
        self.push_undo(
            txn.0,
            UndoOp::CreatedSequence {
                id,
                name: def.name.clone(),
            },
        )?;
        Ok(SequenceId(id))
    }

    fn drop_sequence(&self, txn: TxnId, id: SequenceId) -> Result<()> {
        // Rank order: the txn check (rank 6) precedes the sequence latch (rank 7).
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
        let name = seqs
            .sequences
            .get(&id.0)
            .map(|seq| seq.def.name.clone())
            .ok_or_else(|| sequence_not_found(id))?;
        // Durable delete FIRST: if the append fails, memory is untouched, so a crash never
        // leaves memory saying "dropped" while the log still replays the sequence.
        self.log_durable(&LoggedOp::SeqDrop { id: id.0 }.to_record())?;
        seqs.sequences.remove(&id.0);
        seqs.seq_by_name.remove(&name);
        Ok(())
    }

    fn lookup_sequence(&self, name: &str) -> Result<Option<SequenceId>> {
        let seqs = self.seqs.lock().map_err(|_| poisoned())?;
        Ok(seqs.seq_by_name.get(name).copied().map(SequenceId))
    }

    fn sequence_next(&self, id: SequenceId) -> Result<i64> {
        let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
        let seq = seqs
            .sequences
            .get_mut(&id.0)
            .ok_or_else(|| sequence_not_found(id))?;
        let prior = seq.current;
        let next = advance_sequence(seq)?;
        // Fsync the advance under the sequence latch (concurrent nextvals serialize, the logged
        // value is monotonic — and the group coordinator still shares the fsync with concurrent
        // commits) BEFORE returning the value — a crash after the return can then never hand
        // the same number out twice. Memory rolls back if the append fails.
        if let Err(e) = self.log_durable(
            &LoggedOp::SeqSet {
                id: id.0,
                value: next,
            }
            .to_record(),
        ) {
            if let Some(seq) = seqs.sequences.get_mut(&id.0) {
                seq.current = prior;
            }
            return Err(e);
        }
        Ok(next)
    }

    fn sequence_current(&self, id: SequenceId) -> Result<i64> {
        let seqs = self.seqs.lock().map_err(|_| poisoned())?;
        let current = seqs
            .sequences
            .get(&id.0)
            .ok_or_else(|| sequence_not_found(id))?
            .current;
        current.ok_or_else(|| sequence_error("currval is not yet defined (call nextval first)"))
    }

    fn sequence_set(&self, id: SequenceId, value: i64) -> Result<()> {
        let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
        let seq = seqs
            .sequences
            .get_mut(&id.0)
            .ok_or_else(|| sequence_not_found(id))?;
        let prior = seq.current;
        seq.current = Some(value);
        if let Err(e) = self.log_durable(&LoggedOp::SeqSet { id: id.0, value }.to_record()) {
            if let Some(seq) = seqs.sequences.get_mut(&id.0) {
                seq.current = prior;
            }
            return Err(e);
        }
        Ok(())
    }

    fn create_index(&self, txn: TxnId, def: &IndexDef) -> Result<IndexId> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        if cat.idx_by_name.contains_key(&def.name) {
            return Err(Error::ConstraintViolation(format!(
                "index {} already exists",
                def.name
            )));
        }
        if !cat.tables.contains_key(&def.table.0) {
            return Err(table_not_found(def.table));
        }
        let id = cat.next_index_id;
        cat.next_index_id += 1;
        cat.idx_by_name.insert(def.name.clone(), id);
        cat.indexes.insert(
            id,
            IndexState {
                def: def.clone(),
                // Complete from birth: the creating statement backfills existing rows in the same
                // transaction, and every later write maintains the entries.
                complete: true,
                data: RwLock::new(IndexData::default()),
            },
        );
        self.push_undo(txn.0, UndoOp::CreatedIndex { index: id })?;
        self.log(
            &LoggedOp::CreateIndex {
                txn: txn.0,
                index: id,
                def: def.clone(),
            }
            .to_record(),
        )?;
        Ok(IndexId(id))
    }

    fn drop_index(&self, txn: TxnId, id: IndexId) -> Result<()> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let state = cat
            .indexes
            .remove(&id.0)
            .ok_or_else(|| index_not_found(id))?;
        cat.idx_by_name.remove(&state.def.name);
        self.push_undo(txn.0, UndoOp::DroppedIndex { index: id.0, state })?;
        self.log(
            &LoggedOp::DropIndex {
                txn: txn.0,
                index: id.0,
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn lookup_index(&self, name: &str) -> Result<Option<IndexId>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.idx_by_name.get(name).copied().map(IndexId))
    }

    fn list_indexes(&self, table: TableId) -> Result<Vec<IndexDef>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .indexes
            .values()
            .filter(|i| i.def.table == table)
            .map(|i| i.def.clone())
            .collect())
    }

    fn index_is_complete(&self, index: IndexId) -> Result<bool> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.indexes.get(&index.0).is_some_and(|i| i.complete))
    }

    fn index_insert(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let row_id = row_id_of(tid);
        let idx = cat
            .indexes
            .get(&index.0)
            .ok_or_else(|| index_not_found(index))?;
        let (unique, table, name) = (idx.def.unique, idx.def.table, idx.def.name.as_str());
        // A **constraint-backing** index is exempted from the byte-level uniqueness check (a
        // deliberate layering choice): PRIMARY KEY / UNIQUE semantics are owned by the SQL layer's
        // scan-based checks + key locks (NULL keys never conflict; a statement may pass through
        // a transient duplicate), so backing entries are maintained purely as a lookup structure.
        let backing = cat
            .constraints
            .get(&table.0)
            .is_some_and(|cs| cs.iter().any(|c| c.index == index.0));
        // The index write latch spans check → apply → undo push → WAL append: the uniqueness
        // decision and the entry mutation are one atomic step (two same-key inserters cannot
        // interleave between them), and same-index log order equals apply order.
        let mut data = idx.data.write().map_err(|_| poisoned())?;
        // Uniqueness: reject if `key` already maps to a *live* (newest version not deleted) row
        // other than `tid`. Stale entries (rolled-back / deleted / superseded rows) don't count,
        // so an UPDATE that re-inserts the same key after deleting the old entry is fine.
        if unique && !backing {
            let others: Vec<u64> = data
                .entries
                .get(key)
                .map(|rows| {
                    rows.iter()
                        // Only an alive range can conflict: a dead-stamped one belongs to a
                        // superseded version of its row (the row has since moved to another key).
                        .filter(|&(&r, metas)| {
                            r != row_id && metas.iter().any(|m| m.xmax == mvcc::NO_XMAX)
                        })
                        .map(|(&r, _)| r)
                        .collect()
                })
                .unwrap_or_default();
            if !others.is_empty()
                && let Some(t) = cat.tables.get(&table.0)
            {
                let tree = ClusteredTree::open(&self.store, t.root_id());
                for other in others {
                    if let Some(value) = tree.get(other)? {
                        let (meta, _) =
                            mvcc::decode_row(&value).ok_or_else(|| corrupt_row(other))?;
                        if meta.xmax == mvcc::NO_XMAX {
                            return Err(Error::ConstraintViolation(format!(
                                "duplicate key violates unique index {name}"
                            )));
                        }
                    }
                }
            }
        }
        let owned = key.to_vec();
        let applied = data.apply_insert(&owned, row_id, txn.0);
        // A same-key re-insert (an UPDATE that did not move the key) changed nothing, so nothing
        // may be undone — recording an undo for it is exactly the
        // Bug (rollback would strip the committed entry).
        if let AppliedInsert::Inserted { stamped } = applied {
            self.push_undo(
                txn.0,
                UndoOp::IndexInserted {
                    index: index.0,
                    key: owned.clone(),
                    row_id,
                    stamped,
                },
            )?;
        }
        self.log(
            &LoggedOp::IndexInsert {
                txn: txn.0,
                index: index.0,
                row_id,
                key: owned,
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn index_insert_batch(
        &self,
        txn: TxnId,
        index: IndexId,
        mut entries: Vec<(Vec<u8>, Tid)>,
    ) -> Result<()> {
        // Apply the entries in key order rather than the caller's row order. The index is a sorted
        // map keyed by these bytes, so a key-ordered batch turns the random node descents a bulk
        // load's row order would cause into sequential, cache-warm inserts. Ordering changes neither
        // the final index state nor the uniqueness outcome (two entries for one key still collide
        // once both are seen). Each entry goes through the same per-entry check + apply + undo + WAL
        // as `index_insert`, and the latch is released between entries, so a large batch stays
        // cooperative with concurrent readers and a crash mid-batch rolls back with the transaction —
        // fully indexed or not at all — exactly as the per-row path.
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        for (key, tid) in entries {
            self.index_insert(txn, index, &key, tid)?;
        }
        Ok(())
    }

    fn index_delete(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        let row_id = row_id_of(tid);
        let idx = cat
            .indexes
            .get(&index.0)
            .ok_or_else(|| index_not_found(index))?;
        let mut data = idx.data.write().map_err(|_| poisoned())?;
        let removed = data.apply_delete(key, row_id);
        if let Some(meta) = removed {
            self.push_undo(
                txn.0,
                UndoOp::IndexDeleted {
                    index: index.0,
                    key: key.to_vec(),
                    row_id,
                    meta,
                },
            )?;
            self.log(
                &LoggedOp::IndexDelete {
                    txn: txn.0,
                    index: index.0,
                    row_id,
                    key: key.to_vec(),
                }
                .to_record(),
            )?;
        }
        Ok(())
    }

    fn index_scan(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let (view, serializable) = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            let level = txns
                .txns
                .get(&txn.0)
                .map(|t| t.level)
                .ok_or_else(|| unknown_txn(txn))?;
            (
                txns.view_for(txn.0)?,
                matches!(level, IsolationLevel::Serializable),
            )
        };
        let idx = cat
            .indexes
            .get(&index.0)
            .ok_or_else(|| index_not_found(index))?;
        let table_id = idx.def.table.0;
        let t = cat
            .tables
            .get(&table_id)
            .ok_or_else(|| table_not_found(idx.def.table))?;
        // Ascending key order over `[lo, hi]`. Two visibility hops per entry: the ENTRY's own
        // stamps first — the row keeps its address across versions, so only the stamps can tell
        // this reader whether its visible version carries this key (an `UPDATE` that moved the
        // row dead-stamps the old key's entry and its new key's entry is unseen by older
        // snapshots) — then the base row under the caller's view (a rolled-back or deleted row
        // resolves to nothing). Entry iteration holds the index read latch; the base-row hop is
        // a latch-free tree read under the reclamation gate.
        let mut rows: Vec<(Tid, SharedTuple)> = Vec::new();
        let mut read_ids: Vec<u64> = Vec::new();
        {
            let data = idx.data.read().map_err(|_| poisoned())?;
            let undo = self.reclaim.read().map_err(|_| poisoned())?;
            let tree = ClusteredTree::open(&self.store, t.root_id());
            for (_key, entry_rows) in data
                .entries
                .range::<[u8], _>((as_slice_bound(&lo), as_slice_bound(&hi)))
            {
                for (&row_id, metas) in entry_rows {
                    if !IndexData::entry_visible(metas, &view) {
                        continue;
                    }
                    let Some(value) = tree.get(row_id)? else {
                        continue;
                    };
                    let (meta, tuple) =
                        mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
                    if let Some(visible) = mvcc::visible_tuple(meta, tuple, &undo.arena, &view) {
                        rows.push((tid_of(row_id), SharedTuple::from(visible)));
                        if serializable {
                            read_ids.push(row_id);
                        }
                    }
                }
            }
        }
        // Record the read set for a SERIALIZABLE transaction: an
        // index scan reads only the matching rows, so this is the narrower read set a
        // PK/secondary-key predicate produces.
        if !read_ids.is_empty()
            && let Some(state) = self
                .txns
                .lock()
                .map_err(|_| poisoned())?
                .txns
                .get_mut(&txn.0)
        {
            state
                .reads
                .extend(read_ids.into_iter().map(|row_id| (table_id, row_id)));
        }
        Ok(Box::new(VecScan {
            rows: rows.into_iter(),
        }))
    }

    fn index_scan_committed(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        // Latest-committed visibility (a fresh view), never the frozen snapshot: a uniqueness probe
        // must see a key another transaction committed after this one began (mirrors `scan_committed`).
        // No SERIALIZABLE read tracking — this is a system constraint probe, not a user observation.
        let view = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.fresh_view(txn.0)
        };
        let idx = cat
            .indexes
            .get(&index.0)
            .ok_or_else(|| index_not_found(index))?;
        let table_id = idx.def.table.0;
        let t = cat
            .tables
            .get(&table_id)
            .ok_or_else(|| table_not_found(idx.def.table))?;
        // Same two-hop visibility as `index_scan` (entry stamps, then the base row), but under the
        // fresh view.
        let mut rows: Vec<(Tid, SharedTuple)> = Vec::new();
        {
            let data = idx.data.read().map_err(|_| poisoned())?;
            let undo = self.reclaim.read().map_err(|_| poisoned())?;
            let tree = ClusteredTree::open(&self.store, t.root_id());
            for (_key, entry_rows) in data
                .entries
                .range::<[u8], _>((as_slice_bound(&lo), as_slice_bound(&hi)))
            {
                for (&row_id, metas) in entry_rows {
                    if !IndexData::entry_visible(metas, &view) {
                        continue;
                    }
                    let Some(value) = tree.get(row_id)? else {
                        continue;
                    };
                    let (meta, tuple) =
                        mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
                    if let Some(visible) = mvcc::visible_tuple(meta, tuple, &undo.arena, &view) {
                        rows.push((tid_of(row_id), SharedTuple::from(visible)));
                    }
                }
            }
        }
        Ok(Box::new(VecScan {
            rows: rows.into_iter(),
        }))
    }

    fn add_unique_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        columns: &[String],
        primary: bool,
    ) -> Result<IndexId> {
        // Create the backing unique index first (it takes the state latch internally). If the
        // single-PK check below rejects this, the index was created within this transaction and
        // is undone when the caller rolls back (the undo contract carried over from the predecessor engine).
        let index = self.create_index(
            txn,
            &IndexDef {
                name: name.to_owned(),
                table,
                columns: columns.to_vec(),
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: true,
            },
        )?;
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        // At most one PRIMARY KEY per table — checked under the same guard as the insert below.
        if primary
            && cat
                .constraints
                .get(&table.0)
                .is_some_and(|cs| cs.iter().any(|c| c.primary))
        {
            return Err(Error::ConstraintViolation(format!(
                "table {} already has a primary key",
                table.0
            )));
        }
        cat.constraints
            .entry(table.0)
            .or_default()
            .push(UniqueState {
                name: name.to_owned(),
                columns: columns.to_vec(),
                primary,
                index: index.0,
            });
        self.push_undo(
            txn.0,
            UndoOp::AddedConstraint {
                table: table.0,
                name: name.to_owned(),
            },
        )?;
        self.log(
            &LoggedOp::AddUnique {
                txn: txn.0,
                table: table.0,
                index: index.0,
                name: name.to_owned(),
                columns: columns.to_vec(),
                primary,
            }
            .to_record(),
        )?;
        Ok(index)
    }

    fn add_check_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        expr: &[u8],
    ) -> Result<()> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        if !cat.tables.contains_key(&table.0) {
            return Err(table_not_found(table));
        }
        if cat
            .checks
            .get(&table.0)
            .is_some_and(|cs| cs.iter().any(|c| c.name == name))
        {
            return Err(Error::ConstraintViolation(format!(
                "check constraint {name} already exists on this table"
            )));
        }
        cat.checks.entry(table.0).or_default().push(CheckState {
            name: name.to_owned(),
            expr: expr.to_vec(),
        });
        self.push_undo(
            txn.0,
            UndoOp::AddedCheck {
                table: table.0,
                name: name.to_owned(),
            },
        )?;
        self.log(
            &LoggedOp::AddCheck {
                txn: txn.0,
                table: table.0,
                name: name.to_owned(),
                expr: expr.to_vec(),
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn drop_constraint(&self, txn: TxnId, table: TableId, name: &str) -> Result<()> {
        // A CHECK constraint has no backing index — handle it first.
        let backing_index = {
            let mut cat = self.catalog.write().map_err(|_| poisoned())?;
            if !self.txn_exists(txn.0)? {
                return Err(unknown_txn(txn));
            }
            let removed_check = cat.checks.get_mut(&table.0).and_then(|list| {
                list.iter()
                    .position(|c| c.name == name)
                    .map(|p| list.remove(p))
            });
            if let Some(check) = removed_check {
                self.push_undo(
                    txn.0,
                    UndoOp::DroppedCheck {
                        table: table.0,
                        state: check,
                    },
                )?;
                self.log(
                    &LoggedOp::DropConstraint {
                        txn: txn.0,
                        table: table.0,
                        name: name.to_owned(),
                    }
                    .to_record(),
                )?;
                return Ok(());
            }
            // A FOREIGN KEY declared on this (child) table: remove the record, then drop its
            // child-side backing index below (outside the guard — drop_index re-acquires).
            if cat
                .foreign_keys
                .get(name)
                .is_some_and(|fk| fk.child_table == table.0)
            {
                let Some(fk) = cat.foreign_keys.remove(name) else {
                    return Err(constraint_not_found(table, name));
                };
                let child_index = fk.child_index;
                self.push_undo(txn.0, UndoOp::DroppedForeignKey { state: fk })?;
                self.log(
                    &LoggedOp::DropConstraint {
                        txn: txn.0,
                        table: table.0,
                        name: name.to_owned(),
                    }
                    .to_record(),
                )?;
                child_index
            } else {
                // A UNIQUE / PRIMARY KEY backed by an index. RESTRICT: refuse if a foreign key
                // references its backing index (drop the FK first) — the safe drop order.
                let (pos, this_index) = {
                    let list = cat
                        .constraints
                        .get(&table.0)
                        .ok_or_else(|| constraint_not_found(table, name))?;
                    list.iter()
                        .enumerate()
                        .find(|(_, c)| c.name == name)
                        .map(|(i, c)| (i, c.index))
                        .ok_or_else(|| constraint_not_found(table, name))?
                };
                if let Some(fk) = cat
                    .foreign_keys
                    .values()
                    .find(|f| f.parent_index == this_index)
                {
                    return Err(Error::ConstraintViolation(format!(
                        "cannot drop constraint {name}: foreign key {} references it (drop the foreign key first)",
                        fk.name
                    )));
                }
                let Some(list) = cat.constraints.get_mut(&table.0) else {
                    return Err(constraint_not_found(table, name));
                };
                let state = list.remove(pos);
                let index = state.index;
                self.push_undo(
                    txn.0,
                    UndoOp::DroppedConstraint {
                        table: table.0,
                        state,
                    },
                )?;
                self.log(
                    &LoggedOp::DropConstraint {
                        txn: txn.0,
                        table: table.0,
                        name: name.to_owned(),
                    }
                    .to_record(),
                )?;
                index
            }
        };
        self.drop_index(txn, IndexId(backing_index))
    }

    fn list_constraints(&self, table: TableId) -> Result<Vec<Constraint>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let mut out: Vec<Constraint> = cat.constraints.get(&table.0).map_or_else(Vec::new, |cs| {
            cs.iter()
                .map(|c| Constraint {
                    name: c.name.clone(),
                    table,
                    columns: c.columns.clone(),
                    kind: if c.primary {
                        ConstraintKind::PrimaryKey
                    } else {
                        ConstraintKind::Unique
                    },
                    index: Some(IndexId(c.index)),
                    expr: None,
                })
                .collect()
        });
        for fk in cat
            .foreign_keys
            .values()
            .filter(|f| f.child_table == table.0)
        {
            out.push(Constraint {
                name: fk.name.clone(),
                table,
                columns: fk.child_columns.clone(),
                kind: ConstraintKind::ForeignKey,
                index: Some(IndexId(fk.child_index)),
                expr: None,
            });
        }
        if let Some(cs) = cat.checks.get(&table.0) {
            for c in cs {
                out.push(Constraint {
                    name: c.name.clone(),
                    table,
                    columns: Vec::new(),
                    kind: ConstraintKind::Check,
                    index: None,
                    expr: Some(c.expr.clone()),
                });
            }
        }
        Ok(out)
    }

    fn has_unique_constraint(&self, table: TableId) -> Result<bool> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .constraints
            .get(&table.0)
            .is_some_and(|cs| !cs.is_empty()))
    }

    fn list_foreign_keys(&self, table: TableId) -> Result<Vec<ForeignKeyDef>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .foreign_keys
            .values()
            .filter(|f| f.child_table == table.0 || f.parent_table == table.0)
            .map(|f| {
                // Report the columns of the actual referenced key (which may be a non-PK
                // UNIQUE), resolved through the parent's backing index.
                let parent_columns = cat
                    .constraints
                    .get(&f.parent_table)
                    .and_then(|cs| cs.iter().find(|c| c.index == f.parent_index))
                    .map(|c| c.columns.clone())
                    .unwrap_or_default();
                ForeignKeyDef {
                    name: f.name.clone(),
                    child_table: TableId(f.child_table),
                    child_columns: f.child_columns.clone(),
                    parent_table: TableId(f.parent_table),
                    parent_columns,
                    on_delete: f.on_delete,
                    on_update: f.on_update,
                }
            })
            .collect())
    }

    fn add_foreign_key(&self, txn: TxnId, def: &ForeignKeyDef) -> Result<IndexId> {
        // Validate under a brief read guard before creating the backing index.
        let parent_index = {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            if !self.txn_exists(txn.0)? {
                return Err(unknown_txn(txn));
            }
            if !cat.tables.contains_key(&def.child_table.0) {
                return Err(table_not_found(def.child_table));
            }
            if !cat.tables.contains_key(&def.parent_table.0) {
                return Err(table_not_found(def.parent_table));
            }
            if cat.foreign_keys.contains_key(&def.name) {
                return Err(Error::ConstraintViolation(format!(
                    "foreign key {} already exists",
                    def.name
                )));
            }
            let parents = cat.constraints.get(&def.parent_table.0);
            let referenced = if def.parent_columns.is_empty() {
                // No referenced columns named ⇒ the parent's PRIMARY KEY (preferred) or, failing
                // that, any UNIQUE constraint.
                let pk = parents.and_then(|cs| cs.iter().find(|c| c.primary));
                pk.or_else(|| parents.and_then(|cs| cs.first()))
            } else {
                parents.and_then(|cs| cs.iter().find(|c| c.columns == def.parent_columns))
            };
            let Some(referenced) = referenced else {
                return Err(Error::ConstraintViolation(format!(
                    "foreign key {} references table {} which has no matching primary key or unique constraint",
                    def.name, def.parent_table.0
                )));
            };
            referenced.index
        };
        // The child-side (non-unique) index over the FK columns (re-acquires internally).
        let child_index = self.create_index(
            txn,
            &IndexDef {
                name: def.name.clone(),
                table: def.child_table,
                columns: def.child_columns.clone(),
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: false,
            },
        )?;
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        cat.foreign_keys.insert(
            def.name.clone(),
            FkState {
                name: def.name.clone(),
                child_table: def.child_table.0,
                child_columns: def.child_columns.clone(),
                parent_table: def.parent_table.0,
                parent_index,
                child_index: child_index.0,
                on_delete: def.on_delete,
                on_update: def.on_update,
            },
        );
        self.push_undo(
            txn.0,
            UndoOp::AddedForeignKey {
                name: def.name.clone(),
                child_table: def.child_table.0,
            },
        )?;
        self.log(
            &LoggedOp::AddFk {
                txn: txn.0,
                name: def.name.clone(),
                child_table: def.child_table.0,
                child_columns: def.child_columns.clone(),
                parent_table: def.parent_table.0,
                parent_index,
                child_index: child_index.0,
                on_delete: def.on_delete,
                on_update: def.on_update,
            }
            .to_record(),
        )?;
        Ok(child_index)
    }

    fn fk_check(&self, txn: TxnId, name: &str, key: &[u8]) -> Result<()> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let view = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            if !txns.txns.contains_key(&txn.0) {
                return Err(unknown_txn(txn));
            }
            txns.view_for(txn.0)?
        };
        let fk = cat
            .foreign_keys
            .get(name)
            .ok_or_else(|| fk_not_found(name))?;
        let exists = !self
            .visible_rows_for_index_key(&cat, &view, fk.parent_index, fk.parent_table, key)?
            .is_empty();
        if exists {
            Ok(())
        } else {
            Err(Error::ConstraintViolation(format!(
                "foreign key {name}: referenced key not present in parent"
            )))
        }
    }

    fn fk_on_delete(&self, txn: TxnId, parent_table: TableId, parent_key: &[u8]) -> Result<u64> {
        // Under the read guard: gather the dependent child rows per FK referencing this parent.
        let mut cascade: Vec<(u64, u64)> = Vec::new();
        {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            let view = {
                let txns = self.txns.lock().map_err(|_| poisoned())?;
                if !txns.txns.contains_key(&txn.0) {
                    return Err(unknown_txn(txn));
                }
                txns.view_for(txn.0)?
            };
            for fk in cat
                .foreign_keys
                .values()
                .filter(|f| f.parent_table == parent_table.0)
            {
                let children = self.visible_rows_for_index_key(
                    &cat,
                    &view,
                    fk.child_index,
                    fk.child_table,
                    parent_key,
                )?;
                if children.is_empty() {
                    continue;
                }
                match fk.on_delete {
                    FkAction::Cascade => {
                        for row_id in children {
                            cascade.push((fk.child_table, row_id));
                        }
                    },
                    FkAction::Restrict | FkAction::NoAction => {
                        return Err(Error::ConstraintViolation(format!(
                            "foreign key {}: {} dependent row(s) remain on the referenced row",
                            fk.name,
                            children.len()
                        )));
                    },
                    FkAction::SetNull | FkAction::SetDefault => {
                        return Err(Error::ConstraintViolation(format!(
                            "foreign key {}: SET NULL/SET DEFAULT requires a SQL-layer row rewrite",
                            fk.name
                        )));
                    },
                }
            }
        }
        // Delete the cascaded children outside the guard (delete re-acquires).
        let count = u64::try_from(cascade.len()).unwrap_or(u64::MAX);
        for (child_table, row_id) in cascade {
            self.delete(txn, TableId(child_table), tid_of(row_id))?;
        }
        Ok(count)
    }

    fn analyze_table(&self, txn: TxnId, table: TableId, stats: &TableStats) -> Result<()> {
        let mut cat = self.catalog.write().map_err(|_| poisoned())?;
        if !self.txn_exists(txn.0)? {
            return Err(unknown_txn(txn));
        }
        if !cat.tables.contains_key(&table.0) {
            return Err(table_not_found(table));
        }
        let previous = cat.stats.insert(table.0, stats.clone());
        // Statistics are now fresh: clear the auto-analyze churn tally for this table. (If this
        // transaction later rolls back the stats revert via the undo op below; the churn reset is a
        // benign hint that simply re-accumulates — it never affects correctness.)
        if let Some(state) = cat.tables.get(&table.0) {
            state.reset_churn();
        }
        self.push_undo(
            txn.0,
            UndoOp::AnalyzedTable {
                table: table.0,
                previous: previous.map(Box::new),
            },
        )?;
        self.log(
            &LoggedOp::SetStats {
                txn: txn.0,
                table: table.0,
                stats: stats.clone(),
            }
            .to_record(),
        )?;
        Ok(())
    }

    fn schema_for_version(&self, table: TableId, version: u32) -> Result<Option<TableSchema>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat
            .tables
            .get(&table.0)
            .and_then(|t| t.schema_history.get(&version).cloned()))
    }

    fn current_schema_version(&self, table: TableId) -> Result<Option<u32>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.tables.get(&table.0).map(|t| t.schema_version))
    }

    fn table_stats(&self, table: TableId) -> Result<Option<TableStats>> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.stats.get(&table.0).cloned())
    }

    fn row_count(&self, table: TableId) -> Result<u64> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        // Live + committed: the newest version is undeleted and its creator has ended (an ended
        // transaction present in the tree is committed — rollback erases physically). The active
        // set is snapshotted once: a transaction that ends mid-scan flips rows from "not counted"
        // to "counted" either way, the same race a single-latch count had against a commit
        // waiting on the latch.
        let active = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            txns.active.clone()
        };
        let tree = ClusteredTree::open(&self.store, t.root_id());
        let mut count: u64 = 0;
        for (row_id, value) in tree.scan()? {
            let (meta, _) = mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
            if meta.xmax == mvcc::NO_XMAX && !active.contains(&meta.xmin) {
                count += 1;
            }
        }
        Ok(count)
    }

    fn approx_row_count(&self, table: TableId) -> Result<u64> {
        // Fast path: an initialized counter is a single atomic load under the shared catalog guard.
        let cached = {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            cat.tables
                .get(&table.0)
                .ok_or_else(|| table_not_found(table))?
                .approx_rows_raw()
        };
        if cached != TableState::APPROX_UNINIT {
            return Ok(cached);
        }
        // First access (or post-restart, where the in-memory counter is 0 but the tree holds rows):
        // fill it from an O(n) walk. `row_count` takes its own catalog guard, so ours must be
        // released first (a std `RwLock` read is not reliably re-entrant). A commit that lands
        // between the walk and the store skips the still-uninitialized counter, and `init_approx_rows`
        // only fills if still `UNINIT`, so the estimate is at worst off by the writes committed during
        // the walk — a bounded, one-time approximation, fine for a routing hint.
        let counted = self.row_count(table)?;
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let t = cat
            .tables
            .get(&table.0)
            .ok_or_else(|| table_not_found(table))?;
        t.init_approx_rows(counted);
        Ok(t.approx_rows_raw())
    }

    fn churn_since_analyze(&self, table: TableId) -> Result<u64> {
        // A single atomic load under the shared catalog guard; `0` for an unknown or freshly-analysed
        // table. Maintained per commit ([`TableState::add_churn`]) and reset by `analyze_table`.
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        Ok(cat.tables.get(&table.0).map_or(0, TableState::churn_raw))
    }
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each sharded guard IS the critical section of its domain: dropping it earlier \
              than its last use would race the very invariant it guards (see the latching \
              discipline on the struct docs)"
)]
impl BtreeEngine {
    /// Whether `txn` is a known (begun, not yet ended) transaction — the guard every mutating
    /// call runs first.
    fn txn_exists(&self, txn: u64) -> Result<bool> {
        Ok(self
            .txns
            .lock()
            .map_err(|_| poisoned())?
            .txns
            .contains_key(&txn))
    }

    /// Record the inverse of an applied write on `txn`'s undo list (rank 6, O(1) critical
    /// section). Tolerates an unknown transaction exactly like the old in-latch
    /// `if let Some(t) = txns.get_mut(..)` did.
    fn push_undo(&self, txn: u64, op: UndoOp) -> Result<()> {
        if let Some(t) = self.txns.lock().map_err(|_| poisoned())?.txns.get_mut(&txn) {
            t.undo.push(op);
        }
        Ok(())
    }

    /// Charge `bytes` of uncommitted row memory to `txn` against the optional per-transaction
    /// ceiling, **before** the write mutates anything — so a rejection leaves no partial state and
    /// the transaction aborts through the ordinary undo path. `None` limit (the default) charges
    /// and never rejects. An unknown `txn` is tolerated (the caller's own existence check reports
    /// it); the running total is discarded when the transaction ends.
    fn charge_txn_memory(&self, txn: u64, bytes: u64) -> Result<()> {
        // No limit configured (the default): charge nothing and, crucially, take no lock — so the
        // common bulk-write path keeps its exact prior cost (one `txns` acquisition per row, not two).
        let Some(limit) = self.max_txn_write_bytes else {
            return Ok(());
        };
        let mut guard = self.txns.lock().map_err(|_| poisoned())?;
        if let Some(t) = guard.txns.get_mut(&txn) {
            let next = t.write_bytes.saturating_add(bytes);
            if next > limit {
                return Err(txn_memory_exceeded(limit, next));
            }
            t.write_bytes = next;
        }
        Ok(())
    }

    /// Reject a row `insert` when the in-memory page store has grown to the configured global
    /// resident-memory ceiling, **before** the write mutates anything — so a rejection leaves no
    /// partial state and the transaction aborts through the ordinary undo path. `None` limit (the
    /// default) takes no lock and never rejects, keeping the common bulk-write path at its exact
    /// prior cost. The check is `resident >= limit` rather than a precise per-row projection: the
    /// store grows one 8 KiB page at a time, so it can only overshoot the ceiling by a page or so
    /// before the next `insert` is refused — negligible against a ceiling sized with headroom.
    fn check_resident_memory(&self) -> Result<()> {
        let Some(limit) = self.max_total_resident_bytes else {
            return Ok(());
        };
        let resident = self.store.resident_bytes()?;
        if resident >= limit {
            return Err(resident_memory_exceeded(limit, resident));
        }
        Ok(())
    }

    /// Whether undoing `ops` mutates the catalog maps (DDL-shaped inverses) — those need the
    /// catalog write guard; row/index inverses run under `read` (the per-object latches do the
    /// real exclusion).
    fn undo_needs_catalog_write(ops: &[UndoOp]) -> bool {
        ops.iter().any(|op| {
            !matches!(
                op,
                UndoOp::Inserted { .. }
                    | UndoOp::Updated { .. }
                    | UndoOp::Deleted { .. }
                    | UndoOp::IndexInserted { .. }
                    | UndoOp::IndexDeleted { .. }
                    | UndoOp::CreatedSequence { .. }
            )
        })
    }

    /// Apply the in-memory inverses of `ops` under ONE catalog guard — and, iff `compensate`
    /// (the savepoint path: the transaction may still commit, so replay needs logical inverses),
    /// append the compensation records under that same guard, so replay's view of
    /// catalog-shaped inverses can never interleave with a concurrent DDL. A full abort passes
    /// `false`: replay excludes an uncommitted transaction wholesale, no compensation needed.
    ///
    /// # Errors
    /// Propagates WAL append/fsync and page-store failures.
    fn rollback_tail(&self, txn: TxnId, ops: Vec<UndoOp>, compensate: bool) -> Result<()> {
        if Self::undo_needs_catalog_write(&ops) {
            let mut cat = self.catalog.write().map_err(|_| poisoned())?;
            if compensate {
                self.log_compensations(txn, &ops)?;
            }
            self.undo_ops(&mut CatalogRef::Write(&mut cat), txn.0, ops)
        } else {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            if compensate {
                self.log_compensations(txn, &ops)?;
            }
            self.undo_ops(&mut CatalogRef::Read(&cat), txn.0, ops)
        }
    }

    /// Undo `txn`'s applied writes, release its locks, and neutralize any non-transactional side
    /// effect it logged — the shared body of `rollback` and the `SERIALIZABLE` commit-time abort,
    /// so the two can never drift. The caller has removed `txn` from the txn map (no further ops
    /// can join) but **left it in `active`**: a [`ReadView`] equates "ended and present" with
    /// committed, so the transaction may only leave `active` — here, last — once every one of
    /// its versions is physically gone. Locks release last too, so a constraint path guarded by
    /// a key lock never observes a mid-undo index state.
    ///
    /// Always succeeds or stops the process: the in-memory undo cannot fail except on a poisoned
    /// lock (an already-undefined state), where the only sound response is `process::abort`, and the
    /// durable bookkeeping afterwards is best-effort.
    fn abort(&self, txn: TxnId, state: TxnState) {
        let locks = state.locks;
        let undo = state.undo;
        // Capture which sequences need a durable `SeqDrop` compensation before `undo` is consumed.
        let created_sequences: Vec<u64> = undo
            .iter()
            .filter_map(|op| match op {
                UndoOp::CreatedSequence { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        // 1. Physically erase this transaction's versions FIRST. In Stage-1/2 the page store is
        //    volatile (WAL is the only durable medium), so the undo touches only in-memory state and
        //    CANNOT fail on a full disk — it fails only on a poisoned mutex, which is genuinely
        //    unrecoverable. Doing it before any fallible WAL append means a full disk can never
        //    strand the undo half-done, which was the dirty-read hazard the old ordering guarded
        //    against by killing the process.
        if self.rollback_tail(txn, undo, false).is_err() {
            // The only failure mode here is a poisoned mutex: another thread panicked mid-mutation,
            // so engine state is already undefined and the process cannot continue safely.
            // `process::abort` (not `panic!`) — a panic in the server's `spawn_blocking` task is
            // caught by the runtime and would keep serving corrupt state.
            eprintln!(
                "nusadb-btree: FATAL — transaction abort undo failed (poisoned lock); aborting so \
                 recovery rebuilds a clean state on restart"
            );
            std::process::abort();
        }

        // 2. Versions are gone, so the transaction may now safely leave `active` and drop its locks.
        //    Do this BEFORE the advisory WAL append so a failed append (e.g. ENOSPC) can never
        //    strand the transaction in `active` with its locks held forever.
        let Ok(mut t) = self.txns.lock() else {
            // The undo already erased every version, but a poisoned lock blocks removing the
            // transaction from `active` and releasing its locks — it would stay stranded forever
            // (purge pinned, locks held). A poisoned lock is an undefined-state situation; stop so
            // recovery rebuilds clean, leak-free state on restart.
            eprintln!(
                "nusadb-btree: FATAL — txns lock poisoned during abort teardown; aborting so \
                 recovery rebuilds a clean state on restart"
            );
            std::process::abort();
        };
        t.active.remove(&txn.0);
        t.release_locks(txn.0, &locks);

        // 3. Best-effort durable bookkeeping — MUST NOT abort the process on failure, or a full disk
        //    would take the whole server down in an ENOSPC crash-loop (the disk is still full on
        //    restart). The `AbortTxn` marker is purely advisory: recovery already excludes any
        //    transaction without a `CommitTxn`, so a missing marker changes nothing. The `SeqDrop`
        //    compensation neutralizes a rolled-back non-transactional CREATE SEQUENCE; if it cannot
        //    be logged, that sequence may resurrect on recovery — a rare, benign anomaly we accept
        //    over killing every connection. The disk-full error surfaces to the client that hit it;
        //    the server keeps serving every other connection.
        if let Err(e) = self.log(&WalRecord::AbortTxn { txn }) {
            eprintln!("nusadb-btree: WARN — could not log advisory AbortTxn for {txn:?}: {e}");
        }
        for id in created_sequences {
            if let Err(e) = self.log_durable(&LoggedOp::SeqDrop { id }.to_record()) {
                eprintln!(
                    "nusadb-btree: WARN — could not log SeqDrop compensation for sequence {id}: {e}"
                );
            }
        }
    }

    /// Whether a `SERIALIZABLE` transaction has a read-write antidependency that makes its
    /// schedule non-serializable: a row it read has, since its
    /// snapshot, been created or deleted by a **concurrent transaction that has committed** (one
    /// its `BEGIN` snapshot cannot see and that is no longer active). Row-level detection — it
    /// prevents write-skew over existing rows (the Hermitage `G2` anomaly); predicate/phantom
    /// antidependencies over not-yet-existing rows are the further, predicate-level SSI refinement this engine
    /// owns. A no-op for every other isolation level: `REPEATABLE READ` is snapshot isolation and
    /// permits write-skew by design.
    ///
    /// # Errors
    /// Propagates page-store I/O and corruption-class decode failures.
    ///
    /// Runs under the commit gate. The txn domain is only SNAPSHOTTED (read set, pinned view,
    /// active/staged sets) — its lock is NOT held across the tree walk, so other workers keep
    /// running ops while a big read set is checked. The snapshot is sound because [check →
    /// stage] is serialized by the gate: a committer that staged before this check is in the
    /// staged snapshot (conflicting), and one that stages later runs its own check behind this
    /// gate hold and sees THIS transaction instead. A writer that begins or aborts mid-walk can
    /// only push the verdict toward a spurious conservative abort (the caller retries) — never
    /// toward missing a real conflict. The chain walk holds the reclamation gate so no slot it
    /// can reach is recycled under it.
    fn serializable_read_conflict(&self, txn: u64) -> Result<bool> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let (reads, pinned, active, staged) = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            let Some(state) = txns.txns.get(&txn) else {
                return Ok(false);
            };
            if !matches!(state.level, IsolationLevel::Serializable) || state.reads.is_empty() {
                return Ok(false);
            }
            // SSI narrowing: a table whose write version has not moved since this
            // transaction began provably had no concurrent committer (or stager) write a row
            // in it — no stamp in it can conflict, so every read of it skips validation. The
            // read-mostly workload validates nothing; the check degrades gracefully to the
            // full per-row walk only for tables that were actually written concurrently.
            let reads: Vec<(u64, u64)> = state
                .reads
                .iter()
                .copied()
                .filter(|&(table, _)| {
                    // Skip only when the STAGED-instant version equals the FINISHED-instant
                    // version this reader saw at begin: any writer staged since — including
                    // one still mid-fsync, whose rows the reader could not see — breaks the
                    // equality and forces full validation.
                    let staged_now = txns
                        .table_write_versions_staged
                        .get(&table)
                        .copied()
                        .unwrap_or(0);
                    let finished_at_begin = state
                        .write_versions_at_begin
                        .get(&table)
                        .copied()
                        .unwrap_or(0);
                    staged_now != finished_at_begin
                })
                .collect();
            if reads.is_empty() {
                return Ok(false);
            }
            (
                reads,
                state.pinned.clone(),
                txns.active.clone(),
                txns.staged.clone(),
            )
        };
        let undo = self.reclaim.read().map_err(|_| poisoned())?;
        // A stamp is a conflicting write iff a concurrent transaction (unseen by this one's begin
        // snapshot) made it AND has already committed (is no longer active). An unseen-but-active
        // writer is not itself a conflict — first-committer-wins, checked when it commits — but it
        // must NOT hide a committed writer beneath it: the check walks the version chain from the
        // newest version down to the one this transaction's snapshot saw, so an in-flight write
        // stacked on top of a concurrent-committed write is still caught.
        // A stamp conflicts iff its transaction is committed — or STAGED: a staged commit's
        // marker is already appended, ordering it ahead of this transaction's in the log, so it
        // must count as committed here even though its group fsync has not returned yet (if that
        // fsync ultimately fails, this abort was merely conservative).
        let conflicting = |stamp: u64| {
            stamp != txn
                && !pinned.sees(stamp)
                && (!active.contains(&stamp) || staged.contains(&stamp))
        };
        for &(table, row_id) in &reads {
            let Some(t) = cat.tables.get(&table) else {
                continue; // the table was dropped; nothing left to conflict on
            };
            let tree = ClusteredTree::open(&self.store, t.root_id());
            let Some(value) = tree.get(row_id)? else {
                continue;
            };
            let (mut meta, _) = mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
            loop {
                // A concurrent-committed creation or deletion of a version above what we read is a
                // read-write antidependency.
                if conflicting(meta.xmin) || (meta.xmax != mvcc::NO_XMAX && conflicting(meta.xmax))
                {
                    return Ok(true);
                }
                // Reached the version our snapshot can see (the one we read): everything below is
                // what we read or older — stop.
                if pinned.sees(meta.xmin) || meta.undo == mvcc::NO_UNDO {
                    break;
                }
                let Some(Some(prev)) = undo
                    .arena
                    .get(usize::try_from(meta.undo).unwrap_or(usize::MAX))
                else {
                    break; // a purged slot is unreachable by construction; nothing older to check
                };
                meta = prev.meta;
            }
        }
        Ok(false)
    }

    /// The row-ids of `table` whose entry in `index` under exactly `key` is **visible** to
    /// `view` — the FK lookup primitive (parent-existence and dependent-children checks). The
    /// caller holds the catalog read guard; the index read latch and the reclamation gate are
    /// taken here per lookup.
    fn visible_rows_for_index_key(
        &self,
        cat: &Catalog,
        view: &ReadView,
        index: u64,
        table: u64,
        key: &[u8],
    ) -> Result<Vec<u64>> {
        let Some(idx) = cat.indexes.get(&index) else {
            return Ok(Vec::new());
        };
        let Some(t) = cat.tables.get(&table) else {
            return Ok(Vec::new());
        };
        let data = idx.data.read().map_err(|_| poisoned())?;
        let Some(row_ids) = data.entries.get(key) else {
            return Ok(Vec::new());
        };
        let undo = self.reclaim.read().map_err(|_| poisoned())?;
        let tree = ClusteredTree::open(&self.store, t.root_id());
        let mut out = Vec::new();
        for (&row_id, metas) in row_ids {
            // Entry stamps first (does the reader's visible version of this row carry THIS
            // key?), then the base row — the same 2-hop rule as `index_scan`.
            if !IndexData::entry_visible(metas, view) {
                continue;
            }
            let Some(value) = tree.get(row_id)? else {
                continue;
            };
            let (meta, tuple) = mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
            if mvcc::visible_tuple(meta, tuple, &undo.arena, view).is_some() {
                out.push(row_id);
            }
        }
        Ok(out)
    }

    /// Replay `ops` newest-first, undoing each write by restoring the exact previous encoded
    /// leaf entry (version header included) — an aborted transaction leaves no version behind.
    /// The caller holds the catalog guard (`Write` iff any op is DDL-shaped); row and index
    /// inverses take the per-object latch, so each single undo step is atomic against concurrent
    /// same-object writers — and MVCC keeps the whole span consistent for readers, because the
    /// undoing transaction is still in `active` (its versions invisible) until `abort` ends it.
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-undo-op dispatcher; splitting it would scatter the                   rollback semantics"
    )]
    fn undo_ops(&self, cat: &mut CatalogRef<'_>, txn: u64, mut ops: Vec<UndoOp>) -> Result<()> {
        let store = &self.store;
        while let Some(op) = ops.pop() {
            match op {
                UndoOp::Inserted { table, row_id } => {
                    if let Some(t) = cat.get().tables.get(&table) {
                        let _w = t.write.lock().map_err(|_| poisoned())?;
                        let tree = ClusteredTree::open(store, t.root_id());
                        tree.delete(row_id)?;
                        t.set_root(tree.root());
                    }
                },
                UndoOp::Updated {
                    table,
                    row_id,
                    old,
                    undo_idx,
                } => {
                    if let Some(t) = cat.get().tables.get(&table) {
                        let _w = t.write.lock().map_err(|_| poisoned())?;
                        let mut tree = ClusteredTree::open(store, t.root_id());
                        tree.update(row_id, &old)?;
                        t.set_root(tree.root());
                    }
                    // Restoring `old` disconnected the slot this update parked from every
                    // chain. Queue it for purge to free once the abort settles — freeing here
                    // would race a reader mid-walk from the pre-rollback leaf (the leak this
                    // closes was accidentally shielding that walk).
                    self.reclaim
                        .write()
                        .map_err(|_| poisoned())?
                        .orphans
                        .push((undo_idx, txn));
                },
                UndoOp::Deleted { table, row_id, old } => {
                    if let Some(t) = cat.get().tables.get(&table) {
                        let _w = t.write.lock().map_err(|_| poisoned())?;
                        let mut tree = ClusteredTree::open(store, t.root_id());
                        tree.update(row_id, &old)?;
                        t.set_root(tree.root());
                    }
                },
                UndoOp::CreatedTable { table } => {
                    let cat = cat.get_mut()?;
                    if let Some(state) = cat.tables.remove(&table) {
                        let root = state.root_id();
                        cat.by_name
                            .remove(&(state.schema.schema.clone(), state.schema.name));
                        // The tree was never visible to a committed state: free its pages now
                        // (an aborted CREATE TABLE must not leak them). The catalog write guard
                        // excludes every reader (all hold `read`), so no scan can be walking it.
                        let tree = ClusteredTree::open(store, root);
                        for page in tree.pages()? {
                            store.deallocate_page(page)?;
                        }
                    }
                },
                UndoOp::DroppedTable { table, state } => {
                    // The drop is undone: the tree is live again, so un-queue its reclamation.
                    self.dropped
                        .lock()
                        .map_err(|_| poisoned())?
                        .retain(|d| d.root != state.root_id());
                    let cat = cat.get_mut()?;
                    cat.by_name.insert(
                        (state.schema.schema.clone(), state.schema.name.clone()),
                        table,
                    );
                    cat.tables.insert(table, state);
                },
                UndoOp::CreatedIndex { index } => {
                    let cat = cat.get_mut()?;
                    if let Some(state) = cat.indexes.remove(&index) {
                        cat.idx_by_name.remove(&state.def.name);
                    }
                },
                UndoOp::DroppedIndex { index, state } => {
                    let cat = cat.get_mut()?;
                    cat.idx_by_name.insert(state.def.name.clone(), index);
                    cat.indexes.insert(index, state);
                },
                UndoOp::IndexInserted {
                    index,
                    key,
                    row_id,
                    stamped,
                } => {
                    if let Some(idx) = cat.get().indexes.get(&index) {
                        let mut data = idx.data.write().map_err(|_| poisoned())?;
                        data.remove_inserted(&key, row_id, txn);
                        // Revive the previous alive range this insert dead-stamped (the row's
                        // key move is being undone).
                        if let Some(old_key) = stamped {
                            data.apply_unstamp(&old_key, row_id, txn);
                        }
                    }
                },
                UndoOp::IndexDeleted {
                    index,
                    key,
                    row_id,
                    meta,
                } => {
                    if let Some(idx) = cat.get().indexes.get(&index) {
                        let mut data = idx.data.write().map_err(|_| poisoned())?;
                        data.entries
                            .entry(key.clone())
                            .or_default()
                            .entry(row_id)
                            .or_default()
                            .push(meta);
                        // Only an alive range re-earns the reverse-map slot.
                        if meta.xmax == mvcc::NO_XMAX {
                            data.alive.insert(row_id, key);
                        }
                    }
                },
                UndoOp::AddedConstraint { table, name } => {
                    if let Some(list) = cat.get_mut()?.constraints.get_mut(&table) {
                        list.retain(|c| c.name != name);
                    }
                },
                UndoOp::DroppedConstraint { table, state } => {
                    cat.get_mut()?
                        .constraints
                        .entry(table)
                        .or_default()
                        .push(state);
                },
                UndoOp::AddedCheck { table, name } => {
                    if let Some(list) = cat.get_mut()?.checks.get_mut(&table) {
                        list.retain(|c| c.name != name);
                    }
                },
                UndoOp::DroppedCheck { table, state } => {
                    cat.get_mut()?.checks.entry(table).or_default().push(state);
                },
                UndoOp::AddedForeignKey {
                    name,
                    child_table: _,
                } => {
                    cat.get_mut()?.foreign_keys.remove(&name);
                },
                UndoOp::DroppedForeignKey { state } => {
                    cat.get_mut()?
                        .foreign_keys
                        .insert(state.name.clone(), state);
                },
                UndoOp::AnalyzedTable { table, previous } => {
                    let cat = cat.get_mut()?;
                    match previous {
                        Some(prev) => cat.stats.insert(table, *prev),
                        None => cat.stats.remove(&table),
                    };
                },
                UndoOp::CreatedSequence { id, name } => {
                    let mut seqs = self.seqs.lock().map_err(|_| poisoned())?;
                    seqs.sequences.remove(&id);
                    seqs.seq_by_name.remove(&name);
                },
                UndoOp::AlteredSchema {
                    table,
                    previous,
                    previous_version,
                    new_version,
                } => {
                    let cat = cat.get_mut()?;
                    if let Some(t) = cat.tables.get_mut(&table) {
                        let current_name = t.schema.name.clone();
                        let current_schema = t.schema.schema.clone();
                        // Revert a rename in the by-name index.
                        if current_name != previous.name || current_schema != previous.schema {
                            cat.by_name.remove(&(current_schema, current_name));
                            cat.by_name
                                .insert((previous.schema.clone(), previous.name.clone()), table);
                        }
                        if let Some(t) = cat.tables.get_mut(&table) {
                            t.schema_history.remove(&new_version);
                            t.schema = *previous;
                            t.schema_version = previous_version;
                        }
                    }
                },
                UndoOp::CreatedSchema { id, name } => {
                    let cat = cat.get_mut()?;
                    cat.namespaces.remove(&id);
                    cat.ns_by_name.remove(&name);
                },
                UndoOp::DroppedSchema { id, name } => {
                    let cat = cat.get_mut()?;
                    cat.ns_by_name.insert(name.clone(), id);
                    cat.namespaces.insert(id, name);
                },
            }
        }
        Ok(())
    }

    /// Free the arena chain starting at `idx`, returning how many versions were reclaimed. The
    /// caller holds the reclamation gate exclusively.
    fn free_chain(undo: &mut UndoDomain, mut idx: u64) -> usize {
        let mut freed = 0;
        while idx != mvcc::NO_UNDO {
            let Some(slot) = undo
                .arena
                .get_mut(usize::try_from(idx).unwrap_or(usize::MAX))
            else {
                break;
            };
            let Some(version) = slot.take() else {
                break;
            };
            undo.free.push(idx);
            idx = version.meta.undo;
            freed += 1;
        }
        freed
    }

    /// One purge pass: reclaim every version, row, index entry, and dropped-table page
    /// that **no current or future view can reach**.
    ///
    /// A version stamp is *settled* when its transaction has ended and every active
    /// transaction's pinned view sees it — then it is visible to all current views (pinned or
    /// statement-fresh) and to every future one. Purge then:
    ///
    /// - frees the undo chain below a settled newest version (no reader walks past it);
    /// - physically removes a row whose delete (`xmax`) is settled, plus its chain and its
    ///   stale index entries;
    /// - frees the pages of dropped tables whose dropping transaction is settled.
    ///
    /// Purge is **not logged**: it changes no logical content, and recovery replays committed
    /// history into fresh single versions anyway. Structural leaf reclamation (empty/underfull
    /// pages staying in the chain) arrives with page-store persistence; scheduling (background/
    /// incremental cadence) is wired at the composition root — callers invoke this
    /// explicitly for now.
    ///
    /// Latching: the settled-ness snapshot is taken once — settled is monotone (a settled
    /// stamp can never become unsettled: the transaction has ended and every later view sees
    /// it), so acting on the snapshot stays sound while new transactions begin. Each table is
    /// processed under its writer latch **plus the reclamation gate held exclusively**, so no
    /// in-flight scan can chase a freed arena slot or a deallocated page; index entries follow
    /// per index under their write latch (an entry whose row was just removed resolves to
    /// nothing in the interim — the same tolerance `index_scan` always had).
    ///
    /// # Errors
    /// Propagates page-store I/O errors and corruption-class decode failures.
    #[allow(
        clippy::too_many_lines,
        reason = "one linear pass: batched row reclamation, then index-entry sweep, orphan-slot \
                  reclamation, and dropped-tree reclamation — splitting the phases would scatter \
                  the shared `settled` snapshot they all read"
    )]
    pub fn purge(&self) -> Result<PurgeStats> {
        let mut stats = PurgeStats::default();
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        let (pinned, active) = {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            let pinned: Vec<ReadView> = txns.txns.values().map(|t| t.pinned.clone()).collect();
            (pinned, txns.active.clone())
        };
        let settled = |x: u64| !active.contains(&x) && pinned.iter().all(|v| v.sees(x));

        for (&table, t) in &cat.tables {
            let mut removed_rows: HashSet<u64> = HashSet::new();
            // Incremental row reclamation: process the tree in row-id batches, dropping the writer
            // latch and reclamation gate between batches so a concurrent writer interleaves instead
            // of stalling for the whole pass (the range-UPDATE latch-contention fix). `settled` is
            // monotone — a stamp settled at the snapshot above never un-settles — so the snapshot
            // stays valid across releases. Each batch re-opens the tree (an interleaved write may
            // have moved the root) and resumes at the next row-id via `scan_from_with`; a row that a
            // writer changed in the gap is simply seen fresh in a later batch (its new stamp is not
            // settled by this pass's snapshot, so it is left for the next pass).
            let mut cursor = 0u64;
            let mut batch: Vec<(u64, Vec<u8>)> = Vec::with_capacity(PURGE_ROW_BATCH);
            loop {
                batch.clear();
                let mut last_key = None;
                {
                    let _w = t.write.lock().map_err(|_| poisoned())?;
                    let mut undo = self.reclaim.write().map_err(|_| poisoned())?;
                    // The root cannot move within a batch: `delete` never merges/shrinks (underfull
                    // leaves stay chained here) and the `undo=NO_UNDO` rewrite is byte-identical in
                    // size (never splits). Read the batch's rows first, then reclaim under the same
                    // hold, so the in-batch reclamation never disturbs its own scan.
                    let mut tree = ClusteredTree::open(&self.store, t.root_id());
                    tree.scan_from_with(cursor, |row_id, value| {
                        batch.push((row_id, value.to_vec()));
                        last_key = Some(row_id);
                        Ok(batch.len() < PURGE_ROW_BATCH)
                    })?;
                    for (row_id, value) in &batch {
                        let (meta, tuple) =
                            mvcc::decode_row(value).ok_or_else(|| corrupt_row(*row_id))?;
                        if meta.xmax != mvcc::NO_XMAX && settled(meta.xmax) {
                            // Every view sees the delete: the row and its whole history are dead.
                            stats.versions_reclaimed += Self::free_chain(&mut undo, meta.undo);
                            tree.delete(*row_id)?;
                            removed_rows.insert(*row_id);
                            stats.rows_removed += 1;
                        } else if meta.undo != mvcc::NO_UNDO && settled(meta.xmin) {
                            // Every view sees the newest version: nobody walks the chain below it.
                            stats.versions_reclaimed += Self::free_chain(&mut undo, meta.undo);
                            let unchained = mvcc::encode_row(
                                RowMeta {
                                    xmin: meta.xmin,
                                    xmax: meta.xmax,
                                    undo: mvcc::NO_UNDO,
                                },
                                tuple,
                            );
                            tree.update(*row_id, &unchained)?;
                        }
                    }
                } // release the writer latch + reclamation gate — writers interleave here
                // A short batch means `scan_from_with` reached the end of the tree.
                if batch.len() < PURGE_ROW_BATCH {
                    break;
                }
                match last_key {
                    Some(k) => cursor = k.saturating_add(1),
                    None => break,
                }
            }
            for idx in cat.indexes.values().filter(|i| i.def.table.0 == table) {
                let mut data = idx.data.write().map_err(|_| poisoned())?;
                data.entries.retain(|_, rows| {
                    rows.retain(|r, metas| {
                        // A range is reclaimed with its removed row, or once its dead-stamp is
                        // settled (every present and future view sees the supersession — nobody
                        // can resolve this key to that row through it anymore).
                        let before = metas.len();
                        metas.retain(|m| {
                            !(removed_rows.contains(r)
                                || (m.xmax != mvcc::NO_XMAX && settled(m.xmax)))
                        });
                        stats.index_entries_removed += before - metas.len();
                        !metas.is_empty()
                    });
                    !rows.is_empty()
                });
                data.alive.retain(|r, _| !removed_rows.contains(r));
            }
        }

        // Orphaned arena slots (aborted UPDATEs disconnected their parked versions): freed once
        // the aborting transaction is settled — no view concurrent with it remains, so no reader
        // can still be walking a chain through the pre-rollback leaf into the slot.
        {
            let mut undo = self.reclaim.write().map_err(|_| poisoned())?;
            let orphans = std::mem::take(&mut undo.orphans);
            for (slot, txn) in orphans {
                if settled(txn) {
                    if let Some(entry) = undo
                        .arena
                        .get_mut(usize::try_from(slot).unwrap_or(usize::MAX))
                        && entry.take().is_some()
                    {
                        undo.free.push(slot);
                        stats.versions_reclaimed += 1;
                    }
                } else {
                    undo.orphans.push((slot, txn));
                }
            }
        }

        // Dropped trees: processed in place under the dropped-queue lock (so a concurrent
        // rollback un-queueing its table serializes with this pass) and the reclamation gate
        // (so no in-flight scan of a just-dropped table can touch a deallocated page).
        {
            let mut dropped = self.dropped.lock().map_err(|_| poisoned())?;
            let _gate = self.reclaim.write().map_err(|_| poisoned())?;
            let mut keep: Vec<DroppedPages> = Vec::with_capacity(dropped.len());
            for entry in dropped.drain(..) {
                if settled(entry.txn) {
                    let tree = ClusteredTree::open(&self.store, entry.root);
                    for page in tree.pages()? {
                        self.store.deallocate_page(page)?;
                        stats.pages_reclaimed += 1;
                    }
                    stats.tables_reclaimed += 1;
                } else {
                    keep.push(entry);
                }
            }
            *dropped = keep;
        }
        Ok(stats)
    }

    /// Pages currently allocated in the backing store — observability for purge verification
    /// and ops counters.
    ///
    /// # Errors
    /// Fails only on a poisoned store lock.
    pub fn live_pages(&self) -> Result<usize> {
        self.store.live_pages()
    }

    /// Every stored row version's MVCC stamps — [`VersionMetadata`], the newest version of each
    /// row plus every parked version on its undo chain. The DST prefix-replay oracle's
    /// observability hook: recovery must never mis-stamp version metadata.
    ///
    /// # Errors
    /// Fails on a poisoned latch, page-store I/O, or a corrupt row.
    pub fn version_metadata(&self) -> Result<VersionMetadata> {
        let cat = self.catalog.read().map_err(|_| poisoned())?;
        // The reclamation gate pins every arena slot this walk can reach (the DST oracle runs
        // this on quiesced engines, but the guard keeps it honest under concurrency too).
        let undo = self.reclaim.read().map_err(|_| poisoned())?;
        let mut out = Vec::new();
        for (&table_id, t) in &cat.tables {
            let tree = ClusteredTree::open(&self.store, t.root_id());
            for (row_id, value) in tree.scan()? {
                let (mut meta, _) = mvcc::decode_row(&value).ok_or_else(|| corrupt_row(row_id))?;
                loop {
                    let xmax = (meta.xmax != mvcc::NO_XMAX).then_some(TxnId(meta.xmax));
                    out.push((TableId(table_id), tid_of(row_id), TxnId(meta.xmin), xmax));
                    if meta.undo == mvcc::NO_UNDO {
                        break;
                    }
                    let Some(Some(prev)) = undo
                        .arena
                        .get(usize::try_from(meta.undo).unwrap_or(usize::MAX))
                    else {
                        break;
                    };
                    meta = prev.meta;
                }
            }
        }
        Ok(out)
    }
}

/// One row version's stamps as reported by [`BtreeEngine::version_metadata`]:
/// `(table, tid, xmin, xmax)` — `xmax` is `None` while the version is live.
pub type VersionMetadata = Vec<(TableId, Tid, TxnId, Option<TxnId>)>;

/// What one [`BtreeEngine::purge`] pass reclaimed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PurgeStats {
    /// Undo-arena versions freed (slots recycled through the free list).
    pub versions_reclaimed: usize,
    /// Rows physically removed from their leaves (settled deletes).
    pub rows_removed: usize,
    /// Stale index entries dropped alongside removed rows.
    pub index_entries_removed: usize,
    /// Dropped tables whose trees were reclaimed.
    pub tables_reclaimed: usize,
    /// Pages returned to the store's free list from reclaimed trees.
    pub pages_reclaimed: usize,
}
