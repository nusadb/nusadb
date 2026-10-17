//! Deferrable constraints: `[NOT] DEFERRABLE [INITIALLY DEFERRED | IMMEDIATE]`, `SET CONSTRAINTS`
//! and the checks that run when a transaction ends.
//!
//! # What is deferred
//!
//! A `PRIMARY KEY`, `UNIQUE`, `FOREIGN KEY` or `EXCLUDE` constraint may be declared
//! `DEFERRABLE`. While such a constraint is in deferred mode (`INITIALLY DEFERRED`, or after
//! `SET CONSTRAINTS ... DEFERRED`), a statement that writes does not check it. Instead it records
//! what it wrote in this transaction's queue (`Entry`) and the queue is checked when the
//! transaction commits ([`commit`]), when it is prepared ([`prepare`]), and when
//! `SET CONSTRAINTS ... IMMEDIATE` switches the constraint back. A failed check rolls the
//! transaction back and reports the constraint's usual error: `23505`, `23503` or `23P01`.
//!
//! For a foreign key only the `NO ACTION` check moves: `RESTRICT` and the `CASCADE` / `SET NULL`
//! actions of a deferrable key still run when the parent row is deleted or updated.
//!
//! # The queue
//!
//! An entry names a constraint (`ConstraintRef`) and a key written under it. Its check is
//! idempotent and reads the state the transaction is about to commit: for a key of a unique
//! constraint, at most one live row may hold it; for a foreign key, a live child holding the key
//! needs a live parent; for an exclusion constraint, no two live rows of the key's `=` group may
//! conflict. So an entry that a later statement made stale (the row was deleted again, the key
//! changed back) can never raise a false error, and an entry for a constraint or table dropped
//! since is skipped. Past `WHOLE_CONSTRAINT_AFTER` keys a constraint stops recording keys and is
//! checked over the whole table instead.
//!
//! The queue lives in a process-wide registry keyed by the engine and the transaction, not in a
//! thread-local: the wire server runs consecutive statements of one transaction on different pool
//! threads. `ROLLBACK TO SAVEPOINT` trims the entries recorded since the savepoint and restores the
//! `SET CONSTRAINTS` mode it had ([`rollback_to`]). Every commit, rollback and savepoint operation
//! of the SQL layer and the wire server goes through the wrappers here; a test reads their sources
//! to keep it that way.
//!
//! # Concurrency
//!
//! A deferred write takes the same key locks an immediate one does, at write time, and the locks
//! are held until the transaction ends: an exclusive key lock for a unique or exclusion key, a
//! shared lock on the referenced parent key for a foreign key. The commit-time check runs under
//! those locks and reads the latest committed state plus the transaction's own writes. Any other
//! transaction writing a conflicting key either took its lock first, so this one's write already
//! failed with `40001`, or tries after, and fails itself; one that finished before has committed
//! and is visible to the check. A `40001` may therefore also surface at `COMMIT`.
//!
//! # Catalog
//!
//! Which constraints are deferrable lives in an engine-scoped SQL-layer catalog,
//! `nusadb_constraint_deferral`, one row per deferrable constraint: `(table id, kind, constraint,
//! origin, deferrable, initially deferred)`. `origin` is the constraint a partition's copy was made
//! from, so `SET CONSTRAINTS parent_key` reaches the copies too. Every creation of a constraint
//! writes or clears its row, so a dropped constraint's leftover row can never make a later one of
//! the same name deferrable.
#![allow(clippy::wildcard_imports)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use nusadb_core::{TableId, Tid};

use super::*;

/// Engine-scoped catalog of deferrable constraints.
pub(super) const DEFERRAL_CATALOG: &str = "nusadb_constraint_deferral";

/// `(table_id, kind, constraint, origin, deferrable, initially_deferred)`, all text.
const CATALOG_SCHEMA: [ColumnType; 6] = [ColumnType::Text; 6];

/// A constraint stops recording its keys past this many, and is checked over its whole table.
pub(super) const WHOLE_CONSTRAINT_AFTER: usize = 100_000;

/// The kind of a deferrable constraint, as the catalog stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Kind {
    /// `PRIMARY KEY` or `UNIQUE`.
    Unique,
    /// `FOREIGN KEY`, recorded on the referencing table.
    ForeignKey,
    /// `EXCLUDE`.
    Exclusion,
}

impl Kind {
    const fn code(self) -> &'static str {
        match self {
            Self::Unique => "u",
            Self::ForeignKey => "f",
            Self::Exclusion => "x",
        }
    }

    fn from_code(code: &str) -> Option<Self> {
        match code {
            "u" => Some(Self::Unique),
            "f" => Some(Self::ForeignKey),
            "x" => Some(Self::Exclusion),
            _ => None,
        }
    }
}

/// One constraint whose check a transaction can defer: its kind, the table it is declared on (the
/// referencing table, for a foreign key), and its name. Further kinds of deferred check (a unique
/// key enforced across the partitions of a table, say) take a variant of [`Kind`] and of
/// [`Pending`] of their own.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ConstraintRef {
    pub(super) kind: Kind,
    pub(super) table: TableId,
    pub(super) name: String,
}

impl ConstraintRef {
    pub(super) fn unique(table: TableId, name: &str) -> Self {
        Self {
            kind: Kind::Unique,
            table,
            name: name.to_owned(),
        }
    }

    pub(super) fn foreign_key(child: TableId, name: &str) -> Self {
        Self {
            kind: Kind::ForeignKey,
            table: child,
            name: name.to_owned(),
        }
    }

    pub(super) fn exclusion(table: TableId, name: &str) -> Self {
        Self {
            kind: Kind::Exclusion,
            table,
            name: name.to_owned(),
        }
    }
}

/// What one queue entry asks the commit-time check to verify.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Pending {
    /// A key a statement wrote: a unique or exclusion key, or the referencing key of a child row.
    Key(Vec<ast::Value>),
    /// A referenced key a parent `DELETE` / `UPDATE` removed while children still held it.
    RemovedParentKey(Vec<ast::Value>),
    /// Too many keys to keep: check the constraint over its whole table.
    Whole,
}

/// One deferred check.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Entry {
    pub(super) constraint: ConstraintRef,
    pub(super) pending: Pending,
}

/// The `SET CONSTRAINTS` state of a transaction: `ALL`, then per constraint (a later per-name
/// setting overrides `ALL`; a later `ALL` clears the per-name ones).
#[derive(Debug, Clone, Default)]
struct Mode {
    all: Option<bool>,
    named: HashMap<ConstraintRef, bool>,
}

impl Mode {
    /// Whether `constraint`, declared with `declared`, is deferred now.
    fn defers(&self, constraint: &ConstraintRef, declared: ast::Deferral) -> bool {
        if !declared.deferrable {
            return false;
        }
        self.named
            .get(constraint)
            .copied()
            .or(self.all)
            .unwrap_or(declared.initially_deferred)
    }
}

/// A savepoint's view of the queue: how many entries it had and the mode then.
#[derive(Debug)]
struct Mark {
    name: String,
    len: usize,
    mode: Mode,
}

/// One transaction's deferred state.
#[derive(Debug, Default)]
struct TxnDeferred {
    entries: Vec<Entry>,
    /// Keys recorded per constraint, for the whole-table fallback.
    counts: HashMap<ConstraintRef, usize>,
    /// Constraints checked over the whole table.
    whole: HashSet<ConstraintRef>,
    marks: Vec<Mark>,
    mode: Mode,
}

impl TxnDeferred {
    fn push(&mut self, constraint: ConstraintRef, pending: Pending) {
        if self.whole.contains(&constraint) {
            return;
        }
        let count = self.counts.entry(constraint.clone()).or_insert(0);
        *count += 1;
        if *count > WHOLE_CONSTRAINT_AFTER {
            self.collapse(&constraint);
            return;
        }
        self.entries.push(Entry {
            constraint,
            pending,
        });
    }

    /// Replace every key queued for `constraint` by one whole-table check, in the place of the
    /// first of them: a `ROLLBACK TO` a savepoint taken after that first key keeps the check (some
    /// of its keys are still written), one taken before it drops the check with every key.
    fn collapse(&mut self, constraint: &ConstraintRef) {
        let first = self
            .entries
            .iter()
            .position(|e| e.constraint == *constraint)
            .unwrap_or(self.entries.len());
        // Where each old boundary lands: the entries kept before it, plus the whole-table entry
        // once the boundary is past the first key.
        let mut boundary = Vec::with_capacity(self.entries.len() + 1);
        let mut kept = 0;
        for (i, e) in self.entries.iter().enumerate() {
            boundary.push(kept + usize::from(i > first));
            if e.constraint != *constraint {
                kept += 1;
            }
        }
        boundary.push(kept + usize::from(self.entries.len() > first));
        for mark in &mut self.marks {
            mark.len = boundary.get(mark.len).copied().unwrap_or(mark.len);
        }
        let old = std::mem::take(&mut self.entries);
        let (before, after) = old.split_at(first.min(old.len()));
        self.entries.extend_from_slice(before);
        self.entries.push(Entry {
            constraint: constraint.clone(),
            pending: Pending::Whole,
        });
        self.entries.extend(
            after
                .iter()
                .filter(|e| e.constraint != *constraint)
                .cloned(),
        );
        self.recount();
    }

    /// Rebuild the per-constraint counts and the whole-table set from the entries.
    fn recount(&mut self) {
        self.counts.clear();
        self.whole.clear();
        for e in &self.entries {
            if matches!(e.pending, Pending::Whole) {
                self.whole.insert(e.constraint.clone());
            } else {
                *self.counts.entry(e.constraint.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Keep only the entries `keep` accepts, moving each savepoint's length with them.
    fn retain(&mut self, keep: impl Fn(&Entry) -> bool) {
        let mut kept_before: Vec<usize> = Vec::with_capacity(self.entries.len() + 1);
        let mut kept = 0;
        for e in &self.entries {
            kept_before.push(kept);
            if keep(e) {
                kept += 1;
            }
        }
        kept_before.push(kept);
        for mark in &mut self.marks {
            mark.len = kept_before.get(mark.len).copied().unwrap_or(kept);
        }
        self.entries.retain(keep);
        self.recount();
    }
}

/// The registry key: the engine's address and the transaction.
type Key = (usize, TxnId);

static REGISTRY: LazyLock<Mutex<HashMap<Key, TxnDeferred>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// How many transactions have registry state, so a transaction that never deferred anything never
/// takes the registry lock.
static LIVE: AtomicUsize = AtomicUsize::new(0);

fn registry() -> MutexGuard<'static, HashMap<Key, TxnDeferred>> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

fn key(engine: &dyn StorageEngine, txn: TxnId) -> Key {
    (std::ptr::from_ref(engine).cast::<()>().addr(), txn)
}

fn has_state() -> bool {
    LIVE.load(Ordering::Acquire) > 0
}

/// The registry size from which creating a transaction's state first sweeps out finished ones.
const SWEEP_AT: usize = 64;

/// Run `f` on `txn`'s state, creating it.
fn with_state<T>(
    engine: &dyn StorageEngine,
    txn: TxnId,
    f: impl FnOnce(&mut TxnDeferred) -> T,
) -> T {
    let k = key(engine, txn);
    let sweep = {
        let reg = registry();
        !reg.contains_key(&k) && reg.len() >= SWEEP_AT
    };
    if sweep {
        sweep_finished(engine, txn);
    }
    let mut reg = registry();
    if !reg.contains_key(&k) {
        LIVE.fetch_add(1, Ordering::AcqRel);
    }
    f(reg.entry(k).or_default())
}

/// Drop the state of transactions of `engine` that ended without going through [`commit`] or
/// [`rollback`]. Every end of a transaction forgets its own state through those wrappers (a test
/// holds every call site to that), so this is defence in depth against a leak, never part of
/// correctness.
///
/// Only an engine that reports its transactions can say which ended: `live` is a transaction
/// known to be open, and unless the engine reports it (`txn_isolation` is `Some`), the engine does
/// not track transactions and nothing is swept. A transaction is dropped only when such an engine
/// no longer reports it. The engine is asked outside the registry lock.
fn sweep_finished(engine: &dyn StorageEngine, live: TxnId) {
    if engine.txn_isolation(live).is_none() {
        return;
    }
    let (addr, _) = key(engine, live);
    #[allow(
        clippy::needless_collect,
        reason = "collected so the registry lock is released before the engine is asked"
    )]
    let candidates: Vec<TxnId> = registry()
        .keys()
        .filter(|(a, t)| *a == addr && *t != live)
        .map(|(_, t)| *t)
        .collect();
    let finished: Vec<TxnId> = candidates
        .into_iter()
        .filter(|t| engine.txn_isolation(*t).is_none())
        .collect();
    if finished.is_empty() {
        return;
    }
    let mut reg = registry();
    let mut removed = 0;
    for t in finished {
        if reg.remove(&(addr, t)).is_some() {
            removed += 1;
        }
    }
    drop(reg);
    LIVE.fetch_sub(removed, Ordering::AcqRel);
}

/// Take `txn`'s state out of the registry.
fn take(engine: &dyn StorageEngine, txn: TxnId) -> Option<TxnDeferred> {
    if !has_state() {
        return None;
    }
    let taken = registry().remove(&key(engine, txn));
    if taken.is_some() {
        LIVE.fetch_sub(1, Ordering::AcqRel);
    }
    taken
}

/// Forget `txn`'s deferred state (its transaction ended).
pub(super) fn forget(engine: &dyn StorageEngine, txn: TxnId) {
    let _ = take(engine, txn);
}

// === Transaction boundaries ==============================================================

/// Commit `txn`: run its deferred checks, then commit it. A failed check rolls the transaction
/// back and returns the violation. Every commit of a transaction that ran SQL goes through here.
///
/// # Errors
/// The first deferred violation (`23505` / `23503` / `23P01`), a serialization conflict met while
/// checking, or the engine's commit failure.
pub fn commit(engine: &dyn StorageEngine, txn: TxnId) -> Result<(), Error> {
    if let Some(state) = take(engine, txn)
        && !state.entries.is_empty()
        && let Err(e) = check_entries(engine, txn, &state.entries)
    {
        let _ = engine.rollback(txn);
        return Err(e);
    }
    engine.commit(txn).map_err(Into::into)
}

/// Roll `txn` back, forgetting its deferred state. Every rollback of a transaction that ran SQL
/// goes through here.
///
/// # Errors
/// The engine's rollback failure.
pub fn rollback(engine: &dyn StorageEngine, txn: TxnId) -> nusadb_core::Result<()> {
    forget(engine, txn);
    engine.rollback(txn)
}

/// Run `txn`'s deferred checks before `PREPARE TRANSACTION`.
///
/// The queue is empty afterwards, so the prepared transaction (and a restart while it waits) owes
/// no further check. On a violation the transaction is rolled back.
///
/// # Errors
/// As [`commit`].
pub fn prepare(engine: &dyn StorageEngine, txn: TxnId) -> Result<(), Error> {
    if let Some(state) = take(engine, txn)
        && !state.entries.is_empty()
        && let Err(e) = check_entries(engine, txn, &state.entries)
    {
        let _ = engine.rollback(txn);
        return Err(e);
    }
    Ok(())
}

/// `SAVEPOINT name` in `txn`, remembering the queue length and mode for [`rollback_to`].
///
/// # Errors
/// The engine's refusal.
pub fn savepoint(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> nusadb_core::Result<()> {
    engine.savepoint(txn, name)?;
    if has_state()
        && let Some(state) = registry().get_mut(&key(engine, txn))
    {
        state.marks.push(Mark {
            name: name.to_owned(),
            len: state.entries.len(),
            mode: state.mode.clone(),
        });
    }
    Ok(())
}

/// `ROLLBACK TO SAVEPOINT name` in `txn`.
///
/// The entries recorded since are dropped and the mode returns to what it was. A savepoint taken
/// before the transaction deferred anything has no mark, and rolling back to it clears the whole
/// state.
///
/// # Errors
/// The engine's refusal (an unknown savepoint).
pub fn rollback_to(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> nusadb_core::Result<()> {
    engine.rollback_to(txn, name)?;
    // The rollback may have undone this transaction's own catalog writes: forget its cached read.
    forget_cached_read(engine, txn);
    if !has_state() {
        return Ok(());
    }
    let k = key(engine, txn);
    let mut reg = registry();
    let Some(state) = reg.get_mut(&k) else {
        return Ok(());
    };
    if let Some(at) = state.marks.iter().rposition(|m| m.name == name) {
        state.marks.truncate(at + 1);
        if let Some(mark) = state.marks.get(at) {
            let (len, mode) = (mark.len, mark.mode.clone());
            state.entries.truncate(len);
            state.mode = mode;
        }
        state.recount();
    } else {
        reg.remove(&k);
        drop(reg);
        LIVE.fetch_sub(1, Ordering::AcqRel);
    }
    Ok(())
}

/// `RELEASE SAVEPOINT name` in `txn`: the savepoint and every later one are forgotten; the
/// entries stay.
///
/// # Errors
/// The engine's refusal (an unknown savepoint).
pub fn release_savepoint(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> nusadb_core::Result<()> {
    engine.release_savepoint(txn, name)?;
    if has_state()
        && let Some(state) = registry().get_mut(&key(engine, txn))
    {
        match state.marks.iter().rposition(|m| m.name == name) {
            Some(at) => state.marks.truncate(at),
            None => state.marks.clear(),
        }
    }
    Ok(())
}

/// The number of entries in `txn`'s queue, or `None` when it has no state (for tests).
#[cfg(test)]
fn queued(engine: &dyn StorageEngine, txn: TxnId) -> Option<usize> {
    registry().get(&key(engine, txn)).map(|s| s.entries.len())
}

// === Recording ===========================================================================

/// Record `keys` under `constraint` in `txn`'s queue.
pub(super) fn defer_keys(
    engine: &dyn StorageEngine,
    txn: TxnId,
    constraint: &ConstraintRef,
    keys: impl IntoIterator<Item = Pending>,
) {
    let mut keys = keys.into_iter().peekable();
    if keys.peek().is_none() {
        return;
    }
    with_state(engine, txn, |state| {
        for pending in keys {
            state.push(constraint.clone(), pending);
        }
    });
}

/// The deferral state of one table's constraints for one transaction: which are deferrable, and
/// which are deferred right now.
#[derive(Debug, Default)]
pub(super) struct TableDeferral {
    constraints: Vec<(ConstraintRef, ast::Deferral, bool)>,
}

impl TableDeferral {
    /// Whether `constraint` is deferred right now.
    pub(super) fn defers(&self, constraint: &ConstraintRef) -> bool {
        self.constraints
            .iter()
            .any(|(c, _, deferred)| c == constraint && *deferred)
    }

    /// Whether `constraint` is declared `DEFERRABLE`.
    pub(super) fn deferrable(&self, constraint: &ConstraintRef) -> bool {
        self.constraints
            .iter()
            .any(|(c, d, _)| c == constraint && d.deferrable)
    }

    /// Whether the table has no deferrable constraint at all.
    pub(super) const fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

/// The deferral state of `table`'s constraints in `txn`.
///
/// # Errors
/// Storage errors reading the catalog.
pub(super) fn table_state(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
) -> Result<TableDeferral, Error> {
    let rows = table_rows(engine, txn, table)?;
    if rows.is_empty() {
        return Ok(TableDeferral::default());
    }
    let mode = if has_state() {
        registry()
            .get(&key(engine, txn))
            .map(|s| s.mode.clone())
            .unwrap_or_default()
    } else {
        Mode::default()
    };
    Ok(TableDeferral {
        constraints: rows
            .iter()
            .map(|r| {
                let c = r.constraint_ref();
                let deferred = mode.defers(&c, r.deferral);
                (c, r.deferral, deferred)
            })
            .collect(),
    })
}

// === Catalog =============================================================================

/// One row of [`DEFERRAL_CATALOG`].
#[derive(Debug, Clone)]
pub(super) struct CatalogRow {
    tid: Tid,
    pub(super) table: TableId,
    pub(super) kind: Kind,
    pub(super) name: String,
    pub(super) origin: String,
    pub(super) deferral: ast::Deferral,
}

impl CatalogRow {
    pub(super) fn constraint_ref(&self) -> ConstraintRef {
        ConstraintRef {
            kind: self.kind,
            table: self.table,
            name: self.name.clone(),
        }
    }

    fn decode(tid: Tid, bytes: &[u8]) -> Result<Option<Self>, Error> {
        let row = row::decode(bytes, &CATALOG_SCHEMA)?;
        let text = |i: usize| match row.get(i) {
            Some(ast::Value::Text(s)) => Some(s.as_str()),
            _ => None,
        };
        let (Some(table), Some(kind), Some(name), Some(origin), Some(deferrable), Some(initially)) =
            (text(0), text(1), text(2), text(3), text(4), text(5))
        else {
            return Ok(None);
        };
        let (Ok(table), Some(kind)) = (table.parse::<u64>(), Kind::from_code(kind)) else {
            return Ok(None);
        };
        Ok(Some(Self {
            tid,
            table: TableId(table),
            kind,
            name: name.to_owned(),
            origin: origin.to_owned(),
            deferral: ast::Deferral {
                deferrable: deferrable == "t",
                initially_deferred: initially == "t",
            },
        }))
    }
}

/// The index over the catalog's `table_id` column: a write reads only its own table's rows, so
/// what it costs does not grow with the number of deferrable constraints elsewhere.
const TABLE_INDEX: &str = "nusadb_constraint_deferral_table";

/// Bumped on every catalog write, so a cached read is never reused across one.
static CATALOG_EPOCH: AtomicU64 = AtomicU64::new(0);

/// The catalog rows of the tables one statement has asked about, for one engine, transaction and
/// catalog epoch. Valid within one statement only ([`new_statement`] drops it): the transaction's
/// snapshot does not move inside a statement, so a commit by another transaction cannot make a
/// cached read stale, while across statements a `READ COMMITTED` snapshot moves and the rows are
/// read again (a committed `ALTER CONSTRAINT` is seen by the next statement).
struct CachedReads {
    engine: usize,
    txn: TxnId,
    epoch: u64,
    tables: HashMap<TableId, std::rc::Rc<Vec<CatalogRow>>>,
}

thread_local! {
    static CACHE: RefCell<Option<CachedReads>> = const { RefCell::new(None) };
}

/// A statement (or a statement-like step: a commit-time check, a statement snapshot refresh) starts
/// on this thread: no catalog read cached before it is used again.
pub fn new_statement() {
    CACHE.with(|c| c.borrow_mut().take());
}

/// Drop this thread's cached reads when they belong to `txn`.
fn forget_cached_read(engine: &dyn StorageEngine, txn: TxnId) {
    let (addr, _) = key(engine, txn);
    CACHE.with(|c| {
        let mut cached = c.borrow_mut();
        if cached
            .as_ref()
            .is_some_and(|r| r.engine == addr && r.txn == txn)
        {
            cached.take();
        }
    });
}

/// The catalog rows of `table` that `txn` sees, read through the `table_id` index (one probe) and
/// cached for the rest of the statement.
fn table_rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
) -> Result<std::rc::Rc<Vec<CatalogRow>>, Error> {
    let (addr, _) = key(engine, txn);
    let epoch = CATALOG_EPOCH.load(Ordering::Acquire);
    let hit = CACHE.with(|c| {
        c.borrow().as_ref().and_then(|cached| {
            (cached.engine == addr && cached.txn == txn && cached.epoch == epoch)
                .then(|| cached.tables.get(&table).map(std::rc::Rc::clone))
                .flatten()
        })
    });
    if let Some(rows) = hit {
        return Ok(rows);
    }
    let rows = std::rc::Rc::new(read_table(engine, txn, table)?);
    CACHE.with(|c| {
        let mut cached = c.borrow_mut();
        let fresh = cached
            .as_ref()
            .is_none_or(|r| r.engine != addr || r.txn != txn || r.epoch != epoch);
        if fresh {
            *cached = Some(CachedReads {
                engine: addr,
                txn,
                epoch,
                tables: HashMap::new(),
            });
        }
        if let Some(cached) = cached.as_mut() {
            cached.tables.insert(table, std::rc::Rc::clone(&rows));
        }
    });
    Ok(rows)
}

/// The catalog rows of `table`, uncached: an index probe, or a read of the whole (small) catalog
/// on an engine without secondary indexes.
fn read_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
) -> Result<Vec<CatalogRow>, Error> {
    let Some(catalog) = engine.lookup_table_as_of(txn, DEFERRAL_CATALOG)? else {
        return Ok(Vec::new());
    };
    // The index is created with the catalog, in the same transaction, so a visible catalog has
    // a visible index (or none, on an engine without secondary indexes).
    let mut scan = match engine.lookup_index(TABLE_INDEX)? {
        Some(index) => {
            let key = table_key(table)?;
            engine.index_scan(
                txn,
                index,
                std::ops::Bound::Included(key.clone()),
                std::ops::Bound::Included(key),
            )?
        },
        None => engine.scan(txn, catalog.id)?,
    };
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        if let Some(row) = CatalogRow::decode(tid, &bytes)?
            && row.table == table
        {
            out.push(row);
        }
    }
    Ok(out)
}

/// The `table_id` index key of `table`.
fn table_key(table: TableId) -> Result<Vec<u8>, Error> {
    index_key::encode_index_key(&[ast::Value::Text(table.0.to_string())])
}

/// Every catalog row `txn` sees (introspection and the rare whole-catalog statements).
fn catalog_rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<std::rc::Rc<Vec<CatalogRow>>, Error> {
    Ok(std::rc::Rc::new(read_catalog(engine, txn)?))
}

fn read_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<CatalogRow>, Error> {
    let Some(catalog) = engine.lookup_table_as_of(txn, DEFERRAL_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut scan = engine.scan(txn, catalog.id)?;
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        if let Some(row) = CatalogRow::decode(tid, &bytes)? {
            out.push(row);
        }
    }
    Ok(out)
}

/// Every deferrable constraint `txn` sees.
///
/// # Errors
/// Storage errors.
pub(super) fn all_rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<std::rc::Rc<Vec<CatalogRow>>, Error> {
    catalog_rows(engine, txn)
}

fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<TableId, Error> {
    if let Some(found) = engine.lookup_table_as_of(txn, DEFERRAL_CATALOG)? {
        return Ok(found.id);
    }
    let columns = [
        "table_id",
        "kind",
        "constraint_name",
        "origin_name",
        "deferrable",
        "initially_deferred",
    ]
    .into_iter()
    .map(|name| ColumnDef {
        name: name.to_owned(),
        ty: ColumnType::Text,
        nullable: false,
    })
    .collect();
    let def = TableDef {
        schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
        name: DEFERRAL_CATALOG.to_owned(),
        columns,
    };
    let id = engine.create_table(txn, &def)?;
    let index = nusadb_core::engine::IndexDef {
        name: TABLE_INDEX.to_owned(),
        table: id,
        columns: vec!["table_id".to_owned()],
        key_exprs: Vec::new(),
        predicate: None,
        include: Vec::new(),
        kind: nusadb_core::engine::IndexKind::BTree,
        unique: false,
    };
    match engine.create_index(txn, &index) {
        // An engine without secondary indexes (the in-memory test double) is read by a scan.
        Ok(_) => Ok(id),
        Err(nusadb_core::Error::Io(e)) if e.kind() == std::io::ErrorKind::Unsupported => Ok(id),
        Err(e) => Err(e.into()),
    }
}

/// Delete the catalog rows `doomed` picks, among those of `table` (`None`: of every table).
fn delete_rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: Option<TableId>,
    doomed: impl Fn(&CatalogRow) -> bool,
) -> Result<(), Error> {
    let Some(catalog) = engine.lookup_table_as_of(txn, DEFERRAL_CATALOG)? else {
        return Ok(());
    };
    let rows = match table {
        Some(table) => read_table(engine, txn, table)?,
        None => read_catalog(engine, txn)?,
    };
    let victims: Vec<Tid> = rows
        .into_iter()
        .filter(|r| doomed(r))
        .map(|r| r.tid)
        .collect();
    if victims.is_empty() {
        return Ok(());
    }
    CATALOG_EPOCH.fetch_add(1, Ordering::AcqRel);
    for tid in victims {
        engine.delete(txn, catalog.id, tid)?;
    }
    Ok(())
}

/// Record when the constraint `name` of `kind` on `table` is checked, at its creation or by
/// `ALTER CONSTRAINT`: a deferrable constraint gets a row, any other has none. `origin` is the
/// constraint a partition's copy was made from (`None` for a constraint declared directly).
///
/// # Errors
/// Storage errors.
pub(super) fn declare(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: Kind,
    table: TableId,
    name: &str,
    origin: Option<&str>,
    deferral: ast::Deferral,
) -> Result<(), Error> {
    delete_rows(engine, txn, Some(table), |r| r.name == name)?;
    if !deferral.deferrable {
        return Ok(());
    }
    let catalog = ensure_catalog(engine, txn)?;
    CATALOG_EPOCH.fetch_add(1, Ordering::AcqRel);
    let flag = |b: bool| if b { "t" } else { "f" };
    let row = [
        table.0.to_string(),
        kind.code().to_owned(),
        name.to_owned(),
        origin.unwrap_or(name).to_owned(),
        flag(true).to_owned(),
        flag(deferral.initially_deferred).to_owned(),
    ]
    .map(ast::Value::Text);
    let tid = engine.insert(txn, catalog, &row::encode(&row, &CATALOG_SCHEMA)?)?;
    if let Some(index) = engine.lookup_index(TABLE_INDEX)? {
        engine.index_insert(txn, index, &table_key(table)?, tid)?;
    }
    Ok(())
}

/// The declared timing of the constraint `name` on `table`, and the constraint it was copied
/// from: `NOT DEFERRABLE` and itself when it has no row.
///
/// # Errors
/// Storage errors.
pub(super) fn declared(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
    name: &str,
) -> Result<(ast::Deferral, String), Error> {
    Ok(table_rows(engine, txn, table)?
        .iter()
        .find(|r| r.name == name)
        .map_or_else(
            || (ast::Deferral::IMMEDIATE, name.to_owned()),
            |r| (r.deferral, r.origin.clone()),
        ))
}

/// The drop of a constraint: forget its row.
///
/// # Errors
/// Storage errors.
pub(super) fn forget_constraint(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
    name: &str,
) -> Result<(), Error> {
    delete_rows(engine, txn, Some(table), |r| r.name == name)
}

/// The drop of a table: forget the rows of its constraints.
///
/// # Errors
/// Storage errors.
pub(super) fn forget_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: TableId,
) -> Result<(), Error> {
    delete_rows(engine, txn, Some(table), |_| true)
}

/// A table rebuilt under a new id (`TRUNCATE` of a table other tables reference): its rows follow
/// it.
///
/// # Errors
/// Storage errors.
pub(super) fn retarget_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    from: TableId,
    to: TableId,
) -> Result<(), Error> {
    let moved: Vec<CatalogRow> = table_rows(engine, txn, from)?.as_ref().clone();
    if moved.is_empty() {
        return Ok(());
    }
    forget_table(engine, txn, from)?;
    for r in moved {
        declare(
            engine,
            txn,
            r.kind,
            to,
            &r.name,
            Some(&r.origin),
            r.deferral,
        )?;
    }
    Ok(())
}

/// Forget the rows of tables that no longer exist (the teardown of a schema).
///
/// # Errors
/// Storage errors.
pub(super) fn prune_missing_tables(engine: &dyn StorageEngine, txn: TxnId) -> Result<(), Error> {
    let mut gone = HashSet::new();
    for r in catalog_rows(engine, txn)?.iter() {
        if engine.lookup_table_by_id_as_of(txn, r.table)?.is_none() {
            gone.insert(r.table);
        }
    }
    if gone.is_empty() {
        return Ok(());
    }
    delete_rows(engine, txn, None, |r| gone.contains(&r.table))
}

// === SET CONSTRAINTS =====================================================================

/// `SET CONSTRAINTS {ALL | name, ...} {DEFERRED | IMMEDIATE}` inside `txn`.
///
/// A name resolves among the tables of its schema (or, unqualified, the schemas of the search
/// path), and reaches the copies a partitioned table's constraint made on its partitions. An
/// unknown name is `42704`, a constraint that is not deferrable `55000`. `IMMEDIATE` checks what
/// the named constraints have queued now; a violation fails the statement and leaves the mode as
/// it was.
///
/// # Errors
/// As above, plus the violations of an `IMMEDIATE` check.
pub(super) fn set_constraints(
    stmt: &ast::SetConstraints,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let targets = match &stmt.names {
        None => None,
        Some(names) => Some(resolve_names(names, engine, txn)?),
    };
    with_state(engine, txn, |state| match &targets {
        None => {
            state.mode.all = Some(stmt.deferred);
            state.mode.named.clear();
        },
        Some(targets) => {
            for c in targets {
                state.mode.named.insert(c.clone(), stmt.deferred);
            }
        },
    });
    if stmt.deferred {
        return Ok(ExecutionResult::ConstraintsSet);
    }
    // IMMEDIATE: check what the switched constraints queued, then drop those entries.
    let selected = |e: &Entry| targets.as_ref().is_none_or(|t| t.contains(&e.constraint));
    let due: Vec<Entry> = with_state(engine, txn, |state| {
        state
            .entries
            .iter()
            .filter(|e| selected(e))
            .cloned()
            .collect()
    });
    if !due.is_empty() {
        check_entries(engine, txn, &due)?;
        with_state(engine, txn, |state| state.retain(|e| !selected(e)));
    }
    Ok(ExecutionResult::ConstraintsSet)
}

/// The deferrable constraints `names` designate (a name reaches the partition copies made from
/// it).
fn resolve_names(
    names: &[(Option<String>, String)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<ConstraintRef>, Error> {
    let rows = catalog_rows(engine, txn)?;
    let search_path: Vec<String> = {
        let mut schemas = Vec::new();
        if let Some(temp) = session_ctx::current_temp_schema() {
            schemas.push(temp);
        }
        schemas.extend(crate::search_path_schemas(
            session_ctx::setting("search_path").as_deref(),
        ));
        schemas
    };
    let mut out = Vec::new();
    for (schema, name) in names {
        let in_scope = |table: TableId| -> Result<bool, Error> {
            Ok(engine
                .lookup_table_by_id_as_of(txn, table)?
                .is_some_and(|t| match schema {
                    Some(s) => &t.schema == s,
                    None => search_path.contains(&t.schema),
                }))
        };
        let mut found = false;
        for r in rows.iter() {
            if (r.name == *name || r.origin == *name)
                && in_scope(r.table)?
                && constraint_exists(engine, txn, &r.constraint_ref())?
            {
                found = true;
                out.push(r.constraint_ref());
            }
        }
        if found {
            continue;
        }
        // Not deferrable, or not there at all.
        let display = schema
            .as_ref()
            .map_or_else(|| name.clone(), |s| format!("{s}.{name}"));
        for (table_schema, table_name) in engine.list_tables_qualified_as_of(txn)? {
            let matches_schema = schema.as_ref().map_or_else(
                || search_path.contains(&table_schema),
                |s| *s == table_schema,
            );
            if !matches_schema || table_name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
                continue;
            }
            let Some(t) = engine.lookup_table_as_of_in(txn, &table_schema, &table_name)? else {
                continue;
            };
            let named = engine
                .list_constraints_as_of(txn, t.id)?
                .iter()
                .any(|c| c.name == *name)
                || engine
                    .list_foreign_keys_as_of(txn, t.id)?
                    .iter()
                    .any(|fk| fk.name == *name && fk.child_table == t.id);
            if named {
                return Err(Error::Coded {
                    message: format!("constraint \"{display}\" is not deferrable"),
                    sqlstate: "55000",
                });
            }
        }
        return Err(Error::ObjectNotFound(format!(
            "constraint \"{display}\" does not exist"
        )));
    }
    Ok(out)
}

/// Whether the constraint a catalog row names still exists.
fn constraint_exists(
    engine: &dyn StorageEngine,
    txn: TxnId,
    c: &ConstraintRef,
) -> Result<bool, Error> {
    Ok(match c.kind {
        Kind::Unique => engine
            .list_constraints_as_of(txn, c.table)?
            .iter()
            .any(|k| {
                k.name == c.name
                    && matches!(
                        k.kind,
                        nusadb_core::ConstraintKind::PrimaryKey
                            | nusadb_core::ConstraintKind::Unique
                    )
            }),
        Kind::ForeignKey => engine
            .list_foreign_keys_as_of(txn, c.table)?
            .iter()
            .any(|fk| (fk.name.as_str(), fk.child_table) == (c.name.as_str(), c.table)),
        Kind::Exclusion => engine
            .list_constraints_as_of(txn, c.table)?
            .iter()
            .any(|k| k.name == c.name && super::exclusion::is_exclusion(k)),
    })
}

// === The checks ==========================================================================

/// Read the latest committed state (plus `txn`'s own writes) while alive.
struct FreshReads<'e> {
    engine: &'e dyn StorageEngine,
    txn: TxnId,
}

impl<'e> FreshReads<'e> {
    fn begin(engine: &'e dyn StorageEngine, txn: TxnId) -> Result<Self, Error> {
        engine.begin_fresh_reads(txn)?;
        Ok(Self { engine, txn })
    }
}

impl Drop for FreshReads<'_> {
    fn drop(&mut self) {
        let _ = self.engine.end_fresh_reads(self.txn);
    }
}

/// The keys queued for one constraint, and whether it is checked whole.
#[derive(Default)]
struct Group {
    keys: Vec<Vec<ast::Value>>,
    removed_parent_keys: Vec<Vec<ast::Value>>,
    whole: bool,
}

/// Run the checks `entries` ask for, constraint by constraint in the order first queued, and
/// return the first violation.
fn check_entries(engine: &dyn StorageEngine, txn: TxnId, entries: &[Entry]) -> Result<(), Error> {
    let _fresh = FreshReads::begin(engine, txn)?;
    // The checks read the catalog under fresh reads, not the snapshot a cached read was made under.
    new_statement();
    let mut order: Vec<ConstraintRef> = Vec::new();
    let mut groups: HashMap<ConstraintRef, Group> = HashMap::new();
    for e in entries {
        let group = groups.entry(e.constraint.clone()).or_insert_with(|| {
            order.push(e.constraint.clone());
            Group::default()
        });
        match &e.pending {
            Pending::Key(k) => group.keys.push(k.clone()),
            Pending::RemovedParentKey(k) => group.removed_parent_keys.push(k.clone()),
            Pending::Whole => group.whole = true,
        }
    }
    for c in order {
        let Some(group) = groups.remove(&c) else {
            continue;
        };
        match c.kind {
            Kind::Unique => check_unique(engine, txn, &c, group)?,
            Kind::ForeignKey => check_foreign_key(engine, txn, &c, group)?,
            Kind::Exclusion => check_exclusion(engine, txn, &c, &group)?,
        }
    }
    Ok(())
}

/// Sort and deduplicate keys under the executor's total order.
fn distinct_keys(mut keys: Vec<Vec<ast::Value>>) -> Vec<Vec<ast::Value>> {
    keys.sort_by(|a, b| unique_key_cmp(a, b));
    keys.dedup_by(|a, b| unique_key_cmp(a, b) == std::cmp::Ordering::Equal);
    keys
}

/// A unique constraint: no two live rows may hold one of the queued keys.
fn check_unique(
    engine: &dyn StorageEngine,
    txn: TxnId,
    c: &ConstraintRef,
    group: Group,
) -> Result<(), Error> {
    let Some(table) = engine.lookup_table_by_id_as_of(txn, c.table)? else {
        return Ok(()); // dropped since
    };
    let Some(constraint) = engine
        .list_constraints_as_of(txn, table.id)?
        .into_iter()
        .find(|k| {
            k.name == c.name
                && matches!(
                    k.kind,
                    nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
                )
        })
    else {
        return Ok(());
    };
    let ordinals = constraint_ordinals(&table, &constraint.columns)?;
    let nnd = constraint.nulls_not_distinct;
    let violation =
        || unique_violation_error(constraint.kind, &constraint.name, &constraint.columns);
    let keys = distinct_keys(group.keys);
    let index = if group.whole || nnd || keys.len() > WHOLE_CONSTRAINT_AFTER {
        None
    } else {
        engine.lookup_index(&constraint.name)?
    };
    let probe_safe = keys
        .iter()
        .all(|k| k.iter().all(super::ops::is_hash_safe_value));
    if let (Some(index), true) = (index, probe_safe) {
        let types = column_types(&table);
        for key in &keys {
            let encoded = index_key::encode_index_key(key)?;
            let mut scan = engine.index_scan_committed(
                txn,
                index,
                std::ops::Bound::Included(encoded.clone()),
                std::ops::Bound::Included(encoded),
            )?;
            let mut holders: HashSet<Tid> = HashSet::new();
            while let Some((tid, tuple)) = scan.try_next()? {
                crate::cancel::check()?;
                let row = row::decode(&tuple, &types)?;
                if unique_key(&row, &ordinals, nnd).is_some_and(|k| unique_key_eq(&k, key)) {
                    holders.insert(tid);
                }
            }
            if holders.len() > 1 {
                return Err(violation());
            }
        }
        return Ok(());
    }
    // One read of the table: any two live rows sharing a key (a queued one or not; the committed
    // state holds no duplicate of its own) violate the constraint.
    let mut all: Vec<Vec<ast::Value>> = scan_table_committed(&table, engine, txn)?
        .into_iter()
        .filter_map(|(_, row)| unique_key(&row, &ordinals, nnd))
        .collect();
    all.sort_by(|a, b| unique_key_cmp(a, b));
    if all
        .windows(2)
        .any(|w| matches!(w, [a, b] if unique_key_cmp(a, b) == std::cmp::Ordering::Equal))
    {
        return Err(violation());
    }
    Ok(())
}

/// How the check finds whether a live row of a table holds a key: one read of the whole table
/// (its keys, sorted), or a probe of an index per key.
enum Lookup {
    /// The table's live keys, sorted and deduplicated.
    Keys(Vec<Vec<ast::Value>>),
    /// An index over the key columns, with, per index key column, its position in the key.
    Index(nusadb_core::IndexId, Vec<usize>),
}

impl Lookup {
    /// Probe an index when there are few keys to look for and each is index-safe and an index
    /// over the columns exists (preferring `preferred`); otherwise read the table once.
    fn new(
        table: &TableSchema,
        ordinals: &[usize],
        columns: &[String],
        preferred: Option<&str>,
        keys: &[&[ast::Value]],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<Self, Error> {
        let probe = keys.len() <= 64
            && keys
                .iter()
                .all(|k| k.iter().all(super::ops::is_hash_safe_value));
        if probe && let Some((id, order)) = index_over(table, columns, preferred, engine, txn)? {
            return Ok(Self::Index(id, order));
        }
        Ok(Self::Keys(distinct_keys(
            scan_table_committed(table, engine, txn)?
                .into_iter()
                .filter_map(|(_, row)| unique_key(&row, ordinals, false))
                .collect(),
        )))
    }

    /// Whether a live row of `table` holds `key` at `ordinals`.
    fn holds(
        &self,
        table: &TableSchema,
        ordinals: &[usize],
        key: &[ast::Value],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<bool, Error> {
        match self {
            Self::Keys(all) => Ok(all.binary_search_by(|p| unique_key_cmp(p, key)).is_ok()),
            Self::Index(index, order) => {
                let probe: Vec<ast::Value> =
                    order.iter().filter_map(|&i| key.get(i).cloned()).collect();
                let encoded = index_key::encode_index_key(&probe)?;
                let mut scan = engine.index_scan_committed(
                    txn,
                    *index,
                    std::ops::Bound::Included(encoded.clone()),
                    std::ops::Bound::Included(encoded),
                )?;
                let types = column_types(table);
                while let Some((_, tuple)) = scan.try_next()? {
                    crate::cancel::check()?;
                    let row = row::decode(&tuple, &types)?;
                    if unique_key(&row, ordinals, false).is_some_and(|k| unique_key_eq(&k, key)) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
        }
    }
}

/// An index whose key columns are exactly `columns` (in any order), with, per index key column,
/// its position in `columns`.
fn index_over(
    table: &TableSchema,
    columns: &[String],
    preferred: Option<&str>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<(nusadb_core::IndexId, Vec<usize>)>, Error> {
    let defs = engine.list_indexes_as_of(txn, table.id)?;
    let usable = |def: &nusadb_core::IndexDef| {
        def.kind == nusadb_core::engine::IndexKind::BTree
            && def.key_exprs.is_empty()
            && def.predicate.is_none()
            && def.columns.len() == columns.len()
    };
    let mut ordered: Vec<&nusadb_core::IndexDef> = defs.iter().filter(|d| usable(d)).collect();
    ordered.sort_by_key(|d| Some(d.name.as_str()) != preferred);
    for def in ordered {
        let order: Option<Vec<usize>> = def
            .columns
            .iter()
            .map(|col| columns.iter().position(|c| c == col))
            .collect();
        let Some(order) = order else { continue };
        let distinct: HashSet<usize> = order.iter().copied().collect();
        if distinct.len() != order.len() {
            continue;
        }
        if let Some(id) = engine.lookup_index(&def.name)? {
            return Ok(Some((id, order)));
        }
    }
    Ok(None)
}

/// A foreign key: a live child holding a queued key needs a live parent holding it.
fn check_foreign_key(
    engine: &dyn StorageEngine,
    txn: TxnId,
    c: &ConstraintRef,
    group: Group,
) -> Result<(), Error> {
    let Some(fk) = engine
        .list_foreign_keys_as_of(txn, c.table)?
        .into_iter()
        .find(|fk| (fk.name.as_str(), fk.child_table) == (c.name.as_str(), c.table))
    else {
        return Ok(());
    };
    let (Some(child), Some(parent)) = (
        engine.lookup_table_by_id_as_of(txn, fk.child_table)?,
        engine.lookup_table_by_id_as_of(txn, fk.parent_table)?,
    ) else {
        return Ok(());
    };
    let Some(parent_ordinals) = fk_parent_ordinals(&parent, &fk, engine, txn)? else {
        return Ok(());
    };
    let child_ordinals = constraint_ordinals(&child, &fk.child_columns)?;
    let parent_columns: Vec<String> = parent_ordinals
        .iter()
        .filter_map(|&i| parent.columns.get(i).map(|col| col.name.clone()))
        .collect();
    // Which keys to verify, each with whether a child write (rather than a parent removal) queued
    // it, for the message.
    let child_keys = if group.whole {
        let all: Vec<Vec<ast::Value>> = scan_table_committed(&child, engine, txn)?
            .into_iter()
            .filter_map(|(_, row)| unique_key(&row, &child_ordinals, false))
            .collect();
        distinct_keys(all)
    } else {
        distinct_keys(group.keys)
    };
    let mut keys: Vec<(Vec<ast::Value>, bool)> =
        child_keys.into_iter().map(|k| (k, true)).collect();
    for k in distinct_keys(group.removed_parent_keys) {
        if !keys.iter().any(|(q, _)| unique_key_eq(q, &k)) {
            keys.push((k, false));
        }
    }
    if keys.is_empty() {
        return Ok(());
    }
    // Hold each referenced parent key (as the child write did), then read.
    for (k, _) in &keys {
        lock_referenced_key(
            &parent,
            &parent_ordinals,
            k,
            nusadb_core::engine::RowLockMode::Shared,
            engine,
            txn,
        )?;
    }
    let wanted: Vec<&[ast::Value]> = keys.iter().map(|(k, _)| k.as_slice()).collect();
    let parents = Lookup::new(
        &parent,
        &parent_ordinals,
        &parent_columns,
        None,
        &wanted,
        engine,
        txn,
    )?;
    let mut missing: Vec<(Vec<ast::Value>, bool)> = Vec::new();
    for (k, from_child) in keys {
        if !parents.holds(&parent, &parent_ordinals, &k, engine, txn)? {
            missing.push((k, from_child));
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let wanted: Vec<&[ast::Value]> = missing.iter().map(|(k, _)| k.as_slice()).collect();
    let children = Lookup::new(
        &child,
        &child_ordinals,
        &fk.child_columns,
        Some(&fk.name),
        &wanted,
        engine,
        txn,
    )?;
    for (k, from_child) in missing {
        if children.holds(&child, &child_ordinals, &k, engine, txn)? {
            let message = if from_child {
                format!(
                    "insert or update on \"{}\" violates foreign key \"{}\": no matching row in \
                     \"{}\"",
                    child.name, fk.name, parent.name
                )
            } else {
                format!(
                    "update or delete on \"{}\" violates foreign key \"{}\": a row in \"{}\" \
                     still references the removed key",
                    parent.name, fk.name, child.name
                )
            };
            return Err(nusadb_core::Error::ConstraintViolation(message).into());
        }
    }
    Ok(())
}

/// An exclusion constraint: no two live rows of a queued key's `=` group may conflict.
fn check_exclusion(
    engine: &dyn StorageEngine,
    txn: TxnId,
    c: &ConstraintRef,
    group: &Group,
) -> Result<(), Error> {
    let Some(table) = engine.lookup_table_by_id_as_of(txn, c.table)? else {
        return Ok(());
    };
    let Some(compiled) = super::exclusion::load(&table, engine, txn)?
        .into_iter()
        .find(|x| x.name() == c.name)
    else {
        return Ok(());
    };
    let keys = (!group.whole).then_some(group.keys.as_slice());
    super::exclusion::recheck(&compiled, &table, keys, engine, txn)
}

/// The places in `sources` that end, roll back, prepare or mark a transaction directly instead of
/// through this module's wrappers: a method call (any receiver, also split across lines)
/// `.commit(`, `.rollback(`, `.savepoint(`, `.rollback_to(`, `.release_savepoint(` or
/// `.prepare_transaction(` whose first argument names a transaction (contains `txn`). Test code
/// (an item under `#[cfg(test)]`) is skipped, and so are comments. Each place is reported as a
/// short excerpt of the code with whitespace removed; one is allowed when `allowed` holds a pair
/// of a file suffix and a substring of its excerpt.
#[doc(hidden)]
#[must_use]
pub fn boundary_offenders(sources: &[(&str, &str)], allowed: &[(&str, &str)]) -> Vec<String> {
    const CALLS: [&str; 6] = [
        ".commit(",
        ".rollback(",
        ".savepoint(",
        ".rollback_to(",
        ".release_savepoint(",
        ".prepare_transaction(",
    ];
    let mut out = Vec::new();
    for (file, text) in sources {
        let code: String = production_lines(text)
            .into_iter()
            .map(|line| line.split("//").next().unwrap_or(line))
            .flat_map(str::chars)
            .filter(|c| !c.is_whitespace())
            .collect();
        for call in CALLS {
            for (at, _) in code.match_indices(call) {
                let args = code.get(at + call.len()..).unwrap_or_default();
                let first = args.split([',', ')']).next().unwrap_or_default();
                if !first.to_ascii_lowercase().contains("txn") {
                    continue;
                }
                let from = code.floor_char_boundary(at.saturating_sub(40));
                let to = code.ceil_char_boundary((at + call.len() + 40).min(code.len()));
                let excerpt = code.get(from..to).unwrap_or_default();
                let excused = allowed
                    .iter()
                    .any(|(f, a)| file.ends_with(f) && excerpt.contains(a));
                if !excused {
                    out.push(format!("{file}: ...{excerpt}..."));
                }
            }
        }
    }
    out
}

/// The lines of `text` outside its test code: each item marked `#[cfg(test)]` at the start of a
/// line (with the attributes after the marker) is left out up to its end, a `;`-terminated line
/// or the closing `}` at the start of a line (the shape formatted code has).
fn production_lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim_end() != "#[cfg(test)]" {
            out.push(line);
            continue;
        }
        // Attributes until the item line.
        let mut item = None;
        for next in lines.by_ref() {
            let starts_item = next.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
            if starts_item {
                item = Some(next);
                break;
            }
        }
        if item.is_some_and(|l| !l.trim_end().ends_with(';')) {
            for next in lines.by_ref() {
                if next.trim_end() == "}" {
                    break;
                }
            }
        }
    }
    out
}

/// Every `.rs` file under `dir`, recursively, except test files (`tests.rs`, anything under a
/// `tests` directory), as `(path, text)`.
#[cfg(test)]
fn source_files(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        if path.is_dir() {
            if name != "tests" {
                source_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs")
            && name != "tests.rs"
            && let Ok(text) = std::fs::read_to_string(&path)
        {
            out.push((path.display().to_string(), text));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::panic,
        reason = "unit test asserts via unwrap/panic"
    )]

    use super::*;
    use crate::{Catalog, IndexInfo, Session, analyze, parse, plan};
    use nusadb_btree::BtreeEngine;

    struct Cat<'a>(&'a dyn StorageEngine, Option<TxnId>);
    impl Catalog for Cat<'_> {
        fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
            self.1
                .map_or_else(
                    || self.0.lookup_table(name),
                    |txn| self.0.lookup_table_as_of(txn, name),
                )
                .map_err(Into::into)
        }
        fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
            Ok(Vec::new())
        }
    }

    fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Result<(), Error> {
        let logical = analyze(parse(sql)?, &Cat(engine, session.current_txn()))?;
        session.execute(plan(logical)).map(|_| ())
    }

    fn ok(engine: &dyn StorageEngine, session: &mut Session, sql: &str) {
        run(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }

    #[test]
    fn the_queue_follows_statements_savepoints_and_the_transaction() {
        let engine = BtreeEngine::new();
        let mut s = Session::new(&engine);
        ok(
            &engine,
            &mut s,
            "CREATE TABLE t (id INT PRIMARY KEY DEFERRABLE INITIALLY DEFERRED, \
             v INT CHECK (v > 0))",
        );
        ok(&engine, &mut s, "BEGIN");
        let txn = s.current_txn().unwrap();
        assert_eq!(queued(&engine, txn), None);
        ok(&engine, &mut s, "INSERT INTO t VALUES (1, 1)");
        assert_eq!(queued(&engine, txn), Some(1));
        ok(&engine, &mut s, "SAVEPOINT a");
        ok(&engine, &mut s, "INSERT INTO t VALUES (2, 1), (3, 1)");
        assert_eq!(queued(&engine, txn), Some(3));
        // A statement that fails after queueing its keys leaves nothing queued.
        assert!(run(&engine, &mut s, "INSERT INTO t VALUES (4, -1)").is_err());
        assert_eq!(queued(&engine, txn), Some(3));
        ok(&engine, &mut s, "ROLLBACK TO SAVEPOINT a");
        assert_eq!(queued(&engine, txn), Some(1));
        ok(&engine, &mut s, "SET CONSTRAINTS ALL IMMEDIATE");
        assert_eq!(queued(&engine, txn), Some(0));
        ok(&engine, &mut s, "COMMIT");
        assert_eq!(queued(&engine, txn), None);

        // A rollback forgets the state too, and a failed commit leaves none behind.
        ok(&engine, &mut s, "BEGIN");
        let txn = s.current_txn().unwrap();
        ok(&engine, &mut s, "INSERT INTO t VALUES (1, 1)");
        assert!(run(&engine, &mut s, "COMMIT").is_err());
        assert_eq!(queued(&engine, txn), None);
        ok(&engine, &mut s, "BEGIN");
        let txn = s.current_txn().unwrap();
        ok(&engine, &mut s, "INSERT INTO t VALUES (9, 1)");
        ok(&engine, &mut s, "ROLLBACK");
        assert_eq!(queued(&engine, txn), None);
    }

    #[test]
    fn a_rolled_back_alter_constraint_is_not_served_from_the_cache() {
        let engine = BtreeEngine::new();
        let mut s = Session::new(&engine);
        ok(
            &engine,
            &mut s,
            "CREATE TABLE t (id INT CONSTRAINT t_pk PRIMARY KEY)",
        );
        ok(&engine, &mut s, "INSERT INTO t VALUES (1)");
        ok(&engine, &mut s, "BEGIN");
        ok(&engine, &mut s, "SAVEPOINT a");
        ok(
            &engine,
            &mut s,
            "ALTER TABLE t ALTER CONSTRAINT t_pk DEFERRABLE INITIALLY DEFERRED",
        );
        ok(&engine, &mut s, "INSERT INTO t VALUES (1)");
        ok(&engine, &mut s, "ROLLBACK TO SAVEPOINT a");
        // Immediate again: the duplicate fails at once, not at COMMIT.
        assert!(run(&engine, &mut s, "INSERT INTO t VALUES (1)").is_err());
        ok(&engine, &mut s, "COMMIT");
    }

    fn entry(n: i64) -> Entry {
        Entry {
            constraint: ConstraintRef::unique(TableId(1), "u"),
            pending: Pending::Key(vec![ast::Value::Int(n)]),
        }
    }

    #[test]
    fn retain_moves_savepoint_lengths_with_the_entries() {
        let mut s = TxnDeferred::default();
        for n in 0..4 {
            s.entries.push(entry(n));
        }
        s.marks.push(Mark {
            name: "a".to_owned(),
            len: 2,
            mode: Mode::default(),
        });
        s.marks.push(Mark {
            name: "b".to_owned(),
            len: 4,
            mode: Mode::default(),
        });
        s.retain(|e| e.pending != Pending::Key(vec![ast::Value::Int(0)]));
        assert_eq!(s.entries.len(), 3);
        assert_eq!(s.marks.first().map(|m| m.len), Some(1));
        assert_eq!(s.marks.get(1).map(|m| m.len), Some(3));
    }

    #[test]
    fn mode_resolution_order() {
        let c = ConstraintRef::unique(TableId(1), "u");
        let deferrable = ast::Deferral {
            deferrable: true,
            initially_deferred: false,
        };
        let mut m = Mode::default();
        assert!(!m.defers(&c, deferrable));
        m.all = Some(true);
        assert!(m.defers(&c, deferrable));
        m.named.insert(c.clone(), false);
        assert!(!m.defers(&c, deferrable));
        // Never for a constraint that is not deferrable.
        m.named.insert(c.clone(), true);
        assert!(!m.defers(&c, ast::Deferral::IMMEDIATE));
    }

    #[test]
    fn a_collapsed_check_survives_a_rollback_that_keeps_some_of_its_keys() {
        let other = ConstraintRef::unique(TableId(2), "o");
        let c = ConstraintRef::unique(TableId(1), "u");
        let key = |n: usize| Pending::Key(vec![ast::Value::Int(i64::try_from(n).unwrap())]);
        let mut s = TxnDeferred::default();
        s.push(other.clone(), key(0));
        s.push(c.clone(), key(0));
        // A savepoint after the first key of `c` and one before it.
        s.marks.push(Mark {
            name: "before".to_owned(),
            len: 1,
            mode: Mode::default(),
        });
        s.marks.push(Mark {
            name: "after".to_owned(),
            len: 2,
            mode: Mode::default(),
        });
        for n in 1..=WHOLE_CONSTRAINT_AFTER {
            s.push(c.clone(), key(n));
            if n % 10 == 0 {
                s.push(other.clone(), key(n));
            }
        }
        let whole_at = s
            .entries
            .iter()
            .position(|e| e.pending == Pending::Whole && e.constraint == c)
            .unwrap();
        assert_eq!(whole_at, 1);
        assert_eq!(s.marks.first().map(|m| m.len), Some(1));
        assert_eq!(s.marks.get(1).map(|m| m.len), Some(2));
        // Rolling back to `after` keeps the whole-table check.
        let len = s.marks.get(1).unwrap().len;
        s.entries.truncate(len);
        s.recount();
        assert!(s.whole.contains(&c));
    }

    #[test]
    fn too_many_keys_collapse_to_a_whole_table_check() {
        let mut s = TxnDeferred::default();
        let c = ConstraintRef::unique(TableId(1), "u");
        for n in 0..=WHOLE_CONSTRAINT_AFTER {
            s.push(
                c.clone(),
                Pending::Key(vec![ast::Value::Int(i64::try_from(n).unwrap())]),
            );
        }
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries.first().map(|e| &e.pending), Some(&Pending::Whole));
        s.push(c, Pending::Key(vec![ast::Value::Int(-1)]));
        assert_eq!(s.entries.len(), 1);
    }

    #[test]
    fn the_sweep_drops_only_transactions_the_engine_reports_finished() {
        use nusadb_core::IsolationLevel;
        let engine = BtreeEngine::new();
        let c = ConstraintRef::unique(TableId(1), "u");
        let key = || vec![Pending::Key(vec![ast::Value::Int(1)])];
        // A live transaction with state, and more than enough leaked ones (ended directly on the
        // engine, bypassing the wrappers) to trigger a sweep.
        let live = engine.begin(IsolationLevel::default()).unwrap();
        defer_keys(&engine, live, &c, key());
        let mut leaked = Vec::new();
        for _ in 0..SWEEP_AT {
            let t = engine.begin(IsolationLevel::default()).unwrap();
            defer_keys(&engine, t, &c, key());
            leaked.push(t);
        }
        // While they are all open, a sweep keeps every one.
        let probe = engine.begin(IsolationLevel::default()).unwrap();
        defer_keys(&engine, probe, &c, key());
        assert!(leaked.iter().all(|t| queued(&engine, *t).is_some()));
        forget(&engine, probe);
        engine.rollback(probe).unwrap();
        for t in &leaked {
            engine.rollback(*t).unwrap();
        }
        let fresh = engine.begin(IsolationLevel::default()).unwrap();
        defer_keys(&engine, fresh, &c, key());
        assert!(leaked.iter().all(|t| queued(&engine, *t).is_none()));
        assert_eq!(queued(&engine, live), Some(1));
        assert_eq!(queued(&engine, fresh), Some(1));
        forget(&engine, live);
        forget(&engine, fresh);
        engine.rollback(live).unwrap();
        engine.rollback(fresh).unwrap();
    }

    #[test]
    fn the_scanner_finds_direct_calls_split_across_lines() {
        let source = concat!(
            "fn f() {\n",
            "    engine\n",
            "        .commit(txn)?;\n",
            "    deferred::commit(engine, txn)?;\n",
            "    // engine.rollback(txn) in a comment\n",
            "    let _ = self.engine.rollback(txn);\n",
            "    let _ = other.rollback_to(self.txn, \"s\");\n",
            "    pending.savepoint(name);\n",
            "}\n",
            "#[cfg(test)]\n",
            "fn helper() { engine.commit(txn); }\n",
            "#[cfg(test)]\n",
            "#[allow(\n",
            "    clippy::unwrap_used,\n",
            "    reason = \"test\"\n",
            ")]\n",
            "mod tests {\n",
            "    fn g() {\n",
            "        engine.commit(txn);\n",
            "    }\n",
            "}\n",
            "fn after() { e.prepare_transaction(*txn, gid); }\n",
        );
        let found = boundary_offenders(&[("src/x.rs", source)], &[]);
        assert_eq!(found.len(), 4, "{found:?}");
        assert!(
            boundary_offenders(
                &[("src/x.rs", source)],
                &[
                    ("x.rs", "engine.commit(txn)?;"),
                    ("x.rs", "self.engine.rollback"),
                    ("x.rs", "other.rollback_to"),
                    ("x.rs", "e.prepare_transaction"),
                ]
            )
            .is_empty()
        );
        // An allowance is scoped to its file.
        assert_eq!(
            boundary_offenders(&[("src/y.rs", source)], &[("x.rs", "engine.commit(txn)?;")]).len(),
            4
        );
    }

    /// Every commit, rollback, prepare and savepoint operation on a transaction in the crates
    /// that run SQL on an engine (this one, the wire server, the server binary, the C API and
    /// the client library) goes through the wrappers in this module: a direct engine call would
    /// skip the deferred checks (a commit or prepare) or leave the queue out of step (a rollback
    /// or savepoint). The allowed exceptions are the wrappers themselves and the prepare that
    /// follows the deferred checks.
    #[test]
    fn transaction_boundaries_go_through_the_wrappers() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let mut files = Vec::new();
        for name in [
            "nusadb-sql",
            "nusadb-wire",
            "nusadb-server",
            "nusadb-capi",
            "nusadb-libnusa",
        ] {
            source_files(&crates.join(name).join("src"), &mut files);
        }
        assert!(
            files.iter().any(|(p, _)| p.ends_with("server.rs"))
                && files.iter().any(|(p, _)| p.ends_with("executor/mod.rs")),
            "the scan did not find the sources"
        );
        let sources: Vec<(&str, &str)> = files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let allowed = [
            (
                "executor/deferred.rs",
                "let_=engine.rollback(txn);returnErr(e);",
            ),
            (
                "executor/deferred.rs",
                "engine.commit(txn).map_err(Into::into)",
            ),
            (
                "executor/deferred.rs",
                "forget(engine,txn);engine.rollback(txn)",
            ),
            ("executor/deferred.rs", "engine.savepoint(txn,name)?;"),
            ("executor/deferred.rs", "engine.rollback_to(txn,name)?;"),
            (
                "executor/deferred.rs",
                "engine.release_savepoint(txn,name)?;",
            ),
            (
                "two_phase.rs",
                "deferred::prepare(engine,txn)?;engine.prepare_transaction(txn,gid,owner,database)",
            ),
        ];
        let offenders = boundary_offenders(&sources, &allowed);
        assert!(
            offenders.is_empty(),
            "direct engine transaction calls bypass the deferred-constraint wrappers:\n{}",
            offenders.join("\n")
        );
    }
}
