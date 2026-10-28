//! `PRIMARY KEY` / `UNIQUE` on a partitioned table that leaves out a partitioning column: a key
//! enforced across every partition below the table.
//!
//! # What makes a key global
//!
//! A key that includes every column of the partition key needs nothing more than the copy each
//! partition holds: equal keys always route to the same partition. A key that leaves a partitioning
//! column out (or sits on a table partitioned by an expression) can put two equal keys in two
//! partitions, where no single index sees both. Such a key is enforced over the rows of every
//! partition below the table that declares it, its *scope*.
//!
//! Nothing new is stored. Each partition keeps its own copy of the constraint and that copy's
//! backing index, exactly as for a covering key; whether a copy is global is derived from the
//! catalogs. Walking up from a partition, each ancestor that holds the same key (same columns) is
//! one more level of the chain; the scope is the highest level of the chain whose partition key
//! the key does not cover. Below a covering level, equal keys already route to one subtree, so the
//! highest non-covering level is the widest set of partitions that two equal keys can reach.
//!
//! # Enforcement
//!
//! Every write of a key into a partition takes, besides the partition's own key lock, a no-wait
//! exclusive key lock in the scope's namespace (`lock_key` on the scope's table id), held to the
//! end of the transaction. The key is then remembered by the statement, and when the statement
//! finishes (a nested statement of a function or trigger body included, so an `EXCEPTION` block
//! catches the error) each remembered key is looked up in every partition of the scope, through
//! each partition's backing index with latest-committed visibility plus the transaction's own
//! writes. More than one live row holding the key is `23505`, named after the scope's constraint.
//! Checking at the end of the statement, not row by row, lets one statement move keys between
//! partitions (an `UPDATE` that swaps two keys across partitions succeeds when the final state is
//! valid).
//!
//! The partitions of a scope are listed from the committed partition catalog, read after the key
//! locks are held, never from the transaction's snapshot: a `REPEATABLE READ` transaction that
//! began before a partition was created and filled still sees that partition's keys. Any other
//! transaction writing an equal key either took the scope lock first (this write fails with
//! `40001`) or committed before this one locked it, and is then visible to the check.
//!
//! # Cost
//!
//! Each written key costs one index probe per partition of the scope at the end of the statement:
//! the price of not having a global index, and why creating such a key raises a notice. A statement
//! that writes many keys into a small partition reads that partition once instead.
#![allow(clippy::wildcard_imports)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use nusadb_core::{ConstraintKind, TableId, Tid};

use super::*;

/// A key tuple.
type Key = Vec<ast::Value>;

/// The table whose partitions one global key spans, and the constraint as it declares it.
#[derive(Debug, Clone)]
pub(super) struct Scope {
    /// The partitioned table whose subtree the key must be unique over.
    pub(super) table: TableSchema,
    /// The constraint's name on that table (the name a violation reports).
    pub(super) constraint: String,
    /// `PrimaryKey` or `Unique`.
    pub(super) kind: ConstraintKind,
    /// The key columns.
    pub(super) columns: Vec<String>,
}

impl Scope {
    /// The `23505` a duplicate across the scope's partitions raises.
    fn violation(&self) -> Error {
        dml::unique_violation_error(self.kind, &self.constraint, &self.columns)
    }

    /// Take the exclusive key lock of `key` in the scope's namespace.
    pub(super) fn lock(
        &self,
        key: &[ast::Value],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        let hash = dml::unique_key_hash(&self.table, &self.columns, key);
        engine.lock_key(
            txn,
            self.table.id,
            hash,
            nusadb_core::engine::RowLockMode::Exclusive,
        )?;
        Ok(())
    }

    /// The entry a deferred check of this key is queued under.
    pub(super) fn deferral_ref(&self) -> super::deferred::ConstraintRef {
        super::deferred::ConstraintRef::global_unique(self.table.id, &self.constraint)
    }
}

/// One global key as one partition writes it: the partition's own copy of the constraint, the key
/// columns' positions in the partition's rows, and the scope.
#[derive(Debug)]
pub(super) struct LeafGlobal {
    /// The name of the partition's copy of the constraint.
    pub(super) leaf_constraint: String,
    /// The key columns' ordinals in the partition.
    pub(super) ordinals: Vec<usize>,
    /// The key columns' types.
    types: Vec<ColumnType>,
    /// Where the key must be unique.
    pub(super) scope: Rc<Scope>,
}

impl LeafGlobal {
    /// The key of `row` (a row about to be written), in the form a read of the stored row returns:
    /// a value may still be in its input form (an untyped literal held as text, say), and is
    /// locked, remembered and compared only once brought to the column's own type. `None` for a
    /// row with a NULL key column, which never conflicts.
    ///
    /// # Errors
    /// A value the column type cannot hold.
    pub(super) fn key_of(&self, row: &Row) -> Result<Option<Key>, Error> {
        dml::unique_key(row, &self.ordinals, false)
            .map(|key| stored_key(&key, &self.types))
            .transpose()
    }

    /// [`Self::key_of`] for a key already taken from a row.
    ///
    /// # Errors
    /// A value the column type cannot hold.
    pub(super) fn stored(&self, key: &[ast::Value]) -> Result<Key, Error> {
        stored_key(key, &self.types)
    }
}

/// `key` round-tripped through the row encoding at `types`.
fn stored_key(key: &[ast::Value], types: &[ColumnType]) -> Result<Key, Error> {
    row::decode(&row::encode(key, types)?, types)
}

/// Whether the partition key `parts` is covered by `columns`: every part a plain column among
/// them. An expression part is never covered.
pub(super) fn covers(parts: &[super::partition::KeyPart], columns: &[String]) -> bool {
    parts.iter().all(|part| match part {
        super::partition::KeyPart::Column(name) => columns.contains(name),
        super::partition::KeyPart::Expression(_) => false,
    })
}

/// The table a schema-qualified catalog key names, as `txn` sees the catalog (committed state plus
/// its own changes).
fn lookup(key: &str, engine: &dyn StorageEngine, txn: TxnId) -> Result<Option<TableSchema>, Error> {
    let (schema, name) = crate::analyzer::split_qualified(key);
    Ok(engine.lookup_table_as_of_in(txn, schema.unwrap_or(nusadb_core::PUBLIC_SCHEMA), name)?)
}

/// Whether `c` is a `PRIMARY KEY` / `UNIQUE` constraint.
const fn is_unique(c: &nusadb_core::Constraint) -> bool {
    matches!(c.kind, ConstraintKind::PrimaryKey | ConstraintKind::Unique)
}

/// The scope of the unique constraint `c` of `table`, or `None` when the key is enforced by each
/// partition alone (see the module docs).
fn scope_in(
    tree: &super::partition::CommittedTree,
    table: &TableSchema,
    c: &nusadb_core::Constraint,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<Scope>, Error> {
    const MAX_DEPTH: usize = 64;
    let mut level = table.clone();
    let mut level_constraint = c.clone();
    let mut scope = None;
    for _ in 0..MAX_DEPTH {
        let key = crate::analyzer::qualified_display(&level.schema, &level.name);
        if let Some(parts) = tree.key_parts(&key)
            && !covers(parts, &level_constraint.columns)
        {
            scope = Some(Scope {
                table: level.clone(),
                constraint: level_constraint.name.clone(),
                kind: level_constraint.kind,
                columns: level_constraint.columns.clone(),
            });
        }
        let Some(parent_key) = tree.parent(&key) else {
            break;
        };
        let Some(parent) = lookup(parent_key, engine, txn)? else {
            break;
        };
        let Some(above) = engine
            .list_constraints_as_of(txn, parent.id)?
            .into_iter()
            .find(|p| is_unique(p) && p.columns == c.columns)
        else {
            break;
        };
        level = parent;
        level_constraint = above;
    }
    Ok(scope)
}

/// The scope of the unique constraint `name` of `table`, read now (DDL and `SET CONSTRAINTS`).
///
/// # Errors
/// Storage errors.
pub(super) fn scope_of(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<Scope>, Error> {
    let tree = super::partition::committed_tree(engine, txn)?;
    if tree.is_empty() {
        return Ok(None);
    }
    let Some(c) = engine
        .list_constraints_as_of(txn, table.id)?
        .into_iter()
        .find(|c| is_unique(c) && c.name == name)
    else {
        return Ok(None);
    };
    scope_in(&tree, table, &c, engine, txn)
}

// === Per-statement cache =================================================================

/// Bumped by every statement that may change the partition tree or a table's constraints, so a
/// cached answer is never reused across one.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// What one statement learned about the global keys of the tables it writes, for one engine,
/// transaction and epoch.
struct Cache {
    engine: usize,
    txn: TxnId,
    epoch: u64,
    tree: Option<Arc<super::partition::CommittedTree>>,
    leaves: HashMap<TableId, Rc<Vec<LeafGlobal>>>,
    canon: Option<Arc<super::char_canon::CanonState>>,
    /// Per legacy unique index checked against its table's rows, the keys those rows held when the
    /// statement last read the table (see [`legacy_key_held`]).
    legacy_keys: HashMap<nusadb_core::IndexId, Rc<LegacyKeys>>,
}

thread_local! {
    static CACHE: RefCell<Option<Cache>> = const { RefCell::new(None) };
}

fn engine_addr(engine: &dyn StorageEngine) -> usize {
    std::ptr::from_ref(engine).cast::<()>().addr()
}

/// A statement starts on this thread: forget what an earlier one cached.
pub(super) fn new_statement() {
    CACHE.with(|c| c.borrow_mut().take());
}

/// A statement that may have changed the partition tree or a constraint finished: no cached
/// answer is used again.
pub(super) fn invalidate() {
    EPOCH.fetch_add(1, Ordering::AcqRel);
}

/// Run `f` on this statement's cache, starting a fresh one when it belongs to something else.
fn with_cache<T>(engine: &dyn StorageEngine, txn: TxnId, f: impl FnOnce(&mut Cache) -> T) -> T {
    let addr = engine_addr(engine);
    let epoch = EPOCH.load(Ordering::Acquire);
    CACHE.with(|c| {
        let mut slot = c.borrow_mut();
        match slot.as_mut() {
            Some(cache) if cache.engine == addr && cache.txn == txn && cache.epoch == epoch => {
                f(cache)
            },
            _ => f(slot.insert(Cache {
                engine: addr,
                txn,
                epoch,
                tree: None,
                leaves: HashMap::new(),
                canon: None,
                legacy_keys: HashMap::new(),
            })),
        }
    })
}

// === The partition tree shared across statements ========================================
//
// Every write into a partition asks which of its keys are global, which needs the partition tree.
// Reading the partition catalog for that on every statement would cost a table with covering keys
// a catalog read per write, so the committed tree is kept between statements, per engine, and
// dropped whenever it may have changed:
//
// - a transaction that writes the partition catalog registers itself ([`catalog_written`]); when it
//   ends ([`around_end`]), the tree epoch moves before and after the engine ends it, and while any
//   such end is in flight no tree is cached or served;
// - a prepared transaction ends through `COMMIT PREPARED` / `ROLLBACK PREPARED`, which always move
//   the epoch the same way;
// - a new session or connection moves it too, so an engine created at the address of a dropped one
//   never reads that engine's tree.
//
// A transaction that wrote the catalog itself reads its own tree fresh. The end-of-statement check
// never uses this cache: it lists the partitions of a scope fresh, after the key locks are held.

/// Moves whenever the committed partition tree may have changed.
static TREE_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Ends of catalog-writing transactions in progress.
static TREE_INFLIGHT: AtomicUsize = AtomicUsize::new(0);

/// How many transactions are registered in [`WRITERS`], so the others never take its lock.
static WRITER_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The open transactions that wrote the partition catalog, by engine address.
static WRITERS: LazyLock<Mutex<HashSet<(usize, TxnId)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// The cached tree: engine address, partition-catalog table id, epoch, tree.
type SharedTree = (
    usize,
    Option<TableId>,
    u64,
    Arc<super::partition::CommittedTree>,
);

static SHARED_TREE: Mutex<Option<SharedTree>> = Mutex::new(None);

fn writers() -> std::sync::MutexGuard<'static, HashSet<(usize, TxnId)>> {
    WRITERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn shared_tree() -> std::sync::MutexGuard<'static, Option<SharedTree>> {
    SHARED_TREE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `txn` wrote the partition catalog: no tree it can see is shared, now or until it ends.
pub(super) fn catalog_written(engine: &dyn StorageEngine, txn: TxnId) {
    if writers().insert((engine_addr(engine), txn)) {
        WRITER_COUNT.fetch_add(1, Ordering::AcqRel);
    }
    TREE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

fn is_writer(engine: &dyn StorageEngine, txn: TxnId) -> bool {
    WRITER_COUNT.load(Ordering::Acquire) > 0 && writers().contains(&(engine_addr(engine), txn))
}

/// End `txn` with `end` (a commit, rollback or prepare). When `txn` wrote the partition catalog,
/// the shared tree is invalidated around it.
pub(super) fn around_end<T>(engine: &dyn StorageEngine, txn: TxnId, end: impl FnOnce() -> T) -> T {
    let registered =
        WRITER_COUNT.load(Ordering::Acquire) > 0 && writers().remove(&(engine_addr(engine), txn));
    if !registered {
        return end();
    }
    let out = invalidating(end);
    WRITER_COUNT.fetch_sub(1, Ordering::AcqRel);
    out
}

/// Run `end`, which may change the committed partition tree, with the shared tree invalidated
/// before and after it, and not served while it runs.
pub(super) fn invalidating<T>(end: impl FnOnce() -> T) -> T {
    TREE_INFLIGHT.fetch_add(1, Ordering::AcqRel);
    TREE_EPOCH.fetch_add(1, Ordering::AcqRel);
    let out = end();
    TREE_EPOCH.fetch_add(1, Ordering::AcqRel);
    TREE_INFLIGHT.fetch_sub(1, Ordering::AcqRel);
    out
}

/// A session or connection starts: no tree read before it is served again.
pub(super) fn session_started() {
    TREE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

/// The committed partition tree for the lock-time lookups: from this statement's cache, the
/// shared cache, or read.
fn cached_tree(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Arc<super::partition::CommittedTree>, Error> {
    if let Some(tree) = with_cache(engine, txn, |c| c.tree.clone()) {
        return Ok(tree);
    }
    let tree = shared_or_read(engine, txn)?;
    with_cache(engine, txn, |c| c.tree = Some(Arc::clone(&tree)));
    Ok(tree)
}

fn shared_or_read(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Arc<super::partition::CommittedTree>, Error> {
    if TREE_INFLIGHT.load(Ordering::Acquire) > 0 || is_writer(engine, txn) {
        return Ok(Arc::new(super::partition::committed_tree(engine, txn)?));
    }
    let addr = engine_addr(engine);
    let catalog = engine
        .lookup_table_as_of(txn, super::partition::PARTITION_CATALOG)?
        .map(|t| t.id);
    let epoch = TREE_EPOCH.load(Ordering::Acquire);
    if let Some((a, c, e, tree)) = shared_tree().as_ref()
        && *a == addr
        && *c == catalog
        && *e == epoch
    {
        return Ok(Arc::clone(tree));
    }
    let tree = Arc::new(super::partition::committed_tree(engine, txn)?);
    if TREE_INFLIGHT.load(Ordering::Acquire) == 0 && TREE_EPOCH.load(Ordering::Acquire) == epoch {
        *shared_tree() = Some((addr, catalog, epoch, Arc::clone(&tree)));
    }
    Ok(tree)
}

/// The global keys a write into `table` must lock and check: empty for a table that is not a
/// partition, and for a partition whose keys all cover the partition keys above it. Cached for
/// the statement.
///
/// # Errors
/// Storage errors.
pub(super) fn leaf_globals(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Rc<Vec<LeafGlobal>>, Error> {
    if let Some(found) = with_cache(engine, txn, |c| c.leaves.get(&table.id).map(Rc::clone)) {
        return Ok(found);
    }
    let tree = cached_tree(engine, txn)?;
    let mut out = Vec::new();
    let key = crate::analyzer::qualified_display(&table.schema, &table.name);
    if !tree.is_empty() && tree.parent(&key).is_some() {
        for c in engine.list_constraints_as_of(txn, table.id)? {
            if !is_unique(&c) {
                continue;
            }
            if let Some(scope) = scope_in(&tree, table, &c, engine, txn)? {
                let ordinals = dml::constraint_ordinals(table, &c.columns)?;
                out.push(LeafGlobal {
                    types: ordinals
                        .iter()
                        .filter_map(|&i| table.columns.get(i).map(|col| col.ty))
                        .collect(),
                    ordinals,
                    leaf_constraint: c.name,
                    scope: Rc::new(scope),
                });
            }
        }
    }
    let out = Rc::new(out);
    with_cache(engine, txn, |c| c.leaves.insert(table.id, Rc::clone(&out)));
    Ok(out)
}

// === Statement frames ====================================================================

/// The keys one statement wrote under global constraints, checked when it finishes.
struct Frame {
    engine: usize,
    txn: TxnId,
    pending: Vec<(Rc<Scope>, Key)>,
}

thread_local! {
    static FRAMES: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// Pops the frames above `depth` when dropped, so an unwinding statement leaves none behind.
struct FrameGuard {
    depth: usize,
}

impl Drop for FrameGuard {
    fn drop(&mut self) {
        FRAMES.with(|f| f.borrow_mut().truncate(self.depth));
    }
}

/// Run one statement (`body`), then check every global key it wrote. Statements nest (a function
/// or trigger body runs inside the statement that called it): each level checks its own keys.
///
/// # Errors
/// The statement's own error, or `23505` for a key now held by two rows of a scope.
pub(super) fn statement<T>(
    engine: &dyn StorageEngine,
    txn: TxnId,
    body: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    let depth = FRAMES.with(|f| {
        let mut frames = f.borrow_mut();
        frames.push(Frame {
            engine: engine_addr(engine),
            txn,
            pending: Vec::new(),
        });
        frames.len() - 1
    });
    let guard = FrameGuard { depth };
    let outcome = body();
    let frame = FRAMES.with(|f| {
        let mut frames = f.borrow_mut();
        if frames.len() > depth {
            frames.truncate(depth + 1);
            frames.pop()
        } else {
            None
        }
    });
    drop(guard);
    let value = outcome?;
    if let Some(frame) = frame
        && !frame.pending.is_empty()
    {
        check_pending(frame.pending, engine, txn)?;
    }
    Ok(value)
}

/// Remember `keys` written under `scope` for the end of the running statement. With no statement
/// running on this thread for this transaction (a write path that does not go through one), the
/// keys are queued for the transaction's commit instead: later, but never skipped.
pub(super) fn record(scope: &Rc<Scope>, keys: Vec<Key>, engine: &dyn StorageEngine, txn: TxnId) {
    if keys.is_empty() {
        return;
    }
    let addr = engine_addr(engine);
    let leftover = FRAMES.with(|f| {
        let mut frames = f.borrow_mut();
        match frames.last_mut() {
            Some(frame) if frame.engine == addr && frame.txn == txn => {
                frame
                    .pending
                    .extend(keys.into_iter().map(|k| (Rc::clone(scope), k)));
                None
            },
            _ => Some(keys),
        }
    });
    if let Some(keys) = leftover {
        defer(scope, keys, engine, txn);
    }
}

/// Queue `keys` of `scope` for the commit-time check (a deferred constraint).
pub(super) fn defer(scope: &Scope, keys: Vec<Key>, engine: &dyn StorageEngine, txn: TxnId) {
    super::deferred::defer_keys(
        engine,
        txn,
        &scope.deferral_ref(),
        keys.into_iter().map(super::deferred::Pending::Key),
    );
}

/// Check the keys a statement wrote, scope by scope.
fn check_pending(
    pending: Vec<(Rc<Scope>, Key)>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut groups: Vec<(Rc<Scope>, Vec<Key>)> = Vec::new();
    for (scope, key) in pending {
        match groups
            .iter_mut()
            .find(|(s, _)| s.table.id == scope.table.id && s.constraint == scope.constraint)
        {
            Some((_, keys)) => keys.push(key),
            None => groups.push((scope, vec![key])),
        }
    }
    for (scope, keys) in groups {
        check_keys(&scope, keys, engine, txn)?;
    }
    Ok(())
}

// === The check ===========================================================================

/// One partition of a scope: its table, the key's ordinals in it and its copy's backing index.
struct Leaf {
    table: TableSchema,
    ordinals: Vec<usize>,
    /// The key columns' types.
    types: Vec<ColumnType>,
    index: Option<nusadb_core::IndexId>,
}

/// One end of an index range.
pub(super) type KeyBound = std::ops::Bound<Vec<u8>>;

/// The most spellings of a key probed one by one on an index that may hold non-canonical entries;
/// past it, the rest of the key is covered by one range per spelling probed so far.
pub(super) const MAX_SPELLINGS: usize = 256;

/// The character right after the blank: a value followed by it sorts above the value followed by
/// any number of blanks.
pub(super) const AFTER_BLANK: char = '!';

/// The index ranges that hold every entry of `key` (its columns of `types`) on an index that may
/// hold `CHAR(n)` values with their trailing blanks (written by an earlier release, see
/// `executor::char_canon`): `'x'`, `'x '` and `'x  '` are one key held under three entries.
///
/// For a key with no such `CHAR` value this is the one exact entry. Otherwise each spelling (a
/// `CHAR` value with 0 up to its full width of trailing blanks) of the leading columns is probed
/// exactly, up to [`MAX_SPELLINGS`] combinations; the column where that would be exceeded, and
/// every column after it, are covered by one range per combination, from the rest of the key bare
/// to the rest with every `CHAR` value padded to its width (a blank sorts above the end-of-text
/// marker, so every spelling falls inside). A value with room for more than [`MAX_SPELLINGS`]
/// blanks is not padded: the range ends at it followed by the character after the blank instead,
/// so no range costs more than that many blanks per column however wide the column is. A `CHAR`
/// value ending the key is always covered by such a range, which ends where the key does. The
/// ranges are disjoint; the caller compares every row read with the key (a range may also hold
/// keys that extend the value with blanks and a character below the blank).
///
/// # Errors
/// A value the index cannot encode.
pub(super) fn probe_ranges(
    key: &[ast::Value],
    types: &[ColumnType],
) -> Result<Vec<(KeyBound, KeyBound)>, Error> {
    let slack: Vec<usize> = key
        .iter()
        .zip(types)
        .map(|(value, ty)| match (ty, value) {
            (ColumnType::Char(width), ast::Value::Text(text)) => usize::try_from(*width)
                .unwrap_or(usize::MAX)
                .saturating_sub(text.chars().count()),
            _ => 0,
        })
        .collect();
    let exact = |k: &[ast::Value]| -> Result<(KeyBound, KeyBound), Error> {
        let encoded = index_key::encode_index_key(k)?;
        Ok((
            std::ops::Bound::Included(encoded.clone()),
            std::ops::Bound::Included(encoded),
        ))
    };
    if slack.iter().all(|&s| s == 0) {
        return Ok(vec![exact(key)?]);
    }
    let pad = |k: &mut Key, i: usize, extra: usize| {
        if let Some(ast::Value::Text(text)) = k.get_mut(i) {
            text.push_str(&" ".repeat(extra));
        }
    };
    let last = key.len().saturating_sub(1);
    // `k` from column `from` on, every spelling covered: each CHAR value padded while that adds
    // at most `MAX_SPELLINGS` blanks, the first wider one followed by the character after the
    // blank (above the value followed by any run of blanks) and the columns after it left out.
    let ceiling = |k: &Key, from: usize| -> Key {
        let mut hi: Key = k.get(..from).unwrap_or(k).to_vec();
        for (i, value) in k.iter().enumerate().skip(from) {
            let extra = slack.get(i).copied().unwrap_or(0);
            let mut value = value.clone();
            if let ast::Value::Text(text) = &mut value
                && extra > 0
            {
                if extra > MAX_SPELLINGS {
                    text.push(AFTER_BLANK);
                    hi.push(value);
                    break;
                }
                text.push_str(&" ".repeat(extra));
            }
            hi.push(value);
        }
        hi
    };
    // The rest of the key after column `from`: exact when it holds no CHAR value with room for
    // blanks, else one range from it bare to its ceiling.
    let tail = |k: &Key, from: usize| -> Result<(KeyBound, KeyBound), Error> {
        if slack.iter().skip(from).all(|&s| s == 0) {
            return exact(k);
        }
        Ok((
            std::ops::Bound::Included(index_key::encode_index_key(k)?),
            std::ops::Bound::Included(index_key::encode_index_key(&ceiling(k, from))?),
        ))
    };
    // Spell out each column before the last, every combination, while they stay within the cap.
    let mut combos: Vec<Key> = vec![key.to_vec()];
    for (i, &extra) in slack.iter().enumerate().take(last) {
        if extra == 0 {
            continue;
        }
        let allowed = MAX_SPELLINGS / combos.len().max(1);
        if allowed > extra {
            combos = combos
                .into_iter()
                .flat_map(|k| {
                    (0..=extra).map(move |n| {
                        let mut out = k.clone();
                        if let Some(ast::Value::Text(text)) = out.get_mut(i) {
                            text.push_str(&" ".repeat(n));
                        }
                        out
                    })
                })
                .collect();
            continue;
        }
        // Past the cap at column `i`: its first `allowed` spellings exactly (the columns after it
        // covered as a tail), then one range over its remaining spellings and everything after.
        let allowed = allowed.max(1);
        let mut out = Vec::with_capacity(combos.len() * (allowed + 1));
        for k in &combos {
            for n in 0..allowed {
                let mut spelled = k.clone();
                pad(&mut spelled, i, n);
                out.push(tail(&spelled, i + 1)?);
            }
            let mut lo = k.clone();
            pad(&mut lo, i, allowed);
            out.push((
                std::ops::Bound::Included(index_key::encode_index_key(&lo)?),
                std::ops::Bound::Included(index_key::encode_index_key(&ceiling(k, i))?),
            ));
        }
        return Ok(out);
    }
    combos.iter().map(|k| tail(k, last)).collect()
}

/// Which indexes predate this release (`executor::char_canon`), cached for the statement.
///
/// # Errors
/// Storage errors reading the catalog.
pub(super) fn canon_state(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Arc<super::char_canon::CanonState>, Error> {
    if let Some(state) = with_cache(engine, txn, |c| c.canon.clone()) {
        return Ok(state);
    }
    let state = super::char_canon::state(engine, txn, TREE_EPOCH.load(Ordering::Acquire))?;
    with_cache(engine, txn, |c| c.canon = Some(Arc::clone(&state)));
    Ok(state)
}

/// Which rows hold each encoded key of a legacy unique index: one row's tid, or several.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Holders {
    One(Tid),
    Many,
}

/// The keys the rows of one table held in a legacy unique index when a statement last read it.
type LegacyKeys = HashMap<Vec<u8>, Holders>;

/// The key a row holds in an index (`None` when the index leaves the row out).
pub(super) type KeyOf<'a> = dyn Fn(&Row) -> Result<Option<Vec<u8>>, Error> + 'a;

/// Whether a row other than `own` holds `key` in legacy unique index `index` of `table`, for an
/// index whose entries cannot be probed for a key reliably: one an earlier release keyed on an
/// expression, or gave a predicate over a `CHAR(n)` column (`executor::char_canon`).
///
/// The caller has already probed the exact (canonical) entry, which finds every row written by
/// this build. For the older rows the table is read once per statement, streamed, into a map from
/// each row's key (as `key_of` computes it, `None` for a row the index leaves out) to the rows
/// holding it; later calls answer from the map, so a statement writing many rows reads the table
/// once, not once per row. A key the map gives to a row other than `own` is only a candidate (that
/// row may have changed since): `holds` confirms it against a fresh read of the table, which also
/// replaces the map. The map holds one entry per indexed row of the table.
///
/// # Errors
/// Storage errors, or the errors of `key_of` / `holds`.
#[allow(
    clippy::too_many_arguments,
    reason = "the index, its table, the row being written and the two row tests it needs"
)]
pub(super) fn legacy_key_held(
    engine: &dyn StorageEngine,
    txn: TxnId,
    index: nusadb_core::IndexId,
    table: TableId,
    types: &[ColumnType],
    key: &[u8],
    own: Option<Tid>,
    key_of: &KeyOf<'_>,
    holds: &dyn Fn(Tid, &Row) -> Result<bool, Error>,
) -> Result<bool, Error> {
    if let Some(known) = with_cache(engine, txn, |c| c.legacy_keys.get(&index).map(Rc::clone)) {
        match known.get(key) {
            None => return Ok(false),
            Some(Holders::One(tid)) if Some(*tid) == own => return Ok(false),
            Some(_) => {},
        }
    }
    // First ask, or a candidate to confirm: read the table once, rebuilding the map as it goes.
    let mut keys: LegacyKeys = HashMap::new();
    let mut held = false;
    let mut scan = engine.scan_committed(txn, table)?;
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        let row = row::decode(&tuple, types)?;
        if let Some(k) = key_of(&row)? {
            if !held && k.as_slice() == key && holds(tid, &row)? {
                held = true;
            }
            keys.entry(k)
                .and_modify(|h| *h = Holders::Many)
                .or_insert(Holders::One(tid));
        }
    }
    drop(scan);
    let keys = Rc::new(keys);
    with_cache(engine, txn, |c| c.legacy_keys.insert(index, keys));
    Ok(held)
}

/// The legacy set changed: this statement reads it again.
pub(super) fn invalidate_canon() {
    CACHE.with(|c| {
        if let Some(cache) = c.borrow_mut().as_mut() {
            cache.canon = None;
        }
    });
}

#[cfg(test)]
thread_local! {
    /// Index entries [`key_entries`] read on this thread, for the tests that bound a probe's work.
    pub(super) static ENTRIES_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Index ranges [`key_entries`] probed on this thread.
    pub(super) static RANGES_PROBED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Every index entry of `index` that may hold `key` (see [`probe_ranges`]), latest committed plus
/// `txn`'s own writes. The caller compares each row with the key.
///
/// # Errors
/// Storage errors, or a value the index cannot encode.
pub(super) fn key_entries(
    engine: &dyn StorageEngine,
    txn: TxnId,
    index: nusadb_core::IndexId,
    key: &[ast::Value],
    types: &[ColumnType],
) -> Result<Vec<(Tid, nusadb_core::SharedTuple)>, Error> {
    // An index this release built holds each key under its one stored spelling.
    let ranges = if canon_state(engine, txn)?.is_legacy(index) {
        probe_ranges(key, types)?
    } else {
        let encoded = index_key::encode_index_key(key)?;
        vec![(
            std::ops::Bound::Included(encoded.clone()),
            std::ops::Bound::Included(encoded),
        )]
    };
    #[cfg(test)]
    RANGES_PROBED.with(|n| n.set(n.get() + ranges.len()));
    let mut out = Vec::new();
    for (lo, hi) in ranges {
        crate::cancel::check()?;
        let mut scan = engine.index_scan_committed(txn, index, lo, hi)?;
        while let Some(entry) = scan.try_next()? {
            out.push(entry);
        }
    }
    #[cfg(test)]
    ENTRIES_READ.with(|n| n.set(n.get() + out.len()));
    Ok(out)
}

/// The partitions of `scope` that hold rows, listed from the committed catalog (see the module
/// docs).
fn leaves_of(scope: &Scope, engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Leaf>, Error> {
    let tree = super::partition::committed_tree(engine, txn)?;
    let root = crate::analyzer::qualified_display(&scope.table.schema, &scope.table.name);
    let mut out = Vec::new();
    for key in tree.leaves_below(&root) {
        let Some(table) = lookup(&key, engine, txn)? else {
            continue;
        };
        let copy = engine
            .list_constraints_as_of(txn, table.id)?
            .into_iter()
            .find(|c| is_unique(c) && c.columns == scope.columns);
        let index = match &copy {
            Some(c) => engine.lookup_index(&c.name)?,
            None => None,
        };
        let ordinals = dml::constraint_ordinals(&table, &scope.columns)?;
        out.push(Leaf {
            types: ordinals
                .iter()
                .filter_map(|&i| table.columns.get(i).map(|c| c.ty))
                .collect(),
            ordinals,
            table,
            index,
        });
    }
    Ok(out)
}

/// Whether reading `leaf` whole is cheaper than probing its index once per key.
fn read_whole(leaf: &Leaf, keys: &[Key], engine: &dyn StorageEngine) -> Result<bool, Error> {
    if leaf.index.is_none()
        || keys
            .iter()
            .any(|k| !k.iter().all(super::ops::is_hash_safe_value))
    {
        return Ok(true);
    }
    if keys.len() < 64 {
        return Ok(false);
    }
    let rows = engine.approx_row_count(leaf.table.id)?;
    Ok(u64::try_from(keys.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(4)
        >= rows)
}

/// For each of `keys` (sorted, distinct), how many live rows of `leaf` hold it.
fn count_holders(
    leaf: &Leaf,
    keys: &[Key],
    holders: &mut [usize],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if read_whole(leaf, keys, engine)? {
        for (_, row) in scan_table_committed(&leaf.table, engine, txn)? {
            if let Some(k) = dml::unique_key(&row, &leaf.ordinals, false)
                && let Ok(i) = keys.binary_search_by(|p| dml::unique_key_cmp(p, &k))
                && let Some(h) = holders.get_mut(i)
            {
                *h += 1;
            }
        }
        return Ok(());
    }
    let Some(index) = leaf.index else {
        return Ok(());
    };
    let types = column_types(&leaf.table);
    for (key, h) in keys.iter().zip(holders.iter_mut()) {
        let mut seen: HashSet<Tid> = HashSet::new();
        for (tid, tuple) in key_entries(engine, txn, index, key, &leaf.types)? {
            crate::cancel::check()?;
            let row = row::decode(&tuple, &types)?;
            if dml::unique_key(&row, &leaf.ordinals, false)
                .is_some_and(|k| dml::unique_key_eq(&k, key))
                && seen.insert(tid)
            {
                *h += 1;
            }
        }
    }
    Ok(())
}

/// Sort and deduplicate keys under the executor's total order.
fn distinct(mut keys: Vec<Key>) -> Vec<Key> {
    keys.sort_by(|a, b| dml::unique_key_cmp(a, b));
    keys.dedup_by(|a, b| dml::unique_key_cmp(a, b) == std::cmp::Ordering::Equal);
    keys
}

/// No two live rows of `scope`'s partitions may hold one of `keys`. The caller holds the keys'
/// scope locks.
///
/// # Errors
/// `23505` naming the scope's constraint.
pub(super) fn check_keys(
    scope: &Scope,
    keys: Vec<Key>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let keys = distinct(keys);
    if keys.is_empty() {
        return Ok(());
    }
    let mut holders = vec![0_usize; keys.len()];
    for leaf in leaves_of(scope, engine, txn)? {
        count_holders(&leaf, &keys, &mut holders, engine, txn)?;
        if holders.iter().any(|&h| h > 1) {
            return Err(scope.violation());
        }
    }
    Ok(())
}

/// No two live rows of `scope`'s partitions may share a key: one read of every partition (adding
/// a key to a populated table, a deferred check with too many keys to keep).
///
/// # Errors
/// `23505` naming the scope's constraint.
pub(super) fn check_whole(
    scope: &Scope,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut all: Vec<Key> = Vec::new();
    for leaf in leaves_of(scope, engine, txn)? {
        all.extend(
            scan_table_committed(&leaf.table, engine, txn)?
                .into_iter()
                .filter_map(|(_, row)| dml::unique_key(&row, &leaf.ordinals, false)),
        );
    }
    all.sort_by(|a, b| dml::unique_key_cmp(a, b));
    if all
        .windows(2)
        .any(|w| matches!(w, [a, b] if dml::unique_key_cmp(a, b) == std::cmp::Ordering::Equal))
    {
        return Err(scope.violation());
    }
    Ok(())
}

/// The partitions of a scope, listed once for many lookups. List them after locking the keys to
/// be looked up, so a partition another transaction created and filled with one of them before
/// the lock is listed.
pub(super) struct ScopeLeaves(Vec<Leaf>);

impl ScopeLeaves {
    /// List `scope`'s partitions now.
    ///
    /// # Errors
    /// Storage errors.
    pub(super) fn load(
        scope: &Scope,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<Self, Error> {
        Ok(Self(leaves_of(scope, engine, txn)?))
    }

    /// A live row holding `key`: its partition and tid.
    ///
    /// # Errors
    /// Storage errors.
    pub(super) fn holder(
        &self,
        key: &[ast::Value],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<Option<(TableSchema, Tid)>, Error> {
        let matches = |leaf: &Leaf, row: &Row| {
            dml::unique_key(row, &leaf.ordinals, false).is_some_and(|k| dml::unique_key_eq(&k, key))
        };
        for leaf in &self.0 {
            match leaf.index {
                Some(index) if key.iter().all(super::ops::is_hash_safe_value) => {
                    let types = column_types(&leaf.table);
                    for (tid, tuple) in key_entries(engine, txn, index, key, &leaf.types)? {
                        if matches(leaf, &row::decode(&tuple, &types)?) {
                            return Ok(Some((leaf.table.clone(), tid)));
                        }
                    }
                },
                _ => {
                    for (tid, row) in scan_table_committed(&leaf.table, engine, txn)? {
                        if matches(leaf, &row) {
                            return Ok(Some((leaf.table.clone(), tid)));
                        }
                    }
                },
            }
        }
        Ok(None)
    }
}

/// The commit-time check of a deferred global key (`executor::deferred`): `keys`, or with `whole`
/// every key of the scope. A scope whose table or constraint is gone is skipped.
///
/// # Errors
/// `23505` naming the scope's constraint.
pub(super) fn check_deferred(
    table: TableId,
    constraint: &str,
    keys: Vec<Key>,
    whole: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(schema) = engine.lookup_table_by_id_as_of(txn, table)? else {
        return Ok(());
    };
    let Some(c) = engine
        .list_constraints_as_of(txn, table)?
        .into_iter()
        .find(|c| is_unique(c) && c.name == constraint)
    else {
        return Ok(());
    };
    let scope = Scope {
        table: schema,
        constraint: c.name,
        kind: c.kind,
        columns: c.columns,
    };
    if whole {
        return check_whole(&scope, engine, txn);
    }
    check_keys(&scope, keys, engine, txn)
}

/// Whether the unique constraint `name` on `table` still exists (a deferred entry of a dropped one
/// is skipped).
///
/// # Errors
/// Storage errors.
pub(super) fn constraint_exists(
    table: TableId,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    Ok(engine
        .list_constraints_as_of(txn, table)?
        .iter()
        .any(|c| is_unique(c) && c.name == name))
}

// === DDL =================================================================================

/// The notice a key raises when it becomes global.
pub(super) fn notice_created(constraint: &str, table: &str) {
    crate::notice::notice(format!(
        "unique constraint \"{constraint}\" on partitioned table \"{table}\" does not include \
         every partitioning column, so it is enforced across all of its partitions: every write \
         of its key probes every partition"
    ));
}

/// Refuse a key that would be global but whose checks this release cannot run soundly:
/// `NULLS NOT DISTINCT`, and a floating-point or `NUMERIC` key column (equal values can differ in
/// their index encoding).
///
/// # Errors
/// `0A000`.
pub(super) fn refuse_unsupported(
    table: &TableSchema,
    name: &str,
    columns: &[String],
    nulls_not_distinct: bool,
) -> Result<(), Error> {
    if nulls_not_distinct {
        return Err(Error::Coded {
            message: format!(
                "unique constraint \"{name}\" on partitioned table \"{}\" leaves out a \
                 partitioning column, which is not supported with NULLS NOT DISTINCT",
                table.name
            ),
            sqlstate: "0A000",
        });
    }
    for column in columns {
        if let Some(def) = table.columns.iter().find(|c| &c.name == column)
            && let Some(why) = unsupported_key_type(def.ty)
        {
            return Err(Error::Coded {
                message: format!(
                    "unique constraint \"{name}\" on partitioned table \"{}\" leaves out a \
                     partitioning column, which is not supported for the {why} column \"{column}\"",
                    table.name
                ),
                sqlstate: "0A000",
            });
        }
    }
    Ok(())
}

/// Why a column of type `ty` cannot be part of a key enforced across partitions, or `None` when it
/// can: the check probes each partition's backing index, so every key value must have an index
/// encoding whose bytes are equal exactly when the values are. A floating-point or `NUMERIC` value
/// can compare equal under two encodings, and the other types listed have no index encoding at all.
const fn unsupported_key_type(ty: ColumnType) -> Option<&'static str> {
    match ty {
        ColumnType::Bool
        | ColumnType::Int
        | ColumnType::SmallInt
        | ColumnType::BigInt
        | ColumnType::Text
        | ColumnType::VarChar(_)
        | ColumnType::Char(_)
        | ColumnType::Bytes
        | ColumnType::Date
        | ColumnType::Time
        | ColumnType::TimeTz
        | ColumnType::Timestamp
        | ColumnType::TimestampTz
        | ColumnType::Uuid
        | ColumnType::Macaddr
        | ColumnType::Macaddr8
        | ColumnType::Json
        | ColumnType::Jsonb
        | ColumnType::Enum => None,
        ColumnType::Float | ColumnType::Real | ColumnType::Numeric { .. } => {
            Some("floating-point or NUMERIC")
        },
        ColumnType::Interval => Some("INTERVAL"),
        ColumnType::Array(_) => Some("array"),
        ColumnType::Vector(_) => Some("VECTOR"),
        ColumnType::Inet | ColumnType::Cidr => Some("INET or CIDR"),
        ColumnType::Bit(_) | ColumnType::VarBit(_) => Some("BIT"),
        ColumnType::Range(_) => Some("range"),
        ColumnType::Geometry(_) => Some("geometry"),
        ColumnType::Tsvector | ColumnType::Tsquery => Some("full-text"),
        ColumnType::Xml => Some("XML"),
    }
}

/// After `ATTACH PARTITION`: the keys of the attached table's partitions that are global must not
/// meet an equal key elsewhere in their scope. Every such key is locked in its scope first, so a
/// writer of an equal key elsewhere either already holds it (the attach fails with `40001`) or
/// has committed (the check sees its row).
///
/// # Errors
/// `40001`, or `23505` naming the scope's constraint.
pub(super) fn validate_attached(
    attached: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    new_statement();
    let tree = super::partition::committed_tree(engine, txn)?;
    let key = crate::analyzer::qualified_display(&attached.schema, &attached.name);
    let leaves = if tree.key_parts(&key).is_none() {
        vec![key]
    } else {
        tree.leaves_below(&key)
    };
    for leaf_key in leaves {
        let Some(leaf) = lookup(&leaf_key, engine, txn)? else {
            continue;
        };
        for g in leaf_globals(&leaf, engine, txn)?.iter() {
            let keys: Vec<Key> = scan_table_committed(&leaf, engine, txn)?
                .into_iter()
                .filter_map(|(_, row)| g.key_of(&row).transpose())
                .collect::<Result<_, _>>()?;
            for k in &keys {
                g.scope.lock(k, engine, txn)?;
            }
            check_keys(&g.scope, keys, engine, txn)?;
        }
    }
    Ok(())
}

/// After `ALTER TABLE ... ADD PRIMARY KEY / UNIQUE` on a partitioned table, once every partition
/// holds its copy: when the key is enforced across partitions anywhere below the table, no two
/// rows of the table's partitions may share a key. The caller holds every partition exclusively
/// and reads fresh.
///
/// # Errors
/// `23505` naming the constraint.
pub(super) fn validate_added(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    new_statement();
    let tree = super::partition::committed_tree(engine, txn)?;
    let key = crate::analyzer::qualified_display(&table.schema, &table.name);
    if tree.key_parts(&key).is_none() {
        return Ok(());
    }
    let Some(c) = engine
        .list_constraints_as_of(txn, table.id)?
        .into_iter()
        .find(|c| is_unique(c) && c.name == name)
    else {
        return Ok(());
    };
    let mut global = false;
    for leaf_key in tree.leaves_below(&key) {
        if let Some(leaf) = lookup(&leaf_key, engine, txn)?
            && leaf_globals(&leaf, engine, txn)?
                .iter()
                .any(|g| g.scope.columns == c.columns)
        {
            global = true;
            break;
        }
    }
    if !global {
        return Ok(());
    }
    let scope = Scope {
        table: table.clone(),
        constraint: c.name,
        kind: c.kind,
        columns: c.columns,
    };
    check_whole(&scope, engine, txn)
}

#[cfg(test)]
mod tests {
    use super::super::partition::KeyPart;
    use super::*;

    fn cols(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    #[test]
    fn a_key_covers_only_plain_partition_columns_it_includes() {
        let by_d = [KeyPart::Column("d".to_owned())];
        assert!(covers(&by_d, &cols(&["id", "d"])));
        assert!(!covers(&by_d, &cols(&["id"])));
        let by_two = [
            KeyPart::Column("d".to_owned()),
            KeyPart::Column("r".to_owned()),
        ];
        assert!(!covers(&by_two, &cols(&["d"])));
        assert!(covers(&by_two, &cols(&["r", "d", "id"])));
        let by_expr = [KeyPart::Expression("(k % 10)".to_owned())];
        assert!(!covers(&by_expr, &cols(&["k"])));
    }

    #[test]
    fn a_char_key_is_probed_one_spelling_at_a_time_unless_it_ends_the_key() {
        let text = |s: &str| ast::Value::Text(s.to_owned());
        let is_point = |(lo, hi): &(KeyBound, KeyBound)| lo == hi;
        // No CHAR value with room for blanks: the one exact entry.
        let r = probe_ranges(&[ast::Value::Int(1)], &[ColumnType::Int]).unwrap();
        assert!(r.len() == 1 && r.iter().all(is_point));
        let r = probe_ranges(&[text("abcd")], &[ColumnType::Char(4)]).unwrap();
        assert!(r.len() == 1 && r.iter().all(is_point));
        // A CHAR value ending the key: one range over its spellings.
        let r = probe_ranges(
            &[ast::Value::Int(5), text("a")],
            &[ColumnType::Int, ColumnType::Char(8)],
        )
        .unwrap();
        assert!(r.len() == 1 && !is_point(&r[0]));
        // A CHAR value followed by another column: each spelling exactly, never a range over the
        // later columns.
        let r = probe_ranges(
            &[text("a"), ast::Value::Int(5)],
            &[ColumnType::Char(8), ColumnType::Int],
        )
        .unwrap();
        assert_eq!(r.len(), 8);
        assert!(r.iter().all(is_point));
        // Two of them: every combination.
        let r = probe_ranges(
            &[text("a"), text("b"), ast::Value::Int(5)],
            &[ColumnType::Char(3), ColumnType::Char(2), ColumnType::Int],
        )
        .unwrap();
        assert_eq!(r.len(), 3 * 2);
        assert!(r.iter().all(is_point));
        // Past the cap: as many spellings as fit, exactly, then one range over the rest.
        let r = probe_ranges(
            &[text("a"), ast::Value::Int(5)],
            &[ColumnType::Char(1000), ColumnType::Int],
        )
        .unwrap();
        assert_eq!(r.len(), MAX_SPELLINGS + 1);
        assert_eq!(r.iter().filter(|r| is_point(r)).count(), MAX_SPELLINGS);
        // A column as wide as the type allows: still at most MAX_SPELLINGS + 1 ranges, and no
        // bound holds the padded value (each costs at most MAX_SPELLINGS blanks).
        let size = |r: &[(KeyBound, KeyBound)]| -> usize {
            r.iter()
                .map(|(lo, hi)| {
                    let len = |b: &KeyBound| match b {
                        std::ops::Bound::Included(v) | std::ops::Bound::Excluded(v) => v.len(),
                        std::ops::Bound::Unbounded => 0,
                    };
                    len(lo) + len(hi)
                })
                .sum()
        };
        for types in [
            vec![ColumnType::Char(10_485_760)],
            vec![ColumnType::Char(10_485_760), ColumnType::Char(10_485_760)],
            vec![ColumnType::Char(3), ColumnType::Char(10_485_760)],
        ] {
            let key: Vec<ast::Value> = types.iter().map(|_| text("a")).collect();
            let r = probe_ranges(&key, &types).unwrap();
            assert!(r.len() <= MAX_SPELLINGS + 1, "{types:?}: {}", r.len());
            assert!(size(&r) < 300 * 1024, "{types:?}: {} bytes", size(&r));
        }
    }

    /// The uniqueness probe of a key `(CHAR, INT)` reads only the entries of that key, however
    /// many rows share its CHAR value: a load of `('a', m)` stays linear.
    #[test]
    fn a_composite_char_key_probe_reads_only_its_own_entries() {
        use crate::{Catalog, IndexInfo, Session, analyze, parse, plan};
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
        let engine = nusadb_btree::BtreeEngine::new();
        let mut s = Session::new(&engine);
        let mut run = |sql: &str| {
            let logical = analyze(parse(sql).unwrap(), &Cat(&engine, s.current_txn())).unwrap();
            s.execute(plan(logical)).map(|_| ())
        };
        run("CREATE TABLE ca (k CHAR(8), m INT, UNIQUE (k, m))").unwrap();
        // Descending batches, so every probe lands below the entries already there.
        for batch in (0..20).rev() {
            let values: Vec<String> = (0..100)
                .rev()
                .map(|i| format!("('a', {})", batch * 100 + i))
                .collect();
            run(&format!("INSERT INTO ca VALUES {}", values.join(", "))).unwrap();
        }
        ENTRIES_READ.with(|n| n.set(0));
        run("INSERT INTO ca VALUES ('a', -1)").unwrap();
        assert_eq!(ENTRIES_READ.with(std::cell::Cell::get), 0);
        let e = run("INSERT INTO ca VALUES ('a  ', 7)").unwrap_err();
        assert!(e.to_string().contains("duplicate key"), "{e}");
        // One entry holds the key; nothing else sharing 'a' is read.
        assert_eq!(ENTRIES_READ.with(std::cell::Cell::get), 1);
    }

    /// A `CHAR(n)` key written by an earlier build: the row is stored without its trailing blanks
    /// but its index entry kept them. Every write path must still find it as the same key.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one scenario: the old-format rows, then every write path against them"
    )]
    fn a_char_key_indexed_with_its_blanks_still_conflicts() {
        use crate::{Catalog, IndexInfo, Session, analyze, parse, plan};
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
        fn run(engine: &dyn StorageEngine, s: &mut Session, sql: &str) -> Result<(), Error> {
            let logical = analyze(parse(sql)?, &Cat(engine, s.current_txn()))?;
            s.execute(plan(logical)).map(|_| ())
        }
        fn count(engine: &dyn StorageEngine, table: &str) -> usize {
            let txn = engine.begin(IsolationLevel::default()).unwrap();
            let t = engine.lookup_table_as_of(txn, table).unwrap().unwrap();
            let n = scan_rows(&t, engine, txn).unwrap().len();
            super::super::deferred::rollback(engine, txn).unwrap();
            n
        }
        let engine = nusadb_btree::BtreeEngine::new();
        let mut s = Session::new(&engine);
        for sql in [
            "CREATE TABLE ou (k CHAR(4) UNIQUE, n INT)",
            "CREATE TABLE opk (k CHAR(4) PRIMARY KEY, n INT)",
            "CREATE TABLE ou2 (k CHAR(4), m INT, UNIQUE (k, m))",
            "CREATE TABLE od (k CHAR(4), n INT, CONSTRAINT od_k UNIQUE (k) DEFERRABLE INITIALLY DEFERRED)",
            "INSERT INTO ou VALUES ('b', 2)",
        ] {
            run(&engine, &mut s, sql).unwrap();
        }
        // The tables stand for ones an earlier release built.
        super::super::char_canon::reset_to_legacy(&engine);
        // Write ('a  ', ...) the way an earlier build did: the row through the row encoding (which
        // drops the blanks), the index entry from the value as given.
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        for (table, index, row) in [
            (
                "ou",
                "ou_k_key",
                vec![ast::Value::Text("a  ".into()), ast::Value::Int(1)],
            ),
            (
                "opk",
                "opk_pkey",
                vec![ast::Value::Text("a  ".into()), ast::Value::Int(1)],
            ),
            (
                "ou2",
                "ou2_k_m_key",
                vec![ast::Value::Text("a  ".into()), ast::Value::Int(1)],
            ),
            (
                "od",
                "od_k",
                vec![ast::Value::Text("a  ".into()), ast::Value::Int(1)],
            ),
        ] {
            let t = engine.lookup_table_as_of(txn, table).unwrap().unwrap();
            let tid = engine
                .insert(txn, t.id, &row::encode(&row, &column_types(&t)).unwrap())
                .unwrap();
            let key: Vec<ast::Value> = if table == "ou2" {
                row.clone()
            } else {
                row[..1].to_vec()
            };
            let id = engine.lookup_index(index).unwrap().unwrap();
            engine
                .index_insert(txn, id, &index_key::encode_index_key(&key).unwrap(), tid)
                .unwrap();
        }
        super::super::deferred::commit(&engine, txn).unwrap();

        let refused = |s: &mut Session, sql: &str| {
            let e = run(&engine, s, sql).expect_err(sql);
            assert!(e.to_string().contains("duplicate key"), "{sql}: {e}");
        };
        refused(&mut s, "INSERT INTO ou VALUES ('a', 10)");
        refused(&mut s, "INSERT INTO ou VALUES ('a  ', 10)");
        refused(&mut s, "UPDATE ou SET k = 'a' WHERE n = 2");
        refused(&mut s, "UPDATE ou SET k = 'a   ' WHERE n = 2");
        refused(&mut s, "INSERT INTO opk VALUES ('a', 10)");
        refused(&mut s, "INSERT INTO ou2 VALUES ('a', 1)");
        let crate::ast::Statement::Copy(copy) = parse("COPY ou FROM STDIN").unwrap() else {
            panic!("not a COPY");
        };
        let e = crate::copy_from(&engine, &copy, "a \t30\n").expect_err("copy");
        assert!(e.to_string().contains("duplicate key"), "{e}");
        // The upsert finds the old row as its conflict and updates it, adding none.
        run(
            &engine,
            &mut s,
            "INSERT INTO ou VALUES ('a', 40) ON CONFLICT (k) DO UPDATE SET n = 41",
        )
        .unwrap();
        run(
            &engine,
            &mut s,
            "INSERT INTO ou VALUES ('a ', 50) ON CONFLICT DO NOTHING",
        )
        .unwrap();
        assert_eq!(count(&engine, "ou"), 2);
        // A deferred key is checked at commit through the same range.
        run(&engine, &mut s, "BEGIN").unwrap();
        run(&engine, &mut s, "INSERT INTO od VALUES ('a', 2)").unwrap();
        let e = run(&engine, &mut s, "COMMIT").expect_err("commit");
        assert!(e.to_string().contains("duplicate key"), "{e}");
        assert_eq!(count(&engine, "od"), 1);
    }

    /// Statements nest: an inner statement's keys are checked at its end and never leak into the
    /// outer frame, and an unwinding statement leaves no frame behind.
    #[test]
    fn frames_nest_and_unwind() {
        let engine = nusadb_btree::BtreeEngine::new();
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        let depth = || FRAMES.with(|f| f.borrow().len());
        let scope = Rc::new(Scope {
            table: TableSchema {
                id: TableId(1),
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: Vec::new(),
            },
            constraint: "t_k".to_owned(),
            kind: ConstraintKind::Unique,
            columns: cols(&["k"]),
        });
        let out = statement(&engine, txn, || {
            assert_eq!(depth(), 1);
            statement(&engine, txn, || {
                assert_eq!(depth(), 2);
                Ok::<(), Error>(())
            })?;
            record(&scope, Vec::new(), &engine, txn);
            assert_eq!(depth(), 1);
            Ok::<u8, Error>(7)
        });
        assert_eq!(out.unwrap(), 7);
        assert_eq!(depth(), 0);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = statement(&engine, txn, || -> Result<(), Error> { panic!("boom") });
        }));
        assert!(unwound.is_err());
        assert_eq!(depth(), 0);
    }
}
