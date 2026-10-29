//! Keeping uncommitted DDL out of other transactions' catalog reads.
//!
//! The catalog maps are changed in place when a DDL statement runs and put back by its undo
//! when the transaction rolls back. Read as they are, they would show every transaction the
//! tables, columns and indexes another transaction has changed and not committed: a second
//! session would plan against a column that does not exist for it yet, decode committed rows
//! under a layout they were not written in, or scan an index whose entries it cannot see.
//!
//! So each DDL call also records what it changed in [`UncommittedDdl`]: the committed state of
//! every table a running transaction created, altered, renamed or dropped, and the indexes it
//! created. Resolution for any other transaction then answers from that committed state:
//!
//! - a table another transaction created is absent;
//! - a table another transaction altered or renamed keeps its committed schema and name;
//! - a table another transaction dropped still resolves, and reading or writing it fails with
//!   a serialization conflict (its rows are held by the dropping transaction's undo);
//! - an index another transaction created is not offered to the planner;
//! - a table whose constraints another transaction added or dropped keeps its committed
//!   constraints, and a foreign key another transaction added is not enforced.
//!
//! The transaction that made the change sees it, and everyone sees it once that transaction has
//! committed. Whether an owner is still running is read from the active set, which a
//! transaction leaves only after its commit is durable or its undo has run, so an entry whose
//! owner has left it describes a state the live maps already hold and is dropped lazily.
//!
//! Every table DDL takes the table's exclusive lock until the transaction ends, so at most one
//! running transaction has an entry for a given table.
//!
//! # Schema changes and snapshots
//!
//! Rows are decoded by the SQL layer under the schema it resolved, and a change that rewrites
//! the rows (adding, dropping or retyping a column) or builds an index gives committed rows a
//! layout, or an index a content, that only snapshots taken after it match. Resolution reads the
//! latest committed schema, while a read uses the statement's snapshot, so the two can disagree:
//! a `REPEATABLE READ` snapshot older than the change, or a `READ COMMITTED` statement that
//! resolved the table just before the change committed and took its snapshot just after. For each
//! table such a change was made to, the transactions that made the recent ones are kept in order
//! ([`UncommittedDdl::history`]), together with which of them each transaction's last resolution
//! of the table reflected. A read whose snapshot sees a different set of them than its resolution
//! did fails with a serialization conflict instead of decoding rows under the wrong layout or
//! scanning an index the snapshot cannot use. An entry goes once every running transaction's
//! snapshot sees it, and a rolled-back change removes its own.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use nusadb_core::engine::IndexDef;
use nusadb_core::{Constraint, Error, ForeignKeyDef, Result, TableId, TableSchema, TxnId};

use super::{BtreeEngine, Catalog, constraints_of, foreign_keys_of, poisoned, table_not_found};

/// The catalog changes made by transactions that have not ended. Lives in [`Catalog`]; recorded
/// under the catalog write guard by the DDL call that makes the change.
#[derive(Debug, Default)]
pub(super) struct UncommittedDdl {
    tables: Vec<TableChange>,
    indexes: Vec<IndexChange>,
    constraints: Vec<ConstraintChange>,
    /// Foreign keys a running transaction added, by name, with that transaction.
    foreign_keys: Vec<(String, u64)>,
    /// Per table, oldest first, the transactions whose change rewrote its rows or built an index
    /// on it, while some running transaction's snapshot may not see that change.
    history: HashMap<u64, Vec<u64>>,
    /// `(transaction, table)`: the resolutions of the table the transaction's current and
    /// previous statement made (see [`Resolution`]).
    resolved: HashMap<(u64, u64), Vec<Resolution>>,
    /// Reads and resolutions since the last [`collect`](Self::collect) on their path: they
    /// collect only now and then, which is enough to bound what is kept.
    ticks: u32,
    /// The transactions that own an entry of `tables`, `indexes`, `constraints` or
    /// `foreign_keys`, and the `(table, owner)` pairs of `tables` and of `constraints`. They let
    /// the per-call checks look the set up instead of walking it, so one transaction dropping
    /// thousands of tables does not pay for every earlier drop on each later call.
    owners: HashSet<u64>,
    table_keys: HashSet<(u64, u64)>,
    constraint_keys: HashSet<(u64, u64)>,
}

impl UncommittedDdl {
    /// Whether no running transaction's change is recorded, so name resolution masks nothing.
    const fn no_pending_ddl(&self) -> bool {
        self.tables.is_empty()
            && self.indexes.is_empty()
            && self.constraints.is_empty()
            && self.foreign_keys.is_empty()
    }

    /// Whether nothing is recorded, so resolution may skip this state altogether.
    fn is_empty(&self) -> bool {
        self.no_pending_ddl() && self.history.is_empty() && self.resolved.is_empty()
    }

    /// Mirror what is recorded into the catalog's lock-free flags. Called with the set locked, so
    /// a flag never claims less than the set holds.
    fn publish(&self, cat: &Catalog) {
        cat.has_pending_ddl
            .store(!self.no_pending_ddl(), Ordering::Release);
        cat.has_uncommitted
            .store(!self.is_empty(), Ordering::Release);
    }

    /// Drop the changes of transactions that have ended: they are in the live catalog now, or were
    /// undone, and must not mask it.
    fn drop_ended_changes(&mut self, txns: &super::TxnDomain) {
        let active = &txns.active;
        if self.owners.iter().all(|owner| active.contains(owner)) {
            return;
        }
        self.tables.retain(|c| active.contains(&c.owner));
        self.indexes.retain(|c| active.contains(&c.owner));
        self.constraints.retain(|c| active.contains(&c.owner));
        self.foreign_keys
            .retain(|(_, owner)| active.contains(owner));
        self.owners.retain(|owner| active.contains(owner));
        self.table_keys.retain(|(_, owner)| active.contains(owner));
        self.constraint_keys
            .retain(|(_, owner)| active.contains(owner));
    }

    /// Whether every entry belongs to `txn`, so nothing is hidden from it.
    fn only_owner_is(&self, txn: u64) -> bool {
        self.owners.iter().all(|&owner| owner == txn)
    }

    /// [`collect`](Self::collect) on every 32nd call: the read and resolution paths run per
    /// statement, and nothing they check depends on collecting first.
    fn collect_now_and_then(&mut self, txns: &super::TxnDomain) {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks.is_multiple_of(32) {
            self.collect(txns);
        }
    }

    /// Drop what no longer matters: the changes of transactions that have ended, the
    /// resolutions of transactions that have ended or of their older statements, and the oldest
    /// history entries once every running snapshot sees them and every running transaction's
    /// resolutions reflect them.
    fn collect(&mut self, txns: &super::TxnDomain) {
        self.drop_ended_changes(txns);
        let active = &txns.active;
        self.resolved.retain(|(txn, _), all| {
            txns.txns.get(txn).is_some_and(|state| {
                all.retain(|r| r.statement.saturating_add(1) >= state.statements);
                !all.is_empty()
            })
        });
        let seen_by_all = |owner: u64| {
            !active.contains(&owner)
                && active
                    .iter()
                    .all(|a| txns.txns.get(a).is_none_or(|s| s.pinned.sees(owner)))
        };
        let resolved = &self.resolved;
        for (table, owners) in &mut self.history {
            // An entry may go only while no running transaction resolved the table without it:
            // its read still has to be checked against it.
            let reflected_by_all = |position: usize| {
                resolved.iter().all(|(&(other, t), all)| {
                    t != *table
                        || all.iter().all(|r| {
                            r.reflected
                                .and_then(|r| owners.iter().position(|&o| o == r))
                                .is_some_and(|at| at >= position)
                                || owners.get(position).is_some_and(|&o| o == other)
                        })
                })
            };
            let settled = owners
                .iter()
                .enumerate()
                .take_while(|&(position, &o)| seen_by_all(o) && reflected_by_all(position))
                .count();
            owners.drain(..settled);
        }
        self.history.retain(|_, owners| !owners.is_empty());
    }
}

/// One resolution of a table by a transaction: the newest entry of the table's history it
/// reflected (`None`: none of them), and the transaction's statement count when it was made.
///
/// A statement is resolved before it runs, so its resolutions carry the count of the statement
/// before it; one that started with a [`begin_statement`](nusadb_core::StorageEngine::begin_statement)
/// before resolving gets a count of its own. A read checks the resolutions of the current and the
/// previous count, which cover every resolution of the statement making it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Resolution {
    statement: u64,
    reflected: Option<u64>,
    /// For a resolution assumed when another transaction started a change (see
    /// [`BtreeEngine::note_layout_change`]), that transaction: it goes if the change rolls back.
    assumed_for: Option<u64>,
}

/// The resolutions in `all` that a read at statement count `statement` must agree with.
fn current(all: &[Resolution], statement: u64) -> impl Iterator<Item = &Resolution> {
    all.iter()
        .filter(move |r| r.statement.saturating_add(1) >= statement)
}

/// The newest of `owners` that `reflected` accepts.
fn newest(owners: &[u64], reflected: impl Fn(u64) -> bool) -> Option<u64> {
    owners.iter().rev().copied().find(|&o| reflected(o))
}

/// The constraints of one table before a running transaction first added or dropped one.
#[derive(Debug)]
struct ConstraintChange {
    table: u64,
    owner: u64,
    constraints: Vec<Constraint>,
    foreign_keys: Vec<ForeignKeyDef>,
}

/// One table a running transaction changed.
#[derive(Debug)]
struct TableChange {
    table: u64,
    owner: u64,
    /// The committed schema before `owner` first changed the table, or `None` when `owner`
    /// created it.
    before: Option<TableSchema>,
}

/// One index a running transaction created.
#[derive(Debug)]
struct IndexChange {
    index: u64,
    owner: u64,
}

/// What one transaction must not see: the changes of the other transactions still running.
pub(super) struct Hidden {
    /// Each table such a transaction changed, with its committed schema (`None`: not committed).
    tables: Vec<(u64, Option<TableSchema>)>,
    indexes: HashSet<u64>,
    /// Each table whose constraints such a transaction changed, with the committed ones.
    constraints: Vec<(u64, Vec<Constraint>, Vec<ForeignKeyDef>)>,
    /// The foreign keys such a transaction added.
    foreign_keys: HashSet<String>,
    /// Those transactions.
    owners: HashSet<u64>,
}

/// The committed state of a table another running transaction has changed.
enum Committed<'a> {
    /// That transaction created the table: it has no committed state.
    Absent,
    /// The table as last committed.
    Was(&'a TableSchema),
}

impl Committed<'_> {
    fn schema(&self) -> Option<TableSchema> {
        match self {
            Self::Absent => None,
            Self::Was(schema) => Some((*schema).clone()),
        }
    }
}

impl Hidden {
    /// The committed state of `table` when another running transaction changed it, `None` when
    /// nobody else has it changed.
    fn committed(&self, table: u64) -> Option<Committed<'_>> {
        self.tables
            .iter()
            .find(|(id, _)| *id == table)
            .map(|(_, before)| before.as_ref().map_or(Committed::Absent, Committed::Was))
    }

    /// The committed table known as `(schema, name)` that another running transaction has since
    /// renamed or dropped.
    fn by_committed_name(&self, schema: &str, name: &str) -> Option<&TableSchema> {
        self.tables.iter().find_map(|(_, before)| {
            before
                .as_ref()
                .filter(|b| b.schema == schema && b.name == name)
        })
    }
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard of the pending set is held for exactly the decision it backs: releasing \
              it earlier would let a DDL or a commit change the set between the read and the \
              write that depends on it"
)]
impl BtreeEngine {
    /// Record that `txn` is changing `table`, whose state before the change is `before` (`None`
    /// when `txn` is creating it). Only the first change per transaction is kept: later ones
    /// start from a state nobody else could see. Called under the catalog write guard.
    pub(super) fn note_table_change(
        cat: &Catalog,
        txn: TxnId,
        table: u64,
        before: Option<&TableSchema>,
    ) -> Result<()> {
        {
            let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
            if pending.table_keys.insert((table, txn.0)) {
                pending.owners.insert(txn.0);
                pending.tables.push(TableChange {
                    table,
                    owner: txn.0,
                    before: before.cloned(),
                });
            }
            pending.publish(cat);
        }
        Ok(())
    }

    /// Record that `txn` is about to add or drop a constraint of `table`, keeping the constraints
    /// it has now. Only the first change per transaction is kept. Called under the catalog write
    /// guard.
    pub(super) fn note_constraint_change(cat: &Catalog, txn: TxnId, table: u64) -> Result<()> {
        {
            let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
            if pending.constraint_keys.insert((table, txn.0)) {
                pending.owners.insert(txn.0);
                pending.constraints.push(ConstraintChange {
                    table,
                    owner: txn.0,
                    constraints: constraints_of(cat, TableId(table)),
                    foreign_keys: foreign_keys_of(cat, TableId(table)),
                });
            }
            pending.publish(cat);
        }
        Ok(())
    }

    /// Before `txn` drops the constraint `name` of `table`: take the table exclusively, and for a
    /// foreign key the table it references too, whose rows the key protects (a write admitted
    /// while the constraint is half dropped could break it and still commit if the drop rolls
    /// back), and record the constraints both have now. Called under the catalog write guard.
    pub(super) fn claim_for_constraint_drop(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: TableId,
        name: &str,
    ) -> Result<()> {
        if cat.tables.contains_key(&table.0) {
            self.lock_for_ddl(cat, txn, table.0)?;
        }
        if let Some(parent) = cat
            .foreign_keys
            .get(name)
            .filter(|fk| fk.child_table == table.0)
            .map(|fk| fk.parent_table)
            && cat.tables.contains_key(&parent)
        {
            self.lock_for_ddl(cat, txn, parent)?;
            Self::note_constraint_change(cat, txn, parent)?;
        }
        Self::note_constraint_change(cat, txn, table.0)
    }

    /// Record that `txn` added the foreign key `def`, changing the constraints of its table and of
    /// the table it references. Called under the catalog write guard.
    pub(super) fn note_foreign_key_added(
        cat: &Catalog,
        txn: TxnId,
        def: &ForeignKeyDef,
    ) -> Result<()> {
        Self::note_constraint_change(cat, txn, def.child_table.0)?;
        Self::note_constraint_change(cat, txn, def.parent_table.0)?;
        let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
        pending.owners.insert(txn.0);
        pending.foreign_keys.push((def.name.clone(), txn.0));
        pending.publish(cat);
        Ok(())
    }

    /// Record that `txn` is rewriting `table`'s rows or building an index on it (see the module
    /// docs). Called under the catalog write guard.
    pub(super) fn note_layout_change(&self, cat: &Catalog, txn: TxnId, table: u64) -> Result<()> {
        let mut guard = cat.uncommitted.lock().map_err(|_| poisoned())?;
        let pending = &mut *guard;
        // A table this transaction is creating has no rows an older snapshot could read.
        if pending
            .tables
            .iter()
            .any(|c| c.table == table && c.owner == txn.0 && c.before.is_none())
        {
            return Ok(());
        }
        let txns = self.txns.lock().map_err(|_| poisoned())?;
        pending.collect(&txns);
        let owners = pending.history.entry(table).or_default();
        if !owners.contains(&txn.0) {
            // Every other transaction already running may have resolved the table before this
            // change without it being recorded (no history existed then): assume it resolved the
            // state before the change, unless its current statement has a resolution on record.
            for &other in &txns.active {
                let Some(state) = txns.txns.get(&other) else {
                    continue;
                };
                if other == txn.0 {
                    continue;
                }
                let entry = pending.resolved.entry((other, table)).or_default();
                if current(entry, state.statements).next().is_none() {
                    entry.push(Resolution {
                        statement: state.statements,
                        reflected: newest(owners, |o| o != other && !txns.active.contains(&o)),
                        assumed_for: Some(txn.0),
                    });
                }
            }
            owners.push(txn.0);
        }
        drop(txns);
        pending.publish(cat);
        Ok(())
    }

    /// Undo [`Self::note_layout_change`] for a change `txn` rolled back: the table is as it was,
    /// and nobody has to be checked against it.
    pub(super) fn forget_layout_change(cat: &Catalog, txn: u64, table: u64) -> Result<()> {
        let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
        if let Some(owners) = pending.history.get_mut(&table) {
            owners.retain(|&o| o != txn);
        }
        pending.history.retain(|_, owners| !owners.is_empty());
        for ((_, t), all) in &mut pending.resolved {
            if *t == table {
                all.retain(|r| r.assumed_for != Some(txn));
            }
        }
        pending.resolved.retain(|_, all| !all.is_empty());
        pending.publish(cat);
        Ok(())
    }

    /// Remember which changes of `table`'s history `txn` resolved it with: all but its own (which
    /// it always sees) and those of the transactions `hidden` from it, exactly the ones the
    /// resolution itself masked (a fresh look at the active set could already count one of them
    /// as committed).
    fn record_resolution(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: u64,
        hidden: Option<&Hidden>,
    ) -> Result<()> {
        if !cat.has_uncommitted.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut guard = cat.uncommitted.lock().map_err(|_| poisoned())?;
        let pending = &mut *guard;
        if !pending.history.contains_key(&table) {
            return Ok(());
        }
        let txns = self.txns.lock().map_err(|_| poisoned())?;
        pending.collect_now_and_then(&txns);
        let Some(owners) = pending.history.get(&table) else {
            pending.publish(cat);
            return Ok(());
        };
        let reflected = newest(owners, |o| {
            o != txn.0 && hidden.is_none_or(|h| !h.owners.contains(&o))
        });
        let statement = txns.txns.get(&txn.0).map_or(0, |s| s.statements);
        drop(txns);
        // Keep every resolution of the statement: it resolves its tables while it is analyzed,
        // and its execution may resolve them again, after a change it was not analyzed with has
        // committed. Each of them must match the snapshot it reads under.
        let entry = pending.resolved.entry((txn.0, table)).or_default();
        entry.retain(|r| r.statement.saturating_add(1) >= statement);
        if !entry
            .iter()
            .any(|r| r.statement == statement && r.reflected == reflected)
        {
            entry.push(Resolution {
                statement,
                reflected,
                assumed_for: None,
            });
        }
        pending.publish(cat);
        Ok(())
    }

    /// Refuse a read of `table` by `txn` whose snapshot does not match the schema it resolved
    /// the table with (see the module docs). Called before the read takes `txns`.
    pub(super) fn check_read_snapshot(&self, cat: &Catalog, txn: TxnId, table: u64) -> Result<()> {
        if !cat.has_uncommitted.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut guard = cat.uncommitted.lock().map_err(|_| poisoned())?;
        let pending = &mut *guard;
        if !pending.history.contains_key(&table) {
            return Ok(());
        }
        let txns = self.txns.lock().map_err(|_| poisoned())?;
        pending.collect_now_and_then(&txns);
        pending.publish(cat);
        let Some(owners) = pending.history.get(&table) else {
            return Ok(());
        };
        let Some(state) = txns.txns.get(&txn.0) else {
            return Ok(());
        };
        // A transaction always sees, and resolves with, its own changes: only the others count.
        let seen = newest(owners, |o| o != txn.0 && state.pinned.sees(o));
        // A resolution that reflected an entry since dropped from the history reflected all of
        // the dropped ones, which every snapshot sees.
        let matches = |resolved: Option<u64>| {
            resolved == seen || (seen.is_none() && resolved.is_some_and(|r| !owners.contains(&r)))
        };
        let relevant: Vec<&Resolution> = pending
            .resolved
            .get(&(txn.0, table))
            .map(|all| current(all, state.statements).collect())
            .unwrap_or_default();
        let agrees = if relevant.is_empty() {
            // No resolution on record: the table was resolved before any of this history began.
            matches(newest(owners, |o| o != txn.0 && !txns.active.contains(&o)))
        } else {
            relevant.iter().all(|r| matches(r.reflected))
        };
        if agrees {
            Ok(())
        } else {
            Err(Error::SerializationConflict { txn })
        }
    }

    /// Every table some transaction recorded a change to that has not been cleared yet (cleared
    /// lazily once its owner ends, so this may include a few that are settled).
    pub(super) fn tables_with_uncommitted_ddl(cat: &Catalog) -> Result<HashSet<u64>> {
        Ok(cat
            .uncommitted
            .lock()
            .map_err(|_| poisoned())?
            .tables
            .iter()
            .map(|c| c.table)
            .collect())
    }

    /// Record that `txn` created `index`. Called under the catalog write guard.
    pub(super) fn note_index_created(cat: &Catalog, txn: TxnId, index: u64) -> Result<()> {
        let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
        pending.owners.insert(txn.0);
        pending.indexes.push(IndexChange {
            index,
            owner: txn.0,
        });
        pending.publish(cat);
        Ok(())
    }

    /// The uncommitted changes `txn` must not see, or `None` when there are none. Entries whose
    /// owner has ended are dropped on the way. Lock order: catalog, then the pending set, then
    /// `txns`.
    pub(super) fn hidden_from(&self, cat: &Catalog, txn: TxnId) -> Result<Option<Hidden>> {
        if !cat.has_pending_ddl.load(Ordering::Acquire) {
            return Ok(None);
        }
        let mut pending = cat.uncommitted.lock().map_err(|_| poisoned())?;
        {
            let txns = self.txns.lock().map_err(|_| poisoned())?;
            pending.drop_ended_changes(&txns);
            pending.collect_now_and_then(&txns);
        }
        pending.publish(cat);
        if pending.no_pending_ddl() || pending.only_owner_is(txn.0) {
            return Ok(None);
        }
        let hidden = Hidden {
            tables: pending
                .tables
                .iter()
                .filter(|c| c.owner != txn.0)
                .map(|c| (c.table, c.before.clone()))
                .collect(),
            indexes: pending
                .indexes
                .iter()
                .filter(|c| c.owner != txn.0)
                .map(|c| c.index)
                .collect(),
            constraints: pending
                .constraints
                .iter()
                .filter(|c| c.owner != txn.0)
                .map(|c| (c.table, c.constraints.clone(), c.foreign_keys.clone()))
                .collect(),
            foreign_keys: pending
                .foreign_keys
                .iter()
                .filter(|(_, owner)| *owner != txn.0)
                .map(|(name, _)| name.clone())
                .collect(),
            owners: pending
                .tables
                .iter()
                .map(|c| c.owner)
                .chain(pending.indexes.iter().map(|c| c.owner))
                .chain(pending.constraints.iter().map(|c| c.owner))
                .chain(pending.foreign_keys.iter().map(|(_, owner)| *owner))
                .filter(|&owner| owner != txn.0)
                .collect(),
        };
        drop(pending);
        let empty = hidden.tables.is_empty()
            && hidden.indexes.is_empty()
            && hidden.constraints.is_empty()
            && hidden.foreign_keys.is_empty();
        Ok((!empty).then_some(hidden))
    }

    /// Resolve `(schema, name)` for `txn`: the live catalog, less what other running
    /// transactions changed.
    pub(super) fn resolve_table_for(
        &self,
        cat: &Catalog,
        txn: TxnId,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>> {
        let live = || {
            cat.by_name
                .get(&(schema.to_owned(), name.to_owned()))
                .and_then(|id| cat.tables.get(id))
                .map(|t| t.schema.clone())
        };
        let hidden = self.hidden_from(cat, txn)?;
        let resolved = hidden.as_ref().map_or_else(live, |hidden| {
            if let Some(committed) = hidden.by_committed_name(schema, name) {
                return Some(committed.clone());
            }
            match cat.by_name.get(&(schema.to_owned(), name.to_owned())) {
                // Created, or renamed into this name, by a transaction that has not committed.
                Some(id) if hidden.committed(*id).is_some() => None,
                _ => live(),
            }
        });
        if let Some(table) = &resolved {
            self.record_resolution(cat, txn, table.id.0, hidden.as_ref())?;
        }
        Ok(resolved)
    }

    /// The schema of `table` by id for `txn` (see [`Self::resolve_table_for`]).
    pub(super) fn table_by_id_for(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: u64,
    ) -> Result<Option<TableSchema>> {
        let live = || cat.tables.get(&table).map(|t| t.schema.clone());
        let hidden = self.hidden_from(cat, txn)?;
        let resolved = hidden.as_ref().map_or_else(live, |hidden| {
            hidden
                .committed(table)
                .map_or_else(live, |committed| committed.schema())
        });
        if resolved.is_some() {
            self.record_resolution(cat, txn, table, hidden.as_ref())?;
        }
        Ok(resolved)
    }

    /// Every table `txn` sees, as `(schema, name)` pairs, sorted.
    pub(super) fn tables_for(&self, cat: &Catalog, txn: TxnId) -> Result<Vec<(String, String)>> {
        let hidden = self.hidden_from(cat, txn)?;
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(cat.tables.len());
        for (id, t) in &cat.tables {
            match hidden.as_ref().and_then(|h| h.committed(*id)) {
                Some(Committed::Absent) => {},
                Some(Committed::Was(committed)) => {
                    pairs.push((committed.schema.clone(), committed.name.clone()));
                },
                None => pairs.push((t.schema.schema.clone(), t.schema.name.clone())),
            }
        }
        // Tables another transaction has dropped but not committed are still there for `txn`.
        if let Some(hidden) = &hidden {
            for (id, committed) in &hidden.tables {
                if let Some(committed) = committed
                    && !cat.tables.contains_key(id)
                {
                    pairs.push((committed.schema.clone(), committed.name.clone()));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        Ok(pairs)
    }

    /// The indexes on `table` that `txn` may plan with: those of the live catalog less the ones
    /// another running transaction created.
    pub(super) fn indexes_for(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: TableId,
    ) -> Result<Vec<IndexDef>> {
        let hidden = self.hidden_from(cat, txn)?;
        Ok(cat
            .indexes
            .iter()
            .filter(|(id, i)| {
                i.def.table == table && hidden.as_ref().is_none_or(|h| !h.indexes.contains(id))
            })
            .map(|(_, i)| i.def.clone())
            .collect())
    }

    /// The constraints of `table` that `txn` sees: those it had before another running
    /// transaction added or dropped one, else the live ones.
    pub(super) fn constraints_for(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: TableId,
    ) -> Result<Vec<Constraint>> {
        let hidden = self.hidden_from(cat, txn)?;
        Ok(hidden
            .and_then(|h| {
                h.constraints
                    .into_iter()
                    .find(|(id, _, _)| *id == table.0)
                    .map(|(_, constraints, _)| constraints)
            })
            .unwrap_or_else(|| constraints_of(cat, table)))
    }

    /// The foreign keys of `table` that `txn` sees (see [`Self::constraints_for`]), without any
    /// another running transaction added.
    pub(super) fn foreign_keys_for(
        &self,
        cat: &Catalog,
        txn: TxnId,
        table: TableId,
    ) -> Result<Vec<ForeignKeyDef>> {
        let Some(hidden) = self.hidden_from(cat, txn)? else {
            return Ok(foreign_keys_of(cat, table));
        };
        let keys = hidden
            .constraints
            .iter()
            .find(|(id, _, _)| *id == table.0)
            .map_or_else(|| foreign_keys_of(cat, table), |(_, _, fks)| fks.clone());
        Ok(keys
            .into_iter()
            .filter(|fk| !hidden.foreign_keys.contains(&fk.name))
            .collect())
    }

    /// Whether `(schema, name)` is spoken for by a transaction that has not committed: it names
    /// a table that transaction created, or a committed table it renamed away or dropped (which
    /// its rollback would bring back). Creating or renaming into such a name must wait for the
    /// outcome, so the caller refuses with a serialization conflict.
    pub(super) fn name_held_by_other(
        &self,
        cat: &Catalog,
        txn: TxnId,
        schema: &str,
        name: &str,
    ) -> Result<bool> {
        let Some(hidden) = self.hidden_from(cat, txn)? else {
            return Ok(false);
        };
        if hidden.by_committed_name(schema, name).is_some() {
            return Ok(true);
        }
        Ok(cat
            .by_name
            .get(&(schema.to_owned(), name.to_owned()))
            .is_some_and(|id| hidden.committed(*id).is_some()))
    }

    /// Whether a transaction that is still running dropped `table`, which then still exists for
    /// every other transaction.
    pub(super) fn dropped_by_running(&self, cat: &Catalog, table: TableId) -> Result<bool> {
        // No transaction has id 0, so every running owner counts as another one.
        Ok(self
            .hidden_from(cat, TxnId(0))?
            .is_some_and(|hidden| matches!(hidden.committed(table.0), Some(Committed::Was(_)))))
    }

    /// The error for `table` missing from the live catalog when `txn` reaches it by id: a
    /// serialization conflict when another running transaction dropped it (the table still
    /// exists for `txn`, but its rows are held by that transaction's undo), else table-not-found.
    pub(super) fn missing_table(&self, cat: &Catalog, txn: TxnId, table: TableId) -> Error {
        match self.hidden_from(cat, txn) {
            Ok(Some(hidden)) if matches!(hidden.committed(table.0), Some(Committed::Was(_))) => {
                Error::SerializationConflict { txn }
            },
            Ok(_) => table_not_found(table),
            Err(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use nusadb_core::engine::{AlterOp, IndexKind, IsolationLevel, TableDef};
    use nusadb_core::{ColumnDef, ColumnType, Error, IndexDef, StorageEngine, TableId, TxnId};

    use crate::BtreeEngine;

    fn int(name: &str) -> ColumnDef {
        ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Int,
            nullable: true,
        }
    }

    fn begin(engine: &BtreeEngine) -> TxnId {
        engine.begin(IsolationLevel::ReadCommitted).unwrap()
    }

    fn create(engine: &BtreeEngine, txn: TxnId, schema: &str, name: &str) -> TableId {
        engine
            .create_table(
                txn,
                &TableDef {
                    schema: schema.to_owned(),
                    name: name.to_owned(),
                    columns: vec![int("id"), int("a")],
                },
            )
            .unwrap()
    }

    /// A committed table `public.t (id, a)`.
    fn committed_table(engine: &BtreeEngine) -> TableId {
        let txn = begin(engine);
        let id = create(engine, txn, "public", "t");
        engine.commit(txn).unwrap();
        id
    }

    fn columns(engine: &BtreeEngine, txn: TxnId, name: &str) -> Option<Vec<String>> {
        engine
            .lookup_table_as_of(txn, name)
            .unwrap()
            .map(|t| t.columns.into_iter().map(|c| c.name).collect())
    }

    #[test]
    fn an_uncommitted_alter_is_seen_only_by_its_own_transaction() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let alterer = begin(&engine);
        let other = begin(&engine);
        engine
            .alter_table(alterer, id, &AlterOp::AddColumn(int("c")))
            .unwrap();
        assert_eq!(
            columns(&engine, alterer, "t").unwrap(),
            ["id", "a", "c"],
            "the altering transaction sees its own column"
        );
        assert_eq!(columns(&engine, other, "t").unwrap(), ["id", "a"]);
        let by_id = engine.lookup_table_by_id_as_of(other, id).unwrap().unwrap();
        assert_eq!(by_id.columns.len(), 2, "by id too");
        engine.commit(alterer).unwrap();
        assert_eq!(
            columns(&engine, other, "t").unwrap(),
            ["id", "a", "c"],
            "committed DDL is visible to everyone"
        );
        engine.commit(other).unwrap();
    }

    #[test]
    fn a_rolled_back_alter_never_shows_and_its_version_is_not_reused() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let first = begin(&engine);
        engine
            .alter_table(first, id, &AlterOp::AddColumn(int("c")))
            .unwrap();
        let abandoned = engine.current_schema_version(id).unwrap().unwrap();
        engine.rollback(first).unwrap();
        assert_eq!(engine.current_schema_version(id).unwrap(), Some(0));
        let second = begin(&engine);
        engine
            .alter_table(second, id, &AlterOp::AddColumn(int("d")))
            .unwrap();
        engine.commit(second).unwrap();
        let version = engine.current_schema_version(id).unwrap().unwrap();
        assert!(
            version > abandoned,
            "a new ALTER numbers past the rolled-back one: {version} vs {abandoned}"
        );
        let reader = begin(&engine);
        assert_eq!(columns(&engine, reader, "t").unwrap(), ["id", "a", "d"]);
        engine.commit(reader).unwrap();
    }

    #[test]
    fn create_rename_and_drop_keep_the_committed_names_for_others() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let ddl = begin(&engine);
        let other = begin(&engine);

        create(&engine, ddl, "public", "fresh");
        assert!(columns(&engine, ddl, "fresh").is_some());
        assert!(columns(&engine, other, "fresh").is_none());
        assert!(
            !engine
                .list_tables_as_of(other)
                .unwrap()
                .contains(&"fresh".to_owned())
        );
        // The name is spoken for until the creator ends.
        let err = engine
            .create_table(
                other,
                &TableDef {
                    schema: "public".to_owned(),
                    name: "fresh".to_owned(),
                    columns: vec![int("x")],
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::SerializationConflict { .. }), "{err}");

        engine
            .alter_table(
                ddl,
                id,
                &AlterOp::RenameTable {
                    name: "renamed".to_owned(),
                },
            )
            .unwrap();
        assert!(
            columns(&engine, other, "t").is_some(),
            "old name still resolves"
        );
        assert!(columns(&engine, other, "renamed").is_none());
        assert_eq!(
            engine.list_tables_qualified_as_of(other).unwrap(),
            [("public".to_owned(), "t".to_owned())]
        );

        engine.drop_table(ddl, id).unwrap();
        assert!(columns(&engine, ddl, "renamed").is_none());
        assert!(
            columns(&engine, other, "t").is_some(),
            "a dropped table still exists for the others"
        );
        // Reading it has to wait for the dropping transaction's outcome.
        let err = engine.scan(other, id).err().unwrap();
        assert!(matches!(err, Error::SerializationConflict { .. }), "{err}");
        assert_eq!(engine.approx_row_count(id).unwrap(), 0);

        engine.rollback(ddl).unwrap();
        assert!(columns(&engine, other, "t").is_some());
        assert!(columns(&engine, other, "fresh").is_none());
        assert!(engine.scan(other, id).is_ok());
        engine.commit(other).unwrap();
    }

    #[test]
    fn an_uncommitted_index_is_not_offered_to_other_transactions() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let ddl = begin(&engine);
        let other = begin(&engine);
        engine
            .create_index(
                ddl,
                &IndexDef {
                    name: "t_a".to_owned(),
                    table: id,
                    columns: vec!["a".to_owned()],
                    key_exprs: vec![],
                    predicate: None,
                    include: vec![],
                    kind: IndexKind::BTree,
                    unique: false,
                },
            )
            .unwrap();
        let names = |txn| -> Vec<String> {
            engine
                .list_indexes_as_of(txn, id)
                .unwrap()
                .into_iter()
                .map(|d| d.name)
                .collect()
        };
        assert_eq!(names(ddl), ["t_a"]);
        assert!(names(other).is_empty());
        engine.commit(ddl).unwrap();
        assert_eq!(names(other), ["t_a"]);
        engine.commit(other).unwrap();
    }

    #[test]
    fn a_statement_resolved_before_a_rewrite_committed_cannot_read_after_it() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let reader = begin(&engine);
        // The reader resolves the table for its statement while the change is still running...
        let alterer = begin(&engine);
        engine
            .alter_table(alterer, id, &AlterOp::AddColumn(int("c")))
            .unwrap();
        assert_eq!(columns(&engine, reader, "t").unwrap(), ["id", "a"]);
        // ...the change commits, the statement resolves the table again while it runs, and it
        // reads under a snapshot that sees the change: the first resolution does not match.
        engine.commit(alterer).unwrap();
        assert_eq!(columns(&engine, reader, "t").unwrap(), ["id", "a", "c"]);
        engine.begin_statement(reader).unwrap();
        let err = engine.scan(reader, id).err().unwrap();
        assert!(matches!(err, Error::SerializationConflict { .. }), "{err}");
        engine.rollback(reader).unwrap();

        // A statement that starts and resolves after the commit reads normally.
        let next = begin(&engine);
        engine.begin_statement(next).unwrap();
        assert_eq!(columns(&engine, next, "t").unwrap(), ["id", "a", "c"]);
        engine.begin_statement(next).unwrap();
        assert!(engine.scan(next, id).is_ok());
        engine.commit(next).unwrap();
    }

    #[test]
    fn an_uncommitted_check_is_not_among_the_constraints_others_see() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let ddl = begin(&engine);
        let other = begin(&engine);
        engine
            .add_check_constraint(ddl, id, "ck", b"a > 0")
            .unwrap();
        let names = |txn| -> Vec<String> {
            engine
                .list_constraints_as_of(txn, id)
                .unwrap()
                .into_iter()
                .map(|c| c.name)
                .collect()
        };
        assert_eq!(names(ddl), ["ck"]);
        assert!(names(other).is_empty());
        engine.commit(ddl).unwrap();
        assert_eq!(names(other), ["ck"]);
        engine.commit(other).unwrap();
    }

    #[test]
    fn a_table_resolved_before_any_change_began_is_checked_too() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        // Resolved while nothing was recorded for the table...
        let reader = begin(&engine);
        assert_eq!(columns(&engine, reader, "t").unwrap(), ["id", "a"]);
        // ...then a rewrite starts and commits before the statement reads.
        let alterer = begin(&engine);
        engine
            .alter_table(
                alterer,
                id,
                &AlterOp::DropColumn {
                    name: "a".to_owned(),
                },
            )
            .unwrap();
        engine.commit(alterer).unwrap();
        engine.begin_statement(reader).unwrap();
        let err = engine.scan(reader, id).err().unwrap();
        assert!(matches!(err, Error::SerializationConflict { .. }), "{err}");
        engine.rollback(reader).unwrap();
    }

    fn index_on(engine: &BtreeEngine, txn: TxnId, table: TableId, name: &str) {
        engine
            .create_index(
                txn,
                &IndexDef {
                    name: name.to_owned(),
                    table,
                    columns: vec!["a".to_owned()],
                    key_exprs: vec![],
                    predicate: None,
                    include: vec![],
                    kind: IndexKind::BTree,
                    unique: false,
                },
            )
            .unwrap();
    }

    /// A DDL statement as the SQL layer runs one: start it, resolve the table, change it, and
    /// read it back (the rewrite or index build), then commit.
    fn ddl(engine: &BtreeEngine, id: TableId, change: impl Fn(TxnId)) -> Result<(), Error> {
        let txn = begin(engine);
        engine.begin_statement(txn).unwrap();
        assert!(columns(engine, txn, "t").is_some());
        engine.begin_statement(txn).unwrap();
        change(txn);
        let read = engine.scan(txn, id).map(|_| ());
        if read.is_ok() {
            engine.commit(txn).unwrap();
        } else {
            engine.rollback(txn).unwrap();
        }
        read
    }

    #[test]
    fn an_idle_reader_does_not_block_later_ddl_on_the_table() {
        for level in [
            IsolationLevel::RepeatableRead,
            IsolationLevel::ReadCommitted,
        ] {
            let engine = BtreeEngine::new();
            let id = committed_table(&engine);
            // The idle holder read the table and stays open: its snapshot keeps the history.
            let holder = engine.begin(level).unwrap();
            engine.begin_statement(holder).unwrap();
            assert!(columns(&engine, holder, "t").is_some());
            assert!(engine.scan(holder, id).is_ok());
            for n in 0..5 {
                ddl(&engine, id, |txn| {
                    engine
                        .alter_table(txn, id, &AlterOp::AddColumn(int(&format!("c{n}"))))
                        .unwrap();
                })
                .unwrap_or_else(|e| panic!("{level:?} ADD COLUMN #{n}: {e}"));
                ddl(&engine, id, |txn| {
                    index_on(&engine, txn, id, &format!("i{n}"));
                })
                .unwrap_or_else(|e| panic!("{level:?} CREATE INDEX #{n}: {e}"));
            }
            engine.rollback(holder).unwrap();
        }
    }

    #[test]
    fn rolled_back_changes_leave_nothing_behind() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let holders: Vec<TxnId> = (0..3)
            .map(|_| {
                let h = engine.begin(IsolationLevel::RepeatableRead).unwrap();
                assert!(engine.scan(h, id).is_ok());
                h
            })
            .collect();
        for n in 0..50 {
            let txn = begin(&engine);
            engine
                .alter_table(txn, id, &AlterOp::AddColumn(int(&format!("c{n}"))))
                .unwrap();
            engine.rollback(txn).unwrap();
        }
        {
            let cat = engine.catalog.read().unwrap();
            let (history, resolved) = {
                let pending = cat.uncommitted.lock().unwrap();
                (pending.history.len(), pending.resolved.len())
            };
            assert_eq!(
                (history, resolved),
                (0, 0),
                "nothing left for rolled-back changes"
            );
        }
        for h in holders {
            engine.rollback(h).unwrap();
        }
        // Once nothing is recorded, resolution takes the lock-free path again.
        let txn = begin(&engine);
        assert!(columns(&engine, txn, "t").is_some());
        let pending_ddl = engine
            .catalog
            .read()
            .unwrap()
            .has_pending_ddl
            .load(std::sync::atomic::Ordering::Acquire);
        assert!(!pending_ddl);
    }

    #[test]
    fn an_idle_reader_keeps_one_assumed_resolution_however_many_changes_commit() {
        let engine = BtreeEngine::new();
        let id = committed_table(&engine);
        let holder = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert!(engine.scan(holder, id).is_ok());
        for n in 0..30 {
            let txn = begin(&engine);
            engine
                .alter_table(txn, id, &AlterOp::AddColumn(int(&format!("c{n}"))))
                .unwrap();
            engine.commit(txn).unwrap();
        }
        let kept = {
            let cat = engine.catalog.read().unwrap();
            let pending = cat.uncommitted.lock().unwrap();
            pending.resolved.get(&(holder.0, id.0)).map_or(0, Vec::len)
        };
        assert_eq!(kept, 1);
        engine.rollback(holder).unwrap();
    }
}
