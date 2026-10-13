//! Two-phase commit: `PREPARE TRANSACTION`, `COMMIT PREPARED` and `ROLLBACK PREPARED`.
//!
//! # What a prepared transaction is
//!
//! `PREPARE TRANSACTION 'gid'` ends a transaction's session without ending the transaction. It
//! stays in the active set, so no view sees its writes; it keeps its locks and its pinned view
//! (purge treats it as running); and it waits, under its global identifier, for a
//! `COMMIT PREPARED` or `ROLLBACK PREPARED` from any session, across any number of restarts.
//! Only row and index changes can be prepared: a transaction that changed the schema, a sequence
//! or statistics, or touched a temporary table, is refused and rolled back.
//!
//! # The log
//!
//! A transaction's ordinary records are written as it runs and include the compensations of
//! every savepoint it rolled back to. At `PREPARE`, the transaction's net effect is written again
//! in a canonical form, read off its undo list (which already holds only what survived its
//! savepoints) and the rows' current versions:
//!
//! ```text
//! PrepareBegin(txn)   Insert / Update / Delete / IndexInsert / IndexDelete ...   Prepare(txn, gid, owner, database, time, locks)
//! ```
//!
//! and fsynced. The `Prepare` record is the durability point: without it the begin and what
//! follows are a prepare that never finished, the transaction has no commit marker, and replay
//! skips it as it skips any uncommitted transaction. With it, replay reads the transaction from
//! its prepare records alone; its earlier records are superseded and ignored.
//!
//! `COMMIT PREPARED` appends the ordinary `CommitTxn` marker and fsyncs it; replay then applies
//! the prepare records in place, exactly like any committed operation. `ROLLBACK PREPARED`
//! appends `AbortTxn` and fsyncs it before undoing anything in memory, since unlike an ordinary
//! abort marker it is not advisory.
//!
//! # Recovery
//!
//! A transaction whose `Prepare` record has no commit or abort marker after it is set up again
//! once every other record has replayed: it re-enters the active set under its own id, takes the
//! locks its `Prepare` record lists, and its prepare records are applied through the same paths a
//! live write takes (old versions parked in the undo arena, undo entries pushed), so other
//! transactions read the versions before it, a conflicting writer is refused as before, and a
//! later `ROLLBACK PREPARED` undoes it like any rollback. Other transactions that committed after
//! it could not have touched its rows (it held them), so applying it last is equivalent to
//! applying it in place.
//!
//! # Checkpoints
//!
//! A checkpoint image holds committed state only, and a prepared transaction may wait for as long
//! as its coordinator likes, so it cannot hold checkpoints off the way a running transaction does.
//! Under the checkpoint's quiesce each prepared transaction is set aside: its canonical records
//! are taken, its changes undone in memory, the image written from the committed state with the
//! prepare records of every prepared transaction appended after it, and the transaction applied
//! again from those records. The log the image replaces can then be truncated as usual: whatever
//! a prepared transaction needs is in the new image, and nothing of it is lost if the checkpoint
//! fails at any step (the old log still holds it until the new image is in place).
//!
//! # Standby
//!
//! A standby applies a primary's prepare records like any other: the transaction is set up as
//! prepared on the standby too, and the commit or abort marker a later segment carries finishes
//! it there. A standby refuses `PREPARE TRANSACTION`, `COMMIT PREPARED` and `ROLLBACK PREPARED`
//! of its own, as it refuses every write; once promoted it finishes them like a primary.
//!
//! # Locks and conflict checks
//!
//! The `Prepare` record lists every lock the transaction holds (row, unique-key and table locks,
//! each with its mode), and recovery takes them again before anything else runs, so a writer that
//! was refused before a restart is refused after it. Row writes are refused by version admission
//! as well, since the prepared transaction stays in the active set. For the `SERIALIZABLE`
//! antidependency check a prepared transaction counts as committed (like a staged commit): it
//! passed its own check at `PREPARE`, and a reader whose read it overwrote aborts at its own
//! commit rather than let a skew through. A `ROLLBACK PREPARED` after that makes such an abort
//! merely conservative.
//!
//! # Alternatives considered
//!
//! - *Keep the log from the oldest prepared transaction on.* The image would still have to leave
//!   the transaction's versions out (it holds committed state only, and its pages carry no undo
//!   history), so the same undo and redo were needed anyway, and a forgotten prepared
//!   transaction would have kept the whole log growing without bound.
//! - *Replay the transaction's original records at recovery.* They include savepoint
//!   compensations and could include schema changes undone by a savepoint; the canonical records
//!   read off the undo list hold only what survived, in a form whose replay is plain.
//! - *Refuse a checkpoint while a transaction is prepared.* Simple, and wrong for a feature whose
//!   point is that the transaction may wait on an external coordinator for a long time.
//!
//! # Consequences
//!
//! A prepared transaction's changes are written twice to the log (as it ran, and at `PREPARE`),
//! and every checkpoint while it waits writes them again into the image and briefly undoes and
//! redoes them under its quiesce. Its pinned view holds back purge until it is finished, as a
//! running transaction's does; `--max-prepared-transactions` bounds how many can do so. The log
//! and image records are new kinds, so a directory that may hold them is data format 2.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use nusadb_core::engine::IsolationLevel;
use nusadb_core::{Error, PreparedTransaction, Result, TableId, TxnId};
use nusadb_wal::WalRecord;

use super::{
    BtreeEngine, Catalog, CatalogRef, ClusteredTree, CommitDeltas, LockId, TxnDomain, TxnState,
    UndoOp, append_commit_marker, corrupt_row, index_not_found, poisoned, table_not_found, tid_of,
    tuple_not_found, unix_time_ms, unknown_txn,
};
use crate::mvcc::{self, ReadView, RowMeta, UndoVersion};
use crate::wal::{LoggedOp, PreparedLock};

/// How many prepared transactions may wait at once unless configured otherwise.
pub const DEFAULT_MAX_PREPARED_TRANSACTIONS: usize = 64;

/// The longest global identifier `PREPARE TRANSACTION` accepts, in bytes.
pub const MAX_GID_BYTES: usize = 200;

/// What a prepared transaction carries besides its changes.
#[derive(Debug, Clone)]
pub(super) struct PreparedMeta {
    gid: String,
    owner: String,
    database: String,
    prepared_at_ms: u64,
    /// The order it was prepared in, among the prepared transactions of this process.
    seq: u64,
}

/// The configured ceiling on prepared transactions (`0` refuses every `PREPARE TRANSACTION`).
#[derive(Debug)]
pub(super) struct PreparedLimit(AtomicUsize);

impl Default for PreparedLimit {
    fn default() -> Self {
        Self(AtomicUsize::new(DEFAULT_MAX_PREPARED_TRANSACTIONS))
    }
}

/// A prepared transaction read back from the log or an image: still waiting, with its prepare
/// records to apply.
#[derive(Debug)]
pub(super) struct Recovered {
    pub(super) txn: u64,
    pub(super) gid: String,
    pub(super) owner: String,
    pub(super) database: String,
    pub(super) prepared_at_ms: u64,
    pub(super) locks: Vec<PreparedLock>,
    pub(super) ops: Vec<LoggedOp>,
}

/// A prepared transaction set aside for the length of a checkpoint: its changes are out of
/// memory and `program` puts them back.
#[derive(Debug)]
pub(super) struct Parked {
    txn: u64,
    meta: PreparedMeta,
    locks: Vec<PreparedLock>,
    program: Vec<LoggedOp>,
}

impl Parked {
    /// The prepare records the checkpoint image carries for this transaction.
    pub(super) fn records(&self) -> impl Iterator<Item = WalRecord> + '_ {
        prepare_records(self.txn, &self.meta, &self.locks, &self.program)
    }
}

/// `PrepareBegin`, the canonical operations, then the `Prepare` marker.
fn prepare_records<'a>(
    txn: u64,
    meta: &'a PreparedMeta,
    locks: &'a [PreparedLock],
    program: &'a [LoggedOp],
) -> impl Iterator<Item = WalRecord> + 'a {
    std::iter::once(LoggedOp::PrepareBegin { txn }.to_record())
        .chain(program.iter().map(LoggedOp::to_record))
        .chain(std::iter::once_with(move || {
            LoggedOp::Prepare {
                txn,
                gid: meta.gid.clone(),
                owner: meta.owner.clone(),
                database: meta.database.clone(),
                prepared_at_ms: meta.prepared_at_ms,
                locks: locks.to_vec(),
            }
            .to_record()
        }))
}

/// The error for a session-level `COMMIT`/`ROLLBACK` (or a second `PREPARE`) of a transaction that
/// is already prepared. Kind `InvalidInput`.
pub(super) fn is_prepared(txn: TxnId) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "transaction {} is prepared; finish it with COMMIT PREPARED or ROLLBACK PREPARED",
            txn.0
        ),
    ))
}

/// No prepared transaction has this identifier. Kind `NotFound`.
fn unknown_gid(gid: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("prepared transaction with identifier \"{gid}\" does not exist"),
    ))
}

/// A refusal of `PREPARE TRANSACTION`, of kind `kind` (see the treaty's documentation of the
/// kinds the SQL layer reports).
fn refused(kind: std::io::ErrorKind, message: String) -> Error {
    Error::Io(std::io::Error::new(kind, message))
}

/// A prepare record set that cannot be applied: the log is not what this engine wrote.
fn bad_program(txn: u64) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "nusadb-btree: the prepare records of transaction {txn} hold an operation a prepared \
             transaction cannot have"
        ),
    ))
}

/// The table and row an undo entry changed, for a row change.
const fn row_of(op: &UndoOp) -> Option<(u64, u64)> {
    match op {
        UndoOp::Inserted { table, row_id }
        | UndoOp::Updated { table, row_id, .. }
        | UndoOp::Deleted { table, row_id, .. } => Some((*table, *row_id)),
        _ => None,
    }
}

/// The lock-table entry and hold mode a [`PreparedLock`] names.
const fn lock_id_of(lock: PreparedLock) -> (LockId, bool) {
    match lock {
        PreparedLock::Row {
            table,
            page,
            slot,
            exclusive,
        } => (LockId::Row { table, page, slot }, exclusive),
        PreparedLock::Key {
            table,
            hash,
            exclusive,
        } => (LockId::Key { table, hash }, exclusive),
        PreparedLock::Table { table, exclusive } => (LockId::Table { table }, exclusive),
    }
}

/// Every lock `txn` holds, with its mode, in the order it took them.
fn held_locks(t: &TxnDomain, txn: u64) -> Vec<PreparedLock> {
    let Some(state) = t.txns.get(&txn) else {
        return Vec::new();
    };
    state
        .locks
        .iter()
        .map(|id| {
            let exclusive = t
                .locks
                .get(id)
                .and_then(|h| h.holders.get(&txn))
                .copied()
                .unwrap_or(false);
            match *id {
                LockId::Row { table, page, slot } => PreparedLock::Row {
                    table,
                    page,
                    slot,
                    exclusive,
                },
                LockId::Key { table, hash } => PreparedLock::Key {
                    table,
                    hash,
                    exclusive,
                },
                LockId::Table { table } => PreparedLock::Table { table, exclusive },
            }
        })
        .collect()
}

/// Fold a committed transaction's deltas into the approximate counters under a catalog guard the
/// caller already holds.
fn fold_deltas(cat: &Catalog, deltas: &CommitDeltas) {
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
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "the commit gate held across a prepare or a finish IS what orders it against \
              checkpoints and other committers; dropping it earlier would let a checkpoint see a \
              half-prepared transaction"
)]
impl BtreeEngine {
    /// Set how many prepared transactions may wait at once. `0` refuses every
    /// `PREPARE TRANSACTION`; transactions already prepared are kept whatever the new limit.
    pub fn set_max_prepared_transactions(&self, limit: usize) {
        self.max_prepared.0.store(limit, Ordering::Relaxed);
    }

    /// How many prepared transactions may wait at once.
    pub fn max_prepared_transactions(&self) -> usize {
        self.max_prepared.0.load(Ordering::Relaxed)
    }

    /// Roll `txn` back after a refusal of its prepare, and hand back `error` to report. The commit
    /// gate is released first, as a commit-time abort does.
    fn refuse_prepare(
        &self,
        txn: TxnId,
        gate: std::sync::MutexGuard<'_, ()>,
        error: Error,
    ) -> Error {
        if let Ok(state) = self.take_txn(txn) {
            drop(gate);
            self.abort(txn, state);
        }
        error
    }

    /// `PREPARE TRANSACTION`: see the module docs. On any refusal the transaction is rolled back.
    #[allow(
        clippy::too_many_lines,
        reason = "one ordered sequence of refusals, records and the durability point; split, the \
                  gate held across it would be harder to follow"
    )]
    pub(super) fn prepare_txn(
        &self,
        txn: TxnId,
        gid: &str,
        owner: &str,
        database: &str,
    ) -> Result<()> {
        self.ensure_healthy()?;
        self.finish_open_scans(txn);
        // The gate orders the prepare against committers (their antidependency checks must see
        // it) and against checkpoints (which must find it either running or fully prepared).
        let gate = self.commit_gate.lock().map_err(|_| poisoned())?;
        let refusal = {
            let t = self.txns.lock().map_err(|_| poisoned())?;
            match t.txns.get(&txn.0) {
                None => return Err(unknown_txn(txn)),
                Some(state) if state.prepared.is_some() => return Err(is_prepared(txn)),
                Some(_) => {},
            }
            let limit = self.max_prepared_transactions();
            if self.is_standby() {
                Some(Error::ReadOnly(
                    "this server is a standby; PREPARE TRANSACTION is refused until it is promoted"
                        .to_owned(),
                ))
            } else if gid.len() > MAX_GID_BYTES {
                Some(refused(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "transaction identifier \"{gid}\" is longer than {MAX_GID_BYTES} bytes"
                    ),
                ))
            } else if t.prepared_gids.contains_key(gid) {
                Some(refused(
                    std::io::ErrorKind::AlreadyExists,
                    format!("transaction identifier \"{gid}\" is already in use"),
                ))
            } else if limit == 0 {
                Some(refused(
                    std::io::ErrorKind::InvalidInput,
                    "prepared transactions are disabled (the maximum number of prepared \
                     transactions is 0)"
                        .to_owned(),
                ))
            } else if t.prepared_gids.len() >= limit {
                Some(refused(
                    std::io::ErrorKind::QuotaExceeded,
                    format!(
                        "maximum number of prepared transactions reached ({limit}); finish one \
                         with COMMIT PREPARED or ROLLBACK PREPARED, or raise the limit"
                    ),
                ))
            } else {
                None
            }
        };
        if let Some(error) = refusal {
            return Err(self.refuse_prepare(txn, gate, error));
        }
        // A prepared transaction will commit unless someone rolls it back, so a SERIALIZABLE one
        // must pass its antidependency check now, while it can still be refused.
        match self.serializable_read_conflict(txn.0) {
            Ok(false) => {},
            Ok(true) => {
                return Err(self.refuse_prepare(txn, gate, Error::SerializationConflict { txn }));
            },
            Err(e) => return Err(self.refuse_prepare(txn, gate, e)),
        }
        // The canonical records, read from the undo list (taken out while the trees are read, so
        // the transaction table is not held across them).
        let undo = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            t.txns
                .get_mut(&txn.0)
                .map(|s| std::mem::take(&mut s.undo))
                .unwrap_or_default()
        };
        let program = {
            let cat = self.catalog.read().map_err(|_| poisoned())?;
            self.program_of(&cat, txn.0, &undo)
        };
        let (meta, locks) = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            if let Some(state) = t.txns.get_mut(&txn.0) {
                state.undo = undo;
            }
            let seq = t.next_prepare_seq;
            t.next_prepare_seq += 1;
            let meta = PreparedMeta {
                gid: gid.to_owned(),
                owner: owner.to_owned(),
                database: database.to_owned(),
                prepared_at_ms: unix_time_ms(),
                seq,
            };
            (meta, held_locks(&t, txn.0))
        };
        let program = match program {
            Ok(program) => program,
            Err(e) => return Err(self.refuse_prepare(txn, gate, e)),
        };
        // Appended under the gate; a failure (a full disk) leaves at most a prepare that never
        // finished in the log, which replay ignores, and the transaction is rolled back.
        let durable_at = match &self.wal {
            None => None,
            Some(wal) => {
                let appended: Result<u64> = (|| {
                    let mut w = wal.lock().map_err(|_| poisoned())?;
                    for record in prepare_records(txn.0, &meta, &locks, &program) {
                        w.writer.append(&record)?;
                    }
                    Ok(w.writer.next_lsn().0.saturating_sub(1))
                })();
                match appended {
                    Ok(lsn) => Some(lsn),
                    Err(e) => return Err(self.refuse_prepare(txn, gate, e)),
                }
            },
        };
        drop(program);
        {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            // From here it counts as committed for every other transaction's antidependency
            // check (see `serializable_read_conflict`).
            t.bump_staged_versions(txn.0);
            t.prepared_gids.insert(gid.to_owned(), txn.0);
            if let Some(state) = t.txns.get_mut(&txn.0) {
                state.prepared = Some(Box::new(meta));
                // Its own antidependency check is behind it.
                state.reads = HashSet::new();
                state.predicate_reads = HashSet::new();
            }
        }
        drop(gate);
        if let Some(seq) = durable_at {
            let flushed = self.group.commit(seq, || {
                let Some(wal) = &self.wal else {
                    return Err(std::io::Error::other("group commit without a WAL"));
                };
                self.flush_and_sync(wal)
            });
            if let Err(e) = flushed {
                // The same stance as a failed commit fsync: the prepare record may reach the disk
                // regardless, so reporting a failure and serving on could contradict the restart.
                eprintln!(
                    "nusadb-btree: FATAL: WAL fsync failed during PREPARE TRANSACTION ({e}); \
                     aborting to preserve durability (the database recovers its durable prefix on \
                     restart)"
                );
                std::process::abort();
            }
        }
        Ok(())
    }

    /// `COMMIT PREPARED` (`commit`) or `ROLLBACK PREPARED` of the transaction named `gid`.
    pub(super) fn finish_prepared(
        &self,
        gid: &str,
        owner: Option<&str>,
        commit: bool,
    ) -> Result<()> {
        self.ensure_healthy()?;
        let gate = self.commit_gate.lock().map_err(|_| poisoned())?;
        if self.is_standby() {
            return Err(Error::ReadOnly(
                "this server is a standby; a prepared transaction is finished on the primary"
                    .to_owned(),
            ));
        }
        // Claimed under the transaction table: a second finisher finds the name gone. Until it
        // ends, the transaction counts as staged for antidependency checks, and as running for a
        // checkpoint, which therefore waits for it.
        let txn = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            let txn = *t.prepared_gids.get(gid).ok_or_else(|| unknown_gid(gid))?;
            // The owner is checked under the same lock that claims the name, so the transaction
            // finished is the one that was checked.
            if let Some(expected) = owner {
                let prepared_by = t
                    .txns
                    .get(&txn)
                    .and_then(|s| s.prepared.as_ref())
                    .map(|m| m.owner.as_str());
                if prepared_by != Some(expected) {
                    return Err(refused(
                        std::io::ErrorKind::PermissionDenied,
                        format!(
                            "only the role that prepared transaction \"{gid}\" ({}) or a \
                             superuser may finish it",
                            prepared_by.unwrap_or("unknown")
                        ),
                    ));
                }
            }
            t.prepared_gids.remove(gid);
            t.staged.insert(txn);
            txn
        };
        let unclaim = || -> Result<()> {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            t.staged.remove(&txn);
            t.prepared_gids.insert(gid.to_owned(), txn);
            Ok(())
        };
        let marker_at = match &self.wal {
            None => None,
            Some(wal) => {
                let appended = {
                    let mut w = wal.lock().map_err(|_| poisoned())?;
                    if commit {
                        append_commit_marker(self, &mut w.writer, TxnId(txn))
                    } else {
                        w.writer.append(&WalRecord::AbortTxn { txn: TxnId(txn) })
                    }
                };
                match appended {
                    Ok(lsn) => Some(lsn.0),
                    Err(e) => {
                        // Nothing was written: the transaction is still prepared.
                        unclaim()?;
                        return Err(e);
                    },
                }
            },
        };
        // A commit's fsync runs outside the gate, shared with concurrent committers; a rollback
        // keeps it until its undo is done, so no checkpoint sees the transaction half undone.
        let gate = (!commit).then_some(gate);
        if let Some(seq) = marker_at {
            let flushed = self.group.commit(seq, || {
                let Some(wal) = &self.wal else {
                    return Err(std::io::Error::other("group commit without a WAL"));
                };
                self.flush_and_sync(wal)
            });
            if let Err(e) = flushed {
                // A marker that may or may not reach the disk decides the transaction's fate after
                // a restart; serving on with the other fate in memory would contradict it.
                eprintln!(
                    "nusadb-btree: FATAL: WAL fsync failed while finishing a prepared \
                     transaction ({e}); aborting to preserve durability (the database recovers \
                     its durable prefix on restart)"
                );
                std::process::abort();
            }
        }
        let Ok(mut t) = self.txns.lock() else {
            eprintln!(
                "nusadb-btree: FATAL: txns lock poisoned after a durable prepared-transaction \
                 marker; aborting so recovery replays it on restart"
            );
            std::process::abort();
        };
        t.staged.remove(&txn);
        // Visible to new readers (or gone) from here: the finished instant moves, which also
        // evens out the staged bump the prepare made.
        t.bump_finished_versions(txn);
        if commit {
            let deltas =
                Self::finish_commit(&mut t, TxnId(txn), &self.data_version, &self.txn_ended);
            drop(t);
            return self.apply_commit_deltas(&deltas);
        }
        let state = t.txns.remove(&txn);
        drop(t);
        if let Some(state) = state {
            // The durable abort marker is already written.
            self.abort_with(TxnId(txn), state, false);
        }
        drop(gate);
        Ok(())
    }

    /// Every transaction prepared and not being finished, oldest first.
    pub(super) fn prepared_list(&self) -> Result<Vec<PreparedTransaction>> {
        let t = self.txns.lock().map_err(|_| poisoned())?;
        let mut listed: Vec<(u64, PreparedTransaction)> = t
            .prepared_gids
            .values()
            .filter_map(|&txn| {
                let meta = t.txns.get(&txn)?.prepared.as_ref()?;
                Some((
                    meta.seq,
                    PreparedTransaction {
                        gid: meta.gid.clone(),
                        txn: TxnId(txn),
                        owner: meta.owner.clone(),
                        database: meta.database.clone(),
                        prepared_at_unix_ms: meta.prepared_at_ms,
                    },
                ))
            })
            .collect();
        listed.sort_by_key(|(seq, _)| *seq);
        Ok(listed.into_iter().map(|(_, p)| p).collect())
    }

    /// The canonical records of `txn`'s changes: its undo list read forward, each row change
    /// carrying the row's value right after it (the next change's saved old value, or the row's
    /// current version). Refused for anything but row and index changes of durable tables.
    #[allow(
        clippy::too_many_lines,
        reason = "the refusal checks, the next-change map and the one-arm-per-change translation \
                  share the undo list being read"
    )]
    fn program_of(&self, cat: &Catalog, txn: u64, undo: &[UndoOp]) -> Result<Vec<LoggedOp>> {
        for op in undo {
            let durable = match op {
                UndoOp::Inserted { table, .. }
                | UndoOp::Updated { table, .. }
                | UndoOp::Deleted { table, .. } => cat.table_is_durable(*table),
                UndoOp::IndexInserted { index, .. } | UndoOp::IndexDeleted { index, .. } => {
                    cat.index_is_durable(*index)
                },
                _ => {
                    return Err(refused(
                        std::io::ErrorKind::Unsupported,
                        "PREPARE TRANSACTION of a transaction that changed the schema, a sequence \
                         or table statistics"
                            .to_owned(),
                    ));
                },
            };
            if !durable {
                return Err(refused(
                    std::io::ErrorKind::Unsupported,
                    "PREPARE TRANSACTION of a transaction that has operated on temporary objects"
                        .to_owned(),
                ));
            }
        }
        // For each row change, the position of the next change to the same row.
        let mut next: Vec<Option<usize>> = vec![None; undo.len()];
        let mut later: HashMap<(u64, u64), usize> = HashMap::new();
        for (i, op) in undo.iter().enumerate().rev() {
            if let (Some(row), Some(slot)) = (row_of(op), next.get_mut(i)) {
                *slot = later.insert(row, i);
            }
        }
        let after = |i: usize, table: u64, row_id: u64| -> Result<Vec<u8>> {
            let encoded = match next.get(i).copied().flatten().and_then(|j| undo.get(j)) {
                Some(UndoOp::Updated { old, .. } | UndoOp::Deleted { old, .. }) => old.clone(),
                Some(_) => return Err(corrupt_row(row_id)),
                None => {
                    let t = cat
                        .tables
                        .get(&table)
                        .ok_or_else(|| table_not_found(TableId(table)))?;
                    ClusteredTree::open(&*self.store, t.root_id())
                        .get(row_id)?
                        .ok_or_else(|| corrupt_row(row_id))?
                },
            };
            let (_, tuple) = mvcc::decode_row(&encoded).ok_or_else(|| corrupt_row(row_id))?;
            Ok(tuple.to_vec())
        };
        let mut program = Vec::with_capacity(undo.len());
        for (i, op) in undo.iter().enumerate() {
            // A change to a table or index that is gone has nothing left to apply to. DDL that
            // would remove one under a prepared transaction is refused (`lock_table_for_ddl`); this
            // keeps a checkpoint working even if that guard is ever bypassed.
            let gone = match op {
                UndoOp::Inserted { table, .. }
                | UndoOp::Updated { table, .. }
                | UndoOp::Deleted { table, .. } => !cat.tables.contains_key(table),
                UndoOp::IndexInserted { index, .. } | UndoOp::IndexDeleted { index, .. } => {
                    !cat.indexes.contains_key(index)
                },
                _ => false,
            };
            if gone {
                tracing::warn!(
                    txn,
                    "a prepared transaction's change names a dropped object; skipped"
                );
                continue;
            }
            program.push(match op {
                UndoOp::Inserted { table, row_id } => LoggedOp::Insert {
                    txn,
                    table: *table,
                    row_id: *row_id,
                    tuple: after(i, *table, *row_id)?,
                },
                UndoOp::Updated { table, row_id, .. } => LoggedOp::Update {
                    txn,
                    table: *table,
                    row_id: *row_id,
                    tuple: after(i, *table, *row_id)?,
                },
                UndoOp::Deleted { table, row_id, .. } => LoggedOp::Delete {
                    txn,
                    table: *table,
                    row_id: *row_id,
                },
                UndoOp::IndexInserted {
                    index, key, row_id, ..
                } => LoggedOp::IndexInsert {
                    txn,
                    index: *index,
                    row_id: *row_id,
                    key: key.clone(),
                },
                UndoOp::IndexDeleted {
                    index, key, row_id, ..
                } => LoggedOp::IndexDelete {
                    txn,
                    index: *index,
                    row_id: *row_id,
                    key: key.clone(),
                },
                _ => return Err(bad_program(txn)),
            });
        }
        Ok(program)
    }

    /// Apply one canonical record of prepared transaction `txn` as the live write would: an
    /// uncommitted version, the old one parked, an undo entry pushed. Nothing is logged (the
    /// records are already on disk) and no lock is taken (the transaction's own are restored
    /// whole), and no admission check runs: nothing else can have written these rows.
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-record dispatcher mirroring the live write paths"
    )]
    fn redo(&self, cat: &Catalog, txn: u64, op: &LoggedOp) -> Result<()> {
        // A record naming a table or index the committed history has since dropped is skipped
        // rather than allowed to fail the open: such a drop is refused while the transaction is
        // prepared, so only a log written before that guard existed can hold one, and the
        // database must still open.
        let gone = match op {
            LoggedOp::Insert { table, .. }
            | LoggedOp::Update { table, .. }
            | LoggedOp::Delete { table, .. } => !cat.tables.contains_key(table),
            LoggedOp::IndexInsert { index, .. } | LoggedOp::IndexDelete { index, .. } => {
                !cat.indexes.contains_key(index)
            },
            _ => false,
        };
        if gone {
            tracing::warn!(
                txn,
                "a prepared transaction's record names a dropped object; skipped"
            );
            return Ok(());
        }
        match op {
            LoggedOp::Insert {
                table,
                row_id,
                tuple,
                ..
            } => {
                let t = cat
                    .tables
                    .get(table)
                    .ok_or_else(|| table_not_found(TableId(*table)))?;
                let mut w = t.write.lock().map_err(|_| poisoned())?;
                let mut tree = ClusteredTree::open(&*self.store, t.root_id());
                tree.insert(*row_id, &mvcc::encode_row(RowMeta::fresh(txn), tuple))?;
                t.set_root(tree.root());
                w.next_row_id = w.next_row_id.max(*row_id + 1);
                drop(w);
                self.push_undo(
                    txn,
                    UndoOp::Inserted {
                        table: *table,
                        row_id: *row_id,
                    },
                )
            },
            LoggedOp::Update {
                table,
                row_id,
                tuple,
                ..
            } => {
                let t = cat
                    .tables
                    .get(table)
                    .ok_or_else(|| table_not_found(TableId(*table)))?;
                let w = t.write.lock().map_err(|_| poisoned())?;
                let mut tree = ClusteredTree::open(&*self.store, t.root_id());
                let old = tree
                    .get(*row_id)?
                    .ok_or_else(|| tuple_not_found(tid_of(*row_id)))?;
                let (old_meta, old_tuple) =
                    mvcc::decode_row(&old).ok_or_else(|| corrupt_row(*row_id))?;
                let parked = UndoVersion {
                    meta: old_meta,
                    tuple: old_tuple.to_vec(),
                };
                let undo_idx = {
                    let mut undo = self.reclaim.write().map_err(|_| poisoned())?;
                    if let Some(i) = undo.free.pop() {
                        if let Some(slot) =
                            undo.arena.get_mut(usize::try_from(i).unwrap_or(usize::MAX))
                        {
                            *slot = Some(parked);
                        }
                        i
                    } else {
                        undo.arena.push(Some(parked));
                        u64::try_from(undo.arena.len().saturating_sub(1)).unwrap_or(mvcc::NO_UNDO)
                    }
                };
                let value = mvcc::encode_row(
                    RowMeta {
                        xmin: txn,
                        xmax: mvcc::NO_XMAX,
                        undo: undo_idx,
                    },
                    tuple,
                );
                let retired = tree.update(*row_id, &value)?;
                t.set_root(tree.root());
                drop(w);
                self.retire_pages(txn, retired)?;
                self.push_undo(
                    txn,
                    UndoOp::Updated {
                        table: *table,
                        row_id: *row_id,
                        old,
                        undo_idx,
                    },
                )
            },
            LoggedOp::Delete { table, row_id, .. } => {
                let t = cat
                    .tables
                    .get(table)
                    .ok_or_else(|| table_not_found(TableId(*table)))?;
                let w = t.write.lock().map_err(|_| poisoned())?;
                let tree = ClusteredTree::open(&*self.store, t.root_id());
                let old = tree
                    .get(*row_id)?
                    .ok_or_else(|| tuple_not_found(tid_of(*row_id)))?;
                let (meta, _) = mvcc::decode_row(&old).ok_or_else(|| corrupt_row(*row_id))?;
                let header = mvcc::encode_row(
                    RowMeta {
                        xmin: meta.xmin,
                        xmax: txn,
                        undo: meta.undo,
                    },
                    &[],
                );
                tree.update_prefix(*row_id, &header)?;
                t.set_root(tree.root());
                drop(w);
                self.push_undo(
                    txn,
                    UndoOp::Deleted {
                        table: *table,
                        row_id: *row_id,
                        old,
                    },
                )
            },
            LoggedOp::IndexInsert {
                index, row_id, key, ..
            } => {
                let idx = cat
                    .indexes
                    .get(index)
                    .ok_or_else(|| index_not_found(nusadb_core::IndexId(*index)))?;
                let applied = idx.data.write().map_err(|_| poisoned())?.apply_insert(
                    &self.store,
                    key,
                    *row_id,
                    txn,
                )?;
                if let super::AppliedInsert::Inserted { stamped } = applied {
                    self.push_undo(
                        txn,
                        UndoOp::IndexInserted {
                            index: *index,
                            key: key.clone(),
                            row_id: *row_id,
                            stamped,
                        },
                    )?;
                }
                Ok(())
            },
            LoggedOp::IndexDelete {
                index, row_id, key, ..
            } => {
                let idx = cat
                    .indexes
                    .get(index)
                    .ok_or_else(|| index_not_found(nusadb_core::IndexId(*index)))?;
                let removed = idx.data.write().map_err(|_| poisoned())?.apply_delete(
                    &self.store,
                    key,
                    *row_id,
                )?;
                if let Some(meta) = removed {
                    self.push_undo(
                        txn,
                        UndoOp::IndexDeleted {
                            index: *index,
                            key: key.clone(),
                            row_id: *row_id,
                            meta,
                        },
                    )?;
                }
                Ok(())
            },
            _ => Err(bad_program(txn)),
        }
    }

    /// Set every prepared transaction aside for a checkpoint (see the module docs): take its
    /// canonical records, then undo its changes in memory. Refused, with nothing changed, while
    /// any transaction is still running. The caller holds the commit gate and the catalog
    /// exclusively.
    pub(super) fn park_prepared(&self, cat: &Catalog) -> Result<Vec<Parked>> {
        let taken: Vec<(u64, PreparedMeta, Vec<PreparedLock>, Vec<UndoOp>)> = {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            if t.running() > 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    format!(
                        "nusadb-btree: checkpoint requires a quiesced engine, {} transaction(s) \
                         still active",
                        t.running()
                    ),
                )));
            }
            let mut order: Vec<(u64, u64)> = t
                .prepared_gids
                .values()
                .filter_map(|&txn| Some((t.txns.get(&txn)?.prepared.as_ref()?.seq, txn)))
                .collect();
            order.sort_unstable();
            let mut taken = Vec::with_capacity(order.len());
            for (_, txn) in order {
                let locks = held_locks(&t, txn);
                let Some(state) = t.txns.get_mut(&txn) else {
                    continue;
                };
                let Some(meta) = state.prepared.as_deref().cloned() else {
                    continue;
                };
                taken.push((txn, meta, locks, std::mem::take(&mut state.undo)));
            }
            taken
        };
        if taken.is_empty() {
            return Ok(Vec::new());
        }
        // Every program is read before anything is undone.
        let programs: Vec<Result<Vec<LoggedOp>>> = taken
            .iter()
            .map(|(txn, _, _, undo)| self.program_of(cat, *txn, undo))
            .collect();
        if let Some(position) = programs.iter().position(Result::is_err) {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            for (txn, _, _, undo) in taken {
                if let Some(state) = t.txns.get_mut(&txn) {
                    state.undo = undo;
                }
            }
            drop(t);
            return match programs.into_iter().nth(position) {
                Some(Err(e)) => Err(e),
                _ => Err(poisoned()),
            };
        }
        let mut parked = Vec::with_capacity(taken.len());
        let mut undos = Vec::with_capacity(taken.len());
        for ((txn, meta, locks, undo), program) in taken.into_iter().zip(programs) {
            parked.push(Parked {
                txn,
                meta,
                locks,
                program: program?,
            });
            undos.push((txn, undo));
        }
        // Newest first, like any rollback. A failure leaves memory without part of a prepared
        // transaction: it stops the engine, and a restart rebuilds it from disk.
        for (txn, undo) in undos.into_iter().rev() {
            let undone = self.undo_ops(&mut CatalogRef::Read(cat), txn, undo);
            self.guarded(undone)?;
        }
        Ok(parked)
    }

    /// Apply every parked transaction again from its records, after a checkpoint.
    pub(super) fn unpark_prepared(&self, cat: &Catalog, parked: Vec<Parked>) -> Result<()> {
        for p in parked {
            for op in &p.program {
                self.redo(cat, p.txn, op)?;
            }
        }
        Ok(())
    }

    /// Set up again, at the end of a replay, the prepared transactions the records leave waiting:
    /// each re-enters the active set holding its locks, then its prepare records are applied.
    pub(super) fn restore_prepared(&self, cat: &Catalog, waiting: Vec<Recovered>) -> Result<()> {
        if waiting.is_empty() {
            return Ok(());
        }
        {
            let mut t = self.txns.lock().map_err(|_| poisoned())?;
            for r in &waiting {
                let seq = t.next_prepare_seq;
                t.next_prepare_seq += 1;
                t.active.insert(r.txn);
                t.txns.insert(
                    r.txn,
                    TxnState {
                        undo: Vec::new(),
                        savepoints: Vec::new(),
                        level: IsolationLevel::ReadCommitted,
                        // Replaced below, once every one of them is active.
                        pinned: ReadView {
                            own: r.txn,
                            active: HashSet::new(),
                            horizon: 0,
                        },
                        reads: HashSet::new(),
                        predicate_reads: HashSet::new(),
                        locks: Vec::new(),
                        write_versions_at_begin: HashMap::new(),
                        write_bytes: 0,
                        fresh_reads: None,
                        prepared: Some(Box::new(PreparedMeta {
                            gid: r.gid.clone(),
                            owner: r.owner.clone(),
                            database: r.database.clone(),
                            prepared_at_ms: r.prepared_at_ms,
                            seq,
                        })),
                    },
                );
                t.prepared_gids.insert(r.gid.clone(), r.txn);
                for &lock in &r.locks {
                    let (id, exclusive) = lock_id_of(lock);
                    t.acquire_lock(r.txn, id, exclusive).map_err(|_| {
                        Error::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "nusadb-btree: prepared transaction {} holds a lock another \
                                 prepared transaction holds too",
                                r.txn
                            ),
                        ))
                    })?;
                }
            }
            for r in &waiting {
                let view = t.fresh_view(r.txn);
                if let Some(state) = t.txns.get_mut(&r.txn) {
                    state.pinned = view;
                }
            }
        }
        for r in waiting {
            for op in &r.ops {
                self.redo(cat, r.txn, op)?;
            }
            self.txns
                .lock()
                .map_err(|_| poisoned())?
                .bump_staged_versions(r.txn);
        }
        Ok(())
    }

    /// On a standby, end in memory a transaction an earlier apply left prepared, where its commit
    /// (`commit`) or abort marker stands in the records being applied. The caller holds the
    /// catalog exclusively.
    pub(super) fn finish_applied_prepared(
        &self,
        cat: &mut Catalog,
        txn: u64,
        commit: bool,
    ) -> Result<()> {
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        t.prepared_gids.retain(|_, id| *id != txn);
        t.bump_finished_versions(txn);
        if commit {
            let deltas =
                Self::finish_commit(&mut t, TxnId(txn), &self.data_version, &self.txn_ended);
            drop(t);
            fold_deltas(cat, &deltas);
            return Ok(());
        }
        let Some(state) = t.txns.remove(&txn) else {
            return Ok(());
        };
        drop(t);
        self.undo_ops(&mut CatalogRef::Write(cat), txn, state.undo)?;
        let mut t = self.txns.lock().map_err(|_| poisoned())?;
        t.active.remove(&txn);
        t.release_locks(txn, &state.locks);
        if t.admission_paused > 0 {
            self.txn_ended.notify_all();
        }
        Ok(())
    }
}
