//! Two-phase commit: a prepared transaction keeps its writes invisible and its locks held until
//! `COMMIT PREPARED` or `ROLLBACK PREPARED`, from any session, and stays prepared across a crash,
//! a clean restart, checkpoints that truncate the log, and a standby's apply.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::io::ErrorKind;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::time::Duration;

use nusadb_btree::{BtreeEngine, ShipOutcome, seed_standby, shipped_segments_after};
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, RowLockMode, TableDef};
use nusadb_core::{ColumnType, Error, IndexId, IsolationLevel, StorageEngine, TableId, Tid};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def() -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: "t".to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

fn wal_of(dir: &Path) -> PathBuf {
    dir.join("btree.wal")
}

/// A durable engine with table `t` and an index `t_v` over its values.
fn setup(dir: &Path) -> (BtreeEngine, TableId, IndexId) {
    let engine = BtreeEngine::open(wal_of(dir)).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    let index = engine
        .create_index(
            txn,
            &IndexDef {
                name: "t_v".to_owned(),
                table,
                columns: vec!["v".to_owned()],
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: false,
            },
        )
        .unwrap();
    engine.commit(txn).unwrap();
    (engine, table, index)
}

/// Insert `value` (row and index entry) in its own committed transaction.
fn insert(engine: &BtreeEngine, table: TableId, index: IndexId, value: &[u8]) -> Tid {
    let txn = engine.begin(RC).unwrap();
    let tid = engine.insert(txn, table, value).unwrap();
    engine.index_insert(txn, index, value, tid).unwrap();
    engine.commit(txn).unwrap();
    tid
}

/// Every value a fresh reader sees, sorted.
fn values(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    engine.commit(txn).unwrap();
    out.sort();
    out
}

/// Every value a fresh reader finds through the index, sorted.
fn indexed(engine: &BtreeEngine, index: IndexId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine
        .index_scan(txn, index, Bound::Unbounded, Bound::Unbounded)
        .unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    engine.commit(txn).unwrap();
    out.sort();
    out
}

fn sorted(values: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = values.iter().map(|v| v.to_vec()).collect();
    out.sort();
    out
}

fn gids(engine: &BtreeEngine) -> Vec<String> {
    engine
        .list_prepared()
        .unwrap()
        .into_iter()
        .map(|p| p.gid)
        .collect()
}

fn io_kind(error: &Error) -> Option<ErrorKind> {
    match error {
        Error::Io(e) => Some(e.kind()),
        _ => None,
    }
}

/// Copy the directory tree `from` into `to`, leaving out lock files: the files as a `kill -9`
/// leaves them, while the engine that wrote them is still running.
fn copy_tree(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".lock") {
            continue;
        }
        let target = to.join(&name);
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The directory as a crash right now would leave it, opened by a new process.
fn crash_and_reopen(dir: &Path) -> (tempfile::TempDir, BtreeEngine) {
    let copy = tempfile::tempdir().unwrap();
    copy_tree(dir, copy.path());
    let engine = BtreeEngine::open(wal_of(copy.path())).unwrap();
    (copy, engine)
}

/// The prepared transaction the tests share: over the committed values `a`, `b`, `c`, it
/// inserts `d`, updates `b` to `B` (moving its index entry), deletes `c`, and leaves a savepoint
/// it rolled back to behind. Returns the row ids of `a`, `b` and `c`.
fn prepare_mix(engine: &BtreeEngine, table: TableId, index: IndexId, gid: &str) -> [Tid; 3] {
    let row_a = insert(engine, table, index, b"a");
    let row_b = insert(engine, table, index, b"b");
    let row_c = insert(engine, table, index, b"c");
    let txn = engine.begin(RC).unwrap();
    let row_d = engine.insert(txn, table, b"d").unwrap();
    engine.index_insert(txn, index, b"d", row_d).unwrap();
    engine.update(txn, table, row_b, b"B").unwrap();
    engine.index_insert(txn, index, b"B", row_b).unwrap();
    engine.delete(txn, table, row_c).unwrap();
    engine.savepoint(txn, "sp").unwrap();
    // Undone before the prepare: none of this may come back.
    let row_e = engine.insert(txn, table, b"e").unwrap();
    engine.index_insert(txn, index, b"e", row_e).unwrap();
    engine.update(txn, table, row_a, b"A").unwrap();
    engine.index_insert(txn, index, b"A", row_a).unwrap();
    engine.rollback_to(txn, "sp").unwrap();
    // A row lock on a row it did not change.
    engine
        .lock_row(txn, table, row_a, RowLockMode::Exclusive)
        .unwrap();
    engine
        .prepare_transaction(txn, gid, "alice", "db1")
        .unwrap();
    [row_a, row_b, row_c]
}

const BEFORE: &[&[u8]] = &[b"a", b"b", b"c"];
const AFTER: &[&[u8]] = &[b"B", b"a", b"d"];

/// What every other transaction sees of a prepared `prepare_mix`, and what it may not do.
fn assert_still_prepared(engine: &BtreeEngine, table: TableId, index: IndexId, rows: [Tid; 3]) {
    assert_eq!(gids(engine), vec!["g1".to_owned()]);
    let listed = engine.list_prepared().unwrap().remove(0);
    assert_eq!(
        (listed.owner.as_str(), listed.database.as_str()),
        ("alice", "db1")
    );
    assert_eq!(values(engine, table), sorted(BEFORE));
    assert_eq!(indexed(engine, index), sorted(BEFORE));
    // Its locks hold: a writer of a row it changed, or of the row it only locked, is refused.
    for row in rows {
        let txn = engine.begin(RC).unwrap();
        let refused = engine.update(txn, table, row, b"x").unwrap_err();
        assert!(
            matches!(refused, Error::SerializationConflict { .. }),
            "{refused:?}"
        );
        engine.rollback(txn).unwrap();
    }
    let txn = engine.begin(RC).unwrap();
    let refused = engine
        .lock_row(txn, table, rows[0], RowLockMode::Shared)
        .unwrap_err();
    assert!(
        matches!(refused, Error::SerializationConflict { .. }),
        "{refused:?}"
    );
    engine.rollback(txn).unwrap();
}

#[test]
fn a_prepared_transaction_is_invisible_and_locked_until_commit_prepared() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    assert_still_prepared(&engine, table, index, rows);
    engine.commit_prepared("g1", None).unwrap();
    assert!(gids(&engine).is_empty());
    assert_eq!(values(&engine, table), sorted(AFTER));
    assert_eq!(indexed(&engine, index), sorted(AFTER));
    // The locks went with it.
    let txn = engine.begin(RC).unwrap();
    engine.update(txn, table, rows[0], b"a2").unwrap();
    engine.commit(txn).unwrap();
    // And a clean restart keeps exactly that.
    drop(engine);
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"B", b"a2", b"d"]));
    assert!(gids(&engine).is_empty());
}

#[test]
fn rollback_prepared_discards_the_changes_and_releases_the_locks() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    engine.rollback_prepared("g1", None).unwrap();
    assert!(gids(&engine).is_empty());
    assert_eq!(values(&engine, table), sorted(BEFORE));
    assert_eq!(indexed(&engine, index), sorted(BEFORE));
    for row in rows {
        let txn = engine.begin(RC).unwrap();
        engine
            .lock_row(txn, table, row, RowLockMode::Exclusive)
            .unwrap();
        engine.rollback(txn).unwrap();
    }
    drop(engine);
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_eq!(values(&engine, table), sorted(BEFORE));
    assert!(gids(&engine).is_empty());
}

#[test]
fn a_prepared_transaction_survives_a_crash_and_commits_after_it() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    let (copy, recovered) = crash_and_reopen(dir.path());
    assert_still_prepared(&recovered, table, index, rows);
    recovered.commit_prepared("g1", None).unwrap();
    assert_eq!(values(&recovered, table), sorted(AFTER));
    assert_eq!(indexed(&recovered, index), sorted(AFTER));
    // The commit is as durable as any: a second crash keeps it.
    let (_again, twice) = crash_and_reopen(copy.path());
    assert_eq!(values(&twice, table), sorted(AFTER));
    assert!(gids(&twice).is_empty());
}

#[test]
fn a_prepared_transaction_survives_a_crash_and_rolls_back_after_it() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    let (copy, recovered) = crash_and_reopen(dir.path());
    assert_still_prepared(&recovered, table, index, rows);
    recovered.rollback_prepared("g1", None).unwrap();
    assert_eq!(values(&recovered, table), sorted(BEFORE));
    assert_eq!(indexed(&recovered, index), sorted(BEFORE));
    let (_again, twice) = crash_and_reopen(copy.path());
    assert_eq!(values(&twice, table), sorted(BEFORE));
    assert!(gids(&twice).is_empty());
    drop(engine);
}

#[test]
fn a_clean_restart_keeps_a_prepared_transaction_prepared() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    drop(engine);
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_still_prepared(&engine, table, index, rows);
    drop(engine);
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_still_prepared(&engine, table, index, rows);
    engine.commit_prepared("g1", None).unwrap();
    assert_eq!(values(&engine, table), sorted(AFTER));
}

#[test]
fn a_checkpoint_truncates_the_log_and_carries_the_prepared_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    let before = engine.wal_len().unwrap().unwrap();
    // A prepared transaction does not hold a checkpoint off.
    engine.checkpoint().unwrap();
    assert_eq!(engine.wal_len().unwrap().unwrap(), 0, "was {before}");
    // In memory it is exactly as it was.
    assert_still_prepared(&engine, table, index, rows);
    // Committed work after the checkpoint lands in the new log.
    insert(&engine, table, index, b"z");
    let (copy, recovered) = crash_and_reopen(dir.path());
    assert_eq!(values(&recovered, table), sorted(&[b"a", b"b", b"c", b"z"]));
    assert_eq!(gids(&recovered), vec!["g1".to_owned()]);
    recovered.commit_prepared("g1", None).unwrap();
    assert_eq!(values(&recovered, table), sorted(&[b"B", b"a", b"d", b"z"]));
    assert_eq!(
        indexed(&recovered, index),
        sorted(&[b"B", b"a", b"d", b"z"])
    );
    let (_again, twice) = crash_and_reopen(copy.path());
    assert_eq!(values(&twice, table), sorted(&[b"B", b"a", b"d", b"z"]));
    // The original engine commits it too, from its own memory.
    engine.commit_prepared("g1", None).unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"B", b"a", b"d", b"z"]));
}

#[test]
fn repeated_checkpoints_and_restarts_never_lose_a_prepared_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let rows = prepare_mix(&engine, table, index, "g1");
    engine.checkpoint().unwrap();
    engine.checkpoint().unwrap();
    drop(engine);
    // A restart replays the image; checkpoints after it carry the transaction again.
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_still_prepared(&engine, table, index, rows);
    engine.checkpoint().unwrap();
    let (_copy, recovered) = crash_and_reopen(dir.path());
    assert_still_prepared(&recovered, table, index, rows);
    recovered.rollback_prepared("g1", None).unwrap();
    assert_eq!(values(&recovered, table), sorted(BEFORE));
    assert_eq!(indexed(&recovered, index), sorted(BEFORE));
    // Finished after a checkpoint, then checkpointed and reopened: the end is final.
    engine.commit_prepared("g1", None).unwrap();
    engine.checkpoint().unwrap();
    drop(engine);
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    assert_eq!(values(&engine, table), sorted(AFTER));
    assert!(gids(&engine).is_empty());
}

#[test]
fn a_commit_prepared_marker_after_a_checkpoint_replays_the_image_records() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    prepare_mix(&engine, table, index, "g1");
    engine.checkpoint().unwrap();
    engine.commit_prepared("g1", None).unwrap();
    // The image holds the prepare records, the log only the commit marker.
    let (_copy, recovered) = crash_and_reopen(dir.path());
    assert_eq!(values(&recovered, table), sorted(AFTER));
    assert_eq!(indexed(&recovered, index), sorted(AFTER));
    assert!(gids(&recovered).is_empty());
}

#[test]
fn a_prepare_cut_short_by_a_crash_leaves_the_transaction_uncommitted() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    prepare_mix(&engine, table, index, "g1");
    let copy = tempfile::tempdir().unwrap();
    copy_tree(dir.path(), copy.path());
    // Cut the log just before its last record, the prepare marker, on a record boundary.
    let wal = wal_of(copy.path());
    let bytes = std::fs::read(&wal).unwrap();
    let full = nusadb_wal::recover_prefix(&bytes).unwrap();
    let cut = (1..bytes.len())
        .rev()
        .find(|&len| {
            nusadb_wal::recover_prefix(&bytes[..len]).is_ok_and(|prefix| {
                prefix.good_bytes as usize == len && prefix.records.len() == full.records.len() - 1
            })
        })
        .unwrap();
    std::fs::write(&wal, &bytes[..cut]).unwrap();
    let recovered = BtreeEngine::open(&wal).unwrap();
    assert!(gids(&recovered).is_empty());
    assert_eq!(values(&recovered, table), sorted(BEFORE));
    assert_eq!(indexed(&recovered, index), sorted(BEFORE));
    // Nothing of it holds a lock either.
    let txn = recovered.begin(RC).unwrap();
    let tid = recovered.insert(txn, table, b"free").unwrap();
    recovered.commit(txn).unwrap();
    assert!(values(&recovered, table).contains(&b"free".to_vec()));
    let _ = tid;
}

#[test]
fn prepare_refusals_roll_the_transaction_back() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let first = engine.begin(RC).unwrap();
    engine.insert(first, table, b"one").unwrap();
    engine
        .prepare_transaction(first, "dup", "alice", "db")
        .unwrap();

    // A gid already in use.
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"two").unwrap();
    let e = engine
        .prepare_transaction(txn, "dup", "alice", "db")
        .unwrap_err();
    assert_eq!(io_kind(&e), Some(ErrorKind::AlreadyExists), "{e:?}");
    assert!(
        engine.commit(txn).is_err(),
        "the refused transaction is gone"
    );

    // A gid longer than the limit.
    let txn = engine.begin(RC).unwrap();
    let e = engine
        .prepare_transaction(txn, &"x".repeat(201), "alice", "db")
        .unwrap_err();
    assert_eq!(io_kind(&e), Some(ErrorKind::InvalidData), "{e:?}");

    // A schema change cannot be prepared.
    let txn = engine.begin(RC).unwrap();
    let mut def = table_def();
    def.name = "u".to_owned();
    engine.create_table(txn, &def).unwrap();
    let e = engine
        .prepare_transaction(txn, "ddl", "alice", "db")
        .unwrap_err();
    assert_eq!(io_kind(&e), Some(ErrorKind::Unsupported), "{e:?}");
    assert!(engine.lookup_table("u").unwrap().is_none());

    // The ceiling.
    engine.set_max_prepared_transactions(1);
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"three").unwrap();
    let e = engine
        .prepare_transaction(txn, "over", "alice", "db")
        .unwrap_err();
    assert_eq!(io_kind(&e), Some(ErrorKind::QuotaExceeded), "{e:?}");
    engine.set_max_prepared_transactions(0);
    let txn = engine.begin(RC).unwrap();
    let e = engine
        .prepare_transaction(txn, "off", "alice", "db")
        .unwrap_err();
    assert_eq!(io_kind(&e), Some(ErrorKind::InvalidInput), "{e:?}");

    // A prepared transaction is not the session's to finish.
    assert_eq!(
        io_kind(&engine.commit(first).unwrap_err()),
        Some(ErrorKind::InvalidInput)
    );
    assert_eq!(
        io_kind(&engine.rollback(first).unwrap_err()),
        Some(ErrorKind::InvalidInput)
    );
    // Finishing a name nobody prepared.
    assert_eq!(
        io_kind(&engine.commit_prepared("nope", None).unwrap_err()),
        Some(ErrorKind::NotFound)
    );
    assert_eq!(
        io_kind(&engine.rollback_prepared("nope", None).unwrap_err()),
        Some(ErrorKind::NotFound)
    );
    // None of the refused writes is visible, and the one prepared write is not yet.
    assert!(values(&engine, table).is_empty());
    engine.commit_prepared("dup", None).unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"one"]));
    assert!(
        engine.commit_prepared("dup", None).is_err(),
        "finished once"
    );
    let _ = index;
}

#[test]
fn a_prepared_update_keeps_the_old_version_through_purge() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let a = insert(&engine, table, index, b"a");
    let txn = engine.begin(RC).unwrap();
    engine.update(txn, table, a, b"A").unwrap();
    engine.prepare_transaction(txn, "p", "alice", "db").unwrap();
    // Purge runs while it waits, and later transactions come and go.
    insert(&engine, table, index, b"b");
    engine.purge().unwrap();
    engine.purge().unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"a", b"b"]));
    engine.rollback_prepared("p", None).unwrap();
    engine.purge().unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"a", b"b"]));
}

#[test]
fn a_serializable_reader_of_a_prepared_write_cannot_commit_a_skew() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let a = insert(&engine, table, index, b"a");
    let b = insert(&engine, table, index, b"b");
    // The reader reads `a`, then the writer changes `a` and prepares.
    let reader = engine.begin(IsolationLevel::Serializable).unwrap();
    let mut scan = engine.scan(reader, table).unwrap();
    while scan.try_next().unwrap().is_some() {}
    drop(scan);
    let writer = engine.begin(IsolationLevel::Serializable).unwrap();
    engine.update(writer, table, a, b"A").unwrap();
    engine
        .prepare_transaction(writer, "w", "alice", "db")
        .unwrap();
    // The reader writes `b` and commits: the prepared writer counts as committed, so the
    // reader's antidependency on it aborts the reader.
    engine.update(reader, table, b, b"B").unwrap();
    let e = engine.commit(reader).unwrap_err();
    assert!(matches!(e, Error::SerializationConflict { .. }), "{e:?}");
    engine.commit_prepared("w", None).unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"A", b"b"]));
}

#[test]
fn a_standby_applies_the_prepared_transaction_and_its_end() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let primary =
        BtreeEngine::open_with_archive(wal_of(dir.path()), Some(archive.path().to_path_buf()))
            .unwrap();
    let txn = primary.begin(RC).unwrap();
    let table = primary.create_table(txn, &table_def()).unwrap();
    primary.commit(txn).unwrap();
    let txn = primary.begin(RC).unwrap();
    let a = primary.insert(txn, table, b"a").unwrap();
    primary.commit(txn).unwrap();
    primary.checkpoint().unwrap();
    let standby_dir = tempfile::tempdir().unwrap();
    let standby_wal = wal_of(standby_dir.path());
    seed_standby(archive.path(), &standby_wal).unwrap();
    let standby = BtreeEngine::open_standby(&standby_wal).unwrap();
    let catch_up = |standby: &BtreeEngine| {
        let applied = standby.wal_last_lsn().unwrap().unwrap();
        for (_, path) in shipped_segments_after(archive.path(), applied).unwrap() {
            let bytes = std::fs::read(path).unwrap();
            assert!(matches!(
                standby
                    .apply_shipped_segment(&bytes, Duration::from_secs(2))
                    .unwrap(),
                ShipOutcome::Applied { .. } | ShipOutcome::NothingNew
            ));
        }
    };

    // Prepared on the primary, shipped by a checkpoint: prepared on the standby too.
    let txn = primary.begin(RC).unwrap();
    primary.update(txn, table, a, b"A").unwrap();
    primary.insert(txn, table, b"b").unwrap();
    primary
        .prepare_transaction(txn, "s1", "alice", "db")
        .unwrap();
    primary.checkpoint().unwrap();
    catch_up(&standby);
    assert_eq!(gids(&standby), vec!["s1".to_owned()]);
    assert_eq!(values(&standby, table), sorted(&[b"a"]));
    // The standby finishes nothing of its own.
    assert!(matches!(
        standby.commit_prepared("s1", None).unwrap_err(),
        Error::ReadOnly(_)
    ));
    // A standby restart keeps it prepared, and a standby checkpoint carries it.
    standby.checkpoint().unwrap();
    drop(standby);
    let standby = BtreeEngine::open_standby(&standby_wal).unwrap();
    assert_eq!(gids(&standby), vec!["s1".to_owned()]);
    // The primary commits it; the next segment carries only the marker.
    primary.commit_prepared("s1", None).unwrap();
    primary.checkpoint().unwrap();
    catch_up(&standby);
    assert!(gids(&standby).is_empty());
    assert_eq!(values(&standby, table), sorted(&[b"A", b"b"]));
    drop(standby);
    let standby = BtreeEngine::open_standby(&standby_wal).unwrap();
    assert_eq!(values(&standby, table), sorted(&[b"A", b"b"]));

    // And a rollback ships the same way.
    let txn = primary.begin(RC).unwrap();
    primary.insert(txn, table, b"gone").unwrap();
    primary
        .prepare_transaction(txn, "s2", "alice", "db")
        .unwrap();
    primary.checkpoint().unwrap();
    catch_up(&standby);
    assert_eq!(gids(&standby), vec!["s2".to_owned()]);
    primary.rollback_prepared("s2", None).unwrap();
    primary.checkpoint().unwrap();
    catch_up(&standby);
    assert!(gids(&standby).is_empty());
    assert_eq!(values(&standby, table), sorted(&[b"A", b"b"]));
    // Promoted, a standby holding a prepared transaction finishes it itself.
    let txn = primary.begin(RC).unwrap();
    primary.insert(txn, table, b"late").unwrap();
    primary
        .prepare_transaction(txn, "s3", "alice", "db")
        .unwrap();
    primary.checkpoint().unwrap();
    catch_up(&standby);
    standby.set_standby(false);
    standby.commit_prepared("s3", None).unwrap();
    assert_eq!(values(&standby, table), sorted(&[b"A", b"b", b"late"]));
}

#[test]
fn the_in_memory_engine_prepares_too() {
    let engine = BtreeEngine::new();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"m").unwrap();
    engine.prepare_transaction(txn, "m", "alice", "db").unwrap();
    assert!(values(&engine, table).is_empty());
    engine.commit_prepared("m", None).unwrap();
    assert_eq!(values(&engine, table), sorted(&[b"m"]));
}

/// Whether `result` is the refusal a prepared transaction's table answers DDL with (`55006`).
#[allow(
    clippy::needless_pass_by_value,
    reason = "takes the call's result as it comes, so each check reads as one expression"
)]
fn in_use<T: std::fmt::Debug>(result: nusadb_core::Result<T>) -> bool {
    matches!(result, Err(Error::ObjectInUse(_)))
}

/// DDL that would drop, reshape or re-index a table a prepared transaction wrote is refused while
/// it waits, so its records always find their table and indexes: a checkpoint, a crash and the
/// reopen after it all work, and so does the commit.
#[test]
fn ddl_on_a_prepared_transactions_table_is_refused_and_recovery_stays_whole() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, index) = setup(dir.path());
    let txn = engine.begin(RC).unwrap();
    let tid = engine.insert(txn, table, b"p1").unwrap();
    engine.index_insert(txn, index, b"p1", tid).unwrap();
    engine
        .prepare_transaction(txn, "g1", "alice", "db")
        .unwrap();

    let ddl = engine.begin(RC).unwrap();
    assert!(in_use(engine.drop_table(ddl, table)));
    assert!(in_use(engine.drop_index(ddl, index)));
    assert!(in_use(engine.alter_table(
        ddl,
        table,
        &nusadb_core::engine::AlterOp::AddColumn(ColumnDef {
            name: "w".to_owned(),
            ty: ColumnType::Int,
            nullable: true,
        }),
    )));
    let mut second = IndexDef {
        name: "t_v2".to_owned(),
        table,
        columns: vec!["v".to_owned()],
        key_exprs: Vec::new(),
        predicate: None,
        include: Vec::new(),
        kind: IndexKind::BTree,
        unique: true,
    };
    assert!(in_use(engine.create_index(ddl, &second)));
    assert!(in_use(engine.add_unique_constraint(
        ddl,
        table,
        "t_u",
        &["v".to_owned()],
        false,
        false
    )));
    engine.rollback(ddl).unwrap();

    engine.checkpoint().unwrap();
    let (copy, recovered) = crash_and_reopen(dir.path());
    assert_eq!(gids(&recovered), vec!["g1".to_owned()]);
    // The locks came back with it: still refused after the restart.
    let ddl = recovered.begin(RC).unwrap();
    assert!(in_use(recovered.drop_table(ddl, table)));
    recovered.rollback(ddl).unwrap();
    recovered.commit_prepared("g1", None).unwrap();
    assert_eq!(values(&recovered, table), sorted(&[b"p1"]));
    assert_eq!(indexed(&recovered, index), sorted(&[b"p1"]));
    // Once it is finished the DDL goes through, and an index built now holds the row.
    let ddl = recovered.begin(RC).unwrap();
    second.unique = false;
    recovered.create_index(ddl, &second).unwrap();
    recovered.commit(ddl).unwrap();
    let ddl = recovered.begin(RC).unwrap();
    recovered.drop_table(ddl, table).unwrap();
    recovered.commit(ddl).unwrap();
    drop(recovered);
    let reopened = BtreeEngine::open(wal_of(copy.path())).unwrap();
    assert!(reopened.lookup_table("t").unwrap().is_none());
}

/// An index build next to another transaction's unfinished writes would miss them, so it is
/// refused with the ordinary no-wait conflict while a running writer holds the table.
#[test]
fn create_index_beside_a_running_writer_is_a_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, _) = setup(dir.path());
    let writer = engine.begin(RC).unwrap();
    engine.insert(writer, table, b"w").unwrap();
    let ddl = engine.begin(RC).unwrap();
    let e = engine
        .create_index(
            ddl,
            &IndexDef {
                name: "late".to_owned(),
                table,
                columns: vec!["v".to_owned()],
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: false,
            },
        )
        .unwrap_err();
    assert!(matches!(e, Error::SerializationConflict { .. }), "{e:?}");
    engine.rollback(ddl).unwrap();
    engine.commit(writer).unwrap();
}

/// A log written before DDL was refused under a prepared transaction can hold a drop of a table the
/// transaction wrote. Recovery skips the records naming it rather than failing the open.
#[test]
fn a_prepared_transactions_dropped_table_never_stops_the_open() {
    use nusadb_btree::wal::LoggedOp;
    use nusadb_core::TxnId;
    use nusadb_wal::{WalRecord, WalWriter};

    let dir = tempfile::tempdir().unwrap();
    let wal = wal_of(dir.path());
    let mut writer = WalWriter::new(std::fs::File::create(&wal).unwrap());
    let mut keep = |op: LoggedOp| {
        writer.append(&op.to_record()).unwrap();
    };
    for (table, name) in [(0_u64, "t"), (1, "u")] {
        let mut def = table_def();
        def.name = name.to_owned();
        keep(LoggedOp::CreateTable { txn: 1, table, def });
    }
    writer
        .append(&WalRecord::CommitTxn {
            txn: TxnId(1),
            unix_ms: 1,
        })
        .unwrap();
    let mut keep = |op: LoggedOp| {
        writer.append(&op.to_record()).unwrap();
    };
    keep(LoggedOp::PrepareBegin { txn: 2 });
    for table in [0_u64, 1] {
        keep(LoggedOp::Insert {
            txn: 2,
            table,
            row_id: 0,
            tuple: b"kept".to_vec(),
        });
    }
    keep(LoggedOp::Prepare {
        txn: 2,
        gid: "old".to_owned(),
        owner: "alice".to_owned(),
        database: "db".to_owned(),
        prepared_at_ms: 1,
        locks: Vec::new(),
    });
    keep(LoggedOp::DropTable { txn: 3, table: 0 });
    writer
        .append(&WalRecord::CommitTxn {
            txn: TxnId(3),
            unix_ms: 2,
        })
        .unwrap();
    writer.flush().unwrap();
    writer.get_mut().sync_all().unwrap();
    drop(writer);

    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(gids(&engine), vec!["old".to_owned()]);
    let u = engine.lookup_table("u").unwrap().unwrap().id;
    engine.checkpoint().unwrap();
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    engine.commit_prepared("old", None).unwrap();
    assert_eq!(values(&engine, u), sorted(&[b"kept"]));
}

/// Only the role that prepared a transaction may finish it unless the caller asks as a superuser
/// (`None`); the check is made by the same step that finishes it, and a refusal leaves it prepared,
/// across a crash too.
#[test]
fn finishing_checks_the_owner_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, _) = setup(dir.path());
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"o").unwrap();
    engine
        .prepare_transaction(txn, "own", "alice", "db")
        .unwrap();
    for finish in [
        engine.commit_prepared("own", Some("bob")),
        engine.rollback_prepared("own", Some("bob")),
    ] {
        assert_eq!(
            io_kind(&finish.unwrap_err()),
            Some(ErrorKind::PermissionDenied)
        );
    }
    let (_copy, recovered) = crash_and_reopen(dir.path());
    assert_eq!(gids(&recovered), vec!["own".to_owned()]);
    assert!(recovered.commit_prepared("own", Some("bob")).is_err());
    recovered.commit_prepared("own", Some("alice")).unwrap();
    assert_eq!(values(&recovered, table), sorted(&[b"o"]));
}

/// A CHECK added to a table a prepared transaction wrote would be validated without its rows, which
/// could then commit in violation: refused, before and after a crash, and allowed once it ends.
#[test]
fn a_check_on_a_prepared_transactions_table_is_refused_across_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table, _) = setup(dir.path());
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"2").unwrap();
    engine
        .prepare_transaction(txn, "ck", "alice", "db")
        .unwrap();
    let ddl = engine.begin(RC).unwrap();
    assert!(in_use(
        engine.add_check_constraint(ddl, table, "ck", b"id < 2")
    ));
    engine.rollback(ddl).unwrap();
    engine.checkpoint().unwrap();
    let (_copy, recovered) = crash_and_reopen(dir.path());
    let ddl = recovered.begin(RC).unwrap();
    let refused = recovered.add_check_constraint(ddl, table, "ck", b"id < 2");
    assert!(in_use(refused), "after the crash");
    recovered.rollback(ddl).unwrap();
    recovered.rollback_prepared("ck", None).unwrap();
    let ddl = recovered.begin(RC).unwrap();
    recovered
        .add_check_constraint(ddl, table, "ck", b"id < 2")
        .unwrap();
    recovered.commit(ddl).unwrap();
}

/// A foreign key added over existing child rows is validated against the committed parent rows; a
/// prepared delete of a parent row could then orphan a child. The parent is locked too, so the key
/// is refused while the delete waits, across a crash as well.
#[test]
fn a_foreign_key_over_a_prepared_parent_change_is_refused_across_a_crash() {
    use nusadb_core::{FkAction, ForeignKeyDef};
    let dir = tempfile::tempdir().unwrap();
    let engine = BtreeEngine::open(wal_of(dir.path())).unwrap();
    let txn = engine.begin(RC).unwrap();
    let mut def = table_def();
    def.name = "par".to_owned();
    let par = engine.create_table(txn, &def).unwrap();
    engine
        .add_unique_constraint(txn, par, "par_pk", &["v".to_owned()], true, false)
        .unwrap();
    def.name = "ch".to_owned();
    let ch = engine.create_table(txn, &def).unwrap();
    let parent_row = engine.insert(txn, par, b"1").unwrap();
    engine.insert(txn, ch, b"1").unwrap();
    engine.commit(txn).unwrap();
    let txn = engine.begin(RC).unwrap();
    engine.delete(txn, par, parent_row).unwrap();
    engine
        .prepare_transaction(txn, "del", "alice", "db")
        .unwrap();
    let fk = ForeignKeyDef {
        name: "chfk".to_owned(),
        child_table: ch,
        child_columns: vec!["v".to_owned()],
        parent_table: par,
        parent_columns: vec!["v".to_owned()],
        on_delete: FkAction::NoAction,
        on_update: FkAction::NoAction,
    };
    let ddl = engine.begin(RC).unwrap();
    assert!(in_use(engine.add_foreign_key(ddl, &fk)));
    engine.rollback(ddl).unwrap();
    let (_copy, recovered) = crash_and_reopen(dir.path());
    let ddl = recovered.begin(RC).unwrap();
    assert!(
        in_use(recovered.add_foreign_key(ddl, &fk)),
        "after the crash"
    );
    recovered.rollback(ddl).unwrap();
    recovered.rollback_prepared("del", None).unwrap();
    let ddl = recovered.begin(RC).unwrap();
    recovered.add_foreign_key(ddl, &fk).unwrap();
    recovered.commit(ddl).unwrap();
    // An empty child does not need the parent: a key added to it beside a parent writer is fine.
    let txn = recovered.begin(RC).unwrap();
    def.name = "empty".to_owned();
    let empty = recovered.create_table(txn, &def).unwrap();
    recovered.commit(txn).unwrap();
    let writer = recovered.begin(RC).unwrap();
    recovered.insert(writer, par, b"2").unwrap();
    let ddl = recovered.begin(RC).unwrap();
    recovered
        .add_foreign_key(
            ddl,
            &ForeignKeyDef {
                name: "emptyfk".to_owned(),
                child_table: empty,
                ..fk
            },
        )
        .unwrap();
    recovered.commit(ddl).unwrap();
    recovered.commit(writer).unwrap();
}
