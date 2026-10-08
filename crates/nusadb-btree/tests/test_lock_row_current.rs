//! `lock_row_current`: once the row lock is held, report whether the version the caller's snapshot
//! read is still the row's newest, or whether a transaction that committed after that snapshot
//! updated or deleted it while the lock was free.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, LockedRow, RowLockMode, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId, Tid};

fn table(engine: &BtreeEngine) -> (TableId, Tid) {
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            txn,
            &TableDef {
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: vec![ColumnDef {
                    name: "v".to_owned(),
                    ty: ColumnType::Bytes,
                    nullable: false,
                }],
            },
        )
        .unwrap();
    let tid = engine.insert(txn, table, b"queued").unwrap();
    engine.commit(txn).unwrap();
    (table, tid)
}

fn open() -> (tempfile::TempDir, BtreeEngine) {
    let dir = tempfile::tempdir().unwrap();
    let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    (dir, engine)
}

#[test]
fn an_untouched_row_is_unchanged() {
    let (_dir, engine) = open();
    let (table, tid) = table(&engine);
    for level in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::RepeatableRead,
    ] {
        let txn = engine.begin(level).unwrap();
        engine.begin_statement(txn).unwrap();
        let got = engine
            .lock_row_current(txn, table, tid, RowLockMode::Exclusive)
            .unwrap();
        assert_eq!(got, LockedRow::Unchanged, "{level:?}");
        engine.rollback(txn).unwrap();
    }
}

#[test]
fn an_update_committed_after_the_snapshot_is_reported_with_the_newest_version() {
    for level in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::RepeatableRead,
    ] {
        let (_dir, engine) = open();
        let (table, tid) = table(&engine);
        let reader = engine.begin(level).unwrap();
        engine.begin_statement(reader).unwrap(); // the snapshot the reader scanned under
        let writer = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        engine.update(writer, table, tid, b"done").unwrap();
        engine.commit(writer).unwrap();
        let got = engine
            .lock_row_current(reader, table, tid, RowLockMode::Exclusive)
            .unwrap();
        assert_eq!(got, LockedRow::Updated(b"done".to_vec()), "{level:?}");
        engine.rollback(reader).unwrap();
    }
}

#[test]
fn a_delete_committed_after_the_snapshot_is_reported() {
    let (_dir, engine) = open();
    let (table, tid) = table(&engine);
    let reader = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.begin_statement(reader).unwrap();
    let writer = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.delete(writer, table, tid).unwrap();
    engine.commit(writer).unwrap();
    let got = engine
        .lock_row_current(reader, table, tid, RowLockMode::Exclusive)
        .unwrap();
    assert_eq!(got, LockedRow::Deleted);
    engine.rollback(reader).unwrap();
}

#[test]
fn a_change_the_snapshot_already_sees_is_not_a_change() {
    let (_dir, engine) = open();
    let (table, tid) = table(&engine);
    let writer = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.update(writer, table, tid, b"done").unwrap();
    engine.commit(writer).unwrap();
    // A statement that starts after the commit reads the new version: nothing changed since.
    let reader = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.begin_statement(reader).unwrap();
    let got = engine
        .lock_row_current(reader, table, tid, RowLockMode::Exclusive)
        .unwrap();
    assert_eq!(got, LockedRow::Unchanged);
    // Nor is the reader's own update.
    engine.update(reader, table, tid, b"mine").unwrap();
    let got = engine
        .lock_row_current(reader, table, tid, RowLockMode::Exclusive)
        .unwrap();
    assert_eq!(got, LockedRow::Unchanged);
    engine.rollback(reader).unwrap();
}

#[test]
fn a_lock_held_by_another_transaction_is_a_conflict() {
    let (_dir, engine) = open();
    let (table, tid) = table(&engine);
    let holder = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine
        .lock_row_current(holder, table, tid, RowLockMode::Exclusive)
        .unwrap();
    let other = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.begin_statement(other).unwrap();
    assert!(matches!(
        engine.lock_row_current(other, table, tid, RowLockMode::Exclusive),
        Err(nusadb_core::Error::SerializationConflict { .. })
    ));
    engine.rollback(other).unwrap();
    engine.rollback(holder).unwrap();
}

#[test]
fn a_writer_that_rolled_back_left_nothing_changed() {
    let (_dir, engine) = open();
    let (table, tid) = table(&engine);
    let reader = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.begin_statement(reader).unwrap();
    let updater = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.update(updater, table, tid, b"done").unwrap();
    engine.rollback(updater).unwrap();
    let deleter = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.delete(deleter, table, tid).unwrap();
    engine.rollback(deleter).unwrap();
    let got = engine
        .lock_row_current(reader, table, tid, RowLockMode::Exclusive)
        .unwrap();
    assert_eq!(got, LockedRow::Unchanged);
    engine.rollback(reader).unwrap();
}
