//! A table scan streams: it reads the tree a batch at a time rather than materializing the
//! table at open, and still behaves like a snapshot taken when it opened. Its view stays pinned
//! against purge for its whole life, it never reads rows its own transaction writes afterwards,
//! and a table dropped under it stays readable until it closes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId, Tid};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def(name: &str) -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: name.to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

fn row(i: u64, tag: u8) -> Vec<u8> {
    let mut v = i.to_le_bytes().to_vec();
    v.push(tag);
    v.extend(std::iter::repeat_n(tag, 60));
    v
}

/// A table of `n` rows tagged 0, and each row's tid.
fn table(engine: &BtreeEngine, name: &str, n: u64) -> (TableId, Vec<Tid>) {
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def(name)).unwrap();
    let mut tids = Vec::new();
    for i in 0..n {
        tids.push(engine.insert(txn, table, &row(i, 0)).unwrap());
    }
    engine.commit(txn).unwrap();
    (table, tids)
}

fn drain(scan: &mut Box<dyn nusadb_core::TupleScan>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    out
}

#[test]
fn a_scan_across_many_batches_returns_every_row_once_in_order() {
    let engine = BtreeEngine::new();
    let (table, _) = table(&engine, "t", 5000);
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let got = drain(&mut scan);
    let want: Vec<Vec<u8>> = (0..5000).map(|i| row(i, 0)).collect();
    assert_eq!(got, want);
    drop(scan);
    engine.commit(txn).unwrap();
}

/// A READ COMMITTED transaction keeps a scan open across a later statement (a cursor): the
/// statement refreshes the transaction's view, another transaction rewrites every row and
/// purge runs, and the rest of the scan still reads the rows as of when it opened.
#[test]
fn an_open_scan_keeps_its_snapshot_across_statements_updates_and_purge() {
    let engine = BtreeEngine::new();
    let (table, tids) = table(&engine, "t", 4000);
    let reader = engine.begin(RC).unwrap();
    engine.begin_statement(reader).unwrap();
    let mut scan = engine.scan(reader, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..1500 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    let writer = engine.begin(RC).unwrap();
    for (i, tid) in tids.iter().enumerate() {
        engine
            .update(writer, table, *tid, &row(i as u64, 1))
            .unwrap();
    }
    engine.commit(writer).unwrap();
    // A later statement of the same transaction takes a fresh view, which sees the rewrite: only
    // the open scan still needs the old versions, and purge must leave them.
    engine.begin_statement(reader).unwrap();
    engine.purge().unwrap();
    got.extend(drain(&mut scan));
    let want: Vec<Vec<u8>> = (0..4000).map(|i| row(i, 0)).collect();
    assert_eq!(got, want, "the scan reads the rows as of when it opened");
    drop(scan);
    // Once the scan is closed, the new rows are what a fresh scan reads.
    engine.begin_statement(reader).unwrap();
    let mut fresh = engine.scan(reader, table).unwrap();
    let want: Vec<Vec<u8>> = (0..4000).map(|i| row(i, 1)).collect();
    assert_eq!(drain(&mut fresh), want);
    drop(fresh);
    engine.commit(reader).unwrap();
}

/// Rows the scan's own transaction inserts after it opened are not read by it (a scan feeding
/// an insert into the same table must not read its own output), while rows the transaction
/// inserted before the scan opened are.
#[test]
fn a_scan_never_reads_rows_its_transaction_writes_after_it_opened() {
    let engine = BtreeEngine::new();
    let (table, _) = table(&engine, "t", 3000);
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, &row(9_000, 2)).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut seen = 0;
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        seen += 1;
        // Feed each row back into the table, as INSERT INTO t SELECT * FROM t does.
        engine.insert(txn, table, &tuple).unwrap();
        assert!(seen <= 3001, "the scan is reading its own output");
    }
    assert_eq!(seen, 3001);
    drop(scan);
    engine.commit(txn).unwrap();
}

/// A table dropped and purged while a scan over it is open stays readable until the scan
/// closes: its pages are not freed under the scan's view.
#[test]
fn a_table_dropped_under_an_open_scan_stays_readable_until_it_closes() {
    let engine = BtreeEngine::new();
    let (table, _) = table(&engine, "t", 4000);
    let reader = engine.begin(RC).unwrap();
    let mut scan = engine.scan(reader, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..1000 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    let dropper = engine.begin(RC).unwrap();
    engine.drop_table(dropper, table).unwrap();
    engine.commit(dropper).unwrap();
    engine.begin_statement(reader).unwrap();
    let stats = engine.purge().unwrap();
    assert_eq!(
        stats.tables_reclaimed, 0,
        "the open scan pins the dropped tree"
    );
    got.extend(drain(&mut scan));
    let want: Vec<Vec<u8>> = (0..4000).map(|i| row(i, 0)).collect();
    assert_eq!(got, want);
    drop(scan);
    engine.commit(reader).unwrap();
    let stats = engine.purge().unwrap();
    assert_eq!(
        stats.tables_reclaimed, 1,
        "closed, the scan no longer pins it"
    );
}

/// The scan's own transaction updating and deleting rows ahead of the cursor: the scan still
/// reads every row as it was when it opened, as a scan read whole at open would.
#[test]
fn own_updates_and_deletes_ahead_of_the_cursor_are_not_read() {
    let engine = BtreeEngine::new();
    let (table, tids) = table(&engine, "t", 4000);
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..100 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    // Rewrite the last half and delete a slice in the middle, all ahead of the cursor.
    for (i, tid) in tids.iter().enumerate().skip(2000) {
        engine.update(txn, table, *tid, &row(i as u64, 7)).unwrap();
    }
    for tid in &tids[1000..1100] {
        engine.delete(txn, table, *tid).unwrap();
    }
    got.extend(drain(&mut scan));
    let want: Vec<Vec<u8>> = (0..4000).map(|i| row(i, 0)).collect();
    assert_eq!(got, want);
    drop(scan);
    // A fresh scan reads the transaction's writes.
    let mut fresh = engine.scan(txn, table).unwrap();
    let after = drain(&mut fresh);
    assert_eq!(after.len(), 3900);
    assert!(after.iter().filter(|r| r[8] == 7).count() == 2000);
    drop(fresh);
    engine.commit(txn).unwrap();
}

/// A scan whose own transaction drops the table and commits still returns every row: the commit
/// reads the rest of the scan into its buffer, so purge may free the tree right away.
#[test]
fn a_table_the_scans_own_transaction_dropped_stays_readable_after_commit() {
    let engine = BtreeEngine::new();
    let (table, _) = table(&engine, "t", 3000);
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..500 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    engine.drop_table(txn, table).unwrap();
    engine.commit(txn).unwrap();
    assert_eq!(
        engine.purge().unwrap().tables_reclaimed,
        1,
        "a scan whose transaction ended pins nothing"
    );
    got.extend(drain(&mut scan));
    let want: Vec<Vec<u8>> = (0..3000).map(|i| row(i, 0)).collect();
    assert_eq!(got, want);
}

/// Rolling back to a savepoint while a scan is open does not change what the scan reads: rows the
/// transaction wrote before the scan opened and then undoes are still returned, as a scan read
/// whole at open would return them.
#[test]
fn a_savepoint_rollback_under_an_open_scan_does_not_change_what_it_reads() {
    let engine = BtreeEngine::new();
    let (table, _) = table(&engine, "t", 3000);
    let txn = engine.begin(RC).unwrap();
    engine.savepoint(txn, "s").unwrap();
    for i in 3000..3100 {
        engine.insert(txn, table, &row(i, 0)).unwrap();
    }
    let mut scan = engine.scan(txn, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..10 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    engine.rollback_to(txn, "s").unwrap();
    got.extend(drain(&mut scan));
    let want: Vec<Vec<u8>> = (0..3100).map(|i| row(i, 0)).collect();
    assert_eq!(got, want);
    drop(scan);
    engine.commit(txn).unwrap();
}

/// A scan open on a table its own transaction created and then rolled back still returns what it
/// saw, and purge frees the tree at once: the rollback read the rest of the scan first.
#[test]
fn an_aborted_create_table_under_an_open_scan_is_freed_at_once() {
    let engine = BtreeEngine::new();
    let baseline = engine.live_pages().unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def("t")).unwrap();
    for i in 0..3000 {
        engine.insert(txn, table, &row(i, 0)).unwrap();
    }
    let mut scan = engine.scan(txn, table).unwrap();
    let mut got = Vec::new();
    for _ in 0..10 {
        got.push(scan.try_next().unwrap().unwrap().1.to_vec());
    }
    engine.rollback(txn).unwrap();
    assert_eq!(engine.purge().unwrap().tables_reclaimed, 1);
    assert_eq!(engine.live_pages().unwrap(), baseline);
    got.extend(drain(&mut scan));
    assert_eq!(got.len(), 3000);
}

/// A scan left open after its transaction committed a DROP TABLE does not keep the dropped tree
/// alive through a checkpoint: the image and the store after a restart hold none of its pages.
#[test]
fn a_scan_left_open_past_its_transaction_leaks_no_pages_through_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let baseline = engine.live_pages().unwrap();
    let (table, _) = table(&engine, "t", 3000);
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let first = scan.try_next().unwrap().unwrap().1.to_vec();
    engine.drop_table(txn, table).unwrap();
    engine.commit(txn).unwrap();
    engine.checkpoint().unwrap();
    assert_eq!(engine.live_pages().unwrap(), baseline);
    let mut got = vec![first];
    got.extend(drain(&mut scan));
    assert_eq!(got.len(), 3000);
    drop(scan);
    drop(engine);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(reopened.live_pages().unwrap(), baseline);
}
