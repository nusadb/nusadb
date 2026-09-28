//! Index entries live in pages: a tree ordered by `(key, row)` and a row-to-key map, both in the
//! page store, carried by checkpoint images like table pages. An entry too large for an index
//! page stays in memory and rides the image as a record. Dead ranges are purged without walking
//! the index, dropped indexes give their pages back, and emptied leaves are freed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::ops::Bound;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, ScanDirection, TableDef};
use nusadb_core::{ColumnType, IndexId, IsolationLevel, StorageEngine, TableId, Tid, TxnId};

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

fn index_def(name: &str, table: TableId, unique: bool) -> IndexDef {
    IndexDef {
        name: name.to_owned(),
        table,
        columns: vec!["v".to_owned()],
        key_exprs: Vec::new(),
        predicate: None,
        include: Vec::new(),
        kind: IndexKind::BTree,
        unique,
    }
}

fn key(i: u64) -> Vec<u8> {
    format!("key-{i:08}").into_bytes()
}

/// A table and a unique index whose entries map `key(i)` to row `i`'s tid.
fn indexed(engine: &BtreeEngine, n: u64) -> (TableId, IndexId, Vec<Tid>) {
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def("t")).unwrap();
    let index = engine
        .create_index(txn, &index_def("t_v", table, true))
        .unwrap();
    let mut tids = Vec::new();
    for i in 0..n {
        let tid = engine.insert(txn, table, &key(i)).unwrap();
        engine.index_insert(txn, index, &key(i), tid).unwrap();
        tids.push(tid);
    }
    engine.commit(txn).unwrap();
    (table, index, tids)
}

fn scan(
    engine: &BtreeEngine,
    txn: TxnId,
    index: IndexId,
    lo: Bound<Vec<u8>>,
    hi: Bound<Vec<u8>>,
    direction: ScanDirection,
) -> Vec<(Tid, Vec<u8>)> {
    let mut s = engine
        .index_scan_directed(txn, index, lo, hi, direction)
        .unwrap();
    let mut out = Vec::new();
    while let Some((tid, tuple)) = s.try_next().unwrap() {
        out.push((tid, tuple.to_vec()));
    }
    out
}

fn all(engine: &BtreeEngine, index: IndexId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let rows = scan(
        engine,
        txn,
        index,
        Bound::Unbounded,
        Bound::Unbounded,
        ScanDirection::Forward,
    );
    engine.commit(txn).unwrap();
    rows.into_iter().map(|(_, t)| t).collect()
}

/// Entries survive a checkpoint and a restart through the image's pages, and uniqueness still
/// holds against entries read back from them.
#[test]
fn entries_survive_a_restart_in_pages() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, index, before) = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let (table, index, _) = indexed(&engine, 20_000);
        engine.checkpoint().unwrap();
        (table, index, all(&engine, index))
    };
    assert_eq!(before.len(), 20_000);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(all(&engine, index), before);
    let txn = engine.begin(RC).unwrap();
    let k = key(12_345);
    let hit = scan(
        &engine,
        txn,
        index,
        Bound::Included(k.clone()),
        Bound::Included(k.clone()),
        ScanDirection::Forward,
    );
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].1, k);
    engine.commit(txn).unwrap();
    let txn = engine.begin(RC).unwrap();
    let other = engine.insert(txn, table, &k).unwrap();
    let err = engine.index_insert(txn, index, &k, other).unwrap_err();
    assert!(err.to_string().contains("duplicate"), "{err}");
    engine.rollback(txn).unwrap();
}

/// An entry whose key is too large for an index page lives in memory, merges into scans in key
/// order with the rest, and survives a checkpoint and a restart as an image record.
#[test]
fn an_oversized_key_merges_in_order_and_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let mut big = key(500);
    big.extend(std::iter::repeat_n(b'z', 6000));
    let want = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let (table, index, _) = indexed(&engine, 1000);
        let txn = engine.begin(RC).unwrap();
        let tid = engine.insert(txn, table, &big).unwrap();
        engine.index_insert(txn, index, &big, tid).unwrap();
        engine.commit(txn).unwrap();
        let got = all(&engine, index);
        assert_eq!(got.len(), 1001);
        let mut sorted = got.clone();
        sorted.sort();
        assert_eq!(got, sorted, "the oversized key sits in key order");
        assert_eq!(got.iter().position(|k| *k == big), Some(501));
        engine.checkpoint().unwrap();
        got
    };
    let engine = BtreeEngine::open(&wal).unwrap();
    let index = engine.lookup_index("t_v").unwrap().unwrap();
    assert_eq!(all(&engine, index), want);
    // Backward too.
    let txn = engine.begin(RC).unwrap();
    let back: Vec<Vec<u8>> = scan(
        &engine,
        txn,
        index,
        Bound::Unbounded,
        Bound::Unbounded,
        ScanDirection::Backward,
    )
    .into_iter()
    .map(|(_, t)| t)
    .collect();
    engine.commit(txn).unwrap();
    let mut reversed = want;
    reversed.reverse();
    assert_eq!(back, reversed);
}

/// Rows moved to new keys leave dead ranges under their old ones; purge removes them once
/// settled, and after a checkpoint and a restart only the new keys are found.
#[test]
fn moved_keys_are_purged_and_stay_gone_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let index = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let (table, index, tids) = indexed(&engine, 3000);
        let txn = engine.begin(RC).unwrap();
        for (i, tid) in tids.iter().enumerate() {
            let new_key = key(100_000 + i as u64);
            engine.update(txn, table, *tid, &new_key).unwrap();
            engine.index_insert(txn, index, &new_key, *tid).unwrap();
        }
        engine.commit(txn).unwrap();
        let stats = engine.purge().unwrap();
        assert!(
            stats.index_entries_removed >= 3000,
            "the old keys' dead ranges go: {}",
            stats.index_entries_removed
        );
        engine.checkpoint().unwrap();
        index
    };
    let engine = BtreeEngine::open(&wal).unwrap();
    let got = all(&engine, index);
    let want: Vec<Vec<u8>> = (0..3000).map(|i| key(100_000 + i)).collect();
    assert_eq!(got, want);
    let txn = engine.begin(RC).unwrap();
    let old = scan(
        &engine,
        txn,
        index,
        Bound::Unbounded,
        Bound::Excluded(key(100_000)),
        ScanDirection::Forward,
    );
    engine.commit(txn).unwrap();
    assert!(old.is_empty(), "no old key is reachable");
}

/// A dropped index gives its pages back once the drop settles; a rolled-back drop keeps a
/// working index; a rolled-back create gives its pages back too.
#[test]
fn dropped_and_rolled_back_indexes_give_their_pages_back() {
    let engine = BtreeEngine::new();
    let baseline = engine.live_pages().unwrap();
    let (table, index, _) = indexed(&engine, 5000);
    let with_index = engine.live_pages().unwrap();
    // A rolled-back drop: the index still answers.
    let txn = engine.begin(RC).unwrap();
    engine.drop_index(txn, index).unwrap();
    engine.rollback(txn).unwrap();
    engine.purge().unwrap();
    assert_eq!(all(&engine, index).len(), 5000);
    assert_eq!(engine.live_pages().unwrap(), with_index);
    // A committed drop: the pages come back after purge.
    let txn = engine.begin(RC).unwrap();
    engine.drop_index(txn, index).unwrap();
    engine.commit(txn).unwrap();
    engine.purge().unwrap();
    let without_index = engine.live_pages().unwrap();
    assert!(without_index < with_index);
    // A rolled-back create, entries and all, leaves nothing behind.
    let txn = engine.begin(RC).unwrap();
    let again = engine
        .create_index(txn, &index_def("again", table, false))
        .unwrap();
    let rows = {
        let mut s = engine.scan(txn, table).unwrap();
        let mut rows = Vec::new();
        while let Some((tid, tuple)) = s.try_next().unwrap() {
            rows.push((tid, tuple.to_vec()));
        }
        rows
    };
    for (tid, tuple) in &rows {
        engine.index_insert(txn, again, tuple, *tid).unwrap();
    }
    engine.rollback(txn).unwrap();
    engine.purge().unwrap();
    assert_eq!(engine.live_pages().unwrap(), without_index);
    // The table's own pages are all that remain beyond the baseline.
    assert!(without_index > baseline);
}

/// Rows sharing one key come back in the same relative order forward and backward.
#[test]
fn ties_keep_their_order_in_both_directions() {
    let engine = BtreeEngine::new();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def("t")).unwrap();
    let index = engine
        .create_index(txn, &index_def("t_v", table, false))
        .unwrap();
    for group in 0..3_u64 {
        for n in 0..400_u64 {
            let tuple = format!("{group}-{n:04}").into_bytes();
            let tid = engine.insert(txn, table, &tuple).unwrap();
            engine.index_insert(txn, index, &key(group), tid).unwrap();
        }
    }
    engine.commit(txn).unwrap();
    let txn = engine.begin(RC).unwrap();
    let forward = scan(
        &engine,
        txn,
        index,
        Bound::Unbounded,
        Bound::Unbounded,
        ScanDirection::Forward,
    );
    let backward = scan(
        &engine,
        txn,
        index,
        Bound::Unbounded,
        Bound::Unbounded,
        ScanDirection::Backward,
    );
    engine.commit(txn).unwrap();
    let per_key = |v: &[(Tid, Vec<u8>)], g: u64| -> Vec<Vec<u8>> {
        v.iter()
            .filter(|(_, t)| t.starts_with(format!("{g}-").as_bytes()))
            .map(|(_, t)| t.clone())
            .collect()
    };
    for g in 0..3 {
        assert_eq!(per_key(&forward, g), per_key(&backward, g), "key {g}");
    }
    let first: Vec<u64> = backward
        .iter()
        .map(|(_, t)| u64::from(t[0] - b'0'))
        .collect();
    let mut desc = first.clone();
    desc.sort_by(|a, b| b.cmp(a));
    assert_eq!(first, desc, "keys descend backward");
}
