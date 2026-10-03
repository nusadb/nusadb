//! A forward index range scan outside SERIALIZABLE streams the index a batch at a time. It must
//! return exactly the rows a scan read whole at open returns: every visible entry in the range
//! once, in key order, across batch boundaries (including inside one key's many rows), never a row
//! inserted after it opened (by another transaction or by its own), and never lose a row that a
//! purge running meanwhile would otherwise free.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::ops::Bound;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, TableDef};
use nusadb_core::{ColumnType, IndexId, IsolationLevel, StorageEngine, TableId, TxnId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

/// A table of `rows` rows whose payload is its number, indexed by `key_of(n)`.
fn indexed(engine: &BtreeEngine, rows: u32, key_of: impl Fn(u32) -> Vec<u8>) -> (TableId, IndexId) {
    let txn = engine.begin(RC).unwrap();
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
    for n in 0..rows {
        let tid = engine.insert(txn, table, &n.to_be_bytes()).unwrap();
        engine.index_insert(txn, index, &key_of(n), tid).unwrap();
    }
    engine.commit(txn).unwrap();
    (table, index)
}

/// The payloads an index scan of `[lo, hi]` yields to `txn`, in order.
fn scan(
    engine: &BtreeEngine,
    txn: TxnId,
    index: IndexId,
    lo: Bound<Vec<u8>>,
    hi: Bound<Vec<u8>>,
) -> Vec<Vec<u8>> {
    let mut cursor = engine.index_scan(txn, index, lo, hi).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = cursor.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    out
}

/// The same scan read whole at open (a SERIALIZABLE transaction always reads its range at once).
fn scan_whole(
    engine: &BtreeEngine,
    index: IndexId,
    lo: Bound<Vec<u8>>,
    hi: Bound<Vec<u8>>,
) -> Vec<Vec<u8>> {
    let txn = engine.begin(IsolationLevel::Serializable).unwrap();
    let rows = scan(engine, txn, index, lo, hi);
    engine.commit(txn).unwrap();
    rows
}

#[test]
fn a_streamed_index_scan_matches_a_whole_one_across_many_batches() {
    let engine = BtreeEngine::new();
    // 20 000 rows over 7 keys: thousands of rows under each key, so batches end inside a key.
    let (_, index) = indexed(&engine, 20_000, |n| {
        vec![b'k', u8::try_from(n % 7).unwrap()]
    });
    for (lo, hi) in [
        (Bound::Unbounded, Bound::Unbounded),
        (
            Bound::Included(vec![b'k', 2]),
            Bound::Excluded(vec![b'k', 5]),
        ),
        (
            Bound::Excluded(vec![b'k', 0]),
            Bound::Included(vec![b'k', 6]),
        ),
        (
            Bound::Included(vec![b'k', 3]),
            Bound::Included(vec![b'k', 3]),
        ),
    ] {
        let txn = engine.begin(RC).unwrap();
        let streamed = scan(&engine, txn, index, lo.clone(), hi.clone());
        engine.commit(txn).unwrap();
        let whole = scan_whole(&engine, index, lo.clone(), hi.clone());
        assert!(!whole.is_empty(), "{lo:?}..{hi:?}");
        assert_eq!(streamed, whole, "{lo:?}..{hi:?}");
    }
    // Distinct keys too: one row per key across many batches.
    let engine = BtreeEngine::new();
    let (_, index) = indexed(&engine, 30_000, |n| n.to_be_bytes().to_vec());
    let txn = engine.begin(RC).unwrap();
    let streamed = scan(&engine, txn, index, Bound::Unbounded, Bound::Unbounded);
    engine.commit(txn).unwrap();
    assert_eq!(streamed.len(), 30_000);
    assert_eq!(
        streamed,
        scan_whole(&engine, index, Bound::Unbounded, Bound::Unbounded)
    );
}

#[test]
fn keys_too_large_for_an_index_page_stream_across_batches() {
    // Keys over the in-page entry limit live in memory beside the pages; mix them with ordinary
    // keys so batches resume both inside the in-memory entries and across the two merged.
    let engine = BtreeEngine::new();
    let (_, index) = indexed(&engine, 9_000, |n| {
        let mut key = vec![b'k', u8::try_from(n % 5).unwrap()];
        if n % 3 == 0 {
            key.extend(std::iter::repeat_n(b'x', 2_500));
        }
        key
    });
    let txn = engine.begin(RC).unwrap();
    let streamed = scan(&engine, txn, index, Bound::Unbounded, Bound::Unbounded);
    engine.commit(txn).unwrap();
    assert_eq!(streamed.len(), 9_000);
    assert_eq!(
        streamed,
        scan_whole(&engine, index, Bound::Unbounded, Bound::Unbounded)
    );
}

#[test]
fn an_open_index_scan_never_reaches_rows_inserted_after_it_opened() {
    let engine = BtreeEngine::new();
    let (table, index) = indexed(&engine, 5_000, |n| n.to_be_bytes().to_vec());
    let reader = engine.begin(RC).unwrap();
    let mut cursor = engine
        .index_scan(reader, index, Bound::Unbounded, Bound::Unbounded)
        .unwrap();
    let first = cursor.try_next().unwrap().unwrap().1.to_vec();

    // Another transaction inserts and commits rows inside the range, ahead of the cursor.
    let writer = engine.begin(RC).unwrap();
    for n in 10_000..10_100_u32 {
        let tid = engine.insert(writer, table, &n.to_be_bytes()).unwrap();
        engine
            .index_insert(writer, index, &n.to_be_bytes(), tid)
            .unwrap();
    }
    engine.commit(writer).unwrap();

    // The reader's own transaction inserts into the range too, the shape of an
    // `INSERT INTO t SELECT ... FROM t` reading through the index.
    for n in 20_000..20_100_u32 {
        let tid = engine.insert(reader, table, &n.to_be_bytes()).unwrap();
        engine
            .index_insert(reader, index, &n.to_be_bytes(), tid)
            .unwrap();
    }

    let mut rows = vec![first];
    while let Some((_, tuple)) = cursor.try_next().unwrap() {
        rows.push(tuple.to_vec());
    }
    drop(cursor);
    engine.commit(reader).unwrap();
    let want: Vec<Vec<u8>> = (0..5_000_u32).map(|n| n.to_be_bytes().to_vec()).collect();
    assert_eq!(
        rows, want,
        "the scan read exactly the rows visible when it opened"
    );
}

#[test]
fn a_purge_while_an_index_scan_is_open_frees_nothing_it_still_reads() {
    let engine = BtreeEngine::new();
    let (table, index) = indexed(&engine, 6_000, |n| n.to_be_bytes().to_vec());
    let reader = engine.begin(RC).unwrap();
    let mut cursor = engine
        .index_scan(reader, index, Bound::Unbounded, Bound::Unbounded)
        .unwrap();
    let first = cursor.try_next().unwrap().unwrap().1.to_vec();

    // Delete every row past the first batch and purge: the scan's view still sees them.
    let writer = engine.begin(RC).unwrap();
    let mut doomed = engine
        .index_scan(
            writer,
            index,
            Bound::Included(2_000_u32.to_be_bytes().to_vec()),
            Bound::Unbounded,
        )
        .unwrap();
    let mut tids = Vec::new();
    while let Some((tid, _)) = doomed.try_next().unwrap() {
        tids.push(tid);
    }
    drop(doomed);
    for tid in tids {
        engine.delete(writer, table, tid).unwrap();
    }
    engine.commit(writer).unwrap();
    engine.purge().unwrap();

    let mut rows = vec![first];
    while let Some((_, tuple)) = cursor.try_next().unwrap() {
        rows.push(tuple.to_vec());
    }
    drop(cursor);
    engine.commit(reader).unwrap();
    assert_eq!(rows.len(), 6_000, "every row visible at open is still read");

    // Once the scan is gone the deletes are visible to a new reader.
    let after = engine.begin(RC).unwrap();
    assert_eq!(
        scan(&engine, after, index, Bound::Unbounded, Bound::Unbounded).len(),
        2_000
    );
    engine.commit(after).unwrap();
}

/// Timing evidence that resuming a batch inside one key costs nothing extra: one key holding
/// `n` rows, streamed against the same range read whole. A resume that re-read the key's earlier
/// rows grew quadratically (8.1 s against 0.95 s at 800k rows). Run with
/// `cargo test -p nusadb-btree --release --test test_index_stream -- --ignored --nocapture`.
#[test]
#[ignore = "timing evidence; run in release"]
fn streaming_one_large_key_is_linear() {
    for n in [100_000_u32, 400_000, 800_000] {
        let engine = BtreeEngine::new();
        let (_, index) = indexed(&engine, n, |_| b"same".to_vec());
        let start = std::time::Instant::now();
        let txn = engine.begin(RC).unwrap();
        let streamed = scan(&engine, txn, index, Bound::Unbounded, Bound::Unbounded).len();
        engine.commit(txn).unwrap();
        let stream_time = start.elapsed();
        let start = std::time::Instant::now();
        let whole = scan_whole(&engine, index, Bound::Unbounded, Bound::Unbounded).len();
        let whole_time = start.elapsed();
        assert_eq!(streamed, whole);
        println!("{n} rows under one key: streamed {stream_time:?}, read whole {whole_time:?}");
    }
}
