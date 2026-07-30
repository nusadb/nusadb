//! A backward index scan (`index_scan_directed(ScanDirection::Backward)`) yields exactly the rows a
//! forward scan does over the same bounds, in reverse order — the storage-spine piece that lets the
//! SQL layer serve `ORDER BY <indexed col> DESC LIMIT n` from the index without a sort.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test harness asserts via unwrap; each tuple's first 8 bytes are a fixed-width u64 key"
)]

use std::ops::Bound;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, ScanDirection, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

/// An opaque tuple carrying `key` in its first 8 bytes (the engine never decodes it).
fn tuple(key: u64) -> Vec<u8> {
    key.to_le_bytes().to_vec()
}

/// The keys yielded by a scan, read out of each tuple's first 8 bytes.
fn drain_keys(mut scan: Box<dyn nusadb_core::engine::TupleScan>) -> Vec<u64> {
    let mut keys = Vec::new();
    while let Some((_tid, tuple)) = scan.try_next().unwrap() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&tuple[..8]);
        keys.push(u64::from_le_bytes(bytes));
    }
    keys
}

#[test]
fn backward_index_scan_is_forward_reversed() {
    let engine = BtreeEngine::new();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            txn,
            &TableDef {
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: vec![ColumnDef {
                    name: "row".to_owned(),
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
                name: "t_k".to_owned(),
                table,
                columns: vec!["row".to_owned()],
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: true,
            },
        )
        .unwrap();
    // Insert keys out of order; the index orders them by their big-endian key bytes.
    for key in [30u64, 10, 50, 20, 40] {
        let tid = engine.insert(txn, table, &tuple(key)).unwrap();
        engine
            .index_insert(txn, index, &key.to_be_bytes(), tid)
            .unwrap();
    }
    engine.commit(txn).unwrap();

    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let full = (Bound::Unbounded, Bound::Unbounded);

    // Forward: ascending key order.
    let forward = drain_keys(
        engine
            .index_scan_directed(
                txn,
                index,
                full.0.clone(),
                full.1.clone(),
                ScanDirection::Forward,
            )
            .unwrap(),
    );
    assert_eq!(forward, vec![10, 20, 30, 40, 50]);

    // Backward: exactly the forward rows, reversed.
    let backward = drain_keys(
        engine
            .index_scan_directed(
                txn,
                index,
                full.0.clone(),
                full.1.clone(),
                ScanDirection::Backward,
            )
            .unwrap(),
    );
    let mut expected = forward.clone();
    expected.reverse();
    assert_eq!(backward, expected);
    assert_eq!(backward, vec![50, 40, 30, 20, 10]);

    // A bounded range reverses within the bound too: [20, 40] descending.
    let bounded_back = drain_keys(
        engine
            .index_scan_directed(
                txn,
                index,
                Bound::Included(20u64.to_be_bytes().to_vec()),
                Bound::Included(40u64.to_be_bytes().to_vec()),
                ScanDirection::Backward,
            )
            .unwrap(),
    );
    assert_eq!(bounded_back, vec![40, 30, 20]);

    // The plain `index_scan` still matches the forward-directed scan (delegation is transparent).
    let plain = drain_keys(engine.index_scan(txn, index, full.0, full.1).unwrap());
    assert_eq!(plain, forward);

    engine.commit(txn).unwrap();
}
