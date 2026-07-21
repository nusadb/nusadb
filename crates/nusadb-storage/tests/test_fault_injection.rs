//! DST: the storage-spine `PageStore` consumers (B-tree, buffer pool) run against the
//! `nusadb-sim` fault-injection adapter.
//!
//! This is what substantiates the "same engine code runs against real disk in production and
//! against the simulation adapters in tests" claim — for the **storage spine** (the only layer that
//! consumes `PageStore`). The live engine is file-backed and does not go through
//! `PageStore`; it has its own embedded crash-fault injection, so this test deliberately
//! scopes to the page-store layer.
//!
//! Guarantees asserted: (1) the spine runs correctly over the faultless adapter; (2) an injected
//! fsync failure propagates rather than being swallowed; (3) torn writes never panic or hang the
//! B-tree (liveness under fault — the unchecked-node `try_from_bytes` guard from keeps a torn
//! node's decode in bounds).

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_core::{Error, PageStore};
use nusadb_sim::{FaultRates, FaultingStorage, SimStorage};
use nusadb_storage::{BTree, BufferPool};

fn faultless() -> FaultingStorage {
    FaultingStorage::new(SimStorage::new(), FaultRates::default(), 1)
}

#[test]
fn btree_runs_correctly_over_the_faultless_adapter() {
    // The B-tree is real engine code; here it runs entirely over the simulation PageStore.
    let store = faultless();
    let tree = BTree::create(&store).unwrap();
    let n = 500u64;
    for k in 0..n {
        tree.insert(k, k * 7).unwrap();
    }
    for k in 0..n {
        assert_eq!(tree.get(k).unwrap(), Some(k * 7), "key {k} round-trips");
    }
    let scanned = tree.scan().unwrap();
    assert_eq!(scanned.len(), n as usize);
    assert!(
        scanned.windows(2).all(|w| w[0].0 < w[1].0),
        "scan is ascending"
    );
}

#[test]
fn buffer_pool_round_trips_over_the_faultless_adapter() {
    let pool = BufferPool::new(faultless(), 4); // tiny pool forces eviction + write-back
    let mut ids = Vec::new();
    for i in 0..16u8 {
        let (id, guard) = pool.new_page().unwrap();
        guard.write(|p| p[0] = i);
        ids.push(id);
    }
    pool.flush_all().unwrap();
    for (i, id) in ids.into_iter().enumerate() {
        let got = pool.fetch_page(id).unwrap().read(|p| p[0]);
        assert_eq!(got, i as u8, "page {i} survived eviction + reload");
    }
}

#[test]
fn injected_fsync_failure_propagates() {
    // A spine that relies on fsync for durability must see the failure, not a silent success.
    let store = FaultingStorage::new(
        SimStorage::new(),
        FaultRates {
            fsync_fail: 1.0,
            ..FaultRates::default()
        },
        2,
    );
    assert!(
        matches!(store.fsync(), Err(Error::FsyncFailed(_))),
        "fsync failure must propagate"
    );
}

#[test]
fn torn_writes_do_not_panic_or_hang_the_btree() {
    // Under torn writes a B-tree node may be left half-written. The operations must stay live —
    // succeed or return a clean error — never panic or loop. (B-tree nodes carry no per-node CRC, so
    // a torn node can read back wrong data; the guarantee here is liveness, not recovery.)
    let store = FaultingStorage::new(
        SimStorage::new(),
        FaultRates {
            torn_write: 0.3,
            ..FaultRates::default()
        },
        1234,
    );
    // Even tree creation can hit a torn write; if so, liveness still held (no panic), so just stop.
    let Ok(tree) = BTree::create(&store) else {
        return;
    };
    let mut completed = 0u64;
    for k in 0..1000u64 {
        // Ignore per-op errors (expected under faults); the point is the process stays alive.
        let _ = tree.insert(k, k);
        let _ = tree.get(k % 100);
        completed += 1;
    }
    assert_eq!(
        completed, 1000,
        "every operation returned (no panic / no hang)"
    );
}
