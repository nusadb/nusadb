//! Tests for `btree` (`src/btree/tree.rs`) that exercise only the public `BTree` API over a
//! file-backed store. The one white-box test that inspects internal node structure stays inline
//! in `tree.rs`.

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_core::PageStore;
use nusadb_storage::{BTree, DiskManager};

fn tree_store() -> (tempfile::TempDir, DiskManager) {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("bt.db")).unwrap();
    (dir, dm)
}

#[test]
fn get_missing_is_none() {
    let (_d, dm) = tree_store();
    let tree = BTree::create(&dm).unwrap();
    assert_eq!(tree.get(42).unwrap(), None);
}

#[test]
fn insert_then_get_and_update() {
    let (_d, dm) = tree_store();
    let tree = BTree::create(&dm).unwrap();
    tree.insert(1, 100).unwrap();
    tree.insert(2, 200).unwrap();
    assert_eq!(tree.get(1).unwrap(), Some(100));
    assert_eq!(tree.get(2).unwrap(), Some(200));
    tree.insert(1, 999).unwrap(); // update
    assert_eq!(tree.get(1).unwrap(), Some(999));
    assert_eq!(tree.get(3).unwrap(), None);
}

#[test]
fn concurrent_inserts_and_reads_stay_consistent() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let (_d, dm) = tree_store();
    // Promote to leaked `'static` so reader/writer threads share `&BTree` without a
    // self-referential lifetime dance. Bounded by the test binary's lifetime.
    let dm_ref: &'static DiskManager = Box::leak(Box::new(dm));
    let tree: &'static BTree<'static, DiskManager> =
        Box::leak(Box::new(BTree::create(dm_ref).unwrap()));

    let n: u64 = 5_000;
    // Force coercion `&'static mut → &'static` (shared) so each closure can `Copy` it.
    let done: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));

    // One writer thread populates the tree; four reader threads spin doing point lookups
    // until the writer is done. Each reader iterates the keyspace with a thread-specific
    // stride so the threads hit different keys at different times — without any RNG dep.
    let writer = thread::spawn(move || {
        for k in 0..n {
            tree.insert(k, k * 3).unwrap();
        }
        done.store(true, Ordering::Release);
    });
    let readers: Vec<_> = (0..4u64)
        .map(|tid| {
            thread::spawn(move || {
                // Walk the keyspace until the writer has finished — every observed value
                // must equal the key's expected payload. A bug in latch crabbing would
                // either crash or surface a stale half-update here.
                let stride = (tid * 7 + 3) % n.max(1);
                let mut k = tid % n.max(1);
                while !done.load(Ordering::Acquire) {
                    if let Some(v) = tree.get(k).unwrap() {
                        assert_eq!(v, k * 3, "torn read for key {k}");
                    }
                    k = (k + stride) % n;
                }
                // One final sweep after the writer finishes — every key must now be present.
                for k in 0..n {
                    let v = tree.get(k).unwrap();
                    assert_eq!(v, Some(k * 3), "missing key {k} after writer finished");
                }
            })
        })
        .collect();

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }

    // Final state: every key must be present at its value.
    let scanned = tree.scan().unwrap();
    assert_eq!(scanned.len(), n as usize);
    for (i, &(k, v)) in scanned.iter().enumerate() {
        assert_eq!(k, i as u64);
        assert_eq!(v, i as u64 * 3);
    }
}

#[test]
fn insert_10k_scrambled_scans_in_order() {
    let (_d, dm) = tree_store();
    let tree = BTree::create(&dm).unwrap();
    // Interleave low/high halves to avoid sequential insertion order.
    let n = 10_000u64;
    for step in 0..n {
        let k = if step % 2 == 0 {
            step / 2
        } else {
            n - 1 - step / 2
        };
        tree.insert(k, k + 1).unwrap();
    }
    let scanned = tree.scan().unwrap();
    assert_eq!(scanned.len(), 10_000);
    for (i, &(k, v)) in scanned.iter().enumerate() {
        assert_eq!(k, i as u64);
        assert_eq!(v, i as u64 + 1);
    }
}

// === Delete + rebalance ==========================================

/// A B-tree built from `n` ascending keys (value = key*10), enough to span multiple leaves.
fn filled_tree(dm: &DiskManager, n: u64) -> BTree<'_, DiskManager> {
    let tree = BTree::create(dm).unwrap();
    for k in 0..n {
        tree.insert(k, k * 10).unwrap();
    }
    tree
}

#[test]
fn delete_simple_and_missing() {
    let (_d, dm) = tree_store();
    let tree = BTree::create(&dm).unwrap();
    for k in 0..5u64 {
        tree.insert(k, k * 10).unwrap();
    }
    assert!(
        tree.delete(2).unwrap(),
        "deleting a present key returns true"
    );
    assert_eq!(tree.get(2).unwrap(), None, "deleted key is gone");
    assert_eq!(tree.get(3).unwrap(), Some(30), "neighbours intact");
    assert!(
        !tree.delete(2).unwrap(),
        "deleting an absent key returns false"
    );
    assert!(!tree.delete(99).unwrap(), "never-present key returns false");
    assert_eq!(tree.scan().unwrap().len(), 4);
}

#[test]
fn delete_drains_a_multi_level_tree_to_empty() {
    // Insert enough keys to force several splits (multi-leaf, internal root), then delete every key.
    // This exercises leaf underflow → borrow, → merge, internal underflow propagation, and repeated
    // root-collapse, all validated by the leaf-chain `scan` matching the model at each step.
    let (_d, dm) = tree_store();
    let n = 1500u64; // > MAX_ENTRIES (≈511) → guaranteed multiple levels
    let tree = filled_tree(&dm, n);

    // Delete from the front (forces left-edge underflows + merges that propagate up).
    for k in 0..n {
        assert!(tree.delete(k).unwrap(), "delete {k} present");
        // Spot-check the survivors and ordering every so often (full scan is O(n)).
        if k % 250 == 0 || k + 1 == n {
            let scanned = tree.scan().unwrap();
            let expected: Vec<(u64, u64)> = ((k + 1)..n).map(|j| (j, j * 10)).collect();
            assert_eq!(scanned, expected, "scan after deleting 0..={k}");
        }
    }
    assert_eq!(tree.scan().unwrap(), Vec::new(), "tree is empty");
    assert_eq!(tree.get(0).unwrap(), None);
    assert_eq!(tree.get(n - 1).unwrap(), None);
    // Re-insertion works after a full drain (tree structure is still valid).
    tree.insert(42, 420).unwrap();
    assert_eq!(tree.get(42).unwrap(), Some(420));
}

#[test]
fn merges_recycle_orphaned_pages() {
    // Draining a multi-level tree performs many merges; each merged-away page must be returned to
    // the store's free list instead of being leaked. Before the fix `free_count` stayed at
    // zero through the whole drain.
    let (_d, dm) = tree_store();
    let n = 1500u64; // > MAX_ENTRIES → multiple levels, so deletes force merges
    let tree = filled_tree(&dm, n);
    assert_eq!(dm.free_count(), 0, "a freshly built tree has freed nothing");

    for k in 0..n {
        assert!(tree.delete(k).unwrap(), "delete {k} present");
    }

    assert!(
        dm.free_count() > 0,
        "merges during the drain must recycle orphaned pages, got free_count = {}",
        dm.free_count()
    );
    // The recycled slots are genuinely reusable: allocating now reuses a freed page rather than
    // extending the file.
    let before = dm.free_count();
    dm.allocate_page().unwrap();
    assert_eq!(
        dm.free_count(),
        before - 1,
        "allocate must reuse a freed slot"
    );
}

#[test]
fn delete_interior_keys_keeps_routing_correct() {
    // Delete a scattered subset (every 3rd key) from a multi-leaf tree; the rest must remain
    // findable and ordered (stale separators must still route correctly).
    let (_d, dm) = tree_store();
    let n = 1000u64;
    let tree = filled_tree(&dm, n);
    for k in (0..n).step_by(3) {
        assert!(tree.delete(k).unwrap());
    }
    let expected: Vec<(u64, u64)> = (0..n).filter(|k| k % 3 != 0).map(|k| (k, k * 10)).collect();
    assert_eq!(tree.scan().unwrap(), expected);
    for k in 0..n {
        let want = (k % 3 != 0).then_some(k * 10);
        assert_eq!(
            tree.get(k).unwrap(),
            want,
            "get({k}) after deleting multiples of 3"
        );
    }
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Random insert/delete sequences over a key space wider than a leaf (so splits + rebalances
    /// fire) must leave the B-tree's full ordered contents identical to a `BTreeMap` model, and each
    /// `delete` must agree with the model on whether a key was present.
    #[test]
    fn delete_matches_btreemap_model(
        ops in prop::collection::vec((0u64..1500, any::<bool>(), 0u64..10_000), 0..2500),
    ) {
        let (_d, dm) = tree_store();
        let tree = BTree::create(&dm).unwrap();
        let mut model = std::collections::BTreeMap::new();
        for (k, insert, v) in ops {
            if insert {
                tree.insert(k, v).unwrap();
                model.insert(k, v);
            } else {
                let removed = tree.delete(k).unwrap();
                prop_assert_eq!(removed, model.remove(&k).is_some());
            }
        }
        let scanned = tree.scan().unwrap();
        let expected: Vec<(u64, u64)> = model.into_iter().collect();
        prop_assert_eq!(scanned, expected);
    }
}
