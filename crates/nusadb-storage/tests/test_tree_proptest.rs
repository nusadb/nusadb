//! Property-based tests (proptest) for the B-tree.
//!
//! Treat `BTree` as a persistent `u64 -> u64` map and check it against a reference
//! `BTreeMap` model under randomized operation sequences. The invariants:
//!
//! - **Model equivalence** — after any run of inserts/updates, a point lookup agrees with the
//!   model on every key in the domain (present keys match, absent keys are `None`).
//! - **Sorted scan** — `scan()` yields the model's entries in *strictly* ascending key order
//!   with no duplicate keys.
//! - **Last-write-wins** — re-inserting a key overwrites its value, exactly like `BTreeMap`.
//! - **Insertion-order independence** — the final structure depends only on the final
//!   key→value map, not the order keys arrived in (ascending vs descending insertion agree).

#![allow(
    clippy::unwrap_used,
    reason = "proptest harness asserts via unwrap on infallible test setup"
)]

use std::collections::BTreeMap;

use nusadb_storage::{BTree, DiskManager};
use proptest::prelude::*;

fn fresh_tree() -> (tempfile::TempDir, DiskManager) {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("bt.db")).unwrap();
    (dir, dm)
}

/// The key domain. Deliberately small (`0..KEY_DOMAIN`) so a random run produces real
/// collisions — exercising the last-write-wins update path, not just fresh inserts.
const KEY_DOMAIN: u64 = 64;

/// A randomized sequence of `(key, value)` inserts over the small key domain.
fn op_seq() -> impl Strategy<Value = Vec<(u64, u64)>> {
    prop::collection::vec((0..KEY_DOMAIN, any::<u64>()), 0..400)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// After any sequence of inserts/updates the tree agrees with a `BTreeMap` model on every
    /// key in the domain, the just-written key is immediately visible at its new value, and
    /// `scan()` reproduces the model in ascending order.
    #[test]
    fn matches_btreemap_model(ops in op_seq()) {
        let (_d, dm) = fresh_tree();
        let tree = BTree::create(&dm).unwrap();
        let mut model = BTreeMap::new();

        for &(k, v) in &ops {
            tree.insert(k, v).unwrap();
            model.insert(k, v); // last write wins — same contract as the tree
            // The value just written is visible immediately at its key.
            prop_assert_eq!(tree.get(k).unwrap(), Some(v));
        }

        // Every key in the domain agrees: present keys match, absent keys are `None`.
        for k in 0..KEY_DOMAIN {
            prop_assert_eq!(tree.get(k).unwrap(), model.get(&k).copied());
        }

        // scan() == model entries, in ascending key order.
        let scanned = tree.scan().unwrap();
        let expected: Vec<(u64, u64)> = model.iter().map(|(&k, &v)| (k, v)).collect();
        prop_assert_eq!(scanned, expected);
    }

    /// `scan()` is always strictly ascending by key — no out-of-order pair, no duplicate key —
    /// regardless of insertion order.
    #[test]
    fn scan_is_strictly_sorted(ops in op_seq()) {
        let (_d, dm) = fresh_tree();
        let tree = BTree::create(&dm).unwrap();
        for &(k, v) in &ops {
            tree.insert(k, v).unwrap();
        }
        let scanned = tree.scan().unwrap();
        for w in scanned.windows(2) {
            prop_assert!(
                w[0].0 < w[1].0,
                "scan not strictly ascending: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    /// The final tree state depends only on the final key→value map, not the order the keys
    /// arrived in: inserting the same reduced entries ascending vs descending yields identical
    /// scans.
    #[test]
    fn structure_is_insertion_order_independent(ops in op_seq()) {
        // Reduce to the final model (last write wins), then its sorted entry list.
        let mut model = BTreeMap::new();
        for &(k, v) in &ops {
            model.insert(k, v);
        }
        let entries: Vec<(u64, u64)> = model.iter().map(|(&k, &v)| (k, v)).collect();

        let (_d1, dm1) = fresh_tree();
        let asc = BTree::create(&dm1).unwrap();
        for &(k, v) in &entries {
            asc.insert(k, v).unwrap();
        }

        let (_d2, dm2) = fresh_tree();
        let desc = BTree::create(&dm2).unwrap();
        for &(k, v) in entries.iter().rev() {
            desc.insert(k, v).unwrap();
        }

        prop_assert_eq!(asc.scan().unwrap(), desc.scan().unwrap());
    }
}
