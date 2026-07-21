//! B-tree top-level: create/open, search, insert with split propagation, range scan.
//!
//! A persistent B+tree mapping `u64` keys to `u64` values, one node per page, backed by
//! any [`PageStore`]. Splits push a separator to the parent; a root split grows the tree
//! by one level. Leaves are singly linked for ordered range scans.
//!
//! # Concurrency: latch crabbing
//!
//! Every access takes per-page latches from an internal table (`LatchTable`), keyed by
//! `PageId`. Reads use **lock coupling**: the parent's read latch is held only until the
//! child's read latch has been acquired, then released — so multiple readers can descend
//! disjoint subtrees in parallel. Writes hold every latch on the descent path (write
//! mode) until the operation finishes, since any of them might need to absorb a
//! propagating split. This is the classic, simple variant; the "safe-node" optimization
//! that drops ancestor latches when a child cannot split is a future tweak.
//!
//! Per-page latches are kept in a `Mutex<HashMap>` and lazily allocated; deallocated
//! pages leave stale entries behind, which is harmless until the table is rebuilt — a
//! later concern when deallocation is wired into the B-tree itself.

// Split logic indexes exact-sized temporary vectors; bounds are guaranteed by construction.
#![allow(clippy::indexing_slicing)]
// Latch lifetimes are load-bearing: the *whole point* of crabbing is that a parent latch
// is held precisely until the child latch is acquired (read path) or until the recursive
// call returns (write path). Auto-tightening these scopes — clippy's suggestion — would
// destroy the protocol. Every drop here is deliberate.
#![allow(clippy::significant_drop_tightening)]

use std::collections::HashMap;
use std::sync::Arc;

use nusadb_core::{Error, PageId, PageStore, Result};
use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use parking_lot::{Mutex, RawRwLock, RwLock};

use crate::btree::node::{MIN_ENTRIES, NO_PAGE, Node, NodeType};

/// Table of per-page latches, lazily populated.
///
/// `Arc<RwLock<()>>` so a caller can hold a guard (via `read_arc`/`write_arc`) without
/// keeping the table mutex locked. The unit payload is intentional — the latch protects
/// the page bytes stored in the underlying `PageStore`, not memory inside the latch.
#[derive(Debug, Default)]
struct LatchTable {
    entries: Mutex<HashMap<u64, Arc<RwLock<()>>>>,
}

impl LatchTable {
    fn latch(&self, id: PageId) -> Arc<RwLock<()>> {
        let mut entries = self.entries.lock();
        entries
            .entry(id.0)
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }
}

/// RAII read latch on a single page — held until dropped.
type ReadLatch = ArcRwLockReadGuard<RawRwLock, ()>;
/// RAII write latch on a single page — held until dropped.
type WriteLatch = ArcRwLockWriteGuard<RawRwLock, ()>;

/// A B+tree over a [`PageStore`]. Maps `u64 → u64`. Thread-safe — `get`/`insert`/`scan`
/// all take `&self` and serialize internally via per-page latches.
#[derive(Debug)]
pub struct BTree<'s, S: PageStore> {
    store: &'s S,
    /// Interior-mutable so `insert` can take `&self` and grow the tree on a root split.
    /// The mutex is held only across the (very short) root-pointer swap.
    root: Mutex<PageId>,
    latches: LatchTable,
}

impl<'s, S: PageStore> BTree<'s, S> {
    /// Create a new, empty tree (a single leaf root).
    ///
    /// # Errors
    /// Propagates storage allocation/write errors.
    pub fn create(store: &'s S) -> Result<Self> {
        let root = store.allocate_page()?;
        let node = Node::new(NodeType::Leaf);
        store.write_page(root, node.as_bytes())?;
        Ok(Self {
            store,
            root: Mutex::new(root),
            latches: LatchTable::default(),
        })
    }

    /// Open an existing tree rooted at `root`.
    #[must_use]
    pub fn open(store: &'s S, root: PageId) -> Self {
        Self {
            store,
            root: Mutex::new(root),
            latches: LatchTable::default(),
        }
    }

    /// Current root page id (persist this to reopen the tree later).
    #[must_use]
    pub fn root(&self) -> PageId {
        *self.root.lock()
    }

    /// Read page `id` and validate it as a B-tree node.
    ///
    /// Routes every read-path page through [`Node::try_from_bytes`], so a corrupt page (bad magic
    /// or an out-of-range key count) surfaces as an error rather than panicking deeper in the
    /// traversal when `key_at`/`payload_at` would index past the page.
    fn read_node(&self, id: PageId) -> Result<Node> {
        let bytes = self.store.read_page(id)?;
        Node::try_from_bytes(bytes).ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("corrupt b-tree node at page {id:?} (bad magic or key count)"),
            ))
        })
    }

    /// Look up `key`, returning its value if present.
    ///
    /// Uses read-latch coupling: at most two latches are held at any moment (the parent
    /// and the child being descended into), and only the leaf latch is held while the
    /// final search runs.
    ///
    /// # Errors
    /// Propagates storage read errors.
    pub fn get(&self, key: u64) -> Result<Option<u64>> {
        let mut id = *self.root.lock();
        let mut current: ReadLatch = self.read_latch(id);
        loop {
            let node = self.read_node(id)?;
            match node.node_type() {
                NodeType::Leaf => {
                    let result = node.search(key).ok().map(|i| node.payload_at(i));
                    drop(current);
                    return Ok(result);
                },
                NodeType::Internal => {
                    let next_id = node.child_at(node.child_index(key));
                    // Crabbing: latch the child *before* releasing the parent so the
                    // child page cannot be split out from under us.
                    let next: ReadLatch = self.read_latch(next_id);
                    drop(current);
                    current = next;
                    id = next_id;
                },
            }
        }
    }

    /// Insert or update `key → value`.
    ///
    /// Holds write latches on every page along the descent path until the operation
    /// completes, so any split that propagates upward can mutate the parent it finds.
    /// A root split allocates a fresh root page and atomically swaps the root pointer.
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn insert(&self, key: u64, value: u64) -> Result<()> {
        // Lock the root pointer for the *whole* operation so a concurrent root split
        // does not move the root out from under us between read and write. Concurrent
        // readers are unaffected — they re-read the root inside `get`.
        let mut root_guard = self.root.lock();
        let root_id = *root_guard;
        let root_latch = self.write_latch(root_id);
        let outcome = self.insert_into(root_id, root_latch, key, value)?;
        if let Some((separator, right)) = outcome {
            // Root split: new internal root whose leftmost child is the old root.
            let new_root = self.store.allocate_page()?;
            let mut root = Node::new(NodeType::Internal);
            root.set_link(root_id.0);
            root.insert_entry(0, separator, right.0);
            self.store.write_page(new_root, root.as_bytes())?;
            *root_guard = new_root;
        }
        Ok(())
    }

    /// Recursively insert; returns `Some((separator, new_right_page))` if `id` split.
    /// `latch` is the write latch already held on `id` by the caller; it stays held
    /// until this call returns, so a propagating split can mutate the parent safely.
    fn insert_into(
        &self,
        id: PageId,
        latch: WriteLatch,
        key: u64,
        value: u64,
    ) -> Result<Option<(u64, PageId)>> {
        let mut node = self.read_node(id)?;
        match node.node_type() {
            NodeType::Leaf => {
                let result = match node.search(key) {
                    Ok(i) => {
                        node.set_payload(i, value);
                        self.store.write_page(id, node.as_bytes())?;
                        Ok(None)
                    },
                    Err(i) if !node.is_full() => {
                        node.insert_entry(i, key, value);
                        self.store.write_page(id, node.as_bytes())?;
                        Ok(None)
                    },
                    Err(i) => Ok(Some(self.split_leaf(id, &node, i, key, value)?)),
                };
                drop(latch); // release the leaf latch
                result
            },
            NodeType::Internal => {
                let cidx = node.child_index(key);
                let child = node.child_at(cidx);
                let child_latch = self.write_latch(child);
                let propagated = self.insert_into(child, child_latch, key, value)?;
                let result = match propagated {
                    None => Ok(None),
                    Some((separator, right)) if !node.is_full() => {
                        node.insert_entry(cidx, separator, right.0);
                        self.store.write_page(id, node.as_bytes())?;
                        Ok(None)
                    },
                    Some((separator, right)) => Ok(Some(
                        self.split_internal(id, &node, cidx, separator, right)?,
                    )),
                };
                drop(latch); // release this internal latch after the child has returned
                result
            },
        }
    }

    /// Split a full leaf around a pending insert. Returns the copied-up separator and the
    /// new right leaf's page id. The left half is written back to `id`.
    fn split_leaf(
        &self,
        id: PageId,
        node: &Node,
        ins_idx: usize,
        key: u64,
        value: u64,
    ) -> Result<(u64, PageId)> {
        let mut entries: Vec<(u64, u64)> = Vec::with_capacity(node.len() + 1);
        entries.extend((0..node.len()).map(|i| (node.key_at(i), node.payload_at(i))));
        entries.insert(ins_idx, (key, value));
        let mid = entries.len() / 2;
        let (left_e, right_e) = entries.split_at(mid);

        let right_id = self.store.allocate_page()?;
        let mut left = Node::new(NodeType::Leaf);
        for &(k, v) in left_e {
            left.push_entry(k, v);
        }
        let mut right = Node::new(NodeType::Leaf);
        for &(k, v) in right_e {
            right.push_entry(k, v);
        }
        right.set_link(node.link()); // inherit old sibling
        left.set_link(right_id.0); // left now points at the new right leaf

        self.store.write_page(id, left.as_bytes())?;
        self.store.write_page(right_id, right.as_bytes())?;
        Ok((right_e[0].0, right_id)) // B+tree: separator is the right leaf's first key
    }

    /// Split a full internal node around a pending `(separator, right_child)` insertion at
    /// child position `ins_idx`. Returns the median key (moved up) and the new right node.
    fn split_internal(
        &self,
        id: PageId,
        node: &Node,
        ins_idx: usize,
        separator: u64,
        right_child: PageId,
    ) -> Result<(u64, PageId)> {
        let n = node.len();
        let mut keys: Vec<u64> = (0..n).map(|i| node.key_at(i)).collect();
        // children[0] = leftmost (link); children[i+1] pairs with keys[i].
        let mut children: Vec<u64> = std::iter::once(node.link())
            .chain((0..n).map(|i| node.payload_at(i)))
            .collect();
        keys.insert(ins_idx, separator);
        children.insert(ins_idx + 1, right_child.0);

        let mid = keys.len() / 2;
        let up = keys[mid]; // moves up (not copied)

        let mut left = Node::new(NodeType::Internal);
        left.set_link(children[0]);
        for j in 0..mid {
            left.push_entry(keys[j], children[j + 1]);
        }

        let right_id = self.store.allocate_page()?;
        let mut right = Node::new(NodeType::Internal);
        right.set_link(children[mid + 1]);
        for j in (mid + 1)..keys.len() {
            right.push_entry(keys[j], children[j + 1]);
        }

        self.store.write_page(id, left.as_bytes())?;
        self.store.write_page(right_id, right.as_bytes())?;
        Ok((up, right_id))
    }

    /// Delete `key` if present, returning whether a key was removed.
    ///
    /// Rebalances so every non-root node stays at least [`MIN_ENTRIES`] full: a leaf/internal node
    /// that underflows borrows an entry from an adjacent sibling, or merges with it. Like
    /// [`insert`](Self::insert) it holds the root pointer for the whole operation (serializing
    /// writers) and write-latches each page on the descent path; a borrow/merge also latches the
    /// sibling under the still-held parent latch, so a concurrent reader — which must take the parent
    /// latch first — cannot enter the subtree mid-rebalance. When the internal root loses its last
    /// separator, its sole remaining child becomes the new root and the tree's height shrinks.
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn delete(&self, key: u64) -> Result<bool> {
        let mut root_guard = self.root.lock();
        let root_id = *root_guard;
        let removed = self.delete_into(root_id, self.write_latch(root_id), key)?;
        if removed {
            let _root_latch = self.write_latch(root_id);
            let root = self.read_node(root_id)?;
            if matches!(root.node_type(), NodeType::Internal) && root.is_empty() {
                // Lost its last separator → exactly one child remains (the `link`); promote it.
                *root_guard = PageId(root.link());
            }
        }
        Ok(removed)
    }

    /// Recursively delete `key` under `id` (write-latched by the caller as `latch`). After recursing
    /// into an internal child, re-reads it and — if it dropped below [`MIN_ENTRIES`] — rebalances it
    /// against a sibling, which may in turn shrink this node (the underflow propagates upward).
    fn delete_into(&self, id: PageId, latch: WriteLatch, key: u64) -> Result<bool> {
        let mut node = self.read_node(id)?;
        let removed = match node.node_type() {
            NodeType::Leaf => match node.search(key) {
                Ok(i) => {
                    node.remove_entry(i);
                    self.store.write_page(id, node.as_bytes())?;
                    true
                },
                Err(_) => false,
            },
            NodeType::Internal => {
                let cidx = node.child_index(key);
                let child_id = node.child_at(cidx);
                let removed = self.delete_into(child_id, self.write_latch(child_id), key)?;
                if removed {
                    let child_len = self.read_node(child_id)?.len();
                    if child_len < MIN_ENTRIES {
                        let orphan = self.rebalance_child(&mut node, cidx)?;
                        // Persist the parent first so its on-disk image no longer references the
                        // merged-away page, *then* recycle that page — never the other way round
                        // A crash after the free but before the parent write would leave a
                        // dangling child pointer; this order at worst leaks the page on a crash in
                        // between, which is safe.
                        self.store.write_page(id, node.as_bytes())?;
                        if let Some(dead) = orphan {
                            self.store.deallocate_page(dead)?;
                        }
                    }
                }
                removed
            },
        };
        drop(latch);
        Ok(removed)
    }

    /// Restore the minimum-occupancy invariant for the just-underflowed child at child position
    /// `cidx` of `parent`, by borrowing from or merging with an adjacent sibling. Prefers the left
    /// sibling; the leftmost child (`cidx == 0`) uses its right sibling. Mutates `parent` in place
    /// (the caller persists it) and writes the surviving child/sibling pages here.
    ///
    /// Returns the page id orphaned by a merge, if any, so the caller can recycle it.
    /// The orphan must be freed **only after** the caller persists `parent` — until then `parent`'s
    /// on-disk image still points at the merged-away page, and recycling it first could leave a
    /// dangling reference if a crash struck between the free and the parent write.
    fn rebalance_child(&self, parent: &mut Node, cidx: usize) -> Result<Option<PageId>> {
        let orphan = if cidx > 0 {
            // Rebalance child `cidx` against LEFT sibling `cidx-1`; separator is entry `cidx-1`.
            let sep = cidx - 1;
            let left_id = parent.child_at(sep);
            let child_id = parent.child_at(cidx);
            let _left_latch = self.write_latch(left_id);
            let mut left = self.read_node(left_id)?;
            let mut child = self.read_node(child_id)?;
            if left.len() > MIN_ENTRIES {
                borrow_from_left(parent, sep, &mut left, &mut child);
                self.store.write_page(left_id, left.as_bytes())?;
                self.store.write_page(child_id, child.as_bytes())?;
                None
            } else {
                merge_into_left(parent, sep, &mut left, &child);
                self.store.write_page(left_id, left.as_bytes())?;
                Some(child_id) // `child` page is now orphaned
            }
        } else {
            // Leftmost child underflowed → rebalance against RIGHT sibling `1`; separator is entry 0.
            let child_id = parent.child_at(0);
            let right_id = parent.child_at(1);
            let _right_latch = self.write_latch(right_id);
            let mut child = self.read_node(child_id)?;
            let mut right = self.read_node(right_id)?;
            if right.len() > MIN_ENTRIES {
                borrow_from_right(parent, 0, &mut child, &mut right);
                self.store.write_page(child_id, child.as_bytes())?;
                self.store.write_page(right_id, right.as_bytes())?;
                None
            } else {
                merge_into_left(parent, 0, &mut child, &right);
                self.store.write_page(child_id, child.as_bytes())?;
                Some(right_id) // `right` page is now orphaned
            }
        };
        Ok(orphan)
    }

    /// Collect every `(key, value)` pair in ascending key order by walking the leaf chain.
    ///
    /// # Errors
    /// Propagates storage read errors.
    pub fn scan(&self) -> Result<Vec<(u64, u64)>> {
        // Descend to the leftmost leaf with read-latch coupling.
        let mut id = *self.root.lock();
        let mut current: ReadLatch = self.read_latch(id);
        let leaf_id = loop {
            let node = self.read_node(id)?;
            if node.node_type() == NodeType::Leaf {
                break id;
            }
            let next_id = node.child_at(0);
            let next: ReadLatch = self.read_latch(next_id);
            drop(current);
            current = next;
            id = next_id;
        };
        drop(current); // release the leaf latch — leaf-chain walk re-latches per page

        // Follow the sibling chain, re-latching each leaf in turn.
        let mut out = Vec::new();
        let mut id = leaf_id;
        loop {
            let latch = self.read_latch(id);
            let node = self.read_node(id)?;
            for i in 0..node.len() {
                out.push((node.key_at(i), node.payload_at(i)));
            }
            let next = node.link();
            drop(latch);
            if next == NO_PAGE {
                break;
            }
            id = PageId(next);
        }
        Ok(out)
    }

    fn read_latch(&self, id: PageId) -> ReadLatch {
        self.latches.latch(id).read_arc()
    }

    fn write_latch(&self, id: PageId) -> WriteLatch {
        self.latches.latch(id).write_arc()
    }
}

// === Delete rebalancing primitives ================================
// Pure in-memory node surgery shared by `rebalance_child`; the caller owns reading/writing pages.
// `parent.entry[sep]` is the separator between the two siblings; its payload is the page id of the
// right-hand sibling, so removing it (on merge) drops both the separator key and that child pointer.

/// Move the left sibling's last entry into the front of `child`, fixing the parent separator. Works
/// for a leaf (the separator becomes `child`'s new first key) and an internal node (a rotation: the
/// old separator descends into `child`, the left sibling's last key ascends to the parent).
fn borrow_from_left(parent: &mut Node, sep: usize, left: &mut Node, child: &mut Node) {
    let last = left.len() - 1;
    match child.node_type() {
        NodeType::Leaf => {
            let (k, v) = (left.key_at(last), left.payload_at(last));
            left.remove_entry(last);
            child.insert_entry(0, k, v);
            parent.set_key(sep, k);
        },
        NodeType::Internal => {
            let down = parent.key_at(sep);
            // `child`'s old leftmost child sits to the right of the descended separator.
            child.insert_entry(0, down, child.link());
            child.set_link(left.payload_at(last));
            parent.set_key(sep, left.key_at(last));
            left.remove_entry(last);
        },
    }
}

/// Move the right sibling's first entry into the end of `child`, fixing the parent separator —
/// the mirror of [`borrow_from_left`].
fn borrow_from_right(parent: &mut Node, sep: usize, child: &mut Node, right: &mut Node) {
    match child.node_type() {
        NodeType::Leaf => {
            let (k, v) = (right.key_at(0), right.payload_at(0));
            right.remove_entry(0);
            child.push_entry(k, v);
            parent.set_key(sep, right.key_at(0));
        },
        NodeType::Internal => {
            let down = parent.key_at(sep);
            let up = right.key_at(0);
            child.push_entry(down, right.link()); // right's old leftmost child follows the separator
            right.set_link(right.payload_at(0));
            right.remove_entry(0);
            parent.set_key(sep, up);
        },
    }
}

/// Merge the `right` node into `left` (which survives) and drop the parent's `sep` entry (its
/// separator key + the pointer to `right`). A leaf merge concatenates entries and inherits the leaf
/// link; an internal merge first pulls the parent separator down between the two children sets.
fn merge_into_left(parent: &mut Node, sep: usize, left: &mut Node, right: &Node) {
    if matches!(left.node_type(), NodeType::Internal) {
        left.push_entry(parent.key_at(sep), right.link());
    }
    for i in 0..right.len() {
        left.push_entry(right.key_at(i), right.payload_at(i));
    }
    if matches!(left.node_type(), NodeType::Leaf) {
        left.set_link(right.link()); // keep the sibling chain intact
    }
    parent.remove_entry(sep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiskManager;

    fn tree_store() -> (tempfile::TempDir, DiskManager) {
        let dir = nusadb_test_utils::temp_dir();
        let dm = DiskManager::open(dir.path().join("bt.db")).unwrap();
        (dir, dm)
    }

    // White-box: this one inspects internal node structure (private `Node`/`NodeType`), so it
    // stays inline. The public-API B-tree tests live in `tests/test_tree.rs`.
    #[test]
    fn insert_10k_descending_scans_in_order() {
        let (_d, dm) = tree_store();
        let tree = BTree::create(&dm).unwrap();
        // Descending insertion stresses splits.
        for k in (0..10_000u64).rev() {
            tree.insert(k, k * 10).unwrap();
        }
        let scanned = tree.scan().unwrap();
        assert_eq!(scanned.len(), 10_000);
        for (i, &(k, v)) in scanned.iter().enumerate() {
            assert_eq!(k, i as u64);
            assert_eq!(v, i as u64 * 10);
        }
        assert_eq!(tree.get(5_000).unwrap(), Some(50_000));
        assert_eq!(tree.get(10_000).unwrap(), None);
        // The root must have become internal (tree grew past one page).
        let root = Node::from_bytes(dm.read_page(tree.root()).unwrap());
        assert_eq!(root.node_type(), NodeType::Internal);
    }
}
