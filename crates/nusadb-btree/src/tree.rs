//! The clustered B-link tree (single-version): rows live **in the leaves**, keyed by the
//! table's internal monotonic row-id (ADR 008 §D1/§D2). One tree per table over a shared
//! [`PageStore`].
//!
//! **Split protocol (Lehman–Yao order):** the new right sibling — carrying the moved upper half,
//! the old node's right link, and the old node's high key — is written **before** the split node
//! is rewritten (now linking right with a lowered high key), and only then is the separator
//! published in the parent. Between those steps the tree is *already correct*: a descent that
//! lands on the split node with a key at or beyond its new high key chases the right link
//! (the private `descend` helper). The engine leans on this for real: **writers**
//! serialize per table (the engine's per-table latch — one mutating tree walk at a time), while
//! **readers descend with no latch at all**, riding the publish order plus the page store's
//! atomic per-page reads. Finer-grained (per-page/OLC) writer latching is a later phase.
//!
//! Limitations (documented, by phase design): single-version (no MVCC), not durable (no
//! redo WAL / recovery — a later phase), deletes do not merge underfull leaves (space reclaim), and a
//! tuple must fit one leaf ([`node::MAX_TUPLE`] — TOAST-style overflow later).

use nusadb_core::traits::Page;
use nusadb_core::{PageId, PageStore, Result};

use crate::node;

/// The interior path (root first) plus the target leaf a descent produced.
type Descent = (Vec<(PageId, Page)>, PageId, Page);

/// One table's clustered tree: rows in leaves, keyed by row-id.
pub struct ClusteredTree<'s> {
    store: &'s dyn PageStore,
    root: PageId,
}

impl std::fmt::Debug for ClusteredTree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The store is a trait object; identify the tree by its root page.
        f.debug_struct("ClusteredTree")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

/// The engine-level error for a tuple too large for one leaf (has no overflow pages).
fn tuple_too_large(len: usize) -> nusadb_core::Error {
    nusadb_core::Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "nusadb-btree: tuple of {len} bytes exceeds the single-leaf capacity of {} bytes \
             (overflow pages are a later phase)",
            node::MAX_TUPLE
        ),
    ))
}

/// Internal-corruption error: a structural invariant did not hold. Never expected; loud.
fn corrupt(msg: &str) -> nusadb_core::Error {
    nusadb_core::Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("nusadb-btree: {msg}"),
    ))
}

impl<'s> ClusteredTree<'s> {
    /// Create a new empty tree: a single leaf root.
    pub fn create(store: &'s dyn PageStore) -> Result<Self> {
        let root = store.allocate_page()?;
        let mut page = [0u8; nusadb_core::PAGE_SIZE];
        node::init_leaf(&mut page);
        store.write_page(root, &page)?;
        Ok(Self { store, root })
    }

    /// Open an existing tree at `root`.
    pub const fn open(store: &'s dyn PageStore, root: PageId) -> Self {
        Self { store, root }
    }

    /// The current root page (moves only when the root splits).
    pub const fn root(&self) -> PageId {
        self.root
    }

    /// Descend from the root to the leaf that owns `key`, returning the interior path
    /// (`(page id, page)` pairs, root first) and the leaf. At every level the B-link rule
    /// applies: while `key` is at or beyond the node's high key, follow the right link.
    fn descend(&self, key: u64) -> Result<Descent> {
        let mut path = Vec::new();
        let mut at = self.root;
        loop {
            let mut page = self.store.read_page(at)?;
            // B-link chase: the node split after we picked it — its upper keys moved right.
            while key >= node::high_key(&page) {
                let Some(link) = node::right_link(&page) else {
                    break; // high key is +inf on the rightmost node; nothing to chase.
                };
                at = link;
                page = self.store.read_page(at)?;
            }
            if node::is_leaf(&page) {
                return Ok((path, at, page));
            }
            let next = node::route(&page, key);
            path.push((at, page));
            at = next;
        }
    }

    /// Read-only descent to the leaf that owns `key`: the same walk (and B-link chase) as
    /// [`Self::descend`] minus the interior-path collection only a writer needs for splits —
    /// a point read holds one page copy at a time instead of retaining the whole path.
    fn descend_read(&self, key: u64) -> Result<Page> {
        let mut at = self.root;
        loop {
            let mut page = self.store.read_page(at)?;
            // B-link chase: the node split after we picked it — its upper keys moved right.
            while key >= node::high_key(&page) {
                let Some(link) = node::right_link(&page) else {
                    break; // high key is +inf on the rightmost node; nothing to chase.
                };
                at = link;
                page = self.store.read_page(at)?;
            }
            if node::is_leaf(&page) {
                return Ok(page);
            }
            at = node::route(&page, key);
        }
    }

    /// The tuple stored under `key`, if any — copied out exactly once when found
    /// ([`node::leaf_find`] borrows from the page; profiling found the old form materializing
    /// every entry of the leaf per lookup).
    pub fn get(&self, key: u64) -> Result<Option<Vec<u8>>> {
        let leaf = self.descend_read(key)?;
        Ok(node::leaf_find(&leaf, key).map(<[u8]>::to_vec))
    }

    /// Insert `tuple` under `key`. `key` must not already exist (row-ids are engine-minted and
    /// monotonic, so a duplicate is an internal error, reported loudly).
    pub fn insert(&mut self, key: u64, tuple: &[u8]) -> Result<()> {
        if tuple.len() > node::MAX_TUPLE {
            return Err(tuple_too_large(tuple.len()));
        }
        let (path, leaf_id, leaf) = self.descend(key)?;
        let mut entries = node::leaf_entries(&leaf);
        let Err(at) = entries.binary_search_by_key(&key, |(k, _)| *k) else {
            return Err(corrupt("duplicate row-id insert"));
        };
        entries.insert(at, (key, tuple.to_vec()));
        self.write_leaf_or_split(&path, leaf_id, &leaf, &entries)
    }

    /// Replace the tuple under `key` (which must exist) with `tuple`.
    pub fn update(&mut self, key: u64, tuple: &[u8]) -> Result<()> {
        if tuple.len() > node::MAX_TUPLE {
            return Err(tuple_too_large(tuple.len()));
        }
        let (path, leaf_id, leaf) = self.descend(key)?;
        let mut entries = node::leaf_entries(&leaf);
        match entries.binary_search_by_key(&key, |(k, _)| *k) {
            Ok(pos) => {
                if let Some(slot) = entries.get_mut(pos) {
                    slot.1 = tuple.to_vec();
                }
            },
            Err(_) => return Err(corrupt("update of a missing row-id")),
        }
        self.write_leaf_or_split(&path, leaf_id, &leaf, &entries)
    }

    /// Remove the entry under `key`; `Ok(true)` if it existed. Leaves are not merged when they
    /// underfill (reclaims space); an empty leaf simply stays in the chain.
    pub fn delete(&self, key: u64) -> Result<bool> {
        let (_, leaf_id, leaf) = self.descend(key)?;
        let mut entries = node::leaf_entries(&leaf);
        match entries.binary_search_by_key(&key, |(k, _)| *k) {
            Ok(pos) => {
                entries.remove(pos);
                let mut page = leaf;
                if !node::write_leaf_entries(&mut page, &entries) {
                    return Err(corrupt("shrunken leaf failed to serialize"));
                }
                self.store.write_page(leaf_id, &page)?;
                Ok(true)
            },
            Err(_) => Ok(false),
        }
    }

    /// Every `(key, tuple)` in key order — the full-table scan (leftmost descent, then the leaf
    /// chain via right links).
    pub fn scan(&self) -> Result<Vec<(u64, Vec<u8>)>> {
        let mut out = Vec::new();
        self.scan_with(|key, tuple| {
            out.push((key, tuple.to_vec()));
            Ok(())
        })?;
        Ok(out)
    }

    /// The visitor form of [`scan`](Self::scan): each `(key, tuple)` is handed to `f` with the
    /// tuple BORROWED from the leaf's page copy — the single-copy read path: a read-only
    /// consumer copies each visible tuple exactly once (into its `Arc`), instead of once into a
    /// per-row `Vec` here and again at the caller. Same walk, same pages, same order.
    ///
    /// # Errors
    /// Propagates page-store read errors and any error `f` returns.
    pub fn scan_with<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(u64, &[u8]) -> Result<()>,
    {
        // Leftmost leaf: descend routing every interior by "less than any separator".
        let mut at = self.root;
        let mut page = self.store.read_page(at)?;
        while !node::is_leaf(&page) {
            at = node::interior_entries(&page)
                .first()
                .map_or_else(|| node::rightmost_child(&page), |(_, child)| *child);
            page = self.store.read_page(at)?;
        }
        loop {
            let mut result = Ok(());
            node::for_each_leaf_entry(&page, |key, tuple| {
                if result.is_ok() {
                    result = f(key, tuple);
                }
            });
            result?;
            let Some(link) = node::right_link(&page) else {
                break;
            };
            page = self.store.read_page(link)?;
        }
        Ok(())
    }

    /// Visit every `(key, tuple)` with `key >= start`, in key order, stopping early when the
    /// visitor returns `Ok(false)`. Like [`scan_with`](Self::scan_with) but beginning at the leaf
    /// that would hold `start` (found by a descent) instead of the leftmost leaf — the resumable
    /// form a batched consumer uses to continue a scan after releasing and reacquiring its latches:
    /// the tree may have changed in between, so each call re-descends to `start` in the current
    /// tree and the B-link right-links then cover every later key regardless of concurrent splits.
    ///
    /// # Errors
    /// Propagates page-store read errors and any error the visitor returns.
    pub fn scan_from_with<F>(&self, start: u64, mut f: F) -> Result<()>
    where
        F: FnMut(u64, &[u8]) -> Result<bool>,
    {
        let (_, _, leaf) = self.descend(start)?;
        let mut page = leaf;
        let mut first = true;
        loop {
            let mut stop = false;
            let mut err = None;
            node::for_each_leaf_entry(&page, |key, tuple| {
                if stop || err.is_some() {
                    return;
                }
                // Skip entries before `start` only in the first (descended-to) leaf; every later
                // leaf via the right link is entirely past it.
                if first && key < start {
                    return;
                }
                match f(key, tuple) {
                    Ok(true) => {},
                    Ok(false) => stop = true,
                    Err(e) => err = Some(e),
                }
            });
            first = false;
            if let Some(e) = err {
                return Err(e);
            }
            if stop {
                return Ok(());
            }
            let Some(link) = node::right_link(&page) else {
                break;
            };
            page = self.store.read_page(link)?;
        }
        Ok(())
    }

    /// Every page id reachable from the root — the whole tree, for page reclamation.
    ///
    /// Walks level by level: each level is fully covered by following right links from its
    /// leftmost node, so siblings whose separators were split-published (or not yet) are all
    /// reached regardless of parent state.
    pub fn pages(&self) -> Result<Vec<PageId>> {
        let mut out = Vec::new();
        let mut level_head = Some(self.root);
        while let Some(head) = level_head {
            let mut next_level = None;
            let mut at = head;
            loop {
                let page = self.store.read_page(at)?;
                out.push(at);
                if next_level.is_none() && !node::is_leaf(&page) {
                    next_level = Some(
                        node::interior_entries(&page)
                            .first()
                            .map_or_else(|| node::rightmost_child(&page), |(_, child)| *child),
                    );
                }
                match node::right_link(&page) {
                    Some(link) => at = link,
                    None => break,
                }
            }
            level_head = next_level;
        }
        Ok(out)
    }

    /// Write a leaf back, splitting (possibly cascading up the recorded `path`) when the entries
    /// no longer fit.
    ///
    /// Tuples are variable-length, so the split point is chosen by **accumulated bytes**, not
    /// entry count — a count-midpoint could leave a half that still overflows (e.g. two small
    /// rows and one near-`MAX_TUPLE` row), spuriously failing a legal insert. Worse, no *single*
    /// split point may exist (`[small, huge, small]` fits no two pages), so the overflow is
    /// packed first-fit into as many chunks as needed: chunk 0 stays in the split node, each
    /// later chunk becomes a new right sibling. Every entry fits a page alone (`MAX_TUPLE` is
    /// enforced at the insert/update boundary), so the packing always succeeds.
    fn write_leaf_or_split(
        &mut self,
        path: &[(PageId, Page)],
        leaf_id: PageId,
        leaf: &Page,
        entries: &[(u64, Vec<u8>)],
    ) -> Result<()> {
        let mut page = *leaf;
        if node::write_leaf_entries(&mut page, entries) {
            self.store.write_page(leaf_id, &page)?;
            return Ok(());
        }
        // First-fit chunking by byte size, order-preserving; each chunk fits one page.
        let mut chunks: Vec<Vec<(u64, Vec<u8>)>> = Vec::new();
        let mut current: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut used = 0_usize;
        for entry in entries {
            let size = node::leaf_entry_size(&entry.1);
            if !current.is_empty() && used + size > node::LEAF_CAPACITY {
                chunks.push(std::mem::take(&mut current));
                used = 0;
            }
            used += size;
            current.push(entry.clone());
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        if chunks.len() < 2 {
            // A single chunk that did not fit above means an entry exceeded MAX_TUPLE — checked
            // at the insert/update boundary, so this is unreachable-but-loud.
            return Err(corrupt("leaf overflow produced no split point"));
        }

        // Build the new right siblings RIGHT-TO-LEFT so each node links to its (already-built)
        // right neighbor: the rightmost inherits the split node's old link + high key; every
        // other node links to its neighbor with the neighbor's first key as its high key.
        let mut ids = vec![leaf_id];
        for _ in 1..chunks.len() {
            ids.push(self.store.allocate_page()?);
        }
        let old_link = node::right_link(&page);
        let old_high = node::high_key(&page);
        let mut publishes: Vec<(PageId, u64, PageId)> = Vec::new(); // (left, separator, right)
        for index in (0..chunks.len()).rev() {
            let (Some(id), Some(chunk)) = (ids.get(index), chunks.get(index)) else {
                return Err(corrupt("leaf split chunk bookkeeping out of range"));
            };
            let mut fresh = [0u8; nusadb_core::PAGE_SIZE];
            node::init_leaf(&mut fresh);
            if let (Some(right_id), Some(right_chunk)) = (ids.get(index + 1), chunks.get(index + 1))
            {
                let separator = right_chunk.first().map_or(node::INF_KEY, |(k, _)| *k);
                node::set_right_link(&mut fresh, Some(*right_id));
                node::set_high_key(&mut fresh, separator);
                publishes.push((*id, separator, *right_id));
            } else {
                node::set_right_link(&mut fresh, old_link);
                node::set_high_key(&mut fresh, old_high);
            }
            if !node::write_leaf_entries(&mut fresh, chunk) {
                return Err(corrupt("leaf split chunk failed to serialize"));
            }
            // Lehman–Yao publish order: every right sibling is written before its left
            // neighbor (this loop runs right-to-left), and the parent last (below).
            self.store.write_page(*id, &fresh)?;
        }
        // Publish the separators left-to-right; `publish_separator` locates each left child by
        // chasing parent right-links, so an interior split triggered by an earlier publish
        // cannot orphan a later one.
        publishes.reverse();
        for (left, separator, right) in publishes {
            self.publish_separator(path.to_vec(), left, separator, right)?;
        }
        Ok(())
    }

    /// Insert `(separator, right_id)` into the parent of `left_id`, splitting interior nodes
    /// upward as needed; grow a new root when the old root split.
    fn publish_separator(
        &mut self,
        mut path: Vec<(PageId, Page)>,
        left_id: PageId,
        separator: u64,
        right_id: PageId,
    ) -> Result<()> {
        let Some((mut parent_id, _stale_parent)) = path.pop() else {
            // The root itself split: grow a new interior root over (left, right).
            let new_root = self.store.allocate_page()?;
            let mut page = [0u8; nusadb_core::PAGE_SIZE];
            node::init_interior(&mut page, right_id);
            if !node::write_interior_entries(&mut page, &[(separator, left_id)]) {
                return Err(corrupt("fresh root failed to serialize"));
            }
            self.store.write_page(new_root, &page)?;
            self.root = new_root;
            return Ok(());
        };

        // Locate the parent that actually holds `left_id`'s slot: the recorded path can be stale
        // — an earlier publish (a multi-chunk leaf split issues several) may have split this very
        // parent, moving the slot into a right sibling. Chasing the B-link by CHILD (not by key)
        // is exact: the slot lives in precisely one node of the chain.
        let mut page = self.store.read_page(parent_id)?;
        loop {
            let holds = node::interior_entries(&page)
                .iter()
                .any(|(_, child)| *child == left_id)
                || node::rightmost_child(&page) == left_id;
            if holds {
                break;
            }
            let Some(link) = node::right_link(&page) else {
                return Err(corrupt("split child missing from its parent chain"));
            };
            parent_id = link;
            page = self.store.read_page(parent_id)?;
        }

        // Insert the separator into the parent: the split child's slot is refined. Where the
        // split child was an entry `(sep, left)`, it becomes `(separator, left), (sep, right)`;
        // where it was the rightmost child, `(separator, left)` is appended and `right` becomes
        // the rightmost.
        let mut entries = node::interior_entries(&page);
        if let Some(pos) = entries.iter().position(|(_, child)| *child == left_id) {
            let old_sep = entries.get(pos).map_or(node::INF_KEY, |(sep, _)| *sep);
            if let Some(slot) = entries.get_mut(pos) {
                *slot = (separator, left_id);
            }
            entries.insert(pos + 1, (old_sep, right_id));
        } else {
            if node::rightmost_child(&page) != left_id {
                return Err(corrupt("split child missing from its parent"));
            }
            entries.push((separator, left_id));
            node::set_rightmost_child(&mut page, right_id);
        }
        if node::write_interior_entries(&mut page, &entries) {
            self.store.write_page(parent_id, &page)?;
            return Ok(());
        }

        // The parent overflows too: split it and recurse. Middle separator moves UP (interior
        // split), lower half keeps entries below it, upper half takes those above; the middle
        // entry's child becomes the lower half's rightmost.
        let mid = entries.len() / 2;
        let (mid_sep, mid_child) = entries
            .get(mid)
            .copied()
            .ok_or_else(|| corrupt("interior split midpoint out of range"))?;
        let lower: Vec<_> = entries.get(..mid).unwrap_or_default().to_vec();
        let upper: Vec<_> = entries.get(mid + 1..).unwrap_or_default().to_vec();

        let new_right_id = self.store.allocate_page()?;
        let mut right = [0u8; nusadb_core::PAGE_SIZE];
        node::init_interior(&mut right, node::rightmost_child(&page));
        node::set_right_link(&mut right, node::right_link(&page));
        node::set_high_key(&mut right, node::high_key(&page));
        if !node::write_interior_entries(&mut right, &upper) {
            return Err(corrupt("interior split upper half failed to serialize"));
        }

        let mut left = page;
        node::set_rightmost_child(&mut left, mid_child);
        node::set_right_link(&mut left, Some(new_right_id));
        node::set_high_key(&mut left, mid_sep);
        if !node::write_interior_entries(&mut left, &lower) {
            return Err(corrupt("interior split lower half failed to serialize"));
        }

        self.store.write_page(new_right_id, &right)?;
        self.store.write_page(parent_id, &left)?;
        self.publish_separator(path, parent_id, mid_sep, new_right_id)
    }
}
