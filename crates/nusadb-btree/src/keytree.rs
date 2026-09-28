//! A B+tree over pages, ordered by `(key bytes, row id)`: the page form of a secondary index.
//!
//! Each entry maps a `(key, row)` pair to a small opaque value (the engine stores the entry's
//! visibility ranges there). Leaves are linked both ways, so a range walks forward or backward
//! without going back to the root. The tree is not latch-free: its owner serializes writers and
//! keeps readers out while one runs (the engine's per-index latch), so a split never has to be
//! invisible to a concurrent reader. A leaf that empties is unlinked and freed whenever its parent
//! keeps another child, and a root left with a single child gives way to it, so an index whose
//! keys only grow while old ones are deleted does not accumulate empty pages.
//!
//! Nodes are slotted pages: after the header, a sorted array of 2-byte entry offsets; the
//! entries themselves are packed from the end of the page downward. A lookup is a binary search
//! over the offsets, an insert moves only offsets, and a page is compacted only when the holes
//! deletes left are needed. Pages are read and changed where they sit in the page cache.
//!
//! Layout (little-endian):
//!
//! ```text
//! leaf      [kind u8 = 3][count u16][heap u16][prev u64][next u64]  offsets from 21
//!           entry: [key_len u16][key][row u64][value_len u16][value]
//! interior  [kind u8 = 4][count u16][heap u16][leftmost child u64]   offsets from 13
//!           entry: [key_len u16][key][row u64][child u64]
//! ```
//!
//! `heap` is the lowest offset any entry starts at. Interior semantics: entry `i` is the first
//! `(key, row)` held under its child; a pair below the first separator lives under `leftmost`. An
//! entry larger than [`MAX_ENTRY_BYTES`] is refused by [`KeyTree::put`]; the caller keeps such
//! entries elsewhere ([`fits`] says which).

use std::cmp::Ordering;
use std::ops::Bound;

use nusadb_core::traits::Page;
use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};

use crate::store::PagedStore;

const KIND_LEAF: u8 = 3;
const KIND_INTERIOR: u8 = 4;
const NO_LINK: u64 = u64::MAX;
const LEAF_HEADER: usize = 21;
const INTERIOR_HEADER: usize = 13;
const OFF_COUNT: usize = 1;
const OFF_HEAP: usize = 3;
const OFF_PREV: usize = 5;
const OFF_NEXT: usize = 13;
const OFF_LEFTMOST: usize = 5;

/// The most bytes one encoded leaf entry may take: small enough that a split always leaves both
/// halves within a page and a separator always fits an interior node.
pub const MAX_ENTRY_BYTES: usize = 2000;

/// Whether an entry with a key of `key_len` bytes and a value of `value_len` bytes fits the tree.
pub const fn fits(key_len: usize, value_len: usize) -> bool {
    2 + key_len + 8 + 2 + value_len <= MAX_ENTRY_BYTES
}

fn compare(a_key: &[u8], a_row: u64, b_key: &[u8], b_row: u64) -> Ordering {
    a_key.cmp(b_key).then(a_row.cmp(&b_row))
}

fn corrupt(what: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("nusadb-btree: index page corrupt: {what}"),
    ))
}

fn get_u16(page: &Page, at: usize) -> Result<usize> {
    let b = page
        .get(at..at + 2)
        .ok_or_else(|| corrupt("field past the page end"))?;
    Ok(usize::from(u16::from_le_bytes([
        *b.first().unwrap_or(&0),
        *b.get(1).unwrap_or(&0),
    ])))
}

fn get_u64(page: &Page, at: usize) -> Result<u64> {
    let b = page
        .get(at..at + 8)
        .ok_or_else(|| corrupt("field past the page end"))?;
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    Ok(u64::from_le_bytes(a))
}

fn put_bytes(page: &mut Page, at: usize, bytes: &[u8]) -> Result<()> {
    page.get_mut(at..at + bytes.len())
        .ok_or_else(|| corrupt("write past the page end"))?
        .copy_from_slice(bytes);
    Ok(())
}

fn put_u16(page: &mut Page, at: usize, value: usize) -> Result<()> {
    let value = u16::try_from(value).map_err(|_| corrupt("16-bit field"))?;
    put_bytes(page, at, &value.to_le_bytes())
}

fn is_leaf(page: &Page) -> Result<bool> {
    match page.first() {
        Some(&KIND_LEAF) => Ok(true),
        Some(&KIND_INTERIOR) => Ok(false),
        _ => Err(corrupt("unknown node kind")),
    }
}

const fn header_len(leaf: bool) -> usize {
    if leaf { LEAF_HEADER } else { INTERIOR_HEADER }
}

/// Offset of the `i`-th entry of a node whose header is `header` bytes.
fn slot(page: &Page, header: usize, i: usize) -> Result<usize> {
    get_u16(page, header + 2 * i)
}

/// The `(key, row)` of the entry at `at`.
fn key_row(page: &Page, at: usize) -> Result<(&[u8], u64)> {
    let key_len = get_u16(page, at)?;
    let key = page
        .get(at + 2..at + 2 + key_len)
        .ok_or_else(|| corrupt("key past the page end"))?;
    Ok((key, get_u64(page, at + 2 + key_len)?))
}

/// The value of the leaf entry at `at`.
fn leaf_value(page: &Page, at: usize) -> Result<&[u8]> {
    let key_len = get_u16(page, at)?;
    let value_at = at + 2 + key_len + 8;
    let value_len = get_u16(page, value_at)?;
    page.get(value_at + 2..value_at + 2 + value_len)
        .ok_or_else(|| corrupt("value past the page end"))
}

/// Bytes the entry at `at` takes.
fn entry_len(page: &Page, leaf: bool, at: usize) -> Result<usize> {
    let key_len = get_u16(page, at)?;
    if leaf {
        let value_len = get_u16(page, at + 2 + key_len + 8)?;
        Ok(2 + key_len + 8 + 2 + value_len)
    } else {
        Ok(2 + key_len + 16)
    }
}

/// The child of the interior entry at `at`.
fn child_of(page: &Page, at: usize) -> Result<u64> {
    let key_len = get_u16(page, at)?;
    get_u64(page, at + 2 + key_len + 8)
}

/// Binary search a node for `(key, row)`: `Ok(i)` for the entry holding it, `Err(i)` for where it
/// would go.
fn search(
    page: &Page,
    leaf: bool,
    key: &[u8],
    row: u64,
) -> Result<std::result::Result<usize, usize>> {
    let header = header_len(leaf);
    let (mut lo, mut hi) = (0, get_u16(page, OFF_COUNT)?);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (k, r) = key_row(page, slot(page, header, mid)?)?;
        match compare(k, r, key, row) {
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
            Ordering::Equal => return Ok(Ok(mid)),
        }
    }
    Ok(Err(lo))
}

/// The child of the interior `page` that holds `(key, row)`: the one after the last separator at
/// or below it.
fn child_for(page: &Page, key: &[u8], row: u64) -> Result<u64> {
    let last = match search(page, false, key, row)? {
        Ok(i) => Some(i),
        Err(0) => None,
        Err(i) => Some(i - 1),
    };
    match last {
        None => get_u64(page, OFF_LEFTMOST),
        Some(i) => child_of(page, slot(page, INTERIOR_HEADER, i)?),
    }
}

fn last_child(page: &Page) -> Result<u64> {
    match get_u16(page, OFF_COUNT)? {
        0 => get_u64(page, OFF_LEFTMOST),
        n => child_of(page, slot(page, INTERIOR_HEADER, n - 1)?),
    }
}

/// Bytes the live entries of a node take.
fn live_bytes(page: &Page, leaf: bool) -> Result<usize> {
    let header = header_len(leaf);
    let mut total = 0;
    for i in 0..get_u16(page, OFF_COUNT)? {
        total += entry_len(page, leaf, slot(page, header, i)?)?;
    }
    Ok(total)
}

/// Insert the encoded `entry` as the `i`-th of the node, using the free gap between the offsets
/// and the heap. The caller has checked the gap is large enough.
fn insert_entry(page: &mut Page, leaf: bool, i: usize, entry: &[u8]) -> Result<()> {
    let header = header_len(leaf);
    let count = get_u16(page, OFF_COUNT)?;
    let heap = get_u16(page, OFF_HEAP)?
        .checked_sub(entry.len())
        .ok_or_else(|| corrupt("heap below the entry"))?;
    let slots_end = header + 2 * count;
    if i > count || slots_end + 2 > heap {
        return Err(corrupt("offsets overrun the heap"));
    }
    put_bytes(page, heap, entry)?;
    let from = header + 2 * i;
    page.copy_within(from..slots_end, from + 2);
    put_u16(page, from, heap)?;
    put_u16(page, OFF_COUNT, count + 1)?;
    put_u16(page, OFF_HEAP, heap)
}

/// Remove the `i`-th entry of the node; its bytes become a hole (zeroed), reclaimed at once when
/// it sits at the heap's edge and otherwise by the next compaction.
fn remove_entry(page: &mut Page, leaf: bool, i: usize) -> Result<()> {
    let header = header_len(leaf);
    let count = get_u16(page, OFF_COUNT)?;
    if i >= count || header + 2 * count > PAGE_SIZE {
        return Err(corrupt("offset out of range"));
    }
    let at = slot(page, header, i)?;
    let len = entry_len(page, leaf, at)?;
    page.get_mut(at..at + len)
        .ok_or_else(|| corrupt("entry past the page end"))?
        .fill(0);
    let from = header + 2 * i;
    page.copy_within(from + 2..header + 2 * count, from);
    put_bytes(page, header + 2 * (count - 1), &[0, 0])?;
    put_u16(page, OFF_COUNT, count - 1)?;
    let heap = get_u16(page, OFF_HEAP)?;
    if at == heap {
        put_u16(page, OFF_HEAP, heap + len)?;
    }
    Ok(())
}

/// The free gap between a node's offsets and its heap.
fn gap(page: &Page, leaf: bool) -> Result<usize> {
    let used = header_len(leaf) + 2 * get_u16(page, OFF_COUNT)?;
    Ok(get_u16(page, OFF_HEAP)?.saturating_sub(used))
}

fn encode_leaf_entry(key: &[u8], row: u64, value: &[u8]) -> Result<Vec<u8>> {
    let key_len = u16::try_from(key.len()).map_err(|_| corrupt("key length"))?;
    let value_len = u16::try_from(value.len()).map_err(|_| corrupt("value length"))?;
    let mut out = Vec::with_capacity(12 + key.len() + value.len());
    out.extend_from_slice(&key_len.to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&row.to_le_bytes());
    out.extend_from_slice(&value_len.to_le_bytes());
    out.extend_from_slice(value);
    Ok(out)
}

fn encode_separator(key: &[u8], row: u64, child: u64) -> Result<Vec<u8>> {
    let key_len = u16::try_from(key.len()).map_err(|_| corrupt("key length"))?;
    let mut out = Vec::with_capacity(18 + key.len());
    out.extend_from_slice(&key_len.to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&row.to_le_bytes());
    out.extend_from_slice(&child.to_le_bytes());
    Ok(out)
}

#[derive(Debug, Clone)]
struct Entry {
    key: Vec<u8>,
    row: u64,
    value: Vec<u8>,
}

impl Entry {
    const fn size(&self) -> usize {
        2 + self.key.len() + 8 + 2 + self.value.len()
    }
}

#[derive(Debug, Clone)]
struct Separator {
    key: Vec<u8>,
    row: u64,
    child: u64,
}

impl Separator {
    const fn size(&self) -> usize {
        2 + self.key.len() + 16
    }
}

struct Leaf {
    prev: u64,
    next: u64,
    entries: Vec<Entry>,
}

struct Interior {
    leftmost: u64,
    separators: Vec<Separator>,
}

impl Interior {
    fn children(&self) -> impl Iterator<Item = u64> + '_ {
        std::iter::once(self.leftmost).chain(self.separators.iter().map(|s| s.child))
    }
}

fn decode_leaf(page: &Page) -> Result<Leaf> {
    if !is_leaf(page)? {
        return Err(corrupt("expected a leaf"));
    }
    let count = get_u16(page, OFF_COUNT)?;
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let at = slot(page, LEAF_HEADER, i)?;
        let (key, row) = key_row(page, at)?;
        entries.push(Entry {
            key: key.to_vec(),
            row,
            value: leaf_value(page, at)?.to_vec(),
        });
    }
    Ok(Leaf {
        prev: get_u64(page, OFF_PREV)?,
        next: get_u64(page, OFF_NEXT)?,
        entries,
    })
}

fn decode_interior(page: &Page) -> Result<Interior> {
    if is_leaf(page)? {
        return Err(corrupt("expected an interior node"));
    }
    let count = get_u16(page, OFF_COUNT)?;
    let mut separators = Vec::with_capacity(count);
    for i in 0..count {
        let at = slot(page, INTERIOR_HEADER, i)?;
        let (key, row) = key_row(page, at)?;
        separators.push(Separator {
            key: key.to_vec(),
            row,
            child: child_of(page, at)?,
        });
    }
    Ok(Interior {
        leftmost: get_u64(page, OFF_LEFTMOST)?,
        separators,
    })
}

/// A compact node page from encoded `entries` (in order), or `None` when they do not fit.
fn build(leaf: bool, links: [u64; 2], entries: &[Vec<u8>]) -> Result<Option<Page>> {
    let header = header_len(leaf);
    let total: usize = entries.iter().map(Vec::len).sum::<usize>() + header + 2 * entries.len();
    if total > PAGE_SIZE {
        return Ok(None);
    }
    let mut page = [0u8; PAGE_SIZE];
    put_bytes(
        &mut page,
        0,
        &[if leaf { KIND_LEAF } else { KIND_INTERIOR }],
    )?;
    if leaf {
        put_bytes(&mut page, OFF_PREV, &links[0].to_le_bytes())?;
        put_bytes(&mut page, OFF_NEXT, &links[1].to_le_bytes())?;
    } else {
        put_bytes(&mut page, OFF_LEFTMOST, &links[0].to_le_bytes())?;
    }
    put_u16(&mut page, OFF_COUNT, 0)?;
    put_u16(&mut page, OFF_HEAP, PAGE_SIZE)?;
    for (i, entry) in entries.iter().enumerate() {
        insert_entry(&mut page, leaf, i, entry)?;
    }
    Ok(Some(page))
}

fn encode_leaf(leaf: &Leaf) -> Result<Option<Page>> {
    let entries = leaf
        .entries
        .iter()
        .map(|e| encode_leaf_entry(&e.key, e.row, &e.value))
        .collect::<Result<Vec<_>>>()?;
    build(true, [leaf.prev, leaf.next], &entries)
}

fn encode_interior(node: &Interior) -> Result<Option<Page>> {
    let entries = node
        .separators
        .iter()
        .map(|s| encode_separator(&s.key, s.row, s.child))
        .collect::<Result<Vec<_>>>()?;
    build(false, [node.leftmost, 0], &entries)
}

/// Rewrite a node with its holes squeezed out.
fn compact(page: &mut Page, leaf: bool) -> Result<()> {
    let header = header_len(leaf);
    let count = get_u16(page, OFF_COUNT)?;
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let at = slot(page, header, i)?;
        let len = entry_len(page, leaf, at)?;
        entries.push(
            page.get(at..at + len)
                .ok_or_else(|| corrupt("entry past the page end"))?
                .to_vec(),
        );
    }
    let links = if leaf {
        [get_u64(page, OFF_PREV)?, get_u64(page, OFF_NEXT)?]
    } else {
        [get_u64(page, OFF_LEFTMOST)?, 0]
    };
    *page = build(leaf, links, &entries)?.ok_or_else(|| corrupt("compaction overflow"))?;
    Ok(())
}

/// Insert the encoded `entry` as the `i`-th of the node in place: `true` when it fits (compacting
/// the node first if the free gap alone is too small), `false` when the node must split (it is
/// then untouched).
fn insert_in_place(page: &mut Page, leaf: bool, i: usize, entry: &[u8]) -> Result<bool> {
    if gap(page, leaf)? >= entry.len() + 2 {
        insert_entry(page, leaf, i, entry)?;
        return Ok(true);
    }
    let count = get_u16(page, OFF_COUNT)?;
    let needed = header_len(leaf) + 2 * (count + 1) + live_bytes(page, leaf)? + entry.len();
    if needed > PAGE_SIZE {
        return Ok(false);
    }
    compact(page, leaf)?;
    insert_entry(page, leaf, i, entry)?;
    Ok(true)
}

/// Put `(key, row) -> value` into the leaf `page` in place: `true` when it fit (the page is
/// changed), `false` when the leaf must split (the page is untouched).
fn put_in_place(page: &mut Page, key: &[u8], row: u64, value: &[u8]) -> Result<bool> {
    let entry = encode_leaf_entry(key, row, value)?;
    match search(page, true, key, row)? {
        Ok(i) => {
            let at = slot(page, LEAF_HEADER, i)?;
            let old = entry_len(page, true, at)?;
            if old == entry.len() {
                put_bytes(page, at, &entry)?;
                return Ok(true);
            }
            // A different size: the node must hold the new entry in place of the old one.
            let count = get_u16(page, OFF_COUNT)?;
            let needed = LEAF_HEADER + 2 * count + live_bytes(page, true)? - old + entry.len();
            if needed > PAGE_SIZE {
                return Ok(false);
            }
            remove_entry(page, true, i)?;
            insert_in_place(page, true, i, &entry)
        },
        Err(i) => insert_in_place(page, true, i, &entry),
    }
}

/// What removing an entry from a leaf did.
enum Removal {
    /// The entry was not there; the page is untouched.
    Absent,
    /// Removed; the leaf still holds entries.
    Kept,
    /// Removed, and the leaf is now empty; its `prev` and `next` links.
    Emptied(u64, u64),
}

/// Remove `(key, row)` from the leaf `page` in place.
fn delete_in_place(page: &mut Page, key: &[u8], row: u64) -> Result<Removal> {
    let Ok(i) = search(page, true, key, row)? else {
        return Ok(Removal::Absent);
    };
    remove_entry(page, true, i)?;
    if get_u16(page, OFF_COUNT)? == 0 {
        return Ok(Removal::Emptied(
            get_u64(page, OFF_PREV)?,
            get_u64(page, OFF_NEXT)?,
        ));
    }
    Ok(Removal::Kept)
}

/// The index at which items of `sizes` split into two halves of about equal bytes, each
/// non-empty.
fn split_point(sizes: impl Iterator<Item = usize> + Clone) -> usize {
    let total: usize = sizes.clone().sum();
    let count = sizes.clone().count();
    let mut acc = 0;
    for (i, size) in sizes.enumerate() {
        if acc + size > total / 2 && i > 0 {
            return i.min(count.saturating_sub(1)).max(1);
        }
        acc += size;
    }
    (count / 2).max(1)
}

/// Where a walk starts.
enum Target<'a> {
    First,
    Last,
    At(&'a [u8], u64),
}

/// A B+tree ordered by `(key bytes, row id)`; see the module docs.
pub struct KeyTree<'s> {
    store: &'s PagedStore,
    root: PageId,
}

impl std::fmt::Debug for KeyTree<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyTree")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl<'s> KeyTree<'s> {
    /// A new, empty tree: one empty leaf.
    ///
    /// # Errors
    /// Propagates page-store errors.
    pub fn create(store: &'s PagedStore) -> Result<Self> {
        let root = store.allocate_page()?;
        let page = build(true, [NO_LINK, NO_LINK], &[])?.ok_or_else(|| corrupt("empty leaf"))?;
        store.write_page(root, &page)?;
        Ok(Self { store, root })
    }

    /// The tree rooted at `root`.
    pub const fn open(store: &'s PagedStore, root: PageId) -> Self {
        Self { store, root }
    }

    /// The root page; it changes when the root splits or gives way to its only child.
    pub const fn root(&self) -> PageId {
        self.root
    }

    /// The value under `(key, row)`, if any.
    ///
    /// # Errors
    /// Propagates page-store errors; a malformed page is an error.
    pub fn get(&self, key: &[u8], row: u64) -> Result<Option<Vec<u8>>> {
        let (_, leaf_id) = self.descend(&Target::At(key, row))?;
        self.store
            .with_page(leaf_id, |page| -> Result<Option<Vec<u8>>> {
                match search(page, true, key, row)? {
                    Ok(i) => Ok(Some(
                        leaf_value(page, slot(page, LEAF_HEADER, i)?)?.to_vec(),
                    )),
                    Err(_) => Ok(None),
                }
            })?
    }

    /// Store `value` under `(key, row)`, replacing what is there. The leaf is changed in place;
    /// only a leaf that overflows is decoded and split.
    ///
    /// # Errors
    /// Refuses an entry that does not [`fits`]; propagates page-store errors.
    pub fn put(&mut self, key: &[u8], row: u64, value: &[u8]) -> Result<()> {
        if !fits(key.len(), value.len()) {
            return Err(corrupt("entry too large for an index page"));
        }
        let (path, leaf_id) = self.descend(&Target::At(key, row))?;
        let outcome = self.store.modify_page(leaf_id, |page| {
            match put_in_place(page, key, row, value) {
                Ok(true) => (Ok(None), true),
                Ok(false) => (decode_leaf(page).map(Some), false),
                Err(e) => (Err(e), false),
            }
        })??;
        let Some(mut leaf) = outcome else {
            return Ok(());
        };
        let entry = Entry {
            key: key.to_vec(),
            row,
            value: value.to_vec(),
        };
        match leaf
            .entries
            .binary_search_by(|e| compare(&e.key, e.row, key, row))
        {
            Ok(i) => {
                if let Some(slot) = leaf.entries.get_mut(i) {
                    *slot = entry;
                }
            },
            Err(i) => leaf.entries.insert(i, entry),
        }
        self.split_leaf(path, leaf_id, leaf)
    }

    /// Remove `(key, row)`; whether it was there.
    ///
    /// # Errors
    /// Propagates page-store errors.
    pub fn delete(&mut self, key: &[u8], row: u64) -> Result<bool> {
        let (path, leaf_id) = self.descend(&Target::At(key, row))?;
        let outcome =
            self.store
                .modify_page(leaf_id, |page| match delete_in_place(page, key, row) {
                    Ok(Removal::Absent) => (Ok(Removal::Absent), false),
                    Ok(removal) => (Ok(removal), true),
                    Err(e) => (Err(e), false),
                })??;
        match outcome {
            Removal::Absent => Ok(false),
            Removal::Kept => Ok(true),
            Removal::Emptied(prev, next) => {
                self.remove_empty_leaf(&path, leaf_id, prev, next)?;
                Ok(true)
            },
        }
    }

    /// Take the now-empty leaf `leaf_id` (linked to `prev` and `next`) out of the tree, together
    /// with every ancestor left with no other child: the chain above it is cut at the first
    /// ancestor that keeps another child, the leaf is unlinked from its neighbours, and the leaf
    /// and the chain are freed. When no ancestor keeps another child the leaf is all the tree
    /// holds, so it becomes the root and only the chain above it goes. A root left with a single
    /// child then gives way to it. Whether the leaf itself was removed.
    fn remove_empty_leaf(
        &mut self,
        path: &[PageId],
        leaf_id: PageId,
        prev: u64,
        next: u64,
    ) -> Result<bool> {
        let mut child = leaf_id;
        let mut chain = Vec::new();
        for &node_id in path.iter().rev() {
            let mut node = decode_interior(&self.store.read_page(node_id)?)?;
            if node.separators.is_empty() {
                // `child` is this node's only child: the node empties with it.
                chain.push(node_id);
                child = node_id;
                continue;
            }
            if node.leftmost == child.0 {
                // The next child takes over the lowest range; its separator goes.
                let first = node.separators.remove(0);
                node.leftmost = first.child;
            } else if let Some(at) = node.separators.iter().position(|s| s.child == child.0) {
                node.separators.remove(at);
            } else {
                return Err(corrupt("child not named by its parent"));
            }
            let page = encode_interior(&node)?.ok_or_else(|| corrupt("shrunken interior"))?;
            self.store.write_page(node_id, &page)?;
            for (link, is_next) in [(prev, false), (next, true)] {
                if link == NO_LINK {
                    continue;
                }
                let (field, value) = if is_next {
                    (OFF_PREV, prev)
                } else {
                    (OFF_NEXT, next)
                };
                self.store.modify_page(PageId(link), |page| {
                    (put_bytes(page, field, &value.to_le_bytes()), true)
                })??;
            }
            self.store.deallocate_page(leaf_id)?;
            for id in chain {
                self.store.deallocate_page(id)?;
            }
            self.collapse_root()?;
            return Ok(true);
        }
        // Every ancestor had only this child: the empty leaf is the whole tree.
        if !chain.is_empty() {
            self.root = leaf_id;
            for id in chain {
                self.store.deallocate_page(id)?;
            }
        }
        Ok(false)
    }

    /// Let a root interior with a single child give way to it, as many levels as that holds.
    fn collapse_root(&mut self) -> Result<()> {
        loop {
            let collapse = self
                .store
                .with_page(self.root, |page| -> Result<Option<u64>> {
                    if is_leaf(page)? || get_u16(page, OFF_COUNT)? > 0 {
                        return Ok(None);
                    }
                    get_u64(page, OFF_LEFTMOST).map(Some)
                })??;
            let Some(child) = collapse else {
                return Ok(());
            };
            let old = self.root;
            self.root = PageId(child);
            self.store.deallocate_page(old)?;
        }
    }

    /// Call `f(key, row, value)` for every entry whose key lies within `lo..hi`, in ascending
    /// order (descending when `backward`), until it returns `false`.
    ///
    /// # Errors
    /// Propagates page-store errors and `f`'s errors.
    pub fn scan<F>(
        &self,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        backward: bool,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&[u8], u64, &[u8]) -> Result<bool>,
    {
        let below_lo = |key: &[u8]| match lo {
            Bound::Included(b) => key < b,
            Bound::Excluded(b) => key <= b,
            Bound::Unbounded => false,
        };
        let above_hi = |key: &[u8]| match hi {
            Bound::Included(b) => key > b,
            Bound::Excluded(b) => key >= b,
            Bound::Unbounded => false,
        };
        let start = if backward {
            match hi {
                Bound::Included(b) | Bound::Excluded(b) => Target::At(b, u64::MAX),
                Bound::Unbounded => Target::Last,
            }
        } else {
            match lo {
                Bound::Included(b) | Bound::Excluded(b) => Target::At(b, 0),
                Bound::Unbounded => Target::First,
            }
        };
        let (_, mut leaf_id) = self.descend(&start)?;
        loop {
            let next = self
                .store
                .with_page(leaf_id, |page| -> Result<Option<u64>> {
                    let count = get_u16(page, OFF_COUNT)?;
                    for n in 0..count {
                        let i = if backward { count - 1 - n } else { n };
                        let at = slot(page, LEAF_HEADER, i)?;
                        let (key, row) = key_row(page, at)?;
                        let (skip, stop) = if backward {
                            (above_hi(key), below_lo(key))
                        } else {
                            (below_lo(key), above_hi(key))
                        };
                        if skip {
                            continue;
                        }
                        if stop || !f(key, row, leaf_value(page, at)?)? {
                            return Ok(None);
                        }
                    }
                    let link = get_u64(page, if backward { OFF_PREV } else { OFF_NEXT })?;
                    Ok((link != NO_LINK).then_some(link))
                })??;
            match next {
                Some(link) => leaf_id = PageId(link),
                None => return Ok(()),
            }
        }
    }

    /// Every page of the tree.
    ///
    /// # Errors
    /// Propagates page-store errors.
    pub fn pages(&self) -> Result<Vec<PageId>> {
        let mut out = Vec::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            out.push(id);
            let page = self.store.read_page(id)?;
            if !is_leaf(&page)? {
                stack.extend(decode_interior(&page)?.children().map(PageId));
            }
        }
        Ok(out)
    }

    /// Descend to the leaf for `target`, returning the interior pages passed on the way (root
    /// first) and the leaf. Interior nodes are read in place.
    fn descend(&self, target: &Target<'_>) -> Result<(Vec<PageId>, PageId)> {
        let mut path = Vec::new();
        let mut id = self.root;
        loop {
            let child = self.store.with_page(id, |page| -> Result<Option<u64>> {
                if is_leaf(page)? {
                    return Ok(None);
                }
                Ok(Some(match target {
                    Target::First => get_u64(page, OFF_LEFTMOST)?,
                    Target::Last => last_child(page)?,
                    Target::At(key, row) => child_for(page, key, *row)?,
                }))
            })??;
            let Some(child) = child else {
                return Ok((path, id));
            };
            path.push(id);
            id = PageId(child);
            if path.len() > 64 {
                return Err(corrupt("tree deeper than any valid one"));
            }
        }
    }

    /// Split the over-full `leaf` at `leaf_id` in two and push the new separator up `path`.
    fn split_leaf(&mut self, path: Vec<PageId>, leaf_id: PageId, mut leaf: Leaf) -> Result<()> {
        let mid = split_point(leaf.entries.iter().map(Entry::size));
        let right_entries = leaf.entries.split_off(mid);
        let right_id = self.store.allocate_page()?;
        let first = right_entries
            .first()
            .ok_or_else(|| corrupt("empty half after a split"))?;
        let separator = Separator {
            key: first.key.clone(),
            row: first.row,
            child: right_id.0,
        };
        let right = Leaf {
            prev: leaf_id.0,
            next: leaf.next,
            entries: right_entries,
        };
        if right.next != NO_LINK {
            self.store.modify_page(PageId(right.next), |page| {
                (put_bytes(page, OFF_PREV, &right_id.0.to_le_bytes()), true)
            })??;
        }
        leaf.next = right_id.0;
        let right_page = encode_leaf(&right)?.ok_or_else(|| corrupt("right half"))?;
        let left_page = encode_leaf(&leaf)?.ok_or_else(|| corrupt("left half"))?;
        self.store.write_page(right_id, &right_page)?;
        self.store.write_page(leaf_id, &left_page)?;
        self.insert_separator(path, leaf_id, separator)
    }

    /// Insert `separator` (for the node split off to the right of `left`) into the parent at the
    /// end of `path`, splitting upward as needed; a split root gets a new root above it.
    fn insert_separator(
        &mut self,
        mut path: Vec<PageId>,
        left: PageId,
        separator: Separator,
    ) -> Result<()> {
        let Some(parent_id) = path.pop() else {
            let root = self.store.allocate_page()?;
            let page = encode_interior(&Interior {
                leftmost: left.0,
                separators: vec![separator],
            })?
            .ok_or_else(|| corrupt("new root"))?;
            self.store.write_page(root, &page)?;
            self.root = root;
            return Ok(());
        };
        let entry = encode_separator(&separator.key, separator.row, separator.child)?;
        let inserted = self.store.modify_page(parent_id, |page| {
            let placed = search(page, false, &separator.key, separator.row).and_then(|at| {
                let i = match at {
                    Ok(i) | Err(i) => i,
                };
                insert_in_place(page, false, i, &entry)
            });
            match placed {
                Ok(true) => (Ok(true), true),
                other => (other, false),
            }
        })??;
        if inserted {
            return Ok(());
        }
        // Split the interior: the middle separator moves up, its child leads the right half.
        let mut node = decode_interior(&self.store.read_page(parent_id)?)?;
        let at = node.separators.partition_point(|s| {
            compare(&s.key, s.row, &separator.key, separator.row) == Ordering::Less
        });
        node.separators.insert(at, separator);
        let mid = split_point(node.separators.iter().map(Separator::size));
        let mut right_separators = node.separators.split_off(mid);
        if right_separators.is_empty() {
            return Err(corrupt("interior split with an empty half"));
        }
        let promoted = right_separators.remove(0);
        let right_id = self.store.allocate_page()?;
        let right = Interior {
            leftmost: promoted.child,
            separators: right_separators,
        };
        let right_page = encode_interior(&right)?.ok_or_else(|| corrupt("right interior"))?;
        let left_page = encode_interior(&node)?.ok_or_else(|| corrupt("left interior"))?;
        self.store.write_page(right_id, &right_page)?;
        self.store.write_page(parent_id, &left_page)?;
        self.insert_separator(
            path,
            parent_id,
            Separator {
                key: promoted.key,
                row: promoted.row,
                child: right_id.0,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::store::PagedStore;

    fn key(i: u64, len: usize) -> Vec<u8> {
        let mut k = format!("{i:012}").into_bytes();
        k.resize(len.max(12), b'k');
        k
    }

    /// Random-order puts, replaces and deletes against a `BTreeMap` model: every lookup and
    /// every forward and backward range agrees, across many splits.
    #[test]
    fn matches_an_ordered_map_under_random_operations() {
        let store = PagedStore::default();
        let mut tree = KeyTree::create(&store).unwrap();
        let mut model: BTreeMap<(Vec<u8>, u64), Vec<u8>> = BTreeMap::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for step in 0..20_000_u64 {
            let r = next();
            let k = key(r % 3000, 12 + (r % 7) as usize * 40);
            let row = (r >> 20) % 4;
            if r % 5 == 0 {
                let was = model.remove(&(k.clone(), row)).is_some();
                assert_eq!(tree.delete(&k, row).unwrap(), was, "step {step}");
            } else {
                // Values of varying size, so replacing an entry resizes it and holes appear.
                let value = step.to_le_bytes().repeat(1 + (r % 13) as usize);
                tree.put(&k, row, &value).unwrap();
                model.insert((k, row), value);
            }
        }
        for ((k, row), v) in &model {
            assert_eq!(tree.get(k, *row).unwrap().as_ref(), Some(v));
        }
        assert_eq!(tree.get(b"absent", 0).unwrap(), None);
        let lo = key(700, 12);
        let hi = key(2100, 12);
        let want: Vec<(Vec<u8>, u64)> = model
            .keys()
            .filter(|(k, _)| k.as_slice() >= lo.as_slice() && k.as_slice() < hi.as_slice())
            .cloned()
            .collect();
        let mut got = Vec::new();
        tree.scan(
            Bound::Included(&lo),
            Bound::Excluded(&hi),
            false,
            |k, r, _| {
                got.push((k.to_vec(), r));
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(got, want);
        let mut back = Vec::new();
        tree.scan(
            Bound::Included(&lo),
            Bound::Excluded(&hi),
            true,
            |k, r, _| {
                back.push((k.to_vec(), r));
                Ok(true)
            },
        )
        .unwrap();
        back.reverse();
        assert_eq!(back, want);
        let mut all = 0;
        tree.scan(Bound::Unbounded, Bound::Unbounded, false, |_, _, _| {
            all += 1;
            Ok(true)
        })
        .unwrap();
        assert_eq!(all, model.len());
        assert!(tree.pages().unwrap().len() > 10, "the test must split");
    }

    /// Deleting everything gives every page but one back, and an index whose keys only grow
    /// while the oldest are deleted stays at a bounded size.
    #[test]
    fn emptied_leaves_are_freed() {
        let store = PagedStore::default();
        let mut tree = KeyTree::create(&store).unwrap();
        for i in 0..5000 {
            tree.put(&key(i, 60), 0, b"value").unwrap();
        }
        assert!(tree.pages().unwrap().len() > 20);
        for i in 0..5000 {
            assert!(tree.delete(&key(i, 60), 0).unwrap());
        }
        assert_eq!(
            tree.pages().unwrap().len(),
            1,
            "one empty leaf is all that remains"
        );
        assert_eq!(store.live_pages().unwrap(), 1);
        // A sliding window: insert new keys, delete the oldest.
        for i in 0..20_000_u64 {
            tree.put(&key(i, 60), 0, b"value").unwrap();
            if i >= 1000 {
                assert!(tree.delete(&key(i - 1000, 60), 0).unwrap());
            }
        }
        let pages = tree.pages().unwrap().len();
        assert!(pages < 40, "a window of 1000 entries holds {pages} pages");
        assert_eq!(store.live_pages().unwrap(), pages);
        let mut n = 0;
        tree.scan(Bound::Unbounded, Bound::Unbounded, false, |_, _, _| {
            n += 1;
            Ok(true)
        })
        .unwrap();
        assert_eq!(n, 1000);
        let mut back = 0;
        tree.scan(Bound::Unbounded, Bound::Unbounded, true, |_, _, _| {
            back += 1;
            Ok(true)
        })
        .unwrap();
        assert_eq!(back, 1000);
    }

    /// A tree several levels deep drains back to one page: interiors left with a single, emptied
    /// child go with it, and a long sliding window over it stays bounded.
    #[test]
    fn a_deep_tree_drains_to_one_page_and_a_window_over_it_stays_bounded() {
        let store = PagedStore::default();
        let mut tree = KeyTree::create(&store).unwrap();
        for i in 0..100_000 {
            tree.put(&key(i, 60), 0, b"value").unwrap();
        }
        let loaded = tree.pages().unwrap().len();
        // Shrink to the newest 20,000, then slide that window across 300,000 more keys.
        for i in 0..80_000 {
            assert!(tree.delete(&key(i, 60), 0).unwrap());
        }
        for i in 100_000..400_000_u64 {
            tree.put(&key(i, 60), 0, b"value").unwrap();
            assert!(tree.delete(&key(i - 20_000, 60), 0).unwrap());
        }
        let window = tree.pages().unwrap().len();
        assert!(
            window < loaded / 2,
            "{window} pages for a 20,000 window (loaded {loaded})"
        );
        assert_eq!(store.live_pages().unwrap(), window);
        for i in 380_000..400_000 {
            assert!(tree.delete(&key(i, 60), 0).unwrap());
        }
        assert_eq!(tree.pages().unwrap().len(), 1);
        assert_eq!(store.live_pages().unwrap(), 1);
    }

    /// Exclusive and inclusive bounds on one key, an early stop, and an empty tree.
    #[test]
    fn bounds_and_early_stop() {
        let store = PagedStore::default();
        let mut tree = KeyTree::create(&store).unwrap();
        let mut none = 0;
        tree.scan(Bound::Unbounded, Bound::Unbounded, true, |_, _, _| {
            none += 1;
            Ok(true)
        })
        .unwrap();
        assert_eq!(none, 0);
        for i in 0..500 {
            for row in 0..3 {
                tree.put(&key(i, 100), row, b"v").unwrap();
            }
        }
        let k = key(250, 100);
        let mut rows = Vec::new();
        tree.scan(
            Bound::Included(&k),
            Bound::Included(&k),
            false,
            |_, r, _| {
                rows.push(r);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(rows, vec![0, 1, 2]);
        let mut n = 0;
        tree.scan(
            Bound::Excluded(&k),
            Bound::Excluded(&key(252, 100)),
            false,
            |_, _, _| {
                n += 1;
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(n, 3, "only key 251");
        let mut first = Vec::new();
        tree.scan(Bound::Unbounded, Bound::Unbounded, true, |kk, r, _| {
            first.push((kk.to_vec(), r));
            Ok(first.len() < 2)
        })
        .unwrap();
        assert_eq!(first, vec![(key(499, 100), 2), (key(499, 100), 1)]);
        assert!(!fits(MAX_ENTRY_BYTES, 0));
        assert!(tree.put(&vec![0u8; MAX_ENTRY_BYTES], 0, b"").is_err());
    }
}
