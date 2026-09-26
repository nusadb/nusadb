//! B-link node codec over a raw 8 KiB page (ADR 008 §D2).
//!
//! Every node — leaf or interior — carries the two B-link fields (Lehman–Yao): a **right link**
//! to its right sibling and a **high key** (exclusive upper bound on the keys the node may
//! hold). A descent that finds its key at or beyond a node's high key chases the right link
//! instead of restarting: a concurrent (or crashed-midway) split is therefore
//! invisible to readers. `u64::MAX` in the link field means "no right sibling", and such a
//! node's high key is `+∞` (`u64::MAX`).
//!
//! Layout (little-endian):
//!
//! ```text
//! 0        1              9          17      19
//! [kind u8][right_link u64][high_key u64][count u16][entries …]
//! leaf     entry: [row_id u64][len u16][tuple bytes]         (variable, sorted by row_id)
//! interior header extra: [rightmost_child u64] at 19, entries from 27
//!          entry: [separator u64][child u64]                 (fixed 16 B, sorted by separator)
//! overflow [kind u8][next u64][used u16][payload …]           (a chain page of a large row)
//! ```
//!
//! A row that does not fit one leaf lives in an **overflow chain**: the leaf keeps a stub, flagged
//! by the high bit of the entry's length, holding the first [`OVERFLOW_INLINE_PREFIX`] bytes of
//! the value inline (the engine's per-version header, so visibility never touches the chain), the
//! value's total length, and the first chain page. The rest of the value is spread over chain
//! pages linked by `next`. The chain is written before the stub is published and is only freed
//! once no reader can still hold the stub, so latch-free readers never see a torn value.
//!
//! Interior semantics: entry `i` = `(sep_i, child_i)`; `child_i` covers keys `< sep_i` (and
//! `≥ sep_{i-1}`); keys `≥` the last separator go to `rightmost_child`. The separator pushed up
//! by a split is the right sibling's first key.

// Direct byte indexing/slicing is the point of a page codec: every range is bounded by the
// 8 KiB page-format invariants above (same policy as `nusadb-storage/page.rs`).
#![allow(clippy::indexing_slicing)]

use nusadb_core::{PAGE_SIZE, PageId};

/// "No right sibling" sentinel for the link field.
const NO_LINK: u64 = u64::MAX;
/// The high key of a node with no right sibling: exclusive upper bound `+∞`.
pub const INF_KEY: u64 = u64::MAX;

const KIND_LEAF: u8 = 0;
const KIND_INTERIOR: u8 = 1;
const KIND_OVERFLOW: u8 = 2;

const OFF_KIND: usize = 0;
const OFF_RIGHT_LINK: usize = 1;
const OFF_HIGH_KEY: usize = 9;
const OFF_COUNT: usize = 17;
const LEAF_ENTRIES: usize = 19;
const OFF_RIGHTMOST: usize = 19;
const INTERIOR_ENTRIES: usize = 27;
const INTERIOR_ENTRY: usize = 16;
/// Per-leaf-entry framing: row-id (8) + tuple length (2).
const LEAF_ENTRY_HEADER: usize = 10;
/// High bit of a leaf entry's length field: the entry is an overflow stub, not the value itself.
/// An inline length never exceeds [`MAX_TUPLE`], so the bit is free.
const OVERFLOW_FLAG: u16 = 0x8000;

/// The largest value a leaf entry holds inline (one entry filling an otherwise-empty leaf). A
/// larger value goes to an overflow chain and the leaf keeps a stub.
pub const MAX_TUPLE: usize = PAGE_SIZE - LEAF_ENTRIES - LEAF_ENTRY_HEADER;

/// Bytes of a chained value kept inline in its stub: the engine's fixed per-version header, so
/// deciding a version's visibility never dereferences the chain.
pub const OVERFLOW_INLINE_PREFIX: usize = 24;
/// An overflow stub: the inline prefix, the value's total length (`u32`), the first chain page.
pub const OVERFLOW_STUB: usize = OVERFLOW_INLINE_PREFIX + 4 + 8;

const OFF_OVERFLOW_NEXT: usize = 1;
const OFF_OVERFLOW_USED: usize = 9;
const OVERFLOW_ENTRIES: usize = 11;
/// Bytes of value one overflow chain page carries.
pub const OVERFLOW_PAYLOAD: usize = PAGE_SIZE - OVERFLOW_ENTRIES;

/// Bytes of entry payload a leaf can hold in total (the split packer's budget).
pub const LEAF_CAPACITY: usize = PAGE_SIZE - LEAF_ENTRIES;

/// The on-page size of one leaf entry holding `tuple` (for a chained row: its stub).
pub const fn leaf_entry_size(tuple: &[u8]) -> usize {
    LEAF_ENTRY_HEADER + tuple.len()
}

/// One leaf entry as read from or written to a page: the row-id, the bytes stored in the leaf,
/// and whether those bytes are an overflow stub rather than the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The row-id.
    pub key: u64,
    /// The inline value, or the stub of a chained one.
    pub stored: Vec<u8>,
    /// Whether `stored` is a stub.
    pub overflow: bool,
}

type Page = nusadb_core::traits::Page;

// --- shared header ------------------------------------------------------------------------------

/// Whether `page` is a leaf node.
pub const fn is_leaf(page: &Page) -> bool {
    page[OFF_KIND] == KIND_LEAF
}

/// The node's right sibling, if any (the B-link).
pub fn right_link(page: &Page) -> Option<PageId> {
    let raw = read_u64(page, OFF_RIGHT_LINK);
    (raw != NO_LINK).then_some(PageId(raw))
}

/// Set (or clear) the node's right sibling link.
pub fn set_right_link(page: &mut Page, link: Option<PageId>) {
    write_u64(page, OFF_RIGHT_LINK, link.map_or(NO_LINK, |p| p.0));
}

/// Exclusive upper bound on the keys this node may hold (`INF_KEY` = no bound).
pub fn high_key(page: &Page) -> u64 {
    read_u64(page, OFF_HIGH_KEY)
}

/// Set the node's exclusive upper key bound.
pub fn set_high_key(page: &mut Page, key: u64) {
    write_u64(page, OFF_HIGH_KEY, key);
}

/// How many entries the node holds.
pub fn count(page: &Page) -> usize {
    usize::from(u16::from_le_bytes([page[OFF_COUNT], page[OFF_COUNT + 1]]))
}

fn set_count(page: &mut Page, n: usize) {
    let n = u16::try_from(n).unwrap_or(u16::MAX);
    page[OFF_COUNT..OFF_COUNT + 2].copy_from_slice(&n.to_le_bytes());
}

// --- leaf ---------------------------------------------------------------------------------------

/// Initialize `page` as an empty leaf with no right sibling.
pub fn init_leaf(page: &mut Page) {
    page.fill(0);
    page[OFF_KIND] = KIND_LEAF;
    set_right_link(page, None);
    set_high_key(page, INF_KEY);
    set_count(page, 0);
}

/// The entries of a leaf, in key order.
pub fn leaf_entries(page: &Page) -> Vec<Entry> {
    let mut out = Vec::with_capacity(count(page));
    for_each_leaf_entry(page, |key, stored, overflow| {
        out.push(Entry {
            key,
            stored: stored.to_vec(),
            overflow,
        });
    });
    out
}

/// Decode one entry's length field into `(stored length, is overflow stub)`.
fn entry_len(page: &Page, at: usize) -> (usize, bool) {
    let raw = u16::from_le_bytes([page[at + 8], page[at + 9]]);
    (usize::from(raw & !OVERFLOW_FLAG), raw & OVERFLOW_FLAG != 0)
}

/// Visit each entry of a leaf in key order as `(row_id, stored bytes, is overflow stub)`, the
/// bytes borrowed from the page.
///
/// The zero-copy walk behind the single-copy scan: the owning [`leaf_entries`] allocates a
/// `Vec` per row, which a read-only scan then copied a second time into its `Arc`.
pub fn for_each_leaf_entry<'a>(page: &'a Page, mut f: impl FnMut(u64, &'a [u8], bool)) {
    let mut at = LEAF_ENTRIES;
    for _ in 0..count(page) {
        let row_id = read_u64(page, at);
        let (len, overflow) = entry_len(page, at);
        let start = at + LEAF_ENTRY_HEADER;
        f(row_id, &page[start..start + len], overflow);
        at = start + len;
    }
}

/// The bytes stored under `key` in a leaf, borrowed from the page, and whether they are an
/// overflow stub — `None` if absent.
///
/// Entries are sorted by row-id, so the walk stops at the first key past `key`: the zero-
/// allocation point-lookup counterpart of [`for_each_leaf_entry`] (profiling found `get`
/// materializing every leaf entry per lookup).
pub fn leaf_find(page: &Page, key: u64) -> Option<(&[u8], bool)> {
    let mut at = LEAF_ENTRIES;
    for _ in 0..count(page) {
        let row_id = read_u64(page, at);
        let (len, overflow) = entry_len(page, at);
        let start = at + LEAF_ENTRY_HEADER;
        match row_id.cmp(&key) {
            std::cmp::Ordering::Equal => return Some((&page[start..start + len], overflow)),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => at = start + len,
        }
    }
    None
}

/// Rewrite a leaf's entry area from `entries` (sorted by the caller), preserving the header
/// fields. Returns `false` (leaving the page untouched) if the entries do not fit.
pub fn write_leaf_entries(page: &mut Page, entries: &[Entry]) -> bool {
    let needed: usize = entries
        .iter()
        .map(|e| LEAF_ENTRY_HEADER + e.stored.len())
        .sum();
    if LEAF_ENTRIES + needed > PAGE_SIZE || entries.len() > usize::from(u16::MAX) {
        return false;
    }
    page[LEAF_ENTRIES..].fill(0);
    let mut at = LEAF_ENTRIES;
    for entry in entries {
        write_u64(page, at, entry.key);
        let mut len = u16::try_from(entry.stored.len()).unwrap_or(u16::MAX);
        if entry.overflow {
            len |= OVERFLOW_FLAG;
        }
        page[at + 8..at + 10].copy_from_slice(&len.to_le_bytes());
        page[at + LEAF_ENTRY_HEADER..at + LEAF_ENTRY_HEADER + entry.stored.len()]
            .copy_from_slice(&entry.stored);
        at += LEAF_ENTRY_HEADER + entry.stored.len();
    }
    set_count(page, entries.len());
    true
}

// --- overflow chain ------------------------------------------------------------------------------

/// Encode a stub for a chained value: its first [`OVERFLOW_INLINE_PREFIX`] bytes, its total
/// length, and the first chain page.
pub fn encode_stub(prefix: &[u8], total_len: u32, first: PageId) -> Vec<u8> {
    let mut out = Vec::with_capacity(OVERFLOW_STUB);
    out.extend_from_slice(prefix);
    out.extend_from_slice(&total_len.to_le_bytes());
    out.extend_from_slice(&first.0.to_le_bytes());
    out
}

/// Decode a stub into `(inline prefix, total length, first chain page)`; `None` if malformed.
pub fn decode_stub(stub: &[u8]) -> Option<(&[u8], usize, PageId)> {
    if stub.len() != OVERFLOW_STUB {
        return None;
    }
    let prefix = stub.get(..OVERFLOW_INLINE_PREFIX)?;
    let total = u32::from_le_bytes(
        stub.get(OVERFLOW_INLINE_PREFIX..OVERFLOW_INLINE_PREFIX + 4)?
            .try_into()
            .ok()?,
    );
    let first = u64::from_le_bytes(stub.get(OVERFLOW_INLINE_PREFIX + 4..)?.try_into().ok()?);
    Some((prefix, total as usize, PageId(first)))
}

/// Initialize `page` as an overflow chain page carrying `payload`, linked to `next`.
pub fn init_overflow(page: &mut Page, payload: &[u8], next: Option<PageId>) {
    page.fill(0);
    page[OFF_KIND] = KIND_OVERFLOW;
    write_u64(page, OFF_OVERFLOW_NEXT, next.map_or(NO_LINK, |p| p.0));
    let used = u16::try_from(payload.len()).unwrap_or(u16::MAX);
    page[OFF_OVERFLOW_USED..OFF_OVERFLOW_USED + 2].copy_from_slice(&used.to_le_bytes());
    page[OVERFLOW_ENTRIES..OVERFLOW_ENTRIES + payload.len()].copy_from_slice(payload);
}

/// Whether `page` is an overflow chain page.
#[must_use]
pub const fn is_overflow(page: &Page) -> bool {
    page[OFF_KIND] == KIND_OVERFLOW
}

/// The next page of an overflow chain, `None` at its end.
pub fn overflow_next(page: &Page) -> Option<PageId> {
    match read_u64(page, OFF_OVERFLOW_NEXT) {
        NO_LINK => None,
        id => Some(PageId(id)),
    }
}

/// The value bytes an overflow chain page carries.
pub fn overflow_payload(page: &Page) -> &[u8] {
    let used = usize::from(u16::from_le_bytes([
        page[OFF_OVERFLOW_USED],
        page[OFF_OVERFLOW_USED + 1],
    ]))
    .min(OVERFLOW_PAYLOAD);
    &page[OVERFLOW_ENTRIES..OVERFLOW_ENTRIES + used]
}

// --- interior -----------------------------------------------------------------------------------

/// Initialize `page` as an interior node whose only child is `rightmost` (covers all keys).
pub fn init_interior(page: &mut Page, rightmost: PageId) {
    page.fill(0);
    page[OFF_KIND] = KIND_INTERIOR;
    set_right_link(page, None);
    set_high_key(page, INF_KEY);
    set_count(page, 0);
    write_u64(page, OFF_RIGHTMOST, rightmost.0);
}

/// The child covering keys `≥` every separator.
pub fn rightmost_child(page: &Page) -> PageId {
    PageId(read_u64(page, OFF_RIGHTMOST))
}

/// Replace the child covering keys at or above every separator.
pub fn set_rightmost_child(page: &mut Page, child: PageId) {
    write_u64(page, OFF_RIGHTMOST, child.0);
}

/// The `(separator, child)` pairs of an interior node, in separator order.
pub fn interior_entries(page: &Page) -> Vec<(u64, PageId)> {
    let mut out = Vec::with_capacity(count(page));
    for i in 0..count(page) {
        let at = INTERIOR_ENTRIES + i * INTERIOR_ENTRY;
        out.push((read_u64(page, at), PageId(read_u64(page, at + 8))));
    }
    out
}

/// Rewrite an interior node's separator area (sorted by the caller), preserving the header and
/// rightmost child. Returns `false` (page untouched) if the entries do not fit.
pub fn write_interior_entries(page: &mut Page, entries: &[(u64, PageId)]) -> bool {
    if INTERIOR_ENTRIES + entries.len() * INTERIOR_ENTRY > PAGE_SIZE
        || entries.len() > usize::from(u16::MAX)
    {
        return false;
    }
    page[INTERIOR_ENTRIES..].fill(0);
    for (i, (sep, child)) in entries.iter().enumerate() {
        let at = INTERIOR_ENTRIES + i * INTERIOR_ENTRY;
        write_u64(page, at, *sep);
        write_u64(page, at + 8, child.0);
    }
    set_count(page, entries.len());
    true
}

/// The child an interior node routes `key` to: the first entry whose separator exceeds `key`,
/// else the rightmost child.
///
/// Walks the fixed-size entries in place — routing runs on every descent of every read and
/// write, so it must not materialize the separator list.
pub fn route(page: &Page, key: u64) -> PageId {
    for i in 0..count(page) {
        let at = INTERIOR_ENTRIES + i * INTERIOR_ENTRY;
        if key < read_u64(page, at) {
            return PageId(read_u64(page, at + 8));
        }
    }
    rightmost_child(page)
}

// --- helpers ------------------------------------------------------------------------------------

fn read_u64(page: &Page, at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&page[at..at + 8]);
    u64::from_le_bytes(b)
}

fn write_u64(page: &mut Page, at: usize, value: u64) {
    page[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
