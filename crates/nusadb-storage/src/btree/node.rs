//! B-tree node layout — one node fits exactly in one 8 KiB page.
//!
//! Fixed-size entries keep splits simple and bug-resistant: each entry is two
//! little-endian `u64`s — `(key, payload)`. For a **leaf** the payload is the stored
//! value; for an **internal** node the payload is the page id of the child to the
//! *right* of the key. The leftmost child of an internal node lives in the header
//! `link` field; for a leaf, `link` is the next-leaf pointer (sibling chain for scans).
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic        (0x4254_4E44 = "BTND")
//! 4       1     node_type    (0 = leaf, 1 = internal)
//! 5       1     reserved
//! 6       2     key_count
//! 8       8     link         (leaf: next-leaf page id | internal: leftmost child)
//! 16      ...   entries: [(key u64, payload u64)] × key_count, sorted by key
//! ```

// Byte-level node codec: offset arithmetic + slicing are inherent and bounded by the
// 8 KiB page-format invariants.
#![allow(clippy::indexing_slicing)]

use nusadb_core::{PAGE_SIZE, PageId};

/// Magic identifying a B-tree node page: ASCII `"BTND"`.
pub const BTREE_MAGIC: u32 = 0x4254_4E44;

const HEADER_LEN: usize = 16;
const ENTRY_LEN: usize = 16;

/// Maximum `(key, payload)` entries that fit in one node.
pub const MAX_ENTRIES: usize = (PAGE_SIZE - HEADER_LEN) / ENTRY_LEN;

/// Minimum entries a **non-root** node must retain (delete).
///
/// A delete that drops a node below this triggers a borrow from, or a merge with, a sibling.
/// `MAX_ENTRIES / 2` keeps every merge within capacity: two underflowing siblings hold at most
/// `2·MIN_ENTRIES ≤ MAX_ENTRIES` entries (an internal merge also pulls one separator down, still
/// `≤ MAX_ENTRIES`).
pub const MIN_ENTRIES: usize = MAX_ENTRIES / 2;

/// Sentinel `link` value meaning "no page" (no next leaf).
pub const NO_PAGE: u64 = u64::MAX;

/// Whether a node stores values (leaf) or child pointers (internal).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    /// Stores `(key, value)` pairs and a next-leaf sibling pointer.
    Leaf,
    /// Stores `(separator_key, right_child)` pairs and a leftmost child.
    Internal,
}

/// A B-tree node occupying exactly one 8 KiB page.
#[derive(Clone)]
pub struct Node {
    bytes: [u8; PAGE_SIZE],
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("type", &self.node_type())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl Node {
    /// Create an empty node of the given type. Leaves start with `link = NO_PAGE`.
    #[must_use]
    pub fn new(node_type: NodeType) -> Self {
        let mut node = Self {
            bytes: [0u8; PAGE_SIZE],
        };
        node.bytes[0..4].copy_from_slice(&BTREE_MAGIC.to_le_bytes());
        node.bytes[4] = match node_type {
            NodeType::Leaf => 0,
            NodeType::Internal => 1,
        };
        node.set_link(NO_PAGE);
        node
    }

    /// Wrap raw page bytes read from storage **without validation**.
    // Takes ownership by value: the bytes come straight from `PageStore::read_page` (already
    // owned), so this is a move; passing `&` would force a redundant copy into the new node.
    #[allow(clippy::large_types_passed_by_value)]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; PAGE_SIZE]) -> Self {
        Self { bytes }
    }

    /// Validate raw page bytes as a B-tree node, returning `None` if the page is not one.
    ///
    /// Rejects a wrong [`BTREE_MAGIC`] and a key count larger than [`MAX_ENTRIES`] — the latter is
    /// the case that matters for safety: an out-of-range count would let [`key_at`](Self::key_at) /
    /// [`payload_at`](Self::payload_at) index past the 8 KiB page and panic. Every read that comes
    /// from storage goes through here (never the unchecked [`from_bytes`](Self::from_bytes)), so a
    /// corrupt page becomes a clean error rather than a panic.
    #[allow(
        clippy::large_types_passed_by_value,
        reason = "owned bytes from read_page; see from_bytes"
    )]
    #[must_use]
    pub fn try_from_bytes(bytes: [u8; PAGE_SIZE]) -> Option<Self> {
        let magic = bytemuck::pod_read_unaligned::<u32>(&bytes[0..4]);
        if magic != BTREE_MAGIC {
            return None;
        }
        let key_count = bytemuck::pod_read_unaligned::<u16>(&bytes[6..8]) as usize;
        if key_count > MAX_ENTRIES {
            return None;
        }
        Some(Self { bytes })
    }

    /// Borrow the raw bytes for persisting via `PageStore`.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.bytes
    }

    /// Node kind.
    #[must_use]
    pub const fn node_type(&self) -> NodeType {
        if self.bytes[4] == 0 {
            NodeType::Leaf
        } else {
            NodeType::Internal
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        bytemuck::pod_read_unaligned::<u16>(&self.bytes[6..8]) as usize
    }

    /// Whether the node has zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn set_len(&mut self, n: usize) {
        let n16 = n as u16;
        self.bytes[6..8].copy_from_slice(bytemuck::bytes_of(&n16));
    }

    /// Leaf: next-leaf page id (or [`NO_PAGE`]). Internal: leftmost child page id.
    #[must_use]
    pub fn link(&self) -> u64 {
        bytemuck::pod_read_unaligned::<u64>(&self.bytes[8..16])
    }

    /// Set the `link` field (next-leaf for a leaf, leftmost child for internal).
    pub fn set_link(&mut self, v: u64) {
        self.bytes[8..16].copy_from_slice(bytemuck::bytes_of(&v));
    }

    /// Key at entry `i`.
    #[must_use]
    pub fn key_at(&self, i: usize) -> u64 {
        let o = HEADER_LEN + i * ENTRY_LEN;
        bytemuck::pod_read_unaligned::<u64>(&self.bytes[o..o + 8])
    }

    /// Payload at entry `i` (value for a leaf, right-child page id for internal).
    #[must_use]
    pub fn payload_at(&self, i: usize) -> u64 {
        let o = HEADER_LEN + i * ENTRY_LEN + 8;
        bytemuck::pod_read_unaligned::<u64>(&self.bytes[o..o + 8])
    }

    fn set_entry(&mut self, i: usize, key: u64, payload: u64) {
        let o = HEADER_LEN + i * ENTRY_LEN;
        self.bytes[o..o + 8].copy_from_slice(bytemuck::bytes_of(&key));
        self.bytes[o + 8..o + 16].copy_from_slice(bytemuck::bytes_of(&payload));
    }

    /// Overwrite the payload of an existing entry (used for key updates).
    pub fn set_payload(&mut self, i: usize, payload: u64) {
        let o = HEADER_LEN + i * ENTRY_LEN + 8;
        self.bytes[o..o + 8].copy_from_slice(bytemuck::bytes_of(&payload));
    }

    /// Overwrite the key of an existing entry, keeping its payload (used to fix an internal node's
    /// separator after a delete borrow/rotate).
    pub fn set_key(&mut self, i: usize, key: u64) {
        let o = HEADER_LEN + i * ENTRY_LEN;
        self.bytes[o..o + 8].copy_from_slice(bytemuck::bytes_of(&key));
    }

    /// Whether the node cannot accept another entry.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= MAX_ENTRIES
    }

    /// Binary-search for `key`. `Ok(i)` if present at `i`, else `Err(i)` insertion point.
    ///
    /// # Errors
    /// Never errors; the `Result` encodes found-vs-insertion-point.
    pub fn search(&self, key: u64) -> core::result::Result<usize, usize> {
        let (mut lo, mut hi) = (0usize, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key_at(mid).cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    /// For an internal node: which child to descend for `key` — the number of
    /// separators `<= key`. Child `0` is [`Node::link`]; child `p>=1` is the payload
    /// of entry `p - 1`.
    #[must_use]
    pub fn child_index(&self, key: u64) -> usize {
        let (mut lo, mut hi) = (0usize, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_at(mid) <= key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Resolve child page id at child position `p` for an internal node.
    #[must_use]
    pub fn child_at(&self, p: usize) -> PageId {
        if p == 0 {
            PageId(self.link())
        } else {
            PageId(self.payload_at(p - 1))
        }
    }

    /// Insert `(key, payload)` at sorted position `idx`, shifting later entries right.
    /// The caller guarantees the node is not full.
    pub fn insert_entry(&mut self, idx: usize, key: u64, payload: u64) {
        let n = self.len();
        let start = HEADER_LEN + idx * ENTRY_LEN;
        let end = HEADER_LEN + n * ENTRY_LEN;
        self.bytes.copy_within(start..end, start + ENTRY_LEN);
        self.set_entry(idx, key, payload);
        self.set_len(n + 1);
    }

    /// Append `(key, payload)` after the last entry without shifting. Entries must stay
    /// sorted; used by split routines that append in order.
    pub fn push_entry(&mut self, key: u64, payload: u64) {
        let n = self.len();
        self.set_entry(n, key, payload);
        self.set_len(n + 1);
    }

    /// Remove the entry at `idx`, shifting later entries left to close the gap (delete).
    /// The caller guarantees `idx < len`.
    pub fn remove_entry(&mut self, idx: usize) {
        let n = self.len();
        let gap = HEADER_LEN + idx * ENTRY_LEN;
        let end = HEADER_LEN + n * ENTRY_LEN;
        self.bytes.copy_within(gap + ENTRY_LEN..end, gap);
        self.set_len(n - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_bytes_accepts_a_valid_node() {
        let node = Node::new(NodeType::Leaf);
        assert!(Node::try_from_bytes(*node.as_bytes()).is_some());
    }

    #[test]
    fn try_from_bytes_rejects_bad_magic() {
        let mut bytes = *Node::new(NodeType::Internal).as_bytes();
        bytes[0] ^= 0xFF; // corrupt the magic
        assert!(Node::try_from_bytes(bytes).is_none());
    }

    #[test]
    fn try_from_bytes_rejects_an_out_of_range_key_count() {
        // A corrupt key count past capacity would let `key_at`/`payload_at` index past the page.
        let mut bytes = *Node::new(NodeType::Leaf).as_bytes();
        bytes[6..8].copy_from_slice(&u16::MAX.to_le_bytes()); // 65535 >> MAX_ENTRIES
        assert!(Node::try_from_bytes(bytes).is_none());
        // A count exactly at the limit is still accepted.
        bytes[6..8].copy_from_slice(&(MAX_ENTRIES as u16).to_le_bytes());
        assert!(Node::try_from_bytes(bytes).is_some());
    }
}
