//! TOAST: out-of-line storage for oversized values.
//!
//! A tuple must fit in an 8 KiB page, so a value larger than a page (or large enough to crowd a
//! page) is stored **out of line**: its bytes are split across a singly-linked chain of dedicated
//! TOAST pages, and the tuple keeps only a small fixed-size [`ToastPointer`] (the head page id plus
//! the total length). This is the page-layer primitive a page-based tuple heap uses to spill large
//! attributes out of line so the inline tuple stays page-sized.
//!
//! # Page layout
//!
//! Each TOAST page is one 8 KiB [`PageStore`] page:
//!
//! ```text
//! [magic: u32 = TOAST_MAGIC][next: u64][chunk_len: u32][ chunk bytes ... ]
//! └──────────────────────── 16-byte header ───────────┘└ up to CHUNK_CAP ┘
//! ```
//!
//! `next` is the next page in the chain, or `NO_NEXT` (`u64::MAX`, never a valid allocation) on
//! the last page. `chunk_len` is how many of this page's `CHUNK_CAP` payload bytes are live. The
//! magic ([`TOAST_MAGIC`], ASCII `"TOST"`) is disjoint from the page, catalog, and free-list
//! magics, so a TOAST page is never confused with another kind of page.
//!
//! Storing allocates one page per chunk up front and links them; [`Toast::free`] walks the chain
//! and returns every page to the [`DiskManager`] free list, so deleting a large value reclaims its
//! space.

// Every slice/index here is bounded by construction: payload offsets are `HEADER + n` where
// `n <= CHUNK_CAP` (so `<= PAGE_SIZE`), `pages` always holds at least one element, and a page read
// from disk is always exactly `PAGE_SIZE` bytes. Chunk lengths read back from a page are validated
// against `CHUNK_CAP` before they index. So indexing cannot panic in range.
#![allow(clippy::indexing_slicing)]

use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};

use crate::DiskManager;

/// Magic stamped at the start of every TOAST page. ASCII `"TOST"`, disjoint from `PAGE_MAGIC`
/// (`"NUSA"`), `CATALOG_MAGIC` (`"CATL"`), and [`FREE_PAGE_MAGIC`](crate::disk::FREE_PAGE_MAGIC)
/// (`"FREE"`).
pub const TOAST_MAGIC: u32 = u32::from_le_bytes(*b"TOST");

/// Chain terminator stored in a page's `next` field. `u64::MAX` is never returned by
/// `allocate_page`, so it can never collide with a real page id.
const NO_NEXT: u64 = u64::MAX;

/// Per-page header size: `magic(4) + next(8) + chunk_len(4)`.
const HEADER: usize = 16;

/// Maximum payload bytes carried by one TOAST page.
const CHUNK_CAP: usize = PAGE_SIZE - HEADER;

/// Upper bound on the `Vec::with_capacity` hint when loading, so a corrupt [`ToastPointer`] with an
/// absurd `total_len` cannot trigger an allocation-size panic. The vector still grows past this via
/// `extend` for genuinely large values.
const MAX_PREALLOC: usize = 1 << 24; // 16 MiB

/// A compact reference to an out-of-line value, embedded in a tuple in place of the value itself.
/// Fixed 16 bytes on the wire ([`to_bytes`](ToastPointer::to_bytes)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToastPointer {
    /// First page of the value's chain.
    pub head: PageId,
    /// Total length of the reassembled value, in bytes.
    pub total_len: u64,
}

impl ToastPointer {
    /// Serialize to the fixed 16-byte in-tuple form: `head(8) ++ total_len(8)`, little-endian.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.head.0.to_le_bytes());
        out[8..].copy_from_slice(&self.total_len.to_le_bytes());
        out
    }

    /// Parse the fixed 16-byte in-tuple form produced by [`to_bytes`](ToastPointer::to_bytes).
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let mut head = [0u8; 8];
        head.copy_from_slice(&bytes[..8]);
        let mut len = [0u8; 8];
        len.copy_from_slice(&bytes[8..]);
        Self {
            head: PageId(u64::from_le_bytes(head)),
            total_len: u64::from_le_bytes(len),
        }
    }
}

/// Out-of-line large-value store over a [`PageStore`]. Borrows the store, like
/// [`BTree`](crate::BTree).
#[derive(Debug)]
pub struct Toast<'s, S: PageStore> {
    store: &'s S,
}

impl<'s, S: PageStore> Toast<'s, S> {
    /// Wrap `store`.
    #[must_use]
    pub const fn new(store: &'s S) -> Self {
        Self { store }
    }

    /// Store `value` out of line, returning the [`ToastPointer`] to embed in the tuple.
    ///
    /// Allocates one page per `CHUNK_CAP`-byte chunk (at least one page, even for an empty value),
    /// links them head-to-tail, and writes each. The pages are durable once the caller `fsync`s the
    /// store.
    ///
    /// # Errors
    /// Propagates store allocation/write errors.
    pub fn store_value(&self, value: &[u8]) -> Result<ToastPointer> {
        // `chunks` yields nothing for an empty slice; represent an empty value as one empty page so
        // the pointer always names a real head page.
        let chunk_count = value.len().div_ceil(CHUNK_CAP).max(1);
        let pages: Vec<PageId> = (0..chunk_count)
            .map(|_| self.store.allocate_page())
            .collect::<Result<_>>()?;

        let mut chunks = value.chunks(CHUNK_CAP);
        for (i, &page_id) in pages.iter().enumerate() {
            let chunk = chunks.next().unwrap_or(&[]);
            let next = pages.get(i + 1).map_or(NO_NEXT, |p| p.0);
            let mut buf = [0u8; PAGE_SIZE];
            buf[..4].copy_from_slice(&TOAST_MAGIC.to_le_bytes());
            buf[4..12].copy_from_slice(&next.to_le_bytes());
            buf[12..16].copy_from_slice(&(chunk.len() as u32).to_le_bytes());
            buf[HEADER..HEADER + chunk.len()].copy_from_slice(chunk);
            self.store.write_page(page_id, &buf)?;
        }

        Ok(ToastPointer {
            head: pages[0],
            total_len: value.len() as u64,
        })
    }

    /// Reassemble the value referenced by `ptr` by walking its page chain.
    ///
    /// # Errors
    /// Propagates store read errors, or returns [`Error::InvalidMagic`] if a page in the chain is
    /// not a TOAST page / has an out-of-range chunk length, or [`Error::Io`] if the chain's
    /// reassembled length does not match `ptr.total_len` (corruption / wrong pointer).
    pub fn load(&self, ptr: ToastPointer) -> Result<Vec<u8>> {
        let total = ptr.total_len as usize;
        let mut out = Vec::with_capacity(total.min(MAX_PREALLOC));

        // Bound the walk by the number of pages the declared length implies (+1 slack): a corrupt
        // `next` that forms a cycle or over-long chain is rejected rather than looped forever.
        let max_pages = total.div_ceil(CHUNK_CAP).max(1) + 1;
        let mut current = ptr.head.0;
        for _ in 0..max_pages {
            if current == NO_NEXT {
                break;
            }
            let page_id = PageId(current);
            let page = self.store.read_page(page_id)?;
            if u32::from_le_bytes([page[0], page[1], page[2], page[3]]) != TOAST_MAGIC {
                return Err(Error::InvalidMagic { page_id });
            }
            let next = u64::from_le_bytes(page[4..12].try_into().unwrap_or_default());
            let chunk_len =
                u32::from_le_bytes(page[12..16].try_into().unwrap_or_default()) as usize;
            if chunk_len > CHUNK_CAP {
                return Err(Error::InvalidMagic { page_id });
            }
            out.extend_from_slice(&page[HEADER..HEADER + chunk_len]);
            current = next;
            if next == NO_NEXT {
                break;
            }
        }

        if out.len() != total {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TOAST chain length {} != pointer total_len {total}",
                    out.len()
                ),
            )));
        }
        Ok(out)
    }
}

impl Toast<'_, DiskManager> {
    /// Free every page in `ptr`'s chain, returning them to the [`DiskManager`] free list so a
    /// deleted large value reclaims its space.
    ///
    /// Reclamation is not part of the `PageStore` treaty (which stays minimal), so `free` is
    /// available only over the concrete [`DiskManager`], whose
    /// [`deallocate_page`](DiskManager::deallocate_page) backs it.
    ///
    /// # Errors
    /// Propagates store read/deallocate errors. Stops at a non-TOAST page (treating it as the end
    /// of the owned chain) rather than freeing pages this value does not own.
    pub fn free(&self, ptr: ToastPointer) -> Result<()> {
        let max_pages = (ptr.total_len as usize).div_ceil(CHUNK_CAP).max(1) + 1;
        let mut current = ptr.head.0;
        for _ in 0..max_pages {
            if current == NO_NEXT {
                break;
            }
            let page_id = PageId(current);
            let page = self.store.read_page(page_id)?;
            if u32::from_le_bytes([page[0], page[1], page[2], page[3]]) != TOAST_MAGIC {
                break; // not ours; do not free
            }
            let next = u64::from_le_bytes(page[4..12].try_into().unwrap_or_default());
            self.store.deallocate_page(page_id)?;
            current = next;
        }
        Ok(())
    }
}
