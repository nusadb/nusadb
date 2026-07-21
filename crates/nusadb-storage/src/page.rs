//! 8 KiB page format with header + slot array + heap data.
//!
//! ```text
//! Offset  Size   Field
//! 0       4      magic        (0x4E55_5341 = "NUSA")
//! 4       4      version
//! 8       8      page_id
//! 16      8      lsn          (last WAL LSN that modified this page)
//! 24      4      checksum     (CRC32 of bytes [0..24) || [28..PAGE_SIZE))
//! 28      2      free_space_offset
//! 30      2      slot_count
//! 32      ...    slot array grows down ↓
//! ...     ...    free space
//! ...     ...    row data grows up ↑
//! 8192    end
//! ```
//!
//! See [`ARCHITECTURE.md`] for design rationale.

// This module is a byte-level page codec: offset arithmetic and slicing are inherent, and
// every range is bounded by the 8 KiB page-format invariants. `indexing_slicing` would
// otherwise force `.get()` noise on every field access here.
#![allow(clippy::indexing_slicing)]

use bytemuck::{Pod, Zeroable};
use nusadb_core::{Error, Lsn, PAGE_SIZE, PageId, Result, SlotIdx};

/// Magic bytes identifying a NusaDB page: ASCII `"NUSA"`.
pub const PAGE_MAGIC: u32 = 0x4E55_5341;

/// Current page format version. Bump on any incompatible layout change.
pub const PAGE_VERSION: u32 = 1;

/// Fixed-size header at the start of every 8 KiB page.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PageHeader {
    /// Magic bytes — must equal [`PAGE_MAGIC`].
    pub magic: u32,
    /// Page format version — must equal [`PAGE_VERSION`].
    pub version: u32,
    /// Page identifier within the disk file.
    pub page_id: PageId,
    /// Last WAL LSN that modified this page; `0` for never-written.
    pub lsn: Lsn,
    /// CRC32 of `[0..24) || [28..PAGE_SIZE)`.
    pub checksum: u32,
    /// Offset of the next free byte in the heap, growing upward from the bottom of the page.
    pub free_space_offset: u16,
    /// Number of slots currently used in the slot array.
    pub slot_count: u16,
}

const _: () = assert!(core::mem::size_of::<PageHeader>() == 32);

/// Bytes occupied by [`PageHeader`] at the start of every page.
pub const HEADER_LEN: usize = 32;
/// Size of one slot-array entry.
pub const SLOT_LEN: usize = 4;

// Checksum covers everything except its own 4-byte field at [24..28).
const CK_PREFIX_END: usize = 24;
const CK_FIELD_END: usize = 28;

fn structural() -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "page structure invalid",
    ))
}

/// One slot-array entry: the location and length of a tuple within the page heap.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Slot {
    /// Byte offset of the tuple within the page.
    pub offset: u16,
    /// Length of the tuple in bytes.
    pub len: u16,
}

const _: () = assert!(core::mem::size_of::<Slot>() == SLOT_LEN);

/// A complete 8 KiB page as a flat byte array.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct Page(pub [u8; PAGE_SIZE]);

const _: () = assert!(core::mem::size_of::<Page>() == PAGE_SIZE);

impl std::fmt::Debug for Page {
    // A page is 8 KiB of bytes; dumping them is useless noise. Show the type and size.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("len", &PAGE_SIZE)
            .finish_non_exhaustive()
    }
}

impl Default for Page {
    fn default() -> Self {
        Self([0; PAGE_SIZE])
    }
}

impl Page {
    /// Construct a zero-filled page. Used as a starting buffer; the caller must fill
    /// in a valid header before persisting.
    #[must_use]
    pub fn zeroed() -> Self {
        Self::default()
    }

    /// Construct an empty, initialized page for `page_id` with a valid header + checksum.
    #[must_use]
    pub fn init(page_id: PageId) -> Self {
        let mut page = Self::zeroed();
        {
            let h = page.header_mut();
            h.magic = PAGE_MAGIC;
            h.version = PAGE_VERSION;
            h.page_id = page_id;
            h.lsn = Lsn(0);
            h.checksum = 0;
            // Heap grows downward from the end; the slot array grows upward from HEADER_LEN.
            h.free_space_offset = PAGE_SIZE as u16;
            h.slot_count = 0;
        }
        page.update_checksum();
        page
    }

    /// Read-only view of the page header.
    #[must_use]
    pub fn header(&self) -> &PageHeader {
        bytemuck::from_bytes(&self.0[..HEADER_LEN])
    }

    fn header_mut(&mut self) -> &mut PageHeader {
        bytemuck::from_bytes_mut(&mut self.0[..HEADER_LEN])
    }

    /// Number of tuples currently stored.
    #[must_use]
    pub fn slot_count(&self) -> u16 {
        self.header().slot_count
    }

    /// Bytes available for a new tuple, excluding the slot entry it would consume.
    #[must_use]
    pub fn free_space(&self) -> usize {
        let h = self.header();
        let slot_array_end = HEADER_LEN + h.slot_count as usize * SLOT_LEN;
        (h.free_space_offset as usize).saturating_sub(slot_array_end)
    }

    /// CRC32 over the page, excluding the 4-byte checksum field at `[24..28)`.
    #[must_use]
    pub fn compute_checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&self.0[..CK_PREFIX_END]);
        hasher.update(&self.0[CK_FIELD_END..]);
        hasher.finalize()
    }

    /// Recompute and store the checksum. Call after any mutation, before persisting.
    pub fn update_checksum(&mut self) {
        let checksum = self.compute_checksum();
        self.header_mut().checksum = checksum;
    }

    /// Seal the page for persistence: recompute the checksum so the contents verify.
    ///
    /// Mutation methods like [`insert`](Self::insert) deliberately do **not** checksum each call —
    /// the CRC32 covers ~8 KiB, so re-running it per tuple makes a bulk load `O(n · PAGE_SIZE)`.
    /// Instead the page is treated as a dirty buffer and sealed **once** before it is written
    /// or [`verify`](Self::verify)-ed. This is an alias for [`update_checksum`](Self::update_checksum)
    /// that names the persist-boundary intent.
    pub fn seal(&mut self) {
        self.update_checksum();
    }

    /// Validate the page magic and checksum against `expected` page id.
    ///
    /// # Errors
    /// [`Error::InvalidMagic`] if the header magic is wrong, or
    /// [`Error::ChecksumMismatch`] if the stored checksum does not match the contents.
    pub fn verify(&self, expected: PageId) -> Result<()> {
        let h = self.header();
        if h.magic != PAGE_MAGIC {
            return Err(Error::InvalidMagic { page_id: expected });
        }
        let actual = self.compute_checksum();
        if actual != h.checksum {
            return Err(Error::ChecksumMismatch {
                page_id: expected,
                expected: h.checksum,
                actual,
            });
        }
        Ok(())
    }

    /// Verify a page's integrity on the read path, catching on-disk corruption before it is served
    /// A freshly-allocated page is all-zero — it has no magic or checksum until it is first
    /// written — so it is accepted as a blank frame; every other page must carry a valid magic and a
    /// stored checksum matching its contents, so a bit-flip surfaces as [`Error::ChecksumMismatch`]
    /// (or [`Error::InvalidMagic`]) instead of loading silently. Cheaper than [`try_from_bytes`](
    /// Self::try_from_bytes) — it checks only magic + checksum, not the slot structure.
    ///
    /// # Errors
    /// [`Error::InvalidMagic`] / [`Error::ChecksumMismatch`] for a non-blank page that fails to verify.
    pub fn verify_bytes(bytes: &[u8; PAGE_SIZE], expected: PageId) -> Result<()> {
        if bytes.iter().all(|&b| b == 0) {
            return Ok(());
        }
        Self(*bytes).verify(expected)
    }

    /// Fuzz-safe decoder: take arbitrary bytes and either return a fully-validated
    /// [`Page`] (safe to traverse without panicking) or an `Err`.
    ///
    /// Validates: input length, magic, version, checksum, slot-array fits below
    /// `free_space_offset`, and every slot's `(offset, len)` lies within the page
    /// and at/above `free_space_offset`.
    ///
    /// # Errors
    /// Returns [`Error::Io`] (`InvalidData`) on any structural problem, or
    /// [`Error::InvalidMagic`] / [`Error::ChecksumMismatch`] from [`verify`](Self::verify).
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != PAGE_SIZE {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "page must be exactly PAGE_SIZE bytes",
            )));
        }
        let mut data = [0u8; PAGE_SIZE];
        data.copy_from_slice(bytes);
        let page = Self(data);

        let h = page.header();
        // Magic + checksum. `verify` reports against the header's own page_id so the
        // error names a sensible id even for forged bytes.
        page.verify(h.page_id)?;
        if h.version != PAGE_VERSION {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported page version",
            )));
        }

        let slot_count = h.slot_count as usize;
        let fso = h.free_space_offset as usize;
        let slot_array_end = HEADER_LEN
            .checked_add(slot_count.checked_mul(SLOT_LEN).ok_or_else(structural)?)
            .ok_or_else(structural)?;
        if fso > PAGE_SIZE || fso < slot_array_end {
            return Err(structural());
        }

        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(slot_count);
        for i in 0..slot_count {
            let slot_pos = HEADER_LEN + i * SLOT_LEN;
            let slot: &Slot = bytemuck::from_bytes(&page.0[slot_pos..slot_pos + SLOT_LEN]);
            let off = slot.offset as usize;
            let end = off.checked_add(slot.len as usize).ok_or_else(structural)?;
            if off < fso || end > PAGE_SIZE {
                return Err(structural());
            }
            if end > off {
                ranges.push((off, end));
            }
        }
        // Reject overlapping slots: the per-slot bound check above accepts two slots whose
        // byte ranges intersect, which would alias the same tuple bytes (a forged page returning
        // two logical tuples over one region). Slots are all live here (in-page delete does not
        // exist; deletion is an MVCC tombstone above), so any overlap is structural corruption.
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(structural());
        }
        Ok(page)
    }

    /// Append `data` as a new tuple. Returns its [`SlotIdx`], or `None` if the page is full.
    ///
    /// Does **not** recompute the checksum: the page is a dirty buffer until persisted. Call
    /// [`seal`](Self::seal) once after a batch of inserts (and before [`verify`](Self::verify) or
    /// writing the page out) — sealing per insert would re-hash the whole page every tuple, making
    /// a bulk load `O(n · PAGE_SIZE)`.
    pub fn insert(&mut self, data: &[u8]) -> Option<SlotIdx> {
        if data.len() > u16::MAX as usize || self.free_space() < data.len() + SLOT_LEN {
            return None;
        }
        let header = *self.header();
        let tuple_offset = header.free_space_offset as usize - data.len();
        self.0[tuple_offset..tuple_offset + data.len()].copy_from_slice(data);

        let slot = Slot {
            offset: tuple_offset as u16,
            len: data.len() as u16,
        };
        let slot_pos = HEADER_LEN + header.slot_count as usize * SLOT_LEN;
        self.0[slot_pos..slot_pos + SLOT_LEN].copy_from_slice(bytemuck::bytes_of(&slot));

        let idx = SlotIdx(header.slot_count);
        {
            let h = self.header_mut();
            h.free_space_offset = tuple_offset as u16;
            h.slot_count += 1;
        }
        Some(idx)
    }

    /// Borrow the tuple stored at `idx`, or `None` if the slot does not exist.
    #[must_use]
    pub fn read(&self, idx: SlotIdx) -> Option<&[u8]> {
        if idx.0 >= self.header().slot_count {
            return None;
        }
        let slot_pos = HEADER_LEN + idx.0 as usize * SLOT_LEN;
        let slot: &Slot = bytemuck::from_bytes(&self.0[slot_pos..slot_pos + SLOT_LEN]);
        let (offset, len) = (slot.offset as usize, slot.len as usize);
        Some(&self.0[offset..offset + len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_produces_valid_header() {
        let page = Page::init(PageId(7));
        assert_eq!(page.header().magic, PAGE_MAGIC);
        assert_eq!(page.header().version, PAGE_VERSION);
        assert_eq!(page.header().page_id, PageId(7));
        assert_eq!(page.slot_count(), 0);
        page.verify(PageId(7)).unwrap();
    }

    #[test]
    fn insert_then_read_roundtrips() {
        let mut page = Page::init(PageId(1));
        let a = page.insert(b"hello").unwrap();
        let b = page.insert(b"world!!").unwrap();
        assert_eq!(page.slot_count(), 2);
        assert_eq!(page.read(a).unwrap(), b"hello");
        assert_eq!(page.read(b).unwrap(), b"world!!");
        page.seal(); // insert no longer checksums; seal before verify
        page.verify(PageId(1)).unwrap();
    }

    #[test]
    fn read_unknown_slot_is_none() {
        let page = Page::init(PageId(1));
        assert!(page.read(SlotIdx(0)).is_none());
    }

    #[test]
    fn insert_oversized_tuple_fails() {
        let mut page = Page::init(PageId(1));
        assert!(page.insert(&[0u8; PAGE_SIZE]).is_none());
    }

    #[test]
    fn data_corruption_is_detected() {
        let mut page = Page::init(PageId(3));
        page.insert(b"data").unwrap();
        page.seal(); // valid checksum first…
        page.0[PAGE_SIZE - 1] ^= 0xFF; // …then flip a covered byte without fixing it
        assert!(matches!(
            page.verify(PageId(3)),
            Err(Error::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn bad_magic_is_detected() {
        let mut page = Page::init(PageId(1));
        page.0[0] ^= 0xFF;
        page.update_checksum(); // only the magic is wrong now
        assert!(matches!(
            page.verify(PageId(1)),
            Err(Error::InvalidMagic { .. })
        ));
    }

    #[test]
    fn verify_bytes_allows_blank_and_catches_corruption() {
        // A freshly-allocated, never-written page is all-zero and must be accepted as a blank frame
        // (it carries no checksum yet).
        let blank = [0u8; PAGE_SIZE];
        assert!(Page::verify_bytes(&blank, PageId(5)).is_ok());

        // A sealed page verifies; a single covered-byte flip after sealing is caught as a checksum
        // mismatch instead of being served silently.
        let mut page = Page::init(PageId(5));
        page.insert(b"row").unwrap();
        page.seal();
        assert!(Page::verify_bytes(&page.0, PageId(5)).is_ok());
        let mut corrupt = page.0;
        corrupt[PAGE_SIZE - 1] ^= 0xFF;
        assert!(matches!(
            Page::verify_bytes(&corrupt, PageId(5)),
            Err(Error::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn try_from_bytes_rejects_overlapping_slots() {
        // G26: a forged page whose two slots cover the same bytes would alias one tuple as two.
        // The per-slot bound check passes (both lie within [fso, PAGE_SIZE)); the disjointness
        // check must reject it. A genuine two-tuple page is still accepted.
        let mut good = Page::init(PageId(3));
        good.insert(b"hello").unwrap();
        good.insert(b"world!!").unwrap();
        good.seal();
        Page::try_from_bytes(&good.0).expect("a valid two-slot page round-trips");

        let mut page = Page::init(PageId(3));
        page.insert(b"hello world").unwrap();
        // Forge slot 1 to alias slot 0's exact byte range.
        let slot0: Slot = *bytemuck::from_bytes(&page.0[HEADER_LEN..HEADER_LEN + SLOT_LEN]);
        let slot_pos = HEADER_LEN + SLOT_LEN;
        page.0[slot_pos..slot_pos + SLOT_LEN].copy_from_slice(bytemuck::bytes_of(&slot0));
        page.header_mut().slot_count = 2;
        page.update_checksum(); // checksum is valid; only the structure is corrupt
        assert!(
            Page::try_from_bytes(&page.0).is_err(),
            "overlapping slots must be rejected"
        );
    }

    #[test]
    fn try_from_bytes_accepts_a_valid_page() {
        let mut page = Page::init(PageId(9));
        page.insert(b"hello").unwrap();
        page.seal();
        let decoded = Page::try_from_bytes(&page.0).unwrap();
        assert_eq!(decoded.slot_count(), 1);
        assert_eq!(decoded.read(SlotIdx(0)).unwrap(), b"hello");
    }

    #[test]
    fn try_from_bytes_rejects_wrong_length() {
        assert!(Page::try_from_bytes(&[0u8; PAGE_SIZE - 1]).is_err());
        assert!(Page::try_from_bytes(&[0u8; PAGE_SIZE + 1]).is_err());
    }

    #[test]
    fn try_from_bytes_rejects_bad_magic() {
        // Zero-filled page has magic=0 — not PAGE_MAGIC.
        assert!(matches!(
            Page::try_from_bytes(&[0u8; PAGE_SIZE]),
            Err(Error::InvalidMagic { .. })
        ));
    }

    #[test]
    fn try_from_bytes_rejects_oob_slot() {
        // Build a page whose first slot claims (offset=PAGE_SIZE-2, len=8) — runs off the end.
        let mut page = Page::init(PageId(1));
        page.insert(b"ok").unwrap();
        // Overwrite the slot's len with a number that overruns PAGE_SIZE.
        let bad_slot = Slot {
            offset: (PAGE_SIZE - 2) as u16,
            len: 8,
        };
        page.0[HEADER_LEN..HEADER_LEN + SLOT_LEN].copy_from_slice(bytemuck::bytes_of(&bad_slot));
        page.update_checksum();
        assert!(Page::try_from_bytes(&page.0).is_err());
    }
}
