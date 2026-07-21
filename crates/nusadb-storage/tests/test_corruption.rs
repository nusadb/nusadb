//! Crash-consistency: corruption detection for the 8 KiB page format.
//!
//! The page CRC32 covers the whole 8 KiB except its own 4-byte checksum field, so a
//! single flipped bit anywhere must surface as an `Err` — `InvalidMagic` when the flip
//! lands in the magic bytes, `ChecksumMismatch` everywhere else (including the checksum
//! field itself, where the stored value diverges from the recomputed one). It must never
//! be silently accepted and never panic.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test harness asserts via unwrap and corrupts pages by direct byte indexing"
)]

use nusadb_core::{Error, PAGE_SIZE, PageId, SlotIdx};
use nusadb_storage::Page;

const ID: PageId = PageId(42);

/// A structurally valid page with a few records (including an empty one).
fn valid_page() -> Page {
    let mut p = Page::init(ID);
    p.insert(b"alpha").unwrap();
    p.insert(b"bravo").unwrap();
    p.insert(b"").unwrap(); // zero-length record — still covered by the checksum
    p.seal(); // insert no longer checksums per call; seal once before verify/decode
    p
}

#[test]
fn pristine_page_verifies_and_decodes() {
    let good = valid_page();
    good.verify(ID).unwrap();
    let decoded = Page::try_from_bytes(&good.0).unwrap();
    assert_eq!(decoded.slot_count(), 3);
}

/// Exhaustively flip every one of the 8 KiB × 8 bits and assert the corruption is
/// always caught by both the in-place verifier and the on-load decoder, without panic.
#[test]
fn every_single_bit_flip_in_a_page_is_detected() {
    let good = valid_page();

    for byte in 0..PAGE_SIZE {
        for bit in 0..8u8 {
            let mut corrupt = Page(good.0); // copy the 8 KiB array
            corrupt.0[byte] ^= 1 << bit;

            assert!(
                corrupt.verify(ID).is_err(),
                "flip at byte {byte} bit {bit} slipped past verify()"
            );
            // `try_from_bytes` is the path taken when a page is loaded from disk; it must
            // reject the same corruption (and never panic while walking slots).
            assert!(
                Page::try_from_bytes(&corrupt.0).is_err(),
                "flip at byte {byte} bit {bit} slipped past try_from_bytes()"
            );
        }
    }
}

/// A flip in the magic bytes is reported specifically as `InvalidMagic`; a flip in a
/// checksum-covered region as `ChecksumMismatch`. (Spot-check the variant mapping that
/// the exhaustive sweep above only asserts as "some error".)
#[test]
fn corruption_maps_to_the_right_variant() {
    let good = valid_page();

    // Byte 0 is inside the 4-byte magic.
    let mut bad_magic = Page(good.0);
    bad_magic.0[0] ^= 0x01;
    assert!(matches!(
        bad_magic.verify(ID),
        Err(Error::InvalidMagic { .. })
    ));

    // The last byte is record/heap data — covered by the checksum, outside the magic.
    let mut bad_body = Page(good.0);
    bad_body.0[PAGE_SIZE - 1] ^= 0x01;
    assert!(matches!(
        bad_body.verify(ID),
        Err(Error::ChecksumMismatch { .. })
    ));

    // Flipping a bit inside the checksum field [24..28) makes the stored checksum
    // disagree with the (unchanged) recomputed one — also a mismatch.
    let mut bad_ck = Page(good.0);
    bad_ck.0[24] ^= 0x01;
    assert!(matches!(
        bad_ck.verify(ID),
        Err(Error::ChecksumMismatch { .. })
    ));

    // The decoder ignores empty-slot reads safely too.
    let _ = bad_body.read(SlotIdx(0));
}
