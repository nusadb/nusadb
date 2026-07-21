//! TOAST out-of-line large-value storage: store/load round-trips across a page chain,
//! the in-tuple pointer is a compact fixed-size reference, and freeing a value reclaims its pages.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; slices indexed at constant bounds"
)]

use nusadb_core::PageStore;
use nusadb_storage::{DiskManager, Toast, ToastPointer};

fn store() -> (tempfile::TempDir, DiskManager) {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("toast.db")).unwrap();
    (dir, dm)
}

/// A deterministic pseudo-random byte pattern of length `n` (so a wrong chunk boundary is caught).
fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(31).wrapping_add(7) % 251) as u8)
        .collect()
}

#[test]
fn stores_and_loads_a_100kb_value() {
    let (_d, dm) = store();
    let toast = Toast::new(&dm);

    let value = pattern(100 * 1024); // 100 KiB → spans many 8 KiB pages
    let ptr = toast.store_value(&value).unwrap();
    dm.fsync().unwrap();

    assert_eq!(ptr.total_len, value.len() as u64);
    assert_eq!(
        toast.load(ptr).unwrap(),
        value,
        "100 KiB value must round-trip byte-for-byte"
    );
}

#[test]
fn the_pointer_is_a_compact_16_byte_in_tuple_reference() {
    let (_d, dm) = store();
    let toast = Toast::new(&dm);

    let ptr = toast.store_value(&pattern(50_000)).unwrap();
    // The pointer — not the 50 KB value — is what a tuple embeds. It survives a bytes round-trip.
    let bytes = ptr.to_bytes();
    assert_eq!(bytes.len(), 16);
    assert_eq!(ToastPointer::from_bytes(&bytes), ptr);
    // And the rehydrated pointer still loads the value.
    assert_eq!(
        toast.load(ToastPointer::from_bytes(&bytes)).unwrap().len(),
        50_000
    );
}

#[test]
fn round_trips_boundary_sizes() {
    let (_d, dm) = store();
    let toast = Toast::new(&dm);
    // Empty, sub-page, exactly one page's payload, one byte over, and several pages.
    for &len in &[0usize, 1, 8176, 8177, 16_352, 16_353, 70_000] {
        let value = pattern(len);
        let ptr = toast.store_value(&value).unwrap();
        assert_eq!(toast.load(ptr).unwrap(), value, "len {len} must round-trip");
    }
}

#[test]
fn freeing_a_value_reclaims_its_pages() {
    let (_d, dm) = store();
    let toast = Toast::new(&dm);

    let before = dm.page_count();
    let ptr = toast.store_value(&pattern(40_000)).unwrap(); // ~5 pages
    let allocated = dm.page_count() - before;
    assert!(
        allocated >= 4,
        "a 40 KB value should occupy several pages, got {allocated}"
    );

    toast.free(ptr).unwrap();
    assert_eq!(
        dm.free_count() as u64,
        allocated,
        "every page in the chain returns to the free list"
    );

    // The reclaimed pages are reused by the next large value rather than growing the file.
    let high_water = dm.page_count();
    let _ = toast.store_value(&pattern(40_000)).unwrap();
    assert_eq!(
        dm.page_count(),
        high_water,
        "freed pages are recycled, file does not grow"
    );
}

#[test]
fn a_wrong_pointer_is_rejected_not_misread() {
    let (_d, dm) = store();
    let toast = Toast::new(&dm);

    let value = pattern(20_000);
    let ptr = toast.store_value(&value).unwrap();

    // A pointer claiming the wrong length must not silently return a truncated/garbage value.
    let lied = ToastPointer {
        head: ptr.head,
        total_len: ptr.total_len + 1,
    };
    assert!(
        toast.load(lied).is_err(),
        "a length-mismatched pointer must error"
    );

    // A pointer into a non-TOAST page is rejected on the magic check.
    let id = dm.allocate_page().unwrap();
    dm.write_page(id, &[0u8; nusadb_core::PAGE_SIZE]).unwrap();
    let bogus = ToastPointer {
        head: id,
        total_len: 100,
    };
    assert!(
        toast.load(bogus).is_err(),
        "a non-TOAST head page must error"
    );
}
