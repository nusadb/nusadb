//! Direct-I/O round-trip tests.
//!
//! These only build under the `direct-io` feature; run with:
//! `cargo test -p nusadb-storage --features direct-io --test test_direct_io`. They prove the
//! unbuffered path (aligned bounce buffers + the unbuffered open flag) is byte-for-byte correct:
//! every public `DiskManager` operation — allocate, write, read, deallocate/reuse, and reopen —
//! must behave identically to the buffered path. (The full storage suite is also expected to pass
//! under `--features direct-io`; these add focused, alignment-sensitive coverage.)

#![cfg(feature = "direct-io")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; indexing is bounded by PAGE_SIZE"
)]

use nusadb_core::{PAGE_SIZE, PageId, PageStore};
use nusadb_storage::DiskManager;

#[test]
fn write_then_read_round_trips_under_direct_io() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("dio.db")).unwrap();

    let id = dm.allocate_page().unwrap();
    let mut page = [0u8; PAGE_SIZE];
    for (i, b) in page.iter_mut().enumerate() {
        *b = (i % 251) as u8; // a non-trivial, position-dependent pattern across the whole page
    }
    dm.write_page(id, &page).unwrap();
    dm.fsync().unwrap();

    assert_eq!(
        dm.read_page(id).unwrap(),
        page,
        "direct-io read must match the write byte-for-byte"
    );
}

#[test]
fn many_pages_round_trip_and_stay_independent() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("dio.db")).unwrap();

    let mut ids = Vec::new();
    for n in 0u8..16 {
        let id = dm.allocate_page().unwrap();
        dm.write_page(id, &[n; PAGE_SIZE]).unwrap();
        ids.push((id, n));
    }
    dm.fsync().unwrap();

    for (id, n) in ids {
        assert_eq!(dm.read_page(id).unwrap(), [n; PAGE_SIZE]);
    }
}

#[test]
fn freed_pages_recycle_and_survive_reopen_under_direct_io() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("dio.db");

    let freed = {
        let dm = DiskManager::open(&path).unwrap();
        let a = dm.allocate_page().unwrap();
        let b = dm.allocate_page().unwrap();
        dm.write_page(a, &[1u8; PAGE_SIZE]).unwrap();
        dm.write_page(b, &[2u8; PAGE_SIZE]).unwrap();
        dm.deallocate_page(a).unwrap();
        dm.fsync().unwrap();
        a
    };

    // Reopen: the FREE magic stamped on the deallocated page must be rediscovered (a full-page
    // read under direct I/O), so the next allocation recycles it, zero-filled.
    let dm = DiskManager::open(&path).unwrap();
    assert_eq!(
        dm.free_count(),
        1,
        "deallocated slot must be rediscovered on reopen"
    );
    let recycled = dm.allocate_page().unwrap();
    assert_eq!(recycled, freed, "the freed slot is recycled first");
    assert_eq!(
        dm.read_page(recycled).unwrap(),
        [0u8; PAGE_SIZE],
        "recycled page is zero-filled"
    );

    // Untouched live page is intact.
    assert_eq!(dm.read_page(PageId(1)).unwrap(), [2u8; PAGE_SIZE]);
}
