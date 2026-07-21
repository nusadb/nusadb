//! Tests for `disk` (`src/disk.rs`) — the file-backed `PageStore`. These create real files, so
//! they live in `tests/`; they exercise only the public `DiskManager` / `Page` / `PageStore` API.

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_core::{PageId, PageStore, SlotIdx};
use nusadb_storage::{DiskManager, Page};

#[test]
fn write_then_read_roundtrips() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("t.db")).unwrap();
    let id = dm.allocate_page().unwrap();
    let page = Page::init(id);
    dm.write_page(id, &page.0).unwrap();
    dm.fsync().unwrap();
    let back = dm.read_page(id).unwrap();
    assert_eq!(back, page.0);
    Page(back).verify(id).unwrap();
}

#[test]
fn allocate_returns_dense_ids() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("a.db")).unwrap();
    assert_eq!(dm.allocate_page().unwrap(), PageId(0));
    assert_eq!(dm.allocate_page().unwrap(), PageId(1));
    assert_eq!(dm.page_count(), 2);
}

#[test]
fn reopen_resumes_allocation_and_preserves_data() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("p.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        let id = dm.allocate_page().unwrap();
        let mut page = Page::init(id);
        page.insert(b"persisted").unwrap();
        page.seal(); // insert no longer checksums per call
        dm.write_page(id, &page.0).unwrap();
        dm.fsync().unwrap();
    }
    let dm = DiskManager::open(&path).unwrap();
    assert_eq!(dm.allocate_page().unwrap(), PageId(1)); // resumes after page 0
    let page = Page(dm.read_page(PageId(0)).unwrap());
    page.verify(PageId(0)).unwrap();
    assert_eq!(page.read(SlotIdx(0)).unwrap(), b"persisted");
}

#[test]
fn deallocate_then_allocate_reuses_the_freed_slot() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("r.db")).unwrap();
    let a = dm.allocate_page().unwrap(); // PageId(0)
    let b = dm.allocate_page().unwrap(); // PageId(1)
    assert_eq!(dm.page_count(), 2);

    dm.deallocate_page(a).unwrap();
    assert_eq!(dm.free_count(), 1);

    // The next allocate must reuse `a` rather than extending the file.
    let reused = dm.allocate_page().unwrap();
    assert_eq!(reused, a);
    assert_eq!(dm.page_count(), 2); // unchanged — file did not grow
    assert_eq!(dm.free_count(), 0);

    // `b` is untouched.
    let _ = dm.read_page(b).unwrap();
}

#[test]
fn allocated_after_reuse_returns_zeroed_page() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("z.db")).unwrap();
    let id = dm.allocate_page().unwrap();
    // Put recognizable data on the page, then free it.
    let page = Page::init(id);
    dm.write_page(id, &page.0).unwrap();
    dm.deallocate_page(id).unwrap();

    // Reusing the slot must clear it — caller must see a zero page, not the old contents.
    let reused = dm.allocate_page().unwrap();
    assert_eq!(reused, id);
    let bytes = dm.read_page(reused).unwrap();
    assert!(bytes.iter().all(|&b| b == 0));
}

#[test]
fn free_list_survives_reopen() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("f.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        let _ = dm.allocate_page().unwrap(); // 0
        let one = dm.allocate_page().unwrap(); // 1
        let _ = dm.allocate_page().unwrap(); // 2
        dm.deallocate_page(one).unwrap();
        dm.fsync().unwrap();
    }
    let dm = DiskManager::open(&path).unwrap();
    // The free list was rebuilt from the on-disk FREE marker; the next allocate must
    // return the previously-freed id rather than extending.
    assert_eq!(dm.page_count(), 3);
    assert_eq!(dm.free_count(), 1);
    let reused = dm.allocate_page().unwrap();
    assert_eq!(reused, PageId(1));
}

#[test]
fn fsm_sidecar_is_written_on_fsync_and_drives_reopen() {
    // A successful fsync after a free-list change leaves an `<data>.fsm` sidecar, and the next
    // open() rebuilds the free list from it (verified against the on-disk FREE markers).
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("s.db");
    let fsm = dir.path().join("s.db.fsm");
    {
        let dm = DiskManager::open(&path).unwrap();
        let _ = dm.allocate_page().unwrap(); // 0
        let one = dm.allocate_page().unwrap(); // 1
        let _ = dm.allocate_page().unwrap(); // 2
        dm.deallocate_page(one).unwrap();
        dm.fsync().unwrap();
    }
    assert!(fsm.exists(), "fsync must write the FSM sidecar");

    let dm = DiskManager::open(&path).unwrap();
    assert_eq!(dm.free_count(), 1);
    assert_eq!(dm.allocate_page().unwrap(), PageId(1));
}

#[test]
fn reopen_falls_back_to_scan_when_fsm_is_missing_or_corrupt() {
    // The on-disk FREE markers stay authoritative — a missing or corrupt sidecar just forces
    // the original full scan, never a wrong or lost free list.
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("c.db");
    let fsm = dir.path().join("c.db.fsm");
    {
        let dm = DiskManager::open(&path).unwrap();
        for _ in 0..4 {
            let _ = dm.allocate_page().unwrap();
        }
        dm.deallocate_page(PageId(1)).unwrap();
        dm.deallocate_page(PageId(3)).unwrap();
        dm.fsync().unwrap();
    }

    // Missing sidecar → full scan still finds both freed pages.
    std::fs::remove_file(&fsm).unwrap();
    {
        let dm = DiskManager::open(&path).unwrap();
        assert_eq!(
            dm.free_count(),
            2,
            "scan fallback must rebuild the free list"
        );
        dm.fsync().unwrap(); // re-creates the sidecar
    }

    // Corrupt sidecar (flip a byte) → CRC fails → full scan fallback, still correct.
    let mut bytes = std::fs::read(&fsm).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&fsm, &bytes).unwrap();
    let dm = DiskManager::open(&path).unwrap();
    assert_eq!(dm.free_count(), 2, "corrupt sidecar must fall back to scan");
}

#[test]
fn stale_fsm_entry_for_a_reused_page_is_not_handed_back_as_free() {
    // Crash-safety: if the sidecar lists a page that was reused after the last checkpoint (so it
    // no longer bears the FREE marker), open()'s verification must drop it — a live/reused page is
    // never recycled. We simulate the crash by reusing the slot and dropping the manager without a
    // second fsync, so the on-disk sidecar stays stale.
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("stale.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        let _ = dm.allocate_page().unwrap(); // 0
        let one = dm.allocate_page().unwrap(); // 1
        let _ = dm.allocate_page().unwrap(); // 2
        dm.deallocate_page(one).unwrap();
        dm.fsync().unwrap(); // sidecar now lists page 1 as free
        // Reuse page 1 (zeroes the slot — clears the FREE marker) but DO NOT fsync again, so the
        // sidecar on disk is now stale (still lists 1). This is the crash-before-checkpoint window.
        assert_eq!(dm.allocate_page().unwrap(), PageId(1));
        // drop without fsync
    }

    let dm = DiskManager::open(&path).unwrap();
    assert_eq!(
        dm.free_count(),
        0,
        "verification must drop the stale entry for the reused page"
    );
    // The freshly handed-out id must extend the file, never re-hand the live page 1.
    assert_eq!(dm.allocate_page().unwrap(), PageId(3));
}

#[test]
fn lifo_recycling_returns_newest_freed_first() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("l.db")).unwrap();
    let a = dm.allocate_page().unwrap();
    let b = dm.allocate_page().unwrap();
    let c = dm.allocate_page().unwrap();
    dm.deallocate_page(a).unwrap();
    dm.deallocate_page(b).unwrap();
    dm.deallocate_page(c).unwrap();
    // LIFO — c freed last, so c comes back first.
    assert_eq!(dm.allocate_page().unwrap(), c);
    assert_eq!(dm.allocate_page().unwrap(), b);
    assert_eq!(dm.allocate_page().unwrap(), a);
}

#[test]
fn concurrent_reads_return_each_pages_own_bytes() {
    use std::sync::Arc;
    use std::thread;

    let dir = nusadb_test_utils::temp_dir();
    let dm = Arc::new(DiskManager::open(dir.path().join("conc.db")).unwrap());

    // Lay down 16 distinct pages; page tagged `n` carries [n; 32] in slot 0.
    let mut pages: Vec<(PageId, u8)> = Vec::new();
    for n in 0..16u8 {
        let id = dm.allocate_page().unwrap();
        let mut page = Page::init(id);
        page.insert(&[n; 32]).unwrap();
        page.seal(); // insert no longer checksums per call
        dm.write_page(id, &page.0).unwrap();
        pages.push((id, n));
    }
    dm.fsync().unwrap();

    // 8 threads hammer reads on the shared file. Positioned reads don't touch a shared
    // cursor, so every read must return that page's own bytes — no cross-talk.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let dm = Arc::clone(&dm);
        let pages = pages.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..250 {
                for &(id, n) in &pages {
                    let page = Page(dm.read_page(id).unwrap());
                    page.verify(id).unwrap();
                    assert_eq!(page.read(SlotIdx(0)).unwrap(), &[n; 32]);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn read_pages_into_batches_a_contiguous_run() {
    use nusadb_core::PAGE_SIZE;
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("ra.db")).unwrap();

    // Lay down 32 consecutive pages whose first byte uniquely identifies each.
    let mut ids = Vec::new();
    for n in 0u8..32 {
        let id = dm.allocate_page().unwrap();
        let mut buf = [0u8; PAGE_SIZE];
        buf[0] = n;
        dm.write_page(id, &buf).unwrap();
        ids.push(id);
    }
    dm.fsync().unwrap();

    // A single batched read of the whole run yields each page's own bytes, in order.
    let mut window = vec![0u8; ids.len() * PAGE_SIZE];
    let read = dm.read_pages_into(ids[0], &mut window).unwrap();
    assert_eq!(read, ids.len());
    for (n, chunk) in window.chunks_exact(PAGE_SIZE).enumerate() {
        assert_eq!(
            chunk[0], n as u8,
            "batched read-ahead must preserve per-page bytes and order"
        );
        assert_eq!(
            &dm.read_page(ids[n]).unwrap()[..],
            chunk,
            "batch must match per-page read"
        );
    }

    // A buffer that is not a whole number of pages (or empty) is rejected.
    let mut bad = vec![0u8; PAGE_SIZE + 1];
    assert!(dm.read_pages_into(ids[0], &mut bad).is_err());
    assert!(dm.read_pages_into(ids[0], &mut []).is_err());
}

#[test]
fn out_of_range_page_id_is_rejected_not_aliased() {
    // G1: a page id at/beyond the high-water mark was never allocated. Reading or writing it
    // must error rather than aliasing a live slot or writing a sparse hole past end-of-file.
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("g1.db")).unwrap();
    let id = dm.allocate_page().unwrap(); // PageId(0); next_page == 1
    let blank = Page::init(id);

    // In range: fine.
    dm.write_page(id, &blank.0).unwrap();
    // Exactly at the high-water mark and far beyond: rejected, not aliased to a low offset.
    assert!(dm.read_page(PageId(1)).is_err());
    assert!(dm.write_page(PageId(1), &blank.0).is_err());
    assert!(dm.read_page(PageId(u64::MAX)).is_err());
    assert!(dm.write_page(PageId(u64::MAX), &blank.0).is_err());
    // A read-ahead window that overruns allocated space is rejected as a whole.
    let mut window = vec![0u8; 4 * nusadb_core::PAGE_SIZE];
    assert!(dm.read_pages_into(id, &mut window).is_err());
}

#[test]
fn double_deallocate_is_idempotent() {
    // G2: deallocating the same id twice must not push it onto the free list twice — otherwise
    // two allocate_page calls would hand back the same physical slot (two logical pages, one slot).
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("g2.db")).unwrap();
    let a = dm.allocate_page().unwrap();
    let b = dm.allocate_page().unwrap();

    dm.deallocate_page(a).unwrap();
    dm.deallocate_page(a).unwrap(); // second free is a no-op
    assert_eq!(dm.free_count(), 1, "double-free must not double-count");

    // Only one reuse comes back; the next allocate extends the file rather than re-handing `a`.
    let reused = dm.allocate_page().unwrap();
    assert_eq!(reused, a);
    let fresh = dm.allocate_page().unwrap();
    assert_ne!(fresh, a, "freed slot must not be handed out twice");
    assert_ne!(fresh, b);
}
