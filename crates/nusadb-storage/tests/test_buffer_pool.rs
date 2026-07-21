//! Buffer-pool behavior against a counting in-memory [`PageStore`]: cache hits
//! avoid reloads, dirty pages are written back on eviction and flush, and pinned
//! pages are never evicted.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic and indexes fixed-size page arrays at constant offsets"
)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread;

use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};
use nusadb_storage::BufferPool;

/// In-memory `PageStore` that counts `read_page`/`write_page` calls so tests can
/// observe cache hits and write-backs.
#[derive(Default)]
struct MockStore {
    pages: Mutex<HashMap<u64, [u8; PAGE_SIZE]>>,
    next: AtomicU64,
    reads: AtomicUsize,
    writes: AtomicUsize,
    fail_writes: AtomicBool,
}

impl MockStore {
    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    fn writes(&self) -> usize {
        self.writes.load(Ordering::SeqCst)
    }

    fn stored(&self, id: PageId) -> Option<[u8; PAGE_SIZE]> {
        self.pages.lock().unwrap().get(&id.0).copied()
    }

    /// Seed `id`'s backing bytes with a uniform `byte` pattern without going through the cache.
    fn seed(&self, id: PageId, byte: u8) {
        self.pages.lock().unwrap().insert(id.0, [byte; PAGE_SIZE]);
    }

    /// Make every subsequent `write_page` fail, simulating a disk write error.
    fn set_fail_writes(&self, fail: bool) {
        self.fail_writes.store(fail, Ordering::SeqCst);
    }
}

impl PageStore for MockStore {
    fn read_page(&self, id: PageId) -> Result<[u8; PAGE_SIZE]> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .pages
            .lock()
            .unwrap()
            .get(&id.0)
            .copied()
            .unwrap_or([0u8; PAGE_SIZE]))
    }

    fn write_page(&self, id: PageId, page: &[u8; PAGE_SIZE]) -> Result<()> {
        if self.fail_writes.load(Ordering::SeqCst) {
            return Err(Error::Io(std::io::Error::other("injected write failure")));
        }
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.pages.lock().unwrap().insert(id.0, *page);
        Ok(())
    }

    fn allocate_page(&self) -> Result<PageId> {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        self.pages.lock().unwrap().insert(id, [0u8; PAGE_SIZE]);
        Ok(PageId(id))
    }

    fn fsync(&self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn cache_hit_avoids_reload() {
    let store = MockStore::default();
    let id = store.allocate_page().unwrap();
    let pool = BufferPool::new(store, 4);

    pool.fetch_page(id).unwrap().read(|_| {});
    assert_eq!(pool.store().reads(), 1, "first fetch should read once");

    // Second fetch is a cache hit — no additional store read.
    pool.fetch_page(id).unwrap().read(|_| {});
    assert_eq!(pool.store().reads(), 1, "cache hit must not reload");
}

#[test]
fn stats_count_hits_and_misses() {
    // The pool reports its cache hit/miss counters.
    let store = MockStore::default();
    let a = store.allocate_page().unwrap();
    let b = store.allocate_page().unwrap();
    let pool = BufferPool::new(store, 4);

    pool.fetch_page(a).unwrap().read(|_| {}); // miss (first touch of a)
    pool.fetch_page(a).unwrap().read(|_| {}); // hit
    pool.fetch_page(b).unwrap().read(|_| {}); // miss (first touch of b)
    pool.fetch_page(a).unwrap().read(|_| {}); // hit

    let s = pool.stats();
    assert_eq!(s.misses, 2, "two first-touch misses");
    assert_eq!(s.hits, 2, "two cache hits");
    assert!(
        (s.hit_ratio() - 0.5).abs() < f64::EPSILON,
        "hit ratio = 2/4 = 0.5, got {}",
        s.hit_ratio()
    );
}

#[test]
fn writes_are_visible_within_the_cache() {
    let store = MockStore::default();
    let id = store.allocate_page().unwrap();
    let pool = BufferPool::new(store, 4);

    pool.fetch_page(id).unwrap().write(|p| p[0] = 0xAB);
    // A later fetch (cache hit) sees the modified bytes.
    let got = pool.fetch_page(id).unwrap().read(|p| p[0]);
    assert_eq!(got, 0xAB);
}

#[test]
fn eviction_flushes_dirty_page() {
    let store = MockStore::default();
    let pool = BufferPool::new(store, 1); // a single frame forces eviction

    let (p0, g0) = pool.new_page().unwrap();
    g0.write(|p| p[0] = 0x42);
    drop(g0); // dirty and unpinned

    // Fetching another page evicts p0, which must be written back first.
    let (_p1, _g1) = pool.new_page().unwrap();
    assert_eq!(pool.store().stored(p0).unwrap()[0], 0x42);
}

#[test]
fn failed_writeback_keeps_the_dirty_victim_resident() {
    // If evicting a dirty page fails to write it back, those bytes are its only copy. The fetch must
    // fail, but the dirty victim must stay resident so a later flush can retry it — it must not be
    // silently dropped.
    let store = MockStore::default();
    let pool = BufferPool::new(store, 1); // single frame forces eviction

    let (p0, g0) = pool.new_page().unwrap();
    g0.write(|p| p[0] = 0x42);
    drop(g0); // dirty and unpinned

    // Eviction must flush p0, but the store rejects the write — the fetch fails.
    pool.store().set_fail_writes(true);
    assert!(
        pool.new_page().is_err(),
        "write-back failure must propagate"
    );
    assert_eq!(
        pool.store().stored(p0).unwrap()[0],
        0x00,
        "the failed write must not have reached the store (still the allocate-time zero)"
    );

    // The dirty page is still in the pool: once writes succeed again, a flush persists it intact.
    pool.store().set_fail_writes(false);
    pool.flush_all().unwrap();
    assert_eq!(
        pool.store().stored(p0).unwrap()[0],
        0x42,
        "the dirty victim's bytes must survive the earlier failure"
    );
}

#[test]
fn pinned_page_is_not_evicted() {
    let store = MockStore::default();
    let pool = BufferPool::new(store, 1);

    let (_p0, g0) = pool.new_page().unwrap();
    // Capacity is 1 and p0 is pinned, so no frame is available.
    assert!(pool.new_page().is_err());

    drop(g0);
    // After releasing the pin, the frame can be reused.
    assert!(pool.new_page().is_ok());
}

#[test]
fn flush_all_persists_dirty_pages_once() {
    let store = MockStore::default();
    let pool = BufferPool::new(store, 4);

    let (p0, g0) = pool.new_page().unwrap();
    g0.write(|p| p[1] = 0x99);
    drop(g0);

    pool.flush_all().unwrap();
    assert_eq!(pool.store().stored(p0).unwrap()[1], 0x99);

    // The page is now clean, so a second flush writes nothing more.
    let writes = pool.store().writes();
    pool.flush_all().unwrap();
    assert_eq!(pool.store().writes(), writes);
}

// ---------------------------------------------------------------------------
// Concurrency (per-frame latches): misses on distinct frames run their
// I/O without the meta mutex held, and same-page races resolve to one load.
// ---------------------------------------------------------------------------

#[test]
fn concurrent_first_touch_of_one_page_loads_exactly_once() {
    let store = MockStore::default();
    let id = store.allocate_page().unwrap();
    store.seed(id, 0x5A);
    let pool = BufferPool::new(store, 16);

    let before = pool.store().reads();
    let barrier = Barrier::new(16);
    thread::scope(|s| {
        for _ in 0..16 {
            s.spawn(|| {
                barrier.wait(); // maximize the race on the first touch
                let byte = pool.fetch_page(id).unwrap().read(|p| p[0]);
                assert_eq!(byte, 0x5A, "every racer must observe the loaded bytes");
            });
        }
    });
    // The reservation dedups the miss: exactly one thread reaches the store.
    assert_eq!(
        pool.store().reads() - before,
        1,
        "a same-page race must load once"
    );
}

#[test]
fn concurrent_misses_on_distinct_pages_all_load_correctly() {
    let store = MockStore::default();
    let mut ids = Vec::new();
    for n in 0u8..32 {
        let id = store.allocate_page().unwrap();
        store.seed(id, n);
        ids.push((id, n));
    }
    let pool = BufferPool::new(store, 32); // room for all → no eviction

    let barrier = Barrier::new(ids.len());
    thread::scope(|s| {
        for &(id, n) in &ids {
            let barrier = &barrier;
            let pool = &pool;
            s.spawn(move || {
                barrier.wait();
                assert_eq!(pool.fetch_page(id).unwrap().read(|p| p[0]), n);
            });
        }
    });
}

#[test]
fn concurrent_writes_to_distinct_pages_stay_isolated() {
    let store = MockStore::default();
    let mut ids = Vec::new();
    for _ in 0u8..16 {
        ids.push(store.allocate_page().unwrap());
    }
    let pool = BufferPool::new(store, 16);

    let barrier = Barrier::new(ids.len());
    thread::scope(|s| {
        for (i, &id) in ids.iter().enumerate() {
            let barrier = &barrier;
            let pool = &pool;
            s.spawn(move || {
                barrier.wait();
                pool.fetch_page(id).unwrap().write(|p| p[0] = i as u8);
            });
        }
    });
    pool.flush_all().unwrap();
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            pool.store().stored(id).unwrap()[0],
            i as u8,
            "each page keeps its own write"
        );
    }
}

#[test]
fn concurrent_fetches_under_eviction_pressure_never_corrupt() {
    let store = MockStore::default();
    let mut ids = Vec::new();
    for n in 0u16..64 {
        let id = store.allocate_page().unwrap();
        store.seed(id, (n % 251) as u8); // each page's bytes uniquely derive from its id
        ids.push(id);
    }
    // A pool far smaller than the 64-page working set forces constant eviction churn, but with
    // enough frames (> the 8 concurrent threads) that simultaneous pins never exhaust it.
    let pool = BufferPool::new(store, 16);

    let barrier = Barrier::new(8);
    thread::scope(|s| {
        for t in 0..8u64 {
            let barrier = &barrier;
            let pool = &pool;
            let ids = &ids;
            s.spawn(move || {
                barrier.wait();
                // Deterministic per-thread LCG walk over the page set.
                let mut x = t.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                for _ in 0..2_000 {
                    x = x
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let idx = (x >> 33) as usize % ids.len();
                    let id = ids[idx];
                    let want = (idx as u16 % 251) as u8;
                    let got = pool.fetch_page(id).unwrap().read(|p| p[0]);
                    assert_eq!(
                        got, want,
                        "a page must always read back its own seeded bytes"
                    );
                }
            });
        }
    });
}

#[test]
fn concurrent_readers_and_writers_on_one_page_serialize() {
    // Repeated read/write churn on a single shared page must never tear: a reader sees either the
    // old or the new whole-page value, never a mix, because the frame latch serializes access.
    let store = MockStore::default();
    let id = store.allocate_page().unwrap();
    let pool = BufferPool::new(store, 4);
    pool.fetch_page(id).unwrap().write(|p| p.fill(1)); // start fully = 1

    let barrier = Barrier::new(6);
    thread::scope(|s| {
        // One writer flips the whole page between two uniform values.
        s.spawn(|| {
            barrier.wait();
            for i in 0..5_000u32 {
                let v = if i % 2 == 0 { 2 } else { 3 };
                pool.fetch_page(id).unwrap().write(|p| p.fill(v));
            }
        });
        // Five readers assert the page is always uniform (no torn byte from a half-write).
        for _ in 0..5 {
            let barrier = &barrier;
            let pool = &pool;
            s.spawn(move || {
                barrier.wait();
                for _ in 0..5_000 {
                    let (first, uniform) = pool
                        .fetch_page(id)
                        .unwrap()
                        .read(|p| (p[0], p.iter().all(|&b| b == p[0])));
                    assert!(
                        uniform,
                        "page was torn: byte 0 = {first} but not all bytes match"
                    );
                }
            });
        }
    });
}
