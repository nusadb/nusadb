//! Buffer-pool throughput benchmark for (per-frame latches).
//!
//! ```bash
//! cargo bench -p nusadb-storage --bench buffer_pool
//! ```
//!
//! `fetch_hit` measures the all-hit fast path (single thread). `concurrent_miss_8t` measures eight
//! threads hammering a working set larger than the pool, i.e. constant eviction + load churn — the
//! path that, before per-frame latches, serialized every miss's I/O behind one mutex. With the
//! split design the load I/O runs outside the meta mutex, so this number reflects reduced
//! contention. The backing store is in-memory so the benchmark isolates pool/lock cost rather than
//! disk latency.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    missing_docs,
    reason = "benchmark harness, not production code (criterion_group! macro lacks docs)"
)]

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use criterion::{Criterion, criterion_group, criterion_main};
use nusadb_core::{PAGE_SIZE, PageId, PageStore, Result};
use nusadb_storage::BufferPool;

/// Trivial in-memory `PageStore` so the benchmark measures pool/lock cost, not disk latency.
#[derive(Default)]
struct MemStore {
    pages: Mutex<HashMap<u64, [u8; PAGE_SIZE]>>,
    next: AtomicU64,
}

impl PageStore for MemStore {
    fn read_page(&self, id: PageId) -> Result<[u8; PAGE_SIZE]> {
        Ok(self
            .pages
            .lock()
            .unwrap()
            .get(&id.0)
            .copied()
            .unwrap_or([0u8; PAGE_SIZE]))
    }
    fn write_page(&self, id: PageId, page: &[u8; PAGE_SIZE]) -> Result<()> {
        self.pages.lock().unwrap().insert(id.0, *page);
        Ok(())
    }
    fn allocate_page(&self) -> Result<PageId> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.pages.lock().unwrap().insert(id, [0u8; PAGE_SIZE]);
        Ok(PageId(id))
    }
    fn fsync(&self) -> Result<()> {
        Ok(())
    }
}

fn make_pool(pages: u64, capacity: usize) -> (BufferPool<MemStore>, Vec<PageId>) {
    let store = MemStore::default();
    let ids: Vec<PageId> = (0..pages).map(|_| store.allocate_page().unwrap()).collect();
    (BufferPool::new(store, capacity), ids)
}

fn bench_buffer_pool(c: &mut Criterion) {
    // All-hit path: one resident page fetched repeatedly.
    c.bench_function("fetch_hit", |b| {
        let (pool, ids) = make_pool(1, 8);
        pool.fetch_page(ids[0]).unwrap().read(|_| {}); // prime the cache
        b.iter(|| {
            let v = pool.fetch_page(black_box(ids[0])).unwrap().read(|p| p[0]);
            black_box(v);
        });
    });

    // Contention path: 8 threads over a 256-page set in a 16-frame pool → constant eviction.
    c.bench_function("concurrent_miss_8t", |b| {
        let (pool, ids) = make_pool(256, 16);
        b.iter(|| {
            thread::scope(|s| {
                for t in 0..8u64 {
                    let pool = &pool;
                    let ids = &ids;
                    s.spawn(move || {
                        let mut x = t.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                        for _ in 0..1_000 {
                            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                            let id = ids[(x >> 33) as usize % ids.len()];
                            let v = pool.fetch_page(id).unwrap().read(|p| p[0]);
                            black_box(v);
                        }
                    });
                }
            });
        });
    });
}

criterion_group!(benches, bench_buffer_pool);
criterion_main!(benches);
