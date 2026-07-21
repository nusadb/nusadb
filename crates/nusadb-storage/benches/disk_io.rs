//! Disk page-I/O benchmark for (Direct I/O).
//!
//! Compare buffered vs. unbuffered (direct) I/O by running the same workload under each build:
//!
//! ```bash
//! cargo bench -p nusadb-storage --bench disk_io                    # buffered (default)
//! cargo bench -p nusadb-storage --bench disk_io --features direct-io
//! ```
//!
//! `direct-io` is a compile-time switch on the whole crate, so a single binary can only measure
//! one mode; run both and diff the criterion output. The `sequential_write` workload writes a run
//! of pages then fsyncs (the WAL/flush shape); `random_read` reads pages back in a scattered
//! order (the buffer-pool-miss shape). Direct I/O trades raw single-threaded throughput (a second
//! copy is removed but the OS read-ahead cache is gone) for predictable cache behaviour — the
//! benchmark exists to quantify that trade-off on the target hardware.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    missing_docs,
    reason = "benchmark harness, not production code (criterion_group! macro lacks docs)"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nusadb_core::{PAGE_SIZE, PageId, PageStore};
use nusadb_storage::DiskManager;

/// Number of pages touched per benchmark iteration.
const PAGES: u64 = 512;

/// Allocate `PAGES` pages then overwrite each with a distinct payload, fsyncing once at the end.
fn sequential_write(dm: &DiskManager, page: &[u8; PAGE_SIZE]) {
    let mut ids = Vec::with_capacity(PAGES as usize);
    for _ in 0..PAGES {
        ids.push(dm.allocate_page().unwrap());
    }
    for id in &ids {
        dm.write_page(*id, page).unwrap();
    }
    dm.fsync().unwrap();
}

/// Read `PAGES` pages back in a strided (scattered) order to defeat sequential prefetch.
fn scattered_read(dm: &DiskManager, count: u64) -> u64 {
    let mut acc = 0u64;
    let mut idx = 0u64;
    for _ in 0..count {
        idx = (idx + 97) % count; // 97 is coprime with powers of two → visits every slot
        let page = dm.read_page(PageId(idx)).unwrap();
        acc = acc.wrapping_add(u64::from(page[0]));
    }
    acc
}

fn bench_disk_io(c: &mut Criterion) {
    let payload = [0xABu8; PAGE_SIZE];

    c.bench_function("sequential_write_512p", |b| {
        b.iter_batched(
            || {
                let dir = nusadb_test_utils::temp_dir();
                let dm = DiskManager::open(dir.path().join("bench.db")).unwrap();
                (dir, dm)
            },
            |(_dir, dm)| sequential_write(black_box(&dm), black_box(&payload)),
            criterion::BatchSize::SmallInput,
        );
    });

    // Read-ahead: scanning a contiguous run page-by-page vs. in batched windows. The
    // batched path issues one syscall per window instead of one per page.
    {
        const SCAN_PAGES: u64 = 4_096;
        const WINDOW: usize = 256;
        let dir = nusadb_test_utils::temp_dir();
        let dm = DiskManager::open(dir.path().join("scan.db")).unwrap();
        for _ in 0..SCAN_PAGES {
            let id = dm.allocate_page().unwrap();
            dm.write_page(id, &payload).unwrap();
        }
        dm.fsync().unwrap();

        c.bench_function("sequential_scan_per_page", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for p in 0..SCAN_PAGES {
                    let page = dm.read_page(PageId(p)).unwrap();
                    acc = acc.wrapping_add(u64::from(page[0]));
                }
                black_box(acc);
            });
        });
        c.bench_function("sequential_scan_read_ahead", |b| {
            let mut window = vec![0u8; WINDOW * PAGE_SIZE]; // one reused buffer for the whole scan
            b.iter(|| {
                let mut acc = 0u64;
                let mut start = 0u64;
                while start < SCAN_PAGES {
                    let count = WINDOW.min((SCAN_PAGES - start) as usize);
                    let buf = &mut window[..count * PAGE_SIZE];
                    dm.read_pages_into(PageId(start), buf).unwrap();
                    for page in buf.chunks_exact(PAGE_SIZE) {
                        acc = acc.wrapping_add(u64::from(page[0]));
                    }
                    start += count as u64;
                }
                black_box(acc);
            });
        });
    }

    c.bench_function("scattered_read_512p", |b| {
        // Pre-populate a file once; every iteration reads from it.
        let dir = nusadb_test_utils::temp_dir();
        let dm = DiskManager::open(dir.path().join("bench.db")).unwrap();
        for _ in 0..PAGES {
            let id = dm.allocate_page().unwrap();
            dm.write_page(id, &payload).unwrap();
        }
        dm.fsync().unwrap();
        b.iter(|| black_box(scattered_read(black_box(&dm), black_box(PAGES))));
    });

    // Open() cost on a large file — rebuilding the free list from the FSM sidecar
    // (O(free_count) reads) vs the original full file scan (O(page_count) reads). Build one big
    // file with a handful of free pages, then time open() with the sidecar present vs. removed.
    {
        const BIG_PAGES: u64 = 16_384; // 128 MiB
        let dir = nusadb_test_utils::temp_dir();
        let path = dir.path().join("open.db");
        let fsm = dir.path().join("open.db.fsm");
        {
            let dm = DiskManager::open(&path).unwrap();
            for _ in 0..BIG_PAGES {
                let id = dm.allocate_page().unwrap();
                dm.write_page(id, &payload).unwrap();
            }
            // A few scattered frees — the realistic case (free list is small vs. the file).
            for id in [3u64, 1000, 9999, 16000] {
                dm.deallocate_page(PageId(id)).unwrap();
            }
            dm.fsync().unwrap(); // writes the FSM sidecar
        }

        c.bench_function("open_16384p_with_fsm", |b| {
            b.iter(|| {
                let dm = DiskManager::open(black_box(&path)).unwrap();
                black_box(dm.free_count());
            });
        });

        std::fs::remove_file(&fsm).unwrap(); // force the full-scan fallback
        c.bench_function("open_16384p_full_scan", |b| {
            b.iter_batched(
                // Each scan-fallback open re-arms the dirty flag; a stray fsync would rewrite the
                // sidecar, so remove it before every iteration to keep measuring the scan.
                || {
                    let _ = std::fs::remove_file(&fsm);
                },
                |()| {
                    let dm = DiskManager::open(black_box(&path)).unwrap();
                    black_box(dm.free_count());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(benches, bench_disk_io);
criterion_main!(benches);
