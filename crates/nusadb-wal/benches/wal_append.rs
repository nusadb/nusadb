//! Append-path allocation + time: buffer reuse (per-writer scratch/frame buffers) and skip-compression
//! lz4 for tiny records.
//!
//! Not a criterion bench: it installs a counting global allocator and reports the **allocations per
//! append** (steady state, buffers already grown) and the wall time per append, for a control
//! record (tiny → stored uncompressed) and a full-page record (large → compressed).
//!
//! Previously each append allocated 3-4 times — `encode()` Vec, `compress_prepend_size` Vec, the
//! framing Vec, plus the seal Vec when encrypted. The encode/frame buffers are now reused, and
//! a tiny record skips compression entirely, so a control append settles to ~0 allocations
//! and a large append to ~1 (the compression output).
//!
//! Run: `cargo bench -p nusadb-wal --bench wal_append`.

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    missing_docs,
    reason = "measurement harness, not production code"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nusadb_core::{PageId, TxnId};
use nusadb_wal::{WalRecord, WalWriter};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator and only bumps two atomics around it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Append `record` `iters` times into a single writer (buffers reused across calls), returning
/// (allocations, bytes, total nanoseconds) measured over the loop only — the writer and its buffers
/// are warmed up first so steady-state per-append allocation is what is reported.
fn measure(record: &WalRecord, iters: usize) -> (usize, usize, u128) {
    // A Vec sink reused across the run; warm it and the writer's internal buffers first.
    let mut writer = WalWriter::new(Vec::with_capacity(iters * 64));
    for _ in 0..1000 {
        writer.append(record).unwrap();
    }
    writer.into_inner(); // drop the warmed sink
    let mut writer = WalWriter::new(Vec::with_capacity(iters * 64));
    writer.append(record).unwrap(); // grow the writer's scratch/frame/body once

    let a0 = ALLOCS.load(Ordering::Relaxed);
    let b0 = BYTES.load(Ordering::Relaxed);
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(writer.append(black_box(record)).unwrap());
    }
    let nanos = t0.elapsed().as_nanos();
    let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
    let bytes = BYTES.load(Ordering::Relaxed) - b0;
    black_box(writer.into_inner());
    (allocs, bytes, nanos)
}

fn main() {
    let iters = 100_000usize;

    let control = WalRecord::CommitTxn { txn: TxnId(42) }; // 9 bytes → stored
    let large = WalRecord::FullPageWrite {
        txn: TxnId(42),
        page: PageId(7),
        image: vec![0xAB; 8192],
    }; // → compressed

    for (name, rec) in [
        ("control (9 B, stored)", &control),
        ("full page (8 KiB, lz4)", &large),
    ] {
        let (allocs, bytes, nanos) = measure(rec, iters);
        println!(
            "{name:<24}: {:.3} allocs/append, {:.1} bytes/append, {:.1} ns/append",
            allocs as f64 / iters as f64,
            bytes as f64 / iters as f64,
            nanos as f64 / iters as f64,
        );
    }
}
