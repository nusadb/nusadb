//! Bulk-insert checksum cost for (defer per-insert CRC32).
//!
//! `Page::insert` no longer recomputes the page checksum on every call; the page is a dirty buffer
//! sealed once (`Page::seal`) before it is persisted/verified. Re-hashing ~8 KiB per tuple makes a
//! bulk load `O(n · PAGE_SIZE)`; this bench quantifies the difference between sealing per insert
//! (the old behaviour, reproduced here by calling `seal()` each tuple) and sealing once.
//!
//! Run: `cargo bench -p nusadb-storage --bench page_insert`.

#![allow(
    clippy::unwrap_used,
    missing_docs,
    reason = "benchmark harness, not production code (criterion_group! macro lacks docs)"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nusadb_core::PageId;
use nusadb_storage::page::Page;

/// A small tuple — the realistic shape, where the fixed ~8 KiB checksum dominates the per-tuple
/// copy and the per-insert-seal penalty is largest.
const TUPLE: &[u8] = b"the quick brown fox jumps over"; // 30 bytes

/// Fill a fresh page with as many `TUPLE`s as fit, sealing after **every** insert (old behaviour).
fn fill_seal_each() -> usize {
    let mut page = Page::init(PageId(1));
    let mut n = 0;
    while page.insert(TUPLE).is_some() {
        page.seal();
        n += 1;
    }
    n
}

/// Fill a fresh page with as many `TUPLE`s as fit, sealing **once** at the end (behaviour).
fn fill_seal_once() -> usize {
    let mut page = Page::init(PageId(1));
    let mut n = 0;
    while page.insert(TUPLE).is_some() {
        n += 1;
    }
    page.seal();
    n
}

fn bench_page_insert(c: &mut Criterion) {
    // Sanity: both paths pack the same number of tuples (the seal cadence does not affect layout).
    assert_eq!(fill_seal_each(), fill_seal_once());

    c.bench_function("page_fill_seal_each_insert", |b| {
        b.iter(|| black_box(fill_seal_each()));
    });
    c.bench_function("page_fill_seal_once", |b| {
        b.iter(|| black_box(fill_seal_once()));
    });
}

criterion_group!(benches, bench_page_insert);
criterion_main!(benches);
