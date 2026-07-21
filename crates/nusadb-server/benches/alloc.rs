//! Allocator hot-path benchmark for
//!
//! Run the default (system allocator) and the mimalloc build, then compare:
//!
//! ```bash
//! cargo bench -p nusadb-server --bench alloc                      # system allocator
//! cargo bench -p nusadb-server --bench alloc --features mimalloc  # mimalloc
//! ```
//!
//! The workload mimics the engine's allocation churn: many short-lived per-tuple `Vec<u8>`
//! buffers plus a `BTreeMap` insert/drop cycle (the ordered write-buffer shape). Whichever allocator the
//! binary is compiled with backs every allocation here, so the two runs isolate allocator cost.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    missing_docs,
    reason = "benchmark harness, not production code (criterion_group! macro lacks docs)"
)]

use std::collections::BTreeMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Allocate and drop many small per-tuple byte buffers (the row-encode hot path).
fn tuple_churn(n: usize) -> usize {
    let mut acc = 0usize;
    for i in 0..n {
        let len = 8 + (i % 120);
        let mut v: Vec<u8> = Vec::with_capacity(len);
        v.extend((0..len).map(|b| (b ^ i) as u8));
        acc = acc.wrapping_add(v[len - 1] as usize);
        // `v` drops here — freed immediately, exercising alloc/free pairing.
    }
    acc
}

/// Build then drop an ordered-map `BTreeMap<Vec<u8>, Vec<u8>>` (node + value allocations).
fn btree_map_churn(n: usize) -> usize {
    let mut map: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for i in 0..n {
        let key = format!("key{i:08}").into_bytes();
        let val = vec![(i & 0xff) as u8; 32 + (i % 64)];
        map.insert(key, val);
    }
    map.len()
}

fn bench_alloc(c: &mut Criterion) {
    c.bench_function("tuple_churn_50k", |b| {
        b.iter(|| black_box(tuple_churn(black_box(50_000))));
    });
    c.bench_function("btree_map_churn_50k", |b| {
        b.iter(|| black_box(btree_map_churn(black_box(50_000))));
    });
}

criterion_group!(benches, bench_alloc);
criterion_main!(benches);
