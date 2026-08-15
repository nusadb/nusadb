//! HNSW graph **build time** vs corpus size, and search latency over a built graph — the perf
//! evidence gate for any change to the vector-index construction path.
//!
//! Build cost is the vector index's known weak spot: construction is markedly slower than a mature
//! reference implementation and grows faster than linearly, so *any* build-path change must be shown
//! to lower build time without lowering recall. This bench measures the **time** half of that gate at
//! several corpus sizes, so the per-element cost and its growth exponent are visible side by side; the
//! **recall** half is already gated by the `recall_beats_threshold_for_every_metric` and
//! `each_metric_ranks_the_correct_vector_nearest` tests in `hnsw.rs`, which must be re-run on any
//! build-path change. Together they are the before/after evidence a construction change needs.
//!
//! Vectors are 128-dimensional (a realistic embedding width — a blob past the 8 KB leaf cap, unlike a
//! `VECTOR(3)` toy) and drawn from a `splitmix64` stream, whose distribution makes accidental
//! duplicate points vanishingly unlikely (a periodic generator once injected duplicate points and
//! misled a whole diagnosis — this avoids that class).
//!
//! Run: `cargo bench -p nusadb-sql --bench hnsw_build`.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "criterion_group! generates an undocumented `benches` fn; a bench harness panics on \
              setup failure; corpus sizing casts are exact at these scales"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nusadb_sql::hnsw::{HnswIndex, HnswParams, Metric};

const DIM: usize = 128;
/// Corpus sizes, in nodes. A geometric sweep so the build-time growth exponent is readable from the
/// ratios (each step doubles). Kept modest so the bench is runnable in CI wall-clock; the growth
/// trend, not an absolute large-N number, is what a construction change is judged against.
const SCALES: &[usize] = &[500, 1000, 2000];

/// One deterministic step of a `splitmix64` stream — a high-quality 64-bit sequence with no external
/// dependency, so the corpus is identical run to run (reproducible before/after comparisons).
const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A `[0, 1)` f32 from the stream (top 24 bits over 2^24 → an exactly representable mantissa).
fn next_f32(state: &mut u64) -> f32 {
    (splitmix64(state) >> 40) as f32 / 16_777_216.0
}

/// `n` deterministic `DIM`-dimensional vectors from a stream seeded by `seed`.
fn corpus(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| (0..DIM).map(|_| next_f32(&mut state)).collect())
        .collect()
}

/// Build a fresh index over `vectors` (the measured construction path).
fn build(vectors: &[Vec<f32>], metric: Metric) -> HnswIndex {
    let mut index = HnswIndex::new(DIM, metric, HnswParams::default(), 0x5EED);
    for v in vectors {
        index.insert(v.clone()).unwrap();
    }
    index
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    // Construction is slow (that is the point of the bench), so keep the sample count low and give
    // the slowest scale room to complete without criterion warning about too-few samples.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    for &n in SCALES {
        let vectors = corpus(n, 0x00C0_FFEE);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &vectors, |b, vectors| {
            b.iter(|| black_box(build(black_box(vectors), Metric::L2)));
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    // Search latency over an already-built graph — the read side is fast, so this uses the default
    // sampling. Built once outside the timing loop; only the query is measured.
    let vectors = corpus(2000, 0x00C0_FFEE);
    let index = build(&vectors, Metric::L2);
    let query = corpus(1, 0xD00D).pop().unwrap();
    let mut group = c.benchmark_group("hnsw_search");
    group.throughput(Throughput::Elements(1));
    group.bench_function("k10_ef100_n2000", |b| {
        b.iter(|| black_box(index.search(black_box(&query), 10, 100).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, bench_build, bench_search);
criterion_main!(benches);
