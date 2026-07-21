//! OLC evidence probe (ignored; run manually in release): concurrent point-get throughput
//! plus single-writer insert cost over one shared clustered tree — the workload a per-page
//! seqlock (OLC read half) would have to beat.
//!
//! Permanent record of the 2026-07-12 decision: a full OLC read path WAS built and measured
//! against this probe on the 8-core Windows dev host — per-page seqlock slots (atomic-word
//! pages, version-validated unlatched reads), a zero-copy validated descent for `get`, and a
//! lock-free segmented slot directory (descent wrote ZERO shared cache lines) — and REJECTED:
//! reads landed inside run-to-run noise (8T: 893-937 ns/get vs baseline 897-968; 1T ~4%
//! worse), while inserts paid ~10% (10.0 vs 9.1 µs) for the scalar (non-vectorizable)
//! atomic-word page copy. On this host the random-key descent is memory-latency-bound, not
//! latch-bound, so removing reader-side synchronization bought nothing measurable. Revisit if
//! a many-core rig shows the read plateau is synchronization (re-run this probe there), or
//! together with the zero-copy scan decode (R2 stage-2), which amortizes the seqlock's copy
//! cost by not copying at all.
//!
//! `cargo test -p nusadb-btree --release --test bench_w1b_olc_readers -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    reason = "manual perf probe: asserts via unwrap, reports ns/op as f64"
)]

use std::sync::Arc;
use std::time::Instant;

use nusadb_btree::store::MemPageStore;
use nusadb_btree::tree::ClusteredTree;

const ROWS: u64 = 1_000_000;
const GETS_PER_THREAD: u64 = 1_000_000;

fn tuple(key: u64) -> Vec<u8> {
    let b = u8::try_from(key % 251).unwrap();
    vec![b; 64]
}

#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn w1b_olc_concurrent_point_get_scaling() {
    let store = Arc::new(MemPageStore::default());
    let mut tree = ClusteredTree::create(&*store).unwrap();
    let t = Instant::now();
    for key in 0..ROWS {
        tree.insert(key, &tuple(key)).unwrap();
    }
    println!(
        "setup: {ROWS} inserts in {:.2}s ({:.0} ns/insert)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_nanos() as f64 / ROWS as f64
    );
    let root = tree.root();

    for threads in [1u64, 2, 4, 8] {
        let t = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|r| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let reader = ClusteredTree::open(&*store, root);
                    let mut state = 0x243F_6A88_85A3_08D3u64.wrapping_mul(r * 2 + 1);
                    let mut hits = 0u64;
                    for _ in 0..GETS_PER_THREAD {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let key = state % ROWS;
                        if reader.get(key).unwrap().is_some() {
                            hits += 1;
                        }
                    }
                    assert_eq!(hits, GETS_PER_THREAD, "every seeded key must hit");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let total = threads * GETS_PER_THREAD;
        let secs = t.elapsed().as_secs_f64();
        let per_op = secs * 1e9 / total as f64;
        println!(
            "{threads} reader thread(s): {:.2}M gets/s total, {per_op:.0} ns/get",
            total as f64 / secs / 1e6
        );
    }
}
