//! Point-lookup hop evidence probe (ignored; run manually in release): what does the second hop
//! of a key lookup actually cost?
//!
//! Today a PK point-get is **2-hop**: an in-RAM index-entry probe (key bytes → row-id) and then
//! a clustered-tree descent (row-id → tuple, with MVCC resolution). GAP-1 (key-addressed
//! clustering, an A+B treaty change) would collapse that to one descent. Before paying for the
//! treaty change, we measured the measured overhead ("ukur dulu"). This probe reports:
//!
//! 1. the production 2-hop point lookup (`index_scan` over a single key),
//! 2. the isolated tree descent (`ClusteredTree::get`) — the floor a 1-hop design could reach,
//! 3. the range shape, where the indirection costs a descent **per row** (`index_scan` over a
//!    contiguous key range) against the leaf-chain full scan (the 1-hop range equivalent).
//!
//! `cargo test -p nusadb-btree --release --test bench_point_hops -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "manual perf probe, not a CI gate"
)]

use std::ops::Bound;
use std::time::Instant;

use nusadb_btree::BtreeEngine;
use nusadb_btree::store::MemPageStore;
use nusadb_btree::tree::ClusteredTree;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

const ROWS: u64 = 1_000_000;
const PROBES: u64 = 100_000;
const RANGE_ROWS: u64 = 10_000;

/// Deterministic LCG so the probe keys are reproducible (no `rand` dependency, no wall clock).
const fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 16
}

/// A ~24-byte opaque tuple for `key` (the engine never decodes it), the size class of a small
/// production row.
fn tuple(key: u64) -> Vec<u8> {
    let mut t = Vec::with_capacity(24);
    t.extend_from_slice(&key.to_le_bytes());
    t.extend_from_slice(&(key ^ 0x5555_5555_5555_5555).to_le_bytes());
    t.extend_from_slice(&(key.wrapping_mul(97)).to_le_bytes());
    t
}

#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn r4_point_get_hop_split() {
    // ---- production 2-hop path: engine + unique index maintained like the SQL layer does ----
    let engine = BtreeEngine::new();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            txn,
            &TableDef {
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: vec![ColumnDef {
                    name: "row".to_owned(),
                    ty: ColumnType::Bytes,
                    nullable: false,
                }],
            },
        )
        .unwrap();
    let index = engine
        .create_index(
            txn,
            &IndexDef {
                name: "t_pk".to_owned(),
                table,
                columns: vec!["row".to_owned()],
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: true,
            },
        )
        .unwrap();
    engine.commit(txn).unwrap();

    let t0 = Instant::now();
    for start in (0..ROWS).step_by(10_000) {
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        for key in start..start + 10_000 {
            let tid = engine.insert(txn, table, &tuple(key)).unwrap();
            engine
                .index_insert(txn, index, &key.to_be_bytes(), tid)
                .unwrap();
        }
        engine.commit(txn).unwrap();
    }
    println!("load {ROWS} rows + index: {:?}", t0.elapsed());

    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    // Warm the read path (first scan settles lazy per-tuple commit state).
    let mut scan = engine.scan(txn, table).unwrap();
    let mut warm = 0u64;
    while scan.try_next().unwrap().is_some() {
        warm += 1;
    }
    assert_eq!(warm, ROWS);

    // (1) 2-hop point lookups: single-key index_scan, the exact path a PK point query runs.
    let mut state = 0xA5A5_5A5A_1234_5678u64;
    let mut found = 0u64;
    let t = Instant::now();
    for _ in 0..PROBES {
        let key = (lcg(&mut state) % ROWS).to_be_bytes();
        let mut s = engine
            .index_scan(
                txn,
                index,
                Bound::Included(key.to_vec()),
                Bound::Included(key.to_vec()),
            )
            .unwrap();
        if s.try_next().unwrap().is_some() {
            found += 1;
        }
    }
    let two_hop = t.elapsed().as_nanos() as f64 / PROBES as f64;
    assert_eq!(found, PROBES);
    println!("2-hop point lookup (index_scan single key): {two_hop:.0} ns/op");

    // (3) 2-hop range: one contiguous 10k-key range — the per-row descent shape.
    let lo = (ROWS / 2).to_be_bytes().to_vec();
    let hi = (ROWS / 2 + RANGE_ROWS - 1).to_be_bytes().to_vec();
    let t = Instant::now();
    let mut s = engine
        .index_scan(txn, index, Bound::Included(lo), Bound::Included(hi))
        .unwrap();
    let mut n = 0u64;
    while s.try_next().unwrap().is_some() {
        n += 1;
    }
    let range_2hop = t.elapsed().as_nanos() as f64 / n as f64;
    assert_eq!(n, RANGE_ROWS);
    println!("2-hop range scan ({RANGE_ROWS} keys): {range_2hop:.0} ns/row");

    // Full-scan drain: the leaf-chain walk a 1-hop clustered range read would ride.
    let t = Instant::now();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut n = 0u64;
    while scan.try_next().unwrap().is_some() {
        n += 1;
    }
    let full_scan = t.elapsed().as_nanos() as f64 / n as f64;
    println!("full scan drain (leaf chain, 1-hop range equivalent): {full_scan:.0} ns/row");
    engine.commit(txn).unwrap();

    // (2) isolated descent floor: a raw clustered tree with the same 1M tuples — what a 1-hop
    // key-addressed point-get could cost (plus MVCC resolution, which both designs pay).
    let store = MemPageStore::default();
    let mut tree = ClusteredTree::create(&store).unwrap();
    for key in 0..ROWS {
        tree.insert(key, &tuple(key)).unwrap();
    }
    let mut state = 0xA5A5_5A5A_1234_5678u64;
    let mut found = 0u64;
    let t = Instant::now();
    for _ in 0..PROBES {
        let key = lcg(&mut state) % ROWS;
        if tree.get(key).unwrap().is_some() {
            found += 1;
        }
    }
    let descent = t.elapsed().as_nanos() as f64 / PROBES as f64;
    assert_eq!(found, PROBES);
    println!("isolated tree descent (1-hop floor): {descent:.0} ns/op");

    let hop1 = two_hop - descent;
    println!(
        "=> hop-1 share of a point lookup: {hop1:.0} ns ({:.0}%); \
         range indirection: {range_2hop:.0} vs {full_scan:.0} ns/row ({:.1}x)",
        hop1 / two_hop * 100.0,
        range_2hop / full_scan
    );
}
