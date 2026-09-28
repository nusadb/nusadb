//! Changed pages leave memory for a scratch file when the page cache is full of them, so the
//! pages written between checkpoints are bounded by disk, not by memory. The scratch file is
//! never read by recovery: the log and the image stay the only durable copies.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, PAGE_SIZE, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def(name: &str) -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: name.to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

/// Bytes that do not repeat, so a row costs what it says.
fn payload(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u8::try_from(state & 0xff).unwrap()
        })
        .collect()
}

fn create(engine: &BtreeEngine, name: &str) -> TableId {
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def(name)).unwrap();
    engine.commit(txn).unwrap();
    table
}

fn insert_rows(engine: &BtreeEngine, table: TableId, from: u64, count: u64, len: usize) {
    let txn = engine.begin(RC).unwrap();
    for i in from..from + count {
        engine.insert(txn, table, &payload(i, len)).unwrap();
    }
    engine.commit(txn).unwrap();
}

fn rows(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out.sort();
    out
}

fn expected(from: u64, count: u64, len: usize) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = (from..from + count).map(|i| payload(i, len)).collect();
    out.sort();
    out
}

const CAPACITY_PAGES: u64 = 48;

fn capped(wal: &std::path::Path) -> BtreeEngine {
    BtreeEngine::open(wal)
        .unwrap()
        .with_max_total_resident_bytes(Some(CAPACITY_PAGES * PAGE_SIZE as u64))
}

/// Writing many times the cache between checkpoints succeeds: changed pages spill instead of
/// being refused, the resident set stays near the cache, and every row reads back before and
/// after a checkpoint and a restart.
#[test]
fn writes_far_beyond_the_cache_spill_instead_of_being_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let engine = capped(&wal);
    let table = create(&engine, "t");
    // About 16 times the cache in row bytes, all written before any checkpoint.
    for batch in 0..20 {
        insert_rows(&engine, table, batch * 1000, 1000, 300);
    }
    assert!(engine.spilled_page_bytes() > 0, "nothing was spilled");
    assert!(
        engine.dirty_page_bytes() <= (CAPACITY_PAGES + 8) * PAGE_SIZE as u64,
        "{} dirty bytes resident against a cache of {CAPACITY_PAGES} pages",
        engine.dirty_page_bytes()
    );
    let want = expected(0, 20_000, 300);
    assert_eq!(rows(&engine, table), want);
    engine.checkpoint().unwrap();
    assert_eq!(
        engine.spilled_page_bytes(),
        0,
        "a checkpoint absorbs the spill"
    );
    assert_eq!(rows(&engine, table), want);
    drop(engine);
    let reopened = capped(&wal);
    assert_eq!(rows(&reopened, table), want);
}

/// A crash with pages still spilled loses nothing committed: recovery replays the log over the
/// last image and never reads the scratch file.
#[test]
fn a_restart_with_pages_spilled_recovers_from_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let table;
    {
        let engine = capped(&wal);
        table = create(&engine, "t");
        insert_rows(&engine, table, 0, 2000, 300);
        engine.checkpoint().unwrap();
        for batch in 1..12 {
            insert_rows(&engine, table, batch * 2000, 2000, 300);
        }
        assert!(engine.spilled_page_bytes() > 0);
    }
    let reopened = capped(&wal);
    assert_eq!(rows(&reopened, table), expected(0, 24_000, 300));
}

/// Rewrites of rows whose pages keep spilling and coming back: the newest copy of a page always
/// wins, under concurrent writers and scanners.
#[test]
fn concurrent_rewrites_under_spill_lose_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let engine = Arc::new(capped(&wal));
    let tables: Vec<TableId> = (0..4).map(|i| create(&engine, &format!("t{i}"))).collect();
    for (i, table) in tables.iter().enumerate() {
        insert_rows(&engine, *table, i as u64 * 100_000, 3000, 200);
    }
    let writers: Vec<_> = tables
        .iter()
        .enumerate()
        .map(|(i, &table)| {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                // Rewrite every row of the table with a longer payload, twice.
                for round in 1..=2_u64 {
                    let txn = engine.begin(RC).unwrap();
                    let mut scan = engine.scan(txn, table).unwrap();
                    let mut tids = Vec::new();
                    while let Some((tid, _)) = scan.try_next().unwrap() {
                        tids.push(tid);
                    }
                    drop(scan);
                    for (n, tid) in tids.iter().enumerate() {
                        let seed = i as u64 * 100_000 + round * 10_000 + n as u64;
                        engine
                            .update(txn, table, *tid, &payload(seed, 260))
                            .unwrap();
                    }
                    engine.commit(txn).unwrap();
                }
            })
        })
        .collect();
    let reader = {
        let engine = Arc::clone(&engine);
        let tables = tables.clone();
        std::thread::spawn(move || {
            for _ in 0..10 {
                for &table in &tables {
                    assert_eq!(rows(&engine, table).len(), 3000);
                }
            }
        })
    };
    for w in writers {
        w.join().unwrap();
    }
    reader.join().unwrap();
    let want: Vec<Vec<Vec<u8>>> = (0..4_u64)
        .map(|i| {
            let mut v: Vec<Vec<u8>> = (0..3000)
                .map(|n| payload(i * 100_000 + 20_000 + n, 260))
                .collect();
            v.sort();
            v
        })
        .collect();
    assert!(
        engine.spilled_page_bytes() > 0,
        "the rewrites never spilled"
    );
    for (i, &table) in tables.iter().enumerate() {
        assert_eq!(rows(&engine, table), want[i]);
    }
    engine.checkpoint().unwrap();
    drop(engine);
    let reopened = capped(&wal);
    for (i, &table) in tables.iter().enumerate() {
        assert_eq!(rows(&reopened, table), want[i]);
    }
}
