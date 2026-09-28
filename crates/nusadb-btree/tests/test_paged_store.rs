//! The page store behind a physical checkpoint image: pages load from the image on first use,
//! clean pages leave the cache under a capacity, dirty pages stay until a checkpoint publishes
//! an image that holds them, and the free list and row ids survive the round trip.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, Error, IsolationLevel, PAGE_SIZE, StorageEngine, TableId};

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
    engine.commit(txn).unwrap();
    out.sort();
    out
}

fn expected(from: u64, count: u64, len: usize) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = (from..from + count).map(|i| payload(i, len)).collect();
    out.sort();
    out
}

/// A directory with a table of `count` rows checkpointed into a physical image.
fn checkpointed(count: u64, len: usize) -> (tempfile::TempDir, TableId) {
    let dir = tempfile::tempdir().unwrap();
    let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    let table = create(&engine, "t");
    insert_rows(&engine, table, 0, count, len);
    engine.checkpoint().unwrap();
    (dir, table)
}

#[test]
fn pages_load_from_the_image_on_first_use_not_at_open() {
    let (dir, table) = checkpointed(4000, 500);
    let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    // Nothing has been read yet: the store holds none of the table's pages.
    let at_open = engine.resident_bytes().unwrap();
    assert!(
        at_open < 16 * PAGE_SIZE as u64,
        "resident after open: {at_open} bytes"
    );
    assert_eq!(rows(&engine, table), expected(0, 4000, 500));
    let after_scan = engine.resident_bytes().unwrap();
    assert!(
        after_scan > 4000 * 500 / 2,
        "resident after a full scan: {after_scan} bytes"
    );
    // The image is the page section: a second open reads the same rows again.
    drop(engine);
    let again = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    assert_eq!(rows(&again, table), expected(0, 4000, 500));
}

#[test]
fn clean_pages_leave_the_cache_under_a_capacity() {
    let (dir, table) = checkpointed(4000, 500);
    let capacity = 64 * PAGE_SIZE as u64;
    let engine = BtreeEngine::open(dir.path().join("btree.wal"))
        .unwrap()
        .with_max_total_resident_bytes(Some(capacity));
    // A full scan touches every page of a table far larger than the cache, and completes.
    assert_eq!(rows(&engine, table), expected(0, 4000, 500));
    let resident = engine.resident_bytes().unwrap();
    assert!(
        resident <= capacity,
        "resident {resident} bytes exceeds the capacity {capacity}"
    );
    // Point reads keep working across evictions.
    for _ in 0..3 {
        assert_eq!(rows(&engine, table), expected(0, 4000, 500));
    }
}

#[test]
fn dirty_pages_stay_until_a_checkpoint_and_the_cache_refuses_to_grow_past_them() {
    let (dir, table) = checkpointed(200, 500);
    let capacity = 48 * PAGE_SIZE as u64;
    let engine = BtreeEngine::open(dir.path().join("btree.wal"))
        .unwrap()
        .without_page_spill()
        .with_max_total_resident_bytes(Some(capacity));
    // With spill off, write far more than the cache can hold dirty: the growth is refused.
    let mut refused = false;
    for batch in 0..40 {
        let txn = engine.begin(RC).unwrap();
        let mut failed = false;
        for i in 0..50 {
            match engine.insert(txn, table, &payload(10_000 + batch * 50 + i, 500)) {
                Ok(_) => {},
                Err(Error::OutOfMemory(_)) => {
                    failed = true;
                    break;
                },
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
        if failed {
            engine.rollback(txn).unwrap();
            refused = true;
            break;
        }
        engine.commit(txn).unwrap();
    }
    assert!(refused, "a cache full of dirty pages must refuse to grow");
    let before = rows(&engine, table);
    // A checkpoint publishes an image holding every dirty page; they are clean again and the
    // cache can grow.
    engine.checkpoint().unwrap();
    insert_rows(&engine, table, 50_000, 100, 500);
    let mut after = before;
    after.extend(expected(50_000, 100, 500));
    after.sort();
    assert_eq!(rows(&engine, table), after);
    drop(engine);
    let reopened = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    assert_eq!(rows(&reopened, table), after);
}

#[test]
fn the_log_after_the_image_applies_onto_lazily_loaded_pages() {
    let (dir, table) = checkpointed(3000, 400);
    {
        let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
        insert_rows(&engine, table, 3000, 500, 400);
        // Delete a slice of the checkpointed rows: the delete touches pages loaded from the
        // image and goes to the log, not to a new image.
        let txn = engine.begin(RC).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut victims = Vec::new();
        while let Some((tid, tuple)) = scan.try_next().unwrap() {
            if tuple.first().is_some_and(|b| b % 4 == 0) {
                victims.push(tid);
            }
        }
        drop(scan);
        for tid in &victims {
            engine.delete(txn, table, *tid).unwrap();
        }
        engine.commit(txn).unwrap();
        assert!(!victims.is_empty());
    }
    // A plain reopen: image pages plus the log's inserts and deletes.
    let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    let mut want = expected(0, 3500, 400);
    want.retain(|row| row.first().is_none_or(|b| b % 4 != 0));
    assert_eq!(rows(&engine, table), want);
    // Row ids continue past the image's rows: a fresh insert never collides.
    insert_rows(&engine, table, 9000, 10, 400);
    assert_eq!(rows(&engine, table).len(), want.len() + 10);
}

#[test]
fn freed_pages_are_reused_after_a_restart_and_never_read_as_stale() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let keep = create(&engine, "keep");
    let gone = create(&engine, "gone");
    insert_rows(&engine, keep, 0, 500, 400);
    insert_rows(&engine, gone, 0, 2000, 400);
    let txn = engine.begin(RC).unwrap();
    engine.drop_table(txn, gone).unwrap();
    engine.commit(txn).unwrap();
    // Purge reclaims the dropped tree's pages onto the free list.
    let mut freed = 0;
    for _ in 0..20 {
        engine.purge().unwrap();
        freed = engine.free_pages().unwrap();
        if freed > 0 {
            break;
        }
    }
    assert!(
        freed > 0,
        "the dropped table's pages return to the free list"
    );
    engine.checkpoint().unwrap();
    let pages_at_checkpoint = engine.page_count();
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(
        engine.free_pages().unwrap(),
        freed,
        "the free list rode the image"
    );
    assert_eq!(rows(&engine, keep), expected(0, 500, 400));
    // New pages come from the free list first: the page count does not grow.
    insert_rows(&engine, keep, 500, 300, 400);
    assert_eq!(engine.page_count(), pages_at_checkpoint);
    assert_eq!(rows(&engine, keep), expected(0, 800, 400));
    engine.checkpoint().unwrap();
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, keep), expected(0, 800, 400));
}

/// Writers updating rows and scanners reading them, concurrently, against a cache far smaller
/// than the table: no update is lost to an eviction racing it, live and after a checkpoint and
/// a reopen.
#[test]
fn concurrent_writers_and_scanners_lose_nothing_under_eviction() {
    let (dir, table) = checkpointed(3000, 300);
    let engine = std::sync::Arc::new(
        BtreeEngine::open(dir.path().join("btree.wal"))
            .unwrap()
            .with_max_total_resident_bytes(Some(96 * PAGE_SIZE as u64)),
    );
    // Row id -> tid, from one scan.
    let tids: Vec<nusadb_core::Tid> = {
        let txn = engine.begin(RC).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut out = Vec::new();
        while let Some((tid, _)) = scan.try_next().unwrap() {
            out.push(tid);
        }
        engine.commit(txn).unwrap();
        out
    };
    let tids = std::sync::Arc::new(tids);
    let mut handles = Vec::new();
    for w in 0..4_u64 {
        let engine = std::sync::Arc::clone(&engine);
        let tids = std::sync::Arc::clone(&tids);
        handles.push(std::thread::spawn(move || {
            // Each writer owns every fourth row and rewrites it with a marked payload.
            for (i, tid) in tids
                .iter()
                .enumerate()
                .skip(usize::try_from(w).unwrap())
                .step_by(4)
            {
                let mut row = payload(100_000 + i as u64, 300);
                row[0] = 0xAB;
                loop {
                    let txn = engine.begin(RC).unwrap();
                    match engine.update(txn, table, *tid, &row) {
                        Ok(_) => {
                            engine.commit(txn).unwrap();
                            break;
                        },
                        Err(Error::OutOfMemory(_)) => {
                            engine.rollback(txn).unwrap();
                            // The cache filled with changes: fold them into an image.
                            let _ = engine.checkpoint();
                        },
                        Err(other) => panic!("update failed: {other}"),
                    }
                }
            }
        }));
    }
    for _ in 0..2 {
        let engine = std::sync::Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            for _ in 0..5 {
                assert_eq!(rows(&engine, table).len(), 3000);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let mut want: Vec<Vec<u8>> = (0..3000_u64)
        .map(|i| {
            let mut row = payload(100_000 + i, 300);
            row[0] = 0xAB;
            row
        })
        .collect();
    want.sort();
    assert_eq!(rows(&engine, table), want, "live after the writers");
    let mut attempts = 0;
    while engine.checkpoint().is_err() {
        attempts += 1;
        assert!(attempts < 50, "checkpoint never went through");
    }
    assert_eq!(rows(&engine, table), want, "live after the checkpoint");
    drop(engine);
    let reopened = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    assert_eq!(rows(&reopened, table), want, "after a reopen");
}

/// A row updated before a checkpoint carries a link to its older version in this process's
/// version arena; the image must not carry that link, or after a restart a purge would follow
/// it into another row's versions. Here: update, checkpoint, reopen, then updates and a purge
/// while a snapshot is open; the snapshot keeps its view and the latest state is right.
#[test]
fn undo_links_do_not_survive_into_the_image() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, tids) = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let table = create(&engine, "t");
        insert_rows(&engine, table, 0, 200, 100);
        let txn = engine.begin(RC).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut tids = Vec::new();
        while let Some((tid, _)) = scan.try_next().unwrap() {
            tids.push(tid);
        }
        drop(scan);
        engine.commit(txn).unwrap();
        // Every row gets an older version parked in the arena, then the image is taken
        // without a purge in between.
        let txn = engine.begin(RC).unwrap();
        for (i, tid) in tids.iter().enumerate() {
            engine
                .update(txn, table, *tid, &payload(1000 + i as u64, 100))
                .unwrap();
        }
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
        (table, tids)
    };
    let engine = BtreeEngine::open(&wal).unwrap();
    let before: Vec<Vec<u8>> = expected(1000, 200, 100);
    assert_eq!(rows(&engine, table), before);
    // A snapshot open across new updates and a purge keeps seeing the checkpointed state.
    let snapshot = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    let seen = {
        let mut scan = engine.scan(snapshot, table).unwrap();
        let mut out = Vec::new();
        while let Some((_, tuple)) = scan.try_next().unwrap() {
            out.push(tuple.to_vec());
        }
        out.sort();
        out
    };
    assert_eq!(seen, before);
    for round in 0..3_u64 {
        let txn = engine.begin(RC).unwrap();
        for (i, tid) in tids.iter().enumerate() {
            engine
                .update(
                    txn,
                    table,
                    *tid,
                    &payload(5000 + round * 1000 + i as u64, 100),
                )
                .unwrap();
        }
        engine.commit(txn).unwrap();
        engine.purge().unwrap();
    }
    let still = {
        let mut scan = engine.scan(snapshot, table).unwrap();
        let mut out = Vec::new();
        while let Some((_, tuple)) = scan.try_next().unwrap() {
            out.push(tuple.to_vec());
        }
        out.sort();
        out
    };
    assert_eq!(still, before, "the open snapshot keeps its view");
    engine.commit(snapshot).unwrap();
    engine.purge().unwrap();
    assert_eq!(rows(&engine, table), expected(7000, 200, 100));
}

/// Every page and the directory of the image are checksummed: a flipped byte in a page is
/// refused when that page is read, and a flipped byte in the directory is refused at open.
#[test]
fn a_damaged_page_or_directory_is_refused() {
    let (dir, table) = checkpointed(1000, 400);
    let wal = dir.path().join("btree.wal");
    let image = dir.path().join("btree.wal.ckpt");
    let clean = std::fs::read(&image).unwrap();
    // The first checkpoint writes every page into one segment.
    let segments: Vec<_> = std::fs::read_dir(dir.path().join("btree.wal.pages"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(segments.len(), 1, "{segments:?}");
    let segment = &segments[0];
    let pages = std::fs::read(segment).unwrap();
    // A byte inside the last page (a leaf holding rows).
    let mut damaged = pages.clone();
    let at = pages.len() - PAGE_SIZE / 2;
    damaged[at] ^= 0x40;
    std::fs::write(segment, &damaged).unwrap();
    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    // The damaged page is refused wherever it is first read: opening the scan or a later row.
    let failed = match engine.scan(txn, table) {
        Err(e) => Some(e.to_string()),
        Ok(mut scan) => loop {
            match scan.try_next() {
                Ok(Some(_)) => {},
                Ok(None) => break None,
                Err(e) => break Some(e.to_string()),
            }
        },
    };
    assert!(
        failed.as_deref().is_some_and(|e| e.contains("checksum")),
        "a damaged page must fail its checksum, got {failed:?}"
    );
    let _ = engine.rollback(txn);
    drop(engine);
    std::fs::write(segment, &pages).unwrap();
    // A byte inside the directory: past the 44-byte header and the segment table.
    let mut damaged = clean;
    let name_len = usize::from(u16::from_le_bytes([damaged[44], damaged[45]]));
    damaged[44 + 2 + name_len + 3] ^= 0x01;
    std::fs::write(&image, &damaged).unwrap();
    let err = BtreeEngine::open(&wal).unwrap_err().to_string();
    assert!(err.contains("invalid"), "{err}");
}

/// Rows that grow on update under a small cache: a split never stops half way because the page
/// cache is full. The refusal comes before an update starts, a refused update rolls back
/// cleanly, and after a checkpoint the retry succeeds; every committed row stays reachable at
/// every step, live and after a reopen, at several capacities.
#[test]
fn growing_updates_under_a_small_cache_never_lose_a_row() {
    for capacity in [36_u64, 40] {
        let (dir, table) = checkpointed(3000, 100);
        let wal = dir.path().join("btree.wal");
        let engine = BtreeEngine::open(&wal)
            .unwrap()
            .with_max_total_resident_bytes(Some(capacity * PAGE_SIZE as u64));
        let tids: Vec<nusadb_core::Tid> = {
            let txn = engine.begin(RC).unwrap();
            let mut scan = engine.scan(txn, table).unwrap();
            let mut out = Vec::new();
            while let Some((tid, _)) = scan.try_next().unwrap() {
                out.push(tid);
            }
            drop(scan);
            engine.commit(txn).unwrap();
            out
        };
        for (i, tid) in tids.iter().enumerate() {
            let row = payload(1_000_000 + i as u64, 3000);
            let mut tries = 0;
            loop {
                tries += 1;
                assert!(
                    tries < 20,
                    "capacity {capacity}: row {i} never went through"
                );
                let txn = engine.begin(RC).unwrap();
                match engine.update(txn, table, *tid, &row) {
                    Ok(_) => {
                        engine.commit(txn).unwrap();
                        break;
                    },
                    Err(Error::OutOfMemory(_)) => {
                        engine.rollback(txn).unwrap();
                        assert_eq!(rows(&engine, table).len(), 3000, "capacity {capacity}");
                        engine.checkpoint().unwrap();
                    },
                    Err(other) => panic!("capacity {capacity}: {other}"),
                }
            }
            if i % 250 == 0 {
                assert_eq!(
                    rows(&engine, table).len(),
                    3000,
                    "capacity {capacity}, row {i}"
                );
            }
        }
        let want: Vec<Vec<u8>> = {
            let mut v: Vec<Vec<u8>> = (0..3000_u64)
                .map(|i| payload(1_000_000 + i, 3000))
                .collect();
            v.sort();
            v
        };
        assert_eq!(rows(&engine, table), want, "capacity {capacity}: live");
        engine.checkpoint().unwrap();
        drop(engine);
        let reopened = BtreeEngine::open(&wal).unwrap();
        assert_eq!(
            rows(&reopened, table),
            want,
            "capacity {capacity}: reopened"
        );
    }
}

/// A purge after a mass delete removes every dead row, but it stops to let a checkpoint run once
/// the pages it changed fill half the cache, instead of dirtying the whole table in one pass.
#[test]
fn purge_yields_to_a_checkpoint_under_page_cache_pressure() {
    let (dir, table) = checkpointed(40_000, 200);
    let capacity = 64 * PAGE_SIZE as u64;
    let engine = BtreeEngine::open(dir.path().join("btree.wal"))
        .unwrap()
        .without_page_spill()
        .with_max_total_resident_bytes(Some(capacity));
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut tids = Vec::new();
    while let Some((tid, _)) = scan.try_next().unwrap() {
        tids.push(tid);
    }
    drop(scan);
    for tid in &tids {
        engine.delete(txn, table, *tid).unwrap();
    }
    engine.commit(txn).unwrap();
    engine.checkpoint().unwrap();
    assert_eq!(engine.dirty_page_bytes(), 0);
    // One purge batch can dirty at most the pages its rows sit on; the pass stops once half the
    // cache is dirty, so the overshoot is bounded by one batch, not by the table. (4096 tracks
    // the engine's purge batch size.)
    let bound = capacity / 2 + 4096 * 200 * 2;
    let mut passes = 0;
    loop {
        let stats = engine.purge().unwrap();
        let dirty = engine.dirty_page_bytes();
        assert!(
            dirty <= bound,
            "a purge pass dirtied {dirty} bytes against a cache of {capacity}"
        );
        // Without a checkpoint in between (one held off by a long transaction, say), further
        // passes change nothing more while the cache is under pressure.
        if engine.page_cache_needs_checkpoint() {
            engine.purge().unwrap();
            engine.purge().unwrap();
            assert_eq!(
                engine.dirty_page_bytes(),
                dirty,
                "a pass under pressure dirtied more"
            );
        }
        engine.checkpoint().unwrap();
        passes += 1;
        if stats.rows_removed == 0 {
            break;
        }
        assert!(passes < 200, "purge never finished");
    }
    assert!(passes > 2, "the purge must have yielded at least once");
    assert!(rows(&engine, table).is_empty());
}
