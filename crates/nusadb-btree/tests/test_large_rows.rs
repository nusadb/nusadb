//! Rows larger than one page: stored in overflow chains, logged and imaged logically, so they
//! survive a restart, a checkpoint, and purge, and their pages come back when nothing references
//! them.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, PAGE_SIZE, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def() -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: "t".to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

fn payload(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| u8::try_from((i * 31 + usize::from(seed)) % 256).unwrap())
        .collect()
}

fn rows(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    engine.commit(txn).unwrap();
    out
}

#[test]
fn large_rows_survive_restart_checkpoint_and_purge() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let big = payload(1, PAGE_SIZE * 4 + 123);
    let huge = payload(2, PAGE_SIZE * 40);
    // A chained value keeps its 24-byte header inline and spreads the rest over pages of 8181
    // payload bytes, so the user bytes alone decide the chain length.
    let chain_pages = |user_bytes: usize| user_bytes.div_ceil(PAGE_SIZE - 11);
    let (big_pages, huge_pages) = (chain_pages(big.len()), chain_pages(huge.len()));
    let table = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.insert(txn, table, &big).unwrap();
        engine.insert(txn, table, b"tiny").unwrap();
        engine.insert(txn, table, &huge).unwrap();
        engine.commit(txn).unwrap();
        assert_eq!(
            rows(&engine, table),
            vec![big.clone(), b"tiny".to_vec(), huge.clone()]
        );
        table
    };
    // Restart: the log replays the large rows into fresh chains.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(
        rows(&engine, table),
        vec![big.clone(), b"tiny".to_vec(), huge.clone()]
    );
    // Checkpoint: the image carries them logically; a reopen from image plus empty tail agrees.
    engine.checkpoint().unwrap();
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), vec![big, b"tiny".to_vec(), huge]);

    // Update the huge row to a small one and delete the big one: their chains are retired and
    // come back through purge once the transactions settle, with the data intact meanwhile.
    let txn = engine.begin(RC).unwrap();
    let tids: Vec<_> = {
        let mut scan = engine.scan(txn, table).unwrap();
        let mut out = Vec::new();
        while let Some((tid, _)) = scan.try_next().unwrap() {
            out.push(tid);
        }
        out
    };
    engine.update(txn, table, tids[2], b"shrunk").unwrap();
    engine.delete(txn, table, tids[0]).unwrap();
    engine.commit(txn).unwrap();
    assert_eq!(
        rows(&engine, table),
        vec![b"tiny".to_vec(), b"shrunk".to_vec()]
    );
    // The big row's chain (five pages) and the huge row's (forty-one) both come back, exactly.
    let free_before = engine.free_pages().unwrap();
    let stats = engine.purge().unwrap();
    let expected = big_pages + huge_pages;
    assert_eq!(stats.pages_reclaimed, expected, "{stats:?}");
    assert_eq!(engine.free_pages().unwrap() - free_before, expected);
    // And the reclaimed state survives another restart.
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(
        rows(&engine, table),
        vec![b"tiny".to_vec(), b"shrunk".to_vec()]
    );
}

#[test]
fn a_rolled_back_large_insert_leaves_nothing_behind() {
    let engine = BtreeEngine::new();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    let txn = engine.begin(RC).unwrap();
    engine
        .insert(txn, table, &payload(3, PAGE_SIZE * 6))
        .unwrap();
    engine.rollback(txn).unwrap();
    assert!(rows(&engine, table).is_empty());
    // The rolled-back row's chain is handed back once the abort settles, page for page.
    let free_before = engine.free_pages().unwrap();
    let stats = engine.purge().unwrap();
    let expected = (PAGE_SIZE * 6).div_ceil(PAGE_SIZE - 11);
    assert_eq!(stats.pages_reclaimed, expected, "{stats:?}");
    assert_eq!(engine.free_pages().unwrap() - free_before, expected);
}

/// Hundreds of chained rows: their stubs alone fill and split leaves, every row still reads
/// back, and dropping the table hands back every tree page and every chain page.
#[test]
fn stub_only_leaves_split_and_a_dropped_table_returns_every_chain_page() {
    let engine = BtreeEngine::new();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    let free_before = engine.free_pages().unwrap();
    let count = 400_usize;
    let row = |i: usize| payload(u8::try_from(i % 256).unwrap(), PAGE_SIZE + 500 + i);
    let txn = engine.begin(RC).unwrap();
    for i in 0..count {
        engine.insert(txn, table, &row(i)).unwrap();
    }
    engine.commit(txn).unwrap();
    // The scan yields rows in row-id order, which is insertion order here.
    let want: Vec<Vec<u8>> = (0..count).map(row).collect();
    assert_eq!(rows(&engine, table), want);

    let txn = engine.begin(RC).unwrap();
    engine.drop_table(txn, table).unwrap();
    engine.commit(txn).unwrap();
    let stats = engine.purge().unwrap();
    // Every chain page comes back, plus the handful of leaves and interior nodes the 400 stubs
    // occupied (46 bytes each, about 177 to a leaf).
    let chains: usize = (0..count)
        .map(|i| (PAGE_SIZE + 500 + i).div_ceil(PAGE_SIZE - 11))
        .sum();
    assert!(
        stats.pages_reclaimed >= chains && stats.pages_reclaimed <= chains + 16,
        "{stats:?}, chains {chains}"
    );
    assert_eq!(
        engine.free_pages().unwrap() - free_before,
        stats.pages_reclaimed
    );
}

/// Bytes that do not compress, from a small xorshift generator, so every log record carries
/// its full size.
fn incompressible(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u8::try_from(state & 0xff).unwrap()
        })
        .collect()
}

/// A batch whose rows together exceed what one log record may hold is logged in several records
/// and replays on restart; a single row of the maximum size, incompressible, replays too.
#[test]
fn a_batch_of_large_incompressible_rows_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    // Runs of two rows and one row: the first two fit one 32 MiB record together, the third
    // does not, so the row-id offsets of the second record are not trivial.
    let batch: Vec<Vec<u8>> = [16_usize, 15, 24]
        .iter()
        .enumerate()
        .map(|(i, mib)| incompressible(u64::try_from(i).unwrap() + 1, mib * 1024 * 1024))
        .collect();
    let single = incompressible(99, nusadb_btree::MAX_USER_TUPLE);
    let table = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.insert(txn, table, b"before").unwrap();
        engine.insert_batch(txn, table, &batch).unwrap();
        engine.insert(txn, table, &single).unwrap();
        engine.insert(txn, table, b"after").unwrap();
        engine.commit(txn).unwrap();
        table
    };
    let engine = BtreeEngine::open(&wal).unwrap();
    let mut want = vec![b"before".to_vec()];
    want.extend(batch);
    want.push(single);
    want.push(b"after".to_vec());
    assert_eq!(rows(&engine, table), want);
}

/// A row of exactly the maximum size survives a restart and a checkpoint image.
#[test]
fn a_row_of_the_maximum_size_survives_restart_and_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let largest = payload(9, nusadb_btree::MAX_USER_TUPLE);
    let table = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.insert(txn, table, &largest).unwrap();
        engine.commit(txn).unwrap();
        table
    };
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), vec![largest.clone()]);
    engine.checkpoint().unwrap();
    drop(engine);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), vec![largest]);
}
