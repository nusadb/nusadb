//! A storage error in the middle of a change stops the engine: memory may then disagree with the
//! log, so every later operation is refused and, above all, no checkpoint makes that memory the
//! durable truth. A restart rebuilds the database from its log and last image.
//!
//! The errors here are real ones: pages of the checkpoint image damaged on disk after it was
//! written, read for the first time by the change itself.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic and damages bytes by offset"
)]

use std::path::{Path, PathBuf};

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, PAGE_SIZE, StorageEngine, TableId, Tid};

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

fn rows(engine: &BtreeEngine, table: TableId) -> usize {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut n = 0;
    while scan.try_next().unwrap().is_some() {
        n += 1;
    }
    drop(scan);
    engine.commit(txn).unwrap();
    n
}

/// A database of `n` rows with an index on them, checkpointed and closed: every page is in the
/// image's segments. Returns the table and index ids.
fn checkpointed(wal: &Path, n: u64) -> (TableId, nusadb_core::IndexId) {
    let engine = BtreeEngine::open(wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    let index = engine
        .create_index(
            txn,
            &IndexDef {
                name: "t_v".to_owned(),
                table,
                columns: vec!["v".to_owned()],
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: IndexKind::BTree,
                unique: false,
            },
        )
        .unwrap();
    for i in 0..n {
        let v = format!("row-{i:06}").into_bytes();
        let tid = engine.insert(txn, table, &v).unwrap();
        engine.index_insert(txn, index, &v, tid).unwrap();
    }
    engine.commit(txn).unwrap();
    engine.checkpoint().unwrap();
    (table, index)
}

/// The segment files of the database at `wal`.
fn segments(wal: &Path) -> Vec<PathBuf> {
    let dir = PathBuf::from(format!("{}.pages", wal.display()));
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect()
}

/// Flip a byte in the middle of every page of every segment; return the originals to put back.
fn damage(wal: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    segments(wal)
        .into_iter()
        .map(|path| {
            let clean = std::fs::read(&path).unwrap();
            let mut bad = clean.clone();
            for page in 0..bad.len() / PAGE_SIZE {
                bad[page * PAGE_SIZE + PAGE_SIZE / 2] ^= 0x5A;
            }
            std::fs::write(&path, &bad).unwrap();
            (path, clean)
        })
        .collect()
}

fn repair(originals: &[(PathBuf, Vec<u8>)]) {
    for (path, clean) in originals {
        std::fs::write(path, clean).unwrap();
    }
}

fn assert_stopped(result: nusadb_core::Result<impl Sized>) {
    match result {
        Ok(_) => panic!("a stopped engine must refuse the operation"),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("stopped after a storage error"), "{msg}");
            assert!(msg.contains("restart"), "{msg}");
        },
    }
}

/// An insert that hits a damaged page stops the engine; nothing afterwards is served, no
/// checkpoint rewrites the image, and a restart once the damage is gone has every committed row.
#[test]
fn a_failed_row_change_stops_the_engine_and_a_restart_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, _) = checkpointed(&wal, 2000);
    let originals = damage(&wal);
    let image = dir.path().join("btree.wal.ckpt");
    let image_before = std::fs::read(&image).unwrap();

    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(engine.fault().is_none());
    let txn = engine.begin(RC).unwrap();
    let err = engine.insert(txn, table, b"new").unwrap_err().to_string();
    assert!(err.contains("stopped after a storage error"), "{err}");
    assert!(err.contains("checksum"), "the cause is named: {err}");
    assert!(engine.fault().is_some_and(|f| f.contains("checksum")));

    // Nothing more is served, by the transaction that hit it or by a new one.
    assert_stopped(engine.insert(txn, table, b"again"));
    assert_stopped(engine.scan(txn, table));
    assert_stopped(engine.commit(txn));
    assert_stopped(engine.begin(RC));
    assert_stopped(engine.checkpoint());
    assert_stopped(engine.purge());
    // The transaction can still be ended; the engine stays stopped.
    let _ = engine.rollback(txn);
    assert_stopped(engine.begin(RC));
    drop(engine);
    assert_eq!(
        std::fs::read(&image).unwrap(),
        image_before,
        "no checkpoint made the stopped engine's memory durable"
    );

    // The restart, once the damage is gone, rebuilds the database from its image and log.
    repair(&originals);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(engine.fault().is_none());
    assert_eq!(rows(&engine, table), 2000);
}

/// An index change that hits a damaged page stops the engine the same way.
#[test]
fn a_failed_index_change_stops_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, index) = checkpointed(&wal, 2000);
    let originals = damage(&wal);

    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let err = engine
        .index_insert(
            txn,
            index,
            b"row-000500",
            Tid {
                page: nusadb_core::PageId(0),
                slot: nusadb_core::SlotIdx(1),
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("stopped after a storage error"), "{err}");
    assert_stopped(engine.begin(RC));
    assert_stopped(engine.checkpoint());
    let _ = engine.rollback(txn);
    drop(engine);

    repair(&originals);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), 2000);
}

/// A plain read of a damaged page is an error for that read only: nothing was being changed, so
/// the engine goes on serving once the page is readable. (The background purge is different: its
/// pass changes rows, see below.)
#[test]
fn a_failed_read_does_not_stop_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, _) = checkpointed(&wal, 2000);
    let originals = damage(&wal);
    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let failed = engine.scan(txn, table).map_or(true, |mut scan| {
        loop {
            match scan.try_next() {
                Ok(Some(_)) => {},
                Ok(None) => break false,
                Err(_) => break true,
            }
        }
    });
    assert!(failed, "the damaged pages fail the scan");
    assert!(
        engine.fault().is_none(),
        "a read alone does not stop the engine"
    );
    engine.commit(txn).unwrap();
    drop(engine);
    repair(&originals);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), 2000);
}

/// The background purge walks every table: a damaged page it meets part way through a pass stops
/// the engine, since rows already removed earlier in the pass may still have index entries.
#[test]
fn a_purge_that_meets_a_damaged_page_stops_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, _) = checkpointed(&wal, 2000);
    let originals = damage(&wal);
    let engine = BtreeEngine::open(&wal).unwrap();
    let err = engine.purge().unwrap_err().to_string();
    assert!(err.contains("stopped after a storage error"), "{err}");
    assert_stopped(engine.begin(RC));
    assert_stopped(engine.checkpoint());
    drop(engine);
    repair(&originals);
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine, table), 2000);
}

/// Once stopped, a savepoint rollback and DDL are refused before they change anything, while a
/// full rollback still ends the transaction.
#[test]
fn a_stopped_engine_refuses_savepoint_rollback_and_ddl_but_ends_transactions() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (table, _) = checkpointed(&wal, 2000);
    let _originals = damage(&wal);
    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    engine.savepoint(txn, "s").unwrap();
    assert!(engine.insert(txn, table, b"new").is_err());
    assert!(engine.fault().is_some());
    let tables_before = engine.list_tables().unwrap();
    assert_stopped(engine.rollback_to(txn, "s"));
    assert_stopped(engine.drop_table(txn, table));
    assert_stopped(engine.create_table(txn, &table_def()));
    assert_eq!(
        engine.list_tables().unwrap(),
        tables_before,
        "a refused DDL changes nothing"
    );
    engine.rollback(txn).unwrap();
    assert_stopped(engine.begin(RC));
}
