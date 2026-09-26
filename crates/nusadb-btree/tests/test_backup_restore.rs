//! Backup and restore through the checkpoint image.
//!
//! The image file (`<wal>.ckpt`) is a complete, self-contained copy of the committed state as of
//! the checkpoint's watermark, and it is only ever replaced by an atomic rename: a copy of it
//! taken at any moment after a checkpoint is a consistent point-in-time backup, even while the
//! engine keeps writing. Restoring is placing that copy into an empty data directory; the engine
//! opens it as an image plus an empty log tail.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId};

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

fn insert(engine: &BtreeEngine, table: TableId, payload: &[u8]) {
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, payload).unwrap();
    engine.commit(txn).unwrap();
}

fn payloads(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
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

#[test]
fn a_copy_of_the_checkpoint_image_restores_the_state_as_of_the_checkpoint() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    insert(&engine, table, b"before-1");
    insert(&engine, table, b"before-2");

    // The backup point: checkpoint, then copy the image. Writes that follow belong to the live
    // database only.
    engine.checkpoint().unwrap();
    let backup = tempfile::tempdir().unwrap();
    std::fs::copy(
        live.path().join("btree.wal.ckpt"),
        backup.path().join("btree.wal.ckpt"),
    )
    .unwrap();
    insert(&engine, table, b"after");
    assert_eq!(
        payloads(&engine, table),
        vec![
            b"after".to_vec(),
            b"before-1".to_vec(),
            b"before-2".to_vec()
        ]
    );

    // Restore: the copied image in an otherwise empty directory opens as that point in time.
    let restored = BtreeEngine::open(backup.path().join("btree.wal")).unwrap();
    let restored_table = restored
        .lookup_table("t")
        .unwrap()
        .expect("table in the image");
    assert_eq!(
        payloads(&restored, restored_table.id),
        vec![b"before-1".to_vec(), b"before-2".to_vec()]
    );
    // And the restored database is a working one: it takes new writes and survives a reopen.
    insert(&restored, restored_table.id, b"restored-write");
    drop(restored);
    let reopened = BtreeEngine::open(backup.path().join("btree.wal")).unwrap();
    assert_eq!(payloads(&reopened, restored_table.id).len(), 3);

    // The live database is untouched by the backup.
    assert_eq!(payloads(&engine, table).len(), 3);
}

#[test]
fn the_image_is_stable_while_the_engine_keeps_writing_and_checkpointing() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    insert(&engine, table, b"v1");
    engine.checkpoint().unwrap();

    // Hold the image open as a backup tool would, then let the engine write and checkpoint
    // again: the new image is published by rename, so the open handle still reads the old,
    // complete image.
    let mut held = std::fs::File::open(live.path().join("btree.wal.ckpt")).unwrap();
    insert(&engine, table, b"v2");
    engine.checkpoint().unwrap();
    let backup = tempfile::tempdir().unwrap();
    let mut out = std::fs::File::create(backup.path().join("btree.wal.ckpt")).unwrap();
    std::io::copy(&mut held, &mut out).unwrap();
    drop(out);

    let restored = BtreeEngine::open(backup.path().join("btree.wal")).unwrap();
    let t = restored.lookup_table("t").unwrap().unwrap();
    assert_eq!(payloads(&restored, t.id), vec![b"v1".to_vec()]);
}
