//! Recovery must distinguish a torn WAL *tail* (a crash mid-append — safe to truncate to the last
//! good record) from a *hole in the middle* of the log (bit-rot / a bad sector).
//!
//! The WAL is the sole durable copy of the database in phase 1 (no checkpoint, volatile pages), so
//! truncating at a mid-log hole would silently lose every committed transaction past it AND destroy
//! the still-intact evidence. A hole must therefore make `open`
//! REFUSE loudly and leave the file untouched; only a genuine torn tail may be truncated.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "recovery integration test asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

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

/// Build a durable WAL at `path` with several committed transactions, then close the engine so the
/// log is flushed. Returns the number of committed data rows (each visible after reopen).
fn build_log(path: &std::path::Path, rows: u64) {
    let engine = BtreeEngine::open(path).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    for i in 0..rows {
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, &i.to_le_bytes()).unwrap();
        engine.commit(txn).unwrap();
    }
    // Dropping the engine closes the WAL file; its bytes are already fsynced per commit.
    drop(engine);
}

/// A corruption in the MIDDLE of the log (a flipped byte in an early record's body) — with valid,
/// committed records after it — must make `open` REFUSE, and must NOT truncate the file.
#[test]
fn midlog_corruption_with_valid_records_after_refuses_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wal");
    build_log(&path, 40);
    let original_len = std::fs::metadata(&path).unwrap().len();
    assert!(original_len > 64, "the log should hold many records");

    // Flip the first byte of the FIRST record's body (offset 16 = just past the 16-byte header):
    // the first record's CRC then fails, but every later record stays valid — a mid-log hole.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[16] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let Err(err) = BtreeEngine::open(&path) else {
        panic!("opening a log with a mid-log hole (valid records after corruption) must fail");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("MIDDLE") || msg.contains("refusing to open"),
        "the error must name the mid-log corruption, got: {msg}"
    );
    // The file must be left UNTOUCHED (not truncated) so the operator can restore/repair it.
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        original_len,
        "a refused open must not modify the WAL file"
    );
}

/// A zeroed 512-byte sector in the MIDDLE of the log (a classic bad sector) must make `open` REFUSE.
/// Before the header was covered by the CRC, an all-zero header read as a valid `len=0` record
/// (because `crc32(&[]) == 0`), so the zeroed region masqueraded as empty records and recovery
/// desynced and silently truncated — losing every committed transaction past it
/// Now the zeroed region fails validation and the valid records
/// after it prove a hole.
#[test]
fn zeroed_midlog_sector_refuses_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wal");
    build_log(&path, 200);
    let original_len = std::fs::metadata(&path).unwrap().len();
    assert!(original_len > 2048, "the log should span many sectors");

    // Zero a 512-byte sector near the middle — records committed both before and after it.
    let mut bytes = std::fs::read(&path).unwrap();
    let mid = (bytes.len() / 2) & !0x1ff; // 512-aligned offset in the middle
    for b in &mut bytes[mid..mid + 512] {
        *b = 0;
    }
    std::fs::write(&path, &bytes).unwrap();

    let Err(err) = BtreeEngine::open(&path) else {
        panic!("a zeroed mid-log sector with valid records after it must refuse to open");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("MIDDLE") || msg.contains("refusing to open"),
        "the error must name the mid-log corruption, got: {msg}"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        original_len,
        "a refused open must not modify the WAL file"
    );
}

/// A corruption in the TAIL (a flipped byte in the LAST record, with nothing valid after it) is a
/// torn tail: `open` SUCCEEDS, truncating the bad tail and recovering every prior committed row.
#[test]
fn torn_tail_corruption_still_opens_and_recovers_the_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wal");
    build_log(&path, 40);

    // Flip the very last byte of the file: it lies in the last record's body, so that record's CRC
    // fails and nothing valid follows — a torn tail.
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    // Open succeeds; the corrupt last transaction is dropped, the rest recovered.
    let engine = BtreeEngine::open(&path).expect("a torn tail must still open");
    let txn = engine.begin(RC).unwrap();
    let table = engine
        .lookup_table("t")
        .unwrap()
        .expect("table t recovered");
    let mut scan = engine.scan(txn, table.id).unwrap();
    let mut count = 0u64;
    while scan.try_next().unwrap().is_some() {
        count += 1;
    }
    // At least the create + the first 39 inserts survive (the last insert's commit may be the torn
    // record); the point is that recovery succeeds and keeps the durable prefix.
    assert!(
        count >= 39,
        "a torn tail must recover the committed prefix, got {count} rows"
    );
    engine.commit(txn).unwrap();
}
