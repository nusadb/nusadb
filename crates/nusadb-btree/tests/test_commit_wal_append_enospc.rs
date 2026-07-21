//! DST pin for the disk-full (ENOSPC) commit path — the recoverable twin of the fsync fail-stop
//! (`test_commit_fsync_failstop.rs`).
//!
//! When a WAL commit-marker **append** fails (the write syscall reports ENOSPC *before* the record
//! reaches the log), the honest response is to SELF-HEAL, not fail-stop: nothing durable was
//! written, so `commit` returns an error, the transaction rolls back cleanly with nothing staged
//! and its locks released, the engine keeps serving, and recovery never sees the failed commit.
//! This is the exact opposite of the fsync case — there the bytes reached the file, so the
//! transaction could resurrect and `process::abort` is the only truthful answer; here nothing
//! reached the file, so returning an error and carrying on is correct.
//!
//! We identified this path as carefully designed (commit stages nothing before
//! the marker append; the failed append rolls the txn back rather than stranding it) but never
//! exercised by a test — `FaultRates` injected torn/fsync/power-loss but no write-syscall failure.
//! This pin arms the one-shot `dst-fail-next-wal-append` fault point and proves the four claimed
//! properties: (a) `commit` errors with no partial durable state, (b) the engine stays usable,
//! (c) no corruption, (d) no `process::abort`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "DST pin asserts via unwrap/panic"
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

/// Scan every committed row of `table` in a fresh read transaction, returning the payloads in
/// row-id (insertion) order. Doubles as a corruption check — a broken tree would panic or diverge.
fn scan_payloads(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out
}

#[test]
fn commit_wal_append_enospc_rolls_back_cleanly_and_the_engine_survives() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("enospc.wal");

    let table_id;
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.commit(txn).unwrap();
        table_id = table;

        // A cleanly-committed row is the durable baseline the ENOSPC failure must not disturb.
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, b"committed-before").unwrap();
        engine.commit(txn).unwrap();

        // Arm the disk-full fault: the next WAL append reports ENOSPC before writing anything.
        engine.dst_fail_next_wal_append();
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, b"doomed-by-enospc").unwrap();
        let outcome = engine.commit(txn);
        // (a) The commit surfaces the disk-full error rather than lying "Ok".
        assert!(
            matches!(&outcome, Err(nusadb_core::Error::Io(e)) if e.kind() == std::io::ErrorKind::StorageFull),
            "a disk-full commit-marker append must surface an ENOSPC error, got {outcome:?}"
        );

        // (a)+(c)+(d): the failed transaction left nothing staged — only the baseline row is
        // visible, the tree is intact, and the process is plainly still running these assertions
        // (no fail-stop on this recoverable path).
        assert_eq!(
            scan_payloads(&engine, table),
            vec![b"committed-before".to_vec()],
            "the ENOSPC-failed commit must leave no partial durable state"
        );

        // (b) The engine keeps serving: the one-shot fault cleared when it fired, so a fresh
        // transaction commits normally.
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, b"after-enospc").unwrap();
        engine.commit(txn).unwrap();
        assert_eq!(
            scan_payloads(&engine, table),
            vec![b"committed-before".to_vec(), b"after-enospc".to_vec()],
            "a post-ENOSPC transaction commits normally on the surviving engine"
        );
    }

    // "Restart": recovery replays only the durable prefix. The ENOSPC-failed commit's marker never
    // reached the log, so its row must be absent — exactly the two cleanly-committed rows survive.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(
        scan_payloads(&engine, table_id),
        vec![b"committed-before".to_vec(), b"after-enospc".to_vec()],
        "recovery must see only durably-committed rows, never the ENOSPC-failed commit"
    );
}

/// D-STABLE-ENOSPC rec #3: a restart while the disk is *still* full, then freed, must leave no
/// half-baked state. This composes recovery with the disk-full path: the engine reopens (recovery
/// replays the durable prefix), the very first post-recovery commit hits ENOSPC and fails cleanly
/// (recovery's state untouched), and once space frees the engine resumes committing durably — with
/// a final restart proving only the durable rows survive and the blocked write left nothing.
#[test]
fn restart_with_disk_still_full_then_freed_leaves_no_half_baked_state() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("enospc_restart.wal");

    // Session 1: establish a durable baseline, then close — a clean restart boundary.
    let table_id;
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.commit(txn).unwrap();
        table_id = table;
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, b"durable-baseline").unwrap();
        engine.commit(txn).unwrap();
    }

    // Session 2: reopen with the disk "still full". Recovery replays the durable prefix, then the
    // first post-recovery commit hits ENOSPC and must fail without disturbing the recovered state.
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        assert_eq!(
            scan_payloads(&engine, table_id),
            vec![b"durable-baseline".to_vec()],
            "recovery must replay exactly the durable baseline"
        );

        engine.dst_fail_next_wal_append();
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table_id, b"blocked-while-full").unwrap();
        let outcome = engine.commit(txn);
        assert!(
            matches!(&outcome, Err(nusadb_core::Error::Io(e)) if e.kind() == std::io::ErrorKind::StorageFull),
            "the first commit after recovery, disk still full, must fail with ENOSPC, got {outcome:?}"
        );
        assert_eq!(
            scan_payloads(&engine, table_id),
            vec![b"durable-baseline".to_vec()],
            "a disk-full commit right after recovery must leave only the recovered baseline"
        );

        // Space freed (the one-shot fault already cleared): the engine resumes committing durably.
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table_id, b"after-space-freed").unwrap();
        engine.commit(txn).unwrap();
    }

    // Session 3: a final restart proves the freed-disk write is durable and the blocked one is gone.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(
        scan_payloads(&engine, table_id),
        vec![b"durable-baseline".to_vec(), b"after-space-freed".to_vec()],
        "only durably-committed rows survive the restart cycle; the blocked write left nothing"
    );
}
