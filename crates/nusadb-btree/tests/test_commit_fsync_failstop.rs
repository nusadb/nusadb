//! DST pin for the commit fail-stop: a WAL commit-marker fsync that FAILS after the marker
//! reached the file must `process::abort` — never return an error and keep serving.
//!
//! Why fail-stop is the only honest response (the fsyncgate shape): the failed `fsync` may
//! have left the `CommitTxn` record in the OS page cache with the kernel considering the page
//! clean, so the record can still reach disk and the transaction RESURRECT as committed after
//! a restart. Returning an error would tell the client "commit failed" for a transaction that
//! then comes back — a lie. Retrying the fsync is unsound (a second fsync can report success
//! while the data is already lost).
//!
//! Black-box fault injection cannot time this: a device-level error (dm-error, full disk)
//! always breaks the marker *append* first, which takes the recoverable self-heal path, so this
//! one is routed to an in-repo DST pin. The `dst-fault`
//! feature arms a one-shot fault point that fails the group-leader fsync AFTER its flush, and
//! since `process::abort` kills the test process, the scenario runs in a re-exec'd CHILD:
//!
//! 1. Parent re-runs this same test binary with `NUSADB_DST_CHILD=1` and a temp WAL path.
//! 2. Child: create table + commit (clean), insert a row, arm the fault, COMMIT → the engine
//!    must fail-stop (the child never gets to report the commit outcome).
//! 3. Parent asserts the child died abnormally with the FATAL fsync message on stderr —
//!    NOT a clean exit, which would mean `commit` returned and the server would keep serving.
//! 4. Parent re-opens the WAL ("restart"): the row IS visible — the marker survived in the
//!    file, so the transaction resurrected as committed, proving an error return would have
//!    lied. Recovery replays the durable prefix honestly and the engine serves new writes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "fail-stop integration test asserts via unwrap/panic"
)]

use std::path::PathBuf;
use std::process::Command;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;
const CHILD_ENV: &str = "NUSADB_DST_CHILD";
const WAL_ENV: &str = "NUSADB_DST_WAL";
const TEST_NAME: &str = "commit_fsync_failure_fail_stops_and_resurrects_on_restart";
/// The row whose commit gets the failing fsync — asserted visible again after "restart".
const PAYLOAD: &[u8] = b"poisoned-commit";

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

/// Child scenario — ends in `process::abort` inside `commit` when the pin holds. Every exit
/// from this function is an explicit `process::exit` so the parent can tell "commit returned"
/// (clean exit = pin violated) from the fail-stop it demands.
fn run_child() -> ! {
    let wal = PathBuf::from(std::env::var_os(WAL_ENV).expect("child needs the WAL path"));
    let engine = BtreeEngine::open(&wal).unwrap();

    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap(); // clean commit: the fault is not armed yet

    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, PAYLOAD).unwrap();
    engine.dst_fail_next_fsync();
    // Must never return: the marker append succeeds (the record is in the file), then the
    // group fsync reports failure → the engine must abort the process, not surface an Err
    // (the transaction would resurrect on restart) and not report Ok (durability unproven).
    let outcome = engine.commit(txn);
    eprintln!("child: commit returned {outcome:?} instead of fail-stopping");
    std::process::exit(0);
}

#[test]
fn commit_fsync_failure_fail_stops_and_resurrects_on_restart() {
    if std::env::var_os(CHILD_ENV).is_some() {
        run_child();
    }

    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("failstop.wal");
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe)
        // `--nocapture`: the FATAL eprintln must reach the pipe before the abort kills the
        // process — libtest's capture buffer would die with it.
        .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env(WAL_ENV, &wal)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "the child must fail-stop on a commit-fsync failure, but exited cleanly \
         (commit returned instead of aborting the process) — status {:?}, stderr:\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains("WAL commit fsync failed"),
        "the child died without the FATAL fsync diagnostic — a different failure, not the \
         fail-stop under test — status {:?}, stderr:\n{stderr}",
        out.status
    );

    // "Restart": recovery must replay the durable prefix honestly. The marker reached the
    // file before the fsync report, so the transaction the client never saw confirmed
    // RESURRECTS as committed — exactly why returning an error instead of fail-stopping
    // would have lied.
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = engine
        .lookup_table("t")
        .unwrap()
        .expect("the cleanly committed schema must survive");
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table.id).unwrap();
    let mut payloads = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        payloads.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    assert_eq!(
        payloads,
        vec![PAYLOAD.to_vec()],
        "the fsync-failed commit's marker was durable, so its row must resurrect on restart"
    );

    // And the recovered engine serves new writes — the fail-stop is a clean-slate restart,
    // not a bricked database.
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table.id, b"after-restart").unwrap();
    engine.commit(txn).unwrap();
}
