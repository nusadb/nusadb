//! One database, one engine: opening a database takes an exclusive lock on `<wal>.lock`, so a
//! second engine (in this process or another) is refused while the first is open, and the
//! operating system releases the lock when the holder ends, even when it is killed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

fn refused(result: nusadb_core::Result<BtreeEngine>) -> String {
    match result {
        Ok(_) => panic!("a second open of a database in use must be refused"),
        Err(e) => e.to_string(),
    }
}

fn with_a_row(wal: &Path) {
    let engine = BtreeEngine::open(wal).unwrap();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            txn,
            &TableDef {
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: vec![ColumnDef {
                    name: "v".to_owned(),
                    ty: ColumnType::Bytes,
                    nullable: false,
                }],
            },
        )
        .unwrap();
    engine.insert(txn, table, b"row").unwrap();
    engine.commit(txn).unwrap();
}

/// Every way of opening a database is refused while an engine has it open, with a message that
/// says so, and succeeds once that engine is gone.
#[test]
fn a_second_open_is_refused_until_the_first_engine_closes() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    let first = BtreeEngine::open(&wal).unwrap();
    let err = refused(BtreeEngine::open(&wal));
    assert!(err.contains("already open in another process"), "{err}");
    assert!(err.contains("btree.wal.lock"), "{err}");
    refused(BtreeEngine::open_standby(&wal));
    refused(BtreeEngine::open_with_archive(&wal, None));
    refused(BtreeEngine::open_until(
        &wal,
        nusadb_btree::RecoveryTarget::Latest,
    ));
    drop(first);
    let again = BtreeEngine::open(&wal).unwrap();
    assert_eq!(again.list_tables().unwrap().len(), 1);
}

/// A scan still open after its engine is dropped keeps the database locked: it can still reach
/// the database's files. The lock goes with the last of them.
#[test]
fn the_lock_lasts_as_long_as_anything_can_reach_the_files() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = engine.lookup_table("t").unwrap().unwrap().id;
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let scan = engine.scan(txn, table).unwrap();
    drop(engine);
    let err = refused(BtreeEngine::open(&wal));
    assert!(err.contains("already open"), "{err}");
    drop(scan);
    BtreeEngine::open(&wal).unwrap();
}

/// A restore or a standby seed into a database that is open is refused before it touches
/// anything.
#[test]
fn restore_and_seed_refuse_a_database_that_is_open() {
    let source = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    {
        let engine = BtreeEngine::open_with_archive(
            source.path().join("btree.wal"),
            Some(archive.path().to_path_buf()),
        )
        .unwrap();
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
    }
    let target = tempfile::tempdir().unwrap();
    let out = target.path().join("btree.wal");
    let open = BtreeEngine::open(&out).unwrap();
    let err = BtreeEngine::restore_from_archive(
        archive.path(),
        nusadb_btree::RecoveryTarget::Latest,
        &out,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("already open"), "{err}");
    let err = nusadb_btree::seed_standby(archive.path(), &out)
        .unwrap_err()
        .to_string();
    assert!(err.contains("already open"), "{err}");
    drop(open);
}

/// Set in the environment of the child process `the_lock_holds_across_processes` starts: the
/// database it opens and holds until killed.
const HOLD_ENV: &str = "NUSADB_TEST_HOLD_DATABASE";

/// Not a test on its own: when started by `the_lock_holds_across_processes` with the database
/// in the environment, open it, say so, and hold it until killed, until its parent goes away
/// (its stdin closes), or for two minutes at most, so it can never outlive a failed run.
#[test]
fn lock_holder() {
    let Ok(wal) = std::env::var(HOLD_ENV) else {
        return;
    };
    let wal = PathBuf::from(wal);
    let engine = BtreeEngine::open(&wal).unwrap();
    std::fs::write(wal.with_extension("ready"), b"").unwrap();
    let (gone, parent_gone) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = std::io::Read::read_to_end(&mut std::io::stdin(), &mut sink);
        let _ = gone.send(());
    });
    let _ = parent_gone.recv_timeout(Duration::from_mins(2));
    drop(engine);
}

/// The holder process, killed and reaped however the test ends.
struct Holder(std::process::Child);

impl Drop for Holder {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Another process holding the database keeps this one out; killing it outright (no cleanup
/// runs) releases the lock, and the database opens.
#[test]
fn the_lock_holds_across_processes_and_dies_with_its_holder() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    let mut holder = Holder(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "lock_holder", "--nocapture"])
            .env(HOLD_ENV, &wal)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let ready = wal.with_extension("ready");
    let started = Instant::now();
    while !ready.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the holder never opened the database"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let err = refused(BtreeEngine::open(&wal));
    assert!(err.contains("already open in another process"), "{err}");
    // Killed, not asked to stop: nothing of the holder runs after this.
    holder.0.kill().unwrap();
    holder.0.wait().unwrap();
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(engine.list_tables().unwrap().len(), 1);
}
