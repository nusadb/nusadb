//! A checkpoint that pauses admission: under writers that never leave a quiet instant on their
//! own, holding new transactions for a bounded moment drains the active set, the existing
//! checkpoint runs on the quiesced engine, and the writers carry on afterwards with nothing lost.
//! A transaction that never ends defeats the pause within its budget, and admission resumes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use nusadb_btree::{BtreeEngine, CheckpointOutcome};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn open_with_table(dir: &tempfile::TempDir) -> (Arc<BtreeEngine>, TableId) {
    let engine = Arc::new(BtreeEngine::open(dir.path().join("btree.wal")).unwrap());
    let txn = engine.begin(RC).unwrap();
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
    engine.commit(txn).unwrap();
    (engine, table)
}

fn count_rows(engine: &BtreeEngine, table: TableId) -> usize {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut n = 0;
    while scan.try_next().unwrap().is_some() {
        n += 1;
    }
    engine.commit(txn).unwrap();
    n
}

#[test]
fn pausing_admission_checkpoints_under_writers_that_never_go_quiet() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table) = open_with_table(&dir);
    let stop = Arc::new(AtomicBool::new(false));
    let committed = Arc::new(AtomicUsize::new(0));
    // Four writers in a tight autocommit loop: with this overlap a quiet instant is rare.
    let writers: Vec<_> = (0..4)
        .map(|_| {
            let (engine, stop, committed) = (
                Arc::clone(&engine),
                Arc::clone(&stop),
                Arc::clone(&committed),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let txn = engine.begin(RC).unwrap();
                    engine.insert(txn, table, &[0xAB; 256]).unwrap();
                    engine.commit(txn).unwrap();
                    committed.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(200));

    let before = engine.wal_len().unwrap().unwrap();
    let outcome = engine
        .checkpoint_with_admission_pause(Duration::from_secs(5))
        .unwrap();
    assert!(
        matches!(outcome, CheckpointOutcome::Done { .. }),
        "{outcome:?}"
    );
    let after = engine.wal_len().unwrap().unwrap();
    assert!(after < before, "log did not shrink: {before} -> {after}");

    // Writers were only held, never failed, and keep going after the checkpoint.
    let at_checkpoint = committed.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(200));
    assert!(committed.load(Ordering::Relaxed) > at_checkpoint);
    stop.store(true, Ordering::Relaxed);
    for w in writers {
        w.join().unwrap();
    }
    let total = committed.load(Ordering::Relaxed);
    assert_eq!(count_rows(&engine, table), total);

    // Everything committed survives a reopen from image plus tail.
    drop(engine);
    let reopened = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    assert_eq!(count_rows(&reopened, table), total);
}

#[test]
fn a_transaction_that_never_ends_defeats_the_pause_and_admission_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table) = open_with_table(&dir);
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, b"held open").unwrap();
    let before = engine.wal_len().unwrap().unwrap();

    let outcome = engine
        .checkpoint_with_admission_pause(Duration::from_millis(150))
        .unwrap();
    assert!(
        matches!(outcome, CheckpointOutcome::StillBusy { active: 1, .. }),
        "{outcome:?}"
    );
    assert_eq!(
        engine.wal_len().unwrap().unwrap(),
        before,
        "nothing was written"
    );

    // Admission resumed: a new transaction starts at once, and the held one still commits.
    let later = engine.begin(RC).unwrap();
    engine.commit(later).unwrap();
    engine.commit(txn).unwrap();
    assert!(matches!(
        engine
            .checkpoint_with_admission_pause(Duration::from_millis(150))
            .unwrap(),
        CheckpointOutcome::Done { .. }
    ));
}

#[test]
fn the_in_memory_engine_reports_nothing_to_checkpoint_without_pausing() {
    let engine = BtreeEngine::new();
    assert!(
        engine
            .checkpoint_with_admission_pause(Duration::from_millis(10))
            .is_err()
    );
    // No pause is left behind: transactions begin immediately.
    let txn = engine.begin(RC).unwrap();
    engine.commit(txn).unwrap();
}

/// A `begin` that arrives during a pause waits, and is released the moment the pause ends, even
/// when the pause ends defeated.
#[test]
fn a_begin_arriving_during_a_defeated_pause_is_released_when_the_pause_ends() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, table) = open_with_table(&dir);
    let held = engine.begin(RC).unwrap();
    engine.insert(held, table, b"held open").unwrap();

    let waiter = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            // Give the pause time to start before asking for a transaction.
            std::thread::sleep(Duration::from_millis(50));
            let started = std::time::Instant::now();
            let txn = engine.begin(RC).unwrap();
            engine.commit(txn).unwrap();
            started.elapsed()
        })
    };
    let outcome = engine
        .checkpoint_with_admission_pause(Duration::from_millis(300))
        .unwrap();
    assert!(
        matches!(outcome, CheckpointOutcome::StillBusy { .. }),
        "{outcome:?}"
    );
    let waited = waiter.join().unwrap();
    // Held for the remainder of the pause, released right after it, never failed.
    assert!(waited >= Duration::from_millis(150), "{waited:?}");
    assert!(waited < Duration::from_secs(2), "{waited:?}");
    assert!(!engine.admission_paused());
    engine.commit(held).unwrap();
}
