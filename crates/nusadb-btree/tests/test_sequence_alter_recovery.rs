//! `ALTER SEQUENCE` at the storage-engine layer: a changed definition (and a `RESTART`) is durable,
//! surviving a close/reopen via the `SeqAlter` WAL record — with and without a checkpoint that
//! rewrites the record as the sequence's captured `CREATE` image.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "durability pins assert via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{SequenceChange, SequenceDef, SequenceRestart};
use nusadb_core::{IsolationLevel, SequenceId, StorageEngine};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn def(name: &str) -> SequenceDef {
    SequenceDef {
        name: name.to_owned(),
        start: 1,
        increment: 1,
        min_value: 1,
        max_value: i64::MAX,
        cycle: false,
    }
}

/// Create sequence `s`, advance it once, then change its increment and `RESTART WITH 100`.
fn seed(engine: &BtreeEngine) -> SequenceId {
    let txn = engine.begin(RC).unwrap();
    let id = engine.create_sequence(txn, &def("s")).unwrap();
    engine.commit(txn).unwrap();

    assert_eq!(engine.sequence_next(id).unwrap(), 1);

    let txn = engine.begin(RC).unwrap();
    engine
        .alter_sequence(
            txn,
            id,
            &SequenceChange {
                increment: Some(10),
                restart: Some(SequenceRestart::To(100)),
                ..SequenceChange::default()
            },
        )
        .unwrap();
    engine.commit(txn).unwrap();

    // The change is visible immediately: next is exactly the RESTART target.
    assert_eq!(engine.sequence_next(id).unwrap(), 100);
    // And the new increment applies to the following advance.
    assert_eq!(engine.sequence_next(id).unwrap(), 110);
    id
}

fn assert_altered_definition_survives_restart(checkpoint_before_close: bool) {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("seq.wal");

    let id = {
        let engine = BtreeEngine::open(&wal).unwrap();
        let id = seed(&engine);
        if checkpoint_before_close {
            engine.checkpoint().unwrap();
        }
        id
    };

    // After recovery the altered increment (10) and the counter (last handed out: 110) both persist,
    // so the next advance is 120 — proving the `SeqAlter` definition change replayed, not just the
    // `SeqSet` counter.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(engine.sequence_next(id).unwrap(), 120);
}

#[test]
fn altered_sequence_definition_survives_restart_via_wal() {
    assert_altered_definition_survives_restart(false);
}

#[test]
fn altered_sequence_definition_survives_restart_via_checkpoint() {
    assert_altered_definition_survives_restart(true);
}

#[test]
fn alter_sequence_rejects_invalid_changes() {
    let engine = BtreeEngine::open(tempfile::tempdir().unwrap().path().join("seq.wal")).unwrap();
    let txn = engine.begin(RC).unwrap();
    let id = engine.create_sequence(txn, &def("s")).unwrap();
    engine.commit(txn).unwrap();

    let txn = engine.begin(RC).unwrap();
    // A zero increment is refused.
    assert!(
        engine
            .alter_sequence(
                txn,
                id,
                &SequenceChange {
                    increment: Some(0),
                    ..SequenceChange::default()
                }
            )
            .is_err()
    );
    // Inverted bounds are refused.
    assert!(
        engine
            .alter_sequence(
                txn,
                id,
                &SequenceChange {
                    min_value: Some(10),
                    max_value: Some(5),
                    ..SequenceChange::default()
                }
            )
            .is_err()
    );
    engine.commit(txn).unwrap();

    // A rejected change left the definition intact: the sequence still advances from its start.
    assert_eq!(engine.sequence_next(id).unwrap(), 1);
}
