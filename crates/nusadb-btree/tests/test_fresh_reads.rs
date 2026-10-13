//! `begin_fresh_reads` / `end_fresh_reads`: between them a transaction reads the latest committed
//! rows plus its own, whatever its isolation level; calls nest; the outermost end puts the
//! transaction's snapshot back.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId, TxnId};

fn count(engine: &BtreeEngine, txn: TxnId, table: TableId) -> usize {
    let mut scan = engine.scan(txn, table).unwrap();
    let mut n = 0;
    while scan.try_next().unwrap().is_some() {
        n += 1;
    }
    n
}

#[test]
fn fresh_reads_see_later_commits_and_nest_and_restore_the_snapshot() {
    let engine = BtreeEngine::new();
    let setup = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            setup,
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
    engine.commit(setup).unwrap();
    let rr = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    engine.insert(rr, table, b"own").unwrap();
    let other = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.insert(other, table, b"later").unwrap();
    engine.commit(other).unwrap();
    assert_eq!(
        count(&engine, rr, table),
        1,
        "the snapshot misses the later commit"
    );
    engine.begin_fresh_reads(rr).unwrap();
    assert_eq!(
        count(&engine, rr, table),
        2,
        "fresh: the commit and its own row"
    );
    engine.begin_fresh_reads(rr).unwrap();
    engine.end_fresh_reads(rr).unwrap();
    assert_eq!(
        count(&engine, rr, table),
        2,
        "an inner end keeps the fresh reads"
    );
    engine.end_fresh_reads(rr).unwrap();
    assert_eq!(
        count(&engine, rr, table),
        1,
        "the outermost end restores the snapshot"
    );
    engine.commit(rr).unwrap();
}

/// A table `t` holding one row `old`, then a transaction at `level` that has read it, then another
/// that updates it and deletes it, so only `level`'s snapshot still needs the versions.
fn superseded(level: IsolationLevel) -> (BtreeEngine, TableId, TxnId) {
    let engine = BtreeEngine::new();
    let setup = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine
        .create_table(
            setup,
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
    let tid = engine.insert(setup, table, b"old").unwrap();
    engine.commit(setup).unwrap();
    let reader = engine.begin(level).unwrap();
    assert_eq!(count(&engine, reader, table), 1);
    let writer = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.update(writer, table, tid, b"new").unwrap();
    engine.commit(writer).unwrap();
    let writer = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    engine.delete(writer, table, tid).unwrap();
    engine.commit(writer).unwrap();
    (engine, table, reader)
}

/// Purge running while a transaction reads fresh keeps the versions its saved snapshot needs, at
/// `REPEATABLE READ` and `SERIALIZABLE`, also with the calls nested.
#[test]
fn purge_during_fresh_reads_keeps_the_saved_snapshot() {
    for level in [IsolationLevel::RepeatableRead, IsolationLevel::Serializable] {
        for nested in [false, true] {
            let (engine, table, reader) = superseded(level);
            engine.begin_fresh_reads(reader).unwrap();
            if nested {
                engine.begin_fresh_reads(reader).unwrap();
            }
            assert_eq!(
                count(&engine, reader, table),
                0,
                "fresh: the row is deleted"
            );
            engine.purge().unwrap();
            if nested {
                engine.end_fresh_reads(reader).unwrap();
                engine.purge().unwrap();
            }
            engine.end_fresh_reads(reader).unwrap();
            engine.purge().unwrap();
            let mut scan = engine.scan(reader, table).unwrap();
            let (_, tuple) = scan
                .try_next()
                .unwrap()
                .expect("the snapshot still sees the row");
            assert_eq!(&*tuple, b"old", "{level:?} nested {nested}");
            drop(scan);
            let committed = engine.commit(reader);
            if level == IsolationLevel::Serializable {
                // It read a row two transactions since changed and committed: its own
                // antidependency check, walking the chain the purge kept, refuses it.
                assert!(
                    matches!(
                        committed,
                        Err(nusadb_core::Error::SerializationConflict { .. })
                    ),
                    "{committed:?}"
                );
            } else {
                committed.unwrap();
            }
        }
    }
}
