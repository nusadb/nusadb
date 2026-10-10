//! A foreign key holds under concurrent writers: a child row written while another transaction
//! deletes its parent, or changes the parent's referenced key, never commits beside that change.
//! One of the two loses with a serialization conflict (`40001`) or a foreign key violation
//! (`23503`), whichever order they run in, so no child is left without its parent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/expect/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema, TxnId};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, analyze, execute_in_txn, parse, plan,
};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn run_in(engine: &dyn StorageEngine, txn: TxnId, sql: &str) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    execute_in_txn(plan(logical), engine, txn)
}

fn run(engine: &dyn StorageEngine, sql: &str) {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    run_in(engine, txn, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    engine.commit(txn).unwrap();
}

fn count(engine: &dyn StorageEngine, sql: &str) -> i64 {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    let result = run_in(engine, txn, sql).unwrap();
    engine.commit(txn).unwrap();
    match result {
        ExecutionResult::Rows { rows, .. } => match rows.first().and_then(|r| r.first()) {
            Some(nusadb_sql::ast::Value::Int(n)) => *n,
            other => panic!("expected a count, got {other:?}"),
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

fn setup(engine: &dyn StorageEngine, on_delete: &str) {
    run(
        engine,
        "CREATE TABLE camp (id INT PRIMARY KEY, lvl TEXT NOT NULL, rev INT NOT NULL, \
         CONSTRAINT camp_uq UNIQUE (id, lvl))",
    );
    run(
        engine,
        &format!(
            "CREATE TABLE budget (id INT PRIMARY KEY, camp INT, scope TEXT NOT NULL, \
             CONSTRAINT budget_fk FOREIGN KEY (camp, scope) REFERENCES camp (id, lvl) {on_delete})"
        ),
    );
    run(engine, "INSERT INTO camp VALUES (10, 'ad_group', 1)");
}

/// No budget row may point at a missing `(camp, scope)`.
fn orphans(engine: &dyn StorageEngine) -> i64 {
    count(
        engine,
        "SELECT count(*) FROM budget WHERE NOT EXISTS \
         (SELECT 1 FROM camp WHERE camp.id = budget.camp AND camp.lvl = budget.scope)",
    )
}

fn is_refusal(err: &Error) -> bool {
    matches!(err.sqlstate(), "40001" | "23503")
}

/// Run `first` then `second` in two open transactions (in that order, each statement before the
/// other commits), commit whichever did not fail, and return how many orphans are left.
fn race(engine: &dyn StorageEngine, iso: IsolationLevel, first: &str, second: &str) -> i64 {
    let t1 = engine.begin(iso).unwrap();
    let t2 = engine.begin(iso).unwrap();
    // Both snapshots are taken before either writes.
    run_in(engine, t1, "SELECT count(*) FROM camp").unwrap();
    run_in(engine, t2, "SELECT count(*) FROM budget").unwrap();
    let r1 = run_in(engine, t1, first);
    let r2 = run_in(engine, t2, second);
    for (txn, result, sql) in [(t1, &r1, first), (t2, &r2, second)] {
        match result {
            Ok(_) => {
                if let Err(e) = engine.commit(txn) {
                    assert!(is_refusal(&e.into()), "{sql}: commit failed oddly");
                }
            },
            Err(e) => {
                assert!(is_refusal(e), "{sql}: unexpected error {e}");
                engine.rollback(txn).unwrap();
            },
        }
    }
    orphans(engine)
}

/// Each order of a parent change and a child insert, at each isolation level.
fn every_order(parent_change: &str, on_delete: &str) {
    let child_insert = "INSERT INTO budget VALUES (100, 10, 'ad_group')";
    for iso in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::RepeatableRead,
        IsolationLevel::Serializable,
    ] {
        for parent_first in [true, false] {
            let engine = BtreeEngine::new();
            setup(&engine, on_delete);
            let (first, second) = if parent_first {
                (parent_change, child_insert)
            } else {
                (child_insert, parent_change)
            };
            assert_eq!(
                race(&engine, iso, first, second),
                0,
                "{iso:?}, {first} then {second}: a budget row lost its parent"
            );
        }
    }
}

#[test]
fn deleting_a_parent_races_a_child_insert_without_an_orphan() {
    every_order("DELETE FROM camp WHERE id = 10", "");
}

#[test]
fn changing_a_referenced_key_races_a_child_insert_without_an_orphan() {
    every_order(
        "UPDATE camp SET lvl = 'campaign', rev = rev + 1 WHERE id = 10",
        "",
    );
}

#[test]
fn a_cascading_delete_races_a_child_insert_without_an_orphan() {
    every_order("DELETE FROM camp WHERE id = 10", "ON DELETE CASCADE");
}

/// A child committed after the deleting transaction's snapshot is still found.
#[test]
fn a_child_committed_after_the_parent_snapshot_still_blocks_the_delete() {
    let engine = BtreeEngine::new();
    setup(&engine, "");
    let deleter = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    run_in(&engine, deleter, "SELECT count(*) FROM budget").unwrap();
    run(&engine, "INSERT INTO budget VALUES (100, 10, 'ad_group')");
    let err = run_in(&engine, deleter, "DELETE FROM camp WHERE id = 10")
        .expect_err("the parent has a committed child");
    assert!(is_refusal(&err), "{err}");
    engine.rollback(deleter).unwrap();
    assert_eq!(orphans(&engine), 0);
}

/// The same for an update that changes the referenced key.
#[test]
fn a_child_committed_after_the_parent_snapshot_still_blocks_a_key_change() {
    let engine = BtreeEngine::new();
    setup(&engine, "");
    let updater = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    run_in(&engine, updater, "SELECT count(*) FROM budget").unwrap();
    run(&engine, "INSERT INTO budget VALUES (100, 10, 'ad_group')");
    let err = run_in(
        &engine,
        updater,
        "UPDATE camp SET lvl = 'campaign' WHERE id = 10",
    )
    .expect_err("the key has a committed child");
    assert!(is_refusal(&err), "{err}");
    engine.rollback(updater).unwrap();
    assert_eq!(orphans(&engine), 0);
}

/// A parent deleted after the inserting transaction's snapshot is no parent.
#[test]
fn a_parent_deleted_after_the_child_snapshot_refuses_the_insert() {
    let engine = BtreeEngine::new();
    setup(&engine, "");
    let inserter = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    run_in(&engine, inserter, "SELECT count(*) FROM camp").unwrap();
    run(&engine, "DELETE FROM camp WHERE id = 10");
    let err = run_in(
        &engine,
        inserter,
        "INSERT INTO budget VALUES (100, 10, 'ad_group')",
    )
    .expect_err("the parent is gone");
    assert!(is_refusal(&err), "{err}");
    engine.rollback(inserter).unwrap();
    assert_eq!(orphans(&engine), 0);
}

/// Changing a column the key does not cover is no conflict with a child insert.
#[test]
fn a_non_key_parent_update_and_a_child_insert_both_commit() {
    for parent_first in [true, false] {
        let engine = BtreeEngine::new();
        setup(&engine, "");
        let parent = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let child = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let update = "UPDATE camp SET rev = rev + 1 WHERE id = 10";
        let insert = "INSERT INTO budget VALUES (100, 10, 'ad_group')";
        if parent_first {
            run_in(&engine, parent, update).unwrap();
            run_in(&engine, child, insert).unwrap();
        } else {
            run_in(&engine, child, insert).unwrap();
            run_in(&engine, parent, update).unwrap();
        }
        engine.commit(parent).unwrap();
        engine.commit(child).unwrap();
        assert_eq!(count(&engine, "SELECT count(*) FROM budget"), 1);
    }
}

/// Two children of one parent do not conflict with each other.
#[test]
fn two_child_inserts_under_one_parent_both_commit() {
    let engine = BtreeEngine::new();
    setup(&engine, "");
    let a = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let b = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    run_in(
        &engine,
        a,
        "INSERT INTO budget VALUES (100, 10, 'ad_group')",
    )
    .unwrap();
    run_in(
        &engine,
        b,
        "INSERT INTO budget VALUES (101, 10, 'ad_group')",
    )
    .unwrap();
    engine.commit(a).unwrap();
    engine.commit(b).unwrap();
    assert_eq!(count(&engine, "SELECT count(*) FROM budget"), 2);
}

/// One transaction may reference a key and then remove it itself: its own locks do not conflict.
#[test]
fn one_transaction_adds_a_child_then_removes_its_parent() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE node (id INT PRIMARY KEY, up INT REFERENCES node (id) ON DELETE CASCADE)",
    );
    run(&engine, "INSERT INTO node VALUES (1, NULL)");
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    run_in(&engine, txn, "INSERT INTO node VALUES (2, 1)").unwrap();
    run_in(&engine, txn, "DELETE FROM node WHERE id = 1").unwrap();
    engine.commit(txn).unwrap();
    assert_eq!(count(&engine, "SELECT count(*) FROM node"), 0);
}
