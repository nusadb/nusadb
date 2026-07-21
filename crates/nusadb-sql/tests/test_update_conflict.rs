//! Concurrent-UPDATE conflict classification.
//!
//! An UPDATE that loses a write-write race must surface the retryable `40001` serialization
//! conflict — never a bogus duplicate-key violation (the committed-state uniqueness re-check used
//! to mistake the *newer committed version of the row being rewritten* for a duplicate, because it
//! excluded rewritten rows by tid and a concurrent committer changes the row's tid), and never an
//! internal "tuple not found" (the engine now classifies a write to a version another transaction
//! superseded as the OCC first-updater-wins conflict). A *genuine* duplicate key from an UPDATE
//! stays a constraint violation.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/expect/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, IndexInfo, analyze, execute_in_txn, parse, plan};

/// Minimal analyzer catalog over the engine's schema as seen by `txn`.
struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

/// Run one statement inside `txn` (no commit/rollback here).
fn run_in(engine: &dyn StorageEngine, txn: nusadb_core::TxnId, sql: &str) -> Result<(), Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    execute_in_txn(plan(logical), engine, txn).map(|_| ())
}

/// Run one auto-committed statement.
fn run(engine: &dyn StorageEngine, sql: &str) -> Result<(), Error> {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    match run_in(engine, txn, sql) {
        Ok(()) => {
            engine.commit(txn).unwrap();
            Ok(())
        },
        Err(e) => {
            let _ = engine.rollback(txn);
            Err(e)
        },
    }
}

#[test]
fn losing_a_concurrent_update_race_is_a_serialization_conflict() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE acct (id INT PRIMARY KEY, bal INT)").unwrap();
    run(&engine, "INSERT INTO acct VALUES (1, 100), (2, 100)").unwrap();

    // txn2 freezes its snapshot first (REPEATABLE READ), so after txn1 commits, txn2's scan still
    // resolves the row's OLD tid — the deterministic version of the race the money-transfer
    // harness hits ~13% of the time under READ COMMITTED.
    let txn2 = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    run_in(&engine, txn2, "SELECT * FROM acct").unwrap(); // touch the snapshot

    let txn1 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    run_in(&engine, txn1, "UPDATE acct SET bal = bal - 10 WHERE id = 1").unwrap();
    engine.commit(txn1).unwrap();

    // txn2 now updates the same logical row: it must lose with 40001 — the old code reported a
    // duplicate-key violation on PK id=1 (an UPDATE that never touches the key!).
    let err = run_in(&engine, txn2, "UPDATE acct SET bal = bal - 10 WHERE id = 1")
        .expect_err("the second updater must lose the write-write race");
    assert_eq!(
        err.sqlstate(),
        "40001",
        "a lost update race must be the retryable serialization conflict, got: {err}"
    );
    assert!(
        !err.to_string().to_lowercase().contains("duplicate"),
        "must not misreport as a duplicate key: {err}"
    );
    let _ = engine.rollback(txn2);

    // A concurrent DELETE loser gets the same honest classification.
    let txn3 = engine.begin(IsolationLevel::RepeatableRead).unwrap();
    run_in(&engine, txn3, "SELECT * FROM acct").unwrap();
    let txn4 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    run_in(&engine, txn4, "UPDATE acct SET bal = 0 WHERE id = 2").unwrap();
    engine.commit(txn4).unwrap();
    let err = run_in(&engine, txn3, "DELETE FROM acct WHERE id = 2")
        .expect_err("the deleter must lose the write-write race");
    assert_eq!(err.sqlstate(), "40001", "lost delete race, got: {err}");
    let _ = engine.rollback(txn3);

    // Non-regression: a GENUINE duplicate key from an UPDATE is still a constraint violation.
    let err = run(&engine, "UPDATE acct SET id = 1 WHERE id = 2")
        .expect_err("moving id 2 onto existing id 1 must violate the primary key");
    assert!(
        err.to_string().to_lowercase().contains("duplicate"),
        "a real duplicate stays a duplicate-key violation: {err}"
    );
    assert_ne!(err.sqlstate(), "40001", "a real duplicate is not retryable");
}
