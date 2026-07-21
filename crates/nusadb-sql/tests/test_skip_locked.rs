//! `FOR UPDATE ... SKIP LOCKED` — the job-queue pattern (QA scale/production register).
//!
//! Workers claim rows without blocking on each other: a matched row whose lock another
//! transaction holds is skipped (excluded from the locks taken and from the output) instead of
//! aborting the statement with a serialization conflict.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/expect/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, analyze, execute_in_txn, parse, plan,
};

/// Minimal analyzer catalog over the engine's schema.
struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

/// Run one statement inside `txn`, returning its result (no commit/rollback here).
fn run_in(
    engine: &dyn StorageEngine,
    txn: nusadb_core::TxnId,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    execute_in_txn(plan(logical), engine, txn)
}

/// The single-column `id` values of a row result.
fn ids(result: ExecutionResult) -> Vec<i64> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|row| match row.first() {
                Some(Value::Int(id)) => *id,
                other => panic!("expected an integer id, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

/// Run one auto-committed statement (the analyzer catalog resolves committed schema only).
fn run(engine: &dyn StorageEngine, sql: &str) {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    run_in(engine, txn, sql).unwrap();
    engine.commit(txn).unwrap();
}

#[test]
fn skip_locked_claims_disjoint_rows_without_blocking() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE jobs (id INT PRIMARY KEY, payload TEXT)",
    );
    run(
        &engine,
        "INSERT INTO jobs VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')",
    );

    // Worker 1 claims jobs 1 and 2.
    let worker1 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    assert_eq!(
        ids(run_in(
            &engine,
            worker1,
            "SELECT id FROM jobs WHERE id <= 2 ORDER BY id FOR UPDATE"
        )
        .unwrap()),
        vec![1, 2]
    );

    // Worker 2 with SKIP LOCKED sees (and claims) only the unclaimed jobs — no 40001, no block.
    let worker2 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    assert_eq!(
        ids(run_in(
            &engine,
            worker2,
            "SELECT id FROM jobs ORDER BY id FOR UPDATE SKIP LOCKED"
        )
        .unwrap()),
        vec![3, 4],
        "rows locked by worker 1 must be skipped, not conflicted on"
    );

    // Worker 2 really holds 3 and 4 now: a third worker sees an empty queue.
    let worker3 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    assert_eq!(
        ids(run_in(
            &engine,
            worker3,
            "SELECT id FROM jobs ORDER BY id FOR UPDATE SKIP LOCKED"
        )
        .unwrap()),
        Vec::<i64>::new(),
        "every job is claimed, so SKIP LOCKED returns nothing"
    );

    // A LIMIT fills from lockable rows: release worker 1's claims, then LIMIT 1 takes the
    // lowest unclaimed id.
    engine.rollback(worker1).unwrap();
    assert_eq!(
        ids(run_in(
            &engine,
            worker3,
            "SELECT id FROM jobs ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
        )
        .unwrap()),
        vec![1]
    );

    // Plain FOR UPDATE (no SKIP LOCKED) still conflicts loudly on a claimed row.
    let worker4 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let err = run_in(
        &engine,
        worker4,
        "SELECT id FROM jobs WHERE id = 3 FOR UPDATE",
    )
    .expect_err("worker 2 holds job 3");
    assert_eq!(err.sqlstate(), "40001", "got: {err}");

    let _ = engine.rollback(worker2);
    let _ = engine.rollback(worker3);
    let _ = engine.rollback(worker4);
}
