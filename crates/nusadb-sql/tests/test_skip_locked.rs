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
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema, TxnId};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, analyze, catalog_list_indexes, execute_in_txn,
    parse, plan,
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

/// Like [`Cat`], but reports the engine's real scannable indexes (as the production wire catalog
/// does) so the planner can choose the same index scans a live query would — needed to exercise the
/// ordered-index-scan path under a lock, which a masked-index catalog would hide.
struct IndexedCat<'a>(&'a dyn StorageEngine, TxnId);
impl Catalog for IndexedCat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, Error> {
        catalog_list_indexes(self.0, self.1, name)
    }
}

/// Run one statement inside `txn`, returning its result (no commit/rollback here).
fn run_in(engine: &dyn StorageEngine, txn: TxnId, sql: &str) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    execute_in_txn(plan(logical), engine, txn)
}

/// Like [`run_in`], but analyzes against the index-exposing catalog so the plan uses real indexes.
fn run_in_indexed(
    engine: &dyn StorageEngine,
    txn: TxnId,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &IndexedCat(engine, txn))?;
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

/// Regression: `ORDER BY <indexed NOT NULL col> LIMIT n FOR UPDATE SKIP LOCKED` must fill the LIMIT
/// from *lockable* rows even when a locked row sorts *within* the first `n` — analyzed against a
/// catalog that exposes the real `PRIMARY KEY` index, exactly as the live wire path does.
///
/// This closes the harness gap that let a silent wrong result ship: the ordered-index-scan
/// sort-elimination caps the scan at `n` *visible* rows in the engine, which has no notion of locks,
/// so a locked row inside the cap dropped the result below the LIMIT. Here job 1 is locked and sorts
/// first, so the buggy plan capped the index scan at 1 row (job 1), the executor then skipped it as
/// locked, and the query returned `[]` instead of the next lockable job. The fix disqualifies the
/// ordered index scan under SKIP LOCKED, keeping the Sort+SeqScan path that skips locked rows mid
/// scan. Load-bearing: with the guard removed this returns `[]`.
#[test]
fn skip_locked_limit_fills_past_a_locked_row_within_the_cap() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE jobs (id INT PRIMARY KEY, payload TEXT)",
    );
    run(
        &engine,
        "INSERT INTO jobs VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    );

    // Worker 1 locks job 1 — the lowest id, which an ascending ORDER BY would return first.
    let worker1 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    assert_eq!(
        ids(run_in_indexed(
            &engine,
            worker1,
            "SELECT id FROM jobs WHERE id = 1 FOR UPDATE"
        )
        .unwrap()),
        vec![1]
    );

    // Worker 2 wants one claimable job in id order. Job 1 is locked and sorts first, but there are
    // lockable jobs past it — the LIMIT must fill from job 2, not fall short because the first row
    // in key order happened to be locked.
    let worker2 = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    assert_eq!(
        ids(run_in_indexed(
            &engine,
            worker2,
            "SELECT id FROM jobs ORDER BY id ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
        )
        .unwrap()),
        vec![2],
        "the LIMIT must skip the locked lowest id and take the next lockable job, not return empty"
    );

    let _ = engine.rollback(worker1);
    let _ = engine.rollback(worker2);
}
