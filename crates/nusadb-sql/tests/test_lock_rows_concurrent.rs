//! `SELECT ... FOR UPDATE [SKIP LOCKED]` under concurrency: a job queue claimed by several workers
//! processes every job exactly once, and a claimed row is never one another worker already changed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::single_match_else,
    reason = "integration test harness asserts via unwrap/expect/panic and indexes known result shapes"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema, TxnId};
use nusadb_sql::ast::Value;
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

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

fn run(engine: &dyn StorageEngine, sql: &str) -> Vec<Vec<Value>> {
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let out = match run_in(engine, txn, sql).unwrap() {
        ExecutionResult::Rows { rows, .. } => rows,
        _ => Vec::new(),
    };
    engine.commit(txn).unwrap();
    out
}

fn job_queue(lock: &str) -> (u64, u64) {
    let engine: Arc<BtreeEngine> = Arc::new(BtreeEngine::new());
    run(
        &*engine,
        "CREATE TABLE q (id INT NOT NULL, status TEXT NOT NULL, attempts INT NOT NULL, PRIMARY KEY (id))",
    );
    let jobs = 400;
    let values: Vec<String> = (1..=jobs).map(|i| format!("({i}, 'queued', 0)")).collect();
    run(
        &*engine,
        &format!("INSERT INTO q VALUES {}", values.join(",")),
    );
    let stale = Arc::new(AtomicU64::new(0));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let stale = Arc::clone(&stale);
            let lock = lock.to_owned();
            std::thread::spawn(move || {
                loop {
                    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
                    let claimed = run_in(
                        &*engine,
                        txn,
                        &format!(
                            "SELECT id FROM q WHERE status = 'queued' ORDER BY id LIMIT 1 {lock}"
                        ),
                    );
                    let claimed = match claimed {
                        Ok(result) => rows(result),
                        Err(_) => {
                            engine.rollback(txn).unwrap();
                            continue;
                        },
                    };
                    let Some(Value::Int(id)) = claimed.first().and_then(|r| r.first()).cloned()
                    else {
                        engine.rollback(txn).unwrap();
                        return;
                    };
                    // Holding the lock, the row must still be queued: nobody else may have taken it.
                    let now = rows(
                        run_in(
                            &*engine,
                            txn,
                            &format!("SELECT status FROM q WHERE id = {id}"),
                        )
                        .unwrap(),
                    );
                    if now[0][0] != Value::Text("queued".to_owned()) {
                        stale.fetch_add(1, Ordering::Relaxed);
                    }
                    let updated = run_in(
                        &*engine,
                        txn,
                        &format!(
                            "UPDATE q SET status = 'done', attempts = attempts + 1 WHERE id = {id}"
                        ),
                    );
                    if updated.is_err() {
                        engine.rollback(txn).unwrap();
                        continue;
                    }
                    engine.commit(txn).unwrap();
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    let total = run(&*engine, "SELECT sum(attempts) FROM q");
    let attempts = match &total[0][0] {
        Value::Int(n) => u64::try_from(*n).unwrap(),
        other => panic!("{other:?}"),
    };
    (attempts - jobs, stale.load(Ordering::Relaxed))
}

#[test]
fn skip_locked_job_queue_processes_each_job_once() {
    let (duplicates, stale) = job_queue("FOR UPDATE SKIP LOCKED");
    assert_eq!(
        (duplicates, stale),
        (0, 0),
        "jobs processed twice, stale claims"
    );
}

#[test]
fn for_update_job_queue_processes_each_job_once() {
    let (duplicates, stale) = job_queue("FOR UPDATE");
    assert_eq!(
        (duplicates, stale),
        (0, 0),
        "jobs processed twice, stale claims"
    );
}
