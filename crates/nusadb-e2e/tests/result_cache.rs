//! End-to-end tests for the SQL result cache: a persistent `Session` caches auto-commit
//! `SELECT` results keyed on `(user, plan)` + the engine's data-version, and the cache is correct —
//! a committed write invalidates it, an explicit transaction bypasses it (seeing its own writes), and
//! volatile queries are never cached. Driven through `parse → analyze → plan → Session::execute`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, Session, analyze, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

/// Execute one statement through a persistent session (so its result cache survives across calls).
fn exec(session: &mut Session<'_>, engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    session.execute(plan(logical)).expect("execute")
}

fn ints(result: ExecutionResult) -> Vec<i64> {
    let mut out: Vec<i64> = match result {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|r| match r.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    };
    out.sort_unstable();
    out
}

#[test]
fn a_committed_write_invalidates_the_cache() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");

    // First SELECT populates the cache.
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1]);
    // An identical SELECT (same data-version) returns the same result (served from cache).
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1]);

    // A committed write bumps the data-version; the next identical SELECT must reflect it — proving
    // the cache did NOT serve a stale result.
    exec(&mut s, &engine, "INSERT INTO t VALUES (2)");
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1, 2]);

    // And a DELETE invalidates it too.
    exec(&mut s, &engine, "DELETE FROM t WHERE n = 1");
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![2]);
}

#[test]
fn an_explicit_transaction_bypasses_the_cache_and_sees_its_own_writes() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");
    // Prime the cache in auto-commit.
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1]);

    // Inside a transaction, a SELECT must see the transaction's own uncommitted write — the cache is
    // bypassed entirely while a transaction is open.
    exec(&mut s, &engine, "BEGIN");
    exec(&mut s, &engine, "INSERT INTO t VALUES (2)");
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1, 2]);
    exec(&mut s, &engine, "COMMIT");

    // After commit, the auto-commit SELECT reflects the committed state.
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![1, 2]);
}

#[test]
fn cacheable_selects_populate_but_volatile_ones_do_not() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");
    assert_eq!(s.result_cache_len(), 0);

    // A plain SELECT is cached.
    exec(&mut s, &engine, "SELECT n FROM t");
    assert_eq!(s.result_cache_len(), 1, "a plain SELECT should be cached");
    // Re-running the identical query is a cache hit — it does not add a new entry.
    exec(&mut s, &engine, "SELECT n FROM t");
    assert_eq!(
        s.result_cache_len(),
        1,
        "an identical SELECT hits the cache"
    );

    // A volatile query (RANDOM) must never be cached.
    exec(&mut s, &engine, "SELECT random() FROM t");
    assert_eq!(
        s.result_cache_len(),
        1,
        "a volatile SELECT must not be cached"
    );
    // NOW() is volatile too.
    exec(&mut s, &engine, "SELECT now() FROM t");
    assert_eq!(s.result_cache_len(), 1, "NOW() must not be cached");
    // AGE(ts) is relative to the current date, so its result must not be cached across days
    // (deep-gate) — even with a constant argument.
    exec(
        &mut s,
        &engine,
        "SELECT age(TIMESTAMP '2020-01-01 00:00:00') FROM t",
    );
    assert_eq!(s.result_cache_len(), 1, "AGE() must not be cached");
}

#[test]
fn results_for_different_users_do_not_cross() {
    // The cache key includes the session user, so two users never share a cached row set even for an
    // identical query (a row-level-security safety property).
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (7)");

    s.set_current_user("alice");
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![7]);
    // Switching users still returns the correct rows (a fresh cache key, not alice's entry).
    s.set_current_user("bob");
    assert_eq!(ints(exec(&mut s, &engine, "SELECT n FROM t")), vec![7]);
}
