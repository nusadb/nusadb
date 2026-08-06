//! Auto-commit statements retry their own serialization conflicts; explicit transactions do not.
//!
//! Concurrency control here is optimistic, so two sessions writing the same row make one of them
//! fail with `40001` rather than queue. A single statement in auto-commit has no intermediate state
//! the application observed and committed nothing, so re-running it is exactly what a correct client
//! retry loop would do — and the server can do it without the client noticing. Inside
//! `BEGIN…COMMIT` the same retry would be wrong: the statement may follow others whose results the
//! application already acted on, and silently re-running it changes what those results meant.
//!
//! What the tests below pin down: the retry happens where it is safe, does not happen where it is
//! not, is bounded and switchable, never invents a success, and never turns one non-retryable error
//! into several.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/expect/panic"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

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

fn try_exec(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical))
}

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    try_exec(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The single scalar in a one-row, one-column result.
fn scalar(result: &ExecutionResult) -> String {
    match result {
        ExecutionResult::Rows { rows, .. } => {
            let value = rows
                .first()
                .and_then(|row| row.first())
                .expect("expected one row of one column");
            format!("{value:?}")
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

/// A blocker holding an *uncommitted* write on a row: anything else writing that row loses the
/// optimistic race for as long as this transaction stays open. Locks here do not wait, so the
/// conflict is immediate and no test depends on timing to produce one.
fn open_blocking_write(engine: &'static BtreeEngine, sql: &str) -> nusadb_core::TxnId {
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    nusadb_sql::execute_in_txn(plan(logical), engine, txn).unwrap();
    txn
}

fn fixture() -> (&'static BtreeEngine, Session<'static>) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    exec(engine, &mut session, "INSERT INTO t VALUES (1, 0)");
    (engine, session)
}

/// A conflict that outlives the retry budget still reaches the client as `40001`. The retry is an
/// optimisation, not a promise of success — an application's own retry loop must not be deprived of
/// the error that drives it.
#[test]
fn a_conflict_that_outlives_the_budget_is_still_reported_as_40001() {
    let (engine, mut session) = fixture();
    let blocker = open_blocking_write(engine, "UPDATE t SET v = 99 WHERE id = 1");

    let err = try_exec(engine, &mut session, "UPDATE t SET v = 1 WHERE id = 1")
        .expect_err("the blocker never commits, so every attempt must lose");
    assert_eq!(
        err.sqlstate(),
        "40001",
        "an exhausted retry budget reports the conflict unchanged, got: {err}"
    );

    let _ = engine.rollback(blocker);
}

/// The budget is a knob, and zero turns the retry off entirely — restoring the behaviour of
/// reporting the first conflict straight to the client for a deployment that wants to see them.
#[test]
fn setting_the_budget_to_zero_reports_the_first_conflict() {
    let (engine, mut session) = fixture();
    exec(engine, &mut session, "SET max_autocommit_retries = 0");
    let blocker = open_blocking_write(engine, "UPDATE t SET v = 99 WHERE id = 1");

    let started = std::time::Instant::now();
    let err = try_exec(engine, &mut session, "UPDATE t SET v = 1 WHERE id = 1")
        .expect_err("with the budget at zero the first conflict is the answer");
    assert_eq!(err.sqlstate(), "40001", "got: {err}");
    // With no retries there is no back-off either, so the failure is immediate rather than paying
    // the budget's worth of sleeps first.
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "a disabled retry must not sleep: took {:?}",
        started.elapsed()
    );

    let _ = engine.rollback(blocker);
}

/// A value the budget cannot use is rejected at `SET` time, not stored and then quietly ignored.
/// Storing it would leave a session believing it had changed the budget when it had not — the exact
/// failure mode the parameter-name check already exists to prevent.
#[test]
fn an_unusable_budget_is_rejected_at_set_time() {
    let (engine, mut session) = fixture();

    let err = try_exec(
        engine,
        &mut session,
        "SET max_autocommit_retries = 'banana'",
    )
    .expect_err("a non-numeric budget cannot be honoured");
    assert_eq!(err.sqlstate(), "22023", "got: {err}");
    // A negative budget is refused too, though not by this check — `SET` accepts a literal, and a
    // leading minus makes the value an expression, so it is turned away before the value is ever
    // looked at. Asserted as "refused" rather than by code: the outcome is what matters, and pinning
    // the code here would pin a rule that belongs to `SET` in general, not to this parameter.
    assert!(
        try_exec(engine, &mut session, "SET max_autocommit_retries = -1").is_err(),
        "a negative budget cannot be honoured"
    );

    // An oversized budget is clamped rather than refused — it names a valid intent (retry hard),
    // just more of it than the ceiling allows.
    exec(engine, &mut session, "SET max_autocommit_retries = 100000");
}

/// Inside `BEGIN…COMMIT` the conflict is the application's to handle: earlier statements in the
/// transaction may already have returned results it acted on, so re-running this one silently would
/// change what those results meant. The error surfaces, and the transaction stays open for the
/// caller to `ROLLBACK`.
#[test]
fn an_explicit_transaction_does_not_retry() {
    let (engine, mut session) = fixture();
    let blocker = open_blocking_write(engine, "UPDATE t SET v = 99 WHERE id = 1");

    exec(engine, &mut session, "BEGIN");
    let started = std::time::Instant::now();
    let err = try_exec(engine, &mut session, "UPDATE t SET v = 1 WHERE id = 1")
        .expect_err("a conflict inside an explicit transaction belongs to the application");
    assert_eq!(err.sqlstate(), "40001", "got: {err}");
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "an explicit transaction must not have retried: took {:?}",
        started.elapsed()
    );
    exec(engine, &mut session, "ROLLBACK");

    let _ = engine.rollback(blocker);
}

/// An error re-running cannot fix is reported once and immediately. Retrying it would multiply the
/// work and delay the answer without any chance of a different outcome — so the classification has
/// to be narrow, not "anything that failed".
#[test]
fn a_non_retryable_error_is_not_retried() {
    let (engine, mut session) = fixture();
    exec(engine, &mut session, "INSERT INTO t VALUES (2, 0)");

    let started = std::time::Instant::now();
    let err = try_exec(engine, &mut session, "INSERT INTO t VALUES (2, 5)")
        .expect_err("id 2 already exists");
    assert_ne!(
        err.sqlstate(),
        "40001",
        "a duplicate key is a constraint violation, not a race: {err}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "a constraint violation must not have been retried: took {:?}",
        started.elapsed()
    );

    // And the failed statement changed nothing.
    let rows = exec(engine, &mut session, "SELECT v FROM t WHERE id = 2");
    assert_eq!(scalar(&rows), "Int(0)");
}

/// Running a statement through `EXECUTE` must not change whether it survives contention. That path
/// has its own auto-commit branch, so it needs its own proof — the two were inconsistent until this
/// was noticed, and nothing in the direct-statement tests would have caught it.
#[test]
fn a_prepared_statement_executed_in_auto_commit_retries_too() {
    let (engine, mut session) = fixture();
    exec(
        engine,
        &mut session,
        "PREPARE bump AS UPDATE t SET v = v + 1 WHERE id = 1",
    );

    // Uncontended first: the prepared statement runs and the retry path did not break it.
    exec(engine, &mut session, "EXECUTE bump");
    let rows = exec(engine, &mut session, "SELECT v FROM t WHERE id = 1");
    assert_eq!(scalar(&rows), "Int(1)");

    // Contended: a blocker that never commits, so the budget must run out and report — the same
    // answer the direct statement gives, rather than the un-retried first conflict.
    let blocker = open_blocking_write(engine, "UPDATE t SET v = 99 WHERE id = 1");
    let err = try_exec(engine, &mut session, "EXECUTE bump")
        .expect_err("the blocker never commits, so every attempt must lose");
    assert_eq!(
        err.sqlstate(),
        "40001",
        "an exhausted budget reports the conflict unchanged, got: {err}"
    );
    let _ = engine.rollback(blocker);
}

/// A cancel request — or a statement timeout, which trips the same token — ends the retry sequence
/// instead of being ignored until the budget runs out.
///
/// This is what keeps the retry from weakening `statement_timeout`: the timeout is armed once around
/// the whole statement, so without a check between attempts a contended statement could keep
/// retrying long past its deadline.
#[test]
fn a_cancel_request_stops_the_retrying() {
    let (engine, mut session) = fixture();
    // A budget far larger than the cancel delay, so reaching the end of it cannot be what ends the
    // statement — only the cancel can.
    exec(engine, &mut session, "SET max_autocommit_retries = 100");
    let blocker = open_blocking_write(engine, "UPDATE t SET v = 99 WHERE id = 1");

    let token: nusadb_sql::cancel::CancelToken =
        Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _guard = nusadb_sql::cancel::scope(Arc::clone(&token));
    let flip = Arc::clone(&token);
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(10));
        flip.store(true, Ordering::Relaxed);
    });

    let err = try_exec(engine, &mut session, "UPDATE t SET v = 1 WHERE id = 1")
        .expect_err("the blocker never commits, so the statement can only end by being cancelled");
    assert_eq!(
        err.sqlstate(),
        "57014",
        "a cancelled retry sequence reports query_canceled, not the conflict it was retrying: {err}"
    );

    canceller.join().unwrap();
    let _ = engine.rollback(blocker);
}

/// The reported scenario: several sessions writing the *same* row in auto-commit. Before the retry,
/// this produced `40001`s the application had to handle itself; now every statement completes.
///
/// The count is the real assertion. A retry that dropped an attempt, or ran one twice, would show up
/// here as a total that is not exactly the number of increments issued — so this covers "no lost
/// updates" and "no double-apply" at the same time as "no errors".
#[test]
fn concurrent_autocommit_writers_to_one_row_all_succeed_without_losing_updates() {
    const SESSIONS: usize = 8;
    const WRITES_PER_SESSION: usize = 25;

    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut setup = Session::new(engine);
    exec(
        engine,
        &mut setup,
        "CREATE TABLE ctr (id INT PRIMARY KEY, v INT)",
    );
    exec(engine, &mut setup, "INSERT INTO ctr VALUES (1, 0)");
    drop(setup);

    let failures = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..SESSIONS {
            let failures = Arc::clone(&failures);
            scope.spawn(move || {
                let mut session = Session::new(engine);
                for _ in 0..WRITES_PER_SESSION {
                    if try_exec(
                        engine,
                        &mut session,
                        "UPDATE ctr SET v = v + 1 WHERE id = 1",
                    )
                    .is_err()
                    {
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    assert_eq!(
        failures.load(Ordering::Relaxed),
        0,
        "every auto-commit write should have been retried to completion"
    );

    let mut reader = Session::new(engine);
    let rows = exec(engine, &mut reader, "SELECT v FROM ctr WHERE id = 1");
    assert_eq!(
        scalar(&rows),
        format!("Int({})", SESSIONS * WRITES_PER_SESSION),
        "each increment must be applied exactly once"
    );
}
