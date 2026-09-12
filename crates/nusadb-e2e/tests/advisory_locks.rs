//! End-to-end tests for session-scoped advisory locks across two independent sessions.
//!
//! Advisory locks are process-global and owned by a session, so their defining behavior —
//! exclusion between sessions and release when a session ends — needs two live `Session`s over one
//! engine, which a single-connection SQLLogicTest cannot express. Single-session behavior
//! (re-entrancy, unlock, two-key form, refusals) is covered by `slt/p12_txn/advisory_locks.slt`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        Ok(Vec::new())
    }
}

/// Run a one-row scalar query and return its single `bool` value.
fn ask_bool(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> bool {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    match session.execute(plan(logical)).expect("execute") {
        ExecutionResult::Rows { rows, .. } => match rows.first().and_then(|r| r.first()) {
            Some(Value::Bool(b)) => *b,
            other => panic!("expected one bool, got {other:?}"),
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn advisory_lock_excludes_other_sessions_and_is_reentrant() {
    let engine = BtreeEngine::new();
    let mut a = Session::new(&engine);
    let mut b = Session::new(&engine);

    // A takes key 42; B is then refused.
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_try_advisory_lock(10042)"
    ));
    assert!(!ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(10042)"
    ));

    // A re-enters (hold count 2); B still refused after one unlock.
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_try_advisory_lock(10042)"
    ));
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_advisory_unlock(10042)"
    ));
    assert!(!ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(10042)"
    ));

    // Second unlock drops the last hold; B can now take it.
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_advisory_unlock(10042)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(10042)"
    ));

    // Unlocking a key this session never held reports false.
    assert!(!ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_advisory_unlock(10042)"
    ));
}

#[test]
fn advisory_locks_release_when_the_session_ends() {
    let engine = BtreeEngine::new();
    let mut b = Session::new(&engine);
    {
        let mut a = Session::new(&engine);
        assert!(ask_bool(
            &mut a,
            &engine,
            "SELECT nusadb_try_advisory_lock(20007)"
        ));
        assert!(ask_bool(
            &mut a,
            &engine,
            "SELECT nusadb_try_advisory_lock(20007)"
        )); // held twice
        assert!(!ask_bool(
            &mut b,
            &engine,
            "SELECT nusadb_try_advisory_lock(20007)"
        ));
        // A goes out of scope here without unlocking — every hold must be released.
    }
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(20007)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_advisory_unlock_all()"
    ));
}

#[test]
fn unlock_all_releases_every_key_held_by_the_session() {
    let engine = BtreeEngine::new();
    let mut a = Session::new(&engine);
    let mut b = Session::new(&engine);

    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_try_advisory_lock(30001)"
    ));
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_try_advisory_lock(30002)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_advisory_unlock_all()"
    )); // B holds none: no-op, true
    assert!(!ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(30001)"
    ));

    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_advisory_unlock_all()"
    ));
    // Both of A's keys are free now.
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(30001)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(30002)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_advisory_unlock_all()"
    ));
}

#[test]
fn two_integer_key_is_distinct_from_the_matching_single_key() {
    let engine = BtreeEngine::new();
    let mut a = Session::new(&engine);
    let mut b = Session::new(&engine);

    // The (hi, lo) pair combines to a bigint that is not the small single-key value, so taking the
    // pair does not block the single key 1 in another session.
    assert!(ask_bool(
        &mut a,
        &engine,
        "SELECT nusadb_try_advisory_lock(40001, 2)"
    ));
    assert!(ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(40001)"
    ));
    // But the same pair collides with itself.
    assert!(!ask_bool(
        &mut b,
        &engine,
        "SELECT nusadb_try_advisory_lock(40001, 2)"
    ));

    ask_bool(&mut a, &engine, "SELECT nusadb_advisory_unlock_all()");
    ask_bool(&mut b, &engine, "SELECT nusadb_advisory_unlock_all()");
}
