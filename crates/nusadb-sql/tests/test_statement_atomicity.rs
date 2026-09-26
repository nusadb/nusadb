//! A statement that fails inside an explicit transaction leaves nothing behind: the rows,
//! index entries and trigger side effects it produced before the failure are undone, while the
//! transaction itself stays open for the client to continue or roll back.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session
        .execute(plan(logical))
        .unwrap_or_else(|e| panic!("`{sql}` should succeed: {e}"))
}

fn try_exec(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine))?;
    session.execute(plan(logical))
}

fn count(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> i64 {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    match rows.first().and_then(|r| r.first()) {
        Some(Value::Int(n)) => *n,
        other => panic!("expected an integer count, got {other:?}"),
    }
}

/// A multi-row INSERT refused on its second row: the BEFORE trigger already fired for both rows
/// and the first row was already written. Neither may survive the refusal, and the transaction
/// must still commit whatever else it did.
#[test]
fn a_refused_insert_leaves_no_rows_or_trigger_writes_behind() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    exec(
        &engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    exec(
        &engine,
        &mut session,
        "CREATE TABLE audit (tag TEXT, v INT)",
    );
    exec(
        &engine,
        &mut session,
        "CREATE TRIGGER bi BEFORE INSERT ON t FOR EACH ROW INSERT INTO audit VALUES ('ins', new.v)",
    );
    exec(&engine, &mut session, "INSERT INTO t VALUES (1, 10)");
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM audit"),
        1
    );

    exec(&engine, &mut session, "BEGIN");
    exec(
        &engine,
        &mut session,
        "INSERT INTO audit VALUES ('kept', 0)",
    );
    let err = try_exec(
        &engine,
        &mut session,
        "INSERT INTO t VALUES (2, 20), (1, 99)",
    )
    .expect_err("the duplicate key must refuse the statement");
    assert!(
        err.to_string().contains("duplicate") || err.to_string().contains("unique"),
        "{err}"
    );
    // Still inside the transaction: the failed statement's own writes are gone, the earlier
    // statement's write is not.
    assert_eq!(count(&engine, &mut session, "SELECT count(*) FROM t"), 1);
    assert_eq!(
        count(
            &engine,
            &mut session,
            "SELECT count(*) FROM audit WHERE tag = 'ins'"
        ),
        1
    );
    assert_eq!(
        count(
            &engine,
            &mut session,
            "SELECT count(*) FROM audit WHERE tag = 'kept'"
        ),
        1
    );
    assert!(session.in_transaction());
    exec(&engine, &mut session, "COMMIT");

    assert_eq!(count(&engine, &mut session, "SELECT count(*) FROM t"), 1);
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM audit"),
        2
    );
    // The unique index has no ghost entry for the undone row: inserting it again succeeds.
    exec(&engine, &mut session, "INSERT INTO t VALUES (2, 20)");
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM t WHERE id = 2"),
        1
    );
}

/// The client's own savepoints are untouched by the statement mark: a `ROLLBACK TO` after a
/// failed statement still lands on the client's savepoint, and a `RELEASE` still finds it.
#[test]
fn client_savepoints_survive_a_failed_statement() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    exec(&engine, &mut session, "CREATE TABLE t (id INT PRIMARY KEY)");
    exec(&engine, &mut session, "BEGIN");
    exec(&engine, &mut session, "INSERT INTO t VALUES (1)");
    exec(&engine, &mut session, "SAVEPOINT s");
    exec(&engine, &mut session, "INSERT INTO t VALUES (2)");
    try_exec(&engine, &mut session, "INSERT INTO t VALUES (3), (1)").expect_err("duplicate");
    assert_eq!(count(&engine, &mut session, "SELECT count(*) FROM t"), 2);
    exec(&engine, &mut session, "ROLLBACK TO SAVEPOINT s");
    assert_eq!(count(&engine, &mut session, "SELECT count(*) FROM t"), 1);
    exec(&engine, &mut session, "RELEASE SAVEPOINT s");
    exec(&engine, &mut session, "COMMIT");
    assert_eq!(count(&engine, &mut session, "SELECT count(*) FROM t"), 1);
}

/// A failed UPDATE that had already rewritten earlier rows is undone as a whole.
#[test]
fn a_refused_update_is_undone_as_a_whole() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    exec(
        &engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT CHECK (v < 100))",
    );
    exec(
        &engine,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 95), (3, 10)",
    );
    exec(&engine, &mut session, "BEGIN");
    // Row 2 trips the CHECK; row 1 was (or would have been) rewritten before it.
    try_exec(&engine, &mut session, "UPDATE t SET v = v + 10").expect_err("check violation");
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM t WHERE v = 10"),
        2
    );
    exec(&engine, &mut session, "COMMIT");
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM t WHERE v = 10"),
        2
    );
    assert_eq!(
        count(&engine, &mut session, "SELECT count(*) FROM t WHERE v = 20"),
        0
    );
}

/// An UPDATE that hits a lock held by another transaction on its second row had already
/// rewritten its first: that rewrite must be gone once the statement is refused, so a later
/// COMMIT cannot persist a half-applied update.
#[test]
fn an_update_refused_by_a_lock_conflict_does_not_keep_its_earlier_rows() {
    let engine = BtreeEngine::new();
    let mut holder = Session::new(&engine);
    let mut writer = Session::new(&engine);
    exec(
        &engine,
        &mut holder,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    exec(
        &engine,
        &mut holder,
        "INSERT INTO t VALUES (1, 10), (2, 10), (3, 10)",
    );

    exec(&engine, &mut holder, "BEGIN");
    exec(
        &engine,
        &mut holder,
        "SELECT id FROM t WHERE id = 2 FOR UPDATE",
    );

    exec(&engine, &mut writer, "BEGIN");
    let err = try_exec(&engine, &mut writer, "UPDATE t SET v = v + 1")
        .expect_err("row 2 is locked by the other transaction");
    assert!(!err.to_string().is_empty());
    // Row 1 was rewritten before the conflict on row 2; the refusal must have undone it.
    assert_eq!(
        count(&engine, &mut writer, "SELECT count(*) FROM t WHERE v = 11"),
        0
    );
    exec(&engine, &mut writer, "COMMIT");
    exec(&engine, &mut holder, "COMMIT");
    assert_eq!(
        count(&engine, &mut holder, "SELECT count(*) FROM t WHERE v = 10"),
        3
    );
}
