//! `PREPARE` / `EXECUTE` / `DEALLOCATE` end-to-end. A prepared statement is stored in the
//! session, then re-analyzed and run on `EXECUTE` with its arguments bound to `$1..$n`. Covers
//! parameterized SELECT and INSERT, argument-arity and missing-name errors, DEALLOCATE (single and
//! ALL), re-prepare, and that only a runnable query can be prepared.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Row, Session, analyze, parse, plan};

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
    session.execute(plan(logical)).unwrap()
}

fn try_exec(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine))?;
    session.execute(plan(logical))
}

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn prepare_execute_select_and_insert() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..5 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'x')"),
        );
    }

    // PREPARE a parameterized SELECT, then EXECUTE it with different arguments.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "PREPARE sel AS SELECT a, b FROM t WHERE a = $1"
        ),
        ExecutionResult::Prepared
    ));
    assert_eq!(
        rows(engine, &mut session, "EXECUTE sel (3)"),
        vec![vec![Value::Int(3), Value::Text("x".to_owned())]]
    );
    // The prepared statement persists in the session and can run again with new args.
    assert_eq!(
        rows(engine, &mut session, "EXECUTE sel (0)"),
        vec![vec![Value::Int(0), Value::Text("x".to_owned())]]
    );

    // PREPARE a parameterized INSERT; EXECUTE actually writes.
    exec(
        engine,
        &mut session,
        "PREPARE ins AS INSERT INTO t VALUES ($1, $2)",
    );
    assert!(matches!(
        exec(engine, &mut session, "EXECUTE ins (99, 'z')"),
        ExecutionResult::Inserted(1)
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT a, b FROM t WHERE a = 99"),
        vec![vec![Value::Int(99), Value::Text("z".to_owned())]]
    );

    // A no-FROM constant SELECT works too (negative + text args).
    exec(engine, &mut session, "PREPARE calc AS SELECT $1, $2");
    assert_eq!(
        rows(engine, &mut session, "EXECUTE calc (-7, 'hi')"),
        vec![vec![Value::Int(-7), Value::Text("hi".to_owned())]]
    );
}

#[test]
fn execute_errors_and_deallocate() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT)");
    exec(
        engine,
        &mut session,
        "PREPARE p AS SELECT a FROM t WHERE a = $1",
    );

    // Wrong argument arity is an error.
    assert!(matches!(
        try_exec(engine, &mut session, "EXECUTE p (1, 2)"),
        Err(Error::Unsupported(_))
    ));
    // Executing an unknown prepared statement is an error.
    assert!(matches!(
        try_exec(engine, &mut session, "EXECUTE ghost (1)"),
        Err(Error::Unsupported(_))
    ));

    // DEALLOCATE removes it; afterwards EXECUTE fails.
    assert!(matches!(
        exec(engine, &mut session, "DEALLOCATE p"),
        ExecutionResult::Deallocated
    ));
    assert!(try_exec(engine, &mut session, "EXECUTE p (1)").is_err());
    // DEALLOCATE of a missing name errors; DEALLOCATE ALL is always fine.
    assert!(try_exec(engine, &mut session, "DEALLOCATE p").is_err());
    assert!(matches!(
        exec(engine, &mut session, "DEALLOCATE ALL"),
        ExecutionResult::Deallocated
    ));

    // Re-preparing a name overwrites it.
    exec(
        engine,
        &mut session,
        "PREPARE q AS SELECT a FROM t WHERE a = $1",
    );
    exec(
        engine,
        &mut session,
        "PREPARE q AS SELECT a FROM t WHERE a > $1",
    );
    // (Both are valid; the second definition is the one that runs — a smoke check that it executes.)
    let _ = exec(engine, &mut session, "EXECUTE q (0)");
}

#[test]
fn only_a_query_can_be_prepared() {
    // PREPARE of a non-query (transaction control, DDL, nested EXECUTE) is rejected at parse.
    assert!(parse("PREPARE p AS CREATE TABLE z (a INT)").is_err());
    assert!(parse("PREPARE p AS DEALLOCATE q").is_err());
}
