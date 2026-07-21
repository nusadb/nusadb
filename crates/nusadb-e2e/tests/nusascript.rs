//! End-to-end tests for NusaScript procedure bodies: `BEGIN ... END` blocks with
//! `DECLARE`/`SET` variables, `IF`/`ELSIF`/`ELSE`, `WHILE` loops, `RAISE`, and embedded SQL that
//! reads variables and `$n` parameters — driven through `parse → analyze → plan → execute` against
//! the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, Error> {
    let stmt = parse(sql)?;
    let logical = analyze(stmt, &EngineCatalog(engine))?;
    execute(plan(logical), engine)
}

fn rows(engine: &BtreeEngine, sql: &str) -> Vec<Vec<Value>> {
    match run(engine, sql) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

fn setup(engine: &BtreeEngine) {
    run(engine, "CREATE TABLE out (label TEXT, n INT)");
}

#[test]
fn declare_set_if_drives_branching() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE classify(v INT) AS $$
         BEGIN
           DECLARE label TEXT;
           IF $1 > 100 THEN
             SET label = 'big';
           ELSIF $1 > 10 THEN
             SET label = 'medium';
           ELSE
             SET label = 'small';
           END IF;
           INSERT INTO out VALUES (label, $1);
         END
         $$",
    );
    run(&engine, "CALL classify(150)");
    run(&engine, "CALL classify(50)");
    run(&engine, "CALL classify(5)");
    let mut got = rows(&engine, "SELECT label, n FROM out ORDER BY n");
    got.sort_by_key(|r| match &r[1] {
        Value::Int(n) => *n,
        _ => 0,
    });
    assert_eq!(
        got,
        vec![
            vec![Value::Text("small".to_owned()), Value::Int(5)],
            vec![Value::Text("medium".to_owned()), Value::Int(50)],
            vec![Value::Text("big".to_owned()), Value::Int(150)],
        ]
    );
}

#[test]
fn while_loop_accumulates_with_a_variable() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE countdown(start INT) AS $$
         BEGIN
           DECLARE i INT DEFAULT $1;
           WHILE i > 0 LOOP
             INSERT INTO out VALUES ('x', i);
             SET i = i - 1;
           END LOOP;
         END
         $$",
    );
    run(&engine, "CALL countdown(3)");
    let got = rows(&engine, "SELECT n FROM out ORDER BY n");
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
        ]
    );
}

#[test]
fn a_column_shadows_a_like_named_variable_in_embedded_sql() {
    // Deep-gate: a bare column whose name matches an in-scope variable must resolve to the COLUMN,
    // not be silently clobbered by the variable's value. Previously `INSERT INTO t SELECT n FROM t`
    // with a variable `n` inserted the variable's value for every row instead of each row's column n.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (n INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3)");
    run(
        &engine,
        "CREATE PROCEDURE dup_rows() AS $$
         BEGIN
           DECLARE n INT DEFAULT 99;
           INSERT INTO t SELECT n FROM t;
         END
         $$",
    );
    run(&engine, "CALL dup_rows()");
    // The column `t.n` wins, so the three rows are copied verbatim — not three copies of 99.
    assert_eq!(
        rows(&engine, "SELECT n FROM t ORDER BY n"),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(3)],
        ]
    );
}

#[test]
fn a_variable_binds_in_values_even_when_a_target_column_shares_its_name() {
    // The INSERT target's columns are NOT in scope for its VALUES source, so a variable named like a
    // target column still binds: `INSERT INTO t (n) VALUES (n)` inserts the variable, not column n.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (n INT NOT NULL)");
    run(
        &engine,
        "CREATE PROCEDURE put() AS $$
         BEGIN
           DECLARE n INT DEFAULT 42;
           INSERT INTO t (n) VALUES (n);
         END
         $$",
    );
    run(&engine, "CALL put()");
    assert_eq!(rows(&engine, "SELECT n FROM t"), vec![vec![Value::Int(42)]]);
}

#[test]
fn raise_aborts_and_rolls_back() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE guard(v INT) AS $$
         BEGIN
           INSERT INTO out VALUES ('seen', $1);
           IF $1 < 0 THEN
             RAISE 'negative not allowed';
           END IF;
         END
         $$",
    );
    // A successful call runs to completion.
    run(&engine, "CALL guard(5)");
    assert_eq!(
        rows(&engine, "SELECT n FROM out"),
        vec![vec![Value::Int(5)]]
    );

    // RAISE aborts the call with the message, and the earlier INSERT in the same call rolls back.
    match run_try(&engine, "CALL guard(-1)") {
        Err(Error::Raised(msg)) => assert!(msg.contains("negative")),
        other => panic!("expected Raised, got {other:?}"),
    }
    assert_eq!(
        rows(&engine, "SELECT n FROM out ORDER BY n"),
        vec![vec![Value::Int(5)]],
        "the RAISEd call's INSERT must roll back"
    );
}

#[test]
fn for_loop_iterates_an_integer_range() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE fill(lo INT, hi INT) AS $$
         BEGIN
           FOR k IN $1 TO $2 LOOP
             INSERT INTO out VALUES ('k', k);
           END LOOP;
         END
         $$",
    );
    run(&engine, "CALL fill(2, 5)");
    assert_eq!(
        rows(&engine, "SELECT n FROM out ORDER BY n"),
        vec![
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
        ]
    );
    // An inverted range runs zero iterations.
    run(&engine, "DELETE FROM out");
    run(&engine, "CALL fill(5, 2)");
    assert!(rows(&engine, "SELECT n FROM out").is_empty());
}

#[test]
fn exception_handler_rolls_back_body_and_recovers() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // The body inserts a row, then hits a division-by-zero that aborts it; the EXCEPTION handler
    // rolls the body's writes back (to a savepoint) and inserts a recovery marker instead.
    run(
        &engine,
        "CREATE PROCEDURE safe(v INT) AS $$
         BEGIN
           INSERT INTO out VALUES ('attempt', $1);
           INSERT INTO out VALUES ('boom', $1 / 0);
         EXCEPTION WHEN OTHERS THEN
           INSERT INTO out VALUES ('handled', $1);
         END
         $$",
    );
    run(&engine, "CALL safe(7)");
    // The 'attempt' insert was rolled back; only the handler's row remains, and the call committed.
    assert_eq!(
        rows(&engine, "SELECT label, n FROM out"),
        vec![vec![Value::Text("handled".to_owned()), Value::Int(7),]]
    );
}

#[test]
fn exception_handler_catches_a_raise() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE attempt(v INT) AS $$
         BEGIN
           IF $1 < 0 THEN
             RAISE 'bad input';
           END IF;
           INSERT INTO out VALUES ('ok', $1);
         EXCEPTION WHEN OTHERS THEN
           INSERT INTO out VALUES ('recovered', $1);
         END
         $$",
    );
    // A RAISE is caught by the handler; the call succeeds (does not error out).
    run(&engine, "CALL attempt(-1)");
    assert_eq!(
        rows(&engine, "SELECT label, n FROM out"),
        vec![vec![Value::Text("recovered".to_owned()), Value::Int(-1),]]
    );
    // The non-raising path runs the body normally.
    run(&engine, "DELETE FROM out");
    run(&engine, "CALL attempt(3)");
    assert_eq!(
        rows(&engine, "SELECT label, n FROM out"),
        vec![vec![Value::Text("ok".to_owned()), Value::Int(3),]]
    );
}

#[test]
fn malformed_block_is_rejected_at_create() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // Missing END IF.
    assert!(
        run_try(
            &engine,
            "CREATE PROCEDURE bad() AS $$ BEGIN IF 1 > 0 THEN INSERT INTO out VALUES ('x', 1); END $$",
        )
        .is_err()
    );
}
