//! End-to-end tests for stored procedures: `CREATE`/`DROP PROCEDURE`, `CALL` with
//! positional `$n` arguments, multi-statement bodies, `OR REPLACE`, arg-count + existence checks, and
//! whole-call atomicity — driven through `parse → analyze → plan → execute` against the production
//! `BtreeEngine`.
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

fn select_ints(engine: &BtreeEngine, sql: &str) -> Vec<Vec<i64>> {
    match run(engine, sql) {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|r| {
                r.iter()
                    .map(|v| match v {
                        Value::Int(n) => *n,
                        other => panic!("expected int, got {other:?}"),
                    })
                    .collect()
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn setup(engine: &BtreeEngine) {
    run(engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    run(engine, "CREATE TABLE log (op INT NOT NULL)");
}

#[test]
fn create_call_runs_multi_statement_body_with_params() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE add_pair(a INT, b INT) AS $$
           INSERT INTO t VALUES ($1, $2);
           INSERT INTO log VALUES ($1)
         $$",
    );
    assert!(matches!(
        run(&engine, "CALL add_pair(5, 10)"),
        ExecutionResult::ProcedureCalled
    ));
    assert_eq!(
        select_ints(&engine, "SELECT id, val FROM t"),
        vec![vec![5, 10]]
    );
    assert_eq!(select_ints(&engine, "SELECT op FROM log"), vec![vec![5]]);

    // A second call accumulates (the body re-binds fresh arguments each time).
    run(&engine, "CALL add_pair(7, 8)");
    assert_eq!(
        select_ints(&engine, "SELECT id, val FROM t ORDER BY id"),
        vec![vec![5, 10], vec![7, 8]]
    );
}

#[test]
fn or_replace_and_drop() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE p(a INT) AS $$ INSERT INTO log VALUES ($1) $$",
    );
    run(
        &engine,
        "CREATE OR REPLACE PROCEDURE p(a INT) AS $$ INSERT INTO log VALUES ($1 + 100) $$",
    );
    run(&engine, "CALL p(1)");
    // Only the replacement body ran (1 + 100).
    assert_eq!(select_ints(&engine, "SELECT op FROM log"), vec![vec![101]]);

    assert!(matches!(
        run(&engine, "DROP PROCEDURE p"),
        ExecutionResult::ProcedureDropped
    ));
    assert!(matches!(
        run_try(&engine, "CALL p(1)"),
        Err(Error::ProcedureNotFound { .. })
    ));
    assert!(matches!(
        run(&engine, "DROP PROCEDURE IF EXISTS p"),
        ExecutionResult::ProcedureDropped
    ));
}

#[test]
fn duplicate_and_arg_count_and_unknown_are_rejected() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE p(a INT) AS $$ INSERT INTO log VALUES ($1) $$",
    );
    assert!(matches!(
        run_try(
            &engine,
            "CREATE PROCEDURE p(a INT) AS $$ INSERT INTO log VALUES ($1) $$",
        ),
        Err(Error::ProcedureExists { .. })
    ));
    // Too few / too many arguments are rejected against the declared arity.
    assert!(matches!(
        run_try(&engine, "CALL p()"),
        Err(Error::ProcedureArgCount {
            expected: 1,
            found: 0,
            ..
        })
    ));
    assert!(matches!(
        run_try(&engine, "CALL p(1, 2)"),
        Err(Error::ProcedureArgCount {
            expected: 1,
            found: 2,
            ..
        })
    ));
    assert!(matches!(
        run_try(&engine, "CALL nope(1)"),
        Err(Error::ProcedureNotFound { .. })
    ));
    assert!(matches!(
        run_try(&engine, "DROP PROCEDURE nope"),
        Err(Error::ProcedureNotFound { .. })
    ));
}

#[test]
fn body_statement_failure_rolls_back_whole_call() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // The second statement writes NULL into a NOT NULL column → fails; the first statement's insert
    // must be rolled back with it (the CALL runs in one transaction).
    run(
        &engine,
        "CREATE PROCEDURE half(a INT) AS $$
           INSERT INTO t VALUES ($1, $1);
           INSERT INTO log VALUES (NULL)
         $$",
    );
    assert!(run_try(&engine, "CALL half(1)").is_err());
    assert!(
        select_ints(&engine, "SELECT id FROM t").is_empty(),
        "the first insert must roll back with the failed call"
    );
}

#[test]
fn paramless_procedure() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO log VALUES (1), (2), (3)");
    run(
        &engine,
        "CREATE PROCEDURE clear_log() AS $$ DELETE FROM log $$",
    );
    run(&engine, "CALL clear_log()");
    assert!(select_ints(&engine, "SELECT op FROM log").is_empty());
}

#[test]
fn body_referencing_undeclared_parameter_is_rejected_at_create() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // The body uses $2 but only one parameter is declared.
    assert!(matches!(
        run_try(
            &engine,
            "CREATE PROCEDURE bad(a INT) AS $$ INSERT INTO log VALUES ($2) $$",
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn procedures_compose_via_nested_call() {
    // A procedure body may CALL another procedure, forwarding its own $n arguments.
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE PROCEDURE log_one(v INT) AS $$ INSERT INTO log VALUES ($1) $$",
    );
    run(
        &engine,
        "CREATE PROCEDURE log_both(a INT, b INT) AS $$
           CALL log_one($1);
           CALL log_one($2)
         $$",
    );
    run(&engine, "CALL log_both(10, 20)");
    assert_eq!(
        select_ints(&engine, "SELECT op FROM log ORDER BY op"),
        vec![vec![10], vec![20]]
    );
}

#[test]
fn unbounded_recursive_call_hits_depth_limit() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // A procedure that calls itself with no base case is aborted by the call-depth guard, not a
    // stack overflow, and the whole call rolls back.
    run(
        &engine,
        "CREATE PROCEDURE spin() AS $$ INSERT INTO log VALUES (1); CALL spin() $$",
    );
    assert!(matches!(
        run_try(&engine, "CALL spin()"),
        Err(Error::ProcedureRecursionLimit { .. })
    ));
    assert!(
        select_ints(&engine, "SELECT op FROM log").is_empty(),
        "the aborted recursive call rolls back its inserts"
    );
}

#[test]
fn out_parameters_are_returned_from_call() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // `$1` binds the single IN parameter; the OUT parameter `result` is set in the body and returned.
    run(
        &engine,
        "CREATE PROCEDURE compute(a INT, OUT result INT, OUT label TEXT) AS $$
         BEGIN
           SET result = $1 * 10;
           IF $1 > 0 THEN SET label = 'pos'; ELSE SET label = 'nonpos'; END IF;
         END
         $$",
    );
    // CALL supplies arguments for the IN parameters only; OUT values come back as a one-row result.
    match run(&engine, "CALL compute(4)") {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["result".to_owned(), "label".to_owned()]);
            assert_eq!(
                rows,
                vec![vec![Value::Int(40), Value::Text("pos".to_owned())]]
            );
        },
        other => panic!("expected OUT rows, got {other:?}"),
    }
    // Arity is checked against the IN parameter count (one), not the total.
    assert!(matches!(
        run_try(&engine, "CALL compute(1, 2)"),
        Err(Error::ProcedureArgCount {
            expected: 1,
            found: 2,
            ..
        })
    ));
    // A procedure with no OUT parameters still reports the plain CALL result.
    run(
        &engine,
        "CREATE PROCEDURE noret(a INT) AS $$ INSERT INTO log VALUES ($1) $$",
    );
    assert!(matches!(
        run(&engine, "CALL noret(9)"),
        ExecutionResult::ProcedureCalled
    ));
}
