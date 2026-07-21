//! End-to-end tests for SQL scalar functions: `CREATE`/`DROP FUNCTION` and inlining a call
//! in queries (projection, WHERE, nested, expression arguments), arity + existence checks — driven
//! through `parse → analyze → plan → execute` against the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, FunctionDef, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a BtreeEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<FunctionDef>, Error> {
        // Functions are stored in a committed catalog table; read it in a throwaway transaction.
        let txn = self.0.begin(IsolationLevel::default())?;
        let result = nusadb_sql::lookup_function_definition(self.0, txn, name);
        let _ = self.0.commit(txn);
        result
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

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn sql_function_inlines_in_projection_and_where() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, n INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5), (2, 10)");
    run(
        &engine,
        "CREATE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 * 2 $$",
    );

    // Projection: the call inlines to `n * 2`.
    let out = rows(run(&engine, "SELECT id, dbl(n) FROM t ORDER BY id"));
    assert_eq!(
        out,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );

    // WHERE: the inlined expression composes with comparison.
    let out = rows(run(
        &engine,
        "SELECT id FROM t WHERE dbl(n) > 12 ORDER BY id",
    ));
    assert_eq!(out, vec![vec![Value::Int(2)]]);
}

#[test]
fn sql_function_body_can_reference_parameters_by_name() {
    // A body may reference a declared parameter by its name (not only `$1`). Before the
    // fix `f(5)` failed with `column not found: x`; now the name binds to the argument like `$1` does.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE FUNCTION inc(x INT) RETURNS INT AS $$ SELECT x + 1 $$",
    );
    let out = rows(run(&engine, "SELECT inc(5)"));
    assert_eq!(out, vec![vec![Value::Int(6)]]);

    // Multiple named parameters, in and out of declaration order, mixed with literals.
    run(
        &engine,
        "CREATE FUNCTION wsum(a INT, b INT) RETURNS INT AS $$ SELECT a * 10 + b $$",
    );
    let out = rows(run(&engine, "SELECT wsum(3, 7)"));
    assert_eq!(out, vec![vec![Value::Int(37)]]);

    // The positional `$n` form keeps working, and the two forms coexist across calls.
    run(
        &engine,
        "CREATE FUNCTION pos(x INT) RETURNS INT AS $$ SELECT $1 * 2 $$",
    );
    let out = rows(run(&engine, "SELECT pos(8)"));
    assert_eq!(out, vec![vec![Value::Int(16)]]);

    // A named parameter binds an expression argument too (composes like `$1`).
    run(&engine, "CREATE TABLE t (id INT NOT NULL, n INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5)");
    let out = rows(run(&engine, "SELECT inc(n + 1) FROM t WHERE id = 1"));
    assert_eq!(out, vec![vec![Value::Int(7)]]);
}

#[test]
fn sql_functions_nest_and_take_expression_arguments() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, n INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5)");
    run(
        &engine,
        "CREATE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 * 2 $$",
    );
    run(
        &engine,
        "CREATE FUNCTION inc(x INT) RETURNS INT AS $$ SELECT $1 + 1 $$",
    );
    // dbl(inc(n)) with n=5 → dbl(6) → 12; the argument is itself an expression / function call.
    let out = rows(run(&engine, "SELECT dbl(inc(n + 0)) FROM t WHERE id = 1"));
    assert_eq!(out, vec![vec![Value::Int(12)]]);
}

#[test]
fn function_arity_drop_and_unknown_are_checked() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (n INT)");
    run(&engine, "INSERT INTO t VALUES (3)");
    run(
        &engine,
        "CREATE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 * 2 $$",
    );
    // Wrong arity.
    assert!(matches!(
        run_try(&engine, "SELECT dbl(n, n) FROM t"),
        Err(Error::ArityMismatch { .. })
    ));
    // Duplicate without OR REPLACE.
    assert!(matches!(
        run_try(
            &engine,
            "CREATE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 $$",
        ),
        Err(Error::FunctionExists { .. })
    ));
    // OR REPLACE swaps the body.
    run(
        &engine,
        "CREATE OR REPLACE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 * 3 $$",
    );
    assert_eq!(
        rows(run(&engine, "SELECT dbl(n) FROM t")),
        vec![vec![Value::Int(9)]]
    );
    // DROP, then the name is unknown again.
    assert!(matches!(
        run(&engine, "DROP FUNCTION dbl"),
        ExecutionResult::FunctionDropped
    ));
    assert!(matches!(
        run_try(&engine, "SELECT dbl(n) FROM t"),
        Err(Error::UnknownFunction(_))
    ));
    assert!(matches!(
        run_try(&engine, "DROP FUNCTION dbl"),
        Err(Error::FunctionNotFound { .. })
    ));
}

#[test]
fn function_body_must_be_a_scalar_select() {
    let engine = BtreeEngine::new();
    // A body with FROM (not a bare scalar expression) is rejected at create.
    assert!(
        run_try(
            &engine,
            "CREATE FUNCTION bad() RETURNS INT AS $$ SELECT n FROM t $$",
        )
        .is_err()
    );
    // A body referencing an undeclared parameter is rejected.
    assert!(
        run_try(
            &engine,
            "CREATE FUNCTION bad(x INT) RETURNS INT AS $$ SELECT $2 $$",
        )
        .is_err()
    );
}
