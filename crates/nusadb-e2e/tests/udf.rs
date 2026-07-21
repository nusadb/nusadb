//! End-to-end tests for scalar user-defined functions: registering a Rust function and
//! invoking it from SQL (projection, WHERE, nested), argument type/arity checking, unknown-function
//! rejection, NULL handling, and UDF-raised errors — driven through `parse → analyze → plan → execute`
//! against the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::{ColumnType, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::udf::register_scalar_udf;
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

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn scalar_udf_in_projection_and_where() {
    // A doubling UDF: INT -> INT, NULL-preserving.
    register_scalar_udf(
        "udf_double",
        vec![ColumnType::Int],
        ColumnType::Int,
        Arc::new(|args| match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(n.wrapping_mul(2))),
            Some(Value::Null) => Ok(Value::Null),
            _ => Err("udf_double expects an integer".to_owned()),
        }),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5), (2, 10), (3, 20)");

    // Projection: the UDF runs per row, its INT return type flows through.
    let out = rows(run(
        &engine,
        "SELECT id, udf_double(val) FROM t ORDER BY id",
    ));
    assert_eq!(
        out,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(40)],
        ]
    );

    // WHERE: the UDF result composes with comparison operators.
    let out = rows(run(
        &engine,
        "SELECT id FROM t WHERE udf_double(val) >= 20 ORDER BY id",
    ));
    assert_eq!(out, vec![vec![Value::Int(2)], vec![Value::Int(3)]]);
}

#[test]
fn scalar_udf_multi_arg_and_nested() {
    // A two-argument UDF: concatenate with a separator.
    register_scalar_udf(
        "udf_join2",
        vec![ColumnType::Text, ColumnType::Text],
        ColumnType::Text,
        Arc::new(|args| match args {
            [Value::Text(a), Value::Text(b)] => Ok(Value::Text(format!("{a}-{b}"))),
            _ => Ok(Value::Null),
        }),
    );
    register_scalar_udf(
        "udf_shout",
        vec![ColumnType::Text],
        ColumnType::Text,
        Arc::new(|args| match args.first() {
            Some(Value::Text(s)) => Ok(Value::Text(s.to_uppercase())),
            _ => Ok(Value::Null),
        }),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a TEXT, b TEXT)");
    run(&engine, "INSERT INTO t VALUES ('hi', 'there')");

    // Nested UDF calls + a built-in: udf_shout(udf_join2(a, b)).
    let out = rows(run(&engine, "SELECT udf_shout(udf_join2(a, b)) FROM t"));
    assert_eq!(out, vec![vec![Value::Text("HI-THERE".to_owned())]]);
}

#[test]
fn udf_argument_checks_and_unknown_function() {
    register_scalar_udf(
        "udf_id",
        vec![ColumnType::Int],
        ColumnType::Int,
        Arc::new(|args| Ok(args.first().cloned().unwrap_or(Value::Null))),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (n INT)");
    run(&engine, "INSERT INTO t VALUES (1)");

    // Wrong arity.
    assert!(matches!(
        run_try(&engine, "SELECT udf_id(n, n) FROM t"),
        Err(Error::ArityMismatch { .. })
    ));
    // Wrong argument type (TEXT where INT is declared).
    assert!(matches!(
        run_try(&engine, "SELECT udf_id('x') FROM t"),
        Err(Error::TypeMismatch { .. })
    ));
    // A name with no registered UDF is an unknown function.
    assert!(matches!(
        run_try(&engine, "SELECT no_such_udf(n) FROM t"),
        Err(Error::UnknownFunction(_))
    ));
}

#[test]
fn udf_raised_error_propagates() {
    // A UDF that rejects negative inputs.
    register_scalar_udf(
        "udf_pos",
        vec![ColumnType::Int],
        ColumnType::Int,
        Arc::new(|args| match args.first() {
            Some(Value::Int(n)) if *n < 0 => Err(format!("negative: {n}")),
            Some(Value::Int(n)) => Ok(Value::Int(*n)),
            _ => Ok(Value::Null),
        }),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (n INT)");
    run(&engine, "INSERT INTO t VALUES (5), (-3)");

    match run_try(&engine, "SELECT udf_pos(n) FROM t") {
        Err(Error::UdfFailed { name, .. }) => assert_eq!(name, "udf_pos"),
        other => panic!("expected UdfFailed, got {other:?}"),
    }
}

#[test]
fn udf_receives_null_argument() {
    // A NULL-aware UDF: returns -1 for NULL, the value otherwise.
    register_scalar_udf(
        "udf_or_neg1",
        vec![ColumnType::Int],
        ColumnType::Int,
        Arc::new(|args| match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(*n)),
            _ => Ok(Value::Int(-1)),
        }),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    run(&engine, "INSERT INTO t VALUES (1, 7), (2, NULL)");
    let out = rows(run(&engine, "SELECT udf_or_neg1(val) FROM t ORDER BY id"));
    assert_eq!(out, vec![vec![Value::Int(7)], vec![Value::Int(-1)]]);
}

#[test]
fn scalar_udf_argument_is_coerced_to_the_declared_type() {
    // Deep-gate: the analyzer only checks *assignability*, so an INT argument may reach a FLOAT
    // parameter. The executor must coerce it to FLOAT before invoking the UDF — this one matches only
    // Value::Float, so a raw, un-coerced Value::Int would have fallen through to NULL.
    register_scalar_udf(
        "udf_halve",
        vec![ColumnType::Float],
        ColumnType::Float,
        Arc::new(|args| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f / 2.0)),
            _ => Ok(Value::Null),
        }),
    );
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5)");
    // `val` is INT; the FLOAT parameter must receive a coerced Float(5.0), giving 2.5 (not NULL).
    let out = rows(run(&engine, "SELECT udf_halve(val) FROM t"));
    assert_eq!(out, vec![vec![Value::Float(2.5)]]);
}
