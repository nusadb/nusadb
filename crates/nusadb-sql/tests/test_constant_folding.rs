//! Constant folding: the planner evaluates constant sub-expressions and short-circuits
//! boolean operators with a literal operand. These tests pin that the folding is behavior-preserving
//! — the results match what the un-folded query would produce, including error preservation.

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

fn one(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Value {
    let r = rows(engine, session, sql);
    assert_eq!(r.len(), 1, "expected one row from {sql}");
    r.into_iter().next().unwrap().into_iter().next().unwrap()
}

#[test]
fn constant_predicates_fold_without_changing_results() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT)");
    for i in 0..5 {
        exec(engine, &mut session, &format!("INSERT INTO t VALUES ({i})"));
    }

    // `1 = 1 AND a > 2` — the `1=1` folds to TRUE and the `TRUE AND ...` collapses to `a > 2`.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT a FROM t WHERE 1 = 1 AND a > 2"
        ),
        vec![vec![Value::Int(3)], vec![Value::Int(4)]]
    );
    // `a > 100 OR 2 < 5` — `2 < 5` folds to TRUE and `... OR TRUE` collapses to TRUE → all rows.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT a FROM t WHERE a > 100 OR 2 < 5"
        )
        .len(),
        5
    );
    // A wholly-constant false predicate yields no rows.
    assert!(rows(engine, &mut session, "SELECT a FROM t WHERE 2 > 5").is_empty());
    // `false AND <anything>` collapses to no rows regardless of the column predicate.
    assert!(
        rows(
            engine,
            &mut session,
            "SELECT a FROM t WHERE 1 = 0 AND a < 100"
        )
        .is_empty()
    );
}

#[test]
fn constant_expressions_fold_to_correct_values() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    // Arithmetic precedence is respected when folded.
    assert_eq!(one(engine, &mut session, "SELECT 1 + 2 * 3"), Value::Int(7));
    // A constant searched-CASE folds to the matching branch.
    assert_eq!(
        one(
            engine,
            &mut session,
            "SELECT CASE WHEN 1 > 2 THEN 'a' ELSE 'b' END"
        ),
        Value::Text("b".to_owned())
    );
    // COALESCE over constants picks the first non-NULL (the typed NULL folds, then is skipped).
    assert_eq!(
        one(
            engine,
            &mut session,
            "SELECT COALESCE(CAST(NULL AS INT), 4, 9)"
        ),
        Value::Int(4)
    );
    // A constant IN-list folds to a boolean.
    assert_eq!(
        one(engine, &mut session, "SELECT 3 IN (1, 2, 3)"),
        Value::Bool(true)
    );
}

#[test]
fn folding_preserves_errors_and_their_context() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    // A genuinely-constant division by zero still errors at runtime (it is not folded away).
    assert!(matches!(
        try_exec(engine, &mut session, "SELECT 10 / (5 - 5)"),
        Err(Error::DivisionByZero)
    ));
    // A division by zero in a CASE branch that is never taken must NOT fire — the whole constant
    // CASE folds by short-circuit evaluation to its ELSE result.
    assert_eq!(
        one(
            engine,
            &mut session,
            "SELECT CASE WHEN false THEN 1 / 0 ELSE 7 END"
        ),
        Value::Int(7)
    );
}
