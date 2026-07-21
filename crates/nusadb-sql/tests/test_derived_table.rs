//! Derived tables in FROM (`FROM (SELECT ...) AS x`)increments 3a/3b/3c. A parenthesized
//! subquery with an alias is a relation in its own right: its projection forms the column scope,
//! qualified by the alias. Supported as the FROM base (3a), as a join input (3b), and — with
//! `LATERAL` — correlating to columns on its left, re-evaluated per left row (3c).

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

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn derived_table_in_from_base() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..6 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'r{i}')"),
        );
    }

    // A derived table's projection + filter define the relation.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT a FROM (SELECT a, b FROM t WHERE a > 3) AS x"
        ),
        vec![vec![Value::Int(4)], vec![Value::Int(5)]]
    );
    // Columns are qualified by the alias.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT x.a, x.b FROM (SELECT a, b FROM t WHERE a = 2) AS x"
        ),
        vec![vec![Value::Int(2), Value::Text("r2".to_owned())]]
    );
    // The outer query can aggregate / filter / rename over the derived relation.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT COUNT(*) FROM (SELECT a FROM t WHERE a > 3) AS x"
        ),
        vec![vec![Value::Int(2)]]
    );
    // An inner alias is visible as the derived relation's column.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT y FROM (SELECT a AS y FROM t WHERE a = 0) AS x"
        ),
        vec![vec![Value::Int(0)]]
    );
    // An outer WHERE over the derived relation works.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT a FROM (SELECT a FROM t) AS x WHERE a < 2 ORDER BY a"
        ),
        vec![vec![Value::Int(0)], vec![Value::Int(1)]]
    );
}

#[test]
fn prepared_parameters_inside_a_derived_table_are_bound() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT)");
    for i in 0..5 {
        exec(engine, &mut session, &format!("INSERT INTO t VALUES ({i})"));
    }

    // `$1` lives inside the derived subquery — PREPARE must count it (arity 1) and EXECUTE must
    // substitute it before the subquery is analyzed, otherwise the placeholder leaks through.
    exec(
        engine,
        &mut session,
        "PREPARE p AS SELECT a FROM (SELECT a FROM t WHERE a > $1) AS x ORDER BY a",
    );
    let ExecutionResult::Rows { rows, .. } = exec(engine, &mut session, "EXECUTE p (2)") else {
        panic!("expected rows from EXECUTE");
    };
    assert_eq!(rows, vec![vec![Value::Int(3)], vec![Value::Int(4)]]);
}

#[test]
fn derived_table_as_join_input() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    exec(engine, &mut session, "CREATE TABLE u (a INT, c INT)");
    for i in 0..4 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'r{i}')"),
        );
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO u VALUES ({i}, {})", i * 10),
        );
    }

    // A derived table on the right of a join: only `u` rows with c >= 20 survive the subquery, so
    // the inner join keeps t.a in {2, 3}.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT t.a, x.c FROM t JOIN (SELECT a, c FROM u WHERE c >= 20) AS x ON t.a = x.a"
        ),
        vec![
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ]
    );
    // A LEFT JOIN onto a derived table NULL-pads unmatched left rows.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT t.a, x.c FROM t LEFT JOIN (SELECT a, c FROM u WHERE c >= 20) AS x ON t.a = x.a \
             WHERE t.a < 1"
        ),
        vec![vec![Value::Int(0), Value::Null]]
    );
}

#[test]
fn lateral_join_correlates_to_the_left() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT)");
    exec(engine, &mut session, "CREATE TABLE u (k INT, c INT)");
    for a in 0..4 {
        exec(engine, &mut session, &format!("INSERT INTO t VALUES ({a})"));
    }
    // u has two rows for k=1, one each for k=0 and k=2, and none for k=3.
    for (k, c) in [(0, 100), (1, 101), (1, 111), (2, 102)] {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO u VALUES ({k}, {c})"),
        );
    }

    // CROSS JOIN LATERAL: the subquery filters `u` by the current `t.a` (a correlated reference),
    // so each t row pairs with only its matching u rows. `t.a = 3` matches nothing → dropped.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT t.a, x.c FROM t CROSS JOIN LATERAL (SELECT c FROM u WHERE u.k = t.a) AS x"
        ),
        vec![
            vec![Value::Int(0), Value::Int(100)],
            vec![Value::Int(1), Value::Int(101)],
            vec![Value::Int(1), Value::Int(111)],
            vec![Value::Int(2), Value::Int(102)],
        ]
    );

    // LEFT JOIN LATERAL keeps an unmatched left row, NULL-padded — here `t.a = 3` has no `u` match.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT t.a, x.c FROM t LEFT JOIN LATERAL (SELECT c FROM u WHERE u.k = t.a) AS x ON true \
             WHERE t.a = 3"
        ),
        vec![vec![Value::Int(3), Value::Null]]
    );

    // A LATERAL subquery can also reference the left row in its projection, not just its filter.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT y FROM t CROSS JOIN LATERAL (SELECT t.a + c AS y FROM u WHERE u.k = t.a) AS x"
        ),
        vec![
            vec![Value::Int(100)], // 0 + 100
            vec![Value::Int(102)], // 1 + 101
            vec![Value::Int(104)], // 2 + 102
            vec![Value::Int(112)], // 1 + 111
        ]
    );
}

#[test]
fn derived_table_limitations_are_rejected() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT)");

    let err = |sql: &str| -> Error {
        match parse(sql).and_then(|s| analyze(s, &Cat(engine))) {
            Err(e) => e,
            Ok(_) => panic!("expected error for: {sql}"),
        }
    };

    // A derived table requires an alias.
    assert!(matches!(
        err("SELECT a FROM (SELECT a FROM t)"),
        Error::Unsupported(_)
    ));
    // A LATERAL derived table cannot be the first FROM item.
    assert!(matches!(
        err("SELECT a FROM LATERAL (SELECT a FROM t) AS x"),
        Error::Unsupported(_)
    ));
    // A RIGHT/FULL JOIN LATERAL is meaningless (the right side depends on the left) → rejected.
    assert!(matches!(
        err("SELECT a FROM t RIGHT JOIN LATERAL (SELECT a FROM t) AS x ON true"),
        Error::Unsupported(_)
    ));
}
