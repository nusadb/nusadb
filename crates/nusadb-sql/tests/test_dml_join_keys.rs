//! `UPDATE ... FROM`, `DELETE ... USING` and `MERGE` find a target row's partners through the
//! equalities their condition requires, instead of testing every pair. The result must be the
//! one testing every pair gives: the same rows changed, and for an UPDATE with several matching
//! FROM rows the same first one used. Each statement runs twice on identical tables, once as
//! written and once with its condition wrapped in `(...) OR FALSE`, which offers no equality to
//! key on and so takes the pairwise path; the tables must end equal.

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

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(
        parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}")),
        &Cat(engine),
    )
    .unwrap_or_else(|e| panic!("{sql}: {e}"));
    session
        .execute(plan(logical))
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let ExecutionResult::Rows { mut rows, .. } = run(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

/// Statements over a target `t` and a source `s`, with `$COND` standing for the join condition.
const STATEMENTS: &[(&str, &str)] = &[
    ("UPDATE t SET v = s.v FROM s WHERE $COND", "t.k = s.k"),
    (
        "UPDATE t SET v = s.v + t.v FROM s WHERE $COND",
        "t.k = s.k AND s.v > 100",
    ),
    (
        "UPDATE t SET name = s.name FROM s WHERE $COND",
        "t.name = s.name AND t.k = s.k",
    ),
    (
        "UPDATE t SET v = s.v FROM (SELECT k, v, amount FROM s WHERE v % 3 = 0) AS s WHERE $COND",
        "t.amount = s.amount",
    ),
    (
        "DELETE FROM t USING s WHERE $COND",
        "t.k = s.k AND s.v < 500",
    ),
    ("DELETE FROM t USING s WHERE $COND", "t.amount = s.amount"),
    (
        "MERGE INTO t USING (SELECT DISTINCT ON (k) * FROM s ORDER BY k, v) AS s ON $COND WHEN MATCHED THEN UPDATE SET v = s.v \
         WHEN NOT MATCHED THEN INSERT (k, v, name, amount) VALUES (s.k, s.v, s.name, s.amount)",
        "t.k = s.k",
    ),
    (
        "MERGE INTO t USING (SELECT DISTINCT ON (k) * FROM s ORDER BY k, v) AS s ON $COND WHEN MATCHED AND s.v > 300 THEN DELETE \
         WHEN NOT MATCHED BY SOURCE THEN UPDATE SET v = -1",
        "t.k = s.k AND t.name = s.name",
    ),
];

fn setup(engine: &dyn StorageEngine, session: &mut Session) {
    run(
        engine,
        session,
        "CREATE TABLE t (k INT, v INT, name TEXT, amount NUMERIC(10, 2))",
    );
    run(
        engine,
        session,
        "CREATE TABLE s (k INT, v INT, name TEXT, amount NUMERIC(10, 2))",
    );
    let target = (0..400_i64)
        .map(|i| {
            let k = if i % 13 == 0 {
                "NULL".to_owned()
            } else {
                (i % 150).to_string()
            };
            format!("({k}, {i}, 'n{}', {}.{:02})", i % 7, i % 40, i % 3 * 10)
        })
        .collect::<Vec<_>>()
        .join(",");
    run(engine, session, &format!("INSERT INTO t VALUES {target}"));
    // Several source rows per key, so "the first match" decides an UPDATE's value.
    let source = (0..500_i64)
        .map(|i| {
            let k = if i % 17 == 0 {
                "NULL".to_owned()
            } else {
                (i % 180).to_string()
            };
            format!(
                "({k}, {}, 'n{}', {}.{:03})",
                i * 7 % 1000,
                i % 5,
                i % 45,
                i % 3 * 100
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    run(engine, session, &format!("INSERT INTO s VALUES {source}"));
}

#[test]
fn keyed_dml_joins_change_exactly_what_pairwise_ones_do() {
    for (statement, cond) in STATEMENTS {
        let mut tables = Vec::new();
        for keyed in [true, false] {
            let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
            let mut session = Session::new(engine);
            setup(engine, &mut session);
            let condition = if keyed {
                (*cond).to_owned()
            } else {
                format!("(({cond}) OR FALSE)")
            };
            run(
                engine,
                &mut session,
                &statement.replace("$COND", &condition),
            );
            tables.push(rows(
                engine,
                &mut session,
                "SELECT k, v, name, amount FROM t",
            ));
        }
        assert_eq!(tables[0], tables[1], "{statement} with $COND");
    }
}
