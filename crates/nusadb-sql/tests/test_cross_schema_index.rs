//! A table's indexes and statistics are resolved by its `(schema, name)` identity, never by a
//! bare name that would go back through the search path: `app.t` must not be scanned through
//! an index of `public.t`, alone, under a join, or on the ordered-scan path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, ast::Value, parse, plan,
};

/// The production adapter shape: tables, indexes, and stats resolved from the engine, with the
/// schema-qualified lookups the analyzer uses once a reference is resolved.
struct Cat<'a> {
    engine: &'a BtreeEngine,
}

impl Cat<'_> {
    fn with_txn<T>(
        &self,
        f: impl FnOnce(nusadb_core::TxnId) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = f(txn);
        let _ = self.engine.commit(txn);
        out
    }
}

impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine
            .lookup_table_in(schema, name)
            .map_err(Into::into)
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes(self.engine, txn, table))
    }
    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        self.with_txn(|txn| nusadb_sql::catalog_table_stats(self.engine, txn, table))
    }
    fn list_indexes_in(&self, schema: &str, name: &str) -> Result<Vec<IndexInfo>, Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes_in(self.engine, txn, schema, name))
    }
    fn table_stats_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, Error> {
        self.with_txn(|txn| nusadb_sql::catalog_table_stats_in(self.engine, txn, schema, name))
    }
}

fn run(
    engine: &'static BtreeEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat { engine })?;
    session.execute(plan(logical))
}

fn ints(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> Vec<i64> {
    match run(engine, session, sql).unwrap_or_else(|e| panic!("`{sql}` failed: {e}")) {
        ExecutionResult::Rows { rows, .. } => rows
            .iter()
            .map(|row| match row.first() {
                Some(Value::Int(v)) => *v,
                other => panic!("expected an integer, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn explain(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> String {
    match run(engine, session, &format!("EXPLAIN {sql}")).unwrap() {
        ExecutionResult::Rows { rows, .. } => rows
            .iter()
            .map(|row| match &row[..] {
                [Value::Text(line)] => line.clone(),
                other => panic!("unexpected EXPLAIN row {other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => panic!("expected rows, got {other:?}"),
    }
}

/// `public.t(a, b)` indexed on `a`; `app.t(x, a, y)` with the same name, a different layout and
/// no index; `s(a)` to join against.
fn fixture() -> (&'static BtreeEngine, Session<'static>) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    for sql in [
        "CREATE TABLE t (a INT NOT NULL, b TEXT)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE SCHEMA app",
        "CREATE TABLE app.t (x TEXT, a INT NOT NULL, y INT)",
        "CREATE TABLE s (a INT NOT NULL)",
    ] {
        run(engine, &mut session, sql).unwrap();
    }
    for i in 0..20 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'public{i}')"),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!("INSERT INTO app.t VALUES ('app{i}', {i}, {})", i * 10),
        )
        .unwrap();
        run(engine, &mut session, &format!("INSERT INTO s VALUES ({i})")).unwrap();
    }
    run(engine, &mut session, "ANALYZE t").unwrap();
    (engine, session)
}

#[test]
fn a_same_named_table_in_another_schema_never_uses_the_public_tables_index() {
    let (engine, mut session) = fixture();
    // Positive control: the indexed table itself plans through its index.
    let control = "SELECT a FROM public.t WHERE a = 5";
    assert!(explain(engine, &mut session, control).contains("t_a"));
    assert_eq!(ints(engine, &mut session, control), vec![5]);
    // The other table under the same name has no index and a different row layout.
    for sql in [
        "SELECT y FROM app.t WHERE a = 5",
        "SELECT q.y FROM s JOIN app.t AS q ON s.a = q.a WHERE q.a = 5",
        "SELECT q.y FROM app.t AS q JOIN s ON s.a = q.a WHERE q.a = 5",
        "SELECT y FROM app.t WHERE a BETWEEN 5 AND 6 ORDER BY a LIMIT 1",
    ] {
        let plan = explain(engine, &mut session, sql);
        assert!(!plan.contains("IndexScan"), "{sql}\n{plan}");
        assert_eq!(ints(engine, &mut session, sql), vec![50], "{sql}");
    }
    // Inside a recursive CTE's recursive term the catalog is wrapped by the CTE overlay; the
    // same rule holds there.
    let recursive = "WITH RECURSIVE r(n) AS (SELECT 5 UNION ALL SELECT q.y FROM app.t AS q JOIN r \
                     ON q.a = r.n WHERE q.a = 5) SELECT n FROM r ORDER BY n";
    let plan = explain(engine, &mut session, recursive);
    assert!(!plan.contains("IndexScan"), "{plan}");
    assert_eq!(ints(engine, &mut session, recursive), vec![5, 50]);
}
