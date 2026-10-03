//! Statements that consume a query (`CREATE TABLE AS`, `CREATE` / `REFRESH MATERIALIZED VIEW`,
//! `INSERT ... SELECT`, a buffered selective `SELECT`) must work when the query's input and result
//! are far larger than `work_mem`, given a spill directory, and must leave exactly the rows an
//! unbounded run leaves.
//!
//! `spill_config` is process-wide, so this binary holds a single test.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, SpillConfig, analyze, parse, plan,
    set_spill_config,
};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let txn = self
            .0
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_list_indexes(self.0, txn, table);
        let _ = self.0.commit(txn);
        out
    }
}

fn run(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical))
}

/// The rows of `sql` as a sorted multiset.
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let Ok(ExecutionResult::Rows { mut rows, .. }) = run(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

/// Statements writing into `out_<n>`, each paired with the query that reads the result back.
const STATEMENTS: &[(&str, &str)] = &[
    (
        "CREATE TABLE out_1 AS SELECT g, count(*) AS n, sum(id) AS s FROM src GROUP BY g",
        "SELECT * FROM out_1",
    ),
    (
        "CREATE TABLE out_2 AS SELECT id, s FROM src WHERE id % 97 = 3",
        "SELECT * FROM out_2",
    ),
    (
        "CREATE TABLE out_3 AS SELECT id, row_number() OVER (PARTITION BY g ORDER BY id) AS rn FROM src",
        "SELECT * FROM out_3",
    ),
    (
        "CREATE MATERIALIZED VIEW out_4 AS SELECT g, max(s) AS m FROM src GROUP BY g",
        "SELECT * FROM out_4",
    ),
    (
        "INSERT INTO sink_g SELECT g, count(*) FROM src GROUP BY g",
        "SELECT * FROM sink_g",
    ),
    (
        "INSERT INTO sink_s SELECT s FROM src ORDER BY s",
        "SELECT * FROM sink_s",
    ),
    (
        "INSERT INTO sink_d SELECT DISTINCT g FROM src",
        "SELECT * FROM sink_d",
    ),
    (
        "INSERT INTO sink_w SELECT id, rank() OVER (ORDER BY g) FROM src",
        "SELECT * FROM sink_w",
    ),
    // A key range read through the primary-key index.
    (
        "CREATE TABLE out_5 AS SELECT * FROM keyed WHERE id >= 100 AND id <= 4800",
        "SELECT * FROM out_5",
    ),
    (
        "INSERT INTO sink_k SELECT s FROM keyed WHERE id >= 10 AND id <= 5000",
        "SELECT * FROM sink_k",
    ),
];

#[test]
fn query_consuming_statements_work_past_work_mem() {
    // Run every statement twice, unbounded and then under a small budget, on separate engines.
    let mut results = Vec::new();
    for bounded in [false, true] {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        run(
            engine,
            &mut session,
            "CREATE TABLE src (id INT, g INT, s TEXT)",
        )
        .unwrap();
        for start in (0..6000).step_by(500) {
            let values = (start..start + 500)
                .map(|i: i64| format!("({i}, {}, 'value-{:05}-padding')", i % 900, 6000 - i))
                .collect::<Vec<_>>()
                .join(",");
            run(
                engine,
                &mut session,
                &format!("INSERT INTO src VALUES {values}"),
            )
            .unwrap();
        }
        for ddl in [
            "CREATE TABLE sink_g (g INT, n INT)",
            "CREATE TABLE sink_s (s TEXT)",
            "CREATE TABLE sink_d (g INT)",
            "CREATE TABLE sink_w (id INT, r INT)",
            "CREATE TABLE sink_k (s TEXT)",
            "CREATE TABLE keyed (id INT PRIMARY KEY, s TEXT)",
            "INSERT INTO keyed SELECT id, s FROM src",
        ] {
            run(engine, &mut session, ddl).unwrap();
        }
        if bounded {
            set_spill_config(Some(SpillConfig {
                dir: std::env::temp_dir(),
                threshold_bytes: 64 * 1024 * 1024,
            }));
            run(engine, &mut session, "SET work_mem = '16kB'").unwrap();
        }
        let mut got = Vec::new();
        // The key ranges below must read through the primary-key index.
        let plan = rows(
            engine,
            &mut session,
            "EXPLAIN SELECT * FROM keyed WHERE id >= 100 AND id <= 4800",
        );
        assert!(format!("{plan:?}").contains("IndexScan"), "{plan:?}");
        for (statement, read_back) in STATEMENTS {
            run(engine, &mut session, statement).unwrap_or_else(|e| panic!("{statement}: {e}"));
            run(engine, &mut session, "RESET work_mem").unwrap();
            got.push(rows(engine, &mut session, read_back));
            if bounded {
                run(engine, &mut session, "SET work_mem = '16kB'").unwrap();
            }
        }
        // A refresh after the base table changed, and a selective buffered SELECT.
        run(
            engine,
            &mut session,
            "INSERT INTO src VALUES (7000, 5, 'late')",
        )
        .unwrap();
        run(engine, &mut session, "REFRESH MATERIALIZED VIEW out_4")
            .unwrap_or_else(|e| panic!("REFRESH: {e}"));
        let selective = rows(
            engine,
            &mut session,
            "SELECT id FROM src WHERE id % 1000 = 7",
        );
        run(engine, &mut session, "RESET work_mem").unwrap();
        got.push(rows(engine, &mut session, "SELECT * FROM out_4"));
        got.push(selective);
        set_spill_config(None);
        results.push(got);
    }
    let [unbounded, bounded] = results.as_slice() else {
        panic!("two runs expected");
    };
    for (i, (want, got)) in unbounded.iter().zip(bounded).enumerate() {
        assert!(!want.is_empty(), "result {i} is empty");
        assert_eq!(got, want, "result {i} differs under the budget");
    }
}
