//! `DISTINCT ON` and grouping sets (`ROLLUP` / `CUBE` / `GROUPING SETS`) with spill-to-disk enabled
//! must return exactly the rows the in-memory path returns, whether the input fits the budget or is
//! sorted from disk. A query with an `ORDER BY` must keep its order; the rest are compared as
//! multisets, since SQL leaves their order unspecified.
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
    Catalog, Error, ExecutionResult, IndexInfo, RowSink, Session, SpillConfig, analyze, parse,
    plan, set_spill_config,
};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

#[derive(Default)]
struct Collect(Vec<Vec<Value>>);
impl RowSink for Collect {
    fn columns(&mut self, _columns: &[String]) -> Result<(), Error> {
        Ok(())
    }
    fn row(&mut self, row: &[Value]) -> Result<(), Error> {
        self.0.push(row.to_vec());
        Ok(())
    }
}

fn planned(engine: &dyn StorageEngine, sql: &str) -> nusadb_sql::PhysicalPlan {
    plan(analyze(parse(sql).unwrap(), &Cat(engine)).unwrap())
}

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) {
    session.execute(planned(engine, sql)).unwrap();
}

fn spill(threshold_bytes: usize) {
    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes,
    }));
}

/// Queries whose row order is fixed by an `ORDER BY`.
const ORDERED: &[&str] = &[
    "SELECT DISTINCT ON (p) p, id, x FROM w ORDER BY p, x DESC, id",
    "SELECT DISTINCT ON (p, x) p, x, id FROM w ORDER BY x, p, id DESC",
    "SELECT DISTINCT ON (x) x, v FROM w ORDER BY x, v NULLS FIRST, id",
    "SELECT DISTINCT ON (p) p, id FROM w ORDER BY id DESC",
    // A plain sort with many ties: a spilled sort keeps them in input order, as in memory.
    "SELECT id, p FROM w ORDER BY p",
    "SELECT DISTINCT ON (p, x) p, x, id FROM w ORDER BY p DESC, id",
    "SELECT DISTINCT ON (p, x) p, x, id FROM w ORDER BY p DESC LIMIT 5",
];

/// Queries compared as multisets.
const UNORDERED: &[&str] = &[
    "SELECT DISTINCT ON (p) p, id FROM w",
    "SELECT DISTINCT ON (x % 7) x % 7, id, v FROM w",
    "SELECT p, x % 3, count(*), sum(v), min(id) FROM w GROUP BY ROLLUP (p, x % 3)",
    "SELECT p, x % 4, count(v), max(id), GROUPING(p, x % 4) FROM w GROUP BY CUBE (p, x % 4)",
    "SELECT p, x, count(*) FROM w GROUP BY GROUPING SETS ((p), (x), ())",
    "SELECT p, count(*) FILTER (WHERE v > 100), avg(v) FROM w GROUP BY ROLLUP (p)",
    "SELECT p, string_agg(id::TEXT, ',') FROM w GROUP BY GROUPING SETS ((p), ())",
    "SELECT p, count(*) FROM w WHERE id < 0 GROUP BY ROLLUP (p)",
];

#[test]
fn spilled_distinct_on_and_grouping_sets_match_in_memory() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(
        engine,
        &mut session,
        "CREATE TABLE w (id INT, p INT, x INT, v INT)",
    );
    for start in (0..4000).step_by(500) {
        let values = (start..start + 500)
            .map(|i: i64| {
                let v = if i % 7 == 0 {
                    "NULL".to_owned()
                } else {
                    (i * 3 % 1000).to_string()
                };
                format!("({i}, {}, {}, {v})", i % 4, i % 50)
            })
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO w VALUES {values}"),
        );
    }
    for (sql, ordered) in ORDERED
        .iter()
        .map(|q| (q, true))
        .chain(UNORDERED.iter().map(|q| (q, false)))
    {
        let sort = |mut rows: Vec<Vec<Value>>| {
            if !ordered {
                rows.sort_by_key(|r| format!("{r:?}"));
            }
            rows
        };
        set_spill_config(None);
        let want = sort(raw(engine, &mut session, sql, false));
        for threshold in [1 << 20, 2048] {
            spill(threshold);
            for buffered_path in [false, true] {
                let got = sort(raw(engine, &mut session, sql, buffered_path));
                assert_eq!(
                    got, want,
                    "{sql} (threshold {threshold}, buffered {buffered_path})"
                );
            }
        }
    }
    set_spill_config(None);
}

/// The rows of `sql` in the order produced, through the streaming or the buffered path.
fn raw(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
    buffered_path: bool,
) -> Vec<Vec<Value>> {
    if buffered_path {
        let ExecutionResult::Rows { rows, .. } = session
            .execute(planned(engine, sql))
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
        else {
            panic!("expected rows from: {sql}");
        };
        return rows;
    }
    let mut sink = Collect::default();
    session
        .execute_streaming(planned(engine, sql), &mut sink)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    sink.0
}
