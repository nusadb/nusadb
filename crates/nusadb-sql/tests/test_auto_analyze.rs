//! `auto_analyze_stale_tables` (D-AUTO-ANALYZE): the off-query-path policy that keeps the planner's
//! statistics fresh must analyse a table once its write churn crosses the scale-factor threshold
//! `base + scale * approx_row_count`, populate its statistics, reset its churn, and leave a lightly
//! churned (tiny) table untouched. Proven end-to-end against a real engine so the whole pipeline —
//! churn tracking, the threshold decision, and the ANALYZE it runs — is exercised together.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::fmt::Write as _;

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, auto_analyze_stale_tables, parse,
    plan,
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

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session.execute(plan(logical)).unwrap()
}

/// Insert `n` rows into `table` in one statement, so the whole load commits as `n` churn.
fn insert_rows(engine: &dyn StorageEngine, session: &mut Session, table: &str, n: usize) {
    let mut sql = format!("INSERT INTO {table} VALUES ");
    for i in 0..n {
        if i > 0 {
            sql.push(',');
        }
        write!(sql, "({i},{})", i * 2).unwrap();
    }
    exec(engine, session, &sql);
}

const SCALE: f64 = 0.1; // scale factor: 10% of the row count
const BASE: u64 = 50; // constant churn floor

#[test]
fn auto_analyze_refreshes_a_churned_table_and_resets_its_churn() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT, v INT)");

    // 100 rows → churn 100, over the threshold 50 + 0.1 * 100 = 60.
    insert_rows(engine, &mut session, "t", 100);
    let table = engine.lookup_table("t").unwrap().unwrap();
    assert!(
        engine.table_stats(table.id).unwrap().is_none(),
        "no statistics exist before the sweep"
    );
    assert_eq!(engine.churn_since_analyze(table.id).unwrap(), 100);

    // The sweep analyses the churned table, populating stats and resetting churn.
    let analysed = auto_analyze_stale_tables(engine, SCALE, BASE).unwrap();
    assert!(
        analysed.contains(&"t".to_owned()),
        "the churned table was analysed (got {analysed:?})"
    );
    assert!(
        engine.table_stats(table.id).unwrap().is_some(),
        "statistics are populated after the sweep"
    );
    assert_eq!(
        engine.churn_since_analyze(table.id).unwrap(),
        0,
        "the analyze reset the churn tally"
    );

    // A second sweep with no new writes does nothing — churn is back to zero.
    assert!(
        auto_analyze_stale_tables(engine, SCALE, BASE)
            .unwrap()
            .is_empty(),
        "an unchurned table is not re-analysed"
    );
}

#[test]
fn auto_analyze_leaves_a_lightly_churned_table_alone() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE small (id INT, v INT)");

    // 10 rows → churn 10, below the threshold 50 + 0.1 * 10 = 51: a tiny table needs no statistics.
    insert_rows(engine, &mut session, "small", 10);
    let table = engine.lookup_table("small").unwrap().unwrap();

    let analysed = auto_analyze_stale_tables(engine, SCALE, BASE).unwrap();
    assert!(
        !analysed.contains(&"small".to_owned()),
        "a below-threshold table is skipped (got {analysed:?})"
    );
    assert!(
        engine.table_stats(table.id).unwrap().is_none(),
        "the tiny table gets no statistics"
    );
    assert_eq!(
        engine.churn_since_analyze(table.id).unwrap(),
        10,
        "its churn is left untouched"
    );
}
