//! External merge sort: `ORDER BY` with spill-to-disk enabled must return rows in exactly
//! the same order as the in-memory sort, whether the input fits the budget or overflows it into
//! many runs that are k-way merged. Each ORDER BY ends in a unique tiebreaker (`id`), so the order
//! is total and the comparison is exact-sequence (stronger than a multiset).
//!
//! `spill_config` is a process-wide static, so this lives in its own test binary (run sequentially)
//! and resets the config to `None` before returning.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Row, Session, SpillConfig, analyze, parse, plan,
    set_spill_config,
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

/// Rows in the order the query returned them (no re-sorting — the order is the thing under test).
fn ordered(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows
}

#[test]
fn external_sort_matches_in_memory_then_resets() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT NOT NULL, grp INT, name TEXT)",
    );
    // 100 rows: `grp` has ties and NULLs (every 7th), so multi-key orders exercise the tiebreaker
    // and NULL placement; `name` widens each row so a 64-byte budget forces many runs.
    for i in 0..100 {
        let grp = if i % 7 == 0 {
            "NULL".to_owned()
        } else {
            ((i * 31) % 17).to_string()
        };
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, {grp}, 'row-{i}-padding')"),
        );
    }

    let dir = std::env::temp_dir();
    for order in [
        "ORDER BY id",
        "ORDER BY id DESC",
        "ORDER BY grp, id",
        "ORDER BY grp DESC NULLS FIRST, id",
    ] {
        let sql = format!("SELECT id, grp FROM t {order}");

        set_spill_config(None);
        let baseline = ordered(engine, &mut session, &sql);
        assert_eq!(baseline.len(), 100, "{order}: all rows returned");

        // Build fits a generous budget → in-memory sort fast path.
        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 8 * 1024 * 1024,
        }));
        assert_eq!(
            ordered(engine, &mut session, &sql),
            baseline,
            "{order}: external sort (fits budget) must match in-memory order"
        );

        // Tiny budget → many sorted runs spilled to disk and k-way merged.
        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 64,
        }));
        assert_eq!(
            ordered(engine, &mut session, &sql),
            baseline,
            "{order}: external sort (k-way merge of disk runs) must match in-memory order"
        );
    }

    set_spill_config(None);
}
