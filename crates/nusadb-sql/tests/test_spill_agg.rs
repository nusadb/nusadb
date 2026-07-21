//! Sort-based spilling GROUP BY: a grouped aggregate with spill-to-disk enabled must
//! return the same rows as the in-memory group-by, whether the input fits the budget or overflows
//! into sorted runs that are merged and folded group-by-group.
//!
//! The compared aggregates are order-INSENSITIVE (SUM/COUNT/AVG/MIN/MAX), so the result is identical
//! regardless of within-group row order. (For an order-sensitive aggregate like `ARRAY_AGG` without
//! an explicit `ORDER BY`, SQL leaves the order unspecified, and the sort-based path may legitimately
//! differ from the hash-based path — so those are not part of the oracle.)
//!
//! `spill_config` is process-wide, so this lives in its own test binary (run sequentially) and
//! resets the config to `None` before returning.

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

/// Result rows as a multiset (group order is unspecified, so sort before comparing).
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn spilling_group_by_matches_in_memory_then_resets() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE t (grp INT, v INT, pad TEXT)",
    );
    // ~30 distinct groups (incl. a NULL group every 11th), each with several rows; `pad` widens rows
    // so a 64-byte budget forces many sorted runs that must be merged and folded.
    for i in 0..300 {
        let grp = if i % 11 == 0 {
            "NULL".to_owned()
        } else {
            (i % 30).to_string()
        };
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({grp}, {}, 'pad-{i}')", i * 3),
        );
    }

    let dir = std::env::temp_dir();
    let sql = "SELECT grp, COUNT(*), SUM(v), MIN(v), MAX(v), AVG(v) FROM t GROUP BY grp";

    set_spill_config(None);
    let baseline = rows(engine, &mut session, sql);
    assert!(
        baseline.len() >= 30,
        "expected ~31 groups, got {}",
        baseline.len()
    );

    // Build fits a generous budget → in-memory sort fast path inside the sort-based group-by.
    set_spill_config(Some(SpillConfig {
        dir: dir.clone(),
        threshold_bytes: 8 * 1024 * 1024,
    }));
    assert_eq!(
        rows(engine, &mut session, sql),
        baseline,
        "spilling group-by (fits budget) must match in-memory"
    );

    // Tiny budget → sorted runs spilled and merged, folded group-by-group.
    set_spill_config(Some(SpillConfig {
        dir,
        threshold_bytes: 64,
    }));
    assert_eq!(
        rows(engine, &mut session, sql),
        baseline,
        "spilling group-by (disk runs merged + folded) must match in-memory"
    );

    set_spill_config(None);
}
