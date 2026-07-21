//! Spilling `DISTINCT`: a `SELECT DISTINCT` with spill-to-disk enabled must return the same
//! set of rows as the in-memory `dedupe_rows`, whether the input fits the memory budget or overflows
//! into sorted runs that are merged and deduplicated run-by-run.
//!
//! `DISTINCT` leaves output order unspecified (the in-memory path keeps first-seen order; the
//! spilling path emits in sorted order), so the oracle compares result **multisets** — each variant
//! is sorted before the equality check.
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

/// Result rows as a multiset (DISTINCT order is unspecified, so sort before comparing).
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn spilling_distinct_matches_in_memory_then_resets() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE t (a INT, b TEXT, c INT)",
    );
    // ~40 distinct (a, b, c) triples (incl. a NULL in `a` every 13th), each inserted several times so
    // a tiny budget forces many sorted runs that must be merged and deduplicated. `b` widens rows.
    for i in 0..400 {
        let a = if i % 13 == 0 {
            "NULL".to_owned()
        } else {
            (i % 40).to_string()
        };
        let b = format!("'tag-{}'", i % 7);
        let c = i % 5;
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({a}, {b}, {c})"),
        );
    }

    let dir = std::env::temp_dir();
    let sql = "SELECT DISTINCT a, b, c FROM t";

    set_spill_config(None);
    let baseline = rows(engine, &mut session, sql);
    assert!(
        baseline.len() >= 40,
        "expected many distinct triples, got {}",
        baseline.len()
    );

    // Generous budget → in-memory sort fast path inside the sort-based DISTINCT.
    set_spill_config(Some(SpillConfig {
        dir: dir.clone(),
        threshold_bytes: 8 * 1024 * 1024,
    }));
    assert_eq!(
        rows(engine, &mut session, sql),
        baseline,
        "spilling DISTINCT (fits budget) must match in-memory"
    );

    // Tiny budget → sorted runs spilled to disk, merged, deduplicated run-by-run.
    set_spill_config(Some(SpillConfig {
        dir,
        threshold_bytes: 64,
    }));
    assert_eq!(
        rows(engine, &mut session, sql),
        baseline,
        "spilling DISTINCT (disk runs merged + deduped) must match in-memory"
    );

    set_spill_config(None);
}
