//! Spilling set operations: `UNION`/`INTERSECT`/`EXCEPT [ALL]` with spill-to-disk must
//! return the same multiset as the in-memory `combine_set` path, whether each operand fits the
//! memory budget or overflows into sorted runs that are k-way merged and combined on the fly.
//!
//! The set-op result order is unspecified (the spilling path emits in sorted order; the in-memory
//! path keeps `combine_set`'s order), so the oracle compares result **multisets** — each variant is
//! sorted before the equality check. Every operator is exercised across three budgets, including a
//! 64-byte budget that forces many disk runs, plus a NULL-bearing input (NULL = NULL for set-op
//! membership), duplicates (so `ALL` multiset counts matter), a nested chain, and a multi-column row.
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

/// Result rows as a multiset (set-op order is unspecified, so sort before comparing).
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn spilling_set_ops_match_in_memory_then_reset() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE a (x INT)");
    exec(engine, &mut session, "CREATE TABLE b (x INT)");
    exec(engine, &mut session, "CREATE TABLE p (x INT, y TEXT)");
    exec(engine, &mut session, "CREATE TABLE q (x INT, y TEXT)");

    // `a`: values 0..10 (each ~12×) plus a NULL every 17th row → duplicates + NULL membership.
    for i in 0..120 {
        let x = if i % 17 == 0 {
            "NULL".to_owned()
        } else {
            (i % 10).to_string()
        };
        exec(engine, &mut session, &format!("INSERT INTO a VALUES ({x})"));
    }
    // `b`: values 0..6 (each ~15×) plus a NULL every 13th row → partial overlap with `a`.
    for i in 0..90 {
        let x = if i % 13 == 0 {
            "NULL".to_owned()
        } else {
            (i % 6).to_string()
        };
        exec(engine, &mut session, &format!("INSERT INTO b VALUES ({x})"));
    }
    // `p`/`q`: two-column rows, overlapping with differing multiplicities.
    for i in 0..80 {
        let (px, py) = (i % 5, format!("'t{}'", i % 3));
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO p VALUES ({px}, {py})"),
        );
        let (qx, qy) = (i % 4, format!("'t{}'", i % 2));
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO q VALUES ({qx}, {qy})"),
        );
    }

    let queries = [
        "SELECT x FROM a UNION SELECT x FROM b",
        "SELECT x FROM a UNION ALL SELECT x FROM b",
        "SELECT x FROM a INTERSECT SELECT x FROM b",
        "SELECT x FROM a INTERSECT ALL SELECT x FROM b",
        "SELECT x FROM a EXCEPT SELECT x FROM b",
        "SELECT x FROM a EXCEPT ALL SELECT x FROM b",
        // Nested, left-associative chain: (a ∪ b) ∩ b.
        "SELECT x FROM a UNION SELECT x FROM b INTERSECT SELECT x FROM b",
        // Multi-column multiset operators.
        "SELECT x, y FROM p INTERSECT ALL SELECT x, y FROM q",
        "SELECT x, y FROM p EXCEPT ALL SELECT x, y FROM q",
        "SELECT x, y FROM p UNION SELECT x, y FROM q",
    ];

    for sql in queries {
        set_spill_config(None);
        let baseline = rows(engine, &mut session, sql);
        assert!(
            !baseline.is_empty(),
            "expected non-empty baseline for: {sql}"
        );

        let dir = std::env::temp_dir();
        // Generous budget → each operand sorts in memory (the sorted-merge still runs).
        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 8 * 1024 * 1024,
        }));
        assert_eq!(
            rows(engine, &mut session, sql),
            baseline,
            "spilling set-op (fits budget) must match in-memory: {sql}"
        );

        // Tiny budget → operands spill to many sorted runs, k-way merged + combined run-by-run.
        set_spill_config(Some(SpillConfig {
            dir,
            threshold_bytes: 64,
        }));
        assert_eq!(
            rows(engine, &mut session, sql),
            baseline,
            "spilling set-op (disk runs merged) must match in-memory: {sql}"
        );
    }

    set_spill_config(None);
}
