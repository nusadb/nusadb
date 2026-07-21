//! Holistic spill differential test (correctness net for the spill series).
//!
//! Each per-operator spill test (`test_spill_distinct`, `test_spill_setop`, `test_external_sort`,
//! `test_spill_agg`, `test_grace_join`) checks one operator in isolation. This drives a *broad mix*
//! of queries — multi-way joins (every kind), `GROUP BY ... HAVING`, `DISTINCT`, `ORDER BY`,
//! `UNION`/`INTERSECT`/`EXCEPT [ALL]`, scalar / `IN` / `EXISTS` subqueries, and nested combinations —
//! through the real analyze → plan → execute pipeline under three memory regimes:
//!
//! - spill **off** (the in-memory oracle),
//! - a **generous** budget (8 MiB — the in-memory fast path inside each spilling operator), and
//! - a **tiny** budget (64 B — forces sorted runs to disk and k-way merges).
//!
//! All three must return the same result for every query (a multiset, or an exact sequence when the
//! query has a total `ORDER BY`). A regression in any spilling operator — or an interaction between
//! two of them in one plan — that the isolated tests miss shows up here as a mismatch against the
//! spill-off oracle.
//!
//! `spill_config` is process-wide, so this is its own test binary and resets the config to `None`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "integration test harness asserts via unwrap/panic; one linear query script"
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

/// Result rows as a multiset (sorted by debug form) — unordered queries promise no row order, and
/// the spilling path may emit in a different order than the in-memory path.
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn diverse_queries_match_across_spill_budgets_then_reset() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    // Two related tables with duplicates, NULLs, and partial key overlap, plus a small dimension —
    // enough rows that the 64-byte budget spills every blocking operator to disk.
    exec(
        engine,
        &mut session,
        "CREATE TABLE orders (id INT, cust INT, region TEXT, amount INT)",
    );
    exec(
        engine,
        &mut session,
        "CREATE TABLE refunds (id INT, cust INT, region TEXT, amount INT)",
    );
    exec(
        engine,
        &mut session,
        "CREATE TABLE customers (cust INT, tier TEXT)",
    );

    for i in 0..240 {
        let cust = if i % 19 == 0 {
            "NULL".to_owned()
        } else {
            (i % 12).to_string()
        };
        let region = format!("'r{}'", i % 4);
        let amount = i % 7;
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO orders VALUES ({i}, {cust}, {region}, {amount})"),
        );
    }
    for i in 0..90 {
        let cust = (i % 8).to_string();
        let region = format!("'r{}'", i % 3);
        let amount = i % 5;
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO refunds VALUES ({i}, {cust}, {region}, {amount})"),
        );
    }
    for c in 0..12 {
        let tier = if c % 3 == 0 { "'gold'" } else { "'silver'" };
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO customers VALUES ({c}, {tier})"),
        );
    }

    // A mix that exercises every spilling operator and combinations of them. Each entry is
    // (sql, ordered) — `ordered` queries have a total ORDER BY so the result sequence is
    // deterministic and is compared exactly; the rest are compared as multisets.
    let queries: &[(&str, bool)] = &[
        // GROUP BY + aggregates (+ HAVING) — sort-based spilling aggregate.
        (
            "SELECT cust, COUNT(*), SUM(amount), MIN(amount), MAX(amount) FROM orders GROUP BY cust",
            false,
        ),
        (
            "SELECT region, COUNT(*) FROM orders GROUP BY region HAVING COUNT(*) > 50",
            false,
        ),
        (
            "SELECT cust, region, SUM(amount) FROM orders GROUP BY cust, region",
            false,
        ),
        // DISTINCT — sort-based spilling distinct.
        ("SELECT DISTINCT cust, region FROM orders", false),
        ("SELECT DISTINCT amount FROM orders", false),
        // ORDER BY (total order via unique id) — external merge sort, exact sequence.
        ("SELECT id, cust, amount FROM orders ORDER BY id", true),
        ("SELECT id FROM orders ORDER BY amount, region, id", true),
        // Joins of every kind — grace hash join.
        (
            "SELECT o.id, r.id FROM orders o INNER JOIN refunds r ON o.cust = r.cust",
            false,
        ),
        (
            "SELECT o.id, r.id FROM orders o LEFT JOIN refunds r ON o.cust = r.cust",
            false,
        ),
        (
            "SELECT o.id, r.id FROM orders o RIGHT JOIN refunds r ON o.cust = r.cust",
            false,
        ),
        (
            "SELECT o.id, r.id FROM orders o FULL JOIN refunds r ON o.cust = r.cust",
            false,
        ),
        // Set operations [ALL] — streaming sorted-merge spill.
        (
            "SELECT cust, region FROM orders UNION SELECT cust, region FROM refunds",
            false,
        ),
        (
            "SELECT cust, region FROM orders UNION ALL SELECT cust, region FROM refunds",
            false,
        ),
        (
            "SELECT cust, region FROM orders INTERSECT SELECT cust, region FROM refunds",
            false,
        ),
        (
            "SELECT amount FROM orders INTERSECT ALL SELECT amount FROM refunds",
            false,
        ),
        (
            "SELECT cust, region FROM orders EXCEPT SELECT cust, region FROM refunds",
            false,
        ),
        (
            "SELECT amount FROM orders EXCEPT ALL SELECT amount FROM refunds",
            false,
        ),
        // Subqueries combined with the spilling operators.
        (
            "SELECT DISTINCT cust FROM orders WHERE cust IN (SELECT cust FROM refunds)",
            false,
        ),
        (
            "SELECT o.id FROM orders o WHERE EXISTS (SELECT 1 FROM refunds r WHERE r.cust = o.cust)",
            false,
        ),
        // Join feeding a GROUP BY (two spilling operators stacked in one plan).
        (
            "SELECT o.region, COUNT(*), SUM(o.amount) FROM orders o \
             INNER JOIN customers c ON o.cust = c.cust GROUP BY o.region",
            false,
        ),
        // DISTINCT over a join result.
        (
            "SELECT DISTINCT c.tier, o.region FROM orders o INNER JOIN customers c ON o.cust = c.cust",
            false,
        ),
    ];

    let dir = std::env::temp_dir();
    for (sql, ordered) in queries {
        set_spill_config(None);
        let want = order_aware(engine, &mut session, sql, *ordered);
        assert!(!want.is_empty(), "expected non-empty oracle for: {sql}");

        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 8 * 1024 * 1024,
        }));
        assert_eq!(
            order_aware(engine, &mut session, sql, *ordered),
            want,
            "spill (8 MiB, in-memory fast path) must match the oracle: {sql}"
        );

        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 64,
        }));
        assert_eq!(
            order_aware(engine, &mut session, sql, *ordered),
            want,
            "spill (64 B, disk runs merged) must match the oracle: {sql}"
        );
    }

    set_spill_config(None);
}

/// Rows compared as an exact sequence for a totally-ordered query, else as a sorted multiset.
fn order_aware(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
    ordered: bool,
) -> Vec<Row> {
    if ordered {
        let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
            panic!("expected rows from: {sql}");
        };
        rows
    } else {
        rows(engine, session, sql)
    }
}
