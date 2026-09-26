//! A `WHERE` conjunct pushed onto one side of a join is served from that table's index when
//! one maps onto it, and the join's result is exactly what the unindexed plan produces.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, analyze, execute, parse, plan};

/// A catalog that reports the table's plain secondary indexes, as the server's does.
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn table_stats(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(None);
        };
        self.0.table_stats(schema.id).map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        let backing: std::collections::HashSet<_> = self
            .0
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.0.list_indexes(schema.id)? {
            if self
                .0
                .lookup_index(&def.name)?
                .is_some_and(|id| backing.contains(&id))
                || !def.key_exprs.is_empty()
                || def.predicate.is_some()
            {
                continue;
            }
            out.push(IndexInfo {
                name: def.name,
                columns: def.columns,
                unique: def.unique,
            });
        }
        Ok(out)
    }
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &EngineCatalog(engine))
        .unwrap_or_else(|e| panic!("`{sql}` should analyze: {e}"));
    execute(plan(logical), engine).unwrap_or_else(|e| panic!("`{sql}` should succeed: {e}"))
}

fn ints(engine: &BtreeEngine, sql: &str) -> Vec<i64> {
    let ExecutionResult::Rows { rows, .. } = run(engine, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.into_iter()
        .map(|r| match r.first() {
            Some(Value::Int(n)) => *n,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

fn text(engine: &BtreeEngine, sql: &str) -> String {
    let ExecutionResult::Rows { rows, .. } = run(engine, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.into_iter()
        .filter_map(|r| match r.into_iter().next() {
            Some(Value::Text(s)) => Some(s),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, k INT NOT NULL)",
    );
    run(&engine, "CREATE INDEX t_k ON t (k)");
    for i in 0..2000 {
        run(&engine, &format!("INSERT INTO t VALUES ({i}, {})", i % 50));
    }
    engine
}

/// The rows of `t` as `(id, k)`, the oracle every expectation below is computed from.
fn rows() -> Vec<(i64, i64)> {
    (0..2000).map(|i| (i, i % 50)).collect()
}

#[test]
fn a_join_side_equality_uses_the_index_and_matches_the_oracle() {
    let engine = fixture();
    let plan = text(
        &engine,
        "EXPLAIN SELECT count(*) FROM t a JOIN t b ON a.id = b.id WHERE b.k = 7",
    );
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    let expected = i64::try_from(rows().iter().filter(|(_, k)| *k == 7).count()).unwrap();
    assert_eq!(
        ints(
            &engine,
            "SELECT count(*) FROM t a JOIN t b ON a.id = b.id WHERE b.k = 7"
        ),
        vec![expected]
    );
}

#[test]
fn a_base_side_range_uses_the_index_and_matches_the_oracle() {
    let engine = fixture();
    let sql = "SELECT sum(b.id) FROM t a JOIN t b ON a.id = b.id WHERE a.k BETWEEN 3 AND 5";
    let plan = text(&engine, &format!("EXPLAIN {sql}"));
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    let expected: i64 = rows()
        .iter()
        .filter(|(_, k)| (3..=5).contains(k))
        .map(|(id, _)| id)
        .sum();
    assert_eq!(ints(&engine, sql), vec![expected]);
}

#[test]
fn both_sides_narrowed_still_agree_with_the_oracle() {
    let engine = fixture();
    let sql =
        "SELECT count(*) FROM t a JOIN t b ON a.k = b.k WHERE a.k = 7 AND b.k = 7 AND b.id < 100";
    let plan = text(&engine, &format!("EXPLAIN {sql}"));
    assert_eq!(plan.matches("IndexScan: t using t_k").count(), 2, "{plan}");
    let left = i64::try_from(rows().iter().filter(|(_, k)| *k == 7).count()).unwrap();
    let right =
        i64::try_from(rows().iter().filter(|(id, k)| *k == 7 && *id < 100).count()).unwrap();
    assert_eq!(ints(&engine, sql), vec![left * right]);
}

#[test]
fn an_outer_joins_null_extended_side_is_never_narrowed() {
    let engine = fixture();
    // `b.k = 7` sits above the LEFT join (b may be NULL-extended), so b keeps its full scan, and
    // the count is the rows whose match has k = 7: with a shifted join key nothing matches.
    let sql = "SELECT count(*) FROM t a LEFT JOIN t b ON a.id = b.id + 5000 WHERE b.k = 7";
    let plan = text(&engine, &format!("EXPLAIN {sql}"));
    assert!(!plan.contains("IndexScan"), "{plan}");
    assert_eq!(ints(&engine, sql), vec![0]);
}

#[test]
fn with_statistics_a_bound_that_keeps_most_rows_stays_a_full_scan_on_the_join_side() {
    let engine = fixture();
    // `b.k >= 0` keeps every row. Without statistics the heuristic takes the index; once the
    // table is analyzed the cost gate sees a bound that keeps the whole table and leaves the
    // sequential scan in place, since a random walk through the index would cost more.
    let wide = "SELECT count(*) FROM t a JOIN t b ON a.k = b.k WHERE a.id = 57 AND b.k >= 0";
    let plan = text(&engine, &format!("EXPLAIN {wide}"));
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    run(&engine, "ANALYZE t");
    let plan = text(&engine, &format!("EXPLAIN {wide}"));
    assert!(!plan.contains("IndexScan"), "{plan}");
    // Row 57 has k = 7 and meets every b row with k = 7.
    let with_k7 = i64::try_from(rows().iter().filter(|(_, k)| *k == 7).count()).unwrap();
    assert_eq!(ints(&engine, wide), vec![with_k7]);
    // A selective point bound on the same side still takes its index under statistics.
    let narrow = "SELECT count(*) FROM t a JOIN t b ON a.k = b.k WHERE a.id = 57 AND b.k = 7";
    let plan = text(&engine, &format!("EXPLAIN {narrow}"));
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    assert_eq!(ints(&engine, narrow), vec![with_k7]);
}

/// A CTE that resolves under its own name (recursive, or materialized for volatility) is a
/// working set, not the table it may shadow: it is never scanned through that table's index, as
/// a join input, as a base under a join, or alone.
#[test]
fn a_cte_shadowing_an_indexed_table_is_never_scanned_through_that_index() {
    let engine = fixture();
    run(
        &engine,
        "CREATE TABLE s (id INT PRIMARY KEY, k INT NOT NULL)",
    );
    for i in 0..5 {
        run(&engine, &format!("INSERT INTO s VALUES ({i}, {})", i % 2));
    }
    let recursive_base = "WITH RECURSIVE t(id, k) AS (SELECT 100, 1 UNION ALL SELECT id + 1, k \
                          FROM t WHERE id < 102) SELECT t.id FROM t JOIN s ON s.k = t.k WHERE t.k = 1 \
                          ORDER BY t.id";
    let plan = text(&engine, &format!("EXPLAIN {recursive_base}"));
    assert!(!plan.contains("IndexScan"), "{plan}");
    // s has two rows with k = 1, so each generated id appears twice.
    assert_eq!(
        ints(&engine, recursive_base),
        vec![100, 100, 101, 101, 102, 102]
    );
    let recursive_alone = "WITH RECURSIVE t(id, k) AS (SELECT 100, 1 UNION ALL SELECT id + 1, k \
                           FROM t WHERE id < 102) SELECT t.id FROM t WHERE t.k = 1 ORDER BY t.id";
    let plan = text(&engine, &format!("EXPLAIN {recursive_alone}"));
    assert!(!plan.contains("IndexScan"), "{plan}");
    assert_eq!(ints(&engine, recursive_alone), vec![100, 101, 102]);
    let materialized_join = "WITH t AS (SELECT id + 1000 AS id, k, random() AS r FROM s) \
                             SELECT b.id FROM t a JOIN t b ON a.id = b.id WHERE b.k = 1 ORDER BY b.id";
    let plan = text(&engine, &format!("EXPLAIN {materialized_join}"));
    assert!(!plan.contains("IndexScan"), "{plan}");
    assert_eq!(ints(&engine, materialized_join), vec![1001, 1003]);
}
