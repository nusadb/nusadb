//! `WHERE <range on an indexed column> ORDER BY <that column> LIMIT n` is served by an ordered
//! index scan that stops at the limit, and it returns exactly the rows a full sort would.

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
                key_exprs: Vec::new(),
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
        "CREATE TABLE t (id INT PRIMARY KEY, k INT NOT NULL, v TEXT)",
    );
    run(&engine, "CREATE INDEX t_k ON t (k)");
    // Insert out of key order so the scan order is the index's, not the insertion's.
    for i in (0..1000).rev() {
        run(
            &engine,
            &format!("INSERT INTO t VALUES ({i}, {}, 'r{i}')", (i * 7) % 1000),
        );
    }
    engine
}

#[test]
fn a_range_with_order_by_and_limit_on_the_indexed_column_stops_at_the_limit() {
    let engine = fixture();
    let plan = text(
        &engine,
        "EXPLAIN SELECT k FROM t WHERE k BETWEEN 100 AND 900 ORDER BY k DESC LIMIT 5 OFFSET 2",
    );
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    assert!(!plan.contains("Sort"), "the Sort must be gone:\n{plan}");
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k BETWEEN 100 AND 900 ORDER BY k DESC LIMIT 5 OFFSET 2"
        ),
        vec![898, 897, 896, 895, 894]
    );
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k > 500 ORDER BY k ASC LIMIT 3"
        ),
        vec![501, 502, 503]
    );
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k >= 990 AND k < 995 ORDER BY k LIMIT 10"
        ),
        vec![990, 991, 992, 993, 994]
    );
}

#[test]
fn a_range_the_index_does_not_answer_exactly_still_returns_the_right_rows() {
    let engine = fixture();
    // Two lower bounds: the index keeps the looser one, so the scan must not be capped.
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k > 100 AND k > 500 ORDER BY k LIMIT 3"
        ),
        vec![501, 502, 503]
    );
    // A conjunct on another column filters after the bounds: the first three survivors of the
    // full sorted result, which a wrongly capped scan would miss.
    let full = ints(
        &engine,
        "SELECT k FROM t WHERE k > 100 AND id % 2 = 0 ORDER BY k",
    );
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k > 100 AND id % 2 = 0 ORDER BY k LIMIT 3"
        ),
        full[..3].to_vec()
    );
}

#[test]
fn a_backward_capped_scan_honours_excluded_bounds() {
    let engine = fixture();
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k > 100 AND k < 200 ORDER BY k DESC LIMIT 3"
        ),
        vec![199, 198, 197]
    );
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k < 50 ORDER BY k DESC LIMIT 2"
        ),
        vec![49, 48]
    );
    assert_eq!(
        ints(
            &engine,
            "SELECT k FROM t WHERE k >= 998 ORDER BY k DESC LIMIT 5"
        ),
        vec![999, 998]
    );
}

#[test]
fn a_range_that_can_hold_no_key_is_empty_not_a_crash() {
    let engine = fixture();
    for sql in [
        "SELECT k FROM t WHERE k > 500 AND k < 100",
        "SELECT k FROM t WHERE k BETWEEN 50 AND 10",
        "SELECT k FROM t WHERE k > 5 AND k < 5",
        "SELECT k FROM t WHERE k >= 5 AND k < 5 ORDER BY k LIMIT 3",
        "SELECT k FROM t WHERE k > 500 AND k < 100 ORDER BY k DESC LIMIT 3",
    ] {
        assert_eq!(ints(&engine, sql), Vec::<i64>::new(), "{sql}");
    }
}

#[test]
fn a_wide_range_under_statistics_still_takes_the_capped_ordered_scan() {
    let engine = fixture();
    // The range keeps almost every row, so the cost gate would leave a full read on the
    // sequential scan; with ORDER BY on the same column and a LIMIT, the capped index scan
    // reads only the limit and wins regardless.
    run(&engine, "ANALYZE t");
    let sql = "SELECT k FROM t WHERE k BETWEEN 1 AND 998 ORDER BY k DESC LIMIT 7";
    let plan = text(&engine, &format!("EXPLAIN {sql}"));
    assert!(plan.contains("IndexScan: t using t_k"), "{plan}");
    assert!(!plan.contains("Sort"), "{plan}");
    let mut expected: Vec<i64> = (0..1000).map(|i| (i * 7) % 1000).collect();
    expected.retain(|k| (1..=998).contains(k));
    expected.sort_unstable_by(|a, b| b.cmp(a));
    expected.truncate(7);
    assert_eq!(ints(&engine, sql), expected);
}
