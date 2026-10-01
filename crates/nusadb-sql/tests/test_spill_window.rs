//! Window functions with spill-to-disk enabled must give every row exactly the value the in-memory
//! window gives it, whether each partition fits the budget (evaluated in memory) or not (written to
//! disk and evaluated as it streams back). A frame the streamed evaluation cannot serve over a
//! partition larger than the budget must fail with the budget error, and still work when the
//! partition fits.
//!
//! Rows are compared as multisets: a spilled window emits its rows partition by partition rather
//! than in input order, which SQL leaves unspecified without an `ORDER BY`. Every row carries its
//! unique `id`, so a multiset match is a per-row match.
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

/// The rows of `sql` through the streaming path, as a sorted multiset.
fn streamed(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<Vec<Vec<Value>>, Error> {
    let mut sink = Collect::default();
    session.execute_streaming(planned(engine, sql), &mut sink)?;
    let mut rows = sink.0;
    rows.sort_by_key(|r| format!("{r:?}"));
    Ok(rows)
}

/// The rows of `sql` through the buffered path, as a sorted multiset.
fn buffered(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<Vec<Vec<Value>>, Error> {
    let ExecutionResult::Rows { mut rows, .. } = session.execute(planned(engine, sql))? else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    Ok(rows)
}

fn spill(threshold_bytes: usize) {
    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes,
    }));
}

/// Window queries the streamed evaluation serves over any partition size.
const STREAMABLE: &[&str] = &[
    // Ranking and distribution, with and without ties and partitions.
    "SELECT id, row_number() OVER (ORDER BY id) FROM w",
    "SELECT id, row_number() OVER (ORDER BY x) FROM w",
    "SELECT id, row_number() OVER () FROM w",
    "SELECT id, rank() OVER (ORDER BY x), dense_rank() OVER (ORDER BY x) FROM w",
    "SELECT id, percent_rank() OVER (ORDER BY x), cume_dist() OVER (ORDER BY x) FROM w",
    "SELECT id, rank() OVER (PARTITION BY p ORDER BY x DESC) FROM w",
    "SELECT id, cume_dist() OVER (PARTITION BY p ORDER BY v NULLS FIRST) FROM w",
    "SELECT id, ntile(7) OVER (ORDER BY id), ntile(3) OVER (PARTITION BY p ORDER BY x) FROM w",
    // Navigation.
    "SELECT id, lag(v) OVER (ORDER BY id), lead(v) OVER (ORDER BY id) FROM w",
    "SELECT id, lag(v, 3, -1) OVER (ORDER BY id), lead(v, 5, -2) OVER (PARTITION BY p ORDER BY id) FROM w",
    "SELECT id, lag(v, 0) OVER (ORDER BY id), lead(v, -2) OVER (ORDER BY id) FROM w",
    "SELECT id, lag(v, 10000) OVER (ORDER BY id) FROM w",
    "SELECT id, lag(v, 2000, -3) OVER (ORDER BY id), lead(v, -1500) OVER (ORDER BY x) FROM w",
    "SELECT id, first_value(v) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, first_value(v) OVER (ORDER BY x), last_value(v) OVER (ORDER BY x) FROM w",
    "SELECT id, nth_value(v, 5) OVER (ORDER BY x), nth_value(v, 1) OVER (PARTITION BY p) FROM w",
    "SELECT id, first_value(v) OVER (ORDER BY id ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, last_value(v) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) FROM w",
    "SELECT id, last_value(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, last_value(v) OVER (PARTITION BY p) FROM w",
    "SELECT id, nth_value(v, 2) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w",
    // Aggregates over frames that start at the partition.
    "SELECT id, sum(x) OVER (), count(*) OVER (PARTITION BY p), avg(v) OVER (PARTITION BY p) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id), sum(x) OVER (ORDER BY x) FROM w",
    "SELECT id, sum(v) OVER (PARTITION BY p ORDER BY x), max(v) OVER (PARTITION BY p ORDER BY x) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 2 FOLLOWING) FROM w",
    "SELECT id, min(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, count(v) OVER (ORDER BY x RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, count(*) FILTER (WHERE x > 10) OVER (PARTITION BY p ORDER BY id) FROM w",
    // Several windows with different partitionings in one query.
    "SELECT id, row_number() OVER (ORDER BY x), sum(id) OVER (PARTITION BY p ORDER BY id), lag(x) OVER (ORDER BY id) FROM w",
    // A window over a filtered input, and under an ORDER BY.
    "SELECT id, rank() OVER (PARTITION BY p ORDER BY x) FROM w WHERE id % 3 = 0",
    "SELECT id, row_number() OVER (ORDER BY x) AS rn FROM w ORDER BY rn LIMIT 50",
];

/// Sliding `ROWS` frames: served from disk while each frame fits the budget.
const SLIDING: &[&str] = &[
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) FROM w",
    "SELECT id, avg(v) OVER (PARTITION BY p ORDER BY id ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN 5 PRECEDING AND 2 PRECEDING) FROM w",
    "SELECT id, count(*) OVER (ORDER BY id ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING EXCLUDE CURRENT ROW) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING EXCLUDE GROUP) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING EXCLUDE TIES) FROM w",
    "SELECT id, max(v) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 4 FOLLOWING) FROM w",
    "SELECT id, nth_value(v, 2) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, first_value(v) OVER (ORDER BY id ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING) FROM w",
    "SELECT id, last_value(v) OVER (ORDER BY id ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) FROM w",
    "SELECT id, min(v) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w",
];

/// Frames that need the whole partition in memory: served when partitions fit, refused when not.
const PARTITION_BOUND: &[&str] = &[
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN 3 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY x GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
];

#[test]
fn spilled_windows_match_the_in_memory_window() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(
        engine,
        &mut session,
        "CREATE TABLE w (id INT, p INT, x INT, v INT)",
    );
    for start in (0..3000).step_by(500) {
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

    for sql in STREAMABLE.iter().chain(SLIDING).chain(PARTITION_BOUND) {
        set_spill_config(None);
        let want = buffered(engine, &mut session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert!(!want.is_empty(), "{sql}");
        // 1 MiB: every partition fits and is evaluated in memory.
        spill(1 << 20);
        for (path, got) in [
            ("streamed", streamed(engine, &mut session, sql)),
            ("buffered", buffered(engine, &mut session, sql)),
        ] {
            assert_eq!(
                got.unwrap_or_else(|e| panic!("{sql} ({path}): {e}")),
                want,
                "{sql} ({path}, in memory)"
            );
        }
        // 2 KiB: every partition is evaluated from disk, while a frame of a few rows still fits.
        spill(2048);
        let from_disk = streamed(engine, &mut session, sql);
        if PARTITION_BOUND.contains(sql) {
            let err = from_disk.expect_err(sql).to_string();
            assert!(
                err.contains("work_mem of 2048 bytes exceeded"),
                "{sql}: {err}"
            );
        } else {
            assert_eq!(
                from_disk.unwrap_or_else(|e| panic!("{sql}: {e}")),
                want,
                "{sql} (from disk)"
            );
            let buffered_disk = buffered(engine, &mut session, sql);
            assert_eq!(
                buffered_disk.unwrap_or_else(|e| panic!("{sql}: {e}")),
                want,
                "{sql} (buffered, from disk)"
            );
        }
    }
    set_spill_config(None);

    // No spill file outlives its query.
    let leftover = std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("nusadb-spill-window-{}-", std::process::id()))
        })
        .count();
    assert_eq!(leftover, 0, "spill files left behind");
}
