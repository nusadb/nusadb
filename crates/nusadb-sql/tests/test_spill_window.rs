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
    // Navigation that reads one row chosen per current row: an offset or position that varies
    // per row, and frames of any width.
    "SELECT id, lag(v, id % 5) OVER (ORDER BY id), lead(v, x % 4, -1) OVER (PARTITION BY p ORDER BY id) FROM w",
    "SELECT id, lag(v, x - 25, -9) OVER (ORDER BY x), lead(v, v) OVER (ORDER BY id) FROM w",
    "SELECT id, nth_value(v, id % 6 + 1) OVER (ORDER BY id), nth_value(v, x % 3) OVER (ORDER BY x) FROM w",
    "SELECT id, nth_value(v, id % 4 + 1) OVER (), nth_value(id, p + 1) OVER (PARTITION BY p) FROM w",
    "SELECT id, nth_value(v, 3) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 3 FOLLOWING) FROM w",
    "SELECT id, nth_value(v, 1500) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND 2000 FOLLOWING) FROM w",
    "SELECT id, nth_value(v, p + 1) OVER (PARTITION BY p ORDER BY x ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, nth_value(v, 2) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, first_value(id) OVER (ORDER BY x GROUPS BETWEEN CURRENT ROW AND CURRENT ROW), last_value(id) OVER (ORDER BY x DESC RANGE BETWEEN CURRENT ROW AND CURRENT ROW) FROM w",
    "SELECT id, last_value(v) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND 1000 PRECEDING) FROM w",
    "SELECT id, first_value(v) OVER (ORDER BY id ROWS BETWEEN 1000 FOLLOWING AND 2500 FOLLOWING) FROM w",
    // Aggregates over sliding ROWS frames of any width, added and removed as the frame moves.
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, count(v) OVER (ORDER BY x ROWS BETWEEN 1500 PRECEDING AND 1500 FOLLOWING), count(*) OVER (ORDER BY id ROWS BETWEEN 1000 FOLLOWING AND 2500 FOLLOWING) FROM w",
    "SELECT id, avg(v) OVER (PARTITION BY p ORDER BY id ROWS BETWEEN 300 PRECEDING AND 200 FOLLOWING) FROM w",
    "SELECT id, sum(v::NUMERIC(12, 2)) OVER (ORDER BY id ROWS BETWEEN 900 PRECEDING AND 3 PRECEDING) FROM w",
    "SELECT id, max(id) OVER (ORDER BY id ROWS BETWEEN 2500 PRECEDING AND 10 FOLLOWING), min(id) OVER (ORDER BY id DESC ROWS BETWEEN 2500 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, min(v) OVER (ORDER BY x ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING), max(v) OVER (ORDER BY v ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) FROM w",
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
    // Frames to the partition's end, computed backwards.
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, count(v) OVER (ORDER BY x DESC GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING), min(v) OVER (ORDER BY x DESC GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING), max(v) OVER (ORDER BY x DESC GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, avg(v) OVER (PARTITION BY p ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
];

/// `RANGE` and `GROUPS` frames: served from disk while each frame fits the budget.
const PEER: &[&str] = &[
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN 3 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN 2 PRECEDING AND 2 FOLLOWING) FROM w",
    "SELECT id, count(*) OVER (ORDER BY x DESC RANGE BETWEEN 1 PRECEDING AND 3 FOLLOWING) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY x GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x GROUPS BETWEEN 2 PRECEDING AND 1 PRECEDING) FROM w",
    "SELECT id, count(*) OVER (ORDER BY x GROUPS BETWEEN 1 FOLLOWING AND 2 FOLLOWING) FROM w",
    "SELECT id, count(*) OVER (ORDER BY x DESC GROUPS BETWEEN 2 FOLLOWING AND 3 FOLLOWING) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x RANGE BETWEEN CURRENT ROW AND CURRENT ROW EXCLUDE CURRENT ROW) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING EXCLUDE GROUP) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY x RANGE BETWEEN 2 PRECEDING AND CURRENT ROW EXCLUDE TIES) FROM w",
    "SELECT id, first_value(id) OVER (ORDER BY x RANGE BETWEEN 2 PRECEDING AND 2 FOLLOWING) FROM w",
    "SELECT id, last_value(id) OVER (ORDER BY x GROUPS BETWEEN CURRENT ROW AND 1 FOLLOWING) FROM w",
    "SELECT id, nth_value(id, 3) OVER (ORDER BY x RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, sum(id) OVER (ORDER BY v RANGE BETWEEN 50 PRECEDING AND 50 FOLLOWING) FROM w",
    "SELECT id, count(*) OVER (ORDER BY v DESC RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) FROM w",
    "SELECT id, max(v) OVER (PARTITION BY p ORDER BY x RANGE BETWEEN 4 PRECEDING AND 1 FOLLOWING) FROM w",
];

/// Frames that need the whole partition in memory: served when partitions fit, refused when not.
const PARTITION_BOUND: &[&str] = &[
    "SELECT id, sum(x) OVER (ORDER BY x RANGE BETWEEN UNBOUNDED PRECEDING AND 2 FOLLOWING) FROM w",
    "SELECT id, sum(x::FLOAT8) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM w",
    "SELECT id, sum(x) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) FROM w",
    // A wide frame of an aggregate with no exact sliding form still holds its rows.
    "SELECT id, sum(x::FLOAT8) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND CURRENT ROW) FROM w",
    // A MIN over values that only rise keeps every one of them.
    "SELECT id, min(id) OVER (ORDER BY id ROWS BETWEEN 2000 PRECEDING AND 2000 FOLLOWING) FROM w",
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

    for sql in STREAMABLE
        .iter()
        .chain(SLIDING)
        .chain(PEER)
        .chain(PARTITION_BOUND)
    {
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
        // A RANGE / GROUPS frame spans dozens of rows here, so it needs a little more room.
        spill(if PEER.contains(sql) { 256 * 1024 } else { 2048 });
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

    edge_values(engine, &mut session);

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

/// Values that compare equal but differ, and RANGE offsets that overflow at the ends of BIGINT
/// (see the call site).
fn edge_values(engine: &'static BtreeEngine, session: &mut Session) {
    // Values that compare equal but differ (`1.0` / `1.00`, `1 day` / `24 hours`): MIN and MAX keep
    // the earliest, also when a frame to the partition's end is computed backwards. And RANGE
    // offsets that overflow at the ends of BIGINT: a FOLLOWING start past every value leaves the
    // frame empty, a PRECEDING end before every value too.
    run(
        engine,
        session,
        "CREATE TABLE e (id INT, n NUMERIC, iv INTERVAL, b BIGINT, b2 BIGINT)",
    );
    let edge = (0..3000_i64)
        .map(|i| {
            let n = if i % 2 == 0 { "1.0" } else { "1.00" };
            let iv = if i % 2 == 0 { "1 day" } else { "24 hours" };
            let b = match i {
                0..10 => format!("{}", i64::MIN + i),
                2990.. => format!("{}", i64::MAX - (2999 - i)),
                _ => i.to_string(),
            };
            // `b2` is `b` with a NULL every 11th row, so NULLs sit beside the overflowing ends.
            let b2 = if i % 11 == 0 {
                "NULL".to_owned()
            } else {
                b.clone()
            };
            format!("({i}, {n}, '{iv}', {b}, {b2})")
        })
        .collect::<Vec<_>>()
        .join(",");
    run(engine, session, &format!("INSERT INTO e VALUES {edge}"));
    for sql in [
        "SELECT id, min(n) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM e",
        "SELECT id, max(iv) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM e",
        "SELECT id, max(n) OVER (ORDER BY id % 7 GROUPS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM e",
        "SELECT id, sum(id) OVER (ORDER BY b RANGE BETWEEN 3 FOLLOWING AND 5 FOLLOWING) FROM e",
        "SELECT id, count(*) OVER (ORDER BY b RANGE BETWEEN 5 PRECEDING AND 2 PRECEDING) FROM e",
        "SELECT id, count(*) OVER (ORDER BY b DESC RANGE BETWEEN 2 FOLLOWING AND 4 FOLLOWING) FROM e",
        "SELECT id, count(*) OVER (ORDER BY b2 NULLS FIRST RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) FROM e",
        "SELECT id, count(*) OVER (ORDER BY b2 RANGE BETWEEN CURRENT ROW AND 5 FOLLOWING) FROM e",
        "SELECT id, sum(id) OVER (ORDER BY b2 DESC NULLS LAST RANGE BETWEEN 3 PRECEDING AND 1 FOLLOWING) FROM e",
    ] {
        set_spill_config(None);
        let want = buffered(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        spill(256 * 1024);
        let got = streamed(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        // Compared by debug form: `1.0` and `1.00` are equal values, but which one a row gets
        // is the point.
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "{sql}");
    }
    set_spill_config(None);
    let last = buffered(
        engine,
        session,
        "SELECT id, count(*) OVER (ORDER BY b RANGE BETWEEN 3 FOLLOWING AND 5 FOLLOWING) FROM e",
    )
    .unwrap();
    for row in &last {
        if let [Value::Int(id), Value::Int(count)] = row.as_slice()
            && *id >= 2997
        {
            assert_eq!(
                *count, 0,
                "the frame past the largest value is empty (id {id})"
            );
        }
    }
    // An overflowing PRECEDING start is the first row with a value, not the NULLs before it.
    let first = buffered(
        engine,
        session,
        "SELECT id, count(*) OVER (ORDER BY b2 NULLS FIRST RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) \
         FROM e",
    )
    .unwrap();
    assert!(
        first
            .iter()
            .any(|r| matches!(r.as_slice(), [Value::Int(1), Value::Int(1)])),
        "the smallest value's frame holds only itself"
    );
}
