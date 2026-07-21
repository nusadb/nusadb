//! Grace hash join: an equi-join (INNER / LEFT / RIGHT / FULL) with spill-to-disk enabled
//! must return exactly the same rows as the in-memory hash join, whether the build side fits in the
//! budget or overflows it and partitions to disk (including the NULL-key null bucket).
//!
//! `spill_config` is a process-wide static, so the whole scenario lives in ONE test (this file is its
//! own test binary, run sequentially) and resets the config to `None` before returning, so no other
//! test observes spill enabled.

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
    set_spill_config, set_work_mem,
};

/// Minimal analyzer catalog over the engine's latest-committed schema.
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

/// Run `sql` and return its rows as a multiset (sorted by debug form), since neither the in-memory
/// nor the grace-partitioned join promises a row order for an unordered query.
fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn grace_join_matches_in_memory_for_every_kind_then_resets() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE cust (id INT NOT NULL, name TEXT)",
    );
    exec(
        engine,
        &mut session,
        "CREATE TABLE ord (uid INT, amount INT)",
    );
    // 40 customers (ids 0..40). Orders reference only EVEN uids 0..48, so the data exercises every
    // outer case: odd-id customers have no order (unmatched left), order uids 42..48 have no customer
    // (unmatched right), and even ids 0..40 match — spread across hash partitions + the null bucket.
    for i in 0..40 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO cust VALUES ({i}, 'c{i}')"),
        );
    }
    for i in 0..120 {
        let uid = (i % 25) * 2; // even uids 0..48
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO ord VALUES ({uid}, {})", i * 10),
        );
    }
    // A NULL-keyed row on each side → must land in the null bucket (never matches; surfaces only for
    // the outer side that keeps unmatched rows).
    exec(engine, &mut session, "INSERT INTO ord VALUES (NULL, 999)");
    exec(
        engine,
        &mut session,
        "INSERT INTO cust VALUES (999, 'lonely')",
    );

    let dir = std::env::temp_dir();
    for kind in ["INNER", "LEFT", "RIGHT", "FULL"] {
        let sql =
            format!("SELECT cust.id, ord.amount FROM cust {kind} JOIN ord ON cust.id = ord.uid");

        // Oracle: spill disabled → the in-memory hash join.
        set_spill_config(None);
        let baseline = rows(engine, &mut session, &sql);
        assert!(!baseline.is_empty(), "{kind} join should produce rows");

        // Build side fits a generous budget → grace join's in-memory fast path.
        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 8 * 1024 * 1024,
        }));
        assert_eq!(
            rows(engine, &mut session, &sql),
            baseline,
            "{kind}: grace join (build fits budget) must match in-memory"
        );

        // Tiny budget → the build overflows and both inputs partition to disk (+ null bucket).
        set_spill_config(Some(SpillConfig {
            dir: dir.clone(),
            threshold_bytes: 64,
        }));
        assert_eq!(
            rows(engine, &mut session, &sql),
            baseline,
            "{kind}: grace join (partitioned to disk) must match in-memory"
        );
    }

    // Reset so no other test observes spill enabled.
    set_spill_config(None);
}

/// Residual (QA): with spill configured (the server default) and a small
/// `work_mem`, a join whose PROBE side is far bigger than the budget — but whose build side is a
/// tiny dim table — must stream: `LIMIT` and aggregates over it hold O(build), never O(probe).
/// Before the fix the streaming arm was disabled whenever spill was on, so the materializing
/// path buffered the whole probe input and the stage tripped `work_mem` (QA: `orders(1M) JOIN
/// dim(100)` OOM even with `LIMIT 5`).
#[test]
fn big_probe_small_build_streams_under_work_mem() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE dim (id INT, label TEXT)",
    );
    exec(engine, &mut session, "CREATE TABLE fact (id INT, d INT)");
    for i in 0..100 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO dim VALUES ({i}, 'd{i}')"),
        );
    }
    // ~40k fact rows via batched multi-row inserts (fast enough, big enough to dwarf work_mem).
    for batch in 0..40 {
        let values: Vec<String> = (0..1000)
            .map(|j| {
                let n = batch * 1000 + j;
                format!("({n}, {})", n % 100)
            })
            .collect();
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO fact VALUES {}", values.join(", ")),
        );
    }

    // Spill ON (generous: the 100-row build fits) + a work_mem far below the fact table's
    // materialized size (~40k rows ≫ 64 KiB): any stage that buffers the probe input trips it.
    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes: 8 * 1024 * 1024,
    }));
    set_work_mem(64 * 1024);

    // LIMIT over the join: the exact QA repro shape.
    let got = rows(
        engine,
        &mut session,
        "SELECT fact.id, dim.label FROM fact JOIN dim ON fact.d = dim.id LIMIT 5",
    );
    assert_eq!(got.len(), 5, "LIMIT over a big-probe join must stream");

    // An aggregate over the join: folds the streamed output, O(build) memory.
    let got = rows(
        engine,
        &mut session,
        "SELECT count(*) FROM fact JOIN dim ON fact.d = dim.id",
    );
    assert_eq!(got.len(), 1);
    assert_eq!(
        format!("{:?}", got[0]),
        format!("{:?}", vec![nusadb_sql::ast::Value::Int(40_000)])
    );

    // Reversed join order (dim as probe): still fine — the budget bounds whichever side builds.
    let got = rows(
        engine,
        &mut session,
        "SELECT count(*) FROM dim JOIN fact ON dim.id = fact.d",
    );
    assert_eq!(
        format!("{:?}", got[0]),
        format!("{:?}", vec![nusadb_sql::ast::Value::Int(40_000)])
    );

    // Reset the process-wide knobs so no other test observes them.
    set_work_mem(0);
    set_spill_config(None);
}
