//! Manual scan-throughput probe (A-PERF.SCAN, QA finding `QA_agg_perf_simd_reverify_d7eed161.md`):
//! QA measured aggregate scans at ~1.47µs/row on the durable engine (the comparison ratios live
//! in that QA report) with the fold no longer the bottleneck. This probe splits the cost between the raw engine scan (tuple rehydration
//! plus MVCC materialization) and the SQL layer, on the durable engine where committed tuples are
//! lazy (the QA configuration).
//!
//! NOT a CI gate — `#[ignore]`d; run manually:
//! `cargo test -p nusadb-sql --release --test scan_bench -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    reason = "manual perf probe, not a CI gate"
)]

use std::time::Instant;

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, IndexInfo, Session, analyze, parse, plan};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Result<(), Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical)).map(|_| ())
}

#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn scan_throughput_probe() {
    const N: usize = 1_000_000;
    // A repo-adjacent scratch dir: the system temp dir can deny the WAL open on Windows.
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nusadb_scan_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let engine: &'static BtreeEngine =
        Box::leak(Box::new(BtreeEngine::open(dir.join("bench.wal")).unwrap()));
    let mut session = Session::new(engine);
    run(
        engine,
        &mut session,
        "CREATE TABLE t (id INT, val INT, grp INT)",
    )
    .unwrap();
    // A separate text table keeps the longitudinal numbers of `t` comparable across runs while
    // still probing the text decode path (R2 stage 2b).
    run(engine, &mut session, "CREATE TABLE txt (tag TEXT, val INT)").unwrap();

    let t0 = Instant::now();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({i},{},{})", i % 97, i % 10))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES {values}"),
        )
        .unwrap();
    }
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("('tag{}',{})", i % 25, i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO txt VALUES {values}"),
        )
        .unwrap();
    }
    println!("load 2x{N} rows: {:?}", t0.elapsed());

    let table = engine.lookup_table("t").unwrap().expect("table t").id;

    // Raw engine scan drain: tuple rehydration + MVCC materialization + per-row try_next.
    for label in ["raw cold", "raw warm1", "raw warm2"] {
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let t = Instant::now();
        let mut scan = engine.scan(txn, table).unwrap();
        let open = t.elapsed();
        let t2 = Instant::now();
        let mut n = 0usize;
        while scan.try_next().unwrap().is_some() {
            n += 1;
        }
        let dt = t.elapsed();
        println!(
            "{label}: {n} rows in {dt:?} ({:.0} ns/row; open {open:?} + drain {:?})",
            dt.as_nanos() as f64 / n as f64,
            t2.elapsed()
        );
        engine.commit(txn).unwrap();
    }

    // Full SQL layer on top of the same scan.
    for sql in [
        "SELECT COUNT(*) FROM t",
        "SELECT SUM(val) FROM t",
        "SELECT grp, COUNT(*), SUM(val) FROM t GROUP BY grp",
        // The text decode path (R2 stage 2b): GROUP BY on a TEXT column.
        "SELECT tag, COUNT(*) FROM txt GROUP BY tag",
    ] {
        for round in 1..=3 {
            let t = Instant::now();
            run(engine, &mut session, sql).unwrap();
            let dt = t.elapsed();
            println!(
                "{sql} (round {round}): {dt:?} ({:.0} ns/row)",
                dt.as_nanos() as f64 / N as f64
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Parallel-aggregate evidence probe (separate from the longitudinal probe above — its numbers stay
/// comparable): the parallel grouped aggregate vs the sequential vectorized fold vs the row
/// path, on a 2M-row GROUP BY. Results are asserted identical across all three (order
/// included); timings print for the perf log.
///
/// `cargo test -p nusadb-sql --release --test scan_bench parallel_group_by_probe -- --ignored --nocapture`
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn parallel_group_by_probe() {
    const N: usize = 2_000_000;
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nusadb_par_agg_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let engine: &'static BtreeEngine =
        Box::leak(Box::new(BtreeEngine::open(dir.join("bench.wal")).unwrap()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE p (k INT, v INT)").unwrap();
    let t0 = Instant::now();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({},{})", i % 500, i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO p VALUES {values}"),
        )
        .unwrap();
    }
    println!("load {N} rows: {:?}", t0.elapsed());

    // A fresh session per run: the per-session result cache (family) would otherwise
    // serve every repeat of the identical statement from memory and time nothing.
    let rows_for = |sql: &str| -> Vec<nusadb_sql::Row> {
        let mut session = Session::new(engine);
        let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
        match session.execute(plan(logical)).unwrap() {
            nusadb_sql::ExecutionResult::Rows { rows, .. } => rows,
            other => panic!("expected rows, got {other:?}"),
        }
    };

    let sql = "SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v) FROM p GROUP BY k";
    for label in ["warmup", "row path"] {
        let t = Instant::now();
        let n = rows_for(sql).len();
        println!(
            "{label}: {n} groups in {:?} ({:.0} ns/row)",
            t.elapsed(),
            t.elapsed().as_nanos() as f64 / N as f64
        );
    }
    let sequential = {
        let _v = nusadb_sql::vectorized::scope(true);
        let _p = nusadb_sql::vectorized::parallel_scope(false);
        let t = Instant::now();
        let rows = rows_for(sql);
        println!(
            "vectorized sequential: {} groups in {:?} ({:.0} ns/row)",
            rows.len(),
            t.elapsed(),
            t.elapsed().as_nanos() as f64 / N as f64
        );
        rows
    };
    let parallel = {
        let _v = nusadb_sql::vectorized::scope(true);
        let _p = nusadb_sql::vectorized::parallel_scope(true);
        for _ in 0..2 {
            let t = Instant::now();
            let rows = rows_for(sql);
            println!(
                "parallel: {} groups in {:?} ({:.0} ns/row)",
                rows.len(),
                t.elapsed(),
                t.elapsed().as_nanos() as f64 / N as f64
            );
            assert_eq!(rows, sequential, "parallel != sequential");
        }
        rows_for(sql)
    };
    assert_eq!(parallel.len(), 500);

    // Pushed-down shapes: pushed-down WHERE and scalar aggregation.
    for sql in [
        "SELECT k, COUNT(*), SUM(v) FROM p WHERE v > 48 GROUP BY k",
        "SELECT COUNT(*), SUM(v), MIN(v), MAX(v) FROM p",
    ] {
        let base = {
            let t = Instant::now();
            let rows = rows_for(sql);
            println!(
                "row path `{sql}`: {} rows in {:?} ({:.0} ns/row)",
                rows.len(),
                t.elapsed(),
                t.elapsed().as_nanos() as f64 / N as f64
            );
            rows
        };
        let _v = nusadb_sql::vectorized::scope(true);
        let _p = nusadb_sql::vectorized::parallel_scope(true);
        let t = Instant::now();
        let rows = rows_for(sql);
        println!(
            "parallel `{sql}`: {} rows in {:?} ({:.0} ns/row)",
            rows.len(),
            t.elapsed(),
            t.elapsed().as_nanos() as f64 / N as f64
        );
        assert_eq!(rows, base, "parallel != row path for `{sql}`");
    }
}

/// Split the per-statement cost of `SELECT 1` (the wire round-trip floor) into
/// parse / analyze / plan / execute — the wire layer runs exactly this pipeline per simple
/// query (plus encode + TCP).
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn select1_per_statement_split() {
    const N: u32 = 20_000;
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    let per = |d: std::time::Duration| d.as_nanos() as f64 / f64::from(N);

    let t = Instant::now();
    for _ in 0..N {
        std::hint::black_box(parse("SELECT 1").unwrap());
    }
    println!("parse: {:.0} ns", per(t.elapsed()));

    let t = Instant::now();
    for _ in 0..N {
        let stmt = parse("SELECT 1").unwrap();
        std::hint::black_box(analyze(stmt, &Cat(engine)).unwrap());
    }
    println!("parse+analyze: {:.0} ns", per(t.elapsed()));

    let t = Instant::now();
    for _ in 0..N {
        let logical = analyze(parse("SELECT 1").unwrap(), &Cat(engine)).unwrap();
        std::hint::black_box(plan(logical));
    }
    println!("parse+analyze+plan: {:.0} ns", per(t.elapsed()));

    let t = Instant::now();
    for _ in 0..N {
        let logical = analyze(parse("SELECT 1").unwrap(), &Cat(engine)).unwrap();
        session.execute(plan(logical)).unwrap();
    }
    println!(
        "full pipeline (fresh txn per stmt): {:.0} ns",
        per(t.elapsed())
    );

    // Plan-cache path: what the wire layer actually runs per simple query.
    let mut cache = nusadb_sql::PlanCache::new();
    let t = Instant::now();
    for _ in 0..N {
        let stmt = parse("SELECT 1").unwrap();
        let planned =
            nusadb_sql::plan_cached(&mut cache, "SELECT 1", stmt, &Cat(engine), engine).unwrap();
        session.execute(planned).unwrap();
    }
    println!(
        "wire-equivalent (parse + plan_cached + execute): {:.0} ns",
        per(t.elapsed())
    );
}

/// Evidence: GROUP BY under a spill budget — the sort-based fold vs the
/// statistics-routed DIRECT hash fold (moderate group counts). The many-group case records
/// that statistics do NOT change its route (a grace-partitioned fold measured slower than
/// sort here and was rejected).
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn manygroup_spill_group_by_probe() {
    const N: usize = 1_000_000;
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nusadb_manygroup_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let engine: &'static BtreeEngine =
        Box::leak(Box::new(BtreeEngine::open(dir.join("bench.wal")).unwrap()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE gm (k INT, v INT)").unwrap();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({},{})", i % 200_000, i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO gm VALUES {values}"),
        )
        .unwrap();
    }
    nusadb_sql::set_spill_config(Some(nusadb_sql::SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes: 8 << 20,
    }));
    let rows_for = |sql: &str| -> usize {
        let mut s = Session::new(engine);
        let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
        match s.execute(plan(logical)).unwrap() {
            nusadb_sql::ExecutionResult::Rows { rows, .. } => rows.len(),
            other => panic!("expected rows, got {other:?}"),
        }
    };
    let sql = "SELECT k, COUNT(*), SUM(v) FROM gm GROUP BY k";
    let t = Instant::now();
    let n_sort = rows_for(sql); // no statistics yet → sort-based spilling fold
    println!(
        "many-group sort-based (no stats): {n_sort} groups in {:?}",
        t.elapsed()
    );
    run(engine, &mut session, "ANALYZE gm").unwrap();
    let t = Instant::now();
    // Statistics present but the state estimate exceeds the budget → still the sort fold (the
    // grace-partitioned alternative was measured SLOWER here and deliberately not shipped).
    let n_grace = rows_for(sql);
    println!(
        "many-group post-ANALYZE (stays sort-based): {n_grace} groups in {:?}",
        t.elapsed()
    );
    assert_eq!(n_sort, n_grace);

    // Moderate group count (fits half the budget): statistics route to the DIRECT hash fold —
    // zero disk — vs the sort-based fold without them.
    run(engine, &mut session, "CREATE TABLE gmod (k INT, v INT)").unwrap();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({},{})", i % 10_000, i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO gmod VALUES {values}"),
        )
        .unwrap();
    }
    let sql = "SELECT k, COUNT(*), SUM(v) FROM gmod GROUP BY k";
    let t = Instant::now();
    let a = rows_for(sql);
    println!(
        "moderate sort-based (no stats): {a} groups in {:?}",
        t.elapsed()
    );
    run(engine, &mut session, "ANALYZE gmod").unwrap();
    let t = Instant::now();
    let b = rows_for(sql);
    println!(
        "moderate DIRECT hash (post-ANALYZE): {b} groups in {:?}",
        t.elapsed()
    );
    assert_eq!(a, b);
    nusadb_sql::set_spill_config(None);
}

/// Evidence probe: `ORDER BY <col> LIMIT n` over a 500k-row table times
/// the limit-aware top-N pass against a full `ORDER BY` (the pre-fix cost was a full sort of every
/// row). Prints ns and the top-5 ids so the result is visibly correct.
///
/// `cargo test -p nusadb-sql --release --test scan_bench top_n_orderby_probe -- --ignored --nocapture`
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn top_n_orderby_probe() {
    const N: usize = 500_000;
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nusadb_topn_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let engine: &'static BtreeEngine =
        Box::leak(Box::new(BtreeEngine::open(dir.join("bench.wal")).unwrap()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (id INT, val INT)").unwrap();
    // Insert in shuffled-ish order so the scan (row-id/insertion order) is NOT already sorted by id.
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| {
                let id = (i * 2_654_435_761) % N; // a scramble so scan order != id order
                format!("({id},{})", i % 97)
            })
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES {values}"),
        )
        .unwrap();
    }
    println!("load {N} rows");

    let rows_of = |session: &mut Session, sql: &str| -> Vec<nusadb_sql::Row> {
        let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
        match session.execute(plan(logical)).unwrap() {
            nusadb_sql::ExecutionResult::Rows { rows, .. } => rows,
            other => panic!("expected rows, got {other:?}"),
        }
    };

    // EXPLAIN ANALYZE reports the actual execution time of THIS run (settles cache questions).
    for ea in ["EXPLAIN ANALYZE SELECT id FROM t ORDER BY id LIMIT 5"] {
        let logical = analyze(parse(ea).unwrap(), &Cat(engine)).unwrap();
        if let nusadb_sql::ExecutionResult::Rows { rows, .. } =
            session.execute(plan(logical)).unwrap()
        {
            for r in &rows {
                if let Some(nusadb_sql::ast::Value::Text(line)) = r.first() {
                    println!("EA: {line}");
                }
            }
        }
    }
    for sql in [
        "SELECT id FROM t ORDER BY id LIMIT 5",
        "SELECT id FROM t ORDER BY id LIMIT 100",
        "SELECT id FROM t ORDER BY id DESC LIMIT 5",
        "SELECT id FROM t ORDER BY id", // full sort (baseline cost of sorting every row)
    ] {
        for round in 1..=3 {
            let t = Instant::now();
            let rows = rows_of(&mut session, sql);
            let dt = t.elapsed();
            let head: Vec<i64> = rows
                .iter()
                .take(3)
                .map(|r| match r.first() {
                    Some(nusadb_sql::ast::Value::Int(n)) => *n,
                    _ => -1,
                })
                .collect();
            println!(
                "{sql} (round {round}): {dt:?} [{} rows, head {head:?}]",
                rows.len()
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Evidence probe: a ranking window under `ORDER BY … LIMIT` over a large
/// table. With a work-memory budget set, the pre-fix full materialization of every input row
/// exceeds it and errors (OOM); the limit-aware window computes over only the first `m` rows and
/// completes. Prints the outcome + the (correct) top rows.
///
/// `cargo test -p nusadb-sql --release --test scan_bench window_oom_probe -- --ignored --nocapture`
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn window_oom_probe() {
    const N: usize = 500_000;
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nusadb_window_oom_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let engine: &'static BtreeEngine =
        Box::leak(Box::new(BtreeEngine::open(dir.join("bench.wal")).unwrap()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (id INT, v INT)").unwrap();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({i},{})", i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES {values}"),
        )
        .unwrap();
    }
    println!("load {N} rows");

    // A budget the full 500k-row materialization exceeds but the bounded window fits under.
    nusadb_sql::set_work_mem(64 * 1024 * 1024);
    let sql = "SELECT id, row_number() OVER (ORDER BY id) FROM t ORDER BY id LIMIT 3";
    let t = std::time::Instant::now();
    match run_rows(engine, &mut session, sql) {
        Ok(rows) => println!(
            "ranking window + LIMIT 3: OK in {:?} — {:?}",
            t.elapsed(),
            rows.iter()
                .take(3)
                .map(|r| (
                    match r.first() {
                        Some(nusadb_sql::ast::Value::Int(n)) => *n,
                        _ => -1,
                    },
                    match r.get(1) {
                        Some(nusadb_sql::ast::Value::Int(n)) => *n,
                        _ => -1,
                    }
                ))
                .collect::<Vec<_>>()
        ),
        Err(e) => println!("ranking window + LIMIT 3: ERROR — {e}"),
    }
    nusadb_sql::set_work_mem(0);
    std::fs::remove_dir_all(&dir).ok();
}

/// Run a query and return its rows (probe helper).
fn run_rows(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<Vec<nusadb_sql::Row>, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    match session.execute(plan(logical))? {
        nusadb_sql::ExecutionResult::Rows { rows, .. } => Ok(rows),
        other => panic!("expected rows, got {other:?}"),
    }
}
