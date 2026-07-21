//! /vectorized filter path vs row path, measured (the evidence gate for default-on).
//!
//! The vectorized SELECT path with its SIMD `column <cmp> literal` filter kernels (i64,
//! F64) is opt-in and **default-off** until a measurement shows it beats the row-at-a-time
//! evaluator on a filter-heavy scan. This bench drives the *same* `SELECT ... WHERE` over a
//! `BtreeEngine`, toggling [`vectorized::scope`], so the speedup (or its absence) is visible
//! side by side. The plan is built once and cloned per iteration, so the measurement isolates the
//! execution path (scan → filter → project) from parse/analyze/plan overhead, which both paths share.
//!
//! Two predicate types exercise both kernels: `v > k` over an INT column and `p > k` over a FLOAT
//! column. Selectivity is ~10% (the predicate keeps roughly one row in ten).
//!
//! Run: `cargo bench -p nusadb-sql --bench vectorized_filter`.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    reason = "criterion_group! generates an undocumented `benches` fn; a bench harness panics on setup failure"
)]

use std::fmt::Write as _;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{Catalog, PhysicalPlan, Session, analyze, parse, plan, vectorized};

/// Minimal analyzer catalog over the engine: only table lookup is needed (no secondary indexes, so
/// the planner emits a sequential scan — the shape the vectorized path supports).
struct BenchCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for BenchCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

/// Parse → analyze → plan → execute one statement in `session` (used for DDL/DML setup).
fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) {
    let stmt = parse(sql).unwrap();
    let logical = analyze(stmt, &BenchCatalog(engine)).unwrap();
    session.execute(plan(logical)).unwrap();
}

/// Build a `t(id INT, v INT, p FLOAT)` table of `n` rows. `v = id % 100` and `p = (id % 100)` give a
/// `> 90` predicate a stable ~10% selectivity. The engine is leaked to `'static` (bounded by the
/// bench binary's lifetime) so iterations can borrow it without a self-referential struct.
fn setup(n: usize) -> &'static BtreeEngine {
    // Insert in chunks: one giant VALUES list would bloat the parser, one row per statement would
    // make setup dominate. 512 rows/statement is a balance.
    const CHUNK: usize = 512;

    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT, v INT, p FLOAT)",
    );
    let mut start = 0;
    while start < n {
        let end = (start + CHUNK).min(n);
        let mut sql = String::from("INSERT INTO t VALUES ");
        for id in start..end {
            if id > start {
                sql.push(',');
            }
            let k = id % 100;
            write!(sql, "({id}, {k}, {k}.0)").unwrap();
        }
        exec(engine, &mut session, &sql);
        start = end;
    }
    engine
}

/// The physical plan for `sql` against `engine`, built once for cloning into the measured loop.
fn build_plan(engine: &dyn StorageEngine, sql: &str) -> PhysicalPlan {
    let stmt = parse(sql).unwrap();
    let logical = analyze(stmt, &BenchCatalog(engine)).unwrap();
    plan(logical)
}

/// Execute a pre-built plan under the chosen path; returns the row count so the work isn't optimized
/// away.
fn run(engine: &'static BtreeEngine, plan: &PhysicalPlan, batch_path: bool) -> usize {
    let _guard = vectorized::scope(batch_path);
    let mut session = Session::new(engine);
    match session.execute(plan.clone()).unwrap() {
        nusadb_sql::ExecutionResult::Rows { rows, .. } => rows.len(),
        _ => 0,
    }
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectorized_filter");
    for &n in &[10_000usize, 100_000] {
        let engine = setup(n);
        group.throughput(Throughput::Elements(n as u64));
        for (kind, sql) in [
            ("int", "SELECT id, v FROM t WHERE v > 90"),
            ("float", "SELECT id, p FROM t WHERE p > 90.0"),
        ] {
            let plan = build_plan(engine, sql);
            // Sanity: both paths must agree, else the comparison is meaningless.
            assert_eq!(run(engine, &plan, false), run(engine, &plan, true));
            for (path, batch) in [("row", false), ("batch", true)] {
                let id = BenchmarkId::new(format!("{kind}/{path}"), n);
                group.bench_with_input(id, &n, |b, _| {
                    b.iter(|| black_box(run(engine, &plan, batch)));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);
