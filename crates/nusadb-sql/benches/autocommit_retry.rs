//! What the auto-commit conflict retry costs a statement that never conflicts.
//!
//! The retry only runs after a serialization conflict, so its cost under contention is obvious and
//! intended. The question this bench exists to answer is the other one: whether making a retry
//! *possible* slowed down the overwhelmingly common case where nothing conflicts at all.
//!
//! `dispatch` consumes the plan, so keeping a retry available means keeping a spare copy of it. That
//! copy is the entire added cost on the uncontended path — there is no other difference — so the two
//! groups below measure it directly rather than by subtracting two whole-statement timings.
//!
//! Measuring it by subtraction was tried first and does not work here: executing a statement changes
//! the engine's state (the table grows, version chains lengthen), so per-iteration cost drifts
//! upward *within* a run and the two arms never sample the same work. That produced differences with
//! the wrong sign — a decisive sign the method, not the code, was at fault. Cloning a plan touches no
//! engine state, so it is measurable to a precision the end-to-end path cannot reach here.
//!
//! Read the two groups as a ratio: `plan_clone` over `statement_total` for the same shape is the
//! fraction of an uncontended statement the feature adds.
//!
//! Plan size is varied deliberately. A plan clone is proportional to what the tree holds, so a
//! multi-row `INSERT … VALUES` (which carries every literal) is the shape most able to show the cost
//! and a single-row `UPDATE` the least — if the ratio were going to matter anywhere, it would be at
//! the wide end.
//!
//! Run: `cargo bench -p nusadb-sql --bench autocommit_retry`.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    reason = "criterion_group! generates an undocumented `benches` fn; a bench harness panics on setup failure"
)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, IndexInfo, PhysicalPlan, Session, analyze, parse, plan};
use std::hint::black_box;

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn planned(engine: &dyn StorageEngine, sql: &str) -> PhysicalPlan {
    plan(analyze(parse(sql).unwrap(), &Cat(engine)).unwrap())
}

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) {
    session.execute(planned(engine, sql)).unwrap();
}

/// An `INSERT` carrying `rows` literal tuples — the plan shape whose clone costs the most.
fn insert_sql(rows: usize) -> String {
    use std::fmt::Write as _;
    let mut s = String::from("INSERT INTO t (a, b) VALUES ");
    for i in 0..rows {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "({i}, 'row-{i}')");
    }
    s
}

/// The added work: one clone of the plan the statement is about to run.
fn bench_plan_clone(c: &mut Criterion) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    run(engine, &mut session, "INSERT INTO t VALUES (1, 'x')");

    let mut group = c.benchmark_group("plan_clone");
    for rows in [1_usize, 16, 256] {
        let template = planned(engine, &insert_sql(rows));
        group.bench_with_input(BenchmarkId::new("insert", rows), &rows, |b, _| {
            b.iter(|| black_box(template.clone()));
        });
    }
    let template = planned(engine, "UPDATE t SET a = a + 1 WHERE b = 'x'");
    group.bench_function("update_one_row", |b| {
        b.iter(|| black_box(template.clone()));
    });
    group.finish();
}

/// The denominator: the whole uncontended statement the clone rides on.
///
/// These timings drift as the table grows, so they are an order of magnitude rather than a precise
/// figure — which is all the ratio needs, since the numerator above is precise and thousands of times
/// smaller.
fn bench_statement_total(c: &mut Criterion) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));

    let mut group = c.benchmark_group("statement_total");
    for rows in [1_usize, 16, 256] {
        let mut session = Session::new(engine);
        run(engine, &mut session, "DROP TABLE IF EXISTS t");
        run(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
        let template = planned(engine, &insert_sql(rows));
        group.bench_with_input(BenchmarkId::new("insert", rows), &rows, |b, _| {
            b.iter(|| black_box(session.execute(template.clone()).unwrap()));
        });
    }

    let mut session = Session::new(engine);
    run(engine, &mut session, "DROP TABLE IF EXISTS t");
    run(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    run(engine, &mut session, "INSERT INTO t VALUES (1, 'x')");
    let template = planned(engine, "UPDATE t SET a = a + 1 WHERE b = 'x'");
    group.bench_function("update_one_row", |b| {
        b.iter(|| black_box(session.execute(template.clone()).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, bench_plan_clone, bench_statement_total);
criterion_main!(benches);
