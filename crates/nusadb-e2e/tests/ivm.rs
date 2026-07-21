//! End-to-end tests for incremental view maintenance: a single-table projection +
//! filter materialized view stays in sync with its base table on INSERT/UPDATE/DELETE without a
//! `REFRESH`, and matches what a full `REFRESH` would produce. Ineligible views (joins/aggregates)
//! fall back to full-refresh-only. Driven through `parse → analyze → plan → execute` against the
//! production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        // The matview's backing store is a plain table, so view resolution is not needed here; a
        // default suffices for these tests.
        let _ = name;
        Ok(None)
    }
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

/// The materialized view's backing rows (it is an ordinary table), sorted for comparison.
fn mv_rows(engine: &BtreeEngine, view: &str) -> Vec<Vec<Value>> {
    let mut rows = match run(engine, &format!("SELECT * FROM {view}")) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    };
    rows.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    rows
}

#[test]
fn ivm_tracks_insert_update_delete_for_filtered_projection() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE orders (id INT NOT NULL, amount INT, status TEXT)",
    );
    run(
        &engine,
        "INSERT INTO orders VALUES (1, 100, 'paid'), (2, 50, 'pending'), (3, 200, 'paid')",
    );
    // A single-table projection + filter view → IVM-eligible.
    run(
        &engine,
        "CREATE MATERIALIZED VIEW big_paid AS \
         SELECT id, amount FROM orders WHERE status = 'paid' AND amount >= 100",
    );
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(3), Value::Int(200)],
        ]
    );

    // INSERT a matching row → appears in the view with no REFRESH.
    run(&engine, "INSERT INTO orders VALUES (4, 300, 'paid')");
    // INSERT a non-matching row → does not appear.
    run(&engine, "INSERT INTO orders VALUES (5, 80, 'paid')");
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(3), Value::Int(200)],
            vec![Value::Int(4), Value::Int(300)],
        ]
    );

    // UPDATE that pushes a row out of the filter (amount drops below 100).
    run(&engine, "UPDATE orders SET amount = 10 WHERE id = 1");
    // UPDATE that pulls a row into the filter (status flips to paid, already >= 100).
    run(&engine, "INSERT INTO orders VALUES (6, 150, 'pending')");
    run(&engine, "UPDATE orders SET status = 'paid' WHERE id = 6");
    // UPDATE a value that stays in the filter (amount changes but still qualifies).
    run(&engine, "UPDATE orders SET amount = 250 WHERE id = 3");
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![
            vec![Value::Int(3), Value::Int(250)],
            vec![Value::Int(4), Value::Int(300)],
            vec![Value::Int(6), Value::Int(150)],
        ]
    );

    // DELETE a matching base row → leaves the view.
    run(&engine, "DELETE FROM orders WHERE id = 4");
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![
            vec![Value::Int(3), Value::Int(250)],
            vec![Value::Int(6), Value::Int(150)],
        ]
    );

    // The incrementally-maintained state equals a full REFRESH (the oracle).
    run(&engine, "REFRESH MATERIALIZED VIEW big_paid");
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![
            vec![Value::Int(3), Value::Int(250)],
            vec![Value::Int(6), Value::Int(150)],
        ]
    );
}

#[test]
fn ivm_preserves_duplicate_rows_as_a_bag() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, k INT)");
    run(&engine, "INSERT INTO t VALUES (1, 7), (2, 7), (3, 7)");
    // Projection drops the unique id, so all three rows project to the same view row.
    run(
        &engine,
        "CREATE MATERIALIZED VIEW just_k AS SELECT k FROM t",
    );
    assert_eq!(
        mv_rows(&engine, "just_k"),
        vec![
            vec![Value::Int(7)],
            vec![Value::Int(7)],
            vec![Value::Int(7)],
        ]
    );
    // Deleting one base row must remove exactly one view row (bag semantics), not all three.
    run(&engine, "DELETE FROM t WHERE id = 2");
    assert_eq!(
        mv_rows(&engine, "just_k"),
        vec![vec![Value::Int(7)], vec![Value::Int(7)],]
    );
}

#[test]
fn ineligible_view_is_not_incrementally_maintained() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, k INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    // An aggregate view is not IVM-eligible — it is full-refresh-only.
    run(
        &engine,
        "CREATE MATERIALIZED VIEW total AS SELECT count(*) FROM t",
    );
    assert_eq!(mv_rows(&engine, "total"), vec![vec![Value::Int(2)]]);
    // A base insert does NOT update the aggregate view (it would need a REFRESH).
    run(&engine, "INSERT INTO t VALUES (3, 30)");
    assert_eq!(
        mv_rows(&engine, "total"),
        vec![vec![Value::Int(2)]],
        "an aggregate MV is not incrementally maintained"
    );
    // A REFRESH brings it up to date.
    run(&engine, "REFRESH MATERIALIZED VIEW total");
    assert_eq!(mv_rows(&engine, "total"), vec![vec![Value::Int(3)]]);
}

#[test]
fn view_projecting_volatile_age_is_not_incrementally_maintained() {
    // Deep-gate sibling: `age(value)` is wall-clock-relative, so a view projecting it must NOT
    // be IVM-eligible — otherwise stored rows keep the age computed at insert time and drift stale
    // across days. The fix adds `Age` to the IVM volatility list (it was only on the result-cache
    // list). We assert via row COUNT (the age value itself is date-dependent): the view is
    // full-refresh-only, so a base insert does not append a row until a REFRESH.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, d DATE)");
    run(&engine, "INSERT INTO t VALUES (1, DATE '2000-01-01')");
    run(
        &engine,
        "CREATE MATERIALIZED VIEW aged AS SELECT id, age(d) FROM t",
    );
    assert_eq!(mv_rows(&engine, "aged").len(), 1);

    // Without the fix this insert would be incrementally appended (treating age() as stable).
    run(&engine, "INSERT INTO t VALUES (2, DATE '2000-01-01')");
    assert_eq!(
        mv_rows(&engine, "aged").len(),
        1,
        "a view projecting the volatile age() is not incrementally maintained"
    );

    // A REFRESH recomputes the whole view, bringing in the second row.
    run(&engine, "REFRESH MATERIALIZED VIEW aged");
    assert_eq!(mv_rows(&engine, "aged").len(), 2);
}

#[test]
fn ivm_tracks_upsert_do_update() {
    // Deep-gate: ON CONFLICT DO UPDATE previously skipped IVM, so the materialized view went
    // stale. An upsert must apply both the insert and the update side of its delta to the view.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE orders (id INT PRIMARY KEY, amount INT, status TEXT)",
    );
    run(&engine, "INSERT INTO orders VALUES (1, 100, 'paid')");
    run(
        &engine,
        "CREATE MATERIALIZED VIEW big_paid AS \
         SELECT id, amount FROM orders WHERE status = 'paid' AND amount >= 100",
    );
    assert_eq!(
        mv_rows(&engine, "big_paid"),
        vec![vec![Value::Int(1), Value::Int(100)]]
    );

    // id=1 conflicts → UPDATE its amount (stays in the view at the new value); id=2 is new and
    // matches → INSERT enters the view. Both must reflect immediately, with no REFRESH.
    run(
        &engine,
        "INSERT INTO orders VALUES (1, 250, 'paid'), (2, 300, 'paid') \
         ON CONFLICT (id) DO UPDATE SET amount = EXCLUDED.amount",
    );
    let expected = vec![
        vec![Value::Int(1), Value::Int(250)],
        vec![Value::Int(2), Value::Int(300)],
    ];
    assert_eq!(mv_rows(&engine, "big_paid"), expected);

    // The incrementally-maintained state equals a full REFRESH (the oracle).
    run(&engine, "REFRESH MATERIALIZED VIEW big_paid");
    assert_eq!(mv_rows(&engine, "big_paid"), expected);
}

#[test]
fn ivm_tracks_fk_cascade_delete_on_the_child() {
    // Deep-gate #16: a cascade DELETE on a child table must maintain a materialized view over that
    // child — the cascade previously fired triggers but skipped IVM, leaving the view stale.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT NOT NULL, pid INT REFERENCES parent(id) ON DELETE CASCADE)",
    );
    run(&engine, "INSERT INTO parent VALUES (1), (2)");
    run(
        &engine,
        "INSERT INTO child VALUES (10, 1), (11, 1), (20, 2)",
    );
    run(
        &engine,
        "CREATE MATERIALIZED VIEW child_ids AS SELECT id FROM child",
    );
    assert_eq!(mv_rows(&engine, "child_ids").len(), 3);

    // Deleting parent id=1 cascades to children 10 and 11; the view must drop exactly those two rows.
    run(&engine, "DELETE FROM parent WHERE id = 1");
    assert_eq!(mv_rows(&engine, "child_ids"), vec![vec![Value::Int(20)]]);
    // Equals a full REFRESH (the oracle).
    run(&engine, "REFRESH MATERIALIZED VIEW child_ids");
    assert_eq!(mv_rows(&engine, "child_ids"), vec![vec![Value::Int(20)]]);
}

#[test]
fn ivm_tracks_fk_set_null_on_the_child() {
    // Deep-gate #16: an ON DELETE SET NULL cascade rewrites child rows and must maintain a view that
    // projects the affected column, not leave it stale.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT NOT NULL, pid INT REFERENCES parent(id) ON DELETE SET NULL)",
    );
    run(&engine, "INSERT INTO parent VALUES (1)");
    run(&engine, "INSERT INTO child VALUES (10, 1)");
    run(
        &engine,
        "CREATE MATERIALIZED VIEW child_pids AS SELECT id, pid FROM child",
    );
    assert_eq!(
        mv_rows(&engine, "child_pids"),
        vec![vec![Value::Int(10), Value::Int(1)]]
    );

    // Deleting the parent nulls child.pid; the view's row must reflect the new NULL.
    run(&engine, "DELETE FROM parent WHERE id = 1");
    let expected = vec![vec![Value::Int(10), Value::Null]];
    assert_eq!(mv_rows(&engine, "child_pids"), expected);
    run(&engine, "REFRESH MATERIALIZED VIEW child_pids");
    assert_eq!(mv_rows(&engine, "child_pids"), expected);
}
