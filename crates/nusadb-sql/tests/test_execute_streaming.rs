//! `Session::execute_streaming` (Phase 2 streaming output) must deliver, through a push
//! [`RowSink`], exactly the rows that the buffered `Session::execute` returns — for a linear `SELECT`
//! (truly streamed), a blocking top operator (`ORDER BY`/`DISTINCT`, materialized once then drained),
//! with spill on and off, inside an explicit transaction, and it must report a non-row statement as
//! `StreamOutcome::Other`. A sink error must propagate and abort the statement.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{ColumnType, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, RowSink, Session, SpillConfig, StreamOutcome,
    analyze, parse, plan, set_spill_config,
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

/// A [`RowSink`] that records the columns, their advertised types, and every row it is handed.
/// `typed` records whether the typed `columns_typed` entry point was used (a SELECT or a replayed
/// `RETURNING` row set) versus the untyped `columns` fallback.
#[derive(Default)]
struct Collect {
    columns: Vec<String>,
    types: Vec<ColumnType>,
    typed: bool,
    rows: Vec<Vec<Value>>,
}
impl RowSink for Collect {
    fn columns(&mut self, columns: &[String]) -> Result<(), Error> {
        self.columns = columns.to_vec();
        Ok(())
    }
    fn columns_typed(&mut self, names: &[String], types: &[ColumnType]) -> Result<(), Error> {
        self.columns = names.to_vec();
        self.types = types.to_vec();
        self.typed = true;
        Ok(())
    }
    fn row(&mut self, row: &[Value]) -> Result<(), Error> {
        self.rows.push(row.to_vec());
        Ok(())
    }
}

/// A [`RowSink`] that errors on the `fail_at`-th row, to prove the error aborts the statement.
struct FailAfter {
    seen: usize,
    fail_at: usize,
}
impl RowSink for FailAfter {
    fn columns(&mut self, _columns: &[String]) -> Result<(), Error> {
        Ok(())
    }
    fn row(&mut self, _row: &[Value]) -> Result<(), Error> {
        self.seen += 1;
        if self.seen >= self.fail_at {
            return Err(Error::Unsupported("sink stop".to_owned()));
        }
        Ok(())
    }
}

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session.execute(plan(logical)).unwrap()
}

fn planned(engine: &dyn StorageEngine, sql: &str) -> nusadb_sql::PhysicalPlan {
    plan(analyze(parse(sql).unwrap(), &Cat(engine)).unwrap())
}

/// The buffered `execute` result rows as a sorted multiset.
fn buffered(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> (Vec<String>, Vec<Vec<Value>>) {
    let ExecutionResult::Rows { columns, mut rows } = run(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    (columns, rows)
}

#[test]
fn execute_streaming_matches_buffered_execute() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    run(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..150 {
        let a = i % 12;
        let b = format!("'r{}'", i % 5);
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({a}, {b})"),
        );
    }

    let queries = [
        "SELECT a, b FROM t",                   // linear pipeline → truly streamed
        "SELECT a, b FROM t WHERE a > 5",       // linear + filter
        "SELECT a FROM t ORDER BY a",           // blocking top op (sort)
        "SELECT DISTINCT a FROM t",             // blocking top op (distinct)
        "SELECT a, COUNT(*) FROM t GROUP BY a", // blocking top op (aggregate)
    ];

    for sql in queries {
        for cfg in [None, Some(64usize)] {
            set_spill_config(cfg.map(|threshold_bytes| SpillConfig {
                dir: std::env::temp_dir(),
                threshold_bytes,
            }));

            let (want_cols, want_rows) = buffered(engine, &mut session, sql);

            let mut sink = Collect::default();
            let outcome = session
                .execute_streaming(planned(engine, sql), &mut sink)
                .unwrap();
            let StreamOutcome::Rows { columns, count } = outcome else {
                panic!("expected StreamOutcome::Rows for: {sql}");
            };

            assert_eq!(columns, want_cols, "columns mismatch for: {sql}");
            assert_eq!(count, want_rows.len(), "count mismatch for: {sql}");
            assert_eq!(sink.columns, want_cols, "sink columns mismatch for: {sql}");

            sink.rows.sort_by_key(|r| format!("{r:?}"));
            assert_eq!(
                sink.rows, want_rows,
                "streamed rows must match buffered execute for: {sql} (spill={cfg:?})"
            );
        }
    }
    set_spill_config(None);
}

/// (hash-join build-side selection): a self-join whose LEFT carries a selective
/// pushed-down filter — the gated left-build flip's canonical shape — must stream exactly the
/// buffered result, BOTH before ANALYZE (no statistics → the gate stays off, default
/// build-right) and after (statistics resolve both sides → the flip can fire). An outer-join
/// variant (gate excluded by kind) pins that path too.
#[test]
fn execute_streaming_self_join_left_build_matches_buffered() {
    fn check(
        engine: &'static BtreeEngine,
        session: &mut Session,
        sql: &str,
        expect_count: usize,
        label: &str,
    ) {
        let (want_cols, want_rows) = buffered(engine, session, sql);
        assert_eq!(want_rows.len(), expect_count, "{label}: buffered count");
        let mut sink = Collect::default();
        let outcome = session
            .execute_streaming(planned(engine, sql), &mut sink)
            .unwrap();
        let StreamOutcome::Rows { columns, count } = outcome else {
            panic!("{label}: expected StreamOutcome::Rows");
        };
        assert_eq!(columns, want_cols, "{label}: columns");
        assert_eq!(count, want_rows.len(), "{label}: count");
        sink.rows.sort_by_key(|r| format!("{r:?}"));
        assert_eq!(sink.rows, want_rows, "{label}: rows");
    }

    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(
        engine,
        &mut session,
        "CREATE TABLE orders (id INT, customer_id INT)",
    );
    for i in 0..600 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO orders VALUES ({i}, {})", i % 20),
        );
    }
    // For o1.id = k (k < 5): the o2 rows sharing customer k%20 with a larger id are
    // k+20, k+40, …, k+580 — 29 each; 5 × 29 = 145 pairs.
    let inner = "SELECT o1.id, o2.id FROM orders o1 JOIN orders o2 \
                 ON o1.customer_id = o2.customer_id AND o1.id < o2.id AND o1.id < 5";
    let left = "SELECT o1.id, o2.id FROM orders o1 LEFT JOIN orders o2 \
                ON o1.customer_id = o2.customer_id AND o2.id < 5 WHERE o1.id < 40";

    // No statistics yet: the gate stays off; both paths must already agree.
    check(engine, &mut session, inner, 145, "inner pre-ANALYZE");
    // Each of the 40 left rows matches the (< 5) right rows of its customer: ids 0..5 all have
    // distinct customers, so left rows whose customer is in 0..5 match exactly one, the rest
    // match none (NULL-padded) — 40 rows either way.
    check(engine, &mut session, left, 40, "left pre-ANALYZE");

    run(engine, &mut session, "ANALYZE orders");

    // With statistics the INNER flip can fire (left estimate ≈ 5 rows ≪ 600): identical rows.
    check(engine, &mut session, inner, 145, "inner post-ANALYZE");
    check(engine, &mut session, left, 40, "left post-ANALYZE");
}

#[test]
fn execute_streaming_reports_non_row_statements_as_other() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (a INT)");

    let mut sink = Collect::default();
    let outcome = session
        .execute_streaming(
            planned(engine, "INSERT INTO t VALUES (1), (2), (3)"),
            &mut sink,
        )
        .unwrap();
    match outcome {
        StreamOutcome::Other(ExecutionResult::Inserted(3)) => {},
        other => panic!("expected Other(Inserted(3)), got {other:?}"),
    }
    assert!(
        sink.rows.is_empty(),
        "INSERT must not push rows to the sink"
    );
}

#[test]
fn execute_streaming_resolves_bare_select_sequence_calls() {
    // Regression: a no-FROM `SELECT nextval('s')` is a once-evaluated context, so the sequence
    // built-in must resolve against the engine and advance — on the STREAMING path too, not only the
    // buffered `execute`. The wire/extended-query path streams; previously its Project arm skipped the
    // sequence resolution, so the call fell through to the per-row evaluator and was wrongly rejected
    // as "only supported where it is evaluated once" even though a bare SELECT is exactly that.
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE SEQUENCE s");

    let streamed = |session: &mut Session, sql: &str| -> Vec<Value> {
        let mut sink = Collect::default();
        let outcome = session
            .execute_streaming(planned(engine, sql), &mut sink)
            .unwrap_or_else(|e| panic!("streaming `{sql}` must succeed, got: {e:?}"));
        assert!(
            matches!(outcome, StreamOutcome::Rows { count: 1, .. }),
            "streamed `{sql}` must yield exactly one row, got {outcome:?}"
        );
        sink.rows.into_iter().flatten().collect()
    };

    // nextval advances 1, 2 on the streamed path (the exact form QA saw rejected via a driver).
    assert_eq!(
        streamed(&mut session, "SELECT nextval('s')"),
        vec![Value::Int(1)]
    );
    assert_eq!(
        streamed(&mut session, "SELECT nextval('s')"),
        vec![Value::Int(2)]
    );
    // currval reads the last handed-out value without advancing.
    assert_eq!(
        streamed(&mut session, "SELECT currval('s')"),
        vec![Value::Int(2)]
    );
    // Two calls in one streamed row advance twice.
    assert_eq!(
        streamed(&mut session, "SELECT nextval('s'), nextval('s')"),
        vec![Value::Int(3), Value::Int(4)]
    );
    // setval jumps the sequence; the next streamed nextval returns value + increment.
    assert_eq!(
        streamed(&mut session, "SELECT setval('s', 100)"),
        vec![Value::Int(100)]
    );
    assert_eq!(
        streamed(&mut session, "SELECT nextval('s')"),
        vec![Value::Int(101)]
    );
}

#[test]
fn execute_streaming_returning_reports_real_column_types() {
    // INSERT/UPDATE ... RETURNING streams through the buffered path then replays into the sink, so it
    // must still advertise the projection's real per-column types — like a streamed SELECT — instead
    // of letting every column default to text (QA RETURNING-type finding). A non-text column reported
    // as text would hand a strict-typed driver a string for `RETURNING id`.
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (id INT, label TEXT)");

    let mut sink = Collect::default();
    let outcome = session
        .execute_streaming(
            planned(
                engine,
                "INSERT INTO t VALUES (5, 'x') RETURNING id, label, id + 1",
            ),
            &mut sink,
        )
        .unwrap();
    let StreamOutcome::Rows { columns, count } = outcome else {
        panic!("expected rows from RETURNING");
    };
    assert_eq!(count, 1);
    assert_eq!(columns, vec!["id", "label", "?column?"]);
    assert!(
        sink.typed,
        "RETURNING must announce typed columns (not the untyped text fallback)"
    );
    assert_eq!(
        sink.types,
        vec![ColumnType::Int, ColumnType::Text, ColumnType::Int],
        "RETURNING column types must match the projection, not all-text"
    );
    assert_eq!(
        sink.rows,
        vec![vec![
            Value::Int(5),
            Value::Text("x".to_owned()),
            Value::Int(6)
        ]]
    );

    // An UPDATE ... RETURNING is the same buffered-then-replayed path and must also be typed.
    let mut sink2 = Collect::default();
    session
        .execute_streaming(
            planned(engine, "UPDATE t SET id = 99 RETURNING id"),
            &mut sink2,
        )
        .unwrap();
    assert!(
        sink2.typed,
        "UPDATE ... RETURNING must announce typed columns"
    );
    assert_eq!(sink2.types, vec![ColumnType::Int]);
    assert_eq!(sink2.rows, vec![vec![Value::Int(99)]]);
}

#[test]
fn execute_streaming_sees_uncommitted_rows_in_an_explicit_transaction() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (a INT)");

    run(engine, &mut session, "BEGIN");
    run(engine, &mut session, "INSERT INTO t VALUES (10), (20)");
    let mut sink = Collect::default();
    let outcome = session
        .execute_streaming(planned(engine, "SELECT a FROM t ORDER BY a"), &mut sink)
        .unwrap();
    let StreamOutcome::Rows { count, .. } = outcome else {
        panic!("expected rows");
    };
    assert_eq!(count, 2, "must see the in-transaction inserts");
    assert_eq!(sink.rows, vec![vec![Value::Int(10)], vec![Value::Int(20)]]);
    run(engine, &mut session, "COMMIT");
}

#[test]
fn execute_streaming_propagates_sink_error() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(engine, &mut session, "CREATE TABLE t (a INT)");
    for i in 0..10 {
        run(engine, &mut session, &format!("INSERT INTO t VALUES ({i})"));
    }

    let mut sink = FailAfter {
        seen: 0,
        fail_at: 3,
    };
    let err = session
        .execute_streaming(planned(engine, "SELECT a FROM t"), &mut sink)
        .expect_err("sink error must propagate");
    assert!(matches!(err, Error::Unsupported(_)));

    // The session is usable afterwards (the auto-commit txn was rolled back, not leaked).
    let mut sink2 = Collect::default();
    session
        .execute_streaming(planned(engine, "SELECT a FROM t"), &mut sink2)
        .unwrap();
    assert_eq!(sink2.rows.len(), 10);
}

/// Scale evidence (ignored; run on demand): at 1M rows the gated streamed flip must (a)
/// stay row-identical to the buffered execute, and (b) not be slower than the pre-ANALYZE
/// build-right path — the whole point is replacing a 1M-row hash build with a ~500-row one.
/// Timings print to stderr for the perf log; the assert itself is conservative (no flaky CI).
#[test]
#[ignore = "1M-row scale evidence; run via cargo test -p nusadb-sql --release --test test_execute_streaming selfjoin_build_side_flip_scale -- --ignored --nocapture"]
fn selfjoin_build_side_flip_scale_evidence() {
    fn timed_stream(
        engine: &'static BtreeEngine,
        session: &mut Session,
        q: &str,
        label: &str,
    ) -> (Vec<Vec<Value>>, std::time::Duration) {
        let mut sink = Collect::default();
        let start = std::time::Instant::now();
        let outcome = session
            .execute_streaming(planned(engine, q), &mut sink)
            .unwrap();
        let elapsed = start.elapsed();
        assert!(matches!(outcome, StreamOutcome::Rows { .. }), "{label}");
        eprintln!("{label}: {elapsed:?}");
        (sink.rows, elapsed)
    }

    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    run(
        engine,
        &mut session,
        "CREATE TABLE orders (id INT NOT NULL, customer_id INT)",
    );
    let n = 1_000_000u32;
    let customers = 1_000u32;
    for batch in 0..(n / 1000) {
        let values: Vec<String> = (0..1000)
            .map(|j| {
                let i = batch * 1000 + j;
                format!("({i},{})", i % customers)
            })
            .collect();
        run(
            engine,
            &mut session,
            &format!("INSERT INTO orders VALUES {}", values.join(",")),
        );
    }
    let q = "SELECT count(*) FROM orders o1 JOIN orders o2 \
             ON o1.customer_id = o2.customer_id AND o1.id < o2.id AND o1.id < 500";

    // Pre-ANALYZE: gate off — the build side is the full 1M-row right input.
    let (rows_before, before) = timed_stream(
        engine,
        &mut session,
        q,
        "streamed pre-ANALYZE (build-right, 1M build)",
    );
    run(engine, &mut session, "ANALYZE orders");
    // Post-ANALYZE: the flip fires — the build side is the ~500-row filtered left input.
    let (rows_after, after) = timed_stream(
        engine,
        &mut session,
        q,
        "streamed post-ANALYZE (left-build flip)",
    );

    assert_eq!(rows_before, rows_after, "flip must not change the result");
    // Every o1.id < 500 has customer_id == id (n/customers = 1000 rows per customer, ids 0..999
    // hold distinct customers 0..999): each pairs with the 999 same-customer larger ids.
    assert_eq!(rows_after[0][0], Value::Int(500 * 999));
    assert!(
        after <= before,
        "the ~500-row build must not lose to the 1M-row build (before {before:?}, after {after:?})"
    );
}

/// Spill-bounded engage: under a configured spill budget the parallel grouped fold may
/// replace the bounded-memory sort-based group-by ONLY when ANALYZE statistics bound the group
/// count. No statistics → the gate refuses (the sort-based path runs, `fold_count` unchanged)
/// even when forced; post-ANALYZE with a small NDV → the parallel fold fires and its rows are
/// identical. Fresh sessions per phase defeat the per-session result cache.
#[test]
fn parallel_group_aggregate_under_spill_requires_bounded_stats() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut setup = Session::new(engine);
    run(engine, &mut setup, "CREATE TABLE g (k INT, v INT)");
    for start in (0..3000).step_by(500) {
        let values = (start..start + 500)
            .map(|i: i64| format!("({},{})", i % 10, i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut setup,
            &format!("INSERT INTO g VALUES {values}"),
        );
    }
    let sql = "SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v) FROM g GROUP BY k ORDER BY k";
    let baseline = buffered(engine, &mut Session::new(engine), sql);

    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes: 1 << 16,
    }));
    // No statistics: the parallel fold must refuse (memory proof missing) even when forced.
    {
        let _v = nusadb_sql::vectorized::scope(true);
        let _p = nusadb_sql::vectorized::parallel_scope(true);
        let before = nusadb_sql::vectorized::fold_count();
        let no_stats = buffered(engine, &mut Session::new(engine), sql);
        assert_eq!(
            nusadb_sql::vectorized::fold_count(),
            before,
            "parallel fold must refuse under spill without statistics"
        );
        assert_eq!(no_stats, baseline, "sort-based spill path mismatch");
    }
    // With statistics bounding the groups (NDV 10): the parallel fold fires, rows identical.
    run(engine, &mut setup, "ANALYZE g");
    {
        let _v = nusadb_sql::vectorized::scope(true);
        let _p = nusadb_sql::vectorized::parallel_scope(true);
        let before = nusadb_sql::vectorized::fold_count();
        let with_stats = buffered(engine, &mut Session::new(engine), sql);
        assert_eq!(
            nusadb_sql::vectorized::fold_count(),
            before + 1,
            "parallel fold must fire under spill with bounded statistics"
        );
        assert_eq!(with_stats, baseline, "parallel rows mismatch under spill");
    }
    set_spill_config(None);
}

/// Routing: with statistics bounding the group state within HALF the spill
/// budget, the GROUP BY hash-folds directly (zero disk, measured 2.1x over the sort fold);
/// rows must match the no-spill fold exactly (sorted compare), NULL keys included, and the
/// route is asserted via a counter so the equivalence can never go vacuous. A state estimate
/// past the budget (tiny-budget case) stays on the sort-based fold.
#[test]
fn stats_routed_hash_fold_under_spill_matches() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut setup = Session::new(engine);
    run(engine, &mut setup, "CREATE TABLE gg (k INT, k2 INT, v INT)");
    for start in (0..20_000_i64).step_by(500) {
        let values = (start..start + 500)
            .map(|i| {
                let k = if i % 97 == 0 {
                    "NULL".to_owned()
                } else {
                    (i % 5000).to_string()
                };
                format!("({k},{},{})", i % 7, i % 89)
            })
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut setup,
            &format!("INSERT INTO gg VALUES {values}"),
        );
    }
    run(engine, &mut setup, "ANALYZE gg");
    let queries = [
        "SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v) FROM gg GROUP BY k",
        "SELECT k, k2, COUNT(*), SUM(v) FROM gg GROUP BY k, k2",
        // The audit-caught shape: a WHERE above a projection-narrowed scan (column k pruned,
        // so the group key's narrowed ordinal differs from its source ordinal) — the NDV
        // lookup must map through the scan's kept columns, not misread a pruned column.
        "SELECT k2, COUNT(*), SUM(v) FROM gg WHERE v > 0 GROUP BY k2",
    ];
    for sql in queries {
        set_spill_config(None);
        let want = buffered(engine, &mut Session::new(engine), sql);
        // Budget comfortably above the ~5000-group state estimate → the statistics-routed
        // DIRECT hash fold engages (fires-asserted).
        set_spill_config(Some(SpillConfig {
            dir: std::env::temp_dir(),
            threshold_bytes: 64 << 20,
        }));
        let before = nusadb_sql::executor::agg::stats_hash_agg_count();
        let got = buffered(engine, &mut Session::new(engine), sql);
        assert_eq!(
            nusadb_sql::executor::agg::stats_hash_agg_count(),
            before + 1,
            "stats-routed hash fold did not fire for `{sql}`"
        );
        assert_eq!(got, want, "hash fold under spill mismatch for `{sql}`");
        // A budget far below the estimate keeps the sort-based fold — same rows.
        set_spill_config(Some(SpillConfig {
            dir: std::env::temp_dir(),
            threshold_bytes: 1 << 16,
        }));
        let sorted_path = buffered(engine, &mut Session::new(engine), sql);
        set_spill_config(None);
        assert_eq!(sorted_path, want, "sort fold mismatch for `{sql}`");
    }

    // Route-flip regression for the audit-caught mis-map: group key `k2` (NDV 7, ~2.5KB state)
    // sits AFTER the pruned column `k` (NDV ~5000, ~1.8MB) in the table. With a 16KB budget the
    // correct mapping fires the hash arm; the old bug read `k`'s NDV through the Filter-wrapped
    // narrowed scan and would have routed to sort (counter unchanged).
    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes: 1 << 14,
    }));
    let sql = "SELECT k2, COUNT(*), SUM(v) FROM gg WHERE v >= 0 GROUP BY k2";
    let before = nusadb_sql::executor::agg::stats_hash_agg_count();
    let got = buffered(engine, &mut Session::new(engine), sql);
    set_spill_config(None);
    assert_eq!(
        nusadb_sql::executor::agg::stats_hash_agg_count(),
        before + 1,
        "narrowed-scan NDV mapping regressed (hash arm did not fire)"
    );
    assert_eq!(got, buffered(engine, &mut Session::new(engine), sql));
}
