//! The production catalog adapters expose the engine's indexes —
//! **including the `PRIMARY KEY`/`UNIQUE` constraint-backing ones** — so the
//! fundamental OLTP point-get plans an `IndexScan` (O(log n)) instead of a full-table `SeqScan`
//! (O(n)). The backing indexes are maintained on every write path (INSERT/UPDATE/upsert/COPY,
//! `ALTER` rewrites, matview refresh) and the engine skips its byte-level unique check for them
//! (the SQL layer's scan-based checks + key locks own the constraint semantics), so exposing
//! them changes plans, never results.
//!
//! The harness catalog delegates to [`nusadb_sql::catalog_list_indexes`] /
//! [`nusadb_sql::catalog_table_stats`] — the exact shared body the wire (`EngineCatalog`),
//! `SessionCatalog` (PREPARE/EXECUTE), and `ExecCatalog` (matview refresh) adapters use — so
//! every assertion here exercises the production planning logic.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, ast::Value, parse, plan,
};

/// The production adapter shape: tables, indexes, and stats resolved from the engine.
struct Cat<'a> {
    engine: &'a BtreeEngine,
}

impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_list_indexes(self.engine, txn, table);
        let _ = self.engine.commit(txn);
        out
    }
    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_table_stats(self.engine, txn, table);
        let _ = self.engine.commit(txn);
        out
    }
}

fn run(
    engine: &'static BtreeEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat { engine })?;
    session.execute(plan(logical))
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

/// The EXPLAIN plan text for `sql`.
fn explain(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> String {
    let out = rows(run(engine, session, &format!("EXPLAIN {sql}")).unwrap());
    out.iter()
        .map(|row| match &row[..] {
            [Value::Text(line)] => line.clone(),
            other => panic!("unexpected EXPLAIN row {other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn fresh() -> (&'static BtreeEngine, Session<'static>) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let session = Session::new(engine);
    (engine, session)
}

/// The headline fix: a point-get / range by PRIMARY KEY plans an `IndexScan` over the
/// constraint-backing index (previously always a `SeqScan`), and returns exactly the same rows.
#[test]
fn pk_point_get_and_range_use_the_backing_index() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
    )
    .unwrap();
    for i in 0..50 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'v{i}')"),
        )
        .unwrap();
    }

    let point = explain(engine, &mut session, "SELECT v FROM t WHERE id = 7");
    assert!(
        point.contains("IndexScan"),
        "point-get by PK must plan an IndexScan, got:\n{point}"
    );
    let range = explain(
        engine,
        &mut session,
        "SELECT v FROM t WHERE id > 45 AND id <= 48",
    );
    assert!(
        range.contains("IndexScan"),
        "range by PK must plan an IndexScan (backing index is ordered), got:\n{range}"
    );

    // Results are identical to the sequential semantics.
    let got = rows(run(engine, &mut session, "SELECT v FROM t WHERE id = 7").unwrap());
    assert_eq!(got, vec![vec![Value::Text("v7".to_owned())]]);
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT id FROM t WHERE id > 45 AND id <= 48 ORDER BY id",
        )
        .unwrap(),
    );
    assert_eq!(
        got,
        vec![
            vec![Value::Int(46)],
            vec![Value::Int(47)],
            vec![Value::Int(48)],
        ]
    );
}

/// Stale entries from superseded/deleted versions are visibility-filtered: after UPDATE and
/// DELETE, an index-planned point-get sees exactly the committed state.
#[test]
fn index_scan_respects_update_and_delete_visibility() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
    )
    .unwrap();
    for i in 0..20 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'old')"),
        )
        .unwrap();
    }
    run(engine, &mut session, "UPDATE t SET v = 'new' WHERE id = 5").unwrap();
    run(engine, &mut session, "DELETE FROM t WHERE id = 6").unwrap();

    assert!(
        explain(engine, &mut session, "SELECT v FROM t WHERE id = 5").contains("IndexScan"),
        "sanity: the probe below must actually run through the index plan"
    );
    let got = rows(run(engine, &mut session, "SELECT v FROM t WHERE id = 5").unwrap());
    assert_eq!(got, vec![vec![Value::Text("new".to_owned())]]);
    let got = rows(run(engine, &mut session, "SELECT v FROM t WHERE id = 6").unwrap());
    assert!(
        got.is_empty(),
        "deleted row must not resurface via the index"
    );
    // A key moved by UPDATE is findable at its new value and gone from the old one.
    run(engine, &mut session, "UPDATE t SET id = 100 WHERE id = 7").unwrap();
    let got = rows(run(engine, &mut session, "SELECT id FROM t WHERE id = 100").unwrap());
    assert_eq!(got, vec![vec![Value::Int(100)]]);
    let got = rows(run(engine, &mut session, "SELECT id FROM t WHERE id = 7").unwrap());
    assert!(got.is_empty());
}

/// UNIQUE columns: NULLs never conflict (the backing index no longer runs the engine's byte-level
/// unique check), duplicates still error via the SQL-layer check, and the point-get uses the index.
#[test]
fn unique_backing_index_keeps_sql_null_and_duplicate_semantics() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE u (id INT PRIMARY KEY, k INT UNIQUE, v TEXT)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO u VALUES (1, 10, 'a'), (2, NULL, 'b'), (3, NULL, 'c')",
    )
    .unwrap();
    // Two NULLs in a UNIQUE column are fine (NULLs are distinct).
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT COUNT(*) FROM u WHERE k IS NULL",
        )
        .unwrap(),
    );
    assert_eq!(got, vec![vec![Value::Int(2)]]);
    // A real duplicate still errors (SQL-layer enforcement).
    assert!(
        run(engine, &mut session, "INSERT INTO u VALUES (4, 10, 'dup')").is_err(),
        "duplicate UNIQUE value must still be rejected"
    );
    // And the unique column's point-get plans through its backing index.
    assert!(
        explain(engine, &mut session, "SELECT v FROM u WHERE k = 10").contains("IndexScan"),
        "point-get by UNIQUE column must plan an IndexScan"
    );
    let got = rows(run(engine, &mut session, "SELECT v FROM u WHERE k = 10").unwrap());
    assert_eq!(got, vec![vec![Value::Text("a".to_owned())]]);
}

/// `ALTER TABLE ADD CONSTRAINT UNIQUE` on a populated table backfills the new backing index, so
/// it is immediately scannable and complete.
#[test]
fn alter_add_unique_backfills_the_backing_index() {
    let (engine, mut session) = fresh();
    run(engine, &mut session, "CREATE TABLE a (id INT, v TEXT)").unwrap();
    for i in 0..30 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO a VALUES ({i}, 'v{i}')"),
        )
        .unwrap();
    }
    run(
        engine,
        &mut session,
        "ALTER TABLE a ADD CONSTRAINT a_id_key UNIQUE (id)",
    )
    .unwrap();
    assert!(
        explain(engine, &mut session, "SELECT v FROM a WHERE id = 12").contains("IndexScan"),
        "the backfilled backing index must be scannable"
    );
    let got = rows(run(engine, &mut session, "SELECT v FROM a WHERE id = 12").unwrap());
    assert_eq!(got, vec![vec![Value::Text("v12".to_owned())]]);
}

/// `ALTER TABLE` layout rewrites (ADD/DROP COLUMN, SET TYPE) supersede every row under a new tid;
/// the rewrites re-index, so index plans keep seeing all rows afterward.
#[test]
fn layout_rewrites_keep_indexes_covering() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE r (id INT PRIMARY KEY, v TEXT, dead INT)",
    )
    .unwrap();
    for i in 0..25 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO r VALUES ({i}, 'v{i}', {i})"),
        )
        .unwrap();
    }
    run(engine, &mut session, "ALTER TABLE r ADD COLUMN extra TEXT").unwrap();
    let got = rows(run(engine, &mut session, "SELECT v FROM r WHERE id = 3").unwrap());
    assert_eq!(got, vec![vec![Value::Text("v3".to_owned())]]);
    run(engine, &mut session, "ALTER TABLE r DROP COLUMN dead").unwrap();
    let got = rows(run(engine, &mut session, "SELECT v FROM r WHERE id = 21").unwrap());
    assert_eq!(got, vec![vec![Value::Text("v21".to_owned())]]);
    // The whole table is still reachable through the index path.
    let got = rows(run(engine, &mut session, "SELECT COUNT(*) FROM r WHERE id >= 0").unwrap());
    assert_eq!(got, vec![vec![Value::Int(25)]]);
}

/// Acceptance probe: point-get by `PRIMARY KEY` at 500k rows — QA measured the `SeqScan` plan at
/// 1272ms; the `IndexScan` plan must answer in well under a millisecond. `#[ignore]`d (manual):
/// `cargo test -p nusadb-sql --release --test test_index_access_path -- --ignored --nocapture`
#[test]
#[ignore = "manual perf probe — run with --release -- --ignored --nocapture"]
fn pk_point_get_at_500k_is_sub_millisecond() {
    const N: usize = 500_000;
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE big (id INT PRIMARY KEY, v INT)",
    )
    .unwrap();
    for start in (0..N).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({i},{})", i % 97))
            .collect::<Vec<_>>()
            .join(",");
        run(
            engine,
            &mut session,
            &format!("INSERT INTO big VALUES {values}"),
        )
        .unwrap();
    }
    assert!(
        explain(engine, &mut session, "SELECT v FROM big WHERE id = 250000").contains("IndexScan"),
        "the probe must run the index plan"
    );
    for round in 1..=3 {
        let t = std::time::Instant::now();
        let got = rows(run(engine, &mut session, "SELECT v FROM big WHERE id = 250000").unwrap());
        let dt = t.elapsed();
        assert_eq!(got, vec![vec![Value::Int(250_000 % 97)]]);
        println!("point-get by PK @500k (round {round}): {dt:?}");
    }
}

/// `BETWEEN` is `>= AND <=` — it must drive the index exactly like the spelled-out form
/// (the BETWEEN spelling full-scanned while `>= AND <=` planned an
/// `IndexScan`), with both endpoints inclusive and `NOT BETWEEN` left to the filter.
#[test]
fn between_plans_an_index_scan_with_inclusive_bounds() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE b (id INT PRIMARY KEY, v TEXT)",
    )
    .unwrap();
    for i in 0..60 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO b VALUES ({i}, 'v{i}')"),
        )
        .unwrap();
    }
    let sql = "SELECT id FROM b WHERE id BETWEEN 10 AND 13 ORDER BY id";
    let plan_text = explain(engine, &mut session, sql);
    assert!(
        plan_text.contains("IndexScan"),
        "BETWEEN must plan an IndexScan like its >=/<= spelling, got:\n{plan_text}"
    );
    let got = rows(run(engine, &mut session, sql).unwrap());
    assert_eq!(
        got,
        vec![
            vec![Value::Int(10)],
            vec![Value::Int(11)],
            vec![Value::Int(12)],
            vec![Value::Int(13)],
        ],
        "both BETWEEN endpoints are inclusive"
    );
    // NOT BETWEEN is not a contiguous range — it stays correct via the retained filter.
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT COUNT(*) FROM b WHERE id NOT BETWEEN 10 AND 13",
        )
        .unwrap(),
    );
    assert_eq!(got, vec![vec![Value::Int(56)]]);
}

/// An explicit `CREATE INDEX` keeps working exactly as before through the shared adapter body.
#[test]
fn explicit_index_still_plans_and_answers() {
    let (engine, mut session) = fresh();
    run(engine, &mut session, "CREATE TABLE e (id INT, v TEXT)").unwrap();
    for i in 0..40 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO e VALUES ({i}, 'v{i}')"),
        )
        .unwrap();
    }
    run(engine, &mut session, "CREATE INDEX e_id ON e (id)").unwrap();
    assert!(
        explain(engine, &mut session, "SELECT v FROM e WHERE id = 9").contains("IndexScan"),
        "explicit index must still be offered"
    );
    let got = rows(run(engine, &mut session, "SELECT v FROM e WHERE id = 9").unwrap());
    assert_eq!(got, vec![vec![Value::Text("v9".to_owned())]]);
}

/// A partial or functional index is NOT offered as an equality/range scan candidate (the planner
/// encodes plain-column ascending bounds, which would not match a computed key nor an index holding
/// only the predicate-satisfying rows). It plans a `SeqScan` and returns the full correct result —
/// crucially, a partial index must not hide the rows it does not cover. (Production path: the
/// harness catalog delegates to `catalog_list_indexes`.)
#[test]
fn partial_and_functional_indexes_are_not_scan_candidates() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE t (id INT, a INT, s TEXT, active BOOL)",
    )
    .unwrap();
    for i in 0..30 {
        let active = if i % 2 == 0 { "TRUE" } else { "FALSE" };
        run(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, {}, 's{i}', {active})", i % 5),
        )
        .unwrap();
    }
    run(
        engine,
        &mut session,
        "CREATE INDEX t_a_partial ON t (a) WHERE active",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "CREATE INDEX t_lower_s ON t (lower(s))",
    )
    .unwrap();

    // Neither index is offered → SeqScan.
    let plan = explain(engine, &mut session, "SELECT id FROM t WHERE a = 2");
    assert!(
        plan.contains("SeqScan") && !plan.contains("IndexScan"),
        "a partial index must not be a scan candidate, got:\n{plan}"
    );
    // And the result covers BOTH active and inactive a=2 rows (ids 2,7,12,17,22,27), not just the
    // active ones a partial index would hold.
    let mut got: Vec<i64> =
        rows(run(engine, &mut session, "SELECT id FROM t WHERE a = 2").unwrap())
            .into_iter()
            .map(|r| match r.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect();
    got.sort_unstable();
    assert_eq!(got, vec![2, 7, 12, 17, 22, 27]);
}
