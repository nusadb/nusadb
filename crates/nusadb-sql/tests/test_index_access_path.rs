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

/// Acceptance probe: point-get by `PRIMARY KEY` at 500k rows. The `SeqScan` plan measured
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
fn a_partial_index_is_not_a_scan_candidate() {
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

    // The partial index is not offered (and the expression index does not cover `a`) → SeqScan.
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

#[test]
fn a_composite_index_serves_a_key_prefix_and_returns_the_same_rows_as_a_scan() {
    // Equality on the leading columns, optionally followed by a range on the next one, is served by
    // the composite index (primary key or secondary); the rows are exactly those of the same query
    // over an unindexed copy.
    let (engine, mut session) = fresh();
    for (table, keys) in [("ix", ", PRIMARY KEY (ws, ent, d)"), ("nx", "")] {
        run(
            engine,
            &mut session,
            &format!("CREATE TABLE {table} (ws INT NOT NULL, ent INT NOT NULL, d INT NOT NULL, tag TEXT, v INT{keys})"),
        )
        .unwrap();
        // Several rows per prefix, NULLs in the secondary key, and prefixes on both sides of every
        // bound the predicates use.
        run(
            engine,
            &mut session,
            &format!(
                "INSERT INTO {table} SELECT i % 4, (i / 4) % 5, i / 20, \
                 CASE i % 3 WHEN 0 THEN 'a' WHEN 1 THEN 'b' ELSE NULL END, i % 7 \
                 FROM generate_series(0, 399) AS g(i)"
            ),
        )
        .unwrap();
    }
    run(engine, &mut session, "CREATE INDEX ix_tag_v ON ix (tag, v)").unwrap();
    run(engine, &mut session, "ANALYZE ix").unwrap();

    let mut predicates = Vec::new();
    for ws in [0, 2, 5] {
        predicates.push(format!("ws = {ws}"));
        predicates.push(format!("ws > {ws}"));
        for ent in [0, 3] {
            predicates.push(format!("ws = {ws} AND ent = {ent}"));
            predicates.push(format!("ws = {ws} AND ent >= {ent}"));
            predicates.push(format!("ws = {ws} AND ent < {ent}"));
            for op in ["=", "<", "<=", ">", ">="] {
                predicates.push(format!("ws = {ws} AND ent = {ent} AND d {op} 10"));
            }
            predicates.push(format!("ws = {ws} AND ent = {ent} AND d BETWEEN 5 AND 12"));
        }
    }
    for tag in ["a", "b"] {
        predicates.push(format!("tag = '{tag}'"));
        predicates.push(format!("tag = '{tag}' AND v = 3"));
        predicates.push(format!("tag = '{tag}' AND v > 4"));
    }
    let mut indexed = 0;
    for p in &predicates {
        let plan = explain(engine, &mut session, &format!("SELECT * FROM ix WHERE {p}"));
        if plan.contains("IndexScan") {
            indexed += 1;
        }
        let got = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM ix WHERE {p} ORDER BY ws, ent, d"),
            )
            .unwrap(),
        );
        let want = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM nx WHERE {p} ORDER BY ws, ent, d"),
            )
            .unwrap(),
        );
        assert_eq!(got, want, "{p}: {plan}");
    }
    // A bound that keeps most of the table (`ws > 0`) is rightly left to a scan by the cost gate.
    assert!(
        indexed * 4 >= predicates.len() * 3,
        "{indexed} of {} predicates used an index",
        predicates.len()
    );

    // The full primary key is a unique point lookup; a prefix is a range over the same index.
    let point = explain(
        engine,
        &mut session,
        "SELECT v FROM ix WHERE ws = 1 AND ent = 2 AND d = 3",
    );
    assert!(point.contains("IndexScan: ix using ix_pkey"), "{point}");
    let prefix = explain(
        engine,
        &mut session,
        "SELECT v FROM ix WHERE ws = 1 AND ent = 2",
    );
    assert!(prefix.contains("IndexScan: ix using ix_pkey"), "{prefix}");
    let secondary = explain(
        engine,
        &mut session,
        "SELECT v FROM ix WHERE tag = 'a' AND v = 2",
    );
    assert!(
        secondary.contains("IndexScan: ix using ix_tag_v"),
        "{secondary}"
    );
}

#[test]
fn the_index_bounding_the_most_columns_serves_the_query() {
    let (engine, mut session) = fresh();
    // With a single-column index on the leading column also available, the index that bounds the
    // most columns serves the query.
    let setup = [
        "CREATE TABLE z (a INT NOT NULL, b INT NOT NULL, c INT)",
        "INSERT INTO z SELECT i % 50, i % 7, i FROM generate_series(1, 2000) AS g(i)",
        "CREATE INDEX z_a ON z (a)",
        "CREATE INDEX z_ab ON z (a, b)",
        "ANALYZE z",
    ];
    for sql in setup {
        run(engine, &mut session, sql).unwrap();
    }
    let both = explain(
        engine,
        &mut session,
        "SELECT c FROM z WHERE a = 1 AND b = 2",
    );
    assert!(both.contains("IndexScan: z using z_ab"), "{both}");
}

#[test]
fn a_unique_point_lookup_beats_a_longer_non_unique_match() {
    let (engine, mut session) = fresh();
    for sql in [
        "CREATE TABLE acct (id INT PRIMARY KEY, tenant TEXT NOT NULL, status TEXT NOT NULL, v INT)",
        "INSERT INTO acct SELECT i, 't' || (i % 5), CASE i % 2 WHEN 0 THEN 'on' ELSE 'off' END, i FROM generate_series(1, 500) AS g(i)",
        "CREATE INDEX acct_tenant_status ON acct (tenant, status)",
        "ANALYZE acct",
    ] {
        run(engine, &mut session, sql).unwrap();
    }
    let sql = "SELECT v FROM acct WHERE id = 42 AND tenant = 't2' AND status = 'on'";
    let plan = explain(engine, &mut session, sql);
    assert!(plan.contains("IndexScan: acct using acct_pkey"), "{plan}");
    assert_eq!(
        rows(run(engine, &mut session, sql).unwrap()),
        vec![vec![Value::Int(42)]]
    );
    // The UPDATE finds its row through the same unique lookup.
    run(
        engine,
        &mut session,
        "UPDATE acct SET v = 0 WHERE id = 42 AND tenant = 't2' AND status = 'on'",
    )
    .unwrap();
    assert_eq!(
        rows(run(engine, &mut session, sql).unwrap()),
        vec![vec![Value::Int(0)]]
    );
}

#[test]
fn composite_keys_of_text_numeric_and_date_match_a_scan() {
    let (engine, mut session) = fresh();
    for (table, keys) in [("ck", ", PRIMARY KEY (name, amount, day)"), ("cn", "")] {
        run(
            engine,
            &mut session,
            &format!("CREATE TABLE {table} (name TEXT NOT NULL, amount NUMERIC NOT NULL, day DATE NOT NULL, v INT{keys})"),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!(
                "INSERT INTO {table} SELECT 'n' || (i % 6), CAST(i % 4 AS NUMERIC) / 2, \
                 CAST('2026-01-01' AS DATE) + (i / 12), i FROM generate_series(0, 479) AS g(i)"
            ),
        )
        .unwrap();
    }
    run(engine, &mut session, "ANALYZE ck").unwrap();
    let predicates = [
        "name = 'n2'",
        "name > 'n2'",
        "name BETWEEN 'n1' AND 'n3'",
        "name = 'n2' AND amount = 0.5",
        "name = 'n2' AND amount > 0.5",
        "name = 'n2' AND amount <= 1",
        "name = 'n2' AND amount = 1.5 AND day = '2026-01-05'",
        "name = 'n2' AND amount = 1.5 AND day > '2026-01-10'",
        "name = 'n2' AND amount = 1.5 AND day BETWEEN '2026-01-03' AND '2026-01-12'",
    ];
    for p in predicates {
        let got = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM ck WHERE {p} ORDER BY v"),
            )
            .unwrap(),
        );
        let want = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM cn WHERE {p} ORDER BY v"),
            )
            .unwrap(),
        );
        assert_eq!(got, want, "{p}");
    }
    // UPDATE and DELETE through the full key and through a prefix.
    for (table_sql, check) in [
        (
            "UPDATE TABLE_NAME SET v = -1 WHERE name = 'n3' AND amount = 1 AND day = '2026-01-02'",
            "v = -1",
        ),
        (
            "UPDATE TABLE_NAME SET v = -2 WHERE name = 'n4' AND amount = 0.5",
            "v = -2",
        ),
        ("DELETE FROM TABLE_NAME WHERE name = 'n5'", "name = 'n5'"),
        (
            "DELETE FROM TABLE_NAME WHERE name = 'n1' AND amount > 0.5",
            "name = 'n1'",
        ),
    ] {
        for t in ["ck", "cn"] {
            run(engine, &mut session, &table_sql.replace("TABLE_NAME", t)).unwrap();
        }
        let got = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM ck WHERE {check} ORDER BY v"),
            )
            .unwrap(),
        );
        let want = rows(
            run(
                engine,
                &mut session,
                &format!("SELECT * FROM cn WHERE {check} ORDER BY v"),
            )
            .unwrap(),
        );
        assert_eq!(got, want, "{table_sql}");
    }
    let indexed = rows(run(engine, &mut session, "SELECT * FROM ck ORDER BY v").unwrap());
    let scanned = rows(run(engine, &mut session, "SELECT * FROM cn ORDER BY v").unwrap());
    assert_eq!(indexed, scanned);
}

/// Indexes keyed on expressions, one per key kind: the comparisons the differential runs on each.
const EXPRESSION_KEYS: &[(&str, &[&str])] = &[
    (
        "lower(s)",
        &[
            "= 's7'",
            "= 'S7'",
            "= 'missing'",
            "< 's2'",
            "BETWEEN 's1' AND 's3'",
        ],
    ),
    ("payload->>'k'", &["= 'k3'", "= 'k99'", ">= 'k8'"]),
    ("a + b", &["= 7", "= 0", "> 10", "= 7.0", "= '7'"]),
    ("si + si", &["= 4", "= 9", "< 3"]),
    ("a * 2", &["= 8", "= 9", "= 4.0"]),
    ("n * 2", &["= 3", "= 3.00", "= 2.5", "> 10"]),
    ("n + a", &["= 4.5", "= 5"]),
    ("f * 2", &["= 3", "= 3.0", "< 2"]),
    ("d + 1", &["= DATE '2024-01-05'", "> DATE '2024-01-20'"]),
    ("coalesce(s, 'none')", &["= 'none'", "= 's4'"]),
    ("CAST(a AS TEXT)", &["= '3'", "= 3"]),
    ("length(s)", &["= 2", "= 3", "> 2"]),
    ("abs(a - 5)", &["= 2", "= 0"]),
    (
        "CASE WHEN a > 3 THEN 'hi' ELSE 'lo' END",
        &["= 'hi'", "= 'lo'"],
    ),
    (
        "upper(s) || '-' || CAST(b AS TEXT)",
        &["= 'S7-1'", "> 'S5'"],
    ),
    ("a + b, lower(s)", &["= 7"]),
];

/// Two copies of the same rows: `ex` with one index per [`EXPRESSION_KEYS`] entry, `nx` without,
/// both changed after the indexes were built.
fn expression_tables() -> (&'static BtreeEngine, Session<'static>) {
    let (engine, mut session) = fresh();
    for table in ["ex", "nx"] {
        run(
            engine,
            &mut session,
            &format!(
                "CREATE TABLE {table} (id INT PRIMARY KEY, a INT, b INT, si SMALLINT, \
                 n NUMERIC(10, 2), f DOUBLE PRECISION, d DATE, s TEXT, payload JSONB)"
            ),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!(
                "INSERT INTO {table} SELECT i, i % 9, i % 4, CAST(i % 5 AS SMALLINT), \
                 CAST(i AS NUMERIC(10, 2)) / 4, CAST(i AS DOUBLE PRECISION) / 2, \
                 DATE '2024-01-01' + i % 30, \
                 CASE WHEN i % 11 = 0 THEN NULL WHEN i % 2 = 0 THEN 's' || (i % 13) ELSE 'S' || (i % 13) END, \
                 CASE WHEN i % 7 = 0 THEN CAST('{{}}' AS JSONB) \
                      ELSE CAST('{{\"k\": \"k' || (i % 10) || '\"}}' AS JSONB) END \
                 FROM generate_series(1, 300) AS g(i)"
            ),
        )
        .unwrap();
    }
    for (i, (key, _)) in EXPRESSION_KEYS.iter().enumerate() {
        run(
            engine,
            &mut session,
            &format!("CREATE INDEX ex_k{i} ON ex (({key}))").replace(", lower", "), (lower"),
        )
        .unwrap();
    }
    // Rows written after the build, changed and deleted rows keep every index in step.
    for table in ["ex", "nx"] {
        for sql in [
            format!(
                "INSERT INTO {table} VALUES (1000, 3, 4, 2, 1.5, 1.5, DATE '2024-01-05', 's7', CAST('{{\"k\": \"k3\"}}' AS JSONB))"
            ),
            format!("UPDATE {table} SET s = 's7', a = a + 1 WHERE id % 10 = 3"),
            format!(
                "UPDATE {table} SET payload = CAST('{{\"k\": \"k99\"}}' AS JSONB) WHERE id % 50 = 1"
            ),
            format!("DELETE FROM {table} WHERE id % 17 = 5"),
        ] {
            run(engine, &mut session, &sql).unwrap();
        }
    }
    (engine, session)
}

#[test]
fn an_expression_index_serves_its_expression_and_returns_the_same_rows_as_a_scan() {
    // The index key is evaluated per row on write; a predicate on the same expression is served by
    // that index and must return exactly the rows of the same query over an unindexed copy, for
    // every key type and for a literal of another type than the expression (coerced or refused).
    let (engine, mut session) = expression_tables();
    let mut indexed = 0;
    let mut total = 0;
    for (key, comparisons) in EXPRESSION_KEYS {
        let lead = key.split(", lower").next().unwrap_or(key);
        for cmp in *comparisons {
            let p = format!("{lead} {cmp}");
            // A comparison the analyzer refuses has no plan; both tables must refuse it alike.
            let plan = run(
                engine,
                &mut session,
                &format!("EXPLAIN SELECT id FROM ex WHERE {p}"),
            )
            .map(|r| format!("{:?}", rows(r)))
            .unwrap_or_default();
            total += 1;
            if plan.contains("IndexScan") {
                indexed += 1;
            }
            let query = |t: &str| format!("SELECT id FROM {t} WHERE {p} ORDER BY id");
            let got = run(engine, &mut session, &query("ex"));
            let want = run(engine, &mut session, &query("nx"));
            match (got, want) {
                (Ok(got), Ok(want)) => assert_eq!(rows(got), rows(want), "{p}: {plan}"),
                (Err(got), Err(want)) => assert_eq!(got.to_string(), want.to_string(), "{p}"),
                (got, want) => panic!("{p}: indexed {got:?} vs scan {want:?}\n{plan}"),
            }
        }
    }
    assert!(
        indexed * 3 >= total * 2,
        "{indexed} of {total} predicates used an index"
    );

    let plan = explain(
        engine,
        &mut session,
        "SELECT id FROM ex WHERE payload->>'k' = 'k3'",
    );
    assert!(plan.contains("IndexScan: ex using ex_k1"), "{plan}");
    let plan = explain(
        engine,
        &mut session,
        "SELECT id FROM ex WHERE 'k3' = payload->>'k'",
    );
    assert!(plan.contains("IndexScan: ex using ex_k1"), "{plan}");
    // Pushed onto either side of a join, the expression is that table's own: the index serves it
    // and the joined rows match the same join over unindexed copies.
    run(engine, &mut session, "CREATE TABLE ny AS SELECT * FROM nx").unwrap();
    for (p, flip) in [("lower(X.s) = 's7'", false), ("X.a + X.b = 7", true)] {
        let sql = |x: &str| {
            let (l, r) = if flip { (x, "ny") } else { ("ny", x) };
            let p = p.replace("X.", &format!("{x}."));
            format!(
                "SELECT {l}.id, {r}.id FROM {l} JOIN {r} ON {l}.id = {r}.id + 1 WHERE {p} ORDER BY 1"
            )
        };
        let plan = explain(engine, &mut session, &sql("ex"));
        assert!(plan.contains("IndexScan: ex using"), "{p}: {plan}");
        let got = rows(run(engine, &mut session, &sql("ex")).unwrap());
        assert!(!got.is_empty(), "{p}");
        assert_eq!(
            got,
            rows(run(engine, &mut session, &sql("nx")).unwrap()),
            "{p}"
        );
    }
    // A literal on the left flips the comparison: `'s2' > lower(s)` is `lower(s) < 's2'`.
    for p in ["'s2' > lower(s)", "'k8' <= payload->>'k'", "10 < a + b"] {
        let plan = explain(
            engine,
            &mut session,
            &format!("SELECT id FROM ex WHERE {p}"),
        );
        assert!(plan.contains("IndexScan"), "{p}: {plan}");
        let query = |t: &str| format!("SELECT id FROM {t} WHERE {p} ORDER BY id");
        let got = rows(run(engine, &mut session, &query("ex")).unwrap());
        assert!(!got.is_empty(), "{p}");
        assert_eq!(
            got,
            rows(run(engine, &mut session, &query("nx")).unwrap()),
            "{p}"
        );
    }
}

#[test]
fn a_unique_expression_index_is_a_point_lookup_only_on_its_whole_key() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE eu (id INT PRIMARY KEY, a INT, s TEXT)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO eu VALUES (1, 1, 'X'), (2, 2, 'x'), (3, 3, 'y')",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "CREATE UNIQUE INDEX eu_k ON eu ((lower(s)), (a * 10))",
    )
    .unwrap();
    // Equality on the first key expression alone matches two rows.
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT id FROM eu WHERE lower(s) = 'x' ORDER BY id",
        )
        .unwrap(),
    );
    assert_eq!(got, vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT id FROM eu WHERE lower(s) = 'x' AND a * 10 = 20",
        )
        .unwrap(),
    );
    assert_eq!(got, vec![vec![Value::Int(2)]]);
    // Only equality on every key expression bounds the lookup to one row.
    let point = |sql: &str| {
        let logical = analyze(parse(sql).unwrap(), &Cat { engine }).unwrap();
        nusadb_sql::plan_is_inline_point_get(&plan(logical))
    };
    assert!(point(
        "SELECT id FROM eu WHERE lower(s) = 'x' AND a * 10 = 20"
    ));
    assert!(!point("SELECT id FROM eu WHERE lower(s) = 'x'"));
    let err = run(engine, &mut session, "INSERT INTO eu VALUES (4, 1, 'x')");
    assert!(err.is_err(), "the unique key (x, 10) already exists");
}

#[test]
fn an_expression_index_is_chosen_by_cost_once_the_table_is_analyzed() {
    // With statistics an equality on an indexed expression is still estimated selective (no column
    // statistics describe the expression, so the per-operator default applies) and takes the index;
    // a predicate on a different expression does not.
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE ea (id INT PRIMARY KEY, a INT, b INT, payload JSONB)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO ea SELECT i, i % 100, i % 7, CAST('{\"k\": \"k' || (i % 500) || '\"}' AS JSONB) \
         FROM generate_series(1, 5000) AS g(i)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "CREATE INDEX ea_k ON ea ((payload->>'k'))",
    )
    .unwrap();
    run(engine, &mut session, "CREATE INDEX ea_sum ON ea ((a + b))").unwrap();
    run(engine, &mut session, "ANALYZE ea").unwrap();
    for (p, index) in [("payload->>'k' = 'k42'", "ea_k"), ("a + b = 50", "ea_sum")] {
        let plan = explain(
            engine,
            &mut session,
            &format!("SELECT id FROM ea WHERE {p}"),
        );
        assert!(
            plan.contains(&format!("IndexScan: ea using {index}")),
            "{p}: {plan}"
        );
    }
    for p in ["payload->>'j' = 'k42'", "a - b = 50", "b + a = 50"] {
        let plan = explain(
            engine,
            &mut session,
            &format!("SELECT id FROM ea WHERE {p}"),
        );
        assert!(!plan.contains("IndexScan"), "{p}: {plan}");
    }
    let got = rows(
        run(
            engine,
            &mut session,
            "SELECT count(*) FROM ea WHERE payload->>'k' = 'k42'",
        )
        .unwrap(),
    );
    assert_eq!(got, vec![vec![Value::Int(10)]]);
}

#[test]
fn an_expression_that_depends_on_the_session_or_the_moment_is_not_a_scan_path() {
    // A key computed under one session time zone is not the value a session in another zone
    // computes for the same row, and `now()` / `random()` change between write and read: such an
    // index is maintained but never scanned, so every zone reads the rows a scan reads.
    let (engine, mut session) = fresh();
    for table in ["tz", "tn"] {
        run(
            engine,
            &mut session,
            &format!(
                "CREATE TABLE {table} (id INT PRIMARY KEY, ts TIMESTAMPTZ, lt TIMESTAMP, a INT)"
            ),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!(
                "INSERT INTO {table} SELECT i, \
                 CAST('2024-01-01 00:00:00+00' AS TIMESTAMPTZ) + i * INTERVAL '3 hours', \
                 CAST('2024-01-01 00:00:00' AS TIMESTAMP) + i * INTERVAL '3 hours', i % 10 \
                 FROM generate_series(1, 200) AS g(i)"
            ),
        )
        .unwrap();
    }
    run(engine, &mut session, "SET TIME ZONE 'UTC'").unwrap();
    let keys = [
        "CAST(ts AS DATE)",
        "CAST(ts AS TEXT)",
        "extract(hour FROM ts)",
        "CAST(lt AS TIMESTAMPTZ)",
        "a + random()",
        "CAST(now() AS DATE)",
        "CAST(lt AS DATE)",
        "CAST(random() * 10 AS INT)",
        "CAST(localtimestamp AS DATE)",
        "localtimestamp",
    ];
    for (i, key) in keys.iter().enumerate() {
        run(
            engine,
            &mut session,
            &format!("CREATE INDEX tz_k{i} ON tz (({key}))"),
        )
        .unwrap();
    }
    let predicates = [
        "CAST(ts AS DATE) = DATE '2024-01-05'",
        "CAST(ts AS TEXT) > '2024-01-05'",
        "extract(hour FROM ts) = 5",
        "CAST(lt AS TIMESTAMPTZ) = CAST('2024-01-02 03:00:00+00' AS TIMESTAMPTZ)",
        "CAST(lt AS DATE) = DATE '2024-01-05'",
    ];
    for zone in ["UTC", "+07", "-08"] {
        run(engine, &mut session, &format!("SET TIME ZONE '{zone}'")).unwrap();
        for p in predicates {
            let query = |t: &str| format!("SELECT id FROM {t} WHERE {p} ORDER BY id");
            let got = rows(run(engine, &mut session, &query("tz")).unwrap());
            let want = rows(run(engine, &mut session, &query("tn")).unwrap());
            assert_eq!(got, want, "{zone}: {p}");
        }
    }
    let moment = [
        "a + random() > 5",
        "CAST(now() AS DATE) = DATE '2024-01-05'",
        "CAST(random() * 10 AS INT) = 3",
        "CAST(localtimestamp AS DATE) = DATE '2024-01-05'",
        "localtimestamp > TIMESTAMP '2024-01-05 00:00:00'",
    ];
    for p in predicates[..4].iter().chain(&moment) {
        let plan = explain(
            engine,
            &mut session,
            &format!("SELECT id FROM tz WHERE {p}"),
        );
        assert!(!plan.contains("IndexScan"), "{p}: {plan}");
    }
    // An index on an expression with no zone or moment in it is still a scan path.
    let plan = explain(
        engine,
        &mut session,
        "SELECT id FROM tz WHERE CAST(lt AS DATE) = DATE '2024-01-05'",
    );
    assert!(plan.contains("IndexScan: tz using tz_k6"), "{plan}");
}
