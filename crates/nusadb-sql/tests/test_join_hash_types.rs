//! (QA critical): equi-joins on `BIGINT`/`SMALLINT`/`NUMERIC` (and the
//! other exact types) must plan a `HashJoin`, not the O(n²) `NestedLoopJoin` — `BIGINT` is the
//! standard ID type, so before this fix almost every real-world join was quadratic (QA measured
//! 5k×5k ≈ 6s; 300k×300k hung). The hash is **compare-compatible** by construction: every widened
//! key type maps values the evaluator calls equal to equal `KeyAtom`s (NUMERIC canonicalizes
//! through the trimmed exact decimal, so `1.0` joins `1.00`).

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

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn run(
    engine: &'static BtreeEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical))
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

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

/// Every widened key type plans a `HashJoin` and returns the same matches the nested loop would.
#[test]
fn widened_key_types_plan_hash_join_with_correct_matches() {
    let cases: &[(&str, &str, &str, &str)] = &[
        // (column type, left values, right values, matching key literal)
        ("BIGINT", "(1, 10), (2, 20)", "(2, 200), (3, 300)", "2"),
        ("SMALLINT", "(1, 10), (2, 20)", "(2, 200), (3, 300)", "2"),
        (
            "NUMERIC(10,2)",
            "(1.50, 10), (2.25, 20)",
            "(2.25, 200), (9.00, 300)",
            "2.25",
        ),
        (
            "TIMESTAMP",
            "('2026-01-01 12:00:00', 10), ('2026-06-01 08:30:00', 20)",
            "('2026-06-01 08:30:00', 200), ('2026-12-31 23:59:59', 300)",
            "'2026-06-01 08:30:00'",
        ),
        (
            "DATE",
            "('2026-01-01', 10), ('2026-06-01', 20)",
            "('2026-06-01', 200), ('2026-12-31', 300)",
            "'2026-06-01'",
        ),
        (
            "VARCHAR(20)",
            "('alpha', 10), ('beta', 20)",
            "('beta', 200), ('gamma', 300)",
            "'beta'",
        ),
        (
            "UUID",
            "('11111111-1111-1111-1111-111111111111', 10), ('22222222-2222-2222-2222-222222222222', 20)",
            "('22222222-2222-2222-2222-222222222222', 200), ('33333333-3333-3333-3333-333333333333', 300)",
            "'22222222-2222-2222-2222-222222222222'",
        ),
    ];
    for (ty, left_vals, right_vals, key) in cases {
        let (engine, mut session) = fresh();
        run(
            engine,
            &mut session,
            &format!("CREATE TABLE l (k {ty}, lv INT)"),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!("CREATE TABLE r (k {ty}, rv INT)"),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!("INSERT INTO l VALUES {left_vals}"),
        )
        .unwrap();
        run(
            engine,
            &mut session,
            &format!("INSERT INTO r VALUES {right_vals}"),
        )
        .unwrap();

        let sql = "SELECT l.lv, r.rv FROM l JOIN r ON l.k = r.k";
        let plan_text = explain(engine, &mut session, sql);
        assert!(
            plan_text.contains("HashJoin"),
            "{ty} equi-join must plan a HashJoin, got:\n{plan_text}"
        );
        let got = rows(run(engine, &mut session, sql).unwrap());
        assert_eq!(
            got,
            vec![vec![Value::Int(20), Value::Int(200)]],
            "{ty} join must match exactly the shared key {key}"
        );
    }
}

/// NUMERIC keys hash through the trimmed exact decimal: values of different declared scale that
/// compare equal (`2.5` vs `2.50`) land in one bucket and join.
#[test]
fn numeric_join_matches_across_scales() {
    let (engine, mut session) = fresh();
    run(
        engine,
        &mut session,
        "CREATE TABLE l (k NUMERIC(10,1), lv INT)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "CREATE TABLE r (k NUMERIC(10,1), rv INT)",
    )
    .unwrap();
    run(engine, &mut session, "INSERT INTO l VALUES (2.5, 1)").unwrap();
    // The literal 2.50 rescales into the column; equality must survive whatever spelling reaches
    // the join key.
    run(engine, &mut session, "INSERT INTO r VALUES (2.50, 2)").unwrap();
    let sql = "SELECT l.lv, r.rv FROM l JOIN r ON l.k = r.k";
    assert!(explain(engine, &mut session, sql).contains("HashJoin"));
    let got = rows(run(engine, &mut session, sql).unwrap());
    assert_eq!(got, vec![vec![Value::Int(1), Value::Int(2)]]);
}

/// NULL keys never match (SQL `NULL = NULL` is unknown) — including through the widened types —
/// and LEFT JOIN still NULL-pads the unmatched side.
#[test]
fn null_keys_do_not_match_and_outer_join_pads() {
    let (engine, mut session) = fresh();
    run(engine, &mut session, "CREATE TABLE l (k BIGINT, lv INT)").unwrap();
    run(engine, &mut session, "CREATE TABLE r (k BIGINT, rv INT)").unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO l VALUES (NULL, 1), (7, 2)",
    )
    .unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO r VALUES (NULL, 3), (7, 4), (8, 5)",
    )
    .unwrap();
    let sql = "SELECT l.lv, r.rv FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.lv";
    assert!(explain(engine, &mut session, sql).contains("HashJoin"));
    let got = rows(run(engine, &mut session, sql).unwrap());
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1), Value::Null], // NULL key matches nothing
            vec![Value::Int(2), Value::Int(4)],
        ]
    );
}

/// Float keys stay on the nested loop (NaN / `-0.0` hashing hazards) — the plan must NOT change.
#[test]
fn float_keys_still_fall_back_to_nested_loop() {
    let (engine, mut session) = fresh();
    run(engine, &mut session, "CREATE TABLE l (k FLOAT, lv INT)").unwrap();
    run(engine, &mut session, "CREATE TABLE r (k FLOAT, rv INT)").unwrap();
    run(engine, &mut session, "INSERT INTO l VALUES (1.5e0, 1)").unwrap();
    run(engine, &mut session, "INSERT INTO r VALUES (1.5e0, 2)").unwrap();
    let sql = "SELECT l.lv, r.rv FROM l JOIN r ON l.k = r.k";
    let plan_text = explain(engine, &mut session, sql);
    assert!(
        plan_text.contains("NestedLoopJoin"),
        "float keys must keep the nested loop, got:\n{plan_text}"
    );
    let got = rows(run(engine, &mut session, sql).unwrap());
    assert_eq!(got, vec![vec![Value::Int(1), Value::Int(2)]]);
}
