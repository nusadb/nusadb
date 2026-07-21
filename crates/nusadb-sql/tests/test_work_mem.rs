//! Per-query work-memory budget enforcement.
//!
//! A non-zero `work_mem` makes a query that materializes more than the budget in one executor stage
//! fail with an honest `Error::OutOfMemory` instead of OOM-killing the server. The budget is a
//! process-wide static, so every scenario lives in ONE test (this file is its own test binary, run
//! sequentially) and resets the budget to 0 before returning, so no other test observes a non-zero
//! budget.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, IndexInfo, Session, analyze, parse, plan, set_work_mem, work_mem,
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

/// Run one SQL statement through the full parse → analyze → plan → execute pipeline.
fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Result<(), Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical)).map(|_| ())
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the budget is a process-wide static, so every scenario must live in this ONE \
              sequential test (see the module doc); splitting it would race the budget"
)]
fn work_mem_budget_enforced_then_reset() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    // A table with enough wide rows that a full scan materializes well over a few KiB.
    run(
        engine,
        &mut session,
        "CREATE TABLE big (id INT, payload TEXT)",
    )
    .unwrap();
    for i in 0..400 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO big VALUES ({i}, 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')"),
        )
        .unwrap();
    }

    // Default (work_mem = 0): the full scan succeeds however large.
    assert_eq!(work_mem(), 0, "work_mem defaults to unlimited");
    run(engine, &mut session, "SELECT * FROM big").expect("default budget never refuses");

    // A tight budget makes the same materializing query fail with an honest out-of-memory error.
    set_work_mem(4 * 1024);
    assert_eq!(work_mem(), 4 * 1024);
    match run(engine, &mut session, "SELECT * FROM big") {
        Err(Error::Core(nusadb_core::Error::OutOfMemory(msg))) => {
            assert!(
                msg.contains("work_mem"),
                "message should name work_mem: {msg}"
            );
        },
        other => {
            set_work_mem(0);
            panic!("expected OutOfMemory over work_mem, got {other:?}");
        },
    }

    // A query whose materialized stage is small still runs under the same tight budget. (The
    // row-path executor materializes the full scan before any filter, so a selective predicate on
    // `big` would still buffer all of it — work_mem caps the materialized stage, not the final row
    // count; a genuinely small scan is what fits.)
    run(engine, &mut session, "CREATE TABLE tiny (id INT)").unwrap();
    run(
        engine,
        &mut session,
        "INSERT INTO tiny VALUES (1), (2), (3)",
    )
    .unwrap();
    run(engine, &mut session, "SELECT * FROM tiny")
        .expect("a small materialized stage fits under the budget");

    // `INSERT ... SELECT` streams its source in batches (P-INSERTSEL-OOM): the same budget that
    // refuses materializing `SELECT * FROM big` admits copying `big` into a plain table, because
    // the streaming path holds only one batch at a time. The whole-result copy is what OOMed an
    // ETL-sized backfill before.
    run(
        engine,
        &mut session,
        "CREATE TABLE big_copy (id INT, payload TEXT)",
    )
    .unwrap();
    let logical = analyze(
        parse("INSERT INTO big_copy SELECT id, payload FROM big").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Inserted(n)) => {
            assert_eq!(n, 400, "every source row must be inserted");
        },
        other => {
            set_work_mem(0);
            panic!("streaming INSERT ... SELECT should stay under the budget, got {other:?}");
        },
    }

    // Multi-batch streaming (source > one 1024-row batch), the QA ETL shape: grow the copy to
    // 1200 rows, then stream all of them into a third table — two batches — still under the same
    // tight budget.
    for _ in 0..2 {
        let logical = analyze(
            parse("INSERT INTO big_copy SELECT id, payload FROM big").unwrap(),
            &Cat(engine),
        )
        .unwrap();
        match session.execute(plan(logical)) {
            Ok(nusadb_sql::ExecutionResult::Inserted(400)) => {},
            other => {
                set_work_mem(0);
                panic!("streaming INSERT ... SELECT should succeed, got {other:?}");
            },
        }
    }
    run(engine, &mut session, "CREATE TABLE ser (n INT)").unwrap();
    let logical = analyze(
        parse("INSERT INTO ser SELECT id FROM big_copy").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Inserted(n)) => {
            assert_eq!(n, 1200, "every source row must be inserted across batches");
        },
        other => {
            set_work_mem(0);
            panic!("multi-batch streaming INSERT ... SELECT should succeed, got {other:?}");
        },
    }

    // A literal-integer generate_series source streams lazily (the SRF-source residual): the
    // whole ETL shape `INSERT ... SELECT g FROM generate_series(...)` stays O(1) under the same
    // budget that refuses materializing 3000 rows.
    run(engine, &mut session, "CREATE TABLE ser2 (n INT)").unwrap();
    let logical = analyze(
        parse("INSERT INTO ser2 SELECT g FROM generate_series(1, 3000) g").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Inserted(n)) => {
            assert_eq!(n, 3000, "every generated row must be inserted");
        },
        other => {
            set_work_mem(0);
            panic!("a generate_series source should stream under the budget, got {other:?}");
        },
    }

    // A PRIMARY KEY target streams too (deferred uniqueness): 3000 unique keys insert under the
    // same tight budget; a duplicate inside the stream and a duplicate against a committed row
    // both fail with the honest duplicate-key error (and roll the statement back).
    run(
        engine,
        &mut session,
        "CREATE TABLE pk_t (n INT PRIMARY KEY)",
    )
    .unwrap();
    let logical = analyze(
        parse("INSERT INTO pk_t SELECT n FROM ser2").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Inserted(n)) => {
            assert_eq!(n, 3000, "unique keys stream into a PK table");
        },
        other => {
            set_work_mem(0);
            panic!("PK-target streaming INSERT ... SELECT should succeed, got {other:?}");
        },
    }
    // Intra-stream duplicate: big_copy holds each id three times.
    let logical = analyze(
        parse("INSERT INTO pk_t SELECT id + 10000 FROM big_copy").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Err(e) => assert!(
            e.to_string().contains("duplicate key"),
            "an intra-stream duplicate must fail loudly: {e}"
        ),
        other => {
            set_work_mem(0);
            panic!("expected a duplicate-key error, got {other:?}");
        },
    }
    // Duplicate against a committed row (the end-of-stream committed re-check): key 3000 exists.
    let logical = analyze(
        parse("INSERT INTO pk_t SELECT n + 2999 FROM ser2 WHERE n <= 2").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Err(e) => assert!(
            e.to_string().contains("duplicate key"),
            "a committed duplicate must fail at the deferred check: {e}"
        ),
        other => {
            set_work_mem(0);
            panic!("expected a duplicate-key error, got {other:?}");
        },
    }
    // Both failures rolled back; a fully-new key range still streams in.
    let logical = analyze(
        parse("INSERT INTO pk_t SELECT n + 3000 FROM ser2").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Inserted(n)) => assert_eq!(n, 3000),
        other => {
            set_work_mem(0);
            panic!("fresh keys should stream after the failed statements, got {other:?}");
        },
    }

    // Full-table aggregates fold their streamed input instead of materializing it:
    // the same budget that refuses `SELECT * FROM big` computes
    // count/sum/max — and a GROUP BY — over it, because only accumulators are held.
    for (sql, expected) in [
        (
            "SELECT count(*), sum(id), max(id) FROM big",
            vec![vec![
                nusadb_sql::ast::Value::Int(400),
                nusadb_sql::ast::Value::Int((0..400).sum()),
                nusadb_sql::ast::Value::Int(399),
            ]],
        ),
        (
            "SELECT payload, count(*) FROM big GROUP BY payload",
            vec![vec![
                nusadb_sql::ast::Value::Text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".to_owned()),
                nusadb_sql::ast::Value::Int(400),
            ]],
        ),
        // GROUPING SETS stream too (the residual): the per-key row plus the grand total.
        (
            "SELECT payload, count(*) FROM big GROUP BY ROLLUP (payload)",
            vec![
                vec![
                    nusadb_sql::ast::Value::Text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".to_owned()),
                    nusadb_sql::ast::Value::Int(400),
                ],
                vec![
                    nusadb_sql::ast::Value::Null,
                    nusadb_sql::ast::Value::Int(400),
                ],
            ],
        ),
    ] {
        let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
        match session.execute(plan(logical)) {
            Ok(nusadb_sql::ExecutionResult::Rows { rows, .. }) => {
                assert_eq!(
                    rows, expected,
                    "aggregate must stream under the budget ({sql})"
                );
            },
            other => {
                set_work_mem(0);
                panic!("streaming aggregate should stay under the budget for {sql}, got {other:?}");
            },
        }
    }

    // A hash join's OUTPUT streams: with a budget that admits the tiny
    // build side but NOT the joined output, a LIMIT and an aggregate over the join both succeed —
    // the join holds its build side plus one probe row, never the whole matched output (QA: even
    // `LIMIT 5` over a joined 1M table OOMed because the output materialized first).
    set_work_mem(0);
    run(engine, &mut session, "CREATE TABLE keys (k INT)").unwrap();
    for k in 0..10 {
        run(
            engine,
            &mut session,
            &format!("INSERT INTO keys VALUES ({k})"),
        )
        .unwrap();
    }
    run(
        engine,
        &mut session,
        "CREATE TABLE wide (k INT, payload TEXT)",
    )
    .unwrap();
    for i in 0..400 {
        run(
            engine,
            &mut session,
            &format!(
                "INSERT INTO wide VALUES ({}, 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')",
                i % 10
            ),
        )
        .unwrap();
    }
    // 8 KiB: the 10-row build side fits; the 400-row joined output (~50+ KiB) would not.
    set_work_mem(8192);
    let logical = analyze(
        parse("SELECT wide.k FROM wide JOIN keys ON wide.k = keys.k LIMIT 3").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Rows { rows, .. }) => {
            assert_eq!(
                rows.len(),
                3,
                "LIMIT over the streamed join must stop early under the budget"
            );
        },
        other => {
            set_work_mem(0);
            panic!("LIMIT over a streamed join should stay under the budget, got {other:?}");
        },
    }
    let logical = analyze(
        parse("SELECT count(*) FROM wide JOIN keys ON wide.k = keys.k").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    match session.execute(plan(logical)) {
        Ok(nusadb_sql::ExecutionResult::Rows { rows, .. }) => {
            assert_eq!(
                rows,
                vec![vec![nusadb_sql::ast::Value::Int(400)]],
                "aggregate over the streamed join must fold without materializing the output"
            );
        },
        other => {
            set_work_mem(0);
            panic!("count(*) over a streamed join should stay under the budget, got {other:?}");
        },
    }

    // Lifting the budget admits the large query again — and the streamed copy round-trips.
    set_work_mem(0);
    run(engine, &mut session, "SELECT * FROM big").expect("raising the budget reopens the query");
    let id_sum: i64 = (0..400).sum();
    for (sql, count, sum) in [
        ("SELECT count(*), sum(id) FROM big_copy", 1200, 3 * id_sum),
        ("SELECT count(*), sum(n) FROM ser", 1200, 3 * id_sum),
        // The lazily-streamed series inserted exactly 1..=3000.
        ("SELECT count(*), sum(n) FROM ser2", 3000, 3000 * 3001 / 2),
    ] {
        let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
        match session.execute(plan(logical)) {
            Ok(nusadb_sql::ExecutionResult::Rows { rows, .. }) => {
                assert_eq!(
                    rows,
                    vec![vec![
                        nusadb_sql::ast::Value::Int(count),
                        nusadb_sql::ast::Value::Int(sum),
                    ]],
                    "the streamed copy must hold exactly the source rows ({sql})"
                );
            },
            other => panic!("expected one aggregate row for {sql}, got {other:?}"),
        }
    }

    // The limit-aware top-N must ALSO respect work_mem: a wide-row
    // `ORDER BY ... LIMIT n` retains `n` rows, and a tight budget makes that retention fail loudly
    // rather than silently holding `n` large rows in RAM. `big` has 400
    // wide rows; LIMIT 300 retains ~300 of them, well over 4 KiB.
    set_work_mem(4 * 1024);
    match run(
        engine,
        &mut session,
        "SELECT * FROM big ORDER BY id LIMIT 300",
    ) {
        Err(Error::Core(nusadb_core::Error::OutOfMemory(msg))) => {
            assert!(
                msg.contains("work_mem"),
                "message should name work_mem: {msg}"
            );
        },
        other => {
            set_work_mem(0);
            panic!("top-N ORDER BY ... LIMIT must respect work_mem, got {other:?}");
        },
    }
    // A small top-N fits the same tight budget.
    set_work_mem(64 * 1024);
    run(
        engine,
        &mut session,
        "SELECT * FROM big ORDER BY id LIMIT 3",
    )
    .expect("a small top-N fits work_mem");

    // A cross join materializes its full N×M output. A tight budget must fail it LOUDLY with an
    // out-of-memory error, checked incrementally as the output grows, rather than letting the join
    // balloon (400×400 = 160k joined rows) and OOM-kill the whole server.
    set_work_mem(128 * 1024);
    match run(
        engine,
        &mut session,
        "SELECT count(*) FROM big AS a CROSS JOIN big AS b",
    ) {
        Err(Error::Core(nusadb_core::Error::OutOfMemory(msg))) => {
            assert!(
                msg.contains("work_mem") && msg.contains("join"),
                "a cross-join OOM must name work_mem and the join: {msg}"
            );
        },
        other => {
            set_work_mem(0);
            panic!("a cross join over work_mem must fail loudly, not OOM-kill, got {other:?}");
        },
    }
    set_work_mem(0);
}
