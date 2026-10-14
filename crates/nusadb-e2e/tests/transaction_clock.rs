//! The transaction clock end to end through an embedded `Session`: `now()` and the other
//! transaction-time functions report the transaction's start in every statement of it, while
//! `statement_timestamp()` moves on with each client statement and stays put inside the triggers,
//! procedure bodies and `DO` blocks that statement runs. Neither a prepared statement, a cached plan
//! nor the result cache may serve a `now()` from an earlier transaction.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts by panicking on failure"
)]

use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, PlanCache, Session, analyze, execute, parse, plan, plan_cached,
};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn exec(session: &mut Session<'_>, engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    session.execute(plan(logical)).expect("execute")
}

fn query(session: &mut Session<'_>, engine: &BtreeEngine, sql: &str) -> Vec<Vec<Value>> {
    match exec(session, engine, sql) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

fn micros(value: &Value) -> i64 {
    match value {
        Value::TimestampTz(us) | Value::Timestamp(us) => *us,
        other => panic!("expected a timestamp, got {other:?}"),
    }
}

/// `(now(), statement_timestamp())` of one statement.
fn clocks(session: &mut Session<'_>, engine: &BtreeEngine) -> (i64, i64) {
    let row = &query(session, engine, "SELECT now(), statement_timestamp()")[0];
    (micros(&row[0]), micros(&row[1]))
}

/// Let the wall clock move on, so two statements cannot share a microsecond by chance.
fn tick() {
    std::thread::sleep(Duration::from_millis(3));
}

#[test]
fn now_is_fixed_for_the_whole_transaction() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "BEGIN");
    tick();
    let (now1, stmt1) = clocks(&mut s, &engine);
    tick();
    let (now2, stmt2) = clocks(&mut s, &engine);
    let row = &query(
        &mut s,
        &engine,
        "SELECT transaction_timestamp(), current_timestamp, localtimestamp, current_date, \
         current_time, statement_timestamp(), localtime",
    )[0];
    exec(&mut s, &engine, "COMMIT");

    assert_eq!(
        now1, now2,
        "now() is the transaction's start in every statement"
    );
    assert!(
        now1 < stmt1,
        "the transaction began before its first statement"
    );
    assert!(
        stmt1 < stmt2,
        "statement_timestamp() moves on with each statement"
    );
    assert_eq!(micros(&row[0]), now1);
    assert_eq!(micros(&row[1]), now1);
    assert_eq!(
        micros(&row[2]),
        now1,
        "localtimestamp in the UTC session zone"
    );
    assert_eq!(
        row[3],
        Value::Date(i32::try_from(now1.div_euclid(86_400_000_000)).expect("date"))
    );
    assert!(micros(&row[5]) > stmt2);
    assert_eq!(row[6], Value::Time(now1.rem_euclid(86_400_000_000)));
    let Value::TimeTz(_) = row[4] else {
        panic!("current_time is a timetz, got {:?}", row[4]);
    };

    // The next transaction has its own start.
    tick();
    exec(&mut s, &engine, "BEGIN");
    let (now3, _) = clocks(&mut s, &engine);
    exec(&mut s, &engine, "ROLLBACK");
    assert!(now3 > now2);
}

#[test]
fn autocommit_statements_each_take_their_own_now() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    let (now1, stmt1) = clocks(&mut s, &engine);
    tick();
    let (now2, stmt2) = clocks(&mut s, &engine);
    assert_eq!(
        now1, stmt1,
        "in autocommit the transaction is the statement"
    );
    assert_eq!(now2, stmt2);
    assert!(now2 > now1);
}

#[test]
fn defaults_written_in_one_transaction_share_now() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(
        &mut s,
        &engine,
        "CREATE TABLE ev (id INT, at TIMESTAMPTZ DEFAULT now(), st TIMESTAMPTZ DEFAULT statement_timestamp())",
    );
    exec(&mut s, &engine, "BEGIN");
    exec(&mut s, &engine, "INSERT INTO ev (id) VALUES (1)");
    tick();
    exec(&mut s, &engine, "INSERT INTO ev (id) VALUES (2)");
    let (now, _) = clocks(&mut s, &engine);
    exec(&mut s, &engine, "COMMIT");
    let rows = query(&mut s, &engine, "SELECT at, st FROM ev ORDER BY id");
    assert_eq!(micros(&rows[0][0]), now);
    assert_eq!(micros(&rows[1][0]), now);
    assert!(micros(&rows[0][1]) < micros(&rows[1][1]));
}

#[test]
fn nested_statements_keep_the_client_statement_clock() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(
        &mut s,
        &engine,
        "CREATE TABLE seen (src TEXT, n TIMESTAMPTZ, st TIMESTAMPTZ)",
    );
    exec(&mut s, &engine, "CREATE TABLE t (id INT)");
    exec(
        &mut s,
        &engine,
        "CREATE TRIGGER t_ins AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO seen VALUES ('trigger', now(), statement_timestamp())",
    );
    exec(
        &mut s,
        &engine,
        "CREATE PROCEDURE note() AS $$ \
           INSERT INTO seen VALUES ('procedure', now(), statement_timestamp()); \
           INSERT INTO seen VALUES ('procedure', now(), statement_timestamp()) \
         $$",
    );

    exec(&mut s, &engine, "BEGIN");
    tick();
    let (now, _) = clocks(&mut s, &engine);
    tick();
    exec(&mut s, &engine, "INSERT INTO t VALUES (1), (2)");
    tick();
    exec(&mut s, &engine, "CALL note()");
    tick();
    exec(
        &mut s,
        &engine,
        "DO $$ BEGIN INSERT INTO seen VALUES ('do', now(), statement_timestamp()); \
         INSERT INTO seen VALUES ('do', now(), statement_timestamp()); END $$",
    );
    exec(&mut s, &engine, "COMMIT");

    let rows = query(&mut s, &engine, "SELECT src, n, st FROM seen");
    assert_eq!(rows.len(), 6);
    let mut statements = Vec::new();
    for source in ["trigger", "procedure", "do"] {
        let of: Vec<&Vec<Value>> = rows
            .iter()
            .filter(|r| r[0] == Value::Text(source.to_owned()))
            .collect();
        assert_eq!(of.len(), 2, "{source}");
        for r in &of {
            assert_eq!(
                micros(&r[1]),
                now,
                "{source}: now() is the transaction's start"
            );
        }
        assert_eq!(
            micros(&of[0][2]),
            micros(&of[1][2]),
            "{source}: one client statement, one statement_timestamp()"
        );
        statements.push(micros(&of[0][2]));
    }
    assert!(now < statements[0]);
    assert!(statements[0] < statements[1] && statements[1] < statements[2]);
}

#[test]
fn prepared_cached_and_result_cached_now_follow_the_transaction() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");
    exec(&mut s, &engine, "PREPARE p AS SELECT now() FROM t");

    exec(&mut s, &engine, "BEGIN");
    let (now_a, _) = clocks(&mut s, &engine);
    tick();
    let in_a = micros(&query(&mut s, &engine, "EXECUTE p")[0][0]);
    exec(&mut s, &engine, "COMMIT");
    tick();
    exec(&mut s, &engine, "BEGIN");
    let (now_b, _) = clocks(&mut s, &engine);
    let in_b = micros(&query(&mut s, &engine, "EXECUTE p")[0][0]);
    exec(&mut s, &engine, "COMMIT");
    assert_eq!(in_a, now_a);
    assert_eq!(in_b, now_b);
    assert!(now_b > now_a);

    // Autocommit through the result cache: a fresh value every time.
    let first = micros(&query(&mut s, &engine, "SELECT now() FROM t")[0][0]);
    tick();
    let second = micros(&query(&mut s, &engine, "SELECT now() FROM t")[0][0]);
    assert!(second > first);

    // A cached plan is re-evaluated per run.
    let mut cache = PlanCache::new();
    let sql = "SELECT now() FROM t";
    let mut run = || {
        let physical = plan_cached(
            &mut cache,
            sql,
            parse(sql).expect("parse"),
            &EngineCatalog(&engine),
            &engine,
        )
        .expect("plan_cached");
        match execute(physical, &engine).expect("execute") {
            ExecutionResult::Rows { rows, .. } => micros(&rows[0][0]),
            other => panic!("expected rows, got {other:?}"),
        }
    };
    let first = run();
    tick();
    let second = run();
    assert!(second > first);
    assert_eq!(cache.hits(), 1);
}

#[test]
fn set_transaction_keeps_the_start_of_the_block() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "BEGIN");
    tick();
    // Re-begins the engine transaction at the new level; the block's start stays the BEGIN.
    exec(
        &mut s,
        &engine,
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
    );
    tick();
    let (now, stmt) = clocks(&mut s, &engine);
    let (now2, _) = clocks(&mut s, &engine);
    exec(&mut s, &engine, "COMMIT");
    assert_eq!(now, now2);
    assert!(
        stmt - now >= 6_000,
        "now() is the BEGIN, before both pauses"
    );
}
