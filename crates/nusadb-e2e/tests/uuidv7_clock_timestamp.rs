//! `uuidv7()` and `clock_timestamp()` end to end: version 7 UUIDs grow strictly within a statement,
//! across the statements of a transaction and across rows inserted through a key default, and
//! `clock_timestamp()` reads the wall clock on every evaluation. Neither may be folded into a
//! constant, served from the result cache, or frozen into a cached or prepared plan.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts by panicking on failure"
)]

use std::time::{SystemTime, UNIX_EPOCH};

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

/// Run one statement through a persistent session (transactions and the result cache live there).
fn exec(session: &mut Session<'_>, engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    session.execute(plan(logical)).expect("execute")
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

fn query(session: &mut Session<'_>, engine: &BtreeEngine, sql: &str) -> Vec<Vec<Value>> {
    rows(exec(session, engine, sql))
}

fn uuid(value: &Value) -> [u8; 16] {
    match value {
        Value::Uuid(bytes) => *bytes,
        other => panic!("expected a UUID, got {other:?}"),
    }
}

fn int(value: &Value) -> i64 {
    match value {
        Value::Int(n) => *n,
        other => panic!("expected an integer, got {other:?}"),
    }
}

fn micros(value: &Value) -> i64 {
    match value {
        Value::TimestampTz(us) => *us,
        other => panic!("expected a timestamptz, got {other:?}"),
    }
}

fn unix_millis() -> i64 {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch");
    i64::try_from(elapsed.as_millis()).expect("millis fit in i64")
}

fn embedded_millis(bytes: &[u8; 16]) -> i64 {
    let mut ms = [0u8; 8];
    ms[2..].copy_from_slice(&bytes[..6]);
    i64::from_be_bytes(ms)
}

fn assert_strictly_increasing(uuids: &[[u8; 16]]) {
    for pair in uuids.windows(2) {
        assert!(
            pair[0] < pair[1],
            "{:02x?} is not below {:02x?}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn uuidv7_is_well_formed_and_carries_the_current_time() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    let before = unix_millis();
    let out = query(&mut s, &engine, "SELECT uuidv7()");
    let after = unix_millis();
    let bytes = uuid(&out[0][0]);
    assert_eq!(bytes[6] >> 4, 7, "version nibble");
    assert_eq!(bytes[8] & 0xc0, 0x80, "variant bits");
    let ms = embedded_millis(&bytes);
    // Other tests in this process may push the counter a little ahead of the clock.
    assert!(
        ms >= before && ms <= after + 1000,
        "{before} <= {ms} <= {after}"
    );
}

#[test]
fn uuidv7_strictly_increases_across_ten_thousand_rows_of_one_statement() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    let mut out = query(
        &mut s,
        &engine,
        "SELECT g, uuidv7() FROM generate_series(1, 10000) AS g",
    );
    assert_eq!(out.len(), 10_000);
    out.sort_by_key(|row| int(&row[0]));
    let uuids: Vec<[u8; 16]> = out.iter().map(|row| uuid(&row[1])).collect();
    assert!(uuids.iter().all(|u| u[6] >> 4 == 7 && u[8] & 0xc0 == 0x80));
    assert_strictly_increasing(&uuids);
}

#[test]
fn uuidv7_increases_across_statements_of_one_transaction() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "BEGIN");
    let mut uuids = Vec::new();
    for _ in 0..200 {
        let out = query(&mut s, &engine, "SELECT uuidv7(), uuidv7()");
        uuids.push(uuid(&out[0][0]));
        uuids.push(uuid(&out[0][1]));
    }
    exec(&mut s, &engine, "COMMIT");
    // And on in auto-commit after the transaction.
    for _ in 0..50 {
        uuids.push(uuid(&query(&mut s, &engine, "SELECT uuidv7()")[0][0]));
    }
    assert_strictly_increasing(&uuids);
}

#[test]
fn uuidv7_primary_key_default_sorts_rows_by_insertion_time() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(
        &mut s,
        &engine,
        "CREATE TABLE events (id UUID PRIMARY KEY DEFAULT uuidv7(), n INT NOT NULL)",
    );
    exec(
        &mut s,
        &engine,
        "INSERT INTO events (n) SELECT g FROM generate_series(1, 1000) AS g",
    );
    exec(
        &mut s,
        &engine,
        "INSERT INTO events (n) VALUES (1001), (1002)",
    );
    exec(&mut s, &engine, "BEGIN");
    for n in 1003..=1010 {
        exec(
            &mut s,
            &engine,
            &format!("INSERT INTO events (n) VALUES ({n})"),
        );
    }
    exec(&mut s, &engine, "COMMIT");

    let ordered: Vec<i64> = query(&mut s, &engine, "SELECT n FROM events ORDER BY id")
        .iter()
        .map(|row| int(&row[0]))
        .collect();
    assert_eq!(ordered, (1..=1010).collect::<Vec<_>>());
    let distinct = query(&mut s, &engine, "SELECT count(DISTINCT id) FROM events");
    assert_eq!(int(&distinct[0][0]), 1010);
}

#[test]
fn clock_timestamp_advances_within_one_statement() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);

    // Two calls in one row, with work between them: the second reads a later clock.
    let out = query(
        &mut s,
        &engine,
        "SELECT clock_timestamp(), length(repeat('x', 20000000)), clock_timestamp(), now()",
    );
    let (first, second, now) = (micros(&out[0][0]), micros(&out[0][2]), micros(&out[0][3]));
    assert!(first < second, "{first} should be before {second}");
    assert!(first >= now, "the clock is read after the statement starts");

    // Per row: never going back, and moving forward over the series, while now() stays put.
    let mut out = query(
        &mut s,
        &engine,
        "SELECT g, clock_timestamp(), now() FROM generate_series(1, 200000) AS g",
    );
    out.sort_by_key(|row| int(&row[0]));
    let clocks: Vec<i64> = out.iter().map(|row| micros(&row[1])).collect();
    assert!(clocks.windows(2).all(|w| w[0] <= w[1]));
    assert!(clocks[0] < clocks[clocks.len() - 1]);
    assert!(
        out.iter().all(|row| row[2] == out[0][2]),
        "now() is one instant"
    );

    let counts = query(
        &mut s,
        &engine,
        "SELECT count(DISTINCT now()), count(DISTINCT clock_timestamp()) \
         FROM generate_series(1, 100000)",
    );
    assert_eq!(int(&counts[0][0]), 1);
    assert!(int(&counts[0][1]) > 1);
}

#[test]
fn clock_timestamp_and_uuidv7_are_not_served_from_the_result_cache() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");
    exec(&mut s, &engine, "SELECT n FROM t");
    assert_eq!(s.result_cache_len(), 1, "a plain SELECT is cached");

    let sql = "SELECT clock_timestamp(), uuidv7() FROM t";
    let first = query(&mut s, &engine, sql);
    let second = query(&mut s, &engine, sql);
    assert_eq!(s.result_cache_len(), 1, "a volatile SELECT is not cached");
    assert!(micros(&first[0][0]) <= micros(&second[0][0]));
    assert!(uuid(&first[0][1]) < uuid(&second[0][1]));

    for volatile in ["SELECT clock_timestamp() FROM t", "SELECT uuidv7() FROM t"] {
        exec(&mut s, &engine, volatile);
        assert_eq!(s.result_cache_len(), 1, "{volatile} must not be cached");
    }
}

#[test]
fn clock_timestamp_and_uuidv7_are_not_frozen_into_a_cached_plan() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(&mut s, &engine, "CREATE TABLE t (n INT)");
    exec(&mut s, &engine, "INSERT INTO t VALUES (1)");
    let mut cache = PlanCache::new();
    let sql = "SELECT clock_timestamp(), length(repeat('x', 5000000)), uuidv7() FROM t";
    let run = |cache: &mut PlanCache| {
        let physical = plan_cached(
            cache,
            sql,
            parse(sql).expect("parse"),
            &EngineCatalog(&engine),
            &engine,
        )
        .expect("plan_cached");
        rows(execute(physical, &engine).expect("execute"))
    };
    let first = run(&mut cache);
    let second = run(&mut cache);
    assert_eq!(cache.hits(), 1, "the second run reuses the cached plan");
    assert!(micros(&first[0][0]) < micros(&second[0][0]));
    assert!(uuid(&first[0][2]) < uuid(&second[0][2]));
}

#[test]
fn clock_timestamp_and_uuidv7_are_evaluated_on_each_prepared_execution() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    exec(
        &mut s,
        &engine,
        "PREPARE p AS SELECT clock_timestamp(), length(repeat('x', 5000000)), uuidv7()",
    );
    let first = query(&mut s, &engine, "EXECUTE p");
    let second = query(&mut s, &engine, "EXECUTE p");
    assert!(micros(&first[0][0]) < micros(&second[0][0]));
    assert!(uuid(&first[0][2]) < uuid(&second[0][2]));
}

#[test]
fn clock_timestamp_and_uuidv7_are_refused_as_partition_keys() {
    let engine = BtreeEngine::new();
    for sql in [
        "CREATE TABLE p1 (k INT) PARTITION BY RANGE ((clock_timestamp()))",
        "CREATE TABLE p2 (k INT) PARTITION BY RANGE ((uuidv7()))",
    ] {
        let err = analyze(parse(sql).expect("parse"), &EngineCatalog(&engine))
            .expect_err("a volatile partition key is refused");
        assert!(err.to_string().contains("IMMUTABLE"), "{sql}: {err}");
    }
    // Reading fields out of a stored UUID is deterministic, so it is a valid key.
    let sql = "CREATE TABLE p3 (id UUID) PARTITION BY RANGE ((uuid_extract_timestamp(id)))";
    analyze(parse(sql).expect("parse"), &EngineCatalog(&engine)).expect("a stable key is accepted");
}
