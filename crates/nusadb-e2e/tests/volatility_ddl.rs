//! Function volatility in DDL: an index key or predicate and a generated column must call only
//! immutable built-ins, and a `CHECK` must not call a volatile one (`42P17`). The rules gate new
//! DDL only: a data directory whose catalog already holds such objects, as an earlier release
//! wrote them, still opens and keeps maintaining and enforcing them.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, LogicalPlan, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn analyze_sql(engine: &BtreeEngine, sql: &str) -> Result<LogicalPlan, Error> {
    analyze(parse(sql).expect("parse"), &EngineCatalog(engine))
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let logical = analyze_sql(engine, sql).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, Error> {
    execute(plan(analyze_sql(engine, sql)?), engine)
}

fn rows(engine: &BtreeEngine, sql: &str) -> Vec<Vec<Value>> {
    match run(engine, sql) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

/// Assert `sql` is refused at analysis with `42P17`, naming `function`.
fn refused(engine: &BtreeEngine, sql: &str, function: &str) {
    let err = analyze_sql(engine, sql).expect_err(sql);
    assert_eq!(err.sqlstate(), "42P17", "{sql}: {err}");
    assert!(err.to_string().contains(function), "{sql}: {err}");
}

#[test]
fn index_keys_and_predicates_must_be_immutable() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (x INT, t TIMESTAMPTZ, u UUID)");
    for (sql, function) in [
        ("CREATE INDEX i1 ON t ((uuidv7()))", "uuidv7"),
        (
            "CREATE INDEX i2 ON t ((clock_timestamp()))",
            "clock_timestamp",
        ),
        ("CREATE INDEX i3 ON t ((random()))", "random"),
        (
            "CREATE INDEX i4 ON t ((gen_random_uuid()))",
            "gen_random_uuid",
        ),
        (
            "CREATE INDEX i5 ON t ((x + extract(day FROM now())))",
            "now",
        ),
        ("CREATE INDEX i6 ON t ((current_date))", "current_date"),
        (
            "CREATE INDEX i7 ON t ((statement_timestamp()))",
            "statement_timestamp",
        ),
        ("CREATE INDEX i8 ON t ((current_user))", "current_user"),
        ("CREATE INDEX i9 ON t (x) WHERE t < now()", "now"),
        ("CREATE INDEX i10 ON t (x) WHERE random() < 0.5", "random"),
        ("CREATE UNIQUE INDEX i11 ON t ((age(t)))", "age"),
    ] {
        refused(&engine, sql, function);
    }
    // Immutable keys and predicates are accepted, two-argument age() among them.
    for sql in [
        "CREATE INDEX ok1 ON t ((abs(x)))",
        "CREATE INDEX ok2 ON t ((uuid_extract_timestamp(u)))",
        "CREATE INDEX ok3 ON t (x) WHERE x > 0",
        "CREATE INDEX ok4 ON t ((age(TIMESTAMP '2030-01-01', TIMESTAMP '2020-01-01')))",
    ] {
        run(&engine, sql);
    }
}

#[test]
fn generated_columns_must_be_immutable() {
    let engine = BtreeEngine::new();
    for (sql, function) in [
        (
            "CREATE TABLE g1 (a INT, b FLOAT GENERATED ALWAYS AS (random()) STORED)",
            "random",
        ),
        (
            "CREATE TABLE g2 (a INT, b TIMESTAMPTZ GENERATED ALWAYS AS (now()) STORED)",
            "now",
        ),
        (
            "CREATE TABLE g3 (a INT, b UUID GENERATED ALWAYS AS (uuidv7()) STORED)",
            "uuidv7",
        ),
    ] {
        refused(&engine, sql, function);
    }
    run(
        &engine,
        "CREATE TABLE g4 (a INT, b INT GENERATED ALWAYS AS (a * 2) STORED)",
    );
    refused(
        &engine,
        "ALTER TABLE g4 ADD COLUMN c FLOAT GENERATED ALWAYS AS (random()) STORED",
        "random",
    );
}

#[test]
fn checks_refuse_volatile_but_accept_stable_functions() {
    let engine = BtreeEngine::new();
    for (sql, function) in [
        (
            "CREATE TABLE c1 (t TIMESTAMPTZ CHECK (t <= clock_timestamp()))",
            "clock_timestamp",
        ),
        ("CREATE TABLE c2 (u UUID CHECK (u < uuidv7()))", "uuidv7"),
        ("CREATE TABLE c3 (f FLOAT CHECK (f < random()))", "random"),
        (
            "CREATE TABLE c4 (u UUID, CHECK (u <> gen_random_uuid()))",
            "gen_random_uuid",
        ),
    ] {
        refused(&engine, sql, function);
    }
    run(&engine, "CREATE SEQUENCE s");
    run(&engine, "CREATE TABLE c5 (a INT)");
    refused(
        &engine,
        "ALTER TABLE c5 ADD CONSTRAINT c5_seq CHECK (a < nextval('s'))",
        "nextval",
    );
    refused(
        &engine,
        "ALTER TABLE c5 ADD COLUMN f FLOAT CHECK (f < random())",
        "random",
    );

    // A stable function is accepted: the row is checked against the writing transaction's time.
    run(
        &engine,
        "CREATE TABLE c6 (t TIMESTAMPTZ CHECK (t <= now()), who TEXT CHECK (who = current_user))",
    );
    run(
        &engine,
        "INSERT INTO c6 VALUES (now() - INTERVAL '1 day', current_user)",
    );
    let err = run_try(
        &engine,
        "INSERT INTO c6 VALUES (now() + INTERVAL '1 day', current_user)",
    )
    .expect_err("a future time violates the check");
    assert_eq!(err.sqlstate(), "23514", "{err}");
}

/// Create, the way an earlier release did, a unique index keyed on an expression calling
/// `random()`, a unique partial index whose predicate calls `random()`, a generated column computing
/// `random()` and a CHECK calling `clock_timestamp()`: analyze an accepted statement and swap the
/// volatile call into the plan the executor stores.
fn write_legacy_objects(engine: &BtreeEngine) {
    let swap = |sql: &str, from: &str, to: &str| -> LogicalPlan {
        let logical = analyze_sql(engine, sql).expect("analyze");
        let text = format!("{logical:?}");
        assert!(text.contains(from), "{from} not in {text}");
        logical_swap(logical, from, to)
    };
    let table = swap(
        "CREATE TABLE legacy (id INT PRIMARY KEY, x FLOAT, y INT, \
         g FLOAT GENERATED ALWAYS AS (abs(x)) STORED, \
         t TIMESTAMPTZ CHECK (t <= now()))",
        "now()",
        "clock_timestamp()",
    );
    let table = logical_swap(table, "abs(x)", "random()");
    execute(plan(table), engine).expect("create the legacy table");
    let index = swap(
        "CREATE UNIQUE INDEX legacy_key ON legacy ((abs(x)))",
        "abs(x)",
        "abs(x) + random() * 0",
    );
    execute(plan(index), engine).expect("create the legacy index");
    let partial = swap(
        "CREATE UNIQUE INDEX legacy_partial ON legacy (y) WHERE y < 1000",
        "y < 1000",
        "random() < 2.5",
    );
    execute(plan(partial), engine).expect("create the legacy partial index");
}

/// Replace `from` with `to` in the SQL text a CREATE TABLE / CREATE INDEX plan carries for its
/// checks, generated columns, index keys and index predicate.
fn logical_swap(logical: LogicalPlan, from: &str, to: &str) -> LogicalPlan {
    let swap = |s: &mut String| *s = s.replace(from, to);
    match logical {
        LogicalPlan::CreateTable(mut p) => {
            for check in &mut p.check_constraints {
                swap(&mut check.predicate_sql);
            }
            for (_, sql) in &mut p.defaults {
                swap(sql);
            }
            LogicalPlan::CreateTable(p)
        },
        LogicalPlan::CreateIndex(mut p) => {
            p.def.key_exprs.iter_mut().for_each(swap);
            if let Some(predicate) = &mut p.def.predicate {
                swap(predicate);
            }
            LogicalPlan::CreateIndex(p)
        },
        other => panic!("unexpected plan {other:?}"),
    }
}

fn exercise_legacy_objects(engine: &BtreeEngine, first_id: i64) {
    let (a, b, c) = (first_id, first_id + 1, first_id + 2);
    // Inserts maintain both indexes and fill the generated column; the CHECK is enforced.
    run(
        engine,
        &format!(
            "INSERT INTO legacy (id, x, y, t) VALUES ({a}, {a}, {a}, now() - INTERVAL '1 hour'), \
             ({b}, {b}, {b}, now())"
        ),
    );
    let err = run_try(
        engine,
        &format!(
            "INSERT INTO legacy (id, x, y, t) VALUES ({c}, {c}, {c}, now() + INTERVAL '1 day')"
        ),
    )
    .expect_err("the legacy CHECK still holds");
    assert_eq!(err.sqlstate(), "23514", "{err}");
    // Both unique indexes are still maintained: a repeated key is refused by each.
    for (x, y) in [(a, c), (c, a)] {
        let err = run_try(
            engine,
            &format!("INSERT INTO legacy (id, x, y, t) VALUES ({c}, {x}, {y}, now())"),
        )
        .expect_err("a duplicate key in a legacy unique index");
        assert_eq!(err.sqlstate(), "23505", "x {x} y {y}: {err}");
    }
    run(
        engine,
        &format!("UPDATE legacy SET x = {a} + 0.5 WHERE id = {a}"),
    );
    run(engine, &format!("DELETE FROM legacy WHERE id = {b}"));
    let generated = rows(engine, &format!("SELECT g FROM legacy WHERE id = {a}"));
    let Value::Float(g) = generated[0][0] else {
        panic!("generated value {:?}", generated[0][0]);
    };
    assert!(
        (0.0..1.0).contains(&g),
        "random() fills the generated column: {g}"
    );
}

#[test]
fn a_catalog_holding_volatile_objects_still_opens_and_works() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine = BtreeEngine::open(&path).expect("open");
        write_legacy_objects(&engine);
        exercise_legacy_objects(&engine, 1);
    }
    // Reopen: the catalog loads, and every object is maintained and enforced as before.
    {
        let engine = BtreeEngine::open(&path).expect("reopen");
        exercise_legacy_objects(&engine, 10);
        let names: Vec<String> = engine
            .list_indexes(
                engine
                    .lookup_table("legacy")
                    .expect("lookup")
                    .expect("legacy table")
                    .id,
            )
            .expect("indexes")
            .into_iter()
            .map(|def| def.name)
            .collect();
        assert!(names.iter().any(|n| n == "legacy_key"), "{names:?}");
        assert!(names.iter().any(|n| n == "legacy_partial"), "{names:?}");
        let count = rows(&engine, "SELECT count(*) FROM legacy");
        assert_eq!(count[0][0], Value::Int(2));
        // New DDL of the same kind is refused on the upgraded catalog.
        refused(
            &engine,
            "CREATE INDEX again ON legacy ((uuidv7()))",
            "uuidv7",
        );
    }
    // A third open, after more writes, still works.
    let engine = BtreeEngine::open(&path).expect("third open");
    exercise_legacy_objects(&engine, 20);
}
