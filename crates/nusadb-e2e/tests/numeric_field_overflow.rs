//! A number too large for `NUMERIC(p, s)` is `22003` (`numeric_value_out_of_range`) on every path
//! that stores into or casts to the type, and text that is not a number stays `22P02`. The `.slt`
//! corpus (`slt/p10_types/numeric_field_overflow.slt`) pins the messages; a driver branches on the
//! code, which only this file asserts.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Statement;
use nusadb_sql::{Catalog, ExecutionResult, Session, analyze, copy_from, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn run(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    session.execute(plan(logical))
}

fn ok(session: &mut Session, engine: &dyn StorageEngine, sql: &str) {
    run(session, engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The SQLSTATE and message `sql` is refused with.
fn refusal(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> (&'static str, String) {
    match run(session, engine, sql) {
        Ok(result) => panic!("{sql} succeeded: {result:?}"),
        Err(e) => (e.sqlstate(), e.to_string()),
    }
}

#[test]
fn a_number_too_large_for_the_column_is_22003_on_every_write_path() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "CREATE TABLE tn (v NUMERIC(6,2) UNIQUE)");
    ok(&mut s, &engine, "INSERT INTO tn VALUES (1)");
    ok(&mut s, &engine, "CREATE TABLE tna (a NUMERIC(5,2)[])");
    ok(
        &mut s,
        &engine,
        "CREATE TABLE tnd (v NUMERIC(3,1) DEFAULT 123.4, k INT)",
    );
    ok(&mut s, &engine, "CREATE TABLE tnc (v NUMERIC(10,2))");
    ok(&mut s, &engine, "INSERT INTO tnc VALUES (123456.78)");
    for sql in [
        "INSERT INTO tn VALUES (12345.6)",
        "INSERT INTO tn VALUES ('12345.6')",
        "INSERT INTO tn VALUES (9999.995)",
        "INSERT INTO tn SELECT 123456",
        "UPDATE tn SET v = v * 1000000",
        "INSERT INTO tn VALUES (1) ON CONFLICT (v) DO UPDATE SET v = tn.v * 1000000",
        "MERGE INTO tn USING (SELECT 1 AS k) s ON false WHEN NOT MATCHED THEN INSERT VALUES (123456)",
        "INSERT INTO tna VALUES (ARRAY[1234.5])",
        "INSERT INTO tnd (k) VALUES (1)",
        "ALTER TABLE tnc ALTER COLUMN v TYPE NUMERIC(5,2)",
        "SELECT 12345.6::NUMERIC(6,2)",
    ] {
        let (code, message) = refusal(&mut s, &engine, sql);
        assert_eq!(code, "22003", "{sql}: {message}");
        assert!(
            message.starts_with("numeric field overflow: value exceeds NUMERIC("),
            "{sql}: {message}"
        );
    }

    let Statement::Copy(copy) = parse("COPY tn FROM STDIN").expect("parse copy") else {
        panic!("expected COPY");
    };
    let err = copy_from(&engine, &copy, "12345.6\n").expect_err("does not fit");
    assert_eq!(err.sqlstate(), "22003", "{err}");
}

#[test]
fn text_that_is_not_a_number_stays_22p02_without_debug_formatting() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "CREATE TABLE tn (v NUMERIC(6,2))");
    for sql in [
        "INSERT INTO tn VALUES ('abc')",
        "SELECT 'abc'::NUMERIC(6,2)",
        "SELECT 'abc'::NUMERIC",
    ] {
        let (code, message) = refusal(&mut s, &engine, sql);
        assert_eq!(code, "22P02", "{sql}: {message}");
        assert_eq!(
            message, "invalid input syntax for type numeric: \"abc\"",
            "{sql}"
        );
    }
    let Statement::Copy(copy) = parse("COPY tn FROM STDIN").expect("parse copy") else {
        panic!("expected COPY");
    };
    let err = copy_from(&engine, &copy, "abc\n").expect_err("not a number");
    assert_eq!(err.sqlstate(), "22P02", "{err}");
}

#[test]
fn a_float_or_exponent_number_too_large_is_22003_and_not_22p02() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    ok(
        &mut s,
        &engine,
        "CREATE TABLE tn (v NUMERIC(5,2), a NUMERIC(5,2)[], u NUMERIC)",
    );
    for sql in [
        "SELECT 1e300::FLOAT8::NUMERIC(5,2)",
        "SELECT 1e300::FLOAT8::NUMERIC",
        "SELECT '1e100'::NUMERIC",
        "SELECT '1e100'::NUMERIC(5,2)",
        "SELECT 'Infinity'::FLOAT8::NUMERIC(5,2)",
        "SELECT '-Infinity'::FLOAT8::NUMERIC",
        "SELECT 'Infinity'::NUMERIC(5,2)",
        "INSERT INTO tn (v) VALUES (1e300::FLOAT8)",
        "INSERT INTO tn (v) VALUES ('1e100')",
        "INSERT INTO tn (v) VALUES ('Infinity'::FLOAT8)",
        "INSERT INTO tn (u) VALUES (1e300::FLOAT8)",
        "INSERT INTO tn (a) VALUES ('{1e100}')",
    ] {
        let (code, message) = refusal(&mut s, &engine, sql);
        assert_eq!(code, "22003", "{sql}: {message}");
    }
    for sql in ["SELECT '1e'::NUMERIC", "SELECT '1.2.3e4'::NUMERIC(5,2)"] {
        let (code, message) = refusal(&mut s, &engine, sql);
        assert_eq!(code, "22P02", "{sql}: {message}");
    }
    let Statement::Copy(copy) = parse("COPY tn (v) FROM STDIN").expect("parse copy") else {
        panic!("expected COPY");
    };
    for payload in ["1e100\n", "Infinity\n"] {
        let err = copy_from(&engine, &copy, payload).expect_err("does not fit");
        assert_eq!(err.sqlstate(), "22003", "{payload}: {err}");
    }
    assert_eq!(copy_from(&engine, &copy, "1.5e1\nNaN\n").expect("loads"), 2);
}

#[test]
fn an_exponent_of_any_size_is_read_without_fault() {
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    ok(
        &mut s,
        &engine,
        "CREATE TABLE tx (v NUMERIC, w NUMERIC(5,2))",
    );
    for sql in [
        "SELECT '1e-9223372036854775807'::NUMERIC",
        "SELECT '-1e-9223372036854775808'::NUMERIC",
        "SELECT '1e9223372036854775807'::NUMERIC",
        "SELECT '1e99999999999999999999'::NUMERIC(5,2)",
        "SELECT '1e-1000000000000000000'::NUMERIC(5,2)",
        "INSERT INTO tx (v) VALUES ('1e-9223372036854775790')",
        "INSERT INTO tx (w) VALUES ('9e+9223372036854775807')",
    ] {
        let (code, message) = refusal(&mut s, &engine, sql);
        assert_eq!(code, "22003", "{sql}: {message}");
    }
    let Statement::Copy(copy) = parse("COPY tx (v) FROM STDIN").expect("parse copy") else {
        panic!("expected COPY");
    };
    let err = copy_from(&engine, &copy, "1e-9223372036854775790\n").expect_err("out of range");
    assert_eq!(err.sqlstate(), "22003", "{err}");
    assert_eq!(copy_from(&engine, &copy, "1e-1000\n").expect("loads"), 1);
}
