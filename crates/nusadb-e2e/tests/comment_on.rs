//! `COMMENT ON` against the durable engine: comments survive a reopen, a missing object reports
//! its kind's SQLSTATE, a routine signature must match, and objects dropped by cascade take their
//! comments with them.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, Session, analyze, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table_in(schema, name).map_err(Into::into)
    }
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    Session::new(engine).execute(plan(logical))
}

fn run(engine: &BtreeEngine, sql: &str) {
    run_try(engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn code(engine: &BtreeEngine, sql: &str) -> &'static str {
    run_try(engine, sql).expect_err(sql).sqlstate()
}

/// `(object_type, object_schema, table_name, object_name, description)` per comment, sorted.
fn comments(engine: &BtreeEngine) -> Vec<String> {
    let sql = "SELECT object_type, object_schema, table_name, object_name, description \
               FROM information_schema.nusadb_comments";
    let ExecutionResult::Rows { rows, .. } = run_try(engine, sql).expect(sql) else {
        panic!("expected rows");
    };
    let mut out: Vec<String> = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|v| match v {
                    Value::Text(s) => s,
                    Value::Null => "NULL".to_owned(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    out.sort();
    out
}

fn setup(engine: &BtreeEngine) {
    run(engine, "CREATE SCHEMA app");
    run(
        engine,
        "CREATE TABLE app.t (id INT, v INT, CONSTRAINT t_pk PRIMARY KEY (id), \
         CONSTRAINT v_pos CHECK (v > 0))",
    );
    run(engine, "CREATE INDEX t_v_idx ON app.t (v)");
    run(
        engine,
        "CREATE FUNCTION twice(x INT) RETURNS INT AS 'SELECT x * 2' LANGUAGE SQL",
    );
    run(engine, "CREATE SEQUENCE app.s");
    run(engine, "CREATE TYPE mood AS ENUM ('ok', 'sad')");
}

#[test]
fn comments_survive_a_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine = BtreeEngine::open(&path).expect("open");
        setup(&engine);
        run(&engine, "COMMENT ON TABLE app.t IS 'facts'");
        run(&engine, "COMMENT ON COLUMN app.t.v IS 'value'");
        run(
            &engine,
            "COMMENT ON CONSTRAINT v_pos ON app.t IS 'positive'",
        );
        run(&engine, "COMMENT ON INDEX t_v_idx IS 'by value'");
        run(&engine, "COMMENT ON FUNCTION twice(int) IS 'doubles'");
        run(&engine, "COMMENT ON SCHEMA app IS 'the app'");
        run(&engine, "COMMENT ON SEQUENCE app.s IS 'ids'");
        run(&engine, "COMMENT ON TYPE mood IS 'feelings'");
    }
    let engine = BtreeEngine::open(&path).expect("reopen");
    assert_eq!(
        comments(&engine),
        vec![
            "COLUMN|app|t|v|value",
            "CONSTRAINT|app|t|v_pos|positive",
            "FUNCTION|public|NULL|twice|doubles",
            "INDEX|app|t|t_v_idx|by value",
            "SCHEMA|app|NULL|app|the app",
            "SEQUENCE|app|NULL|s|ids",
            "TABLE|app|t|t|facts",
            "TYPE|public|NULL|mood|feelings",
        ]
    );
    // Still writable after the reopen.
    run(&engine, "COMMENT ON TABLE app.t IS NULL");
    assert!(!comments(&engine).iter().any(|c| c.starts_with("TABLE|")));
}

#[test]
fn missing_objects_report_their_kind() {
    let engine = BtreeEngine::new();
    setup(&engine);
    assert_eq!(code(&engine, "COMMENT ON TABLE app.nope IS 'x'"), "42P01");
    assert_eq!(
        code(&engine, "COMMENT ON COLUMN app.t.nope IS 'x'"),
        "42703"
    );
    assert_eq!(
        code(&engine, "COMMENT ON CONSTRAINT nope ON app.t IS 'x'"),
        "42704"
    );
    assert_eq!(
        code(&engine, "COMMENT ON CONSTRAINT v_pos ON app.nope IS 'x'"),
        "42P01"
    );
    assert_eq!(code(&engine, "COMMENT ON INDEX nope IS 'x'"), "42704");
    assert_eq!(
        code(&engine, "COMMENT ON TRIGGER nope ON app.t IS 'x'"),
        "42704"
    );
    assert_eq!(code(&engine, "COMMENT ON FUNCTION nope IS 'x'"), "42883");
    assert_eq!(code(&engine, "COMMENT ON SCHEMA nope IS 'x'"), "3F000");
    assert_eq!(
        code(&engine, "COMMENT ON SEQUENCE app.nope IS 'x'"),
        "42704"
    );
    assert_eq!(code(&engine, "COMMENT ON TYPE nope IS 'x'"), "42704");
    assert_eq!(code(&engine, "COMMENT ON DOMAIN mood IS 'x'"), "42809");
    // `IS NULL` on a missing object is refused the same way.
    assert_eq!(code(&engine, "COMMENT ON INDEX nope IS NULL"), "42704");
}

#[test]
fn a_routine_signature_must_match() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // Function names are unique, so the bare name is never ambiguous; a signature with the wrong
    // number of arguments names a function that does not exist.
    run(&engine, "COMMENT ON FUNCTION twice IS 'a'");
    run(&engine, "COMMENT ON FUNCTION twice(INTEGER) IS 'b'");
    assert_eq!(code(&engine, "COMMENT ON FUNCTION twice() IS 'x'"), "42883");
    assert_eq!(
        code(&engine, "COMMENT ON FUNCTION twice(int, int) IS 'x'"),
        "42883"
    );
    assert_eq!(comments(&engine), vec!["FUNCTION|public|NULL|twice|b"]);
    // Replacing the function keeps its comment; dropping it removes the comment.
    run(
        &engine,
        "CREATE OR REPLACE FUNCTION twice(x INT) RETURNS INT AS 'SELECT x + x' LANGUAGE SQL",
    );
    assert_eq!(comments(&engine), vec!["FUNCTION|public|NULL|twice|b"]);
    run(&engine, "DROP FUNCTION twice");
    assert!(comments(&engine).is_empty());
    run(
        &engine,
        "CREATE FUNCTION twice(x INT) RETURNS INT AS 'SELECT x * 2' LANGUAGE SQL",
    );
    assert!(comments(&engine).is_empty());
}

#[test]
fn cascaded_drops_take_the_comments_along() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TABLE app.child (id INT, tid INT, CONSTRAINT child_fk FOREIGN KEY (tid) REFERENCES app.t (id))",
    );
    run(
        &engine,
        "COMMENT ON CONSTRAINT child_fk ON app.child IS 'parent link'",
    );
    run(&engine, "COMMENT ON COLUMN app.t.v IS 'value'");
    run(&engine, "COMMENT ON INDEX t_v_idx IS 'by value'");
    run(
        &engine,
        "COMMENT ON CONSTRAINT v_pos ON app.t IS 'positive'",
    );
    // Dropping a column drops the index and check on it, and their comments.
    run(&engine, "ALTER TABLE app.t DROP COLUMN v");
    assert_eq!(
        comments(&engine),
        vec!["CONSTRAINT|app|child|child_fk|parent link"]
    );
    // DROP TABLE ... CASCADE drops the foreign key on the other table, and its comment.
    run(&engine, "DROP TABLE app.t CASCADE");
    assert!(comments(&engine).is_empty());
}

#[test]
fn a_comment_is_private_to_its_transaction_until_commit() {
    let engine = BtreeEngine::new();
    setup(&engine);
    let mut writer = Session::new(&engine);
    let catalog = EngineCatalog(&engine);
    let exec = |session: &mut Session<'_>, sql: &str| {
        let logical = analyze(parse(sql).expect(sql), &catalog).expect(sql);
        session.execute(plan(logical)).expect(sql);
    };
    exec(&mut writer, "BEGIN");
    exec(&mut writer, "COMMENT ON TABLE app.t IS 'pending'");
    assert!(comments(&engine).is_empty(), "visible before commit");
    exec(&mut writer, "COMMIT");
    assert_eq!(comments(&engine), vec!["TABLE|app|t|t|pending"]);
}
