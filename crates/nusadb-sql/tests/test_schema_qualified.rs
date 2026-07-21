//! Schema-qualified names: `public.table` / `public.table.column` resolve to the default namespace,
//! and a non-`public` qualifier `schema.table` resolves to that real schema. Names
//! with more parts (`db.schema.table`) are still rejected (no silent collapse).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Row, Session, analyze, parse, plan};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table_in(schema, name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session.execute(plan(logical)).unwrap()
}

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

fn rejected(sql: &str) -> bool {
    matches!(parse(sql), Err(Error::Unsupported(_)))
}

#[test]
fn public_schema_qualifier_resolves_to_the_bare_table() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    // A `public.`-qualified CREATE / INSERT / DROP all denote the single namespace.
    exec(
        engine,
        &mut session,
        "CREATE TABLE public.t (a INT, b TEXT)",
    );
    exec(engine, &mut session, "INSERT INTO public.t VALUES (1, 'x')");
    exec(engine, &mut session, "INSERT INTO t VALUES (2, 'y')");

    // `public.t` and `t` are the same table; a `public.t.col` column ref resolves like `t.col`.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT public.t.a, t.b FROM public.t WHERE public.t.a = 1"
        ),
        vec![vec![Value::Int(1), Value::Text("x".to_owned())]]
    );
    // A `public.t.*` wildcard expands like `t.*`.
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT public.t.* FROM public.t ORDER BY a"
        ),
        vec![
            vec![Value::Int(1), Value::Text("x".to_owned())],
            vec![Value::Int(2), Value::Text("y".to_owned())],
        ]
    );
    // A bare table mixed with a public-qualified column resolves too.
    assert_eq!(
        rows(engine, &mut session, "SELECT a FROM t WHERE public.t.a = 2"),
        vec![vec![Value::Int(2)]]
    );

    assert!(matches!(
        exec(engine, &mut session, "DROP TABLE public.t"),
        ExecutionResult::Dropped
    ));
}

#[test]
fn non_public_schema_qualifier_resolves_to_that_schema() {
    // `schema.table` resolves to the real schema: a non-public table is created, queried, and
    // dropped, and it is distinct from a same-named table in the default schema.
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE SCHEMA app");
    exec(
        engine,
        &mut session,
        "CREATE TABLE app.users (id INT, name TEXT)",
    );
    exec(
        engine,
        &mut session,
        "CREATE TABLE users (id INT, name TEXT)",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO app.users VALUES (1, 'in_app')",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO users VALUES (2, 'in_public')",
    );

    // Each name resolves to its own table — independent rows.
    assert_eq!(
        rows(engine, &mut session, "SELECT id, name FROM app.users"),
        vec![vec![Value::Int(1), Value::Text("in_app".to_owned())]]
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT id, name FROM users"),
        vec![vec![Value::Int(2), Value::Text("in_public".to_owned())]]
    );

    // Dropping the qualified table leaves the default-schema one intact.
    assert!(matches!(
        exec(engine, &mut session, "DROP TABLE app.users"),
        ExecutionResult::Dropped
    ));
    assert!(matches!(
        analyze(parse("SELECT id FROM app.users").unwrap(), &Cat(engine)),
        Err(Error::TableNotFound { .. })
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM users"),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn deeper_qualifiers_stay_rejected() {
    // A three-part table name (`db.schema.table`) must not silently collapse.
    assert!(rejected("DROP TABLE d.app.users"));
    assert!(rejected("CREATE TABLE d.app.users (id INT)"));
    // A `public`-qualified table with a non-public column qualifier (or extra parts) is rejected.
    assert!(rejected("SELECT a.b.c.d FROM t"));
    assert!(rejected("SELECT public.t.a.extra FROM public.t"));
}
