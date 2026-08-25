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

    // A `public.`-qualified CREATE / INSERT / DROP all denote the same table as the bare name.
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
    // The statements newly routed through the schema-aware helper have to keep refusing a third
    // part. Widening a name path is exactly when this stops being true for free.
    assert!(rejected("ANALYZE d.app.t"));
    assert!(rejected("SHOW COLUMNS FROM d.app.t"));
    assert!(rejected("CREATE TABLE d.app.u AS SELECT 1 AS a"));
    // A `public`-qualified table with a non-public column qualifier (or extra parts) is rejected.
    assert!(rejected("SELECT a.b.c.d FROM t"));
    assert!(rejected("SELECT public.t.a.extra FROM public.t"));
}

/// Statements that name a table must resolve a schema qualifier, not refuse it.
///
/// These went through a narrow name helper that understood only a bare name or `public.`, and
/// answered anything else with "it does not resolve one yet" — while `CREATE TABLE app.t`,
/// `SELECT`, `INSERT` and `DROP` had resolved qualifiers for some time. The refusal was not a
/// design boundary, it was a helper nobody had revisited.
///
/// Each case here targets a table in `app` that is *shadowed* by a same-named table in the default
/// schema, so accepting the qualifier is not enough — it has to reach the right one.
#[test]
fn table_statements_resolve_a_schema_qualifier() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE SCHEMA app");
    exec(engine, &mut session, "CREATE TABLE app.t (a INT, b TEXT)");
    exec(engine, &mut session, "CREATE TABLE t (zzz INT)");
    exec(engine, &mut session, "INSERT INTO app.t VALUES (1, 'x')");

    // Names a column that exists only in `app.t`, so resolving to the default `t` would fail
    // rather than quietly analyse the wrong table.
    exec(engine, &mut session, "ANALYZE app.t (b)");

    // `SHOW COLUMNS` must describe `app.t`, not the default `t` that shadows it.
    let described = rows(engine, &mut session, "SHOW COLUMNS FROM app.t");
    let names: Vec<String> = described
        .iter()
        .map(|r| match r.first() {
            Some(Value::Text(name)) => name.clone(),
            other => panic!("unexpected column-name value: {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["a", "b"], "described the wrong table");

    // The headline case: `CREATE TABLE app.t` had resolved a qualifier for some time while
    // `CREATE TABLE app.u AS SELECT` refused one, and the executor wrote `public` regardless.
    exec(
        engine,
        &mut session,
        "CREATE TABLE app.u AS SELECT a FROM app.t",
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT a FROM app.u"),
        vec![vec![Value::Int(1)]]
    );
    // It landed in `app`, not `public` — the default schema must still have no `u`.
    assert!(
        parse("SELECT a FROM u").is_ok(),
        "the statement itself is well-formed; only resolution should fail"
    );
    let unqualified = analyze(parse("SELECT a FROM u").unwrap(), &Cat(engine));
    assert!(
        matches!(unqualified, Err(Error::TableNotFound { .. })),
        "CTAS created the table in the default schema instead of `app`: {unqualified:?}"
    );
}

/// A qualifier that cannot mean anything must still be refused — and say why.
///
/// `CREATE SCHEMA app.x` is not a feature gap: a schema has no parent schema to sit in. The old
/// message ("it does not resolve one yet") described every refusal as unfinished work, which is
/// wrong here in the direction that matters — it invites a reader to wait for a release that will
/// never change this.
#[test]
fn a_qualifier_that_cannot_mean_anything_is_refused_as_such() {
    let Err(err) = parse("CREATE SCHEMA app.x") else {
        panic!("a schema cannot live inside another schema");
    };
    let message = err.to_string();
    assert!(
        !message.contains("does not resolve one yet"),
        "refused as unfinished work rather than as meaningless: {message}"
    );
}
