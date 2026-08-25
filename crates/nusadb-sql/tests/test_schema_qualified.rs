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

/// Whether `sql` is refused *for having too many name parts*.
///
/// Matching only `Error::Unsupported` would also accept a refusal for an unrelated reason — the
/// statement losing support entirely, say — and quietly go on reporting that three-part names are
/// rejected. The message is what distinguishes the two.
fn rejected(sql: &str) -> bool {
    matches!(parse(sql), Err(Error::Unsupported(m)) if m.contains("more than two parts"))
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
    assert!(rejected("COMMENT ON TABLE d.app.t IS 'x'"));
    assert!(rejected("COPY d.app.t TO STDOUT"));
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

    // `COMMENT ON TABLE` resolves its target and discards the text — nothing persists a comment
    // yet. Routing it through the shared resolver also subjects it to the reserved-catalog and
    // row-level-security gates every other table-resolving statement already passes. Neither is
    // observable here: this `Catalog` reports `is_superuser() == true` and `rls_enabled() == false`
    // by default, so the RLS gate cannot fire in this harness at all.
    //
    // The target has to be a table that exists ONLY in `app`: against the shadowed `app.t`,
    // ignoring the qualifier would still find the default `t` and pass — a test proving the
    // qualifier was accepted rather than resolved.
    exec(engine, &mut session, "CREATE TABLE app.only_here (a INT)");
    exec(
        engine,
        &mut session,
        "COMMENT ON TABLE app.only_here IS 'the app table'",
    );

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

/// A qualifier that the catalog could not honour is refused for that reason, not as unfinished
/// plumbing.
///
/// Policies and triggers record their table by bare name (`nusadb_policies` and `nusadb_triggers`
/// both store it as plain text), so one catalog row would serve two same-named tables in different
/// schemas. Accepting `CREATE POLICY p ON app.t` would parse the qualifier and then govern
/// `public.t` with it — worse than refusing, because the statement would appear to have worked.
///
/// The generic message says a qualifier "does not resolve one yet", which reads as work in
/// progress. Here the blocker is a missing schema column in the object's own catalog.
#[test]
fn a_qualifier_the_catalog_cannot_hold_says_so() {
    for (sql, catalog) in [
        ("CREATE POLICY p ON app.t USING (true)", "policy catalog"),
        ("DROP POLICY p ON app.t", "policy catalog"),
        ("ALTER POLICY p ON app.t TO alice", "policy catalog"),
        (
            "CREATE TRIGGER tg BEFORE INSERT ON app.t EXECUTE FUNCTION f()",
            "trigger catalog",
        ),
        ("DROP TRIGGER tg ON app.t", "trigger catalog"),
        ("ALTER TRIGGER tg ON app.t DISABLE", "trigger catalog"),
        // `public.` is refused here too, where the old helper collapsed it to a bare name. That
        // collapse is not harmless: the analyzer re-resolves the bare name through the search
        // path, so under `SET search_path = app` an explicit `public.t` would have attached the
        // policy to `app.t`. Pinned because it is a narrowing of what parsed before, and a
        // narrowing nobody pinned is one nobody notices.
        ("CREATE POLICY p ON public.t USING (true)", "policy catalog"),
        ("DROP TRIGGER tg ON public.t", "trigger catalog"),
    ] {
        let Err(Error::Unsupported(message)) = parse(sql) else {
            panic!("expected `{sql}` to be refused while the catalog is unqualified");
        };
        assert!(
            message.contains(catalog),
            "`{sql}` should name the {catalog} as the blocker, got: {message}"
        );
        assert!(
            !message.contains("does not resolve one yet"),
            "`{sql}` still reports a catalog limit as unfinished plumbing: {message}"
        );
        // The same generated-continuation defect that has reached a user before.
        assert!(
            !message.contains("  "),
            "`{sql}` carries a run of spaces in its message: {message}"
        );
    }
}
