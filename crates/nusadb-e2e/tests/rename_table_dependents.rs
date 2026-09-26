//! `ALTER TABLE … RENAME TO` and the catalogs that know a table by name: policies, the
//! row-security marker, triggers, column grants, column defaults and column type tags travel
//! with the table, nothing stays attached to the vacated name, and an object that can only find
//! the table by re-reading its SQL text blocks the rename instead of silently breaking.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, Session, analyze, execute, parse, plan};

/// A catalog that answers row-security questions for `user`, the way an authenticated
/// connection's would.
struct RlsCatalog<'a> {
    engine: &'a BtreeEngine,
    superuser: bool,
    user: &'a str,
}

impl Catalog for RlsCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }

    fn is_superuser(&self) -> bool {
        self.superuser
    }

    fn current_user(&self) -> String {
        self.user.to_owned()
    }

    fn rls_enabled(&self, schema: &str, name: &str) -> Result<bool, nusadb_sql::Error> {
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let enabled = nusadb_sql::rls_table_enabled(self.engine, txn, schema, name);
        let _ = self.engine.rollback(txn);
        enabled
    }

    fn lookup_policies(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Vec<nusadb_sql::PolicyDef>, nusadb_sql::Error> {
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let policies = nusadb_sql::lookup_policies_for(self.engine, txn, schema, name);
        let _ = self.engine.rollback(txn);
        policies
    }
}

const fn root_catalog(engine: &BtreeEngine) -> RlsCatalog<'_> {
    RlsCatalog {
        engine,
        superuser: true,
        user: "root",
    }
}

/// Run `sql` as the superuser.
fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &root_catalog(engine))
        .unwrap_or_else(|e| panic!("`{sql}` should analyze: {e}"));
    execute(plan(logical), engine).unwrap_or_else(|e| panic!("`{sql}` should succeed: {e}"))
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
    let logical = analyze(parse(sql).unwrap(), &root_catalog(engine))?;
    execute(plan(logical), engine)
}

/// Run `sql` as a non-superuser, with the policies selected for that user and `CURRENT_USER`
/// evaluating to them.
fn as_user(engine: &BtreeEngine, user: &'static str, sql: &str) -> Vec<Vec<Value>> {
    let logical = analyze(
        parse(sql).unwrap(),
        &RlsCatalog {
            engine,
            superuser: false,
            user,
        },
    )
    .expect("analyze");
    let mut session = Session::new(engine);
    session.set_current_user(user);
    rows(session.execute(plan(logical)).expect("execute"))
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

fn ids(rows: Vec<Vec<Value>>) -> Vec<i64> {
    let mut ids: Vec<i64> = rows
        .into_iter()
        .map(|r| match r.first() {
            Some(Value::Int(n)) => *n,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn count(engine: &BtreeEngine, sql: &str) -> i64 {
    ids(rows(run(engine, sql))).into_iter().next().unwrap()
}

#[test]
fn rename_carries_row_security_and_triggers_and_leaves_the_old_name_bare() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob')");
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT USING (owner = CURRENT_USER)",
    );
    run(&engine, "CREATE TABLE audit (v INT)");
    run(
        &engine,
        "CREATE TRIGGER trg AFTER INSERT ON doc FOR EACH ROW INSERT INTO audit VALUES (new.id)",
    );
    assert_eq!(
        ids(as_user(&engine, "alice", "SELECT id FROM doc")),
        vec![1]
    );

    run(&engine, "ALTER TABLE doc RENAME TO doc2");

    // The renamed table is still protected by its policy, and its trigger still fires.
    assert_eq!(
        ids(as_user(&engine, "alice", "SELECT id FROM doc2")),
        vec![1]
    );
    run(&engine, "INSERT INTO doc2 VALUES (3, 'bob')");
    assert_eq!(count(&engine, "SELECT count(*) FROM audit"), 1);

    // A new table taking the old name inherits none of it: no row security, no policy, no
    // trigger — and the policy name is free again.
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO doc VALUES (9, 'bob')");
    assert_eq!(
        ids(as_user(&engine, "alice", "SELECT id FROM doc")),
        vec![9]
    );
    assert_eq!(count(&engine, "SELECT count(*) FROM audit"), 1);
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT USING (owner = CURRENT_USER)",
    );
}

#[test]
fn rename_carries_column_defaults_and_enum_tags() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')");
    run(
        &engine,
        "CREATE TABLE s (id SERIAL, v INT, m mood DEFAULT 'ok', n INT DEFAULT 7)",
    );
    run(&engine, "INSERT INTO s (v) VALUES (1)");

    run(&engine, "ALTER TABLE s RENAME TO s2");

    // SERIAL keeps counting, DEFAULTs keep filling, and the enum column still resolves labels.
    run(&engine, "INSERT INTO s2 (v) VALUES (2)");
    run(&engine, "INSERT INTO s2 (v, m) VALUES (3, 'happy')");
    assert_eq!(ids(rows(run(&engine, "SELECT id FROM s2"))), vec![1, 2, 3]);
    assert_eq!(count(&engine, "SELECT count(*) FROM s2 WHERE n = 7"), 3);
    // The enum tag moved: labels still render on read and resolve on write.
    let label = |v: i64| {
        format!(
            "{:?}",
            rows(run(&engine, &format!("SELECT m FROM s2 WHERE v = {v}")))
        )
    };
    assert!(label(2).contains("ok"), "{}", label(2));
    assert!(label(3).contains("happy"), "{}", label(3));

    // A table reusing the old name gets no default and no enum tag from its predecessor.
    run(&engine, "CREATE TABLE s (id INT, v INT, m TEXT, n INT)");
    run(
        &engine,
        "INSERT INTO s (id, v, m) VALUES (1, 1, 'whatever')",
    );
    assert_eq!(count(&engine, "SELECT count(*) FROM s WHERE n IS NULL"), 1);
}

#[test]
fn rename_is_refused_while_sql_text_dependents_name_the_table() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL)");
    run(&engine, "CREATE VIEW v AS SELECT id FROM doc");
    let err = run_try(&engine, "ALTER TABLE doc RENAME TO doc2").expect_err("a view names it");
    assert!(
        err.to_string().contains("view \"v\""),
        "the refusal names the dependent: {err}"
    );
    assert_eq!(err.sqlstate(), "2BP01");
    // The table is untouched by the refusal.
    run(&engine, "INSERT INTO doc VALUES (1)");

    run(&engine, "DROP VIEW v");
    run(&engine, "ALTER TABLE doc RENAME TO doc2");
    assert_eq!(count(&engine, "SELECT count(*) FROM doc2"), 1);

    // A trigger on another table whose body spells the name blocks too.
    run(&engine, "CREATE TABLE other (v INT)");
    run(
        &engine,
        "CREATE TRIGGER t2 AFTER INSERT ON other FOR EACH ROW INSERT INTO doc2 VALUES (new.v)",
    );
    let err = run_try(&engine, "ALTER TABLE doc2 RENAME TO doc3").expect_err("a trigger names it");
    assert!(err.to_string().contains("trigger \"t2\""), "{err}");
}
