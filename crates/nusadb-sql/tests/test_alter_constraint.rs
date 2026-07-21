//! `ALTER TABLE ADD/DROP CONSTRAINT` for PRIMARY KEY / UNIQUE / FOREIGN KEY / CHECK: adding
//! validates the existing rows and then enforces the constraint on later writes; dropping releases
//! it. `CREATE TABLE` CHECK constraints are covered here too.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session.execute(plan(logical)).unwrap()
}

fn try_exec(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine))?;
    session.execute(plan(logical))
}

#[test]
fn add_unique_validates_then_enforces_and_drop_releases() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT, b INT)");
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    );

    // The existing rows have distinct `a`, so adding UNIQUE(a) succeeds.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD CONSTRAINT uq_a UNIQUE (a)"
        ),
        ExecutionResult::Altered
    ));
    // The constraint is now enforced — a duplicate `a` is rejected.
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (1, 99)").is_err());

    // Dropping it releases the constraint — the duplicate now inserts.
    assert!(matches!(
        exec(engine, &mut session, "ALTER TABLE t DROP CONSTRAINT uq_a"),
        ExecutionResult::Altered
    ));
    assert!(matches!(
        exec(engine, &mut session, "INSERT INTO t VALUES (1, 99)"),
        ExecutionResult::Inserted(1)
    ));
}

#[test]
fn add_constraint_rejects_violating_existing_rows() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT, b INT)");
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (1, 20)",
    );

    // Existing rows already violate UNIQUE(a) → the ADD is rejected (the constraint is not created).
    assert!(try_exec(engine, &mut session, "ALTER TABLE t ADD UNIQUE (a)").is_err());
    // A PRIMARY KEY over a NULL column is rejected.
    exec(engine, &mut session, "CREATE TABLE n (a INT, b INT)");
    exec(engine, &mut session, "INSERT INTO n VALUES (NULL, 1)");
    assert!(try_exec(engine, &mut session, "ALTER TABLE n ADD PRIMARY KEY (a)").is_err());
}

#[test]
fn drop_constraint_missing_and_unsupported_adds() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (a INT PRIMARY KEY, b INT, parent INT)",
    );

    // DROP CONSTRAINT on a missing name errors; IF EXISTS makes it a no-op.
    assert!(try_exec(engine, &mut session, "ALTER TABLE t DROP CONSTRAINT nope").is_err());
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "ALTER TABLE t DROP CONSTRAINT IF EXISTS nope"
        ),
        ExecutionResult::Altered
    ));
}

#[test]
fn create_table_check_enforces_on_insert_and_update() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT, qty INT CHECK (qty > 0), CHECK (id >= 0))",
    );

    // Rows satisfying both the column-level and table-level CHECK insert fine.
    assert!(matches!(
        exec(engine, &mut session, "INSERT INTO t VALUES (1, 5)"),
        ExecutionResult::Inserted(1)
    ));
    // Violating the column-level CHECK (qty > 0) is rejected…
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (2, 0)").is_err());
    // …as is violating the table-level CHECK (id >= 0).
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (-1, 5)").is_err());

    // UPDATE is enforced on the same paths: a write that breaks the predicate is rejected,
    // and the prior value is preserved.
    assert!(try_exec(engine, &mut session, "UPDATE t SET qty = -3 WHERE id = 1").is_err());
    assert!(matches!(
        exec(engine, &mut session, "UPDATE t SET qty = 9 WHERE id = 1"),
        ExecutionResult::Updated(1)
    ));
}

#[test]
fn check_passes_on_null_and_add_check_validates_then_enforces() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT, qty INT)");
    exec(engine, &mut session, "INSERT INTO t VALUES (1, 5), (2, 10)");

    // A CHECK predicate that evaluates to NULL (here `qty > 0` with a NULL qty) does NOT fail —
    // only an explicit FALSE is a violation (SQL three-valued semantics).
    exec(
        engine,
        &mut session,
        "CREATE TABLE n (qty INT CHECK (qty > 0))",
    );
    assert!(matches!(
        exec(engine, &mut session, "INSERT INTO n VALUES (NULL)"),
        ExecutionResult::Inserted(1)
    ));

    // The existing rows of `t` all satisfy `qty > 0`, so ADD CHECK succeeds and then enforces.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD CONSTRAINT positive_qty CHECK (qty > 0)"
        ),
        ExecutionResult::Altered
    ));
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (3, -1)").is_err());

    // Dropping the constraint releases enforcement.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "ALTER TABLE t DROP CONSTRAINT positive_qty"
        ),
        ExecutionResult::Altered
    ));
    assert!(matches!(
        exec(engine, &mut session, "INSERT INTO t VALUES (3, -1)"),
        ExecutionResult::Inserted(1)
    ));

    // ADD CHECK is rejected when an existing row already violates it.
    assert!(try_exec(engine, &mut session, "ALTER TABLE t ADD CHECK (qty > 0)").is_err());

    // A subquery in a CHECK predicate is rejected at analysis time.
    assert!(matches!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD CHECK (qty > (SELECT 1))"
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn add_foreign_key_validates_and_enforces() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE p (id INT PRIMARY KEY)");
    exec(engine, &mut session, "CREATE TABLE c (cid INT, pid INT)");
    exec(engine, &mut session, "INSERT INTO p VALUES (1), (2)");
    exec(
        engine,
        &mut session,
        "INSERT INTO c VALUES (10, 1), (20, 2)",
    );

    // Existing child rows all reference live parents, so adding the FK succeeds.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "ALTER TABLE c ADD CONSTRAINT fk FOREIGN KEY (pid) REFERENCES p (id)"
        ),
        ExecutionResult::Altered
    ));
    // The FK is now enforced — a child row referencing a missing parent is rejected.
    assert!(try_exec(engine, &mut session, "INSERT INTO c VALUES (30, 99)").is_err());
    // A valid reference still inserts.
    assert!(matches!(
        exec(engine, &mut session, "INSERT INTO c VALUES (30, 1)"),
        ExecutionResult::Inserted(1)
    ));

    // Adding an FK that the existing rows already violate is rejected (constraint not created).
    exec(engine, &mut session, "CREATE TABLE d (did INT, pid INT)");
    exec(engine, &mut session, "INSERT INTO d VALUES (1, 88)");
    assert!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE d ADD CONSTRAINT fk FOREIGN KEY (pid) REFERENCES p (id)"
        )
        .is_err()
    );
}
