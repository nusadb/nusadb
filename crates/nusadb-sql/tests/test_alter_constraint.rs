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

fn column_names(engine: &dyn StorageEngine, table: &str) -> Vec<String> {
    engine
        .lookup_table(table)
        .unwrap()
        .unwrap()
        .columns
        .into_iter()
        .map(|c| c.name)
        .collect()
}

#[test]
fn add_column_with_inline_check_validates_existing_rows() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT PRIMARY KEY)");
    exec(engine, &mut session, "INSERT INTO t VALUES (1), (2)");
    // The DEFAULT fills the existing rows and breaks the CHECK: the statement leaves no column.
    assert!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD COLUMN n INT NOT NULL DEFAULT 0 CHECK (n > 0)"
        )
        .is_err()
    );
    assert_eq!(column_names(engine, "t"), ["id"]);
    exec(
        engine,
        &mut session,
        "ALTER TABLE t ADD COLUMN n INT NOT NULL DEFAULT 1 CHECK (n > 0)",
    );
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (3, 0)").is_err());
    exec(engine, &mut session, "INSERT INTO t VALUES (3, 5)");
    // Named after the table and the column.
    let names: Vec<String> = engine
        .list_constraints(engine.lookup_table("t").unwrap().unwrap().id)
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"t_n_check".to_owned()), "{names:?}");
}

#[test]
fn add_column_with_inline_references_is_enforced_and_cascades() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE p (id INT PRIMARY KEY)");
    exec(engine, &mut session, "INSERT INTO p VALUES (1)");
    exec(engine, &mut session, "CREATE TABLE c (id INT PRIMARY KEY)");
    exec(engine, &mut session, "INSERT INTO c VALUES (10)");
    // An existing row would reference a missing parent: refused, no column left behind.
    assert!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE c ADD COLUMN p INT DEFAULT 9 REFERENCES p (id) ON DELETE CASCADE"
        )
        .is_err()
    );
    assert_eq!(column_names(engine, "c"), ["id"]);
    exec(
        engine,
        &mut session,
        "ALTER TABLE c ADD COLUMN p INT REFERENCES p (id) ON DELETE CASCADE",
    );
    assert!(try_exec(engine, &mut session, "INSERT INTO c VALUES (11, 9)").is_err());
    exec(engine, &mut session, "INSERT INTO c VALUES (11, 1)");
    exec(engine, &mut session, "DELETE FROM p");
    let ExecutionResult::Rows { rows, .. } =
        exec(engine, &mut session, "SELECT id FROM c ORDER BY id")
    else {
        panic!("expected rows");
    };
    assert_eq!(rows.len(), 1, "the referencing row cascaded away: {rows:?}");
}

#[test]
fn add_column_with_inline_unique_is_enforced() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT PRIMARY KEY)");
    exec(
        engine,
        &mut session,
        "ALTER TABLE t ADD COLUMN u INT UNIQUE",
    );
    exec(engine, &mut session, "INSERT INTO t VALUES (1, 5)");
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (2, 5)").is_err());
    // A DEFAULT that gives the existing rows equal values refuses the column; without one the
    // rows hold NULLs, which do not collide.
    exec(engine, &mut session, "INSERT INTO t VALUES (2, 6)");
    assert!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD COLUMN x INT DEFAULT 1 UNIQUE"
        )
        .is_err()
    );
    assert_eq!(column_names(engine, "t"), ["id", "u"]);
    exec(
        engine,
        &mut session,
        "ALTER TABLE t ADD COLUMN x INT UNIQUE",
    );
    exec(engine, &mut session, "INSERT INTO t VALUES (3, 7, 2)");
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (4, 8, 2)").is_err());
}

#[test]
fn add_column_smallint_and_integer_keep_range_bounds() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT PRIMARY KEY)");
    exec(engine, &mut session, "ALTER TABLE t ADD COLUMN s SMALLINT");
    exec(engine, &mut session, "ALTER TABLE t ADD COLUMN i INTEGER");
    exec(
        engine,
        &mut session,
        "ALTER TABLE t ADD COLUMN c VARCHAR(3)",
    );
    assert!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t (id, s) VALUES (1, 100000)"
        )
        .is_err()
    );
    assert!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t (id, i) VALUES (2, 10000000000)"
        )
        .is_err()
    );
    assert!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t (id, c) VALUES (3, 'abcdef')"
        )
        .is_err()
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (4, 32767, 2147483647, 'abc')",
    );
    // A DEFAULT outside the type's range refuses the column.
    assert!(
        try_exec(
            engine,
            &mut session,
            "ALTER TABLE t ADD COLUMN z SMALLINT DEFAULT 40000"
        )
        .is_err()
    );
    assert_eq!(column_names(engine, "t"), ["id", "s", "i", "c"]);
}

#[test]
fn dropping_a_ranged_column_drops_its_range_check() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT, s SMALLINT)");
    exec(engine, &mut session, "ALTER TABLE t ADD COLUMN i INTEGER");
    exec(engine, &mut session, "ALTER TABLE t DROP COLUMN i");
    exec(engine, &mut session, "ALTER TABLE t DROP COLUMN s");
    exec(engine, &mut session, "INSERT INTO t VALUES (1)");
    exec(engine, &mut session, "ALTER TABLE t ADD COLUMN s TEXT");
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (2, 'not a number')",
    );
}

#[test]
fn explain_names_an_added_ranged_column() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (id INT)");
    let ExecutionResult::Rows { rows, .. } = exec(
        engine,
        &mut session,
        "EXPLAIN ALTER TABLE t ADD COLUMN s SMALLINT",
    ) else {
        panic!("expected EXPLAIN rows");
    };
    let text = format!("{rows:?}");
    assert!(text.contains("ADD COLUMN s"), "{text}");
}
