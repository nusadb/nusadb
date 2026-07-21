//! The `--storage-engine btree` SQL bridge (design requirement, user order 2026-07-05): real SQL
//! strings through `parse → analyze → plan → execute` against the clustered B-link/B+tree
//! engine — the `DoD` that unblocks the QA SQL-verify.
//!
//! Covers exactly what QA reported broken: `CREATE TABLE` without a PK (used to fail with
//! `add_check_constraint is not implemented`), with a PK (`add_unique_constraint`), the
//! INSERT/SELECT/`WHERE pk = ?` smoke path, constraint *enforcement* (unique + check + FK,
//! which the SQL layer drives through `list_constraints`), and durability of it all across a
//! crash-reopen.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, analyze, execute, parse, plan};

/// Adapts the engine's schema lookup to the analyzer's narrower `Catalog` port (the same shape
/// `end_to_end.rs` was written against — engine-agnostic by construction).
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table_in(schema, name).map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        let backing: std::collections::HashSet<_> = self
            .0
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.0.list_indexes(schema.id)? {
            if self
                .0
                .lookup_index(&def.name)?
                .is_some_and(|id| backing.contains(&id))
            {
                continue;
            }
            // A functional/expression key or partial predicate is unsafe as a scan candidate —
            // mirror the production `catalog_list_indexes` exclusion.
            if !def.key_exprs.is_empty() || def.predicate.is_some() {
                continue;
            }
            out.push(IndexInfo {
                name: def.name,
                columns: def.columns,
                unique: def.unique,
            });
        }
        Ok(out)
    }

    fn table_stats(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(None);
        };
        self.0.table_stats(schema.id).map_err(Into::into)
    }
}

fn run(engine: &dyn StorageEngine, sql: &str) -> ExecutionResult {
    run_try(engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn run_try(engine: &dyn StorageEngine, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
    let stmt = parse(sql)?;
    let logical = analyze(stmt, &EngineCatalog(engine))?;
    execute(plan(logical), engine)
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected SELECT rows, got {other:?}"),
    }
}

/// The QA `DoD`, statement for statement: CREATE TABLE without a PK, with a PK, INSERT,
/// SELECT, and the point lookup `WHERE pk = ?`.
#[test]
fn create_table_insert_select_point_lookup() {
    let engine = BtreeEngine::new();

    // The exact statement QA reported failing (`add_check_constraint not implemented`).
    run(&engine, "CREATE TABLE plain (id INT, v INT)");
    run(&engine, "INSERT INTO plain VALUES (1, 10), (2, 20)");
    assert_eq!(rows(run(&engine, "SELECT v FROM plain")).len(), 2);

    // And the PK form QA reported failing (`add_unique_constraint not implemented`).
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL, qty INT)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'satu', 100), (2, 'dua', 200), (3, 'tiga', 300)",
    );
    let got = rows(run(&engine, "SELECT name, qty FROM t WHERE id = 2"));
    assert_eq!(
        got,
        vec![vec![Value::Text("dua".to_owned()), Value::Int(200),]]
    );
    // Range + order over the same table for good measure.
    let got = rows(run(
        &engine,
        "SELECT id FROM t WHERE qty >= 200 ORDER BY id DESC",
    ));
    assert_eq!(got, vec![vec![Value::Int(3)], vec![Value::Int(2)]]);
    // UPDATE + DELETE round out the smoke.
    run(&engine, "UPDATE t SET qty = 250 WHERE id = 2");
    run(&engine, "DELETE FROM t WHERE id = 1");
    let got = rows(run(&engine, "SELECT qty FROM t ORDER BY id"));
    assert_eq!(got, vec![vec![Value::Int(250)], vec![Value::Int(300)]]);
}

/// Constraint ENFORCEMENT through SQL (driven by the SQL layer over `list_constraints`):
/// duplicate PK rejected, CHECK rejected, UNIQUE rejected, FK parent-existence enforced.
#[test]
fn constraints_enforce_through_sql() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, age INT CHECK (age >= 0))",
    );
    run(&engine, "INSERT INTO users VALUES (1, 'a@x', 30)");

    let err = run_try(&engine, "INSERT INTO users VALUES (1, 'b@x', 40)")
        .expect_err("duplicate primary key must be rejected");
    assert!(
        err.to_string().to_lowercase().contains("duplicate key"),
        "{err}"
    );

    let err = run_try(&engine, "INSERT INTO users VALUES (2, 'a@x', 40)")
        .expect_err("duplicate unique email must be rejected");
    assert!(err.to_string().to_lowercase().contains("uniq"), "{err}");

    let err = run_try(&engine, "INSERT INTO users VALUES (3, 'c@x', -1)")
        .expect_err("check violation must be rejected");
    assert!(err.to_string().to_lowercase().contains("check"), "{err}");

    // FOREIGN KEY: child insert must reference an existing parent.
    run(
        &engine,
        "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users)",
    );
    run(&engine, "INSERT INTO orders VALUES (10, 1)");
    let err = run_try(&engine, "INSERT INTO orders VALUES (11, 999)")
        .expect_err("fk to a missing parent must be rejected");
    assert!(err.to_string().to_lowercase().contains("foreign"), "{err}");
}

/// The whole SQL-visible catalog — tables, rows, PK/UNIQUE/CHECK/FK constraints — survives a
/// crash-reopen of the durable engine, and enforcement still bites afterward.
#[test]
fn sql_catalog_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, age INT CHECK (age >= 0))",
        );
        run(
            &engine,
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users)",
        );
        run(&engine, "INSERT INTO users VALUES (1, 'a@x', 30)");
        run(&engine, "INSERT INTO orders VALUES (10, 1)");
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    let got = rows(run(&engine, "SELECT email FROM users WHERE id = 1"));
    assert_eq!(got, vec![vec![Value::Text("a@x".to_owned())]]);
    assert_eq!(rows(run(&engine, "SELECT id FROM orders")).len(), 1);

    // Every constraint kind still enforces after recovery.
    assert!(run_try(&engine, "INSERT INTO users VALUES (1, 'z@x', 5)").is_err());
    assert!(run_try(&engine, "INSERT INTO users VALUES (2, 'a@x', 5)").is_err());
    assert!(run_try(&engine, "INSERT INTO users VALUES (3, 'c@x', -9)").is_err());
    assert!(run_try(&engine, "INSERT INTO orders VALUES (11, 999)").is_err());
    run(&engine, "INSERT INTO users VALUES (4, 'd@x', 44)");
    assert_eq!(rows(run(&engine, "SELECT id FROM users")).len(), 2);
}

/// The the design sequence `DoD` via real SQL: `SERIAL PRIMARY KEY`, `GENERATED ALWAYS AS IDENTITY`, and
/// bare `CREATE SEQUENCE` all work on the btree engine (each used to fail with
/// `create_sequence is not implemented`), auto-assigned ids are monotonic, and — the critical
/// property — a crash never repeats an id: post-recovery inserts continue past every id handed
/// out before the crash, committed or not.
#[test]
fn serial_identity_and_sequences_work_and_never_repeat_after_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The exact statements QA reported failing.
        run(&engine, "CREATE TABLE d (id SERIAL PRIMARY KEY, v TEXT)");
        run(
            &engine,
            "CREATE TABLE g (id INT GENERATED ALWAYS AS IDENTITY, v TEXT)",
        );
        run(&engine, "CREATE SEQUENCE counter");

        run(&engine, "INSERT INTO d (v) VALUES ('a'), ('b'), ('c')");
        let got = rows(run(&engine, "SELECT id, v FROM d ORDER BY id"));
        assert_eq!(
            got.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
            "SERIAL ids are monotonic from 1"
        );
        run(&engine, "INSERT INTO g (v) VALUES ('x'), ('y')");
        assert_eq!(rows(run(&engine, "SELECT id FROM g")).len(), 2);
        // A duplicate CREATE SEQUENCE is rejected (the catalog is live)...
        assert!(run_try(&engine, "CREATE SEQUENCE counter").is_err());
        // ...and IF NOT EXISTS tolerates it.
        run(&engine, "CREATE SEQUENCE IF NOT EXISTS counter");
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    // New inserts continue past the pre-crash ids — never a duplicate PK.
    run(&engine, "INSERT INTO d (v) VALUES ('after')");
    let got = rows(run(&engine, "SELECT id FROM d ORDER BY id"));
    let ids: Vec<&Value> = got.iter().map(|r| &r[0]).collect();
    assert_eq!(ids.len(), 4);
    let mut sorted = ids.clone();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        4,
        "no duplicate SERIAL id after recovery: {ids:?}"
    );
    assert!(
        matches!(got[3][0], Value::Int(n) if n >= 4),
        "the post-crash id continues past the pre-crash counter"
    );
    // The bare sequence survived the crash: recreating it without IF NOT EXISTS still errors.
    assert!(run_try(&engine, "CREATE SEQUENCE counter").is_err());
    run(&engine, "DROP SEQUENCE counter");
    run(&engine, "CREATE SEQUENCE counter");
}

/// The the design sequence-value `DoD` via real SQL: `nextval`/`currval`/`setval` advance, read, and set a
/// user `CREATE SEQUENCE`; an advance is durable across a crash (a value never repeats); and the
/// side-effecting calls are loud-rejected in a per-row context rather than silently under-advancing.
#[test]
fn nextval_currval_setval_advance_read_and_survive_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE SEQUENCE s");

        // nextval advances from the start (1); currval follows without advancing.
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(1)
        );
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(2)
        );
        assert_eq!(
            rows(run(&engine, "SELECT currval('s')"))[0][0],
            Value::Int(2)
        );
        assert_eq!(
            rows(run(&engine, "SELECT currval('s')"))[0][0],
            Value::Int(2)
        );

        // Two nextvals in one row advance twice.
        let two = rows(run(&engine, "SELECT nextval('s'), nextval('s')"));
        assert_eq!(two[0], vec![Value::Int(3), Value::Int(4)]);

        // setval jumps; the next nextval returns value + increment; setval returns the set value.
        assert_eq!(
            rows(run(&engine, "SELECT setval('s', 100)"))[0][0],
            Value::Int(100)
        );
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(101)
        );

        // currval before any nextval, in a fresh sequence, is an error.
        run(&engine, "CREATE SEQUENCE fresh");
        assert!(run_try(&engine, "SELECT currval('fresh')").is_err());
        // nextval on a missing sequence is an error, not a silent NULL.
        assert!(run_try(&engine, "SELECT nextval('nope')").is_err());

        // An advancing call over a multi-row scan is rejected (never silently under-advanced).
        run(&engine, "CREATE TABLE t (x INT)");
        run(&engine, "INSERT INTO t VALUES (1), (2), (3)");
        assert!(run_try(&engine, "SELECT nextval('s') FROM t").is_err());
        // The rejected query did not advance the sequence.
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(102)
        );

        // nextval inside INSERT ... VALUES advances once per tuple.
        run(&engine, "CREATE SEQUENCE oseq");
        run(&engine, "CREATE TABLE o (id INT, label TEXT)");
        run(
            &engine,
            "INSERT INTO o VALUES (nextval('oseq'), 'a'), (nextval('oseq'), 'b')",
        );
        let got = rows(run(&engine, "SELECT id, label FROM o ORDER BY id"));
        assert_eq!(got[0], vec![Value::Int(1), Value::Text("a".to_owned())]);
        assert_eq!(got[1], vec![Value::Int(2), Value::Text("b".to_owned())]);
    } // crash: no clean shutdown.

    // After recovery, the next value continues past every value handed out before the crash.
    let engine = BtreeEngine::open(&path).unwrap();
    let after = rows(run(&engine, "SELECT nextval('s')"))[0][0].clone();
    assert!(
        matches!(after, Value::Int(n) if n >= 103),
        "post-crash nextval continues past the pre-crash value: {after:?}"
    );
}

/// The the design+QA DDL-evolution `DoD` via real SQL: `ALTER TABLE` (ADD/DROP/RENAME COLUMN, RENAME
/// TABLE) and `CREATE SCHEMA` (both reported by QA as `XX000: ... not implemented by this
/// StorageEngine`) now work on the btree engine, with existing rows migrated correctly and the
/// whole DDL-evolved catalog surviving a crash-reopen.
#[test]
fn alter_table_and_create_schema_work_and_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE t (id INT, name TEXT)");
        run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");

        // ADD COLUMN with existing rows: the SQL layer migrates them; the new column is NULL.
        run(&engine, "ALTER TABLE t ADD COLUMN qty INT");
        let got = rows(run(&engine, "SELECT id, name, qty FROM t ORDER BY id"));
        assert_eq!(
            got,
            vec![
                vec![Value::Int(1), Value::Text("a".to_owned()), Value::Null],
                vec![Value::Int(2), Value::Text("b".to_owned()), Value::Null],
            ]
        );
        run(&engine, "UPDATE t SET qty = id * 10");
        // RENAME COLUMN + a query against the new name.
        run(&engine, "ALTER TABLE t RENAME COLUMN qty TO amount");
        let got = rows(run(&engine, "SELECT amount FROM t ORDER BY id"));
        assert_eq!(got, vec![vec![Value::Int(10)], vec![Value::Int(20)]]);
        // ADD COLUMN with a DEFAULT backfills existing rows — the
        // fix lives in the SQL layer, so it holds on the btree engine too.
        run(&engine, "ALTER TABLE t ADD COLUMN tag INT DEFAULT 5");
        assert_eq!(
            rows(run(&engine, "SELECT tag FROM t ORDER BY id")),
            vec![vec![Value::Int(5)], vec![Value::Int(5)]],
        );
        run(&engine, "ALTER TABLE t DROP COLUMN tag");
        // DROP COLUMN.
        run(&engine, "ALTER TABLE t DROP COLUMN name");
        assert_eq!(
            rows(run(&engine, "SELECT * FROM t ORDER BY id"))[0].len(),
            2
        );
        // RENAME TABLE.
        run(&engine, "ALTER TABLE t RENAME TO items");
        assert_eq!(rows(run(&engine, "SELECT id FROM items")).len(), 2);

        // CREATE SCHEMA + a qualified table.
        run(&engine, "CREATE SCHEMA sales");
        run(&engine, "CREATE TABLE sales.orders (oid INT, total INT)");
        run(&engine, "INSERT INTO sales.orders VALUES (100, 500)");
        let got = rows(run(
            &engine,
            "SELECT total FROM sales.orders WHERE oid = 100",
        ));
        assert_eq!(got, vec![vec![Value::Int(500)]]);
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    // The DDL-evolved catalog survived: renamed table + column, dropped column, qualified table.
    let got = rows(run(&engine, "SELECT id, amount FROM items ORDER BY id"));
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );
    assert!(
        run_try(&engine, "SELECT * FROM t").is_err(),
        "old table name is gone"
    );
    let got = rows(run(&engine, "SELECT total FROM sales.orders"));
    assert_eq!(got, vec![vec![Value::Int(500)]]);
    // Further evolution still works after recovery.
    run(&engine, "ALTER TABLE items ADD COLUMN note TEXT");
    run(&engine, "INSERT INTO items VALUES (3, 30, 'new')");
    assert_eq!(rows(run(&engine, "SELECT id FROM items")).len(), 3);
}
