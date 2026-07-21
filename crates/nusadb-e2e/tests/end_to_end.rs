//! Drive real SQL strings through the whole engine:
//! `parse → analyze → plan → execute` against the production `BtreeEngine`.
//!
//! These are the convergence tests for the two halves of NusaDB: the SQL *surface*
//! (`nusadb-sql`, the engine layer) and the storage *spine* (`nusadb-btree`, the engine layer),
//! meeting over the `StorageEngine` treaty in `nusadb-core`.

// Integration tests live in their own crate, so the `allow-*-in-tests` clippy carve-outs (which
// only cover `#[cfg(test)]` modules) don't apply here. Panic-on-failure *is* the assertion
// mechanism for this harness.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, execute, parse, plan};

/// Adapts the engine's schema lookup to the analyzer's narrower `Catalog` port.
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        // Only SQL-maintained secondary indexes are safe to scan. Constraint-backing (PK/UNIQUE/FK)
        // indexes are enforced by scanning and not maintained on write, so reading through them
        // would miss updated row versions — exclude them, mirroring `secondary_index_targets`.
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
            // A functional/expression key or a partial predicate makes an index unsafe as an
            // equality/range scan candidate (the planner encodes plain-column ascending bounds) —
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
        // Hand the planner the table's ANALYZE stats for cost-based plan selection.
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(None);
        };
        self.0.table_stats(schema.id).map_err(Into::into)
    }
}

/// A `Catalog` that analyzes as a chosen user (superuser or not) for row-level-security tests,
/// reading the RLS flag and policies from the engine under a fresh read snapshot.
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

    fn rls_enabled(&self, name: &str) -> Result<bool, nusadb_sql::Error> {
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let enabled = nusadb_sql::rls_table_enabled(self.engine, txn, name);
        let _ = self.engine.rollback(txn);
        enabled
    }

    fn lookup_policies(&self, name: &str) -> Result<Vec<nusadb_sql::PolicyDef>, nusadb_sql::Error> {
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let policies = nusadb_sql::lookup_policies_for(self.engine, txn, name);
        let _ = self.engine.rollback(txn);
        policies
    }
}

/// Run one SQL statement end-to-end and return its result.
fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

/// Run one statement end-to-end, returning the `Result` (for asserting expected errors).
fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
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

#[test]
fn create_insert_select_round_trip() {
    let engine = BtreeEngine::new();
    assert!(matches!(
        run(&engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)"),
        ExecutionResult::Created(_)
    ));
    assert!(matches!(
        run(&engine, "INSERT INTO users VALUES (1, 'alice')"),
        ExecutionResult::Inserted(1)
    ));
    assert!(matches!(
        run(&engine, "INSERT INTO users VALUES (2, 'bob')"),
        ExecutionResult::Inserted(1)
    ));

    match run(&engine, "SELECT id, name FROM users") {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_owned(), "name".to_owned()]);
            assert_eq!(rows.len(), 2);
            assert!(rows.contains(&vec![Value::Int(1), Value::Text("alice".to_owned())]));
            assert!(rows.contains(&vec![Value::Int(2), Value::Text("bob".to_owned())]));
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn data_modifying_cte_reports_returning_column_metadata() {
    // A `WITH x AS (INSERT … RETURNING …) SELECT …` query must report the body's column
    // names/types, not an empty set (the WithModifying operator delegates its metadata to the body).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    match run(
        &engine,
        "WITH ins AS (INSERT INTO t VALUES (1, 10), (2, 20) RETURNING id, v) \
         SELECT id, v FROM ins ORDER BY id",
    ) {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_owned(), "v".to_owned()]);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0], vec![Value::Int(1), Value::Int(10)]);
            assert_eq!(rows[1], vec![Value::Int(2), Value::Int(20)]);
        },
        other => panic!("expected rows, got {other:?}"),
    }
    // The insert persisted.
    assert_eq!(
        rows(run(&engine, "SELECT count(*) FROM t")),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn nusa_typeof_reports_the_expression_type() {
    // QA minor (pg_typeof idiom, NusaDB-named): `nusa_typeof(expr)` returns the SQL type name of the
    // expression. Folded to a constant at analysis, so it works without a FROM and over column refs.
    let engine = BtreeEngine::new();
    assert_eq!(
        rows(run(&engine, "SELECT nusa_typeof(1)")),
        vec![vec![Value::Text("integer".to_owned())]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT nusa_typeof('x')")),
        vec![vec![Value::Text("text".to_owned())]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT nusa_typeof(true)")),
        vec![vec![Value::Text("boolean".to_owned())]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT nusa_typeof(1.5)")),
        vec![vec![Value::Text("numeric".to_owned())]]
    );
    // Over a real column: the column's analysis-time type is reported. (Integer widths physicalize to
    // `integer` in the analyzer scope — the K4/K6 design keeps declared width in the codec, not the
    // expression scope — so a TEXT column is used here to show a column reference resolves.)
    run(&engine, "CREATE TABLE t (id INT NOT NULL, label TEXT)");
    run(&engine, "INSERT INTO t VALUES (1, 'a')");
    assert_eq!(
        rows(run(&engine, "SELECT nusa_typeof(label) FROM t")),
        vec![vec![Value::Text("text".to_owned())]]
    );
}

#[test]
fn drop_database_backs_up_tables_then_drops_them() {
    // DROP DATABASE empties the single database's tables, but ONLY when the dropped name matches the
    // current database (`nusadb`) — dropping any other name is a no-op (parity with a multi-database server). The matching
    // form first backs up every table — columns and rows — to {database}_{datetime}_{table} as a
    // data-loss safety net; the FORCE keyword (or the FIX alias) drops without a backup. CREATE/ALTER
    // DATABASE stay single-database no-ops.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)");
    run(&engine, "INSERT INTO users VALUES (1, 'alice'), (2, 'bob')");

    assert!(matches!(
        run(&engine, "CREATE DATABASE shop"),
        ExecutionResult::DatabaseCreated
    ));
    assert!(matches!(
        run(&engine, "ALTER DATABASE shop OWNER TO admin"),
        ExecutionResult::DatabaseAltered
    ));

    // Dropping a NON-current database name is a no-op — the tables survive (multi-database parity).
    assert!(matches!(
        run(&engine, "DROP DATABASE shop"),
        ExecutionResult::DatabaseDropped
    ));
    assert_eq!(
        rows(run(&engine, "SELECT id FROM users ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        "DROP DATABASE of a non-current name must NOT touch the data"
    );

    // Dropping the CURRENT database (`nusadb`) backs the table up, then drops the original.
    assert!(matches!(
        run(&engine, "DROP DATABASE nusadb"),
        ExecutionResult::DatabaseDropped
    ));
    assert!(
        run_try(&engine, "SELECT id FROM users").is_err(),
        "the original table must be gone after dropping the current database"
    );

    // A backup table nusadb_{datetime}_users holds the original rows (the data, not just the schema).
    let backup = engine
        .list_tables()
        .unwrap()
        .into_iter()
        .find(|t| t.starts_with("nusadb_") && t.ends_with("_users"))
        .expect("dropping the current database must leave a backup table for `users`");
    assert_eq!(
        rows(run(
            &engine,
            &format!("SELECT id, name FROM {backup} ORDER BY id")
        )),
        vec![
            vec![Value::Int(1), Value::Text("alice".to_owned())],
            vec![Value::Int(2), Value::Text("bob".to_owned())],
        ]
    );

    // `DROP DATABASE nusadb FORCE` (canonical no-backup form) drops permanently, without a backup.
    run(&engine, "CREATE TABLE again (id INT NOT NULL)");
    run(&engine, "INSERT INTO again VALUES (7)");
    assert!(matches!(
        run(&engine, "DROP DATABASE nusadb FORCE"),
        ExecutionResult::DatabaseDropped
    ));
    assert!(
        run_try(&engine, "SELECT id FROM again").is_err(),
        "DROP DATABASE ... FORCE must drop the table"
    );
    assert!(
        !engine
            .list_tables()
            .unwrap()
            .iter()
            .any(|t| t.ends_with("_again")),
        "DROP DATABASE ... FORCE must NOT create a backup"
    );

    // The `FIX DROP DATABASE name` alias is also accepted, and also skips the backup.
    run(&engine, "CREATE TABLE once_more (id INT NOT NULL)");
    assert!(matches!(
        run(&engine, "FIX DROP DATABASE nusadb"),
        ExecutionResult::DatabaseDropped
    ));
    assert!(run_try(&engine, "SELECT id FROM once_more").is_err());
    assert!(
        !engine
            .list_tables()
            .unwrap()
            .iter()
            .any(|t| t.ends_with("_once_more")),
        "the FIX alias must also skip the backup"
    );
}

#[test]
fn insert_from_select_copies_filtered_rows() {
    // INSERT ... SELECT inserts the subquery's rows; the subquery sees the pre-INSERT state.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE src (id INT NOT NULL, n INT)");
    run(&engine, "CREATE TABLE dst (id INT NOT NULL, n INT)");
    run(&engine, "INSERT INTO src VALUES (1, 10), (2, 20), (3, 30)");

    // Column subset + a WHERE filter on the source.
    assert!(matches!(
        run(
            &engine,
            "INSERT INTO dst (id, n) SELECT id, n FROM src WHERE n >= 20"
        ),
        ExecutionResult::Inserted(2),
    ));
    assert_eq!(
        rows(run(&engine, "SELECT id, n FROM dst ORDER BY id")),
        vec![
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ],
    );

    // RETURNING works on INSERT ... SELECT too.
    match run(
        &engine,
        "INSERT INTO dst SELECT id + 100, n FROM src WHERE id = 1 RETURNING id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(rows, vec![vec![Value::Int(101)]]);
        },
        other => panic!("expected RETURNING rows, got {other:?}"),
    }
}

#[test]
fn insert_returning_projects_the_inserted_rows() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)");

    // RETURNING a subset + an alias.
    match run(
        &engine,
        "INSERT INTO users (id, name) VALUES (1, 'alice') RETURNING id, name AS who",
    ) {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_owned(), "who".to_owned()]);
            assert_eq!(
                rows,
                vec![vec![Value::Int(1), Value::Text("alice".to_owned())]]
            );
        },
        other => panic!("expected RETURNING rows, got {other:?}"),
    }

    // RETURNING * over a multi-row insert returns every inserted row, all columns.
    match run(
        &engine,
        "INSERT INTO users (id, name) VALUES (2, 'bob'), (3, 'carol') RETURNING *",
    ) {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_owned(), "name".to_owned()]);
            assert_eq!(rows.len(), 2);
            assert!(rows.contains(&vec![Value::Int(2), Value::Text("bob".to_owned())]));
            assert!(rows.contains(&vec![Value::Int(3), Value::Text("carol".to_owned())]));
        },
        other => panic!("expected RETURNING rows, got {other:?}"),
    }

    // The rows really landed.
    assert_eq!(rows(run(&engine, "SELECT id FROM users")).len(), 3);

    // Plain INSERT (no RETURNING) still reports a count, not rows.
    assert!(matches!(
        run(&engine, "INSERT INTO users VALUES (4, 'dave')"),
        ExecutionResult::Inserted(1)
    ));
}

#[test]
fn insert_returning_reflects_the_stored_coerced_value_not_the_input() {
    // `INSERT ... RETURNING` must yield the value as *stored* (coerced to the column type),
    // not the un-coerced input literal. A NUMERIC(12,3) rescales 123.4565 -> 123.457, so RETURNING
    // must show 123.457 — exactly what a later SELECT reads back.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE m (id INT, n NUMERIC(12,3))");

    let returned = match run(
        &engine,
        "INSERT INTO m (id, n) VALUES (1, 123.4565) RETURNING n",
    ) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected RETURNING rows, got {other:?}"),
    };
    let stored = rows(run(&engine, "SELECT n FROM m WHERE id = 1"));

    // The returned value is the rescaled, persisted value — and equals what SELECT reads.
    assert_eq!(
        returned,
        vec![vec![Value::Numeric(
            nusadb_sql::numeric::Decimal::parse("123.457").unwrap()
        )]]
    );
    assert_eq!(returned, stored);

    // The value is stored rescaled, so an equality predicate on the rescaled value finds the row
    // (the un-coerced 123.4565 does not) — closes the QA differential "WHERE n=123.457 misses" claim.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM m WHERE n = 123.457")).len(),
        1
    );
    assert_eq!(
        rows(run(&engine, "SELECT id FROM m WHERE n = 123.4565")).len(),
        0
    );
}

#[test]
fn update_and_delete_returning_project_affected_rows() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)");
    run(&engine, "INSERT INTO users VALUES (1, 'alice'), (2, 'bob')");

    // UPDATE RETURNING yields the *post-update* values.
    match run(
        &engine,
        "UPDATE users SET name = 'BOB' WHERE id = 2 RETURNING id, name",
    ) {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["id".to_owned(), "name".to_owned()]);
            assert_eq!(
                rows,
                vec![vec![Value::Int(2), Value::Text("BOB".to_owned())]]
            );
        },
        other => panic!("expected UPDATE RETURNING rows, got {other:?}"),
    }

    // DELETE RETURNING yields the *pre-delete* values.
    match run(
        &engine,
        "DELETE FROM users WHERE id = 1 RETURNING name AS gone",
    ) {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["gone".to_owned()]);
            assert_eq!(rows, vec![vec![Value::Text("alice".to_owned())]]);
        },
        other => panic!("expected DELETE RETURNING rows, got {other:?}"),
    }

    // Plain UPDATE/DELETE still report counts.
    assert!(matches!(
        run(&engine, "UPDATE users SET name = 'x' WHERE id = 2"),
        ExecutionResult::Updated(1)
    ));
    assert_eq!(rows(run(&engine, "SELECT id FROM users")).len(), 1);
}

#[test]
fn create_and_drop_schema_end_to_end() {
    let engine = BtreeEngine::new();
    assert!(matches!(
        run(&engine, "CREATE SCHEMA app"),
        ExecutionResult::SchemaCreated
    ));
    // IF NOT EXISTS makes a pre-existing schema a no-op…
    assert!(matches!(
        run(&engine, "CREATE SCHEMA IF NOT EXISTS app"),
        ExecutionResult::SchemaCreated
    ));
    // …but a plain duplicate is an error.
    assert!(run_try(&engine, "CREATE SCHEMA app").is_err());

    assert!(matches!(
        run(&engine, "DROP SCHEMA app"),
        ExecutionResult::SchemaDropped
    ));
    // DROP of a missing schema errors without IF EXISTS, no-ops with it.
    assert!(run_try(&engine, "DROP SCHEMA app").is_err());
    assert!(matches!(
        run(&engine, "DROP SCHEMA IF EXISTS app"),
        ExecutionResult::SchemaDropped
    ));
}

#[test]
fn create_and_drop_sequence_end_to_end() {
    let engine = BtreeEngine::new();
    assert!(matches!(
        run(&engine, "CREATE SEQUENCE s INCREMENT BY 2 START WITH 10"),
        ExecutionResult::SequenceCreated
    ));
    // IF NOT EXISTS no-op; plain duplicate errors.
    assert!(matches!(
        run(&engine, "CREATE SEQUENCE IF NOT EXISTS s"),
        ExecutionResult::SequenceCreated
    ));
    assert!(run_try(&engine, "CREATE SEQUENCE s").is_err());

    assert!(matches!(
        run(&engine, "DROP SEQUENCE s"),
        ExecutionResult::SequenceDropped
    ));
    // DROP missing errors without IF EXISTS, no-ops with it.
    assert!(run_try(&engine, "DROP SEQUENCE s").is_err());
    assert!(matches!(
        run(&engine, "DROP SEQUENCE IF EXISTS s"),
        ExecutionResult::SequenceDropped
    ));
}

#[test]
fn create_and_drop_index_end_to_end() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");

    assert!(matches!(
        run(&engine, "CREATE INDEX idx_t_name ON t (name)"),
        ExecutionResult::IndexCreated
    ));
    // IF NOT EXISTS no-op; plain duplicate errors.
    assert!(matches!(
        run(&engine, "CREATE INDEX IF NOT EXISTS idx_t_name ON t (name)"),
        ExecutionResult::IndexCreated
    ));
    assert!(run_try(&engine, "CREATE INDEX idx_t_name ON t (name)").is_err());
    // A unique index on a known column also resolves.
    assert!(matches!(
        run(&engine, "CREATE UNIQUE INDEX idx_t_id ON t (id)"),
        ExecutionResult::IndexCreated
    ));
    // Unknown table / column are rejected by the analyzer.
    assert!(run_try(&engine, "CREATE INDEX bad ON ghost (x)").is_err());
    assert!(run_try(&engine, "CREATE INDEX bad ON t (nope)").is_err());

    assert!(matches!(
        run(&engine, "DROP INDEX idx_t_name"),
        ExecutionResult::IndexDropped
    ));
    // DROP missing errors without IF EXISTS, no-ops with it.
    assert!(run_try(&engine, "DROP INDEX idx_t_name").is_err());
    assert!(matches!(
        run(&engine, "DROP INDEX IF EXISTS idx_t_name"),
        ExecutionResult::IndexDropped
    ));
}

#[test]
fn functional_partial_and_desc_indexes_end_to_end() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, s TEXT, a INT, active BOOL)",
    );

    // DESC / NULLS annotations are rejected loudly: only ascending indexes are built and ordered
    // index scans are not implemented, so silently building an ascending index for `(a DESC)`
    // would be a lossy trap. A plain `(a)` index is the alternative.
    assert!(run_try(&engine, "CREATE INDEX idx_a_desc ON t (a DESC)").is_err());
    assert!(run_try(&engine, "CREATE INDEX idx_a_nulls ON t (a NULLS FIRST)").is_err());
    // A user-defined SQL function in a key is rejected at CREATE, not silently unmaintainable: the
    // write-path re-analysis has no function catalog, so accepting it would produce a UNIQUE index
    // that enforces nothing.
    run(
        &engine,
        "CREATE FUNCTION dbl(x INT) RETURNS INT AS $$ SELECT $1 * 2 $$",
    );
    assert!(
        run_try(&engine, "CREATE UNIQUE INDEX idx_udf ON t (dbl(a))").is_err(),
        "a functional index over a SQL UDF must be rejected at CREATE, not silently unmaintainable"
    );

    // Functional UNIQUE index: uniqueness is enforced on the COMPUTED key (lower(s)), so two rows
    // whose `s` differ only in case collide — proving the expression key is evaluated + maintained.
    run(&engine, "CREATE UNIQUE INDEX idx_lower_s ON t (lower(s))");
    run(&engine, "INSERT INTO t VALUES (1, 'Alice', 1, TRUE)");
    assert!(
        run_try(&engine, "INSERT INTO t VALUES (2, 'ALICE', 2, TRUE)").is_err(),
        "a UNIQUE functional index on lower(s) must reject 'ALICE' after 'Alice'"
    );
    // A genuinely different lowercased value is accepted.
    assert!(matches!(
        run_try(&engine, "INSERT INTO t VALUES (3, 'bob', 3, FALSE)"),
        Ok(ExecutionResult::Inserted(1))
    ));

    // Expression index (non-unique) on an arithmetic key: creating and maintaining it must not
    // break inserts or query results (it falls back to a seq scan for reads).
    run(&engine, "CREATE INDEX idx_expr ON t (((a + 1)))");
    assert!(matches!(
        run_try(&engine, "INSERT INTO t VALUES (4, 'carol', 10, TRUE)"),
        Ok(ExecutionResult::Inserted(1))
    ));
    // Results stay correct.
    let got = rows(run(&engine, "SELECT id FROM t WHERE a + 1 = 11"));
    assert_eq!(got, vec![vec![Value::Int(4)]]);

    // Partial UNIQUE index: only rows WHERE active are indexed, so uniqueness of `a` holds only
    // among active rows. Two inactive rows with a=5 are fine; a second ACTIVE a=7 collides.
    run(
        &engine,
        "CREATE TABLE p (id INT NOT NULL, a INT, active BOOL)",
    );
    run(
        &engine,
        "CREATE UNIQUE INDEX idx_p_active ON p (a) WHERE active",
    );
    run(&engine, "INSERT INTO p VALUES (1, 5, FALSE)");
    assert!(
        matches!(
            run_try(&engine, "INSERT INTO p VALUES (2, 5, FALSE)"),
            Ok(ExecutionResult::Inserted(1))
        ),
        "two INACTIVE rows with a=5 must both be allowed (neither is indexed)"
    );
    run(&engine, "INSERT INTO p VALUES (3, 7, TRUE)");
    assert!(
        run_try(&engine, "INSERT INTO p VALUES (4, 7, TRUE)").is_err(),
        "two ACTIVE rows with a=7 must collide on the partial UNIQUE index"
    );
    // An active a=5 is still allowed (no active a=5 yet), proving the predicate gates indexing.
    assert!(matches!(
        run_try(&engine, "INSERT INTO p VALUES (5, 5, TRUE)"),
        Ok(ExecutionResult::Inserted(1))
    ));
    // CRITICAL — a row UPDATEd OUT of the partial predicate must leave the index, so a genuinely
    // new active row with the same key is accepted, not falsely rejected (
    // ENTRY). Row 3 (a=7, active) is the only active a=7; move it out (active=FALSE), then a new
    // active a=7 must succeed.
    run(&engine, "UPDATE p SET active = FALSE WHERE id = 3");
    assert!(
        matches!(
            run_try(&engine, "INSERT INTO p VALUES (6, 7, TRUE)"),
            Ok(ExecutionResult::Inserted(1))
        ),
        "after the only active a=7 row leaves the predicate, a new active a=7 must be accepted"
    );
    // And the partial UNIQUE is still enforced among the (now current) active rows: a second
    // active a=7 collides.
    assert!(
        run_try(&engine, "INSERT INTO p VALUES (7, 7, TRUE)").is_err(),
        "a second active a=7 must still collide"
    );
    // CRITICAL scan-exclusion: a partial index on a plain column must NOT be used to answer a query
    // that does not filter on its predicate — it holds only the active rows, so using it would drop
    // the inactive a=5 rows. `WHERE a = 5` must return ALL three rows (ids 1,2 inactive + 5 active),
    // via a sequential scan.
    let mut got: Vec<i64> = rows(run(&engine, "SELECT id FROM p WHERE a = 5"))
        .into_iter()
        .map(|r| match r.first() {
            Some(Value::Int(n)) => *n,
            other => panic!("expected int id, got {other:?}"),
        })
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![1, 2, 5],
        "a partial index must not hide the non-indexed (inactive) a=5 rows from an unfiltered query"
    );

    // Backfill: a functional UNIQUE index created on a POPULATED table indexes existing rows, so a
    // later colliding insert is rejected.
    run(&engine, "CREATE TABLE b (id INT NOT NULL, s TEXT)");
    run(&engine, "INSERT INTO b VALUES (1, 'Xyz')");
    run(&engine, "CREATE UNIQUE INDEX idx_b_lower ON b (lower(s))");
    assert!(
        run_try(&engine, "INSERT INTO b VALUES (2, 'XYZ')").is_err(),
        "backfill must index the existing 'Xyz' so 'XYZ' collides"
    );
}

#[test]
fn set_operations_end_to_end() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (x INT NOT NULL)");
    run(&engine, "CREATE TABLE b (x INT NOT NULL)");
    run(&engine, "INSERT INTO a VALUES (1), (2), (2), (3)");
    run(&engine, "INSERT INTO b VALUES (2), (3), (4)");

    let ints = |r: ExecutionResult| -> Vec<i64> {
        rows(r)
            .into_iter()
            .map(|row| match row.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect()
    };

    // UNION de-dups; ORDER BY binds to the combined result.
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a UNION SELECT x FROM b ORDER BY x"
        )),
        vec![1, 2, 3, 4]
    );
    // UNION ALL keeps duplicates (4 + 3 rows).
    assert_eq!(
        rows(run(&engine, "SELECT x FROM a UNION ALL SELECT x FROM b")).len(),
        7
    );
    // INTERSECT / EXCEPT (distinct).
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x"
        )),
        vec![2, 3]
    );
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY x"
        )),
        vec![1]
    );
    // Multiset (ALL) forms: a={1,2,2,3}, b={2,3,4}.
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a INTERSECT ALL SELECT x FROM b ORDER BY x"
        )),
        vec![2, 3]
    );
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a EXCEPT ALL SELECT x FROM b ORDER BY x"
        )),
        vec![1, 2] // one 2 removed, one 3 removed
    );
    // LIMIT on the combined result.
    assert_eq!(
        ints(run(
            &engine,
            "SELECT x FROM a UNION SELECT x FROM b ORDER BY x LIMIT 2"
        )),
        vec![1, 2]
    );
}

#[test]
fn order_by_position_and_alias_sort_correctly() {
    // ORDER BY <position> and ORDER BY <alias> must drive the real sort, not no-op.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (3, 10), (1, 30), (2, 20)");

    let ints_col = |r: ExecutionResult, col: usize| -> Vec<i64> {
        rows(r)
            .into_iter()
            .map(|row| match row.get(col) {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect()
    };

    // ORDER BY 2 → sort by the 2nd output column `b` ascending: rows ordered 10, 20, 30.
    assert_eq!(
        ints_col(run(&engine, "SELECT a, b FROM t ORDER BY 2"), 1),
        vec![10, 20, 30]
    );
    // ORDER BY 1 DESC → sort by `a` descending: 3, 2, 1.
    assert_eq!(
        ints_col(run(&engine, "SELECT a, b FROM t ORDER BY 1 DESC"), 0),
        vec![3, 2, 1]
    );
    // ORDER BY <alias> → sort by the aliased computed column.
    assert_eq!(
        ints_col(run(&engine, "SELECT a, b AS bee FROM t ORDER BY bee"), 1),
        vec![10, 20, 30]
    );
    // Out-of-range position is rejected, not silently ignored.
    assert!(run_try(&engine, "SELECT a, b FROM t ORDER BY 5").is_err());
}

#[test]
fn b135_order_by_nulls_first_last_places_nulls() {
    // Explicit NULLS FIRST/LAST pins NULL placement regardless of ASC/DESC. A secondary
    // `id` key makes the NULL-row tie-break deterministic (independent of scan order).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 10), (2, NULL), (3, 5), (4, NULL), (5, 20)",
    );

    let ids = |sql: &str| -> Vec<i64> {
        rows(run(&engine, sql))
            .into_iter()
            .map(|row| match row.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int id, got {other:?}"),
            })
            .collect()
    };

    // ASC NULLS FIRST: NULLs (ids 2,4) lead, then v ascending 5,10,20.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY v ASC NULLS FIRST, id"),
        vec![2, 4, 3, 1, 5]
    );
    // ASC NULLS LAST (also the ASC default): v ascending, then NULLs.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY v ASC NULLS LAST, id"),
        vec![3, 1, 5, 2, 4]
    );
    // DESC defaults NULLs first; NULLS LAST overrides → v descending, then NULLs.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY v DESC NULLS LAST, id"),
        vec![5, 1, 3, 2, 4]
    );
    // DESC NULLS FIRST: NULLs lead, then v descending.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY v DESC NULLS FIRST, id"),
        vec![2, 4, 5, 1, 3]
    );
}

#[test]
fn b138_like_escape_matches_literal_wildcards() {
    // ESCAPE 'c' makes `c%`/`c_` match a literal `%`/`_` instead of a wildcard.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, '100%'), (2, '1000'), (3, 'a_b'), (4, 'axb')",
    );

    let ids = |sql: &str| -> Vec<i64> {
        rows(run(&engine, sql))
            .into_iter()
            .map(|row| match row.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected int id, got {other:?}"),
            })
            .collect()
    };

    // `100!%` ESCAPE '!' → literal '%': matches '100%' (id 1), not '1000'.
    assert_eq!(
        ids("SELECT id FROM t WHERE s LIKE '100!%' ESCAPE '!' ORDER BY id"),
        vec![1]
    );
    // `a!_b` ESCAPE '!' → literal '_': matches 'a_b' (id 3), not 'axb'.
    assert_eq!(
        ids("SELECT id FROM t WHERE s LIKE 'a!_b' ESCAPE '!' ORDER BY id"),
        vec![3]
    );
    // Without ESCAPE, `_` is a wildcard → matches both 'a_b' and 'axb'.
    assert_eq!(
        ids("SELECT id FROM t WHERE s LIKE 'a_b' ORDER BY id"),
        vec![3, 4]
    );
}

#[test]
fn b145_array_literal_and_subscript_end_to_end() {
    // ARRAY[...] constructor and 1-based subscript.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, a INT, b INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10, 20)");

    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // Constructor from columns + a literal.
    assert_eq!(
        one("SELECT ARRAY[a, b, 30] FROM t"),
        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
    );
    // A NULL element is preserved in the array.
    assert_eq!(
        one("SELECT ARRAY[1, NULL, 3] FROM t"),
        Value::Array(vec![Value::Int(1), Value::Null, Value::Int(3)])
    );
    // 1-based subscript.
    assert_eq!(one("SELECT ARRAY[10, 20, 30][2] FROM t"), Value::Int(20));
    // Out-of-range and zero/negative indexes yield NULL (not an error).
    assert_eq!(one("SELECT ARRAY[10, 20, 30][9] FROM t"), Value::Null);
    assert_eq!(one("SELECT ARRAY[10, 20, 30][0] FROM t"), Value::Null);
    // A large negative index (well below 1) is NULL, not an error.
    assert_eq!(one("SELECT ARRAY[10, 20, 30][-100] FROM t"), Value::Null);
    // Subscript of a column-built array.
    assert_eq!(one("SELECT ARRAY[a, b][1] FROM t"), Value::Int(10));
}

#[test]
fn quantified_any_all_over_a_runtime_array_column() {
    // `x <op> ANY/ALL(arr)` where `arr` is a runtime array value (a column). Previously only an
    // `ARRAY[...]` literal or a subquery operand was accepted.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, tags INT[])");
    run(
        &engine,
        "INSERT INTO t VALUES (1, ARRAY[10, 20, 30]), (2, ARRAY[5]), (3, ARRAY[]::INT[]), \
         (4, NULL), (5, ARRAY[10, NULL])",
    );
    let ids = |sql: &str| -> Vec<i64> {
        rows(run(&engine, sql))
            .into_iter()
            .map(|r| match r.into_iter().next() {
                Some(Value::Int(n)) => n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect()
    };

    // `= ANY` is membership: row 1 contains 20; row 5's NULL element leaves 20 = ANY undecided (NULL,
    // excluded by WHERE); the others lack it.
    assert_eq!(
        ids("SELECT id FROM t WHERE 20 = ANY(tags) ORDER BY id"),
        vec![1]
    );
    // A value present alongside a NULL element still matches (the TRUE decides before the NULL).
    assert_eq!(
        ids("SELECT id FROM t WHERE 10 = ANY(tags) ORDER BY id"),
        vec![1, 5]
    );
    // `> ALL`: 40 exceeds every element of {10,20,30} and {5}; the empty array is vacuously TRUE; the
    // NULL array yields NULL; row 5's NULL element makes it undecided.
    assert_eq!(
        ids("SELECT id FROM t WHERE 40 > ALL(tags) ORDER BY id"),
        vec![1, 2, 3]
    );
    // `< ANY`: 6 is below the 10 element (rows 1 and 5 — the NULL element does not matter once the
    // 10 makes ANY true), but not below {5} (row 2).
    assert_eq!(
        ids("SELECT id FROM t WHERE 6 < ANY(tags) ORDER BY id"),
        vec![1, 5]
    );
    // Empty-array identity: `= ALL(empty)` is TRUE (row 3 only); `= ANY(empty)` would be FALSE.
    assert_eq!(
        ids("SELECT id FROM t WHERE 99 = ALL(tags) ORDER BY id"),
        vec![3]
    );
}

#[test]
fn b459_array_ops_and_unnest_end_to_end() {
    // Cardinality, `||` concat, ARRAY_AGG, and UNNEST against the real engine.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // cardinality + concatenation (merge and element append).
    assert_eq!(one("SELECT cardinality(ARRAY[10, 20, 30])"), Value::Int(3));
    assert_eq!(
        one("SELECT ARRAY[1, 2] || ARRAY[3]"),
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
    assert_eq!(
        one("SELECT ARRAY[1, 2] || 3"),
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );

    // ARRAY_AGG collects the column into an array in scan order.
    assert_eq!(
        one("SELECT array_agg(v) FROM t"),
        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
    );

    // UNNEST expands an array into rows, repeating the scalar column.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id, unnest(ARRAY[7, 8]) FROM t WHERE id = 1",
        )),
        vec![
            vec![Value::Int(1), Value::Int(7)],
            vec![Value::Int(1), Value::Int(8)],
        ]
    );
}

#[test]
fn array_agg_and_string_agg_order_by() {
    // ORDER BY inside ARRAY_AGG / STRING_AGG reorders the collected values.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, v INT, name TEXT)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 30, 'c'), (2, 10, 'a'), (3, 20, 'b')",
    );
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // ARRAY_AGG ascending / descending by its own value.
    assert_eq!(
        one("SELECT array_agg(v ORDER BY v) FROM t"),
        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
    );
    assert_eq!(
        one("SELECT array_agg(v ORDER BY v DESC) FROM t"),
        Value::Array(vec![Value::Int(30), Value::Int(20), Value::Int(10)])
    );
    // ORDER BY a different column than the aggregated one (name a,b,c → v 10,20,30).
    assert_eq!(
        one("SELECT array_agg(v ORDER BY name) FROM t"),
        Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
    );
    // STRING_AGG honours its ORDER BY too.
    assert_eq!(
        one("SELECT string_agg(name, ',' ORDER BY name DESC) FROM t"),
        Value::Text("c,b,a".to_owned())
    );
    // Without ORDER BY the collection stays in scan order (unchanged behaviour).
    assert_eq!(
        one("SELECT array_agg(v) FROM t"),
        Value::Array(vec![Value::Int(30), Value::Int(10), Value::Int(20)])
    );
}

#[test]
fn create_type_enum_and_use_as_a_column() {
    // B-ENUM — CREATE TYPE ... AS ENUM registers a user-defined type usable as a column type
    // (stored as TEXT); DROP TYPE removes it; an unknown type name on a column is a loud error.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TYPE mood_enum AS ENUM ('sad', 'ok', 'happy')",
    );

    // Use the enum as a column type, insert, and read back.
    run(&engine, "CREATE TABLE t (id INT NOT NULL, m mood_enum)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'happy'), (2, 'sad'), (3, NULL)",
    );
    assert_eq!(
        rows(run(&engine, "SELECT m FROM t ORDER BY id")),
        vec![
            vec![Value::Text("happy".to_owned())],
            vec![Value::Text("sad".to_owned())],
            vec![Value::Null],
        ]
    );

    // Re-declaring an existing type, and empty/duplicate labels, are rejected.
    assert!(run_try(&engine, "CREATE TYPE mood_enum AS ENUM ('x')").is_err());
    assert!(run_try(&engine, "CREATE TYPE e1 AS ENUM ('a', 'a')").is_err());

    // An unknown user-defined type on a column is a loud error (not silently text).
    assert!(run_try(&engine, "CREATE TABLE bad (x nonexistent_enum)").is_err());

    // DROP TYPE removes it; afterwards a new table can no longer use it; IF EXISTS is a no-op.
    run(&engine, "DROP TYPE mood_enum");
    assert!(run_try(&engine, "CREATE TABLE u (m mood_enum)").is_err());
    run(&engine, "DROP TYPE IF EXISTS mood_enum");
}

#[test]
fn added_scalar_functions_justify_days_encode_decode_regexp_matches() {
    // B-fn — newly added scalar functions: justify_days, encode/decode (bytea), regexp_matches.
    let engine = BtreeEngine::new();
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // justify_days: 35 days -> 1 month + 5 days.
    assert_eq!(
        one("SELECT justify_days(INTERVAL '35 days')::text"),
        Value::Text("1 mon 5 days".to_owned())
    );
    // encode/decode round-trip through hex and escape.
    assert_eq!(
        one(r"SELECT encode('\xdeadbeef'::bytea, 'hex')"),
        Value::Text("deadbeef".to_owned())
    );
    assert_eq!(
        one("SELECT decode('deadbeef', 'hex')"),
        Value::Bytes(vec![0xde, 0xad, 0xbe, 0xef])
    );
    assert_eq!(
        one(r"SELECT encode(decode('414243', 'hex'), 'escape')"),
        Value::Text("ABC".to_owned())
    );
    // regexp_matches is set-returning; without the `g` flag it yields a single row holding the first
    // match's capture groups as a TEXT[].
    assert_eq!(
        one("SELECT regexp_matches('abc123', '([a-z]+)([0-9]+)')"),
        Value::Array(vec![
            Value::Text("abc".to_owned()),
            Value::Text("123".to_owned()),
        ])
    );
    // make_time accepts fractional seconds (its seconds field is double precision); the time renders
    // with full microsecond precision.
    assert_eq!(
        one("SELECT make_time(8, 15, 23.5)::text"),
        Value::Text("08:15:23.500000".to_owned())
    );
}

#[test]
fn regexp_matches_is_set_returning_with_the_global_flag() {
    // regexp_matches is a set-returning function: with the `g` flag it yields one row per match (each
    // a TEXT[] of the capture groups), not a single first-match row. (QA finding: the `g` form was
    // silently returning only the first match.)
    let engine = BtreeEngine::new();
    let arrays = |sql: &str| -> Vec<Value> {
        rows(run(&engine, sql))
            .into_iter()
            .map(|mut r| r.swap_remove(0))
            .collect()
    };

    // Three matches with the `g` flag -> three rows.
    assert_eq!(
        arrays("SELECT regexp_matches('a1b2c3', '([a-z])([0-9])', 'g')"),
        vec![
            Value::Array(vec![
                Value::Text("a".to_owned()),
                Value::Text("1".to_owned())
            ]),
            Value::Array(vec![
                Value::Text("b".to_owned()),
                Value::Text("2".to_owned())
            ]),
            Value::Array(vec![
                Value::Text("c".to_owned()),
                Value::Text("3".to_owned())
            ]),
        ]
    );
    // Without `g`, only the first match -> one row.
    assert_eq!(
        arrays("SELECT regexp_matches('a1b2c3', '([a-z])([0-9])')"),
        vec![Value::Array(vec![
            Value::Text("a".to_owned()),
            Value::Text("1".to_owned()),
        ])]
    );
    // No match -> no rows (the set-returning empty set, not a NULL row).
    assert_eq!(
        arrays("SELECT regexp_matches('xyz', '([0-9])', 'g')"),
        Vec::<Value>::new()
    );
}

#[test]
fn numeric_precision_out_of_range_is_rejected_not_clamped() {
    // K5 — a NUMERIC precision/scale beyond the catalog's u8 limit is rejected, not silently clamped
    // to 255 (a clamp would round values to the wrong scale).
    let engine = BtreeEngine::new();
    assert!(run_try(&engine, "CREATE TABLE a (n NUMERIC(300, 5))").is_err());
    assert!(run_try(&engine, "CREATE TABLE b (n NUMERIC(10, 300))").is_err());
    // In-range precision/scale still work.
    run(&engine, "CREATE TABLE c (n NUMERIC(10, 2))");
    run(&engine, "INSERT INTO c VALUES (123.456)");
    assert_eq!(
        rows(run(&engine, "SELECT n FROM c")),
        vec![vec![Value::Numeric(
            nusadb_sql::numeric::Decimal::parse("123.46").expect("decimal")
        )]]
    );
}

#[test]
fn bitwise_xor_and_power_operators_match_pg() {
    // /: `#` is the reference engine's integer XOR (the generic tokenizer's Sharp is parsed
    // by the dialect hook), and `^` is the reference engine's exponentiation — a double-precision result, associating
    // left-to-right like the reference engine — lowered to the `power()` built-in.
    let engine = BtreeEngine::new();
    let row = rows(run(
        &engine,
        "SELECT 12 & 10, 12 | 10, 12 # 10, 2 ^ 10, 2 ^ 3 ^ 2",
    ))
    .swap_remove(0);
    assert_eq!(
        row,
        vec![
            Value::Int(8),
            Value::Int(14),
            Value::Int(6),
            Value::Float(1024.0),
            // The reference engine: `^` associates LEFT to right (unlike common math), so (2^3)^2 = 64.
            Value::Float(64.0),
        ]
    );
}

#[test]
fn numeric_past_38_digits_is_a_loud_out_of_range() {
    // NUMERIC carries at most 38 significant digits (i128 mantissa) — the reference engine's
    // arbitrary precision would return the 40-digit product. The cap is a documented engine
    // limit surfaced with the standard 22003 code, never a silent rounding.
    let engine = BtreeEngine::new();
    let err = run_try(
        &engine,
        "SELECT 99999999999999999999 * 99999999999999999999",
    )
    .expect_err("a 40-digit exact product exceeds the NUMERIC capacity");
    assert_eq!(err.sqlstate(), "22003", "got: {err}");
    assert!(
        err.to_string().contains("38-digit"),
        "the limit must be named: {err}"
    );
}

#[test]
fn timetz_literal_insert_cast_and_round_trip() {
    // K3 — TIME WITH TIME ZONE was DDL-only: a text literal would not insert and `timetz::text`
    // was unsupported. A timetz value now stores from a text literal and casts both directions.
    let engine = BtreeEngine::new();
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // `'HH:MM:SS+OFFSET'::timetz` keeps its zone (P-TIMETZ, faithful to the reference engine) and renders back with
    // the offset it was entered with.
    assert_eq!(
        one("SELECT '13:45:30+07'::timetz::text"),
        Value::Text("13:45:30+07".to_owned())
    );
    // A bare `Z`/no-offset value is UTC and renders `+00`.
    assert_eq!(
        one("SELECT '06:45:30Z'::timetz::text"),
        Value::Text("06:45:30+00".to_owned())
    );

    // Insert a text literal into a TIMETZ column and read it back: the stored value round-trips
    // with its zone intact.
    run(&engine, "CREATE TABLE t (id INT NOT NULL, ttz TIMETZ)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, '13:45:30+07'), (2, NULL)",
    );
    assert_eq!(
        rows(run(&engine, "SELECT ttz::text FROM t ORDER BY id")),
        vec![
            vec![Value::Text("13:45:30+07".to_owned())],
            vec![Value::Null]
        ]
    );
}

#[test]
fn create_table_accepts_extended_type_aliases() {
    // B-types — types NusaDB does not model natively (currency, object-id, network, bit-string,
    // geometric, range, full-text, XML) are accepted on a column and stored as a base type, so a
    // wide schema loads. A genuinely unknown type name is still rejected.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE wide (
            id            SERIAL PRIMARY KEY,
            col_smallint  SMALLINT,
            col_bigint    BIGINT,
            col_real      REAL,
            col_double    DOUBLE PRECISION,
            col_money     MONEY,
            col_char      CHAR(10),
            col_varchar   VARCHAR(255),
            col_bytea     BYTEA,
            col_timestamp TIMESTAMP,
            col_timetz    TIMETZ,
            col_interval  INTERVAL,
            col_uuid      UUID,
            col_jsonb     JSONB,
            col_int_arr   INTEGER[],
            col_inet      INET,
            col_cidr      CIDR,
            col_macaddr   MACADDR,
            col_bit       BIT(8),
            col_varbit    VARBIT(16),
            col_point     POINT,
            col_polygon   POLYGON,
            col_tsvector  TSVECTOR,
            col_xml       XML,
            col_int4range INT4RANGE,
            col_oid       OID
        )",
    );
    let dec = |s: &str| Value::Numeric(nusadb_sql::numeric::Decimal::parse(s).expect("decimal"));
    // The schema loaded; a money value stores as its decimal and a network/geometric value as text.
    run(
        &engine,
        r"INSERT INTO wide (id, col_money, col_inet, col_point) VALUES (1, 12.34, '192.168.0.1', '(1,2)')",
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT col_money, col_inet, col_point FROM wide WHERE id = 1"
        )),
        vec![vec![
            dec("12.34"),
            Value::Text("192.168.0.1".to_owned()),
            Value::Text("(1,2)".to_owned()),
        ]]
    );
    // A genuinely unknown type name is still a loud error, not silently text.
    assert!(run_try(&engine, "CREATE TABLE bad (x notarealtype)").is_err());
}

#[test]
fn integer_arithmetic_errors_on_overflow_instead_of_wrapping() {
    // K1 — integer arithmetic must error on i64 overflow, not silently wrap (a wrapped
    // counter/financial value is silent data corruption). Matches standard 22003 behaviour.
    let engine = BtreeEngine::new();

    // Overflowing +, -, * all error rather than wrapping. (A literal beyond i64::MAX would parse as
    // NUMERIC, so the operands here stay i64-typed: 9223372036854775807 is exactly i64::MAX.)
    assert!(run_try(&engine, "SELECT 9223372036854775807 + 1").is_err());
    assert!(run_try(&engine, "SELECT -9223372036854775807 - 2").is_err());
    assert!(run_try(&engine, "SELECT 9223372036854775807 * 2").is_err());
    // The i64::MIN / -1 overflow also errors (not a wrap); the inner subtraction reaches i64::MIN.
    assert!(run_try(&engine, "SELECT (-9223372036854775807 - 1) / -1").is_err());
    // The DIV() and MOD() scalar functions guard the same i64::MIN / -1 overflow.
    assert!(run_try(&engine, "SELECT DIV(-9223372036854775807 - 1, -1)").is_err());
    assert!(run_try(&engine, "SELECT MOD(-9223372036854775807 - 1, -1)").is_err());
    // ABS(i64::MIN) and unary -(i64::MIN) have no positive counterpart, so they error too.
    assert!(run_try(&engine, "SELECT ABS(-9223372036854775807 - 1)").is_err());
    assert!(run_try(&engine, "SELECT -(-9223372036854775807 - 1)").is_err());

    // Non-overflowing arithmetic is unaffected.
    assert_eq!(
        rows(run(&engine, "SELECT 2 + 3, 10 - 4, 6 * 7, 20 / 3, 20 % 3")),
        vec![vec![
            Value::Int(5),
            Value::Int(6),
            Value::Int(42),
            Value::Int(6),
            Value::Int(2),
        ]]
    );
}

#[test]
fn bytea_type_cast_store_and_concat() {
    // BYTEA is representable: `\x` hex cast, round-trip storage, text render, `||` concat.
    let engine = BtreeEngine::new();
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // text `\x<hex>` -> bytea, and the empty byte string.
    assert_eq!(
        one(r"SELECT '\xDEADBEEF'::bytea"),
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF])
    );
    assert_eq!(one(r"SELECT '\x'::bytea"), Value::Bytes(vec![]));
    // bytea -> text renders the canonical lowercase `\x<hex>` form.
    assert_eq!(
        one(r"SELECT '\xAABB'::bytea::text"),
        Value::Text(r"\xaabb".to_owned())
    );
    // `||` concatenates two byte strings.
    assert_eq!(
        one(r"SELECT '\xAA'::bytea || '\xBB'::bytea"),
        Value::Bytes(vec![0xAA, 0xBB])
    );

    // Round-trips through stored rows (INSERT a text literal into a BYTEA column, SELECT it back).
    run(&engine, "CREATE TABLE t (id INT NOT NULL, b BYTEA)");
    run(&engine, r"INSERT INTO t VALUES (1, '\x0102ff'), (2, NULL)");
    assert_eq!(
        rows(run(&engine, "SELECT b FROM t ORDER BY id")),
        vec![
            vec![Value::Bytes(vec![0x01, 0x02, 0xFF])],
            vec![Value::Null]
        ]
    );
}

#[test]
fn json_path_operator_coerces_a_text_literal_path() {
    // `#>` / `#>>` accept a bare text literal path (`'{a,b}'`), coerced to text[] (SQL-standard).
    let engine = BtreeEngine::new();
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // `#>` (returns JSON) and `#>>` (returns text) with a string-literal path.
    assert_eq!(
        one(r#"SELECT '{"a":{"b":42}}'::json #>> '{a,b}'"#),
        Value::Text("42".to_owned())
    );
    assert_eq!(
        one(r#"SELECT '{"a":{"b":"x"}}'::json #>> '{a,b}'"#),
        Value::Text("x".to_owned())
    );
    // An explicit text[] path still works (unchanged).
    assert_eq!(
        one(r#"SELECT '{"a":{"b":42}}'::json #>> ARRAY['a','b']"#),
        Value::Text("42".to_owned())
    );
    // A path that misses yields NULL.
    assert_eq!(one(r#"SELECT '{"a":1}'::json #>> '{a,b}'"#), Value::Null);
}

#[test]
fn array_and_json_containment_operators() {
    // `@>` (contains) / `<@` (contained-by) over arrays and JSON (standard containment).
    let engine = BtreeEngine::new();
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // Array containment: every element of the right is present in the left.
    assert_eq!(
        one("SELECT ARRAY[1, 2, 3] @> ARRAY[2, 3]"),
        Value::Bool(true)
    );
    assert_eq!(
        one("SELECT ARRAY[1, 2, 3] @> ARRAY[2, 4]"),
        Value::Bool(false)
    );
    // Order and duplicates do not matter.
    assert_eq!(
        one("SELECT ARRAY[3, 1, 2] @> ARRAY[2, 2, 1]"),
        Value::Bool(true)
    );
    // `<@` is the mirror (`a <@ b` ≡ `b @> a`).
    assert_eq!(
        one("SELECT ARRAY[2, 3] <@ ARRAY[1, 2, 3]"),
        Value::Bool(true)
    );
    assert_eq!(
        one("SELECT ARRAY[2, 4] <@ ARRAY[1, 2, 3]"),
        Value::Bool(false)
    );
    // Text-array containment.
    assert_eq!(
        one("SELECT ARRAY['a', 'b', 'c'] @> ARRAY['c', 'a']"),
        Value::Bool(true)
    );
    // A NULL operand yields NULL (not false).
    assert_eq!(one("SELECT NULL @> ARRAY[1]"), Value::Null);
    assert_eq!(one("SELECT ARRAY[1, 2] <@ NULL"), Value::Null);

    // JSON containment still works through the same operators.
    assert_eq!(
        one(r#"SELECT '{"a":1,"b":2}'::json @> '{"a":1}'::json"#),
        Value::Bool(true)
    );
    assert_eq!(
        one(r#"SELECT '{"a":1}'::json <@ '{"a":1,"b":2}'::json"#),
        Value::Bool(true)
    );
    assert_eq!(
        one(r#"SELECT '{"a":1}'::json @> '{"b":2}'::json"#),
        Value::Bool(false)
    );
}

#[test]
fn json_build_array_constructs_a_json_array() {
    // QA differential limitation: json_build_array / jsonb_build_array were unknown functions. They
    // build a JSON array from the arguments in order; a NULL argument becomes JSON `null` (not a
    // propagated SQL NULL), any arity is accepted (including none), and elements keep their type.
    // (A bare untyped NULL is a separate, pre-existing limitation; a typed NULL flows through.)
    let engine = BtreeEngine::new();
    let jn = |s: &str| Value::Json(s.to_owned());

    assert_eq!(
        rows(run(
            &engine,
            "SELECT json_build_array(1, 'a', true, NULL::int)"
        )),
        vec![vec![jn(r#"[1,"a",true,null]"#)]],
    );
    // The jsonb_ spelling is the same function; an empty call yields an empty array.
    assert_eq!(
        rows(run(&engine, "SELECT jsonb_build_array()")),
        vec![vec![jn("[]")]],
    );
    // Nested: an array element that is itself a built object.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT json_build_array(json_build_object('k', 2), 3)"
        )),
        vec![vec![jn(r#"[{"k":2},3]"#)]],
    );
}

#[test]
fn b458_json_set_returning_functions_end_to_end() {
    // /c — json_array_elements + jsonb_path_query against the real engine.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE j (id INT NOT NULL, doc JSON)");
    run(&engine, "INSERT INTO j VALUES (1, '[10,20,30]')");
    run(
        &engine,
        r#"INSERT INTO j VALUES (2, '{"items":[{"n":1},{"n":2}]}')"#,
    );
    let jn = |s: &str| Value::Json(s.to_owned());

    // json_array_elements expands a JSON array into one JSON value per row.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT json_array_elements(doc) FROM j WHERE id = 1"
        )),
        vec![vec![jn("10")], vec![jn("20")], vec![jn("30")]]
    );
    // jsonb_path_query with a wildcard fans out, then a member step maps each match.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT jsonb_path_query(doc, '$.items[*].n') FROM j WHERE id = 2",
        )),
        vec![vec![jn("1")], vec![jn("2")]]
    );
}

#[test]
fn b134_within_group_ordered_set_aggregates() {
    // PERCENTILE_CONT / PERCENTILE_DISC / MODE WITHIN GROUP.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT, m INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 1, 5), (2, 2, 5), (3, 3, 5), (4, 4, 9), (5, NULL, 9)",
    );
    let one = |sql: &str| -> Value { rows(run(&engine, sql)).swap_remove(0).swap_remove(0) };

    // Scalar over the whole table (the NULL v is ignored): sorted v = [1,2,3,4], m = [5,5,5,9,9].
    assert_eq!(
        rows(run(
            &engine,
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v), \
             PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY v), \
             MODE() WITHIN GROUP (ORDER BY m) FROM t",
        ))[0],
        // CONT(0.5): interpolate between 2 and 3 → 2.5. DISC(0.5): the 2nd value → 2. MODE: 5 (×3).
        vec![Value::Float(2.5), Value::Int(2), Value::Int(5)]
    );

    // Boundary fractions and exact element selection.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0) WITHIN GROUP (ORDER BY v) FROM t"),
        Value::Float(1.0)
    );
    assert_eq!(
        one("SELECT PERCENTILE_CONT(1) WITHIN GROUP (ORDER BY v) FROM t"),
        Value::Float(4.0)
    );
    assert_eq!(
        one("SELECT PERCENTILE_DISC(1) WITHIN GROUP (ORDER BY v) FROM t"),
        Value::Int(4)
    );
    // A group with no non-NULL ordering value → NULL.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v) FROM t WHERE v IS NULL"),
        Value::Null
    );

    // Per group: m=5 has v {1,2,3} → DISC(0.5)=2; m=9 has v {4,NULL}→{4} → DISC(0.5)=4.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT m, PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY v) FROM t GROUP BY m ORDER BY m",
        )),
        vec![
            vec![Value::Int(5), Value::Int(2)],
            vec![Value::Int(9), Value::Int(4)],
        ]
    );

    // DESC ordering reverses the set. CONT(0.25) from the top of [4,3,2,1] interpolates between 4 and
    // 3 at weight 0.75 → 3.25 (vs 1.75 ascending); DISC(0.25) DESC picks the first, 4; the median is
    // direction-symmetric.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY v DESC) FROM t"),
        Value::Float(3.25)
    );
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY v) FROM t"),
        Value::Float(1.75)
    );
    assert_eq!(
        one("SELECT PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY v DESC) FROM t"),
        Value::Int(4)
    );
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v DESC) FROM t"),
        Value::Float(2.5)
    );
    // A NULLS FIRST/LAST clause is accepted and has no effect — NULL ordering values are excluded.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v ASC NULLS FIRST) FROM t"),
        Value::Float(2.5)
    );
    assert_eq!(
        one("SELECT PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY v DESC NULLS LAST) FROM t"),
        Value::Int(4)
    );
}

#[test]
fn set_operation_arity_and_type_mismatch_rejected() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (x INT NOT NULL)");
    run(&engine, "CREATE TABLE c (name TEXT)");
    // Column-count mismatch.
    assert!(run_try(&engine, "SELECT x FROM a UNION SELECT x, x FROM a").is_err());
    // Per-column type mismatch.
    assert!(run_try(&engine, "SELECT x FROM a UNION SELECT name FROM c").is_err());
}

#[test]
fn distinct_on_keeps_first_row_per_key() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (g INT NOT NULL, v INT NOT NULL)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 10), (1, 20), (2, 5), (2, 7), (3, 9)",
    );

    // DISTINCT ON (g) ORDER BY g, v → the smallest v per g.
    match run(&engine, "SELECT DISTINCT ON (g) g, v FROM t ORDER BY g, v") {
        ExecutionResult::Rows { columns, rows } => {
            assert_eq!(columns, vec!["g".to_owned(), "v".to_owned()]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::Int(1), Value::Int(10)],
                    vec![Value::Int(2), Value::Int(5)],
                    vec![Value::Int(3), Value::Int(9)],
                ]
            );
        },
        other => panic!("expected DISTINCT ON rows, got {other:?}"),
    }

    // ORDER BY g, v DESC → the largest v per g.
    match run(
        &engine,
        "SELECT DISTINCT ON (g) g, v FROM t ORDER BY g, v DESC",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![Value::Int(1), Value::Int(20)],
                    vec![Value::Int(2), Value::Int(7)],
                    vec![Value::Int(3), Value::Int(9)],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn select_with_where_filters_rows() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    run(&engine, "INSERT INTO t VALUES (3, 30)");

    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE v = 20")),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn update_is_visible_to_a_later_select() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 100)");
    assert!(matches!(
        run(&engine, "UPDATE t SET v = 200 WHERE id = 1"),
        ExecutionResult::Updated(1)
    ));
    assert_eq!(
        rows(run(&engine, "SELECT v FROM t WHERE id = 1")),
        vec![vec![Value::Int(200)]]
    );
}

#[test]
fn delete_removes_the_matching_row() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1)");
    run(&engine, "INSERT INTO t VALUES (2)");
    assert!(matches!(
        run(&engine, "DELETE FROM t WHERE id = 1"),
        ExecutionResult::Deleted(1)
    ));
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn analyzer_rejects_unknown_table_against_real_catalog() {
    let engine = BtreeEngine::new();
    let stmt = parse("SELECT id FROM ghost").expect("parse");
    assert!(analyze(stmt, &EngineCatalog(&engine)).is_err());
}

#[test]
fn select_distinct_dedupes_rows() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 10), (3, 20)");
    assert_eq!(
        rows(run(&engine, "SELECT DISTINCT v FROM t ORDER BY v")),
        vec![vec![Value::Int(10)], vec![Value::Int(20)]],
    );
}

#[test]
fn distinct_and_set_ops_match_on_hash_and_fallback_paths() {
    // DISTINCT / INTERSECT / EXCEPT use a hash fast path for hash-safe types and a linear
    // fallback for FLOAT/NUMERIC; both must give the same result (NULL not distinct from NULL).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (i INT, f FLOAT)");
    run(
        &engine,
        "INSERT INTO a VALUES (1, 1.5), (1, 1.5), (2, 2.5), (NULL, NULL), (NULL, NULL)",
    );

    // INT column → hash path; the two (1) rows and the two NULL rows each collapse. NULL sorts last.
    assert_eq!(
        rows(run(&engine, "SELECT DISTINCT i FROM a ORDER BY i")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Null]],
    );
    // FLOAT column → linear fallback; same dedup semantics.
    assert_eq!(
        rows(run(&engine, "SELECT DISTINCT f FROM a ORDER BY f")),
        vec![
            vec![Value::Float(1.5)],
            vec![Value::Float(2.5)],
            vec![Value::Null],
        ],
    );

    run(&engine, "CREATE TABLE b (i INT)");
    run(&engine, "INSERT INTO b VALUES (2), (3)");
    // INTERSECT (hash membership): a.i ∩ b.i = {2}.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT i FROM a INTERSECT SELECT i FROM b ORDER BY i",
        )),
        vec![vec![Value::Int(2)]],
    );
    // EXCEPT (hash membership): a.i − b.i = {1, NULL} (NULL sorts last).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT i FROM a EXCEPT SELECT i FROM b ORDER BY i",
        )),
        vec![vec![Value::Int(1)], vec![Value::Null]],
    );
}

#[test]
fn show_tables_and_columns_introspect_the_catalog() {
    // Catalog introspection that backs the CLI's \dt / \d.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE alpha (id INT NOT NULL, name TEXT)");
    run(&engine, "CREATE TABLE beta (x FLOAT)");

    // SHOW TABLES → one `table` column, sorted.
    assert_eq!(
        rows(run(&engine, "SHOW TABLES")),
        vec![
            vec![Value::Text("alpha".to_owned())],
            vec![Value::Text("beta".to_owned())],
        ],
    );

    // SHOW COLUMNS FROM t → (column, type, nullable).
    assert_eq!(
        rows(run(&engine, "SHOW COLUMNS FROM alpha")),
        vec![
            vec![
                Value::Text("id".to_owned()),
                Value::Text("INT".to_owned()),
                Value::Bool(false),
            ],
            vec![
                Value::Text("name".to_owned()),
                Value::Text("TEXT".to_owned()),
                Value::Bool(true),
            ],
        ],
    );

    // A dropped table disappears; an unknown table is rejected.
    run(&engine, "DROP TABLE beta");
    assert_eq!(
        rows(run(&engine, "SHOW TABLES")),
        vec![vec![Value::Text("alpha".to_owned())]],
    );
    assert!(run_try(&engine, "SHOW COLUMNS FROM ghost").is_err());
}

#[test]
fn null_handling_via_is_null() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");
    run(&engine, "INSERT INTO t (id, name) VALUES (1, NULL)");
    run(&engine, "INSERT INTO t (id, name) VALUES (2, 'present')");
    run(&engine, "INSERT INTO t (id, name) VALUES (3, NULL)");

    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE name IS NULL ORDER BY id",
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]],
    );
}

#[test]
fn offset_and_limit_paginate() {
    // OFFSET skips leading rows; combined with LIMIT it paginates; OFFSET alone skips the
    // prefix and returns the rest. Ordered for a deterministic slice.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3), (4), (5)");

    let ids = |sql: &str| -> Vec<i64> {
        rows(run(&engine, sql))
            .into_iter()
            .map(|r| match r.into_iter().next() {
                Some(Value::Int(n)) => n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect()
    };

    // LIMIT + OFFSET: skip 1, take 2 → rows 2,3.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY id LIMIT 2 OFFSET 1"),
        vec![2, 3]
    );
    // OFFSET alone: skip 3 → rows 4,5.
    assert_eq!(ids("SELECT id FROM t ORDER BY id OFFSET 3"), vec![4, 5]);
    // OFFSET past the end → empty.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY id OFFSET 99"),
        Vec::<i64>::new()
    );
    // OFFSET 0 is a no-op.
    assert_eq!(
        ids("SELECT id FROM t ORDER BY id LIMIT 2 OFFSET 0"),
        vec![1, 2]
    );
}

#[test]
fn fetch_first_with_ties_keeps_peers() {
    // FETCH FIRST n ROWS WITH TIES keeps, beyond the first n rows, every following
    // row that ties the n-th row on the ORDER BY keys (the SQL-standard WITH TIES semantics).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, grp INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1,10),(2,10),(3,20),(4,20),(5,20),(6,30)",
    );

    // Collect the returned ids and sort them — WITH TIES fixes the *set* of rows (peers on `grp`),
    // not their relative order within a tie group.
    let sorted_ids = |sql: &str| -> Vec<i64> {
        let mut out: Vec<i64> = rows(run(&engine, sql))
            .into_iter()
            .map(|r| match r.into_iter().next() {
                Some(Value::Int(n)) => n,
                other => panic!("expected int, got {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    };

    // FETCH FIRST 1 ROW WITH TIES: row 1 (grp 10) plus its peer (row 2) → {1,2}.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp FETCH FIRST 1 ROWS WITH TIES"),
        vec![1, 2]
    );
    // FETCH FIRST 3 ROWS WITH TIES: rows 1,2,3 then the peers of row 3 (grp 20) → {1,2,3,4,5}.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp FETCH FIRST 3 ROWS WITH TIES"),
        vec![1, 2, 3, 4, 5]
    );
    // The boundary falls on a non-tie (row 2 is grp 10, row 3 is grp 20) → exactly 2 rows, no extra.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp FETCH FIRST 2 ROWS WITH TIES"),
        vec![1, 2]
    );
    // FETCH FIRST ... ONLY ignores ties → exactly 1 row.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp, id FETCH FIRST 1 ROWS ONLY"),
        vec![1]
    );
    // OFFSET applies before the tie extension: skip 2, take 1 (grp 20) + its peers → {3,4,5}.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp OFFSET 2 ROWS FETCH FIRST 1 ROWS WITH TIES"),
        vec![3, 4, 5]
    );
    // A count past the end returns everything, with no spurious ties.
    assert_eq!(
        sorted_ids("SELECT id FROM t ORDER BY grp FETCH FIRST 99 ROWS WITH TIES"),
        vec![1, 2, 3, 4, 5, 6]
    );

    // WITH TIES requires an ORDER BY, and is rejected (not silently wrong) with DISTINCT.
    assert!(run_try(&engine, "SELECT id FROM t FETCH FIRST 1 ROWS WITH TIES").is_err());
    assert!(
        run_try(
            &engine,
            "SELECT DISTINCT grp FROM t ORDER BY grp FETCH FIRST 1 ROWS WITH TIES"
        )
        .is_err()
    );
}

#[test]
fn truncate_empties_the_table() {
    // TRUNCATE removes every row (desugars to an unfiltered DELETE); the table stays usable.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3)");
    assert_eq!(rows(run(&engine, "SELECT id FROM t")).len(), 3);

    run(&engine, "TRUNCATE TABLE t");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        Vec::<Vec<Value>>::new(),
        "TRUNCATE empties the table"
    );

    // Still insertable + queryable after truncation.
    run(&engine, "INSERT INTO t VALUES (9)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(9)]]
    );
}

#[test]
fn truncate_restart_identity_resets_the_serial_sequence() {
    // `TRUNCATE ... RESTART IDENTITY` resets the backing SERIAL sequence so the next insert
    // restarts at 1; plain TRUNCATE (CONTINUE IDENTITY) keeps the sequence advancing. (QA finding
    // Previously the clause was accepted but silently ignored.)
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id SERIAL PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t (v) VALUES (10), (20)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );

    // Plain TRUNCATE empties the table but the sequence continues → next id is 3.
    run(&engine, "TRUNCATE TABLE t");
    run(&engine, "INSERT INTO t (v) VALUES (30)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(3)]],
        "plain TRUNCATE continues the identity sequence"
    );

    // TRUNCATE ... RESTART IDENTITY rewinds the sequence → next id is 1 again.
    run(&engine, "TRUNCATE TABLE t RESTART IDENTITY");
    run(&engine, "INSERT INTO t (v) VALUES (40)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(1)]],
        "RESTART IDENTITY rewinds the sequence to its start"
    );
}

#[test]
fn explain_returns_a_plan_tree() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    let lines: Vec<String> = rows(run(
        &engine,
        "EXPLAIN SELECT * FROM t WHERE id > 0 ORDER BY id LIMIT 5",
    ))
    .into_iter()
    .map(|row| match row.into_iter().next() {
        Some(Value::Text(s)) => s,
        other => panic!("expected plan text, got {other:?}"),
    })
    .collect();
    let joined = lines.join("\n");
    assert!(joined.contains("Limit 5"), "missing Limit: {joined}");
    assert!(joined.contains("Sort"), "missing Sort: {joined}");
    assert!(joined.contains("Filter"), "missing Filter: {joined}");
    assert!(joined.contains("SeqScan: t"), "missing SeqScan: {joined}");
    // Without ANALYZE there are no statistics, so EXPLAIN omits row estimates.
    assert!(
        !joined.contains("est. rows"),
        "unexpected estimate: {joined}"
    );
}

#[test]
fn explain_annotates_estimated_rows_after_analyze() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    for i in 0..8 {
        run(&engine, &format!("INSERT INTO t VALUES ({i}, {})", i % 2));
    }
    run(&engine, "ANALYZE TABLE t");
    let joined: String = rows(run(&engine, "EXPLAIN SELECT id FROM t WHERE id = 3"))
        .into_iter()
        .map(|row| match row.into_iter().next() {
            Some(Value::Text(s)) => s,
            other => panic!("expected plan text, got {other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    // All 8 rows are scanned; the equality filter on the unique `id` keeps ~1. Each node also
    // carries its estimated subtree cost. Projection pushdown narrows the scan to
    // the single referenced column (`id`), dropping `val`, which EXPLAIN annotates.
    assert!(
        joined.contains("SeqScan: t (project 1/2 cols) (est. rows=8 cost="),
        "missing scan estimate: {joined}",
    );
    assert!(
        joined.contains("Filter (est. rows=1 cost="),
        "missing filter estimate: {joined}",
    );
}

#[test]
fn vectorized_path_matches_row_path_end_to_end() {
    // Wiring — with the opt-in vectorized flag on, a SeqScan/Filter/Project/Sort/Limit query
    // returns exactly what the row path does, against the real engine.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, v INT, name TEXT)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 30, 'a'), (2, NULL, 'b'), (3, 10, 'c'), (4, 20, 'd')",
    );
    let sql = "SELECT id, name FROM t WHERE v >= 10 ORDER BY v DESC LIMIT 2";
    let row_path = rows(run(&engine, sql));
    let batch_path = {
        let _g = nusadb_sql::vectorized::scope(true);
        rows(run(&engine, sql))
    };
    assert_eq!(
        row_path, batch_path,
        "vectorized path diverged from the row path"
    );
    // Sanity: the expected rows (v=30 then v=20; the NULL row is excluded by `>= 10`).
    assert_eq!(
        batch_path,
        vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Int(4), Value::Text("d".to_owned())],
        ]
    );
}

#[test]
fn cost_based_index_vs_seq_scan_selection() {
    // With ANALYZE stats the planner compares costs: a selective equality on an indexed
    // column takes the index; a barely-selective range that keeps most rows takes a sequential scan.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT NOT NULL)");
    run(&engine, "CREATE INDEX t_v ON t (v)");
    // 200 rows with distinct v in [0, 200) — an equality on v is highly selective (~1/200).
    for i in 0..200 {
        run(&engine, &format!("INSERT INTO t VALUES ({i}, {i})"));
    }
    run(&engine, "ANALYZE TABLE t");

    let plan = |sql: &str| -> String {
        rows(run(&engine, sql))
            .into_iter()
            .filter_map(|row| match row.into_iter().next() {
                Some(Value::Text(s)) => Some(s),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Selective: v = 5 keeps ~1 of 200 rows → index scan is cheaper.
    let selective = plan("EXPLAIN SELECT id FROM t WHERE v = 5");
    assert!(
        selective.contains("IndexScan: t using t_v"),
        "selective equality should use the index:\n{selective}",
    );
    // Non-selective: v >= 0 keeps every row → a sequential scan is cheaper than random index fetches.
    let broad = plan("EXPLAIN SELECT id FROM t WHERE v >= 0");
    assert!(
        broad.contains("SeqScan: t") && !broad.contains("IndexScan"),
        "non-selective range should use a sequential scan:\n{broad}",
    );

    // Both still return the correct rows regardless of the chosen plan.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE v = 5")),
        vec![vec![Value::Int(5)]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE v >= 0")).len(),
        200
    );
}

#[test]
fn in_list_filter_works() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')",
    );

    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE id IN (1, 3, 99) ORDER BY id",
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]],
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE id NOT IN (1, 3) ORDER BY id",
        )),
        vec![vec![Value::Int(2)], vec![Value::Int(4)]],
    );
}

#[test]
fn between_filter_works_inclusive() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, age INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
    );

    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE age BETWEEN 20 AND 40 ORDER BY id",
        )),
        vec![
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)]
        ],
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE age NOT BETWEEN 20 AND 40 ORDER BY id",
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(5)]],
    );
}

#[test]
fn like_pattern_matching_works() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'alice'), (2, 'bob'), (3, 'alan'), (4, 'charlie')",
    );

    // prefix match
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE name LIKE 'al%' ORDER BY id",
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]],
    );
    // single-char wildcard
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE name LIKE 'ali_e' ORDER BY id",
        )),
        vec![vec![Value::Int(1)]],
    );
    // NOT LIKE
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE name NOT LIKE 'a%' ORDER BY id",
        )),
        vec![vec![Value::Int(2)], vec![Value::Int(4)]],
    );
    // substring match
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE name LIKE '%li%' ORDER BY id",
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(4)]],
    );
}

#[test]
fn case_searched_form_chooses_first_true_branch() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, score INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 30), (2, 70), (3, 95), (4, NULL)",
    );

    let result = rows(run(
        &engine,
        "SELECT id, \
            CASE WHEN score >= 90 THEN 'A' \
                 WHEN score >= 60 THEN 'B' \
                 WHEN score >= 0  THEN 'F' \
                 ELSE 'NA' END \
         FROM t ORDER BY id",
    ));
    assert_eq!(result[0][1], Value::Text("F".to_owned()));
    assert_eq!(result[1][1], Value::Text("B".to_owned()));
    assert_eq!(result[2][1], Value::Text("A".to_owned()));
    // NULL >= 0 is NULL → no branch matches → ELSE fires.
    assert_eq!(result[3][1], Value::Text("NA".to_owned()));
}

#[test]
fn case_simple_form_matches_against_operand() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, status INT)");
    run(&engine, "INSERT INTO t VALUES (1, 1), (2, 2), (3, 9)");

    let result = rows(run(
        &engine,
        "SELECT id, \
            CASE status WHEN 1 THEN 'active' \
                        WHEN 2 THEN 'pending' \
                        ELSE 'unknown' END \
         FROM t ORDER BY id",
    ));
    assert_eq!(result[0][1], Value::Text("active".to_owned()));
    assert_eq!(result[1][1], Value::Text("pending".to_owned()));
    assert_eq!(result[2][1], Value::Text("unknown".to_owned()));
}

#[test]
fn case_without_else_returns_null_on_no_match() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 5)");
    let result = rows(run(
        &engine,
        "SELECT CASE WHEN v > 100 THEN 'high' END FROM t",
    ));
    assert_eq!(result, vec![vec![Value::Null]]);
}

#[test]
fn coalesce_returns_first_non_null() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, a TEXT, b TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, NULL, 'fallback'), (2, 'primary', 'fallback'), (3, NULL, NULL)",
    );
    let result = rows(run(
        &engine,
        "SELECT id, COALESCE(a, b, 'default') FROM t ORDER BY id",
    ));
    assert_eq!(result[0][1], Value::Text("fallback".to_owned()));
    assert_eq!(result[1][1], Value::Text("primary".to_owned()));
    assert_eq!(result[2][1], Value::Text("default".to_owned()));
}

#[test]
fn row_value_comparison_is_lexicographic() {
    // `(a, b) OP (c, d)` / `ROW(a, b) OP ROW(c, d)` compare row-wise: `=`/`<>`
    // element-wise, the ordering operators lexicographically, all under 3-valued NULL logic.
    let engine = BtreeEngine::new();
    let b = |sql: &str| -> Value {
        rows(run(&engine, &format!("SELECT {sql}")))
            .swap_remove(0)
            .swap_remove(0)
    };

    // Equality / inequality are element-wise.
    assert_eq!(b("(1, 2) = (1, 2)"), Value::Bool(true));
    assert_eq!(b("(1, 2) = (1, 3)"), Value::Bool(false));
    assert_eq!(b("(1, 2) <> (1, 3)"), Value::Bool(true));
    assert_eq!(b("ROW(1, 'a') = ROW(1, 'a')"), Value::Bool(true));

    // Lexicographic ordering: the first field decides; ties fall through to the next.
    assert_eq!(b("(1, 2) < (1, 3)"), Value::Bool(true));
    assert_eq!(b("(1, 9) < (2, 0)"), Value::Bool(true)); // first field 1 < 2
    assert_eq!(b("(2, 0) < (1, 9)"), Value::Bool(false));
    assert_eq!(b("(1, 2) <= (1, 2)"), Value::Bool(true));
    assert_eq!(b("(3, 1) > (2, 9)"), Value::Bool(true));
    assert_eq!(b("(1, 2, 3) >= (1, 2, 3)"), Value::Bool(true));

    // 3-valued NULL logic: a decided first field wins; an undecided comparison propagates NULL.
    assert_eq!(b("(1, NULL) < (2, 1)"), Value::Bool(true)); // 1 < 2 decides
    assert_eq!(b("(1, NULL) < (1, 2)"), Value::Null); // tie, then NULL < 2 is NULL
    assert_eq!(b("(1, NULL) = (1, 2)"), Value::Null);

    // Rows of unequal length are a loud error.
    assert!(run_try(&engine, "SELECT (1, 2) = (1, 2, 3)").is_err());
}

#[test]
fn cast_supports_common_conversions() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, n INT, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 42, '123'), (2, NULL, 'oops')",
    );

    // Int → Float
    assert_eq!(
        rows(run(&engine, "SELECT CAST(n AS FLOAT) FROM t WHERE id = 1"))[0][0],
        Value::Float(42.0),
    );
    // Int → Text
    assert_eq!(
        rows(run(&engine, "SELECT CAST(n AS TEXT) FROM t WHERE id = 1"))[0][0],
        Value::Text("42".to_owned()),
    );
    // Text → Int (success)
    assert_eq!(
        rows(run(&engine, "SELECT CAST(s AS INT) FROM t WHERE id = 1"))[0][0],
        Value::Int(123),
    );
    // NULL preserved
    assert_eq!(
        rows(run(&engine, "SELECT CAST(n AS FLOAT) FROM t WHERE id = 2"))[0][0],
        Value::Null,
    );
}

#[test]
fn numeric_to_integer_cast_rounds_half_away_from_zero() {
    // CAST(numeric AS integer) rounds half-away-from-zero (matching the float -> int cast), not
    // truncate toward zerofound by the QA differential suite.
    let engine = BtreeEngine::new();
    assert_eq!(
        rows(run(
            &engine,
            "SELECT 2.6::int, 3.5::int, 2.5::int, 2.4::int"
        )),
        vec![vec![
            Value::Int(3),
            Value::Int(4),
            Value::Int(3),
            Value::Int(2),
        ]],
    );
    // Negatives round away from zero too, via the CAST(...) form.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT CAST(-2.6 AS INT), CAST(-2.5 AS INT), CAST(-2.4 AS INT)"
        )),
        vec![vec![Value::Int(-3), Value::Int(-3), Value::Int(-2)]],
    );
}

#[test]
fn date_integer_arithmetic_and_date_difference() {
    // Date arithmetic (QA differential): `date + int` adds whole days -> DATE
    // (commutative), `date - int` -> DATE, and `date - date` -> the day count as INTEGER.
    let engine = BtreeEngine::new();
    let d = |s: &str| Value::Date(nusadb_sql::temporal::parse_date(s).unwrap());
    assert_eq!(
        rows(run(&engine, "SELECT DATE '2024-01-01' + 7")),
        vec![vec![d("2024-01-08")]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT 7 + DATE '2024-01-01'")),
        vec![vec![d("2024-01-08")]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT DATE '2024-01-08' - 7")),
        vec![vec![d("2024-01-01")]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT DATE '2024-01-08' - DATE '2024-01-01'")),
        vec![vec![Value::Int(7)]],
    );
    // Crosses a month boundary (2024 is a leap year, so Feb has 29 days).
    assert_eq!(
        rows(run(&engine, "SELECT DATE '2024-02-28' + 2")),
        vec![vec![d("2024-03-01")]],
    );
}

#[test]
fn interval_times_integer_scales_each_component() {
    // Interval scaling (QA differential §Limitasi): `interval * integer` scales every component
    // (commutative); each unit is scaled independently (no remainder spill).
    use nusadb_sql::interval::Interval;
    let engine = BtreeEngine::new();
    assert_eq!(
        rows(run(&engine, "SELECT INTERVAL '1 month 2 days 3 hours' * 3")),
        vec![vec![Value::Interval(Interval {
            months: 3,
            days: 6,
            micros: 9 * 3_600 * 1_000_000,
        })]],
    );
    // Commutative: integer * interval.
    assert_eq!(
        rows(run(&engine, "SELECT 2 * INTERVAL '1 day'")),
        vec![vec![Value::Interval(Interval {
            months: 0,
            days: 2,
            micros: 0,
        })]],
    );
    // Applied to a date (`*` binds tighter than `+`): date + interval*3 = date + 3 days.
    let ts = nusadb_sql::temporal::parse_timestamp("2024-01-04 00:00:00").unwrap();
    assert_eq!(
        rows(run(
            &engine,
            "SELECT DATE '2024-01-01' + INTERVAL '1 day' * 3"
        )),
        vec![vec![Value::Timestamp(ts)]],
    );
}

#[test]
fn varchar_cast_truncates_to_its_declared_length() {
    // An explicit `::varchar(n)` / `CAST(... AS VARCHAR(n))` truncates to n characters (QA
    // differential); a non-text value casts to text first, and NULL stays NULL.
    let engine = BtreeEngine::new();
    let t = |s: &str| Value::Text(s.to_owned());
    assert_eq!(
        rows(run(&engine, "SELECT 'abcdef'::varchar(3)")),
        vec![vec![t("abc")]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT CAST('hello world' AS VARCHAR(5))")),
        vec![vec![t("hello")]],
    );
    // Shorter text is unchanged — VARCHAR does not pad.
    assert_eq!(
        rows(run(&engine, "SELECT 'ab'::varchar(5)")),
        vec![vec![t("ab")]],
    );
    // A non-text value is cast to text first, then truncated.
    assert_eq!(
        rows(run(&engine, "SELECT 12345::varchar(2)")),
        vec![vec![t("12")]],
    );
    // NULL stays NULL.
    assert_eq!(
        rows(run(&engine, "SELECT NULL::varchar(3)")),
        vec![vec![Value::Null]],
    );
    // An unbounded VARCHAR / TEXT cast does not truncate.
    assert_eq!(
        rows(run(&engine, "SELECT 'abcdef'::varchar")),
        vec![vec![t("abcdef")]],
    );
}

#[test]
fn char_cast_truncates_to_its_declared_length() {
    // A fixed `::char(n)` / `CAST(... AS CHAR(n))` truncates to n characters too (QA differential
    // #9): truncate when longer, leave shorter text unchanged (NusaDB does not blank-pad a short
    // value out to n). CHAR(n) is blank-padded (bpchar) — trailing blanks are insignificant and the
    // cast strips them — but every input here is blank-free, so truncation is the only visible
    // effect; the trailing-blank semantics are covered in the p1_ddl/varchar_length SLT.
    let engine = BtreeEngine::new();
    let t = |s: &str| Value::Text(s.to_owned());
    assert_eq!(
        rows(run(&engine, "SELECT 'abcdef'::char(3)")),
        vec![vec![t("abc")]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT CAST('hello world' AS CHARACTER(5))")),
        vec![vec![t("hello")]],
    );
    // Shorter text is unchanged (NusaDB CHAR does not blank-pad).
    assert_eq!(
        rows(run(&engine, "SELECT 'ab'::char(5)")),
        vec![vec![t("ab")]],
    );
    // NULL stays NULL; an unbounded CHAR cast does not truncate.
    assert_eq!(
        rows(run(&engine, "SELECT NULL::char(3)")),
        vec![vec![Value::Null]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT 'abcdef'::char")),
        vec![vec![t("abcdef")]],
    );
}

#[test]
fn array_text_quotes_special_elements_and_round_trips() {
    // QA differential #11: array -> text must quote + escape an element containing a delimiter,
    // quote, backslash, or whitespace (and one that spells NULL) so the `{...}` form re-parses to the
    // same elements. A plain element stays bare; a genuine NULL element is the bare token.
    let engine = BtreeEngine::new();
    let t = |s: &str| Value::Text(s.to_owned());

    assert_eq!(
        rows(run(
            &engine,
            r#"SELECT ARRAY['a', 'b,c', 'd"e', 'x y', 'NULL']::text"#
        )),
        vec![vec![t(r#"{a,"b,c","d\"e","x y","NULL"}"#)]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT ARRAY['a', NULL, 'c']::text")),
        vec![vec![t("{a,NULL,c}")]],
    );
    // Round-trip: the quoted form casts back to text[] and re-renders identically — a quoted "NULL"
    // stays the string, an unquoted NULL stays the null element.
    assert_eq!(
        rows(run(
            &engine,
            r#"SELECT ('{a,"b,c","d\"e","x y","NULL",NULL}'::text[])::text"#
        )),
        vec![vec![t(r#"{a,"b,c","d\"e","x y","NULL",NULL}"#)]],
    );
}

#[test]
fn generate_series_as_a_from_table_function() {
    // A set-returning function in FROM (QA): `FROM generate_series(a, b)` produces one row
    // per value, usable with aliases, WHERE/ORDER, aggregates, and joins.
    let engine = BtreeEngine::new();
    // Bare table function — one column named after the function.
    assert_eq!(
        rows(run(&engine, "SELECT * FROM generate_series(1, 3)")),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ],
    );
    // Alias + WHERE + ORDER over the generated column.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT g FROM generate_series(1, 5) AS g WHERE g % 2 = 1 ORDER BY g"
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(3)],
            vec![Value::Int(5)]
        ],
    );
    // An explicit column-alias list and an aggregate over the series.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT SUM(n) FROM generate_series(1, 4) AS t(n)"
        )),
        vec![vec![Value::Int(10)]],
    );
    // Joined against a real table.
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, label TEXT NOT NULL)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')");
    assert_eq!(
        rows(run(
            &engine,
            "SELECT t.label FROM t JOIN generate_series(1, 2) AS g(id) ON g.id = t.id ORDER BY t.id"
        )),
        vec![
            vec![Value::Text("a".to_owned())],
            vec![Value::Text("b".to_owned())],
        ],
    );
}

#[test]
fn unnest_as_a_from_table_function() {
    // QA differential limitation: `FROM unnest(array)` produces one row per element (the inconsistency
    // with generate_series, which already worked in FROM). Covers bare form, alias + column alias, an
    // aggregate over the elements, and a join.
    let engine = BtreeEngine::new();
    assert_eq!(
        rows(run(&engine, "SELECT * FROM unnest(ARRAY[10, 20, 30])")),
        vec![
            vec![Value::Int(10)],
            vec![Value::Int(20)],
            vec![Value::Int(30)],
        ],
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT SUM(n) FROM unnest(ARRAY[1, 2, 3, 4]) AS t(n)"
        )),
        vec![vec![Value::Int(10)]],
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v FROM unnest(ARRAY['a', 'b', 'c']) AS t(v) WHERE v <> 'b' ORDER BY v"
        )),
        vec![
            vec![Value::Text("a".to_owned())],
            vec![Value::Text("c".to_owned())],
        ],
    );
}

#[test]
fn avg_over_an_exact_type_is_exact_numeric_not_lossy_float() {
    // QA Temuan-4: AVG over an integer (or NUMERIC) column is exact NUMERIC, not a lossy f64.
    // avg(1, 2, 5) = 8/3 must be the exact numeric quotient, not the f64 8.0/3.0.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE m (v INT NOT NULL)");
    run(&engine, "INSERT INTO m VALUES (1), (2), (5)");
    let exact = Value::Numeric(
        nusadb_sql::numeric::Decimal::from_i64(8)
            .checked_div(&nusadb_sql::numeric::Decimal::from_i64(3))
            .unwrap(),
    );
    assert_eq!(
        rows(run(&engine, "SELECT AVG(v) FROM m")),
        vec![vec![exact]]
    );
    // AVG over a FLOAT column stays FLOAT (matching the argument type).
    run(&engine, "CREATE TABLE f (v FLOAT NOT NULL)");
    run(&engine, "INSERT INTO f VALUES (1.0), (2.0), (5.0)");
    assert_eq!(
        rows(run(&engine, "SELECT AVG(v) FROM f")),
        vec![vec![Value::Float(8.0 / 3.0)]],
    );
}

#[test]
fn insert_values_accepts_an_uncorrelated_scalar_subquery() {
    // An uncorrelated scalar subquery is allowed in an INSERT ... VALUES cell (resolved to a literal
    // before the row is built), e.g. `INSERT INTO t VALUES (1, (SELECT max(v) FROM src))`.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE src (v INT NOT NULL)");
    run(&engine, "INSERT INTO src VALUES (10), (30), (20)");
    run(&engine, "CREATE TABLE t (id INT NOT NULL, hi INT, lo INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, (SELECT MAX(v) FROM src), (SELECT MIN(v) FROM src)), \
         (2, (SELECT COUNT(*) FROM src), 0)",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, hi, lo FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(30), Value::Int(10)],
            vec![Value::Int(2), Value::Int(3), Value::Int(0)],
        ],
    );
    // A subquery returning more than one row is an error (scalar context).
    assert!(
        run_try(&engine, "INSERT INTO t VALUES (3, (SELECT v FROM src), 0)").is_err(),
        "a multi-row scalar subquery in a VALUES cell must be rejected",
    );
}

#[test]
fn top_level_values_query_returns_its_rows() {
    // A bare `VALUES (row), ...` is a query that returns its rows directly, with columns
    // named column1, column2, …; ORDER BY / LIMIT bind to the result.
    let engine = BtreeEngine::new();
    assert_eq!(
        rows(run(&engine, "VALUES (1, 'a'), (2, 'b'), (3, 'c')")),
        vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Int(2), Value::Text("b".to_owned())],
            vec![Value::Int(3), Value::Text("c".to_owned())],
        ],
    );
    // ORDER BY the synthetic column name + LIMIT.
    assert_eq!(
        rows(run(
            &engine,
            "VALUES (3), (1), (2) ORDER BY column1 LIMIT 2"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]],
    );
    // A single-row VALUES is a one-row result.
    assert_eq!(
        rows(run(&engine, "VALUES (10)")),
        vec![vec![Value::Int(10)]],
    );
}

#[test]
fn update_from_a_values_source_applies_per_matched_row() {
    // `UPDATE ... FROM (VALUES ...)` — a bulk update mapping ids to new values. A target row
    // with no matching source row is left unchanged.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    run(
        &engine,
        "UPDATE t SET v = u.nv FROM (VALUES (1, 100), (3, 300)) AS u(id, nv) WHERE t.id = u.id",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(20)], // unmatched → unchanged
            vec![Value::Int(3), Value::Int(300)],
        ],
    );
}

#[test]
fn update_from_a_derived_select_source() {
    // `UPDATE ... FROM (SELECT ...)` — the source is a filtered subquery over a real table.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(
        &engine,
        "CREATE TABLE src (id INT NOT NULL, nv INT NOT NULL)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    run(&engine, "INSERT INTO src VALUES (1, 111), (2, 222)");
    run(
        &engine,
        "UPDATE t SET v = s.nv FROM (SELECT id, nv FROM src WHERE nv > 200) AS s WHERE t.id = s.id",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)], // src row 1 (nv=111) filtered out → unchanged
            vec![Value::Int(2), Value::Int(222)],
        ],
    );
}

#[test]
fn delete_using_a_values_source() {
    // `DELETE ... USING (VALUES ...)` — delete the target rows whose key is in the inline source.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    run(
        &engine,
        "DELETE FROM t USING (VALUES (1), (3)) AS d(id) WHERE t.id = d.id",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(2)]],
    );
}

/// Build a `PhysicalPlan` from SQL using the same pipeline `run` uses, but
/// without executing — for the Session-based transaction tests.
fn build_plan(engine: &BtreeEngine, sql: &str) -> nusadb_sql::PhysicalPlan {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    plan(logical)
}

#[test]
fn explicit_transaction_commit_persists() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");

    {
        let mut session = Session::new(&engine);
        assert!(matches!(
            session.execute(build_plan(&engine, "BEGIN")).unwrap(),
            ExecutionResult::TransactionBegun,
        ));
        assert!(session.in_transaction());
        session
            .execute(build_plan(&engine, "INSERT INTO t VALUES (1, 'a')"))
            .unwrap();
        session
            .execute(build_plan(&engine, "INSERT INTO t VALUES (2, 'b')"))
            .unwrap();
        assert!(matches!(
            session.execute(build_plan(&engine, "COMMIT")).unwrap(),
            ExecutionResult::TransactionCommitted,
        ));
        assert!(!session.in_transaction());
    }

    // Outside the session, a fresh auto-commit SELECT sees both rows.
    assert_eq!(rows(run(&engine, "SELECT id FROM t ORDER BY id")).len(), 2);
}

#[test]
fn explicit_transaction_rollback_discards() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1)");

    {
        let mut session = Session::new(&engine);
        session.execute(build_plan(&engine, "BEGIN")).unwrap();
        session
            .execute(build_plan(&engine, "INSERT INTO t VALUES (2)"))
            .unwrap();
        session
            .execute(build_plan(&engine, "INSERT INTO t VALUES (3)"))
            .unwrap();
        assert!(matches!(
            session.execute(build_plan(&engine, "ROLLBACK")).unwrap(),
            ExecutionResult::TransactionRolledBack,
        ));
    }

    // After rollback only the pre-BEGIN row remains.
    let result = rows(run(&engine, "SELECT id FROM t ORDER BY id"));
    assert_eq!(result, vec![vec![Value::Int(1)]]);
}

/// (QA CRITICAL, 2026-07-09) — the exact reported repro: an
/// UPDATE that does not move any indexed key, rolled back, must leave the committed row
/// reachable through EVERY index path (the PK point-lookup and the secondary index), matching
/// the reference behaviour. The bug made the row vanish from index reads while the heap kept it.
#[test]
fn update_rollback_keeps_rows_reachable_via_index() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE h (id INT PRIMARY KEY, k INT, val INT)",
    );
    run(&engine, "CREATE INDEX h_k ON h (k)");
    run(&engine, "INSERT INTO h VALUES (1, 7, 10)");

    let mut session = Session::new(&engine);
    session.execute(build_plan(&engine, "BEGIN")).unwrap();
    session
        .execute(build_plan(&engine, "UPDATE h SET val = 99 WHERE id = 1"))
        .unwrap();
    session.execute(build_plan(&engine, "ROLLBACK")).unwrap();

    let expected = vec![vec![Value::Int(10)]];
    assert_eq!(
        rows(run(&engine, "SELECT val FROM h WHERE id = 1")),
        expected,
        "PK point-lookup must still find the committed row after the rollback"
    );
    assert_eq!(
        rows(run(&engine, "SELECT val FROM h WHERE k = 7")),
        expected,
        "the secondary-index path must still find the committed row after the rollback"
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, val FROM h")),
        vec![vec![Value::Int(1), Value::Int(10)]],
        "the heap agrees"
    );

    // #2: aggregates over an index predicate must not silently under-count after another
    // row's update aborts (the reported failure: COUNT/SUM dropped the touched row's share).
    run(&engine, "INSERT INTO h VALUES (2, 7, 20), (3, 7, 30)");
    session.execute(build_plan(&engine, "BEGIN")).unwrap();
    session
        .execute(build_plan(&engine, "UPDATE h SET val = 999 WHERE id = 2"))
        .unwrap();
    session.execute(build_plan(&engine, "ROLLBACK")).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*), SUM(val) FROM h WHERE k = 7")),
        vec![vec![Value::Int(3), Value::Int(60)]],
        "index-path aggregates must count every committed row after the abort"
    );

    // #7: an aborted key MOVE must restore both paths — the secondary index under the old
    // key AND the PK point-lookup (whose own key never changed).
    session.execute(build_plan(&engine, "BEGIN")).unwrap();
    session
        .execute(build_plan(&engine, "UPDATE h SET k = 99 WHERE id = 1"))
        .unwrap();
    session.execute(build_plan(&engine, "ROLLBACK")).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT val FROM h WHERE id = 1")),
        vec![vec![Value::Int(10)]],
        "the PK lookup survives an aborted key move"
    );
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM h WHERE k = 7")),
        vec![vec![Value::Int(3)]],
        "the old secondary key still finds every row after the aborted move"
    );
}

#[test]
fn savepoint_rollback_to_and_release_work_end_to_end() {
    // ROLLBACK TO undoes work after a savepoint while keeping earlier work; RELEASE
    // discards the marker so a later ROLLBACK TO that name fails.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");

    let mut session = Session::new(&engine);
    session.execute(build_plan(&engine, "BEGIN")).unwrap();
    session
        .execute(build_plan(&engine, "INSERT INTO t VALUES (1)"))
        .unwrap();
    assert!(matches!(
        session
            .execute(build_plan(&engine, "SAVEPOINT sp1"))
            .unwrap(),
        ExecutionResult::SavepointCreated,
    ));
    session
        .execute(build_plan(&engine, "INSERT INTO t VALUES (2)"))
        .unwrap();

    // Roll back to the savepoint: row 2 is undone, row 1 survives.
    assert!(matches!(
        session
            .execute(build_plan(&engine, "ROLLBACK TO SAVEPOINT sp1"))
            .unwrap(),
        ExecutionResult::RolledBackToSavepoint,
    ));
    session
        .execute(build_plan(&engine, "INSERT INTO t VALUES (3)"))
        .unwrap();

    // Release the savepoint: a later ROLLBACK TO the same name must fail.
    assert!(matches!(
        session
            .execute(build_plan(&engine, "RELEASE SAVEPOINT sp1"))
            .unwrap(),
        ExecutionResult::SavepointReleased,
    ));
    assert!(
        session
            .execute(build_plan(&engine, "ROLLBACK TO SAVEPOINT sp1"))
            .is_err(),
        "ROLLBACK TO a released savepoint should fail",
    );

    session.execute(build_plan(&engine, "COMMIT")).unwrap();

    // Rows 1 and 3 committed; row 2 was rolled back.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]],
    );
}

#[test]
fn set_show_reset_session_variable_round_trip() {
    // /SET stores a session variable, SHOW reads it back, RESET clears it to "".
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);

    // Unset variable shows as the empty string.
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW search_path"))
                .unwrap()
        ),
        vec![vec![Value::Text(String::new())]],
    );

    assert!(matches!(
        session
            .execute(build_plan(&engine, "SET search_path = 'reporting'"))
            .unwrap(),
        ExecutionResult::VariableSet,
    ));
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW search_path"))
                .unwrap()
        ),
        vec![vec![Value::Text("reporting".to_owned())]],
    );

    session
        .execute(build_plan(&engine, "RESET search_path"))
        .unwrap();
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW search_path"))
                .unwrap()
        ),
        vec![vec![Value::Text(String::new())]],
    );
}

#[test]
fn show_reports_builtin_defaults_for_well_known_gucs() {
    // A well-known read-only/session GUC that was never SET reports an honest built-in default
    // (instead of the empty string), so client tooling can read server_version / isolation / etc.
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);

    // transaction_isolation reflects the session default (READ COMMITTED).
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW transaction_isolation"))
                .unwrap()
        ),
        vec![vec![Value::Text("read committed".to_owned())]],
    );
    // client_encoding has a fixed default.
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW client_encoding"))
                .unwrap()
        ),
        vec![vec![Value::Text("UTF8".to_owned())]],
    );
    // server_version is non-empty (the engine's own version).
    let ver = rows(
        session
            .execute(build_plan(&engine, "SHOW server_version"))
            .unwrap(),
    );
    assert!(matches!(&ver[0][0], Value::Text(v) if !v.is_empty()));

    // current_setting() agrees with SHOW for a built-in default (server_encoding never SET).
    assert_eq!(
        rows(
            session
                .execute(build_plan(
                    &engine,
                    "SELECT current_setting('server_encoding')"
                ))
                .unwrap()
        ),
        vec![vec![Value::Text("UTF8".to_owned())]],
    );

    // An explicit SET still overrides the built-in default — for both SHOW and current_setting.
    session
        .execute(build_plan(&engine, "SET client_encoding = 'LATIN1'"))
        .unwrap();
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW client_encoding"))
                .unwrap()
        ),
        vec![vec![Value::Text("LATIN1".to_owned())]],
    );
    assert_eq!(
        rows(
            session
                .execute(build_plan(
                    &engine,
                    "SELECT current_setting('client_encoding')"
                ))
                .unwrap()
        ),
        vec![vec![Value::Text("LATIN1".to_owned())]],
    );

    // An unknown variable still reads back as the empty string.
    assert_eq!(
        rows(
            session
                .execute(build_plan(&engine, "SHOW nonexistent_setting"))
                .unwrap()
        ),
        vec![vec![Value::Text(String::new())]],
    );
}

#[test]
fn savepoint_without_an_active_transaction_is_rejected() {
    // Savepoints only make sense inside an explicit transaction.
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    assert!(matches!(
        session.execute(build_plan(&engine, "SAVEPOINT sp1")),
        Err(nusadb_sql::Error::Unsupported(_)),
    ));
}

#[test]
fn read_only_transaction_blocks_writes_but_allows_reads() {
    // BEGIN ... READ ONLY runs reads but refuses DML; the rejected write leaves no trace.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1)");

    let mut session = Session::new(&engine);
    session
        .execute(build_plan(
            &engine,
            "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY",
        ))
        .unwrap();
    // A read works inside the read-only transaction.
    assert!(matches!(
        session
            .execute(build_plan(&engine, "SELECT id FROM t"))
            .unwrap(),
        ExecutionResult::Rows { .. },
    ));
    // A write is refused.
    assert!(matches!(
        session.execute(build_plan(&engine, "INSERT INTO t VALUES (2)")),
        Err(nusadb_sql::Error::Unsupported(_)),
    ));
    session.execute(build_plan(&engine, "COMMIT")).unwrap();

    // The rejected INSERT never reached storage.
    assert_eq!(rows(run(&engine, "SELECT id FROM t")).len(), 1);
}

#[test]
fn set_transaction_sets_the_session_read_only_default() {
    // SET TRANSACTION READ ONLY makes subsequent auto-commit writes in the session fail.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");

    let mut session = Session::new(&engine);
    assert!(matches!(
        session
            .execute(build_plan(&engine, "SET TRANSACTION READ ONLY"))
            .unwrap(),
        ExecutionResult::TransactionCharacteristicsSet,
    ));
    assert!(matches!(
        session.execute(build_plan(&engine, "INSERT INTO t VALUES (1)")),
        Err(nusadb_sql::Error::Unsupported(_)),
    ));
}

#[test]
fn repeatable_read_indexed_select_is_snapshot_stable_after_concurrent_update() {
    // The secondary index is MVCC-aware: under a frozen snapshot (REPEATABLE READ) an
    // indexed SELECT returns the snapshot's rows even after a concurrent committed UPDATE moves the
    // index entry. The old version's entry is kept (removed only by VACUUM) and the index scan
    // filters every entry by per-tid visibility, so this runs through the index — no seq-scan
    // fallback (that workaround is gone).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT NOT NULL)");
    run(&engine, "CREATE INDEX t_v ON t (v)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");

    // The reader freezes a REPEATABLE READ snapshot and reads through the index.
    let mut reader = Session::new(&engine);
    reader
        .execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    assert_eq!(
        rows(
            reader
                .execute(build_plan(&engine, "SELECT id FROM t WHERE v = 10"))
                .unwrap()
        ),
        vec![vec![Value::Int(1)]],
    );

    // A concurrent auto-commit transaction moves the indexed value and commits.
    run(&engine, "UPDATE t SET v = 20 WHERE id = 1");

    // The reader's frozen snapshot is unchanged: v = 10 still finds row 1, v = 20 finds nothing —
    // the index scan must not leak the concurrent update.
    assert_eq!(
        rows(
            reader
                .execute(build_plan(&engine, "SELECT id FROM t WHERE v = 10"))
                .unwrap()
        ),
        vec![vec![Value::Int(1)]],
        "RR snapshot must still see v=10 for row 1",
    );
    assert_eq!(
        rows(
            reader
                .execute(build_plan(&engine, "SELECT id FROM t WHERE v = 20"))
                .unwrap()
        ),
        Vec::<Vec<Value>>::new(),
        "RR snapshot must not see the post-snapshot v=20",
    );
    reader.execute(build_plan(&engine, "COMMIT")).unwrap();

    // A fresh (READ COMMITTED) transaction sees the committed update — the index scan is correct.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE v = 20")),
        vec![vec![Value::Int(1)]],
    );
}

#[test]
fn drop_table_frees_its_indexes_and_constraints_for_recreation() {
    // A-UR.01: DROP TABLE must also drop the table's PRIMARY KEY/UNIQUE backing index and any
    // secondary index, so re-creating the same table (an idempotent migration / redeploy) does not
    // fail with "index `<t>_pkey` already exists" — the index/constraint namespace is global.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "CREATE INDEX t_v ON t (v)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    run(&engine, "DROP TABLE t");

    // The identical CREATE (and its secondary index) now succeed — both the PK backing index and the
    // secondary index were freed.
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "CREATE INDEX t_v ON t (v)");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(2)]],
    );
}

#[test]
fn repeatable_read_insert_rejects_a_concurrently_committed_duplicate_key() {
    // A-QA1b: a uniqueness / PRIMARY KEY check must see the latest committed state, not the txn's
    // frozen snapshot. A REPEATABLE READ txn that began before another committed the same key would
    // otherwise insert a duplicate (its snapshot never sees the other row).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");

    let mut a = Session::new(&engine);
    a.execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    // Freeze A's snapshot with a read before the concurrent commit.
    rows(a.execute(build_plan(&engine, "SELECT id FROM t")).unwrap());

    // B commits id = 1 after A's snapshot was taken.
    run(&engine, "INSERT INTO t VALUES (1, 10)");

    // A inserts the same key: its frozen snapshot does not see B's row, but the constraint check reads
    // latest-committed, so the duplicate is rejected rather than silently committed.
    assert!(
        a.execute(build_plan(&engine, "INSERT INTO t VALUES (1, 20)"))
            .is_err(),
        "a concurrently committed PRIMARY KEY duplicate must be rejected even under REPEATABLE READ",
    );
    let _ = a.execute(build_plan(&engine, "ROLLBACK"));

    // Exactly one row with id = 1 survives (B's).
    assert_eq!(
        rows(run(&engine, "SELECT v FROM t WHERE id = 1")),
        vec![vec![Value::Int(10)]],
    );
}

#[test]
fn repeatable_read_upsert_do_update_rejects_a_concurrently_committed_duplicate_key() {
    // A-QA1d: an ON CONFLICT DO UPDATE that moves a UNIQUE column onto a value another txn committed
    // after a frozen REPEATABLE READ snapshot must be rejected — mirrors the INSERT/UPDATE/MERGE fix.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code TEXT UNIQUE)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 'a')");

    let mut b = Session::new(&engine);
    b.execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    rows(b.execute(build_plan(&engine, "SELECT id FROM t")).unwrap());

    // A commits a row with code = 'z' after B's snapshot.
    run(&engine, "INSERT INTO t VALUES (2, 'z')");

    // B upserts row 1's code to 'z' — colliding with A's committed row 2 (invisible to B's snapshot).
    assert!(
        b.execute(build_plan(
            &engine,
            "INSERT INTO t VALUES (1, 'a') ON CONFLICT (id) DO UPDATE SET code = 'z'"
        ))
        .is_err(),
        "a DO UPDATE onto a concurrently committed unique value must be rejected under REPEATABLE READ",
    );
    let _ = b.execute(build_plan(&engine, "ROLLBACK"));

    // Exactly one row has code = 'z' (A's row 2).
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE code = 'z'")),
        vec![vec![Value::Int(2)]],
    );
}

#[test]
fn repeatable_read_update_rejects_a_concurrently_committed_duplicate_key() {
    // A-QA1b (UPDATE path): an UPDATE that moves a row onto a key another transaction committed after a
    // frozen REPEATABLE READ snapshot must be rejected — the snapshot-based check cannot see that row.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");

    let mut a = Session::new(&engine);
    a.execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    rows(a.execute(build_plan(&engine, "SELECT id FROM t")).unwrap());

    // B commits id = 2 after A's snapshot.
    run(&engine, "INSERT INTO t VALUES (2, 20)");

    // A moves its row 1 to id = 2 — colliding with B's row, which A's snapshot does not see.
    assert!(
        a.execute(build_plan(&engine, "UPDATE t SET id = 2 WHERE id = 1"))
            .is_err(),
        "an UPDATE onto a concurrently committed key must be rejected under REPEATABLE READ",
    );
    let _ = a.execute(build_plan(&engine, "ROLLBACK"));

    // Both rows still exist with distinct keys.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]],
    );
}

#[test]
fn named_window_resolves_an_over_reference() {
    // A `WINDOW w AS (...)` definition is referenced by `OVER w`.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (g INT, id INT NOT NULL, v INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 1, 10), (1, 2, 20), (2, 1, 5)",
    );
    // A running sum within each group, ordered by id — referenced through the named window `w`.
    let got = rows(run(
        &engine,
        "SELECT id, SUM(v) OVER w FROM t WINDOW w AS (PARTITION BY g ORDER BY id) ORDER BY g, id",
    ));
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(30)],
            vec![Value::Int(1), Value::Int(5)],
        ],
    );
    // An OVER reference to an undefined window is rejected.
    assert!(
        run_try(&engine, "SELECT SUM(v) OVER nope FROM t").is_err(),
        "an OVER reference to an undefined window must be rejected",
    );
}

#[test]
fn derived_table_column_aliases_rename_the_output() {
    // A derived table can rename its output columns: (SELECT ...) AS x(a, b).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    // Rename a subquery's output columns, then reference them by the aliases.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a, b FROM (SELECT id, v FROM t) AS d(a, b) ORDER BY a"
        )),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ],
    );
    // Aliases are usable in expressions too.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a + b FROM (SELECT id, v FROM t) AS d(a, b) ORDER BY a"
        )),
        vec![vec![Value::Int(11)], vec![Value::Int(22)]],
    );
    // The wrong number of aliases is rejected.
    assert!(
        run_try(&engine, "SELECT a FROM (SELECT id, v FROM t) AS d(a)").is_err(),
        "an alias-count mismatch must be rejected",
    );
}

#[test]
fn set_operation_cte_is_inlined() {
    // A non-recursive CTE whose body is a set operation is inlined like a set-op
    // derived table.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (x INT NOT NULL)");
    run(&engine, "CREATE TABLE b (y INT NOT NULL)");
    run(&engine, "INSERT INTO a VALUES (1), (2), (3)");
    run(&engine, "INSERT INTO b VALUES (3), (4)");
    // UNION CTE, referenced as the FROM base (column name from the leftmost branch).
    assert_eq!(
        rows(run(
            &engine,
            "WITH u AS (SELECT x FROM a UNION SELECT y FROM b) SELECT x FROM u ORDER BY x"
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
        ],
    );
    // INTERSECT CTE with an explicit column alias list.
    assert_eq!(
        rows(run(
            &engine,
            "WITH c(n) AS (SELECT x FROM a INTERSECT SELECT y FROM b) SELECT n FROM c"
        )),
        vec![vec![Value::Int(3)]],
    );
    // EXCEPT CTE.
    assert_eq!(
        rows(run(
            &engine,
            "WITH e AS (SELECT x FROM a EXCEPT SELECT y FROM b) SELECT x FROM e ORDER BY x"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]],
    );
}

#[test]
fn cte_body_order_by_and_limit_materializes_top_n() {
    // A plain-SELECT CTE body may carry ORDER BY / LIMIT / OFFSET — the CTE materializes that ordered
    // top-N slice (the planner applies the Sort + Limit), rather than the clause being dropped.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3), (4), (5)");

    // Top 2 by id DESC → {5, 4}; the outer query re-sorts ascending to make the slice observable.
    assert_eq!(
        rows(run(
            &engine,
            "WITH top AS (SELECT id FROM t ORDER BY id DESC LIMIT 2) SELECT id FROM top ORDER BY id",
        )),
        vec![vec![Value::Int(4)], vec![Value::Int(5)]],
    );
    // ORDER BY + LIMIT + OFFSET inside the CTE body: skip 1, take 2 of the ascending order → {2, 3}.
    assert_eq!(
        rows(run(
            &engine,
            "WITH page AS (SELECT id FROM t ORDER BY id LIMIT 2 OFFSET 1) SELECT id FROM page ORDER BY id",
        )),
        vec![vec![Value::Int(2)], vec![Value::Int(3)]],
    );
    // A LIMIT with no ORDER BY still caps the CTE's row count (here, 3 of the 5 rows).
    assert_eq!(
        rows(run(
            &engine,
            "WITH lim AS (SELECT id FROM t LIMIT 3) SELECT COUNT(*) FROM lim",
        )),
        vec![vec![Value::Int(3)]],
    );
}

#[test]
fn cte_referenced_in_a_join_is_inlined() {
    // A non-recursive CTE may be referenced in a JOIN (previously rejected — "reference it as the
    // FROM base"): it inlines like a derived-table join input.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE orders (id INT NOT NULL, cust INT NOT NULL)",
    );
    run(&engine, "CREATE TABLE cust (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO orders VALUES (1, 10), (2, 20), (3, 10)",
    );
    run(
        &engine,
        "INSERT INTO cust VALUES (10, 'a'), (20, 'b'), (30, 'c')",
    );

    // INNER JOIN a base table to a grouped CTE.
    assert_eq!(
        rows(run(
            &engine,
            "WITH big AS (SELECT cust AS c, COUNT(*) AS n FROM orders GROUP BY cust) \
             SELECT cust.name, big.n FROM cust JOIN big ON cust.id = big.c ORDER BY cust.name",
        )),
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(2)], // cust 10 -> 2 orders
            vec![Value::Text("b".to_owned()), Value::Int(1)], // cust 20 -> 1 order
        ],
    );
    // LEFT JOIN with the CTE on the right: an unmatched left row survives with NULLs.
    assert_eq!(
        rows(run(
            &engine,
            "WITH big AS (SELECT cust AS c, COUNT(*) AS n FROM orders GROUP BY cust) \
             SELECT cust.name FROM cust LEFT JOIN big ON cust.id = big.c WHERE big.c IS NULL",
        )),
        vec![vec![Value::Text("c".to_owned())]], // cust 30 has no orders
    );
    // A table alias on the CTE join input re-qualifies its columns (`b2.n`).
    assert_eq!(
        rows(run(
            &engine,
            "WITH big AS (SELECT cust AS c, COUNT(*) AS n FROM orders GROUP BY cust) \
             SELECT b2.n FROM cust JOIN big AS b2 ON cust.id = b2.c WHERE cust.id = 10",
        )),
        vec![vec![Value::Int(2)]],
    );
}

#[test]
fn nested_cte_in_a_cte_body_is_resolved() {
    // A CTE body may itself declare a WITH (a nested CTE) — previously rejected. The analyzer resolves
    // the nested CTE with the enclosing CTEs in scope.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");

    // A plain nested CTE.
    assert_eq!(
        rows(run(
            &engine,
            "WITH outer_c AS (WITH inner_c AS (SELECT v FROM t WHERE v > 10) SELECT v FROM inner_c) \
             SELECT v FROM outer_c ORDER BY v",
        )),
        vec![vec![Value::Int(20)], vec![Value::Int(30)]],
    );
    // The nested CTE references an enclosing sibling CTE (`base`), exercising the scope threading.
    assert_eq!(
        rows(run(
            &engine,
            "WITH base AS (SELECT v FROM t WHERE v >= 20), \
                  wrap AS (WITH pick AS (SELECT v FROM base WHERE v = 30) SELECT v FROM pick) \
             SELECT v FROM wrap",
        )),
        vec![vec![Value::Int(30)]],
    );
}

#[test]
fn set_operation_in_from_is_an_inline_relation() {
    // `(SELECT ... UNION/INTERSECT/EXCEPT ...) AS x(a)` is a derived table whose source is the set
    // operation.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (x INT NOT NULL)");
    run(&engine, "CREATE TABLE b (y INT NOT NULL)");
    run(&engine, "INSERT INTO a VALUES (1), (2), (3)");
    run(&engine, "INSERT INTO b VALUES (3), (4)");
    // UNION derived table, deduplicated; the alias renames the output column, ORDER BY over it.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v.n FROM (SELECT x FROM a UNION SELECT y FROM b) AS v(n) ORDER BY n"
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
        ],
    );
    // UNION ALL keeps duplicates (3 is in both tables): 3 + 2 = 5 rows.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT COUNT(*) FROM (SELECT x FROM a UNION ALL SELECT y FROM b) AS v(n)"
        )),
        vec![vec![Value::Int(5)]],
    );
    // INTERSECT keeps only the common value.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT n FROM (SELECT x FROM a INTERSECT SELECT y FROM b) AS v(n)"
        )),
        vec![vec![Value::Int(3)]],
    );
    // A set-op derived table joins like any other relation.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a.x FROM a JOIN (SELECT y FROM b UNION SELECT x FROM a) AS v(k) ON v.k = a.x \
             ORDER BY a.x"
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ],
    );
    // A WHERE filters the set-op relation.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT n FROM (SELECT x FROM a UNION SELECT y FROM b) AS v(n) WHERE n > 2 ORDER BY n"
        )),
        vec![vec![Value::Int(3)], vec![Value::Int(4)]],
    );
}

#[test]
fn values_in_from_is_an_inline_relation() {
    // `(VALUES (row), ...) AS x(a, b)` is a derived table built from inline rows.
    let engine = BtreeEngine::new();
    // No table needed — VALUES is the source.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a, b FROM (VALUES (1, 10), (2, 20), (3, 30)) AS v(a, b) WHERE a >= 2 ORDER BY a"
        )),
        vec![
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ],
    );
    // Column types unify across rows, and a bare NULL takes the column's inferred type.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT x FROM (VALUES (1), (NULL), (3)) AS v(x) ORDER BY x"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)], vec![Value::Null]],
    );
    // A values relation joins like any other derived table.
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, label TEXT NOT NULL)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 'one'), (2, 'two')");
    assert_eq!(
        rows(run(
            &engine,
            "SELECT t.label, v.n FROM t JOIN (VALUES (1, 100), (2, 200)) AS v(id, n) ON v.id = t.id \
             ORDER BY t.id"
        )),
        vec![
            vec![Value::Text("one".to_owned()), Value::Int(100)],
            vec![Value::Text("two".to_owned()), Value::Int(200)],
        ],
    );
    // A row whose arity differs from the rest is rejected.
    assert!(
        run_try(&engine, "SELECT * FROM (VALUES (1, 2), (3)) AS v(a, b)").is_err(),
        "a VALUES row with the wrong arity must be rejected",
    );
    // An all-NULL column has no inferable type → rejected (the user must cast).
    assert!(
        run_try(&engine, "SELECT * FROM (VALUES (NULL), (NULL)) AS v(x)").is_err(),
        "an entirely-NULL VALUES column must be rejected as untyped",
    );
}

#[test]
fn comma_separated_from_is_an_implicit_cross_join() {
    // `FROM a, b` is an implicit CROSS JOIN; a `WHERE` over both filters the product.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (x INT NOT NULL)");
    run(&engine, "CREATE TABLE b (y INT NOT NULL)");
    run(&engine, "INSERT INTO a VALUES (1), (2)");
    run(&engine, "INSERT INTO b VALUES (10), (20)");
    // Bare comma → full cartesian product (2 × 2 = 4 rows).
    assert_eq!(
        rows(run(&engine, "SELECT a.x, b.y FROM a, b ORDER BY a.x, b.y")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(1), Value::Int(20)],
            vec![Value::Int(2), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ],
    );
    // A WHERE over both tables filters the product — the classic implicit-join idiom.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a.x, b.y FROM a, b WHERE b.y = a.x * 10 ORDER BY a.x"
        )),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ],
    );
    // Three-way comma product (2 × 2 × 1 = 4 rows).
    run(&engine, "CREATE TABLE c (z INT NOT NULL)");
    run(&engine, "INSERT INTO c VALUES (100)");
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM a, b, c")),
        vec![vec![Value::Int(4)]],
    );
    // A comma item mixed with an explicit JOIN stays rejected (outer-join precedence is subtle).
    assert!(
        run_try(&engine, "SELECT * FROM a, b JOIN c ON c.z = b.y").is_err(),
        "comma mixed with an explicit JOIN must be rejected, not silently reshaped",
    );
}

#[test]
fn values_default_keyword_fills_each_cell_from_the_column() {
    // A `DEFAULT` cell in a `VALUES` row takes that column's default / serial / NULL fill,
    // exactly as an omitted column would; a concrete cell is used as written.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id SERIAL PRIMARY KEY, c INT DEFAULT 7, n INT, lbl TEXT NOT NULL)",
    );
    // Row 1: every fillable column DEFAULT — id→nextval(1), c→7, n→NULL (nullable, no default).
    run(
        &engine,
        "INSERT INTO t (id, c, n, lbl) VALUES (DEFAULT, DEFAULT, DEFAULT, 'a')",
    );
    // Row 2: DEFAULT serial alongside concrete values — id→nextval(2).
    run(
        &engine,
        "INSERT INTO t (id, c, n, lbl) VALUES (DEFAULT, 99, 5, 'b')",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, c, n, lbl FROM t ORDER BY id")),
        vec![
            vec![
                Value::Int(1),
                Value::Int(7),
                Value::Null,
                Value::Text("a".to_owned()),
            ],
            vec![
                Value::Int(2),
                Value::Int(99),
                Value::Int(5),
                Value::Text("b".to_owned()),
            ],
        ],
    );
    // `DEFAULT` on a NOT NULL column with no default is a violation, like an explicit NULL.
    assert!(
        run_try(&engine, "INSERT INTO t (c, n, lbl) VALUES (1, 2, DEFAULT)").is_err(),
        "DEFAULT on a NOT NULL column without a default must be rejected",
    );
}

#[test]
fn values_default_on_a_generated_always_identity_is_accepted() {
    // `DEFAULT` is the canonical way to fill a GENERATED ALWAYS AS IDENTITY column — it must be
    // accepted (the column auto-generates) while an explicit value is still rejected (#9a).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE g (id INT GENERATED ALWAYS AS IDENTITY, x INT NOT NULL)",
    );
    // Naming the identity column with a DEFAULT cell is allowed: it auto-generates.
    run(&engine, "INSERT INTO g (id, x) VALUES (DEFAULT, 5)");
    assert_eq!(
        rows(run(&engine, "SELECT id, x FROM g")),
        vec![vec![Value::Int(1), Value::Int(5)]],
    );
    // A concrete value for the identity column is still rejected.
    assert!(
        run_try(&engine, "INSERT INTO g (id, x) VALUES (10, 5)").is_err(),
        "an explicit value for a GENERATED ALWAYS column must be rejected",
    );
}

#[test]
fn describe_table_lists_its_columns() {
    // DESCRIBE / DESC <table> is an alias for SHOW COLUMNS — it lists (column, type, nullable).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");
    let expected = vec![
        vec![
            Value::Text("id".to_owned()),
            Value::Text("INT".to_owned()),
            Value::Bool(false),
        ],
        vec![
            Value::Text("name".to_owned()),
            Value::Text("TEXT".to_owned()),
            Value::Bool(true),
        ],
    ];
    assert_eq!(rows(run(&engine, "DESCRIBE t")), expected);
    assert_eq!(rows(run(&engine, "DESC t")), expected);
}

#[test]
fn alter_table_rename_to_renames_the_table() {
    // ALTER TABLE ... RENAME TO renames the table in the catalog (no row rewrite): the old name is
    // freed and the data is reachable under the new name.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE old_name (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO old_name VALUES (1, 10)");
    run(&engine, "ALTER TABLE old_name RENAME TO new_name");

    assert!(
        run_try(&engine, "SELECT id FROM old_name").is_err(),
        "the old table name is gone after RENAME",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM new_name")),
        vec![vec![Value::Int(1), Value::Int(10)]],
    );
    // The freed old name can be reused; renaming onto an existing table is rejected.
    run(&engine, "CREATE TABLE old_name (x INT)");
    assert!(
        run_try(&engine, "ALTER TABLE new_name RENAME TO old_name").is_err(),
        "renaming onto an existing table name must be rejected",
    );
}

#[test]
fn alter_table_add_column_default_backfills_existing_rows() {
    // `ADD COLUMN ... DEFAULT <expr>` fills existing rows with the
    // default (parity with the reference engine), not NULL, and later inserts that omit the column get it too.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");

    run(&engine, "ALTER TABLE t ADD COLUMN b INT DEFAULT 9");
    assert_eq!(
        rows(run(&engine, "SELECT id, b FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(9)],
            vec![Value::Int(2), Value::Int(9)],
        ],
        "existing rows are backfilled with the default, not left NULL",
    );
    // A later insert omitting the new column still gets the default.
    run(&engine, "INSERT INTO t (id, v) VALUES (3, 30)");
    assert_eq!(
        rows(run(&engine, "SELECT b FROM t WHERE id = 3")),
        vec![vec![Value::Int(9)]],
        "the persisted default applies to later inserts too",
    );

    // A NOT NULL column WITH a default backfills fine; without one it is rejected on a non-empty
    // table (parity with the reference engine).
    run(&engine, "ALTER TABLE t ADD COLUMN c INT NOT NULL DEFAULT 7");
    assert_eq!(
        rows(run(&engine, "SELECT c FROM t WHERE id = 1")),
        vec![vec![Value::Int(7)]],
    );
    assert!(
        run_try(&engine, "ALTER TABLE t ADD COLUMN d INT NOT NULL").is_err(),
        "NOT NULL ADD COLUMN with no default on a non-empty table must be rejected",
    );
    // A nullable column with no default still backfills NULL, as before.
    run(&engine, "ALTER TABLE t ADD COLUMN e TEXT");
    assert_eq!(
        rows(run(&engine, "SELECT e FROM t WHERE id = 1")),
        vec![vec![Value::Null]],
    );

    // DROP then re-ADD the same column name without a default: the stale default must not linger
    // (the catalog keys by column name).
    run(&engine, "ALTER TABLE t DROP COLUMN b");
    run(&engine, "ALTER TABLE t ADD COLUMN b INT");
    assert_eq!(
        rows(run(&engine, "SELECT b FROM t WHERE id = 1")),
        vec![vec![Value::Null]],
        "a re-added column with no default must not inherit the dropped column's default",
    );
}

#[test]
fn inner_join_on_clause_pushdown_preserves_results() {
    // Pushing a single-side ON conjunct below an INNER join is a
    // semantics-preserving optimization — the result must be identical to evaluating it as a
    // join residual. Values are hand-computed so a wrong pushdown (dropped/misfiltered rows or a
    // right-side column-shift bug) fails loudly.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE orders (id INT NOT NULL, customer_id INT)",
    );
    // customer 100 has ids {1,2,3}; customer 200 has ids {4,5}.
    run(
        &engine,
        "INSERT INTO orders VALUES (1,100),(2,100),(3,100),(4,200),(5,200)",
    );

    // The QA self-join shape: same customer, o1.id < o2.id. Same-customer ordered pairs:
    // customer 100 → (1,2),(1,3),(2,3) = 3; customer 200 → (4,5) = 1; total 4.
    let base = "SELECT count(*) FROM orders o1 JOIN orders o2 \
                ON o1.customer_id = o2.customer_id AND o1.id < o2.id";
    assert_eq!(rows(run(&engine, base)), vec![vec![Value::Int(4)]]);

    // Left-side pushdown: add `o1.id < 3` (only o1 ids {1,2} qualify). Customer 100 keeps
    // (1,2),(1,3),(2,3); customer 200's (4,5) has o1.id=4 ≥ 3 → dropped. Total 3.
    assert_eq!(
        rows(run(&engine, &format!("{base} AND o1.id < 3"))),
        vec![vec![Value::Int(3)]],
    );

    // Right-side pushdown (exercises the column shift): add `o2.id > 3` (only o2 ids {4,5}).
    // Customer 100's pairs have o2 ∈ {2,3} → none; customer 200's (4,5) has o2.id=5 > 3 → 1.
    assert_eq!(
        rows(run(&engine, &format!("{base} AND o2.id > 3"))),
        vec![vec![Value::Int(1)]],
    );

    // Both sides pushed at once plus a cross-side residual still standing (o1.id < o2.id):
    // o1.id < 4 AND o2.id > 1. Customer 100 → (1,2),(1,3),(2,3); customer 200 → (4,5) has
    // o1.id=4 not < 4 → dropped. Total 3.
    assert_eq!(
        rows(run(&engine, &format!("{base} AND o1.id < 4 AND o2.id > 1"))),
        vec![vec![Value::Int(3)]],
    );
}

#[test]
#[ignore = "perf sanity check; run with --ignored --nocapture"]
fn selfjoin_pushdown_perf_sanity() {
    // Without ON-clause pushdown, all `n` o1 rows probe the hash table (n × fan-out intermediate
    // pairs); with it, only the `o1.id < 500` rows do (500 × fan-out). At n=60k / 600 customers
    // (~100 fan-out) that is ~6M pairs vs ~50k — the un-pushed form takes seconds, the pushed one
    // milliseconds. This test just has to COMPLETE quickly (and return the right count).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE orders (id INT NOT NULL, customer_id INT)",
    );
    let n = 60_000u32;
    let customers = 600u32;
    for batch in 0..(n / 1000) {
        let values: Vec<String> = (0..1000)
            .map(|j| {
                let i = batch * 1000 + j;
                format!("({i},{})", i % customers)
            })
            .collect();
        run(
            &engine,
            &format!("INSERT INTO orders VALUES {}", values.join(",")),
        );
    }
    let q = "SELECT count(*) FROM orders o1 JOIN orders o2 \
             ON o1.customer_id = o2.customer_id AND o1.id < o2.id AND o1.id < 500";
    let start = std::time::Instant::now();
    let got = rows(run(&engine, q));
    eprintln!("self-join with pushdown: {got:?} in {:?}", start.elapsed());
    // 500 qualifying o1 rows, each pairs with the ~100 same-customer rows that have a larger id.
    assert!(matches!(got[0][0], Value::Int(n) if n > 0));
}

#[test]
fn left_join_on_clause_condition_does_not_drop_rows() {
    // An outer join must NOT push a single-side ON conjunct down — every left row is preserved
    // (NULL-padded when the ON predicate fails), so pushing a preserved-side filter would wrongly
    // vanish rows. `left_a` = { (1,10), (2,10), (3,20) }, `right_b` = { (10,'x'), (20,'y') }.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE left_a (id INT NOT NULL, k INT)");
    run(&engine, "CREATE TABLE right_b (k INT NOT NULL, label TEXT)");
    run(&engine, "INSERT INTO left_a VALUES (1,10),(2,10),(3,20)");
    run(&engine, "INSERT INTO right_b VALUES (10,'x'),(20,'y')");

    // `a.id < 3` is on the PRESERVED (left) side of a LEFT join: all 3 left rows must still
    // appear; the condition only decides which get a match vs a NULL pad. Row 3 (id=3, k=20)
    // fails a.id<3 → NULL-padded; rows 1,2 match 'x'.
    let mut got = rows(run(
        &engine,
        "SELECT a.id, b.label FROM left_a a LEFT JOIN right_b b \
         ON a.k = b.k AND a.id < 3 ORDER BY a.id",
    ));
    got.sort_by_key(|r| format!("{r:?}"));
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1), Value::Text("x".to_owned())],
            vec![Value::Int(2), Value::Text("x".to_owned())],
            vec![Value::Int(3), Value::Null],
        ],
        "a LEFT join keeps every left row; the ON condition only gates the match",
    );
}

#[test]
fn update_set_and_where_accept_an_uncorrelated_subquery() {
    // An uncorrelated subquery in an UPDATE's SET value or WHERE is evaluated once and applied to the
    // matched rows (previously a subquery here was rejected at eval).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, x INT)");
    run(&engine, "CREATE TABLE src (v INT)");
    run(&engine, "CREATE TABLE keys (k INT)");
    run(&engine, "INSERT INTO t VALUES (1, 0), (2, 0), (3, 0)");
    run(&engine, "INSERT INTO src VALUES (10), (40), (20)");
    run(&engine, "INSERT INTO keys VALUES (1), (3)");

    // SET from a scalar subquery (max = 40); WHERE filtered by an IN subquery (rows 1 and 3).
    run(
        &engine,
        "UPDATE t SET x = (SELECT max(v) FROM src) WHERE id IN (SELECT k FROM keys)",
    );
    assert_eq!(
        rows(run(&engine, "SELECT id, x FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(40)],
            vec![Value::Int(2), Value::Int(0)],
            vec![Value::Int(3), Value::Int(40)],
        ],
    );

    // DELETE ... WHERE id IN (subquery) is likewise accepted.
    run(&engine, "DELETE FROM t WHERE id IN (SELECT k FROM keys)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(2)]],
    );
}

#[test]
fn a_correlated_subquery_in_update_set_is_rejected_not_silently_nulled() {
    // A correlated subquery (referencing the target row) has no per-row resolution here, so it must
    // be rejected at eval rather than resolved against an unbound row to a wrong NULL.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, x INT)");
    run(&engine, "CREATE TABLE src (sid INT, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 0)");
    run(&engine, "INSERT INTO src VALUES (1, 100)");
    assert!(
        run_try(
            &engine,
            "UPDATE t SET x = (SELECT v FROM src WHERE src.sid = t.id)"
        )
        .is_err(),
        "a correlated subquery in SET must be rejected, not silently set to NULL",
    );
    // The row is unchanged (the failed statement made no write).
    assert_eq!(
        rows(run(&engine, "SELECT x FROM t")),
        vec![vec![Value::Int(0)]],
    );
}

#[test]
fn drop_table_drops_its_hnsw_vector_index() {
    // A-UR.01c: a USING hnsw vector index lives in the SQL-layer catalog (not the engine index
    // namespace); DROP TABLE must remove it so a later same-named table + index does not collide with
    // the orphaned declaration.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE items (id INT NOT NULL, embedding VECTOR(3))",
    );
    run(
        &engine,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );
    run(&engine, "DROP TABLE items");
    // Recreate the table and the same-named vector index — succeeds only if the orphan was cleaned.
    run(
        &engine,
        "CREATE TABLE items (id INT NOT NULL, embedding VECTOR(3))",
    );
    run(
        &engine,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );
}

#[test]
fn drop_table_rejects_a_parent_still_referenced_by_a_foreign_key() {
    // A-UR.01b: dropping a table another table's FOREIGN KEY references is rejected (RESTRICT) so the
    // FK is not left silently dangling. Dropping the child first frees the parent.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT PRIMARY KEY, pid INT REFERENCES parent(id))",
    );
    assert!(
        run_try(&engine, "DROP TABLE parent").is_err(),
        "dropping a referenced parent table must be rejected",
    );
    // After dropping the child, the parent drops cleanly.
    run(&engine, "DROP TABLE child");
    run(&engine, "DROP TABLE parent");
    assert!(run_try(&engine, "SELECT id FROM parent").is_err());
}

#[test]
fn drop_table_drops_its_foreign_key_so_a_child_table_can_be_recreated() {
    // A-UR.01: a table declaring a FOREIGN KEY must remain droppable, and the drop must remove the FK
    // (and its child-side index) so re-creating the child does not collide on the auto-named FK/index.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT PRIMARY KEY, pid INT REFERENCES parent(id))",
    );
    run(&engine, "DROP TABLE child");
    // Re-creating the child with the same FOREIGN KEY succeeds (the FK and its index were freed).
    run(
        &engine,
        "CREATE TABLE child (id INT PRIMARY KEY, pid INT REFERENCES parent(id))",
    );
    run(&engine, "INSERT INTO parent VALUES (1)");
    run(&engine, "INSERT INTO child VALUES (10, 1)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM child")),
        vec![vec![Value::Int(10)]],
    );
}

#[test]
fn a_rolled_back_create_table_frees_its_serial_sequence() {
    // Deep-gate #9d: rolling back a CREATE TABLE with a SERIAL column must not leave its backing
    // sequence behind, or re-creating the same table fails with "sequence already exists".
    let engine = BtreeEngine::new();
    let mut s = Session::new(&engine);
    s.execute(build_plan(&engine, "BEGIN")).unwrap();
    s.execute(build_plan(
        &engine,
        "CREATE TABLE t (id SERIAL PRIMARY KEY, v INT)",
    ))
    .unwrap();
    s.execute(build_plan(&engine, "ROLLBACK")).unwrap();

    // The table and its backing sequence are both gone, so the identical CREATE succeeds and the
    // SERIAL starts fresh at 1.
    run(&engine, "CREATE TABLE t (id SERIAL PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t (v) VALUES (10)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t")),
        vec![vec![Value::Int(1)]],
    );
}

#[test]
fn concurrent_update_to_same_unique_key_aborts_the_second_writer() {
    // Deep-gate #7 /: two transactions that UPDATE different rows to the *same* UNIQUE key each
    // scan a snapshot blind to the other; without a key lock both would commit and leave a duplicate.
    // The matched-UPDATE path now takes the same no-wait key lock the INSERT path does, so the second
    // writer aborts instead of committing the duplicate.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code TEXT UNIQUE)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");

    let mut a = Session::new(&engine);
    a.execute(build_plan(&engine, "BEGIN")).unwrap();
    a.execute(build_plan(&engine, "UPDATE t SET code = 'x' WHERE id = 1"))
        .unwrap();

    // B, concurrently, tries to move its row to the same key. The key lock A holds forces B to abort.
    let mut b = Session::new(&engine);
    b.execute(build_plan(&engine, "BEGIN")).unwrap();
    assert!(
        b.execute(build_plan(&engine, "UPDATE t SET code = 'x' WHERE id = 2"))
            .is_err(),
        "the second writer to the same UNIQUE key must abort on the key lock",
    );

    // A commits; B is rolled back. The table holds exactly one row with code = 'x'.
    a.execute(build_plan(&engine, "COMMIT")).unwrap();
    let _ = b.execute(build_plan(&engine, "ROLLBACK"));
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE code = 'x'")),
        vec![vec![Value::Int(1)]],
    );
}

#[test]
fn a_cancelled_statement_aborts_at_the_scan_boundary() {
    // A tripped cancel token makes the executor abort a scanning statement cooperatively
    // (the statement-timeout timer and an out-of-band cancel both trip this same token).
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3)");

    let token = Arc::new(AtomicBool::new(true)); // already tripped
    {
        let _guard = nusadb_sql::cancel::scope(Arc::clone(&token));
        assert!(
            matches!(
                run_try(&engine, "SELECT id FROM t"),
                Err(nusadb_sql::Error::Cancelled),
            ),
            "a scanning statement must abort while the cancel token is tripped",
        );
    }
    // Once the guard drops, queries run normally again.
    assert_eq!(rows(run(&engine, "SELECT id FROM t ORDER BY id")).len(), 3);
}

#[test]
fn one_shot_execute_rejects_explicit_transaction_control() {
    let engine = BtreeEngine::new();
    let stmt = parse("BEGIN").expect("parse");
    let logical = analyze(stmt, &EngineCatalog(&engine)).expect("analyze");
    let physical = plan(logical);
    let result = execute(physical, &engine);
    assert!(matches!(result, Err(nusadb_sql::Error::Unsupported(_))));
}

#[test]
fn case_insensitive_identifier_works_end_to_end() {
    let engine = BtreeEngine::new();
    // Mixed-case in CREATE; querying with different casing must still work.
    run(&engine, "CREATE TABLE Users (ID INT NOT NULL, Name TEXT)");
    run(&engine, "INSERT INTO USERS VALUES (1, 'alice')");
    run(&engine, "INSERT INTO users (id, name) VALUES (2, 'bob')");

    assert_eq!(
        rows(run(&engine, "SELECT ID, NAME FROM Users ORDER BY id")).len(),
        2,
    );
}

#[test]
fn scalar_aggregates_fold_the_whole_table() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE nums (id INT NOT NULL, v INT, f FLOAT)",
    );
    run(&engine, "INSERT INTO nums VALUES (1, 10, 1.5)");
    run(&engine, "INSERT INTO nums VALUES (2, 20, 2.5)");
    run(&engine, "INSERT INTO nums VALUES (3, 30, NULL)");
    run(&engine, "INSERT INTO nums VALUES (4, NULL, 4.0)");

    // One output row, one column per aggregate, in projection order.
    // COUNT(*) counts every row; COUNT(v) skips the NULL; SUM(Int)→Int,
    // AVG(Int)→exact NUMERIC, MIN/MAX keep the argument's type.
    let result = rows(run(
        &engine,
        "SELECT COUNT(*), COUNT(v), SUM(v), AVG(v), MIN(v), MAX(v) FROM nums",
    ));
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0],
        vec![
            Value::Int(4),  // COUNT(*)
            Value::Int(3),  // COUNT(v) — row 4 is NULL
            Value::Int(60), // SUM(v) = 10+20+30
            // AVG(Int) is exact NUMERIC = 60/3 = 20 (not a lossy f64).
            Value::Numeric(nusadb_sql::numeric::Decimal::parse("20").unwrap()),
            Value::Int(10), // MIN(v)
            Value::Int(30), // MAX(v)
        ],
    );

    // SUM over the FLOAT column skips its NULL and stays Float.
    assert_eq!(
        rows(run(&engine, "SELECT SUM(f) FROM nums"))[0][0],
        Value::Float(8.0), // 1.5 + 2.5 + 4.0
    );
}

#[test]
fn b133_aggregate_distinct_dedupes_per_group() {
    // DISTINCT inside an aggregate folds each distinct non-NULL value once.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT, g INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10, 100)");
    run(&engine, "INSERT INTO t VALUES (2, 10, 100)"); // dup v within g=100
    run(&engine, "INSERT INTO t VALUES (3, 20, 100)");
    run(&engine, "INSERT INTO t VALUES (4, NULL, 200)"); // NULL skipped
    run(&engine, "INSERT INTO t VALUES (5, 20, 200)");

    // Scalar: distinct non-NULL v = {10, 20}. COUNT(v) still counts every non-NULL (4).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT COUNT(DISTINCT v), SUM(DISTINCT v), COUNT(v), COUNT(*) FROM t",
        ))[0],
        vec![
            Value::Int(2),  // COUNT(DISTINCT v) — {10, 20}
            Value::Int(30), // SUM(DISTINCT v) — 10 + 20
            Value::Int(4),  // COUNT(v) — every non-NULL (10,10,20,20)
            Value::Int(5),  // COUNT(*)
        ],
    );

    // Per group: g=100 has {10,10,20}→2 distinct; g=200 has {NULL,20}→1 distinct.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT g, COUNT(DISTINCT v) FROM t GROUP BY g ORDER BY g",
        )),
        vec![
            vec![Value::Int(100), Value::Int(2)],
            vec![Value::Int(200), Value::Int(1)],
        ],
    );
}

#[test]
fn count_star_fast_path_matches_the_visible_row_count() {
    // `COUNT(*)` over a plain scan takes the fast-path (count visible tuples without decoding rows).
    // It must equal the exact visible row count, and stay identical to the fold path for shapes that
    // do not qualify (a WHERE filter, `COUNT(col)` skipping NULLs, `COUNT(*)` over a join).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    // Empty table → 0.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM t")),
        vec![vec![Value::Int(0)]]
    );
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, NULL), (3, 30)");
    // Fast-path: counts every row including the NULL.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM t")),
        vec![vec![Value::Int(3)]]
    );
    // Two COUNT(*) slots both get the same count.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*), COUNT(*) FROM t")),
        vec![vec![Value::Int(3), Value::Int(3)]]
    );
    // Fold path (not the fast-path): COUNT(v) skips the NULL; a filtered COUNT(*) is exact.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(v) FROM t")),
        vec![vec![Value::Int(2)]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM t WHERE v > 15")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn count_star_fast_path_reflects_uncommitted_changes_in_the_same_txn() {
    // The fast-path must honor MVCC exactly like the scan: a transaction's own uncommitted INSERT and
    // DELETE change what its later COUNT(*) sees, and a rolled-back change is not counted afterwards.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3)");

    let mut txn = Session::new(&engine);
    txn.execute(build_plan(&engine, "BEGIN")).unwrap();
    txn.execute(build_plan(&engine, "INSERT INTO t VALUES (4), (5)"))
        .unwrap();
    txn.execute(build_plan(&engine, "DELETE FROM t WHERE id = 1"))
        .unwrap();
    // The txn's own snapshot: 3 - 1 + 2 = 4 visible rows.
    assert_eq!(
        rows(
            txn.execute(build_plan(&engine, "SELECT COUNT(*) FROM t"))
                .unwrap()
        ),
        vec![vec![Value::Int(4)]]
    );
    txn.execute(build_plan(&engine, "ROLLBACK")).unwrap();
    // After rollback the committed count is unchanged.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM t")),
        vec![vec![Value::Int(3)]]
    );
}

#[test]
fn aggregate_arithmetic_in_projection() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");

    // An aggregate buried inside a binary expression is still folded.
    assert_eq!(
        rows(run(&engine, "SELECT SUM(v) + 1 FROM t"))[0][0],
        Value::Int(61),
    );
}

#[test]
fn aggregates_compose_inside_non_binary_expressions() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");

    // CAST wrapping the aggregate itself (not just a binary op): SUM(v)=60 → 60.0.
    assert_eq!(
        rows(run(&engine, "SELECT CAST(SUM(v) AS FLOAT) FROM t"))[0][0],
        Value::Float(60.0),
    );
    // COALESCE wrapping an aggregate.
    assert_eq!(
        rows(run(&engine, "SELECT COALESCE(MIN(v), 0) FROM t"))[0][0],
        Value::Int(10),
    );
    // CASE whose predicate is an aggregate.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT CASE WHEN COUNT(*) > 2 THEN 'many' ELSE 'few' END FROM t",
        ))[0][0],
        Value::Text("many".to_owned()),
    );
    // IS NULL applied to an aggregate result.
    assert_eq!(
        rows(run(&engine, "SELECT SUM(v) IS NULL FROM t"))[0][0],
        Value::Bool(false),
    );
}

#[test]
fn aggregates_over_empty_input() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");

    // No rows: COUNT collapses to 0, every other aggregate is NULL.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT COUNT(*), COUNT(v), SUM(v), AVG(v), MIN(v), MAX(v) FROM t",
        ))[0],
        vec![
            Value::Int(0),
            Value::Int(0),
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
        ],
    );
}

#[test]
fn min_max_preserve_text_type() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'charlie'), (2, 'alice'), (3, 'bob')",
    );

    assert_eq!(
        rows(run(&engine, "SELECT MIN(name), MAX(name) FROM t"))[0],
        vec![
            Value::Text("alice".to_owned()),
            Value::Text("charlie".to_owned()),
        ],
    );
}

#[test]
fn group_by_single_key_with_aggregates() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT, amount INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO sales VALUES ('west', 10), ('west', 20), \
         ('east', 5), ('east', 15), ('east', 30)",
    );

    // One row per group: [group key ++ aggregate results], here ORDER BY region.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT region, COUNT(*), SUM(amount), MIN(amount), MAX(amount) \
             FROM sales GROUP BY region ORDER BY region",
        )),
        vec![
            vec![
                Value::Text("east".to_owned()),
                Value::Int(3),
                Value::Int(50),
                Value::Int(5),
                Value::Int(30),
            ],
            vec![
                Value::Text("west".to_owned()),
                Value::Int(2),
                Value::Int(30),
                Value::Int(10),
                Value::Int(20),
            ],
        ],
    );

    // AVG(Int) returns exact NUMERIC per group (sum/count), not a lossy f64.
    let avg = |sum: i64, n: i64| {
        Value::Numeric(
            nusadb_sql::numeric::Decimal::from_i64(sum)
                .checked_div(&nusadb_sql::numeric::Decimal::from_i64(n))
                .unwrap(),
        )
    };
    assert_eq!(
        rows(run(
            &engine,
            "SELECT region, AVG(amount) FROM sales GROUP BY region ORDER BY region",
        )),
        vec![
            vec![Value::Text("east".to_owned()), avg(50, 3)],
            vec![Value::Text("west".to_owned()), avg(30, 2)],
        ],
    );
}

#[test]
fn group_by_having_filters_groups() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT, amount INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO sales VALUES ('west', 10), ('west', 20), \
         ('east', 5), ('east', 15), ('east', 30)",
    );

    // HAVING on a projected aggregate.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT region, SUM(amount) FROM sales \
             GROUP BY region HAVING SUM(amount) > 40 ORDER BY region",
        )),
        vec![vec![Value::Text("east".to_owned()), Value::Int(50)]],
    );

    // HAVING may reference an aggregate that is not in the projection.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT region FROM sales GROUP BY region HAVING COUNT(*) > 2 ORDER BY region",
        )),
        vec![vec![Value::Text("east".to_owned())]],
    );
}

#[test]
fn group_by_expression_groups_by_the_computed_key() {
    // E (large verticals) — GROUP BY on an arbitrary expression: rows are partitioned by the value
    // of `a + b`, and the projection's matching `a + b` collapses to the grouped key.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT NOT NULL)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 1), (2, 0), (3, 3), (0, 4)",
    );

    // a + b = 2, 2, 6, 4 → keys {2: 2 rows, 4: 1 row, 6: 1 row}.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a + b AS s, COUNT(*) FROM t GROUP BY a + b ORDER BY s",
        )),
        vec![
            vec![Value::Int(2), Value::Int(2)],
            vec![Value::Int(4), Value::Int(1)],
            vec![Value::Int(6), Value::Int(1)],
        ],
    );

    // A bare column that is neither grouped nor aggregated is still rejected.
    assert!(
        run_try(&engine, "SELECT a, COUNT(*) FROM t GROUP BY a + b").is_err(),
        "ungrouped column `a` must be rejected",
    );
}

#[test]
fn group_by_positional_reference_groups_by_the_nth_output_column() {
    // GROUP BY <n> is a positional reference to the Nth output column (like ORDER BY <n>).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (g INT NOT NULL, v INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (1, 20), (2, 30)");

    // GROUP BY 1 groups by the first output column (g) — equivalent to GROUP BY g.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT g, SUM(v) FROM t GROUP BY 1 ORDER BY g"
        )),
        vec![
            vec![Value::Int(1), Value::Int(30)],
            vec![Value::Int(2), Value::Int(30)],
        ]
    );
    // The position resolves to the underlying expression: GROUP BY 1 here means GROUP BY g % 2.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT g % 2 AS parity, COUNT(*) FROM t GROUP BY 1 ORDER BY parity",
        )),
        vec![
            vec![Value::Int(0), Value::Int(1)], // g = 2 -> 0
            vec![Value::Int(1), Value::Int(2)], // g = 1 (x2) -> 1
        ]
    );
    // A position out of range, or one naming an aggregate output column, is a loud error.
    assert!(run_try(&engine, "SELECT g, COUNT(*) FROM t GROUP BY 5").is_err());
    assert!(run_try(&engine, "SELECT COUNT(*) FROM t GROUP BY 1").is_err());
}

#[test]
fn group_by_null_key_groups_together() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (k TEXT, v INT NOT NULL)");
    run(
        &engine,
        "INSERT INTO t VALUES ('a', 1), (NULL, 2), ('a', 3), (NULL, 4)",
    );

    // All NULL keys collapse into one group; NULL sorts last in ASC.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT k, COUNT(*), SUM(v) FROM t GROUP BY k ORDER BY k",
        )),
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(2), Value::Int(4)],
            vec![Value::Null, Value::Int(2), Value::Int(6)],
        ],
    );
}

#[test]
fn group_by_multiple_keys() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (a INT NOT NULL, b INT NOT NULL, v INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 1, 10), (1, 1, 20), (1, 2, 30), (2, 1, 40)",
    );

    assert_eq!(
        rows(run(
            &engine,
            "SELECT a, b, SUM(v) FROM t GROUP BY a, b ORDER BY a, b",
        )),
        vec![
            vec![Value::Int(1), Value::Int(1), Value::Int(30)],
            vec![Value::Int(1), Value::Int(2), Value::Int(30)],
            vec![Value::Int(2), Value::Int(1), Value::Int(40)],
        ],
    );
}

#[test]
fn group_by_only_is_distinct_like() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT, amount INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO sales VALUES ('west', 10), ('east', 5), ('east', 30)",
    );

    // GROUP BY with no aggregate in the projection = distinct group keys.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT region FROM sales GROUP BY region ORDER BY region",
        )),
        vec![
            vec![Value::Text("east".to_owned())],
            vec![Value::Text("west".to_owned())],
        ],
    );
}

#[test]
fn group_by_rejects_ungrouped_column() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT, amount INT NOT NULL)",
    );
    // `amount` is neither a group key nor wrapped in an aggregate.
    let stmt = parse("SELECT region, amount FROM sales GROUP BY region").expect("parse");
    assert!(analyze(stmt, &EngineCatalog(&engine)).is_err());
}

#[test]
fn having_without_aggregation_is_rejected() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    let stmt = parse("SELECT id FROM t HAVING id > 0").expect("parse");
    assert!(analyze(stmt, &EngineCatalog(&engine)).is_err());
}

#[test]
fn explain_shows_group_aggregate() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT, amount INT NOT NULL)",
    );
    let lines: Vec<String> = rows(run(
        &engine,
        "EXPLAIN SELECT region, SUM(amount) FROM sales GROUP BY region HAVING SUM(amount) > 1",
    ))
    .into_iter()
    .map(|row| match row.into_iter().next() {
        Some(Value::Text(s)) => s,
        other => panic!("expected plan text, got {other:?}"),
    })
    .collect();
    let joined = lines.join("\n");
    // One group key (region); SUM(amount) appears in both the projection and
    // HAVING, so two aggregate calls are registered (no CSE dedup yet).
    assert!(
        joined.contains("GroupAggregate (1 key(s), 2 aggregate(s))"),
        "missing GroupAggregate: {joined}",
    );
    assert!(joined.contains("Filter"), "missing HAVING filter: {joined}");
}

#[test]
fn drop_table_then_select_is_rejected() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doomed (a INT)");
    assert!(matches!(
        run(&engine, "DROP TABLE doomed"),
        ExecutionResult::Dropped,
    ));
    // After the drop the analyzer must reject the next SELECT — the catalog
    // really is the storage engine's, not a stale snapshot.
    let stmt = parse("SELECT * FROM doomed").expect("parse");
    assert!(analyze(stmt, &EngineCatalog(&engine)).is_err());
}

/// Seed `users` (3 rows) and `orders` (3 rows; carol has none) for join tests.
fn seed_join_tables(engine: &BtreeEngine) {
    run(engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)");
    run(
        engine,
        "CREATE TABLE orders (id INT NOT NULL, user_id INT NOT NULL, total INT NOT NULL)",
    );
    run(
        engine,
        "INSERT INTO users VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')",
    );
    run(
        engine,
        "INSERT INTO orders VALUES (10, 1, 100), (11, 1, 50), (12, 2, 200)",
    );
}

#[test]
fn cross_using_and_natural_joins_run() {
    // CROSS (Cartesian), USING (cols), and NATURAL (common columns) joins run end-to-end.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t1 (k INT NOT NULL, a INT NOT NULL)");
    run(&engine, "CREATE TABLE t2 (k INT NOT NULL, b INT NOT NULL)");
    run(&engine, "INSERT INTO t1 VALUES (1, 10), (2, 20)");
    run(&engine, "INSERT INTO t2 VALUES (2, 200), (3, 300)");

    // CROSS JOIN = Cartesian product: 2 × 2 = 4 rows.
    assert_eq!(
        rows(run(&engine, "SELECT a FROM t1 CROSS JOIN t2")).len(),
        4
    );

    // USING (k): join where t1.k = t2.k → only k = 2.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a, b FROM t1 JOIN t2 USING (k) ORDER BY a"
        )),
        vec![vec![Value::Int(20), Value::Int(200)]]
    );

    // NATURAL JOIN: the only common column is k → same result as USING (k).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT a, b FROM t1 NATURAL JOIN t2 ORDER BY a"
        )),
        vec![vec![Value::Int(20), Value::Int(200)]]
    );
}

#[test]
fn using_natural_join_coalesce_probe() {
    // The QA-flagged shapes: a USING/NATURAL join must produce ONE merged
    // join column, so `SELECT *` has the coalesced column count, a bare `k` is unambiguous, and a
    // LEFT JOIN USING NULL-pads the non-key columns while keeping the shared key from the left.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE aa (k INT NOT NULL, x TEXT)");
    run(&engine, "CREATE TABLE bb (k INT NOT NULL, y TEXT)");
    run(&engine, "INSERT INTO aa VALUES (1, 'a1'), (2, 'a2')");
    run(&engine, "INSERT INTO bb VALUES (1, 'b1')");

    // SELECT * → one k, then aa's non-key, then bb's non-key = 3 columns (not 4).
    let star = rows(run(&engine, "SELECT * FROM aa JOIN bb USING (k)"));
    assert_eq!(
        star,
        vec![vec![
            Value::Int(1),
            Value::Text("a1".to_owned()),
            Value::Text("b1".to_owned()),
        ]]
    );

    // A bare `k` is unambiguous (the coalesced column), not an "ambiguous column" error.
    assert_eq!(
        rows(run(&engine, "SELECT k FROM aa JOIN bb USING (k)")),
        vec![vec![Value::Int(1)]]
    );

    // LEFT JOIN USING: every left row survives; the unmatched row's bb column is NULL, and the
    // shared key comes from the (preserved) left side.
    let mut left = rows(run(
        &engine,
        "SELECT k, x, y FROM aa LEFT JOIN bb USING (k) ORDER BY k",
    ));
    left.sort_by_key(|r| format!("{r:?}"));
    assert_eq!(
        left,
        vec![
            vec![
                Value::Int(1),
                Value::Text("a1".to_owned()),
                Value::Text("b1".to_owned()),
            ],
            vec![Value::Int(2), Value::Text("a2".to_owned()), Value::Null],
        ]
    );

    // NATURAL JOIN coalesces the same way.
    assert_eq!(
        rows(run(&engine, "SELECT * FROM aa NATURAL JOIN bb")),
        vec![vec![
            Value::Int(1),
            Value::Text("a1".to_owned()),
            Value::Text("b1".to_owned()),
        ]]
    );
}

#[test]
fn inner_join_matches_rows() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);

    // carol (no orders) is excluded by the inner join.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT users.name, orders.total FROM users \
             JOIN orders ON users.id = orders.user_id ORDER BY orders.total",
        )),
        vec![
            vec![Value::Text("alice".to_owned()), Value::Int(50)],
            vec![Value::Text("alice".to_owned()), Value::Int(100)],
            vec![Value::Text("bob".to_owned()), Value::Int(200)],
        ],
    );
}

#[test]
fn left_join_keeps_unmatched_left_rows() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);

    // carol survives with a NULL-padded right side.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT users.name, orders.total FROM users \
             LEFT JOIN orders ON users.id = orders.user_id ORDER BY users.id, orders.total",
        )),
        vec![
            vec![Value::Text("alice".to_owned()), Value::Int(50)],
            vec![Value::Text("alice".to_owned()), Value::Int(100)],
            vec![Value::Text("bob".to_owned()), Value::Int(200)],
            vec![Value::Text("carol".to_owned()), Value::Null],
        ],
    );
}

#[test]
fn join_with_table_aliases_and_where() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);

    assert_eq!(
        rows(run(
            &engine,
            "SELECT u.name, o.total FROM users u \
             JOIN orders o ON u.id = o.user_id WHERE o.total >= 100 ORDER BY o.total",
        )),
        vec![
            vec![Value::Text("alice".to_owned()), Value::Int(100)],
            vec![Value::Text("bob".to_owned()), Value::Int(200)],
        ],
    );
}

#[test]
fn ambiguous_unqualified_column_in_join_is_rejected() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);
    // Both tables have `id`; the bare reference is ambiguous.
    let stmt =
        parse("SELECT id FROM users JOIN orders ON users.id = orders.user_id").expect("parse");
    assert!(analyze(stmt, &EngineCatalog(&engine)).is_err());
}

#[test]
fn join_then_group_by_aggregates_per_group() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);

    // Inner join then group by user: alice has 2 orders (150), bob has 1 (200).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT u.name, COUNT(*), SUM(o.total) FROM users u \
             JOIN orders o ON u.id = o.user_id GROUP BY u.name ORDER BY u.name",
        )),
        vec![
            vec![
                Value::Text("alice".to_owned()),
                Value::Int(2),
                Value::Int(150)
            ],
            vec![
                Value::Text("bob".to_owned()),
                Value::Int(1),
                Value::Int(200)
            ],
        ],
    );
}

#[test]
fn explain_shows_hash_join_for_equi_join_nested_loop_otherwise() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);
    let plan_text = |sql: &str| -> String {
        rows(run(&engine, sql))
            .into_iter()
            .map(|row| match row.into_iter().next() {
                Some(Value::Text(s)) => s,
                other => panic!("expected plan text, got {other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    // An equi-join lowers to a hash join.
    let equi =
        plan_text("EXPLAIN SELECT u.name FROM users u LEFT JOIN orders o ON u.id = o.user_id");
    assert!(equi.contains("HashJoin (Left"), "missing HashJoin: {equi}");
    // A non-equi join still uses a nested-loop join.
    let non_equi =
        plan_text("EXPLAIN SELECT u.name FROM users u LEFT JOIN orders o ON u.id > o.user_id");
    assert!(
        non_equi.contains("NestedLoopJoin (Left)"),
        "missing NestedLoopJoin: {non_equi}",
    );
}

#[test]
fn right_join_keeps_unmatched_right_rows() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);
    // An order whose user_id has no matching user.
    run(&engine, "INSERT INTO orders VALUES (99, 77, 999)");

    // RIGHT JOIN keeps every order; the orphan (99) gets a NULL user.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT orders.id, users.name FROM users \
             RIGHT JOIN orders ON users.id = orders.user_id ORDER BY orders.id",
        )),
        vec![
            vec![Value::Int(10), Value::Text("alice".to_owned())],
            vec![Value::Int(11), Value::Text("alice".to_owned())],
            vec![Value::Int(12), Value::Text("bob".to_owned())],
            vec![Value::Int(99), Value::Null],
        ],
    );
}

#[test]
fn full_join_keeps_unmatched_rows_from_both_sides() {
    let engine = BtreeEngine::new();
    seed_join_tables(&engine);
    // carol has no order; order 99 has no user — one orphan on each side.
    run(&engine, "INSERT INTO orders VALUES (99, 77, 999)");

    assert_eq!(
        rows(run(
            &engine,
            "SELECT users.name, orders.id FROM users \
             FULL JOIN orders ON users.id = orders.user_id ORDER BY users.id, orders.id",
        )),
        vec![
            vec![Value::Text("alice".to_owned()), Value::Int(10)],
            vec![Value::Text("alice".to_owned()), Value::Int(11)],
            vec![Value::Text("bob".to_owned()), Value::Int(12)],
            vec![Value::Text("carol".to_owned()), Value::Null], // unmatched left
            vec![Value::Null, Value::Int(99)],                  // unmatched right
        ],
    );
}

#[test]
fn vacuum_reports_reclaimed_count_from_the_engine() {
    // Drives `VACUUM` through the full SQL pipeline against the real
    // `BtreeEngine` (which overrides `StorageEngine::vacuum`) and asserts the
    // reclamation count surfaces in `ExecutionResult::Vacuumed`. The mock engine
    // used in the unit tests of `nusadb-sql` always returns 0, so this is the only
    // place the wiring through `engine.vacuum()` can be observed end-to-end.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    // UPDATE supersedes the original version; once the auto-committed transaction
    // is closed and no reader holds a snapshot over the old version, vacuum can
    // reclaim it.
    run(&engine, "UPDATE t SET v = 20 WHERE id = 1");

    assert!(matches!(
        run(&engine, "VACUUM"),
        ExecutionResult::Vacuumed(n) if n >= 1
    ));
    // Idempotent: a second VACUUM has nothing left to reclaim.
    assert!(matches!(
        run(&engine, "VACUUM"),
        ExecutionResult::Vacuumed(0)
    ));
}

#[test]
fn group_by_rollup_produces_subtotals_and_grand_total() {
    // ROLLUP(region, product) aggregates at three levels: per (region, product),
    // per region (product rolled up to NULL), and the grand total (both NULL)
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE sales (region TEXT NOT NULL, product TEXT NOT NULL, amt INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO sales VALUES ('east', 'a', 10), ('east', 'b', 20), ('west', 'a', 30)",
    );

    let out = rows(run(
        &engine,
        "SELECT region, product, SUM(amt) FROM sales GROUP BY ROLLUP (region, product)",
    ));
    // 3 detail rows + 2 region subtotals + 1 grand total.
    assert_eq!(out.len(), 6);
    let t = |s: &str| Value::Text(s.to_owned());
    // Detail level.
    assert!(out.contains(&vec![t("east"), t("a"), Value::Int(10)]));
    assert!(out.contains(&vec![t("east"), t("b"), Value::Int(20)]));
    assert!(out.contains(&vec![t("west"), t("a"), Value::Int(30)]));
    // Region subtotals — `product` rolled up to NULL.
    assert!(out.contains(&vec![t("east"), Value::Null, Value::Int(30)]));
    assert!(out.contains(&vec![t("west"), Value::Null, Value::Int(30)]));
    // Grand total — both keys NULL.
    assert!(out.contains(&vec![Value::Null, Value::Null, Value::Int(60)]));
}

#[test]
fn window_functions_rank_and_aggregate_over_partitions() {
    // Window functions annotate each row without collapsing it.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE emp (dept TEXT NOT NULL, name TEXT NOT NULL, sal INT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO emp VALUES ('a', 'x', 10), ('a', 'y', 20), ('a', 'z', 20), ('b', 'w', 30)",
    );
    let t = |s: &str| Value::Text(s.to_owned());

    // ROW_NUMBER: sequential within each partition, ties broken by input order (stable).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY sal) AS rn FROM emp",
        )),
        vec![
            vec![t("x"), Value::Int(1)],
            vec![t("y"), Value::Int(2)],
            vec![t("z"), Value::Int(3)],
            vec![t("w"), Value::Int(1)],
        ]
    );

    // RANK: ties share a rank (the 20-20 pair both rank 2).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name, RANK() OVER (PARTITION BY dept ORDER BY sal) AS rnk FROM emp",
        )),
        vec![
            vec![t("x"), Value::Int(1)],
            vec![t("y"), Value::Int(2)],
            vec![t("z"), Value::Int(2)],
            vec![t("w"), Value::Int(1)],
        ]
    );

    // Aggregate over a partition (no ORDER BY): the partition total, repeated on every row.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name, SUM(sal) OVER (PARTITION BY dept) AS total FROM emp",
        )),
        vec![
            vec![t("x"), Value::Int(50)],
            vec![t("y"), Value::Int(50)],
            vec![t("z"), Value::Int(50)],
            vec![t("w"), Value::Int(30)],
        ]
    );

    // Running aggregate (ORDER BY): the default RANGE frame includes the whole peer group, so the
    // two sal=20 rows share the running total 10+20+20=50.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name, SUM(sal) OVER (ORDER BY sal) AS running FROM emp",
        )),
        vec![
            vec![t("x"), Value::Int(10)],
            vec![t("y"), Value::Int(50)],
            vec![t("z"), Value::Int(50)],
            vec![t("w"), Value::Int(80)],
        ]
    );
}

#[test]
fn grouping_sets_with_only_empty_sets_yields_one_row_per_set() {
    // Regression (the design #7): `GROUPING SETS ((), ())` has an empty key union but must still
    // produce one grand-total row per listed set, not collapse to a single scalar aggregate.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (v INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2), (3)");

    let out = rows(run(
        &engine,
        "SELECT COUNT(*) FROM t GROUP BY GROUPING SETS ((), ())",
    ));
    assert_eq!(out, vec![vec![Value::Int(3)], vec![Value::Int(3)]]);
}

#[test]
fn window_navigation_and_distribution_functions() {
    // LAG/LEAD/FIRST_VALUE/NTILE/CUME_DIST/PERCENT_RANK over a single ordered partition.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE nums (v INT NOT NULL)");
    run(
        &engine,
        "INSERT INTO nums VALUES (10), (20), (30), (40), (50)",
    );

    let out = rows(run(
        &engine,
        "SELECT v, \
           LAG(v) OVER (ORDER BY v) AS lg, \
           LEAD(v) OVER (ORDER BY v) AS ld, \
           FIRST_VALUE(v) OVER (ORDER BY v) AS fst, \
           NTILE(2) OVER (ORDER BY v) AS bucket, \
           CUME_DIST() OVER (ORDER BY v) AS cd, \
           PERCENT_RANK() OVER (ORDER BY v) AS pr \
         FROM nums",
    ));
    let i = Value::Int;
    let f = Value::Float;
    assert_eq!(
        out,
        vec![
            vec![i(10), Value::Null, i(20), i(10), i(1), f(0.2), f(0.0)],
            vec![i(20), i(10), i(30), i(10), i(1), f(0.4), f(0.25)],
            vec![i(30), i(20), i(40), i(10), i(1), f(0.6), f(0.5)],
            vec![i(40), i(30), i(50), i(10), i(2), f(0.8), f(0.75)],
            vec![i(50), i(40), Value::Null, i(10), i(2), f(1.0), f(1.0)],
        ]
    );
}

#[test]
fn lag_lead_default_coerces_to_the_value_type() {
    // A LAG/LEAD default of a different but assignable type (e.g. an INT
    // literal for a NUMERIC or FLOAT value) is coerced to the value's type, as the reference engine does, rather than
    // rejected with a type mismatch.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, amt NUMERIC(10,2) NOT NULL)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10.50), (2, 20.25)");
    // LAG(amt, 1, 0): the first row has no predecessor, so the INT default 0 fills it, coerced to the
    // NUMERIC value type. Like the reference engine, LAG/LEAD's result is an unconstrained NUMERIC, so the default
    // renders with its own scale ("0"), while an actual value keeps its stored scale ("10.50").
    let out = rows(run(
        &engine,
        "SELECT id, LAG(amt, 1, 0) OVER (ORDER BY id)::text FROM t ORDER BY id",
    ));
    assert_eq!(
        out,
        vec![
            vec![Value::Int(1), Value::Text("0".to_owned())],
            vec![Value::Int(2), Value::Text("10.50".to_owned())],
        ]
    );

    // A FLOAT value with an INT default likewise coerces (9 -> 9.0), not rejected.
    run(
        &engine,
        "CREATE TABLE f (id INT NOT NULL, x FLOAT NOT NULL)",
    );
    run(&engine, "INSERT INTO f VALUES (1, 1.5), (2, 2.5)");
    let of = rows(run(
        &engine,
        "SELECT id, LEAD(x, 1, 9) OVER (ORDER BY id) FROM f ORDER BY id",
    ));
    assert_eq!(
        of,
        vec![
            vec![Value::Int(1), Value::Float(2.5)],
            vec![Value::Int(2), Value::Float(9.0)],
        ]
    );

    // An explicit NULL default is typed against the value column, not rejected as an ambiguous NULL:
    // the first row (no predecessor) is NULL, the second is its predecessor's value.
    let n = rows(run(
        &engine,
        "SELECT id, LAG(x, 1, NULL) OVER (ORDER BY id) FROM f ORDER BY id",
    ));
    assert_eq!(
        n,
        vec![
            vec![Value::Int(1), Value::Null],
            vec![Value::Int(2), Value::Float(1.5)],
        ]
    );
}

#[test]
fn cume_dist_is_peer_aware() {
    // Rows with equal ORDER BY keys are peers and share the cumulative distribution.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (v INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (10), (20), (20), (40)");
    let out = rows(run(
        &engine,
        "SELECT v, CUME_DIST() OVER (ORDER BY v) AS cd FROM t",
    ));
    let i = Value::Int;
    let f = Value::Float;
    assert_eq!(
        out,
        vec![
            vec![i(10), f(0.25)],
            vec![i(20), f(0.75)],
            vec![i(20), f(0.75)],
            vec![i(40), f(1.0)],
        ]
    );
}

#[test]
fn non_recursive_cte_as_from_base() {
    // WITH (common table expression) referenced as the FROM base.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE users (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO users VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    );

    // Basic CTE + an outer WHERE over the CTE's columns.
    assert_eq!(
        rows(run(
            &engine,
            "WITH big AS (SELECT id, name FROM users WHERE id >= 2) \
             SELECT name FROM big WHERE id = 3",
        )),
        vec![vec![Value::Text("c".to_owned())]]
    );

    // Explicit CTE column names rename the projected columns.
    assert_eq!(
        rows(run(
            &engine,
            "WITH r (x, y) AS (SELECT id, name FROM users) SELECT x FROM r WHERE y = 'b'",
        )),
        vec![vec![Value::Int(2)]]
    );

    // A CTE body may itself aggregate.
    assert_eq!(
        rows(run(
            &engine,
            "WITH cnt AS (SELECT COUNT(*) AS n FROM users) SELECT n FROM cnt",
        )),
        vec![vec![Value::Int(3)]]
    );

    // WITH RECURSIVE now executes to a fixpoint: a terminating counter yields 1..=4.
    assert_eq!(
        rows(run(
            &engine,
            "WITH RECURSIVE nums AS \
                 (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 4) \
                 SELECT n FROM nums ORDER BY n",
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
        ]
    );

    // A non-terminating recursion (no progress toward a base case) is caught by the depth guard
    // rather than looping forever.
    assert!(run_try(
        &engine,
        "WITH RECURSIVE r AS (SELECT id FROM users UNION ALL SELECT id FROM r) SELECT id FROM r",
    )
    .is_err());
}

#[test]
fn uncorrelated_subqueries_execute() {
    // Scalar, EXISTS / NOT EXISTS, and IN / NOT IN subqueries — uncorrelated,
    // pre-resolved once per query against the live engine snapshot.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, grp INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 10), (2, 10), (3, 20), (4, 20)",
    );
    run(&engine, "CREATE TABLE keep (g INT NOT NULL)");
    run(&engine, "INSERT INTO keep VALUES (10)");
    let i = Value::Int;

    // Scalar subquery in the SELECT list: every row gets the same aggregate value.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id, (SELECT COUNT(*) FROM t) FROM t WHERE id = 1",
        )),
        vec![vec![i(1), i(4)]]
    );

    // Scalar subquery in a WHERE predicate.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE id = (SELECT MAX(id) FROM t)",
        )),
        vec![vec![i(4)]]
    );

    // A scalar subquery with no rows yields NULL (so the `=` is NULL → row filtered out).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE id = (SELECT id FROM t WHERE id > 100)",
        )),
        Vec::<Vec<Value>>::new()
    );

    // IN (subquery): keep rows whose grp is present in `keep`.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE grp IN (SELECT g FROM keep) ORDER BY id",
        )),
        vec![vec![i(1)], vec![i(2)]]
    );

    // NOT IN (subquery): the complement.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE grp NOT IN (SELECT g FROM keep) ORDER BY id",
        )),
        vec![vec![i(3)], vec![i(4)]]
    );

    // EXISTS is true (the subquery has rows) → all rows pass; NOT EXISTS → none.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM keep) ORDER BY id",
        )),
        vec![vec![i(1)], vec![i(2)], vec![i(3)], vec![i(4)]]
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM keep)",
        )),
        Vec::<Vec<Value>>::new()
    );

    // A scalar subquery returning more than one row is a run-time error.
    assert!(run_try(&engine, "SELECT (SELECT id FROM t) FROM t").is_err());
}

#[test]
fn correlated_subqueries_execute() {
    // Correlated subqueries: the subquery body references the outer row and is re-run per
    // outer row against the live engine.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE dept (id INT NOT NULL, name TEXT)");
    run(
        &engine,
        "INSERT INTO dept VALUES (1, 'eng'), (2, 'sales'), (3, 'empty')",
    );
    run(
        &engine,
        "CREATE TABLE emp (id INT NOT NULL, dept INT, salary INT)",
    );
    run(
        &engine,
        "INSERT INTO emp VALUES (1, 1, 100), (2, 1, 200), (3, 2, 50)",
    );
    let i = Value::Int;
    let t = |s: &str| Value::Text(s.to_owned());

    // Correlated EXISTS: departments that have at least one employee.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name FROM dept d \
             WHERE EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.id) ORDER BY name",
        )),
        vec![vec![t("eng")], vec![t("sales")]]
    );

    // Correlated NOT EXISTS: departments with no employees.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name FROM dept d WHERE NOT EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.id)",
        )),
        vec![vec![t("empty")]]
    );

    // Correlated scalar subquery in WHERE: employees paid above their department's average.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM emp e \
             WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept) ORDER BY id",
        )),
        vec![vec![i(2)]]
    );

    // Correlated scalar subquery in the SELECT list: per-department employee count.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT name, (SELECT COUNT(*) FROM emp e WHERE e.dept = d.id) \
             FROM dept d ORDER BY name",
        )),
        vec![
            vec![t("empty"), i(0)],
            vec![t("eng"), i(2)],
            vec![t("sales"), i(1)],
        ]
    );
}

#[test]
fn window_rows_frame_moving_aggregate() {
    // Explicit ROWS window frames: moving / running aggregates + frame-aware FIRST_VALUE.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE nums (v INT NOT NULL)");
    run(
        &engine,
        "INSERT INTO nums VALUES (10), (20), (30), (40), (50)",
    );
    let i = Value::Int;

    // Moving sum over [1 PRECEDING, CURRENT ROW].
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v, SUM(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS ms FROM nums",
        )),
        vec![
            vec![i(10), i(10)],
            vec![i(20), i(30)],
            vec![i(30), i(50)],
            vec![i(40), i(70)],
            vec![i(50), i(90)],
        ]
    );

    // Running total over [UNBOUNDED PRECEDING, CURRENT ROW] = prefix sum.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v, SUM(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rt FROM nums",
        )),
        vec![
            vec![i(10), i(10)],
            vec![i(20), i(30)],
            vec![i(30), i(60)],
            vec![i(40), i(100)],
            vec![i(50), i(150)],
        ]
    );

    // FIRST_VALUE reads the frame's first row.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v, FIRST_VALUE(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS fv FROM nums",
        )),
        vec![
            vec![i(10), i(10)],
            vec![i(20), i(10)],
            vec![i(30), i(20)],
            vec![i(40), i(30)],
            vec![i(50), i(40)],
        ]
    );
}

#[test]
fn window_range_frame_is_peer_aware() {
    // RANGE frames are peer-based: CURRENT ROW spans the whole peer group, so rows with
    // equal ORDER BY keys share the same frame (and value).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (v INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (10), (20), (20), (40)");
    let i = Value::Int;
    // Reverse running sum over [CURRENT ROW, UNBOUNDED FOLLOWING].
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v, SUM(v) OVER (ORDER BY v RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS s FROM t",
        )),
        vec![
            vec![i(10), i(90)],
            vec![i(20), i(80)],
            vec![i(20), i(80)],
            vec![i(40), i(40)],
        ]
    );
}

#[test]
fn primary_key_uniqueness_is_enforced_on_insert() {
    // Column-level PRIMARY KEY is registered at CREATE TABLE and enforced on INSERT:
    // a duplicate key — against an existing row or another row in the same batch — is rejected.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, name TEXT)");
    run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");

    // Duplicate of an already-committed key is rejected.
    assert!(run_try(&engine, "INSERT INTO t VALUES (1, 'c')").is_err());

    // A duplicate *within* one INSERT batch is rejected (validated before any write).
    assert!(run_try(&engine, "INSERT INTO t VALUES (3, 'c'), (3, 'd')").is_err());

    // A distinct key still inserts; the rejected rows above left no partial state.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );
    run(&engine, "INSERT INTO t VALUES (3, 'c')");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ]
    );
}

#[test]
fn primary_key_uniqueness_is_enforced_on_update() {
    // An UPDATE that would make a PRIMARY KEY collide with another row is rejected,
    // validated over the whole post-update table before any write; a free target still updates.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, name TEXT)");
    run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')");

    // Moving id=1 onto the existing id=2 collides → rejected, no partial write.
    assert!(run_try(&engine, "UPDATE t SET id = 2 WHERE id = 1").is_err());
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ]
    );

    // Moving id=1 to a free key (10) succeeds.
    run(&engine, "UPDATE t SET id = 10 WHERE id = 1");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(10)]
        ]
    );

    // Updating a non-key column does not trip the constraint.
    run(&engine, "UPDATE t SET name = 'z' WHERE id = 2");
    assert_eq!(
        rows(run(&engine, "SELECT name FROM t WHERE id = 2")),
        vec![vec![Value::Text("z".to_owned())]]
    );
}

#[test]
fn unique_constraint_is_enforced() {
    // Column-level and table-level UNIQUE are registered at CREATE TABLE and enforced:
    // duplicates rejected on INSERT/UPDATE; multiple NULLs allowed (NULL is distinct in UNIQUE).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE u (id INT PRIMARY KEY, email TEXT UNIQUE, tag TEXT)",
    );
    run(
        &engine,
        "INSERT INTO u VALUES (1, 'a@x', 'p'), (2, 'b@x', 'q')",
    );

    // Duplicate email (column-level UNIQUE) rejected.
    assert!(run_try(&engine, "INSERT INTO u VALUES (3, 'a@x', 'r')").is_err());
    // Duplicate PRIMARY KEY still rejected.
    assert!(run_try(&engine, "INSERT INTO u VALUES (1, 'c@x', 's')").is_err());
    // UPDATE onto an existing email rejected.
    assert!(run_try(&engine, "UPDATE u SET email = 'b@x' WHERE id = 1").is_err());

    // Multiple NULL emails are allowed (UNIQUE treats NULL as distinct).
    run(
        &engine,
        "INSERT INTO u VALUES (10, NULL, 't'), (11, NULL, 'v')",
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM u WHERE email IS NULL ORDER BY id"
        )),
        vec![vec![Value::Int(10)], vec![Value::Int(11)]]
    );

    // A composite table-level UNIQUE (a, b): the pair must be unique, components may repeat.
    run(&engine, "CREATE TABLE c (a INT, b INT, UNIQUE (a, b))");
    run(&engine, "INSERT INTO c VALUES (1, 1), (1, 2), (2, 1)");
    assert!(run_try(&engine, "INSERT INTO c VALUES (1, 1)").is_err());
    run(&engine, "INSERT INTO c VALUES (2, 2)");
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM c")),
        vec![vec![Value::Int(4)]]
    );
}

#[test]
fn unique_index_probe_uses_latest_committed_visibility_under_repeatable_read() {
    // The O(log n) index-probe uniqueness check must use LATEST-COMMITTED visibility, not the txn's
    // frozen snapshot: under REPEATABLE READ a key another transaction commits AFTER T began must
    // still be seen, or T would commit a duplicate. (The per-key lock only serializes concurrent
    // writers; a writer that already committed and released its lock is caught by visibility alone.)
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY)");
    run(&engine, "INSERT INTO t VALUES (1)");

    let mut t = Session::new(&engine);
    t.execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    // Take the snapshot by reading (it now cannot see any later commit).
    t.execute(build_plan(&engine, "SELECT id FROM t")).unwrap();

    // Another transaction commits a NEW key after T's snapshot.
    run(&engine, "INSERT INTO t VALUES (5)");

    // T inserts the same key: its snapshot is blind to `5`, but the probe's latest-committed view
    // sees the committed row, so the duplicate is still rejected.
    assert!(
        t.execute(build_plan(&engine, "INSERT INTO t VALUES (5)"))
            .is_err(),
        "the uniqueness probe must see a key committed after the REPEATABLE READ snapshot began",
    );
    let _ = t.execute(build_plan(&engine, "ROLLBACK"));
    // The committed table holds exactly {1, 5}; the rejected insert left no partial state.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(5)]]
    );
}

#[test]
fn update_unique_probe_uses_latest_committed_visibility_under_repeatable_read() {
    // The UPDATE uniqueness re-check probe must also use latest-committed visibility: a key another
    // transaction commits after an RR snapshot began must still block an UPDATE that moves a row to it.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code INT UNIQUE)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");

    let mut t = Session::new(&engine);
    t.execute(build_plan(&engine, "BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    t.execute(build_plan(&engine, "SELECT code FROM t"))
        .unwrap(); // take the snapshot

    // Another transaction commits a new unique value after T's snapshot.
    run(&engine, "INSERT INTO t VALUES (3, 30)");

    // T moves row 1's code to 30: its snapshot cannot see (3,30), but the probe's latest-committed
    // view does, so the duplicate is rejected.
    assert!(
        t.execute(build_plan(&engine, "UPDATE t SET code = 30 WHERE id = 1"))
            .is_err(),
        "the UPDATE uniqueness probe must see a key committed after the RR snapshot began",
    );
    let _ = t.execute(build_plan(&engine, "ROLLBACK"));
    assert_eq!(
        rows(run(&engine, "SELECT code FROM t ORDER BY code")),
        vec![
            vec![Value::Int(10)],
            vec![Value::Int(20)],
            vec![Value::Int(30)]
        ]
    );
}

#[test]
fn update_keeping_its_own_key_is_not_a_false_self_conflict() {
    // The UPDATE probe must exclude the row being rewritten by its tid: updating a non-key column,
    // or re-setting a key to its own value, must not read the row's own committed index entry as a
    // duplicate of itself.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code INT UNIQUE, v INT)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10, 100), (2, 20, 200)");
    // A non-key update keeps both unique keys — must not self-conflict.
    run(&engine, "UPDATE t SET v = 999 WHERE id = 1");
    // Re-setting the keys to their own values likewise must not self-conflict.
    run(&engine, "UPDATE t SET id = 1, code = 10 WHERE id = 1");
    assert_eq!(
        rows(run(&engine, "SELECT id, code, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10), Value::Int(999)],
            vec![Value::Int(2), Value::Int(20), Value::Int(200)]
        ]
    );
}

#[test]
fn delete_by_primary_key_point_get_deletes_only_that_row() {
    // `DELETE ... WHERE pk = const` finds its row through the backing index (O(log n)) rather than a
    // full scan. The result must be identical to the scan path: exactly the matching row is removed,
    // every other row survives untouched.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    run(&engine, "DELETE FROM t WHERE id = 2");
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(3), Value::Int(30)],
        ]
    );
    // A point-get on an absent key deletes nothing.
    run(&engine, "DELETE FROM t WHERE id = 99");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]]
    );
}

#[test]
fn delete_index_point_get_still_applies_the_residual_predicate() {
    // The index point-get only *narrows* the candidate set to a superset; the full `WHERE` is still
    // re-applied per row. `WHERE id = 1 AND v = 999` must delete nothing when row 1's `v` is not 999,
    // proving the residual predicate runs on top of the index lookup (never a bare index-key match).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    run(&engine, "DELETE FROM t WHERE id = 1 AND v = 999");
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );
    // The matching compound predicate does delete the row.
    run(&engine, "DELETE FROM t WHERE id = 1 AND v = 10");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn delete_by_unique_non_primary_key_column_point_get() {
    // A UNIQUE (non-PK) column is also a point-get path; a range predicate on it falls back to the
    // scan path — both must return exactly the qualifying rows.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code INT UNIQUE)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    run(&engine, "DELETE FROM t WHERE code = 20");
    assert_eq!(
        rows(run(&engine, "SELECT id, code FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(3), Value::Int(30)],
        ]
    );
    // A range predicate (not a point-get) uses the scan fallback: rows with code > 10 go.
    run(&engine, "DELETE FROM t WHERE code > 10");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t ORDER BY id")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn update_by_primary_key_point_get_updates_only_that_row() {
    // `UPDATE ... WHERE pk = const` finds its row through the backing index (O(log n)); the result
    // must match the scan path exactly — only the matching row changes, others are untouched, and a
    // PK/UNIQUE re-check (the point-get row has a UNIQUE column) still holds.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, code INT UNIQUE, v INT)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 10, 100), (2, 20, 200), (3, 30, 300)",
    );
    run(&engine, "UPDATE t SET v = 999 WHERE id = 2");
    assert_eq!(
        rows(run(&engine, "SELECT id, code, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10), Value::Int(100)],
            vec![Value::Int(2), Value::Int(20), Value::Int(999)],
            vec![Value::Int(3), Value::Int(30), Value::Int(300)],
        ]
    );
    // Moving a row's UNIQUE key to a value an untouched row already holds must still be rejected —
    // the whole-table uniqueness invariant is preserved on the index find-path.
    assert!(
        run_try(&engine, "UPDATE t SET code = 30 WHERE id = 1").is_err(),
        "a point-get UPDATE that collides a UNIQUE key with an untouched row must be rejected",
    );
    // A point-get on an absent key updates nothing.
    run(&engine, "UPDATE t SET v = -1 WHERE id = 99");
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(2), Value::Int(999)],
            vec![Value::Int(3), Value::Int(300)],
        ]
    );
}

#[test]
fn update_index_point_get_still_applies_the_residual_predicate() {
    // The index point-get only narrows candidates; the full `WHERE` is re-applied per row. `WHERE
    // id = 1 AND v = 999` must update nothing when row 1's `v` is not 999.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    run(&engine, "UPDATE t SET v = 555 WHERE id = 1 AND v = 999");
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );
    // The matching compound predicate does update the row.
    run(&engine, "UPDATE t SET v = 555 WHERE id = 1 AND v = 10");
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(555)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );
}

#[test]
fn foreign_key_existence_and_referential_actions() {
    // FOREIGN KEY end-to-end: existence check on child INSERT/UPDATE, NO ACTION/RESTRICT
    // reject on parent DELETE, ON DELETE CASCADE propagation, NULL FK allowed (MATCH SIMPLE).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE parent (id INT PRIMARY KEY, name TEXT)",
    );
    run(&engine, "INSERT INTO parent VALUES (1, 'a'), (2, 'b')");
    run(
        &engine,
        "CREATE TABLE child (id INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES parent (id))",
    );

    // Child referencing an existing parent inserts; a NULL FK is allowed; a dangling ref is rejected.
    run(&engine, "INSERT INTO child VALUES (1, 1), (2, 2)");
    run(&engine, "INSERT INTO child VALUES (3, NULL)");
    assert!(run_try(&engine, "INSERT INTO child VALUES (4, 99)").is_err());
    // UPDATE to a dangling parent is rejected; to a valid parent is allowed.
    assert!(run_try(&engine, "UPDATE child SET pid = 99 WHERE id = 1").is_err());
    run(&engine, "UPDATE child SET pid = 2 WHERE id = 1");

    // Deleting a parent that still has children is rejected (default NO ACTION).
    assert!(run_try(&engine, "DELETE FROM parent WHERE id = 2").is_err());
    // The child set is unchanged after the rejected delete.
    assert_eq!(
        rows(run(&engine, "SELECT COUNT(*) FROM child")),
        vec![vec![Value::Int(3)]]
    );

    // ON DELETE CASCADE: deleting the parent removes the referencing children.
    run(
        &engine,
        "CREATE TABLE cc (id INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES parent (id) ON DELETE CASCADE)",
    );
    run(&engine, "INSERT INTO cc VALUES (1, 1), (2, 1), (3, 2)");
    // id=1 still has children in `child` (pid moved to 2 above; id=2,3 reference parent 2)... so
    // delete parent id=1 which only `cc` rows (pid=1) reference now → cascades those away.
    run(&engine, "DELETE FROM parent WHERE id = 1");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM cc ORDER BY id")),
        vec![vec![Value::Int(3)]]
    );
}

#[test]
fn foreign_key_referencing_a_non_pk_unique_column() {
    // A FOREIGN KEY may reference a non-PK UNIQUE column of the parent; existence and
    // referential actions resolve against that key, not the parent's PRIMARY KEY.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE parent (id INT PRIMARY KEY, code TEXT NOT NULL UNIQUE)",
    );
    run(&engine, "INSERT INTO parent VALUES (1, 'A'), (2, 'B')");
    run(
        &engine,
        "CREATE TABLE child (cid INT PRIMARY KEY, pcode TEXT, \
         FOREIGN KEY (pcode) REFERENCES parent (code) ON DELETE CASCADE ON UPDATE CASCADE)",
    );

    // Existence is checked against parent.code, not parent.id.
    run(&engine, "INSERT INTO child VALUES (10, 'A'), (11, 'B')");
    run(&engine, "INSERT INTO child VALUES (12, NULL)"); // NULL FK references nothing (MATCH SIMPLE).
    assert!(
        run_try(&engine, "INSERT INTO child VALUES (13, 'Z')").is_err(),
        "a code with no matching parent.code must be rejected",
    );
    // A parent.id value ('1') is not a parent.code — referencing the PK by mistake would wrongly accept.
    assert!(
        run_try(&engine, "INSERT INTO child VALUES (14, '1')").is_err(),
        "the FK references parent.code, so a parent.id value must not satisfy it",
    );

    // ON UPDATE CASCADE on the referenced UNIQUE value rewrites the child FK.
    run(&engine, "UPDATE parent SET code = 'A2' WHERE id = 1");
    assert_eq!(
        rows(run(&engine, "SELECT pcode FROM child WHERE cid = 10")),
        vec![vec![Value::Text("A2".to_owned())]],
    );

    // ON DELETE CASCADE: deleting a parent removes children referencing its code (NULL FK kept).
    run(&engine, "DELETE FROM parent WHERE id = 2"); // code 'B'
    assert_eq!(
        rows(run(&engine, "SELECT cid FROM child ORDER BY cid")),
        vec![vec![Value::Int(10)], vec![Value::Int(12)]],
    );

    // Referencing a column that is not a PRIMARY KEY or UNIQUE constraint is rejected at CREATE.
    run(&engine, "CREATE TABLE p3 (id INT PRIMARY KEY, name TEXT)");
    assert!(
        run_try(
            &engine,
            "CREATE TABLE bad (x TEXT, FOREIGN KEY (x) REFERENCES p3 (name))"
        )
        .is_err(),
        "a FK referencing a non-unique parent column must be rejected",
    );
}

#[test]
fn foreign_keys_to_different_parent_keys_resolve_independently() {
    // Two FKs at the same parent table but DIFFERENT keys (PK vs a UNIQUE column) must each resolve
    // against their own referenced key — exercises the per-FK key resolution in the cascade path.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE p (id INT PRIMARY KEY, code TEXT NOT NULL UNIQUE)",
    );
    run(&engine, "INSERT INTO p VALUES (1, 'A'), (2, 'B')");
    run(
        &engine,
        "CREATE TABLE c_by_id (x INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES p (id) ON DELETE CASCADE)",
    );
    run(
        &engine,
        "CREATE TABLE c_by_code (x INT PRIMARY KEY, pc TEXT, \
         FOREIGN KEY (pc) REFERENCES p (code) ON DELETE CASCADE)",
    );
    run(&engine, "INSERT INTO c_by_id VALUES (1, 1), (2, 2)");
    run(&engine, "INSERT INTO c_by_code VALUES (1, 'A'), (2, 'B')");
    // Deleting p id=1 (code 'A') must cascade to c_by_id (pid=1) AND c_by_code (pc='A').
    run(&engine, "DELETE FROM p WHERE id = 1");
    assert_eq!(
        rows(run(&engine, "SELECT x FROM c_by_id ORDER BY x")),
        vec![vec![Value::Int(2)]],
    );
    assert_eq!(
        rows(run(&engine, "SELECT x FROM c_by_code ORDER BY x")),
        vec![vec![Value::Int(2)]],
    );
}

#[test]
fn dropping_a_constraint_referenced_by_a_foreign_key_is_rejected() {
    // RESTRICT: a UNIQUE / PRIMARY KEY a foreign key references cannot be dropped while the FK
    // exists — otherwise the FK's referenced key would silently dangle and enforcement would degrade
    // to a different key. Mirrors the DROP TABLE FK guard.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE parent (id INT PRIMARY KEY, code TEXT NOT NULL)",
    );
    run(
        &engine,
        "ALTER TABLE parent ADD CONSTRAINT uc UNIQUE (code)",
    );
    run(&engine, "INSERT INTO parent VALUES (1, 'A')");
    run(
        &engine,
        "CREATE TABLE child (cid INT PRIMARY KEY, pcode TEXT, \n         FOREIGN KEY (pcode) REFERENCES parent (code))",
    );
    run(&engine, "INSERT INTO child VALUES (10, 'A')");
    // Dropping the UNIQUE constraint the FK references is rejected while the FK exists.
    assert!(
        run_try(&engine, "ALTER TABLE parent DROP CONSTRAINT uc").is_err(),
        "dropping a UNIQUE constraint referenced by a foreign key must be rejected",
    );
    // The FK still enforces (proof the constraint is intact): a dangling child code is rejected.
    assert!(run_try(&engine, "INSERT INTO child VALUES (11, 'Z')").is_err());
    // Once the foreign key is gone (its child table dropped), the UNIQUE can be dropped.
    run(&engine, "DROP TABLE child");
    run(&engine, "ALTER TABLE parent DROP CONSTRAINT uc");
}

#[test]
fn drop_table_with_a_self_referencing_foreign_key_succeeds() {
    // A self-referencing FK (child == parent) must not block its own table's DROP: the table's own
    // FKs are torn down before its PRIMARY KEY, so the DROP CONSTRAINT FK-RESTRICT guard does not
    // wrongly fire during teardown. Regression for the guard added alongside FK -> non-PK UNIQUE.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE node (id INT PRIMARY KEY, parent_id INT, \
         FOREIGN KEY (parent_id) REFERENCES node (id))",
    );
    run(&engine, "INSERT INTO node VALUES (1, NULL)");
    run(&engine, "INSERT INTO node VALUES (2, 1)"); // self-reference to the committed row id=1
    run(&engine, "DROP TABLE node");
    // The name (and its PK index) is free to reuse — a clean teardown.
    run(
        &engine,
        "CREATE TABLE node (id INT PRIMARY KEY, parent_id INT, \
         FOREIGN KEY (parent_id) REFERENCES node (id))",
    );
    run(&engine, "INSERT INTO node VALUES (1, NULL)");
    assert_eq!(
        rows(run(&engine, "SELECT id FROM node")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn foreign_key_set_null_and_on_update_actions() {
    // ON DELETE SET NULL (null the child FK), ON UPDATE CASCADE (rewrite child FK to the
    // new parent key), and ON UPDATE NO ACTION (reject a parent-key change with dependents).
    let engine = BtreeEngine::new();

    // --- ON DELETE SET NULL ---
    run(&engine, "CREATE TABLE p (id INT PRIMARY KEY, name TEXT)");
    run(&engine, "INSERT INTO p VALUES (1, 'a'), (2, 'b')");
    run(
        &engine,
        "CREATE TABLE cn (id INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES p (id) ON DELETE SET NULL)",
    );
    run(&engine, "INSERT INTO cn VALUES (1, 1), (2, 1), (3, 2)");
    run(&engine, "DELETE FROM p WHERE id = 1");
    // Children that referenced p=1 are nulled; the p=2 reference is untouched.
    assert_eq!(
        rows(run(&engine, "SELECT id, pid FROM cn ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Null],
            vec![Value::Int(2), Value::Null],
            vec![Value::Int(3), Value::Int(2)],
        ]
    );

    // --- ON UPDATE CASCADE ---
    run(&engine, "CREATE TABLE q (id INT PRIMARY KEY)");
    run(&engine, "INSERT INTO q VALUES (1), (2)");
    run(
        &engine,
        "CREATE TABLE cc (id INT PRIMARY KEY, qid INT, \
         FOREIGN KEY (qid) REFERENCES q (id) ON UPDATE CASCADE)",
    );
    run(&engine, "INSERT INTO cc VALUES (1, 1), (2, 1), (3, 2)");
    run(&engine, "UPDATE q SET id = 10 WHERE id = 1");
    // The children that referenced q=1 follow to the new key 10.
    assert_eq!(
        rows(run(&engine, "SELECT id, qid FROM cc ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(10)],
            vec![Value::Int(3), Value::Int(2)],
        ]
    );

    // --- ON UPDATE NO ACTION (default): reject a parent-key change with dependents ---
    run(&engine, "CREATE TABLE r (id INT PRIMARY KEY)");
    run(&engine, "INSERT INTO r VALUES (1)");
    run(
        &engine,
        "CREATE TABLE cr (id INT PRIMARY KEY, rid INT, FOREIGN KEY (rid) REFERENCES r (id))",
    );
    run(&engine, "INSERT INTO cr VALUES (1, 1)");
    assert!(run_try(&engine, "UPDATE r SET id = 9 WHERE id = 1").is_err());
    // The rejected update left the parent key unchanged.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM r")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn foreign_key_referencing_a_unique_or_pk_column() {
    // A FK may reference the parent's PRIMARY KEY or any UNIQUE column, but a column that is
    // neither is rejected (not silently redirected to the primary key).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE par (id INT PRIMARY KEY, code INT UNIQUE, note TEXT)",
    );
    // Referencing the PK column explicitly is accepted.
    run(
        &engine,
        "CREATE TABLE ok_pk (id INT PRIMARY KEY, pid INT, \
         FOREIGN KEY (pid) REFERENCES par (id))",
    );
    // Referencing a non-PK UNIQUE column is accepted.
    run(
        &engine,
        "CREATE TABLE ok_uq (id INT PRIMARY KEY, pcode INT, \
         FOREIGN KEY (pcode) REFERENCES par (code))",
    );
    // Referencing a column that is neither PRIMARY KEY nor UNIQUE is rejected.
    assert!(
        run_try(
            &engine,
            "CREATE TABLE bad_child (id INT PRIMARY KEY, pnote TEXT, \
         FOREIGN KEY (pnote) REFERENCES par (note))",
        )
        .is_err()
    );
}

#[test]
fn secondary_index_maintained_on_dml() {
    // A CREATE INDEX secondary index is kept in sync by INSERT/UPDATE/DELETE. We read the
    // index back directly (engine.index_scan) to confirm entries track the live table, and check
    // the base table is uncorrupted. (Query routing through the index is a later stage.)
    use nusadb_core::IsolationLevel;
    use nusadb_core::StorageEngine as _;
    use std::ops::Bound;

    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "CREATE INDEX t_v ON t (v)");
    run(&engine, "INSERT INTO t VALUES (1, 30), (2, 10), (3, 20)");

    let index_count = || {
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        let id = engine.lookup_index("t_v").unwrap().expect("index exists");
        let mut scan = engine
            .index_scan(txn, id, Bound::Unbounded, Bound::Unbounded)
            .unwrap();
        let mut n = 0;
        while scan.try_next().unwrap().is_some() {
            n += 1;
        }
        drop(scan);
        engine.commit(txn).unwrap();
        n
    };

    assert_eq!(
        index_count(),
        3,
        "every inserted row should have an index entry"
    );
    run(&engine, "DELETE FROM t WHERE id = 2");
    assert_eq!(index_count(), 2, "DELETE should remove the index entry");
    run(&engine, "UPDATE t SET v = 99 WHERE id = 1");
    assert_eq!(
        index_count(),
        2,
        "UPDATE should move (not duplicate) the index entry"
    );

    // The base table is uncorrupted by index maintenance.
    assert_eq!(
        rows(run(&engine, "SELECT id, v FROM t ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Int(99)],
            vec![Value::Int(3), Value::Int(20)],
        ]
    );
}

#[test]
fn index_scan_operator_reads_in_key_order_with_backfill() {
    // Stage 3: build an IndexScan plan directly and execute it. Rows inserted BEFORE the
    // index exercise CREATE INDEX backfill; the scan returns rows in ascending key order, and
    // range/equality bounds work via the order-preserving key encoding.
    use nusadb_sql::{PhysicalOperator, PhysicalPlan, execute};
    use std::ops::Bound;

    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
    run(&engine, "INSERT INTO t VALUES (1, 30), (2, 10), (3, 20)");
    // Index created AFTER the inserts → must backfill the existing rows.
    run(&engine, "CREATE INDEX t_v ON t (v)");

    let table = engine.lookup_table("t").unwrap().expect("table exists");
    let scan = |lo: Bound<Vec<Value>>, hi: Bound<Vec<Value>>| -> Vec<Vec<Value>> {
        let plan = PhysicalPlan::Select(
            PhysicalOperator::IndexScan {
                table: table.clone(),
                index: "t_v".to_owned(),
                lo,
                hi,
                unique_point: false,
            },
            None,
        );
        rows(execute(plan, &engine).unwrap())
    };
    let r = |id: i64, v: i64| vec![Value::Int(id), Value::Int(v)];

    // Full scan: all (backfilled) rows in ascending v order: 10,20,30 → ids 2,3,1.
    assert_eq!(
        scan(Bound::Unbounded, Bound::Unbounded),
        vec![r(2, 10), r(3, 20), r(1, 30)]
    );
    // Range v >= 20.
    assert_eq!(
        scan(Bound::Included(vec![Value::Int(20)]), Bound::Unbounded),
        vec![r(3, 20), r(1, 30)]
    );
    // Range v > 20 (exclusive lower).
    assert_eq!(
        scan(Bound::Excluded(vec![Value::Int(20)]), Bound::Unbounded),
        vec![r(1, 30)]
    );
    // Equality v = 10.
    assert_eq!(
        scan(
            Bound::Included(vec![Value::Int(10)]),
            Bound::Included(vec![Value::Int(10)])
        ),
        vec![r(2, 10)]
    );
}

#[test]
fn planner_chooses_index_scan_and_returns_correct_rows() {
    // Stage 4: with an index on the table, a WHERE equality/range on the indexed
    // column makes the planner pick an IndexScan in place of the SeqScan — while still returning
    // exactly the rows a sequential scan + filter would.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE items (id INT NOT NULL, v INT)");
    for i in 1..=5 {
        run(
            &engine,
            &format!("INSERT INTO items VALUES ({i}, {})", i * 10),
        );
    }
    run(&engine, "CREATE INDEX items_id_idx ON items (id)");

    // EXPLAIN proves the planner chose the index scan for the equality predicate.
    let plan: String = rows(run(&engine, "EXPLAIN SELECT v FROM items WHERE id = 3"))
        .into_iter()
        .map(|row| match row.into_iter().next() {
            Some(Value::Text(s)) => s,
            other => panic!("expected plan text, got {other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        plan.contains("IndexScan: items using items_id_idx"),
        "planner did not choose the index scan: {plan}"
    );

    // Equality returns the matching row.
    assert_eq!(
        rows(run(&engine, "SELECT v FROM items WHERE id = 3")),
        vec![vec![Value::Int(30)]]
    );
    // A non-indexed predicate added on top still filters correctly above the index scan.
    assert_eq!(
        rows(run(&engine, "SELECT v FROM items WHERE id = 3 AND v > 100")),
        Vec::<Vec<Value>>::new()
    );
    // Range over the index, ordered for a deterministic assertion.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT v FROM items WHERE id >= 4 ORDER BY id"
        )),
        vec![vec![Value::Int(40)], vec![Value::Int(50)]]
    );
    // No matching row.
    assert_eq!(
        rows(run(&engine, "SELECT v FROM items WHERE id = 99")),
        Vec::<Vec<Value>>::new()
    );
}

#[test]
fn set_op_all_multiset_text_null_and_float_fallback() {
    // INTERSECT ALL / EXCEPT ALL multiset semantics must be identical on the hash fast path
    // (Text + NULL) and the linear fallback (Float). NULL is "not distinct" in set ops.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE a (s TEXT)");
    run(
        &engine,
        "INSERT INTO a VALUES ('x'), ('x'), ('y'), (NULL), (NULL)",
    );
    run(&engine, "CREATE TABLE b (s TEXT)");
    run(&engine, "INSERT INTO b VALUES ('x'), (NULL)");
    let t = |s: &str| vec![Value::Text(s.to_owned())];
    let null = || vec![Value::Null];

    // INTERSECT ALL: min(count) per value, left order → 'x' (min 2,1=1), NULL (min 2,1=1).
    assert_eq!(
        rows(run(
            &engine,
            "SELECT s FROM a INTERSECT ALL SELECT s FROM b"
        )),
        vec![t("x"), null()]
    );
    // EXCEPT ALL: max(0, left-right) per value, left order → 'x' (2-1=1), 'y' (1), NULL (2-1=1).
    assert_eq!(
        rows(run(&engine, "SELECT s FROM a EXCEPT ALL SELECT s FROM b")),
        vec![t("x"), t("y"), null()]
    );

    // Float forces the linear fallback (Float is excluded from the hash fast path); same semantics.
    run(&engine, "CREATE TABLE fa (f FLOAT)");
    run(&engine, "INSERT INTO fa VALUES (1.5), (1.5), (2.5)");
    run(&engine, "CREATE TABLE fb (f FLOAT)");
    run(&engine, "INSERT INTO fb VALUES (1.5)");
    assert_eq!(
        rows(run(
            &engine,
            "SELECT f FROM fa INTERSECT ALL SELECT f FROM fb"
        )),
        vec![vec![Value::Float(1.5)]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT f FROM fa EXCEPT ALL SELECT f FROM fb")),
        vec![vec![Value::Float(1.5)], vec![Value::Float(2.5)]]
    );
}

#[test]
fn scalar_string_functions_end_to_end() {
    // String built-ins run through parse → analyze → plan → execute over real rows.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, '  Hello  '), (2, 'world')",
    );

    // UPPER / LOWER / LENGTH (LENGTH is character-based) + TRIM, ordered by id.
    match run(
        &engine,
        "SELECT UPPER(s), LOWER(s), LENGTH(TRIM(s)) FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Text("  HELLO  ".to_owned()),
                        Value::Text("  hello  ".to_owned()),
                        Value::Int(5),
                    ],
                    vec![
                        Value::Text("WORLD".to_owned()),
                        Value::Text("world".to_owned()),
                        Value::Int(5),
                    ],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // SUBSTRING (1-based), REPLACE, LPAD, POSITION composed in one projection.
    match run(
        &engine,
        "SELECT SUBSTRING(TRIM(s), 1, 3), REPLACE(s, 'o', '0'), LPAD(TRIM(s), 7, '*'), \
         POSITION('l' IN s) FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Text("Hel".to_owned()),
                        Value::Text("  Hell0  ".to_owned()),
                        Value::Text("**Hello".to_owned()),
                        Value::Int(5), // 'l' first appears at char 5 of '  Hello  '
                    ],
                    vec![
                        Value::Text("wor".to_owned()),
                        Value::Text("w0rld".to_owned()),
                        Value::Text("**world".to_owned()),
                        Value::Int(4), // 'l' at char 4 of 'world'
                    ],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // A string function in a WHERE predicate filters correctly.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE LENGTH(TRIM(s)) = 5 AND UPPER(s) LIKE '%WORLD%'"
        )),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn b448_string_functions_end_to_end() {
    // CONCAT/CONCAT_WS (NULL-skipping), LEFT/RIGHT, SPLIT_PART, REVERSE end-to-end.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, a TEXT, b TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'foo', 'bar'), (2, 'hello', NULL)",
    );

    // CONCAT skips NULL; CONCAT_WS joins non-NULL with a separator.
    match run(
        &engine,
        "SELECT CONCAT(a, b), CONCAT_WS('-', a, b), REVERSE(a) FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Text("foobar".to_owned()),
                        Value::Text("foo-bar".to_owned()),
                        Value::Text("oof".to_owned()),
                    ],
                    vec![
                        // row 2: b IS NULL → CONCAT yields just 'hello', CONCAT_WS just 'hello'.
                        Value::Text("hello".to_owned()),
                        Value::Text("hello".to_owned()),
                        Value::Text("olleh".to_owned()),
                    ],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // LEFT / RIGHT / SPLIT_PART over a fixed string.
    match run(
        &engine,
        "SELECT LEFT('abcdef', 3), RIGHT('abcdef', 2), SPLIT_PART('x,y,z', ',', 2) FROM t WHERE id = 1",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![vec![
                    Value::Text("abc".to_owned()),
                    Value::Text("ef".to_owned()),
                    Value::Text("y".to_owned()),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn b449_regex_functions_end_to_end() {
    // REGEXP_REPLACE / REGEXP_MATCH end-to-end.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'foo123bar456'), (2, 'no-digits')",
    );

    // REGEXP_REPLACE: global digit-run masking; REGEXP_MATCH: first digit run (TEXT[] / NULL).
    match run(
        &engine,
        "SELECT REGEXP_REPLACE(s, '[0-9]+', '#', 'g'), REGEXP_MATCH(s, '([0-9]+)') FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Text("foo#bar#".to_owned()),
                        Value::Array(vec![Value::Text("123".to_owned())]),
                    ],
                    vec![
                        // row 2: no digit → REGEXP_REPLACE leaves it; REGEXP_MATCH → NULL.
                        Value::Text("no-digits".to_owned()),
                        Value::Null,
                    ],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // An invalid pattern surfaces as an error (not a panic / wrong result).
    assert!(run_try(&engine, "SELECT REGEXP_REPLACE(s, '(', 'x') FROM t").is_err());
}

#[test]
fn b451_date_time_functions_end_to_end() {
    // EXTRACT / DATE_TRUNC / AGE over a real TIMESTAMP column.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, ts TIMESTAMP)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, TIMESTAMP '2024-06-15 13:45:30')",
    );

    match run(
        &engine,
        "SELECT EXTRACT(YEAR FROM ts), EXTRACT(MONTH FROM ts), DATE_TRUNC('month', ts), \
         AGE(TIMESTAMP '2024-03-01 00:00:00', TIMESTAMP '2024-01-15 00:00:00') FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            let month_start = nusadb_sql::temporal::parse_timestamp("2024-06-01 00:00:00").unwrap();
            assert_eq!(
                rows,
                vec![vec![
                    Value::Float(2024.0),
                    Value::Float(6.0),
                    Value::Timestamp(month_start),
                    // 2024-01-15 + 1 month + 15 days = 2024-03-01 (Feb 2024 = 29 days).
                    Value::Interval(nusadb_sql::interval::Interval {
                        months: 1,
                        days: 15,
                        micros: 0,
                    }),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // EXTRACT of a field that does not apply to the value's type is rejected, not silently wrong.
    assert!(run_try(&engine, "SELECT EXTRACT(YEAR FROM CURRENT_TIME) FROM t").is_err());
}

#[test]
fn b452_to_char_to_date_to_timestamp_end_to_end() {
    // TO_CHAR formats; TO_DATE / TO_TIMESTAMP parse, round-tripping a timestamp.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, ts TIMESTAMP)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, TIMESTAMP '2024-06-15 13:45:30')",
    );

    match run(
        &engine,
        "SELECT TO_CHAR(ts, 'YYYY-MM-DD HH24:MI:SS'), TO_CHAR(ts, 'DD Mon YYYY'), \
         TO_DATE('2024-06-15', 'YYYY-MM-DD'), \
         TO_TIMESTAMP('2024-06-15 13:45:30', 'YYYY-MM-DD HH24:MI:SS') FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            let date = nusadb_sql::temporal::parse_date("2024-06-15").unwrap();
            let ts = nusadb_sql::temporal::parse_timestamp("2024-06-15 13:45:30").unwrap();
            assert_eq!(
                rows,
                vec![vec![
                    Value::Text("2024-06-15 13:45:30".to_owned()),
                    Value::Text("15 Jun 2024".to_owned()),
                    Value::Date(date),
                    Value::Timestamp(ts),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // Weekday and day-of-year codes render through the full SQL path (2024-06-15 is a Saturday,
    // day-of-year 167) rather than echoing the template literally.
    match run(
        &engine,
        "SELECT TO_CHAR(ts, 'Day'), TO_CHAR(ts, 'Dy'), TO_CHAR(ts, 'D'), \
         TO_CHAR(ts, 'ID'), TO_CHAR(ts, 'DDD') FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![vec![
                    Value::Text("Saturday ".to_owned()),
                    Value::Text("Sat".to_owned()),
                    Value::Text("7".to_owned()),
                    Value::Text("6".to_owned()),
                    Value::Text("167".to_owned()),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // FM (fill mode) suppresses padding/leading zeros for the next field, and IDDD (ISO day of year)
    // renders through the full SQL path — neither echoes the template literally.
    match run(
        &engine,
        "SELECT TO_CHAR(ts, 'FMDay'), TO_CHAR(ts, 'FMMonth DD, YYYY'), \
         TO_CHAR(ts, 'FMHH12:MI'), TO_CHAR(ts, 'IDDD') FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![vec![
                    Value::Text("Saturday".to_owned()),
                    Value::Text("June 15, 2024".to_owned()),
                    Value::Text("1:45".to_owned()),
                    Value::Text("167".to_owned()),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // Input that does not match the format is an error, not a wrong/NULL value.
    assert!(run_try(&engine, "SELECT TO_DATE('garbage', 'YYYY-MM-DD') FROM t").is_err());
}

#[test]
fn b450_clock_functions_end_to_end() {
    // Niladic clock functions are statement-stable: one instant for the whole statement.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2)");

    match run(
        &engine,
        "SELECT NOW() = CURRENT_TIMESTAMP, NOW(), CURRENT_DATE, CURRENT_TIME FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 2);
            // NOW() and CURRENT_TIMESTAMP are the same instant → equality holds for every row.
            assert_eq!(rows[0][0], Value::Bool(true));
            assert_eq!(rows[1][0], Value::Bool(true));
            // The instant is identical across rows (does not advance row-to-row).
            assert_eq!(rows[0][1], rows[1][1]);

            let Value::TimestampTz(micros) = rows[0][1] else {
                panic!("NOW() should be TIMESTAMPTZ, got {:?}", rows[0][1]);
            };
            assert!(micros > 1_577_836_800_000_000, "now should be past 2020");
            // CURRENT_DATE / CURRENT_TIME are the floor-div / mod-euclid split of that same instant.
            let micros_per_day = 86_400_000_000_i64;
            assert_eq!(
                rows[0][2],
                Value::Date(i32::try_from(micros.div_euclid(micros_per_day)).unwrap())
            );
            assert_eq!(rows[0][3], Value::Time(micros.rem_euclid(micros_per_day)));
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn p12_row_level_security_enable_disable_end_to_end() {
    // ENABLE ROW LEVEL SECURITY makes a table default-deny for a non-superuser: reads with no
    // applicable policy return zero rows (filtered, never errored, never silently exposed), while
    // writes and not-yet-supported read shapes (JOINs over the RLS table) are refused. A superuser
    // bypasses RLS; DISABLE restores access. (Policy-granted reads are covered by
    // `p12_create_policy_filters_rows_end_to_end`.)
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO t VALUES (1, 'alice'), (2, 'bob')");
    // A separate RLS-free table used as an UPDATE/DELETE target with `t` (RLS) as the source.
    run(&engine, "CREATE TABLE dst (k INT NOT NULL)");

    let alice_rows = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        rows(session.execute(plan(logical)).expect("execute"))
    };
    let alice_err = |sql: &str| {
        analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
        .unwrap_err()
    };

    // Before RLS, a non-superuser reads normally.
    assert_eq!(alice_rows("SELECT id FROM t ORDER BY id").len(), 2);

    run(&engine, "ALTER TABLE t ENABLE ROW LEVEL SECURITY");

    // A superuser still sees every row (the default `run` catalog is a superuser).
    assert_eq!(rows(run(&engine, "SELECT id FROM t ORDER BY id")).len(), 2);

    // A non-superuser reads default-deny to zero rows with no policy. Laundering through a CTE still
    // yields zero rows — the RLS base table is filtered wherever it is read, including a recursive
    // CTE's anchor — so there is no bypass.
    for sql in [
        "SELECT id FROM t",
        "WITH c AS (SELECT id FROM t) SELECT id FROM c",
        "WITH RECURSIVE c(id) AS (SELECT id FROM t UNION SELECT id FROM c) SELECT id FROM c",
    ] {
        assert_eq!(alice_rows(sql).len(), 0, "default-deny for `{sql}`");
    }

    // Read shapes whose correct per-relation policy placement is not yet implemented — a JOIN over
    // the RLS table, including a recursive CTE's recursive term — are refused at analysis: fail
    // closed rather than risk an incorrect filter. (SELECT/DELETE/UPDATE/INSERT are now enforced by
    // filtering / WITH CHECK — see the p12_*_policy / p12_*_with_check tests.)
    for sql in [
        "SELECT t.id FROM t JOIN t AS u ON t.id = u.id",
        "WITH RECURSIVE c(id) AS (SELECT 1 UNION SELECT t.id FROM t JOIN c ON t.id = c.id) \
         SELECT id FROM c",
        // UPDATE ... FROM and DELETE ... USING are de-facto joins over the RLS source `t`: a
        // non-superuser must not read its rows through the SET/WHERE, so they are refused too
        // (deep-gate security — previously the source was scanned unfiltered).
        "UPDATE dst SET k = t.id FROM t WHERE dst.k = t.id",
        "DELETE FROM dst USING t WHERE dst.k = t.id",
        // MERGE USING the RLS table `t` as source is the same leak: the source is scanned
        // unfiltered, so a matched action's SET would expose its rows. Refused (deep-gate
        // A-G09.15 security — the target-only RLS check previously let the source through).
        "MERGE INTO dst USING t ON dst.k = t.id WHEN MATCHED THEN UPDATE SET k = t.id",
    ] {
        assert!(
            matches!(alice_err(sql), nusadb_sql::Error::Unsupported(msg) if msg.contains("row-level security")),
            "expected an RLS refusal for `{sql}`",
        );
    }

    // Disabling RLS restores full non-superuser access.
    run(&engine, "ALTER TABLE t DISABLE ROW LEVEL SECURITY");
    assert_eq!(alice_rows("SELECT id FROM t ORDER BY id").len(), 2);
}

#[test]
fn p12_create_policy_filters_rows_end_to_end() {
    // A CREATE POLICY ... USING (...) grants a non-superuser row-filtered SELECT on an
    // RLS-enabled table; with no applicable policy the table is default-deny (zero rows).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE doc (id INT NOT NULL, owner TEXT, body TEXT)",
    );
    run(
        &engine,
        "INSERT INTO doc VALUES (1, 'alice', 'a1'), (2, 'bob', 'b1'), (3, 'alice', 'a2')",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");

    // Run `sql` as a non-superuser `user`: analyze with that user's catalog (so the right policies
    // are selected and the RLS predicate is injected) AND execute in a session set to the same user
    // (so CURRENT_USER inside the predicate evaluates to that user). The two must agree — exactly as
    // a real authenticated connection would.
    let as_user = |user: &'static str, sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user,
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user(user);
        rows(session.execute(plan(logical)).expect("execute"))
    };

    // With RLS on but no policy yet, a non-superuser sees nothing (default-deny), not an error.
    assert_eq!(
        as_user("alice", "SELECT id FROM doc").len(),
        0,
        "default-deny with no policy"
    );

    // A policy that grants each user their own rows (owner = CURRENT_USER).
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    );

    // alice now sees only her two rows.
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]]
    );

    // A different non-superuser the policy does not name (TO alice) still sees nothing.
    assert_eq!(
        as_user("carol", "SELECT id FROM doc").len(),
        0,
        "policy is TO alice, not carol"
    );

    // The superuser bypasses RLS entirely (default `run` catalog is superuser): all three rows.
    assert_eq!(rows(run(&engine, "SELECT id FROM doc")).len(), 3);

    // Dropping the policy returns the table to default-deny for alice.
    run(&engine, "DROP POLICY own ON doc");
    assert_eq!(
        as_user("alice", "SELECT id FROM doc").len(),
        0,
        "default-deny after DROP POLICY"
    );
}

#[test]
fn p12_restrictive_policy_narrows_access_end_to_end() {
    // A RESTRICTIVE policy narrows (AND) what a PERMISSIVE policy grants (OR), per the
    // SQL-standard row-level-security model: alice may read her own rows (permissive) but only the
    // non-classified ones (restrictive). A restrictive policy alone never grants access.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE doc (id INT NOT NULL, owner TEXT, classified INT)",
    );
    run(
        &engine,
        "INSERT INTO doc VALUES (1, 'alice', 0), (2, 'alice', 1), (3, 'bob', 0), (4, 'alice', 0)",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");

    let as_user = |user: &'static str, sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user,
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user(user);
        rows(session.execute(plan(logical)).expect("execute"))
    };

    // Permissive policy first: alice may read her own rows (ids 1, 2, 4).
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    );
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(4)]
        ],
        "permissive policy alone grants all of alice's rows"
    );

    // Add a RESTRICTIVE policy: classified rows are hidden from everyone. alice's view narrows to
    // her non-classified rows (1, 4) — id 2 is classified, id 3 is bob's.
    run(
        &engine,
        "CREATE POLICY hide ON doc AS RESTRICTIVE FOR SELECT USING (classified = 0)",
    );
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)], vec![Value::Int(4)]],
        "restrictive policy ANDs onto the permissive grant"
    );

    // The superuser bypasses RLS entirely: all four rows regardless of either policy.
    assert_eq!(rows(run(&engine, "SELECT id FROM doc")).len(), 4);

    // Dropping the permissive policy leaves only the restrictive one: it never grants access on its
    // own, so alice is back to default-deny (zero rows) even though `classified = 0` matches.
    run(&engine, "DROP POLICY own ON doc");
    assert_eq!(
        as_user("alice", "SELECT id FROM doc").len(),
        0,
        "restrictive policy alone never grants access"
    );
}

#[test]
fn p12_alter_policy_end_to_end() {
    // ALTER POLICY changes a policy's USING and/or TO in place, keeping the command and kind.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(
        &engine,
        "INSERT INTO doc VALUES (1, 'alice'), (2, 'alice'), (3, 'bob')",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    );

    let as_user = |user: &'static str, sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user,
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user(user);
        rows(session.execute(plan(logical)).expect("execute"))
    };

    // Run DDL that must read existing policies (ALTER POLICY) through a policy-aware catalog — the
    // real wire `EngineCatalog` reads policies; the bare test `run` helper does not.
    let run_super = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: true,
                user: "nusa-root",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("nusa-root");
        session.execute(plan(logical)).expect("execute");
    };

    // alice sees her two rows under the original policy.
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );

    // ALTER the USING predicate (keep command SELECT, role alice): alice now sees only id 1.
    run_super("ALTER POLICY own ON doc USING (id = 1)");
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)]],
        "ALTER replaced USING; command and role preserved"
    );

    // ALTER only the role list to bob (USING id = 1 is kept): alice no longer matches, bob does.
    run_super("ALTER POLICY own ON doc TO bob");
    assert_eq!(
        as_user("alice", "SELECT id FROM doc").len(),
        0,
        "role list replaced — alice no longer covered"
    );
    assert_eq!(
        as_user("bob", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)]],
        "USING (id = 1) was preserved across the role-only ALTER"
    );

    // ALTER on a policy that does not exist is an error, not a silent create.
    assert!(
        analyze(
            parse("ALTER POLICY ghost ON doc USING (id = 1)").unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
        .is_err()
    );
}

#[test]
fn p12_alter_policy_preserves_kind_end_to_end() {
    // ALTER POLICY must not change a policy's permissive/restrictive kind: a restrictive policy
    // stays restrictive (AND-narrowing) after an ALTER that only touches its roles.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob')");
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    // A permissive grant for everyone, narrowed by a restrictive own-rows policy.
    run(
        &engine,
        "CREATE POLICY allow ON doc FOR SELECT TO alice USING (TRUE)",
    );
    run(
        &engine,
        "CREATE POLICY hide ON doc AS RESTRICTIVE FOR SELECT USING (owner = CURRENT_USER)",
    );

    let as_user = |user: &'static str, sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user,
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user(user);
        rows(session.execute(plan(logical)).expect("execute"))
    };

    let run_super = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: true,
                user: "nusa-root",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("nusa-root");
        session.execute(plan(logical)).expect("execute");
    };

    // Permissive TRUE AND restrictive owner=alice → alice sees only her row.
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)]]
    );

    // ALTER the restrictive policy's role list only. Had ALTER reset the kind to permissive, the
    // predicate would OR in (no narrowing) and alice would see both rows; it must stay restrictive.
    run_super("ALTER POLICY hide ON doc TO alice");
    assert_eq!(
        as_user("alice", "SELECT id FROM doc ORDER BY id"),
        vec![vec![Value::Int(1)]],
        "restrictive kind preserved across ALTER — still AND-narrows to alice's row"
    );
}

#[test]
fn p12_rls_admin_requires_superuser_end_to_end() {
    // Row-level-security administration (CREATE/ALTER/DROP POLICY and ENABLE/DISABLE RLS) is
    // reserved to superusers. Otherwise a non-superuser — the very session RLS constrains — could
    // lift its own restrictions (disable RLS, loosen a policy, or self-grant) and read everything.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob')");
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    );

    let alice_admin = |sql: &str| {
        analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
    };

    // Every RLS-administration statement run by a non-superuser is permission-denied.
    for sql in [
        "CREATE POLICY hack ON doc FOR SELECT TO alice USING (TRUE)",
        "ALTER POLICY own ON doc USING (TRUE)",
        "DROP POLICY own ON doc",
        "ALTER TABLE doc DISABLE ROW LEVEL SECURITY",
        "ALTER TABLE doc ENABLE ROW LEVEL SECURITY",
    ] {
        assert!(
            matches!(
                alice_admin(sql),
                Err(nusadb_sql::Error::PermissionDenied(_))
            ),
            "expected PermissionDenied for non-superuser `{sql}`",
        );
    }

    // The original policy is intact (no statement above took effect): alice still sees only her row.
    let alice_rows = |sql: &str| {
        let logical = alice_admin(sql).expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        rows(session.execute(plan(logical)).expect("execute"))
    };
    assert_eq!(alice_rows("SELECT id FROM doc"), vec![vec![Value::Int(1)]]);

    // A superuser may administer RLS — the gate is on privilege, not the statement shape.
    let run_super = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: true,
                user: "nusa-root",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("nusa-root");
        session.execute(plan(logical)).expect("execute");
    };
    run_super("ALTER POLICY own ON doc USING (id = 1)");
    run_super("DROP POLICY own ON doc");
}

#[test]
fn p12_rls_no_bypass_via_subquery_or_setop_end_to_end() {
    // A policy filters the RLS table wherever it is read, so a non-superuser cannot launder
    // forbidden rows out through a subquery, set operation, or CTE. Every query block over the base
    // table enforces the policy independently.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(
        &engine,
        "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob'), (3, 'alice')",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO alice USING (owner = CURRENT_USER)",
    );

    let as_alice = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        rows(session.execute(plan(logical)).expect("execute"))
    };

    // Every read shape returns only alice's rows (ids 1 and 3); bob's row (2) never leaks.
    for sql in [
        "SELECT id FROM doc ORDER BY id",
        "SELECT id FROM doc UNION SELECT id FROM doc ORDER BY id",
        "WITH c AS (SELECT id FROM doc) SELECT id FROM c ORDER BY id",
        "SELECT id FROM doc WHERE id IN (SELECT id FROM doc) ORDER BY id",
    ] {
        assert_eq!(
            as_alice(sql),
            vec![vec![Value::Int(1)], vec![Value::Int(3)]],
            "policy must filter every query block for `{sql}`"
        );
    }
}

#[test]
fn p12_restrictive_with_check_narrows_writes_end_to_end() {
    // A RESTRICTIVE WITH CHECK policy narrows what a PERMISSIVE WITH CHECK grants on writes:
    // alice may insert rows she owns (permissive) but only non-classified ones (restrictive).
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE doc (id INT NOT NULL, owner TEXT, classified INT)",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY ins ON doc FOR INSERT TO alice WITH CHECK (owner = CURRENT_USER)",
    );
    run(
        &engine,
        "CREATE POLICY noclass ON doc AS RESTRICTIVE FOR INSERT WITH CHECK (classified = 0)",
    );

    let alice_try = |sql: &str| -> Result<ExecutionResult, nusadb_sql::Error> {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )?;
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        session.execute(plan(logical))
    };

    // Owned and non-classified → passes both policies.
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (1, 'alice', 0)"),
        Ok(ExecutionResult::Inserted(1))
    ));
    // Owned but classified → fails the restrictive WITH CHECK, rejected.
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (2, 'alice', 1)"),
        Err(nusadb_sql::Error::RlsCheckViolation { .. })
    ));
    // Non-classified but owned by someone else → fails the permissive WITH CHECK, rejected.
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (3, 'bob', 0)"),
        Err(nusadb_sql::Error::RlsCheckViolation { .. })
    ));
    // Only the row that satisfied both policies was written.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM doc ORDER BY id")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn p12_delete_policy_filters_rows_end_to_end() {
    // A non-superuser may DELETE only the rows a DELETE/ALL policy grants; with no applicable
    // policy the table is default-deny, so nothing is deleted (never an error, never a full wipe).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(
        &engine,
        "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob'), (3, 'alice')",
    );
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");

    // Run a statement as a non-superuser, analyzing and executing as the same user (matched identity).
    let alice_exec = |sql: &str| {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )
        .expect("analyze");
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        session.execute(plan(logical)).expect("execute")
    };

    // With RLS on but no DELETE policy, a non-superuser's DELETE removes nothing (default-deny).
    assert!(matches!(
        alice_exec("DELETE FROM doc"),
        ExecutionResult::Deleted(0)
    ));
    assert_eq!(
        rows(run(&engine, "SELECT id FROM doc")).len(),
        3,
        "default-deny: no rows deleted without a policy"
    );

    // A DELETE policy lets alice delete only her own rows.
    run(
        &engine,
        "CREATE POLICY del_own ON doc FOR DELETE TO alice USING (owner = CURRENT_USER)",
    );
    assert!(matches!(
        alice_exec("DELETE FROM doc"),
        ExecutionResult::Deleted(2)
    ));
    // Only bob's row (id 2) — which alice's policy does not grant — survives.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM doc ORDER BY id")),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn p12_insert_with_check_end_to_end() {
    // A non-superuser's INSERT must satisfy the INSERT/ALL policies' WITH CHECK; with no
    // applicable policy the table is default-deny, so every non-superuser INSERT is rejected.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");

    // Try `sql` as a non-superuser, returning the Result (so violations can be asserted).
    let alice_try = |sql: &str| -> Result<ExecutionResult, nusadb_sql::Error> {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )?;
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        session.execute(plan(logical))
    };

    // No INSERT policy → default-deny: every non-superuser INSERT is rejected, nothing written.
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (1, 'alice')"),
        Err(nusadb_sql::Error::RlsCheckViolation { .. })
    ));
    assert_eq!(rows(run(&engine, "SELECT id FROM doc")).len(), 0);

    // A WITH CHECK policy lets alice insert only rows she owns.
    run(
        &engine,
        "CREATE POLICY ins ON doc FOR INSERT TO alice WITH CHECK (owner = CURRENT_USER)",
    );
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (1, 'alice')"),
        Ok(ExecutionResult::Inserted(1))
    ));
    // A row owned by someone else fails the WITH CHECK and is rejected.
    assert!(matches!(
        alice_try("INSERT INTO doc VALUES (2, 'bob')"),
        Err(nusadb_sql::Error::RlsCheckViolation { .. })
    ));
    // Only alice's own row was written.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM doc ORDER BY id")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn p12_update_with_check_end_to_end() {
    // UPDATE applies USING (which rows are updatable) AND WITH CHECK (the post-update row must
    // still satisfy the policy, so a row cannot be updated to escape it). With `FOR UPDATE USING`,
    // the USING also serves as the WITH CHECK.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    run(&engine, "INSERT INTO doc VALUES (1, 'alice'), (2, 'bob')");
    run(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    run(
        &engine,
        "CREATE POLICY upd ON doc FOR UPDATE TO alice USING (owner = CURRENT_USER)",
    );

    let alice_try = |sql: &str| -> Result<ExecutionResult, nusadb_sql::Error> {
        let logical = analyze(
            parse(sql).unwrap(),
            &RlsCatalog {
                engine: &engine,
                superuser: false,
                user: "alice",
            },
        )?;
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        session.execute(plan(logical))
    };

    // alice updates a benign column on her own rows (USING matches only owner='alice') → her one row.
    assert!(matches!(
        alice_try("UPDATE doc SET id = 10"),
        Ok(ExecutionResult::Updated(1))
    ));
    // bob's row was untouched (USING excluded it); alice's row now has id 10.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM doc ORDER BY id")),
        vec![vec![Value::Int(2)], vec![Value::Int(10)]]
    );

    // Reassigning ownership away from herself fails the WITH CHECK (the new row would escape the
    // policy) → rejected, and the row is unchanged.
    assert!(matches!(
        alice_try("UPDATE doc SET owner = 'carol'"),
        Err(nusadb_sql::Error::RlsCheckViolation { .. })
    ));
    assert_eq!(
        rows(run(&engine, "SELECT owner FROM doc WHERE id = 10")),
        vec![vec![Value::Text("alice".to_owned())]]
    );
}

#[test]
fn p12_session_functions_end_to_end() {
    // CURRENT_USER / SESSION_USER report the session user (statement-stable); `USER` is a
    // synonym; current_setting(name) reads SET variables and is NULL when unset.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1), (2)");

    // The default session user is reported identically by all three forms, and is stable across rows.
    match run(
        &engine,
        "SELECT CURRENT_USER, SESSION_USER, USER, CURRENT_USER = SESSION_USER FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0][0], Value::Text("nusa-root".to_owned()));
            assert_eq!(rows[0][1], Value::Text("nusa-root".to_owned()));
            assert_eq!(rows[0][2], Value::Text("nusa-root".to_owned()));
            assert_eq!(rows[0][3], Value::Bool(true));
            assert_eq!(rows[0][0], rows[1][0], "session user is stable across rows");
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // A session running as a chosen user reports that user.
    {
        let mut session = Session::new(&engine);
        session.set_current_user("alice");
        match session
            .execute(build_plan(&engine, "SELECT CURRENT_USER"))
            .unwrap()
        {
            ExecutionResult::Rows { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("alice".to_owned()));
            },
            other => panic!("expected rows, got {other:?}"),
        }
    }

    // current_setting reads a SET variable within the same session; an unset name reads back NULL.
    {
        let mut session = Session::new(&engine);
        session
            .execute(build_plan(&engine, "SET tenant = 'acme'"))
            .unwrap();
        match session
            .execute(build_plan(
                &engine,
                "SELECT current_setting('tenant'), current_setting('missing')",
            ))
            .unwrap()
        {
            ExecutionResult::Rows { rows, .. } => {
                assert_eq!(rows[0][0], Value::Text("acme".to_owned()));
                assert_eq!(rows[0][1], Value::Null);
            },
            other => panic!("expected rows, got {other:?}"),
        }
    }
}

#[test]
fn b453_55_math_functions_end_to_end() {
    // Math functions run through the full pipeline, types preserved.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, n INT, f FLOAT)");
    run(&engine, "INSERT INTO t VALUES (1, -7, 2.25), (2, 4, -1.5)");

    // ABS preserves INT/FLOAT; CEIL/FLOOR on FLOAT; SIGN; MOD on INT.
    match run(
        &engine,
        "SELECT ABS(n), ABS(f), CEIL(f), FLOOR(f), SIGN(n), MOD(n, 3) FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Int(7),
                        Value::Float(2.25),
                        Value::Float(3.0),
                        Value::Float(2.0),
                        Value::Int(-1),
                        Value::Int(-1), // -7 mod 3 = -1
                    ],
                    vec![
                        Value::Int(4),
                        Value::Float(1.5),
                        Value::Float(-1.0),
                        Value::Float(-2.0),
                        Value::Int(1),
                        Value::Int(1), // 4 mod 3 = 1
                    ],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // POWER / SQRT compute in FLOAT; usable in a WHERE predicate.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE POWER(n, 2) >= 16 ORDER BY id"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]] // 49>=16 and 16>=16
    );
    match run(&engine, "SELECT SQRT(16.0) FROM t WHERE id = 1") {
        ExecutionResult::Rows { rows, .. } => assert_eq!(rows, vec![vec![Value::Float(4.0)]]),
        other => panic!("expected rows, got {other:?}"),
    }

    // MOD by zero is rejected at run time.
    assert!(run_try(&engine, "SELECT MOD(n, 0) FROM t").is_err());
}

#[test]
fn b457_conditional_functions_end_to_end() {
    // NULLIF / GREATEST / LEAST with NULL-skipping semantics.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, a INT, b INT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 5, 5), (2, 7, 3), (3, 4, NULL)",
    );

    match run(
        &engine,
        "SELECT NULLIF(a, b), GREATEST(a, b), LEAST(a, b) FROM t ORDER BY id",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![
                    // a=b=5 -> NULLIF NULL; greatest/least 5.
                    vec![Value::Null, Value::Int(5), Value::Int(5)],
                    // 7 vs 3.
                    vec![Value::Int(7), Value::Int(7), Value::Int(3)],
                    // b NULL -> NULLIF returns a; GREATEST/LEAST skip NULL -> 4.
                    vec![Value::Int(4), Value::Int(4), Value::Int(4)],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn b456_random_setseed_end_to_end() {
    // RANDOM() volatile FLOAT in [0,1); SETSEED makes the sequence reproducible.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL)");
    run(&engine, "INSERT INTO t VALUES (1)");

    // SETSEED returns BOOL true.
    match run(&engine, "SELECT SETSEED(0.5) FROM t") {
        ExecutionResult::Rows { rows, .. } => assert_eq!(rows, vec![vec![Value::Bool(true)]]),
        other => panic!("expected rows, got {other:?}"),
    }

    // RANDOM() yields a FLOAT in [0, 1).
    match run(&engine, "SELECT RANDOM() FROM t") {
        ExecutionResult::Rows { rows, .. } => match rows.as_slice() {
            [row] => match row.as_slice() {
                [Value::Float(v)] => assert!((0.0..1.0).contains(v), "RANDOM out of range: {v}"),
                other => panic!("expected one Float, got {other:?}"),
            },
            other => panic!("expected one row, got {other:?}"),
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // After re-seeding with the same value, the next RANDOM() repeats (reproducible).
    run(&engine, "SELECT SETSEED(0.25) FROM t");
    let first = rows(run(&engine, "SELECT RANDOM() FROM t"));
    run(&engine, "SELECT SETSEED(0.25) FROM t");
    let second = rows(run(&engine, "SELECT RANDOM() FROM t"));
    assert_eq!(first, second, "SETSEED should make RANDOM reproducible");
}

#[test]
fn b140_regex_match_operator_end_to_end() {
    // `~`/`~*`/`!~`/`!~*` regex match in a WHERE predicate, full pipeline.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'Apple'), (2, 'banana'), (3, 'Cherry42')",
    );

    // Case-sensitive: only lowercase-starting rows.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s ~ '^[a-z]' ORDER BY id"
        )),
        vec![vec![Value::Int(2)]]
    );
    // Case-insensitive matches all starting with a letter.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s ~* '^[a-z]' ORDER BY id"
        )),
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ]
    );
    // Negated: rows that do NOT contain a digit.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s !~ '[0-9]' ORDER BY id"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );
    // Invalid pattern surfaces as an error.
    assert!(run_try(&engine, "SELECT id FROM t WHERE s ~ '('").is_err());
}

#[test]
fn b139_similar_to_end_to_end() {
    // SIMILAR TO is an anchored, SQL-syntax regex match (full pipeline).
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, s TEXT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'abc'), (2, 'abd'), (3, 'xyz'), (4, 'a.c')",
    );

    // `_` matches one char, anchored: 'abc' and 'abd' match 'ab_', 'xyz' does not.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s SIMILAR TO 'ab_' ORDER BY id"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );
    // Alternation + `%` wildcard.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s SIMILAR TO '(abc|xyz)' ORDER BY id"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]]
    );
    // `.` is a LITERAL in SIMILAR TO (unlike POSIX regex): only 'a.c' matches 'a.c', not 'abc'.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s SIMILAR TO 'a.c' ORDER BY id"
        )),
        vec![vec![Value::Int(4)]]
    );
    // Anchored: 'b' does not match the substring of 'abc' (must cover the whole string).
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE s SIMILAR TO 'b'")),
        Vec::<Vec<Value>>::new()
    );
    // NOT SIMILAR TO negates.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE s NOT SIMILAR TO '%c' ORDER BY id"
        )),
        vec![vec![Value::Int(2)], vec![Value::Int(3)]]
    );
}

#[test]
fn b458a_json_path_operators_end_to_end() {
    // #> / #>> path access over a JSON column, full pipeline.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT NOT NULL, doc JSON)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, '{\"a\":{\"b\":42},\"arr\":[10,20,30]}')",
    );

    match run(
        &engine,
        "SELECT doc #> '{a,b}'::text[], doc #>> '{a,b}'::text[], doc #> '{arr,1}'::text[] FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![vec![
                    Value::Json("42".to_owned()),
                    Value::Text("42".to_owned()),
                    Value::Json("20".to_owned()),
                ]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // A missing path yields NULL.
    assert_eq!(
        rows(run(&engine, "SELECT doc #> '{a,z}'::text[] FROM t")),
        vec![vec![Value::Null]]
    );
}

#[test]
fn b132_aggregate_filter_end_to_end() {
    // Aggregate FILTER (WHERE pred): only matching rows contribute, per aggregate.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (id INT NOT NULL, g INT, v INT, flag BOOL)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1,1,10,true),(2,1,20,false),(3,1,30,true),(4,2,40,false)",
    );

    // Scalar: COUNT(*) FILTER, SUM(v) FILTER with different predicates over the same scan.
    match run(
        &engine,
        "SELECT COUNT(*) FILTER (WHERE flag), SUM(v) FILTER (WHERE v >= 20), COUNT(*) FROM t",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            // flag true on rows 1,3 -> 2; v>=20 on rows 2,3,4 -> 20+30+40=90; total rows 4.
            assert_eq!(
                rows,
                vec![vec![Value::Int(2), Value::Int(90), Value::Int(4)]]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }

    // Grouped: FILTER applies within each group.
    match run(
        &engine,
        "SELECT g, COUNT(*) FILTER (WHERE flag) FROM t GROUP BY g ORDER BY g",
    ) {
        ExecutionResult::Rows { rows, .. } => {
            // g=1: rows 1,2,3, flag true on 1,3 -> 2. g=2: row 4, flag false -> 0.
            assert_eq!(
                rows,
                vec![
                    vec![Value::Int(1), Value::Int(2)],
                    vec![Value::Int(2), Value::Int(0)],
                ]
            );
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

// --- decimal literals are exact NUMERIC, with consistent coercion ---

#[test]
fn p0_4_decimal_literal_is_exact_numeric() {
    let engine = BtreeEngine::new();
    // 0.1 + 0.2 is exactly 0.3 as NUMERIC — not the 0.30000000000000004 an f64 literal gives.
    let dec = |s: &str| Value::Numeric(nusadb_sql::numeric::Decimal::parse(s).expect("decimal"));
    assert_eq!(
        rows(run(&engine, "SELECT 0.1 + 0.2")),
        vec![vec![dec("0.3")]]
    );
}

#[test]
fn p0_4_decimal_in_between_case_typecheck_and_filter() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT, v FLOAT)");
    for (i, v) in [(1, "0.5"), (2, "1.5"), (3, "2.5")] {
        run(&engine, &format!("INSERT INTO t VALUES ({i}, {v})"));
    }
    // BETWEEN / IN / simple CASE on a decimal literal must type-check (comparable) and filter
    // correctly rather than raising a spurious TypeMismatch.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM t WHERE v BETWEEN 1.0 AND 2.0")),
        vec![vec![Value::Int(2)]]
    );
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id FROM t WHERE v IN (0.5, 2.5) ORDER BY id"
        )),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]]
    );
    assert!(run_try(&engine, "SELECT CASE v WHEN 1.5 THEN 1 ELSE 0 END FROM t").is_ok());
}

#[test]
fn p0_4_mixed_numeric_case_aggregate_drops_no_rows() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (v FLOAT)");
    for v in ["1.0", "2.0", "3.0"] {
        run(&engine, &format!("INSERT INTO t VALUES ({v})"));
    }
    // CASE mixes a NUMERIC literal (0.5) with a FLOAT column → FLOAT-typed SUM. The accumulator must
    // include the NUMERIC-valued rows in the FLOAT total: 0.5 + 2.0 + 3.0 = 5.5, not 5.0.
    assert_eq!(
        rows(run(
            &engine,
            "SELECT SUM(CASE WHEN v > 1.5 THEN v ELSE 0.5 END) FROM t"
        )),
        vec![vec![Value::Float(5.5)]]
    );

    // NUMERIC-typed direction: an INT branch mixed into a NUMERIC sum must keep the INT rows. Equal
    // to the all-NUMERIC form (both 2.0); if the INT rows were dropped the mixed sum would be 1.0.
    run(&engine, "CREATE TABLE u (id INT)");
    for i in [1, 2, 3] {
        run(&engine, &format!("INSERT INTO u VALUES ({i})"));
    }
    let mixed = rows(run(
        &engine,
        "SELECT SUM(CASE WHEN id <= 2 THEN 0.5 ELSE 1 END) FROM u",
    ));
    let all_numeric = rows(run(
        &engine,
        "SELECT SUM(CASE WHEN id <= 2 THEN 0.5 ELSE 1.0 END) FROM u",
    ));
    assert_eq!(mixed, all_numeric);
}

#[test]
fn p0_4_numeric_value_through_vectorized_float_projection() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (id INT, f FLOAT)");
    run(
        &engine,
        "INSERT INTO t VALUES (1, 1.5), (2, NULL), (3, 2.5)",
    );
    // A FLOAT-typed expression that evaluates to a NUMERIC literal value (COALESCE filling a NULL
    // with 0.5; a CASE with a decimal branch) must round-trip through the vectorized batch path
    // without erroring in row->batch conversion (V-1), and the row and batch paths must agree
    // (both FLOAT, not NUMERIC-by-table-size).
    for sql in [
        "SELECT COALESCE(f, 0.5) FROM t ORDER BY id",
        "SELECT CASE WHEN id = 2 THEN 0.5 ELSE f END FROM t ORDER BY id",
    ] {
        let row_path = rows(run(&engine, sql));
        let batch_path = {
            let _g = nusadb_sql::vectorized::scope(true);
            rows(run(&engine, sql))
        };
        assert_eq!(row_path, batch_path, "vectorized diverged for `{sql}`");
        assert_eq!(
            row_path,
            vec![
                vec![Value::Float(1.5)],
                vec![Value::Float(0.5)],
                vec![Value::Float(2.5)],
            ],
            "wrong result for `{sql}`"
        );
    }
}
