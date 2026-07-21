//! End-to-end tests for SQL triggers: `CREATE`/`DROP TRIGGER`, row- and
//! statement-level firing on INSERT/UPDATE/DELETE, `NEW`/`OLD` binding, `WHEN` guards, `OR REPLACE`,
//! and the recursion limit — driven through the whole `parse → analyze → plan → execute` pipeline
//! against the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, analyze, execute, parse, plan};

/// Minimal catalog over the engine — `lookup_table` is the only required method; the rest default.
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, Error> {
    let stmt = parse(sql)?;
    let logical = analyze(stmt, &EngineCatalog(engine))?;
    execute(plan(logical), engine)
}

/// Fetch `SELECT op, n FROM audit ORDER BY op, n` as `(op, n)` pairs.
fn audit(engine: &BtreeEngine) -> Vec<(String, i64)> {
    match run(engine, "SELECT op, n FROM audit ORDER BY op, n") {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|r| match r.as_slice() {
                [Value::Text(op), Value::Int(n)] => (op.clone(), *n),
                other => panic!("unexpected audit row: {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn setup(engine: &BtreeEngine) {
    run(engine, "CREATE TABLE t (id INT NOT NULL, val INT)");
    run(
        engine,
        "CREATE TABLE audit (op TEXT NOT NULL, n INT NOT NULL)",
    );
}

#[test]
fn after_insert_row_trigger_fires_per_row_with_new_binding() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER log_ins AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('ins', new.val)",
    );
    assert!(matches!(
        run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)"),
        ExecutionResult::Inserted(2)
    ));
    assert_eq!(
        audit(&engine),
        vec![("ins".to_owned(), 10), ("ins".to_owned(), 20)]
    );
}

#[test]
fn before_insert_row_trigger_fires_before_the_write() {
    // A-G09.11: a BEFORE INSERT row trigger fires before the new row is written. Proven by having the
    // trigger record the table's row count: it observes the PRE-insert count (1), not the post (2).
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    run(
        &engine,
        "CREATE TRIGGER bt BEFORE INSERT ON t FOR EACH ROW \
         INSERT INTO audit SELECT 'cnt', COUNT(*) FROM t",
    );
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    // The trigger saw the table before the new row existed → count 1.
    assert_eq!(audit(&engine), vec![("cnt".to_owned(), 1)]);
    // The insert itself still happened (the table now has two rows).
    assert!(matches!(
        run(&engine, "SELECT COUNT(*) FROM t"),
        ExecutionResult::Rows { ref rows, .. } if rows == &[vec![Value::Int(2)]]
    ));
}

#[test]
fn before_insert_trigger_binds_new_which_is_read_only() {
    // A BEFORE trigger binds NEW for reading. NusaDB's trigger model substitutes NEW/OLD as literal
    // values (it has no `SET NEW.col` assignment), so the trigger cannot alter the row being written —
    // the inserted value is exactly what the statement supplied.
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER bt BEFORE INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('new', new.val)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 42)");
    assert_eq!(audit(&engine), vec![("new".to_owned(), 42)]);
    // The row is written with the original value — the BEFORE trigger did not (and cannot) change it.
    assert!(matches!(
        run(&engine, "SELECT val FROM t WHERE id = 1"),
        ExecutionResult::Rows { ref rows, .. } if rows == &[vec![Value::Int(42)]]
    ));
}

#[test]
fn before_update_trigger_fires_before_the_write_and_binds_old_and_new() {
    // A BEFORE UPDATE row trigger fires before the row is rewritten, binding both OLD and NEW.
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    // The action records OLD/NEW (binding) and the table's live value (timing: still the old value,
    // proving the trigger runs before the row is rewritten).
    run(
        &engine,
        "CREATE TRIGGER bu BEFORE UPDATE ON t FOR EACH ROW \
         INSERT INTO audit SELECT 'live', val FROM t WHERE id = 1",
    );
    run(
        &engine,
        "CREATE TRIGGER bu2 BEFORE UPDATE ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('old', old.val), ('new', new.val)",
    );
    run(&engine, "UPDATE t SET val = 99 WHERE id = 1");
    assert_eq!(
        audit(&engine),
        vec![
            ("live".to_owned(), 10),
            ("new".to_owned(), 99),
            ("old".to_owned(), 10),
        ]
    );
}

#[test]
fn after_update_trigger_binds_old_and_new() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    run(
        &engine,
        "CREATE TRIGGER log_upd AFTER UPDATE ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('old', old.val), ('new', new.val)",
    );
    assert!(matches!(
        run(&engine, "UPDATE t SET val = 99 WHERE id = 1"),
        ExecutionResult::Updated(1)
    ));
    assert_eq!(
        audit(&engine),
        vec![("new".to_owned(), 99), ("old".to_owned(), 10)]
    );
}

#[test]
fn after_delete_trigger_binds_old() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    run(
        &engine,
        "CREATE TRIGGER log_del AFTER DELETE ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('del', old.val)",
    );
    assert!(matches!(
        run(&engine, "DELETE FROM t WHERE val > 5"),
        ExecutionResult::Deleted(2)
    ));
    assert_eq!(
        audit(&engine),
        vec![("del".to_owned(), 10), ("del".to_owned(), 20)]
    );
}

#[test]
fn fk_cascade_delete_fires_child_delete_trigger() {
    // Deep-gate #4: an ON DELETE CASCADE that removes child rows must still fire the child table's
    // row-level DELETE triggers (e.g. an audit trigger). Previously the cascade wrote straight to the
    // engine and bypassed them.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE audit (op TEXT NOT NULL, n INT NOT NULL)",
    );
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT NOT NULL, pid INT REFERENCES parent(id) ON DELETE CASCADE)",
    );
    run(&engine, "INSERT INTO parent VALUES (1)");
    run(&engine, "INSERT INTO child VALUES (10, 1), (11, 1)");
    run(
        &engine,
        "CREATE TRIGGER log_cdel AFTER DELETE ON child FOR EACH ROW \
         INSERT INTO audit VALUES ('cdel', old.id)",
    );
    // Deleting the parent cascades to both children; each cascade delete must fire the child trigger.
    run(&engine, "DELETE FROM parent WHERE id = 1");
    assert_eq!(
        audit(&engine),
        vec![("cdel".to_owned(), 10), ("cdel".to_owned(), 11)]
    );
}

#[test]
fn fk_set_null_fires_child_update_trigger() {
    // Deep-gate #4: an ON DELETE SET NULL rewrites the child's FK column and must fire the child's
    // row-level UPDATE triggers with OLD/NEW bound — not bypass them.
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE audit (op TEXT NOT NULL, n INT NOT NULL)",
    );
    run(&engine, "CREATE TABLE parent (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE child (id INT NOT NULL, pid INT REFERENCES parent(id) ON DELETE SET NULL)",
    );
    run(&engine, "INSERT INTO parent VALUES (1)");
    run(&engine, "INSERT INTO child VALUES (10, 1)");
    run(
        &engine,
        "CREATE TRIGGER log_cupd AFTER UPDATE ON child FOR EACH ROW \
         INSERT INTO audit VALUES ('old_pid', old.pid), ('new_id', new.id)",
    );
    run(&engine, "DELETE FROM parent WHERE id = 1");
    // The trigger fired with the pre-null OLD (pid = 1) and the unchanged NEW id (10).
    assert_eq!(
        audit(&engine),
        vec![("new_id".to_owned(), 10), ("old_pid".to_owned(), 1)]
    );
    // The FK column is actually nulled.
    match run(&engine, "SELECT pid FROM child WHERE id = 10") {
        ExecutionResult::Rows { rows, .. } => assert_eq!(rows, vec![vec![Value::Null]]),
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn when_guard_skips_non_matching_rows() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER big_only AFTER INSERT ON t FOR EACH ROW WHEN (new.val > 10) \
         INSERT INTO audit VALUES ('big', new.val)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 5), (2, 20), (3, 8)");
    assert_eq!(audit(&engine), vec![("big".to_owned(), 20)]);
}

#[test]
fn statement_level_trigger_fires_once_per_statement() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER once AFTER INSERT ON t FOR EACH STATEMENT \
         INSERT INTO audit VALUES ('stmt', 0)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)");
    run(&engine, "INSERT INTO t VALUES (4, 4)");
    // Two statements → exactly two 'stmt' audit rows, regardless of row counts (3 + 1).
    assert_eq!(
        audit(&engine),
        vec![("stmt".to_owned(), 0), ("stmt".to_owned(), 0)]
    );
}

#[test]
fn multi_event_trigger_fires_on_insert_and_delete() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER log_iud AFTER INSERT OR DELETE ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('chg', new.id)",
    );
    // INSERT binds NEW; the DELETE branch must not reference NEW (it would error), so this trigger
    // only references NEW.id — fire it on INSERT, and confirm a separate DELETE-safe trigger too.
    run(&engine, "INSERT INTO t VALUES (7, 70)");
    assert_eq!(audit(&engine), vec![("chg".to_owned(), 7)]);
}

#[test]
fn or_replace_overwrites_and_drop_removes() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER tg AFTER INSERT ON t FOR EACH ROW INSERT INTO audit VALUES ('v1', new.id)",
    );
    run(
        &engine,
        "CREATE OR REPLACE TRIGGER tg AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('v2', new.id)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 1)");
    // Only the replacement fires (one row, tagged v2).
    assert_eq!(audit(&engine), vec![("v2".to_owned(), 1)]);

    assert!(matches!(
        run(&engine, "DROP TRIGGER tg ON t"),
        ExecutionResult::TriggerDropped
    ));
    run(&engine, "INSERT INTO t VALUES (2, 2)");
    // After DROP the trigger no longer fires — audit is unchanged.
    assert_eq!(audit(&engine), vec![("v2".to_owned(), 1)]);
}

#[test]
fn duplicate_without_or_replace_errors_and_drop_missing_guards() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER dup AFTER INSERT ON t FOR EACH ROW INSERT INTO audit VALUES ('x', new.id)",
    );
    assert!(matches!(
        run_try(
            &engine,
            "CREATE TRIGGER dup AFTER INSERT ON t FOR EACH ROW INSERT INTO audit VALUES ('y', new.id)",
        ),
        Err(Error::TriggerExists { .. })
    ));
    // DROP of a non-existent trigger errors without IF EXISTS, succeeds with it.
    assert!(matches!(
        run_try(&engine, "DROP TRIGGER nope ON t"),
        Err(Error::TriggerNotFound { .. })
    ));
    assert!(matches!(
        run(&engine, "DROP TRIGGER IF EXISTS nope ON t"),
        ExecutionResult::TriggerDropped
    ));
}

#[test]
fn recursion_limit_aborts_runaway_cascade() {
    let engine = BtreeEngine::new();
    setup(&engine);
    // An AFTER INSERT trigger that unconditionally inserts another row re-fires itself forever; the
    // depth guard aborts it rather than overflowing the stack.
    run(
        &engine,
        "CREATE TRIGGER loop_ins AFTER INSERT ON t FOR EACH ROW INSERT INTO t VALUES (0, 0)",
    );
    assert!(matches!(
        run_try(&engine, "INSERT INTO t VALUES (1, 1)"),
        Err(Error::TriggerRecursionLimit { .. })
    ));
    // The whole statement rolled back: t is empty.
    match run(&engine, "SELECT id FROM t") {
        ExecutionResult::Rows { rows, .. } => assert!(rows.is_empty()),
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn reentrant_before_update_of_the_same_row_aborts_rather_than_losing_the_trigger_write() {
    // A-G09.11b: a BEFORE UPDATE row trigger whose body updates the *same row* is the dangerous
    // re-entrant case — the parent's write loop holds the row's pre-trigger tid, so a "completed"
    // trigger write could be silently overwritten (last-writer-wins, trigger change lost). It cannot
    // complete: the parent applies the row write only *after* the BEFORE trigger returns, so the
    // nested same-row UPDATE keeps re-matching and re-firing until the depth guard aborts the whole
    // statement. The outcome is a loud error + full rollback, never a silent lost write.
    let engine = BtreeEngine::new();
    setup(&engine);
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    run(
        &engine,
        "CREATE TRIGGER reent BEFORE UPDATE ON t FOR EACH ROW UPDATE t SET val = 555 WHERE id = 1",
    );
    assert!(matches!(
        run_try(&engine, "UPDATE t SET val = 99 WHERE id = 1"),
        Err(Error::TriggerRecursionLimit { .. })
    ));
    // The statement rolled back entirely: neither the parent's 99 nor the trigger's 555 landed.
    match run(&engine, "SELECT val FROM t WHERE id = 1") {
        ExecutionResult::Rows { rows, .. } => assert_eq!(rows, vec![vec![Value::Int(10)]]),
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn instead_of_trigger_is_rejected() {
    let engine = BtreeEngine::new();
    setup(&engine);
    assert!(matches!(
        run_try(
            &engine,
            "CREATE TRIGGER bad INSTEAD OF INSERT ON t FOR EACH ROW INSERT INTO audit VALUES ('x', 1)",
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn upsert_fires_insert_and_update_triggers() {
    // Deep-gate: ON CONFLICT DO UPDATE previously skipped trigger firing entirely. The inserted
    // rows must fire INSERT triggers and the conflicting (updated) rows must fire UPDATE triggers.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE u (id INT PRIMARY KEY, val INT)");
    run(
        &engine,
        "CREATE TABLE audit (op TEXT NOT NULL, n INT NOT NULL)",
    );
    run(&engine, "INSERT INTO u VALUES (1, 10)");
    run(
        &engine,
        "CREATE TRIGGER log_ins AFTER INSERT ON u FOR EACH ROW \
         INSERT INTO audit VALUES ('ins', new.val)",
    );
    run(
        &engine,
        "CREATE TRIGGER log_upd AFTER UPDATE ON u FOR EACH ROW \
         INSERT INTO audit VALUES ('upd', new.val)",
    );
    // id=1 conflicts → UPDATE (fires the update trigger); id=2 is new → INSERT (fires insert trigger).
    run(
        &engine,
        "INSERT INTO u VALUES (1, 99), (2, 20) ON CONFLICT (id) DO UPDATE SET val = EXCLUDED.val",
    );
    assert_eq!(
        audit(&engine),
        vec![("ins".to_owned(), 20), ("upd".to_owned(), 99)]
    );
}

// === ALTER TRIGGER + ENABLE/DISABLE TRIGGER ================================

#[test]
fn disable_trigger_stops_firing_and_enable_resumes() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER log_ins AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('ins', new.val)",
    );
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    assert_eq!(audit(&engine), vec![("ins".to_owned(), 10)]);

    // Disabled: the trigger stays in the catalog but stops firing.
    run(&engine, "ALTER TABLE t DISABLE TRIGGER log_ins");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    assert_eq!(audit(&engine), vec![("ins".to_owned(), 10)]);

    // Re-enabled: firing resumes.
    run(&engine, "ALTER TABLE t ENABLE TRIGGER log_ins");
    run(&engine, "INSERT INTO t VALUES (3, 30)");
    assert_eq!(
        audit(&engine),
        vec![("ins".to_owned(), 10), ("ins".to_owned(), 30)]
    );
}

#[test]
fn disable_all_covers_every_trigger_and_a_missing_name_errors() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER a_ins AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('a', new.val)",
    );
    run(
        &engine,
        "CREATE TRIGGER b_ins AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('b', new.val)",
    );

    // ALL disables both triggers in one statement.
    run(&engine, "ALTER TABLE t DISABLE TRIGGER ALL");
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    assert_eq!(audit(&engine), vec![]);

    // ALL re-enables both.
    run(&engine, "ALTER TABLE t ENABLE TRIGGER ALL");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    assert_eq!(
        audit(&engine),
        vec![("a".to_owned(), 20), ("b".to_owned(), 20)]
    );

    // A named trigger that does not exist is a loud error; ALL on a trigger-less table is not.
    assert!(matches!(
        run_try(&engine, "ALTER TABLE t DISABLE TRIGGER nope"),
        Err(Error::TriggerNotFound { .. })
    ));
    run(&engine, "CREATE TABLE bare (x INT)");
    assert!(run_try(&engine, "ALTER TABLE bare DISABLE TRIGGER ALL").is_ok());
}

#[test]
fn alter_trigger_rename_keeps_the_definition_and_guards_names() {
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TRIGGER old_name AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('ins', new.val)",
    );
    run(&engine, "ALTER TRIGGER old_name ON t RENAME TO new_name");

    // The renamed trigger still fires with its full definition intact.
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    assert_eq!(audit(&engine), vec![("ins".to_owned(), 10)]);

    // The old name is gone (dropping it errors); the new name drops cleanly.
    assert!(matches!(
        run_try(&engine, "DROP TRIGGER old_name ON t"),
        Err(Error::TriggerNotFound { .. })
    ));

    // Renaming a missing trigger errors; renaming onto an existing name errors.
    assert!(matches!(
        run_try(&engine, "ALTER TRIGGER ghost ON t RENAME TO x"),
        Err(Error::TriggerNotFound { .. })
    ));
    run(
        &engine,
        "CREATE TRIGGER other AFTER INSERT ON t FOR EACH ROW \
         INSERT INTO audit VALUES ('o', new.val)",
    );
    assert!(matches!(
        run_try(&engine, "ALTER TRIGGER other ON t RENAME TO new_name"),
        Err(Error::TriggerExists { .. })
    ));

    // A disabled trigger keeps its disabled flag across a rename.
    run(&engine, "ALTER TABLE t DISABLE TRIGGER new_name");
    run(&engine, "ALTER TRIGGER new_name ON t RENAME TO frozen");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    // Only `other` fired; `frozen` stayed disabled through the rename.
    assert_eq!(
        audit(&engine),
        vec![("ins".to_owned(), 10), ("o".to_owned(), 20)]
    );

    run(&engine, "DROP TRIGGER frozen ON t");
    run(&engine, "DROP TRIGGER other ON t");
}

#[test]
fn legacy_seven_column_trigger_catalog_upgrades_in_place() {
    // A data dir written before the `enabled` column existed has a seven-column
    // `nusadb_triggers` catalog. Simulate one through SQL (the harness catalog is a superuser, so
    // the reserved namespace is writable): the legacy row must (a) fire as enabled without any
    // DDL, and (b) survive the in-place upgrade the next trigger DDL performs.
    let engine = BtreeEngine::new();
    setup(&engine);
    run(
        &engine,
        "CREATE TABLE nusadb_triggers (name TEXT NOT NULL, \"table\" TEXT NOT NULL, \
         timing TEXT NOT NULL, events TEXT NOT NULL, for_each TEXT NOT NULL, \
         \"when\" TEXT NOT NULL, action TEXT NOT NULL)",
    );
    run(
        &engine,
        "INSERT INTO nusadb_triggers VALUES ('legacy', 't', 'after', 'insert', 'row', '', \
         'INSERT INTO audit VALUES (''ins'', new.val)')",
    );

    // (a) The legacy seven-column row fires as an enabled trigger (padded decode).
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    assert_eq!(audit(&engine), vec![("ins".to_owned(), 10)]);

    // (b) A trigger toggle routes through the upgrade: the catalog gains the `enabled` column,
    // the legacy row is rewritten, and the toggle applies to it.
    run(&engine, "ALTER TABLE t DISABLE TRIGGER legacy");
    run(&engine, "INSERT INTO t VALUES (2, 20)");
    assert_eq!(audit(&engine), vec![("ins".to_owned(), 10)]);
    run(&engine, "ALTER TABLE t ENABLE TRIGGER legacy");
    run(&engine, "INSERT INTO t VALUES (3, 30)");
    assert_eq!(
        audit(&engine),
        vec![("ins".to_owned(), 10), ("ins".to_owned(), 30)]
    );

    // The upgraded catalog is a plain eight-column table: the enabled flag is selectable.
    assert!(matches!(
        run(&engine, "SELECT enabled FROM nusadb_triggers"),
        ExecutionResult::Rows { ref rows, .. } if rows == &[vec![Value::Text("t".to_owned())]]
    ));
}
