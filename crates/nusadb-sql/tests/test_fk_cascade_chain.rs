//! `ON DELETE` referential actions follow the foreign keys of the rows they delete, to any depth:
//! a cascade into a table whose rows are themselves referenced applies *those* foreign keys'
//! actions too, the way a chain of `ON DELETE CASCADE` keys promises.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, ast::Value, parse, plan,
};

struct Cat<'a> {
    engine: &'a BtreeEngine,
}

impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_list_indexes(self.engine, txn, table);
        let _ = self.engine.commit(txn);
        out
    }
}

fn fresh() -> (&'static BtreeEngine, Session<'static>) {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    (engine, Session::new(engine))
}

fn run(
    engine: &'static BtreeEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat { engine })?;
    session.execute(plan(logical))
}

fn ok(engine: &'static BtreeEngine, session: &mut Session, sql: &str) {
    run(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The ids left in `table`, in order.
fn ids(engine: &'static BtreeEngine, session: &mut Session, table: &str) -> Vec<i64> {
    match run(
        engine,
        session,
        &format!("SELECT id FROM {table} ORDER BY id"),
    )
    .unwrap()
    {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|r| match r.first() {
                Some(Value::Int(n)) => *n,
                other => panic!("expected an id, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn chain(engine: &'static BtreeEngine, session: &mut Session, middle: &str, last: &str) {
    for sql in [
        "CREATE TABLE a (id BIGINT PRIMARY KEY)".to_owned(),
        format!(
            "CREATE TABLE b (id BIGINT PRIMARY KEY, a BIGINT REFERENCES a (id) ON DELETE {middle})"
        ),
        format!(
            "CREATE TABLE c (id BIGINT PRIMARY KEY, b BIGINT REFERENCES b (id) ON DELETE {last})"
        ),
        "INSERT INTO a VALUES (1), (2)".to_owned(),
        "INSERT INTO b VALUES (10, 1), (11, 1), (20, 2)".to_owned(),
        "INSERT INTO c VALUES (100, 10), (101, 11), (102, 11), (200, 20)".to_owned(),
    ] {
        ok(engine, session, &sql);
    }
}

#[test]
fn a_cascade_reaches_the_grandchildren() {
    let (engine, mut session) = fresh();
    chain(engine, &mut session, "CASCADE", "CASCADE");
    ok(engine, &mut session, "DELETE FROM a WHERE id = 1");
    assert_eq!(ids(engine, &mut session, "a"), [2]);
    assert_eq!(ids(engine, &mut session, "b"), [20]);
    assert_eq!(ids(engine, &mut session, "c"), [200]);
}

#[test]
fn set_null_below_a_cascade_keeps_the_grandchildren() {
    let (engine, mut session) = fresh();
    chain(engine, &mut session, "CASCADE", "SET NULL");
    ok(engine, &mut session, "DELETE FROM a WHERE id = 1");
    assert_eq!(ids(engine, &mut session, "b"), [20]);
    assert_eq!(ids(engine, &mut session, "c"), [100, 101, 102, 200]);
    let nulled = run(
        engine,
        &mut session,
        "SELECT count(*) FROM c WHERE b IS NULL",
    )
    .unwrap();
    match nulled {
        ExecutionResult::Rows { rows, .. } => assert_eq!(rows, vec![vec![Value::Int(3)]]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn restrict_below_a_cascade_refuses_the_whole_delete() {
    for action in ["RESTRICT", "NO ACTION"] {
        let (engine, mut session) = fresh();
        chain(engine, &mut session, "CASCADE", action);
        let err = run(engine, &mut session, "DELETE FROM a WHERE id = 1").unwrap_err();
        assert!(err.to_string().contains("foreign key"), "{action}: {err}");
        // Nothing of the statement remains: not the parent, not the cascaded children.
        assert_eq!(ids(engine, &mut session, "a"), [1, 2], "{action}");
        assert_eq!(ids(engine, &mut session, "b"), [10, 11, 20], "{action}");
        assert_eq!(
            ids(engine, &mut session, "c"),
            [100, 101, 102, 200],
            "{action}"
        );
        // With no grandchildren in the way the same delete goes through.
        ok(engine, &mut session, "DELETE FROM c WHERE b IN (10, 11)");
        ok(engine, &mut session, "DELETE FROM a WHERE id = 1");
        assert_eq!(ids(engine, &mut session, "b"), [20], "{action}");
    }
}

#[test]
fn a_self_referencing_tree_is_removed_from_its_root_down() {
    let (engine, mut session) = fresh();
    ok(
        engine,
        &mut session,
        "CREATE TABLE node (id INT PRIMARY KEY, parent INT REFERENCES node (id) ON DELETE CASCADE)",
    );
    ok(
        engine,
        &mut session,
        "INSERT INTO node VALUES (1, NULL), (2, 1), (3, 1), (4, 2), (5, 4), (6, 5), (7, NULL), (8, 7)",
    );
    ok(engine, &mut session, "DELETE FROM node WHERE id = 2");
    assert_eq!(ids(engine, &mut session, "node"), [1, 3, 7, 8]);
    ok(engine, &mut session, "DELETE FROM node WHERE id = 1");
    assert_eq!(ids(engine, &mut session, "node"), [7, 8]);
}

#[test]
fn a_long_chain_does_not_exhaust_the_stack() {
    let (engine, mut session) = fresh();
    ok(
        engine,
        &mut session,
        "CREATE TABLE link (id INT PRIMARY KEY, prev INT REFERENCES link (id) ON DELETE CASCADE)",
    );
    ok(engine, &mut session, "INSERT INTO link VALUES (0, NULL)");
    ok(
        engine,
        &mut session,
        "INSERT INTO link SELECT i, i - 1 FROM generate_series(1, 2000) AS g(i)",
    );
    ok(engine, &mut session, "DELETE FROM link WHERE id = 0");
    assert!(ids(engine, &mut session, "link").is_empty());
}

#[test]
fn a_cycle_of_cascades_ends() {
    let (engine, mut session) = fresh();
    for sql in [
        "CREATE TABLE p (id INT PRIMARY KEY, q INT)",
        "CREATE TABLE q (id INT PRIMARY KEY, p INT REFERENCES p (id) ON DELETE CASCADE)",
        "ALTER TABLE p ADD CONSTRAINT p_q FOREIGN KEY (q) REFERENCES q (id) ON DELETE CASCADE",
        "INSERT INTO p VALUES (1, NULL), (2, NULL)",
        "INSERT INTO q VALUES (10, 1), (20, 2)",
        "UPDATE p SET q = 10 WHERE id = 1",
        "UPDATE p SET q = 20 WHERE id = 2",
    ] {
        ok(engine, &mut session, sql);
    }
    ok(engine, &mut session, "DELETE FROM p WHERE id = 1");
    assert_eq!(ids(engine, &mut session, "p"), [2]);
    assert_eq!(ids(engine, &mut session, "q"), [20]);
}

#[test]
fn a_row_reached_by_two_paths_is_deleted_once() {
    let (engine, mut session) = fresh();
    for sql in [
        "CREATE TABLE top (id INT PRIMARY KEY)",
        "CREATE TABLE left_side (id INT PRIMARY KEY, t INT REFERENCES top (id) ON DELETE CASCADE)",
        "CREATE TABLE right_side (id INT PRIMARY KEY, t INT REFERENCES top (id) ON DELETE CASCADE)",
        "CREATE TABLE bottom (id INT PRIMARY KEY, \
         l INT REFERENCES left_side (id) ON DELETE CASCADE, \
         r INT REFERENCES right_side (id) ON DELETE CASCADE)",
        "CREATE TABLE audit (id INT PRIMARY KEY, n INT)",
        "INSERT INTO audit VALUES (1, 0)",
        "CREATE TRIGGER count_bottom AFTER DELETE ON bottom FOR EACH ROW \
         UPDATE audit SET n = n + 1 WHERE id = 1",
        "INSERT INTO top VALUES (1)",
        "INSERT INTO left_side VALUES (1, 1)",
        "INSERT INTO right_side VALUES (1, 1)",
        "INSERT INTO bottom VALUES (1, 1, 1), (2, 1, 1)",
    ] {
        ok(engine, &mut session, sql);
    }
    ok(engine, &mut session, "DELETE FROM top WHERE id = 1");
    assert!(ids(engine, &mut session, "bottom").is_empty());
    match run(engine, &mut session, "SELECT n FROM audit").unwrap() {
        ExecutionResult::Rows { rows, .. } => {
            assert_eq!(
                rows,
                vec![vec![Value::Int(2)]],
                "each bottom row deleted once"
            );
        },
        other => panic!("{other:?}"),
    }
}

fn count(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> i64 {
    match run(engine, session, sql).unwrap() {
        ExecutionResult::Rows { rows, .. } => match rows.first().and_then(|r| r.first()) {
            Some(Value::Int(n)) => *n,
            other => panic!("expected a count, got {other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_refused_delete_inside_a_transaction_changes_nothing() {
    let (engine, mut session) = fresh();
    chain(engine, &mut session, "CASCADE", "RESTRICT");
    ok(engine, &mut session, "BEGIN");
    ok(engine, &mut session, "INSERT INTO a VALUES (3)");
    assert!(run(engine, &mut session, "DELETE FROM a WHERE id = 1").is_err());
    ok(engine, &mut session, "ROLLBACK");
    assert_eq!(ids(engine, &mut session, "a"), [1, 2]);
    assert_eq!(ids(engine, &mut session, "b"), [10, 11, 20]);
    assert_eq!(ids(engine, &mut session, "c"), [100, 101, 102, 200]);
}

#[test]
fn no_action_allows_a_dependant_the_statement_removes_another_way() {
    // `c` references `t` directly (NO ACTION) and through `d` (CASCADE): the cascade removes it,
    // so at the end of the statement nothing references the deleted row. RESTRICT refuses at once.
    for (action, allowed) in [("NO ACTION", true), ("RESTRICT", false)] {
        let (engine, mut session) = fresh();
        for sql in [
            "CREATE TABLE t (id INT PRIMARY KEY)".to_owned(),
            "CREATE TABLE d (id INT PRIMARY KEY, t INT REFERENCES t (id) ON DELETE CASCADE)"
                .to_owned(),
            format!(
                "CREATE TABLE c (id INT PRIMARY KEY, t INT REFERENCES t (id) ON DELETE {action}, \
                 d INT REFERENCES d (id) ON DELETE CASCADE)"
            ),
            "INSERT INTO t VALUES (1), (2)".to_owned(),
            "INSERT INTO d VALUES (10, 1)".to_owned(),
            "INSERT INTO c VALUES (100, 1, 10)".to_owned(),
        ] {
            ok(engine, &mut session, &sql);
        }
        let result = run(engine, &mut session, "DELETE FROM t WHERE id = 1");
        assert_eq!(result.is_ok(), allowed, "{action}: {result:?}");
        let left = if allowed {
            (vec![2], 0)
        } else {
            (vec![1, 2], 1)
        };
        assert_eq!(ids(engine, &mut session, "t"), left.0, "{action}");
        assert_eq!(
            count(engine, &mut session, "SELECT count(*) FROM c"),
            left.1,
            "{action}"
        );
    }
}

#[test]
fn a_row_both_nulled_and_cascaded_is_only_deleted() {
    // `c` is reached by a SET NULL key from `t` and, one level down, by a CASCADE key from `d`.
    // The row goes, and is never rewritten with NULL on its way out.
    let (engine, mut session) = fresh();
    for sql in [
        "CREATE TABLE t (id INT PRIMARY KEY)",
        "CREATE TABLE d (id INT PRIMARY KEY, t INT REFERENCES t (id) ON DELETE CASCADE)",
        "CREATE TABLE c (id INT PRIMARY KEY, t INT REFERENCES t (id) ON DELETE SET NULL, \
         d INT REFERENCES d (id) ON DELETE CASCADE)",
        "CREATE TABLE audit (id INT PRIMARY KEY, updates INT)",
        "INSERT INTO audit VALUES (1, 0)",
        "CREATE TRIGGER c_updates AFTER UPDATE ON c FOR EACH ROW \
         UPDATE audit SET updates = updates + 1 WHERE id = 1",
        "INSERT INTO t VALUES (1)",
        "INSERT INTO d VALUES (10, 1)",
        "INSERT INTO c VALUES (100, 1, 10), (101, 1, NULL)",
    ] {
        ok(engine, &mut session, sql);
    }
    ok(engine, &mut session, "DELETE FROM t WHERE id = 1");
    // 100 is cascaded away; 101 has no `d`, so it stays with its `t` nulled, the one rewrite.
    assert_eq!(ids(engine, &mut session, "c"), [101]);
    assert_eq!(
        count(
            engine,
            &mut session,
            "SELECT count(*) FROM c WHERE t IS NULL"
        ),
        1
    );
    assert_eq!(count(engine, &mut session, "SELECT updates FROM audit"), 1);
}

#[test]
fn two_set_null_keys_on_one_row_both_take_effect() {
    let (engine, mut session) = fresh();
    for sql in [
        "CREATE TABLE p (id INT PRIMARY KEY)",
        "CREATE TABLE r (id INT PRIMARY KEY, x INT REFERENCES p (id) ON DELETE SET NULL, \
         y INT REFERENCES p (id) ON DELETE SET NULL)",
        "INSERT INTO p VALUES (1), (2), (3)",
        "INSERT INTO r VALUES (10, 1, 2), (11, 1, 3)",
    ] {
        ok(engine, &mut session, sql);
    }
    ok(engine, &mut session, "DELETE FROM p WHERE id IN (1, 2)");
    assert_eq!(
        count(
            engine,
            &mut session,
            "SELECT count(*) FROM r WHERE x IS NULL AND y IS NULL"
        ),
        1
    );
    assert_eq!(
        count(
            engine,
            &mut session,
            "SELECT count(*) FROM r WHERE x IS NULL AND y = 3"
        ),
        1
    );
}
