//! `INSERT ... ON CONFLICT`: `DO NOTHING` silently skips a row that would violate a
//! PRIMARY KEY / UNIQUE constraint while the non-conflicting rows still insert; `DO UPDATE`
//! (upsert) updates the existing conflicting row from the proposed `EXCLUDED` row instead.

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

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { mut rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

#[test]
fn do_nothing_skips_conflicting_rows() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b')",
    );

    // Without ON CONFLICT, a duplicate key errors.
    assert!(try_exec(engine, &mut session, "INSERT INTO t VALUES (1, 'x')").is_err());

    // ON CONFLICT DO NOTHING: id 1 conflicts (skipped), id 3 is new (inserted) — one row inserted.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 'x'), (3, 'c') ON CONFLICT DO NOTHING"
        ),
        ExecutionResult::Inserted(1)
    ));
    // The pre-existing id-1 row keeps its original value (DO NOTHING does not overwrite).
    assert_eq!(
        rows(engine, &mut session, "SELECT id, v FROM t ORDER BY id"),
        vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Int(2), Value::Text("b".to_owned())],
            vec![Value::Int(3), Value::Text("c".to_owned())],
        ]
    );

    // A conflict within the same batch: the first (4,'d') is kept, the second (4,'e') is skipped.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (4, 'd'), (4, 'e') ON CONFLICT DO NOTHING"
        ),
        ExecutionResult::Inserted(1)
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT v FROM t WHERE id = 4"),
        vec![vec![Value::Text("d".to_owned())]]
    );

    // RETURNING reports only the rows actually inserted.
    let ExecutionResult::Rows { rows, .. } = exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (4, 'z'), (5, 'e') ON CONFLICT DO NOTHING RETURNING id",
    ) else {
        panic!("expected RETURNING rows");
    };
    assert_eq!(rows, vec![vec![Value::Int(5)]]);
}

#[test]
fn do_nothing_honors_a_unique_constraint() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE u (id INT PRIMARY KEY, email TEXT UNIQUE)",
    );
    exec(engine, &mut session, "INSERT INTO u VALUES (1, 'a@x')");

    // A new id but a duplicate UNIQUE email is skipped.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO u VALUES (2, 'a@x'), (3, 'b@x') ON CONFLICT DO NOTHING"
        ),
        ExecutionResult::Inserted(1)
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM u ORDER BY id"),
        vec![vec![Value::Int(1)], vec![Value::Int(3)]]
    );
}

#[test]
fn do_update_upserts_existing_rows() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b')",
    );

    // ON CONFLICT (id) DO NOTHING (now with a target) skips the conflict and inserts the rest.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 'x'), (3, 'c') ON CONFLICT (id) DO NOTHING"
        ),
        ExecutionResult::Inserted(1)
    ));

    // DO UPDATE: id=2 conflicts → its `v` becomes EXCLUDED.v; id=4 inserts. Two rows affected.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (2, 'B2'), (4, 'd') ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v"
        ),
        ExecutionResult::Inserted(2)
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT id, v FROM t"),
        vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Int(2), Value::Text("B2".to_owned())],
            vec![Value::Int(3), Value::Text("c".to_owned())],
            vec![Value::Int(4), Value::Text("d".to_owned())],
        ]
    );

    // WHERE gates the update: a bare column is the existing row, `EXCLUDED` the proposed one — here
    // the values are equal, so the predicate is false and nothing is affected.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 'a') ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v WHERE t.v <> EXCLUDED.v"
        ),
        ExecutionResult::Inserted(0)
    ));

    // RETURNING projects the final (updated) row.
    let ExecutionResult::Rows { rows, .. } = exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 'A1') ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v RETURNING id, v",
    ) else {
        panic!("expected RETURNING rows");
    };
    assert_eq!(
        rows,
        vec![vec![Value::Int(1), Value::Text("A1".to_owned())]]
    );
}

#[test]
fn do_update_revalidates_secondary_unique() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, u INT, UNIQUE (u))",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 10), (2, 20)",
    );

    // A DO UPDATE that moves a *secondary* unique column (`u`) onto another row's value must be
    // rejected — exactly as a plain UPDATE would be — not silently accepted.
    assert!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 20) ON CONFLICT (id) DO UPDATE SET u = EXCLUDED.u"
        )
        .is_err(),
        "DO UPDATE creating a duplicate secondary-UNIQUE value must be rejected"
    );
    // The conflicting row is unchanged after the rejected upsert.
    assert_eq!(
        rows(engine, &mut session, "SELECT id, u FROM t"),
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );

    // A DO UPDATE to a non-colliding unique value still succeeds.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 99) ON CONFLICT (id) DO UPDATE SET u = EXCLUDED.u"
        ),
        ExecutionResult::Inserted(1)
    ));
    assert_eq!(
        rows(engine, &mut session, "SELECT id, u FROM t WHERE id = 1"),
        vec![vec![Value::Int(1), Value::Int(99)]]
    );
}

#[test]
fn concurrent_insert_same_pk_yields_exactly_one_winner() {
    use std::thread;

    // A `&'static` engine is shared across threads directly. Each thread runs an autocommit INSERT
    // of the SAME primary key; before two READ COMMITTED inserts could both pass the
    // snapshot-based uniqueness scan and commit a duplicate. The key-level lock now serializes them,
    // so exactly one wins and no duplicate is persisted.
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    {
        let mut session = Session::new(engine);
        exec(
            engine,
            &mut session,
            "CREATE TABLE t (id INT PRIMARY KEY, who INT)",
        );
    }

    // Spawn all writers first (they must race), then join — a loop, not collect(), so each thread
    // is live before any is joined.
    let mut handles = Vec::new();
    for k in 0..16i64 {
        handles.push(thread::spawn(move || {
            let mut session = Session::new(engine);
            try_exec(
                engine,
                &mut session,
                &format!("INSERT INTO t VALUES (1, {k})"),
            )
            .is_ok()
        }));
    }
    let mut wins = 0;
    for handle in handles {
        if handle.join().unwrap_or(false) {
            wins += 1;
        }
    }

    assert_eq!(
        wins, 1,
        "exactly one concurrent INSERT of the same primary key must win"
    );
    let mut session = Session::new(engine);
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t"),
        vec![vec![Value::Int(1)]],
        "no duplicate primary key may be persisted",
    );
}

#[test]
fn do_update_honest_rejects() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
    );

    // DO UPDATE requires a conflict target.
    assert!(matches!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 'a') ON CONFLICT DO UPDATE SET v = 'b'"
        ),
        Err(Error::Unsupported(_))
    ));
    // The target columns must match a declared UNIQUE / PRIMARY KEY constraint.
    assert!(
        try_exec(
            engine,
            &mut session,
            "INSERT INTO t VALUES (1, 'a') ON CONFLICT (v) DO UPDATE SET v = 'b'"
        )
        .is_err()
    );
}
