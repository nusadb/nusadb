//! Cursor statements through an embedded `Session`, checking the SQLSTATE of each refusal. The
//! `.slt` corpus (`slt/p5_advanced/cursors.slt`) covers the row results on both the embedded and
//! the wire path, but it can only match error text, and a driver branches on the code.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        Ok(Vec::new())
    }
}

fn run(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    session.execute(plan(logical))
}

fn ok(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> ExecutionResult {
    run(session, engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The SQLSTATE `sql` is refused with.
fn code(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> &'static str {
    match run(session, engine, sql) {
        Ok(result) => panic!("{sql} succeeded: {result:?}"),
        Err(e) => e.sqlstate(),
    }
}

/// The first column of the rows `sql` fetched.
fn fetched(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> Vec<i64> {
    let ExecutionResult::Rows { rows, .. } = ok(session, engine, sql) else {
        panic!("{sql} returned no rows");
    };
    rows.iter()
        .map(|row| match row.first() {
            Some(Value::Int(n)) => *n,
            other => panic!("{sql}: unexpected value {other:?}"),
        })
        .collect()
}

fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    {
        let mut session = Session::new(&engine);
        ok(&mut session, &engine, "CREATE TABLE t (n INT)");
        ok(&mut session, &engine, "INSERT INTO t VALUES (1), (2), (3)");
    }
    engine
}

#[test]
fn a_cursor_without_hold_needs_a_transaction_block() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    assert_eq!(
        code(&mut s, &engine, "DECLARE c CURSOR FOR SELECT n FROM t"),
        "25P01"
    );
    // Nothing was opened by the refused statement.
    assert_eq!(code(&mut s, &engine, "FETCH NEXT FROM c"), "34000");
    assert!(matches!(
        ok(
            &mut s,
            &engine,
            "DECLARE c CURSOR WITH HOLD FOR SELECT n FROM t ORDER BY n"
        ),
        ExecutionResult::CursorDeclared
    ));
    assert_eq!(fetched(&mut s, &engine, "FETCH ABSOLUTE -1 FROM c"), [3]);
}

#[test]
fn redeclaring_an_open_cursor_is_duplicate_cursor_and_keeps_the_original() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "BEGIN");
    ok(
        &mut s,
        &engine,
        "DECLARE c CURSOR FOR SELECT n FROM t ORDER BY n",
    );
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM c"), [1]);
    assert_eq!(
        code(
            &mut s,
            &engine,
            "DECLARE c CURSOR FOR SELECT n FROM t WHERE n > 100"
        ),
        "42P03"
    );
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM c"), [2]);
    ok(&mut s, &engine, "COMMIT");
}

#[test]
fn transaction_end_closes_cursors_that_do_not_outlive_it() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "BEGIN");
    ok(
        &mut s,
        &engine,
        "DECLARE plain CURSOR FOR SELECT n FROM t ORDER BY n",
    );
    ok(
        &mut s,
        &engine,
        "DECLARE held CURSOR WITH HOLD FOR SELECT n FROM t ORDER BY n",
    );
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM held"), [1]);
    ok(&mut s, &engine, "COMMIT");
    assert_eq!(code(&mut s, &engine, "FETCH NEXT FROM plain"), "34000");
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM held"), [2]);

    // A rollback closes what its transaction declared, held or not, and leaves earlier held
    // cursors alone.
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "DECLARE plain2 CURSOR FOR SELECT n FROM t");
    ok(
        &mut s,
        &engine,
        "DECLARE held2 CURSOR WITH HOLD FOR SELECT n FROM t",
    );
    ok(&mut s, &engine, "ROLLBACK");
    assert_eq!(code(&mut s, &engine, "FETCH NEXT FROM plain2"), "34000");
    assert_eq!(code(&mut s, &engine, "FETCH NEXT FROM held2"), "34000");
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM held"), [3]);
}

#[test]
fn move_reports_the_rows_a_fetch_would_have_returned() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "BEGIN");
    ok(
        &mut s,
        &engine,
        "DECLARE c CURSOR FOR SELECT n FROM t ORDER BY n",
    );
    for (sql, count) in [
        ("MOVE FORWARD 100 IN c", 3),
        ("MOVE RELATIVE -2 IN c", 1),
        ("MOVE ABSOLUTE 0 IN c", 0),
        ("MOVE LAST FROM c", 1),
        ("MOVE BACKWARD ALL FROM c", 2),
    ] {
        match ok(&mut s, &engine, sql) {
            ExecutionResult::CursorMoved(n) => assert_eq!(n, count, "{sql}"),
            other => panic!("{sql}: {other:?}"),
        }
    }
    // MOVE positions exactly as FETCH would have: the cursor is now before the first row.
    assert_eq!(fetched(&mut s, &engine, "FETCH NEXT FROM c"), [1]);
    ok(&mut s, &engine, "COMMIT");
}
