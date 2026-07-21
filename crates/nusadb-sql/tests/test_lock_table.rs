//! `LOCK TABLE` end-to-end. The SQL surface resolves the named tables and acquires a
//! table-level lock via the engine's lock manager. This covers the wiring — default and
//! explicit modes, multiple tables, and resolution errors; the lock manager's own conflict/deadlock
//! semantics are covered by the engine's own lock tests.

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
fn lock_table_acquires_locks_and_reports() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE a (x INT)");
    exec(engine, &mut session, "CREATE TABLE b (x INT)");

    // Inside an explicit transaction the locks are held until COMMIT.
    exec(engine, &mut session, "BEGIN");
    assert!(matches!(
        exec(engine, &mut session, "LOCK TABLE a"),
        ExecutionResult::TableLocked
    ));
    assert!(matches!(
        exec(engine, &mut session, "LOCK TABLE a, b IN ACCESS SHARE MODE"),
        ExecutionResult::TableLocked
    ));
    exec(engine, &mut session, "COMMIT");

    // Auto-commit also works (the lock is released at the implicit commit).
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "LOCK TABLE b IN ACCESS EXCLUSIVE MODE"
        ),
        ExecutionResult::TableLocked
    ));

    // Locking a non-existent table is a clean resolution error, not a panic.
    let err = try_exec(engine, &mut session, "LOCK TABLE ghost").expect_err("must error");
    assert!(matches!(err, Error::TableNotFound { .. }), "got {err:?}");
}
