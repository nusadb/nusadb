//! Transaction control and per-connection statements inside a routine body (a procedure, a `DO`
//! block, a NusaScript function or a trigger function) are refused with `0A000`, and the statement
//! that invoked the routine fails as a whole. The `.slt` corpus
//! (`slt/p12_txn/routine_transaction_control.slt`) pins the messages; this file asserts the codes
//! and the state of the caller's transaction afterwards.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, Session, analyze, parse, plan};

struct EngineCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: Option<nusadb_core::TxnId>,
}

impl Catalog for EngineCatalog<'_> {
    nusadb_test_utils::catalog_answers_as_superuser!(nusadb_sql);
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.txn
            .map_or_else(
                || self.engine.lookup_table(name),
                |txn| self.engine.lookup_table_as_of(txn, name),
            )
            .map_err(Into::into)
    }

    fn lookup_function(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_sql::FunctionDef>, nusadb_sql::Error> {
        if let Some(txn) = self.txn {
            return nusadb_sql::lookup_function_definition(self.engine, txn, name);
        }
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let found = nusadb_sql::lookup_function_definition(self.engine, txn, name);
        self.engine.commit(txn)?;
        found
    }
}

fn run(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    let catalog = EngineCatalog {
        engine,
        txn: session.current_txn(),
    };
    let logical = analyze(parse(sql)?, &catalog)?;
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

/// The values of `t.v`, in order.
fn values(session: &mut Session, engine: &dyn StorageEngine) -> Vec<i64> {
    let ExecutionResult::Rows { rows, .. } = ok(session, engine, "SELECT v FROM t ORDER BY v")
    else {
        panic!("no rows");
    };
    rows.iter()
        .map(|row| match row.first() {
            Some(Value::Int(n)) => *n,
            other => panic!("unexpected value {other:?}"),
        })
        .collect()
}

fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    {
        let mut s = Session::new(&engine);
        ok(&mut s, &engine, "CREATE TABLE t (v INT)");
        ok(
            &mut s,
            &engine,
            "CREATE PROCEDURE p_commit() AS $$ BEGIN INSERT INTO t VALUES (1); COMMIT; \
             INSERT INTO t VALUES (2); ROLLBACK; END $$",
        );
    }
    engine
}

#[test]
fn transaction_control_in_every_kind_of_routine_is_0a000() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(
        &mut s,
        &engine,
        "CREATE FUNCTION f_commit() RETURNS INT LANGUAGE nusascript AS $$ BEGIN \
         INSERT INTO t VALUES (3); COMMIT; RETURN 1; END $$",
    );
    ok(
        &mut s,
        &engine,
        "CREATE FUNCTION trg_rollback() RETURNS trigger LANGUAGE nusascript AS $$ BEGIN \
         ROLLBACK; RETURN NEW; END $$",
    );
    ok(&mut s, &engine, "CREATE TABLE trg_t (v INT)");
    ok(
        &mut s,
        &engine,
        "CREATE TRIGGER trg BEFORE INSERT ON trg_t FOR EACH ROW EXECUTE FUNCTION trg_rollback()",
    );
    for sql in [
        "CALL p_commit()",
        "SELECT f_commit()",
        "INSERT INTO trg_t VALUES (1)",
        "DO $$ COMMIT; $$",
        "DO $$ ROLLBACK; $$",
        "DO $$ BEGIN; $$",
        "DO $$ START TRANSACTION; $$",
        "DO $$ SAVEPOINT a; $$",
        "DO $$ RELEASE SAVEPOINT a; $$",
        "DO $$ ROLLBACK TO SAVEPOINT a; $$",
        "DO $$ BEGIN INSERT INTO t VALUES (4); COMMIT; END $$",
        "DO $$ BEGIN SAVEPOINT a; END $$",
    ] {
        assert_eq!(code(&mut s, &engine, sql), "0A000", "{sql}");
    }
    // Nothing the bodies wrote before the refusal survives.
    assert_eq!(values(&mut s, &engine), Vec::<i64>::new());
    let ExecutionResult::Rows { rows, .. } = ok(&mut s, &engine, "SELECT count(*) FROM trg_t")
    else {
        panic!("no rows");
    };
    assert_eq!(rows, vec![vec![Value::Int(0)]]);
}

#[test]
fn connection_statements_in_a_routine_are_0a000() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    for sql in [
        "DO $$ SET search_path = public; $$",
        "DO $$ BEGIN SHOW search_path; END $$",
        "DO $$ BEGIN PREPARE q AS SELECT 1; END $$",
        "DO $$ BEGIN EXECUTE q; END $$",
        "DO $$ BEGIN DEALLOCATE ALL; END $$",
        "DO $$ BEGIN FETCH NEXT FROM c; END $$",
        "DO $$ BEGIN CLOSE c; END $$",
    ] {
        assert_eq!(code(&mut s, &engine, sql), "0A000", "{sql}");
    }
}

#[test]
fn the_callers_transaction_keeps_its_own_writes_and_loses_the_routines() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO t VALUES (100)");
    assert_eq!(code(&mut s, &engine, "CALL p_commit()"), "0A000");
    // The refused CALL neither committed nor ended the transaction: it is still open, holds the
    // caller's own row, and none of the procedure's.
    assert!(s.current_txn().is_some(), "the transaction is still open");
    assert_eq!(values(&mut s, &engine), vec![100]);
    ok(&mut s, &engine, "ROLLBACK");
    assert_eq!(values(&mut s, &engine), Vec::<i64>::new());

    // In auto-commit the CALL is its own transaction: nothing of it is kept.
    assert_eq!(code(&mut s, &engine, "CALL p_commit()"), "0A000");
    assert_eq!(values(&mut s, &engine), Vec::<i64>::new());
}

#[test]
fn an_exception_handler_catches_the_refusal_like_any_error() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(
        &mut s,
        &engine,
        "DO $$ BEGIN INSERT INTO t VALUES (9); COMMIT; \
         EXCEPTION WHEN OTHERS THEN INSERT INTO t VALUES (10); END $$",
    );
    assert_eq!(values(&mut s, &engine), vec![10]);
}
