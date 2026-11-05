//! `COPY (<query>) TO STDOUT` runs its query as the user who issued the `COPY`.
//!
//! The server renders a `COPY` on a worker thread that may last have run another connection's
//! statement. The query must be executed as the COPY user, so `CURRENT_USER` and the row-level
//! security predicates that read it see that user and not whoever ran on the thread before.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::ast::Statement;
use nusadb_sql::{
    Catalog, Error, IndexInfo, Session, analyze, copy_to_bytes_with_columns_in, execute_in_txn_as,
    parse, plan,
};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn copy_out(engine: &BtreeEngine, sql: &str, user: &str, last_user_on_thread: &str) -> Vec<u8> {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    // Another connection's statement ran on this thread last.
    let other = plan(analyze(parse("SELECT 1").unwrap(), &Cat(engine)).unwrap());
    execute_in_txn_as(other, engine, txn, last_user_on_thread).unwrap();
    let Statement::Copy(copy) = parse(sql).unwrap() else {
        panic!("not a COPY: {sql}");
    };
    let (_, _, out) = copy_to_bytes_with_columns_in(engine, &copy, user, txn).unwrap();
    engine.rollback(txn).unwrap();
    out
}

#[test]
fn copy_query_applies_row_security_as_the_copy_user() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    for sql in [
        "CREATE ROLE alice LOGIN",
        "CREATE ROLE bob LOGIN",
        "CREATE TABLE notes (id INT, owner TEXT, body TEXT)",
        "INSERT INTO notes VALUES (1, 'alice', 'for alice'), (2, 'bob', 'for bob')",
        "GRANT SELECT ON notes TO alice, bob",
        "ALTER TABLE notes ENABLE ROW LEVEL SECURITY",
        "CREATE POLICY own ON notes USING (owner = current_user)",
    ] {
        session
            .execute(plan(analyze(parse(sql).unwrap(), &Cat(&engine)).unwrap()))
            .unwrap();
    }
    let sql = "COPY (SELECT body FROM notes ORDER BY id) TO STDOUT";
    assert_eq!(copy_out(&engine, sql, "alice", "bob"), b"for alice\n");
    assert_eq!(copy_out(&engine, sql, "bob", "alice"), b"for bob\n");
    assert_eq!(
        copy_out(
            &engine,
            "COPY (SELECT current_user) TO STDOUT",
            "alice",
            "bob"
        ),
        b"alice\n"
    );
}
