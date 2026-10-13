//! Two-phase commit and notices through embedded `Session`s over one engine: a transaction
//! prepared in one session is finished from another, its writes stay invisible and its rows locked
//! until then, every refusal carries its SQLSTATE, and the `nusadb_prepared_xacts` view shows each
//! session what it may see. Notices raised by a statement are taken from the session that ran it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{PreparedTransaction, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::notice::Severity;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

/// A catalog over the engine as `user` sees it (a superuser unless it says otherwise).
struct EngineCatalog<'a> {
    engine: &'a dyn StorageEngine,
    user: String,
    superuser: bool,
    temp_schema: String,
}

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine
            .lookup_table_in(schema, name)
            .map_err(Into::into)
    }
    fn temp_schema(&self) -> Option<String> {
        Some(self.temp_schema.clone())
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        Ok(Vec::new())
    }
    fn is_superuser(&self) -> bool {
        self.superuser
    }
    fn current_user(&self) -> String {
        self.user.clone()
    }
    fn prepared_transactions(&self) -> Result<Vec<PreparedTransaction>, nusadb_sql::Error> {
        self.engine.list_prepared().map_err(Into::into)
    }
}

fn run(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    let user = session.current_user().to_owned();
    let catalog = EngineCatalog {
        engine,
        superuser: user == nusadb_sql::BOOTSTRAP_SUPERUSER,
        user,
        temp_schema: session.temp_schema_name(),
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

/// The rows `sql` returns, every value rendered as text.
fn rows(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> Vec<Vec<String>> {
    let ExecutionResult::Rows { rows, .. } = ok(session, engine, sql) else {
        panic!("{sql} returned no rows");
    };
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|v| match v {
                    Value::Int(n) => n.to_string(),
                    Value::Text(s) => s.clone(),
                    Value::Null => "NULL".to_owned(),
                    other => format!("{other:?}"),
                })
                .collect()
        })
        .collect()
}

/// The values of table `t`, as a fresh session sees them.
fn values(engine: &dyn StorageEngine) -> Vec<String> {
    let mut s = Session::new(engine);
    rows(&mut s, engine, "SELECT v FROM t ORDER BY v")
        .into_iter()
        .map(|mut row| row.remove(0))
        .collect()
}

fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    {
        let mut s = Session::new(&engine);
        ok(
            &mut s,
            &engine,
            "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
        );
        ok(&mut s, &engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    }
    engine
}

#[test]
fn a_transaction_prepared_in_one_session_is_committed_from_another() {
    let engine = fixture();
    let mut s1 = Session::new(&engine);
    let mut s2 = Session::new(&engine);
    ok(&mut s1, &engine, "BEGIN");
    ok(&mut s1, &engine, "UPDATE t SET v = 'A' WHERE id = 1");
    ok(&mut s1, &engine, "INSERT INTO t VALUES (3, 'c')");
    assert!(matches!(
        ok(&mut s1, &engine, "PREPARE TRANSACTION 'g1'"),
        ExecutionResult::TransactionPrepared
    ));
    // The session is back in auto-commit and may go on with other work.
    assert!(!s1.in_transaction());
    assert_eq!(values(&engine), vec!["a", "b"]);
    // The prepared transaction still holds the row it changed: a second writer is refused at
    // once, as by any concurrent writer.
    ok(&mut s2, &engine, "BEGIN");
    assert_eq!(
        code(&mut s2, &engine, "UPDATE t SET v = 'X' WHERE id = 1"),
        "40001"
    );
    ok(&mut s2, &engine, "ROLLBACK");
    assert!(matches!(
        ok(&mut s2, &engine, "COMMIT PREPARED 'g1'"),
        ExecutionResult::PreparedCommitted
    ));
    assert_eq!(values(&engine), vec!["A", "b", "c"]);
    ok(&mut s2, &engine, "UPDATE t SET v = 'X' WHERE id = 1");
    assert_eq!(values(&engine), vec!["X", "b", "c"]);
}

#[test]
fn rollback_prepared_discards_the_writes() {
    let engine = fixture();
    let mut s1 = Session::new(&engine);
    ok(&mut s1, &engine, "BEGIN");
    ok(&mut s1, &engine, "DELETE FROM t WHERE id = 2");
    ok(&mut s1, &engine, "PREPARE TRANSACTION 'gone'");
    let mut s2 = Session::new(&engine);
    assert!(matches!(
        ok(&mut s2, &engine, "ROLLBACK PREPARED 'gone'"),
        ExecutionResult::PreparedRolledBack
    ));
    assert_eq!(values(&engine), vec!["a", "b"]);
    assert_eq!(code(&mut s2, &engine, "ROLLBACK PREPARED 'gone'"), "42704");
}

#[test]
fn every_refusal_carries_its_sqlstate() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    // No transaction to prepare.
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'x'"), "25P01");
    // Finishing a name nobody prepared.
    assert_eq!(code(&mut s, &engine, "COMMIT PREPARED 'nope'"), "42704");
    // A duplicate identifier: the second transaction is rolled back.
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO t VALUES (10, 'p')");
    ok(&mut s, &engine, "PREPARE TRANSACTION 'dup'");
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO t VALUES (11, 'q')");
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'dup'"), "42710");
    assert!(!s.in_transaction());
    // An identifier past 200 bytes.
    ok(&mut s, &engine, "BEGIN");
    let long = format!("PREPARE TRANSACTION '{}'", "g".repeat(201));
    assert_eq!(code(&mut s, &engine, &long), "22023");
    assert!(!s.in_transaction());
    // COMMIT PREPARED inside a transaction block.
    ok(&mut s, &engine, "BEGIN");
    assert_eq!(code(&mut s, &engine, "COMMIT PREPARED 'dup'"), "25001");
    assert_eq!(code(&mut s, &engine, "ROLLBACK PREPARED 'dup'"), "25001");
    ok(&mut s, &engine, "ROLLBACK");
    // A schema change, and a temporary table, cannot be prepared.
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "CREATE TABLE u (n INT)");
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'ddl'"), "0A000");
    ok(&mut s, &engine, "CREATE TEMP TABLE scratch (n INT)");
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO scratch VALUES (1)");
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'temp'"), "0A000");
    // A cursor that would outlive the transaction.
    ok(&mut s, &engine, "BEGIN");
    ok(
        &mut s,
        &engine,
        "DECLARE c CURSOR WITH HOLD FOR SELECT v FROM t",
    );
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'held'"), "0A000");
    // The ceiling, and disabled prepared transactions.
    engine.set_max_prepared_transactions(1);
    ok(&mut s, &engine, "BEGIN");
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'more'"), "53400");
    engine.set_max_prepared_transactions(0);
    ok(&mut s, &engine, "BEGIN");
    assert_eq!(code(&mut s, &engine, "PREPARE TRANSACTION 'off'"), "55000");
    // Of everything above only the first prepared insert is still pending.
    assert_eq!(values(&engine), vec!["a", "b"]);
    ok(&mut s, &engine, "COMMIT PREPARED 'dup'");
    assert_eq!(values(&engine), vec!["a", "b", "p"]);
}

#[test]
fn only_the_preparing_role_or_a_superuser_may_finish_it() {
    let engine = fixture();
    let mut root = Session::new(&engine);
    ok(&mut root, &engine, "CREATE ROLE alice LOGIN");
    ok(&mut root, &engine, "CREATE ROLE bob LOGIN");
    let mut alice = Session::new(&engine);
    alice.set_current_user("alice");
    let mut bob = Session::new(&engine);
    bob.set_current_user("bob");
    for gid in ["a1", "a2"] {
        ok(&mut alice, &engine, "BEGIN");
        ok(&mut alice, &engine, &format!("PREPARE TRANSACTION '{gid}'"));
    }
    assert_eq!(code(&mut bob, &engine, "COMMIT PREPARED 'a1'"), "42501");
    assert_eq!(code(&mut bob, &engine, "ROLLBACK PREPARED 'a1'"), "42501");
    ok(&mut alice, &engine, "COMMIT PREPARED 'a1'");
    ok(&mut root, &engine, "ROLLBACK PREPARED 'a2'");
    assert!(engine.list_prepared().unwrap().is_empty());
}

#[test]
fn the_view_lists_what_each_role_may_see() {
    let engine = fixture();
    let mut root = Session::new(&engine);
    // Nothing prepared: the view is empty but has its columns.
    let ExecutionResult::Rows {
        columns,
        rows: listed,
        ..
    } = ok(&mut root, &engine, "SELECT * FROM nusadb_prepared_xacts")
    else {
        panic!("no rows");
    };
    assert_eq!(
        columns,
        vec!["transaction", "gid", "prepared", "owner", "database"]
    );
    assert!(listed.is_empty());
    ok(&mut root, &engine, "CREATE ROLE alice LOGIN");
    let mut alice = Session::new(&engine);
    alice.set_current_user("alice");
    ok(&mut alice, &engine, "BEGIN");
    ok(&mut alice, &engine, "PREPARE TRANSACTION 'mine'");
    ok(&mut root, &engine, "BEGIN");
    ok(&mut root, &engine, "PREPARE TRANSACTION 'it''s root''s'");
    assert_eq!(
        rows(
            &mut root,
            &engine,
            "SELECT gid, owner, database FROM nusadb_prepared_xacts ORDER BY gid"
        ),
        vec![
            vec!["it's root's", "nusadb-root", "nusadb"],
            vec!["mine", "alice", "nusadb"],
        ]
    );
    // A role sees its own only, and the view joins and filters like any relation.
    assert_eq!(
        rows(
            &mut alice,
            &engine,
            "SELECT p.gid FROM nusadb_prepared_xacts AS p WHERE p.prepared IS NOT NULL"
        ),
        vec![vec!["mine"]]
    );
    ok(&mut root, &engine, "COMMIT PREPARED 'mine'");
    ok(&mut root, &engine, "COMMIT PREPARED 'it''s root''s'");
    assert!(rows(&mut root, &engine, "SELECT gid FROM nusadb_prepared_xacts").is_empty());
}

/// `(severity, code, message)` of every notice `sql` raised.
fn notices(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> Vec<(Severity, String, String)> {
    let _ = session.take_notices();
    ok(session, engine, sql);
    session
        .take_notices()
        .into_iter()
        .map(|n| (n.severity, n.code, n.message))
        .collect()
}

fn one(severity: Severity, code: &str, message: &str) -> Vec<(Severity, String, String)> {
    vec![(severity, code.to_owned(), message.to_owned())]
}

#[test]
fn if_exists_and_if_not_exists_raise_their_notices() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    assert_eq!(
        notices(&mut s, &engine, "DROP TABLE IF EXISTS missing"),
        one(
            Severity::Notice,
            "00000",
            "table \"missing\" does not exist, skipping"
        )
    );
    assert_eq!(
        notices(&mut s, &engine, "CREATE TABLE IF NOT EXISTS t (n INT)"),
        one(
            Severity::Notice,
            "42P07",
            "relation \"t\" already exists, skipping"
        )
    );
    assert_eq!(
        notices(&mut s, &engine, "ALTER TABLE t DROP COLUMN IF EXISTS ghost"),
        one(
            Severity::Notice,
            "00000",
            "column \"ghost\" of relation \"t\" does not exist, skipping"
        )
    );
    assert_eq!(
        notices(
            &mut s,
            &engine,
            "ALTER TABLE t ADD COLUMN IF NOT EXISTS v TEXT"
        ),
        one(
            Severity::Notice,
            "42701",
            "column \"v\" of relation \"t\" already exists, skipping"
        )
    );
    assert_eq!(
        notices(
            &mut s,
            &engine,
            "ALTER TABLE t DROP CONSTRAINT IF EXISTS nope"
        ),
        one(
            Severity::Notice,
            "00000",
            "constraint \"nope\" of relation \"t\" does not exist, skipping"
        )
    );
    assert_eq!(
        notices(
            &mut s,
            &engine,
            "ALTER TABLE IF EXISTS ghost ADD COLUMN n INT"
        ),
        one(
            Severity::Notice,
            "00000",
            "relation \"ghost\" does not exist, skipping"
        )
    );
    for (sql, what) in [
        ("DROP VIEW IF EXISTS nv", "view \"nv\""),
        ("DROP INDEX IF EXISTS ni", "index \"ni\""),
        ("DROP SEQUENCE IF EXISTS ns", "sequence \"ns\""),
        ("DROP SCHEMA IF EXISTS nsch", "schema \"nsch\""),
        ("DROP TYPE IF EXISTS nt", "type \"nt\""),
        ("DROP FUNCTION IF EXISTS nf", "function nf()"),
        ("DROP ROLE IF EXISTS nr", "role \"nr\""),
    ] {
        assert_eq!(
            notices(&mut s, &engine, sql),
            one(
                Severity::Notice,
                "00000",
                &format!("{what} does not exist, skipping")
            ),
            "{sql}"
        );
    }
    ok(&mut s, &engine, "CREATE SCHEMA app");
    assert_eq!(
        notices(&mut s, &engine, "CREATE SCHEMA IF NOT EXISTS app"),
        one(
            Severity::Notice,
            "42P06",
            "schema \"app\" already exists, skipping"
        )
    );
    // What exists is dropped without a notice, and a plain statement raises none.
    assert!(notices(&mut s, &engine, "DROP SCHEMA IF EXISTS app").is_empty());
    assert!(notices(&mut s, &engine, "SELECT v FROM t").is_empty());
}

#[test]
fn raise_sends_notices_and_raise_exception_aborts() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    let raised = notices(
        &mut s,
        &engine,
        "DO $$ BEGIN \
           RAISE NOTICE 'row % of %', 1, 2; \
           RAISE WARNING 'careful: % %%', NULL; \
           RAISE INFO 'info'; \
           RAISE LOG 'not sent'; \
         END $$",
    );
    assert_eq!(
        raised,
        vec![
            (
                Severity::Notice,
                "00000".to_owned(),
                "row 1 of 2".to_owned()
            ),
            (
                Severity::Warning,
                "01000".to_owned(),
                "careful: <NULL> %".to_owned()
            ),
            (Severity::Info, "00000".to_owned(), "info".to_owned()),
        ]
    );
    let e = run(
        &mut s,
        &engine,
        "DO $$ BEGIN RAISE EXCEPTION 'bad value %', 7; END $$",
    )
    .unwrap_err();
    assert_eq!(e.sqlstate(), "P0001");
    assert!(e.to_string().contains("bad value 7"), "{e}");
    let e = run(&mut s, &engine, "DO $$ BEGIN RAISE NOTICE '% %', 1; END $$").unwrap_err();
    assert_eq!(e.sqlstate(), "42601");
    // The old expression form still aborts.
    let e = run(&mut s, &engine, "DO $$ BEGIN RAISE 'plain'; END $$").unwrap_err();
    assert_eq!(e.sqlstate(), "P0001");
}

#[test]
fn ddl_on_a_table_a_prepared_transaction_wrote_is_refused_with_55006() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(&mut s, &engine, "CREATE SCHEMA app");
    ok(&mut s, &engine, "CREATE TABLE app.u (n INT)");
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO t VALUES (10, 'p1')");
    ok(&mut s, &engine, "INSERT INTO app.u VALUES (1)");
    ok(&mut s, &engine, "PREPARE TRANSACTION 'held'");
    for sql in [
        "DROP TABLE t",
        "TRUNCATE t",
        "CREATE INDEX t_v ON t (v)",
        "CREATE UNIQUE INDEX t_vu ON t (v)",
        "ALTER TABLE t ADD COLUMN w INT",
        "ALTER TABLE t ADD CONSTRAINT t_v_unique UNIQUE (v)",
        "DROP SCHEMA app CASCADE",
    ] {
        assert_eq!(code(&mut s, &engine, sql), "55006", "{sql}");
    }
    ok(&mut s, &engine, "COMMIT PREPARED 'held'");
    // Now the index is built over the committed row and finds it.
    ok(&mut s, &engine, "CREATE INDEX t_v ON t (v)");
    assert_eq!(
        rows(&mut s, &engine, "SELECT id FROM t WHERE v = 'p1'"),
        vec![vec!["10".to_owned()]]
    );
    ok(&mut s, &engine, "TRUNCATE t");
    ok(&mut s, &engine, "DROP SCHEMA app CASCADE");
}

#[test]
fn a_raise_in_a_long_loop_keeps_a_bounded_number_of_notices() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    let raised = notices(
        &mut s,
        &engine,
        "DO $$ BEGIN FOR i IN 1 TO 1500 LOOP RAISE NOTICE 'n %', i; END LOOP; END $$",
    );
    assert_eq!(raised.len(), nusadb_sql::notice::MAX_NOTICES + 1);
    assert_eq!(raised[0].2, "n 1");
    let last = raised.last().unwrap();
    assert_eq!((last.0, last.1.as_str()), (Severity::Warning, "01000"));
    assert!(last.2.starts_with("500 more notices"), "{}", last.2);
}

/// Autocommit DDL that meets a running writer is retried like any autocommit conflict: a writer
/// that ends inside the retry budget lets the DDL through instead of failing it.
#[test]
fn autocommit_ddl_retries_past_a_writer_that_ends() {
    let engine = fixture();
    let began = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut w = Session::new(&engine);
            ok(&mut w, &engine, "BEGIN");
            ok(&mut w, &engine, "INSERT INTO t VALUES (5, 'w')");
            began.wait();
            std::thread::sleep(std::time::Duration::from_millis(40));
            ok(&mut w, &engine, "COMMIT");
        });
        began.wait();
        let mut s = Session::new(&engine);
        ok(&mut s, &engine, "CREATE INDEX t_v ON t (v)");
    });
    let mut s = Session::new(&engine);
    assert_eq!(
        rows(&mut s, &engine, "SELECT id FROM t WHERE v = 'w'"),
        vec![vec!["5".to_owned()]]
    );
}

/// Constraint-adding DDL validates existing rows, which a prepared transaction's unfinished rows
/// are not among: refused with 55006, on a partitioned parent too, and on the parent of a new
/// foreign key whose delete is prepared.
#[test]
fn constraints_over_a_prepared_transactions_rows_are_refused() {
    let engine = fixture();
    let mut s = Session::new(&engine);
    ok(
        &mut s,
        &engine,
        "CREATE TABLE pt (id INT) PARTITION BY RANGE (id)",
    );
    ok(
        &mut s,
        &engine,
        "CREATE TABLE pt1 PARTITION OF pt FOR VALUES FROM (0) TO (100)",
    );
    ok(&mut s, &engine, "CREATE TABLE par (id INT PRIMARY KEY)");
    ok(&mut s, &engine, "CREATE TABLE ch (id INT, pid INT)");
    ok(&mut s, &engine, "INSERT INTO par VALUES (1)");
    ok(&mut s, &engine, "INSERT INTO ch VALUES (1, 1)");
    ok(&mut s, &engine, "BEGIN");
    ok(&mut s, &engine, "INSERT INTO t VALUES (3, 'x')");
    ok(&mut s, &engine, "INSERT INTO pt VALUES (50)");
    ok(&mut s, &engine, "DELETE FROM par WHERE id = 1");
    ok(&mut s, &engine, "PREPARE TRANSACTION 'held'");
    for sql in [
        "ALTER TABLE t ADD CONSTRAINT ck CHECK (id < 3)",
        "ALTER TABLE pt ADD CONSTRAINT pck CHECK (id < 10)",
        "ALTER TABLE ch ADD CONSTRAINT chfk FOREIGN KEY (pid) REFERENCES par (id)",
    ] {
        assert_eq!(code(&mut s, &engine, sql), "55006", "{sql}");
    }
    // The refusal names the table and the prepared transaction.
    let e = run(&mut s, &engine, "DROP TABLE t")
        .unwrap_err()
        .to_string();
    assert!(e.contains("table \"t\"") && e.contains("\"held\""), "{e}");
    ok(&mut s, &engine, "ROLLBACK PREPARED 'held'");
    ok(
        &mut s,
        &engine,
        "ALTER TABLE t ADD CONSTRAINT ck CHECK (id < 3)",
    );
    ok(
        &mut s,
        &engine,
        "ALTER TABLE ch ADD CONSTRAINT chfk FOREIGN KEY (pid) REFERENCES par (id)",
    );
}
