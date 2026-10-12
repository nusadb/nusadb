//! Column DDL and the objects that depend on a column: `DROP COLUMN` (RESTRICT and CASCADE),
//! `RENAME COLUMN`, `ALTER COLUMN ... TYPE` and the owned sequences of identity and serial columns,
//! including rollback and a crash-reopen of the durable engine afterwards.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema, TxnId};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

/// The analyzer's view of the engine, resolved under the session's open transaction when there is
/// one, so DDL earlier in the same `BEGIN` block is visible.
struct TxnCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: Option<TxnId>,
}

impl TxnCatalog<'_> {
    fn with_txn<T>(
        &self,
        f: impl FnOnce(TxnId) -> Result<T, nusadb_sql::Error>,
    ) -> Result<T, nusadb_sql::Error> {
        if let Some(txn) = self.txn {
            return f(txn);
        }
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let out = f(txn);
        let _ = self.engine.commit(txn);
        out
    }
}

impl Catalog for TxnCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.lookup_table_in(nusadb_core::PUBLIC_SCHEMA, name)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.with_txn(|txn| {
            self.engine
                .lookup_table_as_of_in(txn, schema, name)
                .map_err(Into::into)
        })
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes(self.engine, txn, name))
    }

    fn list_indexes_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes_in(self.engine, txn, schema, name))
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_view_definition(self.engine, txn, name))
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_view_columns(self.engine, txn, name))
    }
}

/// One client connection: statements run through a [`Session`], so `BEGIN`/`ROLLBACK` behave as
/// they do on the wire.
struct Conn<'a> {
    engine: &'a BtreeEngine,
    session: Session<'a>,
}

impl<'a> Conn<'a> {
    fn new(engine: &'a BtreeEngine) -> Self {
        Self {
            engine,
            session: Session::new(engine),
        }
    }

    fn try_run(&mut self, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
        let catalog = TxnCatalog {
            engine: self.engine,
            txn: self.session.current_txn(),
        };
        let logical = analyze(parse(sql)?, &catalog)?;
        self.session.execute(plan(logical))
    }

    fn run(&mut self, sql: &str) -> ExecutionResult {
        self.try_run(sql)
            .unwrap_or_else(|e| panic!("`{sql}` failed: {e} ({})", e.sqlstate()))
    }

    /// Run `sql`, which must fail with `sqlstate`; returns the message.
    fn fails(&mut self, sql: &str, sqlstate: &str) -> String {
        match self.try_run(sql) {
            Ok(r) => panic!("`{sql}` should fail with {sqlstate}, got {r:?}"),
            Err(e) => {
                assert_eq!(e.sqlstate(), sqlstate, "`{sql}`: {e}");
                e.to_string()
            },
        }
    }

    fn rows(&mut self, sql: &str) -> Vec<Vec<Value>> {
        match self.run(sql) {
            ExecutionResult::Rows { rows, .. } => rows,
            other => panic!("`{sql}`: expected rows, got {other:?}"),
        }
    }

    /// The single integer `sql` returns.
    fn int(&mut self, sql: &str) -> i64 {
        match self.rows(sql).as_slice() {
            [row] => match row.as_slice() {
                [Value::Int(n)] => *n,
                other => panic!("`{sql}`: expected one integer, got {other:?}"),
            },
            other => panic!("`{sql}`: expected one row, got {other:?}"),
        }
    }
}

/// Run `sql` on a fresh connection to `engine`.
fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    Conn::new(engine).run(sql)
}

/// The constraint names of `table`, the type-range checks left out, sorted.
fn constraints(engine: &BtreeEngine, table: &str) -> Vec<String> {
    let id = engine.lookup_table(table).unwrap().unwrap().id;
    let mut names: Vec<String> = engine
        .list_constraints(id)
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .filter(|n| !n.starts_with(nusadb_sql::SYNTHETIC_TYPE_CHECK_PREFIX))
        .collect();
    names.sort();
    names
}

/// The index names of `table`, sorted.
fn indexes(engine: &BtreeEngine, table: &str) -> Vec<String> {
    let id = engine.lookup_table(table).unwrap().unwrap().id;
    let mut names: Vec<String> = engine
        .list_indexes(id)
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect();
    names.sort();
    names
}

/// The table and its neighbours the drop and rename tests share: a parent `p` with a key and a
/// second unique column, and a table `c1` whose column `c` is named by a foreign key, a
/// single-column and a multi-column index, an expression index, a partial index, a multi-column
/// `UNIQUE`, a `CHECK` on it alone and one together with another column.
fn seed(engine: &BtreeEngine) {
    let mut c = Conn::new(engine);
    for sql in [
        "CREATE TABLE p (k INT PRIMARY KEY, u INT UNIQUE)",
        "INSERT INTO p VALUES (1, 10), (2, 20)",
        "CREATE TABLE c1 (id INT PRIMARY KEY, c INT REFERENCES p(k), d INT, e TEXT, \
         CHECK (c > 0), CHECK (c < d), CHECK (d < 1000), UNIQUE (c, d))",
        "CREATE INDEX c1_c ON c1 (c)",
        "CREATE INDEX c1_cd ON c1 (c, d)",
        "CREATE INDEX c1_expr ON c1 ((c + d))",
        "CREATE INDEX c1_part ON c1 (d) WHERE c > 0",
        "CREATE INDEX c1_d ON c1 (d)",
        "INSERT INTO c1 VALUES (1, 1, 5, 'a'), (2, 2, 6, 'b')",
    ] {
        c.run(sql);
    }
}

/// `DROP COLUMN` takes every index and constraint of the table that involves the column, alone or
/// with other columns, and leaves the rest working.
#[test]
fn drop_column_takes_the_tables_own_dependents() {
    let engine = BtreeEngine::new();
    seed(&engine);
    let mut c = Conn::new(&engine);
    c.run("ALTER TABLE c1 DROP COLUMN c");
    assert_eq!(constraints(&engine, "c1"), ["c1_check3", "c1_pkey"]);
    assert_eq!(indexes(&engine, "c1"), ["c1_d", "c1_pkey"]);
    // The table is fully usable: the reported INSERT, an UPDATE, a read through the index.
    c.run("INSERT INTO c1 VALUES (3, 7, 'c')");
    c.run("UPDATE c1 SET d = 8 WHERE id = 3");
    assert_eq!(c.int("SELECT d FROM c1 WHERE d = 8"), 8);
    assert_eq!(c.int("SELECT count(*) FROM c1"), 3);
    // What did not involve the column still bites.
    c.fails("INSERT INTO c1 VALUES (3, 1, 'x')", "23505");
    c.fails("INSERT INTO c1 VALUES (4, 5000, 'x')", "23514");
    // The parent no longer has a child holding it.
    c.run("DELETE FROM p WHERE k = 1");
}

/// Without `CASCADE` a column another table's foreign key or a view reads is refused with
/// `2BP01`; with it, those go too and the rest keeps working.
#[test]
fn drop_column_restrict_refuses_and_cascade_drops_outside_dependents() {
    let engine = BtreeEngine::new();
    seed(&engine);
    let mut c = Conn::new(&engine);
    c.run("CREATE VIEW pv AS SELECT k, u FROM p");
    c.run("CREATE VIEW pv2 AS SELECT u FROM pv");
    c.run("CREATE VIEW keep AS SELECT k FROM p");
    let msg = c.fails("ALTER TABLE p DROP COLUMN k", "2BP01");
    assert!(msg.contains("c1_fkey"), "{msg}");
    c.fails("ALTER TABLE p DROP COLUMN k RESTRICT", "2BP01");
    let msg = c.fails("ALTER TABLE p DROP COLUMN u", "2BP01");
    assert!(msg.contains("pv"), "{msg}");
    // Refused, so everything is still in place and enforced.
    c.fails("INSERT INTO c1 VALUES (9, 99, 100, 'z')", "23503");
    assert_eq!(c.int("SELECT count(*) FROM pv2"), 2);

    c.run("ALTER TABLE p DROP COLUMN u CASCADE");
    c.fails("SELECT * FROM pv", "42P01");
    c.fails("SELECT * FROM pv2", "42P01");
    assert_eq!(c.int("SELECT count(*) FROM keep"), 2);

    c.run("ALTER TABLE p ADD COLUMN v INT");
    c.run("ALTER TABLE p DROP COLUMN k CASCADE");
    assert_eq!(constraints(&engine, "p"), Vec::<String>::new());
    assert!(!constraints(&engine, "c1").contains(&"c1_fkey".to_owned()));
    // The child's own column stays; only the key that referenced the dropped column went.
    c.run("INSERT INTO c1 VALUES (9, 99, 100, 'z')");
    c.run("INSERT INTO p VALUES (5)");
    assert_eq!(c.int("SELECT count(*) FROM p"), 3);
}

/// A generated column computed from the dropped column cannot be computed any more and goes
/// with it.
#[test]
fn drop_column_takes_generated_columns_reading_it() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE g (id INT, a INT, b INT GENERATED ALWAYS AS (a * 2) STORED)");
    c.run("INSERT INTO g (id, a) VALUES (1, 4)");
    c.run("ALTER TABLE g DROP COLUMN a");
    let table = engine.lookup_table("g").unwrap().unwrap();
    let names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id"]);
    c.run("INSERT INTO g VALUES (2)");
    assert_eq!(c.int("SELECT count(*) FROM g"), 2);
    // Dropping the last other column would leave nothing behind.
    c.run("CREATE TABLE g2 (a INT, b INT GENERATED ALWAYS AS (a + 1) STORED)");
    c.fails("ALTER TABLE g2 DROP COLUMN a", "42P16");
}

/// A rolled-back `DROP COLUMN ... CASCADE` restores the column, its data, and every dependent
/// it took, inside the table and outside it.
#[test]
fn drop_column_rollback_restores_everything() {
    let engine = BtreeEngine::new();
    seed(&engine);
    let mut c = Conn::new(&engine);
    c.run("CREATE VIEW pv AS SELECT k FROM p");
    let before_c1 = (constraints(&engine, "c1"), indexes(&engine, "c1"));
    let before_p = (constraints(&engine, "p"), indexes(&engine, "p"));
    c.run("BEGIN");
    c.run("ALTER TABLE c1 DROP COLUMN c");
    c.run("ALTER TABLE p DROP COLUMN k CASCADE");
    c.run("ROLLBACK");
    assert_eq!(
        (constraints(&engine, "c1"), indexes(&engine, "c1")),
        before_c1
    );
    assert_eq!((constraints(&engine, "p"), indexes(&engine, "p")), before_p);
    assert_eq!(c.int("SELECT count(*) FROM pv"), 2);
    assert_eq!(c.int("SELECT c FROM c1 WHERE id = 2"), 2);
    c.fails("INSERT INTO c1 VALUES (3, 99, 100, 'x')", "23503");
    c.fails("INSERT INTO c1 VALUES (3, 1, 5, 'x')", "23505");
    c.fails("INSERT INTO c1 VALUES (3, 2, 1, 'x')", "23514");
    c.fails("DELETE FROM p WHERE k = 1", "23503");
    c.run("INSERT INTO c1 VALUES (3, 2, 7, 'x')");
    assert_eq!(c.int("SELECT id FROM c1 WHERE c = 2 AND d = 7"), 3);
}

/// `RENAME COLUMN` carries every dependent of the table along: primary key, unique, check,
/// foreign keys on both sides, plain, multi-column, expression and partial indexes, a generated
/// expression. Each keeps enforcing or serving under the new name.
#[test]
fn rename_column_carries_dependents() {
    let engine = BtreeEngine::new();
    seed(&engine);
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE gen (a INT, b INT GENERATED ALWAYS AS (a * 3) STORED)");
    c.run("INSERT INTO gen (a) VALUES (2)");
    let before_c1 = (constraints(&engine, "c1"), indexes(&engine, "c1"));
    let before_p = (constraints(&engine, "p"), indexes(&engine, "p"));
    c.run("ALTER TABLE c1 RENAME COLUMN c TO c2");
    c.run("ALTER TABLE p RENAME COLUMN k TO k2");
    c.run("ALTER TABLE gen RENAME COLUMN a TO a2");
    assert_eq!(
        (constraints(&engine, "c1"), indexes(&engine, "c1")),
        before_c1
    );
    assert_eq!((constraints(&engine, "p"), indexes(&engine, "p")), before_p);
    let fk = engine
        .list_foreign_keys(engine.lookup_table("c1").unwrap().unwrap().id)
        .unwrap();
    assert_eq!(fk[0].child_columns, ["c2"]);
    assert_eq!(fk[0].parent_columns, ["k2"]);
    let id = engine.lookup_table("c1").unwrap().unwrap().id;
    let expr = engine
        .list_indexes(id)
        .unwrap()
        .into_iter()
        .find(|d| d.name == "c1_expr")
        .unwrap();
    assert_eq!(expr.key_exprs, ["(\"c2\" + d)"]);
    // Every constraint still bites under the new names.
    c.fails("INSERT INTO c1 (id, c2, d) VALUES (3, 99, 100)", "23503");
    c.fails("INSERT INTO c1 (id, c2, d) VALUES (3, 1, 5)", "23505");
    c.fails("INSERT INTO c1 (id, c2, d) VALUES (3, 2, 1)", "23514");
    c.fails("INSERT INTO p VALUES (1, 30)", "23505");
    c.fails("DELETE FROM p WHERE k2 = 1", "23503");
    c.run("INSERT INTO c1 (id, c2, d) VALUES (3, 2, 7)");
    c.run("UPDATE c1 SET c2 = 1 WHERE id = 3");
    // The recreated indexes hold the existing rows: reads through them find the old rows too.
    assert_eq!(c.int("SELECT id FROM c1 WHERE c2 = 2"), 2);
    assert_eq!(c.int("SELECT count(*) FROM c1 WHERE c2 + d = 8"), 2);
    assert_eq!(c.int("SELECT b FROM gen"), 6);
    c.run("INSERT INTO gen (a2) VALUES (5)");
    assert_eq!(c.int("SELECT b FROM gen WHERE a2 = 5"), 15);
    // The old name is gone.
    c.fails("SELECT c FROM c1", "42703");
}

/// A rename still refuses while a view, policy or trigger keeps SQL text naming the column (that
/// text is not rewritten), and a rolled-back rename restores the old names everywhere.
#[test]
fn rename_column_refuses_text_dependents_and_rolls_back() {
    let engine = BtreeEngine::new();
    seed(&engine);
    let mut c = Conn::new(&engine);
    // A function in the catalog whose body names other columns does not stand in the way.
    c.run("CREATE FUNCTION fd() RETURNS INT LANGUAGE nusascript AS $$ BEGIN UPDATE c1 SET d = 1; RETURN 1; END $$");
    c.run("ALTER TABLE c1 RENAME COLUMN id TO id2");
    c.run("ALTER TABLE c1 RENAME COLUMN id2 TO id");
    let msg = c.fails("ALTER TABLE c1 RENAME COLUMN d TO d2", "2BP01");
    assert!(msg.contains("fd"), "{msg}");
    c.run("CREATE VIEW cv AS SELECT e FROM c1");
    c.fails("ALTER TABLE c1 RENAME COLUMN e TO e2", "2BP01");
    let before = (constraints(&engine, "c1"), indexes(&engine, "c1"));
    c.run("BEGIN");
    c.run("ALTER TABLE c1 RENAME COLUMN c TO c2");
    c.run("ALTER TABLE p RENAME COLUMN k TO k2");
    c.run("ROLLBACK");
    assert_eq!((constraints(&engine, "c1"), indexes(&engine, "c1")), before);
    c.fails("INSERT INTO c1 VALUES (3, 99, 100, 'x')", "23503");
    c.run("INSERT INTO c1 VALUES (3, 2, 7, 'x')");
    assert_eq!(c.int("SELECT count(*) FROM c1 WHERE c = 2"), 2);
}

/// Changing a column's type replaces the type-range check: widening lifts the old bound,
/// narrowing validates the existing rows and enforces the new bound, and a `VARCHAR(n)` length
/// moves the same way.
#[test]
fn type_change_replaces_the_range_check() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE tg (v SMALLINT, s VARCHAR(3))");
    c.run("INSERT INTO tg VALUES (1, 'abc')");
    c.fails("INSERT INTO tg VALUES (40000, 'a')", "23514");
    c.run("ALTER TABLE tg ALTER COLUMN v TYPE integer");
    c.run("INSERT INTO tg VALUES (40000, 'a')");
    c.fails("INSERT INTO tg VALUES (3000000000, 'a')", "23514");
    c.run("ALTER TABLE tg ALTER COLUMN v TYPE bigint");
    c.run("INSERT INTO tg VALUES (3000000000, 'a')");
    // Narrowing checks the rows already there.
    c.fails("ALTER TABLE tg ALTER COLUMN v TYPE integer", "22003");
    c.run("DELETE FROM tg WHERE v > 32767");
    c.run("ALTER TABLE tg ALTER COLUMN v TYPE smallint");
    c.fails("INSERT INTO tg VALUES (40000, 'a')", "23514");
    // Lengths: widen, then narrow below a stored value.
    c.fails("INSERT INTO tg VALUES (1, 'abcdef')", "22001");
    c.run("ALTER TABLE tg ALTER COLUMN s TYPE varchar(10)");
    c.run("INSERT INTO tg VALUES (2, 'abcdef')");
    c.fails("ALTER TABLE tg ALTER COLUMN s TYPE varchar(4)", "22001");
    c.run("ALTER TABLE tg ALTER COLUMN s TYPE text");
    c.run("INSERT INTO tg VALUES (3, 'a much longer value than before')");
    // A rolled-back change keeps the old bound.
    c.run("BEGIN");
    c.run("ALTER TABLE tg ALTER COLUMN v TYPE bigint");
    c.run("ROLLBACK");
    c.fails("INSERT INTO tg VALUES (40000, 'a')", "23514");
}

/// The sequence of an identity or serial column goes with the column, its default or its table,
/// so the same table can be created again, and a rollback of any of those brings it back.
#[test]
fn owned_sequences_follow_their_column() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    for ty in [
        "INT GENERATED BY DEFAULT AS IDENTITY",
        "INT GENERATED ALWAYS AS IDENTITY",
        "SERIAL",
    ] {
        let create = format!("CREATE TABLE ti (id {ty}, v INT)");
        c.run(&create);
        c.run("ALTER TABLE ti DROP COLUMN id");
        assert!(
            engine.lookup_sequence("ti_id_seq").unwrap().is_none(),
            "{ty}"
        );
        c.run("DROP TABLE ti");
        c.run(&create);
        c.run("ALTER TABLE ti ALTER COLUMN id DROP DEFAULT");
        assert!(
            engine.lookup_sequence("ti_id_seq").unwrap().is_none(),
            "{ty}"
        );
        c.run("DROP TABLE ti");
        c.run(&create);
        c.run("DROP TABLE ti");
    }
}

/// A rolled-back `DROP TABLE`, `DROP COLUMN` or `DROP SCHEMA ... CASCADE` brings back the
/// sequence of the serial and identity columns it took, and the counter carries on where it was.
#[test]
fn rolled_back_drops_restore_owned_sequences() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE seqd (id SERIAL, v INT)");
    c.run("CREATE TABLE idd (id INT GENERATED ALWAYS AS IDENTITY, v INT)");
    c.run("CREATE SCHEMA app");
    c.run("CREATE TABLE app.sch (id SERIAL, v INT)");
    for table in ["seqd", "idd", "app.sch"] {
        c.run(&format!("INSERT INTO {table} (v) VALUES (1)"));
    }
    for (drop, table) in [
        ("DROP TABLE seqd", "seqd"),
        ("ALTER TABLE seqd DROP COLUMN id", "seqd"),
        ("ALTER TABLE seqd ALTER COLUMN id DROP DEFAULT", "seqd"),
        ("DROP TABLE idd", "idd"),
        ("ALTER TABLE idd DROP COLUMN id", "idd"),
        ("DROP SCHEMA app CASCADE", "app.sch"),
    ] {
        c.run("BEGIN");
        c.run(drop);
        c.run("ROLLBACK");
        c.run(&format!("INSERT INTO {table} (v) VALUES (2)"));
    }
    c.run("INSERT INTO seqd DEFAULT VALUES");
    assert_eq!(c.int("SELECT max(id) FROM seqd"), 5);
    assert_eq!(c.int("SELECT max(id) FROM idd"), 3);
    assert_eq!(c.int("SELECT max(id) FROM app.sch"), 2);
    // Committed, they take the sequence: the schema and its table can be made again.
    c.run("DROP SCHEMA app CASCADE");
    c.run("CREATE SCHEMA app");
    c.run("CREATE TABLE app.sch (id SERIAL, v INT)");
}

/// `DROP CONSTRAINT` of a constraint that is not there is `42704`, a no-op with `IF EXISTS`, and
/// a sequence name already taken is `42P07`, not an internal error.
#[test]
fn missing_and_duplicate_objects_report_their_class() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE e2 (a INT)");
    let msg = c.fails("ALTER TABLE e2 DROP CONSTRAINT nosuch", "42704");
    assert!(msg.contains("nosuch"), "{msg}");
    c.run("ALTER TABLE e2 DROP CONSTRAINT IF EXISTS nosuch");
    c.run("CREATE SEQUENCE s1");
    c.fails("CREATE SEQUENCE s1", "42P07");
    c.run("CREATE SEQUENCE IF NOT EXISTS s1");
    c.run("CREATE SEQUENCE e3_id_seq");
    c.fails("CREATE TABLE e3 (id SERIAL)", "42P07");
}

/// After `DROP COLUMN` and `RENAME COLUMN`, the table survives a crash-reopen (WAL replay) and a
/// checkpoint-reopen (image) with its rows, its remaining and carried constraints and indexes,
/// and its sequences, and stays writable. Rolled-back drops and renames leave nothing behind
/// after either.
#[test]
fn column_changes_survive_restart_and_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let expected_c1;
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed(&engine);
        let mut c = Conn::new(&engine);
        c.run("CREATE TABLE seqd (id SERIAL, v INT)");
        c.run("INSERT INTO seqd (v) VALUES (1)");
        c.run("CREATE TABLE tg (v SMALLINT)");
        c.run("ALTER TABLE tg ALTER COLUMN v TYPE integer");
        c.run("ALTER TABLE c1 DROP COLUMN e");
        c.run("ALTER TABLE c1 RENAME COLUMN c TO c2");
        c.run("ALTER TABLE p RENAME COLUMN k TO k2");
        c.run("ALTER TABLE p DROP COLUMN u");
        c.run("BEGIN");
        c.run("ALTER TABLE c1 DROP COLUMN c2");
        c.run("DROP TABLE seqd");
        c.run("ROLLBACK");
        c.run("BEGIN");
        c.run("ALTER TABLE c1 RENAME COLUMN d TO dd");
        c.run("ROLLBACK");
        expected_c1 = (constraints(&engine, "c1"), indexes(&engine, "c1"));
    } // crash: no shutdown, no checkpoint.
    let check = |engine: &BtreeEngine, round: i64| {
        let mut c = Conn::new(engine);
        assert_eq!(
            (constraints(engine, "c1"), indexes(engine, "c1")),
            expected_c1
        );
        let names: Vec<String> = engine
            .lookup_table("c1")
            .unwrap()
            .unwrap()
            .columns
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["id", "c2", "d"]);
        c.fails("INSERT INTO c1 VALUES (100, 99, 100)", "23503");
        c.fails("INSERT INTO c1 VALUES (100, 1, 5)", "23505");
        c.fails("INSERT INTO c1 VALUES (100, 2, 1)", "23514");
        c.run(&format!(
            "INSERT INTO c1 VALUES ({}, 2, {})",
            10 + round,
            100 + round
        ));
        assert_eq!(
            c.int(&format!(
                "SELECT id FROM c1 WHERE c2 = 2 AND d = {}",
                100 + round
            )),
            10 + round
        );
        c.run(&format!("INSERT INTO tg VALUES ({})", 40_000 + round));
        c.run("INSERT INTO seqd (v) VALUES (2)");
        assert_eq!(c.int("SELECT max(id) FROM seqd"), 2 + round);
    };
    {
        let engine = BtreeEngine::open(&path).unwrap();
        check(&engine, 0);
        run(&engine, "CHECKPOINT");
    }
    let engine = BtreeEngine::open(&path).unwrap();
    check(&engine, 1);
}

/// What reads a column without naming it in the select list still depends on it: a `*` (with or
/// without a column list on the view), a `NATURAL` join or `USING` on it, a policy or trigger
/// attached to another table whose text reads it, and a function or procedure body.
#[test]
fn drop_column_finds_indirect_readers() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    for sql in [
        "CREATE TABLE t (id INT, b INT, keep INT)",
        "CREATE TABLE w (b INT, z INT)",
        "CREATE TABLE u (x INT)",
        "INSERT INTO t VALUES (1, 1, 1)",
        "INSERT INTO w VALUES (1, 1), (2, 2)",
    ] {
        c.run(sql);
    }
    let refused = |c: &mut Conn<'_>, setup: &str, teardown: &str, what: &str| {
        c.run(setup);
        let msg = c.fails("ALTER TABLE t DROP COLUMN b", "2BP01");
        assert!(msg.contains(what), "{setup}: {msg}");
        c.run(teardown);
    };
    refused(
        &mut c,
        "CREATE VIEW vstar (p, q, r) AS SELECT * FROM t",
        "DROP VIEW vstar",
        "vstar",
    );
    refused(
        &mut c,
        "CREATE VIEW vs AS SELECT * FROM t",
        "DROP VIEW vs",
        "vs",
    );
    refused(
        &mut c,
        "CREATE VIEW vq AS SELECT t.* FROM t",
        "DROP VIEW vq",
        "vq",
    );
    refused(
        &mut c,
        "CREATE VIEW vn AS SELECT id FROM t NATURAL JOIN w",
        "DROP VIEW vn",
        "vn",
    );
    refused(
        &mut c,
        "CREATE VIEW vu AS SELECT id FROM t JOIN w USING (b)",
        "DROP VIEW vu",
        "vu",
    );
    refused(
        &mut c,
        "CREATE POLICY pu ON u USING (EXISTS (SELECT 1 FROM t WHERE t.b = u.x))",
        "DROP POLICY pu ON u",
        "pu",
    );
    refused(
        &mut c,
        "CREATE TRIGGER tu AFTER INSERT ON u FOR EACH ROW INSERT INTO t (id, b) VALUES (9, 9)",
        "DROP TRIGGER tu ON u",
        "tu",
    );
    // A count(*) or a view naming other columns only does not hold the column.
    c.run("CREATE VIEW vc AS SELECT count(*) AS n FROM t");
    c.run("CREATE VIEW vk AS SELECT keep FROM t");
    // A routine body is the user's code: refused even with CASCADE.
    c.run("CREATE PROCEDURE pb() AS $$ UPDATE t SET b = 0 $$");
    let msg = c.fails("ALTER TABLE t DROP COLUMN b CASCADE", "2BP01");
    assert!(msg.contains("pb"), "{msg}");
    c.run("DROP PROCEDURE pb");
    // CASCADE takes a policy and a trigger attached to another table along.
    c.run("CREATE POLICY pu ON u USING (EXISTS (SELECT 1 FROM t WHERE t.b = u.x))");
    c.run("CREATE TRIGGER tu AFTER INSERT ON u FOR EACH ROW INSERT INTO t (id, b) VALUES (9, 9)");
    c.run("CREATE VIEW vs AS SELECT * FROM t");
    c.run("ALTER TABLE t DROP COLUMN b CASCADE");
    c.fails("DROP POLICY pu ON u", "42704");
    c.fails("DROP TRIGGER tu ON u", "42704");
    c.fails("SELECT * FROM vs", "42P01");
    c.run("INSERT INTO u VALUES (1)");
    assert_eq!(c.int("SELECT n FROM vc"), 1);
    assert_eq!(c.int("SELECT keep FROM vk"), 1);
}

/// A sequence another column's default names through `nextval('...')` outlives the serial column
/// it was made for.
#[test]
fn a_sequence_named_by_another_default_is_kept() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE src (id SERIAL, v INT)");
    c.run("CREATE TABLE other (n INT DEFAULT nextval('src_id_seq'), v INT)");
    c.run("ALTER TABLE src DROP COLUMN id");
    assert!(engine.lookup_sequence("src_id_seq").unwrap().is_some());
    assert_eq!(c.int("SELECT nextval('src_id_seq')"), 1);
    // Once nothing names it, it goes.
    c.run("CREATE TABLE src2 (id SERIAL, v INT)");
    c.run("ALTER TABLE src2 DROP COLUMN id");
    assert!(engine.lookup_sequence("src2_id_seq").unwrap().is_none());
}

/// A sequence name whose drop another session has not committed is not free: creating it is a
/// retryable conflict, and the dropper's rollback leaves the sequence where it was.
#[test]
fn a_pending_sequence_drop_holds_the_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        let mut a = Conn::new(&engine);
        let mut b = Conn::new(&engine);
        a.run("CREATE SEQUENCE rs");
        assert_eq!(a.int("SELECT nextval('rs')"), 1);
        a.run("BEGIN");
        a.run("DROP SEQUENCE rs");
        b.fails("CREATE SEQUENCE rs START 100", "40001");
        a.run("ROLLBACK");
        assert_eq!(b.int("SELECT nextval('rs')"), 2);
    } // crash: no shutdown.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(Conn::new(&engine).int("SELECT nextval('rs')"), 3);
}

/// A vector index is recorded under its table's bare name, so a column change on one of two
/// same-named tables in different schemas refuses rather than move the other's index.
#[test]
fn an_ambiguous_vector_index_refuses_the_drop() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE vt (id INT, emb VECTOR(3))");
    c.run("CREATE INDEX vt_h ON vt USING hnsw (emb vector_l2_ops)");
    c.run("CREATE SCHEMA app");
    c.run("CREATE TABLE app.vt (id INT, other INT, more INT)");
    c.fails("ALTER TABLE app.vt DROP COLUMN id", "0A000");
    // A drop that cannot move it (a later column) is fine.
    c.run("ALTER TABLE app.vt DROP COLUMN more");
}
