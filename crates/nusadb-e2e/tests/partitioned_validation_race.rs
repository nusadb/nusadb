//! A writer that commits while a DDL statement validates, rewrites or indexes a table must never
//! leave a row the statement's result forbids or misses. The statement takes every table it reads
//! exclusively before it reads any, and once it holds them it reads the latest committed rows, not
//! the snapshot its statement began with: a writer that committed between that snapshot and the
//! lock no longer conflicts with the lock, so only a fresh read can see its row.
//!
//! The interleavings are forced, not raced: the engine is wrapped so that the writer's
//! transaction commits underneath the statement at a chosen point, inside its lock call (the
//! window between the snapshot and the lock) or as it opens its scan of a table.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "integration test outside a #[cfg(test)] module; the delegating wrapper is long and \
              one forced interleaving is described by its parts"
)]

use std::ops::Bound;
use std::sync::Mutex;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{
    AlterOp, IndexDef, IsolationLevel, LockedRow, RowLockMode, ScanDirection, SequenceChange,
    SequenceDef, TableDef, TableLockMode, TableStats, Tid, TupleScan,
};
use nusadb_core::{
    Constraint, ForeignKeyDef, IndexId, PreparedTransaction, Result, SchemaId, SequenceId,
    StorageEngine, TableId, TableSchema, TxnId,
};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

/// Where the armed writer commits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Point {
    /// As a scan of the table opens.
    Scan,
    /// Inside a `lock_table` call on the table, before the lock is taken.
    Lock,
    /// Inside a `create_index` call on the table, before the engine takes its lock.
    CreateIndex,
}

/// A [`BtreeEngine`] that commits the armed writer when the statement reaches the armed point.
struct Interleave {
    inner: BtreeEngine,
    armed: Mutex<Option<(Point, TableId, TxnId)>>,
}

impl Interleave {
    fn new() -> Self {
        Self {
            inner: BtreeEngine::new(),
            armed: Mutex::new(None),
        }
    }

    fn fire(&self, point: Point, table: TableId) {
        let writer = self
            .armed
            .lock()
            .unwrap()
            .take_if(|(at, target, _)| *at == point && *target == table)
            .map(|(_, _, writer)| writer);
        if let Some(writer) = writer {
            self.inner.commit(writer).unwrap();
        }
    }
}

impl StorageEngine for Interleave {
    fn begin(&self, level: IsolationLevel) -> Result<TxnId> {
        self.inner.begin(level)
    }

    fn admission_paused(&self) -> bool {
        self.inner.admission_paused()
    }

    fn begin_statement(&self, txn: TxnId) -> Result<()> {
        self.inner.begin_statement(txn)
    }

    fn txn_isolation(&self, txn: TxnId) -> Option<IsolationLevel> {
        self.inner.txn_isolation(txn)
    }

    fn commit(&self, txn: TxnId) -> Result<()> {
        self.inner.commit(txn)
    }

    fn rollback(&self, txn: TxnId) -> Result<()> {
        self.inner.rollback(txn)
    }

    fn savepoint(&self, txn: TxnId, name: &str) -> Result<()> {
        self.inner.savepoint(txn, name)
    }

    fn rollback_to(&self, txn: TxnId, name: &str) -> Result<()> {
        self.inner.rollback_to(txn, name)
    }

    fn release_savepoint(&self, txn: TxnId, name: &str) -> Result<()> {
        self.inner.release_savepoint(txn, name)
    }

    fn create_table(&self, txn: TxnId, def: &TableDef) -> Result<TableId> {
        self.inner.create_table(txn, def)
    }

    fn drop_table(&self, txn: TxnId, table: TableId) -> Result<()> {
        self.inner.drop_table(txn, table)
    }

    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>> {
        self.inner.lookup_table(name)
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        self.inner.list_tables()
    }

    fn list_tables_as_of(&self, txn: TxnId) -> Result<Vec<String>> {
        self.inner.list_tables_as_of(txn)
    }

    fn list_tables_qualified_as_of(&self, txn: TxnId) -> Result<Vec<(String, String)>> {
        self.inner.list_tables_qualified_as_of(txn)
    }

    fn data_version(&self) -> Option<u64> {
        self.inner.data_version()
    }

    fn lookup_table_as_of(&self, txn: TxnId, name: &str) -> Result<Option<TableSchema>> {
        self.inner.lookup_table_as_of(txn, name)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>> {
        self.inner.lookup_table_in(schema, name)
    }

    fn lookup_table_as_of_in(
        &self,
        txn: TxnId,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>> {
        self.inner.lookup_table_as_of_in(txn, schema, name)
    }

    fn insert(&self, txn: TxnId, table: TableId, tuple: &[u8]) -> Result<Tid> {
        self.inner.insert(txn, table, tuple)
    }

    fn insert_batch(&self, txn: TxnId, table: TableId, tuples: &[Vec<u8>]) -> Result<Vec<Tid>> {
        self.inner.insert_batch(txn, table, tuples)
    }

    fn update(&self, txn: TxnId, table: TableId, tid: Tid, tuple: &[u8]) -> Result<Tid> {
        self.inner.update(txn, table, tid, tuple)
    }

    fn delete(&self, txn: TxnId, table: TableId, tid: Tid) -> Result<()> {
        self.inner.delete(txn, table, tid)
    }

    fn scan(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>> {
        let scan = self.inner.scan(txn, table)?;
        self.fire(Point::Scan, table);
        Ok(scan)
    }

    fn scan_committed(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>> {
        self.inner.scan_committed(txn, table)
    }

    fn lock_row(&self, txn: TxnId, table: TableId, tid: Tid, mode: RowLockMode) -> Result<()> {
        self.inner.lock_row(txn, table, tid, mode)
    }

    fn lock_row_current(
        &self,
        txn: TxnId,
        table: TableId,
        tid: Tid,
        mode: RowLockMode,
    ) -> Result<LockedRow> {
        self.inner.lock_row_current(txn, table, tid, mode)
    }

    fn lock_table(&self, txn: TxnId, table: TableId, mode: TableLockMode) -> Result<()> {
        self.fire(Point::Lock, table);
        self.inner.lock_table(txn, table, mode)
    }

    fn lock_key(&self, txn: TxnId, table: TableId, key_hash: u64, mode: RowLockMode) -> Result<()> {
        self.inner.lock_key(txn, table, key_hash, mode)
    }

    fn create_sequence(&self, txn: TxnId, def: &SequenceDef) -> Result<SequenceId> {
        self.inner.create_sequence(txn, def)
    }

    fn drop_sequence(&self, txn: TxnId, id: SequenceId) -> Result<()> {
        self.inner.drop_sequence(txn, id)
    }

    fn lookup_sequence(&self, name: &str) -> Result<Option<SequenceId>> {
        self.inner.lookup_sequence(name)
    }

    fn sequence_next(&self, id: SequenceId) -> Result<i64> {
        self.inner.sequence_next(id)
    }

    fn sequence_current(&self, id: SequenceId) -> Result<i64> {
        self.inner.sequence_current(id)
    }

    fn sequence_set(&self, id: SequenceId, value: i64) -> Result<()> {
        self.inner.sequence_set(id, value)
    }

    fn alter_sequence(&self, txn: TxnId, id: SequenceId, change: &SequenceChange) -> Result<()> {
        self.inner.alter_sequence(txn, id, change)
    }

    fn create_schema(&self, txn: TxnId, name: &str) -> Result<SchemaId> {
        self.inner.create_schema(txn, name)
    }

    fn create_temp_schema(&self, txn: TxnId, name: &str) -> Result<SchemaId> {
        self.inner.create_temp_schema(txn, name)
    }

    fn drop_schema(&self, txn: TxnId, id: SchemaId, cascade: bool) -> Result<()> {
        self.inner.drop_schema(txn, id, cascade)
    }

    fn lookup_schema(&self, name: &str) -> Result<Option<SchemaId>> {
        self.inner.lookup_schema(name)
    }

    fn list_schemas(&self) -> Result<Vec<(SchemaId, String)>> {
        self.inner.list_schemas()
    }

    fn create_index(&self, txn: TxnId, def: &IndexDef) -> Result<IndexId> {
        self.fire(Point::CreateIndex, def.table);
        self.inner.create_index(txn, def)
    }

    fn begin_fresh_reads(&self, txn: TxnId) -> Result<()> {
        self.inner.begin_fresh_reads(txn)
    }

    fn end_fresh_reads(&self, txn: TxnId) -> Result<()> {
        self.inner.end_fresh_reads(txn)
    }

    fn drop_index(&self, txn: TxnId, id: IndexId) -> Result<()> {
        self.inner.drop_index(txn, id)
    }

    fn lookup_index(&self, name: &str) -> Result<Option<IndexId>> {
        self.inner.lookup_index(name)
    }

    fn list_indexes(&self, table: TableId) -> Result<Vec<IndexDef>> {
        self.inner.list_indexes(table)
    }

    fn index_is_complete(&self, index: IndexId) -> Result<bool> {
        self.inner.index_is_complete(index)
    }

    fn index_insert(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        self.inner.index_insert(txn, index, key, tid)
    }

    fn index_insert_batch(
        &self,
        txn: TxnId,
        index: IndexId,
        entries: Vec<(Vec<u8>, Tid)>,
    ) -> Result<()> {
        self.inner.index_insert_batch(txn, index, entries)
    }

    fn index_delete(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        self.inner.index_delete(txn, index, key, tid)
    }

    fn index_scan(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        self.inner.index_scan(txn, index, lo, hi)
    }

    fn index_scan_directed(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
        direction: ScanDirection,
    ) -> Result<Box<dyn TupleScan>> {
        self.inner
            .index_scan_directed(txn, index, lo, hi, direction)
    }

    fn index_scan_directed_limited(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
        direction: ScanDirection,
        limit: Option<usize>,
    ) -> Result<Box<dyn TupleScan>> {
        self.inner
            .index_scan_directed_limited(txn, index, lo, hi, direction, limit)
    }

    fn index_scan_committed(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        self.inner.index_scan_committed(txn, index, lo, hi)
    }

    fn add_unique_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        columns: &[String],
        primary: bool,
        nulls_not_distinct: bool,
    ) -> Result<IndexId> {
        self.inner
            .add_unique_constraint(txn, table, name, columns, primary, nulls_not_distinct)
    }

    fn add_check_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        expr: &[u8],
    ) -> Result<()> {
        self.inner.add_check_constraint(txn, table, name, expr)
    }

    fn drop_constraint(&self, txn: TxnId, table: TableId, name: &str) -> Result<()> {
        self.inner.drop_constraint(txn, table, name)
    }

    fn list_constraints(&self, table: TableId) -> Result<Vec<Constraint>> {
        self.inner.list_constraints(table)
    }

    fn has_unique_constraint(&self, table: TableId) -> Result<bool> {
        self.inner.has_unique_constraint(table)
    }

    fn list_foreign_keys(&self, table: TableId) -> Result<Vec<ForeignKeyDef>> {
        self.inner.list_foreign_keys(table)
    }

    fn add_foreign_key(&self, txn: TxnId, def: &ForeignKeyDef) -> Result<IndexId> {
        self.inner.add_foreign_key(txn, def)
    }

    fn fk_check(&self, txn: TxnId, name: &str, key: &[u8]) -> Result<()> {
        self.inner.fk_check(txn, name, key)
    }

    fn fk_on_delete(&self, txn: TxnId, parent_table: TableId, parent_key: &[u8]) -> Result<u64> {
        self.inner.fk_on_delete(txn, parent_table, parent_key)
    }

    fn analyze_table(&self, txn: TxnId, table: TableId, stats: &TableStats) -> Result<()> {
        self.inner.analyze_table(txn, table, stats)
    }

    fn table_stats(&self, table: TableId) -> Result<Option<TableStats>> {
        self.inner.table_stats(table)
    }

    fn row_count(&self, table: TableId) -> Result<u64> {
        self.inner.row_count(table)
    }

    fn approx_row_count(&self, table: TableId) -> Result<u64> {
        self.inner.approx_row_count(table)
    }

    fn churn_since_analyze(&self, table: TableId) -> Result<u64> {
        self.inner.churn_since_analyze(table)
    }

    fn alter_table(&self, txn: TxnId, table: TableId, op: &AlterOp) -> Result<()> {
        self.inner.alter_table(txn, table, op)
    }

    fn schema_for_version(&self, table: TableId, version: u32) -> Result<Option<TableSchema>> {
        self.inner.schema_for_version(table, version)
    }

    fn current_schema_version(&self, table: TableId) -> Result<Option<u32>> {
        self.inner.current_schema_version(table)
    }

    fn vacuum(&self) -> Result<usize> {
        self.inner.vacuum()
    }

    fn prepare_transaction(
        &self,
        txn: TxnId,
        gid: &str,
        owner: &str,
        database: &str,
    ) -> Result<()> {
        self.inner.prepare_transaction(txn, gid, owner, database)
    }

    fn commit_prepared(&self, gid: &str, owner: Option<&str>) -> Result<()> {
        self.inner.commit_prepared(gid, owner)
    }

    fn rollback_prepared(&self, gid: &str, owner: Option<&str>) -> Result<()> {
        self.inner.rollback_prepared(gid, owner)
    }

    fn list_prepared(&self) -> Result<Vec<PreparedTransaction>> {
        self.inner.list_prepared()
    }

    fn checkpoint(&self) -> Result<()> {
        self.inner.checkpoint()
    }
}

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(
        &self,
        name: &str,
    ) -> std::result::Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> std::result::Result<Vec<IndexInfo>, nusadb_sql::Error> {
        Ok(Vec::new())
    }
}

fn run(
    session: &mut Session,
    engine: &dyn StorageEngine,
    sql: &str,
) -> std::result::Result<ExecutionResult, nusadb_sql::Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    session.execute(plan(logical))
}

fn ok(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> ExecutionResult {
    run(session, engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// One forced interleaving: run `setup`, open the writer's transaction with `writer`, arm it to
/// commit at `point` on table `at`, then run `ddl` with no retry. Afterwards the writer's effect is
/// committed whatever happened, `check` returns `expect` (proving it), and `ddl` must have been
/// refused with one of `allowed`.
fn race(
    setup: &[&str],
    writer: &[&str],
    point: Point,
    at: &str,
    ddl: &str,
    allowed: &[&str],
    check: &str,
    expect: &[Vec<Value>],
) {
    let engine = Interleave::new();
    let mut s = Session::new(&engine);
    for sql in setup {
        ok(&mut s, &engine, sql);
    }
    let target = engine.lookup_table(at).unwrap().unwrap().id;
    let mut w = Session::new(&engine);
    ok(&mut w, &engine, "BEGIN");
    for sql in writer {
        ok(&mut w, &engine, sql);
    }
    *engine.armed.lock().unwrap() = Some((point, target, w.current_txn().unwrap()));
    ok(&mut s, &engine, "SET max_autocommit_retries = 0");
    let outcome = run(&mut s, &engine, ddl);
    let fired = engine.armed.lock().unwrap().take().is_none();
    if !fired {
        engine.inner.commit(w.current_txn().unwrap()).unwrap();
    }
    let ExecutionResult::Rows { rows, .. } = ok(&mut s, &engine, check) else {
        panic!("{check} returned no rows");
    };
    assert_eq!(rows, expect, "{ddl}: the writer's change committed");
    match outcome {
        Err(e) => assert!(
            allowed.contains(&e.sqlstate()),
            "{ddl}: unexpected refusal {e} ({})",
            e.sqlstate()
        ),
        Ok(_) => panic!(
            "{ddl} succeeded although the writer's committed change breaks it (the writer \
             committed at {point:?}, fired: {fired})"
        ),
    }
    drop(w);
}

const PARTITIONED: &[&str] = &[
    "CREATE TABLE pt (k INT, v INT) PARTITION BY RANGE (k)",
    "CREATE TABLE pt1 PARTITION OF pt FOR VALUES FROM (0) TO (100)",
    "INSERT INTO pt VALUES (1, 1), (2, 2)",
];
const PLAIN: &[&str] = &[
    "CREATE TABLE p (k INT, v INT)",
    "INSERT INTO p VALUES (1, 1), (2, 2)",
];

#[test]
fn add_check_on_a_partitioned_table_sees_a_writer_that_committed_before_its_lock() {
    for point in [Point::Lock, Point::Scan] {
        race(
            PARTITIONED,
            &["INSERT INTO pt VALUES (3, -1)"],
            point,
            "pt1",
            "ALTER TABLE pt ADD CONSTRAINT ck CHECK (v > 0)",
            &["40001", "23514"],
            "SELECT k FROM pt1 WHERE v <= 0",
            &[vec![Value::Int(3)]],
        );
    }
}

#[test]
fn add_check_on_a_plain_table_sees_a_writer_that_committed_before_its_lock() {
    race(
        PLAIN,
        &["INSERT INTO p VALUES (3, -1)"],
        Point::Lock,
        "p",
        "ALTER TABLE p ADD CONSTRAINT ck CHECK (v > 0)",
        &["23514"],
        "SELECT k FROM p WHERE v <= 0",
        &[vec![Value::Int(3)]],
    );
}

#[test]
fn set_not_null_sees_a_writer_that_committed_before_its_lock() {
    for (setup, table, at) in [(PARTITIONED, "pt", "pt1"), (PLAIN, "p", "p")] {
        for point in [Point::Lock, Point::Scan] {
            race(
                setup,
                &[&format!("INSERT INTO {table} VALUES (3, NULL)")],
                point,
                at,
                &format!("ALTER TABLE {table} ALTER COLUMN v SET NOT NULL"),
                &["40001", "23502"],
                &format!("SELECT k FROM {at} WHERE v IS NULL"),
                &[vec![Value::Int(3)]],
            );
        }
    }
}

#[test]
fn add_unique_sees_a_duplicate_committed_before_its_lock() {
    race(
        PLAIN,
        &["INSERT INTO p VALUES (3, 1)"],
        Point::Lock,
        "p",
        "ALTER TABLE p ADD CONSTRAINT pu UNIQUE (v)",
        &["23505"],
        "SELECT k FROM p WHERE v = 1 ORDER BY k",
        &[vec![Value::Int(1)], vec![Value::Int(3)]],
    );
}

#[test]
fn add_foreign_key_sees_parent_and_child_changes_committed_before_its_lock() {
    race(
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "CREATE TABLE ch (k INT, pid INT)",
            "INSERT INTO par VALUES (1), (2)",
            "INSERT INTO ch VALUES (1, 1)",
        ],
        &["DELETE FROM par WHERE id = 1"],
        Point::Lock,
        "ch",
        "ALTER TABLE ch ADD CONSTRAINT chfk FOREIGN KEY (pid) REFERENCES par (id)",
        &["23503"],
        "SELECT id FROM par",
        &[vec![Value::Int(2)]],
    ); // The child side: an orphan child row committed before the lock.
    race(
        &[
            "CREATE TABLE par (id INT PRIMARY KEY)",
            "CREATE TABLE ch (k INT, pid INT)",
            "INSERT INTO par VALUES (1)",
            "INSERT INTO ch VALUES (1, 1)",
        ],
        &["INSERT INTO ch VALUES (2, 9)"],
        Point::Lock,
        "ch",
        "ALTER TABLE ch ADD CONSTRAINT chfk FOREIGN KEY (pid) REFERENCES par (id)",
        &["23503"],
        "SELECT k FROM ch WHERE pid = 9",
        &[vec![Value::Int(2)]],
    );
}

/// An index built over a row a writer committed just before the build took the table holds that
/// row: the backfill reads the committed rows, not the snapshot from before the lock.
#[test]
fn create_index_backfills_a_row_committed_before_its_lock() {
    let engine = Interleave::new();
    let mut s = Session::new(&engine);
    for sql in PLAIN {
        ok(&mut s, &engine, sql);
    }
    let p = engine.lookup_table("p").unwrap().unwrap().id;
    let mut w = Session::new(&engine);
    ok(&mut w, &engine, "BEGIN");
    ok(&mut w, &engine, "INSERT INTO p VALUES (3, 30)");
    *engine.armed.lock().unwrap() = Some((Point::CreateIndex, p, w.current_txn().unwrap()));
    ok(&mut s, &engine, "SET max_autocommit_retries = 0");
    ok(&mut s, &engine, "CREATE INDEX p_v ON p (v)");
    assert!(
        engine.armed.lock().unwrap().is_none(),
        "the writer committed inside the build"
    );
    let index = engine.lookup_index("p_v").unwrap().unwrap();
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let mut scan = engine
        .index_scan(txn, index, Bound::Unbounded, Bound::Unbounded)
        .unwrap();
    let mut entries = 0;
    while scan.try_next().unwrap().is_some() {
        entries += 1;
    }
    drop(scan);
    engine.commit(txn).unwrap();
    assert_eq!(entries, 3, "the index covers every committed row");
    drop(w);
}

/// A table rewrite (a column added with a default, a column's type changed) takes in a row a
/// writer committed just before the rewrite took the table, rather than leaving it behind.
#[test]
fn a_rewrite_takes_in_a_row_committed_before_its_lock() {
    for (ddl, check, want) in [
        (
            "ALTER TABLE p ADD COLUMN w INT DEFAULT 7",
            "SELECT k, w FROM p ORDER BY k",
            vec![
                vec![Value::Int(1), Value::Int(7)],
                vec![Value::Int(2), Value::Int(7)],
                vec![Value::Int(3), Value::Int(7)],
            ],
        ),
        (
            "ALTER TABLE p ALTER COLUMN v TYPE TEXT",
            "SELECT k, v FROM p ORDER BY k",
            vec![
                vec![Value::Int(1), Value::Text("1".to_owned())],
                vec![Value::Int(2), Value::Text("2".to_owned())],
                vec![Value::Int(3), Value::Text("3".to_owned())],
            ],
        ),
    ] {
        let engine = Interleave::new();
        let mut s = Session::new(&engine);
        for sql in PLAIN {
            ok(&mut s, &engine, sql);
        }
        let p = engine.lookup_table("p").unwrap().unwrap().id;
        let mut w = Session::new(&engine);
        ok(&mut w, &engine, "BEGIN");
        ok(&mut w, &engine, "INSERT INTO p VALUES (3, 3)");
        *engine.armed.lock().unwrap() = Some((Point::Lock, p, w.current_txn().unwrap()));
        ok(&mut s, &engine, "SET max_autocommit_retries = 0");
        ok(&mut s, &engine, ddl);
        assert!(
            engine.armed.lock().unwrap().is_none(),
            "{ddl}: the writer committed"
        );
        let ExecutionResult::Rows { rows, .. } = ok(&mut s, &engine, check) else {
            panic!("no rows");
        };
        assert_eq!(rows, want, "{ddl}");
        drop(w);
    }
}
