use std::collections::HashMap;
use std::sync::Mutex;

use nusadb_core::engine::{Constraint, ConstraintKind};
use nusadb_core::{
    ColumnType, Error as CoreError, IsolationLevel, PageId, Result as CoreResult, SharedTuple,
    SlotIdx, StorageEngine, TableDef, TableId, TableSchema, Tid, TupleScan, TxnId,
};

use super::{
    ExecutionResult, Row, describe_column_types, describe_columns, execute, execute_in_txn,
};
use crate::analyzer::{Catalog, analyze};
use crate::ast::Value;
use crate::error::Error;
use crate::parser::parse;
use crate::planner::plan;

// --- In-memory mock StorageEngine ----------------------------------

struct MockEngine {
    state: Mutex<MockState>,
}

struct MockState {
    next_table_id: u64,
    next_txn_id: u64,
    next_slot: u16,
    tables_by_id: HashMap<TableId, MockTable>,
    tables_by_name: HashMap<String, TableId>,
    /// How many times [`StorageEngine::begin_statement`] has been called — lets a test assert the
    /// per-statement snapshot refresh actually fires at the execution choke-point.
    begin_statements: u64,
    /// When set, `commit` fails with `FsyncFailed` (simulating a group-fsync error at the
    /// durability point) so the rollback-on-commit-failure path is testable.
    fail_commit: bool,
    /// Number of `rollback` calls the engine has seen — a leak test asserts the SQL layer rolls a
    /// failed commit back instead of stranding the transaction.
    rollback_count: u32,
}

struct MockTable {
    schema: TableSchema,
    rows: Vec<(Tid, Vec<u8>)>,
    stats: Option<nusadb_core::TableStats>,
    /// `CHECK` constraints (incl. the synthetic type-bound ones), so CREATE TABLE with a check
    /// succeeds and the predicate is enforced on write — mirroring the production engine.
    checks: Vec<Constraint>,
}

impl MockEngine {
    fn new() -> Self {
        Self {
            state: Mutex::new(MockState {
                next_table_id: 1,
                next_txn_id: 1,
                next_slot: 1,
                tables_by_id: HashMap::new(),
                tables_by_name: HashMap::new(),
                begin_statements: 0,
                fail_commit: false,
                rollback_count: 0,
            }),
        }
    }

    /// Make every subsequent `commit` fail (as a group-fsync error would), so a test can seed data
    /// with working commits first and then drive a commit-failure and read
    /// [`rollback_count`](Self::rollback_count).
    fn set_failing_commit(&self, fail: bool) {
        self.state.lock().unwrap().fail_commit = fail;
    }

    /// How many times `rollback` has been called on the engine.
    fn rollback_count(&self) -> u32 {
        self.state.lock().unwrap().rollback_count
    }
}

impl StorageEngine for MockEngine {
    fn begin(&self, _level: IsolationLevel) -> CoreResult<TxnId> {
        let mut s = self.state.lock().unwrap();
        let id = TxnId(s.next_txn_id);
        s.next_txn_id += 1;
        Ok(id)
    }
    fn begin_statement(&self, _txn: TxnId) -> CoreResult<()> {
        // No MVCC snapshots in this in-memory double: every read already sees latest state. Count
        // the calls so a test can assert the refresh fires at the execution choke-point.
        self.state.lock().unwrap().begin_statements += 1;
        Ok(())
    }
    fn commit(&self, _txn: TxnId) -> CoreResult<()> {
        if self.state.lock().unwrap().fail_commit {
            return Err(CoreError::FsyncFailed(
                "mock commit fsync failed".to_owned(),
            ));
        }
        Ok(())
    }
    fn rollback(&self, _txn: TxnId) -> CoreResult<()> {
        self.state.lock().unwrap().rollback_count += 1;
        Ok(())
    }
    fn savepoint(&self, _txn: TxnId, _name: &str) -> CoreResult<()> {
        Ok(())
    }
    fn rollback_to(&self, _txn: TxnId, _name: &str) -> CoreResult<()> {
        Ok(())
    }

    fn create_table(&self, _txn: TxnId, def: &TableDef) -> CoreResult<TableId> {
        let mut s = self.state.lock().unwrap();
        if s.tables_by_name.contains_key(&def.name) {
            return Err(CoreError::TableExists {
                name: def.name.clone(),
            });
        }
        let id = TableId(s.next_table_id);
        s.next_table_id += 1;
        let schema = TableSchema {
            schema: "public".to_owned(),
            id,
            name: def.name.clone(),
            columns: def.columns.clone(),
        };
        s.tables_by_name.insert(def.name.clone(), id);
        s.tables_by_id.insert(
            id,
            MockTable {
                schema,
                rows: Vec::new(),
                stats: None,
                checks: Vec::new(),
            },
        );
        Ok(id)
    }

    fn drop_table(&self, _txn: TxnId, table: TableId) -> CoreResult<()> {
        let mut s = self.state.lock().unwrap();
        if let Some(t) = s.tables_by_id.remove(&table) {
            s.tables_by_name.remove(&t.schema.name);
        }
        Ok(())
    }

    fn add_check_constraint(
        &self,
        _txn: TxnId,
        table: TableId,
        name: &str,
        expr: &[u8],
    ) -> CoreResult<()> {
        let mut s = self.state.lock().unwrap();
        let Some(t) = s.tables_by_id.get_mut(&table) else {
            return Err(CoreError::TableNotFound {
                name: format!("add_check_constraint: unknown table {table:?}"),
            });
        };
        t.checks.push(Constraint {
            name: name.to_owned(),
            table,
            columns: Vec::new(),
            kind: ConstraintKind::Check,
            index: None,
            expr: Some(expr.to_vec()),
        });
        Ok(())
    }

    fn drop_constraint(&self, _txn: TxnId, table: TableId, name: &str) -> CoreResult<()> {
        let mut s = self.state.lock().unwrap();
        if let Some(t) = s.tables_by_id.get_mut(&table) {
            t.checks.retain(|c| c.name != name);
        }
        Ok(())
    }

    fn list_constraints(&self, table: TableId) -> CoreResult<Vec<Constraint>> {
        let s = self.state.lock().unwrap();
        Ok(s.tables_by_id
            .get(&table)
            .map(|t| t.checks.clone())
            .unwrap_or_default())
    }

    fn lookup_table(&self, name: &str) -> CoreResult<Option<TableSchema>> {
        let s = self.state.lock().unwrap();
        Ok(s.tables_by_name
            .get(name)
            .and_then(|id| s.tables_by_id.get(id))
            .map(|t| t.schema.clone()))
    }

    fn insert(&self, _txn: TxnId, table: TableId, tuple: &[u8]) -> CoreResult<Tid> {
        let mut s = self.state.lock().unwrap();
        let slot = s.next_slot;
        s.next_slot += 1;
        let tid = Tid {
            page: PageId(0),
            slot: SlotIdx(slot),
        };
        let t = s
            .tables_by_id
            .get_mut(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        t.rows.push((tid, tuple.to_vec()));
        Ok(tid)
    }

    fn update(&self, _txn: TxnId, table: TableId, tid: Tid, tuple: &[u8]) -> CoreResult<Tid> {
        let mut s = self.state.lock().unwrap();
        let t = s
            .tables_by_id
            .get_mut(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        for row in &mut t.rows {
            if row.0 == tid {
                row.1 = tuple.to_vec();
                return Ok(tid);
            }
        }
        Err(CoreError::TableNotFound {
            name: format!("no row at {tid:?}"),
        })
    }

    fn delete(&self, _txn: TxnId, table: TableId, tid: Tid) -> CoreResult<()> {
        let mut s = self.state.lock().unwrap();
        let t = s
            .tables_by_id
            .get_mut(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        t.rows.retain(|(t, _)| *t != tid);
        Ok(())
    }

    fn scan(&self, _txn: TxnId, table: TableId) -> CoreResult<Box<dyn TupleScan>> {
        let s = self.state.lock().unwrap();
        let t = s
            .tables_by_id
            .get(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        Ok(Box::new(MockScan {
            snapshot: t.rows.clone(),
            pos: 0,
        }))
    }

    fn alter_table(
        &self,
        _txn: TxnId,
        table: TableId,
        op: &nusadb_core::AlterOp,
    ) -> CoreResult<()> {
        use nusadb_core::AlterOp;
        let mut s = self.state.lock().unwrap();
        if let AlterOp::RenameTable { name } = op {
            let old = s.tables_by_id.get(&table).map(|t| t.schema.name.clone());
            if let Some(old) = old {
                s.tables_by_name.remove(&old);
                s.tables_by_name.insert(name.clone(), table);
                if let Some(t) = s.tables_by_id.get_mut(&table) {
                    t.schema.name = name.clone();
                }
            }
            return Ok(());
        }
        let t = s
            .tables_by_id
            .get_mut(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        match op {
            AlterOp::AddColumn(c) => t.schema.columns.push(c.clone()),
            AlterOp::DropColumn { name } => t.schema.columns.retain(|c| &c.name != name),
            AlterOp::RenameColumn { from, to } => {
                for c in &mut t.schema.columns {
                    if &c.name == from {
                        c.name = to.clone();
                    }
                }
            },
            AlterOp::AlterColumnType { column, ty } => {
                for c in &mut t.schema.columns {
                    if &c.name == column {
                        c.ty = *ty;
                    }
                }
            },
            AlterOp::SetNotNull { column } => {
                for c in &mut t.schema.columns {
                    if &c.name == column {
                        c.nullable = false;
                    }
                }
            },
            AlterOp::DropNotNull { column } => {
                for c in &mut t.schema.columns {
                    if &c.name == column {
                        c.nullable = true;
                    }
                }
            },
            // Handled by the early return above.
            AlterOp::RenameTable { .. } => {},
        }
        Ok(())
    }

    fn row_count(&self, table: TableId) -> CoreResult<u64> {
        let s = self.state.lock().unwrap();
        Ok(s.tables_by_id
            .get(&table)
            .map_or(0, |t| t.rows.len() as u64))
    }

    fn analyze_table(
        &self,
        _txn: TxnId,
        table: TableId,
        stats: &nusadb_core::TableStats,
    ) -> CoreResult<()> {
        let mut s = self.state.lock().unwrap();
        let t = s
            .tables_by_id
            .get_mut(&table)
            .ok_or_else(|| CoreError::TableNotFound {
                name: format!("{table:?}"),
            })?;
        t.stats = Some(stats.clone());
        Ok(())
    }

    fn table_stats(&self, table: TableId) -> CoreResult<Option<nusadb_core::TableStats>> {
        let s = self.state.lock().unwrap();
        Ok(s.tables_by_id.get(&table).and_then(|t| t.stats.clone()))
    }
}

impl Catalog for MockEngine {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        StorageEngine::lookup_table(self, name).map_err(Error::from)
    }

    fn table_stats(&self, name: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        // Mirror the production adapters (wire/e2e): resolve the table by name, then fetch its
        // stored ANALYZE stats. Lets the planner annotate a SELECT with its scanned-row estimate.
        let Some(schema) = StorageEngine::lookup_table(self, name).map_err(Error::from)? else {
            return Ok(None);
        };
        StorageEngine::table_stats(self, schema.id).map_err(Error::from)
    }
}

struct MockScan {
    snapshot: Vec<(Tid, Vec<u8>)>,
    pos: usize,
}

impl TupleScan for MockScan {
    fn try_next(&mut self) -> CoreResult<Option<(Tid, SharedTuple)>> {
        let item = self
            .snapshot
            .get(self.pos)
            .map(|(tid, bytes)| (*tid, SharedTuple::from(bytes.as_slice())));
        if item.is_some() {
            self.pos += 1;
        }
        Ok(item)
    }
}

// --- End-to-end helpers --------------------------------------------

fn run(sql: &str, engine: &MockEngine) -> Result<ExecutionResult, Error> {
    let stmt = parse(sql)?;
    let logical = analyze(stmt, engine)?;
    execute(plan(logical), engine)
}

fn rows_of(result: ExecutionResult) -> (Vec<String>, Vec<Row>) {
    match result {
        ExecutionResult::Rows { columns, rows } => (columns, rows),
        other => panic!("expected Rows, got {other:?}"),
    }
}

fn inserted_count(result: ExecutionResult) -> usize {
    match result {
        ExecutionResult::Inserted(n) => n,
        other => panic!("expected Inserted, got {other:?}"),
    }
}

fn updated_count(result: ExecutionResult) -> usize {
    match result {
        ExecutionResult::Updated(n) => n,
        other => panic!("expected Updated, got {other:?}"),
    }
}

fn deleted_count(result: ExecutionResult) -> usize {
    match result {
        ExecutionResult::Deleted(n) => n,
        other => panic!("expected Deleted, got {other:?}"),
    }
}

fn setup() -> MockEngine {
    let engine = MockEngine::new();
    run(
        "CREATE TABLE users (id INT NOT NULL, name TEXT, age INT, score FLOAT, active BOOL)",
        &engine,
    )
    .unwrap();
    engine
}

// --- Statement snapshot --------------------

#[test]
fn execute_in_txn_refreshes_the_statement_snapshot() {
    // Regression for the reopened: a statement run inside an already-open
    // transaction must refresh the RC/RU statement snapshot at the shared execution choke-point.
    // The extended-query / prepared-statement path reaches the engine only through `execute_in_txn`
    // (via `execute_in_txn_as_with_settings`); an earlier fix that refreshed only on the
    // simple-query call-sites let a prepared `UPDATE` silently skip rows against a stale snapshot.
    // Assert `execute_in_txn` calls `begin_statement` exactly once per statement.
    let engine = setup();
    run("INSERT INTO users VALUES (1, 'a', 30, 1.0, true)", &engine).unwrap();

    let before = engine.state.lock().unwrap().begin_statements;
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    let stmt = parse("SELECT * FROM users").unwrap();
    let logical = analyze(stmt, &engine).unwrap();
    execute_in_txn(plan(logical), &engine, txn).unwrap();
    let after = engine.state.lock().unwrap().begin_statements;

    assert_eq!(
        after - before,
        1,
        "execute_in_txn must refresh the statement snapshot exactly once per statement"
    );
}

// --- Commit-failure rollback ---------------

/// Drive one statement through an explicit `Session` (so `BEGIN`/`COMMIT`/`ROLLBACK` and the
/// in-transaction dispatch paths are exercised, unlike the free `execute` used by `run`).
fn session_run(
    session: &mut super::Session,
    engine: &MockEngine,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, engine)?;
    session.execute(plan(logical))
}

/// A commit that fails at the durability point (group-fsync error) must be
/// rolled back, not stranded. A stranded transaction stays in the engine's active set — pinning the
/// purge horizon forever — and never releases its locks (a permanent 40001 storm on those rows). We
/// drive both the auto-commit path (a bare INSERT) and the explicit `COMMIT` path and assert each
/// surfaces the error AND issues exactly one rollback.
#[test]
fn failed_commit_rolls_back_instead_of_leaking_txn() {
    // Auto-commit path: a bare INSERT begins a one-statement transaction and commits it.
    let engine = setup();
    engine.set_failing_commit(true);
    let err = run("INSERT INTO users VALUES (1, 'a', 1, 1.0, TRUE)", &engine)
        .expect_err("a failed commit must surface the error, not be swallowed");
    assert!(
        matches!(err, Error::Core(CoreError::FsyncFailed(_))),
        "expected the fsync failure to propagate, got {err:?}"
    );
    assert_eq!(
        engine.rollback_count(),
        1,
        "auto-commit must roll a failed commit back exactly once"
    );

    // Explicit COMMIT path: BEGIN … INSERT … COMMIT, where COMMIT is the failing step. The INSERT
    // runs inside the open transaction (no commit yet); only the explicit COMMIT triggers fsync.
    let engine = setup();
    let mut session = super::Session::new(&engine);
    session_run(&mut session, &engine, "BEGIN").unwrap();
    session_run(
        &mut session,
        &engine,
        "INSERT INTO users VALUES (2, 'b', 2, 2.0, TRUE)",
    )
    .unwrap();
    engine.set_failing_commit(true);
    let err = session_run(&mut session, &engine, "COMMIT")
        .expect_err("a failed explicit COMMIT must surface the error");
    assert!(
        matches!(err, Error::Core(CoreError::FsyncFailed(_))),
        "expected the fsync failure to propagate, got {err:?}"
    );
    assert_eq!(
        engine.rollback_count(),
        1,
        "explicit COMMIT must roll a failed commit back exactly once (not strand the txn)"
    );
    // The session must not still hold the transaction: a follow-up COMMIT has nothing to commit.
    let follow_up = session_run(&mut session, &engine, "COMMIT")
        .expect_err("the transaction is over, so a second COMMIT must fail");
    assert!(
        matches!(follow_up, Error::Unsupported(_)),
        "expected 'COMMIT without an active transaction', got {follow_up:?}"
    );
}

// --- work_mem session tunability ------------------------------------

#[test]
fn parse_work_mem_follows_pg_units() {
    for (input, expect) in [
        ("4MB", Some(4 * 1024 * 1024)),
        ("512kB", Some(512 * 1024)),
        ("512KB", Some(512 * 1024)), // unit is case-insensitive
        ("1GB", Some(1024 * 1024 * 1024)),
        ("1TB", Some(1024_usize * 1024 * 1024 * 1024)),
        ("4096", Some(4096 * 1024)), // bare integer = kilobytes (conventional memory-GUC unit)
        (" 4 MB ", Some(4 * 1024 * 1024)),
        ("0", Some(0)), // unlimited, matching --work-mem 0
        ("abc", None),
        ("4XB", None),
        ("-1", None),
        ("1.5MB", None), // fractional values are rejected, not rounded
        ("", None),
        ("999999999999999999TB", None), // overflow
    ] {
        assert_eq!(
            super::parse_work_mem(input),
            expect,
            "parse_work_mem({input:?})"
        );
    }
}

/// The executor budget — both the per-stage materialization check and the recursive-CTE guard —
/// must honour a session `SET work_mem`, not only the process-wide `--work-mem` default it used to
/// read exclusively (the decisive probe: a recursion far above `SET work_mem` completed regardless
/// of the setting, and the OOM message cited the server default).
#[test]
fn set_work_mem_governs_the_statement_budget() {
    let engine = setup();
    let mut session = super::Session::new(&engine);
    // Seed a row wide enough to blow a 1kB budget — inserted while the budget is still unlimited.
    let wide = "x".repeat(4096);
    session_run(
        &mut session,
        &engine,
        &format!("INSERT INTO users VALUES (1, '{wide}', 1, 1.0, TRUE)"),
    )
    .unwrap();

    session_run(&mut session, &engine, "SET work_mem = '1kB'").unwrap();
    // Per-stage budget: scanning the wide row materializes far more than 1kB.
    let err = session_run(&mut session, &engine, "SELECT * FROM users")
        .expect_err("a 1kB session budget must trip on a 4kB row");
    assert!(
        matches!(err, Error::Core(CoreError::OutOfMemory(_))),
        "expected OutOfMemory, got {err:?}"
    );
    // The message must cite the SESSION budget (1024 bytes), proving the SET was honoured —
    // QA's finding was precisely that the message cited the server default instead.
    let msg = err.to_string();
    assert!(msg.contains("1024"), "budget in message: {msg}");

    // The recursive-CTE guard (the reported finding) reads the same session budget.
    let err = session_run(
        &mut session,
        &engine,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 100000) \
         SELECT count(*) FROM r",
    )
    .expect_err("a 1kB session budget must stop the recursion");
    let msg = err.to_string();
    assert!(
        msg.contains("recursive CTE exceeded work_mem of 1024"),
        "recursion budget in message: {msg}"
    );

    // RESET returns to the process default (unlimited in this test binary): both statements
    // succeed again, including a recursion the 1kB budget would have stopped.
    session_run(&mut session, &engine, "RESET work_mem").unwrap();
    session_run(&mut session, &engine, "SELECT * FROM users").unwrap();
    session_run(
        &mut session,
        &engine,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 2000) \
         SELECT count(*) FROM r",
    )
    .unwrap();
}

/// An unparseable `work_mem` is rejected loudly at SET time (SQLSTATE 22023,
/// `invalid_parameter_value`) instead of being stored and then silently ignored by the budget
/// check's fallback.
#[test]
fn set_work_mem_rejects_an_invalid_value() {
    let engine = setup();
    let mut session = super::Session::new(&engine);
    let err = session_run(&mut session, &engine, "SET work_mem = 'lots'")
        .expect_err("an unparseable work_mem must be rejected at SET time");
    assert!(
        matches!(
            &err,
            Error::Coded {
                sqlstate: "22023",
                ..
            }
        ),
        "expected 22023 invalid_parameter_value, got {err:?}"
    );
    // A valid value is accepted.
    session_run(&mut session, &engine, "SET work_mem = '64MB'").unwrap();

    // `statement_timeout` gets the same SET-time validation (it arms the wire server's
    // per-statement cancel timer; a typo must not silently disable it).
    let err = session_run(&mut session, &engine, "SET statement_timeout = 'banana'")
        .expect_err("an unparseable statement_timeout must be rejected at SET time");
    assert!(
        matches!(
            &err,
            Error::Coded {
                sqlstate: "22023",
                ..
            }
        ),
        "expected 22023 invalid_parameter_value, got {err:?}"
    );
    session_run(&mut session, &engine, "SET statement_timeout = '100ms'").unwrap();
}

// --- DDL -----------------------------------------------------------

#[test]
fn create_and_drop_table() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (a INT)", &engine).unwrap();
    assert!(matches!(
        run("SELECT * FROM t", &engine).unwrap(),
        ExecutionResult::Rows { .. }
    ));
    run("DROP TABLE t", &engine).unwrap();
    // After drop, table is gone — analyzer rejects.
    assert!(matches!(
        run("SELECT * FROM t", &engine),
        Err(Error::TableNotFound { .. })
    ));
}

#[test]
fn create_if_not_exists_on_existing_is_noop() {
    let engine = setup();
    assert!(matches!(
        run("CREATE TABLE IF NOT EXISTS users (id INT)", &engine).unwrap(),
        ExecutionResult::Created(_)
    ));
}

#[test]
fn comment_on_table_and_column_execute() {
    let engine = setup();
    assert!(matches!(
        run("COMMENT ON TABLE users IS 'accounts'", &engine).unwrap(),
        ExecutionResult::Commented
    ));
    assert!(matches!(
        run("COMMENT ON COLUMN users.name IS NULL", &engine).unwrap(),
        ExecutionResult::Commented
    ));
    // The target must exist — the analyzer rejects a missing column before execution.
    assert!(matches!(
        run("COMMENT ON COLUMN users.ghost IS 'x'", &engine),
        Err(Error::ColumnNotFound { .. })
    ));
}

#[test]
fn explain_comment_renders_target() {
    let engine = setup();
    let (_, rows) = rows_of(run("EXPLAIN COMMENT ON COLUMN users.name IS 'x'", &engine).unwrap());
    let plan = format!("{rows:?}");
    assert!(plan.contains("Comment: users.name"), "{plan}");
}

#[test]
fn drop_if_exists_missing_is_noop() {
    let engine = MockEngine::new();
    assert!(matches!(
        run("DROP TABLE IF EXISTS ghost", &engine).unwrap(),
        ExecutionResult::Dropped
    ));
}

// --- ALTER TABLE ---------------------------------------------------

fn altered(result: ExecutionResult) {
    match result {
        ExecutionResult::Altered => {},
        other => panic!("expected Altered, got {other:?}"),
    }
}

/// `users` with two rows; one has a NULL `age`.
fn seeded_users() -> MockEngine {
    let engine = setup();
    run(
        "INSERT INTO users VALUES (1, 'alice', 30, 9.5, TRUE)",
        &engine,
    )
    .unwrap();
    run(
        "INSERT INTO users VALUES (2, 'bob', NULL, 1.0, FALSE)",
        &engine,
    )
    .unwrap();
    engine
}

/// Phase 1 oracle: `stream_op` must yield exactly what `execute_op` does, in the same order —
/// linear operators stream, blocking ones fall back to materialize. The materializing path is the
/// oracle.
fn assert_stream_eq_execute(sql: &str, engine: &MockEngine) {
    use crate::planner::PhysicalPlan;

    let logical = analyze(parse(sql).unwrap(), engine).unwrap();
    let PhysicalPlan::Select(op, _) = plan(logical) else {
        panic!("expected a SELECT plan for: {sql}");
    };
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let materialized = super::ops::execute_op(&op, engine, txn).unwrap();
    let mut source = super::stream::stream_op(&op, engine, txn).unwrap();
    let mut streamed = Vec::new();
    while let Some(row) = super::stream::RowSource::try_next(&mut *source).unwrap() {
        streamed.push(row);
    }
    engine.commit(txn).unwrap();
    assert_eq!(streamed, materialized, "stream_op != execute_op for: {sql}");
}

#[test]
fn stream_op_matches_execute_op_for_linear_and_blocking_plans() {
    let engine = seeded_users();
    for sql in [
        "SELECT * FROM users",                             // SeqScan
        "SELECT id, name FROM users",                      // Project
        "SELECT id FROM users WHERE age > 10",             // Filter + Project
        "SELECT id FROM users WHERE age IS NULL",          // Filter (NULL handling)
        "SELECT id, age FROM users WHERE age > 0 LIMIT 1", // Limit
        "SELECT id FROM users LIMIT 1 OFFSET 1",           // Limit with offset
        "SELECT id FROM users ORDER BY id",                // Sort -> materialize fallback
        "SELECT age, COUNT(*) FROM users GROUP BY age",    // GroupAggregate -> fallback
        "SELECT DISTINCT active FROM users",               // Distinct -> fallback
        // HashJoin streams its output: every kind must yield exactly the
        // materializing join's rows in the same order — matches per probe row, LEFT/FULL pads,
        // then the RIGHT/FULL unmatched-build drain, with USING coalescing applied per row.
        "SELECT u.name, v.name FROM users u JOIN users v ON u.id = v.id",
        "SELECT u.name, v.name FROM users u LEFT JOIN users v ON u.id = v.age",
        "SELECT u.name, v.name FROM users u RIGHT JOIN users v ON u.id = v.age",
        "SELECT u.name, v.name FROM users u FULL JOIN users v ON u.id = v.age",
        "SELECT u.id, v.id FROM users u JOIN users v ON u.id = v.id AND u.age > 10",
        "SELECT id, u.name FROM users u JOIN users v USING (id)",
        "SELECT COUNT(*) FROM users u JOIN users v ON u.id = v.id",
    ] {
        assert_stream_eq_execute(sql, &engine);
    }
}

#[test]
fn float_to_int_cast_rounds_and_rejects_out_of_range() {
    // FLOAT→INT must round and ERROR on non-finite / out-of-range, not silently saturate
    // (NaN→0, overflow→i64::MAX) as a successful wrong value.
    let engine = MockEngine::new();
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;
    // Exponent literals are FLOAT (a bare `2.6` is exact NUMERIC); round half-away-zero.
    assert_eq!(one("SELECT CAST(26e-1 AS INT)"), vec![vec![Value::Int(3)]]);
    assert_eq!(
        one("SELECT CAST(-26e-1 AS INT)"),
        vec![vec![Value::Int(-3)]]
    );
    assert_eq!(one("SELECT CAST(24e-1 AS INT)"), vec![vec![Value::Int(2)]]);
    // 1e30 is far past i64::MAX → an honest error, not a saturated i64::MAX.
    assert!(
        matches!(
            run("SELECT CAST(1e30 AS INT)", &engine),
            Err(Error::InvalidValue { .. })
        ),
        "an out-of-range float→int cast must error"
    );
}

#[test]
fn alter_add_column_backfills_null_and_rows_still_decode() {
    let engine = seeded_users();
    altered(run("ALTER TABLE users ADD COLUMN tag TEXT", &engine).unwrap());
    // Every pre-existing row decodes under the widened schema with a NULL tag.
    let (cols, rows) = rows_of(run("SELECT id, tag FROM users", &engine).unwrap());
    assert_eq!(cols, ["id", "tag"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(1), Value::Null],
            vec![Value::Int(2), Value::Null],
        ],
    );
    // A new row can populate the added column.
    run(
        "INSERT INTO users VALUES (3, 'carol', 40, 2.0, TRUE, 'vip')",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT tag FROM users WHERE id = 3", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Text("vip".to_owned())]]);
}

#[test]
fn string_literal_date_filter_returns_correct_rows_end_to_end() {
    // Full pipeline (analyze → plan → execute) for the ORM date-filter pattern. After the
    // extended-query `Bind` substitutes a `$1` date parameter (sent by the driver as text), the
    // statement is `WHERE d >= '<date>'` — an implicit-cast comparison. Assert it not only
    // type-checks but returns exactly the right rows, and that a range/IN bound works too.
    let engine = MockEngine::new();
    run("CREATE TABLE events (id INT NOT NULL, d DATE)", &engine).unwrap();
    for (id, d) in [(1, "2026-01-01"), (2, "2026-06-15"), (3, "2026-12-31")] {
        run(&format!("INSERT INTO events VALUES ({id}, '{d}')"), &engine).unwrap();
    }
    let ids = |sql: &str| -> Vec<Value> {
        rows_of(run(sql, &engine).unwrap())
            .1
            .into_iter()
            .map(|r| r[0].clone())
            .collect()
    };
    assert_eq!(
        ids("SELECT id FROM events WHERE d >= '2026-06-15' ORDER BY id"),
        vec![Value::Int(2), Value::Int(3)],
    );
    assert_eq!(
        ids("SELECT id FROM events WHERE d = '2026-01-01'"),
        vec![Value::Int(1)],
    );
    assert_eq!(
        ids("SELECT id FROM events WHERE d BETWEEN '2026-01-01' AND '2026-06-30' ORDER BY id"),
        vec![Value::Int(1), Value::Int(2)],
    );
    assert_eq!(
        ids("SELECT id FROM events WHERE d IN ('2026-01-01', '2026-12-31') ORDER BY id"),
        vec![Value::Int(1), Value::Int(3)],
    );
    // An unparseable date string still loud-rejects at evaluation — never a silent wrong row.
    assert!(run("SELECT id FROM events WHERE d >= 'not-a-date'", &engine).is_err());
}

#[test]
fn simple_case_with_untyped_null_operand_evaluates() {
    // A simple CASE whose operand is a bare untyped NULL (`CASE NULL WHEN … END`) must type the
    // operand from the WHEN values rather than raising "cannot infer the type of NULL". `NULL = <any>`
    // is NULL (never matches), so such a CASE falls through to ELSE — matching the reference engine.
    let engine = MockEngine::new();
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;
    // Operand and WHEN both untyped NULL: `NULL = NULL` is NULL, so no branch matches → ELSE 2.
    assert_eq!(
        one("SELECT CASE NULL WHEN NULL THEN 1 ELSE 2 END"),
        vec![vec![Value::Int(2)]]
    );
    // Operand NULL typed from a concrete WHEN (int, then text); NULL never equals it → ELSE.
    assert_eq!(
        one("SELECT CASE NULL WHEN 5 THEN 1 ELSE 2 END"),
        vec![vec![Value::Int(2)]]
    );
    assert_eq!(
        one("SELECT CASE NULL WHEN 'a' THEN 1 ELSE 2 END"),
        vec![vec![Value::Int(2)]]
    );
    // No ELSE → the fall-through result is NULL, typed from the THEN branch (INT).
    assert_eq!(
        one("SELECT CASE NULL WHEN NULL THEN 1 END"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn any_with_text_array_literal_filters_end_to_end() {
    // A bound array parameter `WHERE id = ANY($1)` becomes `id = ANY('{...}')` once the driver's
    // `{...}` text form is substituted. The array operand must coerce to an array of the probe's type
    // and match — exactly like the explicit `$1::int[]` form — returning the right rows.
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, name TEXT)", &engine).unwrap();
    for (id, name) in [(1, "a"), (2, "b"), (3, "c"), (4, "d")] {
        run(&format!("INSERT INTO t VALUES ({id}, '{name}')"), &engine).unwrap();
    }
    let ids = |sql: &str| -> Vec<Value> {
        let mut r = rows_of(run(sql, &engine).unwrap())
            .1
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>();
        r.sort_by_key(|v| format!("{v:?}"));
        r
    };
    // int[] probe: only ids in the array match.
    assert_eq!(
        ids("SELECT id FROM t WHERE id = ANY('{1,3}')"),
        vec![Value::Int(1), Value::Int(3)]
    );
    // text[] probe against the name column.
    assert_eq!(
        ids("SELECT id FROM t WHERE name = ANY('{b,d}')"),
        vec![Value::Int(2), Value::Int(4)]
    );
    // ALL quantifier over a text array literal still type-checks and evaluates.
    assert_eq!(
        ids("SELECT id FROM t WHERE id <> ALL('{2,4}')"),
        vec![Value::Int(1), Value::Int(3)]
    );
    // An unparseable array literal loud-rejects at evaluation — never a silent wrong row.
    assert!(run("SELECT id FROM t WHERE id = ANY('{not,ints}')", &engine).is_err());
}

#[test]
fn window_function_over_group_by_ranks_aggregated_rows() {
    // A window function runs over the post-aggregation rows: its ORDER BY / PARTITION BY may name a
    // grouping aggregate (`rank() OVER (ORDER BY sum(x))`) or a group key, exactly the OLAP
    // "rank-groups-by-aggregate" shape. Previously rejected as "window functions together with
    // GROUP BY".
    let engine = MockEngine::new();
    run("CREATE TABLE emp (dept TEXT, sal INT)", &engine).unwrap();
    for (d, s) in [("a", 100), ("a", 200), ("b", 50), ("b", 50), ("c", 300)] {
        run(&format!("INSERT INTO emp VALUES ('{d}', {s})"), &engine).unwrap();
    }
    // rank() OVER (ORDER BY sum(sal) DESC): sums a=300, b=100, c=300 → 300s tie at rank 1, 100 at 3.
    assert_eq!(
        rows_of(
            run(
                "SELECT dept, sum(sal), rank() OVER (ORDER BY sum(sal) DESC) \
                 FROM emp GROUP BY dept ORDER BY dept",
                &engine
            )
            .unwrap()
        )
        .1,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(300), Value::Int(1)],
            vec![Value::Text("b".to_owned()), Value::Int(100), Value::Int(3)],
            vec![Value::Text("c".to_owned()), Value::Int(300), Value::Int(1)],
        ]
    );
    // Window ORDER BY a group key.
    assert_eq!(
        rows_of(
            run(
                "SELECT dept, rank() OVER (ORDER BY dept) FROM emp GROUP BY dept ORDER BY dept",
                &engine
            )
            .unwrap()
        )
        .1,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(1)],
            vec![Value::Text("b".to_owned()), Value::Int(2)],
            vec![Value::Text("c".to_owned()), Value::Int(3)],
        ]
    );
    // A scalar aggregate (no GROUP BY) is one group: the window runs over that single row.
    assert_eq!(
        rows_of(
            run(
                "SELECT row_number() OVER (ORDER BY count(*)) FROM emp",
                &engine
            )
            .unwrap()
        )
        .1,
        vec![vec![Value::Int(1)]]
    );
    // PARTITION BY a grouping aggregate: {a,b} share count 2, {c} is count 1.
    assert_eq!(
        rows_of(
            run(
                "SELECT dept, row_number() OVER (PARTITION BY count(*) ORDER BY dept) \
                 FROM emp GROUP BY dept ORDER BY dept",
                &engine
            )
            .unwrap()
        )
        .1,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(1)],
            vec![Value::Text("b".to_owned()), Value::Int(2)],
            vec![Value::Text("c".to_owned()), Value::Int(1)],
        ]
    );
    // A projected bare column that is neither grouped nor aggregated is still rejected.
    assert!(
        run(
            "SELECT sal, rank() OVER (ORDER BY count(*)) FROM emp GROUP BY dept",
            &engine
        )
        .is_err()
    );
}

#[test]
fn jsonb_agg_collects_values_into_a_json_array() {
    // JSONB_AGG collects every input value into a JSON array in input order, a NULL becoming JSON
    // `null` (unlike the NULL-skipping numeric aggregates); an empty group yields NULL; and an
    // in-aggregate ORDER BY reorders the elements before serialization.
    let engine = MockEngine::new();
    run("CREATE TABLE t (g INT, v INT)", &engine).unwrap();
    for (g, v) in [(1, "10"), (1, "20"), (1, "NULL"), (2, "30")] {
        run(&format!("INSERT INTO t VALUES ({g}, {v})"), &engine).unwrap();
    }
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;
    // Input order, NULL kept as JSON null.
    assert_eq!(
        one("SELECT jsonb_agg(v) FROM t WHERE g = 1"),
        vec![vec![Value::Json("[10,20,null]".to_owned())]]
    );
    // ORDER BY v DESC: NULLs sort first (the DESC default), then 20, 10.
    assert_eq!(
        one("SELECT jsonb_agg(v ORDER BY v DESC) FROM t WHERE g = 1"),
        vec![vec![Value::Json("[null,20,10]".to_owned())]]
    );
    // An empty group is NULL, not an empty array.
    assert_eq!(
        one("SELECT jsonb_agg(v) FROM t WHERE g = 99"),
        vec![vec![Value::Null]]
    );
    // Per group, and the `json_agg` spelling is the same aggregate.
    assert_eq!(
        one("SELECT json_agg(v) FROM t WHERE g = 2"),
        vec![vec![Value::Json("[30]".to_owned())]]
    );
}

#[test]
fn jsonb_functions_accept_bare_string_literals() {
    // A bare string literal for a JSON (or `text[]` path) argument is the unknown-literal form an
    // ad-hoc query and many drivers use: `jsonb_object_keys('{...}')`, `jsonb_set('{...}', '{a}',
    // '9')`. It must type-check like the explicit `'...'::json` / `'{a}'::text[]` cast, not raise
    // `TypeMismatch: expected Json, found Text`.
    let engine = MockEngine::new();
    let rows = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;

    // jsonb_object_keys (set-returning) — one row per top-level key.
    assert_eq!(
        rows(r#"SELECT jsonb_object_keys('{"a":1,"b":2}')"#),
        vec![
            vec![Value::Text("a".to_owned())],
            vec![Value::Text("b".to_owned())],
        ]
    );
    // jsonb_set (scalar, text path + text new-value literals) — mutate the value at the path.
    assert_eq!(
        rows(r#"SELECT jsonb_set('{"a":1,"b":2}', '{a}', '9')"#),
        vec![vec![Value::Json(r#"{"a":9,"b":2}"#.to_owned())]]
    );
    // The optional 4th (create-missing) argument still binds.
    assert_eq!(
        rows(r#"SELECT jsonb_set('{"a":1}', '{c}', '3', true)"#),
        vec![vec![Value::Json(r#"{"a":1,"c":3}"#.to_owned())]]
    );
    // An unparseable JSON literal loud-rejects at evaluation — never a silent wrong row.
    assert!(run("SELECT jsonb_object_keys('not json')", &engine).is_err());
}

#[test]
fn percentile_within_group_array_of_fractions_returns_an_array() {
    // `PERCENTILE_CONT(ARRAY[f1, f2, ...]) WITHIN GROUP (ORDER BY x)` returns one percentile per
    // fraction, as an array — the multi-quantile form a dashboard asks for in a single scan. It
    // desugars to one scalar percentile aggregate per fraction, collected into an array.
    let engine = MockEngine::new();
    run("CREATE TABLE t (v INT)", &engine).unwrap();
    for v in 1..=5 {
        run(&format!("INSERT INTO t VALUES ({v})"), &engine).unwrap();
    }
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;
    // Continuous interpolation over 1..5: q0=1, q0.25=2, q0.5=3, q1=5 — all FLOAT.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(ARRAY[0, 0.25, 0.5, 1]) WITHIN GROUP (ORDER BY v) FROM t"),
        vec![vec![Value::Array(vec![
            Value::Float(1.0),
            Value::Float(2.0),
            Value::Float(3.0),
            Value::Float(5.0),
        ])]]
    );
    // Discrete percentile returns actual data points (INT elements): d0=1, d0.5=3, d1=5.
    assert_eq!(
        one("SELECT PERCENTILE_DISC(ARRAY[0, 0.5, 1]) WITHIN GROUP (ORDER BY v) FROM t"),
        vec![vec![Value::Array(vec![
            Value::Int(1),
            Value::Int(3),
            Value::Int(5),
        ])]]
    );
    // The scalar form is unaffected — still a bare FLOAT, not a one-element array.
    assert_eq!(
        one("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v) FROM t"),
        vec![vec![Value::Float(3.0)]]
    );
    // A fraction outside [0, 1] anywhere in the array still loud-rejects.
    assert!(
        run(
            "SELECT PERCENTILE_CONT(ARRAY[0.5, 2.0]) WITHIN GROUP (ORDER BY v) FROM t",
            &engine
        )
        .is_err()
    );
}

#[test]
fn cte_is_visible_inside_subqueries_and_set_operation_branches() {
    // A `WITH` CTE scopes over the whole statement, not just the top-level FROM: a subquery (scalar /
    // IN / EXISTS) and every branch of a UNION/INTERSECT/EXCEPT can reference it. Previously each of
    // these raised "table a not found" (subqueries) or was rejected outright (set operations).
    let engine = MockEngine::new();
    run("CREATE TABLE t (x INT)", &engine).unwrap();
    run("INSERT INTO t VALUES (1), (2), (3)", &engine).unwrap();
    let ids = |sql: &str| -> Vec<Value> {
        rows_of(run(sql, &engine).unwrap())
            .1
            .into_iter()
            .map(|r| r[0].clone())
            .collect()
    };
    // Scalar subquery references the CTE.
    assert_eq!(
        ids("WITH a AS (SELECT x FROM t WHERE x = 1) SELECT (SELECT x FROM a)"),
        vec![Value::Int(1)]
    );
    // IN (subquery) references the CTE.
    assert_eq!(
        ids(
            "WITH a AS (SELECT x FROM t WHERE x < 3) SELECT x FROM t WHERE x IN (SELECT x FROM a) ORDER BY x"
        ),
        vec![Value::Int(1), Value::Int(2)]
    );
    // EXISTS (correlated subquery) references the CTE.
    assert_eq!(
        ids(
            "WITH a AS (SELECT x FROM t WHERE x = 2) SELECT x FROM t WHERE EXISTS (SELECT 1 FROM a WHERE a.x = t.x)"
        ),
        vec![Value::Int(2)]
    );
    // The CTE scopes over a set-operation branch (here the right branch of a UNION ALL).
    assert_eq!(
        ids(
            "WITH a AS (SELECT x FROM t WHERE x < 3) SELECT x FROM t WHERE x = 3 UNION ALL SELECT x FROM a ORDER BY x"
        ),
        vec![Value::Int(1), Value::Int(2), Value::Int(3)]
    );
    // The CTE is visible to both branches; UNION (distinct) dedups the two identical branches.
    assert_eq!(
        ids("WITH a AS (SELECT x FROM t WHERE x = 2) SELECT x FROM a UNION SELECT x FROM a"),
        vec![Value::Int(2)]
    );
}

#[test]
fn alter_add_not_null_column_on_nonempty_table_is_rejected() {
    let engine = seeded_users();
    assert!(matches!(
        run("ALTER TABLE users ADD COLUMN flag BOOL NOT NULL", &engine),
        Err(Error::NotNullViolation { .. }),
    ));
}

#[test]
fn alter_add_not_null_column_on_empty_table_is_ok() {
    let engine = setup();
    altered(run("ALTER TABLE users ADD COLUMN flag BOOL NOT NULL", &engine).unwrap());
}

#[test]
fn alter_drop_column_rewrites_rows() {
    let engine = seeded_users();
    altered(run("ALTER TABLE users DROP COLUMN age", &engine).unwrap());
    // The dropped column is gone and the remaining values stay aligned.
    let (cols, rows) = rows_of(run("SELECT id, name, score FROM users", &engine).unwrap());
    assert_eq!(cols, ["id", "name", "score"]);
    assert_eq!(
        rows,
        vec![
            vec![
                Value::Int(1),
                Value::Text("alice".to_owned()),
                Value::Float(9.5)
            ],
            vec![
                Value::Int(2),
                Value::Text("bob".to_owned()),
                Value::Float(1.0)
            ],
        ],
    );
    // `age` no longer resolves.
    assert!(matches!(
        run("SELECT age FROM users", &engine),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn alter_column_type_casts_stored_values() {
    let engine = seeded_users();
    altered(
        run(
            "ALTER TABLE users ALTER COLUMN score SET DATA TYPE TEXT",
            &engine,
        )
        .unwrap(),
    );
    let (_, rows) = rows_of(run("SELECT score FROM users WHERE id = 1", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Text("9.5".to_owned())]]);
}

#[test]
fn alter_set_not_null_rejects_existing_nulls() {
    let engine = seeded_users();
    // Row 2 has a NULL age, so the validating scan must fail.
    assert!(matches!(
        run("ALTER TABLE users ALTER COLUMN age SET NOT NULL", &engine),
        Err(Error::NotNullViolation { .. }),
    ));
}

#[test]
fn alter_rename_column_is_catalog_only() {
    let engine = seeded_users();
    altered(run("ALTER TABLE users RENAME COLUMN name TO full_name", &engine).unwrap());
    let (cols, _) = rows_of(run("SELECT id, full_name FROM users", &engine).unwrap());
    assert_eq!(cols, ["id", "full_name"]);
    assert!(matches!(
        run("SELECT name FROM users", &engine),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn alter_missing_table_if_exists_is_noop() {
    let engine = setup();
    altered(run("ALTER TABLE IF EXISTS ghost ADD COLUMN x INT", &engine).unwrap());
}

// --- ANALYZE -------------------------------------------------------

#[test]
fn analyze_persists_table_and_column_stats() {
    let engine = setup();
    for (id, age) in [(1, 30), (2, 30), (3, 25)] {
        run(
            &format!("INSERT INTO users VALUES ({id}, 'n', {age}, 1.0, TRUE)"),
            &engine,
        )
        .unwrap();
    }
    match run("ANALYZE TABLE users", &engine).unwrap() {
        ExecutionResult::Analyzed { table, columns } => {
            assert_eq!(table, "users");
            assert_eq!(columns, 5);
        },
        other => panic!("expected Analyzed, got {other:?}"),
    }
    let id = StorageEngine::lookup_table(&engine, "users")
        .unwrap()
        .unwrap()
        .id;
    let stats = StorageEngine::table_stats(&engine, id).unwrap().unwrap();
    assert_eq!(stats.row_count, 3);
    assert_eq!(stats.columns.len(), 5);
    // `age` column: 2 distinct values, no NULLs.
    let age = stats.columns.iter().find(|c| c.column == "age").unwrap();
    assert_eq!(age.distinct_count, 2);
    assert_eq!(age.null_count, 0);
}

#[test]
fn analyze_specific_columns_only() {
    let engine = setup();
    run("INSERT INTO users VALUES (1, 'a', 1, 1.0, TRUE)", &engine).unwrap();
    match run("ANALYZE TABLE users FOR COLUMNS id, age", &engine).unwrap() {
        ExecutionResult::Analyzed { columns, .. } => assert_eq!(columns, 2),
        other => panic!("expected Analyzed, got {other:?}"),
    }
    let id = StorageEngine::lookup_table(&engine, "users")
        .unwrap()
        .unwrap()
        .id;
    let stats = StorageEngine::table_stats(&engine, id).unwrap().unwrap();
    assert_eq!(stats.columns.len(), 2);
    assert!(
        stats
            .columns
            .iter()
            .all(|c| c.column == "id" || c.column == "age")
    );
}

#[test]
fn analyze_missing_table_is_rejected() {
    let engine = setup();
    assert!(matches!(
        run("ANALYZE TABLE ghost", &engine),
        Err(Error::TableNotFound { .. }),
    ));
}

/// The limit-aware top-N pass is **result-identical** to the full sort's
/// first rows — same rows, same order, including tie stability, DESC, and NULL placement. Proven
/// differentially: `ORDER BY … LIMIT k` (top-N path) equals `ORDER BY …` (full sort) truncated to
/// `k`, over data with duplicate keys and NULLs.
#[test]
fn top_n_sort_is_result_identical_to_full_sort() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, grp INT)", &engine).unwrap();
    // 60 rows: `grp` has heavy duplicates (id % 7) plus some NULLs, so ties and null-ordering are
    // exercised; `id` is unique so the tie-break is observable through the projected `id`.
    for id in 0..60 {
        let grp = if id % 11 == 0 {
            "NULL".to_owned()
        } else {
            (id % 7).to_string()
        };
        run(&format!("INSERT INTO t VALUES ({id}, {grp})"), &engine).unwrap();
    }

    // For each ORDER BY shape, the top-N result must equal the full-sort result truncated to k.
    for order in [
        "grp",
        "grp DESC",
        "grp NULLS FIRST",
        "grp DESC NULLS LAST",
        "grp, id DESC",
    ] {
        let full =
            rows_of(run(&format!("SELECT id, grp FROM t ORDER BY {order}"), &engine).unwrap()).1;
        for k in [0_usize, 1, 5, 13, 60, 100] {
            let topn = rows_of(
                run(
                    &format!("SELECT id, grp FROM t ORDER BY {order} LIMIT {k}"),
                    &engine,
                )
                .unwrap(),
            )
            .1;
            let expected: Vec<Row> = full.iter().take(k).cloned().collect();
            assert_eq!(
                topn, expected,
                "top-{k} of `ORDER BY {order}` must equal the full sort's first {k} rows"
            );
        }
    }

    // The finding's exact repro plans (and EXPLAINs) as a bounded top-N, not a full sort.
    let plan = explain_lines("EXPLAIN SELECT id FROM t ORDER BY id LIMIT 5", &engine).join("\n");
    assert!(
        plan.contains("Sort (1 key(s)) top-5"),
        "ORDER BY id LIMIT 5 must plan a top-5 sort, got:\n{plan}"
    );
}

/// The limit-aware ranking window (computing over only the first `m`
/// rows) is **result-identical** to the full materializing computation's first `m` rows — same
/// rows, same order, same ranking values — including ties in the window order. Proven
/// differentially: `... OVER (ORDER BY grp) ... ORDER BY grp LIMIT k` (top-N path) equals the same
/// query without the LIMIT (full path) truncated to `k`, for `ROW_NUMBER` / `RANK` / `DENSE_RANK`.
#[test]
fn top_n_ranking_window_is_result_identical_to_full() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, grp INT)", &engine).unwrap();
    // Heavy ties in `grp` (id % 6) so the ranking's peer handling and the tie-break are exercised.
    for id in 0..50 {
        run(&format!("INSERT INTO t VALUES ({id}, {})", id % 6), &engine).unwrap();
    }
    for func in ["row_number()", "rank()", "dense_rank()"] {
        let full = rows_of(
            run(
                &format!("SELECT id, {func} OVER (ORDER BY grp) FROM t ORDER BY grp"),
                &engine,
            )
            .unwrap(),
        )
        .1;
        for k in [0_usize, 1, 5, 13, 50, 80] {
            let topn = rows_of(
                run(
                    &format!("SELECT id, {func} OVER (ORDER BY grp) FROM t ORDER BY grp LIMIT {k}"),
                    &engine,
                )
                .unwrap(),
            )
            .1;
            let expected: Vec<Row> = full.iter().take(k).cloned().collect();
            assert_eq!(
                topn, expected,
                "top-{k} ranking window `{func}` must equal the full computation's first {k} rows"
            );
        }
    }

    // The plan shows the bounded window (EXPLAIN diagnostic QA uses).
    let plan = explain_lines(
        "EXPLAIN SELECT id, row_number() OVER (ORDER BY grp) FROM t ORDER BY grp LIMIT 3",
        &engine,
    )
    .join("\n");
    assert!(
        plan.contains("Window (1 function(s)) top-3"),
        "the ranking window must plan a top-3, got:\n{plan}"
    );
}

#[test]
fn analyze_is_deterministic() {
    let engine = setup();
    for i in 0..20 {
        run(
            &format!("INSERT INTO users VALUES ({i}, 'n', {}, 1.0, TRUE)", i % 4),
            &engine,
        )
        .unwrap();
    }
    let id = StorageEngine::lookup_table(&engine, "users")
        .unwrap()
        .unwrap()
        .id;
    run("ANALYZE TABLE users", &engine).unwrap();
    let first = StorageEngine::table_stats(&engine, id).unwrap().unwrap();
    run("ANALYZE TABLE users", &engine).unwrap();
    let second = StorageEngine::table_stats(&engine, id).unwrap().unwrap();
    assert_eq!(first, second);
}

// --- EXPLAIN cost annotation ----------------------------

fn explain_lines(sql: &str, engine: &MockEngine) -> Vec<String> {
    let (_, rows) = rows_of(run(sql, engine).unwrap());
    rows.into_iter()
        .map(|r| match r.into_iter().next() {
            Some(Value::Text(s)) => s,
            other => panic!("expected text plan line, got {other:?}"),
        })
        .collect()
}

#[test]
fn explain_without_analyze_has_no_estimates() {
    let engine = setup();
    run("INSERT INTO users VALUES (1, 'a', 1, 1.0, TRUE)", &engine).unwrap();
    let lines = explain_lines("EXPLAIN SELECT * FROM users WHERE id = 1", &engine);
    assert!(
        lines.iter().all(|l| !l.contains("est. rows")),
        "unexpected estimate before ANALYZE: {lines:?}",
    );
}

#[test]
fn explain_after_analyze_annotates_estimated_rows() {
    let engine = setup();
    for i in 0..8 {
        run(
            &format!("INSERT INTO users VALUES ({i}, 'n', {i}, 1.0, TRUE)"),
            &engine,
        )
        .unwrap();
    }
    run("ANALYZE TABLE users", &engine).unwrap();
    let lines = explain_lines("EXPLAIN SELECT * FROM users WHERE id = 3", &engine);
    // SeqScan sees all 8 rows; the equality filter on a unique column keeps ~1.
    let seqscan = lines.iter().find(|l| l.contains("SeqScan")).unwrap();
    assert!(seqscan.contains("est. rows=8"), "seqscan: {seqscan}");
    let filter = lines.iter().find(|l| l.contains("Filter")).unwrap();
    assert!(filter.contains("est. rows=1"), "filter: {filter}");
}

// --- Column encryption ----------------------------------

#[test]
fn encrypt_decrypt_round_trips_through_storage() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, secret TEXT)", &engine).unwrap();
    run("INSERT INTO t VALUES (1, encrypt('ssn-123', 'k'))", &engine).unwrap();

    // At rest the value is opaque ciphertext, not the plaintext.
    let (_, stored) = rows_of(run("SELECT secret FROM t", &engine).unwrap());
    let Value::Text(ct) = stored.first().and_then(|r| r.first()).unwrap() else {
        panic!("expected text ciphertext");
    };
    assert_ne!(ct, "ssn-123");

    // decrypt(...) with the right key recovers it.
    let (_, plain) = rows_of(run("SELECT decrypt(secret, 'k') FROM t", &engine).unwrap());
    assert_eq!(plain, vec![vec![Value::Text("ssn-123".to_owned())]]);
}

#[test]
fn decrypt_with_wrong_key_errors() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, secret TEXT)", &engine).unwrap();
    run(
        "INSERT INTO t VALUES (1, encrypt('secret', 'right'))",
        &engine,
    )
    .unwrap();
    assert!(matches!(
        run("SELECT decrypt(secret, 'wrong') FROM t", &engine),
        Err(Error::Decryption(_)),
    ));
}

#[test]
fn encrypt_is_deterministic_across_rows() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, secret TEXT)", &engine).unwrap();
    run("INSERT INTO t VALUES (1, encrypt('x', 'k'))", &engine).unwrap();
    run("INSERT INTO t VALUES (2, encrypt('x', 'k'))", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT secret FROM t ORDER BY id", &engine).unwrap());
    assert_eq!(
        rows[0], rows[1],
        "same key+plaintext must encrypt identically"
    );
}

#[test]
fn encrypt_of_null_is_null() {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, secret TEXT)", &engine).unwrap();
    run("INSERT INTO t VALUES (1, NULL)", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT encrypt(secret, 'k') FROM t", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Null]]);
}

// --- INSERT --------------------------------------------------------

#[test]
fn insert_then_select_roundtrip() {
    let engine = setup();
    assert_eq!(
        inserted_count(
            run(
                "INSERT INTO users VALUES (1, 'alice', 30, 9.5, TRUE)",
                &engine,
            )
            .unwrap(),
        ),
        1,
    );
    let (cols, rows) = rows_of(run("SELECT * FROM users", &engine).unwrap());
    assert_eq!(cols, ["id", "name", "age", "score", "active"]);
    assert_eq!(
        rows,
        vec![vec![
            Value::Int(1),
            Value::Text("alice".to_owned()),
            Value::Int(30),
            Value::Float(9.5),
            Value::Bool(true),
        ]],
    );
}

#[test]
fn insert_multiple_rows() {
    let engine = setup();
    run(
        "INSERT INTO users (id, name) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users", &engine).unwrap());
    assert_eq!(rows.len(), 3);
}

#[test]
fn insert_null_into_nullable_column() {
    let engine = setup();
    run("INSERT INTO users (id, name) VALUES (1, NULL)", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT name FROM users", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn insert_missing_not_null_column_is_rejected() {
    let engine = setup();
    // `id` is NOT NULL but the column list omits it.
    let result = run("INSERT INTO users (name) VALUES ('alice')", &engine);
    assert!(matches!(result, Err(Error::NotNullViolation { .. })));
}

// --- SELECT --------------------------------------------------------

#[test]
fn select_where_filters_rows() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 10), (2, 30), (3, 50)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users WHERE age > 20", &engine).unwrap());
    let ids: Vec<&Value> = rows.iter().map(|r| &r[0]).collect();
    assert_eq!(ids, vec![&Value::Int(2), &Value::Int(3)]);
}

#[test]
fn select_order_by_sorts_rows() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 30), (2, 10), (3, 20)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users ORDER BY age", &engine).unwrap());
    let ids: Vec<&Value> = rows.iter().map(|r| &r[0]).collect();
    assert_eq!(ids, vec![&Value::Int(2), &Value::Int(3), &Value::Int(1)]);
}

#[test]
fn select_order_by_desc_sorts_descending() {
    let engine = setup();
    run("INSERT INTO users (id) VALUES (1), (2), (3)", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users ORDER BY id DESC", &engine).unwrap());
    let ids: Vec<&Value> = rows.iter().map(|r| &r[0]).collect();
    assert_eq!(ids, vec![&Value::Int(3), &Value::Int(2), &Value::Int(1)]);
}

#[test]
fn select_limit_caps_row_count() {
    let engine = setup();
    run(
        "INSERT INTO users (id) VALUES (1), (2), (3), (4), (5)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users LIMIT 2", &engine).unwrap());
    assert_eq!(rows.len(), 2);
}

#[test]
fn select_computed_projection() {
    let engine = setup();
    run("INSERT INTO users (id, age) VALUES (1, 30)", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT age + 1 FROM users", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(31)]]);
}

#[test]
fn select_is_null_handling() {
    let engine = setup();
    run(
        "INSERT INTO users (id, name) VALUES (1, NULL), (2, 'x')",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users WHERE name IS NULL", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(1)]]);
}

#[test]
fn select_without_from_returns_literal() {
    let engine = MockEngine::new();
    let (cols, rows) = rows_of(run("SELECT 1", &engine).unwrap());
    assert_eq!(cols.len(), 1);
    assert_eq!(rows, vec![vec![Value::Int(1)]]);
}

#[test]
fn select_null_sorts_last_in_ascending() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, NULL), (2, 10), (3, NULL), (4, 20)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users ORDER BY age", &engine).unwrap());
    // age order: 10, 20, NULL, NULL → ids 2, 4, then (1, 3) in either order.
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| if let Value::Int(i) = &r[0] { *i } else { 0 })
        .collect();
    assert_eq!(&ids[..2], &[2, 4]);
    assert!(ids[2..].contains(&1) && ids[2..].contains(&3));
}

// --- UPDATE / DELETE ----------------------------------------------

#[test]
fn update_with_filter() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 10), (2, 20), (3, 30)",
        &engine,
    )
    .unwrap();
    let count = updated_count(run("UPDATE users SET age = 99 WHERE id = 2", &engine).unwrap());
    assert_eq!(count, 1);
    let (_, rows) = rows_of(run("SELECT age FROM users WHERE id = 2", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(99)]]);
}

#[test]
fn update_no_match_returns_zero() {
    let engine = setup();
    run("INSERT INTO users (id) VALUES (1)", &engine).unwrap();
    let count = updated_count(run("UPDATE users SET age = 1 WHERE id = 999", &engine).unwrap());
    assert_eq!(count, 0);
}

#[test]
fn update_can_reference_columns() {
    let engine = setup();
    run("INSERT INTO users (id, age) VALUES (1, 10)", &engine).unwrap();
    run("UPDATE users SET age = age + 5", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT age FROM users", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(15)]]);
}

#[test]
fn delete_with_filter() {
    let engine = setup();
    run("INSERT INTO users (id) VALUES (1), (2), (3)", &engine).unwrap();
    let count = deleted_count(run("DELETE FROM users WHERE id = 2", &engine).unwrap());
    assert_eq!(count, 1);
    let (_, rows) = rows_of(run("SELECT id FROM users", &engine).unwrap());
    assert_eq!(rows.len(), 2);
}

#[test]
fn delete_all_rows() {
    let engine = setup();
    run("INSERT INTO users (id) VALUES (1), (2)", &engine).unwrap();
    let count = deleted_count(run("DELETE FROM users", &engine).unwrap());
    assert_eq!(count, 2);
    let (_, rows) = rows_of(run("SELECT id FROM users", &engine).unwrap());
    assert!(rows.is_empty());
}

// --- Runtime errors ------------------------------------------------

// --- EXPLAIN -------------------------------------------------------

#[test]
fn explain_select_returns_plan_lines() {
    let engine = setup();
    let (cols, rows) = rows_of(
        run(
            "EXPLAIN SELECT * FROM users WHERE id = 1 ORDER BY id LIMIT 5",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["plan"]);
    let text: Vec<String> = rows
        .iter()
        .map(|r| {
            if let Value::Text(s) = &r[0] {
                s.clone()
            } else {
                panic!("expected Text")
            }
        })
        .collect();
    let joined = text.join("\n");
    assert!(joined.contains("Limit 5"));
    assert!(joined.contains("Project"));
    assert!(joined.contains("Sort"));
    assert!(joined.contains("Filter"));
    assert!(joined.contains("SeqScan: users"));
}

#[test]
fn explain_format_json_emits_a_plan_tree() {
    let engine = setup();
    let (cols, rows) = rows_of(
        run(
            "EXPLAIN FORMAT JSON SELECT * FROM users WHERE id = 1 ORDER BY id LIMIT 5",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["plan"]);
    // A single row holding the whole pretty-printed document.
    assert_eq!(rows.len(), 1);
    let Value::Text(doc) = &rows[0][0] else {
        panic!("expected a JSON text document");
    };
    let parsed: serde_json::Value = serde_json::from_str(doc).expect("valid JSON");
    // The root is the outermost operator; children nest the rest of the pipeline.
    let root = &parsed["plan"];
    assert_eq!(root["node"], "Limit 5");
    // Walk down and confirm the scan sits at the bottom of the `children` chain.
    let mut node = root;
    let mut labels = vec![node["node"].as_str().unwrap_or("").to_owned()];
    while let Some(child) = node["children"].as_array().and_then(|c| c.first()) {
        node = child;
        labels.push(node["node"].as_str().unwrap_or("").to_owned());
    }
    assert!(
        labels.iter().any(|l| l == "SeqScan: users"),
        "scan missing from tree: {labels:?}",
    );
    assert!(labels.iter().any(|l| l.starts_with("Sort")), "{labels:?}");
    assert!(labels.iter().any(|l| l == "Filter"), "{labels:?}");
}

#[test]
fn explain_format_json_verbose_includes_output_columns() {
    let engine = setup();
    let lines = explain_lines("EXPLAIN VERBOSE SELECT id, name FROM users", &engine);
    // Sanity: the text form carries an Output line; the JSON form lifts it to an `output` array.
    assert!(lines.iter().any(|l| l.starts_with("Output:")));

    let (_, rows) = rows_of(
        run(
            "EXPLAIN VERBOSE FORMAT JSON SELECT id, name FROM users",
            &engine,
        )
        .unwrap(),
    );
    let Value::Text(doc) = &rows[0][0] else {
        panic!("expected JSON text");
    };
    let parsed: serde_json::Value = serde_json::from_str(doc).expect("valid JSON");
    let output = parsed["output"].as_array().expect("output array");
    let names: Vec<&str> = output.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, ["id", "name"]);
}

#[test]
fn explain_format_graphviz_is_rejected() {
    let engine = setup();
    assert!(matches!(
        run("EXPLAIN FORMAT GRAPHVIZ SELECT * FROM users", &engine),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn explain_propagates_analyzer_errors() {
    let engine = setup();
    assert!(matches!(
        run("EXPLAIN SELECT * FROM ghost", &engine),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn explain_does_not_execute() {
    let engine = setup();
    // EXPLAIN INSERT should NOT actually insert.
    run("EXPLAIN INSERT INTO users (id) VALUES (99)", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT id FROM users", &engine).unwrap());
    assert!(rows.is_empty());
}

#[test]
fn division_by_zero_aborts_select() {
    let engine = setup();
    run("INSERT INTO users (id, age) VALUES (1, 5)", &engine).unwrap();
    let result = run("SELECT age / 0 FROM users", &engine);
    assert!(matches!(result, Err(Error::DivisionByZero)));
}

// --- Hash join -----------------------------------------------------

/// `cust(id INT NOT NULL, name TEXT)` joined with `ord(uid INT, amount INT)`
/// — `uid` nullable so the NULL-key case is exercisable. Base data has no
/// NULL keys: customers 1,2,3; orders for 1 (×2), 2, and 5 (no customer).
fn setup_join() -> MockEngine {
    let engine = MockEngine::new();
    run("CREATE TABLE cust (id INT NOT NULL, name TEXT)", &engine).unwrap();
    run("CREATE TABLE ord (uid INT, amount INT)", &engine).unwrap();
    run(
        "INSERT INTO cust (id, name) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
        &engine,
    )
    .unwrap();
    run(
        "INSERT INTO ord (uid, amount) VALUES (1, 100), (1, 50), (2, 200), (5, 999)",
        &engine,
    )
    .unwrap();
    engine
}

#[test]
fn hash_inner_join_matches_pairs_in_order() {
    let engine = setup_join();
    let (_, rows) = rows_of(
        run(
            "SELECT cust.name, ord.amount FROM cust JOIN ord ON cust.id = ord.uid",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(100)],
            vec![Value::Text("a".to_owned()), Value::Int(50)],
            vec![Value::Text("b".to_owned()), Value::Int(200)],
        ],
    );
}

#[test]
fn hash_left_join_keeps_unmatched_left_null_padded() {
    let engine = setup_join();
    let (_, rows) = rows_of(
        run(
            "SELECT cust.name, ord.amount FROM cust LEFT JOIN ord ON cust.id = ord.uid",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(100)],
            vec![Value::Text("a".to_owned()), Value::Int(50)],
            vec![Value::Text("b".to_owned()), Value::Int(200)],
            vec![Value::Text("c".to_owned()), Value::Null], // customer 3, no order
        ],
    );
}

#[test]
fn hash_full_join_keeps_both_unmatched_sides() {
    let engine = setup_join();
    let (_, rows) = rows_of(
        run(
            "SELECT cust.name, ord.amount FROM cust FULL JOIN ord ON cust.id = ord.uid",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(100)],
            vec![Value::Text("a".to_owned()), Value::Int(50)],
            vec![Value::Text("b".to_owned()), Value::Int(200)],
            vec![Value::Text("c".to_owned()), Value::Null], // customer 3, no order
            vec![Value::Null, Value::Int(999)],             // order uid 5, no customer
        ],
    );
}

#[test]
fn hash_join_residual_filters_pairs() {
    let engine = setup_join();
    // amount > 60 drops the (1, 50) order from the matches.
    let (_, rows) = rows_of(
        run(
            "SELECT cust.name, ord.amount FROM cust JOIN ord \
                 ON cust.id = ord.uid AND ord.amount > 60",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("a".to_owned()), Value::Int(100)],
            vec![Value::Text("b".to_owned()), Value::Int(200)],
        ],
    );
}

#[test]
fn hash_join_null_key_never_matches() {
    let engine = MockEngine::new();
    run("CREATE TABLE cust (id INT NOT NULL, name TEXT)", &engine).unwrap();
    run("CREATE TABLE ord (uid INT, amount INT)", &engine).unwrap();
    run("INSERT INTO cust (id, name) VALUES (1, 'a')", &engine).unwrap();
    // A NULL uid must not join to anything, even though both are "absent".
    run(
        "INSERT INTO ord (uid, amount) VALUES (1, 100), (NULL, 999)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(
        run(
            "SELECT cust.name, ord.amount FROM cust JOIN ord ON cust.id = ord.uid",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![vec![Value::Text("a".to_owned()), Value::Int(100)]],
    );
}

// --- SELECT DISTINCT ----------------------------------------------

#[test]
fn distinct_removes_duplicates_first_seen_order() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 10), (2, 20), (3, 10), (4, 20)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT DISTINCT age FROM users", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(10)], vec![Value::Int(20)]]);
}

#[test]
fn distinct_treats_null_as_not_distinct_from_null() {
    let engine = setup();
    run(
        "INSERT INTO users (id, name) VALUES (1, NULL), (2, NULL), (3, 'x')",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT DISTINCT name FROM users", &engine).unwrap());
    assert_eq!(
        rows,
        vec![vec![Value::Null], vec![Value::Text("x".to_owned())]],
    );
}

#[test]
fn distinct_over_multiple_columns() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 10), (1, 10), (1, 20)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT DISTINCT id, age FROM users", &engine).unwrap());
    assert_eq!(rows.len(), 2);
}

#[test]
fn distinct_then_limit_caps_deduped_rows() {
    let engine = setup();
    run(
        "INSERT INTO users (id, age) VALUES (1, 10), (2, 10), (3, 20), (4, 30)",
        &engine,
    )
    .unwrap();
    let (_, rows) = rows_of(run("SELECT DISTINCT age FROM users LIMIT 2", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(10)], vec![Value::Int(20)]]);
}

// --- VACUUM --------------------------------------------------------

#[test]
fn vacuum_executes_as_recognized_statement() {
    let engine = MockEngine::new();
    // The SQL pathway is complete; the count is 0 until the storage engine
    // exposes reclamation through the treaty.
    assert!(matches!(
        run("VACUUM", &engine).unwrap(),
        ExecutionResult::Vacuumed(0),
    ));
}

#[test]
fn reindex_executes_as_a_noop() {
    // REINDEX is accepted and runs as a no-op (NusaDB's B-tree indexes are always consistent).
    let engine = MockEngine::new();
    assert!(matches!(
        run("REINDEX TABLE t", &engine).unwrap(),
        ExecutionResult::Reindexed,
    ));
}

#[test]
fn explain_vacuum_formats_plan() {
    let engine = MockEngine::new();
    let (cols, rows) = rows_of(run("EXPLAIN VACUUM", &engine).unwrap());
    assert_eq!(cols, ["plan"]);
    let text: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.first() {
            Some(Value::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(text.join("\n").contains("Vacuum"));
}

#[test]
fn explain_picks_hash_join_for_equi_and_nested_loop_otherwise() {
    let engine = setup_join();
    let plan_text = |sql: &str| -> String {
        let (_, rows) = rows_of(run(sql, &engine).unwrap());
        rows.iter()
            .filter_map(|r| match r.first() {
                Some(Value::Text(s)) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        plan_text("EXPLAIN SELECT cust.name FROM cust JOIN ord ON cust.id = ord.uid")
            .contains("HashJoin"),
    );
    assert!(
        plan_text("EXPLAIN SELECT cust.name FROM cust JOIN ord ON cust.id > ord.uid")
            .contains("NestedLoopJoin"),
    );
}

// --- WITH RECURSIVE ----------------------------------------

#[test]
fn recursive_cte_counts_up_with_union_all() {
    // The canonical counter: base = 1, recursive adds 1 while n < 5 → rows 1..=5.
    let engine = MockEngine::new();
    let (cols, rows) = rows_of(
        run(
            "WITH RECURSIVE nums AS \
                 (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 5) \
                 SELECT n FROM nums ORDER BY n",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["n"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
        ],
    );
}

#[test]
fn recursive_cte_union_distinct_reaches_fixpoint() {
    // UNION (distinct) over a recursion that would otherwise loop: once n reaches its cap the
    // recursive term keeps producing the same row, which UNION drops — so the fixpoint terminates.
    let engine = MockEngine::new();
    let (_, rows) = rows_of(
        run(
            "WITH RECURSIVE nums AS \
                 (SELECT 1 AS n UNION SELECT CASE WHEN n < 3 THEN n + 1 ELSE n END FROM nums) \
                 SELECT n FROM nums ORDER BY n",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)]
        ],
    );
}

#[test]
fn recursive_cte_traverses_a_seeded_hierarchy() {
    // A classic transitive closure: walk an adjacency table from a root down every edge.
    let engine = MockEngine::new();
    run("CREATE TABLE edge (parent INT, child INT)", &engine).unwrap();
    for (p, c) in [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6)] {
        run(&format!("INSERT INTO edge VALUES ({p}, {c})"), &engine).unwrap();
    }
    let (_, rows) = rows_of(
        run(
            "WITH RECURSIVE reach AS \
                 (SELECT child AS node FROM edge WHERE parent = 1 \
                  UNION ALL \
                  SELECT e.child FROM edge AS e JOIN reach ON e.parent = reach.node) \
             SELECT node FROM reach ORDER BY node",
            &engine,
        )
        .unwrap(),
    );
    // From root 1: 2, 3 (direct), then 4, 5 (their children), then 6 (under 4).
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(4)],
            vec![Value::Int(5)],
            vec![Value::Int(6)],
        ],
    );
}

#[test]
fn recursive_cte_explains_as_with_recursive() {
    let engine = MockEngine::new();
    let (_, rows) = rows_of(
        run(
            "EXPLAIN WITH RECURSIVE nums AS \
                 (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 3) \
                 SELECT n FROM nums",
            &engine,
        )
        .unwrap(),
    );
    let text = rows
        .iter()
        .filter_map(|r| match r.first() {
            Some(Value::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("WithRecursive"), "plan was:\n{text}");
    assert!(
        text.contains("RecursiveCte (UNION ALL)"),
        "plan was:\n{text}"
    );
}

// --- Correlated subqueries --------------------------------

fn seeded_dept_emp() -> MockEngine {
    let engine = MockEngine::new();
    run("CREATE TABLE dept (id INT NOT NULL, name TEXT)", &engine).unwrap();
    run(
        "CREATE TABLE emp (id INT NOT NULL, dept INT, salary INT)",
        &engine,
    )
    .unwrap();
    for (id, name) in [(1, "eng"), (2, "sales"), (3, "empty")] {
        run(
            &format!("INSERT INTO dept VALUES ({id}, '{name}')"),
            &engine,
        )
        .unwrap();
    }
    for (id, dept, salary) in [(1, 1, 100), (2, 1, 200), (3, 2, 50)] {
        run(
            &format!("INSERT INTO emp VALUES ({id}, {dept}, {salary})"),
            &engine,
        )
        .unwrap();
    }
    engine
}

#[test]
fn correlated_exists_filters_on_outer_row() {
    let engine = seeded_dept_emp();
    // Departments that have at least one employee (the subquery references the outer `d.id`).
    let (_, rows) = rows_of(
        run(
            "SELECT name FROM dept d \
             WHERE EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.id) \
             ORDER BY name",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("eng".to_owned())],
            vec![Value::Text("sales".to_owned())],
        ],
    );
}

#[test]
fn correlated_not_exists_keeps_unmatched_outer_rows() {
    let engine = seeded_dept_emp();
    let (_, rows) = rows_of(
        run(
            "SELECT name FROM dept d \
             WHERE NOT EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.id)",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows, vec![vec![Value::Text("empty".to_owned())]]);
}

#[test]
fn correlated_scalar_subquery_in_where_compares_per_group_aggregate() {
    let engine = seeded_dept_emp();
    // Employees paid above their own department's average salary.
    let (_, rows) = rows_of(
        run(
            "SELECT id FROM emp e \
             WHERE salary > (SELECT AVG(salary) FROM emp WHERE dept = e.dept) \
             ORDER BY id",
            &engine,
        )
        .unwrap(),
    );
    // dept 1 avg = 150 → emp 2 (200) qualifies, emp 1 (100) does not; dept 2 avg = 50 → none above.
    assert_eq!(rows, vec![vec![Value::Int(2)]]);
}

#[test]
fn correlated_scalar_subquery_in_select_list_counts_per_outer_row() {
    let engine = seeded_dept_emp();
    let (cols, rows) = rows_of(
        run(
            "SELECT name, (SELECT COUNT(*) FROM emp e WHERE e.dept = d.id) AS n \
             FROM dept d ORDER BY name",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["name", "n"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("empty".to_owned()), Value::Int(0)],
            vec![Value::Text("eng".to_owned()), Value::Int(2)],
            vec![Value::Text("sales".to_owned()), Value::Int(1)],
        ],
    );
}

#[test]
fn uncorrelated_subquery_still_resolves_once() {
    // A subquery with no outer reference keeps the resolve-once fast path (regression guard).
    let engine = seeded_dept_emp();
    let (_, rows) = rows_of(
        run(
            "SELECT id FROM emp WHERE salary > (SELECT AVG(salary) FROM emp) ORDER BY id",
            &engine,
        )
        .unwrap(),
    );
    // Overall avg = (100+200+50)/3 ≈ 116.67 → only emp 2 (200) is above.
    assert_eq!(rows, vec![vec![Value::Int(2)]]);
}

// --- Array operations: cardinality + `||` concat -----------

fn seeded_arrays() -> MockEngine {
    let engine = MockEngine::new();
    run("CREATE TABLE t (id INT NOT NULL, tags INT[])", &engine).unwrap();
    run("INSERT INTO t VALUES (1, '{10,20,30}')", &engine).unwrap();
    run("INSERT INTO t VALUES (2, '{}')", &engine).unwrap();
    run("INSERT INTO t VALUES (3, NULL)", &engine).unwrap();
    engine
}

#[test]
fn cardinality_counts_array_elements() {
    let engine = seeded_arrays();
    let (cols, rows) =
        rows_of(run("SELECT id, cardinality(tags) FROM t ORDER BY id", &engine).unwrap());
    assert_eq!(cols, ["id", "?column?"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(1), Value::Int(3)],
            vec![Value::Int(2), Value::Int(0)],
            // A NULL array yields a NULL count.
            vec![Value::Int(3), Value::Null],
        ],
    );
}

#[test]
fn cardinality_rejects_a_non_array_argument() {
    let engine = seeded_arrays();
    assert!(matches!(
        run("SELECT cardinality(id) FROM t", &engine),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn array_concat_merges_arrays_and_elements() {
    let engine = seeded_arrays();
    let int = |n| Value::Int(n);
    let arr = |xs: &[i64]| Value::Array(xs.iter().copied().map(Value::Int).collect());

    // Array || array.
    let (_, rows) = rows_of(run("SELECT tags || tags FROM t WHERE id = 1", &engine).unwrap());
    assert_eq!(rows, vec![vec![arr(&[10, 20, 30, 10, 20, 30])]]);

    // Array || element (append) and element || array (prepend).
    let (_, rows) = rows_of(run("SELECT tags || 40 FROM t WHERE id = 1", &engine).unwrap());
    assert_eq!(rows, vec![vec![arr(&[10, 20, 30, 40])]]);
    let (_, rows) = rows_of(run("SELECT 5 || tags FROM t WHERE id = 1", &engine).unwrap());
    assert_eq!(rows, vec![vec![arr(&[5, 10, 20, 30])]]);

    // Concatenating onto the empty array, and NULL-strictness.
    let (_, rows) = rows_of(
        run(
            "SELECT cardinality(tags || tags) FROM t WHERE id = 2",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows, vec![vec![int(0)]]);
    let (_, rows) = rows_of(run("SELECT tags || tags FROM t WHERE id = 3", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn array_concat_rejects_a_mismatched_element_type() {
    let engine = seeded_arrays();
    assert!(matches!(
        run("SELECT tags || 'x' FROM t WHERE id = 1", &engine),
        Err(Error::TypeMismatch { .. }),
    ));
}

fn seeded_groups() -> MockEngine {
    let engine = MockEngine::new();
    run("CREATE TABLE g (grp INT NOT NULL, v INT)", &engine).unwrap();
    for (grp, v) in [(1, "10"), (1, "20"), (1, "NULL"), (2, "30")] {
        run(&format!("INSERT INTO g VALUES ({grp}, {v})"), &engine).unwrap();
    }
    engine
}

#[test]
fn array_agg_collects_values_per_group_including_nulls() {
    let engine = seeded_groups();
    let (cols, rows) = rows_of(
        run(
            "SELECT grp, array_agg(v) FROM g GROUP BY grp ORDER BY grp",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["grp", "?column?"]);
    // Group 1 keeps input order and the NULL; group 2 has a single value.
    assert_eq!(
        rows,
        vec![
            vec![
                Value::Int(1),
                Value::Array(vec![Value::Int(10), Value::Int(20), Value::Null]),
            ],
            vec![Value::Int(2), Value::Array(vec![Value::Int(30)])],
        ],
    );
}

#[test]
fn array_agg_empty_group_is_null_and_distinct_dedups() {
    let engine = seeded_groups();
    // A scalar ARRAY_AGG over zero rows yields NULL, not an empty array.
    let (_, rows) = rows_of(run("SELECT array_agg(v) FROM g WHERE grp = 99", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Null]]);
    // DISTINCT drops duplicate values; a single NULL is kept.
    let (_, rows) =
        rows_of(run("SELECT array_agg(DISTINCT v) FROM g WHERE grp = 1", &engine).unwrap());
    assert_eq!(
        rows,
        vec![vec![Value::Array(vec![
            Value::Int(10),
            Value::Int(20),
            Value::Null
        ])]],
    );
}

#[test]
fn array_agg_coerces_numeric_to_float_and_rejects_non_arrayable() {
    let engine = MockEngine::new();
    run("CREATE TABLE n (x NUMERIC, j JSON)", &engine).unwrap();
    // NUMERIC is not a supported array element type, so ARRAY_AGG(NUMERIC) coerces to FLOAT[] —
    // the same NUMERIC→FLOAT coercion assignment allows — rather than silently a NUMERIC[] or, as
    // beforeerroring on a plain decimal literal. (Exact NUMERIC[] is a separate feature.)
    assert!(run("SELECT array_agg(x) FROM n", &engine).is_ok());
    // A genuinely non-arrayable element type (JSON) is still rejected.
    assert!(matches!(
        run("SELECT array_agg(j) FROM n", &engine),
        Err(Error::Unsupported(_)),
    ));
}

// --- UNNEST set-returning function -------------------------

#[test]
fn unnest_expands_an_array_into_rows() {
    let engine = seeded_arrays();
    // One output row per element, with the scalar column repeated; empty/NULL arrays emit no rows.
    let (cols, rows) = rows_of(run("SELECT id, unnest(tags) FROM t", &engine).unwrap());
    assert_eq!(cols, ["id", "unnest"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(1), Value::Int(20)],
            vec![Value::Int(1), Value::Int(30)],
        ],
    );
}

#[test]
fn unnest_alone_and_composed_with_concat() {
    let engine = seeded_arrays();
    // Bare UNNEST yields a single "unnest" column.
    let (cols, rows) = rows_of(run("SELECT unnest(tags) FROM t WHERE id = 1", &engine).unwrap());
    assert_eq!(cols, ["unnest"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(10)],
            vec![Value::Int(20)],
            vec![Value::Int(30)]
        ],
    );
    // UNNEST over a concatenated array expands all six elements.
    let (_, rows) =
        rows_of(run("SELECT unnest(tags || tags) FROM t WHERE id = 1", &engine).unwrap());
    assert_eq!(rows.len(), 6);
}

#[test]
fn unnest_is_rejected_outside_the_select_list() {
    let engine = seeded_arrays();
    // In a WHERE predicate (not the SELECT list).
    assert!(matches!(
        run("SELECT id FROM t WHERE unnest(tags) > 5", &engine),
        Err(Error::Unsupported(_)),
    ));
    // Nested inside another expression in the SELECT list.
    assert!(matches!(
        run("SELECT unnest(tags) + 1 FROM t", &engine),
        Err(Error::Unsupported(_)),
    ));
    // Combined with aggregation.
    assert!(matches!(
        run("SELECT count(*), unnest(tags) FROM t", &engine),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn unnest_rejects_a_non_array_argument() {
    let engine = seeded_arrays();
    assert!(matches!(
        run("SELECT unnest(id) FROM t", &engine),
        Err(Error::Unsupported(_)),
    ));
}

// --- JSON set-returning functions (c) -----------------------

fn seeded_json() -> MockEngine {
    let engine = MockEngine::new();
    run("CREATE TABLE j (id INT NOT NULL, doc JSON)", &engine).unwrap();
    run("INSERT INTO j VALUES (1, '[10,20,30]')", &engine).unwrap();
    run(
        r#"INSERT INTO j VALUES (2, '{"items":[{"n":1},{"n":2}],"k":"v"}')"#,
        &engine,
    )
    .unwrap();
    engine
}

#[test]
fn json_array_elements_expands_a_json_array() {
    let engine = seeded_json();
    let (cols, rows) = rows_of(
        run(
            "SELECT json_array_elements(doc) FROM j WHERE id = 1",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["json_array_elements"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Json("10".to_owned())],
            vec![Value::Json("20".to_owned())],
            vec![Value::Json("30".to_owned())],
        ],
    );
    // A non-array JSON document yields no rows.
    let (_, rows) = rows_of(
        run(
            "SELECT json_array_elements(doc) FROM j WHERE id = 2",
            &engine,
        )
        .unwrap(),
    );
    assert!(rows.is_empty());
}

#[test]
fn jsonb_path_query_yields_matches() {
    let engine = seeded_json();
    // Wildcard fan-out then member access.
    let (cols, rows) = rows_of(
        run(
            "SELECT jsonb_path_query(doc, '$.items[*].n') FROM j WHERE id = 2",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(cols, ["jsonb_path_query"]);
    assert_eq!(
        rows,
        vec![
            vec![Value::Json("1".to_owned())],
            vec![Value::Json("2".to_owned())]
        ],
    );
    // A path that matches nothing yields no rows.
    let (_, rows) = rows_of(
        run(
            "SELECT jsonb_path_query(doc, '$.nope') FROM j WHERE id = 2",
            &engine,
        )
        .unwrap(),
    );
    assert!(rows.is_empty());
}

#[test]
fn jsonb_path_query_rejects_an_unsupported_path_at_runtime() {
    let engine = seeded_json();
    assert!(matches!(
        run(
            "SELECT jsonb_path_query(doc, 'items') FROM j WHERE id = 2",
            &engine,
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn json_srf_rejects_wrong_argument_type_and_arity() {
    let engine = seeded_json();
    // json_array_elements over a non-JSON argument.
    assert!(matches!(
        run("SELECT json_array_elements(id) FROM j", &engine),
        Err(Error::TypeMismatch { .. } | Error::Unsupported(_)),
    ));
    // jsonb_path_query needs two arguments.
    assert!(matches!(
        run("SELECT jsonb_path_query(doc) FROM j", &engine),
        Err(Error::Unsupported(_)),
    ));
}

// --- Vectorized execution path (wiring): opt-in, equivalent to the row path ---

#[test]
fn vectorized_path_matches_row_path() {
    let engine = seeded_users();
    run(
        "INSERT INTO users VALUES (3, 'carol', 40, 2.5, TRUE)",
        &engine,
    )
    .unwrap();
    run(
        "INSERT INTO users VALUES (4, 'dave', 22, 7.0, FALSE)",
        &engine,
    )
    .unwrap();

    // Plans built only from SeqScan / Filter / Project / Sort / Limit — the vectorized-supported
    // shapes. Each must return identical columns+rows on the row path and the batch path.
    let queries = [
        "SELECT id, name FROM users ORDER BY id",
        "SELECT id FROM users WHERE age > 25 ORDER BY id",
        "SELECT name, age FROM users WHERE active = TRUE ORDER BY name",
        "SELECT id FROM users WHERE age IS NULL",
        "SELECT id FROM users ORDER BY id DESC LIMIT 2 OFFSET 1",
        "SELECT id, age + 1 FROM users WHERE id <= 3 ORDER BY id",
        // Scalar aggregates (no GROUP BY) — the vectorized ScalarAggregate's SIMD COUNT/SUM/MIN/MAX
        // must match the row path exactly.
        "SELECT COUNT(*), COUNT(age), SUM(age), MIN(age), MAX(age) FROM users",
        "SELECT SUM(age) FROM users WHERE active = TRUE",
        "SELECT MIN(score), MAX(score), SUM(score) FROM users",
        // Grouped aggregates (A-PERF.AGG6 / F2c) — the vectorized GroupedAggregate must match the
        // row path's output multiset AND its first-seen emission order (hence no ORDER BY on the
        // first two), including float SUM rounding, DISTINCT, and NULL-valued inputs.
        "SELECT active, COUNT(*) FROM users GROUP BY active",
        "SELECT age, COUNT(*), SUM(score), MIN(name), MAX(score) FROM users GROUP BY age",
        "SELECT active, COUNT(DISTINCT age), SUM(age) FROM users GROUP BY active ORDER BY active",
        "SELECT age, COUNT(*) FROM users WHERE score > 1.0 GROUP BY age ORDER BY age",
        "SELECT active, AVG(age) FROM users GROUP BY active ORDER BY active",
    ];
    for sql in queries {
        let row_path = rows_of(run(sql, &engine).unwrap());
        let batch_path = {
            let _g = crate::vectorized::scope(true);
            rows_of(run(sql, &engine).unwrap())
        };
        assert_eq!(row_path, batch_path, "vectorized mismatch for `{sql}`");
    }
}

#[test]
fn vectorized_path_matches_row_path_for_bytea() {
    // A present BYTEA value must materialize + read back identically on the row path and the
    // columnar (vectorized) Binary-array path. Regression guard for the batch BYTEA arm.
    let engine = MockEngine::new();
    run("CREATE TABLE blobs (id INT NOT NULL, b BYTEA)", &engine).unwrap();
    run(
        r"INSERT INTO blobs VALUES (1, '\x0102ff'), (2, NULL), (3, '\xdead')",
        &engine,
    )
    .unwrap();
    let sql = "SELECT id, b FROM blobs ORDER BY id";
    let row_path = rows_of(run(sql, &engine).unwrap());
    let batch_path = {
        let _g = crate::vectorized::scope(true);
        rows_of(run(sql, &engine).unwrap())
    };
    assert_eq!(row_path, batch_path, "vectorized BYTEA mismatch");
}

#[test]
fn vectorized_path_falls_back_for_unsupported_shapes() {
    // With the flag ON, plans with no vectorized operator (grouping sets, a FILTERed grouped call,
    // a scalar subquery predicate) still run correctly via the row-path fallback (try_build
    // returns None).
    let engine = seeded_users();
    let _g = crate::vectorized::scope(true);
    // ROLLUP plans as GroupingSetsAggregate, which has no vectorized operator.
    let (_, rows) = rows_of(
        run(
            "SELECT active, COUNT(*) FROM users GROUP BY ROLLUP(active)",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows.len(), 3); // two `active` groups + the super-aggregate row
    // A grouped call with FILTER needs row evaluation → GroupedAggregate refuses, row path runs.
    let (_, rows) = rows_of(
        run(
            "SELECT active, COUNT(*) FILTER (WHERE age > 25) FROM users GROUP BY active",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows.len(), 2);
    // A scalar-subquery predicate is not vectorizable → the whole plan falls back.
    let (_, rows) = rows_of(
        run(
            "SELECT id FROM users WHERE id = (SELECT MAX(id) FROM users)",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows, vec![vec![Value::Int(2)]]);
}

#[test]
fn vectorized_routing_uses_plan_time_scan_estimate() {
    // Selective routing: the planner annotates a single-table SELECT with
    // the scanned-row estimate from ANALYZE stats, and the executor routes to the batch path only
    // at/above the threshold — with no run-time stats fetch.
    let engine = seeded_users();
    let est_of = |engine: &MockEngine| -> Option<u64> {
        let logical = analyze(
            parse("SELECT id FROM users WHERE age > 25").unwrap(),
            engine,
        )
        .unwrap();
        match plan(logical) {
            crate::planner::PhysicalPlan::Select(_, est) => est,
            other => panic!("expected a SELECT plan, got {other:?}"),
        }
    };

    // Un-analyzed → no estimate → row path.
    assert_eq!(est_of(&engine), None);
    assert!(!super::meets_vectorize_threshold(0));

    let users = Catalog::lookup_table(&engine, "users")
        .unwrap()
        .expect("users table");
    let analyze_with = |rows: u64| {
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        engine
            .analyze_table(
                txn,
                users.id,
                &nusadb_core::TableStats {
                    row_count: rows,
                    page_count: 0,
                    columns: vec![],
                },
            )
            .unwrap();
        engine.commit(txn).unwrap();
    };

    // Analyzed at/above the threshold → estimate present and routes to the batch path.
    analyze_with(60_000);
    let est = est_of(&engine).expect("estimate present after ANALYZE");
    assert_eq!(est, 60_000);
    assert!(super::meets_vectorize_threshold(est));

    // A small analyzed count stays on the row path.
    analyze_with(100);
    let est = est_of(&engine).expect("estimate present");
    assert_eq!(est, 100);
    assert!(!super::meets_vectorize_threshold(est));
}

// --- per-column output types -----------------------------

#[test]
fn describe_column_types_align_with_names_and_resolve_types() {
    let engine = MockEngine::new();
    run(
        "CREATE TABLE typed (id INT NOT NULL, name TEXT, active BOOL, ts TIMESTAMP)",
        &engine,
    )
    .unwrap();
    let logical = analyze(
        parse("SELECT id, name, active, ts FROM typed").unwrap(),
        &engine,
    )
    .unwrap();
    let physical = plan(logical);

    let names = describe_columns(&physical);
    let types = describe_column_types(&physical);
    // Names and types describe the same columns, in the same order.
    assert_eq!(names, vec!["id", "name", "active", "ts"]);
    assert_eq!(
        types,
        vec![
            ColumnType::Int,
            ColumnType::Text,
            ColumnType::Bool,
            ColumnType::Timestamp,
        ]
    );
    assert_eq!(names.len(), types.len());
}

#[test]
fn describe_column_types_cover_returning_and_set_ops() {
    let engine = MockEngine::new();
    run("CREATE TABLE r (id INT NOT NULL, name TEXT)", &engine).unwrap();

    // INSERT ... RETURNING reports the projected column types.
    let returning = plan(
        analyze(
            parse("INSERT INTO r (id, name) VALUES (1, 'a') RETURNING id, name").unwrap(),
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        describe_columns(&returning).len(),
        describe_column_types(&returning).len()
    );
    assert_eq!(
        describe_column_types(&returning),
        vec![ColumnType::Int, ColumnType::Text]
    );

    // A set operation's types come from its leftmost branch and line up with its names.
    let set_op = plan(
        analyze(
            parse("SELECT id FROM r UNION SELECT id FROM r").unwrap(),
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(
        describe_columns(&set_op).len(),
        describe_column_types(&set_op).len()
    );
    assert_eq!(describe_column_types(&set_op), vec![ColumnType::Int]);
}

/// Evidence probe (run manually): how much of the hash-join cost is the generic
/// key path (`key_atoms`: expression eval + per-row `Vec<KeyAtom>` alloc + enum hashing)
/// versus an ideal single-int-key path (direct column access into a `HashMap<i64, _>`)?
/// Prints ns/row for build and probe on both paths over the same 1M-row data.
///
/// `cargo test -p nusadb-sql --release --lib apj2_int_key_path_evidence -- --ignored --nocapture`
#[test]
#[ignore = "manual evidence probe: 1M rows, run in release"]
fn apj2_int_key_path_evidence() {
    use std::time::Instant;

    use super::join::{KeySide, build_right_index, key_atoms};
    use crate::planner::{HashKey, TypedExpr, TypedExprKind};

    const N: usize = 1_000_000;
    const LEFT_WIDTH: usize = 2;

    // Two int columns per side; the join key (col 1) has ~10 duplicates per value.
    #[allow(clippy::cast_possible_wrap)]
    let rows: Vec<Row> = (0..N)
        .map(|i| vec![Value::Int(i as i64), Value::Int((i % 100_000) as i64)])
        .collect();
    let keys = vec![HashKey {
        left: TypedExpr {
            kind: TypedExprKind::Column(1),
            ty: ColumnType::Int,
        },
        right: TypedExpr {
            kind: TypedExprKind::Column(LEFT_WIDTH + 1),
            ty: ColumnType::Int,
        },
    }];
    let per_row = |d: std::time::Duration| d.as_nanos() / (N as u128);

    // Generic path, exactly as run_hash_join executes it today.
    let t = Instant::now();
    let generic_table = build_right_index(&rows, &keys, LEFT_WIDTH).unwrap();
    let generic_build = t.elapsed();
    let t = Instant::now();
    let mut generic_matches = 0usize;
    for row in &rows {
        if let Some(key) = key_atoms(&keys, row, KeySide::Left).unwrap()
            && let Some(indices) = generic_table.get(&key)
        {
            generic_matches += indices.len();
        }
    }
    let generic_probe = t.elapsed();

    // Ideal single-int-key path: direct column access, i64-keyed map, no eval / alloc / enum.
    let t = Instant::now();
    let mut int_table: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, row) in rows.iter().enumerate() {
        if let Some(Value::Int(k)) = row.get(1) {
            int_table.entry(*k).or_default().push(index);
        }
    }
    let int_build = t.elapsed();
    let t = Instant::now();
    let mut int_matches = 0usize;
    for row in &rows {
        if let Some(Value::Int(k)) = row.get(1)
            && let Some(indices) = int_table.get(k)
        {
            int_matches += indices.len();
        }
    }
    let int_probe = t.elapsed();

    assert_eq!(generic_matches, int_matches);
    println!(
        "AP-J2 evidence over {N} rows ({generic_matches} matches):\n\
         generic build {} ns/row · generic probe {} ns/row\n\
         int     build {} ns/row · int     probe {} ns/row",
        per_row(generic_build),
        per_row(generic_probe),
        per_row(int_build),
        per_row(int_probe),
    );
}

/// Pins: the single-int-key fast path must (a) fire exactly on its gate — one key, both
/// sides bare Int-physical columns — and (b) produce probe results identical to the generic
/// [`key_atoms`] index over the same rows, including NULL keys (match nothing) and duplicates.
#[test]
fn apj2_join_index_int_path_gate_and_equivalence() {
    use super::join::{JoinIndex, build_left_index, build_right_index};
    use crate::planner::{HashKey, TypedExpr, TypedExprKind};

    let col = |ordinal: usize, ty: ColumnType| TypedExpr {
        kind: TypedExprKind::Column(ordinal),
        ty,
    };
    let int_keys = |l: usize, r: usize| {
        vec![HashKey {
            left: col(l, ColumnType::Int),
            right: col(r, ColumnType::Int),
        }]
    };

    // Gate fires: one key, bare Int columns (SMALLINT/BIGINT are Int-physical too).
    let rows: Vec<Row> = vec![
        vec![Value::Int(1), Value::Int(10)],
        vec![Value::Int(2), Value::Null],
        vec![Value::Int(3), Value::Int(10)],
        vec![Value::Int(4), Value::Int(7)],
    ];
    let fired = JoinIndex::build_right(&rows, &int_keys(1, 3), 2).unwrap();
    assert!(matches!(fired, JoinIndex::Int { .. }));
    let smallint = vec![HashKey {
        left: col(1, ColumnType::SmallInt),
        right: col(3, ColumnType::BigInt),
    }];
    assert!(matches!(
        JoinIndex::build_right(&rows, &smallint, 2).unwrap(),
        JoinIndex::Int { .. }
    ));

    // Gate refuses: Text key, NUMERIC key (decimal canonicalization), two keys, non-column expr.
    let text_keys = vec![HashKey {
        left: col(1, ColumnType::Text),
        right: col(3, ColumnType::Text),
    }];
    assert!(matches!(
        JoinIndex::build_right(&rows, &text_keys, 2).unwrap(),
        JoinIndex::Generic(_)
    ));
    let numeric = ColumnType::Numeric {
        precision: 10,
        scale: 2,
    };
    let numeric_keys = vec![HashKey {
        left: col(1, numeric),
        right: col(3, numeric),
    }];
    assert!(matches!(
        JoinIndex::build_right(&rows, &numeric_keys, 2).unwrap(),
        JoinIndex::Generic(_)
    ));
    let two = vec![int_keys(0, 2).remove(0), int_keys(1, 3).remove(0)];
    assert!(matches!(
        JoinIndex::build_right(&rows, &two, 2).unwrap(),
        JoinIndex::Generic(_)
    ));
    let literal_key = vec![HashKey {
        left: TypedExpr {
            kind: TypedExprKind::Literal(Value::Int(1)),
            ty: ColumnType::Int,
        },
        right: col(3, ColumnType::Int),
    }];
    assert!(matches!(
        JoinIndex::build_right(&rows, &literal_key, 2).unwrap(),
        JoinIndex::Generic(_)
    ));

    // Probe equivalence, both directions, against the generic index over the same data:
    // duplicates keep identical index lists, NULL and absent keys match nothing.
    let keys = int_keys(1, 3);
    let left_probes: Vec<Row> = vec![
        vec![Value::Int(9), Value::Int(10)], // two matches
        vec![Value::Int(9), Value::Int(7)],  // one match
        vec![Value::Int(9), Value::Int(99)], // no match
        vec![Value::Int(9), Value::Null],    // NULL never matches
    ];
    let int_right = JoinIndex::build_right(&rows, &keys, 2).unwrap();
    let generic_right = JoinIndex::Generic(build_right_index(&rows, &keys, 2).unwrap());
    for probe in &left_probes {
        assert_eq!(
            int_right.probe_left(&keys, probe).unwrap(),
            generic_right.probe_left(&keys, probe).unwrap(),
            "probe_left diverged on {probe:?}"
        );
    }
    let int_left = JoinIndex::build_left(&rows, &keys, 2).unwrap();
    let generic_left = JoinIndex::Generic(build_left_index(&rows, &keys).unwrap());
    let mut padded_a: Row = vec![Value::Null; 2];
    let mut padded_b: Row = vec![Value::Null; 2];
    for probe in &left_probes {
        assert_eq!(
            int_left
                .probe_right(&keys, probe, 2, &mut padded_a)
                .unwrap(),
            generic_left
                .probe_right(&keys, probe, 2, &mut padded_b)
                .unwrap(),
            "probe_right diverged on {probe:?}"
        );
    }
}

/// Pin: the parallel grouped aggregate must be **bit-identical** to the sequential
/// vectorized fold and the row path — same rows, same first-seen emission order — across
/// multi-chunk inputs (several dealing rounds over ≥2 workers), NULL group keys, NULL values,
/// multi-column keys, and repeated runs (determinism). Shapes the parallel gate refuses (AVG)
/// must still return correct results through the sequential fallback with the force flag on.
#[test]
fn parallel_grouped_aggregate_matches_sequential_bit_for_bit() {
    let engine = MockEngine::new();
    run("CREATE TABLE m (k INT, txt TEXT, v INT)", &engine).unwrap();
    // ~3.5k rows spread over several 1024-row chunks; NULLs in keys and values; negatives.
    for start in (0..3500).step_by(500) {
        let values = (start..start + 500)
            .map(|i: i64| {
                let k = if i % 13 == 0 {
                    "NULL".to_owned()
                } else {
                    (i * 37 % 11).to_string()
                };
                let t = if i % 17 == 0 {
                    "NULL".to_owned()
                } else {
                    format!("'tag{}'", i % 7)
                };
                let v = if i % 7 == 0 {
                    "NULL".to_owned()
                } else {
                    (i % 97 - 48).to_string()
                };
                format!("({k},{t},{v})")
            })
            .collect::<Vec<_>>()
            .join(",");
        run(&format!("INSERT INTO m VALUES {values}"), &engine).unwrap();
    }
    let queries = [
        "SELECT k, COUNT(*), COUNT(v), SUM(v), MIN(v), MAX(v) FROM m GROUP BY k",
        "SELECT txt, COUNT(*), MIN(txt), MAX(v) FROM m GROUP BY txt",
        "SELECT k, txt, SUM(v), COUNT(*) FROM m GROUP BY k, txt",
        // ORDER BY above the aggregate: the Sort keys reference the synthesized row via
        // AggregateRefs, rewritten to columns like the projection (key and aggregate order).
        "SELECT k, COUNT(*), SUM(v) FROM m GROUP BY k ORDER BY k",
        "SELECT k, COUNT(*) FROM m GROUP BY k ORDER BY COUNT(*), k LIMIT 5",
        // AVG is outside the mergeable gate: with the force flag on it must fall back to the
        // sequential fold and still match.
        "SELECT k, AVG(v) FROM m GROUP BY k",
    ];
    for sql in queries {
        let row_path = rows_of(run(sql, &engine).unwrap());
        let sequential = {
            let _v = crate::vectorized::scope(true);
            let _p = crate::vectorized::parallel_scope(false);
            rows_of(run(sql, &engine).unwrap())
        };
        let (parallel, parallel_again, folds) = {
            let _v = crate::vectorized::scope(true);
            let _p = crate::vectorized::parallel_scope(true);
            let before = crate::vectorized::fold_count();
            let a = rows_of(run(sql, &engine).unwrap());
            let b = rows_of(run(sql, &engine).unwrap());
            (a, b, crate::vectorized::fold_count() - before)
        };
        // AVG is refused by the mergeable gate; every other query must actually run parallel
        // (a silent fallback would make this pin vacuous).
        if sql.contains("AVG") {
            assert_eq!(folds, 0, "AVG must fall back sequentially for `{sql}`");
        } else {
            assert_eq!(folds, 2, "parallel fold did not fire for `{sql}`");
        }
        assert_eq!(parallel, sequential, "parallel vs sequential for `{sql}`");
        assert_eq!(parallel, row_path, "parallel vs row path for `{sql}`");
        assert_eq!(
            parallel, parallel_again,
            "nondeterministic rerun for `{sql}`"
        );
    }
}

/// (QA, silent-wrong): with an ORDER BY and no explicit frame, the
/// default window frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` — so
/// `LAST_VALUE` runs through the current peer group (not "last of partition") and
/// `NTH_VALUE(v, n)` is NULL until the frame holds `n` rows. Ties share a frame end (peers);
/// an explicit frame and the no-ORDER-BY whole-partition default stay as they were.
#[test]
fn window_default_frame_runs_through_current_peer_group() {
    use Value::{Int, Null};
    let engine = MockEngine::new();
    run("CREATE TABLE w (id INT, v INT)", &engine).unwrap();
    run("INSERT INTO w VALUES (1,10),(2,20),(3,30)", &engine).unwrap();
    let rows = |sql: &str| rows_of(run(sql, &engine).unwrap()).1;

    // Running LAST_VALUE: each row is its own frame end (unique keys).
    assert_eq!(
        rows("SELECT id, LAST_VALUE(v) OVER (ORDER BY id) FROM w ORDER BY id"),
        vec![
            vec![Int(1), Int(10)],
            vec![Int(2), Int(20)],
            vec![Int(3), Int(30)],
        ]
    );
    // NTH_VALUE(2): NULL while the frame holds fewer than 2 rows.
    assert_eq!(
        rows("SELECT id, NTH_VALUE(v, 2) OVER (ORDER BY id) FROM w ORDER BY id"),
        vec![
            vec![Int(1), Null],
            vec![Int(2), Int(20)],
            vec![Int(3), Int(20)],
        ]
    );
    // FIRST_VALUE was already right (frame start is partition start either way).
    assert_eq!(
        rows("SELECT id, FIRST_VALUE(v) OVER (ORDER BY id) FROM w ORDER BY id"),
        vec![
            vec![Int(1), Int(10)],
            vec![Int(2), Int(10)],
            vec![Int(3), Int(10)],
        ]
    );
    // An explicit frame overrides the default exactly as before.
    assert_eq!(
        rows(
            "SELECT id, LAST_VALUE(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING               AND UNBOUNDED FOLLOWING) FROM w ORDER BY id"
        ),
        vec![
            vec![Int(1), Int(30)],
            vec![Int(2), Int(30)],
            vec![Int(3), Int(30)],
        ]
    );
    // No ORDER BY: every row is a peer, the default frame is the whole partition.
    assert_eq!(
        rows("SELECT id, LAST_VALUE(v) OVER () FROM w ORDER BY id"),
        vec![
            vec![Int(1), Int(30)],
            vec![Int(2), Int(30)],
            vec![Int(3), Int(30)],
        ]
    );

    // Ties: peers share the frame end — both id=2 rows see the last peer of their group.
    run("CREATE TABLE t2 (k INT, v INT)", &engine).unwrap();
    run("INSERT INTO t2 VALUES (1,10),(2,20),(2,25),(3,30)", &engine).unwrap();
    assert_eq!(
        rows("SELECT k, LAST_VALUE(v) OVER (ORDER BY k) FROM t2 ORDER BY k, v"),
        vec![
            vec![Int(1), Int(10)],
            vec![Int(2), Int(25)],
            vec![Int(2), Int(25)],
            vec![Int(3), Int(30)],
        ]
    );
}

/// Pins: the parallel fold now covers a pushed-down thread-pure WHERE (SIMD-shape and
/// complex predicates alike — survivors keep their global positions, so first-seen order stays
/// bit-identical) and scalar aggregates (one merged group; one row even over an empty or
/// fully-filtered input). An impure predicate (`random()`) must fall back sequentially — and
/// still return correct rows.
#[test]
fn parallel_filtered_and_scalar_aggregate_match_sequential() {
    let engine = MockEngine::new();
    run("CREATE TABLE f (k INT, v INT)", &engine).unwrap();
    for start in (0..3000_i64).step_by(500) {
        let values = (start..start + 500)
            .map(|i| {
                let v = if i % 11 == 0 {
                    "NULL".to_owned()
                } else {
                    (i % 89 - 44).to_string()
                };
                format!("({},{v})", i % 7)
            })
            .collect::<Vec<_>>()
            .join(",");
        run(&format!("INSERT INTO f VALUES {values}"), &engine).unwrap();
    }
    run("CREATE TABLE empty_f (k INT, v INT)", &engine).unwrap();
    let fires = [
        // SIMD-shape predicate (column <cmp> literal).
        "SELECT k, COUNT(*), SUM(v), MIN(v), MAX(v) FROM f WHERE v > 3 GROUP BY k",
        // Complex (non-SIMD) but thread-pure predicate.
        "SELECT k, COUNT(*), SUM(v) FROM f WHERE v > -20 AND k < 6 GROUP BY k ORDER BY k",
        // Predicate dropping every row: zero groups.
        "SELECT k, COUNT(*) FROM f WHERE v > 1000 GROUP BY k",
        // Scalar aggregates: bare, filtered, and over an empty table (one row, COUNT 0).
        "SELECT COUNT(*), COUNT(v), SUM(v), MIN(v), MAX(v) FROM f",
        "SELECT COUNT(*), SUM(v) FROM f WHERE v > 3",
        "SELECT COUNT(*), SUM(v), MIN(v) FROM empty_f",
    ];
    for sql in fires {
        let row_path = rows_of(run(sql, &engine).unwrap());
        let (parallel, folds) = {
            let _v = crate::vectorized::scope(true);
            let _p = crate::vectorized::parallel_scope(true);
            let before = crate::vectorized::fold_count();
            let rows = rows_of(run(sql, &engine).unwrap());
            (rows, crate::vectorized::fold_count() - before)
        };
        assert_eq!(folds, 1, "parallel fold did not fire for `{sql}`");
        assert_eq!(parallel, row_path, "parallel vs row path for `{sql}`");
    }
    // An impure predicate (random() is thread-local RNG) must refuse the parallel fold and
    // still produce a correct result through the sequential paths.
    let sql = "SELECT k, COUNT(*) FROM f WHERE v > random() * 0 - 1 GROUP BY k";
    let row_path = rows_of(run(sql, &engine).unwrap());
    let (sequential, folds) = {
        let _v = crate::vectorized::scope(true);
        let _p = crate::vectorized::parallel_scope(true);
        let before = crate::vectorized::fold_count();
        let rows = rows_of(run(sql, &engine).unwrap());
        (rows, crate::vectorized::fold_count() - before)
    };
    assert_eq!(folds, 0, "impure predicate must not run on workers");
    assert_eq!(sequential, row_path, "fallback mismatch for `{sql}`");
}

/// SQL-surface batch (user directive "impl sintaks yang belum"): `SUBSTRING(s FOR n)` defaults
/// its start to 1; `CREATE [MATERIALIZED] VIEW IF NOT EXISTS` is accepted (no-op contract
/// pinned end-to-end in `p11_views/plain.slt`); explicit `UNIQUE ... NULLS DISTINCT` (the
/// default semantic) and `TRUNCATE ... RESTRICT` are accepted.
#[test]
fn surface_batch_substring_view_if_not_exists_cte_hints() {
    let engine = MockEngine::new();
    run("CREATE TABLE s (t TEXT, v INT)", &engine).unwrap();
    run("INSERT INTO s VALUES ('abcdef', 1), ('xy', 2)", &engine).unwrap();

    // SUBSTRING(s FOR n) == SUBSTRING(s FROM 1 FOR n).
    let (_, rows) = rows_of(
        run(
            "SELECT SUBSTRING(t FOR 3), SUBSTRING(t FROM 1 FOR 3) FROM s ORDER BY v",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows[0][0], rows[0][1]);
    assert_eq!(rows[0][0], Value::Text("abc".into()));
    assert_eq!(rows[1][0], Value::Text("xy".into()));

    // CREATE VIEW IF NOT EXISTS: accepted (view reads resolve only through the Session, so
    // the no-op/no-clobber contract is pinned end-to-end in p11_views/plain.slt); combining
    // OR REPLACE with IF NOT EXISTS is refused.
    run(
        "CREATE VIEW IF NOT EXISTS v1 AS SELECT v FROM s WHERE v > 1",
        &engine,
    )
    .unwrap();
    run("CREATE VIEW IF NOT EXISTS s AS SELECT 1", &engine).unwrap(); // table name: no-op
    assert!(
        run(
            "CREATE OR REPLACE VIEW IF NOT EXISTS v1 AS SELECT 1",
            &engine
        )
        .is_err()
    );

    // UNIQUE NULLS DISTINCT = the default semantic (parse-level: constraints need the real
    // engine, exercised via SLT); NULLS NOT DISTINCT stays refused.
    parse("CREATE TABLE u1 (a INT, UNIQUE NULLS DISTINCT (a))").unwrap();
    assert!(parse("CREATE TABLE u2 (a INT, UNIQUE NULLS NOT DISTINCT (a))").is_err());

    // TRUNCATE ... RESTRICT (the default) is accepted; CASCADE stays refused.
    run("TRUNCATE TABLE s RESTRICT", &engine).unwrap();
    let (_, rows) = rows_of(run("SELECT COUNT(*) FROM s", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(0)]]);
    assert!(parse("TRUNCATE TABLE s CASCADE").is_err());
}

/// Multi-object `DROP <kind> a, b` (internal Batch desugar — atomic within the statement) and
/// `SELECT ... INTO t` (the standard's CTAS spelling, desugared to CREATE TABLE AS).
#[test]
fn surface_batch_multi_drop_and_select_into() {
    let engine = MockEngine::new();
    run("CREATE TABLE d1 (a INT)", &engine).unwrap();
    run("CREATE TABLE d2 (a INT)", &engine).unwrap();
    run("CREATE TABLE d3 (a INT)", &engine).unwrap();

    // Multi-drop removes both; a later reference fails.
    run("DROP TABLE d1, d2", &engine).unwrap();
    assert!(run("SELECT * FROM d1", &engine).is_err());
    assert!(run("SELECT * FROM d2", &engine).is_err());
    let (_, rows) = rows_of(run("SELECT COUNT(*) FROM d3", &engine).unwrap());
    assert_eq!(rows, vec![vec![Value::Int(0)]]);

    // A missing name mid-list fails the whole statement — atomically: d3 (dropped before the
    // error inside the statement transaction) must survive the rollback.
    assert!(run("DROP TABLE d3, nope", &engine).is_err());
    let (_, rows) = rows_of(run("SELECT COUNT(*) FROM d3", &engine).unwrap());
    assert_eq!(
        rows,
        vec![vec![Value::Int(0)]],
        "failed multi-drop must roll back atomically"
    );
    run("DROP TABLE IF EXISTS d3, nope", &engine).unwrap();
    assert!(run("SELECT * FROM d3", &engine).is_err());

    // SELECT INTO = CTAS: table created with the query's rows; re-INTO the same name errors.
    run("CREATE TABLE src (v INT)", &engine).unwrap();
    run("INSERT INTO src VALUES (1), (2), (3)", &engine).unwrap();
    run("SELECT v * 10 AS w INTO dst FROM src WHERE v > 1", &engine).unwrap();
    let (cols, rows) = rows_of(run("SELECT w FROM dst ORDER BY w", &engine).unwrap());
    assert_eq!(cols, vec!["w"]);
    assert_eq!(rows, vec![vec![Value::Int(20)], vec![Value::Int(30)]]);
    assert!(run("SELECT v INTO dst FROM src", &engine).is_err());
    assert!(parse("SELECT v INTO TEMPORARY t2 FROM src").is_err());
}

/// Gaps (QA routing): `||`/CONCAT coerce textout-able scalars to text — booleans render
/// `t`/`f` via the output function, unlike CAST's `true`/`false`; a bare JSON
/// number/boolean casts to the numeric/bool target while objects/arrays/strings stay refused;
/// `LENGTH`/`OCTET_LENGTH` over BYTEA count octets, `BIT_LENGTH` 8x.
#[test]
fn q47_concat_coerce_json_scalar_cast_bytea_length() {
    let engine = MockEngine::new();
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1.remove(0).remove(0);

    // QA's exact pins.
    assert_eq!(one("SELECT 'x' || 5"), Value::Text("x5".into()));
    assert_eq!(
        one("SELECT CONCAT('a', 1, TRUE)"),
        Value::Text("a1t".into())
    );
    assert_eq!(one("SELECT 5 || 'x'"), Value::Text("5x".into()));
    assert_eq!(one("SELECT 'v: ' || 1.5"), Value::Text("v: 1.5".into()));
    assert_eq!(one("SELECT 'b=' || FALSE"), Value::Text("b=f".into()));
    assert_eq!(
        one("SELECT CONCAT_WS('-', 'a', 2, NULL, TRUE)"),
        Value::Text("a-2-t".into())
    );
    // NULL stays NULL-strict for `||`, skipped for CONCAT.
    assert_eq!(one("SELECT 'x' || NULL"), Value::Null);
    assert_eq!(
        one("SELECT CONCAT('x', NULL, 'y')"),
        Value::Text("xy".into())
    );
    // CAST keeps its own boolean rendering — the output function differs by design.
    assert_eq!(one("SELECT CAST(TRUE AS TEXT)"), Value::Text("true".into()));

    // QA's exact pin + kind coverage; non-scalars stay loudly refused.
    assert_eq!(one("SELECT ('[1,2,3]'::jsonb->0)::int"), Value::Int(1));
    assert_eq!(
        one("SELECT ('{\"a\": 2.5}'::jsonb->'a')::float"),
        Value::Float(2.5)
    );
    assert_eq!(one("SELECT ('[true]'::jsonb->0)::bool"), Value::Bool(true));
    assert!(run("SELECT ('[[1]]'::jsonb->0)::int", &engine).is_err());
    assert!(run("SELECT ('[\"5\"]'::jsonb->0)::int", &engine).is_err());
    assert!(run("SELECT ('[null]'::jsonb->0)::int", &engine).is_err());

    // BYTEA length.
    assert_eq!(one(r"SELECT length('\x010203'::bytea)"), Value::Int(3));
    assert_eq!(one(r"SELECT octet_length('\x0102'::bytea)"), Value::Int(2));
    assert_eq!(one(r"SELECT bit_length('\x01'::bytea)"), Value::Int(8));
}

/// (QA): a `DELETE ... RETURNING` CTE — the archive-then-remove pattern —
/// runs once, its returned rows form the relation, and the deletion is real. INSERT/UPDATE
/// CTEs already worked; DELETE was rejected only because a stale parser comment predated the
/// vendored parser gaining the variant.
#[test]
fn q50_delete_returning_cte() {
    let engine = MockEngine::new();
    run("CREATE TABLE live (id INT, v INT)", &engine).unwrap();
    run("INSERT INTO live VALUES (1, 10), (2, 20), (3, 30)", &engine).unwrap();

    // QA's exact shape: the deleted rows form the CTE relation, and the deletion is real.
    let sql = "WITH d AS (DELETE FROM live WHERE v >= 20 RETURNING id, v) \
               SELECT id, v FROM d ORDER BY id";
    let (_, rows) = rows_of(run(sql, &engine).unwrap());
    assert_eq!(
        rows,
        vec![
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ]
    );
    let (_, live) = rows_of(run("SELECT id FROM live", &engine).unwrap());
    assert_eq!(live, vec![vec![Value::Int(1)]], "deletion must be real");

    // Aggregation over the relation works; RETURNING stays required.
    let (_, rows) = rows_of(
        run(
            "WITH d AS (DELETE FROM live RETURNING id) SELECT COUNT(*) FROM d",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows, vec![vec![Value::Int(1)]]);
    assert!(run("WITH d AS (DELETE FROM live) SELECT 1", &engine).is_err());
}

/// (QA): a WITH prefix on a top-level `INSERT ... SELECT` — completing the
/// archive-then-remove pattern end-to-end: the DELETE's RETURNING rows flow through the CTE
/// into the INSERT, the deletion is real, and the archive receives exactly those rows.
#[test]
fn q51_with_prefix_on_insert() {
    let engine = MockEngine::new();
    run("CREATE TABLE live2 (id INT, v INT)", &engine).unwrap();
    run("CREATE TABLE archive2 (id INT, v INT)", &engine).unwrap();
    run(
        "INSERT INTO live2 VALUES (1, 10), (2, 20), (3, 30)",
        &engine,
    )
    .unwrap();

    run(
        "WITH m AS (DELETE FROM live2 WHERE v >= 20 RETURNING id, v) \
         INSERT INTO archive2 SELECT id, v FROM m",
        &engine,
    )
    .unwrap();
    let (_, live) = rows_of(run("SELECT id FROM live2", &engine).unwrap());
    assert_eq!(live, vec![vec![Value::Int(1)]], "deletion must be real");
    let (_, archived) = rows_of(run("SELECT id, v FROM archive2 ORDER BY id", &engine).unwrap());
    assert_eq!(
        archived,
        vec![
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ]
    );

    // A plain (read-only) CTE prefix works too.
    run(
        "WITH src AS (SELECT 9 AS id, 90 AS v) INSERT INTO archive2 SELECT id, v FROM src",
        &engine,
    )
    .unwrap();
    let (_, n) = rows_of(run("SELECT COUNT(*) FROM archive2", &engine).unwrap());
    assert_eq!(n, vec![vec![Value::Int(3)]]);

    // The unwired shapes stay loud: WITH on UPDATE/DELETE, WITH onto a VALUES-source INSERT.
    assert!(run("WITH c AS (SELECT 1) UPDATE live2 SET v = 0", &engine).is_err());
    assert!(run("WITH c AS (SELECT 1) DELETE FROM live2", &engine).is_err());
    assert!(
        run(
            "WITH c AS (SELECT 1) INSERT INTO archive2 VALUES (5, 50)",
            &engine
        )
        .is_err()
    );
}

/// Gaps: `TIME ± INTERVAL` wraps within the 24-hour clock using the interval's sub-day
/// microseconds (whole days/months contribute nothing to a clock time), `TIME - TIME` is the
/// signed elapsed INTERVAL (QA-pinned); and a set operation's described
/// column type is the branches' UNIFIED type, not the leftmost leaf's (
/// `1 UNION 2.5` is a NUMERIC column, top-level and as a derived table).
#[test]
fn q52_time_interval_arith_and_union_type_unify() {
    let engine = MockEngine::new();
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1.remove(0).remove(0);

    // QA's exact case: 23:00 + 2h wraps to 01:00.
    assert_eq!(
        one("SELECT (TIME '23:00' + INTERVAL '2 hours')::text"),
        Value::Text("01:00:00".into())
    );
    assert_eq!(
        one("SELECT (TIME '01:00' - INTERVAL '2 hours')::text"),
        Value::Text("23:00:00".into())
    );
    // A whole-day component contributes nothing to a clock time.
    assert_eq!(
        one("SELECT (TIME '01:00' + INTERVAL '1 day 2 hours')::text"),
        Value::Text("03:00:00".into())
    );
    // A pathologically large interval must not overflow (audit catch): the sub-day reduction
    // happens before the add, so the wrapped clock time is exact — never panic, never garbage.
    assert_eq!(
        one("SELECT (TIME '12:00' + INTERVAL '9223372036854 seconds')::text"),
        {
            let sub_day = 9_223_372_036_854_i64 * 1_000_000 % (24 * 3600 * 1_000_000);
            let total = (12 * 3600 * 1_000_000 + sub_day) % (24 * 3600 * 1_000_000);
            let (h, m, sec) = (
                total / 3_600_000_000,
                total / 60_000_000 % 60,
                total / 1_000_000 % 60,
            );
            Value::Text(format!("{h:02}:{m:02}:{sec:02}"))
        }
    );

    // Commutative plus; TIME - TIME = signed INTERVAL.
    assert_eq!(
        one("SELECT (INTERVAL '30 minutes' + TIME '10:00')::text"),
        Value::Text("10:30:00".into())
    );
    assert_eq!(
        one("SELECT (TIME '10:30' - TIME '10:00')::text"),
        Value::Text("00:30:00".into())
    );

    // UNION type unification: the described column is NUMERIC, top-level and derived.
    let logical = analyze(parse("SELECT 1 UNION SELECT 2.5").unwrap(), &engine).unwrap();
    let types = describe_column_types(&plan(logical));
    assert!(
        matches!(types.as_slice(), [ColumnType::Numeric { .. }]),
        "got {types:?}"
    );
    let (_, rows) = rows_of(
        run(
            "SELECT SUM(x) FROM (SELECT 1 AS x UNION SELECT 2.5) t",
            &engine,
        )
        .unwrap(),
    );
    assert_eq!(rows.len(), 1); // a NUMERIC-typed sum over the unified column analyzes and runs
}

/// Gaps: `substring(s FROM 'pattern')` is the POSIX-regex form (
/// first capture group when present, whole first match otherwise, NULL on no match; the
/// positional INT form is untouched), and an untyped `NULL` types from context under `NOT`
/// and as an `IN` probe (both evaluate to NULL, three-valued).
#[test]
fn q52_substring_regex_and_null_type_infer() {
    let engine = MockEngine::new();
    let one = |sql: &str| rows_of(run(sql, &engine).unwrap()).1.remove(0).remove(0);

    // QA's exact cases: whole match without groups, first capture group with one.
    assert_eq!(
        one("SELECT substring('foobar' FROM 'o.b')"),
        Value::Text("oob".into())
    );
    assert_eq!(
        one("SELECT substring('foobar' FROM 'o(.)b')"),
        Value::Text("o".into())
    );
    // No match → NULL; the positional form still slices by character position.
    assert_eq!(one("SELECT substring('foobar' FROM 'xyz')"), Value::Null);
    assert_eq!(
        one("SELECT substring('foobar' FROM 2 FOR 3)"),
        Value::Text("oob".into())
    );
    // The SIMILAR-TO escape form (all-TEXT, 3 args) stays a loud reject.
    assert!(run("SELECT substring('foobar' FROM 'o.b' FOR '#')", &engine).is_err());
    // A malformed pattern is a loud error, not NULL.
    assert!(run("SELECT substring('foobar' FROM '(unclosed')", &engine).is_err());

    // NOT NULL is NULL (three-valued), and so is NULL IN (1, 2) / NULL NOT IN (1, 2).
    assert_eq!(one("SELECT NOT NULL"), Value::Null);
    assert_eq!(one("SELECT NULL IN (1, 2)"), Value::Null);
    assert_eq!(one("SELECT NULL NOT IN (1, 2)"), Value::Null);
    // NOT over a non-BOOL operand is still rejected.
    assert!(run("SELECT NOT 'abc'", &engine).is_err());
}
