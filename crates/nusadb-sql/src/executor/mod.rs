//! Execute a [`PhysicalPlan`] against a [`StorageEngine`].
//!
//! This is the first-iteration executor: materialize-based and row-at-a-time.
//! It is deliberately *not* the vectorized 1024-row batch + SIMD AVX2 design
//! that [`crate::BATCH_SIZE`] points at — that is an explicit perf follow-up
//! once correctness is locked in. Statements run inside an auto-committed
//! transaction the executor opens itself; an error during execution rolls
//! back.
//!
//! The operators live in per-concern submodules (ADR 007: `ddl`, `dml`, `ops`, `scan`); they call
//! each other and the dispatch/format helpers in this file through a glob re-export.
#![allow(clippy::wildcard_imports)]

use std::collections::{HashMap, HashSet};

use nusadb_core::{
    AlterOp, ColumnDef, ColumnType, IsolationLevel, StorageEngine, TableDef, TableSchema,
    TableStats, Tid, TxnId,
};

use crate::ast;
use crate::error::Error;
use crate::planner::{
    AggregateCall, AlterColumnOp, AlterTablePlan, AnalyzePlan, Assignment, ConflictArbiter,
    CreateIndexPlan, CreatePlainViewPlan, CreateSchemaPlan, CreateSequencePlan, CreateTablePlan,
    DeletePlan, DropIndexPlan, DropSchemaPlan, DropSequencePlan, DropTablePlan, DropViewPlan,
    FrameBound, HashKey, InsertPlan, InsertSource, OnConflictPlan, OrderByKey,
    PhysicalCreateTableAs, PhysicalMaterializedView, PhysicalOperator, PhysicalPlan,
    PhysicalRecursiveCte, PhysicalSetOp, SetOpTree, TxnCharacteristics, TypedExpr, UpdatePlan,
    WindowExpr, WindowFrame,
};

mod clock;
pub mod cost;
mod crypto;
pub mod eval;
mod rng;
pub mod row;
mod session_ctx;
mod stats;

// Operator submodules (ADR 007), glob-re-exported below. agg/join remain stubs whose operators
// currently live in `ops` (a follow-up may hoist them into their own files).
pub mod agg;
pub(crate) mod coldefault;
mod ddl;
mod dml;
mod function;
mod index_key;
mod instrument;
mod ivm;
pub mod join;
mod lock_skip;
pub mod ops;
mod procedure;
mod recursive;
pub mod scan;
mod script;
mod spill_setop;
mod spill_sort;
mod trigger;
// Spill-to-disk substrate. The consumers (streaming `RowSource`, grace hash join, external
// merge sort) land in later commits; this commit is the
// foundation, so most of it is exercised only by its own unit tests until then.
#[allow(
    dead_code,
    reason = "Phase 0 foundation; first operator consumer lands in the next commit"
)]
mod spill;
// Phase 1: pull-based streaming for the linear pipeline. The consumer (the spilling grace hash
// join / external sort) lands next; for now `stream_op` is exercised by an oracle test that asserts
// it yields exactly what `execute_op` does.
#[allow(
    dead_code,
    reason = "Phase 1 streaming substrate; the spilling-operator consumer lands in the next commit"
)]
mod stream;
use agg::*;
use ddl::*;
use dml::*;
use join::*;
use ops::*;
use scan::*;

pub use function::lookup_function_definition;
pub use ops::{parse_work_mem, set_work_mem, work_mem};
pub use row::Row;
pub use spill::{SpillConfig, set_spill_config};

/// Whether spill-to-disk is configured for this process. Crate-visible so the vectorized
/// group-by routing can defer to the row path's bounded-memory sort-based group-by whenever spill
/// is on (A-PERF.AGG6) instead of holding O(groups) state past the budget.
pub(crate) fn spill_is_configured() -> bool {
    spill::spill_config().is_some()
}

/// The outcome of executing a single SQL statement.
#[derive(Debug)]
pub enum ExecutionResult {
    /// `CREATE TABLE` — the new (or existing-if-`IF NOT EXISTS`) table id.
    Created(nusadb_core::TableId),
    /// `DROP TABLE` succeeded (or `IF EXISTS` made it a no-op).
    Dropped,
    /// `ALTER TABLE` succeeded (or an `IF [NOT] EXISTS` guard made it a no-op).
    Altered,
    /// `INSERT` — the number of rows written.
    Inserted(usize),
    /// `UPDATE` — the number of rows matched by the predicate and updated.
    Updated(usize),
    /// `DELETE` — the number of rows matched by the predicate and removed.
    Deleted(usize),
    /// `MERGE` — the number of source rows that drove an applied `WHEN` action.
    Merged(usize),
    /// `SELECT` — projected output rows and their column names.
    Rows {
        /// Names of the output columns, in order.
        columns: Vec<String>,
        /// One row per output tuple, each with `columns.len()` entries.
        rows: Vec<Row>,
    },
    /// `BEGIN` — an explicit transaction is now active in the session.
    TransactionBegun,
    /// `COMMIT` — the active transaction was committed.
    TransactionCommitted,
    /// `ROLLBACK` — the active transaction was rolled back.
    TransactionRolledBack,
    /// `SET TRANSACTION` — the session's default transaction characteristics were updated.
    TransactionCharacteristicsSet,
    /// `SET` / `RESET` — a session variable was set or reset.
    VariableSet,
    /// `SAVEPOINT` — a savepoint was established in the active transaction.
    SavepointCreated,
    /// `ROLLBACK TO SAVEPOINT` — the active transaction was rolled back to a savepoint.
    RolledBackToSavepoint,
    /// `RELEASE SAVEPOINT` — a savepoint was released from the active transaction.
    SavepointReleased,
    /// `VACUUM` — the number of dead row versions reclaimed.
    Vacuumed(usize),
    /// `REINDEX` — accepted as a no-op (NusaDB's B-tree indexes are always consistent).
    Reindexed,
    /// `ANALYZE` — statistics recomputed for the given number of columns.
    Analyzed {
        /// Table whose statistics were recomputed.
        table: String,
        /// Number of columns analyzed.
        columns: usize,
    },
    /// `COMMENT ON` — the target was resolved and the statement accepted.
    Commented,
    /// `LOCK TABLE` — the requested table locks were acquired.
    TableLocked,
    /// `PREPARE` — a prepared statement was stored in the session.
    Prepared,
    /// `DEALLOCATE` — one or all prepared statements were discarded.
    Deallocated,
    /// `CREATE SCHEMA` succeeded (or `IF NOT EXISTS` made it a no-op).
    SchemaCreated,
    /// `DROP SCHEMA` succeeded (or `IF EXISTS` made it a no-op).
    SchemaDropped,
    /// `CREATE DATABASE` accepted — a single-database compatibility no-op (NusaDB is one database
    /// per data directory).
    DatabaseCreated,
    /// `ALTER DATABASE` accepted — a single-database compatibility no-op.
    DatabaseAltered,
    /// `DROP DATABASE` dropped every table in the single database (backing each up first unless
    /// `FIX DROP DATABASE` skipped the backup).
    DatabaseDropped,
    /// `CREATE SEQUENCE` succeeded (or `IF NOT EXISTS` made it a no-op).
    SequenceCreated,
    /// `DROP SEQUENCE` succeeded (or `IF EXISTS` made it a no-op).
    SequenceDropped,
    /// `CREATE INDEX` succeeded (or `IF NOT EXISTS` made it a no-op).
    IndexCreated,
    /// `DROP INDEX` succeeded (or `IF EXISTS` made it a no-op).
    IndexDropped,
    /// `CREATE TRIGGER` succeeded (or `OR REPLACE` replaced an existing one).
    TriggerCreated,
    /// `DROP TRIGGER` succeeded (or `IF EXISTS` made it a no-op).
    TriggerDropped,
    /// `ALTER TRIGGER ... RENAME TO` succeeded.
    TriggerAltered,
    /// `CREATE PROCEDURE` succeeded (or `OR REPLACE` replaced an existing one).
    ProcedureCreated,
    /// `DROP PROCEDURE` succeeded (or `IF EXISTS` made it a no-op).
    ProcedureDropped,
    /// `CALL` ran a stored procedure to completion.
    ProcedureCalled,
    /// `CREATE FUNCTION` succeeded (or `OR REPLACE` replaced an existing one).
    FunctionCreated,
    /// `DROP FUNCTION` succeeded (or `IF EXISTS` made it a no-op).
    FunctionDropped,
}

/// A push-based receiver for a streamed statement's output (Phase 2 streaming output).
///
/// [`Session::execute_streaming`] drives this instead of collecting a `Vec<Row>`: [`columns`] is
/// called exactly once (before any row) for a row-producing statement, then [`row`] is called once
/// per output row **as it is produced**. A `SELECT` whose top operator is linear (scan/filter/
/// project/limit) is delivered one row at a time, so the executor never holds the whole result at
/// once; a blocking top operator (sort/aggregate/distinct/set-op) still materializes once internally
/// (spilling under `work_mem` per Phase 1) but is drained into the sink without a second copy.
///
/// A sink that fails (e.g. a wire-layer write error) aborts the statement: the error propagates and,
/// in auto-commit, rolls the implicit transaction back.
///
/// [`columns`]: RowSink::columns
/// [`row`]: RowSink::row
pub trait RowSink {
    /// Announce the output column names. Called once, before the first [`row`](RowSink::row), for a
    /// row-producing statement only.
    ///
    /// # Errors
    /// Any sink-side error (propagated to abort the statement).
    fn columns(&mut self, columns: &[String]) -> Result<(), Error>;

    /// Announce the output columns **with their resolved types**. `names` and `types` line
    /// up element-for-element (`types` from [`describe_column_types`]). The default ignores the types
    /// and delegates to [`columns`](RowSink::columns), so a names-only sink keeps working; a typed
    /// sink (e.g. the wire server at protocol `minor >= 1`) overrides this to emit the per-column type
    /// tags. Called once, before the first [`row`](RowSink::row), in place of `columns`.
    ///
    /// # Errors
    /// Any sink-side error (propagated to abort the statement).
    fn columns_typed(&mut self, names: &[String], _types: &[ColumnType]) -> Result<(), Error> {
        self.columns(names)
    }

    /// Receive one output row.
    ///
    /// # Errors
    /// Any sink-side error (propagated to abort the statement).
    fn row(&mut self, row: &[ast::Value]) -> Result<(), Error>;
}

/// The outcome of [`Session::execute_streaming`].
#[derive(Debug)]
pub enum StreamOutcome {
    /// A row-producing statement: the output column names and the number of rows delivered to the
    /// sink (the rows themselves went to the sink, not here).
    Rows {
        /// Output column names, as also passed to [`RowSink::columns`].
        columns: Vec<String>,
        /// How many rows were delivered to the sink.
        count: usize,
    },
    /// A non-row statement (DDL/DML/transaction control): its ordinary [`ExecutionResult`], so a
    /// caller can derive the same command tag it would from [`Session::execute`].
    Other(ExecutionResult),
}

/// Run a [`PhysicalPlan`] against `engine`, returning what happened.
///
/// One-shot convenience: equivalent to creating a [`Session`] and executing a
/// single plan in it. Explicit transaction-control plans
/// (`BEGIN`/`COMMIT`/`ROLLBACK`) are rejected here — they only make sense
/// across multiple statements, so a [`Session`] is required.
pub fn execute(plan: PhysicalPlan, engine: &dyn StorageEngine) -> Result<ExecutionResult, Error> {
    match plan {
        PhysicalPlan::BeginTransaction(_)
        | PhysicalPlan::Commit
        | PhysicalPlan::Rollback
        | PhysicalPlan::SetTransaction(_) => Err(Error::Unsupported(
            "explicit BEGIN/COMMIT/ROLLBACK requires a Session — use Session::execute".to_owned(),
        )),
        other => Session::new(engine).execute(other),
    }
}

/// Auto-analyze policy (D-AUTO-ANALYZE): keep the planner's statistics fresh without a manual
/// `ANALYZE`.
///
/// Runs `ANALYZE` on every table whose write churn since its last analyze has crossed the
/// scale-factor threshold `base + scale * approx_row_count`, and returns the names of the tables
/// it analysed, so the planner's histogram/MCV statistics never go stale on a live database.
///
/// Designed to run **off the query path** — e.g. from a background scheduler — so a client query is
/// never blocked or perturbed: each `ANALYZE` runs in its own autocommitted transaction, and a
/// failure on one table is logged and skipped so a single bad table never stalls the sweep. It is a
/// planning hint only; it never changes query *results*, only how well they are planned.
///
/// # Errors
/// Propagates a failure to enumerate the tables or read one's churn/row-count (engine-lock poison —
/// the whole sweep is moot); an individual table's `ANALYZE` failure is skipped, not propagated.
pub fn auto_analyze_stale_tables(
    engine: &dyn StorageEngine,
    scale: f64,
    base: u64,
) -> Result<Vec<String>, Error> {
    let mut analysed = Vec::new();
    for name in engine.list_tables()? {
        let Some(table) = engine.lookup_table(&name)? else {
            continue;
        };
        let churn = engine.churn_since_analyze(table.id)?;
        if churn == 0 {
            continue; // untouched since the last analyze — nothing to refresh
        }
        let rows = engine.approx_row_count(table.id)?;
        // The scale-factor churn formula: analyze once churn passes `base + scale * row_count`.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "the auto-analyze threshold is an approximate heuristic; `scale` and `rows` are \
                      non-negative and precision loss on a billion-row table never changes the \
                      cross-the-threshold decision"
        )]
        let limit = base.saturating_add((scale * rows as f64) as u64);
        if churn < limit {
            continue;
        }
        // Refresh via the normal ANALYZE path in its own autocommitted transaction, off any query.
        let columns = (0..table.columns.len()).collect();
        let plan = AnalyzePlan { table, columns };
        let Ok(txn) = engine.begin(IsolationLevel::default()) else {
            continue;
        };
        match run_analyze(plan, engine, txn).and_then(|_| engine.commit(txn).map_err(Into::into)) {
            Ok(()) => analysed.push(name),
            Err(e) => {
                let _ = engine.rollback(txn);
                tracing::warn!(table = %name, error = %e, "auto-analyze skipped a table");
            },
        }
    }
    Ok(analysed)
}

/// Execute `COPY <table> FROM STDIN`: bulk-load the text-format `data` into the table in a
/// single auto-committed transaction, returning the number of rows inserted.
///
/// The wire server collects the `CopyData` stream and calls this once the client sends `CopyDone`.
/// The load is all-or-nothing: a malformed row or a constraint violation rolls the whole copy back
/// (the same semantics a single multi-row `INSERT` would have).
///
/// # Errors
/// Propagates parse, type, and constraint errors; the transaction is rolled back on any failure.
pub fn copy_from(engine: &dyn StorageEngine, copy: &ast::Copy, data: &str) -> Result<usize, Error> {
    let txn = engine.begin(IsolationLevel::default())?;
    match run_copy_from(copy, data, engine, txn) {
        // A failed commit (e.g. ENOSPC appending CommitTxn) must roll back, or the transaction leaks
        // in `active` with its locks held and purge stalls forever. The `?`
        // that used to be here threw without rolling back.
        Ok(count) => match engine.commit(txn) {
            Ok(()) => Ok(count),
            Err(err) => {
                let _ = engine.rollback(txn);
                Err(err.into())
            },
        },
        Err(err) => {
            let _ = engine.rollback(txn);
            Err(err)
        },
    }
}

/// Execute `COPY <table> TO STDOUT`: render the table's rows in the text format in a single
/// read-only transaction, returning the row count and the rendered payload (newline-terminated).
///
/// The wire server streams the payload back as `CopyData` framed by `CopyOutResponse`/`CopyDone`.
///
/// # Errors
/// Propagates lookup and rendering errors; the transaction is rolled back on any failure.
pub fn copy_to(engine: &dyn StorageEngine, copy: &ast::Copy) -> Result<(usize, String), Error> {
    let txn = engine.begin(IsolationLevel::default())?;
    match run_copy_to(copy, engine, txn) {
        // A failed commit must roll back rather than leak the transaction,
        // mirroring `copy_from` above — even though a read-only COPY TO commits with an empty undo
        // set and so rarely reaches a failing durability point.
        Ok(out) => match engine.commit(txn) {
            Ok(()) => Ok(out),
            Err(err) => {
                let _ = engine.rollback(txn);
                Err(err.into())
            },
        },
        Err(err) => {
            let _ = engine.rollback(txn);
            Err(err)
        },
    }
}

/// Execute `plan` inside an already-open transaction `txn` — the caller owns the transaction
/// lifecycle (begin / commit / rollback).
///
/// This lets a caller run **analysis and execution under one transaction** so schema resolution and
/// the statement see the same snapshot: begin a txn, `analyze` through a catalog bound to it, then
/// `execute_in_txn` the resulting plan. Transaction-control plans (`BEGIN`/`COMMIT`/`ROLLBACK`) are
/// rejected here exactly as [`execute`] rejects them — they require a [`Session`].
///
/// # Errors
/// Propagates execution errors; never commits or rolls back `txn`.
pub fn execute_in_txn(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // Refresh the READ COMMITTED / READ UNCOMMITTED statement snapshot exactly once, here at the
    // buffered-execution choke-point every protocol reaches: simple-query (Session), extended-query
    // / prepared statement (wire `run_stmt_in_state` → `execute_in_txn_as_with_settings`), and
    // auto-commit. `dispatch` must NOT do this — it recurses for `Batch` desugars and is re-entered
    // by triggers / stored-procedure / DO bodies, which run within the SAME statement snapshot
    // `execute_in_txn` is never re-entered per-operator, so it fires once.
    engine.begin_statement(txn)?;
    match plan {
        PhysicalPlan::BeginTransaction(_)
        | PhysicalPlan::Commit
        | PhysicalPlan::Rollback
        | PhysicalPlan::SetTransaction(_) => Err(Error::Unsupported(
            "explicit BEGIN/COMMIT/ROLLBACK requires a Session — use Session::execute".to_owned(),
        )),
        other => dispatch(other, engine, txn),
    }
}

/// Like [`execute_in_txn`], but runs as session `user`.
///
/// `CURRENT_USER` / `SESSION_USER` and row-level-security predicates then observe the connection's
/// authenticated identity, rather than the default bootstrap user.
///
/// The wire server calls this with the user it authenticated; that same user must drive the
/// [`Catalog`](crate::Catalog) used for analysis (so policy selection and the RLS superuser bypass
/// agree with the predicate's `CURRENT_USER` at execution).
///
/// # Errors
/// Propagates execution errors; never commits or rolls back `txn`.
pub fn execute_in_txn_as(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
) -> Result<ExecutionResult, Error> {
    session_ctx::set_session_user(user);
    execute_in_txn(plan, engine, txn)
}

/// Like [`execute_in_txn_as`], but also pins the connection's `SET` variables.
///
/// `current_setting` and the session built-ins then observe an earlier `SET name = …`. The
/// wire server keeps a per-connection settings map (session-state-over-wire) and passes it here for
/// every statement.
///
/// # Errors
/// Propagates execution errors; never commits or rolls back `txn`.
#[allow(
    clippy::implicit_hasher,
    reason = "the wire server's per-connection GUC store is a default-hasher HashMap; generalising \
              the hasher would force every caller to name a type parameter for no benefit"
)]
pub fn execute_in_txn_as_with_settings(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    settings: &HashMap<String, String>,
) -> Result<ExecutionResult, Error> {
    session_ctx::set_session_user_with_settings(user, settings);
    execute_in_txn(plan, engine, txn)
}

/// Render `SHOW name` against an explicit per-connection `SET` store (session-state-over-wire).
///
/// An explicit `SET` wins, else a well-known read-only/session GUC reports its honest built-in default
/// (so `SHOW server_version` / `SHOW transaction_isolation` are useful), else the empty string. Kept
/// consistent with [`execute_in_txn_as_with_settings`]'s `current_setting`.
#[must_use]
#[allow(
    clippy::implicit_hasher,
    reason = "the wire server's per-connection GUC store is a default-hasher HashMap"
)]
pub fn show_session_variable(name: &str, settings: &HashMap<String, String>) -> ExecutionResult {
    let value = settings
        .get(name)
        .cloned()
        .or_else(|| {
            if matches!(
                name,
                "transaction_isolation" | "default_transaction_isolation"
            ) {
                Some(isolation_guc_text(IsolationLevel::default()).to_owned())
            } else {
                session_ctx::builtin_guc_static_default(name).map(ToOwned::to_owned)
            }
        })
        .unwrap_or_default();
    ExecutionResult::Rows {
        columns: vec![name.to_owned()],
        rows: vec![vec![ast::Value::Text(value)]],
    }
}

/// Streaming counterpart of [`execute_in_txn_as`] (Phase 2).
///
/// Delivers a `SELECT`'s rows to `sink` one at a time instead of returning a `Vec`, so the wire
/// server can write `DataRow` frames to the socket as rows are produced rather than buffering the
/// whole result set. `txn` is caller-managed (never committed or rolled back here), exactly like
/// `execute_in_txn_as`. A plain `SELECT` streams through `stream_op`; every other statement runs
/// buffered and is replayed into the sink, so the observable result (and `StreamOutcome`) matches
/// what `execute_in_txn_as` would produce.
///
/// # Errors
/// Propagates execution errors and any error returned by `sink`; never commits or rolls back `txn`.
pub fn execute_in_txn_as_streaming(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    sink: &mut dyn RowSink,
) -> Result<StreamOutcome, Error> {
    session_ctx::set_session_user(user);
    if let PhysicalPlan::Select(op, _est) = plan {
        // Streaming SELECT skips `execute_in_txn`, so refresh the RC/RU statement snapshot here too
        // A stale-snapshot read is only a SELECT, but keeping every
        // protocol's reads on one statement snapshot is the whole point.
        engine.begin_statement(txn)?;
        // Mirror `dispatch`'s per-statement clock pin (bypassed here because we skip `dispatch`).
        clock::set_statement_now();
        return stream_select_rows(&op, engine, txn, sink);
    }
    // Capture the row shape before the plan is consumed so a replayed RETURNING set is typed.
    let types = describe_column_types(&plan);
    replay_into_sink(execute_in_txn(plan, engine, txn)?, &types, sink)
}

/// Like [`execute_in_txn_as_streaming`], but also pins the connection's `SET` variables.
///
/// A streamed `SELECT current_setting(…)` then reflects an earlier `SET`. Used by the wire server's
/// extended-query path, which carries a per-connection settings map.
///
/// # Errors
/// Propagates execution errors and any error returned by `sink`; never commits or rolls back `txn`.
#[allow(
    clippy::implicit_hasher,
    reason = "the wire server's per-connection GUC store is a default-hasher HashMap"
)]
pub fn execute_in_txn_as_streaming_with_settings(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    settings: &HashMap<String, String>,
    sink: &mut dyn RowSink,
) -> Result<StreamOutcome, Error> {
    session_ctx::set_session_user_with_settings(user, settings);
    if let PhysicalPlan::Select(op, _est) = plan {
        // Streaming SELECT skips `execute_in_txn` — refresh the RC/RU statement snapshot here too
        // So the extended-query streaming path stays on one snapshot.
        engine.begin_statement(txn)?;
        clock::set_statement_now();
        return stream_select_rows(&op, engine, txn, sink);
    }
    // Capture the row shape before the plan is consumed so a replayed RETURNING set is typed.
    let types = describe_column_types(&plan);
    replay_into_sink(execute_in_txn(plan, engine, txn)?, &types, sink)
}

/// Replay a finished [`ExecutionResult`] into `sink` for a non-streamed statement: a row result
/// announces its columns then pushes each row; anything else passes through as
/// [`StreamOutcome::Other`]. Keeps the streaming entry points behaviourally identical to the
/// buffered ones.
///
/// `types` are the plan's output column types (from [`describe_column_types`]), captured before the
/// plan was executed. When they line up element-for-element with the produced columns the typed
/// `RowSink::columns_typed` is used so a buffered-but-replayed row set (notably
/// `INSERT/UPDATE/DELETE ... RETURNING`) advertises its real per-column types over the wire, exactly
/// like a streamed `SELECT` — without them a `RETURNING <int col>` would be reported as text. A
/// length mismatch (a row-producing plan `describe_column_types` does not cover) falls back to the
/// untyped form rather than risk a name/type misalignment.
///
/// # Errors
/// Propagates any error returned by `sink`.
fn replay_into_sink(
    result: ExecutionResult,
    types: &[ColumnType],
    sink: &mut dyn RowSink,
) -> Result<StreamOutcome, Error> {
    match result {
        ExecutionResult::Rows { columns, rows } => {
            if types.len() == columns.len() {
                sink.columns_typed(&columns, types)?;
            } else {
                sink.columns(&columns)?;
            }
            for row in &rows {
                sink.row(row)?;
            }
            Ok(StreamOutcome::Rows {
                columns,
                count: rows.len(),
            })
        },
        other => Ok(StreamOutcome::Other(other)),
    }
}

/// The output column names a [`PhysicalPlan`] would produce, **without executing it**.
///
/// For the extended-query `Describe` metadata path: a `Describe(Portal)` must report the row
/// shape without running the statement — running it there would apply the statement's side effects
/// (INSERT/UPDATE/DELETE/DDL commit) before `Execute`, violating the protocol. Returns the
/// column names for a row-producing plan (`SELECT`, or `EXPLAIN`'s single text column) and an empty
/// list for a command that returns no rows.
#[must_use]
pub fn describe_columns(plan: &PhysicalPlan) -> Vec<String> {
    match plan {
        PhysicalPlan::Select(op, _) => output_columns(op),
        // Must match the column name `execute` emits for EXPLAIN so a Describe-before-Execute and
        // the executed result agree.
        PhysicalPlan::Explain(..) => vec!["plan".to_owned()],
        // A set operation produces a row set; its output columns come from the left branch.
        PhysicalPlan::SetOperation(p) => p.columns.clone(),
        // `INSERT/UPDATE/DELETE ... RETURNING` produce a row set; report projected columns.
        PhysicalPlan::Insert(p) => p.returning.iter().map(|r| r.name.clone()).collect(),
        PhysicalPlan::Update(p) => p.returning.iter().map(|r| r.name.clone()).collect(),
        PhysicalPlan::Delete(p) => p.returning.iter().map(|r| r.name.clone()).collect(),
        // `SHOW name` reports one row in a column named after the variable.
        PhysicalPlan::ShowVariable(name) => vec![name.clone()],
        // Catalog introspection row shapes.
        PhysicalPlan::ShowTables => vec!["table".to_owned()],
        PhysicalPlan::ShowColumns(_) => {
            vec![
                "column".to_owned(),
                "type".to_owned(),
                "nullable".to_owned(),
            ]
        },
        _ => Vec::new(),
    }
}

/// The output column **types** a [`PhysicalPlan`] would produce, parallel to [`describe_columns`].
///
/// Same plan coverage and column order as the names, so for a row-producing plan
/// `describe_columns(plan)` and `describe_column_types(plan)` line up element-for-element. The wire
/// `RowDescription` carries only names today; this exposes the per-column types the analyzer already
/// resolved so the protocol can pair them once it advertises types. Catalog-introspection
/// and `SHOW`/`EXPLAIN` shapes emit text columns.
#[must_use]
pub fn describe_column_types(plan: &PhysicalPlan) -> Vec<ColumnType> {
    match plan {
        PhysicalPlan::Select(op, _) => output_column_types(op),
        // `EXPLAIN`'s single "plan" column, `SHOW name`'s value, and `SHOW TABLES`' table name are
        // all text.
        PhysicalPlan::Explain(..) | PhysicalPlan::ShowVariable(_) | PhysicalPlan::ShowTables => {
            vec![ColumnType::Text]
        },
        // A set operation's row shape is its branches' UNIFIED typing;
        // the column NAMES still come from the leftmost branch.
        PhysicalPlan::SetOperation(p) => p.column_types.clone(),
        PhysicalPlan::Insert(p) => p.returning.iter().map(|r| r.expr.ty).collect(),
        PhysicalPlan::Update(p) => p.returning.iter().map(|r| r.expr.ty).collect(),
        PhysicalPlan::Delete(p) => p.returning.iter().map(|r| r.expr.ty).collect(),
        // `SHOW COLUMNS` reports column / type / nullable, all text.
        PhysicalPlan::ShowColumns(_) => {
            vec![ColumnType::Text, ColumnType::Text, ColumnType::Text]
        },
        _ => Vec::new(),
    }
}

/// A result-cache key: `(session user, plan rendering)`.
type ResultCacheKey = (String, String);

/// A cached query result: the data-version it was computed at, plus its output column names and rows.
type CachedQueryResult = (u64, Vec<String>, Vec<Row>);

/// The SQL-standard text for an isolation level, as `SHOW transaction_isolation` reports it.
const fn isolation_guc_text(level: IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "read uncommitted",
        IsolationLevel::ReadCommitted => "read committed",
        IsolationLevel::RepeatableRead => "repeatable read",
        IsolationLevel::Serializable => "serializable",
    }
}

/// Stateful executor across many statements. Holds at most one active
/// explicit transaction so `BEGIN ... statements ... COMMIT` runs every
/// intermediate statement inside that one transaction.
///
/// When no explicit transaction is open, statements auto-commit individually
/// (the same behavior the one-shot [`execute`] gives).
///
/// `Drop` best-effort-rolls back any still-open transaction so a panicking
/// caller does not leak a transaction in the engine.
pub struct Session<'engine> {
    engine: &'engine dyn StorageEngine,
    current_txn: Option<TxnId>,
    /// Default isolation for transactions this session starts, set by `SET TRANSACTION`.
    default_isolation: IsolationLevel,
    /// Default access mode (`true` = read-only) for transactions this session starts.
    default_read_only: bool,
    /// Whether the currently-active explicit transaction is read-only (set at `BEGIN`).
    txn_read_only: bool,
    /// Generic session-variable store for `SET`/`RESET`/`SHOW`. Variables are
    /// remembered and echoed back, and read by `current_setting(name)`.
    variables: HashMap<String, String>,
    /// The session user reported by `CURRENT_USER`/`SESSION_USER` and used to evaluate row-level
    /// security policies. Defaults to [`session_ctx::DEFAULT_USER`]; authentication does not yet
    /// feed a per-connection user into the SQL session (a documented follow-up).
    current_user: String,
    /// The current database name, reported by `CURRENT_DATABASE()`. A single-node NusaDB
    /// instance has one logical database, so this is a static label.
    current_database: String,
    /// The current schema name, reported by `CURRENT_SCHEMA()`. Defaults to `"public"`.
    current_schema: String,
    /// Statements prepared in this session via `PREPARE`, keyed by name. Re-analyzed and run
    /// on `EXECUTE` with the supplied arguments bound to their `$n` placeholders.
    prepared: HashMap<String, PreparedStatement>,
    /// Result cache: maps `(user, plan)` to the data-version it was computed at plus its
    /// rows. A cached entry is served only in auto-commit, only while the engine's `data_version` is
    /// unchanged (any committed write invalidates every entry), and only for non-volatile plans —
    /// keeping it from ever serving stale, snapshot-wrong, cross-user, or non-deterministic results.
    result_cache: HashMap<ResultCacheKey, CachedQueryResult>,
}

/// A statement stored by `PREPARE`: the un-analyzed body plus its placeholder count.
struct PreparedStatement {
    statement: ast::Statement,
    param_count: usize,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Skip the `&dyn StorageEngine` field — it has no `Debug` impl on the
        // trait object and printing engine state is not the point here.
        f.debug_struct("Session")
            .field("current_txn", &self.current_txn)
            .finish_non_exhaustive()
    }
}

/// Largest row set that the result cache will store — a memory bound.
const MAX_CACHE_ROWS: usize = 10_000;

/// Most entries the result cache holds before it is cleared — a memory bound. A true LRU is
/// a follow-up; clearing on overflow stays correct (entries are only ever served on a version match).
const MAX_CACHE_ENTRIES: usize = 256;

/// Markers in a plan's debug rendering that make its result non-cacheable: volatile or
/// session-dependent built-ins (`NOW`, `RANDOM`, `CURRENT_USER`, …) and user-defined functions (which
/// may be non-deterministic). These are the exact-case [`ast::ScalarFunc`] variant names plus
/// `ScalarUdf`, so a genuine volatile call is never missed; an incidental match (e.g. a quoted
/// identifier) only skips caching, which is always safe.
const VOLATILE_PLAN_MARKERS: [&str; 18] = [
    "ScalarUdf",
    // Sequence built-ins advance / read engine + session state per call, so a memoized result would
    // hand out a stale (or duplicate) value — never cache a plan that mentions one.
    "SequenceNext",
    "SequenceCurrent",
    "SequenceSet",
    "Now",
    "CurrentTimestamp",
    "CurrentDate",
    "CurrentTime",
    "CurrentUser",
    "SessionUser",
    "CurrentSetting",
    "Random",
    "Setseed",
    "UuidGenerateV4",
    "Version",
    "CurrentDatabase",
    "CurrentSchema",
    // `AGE(ts)` (one-argument) is relative to the current date, so its result must not be cached
    // across days (the two-argument form is deterministic, but the plan rendering cannot distinguish
    // them by name; skipping the cache for both is safe). The CamelCase `Age` is case-sensitive, so a
    // lowercase column named `age` (or words like `average`/`page`) never collides (deep-gate).
    "Age",
];

/// Whether a plan's debug rendering mentions a volatile/non-deterministic call, so its result must not
/// be cached.
fn plan_is_volatile(rendered: &str) -> bool {
    VOLATILE_PLAN_MARKERS
        .iter()
        .any(|marker| rendered.contains(marker))
}

impl<'engine> Session<'engine> {
    /// Open a new session with no active transaction.
    pub fn new(engine: &'engine dyn StorageEngine) -> Self {
        Self {
            engine,
            current_txn: None,
            default_isolation: IsolationLevel::default(),
            default_read_only: false,
            txn_read_only: false,
            variables: HashMap::new(),
            current_user: session_ctx::DEFAULT_USER.to_owned(),
            current_database: "nusadb".to_owned(),
            current_schema: "public".to_owned(),
            prepared: HashMap::new(),
            result_cache: HashMap::new(),
        }
    }

    /// Whether the session is inside an explicit `BEGIN ... COMMIT/ROLLBACK`.
    #[must_use]
    pub const fn in_transaction(&self) -> bool {
        self.current_txn.is_some()
    }

    /// The session's open explicit transaction, or `None` in auto-commit.
    ///
    /// A caller that analyzes a statement *before* handing the plan to [`execute`](Self::execute)
    /// must resolve schema names under this transaction's snapshot (via the engine's
    /// `lookup_table_as_of`) when it is `Some`, so DDL done earlier in the same explicit transaction
    /// (`BEGIN; CREATE TABLE t; INSERT INTO t ...`) is visible to the later statements. With `None`
    /// (auto-commit) the latest-committed view (`lookup_table`) is correct, since each prior
    /// statement committed. The wire server threads its per-statement transaction the same way
    /// (see `EngineCatalog`).
    #[must_use]
    pub const fn current_txn(&self) -> Option<TxnId> {
        self.current_txn
    }

    /// The session user reported by `CURRENT_USER`/`SESSION_USER`.
    #[must_use]
    pub fn current_user(&self) -> &str {
        &self.current_user
    }

    /// Set the session user reported by `CURRENT_USER`/`SESSION_USER` (and used for row-level
    /// security). The wire layer calls this once it authenticates a connection; tests use it to run
    /// statements as a chosen user.
    pub fn set_current_user(&mut self, user: impl Into<String>) {
        self.current_user = user.into();
    }

    /// The number of entries currently in the result cache — for monitoring and tests.
    #[must_use]
    pub fn result_cache_len(&self) -> usize {
        self.result_cache.len()
    }

    /// The current database name, reported by `CURRENT_DATABASE()`.
    #[must_use]
    pub fn current_database(&self) -> &str {
        &self.current_database
    }

    /// The current schema name, reported by `CURRENT_SCHEMA()`.
    #[must_use]
    pub fn current_schema(&self) -> &str {
        &self.current_schema
    }

    /// The session's ordered `search_path` schemas — the schemas an unqualified name resolves
    /// through, derived from `SET search_path` (always ending in `public`).
    #[must_use]
    pub fn search_path(&self) -> Vec<String> {
        crate::search_path_schemas(self.variables.get("search_path").map(String::as_str))
    }

    /// Execute one plan in this session's transaction context.
    pub fn execute(&mut self, plan: PhysicalPlan) -> Result<ExecutionResult, Error> {
        // A read-only `SELECT` in auto-commit may be served from / stored in the result cache.
        if matches!(plan, PhysicalPlan::Select(..)) && self.current_txn.is_none() {
            return self.execute_select_cached(plan);
        }
        match plan {
            PhysicalPlan::BeginTransaction(c) => self.begin(c),
            PhysicalPlan::Commit => self.commit(),
            PhysicalPlan::Rollback => self.rollback(),
            PhysicalPlan::SetTransaction(c) => self.set_transaction(c),
            PhysicalPlan::Savepoint(name) => self.savepoint(&name),
            PhysicalPlan::RollbackToSavepoint(name) => self.rollback_to_savepoint(&name),
            PhysicalPlan::ReleaseSavepoint(name) => self.release_savepoint(&name),
            PhysicalPlan::SetVariable { name, value } => self.set_variable(name, value),
            PhysicalPlan::ShowVariable(name) => Ok(self.show_variable(&name)),
            // Prepared-statement control needs the session's statement store.
            PhysicalPlan::Prepare {
                name,
                statement,
                param_count,
            } => {
                self.prepared.insert(
                    name,
                    PreparedStatement {
                        statement: *statement,
                        param_count,
                    },
                );
                Ok(ExecutionResult::Prepared)
            },
            PhysicalPlan::Execute { name, args } => self.execute_prepared(&name, &args),
            PhysicalPlan::Deallocate(target) => {
                match target {
                    ast::DeallocateTarget::All => self.prepared.clear(),
                    ast::DeallocateTarget::Name(name) => {
                        if self.prepared.remove(&name).is_none() {
                            return Err(Error::Unsupported(format!(
                                "prepared statement \"{name}\" does not exist"
                            )));
                        }
                    },
                }
                Ok(ExecutionResult::Deallocated)
            },
            other => self.run_within_txn(other),
        }
    }

    /// Execute an auto-commit `SELECT`, consulting the result cache.
    ///
    /// Safe by construction: caching is enabled only when the engine reports a `data_version` (so any
    /// committed write bumps it and invalidates every entry); a cached row set is reused only while
    /// that version is unchanged; the cache key includes the session user (so row-level security can
    /// never leak across users); and a plan whose rendering mentions a volatile/non-deterministic
    /// built-in or UDF is never cached (so `NOW()`/`RANDOM()`/`CURRENT_USER`/… are always recomputed).
    fn execute_select_cached(&mut self, plan: PhysicalPlan) -> Result<ExecutionResult, Error> {
        // Pin the session context up front (run_within_txn re-pins identically on a miss) so the
        // cache-hit path's work_mem enforcement below reads THIS statement's session settings —
        // not whatever the previous statement on this thread pinned.
        session_ctx::set_session_context(
            &self.current_user,
            &self.variables,
            &self.current_database,
            &self.current_schema,
        );
        let Some(version) = self.engine.data_version() else {
            // The engine does not track a version, so caching cannot be invalidated safely.
            return self.run_within_txn(plan);
        };
        let rendered = format!("{plan:?}");
        if plan_is_volatile(&rendered) {
            return self.run_within_txn(plan);
        }
        let key = (self.current_user.clone(), rendered);
        if let Some((cached_version, columns, rows)) = self.result_cache.get(&key)
            && *cached_version == version
        {
            // The work-memory budget bounds a materialized result regardless of where it came from:
            // a cached row set still occupies memory when served, so enforce it here too — otherwise
            // a query that would fail the budget on a cold run would succeed on a cache hit.
            ops::enforce_work_mem(rows)?;
            return Ok(ExecutionResult::Rows {
                columns: columns.clone(),
                rows: rows.clone(),
            });
        }
        let result = self.run_within_txn(plan)?;
        if let ExecutionResult::Rows { columns, rows } = &result
            && rows.len() <= MAX_CACHE_ROWS
        {
            // Bound memory: a simple cap with a clear-on-full policy (a refinement to LRU is a
            // follow-up). Stale entries (older `version`) are simply never served.
            if self.result_cache.len() >= MAX_CACHE_ENTRIES {
                self.result_cache.clear();
            }
            self.result_cache
                .insert(key, (version, columns.clone(), rows.clone()));
        }
        Ok(result)
    }

    /// Run a previously `PREPARE`d statement with `args` bound to its `$1..$n` placeholders.
    /// The bound statement is re-analyzed and executed in this session's transaction context, with
    /// the session user driving row-level security (so EXECUTE cannot bypass it).
    fn execute_prepared(&self, name: &str, args: &[ast::Value]) -> Result<ExecutionResult, Error> {
        let (statement, param_count) = {
            let prepared = self.prepared.get(name).ok_or_else(|| {
                Error::Unsupported(format!("prepared statement \"{name}\" does not exist"))
            })?;
            (prepared.statement.clone(), prepared.param_count)
        };
        if args.len() != param_count {
            return Err(Error::Unsupported(format!(
                "prepared statement \"{name}\" expects {param_count} parameter(s), got {}",
                args.len()
            )));
        }
        let bound = crate::params::substitute_values(statement, args)?;
        self.run_substituted(bound)
    }

    /// Analyze and execute a bound statement (from `EXECUTE`) in this session's transaction context,
    /// mirroring [`run_within_txn`](Self::run_within_txn) but performing analysis inside the
    /// transaction with a user-aware catalog (RLS-correct) and enforcing READ ONLY.
    fn run_substituted(&self, stmt: ast::Statement) -> Result<ExecutionResult, Error> {
        session_ctx::set_session_context(
            &self.current_user,
            &self.variables,
            &self.current_database,
            &self.current_schema,
        );
        if let Some(txn) = self.current_txn {
            // Refresh the READ COMMITTED statement snapshot.
            self.engine.begin_statement(txn)?;
            let read_only = self.txn_read_only;
            self.analyze_plan_dispatch(stmt, txn, read_only)
        } else {
            let txn = self.engine.begin(self.default_isolation)?;
            match self.analyze_plan_dispatch(stmt, txn, self.default_read_only) {
                Ok(result) => match self.engine.commit(txn) {
                    Ok(()) => Ok(result),
                    // A failed auto-commit fsync must roll the transaction back, not leak it
                    // Otherwise its view pins purge forever and its locks
                    // are never released.
                    Err(e) => {
                        let _ = self.engine.rollback(txn);
                        Err(e.into())
                    },
                },
                Err(err) => {
                    let _ = self.engine.rollback(txn);
                    Err(err)
                },
            }
        }
    }

    /// Analyze `stmt` under `txn` (with the session-user catalog for RLS), enforce READ ONLY, then
    /// dispatch the resulting plan.
    fn analyze_plan_dispatch(
        &self,
        stmt: ast::Statement,
        txn: TxnId,
        read_only: bool,
    ) -> Result<ExecutionResult, Error> {
        let catalog = SessionCatalog {
            engine: self.engine,
            txn,
            user: &self.current_user,
            search_path: self.search_path(),
        };
        let logical = crate::analyze(stmt, &catalog)?;
        let physical = crate::plan(logical);
        if read_only && plan_modifies_data(&physical) {
            return Err(Error::Unsupported(
                "cannot execute a data-modifying statement in a READ ONLY transaction".to_owned(),
            ));
        }
        dispatch(physical, self.engine, txn)
    }

    /// Execute one plan, delivering any output rows to `sink` one at a time instead of collecting
    /// them into a `Vec` (Phase 2 streaming output).
    ///
    /// A plain `SELECT` streams row-by-row through `stream_op` (bounded to a
    /// single row for a linear pipeline; a blocking top operator still materializes once but spills
    /// under `work_mem`). Every other statement runs through the ordinary [`execute`](Self::execute)
    /// path and, if it produced rows, replays them into the sink — so the observable result is
    /// identical, only the memory profile of a large `SELECT` differs.
    ///
    /// Transaction handling matches `execute`: inside an explicit transaction the statement runs in
    /// it (no commit); in auto-commit a one-statement transaction is begun, committed once the sink
    /// has consumed every row, and rolled back if the sink or execution errors.
    ///
    /// # Errors
    /// Propagates planning/storage/evaluation errors and any error returned by `sink`.
    pub fn execute_streaming(
        &mut self,
        plan: PhysicalPlan,
        sink: &mut dyn RowSink,
    ) -> Result<StreamOutcome, Error> {
        // Only a plain SELECT can stream; route everything else through the buffered path and replay
        // any rows into the sink so behaviour is identical to `execute`.
        if let PhysicalPlan::Select(op, _est) = plan {
            return self.stream_select(&op, sink);
        }
        // Capture the row shape before the plan is consumed so a replayed RETURNING set is typed.
        let types = describe_column_types(&plan);
        replay_into_sink(self.execute(plan)?, &types, sink)
    }

    /// Stream a `SELECT`'s rows into `sink` under the session's transaction context (auto-commit
    /// begins a one-statement transaction and commits only after every row is consumed). A `SELECT`
    /// never modifies data, so the READ ONLY guard does not apply.
    fn stream_select(
        &self,
        op: &PhysicalOperator,
        sink: &mut dyn RowSink,
    ) -> Result<StreamOutcome, Error> {
        // Pin the session context + wall clock for the statement, exactly as `run_within_txn` /
        // `dispatch` do for the buffered path.
        session_ctx::set_session_context(
            &self.current_user,
            &self.variables,
            &self.current_database,
            &self.current_schema,
        );
        clock::set_statement_now();
        if let Some(txn) = self.current_txn {
            // Refresh the READ COMMITTED statement snapshot so this SELECT's reads are consistent.
            self.engine.begin_statement(txn)?;
            stream_select_rows(op, self.engine, txn, sink)
        } else {
            let txn = self.engine.begin(self.default_isolation)?;
            match stream_select_rows(op, self.engine, txn, sink) {
                Ok(outcome) => match self.engine.commit(txn) {
                    Ok(()) => Ok(outcome),
                    // A failed auto-commit must roll the transaction back, not leak it.
                    Err(e) => {
                        let _ = self.engine.rollback(txn);
                        Err(e.into())
                    },
                },
                Err(err) => {
                    let _ = self.engine.rollback(txn);
                    Err(err)
                },
            }
        }
    }

    /// `SET name = value` stores the variable; `RESET name` (value `None`) clears it.
    fn set_variable(
        &mut self,
        name: String,
        value: Option<String>,
    ) -> Result<ExecutionResult, Error> {
        // `work_mem` feeds the executor's budget checks, so an unparseable value is rejected here
        // at SET time instead of being stored and then silently ignored by
        // `effective_work_mem`'s fallback.
        if name.eq_ignore_ascii_case("work_mem")
            && let Some(v) = &value
            && ops::parse_work_mem(v).is_none()
        {
            return Err(Error::Coded {
                message: format!(
                    "invalid value for parameter \"work_mem\": {v:?} — expected an integer with \
                     an optional kB/MB/GB/TB unit (a bare integer is kilobytes; 0 = unlimited)"
                ),
                sqlstate: "22023", // invalid_parameter_value
            });
        }
        // `statement_timeout` arms the wire server's per-statement cancel timer; same loud
        // SET-time rejection so a typo cannot silently disable the timeout.
        if name.eq_ignore_ascii_case("statement_timeout")
            && let Some(v) = &value
            && crate::cancel::parse_statement_timeout(v).is_none()
        {
            return Err(Error::Coded {
                message: format!(
                    "invalid value for parameter \"statement_timeout\": {v:?} — expected an \
                     integer with an optional us/ms/s/min/h/d unit (a bare integer is \
                     milliseconds; 0 = no timeout)"
                ),
                sqlstate: "22023", // invalid_parameter_value
            });
        }
        let is_search_path = name == "search_path";
        match value {
            Some(v) => {
                self.variables.insert(name, v);
            },
            None => {
                self.variables.remove(&name);
            },
        }
        // `SET search_path = …` / `RESET search_path` moves the session's current schema: an
        // unqualified name is created in, and resolved through, this schema (then `public`).
        if is_search_path {
            self.current_schema = crate::current_schema_for_search_path(
                self.variables.get("search_path").map(String::as_str),
            );
        }
        Ok(ExecutionResult::VariableSet)
    }

    /// `SHOW name` reports the variable's current value as a single row. An explicit `SET` wins;
    /// otherwise a well-known read-only/session GUC reports an honest built-in default (so e.g.
    /// `SHOW server_version` / `SHOW transaction_isolation` return useful values to client tooling
    /// instead of an empty string); any other unset variable still reads back as the empty string.
    fn show_variable(&self, name: &str) -> ExecutionResult {
        let value = self
            .variables
            .get(name)
            .cloned()
            .or_else(|| self.builtin_guc_default(name))
            .unwrap_or_default();
        ExecutionResult::Rows {
            columns: vec![name.to_owned()],
            rows: vec![vec![ast::Value::Text(value)]],
        }
    }

    /// Built-in default for a well-known GUC that the user has not explicitly `SET`. Names arrive
    /// folded (lower-case) from the parser. `transaction_isolation` reflects the session's current
    /// default level (resolved here); every other default is a shared constant resolved by
    /// [`session_ctx::builtin_guc_static_default`] (so `SHOW` and `current_setting` agree). Returns
    /// `None` for an unknown variable, leaving the empty-string fallback in place.
    fn builtin_guc_default(&self, name: &str) -> Option<String> {
        if matches!(
            name,
            "transaction_isolation" | "default_transaction_isolation"
        ) {
            return Some(isolation_guc_text(self.default_isolation).to_owned());
        }
        session_ctx::builtin_guc_static_default(name).map(ToOwned::to_owned)
    }

    fn savepoint(&self, name: &str) -> Result<ExecutionResult, Error> {
        let txn = self.active_txn("SAVEPOINT")?;
        self.engine.savepoint(txn, name)?;
        Ok(ExecutionResult::SavepointCreated)
    }

    fn rollback_to_savepoint(&self, name: &str) -> Result<ExecutionResult, Error> {
        let txn = self.active_txn("ROLLBACK TO SAVEPOINT")?;
        self.engine.rollback_to(txn, name)?;
        Ok(ExecutionResult::RolledBackToSavepoint)
    }

    fn release_savepoint(&self, name: &str) -> Result<ExecutionResult, Error> {
        let txn = self.active_txn("RELEASE SAVEPOINT")?;
        self.engine.release_savepoint(txn, name)?;
        Ok(ExecutionResult::SavepointReleased)
    }

    /// The active explicit transaction, or an error naming the `stmt` that requires one.
    /// Savepoints only make sense inside `BEGIN ... COMMIT/ROLLBACK`.
    fn active_txn(&self, stmt: &str) -> Result<TxnId, Error> {
        self.current_txn
            .ok_or_else(|| Error::Unsupported(format!("{stmt} without an active transaction")))
    }

    fn begin(&mut self, characteristics: TxnCharacteristics) -> Result<ExecutionResult, Error> {
        if self.current_txn.is_some() {
            return Err(Error::Unsupported(
                "nested BEGIN — already inside a transaction".to_owned(),
            ));
        }
        // Explicit BEGIN characteristics win; otherwise fall back to the session defaults.
        let isolation = characteristics.isolation.unwrap_or(self.default_isolation);
        let read_only = characteristics.read_only.unwrap_or(self.default_read_only);
        let txn = self.engine.begin(isolation)?;
        self.current_txn = Some(txn);
        self.txn_read_only = read_only;
        Ok(ExecutionResult::TransactionBegun)
    }

    fn set_transaction(
        &mut self,
        characteristics: TxnCharacteristics,
    ) -> Result<ExecutionResult, Error> {
        // The engine fixes a transaction's isolation/access mode at BEGIN, so SET TRANSACTION can
        // only configure transactions started later — matching the SQL rule that it precedes the
        // transaction's first statement. Reject it inside an active transaction rather than
        // silently ignoring it.
        if self.current_txn.is_some() {
            return Err(Error::Unsupported(
                "SET TRANSACTION must run before the transaction's first statement; \
                 characteristics are fixed at BEGIN"
                    .to_owned(),
            ));
        }
        if let Some(isolation) = characteristics.isolation {
            self.default_isolation = isolation;
        }
        if let Some(read_only) = characteristics.read_only {
            self.default_read_only = read_only;
        }
        Ok(ExecutionResult::TransactionCharacteristicsSet)
    }

    fn commit(&mut self) -> Result<ExecutionResult, Error> {
        let txn = self
            .current_txn
            .take()
            .ok_or_else(|| Error::Unsupported("COMMIT without an active transaction".to_owned()))?;
        // A failed COMMIT (e.g. a failed group fsync) must NOT leak the transaction: roll it back so
        // the engine releases it and its locks and purge is not blocked forever
        // `current_txn` is already cleared, so the session returns to
        // auto-commit either way.
        if let Err(e) = self.engine.commit(txn) {
            let _ = self.engine.rollback(txn);
            self.txn_read_only = false;
            return Err(e.into());
        }
        self.txn_read_only = false;
        Ok(ExecutionResult::TransactionCommitted)
    }

    fn rollback(&mut self) -> Result<ExecutionResult, Error> {
        let txn = self.current_txn.take().ok_or_else(|| {
            Error::Unsupported("ROLLBACK without an active transaction".to_owned())
        })?;
        self.engine.rollback(txn)?;
        self.txn_read_only = false;
        Ok(ExecutionResult::TransactionRolledBack)
    }

    fn run_within_txn(&self, plan: PhysicalPlan) -> Result<ExecutionResult, Error> {
        // Pin this session's user and settings for the statement so every CURRENT_USER /
        // SESSION_USER / current_setting() it contains observes the same snapshot (statement
        // stability, mirroring the wall clock).
        session_ctx::set_session_context(
            &self.current_user,
            &self.variables,
            &self.current_database,
            &self.current_schema,
        );
        // Enforce READ ONLY: reject data-modifying statements. Inside an explicit
        // transaction the flag is the one captured at BEGIN; in auto-commit it is the session
        // default carried by `SET TRANSACTION`.
        let read_only = if self.current_txn.is_some() {
            self.txn_read_only
        } else {
            self.default_read_only
        };
        if read_only && plan_modifies_data(&plan) {
            return Err(Error::Unsupported(
                "cannot execute a data-modifying statement in a READ ONLY transaction".to_owned(),
            ));
        }
        if let Some(txn) = self.current_txn {
            // Inside an explicit transaction: refresh the READ COMMITTED statement snapshot so
            // every read in this statement sees ONE consistent view, then
            // run the statement, do *not* commit. On error the transaction stays open so the caller
            // can choose ROLLBACK (matching standard SQL semantics).
            self.engine.begin_statement(txn)?;
            dispatch(plan, self.engine, txn)
        } else {
            // Auto-commit: one transaction per statement, at the session default isolation — its
            // `begin` already takes a fresh snapshot, so it needs no `begin_statement`.
            let txn = self.engine.begin(self.default_isolation)?;
            match dispatch(plan, self.engine, txn) {
                Ok(result) => match self.engine.commit(txn) {
                    Ok(()) => Ok(result),
                    // A failed auto-commit fsync must roll the transaction back, not leak it
                    // Otherwise its view pins purge forever and its locks
                    // are never released.
                    Err(e) => {
                        let _ = self.engine.rollback(txn);
                        Err(e.into())
                    },
                },
                Err(err) => {
                    let _ = self.engine.rollback(txn);
                    Err(err)
                },
            }
        }
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        if let Some(txn) = self.current_txn.take() {
            // Best-effort rollback: a session dropped mid-transaction would
            // otherwise leak the txn inside the engine.
            let _ = self.engine.rollback(txn);
        }
    }
}

/// Whether a plan modifies data or schema — used to enforce READ ONLY transactions.
///
/// Only pure reads are permitted: `SELECT`, set operations, and `EXPLAIN` (which formats a plan
/// without executing its side effects). Everything else (DML, DDL, `VACUUM`, `ANALYZE`,
/// `COMMENT`) counts as a write.
const fn plan_modifies_data(plan: &PhysicalPlan) -> bool {
    // `LOCK TABLE` acquires a lock but does not modify data, so it is permitted in a READ ONLY
    // transaction.
    !matches!(
        plan,
        PhysicalPlan::Select(..)
            | PhysicalPlan::SetOperation(_)
            | PhysicalPlan::Explain(..)
            | PhysicalPlan::LockTable { .. }
    )
}

/// Stream a `SELECT` operator's output rows into `sink` (Phase 2). Emits the column names once,
/// then pulls rows one at a time from [`stream_op`](stream::stream_op) — which streams the linear
/// pipeline truly and materializes only an inherently blocking top operator (which spills under
/// `work_mem`). Yields exactly the rows the buffered `run_select` would, in the same order.
///
/// # Errors
/// Propagates evaluation/storage errors from producing rows and any error returned by `sink`.
fn stream_select_rows(
    op: &PhysicalOperator,
    engine: &dyn StorageEngine,
    txn: TxnId,
    sink: &mut dyn RowSink,
) -> Result<StreamOutcome, Error> {
    let columns = output_columns(op);
    sink.columns_typed(&columns, &output_column_types(op))?;
    let mut source = stream::stream_op(op, engine, txn)?;
    let mut count = 0;
    while let Some(row) = source.try_next()? {
        sink.row(&row)?;
        count += 1;
    }
    Ok(StreamOutcome::Rows { columns, count })
}

fn dispatch(
    plan: PhysicalPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // Pin the wall clock for this statement so every NOW()/CURRENT_TIMESTAMP/CURRENT_DATE/
    // CURRENT_TIME it contains observes the same instant (SQL statement stability).
    clock::set_statement_now();
    match plan {
        // The multi-object DDL desugar: children run in order within this same statement
        // transaction (so `DROP TABLE a, b` is atomic); the first error aborts the rest and
        // the last child's result is the statement's.
        PhysicalPlan::Batch(children) => {
            let mut last = None;
            for child in children {
                last = Some(dispatch(child, engine, txn)?);
            }
            last.ok_or_else(|| Error::Unsupported("internal: empty statement batch".to_owned()))
        },
        PhysicalPlan::CreateTable(p) => run_create_table(p, engine, txn),
        PhysicalPlan::CreateTableAs(p) => run_create_table_as(p, engine, txn),
        PhysicalPlan::DropTable(p) => run_drop_table(&p, engine, txn),
        PhysicalPlan::CreateMaterializedView(p) => run_create_materialized_view(p, engine, txn),
        PhysicalPlan::CreateView(p) => run_create_view(&p, engine, txn),
        PhysicalPlan::DropView(p) => run_drop_view(&p, engine, txn),
        PhysicalPlan::CreateEnum(p) => run_create_enum(&p, engine, txn),
        PhysicalPlan::DropType(p) => run_drop_type(&p, engine, txn),
        PhysicalPlan::CreateTrigger(p) => trigger::run_create_trigger(&p, engine, txn),
        PhysicalPlan::DropTrigger(p) => trigger::run_drop_trigger(&p, engine, txn),
        PhysicalPlan::AlterTrigger(p) => trigger::run_alter_trigger(&p, engine, txn),
        PhysicalPlan::CreateProcedure(p) => procedure::run_create_procedure(&p, engine, txn),
        PhysicalPlan::DropProcedure(p) => procedure::run_drop_procedure(&p, engine, txn),
        PhysicalPlan::Call(p) => procedure::run_call(&p, engine, txn),
        PhysicalPlan::CreateFunction(p) => function::run_create_function(&p, engine, txn),
        PhysicalPlan::DropFunction(p) => function::run_drop_function(&p, engine, txn),
        PhysicalPlan::RefreshMaterializedView(name) => {
            run_refresh_materialized_view(&name, engine, txn)
        },
        PhysicalPlan::CreatePolicy(p) => run_create_policy(&p, engine, txn),
        PhysicalPlan::DropPolicy(p) => run_drop_policy(&p, engine, txn),
        PhysicalPlan::AlterTable(p) => run_alter_table(p, engine, txn),
        PhysicalPlan::Insert(p) => run_insert(&p, engine, txn),
        PhysicalPlan::Select(op, est_scan_rows) => run_select(&op, est_scan_rows, engine, txn),
        PhysicalPlan::Update(p) => run_update(&p, engine, txn),
        PhysicalPlan::Delete(p) => run_delete(&p, engine, txn),
        PhysicalPlan::Merge(p) => dml::run_merge(&p, engine, txn),
        PhysicalPlan::Explain(inner, options) => run_explain(&inner, options, engine, txn),
        PhysicalPlan::Vacuum(options) => run_vacuum(options, engine, txn),
        // REINDEX is a no-op: NusaDB's B-tree indexes are always consistent (MVCC + purge).
        PhysicalPlan::Reindex => Ok(ExecutionResult::Reindexed),
        PhysicalPlan::LockTable { tables, mode } => run_lock_table(&tables, mode, engine, txn),
        PhysicalPlan::ShowTables => run_show_tables(engine, txn),
        PhysicalPlan::ShowColumns(schema) => Ok(run_show_columns(&schema)),
        PhysicalPlan::Analyze(p) => run_analyze(p, engine, txn),
        PhysicalPlan::CreateSchema(p) => run_create_schema(&p, engine, txn),
        PhysicalPlan::DropSchema(p) => run_drop_schema(&p, engine, txn),
        PhysicalPlan::CreateDatabase(_) => Ok(ExecutionResult::DatabaseCreated),
        PhysicalPlan::AlterDatabase(_) => Ok(ExecutionResult::DatabaseAltered),
        PhysicalPlan::DropDatabase(p) => run_drop_database(&p, engine, txn),
        PhysicalPlan::CreateSequence(p) => run_create_sequence(&p, engine, txn),
        PhysicalPlan::DropSequence(p) => run_drop_sequence(&p, engine, txn),
        PhysicalPlan::CreateIndex(p) => run_create_index(&p, engine, txn),
        PhysicalPlan::DropIndex(p) => run_drop_index(&p, engine, txn),
        PhysicalPlan::SetOperation(p) => run_set_operation(&p, engine, txn),
        // The analyzer already resolved the target's existence. Persisting the comment in catalog
        // metadata is optional treaty work (DoD), so accept it as a metadata no-op for now.
        PhysicalPlan::Comment(_) => Ok(ExecutionResult::Commented),
        PhysicalPlan::BeginTransaction(_)
        | PhysicalPlan::Commit
        | PhysicalPlan::Rollback
        | PhysicalPlan::SetTransaction(_)
        | PhysicalPlan::Savepoint(_)
        | PhysicalPlan::RollbackToSavepoint(_)
        | PhysicalPlan::ReleaseSavepoint(_)
        | PhysicalPlan::SetVariable { .. }
        | PhysicalPlan::ShowVariable(_)
        | PhysicalPlan::Prepare { .. }
        | PhysicalPlan::Execute { .. }
        | PhysicalPlan::Deallocate(_) => {
            // Transaction-, session-, and prepared-statement-control plans are intercepted by
            // `Session::execute` before reaching `dispatch` (the latter need the session's statement
            // store); this arm is defensive. PREPARE/EXECUTE/DEALLOCATE require a `Session`.
            Err(Error::Unsupported(
                "session-control plan reached executor dispatch (requires a Session)".to_owned(),
            ))
        },
    }
}

/// `LOCK TABLE name [, ...] [IN <mode> MODE]`: acquire a table-level lock on each resolved
/// table, held until the end of the transaction. In auto-commit the lock is released at the implicit
/// commit, so it only protects across statements inside an explicit `BEGIN ... COMMIT`.
///
/// # Errors
/// Propagates lock-acquisition errors (including a deadlock or a lock-manager rejection).
fn run_lock_table(
    tables: &[TableSchema],
    mode: ast::LockMode,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let mode = match mode {
        ast::LockMode::AccessShare => nusadb_core::engine::TableLockMode::AccessShare,
        ast::LockMode::AccessExclusive => nusadb_core::engine::TableLockMode::AccessExclusive,
    };
    for table in tables {
        engine.lock_table(txn, table.id, mode)?;
    }
    Ok(ExecutionResult::TableLocked)
}

/// `VACUUM [FULL] [ANALYZE]`: reclaim dead row versions across all tables, then — for
/// `ANALYZE` — recompute statistics for every user table. `FULL` currently behaves as the standard
/// reclaim (an aggressive rewrite/deep compaction is a follow-up); it is accepted so the surface is
/// complete. Returns the number of reclaimed versions.
///
/// # Errors
/// Propagates reclaim, table-listing, schema-lookup, and per-table `ANALYZE` errors.
fn run_vacuum(
    options: ast::VacuumOptions,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let reclaimed = engine.vacuum()?;
    if options.analyze {
        // Recompute statistics for every user table (skip internal/system tables, which carry the
        // reserved prefix and are not user-analyzable). Enumerate under this txn's snapshot so a table
        // committed by an earlier statement on the same connection is reliably included.
        for name in engine.list_tables_as_of(txn)? {
            if name.starts_with(crate::SYSTEM_TABLE_PREFIX) {
                continue;
            }
            if let Some(table) = engine.lookup_table_as_of(txn, &name)? {
                let columns: Vec<usize> = (0..table.columns.len()).collect();
                run_analyze(AnalyzePlan { table, columns }, engine, txn)?;
            }
        }
    }
    Ok(ExecutionResult::Vacuumed(reclaimed))
}

fn run_explain(
    plan: &PhysicalPlan,
    options: ast::ExplainOptions,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // Build a single-table cost context when the plan scans exactly one table
    // that has been analyzed; EXPLAIN then annotates each operator with its
    // estimated row count. Multi-table plans and un-analyzed tables format
    // without estimates (no `table_stats` → `ctx` stays `None`).
    let scan_table = single_scan_table(plan);
    let fetched = match scan_table {
        Some(table) => engine.table_stats(table.id)?,
        None => None,
    };
    let ctx = match (scan_table, &fetched) {
        (Some(schema), Some(stats)) => Some(cost::ScanStats::new(schema, stats)),
        _ => None,
    };

    // ANALYZE: actually run a *read-only* statement and report its real row count and total
    // wall time, alongside the estimated plan. A data-modifying statement is rejected rather than
    // executed (its side effects would commit before the EXPLAIN result is even returned).
    // A SELECT runs instrumented: every operator records the rows it actually produced,
    // keyed by its node's address in this same plan tree — so the run executes *this* tree (not a
    // clone) through the row-at-a-time path, whose surfaces carry the per-node counters (the
    // vectorized batch path does not).
    let (execution, actuals) = if options.analyze {
        if plan_modifies_data(plan) {
            return Err(Error::Unsupported(
                "EXPLAIN ANALYZE on a data-modifying statement is not supported; \
                 only read-only statements (SELECT / set operations) can be analyzed"
                    .to_owned(),
            ));
        }
        clock::set_statement_now();
        let started = std::time::Instant::now();
        let (actual_rows, actuals) = if let PhysicalPlan::Select(op, _est) = plan {
            let session = instrument::Session::begin();
            let rows = ops::execute_op(op, engine, txn)?;
            (rows.len(), Some(session.take()))
        } else {
            let actual_rows = match dispatch(plan.clone(), engine, txn)? {
                ExecutionResult::Rows { rows, .. } => rows.len(),
                _ => 0,
            };
            (actual_rows, None)
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        (Some((actual_rows, elapsed_ms)), actuals)
    } else {
        (None, None)
    };
    let plan_lines = format_plan(plan, 0, ctx.as_ref(), actuals.as_ref());

    // VERBOSE: the plan's output column names for a row-producing statement.
    let output_columns = if options.verbose {
        describe_columns(plan)
    } else {
        Vec::new()
    };

    let rows = match options.format {
        ast::ExplainFormat::Text => {
            let mut lines = plan_lines;
            if !output_columns.is_empty() {
                lines.push(format!("Output: {}", output_columns.join(", ")));
            }
            if let Some((actual_rows, elapsed_ms)) = execution {
                lines.push(format!(
                    "Execution: actual rows={actual_rows}, total time={elapsed_ms:.3} ms"
                ));
            }
            lines
                .into_iter()
                .map(|line| vec![ast::Value::Text(line)])
                .collect()
        },
        // Structured JSON: one row holding a pretty-printed document.
        ast::ExplainFormat::Json => {
            let mut obj = serde_json::Map::new();
            obj.insert("plan".to_owned(), plan_tree_json(&plan_lines));
            if !output_columns.is_empty() {
                obj.insert(
                    "output".to_owned(),
                    serde_json::Value::Array(
                        output_columns
                            .into_iter()
                            .map(serde_json::Value::String)
                            .collect(),
                    ),
                );
            }
            if let Some((actual_rows, elapsed_ms)) = execution {
                obj.insert(
                    "execution".to_owned(),
                    serde_json::json!({ "actual_rows": actual_rows, "total_time_ms": elapsed_ms }),
                );
            }
            let doc = serde_json::to_string_pretty(&serde_json::Value::Object(obj))
                .unwrap_or_else(|_| "{}".to_owned());
            vec![vec![ast::Value::Text(doc)]]
        },
    };
    Ok(ExecutionResult::Rows {
        columns: vec!["plan".to_owned()],
        rows,
    })
}

/// Build the `EXPLAIN (FORMAT JSON)` plan tree from the rendered text lines.
///
/// The text formatter emits exactly one self-line per operator at its tree depth, with children at
/// `depth + 1`, so the indentation faithfully encodes the tree: leading two-space groups give the
/// depth and the trimmed remainder is the node label. Reusing the rendered lines keeps the JSON in
/// lock-step with the text plan (no second formatter to drift). Each node becomes
/// `{ "node": <label>, "children": [ … ] }`; an empty plan yields `null`.
fn plan_tree_json(lines: &[String]) -> serde_json::Value {
    struct Node {
        label: String,
        children: Vec<usize>,
    }

    fn build(arena: &[Node], i: usize) -> serde_json::Value {
        let Some(node) = arena.get(i) else {
            return serde_json::Value::Null;
        };
        serde_json::json!({
            "node": node.label,
            "children": node
                .children
                .iter()
                .map(|&c| build(arena, c))
                .collect::<Vec<_>>(),
        })
    }

    let mut arena: Vec<Node> = Vec::with_capacity(lines.len());
    // `stack[d]` is the index of the current ancestor at depth `d`.
    let mut stack: Vec<usize> = Vec::new();
    for line in lines {
        let depth = line.bytes().take_while(|b| *b == b' ').count() / 2;
        let idx = arena.len();
        arena.push(Node {
            label: line.trim_start().to_owned(),
            children: Vec::new(),
        });
        stack.truncate(depth);
        if let Some(&parent) = stack.last()
            && let Some(node) = arena.get_mut(parent)
        {
            node.children.push(idx);
        }
        stack.push(idx);
    }

    if arena.is_empty() {
        serde_json::Value::Null
    } else {
        build(&arena, 0)
    }
}

/// The scanned-row count at or above which a supported single-table SELECT is routed to the
/// vectorized batch path by default (selective routing, per the the design recommendation). The batch
/// path's columnar materialization only pays off at scale — measured a ~13–15% win at 100k rows and
/// parity at ~10k — so smaller scans stay on the
/// row path, where they are at least as fast. The metric is the *scanned* row count (the rows the
/// batch operators process), not the post-filter output, because that is what the per-batch overhead
/// is amortized over.
const VECTORIZED_MIN_ROWS: u64 = 50_000;

/// Whether a plan-time scanned-row estimate is large enough to route to the vectorized batch path.
/// Chooses only the *execution strategy* — the rows produced are identical either way, and
/// [`vectorized::execute`](crate::vectorized::execute) still falls back to the row path for any plan
/// shape it does not support.
pub(super) const fn meets_vectorize_threshold(est_scan_rows: u64) -> bool {
    est_scan_rows >= VECTORIZED_MIN_ROWS
}

/// The lone table a plan scans, or `None` when it scans zero or more than one
/// (joins) — the single-table case the cost context can resolve ordinals for.
fn single_scan_table(plan: &PhysicalPlan) -> Option<&TableSchema> {
    match plan {
        PhysicalPlan::Select(op, _) => {
            let mut found = None;
            let mut count = 0usize;
            collect_scan_tables(op, &mut found, &mut count);
            if count == 1 { found } else { None }
        },
        PhysicalPlan::Explain(inner, _) => single_scan_table(inner),
        _ => None,
    }
}

fn collect_scan_tables<'a>(
    op: &'a PhysicalOperator,
    found: &mut Option<&'a TableSchema>,
    count: &mut usize,
) {
    match op {
        PhysicalOperator::SeqScan { table, .. }
        | PhysicalOperator::IndexScan { table, .. }
        | PhysicalOperator::VectorKnn { table, .. } => {
            *count += 1;
            if found.is_none() {
                *found = Some(table);
            }
        },
        PhysicalOperator::OneRow
        | PhysicalOperator::InfoSchemaScan { .. }
        | PhysicalOperator::Values { .. }
        | PhysicalOperator::SetOperation(_) => {},
        PhysicalOperator::Filter { input, .. }
        | PhysicalOperator::Sort { input, .. }
        | PhysicalOperator::Project { input, .. }
        | PhysicalOperator::ProjectSet { input, .. }
        | PhysicalOperator::Limit { input, .. }
        | PhysicalOperator::Distinct { input }
        | PhysicalOperator::DistinctOn { input, .. }
        | PhysicalOperator::ScalarAggregate { input, .. }
        | PhysicalOperator::GroupAggregate { input, .. }
        | PhysicalOperator::GroupingSetsAggregate { input, .. }
        | PhysicalOperator::Window { input, .. }
        | PhysicalOperator::LockRows { input, .. } => {
            collect_scan_tables(input, found, count);
        },
        PhysicalOperator::NestedLoopJoin { left, right, .. }
        | PhysicalOperator::HashJoin { left, right, .. }
        | PhysicalOperator::LateralJoin { left, right, .. } => {
            collect_scan_tables(left, found, count);
            collect_scan_tables(right, found, count);
        },
        PhysicalOperator::WithRecursive { ctes, body } => {
            // Walk the body and every CTE term. A recursive term always scans its synthetic table,
            // so a recursive query reports more than one scan and never takes the single-table
            // cost path — which is correct: a synthetic CTE table has no `table_stats`.
            collect_scan_tables(body, found, count);
            for cte in ctes {
                collect_scan_tables(&cte.base, found, count);
                collect_scan_tables(&cte.recursive, found, count);
            }
        },
        // A data-modifying CTE's statement runs separately; only the body contributes outer scans.
        PhysicalOperator::WithModifying { body, .. } => collect_scan_tables(body, found, count),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive match arm per physical plan type — inherently long, but splitting it \
              would scatter the EXPLAIN formatting across many tiny helpers with no clarity gain"
)]
fn format_plan(
    plan: &PhysicalPlan,
    depth: usize,
    ctx: Option<&cost::ScanStats>,
    actuals: Option<&HashMap<usize, u64>>,
) -> Vec<String> {
    let indent = "  ".repeat(depth);
    match plan {
        PhysicalPlan::Batch(children) => children
            .iter()
            .flat_map(|c| format_plan(c, depth, ctx, actuals))
            .collect(),
        PhysicalPlan::CreateTable(p) => vec![format!("{indent}CreateTable: {}", p.table)],
        PhysicalPlan::CreateTableAs(p) => {
            let mut lines = vec![format!("{indent}CreateTableAs: {}", p.name)];
            lines.extend(format_op(&p.body, depth + 1, ctx, actuals));
            lines
        },
        PhysicalPlan::DropTable(p) => vec![format!("{indent}DropTable: {}", p.table)],
        PhysicalPlan::CreateMaterializedView(p) => {
            let mut lines = vec![format!("{indent}CreateMaterializedView: {}", p.name)];
            lines.extend(format_op(&p.body, depth + 1, ctx, actuals));
            lines
        },
        PhysicalPlan::CreateView(p) => vec![format!("{indent}CreateView: {}", p.name)],
        PhysicalPlan::DropView(p) => vec![format!("{indent}DropView: {}", p.name)],
        PhysicalPlan::CreateEnum(p) => vec![format!("{indent}CreateEnum: {}", p.name)],
        PhysicalPlan::DropType(p) => vec![format!("{indent}DropType: {}", p.name)],
        PhysicalPlan::CreateTrigger(p) => {
            vec![format!("{indent}CreateTrigger: {} ON {}", p.name, p.table)]
        },
        PhysicalPlan::AlterTrigger(p) => {
            vec![format!(
                "{indent}AlterTrigger: {} ON {} RENAME TO {}",
                p.name, p.table, p.new_name
            )]
        },
        PhysicalPlan::DropTrigger(p) => {
            vec![format!("{indent}DropTrigger: {} ON {}", p.name, p.table)]
        },
        PhysicalPlan::CreateProcedure(p) => {
            vec![format!("{indent}CreateProcedure: {}", p.name)]
        },
        PhysicalPlan::DropProcedure(p) => vec![format!("{indent}DropProcedure: {}", p.name)],
        PhysicalPlan::Call(p) => vec![format!("{indent}Call: {}", p.name)],
        PhysicalPlan::CreateFunction(p) => vec![format!("{indent}CreateFunction: {}", p.name)],
        PhysicalPlan::DropFunction(p) => vec![format!("{indent}DropFunction: {}", p.name)],
        PhysicalPlan::RefreshMaterializedView(name) => {
            vec![format!("{indent}RefreshMaterializedView: {name}")]
        },
        PhysicalPlan::CreatePolicy(p) => {
            vec![format!("{indent}CreatePolicy: {} ON {}", p.name, p.table)]
        },
        PhysicalPlan::DropPolicy(p) => {
            vec![format!("{indent}DropPolicy: {} ON {}", p.name, p.table)]
        },
        PhysicalPlan::AlterTable(p) => vec![format!("{indent}AlterTable{}", format_alter(p))],
        PhysicalPlan::Insert(p) => {
            let source = match &p.source {
                InsertSource::Values(rows) => format!("{} row(s)", rows.len()),
                InsertSource::Select(_) => "select".to_owned(),
            };
            vec![format!("{indent}Insert into {} ({source})", p.table.name)]
        },
        PhysicalPlan::Update(p) => vec![format!(
            "{indent}Update {} ({} assignment(s){})",
            p.table.name,
            p.assignments.len(),
            if p.filter.is_some() { ", filtered" } else { "" },
        )],
        PhysicalPlan::Merge(p) => vec![format!(
            "{indent}Merge into {} using {} ({} when-clause(s))",
            p.table.name,
            p.source.name,
            p.whens.len(),
        )],
        PhysicalPlan::Delete(p) => vec![format!(
            "{indent}Delete from {}{}",
            p.table.name,
            if p.filter.is_some() {
                " (filtered)"
            } else {
                ""
            },
        )],
        PhysicalPlan::Select(op, _) => format_op(op, depth, ctx, actuals),
        PhysicalPlan::Explain(inner, _) => {
            let mut lines = vec![format!("{indent}Explain")];
            lines.extend(format_plan(inner, depth + 1, ctx, actuals));
            lines
        },
        PhysicalPlan::BeginTransaction(_) => vec![format!("{indent}BeginTransaction")],
        PhysicalPlan::Commit => vec![format!("{indent}Commit")],
        PhysicalPlan::Rollback => vec![format!("{indent}Rollback")],
        PhysicalPlan::SetTransaction(_) => vec![format!("{indent}SetTransaction")],
        PhysicalPlan::Savepoint(name) => vec![format!("{indent}Savepoint {name}")],
        PhysicalPlan::RollbackToSavepoint(name) => {
            vec![format!("{indent}RollbackToSavepoint {name}")]
        },
        PhysicalPlan::ReleaseSavepoint(name) => vec![format!("{indent}ReleaseSavepoint {name}")],
        PhysicalPlan::SetVariable { name, .. } => vec![format!("{indent}SetVariable {name}")],
        PhysicalPlan::ShowVariable(name) => vec![format!("{indent}ShowVariable {name}")],
        PhysicalPlan::ShowTables => vec![format!("{indent}ShowTables")],
        PhysicalPlan::ShowColumns(schema) => {
            vec![format!("{indent}ShowColumns {}", schema.name)]
        },
        PhysicalPlan::Vacuum(options) => vec![format!(
            "{indent}Vacuum{}{}",
            if options.full { " FULL" } else { "" },
            if options.analyze { " ANALYZE" } else { "" }
        )],
        PhysicalPlan::Reindex => vec![format!("{indent}Reindex (no-op)")],
        PhysicalPlan::Analyze(p) => vec![format!(
            "{indent}Analyze: {} ({} column(s))",
            p.table.name,
            p.columns.len(),
        )],
        PhysicalPlan::LockTable { tables, mode } => {
            let names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
            let mode = match mode {
                crate::ast::LockMode::AccessShare => "ACCESS SHARE",
                crate::ast::LockMode::AccessExclusive => "ACCESS EXCLUSIVE",
            };
            vec![format!(
                "{indent}LockTable {} IN {mode} MODE",
                names.join(", ")
            )]
        },
        PhysicalPlan::Prepare { name, .. } => vec![format!("{indent}Prepare: {name}")],
        PhysicalPlan::Execute { name, args } => {
            vec![format!("{indent}Execute: {name} ({} arg(s))", args.len())]
        },
        PhysicalPlan::Deallocate(target) => match target {
            crate::ast::DeallocateTarget::All => vec![format!("{indent}Deallocate: ALL")],
            crate::ast::DeallocateTarget::Name(name) => {
                vec![format!("{indent}Deallocate: {name}")]
            },
        },
        PhysicalPlan::Comment(p) => {
            let target = p
                .column
                .as_ref()
                .map_or_else(|| p.table.clone(), |c| format!("{}.{c}", p.table));
            vec![format!("{indent}Comment: {target}")]
        },
        PhysicalPlan::CreateSchema(p) => vec![format!("{indent}CreateSchema: {}", p.name)],
        PhysicalPlan::DropSchema(p) => vec![format!("{indent}DropSchema: {}", p.name)],
        PhysicalPlan::CreateDatabase(p) => vec![format!("{indent}CreateDatabase: {}", p.name)],
        PhysicalPlan::AlterDatabase(p) => vec![format!("{indent}AlterDatabase: {}", p.name)],
        PhysicalPlan::DropDatabase(p) => {
            let how = if p.force { "force" } else { "backup-then-drop" };
            vec![format!("{indent}DropDatabase: {} ({how})", p.name)]
        },
        PhysicalPlan::CreateSequence(p) => {
            vec![format!("{indent}CreateSequence: {}", p.def.name)]
        },
        PhysicalPlan::DropSequence(p) => vec![format!("{indent}DropSequence: {}", p.name)],
        PhysicalPlan::CreateIndex(p) => vec![format!(
            "{indent}CreateIndex: {} on {}",
            p.def.name, p.def.table.0
        )],
        PhysicalPlan::DropIndex(p) => vec![format!("{indent}DropIndex: {}", p.name)],
        PhysicalPlan::SetOperation(p) => {
            let mut lines = vec![format!("{indent}SetOperation")];
            format_set_tree(&p.tree, depth + 1, &mut lines, actuals);
            lines
        },
    }
}

/// Format a set-operation tree for `EXPLAIN`.
fn format_set_tree(
    tree: &SetOpTree<PhysicalOperator>,
    depth: usize,
    lines: &mut Vec<String>,
    actuals: Option<&HashMap<usize, u64>>,
) {
    let indent = "  ".repeat(depth);
    match tree {
        SetOpTree::Leaf(op) => lines.extend(format_op(op, depth, None, actuals)),
        SetOpTree::Node {
            op,
            all,
            left,
            right,
        } => {
            let all = if *all { " ALL" } else { "" };
            lines.push(format!("{indent}{op:?}{all}"));
            format_set_tree(left, depth + 1, lines, actuals);
            format_set_tree(right, depth + 1, lines, actuals);
        },
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-operator EXPLAIN formatter; length tracks the operator set"
)]
fn format_op(
    op: &PhysicalOperator,
    depth: usize,
    ctx: Option<&cost::ScanStats>,
    actuals: Option<&HashMap<usize, u64>>,
) -> Vec<String> {
    let indent = "  ".repeat(depth);
    let mut lines = Vec::new();
    match op {
        PhysicalOperator::SeqScan { table, columns } => {
            // Show the projection-pushdown narrowing when present.
            if columns.is_empty() {
                lines.push(format!("{indent}SeqScan: {}", table.name));
            } else {
                lines.push(format!(
                    "{indent}SeqScan: {} (project {}/{} cols)",
                    table.name,
                    columns.len(),
                    table.columns.len()
                ));
            }
        },
        PhysicalOperator::VectorKnn { table, k, .. } => {
            lines.push(format!("{indent}VectorKnn: {} (k={k})", table.name));
        },
        PhysicalOperator::IndexScan { table, index, .. } => {
            lines.push(format!("{indent}IndexScan: {} using {index}", table.name));
        },
        PhysicalOperator::OneRow => lines.push(format!("{indent}OneRow")),
        PhysicalOperator::Values { rows } => {
            lines.push(format!("{indent}Values ({} rows)", rows.len()));
        },
        PhysicalOperator::SetOperation(set_op) => {
            lines.push(format!("{indent}SetOperation"));
            format_set_tree(&set_op.tree, depth + 1, &mut lines, actuals);
        },
        PhysicalOperator::InfoSchemaScan { view } => {
            lines.push(format!("{indent}InfoSchemaScan: {}", view.view_name()));
        },
        PhysicalOperator::Filter { input, .. } => {
            lines.push(format!("{indent}Filter"));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::LockRows {
            input, table, mode, ..
        } => {
            let strength = match mode {
                nusadb_core::engine::RowLockMode::Exclusive => "FOR UPDATE",
                nusadb_core::engine::RowLockMode::Shared => "FOR SHARE",
            };
            lines.push(format!("{indent}LockRows ({strength} on {})", table.name));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::Sort {
            input,
            keys,
            limit_ties,
            top_n,
        } => {
            let suffix = if limit_ties.is_some() {
                " with ties".to_owned()
            } else if let Some(m) = top_n {
                // Limit-aware top-N: a bounded selection, not a full sort.
                format!(" top-{m}")
            } else {
                String::new()
            };
            lines.push(format!("{indent}Sort ({} key(s)){suffix}", keys.len()));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::Project { input, columns } => {
            lines.push(format!("{indent}Project ({} column(s))", columns.len()));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::ProjectSet { input, columns, .. } => {
            lines.push(format!("{indent}ProjectSet ({} column(s))", columns.len()));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::Limit {
            input,
            count,
            offset,
        } => {
            if *offset > 0 {
                lines.push(format!("{indent}Limit {count} offset {offset}"));
            } else {
                lines.push(format!("{indent}Limit {count}"));
            }
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::Distinct { input } => {
            lines.push(format!("{indent}Distinct"));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::DistinctOn { input, keys } => {
            lines.push(format!("{indent}DistinctOn ({} key(s))", keys.len()));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::ScalarAggregate { input, calls } => {
            lines.push(format!(
                "{indent}ScalarAggregate ({} aggregate(s))",
                calls.len()
            ));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::GroupAggregate {
            input,
            group_keys,
            calls,
        } => {
            lines.push(format!(
                "{indent}GroupAggregate ({} key(s), {} aggregate(s))",
                group_keys.len(),
                calls.len()
            ));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::GroupingSetsAggregate {
            input,
            group_keys,
            grouping_sets,
            calls,
        } => {
            lines.push(format!(
                "{indent}GroupingSetsAggregate ({} set(s), {} key(s), {} aggregate(s))",
                grouping_sets.len(),
                group_keys.len(),
                calls.len()
            ));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::Window {
            input,
            windows,
            top_n,
        } => {
            let suffix = top_n.map_or(String::new(), |m| format!(" top-{m}"));
            lines.push(format!(
                "{indent}Window ({} function(s)){suffix}",
                windows.len()
            ));
            lines.extend(format_op(input, depth + 1, ctx, actuals));
        },
        PhysicalOperator::NestedLoopJoin {
            left, right, kind, ..
        } => {
            let kind_label = match kind {
                ast::JoinKind::Inner => "Inner",
                ast::JoinKind::Left => "Left",
                ast::JoinKind::Right => "Right",
                ast::JoinKind::Full => "Full",
                ast::JoinKind::Cross => "Cross",
            };
            lines.push(format!("{indent}NestedLoopJoin ({kind_label})"));
            lines.extend(format_op(left, depth + 1, ctx, actuals));
            lines.extend(format_op(right, depth + 1, ctx, actuals));
        },
        PhysicalOperator::HashJoin {
            left,
            right,
            keys,
            kind,
            ..
        } => {
            let kind_label = match kind {
                ast::JoinKind::Inner => "Inner",
                ast::JoinKind::Left => "Left",
                ast::JoinKind::Right => "Right",
                ast::JoinKind::Full => "Full",
                ast::JoinKind::Cross => "Cross",
            };
            lines.push(format!(
                "{indent}HashJoin ({kind_label}, {} key(s))",
                keys.len()
            ));
            lines.extend(format_op(left, depth + 1, ctx, actuals));
            lines.extend(format_op(right, depth + 1, ctx, actuals));
        },
        PhysicalOperator::LateralJoin {
            left, right, kind, ..
        } => {
            let kind_label = match kind {
                ast::JoinKind::Inner => "Inner",
                ast::JoinKind::Left => "Left",
                ast::JoinKind::Right => "Right",
                ast::JoinKind::Full => "Full",
                ast::JoinKind::Cross => "Cross",
            };
            lines.push(format!("{indent}LateralJoin ({kind_label})"));
            lines.extend(format_op(left, depth + 1, ctx, actuals));
            lines.extend(format_op(right, depth + 1, ctx, actuals));
        },
        PhysicalOperator::WithRecursive { ctes, body } => {
            lines.push(format!("{indent}WithRecursive ({} CTE(s))", ctes.len()));
            for cte in ctes {
                let label = if cte.union_all { "UNION ALL" } else { "UNION" };
                lines.push(format!("{indent}  RecursiveCte ({label})"));
                lines.extend(format_op(&cte.base, depth + 2, ctx, actuals));
                lines.extend(format_op(&cte.recursive, depth + 2, ctx, actuals));
            }
            lines.extend(format_op(body, depth + 1, ctx, actuals));
        },
        PhysicalOperator::WithModifying { ctes, body } => {
            lines.push(format!("{indent}WithModifying ({} CTE(s))", ctes.len()));
            lines.extend(format_op(body, depth + 1, ctx, actuals));
        },
    }
    // Annotate this operator's own line with its estimated row count and subtree cost when a cost
    // context is available (i.e. the single scanned table has been analyzed).
    if let (Some(ctx), Some(first)) = (ctx, lines.first_mut()) {
        use std::fmt::Write as _;
        let _ = write!(
            first,
            " (est. rows={:.0} cost={:.1})",
            cost::estimate_rows(op, Some(ctx)),
            cost::estimate_cost(op, Some(ctx)),
        );
    }
    // EXPLAIN ANALYZE: annotate with the rows the node actually produced during the
    // instrumented run — next to the estimate above, that is the est-vs-actual comparison. A node
    // with no recorded count never ran (e.g. a branch short-circuited away).
    if let (Some(actuals), Some(first)) = (actuals, lines.first_mut()) {
        use std::fmt::Write as _;
        match actuals.get(&instrument::key(op)) {
            Some(n) => {
                let _ = write!(first, " (actual rows={n})");
            },
            None => first.push_str(" (never executed)"),
        }
    }
    lines
}

/// Engine-scoped system catalog of materialized views' defining SQL, so `REFRESH` can recompute
/// them. A two-text-column table (`name`, `def`) created lazily — no treaty change.
const MATVIEW_CATALOG: &str = "nusadb_matviews";

/// Engine-scoped system catalog of non-materialized views' defining SQL, inlined by the analyzer on
/// read. Same shape as [`MATVIEW_CATALOG`].
const VIEW_CATALOG: &str = "nusadb_views";

/// Engine-scoped system catalog of non-materialized views' explicit column-name lists, keyed by view
/// name. Present only for views created with `CREATE VIEW name (cols) AS ...`; `def` is the
/// tab-separated column names. Same `(name, def)` shape as [`VIEW_CATALOG`].
const VIEW_COLUMNS_CATALOG: &str = "nusadb_view_columns";

/// `(name, def)` text schema of the view catalog tables.
const VIEW_CATALOG_SCHEMA: [ColumnType; 2] = [ColumnType::Text, ColumnType::Text];

/// Separator joining a view's explicit column names in [`VIEW_COLUMNS_CATALOG`]'s `def` column.
const VIEW_COLUMN_SEP: char = '\t';

/// Look up the named view-catalog table, creating it (lazily) if it does not exist yet.
fn ensure_view_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
    catalog: &str,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, catalog)? {
        return Ok(schema.id);
    }
    let def = TableDef {
        schema: "public".to_owned(),
        name: catalog.to_owned(),
        columns: vec![
            ColumnDef {
                name: "name".to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            },
            ColumnDef {
                name: "def".to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            },
        ],
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Persist (or replace) a view's defining SQL in `catalog`.
fn store_view_def(
    engine: &dyn StorageEngine,
    txn: TxnId,
    catalog: &str,
    name: &str,
    def_sql: &str,
) -> Result<(), Error> {
    let cat = ensure_view_catalog(engine, txn, catalog)?;
    delete_view_def(engine, txn, catalog, name)?;
    let row = [
        ast::Value::Text(name.to_owned()),
        ast::Value::Text(def_sql.to_owned()),
    ];
    let bytes = row::encode(&row, &VIEW_CATALOG_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Fetch a view's defining SQL from `catalog`, or `None` if it has none recorded.
fn load_view_def(
    engine: &dyn StorageEngine,
    txn: TxnId,
    catalog: &str,
    name: &str,
) -> Result<Option<String>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, catalog)? else {
        return Ok(None);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &VIEW_CATALOG_SCHEMA)?;
        if let [ast::Value::Text(n), ast::Value::Text(def)] = row.as_slice()
            && n == name
        {
            return Ok(Some(def.clone()));
        }
    }
    Ok(None)
}

/// Remove a view's row from `catalog`, returning whether one was deleted.
fn delete_view_def(
    engine: &dyn StorageEngine,
    txn: TxnId,
    catalog: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, catalog)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &VIEW_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == name) {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}

/// The **scannable** indexes of table `name` under `txn`'s snapshot, as planner
/// [`IndexInfo`](crate::IndexInfo)s.
///
/// The shared body of every production `Catalog::list_indexes` adapter (the session/EXECUTE
/// catalogs and the wire's).
/// Since the backing-index unification, this includes the `PRIMARY KEY` / `UNIQUE` constraint-backing indexes (maintained
/// on every write like explicit ones), so the fundamental point-get plans an `IndexScan` instead
/// of a full-table `SeqScan`. Only a **complete** index (every live row has an entry —
/// [`StorageEngine::index_is_complete`]) is offered: a stale one, e.g. recovered from a data dir
/// written before backing-index maintenance existed, would silently drop rows if scanned, so it
/// stays catalog-visible but is never a scan candidate. The full `WHERE` filter is always
/// re-applied above the scan, so an offered index only narrows it.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn catalog_list_indexes(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Vec<crate::IndexInfo>, Error> {
    let Some(schema) = engine.lookup_table_as_of(txn, name)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for def in engine.list_indexes(schema.id)? {
        let Some(id) = engine.lookup_index(&def.name)? else {
            continue;
        };
        if !engine.index_is_complete(id)? {
            continue;
        }
        // A functional/expression key (`key_exprs`) or a partial predicate makes this index
        // unsafe as an equality/range scan candidate: the planner encodes scan bounds from the
        // query's plain-column values in ascending key order, which would not match a key computed
        // from an expression, nor an index that holds only the rows satisfying a predicate. Such
        // indexes are still maintained (and enforce uniqueness) — they are simply not offered as a
        // scan path, so the query falls back to a sequential scan, which is correct. (Matching a
        // functional index to a `WHERE lower(s) = …` predicate is possible future work.)
        if !def.key_exprs.is_empty() || def.predicate.is_some() {
            continue;
        }
        out.push(crate::IndexInfo {
            name: def.name,
            columns: def.columns,
            unique: def.unique,
        });
    }
    Ok(out)
}

/// The ANALYZE statistics of table `name` under `txn`'s snapshot, for cost-based planning
/// — the shared body of every production `Catalog::table_stats` adapter.
///
/// `None` (never analyzed) leaves planning heuristic.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn catalog_table_stats(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<nusadb_core::TableStats>, Error> {
    let Some(schema) = engine.lookup_table_as_of(txn, name)? else {
        return Ok(None);
    };
    engine.table_stats(schema.id).map_err(Into::into)
}

/// The `O(1)` approximate live-row count of table `name` under `txn`'s snapshot — the shared body of
/// every production `Catalog::approx_row_count` adapter.
///
/// The vectorized-routing cardinality fallback used when a table has no `ANALYZE` stats. `0` when the
/// table is unknown (the caller then gives no hint).
///
/// # Errors
/// Propagates storage errors.
pub fn catalog_approx_row_count(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<u64, Error> {
    let Some(schema) = engine.lookup_table_as_of(txn, name)? else {
        return Ok(0);
    };
    engine.approx_row_count(schema.id).map_err(Into::into)
}

/// The defining SQL of the non-materialized view `name`, read from the view catalog under `txn`'s
/// snapshot. Used by the analyzer's `Catalog::lookup_view` adapters to inline the view body.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn lookup_view_definition(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<String>, Error> {
    load_view_def(engine, txn, VIEW_CATALOG, name)
}

/// The explicit output column names of the non-materialized view `name`, or empty if none.
///
/// Declared via `CREATE VIEW name (cols) AS ...`. Used by the analyzer's
/// `Catalog::lookup_view_columns` adapters to rename the inlined view body positionally.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn lookup_view_columns(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Vec<String>, Error> {
    Ok(
        load_view_def(engine, txn, VIEW_COLUMNS_CATALOG, name)?.map_or_else(Vec::new, |joined| {
            joined.split(VIEW_COLUMN_SEP).map(str::to_owned).collect()
        }),
    )
}

/// Engine-scoped system catalog of `USING hnsw` vector-index declarations. Same `(name, def)`
/// two-text-column shape as the view catalog; `def` is tab-separated `table`, column ordinal, and
/// dimension. Only the *declaration* is persisted here — the HNSW graph is rebuilt in memory on
/// demand — so a vector index survives a restart even though its graph does not.
const VECTOR_INDEX_CATALOG: &str = "nusadb_vector_indexes";

/// A vector-index declaration read back from [`VECTOR_INDEX_CATALOG`], for the table column
/// the lookup matched on.
pub(super) struct VectorIndexEntry {
    /// The vector dimension `n`.
    pub dim: usize,
}

/// Encode a vector index's `def` column: tab-separated `table`, column ordinal, dimension.
fn encode_vector_index_def(spec: &crate::planner::VectorIndexSpec) -> String {
    format!("{}\t{}\t{}", spec.table, spec.column_ordinal, spec.dim)
}

/// Parse a `def` string written by [`encode_vector_index_def`] into its `(table, ordinal, dim)`.
fn parse_vector_index_def(def: &str) -> Option<(String, usize, usize)> {
    let mut parts = def.split('\t');
    let table = parts.next()?.to_owned();
    let ordinal = parts.next()?.parse().ok()?;
    let dim = parts.next()?.parse().ok()?;
    Some((table, ordinal, dim))
}

/// Record (or replace) a `USING hnsw` vector index in the catalog.
pub(super) fn store_vector_index(
    engine: &dyn StorageEngine,
    txn: TxnId,
    spec: &crate::planner::VectorIndexSpec,
) -> Result<(), Error> {
    store_view_def(
        engine,
        txn,
        VECTOR_INDEX_CATALOG,
        &spec.name,
        &encode_vector_index_def(spec),
    )
}

/// Whether a vector index named `name` is already declared (for `IF NOT EXISTS`).
pub(super) fn vector_index_exists(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<bool, Error> {
    Ok(load_view_def(engine, txn, VECTOR_INDEX_CATALOG, name)?.is_some())
}

/// Remove a vector index's declaration, returning whether one was deleted.
pub(super) fn delete_vector_index(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<bool, Error> {
    delete_view_def(engine, txn, VECTOR_INDEX_CATALOG, name)
}

/// Remove every `USING hnsw` vector index declared on `table_name` (A-UR.01c), so a
/// `DROP TABLE` does not leave an orphaned vector-index declaration behind (a later same-named table
/// would otherwise inherit a stale index). Names are collected before deleting so the catalog scan is
/// not mutated mid-iteration.
pub(super) fn delete_vector_indexes_for_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table_name: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, VECTOR_INDEX_CATALOG)? else {
        return Ok(());
    };
    let mut names: Vec<String> = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &VIEW_CATALOG_SCHEMA)?;
        if let [ast::Value::Text(name), ast::Value::Text(def)] = row.as_slice()
            && parse_vector_index_def(def).is_some_and(|(table, _, _)| table == table_name)
        {
            names.push(name.clone());
        }
    }
    drop(scan);
    for name in names {
        delete_vector_index(engine, txn, &name)?;
    }
    Ok(())
}

/// Find an hnsw vector index declared on `table_name`'s column `ordinal`, if any. Used by
/// the executor to decide whether `ORDER BY col <=> q LIMIT k` can use an HNSW search.
pub(super) fn vector_index_for_column(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table_name: &str,
    ordinal: usize,
) -> Result<Option<VectorIndexEntry>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, VECTOR_INDEX_CATALOG)? else {
        return Ok(None);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &VIEW_CATALOG_SCHEMA)?;
        if let [ast::Value::Text(_), ast::Value::Text(def)] = row.as_slice()
            && let Some((table, column_ordinal, dim)) = parse_vector_index_def(def)
            && table == table_name
            && column_ordinal == ordinal
        {
            return Ok(Some(VectorIndexEntry { dim }));
        }
    }
    Ok(None)
}

/// Engine-scoped system catalog of tables with row-level security enabled. A single-text-column
/// table (`table`) created lazily; a row's presence means RLS is on for that table.
const RLS_CATALOG: &str = "nusadb_rls";

/// `(table)` text schema of the RLS catalog table.
const RLS_CATALOG_SCHEMA: [ColumnType; 1] = [ColumnType::Text];

/// Look up the RLS catalog table, creating it (lazily) if it does not exist yet.
fn ensure_rls_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, RLS_CATALOG)? {
        return Ok(schema.id);
    }
    let def = TableDef {
        schema: "public".to_owned(),
        name: RLS_CATALOG.to_owned(),
        columns: vec![ColumnDef {
            name: "table".to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        }],
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Remove `table`'s RLS marker, returning whether one was present.
fn clear_rls_marker(engine: &dyn StorageEngine, txn: TxnId, table: &str) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, RLS_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &RLS_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == table) {
            victims.push(tid);
        }
    }
    let present = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(present)
}

/// Enable or disable row-level security for `table` in the RLS catalog. Idempotent: the marker
/// is cleared first, so enabling an already-enabled table never duplicates the row.
pub(super) fn set_table_rls(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    enabled: bool,
) -> Result<(), Error> {
    clear_rls_marker(engine, txn, table)?;
    if enabled {
        let cat = ensure_rls_catalog(engine, txn)?;
        let bytes = row::encode(&[ast::Value::Text(table.to_owned())], &RLS_CATALOG_SCHEMA)?;
        engine.insert(txn, cat, &bytes)?;
    }
    Ok(())
}

/// Whether row-level security is enabled on `table`, read under `txn`'s snapshot.
///
/// Used by the analyzer's [`Catalog::rls_enabled`](crate::Catalog::rls_enabled) adapters. Cheap when
/// no table has ever enabled RLS — the catalog table does not exist, so this returns `false` after
/// one lookup.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn rls_table_enabled(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, RLS_CATALOG)? else {
        return Ok(false);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &RLS_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == table) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Engine-scoped system catalog of row-level-security policies. Seven text columns:
/// `(table, name, command, roles, using, check, permissive)`, created lazily. `roles` is a
/// comma-joined list (empty = `PUBLIC`); `using`/`check` are canonical SQL (empty = absent);
/// `permissive` is `"t"` for a permissive policy or `"f"` for a restrictive one.
const POLICY_CATALOG: &str = "nusadb_policies";

/// `(table, name, command, roles, using, check, permissive)` text schema of the policy catalog.
const POLICY_CATALOG_SCHEMA: [ColumnType; 7] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];

/// Look up the policy catalog table, creating it (lazily) if it does not exist yet.
fn ensure_policy_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, POLICY_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = [
        "table",
        "name",
        "command",
        "roles",
        "using",
        "check",
        "permissive",
    ]
    .into_iter()
    .map(|name| ColumnDef {
        name: name.to_owned(),
        ty: ColumnType::Text,
        nullable: false,
    })
    .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: POLICY_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Remove the `(table, name)` policy row, returning whether one was present.
fn delete_policy_row(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, POLICY_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &POLICY_CATALOG_SCHEMA)?;
        if matches!((row.first(), row.get(1)),
            (Some(ast::Value::Text(t)), Some(ast::Value::Text(n))) if t == table && n == name)
        {
            victims.push(tid);
        }
    }
    let present = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(present)
}

/// Remove every RLS policy declared on `table`. Called when the table is dropped so its
/// policies are not orphaned in the catalog — otherwise a later same-named table fails to re-create a
/// policy of the same name ("policy already exists"), breaking idempotent migrations.
pub(super) fn delete_policies_for_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, POLICY_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &POLICY_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(t)) if t == table) {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// `CREATE POLICY`: persist a validated policy. A duplicate name on the same table is a
/// (standard SQL) error, so a policy is never silently overwritten. When `p.replace` is set (an
/// `ALTER POLICY` lowered to a row rewrite), any existing policy of the same name is dropped first.
fn run_create_policy(
    p: &crate::planner::CreatePolicyPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let cat = ensure_policy_catalog(engine, txn)?;
    if p.replace {
        delete_policy_row(engine, txn, &p.table, &p.name)?;
    } else if lookup_policies_for(engine, txn, &p.table)?
        .iter()
        .any(|existing| existing.name == p.name)
    {
        return Err(Error::Unsupported(format!(
            "policy `{}` already exists on `{}`",
            p.name, p.table
        )));
    }
    let row = [
        ast::Value::Text(p.table.clone()),
        ast::Value::Text(p.name.clone()),
        ast::Value::Text(p.command.as_str().to_owned()),
        ast::Value::Text(p.roles.join(",")),
        ast::Value::Text(p.using.clone().unwrap_or_default()),
        ast::Value::Text(p.check.clone().unwrap_or_default()),
        ast::Value::Text(if p.permissive { "t" } else { "f" }.to_owned()),
    ];
    let bytes = row::encode(&row, &POLICY_CATALOG_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(ExecutionResult::Created(nusadb_core::TableId(0)))
}

/// `DROP POLICY [IF EXISTS] name ON table`.
fn run_drop_policy(
    p: &crate::planner::DropPolicyPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let removed = delete_policy_row(engine, txn, &p.table, &p.name)?;
    if !removed && !p.if_exists {
        return Err(Error::Unsupported(format!(
            "policy `{}` does not exist on `{}`",
            p.name, p.table
        )));
    }
    Ok(ExecutionResult::Dropped)
}

/// The row-level-security policies defined on `table`, read under `txn`'s snapshot. Used by the
/// analyzer's [`Catalog::lookup_policies`](crate::Catalog::lookup_policies) adapters.
///
/// # Errors
/// Propagates storage/decode errors.
pub fn lookup_policies_for(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<Vec<crate::analyzer::PolicyDef>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, POLICY_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &POLICY_CATALOG_SCHEMA)?;
        if let [
            ast::Value::Text(t),
            ast::Value::Text(name),
            ast::Value::Text(command),
            ast::Value::Text(roles),
            ast::Value::Text(using),
            ast::Value::Text(check),
            ast::Value::Text(permissive),
        ] = row.as_slice()
            && t == table
        {
            out.push(crate::analyzer::PolicyDef {
                name: name.clone(),
                // Any value other than the restrictive marker `"f"` reads as permissive, so a policy
                // is never accidentally treated as restrictive.
                permissive: permissive != "f",
                command: parse_policy_command(command),
                roles: if roles.is_empty() {
                    Vec::new()
                } else {
                    roles.split(',').map(str::to_owned).collect()
                },
                using: (!using.is_empty()).then(|| using.clone()),
                check: (!check.is_empty()).then(|| check.clone()),
            });
        }
    }
    Ok(out)
}

/// Decode a policy command keyword from its catalog text (unknown text defaults to `ALL`).
fn parse_policy_command(text: &str) -> ast::PolicyCommand {
    use ast::PolicyCommand as C;
    match text {
        "SELECT" => C::Select,
        "INSERT" => C::Insert,
        "UPDATE" => C::Update,
        "DELETE" => C::Delete,
        _ => C::All,
    }
}

/// A minimal [`Catalog`] over a [`StorageEngine`] under one transaction, used to re-analyze a view's
/// stored SQL on `REFRESH`. Resolves base tables, non-materialized views, and — since the backing-index unification — the
/// engine's scannable indexes + ANALYZE stats, so the re-planned body gets the same index-scan
/// candidates a direct query would.
struct ExecCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: TxnId,
}

impl crate::Catalog for ExecCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine
            .lookup_table_as_of(self.txn, name)
            .map_err(Into::into)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine
            .lookup_table_as_of_in(self.txn, schema, name)
            .map_err(Into::into)
    }

    fn list_indexes(&self, table: &str) -> Result<Vec<crate::IndexInfo>, Error> {
        catalog_list_indexes(self.engine, self.txn, table)
    }

    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        catalog_table_stats(self.engine, self.txn, table)
    }

    fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
        catalog_approx_row_count(self.engine, self.txn, table)
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        lookup_view_definition(self.engine, self.txn, name)
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, Error> {
        lookup_view_columns(self.engine, self.txn, name)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<crate::FunctionDef>, Error> {
        function::lookup_function_definition(self.engine, self.txn, name)
    }
}

/// A [`Catalog`](crate::Catalog) over a [`StorageEngine`] that re-analyzes a bound `EXECUTE`
/// statement under one transaction *as the session user*. Carrying the user is what keeps
/// `EXECUTE` from bypassing row-level security: `is_superuser`/`rls_enabled`/`lookup_policies`
/// resolve exactly as the wire's production catalog would for a direct query. Since the backing-index unification it also
/// reports the engine's scannable indexes + ANALYZE stats, so a direct [`Session`] query or an
/// `EXECUTE`d statement plans the same index scans the wire path does — a point-get by
/// `PRIMARY KEY`/`UNIQUE` is `O(log n)`, not a full-table `SeqScan`.
struct SessionCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: TxnId,
    user: &'a str,
    /// The session's ordered `search_path` schemas (from `SET search_path`), so an unqualified name
    /// in a re-analyzed statement (`EXECUTE`, a direct [`Session`] call) resolves through them in
    /// order before `public`.
    search_path: Vec<String>,
}

impl crate::Catalog for SessionCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine
            .lookup_table_as_of(self.txn, name)
            .map_err(Into::into)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine
            .lookup_table_as_of_in(self.txn, schema, name)
            .map_err(Into::into)
    }

    fn search_path(&self) -> Vec<String> {
        self.search_path.clone()
    }

    fn list_indexes(&self, table: &str) -> Result<Vec<crate::IndexInfo>, Error> {
        catalog_list_indexes(self.engine, self.txn, table)
    }

    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        catalog_table_stats(self.engine, self.txn, table)
    }

    fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
        catalog_approx_row_count(self.engine, self.txn, table)
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        lookup_view_definition(self.engine, self.txn, name)
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, Error> {
        lookup_view_columns(self.engine, self.txn, name)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<crate::FunctionDef>, Error> {
        function::lookup_function_definition(self.engine, self.txn, name)
    }

    fn is_superuser(&self) -> bool {
        self.user == crate::BOOTSTRAP_SUPERUSER
    }

    fn current_user(&self) -> String {
        self.user.to_owned()
    }

    fn rls_enabled(&self, name: &str) -> Result<bool, Error> {
        rls_table_enabled(self.engine, self.txn, name)
    }

    fn lookup_policies(&self, name: &str) -> Result<Vec<crate::PolicyDef>, Error> {
        lookup_policies_for(self.engine, self.txn, name)
    }
}

/// Replace every row of `table_id` with `rows`, encoded against `schema`. Used to refresh a view's
/// backing table in place (keeping its `TableId` stable).
fn replace_rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table_id: nusadb_core::TableId,
    schema: &[ColumnType],
    rows: &[Row],
) -> Result<(), Error> {
    let mut old = Vec::new();
    let mut scan = engine.scan(txn, table_id)?;
    while let Some((tid, _)) = scan.try_next()? {
        old.push(tid);
    }
    for tid in old {
        engine.delete(txn, table_id, tid)?;
    }
    // Maintain the table's indexes for the re-inserted rows: a materialized view's backing
    // table can carry explicit indexes, and the refresh re-inserts every row under a new tid —
    // without re-indexing, an index scan would miss every refreshed row. (The deleted versions'
    // entries stay until VACUUM; stale-extra entries are visibility-filtered, only missing ones
    // lose rows.)
    let index_targets = match dml::schema_by_id(engine, table_id)? {
        Some(table) => dml::secondary_index_targets(&table, engine)?,
        None => Vec::new(),
    };
    for row in rows {
        let tid = engine.insert(txn, table_id, &row::encode(row, schema)?)?;
        dml::insert_into_indexes(&index_targets, row, tid, engine, txn)?;
    }
    Ok(())
}

/// `CREATE MATERIALIZED VIEW`: run the body once and store its rows in a backing table named
/// after the view, so reads are ordinary table scans, and record its defining SQL so it can be
/// refreshed. `OR REPLACE` drops an existing same-named table first; otherwise a name clash errors.
fn run_create_materialized_view(
    p: PhysicalMaterializedView,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // IF NOT EXISTS must also no-op on an existing PLAIN view (it has no backing table, so the
    // table lookup below cannot see it) — otherwise the new backing table would silently shadow
    // the view (audit catch: reads resolve tables before views).
    if p.if_not_exists && load_view_def(engine, txn, VIEW_CATALOG, &p.name)?.is_some() {
        return Ok(ExecutionResult::Created(nusadb_core::TableId(0)));
    }
    match engine.lookup_table_as_of(txn, &p.name)? {
        Some(existing) if p.or_replace => engine.drop_table(txn, existing.id)?,
        Some(existing) if p.if_not_exists => {
            return Ok(ExecutionResult::Created(existing.id));
        },
        Some(_) => return Err(Error::TableExists { name: p.name }),
        None => {},
    }
    let def = TableDef {
        schema: "public".to_owned(),
        name: p.name.clone(),
        columns: p
            .columns
            .iter()
            .map(|(name, ty)| ColumnDef {
                name: name.clone(),
                ty: *ty,
                nullable: true,
            })
            .collect(),
    };
    let table_id = engine.create_table(txn, &def)?;
    let rows = execute_op(&p.body, engine, txn)?;
    let schema: Vec<ColumnType> = p.columns.iter().map(|(_, ty)| *ty).collect();
    for row in &rows {
        engine.insert(txn, table_id, &row::encode(row, &schema)?)?;
    }
    store_view_def(engine, txn, MATVIEW_CATALOG, &p.name, &p.definition_sql)?;
    // Register for incremental maintenance when the body is IVM-eligible; otherwise the view
    // is full-refresh-only. `OR REPLACE` re-registers (or, for a now-ineligible body, clears any prior
    // registration).
    if let Some(base) = &p.ivm_base {
        ivm::register_ivm_view(engine, txn, &p.name, base)?;
    } else {
        ivm::unregister_ivm_view(engine, txn, &p.name)?;
    }
    Ok(ExecutionResult::Created(table_id))
}

/// `DROP DATABASE name [FORCE]` (or the `FIX DROP DATABASE name` alias): drop every user table in the
/// single database (NusaDB is one database per data directory).
///
/// **Parity guard:** this only wipes when `name` matches the current database
/// ([`session_ctx::current_database`]); dropping any *other* name is an accepted no-op, exactly as
/// `DROP DATABASE other` against a multi-database server leaves the connected database's tables
/// untouched. So a portable script that brackets work with `CREATE DATABASE x; … ; DROP DATABASE x`
/// never wipes the real data.
///
/// Unless `force` (the `FORCE` keyword / `FIX` alias), each table is **backed up** first — its current columns and all of
/// its rows (a data safety net, not a full-fidelity clone: constraints and secondary indexes are not
/// copied) — into a fresh table `{name}_{datetime}_{table}`, so an accidental `DROP DATABASE` is
/// recoverable. Rows are read through [`scan::scan_table`], which already transforms each stored row to
/// the table's current schema, then re-encoded for the backup — so a table with `ALTER TABLE` history
/// backs up correctly.
///
/// System catalog tables (the [`SYSTEM_TABLE_PREFIX`](crate::analyzer::SYSTEM_TABLE_PREFIX)) are never
/// touched. The table list is snapshotted before any backup is created, so a backup is never itself
/// backed up or dropped. Everything runs in the caller's transaction, so any failure rolls the whole
/// operation back — there is no half-dropped database.
///
/// # Errors
/// Propagates engine list / lookup / create / scan / insert / drop errors.
fn run_drop_database(
    plan: &crate::planner::DropDatabasePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // Parity guard: NusaDB serves a single database, so only dropping *that* database — by its current
    // name — wipes its tables. Dropping any other name is an accepted no-op, exactly as
    // `DROP DATABASE other` against a multi-database server leaves the connected database's tables
    // untouched. This keeps a portable script that brackets work with
    // `CREATE DATABASE x; … ; DROP DATABASE x` (a common ecosystem / differential-test idiom) from
    // wiping the real data.
    if !plan
        .name
        .eq_ignore_ascii_case(&session_ctx::current_database())
    {
        return Ok(ExecutionResult::DatabaseDropped);
    }

    // The user tables to drop, enumerated under THIS transaction's snapshot (not the
    // non-transactional `list_tables`, whose latest-committed view races a just-committed prior
    // statement on a reused connection → an empty list → a silent no-op). Snapshotted before any
    // backup table is created so backups are never themselves backed up or re-dropped. System catalog
    // tables are excluded.
    let originals: Vec<String> = engine
        .list_tables_as_of(txn)?
        .into_iter()
        .filter(|name| !name.starts_with(crate::analyzer::SYSTEM_TABLE_PREFIX))
        .collect();

    if !plan.force {
        let stamp = crate::temporal::compact_stamp(clock::statement_now_micros());
        for name in &originals {
            let Some(schema) = engine.lookup_table_as_of(txn, name)? else {
                continue; // vanished concurrently — nothing to back up
            };
            // The backup carries the original's current columns; rows come from `scan_table` already
            // normalised to that schema, then re-encoded, so the copy is faithful even after ALTERs.
            let backup = TableDef {
                schema: "public".to_owned(),
                name: format!("{}_{}_{}", plan.name, stamp, name),
                columns: schema.columns.clone(),
            };
            let col_types: Vec<ColumnType> = schema.columns.iter().map(|c| c.ty).collect();
            let rows = scan::scan_table(&schema, engine, txn)?;
            let backup_id = engine.create_table(txn, &backup)?;
            for (_tid, values) in rows {
                engine.insert(txn, backup_id, &row::encode(&values, &col_types)?)?;
            }
        }
    }

    // Drop the originals (never the freshly-created backups — they are not in `originals`).
    for name in &originals {
        if let Some(schema) = engine.lookup_table_as_of(txn, name)? {
            engine.drop_table(txn, schema.id)?;
        }
    }
    Ok(ExecutionResult::DatabaseDropped)
}

/// `CREATE TABLE [IF NOT EXISTS] name [(cols)] AS <select>`: create a fresh table from the
/// query's derived schema and seed it with the query's rows. Unlike a materialized view, the result
/// is an independent table — no recorded definition, no incremental maintenance. With `IF NOT EXISTS`
/// an already-existing table makes the statement a no-op (the query is not run again).
fn run_create_table_as(
    p: PhysicalCreateTableAs,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if let Some(existing) = engine.lookup_table_as_of(txn, &p.name)? {
        if p.if_not_exists {
            return Ok(ExecutionResult::Created(existing.id));
        }
        // The analyzer already rejected a name clash without IF NOT EXISTS; guard defensively in case
        // the table was created concurrently after analysis.
        return Err(Error::TableExists { name: p.name });
    }
    let def = TableDef {
        schema: "public".to_owned(),
        name: p.name.clone(),
        columns: p
            .columns
            .iter()
            .map(|(name, ty)| ColumnDef {
                name: name.clone(),
                ty: *ty,
                nullable: true,
            })
            .collect(),
    };
    let table_id = engine.create_table(txn, &def)?;
    // Compute the source rows before inserting; the body never reads the new (empty) table.
    let rows = execute_op(&p.body, engine, txn)?;
    let schema: Vec<ColumnType> = p.columns.iter().map(|(_, ty)| *ty).collect();
    for row in &rows {
        engine.insert(txn, table_id, &row::encode(row, &schema)?)?;
    }
    Ok(ExecutionResult::Created(table_id))
}

/// `CREATE [OR REPLACE] VIEW name AS <select>` (M6, non-materialized): record the defining SQL in the
/// view catalog so reads can inline it; no backing table is created. `OR REPLACE` overwrites an
/// existing view; otherwise the analyzer has already rejected a name clash.
fn run_create_view(
    p: &CreatePlainViewPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // IF NOT EXISTS: re-checked here under the statement's own snapshot (the analyzer skipped
    // its clash error) — an existing view or table of the name makes this a no-op.
    if p.if_not_exists
        && (load_view_def(engine, txn, VIEW_CATALOG, &p.name)?.is_some()
            || engine.lookup_table_as_of(txn, &p.name)?.is_some())
    {
        return Ok(ExecutionResult::Created(nusadb_core::TableId(0)));
    }
    store_view_def(engine, txn, VIEW_CATALOG, &p.name, &p.definition_sql)?;
    // Record (or, under OR REPLACE, clear) the explicit column-name list. Always remove the prior
    // entry first so replacing a `VIEW v(a,b)` with a plain `VIEW v` drops the stale rename.
    if p.columns.is_empty() {
        delete_view_def(engine, txn, VIEW_COLUMNS_CATALOG, &p.name)?;
    } else {
        let joined = p.columns.join(&VIEW_COLUMN_SEP.to_string());
        store_view_def(engine, txn, VIEW_COLUMNS_CATALOG, &p.name, &joined)?;
    }
    Ok(ExecutionResult::Created(nusadb_core::TableId(0)))
}

/// `REFRESH MATERIALIZED VIEW name`: re-execute the view's stored definition and replace its
/// backing rows in place. Errors if the name is not a materialized view (no recorded definition).
fn run_refresh_materialized_view(
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let Some(def_sql) = load_view_def(engine, txn, MATVIEW_CATALOG, name)? else {
        return Err(Error::Unsupported(format!(
            "`{name}` is not a materialized view (no stored definition to refresh)"
        )));
    };
    let table = engine
        .lookup_table_as_of(txn, name)?
        .ok_or_else(|| Error::TableNotFound {
            name: name.to_owned(),
        })?;
    // Re-analyze and re-plan the stored SELECT against the current catalog, then run it.
    let logical = crate::analyze(crate::parse(&def_sql)?, &ExecCatalog { engine, txn })?;
    let PhysicalPlan::Select(op, _) = crate::plan(logical) else {
        return Err(Error::Unsupported(
            "materialized view definition is not a SELECT".to_owned(),
        ));
    };
    let rows = execute_op(&op, engine, txn)?;
    let schema = column_types(&table);
    replace_rows(engine, txn, table.id, &schema, &rows)?;
    Ok(ExecutionResult::Updated(rows.len()))
}

/// `DROP [MATERIALIZED] VIEW` — drop the view's backing table and forget its definition (sqlparser
/// maps both `DROP VIEW` and `DROP MATERIALIZED VIEW` to the same statement).
fn run_drop_view(
    p: &DropViewPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, &p.name)? {
        // A materialized view's backing table — drop it, forget its definition, and stop maintaining
        // it incrementally.
        engine.drop_table(txn, schema.id)?;
        delete_view_def(engine, txn, MATVIEW_CATALOG, &p.name)?;
        ivm::unregister_ivm_view(engine, txn, &p.name)?;
    } else {
        // No backing table — it may be a non-materialized view (catalog entry only).
        let removed = delete_view_def(engine, txn, VIEW_CATALOG, &p.name)?;
        // Forget any explicit column-name list (no-op if the view declared none).
        delete_view_def(engine, txn, VIEW_COLUMNS_CATALOG, &p.name)?;
        if !removed && !p.if_exists {
            return Err(Error::TableNotFound {
                name: p.name.clone(),
            });
        }
    }
    Ok(ExecutionResult::Dropped)
}

/// Engine-scoped system catalog of user-defined `ENUM` types: `(name, def)` where `def` is the
/// labels joined by [`ENUM_LABEL_SEP`] in declaration order (B-ENUM). Same `(name, def)` text shape
/// as [`VIEW_CATALOG`].
const ENUM_CATALOG: &str = "nusadb_enums";

/// Separator joining an enum's labels in [`ENUM_CATALOG`]'s `def` column — the ASCII unit separator,
/// which does not occur in ordinary enum labels.
const ENUM_LABEL_SEP: char = '\u{1f}';

/// `CREATE TYPE name AS ENUM (...)` — persist the label set (B-ENUM). Rejects a name already taken by
/// an enum or an existing table, mirroring catalog-object uniqueness.
fn run_create_enum(
    p: &ast::CreateEnum,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if lookup_enum(engine, txn, &p.name)?.is_some()
        || engine.lookup_table_as_of(txn, &p.name)?.is_some()
    {
        return Err(Error::Unsupported(format!(
            "type {:?} already exists",
            p.name
        )));
    }
    let joined = p.labels.join(&ENUM_LABEL_SEP.to_string());
    store_view_def(engine, txn, ENUM_CATALOG, &p.name, &joined)?;
    Ok(ExecutionResult::Created(nusadb_core::TableId(0)))
}

/// `DROP TYPE [IF EXISTS] name` — forget a user-defined enum (B-ENUM).
fn run_drop_type(
    p: &ast::DropType,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let removed = delete_view_def(engine, txn, ENUM_CATALOG, &p.name)?;
    if !removed && !p.if_exists {
        return Err(Error::Unsupported(format!(
            "type {:?} does not exist",
            p.name
        )));
    }
    Ok(ExecutionResult::Dropped)
}

/// The ordered labels of a user-defined enum type, or `None` if no such type exists (B-ENUM).
pub(crate) fn lookup_enum(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<Vec<String>>, Error> {
    Ok(load_view_def(engine, txn, ENUM_CATALOG, name)?
        .map(|def| def.split(ENUM_LABEL_SEP).map(str::to_owned).collect()))
}

// === Internal helpers =====================================================

fn column_types(table: &TableSchema) -> Vec<ColumnType> {
    table.columns.iter().map(|c| c.ty).collect()
}

fn column_at(table: &TableSchema, index: usize) -> Result<&ColumnDef, Error> {
    table
        .columns
        .get(index)
        .ok_or_else(|| internal_index(index))
}

/// Error for an out-of-range ordinal the analyzer should have ruled out — a
/// planner/analyzer bug, surfaced rather than panicking.
fn internal_index(index: usize) -> Error {
    Error::Unsupported(format!("internal: row/column index {index} out of bounds"))
}

fn set_at(row: &mut Row, index: usize, value: ast::Value) -> Result<(), Error> {
    let slot = row
        .get_mut(index)
        .ok_or_else(|| Error::Unsupported(format!("internal: row index {index} out of bounds")))?;
    *slot = value;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "in-memory test mock; lock hold time is not a correctness concern"
)]
mod tests;
