//! Top-level statement enum + COMMENT/ANALYZE.
//!
//! Pure AST types split verbatim out of `ast/mod.rs` (ADR 007). Sibling types resolve via
//! `use super::*` (re-exported by the parent).
#![allow(clippy::wildcard_imports)]

use super::*;

/// Output format for an `EXPLAIN` statement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExplainFormat {
    /// Human-readable indented plan tree (the default).
    #[default]
    Text,
    /// Structured JSON: a nested `{ "node", "children" }` plan tree, plus an `output` array under
    /// `VERBOSE` and an `execution` object under `ANALYZE`. Emitted as one pretty-printed document.
    Json,
}

/// Options on an `EXPLAIN` statement. `FORMAT` accepts `TEXT` (default) and `JSON`;
/// other formats (e.g. `GRAPHVIZ`) are rejected at parse time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExplainOptions {
    /// `EXPLAIN ANALYZE` — also execute the (read-only) statement and report actual rows + time.
    pub analyze: bool,
    /// `EXPLAIN VERBOSE` — include the plan's output columns.
    pub verbose: bool,
    /// `EXPLAIN (FORMAT …)` — the output format.
    pub format: ExplainFormat,
}

/// Options on a `VACUUM` statement.
///
/// `FULL` requests an aggressive rewrite (treated as the standard reclaim for now; deep compaction
/// is a follow-up); `ANALYZE` recomputes statistics for every table after reclaiming.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumOptions {
    /// `VACUUM FULL` — aggressive reclaim/rewrite.
    pub full: bool,
    /// `VACUUM ANALYZE` — recompute statistics for all tables afterwards.
    pub analyze: bool,
}

/// A table-level lock mode for `LOCK TABLE`. NusaDB's lock manager supports the two
/// extremes of the standard table-lock lattice; `ACCESS EXCLUSIVE` is the default when none is given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LockMode {
    /// `ACCESS SHARE` — the weakest lock; conflicts only with `ACCESS EXCLUSIVE`.
    AccessShare,
    /// `ACCESS EXCLUSIVE` — the strongest lock; conflicts with every other table lock (the default).
    #[default]
    AccessExclusive,
}

/// The target of a `DEALLOCATE` statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeallocateTarget {
    /// `DEALLOCATE name` — discard one prepared statement.
    Name(String),
    /// `DEALLOCATE ALL` — discard every prepared statement in the session.
    All,
}

/// Whether `stmt` is a FROM-less `SELECT` whose every expression is pure bounded CPU.
///
/// Over the single synthesized row — no table access, no subquery, no set-returning function,
/// no user-defined function (whose body could scan tables) — such a statement cannot touch
/// disk and produces at most one output row, so a server may run it inline on its I/O thread
/// instead of paying two thread hops to the blocking pool. Default-deny: any
/// unrecognized expression shape keeps the statement on the pool, which is always correct.
#[must_use]
pub fn from_less_pure_select(stmt: &Statement) -> bool {
    let Statement::Select(s) = stmt else {
        return false;
    };
    if s.from.is_some() || !s.with.is_empty() {
        return false;
    }
    let exprs_pure = |exprs: &[Expr]| exprs.iter().all(pure_row_expr);
    s.projection.iter().all(|item| match item {
        SelectItem::Expr { expr, .. } => pure_row_expr(expr),
        // A wildcard without FROM fails analysis immediately — bounded either way.
        _ => true,
    }) && s.filter.as_ref().is_none_or(pure_row_expr)
        && s.having.as_ref().is_none_or(pure_row_expr)
        && s.order_by.iter().all(|k| pure_row_expr(&k.expr))
        && match &s.group_by {
            GroupBy::Expressions(list) => exprs_pure(list),
            _ => false,
        }
}

/// Whether `stmt` is a *candidate* for the reactor-inline point-get path.
///
/// A candidate is a plain single-table `SELECT` (no CTE/join/derived-table/set-op, no
/// `DISTINCT`/`GROUP BY`/`HAVING`, no `FOR UPDATE`) whose `WHERE` has at least one top-level
/// `column = literal` conjunct and whose expressions all pass the same closed pure-built-in
/// walk as the FROM-less gate — no subqueries, UDFs, aggregates, windows, or set-returning
/// calls. Default-deny.
///
/// A candidate is NOT yet admitted: the wire layer pre-flight-plans it and requires
/// [`crate::plan_is_inline_point_get`] (a unique-index point bound), so a candidate that plans
/// to anything unbounded falls back to the blocking pool.
#[must_use]
pub fn point_get_candidate(stmt: &Statement) -> bool {
    let Statement::Select(s) = stmt else {
        return false;
    };
    let Some(from) = &s.from else {
        return false;
    };
    // A plain named base table only — a derived table / VALUES / set-op / ordinality base (or
    // any join) can hide unbounded work behind the same syntactic shape.
    if !from.joins.is_empty()
        || from.base.subquery.is_some()
        || from.base.values.is_some()
        || from.base.set_op.is_some()
        || from.base.with_ordinality
        || from.base.lateral
    {
        return false;
    }
    if !s.with.is_empty()
        || s.distinct.is_some()
        || s.having.is_some()
        || s.lock.is_some()
        || !matches!(&s.group_by, GroupBy::Expressions(list) if list.is_empty())
    {
        return false;
    }
    let Some(filter) = &s.filter else {
        return false;
    };
    has_eq_conjunct(filter)
        && pure_row_expr(filter)
        && s.projection.iter().all(|item| match item {
            SelectItem::Expr { expr, .. } => pure_row_expr(expr),
            // A wildcard over the single base table is one bounded row's columns.
            _ => true,
        })
        && s.order_by.iter().all(|k| pure_row_expr(&k.expr))
}

/// Whether `filter`'s top-level `AND` chain contains at least one `column = literal` conjunct —
/// the shape that can drive a unique-index point bound. Purely a cheap pre-filter so obvious
/// range-only scans skip the pre-flight plan.
fn has_eq_conjunct(filter: &Expr) -> bool {
    match filter {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => has_eq_conjunct(left) || has_eq_conjunct(right),
        Expr::Binary {
            left,
            op: BinaryOp::Eq,
            right,
        } => matches!(
            (left.as_ref(), right.as_ref()),
            (Expr::Column(_), Expr::Literal(_)) | (Expr::Literal(_), Expr::Column(_))
        ),
        _ => false,
    }
}

/// The allow-list walk behind [`from_less_pure_select`]: recognized pure value shapes recurse,
/// everything else — subqueries, set-returning calls, unresolved (possibly user-defined)
/// function calls — denies.
fn pure_row_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) | Expr::Column(_) => true,
        Expr::Binary { left, right, .. } => pure_row_expr(left) && pure_row_expr(right),
        Expr::Unary { expr, .. }
        | Expr::IsNull { expr, .. }
        | Expr::IsJson { operand: expr, .. }
        | Expr::Cast { expr, .. } => pure_row_expr(expr),
        Expr::Between {
            expr, low, high, ..
        } => pure_row_expr(expr) && pure_row_expr(low) && pure_row_expr(high),
        Expr::InList { expr, list, .. } => pure_row_expr(expr) && list.iter().all(pure_row_expr),
        Expr::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(pure_row_expr)
                && branches
                    .iter()
                    .all(|b| pure_row_expr(&b.when) && pure_row_expr(&b.then))
                && default.as_deref().is_none_or(pure_row_expr)
        },
        // The built-in scalar set is row-local CPU (clock/random read thread-locals pinned on
        // the executing thread); sequence access has no scalar built-in, and user-defined
        // functions parse as `FunctionCall`, which the default arm denies.
        Expr::Coalesce(args) | Expr::ArrayLiteral(args) | Expr::ScalarFunction { args, .. } => {
            args.iter().all(pure_row_expr)
        },
        _ => false,
    }
}

/// A single parsed SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A sequence of statements executed in order within one statement transaction — the
    /// internal desugar container for multi-object DDL (`DROP TABLE a, b`); never produced
    /// directly by user-visible syntax.
    Batch(Vec<Self>),
    /// `CREATE TABLE`.
    CreateTable(CreateTable),
    /// `CREATE TABLE [IF NOT EXISTS] name [(cols)] AS <select>`.
    CreateTableAs(CreateTableAs),
    /// `DROP TABLE`.
    DropTable(DropTable),
    /// `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (cols) [INCLUDE (cols)]`.
    CreateIndex(CreateIndex),
    /// `DROP INDEX [IF EXISTS] name`.
    DropIndex(DropIndex),
    /// `ALTER TABLE [IF EXISTS] name <action>` — a single column action.
    AlterTable(AlterTable),
    /// `CREATE [OR REPLACE] VIEW name [(columns)] AS <select>`.
    CreateView(CreateView),
    /// `DROP VIEW [IF EXISTS] name`.
    DropView(DropView),
    /// `CREATE TYPE name AS ENUM (...)` — a user-defined enum type (B-ENUM). Recognized by a custom
    /// parser pass (sqlparser 0.51 only models the composite `CREATE TYPE name AS (...)` form).
    CreateEnum(CreateEnum),
    /// `DROP TYPE [IF EXISTS] name` (B-ENUM / composite). Custom-parsed for the `DROP` form; a
    /// composite type and an enum share the one type namespace, so both are dropped here.
    DropType(DropType),
    /// `CREATE TYPE name AS (field type, ...)` — a user-defined composite (row) type. Parsed by
    /// sqlparser's generic `CREATE TYPE` grammar and converted in the statement converter.
    CreateComposite(CreateComposite),
    /// `CREATE DOMAIN name AS base_type [NOT NULL] [CHECK (VALUE …)]` — a constrained base type.
    /// Recognized by a custom parser pass.
    CreateDomain(CreateDomain),
    /// `DROP DOMAIN [IF EXISTS] name`. Custom-parsed.
    DropDomain(DropDomain),
    /// `CREATE [OR REPLACE] TRIGGER ...` — a triggered SQL action. Recognized by a custom
    /// parser pass (sqlparser 0.51 only models the `EXECUTE FUNCTION` trigger form).
    CreateTrigger(CreateTrigger),
    /// `DROP TRIGGER [IF EXISTS] name ON table`. Custom-parsed.
    DropTrigger(DropTrigger),
    /// `ALTER TRIGGER name ON table RENAME TO new_name`. Custom-parsed.
    AlterTrigger(AlterTrigger),
    /// `CREATE [OR REPLACE] PROCEDURE ...` — a stored procedure with a SQL body. Custom-parsed.
    CreateProcedure(CreateProcedure),
    /// `DROP PROCEDURE [IF EXISTS] name`. Custom-parsed.
    DropProcedure(DropProcedure),
    /// `CALL name(args)` — invoke a stored procedure. Custom-parsed.
    Call(Call),
    /// `DO [LANGUAGE ...] $$ body $$` — run an anonymous code block: the same body grammar as a
    /// procedure (a NusaScript `BEGIN ... END` block, or a plain statement sequence), executed once
    /// with no parameters. Custom-parsed; the body is the raw text.
    Do(String),
    /// `CREATE [OR REPLACE] FUNCTION ...` — a SQL scalar function. Custom-parsed.
    CreateFunction(CreateFunction),
    /// `DROP FUNCTION [IF EXISTS] name`. Custom-parsed.
    DropFunction(DropFunction),
    /// `REFRESH MATERIALIZED VIEW name` — recompute the view's stored rows. Recognized by a
    /// custom parser pass (sqlparser has no `REFRESH` keyword).
    RefreshMaterializedView(String),
    /// `CREATE POLICY ...` — a row-level-security policy. Recognized by a custom parser pass
    /// (sqlparser 0.51 does not model RLS policies).
    CreatePolicy(CreatePolicy),
    /// `DROP POLICY [IF EXISTS] name ON table`. Custom-parsed.
    DropPolicy(DropPolicy),
    /// `ALTER POLICY name ON table [TO ...] [USING ...] [WITH CHECK ...]`. Custom-parsed.
    AlterPolicy(AlterPolicy),
    /// `GRANT privileges ON objects TO grantees [WITH GRANT OPTION]`.
    Grant(Grant),
    /// `REVOKE [GRANT OPTION FOR] privileges ON objects FROM grantees [CASCADE | RESTRICT]`.
    Revoke(Revoke),
    /// `GRANT role[, ...] TO member[, ...] [WITH ADMIN OPTION]` — role membership. Told apart from
    /// [`Statement::Grant`] by the absence of `ON`.
    GrantRole(GrantRole),
    /// `REVOKE [ADMIN OPTION FOR] role[, ...] FROM member[, ...]`.
    RevokeRole(RevokeRole),
    /// `CREATE ROLE | USER [IF NOT EXISTS] name [options]`.
    CreateRole(CreateRole),
    /// `DROP ROLE | USER [IF EXISTS] name[, ...]`.
    DropRole(DropRole),
    /// `ALTER ROLE name [options]`.
    AlterRole(AlterRole),
    /// `SET ROLE name | NONE` / `RESET ROLE` — switch the privileges the session runs with.
    SetRole(SetRole),
    /// `CREATE SCHEMA [IF NOT EXISTS] name`.
    CreateSchema(CreateSchema),
    /// `DROP SCHEMA [IF EXISTS] name`.
    DropSchema(DropSchema),
    /// `CREATE DATABASE [IF NOT EXISTS] name` — single-database compatibility no-op.
    CreateDatabase(CreateDatabase),
    /// `DROP DATABASE [IF EXISTS] name` — drop every table in the single database (backing them up
    /// first unless `FIX DROP DATABASE`).
    DropDatabase(DropDatabase),
    /// `ALTER DATABASE name ...` — single-database compatibility no-op.
    AlterDatabase(AlterDatabase),
    /// `CREATE EXTENSION [IF NOT EXISTS] name [WITH] [SCHEMA ..] [VERSION ..] [CASCADE]` — accepted
    /// as a no-op for client/tool compatibility. NusaDB has no extension system, so nothing is
    /// installed; the `name` is kept only for `EXPLAIN`. Options (`SCHEMA`/`VERSION`/`CASCADE`) are
    /// ignored.
    CreateExtension {
        /// The extension name (as written).
        name: String,
    },
    /// `CREATE SEQUENCE [IF NOT EXISTS] name [options...]`.
    CreateSequence(CreateSequence),
    /// `ALTER SEQUENCE [IF EXISTS] name <options...>`.
    AlterSequence(AlterSequence),
    /// `DROP SEQUENCE [IF EXISTS] name`.
    DropSequence(DropSequence),
    /// `TRUNCATE [TABLE] name [RESTART IDENTITY | CONTINUE IDENTITY]`.
    Truncate(TruncateTable),
    /// `INSERT INTO ... VALUES ...`.
    Insert(Insert),
    /// `SELECT ...`.
    Select(Select),
    /// `<select> {UNION | INTERSECT | EXCEPT} [ALL] <select> [...]` — a set-operation
    /// tree over two or more `SELECT` branches.
    SetOperation(SetOperation),
    /// `UPDATE ... SET ...`.
    Update(Update),
    /// `DELETE FROM ...`.
    Delete(Delete),
    /// `COPY table [(cols)] FROM STDIN | TO STDOUT` — bulk load / export.
    Copy(Copy),
    /// `MERGE INTO target USING source ON cond WHEN [NOT] MATCHED THEN ...`.
    Merge(Merge),
    /// `EXPLAIN [ANALYZE] [VERBOSE] <statement>` — describe the plan for the wrapped statement.
    /// `ANALYZE` also executes a read-only statement and reports its actual row count and total time;
    /// `VERBOSE` adds the plan's output columns.
    Explain(Box<Self>, ExplainOptions),
    /// `BEGIN [opts]` / `START TRANSACTION [opts]` — open an explicit transaction; later
    /// statements run inside it until `COMMIT` or `ROLLBACK`. `opts` carries optional
    /// `ISOLATION LEVEL` / `READ ONLY|WRITE` characteristics.
    BeginTransaction(TransactionSettings),
    /// `COMMIT` — commit the explicit transaction opened by `BEGIN`.
    Commit,
    /// `ROLLBACK` — abort the explicit transaction opened by `BEGIN`.
    Rollback,
    /// `SAVEPOINT name` — mark a rollback point within the current transaction.
    Savepoint(String),
    /// `ROLLBACK TO [SAVEPOINT] name` — undo back to a named savepoint.
    RollbackToSavepoint(String),
    /// `RELEASE [SAVEPOINT] name` — discard a named savepoint.
    ReleaseSavepoint(String),
    /// `SET [SESSION] TRANSACTION ...` — set characteristics for the transaction.
    SetTransaction(TransactionSettings),
    /// `LISTEN channel` — subscribe this connection to a notification channel. The channel name is
    /// a folded identifier. The server owns the cross-connection registry, so this is intercepted at
    /// the wire layer rather than executed by the SQL engine.
    Listen(String),
    /// `UNLISTEN channel` / `UNLISTEN *` — stop listening on one channel (`Some(name)`) or all of
    /// them (`None`, the `*` form).
    Unlisten(Option<String>),
    /// `NOTIFY channel [, 'payload']` — send a notification to every connection listening on the
    /// channel. `payload` is the optional string literal (`None` when omitted).
    Notify {
        /// The channel name (a folded identifier).
        channel: String,
        /// The optional payload string.
        payload: Option<String>,
    },
    /// `SET name = value` / `RESET name` — set or reset a session variable.
    SetVariable(SetVariable),
    /// `SHOW name` — display a session/runtime variable.
    Show(String),
    /// `SHOW TABLES` — list the database's tables (catalog introspection).
    ShowTables,
    /// `SHOW COLUMNS FROM table` — list a table's columns (catalog introspection).
    ShowColumns(Option<String>, String),
    /// `VACUUM [FULL] [ANALYZE]` — reclaim dead row versions across all tables; `ANALYZE` also
    /// recomputes statistics for every table. The bare, table-less form (any options) is supported.
    Vacuum(VacuumOptions),
    /// `REINDEX { INDEX | TABLE | SCHEMA | DATABASE | SYSTEM } name` — accepted as a no-op. NusaDB's
    /// clustered B-tree indexes are kept consistent by MVCC and background purge, so there is nothing
    /// to rebuild; but migration tools and ORM health-checks emit `REINDEX`, so it is accepted rather
    /// than rejected (which would break their scripts).
    Reindex,
    /// `CHECKPOINT` — fold the durable log into a checkpoint image and truncate it, so recovery
    /// replays only what was written since. Requires a quiesced engine, so it may be refused when a
    /// transaction is active. The bare, argument-less form.
    Checkpoint,
    /// `ANALYZE [TABLE] name [(columns)]` — recompute table/column statistics.
    Analyze(Analyze),
    /// `PREPARE name [(types)] AS <statement>` — store a parameterized statement under `name` for
    /// later `EXECUTE`. The declared parameter types are accepted but not retained; the
    /// parameter count is inferred from the `$n` placeholders in `statement`.
    Prepare {
        /// The prepared statement's name.
        name: String,
        /// The statement to prepare (kept un-analyzed; its `$n` placeholders are bound at `EXECUTE`).
        statement: Box<Self>,
    },
    /// `EXECUTE name [(args)]` — run a previously `PREPARE`d statement with `args` bound to its
    /// `$1..$n` placeholders.
    Execute {
        /// The prepared statement's name.
        name: String,
        /// Constant argument expressions, one per placeholder, in order.
        args: Vec<Expr>,
    },
    /// `DEALLOCATE [PREPARE] {name | ALL}` — discard a prepared statement, or all of them.
    Deallocate(DeallocateTarget),
    /// `LOCK [TABLE] name [, ...] [IN {ACCESS SHARE | ACCESS EXCLUSIVE} MODE]` — acquire a
    /// table-level lock held until the end of the transaction. Bare `LOCK TABLE t` defaults
    /// to `ACCESS EXCLUSIVE` (the strongest mode, the conventional default).
    LockTable {
        /// Tables to lock, in the order given.
        tables: Vec<String>,
        /// The requested lock mode.
        mode: LockMode,
    },
    /// `COMMENT ON {TABLE | COLUMN} <object> IS {'text' | NULL}` — attach (or with
    /// `IS NULL`, clear) a human-readable description on a table or column.
    CommentOn(CommentOn),
    /// `DECLARE name [SCROLL] CURSOR [WITH HOLD] FOR <query>` — open a cursor over a query.
    DeclareCursor(DeclareCursor),
    /// `FETCH <direction> [FROM | IN] name` — read rows from an open cursor.
    FetchCursor(FetchCursor),
    /// `CLOSE {name | ALL}` — discard an open cursor, or all of them.
    CloseCursor(CloseTarget),
}

/// Transaction characteristics for `BEGIN ...` and `SET TRANSACTION ...`.
///
/// An all-`None` value is the engine default (plain `BEGIN`). The analyzer threads non-default
/// settings into the plan: `BEGIN` applies them to the transaction it opens and `SET TRANSACTION`
/// updates the session defaults.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransactionSettings {
    /// `ISOLATION LEVEL ...`; `None` means the engine default.
    pub isolation: Option<IsolationLevel>,
    /// `READ ONLY` / `READ WRITE`; `None` means the engine default.
    pub access_mode: Option<AccessMode>,
}

impl TransactionSettings {
    /// `true` when no explicit characteristics were given (a plain `BEGIN`).
    #[must_use]
    pub const fn is_default(&self) -> bool {
        self.isolation.is_none() && self.access_mode.is_none()
    }
}

/// SQL transaction isolation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// `READ UNCOMMITTED`.
    ReadUncommitted,
    /// `READ COMMITTED`.
    ReadCommitted,
    /// `REPEATABLE READ`.
    RepeatableRead,
    /// `SERIALIZABLE`.
    Serializable,
}

/// Transaction access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// `READ ONLY`.
    ReadOnly,
    /// `READ WRITE`.
    ReadWrite,
}

/// `SET name = value` / `RESET name` — a session-variable assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetVariable {
    /// Variable name, folded to lowercase.
    pub name: String,
    /// `Some(text)` for `SET name = value`; `None` for `RESET name`.
    pub value: Option<String>,
}

/// `COMMENT ON {TABLE | COLUMN} <object> IS {'text' | NULL}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentOn {
    /// What the comment is attached to.
    pub target: CommentTarget,
    /// The comment text; `None` for `IS NULL`, which clears any existing comment.
    pub comment: Option<String>,
}

/// The object a [`CommentOn`] targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommentTarget {
    /// `COMMENT ON TABLE <table>`.
    Table {
        /// Schema qualifier, when the statement carried one.
        schema: Option<String>,
        /// Target table name.
        table: String,
    },
    /// `COMMENT ON COLUMN <table>.<column>`.
    Column {
        /// Table owning the column.
        table: String,
        /// Target column name.
        column: String,
    },
}

/// `ANALYZE [TABLE] name [(columns)]` — recompute statistics for a table, or a
/// chosen subset of its columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analyze {
    /// Schema qualifier on the target, when the statement carried one; `None` resolves through the
    /// session search path.
    pub schema: Option<String>,
    /// Target table name; `None` is the bare `ANALYZE` that refreshes every user table.
    pub table: Option<String>,
    /// Columns to analyze; empty means every column. Always empty when `table` is `None`.
    pub columns: Vec<String>,
}
