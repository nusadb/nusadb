//! Plan IR types: `LogicalPlan` / `PhysicalPlan` and every per-statement plan + typed-expression node.
//!
//! Split verbatim out of `planner/mod.rs` (ADR 007). Resolves siblings via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === information_schema views ==================================

/// The `information_schema` views NusaDB exposes for driver/ORM introspection. Each
/// variant maps to a synthetic table whose rows are produced by the executor from engine metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoSchemaView {
    /// `information_schema.tables` — one row per table/view with catalog/schema/name/type.
    Tables,
    /// `information_schema.columns` — one row per column with name/type/nullability/ordinal.
    Columns,
    /// `information_schema.schemata` — one row per schema (catalog/schema/owner).
    Schemata,
    /// `information_schema.views` — one row per view with catalog/schema/name/definition.
    Views,
    /// `information_schema.table_constraints` — one row per PK/UNIQUE/FK/CHECK constraint.
    TableConstraints,
    /// `information_schema.key_column_usage` — one row per (constraint, key column) with the column's
    /// position in the key; backs JDBC `getPrimaryKeys`.
    KeyColumnUsage,
    /// `information_schema.statistics` — one row per (index, key column); an index-metadata view
    /// that backs JDBC `getIndexInfo`.
    Statistics,
}

impl InfoSchemaView {
    /// The name of the view, e.g. `"tables"` for `information_schema.tables`.
    #[must_use]
    pub const fn view_name(self) -> &'static str {
        match self {
            Self::Tables => "tables",
            Self::Columns => "columns",
            Self::Schemata => "schemata",
            Self::Views => "views",
            Self::TableConstraints => "table_constraints",
            Self::KeyColumnUsage => "key_column_usage",
            Self::Statistics => "statistics",
        }
    }

    /// The full canonical name, e.g. `"information_schema.tables"`.
    #[must_use]
    pub fn full_name(self) -> String {
        format!("information_schema.{}", self.view_name())
    }

    /// The synthetic `TableSchema` for this view — defines the column names and types the view
    /// exposes. All columns are nullable to match SQL standard semantics.
    #[must_use]
    pub fn table_schema(self) -> nusadb_core::TableSchema {
        use nusadb_core::{ColumnDef, ColumnType, TableSchema};
        // All `information_schema` columns are nullable TEXT, except a couple of integer ones; these
        // builders keep each view's column list to a single readable line.
        let text = |name: &str| ColumnDef {
            name: name.into(),
            ty: ColumnType::Text,
            nullable: true,
        };
        let int = |name: &str| ColumnDef {
            name: name.into(),
            ty: ColumnType::Int,
            nullable: true,
        };
        let columns = match self {
            Self::Tables => vec![
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                text("table_type"),
            ],
            Self::Columns => vec![
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                text("column_name"),
                int("ordinal_position"),
                text("data_type"),
                text("is_nullable"),
                // Standard reflection columns — NULL when not applicable to the type.
                int("character_maximum_length"),
                int("numeric_precision"),
                int("numeric_scale"),
                // The column's DEFAULT expression as SQL text, or NULL if it has none. A SERIAL
                // / IDENTITY column reports its `nextval('<sequence>')` default.
                text("column_default"),
            ],
            Self::Schemata => vec![
                text("catalog_name"),
                text("schema_name"),
                text("schema_owner"),
            ],
            Self::Views => vec![
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                text("view_definition"),
            ],
            Self::TableConstraints => vec![
                text("constraint_catalog"),
                text("constraint_schema"),
                text("constraint_name"),
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                text("constraint_type"),
            ],
            Self::KeyColumnUsage => vec![
                text("constraint_catalog"),
                text("constraint_schema"),
                text("constraint_name"),
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                text("column_name"),
                int("ordinal_position"),
            ],
            Self::Statistics => vec![
                text("table_catalog"),
                text("table_schema"),
                text("table_name"),
                int("non_unique"),
                text("index_name"),
                int("seq_in_index"),
                text("column_name"),
            ],
        };
        TableSchema {
            schema: "public".to_owned(),
            id: info_schema_table_id(),
            name: self.full_name(),
            columns,
        }
    }

    /// Resolve `name` (e.g. `"information_schema.tables"`) to an `InfoSchemaView`, or `None` if it
    /// is not a recognised `information_schema` view.
    #[must_use]
    pub fn from_full_name(name: &str) -> Option<Self> {
        match name {
            "information_schema.tables" => Some(Self::Tables),
            "information_schema.columns" => Some(Self::Columns),
            "information_schema.schemata" => Some(Self::Schemata),
            "information_schema.views" => Some(Self::Views),
            "information_schema.table_constraints" => Some(Self::TableConstraints),
            "information_schema.key_column_usage" => Some(Self::KeyColumnUsage),
            "information_schema.statistics" => Some(Self::Statistics),
            _ => None,
        }
    }
}

/// A placeholder [`nusadb_core::TableId`] for the synthetic `information_schema` view schemas
/// It is never used to *detect* an info-schema scan — that is done by name at lowering,
/// since a real table or CTE can never be named `information_schema.*` — so this id is inert and
/// only fills the `TableSchema::id` field.
const fn info_schema_table_id() -> nusadb_core::TableId {
    nusadb_core::TableId(u64::MAX - 100)
}

/// Transaction characteristics requested by `BEGIN ...` / `SET TRANSACTION ...`.
///
/// `isolation == None` keeps the session default isolation; `read_only == None` keeps the
/// session default access mode. `BEGIN` applies these to the transaction it opens; `SET
/// TRANSACTION` updates the session defaults used by subsequently-started transactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TxnCharacteristics {
    /// Requested isolation level, or `None` to keep the session default.
    pub isolation: Option<nusadb_core::engine::IsolationLevel>,
    /// `Some(true)` = `READ ONLY`, `Some(false)` = `READ WRITE`, `None` = session default.
    pub read_only: Option<bool>,
}

/// A validated, type-checked statement — the analyzer's output.
///
/// Every table/column reference is resolved and every expression is typed, so
/// downstream stages never need the catalog to interpret a `LogicalPlan`.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// `CREATE TABLE`.
    CreateTable(CreateTablePlan),
    /// `CREATE TABLE ... AS <select>` — schema derived from the analyzed source query.
    CreateTableAs(CreateTableAsPlan),
    /// `DROP TABLE`.
    DropTable(DropTablePlan),
    /// A sequence of plans executed in order within one statement transaction (the multi-object
    /// DDL desugar, e.g. `DROP TABLE a, b`); the result is the last child's.
    Batch(Vec<Self>),
    /// `CREATE MATERIALIZED VIEW` — body analyzed; materialized at execution.
    CreateMaterializedView(MaterializedViewPlan),
    /// `CREATE VIEW` — non-materialized; stores the defining SQL, inlined on read.
    CreateView(CreatePlainViewPlan),
    /// `DROP [MATERIALIZED] VIEW` — drops the materialized view's backing table.
    DropView(DropViewPlan),
    /// `CREATE TYPE name AS ENUM (...)` — persist a user-defined enum type (B-ENUM).
    CreateEnum(ast::CreateEnum),
    /// `DROP TYPE [IF EXISTS] name` — drop a user-defined type (B-ENUM).
    DropType(ast::DropType),
    /// `CREATE [OR REPLACE] TRIGGER ...` — a triggered SQL action to persist.
    CreateTrigger(CreateTriggerPlan),
    /// `DROP TRIGGER [IF EXISTS] name ON table`.
    DropTrigger(DropTriggerPlan),
    /// `ALTER TRIGGER name ON table RENAME TO new_name`.
    AlterTrigger(AlterTriggerPlan),
    /// `CREATE [OR REPLACE] PROCEDURE ...` — a stored procedure to persist.
    CreateProcedure(CreateProcedurePlan),
    /// `DROP PROCEDURE [IF EXISTS] name`.
    DropProcedure(DropProcedurePlan),
    /// `CALL name(args)` — invoke a stored procedure with constant arguments.
    Call(CallPlan),
    /// `CREATE [OR REPLACE] FUNCTION ...` — a SQL scalar function to persist.
    CreateFunction(CreateFunctionPlan),
    /// `DROP FUNCTION [IF EXISTS] name`.
    DropFunction(DropFunctionPlan),
    /// `REFRESH MATERIALIZED VIEW name` — recompute the named view's stored rows.
    RefreshMaterializedView(String),
    /// `CREATE POLICY` — a validated row-level-security policy to persist.
    CreatePolicy(CreatePolicyPlan),
    /// `DROP POLICY [IF EXISTS] name ON table`.
    DropPolicy(DropPolicyPlan),
    /// `ALTER TABLE`.
    AlterTable(AlterTablePlan),
    /// `INSERT`.
    Insert(InsertPlan),
    /// `SELECT`. Boxed: a [`SelectPlan`] is by far the largest plan variant, so
    /// boxing keeps `LogicalPlan` small for the many lightweight statements.
    Select(Box<SelectPlan>),
    /// `UPDATE`.
    Update(UpdatePlan),
    /// `DELETE`.
    Delete(DeletePlan),
    /// `MERGE INTO ... USING ... ON ... WHEN [NOT] MATCHED ...`.
    Merge(MergePlan),
    /// `EXPLAIN [ANALYZE] [VERBOSE] <statement>` — wraps the validated plan of the inner statement
    /// so the executor can format it (and, for `ANALYZE`, also execute and time it).
    Explain(Box<Self>, crate::ast::ExplainOptions),
    /// `BEGIN` — open an explicit transaction in the current session, applying the requested
    /// isolation / access-mode characteristics.
    BeginTransaction(TxnCharacteristics),
    /// `COMMIT` — commit the explicit transaction.
    Commit,
    /// `ROLLBACK` — abort the explicit transaction.
    Rollback,
    /// `SET TRANSACTION` — update the session's default transaction characteristics.
    SetTransaction(TxnCharacteristics),
    /// `SAVEPOINT name` — mark a rollback point in the current transaction.
    Savepoint(String),
    /// `ROLLBACK TO [SAVEPOINT] name` — undo back to a named savepoint.
    RollbackToSavepoint(String),
    /// `RELEASE [SAVEPOINT] name` — discard a named savepoint, keeping its writes.
    ReleaseSavepoint(String),
    /// `SET name = value` (`value: Some`) / `RESET name` (`value: None`) — a session variable.
    SetVariable {
        /// Variable name, folded to lowercase.
        name: String,
        /// New value for `SET`, or `None` to `RESET` it to the unset default.
        value: Option<String>,
    },
    /// `SHOW name` — report a session variable's current value.
    ShowVariable(String),
    /// `SHOW TABLES` — list the database's tables.
    ShowTables,
    /// `SHOW COLUMNS FROM t` — list the resolved table's columns.
    ShowColumns(TableSchema),
    /// `VACUUM [FULL] [ANALYZE]` — reclaim dead row versions across all tables (and, for `ANALYZE`,
    /// recompute every table's statistics).
    Vacuum(crate::ast::VacuumOptions),
    /// `REINDEX ...` — accepted as a no-op (NusaDB's B-tree indexes are always consistent).
    Reindex,
    /// `ANALYZE` — recompute statistics for a table's columns.
    Analyze(AnalyzePlan),
    /// `LOCK TABLE` — acquire a table-level lock on each resolved table.
    LockTable {
        /// Resolved tables to lock, in order.
        tables: Vec<TableSchema>,
        /// The lock mode.
        mode: crate::ast::LockMode,
    },
    /// `PREPARE` — store a parameterized statement in the session.
    Prepare {
        /// The prepared statement's name.
        name: String,
        /// The un-analyzed statement (its `$n` placeholders are bound at `EXECUTE`).
        statement: Box<crate::ast::Statement>,
        /// Number of `$1..$n` placeholders the statement references.
        param_count: usize,
    },
    /// `EXECUTE` — run a prepared statement with constant arguments.
    Execute {
        /// The prepared statement's name.
        name: String,
        /// Constant argument values, one per placeholder, in order.
        args: Vec<ast::Value>,
    },
    /// `DEALLOCATE` — discard a prepared statement, or all of them.
    Deallocate(crate::ast::DeallocateTarget),
    /// `COMMENT ON` — target resolved against the catalog.
    Comment(CommentPlan),
    /// `CREATE SCHEMA`.
    CreateSchema(CreateSchemaPlan),
    /// `DROP SCHEMA`.
    DropSchema(DropSchemaPlan),
    /// `CREATE DATABASE` — single-database compatibility no-op.
    CreateDatabase(CreateDatabasePlan),
    /// `ALTER DATABASE` — single-database compatibility no-op.
    AlterDatabase(AlterDatabasePlan),
    /// `DROP DATABASE` — drop every table in the single database (backup-then-drop, or forced).
    DropDatabase(DropDatabasePlan),
    /// `CREATE SEQUENCE`.
    CreateSequence(CreateSequencePlan),
    /// `DROP SEQUENCE`.
    DropSequence(DropSequencePlan),
    /// `CREATE INDEX`.
    CreateIndex(CreateIndexPlan),
    /// `DROP INDEX`.
    DropIndex(DropIndexPlan),
    /// `UNION` / `INTERSECT` / `EXCEPT` set operation.
    SetOperation(SetOpPlan),
}

/// A set-operation tree, left-associative as the parser built it. Generic over the leaf type so the
/// analyzer's tree (`SetOpTree<SelectPlan>`) lowers to the executor's (`SetOpTree<PhysicalOperator>`).
#[derive(Debug, Clone, PartialEq)]
pub enum SetOpTree<L> {
    /// A leaf `SELECT` branch.
    Leaf(Box<L>),
    /// `left {op} [ALL] right`.
    Node {
        /// The set operator.
        op: ast::SetOp,
        /// `true` for `… ALL` (keep duplicates / multiset semantics).
        all: bool,
        /// Left operand.
        left: Box<Self>,
        /// Right operand.
        right: Box<Self>,
    },
}

/// `UNION`/`INTERSECT`/`EXCEPT` — the resolved tree plus the combined `ORDER BY`/`LIMIT`.
#[derive(Debug, Clone, PartialEq)]
pub struct SetOpPlan {
    /// The set-operation tree; leaves are resolved `SELECT` plans.
    pub tree: SetOpTree<SelectPlan>,
    /// Output column names (taken from the leftmost branch).
    pub columns: Vec<String>,
    /// Output column types, UNIFIED across every branch: mixed numeric
    /// branches widen (`1 UNION 2.5` is NUMERIC), so the described row shape matches the
    /// values the branches actually produce — not just the leftmost leaf's typing.
    pub column_types: Vec<ColumnType>,
    /// `ORDER BY` over the combined result, resolved against the output columns.
    pub order_by: Vec<OrderByKey>,
    /// `LIMIT` on the combined result.
    pub limit: Option<u64>,
}

/// Physical form of [`SetOpPlan`]: leaves lowered to operator pipelines.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSetOp {
    /// The set-operation tree; leaves are operator pipelines.
    pub tree: SetOpTree<PhysicalOperator>,
    /// Output column names.
    pub columns: Vec<String>,
    /// Output column types, unified across every branch.
    pub column_types: Vec<ColumnType>,
    /// `ORDER BY` over the combined result.
    pub order_by: Vec<OrderByKey>,
    /// `LIMIT` on the combined result.
    pub limit: Option<u64>,
}

/// `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (cols) [INCLUDE (cols)]` — table and
/// columns resolved against the catalog into an [`IndexDef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateIndexPlan {
    /// Resolved index definition (name, target table id, key/include columns, kind, uniqueness).
    /// Used by the default B-tree path; ignored when [`vector`](Self::vector) is `Some`.
    pub def: IndexDef,
    /// `Some` for a `USING hnsw` vector index: the executor records it in the SQL-layer
    /// vector-index catalog instead of creating an engine B-tree index. `None` for a B-tree index.
    pub vector: Option<VectorIndexSpec>,
    /// Whether `IF NOT EXISTS` was given.
    pub if_not_exists: bool,
}

/// A resolved `CREATE INDEX ... USING hnsw (col)` vector index.
///
/// The graph itself is built on demand from a table scan and cached in memory; only this declaration
/// is persisted (in the SQL-layer vector-index catalog), so the index survives restarts even though
/// its graph does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorIndexSpec {
    /// Index name (the catalog key).
    pub name: String,
    /// Target table name (the catalog stores it by name for read-path lookup).
    pub table: String,
    /// The single indexed `VECTOR(n)` column's name.
    pub column: String,
    /// The indexed column's ordinal in the table.
    pub column_ordinal: usize,
    /// The vector dimension `n` (every indexed value must match it).
    pub dim: usize,
}

/// `DROP INDEX [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropIndexPlan {
    /// Index name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CREATE SEQUENCE [IF NOT EXISTS] name [options]` — options folded into a [`SequenceDef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSequencePlan {
    /// Resolved sequence definition (name + start/increment/bounds/cycle).
    pub def: SequenceDef,
    /// Whether `IF NOT EXISTS` was given.
    pub if_not_exists: bool,
}

/// `DROP SEQUENCE [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSequencePlan {
    /// Sequence name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CREATE SCHEMA [IF NOT EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSchemaPlan {
    /// Schema name.
    pub name: String,
    /// Whether `IF NOT EXISTS` was given.
    pub if_not_exists: bool,
}

/// `DROP SCHEMA [IF EXISTS] name [CASCADE | RESTRICT]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSchemaPlan {
    /// Schema name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
    /// `CASCADE` drops the schema's member tables with it; the default (`RESTRICT`) refuses a
    /// non-empty schema.
    pub cascade: bool,
}

/// `CREATE DATABASE [IF NOT EXISTS] name` — a single-database compatibility no-op (NusaDB is one
/// database per data directory; the name is recorded only for the command tag / EXPLAIN).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateDatabasePlan {
    /// Database name.
    pub name: String,
    /// Whether `IF NOT EXISTS` was given.
    pub if_not_exists: bool,
}

/// `DROP DATABASE [IF EXISTS] name` — drop every table in the single database. Plain form backs each
/// table up to `{name}_{datetime}_{table}` first; `force` (`FIX DROP DATABASE`) skips the backup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropDatabasePlan {
    /// Database name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
    /// `true` for `FIX DROP DATABASE` — drop permanently, skipping the safety backup.
    pub force: bool,
}

/// `ALTER DATABASE name ...` — the single-database compatibility no-op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterDatabasePlan {
    /// Database name (recorded for the command tag / EXPLAIN; otherwise unused).
    pub name: String,
}

/// `COMMENT ON {TABLE | COLUMN}` — target resolved (existence-checked) against the catalog.
///
/// Persisting the comment in catalog metadata is optional (treaty work); the executor
/// validates and reports success as a metadata no-op for now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentPlan {
    /// Resolved target table name.
    pub table: String,
    /// Resolved target column, or `None` when the comment targets the table itself.
    pub column: Option<String>,
    /// The comment text; `None` for `IS NULL` (clear the comment).
    pub comment: Option<String>,
}

/// `ANALYZE` — table resolved, target columns resolved to ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzePlan {
    /// Resolved target table.
    pub table: TableSchema,
    /// Ordinals (into `table.columns`) to compute statistics for; never empty
    /// (the analyzer expands a bare `ANALYZE t` to every column).
    pub columns: Vec<usize>,
}

/// `CREATE TABLE` — validated for unique column names and name availability.
// Not `Eq`: `columns` carry `ast::ColumnDef`, whose `default`/`generated` hold `Expr` (PartialEq).
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTablePlan {
    /// Schema (namespace) to create the table in; `public` when unqualified.
    pub schema: String,
    /// Table name.
    pub table: String,
    /// Columns in declaration order.
    pub columns: Vec<ast::ColumnDef>,
    /// Resolved `PRIMARY KEY` / `UNIQUE` constraints to register (column-level + table-level),
    /// which the executor declares via `add_unique_constraint` so INSERT/UPDATE enforce them.
    pub unique_constraints: Vec<UniqueConstraintSpec>,
    /// Resolved `FOREIGN KEY` constraints to register via `add_foreign_key`; the parent
    /// table name is resolved to a `TableId` by the executor against the live catalog.
    pub foreign_keys: Vec<ForeignKeySpec>,
    /// Resolved `CHECK` constraints: the predicate's SQL text, persisted via
    /// `add_check_constraint` and re-parsed + enforced per row on later `INSERT`/`UPDATE`.
    pub check_constraints: Vec<CheckSpec>,
    /// Resolved column `DEFAULT`s as `(column name, default SQL text)`. Persisted in the
    /// column-default catalog and re-parsed to fill an omitted column on later `INSERT`.
    pub defaults: Vec<(String, String)>,
    /// Whether `IF NOT EXISTS` was given (the table may already exist).
    pub if_not_exists: bool,
}

/// A resolved `CHECK` constraint.
///
/// Carries its name and the predicate's canonical SQL text. The predicate was type-checked
/// (boolean, columns exist) at analysis time; the text is what is persisted and re-parsed for
/// per-row enforcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckSpec {
    /// Constraint name (generated if the user gave none).
    pub name: String,
    /// The predicate as canonical SQL text.
    pub predicate_sql: String,
}

/// `CREATE MATERIALIZED VIEW name AS <select>` (M6, logical).
///
/// `body` is the analyzed `SELECT`; `columns` is its derived output schema (name + type), with any
/// explicit column-name override already applied. Materialization (run the body, store the rows)
/// happens at execution.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedViewPlan {
    /// Materialized view name (also its backing table name).
    pub name: String,
    /// Whether `OR REPLACE` was given (drop an existing view of the same name first).
    pub or_replace: bool,
    /// Whether `IF NOT EXISTS` was given: an existing view/table of the name makes this a no-op.
    pub if_not_exists: bool,
    /// Output schema: `(column name, type)` in projection order.
    pub columns: Vec<(String, ColumnType)>,
    /// The analyzed view body.
    pub body: Box<SelectPlan>,
    /// The view body as canonical SQL, persisted so `REFRESH` can re-execute it.
    pub definition_sql: String,
    /// `Some(base_table)` when the body is eligible for incremental view maintenance: a
    /// single base table, projection + filter only (no join/aggregate/distinct/group/window/limit),
    /// and stable (subquery- and volatile-free) expressions. The executor then maintains the view
    /// incrementally on writes to `base_table` instead of requiring a full `REFRESH`. `None` =
    /// full-refresh-only.
    pub ivm_base: Option<String>,
}

/// `CREATE MATERIALIZED VIEW` lowered for execution: the body is an executable operator tree.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalMaterializedView {
    /// Materialized view name (also its backing table name).
    pub name: String,
    /// Whether `OR REPLACE` was given.
    pub or_replace: bool,
    /// Whether `IF NOT EXISTS` was given: an existing view/table of the name makes this a no-op.
    pub if_not_exists: bool,
    /// Output schema: `(column name, type)` in projection order.
    pub columns: Vec<(String, ColumnType)>,
    /// The lowered view body.
    pub body: Box<PhysicalOperator>,
    /// The view body as canonical SQL, persisted so `REFRESH` can re-execute it.
    pub definition_sql: String,
    /// `Some(base_table)` when the view is incrementally maintainable; see
    /// [`MaterializedViewPlan::ivm_base`].
    pub ivm_base: Option<String>,
}

/// `CREATE TABLE ... AS <select>`: an analyzed source query plus the derived table schema.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableAsPlan {
    /// New table name.
    pub name: String,
    /// Derived schema: `(column name, type)` in projection order.
    pub columns: Vec<(String, ColumnType)>,
    /// The analyzed source query whose rows seed the table.
    pub body: Box<SelectPlan>,
    /// Whether `IF NOT EXISTS` was given (an existing table makes the statement a no-op).
    pub if_not_exists: bool,
}

/// `CREATE TABLE ... AS <select>` lowered for execution: the source query is an operator tree.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalCreateTableAs {
    /// New table name.
    pub name: String,
    /// Derived schema: `(column name, type)` in projection order.
    pub columns: Vec<(String, ColumnType)>,
    /// The lowered source query.
    pub body: Box<PhysicalOperator>,
    /// Whether `IF NOT EXISTS` was given.
    pub if_not_exists: bool,
}

/// `DROP [MATERIALIZED] VIEW name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropViewPlan {
    /// View name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CREATE [OR REPLACE] TRIGGER ...` — a validated trigger ready to persist.
///
/// The `WHEN` guard and the action are kept as canonical SQL and re-analyzed (with `NEW`/`OLD`
/// substituted) when the trigger fires. The same struct is the analyzer's and the executor's view
/// (no lowering needed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTriggerPlan {
    /// Trigger name (unique per table).
    pub name: String,
    /// Whether `OR REPLACE` was given (replace a same-named trigger on the table).
    pub or_replace: bool,
    /// The table the trigger is attached to.
    pub table: String,
    /// When the trigger fires relative to the write.
    pub timing: crate::ast::TriggerTiming,
    /// The events it fires on (one or more); never empty.
    pub events: Vec<crate::ast::TriggerEvent>,
    /// Per-row or per-statement granularity.
    pub for_each: crate::ast::TriggerForEach,
    /// The `WHEN (cond)` guard as raw SQL, or `None`.
    pub when: Option<String>,
    /// The triggered action as raw SQL (a single data statement).
    pub action: String,
}

/// `DROP TRIGGER [IF EXISTS] name ON table`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTriggerPlan {
    /// Trigger name.
    pub name: String,
    /// The table the trigger is attached to.
    pub table: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `ALTER TRIGGER name ON table RENAME TO new_name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterTriggerPlan {
    /// Current trigger name.
    pub name: String,
    /// The table the trigger is attached to.
    pub table: String,
    /// The new trigger name.
    pub new_name: String,
}

/// `CREATE [OR REPLACE] PROCEDURE ...` — a validated stored procedure ready to persist.
///
/// The body is kept as canonical SQL (one or more `;`-separated data statements) and re-parsed +
/// run with `$1..$n` bound to the call arguments when the procedure is invoked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateProcedurePlan {
    /// Procedure name (unique).
    pub name: String,
    /// Whether `OR REPLACE` was given.
    pub or_replace: bool,
    /// The number of declared `IN` parameters (the call must supply exactly this many arguments).
    pub param_count: usize,
    /// The `OUT` parameter names, in declaration order; `CALL` returns their final values.
    pub out_params: Vec<String>,
    /// The procedure body as raw SQL.
    pub body: String,
}

/// `DROP PROCEDURE [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropProcedurePlan {
    /// Procedure name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CALL name(args)` — a procedure invocation with its arguments reduced to constants.
#[derive(Debug, Clone, PartialEq)]
pub struct CallPlan {
    /// The procedure name.
    pub name: String,
    /// The constant argument values, in order, bound to the body's `$1..$n`.
    pub args: Vec<crate::ast::Value>,
}

/// `CREATE [OR REPLACE] FUNCTION ...` — a validated SQL scalar function ready to persist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateFunctionPlan {
    /// Function name (unique).
    pub name: String,
    /// Whether `OR REPLACE` was given.
    pub or_replace: bool,
    /// The number of declared parameters.
    pub param_count: usize,
    /// Declared parameter names, in order (lowercase-folded) — persisted so a call can bind arguments
    /// to the body's named references as well as `$1`..`$n`.
    pub param_names: Vec<String>,
    /// The function body as canonical SQL (a `SELECT <expr>`).
    pub body: String,
}

/// `DROP FUNCTION [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropFunctionPlan {
    /// Function name.
    pub name: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CREATE POLICY` — a validated row-level-security policy ready to persist.
///
/// The `USING` / `WITH CHECK` predicates are kept as canonical SQL (already type-checked against the
/// table as boolean) and re-analyzed against the table when the policy is enforced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatePolicyPlan {
    /// Policy name (unique per table).
    pub name: String,
    /// The table the policy governs.
    pub table: String,
    /// `true` when persisting should replace any existing policy of this `name` (an `ALTER POLICY`
    /// lowered to a full row rewrite); `false` for `CREATE POLICY`, where a duplicate is an error.
    pub replace: bool,
    /// `true` for a permissive policy (grants access, `OR`-combined); `false` for a restrictive
    /// policy (narrows access, `AND`-combined on top of the permissive predicate).
    pub permissive: bool,
    /// The command the policy applies to.
    pub command: crate::ast::PolicyCommand,
    /// Roles the policy applies to; empty means `PUBLIC`.
    pub roles: Vec<String>,
    /// `USING` row-visibility predicate as canonical SQL, or `None`.
    pub using: Option<String>,
    /// `WITH CHECK` write predicate as canonical SQL, or `None`.
    pub check: Option<String>,
}

/// `DROP POLICY [IF EXISTS] name ON table`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropPolicyPlan {
    /// Policy name.
    pub name: String,
    /// The table the policy is attached to.
    pub table: String,
    /// Whether `IF EXISTS` was given.
    pub if_exists: bool,
}

/// `CREATE [OR REPLACE] VIEW name AS <select>` — a non-materialized view.
///
/// Only its defining SQL is stored; querying it re-evaluates the body (inlined by the analyzer), so
/// there is no backing table and no derived schema to carry here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatePlainViewPlan {
    /// View name.
    pub name: String,
    /// Whether `OR REPLACE` was given.
    pub or_replace: bool,
    /// Whether `IF NOT EXISTS` was given: an existing view/table of the name makes this a no-op.
    pub if_not_exists: bool,
    /// The view body as canonical SQL, stored so reads can re-parse and inline it.
    pub definition_sql: String,
    /// Explicit output column names from `CREATE VIEW name (cols) AS ...`, or empty to use the
    /// body's inferred projection names. Applied positionally when the view is inlined on read.
    pub columns: Vec<String>,
}

/// A resolved `PRIMARY KEY` / `UNIQUE` constraint to declare at `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniqueConstraintSpec {
    /// Catalog name for the constraint (and its backing unique index).
    pub name: String,
    /// The constrained columns, in declaration order (validated to exist).
    pub columns: Vec<String>,
    /// `true` for `PRIMARY KEY` (also enforces at-most-one-per-table), `false` for `UNIQUE`.
    pub primary: bool,
}

/// A resolved `FOREIGN KEY` constraint to declare at `CREATE TABLE`. The child columns
/// reference the parent table's primary key; the parent name is resolved to a `TableId` at execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignKeySpec {
    /// Catalog name for the constraint (and its backing child-side index).
    pub name: String,
    /// The referencing (child) columns, in order (validated to exist on the new table).
    pub columns: Vec<String>,
    /// The referenced (parent) table name.
    pub parent_table: String,
    /// Explicitly named referenced (parent) columns, or empty for "the parent's primary key".
    /// The executor rejects an explicit list that is not exactly the parent's primary key (v1
    /// references the PK only — silently redirecting to the PK would mis-enforce).
    pub referred_columns: Vec<String>,
    /// Action when a referenced parent row is deleted.
    pub on_delete: nusadb_core::FkAction,
    /// Action when a referenced parent key is updated.
    pub on_update: nusadb_core::FkAction,
}

/// `DROP TABLE` — validated for table existence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTablePlan {
    /// Schema (namespace) the table lives in; `public` when unqualified.
    pub schema: String,
    /// Table name.
    pub table: String,
    /// Whether `IF EXISTS` was given (the table may be absent).
    pub if_exists: bool,
    /// Whether `CASCADE` was given: referencing FOREIGN KEYs drop instead of blocking.
    pub cascade: bool,
}

/// `ALTER TABLE` — one resolved schema mutation against a resolved table.
///
/// `IF EXISTS` on a missing table, `ADD COLUMN IF NOT EXISTS` on an existing
/// column, and `DROP COLUMN IF EXISTS` on a missing column all collapse to
/// [`AlterTablePlan::Noop`] at analysis time, so the executor never re-checks
/// those conditions.
// Not `Eq`: `Apply.op` is `AlterColumnOp`, which can carry an `Expr`-bearing `ColumnDef`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterTablePlan {
    /// Apply `op` to the resolved `table` (its pre-alter schema).
    Apply {
        /// The target table's schema *before* the alteration. The executor
        /// decodes existing rows with this schema when a rewrite is required.
        table: TableSchema,
        /// The single operation to apply.
        op: AlterColumnOp,
    },
    /// An `IF [NOT] EXISTS` guard made the statement a no-op.
    Noop,
    /// `ENABLE`/`DISABLE ROW LEVEL SECURITY` — toggle the table's row-level-security flag in the
    /// SQL-layer catalog. Not a storage-engine schema change, so it does not flow through the
    /// `AlterOp` treaty path.
    SetRls {
        /// The target table's name (already resolved to exist at analysis time).
        table: String,
        /// `true` to enable row-level security, `false` to disable it.
        enabled: bool,
    },
    /// `{ENABLE|DISABLE} TRIGGER {name|ALL}` — flip a trigger's enabled flag in the SQL-layer
    /// trigger catalog. Not a storage-engine schema change.
    SetTriggerEnabled {
        /// The target table's name (already resolved to exist at analysis time).
        table: String,
        /// The trigger to toggle, or `None` for every trigger on the table.
        name: Option<String>,
        /// `true` to enable, `false` to disable.
        enabled: bool,
    },
    /// `ADD [CONSTRAINT name] PRIMARY KEY/UNIQUE (cols)` — the executor first validates that
    /// the table's existing rows satisfy the constraint, then registers it. Not a row rewrite, so it
    /// does not flow through the `AlterOp` treaty path.
    AddUniqueConstraint {
        /// Target table schema (used to validate existing rows and resolve the column ordinals).
        table: TableSchema,
        /// The constraint name (generated from the table and columns if the user gave none).
        name: String,
        /// The key column names, in declaration order.
        columns: Vec<String>,
        /// `true` for `PRIMARY KEY` (key columns must also be non-`NULL`), `false` for `UNIQUE`.
        primary: bool,
    },
    /// `ADD [CONSTRAINT name] FOREIGN KEY (cols) REFERENCES parent (...)` — the executor
    /// registers the FK, then validates that the table's existing rows reference live parent rows.
    AddForeignKey {
        /// Target (child) table schema — used to validate existing rows after registration.
        table: TableSchema,
        /// The resolved foreign-key spec (child columns already validated to exist).
        fk: ForeignKeySpec,
    },
    /// `ADD [CONSTRAINT name] CHECK (expr)` — the executor validates the table's existing
    /// rows against `predicate`, then persists `predicate_sql` so later writes re-parse and enforce
    /// it. A row passes when the predicate is `TRUE` or `NULL` (only `FALSE` fails).
    AddCheck {
        /// Target table schema — used to validate existing rows after registration.
        table: TableSchema,
        /// The constraint name (generated if the user gave none).
        name: String,
        /// The predicate's canonical SQL text (persisted in the constraint catalog).
        predicate_sql: String,
        /// The type-checked predicate, evaluated against each existing row to validate the ADD.
        predicate: TypedExpr,
    },
    /// `DROP CONSTRAINT [IF EXISTS] name` — drop a named constraint and its backing index.
    DropConstraint {
        /// Target table id.
        table: nusadb_core::TableId,
        /// The constraint name.
        name: String,
        /// Whether `IF EXISTS` was given (a missing constraint is then a no-op).
        if_exists: bool,
    },
    /// `RENAME TO name` — rename the table in the catalog (no row rewrite).
    RenameTable {
        /// Target table id.
        table: nusadb_core::TableId,
        /// The new table name (validated free at analysis time).
        name: String,
    },
}

/// A resolved single `ALTER TABLE` action.
///
/// Column references are resolved to ordinals against the pre-alter schema so
/// the executor can rewrite rows without consulting the catalog again. Only the
/// operations the [`AlterOp`](nusadb_core::AlterOp) treaty models are
/// representable; column `DEFAULT` and constraint actions are rejected earlier
/// (the catalog has no column-default or analysis-time constraint hook yet).
// Not `Eq`: `AddColumn` carries `ast::ColumnDef`, whose `default`/`generated` hold `Expr`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterColumnOp {
    /// `ADD COLUMN` — append `column`. Existing rows gain a trailing slot
    /// (`NULL`, or rejected at execution if the column is `NOT NULL` and the
    /// table is non-empty, since there is no default to backfill).
    AddColumn(ast::ColumnDef),
    /// `DROP COLUMN` at ordinal `index`.
    DropColumn {
        /// Ordinal of the column to remove.
        index: usize,
    },
    /// `RENAME COLUMN` — catalog-only (no row rewrite).
    RenameColumn {
        /// Ordinal of the column to rename.
        index: usize,
        /// New column name.
        to: String,
    },
    /// `ALTER COLUMN … SET DATA TYPE` — each row's value at `index` is cast to
    /// `ty` and re-encoded.
    SetType {
        /// Ordinal of the column to retype.
        index: usize,
        /// New physical type.
        ty: ColumnType,
    },
    /// `ALTER COLUMN … SET NOT NULL` — validated (no existing `NULL`s) then a
    /// catalog-only flag change.
    SetNotNull {
        /// Ordinal of the column to mark `NOT NULL`.
        index: usize,
    },
    /// `ALTER COLUMN … DROP NOT NULL` — catalog-only.
    DropNotNull {
        /// Ordinal of the column to make nullable.
        index: usize,
    },
    /// `ALTER COLUMN … SET DEFAULT <expr>` — upsert the column's default in the column-default
    /// catalog. Validated assignable + subquery-free at analysis time.
    SetDefault {
        /// The column whose default is being set.
        column: String,
        /// The default's canonical SQL text to persist.
        default_sql: String,
    },
    /// `ALTER COLUMN … DROP DEFAULT` — remove the column's default from the catalog.
    DropDefault {
        /// The column whose default is being removed.
        column: String,
    },
}

/// `INSERT` — table resolved, target columns resolved, rows type-checked.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertPlan {
    /// Resolved target table.
    pub table: TableSchema,
    /// Ordinals (into `table.columns`) each supplied value maps to.
    pub columns: Vec<usize>,
    /// Where the rows to insert come from: literal `VALUES` or a `SELECT`.
    pub source: InsertSource,
    /// `RETURNING` output columns, resolved against the inserted row's columns. Empty when
    /// the statement has no `RETURNING` clause (then the result is a row count, not a row set).
    pub returning: Vec<Projection>,
    /// Row-level-security `WITH CHECK` predicate every inserted row must satisfy, resolved
    /// against the table's columns. `None` for a superuser or an RLS-free table; `FALSE` (default
    /// deny) when RLS is on but no `INSERT`/`ALL` policy grants the write.
    pub rls_check: Option<TypedExpr>,
    /// The `ON CONFLICT` clause, or `None` for a plain `INSERT`. `DoNothing` skips a row
    /// that would violate a `PRIMARY KEY`/`UNIQUE` constraint; `DoUpdate` updates the existing
    /// conflicting row instead (upsert).
    pub on_conflict: Option<OnConflictPlan>,
}

/// The resolved `ON CONFLICT` action of an `INSERT`.
#[derive(Debug, Clone, PartialEq)]
pub enum OnConflictPlan {
    /// `DO NOTHING` — silently skip a row that would violate a `PRIMARY KEY`/`UNIQUE` constraint.
    DoNothing,
    /// `DO UPDATE SET ... [WHERE ...]` — update the existing conflicting row from the proposed one.
    DoUpdate {
        /// The conflict arbiter; the executor resolves it to the matching `PRIMARY KEY`/`UNIQUE`
        /// constraint whose key the proposed row collides on.
        target: ConflictArbiter,
        /// `SET` assignments as `(target-column ordinal, value)`. Each value is resolved against the
        /// combined scope — the existing row's columns at ordinals `[0, n)` and the proposed
        /// (`EXCLUDED`) row's columns at `[n, 2n)`, where `n = table.columns.len()`.
        assignments: Vec<(usize, TypedExpr)>,
        /// Optional `WHERE` predicate over the same combined scope; the update is applied only when
        /// it evaluates to `TRUE`.
        filter: Option<TypedExpr>,
    },
}

/// How an `ON CONFLICT DO UPDATE` names the unique constraint to arbitrate on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictArbiter {
    /// `ON CONFLICT (col, ...)` — the target columns' ordinals; the executor checks they form a
    /// declared `PRIMARY KEY`/`UNIQUE` key.
    Columns(Vec<usize>),
    /// `ON CONFLICT ON CONSTRAINT name` — a named constraint the executor looks up.
    Constraint(String),
}

/// The row source of an `INSERT`.
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// `INSERT ... VALUES` — one typed expression tuple per row; each row has `columns.len()`
    /// entries, evaluated against an empty input row. A `None` entry is an explicit `DEFAULT`
    /// cell: the executor fills it from the target column's default/serial/NULL at insert time.
    Values(Vec<Vec<Option<TypedExpr>>>),
    /// `INSERT ... SELECT` — rows produced by a subquery whose output columns align with the
    /// target `columns`. Carries the logical [`SelectPlan`]; the executor lowers it with
    /// [`plan_select`](crate::planner::plan_select) and runs it (an `INSERT` is not a hot path).
    Select(Box<SelectPlan>),
}

/// `SELECT` — source resolved, projection and predicates type-checked.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectPlan {
    /// Resolved base source table; `None` for a `SELECT` without `FROM` or one whose base source is
    /// a CTE (see `from_cte`).
    pub table: Option<TableSchema>,
    /// Inline `VALUES` rows for a `(VALUES ...) AS x` derived-table base; empty for every other plan.
    /// When non-empty, the planner emits a [`PhysicalOperator::Values`] as the base source (in place
    /// of `OneRow`/`SeqScan`), and `projection` selects from those columns. Each row has the same
    /// arity as the others; the column types are unified per column at analysis time.
    pub values: Vec<Vec<TypedExpr>>,
    /// Set-operation source for a `(SELECT ... UNION ...) AS x` derived-table base; `None` for every
    /// other plan. When `Some`, the planner emits a [`PhysicalOperator::SetOperation`] as the base
    /// source, and `projection` selects from its output columns. Mutually exclusive with `table`,
    /// `from_cte`, and `values`.
    pub set_op_source: Option<Box<SetOpPlan>>,
    /// `WITH`-clause base source: when the `FROM` base names a non-recursive CTE, this holds
    /// the CTE's already-planned `SELECT`, inlined by the planner in place of a `SeqScan`. Its output
    /// columns (in projection order) form the base scope. Mutually exclusive with `table`.
    pub from_cte: Option<Box<Self>>,
    /// Joins applied left-to-right onto `table`. Empty for the single-table
    /// (or no-`FROM`) case. Column ordinals throughout the plan index the
    /// concatenated joined row `[table cols ++ join0 cols ++ ...]`.
    pub joins: Vec<JoinPlan>,
    /// `SELECT DISTINCT` — when set, duplicate output rows are removed after
    /// projection. Mutually exclusive with `distinct_on`.
    pub distinct: bool,
    /// `SELECT DISTINCT ON (keys)` — when non-empty, keep the first source row per distinct
    /// `keys` tuple (evaluated against the source row, ordered by `order_by`). Empty otherwise.
    pub distinct_on: Vec<TypedExpr>,
    /// Output columns, in order. After scalar aggregation, projection
    /// expressions reference computed aggregates via
    /// [`TypedExprKind::AggregateRef`].
    pub projection: Vec<Projection>,
    /// `WHERE` predicate (guaranteed boolean-typed).
    pub filter: Option<TypedExpr>,
    /// `ORDER BY` keys, in priority order. For an aggregated `SELECT` these
    /// reference the synthesized post-aggregation row (see `aggregates`).
    pub order_by: Vec<OrderByKey>,
    /// `LIMIT` row cap.
    pub limit: Option<u64>,
    /// `OFFSET` — rows to skip before `LIMIT`. `None` for none.
    pub offset: Option<u64>,
    /// `GROUP BY` key expressions (evaluated against source rows). When
    /// non-empty the planner inserts a [`PhysicalOperator::GroupAggregate`].
    /// For `ROLLUP`/`CUBE`/`GROUPING SETS` this is the *union* of every column
    /// mentioned by any grouping set (see `grouping_sets`).
    pub group_keys: Vec<TypedExpr>,
    /// `ROLLUP`/`CUBE`/`GROUPING SETS`. Empty for a plain `GROUP BY`.
    /// When non-empty, each entry is the set of indices into `group_keys` that
    /// the grouping set activates; columns absent from a set are emitted as
    /// `NULL`. The planner inserts a [`PhysicalOperator::GroupingSetsAggregate`].
    pub grouping_sets: Vec<Vec<usize>>,
    /// Window functions extracted from the projection. When
    /// non-empty the planner inserts a [`PhysicalOperator::Window`] that appends
    /// one column per entry; the projection references them by appended ordinal.
    /// Mutually exclusive with aggregation in v1.
    pub windows: Vec<WindowExpr>,
    /// `HAVING` predicate, applied to the post-aggregation rows. References the
    /// synthesized row via [`TypedExprKind::AggregateRef`].
    pub having: Option<TypedExpr>,
    /// Aggregate calls collected from the projection (and `HAVING`/`ORDER BY`).
    /// When `group_keys` is empty but this is non-empty, the `SELECT` is a
    /// *scalar aggregate* (one global group) and the planner inserts a
    /// [`PhysicalOperator::ScalarAggregate`]. Either way the aggregation
    /// operator's output row is laid out as `[group keys ++ aggregate
    /// results]`, and projection/`HAVING`/`ORDER BY` reference its columns via
    /// [`TypedExprKind::AggregateRef`].
    pub aggregates: Vec<AggregateCall>,
    /// Secondary indexes available on the base `table`, resolved to column ordinals. The
    /// planner consults these to replace the base `SeqScan` with a [`PhysicalOperator::IndexScan`]
    /// when a `WHERE` predicate maps onto one. Empty when the base is a CTE, has no indexes, or the
    /// catalog does not expose any.
    pub indexes: Vec<IndexMeta>,
    /// The base table's ANALYZE statistics, when available. The planner uses them for
    /// cost-based plan selection — currently the index-vs-sequential-scan choice. `None` for a CTE
    /// base, an un-analyzed table, or a catalog that exposes no stats (then planning is heuristic).
    pub table_stats: Option<nusadb_core::TableStats>,
    /// The base table's `O(1)` approximate live-row count, when the catalog exposes one — the
    /// vectorized-routing cardinality fallback used when [`table_stats`](Self::table_stats) is
    /// absent (no `ANALYZE`), so a scan of a large un-analyzed table still vectorizes. `None` for a
    /// CTE base, a join, or a catalog with no cheap estimate. A routing hint only, never a
    /// correctness input.
    pub approx_scan_rows: Option<u64>,
    /// `WITH RECURSIVE` CTEs this query introduces. Each is materialized to a fixpoint
    /// before the body runs and bound to its synthetic table id; the body and the recursive term
    /// reference it as a table. Empty for a query with no recursive CTE.
    pub recursive_ctes: Vec<RecursiveCteDef>,
    /// Data-modifying CTEs this query introduces. Each runs once before the body and its
    /// `RETURNING` rows are bound to its synthetic table id; the body (and later siblings) reference
    /// it as a table. Empty for a query with no data-modifying CTE.
    pub modifying_ctes: Vec<ModifyingCteDef>,
    /// `SELECT ... FOR UPDATE` / `FOR SHARE` row lock. `Some` only for the supported shape —
    /// a single base table with no join/aggregate/grouping/window/distinct and a subquery-free
    /// predicate — validated by the analyzer; the planner wraps the pipeline in a
    /// [`PhysicalOperator::LockRows`] that locks each matched base row before output; the `bool`
    /// is `SKIP LOCKED` (skip rows another transaction holds locked instead of conflicting).
    /// `None` = no row locking.
    pub row_lock: Option<(nusadb_core::engine::RowLockMode, bool)>,
    /// `WITH ORDINALITY` on a `FROM` set-returning function: when `true`, this plan is a
    /// set-returning derived table whose `ProjectSet` appends a 1-based `ordinality` column. The
    /// analyzer sets it only for such a derived table (and requires a set-returning projection);
    /// `false` for every other plan.
    pub ordinality: bool,
    /// `FETCH FIRST n ROWS WITH TIES`: when `true`, the row cap in `limit` keeps,
    /// in addition to the first `limit` rows, every following row that ties the last kept row on the
    /// `ORDER BY` keys. The analyzer sets it only for a query with an `ORDER BY` and no
    /// `DISTINCT`/`DISTINCT ON`/set-returning projection (those change row identity above the sort);
    /// `false` for a plain `FETCH ... ONLY` / `LIMIT`.
    pub limit_with_ties: bool,
}

/// A `WITH RECURSIVE` CTE: a base term unioned with a recursive term that references the CTE
/// itself, evaluated to a fixpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveCteDef {
    /// Synthetic table id the CTE is bound to (reserved high range).
    pub id: nusadb_core::TableId,
    /// The non-recursive base term.
    pub base: Box<SelectPlan>,
    /// The recursive term, which scans the CTE's working set.
    pub recursive: Box<SelectPlan>,
    /// `UNION ALL` (keep duplicates) vs `UNION` (distinct rows only).
    pub union_all: bool,
}

/// A data-modifying CTE: `WITH x AS (INSERT/UPDATE … RETURNING …)`.
///
/// The statement runs once at execution; its RETURNING rows are bound to the synthetic table `id` so
/// the outer query (and later siblings) read them as a relation.
#[derive(Debug, Clone, PartialEq)]
pub struct ModifyingCteDef {
    /// Synthetic table id the CTE's RETURNING rows are bound to (reserved high range).
    pub id: nusadb_core::TableId,
    /// The data-modifying statement's logical plan (`LogicalPlan::Insert`/`Update`), planned and run
    /// once; its `RETURNING` rows form the relation.
    pub plan: Box<LogicalPlan>,
}

/// A base-table index, resolved for planning: its name plus the table-column ordinals it keys, in
/// index order. The resolved form of [`crate::analyzer::IndexInfo`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMeta {
    /// Index name, passed to [`PhysicalOperator::IndexScan`] and resolved to an `IndexId` at exec.
    pub name: String,
    /// Key column ordinals into `table.columns`, in index order.
    pub columns: Vec<usize>,
    /// Whether the index enforces key uniqueness (from [`crate::analyzer::IndexInfo::unique`]).
    pub unique: bool,
}

/// One resolved join in a [`SelectPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct JoinPlan {
    /// The joined-in relation's schema. For a named table this is the catalog schema; for a derived
    /// table `JOIN (SELECT ...) AS x` it is the subquery's projection schema (qualified by the
    /// alias), and the rows come from `input_cte` rather than a table scan.
    pub table: TableSchema,
    /// Join kind (`INNER` / `LEFT`).
    pub kind: ast::JoinKind,
    /// `ON` predicate (boolean-typed), resolved against the columns of the
    /// base table through this join's table.
    pub on: TypedExpr,
    /// `USING`/`NATURAL` merged-column `(kept-left, hidden-right)` ordinal pairs over the
    /// concatenated `[left ++ right]` row. The executor surfaces each merge as `coalesce(left,
    /// right)` (filling the left slot from the right when the left is NULL), which is correct for a
    /// RIGHT/FULL join's unmatched rows and a no-op for INNER/LEFT. Empty for `ON`/`CROSS` joins.
    pub coalesce: Vec<(usize, usize)>,
    /// For a derived-table join input `JOIN (SELECT ...) AS x`, the inlined subquery plan whose
    /// output forms this join's right side; `None` for a named table (scanned via `table`).
    pub input_cte: Option<Box<SelectPlan>>,
    /// `true` for a `JOIN LATERAL (SELECT ...)` whose subquery may reference columns from the FROM
    /// items to its left — the right side is re-executed per left row. Always has an
    /// `input_cte`; lowers to [`PhysicalOperator::LateralJoin`] rather than a materialized join.
    pub lateral: bool,
}

/// One output column of a [`SelectPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    /// The expression producing this column's value.
    pub expr: TypedExpr,
    /// The output column name (alias, source column name, or `?column?`).
    pub name: String,
}

/// One `ORDER BY` sort key of a [`SelectPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByKey {
    /// Sort-key expression.
    pub expr: TypedExpr,
    /// `true` for `ASC`, `false` for `DESC`.
    pub ascending: bool,
    /// `NULLS FIRST` / `NULLS LAST` placement; `Default` keeps the SQL default (NULLs last for
    /// `ASC`, first for `DESC`).
    pub nulls: ast::NullOrdering,
}

/// The `FETCH FIRST n ROWS WITH TIES` cap carried on a [`Sort`](PhysicalOperator::Sort).
///
/// The sort skips `offset` rows, keeps `count` rows, then extends the result with
/// the trailing peers of the last kept row (every following row equal to it on the sort keys).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiesLimit {
    /// Rows to skip before the kept window (`OFFSET`); `0` for none.
    pub offset: u64,
    /// Rows to keep before the tie extension (the `FETCH FIRST n` count).
    pub count: u64,
}

/// `UPDATE` — table resolved, assignments and predicate type-checked.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdatePlan {
    /// Resolved target table.
    pub table: TableSchema,
    /// `SET` assignments.
    pub assignments: Vec<Assignment>,
    /// `UPDATE ... FROM <source>` source: an additional relation the `SET` values and `WHERE`
    /// may reference. The value/`WHERE` expressions are type-checked against the concatenated
    /// `target ++ from` row, so a FROM column has ordinal `target_width + j`. `None` for a plain
    /// single-table `UPDATE`. Carries the source's schema (for the column scope); a derived source
    /// (`(VALUES ...)` / `(SELECT ...)` / set operation) additionally carries [`from_plan`](Self::from_plan).
    pub from: Option<TableSchema>,
    /// For an `UPDATE ... FROM (VALUES ...)` / `FROM (SELECT ...)` derived source, the inlined plan
    /// that produces its rows; `None` for a plain named FROM table (which the executor scans).
    pub from_plan: Option<Box<SelectPlan>>,
    /// `WHERE` predicate (guaranteed boolean-typed). With a `FROM`, evaluated against the
    /// concatenated `target ++ from` row.
    pub filter: Option<TypedExpr>,
    /// `RETURNING` output columns, resolved against the table's columns and evaluated
    /// against each row's **post-update** values. Empty when there is no `RETURNING` clause.
    pub returning: Vec<Projection>,
    /// Row-level-security `WITH CHECK` predicate each **post-update** row must satisfy,
    /// resolved against the table's columns. `None` for a superuser or an RLS-free table; the
    /// `USING` side (which rows are updatable) is folded into [`filter`](Self::filter) instead.
    pub rls_check: Option<TypedExpr>,
}

/// One `SET column = value` assignment of an [`UpdatePlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Ordinal (into the table's columns) of the column being assigned.
    pub column: usize,
    /// The new value (type-checked against the column's type).
    pub value: TypedExpr,
}

/// `DELETE` — table resolved, predicate type-checked.
#[derive(Debug, Clone, PartialEq)]
pub struct DeletePlan {
    /// Resolved target table.
    pub table: TableSchema,
    /// `DELETE ... USING <source>` source: an additional relation the `WHERE` may reference.
    /// The predicate is type-checked against the concatenated `target ++ using` row, so a USING column
    /// has ordinal `target_width + j`. `None` for a plain `DELETE`. Carries the source's schema; a
    /// derived source additionally carries [`using_plan`](Self::using_plan).
    pub using: Option<TableSchema>,
    /// For a `DELETE ... USING (VALUES ...)` / `USING (SELECT ...)` derived source, the inlined plan
    /// that produces its rows; `None` for a plain named USING table (which the executor scans).
    pub using_plan: Option<Box<SelectPlan>>,
    /// `WHERE` predicate (guaranteed boolean-typed).
    pub filter: Option<TypedExpr>,
    /// `RETURNING` output columns, resolved against the table's columns and evaluated
    /// against each row's **pre-delete** values. Empty when there is no `RETURNING` clause.
    pub returning: Vec<Projection>,
    /// `TRUNCATE ... RESTART IDENTITY`: after emptying the table, reset the backing sequence
    /// of every `SERIAL`/`IDENTITY` column so the next insert restarts at the sequence's start value.
    /// Always `false` for an ordinary `DELETE` and for `TRUNCATE` / `CONTINUE IDENTITY`.
    pub restart_identity: bool,
}

/// `MERGE INTO target USING source ON ... WHEN [NOT] MATCHED ...`.
///
/// Every clause expression is type-checked against the concatenated `target ++ source` row, so a
/// source column has ordinal `target_width + j`. For a `WHEN NOT MATCHED` clause (no matched target),
/// the executor evaluates against a row whose target half is `NULL`. v1 supports a single named target
/// and source table (a subquery / join source is rejected).
#[derive(Debug, Clone, PartialEq)]
pub struct MergePlan {
    /// Resolved target table (the one rows are inserted into / updated / deleted from).
    pub table: TableSchema,
    /// Resolved source relation joined against the target. For a derived source (`VALUES` /
    /// subquery / set operation) this carries the source's projection schema (column scope); the
    /// rows come from [`source_plan`](Self::source_plan). For a plain named table it is the catalog
    /// schema and the executor scans it directly.
    pub source: TableSchema,
    /// For a `MERGE ... USING (VALUES ...)` / `USING (SELECT ...)` derived source, the inlined plan
    /// that produces its rows; `None` for a plain named source (which the executor scans). Mirrors
    /// `UpdatePlan::from_plan` / `DeletePlan::using_plan`.
    pub source_plan: Option<Box<SelectPlan>>,
    /// The `ON` join condition (boolean), over `target ++ source`.
    pub on: TypedExpr,
    /// The ordered `WHEN` clauses; the first whose `MATCHED`/`NOT MATCHED` status and `AND` guard fit
    /// a row is applied.
    pub whens: Vec<MergeWhen>,
}

/// One `WHEN` clause of a [`MergePlan`].
#[derive(Debug, Clone, PartialEq)]
pub enum MergeWhen {
    /// `WHEN MATCHED [AND pred] THEN {UPDATE SET ... | DELETE}`.
    Matched {
        /// Optional `AND` guard (boolean over `target ++ source`).
        pred: Option<TypedExpr>,
        /// The action applied to the matched target row.
        action: MergeMatchedAction,
    },
    /// `WHEN NOT MATCHED [AND pred] THEN INSERT (cols) VALUES (...)`.
    NotMatched {
        /// Optional `AND` guard (boolean over the `NULL-target ++ source` row).
        pred: Option<TypedExpr>,
        /// Target column ordinals each inserted value maps to.
        columns: Vec<usize>,
        /// The values to insert, over `target ++ source` (target half `NULL`).
        values: Vec<TypedExpr>,
    },
}

/// The action of a `WHEN MATCHED` clause.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeMatchedAction {
    /// `UPDATE SET column = value, ...` — values type-checked over `target ++ source`.
    Update {
        /// The `SET` assignments (column ordinal + value).
        assignments: Vec<Assignment>,
    },
    /// `DELETE` the matched target row.
    Delete,
}

/// A type-checked scalar expression.
///
/// Unlike [`ast::Expr`], every node carries a concrete result [`ColumnType`]
/// and every column reference is resolved to an ordinal.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    /// The expression shape.
    pub kind: TypedExprKind,
    /// The concrete type this expression evaluates to.
    pub ty: ColumnType,
}

/// The shape of a [`TypedExpr`].
#[derive(Debug, Clone, PartialEq)]
pub enum TypedExprKind {
    /// A literal constant. A [`ast::Value::Null`] takes its enclosing
    /// [`TypedExpr::ty`] from context.
    Literal(ast::Value),
    /// A column reference, resolved to an ordinal into the source table.
    Column(usize),
    /// A correlated reference to an enclosing query's row. `level` counts query nestings
    /// outward (`1` = the immediately enclosing query), `ordinal` indexes that row. Appears only
    /// inside a subquery plan; the executor binds each outer row before running the subquery, and
    /// [`crate::executor::eval`] reads it from the outer-row stack.
    OuterColumn {
        /// How many query levels out the referenced row is (`1` = immediate parent).
        level: usize,
        /// Column ordinal into that outer row.
        ordinal: usize,
    },
    /// A binary operation.
    Binary {
        /// Left operand.
        left: Box<TypedExpr>,
        /// Operator.
        op: ast::BinaryOp,
        /// Right operand.
        right: Box<TypedExpr>,
    },
    /// A unary operation.
    Unary {
        /// Operator.
        op: ast::UnaryOp,
        /// Operand.
        expr: Box<TypedExpr>,
    },
    /// `expr IS [NOT] NULL`.
    IsNull {
        /// Operand being tested.
        expr: Box<TypedExpr>,
        /// `true` for `IS NOT NULL`, `false` for `IS NULL`.
        negated: bool,
    },
    /// `left IS [NOT] DISTINCT FROM right` — NULL-aware comparison
    /// yielding a non-NULL boolean.
    IsDistinctFrom {
        /// Left operand.
        left: Box<TypedExpr>,
        /// Right operand.
        right: Box<TypedExpr>,
        /// `true` for `IS NOT DISTINCT FROM`.
        negated: bool,
    },
    /// `expr IS [NOT] {TRUE|FALSE|UNKNOWN}` — three-valued truth test
    /// yielding a non-NULL boolean.
    IsBool {
        /// Boolean operand being tested.
        expr: Box<TypedExpr>,
        /// Which truth value is tested for.
        truth: ast::TruthValue,
        /// `true` for the `IS NOT …` form.
        negated: bool,
    },
    /// `expr [NOT] IN (list...)`.
    InList {
        /// Value being tested.
        expr: Box<TypedExpr>,
        /// Typed list of values to test membership against.
        list: Vec<TypedExpr>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] BETWEEN low AND high`.
    Between {
        /// Value being tested.
        expr: Box<TypedExpr>,
        /// Lower bound (inclusive).
        low: Box<TypedExpr>,
        /// Upper bound (inclusive).
        high: Box<TypedExpr>,
        /// `true` for `NOT BETWEEN`.
        negated: bool,
    },
    /// `expr [NOT] LIKE pattern` — `%` = any chars, `_` = one char.
    Like {
        /// Subject string (must be `Text`).
        expr: Box<TypedExpr>,
        /// Pattern string (must be `Text`).
        pattern: Box<TypedExpr>,
        /// `true` for `NOT LIKE`.
        negated: bool,
        /// `ESCAPE 'c'` character: in the pattern, `c` makes the next `%`/`_`/`c` a literal. `None`
        /// (no `ESCAPE` clause) keeps the default — no escape, so `%`/`_` are always wildcards.
        escape: Option<char>,
        /// `true` for `ILIKE`: letters match case-insensitively, per character in the matcher
        /// (deep-gate #12).
        case_insensitive: bool,
    },
    /// `expr ~ pattern` / `~*` / `!~` / `!~*` — POSIX regular-expression match. Both
    /// operands are `Text`; the result is `Bool`.
    RegexMatch {
        /// Subject string (must be `Text`).
        expr: Box<TypedExpr>,
        /// POSIX regex pattern (must be `Text`).
        pattern: Box<TypedExpr>,
        /// `true` for the case-sensitive forms `~` / `!~`; `false` for `~*` / `!~*`.
        case_sensitive: bool,
        /// `true` for the negated forms `!~` / `!~*`.
        negated: bool,
    },
    /// `expr [NOT] SIMILAR TO pattern` — SQL-standard regex match. Both operands are
    /// `Text`; the result is `Bool`. The pattern uses SQL `SIMILAR TO` syntax (`%`/`_` wildcards
    /// plus the regex metacharacters `| * + ? ( ) { } [ ]`), translated to a POSIX regex matched
    /// against the **whole** subject (anchored); the default escape character is `\`.
    SimilarTo {
        /// Subject string (must be `Text`).
        expr: Box<TypedExpr>,
        /// `SIMILAR TO` pattern (must be `Text`).
        pattern: Box<TypedExpr>,
        /// `true` for `NOT SIMILAR TO`.
        negated: bool,
    },
    /// `CASE` expression — simple form (`operand = Some(...)`) or searched
    /// form (`operand = None`). All branch results plus the optional default
    /// share one resolved type stored in the enclosing [`TypedExpr::ty`].
    Case {
        /// Match value for the simple form; `None` for the searched form.
        operand: Option<Box<TypedExpr>>,
        /// `WHEN ... THEN ...` branches in declaration order.
        branches: Vec<TypedCaseBranch>,
        /// Optional `ELSE` result. When absent, no-match yields `NULL`.
        default: Option<Box<TypedExpr>>,
    },
    /// `COALESCE(a, b, ...)` — first non-NULL argument. All arguments share
    /// the resolved type in the enclosing [`TypedExpr::ty`].
    Coalesce(Vec<TypedExpr>),
    /// `ARRAY[a, b, ...]` — array constructor. Elements unify to one scalar type; the
    /// enclosing [`TypedExpr::ty`] is the `ColumnType::Array` of that element type.
    ArrayLiteral(Vec<TypedExpr>),
    /// `base[index]` — 1-based array element access. `base` is array-typed and `index` is
    /// `Int`; the result is the element type (NULL for an out-of-range or NULL index/array).
    Subscript {
        /// The array-valued operand.
        base: Box<TypedExpr>,
        /// The 1-based index (`Int`).
        index: Box<TypedExpr>,
    },
    /// `base[lower:upper]` — a 1-based inclusive array slice (B-fn). `base` is array-typed and the
    /// bounds are `Int`; the result is the same array type. An omitted bound defaults to the array's
    /// first / last element.
    ArraySlice {
        /// The array-valued operand.
        base: Box<TypedExpr>,
        /// The lower bound (1-based, inclusive), or `None` for the array start.
        lower: Option<Box<TypedExpr>>,
        /// The upper bound (1-based, inclusive), or `None` for the array end.
        upper: Option<Box<TypedExpr>>,
    },
    /// `CAST(expr AS T)` — runtime type conversion. The target type `T` is
    /// the enclosing [`TypedExpr::ty`]; the executor handles the conversion. The `bool` is
    /// `try_cast`: `true` for `TRY_CAST`/`SAFE_CAST`, where a failed conversion yields `NULL`.
    Cast(Box<TypedExpr>, bool),
    /// Reference to the *Nth* aggregate slot computed by an upstream
    /// [`PhysicalOperator::ScalarAggregate`]. The analyzer extracts each
    /// aggregate call out of the projection list and replaces it here with
    /// the slot index.
    AggregateRef(usize),
    /// `encrypt(value, key)` / `decrypt(value, key)`. Both operands
    /// are `Text`; the enclosing [`TypedExpr::ty`] is `Text`.
    Crypto {
        /// Encrypt vs. decrypt.
        op: CryptoOp,
        /// The value to (en/de)crypt.
        value: Box<TypedExpr>,
        /// The key string.
        key: Box<TypedExpr>,
    },
    /// A scalar built-in function call, e.g. `UPPER(name)` or `SUBSTRING(s, 2, 3)`. The
    /// analyzer validated arity + argument types; the result type is the enclosing
    /// [`TypedExpr::ty`]. Evaluated per row with SQL `NULL` propagation.
    ScalarFunction {
        /// Which built-in.
        func: ast::ScalarFunc,
        /// Argument expressions, in positional order.
        args: Vec<TypedExpr>,
    },
    /// A call to a registered scalar user-defined function, resolved by name. The executor
    /// evaluates each argument and invokes the registered Rust function per row; `ty` is the UDF's
    /// declared return type.
    ScalarUdf {
        /// The UDF name (folded), looked up in the registry at evaluation time.
        name: String,
        /// Argument expressions, in positional order (already checked against the declared types).
        args: Vec<TypedExpr>,
        /// The UDF's declared parameter types, captured at analysis time so the executor coerces each
        /// argument to the contract type without a per-row registry read (deep-gate).
        arg_types: Vec<ColumnType>,
    },
    /// A set-returning function at the top of a `SELECT`-list item; `ty` is the per-row
    /// element type it produces. It yields multiple values per input row, so the scalar evaluator
    /// never evaluates it — the [`PhysicalOperator::ProjectSet`] operator expands it instead.
    SetReturning {
        /// Which set-returning built-in.
        func: ast::SetReturningFunc,
        /// Argument expressions (e.g. the array for `UNNEST`; the document + path for
        /// `JSONB_PATH_QUERY`).
        args: Vec<TypedExpr>,
    },
    /// Uncorrelated scalar subquery `(SELECT ...)`. The plan projects
    /// exactly one column; the executor runs it once and substitutes the single
    /// result value (NULL for zero rows, error for more than one row) before any
    /// row evaluation, so the per-row evaluator never sees this node.
    ScalarSubquery(Box<SelectPlan>),
    /// `[NOT] EXISTS (SELECT ...)`. The executor runs the plan once and
    /// substitutes a boolean (row presence XOR `negated`) before row evaluation.
    Exists {
        /// The subquery whose row presence is tested.
        plan: Box<SelectPlan>,
        /// `true` for `NOT EXISTS`.
        negated: bool,
    },
    /// `expr <op> ANY/ALL ((SELECT ...))` — a quantified comparison against a single-column
    /// subquery. The executor runs the subquery and rewrites this into an OR (`ANY`) / AND (`ALL`)
    /// chain of comparisons, so empty-set and `NULL` three-valued semantics fall out of the existing
    /// binary evaluator.
    QuantifiedSubquery {
        /// The probed expression (left operand).
        expr: Box<TypedExpr>,
        /// The comparison operator applied against each subquery row.
        op: ast::BinaryOp,
        /// `true` for `ALL`, `false` for `ANY`/`SOME`.
        all: bool,
        /// The single-column subquery.
        plan: Box<SelectPlan>,
    },
    /// `expr <op> ANY/ALL (array)` — a quantified comparison against every element of a **runtime**
    /// array value. The executor evaluates the array and folds `expr <op> element` over the elements
    /// with `OR` (`ANY`) / `AND` (`ALL`), so empty-array and `NULL` three-valued semantics fall out of
    /// the existing binary evaluator (empty → `ANY` = FALSE, `ALL` = TRUE; a `NULL` array → `NULL`).
    QuantifiedArray {
        /// The probed expression (left operand).
        expr: Box<TypedExpr>,
        /// The comparison operator applied against each element.
        op: ast::BinaryOp,
        /// `true` for `ALL`, `false` for `ANY`/`SOME`.
        all: bool,
        /// The array-valued right operand.
        array: Box<TypedExpr>,
    },
    /// `expr [NOT] IN (SELECT ...)`. The plan projects exactly one
    /// column; the executor materializes its result rows into a literal
    /// [`TypedExprKind::InList`] before row evaluation (so `NULL` membership
    /// semantics fall out of the existing `InList` evaluator).
    InSubquery {
        /// The probed expression.
        expr: Box<TypedExpr>,
        /// The single-column subquery providing the membership set.
        plan: Box<SelectPlan>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
}

/// Direction of a [`TypedExprKind::Crypto`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoOp {
    /// `encrypt(value, key)`.
    Encrypt,
    /// `decrypt(value, key)`.
    Decrypt,
}

/// One aggregate function call collected from a `SELECT` projection. The
/// executor's [`PhysicalOperator::ScalarAggregate`] folds the input stream
/// across all calls in one pass.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateCall {
    /// Which aggregate function.
    pub func: ast::AggregateFunc,
    /// Argument expression; `None` for `COUNT(*)`.
    pub arg: Option<TypedExpr>,
    /// Result type the executor must emit for this slot.
    pub result_ty: ColumnType,
    /// `true` for `COUNT(DISTINCT x)` / `SUM(DISTINCT x)` etc. — fold only over the distinct
    /// (non-`NULL`) argument values within each group.
    pub distinct: bool,
    /// The fraction `f ∈ [0, 1]` for the ordered-set percentile aggregates `PERCENTILE_CONT` /
    /// `PERCENTILE_DISC`; `None` for every other aggregate (including `MODE`). `arg` carries
    /// the `WITHIN GROUP (ORDER BY ...)` value expression.
    pub fraction: Option<f64>,
    /// `true` when an ordered-set aggregate's `WITHIN GROUP (ORDER BY ...)` key is `DESC` — the
    /// executor sorts the collected values descending before applying the percentile/mode. `false`
    /// for `ASC` and for every non-ordered-set aggregate. (`NULLS FIRST/LAST` needs no flag: the
    /// ordered set excludes `NULL` ordering values, so their placement never affects the result.)
    pub ordered_set_descending: bool,
    /// `FILTER (WHERE pred)` — a per-row boolean predicate (resolved against the pre-aggregation
    /// scope); a row contributes to this call only when it evaluates to `TRUE`. `None` when
    /// the clause is absent.
    pub filter: Option<TypedExpr>,
    /// The constant separator of `STRING_AGG(expr, separator)`; `None` for every other aggregate.
    pub separator: Option<String>,
    /// The second per-row argument expression of the two-argument statistical aggregates
    /// (`CORR`/`COVAR_POP`/`COVAR_SAMP`), evaluated alongside [`arg`](Self::arg) per row;
    /// `None` for every other aggregate. The pair contributes only when both are non-`NULL`.
    pub arg2: Option<TypedExpr>,
    /// `ORDER BY` inside the aggregate (`array_agg(x ORDER BY y)`) — the sort keys the executor
    /// applies to the collected values before producing the result. Only `ARRAY_AGG` / `STRING_AGG`
    /// populate this; empty for every other aggregate (and when the clause is absent).
    pub order_by: Vec<OrderByKey>,
    /// For the synthetic `GROUPING(...)` call ([`ast::AggregateFunc::Grouping`]): the indices
    /// into the query's `group_keys` that the `GROUPING` arguments name, leftmost = most-significant
    /// bit. The executor emits, per super-aggregate row, a bitmask whose bit is `1` when that key was
    /// grouped away in the current set. Empty for every ordinary aggregate (which folds row values).
    pub grouping_args: Vec<usize>,
}

/// One window-function call extracted from a `SELECT` projection.
///
/// The executor's [`PhysicalOperator::Window`] computes one output column per
/// `WindowExpr` and appends it after the input row's columns; the projection
/// references that column by its appended ordinal (a plain
/// [`TypedExprKind::Column`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExpr {
    /// Which window function (ranking, aggregate, navigation, or distribution).
    pub func: ast::WindowFunc,
    /// Argument expressions: empty for the arg-less ranking/distribution functions
    /// and `COUNT(*)`; one for an aggregate / `FIRST_VALUE`; `[expr, offset, default]`
    /// for `LAG`/`LEAD`; `[expr, n]` for `NTH_VALUE`; `[n]` for `NTILE`.
    pub args: Vec<TypedExpr>,
    /// `PARTITION BY` key expressions, evaluated per input row; empty = one
    /// partition over the whole input.
    pub partition: Vec<TypedExpr>,
    /// `ORDER BY` keys within each partition; empty = unordered.
    pub order: Vec<OrderByKey>,
    /// Explicit `ROWS` window frame; `None` = the default frame (whole partition unordered,
    /// or running through the current peer group when ordered).
    pub frame: Option<WindowFrame>,
    /// Result type of the produced column.
    pub result_ty: ColumnType,
}

/// A resolved window frame.
///
/// `ROWS` frames use physical row offsets; `RANGE`/`GROUPS` frames are **peer-based** (`CURRENT ROW`
/// spans the current peer group). v1 supports numeric `n PRECEDING`/`FOLLOWING` offsets for `ROWS`
/// only — `RANGE`/`GROUPS` accept only the `UNBOUNDED`/`CURRENT ROW` bounds (the analyzer rejects
/// offsets there).
// Not `Eq`: a `RANGE` value offset (`FrameBound::Range*`) carries an `ast::Value`, which is only
// `PartialEq` (it can hold a float).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    /// Lower frame bound.
    pub start: FrameBound,
    /// Upper frame bound.
    pub end: FrameBound,
    /// `true` for `RANGE`/`GROUPS` (peer-aware `CURRENT ROW`); `false` for `ROWS` (physical).
    pub peer_based: bool,
    /// `true` when a `RANGE` value-offset frame's single ORDER BY column is `DESC`: the executor
    /// then reverses the value-boundary direction (preceding rows have larger keys) and the scan
    /// comparison. `false` for `ASC` / non-`RANGE`-value frames.
    pub range_descending: bool,
}

/// One bound of a [`WindowFrame`]. A `ROWS`/`GROUPS` offset is a non-negative count (rows or peer
/// groups); a `RANGE` offset is a value added to / subtracted from the single ordering column.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameBound {
    /// `UNBOUNDED PRECEDING` — the partition start.
    UnboundedPreceding,
    /// `<n> PRECEDING` — `n` rows (`ROWS`) or peer groups (`GROUPS`) before the current row.
    Preceding(u64),
    /// `CURRENT ROW`.
    CurrentRow,
    /// `<n> FOLLOWING` — `n` rows (`ROWS`) or peer groups (`GROUPS`) after the current row.
    Following(u64),
    /// `UNBOUNDED FOLLOWING` — the partition end.
    UnboundedFollowing,
    /// `RANGE <v> PRECEDING` — the bound value is the current ordering value minus `v` (plus `v` when
    /// the ordering is `DESC`); the frame includes rows within that value distance.
    RangePreceding(crate::ast::Value),
    /// `RANGE <v> FOLLOWING` — the bound value is the current ordering value plus `v` (minus `v` when
    /// the ordering is `DESC`).
    RangeFollowing(crate::ast::Value),
}

/// One `WHEN ... THEN ...` clause inside a [`TypedExprKind::Case`].
#[derive(Debug, Clone, PartialEq)]
pub struct TypedCaseBranch {
    /// Match value (simple form) or boolean predicate (searched form).
    pub when: TypedExpr,
    /// Result expression returned when this branch matches.
    pub then: TypedExpr,
}

// `Coalesce(Vec<TypedExpr>)` and `Cast { expr, target }` are appended to
// `TypedExprKind` above; see those variants for documentation.

/// A physical execution plan — the planner's output, run by the executor.
///
/// DDL, `INSERT`, `UPDATE`, and `DELETE` have exactly one execution strategy
/// each, so they carry the analyzer's plan structs unchanged. Only `SELECT` is
/// lowered into a [`PhysicalOperator`] pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalPlan {
    /// A sequence of plans executed in order within one statement transaction (the multi-object
    /// DDL desugar); the result is the last child's.
    Batch(Vec<Self>),
    /// `CREATE TABLE`.
    CreateTable(CreateTablePlan),
    /// `CREATE TABLE ... AS <select>` — the source query lowered to an executable operator.
    CreateTableAs(PhysicalCreateTableAs),
    /// `DROP TABLE`.
    DropTable(DropTablePlan),
    /// `CREATE MATERIALIZED VIEW` — the body lowered to an executable operator.
    CreateMaterializedView(PhysicalMaterializedView),
    /// `CREATE VIEW` — non-materialized; stores the defining SQL, inlined on read.
    CreateView(CreatePlainViewPlan),
    /// `DROP [MATERIALIZED] VIEW` — drops the materialized view's backing table.
    DropView(DropViewPlan),
    /// `CREATE TYPE name AS ENUM (...)` — persist a user-defined enum type (B-ENUM).
    CreateEnum(ast::CreateEnum),
    /// `DROP TYPE [IF EXISTS] name` — drop a user-defined type (B-ENUM).
    DropType(ast::DropType),
    /// `CREATE [OR REPLACE] TRIGGER ...` — a triggered SQL action to persist.
    CreateTrigger(CreateTriggerPlan),
    /// `DROP TRIGGER [IF EXISTS] name ON table`.
    DropTrigger(DropTriggerPlan),
    /// `ALTER TRIGGER name ON table RENAME TO new_name`.
    AlterTrigger(AlterTriggerPlan),
    /// `CREATE [OR REPLACE] PROCEDURE ...` — a stored procedure to persist.
    CreateProcedure(CreateProcedurePlan),
    /// `DROP PROCEDURE [IF EXISTS] name`.
    DropProcedure(DropProcedurePlan),
    /// `CALL name(args)` — invoke a stored procedure with constant arguments.
    Call(CallPlan),
    /// `CREATE [OR REPLACE] FUNCTION ...` — a SQL scalar function to persist.
    CreateFunction(CreateFunctionPlan),
    /// `DROP FUNCTION [IF EXISTS] name`.
    DropFunction(DropFunctionPlan),
    /// `REFRESH MATERIALIZED VIEW name` — recompute the named view's stored rows.
    RefreshMaterializedView(String),
    /// `CREATE POLICY` — a validated row-level-security policy to persist.
    CreatePolicy(CreatePolicyPlan),
    /// `DROP POLICY [IF EXISTS] name ON table`.
    DropPolicy(DropPolicyPlan),
    /// `ALTER TABLE`.
    AlterTable(AlterTablePlan),
    /// `INSERT`.
    Insert(InsertPlan),
    /// `SELECT` — the root of an operator pipeline. The second field is the estimated scanned-row
    /// count for a single-table plan (from ANALYZE stats, computed at plan time), or `None` for a
    /// multi-table / un-analyzed plan; the executor uses it for vectorized selective routing
    /// without re-fetching stats at run time.
    Select(PhysicalOperator, Option<u64>),
    /// `UPDATE`.
    Update(UpdatePlan),
    /// `DELETE`.
    Delete(DeletePlan),
    /// `MERGE INTO ... USING ... ON ... WHEN [NOT] MATCHED ...`.
    Merge(MergePlan),
    /// `EXPLAIN [ANALYZE] [VERBOSE] <statement>` — the executor formats the wrapped plan (and, for
    /// `ANALYZE`, also executes and times it) instead of just running it.
    Explain(Box<Self>, crate::ast::ExplainOptions),
    /// `BEGIN` — open an explicit transaction with the requested characteristics.
    BeginTransaction(TxnCharacteristics),
    /// `COMMIT` — commit the explicit transaction.
    Commit,
    /// `ROLLBACK` — abort the explicit transaction.
    Rollback,
    /// `SET TRANSACTION` — update the session's default transaction characteristics.
    SetTransaction(TxnCharacteristics),
    /// `SAVEPOINT name` — mark a rollback point in the current transaction.
    Savepoint(String),
    /// `ROLLBACK TO [SAVEPOINT] name` — undo back to a named savepoint.
    RollbackToSavepoint(String),
    /// `RELEASE [SAVEPOINT] name` — discard a named savepoint, keeping its writes.
    ReleaseSavepoint(String),
    /// `SET name = value` / `RESET name` — a session variable.
    SetVariable {
        /// Variable name, folded to lowercase.
        name: String,
        /// New value for `SET`, or `None` to `RESET` it to the unset default.
        value: Option<String>,
    },
    /// `SHOW name` — report a session variable's current value.
    ShowVariable(String),
    /// `SHOW TABLES` — list the database's tables.
    ShowTables,
    /// `SHOW COLUMNS FROM t` — list the resolved table's columns.
    ShowColumns(TableSchema),
    /// `VACUUM [FULL] [ANALYZE]` — reclaim dead row versions across all tables (and, for `ANALYZE`,
    /// recompute every table's statistics).
    Vacuum(crate::ast::VacuumOptions),
    /// `REINDEX ...` — accepted as a no-op (NusaDB's B-tree indexes are always consistent).
    Reindex,
    /// `ANALYZE` — recompute statistics for a table's columns.
    Analyze(AnalyzePlan),
    /// `LOCK TABLE` — acquire a table-level lock on each resolved table.
    LockTable {
        /// Resolved tables to lock, in order.
        tables: Vec<TableSchema>,
        /// The lock mode.
        mode: crate::ast::LockMode,
    },
    /// `PREPARE` — store a parameterized statement in the session.
    Prepare {
        /// The prepared statement's name.
        name: String,
        /// The un-analyzed statement (its `$n` placeholders are bound at `EXECUTE`).
        statement: Box<crate::ast::Statement>,
        /// Number of `$1..$n` placeholders the statement references.
        param_count: usize,
    },
    /// `EXECUTE` — run a prepared statement with constant arguments.
    Execute {
        /// The prepared statement's name.
        name: String,
        /// Constant argument values, one per placeholder, in order.
        args: Vec<ast::Value>,
    },
    /// `DEALLOCATE` — discard a prepared statement, or all of them.
    Deallocate(crate::ast::DeallocateTarget),
    /// `COMMENT ON` — target resolved; a metadata no-op at execution.
    Comment(CommentPlan),
    /// `CREATE SCHEMA`.
    CreateSchema(CreateSchemaPlan),
    /// `DROP SCHEMA`.
    DropSchema(DropSchemaPlan),
    /// `CREATE DATABASE` — single-database compatibility no-op.
    CreateDatabase(CreateDatabasePlan),
    /// `ALTER DATABASE` — single-database compatibility no-op.
    AlterDatabase(AlterDatabasePlan),
    /// `DROP DATABASE` — drop every table in the single database (backup-then-drop, or forced).
    DropDatabase(DropDatabasePlan),
    /// `CREATE SEQUENCE`.
    CreateSequence(CreateSequencePlan),
    /// `DROP SEQUENCE`.
    DropSequence(DropSequencePlan),
    /// `CREATE INDEX`.
    CreateIndex(CreateIndexPlan),
    /// `DROP INDEX`.
    DropIndex(DropIndexPlan),
    /// `UNION` / `INTERSECT` / `EXCEPT` set operation.
    SetOperation(PhysicalSetOp),
}

/// A node in a `SELECT` execution pipeline.
///
/// Each operator pulls rows from its `input` — a pull-based iterator model the
/// executor implements. Today the only scan is [`PhysicalOperator::SeqScan`];
/// an index scan needs index metadata and table statistics that do not exist
/// yet, and join operators need `JOIN` support — both are future work.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalOperator {
    /// Sequential scan over every row of a table.
    SeqScan {
        /// The table to scan.
        table: TableSchema,
        /// Projection pushdown: the source column ordinals this scan
        /// materializes, in ascending order. **Empty = the identity** (decode
        /// every column, the row keeps its full table width). When non-empty it
        /// is a strict subset and the scan yields a *narrowed* row holding only
        /// those columns, in this order — every `Column` reference above the
        /// scan has been rewritten to index the narrowed row. Set only for the
        /// simple single-table shape by the planner's projection-pushdown pass
        /// (`planner::pushdown`).
        columns: Vec<usize>,
    },
    /// Index scan over a named index: yields the visible rows whose key falls in
    /// `[lo, hi]`, in ascending key order. The executor encodes the bound *values* into the index's
    /// order-preserving key bytes; an `Unbounded` end is open. The planner emits this in place of a
    /// `SeqScan` when a predicate maps onto an index.
    IndexScan {
        /// The scanned table (used to decode the fetched tuples into rows).
        table: TableSchema,
        /// The index name, resolved to an `IndexId` at execution.
        index: String,
        /// Lower key bound as key-prefix column values; `Unbounded` for an open start.
        lo: std::ops::Bound<Vec<ast::Value>>,
        /// Upper key bound; `Unbounded` for an open end.
        hi: std::ops::Bound<Vec<ast::Value>>,
        /// Whether this scan is a *unique point lookup* — an equality bound covering the whole
        /// key of a unique index, so it matches **at most one row**. Informational: the executor
        /// scans identically either way; the reactor-inline point-get gate
        /// ([`crate::plan_is_inline_point_get`]) requires it to bound the work it
        /// admits onto the reactor.
        unique_point: bool,
    },
    /// `SELECT ... FOR UPDATE` / `FOR SHARE`: take a row lock on every base-table row that
    /// satisfies `predicate`, then yield `input`'s rows unchanged. Wraps the whole single-table
    /// pipeline; the executor re-scans the base table under the statement's snapshot, locks each
    /// matched `Tid` via `StorageEngine::lock_row`, and only then runs `input` — so a concurrent
    /// writer of a locked row blocks (the lost-update escape hatch). Held until the transaction ends.
    LockRows {
        /// The pipeline whose output rows are returned unchanged.
        input: Box<Self>,
        /// The base table whose rows are locked.
        table: TableSchema,
        /// `WHERE` predicate over the base row (subquery-free); `None` locks every row.
        predicate: Option<TypedExpr>,
        /// `FOR UPDATE` → `Exclusive`, `FOR SHARE` → `Shared`.
        mode: nusadb_core::engine::RowLockMode,
        /// `SKIP LOCKED` (the job-queue pattern): a matched row whose lock another transaction
        /// holds is skipped — excluded from both the locks taken and the pipeline's output —
        /// instead of aborting with a serialization conflict.
        skip_locked: bool,
    },
    /// Produces exactly one row with no columns — the source for a `SELECT`
    /// without a `FROM` clause (e.g. `SELECT 1`).
    OneRow,
    /// Produces one row per entry in `rows`, each built by evaluating its expressions against an empty
    /// input — the source for a `(VALUES ...) AS x` derived table. Every row has the same width.
    Values {
        /// The inline rows, in source order; each is a tuple of column expressions.
        rows: Vec<Vec<TypedExpr>>,
    },
    /// Produces the rows of a `UNION`/`INTERSECT`/`EXCEPT` set operation — the source for a
    /// `(SELECT ... UNION ...) AS x` derived table. Executed by the same path as a top-level set
    /// operation (`run_set_operation`), including its own `ORDER BY`/`LIMIT` and spill handling.
    SetOperation(Box<PhysicalSetOp>),
    /// Drops input rows for which `predicate` does not evaluate to true.
    Filter {
        /// Upstream operator.
        input: Box<Self>,
        /// Boolean row predicate.
        predicate: TypedExpr,
    },
    /// Sorts input rows by the given keys, in priority order.
    Sort {
        /// Upstream operator.
        input: Box<Self>,
        /// Sort keys; they reference source-table columns.
        keys: Vec<OrderByKey>,
        /// `FETCH FIRST n ROWS WITH TIES` cap: when `Some`, after sorting the
        /// operator skips `offset` rows, keeps `count` rows, then extends the result with every
        /// following row that ties the last kept row on `keys`. `None` for an ordinary sort (the
        /// separate [`Limit`](Self::Limit) operator caps the stream instead).
        limit_ties: Option<TiesLimit>,
        /// Limit-aware top-N cap: when `Some(m)`, an enclosing
        /// `LIMIT`/`OFFSET` needs only the first `m = offset + limit` rows in sort order, and
        /// nothing between this sort and that `LIMIT` changes the row count — so the executor
        /// selects the `m` smallest instead of a full O(N log N) sort. The selection is
        /// **result-identical** to the full sort's first `m` rows (same key comparator, same tie
        /// order). Every input row is still consumed, so a `SERIALIZABLE` scan's read set is
        /// unchanged.
        ///
        /// On the **row path** this is a streaming O(N log m) pass retaining O(m) rows, bounded by
        /// `work_mem` (bytes) so a wide-row `LIMIT` cannot hold `m` large rows unchecked. On the
        /// **vectorized path** the operator already buffers its input, so the win is CPU only
        /// (a partial selection instead of a full sort) — the memory profile is that of the
        /// vectorized sort. `None` = full sort required: no bounding `LIMIT`, a `WITH TIES` cap
        /// (that is `limit_ties`), a `DISTINCT`/`DISTINCT ON` dedup or a set-returning projection
        /// between this sort and the `LIMIT`, or an `m` beyond the cap that keeps the bounded set
        /// small. Never set together with `limit_ties`.
        top_n: Option<u64>,
    },
    /// Computes the output columns of each input row.
    Project {
        /// Upstream operator.
        input: Box<Self>,
        /// Output columns, in order.
        columns: Vec<Projection>,
    },
    /// Projection containing a set-returning function: for each input row, the SRF column is
    /// expanded to one output row per produced element, with the scalar columns repeated. Used
    /// instead of [`Project`](Self::Project) when a projection holds a
    /// [`TypedExprKind::SetReturning`]. v1 supports a single SRF per projection.
    ProjectSet {
        /// Upstream operator.
        input: Box<Self>,
        /// Output columns, in order; exactly one carries a `SetReturning` expression.
        columns: Vec<Projection>,
        /// `WITH ORDINALITY`: when `true`, each emitted row appends a 1-based `BIGINT` row
        /// number counting the produced elements per input row.
        ordinality: bool,
    },
    /// Passes through at most `count` rows, then stops.
    Limit {
        /// Upstream operator.
        input: Box<Self>,
        /// Maximum number of rows to emit (`u64::MAX` ≈ unbounded — used for an `OFFSET` with no
        /// `LIMIT`).
        count: u64,
        /// Number of leading rows to skip before emitting (`OFFSET`); `0` for none.
        offset: u64,
    },
    /// Removes duplicate rows from its input (`SELECT DISTINCT`), keeping the
    /// first occurrence of each distinct row. Two rows are duplicates when every
    /// column is "not distinct" (SQL semantics: `NULL` is not distinct from
    /// `NULL`).
    Distinct {
        /// Upstream operator (typically the `Project`).
        input: Box<Self>,
    },
    /// `SELECT DISTINCT ON (keys)` — keeps the **first** input row per distinct `keys` tuple.
    /// Sits below `Project` (above any `Sort`), so `keys` are evaluated against the source
    /// row, and "first" follows the `ORDER BY` the planner placed beneath it. NULL is not distinct
    /// from NULL.
    DistinctOn {
        /// Upstream operator (the `Sort`, or the unsorted source).
        input: Box<Self>,
        /// The `DISTINCT ON` key expressions, evaluated against the input (source) row.
        keys: Vec<TypedExpr>,
    },
    /// Scalar aggregate (no `GROUP BY`): folds the entire input stream into
    /// one output row, one column per [`AggregateCall`].
    ScalarAggregate {
        /// Upstream operator producing rows to fold.
        input: Box<Self>,
        /// Aggregate functions to compute in a single pass.
        calls: Vec<AggregateCall>,
    },
    /// Grouped aggregate (`GROUP BY`): partitions the input stream by the
    /// evaluated `group_keys`, then emits one row per group laid out as
    /// `[group key values ++ aggregate results]`.
    GroupAggregate {
        /// Upstream operator producing rows to group.
        input: Box<Self>,
        /// Group-key expressions, evaluated per input row to form the key.
        group_keys: Vec<TypedExpr>,
        /// Aggregate functions to compute per group, in a single pass.
        calls: Vec<AggregateCall>,
    },
    /// Multi-grouping-set aggregate (`ROLLUP`/`CUBE`/`GROUPING SETS`):
    /// runs the grouped aggregation once per grouping set and unions the
    /// results. Each output row keeps the full `[group key values ++ aggregate
    /// results]` layout of [`GroupAggregate`](Self::GroupAggregate); columns not
    /// activated by a given set are emitted as `NULL`.
    GroupingSetsAggregate {
        /// Upstream operator producing rows to group.
        input: Box<Self>,
        /// The union of all grouping-set key expressions, evaluated per input
        /// row. A grouping set selects a subset of these by index.
        group_keys: Vec<TypedExpr>,
        /// One entry per grouping set: the indices into `group_keys` it groups
        /// by (empty = the grand total over the whole input).
        grouping_sets: Vec<Vec<usize>>,
        /// Aggregate functions to compute per group, in a single pass.
        calls: Vec<AggregateCall>,
    },
    /// Window-function evaluation: computes one column per
    /// [`WindowExpr`] (ranking or aggregate-over-window) and appends it after
    /// the input row's columns, preserving input row order. Each window is
    /// evaluated independently over its own `PARTITION BY` / `ORDER BY`.
    Window {
        /// Upstream operator producing the rows to annotate.
        input: Box<Self>,
        /// Window functions, one appended output column each.
        windows: Vec<WindowExpr>,
        /// Limit-aware top-N cap: when `Some(m)`, an enclosing
        /// `ORDER BY … LIMIT` needs only the first `m` rows in the windows' shared order, and the
        /// windows are **ranking-only** (`ROW_NUMBER`/`RANK`/`DENSE_RANK`) over a **single
        /// partition** — whose value at position `k` depends only on rows at positions `≤ k` — so
        /// the operator computes over just the `m` smallest rows (bounded memory) instead of
        /// materializing the whole input. Result-identical to the full computation for those `m`
        /// rows. The planner sets it only when the outer `ORDER BY` equals the window order (so the
        /// first `m` by window order are exactly the first `m` the `LIMIT` wants). `None` = the full
        /// materializing computation (any partition, navigation/aggregate/distribution function, an
        /// explicit frame, or an order the outer `LIMIT` does not match).
        top_n: Option<u64>,
    },
    /// Nested-loop join: for each `left` row, scan `right` and emit the
    /// concatenated row `[left ++ right]` where `predicate` holds. Outer joins
    /// also emit unmatched rows NULL-padded on the absent side: `Left`/`Full`
    /// keep unmatched left rows (right padded with `right_width` NULLs);
    /// `Right`/`Full` keep unmatched right rows (left padded with `left_width`
    /// NULLs).
    NestedLoopJoin {
        /// Left (outer) input.
        left: Box<Self>,
        /// Right (inner) input — fully scanned once per left row.
        right: Box<Self>,
        /// `ON` predicate over the concatenated row.
        predicate: TypedExpr,
        /// Join kind.
        kind: ast::JoinKind,
        /// Number of columns the left input produces (for NULL-padding the
        /// left side of unmatched right rows in `Right`/`Full` joins).
        left_width: usize,
        /// Number of columns the right input produces (for NULL-padding the
        /// right side of unmatched left rows in `Left`/`Full` joins).
        right_width: usize,
        /// `USING`/`NATURAL` merged-column `(kept-left, hidden-right)` ordinal pairs: each emitted
        /// row's left slot is set to `coalesce(left, right)` (taking the right value when the left
        /// is NULL). Correct for RIGHT/FULL unmatched rows, a no-op for INNER/LEFT. Empty for `ON`.
        coalesce_pairs: Vec<(usize, usize)>,
    },
    /// Hash join: builds a hash table on the `right` input keyed by the
    /// right-hand side of each equi-key, then probes it once per `left` row.
    /// Equivalent in result to a [`NestedLoopJoin`](Self::NestedLoopJoin) whose
    /// predicate is `(keys[0].left = keys[0].right) AND ... AND residual`, but
    /// `O(left + right)` instead of `O(left × right)`.
    ///
    /// A key with a `NULL` on either side never matches (SQL `NULL = NULL` is
    /// unknown), so such rows are treated as unmatched — preserving correct
    /// outer-join NULL-padding.
    HashJoin {
        /// Left (probe) input.
        left: Box<Self>,
        /// Right (build) input — fully materialized into the hash table.
        right: Box<Self>,
        /// Equi-join keys. Each `left`/`right` expression references the
        /// concatenated row's ordinals (left columns `< left_width`, right
        /// columns `>= left_width`); the executor evaluates them against the
        /// appropriate padded side.
        keys: Vec<HashKey>,
        /// Leftover (non-equi) `ON` conjuncts applied to each hash-matched pair,
        /// over the concatenated row. `None` when the predicate was purely
        /// equi-joins.
        residual: Option<TypedExpr>,
        /// Join kind.
        kind: ast::JoinKind,
        /// Columns the left input produces (NULL-padding for `Right`/`Full`).
        left_width: usize,
        /// Columns the right input produces (NULL-padding for `Left`/`Full`).
        right_width: usize,
        /// `USING`/`NATURAL` merged-column `(kept-left, hidden-right)` ordinal pairs: each emitted
        /// row's left slot is set to `coalesce(left, right)` (taking the right value when the left
        /// is NULL). Correct for RIGHT/FULL unmatched rows, a no-op for INNER/LEFT. Empty for `ON`.
        coalesce_pairs: Vec<(usize, usize)>,
    },
    /// `LATERAL` join: a dependent join where the `right` input is re-executed once per
    /// `left` row, with that row bound as the enclosing scope so the subquery's correlated
    /// references (`OuterColumn`) read it. Unlike [`NestedLoopJoin`](Self::NestedLoopJoin), whose
    /// `right` is scanned once and reused, `LATERAL` parameterizes `right` by the current left row.
    /// Only `Inner`/`Left`/`Cross` are valid (a right/full lateral join is meaningless — the right
    /// side depends on the left).
    LateralJoin {
        /// Left (outer) input — materialized; each row parameterizes `right`.
        left: Box<Self>,
        /// Right (inner) input — re-executed per left row with that row bound as the outer scope.
        right: Box<Self>,
        /// `ON` predicate over the concatenated `[left ++ right]` row (`true` for `CROSS`).
        predicate: TypedExpr,
        /// Join kind — `Inner`, `Left` (NULL-pad unmatched left rows), or `Cross`.
        kind: ast::JoinKind,
        /// Number of columns the right input produces (for NULL-padding a `Left` join).
        right_width: usize,
    },
    /// An `information_schema` view scan: yields metadata rows from the engine for the
    /// requested view (`tables`/`columns`/`schemata`/`views`). Behaves like a `SeqScan` over a
    /// synthetic table whose rows come from engine introspection rather than storage, so the entire
    /// SQL pipeline (WHERE/Project/Sort/Limit) applies naturally on top.
    InfoSchemaScan {
        /// Which `information_schema` view to produce metadata for.
        view: InfoSchemaView,
    },
    /// Vector k-nearest-neighbours: the top-`k` rows of `table` ordered by `column <=> query`
    /// (cosine distance), ascending. The planner emits this for `ORDER BY col <=> q LIMIT k`
    /// over a plain single-table scan. At execution it uses an HNSW index when one is declared on the
    /// column (approximate, fast) and otherwise falls back to an exact scan+sort — the result is the
    /// `k` nearest rows in distance order either way.
    VectorKnn {
        /// The scanned table (its rows are the candidates; also the vector-index catalog lookup key).
        table: TableSchema,
        /// Ordinal of the indexed `VECTOR(n)` column in `table`.
        column_ordinal: usize,
        /// The query vector expression (a constant — references no row column).
        query: TypedExpr,
        /// Number of nearest rows to return.
        k: u64,
        /// Optional `WHERE` predicate over the source row: only rows passing it are
        /// returned. With an HNSW index the search over-fetches and post-filters; if too few pass it
        /// falls back to an exact filtered scan, so a selective filter never under-returns.
        filter: Option<TypedExpr>,
    },
    /// `WITH RECURSIVE`: materialize each CTE to a fixpoint, bind it to its synthetic table,
    /// then run `body`. The bindings are dropped after `body` produces its rows.
    WithRecursive {
        /// The recursive CTEs to materialize, in declaration order.
        ctes: Vec<PhysicalRecursiveCte>,
        /// The query body, which scans the bound CTE tables.
        body: Box<Self>,
    },
    /// Data-modifying CTEs: run each statement once, bind its `RETURNING` rows to the
    /// CTE's synthetic table, then run `body` (which scans those tables). Bindings drop after the body.
    WithModifying {
        /// The data-modifying CTEs to run, in declaration order.
        ctes: Vec<PhysicalModifyingCte>,
        /// The query body, which reads the bound CTE relations.
        body: Box<Self>,
    },
}

/// A lowered [`ModifyingCteDef`]: the data-modifying statement as a runnable plan, plus the synthetic
/// table id its `RETURNING` rows bind to.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalModifyingCte {
    /// Synthetic table id the `RETURNING` rows bind to.
    pub id: nusadb_core::TableId,
    /// The planned data-modifying statement (`PhysicalPlan::Insert`/`Update`/`Delete`), run once.
    pub plan: PhysicalPlan,
}

/// A lowered [`RecursiveCteDef`]: the base + recursive terms as physical operators.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalRecursiveCte {
    /// Synthetic table id the CTE is bound to.
    pub id: nusadb_core::TableId,
    /// The non-recursive base term.
    pub base: Box<PhysicalOperator>,
    /// The recursive term, which scans the CTE's working set.
    pub recursive: Box<PhysicalOperator>,
    /// `UNION ALL` vs `UNION` (distinct).
    pub union_all: bool,
}

/// One equi-join key of a [`PhysicalOperator::HashJoin`]: a pair of expressions
/// that must compare equal, where `left` references only left-input columns and
/// `right` references only right-input columns.
#[derive(Debug, Clone, PartialEq)]
pub struct HashKey {
    /// Left-side key expression (references left-input columns only).
    pub left: TypedExpr,
    /// Right-side key expression (references right-input columns only).
    pub right: TypedExpr,
}
