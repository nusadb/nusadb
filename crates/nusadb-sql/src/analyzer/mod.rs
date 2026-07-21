//! Semantic analysis: resolve names against the catalog, type-check
//! expressions, and lower the parser's [`ast`] into a typed
//! [`LogicalPlan`].
//!
//! The analyzer is the gate that rejects unknown tables/columns, arity errors,
//! and type mismatches *before* the planner runs. Its only outside dependency
//! is the [`Catalog`] port, so it can be tested against a trivial in-memory
//! catalog with no real storage engine.
//!
//! # NULL typing
//!
//! A bare `NULL` literal has no intrinsic type. The analyzer types it from
//! context: the target column of an `INSERT`/`UPDATE`, the other operand of a
//! binary operator, or the boolean a `WHERE` clause expects. Where no context
//! supplies a type (`SELECT NULL`, `NULL = NULL`) it is rejected with
//! [`Error::AmbiguousNull`].

// The analyzer is one tightly-connected web of free functions split across per-concern submodules
// (ADR 007); each resolves its siblings + shared helpers through a glob re-export.
#![allow(clippy::wildcard_imports)]

use std::collections::HashSet;

use nusadb_core::{ColumnDef, ColumnType, TableSchema};

use crate::ast;
use crate::error::Error;
use crate::planner::{
    AggregateCall, AlterColumnOp, AlterDatabasePlan, AlterTablePlan, AlterTriggerPlan, AnalyzePlan,
    Assignment, CallPlan, CheckSpec, CommentPlan, ConflictArbiter, CreateDatabasePlan,
    CreateFunctionPlan, CreateIndexPlan, CreatePlainViewPlan, CreatePolicyPlan,
    CreateProcedurePlan, CreateSchemaPlan, CreateSequencePlan, CreateTableAsPlan, CreateTablePlan,
    CreateTriggerPlan, CryptoOp, DeletePlan, DropDatabasePlan, DropFunctionPlan, DropIndexPlan,
    DropPolicyPlan, DropProcedurePlan, DropSchemaPlan, DropSequencePlan, DropTablePlan,
    DropTriggerPlan, DropViewPlan, ForeignKeySpec, FrameBound, IndexMeta, InsertPlan, InsertSource,
    JoinPlan, LogicalPlan, MaterializedViewPlan, MergeMatchedAction, MergePlan, MergeWhen,
    ModifyingCteDef, OnConflictPlan, OrderByKey, Projection, RecursiveCteDef, SelectPlan,
    SetOpPlan, SetOpTree, TxnCharacteristics, TypedCaseBranch, TypedExpr, TypedExprKind,
    UniqueConstraintSpec, UpdatePlan, VectorIndexSpec, WindowExpr, WindowFrame,
};

mod ddl;
mod dml;
mod expr;
mod select;
mod typecheck;
mod window;
use ddl::*;
use dml::*;
use expr::*;
use select::*;
use typecheck::*;
use window::*;

/// Schema-resolution port used by the analyzer.
///
/// The analyzer only needs to resolve table names to schemas; it deliberately
/// does not depend on the full `StorageEngine` surface. The production engine
/// is adapted to this trait at wire-up time; tests use a small in-memory
/// implementation.
pub trait Catalog {
    /// Resolve a table name to its schema, or `Ok(None)` if it does not exist. Resolves in the
    /// default `public` namespace; use [`lookup_table_in`](Self::lookup_table_in) for a qualifier.
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error>;

    /// Resolve `(schema, name)` to its schema, or `Ok(None)` if it does not exist. The default
    /// delegates to [`lookup_table`](Self::lookup_table) for the `public` schema and reports
    /// `Ok(None)` for any other — correct for single-namespace test doubles; the production adapter
    /// overrides it to resolve the real `(schema, name)` catalog key under the transaction snapshot.
    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        if schema == nusadb_core::PUBLIC_SCHEMA {
            self.lookup_table(name)
        } else {
            Ok(None)
        }
    }

    /// The ordered schemas an unqualified name resolves through (NS3, the session `search_path`),
    /// first to last. The default is `[public]`; the production adapter reports the connection's
    /// `SET search_path = …` list (always ending in `public`).
    fn search_path(&self) -> Vec<String> {
        vec![nusadb_core::PUBLIC_SCHEMA.to_owned()]
    }

    /// The session's current schema — where an unqualified name is *created* (the first entry of the
    /// [`search_path`](Self::search_path)). Defaults to deriving from the search path.
    fn current_schema(&self) -> String {
        self.search_path()
            .into_iter()
            .next()
            .unwrap_or_else(|| nusadb_core::PUBLIC_SCHEMA.to_owned())
    }

    /// The secondary indexes declared on `table`, for index-scan planning. Default empty so
    /// a minimal catalog — or one whose engine has no indexes — simply plans sequential scans; the
    /// production adapter overrides it to expose the engine's indexes. Called only for a real base
    /// table (never a CTE), so a name that does not resolve may return an empty list.
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let _ = table;
        Ok(Vec::new())
    }

    /// The table's ANALYZE statistics for cost-based planning, or `None` if it has not been
    /// analyzed. Default `None` so a minimal catalog plans heuristically; the production adapter
    /// returns the engine's stored stats. Used only for a real base table (never a CTE).
    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        let _ = table;
        Ok(None)
    }

    /// The base table's `O(1)` approximate live-row count (see
    /// [`StorageEngine::approx_row_count`](nusadb_core::StorageEngine::approx_row_count)) — the
    /// vectorized-routing cardinality fallback when `ANALYZE` stats are absent. Default `Ok(0)` so a
    /// minimal catalog gives no hint; the production adapter returns the engine's estimate. A routing
    /// hint only, never a correctness input.
    fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
        let _ = table;
        Ok(0)
    }

    /// The defining SQL of a non-materialized view named `name`, or `None` if no such view exists.
    /// Default `None` so a minimal catalog has no views; the production adapter reads the view
    /// catalog. The analyzer inlines the parsed body in place of a `FROM` base, like a CTE.
    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        let _ = name;
        Ok(None)
    }

    /// The explicit output column names of the non-materialized view `name`, as declared in
    /// `CREATE VIEW name (cols) AS ...`, or an empty list if the view declared none (then the body's
    /// inferred projection names are used). Default empty; the production adapter reads the view
    /// column catalog. Applied positionally when the view body is inlined on read.
    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, Error> {
        let _ = name;
        Ok(Vec::new())
    }

    /// The definition of a SQL scalar function named `name`, or `None` if no such function exists.
    /// Default `None` so a minimal catalog has no functions; the production adapter reads
    /// the function catalog. The analyzer inlines the function body in place of a call to it.
    fn lookup_function(&self, name: &str) -> Result<Option<FunctionDef>, Error> {
        let _ = name;
        Ok(None)
    }

    /// Whether the session running this statement is a superuser, which bypasses row-level security.
    /// Default `true` so a minimal catalog (and the analyzer's own unit tests) behaves as a
    /// superuser — RLS never restricts it, keeping existing behavior unchanged. The production
    /// adapter reports this from the session user.
    fn is_superuser(&self) -> bool {
        true
    }

    /// Whether row-level security is enabled on base table `name`. Default `false` so a
    /// minimal catalog never restricts; the production adapter reads the RLS catalog. Consulted only
    /// for a non-superuser (the caller short-circuits on [`Catalog::is_superuser`] first), so a
    /// superuser session never pays for this lookup.
    fn rls_enabled(&self, name: &str) -> Result<bool, Error> {
        let _ = name;
        Ok(false)
    }

    /// The session user, used to match a policy's `TO role` list. Default is the bootstrap
    /// superuser name; consulted only when building a non-superuser's policy predicate.
    fn current_user(&self) -> String {
        "nusa-root".to_owned()
    }

    /// The row-level-security policies defined on base table `name`. Default empty so a
    /// minimal catalog has none; the production adapter reads the policy catalog. Consulted only for
    /// a non-superuser whose query targets a single RLS-enabled base table.
    fn lookup_policies(&self, name: &str) -> Result<Vec<PolicyDef>, Error> {
        let _ = name;
        Ok(Vec::new())
    }
}

/// Re-parse and type-check a stored CHECK predicate against `table`'s columns, for the executor to
/// evaluate per row. The predicate text comes from the constraint catalog; it was validated
/// to be a subquery-free boolean expression when the constraint was created, so `catalog` is only
/// consulted defensively (a single-table CHECK references no other tables).
pub(crate) fn analyze_check_predicate(
    predicate_sql: &str,
    table: &TableSchema,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    let expr = crate::parser::parse_expression(predicate_sql)?;
    analyze_expr(
        &expr,
        &single_table_scope(table),
        catalog,
        Some(ColumnType::Bool),
    )
}

/// Re-parse and type-check a stored functional/expression index key against `table`'s columns, for
/// the executor to evaluate per row to build the index key. The text comes from the index catalog;
/// it was validated subquery- and aggregate-free at `CREATE INDEX` time, so `catalog` is consulted
/// only defensively (a single-table index key references no other table).
pub(crate) fn analyze_index_key_expr(
    expr_sql: &str,
    table: &TableSchema,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    let expr = crate::parser::parse_expression(expr_sql)?;
    analyze_expr(&expr, &single_table_scope(table), catalog, None)
}

/// Re-parse and type-check a stored column `DEFAULT` expression to the target type. The text
/// comes from the column-default catalog; it was validated subquery-free and assignable at `CREATE`
/// time, so it is analyzed against an empty scope (a default references no column) and `catalog` is
/// consulted only defensively.
pub(crate) fn analyze_default_expr(
    default_sql: &str,
    target: ColumnType,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    let expr = crate::parser::parse_expression(default_sql)?;
    analyze_expr(&expr, &[], catalog, Some(target))
}

/// Re-parse and type-check a stored `GENERATED ALWAYS AS (<expr>) STORED` column expression against
/// `table`'s columns, to the target column type. Unlike a `DEFAULT`, a generated expression
/// references the row's other columns, so it is analyzed against the table scope (like a CHECK). It
/// was validated subquery-free, immutable, and assignable at `CREATE` time, so `catalog` is consulted
/// only defensively (a generated expression references only its own table's columns).
pub(crate) fn analyze_generated_expr(
    expr_sql: &str,
    table: &TableSchema,
    target: ColumnType,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    let expr = crate::parser::parse_expression(expr_sql)?;
    analyze_expr(&expr, &single_table_scope(table), catalog, Some(target))
}

/// A row-level-security policy as the [`Catalog`] reports it.
///
/// The `using`/`check` predicates are canonical SQL text, re-parsed and analyzed against the target
/// table when the policy is enforced — mirroring how non-materialized views store their body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDef {
    /// Policy name (unique per table).
    pub name: String,
    /// `true` for a permissive policy (grants access, `OR`-combined with sibling permissive
    /// policies); `false` for a restrictive policy (`AND`-combined to narrow access).
    pub permissive: bool,
    /// The command the policy applies to.
    pub command: crate::ast::PolicyCommand,
    /// Roles the policy applies to; empty means `PUBLIC` (every role).
    pub roles: Vec<String>,
    /// `USING` row-visibility predicate as canonical SQL, or `None`.
    pub using: Option<String>,
    /// `WITH CHECK` write predicate as canonical SQL, or `None`.
    pub check: Option<String>,
}

/// A SQL scalar function as the [`Catalog`] reports it for inlining.
///
/// The body is the canonical SQL of a `SELECT <expr>` (a single scalar projection, no `FROM`). When a
/// call `name(args)` is analyzed, the body's projection expression is extracted, its `$1..$n`
/// placeholders are replaced with the call's argument expressions, and the result is analyzed in
/// place of the call — so a SQL function composes exactly like a built-in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDef {
    /// The number of declared parameters (the call must supply exactly this many arguments).
    pub param_count: usize,
    /// Declared parameter names, in order (lowercase-folded). A body may reference a parameter either
    /// positionally as `$1`..`$n` or by these names; both bind to the call arguments at
    /// inline time. Empty for functions persisted before names were tracked (positional-only).
    pub param_names: Vec<String>,
    /// The function body as canonical SQL (a `SELECT <expr>`). The result type is the inlined body's
    /// type; the declared `RETURNS` type is accepted but not enforced (a follow-up, like `PREPARE`).
    pub body: String,
}

/// A secondary index as the [`Catalog`] reports it for planning.
///
/// Carries the index name and the table columns it keys, by name (the analyzer resolves the names
/// to ordinals) — only what the planner needs to match a predicate onto the index, not the full
/// engine `IndexDef`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexInfo {
    /// Index name, used to resolve the `IndexId` at execution time.
    pub name: String,
    /// Key column names, in index order.
    pub columns: Vec<String>,
    /// Whether the index enforces key uniqueness — an equality bound covering the whole key
    /// then matches at most one row, which the reactor-inline point-get gate relies on.
    pub unique: bool,
}

/// Analyze a parsed [`ast::Statement`] into a typed [`LogicalPlan`].
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-statement dispatch; length tracks the statement set, not complexity"
)]
pub fn analyze(stmt: ast::Statement, catalog: &dyn Catalog) -> Result<LogicalPlan, Error> {
    match stmt {
        ast::Statement::CreateTable(ct) => {
            analyze_create_table(ct, catalog).map(LogicalPlan::CreateTable)
        },
        ast::Statement::CreateTableAs(ct) => analyze_create_table_as(ct, catalog),
        ast::Statement::DropTable(dt) => {
            analyze_drop_table(dt, catalog).map(LogicalPlan::DropTable)
        },
        // The multi-object DDL desugar (`DROP TABLE a, b`): analyze each child in order. Every
        // child resolves against the same pre-statement catalog snapshot — exactly how the
        // standard treats the list (a duplicate name simply drops-then-misses at execution,
        // surfaced by the second child's IF EXISTS handling).
        ast::Statement::Batch(stmts) => Ok(LogicalPlan::Batch(
            stmts
                .into_iter()
                .map(|s| analyze(s, catalog))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        // CREATE INDEX is recognized by the parser today but cannot be
        // executed end-to-end yet — the analyzer needs the catalog hooks
        // tracked as task + to validate the target table and
        // columns, and the engine needs to actually build the index.
        // Until those land, surface a clear `Unsupported` so the parser
        // surface stays honest about what the engine will run.
        // CREATE INDEX: the engine path exists; resolve the target table + columns
        // against the catalog into an IndexDef. The executor calls the engine.
        ast::Statement::CreateIndex(ci) => {
            analyze_create_index(ci, catalog).map(LogicalPlan::CreateIndex)
        },
        // DROP INDEX is recognized by the parser but cannot run end-to-end
        // until the index catalog/engine path exists. Resolution lands in
        // (analyze) on top of the catalog hook; reject clearly
        // until then so the surface stays honest.
        ast::Statement::DropIndex(di) => Ok(LogicalPlan::DropIndex(DropIndexPlan {
            name: di.name,
            if_exists: di.if_exists,
        })),
        ast::Statement::AlterTable(at) => {
            analyze_alter_table(at, catalog).map(LogicalPlan::AlterTable)
        },
        // CREATE MATERIALIZED VIEW: analyze the body and derive its output schema; the executor
        // computes and stores the rows into a backing table. Plain (non-materialized) views still
        // re-evaluate on read and are not yet implemented.
        ast::Statement::CreateView(cv) => analyze_create_materialized_view(cv, catalog),
        // DROP [MATERIALIZED] VIEW drops the backing table (sqlparser does not distinguish the two).
        ast::Statement::DropView(dv) => {
            // The backing store is an ordinary table, so a system-catalog name must not be
            // droppable through the view path either.
            enforce_system_catalog(&dv.name, catalog)?;
            Ok(LogicalPlan::DropView(DropViewPlan {
                name: dv.name,
                if_exists: dv.if_exists,
            }))
        },
        // CREATE TYPE ... AS ENUM (B-ENUM): validate the labels (at least one, distinct) and persist.
        ast::Statement::CreateEnum(ce) => {
            if ce.labels.is_empty() {
                return Err(Error::Unsupported(
                    "CREATE TYPE ... AS ENUM requires at least one label".to_owned(),
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for label in &ce.labels {
                if !seen.insert(label.as_str()) {
                    return Err(Error::Unsupported(format!(
                        "duplicate ENUM label {label:?}"
                    )));
                }
            }
            Ok(LogicalPlan::CreateEnum(ce))
        },
        ast::Statement::DropType(dt) => Ok(LogicalPlan::DropType(dt)),
        // CREATE/DROP TRIGGER: validate the target table; the action + WHEN bodies
        // are kept as text and re-analyzed (with NEW/OLD bound) when the trigger fires.
        ast::Statement::CreateTrigger(ct) => analyze_create_trigger(ct, catalog),
        ast::Statement::DropTrigger(dt) => Ok(LogicalPlan::DropTrigger(DropTriggerPlan {
            name: dt.name,
            table: dt.table,
            if_exists: dt.if_exists,
        })),
        // ALTER TRIGGER ... RENAME TO: existence of the trigger (and absence of the new name) is
        // checked by the executor against the trigger catalog, like DROP TRIGGER.
        ast::Statement::AlterTrigger(at) => Ok(LogicalPlan::AlterTrigger(AlterTriggerPlan {
            name: at.name,
            table: at.table,
            new_name: at.new_name,
        })),
        // CREATE/DROP PROCEDURE + CALL: the body is kept as text and re-parsed +
        // run (with `$n` bound to the call arguments) by the executor's procedure module.
        ast::Statement::CreateProcedure(cp) => {
            enforce_system_catalog(&cp.name, catalog)?;
            let param_count = cp.params.iter().filter(|p| !p.out).count();
            let out_params = cp
                .params
                .iter()
                .filter(|p| p.out)
                .map(|p| p.name.clone())
                .collect();
            Ok(LogicalPlan::CreateProcedure(CreateProcedurePlan {
                name: cp.name,
                or_replace: cp.or_replace,
                param_count,
                out_params,
                body: cp.body,
            }))
        },
        ast::Statement::DropProcedure(dp) => Ok(LogicalPlan::DropProcedure(DropProcedurePlan {
            name: dp.name,
            if_exists: dp.if_exists,
        })),
        ast::Statement::Call(call) => {
            // Arguments bind `$1..$n` before the body is analyzed, so they must be constants.
            let args = call
                .args
                .iter()
                .map(const_value)
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(LogicalPlan::Call(CallPlan {
                name: call.name,
                args,
            }))
        },
        // CREATE/DROP FUNCTION: a SQL scalar function, inlined at call sites by the analyzer.
        ast::Statement::CreateFunction(cf) => {
            enforce_system_catalog(&cf.name, catalog)?;
            // The declared RETURNS type is accepted at parse time but not stored/enforced (v1).
            Ok(LogicalPlan::CreateFunction(CreateFunctionPlan {
                name: cf.name,
                or_replace: cf.or_replace,
                param_count: cf.params.len(),
                param_names: cf.params.iter().map(|p| p.name.clone()).collect(),
                body: cf.body,
            }))
        },
        ast::Statement::DropFunction(df) => Ok(LogicalPlan::DropFunction(DropFunctionPlan {
            name: df.name,
            if_exists: df.if_exists,
        })),
        // REFRESH MATERIALIZED VIEW: recompute the named view at execution.
        ast::Statement::RefreshMaterializedView(name) => {
            // Belt-and-suspenders: a `nusadb_*` name can never be a matview (creation is guarded),
            // so refuse here too rather than rely on the executor's not-a-matview rejection.
            enforce_system_catalog(&name, catalog)?;
            Ok(LogicalPlan::RefreshMaterializedView(name))
        },
        // CREATE/DROP POLICY: validate the policy against its table, then persist it.
        ast::Statement::CreatePolicy(cp) => analyze_create_policy(cp, catalog),
        ast::Statement::DropPolicy(dp) => {
            require_rls_admin(catalog, "drop a row-level-security policy")?;
            Ok(LogicalPlan::DropPolicy(DropPolicyPlan {
                name: dp.name,
                table: dp.table,
                if_exists: dp.if_exists,
            }))
        },
        // ALTER POLICY: merge the changed clauses onto the existing policy, then re-persist.
        ast::Statement::AlterPolicy(ap) => analyze_alter_policy(ap, catalog),
        // CREATE/DROP SCHEMA: the catalog/engine path exists, so resolve the
        // statement shape; the executor calls the engine (and resolves the name → id for DROP).
        ast::Statement::CreateSchema(cs) => Ok(LogicalPlan::CreateSchema(CreateSchemaPlan {
            name: cs.name,
            if_not_exists: cs.if_not_exists,
        })),
        ast::Statement::DropSchema(ds) => Ok(LogicalPlan::DropSchema(DropSchemaPlan {
            name: ds.name,
            if_exists: ds.if_exists,
            cascade: ds.cascade,
        })),
        // CREATE/ALTER DATABASE: NusaDB is single-database per data dir, so these are accepted as a
        // compatibility no-op (no catalog work) — the executor just reports success.
        ast::Statement::CreateDatabase(cd) => Ok(LogicalPlan::CreateDatabase(CreateDatabasePlan {
            name: cd.name,
            if_not_exists: cd.if_not_exists,
        })),
        ast::Statement::AlterDatabase(ad) => Ok(LogicalPlan::AlterDatabase(AlterDatabasePlan {
            name: ad.name,
        })),
        // DROP DATABASE empties the single database's tables (backing them up first unless FIX); the
        // executor does the work against the engine.
        ast::Statement::DropDatabase(dd) => Ok(LogicalPlan::DropDatabase(DropDatabasePlan {
            name: dd.name,
            if_exists: dd.if_exists,
            force: dd.force,
        })),
        // CREATE/DROP SEQUENCE: the engine path exists; fold the options into a
        // SequenceDef. The executor calls the engine (resolving name → id for DROP).
        ast::Statement::CreateSequence(cs) => {
            analyze_create_sequence(cs).map(LogicalPlan::CreateSequence)
        },
        ast::Statement::DropSequence(ds) => Ok(LogicalPlan::DropSequence(DropSequencePlan {
            name: ds.name,
            if_exists: ds.if_exists,
        })),
        // TRUNCATE: desugar to an unfiltered DELETE — the same MVCC delete path empties the
        // table (honouring FK references + maintaining secondary indexes). `RESTART IDENTITY` is
        // carried through so the executor resets the backing sequence of each SERIAL/IDENTITY column
        // after the rows are removed (`CONTINUE IDENTITY` / unspecified leaves the sequence advancing).
        ast::Statement::Truncate(t) => {
            // Resolve the target through the search path (an explicit qualifier wins), exactly like a
            // DELETE target, so TRUNCATE reaches a non-public table too.
            let table = resolve_table(t.schema.as_deref(), &t.name, catalog)?;
            Ok(LogicalPlan::Delete(DeletePlan {
                table,
                using: None,
                using_plan: None,
                filter: None,
                returning: Vec::new(),
                restart_identity: t.restart_identity,
            }))
        },
        ast::Statement::Insert(ins) => analyze_insert(ins, catalog).map(LogicalPlan::Insert),
        ast::Statement::Select(sel) => {
            analyze_select(sel, catalog).map(|p| LogicalPlan::Select(Box::new(p)))
        },
        // Set operations are parsed but the column-compat / type-promotion resolver and
        // the physical Union/Intersect/Except path are not yet wired.
        // Set operations: analyze each leaf SELECT, check column-count + per-column type
        // compatibility across operands, and resolve ORDER BY/LIMIT against the output columns.
        ast::Statement::SetOperation(so) => {
            analyze_set_operation(so, catalog).map(LogicalPlan::SetOperation)
        },
        ast::Statement::Update(upd) => analyze_update(upd, catalog).map(LogicalPlan::Update),
        ast::Statement::Delete(del) => analyze_delete(del, catalog).map(LogicalPlan::Delete),
        // COPY is parsed but its data rides the wire's COPY sub-protocol, so the
        // executor/wire pipeline drives it rather than the normal plan path.
        ast::Statement::Copy(_) => Err(Error::Unsupported(
            "COPY runs over the wire protocol's COPY sub-protocol \
             — issue it through a client connection"
                .to_owned(),
        )),
        // MERGE: a join-driven matched/not-matched apply over a single named source table.
        ast::Statement::Merge(m) => analyze_merge(m, catalog).map(LogicalPlan::Merge),
        ast::Statement::Explain(inner, options) => {
            let inner_plan = analyze(*inner, catalog)?;
            Ok(LogicalPlan::Explain(Box::new(inner_plan), options))
        },
        // BEGIN applies the requested ISOLATION LEVEL / READ ONLY|WRITE characteristics to the
        // transaction it opens. A plain BEGIN keeps the session defaults.
        ast::Statement::BeginTransaction(settings) => Ok(LogicalPlan::BeginTransaction(
            txn_characteristics(&settings),
        )),
        ast::Statement::Commit => Ok(LogicalPlan::Commit),
        ast::Statement::Rollback => Ok(LogicalPlan::Rollback),
        // Savepoints: the engine implements savepoint / rollback_to / release_savepoint, so
        // carry the name through to the executor, which runs it against the active transaction.
        ast::Statement::Savepoint(name) => Ok(LogicalPlan::Savepoint(name)),
        ast::Statement::RollbackToSavepoint(name) => Ok(LogicalPlan::RollbackToSavepoint(name)),
        ast::Statement::ReleaseSavepoint(name) => Ok(LogicalPlan::ReleaseSavepoint(name)),
        // LISTEN / UNLISTEN / NOTIFY (async pub/sub): the channel registry spans connections and lives
        // in the wire server, which intercepts these before analysis. They have no engine-level plan,
        // so reaching the analyzer means an execution path that does not intercept them (e.g. inside a
        // procedure body or a test double); reject loudly rather than silently doing nothing.
        ast::Statement::Listen(_) | ast::Statement::Unlisten(_) | ast::Statement::Notify { .. } => {
            Err(Error::Unsupported(
                "LISTEN/UNLISTEN/NOTIFY are only valid on a live client connection".to_owned(),
            ))
        },
        // SET TRANSACTION updates the session's default characteristics for subsequently-started
        // transactions.
        ast::Statement::SetTransaction(settings) => {
            Ok(LogicalPlan::SetTransaction(txn_characteristics(&settings)))
        },
        // SET/RESET session variables: the session keeps a generic variable store; carry the
        // name + value (None = RESET) through to the executor.
        ast::Statement::SetVariable(sv) => Ok(LogicalPlan::SetVariable {
            name: sv.name,
            value: sv.value,
        }),
        // SHOW: report the session variable's current value.
        ast::Statement::Show(name) => Ok(LogicalPlan::ShowVariable(name)),
        // SHOW TABLES / SHOW COLUMNS: catalog introspection. TABLES needs no resolution
        // (the executor lists the engine's tables); COLUMNS resolves the table to its schema.
        ast::Statement::ShowTables => Ok(LogicalPlan::ShowTables),
        ast::Statement::ShowColumns(table) => {
            let schema = resolve_table(None, &table, catalog)?;
            Ok(LogicalPlan::ShowColumns(schema))
        },
        ast::Statement::Vacuum(options) => Ok(LogicalPlan::Vacuum(options)),
        ast::Statement::Reindex => Ok(LogicalPlan::Reindex),
        ast::Statement::Analyze(an) => analyze_analyze(an, catalog).map(LogicalPlan::Analyze),
        ast::Statement::LockTable { tables, mode } => {
            // Resolve every named table (each must exist); the executor then acquires the lock.
            let tables = tables
                .iter()
                .map(|name| resolve_table(None, name, catalog))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(LogicalPlan::LockTable { tables, mode })
        },
        ast::Statement::Prepare { name, statement } => {
            // The body is stored un-analyzed (its `$n` types are only known at EXECUTE); count its
            // placeholders so EXECUTE can check the argument arity.
            let param_count = crate::params::parameter_count(&statement);
            Ok(LogicalPlan::Prepare {
                name,
                statement,
                param_count,
            })
        },
        ast::Statement::Execute { name, args } => {
            // The arguments must be constants (they bind `$n` before the statement is analyzed).
            let args = args
                .iter()
                .map(const_value)
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(LogicalPlan::Execute { name, args })
        },
        ast::Statement::Deallocate(target) => Ok(LogicalPlan::Deallocate(target)),
        ast::Statement::CommentOn(c) => analyze_comment(c, catalog).map(LogicalPlan::Comment),
    }
}

/// Reduce a constant `EXECUTE` argument expression to a literal [`ast::Value`]. Only literals
/// and a unary minus on a numeric literal are accepted; anything that would need a row context or
/// catalog (a column reference, function call, subquery, …) is rejected.
fn const_value(expr: &ast::Expr) -> Result<ast::Value, Error> {
    match expr {
        ast::Expr::Literal(value) => Ok(value.clone()),
        ast::Expr::Unary {
            op: ast::UnaryOp::Negate,
            expr,
        } => match expr.as_ref() {
            ast::Expr::Literal(ast::Value::Int(n)) => Ok(ast::Value::Int(n.wrapping_neg())),
            ast::Expr::Literal(ast::Value::Float(f)) => Ok(ast::Value::Float(-f)),
            ast::Expr::Literal(ast::Value::Numeric(d)) => {
                Ok(ast::Value::Numeric(crate::numeric::Decimal {
                    mantissa: d.mantissa.wrapping_neg(),
                    scale: d.scale,
                }))
            },
            _ => Err(Error::Unsupported(
                "EXECUTE argument must be a constant value".to_owned(),
            )),
        },
        _ => Err(Error::Unsupported(
            "EXECUTE argument must be a constant value".to_owned(),
        )),
    }
}

/// Analyze `CREATE [OR REPLACE] TRIGGER ...`. The target table must exist and must not be a
/// system catalog. The action and `WHEN` bodies stay as text — they reference the `NEW`/`OLD`
/// pseudo-rows, which only exist when the trigger fires, so they are re-parsed, substituted, and
/// analyzed at fire time (see the executor's trigger module) rather than now.
fn analyze_create_trigger(
    ct: ast::CreateTrigger,
    catalog: &dyn Catalog,
) -> Result<LogicalPlan, Error> {
    enforce_system_catalog(&ct.table, catalog)?;
    // The target table must exist (resolve discards the schema — the executor re-resolves at fire
    // time under the live snapshot).
    resolve_table(None, &ct.table, catalog)?;
    Ok(LogicalPlan::CreateTrigger(CreateTriggerPlan {
        name: ct.name,
        or_replace: ct.or_replace,
        table: ct.table,
        timing: ct.timing,
        events: ct.events,
        for_each: ct.for_each,
        when: ct.when,
        action: ct.action,
    }))
}

/// Analyze `CREATE TABLE [IF NOT EXISTS] name AS <select>`. The body is analyzed like any
/// `SELECT` and the new table's schema is derived from its projection (name + type); column aliases in
/// the `SELECT` name the output columns. The executor creates the table and inserts the query's rows.
/// Unlike a materialized view, the result is an independent table with no recorded definition and no
/// incremental maintenance.
fn analyze_create_table_as(
    ct: ast::CreateTableAs,
    catalog: &dyn Catalog,
) -> Result<LogicalPlan, Error> {
    // A new table named like a system catalog (or an existing view) would shadow it.
    enforce_system_catalog(&ct.name, catalog)?;
    let exists =
        catalog.lookup_table(&ct.name)?.is_some() || catalog.lookup_view(&ct.name)?.is_some();
    if exists && !ct.if_not_exists {
        return Err(Error::TableExists { name: ct.name });
    }
    let body = analyze_select((*ct.query).clone(), catalog)?;
    let columns: Vec<(String, ColumnType)> = body
        .projection
        .iter()
        .map(|p| (p.name.clone(), p.expr.ty))
        .collect();
    if columns.is_empty() {
        return Err(Error::Unsupported(
            "CREATE TABLE ... AS SELECT with no output columns".to_owned(),
        ));
    }
    // A table cannot have two columns of the same name (e.g. `SELECT id, id`).
    let mut seen = std::collections::HashSet::with_capacity(columns.len());
    for (name, _) in &columns {
        if !seen.insert(name.as_str()) {
            return Err(Error::Unsupported(format!(
                "CREATE TABLE ... AS SELECT: duplicate output column name {name:?} (alias the \
                 SELECT columns to make them distinct)"
            )));
        }
    }
    Ok(LogicalPlan::CreateTableAs(CreateTableAsPlan {
        name: ct.name,
        columns,
        body: Box::new(body),
        if_not_exists: ct.if_not_exists,
    }))
}

/// Analyze `CREATE MATERIALIZED VIEW name [(cols)] AS <select>`. The body is analyzed like any
/// `SELECT`; the output schema is derived from its projection (name + type), with an explicit column
/// list overriding the inferred names. The executor computes the rows and stores them in a backing
/// table named `name`, so querying the view is an ordinary table scan.
fn analyze_create_materialized_view(
    cv: ast::CreateView,
    catalog: &dyn Catalog,
) -> Result<LogicalPlan, Error> {
    // A view's definition (and a matview's rows) back onto `nusadb_*` catalog tables, so creating
    // one with a system-catalog name would squat (or, with OR REPLACE, forge) a catalog table.
    enforce_system_catalog(&cv.name, catalog)?;
    // The view name must be free unless OR REPLACE (the executor drops the old one first) or
    // IF NOT EXISTS (the executor re-checks under its own snapshot and no-ops on a clash).
    if !cv.or_replace
        && !cv.if_not_exists
        && (catalog.lookup_table(&cv.name)?.is_some() || catalog.lookup_view(&cv.name)?.is_some())
    {
        return Err(Error::TableExists { name: cv.name });
    }
    let body = analyze_select((*cv.query).clone(), catalog)?;
    if !cv.materialized {
        // A plain view re-evaluates on read: validate the body now (done above) and store its SQL
        // plus any explicit column-name list; querying it inlines the parsed body and renames its
        // columns positionally (see `resolve_view`). No backing table. Validate the arity here for
        // an early error rather than deferring it to the first read.
        if !cv.columns.is_empty() && cv.columns.len() != body.projection.len() {
            return Err(Error::ArityMismatch {
                context: "CREATE VIEW column list".to_owned(),
                expected: body.projection.len(),
                found: cv.columns.len(),
            });
        }
        return Ok(LogicalPlan::CreateView(CreatePlainViewPlan {
            name: cv.name,
            or_replace: cv.or_replace,
            if_not_exists: cv.if_not_exists,
            definition_sql: cv.definition_sql,
            columns: cv.columns,
        }));
    }
    let mut columns: Vec<(String, ColumnType)> = body
        .projection
        .iter()
        .map(|p| (p.name.clone(), p.expr.ty))
        .collect();
    if !cv.columns.is_empty() {
        if cv.columns.len() != columns.len() {
            return Err(Error::ArityMismatch {
                context: "CREATE MATERIALIZED VIEW column list".to_owned(),
                expected: columns.len(),
                found: cv.columns.len(),
            });
        }
        for (col, name) in columns.iter_mut().zip(cv.columns) {
            col.0 = name;
        }
    }
    let ivm_base = ivm_base_table(&body);
    Ok(LogicalPlan::CreateMaterializedView(MaterializedViewPlan {
        name: cv.name,
        or_replace: cv.or_replace,
        if_not_exists: cv.if_not_exists,
        columns,
        body: Box::new(body),
        definition_sql: cv.definition_sql,
        ivm_base,
    }))
}

/// The base table a materialized view body can be *incrementally* maintained over, or
/// `None` if the body is not IVM-eligible (then it is full-refresh-only).
///
/// Eligible = a single base table with **only** a projection and an optional `WHERE`: no join,
/// aggregate, `GROUP BY`/grouping set, window, `DISTINCT`, `HAVING`, `LIMIT`/`OFFSET`, recursive CTE,
/// or CTE base — and every projection/filter expression is *stable* (subquery-free and
/// non-volatile). Under these conditions each base row contributes exactly one view row (bag
/// semantics), so an insert appends one row and a delete removes one — keeping the view byte-for-byte
/// what a full `REFRESH` would produce.
fn ivm_base_table(body: &SelectPlan) -> Option<String> {
    let table = body.table.as_ref()?;
    let structurally_simple = body.from_cte.is_none()
        && body.joins.is_empty()
        && !body.distinct
        && body.distinct_on.is_empty()
        && body.group_keys.is_empty()
        && body.grouping_sets.is_empty()
        && body.aggregates.is_empty()
        && body.windows.is_empty()
        && body.having.is_none()
        && body.limit.is_none()
        && body.offset.is_none()
        && body.recursive_ctes.is_empty();
    if !structurally_simple {
        return None;
    }
    if body.filter.as_ref().is_some_and(|f| !expr_is_ivm_stable(f)) {
        return None;
    }
    if !body.projection.iter().all(|p| expr_is_ivm_stable(&p.expr)) {
        return None;
    }
    Some(table.name.clone())
}

/// Whether a typed expression is stable enough to maintain incrementally: it always yields
/// the same value for the same base row. Rejects subqueries, correlated/aggregate references, scalar
/// UDFs (which may be non-deterministic), set-returning calls, and volatile built-ins (`NOW`,
/// `RANDOM`, `gen_random_uuid`, session functions, …); recurses through every other node's children.
fn expr_is_ivm_stable(expr: &TypedExpr) -> bool {
    use crate::planner::TypedExprKind as K;
    match &expr.kind {
        K::Literal(_) | K::Column(_) => true,
        // Non-deterministic / context-dependent / can't-incrementalize nodes.
        K::OuterColumn { .. }
        | K::AggregateRef(_)
        | K::ScalarUdf { .. }
        | K::SetReturning { .. }
        | K::ScalarSubquery(_)
        | K::Exists { .. }
        | K::InSubquery { .. }
        | K::QuantifiedSubquery { .. } => false,
        K::ScalarFunction { func, args } => {
            !is_volatile_scalar_func(*func) && args.iter().all(expr_is_ivm_stable)
        },
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            expr_is_ivm_stable(left) && expr_is_ivm_stable(right)
        },
        K::QuantifiedArray { expr, array, .. } => {
            expr_is_ivm_stable(expr) && expr_is_ivm_stable(array)
        },
        K::Unary { expr, .. }
        | K::IsNull { expr, .. }
        | K::IsBool { expr, .. }
        | K::Cast(expr, _) => expr_is_ivm_stable(expr),
        K::InList { expr, list, .. } => {
            expr_is_ivm_stable(expr) && list.iter().all(expr_is_ivm_stable)
        },
        K::Between {
            expr, low, high, ..
        } => expr_is_ivm_stable(expr) && expr_is_ivm_stable(low) && expr_is_ivm_stable(high),
        K::Like { expr, pattern, .. }
        | K::SimilarTo { expr, pattern, .. }
        | K::RegexMatch { expr, pattern, .. } => {
            expr_is_ivm_stable(expr) && expr_is_ivm_stable(pattern)
        },
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(expr_is_ivm_stable)
                && branches
                    .iter()
                    .all(|b| expr_is_ivm_stable(&b.when) && expr_is_ivm_stable(&b.then))
                && default.as_deref().is_none_or(expr_is_ivm_stable)
        },
        K::Coalesce(args) | K::ArrayLiteral(args) => args.iter().all(expr_is_ivm_stable),
        K::Crypto { value, key, .. } => expr_is_ivm_stable(value) && expr_is_ivm_stable(key),
        K::Subscript { base, index } => expr_is_ivm_stable(base) && expr_is_ivm_stable(index),
        K::ArraySlice { base, lower, upper } => {
            expr_is_ivm_stable(base)
                && lower.as_deref().is_none_or(expr_is_ivm_stable)
                && upper.as_deref().is_none_or(expr_is_ivm_stable)
        },
    }
}

/// Whether a scalar built-in is volatile (a fresh/contextual value per evaluation), so a view using
/// it cannot be incrementally maintained.
const fn is_volatile_scalar_func(func: ast::ScalarFunc) -> bool {
    use ast::ScalarFunc as F;
    matches!(
        func,
        F::Now
            | F::CurrentTimestamp
            | F::CurrentDate
            | F::CurrentTime
            // `AGE(value)` (one-argument) is relative to the current date, so a view projecting it
            // cannot be incrementally maintained: stored rows would keep the age computed at insert
            // time and drift stale across days. Mirrors the result-cache volatility list so the two
            // denylists do not disagree (deep-gate sibling).
            | F::Age
            | F::CurrentUser
            | F::SessionUser
            | F::CurrentSetting
            | F::Random
            | F::Setseed
            | F::UuidGenerateV4
            | F::Version
            | F::CurrentDatabase
            | F::CurrentSchema
    )
}

/// Translate parsed `BEGIN` / `SET TRANSACTION` characteristics into the plan-level
/// [`TxnCharacteristics`]: map the SQL isolation enum onto the engine's canonical
/// `IsolationLevel` and fold `READ ONLY` / `READ WRITE` into a tri-state flag.
fn txn_characteristics(settings: &ast::TransactionSettings) -> TxnCharacteristics {
    use nusadb_core::engine::IsolationLevel as Engine;

    let isolation = settings.isolation.map(|level| match level {
        ast::IsolationLevel::ReadUncommitted => Engine::ReadUncommitted,
        ast::IsolationLevel::ReadCommitted => Engine::ReadCommitted,
        ast::IsolationLevel::RepeatableRead => Engine::RepeatableRead,
        ast::IsolationLevel::Serializable => Engine::Serializable,
    });
    let read_only = settings
        .access_mode
        .map(|mode| matches!(mode, ast::AccessMode::ReadOnly));
    TxnCharacteristics {
        isolation,
        read_only,
    }
}

/// Resolve a `COMMENT ON` target against the catalog: the table (and, for a column comment, the
/// column) must exist. The comment text itself needs no validation. Persisting the comment is
/// optional treaty work, so the resolved plan carries only the target names + text.
fn analyze_comment(c: ast::CommentOn, catalog: &dyn Catalog) -> Result<CommentPlan, Error> {
    let (table_name, column) = match c.target {
        ast::CommentTarget::Table { table } => (table, None),
        ast::CommentTarget::Column { table, column } => (table, Some(column)),
    };
    enforce_system_catalog(&table_name, catalog)?;
    let schema = catalog
        .lookup_table(&table_name)?
        .ok_or_else(|| Error::TableNotFound {
            name: table_name.clone(),
        })?;
    if let Some(column) = &column
        && !schema.columns.iter().any(|c| &c.name == column)
    {
        return Err(Error::ColumnNotFound {
            table: table_name,
            column: column.clone(),
        });
    }
    Ok(CommentPlan {
        table: table_name,
        column,
        comment: c.comment,
    })
}

// === Name resolution ======================================================

/// Name prefix every engine-internal system-catalog table shares.
///
/// Covers `nusadb_policies`, `nusadb_rls`, `nusadb_views`, `nusadb_matviews`, ... Public so
/// out-of-analyzer entry points (the wire COPY sub-protocol, which bypasses the analyzer) apply
/// the same namespace reservation.
pub const SYSTEM_TABLE_PREFIX: &str = "nusadb_";

/// Name prefix marking a *synthetic* type-bound `CHECK`.
///
/// It tags the desugared enforcement of a `VARCHAR(n)` length or a narrow-integer width — an
/// implementation detail (the bound is a property of the declared type, not a user constraint). Such
/// checks are still enforced on every write but hidden from constraint introspection
/// (`information_schema.table_constraints`).
pub const SYNTHETIC_TYPE_CHECK_PREFIX: &str = "nusadb_typeck_";

/// Reserve the `nusadb_*` system-catalog namespace to superusers.
///
/// The RLS and view catalogs are ordinary engine tables, so without this guard any authenticated
/// user could read or rewrite them with plain SQL — e.g. `INSERT` a permissive policy into
/// `nusadb_policies`, `DELETE` the `nusadb_rls` toggle rows, or widen a policy with `UPDATE` —
/// silently disabling row-level security for every table (the system-catalog DML bypass the design
/// demonstrated). Applied at every site a user statement resolves, creates, drops, or alters a
/// table by name (DML targets, `SELECT`/join bases, DDL, and view names, which back onto tables);
/// engine-internal accesses go through the catalog port / executor directly and are unaffected.
/// Superusers keep full access (introspection + administration escape hatch).
pub(super) fn enforce_system_catalog(name: &str, catalog: &dyn Catalog) -> Result<(), Error> {
    if name.starts_with(SYSTEM_TABLE_PREFIX) && !catalog.is_superuser() {
        return Err(Error::PermissionDenied(format!(
            "`{name}` is in the reserved system-catalog namespace (`{SYSTEM_TABLE_PREFIX}*`); \
             only a superuser may reference it"
        )));
    }
    Ok(())
}

/// A table reference for an error message: `schema.name` when qualified by a non-default schema,
/// otherwise the bare name.
pub(super) fn qualified_display(schema: &str, name: &str) -> String {
    if schema == nusadb_core::PUBLIC_SCHEMA {
        name.to_owned()
    } else {
        format!("{schema}.{name}")
    }
}

/// Like [`qualified_display`], for an optional qualifier: `schema.name` when explicitly qualified by
/// a non-default schema, otherwise the bare name.
pub(super) fn qualified_display_opt(schema: Option<&str>, name: &str) -> String {
    schema.map_or_else(|| name.to_owned(), |s| qualified_display(s, name))
}

/// Resolve a table reference to its schema, or `Ok(None)` if no such table exists. An explicit
/// qualifier resolves in exactly that schema; an unqualified name walks the session
/// [`search_path`](Catalog::search_path), taking the first schema that has it.
fn lookup_table_ref(
    schema: Option<&str>,
    name: &str,
    catalog: &dyn Catalog,
) -> Result<Option<TableSchema>, Error> {
    // A data-modifying CTE's target may not be resolved anywhere but the CTE itself: the
    // outer query and siblings — including their subqueries, which resolve every base table through
    // here — must not observe the CTE's writes. The DML lifts its own target while it is analyzed, so
    // this fires only for an *outside* read. Reject loudly rather than return snapshot-divergent rows.
    if FORBIDDEN_DML_TARGETS.with(|m| m.borrow().get(name).is_some_and(|&c| c > 0)) {
        return Err(Error::Unsupported(format!(
            "table `{name}` is modified by a data-modifying CTE in this statement and cannot also be \
             read here (reference the CTE's RETURNING rows instead)"
        )));
    }
    if let Some(s) = schema {
        return catalog.lookup_table_in(s, name);
    }
    // Unqualified: walk the search path, taking the first schema that has the table.
    for s in catalog.search_path() {
        if let Some(table) = catalog.lookup_table_in(&s, name)? {
            return Ok(Some(table));
        }
    }
    Ok(None)
}

// A data-modifying CTE's target table, while it is forbidden, is keyed by folded name to a positive
// count (so overlapping forbids and the per-DML "lift my own target" nest correctly). Resolving such
// a table anywhere other than the data-modifying CTE itself would diverge from the statement-snapshot
// semantics (the outer query must not see the CTE's writes), so it is rejected.
thread_local! {
    static FORBIDDEN_DML_TARGETS: std::cell::RefCell<std::collections::HashMap<String, usize>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Forbid resolving each `names` table for the guard's lifetime; the count is restored on
/// drop, so nested/overlapping forbids compose.
#[must_use]
pub(super) fn forbid_dml_targets(names: &[String]) -> ForbidGuard {
    FORBIDDEN_DML_TARGETS.with(|m| {
        let mut m = m.borrow_mut();
        for n in names {
            *m.entry(n.clone()).or_insert(0) += 1;
        }
    });
    ForbidGuard {
        names: names.to_vec(),
    }
}

/// Temporarily *lift* the forbid on `name` for the guard's lifetime — the one table a data-modifying
/// CTE is itself allowed to resolve (it is the modification target). Restored on drop.
#[must_use]
pub(super) fn allow_dml_target(name: &str) -> Option<AllowGuard> {
    let lifted = FORBIDDEN_DML_TARGETS.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(c) = m.get_mut(name)
            && *c > 0
        {
            *c -= 1;
            return true;
        }
        false
    });
    lifted.then(|| AllowGuard {
        name: name.to_owned(),
    })
}

/// Restores the forbid counts for [`forbid_dml_targets`] on drop.
pub(super) struct ForbidGuard {
    names: Vec<String>,
}
impl Drop for ForbidGuard {
    fn drop(&mut self) {
        FORBIDDEN_DML_TARGETS.with(|m| {
            let mut m = m.borrow_mut();
            for n in &self.names {
                if let Some(c) = m.get_mut(n) {
                    *c -= 1;
                    if *c == 0 {
                        m.remove(n);
                    }
                }
            }
        });
    }
}

/// Re-applies the forbid lifted by [`allow_dml_target`] on drop.
pub(super) struct AllowGuard {
    name: String,
}
impl Drop for AllowGuard {
    fn drop(&mut self) {
        FORBIDDEN_DML_TARGETS.with(|m| {
            *m.borrow_mut().entry(self.name.clone()).or_insert(0) += 1;
        });
    }
}

/// Resolve a table reference `(schema, name)` to its schema, erroring if absent. An explicit
/// qualifier resolves in that schema; an unqualified name walks the session search path.
fn resolve_table(
    schema: Option<&str>,
    name: &str,
    catalog: &dyn Catalog,
) -> Result<TableSchema, Error> {
    enforce_system_catalog(name, catalog)?;
    let table = lookup_table_ref(schema, name, catalog)?.ok_or_else(|| Error::TableNotFound {
        name: qualified_display_opt(schema, name),
    })?;
    enforce_table_rls(name, catalog)?;
    Ok(table)
}

/// Fail closed when a non-superuser touches a base table with row-level security enabled.
///
/// Until policy-based access lands, an RLS-enabled table is superuser-only: refusing is the
/// safe choice — never silently return every row to a user the policy would restrict. The check
/// short-circuits on [`Catalog::is_superuser`] first, so a superuser session never pays for the RLS
/// lookup, and so a minimal catalog (default superuser) is unaffected. Applied at every base-table
/// resolution site (DML targets via [`resolve_table`], and the `SELECT` `FROM` base).
pub(super) fn enforce_table_rls(name: &str, catalog: &dyn Catalog) -> Result<(), Error> {
    if !catalog.is_superuser() && catalog.rls_enabled(name)? {
        return Err(Error::Unsupported(format!(
            "row-level security is enabled on `{name}`; this access is not yet supported under RLS \
             (only single-table reads are policy-filtered), so it is allowed only for a superuser"
        )));
    }
    Ok(())
}

/// Validate a `CREATE POLICY` against its table and produce the plan to persist it. The table
/// must exist; each of `USING` / `WITH CHECK` (when present) must type-check as a boolean predicate
/// over the table's columns — caught here so a malformed policy is rejected at creation, not silently
/// stored.
fn analyze_create_policy(
    cp: ast::CreatePolicy,
    catalog: &dyn Catalog,
) -> Result<LogicalPlan, Error> {
    require_rls_admin(catalog, "create a row-level-security policy")?;
    let table = catalog
        .lookup_table(&cp.table)?
        .ok_or_else(|| Error::TableNotFound {
            name: cp.table.clone(),
        })?;
    let scope = single_table_scope(&table);
    for predicate in [cp.using.as_deref(), cp.check.as_deref()]
        .into_iter()
        .flatten()
    {
        let expr = crate::parser::parse_expression(predicate)?;
        // `analyze_predicate` type-checks the expression as boolean against the table's columns.
        analyze_predicate(Some(expr), &scope, catalog)?;
    }
    Ok(LogicalPlan::CreatePolicy(CreatePolicyPlan {
        name: cp.name,
        table: table.name,
        replace: false,
        permissive: cp.permissive,
        command: cp.command,
        roles: cp.roles,
        using: cp.using,
        check: cp.check,
    }))
}

/// Validate an `ALTER POLICY` against its table and produce a (replacing) [`CreatePolicyPlan`].
///
/// The policy must already exist on the table; the omitted clauses (`TO` / `USING` / `WITH CHECK`)
/// are filled from the existing definition, and the kind (permissive/restrictive) and command are
/// preserved (they cannot be altered). Any new `USING` / `WITH CHECK` must type-check as a boolean
/// over the table's columns. Lowered to a row replacement so it reuses the `CreatePolicy` path.
fn analyze_alter_policy(ap: ast::AlterPolicy, catalog: &dyn Catalog) -> Result<LogicalPlan, Error> {
    require_rls_admin(catalog, "alter a row-level-security policy")?;
    let table = catalog
        .lookup_table(&ap.table)?
        .ok_or_else(|| Error::TableNotFound {
            name: ap.table.clone(),
        })?;
    let existing = catalog
        .lookup_policies(&table.name)?
        .into_iter()
        .find(|p| p.name == ap.name)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "policy `{}` does not exist on `{}`",
                ap.name, table.name
            ))
        })?;

    // Omitted clauses keep the existing policy's parts; a `Some` replaces them.
    let roles = ap.roles.unwrap_or(existing.roles);
    let using = ap.using.or(existing.using);
    let check = ap.check.or(existing.check);

    let scope = single_table_scope(&table);
    for predicate in [using.as_deref(), check.as_deref()].into_iter().flatten() {
        let expr = crate::parser::parse_expression(predicate)?;
        analyze_predicate(Some(expr), &scope, catalog)?;
    }
    Ok(LogicalPlan::CreatePolicy(CreatePolicyPlan {
        name: ap.name,
        table: table.name,
        replace: true,
        permissive: existing.permissive,
        command: existing.command,
        roles,
        using,
        check,
    }))
}

/// Reserve a row-level-security administration statement to superusers.
///
/// Creating/altering/dropping a policy and toggling a table's RLS are security administration: a
/// non-superuser must not run them, or the very session RLS is meant to constrain could lift its own
/// restrictions (e.g. `ALTER TABLE t DISABLE ROW LEVEL SECURITY`, or `ALTER POLICY ... USING (TRUE)`).
/// Full role-based access control is deferred; this guards the security-critical RLS cases.
pub(super) fn require_rls_admin(catalog: &dyn Catalog, action: &str) -> Result<(), Error> {
    if catalog.is_superuser() {
        Ok(())
    } else {
        Err(Error::PermissionDenied(format!(
            "only a superuser may {action}"
        )))
    }
}

fn find_column<'a>(
    columns: &'a [ColumnDef],
    name: &str,
    table: &str,
) -> Result<(usize, &'a ColumnDef), Error> {
    columns
        .iter()
        .enumerate()
        .find(|(_, column)| column.name == name)
        .ok_or_else(|| Error::ColumnNotFound {
            table: table.to_owned(),
            column: name.to_owned(),
        })
}

/// One column visible in a `SELECT`'s scope, tagged with the table name or
/// alias that owns it. The column's position in the scope slice is its ordinal
/// in the (possibly joined) row the executor produces.
#[derive(Clone)]
pub(crate) struct ScopedColumn {
    /// Owning table name or alias (the qualifier in `table.column`).
    qualifier: String,
    /// The column definition.
    def: ColumnDef,
    /// When `true`, the column is reachable only through its qualifier (`excluded.col` / `right.col`),
    /// never as a bare `col`, and is omitted from `SELECT *`. Two uses: the `EXCLUDED` pseudo-relation
    /// of `ON CONFLICT DO UPDATE`, whose names mirror the target table's so a bare reference
    /// must resolve to the target row; and the right side's copy of a `USING`/`NATURAL` join column,
    /// which is merged into one output column reached via the left side.
    /// `false` for an ordinary table/CTE column.
    qualified_only: bool,
}

/// Build the scope for a single table (the common, non-join case).
fn scope_of(table: &TableSchema) -> Vec<ScopedColumn> {
    scope_of_aliased(table, &table.name)
}

/// Like [`scope_of`] but with an explicit `qualifier` (an alias) for the columns — for the secondary
/// table of `UPDATE ... FROM` / `DELETE ... USING` / `MERGE`, where the same table may appear under
/// an alias.
pub(crate) fn scope_of_aliased(table: &TableSchema, qualifier: &str) -> Vec<ScopedColumn> {
    table
        .columns
        .iter()
        .map(|def| ScopedColumn {
            qualifier: qualifier.to_owned(),
            def: def.clone(),
            qualified_only: false,
        })
        .collect()
}

/// Resolve a column reference against `scope`, returning its row ordinal and
/// type. `qualifier = None` matches across all tables (more than one match is
/// ambiguous); `Some(q)` restricts to columns owned by table/alias `q`.
fn resolve_scoped(
    scope: &[ScopedColumn],
    qualifier: Option<&str>,
    name: &str,
) -> Result<(usize, ColumnType), Error> {
    let mut found: Option<(usize, ColumnType)> = None;
    for (index, col) in scope.iter().enumerate() {
        if col.def.name != name {
            continue;
        }
        if qualifier.is_some_and(|q| col.qualifier != q) {
            continue;
        }
        // An `EXCLUDED` column is invisible to a bare (unqualified) reference.
        if qualifier.is_none() && col.qualified_only {
            continue;
        }
        if found.is_some() {
            return Err(Error::Unsupported(format!(
                "ambiguous column reference `{name}` — qualify it with a table name"
            )));
        }
        // A column *reference* carries its expression type: `VARCHAR(n)`/`CHAR(n)` behave as `TEXT`
        // (the declared length is catalog metadata only), but the integer widths are kept so a
        // `BIGINT` column is not falsely bounded at int4 and a `SMALLINT`/`INT` column enforces its
        // range in arithmetic.
        found = Some((index, expr_type(col.def.ty)));
    }
    found.ok_or_else(|| Error::ColumnNotFound {
        table: qualifier
            .map(str::to_owned)
            .or_else(|| scope.first().map(|c| c.qualifier.clone()))
            .unwrap_or_default(),
        column: name.to_owned(),
    })
}

thread_local! {
    /// Stack of enclosing-query scopes while analyzing a correlated subquery body. The top
    /// frame is the immediately enclosing query; a column that misses the subquery's own scope is
    /// resolved against these, innermost-first, yielding a [`TypedExprKind::OuterColumn`].
    static OUTER_SCOPES: std::cell::RefCell<Vec<Vec<ScopedColumn>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Push `scope` as the enclosing scope for the duration of the returned guard, so a subquery body
/// analyzed within it can resolve correlated references to `scope`'s columns.
#[must_use]
fn push_outer_scope(scope: &[ScopedColumn]) -> OuterScopeGuard {
    OUTER_SCOPES.with(|stack| stack.borrow_mut().push(scope.to_vec()));
    OuterScopeGuard
}

/// Pops the enclosing scope pushed by [`push_outer_scope`] on drop.
struct OuterScopeGuard;

impl Drop for OuterScopeGuard {
    fn drop(&mut self) {
        OUTER_SCOPES.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Resolve a column reference against the local `scope` first, then — for a correlated subquery
/// — against the enclosing scopes pushed by [`push_outer_scope`]. A local hit is a
/// [`TypedExprKind::Column`]; an enclosing hit is a [`TypedExprKind::OuterColumn`] tagged with how
/// many levels out it resolved. A local *ambiguity* is an error even if an outer scope would match
/// (the nearest scope wins, and within it the reference must be unambiguous).
fn resolve_scoped_or_outer(
    scope: &[ScopedColumn],
    qualifier: Option<&str>,
    name: &str,
) -> Result<(TypedExprKind, ColumnType), Error> {
    match resolve_scoped(scope, qualifier, name) {
        Ok((ordinal, ty)) => Ok((TypedExprKind::Column(ordinal), ty)),
        Err(not_found @ Error::ColumnNotFound { .. }) => OUTER_SCOPES.with(|stack| {
            for (depth, frame) in stack.borrow().iter().rev().enumerate() {
                if let Ok((ordinal, ty)) = resolve_scoped(frame, qualifier, name) {
                    return Ok((
                        TypedExprKind::OuterColumn {
                            level: depth + 1,
                            ordinal,
                        },
                        ty,
                    ));
                }
            }
            Err(not_found)
        }),
        Err(other) => Err(other),
    }
}

#[cfg(test)]
mod tests;
