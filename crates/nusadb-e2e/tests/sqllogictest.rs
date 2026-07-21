//! `sqllogictest` correctness suite for the NusaDB SQL surface.
//!
//! Each `.slt` file under `tests/slt/` is a self-contained scenario: it builds
//! its own [`BtreeEngine`] (fresh state, no cross-file leakage), drives
//! SQL through `parse → analyze → plan → execute`, and asserts the engine's
//! output matches the file's expected rows.
//!
//! This is the *SQL correctness* gate — one of the six test layers
//! per the SQL support policy. The corpus mirrors the SQL Support Priority
//! ordering: `p1_ddl`, `p2_dml`, `p3_filter`, `p4_join_agg`.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "integration test harness: panic-on-failure IS the assertion mechanism"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};
use sqllogictest::{DBOutput, DefaultColumnType, Runner};

/// One `.slt` connection: a freshly-built [`BtreeEngine`] plus a
/// [`Session`] that survives across `statement` blocks so explicit
/// `BEGIN ... COMMIT` works the way a real client would observe.
///
/// The engine is `Box::leak`-promoted to `&'static` so the session can borrow
/// it without a self-referential struct dance. Each `.slt` file allocates
/// exactly one engine — the leak is bounded by the test binary's lifetime.
struct SltConnection {
    engine: &'static BtreeEngine,
    session: Session<'static>,
}

impl SltConnection {
    fn new() -> Self {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let session = Session::new(engine);
        Self { engine, session }
    }
}

/// Bridges the analyzer's narrow [`Catalog`] port to the full engine.
///
/// `txn` is the session's open explicit transaction (if any): when set, schema names resolve under
/// that transaction's snapshot (`lookup_table_as_of`) so DDL done earlier in the same `BEGIN ...`
/// block is visible to later statements — matching the wire server's `EngineCatalog`. In
/// auto-commit (`None`) the latest-committed view is correct.
struct SltCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: Option<nusadb_core::TxnId>,
    /// The session's ordered `search_path` schemas (from `SET search_path`), so a bare name resolves
    /// through them in order before `public`.
    search_path: Vec<String>,
}

impl SltCatalog<'_> {
    fn resolve(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.resolve_in(nusadb_core::PUBLIC_SCHEMA, name)
    }

    fn resolve_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.txn
            .map_or_else(
                || self.engine.lookup_table_in(schema, name),
                |txn| self.engine.lookup_table_as_of_in(txn, schema, name),
            )
            .map_err(Into::into)
    }
}

impl Catalog for SltCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.resolve(name)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.resolve_in(schema, name)
    }

    fn search_path(&self) -> Vec<String> {
        self.search_path.clone()
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.resolve(name)? else {
            return Ok(Vec::new());
        };
        // Constraint-backing (PK/UNIQUE/FK) indexes are enforced by scanning and not maintained on
        // write, so they are unsafe to scan — expose only SQL-maintained secondary indexes.
        let backing: std::collections::HashSet<_> = self
            .engine
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.engine.list_indexes(schema.id)? {
            if self
                .engine
                .lookup_index(&def.name)?
                .is_some_and(|id| backing.contains(&id))
            {
                continue;
            }
            // A functional/expression key or partial predicate is unsafe as a scan candidate —
            // mirror the production `catalog_list_indexes` exclusion.
            if !def.key_exprs.is_empty() || def.predicate.is_some() {
                continue;
            }
            out.push(IndexInfo {
                name: def.name,
                columns: def.columns,
                unique: def.unique,
            });
        }
        Ok(out)
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, nusadb_sql::Error> {
        // The view catalog is a regular table. Reuse the session's open transaction when there is
        // one so a view created earlier in the same explicit transaction is visible; otherwise read
        // in a throwaway transaction (each prior DDL committed in auto-commit).
        if let Some(txn) = self.txn {
            nusadb_sql::lookup_view_definition(self.engine, txn, name)
        } else {
            let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
            let result = nusadb_sql::lookup_view_definition(self.engine, txn, name);
            let _ = self.engine.commit(txn);
            result
        }
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, nusadb_sql::Error> {
        // Explicit `CREATE VIEW name (cols)` list. Same transaction handling as `lookup_view` so a
        // view created earlier in the same explicit transaction is visible.
        if let Some(txn) = self.txn {
            nusadb_sql::lookup_view_columns(self.engine, txn, name)
        } else {
            let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
            let result = nusadb_sql::lookup_view_columns(self.engine, txn, name);
            let _ = self.engine.commit(txn);
            result
        }
    }
}

impl sqllogictest::DB for SltConnection {
    type Error = nusadb_sql::Error;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        let stmt = parse(sql)?;
        // Resolve schema under the session's open transaction (if any) so DDL done earlier in the
        // same explicit `BEGIN ...` block is visible to later statements, matching the wire
        // server's `EngineCatalog`.
        let catalog = SltCatalog {
            engine: self.engine,
            txn: self.session.current_txn(),
            search_path: self.session.search_path(),
        };
        let logical = analyze(stmt, &catalog)?;
        let physical = plan(logical);
        let result = self.session.execute(physical)?;
        Ok(to_slt_output(result))
    }

    fn engine_name(&self) -> &'static str {
        "nusadb"
    }
}

/// Map our [`ExecutionResult`] into `sqllogictest`'s [`DBOutput`].
///
/// Non-row statements report their affected-row count (or 0 for DDL /
/// transaction control). Row statements format every [`Value`] as the canonical
/// `sqllogictest` text: NULL → `"NULL"`, bool → `0`/`1`, float → `{:.3}`.
fn to_slt_output(result: ExecutionResult) -> DBOutput<DefaultColumnType> {
    match result {
        ExecutionResult::Created(_)
        | ExecutionResult::Dropped
        | ExecutionResult::Altered
        | ExecutionResult::Analyzed { .. }
        | ExecutionResult::Commented
        | ExecutionResult::SchemaCreated
        | ExecutionResult::SchemaDropped
        | ExecutionResult::DatabaseCreated
        | ExecutionResult::DatabaseAltered
        | ExecutionResult::DatabaseDropped
        | ExecutionResult::SequenceCreated
        | ExecutionResult::SequenceDropped
        | ExecutionResult::IndexCreated
        | ExecutionResult::IndexDropped
        | ExecutionResult::TriggerCreated
        | ExecutionResult::TriggerDropped
        | ExecutionResult::TriggerAltered
        | ExecutionResult::ProcedureCreated
        | ExecutionResult::ProcedureDropped
        | ExecutionResult::ProcedureCalled
        | ExecutionResult::FunctionCreated
        | ExecutionResult::FunctionDropped
        | ExecutionResult::TransactionBegun
        | ExecutionResult::TransactionCommitted
        | ExecutionResult::TransactionRolledBack
        | ExecutionResult::TransactionCharacteristicsSet
        | ExecutionResult::SavepointCreated
        | ExecutionResult::RolledBackToSavepoint
        | ExecutionResult::SavepointReleased
        | ExecutionResult::TableLocked
        | ExecutionResult::Prepared
        | ExecutionResult::Deallocated
        | ExecutionResult::Reindexed
        | ExecutionResult::VariableSet => DBOutput::StatementComplete(0),
        ExecutionResult::Inserted(n)
        | ExecutionResult::Updated(n)
        | ExecutionResult::Deleted(n)
        | ExecutionResult::Merged(n)
        | ExecutionResult::Vacuumed(n) => DBOutput::StatementComplete(n as u64),
        ExecutionResult::Rows { rows, .. } => {
            let types = rows
                .first()
                .map_or_else(Vec::new, |row| row.iter().map(infer_column_type).collect());
            let rows = rows
                .into_iter()
                .map(|row| row.iter().map(format_value).collect())
                .collect();
            DBOutput::Rows { types, rows }
        },
    }
}

const fn infer_column_type(v: &Value) -> DefaultColumnType {
    match v {
        Value::Int(_) | Value::Bool(_) => DefaultColumnType::Integer,
        Value::Float(_) => DefaultColumnType::FloatingPoint,
        // Text, temporal, UUID, NUMERIC, JSON, INTERVAL, and arrays all render as text.
        Value::Text(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::TimeTz(_)
        | Value::Timestamp(_)
        | Value::TimestampTz(_)
        | Value::Uuid(_)
        | Value::Numeric(_)
        | Value::Json(_)
        | Value::Interval(_)
        | Value::Array(_)
        | Value::Vector(_)
        | Value::Bytes(_) => DefaultColumnType::Text,
        Value::Null => DefaultColumnType::Any,
    }
}

fn format_value(v: &Value) -> String {
    use nusadb_sql::temporal;
    match v {
        Value::Null => "NULL".to_owned(),
        Value::Bool(true) => "1".to_owned(),
        Value::Bool(false) => "0".to_owned(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{f:.3}"),
        Value::Text(s) if s.is_empty() => "(empty)".to_owned(),
        Value::Text(s) => s.clone(),
        // JSON renders in the spaced display form (`{"a": 1}`), matching standard jsonb output.
        Value::Json(s) => nusadb_sql::json::display_form(s),
        // Temporal + UUID in canonical text form.
        Value::Date(d) => temporal::format_date(*d),
        Value::Time(t) => temporal::format_time(*t),
        Value::Timestamp(t) => temporal::format_timestamp(*t),
        Value::TimestampTz(t) => temporal::format_timestamptz(*t),
        Value::TimeTz(t) => temporal::format_timetz(*t),
        Value::Uuid(u) => temporal::format_uuid(u),
        Value::Numeric(d) => d.format(),
        Value::Interval(iv) => iv.format(),
        Value::Array(items) => nusadb_sql::display::array_text(items),
        Value::Vector(v) => nusadb_sql::vector::format(v),
        Value::Bytes(b) => nusadb_sql::display::bytea_hex(b),
    }
}

/// Run one `.slt` file with a fresh engine. Each file owns its own scenario;
/// failures panic with the file/line where the comparison broke.
fn run_slt(path: &str) {
    let mut runner = Runner::new(|| async { Ok::<_, nusadb_sql::Error>(SltConnection::new()) });
    runner.run_file(path).expect("sqllogictest scenario failed");
}

// === The corpus =============================================================
//
// One `#[test]` per `.slt` so a failure points straight at the file. As
// coverage grows, add a new file under `tests/slt/<priority>/` and a new
// `#[test]` line here.

#[test]
fn slt_p1_create_drop() {
    run_slt("tests/slt/p1_ddl/create_drop.slt");
}

#[test]
fn slt_p1_drop_cascade() {
    run_slt("tests/slt/p1_ddl/drop_cascade.slt");
}

#[test]
fn slt_p1_create_table_as() {
    run_slt("tests/slt/p1_ddl/create_table_as.slt");
}

#[test]
fn slt_p1_schema_qualified() {
    run_slt("tests/slt/p1_ddl/schema_qualified.slt");
}

#[test]
fn slt_p1_information_schema() {
    run_slt("tests/slt/p1_ddl/information_schema.slt");
}

#[test]
fn slt_p1_information_schema_constraints() {
    run_slt("tests/slt/p1_ddl/information_schema_constraints.slt");
}

#[test]
fn slt_p1_varchar_length() {
    run_slt("tests/slt/p1_ddl/varchar_length.slt");
}

#[test]
fn slt_p1_int_range() {
    run_slt("tests/slt/p1_ddl/int_range.slt");
}

#[test]
fn slt_p1_real_jsonb_fidelity() {
    run_slt("tests/slt/p1_ddl/real_jsonb_fidelity.slt");
}

#[test]
fn slt_p2_insert_select() {
    run_slt("tests/slt/p2_dml/insert_select.slt");
}

#[test]
fn slt_p2_update_delete() {
    run_slt("tests/slt/p2_dml/update_delete.slt");
}

#[test]
fn slt_p2_upsert() {
    run_slt("tests/slt/p2_dml/upsert.slt");
}

#[test]
fn slt_p3_where_orderby_limit() {
    run_slt("tests/slt/p3_filter/where_orderby_limit.slt");
}

#[test]
fn slt_p3_null_handling() {
    run_slt("tests/slt/p3_filter/null_handling.slt");
}

#[test]
fn slt_p3_is_predicates() {
    run_slt("tests/slt/p3_filter/is_predicates.slt");
}

#[test]
fn slt_p3_quantified() {
    run_slt("tests/slt/p3_filter/quantified.slt");
}

#[test]
fn slt_p3_distinct_on() {
    run_slt("tests/slt/p3_filter/distinct_on.slt");
}

#[test]
fn slt_p3_nulls_ordering() {
    run_slt("tests/slt/p3_filter/nulls_ordering.slt");
}

#[test]
fn slt_p4_scalar_aggregate() {
    run_slt("tests/slt/p4_join_agg/scalar_aggregate.slt");
}

#[test]
fn slt_p4_stat_aggregates() {
    run_slt("tests/slt/p4_join_agg/stat_aggregates.slt");
}

#[test]
fn slt_p4_group_by_having() {
    run_slt("tests/slt/p4_join_agg/group_by_having.slt");
}

#[test]
fn slt_p4_grouping_sets() {
    run_slt("tests/slt/p4_join_agg/grouping_sets.slt");
}

#[test]
fn slt_p4_aggregate_modifiers() {
    run_slt("tests/slt/p4_join_agg/aggregate_modifiers.slt");
}

#[test]
fn slt_p4_join() {
    run_slt("tests/slt/p4_join_agg/join.slt");
}

#[test]
fn slt_p4_qualified_wildcard() {
    run_slt("tests/slt/p4_join_agg/qualified_wildcard.slt");
}

#[test]
fn slt_p4_sum_bigint_precision() {
    run_slt("tests/slt/p4_join_agg/sum_bigint_precision.slt");
}

#[test]
fn slt_p5_alter_table() {
    run_slt("tests/slt/p5_advanced/alter_table.slt");
}

#[test]
fn slt_p5_column_default() {
    run_slt("tests/slt/p5_advanced/column_default.slt");
}

#[test]
fn slt_p5_generated_columns() {
    run_slt("tests/slt/p5_advanced/generated_columns.slt");
}

#[test]
fn slt_p5_serial() {
    run_slt("tests/slt/p5_advanced/serial.slt");
}

#[test]
fn slt_p5_sequences() {
    run_slt("tests/slt/p5_advanced/sequences.slt");
}

#[test]
fn slt_p5_hot_update() {
    run_slt("tests/slt/p5_advanced/hot_update.slt");
}

#[test]
fn slt_p5_update_from() {
    run_slt("tests/slt/p5_advanced/update_from.slt");
}

#[test]
fn slt_p5_delete_using() {
    run_slt("tests/slt/p5_advanced/delete_using.slt");
}

#[test]
fn slt_p5_merge() {
    run_slt("tests/slt/p5_advanced/merge.slt");
}

#[test]
fn slt_p5_analyze() {
    run_slt("tests/slt/p5_advanced/analyze.slt");
}

#[test]
fn slt_p10_temporal() {
    run_slt("tests/slt/p10_types/temporal.slt");
}

#[test]
fn slt_p10_uuid() {
    run_slt("tests/slt/p10_types/uuid.slt");
}

#[test]
fn slt_p10_numeric() {
    run_slt("tests/slt/p10_types/numeric.slt");
}

#[test]
fn slt_p10_int_overflow() {
    run_slt("tests/slt/p10_types/int_overflow.slt");
}

#[test]
fn slt_p10_bool_cast() {
    run_slt("tests/slt/p10_types/bool_cast.slt");
}

#[test]
fn slt_p10_json() {
    run_slt("tests/slt/p10_types/json.slt");
}

#[test]
fn slt_p10_array() {
    run_slt("tests/slt/p10_types/array.slt");
}

#[test]
fn slt_p10_interval() {
    run_slt("tests/slt/p10_types/interval.slt");
}

#[test]
fn slt_p11_vector() {
    run_slt("tests/slt/p11_vector/vector.slt");
}

#[test]
fn slt_p11_hybrid_rrf() {
    run_slt("tests/slt/p11_vector/hybrid_rrf.slt");
}

#[test]
fn slt_p11_graphrag() {
    run_slt("tests/slt/p11_vector/graphrag.slt");
}

#[test]
fn slt_p5_encryption() {
    run_slt("tests/slt/p5_advanced/encryption.slt");
}

#[test]
fn slt_p13_string_concat() {
    run_slt("tests/slt/p13_functions/string_concat.slt");
}

#[test]
fn slt_p8_cte_and_subquery() {
    run_slt("tests/slt/p8_cte/cte_and_subquery.slt");
}

#[test]
fn slt_p8_data_modifying_cte() {
    run_slt("tests/slt/p8_cte/data_modifying_cte.slt");
}

#[test]
fn slt_p10_array_ops() {
    run_slt("tests/slt/p10_types/array_ops.slt");
}

#[test]
fn slt_p13_json_set_returning() {
    run_slt("tests/slt/p13_functions/json_set_returning.slt");
}

#[test]
fn slt_p14_tpch_subset() {
    run_slt("tests/slt/p14_tpch/tpch_subset.slt");
}

#[test]
fn slt_p6_set_ops() {
    run_slt("tests/slt/p6_setops/set_ops.slt");
}

#[test]
fn slt_p9_window() {
    run_slt("tests/slt/p9_window/window.slt");
}

#[test]
fn slt_p9_window_frames_and_value_functions() {
    run_slt("tests/slt/p9_window/frames_and_value_functions.slt");
}

#[test]
fn slt_p11_materialized_view() {
    run_slt("tests/slt/p11_views/materialized.slt");
}

#[test]
fn slt_p13_case_expr() {
    run_slt("tests/slt/p13_functions/case_expr.slt");
}

#[test]
fn slt_p13_scalar() {
    run_slt("tests/slt/p13_functions/scalar.slt");
}

#[test]
fn slt_p13_fts() {
    run_slt("tests/slt/p13_functions/fts.slt");
}

#[test]
fn slt_p11_plain_view() {
    run_slt("tests/slt/p11_views/plain.slt");
}

#[test]
fn slt_p12_transactions() {
    run_slt("tests/slt/p12_txn/transactions.slt");
}

#[test]
fn slt_p10_timetz() {
    run_slt("tests/slt/p10_types/timetz.slt");
}

#[test]
fn slt_p7_subquery() {
    run_slt("tests/slt/p7_subquery/subquery.slt");
}

#[test]
fn slt_p7_order_by_subquery() {
    run_slt("tests/slt/p7_subquery/order_by_subquery.slt");
}

#[test]
fn slt_p6_setops_null_dup() {
    run_slt("tests/slt/p6_setops/setops_null_dup.slt");
}

#[test]
fn slt_p12_savepoints_nested() {
    run_slt("tests/slt/p12_txn/savepoints_nested.slt");
}

#[test]
fn slt_p12_ddl_in_transaction() {
    run_slt("tests/slt/p12_txn/ddl_in_transaction.slt");
}

#[test]
fn slt_p12_for_update() {
    run_slt("tests/slt/p12_txn/for_update.slt");
}

#[test]
fn slt_p12_isolation_levels() {
    run_slt("tests/slt/p12_txn/isolation_levels.slt");
}

#[test]
fn slt_p12_savepoint_errors() {
    run_slt("tests/slt/p12_txn/savepoint_errors.slt");
}

#[test]
fn slt_p13_math() {
    run_slt("tests/slt/p13_functions/math.slt");
}

#[test]
fn slt_p13_date_time() {
    run_slt("tests/slt/p13_functions/date_time.slt");
}

#[test]
fn slt_p13_string_extra() {
    run_slt("tests/slt/p13_functions/string_extra.slt");
}

#[test]
fn slt_p13_pattern_match() {
    run_slt("tests/slt/p13_functions/pattern_match.slt");
}

#[test]
fn slt_p13_uuid_gen() {
    run_slt("tests/slt/p13_functions/uuid_gen.slt");
}

#[test]
fn slt_p13_system() {
    run_slt("tests/slt/p13_functions/system.slt");
}
