//! L3 — SQL engine.
//!
//! Pipeline: [`parser`] → [`analyzer`] → [`planner`] → [`executor`]. Executor processes
//! data in 1024-row vectorized batches; SIMD AVX2 used in hot paths (filter, project,
//! hash). The parser is a thin wrapper around `sqlparser-rs`; downstream code uses
//! our internal [`ast`] types, never `sqlparser` types directly (anti-corruption layer).
//!
//! # Stage
//!
//! the SQL engine layer (parser, analyzer, planner, executor).

#![warn(missing_docs)]

pub mod analyzer;
pub mod ast;
pub mod batch;
pub mod cancel;
pub mod copy;
pub mod display;
pub mod error;
pub mod executor;
pub mod fts;
pub mod hnsw;
pub mod interval;
pub mod json;
pub mod numeric;
pub mod params;
pub mod parser;
pub mod plan_cache;
pub mod planner;
pub mod temporal;
pub mod udf;
pub mod vector;
pub mod vectorized;

pub use analyzer::{
    Catalog, FunctionDef, IndexInfo, PolicyDef, SYNTHETIC_TYPE_CHECK_PREFIX, SYSTEM_TABLE_PREFIX,
    analyze,
};
pub use batch::{
    Array, ArrayRef, BinaryArray, BooleanArray, DateArray, DecimalArray, Field, Float64Array,
    Int64Array, IntervalArray, JsonArray, ListArray, PrimitiveArray, PrimitiveType, RecordBatch,
    RecordBatchScan, Schema, StringArray, TemporalArray, TemporalKind, TimeArray, TimestampArray,
    TimestampTzArray, Uuid, UuidArray, schema_from_columns,
};
pub use error::Error;
pub use executor::{
    ExecutionResult, Row, RowSink, Session, SpillConfig, StreamOutcome, auto_analyze_stale_tables,
    catalog_approx_row_count, catalog_list_indexes, catalog_table_stats, copy_from, copy_to,
    describe_column_types, describe_columns, execute, execute_in_txn, execute_in_txn_as,
    execute_in_txn_as_streaming, execute_in_txn_as_streaming_with_settings,
    execute_in_txn_as_with_settings, lookup_function_definition, lookup_policies_for,
    lookup_view_columns, lookup_view_definition, maintenance_work_mem, parse_work_mem,
    rls_table_enabled, set_maintenance_work_mem, set_spill_config, set_work_mem,
    show_session_variable, work_mem,
};

/// The bootstrap database superuser, which bypasses row-level security.
///
/// NusaDB does not yet model per-role `SUPERUSER` attributes, so a session is a superuser exactly
/// when it runs as this user. The wire server reports it from the connection's authenticated user.
pub const BOOTSTRAP_SUPERUSER: &str = "nusa-root";

/// The reserved settings-snapshot key the wire stamps with the connection's database.
///
/// Read by the session so `CURRENT_DATABASE()` reflects which physical database the connection is in.
/// A client cannot make `current_database()` lie: the wire re-stamps this key with the real
/// connection database into a fresh per-statement snapshot *after* cloning the GUC store, so any
/// value a `SET` might have placed there is overwritten before the session reads it. The leading NUL
/// also keeps the key out of normal identifier space.
pub const CONNECTION_DATABASE_SETTING: &str = "\u{0}conn_database";

/// The ordered list of schemas an unqualified name resolves through, derived from a `search_path`
/// GUC value — the heart of search-path name resolution.
///
/// Splits the (possibly comma-separated) `search_path` into its entries, trimming quotes/whitespace
/// and dropping empties/duplicates while preserving order. `public` is always appended as the final
/// fallback if it is not already listed, so a bare name still resolves there. An unset or empty
/// `search_path` yields `[public]`.
#[must_use]
pub fn search_path_schemas(search_path: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(sp) = search_path {
        for part in sp.split(',') {
            let s = part.trim().trim_matches(['"', '\'']).trim();
            if !s.is_empty() && !out.iter().any(|x| x == s) {
                out.push(s.to_owned());
            }
        }
    }
    if !out.iter().any(|x| x == nusadb_core::PUBLIC_SCHEMA) {
        out.push(nusadb_core::PUBLIC_SCHEMA.to_owned());
    }
    out
}

/// The session's current schema derived from a `search_path` GUC value — the first entry of
/// the [`search_path_schemas`] list, i.e. where an unqualified name is created. `public` when unset.
#[must_use]
pub fn current_schema_for_search_path(search_path: Option<&str>) -> String {
    search_path_schemas(search_path)
        .into_iter()
        .next()
        .unwrap_or_else(|| nusadb_core::PUBLIC_SCHEMA.to_owned())
}
pub use params::{bind_parameters, parameter_count};
pub use parser::parse;
pub use plan_cache::{PlanCache, plan_cached};
pub use planner::{LogicalPlan, PhysicalOperator, PhysicalPlan, plan, plan_is_inline_point_get};
pub use vectorized::{Filter, Limit, Operator, Project, SeqScan, Sort};

/// Vectorized batch size — every operator processes exactly this many rows at once.
///
/// Matches AVX2 lane multiples (4 × i64 = 32 lanes; 8 × i32 = 128 lanes; etc.).
pub const BATCH_SIZE: usize = 1024;

#[cfg(test)]
mod search_path_tests {
    use super::{current_schema_for_search_path as cs, search_path_schemas as sp};

    #[test]
    fn current_schema_derives_from_search_path() {
        // Unset / empty fall back to the default namespace.
        assert_eq!(cs(None), "public");
        assert_eq!(cs(Some("")), "public");
        assert_eq!(cs(Some("   ")), "public");
        // A single schema, with or without quotes/whitespace.
        assert_eq!(cs(Some("app")), "app");
        assert_eq!(cs(Some("  app  ")), "app");
        assert_eq!(cs(Some("'app'")), "app");
        assert_eq!(cs(Some("\"app\"")), "app");
        // A list takes the first entry (NS3 current-schema model).
        assert_eq!(cs(Some("app, public")), "app");
        assert_eq!(cs(Some("reporting,app,public")), "reporting");
    }

    #[test]
    fn search_path_is_an_ordered_deduped_list_ending_in_public() {
        // Unset / empty → just public.
        assert_eq!(sp(None), vec!["public"]);
        assert_eq!(sp(Some("")), vec!["public"]);
        // A single schema gains public as the trailing fallback.
        assert_eq!(sp(Some("app")), vec!["app", "public"]);
        // An explicit list preserves order; public is not re-appended if already present.
        assert_eq!(sp(Some("a, b, public")), vec!["a", "b", "public"]);
        assert_eq!(sp(Some("public, a")), vec!["public", "a"]);
        // Whitespace/quotes trimmed; duplicates dropped preserving first occurrence.
        assert_eq!(sp(Some(" 'a' , a , b ")), vec!["a", "b", "public"]);
    }
}
