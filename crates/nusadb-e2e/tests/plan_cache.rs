//! Plan cache against the real `BtreeEngine`: a repeated read reuses its planned form,
//! a schema change invalidates it (never serving a stale plan), and plans with an unversioned
//! dependency (an index that could be dropped) are not cached.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, PlanCache, execute, parse, plan_cached};

/// Adapts the engine's schema lookup to the analyzer's `Catalog` port, exposing user-maintained
/// secondary indexes (so the plan cache's index guard sees them). Mirrors the `end_to_end` adapter.
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        let backing: std::collections::HashSet<_> = self
            .0
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.0.list_indexes(schema.id)? {
            if self
                .0
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
}

/// Run a DDL/DML statement straight through (no caching).
fn run(engine: &BtreeEngine, sql: &str) {
    let stmt = parse(sql).expect("parse");
    let logical = nusadb_sql::analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(nusadb_sql::plan(logical), engine).expect("execute");
}

/// Run a query through the plan cache and return its rows.
fn run_cached(engine: &BtreeEngine, cache: &mut PlanCache, sql: &str) -> Vec<Vec<Value>> {
    let stmt = parse(sql).expect("parse");
    let physical =
        plan_cached(cache, sql, stmt, &EngineCatalog(engine), engine).expect("plan_cached");
    match execute(physical, engine).expect("execute") {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected SELECT rows, got {other:?}"),
    }
}

#[test]
fn repeated_select_is_served_from_the_plan_cache() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    let mut cache = PlanCache::new();

    let sql = "SELECT a, b FROM t WHERE a > 1";
    let first = run_cached(&engine, &mut cache, sql);
    assert_eq!(cache.misses(), 1, "first run plans (miss)");
    assert_eq!(cache.hits(), 0);

    let second = run_cached(&engine, &mut cache, sql);
    assert_eq!(cache.hits(), 1, "second run reuses the cached plan");
    assert_eq!(cache.misses(), 1);
    assert_eq!(first, second, "cached plan yields identical rows");
    assert_eq!(first.len(), 2);
}

#[test]
fn schema_change_invalidates_the_cached_plan() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10)");
    let mut cache = PlanCache::new();

    // Cache a `SELECT *` plan: its projection is the two columns that exist now.
    let sql = "SELECT * FROM t";
    let before = run_cached(&engine, &mut cache, sql);
    assert_eq!(before[0].len(), 2, "two columns before the ALTER");
    assert_eq!(cache.misses(), 1);

    // Add a column: this bumps the table's schema version, so the cached `SELECT *` plan (still
    // 2-wide) must be discarded rather than served stale.
    run(&engine, "ALTER TABLE t ADD COLUMN c INT");
    let after = run_cached(&engine, &mut cache, sql);
    assert_eq!(cache.hits(), 0, "the stale plan was not reused");
    assert_eq!(
        cache.misses(),
        2,
        "the query was re-planned after the schema change"
    );
    assert_eq!(
        after[0].len(),
        3,
        "the re-planned SELECT * reflects the new column (no stale plan served)"
    );
}

#[test]
fn dropping_a_table_invalidates_the_cached_plan() {
    // Regression for the stale-read bug: once a table is dropped, its cached SELECT plan must
    // not be served. The engine reports a committed-dropped table's schema version as `None`, so the
    // fingerprint stops matching and the query is re-planned — which now fails to resolve the table —
    // instead of scanning the MVCC tombstone and returning 0 rows as if the table were empty.
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    let mut cache = PlanCache::new();

    let sql = "SELECT a, b FROM t";
    let before = run_cached(&engine, &mut cache, sql);
    assert_eq!(before.len(), 2, "two rows before the drop");
    assert_eq!(cache.misses(), 1);

    run(&engine, "DROP TABLE t");

    // Re-planning the same SQL must now fail to resolve `t` rather than serve the stale plan.
    let stmt = parse(sql).expect("parse");
    let replanned = plan_cached(&mut cache, sql, stmt, &EngineCatalog(&engine), &engine);
    assert!(
        replanned.is_err(),
        "a cached SELECT over a dropped table must be re-planned and fail, not served stale"
    );
    assert_eq!(
        cache.hits(),
        0,
        "the dropped table's plan was never a cache hit"
    );
}

#[test]
fn plans_over_an_indexed_table_are_not_cached() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE t (a INT NOT NULL, b INT)");
    run(&engine, "CREATE INDEX t_a ON t (a)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    let mut cache = PlanCache::new();

    // An index could be dropped without bumping the schema version, so a plan that can see one is not
    // cached: both runs are misses.
    let sql = "SELECT a, b FROM t WHERE a = 1";
    let _ = run_cached(&engine, &mut cache, sql);
    let _ = run_cached(&engine, &mut cache, sql);
    assert_eq!(cache.hits(), 0, "indexed-table plans are not cached");
    assert_eq!(cache.misses(), 2);
}

/// A catalog with a configurable `search_path` over the real engine, for the
/// regression: both qualified and unqualified resolution go to the real `(schema, name)` engine
/// lookup, and the search path drives where a bare name binds.
struct PathCatalog<'a> {
    engine: &'a dyn StorageEngine,
    search_path: Vec<String>,
}

impl Catalog for PathCatalog<'_> {
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

    fn search_path(&self) -> Vec<String> {
        self.search_path.clone()
    }
}

fn path_catalog<'a>(engine: &'a BtreeEngine, search_path: &[&str]) -> PathCatalog<'a> {
    PathCatalog {
        engine,
        search_path: search_path.iter().map(|s| (*s).to_owned()).collect(),
    }
}

fn run_path(engine: &BtreeEngine, search_path: &[&str], sql: &str) {
    let stmt = parse(sql).expect("parse");
    let logical = nusadb_sql::analyze(stmt, &path_catalog(engine, search_path)).expect("analyze");
    execute(nusadb_sql::plan(logical), engine).expect("execute");
}

fn run_cached_path(
    engine: &BtreeEngine,
    cache: &mut PlanCache,
    search_path: &[&str],
    sql: &str,
) -> Vec<Vec<Value>> {
    let stmt = parse(sql).expect("parse");
    let physical = plan_cached(cache, sql, stmt, &path_catalog(engine, search_path), engine)
        .expect("plan_cached");
    match execute(physical, engine).expect("execute") {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected SELECT rows, got {other:?}"),
    }
}

#[test]
fn search_path_is_part_of_the_plan_cache_key() {
    // The same unqualified SQL must not be served a cached plan that bound the
    // name to a different schema after `search_path` changed which schema the bare name resolves to.
    let engine = BtreeEngine::new();
    run_path(&engine, &["public"], "CREATE SCHEMA spo");
    run_path(
        &engine,
        &["public"],
        "CREATE TABLE spo.spt (v INT NOT NULL)",
    );
    run_path(&engine, &["public"], "INSERT INTO spo.spt VALUES (100)");
    run_path(
        &engine,
        &["public"],
        "CREATE TABLE public.spt (v INT NOT NULL)",
    );
    run_path(&engine, &["public"], "INSERT INTO public.spt VALUES (200)");
    let mut cache = PlanCache::new();

    // Under the default path a bare name resolves to public (200) and is cached.
    let sql = "SELECT v FROM spt";
    assert_eq!(
        run_cached_path(&engine, &mut cache, &["public"], sql),
        vec![vec![Value::Int(200)]]
    );

    // With `spo` first in the path the SAME SQL must resolve to spo (100) — a fresh key, not the
    // stale public plan. Before the fix this returned 200 (the cached public plan).
    assert_eq!(
        run_cached_path(&engine, &mut cache, &["spo", "public"], sql),
        vec![vec![Value::Int(100)]],
        "search_path order is honored through the plan cache"
    );

    // The original path still resolves public from its own cache entry.
    assert_eq!(
        run_cached_path(&engine, &mut cache, &["public"], sql),
        vec![vec![Value::Int(200)]]
    );
}
