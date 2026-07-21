//! Per-connection plan cache.
//!
//! Re-planning every statement re-parses, re-analyzes, and re-lowers it from scratch. A client that
//! issues the same query repeatedly (dashboards, ORMs, prepared-statement-style loops) pays that cost
//! each time even though the plan is identical. [`plan_cached`] caches the planned form of a read
//! statement, keyed by its SQL text, and reuses it until the schema it was planned against changes.
//!
//! ## Correctness
//!
//! A stale plan is a wrong answer, so caching is gated by three rules, all enforced here:
//!
//! 1. **Read-only statements only.** Only `SELECT` / set-operations are cached; DDL and DML are
//!    cheap to plan and must never be served from a snapshot of an earlier schema.
//! 2. **No unversioned dependency.** A plan can bake things that change *without* bumping any table's
//!    schema version: an inlined view/UDF body, a row-level-security policy predicate, or an index
//!    choice (`CREATE`/`DROP INDEX` is unversioned). The recording catalog flags all of these during
//!    analysis, and such plans are not cached at all (conservative, always safe).
//! 3. **Schema-version fingerprint.** Each entry records the schema version of *every* base table the
//!    analyzer resolved (captured by a recording catalog, which is complete by construction — the
//!    analyzer cannot resolve a table without looking it up). The entry is reused only while every one
//!    of those versions is unchanged, so any `ALTER`/`DROP`/recreate of a referenced table discards it.
//!
//! The cache must be **per connection** (hence per user): a plan bakes row-level-security predicates
//! for the analyzing user, so the wire server holds one `PlanCache` per connection and never shares it
//! across users.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use nusadb_core::{StorageEngine, TableId, TableSchema, TableStats};

use crate::analyzer::{Catalog, FunctionDef, IndexInfo, PolicyDef};
use crate::ast;
use crate::error::Error;
use crate::planner::PhysicalPlan;

/// Most entries the cache holds before it is cleared (a memory bound). Clearing on overflow stays
/// correct — a later query simply re-analyzes and repopulates it; a true LRU is a follow-up.
const MAX_PLAN_CACHE_ENTRIES: usize = 256;

/// A planned read statement plus the schema fingerprint it was planned under.
struct CachedPlan {
    /// `(table, schema version)` for every base table the plan references, sorted. The plan is reused
    /// only while each table still reports this exact version.
    fingerprint: Vec<(TableId, u32)>,
    plan: PhysicalPlan,
}

/// A per-(connection, user) cache of planned read statements. See the module docs for the
/// correctness rules.
#[derive(Default)]
pub struct PlanCache {
    entries: HashMap<String, CachedPlan>,
    hits: u64,
    misses: u64,
}

impl std::fmt::Debug for PlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanCache")
            .field("entries", &self.entries.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl PlanCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many cacheable reads were served from the cache (for tests/metrics).
    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// How many cacheable reads had to be planned (cache miss or stale entry).
    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }
}

/// Wraps a catalog to record which base tables a statement resolves and whether it inlines a view or
/// SQL function. The recorded set is complete because the analyzer resolves every referenced table
/// through [`Catalog::lookup_table`]; views and functions are flagged so their plans are never cached.
struct RecordingCatalog<'a> {
    inner: &'a dyn Catalog,
    tables: RefCell<Vec<TableId>>,
    saw_view: Cell<bool>,
    saw_function: Cell<bool>,
    /// Set if the plan may bake an index choice or a row-level-security predicate. Index DDL
    /// (`CREATE`/`DROP INDEX`) and policy changes do **not** bump a table's schema version, so a plan
    /// that depends on either cannot be safely invalidated by the fingerprint — such plans are not
    /// cached at all (conservative: it also skips index-bearing tables whose plan chose a seq scan).
    saw_unversioned_dep: Cell<bool>,
}

impl<'a> RecordingCatalog<'a> {
    fn new(inner: &'a dyn Catalog) -> Self {
        Self {
            inner,
            tables: RefCell::new(Vec::new()),
            saw_view: Cell::new(false),
            saw_function: Cell::new(false),
            saw_unversioned_dep: Cell::new(false),
        }
    }
}

impl Catalog for RecordingCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        let schema = self.inner.lookup_table(name)?;
        if let Some(s) = &schema {
            self.tables.borrow_mut().push(s.id);
        }
        Ok(schema)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        let resolved = self.inner.lookup_table_in(schema, name)?;
        if let Some(s) = &resolved {
            self.tables.borrow_mut().push(s.id);
        }
        Ok(resolved)
    }

    fn search_path(&self) -> Vec<String> {
        // Forward the session search path: an unqualified name must resolve under the real
        // session path during cached-plan analysis, not the default `public`, or the wire's read
        // path would bind bare names to the wrong schema.
        self.inner.search_path()
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        let view = self.inner.lookup_view(name)?;
        if view.is_some() {
            self.saw_view.set(true);
        }
        Ok(view)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<FunctionDef>, Error> {
        let func = self.inner.lookup_function(name)?;
        if func.is_some() {
            self.saw_function.set(true);
        }
        Ok(func)
    }

    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let indexes = self.inner.list_indexes(table)?;
        // A plan that can see an index may lower to an `IndexScan`; index DDL is unversioned, so do
        // not cache it. (Conservative — fires even when the planner ultimately picks a seq scan.)
        if !indexes.is_empty() {
            self.saw_unversioned_dep.set(true);
        }
        Ok(indexes)
    }
    fn rls_enabled(&self, name: &str) -> Result<bool, Error> {
        let enabled = self.inner.rls_enabled(name)?;
        // An RLS-enabled table bakes a policy predicate into the plan; policy changes are unversioned,
        // so the plan cannot be safely reused — do not cache it.
        if enabled {
            self.saw_unversioned_dep.set(true);
        }
        Ok(enabled)
    }

    // Everything else delegates unchanged.
    fn table_stats(&self, table: &str) -> Result<Option<TableStats>, Error> {
        self.inner.table_stats(table)
    }
    fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
        self.inner.approx_row_count(table)
    }
    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, Error> {
        self.inner.lookup_view_columns(name)
    }
    fn is_superuser(&self) -> bool {
        self.inner.is_superuser()
    }
    fn current_user(&self) -> String {
        self.inner.current_user()
    }
    fn lookup_policies(&self, name: &str) -> Result<Vec<PolicyDef>, Error> {
        self.inner.lookup_policies(name)
    }
}

/// Whether `stmt` is a pure read whose plan is safe to cache. DDL/DML and session-control statements
/// are excluded — they are cheap to plan and must never run against a stale schema snapshot.
const fn is_cacheable(stmt: &ast::Statement) -> bool {
    matches!(
        stmt,
        ast::Statement::Select(_) | ast::Statement::SetOperation(_)
    )
}

/// Whether every table in `fingerprint` still reports the recorded schema version. A table whose
/// version is unknown (dropped, or untracked) counts as changed, so the plan is rebuilt.
fn fingerprint_unchanged(engine: &dyn StorageEngine, fingerprint: &[(TableId, u32)]) -> bool {
    fingerprint
        .iter()
        .all(|&(id, version)| matches!(engine.current_schema_version(id), Ok(Some(cur)) if cur == version))
}

/// A fingerprint from the recorded tables, or `None` if any table has no tracked version — then the
/// plan cannot be safely invalidated and must not be cached (e.g. an engine that does not version
/// schemas, which keeps the cache disabled rather than risking a stale plan).
fn build_fingerprint(engine: &dyn StorageEngine, ids: &[TableId]) -> Option<Vec<(TableId, u32)>> {
    let mut fingerprint = Vec::with_capacity(ids.len());
    for &id in ids {
        match engine.current_schema_version(id) {
            Ok(Some(version)) => fingerprint.push((id, version)),
            _ => return None,
        }
    }
    fingerprint.sort_unstable();
    fingerprint.dedup();
    Some(fingerprint)
}

/// Analyze and lower `stmt`, reusing a cached plan when the same `sql` was planned before and none of
/// the tables it references have changed schema.
///
/// `sql` is the cache key; `catalog` analyzes the statement and **must be scoped to the same user**
/// across calls on one `cache` (the wire server holds one cache per connection). `engine` supplies the
/// per-table schema versions used to validate and invalidate entries.
///
/// Non-read statements, and reads that inline a view or SQL function, bypass the cache entirely (they
/// are always planned fresh), so caching can never serve a stale or user-crossed plan.
///
/// # Errors
/// Propagates any analysis or planning error.
pub fn plan_cached(
    cache: &mut PlanCache,
    sql: &str,
    stmt: ast::Statement,
    catalog: &dyn Catalog,
    engine: &dyn StorageEngine,
) -> Result<PhysicalPlan, Error> {
    if !is_cacheable(&stmt) {
        return Ok(crate::plan(crate::analyze(stmt, catalog)?));
    }
    // The session search path is part of a plan's identity: the same SQL text
    // resolves an unqualified name to a different table under a different `search_path`, so it must
    // key the cache — otherwise a plan resolved under one path is served stale after `SET
    // search_path` changes which schema a bare name binds to.
    let key = cache_key(catalog, sql);
    if let Some(entry) = cache.entries.get(&key) {
        if fingerprint_unchanged(engine, &entry.fingerprint) {
            cache.hits += 1;
            return Ok(entry.plan.clone());
        }
        // Stale: a referenced table changed schema since this was planned.
        cache.entries.remove(&key);
    }
    cache.misses += 1;
    let recorder = RecordingCatalog::new(catalog);
    let physical = crate::plan(crate::analyze(stmt, &recorder)?);
    let cacheable = !recorder.saw_view.get()
        && !recorder.saw_function.get()
        && !recorder.saw_unversioned_dep.get();
    let ids = recorder.tables.into_inner();
    if cacheable
        && !ids.is_empty()
        && let Some(fingerprint) = build_fingerprint(engine, &ids)
    {
        if cache.entries.len() >= MAX_PLAN_CACHE_ENTRIES {
            cache.entries.clear();
        }
        cache.entries.insert(
            key,
            CachedPlan {
                fingerprint,
                plan: physical.clone(),
            },
        );
    }
    Ok(physical)
}

/// The plan-cache key: the session `search_path` followed by the SQL text. Two
/// otherwise-identical queries planned under different search paths get distinct entries, so an
/// unqualified name is never served a plan that bound it to a different schema. A `\u{1}` separates
/// path entries and a `\u{0}` separates the path from the SQL — neither can appear in a schema name
/// or in normal SQL, so the key is unambiguous.
fn cache_key(catalog: &dyn Catalog, sql: &str) -> String {
    let mut key = String::new();
    for schema in catalog.search_path() {
        key.push_str(&schema);
        key.push('\u{1}');
    }
    key.push('\u{0}');
    key.push_str(sql);
    key
}
