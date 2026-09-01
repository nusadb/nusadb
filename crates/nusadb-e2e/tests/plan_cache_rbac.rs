//! Plan caching for role-based sessions: a non-superuser's read plan is cached (it used to be
//! marked uncacheable, so every statement re-analyzed on top of the RBAC overhead), a `REVOKE`
//! invalidates it (the plan is not served to a role that may no longer run it), and an ordinary
//! data write does not invalidate it. A cache hit re-verifies the plan's privilege decisions
//! against the live catalog, so the re-check reads the same snapshot a re-analysis would — a
//! committed `REVOKE` is always seen, with no cross-transaction staleness. A row-level-security
//! SELECT bakes its policy predicate into the plan and so is not cached at all.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::{ObjectKind, Privilege};
use nusadb_sql::{
    Catalog, ExecutionResult, PlanCache, Session, analyze, execute, parse, plan, plan_cached,
};

/// A catalog that answers privilege questions from the real RBAC catalogs for a fixed user — the
/// same shape the wire server builds per connection.
struct RbacCatalog<'a> {
    engine: &'a BtreeEngine,
    user: &'a str,
}

impl RbacCatalog<'_> {
    fn with_txn<T>(
        &self,
        f: impl FnOnce(nusadb_core::TxnId) -> Result<T, nusadb_sql::Error>,
    ) -> Result<T, nusadb_sql::Error> {
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let out = f(txn);
        let _ = self.engine.rollback(txn);
        out
    }
}

impl Catalog for RbacCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn is_superuser(&self) -> bool {
        self.with_txn(|txn| nusadb_sql::rbac::principal(self.engine, txn, self.user))
            .is_ok_and(|p| p.superuser)
    }
    fn current_user(&self) -> String {
        self.user.to_owned()
    }
    fn has_privilege(
        &self,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::has_privilege(self.engine, txn, self.user, kind, object, privilege)
        })
    }
    fn has_column_privilege(
        &self,
        object: &str,
        column: &str,
        privilege: Privilege,
    ) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::has_column_privilege(
                self.engine,
                txn,
                self.user,
                object,
                column,
                privilege,
            )
        })
    }
    fn has_any_column_privilege(
        &self,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::has_any_column_privilege(
                self.engine,
                txn,
                self.user,
                object,
                privilege,
            )
        })
    }
    fn owns_object(&self, kind: ObjectKind, object: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::owns_object(self.engine, txn, self.user, kind, object)
        })
    }
    fn rls_enabled(&self, schema: &str, name: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::rls_table_enabled(self.engine, txn, schema, name))
    }
    fn lookup_policies(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Vec<nusadb_sql::PolicyDef>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_policies_for(self.engine, txn, schema, name))
    }
}

/// Run a statement as `nusadb-root` (a permissive superuser catalog for setup).
fn root(engine: &BtreeEngine, sql: &str) {
    struct Root<'a>(&'a BtreeEngine);
    impl Catalog for Root<'_> {
        fn lookup_table(&self, n: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
            self.0.lookup_table(n).map_err(Into::into)
        }
        fn is_superuser(&self) -> bool {
            true
        }
        fn current_user(&self) -> String {
            "nusadb-root".to_owned()
        }
    }
    let stmt = parse(sql).unwrap();
    let logical = analyze(stmt, &Root(engine)).expect(sql);
    let mut s = Session::new(engine);
    s.set_current_user("nusadb-root");
    s.execute(plan(logical)).expect(sql);
}

/// Plan `sql` as `user` through the cache, execute it, and return the result.
fn cached_as(
    engine: &BtreeEngine,
    cache: &mut PlanCache,
    user: &str,
    sql: &str,
) -> Result<ExecutionResult, String> {
    let stmt = parse(sql).map_err(|e| e.to_string())?;
    let physical = plan_cached(cache, sql, stmt, &RbacCatalog { engine, user }, engine)
        .map_err(|e| e.to_string())?;
    execute(physical, engine).map_err(|e| e.to_string())
}

#[test]
fn a_role_plan_is_cached_reused_and_invalidated_by_a_revoke_but_not_by_a_write() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE orders (id INT NOT NULL, total INT)");
    root(&engine, "INSERT INTO orders VALUES (1, 100), (2, 200)");
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "GRANT SELECT ON orders TO app");

    let mut cache = PlanCache::new();
    let sql = "SELECT id FROM orders ORDER BY id";

    // First run as the non-superuser: a miss that is now CACHED (previously the privilege check
    // marked such plans uncacheable, so they re-analyzed on every statement).
    cached_as(&engine, &mut cache, "app", sql).expect("app may read");
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 0);

    // Second run: served from the cache — the fix's whole point.
    cached_as(&engine, &mut cache, "app", sql).expect("app may still read");
    assert_eq!(
        cache.hits(),
        1,
        "a role's SELECT plan must be cached, not re-analyzed each statement"
    );

    // An ordinary data write must NOT invalidate the plan: the re-check re-verifies the privilege
    // (still granted), and the schema version is unchanged. The next read is still a cache hit.
    root(&engine, "INSERT INTO orders VALUES (3, 300)");
    cached_as(&engine, &mut cache, "app", sql).expect("still readable after a write");
    assert_eq!(
        cache.hits(),
        2,
        "a plain INSERT must not evict the cached plan"
    );

    // A REVOKE flips the recorded privilege decision, so the reuse re-check fails, the entry is
    // discarded, and the re-analysis denies the read — the plan a role may no longer run is never
    // served from cache.
    root(&engine, "REVOKE SELECT ON orders FROM app");
    let denied = cached_as(&engine, &mut cache, "app", sql);
    assert!(
        denied
            .as_ref()
            .is_err_and(|m| m.contains("permission denied")),
        "after REVOKE the cached plan must be invalidated and the re-analysis must deny: {denied:?}"
    );
}

/// A SELECT on a row-level-security table is never cached: its policy predicate is baked into the
/// plan and a `CREATE`/`DROP`/`ALTER POLICY` changes it without moving any table's schema version,
/// so a cached plan could return rows a changed policy must now hide. The plan re-analyzes every
/// time (correct, if not cached), so there is no stale-plan vector for a policy change.
#[test]
fn a_row_level_security_select_is_not_cached() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE doc (id INT NOT NULL, owner TEXT)");
    root(&engine, "INSERT INTO doc VALUES (1, 'app'), (2, 'other')");
    root(&engine, "ALTER TABLE doc ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own ON doc FOR SELECT TO app USING (owner = CURRENT_USER)",
    );
    root(&engine, "GRANT SELECT ON doc TO app");

    let mut cache = PlanCache::new();
    let sql = "SELECT id FROM doc";
    // Every run is a miss — the RLS plan is not cached, so a later policy change can never serve a
    // stale predicate.
    cached_as(&engine, &mut cache, "app", sql).expect("app may read its rows");
    cached_as(&engine, &mut cache, "app", sql).expect("app may read its rows");
    cached_as(&engine, &mut cache, "app", sql).expect("app may read its rows");
    assert_eq!(
        cache.hits(),
        0,
        "an RLS SELECT must never be served from cache"
    );
    assert_eq!(cache.misses(), 3);
}
