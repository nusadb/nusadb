//! End-to-end tests for access control: roles, privileges, ownership, and the statements that
//! manage them.
//!
//! These drive the real enforcement path. The catalog below answers privilege questions from the
//! actual role and privilege catalogs, exactly as the wire server's adapter does — a permissive
//! stand-in would make every assertion here vacuous, since the `Catalog` trait's defaults grant
//! everything so that embeddings without a role catalog behave as they did before.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests live outside #[cfg(test)], so the test-only lint relaxations in \
              clippy.toml do not reach them"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::{ObjectKind, Privilege, Value};
use nusadb_sql::{Catalog, ExecutionResult, Session, analyze, parse, plan};

/// A catalog that resolves privileges against the real catalogs, one snapshot per question — the
/// same shape as the per-statement catalog the server builds for a connection.
struct RbacCatalog<'a> {
    engine: &'a BtreeEngine,
    user: &'a str,
}

impl RbacCatalog<'_> {
    /// Run `f` against a fresh read snapshot, rolling it back afterwards.
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

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine
            .lookup_table_in(schema, name)
            .map_err(Into::into)
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

    fn may_grant_object(
        &self,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::may_grant(self.engine, txn, self.user, kind, object, privilege)
        })
    }

    fn may_create_role(&self) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::rbac::may_create_role(self.engine, txn, self.user))
    }

    fn may_administer_role(&self, role: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::may_administer_role(self.engine, txn, self.user, role)
        })
    }

    fn may_assume_role(&self, role: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            Ok(
                nusadb_sql::rbac::principal(self.engine, txn, self.user)?.superuser
                    || nusadb_sql::rbac::effective_roles(self.engine, txn, self.user)?
                        .contains(role),
            )
        })
    }

    fn role_exists(&self, name: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            Ok(name == nusadb_sql::BOOTSTRAP_SUPERUSER
                || nusadb_sql::rbac::lookup_role(self.engine, txn, name)?.is_some())
        })
    }

    fn owns_object(&self, kind: ObjectKind, object: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| {
            nusadb_sql::rbac::owns_object(self.engine, txn, self.user, kind, object)
        })
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_view_definition(self.engine, txn, name))
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_view_columns(self.engine, txn, name))
    }
}

/// The bootstrap superuser, used for every setup statement.
const ROOT: &str = "nusadb-root";

/// Run `sql` as `user`, returning the result or the error text.
fn as_role(engine: &BtreeEngine, user: &str, sql: &str) -> Result<ExecutionResult, String> {
    let stmt = parse(sql).map_err(|e| e.to_string())?;
    let logical = analyze(stmt, &RbacCatalog { engine, user }).map_err(|e| e.to_string())?;
    let mut session = Session::new(engine);
    session.set_current_user(user);
    session.execute(plan(logical)).map_err(|e| e.to_string())
}

/// Run `sql` as `user` and expect success.
fn ok_as(engine: &BtreeEngine, user: &str, sql: &str) -> ExecutionResult {
    as_role(engine, user, sql).unwrap_or_else(|e| panic!("`{sql}` as {user} should succeed: {e}"))
}

/// Run a setup statement as the superuser.
fn root(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    ok_as(engine, ROOT, sql)
}

/// Run `sql` as `user` and expect a permission denial.
fn denied_as(engine: &BtreeEngine, user: &str, sql: &str) {
    match as_role(engine, user, sql) {
        Err(msg) => assert!(
            msg.contains("permission denied"),
            "`{sql}` as {user} should be denied, got: {msg}"
        ),
        Ok(other) => panic!("`{sql}` as {user} should have been denied, got {other:?}"),
    }
}

/// The rows of a `SELECT` result.
fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected SELECT rows, got {other:?}"),
    }
}

/// An engine with one superuser-owned table and a bare `app` role that holds nothing.
fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE orders (id INT, total INT)");
    root(&engine, "INSERT INTO orders VALUES (1, 100), (2, 200)");
    root(&engine, "CREATE ROLE app LOGIN");
    engine
}

#[test]
fn denies_until_granted_and_denies_again_after_revoke() {
    let engine = fixture();

    // This is the finding being closed: a connected role used to reach everything.
    denied_as(&engine, "app", "SELECT id FROM orders");
    denied_as(&engine, "app", "INSERT INTO orders VALUES (3, 300)");

    root(&engine, "GRANT SELECT ON orders TO app");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders ORDER BY id")),
        vec![vec![Value::Int(1)], vec![Value::Int(2)]]
    );
    // One privilege must not imply the others.
    denied_as(&engine, "app", "INSERT INTO orders VALUES (3, 300)");

    root(&engine, "REVOKE SELECT ON orders FROM app");
    denied_as(&engine, "app", "SELECT id FROM orders");
}

#[test]
fn grant_all_covers_every_table_privilege() {
    let engine = fixture();
    root(&engine, "GRANT ALL PRIVILEGES ON orders TO app");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
    ok_as(&engine, "app", "INSERT INTO orders VALUES (3, 300)");
    ok_as(&engine, "app", "DELETE FROM orders WHERE id = 3");
}

#[test]
fn privileges_flow_through_role_membership() {
    let engine = fixture();
    root(&engine, "CREATE ROLE reader");
    root(&engine, "GRANT SELECT ON orders TO reader");

    denied_as(&engine, "app", "SELECT id FROM orders");
    root(&engine, "GRANT reader TO app");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );

    root(&engine, "REVOKE reader FROM app");
    denied_as(&engine, "app", "SELECT id FROM orders");
}

#[test]
fn public_grant_reaches_every_role() {
    let engine = fixture();
    root(&engine, "CREATE ROLE other LOGIN");
    root(&engine, "GRANT SELECT ON orders TO PUBLIC");
    // Including a role never named in the grant.
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
    assert_eq!(
        rows(ok_as(&engine, "other", "SELECT id FROM orders")).len(),
        2
    );
    root(&engine, "REVOKE SELECT ON orders FROM PUBLIC");
    denied_as(&engine, "app", "SELECT id FROM orders");
}

#[test]
fn holding_a_privilege_does_not_confer_the_right_to_grant_it() {
    let engine = fixture();
    root(&engine, "CREATE ROLE third LOGIN");
    root(&engine, "GRANT SELECT ON orders TO app");

    denied_as(&engine, "app", "GRANT SELECT ON orders TO third");
    denied_as(&engine, "third", "SELECT id FROM orders");

    root(&engine, "GRANT SELECT ON orders TO app WITH GRANT OPTION");
    ok_as(&engine, "app", "GRANT SELECT ON orders TO third");
    assert_eq!(
        rows(ok_as(&engine, "third", "SELECT id FROM orders")).len(),
        2
    );
}

#[test]
fn revoke_restrict_refuses_dependents_and_cascade_unwinds_them() {
    let engine = fixture();
    root(&engine, "CREATE ROLE third LOGIN");
    root(&engine, "GRANT SELECT ON orders TO app WITH GRANT OPTION");
    ok_as(&engine, "app", "GRANT SELECT ON orders TO third");

    let err = as_role(&engine, ROOT, "REVOKE SELECT ON orders FROM app")
        .expect_err("RESTRICT must refuse while a dependent grant exists");
    assert!(
        err.contains("CASCADE"),
        "message should name the way out: {err}"
    );
    assert_eq!(
        rows(ok_as(&engine, "third", "SELECT id FROM orders")).len(),
        2,
        "a refused revoke must not have partially applied"
    );

    root(&engine, "REVOKE SELECT ON orders FROM app CASCADE");
    denied_as(&engine, "app", "SELECT id FROM orders");
    denied_as(&engine, "third", "SELECT id FROM orders");
}

#[test]
fn filtered_writes_need_select_on_top_of_the_write_privilege() {
    let engine = fixture();
    root(&engine, "GRANT UPDATE, DELETE ON orders TO app");

    // A filtered write reads rows to decide what to change.
    denied_as(&engine, "app", "UPDATE orders SET total = 1 WHERE id = 1");
    denied_as(&engine, "app", "DELETE FROM orders WHERE id = 1");
    // An unconditional one reads nothing.
    ok_as(&engine, "app", "UPDATE orders SET total = 1");

    root(&engine, "GRANT SELECT ON orders TO app");
    ok_as(&engine, "app", "UPDATE orders SET total = 2 WHERE id = 1");
}

#[test]
fn truncate_is_not_implied_by_delete() {
    let engine = fixture();
    root(&engine, "GRANT DELETE, SELECT ON orders TO app");
    denied_as(&engine, "app", "TRUNCATE orders");
    root(&engine, "GRANT TRUNCATE ON orders TO app");
    ok_as(&engine, "app", "TRUNCATE orders");
}

#[test]
fn a_join_needs_select_on_every_table_it_reads() {
    let engine = fixture();
    root(&engine, "CREATE TABLE secret (id INT, code INT)");
    root(&engine, "INSERT INTO secret VALUES (1, 42)");
    root(&engine, "GRANT SELECT ON orders TO app");

    // Joining must not become a way to read a table the session may not read directly.
    denied_as(
        &engine,
        "app",
        "SELECT o.id FROM orders o JOIN secret s ON o.id = s.id",
    );
    root(&engine, "GRANT SELECT ON secret TO app");
    assert_eq!(
        rows(ok_as(
            &engine,
            "app",
            "SELECT o.id FROM orders o JOIN secret s ON o.id = s.id"
        ))
        .len(),
        1
    );
}

#[test]
fn a_creator_owns_its_table_and_needs_no_self_grant() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE ROLE app LOGIN");
    ok_as(&engine, "app", "CREATE TABLE mine (id INT)");
    ok_as(&engine, "app", "INSERT INTO mine VALUES (1)");
    assert_eq!(rows(ok_as(&engine, "app", "SELECT id FROM mine")).len(), 1);
}

#[test]
fn dropping_a_table_takes_its_grants_with_it() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app");
    root(&engine, "DROP TABLE orders");
    root(&engine, "CREATE TABLE orders (id INT, total INT)");
    // The recreated table is a different object; the old grant must not carry over.
    denied_as(&engine, "app", "SELECT id FROM orders");
}

#[test]
fn dropping_and_altering_need_ownership_which_no_grant_confers() {
    let engine = fixture();
    // Every table privilege there is, still not enough: DROP and ALTER are the owner's rights.
    root(
        &engine,
        "GRANT ALL PRIVILEGES ON orders TO app WITH GRANT OPTION",
    );
    denied_as(&engine, "app", "DROP TABLE orders");
    denied_as(&engine, "app", "ALTER TABLE orders ADD COLUMN note TEXT");

    // On a table it owns, both work without any grant at all.
    ok_as(&engine, "app", "CREATE TABLE mine (id INT)");
    ok_as(&engine, "app", "ALTER TABLE mine ADD COLUMN note TEXT");
    ok_as(&engine, "app", "DROP TABLE mine");
}

#[test]
fn ownership_reaches_through_role_membership() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE ROLE team");
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "GRANT team TO app");
    ok_as(&engine, "team", "CREATE TABLE shared (id INT)");
    // `app` inherits `team`, so it inherits what `team` owns.
    ok_as(&engine, "app", "ALTER TABLE shared ADD COLUMN note TEXT");
    ok_as(&engine, "app", "DROP TABLE shared");
}

#[test]
fn a_view_is_not_a_way_around_the_check_on_its_base_table() {
    let engine = fixture();
    root(&engine, "CREATE VIEW order_ids AS SELECT id FROM orders");

    // A view's body is re-analyzed as the querying role, so reading through it needs the same
    // privilege reading the table directly would. This is the stricter of the two readings a SQL
    // engine can take (the alternative runs the body as the view's owner), and it is the one that
    // cannot turn a view into a privilege-laundering step.
    denied_as(&engine, "app", "SELECT id FROM order_ids");

    root(&engine, "GRANT SELECT ON orders TO app");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM order_ids")).len(),
        2
    );
}

// === Regressions for the bypasses the pre-push audit found ===============

#[test]
fn a_nested_body_does_not_run_as_superuser() {
    let engine = fixture();
    // The catalog used to analyze a `DO` / `CALL` / trigger body took the trait's permissive
    // defaults, so `is_superuser()` was true inside it and the body could write the role catalog —
    // handing any authenticated role a superuser row. The body must be analyzed as the same role
    // that reached it.
    let err = as_role(
        &engine,
        "app",
        "DO $$ INSERT INTO nusadb_roles VALUES ('app','t','t','t','t','t') $$",
    )
    .expect_err("a nested body must not reach the role catalog");
    assert!(
        !err.is_empty(),
        "writing the role catalog from a nested body must be refused"
    );
    // The decisive check: `app` must still not be a superuser afterwards.
    denied_as(&engine, "app", "SELECT id FROM orders");
}

#[test]
fn merge_needs_privileges_on_its_target() {
    let engine = fixture();
    // MERGE reads the target through its ON condition and writes it through the WHEN arms, but had
    // no check at all on the target — only on the USING source.
    denied_as(
        &engine,
        "app",
        "MERGE INTO orders t USING (VALUES (1)) s(x) ON t.id = s.x \
         WHEN MATCHED THEN UPDATE SET total = 0",
    );
    denied_as(
        &engine,
        "app",
        "MERGE INTO orders t USING (VALUES (9)) s(x) ON t.id = s.x \
         WHEN NOT MATCHED THEN INSERT (id, total) VALUES (9, 9)",
    );

    // SELECT alone is not enough — the write side still needs its own privilege.
    root(&engine, "GRANT SELECT ON orders TO app");
    denied_as(
        &engine,
        "app",
        "MERGE INTO orders t USING (VALUES (1)) s(x) ON t.id = s.x \
         WHEN MATCHED THEN UPDATE SET total = 0",
    );

    root(&engine, "GRANT UPDATE ON orders TO app");
    ok_as(
        &engine,
        "app",
        "MERGE INTO orders t USING (VALUES (1)) s(x) ON t.id = s.x \
         WHEN MATCHED THEN UPDATE SET total = 0",
    );
}

#[test]
fn renaming_a_table_carries_its_ownership_and_grants() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "CREATE ROLE reader LOGIN");
    ok_as(&engine, "app", "CREATE TABLE t (id INT)");
    ok_as(&engine, "app", "GRANT SELECT ON t TO reader");

    ok_as(&engine, "app", "ALTER TABLE t RENAME TO t2");
    // The owner must not be locked out of its own table by renaming it, and the grant must follow.
    ok_as(&engine, "app", "INSERT INTO t2 VALUES (1)");
    assert_eq!(rows(ok_as(&engine, "reader", "SELECT id FROM t2")).len(), 1);

    // And the vacated name must not carry the old permissions to whatever takes it next.
    root(&engine, "CREATE TABLE t (id INT)");
    denied_as(&engine, "reader", "SELECT id FROM t");
}

#[test]
fn a_cascade_revoke_terminates_on_a_grantor_cycle() {
    let engine = fixture();
    root(&engine, "CREATE ROLE a LOGIN");
    root(&engine, "CREATE ROLE b LOGIN");
    root(&engine, "GRANT SELECT ON orders TO a WITH GRANT OPTION");
    ok_as(
        &engine,
        "a",
        "GRANT SELECT ON orders TO b WITH GRANT OPTION",
    );
    // Re-granting to `a` re-parents `a`'s row to `b`, closing a grantor cycle a <-> b. The cascade
    // walked that loop forever and overflowed the stack — an abort, not a catchable error.
    ok_as(
        &engine,
        "b",
        "GRANT SELECT ON orders TO a WITH GRANT OPTION",
    );

    root(&engine, "REVOKE SELECT ON orders FROM a CASCADE");
    denied_as(&engine, "a", "SELECT id FROM orders");
    denied_as(&engine, "b", "SELECT id FROM orders");
}

#[test]
fn a_cached_plan_is_not_reused_across_a_role_switch() {
    let engine = fixture();
    root(&engine, "CREATE ROLE weak LOGIN");
    root(&engine, "GRANT SELECT ON orders TO app");
    root(&engine, "GRANT weak TO app");

    // Planning as `app` (which may read) must not leave a plan that `weak` (which may not) can be
    // served. Both directions are checked through the same statement text.
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
    denied_as(&engine, "weak", "SELECT id FROM orders");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
}

#[test]
fn set_role_requires_membership() {
    let engine = fixture();
    root(&engine, "CREATE ROLE elevated");
    // Otherwise SET ROLE walks straight past every other check.
    denied_as(&engine, "app", "SET ROLE elevated");
    root(&engine, "GRANT elevated TO app");
    ok_as(&engine, "app", "SET ROLE elevated");
}

#[test]
fn membership_cycles_are_refused() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE ROLE a");
    root(&engine, "CREATE ROLE b");
    root(&engine, "CREATE ROLE c");

    // Self-membership.
    assert!(
        as_role(&engine, ROOT, "GRANT a TO a").is_err(),
        "a role may not be made a member of itself"
    );

    // The two-role loop, which the first version of the check walked past.
    root(&engine, "GRANT a TO b");
    let err =
        as_role(&engine, ROOT, "GRANT b TO a").expect_err("a membership cycle must be refused");
    assert!(err.contains("cycle"), "message should say why: {err}");

    // And a longer chain: a -> b -> c, so c must not flow back into a.
    root(&engine, "GRANT b TO c");
    assert!(
        as_role(&engine, ROOT, "GRANT c TO a").is_err(),
        "a three-role cycle must be refused too"
    );

    // A diamond is not a cycle: both b and c already inherit from a, and c inheriting from b as
    // well keeps the graph acyclic. Refusing this would be over-strict.
    root(&engine, "CREATE ROLE d");
    root(&engine, "GRANT a TO d");
    root(&engine, "GRANT d TO c");
}

#[test]
fn creating_roles_requires_authority_and_is_not_a_path_to_superuser() {
    let engine = fixture();
    denied_as(&engine, "app", "CREATE ROLE sneaky");

    root(&engine, "ALTER ROLE app CREATEROLE");
    ok_as(&engine, "app", "CREATE ROLE ordinary");
    // A CREATEROLE role that could mint a superuser would hold superuser authority by two steps.
    denied_as(&engine, "app", "CREATE ROLE godmode SUPERUSER");
}

#[test]
fn a_wildcard_grant_skips_the_access_control_catalogs() {
    let engine = fixture();
    root(
        &engine,
        "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app",
    );
    // The wildcard reached the user's table...
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );

    // ...but a grantee that could write the privilege catalog could rewrite its own permissions.
    let txn = engine
        .begin(nusadb_core::IsolationLevel::default())
        .expect("begin");
    let grants = nusadb_sql::rbac::all_grants(&engine, txn).expect("read grants");
    let _ = engine.rollback(txn);
    assert!(
        !grants
            .iter()
            .any(|g| g.object.ends_with(nusadb_sql::rbac::PRIVILEGE_CATALOG)),
        "a wildcard grant must not cover the privilege catalog itself"
    );
}

#[test]
fn superuser_bypasses_every_check_directly_and_through_membership() {
    let engine = fixture();
    // The bootstrap user holds nothing explicitly and still reaches everything.
    assert_eq!(rows(ok_as(&engine, ROOT, "SELECT id FROM orders")).len(), 2);
    root(&engine, "CREATE ROLE admin SUPERUSER");
    root(&engine, "GRANT admin TO app");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
}

#[test]
fn a_role_cannot_be_dropped_while_it_owns_objects() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE ROLE app LOGIN");
    ok_as(&engine, "app", "CREATE TABLE mine (id INT)");
    let err = as_role(&engine, ROOT, "DROP ROLE app")
        .expect_err("dropping an owner would orphan its tables");
    assert!(err.contains("still owns"), "message should say why: {err}");
    root(&engine, "DROP TABLE mine");
    root(&engine, "DROP ROLE app");
}

#[test]
fn public_is_reserved_and_password_is_refused_rather_than_ignored() {
    let engine = BtreeEngine::new();
    // A real role named `public` would make `GRANT ... TO public` ambiguous, in the direction of
    // more access.
    assert!(
        as_role(&engine, ROOT, "CREATE ROLE public").is_err(),
        "`public` must not be creatable as a real role"
    );
    // A password set here would never be checked at login.
    let err = as_role(&engine, ROOT, "CREATE ROLE app LOGIN PASSWORD 'x'")
        .expect_err("PASSWORD must be refused, not silently dropped");
    assert!(
        err.contains("auth-user"),
        "message should point at the real knob: {err}"
    );
}

#[test]
fn revoking_the_grant_option_keeps_the_privilege() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app WITH GRANT OPTION");
    root(&engine, "REVOKE GRANT OPTION FOR SELECT ON orders FROM app");
    // The privilege itself survives; only the right to pass it on is gone.
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT id FROM orders")).len(),
        2
    );
    root(&engine, "CREATE ROLE third LOGIN");
    denied_as(&engine, "app", "GRANT SELECT ON orders TO third");
}

#[test]
fn table_privileges_view_lists_what_was_granted() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app");
    let listed = rows(ok_as(
        &engine,
        ROOT,
        "SELECT grantee, table_name, privilege_type FROM information_schema.table_privileges",
    ));
    assert!(
        listed.iter().any(|r| {
            r == &[
                Value::Text("app".to_owned()),
                Value::Text("orders".to_owned()),
                Value::Text("SELECT".to_owned()),
            ]
        }),
        "the grant should be visible through information_schema: {listed:?}"
    );
}

// ---- Role-administration and metadata regressions ---------------------------------------
// Six ways the first RBAC cut let a non-superuser reach past its grants. Each test drives the
// real enforcement path and fails on the pre-fix code.

/// B1 — a `CREATEROLE` role cannot grant itself a SUPERUSER role and thereby become superuser.
/// Minting a superuser was already blocked; membership was the unguarded second door.
#[test]
fn createrole_cannot_escalate_via_membership_in_a_superuser_role() {
    let engine = fixture();
    root(&engine, "CREATE ROLE admin SUPERUSER");
    root(&engine, "ALTER ROLE app CREATEROLE");
    // The escalation attempt is refused...
    denied_as(&engine, "app", "GRANT admin TO app");
    // ...and app still cannot read a table it was never granted.
    denied_as(&engine, "app", "SELECT id FROM orders");
}

/// B2 — the bootstrap superuser's name is reserved, so no catalog role can be created under it to
/// inherit ownership of every unowned object.
#[test]
fn the_bootstrap_superuser_name_is_reserved() {
    let engine = fixture();
    root(&engine, "ALTER ROLE app CREATEROLE");
    let err = as_role(&engine, "app", "CREATE ROLE \"nusadb-root\"").unwrap_err();
    assert!(
        err.contains("reserved"),
        "creating a role named after the bootstrap superuser must be refused: {err}"
    );
}

/// B3 — `CREATE TABLE AS` records the creator as owner, so the creator can read its own table.
#[test]
fn create_table_as_is_owned_by_its_creator() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app");
    root(&engine, "GRANT CREATE ON SCHEMA public TO app");
    ok_as(&engine, "app", "CREATE TABLE mine AS SELECT id FROM orders");
    // Owned by app: app reads it without any further grant.
    let got = rows(ok_as(&engine, "app", "SELECT id FROM mine"));
    assert_eq!(got.len(), 2);
}

/// B4 — `DROP SCHEMA` is owner-or-superuser only: a role cannot drop another's schema.
#[test]
fn drop_schema_requires_ownership() {
    let engine = fixture();
    root(&engine, "CREATE SCHEMA victim");
    root(&engine, "CREATE TABLE victim.secrets (id INT)");
    denied_as(&engine, "app", "DROP SCHEMA victim CASCADE");
    // The schema survived the denied attempt: the owner can still drop it (a no-op the second
    // time would error with "schema not found").
    ok_as(&engine, ROOT, "DROP SCHEMA victim CASCADE");
}

/// B4 — a schema's own creator (not only a superuser) may drop it, because CREATE SCHEMA now
/// records ownership.
#[test]
fn a_schema_can_be_dropped_by_its_creator() {
    let engine = fixture();
    root(&engine, "GRANT CREATE ON SCHEMA public TO app"); // not required, but mirrors real use
    root(&engine, "ALTER ROLE app CREATEDB");
    // app creates and then drops its own schema.
    ok_as(&engine, "app", "CREATE SCHEMA mine");
    ok_as(&engine, "app", "DROP SCHEMA mine");
}

/// B4 — `DROP DATABASE` is superuser-only.
#[test]
fn drop_database_requires_superuser() {
    let engine = fixture();
    denied_as(&engine, "app", "DROP DATABASE nusadb");
}

/// B6 — `ANALYZE` needs SELECT on the table: it reads every row and persists column values.
#[test]
fn analyze_requires_select_privilege() {
    let engine = fixture();
    denied_as(&engine, "app", "ANALYZE orders");
    root(&engine, "GRANT SELECT ON orders TO app");
    // With SELECT it is allowed.
    ok_as(&engine, "app", "ANALYZE orders");
}

/// W7 — `REVOKE` of a non-existent role is a loud error, like its `GRANT` twin, not a silent
/// no-op.
#[test]
fn revoke_of_a_missing_role_is_an_error() {
    let engine = fixture();
    let err = as_role(&engine, ROOT, "REVOKE ghost FROM app").unwrap_err();
    assert!(
        err.contains("does not exist"),
        "revoking a missing role should error: {err}"
    );
}

/// B5 — dropping a schema CASCADE clears its member tables' grants, so a later same-named table
/// does not inherit the dropped one's permissions. Without the cleanup a role keeps SELECT on the
/// reincarnated table.
#[test]
fn drop_schema_cascade_clears_member_grants() {
    let engine = fixture();
    root(&engine, "CREATE SCHEMA s");
    root(&engine, "CREATE TABLE s.t (id INT)");
    root(&engine, "GRANT SELECT ON s.t TO app");
    // app can read it now.
    ok_as(&engine, "app", "SELECT id FROM s.t");
    // Drop the whole schema, then recreate the same names.
    root(&engine, "DROP SCHEMA s CASCADE");
    root(&engine, "CREATE SCHEMA s");
    root(&engine, "CREATE TABLE s.t (id INT)");
    // The stale grant must be gone: app cannot read the reincarnated table.
    denied_as(&engine, "app", "SELECT id FROM s.t");
}

/// SHOW COLUMNS must not leak a table's shape to a role with no relationship to it; any single
/// privilege (or ownership) is enough to see it.
#[test]
fn show_columns_requires_some_privilege() {
    let engine = fixture();
    denied_as(&engine, "app", "SHOW COLUMNS FROM orders");
    root(&engine, "GRANT SELECT ON orders TO app");
    // Any privilege suffices — the column list is visible now.
    ok_as(&engine, "app", "SHOW COLUMNS FROM orders");
}

/// CREATE INDEX restructures a table — the owner's right, denied to a mere reader.
#[test]
fn create_index_requires_ownership() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app");
    denied_as(&engine, "app", "CREATE INDEX ord_total ON orders (total)");
}

/// CREATE TRIGGER needs the TRIGGER privilege, not merely the ability to read the table.
#[test]
fn create_trigger_requires_trigger_privilege() {
    let engine = fixture();
    root(&engine, "CREATE TABLE audit (msg TEXT)");
    root(&engine, "GRANT SELECT ON orders TO app");
    denied_as(
        &engine,
        "app",
        "CREATE TRIGGER t AFTER INSERT ON orders FOR EACH ROW INSERT INTO audit VALUES ('x')",
    );
    root(&engine, "GRANT TRIGGER ON orders TO app");
    ok_as(
        &engine,
        "app",
        "CREATE TRIGGER t AFTER INSERT ON orders FOR EACH ROW INSERT INTO audit VALUES ('x')",
    );
}

/// LOCK TABLE guards a write, so a read-only role cannot take one; a write privilege (or
/// ownership) grants it.
#[test]
fn lock_table_requires_write_or_ownership() {
    let engine = fixture();
    root(&engine, "GRANT SELECT ON orders TO app");
    denied_as(&engine, "app", "LOCK TABLE orders IN ACCESS EXCLUSIVE MODE");
    root(&engine, "GRANT UPDATE ON orders TO app");
    ok_as(&engine, "app", "LOCK TABLE orders IN ACCESS EXCLUSIVE MODE");
}

/// A `NOINHERIT` member does not automatically wield its roles' privileges, but it may still
/// `SET ROLE` into them — that is exactly what `NOINHERIT` is for. Eligibility to switch follows
/// membership, not the inherit flag.
#[test]
fn noinherit_member_may_assume_but_does_not_inherit() {
    let engine = fixture();
    root(&engine, "CREATE ROLE parent");
    root(&engine, "GRANT SELECT ON orders TO parent");
    root(&engine, "CREATE ROLE child LOGIN NOINHERIT");
    root(&engine, "GRANT parent TO child");

    // NOINHERIT: child does not inherit parent's SELECT, so a plain read is refused.
    denied_as(&engine, "child", "SELECT id FROM orders");

    let txn = engine
        .begin(nusadb_core::IsolationLevel::default())
        .unwrap();
    // ...yet child may SET ROLE into parent (membership, not inheritance, governs this).
    assert!(
        nusadb_sql::rbac::may_assume_role(&engine, txn, "child", "parent").unwrap(),
        "a NOINHERIT member must still be able to SET ROLE into its role"
    );
    // And the inherit-gated effective set correctly excludes parent.
    assert!(
        !nusadb_sql::rbac::effective_roles(&engine, txn, "child")
            .unwrap()
            .contains("parent"),
        "a NOINHERIT member must not silently inherit its role's privileges"
    );
    let _ = engine.rollback(txn);
}

/// Unquoted `PUBLIC` is the pseudo-role every session belongs to; quoted `"PUBLIC"` is an ordinary
/// (case-preserving) role identifier, not the pseudo-role. Treating the quoted spelling as the
/// pseudo-role would widen a grant from "one specific role" to "everyone".
#[test]
fn quoted_public_is_a_role_identifier_not_the_pseudo_role() {
    use nusadb_sql::ast::{Grantee, Statement};
    let unquoted = parse("GRANT SELECT ON t TO PUBLIC").unwrap();
    let Statement::Grant(g) = unquoted else {
        panic!("expected a GRANT, got {unquoted:?}")
    };
    assert_eq!(g.grantees, vec![Grantee::Public]);

    let quoted = parse("GRANT SELECT ON t TO \"PUBLIC\"").unwrap();
    let Statement::Grant(g) = quoted else {
        panic!("expected a GRANT, got {quoted:?}")
    };
    assert_eq!(
        g.grantees,
        vec![Grantee::Role("PUBLIC".to_owned())],
        "quoted \"PUBLIC\" must be a role identifier, not the pseudo-role"
    );
}

// === Column-level privileges ============================================================

/// An engine with a three-column table and a bare `clerk` role.
fn column_fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE t (a INT, b INT, secret INT)");
    root(&engine, "INSERT INTO t VALUES (1, 2, 99)");
    root(&engine, "CREATE ROLE clerk LOGIN");
    engine
}

#[test]
fn column_select_grant_admits_only_the_named_columns() {
    let engine = column_fixture();
    root(&engine, "GRANT SELECT (a, b) ON t TO clerk");
    // The granted columns read; the ungranted one, and a `*` that would expand it, are denied.
    assert_eq!(rows(ok_as(&engine, "clerk", "SELECT a, b FROM t")).len(), 1);
    assert_eq!(rows(ok_as(&engine, "clerk", "SELECT a FROM t")).len(), 1);
    denied_as(&engine, "clerk", "SELECT secret FROM t");
    denied_as(&engine, "clerk", "SELECT * FROM t");
    // A predicate on an ungranted column is a read of it, so it is denied too.
    denied_as(&engine, "clerk", "SELECT a FROM t WHERE secret = 99");
}

#[test]
fn column_select_grant_allows_a_column_free_count() {
    let engine = column_fixture();
    root(&engine, "GRANT SELECT (a) ON t TO clerk");
    // `count(*)` reads no specific column, and holding SELECT on *some* column is enough — matching
    // the reference engine.
    assert_eq!(
        rows(ok_as(&engine, "clerk", "SELECT count(*) FROM t")),
        vec![vec![Value::Int(1)]]
    );
}

#[test]
fn no_column_grant_still_denies_the_whole_table() {
    let engine = column_fixture();
    // `clerk` holds nothing: neither a bare column read nor `count(*)` is allowed.
    denied_as(&engine, "clerk", "SELECT a FROM t");
    denied_as(&engine, "clerk", "SELECT count(*) FROM t");
}

#[test]
fn column_insert_grant_admits_only_the_named_columns() {
    let engine = column_fixture();
    root(&engine, "GRANT INSERT (a) ON t TO clerk");
    ok_as(&engine, "clerk", "INSERT INTO t (a) VALUES (10)");
    denied_as(&engine, "clerk", "INSERT INTO t (b) VALUES (10)");
    // A column-less INSERT targets every column, so a single column grant cannot satisfy it.
    denied_as(&engine, "clerk", "INSERT INTO t VALUES (1, 2, 3)");
}

#[test]
fn column_update_grant_admits_only_the_named_columns() {
    let engine = column_fixture();
    root(&engine, "GRANT UPDATE (b) ON t TO clerk");
    // An unconditional UPDATE of the granted column reads no rows, so it needs only column UPDATE.
    ok_as(&engine, "clerk", "UPDATE t SET b = 5");
    denied_as(&engine, "clerk", "UPDATE t SET a = 5");
}

#[test]
fn revoking_a_column_grant_denies_again() {
    let engine = column_fixture();
    root(&engine, "GRANT SELECT (a) ON t TO clerk");
    ok_as(&engine, "clerk", "SELECT a FROM t");
    root(&engine, "REVOKE SELECT (a) ON t FROM clerk");
    denied_as(&engine, "clerk", "SELECT a FROM t");
}

#[test]
fn a_table_wide_grant_covers_every_column() {
    let engine = column_fixture();
    root(&engine, "GRANT SELECT ON t TO clerk");
    // A plain table grant reads every column, including one no column grant named.
    assert_eq!(rows(ok_as(&engine, "clerk", "SELECT * FROM t")).len(), 1);
    assert_eq!(
        rows(ok_as(&engine, "clerk", "SELECT secret FROM t")).len(),
        1
    );
}

#[test]
fn granting_a_column_that_does_not_exist_is_an_error() {
    let engine = column_fixture();
    match as_role(&engine, ROOT, "GRANT SELECT (nope) ON t TO clerk") {
        Err(msg) => assert!(
            msg.contains("column not found") || msg.contains("nope"),
            "expected a column-not-found error, got: {msg}"
        ),
        Ok(other) => panic!("granting on a missing column should fail, got {other:?}"),
    }
}

#[test]
fn a_column_grantee_cannot_regrant_without_grant_option() {
    let engine = column_fixture();
    root(&engine, "GRANT SELECT (a) ON t TO clerk");
    // `clerk` holds the column privilege but not the grant option, so it may not pass it on.
    denied_as(&engine, "clerk", "GRANT SELECT (a) ON t TO app");
}
