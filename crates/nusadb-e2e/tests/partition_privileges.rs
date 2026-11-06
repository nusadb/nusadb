//! Access control on partitioned and inheritance parents.
//!
//! A statement on a parent (without `ONLY`) checks the privileges of the table it names and
//! applies that table's row-level security to every row it reaches, wherever the row is stored. The
//! partitions and children it reaches are not checked separately. Naming a partition or child
//! directly checks that table's own privileges and applies its own policies, so a grant on the
//! parent never opens a partition to direct access.
//!
//! The catalog below answers privilege, row-level-security and inheritance questions from the real
//! catalogs, the way the wire server's per-statement catalog does.

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
use nusadb_sql::{Catalog, ExecutionResult, PolicyDef, Session, analyze, parse, plan};

/// A catalog that resolves privileges, policies and the partition tree against the real catalogs,
/// one snapshot per question.
struct RoleCatalog<'a> {
    engine: &'a BtreeEngine,
    user: &'a str,
    /// The session's search path; empty for the default (`public`).
    path: &'a [&'a str],
}

impl RoleCatalog<'_> {
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

impl Catalog for RoleCatalog<'_> {
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
        if self.path.is_empty() {
            vec![nusadb_core::PUBLIC_SCHEMA.to_owned()]
        } else {
            self.path.iter().map(|s| (*s).to_owned()).collect()
        }
    }

    fn is_superuser(&self) -> bool {
        self.with_txn(|txn| nusadb_sql::rbac::principal(self.engine, txn, self.user))
            .is_ok_and(|p| p.superuser)
    }

    fn current_user(&self) -> String {
        self.user.to_owned()
    }

    fn any_inheritance(&self) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::inheritance_any(self.engine, txn))
    }

    fn inheritance_descendants(&self, table: &str) -> Result<Vec<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::inheritance_descendants(self.engine, txn, table))
    }

    fn partition_key_columns(&self, table: &str) -> Result<Option<Vec<String>>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::partition_key_columns(self.engine, txn, table))
    }

    fn partitions_to_prune(
        &self,
        parent: &str,
        constraints: &[nusadb_sql::PruneConstraint],
    ) -> Result<Vec<String>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::partitions_to_prune(self.engine, txn, parent, constraints))
    }

    fn rls_enabled(&self, schema: &str, name: &str) -> Result<bool, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::rls_table_enabled(self.engine, txn, schema, name))
    }

    fn lookup_policies(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Vec<PolicyDef>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::lookup_policies_for(self.engine, txn, schema, name))
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
}

/// The bootstrap superuser, used for every setup statement.
const ROOT: &str = "nusadb-root";

/// Run `sql` as `user`, returning the result or the error text.
fn as_role(engine: &BtreeEngine, user: &str, sql: &str) -> Result<ExecutionResult, String> {
    as_role_in(engine, user, &[], sql)
}

/// Run `sql` as `user` under the search path `path`, returning the result or the error text.
fn as_role_in(
    engine: &BtreeEngine,
    user: &str,
    path: &[&str],
    sql: &str,
) -> Result<ExecutionResult, String> {
    let stmt = parse(sql).map_err(|e| e.to_string())?;
    let logical = analyze(stmt, &RoleCatalog { engine, user, path }).map_err(|e| e.to_string())?;
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

/// Run `sql` as `user` and expect a permission denial naming `object`.
fn denied_as(engine: &BtreeEngine, user: &str, sql: &str, object: &str) {
    match as_role(engine, user, sql) {
        Err(msg) => assert!(
            msg.contains("permission denied") && msg.contains(object),
            "`{sql}` as {user} should be denied on {object}, got: {msg}"
        ),
        Ok(other) => panic!("`{sql}` as {user} should have been denied, got {other:?}"),
    }
}

/// The rows of a `SELECT` result.
fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

/// The first column of every row, as integers.
fn ids(result: ExecutionResult) -> Vec<i64> {
    rows(result)
        .into_iter()
        .map(|row| match row.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

/// The number of rows a write reports.
fn count(result: &ExecutionResult) -> usize {
    match result {
        ExecutionResult::Inserted(n)
        | ExecutionResult::Updated(n)
        | ExecutionResult::Deleted(n) => *n,
        other => panic!("expected a row count, got {other:?}"),
    }
}

/// A partitioned table with two partitions, one of them sub-partitioned, holding one row in each
/// leaf, and a role `app` that holds nothing yet.
fn fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    root(
        &engine,
        "CREATE TABLE g (g_id BIGINT NOT NULL, k BIGINT, s TEXT, d DATE NOT NULL, v BIGINT NOT \
         NULL, CONSTRAINT g_pkey PRIMARY KEY (g_id), CONSTRAINT g_nat_key UNIQUE (k, s, d)) \
         PARTITION BY RANGE (d)",
    );
    root(
        &engine,
        "CREATE TABLE g_p202610 PARTITION OF g FOR VALUES FROM ('2026-10-01') TO ('2026-11-01')",
    );
    root(
        &engine,
        "CREATE TABLE g_p202611 PARTITION OF g FOR VALUES FROM ('2026-11-01') TO ('2026-12-01') \
         PARTITION BY RANGE (v)",
    );
    root(
        &engine,
        "CREATE TABLE g_p202611_a PARTITION OF g_p202611 FOR VALUES FROM (0) TO (1000)",
    );
    root(
        &engine,
        "INSERT INTO g VALUES (1, 1, 'a', '2026-10-05', 1), (2, 2, 'b', '2026-11-05', 2)",
    );
    root(&engine, "CREATE ROLE app LOGIN");
    engine
}

#[test]
fn a_grant_on_the_parent_covers_every_partition_it_reaches() {
    let engine = fixture();
    root(&engine, "GRANT SELECT, INSERT, UPDATE ON g TO app");

    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT g_id FROM g ORDER BY g_id")),
        vec![1, 2]
    );
    // Pruned to one partition, joined, aggregated, in a subquery: all read through the parent.
    assert_eq!(
        ids(ok_as(
            &engine,
            "app",
            "SELECT g_id FROM g WHERE d = '2026-11-05'"
        )),
        vec![2]
    );
    assert_eq!(
        ids(ok_as(
            &engine,
            "app",
            "SELECT a.g_id FROM g a JOIN g b ON a.g_id = b.g_id ORDER BY a.g_id"
        )),
        vec![1, 2]
    );
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT count(*) FROM g")),
        vec![2]
    );
    assert_eq!(
        ids(ok_as(
            &engine,
            "app",
            "SELECT g_id FROM g WHERE g_id IN (SELECT g_id FROM g WHERE v > 1)"
        )),
        vec![2]
    );

    assert_eq!(count(&ok_as(&engine, "app", "UPDATE g SET v = v + 10")), 2);
    assert_eq!(
        count(&ok_as(&engine, "app", "UPDATE g SET v = 7 WHERE g_id = 2")),
        1
    );
    ok_as(
        &engine,
        "app",
        "INSERT INTO g VALUES (3, 3, 'c', '2026-10-06', 3)",
    );
    ok_as(
        &engine,
        "app",
        "INSERT INTO g VALUES (4, 1, 'a', '2026-10-05', 5) ON CONFLICT (k, s, d) DO UPDATE SET v \
         = excluded.v",
    );
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT v FROM g WHERE g_id = 1")),
        vec![vec![Value::Int(5)]]
    );
    // An UPDATE whose new key moves the row to another partition.
    ok_as(
        &engine,
        "app",
        "UPDATE g SET d = '2026-11-20' WHERE g_id = 3",
    );
    assert_eq!(
        ids(ok_as(
            &engine,
            "app",
            "SELECT g_id FROM g WHERE d >= '2026-11-01' ORDER BY g_id"
        )),
        vec![2, 3]
    );

    // What was not granted on the parent stays denied, checked on the parent.
    denied_as(&engine, "app", "DELETE FROM g WHERE g_id = 1", "public.g");
    denied_as(&engine, "app", "TRUNCATE g", "public.g");
    root(&engine, "GRANT DELETE, TRUNCATE ON g TO app");
    assert_eq!(
        count(&ok_as(&engine, "app", "DELETE FROM g WHERE g_id = 1")),
        1
    );
    ok_as(&engine, "app", "TRUNCATE g");
    assert!(rows(ok_as(&engine, "app", "SELECT g_id FROM g")).is_empty());
}

#[test]
fn a_grant_on_the_parent_does_not_open_a_partition_to_direct_access() {
    let engine = fixture();
    root(&engine, "GRANT ALL PRIVILEGES ON g TO app");

    for sql in [
        "SELECT g_id FROM g_p202610",
        "SELECT g_id FROM ONLY g_p202610",
        "INSERT INTO g_p202610 VALUES (9, 9, 'z', '2026-10-09', 9)",
        "UPDATE g_p202610 SET v = 0",
        "DELETE FROM g_p202610",
        "TRUNCATE g_p202610",
        "SELECT g_id FROM g_p202611_a",
        "SELECT g_id FROM g_p202611",
    ] {
        match as_role(&engine, "app", sql) {
            Err(msg) => assert!(msg.contains("permission denied"), "{sql}: {msg}"),
            Ok(other) => panic!("`{sql}` should have been denied, got {other:?}"),
        }
    }
    // A statement on the parent may not read a partition by name in its other clauses either.
    denied_as(
        &engine,
        "app",
        "UPDATE g SET v = (SELECT max(v) FROM g_p202610)",
        "g_p202610",
    );
    denied_as(
        &engine,
        "app",
        "SELECT g_id FROM g WHERE g_id IN (SELECT g_id FROM g_p202611_a)",
        "g_p202611_a",
    );
    denied_as(
        &engine,
        "app",
        "DELETE FROM g WHERE g_id IN (SELECT g_id FROM g_p202610)",
        "g_p202610",
    );

    // The reverse: a grant on a partition reaches that partition only.
    let engine = fixture();
    root(&engine, "GRANT SELECT ON g_p202610 TO app");
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT g_id FROM g_p202610")),
        vec![1]
    );
    denied_as(&engine, "app", "SELECT g_id FROM g", "public.g");
}

#[test]
fn column_grants_on_the_parent_are_checked_against_the_columns_used() {
    let engine = fixture();
    root(&engine, "GRANT SELECT (g_id, v), UPDATE (v) ON g TO app");

    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT g_id, v FROM g ORDER BY g_id")),
        vec![1, 2]
    );
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT count(*) FROM g")),
        vec![2]
    );
    denied_as(&engine, "app", "SELECT k FROM g", "`k`");
    denied_as(&engine, "app", "SELECT * FROM g", "`k`");
    denied_as(&engine, "app", "SELECT g_id FROM g WHERE s = 'a'", "`s`");

    assert_eq!(
        count(&ok_as(&engine, "app", "UPDATE g SET v = 5 WHERE g_id = 1")),
        1
    );
    denied_as(&engine, "app", "UPDATE g SET k = 5 WHERE g_id = 1", "`k`");
    denied_as(&engine, "app", "UPDATE g SET v = 5 WHERE k = 1", "`k`");
    assert_eq!(
        rows(ok_as(&engine, "app", "SELECT v FROM g WHERE g_id = 1")),
        vec![vec![Value::Int(5)]]
    );
}

#[test]
fn a_partitioned_source_of_a_write_needs_select_on_the_parent_only() {
    let engine = fixture();
    root(&engine, "CREATE TABLE t (id BIGINT, v BIGINT)");
    root(&engine, "INSERT INTO t VALUES (1, 0), (2, 0)");
    root(&engine, "GRANT SELECT, INSERT, UPDATE ON t TO app");
    denied_as(
        &engine,
        "app",
        "UPDATE t SET v = g.v FROM g WHERE g.g_id = t.id",
        "public.g",
    );
    root(&engine, "GRANT SELECT ON g TO app");
    assert_eq!(
        count(&ok_as(
            &engine,
            "app",
            "UPDATE t SET v = g.v FROM g WHERE g.g_id = t.id"
        )),
        2
    );
    ok_as(&engine, "app", "INSERT INTO t SELECT g_id + 10, v FROM g");
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM t ORDER BY id")),
        vec![1, 2, 11, 12]
    );
}

#[test]
fn foreign_key_checks_need_no_privilege_on_either_side() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE fkp (id BIGINT PRIMARY KEY)");
    root(&engine, "INSERT INTO fkp VALUES (1)");
    root(
        &engine,
        "CREATE TABLE fkc (id BIGINT NOT NULL, r BIGINT REFERENCES fkp (id)) PARTITION BY RANGE \
         (id)",
    );
    root(
        &engine,
        "CREATE TABLE fkc_1 PARTITION OF fkc FOR VALUES FROM (0) TO (100)",
    );
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "GRANT INSERT ON fkc TO app");
    ok_as(&engine, "app", "INSERT INTO fkc VALUES (1, 1)");
    match as_role(&engine, "app", "INSERT INTO fkc VALUES (2, 9)") {
        Err(msg) => assert!(msg.contains("violates foreign key"), "{msg}"),
        Ok(other) => panic!("expected a foreign key violation, got {other:?}"),
    }
}

#[test]
fn the_parent_policies_apply_to_every_row_read_or_written_through_it() {
    let engine = BtreeEngine::new();
    root(
        &engine,
        "CREATE TABLE r (id INT NOT NULL, owner TEXT, d INT NOT NULL) PARTITION BY RANGE (d)",
    );
    root(
        &engine,
        "CREATE TABLE r1 PARTITION OF r FOR VALUES FROM (0) TO (100)",
    );
    root(
        &engine,
        "CREATE TABLE r2 PARTITION OF r FOR VALUES FROM (100) TO (200)",
    );
    root(
        &engine,
        "INSERT INTO r VALUES (1, 'app', 1), (2, 'other', 2), (3, 'app', 150), (4, 'other', 150)",
    );
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "GRANT SELECT, UPDATE, DELETE ON r TO app");
    root(&engine, "GRANT SELECT ON r1 TO app");
    root(&engine, "ALTER TABLE r ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own ON r USING (owner = current_user)",
    );

    // The parent's policy filters the rows of every partition.
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM r ORDER BY id")),
        vec![1, 3]
    );
    // A partition named directly applies its own policies: none, so every row.
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM r1 ORDER BY id")),
        vec![1, 2]
    );

    // A partition's own policies do not apply through the parent.
    root(&engine, "ALTER TABLE r1 ENABLE ROW LEVEL SECURITY");
    root(&engine, "CREATE POLICY only_two ON r1 USING (id = 2)");
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM r ORDER BY id")),
        vec![1, 3]
    );
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM r1 ORDER BY id")),
        vec![2]
    );

    // Writes through the parent reach only the rows its policy admits.
    assert_eq!(count(&ok_as(&engine, "app", "UPDATE r SET d = d")), 2);
    assert_eq!(count(&ok_as(&engine, "app", "DELETE FROM r")), 2);
    assert_eq!(
        ids(ok_as(&engine, ROOT, "SELECT id FROM r ORDER BY id")),
        vec![2, 4]
    );
}

#[test]
fn an_inheritance_parent_checks_and_filters_the_same_way() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE ip (id INT, owner TEXT)");
    root(&engine, "CREATE TABLE ic (extra INT) INHERITS (ip)");
    root(&engine, "INSERT INTO ip VALUES (1, 'app')");
    root(
        &engine,
        "INSERT INTO ic VALUES (2, 'other', 9), (3, 'app', 9)",
    );
    root(&engine, "CREATE ROLE app LOGIN");
    root(&engine, "GRANT SELECT, UPDATE ON ip TO app");
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM ip ORDER BY id")),
        vec![1, 2, 3]
    );
    assert_eq!(
        count(&ok_as(&engine, "app", "UPDATE ip SET owner = owner")),
        3
    );
    denied_as(&engine, "app", "SELECT id FROM ic", "public.ic");

    root(&engine, "ALTER TABLE ip ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own ON ip USING (owner = current_user)",
    );
    assert_eq!(
        ids(ok_as(&engine, "app", "SELECT id FROM ip ORDER BY id")),
        vec![1, 3]
    );
}

/// The text values of the second column of `SELECT id, <text column> ... ORDER BY id`.
fn texts(result: ExecutionResult) -> Vec<Option<String>> {
    rows(result)
        .into_iter()
        .map(|row| match row.get(1) {
            Some(Value::Text(s)) => Some(s.clone()),
            Some(Value::Null) => None,
            other => panic!("expected text, got {other:?}"),
        })
        .collect()
}

/// `docs` partitioned on `k` into `docs_a` and `docs_b`, a hidden row of `bob` in `docs_b`, and a
/// role `alice` that may read and write `docs` and read `docs_b`, whose policy shows her only her
/// own rows.
fn docs_fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    root(
        &engine,
        "CREATE TABLE docs (id INT, k INT, owner TEXT, body TEXT) PARTITION BY RANGE (k)",
    );
    root(
        &engine,
        "CREATE TABLE docs_a PARTITION OF docs FOR VALUES FROM (0) TO (10)",
    );
    root(
        &engine,
        "CREATE TABLE docs_b PARTITION OF docs FOR VALUES FROM (10) TO (20)",
    );
    root(
        &engine,
        "INSERT INTO docs VALUES (1, 1, 'alice', 'mine'), (2, 15, 'bob', 'TOP-SECRET'), (3, 11, \
         'alice', 'hers')",
    );
    root(&engine, "CREATE ROLE alice LOGIN");
    root(&engine, "GRANT SELECT, UPDATE, DELETE ON docs TO alice");
    root(&engine, "GRANT SELECT ON docs_b TO alice");
    root(&engine, "ALTER TABLE docs_b ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own_b ON docs_b USING (owner = current_user)",
    );
    engine
}

/// The bodies of `docs`, by id, as the superuser sees them.
fn bodies(engine: &BtreeEngine) -> Vec<Option<String>> {
    texts(ok_as(engine, ROOT, "SELECT id, body FROM docs ORDER BY id"))
}

#[test]
fn a_partition_read_inside_a_write_on_the_parent_keeps_its_own_policies() {
    let engine = docs_fixture();
    // The subquery reads docs_b as itself, so its policy hides bob's row; the update itself
    // reaches every row through the parent, which has no policy.
    ok_as(
        &engine,
        "alice",
        "UPDATE docs SET body = (SELECT string_agg(body, ',' ORDER BY body) FROM docs_b)",
    );
    assert_eq!(bodies(&engine), vec![Some("hers".to_owned()); 3]);

    // A count in the WHERE does not reveal the hidden row either.
    let engine = docs_fixture();
    assert_eq!(
        count(&ok_as(
            &engine,
            "alice",
            "UPDATE docs SET body = 'w' WHERE (SELECT count(*) FROM docs_b) > 1"
        )),
        0
    );
    assert_eq!(
        count(&ok_as(
            &engine,
            "alice",
            "DELETE FROM docs WHERE (SELECT count(*) FROM docs_b) > 1"
        )),
        0
    );
    assert_eq!(
        count(&ok_as(
            &engine,
            "alice",
            "DELETE FROM docs WHERE (SELECT count(*) FROM docs_b) = 1 AND id = 1"
        )),
        1
    );

    // Column grants: the copied value comes from a row the policy hides.
    let engine = docs_fixture();
    root(&engine, "REVOKE SELECT ON docs_b FROM alice");
    root(&engine, "GRANT SELECT (id, owner, body) ON docs_b TO alice");
    ok_as(
        &engine,
        "alice",
        "UPDATE docs SET body = (SELECT body FROM docs_b WHERE id = 2) WHERE id = 1",
    );
    assert_eq!(
        bodies(&engine),
        vec![None, Some("TOP-SECRET".to_owned()), Some("hers".to_owned())]
    );

    // A partition whose policy admits nothing, read for a row stored in another partition, and the
    // value handed back by RETURNING.
    let engine = docs_fixture();
    root(&engine, "GRANT SELECT ON docs_a TO alice");
    root(&engine, "ALTER TABLE docs_a ENABLE ROW LEVEL SECURITY");
    root(&engine, "CREATE POLICY nothing ON docs_a USING (false)");
    assert_eq!(
        texts(ok_as(
            &engine,
            "alice",
            "UPDATE docs SET body = (SELECT string_agg(body, ',') FROM docs_a) WHERE id = 3 \
             RETURNING id, body"
        )),
        vec![None]
    );
}

#[test]
fn a_partition_below_a_sub_partitioned_parent_keeps_its_own_policies() {
    let engine = BtreeEngine::new();
    root(
        &engine,
        "CREATE TABLE sp (id INT, k INT, owner TEXT, body TEXT) PARTITION BY RANGE (k)",
    );
    root(
        &engine,
        "CREATE TABLE sp_a PARTITION OF sp FOR VALUES FROM (0) TO (10)",
    );
    root(
        &engine,
        "CREATE TABLE sp_b PARTITION OF sp FOR VALUES FROM (10) TO (20) PARTITION BY RANGE (id)",
    );
    root(
        &engine,
        "CREATE TABLE sp_b1 PARTITION OF sp_b FOR VALUES FROM (0) TO (100)",
    );
    root(
        &engine,
        "INSERT INTO sp VALUES (1, 1, 'alice', 'mine'), (2, 15, 'bob', 'TOP-SECRET'), (3, 11, \
         'alice', 'hers')",
    );
    root(&engine, "CREATE ROLE alice LOGIN");
    root(&engine, "GRANT SELECT, UPDATE ON sp TO alice");
    root(&engine, "GRANT SELECT ON sp_b, sp_b1 TO alice");
    root(&engine, "ALTER TABLE sp_b1 ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own ON sp_b1 USING (owner = current_user)",
    );
    // Read through sp_b, sp_b1 takes sp_b's policies (none); read by name, its own.
    ok_as(
        &engine,
        "alice",
        "UPDATE sp SET body = (SELECT string_agg(body, ',' ORDER BY body) FROM sp_b1) WHERE id = 1",
    );
    ok_as(
        &engine,
        "alice",
        "UPDATE sp SET body = (SELECT string_agg(body, ',' ORDER BY body) FROM sp_b) WHERE id = 3",
    );
    assert_eq!(
        texts(ok_as(&engine, ROOT, "SELECT id, body FROM sp ORDER BY id")),
        vec![
            Some("hers".to_owned()),
            Some("TOP-SECRET".to_owned()),
            Some("TOP-SECRET,hers".to_owned())
        ]
    );
}

#[test]
fn a_child_read_inside_a_write_on_an_inheritance_parent_keeps_its_own_policies() {
    let engine = BtreeEngine::new();
    root(&engine, "CREATE TABLE ip (id INT, owner TEXT, body TEXT)");
    root(&engine, "CREATE TABLE ic_a () INHERITS (ip)");
    root(&engine, "CREATE TABLE ic_b () INHERITS (ip)");
    root(&engine, "INSERT INTO ic_a VALUES (1, 'alice', 'mine')");
    root(
        &engine,
        "INSERT INTO ic_b VALUES (2, 'bob', 'TOP-SECRET'), (3, 'alice', 'hers')",
    );
    root(&engine, "CREATE ROLE alice LOGIN");
    root(&engine, "GRANT UPDATE ON ip TO alice");
    root(&engine, "GRANT SELECT ON ic_b TO alice");
    root(&engine, "ALTER TABLE ic_b ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY own_b ON ic_b USING (owner = current_user)",
    );
    ok_as(
        &engine,
        "alice",
        "UPDATE ip SET body = (SELECT string_agg(body, ',' ORDER BY body) FROM ic_b)",
    );
    assert_eq!(
        texts(ok_as(&engine, ROOT, "SELECT id, body FROM ip ORDER BY id")),
        vec![Some("hers".to_owned()); 3]
    );
}

#[test]
fn a_parent_policy_that_reads_a_partition_reads_it_under_its_own_policies() {
    let engine = docs_fixture();
    root(&engine, "INSERT INTO docs VALUES (4, 12, 'bob', 'b2')");
    root(&engine, "ALTER TABLE docs ENABLE ROW LEVEL SECURITY");
    root(
        &engine,
        "CREATE POLICY via_b ON docs USING (k < 10 OR id IN (SELECT id FROM docs_b))",
    );
    // The policy's subquery sees only alice's rows of docs_b, so of docs_b's rows only id 3 passes.
    assert_eq!(
        ids(ok_as(&engine, "alice", "SELECT id FROM docs ORDER BY id")),
        vec![1, 3]
    );
}

#[test]
fn column_grants_on_a_partitioned_source_of_a_write_are_checked() {
    let engine = BtreeEngine::new();
    root(
        &engine,
        "CREATE TABLE p (id INT, k INT, v TEXT) PARTITION BY RANGE (k)",
    );
    root(
        &engine,
        "CREATE TABLE p1 PARTITION OF p FOR VALUES FROM (0) TO (10)",
    );
    root(&engine, "INSERT INTO p VALUES (1, 1, 'secret')");
    root(&engine, "CREATE TABLE t_other (id INT, v TEXT)");
    root(&engine, "INSERT INTO t_other VALUES (1, 'x')");
    root(&engine, "CREATE ROLE alice LOGIN");
    root(&engine, "GRANT SELECT (id, k) ON p TO alice");
    root(&engine, "GRANT ALL ON t_other TO alice");
    denied_as(
        &engine,
        "alice",
        "UPDATE t_other SET v = p.v FROM p WHERE p.id = t_other.id",
        "`v`",
    );
    denied_as(
        &engine,
        "alice",
        "DELETE FROM t_other USING p WHERE p.v = 'secret' AND p.id = t_other.id",
        "`v`",
    );
    assert_eq!(
        texts(ok_as(
            &engine,
            ROOT,
            "SELECT id, v FROM t_other ORDER BY id"
        )),
        vec![Some("x".to_owned())]
    );
    // The granted columns of the parent read, as for any other source.
    assert_eq!(
        count(&ok_as(
            &engine,
            "alice",
            "UPDATE t_other SET v = 'y' FROM p WHERE p.id = t_other.id AND p.k = 1"
        )),
        1
    );
}

/// A parent reaches the partitions recorded under it, whatever the session's search path: a table
/// of a partition's name in a schema ahead of the partition's own is neither read nor written, so a
/// write on the parent cannot reach a table its privileges do not cover.
#[test]
fn a_parent_reaches_its_own_partitions_whatever_the_search_path() {
    let engine = fixture();
    root(&engine, "CREATE SCHEMA other");
    root(
        &engine,
        "CREATE TABLE other.g_p202610 (g_id BIGINT, k BIGINT, s TEXT, d DATE, v BIGINT)",
    );
    root(
        &engine,
        "INSERT INTO other.g_p202610 VALUES (9, 9, 'z', '2026-10-09', 9)",
    );
    root(&engine, "GRANT SELECT, UPDATE, DELETE ON g TO app");
    let path = ["other", "public"];
    let ok_in = |sql: &str| {
        as_role_in(&engine, "app", &path, sql)
            .unwrap_or_else(|e| panic!("`{sql}` as app should succeed: {e}"))
    };

    assert_eq!(
        count(&ok_in("UPDATE g SET v = v + 10 WHERE g_id IN (1, 9)")),
        1
    );
    assert_eq!(count(&ok_in("DELETE FROM g WHERE g_id = 9")), 0);
    assert_eq!(
        ids(root(&engine, "SELECT g_id FROM other.g_p202610")),
        vec![9]
    );
    assert_eq!(
        rows(root(&engine, "SELECT v FROM other.g_p202610")),
        vec![vec![Value::Int(9)]]
    );
    assert_eq!(
        rows(root(&engine, "SELECT v FROM g_p202610")),
        vec![vec![Value::Int(11)]]
    );
    assert_eq!(ids(ok_in("SELECT g_id FROM g ORDER BY g_id")), vec![1, 2]);
}

/// The column names and the rows, sorted, of a `RETURNING` result.
fn returned(result: ExecutionResult) -> (Vec<String>, Vec<Vec<Value>>) {
    match result {
        ExecutionResult::Rows {
            columns, mut rows, ..
        } => {
            rows.sort_by_key(|row| format!("{row:?}"));
            (columns, rows)
        },
        other => panic!("expected rows, got {other:?}"),
    }
}

/// Rows of integers.
fn ints(rows: &[&[i64]]) -> Vec<Vec<Value>> {
    rows.iter()
        .map(|row| row.iter().map(|&v| Value::Int(v)).collect())
        .collect()
}

/// An inheritance parent `ip (id, v)` whose child adds `extra`, a partitioned `pp (id, v)`, each
/// holding two rows, and two roles on both: `whole` with table privileges, `narrow` with column
/// grants on the parent's columns only. Neither holds anything on the child or the partitions.
fn returning_fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    for sql in [
        "CREATE TABLE ip (id INT, v INT)",
        "CREATE TABLE ic (extra TEXT) INHERITS (ip)",
        "INSERT INTO ip VALUES (1, 10)",
        "INSERT INTO ic VALUES (2, 20, 'secret')",
        "CREATE TABLE pp (id INT, v INT) PARTITION BY RANGE (id)",
        "CREATE TABLE pp1 PARTITION OF pp FOR VALUES FROM (0) TO (10)",
        "CREATE TABLE pp2 PARTITION OF pp FOR VALUES FROM (10) TO (20)",
        "INSERT INTO pp VALUES (1, 10), (11, 110)",
        "CREATE ROLE whole LOGIN",
        "CREATE ROLE narrow LOGIN",
        "GRANT SELECT, UPDATE, DELETE ON ip TO whole",
        "GRANT SELECT, UPDATE, DELETE ON pp TO whole",
        "GRANT SELECT (id, v), UPDATE (v), DELETE ON ip TO narrow",
        "GRANT SELECT (id, v), UPDATE (v), DELETE ON pp TO narrow",
    ] {
        root(&engine, sql);
    }
    engine
}

/// `RETURNING *` and `RETURNING t.*` through a parent stand for the parent's columns on every row,
/// whatever columns the table holding the row adds, for table and column grants alike.
#[test]
fn a_wildcard_returning_through_a_parent_is_the_parents_columns() {
    type Ints = &'static [&'static [i64]];
    let id_v = vec!["id".to_owned(), "v".to_owned()];
    let topologies: [(&str, Ints, Ints); 2] = [
        ("ip", &[&[1, 10], &[2, 20]], &[&[1, 11], &[2, 21]]),
        ("pp", &[&[1, 10], &[11, 110]], &[&[1, 11], &[11, 111]]),
    ];
    for (parent, before, after) in topologies {
        for user in [ROOT, "whole", "narrow"] {
            for (sql, expected) in [
                (format!("UPDATE {parent} SET v = v + 1 RETURNING *"), after),
                (
                    format!("UPDATE {parent} AS t SET v = v + 1 RETURNING t.*"),
                    after,
                ),
                (
                    format!("UPDATE {parent} SET v = v + 1 RETURNING {parent}.*"),
                    after,
                ),
                (format!("DELETE FROM {parent} RETURNING *"), before),
                (format!("DELETE FROM {parent} AS t RETURNING t.*"), before),
            ] {
                let engine = returning_fixture();
                assert_eq!(
                    returned(ok_as(&engine, user, &sql)),
                    (id_v.clone(), ints(expected)),
                    "`{sql}` as {user}"
                );
            }
        }
    }
}

/// A wildcard through a parent still needs SELECT on each of the parent's columns, and a column
/// only the child has is not reachable by name, as a column the parent lacks.
#[test]
fn a_wildcard_returning_through_a_parent_needs_select_on_its_columns() {
    for parent in ["ip", "pp"] {
        let engine = returning_fixture();
        root(
            &engine,
            &format!("REVOKE SELECT (v) ON {parent} FROM narrow"),
        );
        for sql in [
            format!("UPDATE {parent} SET v = 5 RETURNING *"),
            format!("UPDATE {parent} AS t SET v = 5 RETURNING t.*"),
            format!("DELETE FROM {parent} RETURNING *"),
        ] {
            denied_as(&engine, "narrow", &sql, "`v`");
        }
        let (columns, returned_rows) = returned(ok_as(
            &engine,
            "narrow",
            &format!("UPDATE {parent} SET v = 5 RETURNING id"),
        ));
        assert_eq!(columns, vec!["id".to_owned()]);
        assert_eq!(returned_rows.len(), 2);
    }
    let engine = returning_fixture();
    for user in [ROOT, "whole", "narrow"] {
        for sql in [
            "UPDATE ip SET v = 5 RETURNING extra",
            "UPDATE ip AS t SET v = 5 RETURNING t.extra",
            "DELETE FROM ip RETURNING extra",
        ] {
            let refused = as_role(&engine, user, sql).expect_err(sql);
            assert!(refused.contains("extra"), "`{sql}` as {user}: {refused}");
        }
    }
}

/// The wildcard follows the parent by column name: a column added to the parent lands after the
/// child's own in the child's rows, and is read from there.
#[test]
fn a_wildcard_returning_through_a_parent_matches_columns_by_name() {
    let engine = returning_fixture();
    root(&engine, "CREATE TABLE ig (more TEXT) INHERITS (ic)");
    root(&engine, "INSERT INTO ig VALUES (3, 30, 'hidden', 'also')");
    root(&engine, "ALTER TABLE ip ADD COLUMN z INT DEFAULT 7");
    let columns: Vec<String> = ["id", "v", "z"].map(str::to_owned).to_vec();
    assert_eq!(
        returned(ok_as(&engine, "whole", "UPDATE ip SET v = 0 RETURNING *")),
        (columns, ints(&[&[1, 0, 7], &[2, 0, 7], &[3, 0, 7]]))
    );
}

/// Only a statement through a parent narrows the wildcard: `ONLY` the parent and the child named
/// directly return their own columns, and a writable `WITH` hands on what the statement returned.
#[test]
fn a_wildcard_returning_keeps_the_columns_of_the_table_named() {
    let engine = returning_fixture();
    assert_eq!(
        returned(ok_as(&engine, ROOT, "UPDATE ONLY ip SET v = 1 RETURNING *")),
        (vec!["id".to_owned(), "v".to_owned()], ints(&[&[1, 1]]))
    );
    assert_eq!(
        returned(ok_as(&engine, ROOT, "UPDATE ic SET v = 2 RETURNING *")),
        (
            vec!["id".to_owned(), "v".to_owned(), "extra".to_owned()],
            vec![vec![
                Value::Int(2),
                Value::Int(2),
                Value::Text("secret".to_owned())
            ]]
        )
    );
    assert_eq!(
        returned(ok_as(
            &engine,
            "whole",
            "WITH u AS (UPDATE ip SET v = 3 RETURNING *) SELECT * FROM u"
        )),
        (
            vec!["id".to_owned(), "v".to_owned()],
            ints(&[&[1, 3], &[2, 3]])
        )
    );
}

/// `row_to_json` of a parent a write goes through stands for the parent's columns, in the parent's
/// order, however the table holding the row lays them out, as `SELECT row_to_json(ip) FROM ip` does.
#[test]
fn a_whole_row_through_a_parent_is_the_parents_columns() {
    let engine = returning_fixture();
    root(&engine, "CREATE TABLE ig (more TEXT) INHERITS (ic)");
    root(&engine, "INSERT INTO ig VALUES (3, 30, 'hidden', 'also')");
    root(&engine, "ALTER TABLE ip ADD COLUMN z INT DEFAULT 7");
    root(&engine, "GRANT SELECT (z) ON ip TO narrow");
    let json = |rows: Vec<&str>| {
        rows.into_iter()
            .map(|r| vec![Value::Text(r.to_owned())])
            .collect::<Vec<_>>()
    };
    let expected = json(vec![
        r#"{"id": 1, "v": 0, "z": 7}"#,
        r#"{"id": 2, "v": 0, "z": 7}"#,
        r#"{"id": 3, "v": 0, "z": 7}"#,
    ]);
    for role in ["whole", "narrow"] {
        for sql in [
            "UPDATE ip SET v = 0 RETURNING row_to_json(ip)::text",
            "UPDATE ip AS x SET v = 0 RETURNING row_to_json(x)::text",
        ] {
            assert_eq!(rows(ok_as(&engine, role, sql)), expected, "{role}: {sql}");
        }
        assert_eq!(
            rows(ok_as(
                &engine,
                role,
                "SELECT row_to_json(ip)::text FROM ip ORDER BY id"
            )),
            expected,
            "{role}"
        );
    }
    // A column only a child adds decides nothing: it is not part of the row read.
    let hidden = "UPDATE ip SET v = 1 WHERE row_to_json(ip)::text LIKE '%hidden%'";
    assert!(matches!(
        ok_as(&engine, "narrow", hidden),
        ExecutionResult::Updated(0)
    ));
    let hidden = "DELETE FROM ip WHERE row_to_json(ip)::text LIKE '%secret%'";
    assert!(matches!(
        ok_as(&engine, "narrow", hidden),
        ExecutionResult::Deleted(0)
    ));
    assert_eq!(
        rows(ok_as(
            &engine,
            "narrow",
            "DELETE FROM ip WHERE id = 3 RETURNING row_to_json(ip)::text"
        )),
        json(vec![r#"{"id": 3, "v": 0, "z": 7}"#])
    );
    // Named directly, a table's whole row is its own.
    assert_eq!(
        rows(ok_as(
            &engine,
            ROOT,
            "UPDATE ONLY ic SET v = 5 RETURNING row_to_json(ic)::text"
        )),
        json(vec![r#"{"id": 2, "v": 5, "extra": "secret", "z": 7}"#])
    );
}

/// A whole row through a parent still needs SELECT on each of the parent's columns.
#[test]
fn a_whole_row_through_a_parent_needs_select_on_its_columns() {
    let engine = returning_fixture();
    root(&engine, "CREATE ROLE blind LOGIN");
    root(
        &engine,
        "GRANT SELECT (id), UPDATE (v), DELETE ON ip TO blind",
    );
    root(
        &engine,
        "GRANT SELECT (id), UPDATE (v), DELETE ON pp TO blind",
    );
    for sql in [
        "UPDATE ip SET v = 1 RETURNING row_to_json(ip)",
        "UPDATE pp SET v = 1 RETURNING row_to_json(pp)",
        "UPDATE ip SET v = 1 WHERE row_to_json(ip)::text LIKE '%1%'",
        "DELETE FROM ip RETURNING row_to_json(ip)",
        "DELETE FROM pp WHERE row_to_json(pp)::text LIKE '%1%'",
        "SELECT row_to_json(ip) FROM ip",
    ] {
        denied_as(&engine, "blind", sql, "SELECT on column `v`");
    }
}

/// A table `victim` owned by the superuser, partitioned, with a partition holding a row, a plain
/// inheritance parent `base`, a table `loose` of the same shape, and a role `mallory` that may create
/// tables in the schema and holds nothing on any of them.
fn linking_fixture() -> BtreeEngine {
    let engine = BtreeEngine::new();
    for sql in [
        "CREATE TABLE victim (id INT, v TEXT) PARTITION BY RANGE (id)",
        "CREATE TABLE victim_low PARTITION OF victim FOR VALUES FROM (0) TO (10)",
        "INSERT INTO victim VALUES (1, 'kept')",
        "CREATE TABLE base (id INT, v TEXT)",
        "CREATE TABLE loose (id INT, v TEXT)",
        "INSERT INTO loose VALUES (50, 'loose')",
        "CREATE ROLE mallory LOGIN",
        "CREATE ROLE owner LOGIN",
        "GRANT CREATE ON SCHEMA public TO mallory",
        "GRANT CREATE ON SCHEMA public TO owner",
    ] {
        root(&engine, sql);
    }
    engine
}

/// The tables below `parent`.
fn children_of(engine: &BtreeEngine, parent: &str) -> Vec<String> {
    RoleCatalog {
        engine,
        user: ROOT,
        path: &[],
    }
    .inheritance_descendants(parent)
    .unwrap()
}

/// Whether no table `name` exists.
fn table_is_absent(engine: &BtreeEngine, name: &str) -> bool {
    engine.lookup_table(name).unwrap().is_none()
}

#[test]
fn only_the_owner_of_a_parent_may_add_a_child_to_it() {
    let engine = linking_fixture();
    for sql in [
        "CREATE TABLE stolen PARTITION OF victim FOR VALUES FROM (10) TO (20)",
        "CREATE TABLE stolen PARTITION OF victim_low FOR VALUES FROM (0) TO (5)",
        "CREATE TABLE IF NOT EXISTS stolen PARTITION OF victim FOR VALUES FROM (10) TO (20)",
        "CREATE TABLE stolen (extra INT) INHERITS (base)",
        "CREATE TABLE stolen (extra INT) INHERITS (victim)",
    ] {
        denied_as(&engine, "mallory", sql, "owner");
    }
    assert!(table_is_absent(&engine, "stolen"));
    assert!(children_of(&engine, "base").is_empty());
    assert_eq!(children_of(&engine, "victim"), ["victim_low"]);
    // A second parent is no way around the first: every parent must be the role's own.
    ok_as(&engine, "mallory", "CREATE TABLE mine (id INT, v TEXT)");
    denied_as(
        &engine,
        "mallory",
        "CREATE TABLE stolen (extra INT) INHERITS (mine, base)",
        "base",
    );
    assert!(table_is_absent(&engine, "stolen"));
    // Holding every privilege on the parent is not owning it.
    root(&engine, "GRANT ALL ON base TO mallory");
    root(&engine, "GRANT ALL ON victim TO mallory");
    denied_as(
        &engine,
        "mallory",
        "CREATE TABLE stolen (extra INT) INHERITS (base)",
        "base",
    );
    denied_as(
        &engine,
        "mallory",
        "CREATE TABLE stolen PARTITION OF victim FOR VALUES FROM (10) TO (20)",
        "victim",
    );
}

#[test]
fn a_foreign_child_never_receives_or_shows_the_rows_of_a_parent() {
    let engine = linking_fixture();
    root(&engine, "GRANT SELECT, INSERT ON victim TO mallory");
    ok_as(
        &engine,
        "mallory",
        "CREATE TABLE mine (id INT, v TEXT) PARTITION BY RANGE (id)",
    );
    ok_as(&engine, "mallory", "CREATE TABLE trap (id INT, v TEXT)");
    // Neither direction of ATTACH works without owning both tables.
    for sql in [
        "ALTER TABLE victim ATTACH PARTITION trap FOR VALUES FROM (10) TO (20)",
        "ALTER TABLE mine ATTACH PARTITION victim_low FOR VALUES FROM (0) TO (10)",
        "ALTER TABLE mine ATTACH PARTITION loose FOR VALUES FROM (0) TO (100)",
    ] {
        denied_as(&engine, "mallory", sql, "owner");
    }
    root(&engine, "INSERT INTO victim VALUES (2, 'still kept')");
    assert_eq!(
        rows(root(&engine, "SELECT id FROM victim_low ORDER BY id")).len(),
        2
    );
    assert!(rows(ok_as(&engine, "mallory", "SELECT id FROM trap")).is_empty());
    assert_eq!(rows(root(&engine, "SELECT id FROM loose")).len(), 1);
    assert_eq!(children_of(&engine, "victim"), ["victim_low"]);
    assert!(children_of(&engine, "mine").is_empty());
}

#[test]
fn the_owner_of_a_parent_links_its_own_tables_only() {
    let engine = linking_fixture();
    for sql in [
        "CREATE TABLE top (id INT, v TEXT) PARTITION BY RANGE (id)",
        "CREATE TABLE lower_part PARTITION OF top FOR VALUES FROM (0) TO (10)",
        "CREATE TABLE plain (id INT, v TEXT)",
        "CREATE TABLE sub (extra INT) INHERITS (plain)",
        "CREATE TABLE free (id INT, v TEXT)",
        "ALTER TABLE top ATTACH PARTITION free FOR VALUES FROM (10) TO (20)",
        "ALTER TABLE top DETACH PARTITION free",
    ] {
        ok_as(&engine, "owner", sql);
    }
    // The parent's owner takes no table of another role through an ATTACH.
    denied_as(
        &engine,
        "owner",
        "ALTER TABLE top ATTACH PARTITION loose FOR VALUES FROM (40) TO (60)",
        "loose",
    );
    assert_eq!(children_of(&engine, "top"), ["lower_part"]);
    // A superuser links what it does not own.
    root(
        &engine,
        "ALTER TABLE top ATTACH PARTITION loose FOR VALUES FROM (40) TO (60)",
    );
    assert_eq!(children_of(&engine, "top"), ["lower_part", "loose"]);
}
