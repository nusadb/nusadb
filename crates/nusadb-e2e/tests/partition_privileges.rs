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
    let stmt = parse(sql).map_err(|e| e.to_string())?;
    let logical = analyze(stmt, &RoleCatalog { engine, user }).map_err(|e| e.to_string())?;
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
}
