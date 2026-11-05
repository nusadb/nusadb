//! An index on a partitioned table racing a partition that joins it.
//!
//! `CREATE INDEX` on a partitioned table builds the index on every partition it has, and a table
//! that joins it later (`PARTITION OF`, `ATTACH PARTITION`) gets the index when it joins. When the
//! two run in concurrent transactions, neither sees the other's uncommitted work: the index would
//! not be built on the new partition, and the partition would not see the index. One of them must
//! fail with the no-wait `40001` instead, and the retry after the other commits gives the
//! partition its index.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema, TxnId};
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

/// The analyzer's view of the engine, under the session's open transaction when there is one.
struct TxnCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: Option<TxnId>,
}

impl TxnCatalog<'_> {
    fn with_txn<T>(
        &self,
        f: impl FnOnce(TxnId) -> Result<T, nusadb_sql::Error>,
    ) -> Result<T, nusadb_sql::Error> {
        if let Some(txn) = self.txn {
            return f(txn);
        }
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let out = f(txn);
        let _ = self.engine.commit(txn);
        out
    }
}

impl Catalog for TxnCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.lookup_table_in(nusadb_core::PUBLIC_SCHEMA, name)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.with_txn(|txn| {
            self.engine
                .lookup_table_as_of_in(txn, schema, name)
                .map_err(Into::into)
        })
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes(self.engine, txn, name))
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
}

/// One client connection.
struct Conn<'a> {
    engine: &'a BtreeEngine,
    session: Session<'a>,
}

impl<'a> Conn<'a> {
    fn new(engine: &'a BtreeEngine) -> Self {
        Self {
            engine,
            session: Session::new(engine),
        }
    }

    fn try_run(&mut self, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
        let catalog = TxnCatalog {
            engine: self.engine,
            txn: self.session.current_txn(),
        };
        let logical = analyze(parse(sql)?, &catalog)?;
        self.session.execute(plan(logical))
    }

    fn run(&mut self, sql: &str) -> ExecutionResult {
        self.try_run(sql)
            .unwrap_or_else(|e| panic!("`{sql}` failed: {e} ({})", e.sqlstate()))
    }

    fn refused(&mut self, sql: &str) {
        let err = self
            .try_run(sql)
            .expect_err("a concurrent index build on the parent refuses it");
        assert_eq!(err.sqlstate(), "40001", "`{sql}`: {err}");
    }
}

/// The names of the committed indexes of `table`, sorted.
fn indexes(engine: &BtreeEngine, table: &str) -> Vec<String> {
    let id = engine.lookup_table(table).unwrap().unwrap().id;
    let mut names: Vec<String> = engine
        .list_indexes(id)
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect();
    names.sort();
    names
}

fn fixture(engine: &BtreeEngine) {
    let mut c = Conn::new(engine);
    c.run("CREATE TABLE pt (a INT, b INT) PARTITION BY RANGE (a)");
    c.run("CREATE TABLE p1 PARTITION OF pt FOR VALUES FROM (0) TO (10)");
    c.run("CREATE TABLE ext (a INT, b INT)");
    c.run("INSERT INTO ext VALUES (25, 1)");
}

#[test]
fn a_partition_created_beside_an_uncommitted_index_build_is_refused() {
    let engine = BtreeEngine::new();
    fixture(&engine);
    let mut a = Conn::new(&engine);
    let mut b = Conn::new(&engine);
    a.run("BEGIN");
    a.run("CREATE INDEX pt_a ON pt (a)");
    b.refused("CREATE TABLE p2 PARTITION OF pt FOR VALUES FROM (10) TO (20)");
    b.refused("ALTER TABLE pt ATTACH PARTITION ext FOR VALUES FROM (20) TO (30)");
    a.run("COMMIT");
    b.run("CREATE TABLE p2 PARTITION OF pt FOR VALUES FROM (10) TO (20)");
    b.run("ALTER TABLE pt ATTACH PARTITION ext FOR VALUES FROM (20) TO (30)");
    assert_eq!(indexes(&engine, "p1"), vec!["p1_a_idx"]);
    assert_eq!(indexes(&engine, "p2"), vec!["p2_a_idx"]);
    assert_eq!(indexes(&engine, "ext"), vec!["ext_a_idx"]);
}

#[test]
fn an_index_build_beside_an_uncommitted_partition_is_refused() {
    for join in [
        "CREATE TABLE p2 PARTITION OF pt FOR VALUES FROM (10) TO (20)",
        "ALTER TABLE pt ATTACH PARTITION ext FOR VALUES FROM (20) TO (30)",
    ] {
        let engine = BtreeEngine::new();
        fixture(&engine);
        let mut a = Conn::new(&engine);
        let mut b = Conn::new(&engine);
        b.run("BEGIN");
        b.run(join);
        a.refused("CREATE INDEX pt_a ON pt (a)");
        b.run("COMMIT");
        a.run("CREATE INDEX pt_a ON pt (a)");
        let joined = if join.contains("ext") { "ext" } else { "p2" };
        assert_eq!(indexes(&engine, joined), vec![format!("{joined}_a_idx")]);
    }
}

#[test]
fn a_writer_through_the_parent_does_not_hold_up_a_new_partition() {
    let engine = BtreeEngine::new();
    fixture(&engine);
    let mut c = Conn::new(&engine);
    c.run("CREATE INDEX pt_a ON pt (a)");
    let mut writer = Conn::new(&engine);
    writer.run("BEGIN");
    writer.run("INSERT INTO pt VALUES (1, 1)");
    c.run("CREATE TABLE p2 PARTITION OF pt FOR VALUES FROM (10) TO (20)");
    c.run("ALTER TABLE pt ATTACH PARTITION ext FOR VALUES FROM (20) TO (30)");
    writer.run("COMMIT");
    assert_eq!(indexes(&engine, "p2"), vec!["p2_a_idx"]);
    assert_eq!(indexes(&engine, "ext"), vec!["ext_a_idx"]);
}
