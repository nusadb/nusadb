//! `CREATE INDEX` external merge sort: with spill-to-disk configured, building an index on a table
//! larger than the maintenance-memory budget streams entries into sorted runs on disk, k-way merges
//! them, and applies the index in one global key order. The index must still be complete and enforce
//! uniqueness across run boundaries — identical to the in-memory chunked build.
//!
//! `spill_config` and the maintenance budget are process-wide statics, so this lives in its own test
//! binary (run sequentially) as a single test that resets both before returning.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::fmt::Write as _;
use std::ops::Bound;

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::ast::Statement;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, SpillConfig, analyze, copy_from, parse,
    plan, set_maintenance_work_mem, set_spill_config,
};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
        Ok(Vec::new())
    }
}

fn exec(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
    session.execute(plan(logical)).unwrap()
}

fn copy(engine: &dyn StorageEngine, sql: &str, data: &str) -> Result<usize, Error> {
    let Statement::Copy(copy) = parse(sql).unwrap() else {
        panic!("not a COPY statement: {sql}");
    };
    copy_from(engine, &copy, data)
}

/// Count the live entries in a secondary index by scanning it directly through the engine.
fn count_index_entries(engine: &dyn StorageEngine, index_name: &str) -> usize {
    let txn = engine.begin(IsolationLevel::default()).unwrap();
    let id = engine
        .lookup_index(index_name)
        .unwrap()
        .expect("index exists");
    let mut scan = engine
        .index_scan(txn, id, Bound::Unbounded, Bound::Unbounded)
        .unwrap();
    let mut count = 0;
    while scan.try_next().unwrap().is_some() {
        count += 1;
    }
    engine.commit(txn).unwrap();
    count
}

/// Enough entries that, under the small budget below, the build spills into many sorted runs.
const ROWS: usize = 20_000;

#[test]
fn create_index_external_sort_builds_complete_index_and_enforces_uniqueness() {
    // Enable spill (so `CREATE INDEX` takes the external-sort path) with a small maintenance budget,
    // so `ROWS` entries spill into many sorted runs that must be k-way merged.
    set_spill_config(Some(SpillConfig {
        dir: std::env::temp_dir(),
        threshold_bytes: 64,
    }));
    set_maintenance_work_mem(128 << 10); // 128 KiB

    // 1. Completeness: rows loaded with v in reverse key order, so run order is not global key
    //    order and the merge does real work; every row must end up with an index entry.
    {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        exec(
            engine,
            &mut session,
            "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
        );
        let mut data = String::new();
        for i in 0..ROWS {
            writeln!(data, "{i}\t{}", ROWS - i).unwrap();
        }
        copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();

        exec(engine, &mut session, "CREATE INDEX t_v ON t (v)");
        assert_eq!(
            count_index_entries(engine, "t_v"),
            ROWS,
            "the external-sort backfill built an entry for every row across merged runs"
        );
    }

    // 2. Uniqueness across runs: two rows share a value, so the merged stream places the duplicate
    //    keys adjacent and the unique index rejects the build.
    {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        exec(
            engine,
            &mut session,
            "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
        );
        let mut data = String::new();
        for i in 0..ROWS {
            let v = if i == 15_000 { 5 } else { i };
            writeln!(data, "{i}\t{v}").unwrap();
        }
        copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();

        let logical = analyze(
            parse("CREATE UNIQUE INDEX t_v_uniq ON t (v)").unwrap(),
            &Cat(engine),
        )
        .unwrap();
        let err = session.execute(plan(logical)).unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("unique") || msg.contains("duplicate"),
            "external-sort CREATE UNIQUE INDEX over a cross-run duplicate is rejected: {err}"
        );
    }

    set_spill_config(None);
    set_maintenance_work_mem(0);
}
