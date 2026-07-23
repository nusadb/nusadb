//! `COPY ... FROM STDIN` streams its payload in bounded batches: rather than
//! materializing every parsed row up front — gigabytes of `ast::Value` for a multi-million-row load,
//! which OOM-killed the server before the per-transaction write ceiling could reject it — it inserts
//! in `INSERT_SELECT_BATCH`-row batches. These pins prove the batching preserves semantics:
//!
//! 1. A load larger than one batch inserts every row correctly across the batch boundary.
//! 2. A duplicate key spanning two batches is still rejected atomically — the deferred-unique
//!    enforcement must see prior batches' still-uncommitted keys (a naive per-batch immediate check,
//!    which only sees committed data, would let the duplicate through). This is the same mechanism
//!    `INSERT ... SELECT` streaming uses.

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
use nusadb_sql::ast::{Statement, Value};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Row, Session, analyze, copy_from, parse, plan,
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

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Row> {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows
}

/// Parse a `COPY <table> ... FROM STDIN` statement and drive `data` through the executor's
/// `copy_from` — the same entry point the wire server calls with the client's `CopyData` payload.
fn copy(engine: &dyn StorageEngine, sql: &str, data: &str) -> Result<usize, Error> {
    let Statement::Copy(copy) = parse(sql).unwrap() else {
        panic!("not a COPY statement: {sql}");
    };
    copy_from(engine, &copy, data)
}

/// Count the live entries in a secondary index by scanning it directly through the engine — used to
/// prove a bulk load left the index complete rather than partial.
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

/// 2500 rows > one 1024-row batch, so the load flushes three times (two full batches + a remainder)
/// — the boundary a single-Vec load never exercised.
const N: usize = 2500;

#[test]
fn copy_inserts_every_row_across_the_batch_boundary() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );

    let mut data = String::new();
    for i in 0..N {
        // tab-delimited (the COPY default), v = id * 2 so a spot check is unambiguous.
        writeln!(data, "{i}\t{}", i * 2).unwrap();
    }
    let inserted = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();
    assert_eq!(inserted, N, "every row reports inserted");

    // Every row is present (counted directly, independent of COUNT's value type).
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t").len(),
        N,
        "all rows are durable across the batch boundary"
    );
    // A row straddling the boundary carries the right value — no batch was dropped or mis-inserted.
    assert_eq!(
        rows(engine, &mut session, "SELECT v FROM t WHERE id = 1500"),
        vec![vec![Value::Int(3000)]],
        "the row at the second batch boundary is correct"
    );
}

#[test]
fn copy_rejects_a_duplicate_key_spanning_two_batches() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );

    // Row 2000 reuses id = 5 from the first batch. The duplicate is only detectable if the second
    // batch's uniqueness check sees the first batch's still-uncommitted key.
    let mut data = String::new();
    for i in 0..N {
        let id = if i == 2000 { 5 } else { i };
        writeln!(data, "{id}\t{i}").unwrap();
    }
    let err = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap_err();

    // The whole COPY rolled back atomically — a failed load commits nothing.
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t").len(),
        0,
        "a rejected COPY leaves the table empty (error was: {err:?})"
    );
}

/// A `COPY` into a table with a secondary index builds that index through the batched, key-sorted
/// path. Every loaded row must have an index entry across the batch boundary — a complete index,
/// not a partial one. The rows are loaded in the opposite of key order so the build must sort.
#[test]
fn copy_builds_a_complete_secondary_index() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    exec(engine, &mut session, "CREATE INDEX t_v ON t (v)");

    let mut data = String::new();
    for i in 0..N {
        // Descending v: the row order is the reverse of key order, so the sorted build reorders.
        writeln!(data, "{i}\t{}", N - i).unwrap();
    }
    let inserted = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();
    assert_eq!(inserted, N);

    assert_eq!(
        count_index_entries(engine, "t_v"),
        N,
        "the secondary index has an entry for every loaded row"
    );
}

/// A `COPY` whose data duplicates a key on a UNIQUE secondary index — the pair split across two
/// batches — is rejected by the batched build's uniqueness enforcement, and the whole load rolls
/// back. The second batch's build must see the first batch's already-applied entry.
#[test]
fn copy_rejects_a_duplicate_on_a_secondary_unique_index() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    exec(
        engine,
        &mut session,
        "CREATE UNIQUE INDEX t_v_uniq ON t (v)",
    );

    // Every row has a distinct v = i except row 2000, which reuses v = 7 (loaded in the first
    // batch) — a unique-index duplicate that spans the batch boundary.
    let mut data = String::new();
    for i in 0..N {
        let v = if i == 2000 { 7 } else { i };
        writeln!(data, "{i}\t{v}").unwrap();
    }
    let err = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap_err();
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t").len(),
        0,
        "a rejected COPY leaves the table empty (error was: {err:?})"
    );
}

/// `CREATE INDEX` on an already-populated table backfills every existing row through the batched,
/// key-sorted path, so the new index is complete even though the rows are not stored in key order.
#[test]
fn create_index_backfills_a_populated_table_completely() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    // Load the rows first (no secondary index yet), with v in the reverse of key order.
    let mut data = String::new();
    for i in 0..N {
        writeln!(data, "{i}\t{}", N - i).unwrap();
    }
    copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();

    // Build the index over the existing rows — the batched backfill.
    exec(engine, &mut session, "CREATE INDEX t_v ON t (v)");
    assert_eq!(
        count_index_entries(engine, "t_v"),
        N,
        "the backfill built an entry for every existing row"
    );
}

/// `CREATE UNIQUE INDEX` on a column that already holds a duplicate value is rejected by the
/// batched backfill's uniqueness check (the sorted build places the equal keys adjacent).
#[test]
fn create_unique_index_rejects_an_existing_duplicate() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    // Two rows share v = 7, so a unique index over v cannot be built.
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 7), (2, 7), (3, 9)",
    );
    let logical = analyze(
        parse("CREATE UNIQUE INDEX t_v_uniq ON t (v)").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    let err = session.execute(plan(logical)).unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("unique") || msg.contains("duplicate"),
        "CREATE UNIQUE INDEX over duplicate data is rejected: {err}"
    );
}
