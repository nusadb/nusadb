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

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
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
