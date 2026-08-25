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
    Catalog, Error, ExecutionResult, IndexInfo, Row, Session, analyze, copy_from, copy_to, parse,
    plan, set_maintenance_work_mem,
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
/// `copy_from`.
///
/// This is the *load*, not the whole wire path: the server runs its access check first and only
/// then calls `copy_from_in`. Nothing asserted here can speak for privileges or row-level
/// security — those are pinned in the wire crate's own tests, and a COPY change has to be
/// checked there as well as here.
fn copy(engine: &dyn StorageEngine, sql: &str, data: &str) -> Result<usize, Error> {
    let Statement::Copy(copy) = parse(sql).unwrap() else {
        panic!("not a COPY statement: {sql}");
    };
    copy_from(engine, &copy, data)
}

/// Parse a `COPY <table> ... TO STDOUT` statement and render the table through `copy_to`, returning
/// the payload the wire server would stream back.
fn copy_out(engine: &dyn StorageEngine, sql: &str) -> String {
    let Statement::Copy(copy) = parse(sql).unwrap() else {
        panic!("not a COPY statement: {sql}");
    };
    copy_to(engine, &copy).unwrap().1
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

/// The index-probe uniqueness path (integer/text keys) keeps no per-statement key state, checking
/// each batch against the backing index. A duplicate against a row committed *before* the COPY must
/// still be rejected — the probe's latest-committed view sees it — and the pre-existing row survives.
#[test]
fn copy_rejects_a_duplicate_against_a_precommitted_row() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    // A committed row the COPY will collide with (id = 5), in its own transaction.
    exec(engine, &mut session, "INSERT INTO t VALUES (5, 999)");

    let mut data = String::new();
    for i in 0..N {
        writeln!(data, "{i}\t{i}").unwrap();
    }
    let err = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap_err();

    // The COPY rolled back atomically, leaving exactly the one pre-committed row untouched.
    let ids = rows(engine, &mut session, "SELECT id FROM t");
    assert_eq!(
        ids,
        vec![vec![Value::Int(5)]],
        "a rejected COPY leaves only the pre-committed row (error was: {err:?})"
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT v FROM t WHERE id = 5"),
        vec![vec![Value::Int(999)]],
        "the pre-committed row keeps its original value"
    );
}

/// A `NUMERIC` key is *not* index-probe-eligible (its values encode inconsistently), so the COPY
/// uses the accumulating fallback: per-statement key state plus one end-of-stream scan. A duplicate
/// key spanning two batches must still be rejected atomically through that path.
#[test]
fn copy_rejects_a_cross_batch_duplicate_on_a_numeric_pk() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id NUMERIC PRIMARY KEY, v INT)",
    );

    // Row 2000 reuses id = 5 from the first batch — a cross-batch duplicate the accumulating path
    // catches against the keys it has already seen this statement.
    let mut data = String::new();
    for i in 0..N {
        let id = if i == 2000 { 5 } else { i };
        writeln!(data, "{id}\t{i}").unwrap();
    }
    let err = copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap_err();
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t").len(),
        0,
        "a rejected COPY on a NUMERIC PK leaves the table empty (error was: {err:?})"
    );
}

/// `COPY ... FROM STDIN WITH (FORMAT csv)` loads comma-delimited data, honoring quoted fields (which
/// may carry the delimiter, a newline, or a doubled quote) and the CSV NULL rule: an unquoted empty
/// field is NULL, while a quoted empty field is the empty string.
#[test]
fn copy_from_csv_honors_quoting_and_null_distinction() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name TEXT, note TEXT)",
    );

    // id 1: a quoted field with an embedded comma.
    // id 2: a quoted field with an embedded newline; trailing unquoted-empty note = NULL.
    // id 3: unquoted-empty name = NULL; a quoted note with a doubled quote → a literal quote.
    // id 4: a quoted empty note = the empty string (distinct from id 2's NULL).
    let data =
        "1,alice,\"hello, world\"\n2,\"bob\nsmith\",\n3,,\"literal \"\"q\"\"\"\n4,dan,\"\"\n";
    assert_eq!(
        copy(engine, "COPY t FROM STDIN WITH (FORMAT csv)", data).unwrap(),
        4
    );

    assert_eq!(
        rows(engine, &mut session, "SELECT note FROM t WHERE id = 1"),
        vec![vec![Value::Text("hello, world".into())]],
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT name FROM t WHERE id = 2"),
        vec![vec![Value::Text("bob\nsmith".into())]],
        "a quoted field carries an embedded newline as one value",
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t WHERE note IS NULL"),
        vec![vec![Value::Int(2)]],
        "only the unquoted-empty note is NULL",
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t WHERE name IS NULL"),
        vec![vec![Value::Int(3)]],
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT note FROM t WHERE id = 3"),
        vec![vec![Value::Text("literal \"q\"".into())]],
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT note FROM t WHERE id = 4"),
        vec![vec![Value::Text(String::new())]],
        "a quoted empty field is the empty string, not NULL",
    );
}

/// `COPY ... TO STDOUT WITH (FORMAT csv)` renders CSV that round-trips back through `FROM ... csv`:
/// values needing it are quoted, and the NULL-vs-empty-string distinction survives the export/reload.
#[test]
fn copy_to_csv_round_trips_through_from_csv() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, name TEXT, note TEXT)",
    );
    // A value with a comma+quote, an empty string, and a NULL — the cases CSV must disambiguate.
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (1, 'a,b\"c', '')",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO t VALUES (2, 'plain', NULL)",
    );

    let exported = copy_out(engine, "COPY t TO STDOUT WITH (FORMAT csv)");

    // Reload into an identical table and compare — the round-trip must preserve every value,
    // including the empty-string vs NULL distinction.
    exec(
        engine,
        &mut session,
        "CREATE TABLE t2 (id INT PRIMARY KEY, name TEXT, note TEXT)",
    );
    assert_eq!(
        copy(engine, "COPY t2 FROM STDIN WITH (FORMAT csv)", &exported).unwrap(),
        2,
    );
    assert_eq!(
        rows(
            engine,
            &mut session,
            "SELECT id, name, note FROM t ORDER BY id"
        ),
        rows(
            engine,
            &mut session,
            "SELECT id, name, note FROM t2 ORDER BY id"
        ),
        "CSV export reloads to an identical table",
    );
    // The empty string stayed an empty string and the NULL stayed NULL after the round-trip.
    assert_eq!(
        rows(engine, &mut session, "SELECT id FROM t2 WHERE note IS NULL"),
        vec![vec![Value::Int(2)]],
    );
    assert_eq!(
        rows(engine, &mut session, "SELECT note FROM t2 WHERE id = 1"),
        vec![vec![Value::Text(String::new())]],
    );
}

/// Enough rows that, under the small maintenance budget the `CREATE INDEX` tests set, the build
/// buffers past it several times — exercising the chunk boundary the way `N` does for COPY.
const BACKFILL_ROWS: usize = 20_000;

/// A deliberately small maintenance-memory budget so `BACKFILL_ROWS` entries span several backfill
/// chunks (the default budget would hold them all in one).
const SMALL_MAINTENANCE_BUDGET: usize = 128 << 10; // 128 KiB

/// `CREATE INDEX` on an already-populated table streams the rows and backfills every one through the
/// chunked, key-sorted path, so the new index is complete across chunk boundaries even though the
/// rows are not stored in key order.
#[test]
fn create_index_backfills_a_populated_table_completely() {
    set_maintenance_work_mem(SMALL_MAINTENANCE_BUDGET);
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    // Load the rows first (no secondary index yet), with v in the reverse of key order.
    let mut data = String::new();
    for i in 0..BACKFILL_ROWS {
        writeln!(data, "{i}\t{}", BACKFILL_ROWS - i).unwrap();
    }
    copy(engine, "COPY t (id, v) FROM STDIN", &data).unwrap();

    // Build the index over the existing rows — the streaming, chunked backfill.
    exec(engine, &mut session, "CREATE INDEX t_v ON t (v)");
    assert_eq!(
        count_index_entries(engine, "t_v"),
        BACKFILL_ROWS,
        "the backfill built an entry for every existing row, across chunk boundaries"
    );
}

/// `CREATE UNIQUE INDEX` on a column that already holds a duplicate is rejected by the backfill's
/// uniqueness check — even when the duplicate pair straddles two streamed chunks, since each chunk is
/// applied before the next is read.
#[test]
fn create_unique_index_rejects_an_existing_duplicate() {
    set_maintenance_work_mem(SMALL_MAINTENANCE_BUDGET);
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE t (id INT PRIMARY KEY, v INT)",
    );
    // Distinct v = i for every row except id 15000, which reuses id 5's value — a duplicate whose
    // two rows fall in different backfill chunks.
    let mut data = String::new();
    for i in 0..BACKFILL_ROWS {
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
        "CREATE UNIQUE INDEX over duplicate data is rejected: {err}"
    );
}

/// `COPY` resolves a schema qualifier, in both directions.
///
/// It went through a name helper that understood a bare name or `public.` and refused anything
/// else — so a user who could already `SELECT ... FROM app.t` had no way to export it, with the
/// working alternative in plain sight.
///
/// The target is shadowed by a same-named table in the default schema holding different rows, so
/// accepting the qualifier is not enough: it has to reach the right table in each direction.
#[test]
fn copy_resolves_a_schema_qualifier_in_both_directions() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE SCHEMA app");
    exec(engine, &mut session, "CREATE TABLE app.t (id INT NOT NULL)");
    exec(engine, &mut session, "CREATE TABLE t (id INT NOT NULL)");
    exec(engine, &mut session, "INSERT INTO t VALUES (999)");

    assert_eq!(
        copy(engine, "COPY app.t FROM STDIN", "1\n2\n").unwrap(),
        2,
        "COPY FROM refused or mis-resolved the qualifier"
    );

    // Reading back through the qualifier must show the loaded rows, not the shadowing table's.
    assert_eq!(copy_out(engine, "COPY app.t TO STDOUT"), "1\n2\n");
    // And the default-schema table must be untouched by a qualified load.
    assert_eq!(copy_out(engine, "COPY t TO STDOUT"), "999\n");
}

/// A name with more parts than `schema.table` stays rejected on the path just widened.
#[test]
fn copy_still_rejects_a_three_part_name() {
    // The message, not just the variant: `Error::Unsupported` alone would also be satisfied by
    // COPY-to-STDOUT support being withdrawn, and this test would go on claiming to pin name
    // arity.
    assert!(matches!(
        parse("COPY d.app.t TO STDOUT"),
        Err(Error::Unsupported(m)) if m.contains("more than two parts")
    ));
}
