//! End-to-end spill of a `VECTOR` column (the design finding on `executor/spill/codec.rs` tag-15). The
//! spill codec's `Vector` path is exercised in production when a vector workload spills under a
//! tight memory budget (the 1 vCPU / 1 GB target), yet only the byte round-trip was covered. This
//! drives a real spilling `ORDER BY` whose rows carry `VECTOR(3)` values through `encode_row` /
//! `decode_row`, and asserts the result equals the in-memory path.
//!
//! `spill_config` is process-wide, so this lives in its own test binary and resets to `None`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Row, Session, SpillConfig, analyze, parse, plan,
    set_spill_config,
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

#[test]
fn spilling_order_by_round_trips_vector_column_then_resets() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(
        engine,
        &mut session,
        "CREATE TABLE items (id INT NOT NULL, embedding VECTOR(3))",
    );
    // Enough wide rows that a tiny budget spills to many sorted runs — each carrying a VECTOR(3)
    // value that must survive `encode_row`/`decode_row`.
    for i in 0..200 {
        let (a, b, c) = (i % 7, (i * 3) % 11, i % 5);
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO items VALUES ({i}, '[{a},{b},{c}]')"),
        );
    }

    // `ORDER BY id` gives a total order, so the result sequence is deterministic across paths —
    // a stronger oracle than a multiset comparison.
    let sql = "SELECT id, embedding FROM items ORDER BY id";

    set_spill_config(None);
    let baseline = rows(engine, &mut session, sql);
    assert_eq!(baseline.len(), 200);

    let dir = std::env::temp_dir();
    // Tiny budget → external merge sort spills Vector-bearing rows to disk and merges them back.
    set_spill_config(Some(SpillConfig {
        dir,
        threshold_bytes: 64,
    }));
    assert_eq!(
        rows(engine, &mut session, sql),
        baseline,
        "spilled ORDER BY carrying a VECTOR column must match the in-memory result"
    );

    set_spill_config(None);
}
