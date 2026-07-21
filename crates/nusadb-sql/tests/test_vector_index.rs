//! `CREATE INDEX ... USING hnsw` + `ORDER BY col <=> q LIMIT k` routed to a vector search.
//! The planner emits a `VectorKnn` for the k-NN shape; the executor uses the declared HNSW index
//! (cached, approximate) when present and an exact scan otherwise — both return the k nearest rows in
//! ascending cosine-distance order. On a tiny index the HNSW search is exact, so the order is pinned.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Row, Session, analyze, parse, plan};

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

fn try_analyze(engine: &dyn StorageEngine, sql: &str) -> Result<(), Error> {
    analyze(parse(sql)?, &Cat(engine)).map(|_| ())
}

/// Run a query and return the `id` (first column) of each row, in result order (NOT sorted).
fn ids_in_order(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<i64> {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.iter()
        .map(|r: &Row| match r.first() {
            Some(Value::Int(i)) => *i,
            other => panic!("expected an Int id, got {other:?}"),
        })
        .collect()
}

fn seed(engine: &dyn StorageEngine, session: &mut Session) {
    exec(
        engine,
        session,
        "CREATE TABLE items (id INT NOT NULL, embedding VECTOR(3))",
    );
    // id1 = the query point; id4 is very close to it; id5 leans toward id2; id2/id3 are orthogonal.
    for (id, v) in [
        (1, "[1,0,0]"),
        (2, "[0,1,0]"),
        (3, "[0,0,1]"),
        (4, "[0.9,0.1,0]"),
        (5, "[0.1,0.9,0]"),
    ] {
        exec(
            engine,
            session,
            &format!("INSERT INTO items VALUES ({id}, '{v}'::VECTOR(3))"),
        );
    }
}

#[test]
fn hnsw_index_routes_knn_and_matches_exact() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    seed(engine, &mut session);

    let knn2 = "SELECT id FROM items ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";
    let knn3 = "SELECT id FROM items ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 3";

    // Before any index: the exact scan path returns the true nearest neighbours.
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);
    assert_eq!(ids_in_order(engine, &mut session, knn3), vec![1, 4, 5]);

    // Declare an HNSW index; the query now routes through it. On a 5-row index the search is exact,
    // so the order is identical to the brute-force scan above.
    exec(
        engine,
        &mut session,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);
    assert_eq!(ids_in_order(engine, &mut session, knn3), vec![1, 4, 5]);

    // A new row nearer the query than id4 must surface — the cache rebuilds when the table changes.
    exec(
        engine,
        &mut session,
        "INSERT INTO items VALUES (6, '[0.99,0.01,0]'::VECTOR(3))",
    );
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 6]);

    // A same-row-count UPDATE must still invalidate the cache (an MVCC update supersedes the row with
    // a new tid, changing the table signature): move id6 far from the query, so id4 returns instead.
    exec(
        engine,
        &mut session,
        "UPDATE items SET embedding = '[0,0,1]'::VECTOR(3) WHERE id = 6",
    );
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);

    // Dropping the index falls back to the exact scan — same answer.
    exec(engine, &mut session, "DROP INDEX items_emb");
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);
}

#[test]
fn filtered_knn_applies_where_and_matches_exact() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    seed(engine, &mut session);

    // Distances to [1,0,0]: id1 (0) < id4 < id5 < id2 = id3.
    let with_index = |session: &mut Session, sql: &str| ids_in_order(engine, session, sql);

    exec(
        engine,
        &mut session,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );

    // Excluding the nearest row leaves the next-nearest that pass the filter.
    let excl1 =
        "SELECT id FROM items WHERE id <> 1 ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";
    assert_eq!(with_index(&mut session, excl1), vec![4, 5]);

    // Excluding a non-nearest row from the middle.
    let excl4 =
        "SELECT id FROM items WHERE id <> 4 ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";
    assert_eq!(with_index(&mut session, excl4), vec![1, 5]);

    // A filter selective enough that fewer than k rows match returns just those (exact fallback).
    let only1 =
        "SELECT id FROM items WHERE id = 1 ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";
    assert_eq!(with_index(&mut session, only1), vec![1]);

    // A filter carrying a subquery is left on the exact pipeline (which resolves subqueries) rather
    // than routed — it must still return the correct nearest matching rows, not error.
    let sub = "SELECT id FROM items WHERE id IN (SELECT id FROM items WHERE id <> 1) \
         ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";
    assert_eq!(with_index(&mut session, sub), vec![4, 5]);

    // Dropping the index runs the same queries via the exact path — identical filtered answers.
    exec(engine, &mut session, "DROP INDEX items_emb");
    assert_eq!(ids_in_order(engine, &mut session, excl1), vec![4, 5]);
    assert_eq!(ids_in_order(engine, &mut session, only1), vec![1]);
}

#[test]
fn incremental_append_and_delete_stay_correct() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    seed(engine, &mut session);
    exec(
        engine,
        &mut session,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );

    let knn4 = "SELECT id FROM items ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 4";

    // Two appends (pure inserts) are folded into the cached graph incrementally; both surface.
    exec(
        engine,
        &mut session,
        "INSERT INTO items VALUES (6, '[0.95,0.05,0]'::VECTOR(3))",
    );
    exec(
        engine,
        &mut session,
        "INSERT INTO items VALUES (7, '[0.8,0.2,0]'::VECTOR(3))",
    );
    // Order by closeness to [1,0,0]: id1 (0) < id6 < id4 < id7.
    assert_eq!(ids_in_order(engine, &mut session, knn4), vec![1, 6, 4, 7]);

    // A DELETE removes a node, which the graph cannot do incrementally → full rebuild. id6 is gone.
    exec(engine, &mut session, "DELETE FROM items WHERE id = 6");
    assert_eq!(
        ids_in_order(
            engine,
            &mut session,
            "SELECT id FROM items ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 3"
        ),
        vec![1, 4, 7]
    );
}

#[test]
fn ef_search_hint_is_accepted_and_safe() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    seed(engine, &mut session);
    exec(
        engine,
        &mut session,
        "CREATE INDEX items_emb ON items USING hnsw (embedding)",
    );

    let knn2 = "SELECT id FROM items ORDER BY embedding <=> '[1,0,0]'::VECTOR(3) LIMIT 2";

    // A wider beam (higher recall, more work) does not change a correct result.
    exec(engine, &mut session, "SET hnsw_ef_search = 200");
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);

    // A beam below k is clamped up to k, so the query still returns k rows correctly.
    exec(engine, &mut session, "SET hnsw_ef_search = 1");
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);

    // Clearing the hint falls back to the default beam — still correct.
    exec(engine, &mut session, "SET hnsw_ef_search = 0");
    assert_eq!(ids_in_order(engine, &mut session, knn2), vec![1, 4]);
}

#[test]
fn hnsw_index_creation_is_validated() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE items (id INT NOT NULL, embedding VECTOR(3), label TEXT)",
    );

    // An unknown access method is rejected (only `hnsw` is supported).
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i ON items USING gin (embedding)"),
        Err(Error::Unsupported(_))
    ));
    // An hnsw index over a non-vector column is rejected.
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i ON items USING hnsw (label)"),
        Err(Error::Unsupported(_))
    ));
    // An hnsw index over more than one column is rejected.
    assert!(matches!(
        try_analyze(
            engine,
            "CREATE INDEX i ON items USING hnsw (embedding, label)"
        ),
        Err(Error::Unsupported(_))
    ));
    // A plain (B-tree) index is unaffected by the new `USING` surface.
    assert!(matches!(
        exec(engine, &mut session, "CREATE INDEX items_id ON items (id)"),
        ExecutionResult::IndexCreated
    ));
}
