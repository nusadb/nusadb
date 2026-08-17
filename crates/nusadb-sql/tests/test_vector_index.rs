//! `CREATE INDEX ... USING hnsw` + `ORDER BY col <=> q LIMIT k` routed to a vector search.
//! The planner emits a `VectorKnn` for the k-NN shape; the executor uses the declared HNSW index
//! (cached, approximate) when it was built under that same metric and an exact scan otherwise — both
//! return the k nearest rows in
//! ascending distance order under the metric the query's operator names. On a tiny index the HNSW search is exact, so the order is pinned.

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

    // An unknown access method really is a feature NusaDB has not built: `0A000`.
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i ON items USING gin (embedding)"),
        Err(Error::Unsupported(_))
    ));
    // The two below are the caller's mistake, not a gap — `hnsw` exists and these calls misuse it —
    // so they report `42601` rather than telling a migration tool the feature is missing.
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i ON items USING hnsw (label)"),
        Err(Error::InvalidStatement(_))
    ));
    assert!(matches!(
        try_analyze(
            engine,
            "CREATE INDEX i ON items USING hnsw (embedding, label)"
        ),
        Err(Error::InvalidStatement(_))
    ));
    // A plain (B-tree) index is unaffected by the new `USING` surface.
    assert!(matches!(
        exec(engine, &mut session, "CREATE INDEX items_id ON items (id)"),
        ExecutionResult::IndexCreated
    ));
}

/// An index answers exactly the distance it was built under.
///
/// The nearest neighbours under L2 are not the nearest under cosine — cosine ignores magnitude, L2 is
/// dominated by it. So a `<->` query must not be served from a cosine graph. This pins the behaviour
/// on data engineered to tell the two apart: `far` points the same *direction* as the query (cosine
/// distance 0, but a long way off in space), while `near` sits close in space but off-axis. Cosine
/// ranks `far` first, L2 ranks `near` first.
#[test]
fn a_query_is_never_answered_from_another_metrics_graph() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE pts (id INT NOT NULL, v VECTOR(2))",
    );
    for (id, v) in [(1, "[10,0]"), (2, "[1,1]")] {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO pts VALUES ({id}, '{v}')"),
        );
    }
    let cosine_first = "SELECT id FROM pts ORDER BY v <=> '[1,0]'::VECTOR(2) LIMIT 1";
    let l2_first = "SELECT id FROM pts ORDER BY v <-> '[1,0]'::VECTOR(2) LIMIT 1";

    // With no index at all, both are exact scans — this is the ground truth the index must not change.
    assert_eq!(ids_in_order(engine, &mut session, cosine_first), vec![1]);
    assert_eq!(ids_in_order(engine, &mut session, l2_first), vec![2]);

    // Declaring a cosine index must leave the L2 answer alone. Before the metric was recorded, the
    // `<->` query would have been handed the cosine graph and confidently returned id 1.
    exec(
        engine,
        &mut session,
        "CREATE INDEX pts_cos ON pts USING hnsw (v vector_cosine_ops)",
    );
    assert_eq!(ids_in_order(engine, &mut session, cosine_first), vec![1]);
    assert_eq!(
        ids_in_order(engine, &mut session, l2_first),
        vec![2],
        "an L2 query must not be answered from a cosine graph"
    );
}

/// Each operator class selects its own metric, an unknown one is rejected rather than quietly
/// treated as the default, and omitting it keeps the cosine an existing index already had.
#[test]
fn operator_class_selects_the_metric() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(
        engine,
        &mut session,
        "CREATE TABLE v (id INT NOT NULL, e VECTOR(2))",
    );
    for name in [
        "vector_l2_ops",
        "vector_cosine_ops",
        "vector_ip_ops",
        "vector_l1_ops",
    ] {
        assert!(
            matches!(
                exec(
                    engine,
                    &mut session,
                    &format!("CREATE INDEX i_{name} ON v USING hnsw (e {name})"),
                ),
                ExecutionResult::IndexCreated
            ),
            "{name} should select a metric"
        );
    }
    // Omitting the operator class keeps the historical default.
    assert!(matches!(
        exec(
            engine,
            &mut session,
            "CREATE INDEX i_def ON v USING hnsw (e)"
        ),
        ExecutionResult::IndexCreated
    ));
    // An unrecognized class is refused — accepting it and silently building a cosine graph is exactly
    // the trap this whole change removes.
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i_bad ON v USING hnsw (e text_ops)"),
        Err(Error::ObjectNotFound(_))
    ));
    // An operator class on a plain B-tree index is still refused.
    assert!(matches!(
        try_analyze(engine, "CREATE INDEX i_bt ON v (id int4_ops)"),
        Err(Error::Unsupported(_))
    ));
}

/// Two indexes on the *same* column under different metrics must each answer only their own queries.
///
/// This is the case that broke the first attempt at metric routing: the built-graph cache was keyed
/// by `(engine, table, column)` alone, so both indexes shared one slot and whichever was created last
/// silently answered for both. Checking the query's metric against the *catalog* entry was not
/// enough — the catalog and the cached graph were independent. Both creation orders are exercised,
/// because only the second index was previously wrong.
#[test]
fn two_metrics_on_one_column_do_not_share_a_graph() {
    for cosine_first in [true, false] {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        exec(
            engine,
            &mut session,
            "CREATE TABLE pts (id INT NOT NULL, v VECTOR(2))",
        );
        for (id, v) in [(1, "[10,0]"), (2, "[1,1]")] {
            exec(
                engine,
                &mut session,
                &format!("INSERT INTO pts VALUES ({id}, '{v}')"),
            );
        }
        let creates = [
            "CREATE INDEX pts_cos ON pts USING hnsw (v vector_cosine_ops)",
            "CREATE INDEX pts_l2 ON pts USING hnsw (v vector_l2_ops)",
        ];
        for sql in if cosine_first {
            creates
        } else {
            [creates[1], creates[0]]
        } {
            exec(engine, &mut session, sql);
        }
        // An INSERT drives the incremental-maintenance path, which previously wrote one index's graph
        // into the other's persisted blob.
        exec(engine, &mut session, "INSERT INTO pts VALUES (3, '[9,0]')");

        assert_eq!(
            ids_in_order(
                engine,
                &mut session,
                "SELECT id FROM pts ORDER BY v <=> '[1,0]'::VECTOR(2) LIMIT 1"
            ),
            vec![1],
            "cosine query, cosine_first={cosine_first}"
        );
        assert_eq!(
            ids_in_order(
                engine,
                &mut session,
                "SELECT id FROM pts ORDER BY v <-> '[1,0]'::VECTOR(2) LIMIT 1"
            ),
            vec![2],
            "L2 query must not be served from the cosine graph, cosine_first={cosine_first}"
        );
    }
}

/// A row whose vector is NULL has no distance, but `ORDER BY` puts it last rather than dropping it —
/// so the k-NN operator that stands in for that `Sort` + `Limit` must do the same, for every
/// operator, with and without an index.
///
/// This is a regression pin: widening the routing beyond `<=>` made three more operators reach the
/// k-NN path, which had been silently discarding these rows.
#[test]
fn a_null_vector_row_is_ranked_last_not_dropped() {
    for index_sql in [
        None,
        Some("CREATE INDEX p_l2 ON p USING hnsw (v vector_l2_ops)"),
    ] {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        exec(
            engine,
            &mut session,
            "CREATE TABLE p (id INT NOT NULL, v VECTOR(2))",
        );
        exec(engine, &mut session, "INSERT INTO p VALUES (1, '[1,0]')");
        exec(engine, &mut session, "INSERT INTO p VALUES (2, NULL)");
        exec(engine, &mut session, "INSERT INTO p VALUES (3, '[0,1]')");
        if let Some(sql) = index_sql {
            exec(engine, &mut session, sql);
        }
        for op in ["<->", "<#>", "<+>", "<=>"] {
            let got = ids_in_order(
                engine,
                &mut session,
                &format!("SELECT id FROM p ORDER BY v {op} '[1,0]'::VECTOR(2) LIMIT 5"),
            );
            assert_eq!(
                got.len(),
                3,
                "{op} dropped the NULL-vector row (indexed={})",
                index_sql.is_some()
            );
            assert_eq!(
                got[2],
                2,
                "{op} must rank the NULL-vector row last (indexed={})",
                index_sql.is_some()
            );
        }
    }
}
