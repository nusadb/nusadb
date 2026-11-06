//! A partial index as a scan path, through the production catalog adapters
//! ([`nusadb_sql::catalog_list_indexes`] / [`nusadb_sql::catalog_table_stats`]).
//!
//! Covered here, beside the plan shapes in `p3_filter/partial_index_scan.slt`:
//!
//! - snapshots: a row that leaves the predicate is still found through the index by every
//!   transaction that sees its old version (while the update is open, and by a REPEATABLE READ
//!   reader after it commits), and a rollback leaves it in place;
//! - a partial unique index whose row leaves it frees the key, in the same transaction and in the
//!   next, and the row may come back;
//! - cost: with statistics, a partial index that holds most of the table loses to a sequential scan,
//!   and one whose predicate is selective wins over a full index on the same key;
//! - the plan cache keeps a plan for a table whose only indexes are partial, and drops it when
//!   that table's index listing changes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, PlanCache, Session, analyze, ast::Value, parse,
    plan,
};

/// The production adapter shape: tables, indexes, and stats resolved from the engine.
struct Cat<'a> {
    engine: &'a BtreeEngine,
}

impl Catalog for Cat<'_> {
    nusadb_test_utils::catalog_answers_as_superuser!(nusadb_sql);
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.engine.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_list_indexes(self.engine, txn, table);
        let _ = self.engine.commit(txn);
        out
    }
    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        let txn = self
            .engine
            .begin(nusadb_core::IsolationLevel::ReadCommitted)
            .map_err(Error::from)?;
        let out = nusadb_sql::catalog_table_stats(self.engine, txn, table);
        let _ = self.engine.commit(txn);
        out
    }
}

fn run(
    engine: &'static BtreeEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat { engine })?;
    session.execute(plan(logical))
}

fn ok(engine: &'static BtreeEngine, session: &mut Session, sql: &str) {
    run(engine, session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn ids(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> Vec<i64> {
    match run(engine, session, sql).unwrap() {
        ExecutionResult::Rows { rows, .. } => rows
            .into_iter()
            .map(|row| match row.first() {
                Some(Value::Int(v)) => *v,
                other => panic!("expected an integer id, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn explain(engine: &'static BtreeEngine, session: &mut Session, sql: &str) -> String {
    match run(engine, session, &format!("EXPLAIN {sql}")).unwrap() {
        ExecutionResult::Rows { rows, .. } => rows
            .iter()
            .map(|row| match &row[..] {
                [Value::Text(line)] => line.clone(),
                other => panic!("unexpected EXPLAIN row {other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => panic!("expected rows, got {other:?}"),
    }
}

fn fresh() -> &'static BtreeEngine {
    Box::leak(Box::new(BtreeEngine::new()))
}

/// The claim-queue table: every tenth row pending, a partial index over the pending rows' due time.
fn queue(engine: &'static BtreeEngine, session: &mut Session) {
    ok(
        engine,
        session,
        "CREATE TABLE q (id INT PRIMARY KEY, status TEXT NOT NULL, due INT NOT NULL)",
    );
    ok(
        engine,
        session,
        "INSERT INTO q SELECT g, CASE WHEN g % 10 = 0 THEN 'pending' ELSE 'done' END, g \
         FROM generate_series(1, 200) g",
    );
    ok(
        engine,
        session,
        "CREATE INDEX q_due_pending ON q (due) WHERE status = 'pending'",
    );
}

const PENDING: &str = "SELECT id FROM q WHERE status = 'pending' AND due >= 0 ORDER BY due";
const PENDING_SEQ: &str =
    "SELECT id FROM q WHERE status || '' = 'pending' AND due + 0 >= 0 ORDER BY due";

/// A row leaving the predicate stays visible through the index to every snapshot that still sees
/// its old version, exactly as a sequential scan shows it.
#[test]
fn a_departed_row_stays_visible_to_older_snapshots() {
    let engine = fresh();
    let mut writer = Session::new(engine);
    let mut reader = Session::new(engine);
    queue(engine, &mut writer);
    assert!(explain(engine, &mut reader, PENDING).contains("IndexScan: q using q_due_pending"));
    assert!(explain(engine, &mut reader, PENDING_SEQ).contains("SeqScan"));
    let all: Vec<i64> = (1..=20).map(|i| i * 10).collect();
    assert_eq!(ids(engine, &mut reader, PENDING), all);

    ok(engine, &mut writer, "BEGIN");
    ok(
        engine,
        &mut writer,
        "UPDATE q SET status = 'done' WHERE id = 40",
    );
    // The writer no longer sees row 40 as pending.
    let without: Vec<i64> = all.iter().copied().filter(|&i| i != 40).collect();
    assert_eq!(ids(engine, &mut writer, PENDING), without);
    // Every other transaction still does, through the index as through a scan.
    assert_eq!(ids(engine, &mut reader, PENDING), all);
    ok(engine, &mut reader, "BEGIN");
    assert_eq!(ids(engine, &mut reader, PENDING), all);
    ok(engine, &mut reader, "COMMIT");
    ok(engine, &mut reader, "BEGIN ISOLATION LEVEL REPEATABLE READ");
    assert_eq!(ids(engine, &mut reader, PENDING), all);
    ok(engine, &mut writer, "COMMIT");
    // The REPEATABLE READ snapshot predates the commit: row 40 is still pending to it.
    assert_eq!(ids(engine, &mut reader, PENDING), all);
    assert_eq!(ids(engine, &mut reader, PENDING_SEQ), all);
    ok(engine, &mut reader, "COMMIT");
    // A new snapshot sees the commit.
    assert_eq!(ids(engine, &mut reader, PENDING), without);
    assert_eq!(ids(engine, &mut reader, PENDING_SEQ), without);
}

/// A rolled-back departure leaves the row in the index, and a row that comes back is found again.
#[test]
fn a_rolled_back_departure_and_a_return() {
    let engine = fresh();
    let mut s = Session::new(engine);
    queue(engine, &mut s);
    let all: Vec<i64> = (1..=20).map(|i| i * 10).collect();
    ok(engine, &mut s, "BEGIN");
    ok(engine, &mut s, "UPDATE q SET status = 'done' WHERE id = 40");
    ok(engine, &mut s, "ROLLBACK");
    assert_eq!(ids(engine, &mut s, PENDING), all);

    ok(engine, &mut s, "UPDATE q SET status = 'done' WHERE id = 40");
    ok(
        engine,
        &mut s,
        "UPDATE q SET status = 'pending' WHERE id = 40",
    );
    ok(
        engine,
        &mut s,
        "UPDATE q SET status = 'pending' WHERE id = 41",
    );
    let mut with_41 = all;
    with_41.insert(4, 41);
    assert_eq!(ids(engine, &mut s, PENDING), with_41);
    assert_eq!(ids(engine, &mut s, PENDING_SEQ), with_41);
}

/// A partial unique index: a row leaving the predicate frees its key for another row, in the same
/// transaction and in the next one, and the key is enforced again once the row returns.
#[test]
fn a_partial_unique_key_is_freed_by_a_departure() {
    let engine = fresh();
    let mut s = Session::new(engine);
    ok(
        engine,
        &mut s,
        "CREATE TABLE acct (id INT PRIMARY KEY, email TEXT, state TEXT NOT NULL)",
    );
    ok(
        engine,
        &mut s,
        "CREATE UNIQUE INDEX acct_live_email ON acct (email) WHERE state = 'live'",
    );
    ok(engine, &mut s, "INSERT INTO acct VALUES (1, 'a@x', 'live')");
    assert!(
        run(engine, &mut s, "INSERT INTO acct VALUES (2, 'a@x', 'live')").is_err(),
        "the key is taken"
    );
    let live = "SELECT id FROM acct WHERE email = 'a@x' AND state = 'live'";
    assert!(explain(engine, &mut s, live).contains("IndexScan: acct using acct_live_email"));

    ok(engine, &mut s, "BEGIN");
    ok(
        engine,
        &mut s,
        "UPDATE acct SET state = 'closed' WHERE id = 1",
    );
    ok(engine, &mut s, "INSERT INTO acct VALUES (2, 'a@x', 'live')");
    ok(engine, &mut s, "COMMIT");
    assert_eq!(ids(engine, &mut s, live), vec![2]);

    ok(
        engine,
        &mut s,
        "UPDATE acct SET state = 'closed' WHERE id = 2",
    );
    ok(engine, &mut s, "INSERT INTO acct VALUES (3, 'a@x', 'live')");
    assert_eq!(ids(engine, &mut s, live), vec![3]);
    assert!(
        run(
            engine,
            &mut s,
            "UPDATE acct SET state = 'live' WHERE id = 1"
        )
        .is_err(),
        "a returning row meets the key it left behind taken"
    );
    ok(
        engine,
        &mut s,
        "DELETE FROM acct WHERE email = 'a@x' AND state = 'live'",
    );
    ok(
        engine,
        &mut s,
        "UPDATE acct SET state = 'live' WHERE id = 1",
    );
    assert_eq!(ids(engine, &mut s, live), vec![1]);
    assert_eq!(
        ids(engine, &mut s, "SELECT id FROM acct ORDER BY id"),
        vec![1, 2]
    );
}

/// With statistics the planner weighs what a partial index holds: one holding nearly every row
/// loses to a sequential scan for a wide bound, a selective one wins over a full index on the same
/// key, and a selective bound takes the partial index as it would a full one.
#[test]
fn statistics_weigh_the_predicate() {
    let engine = fresh();
    let mut s = Session::new(engine);
    ok(
        engine,
        &mut s,
        "CREATE TABLE w (id INT NOT NULL, k INT NOT NULL, status TEXT NOT NULL)",
    );
    ok(
        engine,
        &mut s,
        "INSERT INTO w SELECT g, g % 100, CASE WHEN g % 100 = 0 THEN 'rare' ELSE 'common' END \
         FROM generate_series(1, 5000) g",
    );
    ok(
        engine,
        &mut s,
        "CREATE INDEX w_id_common ON w (id) WHERE status = 'common'",
    );
    ok(engine, &mut s, "CREATE INDEX w_k ON w (k)");
    ok(
        engine,
        &mut s,
        "CREATE INDEX w_k_rare ON w (k) WHERE status = 'rare'",
    );
    ok(engine, &mut s, "ANALYZE w");

    let wide = "SELECT id FROM w WHERE id > 10 AND status = 'common'";
    assert!(
        explain(engine, &mut s, wide).contains("SeqScan"),
        "{}",
        explain(engine, &mut s, wide)
    );
    let narrow = "SELECT id FROM w WHERE id < 30 AND status = 'common'";
    assert!(
        explain(engine, &mut s, narrow).contains("IndexScan: w using w_id_common"),
        "{}",
        explain(engine, &mut s, narrow)
    );
    let rare = "SELECT id FROM w WHERE k > 50 AND status = 'rare'";
    assert!(
        explain(engine, &mut s, rare).contains("IndexScan: w using w_k_rare"),
        "{}",
        explain(engine, &mut s, rare)
    );
    let common = "SELECT id FROM w WHERE k = 7 AND status = 'common'";
    assert!(
        explain(engine, &mut s, common).contains("IndexScan: w using w_k"),
        "{}",
        explain(engine, &mut s, common)
    );
    // Same rows as a scan the planner cannot narrow.
    for (query, scan) in [
        (
            wide,
            "SELECT id FROM w WHERE id + 0 > 10 AND status || '' = 'common'",
        ),
        (
            narrow,
            "SELECT id FROM w WHERE id + 0 < 30 AND status || '' = 'common'",
        ),
        (
            rare,
            "SELECT id FROM w WHERE k + 0 > 50 AND status || '' = 'rare'",
        ),
    ] {
        let mut a = ids(engine, &mut s, query);
        let mut b = ids(engine, &mut s, scan);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "{query}");
    }
}

/// A plan over a table whose only indexes are partial is cached, and it is discarded as soon as
/// the table's index listing changes (index DDL moves no schema version, so the cache lists the
/// indexes again on each hit): a `DROP INDEX`, or one recreated under the same name with another
/// predicate, never leaves a stale plan. A table with an index over every row is not cached.
#[test]
fn the_plan_cache_rechecks_a_table_with_only_partial_indexes() {
    let engine = fresh();
    let mut s = Session::new(engine);
    ok(
        engine,
        &mut s,
        "CREATE TABLE c (id INT NOT NULL, status TEXT NOT NULL)",
    );
    ok(
        engine,
        &mut s,
        "INSERT INTO c SELECT g, CASE WHEN g % 2 = 0 THEN 'on' ELSE 'off' END \
         FROM generate_series(1, 20) g",
    );
    let sql = "SELECT id FROM c WHERE id = 4 AND status = 'on'";
    let run_cached = |cache: &mut PlanCache, s: &mut Session| -> Vec<i64> {
        let planned =
            nusadb_sql::plan_cached(cache, sql, parse(sql).unwrap(), &Cat { engine }, engine)
                .unwrap();
        match s.execute(planned).unwrap() {
            ExecutionResult::Rows { rows, .. } => rows
                .into_iter()
                .map(|row| match row.first() {
                    Some(Value::Int(v)) => *v,
                    other => panic!("expected an integer id, got {other:?}"),
                })
                .collect(),
            other => panic!("expected rows, got {other:?}"),
        }
    };
    let mut cache = PlanCache::new();
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(cache.hits(), 1, "a plan over an index-free table is cached");

    ok(
        engine,
        &mut s,
        "CREATE INDEX c_on ON c (id) WHERE status = 'on'",
    );
    assert!(explain(engine, &mut s, sql).contains("IndexScan: c using c_on"));
    let mut cache = PlanCache::new();
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(
        cache.hits(),
        1,
        "a plan over only partial indexes is cached"
    );

    // Dropped: the listing changed, so the plan through `c_on` is not served.
    ok(engine, &mut s, "DROP INDEX c_on");
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!((cache.hits(), cache.misses()), (1, 2));

    // Recreated under the same name with a predicate the query does not imply.
    let mut cache = PlanCache::new();
    ok(
        engine,
        &mut s,
        "CREATE INDEX c_on ON c (id) WHERE status = 'on'",
    );
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    ok(engine, &mut s, "DROP INDEX c_on");
    ok(
        engine,
        &mut s,
        "CREATE INDEX c_on ON c (id) WHERE status = 'off'",
    );
    assert!(!explain(engine, &mut s, sql).contains("using c_on"));
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!((cache.hits(), cache.misses()), (0, 2));

    // An index over every row: not cached.
    ok(engine, &mut s, "CREATE INDEX c_id ON c (id)");
    let mut cache = PlanCache::new();
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(run_cached(&mut cache, &mut s), vec![4]);
    assert_eq!(
        cache.hits(),
        0,
        "a plan that sees a full index is not cached"
    );
}
