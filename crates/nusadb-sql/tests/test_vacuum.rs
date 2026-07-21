//! `VACUUM [FULL] [ANALYZE]` executor behavior. Plain `VACUUM` and `VACUUM FULL` reclaim
//! dead row versions; `VACUUM ANALYZE` additionally recomputes statistics for every user table —
//! proven by checking the engine has table stats only after `VACUUM ANALYZE` runs.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

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

#[test]
fn vacuum_reclaims_and_analyze_recomputes_stats() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..30 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({i}, 'x')"),
        );
    }
    let table_id = engine.lookup_table("t").unwrap().unwrap().id;

    // Plain VACUUM and VACUUM FULL both succeed (reclaim path).
    assert!(matches!(
        exec(engine, &mut session, "VACUUM"),
        ExecutionResult::Vacuumed(_)
    ));
    assert!(matches!(
        exec(engine, &mut session, "VACUUM FULL"),
        ExecutionResult::Vacuumed(_)
    ));

    // No stats yet — neither VACUUM nor VACUUM FULL analyzes.
    assert!(
        engine.table_stats(table_id).unwrap().is_none(),
        "stats must not exist before VACUUM ANALYZE"
    );

    // VACUUM ANALYZE recomputes stats for every user table.
    assert!(matches!(
        exec(engine, &mut session, "VACUUM ANALYZE"),
        ExecutionResult::Vacuumed(_)
    ));
    let stats = engine
        .table_stats(table_id)
        .unwrap()
        .expect("VACUUM ANALYZE must populate table stats");
    assert_eq!(
        stats.row_count, 30,
        "analyzed row count must match the table"
    );
    assert_eq!(stats.columns.len(), 2, "both columns must be analyzed");
}
