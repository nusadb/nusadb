//! `UPDATE ... FROM`, `DELETE ... USING` and `MERGE` whose source is larger than `work_mem` spill
//! it, given a spill directory, and must end exactly as an unbounded run ends: the same rows
//! changed (for an UPDATE with several matching source rows, by the same first one), or the same
//! error. The source mixes a key held by over a thousand rows (larger than the budget at every
//! split), keys spread over many partitions, and `NULL` keys; conditions with no equality to key on
//! read the source from disk.
//!
//! `spill_config` is process-wide, so this binary holds a single test.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, Error, ExecutionResult, IndexInfo, Session, SpillConfig, analyze, parse, plan,
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

fn run(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(
        parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}")),
        &Cat(engine),
    )
    .unwrap_or_else(|e| panic!("{sql}: {e}"));
    session
        .execute(plan(logical))
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn try_run(
    engine: &dyn StorageEngine,
    session: &mut Session,
    sql: &str,
) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &Cat(engine))?;
    session.execute(plan(logical))
}

fn rows(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    let ExecutionResult::Rows { mut rows, .. } = run(engine, session, sql) else {
        panic!("expected rows from: {sql}");
    };
    rows.sort_by_key(|r| format!("{r:?}"));
    rows
}

const STATEMENTS: &[&str] = &[
    "UPDATE t SET v = s.v FROM s WHERE t.k = s.k",
    "UPDATE t SET v = s.v + t.v, name = s.name FROM s WHERE t.k = s.k AND s.v > 300",
    "UPDATE t SET v = s.v FROM s WHERE t.name = s.name AND t.k = s.k",
    "UPDATE t SET v = s.v FROM (SELECT k, v, name FROM s WHERE v % 3 = 0) AS s WHERE t.k = s.k",
    "UPDATE t SET v = s.v FROM s WHERE t.k + 1 = s.k + 1 AND t.name <> s.name",
    "UPDATE t SET v = -s.v FROM s WHERE t.k < s.k AND s.v = 21",
    "DELETE FROM t USING s WHERE t.k = s.k AND s.v < 200",
    "DELETE FROM t USING s WHERE t.name = s.name AND t.k = s.k",
    "DELETE FROM t USING (SELECT k FROM s WHERE v % 7 = 1) AS s WHERE t.k = s.k",
    "DELETE FROM t USING s WHERE t.v > s.v + 990",
    // MERGE over a source with one row per key (a repeated key would affect a row twice).
    "MERGE INTO t USING (SELECT DISTINCT ON (k) * FROM s ORDER BY k, v) AS s ON t.k = s.k \
     WHEN MATCHED AND s.v > 500 THEN UPDATE SET v = s.v WHEN MATCHED THEN DELETE \
     WHEN NOT MATCHED THEN INSERT (k, v, name) VALUES (s.k, s.v, s.name)",
    "MERGE INTO t USING (SELECT DISTINCT ON (k) * FROM s ORDER BY k, v DESC) AS s ON t.k = s.k AND t.name = s.name \
     WHEN MATCHED THEN UPDATE SET v = s.v + 1 \
     WHEN NOT MATCHED BY SOURCE AND t.v % 3 = 0 THEN DELETE \
     WHEN NOT MATCHED BY SOURCE THEN UPDATE SET v = -1",
    "MERGE INTO t USING (SELECT k, v, name FROM s WHERE v % 50 = 0) AS s ON t.k + 1000 = s.v \
     WHEN MATCHED THEN UPDATE SET name = s.name WHEN NOT MATCHED THEN INSERT (k, v) VALUES (s.k, s.v) \
     WHEN NOT MATCHED BY SOURCE AND t.k = 7 THEN DELETE",
    "MERGE INTO t USING (SELECT DISTINCT ON (v) * FROM s ORDER BY v, k) AS s ON t.v > s.v * 2 AND t.v <= s.v * 2 + 2 AND t.name = 'n1' \
     WHEN MATCHED THEN UPDATE SET v = 0 WHEN NOT MATCHED BY SOURCE AND t.v % 7 = 0 THEN DELETE",
    // Each source row matches several target rows (and no target two source rows): it takes the
    // first in scan order.
    "MERGE INTO t USING (SELECT DISTINCT ON (v) *, repeat('x', 400) AS pad FROM s WHERE v % 10 = 0 ORDER BY v, k) AS s \
     ON t.v > s.v * 2 AND t.v <= s.v * 2 + 6 \
     WHEN MATCHED THEN UPDATE SET v = s.v + t.v WHEN NOT MATCHED BY SOURCE AND t.v % 9 = 0 THEN DELETE",
    // A source key that matches many target rows: the source row takes its first one; and a
    // source with repeated keys, which must fail the same way.
    "MERGE INTO t USING (SELECT DISTINCT ON (k) * FROM s ORDER BY k, v) AS s ON t.k = s.k \
     WHEN MATCHED AND t.k = 7 THEN UPDATE SET v = s.v + t.v",
    // One key larger than the budget at every split, its rows told apart by the rest of ON.
    "MERGE INTO t USING (SELECT DISTINCT ON (v) *, repeat('x', 400) AS pad FROM s WHERE k = 7 AND v % 10 = 0 ORDER BY v, name) AS s \
     ON t.k = s.k AND t.v > s.v * 2 AND t.v <= s.v * 2 + 15 \
     WHEN MATCHED THEN UPDATE SET name = s.name \
     WHEN NOT MATCHED AND s.v % 4 = 0 THEN INSERT (k, v) VALUES (s.k, s.v) \
     WHEN NOT MATCHED BY SOURCE AND t.k = 7 AND t.v % 3 = 0 THEN DELETE",
    "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET v = s.v",
];

fn setup(engine: &dyn StorageEngine, session: &mut Session) {
    run(engine, session, "CREATE TABLE t (k INT, v INT, name TEXT)");
    run(engine, session, "CREATE TABLE s (k INT, v INT, name TEXT)");
    let target = (0..2000_i64)
        .map(|i| {
            let k = if i % 29 == 0 {
                "NULL".to_owned()
            } else if i % 5 == 0 {
                "7".to_owned()
            } else {
                (i % 700).to_string()
            };
            format!("({k}, {i}, 'n{}')", i % 6)
        })
        .collect::<Vec<_>>()
        .join(",");
    run(engine, session, &format!("INSERT INTO t VALUES {target}"));
    // Key 7 takes every third source row (over a thousand rows); the rest spread over 900 keys.
    for start in (0..3600_i64).step_by(600) {
        let source = (start..start + 600)
            .map(|i| {
                let k = if i % 31 == 0 {
                    "NULL".to_owned()
                } else if i % 3 == 0 {
                    "7".to_owned()
                } else {
                    (i % 900).to_string()
                };
                format!("({k}, {}, 'n{}-padding-padding')", i * 7 % 1000, i % 4)
            })
            .collect::<Vec<_>>()
            .join(",");
        run(engine, session, &format!("INSERT INTO s VALUES {source}"));
    }
    // Names that match `t.name`, so the name-keyed statements find partners.
    run(
        engine,
        session,
        "UPDATE s SET name = 'n' || (v % 6)::TEXT WHERE v % 2 = 0",
    );
}

#[test]
fn spilled_dml_joins_change_exactly_what_unbounded_ones_do() {
    let dir = std::env::temp_dir().join(format!("nusadb-dml-spill-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for statement in STATEMENTS {
        let mut tables = Vec::new();
        for bounded in [false, true] {
            let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
            let mut session = Session::new(engine);
            setup(engine, &mut session);
            if bounded {
                set_spill_config(Some(SpillConfig {
                    dir: dir.clone(),
                    threshold_bytes: 64 * 1024 * 1024,
                }));
                run(engine, &mut session, "SET work_mem = '16kB'");
            }
            let outcome = try_run(engine, &mut session, statement);
            run(engine, &mut session, "RESET work_mem");
            set_spill_config(None);
            tables.push(match outcome {
                Ok(_) => Ok(rows(engine, &mut session, "SELECT k, v, name FROM t")),
                Err(e) => Err(e.to_string()),
            });
        }
        assert_eq!(tables[0], tables[1], "{statement}");
        if statement.ends_with("SET v = s.v") {
            assert!(tables[0].is_err(), "{statement} should affect a row twice");
        } else {
            assert!(tables[0].is_ok(), "{statement}: {:?}", tables[0]);
        }
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "{statement} left spill files behind"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
