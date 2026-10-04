//! `UPDATE ... FROM` and `DELETE ... USING` whose source is larger than `work_mem` spill it, given
//! a spill directory, and must change exactly the rows an unbounded run changes: the same rows, and
//! for an UPDATE with several matching source rows the same first one. The source mixes a key held
//! by over a thousand rows (larger than the budget at every split), keys spread over many
//! partitions, and `NULL` keys; conditions with no equality to key on read the source from disk.
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
            run(engine, &mut session, statement);
            run(engine, &mut session, "RESET work_mem");
            set_spill_config(None);
            tables.push(rows(engine, &mut session, "SELECT k, v, name FROM t"));
        }
        assert_eq!(tables[0], tables[1], "{statement}");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "{statement} left spill files behind"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
