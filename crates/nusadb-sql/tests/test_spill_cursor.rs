//! A cursor whose result is larger than `work_mem` keeps its rows in a spill file, given a spill
//! directory, and every `FETCH` direction must return exactly the rows the same cursor held in
//! memory returns. Closing the cursor removes the file.
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

fn fetched(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<Vec<Value>> {
    match run(engine, session, sql) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("{sql}: expected rows, got {other:?}"),
    }
}

/// Cursor files left in `dir`.
fn cursor_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("nusadb-spill-cursor-")
        })
        .count()
}

const FETCHES: &[&str] = &[
    "FETCH 3 FROM cur",
    "FETCH LAST FROM cur",
    "FETCH PRIOR FROM cur",
    "FETCH ABSOLUTE 2500 FROM cur",
    "FETCH BACKWARD 4 FROM cur",
    "FETCH ABSOLUTE 7 FROM cur",
    "FETCH RELATIVE 100 FROM cur",
    "FETCH FORWARD 2000 FROM cur",
    "FETCH NEXT FROM cur",
    "FETCH ABSOLUTE 0 FROM cur",
    "FETCH FIRST FROM cur",
    "FETCH ABSOLUTE 99999 FROM cur",
    "FETCH BACKWARD 2 FROM cur",
    "FETCH FORWARD ALL FROM cur",
    "FETCH BACKWARD ALL FROM cur",
    "FETCH NEXT FROM cur",
];

#[test]
fn a_cursor_larger_than_work_mem_fetches_every_direction_from_disk() {
    let dir = std::env::temp_dir().join(format!("nusadb-cursor-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut results = Vec::new();
    for bounded in [false, true] {
        let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
        let mut session = Session::new(engine);
        run(engine, &mut session, "CREATE TABLE t (id INT, s TEXT)");
        for start in (0..5000).step_by(500) {
            let values = (start..start + 500)
                .map(|i: i64| format!("({i}, 'row-{:05}-padding')", (i * 7919) % 5000))
                .collect::<Vec<_>>()
                .join(",");
            run(
                engine,
                &mut session,
                &format!("INSERT INTO t VALUES {values}"),
            );
        }
        if bounded {
            set_spill_config(Some(SpillConfig {
                dir: dir.clone(),
                threshold_bytes: 64 * 1024 * 1024,
            }));
            run(engine, &mut session, "SET work_mem = '16kB'");
        }
        run(engine, &mut session, "BEGIN");
        run(
            engine,
            &mut session,
            "DECLARE cur SCROLL CURSOR FOR SELECT id, s FROM t WHERE id % 3 <> 1 ORDER BY s, id",
        );
        if bounded {
            assert_eq!(cursor_files(&dir), 2, "the cursor's rows went to disk");
        }
        let mut got = Vec::new();
        for fetch in FETCHES {
            got.push(fetched(engine, &mut session, fetch));
        }
        run(engine, &mut session, "CLOSE cur");
        run(engine, &mut session, "COMMIT");
        assert_eq!(cursor_files(&dir), 0, "closing the cursor removes its file");
        set_spill_config(None);
        results.push(got);
    }
    let [memory, spilled] = results.as_slice() else {
        panic!("two runs expected");
    };
    assert!(memory.iter().any(|rows| rows.len() > 1000));
    for ((fetch, want), got) in FETCHES.iter().zip(memory).zip(spilled) {
        assert_eq!(got, want, "{fetch}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
