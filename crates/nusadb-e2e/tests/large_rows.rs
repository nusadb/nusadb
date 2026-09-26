//! Rows larger than a page through SQL: stored whole, read back whole, updated and deleted like
//! any other row.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, parse, plan};

struct Cat<'a>(&'a dyn StorageEngine);
impl Catalog for Cat<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        Ok(Vec::new())
    }
}

fn exec(engine: &BtreeEngine, session: &mut Session, sql: &str) -> ExecutionResult {
    let logical = analyze(parse(sql).unwrap(), &Cat(engine))
        .unwrap_or_else(|e| panic!("`{}` should analyze: {e}", &sql[..sql.len().min(80)]));
    session
        .execute(plan(logical))
        .unwrap_or_else(|e| panic!("`{}` should succeed: {e}", &sql[..sql.len().min(80)]))
}

fn texts(engine: &BtreeEngine, session: &mut Session, sql: &str) -> Vec<String> {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("expected rows");
    };
    rows.into_iter()
        .map(|r| match r.into_iter().next() {
            Some(Value::Text(s)) => s,
            other => panic!("expected text, got {other:?}"),
        })
        .collect()
}

#[test]
fn a_text_value_far_larger_than_a_page_round_trips_through_sql() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    exec(
        &engine,
        &mut session,
        "CREATE TABLE docs (id INT PRIMARY KEY, body TEXT)",
    );
    // 200 KiB of text, with structure so a reassembly mistake shows.
    let body = (0..200 * 1024 / 8).fold(String::new(), |mut acc, i| {
        use std::fmt::Write as _;
        let _ = writeln!(acc, "{i:07}");
        acc
    });
    exec(
        &engine,
        &mut session,
        &format!("INSERT INTO docs VALUES (1, '{body}'), (2, 'short')"),
    );
    let got = texts(&engine, &mut session, "SELECT body FROM docs WHERE id = 1");
    assert_eq!(got, vec![body.clone()]);
    assert_eq!(
        texts(
            &engine,
            &mut session,
            "SELECT length(body)::text FROM docs ORDER BY id"
        ),
        vec![body.len().to_string(), "5".to_owned()]
    );

    // Grow it, then shrink it back below a page; delete it; the table keeps working.
    let bigger = format!("{body}{body}");
    exec(
        &engine,
        &mut session,
        &format!("UPDATE docs SET body = '{bigger}' WHERE id = 1"),
    );
    assert_eq!(
        texts(&engine, &mut session, "SELECT body FROM docs WHERE id = 1"),
        vec![bigger]
    );
    exec(
        &engine,
        &mut session,
        "UPDATE docs SET body = 'tiny' WHERE id = 1",
    );
    assert_eq!(
        texts(&engine, &mut session, "SELECT body FROM docs ORDER BY id"),
        vec!["tiny".to_owned(), "short".to_owned()]
    );
    exec(&engine, &mut session, "DELETE FROM docs WHERE id = 1");
    assert_eq!(
        texts(&engine, &mut session, "SELECT body FROM docs"),
        vec!["short".to_owned()]
    );
}
