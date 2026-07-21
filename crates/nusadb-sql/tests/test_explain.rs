//! `EXPLAIN [ANALYZE] [VERBOSE]`. Plain `EXPLAIN` formats the plan without running it;
//! `VERBOSE` appends the output columns; `ANALYZE` additionally executes a read-only statement and
//! reports the real row count + total time. `ANALYZE` on a data-modifying statement is rejected
//! (its side effects would commit before the result is returned), and a non-text `FORMAT` is rejected
//! at parse time (structured output is a follow-up).

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

/// The `plan` column text lines of an EXPLAIN result.
fn explain_lines(engine: &dyn StorageEngine, session: &mut Session, sql: &str) -> Vec<String> {
    let ExecutionResult::Rows { rows, .. } = exec(engine, session, sql) else {
        panic!("EXPLAIN must return rows: {sql}");
    };
    rows.into_iter()
        .map(|r| match r.into_iter().next() {
            Some(nusadb_sql::ast::Value::Text(s)) => s,
            other => panic!("expected text plan line, got {other:?}"),
        })
        .collect()
}

#[test]
fn explain_variants() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..20 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({}, 'x')", i % 4),
        );
    }

    // Plain EXPLAIN: a non-empty plan, and it must NOT have executed (no Execution line, no Output).
    let plain = explain_lines(engine, &mut session, "EXPLAIN SELECT a FROM t WHERE a > 1");
    assert!(!plain.is_empty());
    assert!(!plain.iter().any(|l| l.starts_with("Execution:")));
    assert!(!plain.iter().any(|l| l.starts_with("Output:")));

    // VERBOSE: appends the output columns, still no execution.
    let verbose = explain_lines(engine, &mut session, "EXPLAIN VERBOSE SELECT a, b FROM t");
    assert!(
        verbose.iter().any(|l| l == "Output: a, b"),
        "VERBOSE must list output columns, got {verbose:?}"
    );
    assert!(!verbose.iter().any(|l| l.starts_with("Execution:")));

    // ANALYZE: actually runs the read-only query and reports the real row count. `a > 1` keeps
    // values 2 and 3 → 10 of the 20 rows.
    let analyzed = explain_lines(
        engine,
        &mut session,
        "EXPLAIN ANALYZE SELECT a FROM t WHERE a > 1",
    );
    let exec_line = analyzed
        .iter()
        .find(|l| l.starts_with("Execution:"))
        .expect("ANALYZE must report execution");
    assert!(
        exec_line.contains("actual rows=10"),
        "ANALYZE row count wrong: {exec_line}"
    );

    // ANALYZE + VERBOSE compose.
    let both = explain_lines(
        engine,
        &mut session,
        "EXPLAIN ANALYZE VERBOSE SELECT a FROM t",
    );
    assert!(both.iter().any(|l| l == "Output: a"));
    assert!(
        both.iter()
            .any(|l| l.starts_with("Execution: actual rows=20"))
    );
}

#[test]
fn explain_analyze_reports_per_node_actual_rows() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);

    exec(engine, &mut session, "CREATE TABLE t (a INT, b TEXT)");
    for i in 0..20 {
        exec(
            engine,
            &mut session,
            &format!("INSERT INTO t VALUES ({}, 'x')", i % 4),
        );
    }

    // Per-node actuals: every operator line carries the rows it actually produced. The
    // scan reads all 20 rows; `a > 1` keeps values 2 and 3 → 10 rows through the filter and out.
    let analyzed = explain_lines(
        engine,
        &mut session,
        "EXPLAIN ANALYZE SELECT a FROM t WHERE a > 1",
    );
    let node_lines: Vec<&String> = analyzed
        .iter()
        .take_while(|l| !l.starts_with("Execution:"))
        .collect();
    assert!(
        node_lines
            .iter()
            .all(|l| l.contains("(actual rows=") || l.contains("(never executed)")),
        "every plan node must be annotated: {analyzed:?}"
    );
    let scan = node_lines
        .iter()
        .find(|l| l.contains("SeqScan"))
        .expect("plan must contain the scan");
    assert!(
        scan.contains("(actual rows=20)"),
        "the scan reads all 20 rows: {scan}"
    );
    assert!(
        node_lines.iter().any(|l| l.contains("(actual rows=10)")),
        "a post-filter node must show the 10 surviving rows: {analyzed:?}"
    );

    // The streaming surface is instrumented too: an aggregate folds its input from a stream, so
    // the scan under it records through the counting source, and the aggregate emits one row.
    let agg = explain_lines(
        engine,
        &mut session,
        "EXPLAIN ANALYZE SELECT count(*) FROM t",
    );
    let scan = agg
        .iter()
        .find(|l| l.contains("SeqScan"))
        .expect("plan must contain the scan");
    assert!(
        scan.contains("(actual rows=20)"),
        "the streamed scan still records its rows: {scan}"
    );
    let agg_node = agg
        .iter()
        .find(|l| l.contains("ScalarAggregate"))
        .expect("plan must contain the aggregate");
    assert!(
        agg_node.contains("(actual rows=1)"),
        "the aggregate produces one row: {agg_node}"
    );

    // Plain EXPLAIN (no ANALYZE) never carries actual-row annotations.
    let plain = explain_lines(engine, &mut session, "EXPLAIN SELECT a FROM t WHERE a > 1");
    assert!(
        !plain.iter().any(|l| l.contains("actual rows=")),
        "plain EXPLAIN must not execute or annotate: {plain:?}"
    );
}

#[test]
fn explain_analyze_rejects_data_modifying_statements() {
    let engine: &'static BtreeEngine = Box::leak(Box::new(BtreeEngine::new()));
    let mut session = Session::new(engine);
    exec(engine, &mut session, "CREATE TABLE t (a INT)");

    // EXPLAIN ANALYZE INSERT must be refused (executing it would commit the insert).
    let logical = analyze(
        parse("EXPLAIN ANALYZE INSERT INTO t VALUES (1)").unwrap(),
        &Cat(engine),
    )
    .unwrap();
    let err = session.execute(plan(logical)).expect_err("must reject");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

    // The plain EXPLAIN of the same INSERT is fine (it does not execute) and inserts nothing.
    let _ = exec(engine, &mut session, "EXPLAIN INSERT INTO t VALUES (1)");
    let ExecutionResult::Rows { rows, .. } = exec(engine, &mut session, "SELECT a FROM t") else {
        panic!("expected rows");
    };
    assert!(
        rows.is_empty(),
        "EXPLAIN (no ANALYZE) must not have inserted"
    );
}

#[test]
fn explain_non_text_format_is_rejected() {
    // Structured FORMAT output is a follow-up; the parser must reject it rather than
    // silently producing text. (Either an Unsupported or a parser Syntax error is acceptable.)
    assert!(parse("EXPLAIN (FORMAT JSON) SELECT 1").is_err());
}
