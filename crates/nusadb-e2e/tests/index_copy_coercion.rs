//! Regression over the `COPY` path: a bulk-loaded row whose value's text form differs from
//! the column type (an integer `45` into a `NUMERIC` column) must be found through an index on that
//! column, not silently missed. COPY shares the insert path that adopts the column type before
//! indexing, so this guards the same fix as `slt/p1_ddl/index_type_coercion.slt` does for `INSERT`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test outside a #[cfg(test)] module"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::{Statement, Value};
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, copy_from, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        let backing: std::collections::HashSet<_> = self
            .0
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.0.list_indexes(schema.id)? {
            if self
                .0
                .lookup_index(&def.name)?
                .is_some_and(|id| backing.contains(&id))
            {
                continue;
            }
            if !def.key_exprs.is_empty() || def.predicate.is_some() {
                continue;
            }
            out.push(IndexInfo {
                name: def.name,
                columns: def.columns,
                unique: def.unique,
            });
        }
        Ok(out)
    }
}

fn exec(session: &mut Session, engine: &dyn StorageEngine, sql: &str) {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    session.execute(plan(logical)).expect("execute");
}

fn ids_where(session: &mut Session, engine: &dyn StorageEngine, sql: &str) -> Vec<i64> {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    match session.execute(plan(logical)).expect("execute") {
        ExecutionResult::Rows { rows, .. } => rows
            .iter()
            .map(|r| match r.first() {
                Some(Value::Int(v)) => *v,
                other => panic!("expected int id, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn copy_loaded_numeric_rows_are_found_through_the_index() {
    let engine = BtreeEngine::new();
    let mut session = Session::new(&engine);
    exec(
        &mut session,
        &engine,
        "CREATE TABLE ix (id INT, v NUMERIC(12,2))",
    );
    exec(&mut session, &engine, "CREATE INDEX ix_v ON ix (v)");

    // Bulk-load two rows whose numeric column arrives as the bare integer text `45`.
    let Statement::Copy(copy) = parse("COPY ix FROM STDIN").expect("parse copy") else {
        panic!("expected COPY");
    };
    let n = copy_from(&engine, &copy, "1\t45\n2\t45\n").expect("copy");
    assert_eq!(n, 2, "two rows loaded");

    // Both must be found through the index (WHERE v = 45 plans as an IndexScan on ix_v) and match
    // the sequential scan — the bug returned zero here.
    let mut via_index = ids_where(
        &mut session,
        &engine,
        "SELECT id FROM ix WHERE v = 45 ORDER BY id",
    );
    let mut via_seq = ids_where(
        &mut session,
        &engine,
        "SELECT id FROM ix WHERE v + 0 = 45 ORDER BY id",
    );
    via_index.sort_unstable();
    via_seq.sort_unstable();
    assert_eq!(
        via_index,
        vec![1, 2],
        "index scan must find both COPY-loaded rows"
    );
    assert_eq!(via_index, via_seq, "index and sequential scan must agree");
}
