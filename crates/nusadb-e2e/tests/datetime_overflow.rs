//! Date and time arithmetic past the representable range fails with `22008`
//! (`datetime_field_overflow`) instead of clamping or wrapping.

#![allow(
    clippy::expect_used,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::{Catalog, Error, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<(), Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    execute(plan(logical), engine).map(|_| ())
}

#[test]
fn out_of_range_datetime_results_are_22008() {
    let engine = BtreeEngine::new();
    run_try(&engine, "CREATE TABLE big_iv (d INTERVAL)").expect("create");
    run_try(
        &engine,
        "INSERT INTO big_iv VALUES (INTERVAL '2000000000 days'), (INTERVAL '2000000000 days')",
    )
    .expect("insert");
    for sql in [
        "SELECT now() + (make_interval(days => -2147483647) + INTERVAL '2562047788 hours')",
        "SELECT now() - (make_interval(days => -2147483647) + INTERVAL '2562047788 hours')",
        "SELECT uuidv7(make_interval(days => -2147483647) + INTERVAL '2562047788 hours')",
        "SELECT TIMESTAMP '2000-01-01' + INTERVAL '300000 years'",
        "SELECT TIMESTAMPTZ '2000-01-01 00:00:00+00' - INTERVAL '300000 years'",
        "SELECT DATE '2000-01-01' + INTERVAL '300000 years'",
        "SELECT INTERVAL '300000 years' + DATE '2000-01-01'",
        "SELECT DATE '5874897-01-01' + INTERVAL '1 day'",
        "SELECT INTERVAL '2147483647 months' + INTERVAL '1 month'",
        "SELECT INTERVAL '2147483647 days' - INTERVAL '-1 day'",
        "SELECT INTERVAL '2000000000 days' * 2",
        "SELECT make_interval(years => 1000000000)",
        "SELECT make_interval(weeks => 2147483647)",
        "SELECT justify_hours(make_interval(days => 2147483647, hours => 24))",
        "SELECT TIMESTAMP '294247-01-10 04:00:54.775807' - TIMESTAMP '0001-01-01'",
        "SELECT SUM(d) FROM big_iv",
    ] {
        let err = run_try(&engine, sql).expect_err(sql);
        assert_eq!(err.sqlstate(), "22008", "{sql}: {err}");
    }
}
