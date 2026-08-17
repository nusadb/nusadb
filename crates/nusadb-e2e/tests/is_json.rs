//! End-to-end tests for the SQL `IS JSON` predicate:
//! `<expr> IS [NOT] JSON [VALUE | SCALAR | ARRAY | OBJECT] [WITH | WITHOUT UNIQUE KEYS]`.
//!
//! Every case below is an oracle row verified against a live reference engine: validity of
//! each JSON item type, `VALUE`/`SCALAR`/`ARRAY`/`OBJECT`, the `NOT` negation, three-valued
//! `NULL` handling, and `WITH`/`WITHOUT UNIQUE KEYS` (including a NESTED duplicate key and a
//! non-object operand that is trivially unique). Driven through `parse → analyze → plan → execute`
//! against the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, Error, ExecutionResult, FunctionDef, analyze, execute, parse, plan};

struct EngineCatalog<'a>(&'a BtreeEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<FunctionDef>, Error> {
        let txn = self.0.begin(IsolationLevel::default())?;
        let result = nusadb_sql::lookup_function_definition(self.0, txn, name);
        let _ = self.0.commit(txn);
        result
    }
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    let stmt = parse(sql).expect("parse");
    let logical = analyze(stmt, &EngineCatalog(engine)).expect("analyze");
    execute(plan(logical), engine).expect("execute")
}

/// Evaluate a single scalar `SELECT <expr>` and return its one value.
fn scalar(engine: &BtreeEngine, sql: &str) -> Value {
    let ExecutionResult::Rows { mut rows, .. } = run(engine, sql) else {
        panic!("expected rows for `{sql}`");
    };
    assert_eq!(rows.len(), 1, "expected exactly one row for `{sql}`");
    let mut row = rows.pop().expect("one row");
    assert_eq!(row.len(), 1, "expected exactly one column for `{sql}`");
    row.pop().expect("one column")
}

const fn t() -> Value {
    Value::Bool(true)
}
const fn f() -> Value {
    Value::Bool(false)
}

#[test]
fn is_json_validity_and_item_types() {
    let engine = BtreeEngine::new();

    // An object: valid JSON of type OBJECT (also VALUE); not an ARRAY or SCALAR.
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS JSON"#), t());
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS JSON VALUE"#), t());
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS JSON OBJECT"#), t());
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS JSON ARRAY"#), f());
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS JSON SCALAR"#), f());

    // An array: JSON + ARRAY, not OBJECT.
    assert_eq!(scalar(&engine, "SELECT '[1,2]' IS JSON"), t());
    assert_eq!(scalar(&engine, "SELECT '[1,2]' IS JSON ARRAY"), t());
    assert_eq!(scalar(&engine, "SELECT '[1,2]' IS JSON OBJECT"), f());

    // Scalars: a bare number and a JSON string are both JSON + SCALAR.
    assert_eq!(scalar(&engine, "SELECT '5' IS JSON"), t());
    assert_eq!(scalar(&engine, "SELECT '5' IS JSON SCALAR"), t());
    assert_eq!(scalar(&engine, r#"SELECT '"s"' IS JSON"#), t());
    assert_eq!(scalar(&engine, r#"SELECT '"s"' IS JSON SCALAR"#), t());
    // A scalar is neither OBJECT nor ARRAY.
    assert_eq!(scalar(&engine, "SELECT '5' IS JSON OBJECT"), f());
    assert_eq!(scalar(&engine, "SELECT '5' IS JSON ARRAY"), f());
}

#[test]
fn is_json_invalid_and_negation() {
    let engine = BtreeEngine::new();

    // `hello` is not valid JSON (an unquoted bare word).
    assert_eq!(scalar(&engine, "SELECT 'hello' IS JSON"), f());
    assert_eq!(scalar(&engine, "SELECT 'hello' IS NOT JSON"), t());

    // `IS NOT JSON` negates a valid document.
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}' IS NOT JSON"#), f());
    assert_eq!(scalar(&engine, "SELECT '[1,2]' IS NOT JSON ARRAY"), f());
    // Negation of a type mismatch is true (an object IS NOT a JSON array).
    assert_eq!(
        scalar(&engine, r#"SELECT '{"a":1}' IS NOT JSON ARRAY"#),
        t()
    );
}

#[test]
fn is_json_null_is_three_valued() {
    let engine = BtreeEngine::new();

    // NULL in → NULL out, and the negation of NULL is still NULL (never flipped to a boolean).
    assert_eq!(scalar(&engine, "SELECT NULL::text IS JSON"), Value::Null);
    assert_eq!(
        scalar(&engine, "SELECT NULL::text IS NOT JSON"),
        Value::Null
    );
    assert_eq!(
        scalar(&engine, "SELECT NULL::text IS JSON OBJECT"),
        Value::Null
    );
    assert_eq!(
        scalar(&engine, "SELECT NULL::text IS JSON WITH UNIQUE KEYS"),
        Value::Null
    );
}

#[test]
fn is_json_with_unique_keys() {
    let engine = BtreeEngine::new();

    // A top-level duplicate key fails `WITH UNIQUE KEYS`.
    assert_eq!(
        scalar(
            &engine,
            r#"SELECT '{"a":1,"a":2}' IS JSON WITH UNIQUE KEYS"#
        ),
        f()
    );
    // A NESTED duplicate key also fails — the uniqueness check is recursive.
    assert_eq!(
        scalar(
            &engine,
            r#"SELECT '{"a":{"b":1,"b":2}}' IS JSON WITH UNIQUE KEYS"#
        ),
        f()
    );
    // All-distinct keys pass.
    assert_eq!(
        scalar(
            &engine,
            r#"SELECT '{"a":1,"b":2}' IS JSON WITH UNIQUE KEYS"#
        ),
        t()
    );
    // A non-object (here an array) is trivially unique.
    assert_eq!(
        scalar(&engine, "SELECT '[1,2]' IS JSON WITH UNIQUE KEYS"),
        t()
    );

    // `WITHOUT UNIQUE KEYS` is the default: duplicates do not disqualify a valid document.
    assert_eq!(
        scalar(
            &engine,
            r#"SELECT '{"a":1,"a":2}' IS JSON WITHOUT UNIQUE KEYS"#
        ),
        t()
    );
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1,"a":2}' IS JSON"#), t());
}

#[test]
fn is_json_accepts_json_typed_operand() {
    let engine = BtreeEngine::new();

    // A JSON-typed operand is already valid JSON; the type check still applies.
    assert_eq!(scalar(&engine, r#"SELECT '{"a":1}'::json IS JSON"#), t());
    assert_eq!(
        scalar(&engine, r#"SELECT '{"a":1}'::json IS JSON OBJECT"#),
        t()
    );
    assert_eq!(
        scalar(&engine, r#"SELECT '{"a":1}'::json IS JSON ARRAY"#),
        f()
    );
    assert_eq!(scalar(&engine, "SELECT '[1,2]'::jsonb IS JSON ARRAY"), t());
}

#[test]
fn is_json_over_a_column_and_two_predicates() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE docs (id INT NOT NULL, body TEXT)");
    run(
        &engine,
        r#"INSERT INTO docs VALUES (1, '{"a":1}'), (2, '[1,2]'), (3, 'nope'), (4, NULL)"#,
    );

    // The predicate applies per row against a column; row 4's NULL body yields NULL (excluded).
    let ExecutionResult::Rows { rows, .. } = run(
        &engine,
        "SELECT id FROM docs WHERE body IS JSON ORDER BY id",
    ) else {
        panic!("expected rows");
    };
    assert_eq!(rows, vec![vec![Value::Int(1)], vec![Value::Int(2)]]);

    // Two predicates in one statement, each with its own item type.
    let ExecutionResult::Rows { rows, .. } = run(
        &engine,
        "SELECT id FROM docs WHERE body IS JSON OBJECT OR body IS JSON ARRAY ORDER BY id",
    ) else {
        panic!("expected rows");
    };
    assert_eq!(rows, vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
}
