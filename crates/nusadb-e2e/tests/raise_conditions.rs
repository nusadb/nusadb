//! NusaScript `RAISE` with a condition: `USING ERRCODE`, `RAISE SQLSTATE '...'`, `RAISE
//! condition_name`, and the `MESSAGE`, `DETAIL` and `HINT` options, on errors, notices, exception
//! handlers and BEFORE triggers (`INSERT`, `UPDATE`, `COPY`, `MERGE`), driven through
//! `parse → analyze → plan → execute` against the production `BtreeEngine`.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::{Statement, Value};
use nusadb_sql::notice::{Notice, Severity};
use nusadb_sql::{Catalog, Error, ExecutionResult, analyze, copy_from, execute, parse, plan};

struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    nusadb_test_utils::catalog_answers_as_superuser!(nusadb_sql);
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_function(&self, name: &str) -> Result<Option<nusadb_sql::FunctionDef>, Error> {
        let txn = self.0.begin(nusadb_core::IsolationLevel::default())?;
        let out = nusadb_sql::lookup_function_definition(self.0, txn, name);
        let _ = self.0.commit(txn);
        out
    }
}

fn run_try(engine: &BtreeEngine, sql: &str) -> Result<ExecutionResult, Error> {
    let logical = analyze(parse(sql)?, &EngineCatalog(engine))?;
    execute(plan(logical), engine)
}

fn run(engine: &BtreeEngine, sql: &str) -> ExecutionResult {
    run_try(engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn rows(engine: &BtreeEngine, sql: &str) -> Vec<Vec<Value>> {
    match run(engine, sql) {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

/// The error `sql` fails with: its SQLSTATE and message.
fn fails(engine: &BtreeEngine, sql: &str) -> (String, String) {
    match run_try(engine, sql) {
        Err(e) => (e.code().into_owned(), e.to_string()),
        Ok(other) => panic!("{sql}: expected an error, got {other:?}"),
    }
}

/// A procedure whose body is `body`, created afresh as `p`.
fn procedure(engine: &BtreeEngine, body: &str) {
    let _ = run_try(engine, "DROP PROCEDURE p");
    run(
        engine,
        &format!("CREATE PROCEDURE p() AS $$ BEGIN {body} END $$"),
    );
}

/// The error `CALL p()` fails with once `p` runs `body`.
fn raised(engine: &BtreeEngine, body: &str) -> (String, String) {
    procedure(engine, body);
    fails(engine, "CALL p()")
}

fn pair(code: &str, message: &str) -> (String, String) {
    (code.to_owned(), message.to_owned())
}

#[test]
fn errcode_sqlstate_and_condition_names_set_the_code() {
    let engine = BtreeEngine::new();
    let e = &engine;
    assert_eq!(
        raised(e, "RAISE 'quota exceeded' USING ERRCODE = 'MA001';"),
        pair("MA001", "raised exception: quota exceeded")
    );
    assert_eq!(
        raised(
            e,
            "RAISE EXCEPTION 'quota % of %', 3, 5 USING ERRCODE = 'MA001';"
        ),
        pair("MA001", "raised exception: quota 3 of 5")
    );
    assert_eq!(
        raised(e, "RAISE SQLSTATE '22023' USING MESSAGE = 'via sqlstate';"),
        pair("22023", "raised exception: via sqlstate")
    );
    assert_eq!(
        raised(e, "RAISE SQLSTATE '22023';"),
        pair("22023", "raised exception: 22023")
    );
    assert_eq!(
        raised(e, "RAISE unique_violation;"),
        pair("23505", "raised exception: unique_violation")
    );
    assert_eq!(
        raised(e, "RAISE unique_violation USING MESSAGE = 'dup!';"),
        pair("23505", "raised exception: dup!")
    );
    assert_eq!(
        raised(e, "RAISE EXCEPTION division_by_zero;"),
        pair("22012", "raised exception: division_by_zero")
    );
    for (name, code) in [
        ("check_violation", "23514"),
        ("invalid_parameter_value", "22023"),
        ("raise_exception", "P0001"),
        ("string_data_right_truncation", "22001"),
    ] {
        assert_eq!(
            raised(e, &format!("RAISE 'x' USING ERRCODE = '{name}';")),
            pair(code, "raised exception: x"),
            "{name}"
        );
    }
    assert_eq!(
        raised(
            e,
            "RAISE EXCEPTION USING MESSAGE = 'm30', ERRCODE = 'MA030', HINT = 'h30';"
        ),
        pair("MA030", "raised exception: m30")
    );
    assert_eq!(
        raised(e, "RAISE EXCEPTION USING ERRCODE = 'MA034';"),
        pair("MA034", "raised exception: MA034")
    );
    assert_eq!(
        raised(e, "RAISE EXCEPTION USING ERRCODE = 'unique_violation';"),
        pair("23505", "raised exception: unique_violation")
    );
    // The option is an expression, evaluated when the RAISE runs.
    assert_eq!(
        raised(
            e,
            "DECLARE c TEXT DEFAULT 'MA0'; RAISE 'dyn' USING ERRCODE = c || '02', MESSAGE := 'set';"
        ),
        pair("42601", "RAISE option already specified: MESSAGE")
    );
    assert_eq!(
        raised(
            e,
            "DECLARE c TEXT DEFAULT 'MA0'; RAISE USING ERRCODE = c || '02';"
        ),
        pair("MA002", "raised exception: MA002")
    );
}

#[test]
fn detail_and_hint_travel_with_the_error() {
    let engine = BtreeEngine::new();
    procedure(
        &engine,
        "RAISE EXCEPTION 'bad % and %', 1, 'x' USING ERRCODE = '22023', DETAIL = 'the detail', \
         HINT = 'the hint';",
    );
    match run_try(&engine, "CALL p()") {
        Err(Error::RaisedWith {
            sqlstate,
            message,
            detail,
            hint,
        }) => {
            assert_eq!(sqlstate, "22023");
            assert_eq!(message, "bad 1 and x");
            assert_eq!(detail.as_deref(), Some("the detail"));
            assert_eq!(hint.as_deref(), Some("the hint"));
        },
        other => panic!("expected RaisedWith, got {other:?}"),
    }
    // A detail alone keeps the default condition.
    procedure(&engine, "RAISE 'p' USING DETAIL = 'pd';");
    match run_try(&engine, "CALL p()") {
        Err(e @ Error::RaisedWith { .. }) => {
            assert_eq!(e.sqlstate(), "P0001");
            assert_eq!(e.to_string(), "raised exception: p");
        },
        other => panic!("expected RaisedWith, got {other:?}"),
    }
}

#[test]
fn plain_raise_is_unchanged() {
    let engine = BtreeEngine::new();
    procedure(&engine, "RAISE 'plain';");
    match run_try(&engine, "CALL p()") {
        Err(e @ Error::Raised(_)) => {
            assert_eq!(e.sqlstate(), "P0001");
            assert_eq!(e.to_string(), "raised exception: plain");
        },
        other => panic!("expected Raised, got {other:?}"),
    }
    // A variable is still the message, not a condition name.
    assert_eq!(
        raised(&engine, "DECLARE m TEXT DEFAULT 'from var'; RAISE m;"),
        pair("P0001", "raised exception: from var")
    );
    // Without a level and without arguments the text is taken as it is.
    assert_eq!(
        raised(&engine, "RAISE 'a %%b';"),
        pair("P0001", "raised exception: a %%b")
    );
    assert_eq!(
        raised(&engine, "RAISE 'a % b', 1;"),
        pair("P0001", "raised exception: a 1 b")
    );
    // Naming the default condition changes nothing.
    procedure(&engine, "RAISE 'x' USING ERRCODE = 'P0001';");
    assert!(matches!(run_try(&engine, "CALL p()"), Err(Error::Raised(m)) if m == "x"));
}

#[test]
fn success_and_warning_classes() {
    let engine = BtreeEngine::new();
    // An error cannot report success: `00000` raises the default condition.
    assert_eq!(
        raised(&engine, "RAISE 'x' USING ERRCODE = '00000';"),
        pair("P0001", "raised exception: x")
    );
    assert_eq!(
        raised(&engine, "RAISE SQLSTATE '00000';"),
        pair("P0001", "raised exception: 00000")
    );
    // A warning-class code on an EXCEPTION is still an error, with that code.
    assert_eq!(
        raised(&engine, "RAISE 'warnclass' USING ERRCODE = '01000';"),
        pair("01000", "raised exception: warnclass")
    );
}

#[test]
fn invalid_codes_and_options() {
    let engine = BtreeEngine::new();
    let e = &engine;
    // A string ERRCODE is checked when the RAISE runs.
    for bad in [
        "MA00",
        "ma001",
        "M-001",
        "no_such_condition",
        "Unique_Violation",
        "p0001",
    ] {
        assert_eq!(
            raised(e, &format!("RAISE 'x' USING ERRCODE = '{bad}';")),
            pair(
                "42704",
                &format!("unrecognized exception condition \"{bad}\"")
            ),
            "{bad}"
        );
    }
    // Even on a level that sends nothing.
    assert_eq!(
        raised(e, "RAISE LOG 'x' USING ERRCODE = 'bad';"),
        pair("42704", "unrecognized exception condition \"bad\"")
    );
    // A written SQLSTATE, a condition name after a level and an option name are checked when
    // the routine is created.
    let create = |body: &str| {
        fails(
            e,
            &format!("CREATE PROCEDURE q() AS $$ BEGIN {body} END $$"),
        )
    };
    let (code, message) = create("RAISE SQLSTATE 'MA00';");
    assert_eq!(code, "42601", "{message}");
    assert!(message.contains("invalid SQLSTATE code"), "{message}");
    assert_eq!(
        create("RAISE EXCEPTION no_such_condition;"),
        pair(
            "42704",
            "unrecognized exception condition \"no_such_condition\""
        )
    );
    let (code, message) = create("RAISE EXCEPTION 'x' USING foo = 'y';");
    assert_eq!(code, "42601", "{message}");
    assert!(
        message.contains("unrecognized RAISE statement option"),
        "{message}"
    );
    // Options given twice, or beside what they would replace, and NULL options fail when run.
    assert_eq!(
        raised(e, "RAISE 'a' USING MESSAGE = 'b';"),
        pair("42601", "RAISE option already specified: MESSAGE")
    );
    assert_eq!(
        raised(e, "RAISE 'a' USING ERRCODE = 'MA001', ERRCODE = 'MA002';"),
        pair("42601", "RAISE option already specified: ERRCODE")
    );
    assert_eq!(
        raised(e, "RAISE unique_violation USING ERRCODE = 'MA001';"),
        pair("42601", "RAISE option already specified: ERRCODE")
    );
    assert_eq!(
        raised(e, "RAISE 'x' USING DETAIL = 'd', DETAIL = 'e';"),
        pair("42601", "RAISE option already specified: DETAIL")
    );
    assert_eq!(
        raised(e, "RAISE 'x' USING ERRCODE = NULL;"),
        pair("22004", "RAISE statement option cannot be null")
    );
    assert_eq!(
        raised(e, "RAISE 'x' USING DETAIL = NULL;"),
        pair("22004", "RAISE statement option cannot be null")
    );
}

#[test]
fn notices_carry_the_condition() {
    let engine = BtreeEngine::new();
    let notice = |body: &str| -> Vec<Notice> {
        procedure(&engine, body);
        let mut session = nusadb_sql::Session::new(&engine);
        let _ = session.take_notices();
        let logical =
            analyze(parse("CALL p()").expect("parse"), &EngineCatalog(&engine)).expect("analyze");
        session.execute(plan(logical)).expect("CALL p()");
        session.take_notices()
    };
    let got = notice("RAISE NOTICE 'note %', 5 USING ERRCODE = '01ABC', HINT = 'h';");
    assert_eq!(got.len(), 1);
    assert_eq!(
        (
            got[0].severity,
            got[0].code.as_str(),
            got[0].message.as_str()
        ),
        (Severity::Notice, "01ABC", "note 5")
    );
    assert_eq!(got[0].hint.as_deref(), Some("h"));
    let got = notice("RAISE WARNING 'warn' USING ERRCODE = 'MA001', DETAIL = 'd';");
    assert_eq!(
        (
            got[0].severity,
            got[0].code.as_str(),
            got[0].message.as_str()
        ),
        (Severity::Warning, "MA001", "warn")
    );
    assert_eq!(got[0].detail.as_deref(), Some("d"));
    let got = notice("RAISE NOTICE USING MESSAGE = 'n35';");
    assert_eq!(
        (got[0].code.as_str(), got[0].message.as_str()),
        ("00000", "n35")
    );
    let got = notice("RAISE NOTICE USING DETAIL = 'only detail';");
    assert_eq!(
        (got[0].code.as_str(), got[0].message.as_str()),
        ("00000", "00000")
    );
    let got = notice("RAISE NOTICE 'n' USING ERRCODE = 'unique_violation';");
    assert_eq!(
        (got[0].code.as_str(), got[0].message.as_str()),
        ("23505", "n")
    );
    let got = notice("RAISE INFO 'i' USING ERRCODE = 'MA002';");
    assert_eq!(
        (got[0].severity, got[0].code.as_str()),
        (Severity::Info, "MA002")
    );
    let got = notice("RAISE WARNING USING ERRCODE = 'MA001';");
    assert_eq!(
        (got[0].code.as_str(), got[0].message.as_str()),
        ("MA001", "MA001")
    );
    // The plain forms keep their codes.
    let got = notice("RAISE WARNING 'w';");
    assert_eq!(got[0].code, "01000");
    assert!(notice("RAISE LOG 'x' USING ERRCODE = 'MA001';").is_empty());
}

#[test]
fn when_others_catches_a_named_condition() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE log (what TEXT)");
    procedure(
        &engine,
        "BEGIN INSERT INTO log VALUES ('body'); RAISE 'q' USING ERRCODE = 'MA001'; \
         EXCEPTION WHEN OTHERS THEN INSERT INTO log VALUES ('handler'); END;",
    );
    run(&engine, "CALL p()");
    assert_eq!(
        rows(&engine, "SELECT what FROM log"),
        vec![vec![Value::Text("handler".to_owned())]]
    );
    procedure(
        &engine,
        "BEGIN RAISE unique_violation; EXCEPTION WHEN OTHERS THEN INSERT INTO log VALUES ('dup'); \
         END;",
    );
    run(&engine, "CALL p()");
    assert_eq!(
        rows(&engine, "SELECT count(*) FROM log"),
        vec![vec![Value::Int(2)]]
    );
}

#[test]
fn a_function_reports_the_condition() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE FUNCTION guard(n INT) RETURNS INT LANGUAGE nusascript AS $$ BEGIN \
         IF n > 10 THEN RAISE 'quota % exceeded', n USING ERRCODE = 'MA001', HINT = 'ask'; \
         END IF; RETURN n; END $$",
    );
    assert_eq!(rows(&engine, "SELECT guard(3)"), vec![vec![Value::Int(3)]]);
    assert_eq!(
        fails(&engine, "SELECT guard(11)"),
        pair("MA001", "raised exception: quota 11 exceeded")
    );
}

/// A BEFORE trigger's custom condition reaches the client on every write path, and the statement
/// leaves nothing behind.
#[test]
fn a_before_trigger_reports_the_condition_and_the_statement_rolls_back() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE TABLE tt (id INT PRIMARY KEY, v INT)");
    run(
        &engine,
        "CREATE FUNCTION tg() RETURNS trigger LANGUAGE nusascript AS $$ BEGIN \
         IF NEW.v < 0 THEN RAISE 'negative v' USING ERRCODE = 'MA100', DETAIL = 'v was ' || NEW.v; \
         END IF; RETURN NEW; END $$",
    );
    run(
        &engine,
        "CREATE TRIGGER tg BEFORE INSERT OR UPDATE ON tt FOR EACH ROW EXECUTE FUNCTION tg()",
    );
    run(&engine, "INSERT INTO tt VALUES (1, 1)");
    let negative = pair("MA100", "raised exception: negative v");
    assert_eq!(
        fails(&engine, "INSERT INTO tt VALUES (2, 5), (3, -1)"),
        negative
    );
    assert_eq!(
        fails(&engine, "UPDATE tt SET v = -5 WHERE id = 1"),
        negative
    );
    assert_eq!(
        fails(
            &engine,
            "MERGE INTO tt USING (SELECT 1 AS id, -9 AS v) s ON tt.id = s.id \
             WHEN MATCHED THEN UPDATE SET v = s.v"
        ),
        negative
    );
    let copy = match parse("COPY tt FROM STDIN").expect("parse") {
        Statement::Copy(copy) => copy,
        other => panic!("expected COPY, got {other:?}"),
    };
    match copy_from(&engine, &copy, "4\t4\n5\t-3\n") {
        Err(e) => assert_eq!(
            (e.code().into_owned(), e.to_string()),
            ("MA100".to_owned(), negative.1)
        ),
        Ok(n) => panic!("COPY loaded {n} rows"),
    }
    match run_try(&engine, "INSERT INTO tt VALUES (6, -6)") {
        Err(Error::RaisedWith { detail, .. }) => assert_eq!(detail.as_deref(), Some("v was -6")),
        other => panic!("expected RaisedWith, got {other:?}"),
    }
    assert_eq!(
        rows(&engine, "SELECT id, v FROM tt ORDER BY id"),
        vec![vec![Value::Int(1), Value::Int(1)]]
    );
}

/// A `RAISE` naming a retryable code (`40001`) is the routine's own decision: the automatic retry
/// of a conflicting auto-commit statement does not run the body again.
#[test]
fn a_raised_serialization_failure_is_not_retried() {
    let engine = BtreeEngine::new();
    run(&engine, "CREATE SEQUENCE ran");
    procedure(
        &engine,
        "PERFORM nextval('ran'); RAISE serialization_failure;",
    );
    let mut session = nusadb_sql::Session::new(&engine);
    let logical =
        analyze(parse("CALL p()").expect("parse"), &EngineCatalog(&engine)).expect("analyze");
    let err = session
        .execute(plan(logical))
        .expect_err("the RAISE fails the call");
    assert_eq!(err.sqlstate(), "40001");
    assert_eq!(
        rows(&engine, "SELECT nextval('ran')"),
        vec![vec![Value::Int(2)]],
        "the body ran once"
    );
}
