//! Column grants over a connection. The wire path answers column checks from one read of the grants
//! per statement, so these tests pin that the read lasts no longer than the statement: a grant or a
//! revoke reaches the next statement of the same transaction, and from another connection too. They
//! also pin that a read of a whole row needs SELECT on every column it expands, wherever a statement
//! makes it, over the simple and the extended protocol alike.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic over linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, handle_client};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn next(conn: &mut Conn) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Connect as `user` and consume the handshake up to the first `ReadyForQuery`.
async fn open(engine: &Arc<dyn StorageEngine>, user: &str) -> Conn {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: user.to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    while !matches!(next(&mut conn).await, BackendMessage::ReadyForQuery(_)) {}
    conn
}

/// What a statement came back with: the SQLSTATE of its error, or the fields of its rows.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Refused(String),
    Rows(Vec<Vec<Option<String>>>),
}

async fn run(conn: &mut Conn, sql: &str) -> Outcome {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    outcome(conn).await
}

/// Prepare `sql` as the statement `name` (the unnamed one when empty), without running it.
async fn parse(conn: &mut Conn, name: &str, sql: &str) {
    send(
        conn,
        FrontendMessage::Parse {
            name: name.to_owned(),
            sql: sql.to_owned(),
            param_types: Vec::new(),
        },
    )
    .await;
}

/// Bind the prepared statement `name` and run it to the end of the pipeline.
async fn execute(conn: &mut Conn, name: &str) -> Outcome {
    send(
        conn,
        FrontendMessage::Bind {
            portal: String::new(),
            statement: name.to_owned(),
            params: Vec::new(),
            result_formats: Vec::new(),
        },
    )
    .await;
    send(
        conn,
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 0,
        },
    )
    .await;
    send(conn, FrontendMessage::Sync).await;
    outcome(conn).await
}

/// Run `sql` over the extended protocol: Parse, Bind, Execute, Sync.
async fn run_extended(conn: &mut Conn, sql: &str) -> Outcome {
    parse(conn, "", sql).await;
    execute(conn, "").await
}

/// The rows and the refusal a statement answered with, up to `ReadyForQuery`.
async fn outcome(conn: &mut Conn) -> Outcome {
    let mut rows = Vec::new();
    let mut refused = None;
    loop {
        match next(conn).await {
            BackendMessage::DataRow { values } => rows.push(
                values
                    .into_iter()
                    .map(|v| v.map(|bytes| String::from_utf8(bytes).unwrap()))
                    .collect(),
            ),
            BackendMessage::Error { code, .. } => refused = Some(code),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    refused.map_or(Outcome::Rows(rows), Outcome::Refused)
}

/// Run `sql` and expect it not to be refused.
async fn ok(conn: &mut Conn, sql: &str) -> Vec<Vec<Option<String>>> {
    match run(conn, sql).await {
        Outcome::Rows(rows) => rows,
        Outcome::Refused(code) => panic!("`{sql}` was refused with {code}"),
    }
}

/// Run `sql` and expect a permission denial.
async fn denied(conn: &mut Conn, sql: &str) {
    assert_eq!(
        run(conn, sql).await,
        Outcome::Refused("42501".to_owned()),
        "{sql}"
    );
}

/// Expect a permission denial over both protocols. Outside a transaction block only: the first
/// refusal would abort it.
async fn denied_both(conn: &mut Conn, sql: &str) {
    denied(conn, sql).await;
    assert_eq!(
        run_extended(conn, sql).await,
        Outcome::Refused("42501".to_owned()),
        "{sql} over the extended protocol"
    );
}

/// An engine with `t (id, v, s)`, one row, and a role that may read `id` and `s` and write `v`.
async fn fixture() -> (Arc<dyn StorageEngine>, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    for sql in [
        "CREATE TABLE t (id INT, v INT, s INT)",
        "INSERT INTO t VALUES (1, 2, 3)",
        "CREATE ROLE clerk LOGIN",
        "GRANT SELECT (id, s), UPDATE (v) ON t TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    (engine, root)
}

#[tokio::test]
async fn a_revoke_from_another_connection_reaches_the_next_statement_of_an_open_transaction() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    let read = "UPDATE t SET v = s";

    ok(&mut clerk, "BEGIN").await;
    ok(&mut clerk, read).await;
    ok(&mut clerk, read).await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    denied(&mut clerk, read).await;
    ok(&mut clerk, "ROLLBACK").await;

    denied(&mut clerk, read).await;
    ok(&mut root, "GRANT SELECT (s) ON t TO clerk").await;
    ok(&mut clerk, read).await;
}

#[tokio::test]
async fn a_revoke_reaches_the_next_statement_of_its_own_transaction() {
    let (_engine, mut root) = fixture().await;
    let read = "UPDATE t SET v = s";

    ok(&mut root, "BEGIN").await;
    ok(&mut root, "SET ROLE clerk").await;
    ok(&mut root, read).await;
    ok(&mut root, "RESET ROLE").await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    ok(&mut root, "SET ROLE clerk").await;
    denied(&mut root, read).await;
}

/// What a statement read of the table grants lasts as long as the statement, like its column grants.
#[tokio::test]
async fn a_table_wide_revoke_reaches_the_next_statement_of_an_open_transaction() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "CREATE TABLE u (id INT PRIMARY KEY, v INT)",
        "GRANT SELECT, INSERT ON u TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let upsert = "INSERT INTO u VALUES (1, 1) ON CONFLICT (id) DO NOTHING RETURNING *";

    ok(&mut clerk, "BEGIN").await;
    ok(&mut clerk, upsert).await;
    ok(&mut clerk, upsert).await;
    ok(&mut root, "REVOKE SELECT ON u FROM clerk").await;
    denied(&mut clerk, upsert).await;
    ok(&mut clerk, "ROLLBACK").await;

    ok(&mut root, "GRANT SELECT ON u TO clerk").await;
    ok(&mut clerk, upsert).await;
}

/// The same within one transaction, where the next statement sees the writes of the last one.
#[tokio::test]
async fn a_table_wide_grant_reaches_the_next_statement_of_its_own_transaction() {
    let (_engine, mut root) = fixture().await;
    let read = "SELECT * FROM u";
    ok(&mut root, "CREATE TABLE u (id INT, v INT)").await;
    ok(&mut root, "BEGIN").await;
    ok(&mut root, "GRANT SELECT ON u TO clerk").await;
    ok(&mut root, "SET ROLE clerk").await;
    ok(&mut root, read).await;
    ok(&mut root, read).await;
    ok(&mut root, "RESET ROLE").await;
    ok(&mut root, "REVOKE SELECT ON u FROM clerk").await;
    ok(&mut root, "SET ROLE clerk").await;
    denied(&mut root, read).await;
}

/// A nested body's statement is a statement too: its check does not outlive it.
#[tokio::test]
async fn a_revoke_reaches_the_next_nested_body_of_an_open_transaction() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    let body = "DO $$ UPDATE t SET v = s $$";
    ok(&mut clerk, "BEGIN").await;
    ok(&mut clerk, body).await;
    ok(&mut clerk, body).await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    denied(&mut clerk, body).await;
    ok(&mut clerk, "ROLLBACK").await;
}

#[tokio::test]
async fn a_revoke_reaches_the_next_execution_of_a_prepared_statement() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    for (name, sql) in [
        ("write", "UPDATE t SET v = s"),
        ("read", "SELECT s FROM t"),
        ("back", "UPDATE t SET v = 1 WHERE s = 3 RETURNING id, s"),
    ] {
        parse(&mut clerk, name, sql).await;
    }
    send(&mut clerk, FrontendMessage::Sync).await;
    assert_eq!(outcome(&mut clerk).await, Outcome::Rows(Vec::new()));
    for name in ["write", "read", "back"] {
        assert!(matches!(execute(&mut clerk, name).await, Outcome::Rows(_)));
    }
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    for name in ["write", "read", "back"] {
        assert_eq!(
            execute(&mut clerk, name).await,
            Outcome::Refused("42501".to_owned()),
            "{name}"
        );
    }
    ok(&mut root, "GRANT SELECT (s) ON t TO clerk").await;
    for name in ["write", "read", "back"] {
        assert!(matches!(execute(&mut clerk, name).await, Outcome::Rows(_)));
    }
}

/// A whole row names no column, and a revoke of any one of them reaches it all the same.
#[tokio::test]
async fn a_revoke_reaches_the_next_execution_of_a_prepared_whole_row_read() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "GRANT SELECT (v) ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    let statements = [
        ("object", "SELECT row_to_json(t) FROM t"),
        ("star", "SELECT * FROM t"),
        ("written", "UPDATE t SET v = 2 RETURNING row_to_json(t)"),
        ("all", "UPDATE t SET v = 2 RETURNING *"),
    ];
    for (name, sql) in statements {
        parse(&mut clerk, name, sql).await;
    }
    send(&mut clerk, FrontendMessage::Sync).await;
    assert_eq!(outcome(&mut clerk).await, Outcome::Rows(Vec::new()));
    for (name, _) in statements {
        let outcome = execute(&mut clerk, name).await;
        assert!(
            matches!(outcome, Outcome::Rows(ref rows) if rows.len() == 1),
            "{name}: {outcome:?}"
        );
    }
    ok(&mut root, "REVOKE SELECT (v) ON t FROM clerk").await;
    for (name, _) in statements {
        assert_eq!(
            execute(&mut clerk, name).await,
            Outcome::Refused("42501".to_owned()),
            "{name}"
        );
    }
    ok(&mut root, "GRANT SELECT (v) ON t TO clerk").await;
    for (name, _) in statements {
        assert!(
            matches!(execute(&mut clerk, name).await, Outcome::Rows(_)),
            "{name}"
        );
    }
}

/// A name that is no key of the table does not exist, whatever the session may read of the table,
/// and a real key whose columns the session may not read is still a privilege it lacks.
#[tokio::test]
async fn an_arbiter_name_that_is_no_key_is_undefined_for_every_session() {
    let (engine, mut root) = fixture().await;
    for sql in [
        "CREATE TABLE oc (id INT, k INT, w TEXT, CONSTRAINT oc_k UNIQUE (k))",
        "CREATE UNIQUE INDEX oc_e ON oc ((k + id))",
        "CREATE UNIQUE INDEX oc_i ON oc (id)",
        "GRANT INSERT ON oc TO clerk",
        "GRANT SELECT (id) ON oc TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    let insert = |name: &str| {
        format!("INSERT INTO oc VALUES (1, 1, 'x') ON CONFLICT ON CONSTRAINT {name} DO NOTHING")
    };
    for name in ["no_such_key", "oc_e", "oc_i"] {
        for conn in [&mut root, &mut clerk] {
            assert_eq!(
                run(conn, &insert(name)).await,
                Outcome::Refused("42704".to_owned()),
                "{name}"
            );
        }
    }
    denied_both(&mut clerk, &insert("oc_k")).await;
    ok(&mut root, "GRANT SELECT (k) ON oc TO clerk").await;
    ok(&mut clerk, &insert("oc_k")).await;
}

#[tokio::test]
async fn a_revoke_reaches_the_next_extended_statement_of_an_open_transaction() {
    let (engine, mut root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    let read = "UPDATE t SET v = s";

    ok(&mut clerk, "BEGIN").await;
    assert_eq!(
        run_extended(&mut clerk, read).await,
        Outcome::Rows(Vec::new())
    );
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    assert_eq!(
        run_extended(&mut clerk, read).await,
        Outcome::Refused("42501".to_owned())
    );
    ok(&mut clerk, "ROLLBACK").await;

    ok(&mut root, "GRANT SELECT (s) ON t TO clerk").await;
    assert_eq!(
        run_extended(&mut clerk, read).await,
        Outcome::Rows(Vec::new())
    );
}

#[tokio::test]
async fn column_reads_are_checked_where_a_write_makes_them_over_the_extended_protocol() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "REVOKE SELECT (s) ON t FROM clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "UPDATE t SET v = s",
        "UPDATE t SET v = 1 WHERE s = 3",
        "UPDATE t SET v = 1 RETURNING s",
        "UPDATE t SET v = 1 RETURNING *",
        "SELECT * FROM t",
    ] {
        denied_both(&mut clerk, sql).await;
    }
    assert_eq!(
        run_extended(&mut clerk, "UPDATE t SET v = 9 RETURNING id").await,
        Outcome::Rows(vec![vec![Some("1".to_owned())]])
    );
}

#[tokio::test]
async fn a_wildcard_returning_through_a_parent_is_the_parents_columns_over_the_wire() {
    let (engine, mut root) = fixture().await;
    for sql in [
        "CREATE TABLE ip (id INT, v INT)",
        "CREATE TABLE ic (extra TEXT) INHERITS (ip)",
        "INSERT INTO ip VALUES (1, 10)",
        "INSERT INTO ic VALUES (2, 20, 'secret')",
        "GRANT SELECT (id, v), UPDATE (v), DELETE ON ip TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    let field = |v: &str| Some(v.to_owned());
    assert_eq!(
        ok(&mut clerk, "UPDATE ip AS t SET v = v + 1 RETURNING t.*").await,
        vec![vec![field("1"), field("11")], vec![field("2"), field("21")]]
    );
    assert_eq!(
        run_extended(&mut clerk, "UPDATE ip SET v = v + 1 RETURNING *").await,
        Outcome::Rows(vec![
            vec![field("1"), field("12")],
            vec![field("2"), field("22")]
        ])
    );
    let rows = vec![
        vec![field(r#"{"id": 1, "v": 13}"#)],
        vec![field(r#"{"id": 2, "v": 23}"#)],
    ];
    let whole = "UPDATE ip SET v = v + 1 RETURNING row_to_json(ip)";
    assert_eq!(run_extended(&mut clerk, whole).await, Outcome::Rows(rows));
    assert_eq!(
        ok(&mut clerk, "DELETE FROM ip WHERE id = 2 RETURNING *").await,
        vec![vec![field("2"), field("23")]]
    );
}

/// An engine with `t (id, a, secret)`, one row, a table `s` to read from, a full view of `t`, and a
/// role with no privilege yet.
async fn whole_row_fixture() -> (Arc<dyn StorageEngine>, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    for sql in [
        "CREATE TABLE t (id INT PRIMARY KEY, a INT, secret TEXT)",
        "INSERT INTO t VALUES (1, 2, 'hunter2')",
        "CREATE TABLE s (id INT, x INT, hid TEXT)",
        "INSERT INTO s VALUES (1, 7, 'hid')",
        "CREATE VIEW v AS SELECT * FROM t",
        "CREATE ROLE clerk LOGIN",
    ] {
        ok(&mut root, sql).await;
    }
    (engine, root)
}

/// What the superuser sees of `t`, to show a refused statement changed nothing.
async fn table_t(root: &mut Conn) -> Vec<Vec<Option<String>>> {
    ok(root, "SELECT id, a, secret FROM t").await
}

#[tokio::test]
async fn a_write_reads_no_whole_row_without_select_on_every_column() {
    let (engine, mut root) = whole_row_fixture().await;
    ok(&mut root, "GRANT UPDATE (a), DELETE ON t TO clerk").await;
    let before = table_t(&mut root).await;
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "UPDATE t SET a = 1 RETURNING row_to_json(t)",
        "UPDATE t AS x SET a = 1 RETURNING row_to_json(x)",
        "WITH u AS (UPDATE t SET a = 5 RETURNING row_to_json(t) j) SELECT j FROM u",
        "DELETE FROM t RETURNING row_to_json(t)",
        "DELETE FROM t AS x RETURNING row_to_json(x)",
        "UPDATE t SET a = 1 WHERE row_to_json(t)::text LIKE '%hunter2%'",
        "UPDATE t SET a = length(row_to_json(t)::text)",
        "DELETE FROM t WHERE row_to_json(t)::text LIKE '%hunter2%'",
        "UPDATE v SET a = 5 RETURNING row_to_json(t)",
        "UPDATE v SET a = 5 WHERE row_to_json(t)::text LIKE '%hunter2%'",
        "DELETE FROM v RETURNING row_to_json(t)",
    ] {
        denied_both(&mut clerk, sql).await;
    }
    assert_eq!(table_t(&mut root).await, before);
}

#[tokio::test]
async fn a_write_reads_a_whole_row_with_select_on_every_column() {
    let (engine, mut root) = whole_row_fixture().await;
    ok(&mut root, "GRANT UPDATE (a) ON t TO clerk").await;
    ok(&mut root, "GRANT SELECT (id, a) ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    let sql = "UPDATE t SET a = 3 RETURNING row_to_json(t)";
    denied(&mut clerk, sql).await;

    // Each column granted is as good as the table.
    ok(&mut root, "GRANT SELECT (secret) ON t TO clerk").await;
    let row = r#"{"id": 1, "a": 3, "secret": "hunter2"}"#.to_owned();
    assert_eq!(ok(&mut clerk, sql).await, vec![vec![Some(row.clone())]]);
    assert_eq!(
        run_extended(&mut clerk, sql).await,
        Outcome::Rows(vec![vec![Some(row.clone())]])
    );
    ok(&mut root, "REVOKE SELECT (id, a, secret) ON t FROM clerk").await;
    ok(&mut root, "GRANT SELECT ON t TO clerk").await;
    assert_eq!(ok(&mut clerk, sql).await, vec![vec![Some(row)]]);
}

#[tokio::test]
async fn a_row_of_named_columns_needs_select_on_those_columns_only() {
    let (engine, mut root) = whole_row_fixture().await;
    ok(&mut root, "GRANT SELECT (id, a) ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    assert_eq!(
        ok(&mut clerk, "SELECT row_to_json(ROW(id, a)) FROM t").await,
        vec![vec![Some(r#"{"f1": 1, "f2": 2}"#.to_owned())]]
    );
    assert_eq!(
        ok(
            &mut clerk,
            "SELECT row_to_json(d) FROM (SELECT id, a FROM t) d"
        )
        .await,
        vec![vec![Some(r#"{"id": 1, "a": 2}"#.to_owned())]]
    );
    denied_both(&mut clerk, "SELECT row_to_json(ROW(id, secret)) FROM t").await;
}

#[tokio::test]
async fn a_select_reads_no_whole_row_without_select_on_every_column() {
    let (engine, mut root) = whole_row_fixture().await;
    for sql in [
        "GRANT SELECT (id) ON t TO clerk",
        "GRANT SELECT (id, x) ON s TO clerk",
        "CREATE TABLE kept (j TEXT)",
        "GRANT INSERT ON kept TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "SELECT row_to_json(t) FROM t",
        "SELECT row_to_json(x) FROM t x",
        "SELECT id FROM t WHERE row_to_json(t)::text LIKE '%hunter2%'",
        "SELECT id FROM t ORDER BY row_to_json(t)::text",
        "SELECT row_to_json(t)::text, count(*) FROM t GROUP BY 1",
        "SELECT count(*) FILTER (WHERE row_to_json(t)::text LIKE '%h%') FROM t",
        "SELECT json_agg(row_to_json(t)) FROM t",
        "SELECT row_number() OVER (ORDER BY row_to_json(t)::text) FROM t",
        "SELECT j FROM (SELECT row_to_json(t) j FROM t) d",
        "WITH d AS (SELECT row_to_json(t) j FROM t) SELECT j FROM d",
        "SELECT (SELECT row_to_json(x) FROM t x) FROM s",
        "SELECT s.id FROM s JOIN t ON row_to_json(t)::text LIKE '%h%'",
        "SELECT t.id FROM t JOIN s ON row_to_json(s)::text LIKE '%hid%'",
        "SELECT * FROM generate_series(1, (SELECT length(row_to_json(t)::text) FROM t))",
        "SELECT jsonb_each(row_to_json(t)) FROM t",
        "INSERT INTO kept SELECT row_to_json(t)::text FROM t",
        "CREATE TABLE stolen AS SELECT row_to_json(t) j FROM t",
        "CREATE MATERIALIZED VIEW stolen AS SELECT row_to_json(t) j FROM t",
        "DO $$ INSERT INTO kept SELECT row_to_json(t)::text FROM t $$",
    ] {
        denied_both(&mut clerk, sql).await;
    }
    denied(&mut clerk, "COPY (SELECT row_to_json(t) FROM t) TO STDOUT").await;
    denied(&mut clerk, "EXPLAIN SELECT row_to_json(t) FROM t").await;
    ok(&mut clerk, "PREPARE whole AS SELECT row_to_json(t) FROM t").await;
    denied(&mut clerk, "EXECUTE whole").await;
}

#[tokio::test]
async fn a_cached_whole_row_read_is_refused_after_a_revoke() {
    let (engine, mut root) = whole_row_fixture().await;
    ok(&mut root, "GRANT SELECT (id, a, secret) ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    let sql = "SELECT row_to_json(t) FROM t";
    let row = r#"{"id": 1, "a": 2, "secret": "hunter2"}"#.to_owned();
    for _ in 0..2 {
        assert_eq!(ok(&mut clerk, sql).await, vec![vec![Some(row.clone())]]);
    }
    ok(&mut root, "REVOKE SELECT (secret) ON t FROM clerk").await;
    denied(&mut clerk, sql).await;
}

#[tokio::test]
async fn a_write_reads_no_whole_row_of_a_source_without_select_on_every_column() {
    let (engine, mut root) = whole_row_fixture().await;
    for sql in [
        "GRANT SELECT, INSERT, UPDATE, DELETE ON t TO clerk",
        "GRANT SELECT (id, x) ON s TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "UPDATE t SET a = length(row_to_json(s)::text) FROM s",
        "UPDATE t SET a = 1 FROM s WHERE row_to_json(s)::text LIKE '%hid%'",
        "UPDATE t SET a = 1 FROM s AS q WHERE row_to_json(q)::text LIKE '%hid%'",
        "DELETE FROM t USING s WHERE row_to_json(s)::text LIKE '%hid%'",
        "MERGE INTO t USING s ON row_to_json(s)::text LIKE '%hid%' \
         WHEN MATCHED THEN UPDATE SET a = 1",
        "MERGE INTO t USING s ON t.id = s.id \
         WHEN MATCHED AND row_to_json(s)::text LIKE '%hid%' THEN DELETE",
        "MERGE INTO t USING s ON t.id = s.id \
         WHEN MATCHED THEN UPDATE SET a = length(row_to_json(s)::text)",
        "MERGE INTO t USING s ON t.id = 9 \
         WHEN NOT MATCHED THEN INSERT (id, a) VALUES (s.id, length(row_to_json(s)::text))",
    ] {
        denied_both(&mut clerk, sql).await;
    }
    ok(&mut root, "GRANT SELECT (hid) ON s TO clerk").await;
    ok(
        &mut clerk,
        "UPDATE t SET a = length(row_to_json(s)::text) FROM s",
    )
    .await;
}

#[tokio::test]
async fn an_insert_reads_no_whole_row_without_select_on_every_column() {
    let (engine, mut root) = whole_row_fixture().await;
    for sql in [
        "CREATE UNIQUE INDEX t_part ON t (a) WHERE id > 100",
        "GRANT INSERT, UPDATE (a), SELECT (id, a) ON t TO clerk",
    ] {
        ok(&mut root, sql).await;
    }
    let before = table_t(&mut root).await;
    let mut clerk = open(&engine, "clerk").await;
    for sql in [
        "INSERT INTO t VALUES (2, 3, 'x') RETURNING row_to_json(t)",
        "INSERT INTO t VALUES (1, 3, 'x') ON CONFLICT (id) DO NOTHING RETURNING row_to_json(t)",
        "INSERT INTO t VALUES (1, 3, 'x') ON CONFLICT (id) DO UPDATE SET a = 1 \
         RETURNING row_to_json(t)",
        "INSERT INTO t VALUES (1, 3, 'x') ON CONFLICT (id) DO UPDATE SET a = 1 \
         WHERE row_to_json(t)::text LIKE '%hunter2%'",
        "INSERT INTO t VALUES (1, 3, 'x') ON CONFLICT (id) DO UPDATE \
         SET a = length(row_to_json(t)::text)",
        "INSERT INTO t VALUES (200, 3, 'x') ON CONFLICT (a) \
         WHERE row_to_json(t)::text LIKE '%hunter2%' DO NOTHING",
        "INSERT INTO t VALUES (200, 3, 'x') ON CONFLICT (a) \
         WHERE secret = 'hunter2' DO NOTHING",
        "INSERT INTO t VALUES (200, 3, 'x') ON CONFLICT (a) \
         WHERE EXISTS (SELECT 1 WHERE secret = 'hunter2') DO NOTHING",
    ] {
        denied_both(&mut clerk, sql).await;
    }
    assert_eq!(table_t(&mut root).await, before);
}

/// A temporary table is a table of the session, and still no way to add a child to a parent of
/// another role.
#[tokio::test]
async fn a_temporary_table_adds_no_child_to_a_parent_of_another_role() {
    let (engine, mut root) = whole_row_fixture().await;
    for sql in [
        "CREATE TABLE p (id INT, v TEXT) PARTITION BY RANGE (id)",
        "CREATE TABLE plain (id INT, v TEXT)",
    ] {
        ok(&mut root, sql).await;
    }
    let mut clerk = open(&engine, "clerk").await;
    denied(
        &mut clerk,
        "CREATE TEMPORARY TABLE stolen PARTITION OF p FOR VALUES FROM (0) TO (10)",
    )
    .await;
    denied(
        &mut clerk,
        "CREATE TEMPORARY TABLE stolen (extra INT) INHERITS (plain)",
    )
    .await;
}
