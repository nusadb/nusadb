//! Access control inside a `WITH RECURSIVE`, and the names a query's own `WITH` can reach.
//!
//! The recursive term of a CTE is analyzed under a catalog that adds the CTE's working table to the
//! real one. These tests pin that nothing else about the catalog changes there: a term needs the
//! same privileges as any other read, sees the same policies, parents, views and functions, and
//! does not need a grant for its own working table. The last group pins that the names a query
//! declares stay in that query: a stored definition keeps meaning what it meant when it was created.

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
type Rows = Vec<Vec<Option<String>>>;

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

/// What a statement came back with: the SQLSTATE of its error, or its rows.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Refused(String),
    Rows(Rows),
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

/// Prepare `sql` as the statement `name`, without running it.
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

/// Run `sql` and expect it not to be refused.
async fn ok(conn: &mut Conn, sql: &str) -> Rows {
    match run(conn, sql).await {
        Outcome::Rows(rows) => rows,
        Outcome::Refused(code) => panic!("`{sql}` was refused with {code}"),
    }
}

/// Run each statement and expect it not to be refused.
async fn setup(conn: &mut Conn, statements: &[&str]) {
    for sql in statements {
        ok(conn, sql).await;
    }
}

/// Expect the same refusal over both protocols.
async fn refused_both(conn: &mut Conn, sql: &str, code: &str) {
    let expected = Outcome::Refused(code.to_owned());
    assert_eq!(run(conn, sql).await, expected, "{sql}");
    assert_eq!(
        run_extended(conn, sql).await,
        expected,
        "{sql} over the extended protocol"
    );
}

/// Expect the same rows over both protocols.
async fn rows_both(conn: &mut Conn, sql: &str, expected: &[&[Option<&str>]]) {
    let expected = Outcome::Rows(
        expected
            .iter()
            .map(|row| row.iter().map(|v| v.map(str::to_owned)).collect())
            .collect(),
    );
    assert_eq!(run(conn, sql).await, expected, "{sql}");
    assert_eq!(
        run_extended(conn, sql).await,
        expected,
        "{sql} over the extended protocol"
    );
}

/// An engine with `t (id, secret)`, one row, and a role with no privilege on it.
async fn fixture() -> (Arc<dyn StorageEngine>, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    setup(
        &mut root,
        &[
            "CREATE TABLE t (id INT, secret TEXT)",
            "INSERT INTO t VALUES (1, 'kept')",
            "CREATE ROLE clerk LOGIN",
        ],
    )
    .await;
    (engine, root)
}

/// A term with an empty anchor row: `r (n, s)`, where `s` is what the term reads, one row deep.
fn term(projection: &str, from: &str) -> String {
    format!(
        "WITH RECURSIVE r(n, s) AS (SELECT 1, ''::text UNION ALL \
         SELECT n + 1, {projection} FROM {from} WHERE n < 2) SELECT * FROM r"
    )
}

/// Every way a recursive term reads `t` asks for the same privilege a plain read does.
#[tokio::test]
async fn a_recursive_term_needs_select_like_any_other_read() {
    let (engine, _root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    let reads = [
        term("x.secret", "r, t x"),
        term("x.secret", "t x, r"),
        term("row_to_json(x)::text", "r, t x"),
        term("(SELECT min(secret) FROM t)", "r"),
        term("(SELECT y.s FROM (SELECT secret AS s FROM t) y)", "r"),
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r \
         WHERE n < 2 AND EXISTS (SELECT 1 FROM t WHERE secret LIKE 'k%')) SELECT * FROM r"
            .to_owned(),
        format!("EXPLAIN {}", term("x.secret", "r, t x")),
    ];
    for sql in &reads {
        refused_both(&mut clerk, sql, "42501").await;
    }
    refused_both(&mut clerk, "SELECT * FROM t", "42501").await;
}

/// A partition read by its own name is that table's privilege, not its parent's.
#[tokio::test]
async fn a_partition_named_in_a_term_is_checked_as_itself() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE TABLE tp (id INT, v TEXT) PARTITION BY RANGE (id)",
            "CREATE TABLE tp1 PARTITION OF tp FOR VALUES FROM (0) TO (100)",
            "INSERT INTO tp VALUES (1, 'kept')",
            "GRANT SELECT ON tp TO clerk",
        ],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    refused_both(&mut clerk, &term("x.v", "r, tp1 x"), "42501").await;
}

/// The working table takes no grant, whatever else the term reads and under whatever name.
#[tokio::test]
async fn a_term_needs_no_grant_on_its_own_working_table() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "GRANT SELECT ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    let expected: &[&[Option<&str>]] = &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]];
    rows_both(&mut clerk, &term("x.secret", "r, t x"), expected).await;
    rows_both(&mut clerk, &term("x.secret", "t x, r"), expected).await;
    rows_both(
        &mut clerk,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
         SELECT count(*) FROM r",
        &[&[Some("3")]],
    )
    .await;
    // The self-reference in a nested subquery and a derived table of the term.
    rows_both(
        &mut clerk,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT d.n + 1 FROM (SELECT n FROM r) d \
         WHERE d.n < 3 AND EXISTS (SELECT 1 FROM r r2 WHERE r2.n = d.n)) SELECT max(n) FROM r",
        &[&[Some("3")]],
    )
    .await;
    // Its whole row, its columns by a column list of its own, and a CTE named after a real table.
    rows_both(
        &mut clerk,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r \
         WHERE length(row_to_json(r)::text) > 0 AND n < 2) SELECT max(n) FROM r",
        &[&[Some("2")]],
    )
    .await;
    rows_both(
        &mut clerk,
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 2) \
         SELECT max(n) FROM t",
        &[&[Some("2")]],
    )
    .await;
    // Reached through the session's temp schema probe too.
    ok(&mut clerk, "CREATE TEMP TABLE scratch (a INT)").await;
    rows_both(
        &mut clerk,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 2) \
         SELECT max(n) FROM r",
        &[&[Some("2")]],
    )
    .await;
}

/// A CTE named after a table does not lend its grant to the same name in another schema.
#[tokio::test]
async fn the_working_table_grants_nothing_on_a_table_of_its_name_elsewhere() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE SCHEMA app",
            "CREATE TABLE app.r (a INT, s TEXT)",
            "INSERT INTO app.r VALUES (1, 'kept')",
            "GRANT USAGE ON SCHEMA app TO clerk",
        ],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    refused_both(&mut clerk, &term("x.s", "r, app.r x"), "42501").await;
    refused_both(
        &mut clerk,
        &term("(SELECT max(s) FROM app.r)", "r"),
        "42501",
    )
    .await;
    ok(&mut root, "GRANT SELECT ON app.r TO clerk").await;
    rows_both(
        &mut clerk,
        &term("x.s", "r, app.r x"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]],
    )
    .await;
}

/// A column grant is honoured per column in a term, as it is outside one.
#[tokio::test]
async fn a_column_grant_in_a_term_covers_its_columns_and_no_others() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "GRANT SELECT (id) ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    refused_both(&mut clerk, &term("x.secret", "r, t x"), "42501").await;
    refused_both(&mut clerk, &term("row_to_json(x)::text", "r, t x"), "42501").await;
    refused_both(
        &mut clerk,
        &term("(SELECT min(secret) FROM t)", "r"),
        "42501",
    )
    .await;
    rows_both(
        &mut clerk,
        &term("x.id::text", "r, t x"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("1")]],
    )
    .await;
    rows_both(
        &mut clerk,
        &term("(SELECT count(*) FROM t)::text", "r"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("1")]],
    )
    .await;
}

/// The owner and a superuser read what they read anywhere else.
#[tokio::test]
async fn the_owner_and_a_superuser_are_not_affected() {
    let (engine, mut root) = fixture().await;
    let expected: &[&[Option<&str>]] = &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]];
    rows_both(&mut root, &term("x.secret", "r, t x"), expected).await;
    setup(
        &mut root,
        &[
            "CREATE ROLE maker LOGIN CREATEDB",
            "GRANT CREATE ON SCHEMA public TO maker",
        ],
    )
    .await;
    let mut maker = open(&engine, "maker").await;
    setup(
        &mut maker,
        &[
            "CREATE TABLE own (id INT, secret TEXT)",
            "INSERT INTO own VALUES (1, 'kept')",
        ],
    )
    .await;
    rows_both(&mut maker, &term("x.secret", "r, own x"), expected).await;
}

/// A grant taken away reaches a term the next time it runs, cached or prepared.
#[tokio::test]
async fn a_revoke_reaches_the_next_run_of_a_term() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "GRANT SELECT ON t TO clerk").await;
    let mut clerk = open(&engine, "clerk").await;
    let sql = term("x.secret", "r, t x");
    parse(&mut clerk, "held", &sql).await;
    send(&mut clerk, FrontendMessage::Sync).await;
    assert_eq!(outcome(&mut clerk).await, Outcome::Rows(Vec::new()));
    for _ in 0..3 {
        assert!(matches!(run(&mut clerk, &sql).await, Outcome::Rows(_)));
        assert!(matches!(
            execute(&mut clerk, "held").await,
            Outcome::Rows(_)
        ));
    }
    ok(&mut root, "REVOKE SELECT ON t FROM clerk").await;
    assert_eq!(
        run(&mut clerk, &sql).await,
        Outcome::Refused("42501".to_owned())
    );
    assert_eq!(
        execute(&mut clerk, "held").await,
        Outcome::Refused("42501".to_owned())
    );
    ok(&mut root, "GRANT SELECT ON t TO clerk").await;
    assert!(matches!(run(&mut clerk, &sql).await, Outcome::Rows(_)));
    assert!(matches!(
        execute(&mut clerk, "held").await,
        Outcome::Rows(_)
    ));
}

/// A plan one role cached in a term does not answer for another.
#[tokio::test]
async fn a_term_is_not_shared_across_roles() {
    let (engine, mut root) = fixture().await;
    ok(&mut root, "GRANT SELECT ON t TO clerk").await;
    ok(&mut root, "CREATE ROLE stranger LOGIN").await;
    let mut clerk = open(&engine, "clerk").await;
    let mut stranger = open(&engine, "stranger").await;
    let sql = term("x.secret", "r, t x");
    assert!(matches!(run(&mut clerk, &sql).await, Outcome::Rows(_)));
    assert!(matches!(run(&mut clerk, &sql).await, Outcome::Rows(_)));
    refused_both(&mut stranger, &sql, "42501").await;
}

/// A parent read in a term is its whole tree, as it is outside one.
#[tokio::test]
async fn a_parent_read_in_a_term_returns_the_rows_of_its_partitions() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE TABLE tp (id INT, v TEXT) PARTITION BY RANGE (id)",
            "CREATE TABLE tp1 PARTITION OF tp FOR VALUES FROM (0) TO (100)",
            "INSERT INTO tp VALUES (1, 'kept')",
            "CREATE TABLE ip (id INT, v TEXT)",
            "CREATE TABLE ic (extra INT) INHERITS (ip)",
            "INSERT INTO ic VALUES (1, 'child', 5)",
            "GRANT SELECT ON tp, ip TO clerk",
        ],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    for conn in [&mut root, &mut clerk] {
        rows_both(
            conn,
            &term("x.v", "r, tp x"),
            &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]],
        )
        .await;
        rows_both(
            conn,
            &term("x.v", "r, ip x"),
            &[&[Some("1"), Some("")], &[Some("2"), Some("child")]],
        )
        .await;
        rows_both(
            conn,
            &term("(SELECT min(v) FROM tp)", "r"),
            &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]],
        )
        .await;
    }
}

/// Row-level security applies in a term with the session's own role, not a fallback one.
#[tokio::test]
async fn a_term_sees_the_rows_a_policy_gives_the_session() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE TABLE rt (id INT, owner TEXT, secret TEXT)",
            "INSERT INTO rt VALUES (1, 'clerk', 'mine'), (2, 'other', 'theirs')",
            "GRANT SELECT ON rt TO clerk",
            "ALTER TABLE rt ENABLE ROW LEVEL SECURITY",
            "CREATE POLICY own ON rt USING (owner = current_user)",
            "CREATE TABLE rn (id INT, secret TEXT)",
            "INSERT INTO rn VALUES (1, 'named')",
            "GRANT SELECT ON rn TO clerk",
            "ALTER TABLE rn ENABLE ROW LEVEL SECURITY",
            "CREATE POLICY named ON rn TO clerk USING (true)",
        ],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    let single = "WITH RECURSIVE r(n, s) AS (SELECT 1, ''::text UNION \
                  SELECT 2, x.secret FROM rt x) SELECT * FROM r";
    rows_both(
        &mut clerk,
        single,
        &[&[Some("1"), Some("")], &[Some("2"), Some("mine")]],
    )
    .await;
    rows_both(
        &mut clerk,
        &term("(SELECT string_agg(secret, ',') FROM rt)", "r"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("mine")]],
    )
    .await;
    rows_both(
        &mut clerk,
        &term("(SELECT min(secret) FROM rn)", "r"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("named")]],
    )
    .await;
    refused_both(&mut clerk, &term("x.secret", "r, rt x"), "0A000").await;
    // A superuser is not filtered in a term.
    rows_both(
        &mut root,
        &term("(SELECT string_agg(secret, ',' ORDER BY id) FROM rt)", "r"),
        &[&[Some("1"), Some("")], &[Some("2"), Some("mine,theirs")]],
    )
    .await;
}

/// The system catalogs stay closed in a term, and the CTE's own reserved-looking name is no way in.
#[tokio::test]
async fn the_reserved_catalogs_stay_closed_in_a_term() {
    let (engine, _root) = fixture().await;
    let mut clerk = open(&engine, "clerk").await;
    refused_both(&mut clerk, &term("x.name", "r, nusadb_policies x"), "42501").await;
    refused_both(
        &mut clerk,
        &term("(SELECT min(name) FROM nusadb_roles)", "r"),
        "42501",
    )
    .await;
}

/// A view and a function resolve in a term, with the privileges they need anywhere else.
#[tokio::test]
async fn views_and_functions_resolve_in_a_term() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE VIEW tv AS SELECT id, secret FROM t",
            "CREATE FUNCTION addone(x INT) RETURNS INT LANGUAGE SQL AS 'SELECT $1 + 1'",
            "GRANT SELECT ON tv TO clerk",
        ],
    )
    .await;
    let expected: &[&[Option<&str>]] = &[&[Some("1"), Some("")], &[Some("2"), Some("kept")]];
    rows_both(
        &mut root,
        &term("(SELECT min(secret) FROM tv)", "r"),
        expected,
    )
    .await;
    rows_both(
        &mut root,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT addone(n) FROM r WHERE n < 3) \
         SELECT max(n) FROM r",
        &[&[Some("3")]],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    refused_both(&mut clerk, "SELECT * FROM tv", "42501").await;
    refused_both(
        &mut clerk,
        &term("(SELECT min(secret) FROM tv)", "r"),
        "42501",
    )
    .await;
}

/// An engine with `doc` filtered by a policy that reads `allow_t`, and a role that may read both.
async fn policy_fixture() -> (Arc<dyn StorageEngine>, Conn, Conn) {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE TABLE allow_t (id INT)",
            "CREATE TABLE doc (id INT, body TEXT)",
            "INSERT INTO doc VALUES (1, 'shown'), (2, 'filtered')",
            "INSERT INTO allow_t VALUES (1)",
            "GRANT SELECT ON doc, allow_t TO clerk",
            "GRANT UPDATE, DELETE ON doc TO clerk",
            "ALTER TABLE doc ENABLE ROW LEVEL SECURITY",
            "CREATE POLICY p ON doc USING (id IN (SELECT id FROM allow_t))",
        ],
    )
    .await;
    let clerk = open(&engine, "clerk").await;
    (engine, root, clerk)
}

/// A policy names catalog tables. A `WITH` of the query it filters is not in its scope.
#[tokio::test]
async fn a_query_cannot_rename_the_tables_a_policy_reads() {
    let (_engine, _root, mut clerk) = policy_fixture().await;
    let shown: &[&[Option<&str>]] = &[&[Some("1"), Some("shown")]];
    rows_both(&mut clerk, "SELECT * FROM doc", shown).await;
    for sql in [
        "WITH allow_t(id) AS (SELECT 2) SELECT * FROM doc",
        "WITH allow_t(id) AS (SELECT 2) SELECT * FROM (SELECT * FROM doc) d",
        "WITH RECURSIVE allow_t(id) AS (SELECT 2 UNION ALL SELECT id FROM allow_t WHERE false) \
         SELECT * FROM doc",
    ] {
        rows_both(&mut clerk, sql, shown).await;
    }
    rows_both(
        &mut clerk,
        "WITH allow_t(id) AS (SELECT 2) SELECT (SELECT string_agg(body, ',') FROM doc)",
        &[&[Some("shown")]],
    )
    .await;
    // Inside a recursive term, where the name reaches the catalog through the working table.
    rows_both(
        &mut clerk,
        "WITH RECURSIVE allow_t(id, s) AS (SELECT 2, ''::text UNION \
         SELECT id, body FROM doc) SELECT * FROM allow_t",
        &[&[Some("2"), Some("")], &[Some("1"), Some("shown")]],
    )
    .await;
}

/// The same for the rows a write may touch, which the policy selects too.
#[tokio::test]
async fn a_query_cannot_rename_the_tables_a_write_policy_reads() {
    let (_engine, mut root, mut clerk) = policy_fixture().await;
    for sql in [
        "WITH allow_t(id) AS (SELECT 2), w AS (UPDATE doc SET body = 'x' WHERE id = 2 RETURNING id) \
         SELECT * FROM w",
        "WITH allow_t(id) AS (SELECT 2), w AS (DELETE FROM doc WHERE id = 2 RETURNING id) \
         SELECT * FROM w",
    ] {
        assert_eq!(
            run(&mut clerk, sql).await,
            Outcome::Rows(Vec::new()),
            "{sql}"
        );
    }
    assert_eq!(
        ok(&mut root, "SELECT body FROM doc WHERE id = 2").await,
        vec![vec![Some("filtered".to_owned())]]
    );
}

/// A view's body names catalog tables too.
#[tokio::test]
async fn a_query_cannot_rename_the_tables_a_view_reads() {
    let (_engine, mut root, _clerk) = policy_fixture().await;
    setup(
        &mut root,
        &["CREATE VIEW dv AS SELECT id FROM t WHERE id IN (SELECT id FROM allow_t)"],
    )
    .await;
    let expected: &[&[Option<&str>]] = &[&[Some("1")]];
    rows_both(&mut root, "SELECT * FROM dv", expected).await;
    rows_both(
        &mut root,
        "WITH allow_t(id) AS (SELECT 2) SELECT * FROM dv",
        expected,
    )
    .await;
    // Under a working table of that name as well, where a view must not see the term's rows.
    rows_both(
        &mut root,
        "WITH RECURSIVE allow_t(id) AS (SELECT 2 UNION ALL \
         SELECT id + (SELECT max(id) FROM dv) FROM allow_t WHERE id < 3) SELECT id FROM allow_t",
        &[&[Some("2")], &[Some("3")]],
    )
    .await;
}

/// A partition is read as the partition, whatever the query calls its own rows.
#[tokio::test]
async fn a_query_cannot_rename_the_partitions_of_a_parent_it_reads() {
    let (engine, mut root) = fixture().await;
    setup(
        &mut root,
        &[
            "CREATE TABLE tp (id INT, v TEXT) PARTITION BY RANGE (id)",
            "CREATE TABLE tp1 PARTITION OF tp FOR VALUES FROM (0) TO (100)",
            "INSERT INTO tp VALUES (1, 'kept')",
            "GRANT SELECT ON tp TO clerk",
        ],
    )
    .await;
    let mut clerk = open(&engine, "clerk").await;
    rows_both(
        &mut clerk,
        "WITH tp1(id, v) AS (SELECT 9, 'mine') SELECT id, v FROM tp",
        &[&[Some("1"), Some("kept")]],
    )
    .await;
    rows_both(
        &mut clerk,
        "WITH RECURSIVE tp1(id, v) AS (SELECT 9, 'mine'::text UNION \
         SELECT x.id, x.v FROM tp x) SELECT * FROM tp1",
        &[&[Some("9"), Some("mine")], &[Some("1"), Some("kept")]],
    )
    .await;
}
