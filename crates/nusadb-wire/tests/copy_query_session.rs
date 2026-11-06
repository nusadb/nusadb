//! `COPY (<query>) TO STDOUT` over a connection runs its query under the connection's session: its
//! time zone, its other settings, the block's start time and the role it acts as, for the plan and
//! for the run alike. COPY is rendered on a blocking worker that other connections use too.

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

/// Run `sql` and return what it sent back: the fields of its rows or its `COPY` payload, one line
/// per row, or the SQLSTATE of its error.
async fn run(conn: &mut Conn, sql: &str) -> Result<String, String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut out = String::new();
    let mut refused = None;
    loop {
        match next(conn).await {
            BackendMessage::DataRow { values } => {
                let fields: Vec<String> = values
                    .into_iter()
                    .map(|v| String::from_utf8(v.unwrap_or_default()).unwrap())
                    .collect();
                out.push_str(&fields.join("\t"));
                out.push('\n');
            },
            BackendMessage::CopyData { data } => {
                out.push_str(&String::from_utf8(data).unwrap());
            },
            BackendMessage::Error { code, .. } => refused = Some(code),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    refused.map_or(Ok(out), Err)
}

/// Run `sql` and expect it to succeed.
async fn ok(conn: &mut Conn, sql: &str) -> String {
    run(conn, sql)
        .await
        .unwrap_or_else(|code| panic!("`{sql}` failed with {code}"))
}

/// An engine with `ts_t` holding one instant, and a connection in zone `+07`.
async fn zone_fixture() -> (Arc<dyn StorageEngine>, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut conn = open(&engine, "nusadb-root").await;
    ok(&mut conn, "CREATE TABLE ts_t (at TIMESTAMPTZ)").await;
    ok(
        &mut conn,
        "INSERT INTO ts_t VALUES ('2026-01-01 00:00:00+00')",
    )
    .await;
    ok(&mut conn, "SET TIME ZONE '+07'").await;
    (engine, conn)
}

#[tokio::test]
async fn copy_query_renders_values_in_the_session_zone() {
    let (_engine, mut conn) = zone_fixture().await;
    let selected = ok(&mut conn, "SELECT at FROM ts_t").await;
    assert!(selected.contains("07:00:00+07"), "{selected}");
    assert_eq!(
        ok(&mut conn, "COPY (SELECT at FROM ts_t) TO STDOUT").await,
        selected
    );
    assert_eq!(ok(&mut conn, "COPY ts_t TO STDOUT").await, selected);
}

#[tokio::test]
async fn copy_query_evaluates_in_the_session_zone() {
    let (_engine, mut conn) = zone_fixture().await;
    let matched = "(SELECT at::timestamp FROM ts_t WHERE at::timestamp = '2026-01-01 07:00:00')";
    assert_eq!(
        ok(&mut conn, &format!("COPY {matched} TO STDOUT")).await,
        "2026-01-01 07:00:00\n"
    );
}

#[tokio::test]
async fn copy_query_reports_the_session_zone() {
    let (_engine, mut conn) = zone_fixture().await;
    let shown = ok(&mut conn, "SELECT current_setting('timezone')").await;
    assert_eq!(
        ok(
            &mut conn,
            "COPY (SELECT current_setting('timezone')) TO STDOUT"
        )
        .await,
        shown
    );
    assert_eq!(shown, "<+07>-07\n");
}

/// A zone at offset zero is still the zone the session spelled.
#[tokio::test]
async fn copy_query_keeps_the_zone_as_the_session_spelled_it() {
    let (_engine, mut conn) = zone_fixture().await;
    ok(&mut conn, "SET TIME ZONE 'Etc/GMT'").await;
    let selected = ok(
        &mut conn,
        "SELECT at, current_setting('timezone') FROM ts_t",
    )
    .await;
    assert!(selected.ends_with("Etc/GMT\n"), "{selected}");
    assert_eq!(
        ok(
            &mut conn,
            "COPY (SELECT at, current_setting('timezone') FROM ts_t) TO STDOUT"
        )
        .await,
        selected
    );
}

/// The block's start time reaches the query, as for any statement of the block.
#[tokio::test]
async fn copy_query_in_a_block_sees_the_start_of_the_block() {
    let (_engine, mut conn) = zone_fixture().await;
    ok(&mut conn, "BEGIN").await;
    let started = ok(&mut conn, "SELECT now()").await;
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert_eq!(
        ok(&mut conn, "COPY (SELECT now()) TO STDOUT").await,
        started
    );
    assert_eq!(ok(&mut conn, "SELECT now()").await, started);
    ok(&mut conn, "ROLLBACK").await;
}

#[tokio::test]
async fn copy_query_reads_the_session_settings() {
    let (_engine, mut conn) = zone_fixture().await;
    ok(&mut conn, "SET my.tag = 'blue'").await;
    assert_eq!(
        ok(
            &mut conn,
            "COPY (SELECT current_setting('my.tag')) TO STDOUT"
        )
        .await,
        "blue\n"
    );
    ok(
        &mut conn,
        "SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE",
    )
    .await;
    let shown = ok(&mut conn, "SELECT current_setting('transaction_isolation')").await;
    assert_eq!(
        ok(
            &mut conn,
            "COPY (SELECT current_setting('transaction_isolation')) TO STDOUT"
        )
        .await,
        shown
    );
}

#[tokio::test]
async fn copy_query_resolves_names_like_the_session() {
    let (_engine, mut conn) = zone_fixture().await;
    ok(&mut conn, "CREATE SCHEMA app").await;
    ok(&mut conn, "CREATE TABLE app.only_here (id INT)").await;
    ok(&mut conn, "INSERT INTO app.only_here VALUES (4)").await;
    ok(&mut conn, "SET search_path = app").await;
    let query = "SELECT id, current_schema() FROM only_here";
    let selected = ok(&mut conn, query).await;
    assert_eq!(selected, "4\tapp\n");
    assert_eq!(
        ok(&mut conn, &format!("COPY ({query}) TO STDOUT")).await,
        selected
    );
    // A temporary table shadows through the session, on whichever worker the COPY lands.
    ok(&mut conn, "CREATE TEMP TABLE scratch (id INT)").await;
    ok(&mut conn, "INSERT INTO scratch VALUES (5)").await;
    assert_eq!(
        ok(&mut conn, "COPY (SELECT id FROM scratch) TO STDOUT").await,
        "5\n"
    );
}

/// A role the session switched to is the user the query runs as.
#[tokio::test]
async fn copy_query_runs_as_the_role_the_session_acts_as() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    ok(&mut root, "CREATE ROLE alice LOGIN").await;
    ok(&mut root, "SET TIME ZONE '+07'").await;
    ok(&mut root, "SET ROLE alice").await;
    assert_eq!(
        ok(
            &mut root,
            "COPY (SELECT current_user, current_setting('timezone')) TO STDOUT"
        )
        .await,
        "alice\t<+07>-07\n"
    );
    ok(&mut root, "RESET ROLE").await;
    assert_eq!(
        ok(&mut root, "COPY (SELECT current_user) TO STDOUT").await,
        "nusadb-root\n"
    );
}

/// Connections taking turns on the blocking workers each keep their own user and their own zone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_copies_keep_their_own_user_and_zone() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut root = open(&engine, "nusadb-root").await;
    ok(&mut root, "CREATE ROLE alice LOGIN").await;
    ok(&mut root, "CREATE ROLE bob LOGIN").await;
    let copy = "COPY (SELECT current_user, current_setting('timezone')) TO STDOUT";
    let mut tasks = Vec::new();
    for (user, zone, shown) in [("alice", "+07", "<+07>-07"), ("bob", "-03", "<-03>+03")] {
        let engine = Arc::clone(&engine);
        tasks.push(tokio::spawn(async move {
            let mut conn = open(&engine, user).await;
            ok(&mut conn, &format!("SET TIME ZONE '{zone}'")).await;
            let expected = format!("{user}\t{shown}\n");
            for _ in 0..200 {
                assert_eq!(ok(&mut conn, copy).await, expected);
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
}
