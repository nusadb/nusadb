//! Multi-statement simple queries, and statements of an open transaction block run through the
//! extended protocol and COPY, driven over an in-memory duplex stream against a real
//! `BtreeEngine`.

#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic over linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, DescribeTarget, FrontendMessage, TxnStatus, handle_client,
};
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
async fn open(
    engine: &Arc<dyn StorageEngine>,
    user: &str,
) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
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
    (conn, handle)
}

/// Render one backend message compactly: `C:<tag>`, `E:<sqlstate>`, `D:<fields>`, `T` for a row
/// description, `Z:<I|T|E>`, or the message's debug form.
fn render(msg: &BackendMessage) -> String {
    match msg {
        BackendMessage::CommandComplete { tag } => format!("C:{tag}"),
        BackendMessage::Error { code, .. } => format!("E:{code}"),
        BackendMessage::DataRow { values } => {
            let fields: Vec<String> = values
                .iter()
                .map(|v| {
                    v.as_ref().map_or_else(
                        || "NULL".to_owned(),
                        |b| String::from_utf8_lossy(b).into_owned(),
                    )
                })
                .collect();
            format!("D:{}", fields.join(","))
        },
        BackendMessage::RowDescription { .. } | BackendMessage::RowDescriptionTyped { .. } => {
            "T".to_owned()
        },
        BackendMessage::ReadyForQuery(status) => match status {
            TxnStatus::Idle => "Z:I".to_owned(),
            TxnStatus::InTransaction => "Z:T".to_owned(),
            TxnStatus::Failed => "Z:E".to_owned(),
        },
        other => format!("{other:?}"),
    }
}

/// Every message up to and including `ReadyForQuery`, rendered.
async fn until_ready(conn: &mut Conn) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(render(&msg));
        if done {
            return out;
        }
    }
}

/// Send a simple query and return its rendered response.
async fn q(conn: &mut Conn, sql: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    until_ready(conn).await
}

/// Run `sql` through Parse / Bind / Describe(portal) / Execute / Sync with text parameters.
async fn ext(conn: &mut Conn, sql: &str, params: &[&str]) -> Vec<String> {
    for msg in [
        FrontendMessage::Parse {
            name: String::new(),
            sql: sql.to_owned(),
            param_types: vec![],
        },
        FrontendMessage::Bind {
            portal: String::new(),
            statement: String::new(),
            params: params.iter().map(|p| Some(p.as_bytes().to_vec())).collect(),
            result_formats: vec![],
        },
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: String::new(),
        },
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 0,
        },
        FrontendMessage::Sync,
    ] {
        send(conn, msg).await;
    }
    until_ready(conn)
        .await
        .into_iter()
        .filter(|m| m != "ParseComplete" && m != "BindComplete" && m != "NoData")
        .collect()
}

/// `COPY ... FROM STDIN` with `data` as the whole payload.
async fn copy_in(conn: &mut Conn, sql: &str, data: &[u8]) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    match next(conn).await {
        BackendMessage::CopyInResponse { .. } => {},
        other => {
            let mut out = vec![render(&other)];
            if !matches!(other, BackendMessage::ReadyForQuery(_)) {
                out.extend(until_ready(conn).await);
            }
            return out;
        },
    }
    send(
        conn,
        FrontendMessage::CopyData {
            data: data.to_vec(),
        },
    )
    .await;
    send(conn, FrontendMessage::CopyDone).await;
    until_ready(conn).await
}

#[tokio::test]
async fn multi_statement_query_runs_each_statement_in_one_implicit_transaction() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;

    // Each statement sends its own result; one ReadyForQuery ends the message.
    assert_eq!(
        q(&mut c, "SELECT 1; SELECT 2").await,
        ["T", "D:1", "C:SELECT 1", "T", "D:2", "C:SELECT 1", "Z:I"]
    );
    assert_eq!(
        q(
            &mut c,
            "CREATE TABLE ms (id INT); INSERT INTO ms VALUES (1);"
        )
        .await,
        ["C:CREATE TABLE", "C:INSERT 1", "Z:I"]
    );
    // A `;` inside a string, a dollar-quoted body or a comment does not split.
    assert_eq!(
        q(&mut c, "SELECT 'a;b'; SELECT $$x;y$$ -- c; d\n").await,
        [
            "T",
            "D:a;b",
            "C:SELECT 1",
            "T",
            "D:x;y",
            "C:SELECT 1",
            "Z:I"
        ]
    );

    // An error stops the query and rolls back everything it ran.
    assert_eq!(
        q(
            &mut c,
            "INSERT INTO ms VALUES (2); SELECT 1/0; INSERT INTO ms VALUES (3)"
        )
        .await,
        ["C:INSERT 1", "T", "E:22012", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "SELECT id FROM ms ORDER BY id").await,
        ["T", "D:1", "C:SELECT 1", "Z:I"]
    );

    // COMMIT ends the implicit transaction early; what follows runs in a new one.
    assert_eq!(
        q(
            &mut c,
            "INSERT INTO ms VALUES (4); COMMIT; INSERT INTO ms VALUES (5); SELECT 1/0"
        )
        .await,
        [
            "C:INSERT 1",
            "C:COMMIT",
            "C:INSERT 1",
            "T",
            "E:22012",
            "Z:I"
        ]
    );
    assert_eq!(
        q(&mut c, "SELECT id FROM ms ORDER BY id").await,
        ["T", "D:1", "D:4", "C:SELECT 2", "Z:I"]
    );

    // BEGIN turns the implicit transaction into a block that keeps what already ran, and the
    // block stays open after the query.
    assert_eq!(
        q(
            &mut c,
            "INSERT INTO ms VALUES (6); BEGIN; INSERT INTO ms VALUES (7)"
        )
        .await,
        ["C:INSERT 1", "C:BEGIN", "C:INSERT 1", "Z:T"]
    );
    assert_eq!(q(&mut c, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);
    assert_eq!(
        q(&mut c, "SELECT count(*) FROM ms").await,
        ["T", "D:2", "C:SELECT 1", "Z:I"]
    );

    // Savepoints need a transaction block, and BEGIN options can only come first.
    assert_eq!(
        q(&mut c, "SELECT 1; SAVEPOINT a").await,
        ["T", "D:1", "C:SELECT 1", "E:25P01", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "SELECT 1; BEGIN ISOLATION LEVEL SERIALIZABLE").await,
        ["T", "D:1", "C:SELECT 1", "E:25001", "Z:I"]
    );
    assert_eq!(
        q(
            &mut c,
            "BEGIN; INSERT INTO ms VALUES (8); SAVEPOINT a; INSERT INTO ms VALUES (9); \
             ROLLBACK TO a; COMMIT"
        )
        .await,
        [
            "C:BEGIN",
            "C:INSERT 1",
            "C:SAVEPOINT",
            "C:INSERT 1",
            "C:ROLLBACK",
            "C:COMMIT",
            "Z:I"
        ]
    );

    // A statement refused before it reaches the engine still rolls back what ran before it.
    assert_eq!(
        q(&mut c, "INSERT INTO ms VALUES (13); SAVEPOINT a").await,
        ["C:INSERT 1", "E:25P01", "Z:I"]
    );

    // Inside an explicit block the statements run in it, and an error fails the block.
    assert_eq!(q(&mut c, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        q(&mut c, "INSERT INTO ms VALUES (10); SELECT 1/0; SELECT 2").await,
        ["C:INSERT 1", "T", "E:22012", "Z:E"]
    );
    assert_eq!(q(&mut c, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);

    // Nothing runs when any statement fails to parse, and COPY must be sent on its own.
    assert_eq!(
        q(&mut c, "INSERT INTO ms VALUES (11); SELEC 2").await,
        ["E:42601", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "INSERT INTO ms VALUES (12); COPY ms FROM STDIN").await,
        ["E:0A000", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "SELECT id FROM ms ORDER BY id").await,
        ["T", "D:1", "D:4", "D:8", "C:SELECT 3", "Z:I"]
    );

    // A query with no statement at all is an error; a dollar-quoted string with a digit tag is
    // one string, not three statements.
    assert_eq!(q(&mut c, ";;").await, ["E:42601", "Z:I"]);
    assert_eq!(q(&mut c, "-- only a comment").await, ["E:42601", "Z:I"]);
    assert_eq!(
        q(&mut c, "SELECT $1$; SELECT 2; SELECT $1$").await,
        ["T", "D:; SELECT 2; SELECT ", "C:SELECT 1", "Z:I"]
    );

    // The extended protocol still takes exactly one statement.
    assert_eq!(
        ext(&mut c, "SELECT 1; SELECT 2", &[]).await,
        ["E:42601", "Z:I"]
    );

    drop(c);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn extended_protocol_and_copy_see_the_blocks_own_uncommitted_table() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut root, root_handle) = open(&engine, "nusadb-root").await;
    for sql in [
        "CREATE ROLE alice LOGIN",
        "GRANT CREATE ON SCHEMA public TO alice",
        "CREATE TABLE secret (id INT)",
        "CREATE TABLE shared (id INT)",
    ] {
        assert_eq!(q(&mut root, sql).await.last().unwrap(), "Z:I", "{sql}");
    }

    let (mut a, a_handle) = open(&engine, "alice").await;
    assert_eq!(q(&mut a, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        q(&mut a, "CREATE TABLE own_t (id INT)").await,
        ["C:CREATE TABLE", "Z:T"]
    );
    // The owner of a table created in this block may use it through Parse/Bind/Execute, exactly as
    // through a simple query.
    assert_eq!(
        ext(&mut a, "INSERT INTO own_t VALUES ($1)", &["5"]).await,
        ["C:INSERT 1", "Z:T"]
    );
    assert_eq!(
        ext(&mut a, "UPDATE own_t SET id = $1 WHERE id = 5", &["6"]).await,
        ["C:UPDATE 1", "Z:T"]
    );
    assert_eq!(
        ext(&mut a, "SELECT id FROM own_t WHERE id = $1", &["6"]).await,
        ["T", "D:6", "C:SELECT 1", "Z:T"]
    );
    assert_eq!(
        ext(&mut a, "DELETE FROM own_t WHERE id = $1", &["6"]).await,
        ["C:DELETE 1", "Z:T"]
    );
    // COPY runs in the block too.
    assert_eq!(
        copy_in(&mut a, "COPY own_t FROM STDIN", b"7\n").await,
        ["C:COPY 1", "Z:T"]
    );
    assert_eq!(
        q(&mut a, "SELECT id FROM own_t").await,
        ["T", "D:7", "C:SELECT 1", "Z:T"]
    );
    // The block's transaction grants nothing the role lacks.
    assert_eq!(
        ext(&mut a, "SELECT id FROM secret WHERE id = $1", &["1"])
            .await
            .first()
            .map(String::as_str),
        Some("E:42501")
    );
    assert_eq!(q(&mut a, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);
    assert_eq!(q(&mut a, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        copy_in(&mut a, "COPY secret FROM STDIN", b"1\n").await,
        ["E:42501", "Z:E"]
    );
    assert_eq!(q(&mut a, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);
    // The rolled-back table is gone.
    assert_eq!(q(&mut a, "SELECT 1 FROM own_t").await, ["E:42P01", "Z:I"]);

    // A COPY inside a block is undone by its ROLLBACK and kept by its COMMIT.
    assert_eq!(q(&mut root, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        copy_in(&mut root, "COPY shared FROM STDIN", b"1\n").await,
        ["C:COPY 1", "Z:T"]
    );
    assert_eq!(q(&mut root, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);
    assert_eq!(q(&mut root, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        copy_in(&mut root, "COPY shared FROM STDIN", b"2\n").await,
        ["C:COPY 1", "Z:T"]
    );
    assert_eq!(q(&mut root, "COMMIT").await, ["C:COMMIT", "Z:I"]);
    assert_eq!(
        q(&mut root, "SELECT id FROM shared").await,
        ["T", "D:2", "C:SELECT 1", "Z:I"]
    );
    // A failed COPY fails the block; in a failed block COPY is refused.
    assert_eq!(q(&mut root, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    assert_eq!(
        copy_in(&mut root, "COPY shared FROM STDIN", b"x\n").await,
        ["E:22P02", "Z:E"]
    );
    assert_eq!(
        copy_in(&mut root, "COPY shared FROM STDIN", b"3\n").await,
        ["E:25P02", "Z:E"]
    );
    assert_eq!(q(&mut root, "ROLLBACK").await, ["C:ROLLBACK", "Z:I"]);

    drop(a);
    drop(root);
    a_handle.await.unwrap().unwrap();
    root_handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn copy_into_a_composite_column_checks_each_value() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;
    for sql in [
        "CREATE TYPE inner_t AS (x INT, t TEXT)",
        "CREATE TYPE outer_t AS (y inner_t, z INT)",
        "CREATE TYPE cx AS (b BYTEA, i INTERVAL, a INT[])",
        "CREATE TABLE ot (id INT, v outer_t)",
        "CREATE TABLE cxt (id INT, v cx)",
    ] {
        assert_eq!(q(&mut c, sql).await.last().unwrap(), "Z:I", "{sql}");
    }
    // A value that does not fit the type is refused, at any nesting level.
    assert_eq!(
        copy_in(&mut c, "COPY ot FROM STDIN", b"1\t(\"(abc,x)\",3)\n").await,
        ["E:22023", "Z:I"]
    );
    assert_eq!(
        copy_in(
            &mut c,
            "COPY cxt FROM STDIN",
            b"1\t(,\"forever\",\"{1}\")\n"
        )
        .await,
        ["E:22023", "Z:I"]
    );
    // A valid one is stored in its canonical form and reads back field by field.
    assert_eq!(
        copy_in(&mut c, "COPY ot FROM STDIN", b"2\t(\"(7,\"\"w\"\")\",1)\n").await,
        ["C:COPY 1", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "SELECT v, (v).y.x FROM ot").await,
        ["T", "D:(\"(7,w)\",1),7", "C:SELECT 1", "Z:I"]
    );
    assert_eq!(
        copy_in(
            &mut c,
            "COPY cxt FROM STDIN",
            b"3\t(\"\\\\\\\\x03\",\"3 days\",\"{4,5}\")\n"
        )
        .await,
        ["C:COPY 1", "Z:I"]
    );
    assert_eq!(
        q(&mut c, "SELECT (v).b, (v).i, (v).a FROM cxt").await,
        ["T", "D:\\x03,3 days,{4,5}", "C:SELECT 1", "Z:I"]
    );
    drop(c);
    handle.await.unwrap().unwrap();
}
