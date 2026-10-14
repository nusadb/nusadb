//! The transaction clock over the wire: `now()` reports the start of the open transaction block in
//! every statement of it, whichever way a statement arrives (simple query, a multi-statement query
//! running as one implicit transaction, the extended protocol, a prepared statement, `COPY`), while
//! `statement_timestamp()` moves on with each statement. Outside a block each statement is its own
//! transaction.

#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic/indexing over linear protocol scripts"
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

/// The single value of a one-row, one-column result.
async fn scalar(conn: &mut Conn, sql: &str) -> String {
    let out = q(conn, sql).await;
    let values: Vec<&String> = out.iter().filter(|m| m.starts_with("D:")).collect();
    assert_eq!(values.len(), 1, "{sql}: {out:?}");
    values[0].trim_start_matches("D:").to_owned()
}

/// Let the wall clock move on, so two statements cannot share a microsecond by chance.
async fn tick() {
    tokio::time::sleep(std::time::Duration::from_millis(3)).await;
}

#[tokio::test]
async fn now_is_fixed_for_an_explicit_transaction_block() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;

    assert_eq!(q(&mut c, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    tick().await;
    let first = scalar(&mut c, "SELECT now()").await;
    tick().await;
    let second = scalar(&mut c, "SELECT now()").await;
    assert_eq!(
        first, second,
        "now() is the block's start in each statement"
    );
    assert_eq!(
        scalar(&mut c, "SELECT transaction_timestamp()").await,
        first
    );
    assert_eq!(scalar(&mut c, "SELECT current_timestamp").await, first);
    assert_eq!(
        scalar(
            &mut c,
            "SELECT CASE WHEN statement_timestamp() > now() THEN 'later' ELSE 'same' END"
        )
        .await,
        "later"
    );
    // The extended protocol reads the same instant.
    assert_eq!(
        ext(&mut c, "SELECT now()", &[]).await,
        ["T", format!("D:{first}").as_str(), "C:SELECT 1", "Z:T"]
    );
    assert_eq!(q(&mut c, "COMMIT").await, ["C:COMMIT", "Z:I"]);

    // A new block starts later; in autocommit each statement is its own transaction.
    tick().await;
    let next = scalar(&mut c, "SELECT now()").await;
    assert_ne!(next, first);
    assert_eq!(
        scalar(
            &mut c,
            "SELECT CASE WHEN now() = statement_timestamp() THEN 'same' ELSE 'apart' END"
        )
        .await,
        "same"
    );
    drop(c);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_multi_statement_query_shares_one_now() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;

    let out = q(
        &mut c,
        "SELECT now(); SELECT count(*) FROM generate_series(1, 300000); \
         SELECT now(), CASE WHEN statement_timestamp() > now() THEN 'later' ELSE 'same' END",
    )
    .await;
    let rows: Vec<&String> = out.iter().filter(|m| m.starts_with("D:")).collect();
    assert_eq!(rows.len(), 3, "{out:?}");
    let first = rows[0].trim_start_matches("D:");
    assert_eq!(*rows[2], format!("D:{first},later"), "{out:?}");
    assert_eq!(out.last().map(String::as_str), Some("Z:I"));

    // The next query is a new implicit transaction.
    tick().await;
    assert_ne!(scalar(&mut c, "SELECT now()").await, first);
    drop(c);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn defaults_copy_and_prepared_statements_use_the_block_start() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;
    q(
        &mut c,
        "CREATE TABLE ev (id INT, at TIMESTAMPTZ DEFAULT now())",
    )
    .await;
    q(&mut c, "PREPARE p AS SELECT now()").await;

    q(&mut c, "BEGIN").await;
    let start = scalar(&mut c, "SELECT now()").await;
    tick().await;
    q(&mut c, "INSERT INTO ev (id) VALUES (1)").await;
    tick().await;
    assert_eq!(
        copy_in(&mut c, "COPY ev (id) FROM STDIN", b"2\n").await,
        ["C:COPY 1", "Z:T"]
    );
    tick().await;
    assert_eq!(
        ext(&mut c, "INSERT INTO ev (id) VALUES ($1)", &["3"]).await,
        ["C:INSERT 1", "Z:T"]
    );
    assert_eq!(scalar(&mut c, "EXECUTE p").await, start);
    assert_eq!(
        scalar(
            &mut c,
            "SELECT count(DISTINCT at) || ' ' || CASE WHEN min(at) = now() THEN 'start' ELSE 'other' END FROM ev"
        )
        .await,
        "1 start"
    );
    q(&mut c, "COMMIT").await;

    // The prepared statement is evaluated afresh in the next block.
    tick().await;
    q(&mut c, "BEGIN").await;
    let later = scalar(&mut c, "EXECUTE p").await;
    assert_eq!(scalar(&mut c, "SELECT now()").await, later);
    q(&mut c, "COMMIT").await;
    assert_ne!(later, start);
    drop(c);
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn recovery_to_a_savepoint_keeps_the_block_start() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, handle) = open(&engine, "nusadb-root").await;
    q(&mut c, "BEGIN").await;
    let start = scalar(&mut c, "SELECT now()").await;
    q(&mut c, "SAVEPOINT s").await;
    let failed = q(&mut c, "SELECT 1 / 0").await;
    assert_eq!(failed.last().map(String::as_str), Some("Z:E"), "{failed:?}");
    q(&mut c, "ROLLBACK TO SAVEPOINT s").await;
    tick().await;
    assert_eq!(scalar(&mut c, "SELECT now()").await, start);
    q(&mut c, "COMMIT").await;
    drop(c);
    handle.await.unwrap().unwrap();
}
