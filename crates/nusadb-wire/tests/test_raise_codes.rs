//! The SQLSTATE a NusaScript `RAISE` names reaches the client exactly, on the error and on the
//! notice, and no session can change what another one's codes read as: the code travels with the
//! error, so one session raising thousands of distinct codes leaves another's untouched.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, TxnStatus, handle_client};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn next(conn: &mut Conn) -> BackendMessage {
    BackendMessage::decode(&conn.read_frame().await.unwrap().unwrap()).unwrap()
}

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn connect(engine: &Arc<dyn StorageEngine>) -> Conn {
    let (client, server) = tokio::io::duplex(256 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 3,
            user: "nusadb-root".to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    loop {
        match next(&mut conn).await {
            BackendMessage::ReadyForQuery(TxnStatus::Idle) => return conn,
            BackendMessage::BackendKeyData { .. } | BackendMessage::ParameterStatus { .. } => {},
            other => panic!("unexpected handshake message: {other:?}"),
        }
    }
}

/// Every message of one simple query's response, up to and including its `ReadyForQuery`.
async fn run(conn: &mut Conn, sql: &str) -> Vec<BackendMessage> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(msg);
        if done {
            return out;
        }
    }
}

/// The `(code, message)` of the error in a response.
fn error(response: &[BackendMessage]) -> (String, String) {
    response
        .iter()
        .find_map(|m| match m {
            BackendMessage::Error { code, message } => Some((code.clone(), message.clone())),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no error in {response:?}"))
}

fn raise(code: &str) -> String {
    format!("DO $$ BEGIN RAISE 'from {code}' USING ERRCODE = '{code}'; END $$")
}

#[tokio::test]
async fn a_raised_code_reaches_the_client_exactly() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut conn = connect(&engine).await;
    assert_eq!(
        error(&run(&mut conn, &raise("MA001")).await),
        (
            "MA001".to_owned(),
            "raised exception: from MA001".to_owned()
        )
    );
    assert_eq!(
        error(&run(&mut conn, "DO $$ BEGIN RAISE unique_violation; END $$").await).0,
        "23505"
    );
    let notices: Vec<_> = run(
        &mut conn,
        "DO $$ BEGIN RAISE NOTICE 'n' USING ERRCODE = 'MA900'; END $$",
    )
    .await
    .into_iter()
    .filter_map(|m| match m {
        BackendMessage::NoticeResponse { code, message, .. } => Some((code, message)),
        _ => None,
    })
    .collect();
    assert_eq!(notices, vec![("MA900".to_owned(), "n".to_owned())]);
}

/// One session raising far more distinct codes than any table of them could hold leaves another
/// session's code exact, and its own too.
#[tokio::test]
async fn one_session_flooding_codes_does_not_change_another_sessions_codes() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut flood = connect(&engine).await;
    let mut other = connect(&engine).await;
    for i in 0..5000 {
        let code = format!("Q{i:04}");
        assert_eq!(error(&run(&mut flood, &raise(&code)).await).0, code);
    }
    assert_eq!(error(&run(&mut other, &raise("ZZ999")).await).0, "ZZ999");
    assert_eq!(error(&run(&mut flood, &raise("ZY123")).await).0, "ZY123");
}
