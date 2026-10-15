//! The server's reserved settings keys (a leading NUL: the transaction start, the temp schema, the
//! connection database) cannot be read back by a client, whichever protocol it uses: a quoted
//! identifier can carry the NUL, so the name alone does not keep them private.

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

#[tokio::test]
async fn reserved_settings_cannot_be_shown_or_read() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut c, _) = open(&engine, "nusadb-root").await;
    // A temporary table makes the connection stamp its temp schema.
    q(&mut c, "CREATE TEMP TABLE scratch (x INT)").await;
    assert_eq!(q(&mut c, "BEGIN").await, ["C:BEGIN", "Z:T"]);
    q(&mut c, "SELECT now()").await;
    for key in [
        nusadb_sql::TRANSACTION_START_SETTING,
        nusadb_sql::CONNECTION_TEMP_SCHEMA_SETTING,
        nusadb_sql::CONNECTION_DATABASE_SETTING,
    ] {
        let show = format!("SHOW \"{key}\"");
        let simple = q(&mut c, &show).await;
        assert!(
            simple.contains(&"E:42704".to_owned()),
            "{key:?}: {simple:?}"
        );
        q(&mut c, "ROLLBACK").await;
        q(&mut c, "BEGIN").await;
        let extended = ext(&mut c, &show, &[]).await;
        assert!(
            extended.contains(&"E:42704".to_owned()),
            "{key:?}: {extended:?}"
        );
        q(&mut c, "ROLLBACK").await;
        q(&mut c, "BEGIN").await;
        q(&mut c, "SELECT now()").await;
        assert_eq!(
            q(&mut c, &format!("SELECT current_setting('{key}')")).await,
            ["T", "D:NULL", "C:SELECT 1", "Z:T"],
            "{key:?}"
        );
    }
    q(&mut c, "ROLLBACK").await;
    // An ordinary parameter still reads back.
    q(&mut c, "SET search_path = public").await;
    assert_eq!(
        q(&mut c, "SHOW search_path").await,
        ["T", "D:public", "C:SHOW", "Z:I"]
    );
}
