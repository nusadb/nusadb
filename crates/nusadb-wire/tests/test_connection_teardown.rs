//! A connection that ends without saying goodbye must still give back what its session held.
//!
//! Each test opens a block that holds a row lock, then ends the connection some way other than a
//! clean `Terminate` or end of stream: a frame the server cannot decode, a frame cut off by the
//! client closing, or a server shutdown that aborts a connection still busy with a statement. In
//! every case another session must be able to take the row afterwards.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/expect/panic"
)]

use std::sync::Arc;
use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, FrontendMessage, ServerConfig, handle_client, serve_with_shutdown,
};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};
use tokio::net::{TcpListener, TcpStream};

/// How long a test waits for a teardown that runs on another thread to take effect.
const SETTLE: Duration = Duration::from_secs(3);

async fn send<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
    conn.flush_now().await.unwrap();
}

/// Run `sql` and return its command tag, or the SQLSTATE it failed with.
async fn run<S: AsyncRead + AsyncWrite + Unpin>(
    conn: &mut Connection<S>,
    sql: &str,
) -> Result<String, String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut outcome = None;
    loop {
        let frame = conn
            .read_frame()
            .await
            .unwrap()
            .expect("server closed before ReadyForQuery");
        match BackendMessage::decode(&frame).unwrap() {
            BackendMessage::CommandComplete { tag } => outcome = Some(Ok(tag)),
            BackendMessage::Error { code, .. } => outcome = Some(Err(code)),
            BackendMessage::ReadyForQuery(_) => {
                return outcome.unwrap_or_else(|| panic!("no outcome for: {sql}"));
            },
            _ => {},
        }
    }
}

async fn start<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    send(
        conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: String::new(),
        },
    )
    .await;
    loop {
        let frame = conn.read_frame().await.unwrap().unwrap();
        if matches!(
            BackendMessage::decode(&frame).unwrap(),
            BackendMessage::ReadyForQuery(_)
        ) {
            return;
        }
    }
}

/// The client end of a fresh in-process connection, before its handshake.
fn open(engine: &Arc<dyn StorageEngine>) -> DuplexStream {
    let (client, server) = tokio::io::duplex(64 * 1024);
    drop(tokio::spawn(handle_client(server, Arc::clone(engine))));
    client
}

async fn connect(engine: &Arc<dyn StorageEngine>) -> Connection<DuplexStream> {
    let mut conn = Connection::new(open(engine));
    start(&mut conn).await;
    conn
}

/// A table with one row, and a session that holds that row's lock in an open block.
async fn locked_row<S: AsyncRead + AsyncWrite + Unpin>(
    engine: &Arc<dyn StorageEngine>,
    a: &mut Connection<S>,
) -> Connection<DuplexStream> {
    let mut b = connect(engine).await;
    run(&mut b, "CREATE TABLE t (id INT PRIMARY KEY, n INT)")
        .await
        .unwrap();
    run(&mut b, "INSERT INTO t VALUES (1, 0)").await.unwrap();
    run(a, "BEGIN").await.unwrap();
    run(a, "UPDATE t SET n = 5 WHERE id = 1").await.unwrap();
    assert_eq!(
        run(&mut b, "UPDATE t SET n = 6 WHERE id = 1").await,
        Err("40001".to_owned()),
        "the open block holds the row"
    );
    b
}

/// Retry the update of the locked row until it goes through or [`SETTLE`] passes.
async fn row_is_released<S: AsyncRead + AsyncWrite + Unpin>(b: &mut Connection<S>) -> bool {
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        if run(b, "UPDATE t SET n = 6 WHERE id = 1").await.is_ok() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A frame whose declared length is below the frame header ends the connection with a protocol
/// error. The block it had open must be rolled back.
#[tokio::test]
async fn a_malformed_frame_does_not_leak_the_open_block() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut raw = open(&engine);
    let mut a = Connection::new(&mut raw);
    start(&mut a).await;
    let mut b = locked_row(&engine, &mut a).await;
    drop(a);

    raw.write_all(&[b'Q', 0, 0, 0, 2]).await.unwrap();
    raw.flush().await.unwrap();

    assert!(
        row_is_released(&mut b).await,
        "the row lock of a session that sent a malformed frame is still held"
    );
}

/// A client that closes in the middle of a frame ends the connection with an unexpected end of
/// stream. The block it had open must be rolled back.
#[tokio::test]
async fn a_frame_cut_off_by_the_client_does_not_leak_the_open_block() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut raw = open(&engine);
    let mut a = Connection::new(&mut raw);
    start(&mut a).await;
    let mut b = locked_row(&engine, &mut a).await;
    drop(a);

    raw.write_all(&[b'Q', 0, 0, 0, 64, 0]).await.unwrap();
    raw.flush().await.unwrap();
    drop(raw);

    assert!(
        row_is_released(&mut b).await,
        "the row lock of a session cut off mid-frame is still held"
    );
}

/// A shutdown whose drain window runs out aborts the connections still busy. One stalled on a
/// client that stopped reading a large result, inside a block, must still have that block rolled
/// back, and only once the statement still running for it has stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connection_aborted_by_shutdown_does_not_leak_the_open_block() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        Arc::clone(&engine),
        ServerConfig {
            drain_timeout: Some(Duration::from_millis(200)),
            ..ServerConfig::default()
        },
        async move {
            let _ = stopped.await;
        },
    ));

    let mut a = Connection::new(TcpStream::connect(addr).await.unwrap());
    start(&mut a).await;
    let mut b = locked_row(&engine, &mut a).await;

    // A result far larger than the socket buffers, which this client never reads: the connection
    // is left blocked writing it, mid-statement, when the shutdown comes.
    send(
        &mut a,
        FrontendMessage::Query {
            sql: "SELECT repeat('x', 1000) FROM generate_series(1, 200000)".to_owned(),
        },
    )
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stopping = tokio::time::Instant::now();
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("the drain must give up on the stalled connection")
        .unwrap()
        .unwrap();
    assert!(
        stopping.elapsed() >= Duration::from_millis(200),
        "the connection drained by itself, so it was never aborted"
    );

    assert!(
        row_is_released(&mut b).await,
        "the row lock of a connection aborted by shutdown is still held"
    );
    drop(a);
}

/// Parse, bind and execute `sql` in the unnamed portal, without waiting for a reply.
async fn execute<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>, sql: &str) {
    send(
        conn,
        FrontendMessage::Parse {
            name: String::new(),
            sql: sql.to_owned(),
            param_types: Vec::new(),
        },
    )
    .await;
    send(
        conn,
        FrontendMessage::Bind {
            portal: String::new(),
            statement: String::new(),
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
}

/// A statement that opens a block, and whose reply is the write that finds the client gone. The
/// block it opened must be rolled back, which only happens if the teardown sees the state the
/// statement left rather than the one it found.
#[tokio::test]
async fn a_block_opened_by_a_statement_whose_reply_fails_is_rolled_back() {
    // A durable engine: only it checkpoints.
    let dir = nusadb_test_utils::temp_dir();
    let engine: Arc<dyn StorageEngine> =
        Arc::new(BtreeEngine::open(dir.path().join("db")).unwrap());
    let mut b = connect(&engine).await;
    assert_eq!(run(&mut b, "CHECKPOINT").await, Ok("CHECKPOINT".to_owned()));

    let mut raw = open(&engine);
    let mut a = Connection::new(&mut raw);
    start(&mut a).await;
    // The first reply leaves the server's output just below its 64 KiB flush threshold, so the
    // reply to BEGIN is the write that sends it, after the client has gone.
    execute(&mut a, "SELECT repeat('x', 65480)").await;
    execute(&mut a, "BEGIN").await;
    send(&mut a, FrontendMessage::Sync).await;
    drop(a);
    drop(raw);

    // A checkpoint refuses while any transaction is open, so it shows whether one leaked.
    let deadline = tokio::time::Instant::now() + SETTLE;
    let mut last = run(&mut b, "CHECKPOINT").await;
    while last.is_err() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
        last = run(&mut b, "CHECKPOINT").await;
    }
    assert_eq!(
        last,
        Ok("CHECKPOINT".to_owned()),
        "the block the statement opened outlived the session"
    );
}
