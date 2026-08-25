//! The SQLLogicTest corpus, run through a real connection.
//!
//! `sqllogictest.rs` drives the same files through `Session::execute` in-process. That path never
//! goes near the wire server, and the wire server never goes through `Session` — it calls
//! `execute_in_txn_as_*` directly. So everything `Session::execute` intercepts is structurally
//! invisible to that corpus: `PREPARE`/`EXECUTE`/`DEALLOCATE`, transaction control, savepoints,
//! `SET`/`SHOW`. Command tags are invisible too, because the embedded API returns a typed
//! `ExecutionResult` and never formats one.
//!
//! That is not a hypothesis. Two defects found in one week were wire-only, and one of them —
//! SQL-level `PREPARE` failing outright over a connection — was pinned as *working* in this very
//! corpus and reported as matching the reference engine, because both checks ran the embedded path.
//!
//! This file closes the hole by construction rather than by documenting it: the same `.slt` files,
//! a second `DB` implementation, a real server on a real socket. What it reports when a file
//! disagrees between the two paths is the thing nobody had — a list of what the corpus was hiding.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test: unwrap/panic-on-failure is the assertion mechanism"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::messages::{BackendMessage, FrontendMessage};
use nusadb_wire::server::{ServerConfig, serve_with_shutdown};
use nusadb_wire::{Connection, PROTOCOL_VERSION};
use sqllogictest::{DBOutput, DefaultColumnType, Runner};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;

/// A failure reported by the wire path — the SQLSTATE and message a client would see.
#[derive(Debug)]
struct SltWireError(String);

impl std::fmt::Display for SltWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SltWireError {}

/// A `.slt` connection that speaks the wire protocol to a server of its own.
///
/// One server and one connection per file, matching the in-process runner's "fresh engine per
/// file" rule, so a scenario cannot be perturbed by another.
struct WireConnection {
    /// Declared before `runtime` on purpose: a tokio IO resource must not outlive the runtime it
    /// was created on, and struct fields drop in declaration order.
    client: Connection<TcpStream>,
    runtime: Runtime,
}

impl WireConnection {
    fn new() -> Self {
        // `futures::executor::block_on` drives sqllogictest, so this is not a tokio context and a
        // current-thread runtime still drives the spawned server during `Runtime::block_on`.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let client = runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr");
            let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
            tokio::spawn(serve_with_shutdown(
                listener,
                engine,
                ServerConfig::default(),
                std::future::pending::<()>(),
            ));
            connect(addr).await
        });
        Self { client, runtime }
    }
}

async fn connect(addr: std::net::SocketAddr) -> Connection<TcpStream> {
    let mut client = Connection::new(TcpStream::connect(addr).await.expect("connect"));
    client
        .write_frame(
            &FrontendMessage::Startup {
                major: PROTOCOL_VERSION.0,
                minor: PROTOCOL_VERSION.1,
                user: "nusadb-root".to_owned(),
                database: "nusadb".to_owned(),
            }
            .encode()
            .expect("encode startup"),
        )
        .await
        .expect("send startup");
    loop {
        let frame = client
            .read_frame()
            .await
            .expect("read")
            .expect("server closed during startup");
        match BackendMessage::decode(&frame).expect("decode") {
            BackendMessage::ReadyForQuery(_) => return client,
            BackendMessage::Error { code, message } => {
                panic!("startup refused: {code}: {message}")
            },
            _ => {},
        }
    }
}

impl sqllogictest::DB for WireConnection {
    type Error = SltWireError;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        // Only a server `Error` frame may become `Self::Error`. Everything else — a dead socket, a
        // half-written frame, a decode failure — panics instead, because `statement error` in the
        // corpus is satisfied by *any* `Err`: laundering a transport failure into one would turn
        // "the connection broke" into a passing assertion, in the one harness whose entire purpose
        // is to prove the connection was used.
        let client = &mut self.client;
        self.runtime.block_on(async move {
            let frame = FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap_or_else(|e| panic!("encoding a query failed: {e}"));
            client
                .write_frame(&frame)
                .await
                .unwrap_or_else(|e| panic!("wire write failed: {e}"));

            // A wire-side hang is a live defect class for a harness built to catch wire-only
            // defects. The deadline has to wrap the read itself: a check before the await never
            // runs, because a pending read parks the runtime and control never returns to it.
            let deadline = std::time::Instant::now() + RESPONSE_DEADLINE;
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut columns = 0usize;
            let mut tags: Vec<u8> = Vec::new();
            let mut affected = 0u64;
            let mut error: Option<SltWireError> = None;
            loop {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                let frame = match tokio::time::timeout(remaining, client.read_frame()).await {
                    Err(tokio::time::error::Elapsed { .. }) => panic!(
                        "no frame within {RESPONSE_DEADLINE:?} — the wire hung rather than answered"
                    ),
                    Ok(Ok(Some(frame))) => frame,
                    Ok(Ok(None)) => {
                        panic!("wire closed mid-query — a transport failure, not a SQL error")
                    },
                    Ok(Err(e)) => panic!("wire read failed: {e}"),
                };
                let decoded = BackendMessage::decode(&frame)
                    .unwrap_or_else(|e| panic!("undecodable frame from the server: {e}"));
                match decoded {
                    BackendMessage::RowDescription { columns: c } => {
                        columns = c.len();
                        tags = vec![0; columns];
                    },
                    BackendMessage::RowDescriptionTyped { columns: c } => {
                        columns = c.len();
                        tags = c.iter().map(|(_, tag)| *tag).collect();
                    },
                    BackendMessage::DataRow { values } => rows.push(
                        values
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| {
                                // The corpus writes SQL NULL as `NULL`; the wire sends it absent.
                                v.map_or_else(
                                    || "NULL".to_owned(),
                                    |b| {
                                        canonicalize(
                                            &String::from_utf8_lossy(&b),
                                            tags.get(i).copied().unwrap_or(0),
                                        )
                                    },
                                )
                            })
                            .collect(),
                    ),
                    BackendMessage::Error { code, message } => {
                        error = Some(SltWireError(format!("{code}: {message}")));
                    },
                    BackendMessage::CommandComplete { tag } => {
                        // The corpus can assert an affected-row count, and the tag is the only
                        // place the wire carries one. Discarding it would report 0 for every
                        // write while the in-process path reports the real number — the module
                        // doc claims this harness covers command tags, so it has to read them.
                        affected = tag
                            .rsplit(' ')
                            .next()
                            .and_then(|n| n.parse::<u64>().ok())
                            .unwrap_or(0);
                    },
                    BackendMessage::ReadyForQuery(_) => break,
                    _ => {},
                }
            }
            if let Some(err) = error {
                return Err(err);
            }
            if rows.is_empty() && columns == 0 {
                return Ok(DBOutput::StatementComplete(affected));
            }
            Ok(DBOutput::Rows {
                types: vec![DefaultColumnType::Any; columns.max(rows.first().map_or(0, Vec::len))],
                rows,
            })
        })
    }

    fn engine_name(&self) -> &'static str {
        "nusadb-wire"
    }
}

/// Render a wire value the way the corpus writes it.
///
/// The corpus is written in sqllogictest's canonical text, which the in-process runner produces
/// directly from typed values. The wire sends the client-facing rendering instead — `true` where
/// the corpus writes `1`, `1.5` where it writes `1.500` — so the same row disagrees for reasons
/// that have nothing to do with the engine. Converting here keeps the *files* the single source of
/// truth for both paths, which is the point of running them twice.
///
/// The type tag comes from `RowDescriptionTyped`; an untyped description leaves the text alone.
///
/// The tags are derived from the taxonomy rather than transcribed: a hand-copied number that drifts
/// would let a TEXT column inherit the BOOL tag and silently rewrite a genuine `'true'` to `1` —
/// the corruption the `int_range.slt` note cites as the reason not to guess from text.
fn canonicalize(text: &str, type_tag: u8) -> String {
    match type_tag {
        BOOL_TAG => match text {
            "true" => "1".to_owned(),
            "false" => "0".to_owned(),
            other => other.to_owned(),
        },
        FLOAT_TAG => text
            .parse::<f64>()
            .map_or_else(|_| text.to_owned(), |f| format!("{f:.3}")),
        _ => text.to_owned(),
    }
}

/// How long one statement may take before the harness calls it a hang. Named so the panic can
/// report the actual budget: `tokio`'s `Elapsed` is a unit type whose `Display` is the fixed
/// string "deadline has elapsed", which names nothing.
const RESPONSE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Compile-time, from the taxonomy — see [`canonicalize`].
const BOOL_TAG: u8 = nusadb_wire::column_type_tag(nusadb_core::ColumnType::Bool);
/// Compile-time, from the taxonomy — see [`canonicalize`].
const FLOAT_TAG: u8 = nusadb_wire::column_type_tag(nusadb_core::ColumnType::Float);

/// Run one `.slt` file over a real connection.
fn run_over_wire(path: &str) {
    let mut runner = Runner::new(|| async { Ok::<_, SltWireError>(WireConnection::new()) });
    runner
        .run_file(path)
        .expect("sqllogictest scenario failed over the wire");
}

// === Corpus over the wire ===================================================
//
// Two beachhead files prove the plumbing on ground the in-process runner already covers, so a
// failure here is a harness fault rather than an engine one. Everything after them is chosen for
// the opposite reason: `p12_txn` is transaction control and savepoints, which `Session::execute`
// intercepts and the in-process corpus therefore reports on without ever having exercised the path
// a client takes. Those green results were unearned until now.

#[test]
fn wire_slt_p1_create_table_as() {
    run_over_wire("tests/slt/p1_ddl/create_table_as.slt");
}

#[test]
fn wire_slt_p1_drop_cascade() {
    run_over_wire("tests/slt/p1_ddl/drop_cascade.slt");
}

#[test]
fn wire_slt_p12_transactions() {
    run_over_wire("tests/slt/p12_txn/transactions.slt");
}

#[test]
fn wire_slt_p12_savepoints_nested() {
    run_over_wire("tests/slt/p12_txn/savepoints_nested.slt");
}

// Not yet runnable over the wire, each with a measured reason rather than a guess:
//
// * `p1_ddl/int_range.slt` — `SHOW COLUMNS` sends an untyped `RowDescription`, so its boolean
//   column arrives as `true`/`false` with no tag to canonicalise by, and the corpus writes `1`/`0`.
//   Guessing from the text would corrupt a genuine TEXT value of `'true'`, so the runner leaves it
//   alone. Either that statement should send a typed description, or the corpus should not assume
//   the in-process rendering.
// * `p13_functions/string_extra.slt` — the in-process runner renders an empty TEXT value as
//   `(empty)`, which is its own convention rather than anything `sqllogictest` normalises; the wire
//   sends an empty field. Same shape as the boolean case, and the same choice: no guessing.
// * Any file using `CREATE DATABASE` — this drives the wire loop directly
//   (`nusadb_wire::server::serve_with_shutdown`), not the composition root in `nusadb-server`, so
//   the cluster manager that statement needs is not present. That also bounds what this harness
//   covers: the wire protocol, not the server's own wiring.
// * Anything using `LISTEN`/`NOTIFY` — the notify registry is process-global and keyed on database
//   name, and every file here connects as `nusadb`, so concurrently-running files would cross.

/// Pin the wire's own rendering, so canonicalisation cannot quietly absorb a change to it.
///
/// `canonicalize` bridges the wire's text to the corpus's convention, which is what keeps the
/// `.slt` files the single source of truth for both paths. The cost of that bridge is a blind spot
/// exactly one line of wire code wide: if `value_to_field` started formatting floats as `{:.2}`,
/// the value would round-trip through `parse::<f64>()` to a different number that still re-renders
/// as `1.500` whenever the third decimal is zero, and every file would stay green.
///
/// This asserts the raw bytes instead, closing that residual.
#[test]
fn the_wire_renders_values_the_way_the_canonicaliser_assumes() {
    let mut conn = WireConnection::new();
    // A real FLOAT column, not a decimal literal: `1.5` written bare is NUMERIC, which neither
    // renderer reformats. Writing this test against the literal asserted nothing about the float
    // path — it is the first thing this test caught, and the reason it exists.
    sqllogictest::DB::run(&mut conn, "CREATE TABLE r (b BOOL, f FLOAT, s TEXT)").expect("create");
    sqllogictest::DB::run(&mut conn, "INSERT INTO r VALUES (true, 1.5, '')").expect("insert");
    let out = sqllogictest::DB::run(&mut conn, "SELECT b, NOT b, f, s FROM r").expect("select");
    let DBOutput::Rows { rows, .. } = out else {
        panic!("expected rows from a constant projection");
    };

    // Canonicalised, because that is what the corpus compares against.
    // The first three are the corpus's canonical form. The fourth is not: the corpus writes an
    // empty TEXT value as `(empty)`, which is the in-process runner's own convention and the
    // reason `string_extra.slt` is on the blocked list above. Asserting `""` here records the
    // divergence rather than implying the corpus expects it.
    assert_eq!(
        rows,
        vec![vec!["1", "0", "1.500", ""]],
        "canonical form for bool and float; empty TEXT is the known divergence"
    );

    // And the raw form the canonicaliser is written against. Asserted separately so a change to
    // either side is a failure here rather than a silent agreement somewhere else.
    let raw = raw_row(&mut conn, "SELECT b, NOT b, f, s FROM r");
    assert_eq!(
        raw,
        vec![
            Some("true".to_owned()),
            Some("false".to_owned()),
            Some("1.5".to_owned()),
            Some(String::new())
        ],
        "the wire's own rendering changed; `canonicalize` is written against it"
    );

    // Last, because it adds a row and the pin above reads the last one. The command tag carries
    // the only affected-row count the wire sends, and nothing in the corpus consumes one today —
    // so without this the parse would run unchecked, which is the shape this file exists to
    // refuse. Demonstrated by output, not claimed in a comment.
    let inserted =
        sqllogictest::DB::run(&mut conn, "INSERT INTO r VALUES (false, 2.5, 'x')").expect("insert");
    assert!(
        matches!(inserted, DBOutput::StatementComplete(1)),
        "the affected-row count must come from the command tag"
    );
}

/// The *last* row of a result, exactly as the wire sends it, with no canonicalisation.
/// Used for single-row pins, where last and only coincide.
fn raw_row(conn: &mut WireConnection, sql: &str) -> Vec<Option<String>> {
    let client = &mut conn.client;
    conn.runtime.block_on(async move {
        let frame = FrontendMessage::Query {
            sql: sql.to_owned(),
        }
        .encode()
        .expect("encode");
        client.write_frame(&frame).await.expect("write");
        let mut out = Vec::new();
        loop {
            let frame = client
                .read_frame()
                .await
                .expect("read")
                .expect("server closed mid-query");
            match BackendMessage::decode(&frame).expect("decode") {
                BackendMessage::DataRow { values } => {
                    out = values
                        .into_iter()
                        .map(|v| v.map(|b| String::from_utf8_lossy(&b).into_owned()))
                        .collect();
                },
                BackendMessage::ReadyForQuery(_) => break,
                _ => {},
            }
        }
        out
    })
}
