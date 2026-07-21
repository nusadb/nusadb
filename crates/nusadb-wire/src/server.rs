//! TCP accept loop + per-connection state machine.
//!
//! [`serve`] accepts connections and drives each through [`handle_client`]:
//! `Startup → (auth) → ReadyForQuery → {Query → results → ReadyForQuery}* → Terminate`.
//!
//! The wire layer is async (tokio); the SQL engine is synchronous, so each query is run on a
//! blocking pool thread via [`tokio::task::spawn_blocking`], keeping the reactor free for I/O.
//!
//! Auth is currently trust-on-startup; the SCRAM-SHA-256 handshake ([`auth`](crate::auth)) and
//! TLS upgrade ([`tls`](crate::tls)) slot in before `AuthOk` in a later batch.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use bytes::BytesMut;
use nusadb_core::{IsolationLevel, StorageEngine, TableSchema, TxnId};
use nusadb_sql::ast::Value;
use nusadb_sql::{
    Catalog, ExecutionResult, IndexInfo, RowSink, StreamOutcome, analyze, bind_parameters,
    copy_from, copy_to, describe_column_types, describe_columns,
    execute_in_txn_as_streaming_with_settings, execute_in_txn_as_with_settings, parameter_count,
    parse, plan, show_session_variable,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, mpsc, watch};
use tokio::task::JoinSet;

use crate::PROTOCOL_VERSION;
use crate::auth::{AuthStore, scram};
use crate::cancel;
use crate::cluster::DatabaseCluster;
use crate::frame::Frame;
use crate::messages::{BackendMessage, DescribeTarget, FrontendMessage, TxnStatus};
use crate::metrics::Metrics;
use crate::notify;

/// Queued output beyond this many bytes is transmitted immediately rather than held for the next
/// coalescing point, so a large result set (or `COPY TO STDOUT`) streams in bounded chunks instead
/// of accumulating in memory.
const OUT_FLUSH_THRESHOLD: usize = 64 * 1024;

/// A framed connection over an async byte stream: incremental frame reads + coalesced frame writes.
///
/// Writes are buffered: [`write_frame`](Self::write_frame) queues a frame and the queue is
/// transmitted as **one** write+flush at the next [`flush_now`](Self::flush_now) or — automatically
/// — the moment [`read_frame`](Self::read_frame) would wait for input. One logical response
/// (e.g. `RowDescription → DataRow… → CommandComplete → ReadyForQuery`) therefore leaves as a
/// single TCP segment instead of one segment per frame, which avoids the client's delayed-ACK
/// ping-pong that put a ~40ms floor under every query. The read-side auto-flush makes the scheme
/// deadlock-free by construction: this end never waits for the peer while holding bytes the peer
/// is waiting for.
#[derive(Debug)]
pub struct Connection<S> {
    stream: S,
    buf: BytesMut,
    out: BytesMut,
}

impl<S: AsyncRead + AsyncWrite + Unpin> Connection<S> {
    /// Wrap a stream.
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            buf: BytesMut::with_capacity(8 * 1024),
            out: BytesMut::with_capacity(8 * 1024),
        }
    }

    /// Read the next frame, or `Ok(None)` at a clean end-of-stream.
    ///
    /// Any queued output is flushed before waiting on the socket, so the peer — which may be
    /// blocked on exactly those bytes — can always make progress.
    ///
    /// # Errors
    /// I/O errors, or a malformed/oversized frame (surfaced as `io::Error`).
    pub async fn read_frame(&mut self) -> io::Result<Option<Frame>> {
        loop {
            if let Some(frame) = Frame::decode(&mut self.buf).map_err(io::Error::other)? {
                return Ok(Some(frame));
            }
            self.flush_now().await?;
            if self.stream.read_buf(&mut self.buf).await? == 0 {
                return if self.buf.is_empty() {
                    Ok(None)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "partial frame at EOF",
                    ))
                };
            }
        }
    }

    /// Queue one frame for output. It is transmitted at the next [`flush_now`](Self::flush_now)
    /// (or automatically when [`read_frame`](Self::read_frame) is about to wait), except that a
    /// queue past the 64 KiB flush threshold is flushed here so a large streamed result never
    /// accumulates without bound.
    ///
    /// # Errors
    /// Propagates write/flush errors from a threshold flush.
    pub async fn write_frame(&mut self, frame: &Frame) -> io::Result<()> {
        frame.encode(&mut self.out);
        if self.out.len() >= OUT_FLUSH_THRESHOLD {
            self.flush_now().await?;
        }
        Ok(())
    }

    /// Transmit everything queued by [`write_frame`](Self::write_frame) as one write, then flush.
    /// No-op when nothing is queued. Cancellation-safe: a partial write consumes exactly the
    /// written bytes from the queue, so a resumed call picks up where it left off.
    ///
    /// # Errors
    /// Propagates write/flush errors.
    pub async fn flush_now(&mut self) -> io::Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        self.stream.write_all_buf(&mut self.out).await?;
        self.stream.flush().await
    }
}

/// Tunables for [`serve_with_shutdown`].
///
/// [`Default`] is **safe-by-default**: [`handshake_timeout`](Self::handshake_timeout) and
/// [`copy_from_max_bytes`](Self::copy_from_max_bytes) carry non-`None` guards so the shipped
/// configuration is not trivially DoS-able even when every other field is left unset.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Close a connection that has been idle (awaiting a request) for longer than this. `None`
    /// disables the idle timeout (a connection may wait indefinitely).
    pub idle_timeout: Option<Duration>,
    /// On shutdown, how long to wait for in-flight connections to drain before forcibly aborting
    /// the stragglers. `None` waits indefinitely.
    pub drain_timeout: Option<Duration>,
    /// Maximum number of connections served concurrently. Excess connections wait (queued in the
    /// kernel backlog) until a slot frees. `None` is unbounded.
    pub max_connections: Option<usize>,
    /// What happens to a connection past [`max_connections`](Self::max_connections) (P-CONNCAP):
    /// `false` (default) queues it until a slot frees — graceful, but a connection storm shows up
    /// as client-side hangs; `true` refuses it immediately with SQLSTATE `53300`
    /// (`too_many_connections`, the reference engine's behaviour), so a pool sees an honest error instead of a
    /// stall. No effect when `max_connections` is `None`.
    pub reject_excess_connections: bool,
    /// Shared counters to update as connections open/close and queries run. `None` disables
    /// metrics.
    pub metrics: Option<Arc<Metrics>>,
    /// When set, every accepted connection is wrapped in a rustls server session before any frame
    /// is read (implicit TLS). `None` serves plaintext. Build one with
    /// [`tls::server_config`](crate::tls::server_config).
    pub tls: Option<Arc<rustls::ServerConfig>>,
    /// When set, every connection must complete a SCRAM-SHA-256 handshake against this credential
    /// store before any query. `None` is trust-on-startup (the connection's declared user
    /// is accepted without a password — suitable for local/dev or a trusted network).
    pub auth: Option<Arc<AuthStore>>,
    /// Cancel a statement that runs longer than this. Enforced cooperatively — the executor
    /// aborts at the next scan/loop boundary — so the wall-clock cap is approximate. `None` = no
    /// limit.
    pub statement_timeout: Option<Duration>,
    /// Bound the pre-query handshake — reading the `Startup` frame plus completing authentication.
    /// Unlike [`idle_timeout`](Self::idle_timeout), this **always** caps how long a connection may
    /// occupy a slot before it reaches the query loop, so an unauthenticated client that connects
    /// and then stalls (slowloris) cannot hold a slot indefinitely. `None` disables it (not
    /// recommended; reintroduces the pre-Startup DoS).
    pub handshake_timeout: Option<Duration>,
    /// Maximum cumulative bytes buffered for a single `COPY ... FROM STDIN`. A load that streams
    /// more than this is aborted with an error instead of being buffered without bound (which an
    /// authenticated client could otherwise exploit to OOM the server). `None` is unbounded (not
    /// recommended).
    pub copy_from_max_bytes: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            idle_timeout: None,
            drain_timeout: None,
            max_connections: None,
            reject_excess_connections: false,
            metrics: None,
            tls: None,
            auth: None,
            statement_timeout: None,
            // Safe-by-default DoS guards: these apply even when the operator leaves
            // everything else unset, so the shipped default is not trivially DoS-able. A real
            // handshake (Startup + TLS + SCRAM) completes in well under a minute; a legitimate bulk
            // COPY rarely exceeds a gibibyte in one statement (split larger loads).
            handshake_timeout: Some(Duration::from_mins(1)),
            copy_from_max_bytes: Some(1 << 30), // 1 GiB
        }
    }
}

/// Accept connections forever, serving each against `engine` (no idle timeout, never shuts down).
///
/// # Errors
/// Propagates listener accept errors. Per-connection errors are logged, not propagated.
pub async fn serve(listener: TcpListener, engine: Arc<dyn StorageEngine>) -> io::Result<()> {
    serve_with_shutdown(
        listener,
        engine,
        ServerConfig::default(),
        std::future::pending::<()>(),
    )
    .await
}

/// Accept and serve connections until `shutdown` resolves, then drain and return.
///
/// On shutdown the server **stops accepting**, signals every in-flight connection to finish its
/// current exchange, and **drains** — waiting for them to complete (bounded by
/// [`ServerConfig::drain_timeout`]) before returning. Each connection also honours
/// [`ServerConfig::idle_timeout`].
///
/// # Errors
/// Propagates listener accept errors. Per-connection errors are logged, not propagated.
pub async fn serve_with_shutdown<F>(
    listener: TcpListener,
    engine: Arc<dyn StorageEngine>,
    config: ServerConfig,
    shutdown: F,
) -> io::Result<()>
where
    F: Future<Output = ()> + Send,
{
    // A bare engine is single-database mode: serve it as a one-database cluster.
    serve_cluster_with_shutdown(
        listener,
        Arc::new(crate::cluster::SingleDatabase::new(engine)),
        config,
        shutdown,
    )
    .await
}

/// Like [`serve_with_shutdown`], but over a [`DatabaseCluster`].
///
/// Each connection resolves its startup database to one engine, so the server hosts
/// physically-isolated multiple databases. The single-engine [`serve_with_shutdown`] is this with a
/// one-database cluster.
///
/// # Errors
/// Propagates listener accept errors. Per-connection errors are logged, not propagated.
#[allow(
    clippy::too_many_lines,
    reason = "one linear accept loop: slot policy (queue vs fast-reject, P-CONNCAP), accept, and \
              the per-connection spawn; splitting it would scatter the loop's shutdown/permit \
              interplay across helpers"
)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "the queue-mode permit is deliberately held across the accept and moved into the \
              connection task (it IS the connection slot); the lint's suggested early drop would \
              release the slot before the connection ends"
)]
pub async fn serve_cluster_with_shutdown<F>(
    listener: TcpListener,
    cluster: Arc<dyn DatabaseCluster>,
    config: ServerConfig,
    shutdown: F,
) -> io::Result<()>
where
    F: Future<Output = ()> + Send,
{
    // `false` while running; flipped to `true` to tell every connection to wind down.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let mut tasks: JoinSet<()> = JoinSet::new();
    // Connection limiter: a permit per in-flight connection, held for its lifetime.
    let limiter = config.max_connections.map(|n| Arc::new(Semaphore::new(n)));
    if let Some(n) = config.max_connections {
        // Expose the effective limit + policy (P-CONNCAP) so an operator can see what a
        // connection storm will do without reading the source.
        tracing::info!(
            max_connections = n,
            policy = if config.reject_excess_connections {
                "reject (53300)"
            } else {
                "queue"
            },
            "connection limit active"
        );
    }
    tokio::pin!(shutdown);

    loop {
        // In queue mode, reserve a connection slot first: when the limit is reached this
        // waits here, so a surplus connection simply stays in the kernel accept backlog until a
        // slot frees. In reject mode the slot is probed *after* accept, so the surplus connection
        // can be refused with an honest error instead of queueing (P-CONNCAP).
        let queued_permit = match &limiter {
            Some(sem) if !config.reject_excess_connections => {
                let acquired = tokio::select! {
                    biased;
                    () = &mut shutdown => break,
                    p = Arc::clone(sem).acquire_owned() => p,
                };
                match acquired {
                    Ok(permit) => Some(permit),
                    Err(_closed) => break, // semaphore is never closed; treat as shutdown
                }
            },
            _ => None,
        };

        let (socket, _peer) = tokio::select! {
            // Prefer shutdown over accepting a fresh connection when both are ready.
            biased;
            () = &mut shutdown => break,
            accept = listener.accept() => accept?,
        };

        let permit = match (&limiter, queued_permit) {
            (Some(sem), None) if config.reject_excess_connections => {
                match Arc::clone(sem).try_acquire_owned() {
                    Ok(permit) => Some(permit),
                    Err(_no_slot) => {
                        // Fast-reject (P-CONNCAP): tell the client `53300 too many clients` and
                        // close, exactly what the reference engine does — a pool retries/backs off instead of
                        // hanging in the backlog. Runs in its own task (the TLS handshake, when
                        // configured, must complete before the error is readable) so a slow
                        // client cannot stall the accept loop; it holds no connection slot.
                        tokio::spawn(reject_connection(
                            socket,
                            config.tls.clone(),
                            config.handshake_timeout,
                        ));
                        continue;
                    },
                }
            },
            (_, queued) => queued,
        };

        // Disable Nagle's algorithm so a small request/response is not delayed waiting to coalesce
        // with more data — the protocol is request/response, so Nagle (interacting with delayed-ACK)
        // adds tens of milliseconds per round trip to a point query (A-NET.1). A failure to set the
        // option is non-fatal: the connection still works, just with Nagle left on.
        if let Err(e) = socket.set_nodelay(true) {
            tracing::debug!("set_nodelay failed on accepted connection: {e}");
        }

        let cluster = Arc::clone(&cluster);
        let rx = shutdown_rx.clone();
        let idle = config.idle_timeout;
        let metrics = config.metrics.clone();
        let tls = config.tls.clone();
        let auth = config.auth.clone();
        let stmt_timeout = config.statement_timeout;
        let handshake = config.handshake_timeout;
        let copy_max = config.copy_from_max_bytes;
        tasks.spawn(async move {
            let _permit = permit; // released (slot freed) when the connection task ends
            // RAII gauge: increment now, decrement on drop — so the active-connection count stays
            // balanced even if this task panics mid-connection, where a manual
            // `connection_closed()` after the await would be skipped.
            let _conn_guard = ConnectionGuard::new(metrics.clone());
            // Run the (optional) TLS handshake inside the per-connection task so a slow handshake
            // never stalls the accept loop. A handshake failure drops the connection.
            let result = match tls {
                Some(cfg) => {
                    serve_tls(
                        socket,
                        cfg,
                        cluster,
                        idle,
                        rx,
                        metrics.clone(),
                        auth,
                        stmt_timeout,
                        handshake,
                        copy_max,
                    )
                    .await
                },
                None => {
                    serve_connection(
                        socket,
                        cluster,
                        idle,
                        Some(rx),
                        metrics.clone(),
                        auth,
                        stmt_timeout,
                        handshake,
                        copy_max,
                    )
                    .await
                },
            };
            if let Err(e) = result {
                tracing::warn!("connection error: {e}");
            }
        });
        // Reap finished connection tasks so the set doesn't grow without bound.
        while tasks.try_join_next().is_some() {}
    }

    tracing::info!(
        in_flight = tasks.len(),
        "shutdown requested — draining connections"
    );
    let _ = shutdown_tx.send(true); // wake idle connections so they close promptly
    drain(tasks, config.drain_timeout).await;
    Ok(())
}

/// Wait for every connection task to finish, bounded by `timeout`; abort the stragglers if it
/// elapses.
async fn drain(mut tasks: JoinSet<()>, timeout: Option<Duration>) {
    match timeout {
        Some(d) => {
            let drained =
                tokio::time::timeout(d, async { while tasks.join_next().await.is_some() {} }).await;
            if drained.is_err() {
                tracing::warn!(
                    remaining = tasks.len(),
                    "drain timeout — aborting connections"
                );
                tasks.shutdown().await;
            }
        },
        None => while tasks.join_next().await.is_some() {},
    }
}

/// Drive one client connection through its full lifecycle (no idle timeout, no shutdown signal).
///
/// The safe-by-default DoS guards from [`ServerConfig::default`] still apply: the handshake is
/// bounded by a timeout and a `COPY ... FROM STDIN` is capped at a cumulative byte limit.
///
/// # Errors
/// I/O or protocol errors that terminate the connection. SQL errors are reported to the client
/// as an `Error` message, not returned here.
pub async fn handle_client<S>(stream: S, engine: Arc<dyn StorageEngine>) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    // Inherit the safe-by-default DoS guards (handshake timeout + COPY cap) from `ServerConfig`.
    let defaults = ServerConfig::default();
    handle_client_with(
        stream,
        engine,
        None,
        None,
        None,
        None,
        None,
        defaults.handshake_timeout,
        defaults.copy_from_max_bytes,
    )
    .await
}

/// Refuse a connection that arrived past `max_connections` in reject mode (fast-reject,
/// P-CONNCAP): complete the (optional) TLS handshake so the error is readable by the client's
/// protocol stack, send SQLSTATE `53300` (`too_many_connections`, the reference engine's message), and close.
/// Best-effort and bounded by the handshake timeout: any failure just drops the socket, which is
/// where this connection was headed anyway.
async fn reject_connection(
    socket: tokio::net::TcpStream,
    tls: Option<Arc<rustls::ServerConfig>>,
    handshake_timeout: Option<Duration>,
) {
    async fn refuse<S: AsyncRead + AsyncWrite + Unpin>(stream: S) {
        let mut conn = Connection::new(stream);
        let Ok(frame) = error_response_coded("sorry, too many clients already", "53300").encode()
        else {
            return;
        };
        let _ = conn.write_frame(&frame).await;
        let _ = conn.flush_now().await;
    }
    let work = async {
        match tls {
            Some(cfg) => {
                if let Ok(stream) = tokio_rustls::TlsAcceptor::from(cfg).accept(socket).await {
                    refuse(stream).await;
                }
            },
            None => refuse(socket).await,
        }
    };
    // Even a refusal must not be stallable indefinitely by a slow client.
    let _ = tokio::time::timeout(
        handshake_timeout.unwrap_or_else(|| Duration::from_secs(30)),
        work,
    )
    .await;
}

/// Complete the TLS handshake on `socket`, then serve the encrypted stream. The
/// per-connection handler is generic, so it runs unchanged over the rustls stream.
///
/// # Errors
/// A handshake failure (untrusted client, protocol mismatch) returns the I/O error, which the
/// caller logs before dropping the connection.
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection settings forwarded verbatim to handle_client_with after the TLS accept"
)]
async fn serve_tls(
    socket: tokio::net::TcpStream,
    tls: Arc<rustls::ServerConfig>,
    cluster: Arc<dyn DatabaseCluster>,
    idle: Option<Duration>,
    shutdown: watch::Receiver<bool>,
    metrics: Option<Arc<Metrics>>,
    auth: Option<Arc<AuthStore>>,
    statement_timeout: Option<Duration>,
    handshake_timeout: Option<Duration>,
    copy_from_max_bytes: Option<usize>,
) -> io::Result<()> {
    // The TLS handshake is also pre-Startup, so bound it by the same timeout: a client that
    // opens a socket and then stalls the rustls handshake must not hold a connection slot forever.
    // This window is separate from the Startup+auth one below, so the TLS path's total pre-query
    // budget is up to twice `handshake_timeout` — still bounded, which is what matters here.
    let accept = tokio_rustls::TlsAcceptor::from(tls).accept(socket);
    let stream = match handshake_timeout {
        Some(d) => match tokio::time::timeout(d, accept).await {
            Ok(stream) => stream?,
            Err(_elapsed) => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "TLS handshake did not complete within the handshake timeout",
                ));
            },
        },
        None => accept.await?,
    };
    serve_connection(
        stream,
        cluster,
        idle,
        Some(shutdown),
        metrics,
        auth,
        statement_timeout,
        handshake_timeout,
        copy_from_max_bytes,
    )
    .await
}

/// Read the next frame, returning `Ok(None)` to close the connection on a clean EOF, an idle
/// timeout, or a shutdown-while-idle. A request already in progress is never
/// interrupted — only the wait for the *next* request is cancellable, so in-flight work drains.
async fn read_next<S>(
    conn: &mut Connection<S>,
    idle_timeout: Option<Duration>,
    shutdown: &mut Option<watch::Receiver<bool>>,
) -> io::Result<Option<Frame>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // Shutdown already in progress — close before blocking on a read.
    if let Some(rx) = shutdown.as_ref()
        && *rx.borrow()
    {
        return Ok(None);
    }
    match shutdown.as_mut() {
        Some(rx) => {
            tokio::select! {
                biased;
                _ = rx.changed() => Ok(None), // shutdown signalled while idle → graceful close
                frame = read_with_idle(conn, idle_timeout) => frame,
            }
        },
        None => read_with_idle(conn, idle_timeout).await,
    }
}

/// `read_frame` bounded by an optional idle timeout; an elapsed timeout closes the connection.
async fn read_with_idle<S>(
    conn: &mut Connection<S>,
    idle_timeout: Option<Duration>,
) -> io::Result<Option<Frame>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match idle_timeout {
        Some(d) => match tokio::time::timeout(d, conn.read_frame()).await {
            Ok(frame) => frame,
            Err(_elapsed) => {
                tracing::debug!("closing connection after idle timeout");
                Ok(None)
            },
        },
        None => conn.read_frame().await,
    }
}

/// The run-time parameters reported to a client during the startup handshake (via
/// [`BackendMessage::ParameterStatus`]). `server_version` is the engine's own version (the wire
/// protocol is the Nusa protocol — the honest engine version is the right answer); the rest mirror
/// the SQL `SHOW` / `current_setting` built-in defaults so a client sees one consistent story.
const fn startup_parameter_status() -> [(&'static str, &'static str); 6] {
    [
        ("server_version", env!("CARGO_PKG_VERSION")),
        ("server_encoding", "UTF8"),
        ("client_encoding", "UTF8"),
        ("standard_conforming_strings", "on"),
        ("DateStyle", "ISO, MDY"),
        ("TimeZone", "UTC"),
    ]
}

/// Drive one client connection, honouring an optional idle timeout and shutdown signal.
///
/// # Errors
/// I/O or protocol errors that terminate the connection. SQL errors are reported to the client
/// as an `Error` message, not returned here.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-message protocol dispatch (simple + extended query); length scales with \
              the message taxonomy, not branching depth"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection settings forwarded verbatim from ServerConfig; bundling them into a \
              struct would only move the field list, not shrink it"
)]
pub async fn handle_client_with<S>(
    stream: S,
    engine: Arc<dyn StorageEngine>,
    idle_timeout: Option<Duration>,
    shutdown: Option<watch::Receiver<bool>>,
    metrics: Option<Arc<Metrics>>,
    auth: Option<Arc<AuthStore>>,
    statement_timeout: Option<Duration>,
    handshake_timeout: Option<Duration>,
    copy_from_max_bytes: Option<usize>,
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    // A bare engine is single-database mode: wrap it so every requested database resolves to it.
    serve_connection(
        stream,
        Arc::new(crate::cluster::SingleDatabase::new(engine)),
        idle_timeout,
        shutdown,
        metrics,
        auth,
        statement_timeout,
        handshake_timeout,
        copy_from_max_bytes,
    )
    .await
}

/// Drive one client connection against a [`DatabaseCluster`]: after the startup handshake, the
/// connection's requested database name is resolved to one engine (the connection then only ever
/// touches that engine — physical isolation, DB2). A request for a database that does not exist is
/// refused with a fatal `3D000` before the query loop.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-message protocol dispatch (simple + extended query); length scales with \
              the message taxonomy, not branching depth"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection settings forwarded verbatim from ServerConfig; bundling them into a \
              struct would only move the field list, not shrink it"
)]
async fn serve_connection<S>(
    stream: S,
    cluster: Arc<dyn DatabaseCluster>,
    idle_timeout: Option<Duration>,
    mut shutdown: Option<watch::Receiver<bool>>,
    metrics: Option<Arc<Metrics>>,
    auth: Option<Arc<AuthStore>>,
    statement_timeout: Option<Duration>,
    handshake_timeout: Option<Duration>,
    copy_from_max_bytes: Option<usize>,
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut conn = Connection::new(stream);

    // --- Startup + authentication (bounded by `handshake_timeout`) ---
    //
    // This whole phase has a deadline that applies independently of `idle_timeout`, so a client
    // that connects and then stalls — never finishing Startup or the SCRAM handshake — cannot hold
    // a connection slot indefinitely (an unauthenticated slowloris). The future yields `Ok(None)`
    // to close the connection cleanly, or `Ok(Some(user))` to proceed to the query loop.
    let handshake = async {
        let Some(frame) = read_next(&mut conn, idle_timeout, &mut shutdown).await? else {
            return io::Result::Ok(None); // disconnected before sending anything
        };
        let (user, client_minor, database) =
            match FrontendMessage::decode(&frame).map_err(io::Error::other)? {
                FrontendMessage::Startup {
                    major,
                    minor,
                    user,
                    database,
                } => {
                    if major != PROTOCOL_VERSION.0 {
                        conn.write_frame(
                            &error_response("unsupported protocol major version").encode()?,
                        )
                        .await?;
                        conn.flush_now().await?;
                        return Ok(None);
                    }
                    (user, minor, database)
                },
                // An out-of-band cancel request: trip the target connection's statement and
                // close — no response, the requester just disconnects (matching the cancel protocol).
                FrontendMessage::CancelRequest { pid, secret } => {
                    cancel::cancel(pid, secret);
                    return Ok(None);
                },
                _ => {
                    conn.write_frame(&error_response("expected Startup message").encode()?)
                        .await?;
                    conn.flush_now().await?;
                    return Ok(None);
                },
            };
        // Authenticate: SCRAM-SHA-256 when an `AuthStore` is configured, else trust-on-startup. A
        // failed/abandoned handshake drops the connection without reaching the query loop.
        match &auth {
            None => conn.write_frame(&BackendMessage::AuthOk.encode()?).await?,
            Some(store) => {
                if !authenticate(&mut conn, &user, store, idle_timeout, &mut shutdown).await? {
                    return Ok(None);
                }
            },
        }
        Ok(Some((user, client_minor, database)))
    };
    let negotiated = match handshake_timeout {
        Some(d) => match tokio::time::timeout(d, handshake).await {
            Ok(result) => result?,
            Err(_elapsed) => {
                tracing::debug!("closing connection: handshake not completed within timeout");
                return Ok(());
            },
        },
        None => handshake.await?,
    };
    let Some((user, client_minor, database)) = negotiated else {
        return Ok(());
    };
    // Resolve the connection's database: an empty startup database lands in the cluster default.
    // The connection binds to exactly this engine for its lifetime — it never touches another, so
    // databases are physically isolated. A request for a database that does not exist is fatal.
    let database = if database.is_empty() {
        cluster.default_database()
    } else {
        database
    };
    let engine = match cluster.open(&database) {
        Ok(Some(engine)) => engine,
        Ok(None) => {
            conn.write_frame(
                &error_response_coded(&format!("database \"{database}\" does not exist"), "3D000")
                    .encode()?,
            )
            .await?;
            conn.flush_now().await?;
            return Ok(());
        },
        Err(e) => {
            conn.write_frame(&error_response_coded(&e.to_string(), e.sqlstate()).encode()?)
                .await?;
            conn.flush_now().await?;
            return Ok(());
        },
    };
    // Effective protocol minor for this connection: the lower of what the client asked for and what
    // this server supports. `typed >= 1` opts the client into the typed `RowDescriptionTyped`
    // metadata; `minor = 0` keeps the byte-identical 1.0 surface.
    let typed_row_description = client_minor.min(PROTOCOL_VERSION.1) >= 1;
    // `minor >= 2` additionally carries an ARRAY column's element type in its tag (protocol 1.2), so a
    // client decodes the elements at their real type rather than as text.
    let array_element_types = client_minor.min(PROTOCOL_VERSION.1) >= 2;

    // Register this connection's cancel token and hand the client its key. The token is
    // checked by the executor; the registration deregisters on drop at connection end.
    let cancel_token: nusadb_sql::cancel::CancelToken = Arc::new(AtomicBool::new(false));
    let (key, _registration) = cancel::register(Arc::clone(&cancel_token));
    conn.write_frame(
        &BackendMessage::BackendKeyData {
            pid: key.pid,
            secret: key.secret,
        }
        .encode()?,
    )
    .await?;

    // Report the run-time parameters a client reads during startup (server_version, the encodings,
    // standard_conforming_strings, …) so drivers configure themselves without a round-trip query. The
    // values mirror the SQL `SHOW`/`current_setting` built-in defaults (honest engine version).
    for (name, value) in startup_parameter_status() {
        conn.write_frame(
            &BackendMessage::ParameterStatus {
                name: name.to_owned(),
                value: value.to_owned(),
            }
            .encode()?,
        )
        .await?;
    }

    conn.write_frame(&BackendMessage::ReadyForQuery(TxnStatus::Idle).encode()?)
        .await?;

    // --- Query loop (simple query + extended query) ---
    //
    // Extended query: `Parse` stores a prepared statement, `Bind` makes a portal,
    // `Describe`/`Execute` lazily run the portal once (caching its result) and stream it, and
    // `Sync` ends the pipeline with `ReadyForQuery`. After an error the server skips messages until
    // the next `Sync` (skip-until-Sync error semantics), tracked by `failed`.
    let mut statements: HashMap<String, String> = HashMap::new();
    let mut portals: HashMap<String, Portal> = HashMap::new();
    let mut failed = false;
    // Explicit-transaction state across statements on this connection (transaction-over-wire). The
    // simple- and extended-query paths both thread it, so `BEGIN ... COMMIT` spans either.
    let mut txn_state = TxnState::Auto;
    // Per-connection session GUC store (session-state-over-wire): `SET name = value`
    // records here, `RESET` clears, `SHOW name` / `current_setting(name)` read it back. Scoped to this
    // connection so one client's settings never leak to another, and read into every statement so a
    // later `current_setting` reflects an earlier `SET`. Shared into each `spawn_blocking` task via the
    // `Arc`; statements on one connection run strictly serially, so the `Mutex` is never contended.
    let settings: Arc<std::sync::Mutex<HashMap<String, String>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    // Per-connection plan cache: reuses a planned read query when the same SQL is issued again
    // and none of its tables changed schema. Scoped to this connection (hence this user), so a cached
    // plan — which bakes RLS predicates for the analyzing user — is never served across users.
    let mut plan_cache = nusadb_sql::PlanCache::new();
    // LISTEN/NOTIFY (async pub/sub): this connection's notification mailbox. `LISTEN` subscribes its
    // pid in the process-global registry; a `NOTIFY` from any connection in the same database pushes a
    // `Notification` here, which the loop below drains and writes as a `NotificationResponse` while the
    // connection is idle. The registration removes the pid (and its subscriptions) when it ends.
    let (notif_tx, mut notif_rx) = tokio::sync::mpsc::unbounded_channel::<notify::Notification>();
    let _notif_registration = notify::register(key.pid, database.clone(), notif_tx);
    // NOTIFY is transactional, like the reference engine: a NOTIFY issued inside an explicit transaction is queued here
    // and delivered only when the transaction COMMITs; ROLLBACK (or ROLLBACK TO SAVEPOINT) discards
    // the queued notifications. Empty (and untouched) in autocommit, where NOTIFY delivers eagerly.
    let mut pending_notify = PendingNotifications::default();

    // `idle` is true whenever the connection sits at a `ReadyForQuery` boundary (after startup, a
    // simple query, or a `Sync`) — the only safe point to inject an unsolicited notification, never
    // mid-extended-query-pipeline. While idle the `select!` delivers a queued notification (then loops
    // back, still idle); otherwise it reads the next client frame.
    let mut idle = true;
    loop {
        let frame = tokio::select! {
            maybe_notif = notif_rx.recv(), if idle => {
                match maybe_notif {
                    Some(n) => {
                        conn.write_frame(
                            &BackendMessage::NotificationResponse {
                                pid: n.pid,
                                channel: n.channel,
                                payload: n.payload,
                            }
                            .encode()?,
                        )
                        .await?;
                        continue;
                    },
                    // The sender is only dropped when our own registration drops (connection end),
                    // which cannot happen while this loop runs; treat a close defensively as no-op.
                    None => continue,
                }
            },
            frame = read_next(&mut conn, idle_timeout, &mut shutdown) => frame?,
        };
        let Some(frame) = frame else { break };
        idle = false;
        match FrontendMessage::decode(&frame).map_err(io::Error::other)? {
            FrontendMessage::Query { sql } => {
                failed = false; // a simple query abandons any half-built extended pipeline
                if let Some(copy) = copy_statement(&sql) {
                    // COPY ... FROM STDIN / TO STDOUT drives the COPY sub-protocol. COPY does not pass
                    // through the RLS-aware analyzer, so refuse it on an RLS-enabled table for a
                    // non-superuser — fail closed, never bypass the policy.
                    let outcome = if let Some(msg) = copy_rls_block(engine.as_ref(), &copy.table, &user)
                    {
                        Err(msg)
                    } else {
                        match copy.direction {
                            nusadb_sql::ast::CopyDirection::From => {
                                handle_copy_in(
                                    &mut conn,
                                    &engine,
                                    copy,
                                    idle_timeout,
                                    &mut shutdown,
                                    copy_from_max_bytes,
                                )
                                .await?
                            },
                            nusadb_sql::ast::CopyDirection::To => {
                                handle_copy_out(&mut conn, &engine, copy).await?
                            },
                        }
                    };
                    if let Some(m) = &metrics {
                        m.query(outcome.is_ok());
                    }
                    match outcome {
                        Ok(count) => {
                            conn.write_frame(&command_complete(&format!("COPY {count}")).encode()?)
                                .await?;
                        },
                        Err(message) => {
                            conn.write_frame(&error_response(&message).encode()?)
                                .await?;
                        },
                    }
                } else if let Some(stmt) = pubsub_statement(&sql) {
                    // LISTEN / UNLISTEN / NOTIFY (async pub/sub): intercepted here because the channel
                    // registry spans connections and lives in the wire server, not the SQL engine. The
                    // reply is a bare command tag; any actual notification arrives asynchronously via
                    // the `select!` above. NOTIFY inside a transaction is queued in `pending_notify`
                    // and flushed on COMMIT below (transactional NOTIFY).
                    handle_pubsub(
                        &mut conn,
                        key.pid,
                        &database,
                        txn_state.notify_phase(),
                        &mut pending_notify,
                        stmt,
                    )
                    .await?;
                } else {
                    // Classify a transaction-control statement before the SQL text is moved into the
                    // streaming call, and note whether a transaction was actually open: a COMMIT of an
                    // *active* transaction flushes the queued NOTIFYs, but a COMMIT of a *failed* one
                    // rolls back (so it must discard them instead).
                    let control = txn_control_kind(&sql);
                    let was_active = matches!(txn_state, TxnState::Active { .. });
                    // Stream the result's rows to the socket as they are produced (Phase 2):
                    // `RowDescription`, then each `DataRow` as the executor yields it, then
                    // `CommandComplete` — bounding the wire layer's memory to the channel capacity
                    // instead of buffering the whole result set. The frame sequence and bytes are
                    // identical to the buffered path.
                    let (ok, new_state, new_cache) = stream_query_to_conn(
                        &mut conn,
                        &engine,
                        &cluster,
                        &database,
                        sql,
                        &[], // simple query: no bound parameters
                        user.clone(),
                        Arc::clone(&cancel_token),
                        statement_timeout,
                        txn_state,
                        typed_row_description,
                        array_element_types,
                        plan_cache,
                        &settings,
                        false, // simple query sends its own RowDescription
                    )
                    .await?;
                    txn_state = new_state;
                    plan_cache = new_cache;
                    // Apply the queued-NOTIFY effect of a transaction-control statement that ran ok.
                    if ok {
                        match control {
                            Some(TxnControl::Commit) if was_active => {
                                pending_notify.flush(key.pid, &database);
                            },
                            // COMMIT of a failed transaction is a rollback; ROLLBACK and a fresh BEGIN
                            // both drop anything queued.
                            Some(TxnControl::Commit | TxnControl::Rollback | TxnControl::Begin) => {
                                pending_notify.discard();
                            },
                            Some(TxnControl::Savepoint(name)) => pending_notify.savepoint(name),
                            Some(TxnControl::Release(name)) => pending_notify.release(&name),
                            Some(TxnControl::RollbackTo(name)) => pending_notify.rollback_to(&name),
                            None => {},
                        }
                    }
                    if let Some(m) = &metrics {
                        m.query(ok);
                    }
                }
                conn.write_frame(&BackendMessage::ReadyForQuery(txn_state.status()).encode()?)
                    .await?;
                idle = true;
            },
            FrontendMessage::Parse { name, sql, .. } => {
                if failed {
                    continue;
                }
                statements.insert(name, sql);
                conn.write_frame(&BackendMessage::ParseComplete.encode()?)
                    .await?;
            },
            FrontendMessage::Bind {
                portal,
                statement,
                params,
                result_formats,
            } => {
                if failed {
                    continue;
                }
                if let Some(sql) = statements.get(&statement) {
                    portals.insert(
                        portal,
                        Portal {
                            sql: sql.clone(),
                            params,
                            result_formats,
                            result: None,
                        },
                    );
                    conn.write_frame(&BackendMessage::BindComplete.encode()?)
                        .await?;
                } else {
                    let msg = format!("unknown prepared statement {statement:?}");
                    fail(&mut conn, &mut failed, &msg).await?;
                }
            },
            FrontendMessage::Describe { target, name } => {
                if failed {
                    continue;
                }
                match target {
                    DescribeTarget::Statement => {
                        if let Some(sql) = statements.get(&name) {
                            // Report the real number of `$n` placeholders, not a hard-coded 0.
                            // A statement that fails to parse describes as 0 params — its parse
                            // error surfaces later at Execute. Row metadata stays per-portal (NoData).
                            let count = match parse(sql).ok().map(|stmt| parameter_count(&stmt)) {
                                None => 0,
                                Some(n) => {
                                    if let Ok(c) = u16::try_from(n) {
                                        c
                                    } else {
                                        // More `$n` placeholders than the wire's u16 count can
                                        // carry: fail cleanly rather than silently saturating to
                                        // u16::MAX, which would under-report the real count.
                                        let msg = format!(
                                            "prepared statement {name:?} declares {n} parameters, \
                                             exceeding the protocol limit of {}",
                                            u16::MAX
                                        );
                                        fail(&mut conn, &mut failed, &msg).await?;
                                        continue;
                                    }
                                },
                            };
                            conn.write_frame(
                                &BackendMessage::ParameterDescription { count }.encode()?,
                            )
                            .await?;
                            conn.write_frame(&BackendMessage::NoData.encode()?).await?;
                        } else {
                            let msg = format!("unknown prepared statement {name:?}");
                            fail(&mut conn, &mut failed, &msg).await?;
                        }
                    },
                    // Plan-only: report the row shape without executing (side effects wait for
                    // Execute).
                    DescribeTarget::Portal => match describe_portal(
                        &engine, &portals, &name, &user,
                    )
                    .await
                    {
                        Ok((columns, _types)) if columns.is_empty() => {
                            conn.write_frame(&BackendMessage::NoData.encode()?).await?;
                        },
                        Ok((columns, types)) => {
                            // At protocol minor >= 1 emit the typed metadata when types are known
                            // (a Describe-before-Execute resolves them from the plan); otherwise the
                            // classic names-only RowDescription (byte-identical to 1.0).
                            let msg = if typed_row_description && columns.len() == types.len() {
                                BackendMessage::RowDescriptionTyped {
                                    columns: columns
                                        .into_iter()
                                        .zip(types)
                                        .map(|(name, ty)| (name, type_tag_for(ty, array_element_types)))
                                        .collect(),
                                }
                            } else {
                                BackendMessage::RowDescription { columns }
                            };
                            conn.write_frame(&msg.encode()?).await?;
                        },
                        Err(msg) => fail(&mut conn, &mut failed, &msg).await?,
                    },
                }
            },
            FrontendMessage::Execute { portal, max_rows } => {
                if failed {
                    continue;
                }
                // A fresh fetch-all `Execute` (`max_rows == 0`) over a text-format portal
                // streams its rows straight to the socket instead of
                // materializing the whole result set in memory first — this closes the driver
                // fetch-all OOM. Anything else (already executed, paginated `max_rows > 0`, or a
                // binary result format we cannot stream text for) keeps the buffered
                // materialize-then-drain path below unchanged.
                let stream_job = (max_rows == 0)
                    .then(|| portals.get(&portal))
                    .flatten()
                    .filter(|p| p.result.is_none() && p.result_formats.iter().all(|&f| f == 0))
                    .map(|p| (p.sql.clone(), p.params.clone()));
                if let Some((sql, params)) = stream_job {
                    let (ok, new_state, new_cache) = stream_query_to_conn(
                        &mut conn,
                        &engine,
                        &cluster,
                        &database,
                        sql,
                        &params,
                        user.clone(),
                        Arc::clone(&cancel_token),
                        statement_timeout,
                        txn_state,
                        typed_row_description,
                        array_element_types,
                        plan_cache,
                        &settings,
                        true, // portal path: `Describe` already sent `RowDescription`
                    )
                    .await?;
                    txn_state = new_state;
                    plan_cache = new_cache;
                    if let Some(m) = &metrics {
                        m.query(ok);
                    }
                    if ok {
                        // Mark the portal drained so a re-`Execute` before `Sync` is a no-op
                        // rather than re-running the query against the (now advanced) transaction.
                        if let Some(p) = portals.get_mut(&portal) {
                            p.result = Some(PortalResult {
                                columns: Vec::new(),
                                rows: VecDeque::new(),
                                tag: String::new(),
                                completed: true,
                            });
                        }
                    } else {
                        // The stream already wrote its `ErrorResponse`; enter the failed state so
                        // frames up to the next `Sync` are skipped (matching the buffered error
                        // path). Leave `result` unset so a post-`Sync` re-`Execute` re-runs.
                        failed = true;
                    }
                    continue;
                }
                // Count the query exactly once — on the execution itself, not on every `Execute` of
                // the same portal (a re-`Execute` after the rows were drained must not re-count).
                let already_executed = portals.get(&portal).is_some_and(|p| p.result.is_some());
                let (executed, new_state) = ensure_executed(
                    &engine,
                    &cluster,
                    &database,
                    &mut portals,
                    &portal,
                    &user,
                    Arc::clone(&cancel_token),
                    statement_timeout,
                    txn_state,
                    &settings,
                )
                .await;
                txn_state = new_state;
                match executed {
                    Ok(_columns) => {
                        if !already_executed && let Some(m) = &metrics {
                            m.query(true);
                        }
                        if let Some(p) = portals.get_mut(&portal) {
                            for msg in drain_portal(p, max_rows) {
                                conn.write_frame(&msg.encode()?).await?;
                            }
                        }
                    },
                    Err((msg, code)) => {
                        if !already_executed && let Some(m) = &metrics {
                            m.query(false);
                        }
                        fail_coded(&mut conn, &mut failed, &msg, code).await?;
                    },
                }
            },
            FrontendMessage::Sync => {
                failed = false;
                conn.write_frame(&BackendMessage::ReadyForQuery(txn_state.status()).encode()?)
                    .await?;
                idle = true;
            },
            FrontendMessage::Close { target, name } => {
                if failed {
                    continue;
                }
                match target {
                    DescribeTarget::Statement => {
                        statements.remove(&name);
                    },
                    DescribeTarget::Portal => {
                        portals.remove(&name);
                    },
                }
                conn.write_frame(&BackendMessage::CloseComplete.encode()?)
                    .await?;
            },
            // COPY and SASL messages are consumed by their own handlers (`handle_copy_in` /
            // `authenticate`); any that arrive outside that exchange are stray (e.g. a client error)
            // and are harmlessly dropped — each frame is length-delimited, so ignoring one cannot
            // desync the stream.
            FrontendMessage::CopyData { .. }
            | FrontendMessage::CopyDone
            | FrontendMessage::CopyFail { .. }
            | FrontendMessage::SaslInitialResponse { .. }
            | FrontendMessage::SaslResponse { .. }
            // A CancelRequest is only meaningful as the first frame of a fresh connection; one
            // arriving mid-session is stray and harmlessly ignored.
            | FrontendMessage::CancelRequest { .. } => {},
            // Terminate ends the session normally; Startup mid-session is a
            // protocol violation we treat as termination rather than crash.
            FrontendMessage::Terminate | FrontendMessage::Startup { .. } => break,
        }
    }

    // The connection is ending: push out anything still queued (e.g. a response to the frame that
    // preceded `Terminate`) before the socket drops.
    conn.flush_now().await?;

    // Roll back an explicit transaction left open when the connection ends, so a disconnect mid
    // `BEGIN ... ` does not leak it in the engine.
    if let Some(txn) = txn_state.open_txn() {
        let engine = Arc::clone(&engine);
        let _ = tokio::task::spawn_blocking(move || engine.rollback(txn)).await;
    }
    Ok(())
}

/// If `sql` is a `COPY ... FROM STDIN` / `TO STDOUT`, parse and return it; otherwise `None` (the
/// normal execute path handles every other statement). The cheap prefix check avoids parsing every
/// simple query twice.
fn copy_statement(sql: &str) -> Option<nusadb_sql::ast::Copy> {
    if !sql.trim_start().get(..4)?.eq_ignore_ascii_case("copy") {
        return None;
    }
    match parse(sql) {
        Ok(nusadb_sql::ast::Statement::Copy(copy)) => Some(copy),
        _ => None,
    }
}

/// If `sql` is a `LISTEN` / `UNLISTEN` / `NOTIFY` statement, parse and return it; otherwise `None`
/// (the normal execute path handles everything else). The cheap prefix check avoids parsing every
/// simple query twice; a false-positive prefix (e.g. an identifier starting with `listen`) parses to
/// some other statement and returns `None`.
fn pubsub_statement(sql: &str) -> Option<nusadb_sql::ast::Statement> {
    use nusadb_sql::ast::Statement;
    let head = sql.trim_start();
    let looks_pubsub = ["listen", "unlisten", "notify"].iter().any(|kw| {
        head.get(..kw.len())
            .is_some_and(|p| p.eq_ignore_ascii_case(kw))
    });
    if !looks_pubsub {
        return None;
    }
    match parse(sql) {
        Ok(s @ (Statement::Listen(_) | Statement::Unlisten(_) | Statement::Notify { .. })) => {
            Some(s)
        },
        _ => None,
    }
}

/// Handle an intercepted `LISTEN` / `UNLISTEN` / `NOTIFY` against the process-global notification
/// registry ([`crate::notify`]), then reply with the bare command tag. `pid` is this connection's
/// backend id and `database` the database it is bound to (notifications are delivered only within a
/// database, like the reference engine). `phase` gates `NOTIFY` for transactional delivery: in autocommit it delivers
/// immediately; inside an active transaction it is queued in `pending` (flushed by the caller on
/// COMMIT, discarded on ROLLBACK); inside a failed transaction it is rejected, like any other command.
/// `LISTEN`/`UNLISTEN` take effect immediately (a minor deviation from the reference engine, which also undoes them on
/// rollback) — they have no external side effect a rollback must take back.
async fn handle_pubsub<S>(
    conn: &mut Connection<S>,
    pid: u32,
    database: &str,
    phase: NotifyPhase,
    pending: &mut PendingNotifications,
    stmt: nusadb_sql::ast::Statement,
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use nusadb_sql::ast::Statement;
    let outcome: Result<&str, (String, &str)> = match stmt {
        Statement::Listen(channel) => {
            notify::listen(pid, channel);
            Ok("LISTEN")
        },
        Statement::Unlisten(channel) => {
            notify::unlisten(pid, channel.as_deref());
            Ok("UNLISTEN")
        },
        Statement::Notify { channel, payload } => {
            let payload = payload.unwrap_or_default();
            match phase {
                NotifyPhase::Aborted => Err((
                    "current transaction is aborted, commands ignored until end of transaction block"
                        .to_owned(),
                    "25P02", // in_failed_sql_transaction
                )),
                // Queue until COMMIT: a later ROLLBACK must not leave a notification sent for work
                // that never committed.
                NotifyPhase::InTransaction => {
                    pending.enqueue(channel, payload);
                    Ok("NOTIFY")
                },
                NotifyPhase::Autocommit => {
                    let notification = notify::Notification {
                        pid,
                        channel: channel.clone(),
                        payload,
                    };
                    notify::notify(database, &channel, &notification);
                    Ok("NOTIFY")
                },
            }
        },
        _ => unreachable!("handle_pubsub called with a non-pubsub statement"),
    };
    let frame = match outcome {
        Ok(tag) => BackendMessage::CommandComplete {
            tag: tag.to_owned(),
        },
        Err((message, code)) => error_response_coded(&message, code),
    };
    conn.write_frame(&frame.encode()?).await
}

/// Drive the `COPY ... FROM STDIN` sub-protocol: send `CopyInResponse`, gather the client's
/// `CopyData` stream until `CopyDone` (or `CopyFail`), then bulk-load it. Returns the row count, or
/// an error message to relay to the client. Wire I/O errors propagate; a load failure (bad row,
/// constraint, client abort) returns `Err(message)` without tearing down the connection.
async fn handle_copy_in<S>(
    conn: &mut Connection<S>,
    engine: &Arc<dyn StorageEngine>,
    copy: nusadb_sql::ast::Copy,
    idle_timeout: Option<Duration>,
    shutdown: &mut Option<watch::Receiver<bool>>,
    max_bytes: Option<usize>,
) -> io::Result<Result<usize, String>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // Advisory column count for the client; the real load arity is validated against the table in
    // `copy_from`, so an (implausible) >65535-column overflow harmlessly reports 0 here.
    let columns = u16::try_from(copy.columns.len()).unwrap_or(0);
    conn.write_frame(&BackendMessage::CopyInResponse { columns }.encode()?)
        .await?;

    // Accumulate the client's `CopyData` stream, but cap the cumulative size: without a cap
    // an authenticated client could stream without bound and OOM the server. Once the cap is
    // exceeded we stop buffering and free what we held, yet keep reading until `CopyDone`/`CopyFail`
    // so the connection stays in protocol sync (the leftover frames don't desync the next query),
    // then report the error.
    let mut buf: Vec<u8> = Vec::new();
    let mut overflow = false;
    loop {
        let Some(frame) = read_next(conn, idle_timeout, shutdown).await? else {
            return Ok(Err("connection closed during COPY FROM".to_owned()));
        };
        match FrontendMessage::decode(&frame).map_err(io::Error::other)? {
            FrontendMessage::CopyData { data } => {
                if overflow {
                    continue; // already over the cap — drain and discard
                }
                match max_bytes {
                    Some(max) if buf.len().saturating_add(data.len()) > max => {
                        overflow = true;
                        buf = Vec::new(); // release the buffered data promptly
                    },
                    _ => buf.extend_from_slice(&data),
                }
            },
            FrontendMessage::CopyDone => break,
            FrontendMessage::CopyFail { message } => {
                return Ok(Err(format!("COPY aborted by client: {message}")));
            },
            // Any other message mid-copy is a protocol violation; abort the load.
            _ => return Ok(Err("unexpected message during COPY FROM".to_owned())),
        }
    }
    if overflow {
        // `overflow` is only set in the `Some(max)` arm, so the cap is known.
        return Ok(Err(format!(
            "COPY data exceeds the {}-byte limit",
            max_bytes.unwrap_or(0)
        )));
    }

    let Ok(data) = String::from_utf8(buf) else {
        return Ok(Err("COPY data is not valid UTF-8".to_owned()));
    };
    let engine = Arc::clone(engine);
    let result = tokio::task::spawn_blocking(move || {
        copy_from(engine.as_ref(), &copy, &data).map_err(|e| e.to_string())
    })
    .await
    .unwrap_or_else(|_join| Err("internal execution error".to_owned()));
    Ok(result)
}

/// Drive the `COPY ... TO STDOUT` sub-protocol: render the rows, then stream them as
/// `CopyOutResponse` + one `CopyData` + `CopyDone`. Returns the row count, or an error message to
/// relay (sent before any copy-out frame, so a failed render never starts a half-stream).
async fn handle_copy_out<S>(
    conn: &mut Connection<S>,
    engine: &Arc<dyn StorageEngine>,
    copy: nusadb_sql::ast::Copy,
) -> io::Result<Result<usize, String>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let columns = u16::try_from(copy.columns.len()).unwrap_or(0);
    let engine = Arc::clone(engine);
    let rendered = tokio::task::spawn_blocking(move || {
        copy_to(engine.as_ref(), &copy).map_err(|e| e.to_string())
    })
    .await
    .unwrap_or_else(|_join| Err("internal execution error".to_owned()));

    let (count, payload) = match rendered {
        Ok(out) => out,
        Err(message) => return Ok(Err(message)),
    };
    conn.write_frame(&BackendMessage::CopyOutResponse { columns }.encode()?)
        .await?;
    conn.write_frame(
        &BackendMessage::CopyData {
            data: payload.into_bytes(),
        }
        .encode()?,
    )
    .await?;
    conn.write_frame(&BackendMessage::CopyDone.encode()?)
        .await?;
    Ok(Ok(count))
}

/// RAII guard for the active-connection gauge: increments on construction and decrements on drop,
/// so the count is balanced even if the connection task unwinds on a panic.
struct ConnectionGuard(Option<Arc<Metrics>>);

impl ConnectionGuard {
    fn new(metrics: Option<Arc<Metrics>>) -> Self {
        if let Some(m) = &metrics {
            m.connection_opened();
        }
        Self(metrics)
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        if let Some(m) = &self.0 {
            m.connection_closed();
        }
    }
}

/// A bound portal: the resolved SQL, its bound parameters, and its lazily-executed result.
struct Portal {
    /// SQL resolved from the prepared statement at `Bind` time.
    sql: String,
    /// Wire-format `$n` parameter values supplied by `Bind`.
    params: Vec<Option<Vec<u8>>>,
    /// Per-column result format codes from `Bind`: `0` = text, `1` = binary. Empty = all
    /// text; a single entry applies to every column.
    result_formats: Vec<u16>,
    /// Filled on the first `Describe`/`Execute`; `Execute` drains its rows.
    result: Option<PortalResult>,
}

/// A portal's executed result, ready to stream.
struct PortalResult {
    /// Output column names (empty for a non-row command).
    columns: Vec<String>,
    /// Rows still to emit (drained by `Execute`).
    rows: VecDeque<Vec<Option<Vec<u8>>>>,
    /// The `CommandComplete` tag, computed once at execution.
    tag: String,
    /// Set once the portal has been fully drained and its `CommandComplete` emitted, so a second
    /// `Execute` on the same portal does not replay rows or re-send the command tag.
    completed: bool,
}

/// The statement-effective timeout: a session `SET statement_timeout` overrides the server's
/// `--statement-timeout` default, and a session `0` disables the timeout entirely. An absent —
/// or, defensively, unparseable — session value falls back to the server default (`SET`-time
/// validation in [`apply_set_variable`] rejects unparseable values, so the fallback is defense in
/// depth, not policy). Read once per statement, right before the cancel timer is armed.
fn effective_statement_timeout(
    settings: &std::sync::Mutex<HashMap<String, String>>,
    server_default: Option<Duration>,
) -> Option<Duration> {
    let session = settings
        .lock()
        .ok()
        .and_then(|s| s.get("statement_timeout").cloned());
    match session
        .as_deref()
        .and_then(nusadb_sql::cancel::parse_statement_timeout)
    {
        Some(timeout) if timeout.is_zero() => None, // session opted out of any timeout
        Some(timeout) => Some(timeout),
        None => server_default,
    }
}

/// Run one statement on the blocking pool, flattening the join error.
#[allow(
    clippy::too_many_arguments,
    reason = "threads engine, cluster, database, statement, params, user, cancellation, timeout, \
              txn state, and the per-connection GUC store — each a distinct per-statement concern"
)]
async fn run_blocking(
    engine: &Arc<dyn StorageEngine>,
    cluster: &Arc<dyn DatabaseCluster>,
    database: &str,
    sql: String,
    params: Vec<Option<Vec<u8>>>,
    user: String,
    cancel: nusadb_sql::cancel::CancelToken,
    statement_timeout: Option<Duration>,
    state: TxnState,
    settings: &Arc<std::sync::Mutex<HashMap<String, String>>>,
) -> (Result<ExecutionResult, (String, &'static str)>, TxnState) {
    // The per-connection cancel flag. Reset it for this statement: a cancel that
    // arrived between statements (when nothing was running) must not abort this one. It is tripped
    // by the statement-timeout timer below or by an out-of-band `CancelRequest` mid-statement.
    cancel.store(false, Ordering::Relaxed);
    let timer = effective_statement_timeout(settings, statement_timeout).map(|deadline| {
        let token = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::time::sleep(deadline).await;
            token.store(true, Ordering::Relaxed);
        })
    });

    let engine = Arc::clone(engine);
    let cluster = Arc::clone(cluster);
    let database = database.to_owned();
    let settings = Arc::clone(settings);
    let (result, new_state) = tokio::task::spawn_blocking(move || {
        let _cancel_guard = nusadb_sql::cancel::scope(cancel);
        let (outcome, new_state) = run_query_txn(
            engine.as_ref(),
            cluster.as_ref(),
            &database,
            &sql,
            &params,
            &user,
            state,
            &settings,
        );
        (
            outcome.map_err(|e| (e.to_string(), e.sqlstate())),
            new_state,
        )
    })
    .await
    .unwrap_or_else(|_join| (Err(("internal execution error".to_owned(), "XX000")), state));

    // The statement finished (or failed); stop the timer so it cannot trip a later statement.
    if let Some(timer) = timer {
        timer.abort();
    }
    (result, new_state)
}

/// Compute a portal's output column names **without executing it**: parse → bind → analyze →
/// plan, then read the plan's output shape. `Describe(Portal)` must report row metadata without
/// running the statement, so its side effects are deferred to `Execute`.
async fn describe_portal(
    engine: &Arc<dyn StorageEngine>,
    portals: &HashMap<String, Portal>,
    name: &str,
    user: &str,
) -> Result<(Vec<String>, Vec<nusadb_core::ColumnType>), String> {
    let (sql, params) = match portals.get(name) {
        // A portal already executed (its rows are cached) reuses the executed column names. Types are
        // not cached, so a Describe after Execute falls back to the untyped form (empty type list).
        Some(p) => match &p.result {
            Some(result) => return Ok((result.columns.clone(), Vec::new())),
            None => (p.sql.clone(), p.params.clone()),
        },
        None => return Err(format!("unknown portal {name:?}")),
    };
    let engine = Arc::clone(engine);
    let user = user.to_owned();
    tokio::task::spawn_blocking(move || {
        let stmt = bind_parameters(parse(&sql)?, &params)?;
        // Describe resolves the row shape without running the statement, but analysis still
        // needs a transaction to resolve schema visibility. Use a short read-only
        // transaction and roll it back — Describe must have no side effects. The connection's user
        // is carried so RLS-restricted columns analyze the same way they will at execution.
        let txn = engine.begin(IsolationLevel::default())?;
        let result = analyze(
            stmt,
            &EngineCatalog::new(engine.as_ref(), txn, &user, &HashMap::new()),
        )
        .map(|logical| {
            let physical = plan(logical);
            (
                describe_columns(&physical),
                describe_column_types(&physical),
            )
        });
        let _ = engine.rollback(txn);
        result
    })
    .await
    .unwrap_or_else(|_join| Err(nusadb_sql::Error::Unsupported("internal error".to_owned())))
    .map_err(|e: nusadb_sql::Error| e.to_string())
}

/// Ensure the named portal has been executed (running it once, lazily), returning its output
/// column names. Side effects of the statement happen here, exactly once.
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection context threaded into the portal execution (engine, cluster, database, \
              portals, portal name, user, cancellation, timeout, txn state, GUC store)"
)]
async fn ensure_executed(
    engine: &Arc<dyn StorageEngine>,
    cluster: &Arc<dyn DatabaseCluster>,
    database: &str,
    portals: &mut HashMap<String, Portal>,
    name: &str,
    user: &str,
    cancel: nusadb_sql::cancel::CancelToken,
    statement_timeout: Option<Duration>,
    state: TxnState,
    settings: &Arc<std::sync::Mutex<HashMap<String, String>>>,
) -> (Result<Vec<String>, (String, &'static str)>, TxnState) {
    let job = match portals.get(name) {
        Some(p) if p.result.is_none() => {
            Some((p.sql.clone(), p.params.clone(), p.result_formats.clone()))
        },
        Some(_) => None, // already executed
        None => return (Err((format!("unknown portal {name:?}"), "XX000")), state),
    };
    let mut state = state;
    if let Some((sql, params, formats)) = job {
        // The borrow on `portals` is dropped before this await (we cloned `sql`/`params`).
        let (outcome, new_state) = run_blocking(
            engine,
            cluster,
            database,
            sql,
            params,
            user.to_owned(),
            cancel,
            statement_timeout,
            state,
            settings,
        )
        .await;
        state = new_state;
        match outcome {
            Ok(exec_result) => {
                let result = into_portal_result(exec_result, &formats);
                if let Some(p) = portals.get_mut(name) {
                    p.result = Some(result);
                }
            },
            Err(message) => return (Err(message), state),
        }
    }
    let columns = portals
        .get(name)
        .and_then(|p| p.result.as_ref())
        .map(|r| r.columns.clone())
        .unwrap_or_default();
    (Ok(columns), state)
}

/// Emit up to `max_rows` (`0` = all) buffered rows of `portal` as `DataRow`s, then either
/// `CommandComplete` (drained) or `PortalSuspended` (rows remain). Extended-protocol `Execute`
/// does not repeat `RowDescription` — that comes from `Describe`.
fn drain_portal(portal: &mut Portal, max_rows: u32) -> Vec<BackendMessage> {
    let Some(result) = portal.result.as_mut() else {
        return Vec::new();
    };
    let limit = if max_rows == 0 {
        usize::MAX
    } else {
        max_rows as usize
    };
    // A portal already fully drained must not replay its rows or re-send `CommandComplete`.
    if result.completed {
        return Vec::new();
    }
    let mut msgs = Vec::new();
    while msgs.len() < limit {
        match result.rows.pop_front() {
            Some(values) => msgs.push(BackendMessage::DataRow { values }),
            None => break,
        }
    }
    if result.rows.is_empty() {
        result.completed = true;
        msgs.push(command_complete(&result.tag));
    } else {
        msgs.push(BackendMessage::PortalSuspended);
    }
    msgs
}

/// Convert an executed result into the portal's drainable form (columns + row fields + tag).
fn into_portal_result(result: ExecutionResult, result_formats: &[u16]) -> PortalResult {
    let tag = command_tag(&result);
    match result {
        ExecutionResult::Rows { columns, rows } => PortalResult {
            columns,
            rows: rows
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .enumerate()
                        .map(|(col, value)| encode_field(value, column_format(result_formats, col)))
                        .collect()
                })
                .collect(),
            tag,
            completed: false,
        },
        _ => PortalResult {
            columns: Vec::new(),
            rows: VecDeque::new(),
            tag,
            completed: false,
        },
    }
}

/// Run the SCRAM-SHA-256 SASL handshake. Returns `Ok(true)` on success (the client proved
/// it knows the password) or `Ok(false)` if the handshake fails or is abandoned — in which case an
/// error has been sent and the caller drops the connection. Wire I/O errors propagate.
///
/// The flow is: server offers `SCRAM-SHA-256` → client sends `client-first` → server replies
/// `server-first` (combined nonce + the user's salt + iteration count) → client sends `client-final`
/// (its proof) → server verifies the proof in constant time and replies `server-final` + `AuthOk`.
async fn authenticate<S>(
    conn: &mut Connection<S>,
    user: &str,
    auth: &AuthStore,
    idle_timeout: Option<Duration>,
    shutdown: &mut Option<watch::Receiver<bool>>,
) -> io::Result<bool>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    const MECHANISM: &str = "SCRAM-SHA-256";

    conn.write_frame(
        &BackendMessage::AuthSasl {
            mechanisms: vec![MECHANISM.to_owned()],
        }
        .encode()?,
    )
    .await?;

    // --- client-first ---
    let Some(frame) = read_next(conn, idle_timeout, shutdown).await? else {
        return Ok(false);
    };
    let FrontendMessage::SaslInitialResponse { mechanism, data } =
        FrontendMessage::decode(&frame).map_err(io::Error::other)?
    else {
        return auth_fail(conn, "expected a SASLInitialResponse").await;
    };
    if mechanism != MECHANISM {
        return auth_fail(conn, "unsupported SASL mechanism").await;
    }
    let Some(client_first) = std::str::from_utf8(&data)
        .ok()
        .and_then(|s| scram::ClientFirst::parse(s).ok())
    else {
        return auth_fail(conn, "malformed client-first message").await;
    };
    // The SCRAM username must match the connection's declared user (no cross-user auth). Look the
    // credentials up; a missing user fails with the same generic message to avoid user enumeration.
    let creds = match auth.lookup(user) {
        Some(creds) if client_first.username == user => creds,
        _ => return auth_fail(conn, "authentication failed").await,
    };

    // --- server-first ---
    let Ok(server_nonce) = scram::generate_nonce() else {
        return auth_fail(conn, "internal authentication error").await;
    };
    let server_first = scram::ServerFirst::build(
        &client_first,
        creds.salt.clone(),
        creds.iterations,
        &server_nonce,
    );
    let server_first_msg = server_first.to_message();
    conn.write_frame(
        &BackendMessage::AuthSaslContinue {
            data: server_first_msg.clone().into_bytes(),
        }
        .encode()?,
    )
    .await?;

    // --- client-final ---
    let Some(frame) = read_next(conn, idle_timeout, shutdown).await? else {
        return Ok(false);
    };
    let FrontendMessage::SaslResponse { data } =
        FrontendMessage::decode(&frame).map_err(io::Error::other)?
    else {
        return auth_fail(conn, "expected a SASLResponse").await;
    };
    let Some(client_final) = std::str::from_utf8(&data)
        .ok()
        .and_then(|s| scram::ClientFinal::parse(s).ok())
    else {
        return auth_fail(conn, "malformed client-final message").await;
    };
    // The nonce must echo the combined nonce, and the channel-binding `c=` must echo the GS2 header
    // (no channel binding in this build). Either mismatch fails authentication.
    if client_final.combined_nonce != server_first.combined_nonce
        || client_final.channel_binding != client_first.gs2_header.as_bytes()
    {
        return auth_fail(conn, "authentication failed").await;
    }
    let auth_msg = scram::auth_message(
        &client_first.bare,
        &server_first_msg,
        &client_final.without_proof,
    );
    let Ok(server_final) = scram::verify_client_proof(creds, &auth_msg, &client_final.proof) else {
        return auth_fail(conn, "authentication failed").await;
    };
    conn.write_frame(
        &BackendMessage::AuthSaslFinal {
            data: server_final.into_bytes(),
        }
        .encode()?,
    )
    .await?;
    conn.write_frame(&BackendMessage::AuthOk.encode()?).await?;
    Ok(true)
}

/// Send an authentication error and return `Ok(false)` (the handshake failed).
///
/// The connection closes right after, so the error is flushed here — it would otherwise die in
/// the output queue.
async fn auth_fail<S>(conn: &mut Connection<S>, message: &str) -> io::Result<bool>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    conn.write_frame(&error_response(message).encode()?).await?;
    conn.flush_now().await?;
    Ok(false)
}

/// Write an error response and enter the "skip until Sync" state.
async fn fail<S>(conn: &mut Connection<S>, failed: &mut bool, message: &str) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fail_coded(conn, failed, message, "XX000").await
}

/// Like [`fail`], but with an explicit SQLSTATE class code (B-QA SQLSTATE) so a serialization
/// conflict / deadlock surfaced from statement execution reaches the client as a retryable error.
async fn fail_coded<S>(
    conn: &mut Connection<S>,
    failed: &mut bool,
    message: &str,
    code: &str,
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    *failed = true;
    conn.write_frame(&error_response_coded(message, code).encode()?)
        .await
}

/// Refuse `COPY` on an RLS-enabled or system-catalog `table` for a non-superuser.
///
/// `COPY` does not pass through the RLS-aware analyzer, so — like writes and joins over an RLS table
/// — it fails closed rather than read or write rows a policy would restrict. The reserved
/// `nusadb_*` namespace is likewise refused here, or COPY would be the one remaining path a user
/// could read or rewrite the policy/RLS catalogs through. Returns an error message to relay, or
/// `None` to proceed. If the RLS flag cannot be read, it fails closed.
fn copy_rls_block(engine: &dyn StorageEngine, table: &str, user: &str) -> Option<String> {
    if user == nusadb_sql::BOOTSTRAP_SUPERUSER {
        return None;
    }
    if table.starts_with(nusadb_sql::SYSTEM_TABLE_PREFIX) {
        return Some(format!(
            "`{table}` is in the reserved system-catalog namespace (`{}*`); only a superuser may \
             reference it",
            nusadb_sql::SYSTEM_TABLE_PREFIX
        ));
    }
    let Ok(txn) = engine.begin(IsolationLevel::default()) else {
        return Some("could not verify row-level security for COPY".to_owned());
    };
    let enabled = nusadb_sql::rls_table_enabled(engine, txn, table);
    let _ = engine.rollback(txn);
    match enabled {
        Ok(false) => None,
        Ok(true) => Some(format!(
            "row-level security is enabled on `{table}`; COPY is not yet supported under RLS, so it \
             is allowed only for a superuser"
        )),
        Err(_) => Some("could not verify row-level security for COPY".to_owned()),
    }
}

/// Bound on the number of result frames buffered in flight between the blocking executor thread and
/// the async socket writer (Phase 2). A full channel back-pressures the executor (its
/// `blocking_send` parks) until the writer drains, so a large result set never piles up in memory.
const ROW_STREAM_CHANNEL_CAP: usize = 16;

/// Messages buffered in the [`ChannelSink`] before a chunk crosses the channel: one
/// cross-thread send per chunk instead of per message, and a result smaller than one chunk
/// crosses ZERO times — it returns with the outcome. In-flight bound stays
/// `ROW_STREAM_CHANNEL_CAP × SINK_CHUNK` messages (~2048, the old per-message cap's order).
const SINK_CHUNK: usize = 128;

/// The `RowDescriptionTyped` type tag for column type `ty`: the protocol-1.2 element-typed array tag
/// when `array_elements` (negotiated `minor >= 2`), else the plain tag (an `ARRAY` stays `0x0F`).
const fn type_tag_for(ty: nusadb_core::ColumnType, array_elements: bool) -> u8 {
    if array_elements {
        crate::column_type_tag_v2(ty)
    } else {
        crate::column_type_tag(ty)
    }
}

/// A [`RowSink`] that forwards a statement's output rows to the async socket writer as backend
/// A streamed statement's outcome: the `CommandComplete` tag (or error), the buffered message
/// tail (see [`ChannelSink::into_tail`]), the connection's next transaction state, and the
/// moved-back plan cache.
type StreamedOutcome = (
    Result<String, (String, &'static str)>,
    Vec<BackendMessage>,
    TxnState,
    nusadb_sql::PlanCache,
);

/// The result of [`run_query_streaming`] under an optional inline plan-shape gate:
/// either the statement ran (`Done`), or the gate refused it BEFORE execution and the caller
/// must re-dispatch to the blocking pool (`Punt` — guaranteed side-effect-free: the plan is
/// cached but no row was read, no state changed, and any probe transaction was rolled back).
enum StreamedRun {
    /// The statement ran to an outcome (success or error) — the ordinary result.
    Done(StreamedOutcome),
    /// The plan-shape gate refused inline execution; re-dispatch to the pool. Carries the
    /// moved-in plan cache back (now holding the statement's plan, so the pool re-plan is a
    /// cache hit).
    Punt(nusadb_sql::PlanCache),
}

/// frames over a bounded channel (Phase 2 streaming output). Runs on the `spawn_blocking`
/// executor thread, so it uses `blocking_send`; a closed receiver (the writer stopped — client gone
/// or a socket write failed) surfaces as an error that aborts the statement.
/// How a [`ChannelSink`]'s overflow chunks travel. `Pool` is the ordinary blocking-thread
/// path; `Inline` backs the reactor-inline statements, whose gate guarantees at
/// most one output row — a flush there is an internal-invariant breach reported loudly (a
/// `blocking_send` on the reactor would panic the runtime instead).
enum SinkTx {
    Pool(mpsc::Sender<Vec<BackendMessage>>),
    Inline,
}

struct ChannelSink {
    /// Messages buffered until [`SINK_CHUNK`] is reached (then flushed as one channel send) or
    /// the statement ends (then returned to the async side with the outcome).
    buf: Vec<BackendMessage>,
    tx: SinkTx,
    /// When the connection negotiated protocol `minor >= 1`, emit the typed `RowDescriptionTyped`
    /// Otherwise the classic names-only `RowDescription` (byte-identical to 1.0).
    typed: bool,
    /// When the connection negotiated protocol `minor >= 2`, an `ARRAY` column's tag carries its
    /// element type (protocol 1.2). Implies `typed`.
    array_elements: bool,
}

impl ChannelSink {
    fn send(&mut self, msg: BackendMessage) -> Result<(), nusadb_sql::Error> {
        self.buf.push(msg);
        if self.buf.len() >= SINK_CHUNK {
            match &self.tx {
                SinkTx::Pool(tx) => tx
                    .blocking_send(std::mem::replace(
                        &mut self.buf,
                        Vec::with_capacity(SINK_CHUNK),
                    ))
                    .map_err(|_| {
                        nusadb_sql::Error::Unsupported("client connection closed".to_owned())
                    })?,
                SinkTx::Inline => {
                    return Err(nusadb_sql::Error::Unsupported(
                        "internal: inline statement exceeded its buffered output".to_owned(),
                    ));
                },
            }
        }
        Ok(())
    }

    /// The buffered tail — everything since the last chunk flush — handed back to the async
    /// side with the statement outcome, so a small result never crosses the channel at all.
    fn into_tail(self) -> Vec<BackendMessage> {
        self.buf
    }
}

impl RowSink for ChannelSink {
    fn columns(&mut self, columns: &[String]) -> Result<(), nusadb_sql::Error> {
        self.send(BackendMessage::RowDescription {
            columns: columns.to_vec(),
        })
    }

    fn columns_typed(
        &mut self,
        names: &[String],
        types: &[nusadb_core::ColumnType],
    ) -> Result<(), nusadb_sql::Error> {
        if self.typed && names.len() == types.len() {
            let columns = names
                .iter()
                .zip(types)
                .map(|(name, ty)| (name.clone(), type_tag_for(*ty, self.array_elements)))
                .collect();
            self.send(BackendMessage::RowDescriptionTyped { columns })
        } else {
            self.columns(names)
        }
    }

    fn row(&mut self, row: &[Value]) -> Result<(), nusadb_sql::Error> {
        let values = row.iter().map(|v| value_to_field(v.clone())).collect();
        self.send(BackendMessage::DataRow { values })
    }
}

/// The `CommandComplete` tag for a streamed statement: `SELECT n` for a row result (the rows already
/// went to the sink), otherwise the ordinary [`command_tag`].
fn stream_command_tag(outcome: &StreamOutcome) -> String {
    match outcome {
        StreamOutcome::Rows { count, .. } => format!("SELECT {count}"),
        StreamOutcome::Other(result) => command_tag(result),
    }
}

/// Run one statement against `engine` as `user`, streaming any output rows into `tx` as backend
/// frames (Phase 2). Mirrors [`run_query`]'s one-transaction-per-statement discipline (analyze
/// and execute share a snapshot, then auto-commit or roll back). Returns the `CommandComplete` tag on
/// success. Runs on a blocking thread; `tx` is dropped on return, closing the channel.
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection context (engine, cluster, database, statement text, params, user, the \
              row channel, txn state, the typed flag, and the move-in/move-out plan cache)"
)]
fn run_query_streaming(
    engine: &dyn StorageEngine,
    cluster: &dyn DatabaseCluster,
    database: &str,
    sql: &str,
    stmt: nusadb_sql::ast::Statement,
    user: &str,
    tx: SinkTx,
    state: TxnState,
    typed: bool,
    array_elements: bool,
    mut plan_cache: nusadb_sql::PlanCache,
    settings: &std::sync::Mutex<HashMap<String, String>>,
    point_get_gate: bool,
) -> StreamedRun {
    use nusadb_sql::ast::Statement;
    // Parsed (and parameter-bound) by the caller on the reactor — a parse error never reaches
    // this function.
    let mut stmt = stmt;
    // CREATE/DROP DATABASE are cluster operations handled by the server, not the engine (DB3/DB4).
    // They produce a command tag (no rows), like a transaction-control statement.
    if let Some(result) =
        intercept_database_stmt(cluster, database, in_transaction_block(&state), &stmt)
    {
        return StreamedRun::Done((
            result
                .map(|r| command_tag(&r))
                .map_err(|e| (e.to_string(), e.sqlstate())),
            Vec::new(),
            state,
            plan_cache,
        ));
    }
    // `SELECT ... FROM nusadb_databases` lists the cluster (cross-engine wire-layer state); rewrite
    // it to an inline VALUES relation so the executor streams it with full SQL semantics.
    rewrite_database_catalog(&mut stmt, cluster);
    // Transaction-control statements produce no rows; run them through the same state machine the
    // buffered (extended-query) path uses, then report the tag. The channel sender is dropped
    // unused, so the connection side immediately sees an empty row stream.
    let control = match &stmt {
        Statement::BeginTransaction(ts) => Some(begin_txn(engine, state, settings, ts)),
        Statement::Commit => Some(commit_txn(engine, state)),
        Statement::Rollback => Some(rollback_txn(engine, state)),
        // `SET [SESSION CHARACTERISTICS AS] TRANSACTION ...` (P-ISOLATION): session default in
        // autocommit, re-begin in an untouched transaction, refused after any query.
        Statement::SetTransaction(ts) => Some(set_transaction_txn(engine, settings, ts, state)),
        savepoint @ (Statement::Savepoint(_)
        | Statement::RollbackToSavepoint(_)
        | Statement::ReleaseSavepoint(_)) => Some(savepoint_txn(engine, savepoint, state)),
        _ => None,
    };
    if let Some((result, new_state)) = control {
        return StreamedRun::Done((
            result
                .map(|r| command_tag(&r))
                .map_err(|e| (e.to_string(), e.sqlstate())),
            Vec::new(),
            new_state,
            plan_cache,
        ));
    }
    // Session-variable control over-wire: handled against this connection's GUC store.
    match stmt {
        Statement::SetVariable(sv) => {
            let result = apply_set_variable(settings, sv);
            return StreamedRun::Done((
                result
                    .map(|r| command_tag(&r))
                    .map_err(|e| (e.to_string(), e.sqlstate())),
                Vec::new(),
                state,
                plan_cache,
            ));
        },
        // `SHOW name` produces one row; render it straight into the sink (the executor has no
        // per-connection session to run it), then report the `SHOW` tag.
        Statement::Show(name) => {
            let (result, tail) =
                run_show_streaming(&name, settings, state, tx, typed, array_elements);
            return StreamedRun::Done((result, tail, state, plan_cache));
        },
        _ => {},
    }

    // A row/DML statement: stream it within the connection's transaction context. The snapshot
    // carries the connection database for `CURRENT_DATABASE()` and the effective
    // transaction isolation.
    let mut snapshot = settings_snapshot(settings, database);
    stamp_transaction_isolation(&mut snapshot, state);
    let mut sink = ChannelSink {
        buf: Vec::with_capacity(SINK_CHUNK),
        tx,
        typed,
        array_elements,
    };
    let run = stream_stmt_in_state(
        engine,
        stmt,
        sql,
        user,
        state,
        &mut sink,
        &mut plan_cache,
        &snapshot,
        point_get_gate,
    );
    let (outcome, new_state) = match run {
        StmtRun::Done(outcome, new_state) => (outcome, new_state),
        // The gate refused before execution: the sink is untouched (nothing buffered), state
        // is unchanged, and the plan is now cached for the pool's re-plan.
        StmtRun::Punt => return StreamedRun::Punt(plan_cache),
    };
    StreamedRun::Done((
        outcome
            .map(|o| stream_command_tag(&o))
            .map_err(|e| (e.to_string(), e.sqlstate())),
        sink.into_tail(),
        new_state,
        plan_cache,
    ))
}

/// `SHOW name` over the streaming path: render the one-row result into a fresh sink and hand
/// back its buffered tail with the `SHOW` tag (the body of `run_query_streaming`'s Show arm).
fn run_show_streaming(
    name: &str,
    settings: &std::sync::Mutex<HashMap<String, String>>,
    state: TxnState,
    tx: SinkTx,
    typed: bool,
    array_elements: bool,
) -> (Result<String, (String, &'static str)>, Vec<BackendMessage>) {
    let mut snapshot = settings.lock().map(|s| s.clone()).unwrap_or_default();
    stamp_transaction_isolation(&mut snapshot, state);
    let mut sink = ChannelSink {
        buf: Vec::with_capacity(SINK_CHUNK),
        tx,
        typed,
        array_elements,
    };
    let pushed = show_result(name, &snapshot).and_then(|r| push_result_rows(r, &mut sink));
    (
        pushed
            .map(|()| "SHOW".to_owned())
            .map_err(|e| (e.to_string(), e.sqlstate())),
        sink.into_tail(),
    )
}

/// Push a server-rendered row result (`SHOW ...`) into the streaming sink; any other
/// [`ExecutionResult`] carries no rows and is a no-op.
fn push_result_rows(
    result: ExecutionResult,
    sink: &mut ChannelSink,
) -> Result<(), nusadb_sql::Error> {
    let ExecutionResult::Rows { columns, rows } = result else {
        return Ok(());
    };
    sink.columns(&columns)?;
    for row in &rows {
        sink.row(row)?;
    }
    Ok(())
}

/// Streaming counterpart of [`run_stmt_in_state`]: rejected in a failed transaction, streamed into
/// the open transaction (no commit, abort-to-`Failed` on error), or auto-committed in `Auto`.
#[allow(
    clippy::too_many_arguments,
    reason = "threads engine, statement, statement text, user, txn state, the row sink, the plan \
              cache, and the per-connection GUC snapshot — each a distinct per-statement concern"
)]
/// The result of [`stream_stmt_in_state`]: the statement ran (`Done`), or the inline point-get
/// plan-shape gate refused it BEFORE any execution side effect (`Punt` — the connection state
/// is unchanged and any auto-commit probe transaction was rolled back).
enum StmtRun {
    /// The statement executed; the ordinary (outcome, next-state) pair.
    Done(Result<StreamOutcome, nusadb_sql::Error>, TxnState),
    /// The plan-shape gate refused inline execution; nothing ran.
    Punt,
}

#[allow(
    clippy::too_many_arguments,
    reason = "threads the per-statement transaction context plus the inline point-get gate flag;               each is a distinct concern"
)]
fn stream_stmt_in_state(
    engine: &dyn StorageEngine,
    stmt: nusadb_sql::ast::Statement,
    sql: &str,
    user: &str,
    state: TxnState,
    sink: &mut ChannelSink,
    plan_cache: &mut nusadb_sql::PlanCache,
    snapshot: &HashMap<String, String>,
    point_get_gate: bool,
) -> StmtRun {
    match state {
        TxnState::Failed { .. } => StmtRun::Done(
            Err(nusadb_sql::Error::Unsupported(
                "current transaction is aborted, commands ignored until end of transaction block"
                    .to_owned(),
            )),
            state,
        ),
        TxnState::Active { txn, isolation, .. } => {
            // The RC/RU statement snapshot is refreshed at the execution choke-point
            // (`execute_in_txn_as_streaming_with_settings` → `execute_in_txn` / `stream_select_rows`),
            // identically for the simple- and extended-query paths, so no protocol can drift onto a
            // stale snapshot. Planning below reads the catalog under the
            // transaction's current view, matching the simple-query path which also plans pre-refresh.
            let physical = match nusadb_sql::plan_cached(
                plan_cache,
                sql,
                stmt,
                &EngineCatalog::new(engine, txn, user, snapshot),
                engine,
            ) {
                Ok(physical) => physical,
                // A plan error is reported identically on either path — no reason to punt.
                Err(e) => return StmtRun::Done(Err(e), TxnState::Failed { txn, isolation }),
            };
            // Inline point-get gate: admit only a bounded unique-key point lookup onto the reactor. The
            // check runs where the plan already exists, so an admitted statement pays ZERO
            // extra work over the pool path. Nothing has executed yet — punting leaves the
            // open transaction exactly as it was.
            if point_get_gate && !nusadb_sql::plan_is_inline_point_get(&physical) {
                return StmtRun::Punt;
            }
            let outcome = execute_in_txn_as_streaming_with_settings(
                physical, engine, txn, user, snapshot, sink,
            );
            match outcome {
                Ok(o) => StmtRun::Done(
                    Ok(o),
                    TxnState::Active {
                        txn,
                        dirty: true,
                        isolation,
                    },
                ),
                Err(e) => StmtRun::Done(Err(e), TxnState::Failed { txn, isolation }),
            }
        },
        TxnState::Auto => {
            let txn = match engine.begin(session_isolation(snapshot)) {
                Ok(txn) => txn,
                Err(e) => return StmtRun::Done(Err(e.into()), TxnState::Auto),
            };
            let physical = match nusadb_sql::plan_cached(
                plan_cache,
                sql,
                stmt,
                &EngineCatalog::new(engine, txn, user, snapshot),
                engine,
            ) {
                Ok(physical) => physical,
                Err(e) => {
                    let _ = engine.rollback(txn);
                    return StmtRun::Done(Err(e), TxnState::Auto);
                },
            };
            // Inline point-get (see the Active arm): plan-only so far — roll the probe transaction
            // back and punt with no visible effect.
            if point_get_gate && !nusadb_sql::plan_is_inline_point_get(&physical) {
                let _ = engine.rollback(txn);
                return StmtRun::Punt;
            }
            let outcome = execute_in_txn_as_streaming_with_settings(
                physical, engine, txn, user, snapshot, sink,
            );
            match outcome {
                Ok(o) => match engine.commit(txn) {
                    Ok(()) => StmtRun::Done(Ok(o), TxnState::Auto),
                    Err(e) => {
                        let _ = engine.rollback(txn);
                        StmtRun::Done(Err(e.into()), TxnState::Auto)
                    },
                },
                Err(err) => {
                    let _ = engine.rollback(txn);
                    StmtRun::Done(Err(err), TxnState::Auto)
                },
            }
        },
    }
}

/// Execute `sql` and stream its result frames to `conn` as they are produced (simple
/// query path). The statement runs on a blocking thread feeding a bounded channel; this side writes
/// each `RowDescription`/`DataRow` to the socket as it arrives, then a final `CommandComplete` (on
/// success) or `Error` (on failure). Returns whether the statement succeeded, for metrics.
///
/// A statement-timeout timer trips the cancel token exactly as [`run_blocking`] does. A socket write
/// failure mid-stream aborts: the channel is dropped (unblocking the producer) and the error is
/// returned to the connection loop.
///
/// # Errors
/// Propagates socket write/encode errors.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "a streaming simple-query handler threads connection, engine, statement, user, \
              cancellation, timeout, txn state, and the typed-RowDescription flag — each a distinct \
              concern not worth bundling into a parameter struct; the length is the linear \
              parse → inline-gate → pool-dispatch → frame-pump protocol script"
)]
async fn stream_query_to_conn<S>(
    conn: &mut Connection<S>,
    engine: &Arc<dyn StorageEngine>,
    cluster: &Arc<dyn DatabaseCluster>,
    database: &str,
    sql: String,
    // Bound `$n` parameters. Empty for the simple-query path; the extended-query (portal) path passes
    // the values `Bind` supplied. `bind_parameters` accepts the wire-format bytes directly.
    params: &[Option<Vec<u8>>],
    user: String,
    cancel: nusadb_sql::cancel::CancelToken,
    statement_timeout: Option<Duration>,
    state: TxnState,
    typed: bool,
    array_elements: bool,
    mut plan_cache: nusadb_sql::PlanCache,
    settings: &Arc<std::sync::Mutex<HashMap<String, String>>>,
    // Extended-query `Execute` must NOT repeat `RowDescription` (that is `Describe`'s job) — set for
    // the portal path so the streamed frames carry only `DataRow`s + `CommandComplete`.
    suppress_row_description: bool,
) -> io::Result<(bool, TxnState, nusadb_sql::PlanCache)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // Reset the per-connection cancel flag for this statement (matching `run_blocking`): a cancel
    // that arrived between statements must not abort this one.
    cancel.store(false, Ordering::Relaxed);

    // Parse on the reactor (pure CPU, microseconds) — the parse error path writes its response
    // here exactly as the blocking path used to after the join, and the parsed statement drives
    // the inline gate below.
    let stmt = match parse(&sql).and_then(|s| bind_parameters(s, params)) {
        Ok(stmt) => stmt,
        Err(e) => {
            conn.write_frame(&error_response_coded(&e.to_string(), e.sqlstate()).encode()?)
                .await?;
            return Ok((false, state, plan_cache));
        },
    };

    // Reactor-inline statements skip the measured ~36us spawn_blocking dispatch+join hop:
    // FROM-less pure SELECTs (bounded CPU over one synthesized row) and unique-key
    // point lookups (at most one row off a few B-tree pages — the descent may
    // fault pages in from disk on the reactor, the accepted trade for skipping the hop). Both
    // gates are default-deny. A point-get candidate is admitted INSIDE the execution path, where
    // its plan exists anyway (`plan_is_inline_point_get` — a unique-index point bound): an
    // admitted statement pays zero extra work over the pool path, and a refused one PUNTS
    // side-effect-free back to the pool below with its plan already cached.
    let mut stmt = stmt;
    let from_less = nusadb_sql::ast::from_less_pure_select(&stmt);
    let point_get = !from_less && nusadb_sql::ast::point_get_candidate(&stmt);
    if from_less || point_get {
        // Keep the parsed statement for the (rare) punt re-dispatch; candidates that don't
        // punt pay one small AST clone.
        let backup = point_get.then(|| stmt.clone());
        let run = {
            let _cancel_guard = nusadb_sql::cancel::scope(Arc::clone(&cancel));
            run_query_streaming(
                engine.as_ref(),
                cluster.as_ref(),
                database,
                &sql,
                stmt,
                &user,
                SinkTx::Inline,
                state,
                typed,
                array_elements,
                plan_cache,
                settings,
                point_get,
            )
        };
        match run {
            StreamedRun::Done((outcome, tail, new_state, new_cache)) => {
                if point_get {
                    INLINE_POINT_GET_RUNS.fetch_add(1, Ordering::Relaxed);
                }
                let ok =
                    write_statement_outcome(conn, outcome, tail, suppress_row_description).await?;
                return Ok((ok, new_state, new_cache));
            },
            StreamedRun::Punt(cache) => {
                plan_cache = cache;
                stmt = match backup {
                    Some(stmt) => stmt,
                    // Unreachable by construction (only the point-get gate punts, and it always
                    // has a backup); re-parse rather than panic if it ever regresses.
                    None => match parse(&sql).and_then(|s| bind_parameters(s, params)) {
                        Ok(stmt) => stmt,
                        Err(e) => {
                            conn.write_frame(
                                &error_response_coded(&e.to_string(), e.sqlstate()).encode()?,
                            )
                            .await?;
                            return Ok((false, state, plan_cache));
                        },
                    },
                };
            },
        }
    }

    let timer = effective_statement_timeout(settings, statement_timeout).map(|deadline| {
        let token = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::time::sleep(deadline).await;
            token.store(true, Ordering::Relaxed);
        })
    });

    let (tx, mut rx) = mpsc::channel::<Vec<BackendMessage>>(ROW_STREAM_CHANNEL_CAP);
    let engine = Arc::clone(engine);
    let cluster = Arc::clone(cluster);
    let database = database.to_owned();
    let settings = Arc::clone(settings);
    let task = tokio::task::spawn_blocking(move || {
        let _cancel_guard = nusadb_sql::cancel::scope(cancel);
        match run_query_streaming(
            engine.as_ref(),
            cluster.as_ref(),
            &database,
            &sql,
            stmt,
            &user,
            SinkTx::Pool(tx),
            state,
            typed,
            array_elements,
            plan_cache,
            &settings,
            false,
        ) {
            StreamedRun::Done(outcome) => outcome,
            // Structurally unreachable: only the point-get gate punts, and the pool path runs
            // with the gate off.
            StreamedRun::Punt(plan_cache) => (
                Err((
                    "internal: the pool path cannot punt".to_owned(),
                    nusadb_sql::Error::Unsupported(String::new()).sqlstate(),
                )),
                Vec::new(),
                state,
                plan_cache,
            ),
        }
    });

    // Write frames to the socket as the executor produces them — one channel hop per
    // SINK_CHUNK-message chunk, bounded by the channel capacity.
    let mut write_err: Option<io::Error> = None;
    'chunks: while let Some(chunk) = rx.recv().await {
        for msg in chunk {
            // The portal path suppresses `RowDescription` — `Describe` already sent it.
            if suppress_row_description
                && matches!(
                    msg,
                    BackendMessage::RowDescription { .. }
                        | BackendMessage::RowDescriptionTyped { .. }
                )
            {
                continue;
            }
            let frame = match msg.encode() {
                Ok(frame) => frame,
                Err(e) => {
                    write_err = Some(io::Error::other(format!("frame encode failed: {e}")));
                    break 'chunks;
                },
            };
            if let Err(e) = conn.write_frame(&frame).await {
                write_err = Some(e);
                break 'chunks;
            }
        }
    }
    // Drop the receiver so a producer parked on a full channel (or still running after a write
    // error) unblocks, then collect its outcome.
    drop(rx);
    let (outcome, tail, new_state, new_cache) = task.await.unwrap_or_else(|_join| {
        // The blocking task panicked: its moved-in plan cache is gone, so resume with an empty one
        // (cold but correct). A panic mid-statement is catastrophic regardless.
        (
            Err(("internal execution error".to_owned(), "XX000")),
            Vec::new(),
            state,
            nusadb_sql::PlanCache::new(),
        )
    });
    if let Some(timer) = timer {
        timer.abort();
    }

    if let Some(err) = write_err {
        return Err(err);
    }
    let ok = write_statement_outcome(conn, outcome, tail, suppress_row_description).await?;
    Ok((ok, new_state, new_cache))
}

/// Write a streamed statement's buffered tail (strictly after every flushed chunk) and its
/// closing `CommandComplete`/`Error` frame; returns whether the statement succeeded. Shared by
/// the blocking-pool path and the reactor-inline path.
async fn write_statement_outcome<S>(
    conn: &mut Connection<S>,
    outcome: Result<String, (String, &'static str)>,
    tail: Vec<BackendMessage>,
    // The portal path suppresses `RowDescription` in the buffered tail too (see the reactor loop).
    suppress_row_description: bool,
) -> io::Result<bool>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    for msg in tail {
        if suppress_row_description
            && matches!(
                msg,
                BackendMessage::RowDescription { .. } | BackendMessage::RowDescriptionTyped { .. }
            )
        {
            continue;
        }
        conn.write_frame(&msg.encode()?).await?;
    }
    match outcome {
        Ok(tag) => {
            conn.write_frame(&command_complete(&tag).encode()?).await?;
            Ok(true)
        },
        Err((message, code)) => {
            conn.write_frame(&error_response_coded(&message, code).encode()?)
                .await?;
            Ok(false)
        },
    }
}

/// The wire connection's transaction state across statements (transaction-over-wire).
///
/// `Auto` means every statement is its own auto-committed transaction (the historical behaviour);
/// `Active`/`Failed` hold the explicit transaction opened by `BEGIN` until `COMMIT`/`ROLLBACK`.
/// `Failed` is an errored transaction that, per standard SQL semantics, only `COMMIT`/`ROLLBACK` can
/// leave (intervening statements are rejected).
#[derive(Clone, Copy, Debug)]
enum TxnState {
    Auto,
    Active {
        txn: TxnId,
        /// Whether any statement (or savepoint operation) has run inside this transaction. The
        /// engine fixes isolation at `begin`, so `SET TRANSACTION ISOLATION LEVEL` is honored by
        /// re-beginning the transaction — observably equivalent **only** while it is untouched
        /// (the reference engine likewise requires it "before any query"); afterwards it is refused (P-ISOLATION).
        dirty: bool,
        /// The level the engine transaction was begun with, carried so `SHOW
        /// transaction_isolation` / `current_setting` report the level actually enforced,
        /// not the session default.
        isolation: IsolationLevel,
    },
    Failed {
        txn: TxnId,
        /// Kept through failure so a `ROLLBACK TO SAVEPOINT` recovery (back to `Active`) and
        /// introspection still know the transaction's level.
        isolation: IsolationLevel,
    },
}

impl TxnState {
    /// The `ReadyForQuery` status byte (`I`/`T`/`E`) for this state.
    const fn status(self) -> TxnStatus {
        match self {
            Self::Auto => TxnStatus::Idle,
            Self::Active { .. } => TxnStatus::InTransaction,
            Self::Failed { .. } => TxnStatus::Failed,
        }
    }

    /// The open transaction (if any) — used to roll back at connection teardown.
    const fn open_txn(self) -> Option<TxnId> {
        match self {
            Self::Auto => None,
            Self::Active { txn, .. } | Self::Failed { txn, .. } => Some(txn),
        }
    }

    /// How a `NOTIFY` issued in this state is handled (transactional NOTIFY): delivered eagerly in
    /// `Auto`, queued in an active transaction, rejected in a failed one.
    const fn notify_phase(self) -> NotifyPhase {
        match self {
            Self::Auto => NotifyPhase::Autocommit,
            Self::Active { .. } => NotifyPhase::InTransaction,
            Self::Failed { .. } => NotifyPhase::Aborted,
        }
    }
}

/// Stamp the connection's **effective** transaction isolation into a GUC snapshot:
/// inside a transaction block, the level the engine transaction was begun
/// with; in autocommit, the session default the next statement's transaction will use. Written
/// after the store copy, so it always wins — `SHOW transaction_isolation` / `current_setting`
/// cannot misreport the enforced level.
fn stamp_transaction_isolation(snapshot: &mut HashMap<String, String>, state: TxnState) {
    let level = match state {
        TxnState::Active { isolation, .. } | TxnState::Failed { isolation, .. } => isolation,
        TxnState::Auto => session_isolation(snapshot),
    };
    snapshot.insert(
        "transaction_isolation".to_owned(),
        isolation_guc_text(level).to_owned(),
    );
}

/// The isolation level the connection's next transaction begins with (P-ISOLATION): an explicit
/// `SET default_transaction_isolation` / `SET [SESSION CHARACTERISTICS AS] TRANSACTION ISOLATION
/// LEVEL` recorded in the GUC store wins; otherwise the engine default. An unparseable stored
/// value cannot happen ([`apply_set_variable`] validates the GUC on `SET`), but fail safe to the
/// default anyway.
fn session_isolation(snapshot: &HashMap<String, String>) -> IsolationLevel {
    snapshot
        .get("default_transaction_isolation")
        .and_then(|text| parse_isolation_guc(text))
        .unwrap_or_default()
}

/// The engine-level isolation for a parsed `BEGIN`/`SET TRANSACTION` level (the AST keeps its own
/// enum so the parser does not depend on `nusadb-core` types).
const fn core_isolation(level: nusadb_sql::ast::IsolationLevel) -> IsolationLevel {
    match level {
        nusadb_sql::ast::IsolationLevel::ReadUncommitted => IsolationLevel::ReadUncommitted,
        nusadb_sql::ast::IsolationLevel::ReadCommitted => IsolationLevel::ReadCommitted,
        nusadb_sql::ast::IsolationLevel::RepeatableRead => IsolationLevel::RepeatableRead,
        nusadb_sql::ast::IsolationLevel::Serializable => IsolationLevel::Serializable,
    }
}

/// Parse the GUC spelling of an isolation level (case-insensitive, the reference engine's forms).
fn parse_isolation_guc(text: &str) -> Option<IsolationLevel> {
    match text.trim().to_ascii_lowercase().as_str() {
        "read uncommitted" => Some(IsolationLevel::ReadUncommitted),
        "read committed" => Some(IsolationLevel::ReadCommitted),
        "repeatable read" => Some(IsolationLevel::RepeatableRead),
        "serializable" => Some(IsolationLevel::Serializable),
        _ => None,
    }
}

/// The GUC spelling of an isolation level (what `SHOW`/`current_setting` report).
const fn isolation_guc_text(level: IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "read uncommitted",
        IsolationLevel::ReadCommitted => "read committed",
        IsolationLevel::RepeatableRead => "repeatable read",
        IsolationLevel::Serializable => "serializable",
    }
}

/// `SET [SESSION CHARACTERISTICS AS] TRANSACTION ...` over the wire (P-ISOLATION).
///
/// - In autocommit: records the isolation as the connection's `default_transaction_isolation`
///   GUC, so every later `BEGIN` / auto-committed statement begins at that level (both spellings
///   land here — the session default — matching the embedded `Session`; a documented deviation
///   from the reference engine's txn-scoped plain `SET TRANSACTION`, which is a no-op warning outside a block).
/// - In an **untouched** transaction: re-begins the engine transaction at the requested level
///   (observably equivalent — nothing has run; the reference engine likewise requires this "before any query").
///   The session default is left alone, so a later transaction reverts, like the reference engine.
/// - After the transaction has run a statement: refused with SQLSTATE `25001` and the
///   transaction aborts (an error inside a block aborts it, like any other).
/// - `READ ONLY` is refused loudly — the wire layer does not enforce access modes yet, and
///   silently granting a writable "read-only" transaction would be worse than an error.
fn set_transaction_txn(
    engine: &dyn StorageEngine,
    settings: &std::sync::Mutex<HashMap<String, String>>,
    ts: &nusadb_sql::ast::TransactionSettings,
    state: TxnState,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    if matches!(ts.access_mode, Some(nusadb_sql::ast::AccessMode::ReadOnly)) {
        let err = nusadb_sql::Error::Unsupported(
            "READ ONLY transactions are not supported over the wire protocol yet".to_owned(),
        );
        let new_state = match state {
            TxnState::Active { txn, isolation, .. } => TxnState::Failed { txn, isolation },
            other => other,
        };
        return (Err(err), new_state);
    }
    match state {
        TxnState::Failed { .. } => (
            Err(nusadb_sql::Error::Unsupported(
                "current transaction is aborted, commands ignored until end of transaction block"
                    .to_owned(),
            )),
            state,
        ),
        TxnState::Active {
            txn,
            dirty: true,
            isolation,
        } => (
            Err(nusadb_sql::Error::Coded {
                message: "SET TRANSACTION ISOLATION LEVEL must be called before any query"
                    .to_owned(),
                sqlstate: "25001", // active_sql_transaction
            }),
            TxnState::Failed { txn, isolation },
        ),
        TxnState::Active {
            txn,
            dirty: false,
            isolation,
        } => {
            // Nothing ran yet: an engine transaction at the new level is indistinguishable from
            // this one, so swap them. Without an isolation change there is nothing to do.
            let Some(level) = ts.isolation else {
                return (
                    Ok(ExecutionResult::TransactionCharacteristicsSet),
                    TxnState::Active {
                        txn,
                        dirty: false,
                        isolation,
                    },
                );
            };
            let _ = engine.rollback(txn);
            let level = core_isolation(level);
            match engine.begin(level) {
                Ok(new_txn) => (
                    Ok(ExecutionResult::TransactionCharacteristicsSet),
                    TxnState::Active {
                        txn: new_txn,
                        dirty: false,
                        isolation: level,
                    },
                ),
                Err(e) => (Err(e.into()), TxnState::Auto),
            }
        },
        TxnState::Auto => {
            if let Some(level) = ts.isolation
                && let Ok(mut store) = settings.lock()
            {
                store.insert(
                    "default_transaction_isolation".to_owned(),
                    isolation_guc_text(core_isolation(level)).to_owned(),
                );
            }
            (
                Ok(ExecutionResult::TransactionCharacteristicsSet),
                TxnState::Auto,
            )
        },
    }
}

/// How the current transaction state treats a `NOTIFY` (see [`TxnState::notify_phase`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NotifyPhase {
    /// No explicit transaction: deliver immediately.
    Autocommit,
    /// Inside an active transaction: queue until COMMIT.
    InTransaction,
    /// Inside a failed transaction: reject (commands are ignored until the block ends).
    Aborted,
}

/// A transaction-control statement, classified from its SQL so the connection loop can apply its
/// effect on the queued `NOTIFY`s ([`PendingNotifications`]) after the statement runs.
enum TxnControl {
    Begin,
    Commit,
    Rollback,
    Savepoint(String),
    Release(String),
    RollbackTo(String),
}

/// Classify `sql` as a transaction-control statement, or `None` for anything else. A cheap first-word
/// guard avoids parsing every simple query twice; only a `BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT`/
/// `RELEASE`/`START`/`END`/`ABORT` prefix reaches the parse.
fn txn_control_kind(sql: &str) -> Option<TxnControl> {
    use nusadb_sql::ast::Statement;
    let head = sql.trim_start();
    let looks = [
        "begin",
        "start",
        "commit",
        "end",
        "rollback",
        "abort",
        "savepoint",
        "release",
    ]
    .iter()
    .any(|kw| {
        head.get(..kw.len())
            .is_some_and(|p| p.eq_ignore_ascii_case(kw))
    });
    if !looks {
        return None;
    }
    match parse(sql).ok()? {
        Statement::BeginTransaction(_) => Some(TxnControl::Begin),
        Statement::Commit => Some(TxnControl::Commit),
        Statement::Rollback => Some(TxnControl::Rollback),
        Statement::Savepoint(name) => Some(TxnControl::Savepoint(name)),
        Statement::ReleaseSavepoint(name) => Some(TxnControl::Release(name)),
        Statement::RollbackToSavepoint(name) => Some(TxnControl::RollbackTo(name)),
        _ => None,
    }
}

/// A connection's queue of `NOTIFY`s issued inside the current explicit transaction (transactional
/// NOTIFY, like the reference engine). Notifications are delivered only on `COMMIT` ([`flush`](Self::flush)) and
/// discarded on `ROLLBACK` ([`discard`](Self::discard)). Savepoint marks let `ROLLBACK TO SAVEPOINT`
/// discard only the notifications queued *after* that savepoint, mirroring the reference engine.
#[derive(Default)]
struct PendingNotifications {
    /// Queued `(channel, payload)` in issue order.
    queue: Vec<(String, String)>,
    /// `(savepoint name, queue length when the savepoint was established)`, innermost last.
    marks: Vec<(String, usize)>,
}

impl PendingNotifications {
    /// Queue a `NOTIFY` issued in the current transaction.
    fn enqueue(&mut self, channel: String, payload: String) {
        self.queue.push((channel, payload));
    }

    /// Record a `SAVEPOINT`, remembering the queue length so a later `ROLLBACK TO` can trim back to it.
    fn savepoint(&mut self, name: String) {
        self.marks.push((name, self.queue.len()));
    }

    /// `RELEASE SAVEPOINT name`: drop the named savepoint and any established after it; the queued
    /// notifications stay (they merge into the enclosing transaction, like the reference engine).
    fn release(&mut self, name: &str) {
        if let Some(pos) = self.marks.iter().rposition(|(n, _)| n == name) {
            self.marks.truncate(pos);
        }
    }

    /// `ROLLBACK TO SAVEPOINT name`: discard notifications queued after the savepoint; keep the
    /// savepoint itself (it can be rolled back to again) and drop any established after it.
    fn rollback_to(&mut self, name: &str) {
        if let Some(pos) = self.marks.iter().rposition(|(n, _)| n == name)
            && let Some(&(_, len)) = self.marks.get(pos)
        {
            self.queue.truncate(len);
            self.marks.truncate(pos + 1);
        }
    }

    /// Discard the whole queue (`ROLLBACK`, or a fresh `BEGIN`).
    fn discard(&mut self) {
        self.queue.clear();
        self.marks.clear();
    }

    /// Deliver the queued notifications to their listeners and clear the queue (`COMMIT`).
    /// Identical `(channel, payload)` pairs are collapsed to a single delivery, like the reference engine.
    fn flush(&mut self, pid: u32, database: &str) {
        let mut seen = std::collections::HashSet::new();
        for (channel, payload) in self.queue.drain(..) {
            if seen.insert((channel.clone(), payload.clone())) {
                let notification = notify::Notification {
                    pid,
                    channel: channel.clone(),
                    payload,
                };
                notify::notify(database, &channel, &notification);
            }
        }
        self.marks.clear();
    }
}

/// Run one statement honouring the connection's transaction state, returning the result and the new
/// state. Explicit `BEGIN`/`COMMIT`/`ROLLBACK` open/close the transaction; every other statement
/// runs in the open transaction (no commit) or, in `Auto`, in its own auto-committed transaction.
#[allow(
    clippy::too_many_arguments,
    reason = "per-connection context (engine, cluster, database, user, settings) threaded verbatim"
)]
fn run_query_txn(
    engine: &dyn StorageEngine,
    cluster: &dyn DatabaseCluster,
    database: &str,
    sql: &str,
    params: &[Option<Vec<u8>>],
    user: &str,
    state: TxnState,
    settings: &std::sync::Mutex<HashMap<String, String>>,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    use nusadb_sql::ast::Statement;
    let mut stmt = match parse(sql).and_then(|s| bind_parameters(s, params)) {
        Ok(stmt) => stmt,
        Err(e) => return (Err(e), state),
    };
    // CREATE/DROP DATABASE are cluster operations handled by the server, not the engine (DB3/DB4).
    if let Some(result) =
        intercept_database_stmt(cluster, database, in_transaction_block(&state), &stmt)
    {
        return (result, state);
    }
    // `SELECT ... FROM nusadb_databases` lists the cluster (cross-engine wire-layer state); rewrite
    // it to an inline VALUES relation so the executor serves it with full SQL semantics.
    rewrite_database_catalog(&mut stmt, cluster);
    match stmt {
        Statement::BeginTransaction(ts) => begin_txn(engine, state, settings, &ts),
        Statement::Commit => commit_txn(engine, state),
        Statement::Rollback => rollback_txn(engine, state),
        // `SET [SESSION CHARACTERISTICS AS] TRANSACTION ...` (P-ISOLATION): session default in
        // autocommit, re-begin in an untouched transaction, refused after any query.
        Statement::SetTransaction(ts) => set_transaction_txn(engine, settings, &ts, state),
        savepoint @ (Statement::Savepoint(_)
        | Statement::RollbackToSavepoint(_)
        | Statement::ReleaseSavepoint(_)) => savepoint_txn(engine, &savepoint, state),
        // Session-variable control: handled against this connection's GUC store, not the
        // executor (which has no per-connection session). `SET` records / `RESET` clears; `SHOW` reads
        // back the value (with built-in defaults) — kept consistent with `current_setting`.
        Statement::SetVariable(sv) => (apply_set_variable(settings, sv), state),
        Statement::Show(name) => {
            let mut snapshot = settings.lock().map(|s| s.clone()).unwrap_or_default();
            stamp_transaction_isolation(&mut snapshot, state);
            (show_result(&name, &snapshot), state)
        },
        other => run_stmt_in_state(engine, database, other, user, state, settings),
    }
}

/// Apply `SET name = value` / `RESET name` to a connection's GUC store: `Some` records the
/// value, `None` (RESET) clears it. A poisoned lock is treated as empty (the connection is already
/// being torn down on any panic that would poison it).
///
/// `default_transaction_isolation` is validated on write (P-ISOLATION): the value steers what
/// isolation later transactions actually begin with, so a typo must fail loudly here rather than
/// silently fall back to the default level at `BEGIN`.
fn apply_set_variable(
    settings: &std::sync::Mutex<HashMap<String, String>>,
    sv: nusadb_sql::ast::SetVariable,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    if sv
        .name
        .eq_ignore_ascii_case("default_transaction_isolation")
        && let Some(value) = &sv.value
        && parse_isolation_guc(value).is_none()
    {
        return Err(nusadb_sql::Error::Coded {
            message: format!(
                "invalid value for parameter \"default_transaction_isolation\": {value:?}"
            ),
            sqlstate: "22023", // invalid_parameter_value
        });
    }
    // `work_mem` feeds the executor's statement-effective budget; reject an unparseable value at
    // SET time instead of storing a string the budget check would then fall back past.
    if sv.name.eq_ignore_ascii_case("work_mem")
        && let Some(value) = &sv.value
        && nusadb_sql::parse_work_mem(value).is_none()
    {
        return Err(nusadb_sql::Error::Coded {
            message: format!(
                "invalid value for parameter \"work_mem\": {value:?} — expected an integer with \
                 an optional kB/MB/GB/TB unit (a bare integer is kilobytes; 0 = unlimited)"
            ),
            sqlstate: "22023", // invalid_parameter_value
        });
    }
    // `statement_timeout` arms the per-statement cancel timer; same loud SET-time rejection so a
    // typo cannot silently disable the timeout.
    if sv.name.eq_ignore_ascii_case("statement_timeout")
        && let Some(value) = &sv.value
        && nusadb_sql::cancel::parse_statement_timeout(value).is_none()
    {
        return Err(nusadb_sql::Error::Coded {
            message: format!(
                "invalid value for parameter \"statement_timeout\": {value:?} — expected an \
                 integer with an optional us/ms/s/min/h/d unit (a bare integer is milliseconds; \
                 0 = no timeout)"
            ),
            sqlstate: "22023", // invalid_parameter_value
        });
    }
    if let Ok(mut store) = settings.lock() {
        match sv.value {
            Some(value) => {
                store.insert(sv.name, value);
            },
            None => {
                store.remove(&sv.name);
            },
        }
    }
    Ok(ExecutionResult::VariableSet)
}

/// The name of the wire-level system catalog that lists the cluster's databases, following the
/// engine's own `nusadb_*` catalog convention (`nusadb_functions`/`nusadb_policies`/…). A query
/// `SELECT name FROM nusadb_databases` is the supported way to list the cluster (see
/// [`rewrite_database_catalog`]).
const DATABASES_CATALOG: &str = "nusadb_databases";

/// Render `SHOW name`. `SHOW` reads only a configuration parameter (a GUC), so `SHOW DATABASES` —
/// a non-standard listing statement, not a parameter — is rejected loudly here, exactly as an
/// unrecognized parameter is rejected; the cluster's databases are listed instead via
/// `SELECT name FROM nusadb_databases` (see [`rewrite_database_catalog`]). Every other name reads
/// back the connection's session variable, as before.
fn show_result(
    name: &str,
    snapshot: &HashMap<String, String>,
) -> Result<ExecutionResult, nusadb_sql::Error> {
    if name.eq_ignore_ascii_case("databases") {
        return Err(nusadb_sql::Error::Coded {
            message: format!(
                "unrecognized configuration parameter \"{name}\" \
                 (SHOW DATABASES is not supported; use SELECT name FROM {DATABASES_CATALOG})"
            ),
            sqlstate: "42704", // undefined_object
        });
    }
    Ok(show_session_variable(name, snapshot))
}

/// Rewrite a query's reference to the [`DATABASES_CATALOG`] (`nusadb_databases`) system catalog
/// into an inline `(VALUES ...) AS nusadb_databases(name)` derived table sourced from
/// `cluster.list()`.
///
/// The cluster's database list is wire-layer state (cross-database, spanning every engine), not
/// any single engine's catalog, so a `SELECT ... FROM nusadb_databases` cannot resolve through the
/// SQL executor. Rewriting the `FROM` reference to a literal `VALUES` relation before the executor
/// runs gives the query full SQL semantics — projection, `WHERE`, `ORDER BY`, aliases — for free.
/// Rewrites the top-level `SELECT`'s `FROM` base and its joins; returns `true` if anything changed.
fn rewrite_database_catalog(
    stmt: &mut nusadb_sql::ast::Statement,
    cluster: &dyn DatabaseCluster,
) -> bool {
    use nusadb_sql::ast::Statement;
    let Statement::Select(select) = stmt else {
        return false;
    };
    let Some(from) = select.from.as_mut() else {
        return false;
    };
    let mut rewrote = rewrite_databases_table_ref(&mut from.base, cluster);
    for join in &mut from.joins {
        rewrote |= rewrite_databases_table_ref(&mut join.table, cluster);
    }
    rewrote
}

/// Replace a bare `nusadb_databases` table reference with a `VALUES` relation of the cluster's
/// database names (one `name` column). A no-op for anything else (including a derived table that
/// merely shares the name). Never rewrites to an empty `VALUES` (unsupported) — there is always at
/// least the default database, but the guard keeps it correct if the list is somehow empty.
fn rewrite_databases_table_ref(
    table: &mut nusadb_sql::ast::TableRef,
    cluster: &dyn DatabaseCluster,
) -> bool {
    let is_plain_named = table.schema.is_none()
        && table.subquery.is_none()
        && table.values.is_none()
        && table.set_op.is_none()
        && table.name.eq_ignore_ascii_case(DATABASES_CATALOG);
    if !is_plain_named {
        return false;
    }
    let rows: Vec<Vec<nusadb_sql::ast::Expr>> = cluster
        .list()
        .into_iter()
        .map(|db| vec![nusadb_sql::ast::Expr::Literal(Value::Text(db))])
        .collect();
    if rows.is_empty() {
        return false;
    }
    // The alias the query qualifies columns with (`nusadb_databases.name` or the `AS` alias).
    let alias = table
        .alias
        .clone()
        .unwrap_or_else(|| DATABASES_CATALOG.to_owned());
    table.schema = None;
    table.name.clone_from(&alias);
    table.alias = Some(alias);
    table.values = Some(rows);
    table.column_aliases = vec!["name".to_owned()];
    true
}

/// Snapshot the connection's GUC store for a statement, stamping in the connection `database` under
/// the reserved key the session reads for `CURRENT_DATABASE()`. The stamp is written into the
/// fresh clone after copying the store, so it always wins over any value a `SET` may have placed
/// there — a client cannot make `current_database()` lie.
fn settings_snapshot(
    settings: &std::sync::Mutex<HashMap<String, String>>,
    database: &str,
) -> HashMap<String, String> {
    let mut snapshot = settings.lock().map(|s| s.clone()).unwrap_or_default();
    snapshot.insert(
        nusadb_sql::CONNECTION_DATABASE_SETTING.to_owned(),
        database.to_owned(),
    );
    snapshot
}

/// How many statements have run on the reactor-inline point-get path, process-wide.
/// Observability for tests: the integration pins assert it advances across a unique-key lookup
/// round trip, proving the inline path actually fired (anti-vacuous-gate discipline). Tests run
/// in parallel in one process, so pins assert a monotonic increase, never an exact value.
static INLINE_POINT_GET_RUNS: AtomicU64 = AtomicU64::new(0);

/// The current [`INLINE_POINT_GET_RUNS`] reading (see its doc; test observability only).
#[doc(hidden)]
#[must_use]
pub fn inline_point_get_count() -> u64 {
    INLINE_POINT_GET_RUNS.load(Ordering::Relaxed)
}

/// Whether the connection is inside an open transaction block — either active or
/// aborted-but-still-open (`Failed`). A non-transactional statement (`CREATE`/`DROP DATABASE`) must
/// be refused in **both**, so the predicate is "not auto-commit", computed in one place so the two
/// call sites cannot drift (a `Failed`-block miss would let an irreversible drop run mid-transaction).
const fn in_transaction_block(state: &TxnState) -> bool {
    !matches!(state, TxnState::Auto)
}

/// Intercept `CREATE`/`DROP DATABASE` at the wire (model B, DB3/DB4): these are cluster operations a
/// single-engine SQL executor cannot perform, so the server runs them against its [`DatabaseCluster`]
/// — mirroring the `SET`/`SHOW` interception. Returns `Some(result)` when `stmt` is a database
/// statement (the caller skips the executor), `None` otherwise. `connected` is the connection's
/// database (a connection cannot drop the database it is in); `in_transaction` rejects the operation
/// inside a transaction block (active or aborted), since these are non-transactional.
fn intercept_database_stmt(
    cluster: &dyn DatabaseCluster,
    connected: &str,
    in_transaction: bool,
    stmt: &nusadb_sql::ast::Statement,
) -> Option<Result<ExecutionResult, nusadb_sql::Error>> {
    use nusadb_sql::ast::Statement;
    let (verb, name, flag, created) = match stmt {
        Statement::CreateDatabase(cd) => ("CREATE DATABASE", &cd.name, cd.if_not_exists, true),
        Statement::DropDatabase(dd) => ("DROP DATABASE", &dd.name, dd.if_exists, false),
        _ => return None,
    };
    if in_transaction {
        return Some(Err(nusadb_sql::Error::Coded {
            message: format!("{verb} cannot run inside a transaction block"),
            sqlstate: "25001", // active_sql_transaction
        }));
    }
    let outcome = if created {
        cluster
            .create(name, flag)
            .map(|_| ExecutionResult::DatabaseCreated)
    } else {
        cluster
            .drop_database(name, flag, connected)
            .map(|_| ExecutionResult::DatabaseDropped)
    };
    Some(outcome.map_err(|e| nusadb_sql::Error::Coded {
        message: e.to_string(),
        sqlstate: e.sqlstate(),
    }))
}

/// `BEGIN [ISOLATION LEVEL ...]`: open a transaction in `Auto` — at the explicitly requested
/// isolation, falling back to the connection's `default_transaction_isolation` GUC, then the
/// engine default (P-ISOLATION). Otherwise keep the open one (a redundant `BEGIN` is a no-op,
/// not an error; the reference engine ignores its characteristics with a warning). `BEGIN READ ONLY` is refused
/// loudly — the wire layer does not enforce access modes yet, and silently opening a writable
/// "read-only" transaction would be worse than an error.
fn begin_txn(
    engine: &dyn StorageEngine,
    state: TxnState,
    settings: &std::sync::Mutex<HashMap<String, String>>,
    requested: &nusadb_sql::ast::TransactionSettings,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    if matches!(
        requested.access_mode,
        Some(nusadb_sql::ast::AccessMode::ReadOnly)
    ) {
        return (
            Err(nusadb_sql::Error::Unsupported(
                "READ ONLY transactions are not supported over the wire protocol yet".to_owned(),
            )),
            state,
        );
    }
    match state {
        TxnState::Auto => {
            let level = requested.isolation.map_or_else(
                || {
                    let snapshot = settings.lock().map(|s| s.clone()).unwrap_or_default();
                    session_isolation(&snapshot)
                },
                core_isolation,
            );
            match engine.begin(level) {
                Ok(txn) => (
                    Ok(ExecutionResult::TransactionBegun),
                    TxnState::Active {
                        txn,
                        dirty: false,
                        isolation: level,
                    },
                ),
                Err(e) => (Err(e.into()), TxnState::Auto),
            }
        },
        already_open => (Ok(ExecutionResult::TransactionBegun), already_open),
    }
}

/// `COMMIT`: commit an active transaction; roll back an aborted one (standard SQL semantics); no-op
/// in `Auto`.
fn commit_txn(
    engine: &dyn StorageEngine,
    state: TxnState,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    match state {
        TxnState::Active { txn, .. } => match engine.commit(txn) {
            Ok(()) => (Ok(ExecutionResult::TransactionCommitted), TxnState::Auto),
            Err(e) => {
                let _ = engine.rollback(txn);
                (Err(e.into()), TxnState::Auto)
            },
        },
        TxnState::Failed { txn, .. } => {
            let _ = engine.rollback(txn);
            (Ok(ExecutionResult::TransactionRolledBack), TxnState::Auto)
        },
        TxnState::Auto => (Ok(ExecutionResult::TransactionCommitted), TxnState::Auto),
    }
}

/// `ROLLBACK`: roll back any open transaction; no-op in `Auto`.
fn rollback_txn(
    engine: &dyn StorageEngine,
    state: TxnState,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    if let Some(txn) = state.open_txn() {
        let _ = engine.rollback(txn);
    }
    (Ok(ExecutionResult::TransactionRolledBack), TxnState::Auto)
}

/// `SAVEPOINT` / `RELEASE SAVEPOINT` / `ROLLBACK TO SAVEPOINT` against the connection's open
/// transaction. Outside a transaction block (`Auto`) all three error, like the standard. In a failed
/// transaction only `ROLLBACK TO SAVEPOINT` is allowed — and on success it *recovers* the transaction
/// (back to `Active`), undoing the statement that aborted it; the others stay rejected until the block
/// ends. `name` is the savepoint identifier carried by the statement (A-UR.03).
fn savepoint_txn(
    engine: &dyn StorageEngine,
    stmt: &nusadb_sql::ast::Statement,
    state: TxnState,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    use nusadb_sql::ast::Statement;
    let aborted = || {
        nusadb_sql::Error::Unsupported(
            "current transaction is aborted, commands ignored until end of transaction block"
                .to_owned(),
        )
    };
    match state {
        TxnState::Active { txn, isolation, .. } => {
            let result = match stmt {
                Statement::Savepoint(name) => engine
                    .savepoint(txn, name)
                    .map(|()| ExecutionResult::SavepointCreated),
                Statement::ReleaseSavepoint(name) => engine
                    .release_savepoint(txn, name)
                    .map(|()| ExecutionResult::SavepointReleased),
                Statement::RollbackToSavepoint(name) => engine
                    .rollback_to(txn, name)
                    .map(|()| ExecutionResult::RolledBackToSavepoint),
                _ => unreachable!("savepoint_txn called with a non-savepoint statement"),
            };
            // A savepoint pins transaction structure a re-begin could not reproduce, so the
            // transaction no longer accepts `SET TRANSACTION` (dirty), whatever it was before.
            (
                result.map_err(Into::into),
                TxnState::Active {
                    txn,
                    dirty: true,
                    isolation,
                },
            )
        },
        // A failed transaction can be recovered by rolling back to a savepoint taken before the error.
        TxnState::Failed { txn, isolation } => match stmt {
            Statement::RollbackToSavepoint(name) => match engine.rollback_to(txn, name) {
                Ok(()) => (
                    Ok(ExecutionResult::RolledBackToSavepoint),
                    TxnState::Active {
                        txn,
                        dirty: true,
                        isolation,
                    },
                ),
                Err(e) => (Err(e.into()), TxnState::Failed { txn, isolation }),
            },
            _ => (Err(aborted()), TxnState::Failed { txn, isolation }),
        },
        TxnState::Auto => (
            Err(nusadb_sql::Error::Unsupported(
                "SAVEPOINT can only be used in transaction blocks".to_owned(),
            )),
            TxnState::Auto,
        ),
    }
}

/// Run a non-transaction-control statement: rejected in a failed transaction; executed in the open
/// transaction (no commit, abort-to-`Failed` on error); auto-committed in `Auto` (the historical
/// one-transaction-per-statement path — schema resolution and execution share one snapshot,
/// both as the connection `user` so row-level security agrees).
fn run_stmt_in_state(
    engine: &dyn StorageEngine,
    database: &str,
    stmt: nusadb_sql::ast::Statement,
    user: &str,
    state: TxnState,
    settings: &std::sync::Mutex<HashMap<String, String>>,
) -> (Result<ExecutionResult, nusadb_sql::Error>, TxnState) {
    // Snapshot the connection's SET variables so `current_setting(name)` reflects them, with
    // the connection database stamped in for `CURRENT_DATABASE()` and the effective
    // transaction isolation stamped in. Cheap (the map is tiny); cloned so the
    // store lock is not held across execution.
    let mut snapshot = settings_snapshot(settings, database);
    stamp_transaction_isolation(&mut snapshot, state);
    match state {
        TxnState::Failed { .. } => (
            Err(nusadb_sql::Error::Unsupported(
                "current transaction is aborted, commands ignored until end of transaction block"
                    .to_owned(),
            )),
            state,
        ),
        TxnState::Active { txn, isolation, .. } => {
            let outcome = analyze(stmt, &EngineCatalog::new(engine, txn, user, &snapshot))
                .and_then(|logical| {
                    execute_in_txn_as_with_settings(plan(logical), engine, txn, user, &snapshot)
                });
            match outcome {
                Ok(result) => (
                    Ok(result),
                    TxnState::Active {
                        txn,
                        dirty: true,
                        isolation,
                    },
                ),
                // A failed statement aborts the transaction until COMMIT/ROLLBACK.
                Err(e) => (Err(e), TxnState::Failed { txn, isolation }),
            }
        },
        TxnState::Auto => {
            let txn = match engine.begin(session_isolation(&snapshot)) {
                Ok(txn) => txn,
                Err(e) => return (Err(e.into()), TxnState::Auto),
            };
            let outcome = analyze(stmt, &EngineCatalog::new(engine, txn, user, &snapshot))
                .and_then(|logical| {
                    execute_in_txn_as_with_settings(plan(logical), engine, txn, user, &snapshot)
                });
            match outcome {
                Ok(result) => match engine.commit(txn) {
                    Ok(()) => (Ok(result), TxnState::Auto),
                    Err(e) => {
                        let _ = engine.rollback(txn);
                        (Err(e.into()), TxnState::Auto)
                    },
                },
                Err(err) => {
                    let _ = engine.rollback(txn);
                    (Err(err), TxnState::Auto)
                },
            }
        },
    }
}

/// Adapts the engine's schema lookup to the analyzer's narrower [`Catalog`] port, resolving names
/// under `txn`'s snapshot so analysis sees the same schema visibility execution will.
///
/// Carries the connection's authenticated `user` so row-level security is enforced for real
/// connections: a non-superuser's policy selection and superuser bypass here agree with the
/// `CURRENT_USER` the predicate evaluates against at execution (the wire runs the statement with the
/// same user via [`execute_in_txn_as`](nusadb_sql::execute_in_txn_as)).
struct EngineCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: TxnId,
    user: &'a str,
    /// The session's ordered `search_path` schemas, derived from `SET search_path`; `[public]`
    /// when unset. An unqualified name is created in the first entry and resolved through the list.
    search_path: Vec<String>,
}

impl<'a> EngineCatalog<'a> {
    /// Build a catalog whose search path reflects the connection's `search_path` setting.
    fn new(
        engine: &'a dyn StorageEngine,
        txn: TxnId,
        user: &'a str,
        settings: &HashMap<String, String>,
    ) -> Self {
        let search_path =
            nusadb_sql::search_path_schemas(settings.get("search_path").map(String::as_str));
        Self {
            engine,
            txn,
            user,
            search_path,
        }
    }
}

impl Catalog for EngineCatalog<'_> {
    fn search_path(&self) -> Vec<String> {
        self.search_path.clone()
    }

    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine
            .lookup_table_as_of(self.txn, name)
            .map_err(Into::into)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.engine
            .lookup_table_as_of_in(self.txn, schema, name)
            .map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        // The shared production adapter body: every complete engine index — since the backing-index unification
        // including the PK/UNIQUE constraint-backing ones, which are now maintained on every
        // write — so a point-get by PRIMARY KEY plans an IndexScan. Execution re-resolves the
        // chosen index by name, and the full WHERE filter is always kept, so an index the planner
        // picks here only narrows the scan.
        nusadb_sql::catalog_list_indexes(self.engine, self.txn, name)
    }

    fn table_stats(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, nusadb_sql::Error> {
        // The shared production adapter body: the planner gets the table's ANALYZE stats for
        // cost-based plan selection. `None` (never analyzed) leaves planning heuristic.
        nusadb_sql::catalog_table_stats(self.engine, self.txn, name)
    }

    fn approx_row_count(&self, name: &str) -> Result<u64, nusadb_sql::Error> {
        // The O(1) approximate row count — the vectorized-routing cardinality fallback when the
        // table was never analyzed, so a large un-analyzed table still vectorizes.
        nusadb_sql::catalog_approx_row_count(self.engine, self.txn, name)
    }

    fn lookup_view(&self, name: &str) -> Result<Option<String>, nusadb_sql::Error> {
        // Non-materialized views: read the stored defining SQL so the analyzer can inline it.
        nusadb_sql::lookup_view_definition(self.engine, self.txn, name)
    }

    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, nusadb_sql::Error> {
        // Explicit `CREATE VIEW name (cols)` list, so the inlined view body is renamed positionally.
        nusadb_sql::lookup_view_columns(self.engine, self.txn, name)
    }

    fn lookup_function(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_sql::FunctionDef>, nusadb_sql::Error> {
        // SQL functions: read the stored definition so the analyzer can inline the call.
        nusadb_sql::lookup_function_definition(self.engine, self.txn, name)
    }

    fn is_superuser(&self) -> bool {
        // Row-level security is bypassed only for the bootstrap superuser. Every other
        // authenticated connection is a regular user and is subject to RLS.
        self.user == nusadb_sql::BOOTSTRAP_SUPERUSER
    }

    fn current_user(&self) -> String {
        self.user.to_owned()
    }

    fn rls_enabled(&self, name: &str) -> Result<bool, nusadb_sql::Error> {
        // Row-level security: read the table's RLS flag under `txn`'s snapshot. Consulted only
        // for a non-superuser (`is_superuser` short-circuits first).
        nusadb_sql::rls_table_enabled(self.engine, self.txn, name)
    }

    fn lookup_policies(&self, name: &str) -> Result<Vec<nusadb_sql::PolicyDef>, nusadb_sql::Error> {
        // Row-level security policies for `name` under `txn`'s snapshot. Reached only for a
        // non-superuser whose query targets this RLS-enabled table.
        nusadb_sql::lookup_policies_for(self.engine, self.txn, name)
    }
}

/// The `CommandComplete` tag for a result (e.g. `SELECT 3`, `INSERT 1`, `CREATE TABLE`).
fn command_tag(result: &ExecutionResult) -> String {
    match result {
        ExecutionResult::Rows { rows, .. } => format!("SELECT {}", rows.len()),
        ExecutionResult::Created(_) => "CREATE TABLE".to_owned(),
        ExecutionResult::Dropped => "DROP TABLE".to_owned(),
        ExecutionResult::Altered => "ALTER TABLE".to_owned(),
        ExecutionResult::Inserted(n) => format!("INSERT {n}"),
        ExecutionResult::Updated(n) => format!("UPDATE {n}"),
        ExecutionResult::Deleted(n) => format!("DELETE {n}"),
        ExecutionResult::Merged(n) => format!("MERGE {n}"),
        ExecutionResult::TransactionBegun => "BEGIN".to_owned(),
        ExecutionResult::TransactionCommitted => "COMMIT".to_owned(),
        // A full ROLLBACK and a ROLLBACK TO SAVEPOINT both report the standard `ROLLBACK` tag.
        ExecutionResult::TransactionRolledBack | ExecutionResult::RolledBackToSavepoint => {
            "ROLLBACK".to_owned()
        },
        // SET TRANSACTION and SET/RESET <var> both report the standard `SET` tag.
        ExecutionResult::TransactionCharacteristicsSet | ExecutionResult::VariableSet => {
            "SET".to_owned()
        },
        ExecutionResult::SavepointCreated => "SAVEPOINT".to_owned(),
        ExecutionResult::SavepointReleased => "RELEASE".to_owned(),
        ExecutionResult::Vacuumed(n) => format!("VACUUM {n}"),
        ExecutionResult::Reindexed => "REINDEX".to_owned(),
        ExecutionResult::Analyzed { .. } => "ANALYZE".to_owned(),
        ExecutionResult::Commented => "COMMENT".to_owned(),
        ExecutionResult::TableLocked => "LOCK TABLE".to_owned(),
        ExecutionResult::Prepared => "PREPARE".to_owned(),
        ExecutionResult::Deallocated => "DEALLOCATE".to_owned(),
        ExecutionResult::SchemaCreated => "CREATE SCHEMA".to_owned(),
        ExecutionResult::SchemaDropped => "DROP SCHEMA".to_owned(),
        ExecutionResult::DatabaseCreated => "CREATE DATABASE".to_owned(),
        ExecutionResult::DatabaseAltered => "ALTER DATABASE".to_owned(),
        ExecutionResult::DatabaseDropped => "DROP DATABASE".to_owned(),
        ExecutionResult::SequenceCreated => "CREATE SEQUENCE".to_owned(),
        ExecutionResult::SequenceDropped => "DROP SEQUENCE".to_owned(),
        ExecutionResult::IndexCreated => "CREATE INDEX".to_owned(),
        ExecutionResult::IndexDropped => "DROP INDEX".to_owned(),
        ExecutionResult::TriggerCreated => "CREATE TRIGGER".to_owned(),
        ExecutionResult::TriggerDropped => "DROP TRIGGER".to_owned(),
        ExecutionResult::TriggerAltered => "ALTER TRIGGER".to_owned(),
        ExecutionResult::ProcedureCreated => "CREATE PROCEDURE".to_owned(),
        ExecutionResult::ProcedureDropped => "DROP PROCEDURE".to_owned(),
        ExecutionResult::ProcedureCalled => "CALL".to_owned(),
        ExecutionResult::FunctionCreated => "CREATE FUNCTION".to_owned(),
        ExecutionResult::FunctionDropped => "DROP FUNCTION".to_owned(),
    }
}

/// The format code negotiated for result column `col`. With no codes every column is
/// text; a single code applies to all columns; otherwise it is one code per column (a missing
/// entry defaults to text). Any code other than `1` (binary) is treated as text.
fn column_format(result_formats: &[u16], col: usize) -> u16 {
    match result_formats {
        [] => 0,
        [only] => *only,
        many => many.get(col).copied().unwrap_or(0),
    }
}

/// Encode one field in the negotiated format: binary (`1`) via [`encode_binary`], otherwise the
/// text rendering.
fn encode_field(value: Value, format: u16) -> Option<Vec<u8>> {
    if format == 1 {
        crate::encode_binary(&value)
    } else {
        value_to_field(value)
    }
}

/// Render a SQL value as the wire's text-format field bytes (`None` = NULL).
fn value_to_field(value: Value) -> Option<Vec<u8>> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(if b {
            b"true".to_vec()
        } else {
            b"false".to_vec()
        }),
        Value::Int(i) => Some(i.to_string().into_bytes()),
        Value::Float(f) => Some(f.to_string().into_bytes()),
        Value::Text(s) => Some(s.into_bytes()),
        // JSON is sent in the spaced display form (`{"a": 1}`), matching standard jsonb text output.
        Value::Json(s) => Some(nusadb_sql::json::display_form(&s).into_bytes()),
        // Temporal + UUID render in their canonical text form.
        Value::Date(d) => Some(nusadb_sql::temporal::format_date(d).into_bytes()),
        Value::Time(t) => Some(nusadb_sql::temporal::format_time(t).into_bytes()),
        Value::Timestamp(t) => Some(nusadb_sql::temporal::format_timestamp(t).into_bytes()),
        Value::TimestampTz(t) => Some(nusadb_sql::temporal::format_timestamptz(t).into_bytes()),
        Value::TimeTz(t) => Some(nusadb_sql::temporal::format_timetz(t).into_bytes()),
        Value::Uuid(u) => Some(nusadb_sql::temporal::format_uuid(&u).into_bytes()),
        Value::Numeric(d) => Some(d.format().into_bytes()),
        Value::Interval(iv) => Some(iv.format().into_bytes()),
        Value::Array(ref items) => Some(nusadb_sql::display::array_text(items).into_bytes()),
        Value::Vector(ref v) => Some(nusadb_sql::vector::format(v).into_bytes()),
        // BYTEA renders in the canonical `\x<hex>` text form.
        Value::Bytes(ref b) => Some(nusadb_sql::display::bytea_hex(b).into_bytes()),
    }
}

fn command_complete(tag: &str) -> BackendMessage {
    BackendMessage::CommandComplete {
        tag: tag.to_owned(),
    }
}

fn error_response(message: &str) -> BackendMessage {
    // XX000 = internal_error (SQLSTATE). Use `error_response_coded` for an error whose SQLSTATE the
    // engine classifies (e.g. a serialization conflict → 40001).
    error_response_coded(message, "XX000")
}

/// An `ErrorResponse` carrying an explicit SQLSTATE class code (B-QA SQLSTATE). A serialization
/// conflict (`40001`) / deadlock (`40P01`) reaches client retry middleware as a *retryable* error
/// rather than the opaque `XX000`.
fn error_response_coded(message: &str, code: &str) -> BackendMessage {
    BackendMessage::Error {
        code: code.to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::*;

    fn settings_with(entries: &[(&str, &str)]) -> std::sync::Mutex<HashMap<String, String>> {
        std::sync::Mutex::new(
            entries
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        )
    }

    /// A session `SET statement_timeout` must win over the server default, `0` must disable the
    /// timeout, and an absent (or, defensively, unparseable) value must fall back to the default.
    #[test]
    fn session_statement_timeout_overrides_the_server_default() {
        let default = Some(Duration::from_secs(30));
        // Session value wins — both shorter and longer than the default.
        for (value, expect) in [
            ("100", Some(Duration::from_millis(100))),
            ("100ms", Some(Duration::from_millis(100))),
            ("5min", Some(Duration::from_mins(5))),
            // 0 = the session opts out of any timeout, even when the server has one.
            ("0", None),
        ] {
            assert_eq!(
                effective_statement_timeout(
                    &settings_with(&[("statement_timeout", value)]),
                    default
                ),
                expect,
                "session statement_timeout = {value:?}"
            );
        }
        // No session value → the server default applies.
        assert_eq!(
            effective_statement_timeout(&settings_with(&[]), default),
            default
        );
        assert_eq!(effective_statement_timeout(&settings_with(&[]), None), None);
        // Defensive: an unparseable stored value (SET-time validation normally rejects it) falls
        // back to the server default rather than silently disabling the timeout.
        assert_eq!(
            effective_statement_timeout(
                &settings_with(&[("statement_timeout", "banana")]),
                default
            ),
            default
        );
    }
}
