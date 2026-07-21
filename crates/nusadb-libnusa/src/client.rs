//! The [`Client`]: one authenticated connection over the Nusa Wire Protocol.
//!
//! A client wraps a [`Transport`] (plaintext or TLS), performs the Startup + auth handshake
//! (trust-on-startup or SCRAM-SHA-256), and drives the simple- and extended-query paths described
//! in `docs/wire-protocol.md`. It captures the connection's `BackendKeyData` so an in-flight
//! statement can be cancelled out of band (§13).

use std::sync::Arc;

use nusadb_wire::auth::scram;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, PROTOCOL_VERSION};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

use crate::error::{Error, Result};
use crate::transport::Transport;
use crate::value::{Param, QueryResult, Row};

const SCRAM_MECHANISM: &str = "SCRAM-SHA-256";

/// How to reach and authenticate to a server.
///
/// Build with [`Config::new`] and the chainable setters; pass it to [`Client::connect`] or a
/// [`Pool`](crate::Pool).
#[derive(Clone)]
pub struct Config {
    host: String,
    port: u16,
    user: String,
    database: String,
    password: Option<String>,
    tls: Option<Arc<rustls::ClientConfig>>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the password.
        f.debug_struct("Config")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("user", &self.user)
            .field("database", &self.database)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field("tls", &self.tls.is_some())
            .finish()
    }
}

impl Config {
    /// A configuration for `user`@`host:port`/`database` (plaintext, trust-on-startup by default).
    pub fn new(
        host: impl Into<String>,
        port: u16,
        user: impl Into<String>,
        database: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            port,
            user: user.into(),
            database: database.into(),
            password: None,
            tls: None,
        }
    }

    /// Set the SCRAM password (required when the server runs with authentication).
    #[must_use]
    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(password.into());
        self
    }

    /// Enable TLS using `config` (build one with [`tls_client_config`](crate::tls_client_config)).
    #[must_use]
    pub fn tls(mut self, config: Arc<rustls::ClientConfig>) -> Self {
        self.tls = Some(config);
        self
    }

    /// The configured user.
    #[must_use]
    pub fn user(&self) -> &str {
        &self.user
    }
}

/// A connection's out-of-band cancellation key (`docs/wire-protocol.md` §8 / §13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKey {
    /// The connection's backend process id.
    pub pid: u32,
    /// The secret proving the holder observed this connection's `BackendKeyData`.
    pub secret: u32,
}

/// A prepared statement handle (a named server-side statement created by [`Client::prepare`]).
#[derive(Debug, Clone)]
pub struct Statement {
    name: String,
}

impl Statement {
    /// The server-side statement name.
    #[must_use]
    pub const fn name(&self) -> &str {
        self.name.as_str()
    }
}

/// One authenticated connection to a NusaDB server.
#[derive(Debug)]
pub struct Client {
    conn: Connection<Transport>,
    backend_key: Option<BackendKey>,
    config: Config,
    next_id: u64,
}

impl Client {
    /// Connect, perform the handshake, and authenticate, returning a ready client.
    ///
    /// # Errors
    /// [`Error::Io`] on a transport failure, [`Error::Tls`] on a TLS-handshake failure,
    /// [`Error::Auth`] if authentication is required but no/incorrect password is configured, or
    /// [`Error::Server`] if the server rejects the Startup.
    pub async fn connect(config: &Config) -> Result<Self> {
        let tcp = TcpStream::connect((config.host.as_str(), config.port)).await?;
        let transport = match &config.tls {
            None => Transport::Plain(tcp),
            Some(tls) => {
                let connector = TlsConnector::from(Arc::clone(tls));
                let name = rustls::pki_types::ServerName::try_from(config.host.clone())
                    .map_err(|_| Error::Tls(format!("invalid server name {:?}", config.host)))?;
                Transport::Tls(Box::new(connector.connect(name, tcp).await?))
            },
        };
        let mut client = Self {
            conn: Connection::new(transport),
            backend_key: None,
            config: config.clone(),
            next_id: 0,
        };
        client.handshake().await?;
        Ok(client)
    }

    /// This connection's cancellation key, captured during the handshake.
    #[must_use]
    pub const fn backend_key(&self) -> Option<BackendKey> {
        self.backend_key
    }

    /// Run one SQL string via the simple-query protocol and collect its result.
    ///
    /// # Errors
    /// [`Error::Server`] if the server rejects the statement; [`Error::Io`]/[`Error::Protocol`] on
    /// a transport or protocol failure.
    pub async fn simple_query(&mut self, sql: &str) -> Result<QueryResult> {
        self.send(FrontendMessage::Query {
            sql: sql.to_owned(),
        })
        .await?;
        self.collect().await
    }

    /// Run a statement that returns no rows (DDL/DML) and report the affected-row count parsed from
    /// its command tag (`0` when the tag carries no count).
    ///
    /// # Errors
    /// As for [`simple_query`](Self::simple_query).
    pub async fn execute(&mut self, sql: &str, params: &[Param]) -> Result<u64> {
        let result = if params.is_empty() {
            self.simple_query(sql).await?
        } else {
            self.query(sql, params).await?
        };
        Ok(result.affected().unwrap_or(0))
    }

    /// Run a one-shot parameterised query via the extended-query protocol (Parse/Bind/Describe/
    /// Execute/Sync on the unnamed statement and portal).
    ///
    /// # Errors
    /// As for [`simple_query`](Self::simple_query).
    pub async fn query(&mut self, sql: &str, params: &[Param]) -> Result<QueryResult> {
        self.send(FrontendMessage::Parse {
            name: String::new(),
            sql: sql.to_owned(),
            param_types: Vec::new(),
        })
        .await?;
        self.bind_describe_execute_sync(String::new(), String::new(), params)
            .await?;
        self.collect().await
    }

    /// Prepare a named statement for repeated execution with [`query_prepared`](Self::query_prepared).
    ///
    /// # Errors
    /// [`Error::Server`] if the SQL fails to parse on the server; transport/protocol errors.
    pub async fn prepare(&mut self, sql: &str) -> Result<Statement> {
        let name = self.fresh_name("stmt");
        self.send(FrontendMessage::Parse {
            name: name.clone(),
            sql: sql.to_owned(),
            param_types: Vec::new(),
        })
        .await?;
        self.send(FrontendMessage::Sync).await?;
        // Parse errors surface here as a server Error followed by ReadyForQuery.
        self.collect().await?;
        Ok(Statement { name })
    }

    /// Bind `params` into a fresh portal of `stmt`, execute it, and collect the result.
    ///
    /// # Errors
    /// As for [`query`](Self::query).
    pub async fn query_prepared(
        &mut self,
        stmt: &Statement,
        params: &[Param],
    ) -> Result<QueryResult> {
        let portal = self.fresh_name("portal");
        self.bind_describe_execute_sync(portal, stmt.name.clone(), params)
            .await?;
        self.collect().await
    }

    /// Run one statement once per parameter set, reusing a single prepared statement — the bulk
    /// insert/update path. Returns the per-set affected-row counts (`None` for a set that produced
    /// rows or carried no command tag). One `Parse`, then one bind/execute per set (the wire
    /// protocol has no batch pipeline, so this is N round-trips, not one); the first failing set
    /// returns its error.
    ///
    /// # Errors
    /// As for [`query`](Self::query).
    pub async fn execute_many(
        &mut self,
        sql: &str,
        param_sets: &[Vec<Param>],
    ) -> Result<Vec<Option<u64>>> {
        let stmt = self.prepare(sql).await?;
        let mut counts = Vec::with_capacity(param_sets.len());
        for params in param_sets {
            let result = self.query_prepared(&stmt, params).await?;
            counts.push(result.affected());
        }
        Ok(counts)
    }

    /// Begin an explicit transaction (`BEGIN`). Subsequent statements run inside it until
    /// [`commit`](Self::commit) or [`rollback`](Self::rollback); the server reports the
    /// transaction status in each `ReadyForQuery`.
    ///
    /// # Errors
    /// As for [`simple_query`](Self::simple_query).
    pub async fn begin(&mut self) -> Result<()> {
        self.simple_query("BEGIN").await.map(drop)
    }

    /// Commit the current transaction (`COMMIT`).
    ///
    /// # Errors
    /// As for [`simple_query`](Self::simple_query).
    pub async fn commit(&mut self) -> Result<()> {
        self.simple_query("COMMIT").await.map(drop)
    }

    /// Roll back the current transaction (`ROLLBACK`), discarding its uncommitted changes. Also the
    /// way to leave a failed transaction (one whose statement errored).
    ///
    /// # Errors
    /// As for [`simple_query`](Self::simple_query).
    pub async fn rollback(&mut self) -> Result<()> {
        self.simple_query("ROLLBACK").await.map(drop)
    }

    /// Run `f` inside a transaction: `BEGIN`, then the closure's statements, then `COMMIT` — or
    /// `ROLLBACK` if the closure returns an error (the error is propagated).
    ///
    /// # Errors
    /// The closure's error (after rolling back), or a transport/server error from
    /// `BEGIN`/`COMMIT`/`ROLLBACK`.
    pub async fn transaction<F, T>(&mut self, f: F) -> Result<T>
    where
        F: AsyncFnOnce(&mut Self) -> Result<T>,
    {
        self.begin().await?;
        match f(self).await {
            Ok(value) => {
                self.commit().await?;
                Ok(value)
            },
            Err(e) => {
                let _ = self.rollback().await;
                Err(e)
            },
        }
    }

    /// Politely close the connection (`Terminate`).
    ///
    /// # Errors
    /// [`Error::Io`] if the final write fails.
    pub async fn close(mut self) -> Result<()> {
        self.send(FrontendMessage::Terminate).await?;
        // The connection drops right after: force the queued frame onto the wire.
        self.conn.flush_now().await?;
        Ok(())
    }

    /// Cancel an in-flight statement on another connection, identified by its [`BackendKey`].
    ///
    /// Opens a fresh connection to the same server and sends a `CancelRequest` in place of Startup
    /// (`docs/wire-protocol.md` §13), then disconnects. A wrong/stale key is a silent no-op on the
    /// server side.
    ///
    /// # Errors
    /// [`Error::Io`]/[`Error::Tls`] if the cancel connection cannot be opened.
    pub async fn cancel(config: &Config, key: BackendKey) -> Result<()> {
        let tcp = TcpStream::connect((config.host.as_str(), config.port)).await?;
        let transport = match &config.tls {
            None => Transport::Plain(tcp),
            Some(tls) => {
                let connector = TlsConnector::from(Arc::clone(tls));
                let name = rustls::pki_types::ServerName::try_from(config.host.clone())
                    .map_err(|_| Error::Tls(format!("invalid server name {:?}", config.host)))?;
                Transport::Tls(Box::new(connector.connect(name, tcp).await?))
            },
        };
        let mut conn = Connection::new(transport);
        conn.write_frame(
            &FrontendMessage::CancelRequest {
                pid: key.pid,
                secret: key.secret,
            }
            .encode()?,
        )
        .await?;
        // This connection never reads (the cancel protocol is fire-and-disconnect), so the frame
        // must be flushed explicitly or it would die in the output queue.
        conn.flush_now().await?;
        Ok(())
    }

    // --- internals ---

    /// Send the Startup, run authentication if challenged, and capture `BackendKeyData`, returning
    /// once the server is `ReadyForQuery`.
    async fn handshake(&mut self) -> Result<()> {
        self.send(FrontendMessage::Startup {
            major: PROTOCOL_VERSION.0,
            minor: PROTOCOL_VERSION.1,
            user: self.config.user.clone(),
            database: self.config.database.clone(),
        })
        .await?;
        loop {
            match self.next_message().await? {
                BackendMessage::ReadyForQuery(_) => return Ok(()),
                BackendMessage::AuthSasl { mechanisms } => self.scram(&mechanisms).await?,
                BackendMessage::BackendKeyData { pid, secret } => {
                    self.backend_key = Some(BackendKey { pid, secret });
                },
                BackendMessage::Error { code, message } => {
                    return Err(Error::Server { code, message });
                },
                // AuthOk and any other pre-ready acknowledgement.
                _ => {},
            }
        }
    }

    /// Drive the client side of the SCRAM-SHA-256 exchange, verifying the server's signature
    /// (mutual auth) before returning.
    async fn scram(&mut self, mechanisms: &[String]) -> Result<()> {
        if !mechanisms.iter().any(|m| m == SCRAM_MECHANISM) {
            return Err(Error::Auth(format!(
                "server offered no supported SASL mechanism (got {mechanisms:?})"
            )));
        }
        let password =
            self.config.password.clone().ok_or_else(|| {
                Error::Auth("server requires a password but none was set".to_owned())
            })?;
        let user = self.config.user.clone();

        // client-first: GS2 header `n,,` (no channel binding) + `n=<user>,r=<nonce>`.
        let client_nonce = scram::generate_nonce().map_err(|e| Error::Auth(e.to_string()))?;
        let gs2_header = "n,,";
        let client_first_bare = format!("n={user},r={client_nonce}");
        self.send(FrontendMessage::SaslInitialResponse {
            mechanism: SCRAM_MECHANISM.to_owned(),
            data: format!("{gs2_header}{client_first_bare}").into_bytes(),
        })
        .await?;

        // server-first.
        let BackendMessage::AuthSaslContinue { data } = self.next_message().await? else {
            return Err(Error::Protocol(
                "expected a SASL continue message".to_owned(),
            ));
        };
        let server_first_msg = String::from_utf8(data)
            .map_err(|_| Error::Protocol("server-first is not valid UTF-8".to_owned()))?;
        let server_first =
            scram::ServerFirst::parse(&server_first_msg).map_err(|e| Error::Auth(e.to_string()))?;

        // client-final with proof.
        let client_final = scram::client_final_message(
            &password,
            gs2_header,
            &client_first_bare,
            &server_first_msg,
            &server_first,
        )
        .map_err(|e| Error::Auth(e.to_string()))?;
        let without_proof = client_final
            .rsplit_once(",p=")
            .map_or(client_final.as_str(), |(head, _)| head);
        let auth_message =
            scram::auth_message(&client_first_bare, &server_first_msg, without_proof);
        self.send(FrontendMessage::SaslResponse {
            data: client_final.clone().into_bytes(),
        })
        .await?;

        // server-final: verify the server's signature.
        match self.next_message().await? {
            BackendMessage::AuthSaslFinal { data } => {
                let server_final = String::from_utf8(data)
                    .map_err(|_| Error::Protocol("server-final is not valid UTF-8".to_owned()))?;
                scram::verify_server_signature(
                    &password,
                    &server_first.salt,
                    server_first.iterations,
                    &auth_message,
                    &server_final,
                )
                .map_err(|_| Error::Auth("server signature did not verify".to_owned()))?;
                Ok(())
            },
            BackendMessage::Error { code, message } => Err(Error::Server { code, message }),
            _ => Err(Error::Protocol("expected a SASL final message".to_owned())),
        }
    }

    /// Send `Bind` + `Describe(Portal)` + `Execute` + `Sync` for the extended-query path.
    async fn bind_describe_execute_sync(
        &mut self,
        portal: String,
        statement: String,
        params: &[Param],
    ) -> Result<()> {
        let wire_params = params.iter().map(Param::to_wire).collect();
        self.send(FrontendMessage::Bind {
            portal: portal.clone(),
            statement,
            params: wire_params,
            result_formats: Vec::new(),
        })
        .await?;
        self.send(FrontendMessage::Describe {
            target: nusadb_wire::DescribeTarget::Portal,
            name: portal.clone(),
        })
        .await?;
        self.send(FrontendMessage::Execute {
            portal,
            max_rows: 0,
        })
        .await?;
        self.send(FrontendMessage::Sync).await
    }

    /// Read backend messages until `ReadyForQuery`, assembling a [`QueryResult`]. A server `Error`
    /// is captured and returned (after the stream resynchronises at `ReadyForQuery`).
    async fn collect(&mut self) -> Result<QueryResult> {
        let mut columns: Arc<[String]> = Arc::from([]);
        let mut column_types: Arc<[Option<String>]> = Arc::from([]);
        let mut rows: Vec<Row> = Vec::new();
        let mut tag: Option<String> = None;
        let mut error: Option<Error> = None;
        loop {
            match self.next_message().await? {
                BackendMessage::RowDescription { columns: cols } => {
                    column_types = vec![None; cols.len()].into();
                    columns = Arc::from(cols);
                },
                // Protocol 1.1 typed metadata: the client negotiates `minor = 1`, so
                // the server may answer with the typed form — keep the names and the per-column type
                // names (the tag taxonomy, wire-protocol.md §9.2).
                BackendMessage::RowDescriptionTyped { columns: cols } => {
                    columns = cols.iter().map(|(name, _)| name.clone()).collect();
                    column_types = cols
                        .iter()
                        .map(|(_, tag)| Some(nusadb_wire::column_type_name(*tag).to_owned()))
                        .collect();
                },
                BackendMessage::DataRow { values } => {
                    rows.push(Row::new(Arc::clone(&columns), values));
                },
                BackendMessage::CommandComplete { tag: t } => tag = Some(t),
                BackendMessage::Error { code, message } => {
                    error = Some(Error::Server { code, message });
                },
                BackendMessage::ReadyForQuery(_) => break,
                // Extended-query acknowledgements carry no result data.
                _ => {},
            }
        }
        error.map_or_else(
            || {
                Ok(QueryResult {
                    columns,
                    column_types,
                    rows,
                    tag,
                })
            },
            Err,
        )
    }

    /// A fresh server-side statement/portal name unique to this connection.
    fn fresh_name(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        format!("nusa_{prefix}_{id}")
    }

    async fn send(&mut self, msg: FrontendMessage) -> Result<()> {
        self.conn.write_frame(&msg.encode()?).await?;
        Ok(())
    }

    async fn next_message(&mut self) -> Result<BackendMessage> {
        let frame = self
            .conn
            .read_frame()
            .await?
            .ok_or_else(|| Error::Protocol("server closed the connection".to_owned()))?;
        Ok(BackendMessage::decode(&frame)?)
    }
}
