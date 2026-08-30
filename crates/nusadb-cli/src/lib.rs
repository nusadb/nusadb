//! nusadb-cli client library: the Nusa Wire Protocol handshake, query execution, and result
//! rendering — factored out of `main.rs` so it can be tested against a real server.

use std::fmt::Write as _;
use std::io;

use nusadb_wire::{BackendMessage, Connection, FrontendMessage, PROTOCOL_VERSION};
use tokio::io::{AsyncRead, AsyncWrite};

fn unexpected_eof() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "server closed the connection")
}

/// Build a [`rustls::ClientConfig`] that trusts the certificate(s) in `ca_pem` as its only roots.
///
/// This is how the CLI connects to a server presenting a self-signed or private-CA certificate
/// (e.g. one made for `nusadb-server --tls-cert`); there is no system-trust-store fallback, so the
/// operator decides exactly what to trust.
///
/// # Errors
/// An [`io::Error`] if the PEM holds no certificate, a certificate fails to parse, or rustls
/// rejects it as a trust anchor.
pub fn tls_client_config(ca_pem: &[u8]) -> io::Result<rustls::ClientConfig> {
    use rustls::pki_types::CertificateDer;
    use rustls::pki_types::pem::PemObject as _;

    let mut roots = rustls::RootCertStore::empty();
    let mut added = 0usize;
    for cert in CertificateDer::pem_slice_iter(ca_pem) {
        let cert = cert.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        roots
            .add(cert)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        added += 1;
    }
    if added == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CA PEM contained no certificates",
        ));
    }
    Ok(rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth())
}

/// Send the Startup message and wait until the server reports it is ready, completing a
/// SCRAM-SHA-256 authentication exchange if the server requests one (client side).
///
/// A server started with `--auth-user` answers Startup with a SASL challenge; `password` (from
/// `--password` / `NUSADB_PASSWORD`) is then required. A trust-on-startup server never challenges,
/// so `password` is ignored there.
///
/// # Errors
/// I/O errors, an early disconnect, a server `Error` during startup, a server that requests
/// authentication when no password was supplied, or a failed SCRAM exchange (wrong password or a
/// server signature that does not verify).
pub async fn handshake<S>(
    conn: &mut Connection<S>,
    user: &str,
    database: &str,
    password: Option<&str>,
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    conn.write_frame(
        &FrontendMessage::Startup {
            major: PROTOCOL_VERSION.0,
            minor: PROTOCOL_VERSION.1,
            user: user.to_owned(),
            database: database.to_owned(),
        }
        .encode()?,
    )
    .await?;
    loop {
        let frame = conn.read_frame().await?.ok_or_else(unexpected_eof)?;
        match BackendMessage::decode(&frame).map_err(io::Error::other)? {
            BackendMessage::ReadyForQuery(_) => return Ok(()),
            BackendMessage::AuthSasl { mechanisms } => {
                let password = password.ok_or_else(|| {
                    io::Error::other(
                        "server requires authentication; supply a password with --password or \
                         the NUSADB_PASSWORD environment variable",
                    )
                })?;
                scram_authenticate(conn, user, password, &mechanisms).await?;
            },
            BackendMessage::Error { code, message } => {
                return Err(io::Error::other(format!("{code}: {message}")));
            },
            _ => {}, // AuthOk and any pre-ready chatter
        }
    }
}

/// Drive the client side of the SCRAM-SHA-256 SASL handshake after the server's `AuthSasl` offer.
///
/// `client-first → server-first → client-final → server-final`, then the server's `AuthOk` and
/// `ReadyForQuery` are read by the caller's loop. The server's final signature is verified so the
/// client also authenticates the server (mutual auth), not just the other way round.
async fn scram_authenticate<S>(
    conn: &mut Connection<S>,
    user: &str,
    password: &str,
    mechanisms: &[String],
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use nusadb_wire::auth::scram;

    const MECHANISM: &str = "SCRAM-SHA-256";
    if !mechanisms.iter().any(|m| m == MECHANISM) {
        return Err(io::Error::other(format!(
            "server offered no supported SASL mechanism (got {mechanisms:?}, need {MECHANISM})"
        )));
    }

    // --- client-first ---
    // GS2 header `n,,` = no channel binding, no authzid; the bare message is `n=<user>,r=<nonce>`.
    let client_nonce = scram::generate_nonce().map_err(io::Error::other)?;
    let gs2_header = "n,,";
    let client_first_bare = format!("n={user},r={client_nonce}");
    let client_first = format!("{gs2_header}{client_first_bare}");
    conn.write_frame(
        &FrontendMessage::SaslInitialResponse {
            mechanism: MECHANISM.to_owned(),
            data: client_first.into_bytes(),
        }
        .encode()?,
    )
    .await?;

    // --- server-first ---
    let frame = conn.read_frame().await?.ok_or_else(unexpected_eof)?;
    let BackendMessage::AuthSaslContinue { data } =
        BackendMessage::decode(&frame).map_err(io::Error::other)?
    else {
        return Err(io::Error::other(
            "expected a SASL continue (server-first) message",
        ));
    };
    let server_first_msg = String::from_utf8(data)
        .map_err(|_| io::Error::other("server-first message is not valid UTF-8"))?;
    let server_first = scram::ServerFirst::parse(&server_first_msg).map_err(io::Error::other)?;

    // --- client-final (with proof) ---
    let client_final = scram::client_final_message(
        password,
        gs2_header,
        &client_first_bare,
        &server_first_msg,
        &server_first,
    )
    .map_err(io::Error::other)?;
    // The AuthMessage the server signs is `client-first-bare , server-first , client-final-no-proof`;
    // reconstruct the no-proof prefix (everything before `,p=`) to verify the server's signature.
    let without_proof = client_final
        .rsplit_once(",p=")
        .map_or(client_final.as_str(), |(head, _proof)| head);
    let auth_message = scram::auth_message(&client_first_bare, &server_first_msg, without_proof);
    conn.write_frame(
        &FrontendMessage::SaslResponse {
            data: client_final.clone().into_bytes(),
        }
        .encode()?,
    )
    .await?;

    // --- server-final: verify the server's signature (mutual auth) ---
    let frame = conn.read_frame().await?.ok_or_else(unexpected_eof)?;
    match BackendMessage::decode(&frame).map_err(io::Error::other)? {
        BackendMessage::AuthSaslFinal { data } => {
            let server_final = String::from_utf8(data)
                .map_err(|_| io::Error::other("server-final message is not valid UTF-8"))?;
            scram::verify_server_signature(
                password,
                &server_first.salt,
                server_first.iterations,
                &auth_message,
                &server_final,
            )
            .map_err(|_| io::Error::other("server signature did not verify"))?;
            Ok(())
        },
        BackendMessage::Error { code, message } => {
            Err(io::Error::other(format!("{code}: {message}")))
        },
        _ => Err(io::Error::other("expected a SASL final message")),
    }
}

/// The structured result of one statement.
///
/// Holds the column header (empty for a non-`SELECT`), the raw row values, the command tag, and
/// the first error (if the server rejected it). Produced by [`collect_result`] and rendered by
/// [`format_result`] (or the legacy [`run_query`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryResult {
    /// Column names from `RowDescription` (empty for DDL/DML with no result set).
    pub columns: Vec<String>,
    /// Row values, each a column-aligned list of optional (NULL = `None`) raw bytes.
    pub rows: Vec<Vec<Option<Vec<u8>>>>,
    /// `CommandComplete` tag (e.g. `SELECT 1`, `CREATE TABLE`), if the statement completed.
    pub tag: Option<String>,
    /// The rendered error line if the statement failed: `ERROR <code>: <message>` when the server
    /// rejected it, or `ERROR: <message>` for a refusal the client raised on its own.
    pub error: Option<String>,
    /// Set when this statement streamed `COPY … TO STDOUT` rows into the caller's sink. The tag
    /// then must not be written to that same stream — appending `COPY 3` to exported rows makes
    /// the file fail to load back.
    ///
    /// A `COPY … TO STDOUT` with no sink leaves this `false` and sets [`error`](Self::error), but
    /// the server counts the export a success, so [`tag`](Self::tag) is also set. A caller reading
    /// the fields directly should test `error` first.
    pub copied_out: bool,
    /// Set when this statement read the caller's source stream to feed `COPY … FROM STDIN`. A
    /// single stream serves one load, so a caller running a batch knows not to offer it again.
    pub copied_in: bool,
}

/// Stream `source` to the server as `CopyData` frames, then `CopyDone`. A read failure sends
/// `CopyFail` instead, which keeps the connection in protocol sync (the server rolls the load back
/// and returns to ready) rather than leaving it mid-`COPY`.
///
/// No explicit flush is needed: frames below the write buffer's threshold stay buffered, and the
/// caller's next `read_frame` flushes before it waits. A small payload would otherwise sit in the
/// buffer while both sides waited for each other.
async fn send_copy_data<S>(
    conn: &mut Connection<S>,
    source: &mut (dyn AsyncRead + Unpin + Send),
) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use tokio::io::AsyncReadExt as _;

    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match source.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                let chunk = buf.get(..n).unwrap_or(&[]).to_vec();
                conn.write_frame(&FrontendMessage::CopyData { data: chunk }.encode()?)
                    .await?;
            },
            Err(e) => {
                conn.write_frame(
                    &FrontendMessage::CopyFail {
                        message: format!("client read error during COPY: {e}"),
                    }
                    .encode()?,
                )
                .await?;
                return Ok(());
            },
        }
    }
    conn.write_frame(&FrontendMessage::CopyDone.encode()?).await
}

/// Abandon a `COPY … FROM STDIN` the client cannot serve. `CopyFail` puts the server back in
/// charge: it rolls the load back and echoes `message` inside its own `ErrorResponse`, which the
/// collector records — so no client-side error is set here, or it would be overwritten anyway.
async fn fail_copy_in<S>(conn: &mut Connection<S>, message: &str) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    conn.write_frame(
        &FrontendMessage::CopyFail {
            message: message.to_owned(),
        }
        .encode()?,
    )
    .await
}

/// The data streams a `COPY` statement needs.
///
/// `source` feeds `COPY … FROM STDIN`, `sink` receives `COPY … TO STDOUT`. Either may be absent,
/// in which case a `COPY` needing it is failed with an explanatory error rather than left waiting
/// — the server is mid-`COPY` and would otherwise block forever for data that is never coming.
#[derive(Default)]
#[expect(
    missing_debug_implementations,
    reason = "holds borrowed trait objects with no Debug bound; the streams have nothing to print"
)]
pub struct CopyIo<'a> {
    /// Bytes to send as the `COPY … FROM STDIN` payload (server text format: tab-delimited,
    /// `\N` for NULL, unless the statement's `WITH` options say otherwise).
    pub source: Option<&'a mut (dyn AsyncRead + Unpin + Send)>,
    /// Where to write the rows a `COPY … TO STDOUT` streams back.
    pub sink: Option<&'a mut (dyn AsyncWrite + Unpin + Send)>,
}

/// Run one SQL statement and collect its result into a [`QueryResult`].
///
/// A `COPY` that needs a data stream is refused (see [`collect_result_with_copy`], which this
/// delegates to with no streams attached).
///
/// # Errors
/// I/O errors or an early disconnect.
pub async fn collect_result<S>(conn: &mut Connection<S>, sql: &str) -> io::Result<QueryResult>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    collect_result_with_copy(conn, sql, CopyIo::default()).await
}

/// Run one SQL statement, streaming a `COPY`'s data through `copy` if it turns out to be one.
///
/// The `COPY` sub-protocol is a mode switch, not a normal reply: after `CopyInResponse` the server
/// reads `CopyData` frames until `CopyDone`, and after `CopyOutResponse` it writes them. A client
/// that ignores those replies and waits for `ReadyForQuery` deadlocks against a server waiting for
/// data. With no `source` to feed a `COPY … FROM STDIN` the load is abandoned with `CopyFail`; with
/// no `sink` for a `COPY … TO STDOUT` the rows are read and discarded, since that direction never
/// reads from the client. Either way the caller gets an error instead of a session that hangs.
///
/// # Errors
/// I/O errors or an early disconnect. A `COPY` the server refuses, or one with no stream to serve
/// it, is reported in [`QueryResult::error`] with the connection left ready for the next statement.
pub async fn collect_result_with_copy<S>(
    conn: &mut Connection<S>,
    sql: &str,
    mut copy: CopyIo<'_>,
) -> io::Result<QueryResult>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    conn.write_frame(
        &FrontendMessage::Query {
            sql: sql.to_owned(),
        }
        .encode()?,
    )
    .await?;
    let mut result = QueryResult::default();
    loop {
        let frame = conn.read_frame().await?.ok_or_else(unexpected_eof)?;
        match BackendMessage::decode(&frame).map_err(io::Error::other)? {
            // COPY … FROM STDIN: stream `source` up in 64 KiB chunks, then `CopyDone`. The server
            // replies `CommandComplete` + `ReadyForQuery`, so the loop carries on from here.
            BackendMessage::CopyInResponse { .. } => {
                if let Some(source) = copy.source.as_deref_mut() {
                    send_copy_data(conn, source).await?;
                    result.copied_in = true;
                } else {
                    fail_copy_in(
                        conn,
                        "COPY … FROM STDIN needs input; redirect a file into the command (one \
                         redirect feeds one load)",
                    )
                    .await?;
                }
            },
            // COPY … TO STDOUT: the server writes `CopyData` frames and never reads back, so an
            // absent sink is handled by discarding them below rather than by replying — a
            // `CopyFail` here is a frame the server documents as a client bug and drops.
            BackendMessage::CopyOutResponse { .. } => {
                result.copied_out = copy.sink.is_some();
                if copy.sink.is_none() {
                    result.error = Some(
                        "ERROR: COPY … TO STDOUT had no sink for the exported rows; they were \
                         discarded"
                            .to_owned(),
                    );
                }
            },
            BackendMessage::CopyData { data } => {
                if let Some(sink) = copy.sink.as_deref_mut() {
                    use tokio::io::AsyncWriteExt as _;
                    sink.write_all(&data).await?;
                }
            },
            BackendMessage::CopyDone => {
                if let Some(sink) = copy.sink.as_deref_mut() {
                    use tokio::io::AsyncWriteExt as _;
                    sink.flush().await?;
                }
            },
            BackendMessage::RowDescription { columns } => result.columns = columns,
            // Protocol 1.1 typed metadata: take the names; the CLI renders untyped text.
            BackendMessage::RowDescriptionTyped { columns } => {
                result.columns = columns.into_iter().map(|(name, _type_tag)| name).collect();
            },
            BackendMessage::DataRow { values } => result.rows.push(values),
            BackendMessage::CommandComplete { tag } => result.tag = Some(tag),
            BackendMessage::Error { code, message } => {
                result.error = Some(format!("ERROR {code}: {message}"));
            },
            BackendMessage::ReadyForQuery(_) => return Ok(result),
            // Auth and extended-query replies do not drive this simple-query collector. An
            // asynchronous LISTEN/NOTIFY delivery is not part of a query's result stream (the server
            // sends it only between statements), so it is ignored here; surfacing it interactively is
            // a LISTEN/NOTIFY follow-up.
            BackendMessage::AuthOk
            | BackendMessage::AuthSasl { .. }
            | BackendMessage::AuthSaslContinue { .. }
            | BackendMessage::AuthSaslFinal { .. }
            | BackendMessage::ParseComplete
            | BackendMessage::BindComplete
            | BackendMessage::CloseComplete
            | BackendMessage::ParameterDescription { .. }
            | BackendMessage::NoData
            | BackendMessage::PortalSuspended
            | BackendMessage::ParameterStatus { .. }
            | BackendMessage::NotificationResponse { .. }
            | BackendMessage::BackendKeyData { .. } => {},
        }
    }
}

/// Run one SQL statement and return its rendered lines in the default ` | `-separated format.
///
/// A `RowDescription` header, one line per row, then the command tag — or a single error line.
/// Equivalent to `format_result(&collect_result(..), OutputFormat::Pipe)`.
///
/// # Errors
/// I/O errors or an early disconnect.
pub async fn run_query<S>(conn: &mut Connection<S>, sql: &str) -> io::Result<Vec<String>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    Ok(format_result(
        &collect_result(conn, sql).await?,
        OutputFormat::Pipe,
    ))
}

/// Render one data row as ` | `-separated fields, with `NULL` for absent values.
#[must_use]
pub fn render_data_row(values: &[Option<Vec<u8>>]) -> String {
    values
        .iter()
        .map(|v| cell_text(v.as_deref()))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// How query results are rendered to the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// The default ` | `-separated layout (also what [`run_query`] emits).
    #[default]
    Pipe,
    /// Column-aligned table with a header rule.
    Aligned,
    /// One `name | value` line per column, grouped per record (expanded/vertical layout).
    Expanded,
    /// Comma-separated values with a header row; NULL is an empty field.
    Csv,
    /// A single JSON array of `{column: value}` objects; NULL is JSON `null`.
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "pipe" => Ok(Self::Pipe),
            "aligned" => Ok(Self::Aligned),
            "expanded" => Ok(Self::Expanded),
            "csv" => Ok(Self::Csv),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unknown format {other:?} (expected aligned, expanded, csv, or json)"
            )),
        }
    }
}

/// Render a `QueryResult` to output lines in the chosen format.
///
/// An error renders as the single error line in every format; a statement with no result set
/// (DDL/DML) renders as just its command tag. Otherwise rows are laid out per `format` —
/// `Pipe`/`Aligned`/`Expanded` append the command tag, while `Csv`/`Json` stay pure data.
#[must_use]
pub fn format_result(result: &QueryResult, format: OutputFormat) -> Vec<String> {
    if let Some(err) = &result.error {
        return vec![err.clone()];
    }
    if result.columns.is_empty() {
        // DDL/DML: just the command tag (if any), regardless of format.
        return result.tag.iter().cloned().collect();
    }
    let mut out = match format {
        OutputFormat::Pipe => format_pipe(result),
        OutputFormat::Aligned => format_aligned(result),
        OutputFormat::Expanded => format_expanded(result),
        OutputFormat::Csv => return format_csv(result),
        OutputFormat::Json => return vec![format_json(result)],
    };
    if let Some(tag) = &result.tag {
        out.push(tag.clone());
    }
    out
}

/// A cell as display text: `NULL` for absent values, lossy UTF-8 otherwise.
fn cell_text(value: Option<&[u8]>) -> String {
    value.map_or_else(
        || "NULL".to_owned(),
        |bytes| String::from_utf8_lossy(bytes).into_owned(),
    )
}

fn format_pipe(result: &QueryResult) -> Vec<String> {
    let mut out = vec![result.columns.join(" | ")];
    for row in &result.rows {
        out.push(render_data_row(row));
    }
    out
}

fn format_aligned(result: &QueryResult) -> Vec<String> {
    // Column width = the widest of the header and every cell in that column.
    let mut widths: Vec<usize> = result.columns.iter().map(String::len).collect();
    let cells: Vec<Vec<String>> = result
        .rows
        .iter()
        .map(|row| row.iter().map(|v| cell_text(v.as_deref())).collect())
        .collect();
    for row in &cells {
        for (i, cell) in row.iter().enumerate() {
            if let Some(w) = widths.get_mut(i) {
                *w = (*w).max(cell.len());
            }
        }
    }
    let pad = |s: &str, w: usize| format!("{s:<w$}");
    let header = result
        .columns
        .iter()
        .zip(&widths)
        .map(|(c, w)| pad(c, *w))
        .collect::<Vec<_>>()
        .join(" | ");
    let rule = widths
        .iter()
        .map(|w| "-".repeat(*w))
        .collect::<Vec<_>>()
        .join("-+-");
    let mut out = vec![header, rule];
    for row in &cells {
        out.push(
            row.iter()
                .enumerate()
                .map(|(i, c)| pad(c, *widths.get(i).unwrap_or(&0)))
                .collect::<Vec<_>>()
                .join(" | "),
        );
    }
    out
}

fn format_expanded(result: &QueryResult) -> Vec<String> {
    let name_w = result.columns.iter().map(String::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for (n, row) in result.rows.iter().enumerate() {
        out.push(format!("-[ RECORD {} ]-", n + 1));
        for (i, col) in result.columns.iter().enumerate() {
            let value = cell_text(row.get(i).and_then(|v| v.as_deref()));
            out.push(format!("{col:<name_w$} | {value}"));
        }
    }
    out
}

fn format_csv(result: &QueryResult) -> Vec<String> {
    let mut out = vec![
        result
            .columns
            .iter()
            .map(|c| csv_field(c))
            .collect::<Vec<_>>()
            .join(","),
    ];
    for row in &result.rows {
        out.push(
            row.iter()
                // NULL renders as an empty (unquoted) field.
                .map(|v| v.as_deref().map_or_else(String::new, |b| csv_field(&String::from_utf8_lossy(b))))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    out
}

/// Quote a CSV field if it contains a comma, quote, CR, or LF; double any embedded quotes.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

fn format_json(result: &QueryResult) -> String {
    let objects: Vec<String> = result
        .rows
        .iter()
        .map(|row| {
            let fields: Vec<String> = result
                .columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    let value = row.get(i).and_then(|v| v.as_deref()).map_or_else(
                        || "null".to_owned(),
                        |b| format!("\"{}\"", json_escape(&String::from_utf8_lossy(b))),
                    );
                    format!("\"{}\":{value}", json_escape(col))
                })
                .collect();
            format!("{{{}}}", fields.join(","))
        })
        .collect();
    format!("[{}]", objects.join(","))
}

/// Escape a string for embedding in a JSON double-quoted string.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            },
            c => out.push(c),
        }
    }
    out
}

/// If a dollar-quote delimiter (`$$` or `$tag$`) begins at `chars[i]` (which must be `$`), return its
/// tag (the text between the two dollar signs, empty for `$$`) and the delimiter's length in `char`s.
///
/// A tag follows unquoted-identifier rules (a letter or underscore, then letters/digits/underscores),
/// so a positional parameter like `$1` is not a delimiter and `$` there stays literal.
fn dollar_delimiter_at(chars: &[char], i: usize) -> Option<(String, usize)> {
    debug_assert_eq!(chars.get(i), Some(&'$'));
    let mut j = i + 1;
    while chars
        .get(j)
        .is_some_and(|&c| c.is_ascii_alphanumeric() || c == '_')
    {
        j += 1;
    }
    // The candidate tag is `chars[i + 1..j]`; a real delimiter closes it with a second `$`.
    if chars.get(j) != Some(&'$') {
        return None;
    }
    let tag: String = chars.iter().skip(i + 1).take(j - (i + 1)).collect();
    // A non-empty tag cannot start with a digit (identifier rule); this rejects `$1$` too.
    if tag.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((tag, j - i + 1))
}

/// Split a batch of SQL (e.g. a `--file` body) into individual statements on `;`.
///
/// Semicolons inside single- or double-quoted strings (with `''`/`""` doubling) do not split, nor do
/// semicolons inside a dollar-quoted string (`$$ … $$` or `$tag$ … $tag$`) — so a function or
/// procedure body written with dollar-quoting survives as one statement. Whitespace-only fragments
/// are dropped; trailing input without a `;` is returned as a final statement.
#[must_use]
pub fn split_statements(input: &str) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    // The tag of the dollar-quoted string currently open (empty for `$$`), or `None` when not inside
    // one; only its matching `$tag$` closes it, and quotes/semicolons are literal within.
    let mut dollar_tag: Option<String> = None;
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if let Some(tag) = &dollar_tag {
            if c == '$'
                && let Some((closing, len)) = dollar_delimiter_at(&chars, i)
                && &closing == tag
            {
                cur.extend(chars.iter().skip(i).take(len));
                i += len;
                dollar_tag = None;
                continue;
            }
            cur.push(c);
            i += 1;
            continue;
        }
        match c {
            '\'' if !in_double => {
                cur.push(c);
                if in_single && chars.get(i + 1) == Some(&'\'') {
                    cur.push('\''); // escaped '' — stays inside the string
                    i += 1;
                } else {
                    in_single = !in_single;
                }
                i += 1;
            },
            '"' if !in_single => {
                cur.push(c);
                if in_double && chars.get(i + 1) == Some(&'"') {
                    cur.push('"'); // escaped "" — stays inside the identifier
                    i += 1;
                } else {
                    in_double = !in_double;
                }
                i += 1;
            },
            '$' if !in_single && !in_double => {
                if let Some((tag, len)) = dollar_delimiter_at(&chars, i) {
                    cur.extend(chars.iter().skip(i).take(len));
                    i += len;
                    dollar_tag = Some(tag);
                } else {
                    cur.push(c);
                    i += 1;
                }
            },
            ';' if !in_single && !in_double => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_owned());
                }
                cur.clear();
                i += 1;
            },
            _ => {
                cur.push(c);
                i += 1;
            },
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

/// A REPL meta-command — handled client-side, never sent verbatim to the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Meta {
    /// End the session (`\q`, `\quit`, or `quit`).
    Quit,
    /// Print the meta-command help (`\?` / `\h`).
    Help,
    /// List databases (`\l`) — NusaDB is single-database, so the client prints the connected one.
    Databases,
    /// Expand to a SQL statement the client runs and renders (`\dt` → `SHOW TABLES`,
    /// `\d t` → `SHOW COLUMNS FROM t`).
    Sql(String),
}

/// The help text printed by `\?`.
pub const META_HELP: &str = "\\?            show this help\n\
     \\dt           list tables\n\
     \\d            list tables\n\
     \\d NAME       describe table NAME's columns\n\
     \\l            list databases\n\
     \\q, \\quit     quit";

/// Parse the *first* line of an input as a client-side meta-command, if it is one.
///
/// Recognised only as a standalone first line (so a `;`-terminated statement is never mistaken for
/// one). `\dt` / `\d` / `\d NAME` / `\l` expand to a catalog-introspection query; `\?` prints help;
/// `\q` / `\quit` / `quit` end the session.
#[must_use]
pub fn parse_meta(line: &str) -> Option<Meta> {
    let trimmed = line.trim();
    match trimmed {
        "\\q" | "\\quit" => Some(Meta::Quit),
        s if s.eq_ignore_ascii_case("quit") => Some(Meta::Quit),
        "\\?" | "\\h" => Some(Meta::Help),
        "\\l" | "\\list" => Some(Meta::Databases),
        "\\dt" | "\\d" => Some(Meta::Sql("SHOW TABLES".to_owned())),
        _ => trimmed
            .strip_prefix("\\d ")
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| Meta::Sql(format!("SHOW COLUMNS FROM {name}"))),
    }
}

/// Whether the accumulated multi-line input forms a complete statement ready to send.
///
/// True for a non-empty input whose last non-whitespace character is the `;` terminator. Until this
/// holds, the REPL keeps reading continuation lines.
#[must_use]
pub fn is_complete_statement(buf: &str) -> bool {
    buf.trim_end().ends_with(';')
}

/// Strip the trailing `;` terminator (and surrounding whitespace) from a completed statement.
///
/// The server's parser is fed one bare statement, exactly as it was before multi-line input.
#[must_use]
pub fn strip_terminator(buf: &str) -> &str {
    let trimmed = buf.trim();
    trimmed.strip_suffix(';').map_or(trimmed, str::trim_end)
}

#[cfg(test)]
mod tls_tests {
    use super::tls_client_config;

    const CA_PEM: &[u8] = include_bytes!("../tests/data/localhost-cert.pem");

    #[test]
    fn builds_from_a_valid_ca_certificate() {
        assert!(tls_client_config(CA_PEM).is_ok());
    }

    #[test]
    fn empty_pem_is_rejected() {
        assert!(tls_client_config(b"").is_err());
    }

    #[test]
    fn garbage_pem_is_rejected() {
        assert!(
            tls_client_config(
                b"-----BEGIN CERTIFICATE-----\nnot base64\n-----END CERTIFICATE-----\n"
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Meta, OutputFormat, QueryResult, format_result, is_complete_statement, parse_meta,
        split_statements, strip_terminator,
    };

    /// A two-column, two-row result with one NULL — the fixture for the format tests.
    fn sample() -> QueryResult {
        QueryResult {
            columns: vec!["id".to_owned(), "name".to_owned()],
            rows: vec![
                vec![Some(b"5".to_vec()), Some(b"alice".to_vec())],
                vec![Some(b"42".to_vec()), None],
            ],
            tag: Some("SELECT 2".to_owned()),
            error: None,
            ..QueryResult::default()
        }
    }

    #[test]
    fn meta_commands_are_recognised_case_insensitively() {
        assert_eq!(parse_meta("\\q"), Some(Meta::Quit));
        assert_eq!(parse_meta("\\quit"), Some(Meta::Quit));
        assert_eq!(parse_meta("  quit  "), Some(Meta::Quit));
        assert_eq!(parse_meta("QUIT"), Some(Meta::Quit));
        assert_eq!(parse_meta("select 1;"), None);
        assert_eq!(parse_meta("quitx"), None);
    }

    #[test]
    fn dot_commands_expand_to_introspection_queries() {
        // \dt / \d list tables; \d NAME describes a table; \l and \? are client-side.
        assert_eq!(
            parse_meta("\\dt"),
            Some(Meta::Sql("SHOW TABLES".to_owned()))
        );
        assert_eq!(parse_meta("\\d"), Some(Meta::Sql("SHOW TABLES".to_owned())));
        assert_eq!(
            parse_meta("\\d users"),
            Some(Meta::Sql("SHOW COLUMNS FROM users".to_owned())),
        );
        assert_eq!(parse_meta("  \\d  users  "), parse_meta("\\d users"));
        assert_eq!(parse_meta("\\l"), Some(Meta::Databases));
        assert_eq!(parse_meta("\\?"), Some(Meta::Help));
        // Not a meta-command.
        assert_eq!(parse_meta("\\dz"), None);
        assert_eq!(parse_meta("SELECT 1"), None);
    }

    #[test]
    fn a_statement_is_complete_only_when_semicolon_terminated() {
        assert!(!is_complete_statement("select 1"));
        assert!(!is_complete_statement("select 1\n  from t"));
        assert!(is_complete_statement("select 1;"));
        assert!(is_complete_statement("select 1\nfrom t;  \n"));
        assert!(!is_complete_statement("")); // nothing yet
    }

    #[test]
    fn the_terminator_is_stripped_before_sending() {
        assert_eq!(strip_terminator("select 1;"), "select 1");
        assert_eq!(strip_terminator("select 1 ;  "), "select 1");
        assert_eq!(strip_terminator("select 1\nfrom t;\n"), "select 1\nfrom t");
        // No terminator (e.g. a one-shot --command): returned trimmed, unchanged.
        assert_eq!(strip_terminator("  select 1  "), "select 1");
    }

    #[test]
    fn output_format_parses_names_case_insensitively_and_rejects_junk() {
        assert_eq!("CSV".parse(), Ok(OutputFormat::Csv));
        assert_eq!("Json".parse(), Ok(OutputFormat::Json));
        assert_eq!("aligned".parse(), Ok(OutputFormat::Aligned));
        assert_eq!("expanded".parse(), Ok(OutputFormat::Expanded));
        assert!("yaml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn pipe_format_matches_the_legacy_rendering() {
        // header, one line per row (NULL spelled out), then the tag.
        assert_eq!(
            format_result(&sample(), OutputFormat::Pipe),
            vec!["id | name", "5 | alice", "42 | NULL", "SELECT 2"]
        );
    }

    #[test]
    fn aligned_format_pads_columns_and_draws_a_rule() {
        assert_eq!(
            format_result(&sample(), OutputFormat::Aligned),
            vec![
                "id | name ",
                "---+------",
                "5  | alice",
                "42 | NULL ",
                "SELECT 2"
            ]
        );
    }

    #[test]
    fn expanded_format_groups_one_record_per_block() {
        assert_eq!(
            format_result(&sample(), OutputFormat::Expanded),
            vec![
                "-[ RECORD 1 ]-",
                "id   | 5",
                "name | alice",
                "-[ RECORD 2 ]-",
                "id   | 42",
                "name | NULL",
                "SELECT 2",
            ]
        );
    }

    #[test]
    fn csv_format_quotes_only_when_needed_and_blanks_null() {
        let mut r = sample();
        r.rows[0][1] = Some(b"a,\"b\"".to_vec()); // needs quoting + quote-doubling
        assert_eq!(
            format_result(&r, OutputFormat::Csv),
            vec!["id,name", "5,\"a,\"\"b\"\"\"", "42,"] // no tag line for csv
        );
    }

    #[test]
    fn json_format_emits_one_array_with_null_and_escaping() {
        let mut r = sample();
        r.rows[0][1] = Some(b"a\"b".to_vec());
        assert_eq!(
            format_result(&r, OutputFormat::Json),
            vec![r#"[{"id":"5","name":"a\"b"},{"id":"42","name":null}]"#]
        );
    }

    #[test]
    fn a_command_result_renders_as_just_the_tag_in_every_format() {
        let ddl = QueryResult {
            tag: Some("CREATE TABLE".to_owned()),
            ..QueryResult::default()
        };
        for f in [
            OutputFormat::Pipe,
            OutputFormat::Aligned,
            OutputFormat::Csv,
            OutputFormat::Json,
        ] {
            assert_eq!(format_result(&ddl, f), vec!["CREATE TABLE".to_owned()]);
        }
    }

    #[test]
    fn statements_split_on_semicolons_outside_quotes() {
        assert_eq!(split_statements("a; b; c"), vec!["a", "b", "c"]);
        assert_eq!(split_statements("select 1;"), vec!["select 1"]);
        assert!(split_statements("   \n ").is_empty());
        // `;` inside a string literal does not split; `''` is an escaped quote.
        assert_eq!(
            split_statements("INSERT INTO t VALUES ('a;b'); SELECT 'it''s; ok'"),
            vec!["INSERT INTO t VALUES ('a;b')", "SELECT 'it''s; ok'"]
        );
        // trailing statement without a terminator is kept.
        assert_eq!(split_statements("a; b"), vec!["a", "b"]);
    }

    #[test]
    fn statements_do_not_split_inside_dollar_quotes() {
        // A `;` inside a `$$ … $$` body keeps the whole `CREATE FUNCTION` as one statement.
        assert_eq!(
            split_statements(
                "CREATE FUNCTION f(a int) RETURNS int LANGUAGE nusascript \
                 AS $$ BEGIN RETURN a * 3; END $$; SELECT f(2)"
            ),
            vec![
                "CREATE FUNCTION f(a int) RETURNS int LANGUAGE nusascript \
                 AS $$ BEGIN RETURN a * 3; END $$",
                "SELECT f(2)",
            ]
        );
        // A custom tag closes only on the same tag; a `;` and a bare `$$` inside it are literal.
        assert_eq!(
            split_statements("DO $body$ SELECT 1; SELECT '$$' $body$; SELECT 2"),
            vec!["DO $body$ SELECT 1; SELECT '$$' $body$", "SELECT 2"]
        );
        // A positional parameter (`$1`) is not a dollar-quote, so `;` still splits around it.
        assert_eq!(
            split_statements("SELECT $1; SELECT $2"),
            vec!["SELECT $1", "SELECT $2"]
        );
        // An empty-tag body with no internal `;` still round-trips as one statement.
        assert_eq!(
            split_statements("CREATE FUNCTION g() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$"),
            vec!["CREATE FUNCTION g() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$"]
        );
    }
}
