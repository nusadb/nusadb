//! nusadb-cli — interactive SQL shell for NusaDB.
//!
//! Connects to a `nusadb-server` over the Nusa Wire Protocol, performs the Startup handshake,
//! then either runs a batch (`--command` / `--file`) and exits, or starts a `rustyline`
//! REPL: statements are assembled across continuation lines until `;`-terminated, with
//! line editing and a persistent history. Results render in the chosen `--format`
//! (aligned/expanded/csv/json). Backslash dot-commands are handled client-side:
//! `\dt`/`\d` list tables, `\d NAME` describes a table, `\l` lists databases, `\?` shows help.
//! `\q`, `\quit`, `quit`, or EOF (Ctrl-D) ends the session; Ctrl-C abandons the statement typed.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use nusadb_cli::{
    CopyIo, META_HELP, Meta, OutputFormat, collect_result, collect_result_with_copy, format_result,
    handshake, is_complete_statement, parse_meta, split_statements, strip_terminator,
    tls_client_config,
};
use nusadb_wire::{Connection, FrontendMessage};
use rustls::pki_types::ServerName;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

/// nusadb-cli — interactive SQL shell for NusaDB.
#[derive(Debug, Parser)]
#[command(name = "nusadb-cli", version, about)]
struct Args {
    /// Server to connect to.
    #[arg(long, default_value = "127.0.0.1:5678")]
    host: String,

    /// User to connect as.
    #[arg(short, long, default_value = "nusadb-root")]
    user: String,

    /// Database to open.
    #[arg(short, long, default_value = "nusadb")]
    database: String,

    /// Password for SCRAM-SHA-256 authentication, required when the server was started with
    /// `--auth-user`. Prefer the `NUSADB_PASSWORD` environment variable, which takes effect when
    /// this flag is omitted, to keep the secret out of the process list and shell history.
    #[arg(short = 'W', long)]
    password: Option<String>,

    /// Run a single batch of SQL and exit (statements separated by `;`).
    #[arg(short, long, conflicts_with = "file")]
    command: Option<String>,

    /// Run the SQL in a file and exit (statements separated by `;`).
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Output format: aligned, expanded, csv, or json.
    #[arg(short = 'F', long, default_value = "aligned")]
    format: OutputFormat,

    /// Connect using TLS. Requires `--tls-ca` (there is no system trust store).
    #[arg(long)]
    tls: bool,

    /// PEM certificate to trust for TLS — a self-signed cert or private CA. Implies `--tls`.
    #[arg(long, value_name = "PATH")]
    tls_ca: Option<PathBuf>,

    /// Server name to verify the certificate against (default: `--host` without its port).
    #[arg(long, value_name = "NAME")]
    tls_domain: Option<String>,
}

/// The host portion of `host:port` (the name a TLS certificate is verified against).
fn host_name(host: &str) -> &str {
    host.rsplit_once(':').map_or(host, |(name, _port)| name)
}

/// Where to persist command history (`~/.nusadb_history`), or `None` if the home dir is unknown.
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".nusadb_history"))
}

/// Drop a leading byte-order mark from SQL text.
///
/// An editor on Windows normally writes one, and it is not SQL — left in place it fails on the very
/// first character. Only offset zero is considered, and that position cannot be inside a string
/// literal, so the same character appearing in a value is untouched.
fn strip_bom(sql: &str) -> &str {
    sql.strip_prefix('\u{feff}').unwrap_or(sql)
}

/// Run a batch of `;`-separated statements, printing each result in `format`. A server error is
/// printed to stderr and does not stop the batch — but it is not forgotten either: the return
/// value says whether every statement succeeded, so the process can exit non-zero and a caller
/// script does not sail past a failed load believing it worked.
///
/// A `COPY … FROM STDIN` reads the process's standard input, and a `COPY … TO STDOUT` writes to
/// standard output, so the shell forms work as they read:
///
/// ```text
/// nusadb-cli -c "COPY t FROM STDIN" < rows.tsv
/// nusadb-cli -c "COPY t TO STDOUT"  > rows.tsv
/// ```
///
/// Both batch forms read their SQL from elsewhere — the command line or a file — so standard input
/// is free to carry the data either way. The interactive REPL is the exception: there stdin is the
/// user's keyboard, so a `COPY` typed at the prompt is refused rather than fed the session itself.
async fn run_batch<S>(
    conn: &mut Connection<S>,
    sql: &str,
    format: OutputFormat,
) -> std::io::Result<bool>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut all_ok = true;
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    // One redirect feeds one load: after a COPY has drained stdin to EOF, a later one in the same
    // batch would read nothing and report `COPY 0` as if it had succeeded. Withhold the stream so
    // it is refused out loud instead.
    let mut stdin_unread = true;
    for stmt in split_statements(sql) {
        let source: Option<&mut (dyn AsyncRead + Unpin + Send)> =
            if stdin_unread { Some(&mut stdin) } else { None };
        let copy = CopyIo {
            source,
            sink: Some(&mut stdout),
        };
        let result = collect_result_with_copy(conn, &stmt, copy).await?;
        stdin_unread &= !result.copied_in;
        all_ok &= result.error.is_none();
        for line in format_result(&result, format) {
            // A statement that streamed rows to stdout must not have its tag land there too:
            // `COPY 3` appended to the exported rows makes the file fail to load back. An error
            // line goes to stderr for the same reason — it is diagnostics, not data.
            if result.copied_out || result.error.is_some() {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
        }
    }
    Ok(all_ok)
}

/// The interactive `rustyline` REPL.
async fn repl<S>(
    conn: &mut Connection<S>,
    format: OutputFormat,
    database: &str,
) -> Result<(), ReadlineError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // rustyline gives line editing + history (up/down, Ctrl-R) and reads blocking — fine for an
    // interactive REPL that does nothing else while it waits for the user to type.
    let mut rl = DefaultEditor::new()?;
    let history = history_path();
    if let Some(path) = &history {
        let _ = rl.load_history(path); // absent on first run — not an error
    }

    'session: loop {
        // Assemble one statement across continuation lines until it is `;`-terminated.
        let mut buf = String::new();
        let sql_owned = loop {
            let prompt = if buf.is_empty() { "nusa> " } else { "  ...> " };
            match rl.readline(prompt) {
                Ok(line) => {
                    // A meta-command (`\q`, `\dt`, `\d`, `\l`, `\?`) is only honoured as a
                    // standalone first line.
                    if buf.is_empty() {
                        if line.trim().is_empty() {
                            continue; // empty prompt → re-prompt fresh
                        }
                        match parse_meta(&line) {
                            Some(Meta::Quit) => break 'session,
                            Some(Meta::Help) => {
                                println!("{META_HELP}");
                                continue;
                            },
                            Some(Meta::Databases) => {
                                println!("{database}");
                                continue;
                            },
                            Some(Meta::Sql(sql)) => {
                                match collect_result(conn, &sql).await {
                                    Ok(result) => {
                                        for out in format_result(&result, format) {
                                            println!("{out}");
                                        }
                                    },
                                    Err(e) => eprintln!("error: {e}"),
                                }
                                continue;
                            },
                            None => {},
                        }
                    }
                    buf.push_str(&line);
                    buf.push('\n');
                    if is_complete_statement(&buf) {
                        break strip_terminator(&buf).to_owned();
                    }
                },
                // Ctrl-C abandons the statement in progress and returns to a fresh prompt.
                Err(ReadlineError::Interrupted) => continue 'session,
                // Ctrl-D ends the session.
                Err(ReadlineError::Eof) => break 'session,
                Err(e) => {
                    eprintln!("error: {e}");
                    break 'session;
                },
            }
        };

        if sql_owned.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(buf.trim());
        match collect_result(conn, &sql_owned).await {
            Ok(result) => {
                for line in format_result(&result, format) {
                    println!("{line}");
                }
            },
            Err(e) => eprintln!("error: {e}"),
        }
    }

    if let Some(path) = &history {
        let _ = rl.save_history(path);
    }
    Ok(())
}

/// Run the full session over an established (plain or TLS) connection: handshake, then a batch
/// (`--command`/`--file`) or the interactive REPL, then `Terminate`. Generic over the stream so
/// the same path serves plaintext [`TcpStream`] and a `tokio-rustls` TLS stream.
async fn run_session<S>(
    mut conn: Connection<S>,
    args: &Args,
    interactive: bool,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // The password comes from --password, falling back to NUSADB_PASSWORD (preferred — keeps the
    // secret off the command line), then to the canonical default `nusadb-root`. Only used if the
    // server requests authentication (a trust-on-startup server ignores it).
    let password = args
        .password
        .clone()
        .or_else(|| std::env::var("NUSADB_PASSWORD").ok())
        .or_else(|| Some("nusadb-root".to_owned()));
    handshake(&mut conn, &args.user, &args.database, password.as_deref()).await?;
    if interactive {
        let scheme = if args.tls || args.tls_ca.is_some() {
            " over TLS"
        } else {
            ""
        };
        println!("connected to {} as {}{scheme}", args.host, args.user);
    }

    let mut batch_ok = true;
    if let Some(command) = &args.command {
        batch_ok = run_batch(&mut conn, strip_bom(command), args.format).await?;
    } else if let Some(path) = &args.file {
        let body = std::fs::read_to_string(path)?;
        batch_ok = run_batch(&mut conn, strip_bom(&body), args.format).await?;
    } else {
        repl(&mut conn, args.format, &args.database).await?;
    }

    conn.write_frame(&FrontendMessage::Terminate.encode()?)
        .await?;
    // The connection drops right after: force the queued frame onto the wire.
    conn.flush_now().await?;
    if !batch_ok {
        // Every result was printed; the exit status must still say the batch was not clean, or a
        // calling script keeps going as if its load had worked.
        std::process::exit(1);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let interactive = args.command.is_none() && args.file.is_none();
    if interactive {
        println!("nusadb-cli (NusaDB) — type \\q to quit");
    }

    let tcp = TcpStream::connect(&args.host).await?;
    // Disable Nagle's algorithm: the wire protocol is request/response, so coalescing would add a
    // round-trip delay to each interactive query. Non-fatal if it fails.
    if let Err(e) = tcp.set_nodelay(true) {
        tracing::debug!("set_nodelay failed on client connection: {e}");
    }

    // TLS is requested by `--tls` or implicitly by supplying `--tls-ca`. With no system trust
    // store, a trusted certificate (`--tls-ca`) is required.
    if args.tls || args.tls_ca.is_some() {
        let ca_path = args
            .tls_ca
            .as_ref()
            .ok_or("TLS requested but no --tls-ca certificate was provided")?;
        let ca_pem = std::fs::read(ca_path)?;
        let connector = TlsConnector::from(Arc::new(tls_client_config(&ca_pem)?));
        let domain = args
            .tls_domain
            .clone()
            .unwrap_or_else(|| host_name(&args.host).to_owned());
        let server_name = ServerName::try_from(domain)?;
        let stream = connector.connect(server_name, tcp).await?;
        run_session(Connection::new(stream), &args, interactive).await
    } else {
        run_session(Connection::new(tcp), &args, interactive).await
    }
}

#[cfg(test)]
mod bom_tests {
    use super::strip_bom;

    #[test]
    fn a_leading_mark_goes_and_nothing_else_does() {
        assert_eq!(strip_bom("\u{feff}SELECT 1"), "SELECT 1");
        assert_eq!(strip_bom("SELECT 1"), "SELECT 1");
        // Only the first one: a second is real text, and one inside a value is left alone.
        assert_eq!(strip_bom("\u{feff}\u{feff}SELECT 1"), "\u{feff}SELECT 1");
        assert_eq!(strip_bom("SELECT '\u{feff}'"), "SELECT '\u{feff}'");
        assert_eq!(strip_bom(""), "");
    }
}

#[cfg(test)]
mod batch_exit_tests {
    use std::sync::Arc;

    use nusadb_btree::BtreeEngine;
    use nusadb_cli::handshake;
    use nusadb_core::StorageEngine;
    use nusadb_wire::{Connection, serve};
    use tokio::net::{TcpListener, TcpStream};

    use super::{OutputFormat, run_batch};

    /// The bug this pins: a batch whose statement failed used to exit 0, so a caller script — one
    /// loading data in steps under `set -e` — sailed on believing the load worked.
    #[tokio::test]
    async fn a_batch_with_a_failed_statement_reports_not_clean() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
        let server = tokio::spawn(serve(listener, engine));

        let mut conn = Connection::new(TcpStream::connect(addr).await.unwrap());
        handshake(&mut conn, "u", "nusadb", None).await.unwrap();

        // A clean batch is clean.
        let ok = run_batch(
            &mut conn,
            "CREATE TABLE t (id INT); INSERT INTO t VALUES (1)",
            OutputFormat::Aligned,
        )
        .await
        .unwrap();
        assert!(ok, "a batch of successful statements must report clean");

        // One failing statement marks the whole batch, and later statements still ran.
        let ok = run_batch(
            &mut conn,
            "INSERT INTO t VALUES (2); SELECT nope FROM missing; INSERT INTO t VALUES (3)",
            OutputFormat::Aligned,
        )
        .await
        .unwrap();
        assert!(
            !ok,
            "a batch containing a server error must report not-clean"
        );
        let clean = run_batch(&mut conn, "SELECT count(*) FROM t", OutputFormat::Aligned)
            .await
            .unwrap();
        assert!(clean, "the session stays usable after a failed statement");
        // And the statements around the failure really applied: rows 1, 2 and 3 all landed.
        let count = nusadb_cli::collect_result(&mut conn, "SELECT count(*) FROM t")
            .await
            .unwrap();
        assert_eq!(count.rows, vec![vec![Some(b"3".to_vec())]]);

        server.abort();
    }
}
