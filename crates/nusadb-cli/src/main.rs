//! nusa-cli — interactive SQL shell for NusaDB.
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
    META_HELP, Meta, OutputFormat, collect_result, format_result, handshake, is_complete_statement,
    parse_meta, split_statements, strip_terminator, tls_client_config,
};
use nusadb_wire::{Connection, FrontendMessage};
use rustls::pki_types::ServerName;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

/// nusa-cli — interactive SQL shell for NusaDB.
#[derive(Debug, Parser)]
#[command(name = "nusa-cli", version, about)]
struct Args {
    /// Server to connect to.
    #[arg(long, default_value = "127.0.0.1:5678")]
    host: String,

    /// User to connect as.
    #[arg(short, long, default_value = "nusa-root")]
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

/// Where to persist command history (`~/.nusa_history`), or `None` if the home dir is unknown.
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".nusa_history"))
}

/// Run a batch of `;`-separated statements, printing each result in `format`. Server errors are
/// printed and do not stop the batch.
async fn run_batch<S>(
    conn: &mut Connection<S>,
    sql: &str,
    format: OutputFormat,
) -> std::io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    for stmt in split_statements(sql) {
        for line in format_result(&collect_result(conn, &stmt).await?, format) {
            println!("{line}");
        }
    }
    Ok(())
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
    // secret off the command line), then to the canonical default `nusa-root`. Only used if the
    // server requests authentication (a trust-on-startup server ignores it).
    let password = args
        .password
        .clone()
        .or_else(|| std::env::var("NUSADB_PASSWORD").ok())
        .or_else(|| Some("nusa-root".to_owned()));
    handshake(&mut conn, &args.user, &args.database, password.as_deref()).await?;
    if interactive {
        let scheme = if args.tls || args.tls_ca.is_some() {
            " over TLS"
        } else {
            ""
        };
        println!("connected to {} as {}{scheme}", args.host, args.user);
    }

    if let Some(command) = &args.command {
        run_batch(&mut conn, command, args.format).await?;
    } else if let Some(path) = &args.file {
        let body = std::fs::read_to_string(path)?;
        run_batch(&mut conn, &body, args.format).await?;
    } else {
        repl(&mut conn, args.format, &args.database).await?;
    }

    conn.write_frame(&FrontendMessage::Terminate.encode()?)
        .await?;
    // The connection drops right after: force the queued frame onto the wire.
    conn.flush_now().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let interactive = args.command.is_none() && args.file.is_none();
    if interactive {
        println!("nusa-cli (NusaDB) — type \\q to quit");
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
