//! NusaDB server binary entry point.
//!
//! The composition root: opens a durable `nusadb-btree` engine per database over the data
//! directory, binds a TCP listener, and serves clients with the Nusa Wire Protocol via
//! [`nusadb_wire::serve`]. The wire layer is async; query execution bridges to the synchronous
//! engine inside `serve`.

use std::sync::Arc;

/// Process-wide allocator. With `--features mimalloc` the whole server runs on mimalloc
/// (faster, lower-fragmentation for the engine's allocation churn); otherwise the platform
/// system allocator is used. Feature-gated so the default build pulls in no C allocator.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::Duration;

mod database_manager;
mod tuning;

use clap::Parser;
use nusadb_wire::{AuthStore, Metrics, ServerConfig, serve_cluster_with_shutdown};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// NusaDB server.
#[derive(Debug, Parser)]
#[command(name = "nusadb-server", version, about)]
struct Args {
    /// TCP listen address.
    #[arg(long, default_value = "0.0.0.0:5678")]
    listen: String,

    /// Data directory (holds the durable WAL).
    #[arg(long, default_value = "./data")]
    data_dir: String,

    /// Storage engine for every database in the cluster: the clustered B-link/B+tree engine
    /// (ADR 008) — the sole engine (owner decision, 2026-07-08). `btree` is the only accepted
    /// value; the flag remains so existing `--storage-engine btree` invocations keep working,
    /// while the removed `lsm` value fails loudly at parse. A data directory written by the
    /// removed lsm engine is refused at open with a migration hint.
    #[arg(long, value_enum, default_value = "btree")]
    storage_engine: database_manager::EngineKind,

    /// Close a connection idle for this many seconds (0 = no idle timeout).
    #[arg(long, default_value_t = 0)]
    idle_timeout: u64,

    /// On Ctrl-C, wait up to this many seconds for in-flight connections to drain before
    /// aborting them (0 = wait indefinitely).
    #[arg(long, default_value_t = 30)]
    drain_timeout: u64,

    /// Maximum concurrent connections; excess connections queue until a slot frees (0 = unlimited).
    /// Defaults to 25 (small-safe Tier-1): an unbounded default is the fastest path to OOM
    /// on a 2 GB host, since each connection costs memory and a connection storm multiplies it.
    /// Raise it on a larger host; an app that needs many connections should front the server with a
    /// pooler.
    #[arg(long, default_value_t = 25)]
    max_connections: usize,

    /// Refuse a connection past --max-connections immediately with SQLSTATE 53300
    /// (`too many clients already`, the reference engine's behaviour) instead of queueing it until a slot frees
    /// (the default). Pick this when clients run behind a pool that should retry/back off on an
    /// honest error rather than hang in the accept backlog during a connection storm. (P-CONNCAP)
    #[arg(long, default_value_t = false)]
    reject_excess_connections: bool,

    /// Total memory budget in bytes that drives RAM-aware auto-tuning: the engine derives
    /// a bounded per-query `work_mem`, an engine footprint cap, and
    /// turns spill-to-disk on by default, so a large query degrades gracefully instead of OOM-killing
    /// the process. `0` (the default) auto-detects the budget on Linux as `min(host RAM, cgroup
    /// limit)` — container-aware, so a cloud free-tier limit is honoured; on a non-Linux host with no
    /// explicit value, auto-tuning is skipped and only explicit `--work-mem`/`--spill-dir` apply. Set
    /// a positive value to pin the budget. Explicit `--work-mem`/`--spill-dir` override the derived
    /// values.
    #[arg(long, default_value_t = 0)]
    mem_budget: usize,

    /// Per-query work-memory budget in bytes; a query that materializes more than this in one
    /// executor stage (a big sort / aggregate / join) is failed with an honest error rather than
    /// OOM-killing the server (0 = unlimited, the default). Set it on a constrained host to bound
    /// any single query's memory.
    #[arg(long, default_value_t = 0)]
    work_mem: usize,

    /// Directory for transient spill-to-disk files (external sort / hash join). When set, blocking
    /// operators over more than the spill threshold stream the overflow to this directory instead of
    /// failing; unset (the default) keeps the in-memory `work_mem` behavior. Stale files from a prior
    /// crash are swept on startup.
    #[arg(long)]
    spill_dir: Option<String>,

    /// Serve Prometheus metrics on this address (e.g. `127.0.0.1:9100`); disabled if unset.
    #[arg(long)]
    metrics_listen: Option<String>,

    /// PEM certificate chain for TLS; enables TLS when set (requires `--tls-key`).
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<String>,

    /// PEM private key for TLS (requires `--tls-cert`).
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<String>,

    /// PEM CA certificate for mutual TLS: when set, every client must present a certificate signed
    /// by this CA (requires `--tls-cert`/`--tls-key`). (mTLS)
    #[arg(long, requires = "tls_cert")]
    tls_client_ca: Option<String>,

    /// Require SCRAM-SHA-256 authentication for a `user:password` pair (repeatable). When any are
    /// given, every connection must authenticate; otherwise auth is trust-on-startup.
    #[arg(long = "auth-user", value_name = "USER:PASSWORD")]
    auth_users: Vec<String>,

    /// Cancel any statement that runs longer than this many seconds (0 = no limit). Enforced
    /// cooperatively, so the cap is approximate.
    #[arg(long, default_value_t = 0)]
    statement_timeout: u64,

    /// Drop a connection that has not finished the Startup + authentication handshake within this
    /// many seconds (0 = no limit, not recommended). Always applies, independent of
    /// `--idle-timeout`, so an unauthenticated client cannot hold a slot by stalling the handshake
    /// (slowloris). Defaults to 60.
    #[arg(long, default_value_t = 60)]
    handshake_timeout: u64,

    /// Maximum cumulative bytes buffered for a single `COPY ... FROM STDIN`; a larger load is
    /// aborted with an error rather than buffered without bound (which could OOM the server). Left
    /// unset, it auto-derives from the memory budget (about 20%, capped at 1 GiB) so the transient
    /// buffer plus the resident store cannot exceed RAM on a constrained host; with no budget it
    /// falls back to 1 GiB. Pass an explicit value to pin it (raise it on a larger host or split the
    /// load), or `0` for unbounded (not recommended).
    #[arg(long)]
    copy_max_bytes: Option<usize>,

    /// Maximum cumulative bytes one transaction may buffer for its uncommitted writes before it is
    /// failed with an honest out-of-memory error (SQLSTATE XX000) instead of being allowed to grow
    /// the process until the host OOM-kills it. Bounds a runaway single-transaction bulk
    /// `INSERT`/`UPDATE`/`DELETE` (including `INSERT ... SELECT`), so one client cannot take the
    /// whole server down. `0` (the default) derives a protective ceiling of 25% of the memory
    /// budget (floor 128 MiB) when a budget is known — the same budget `--work-mem` uses — and is
    /// unlimited only when no budget can be determined (a non-Linux host with no `--mem-budget`). A
    /// non-zero value overrides the derived default. This is a *per-transaction* bound, not a global
    /// one. Complements `--work-mem` (per-query executor memory) and `--copy-max-bytes` (single COPY
    /// load).
    #[arg(long, default_value_t = 0)]
    max_txn_write_bytes: u64,

    /// Global resident-memory ceiling (bytes) for each database's in-memory page store. Once its
    /// total footprint reaches this, a row `INSERT` is failed with an honest out-of-memory error
    /// instead of the store growing until the host OOM-kills the server. Unlike
    /// `--max-txn-write-bytes` (one in-flight transaction) this bounds *committed-resident* data
    /// across the whole store — the streamed bulk load that accumulates past the per-transaction
    /// ceiling. `DELETE`/`TRUNCATE` stay available at the ceiling so space can be freed. `0` (the
    /// default) derives a protective ceiling from the memory budget (floor 256 MiB) when a budget is
    /// known — the engine's page share, reduced because the meter counts logical page bytes while the
    /// real footprint runs larger, so the ceiling trips before the OS out-of-memory killer would —
    /// and is unlimited only when no budget can be determined (a non-Linux host with no
    /// `--mem-budget`). A non-zero value overrides the derived default.
    #[arg(long, default_value_t = 0)]
    max_resident_bytes: u64,

    /// How often (seconds) the background auto-analyze scheduler sweeps each database for tables whose
    /// planner statistics have gone stale and re-`ANALYZE`s them, so the cost-based optimizer stays
    /// accurate on a live database without a manual `ANALYZE`. `0` disables auto-analyze. Defaults to
    /// 60 seconds.
    #[arg(long, default_value_t = 60)]
    autoanalyze_interval: u64,

    /// Auto-analyze scale factor: a table is re-analyzed once its write churn since the last analyze
    /// exceeds `--autoanalyze-threshold + this * row_count` — a fraction of the table's size.
    #[arg(long, default_value_t = 0.1)]
    autoanalyze_scale: f64,

    /// Auto-analyze churn threshold: the constant floor added to the scaled part of the auto-analyze
    /// trigger (see `--autoanalyze-scale`), so small tables are not re-analyzed on trivial churn.
    #[arg(long, default_value_t = 50)]
    autoanalyze_threshold: u64,
}

/// `0` means "disabled / unbounded"; any other value is that many seconds.
fn secs(n: u64) -> Option<Duration> {
    (n > 0).then(|| Duration::from_secs(n))
}

/// Remove spill files orphaned by a prior crash. Spilling operators clean up their own files
/// via RAII, so anything left in the scratch dir on startup is a leak from a hard crash. Only files
/// whose name carries the `nusadb-spill-` prefix are touched, so an operator-shared directory keeps
/// any unrelated contents.
fn sweep_stale_spill_files(dir: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("nusadb-spill-")
        {
            continue;
        }
        if let Err(e) = std::fs::remove_file(entry.path()) {
            tracing::warn!(path = %entry.path().display(), error = %e, "could not sweep stale spill file");
        }
    }
}

/// Resolve the effective `USER:PASSWORD` credential pairs. `--auth-user` pairs take
/// precedence; with none, fall back to the `NUSADB_USER` + `NUSADB_PASSWORD` env pair (so a
/// container can require auth without baking a secret into the image). No pairs and no env → empty
/// (trust-on-startup). Setting only one of the two env values is a configuration error. Pure (env is
/// read by the caller) so the precedence/error logic is unit-testable.
fn resolve_auth_pairs(
    cli_pairs: &[String],
    env_user: Option<String>,
    env_password: Option<String>,
) -> Result<Vec<String>, String> {
    if !cli_pairs.is_empty() {
        return Ok(cli_pairs.to_vec());
    }
    match (env_user, env_password) {
        (Some(user), Some(password)) => Ok(vec![format!("{user}:{password}")]),
        (Some(_), None) | (None, Some(_)) => {
            Err("set both NUSADB_USER and NUSADB_PASSWORD (or neither), not just one".to_owned())
        },
        (None, None) => Ok(Vec::new()),
    }
}

/// Build the optional SCRAM credential store from the resolved credential pairs (see
/// [`resolve_auth_pairs`]). Empty → `None` (trust-on-startup).
fn build_auth(pairs: &[String]) -> Result<Option<Arc<AuthStore>>, Box<dyn std::error::Error>> {
    let pairs = resolve_auth_pairs(
        pairs,
        std::env::var("NUSADB_USER").ok(),
        std::env::var("NUSADB_PASSWORD").ok(),
    )?;
    if pairs.is_empty() {
        return Ok(None);
    }
    let mut creds = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let (user, password) = pair
            .split_once(':')
            .ok_or_else(|| format!("--auth-user must be USER:PASSWORD, got `{pair}`"))?;
        creds.push((user.to_owned(), password.to_owned()));
    }
    Ok(Some(Arc::new(AuthStore::from_passwords(creds)?)))
}

/// A tiny Prometheus scrape endpoint: respond to any request with the current metrics in the text
/// exposition format. Runs until the task is aborted (on server shutdown).
async fn serve_metrics(listener: TcpListener, metrics: std::sync::Arc<Metrics>) {
    loop {
        let mut socket = match listener.accept().await {
            Ok((socket, _peer)) => socket,
            Err(e) => {
                // Back off briefly so a persistent accept error (e.g. fd exhaustion) does not spin
                // this loop at 100% CPU (round-3 audit).
                tracing::warn!("metrics endpoint accept error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            },
        };
        let metrics = std::sync::Arc::clone(&metrics);
        tokio::spawn(async move {
            // Bound the whole exchange with a timeout: an unauthenticated client that connects
            // and never finishes sending (slowloris) must not park this task — and leak the socket /
            // FD — indefinitely. A real scraper sends a tiny `GET /metrics` and reads a small body
            // well within this window.
            let exchange = async move {
                // Best-effort read of the request; its contents are ignored.
                let mut scratch = [0u8; 1024];
                let _ = socket.read(&mut scratch).await;
                let body = metrics.render_prometheus();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            };
            if tokio::time::timeout(Duration::from_secs(10), exchange)
                .await
                .is_err()
            {
                tracing::warn!("metrics request timed out");
            }
        });
    }
}

/// Apply the **process-global** memory configuration once at startup.
///
/// Picks a memory budget — an explicit `--mem-budget`, else the detected container/host limit on
/// Linux — and derives safe defaults from it ([`tuning::derive`]): a bounded per-query `work_mem`
/// and spill-to-disk ON to `<data-dir>/tmp` (including the one-time sweep of stale spill files), so
/// a large query degrades gracefully instead of OOM-killing the process. Explicit `--work-mem` /
/// `--spill-dir` override the derived values. With no explicit budget and no detection (non-Linux),
/// auto-tuning is skipped and only explicit flags apply.
///
/// Returns the resolved per-transaction write ceiling to hand every database's engine: an explicit
/// `--max-txn-write-bytes` (non-zero) wins; otherwise, when a budget is known, the protective 25%
/// default ([`tuning::derive`]); with no budget, `None` (unlimited), consistent with auto-tuning
/// being skipped. This is the composition root's single source of truth for the ceiling.
/// The resolved engine memory ceilings the composition root hands every database's engine: the
/// per-transaction write ceiling and the global resident-memory ceiling. `None` on either means
/// unlimited (no budget and no explicit flag).
#[allow(
    clippy::struct_field_names,
    reason = "each field carries a byte cap and mirrors the name of the `--*-bytes` flag it resolves; \
              the shared suffix is the point, not noise"
)]
struct MemoryCeilings {
    max_txn_write_bytes: Option<u64>,
    max_resident_bytes: Option<u64>,
    /// Cap for one `COPY ... FROM STDIN` wire buffer: `None` = unbounded (explicit `0`), else the
    /// byte cap. Auto-derived from the budget when the flag is unset (see [`tuning::DerivedKnobs`]).
    copy_max_bytes: Option<u64>,
}

/// Resolve the single-`COPY` buffer cap from the flag and an optional budget-derived default. An
/// explicit `--copy-max-bytes` wins verbatim (`Some(0)` = unbounded → `None`); when the flag is unset
/// the derived value applies, and with no budget it falls back to the historical 1 GiB.
fn resolve_copy_max_bytes(flag: Option<usize>, derived: Option<u64>) -> Option<u64> {
    match flag {
        Some(0) => None, // explicit opt-out: unbounded (not recommended)
        Some(explicit) => Some(explicit as u64), // explicit cap wins, even over a derived default
        None => derived.or(Some(1 << 30)), // unset: budget-derived, else the historical 1 GiB
    }
}

fn apply_memory_config(args: &Args) -> Result<MemoryCeilings, Box<dyn std::error::Error>> {
    let budget = if args.mem_budget != 0 {
        Some(args.mem_budget)
    } else {
        tuning::detect_budget()
    };
    let Some(budget) = budget else {
        // No budget (non-Linux without --mem-budget): legacy behaviour — only explicit flags apply.
        tracing::info!(
            "auto-tuning off (no memory budget detected); pass --mem-budget to enable on this host"
        );
        if args.work_mem != 0 {
            nusadb_sql::set_work_mem(args.work_mem);
            tracing::info!(work_mem = args.work_mem, "per-query work_mem budget set");
        }
        if let Some(spill_dir) = &args.spill_dir {
            std::fs::create_dir_all(spill_dir)?;
            sweep_stale_spill_files(spill_dir);
            // Without an explicit work_mem the threshold defaults to 64 MiB so spill actually triggers.
            let threshold = if args.work_mem != 0 {
                args.work_mem
            } else {
                64 * 1024 * 1024
            };
            nusadb_sql::set_spill_config(Some(nusadb_sql::SpillConfig {
                dir: std::path::PathBuf::from(spill_dir),
                threshold_bytes: threshold,
            }));
            tracing::info!(spill_dir, threshold, "spill-to-disk enabled");
        }
        // No budget to derive from: each ceiling honours only its explicit flag, else stays unlimited.
        return Ok(MemoryCeilings {
            max_txn_write_bytes: (args.max_txn_write_bytes != 0)
                .then_some(args.max_txn_write_bytes),
            max_resident_bytes: (args.max_resident_bytes != 0).then_some(args.max_resident_bytes),
            // No budget to derive from: honour an explicit flag, else the historical 1 GiB fallback.
            copy_max_bytes: resolve_copy_max_bytes(args.copy_max_bytes, None),
        });
    };

    let knobs = tuning::derive(budget, args.max_connections);
    if tuning::work_pool_overcommits(knobs, budget, args.max_connections) {
        tracing::warn!(
            budget,
            max_connections = args.max_connections,
            work_mem = knobs.work_mem,
            "memory budget is small for this many connections; concurrent large queries may exceed \
             it — lower --max-connections or raise --mem-budget"
        );
    }
    // Per-query work_mem (also the spill threshold): an explicit flag wins over the derived value.
    let work_mem = if args.work_mem != 0 {
        args.work_mem
    } else {
        knobs.work_mem
    };
    nusadb_sql::set_work_mem(work_mem);
    // Spill ON by default: an explicit --spill-dir wins, else `<data-dir>/tmp`. Failing to prepare
    // the *default* scratch dir is non-fatal (log + skip spill); an explicit --spill-dir failure is
    // fatal (the operator asked for it).
    let spill_dir: Option<std::path::PathBuf> = if let Some(dir) = &args.spill_dir {
        std::fs::create_dir_all(dir)?;
        Some(std::path::PathBuf::from(dir))
    } else {
        let dir = std::path::Path::new(&args.data_dir).join("tmp");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(path = %dir.display(), error = %e,
                "could not create default spill dir; spill disabled");
            None
        } else {
            Some(dir)
        }
    };
    if let Some(dir) = spill_dir {
        sweep_stale_spill_files(dir.to_string_lossy().as_ref());
        nusadb_sql::set_spill_config(Some(nusadb_sql::SpillConfig {
            dir,
            threshold_bytes: work_mem,
        }));
    }
    // Per-transaction write ceiling: an explicit flag wins; otherwise the protective derived
    // default (25% of budget). The engine fails a transaction that buffers more than this loudly,
    // so one runaway bulk write cannot OOM-kill the server and take every client down with it.
    let max_txn_write_bytes = if args.max_txn_write_bytes != 0 {
        args.max_txn_write_bytes
    } else {
        knobs.max_txn_write_bytes as u64
    };
    // Global resident ceiling: an explicit flag wins; otherwise the protective derived default (the
    // page share of the budget, reduced for the real-vs-logical footprint gap). The engine fails a
    // row insert that would grow the store past this loudly, so a bulk load bigger than RAM degrades
    // to an error instead of an OS OOM-kill of the whole server.
    let max_resident_bytes = if args.max_resident_bytes != 0 {
        args.max_resident_bytes
    } else {
        knobs.max_resident_bytes as u64
    };
    // Single-COPY buffer cap: an explicit flag wins; otherwise the derived value scaled to the budget
    // (about 20%, capped at 1 GiB), so the transient wire buffer plus the resident store leave RAM
    // headroom instead of together overrunning a constrained host.
    let copy_max_bytes =
        resolve_copy_max_bytes(args.copy_max_bytes, Some(knobs.max_copy_bytes as u64));
    tracing::info!(
        budget,
        work_mem,
        max_txn_write_bytes,
        max_resident_bytes,
        copy_max_bytes = copy_max_bytes.unwrap_or(0),
        "RAM-aware auto-tuning applied"
    );
    Ok(MemoryCeilings {
        max_txn_write_bytes: Some(max_txn_write_bytes),
        max_resident_bytes: Some(max_resident_bytes),
        copy_max_bytes,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    std::fs::create_dir_all(&args.data_dir)?;
    // Apply process-global memory config once (per-query work_mem + spill-to-disk), and resolve the
    // per-transaction and global resident write ceilings every database's engine will enforce.
    let ceilings = apply_memory_config(&args)?;
    // Auto-analyze keeps the planner's statistics fresh on a live database (0 interval = off).
    let autoanalyze = database_manager::AutoAnalyzeConfig {
        interval: secs(args.autoanalyze_interval),
        scale: args.autoanalyze_scale,
        base: args.autoanalyze_threshold,
    };
    // The physical multi-database cluster: each database is its own engine under `base/<db>/`,
    // bootstrapping the default database on a fresh data directory. Dead-version reclamation is
    // the per-database purge scheduler the manager wires as each engine opens.
    let cluster: Arc<dyn nusadb_wire::DatabaseCluster> =
        Arc::new(database_manager::DatabaseManager::open(
            &args.data_dir,
            nusadb_wire::cluster::DEFAULT_DATABASE,
            ceilings.max_txn_write_bytes,
            ceilings.max_resident_bytes,
            autoanalyze,
        )?);
    tracing::info!(
        data_dir = %args.data_dir,
        storage_engine = ?args.storage_engine,
        databases = ?cluster.list(),
        "opened durable database cluster catalog; per-database engines open lazily on first \
         connection — a corrupt WAL is detected and refused then (logged at ERROR), not at startup"
    );

    let listener = TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, "nusadb-server listening");

    // TLS is enabled when both --tls-cert and --tls-key are given (clap's `requires` enforces the
    // pair). With --tls-client-ca it becomes mutual TLS (clients must present a CA-signed cert).
    // Build the rustls config up front so bad cert/key/CA material fails fast before accepting.
    let tls = match (&args.tls_cert, &args.tls_key) {
        (Some(cert), Some(key)) => {
            let config = if let Some(ca) = &args.tls_client_ca {
                let config = nusadb_wire::tls::server_config_mtls_from_files(
                    std::path::Path::new(cert),
                    std::path::Path::new(key),
                    std::path::Path::new(ca),
                )?;
                tracing::info!(cert = %cert, client_ca = %ca, "mutual TLS enabled");
                config
            } else {
                let config = nusadb_wire::tls::server_config_from_files(
                    std::path::Path::new(cert),
                    std::path::Path::new(key),
                )?;
                tracing::info!(cert = %cert, "TLS enabled");
                config
            };
            Some(config)
        },
        _ => None,
    };

    let metrics = Arc::new(Metrics::new());
    let auth = build_auth(&args.auth_users)?;
    if auth.is_none() {
        // Trust-on-startup accepts any connecting user without a password. Fine for local/dev/test,
        // but never silent in a shared or public deployment — say so loudly (the conventional
        // pattern of requiring a password via env/flag). Set --auth-user or
        // NUSADB_USER/NUSADB_PASSWORD to require SCRAM authentication.
        tracing::warn!(
            "running UNAUTHENTICATED (trust-on-startup): any client is accepted without a \
             password. For local/dev/testing only — set --auth-user USER:PASSWORD or the \
             NUSADB_USER/NUSADB_PASSWORD environment variables for any shared or public deployment."
        );
    }
    let config = ServerConfig {
        idle_timeout: secs(args.idle_timeout),
        drain_timeout: secs(args.drain_timeout),
        max_connections: (args.max_connections > 0).then_some(args.max_connections),
        reject_excess_connections: args.reject_excess_connections,
        metrics: Some(Arc::clone(&metrics)),
        tls,
        auth,
        statement_timeout: secs(args.statement_timeout),
        handshake_timeout: secs(args.handshake_timeout),
        copy_from_max_bytes: ceilings.copy_max_bytes.map(|b| b as usize),
    };

    // Optional Prometheus scrape endpoint; aborted when the server stops.
    let metrics_task = match &args.metrics_listen {
        Some(addr) => {
            let metrics_listener = TcpListener::bind(addr).await?;
            tracing::info!(metrics = %addr, "metrics endpoint listening");
            Some(tokio::spawn(serve_metrics(
                metrics_listener,
                Arc::clone(&metrics),
            )))
        },
        None => None,
    };

    // Graceful shutdown on Ctrl-C: stop accepting, drain in-flight connections, then exit.
    let shutdown = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to listen for Ctrl-C: {e}");
        } else {
            tracing::info!("Ctrl-C received — shutting down gracefully");
        }
    };
    serve_cluster_with_shutdown(listener, cluster, config, shutdown).await?;

    if let Some(task) = metrics_task {
        task.abort();
    }
    tracing::info!("server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::resolve_auth_pairs;

    #[test]
    fn cli_pairs_take_precedence_over_env() {
        let r = resolve_auth_pairs(
            &["admin:secret".to_owned()],
            Some("envuser".to_owned()),
            Some("envpass".to_owned()),
        )
        .unwrap();
        assert_eq!(r, vec!["admin:secret".to_owned()]);
    }

    #[test]
    fn env_pair_used_when_no_cli_pairs() {
        let r =
            resolve_auth_pairs(&[], Some("alice".to_owned()), Some("secret".to_owned())).unwrap();
        assert_eq!(r, vec!["alice:secret".to_owned()]);
    }

    #[test]
    fn only_one_env_var_set_is_an_error() {
        assert!(resolve_auth_pairs(&[], Some("alice".to_owned()), None).is_err());
        assert!(resolve_auth_pairs(&[], None, Some("secret".to_owned())).is_err());
    }

    #[test]
    fn no_credentials_means_trust_on_startup() {
        assert!(resolve_auth_pairs(&[], None, None).unwrap().is_empty());
    }

    #[test]
    fn password_may_contain_colons() {
        // The env pair is joined as user:password; build_auth's split_once(':') splits on the first
        // colon, so a password with further colons is preserved verbatim.
        let r = resolve_auth_pairs(&[], Some("u".to_owned()), Some("p:a:ss".to_owned())).unwrap();
        assert_eq!(r, vec!["u:p:a:ss".to_owned()]);
        let (user, password) = r[0].split_once(':').unwrap();
        assert_eq!(user, "u");
        assert_eq!(password, "p:a:ss");
    }
}
