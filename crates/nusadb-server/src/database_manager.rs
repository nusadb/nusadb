//! The physical multi-database manager.
//!
//! Lays out a cluster as
//!
//! ```text
//! <data-dir>/
//!   global/databases     ← cluster catalog: one database name per line
//!   base/<db>/btree.wal   ← one BtreeEngine per database
//! ```
//!
//! Each database is opened **lazily** (on first connection) to keep idle databases off the heap on
//! minimal hardware, and cached for the cluster's lifetime. A connection resolves its database name
//! to one engine and only ever touches that engine, so databases are physically isolated (separate
//! WAL / MVCC / recovery / backup) — the reason the physical model was chosen.

#![allow(
    clippy::redundant_pub_crate,
    reason = "this is a private module of the server binary; its items are `pub(crate)` so the \
              crate root (main.rs) can use them, which `unreachable_pub` requires — the two lints \
              are mutually exclusive here"
)]
#![allow(
    clippy::significant_drop_tightening,
    reason = "the state lock is held deliberately across the catalog mutation and its filesystem \
              effects (create/remove the database directory, persist the catalog) so a database's \
              registration and its storage stay consistent under concurrent cluster operations"
)]

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{ClusterError, DatabaseCluster, is_valid_database_name};

/// Which storage engine the cluster runs. The clustered B-link/B+tree engine (ADR 008) is the
/// sole engine (owner decision, 2026-07-08): the former `lsm` value is no longer accepted, so a
/// stale script passing it fails loudly at argument parsing instead of silently running the
/// wrong engine. Every database directory records its engine in an `engine` marker file; a
/// directory recorded (or inferred) as `lsm` is refused at open — its data needs a dump/restore
/// migration, never a silent cross-engine read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum EngineKind {
    /// The clustered B-link/B+tree `BtreeEngine` (ADR 008) — the only engine.
    Btree,
}

impl EngineKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Btree => "btree",
        }
    }
}

/// The cluster's mutable state behind one lock: the registered database names (the catalog) and the
/// lazily-opened engines.
struct ManagerState {
    /// Registered database names — the persisted cluster catalog. `BTreeSet` keeps `list()` sorted.
    databases: BTreeSet<String>,
    /// Engines opened so far, keyed by database name. Absent until the first connection opens one.
    /// Type-erased: the whole surface above this point is engine-agnostic by construction.
    engines: HashMap<String, Arc<dyn StorageEngine>>,
}

/// A cluster of physically-isolated databases, each a `BtreeEngine` under `base/<db>/`.
pub(crate) struct DatabaseManager {
    root: PathBuf,
    /// The exclusive lock on `<root>/global/cluster.lock`, held for the server's life so no second
    /// server runs on the same data directory. The operating system releases it when the process
    /// ends, however it ends.
    _cluster_lock: std::fs::File,
    default_name: String,
    /// Back-compat: a pre-multi-database data directory has its WAL at the root rather than under
    /// `base/`. When that layout is detected, the default database's engine stays at the root so
    /// existing data is preserved without a migration; other databases use `base/`. A root written
    /// by the removed `lsm` engine is refused at open by the engine marker, not silently read.
    legacy_root: bool,
    /// Per-transaction uncommitted-write ceiling applied to every database's engine as it opens
    /// (`--max-txn-write-bytes`). `None` (the default) leaves each engine unbounded, exactly as
    /// before the flag existed.
    max_txn_write_bytes: Option<u64>,
    /// Global resident-memory ceiling applied to every database's engine as it opens
    /// (`--max-resident-bytes`). `None` (the default) leaves each engine's page store unbounded.
    max_resident_bytes: Option<u64>,
    /// Auto-analyze policy applied to every database's background scheduler as its engine opens.
    autoanalyze: AutoAnalyzeConfig,
    /// Runtime checkpoint policy applied to every database as its engine opens; `None` leaves the
    /// log growing until an operator issues `CHECKPOINT` or the server restarts.
    checkpoint: Option<CheckpointConfig>,
    /// Root of the write-ahead-log archive (`--wal-archive-dir`); each database archives its
    /// checkpoints under `<root>/<database>/`. `None` keeps no archive.
    wal_archive: Option<PathBuf>,
    /// Set when this server is a standby following a primary's archive: every database is
    /// read-only, seeded from `<root>/<database>/` and kept up to date by its apply scheduler.
    standby: Option<StandbyConfig>,
    state: Mutex<ManagerState>,
}

impl std::fmt::Debug for DatabaseManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseManager")
            .field("root", &self.root)
            .field("default_name", &self.default_name)
            .finish_non_exhaustive()
    }
}

impl DatabaseManager {
    /// Open (or bootstrap) a cluster rooted at `data_dir`. On a fresh directory this creates the
    /// `global/` catalog and the default database `default_name`.
    pub(crate) fn open(
        data_dir: impl AsRef<Path>,
        default_name: impl Into<String>,
        max_txn_write_bytes: Option<u64>,
        max_resident_bytes: Option<u64>,
        autoanalyze: AutoAnalyzeConfig,
        durability: DurabilityOptions,
    ) -> io::Result<Self> {
        let DurabilityOptions {
            checkpoint,
            wal_archive,
            standby,
        } = durability;
        let root = data_dir.as_ref().to_path_buf();
        let default_name = default_name.into();
        // Detect the legacy single-database layout (a WAL at the root) before creating `base/`.
        let legacy_root = root.join("nusadb.wal").exists();
        std::fs::create_dir_all(root.join("global"))?;
        // Before anything under the data directory is read or written: no other server may be
        // running on it.
        let cluster_lock = lock_cluster(&root)?;
        std::fs::create_dir_all(root.join("base"))?;

        let mut databases = load_catalog(&root)?;
        if let Some(cfg) = &standby {
            if legacy_root {
                return Err(io::Error::other(
                    "a standby needs the per-database layout; this data directory is the legacy \
                     single-database one",
                ));
            }
            // Every database the primary archives exists here too, seeded from its newest
            // archived image when its directory is still empty.
            for name in archived_database_names(&cfg.root)? {
                let wal = base_dir(&root, &name).join("btree.wal");
                if !wal.exists() && !wal.with_extension("wal.ckpt").exists() {
                    std::fs::create_dir_all(base_dir(&root, &name))?;
                    let covered = nusadb_btree::seed_standby(&cfg.root.join(&name), &wal)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    tracing::info!(database = %name, position = covered, "standby seeded from the primary's archive");
                }
                databases.insert(name);
            }
            // The default database exists on every cluster, standby included, so a connection
            // that names no database has somewhere to go; it follows the primary's archive of
            // it once one appears.
            std::fs::create_dir_all(base_dir(&root, &default_name))?;
            databases.insert(default_name.clone());
            save_catalog(&root, &databases)?;
        }
        if databases.is_empty() {
            // Fresh (or legacy) cluster: register the default database so a first connection has
            // somewhere to go. A fresh cluster also creates its `base/<default>/` directory; a legacy
            // one keeps the existing root WAL in place.
            if !legacy_root {
                std::fs::create_dir_all(base_dir(&root, &default_name))?;
            }
            databases.insert(default_name.clone());
            save_catalog(&root, &databases)?;
        }

        Ok(Self {
            root,
            _cluster_lock: cluster_lock,
            default_name,
            legacy_root,
            max_txn_write_bytes,
            max_resident_bytes,
            autoanalyze,
            checkpoint,
            wal_archive,
            standby,
            state: Mutex::new(ManagerState {
                databases,
                engines: HashMap::new(),
            }),
        })
    }

    /// Whether this server is a standby (read-only, following a primary's archive).
    pub(crate) const fn is_standby(&self) -> bool {
        self.standby.is_some()
    }

    /// The WAL path of a registered database, for an offline restore into its directory; `None`
    /// for a name the cluster does not know.
    pub(crate) fn registered_wal_path(&self, name: &str) -> Option<PathBuf> {
        let state = self.state.lock().ok()?;
        state
            .databases
            .contains(name)
            .then(|| self.db_wal_path(name))
    }

    /// Move the archive directory of `name`, if any, to `<name>.dropped-<moment>` beside it.
    fn set_archive_aside(&self, name: &str) -> Result<(), ClusterError> {
        let Some(archive) = self.archive_dir(name) else {
            return Ok(());
        };
        if !archive.is_dir() {
            return Ok(());
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let aside = archive.with_file_name(format!("{name}.dropped-{stamp}"));
        std::fs::rename(&archive, &aside).map_err(|e| ClusterError::Io(e.to_string()))
    }

    /// The archive directory of database `name` under the configured archive root, if any.
    pub(crate) fn archive_dir(&self, name: &str) -> Option<PathBuf> {
        self.wal_archive.as_ref().map(|root| root.join(name))
    }

    /// The WAL path for database `name` (`btree.wal`): under `base/<name>/`, or at the root for
    /// the default database under the legacy single-database layout.
    fn db_wal_path(&self, name: &str) -> PathBuf {
        if self.legacy_root && name == self.default_name {
            self.root.join("btree.wal")
        } else {
            base_dir(&self.root, name).join("btree.wal")
        }
    }

    /// Lazily open `name`'s engine, caching it. The caller holds `state` locked.
    fn engine_for(
        &self,
        state: &mut ManagerState,
        name: &str,
    ) -> io::Result<Arc<dyn StorageEngine>> {
        if let Some(engine) = state.engines.get(name) {
            return Ok(Arc::clone(engine));
        }
        let wal = self.db_wal_path(name);
        if let Some(parent) = wal.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let dir = wal
            .parent()
            .map_or_else(|| self.root.clone(), Path::to_path_buf);
        check_engine_marker(&dir)?;
        // Vacuum's btree equivalent is purge, scheduled right here ('s contract: scheduling is
        // the composition root's job).
        let engine = Arc::new(
            // The archive goes in with the open: recovery may checkpoint before returning, and
            // that checkpoint must archive the segment it truncates like any other. A standby
            // opens as one from the first moment for the same reason.
            if self.is_standby() {
                BtreeEngine::open_standby(wal)
            } else {
                BtreeEngine::open_with_archive(wal, self.archive_dir(name))
            }
                // Apply the per-transaction and global resident write ceilings as the engine opens;
                // `None` leaves each unbounded (the pre-flag behavior). Both checks short-circuit
                // before any locking when unset, so an unconfigured server pays nothing. Applying the
                // ceilings AFTER open() means recovery (which does not go through `insert`) always
                // replays the full committed log even when the data exceeds the resident ceiling.
                .map(|e| {
                    e.with_max_txn_write_bytes(self.max_txn_write_bytes)
                        .with_max_total_resident_bytes(self.max_resident_bytes)
                })
                .map_err(|e| {
                    // Surface a refused open (e.g. a corrupt WAL mid-log hole) in the SERVER log,
                    // not only to the connecting client — the lazy per-database open means this is
                    // the first point an operator watching the logs learns the database is unhealthy.
                    tracing::error!(database = name, error = %e, "failed to open database engine");
                    io::Error::other(e.to_string())
                })?,
        );
        spawn_purge_scheduler(&engine, name);
        // Keep this database's planner statistics fresh in the background (no-op if disabled).
        // (A standby cannot write statistics; ANALYZE is a write, refused like any other.)
        if !self.is_standby() {
            spawn_analyze_scheduler(&engine, name, self.autoanalyze);
        }
        // Bound the log on a server that never restarts (no thread when disabled).
        if let Some(policy) = self.checkpoint {
            spawn_checkpoint_scheduler(&engine, name, policy);
        }
        // A standby applies what the primary archives and commits nothing of its own.
        if let Some(cfg) = &self.standby {
            spawn_standby_scheduler(&engine, name, cfg);
        }
        let engine: Arc<dyn StorageEngine> = engine;
        state.engines.insert(name.to_owned(), Arc::clone(&engine));
        Ok(engine)
    }
}

/// Base purge cadence: the wait after a pass that reclaimed only a modest backlog. Under sustained
/// churn a single pass reclaims far more, and [`next_purge_delay`] shortens the wait toward
/// [`PURGE_MIN_INTERVAL`] so the version store stays bounded between passes instead of growing.
const PURGE_INTERVAL: Duration = Duration::from_secs(10);

/// Shortest wait between purge passes, held while a pass keeps reclaiming a large backlog. It is a
/// floor, not a busy-loop: the batched, latch-dropping purge is cooperative with foreground work, and
/// the floor bounds purge's CPU/latch share.
const PURGE_MIN_INTERVAL: Duration = Duration::from_secs(2);

/// A pass reclaiming at least this many versions-plus-rows is read as "still under churn", so the
/// next pass follows promptly rather than after the full base interval. One purge row-batch, so the
/// fast cadence engages only when there is genuinely a batch-plus of work to keep up with.
const PURGE_BACKLOG_HIGH_WATER: usize = 4096;

/// Pick the wait before the next purge pass from how much the last pass reclaimed. A large reclaim
/// means the backlog is still building under churn, so check back soon; otherwise use the base
/// cadence. The result is always within `[PURGE_MIN_INTERVAL, PURGE_INTERVAL]` — the cadence can only
/// shorten the wait, never lengthen it past the base, so it never delays reclamation relative to the
/// old fixed schedule. A pure function of the reclaim count, kept separate for testing.
const fn next_purge_delay(reclaimed: usize) -> Duration {
    if reclaimed >= PURGE_BACKLOG_HIGH_WATER {
        PURGE_MIN_INTERVAL
    } else {
        PURGE_INTERVAL
    }
}

/// Run purge for one btree database on a cadence (scheduling wired at the composition
/// root): a detached thread holding only a weak reference, so it exits when the engine drops
/// and never keeps a dropped database alive.
fn spawn_purge_scheduler(engine: &Arc<BtreeEngine>, db: &str) {
    let weak = Arc::downgrade(engine);
    let db = db.to_owned();
    let thread_name = format!("purge-{db}");
    let spawned = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            while let Some(engine) = weak.upgrade() {
                let delay = match engine.purge() {
                    // Follow the backlog: a pass that reclaimed a batch-plus keeps the fast cadence
                    // so churn cannot outrun purge; a quiet pass falls back to the base.
                    Ok(stats) => {
                        let delay = next_purge_delay(stats.versions_reclaimed + stats.rows_removed);
                        tracing::trace!(?stats, ?delay, db = %db, "purge pass");
                        delay
                    },
                    // On a failed pass, wait the base interval before retrying rather than spinning.
                    Err(e) => {
                        tracing::warn!(error = %e, db = %db, "purge pass failed");
                        PURGE_INTERVAL
                    },
                };
                drop(engine);
                std::thread::sleep(delay);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "purge scheduler did not start; run purge manually");
    }
}

/// How a standby follows its primary: where the primary's archive root is, how often each
/// database looks for new segments, and how long new transactions may be held while a segment
/// waits for the running ones to end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StandbyConfig {
    /// The primary's `--wal-archive-dir` root, one subdirectory per database.
    pub(crate) root: PathBuf,
    /// How often each database polls the archive for new segments.
    pub(crate) poll: Duration,
    /// The longest a segment holds new transactions while the running ones end.
    pub(crate) max_pause: Duration,
}

impl StandbyConfig {
    /// The standby policy the `--standby-*` flags describe; `None` without a root.
    pub(crate) fn from_flags(
        root: Option<&str>,
        poll_secs: u64,
        max_pause_secs: u64,
    ) -> Option<Self> {
        Some(Self {
            root: PathBuf::from(root?),
            poll: Duration::from_secs(poll_secs.max(1)),
            max_pause: Duration::from_secs(max_pause_secs.min(MAX_PAUSE_SECS)),
        })
    }
}

/// The databases a primary has archived under `root`: one subdirectory each, leaving out the
/// `<name>.dropped-<moment>` directories of dropped databases and anything that is not a
/// valid database name.
fn archived_database_names(root: &Path) -> io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_valid_database_name(&name) && !name.contains(".dropped-") {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// What one standby poll of a database's archive did.
#[derive(Debug)]
enum StandbyTick {
    /// Nothing new past the applied position.
    Idle {
        /// The applied position.
        position: u64,
    },
    /// `segments` segments holding `records` records were applied; the position is now `position`.
    Applied {
        /// Segments applied this tick.
        segments: usize,
        /// Records applied this tick.
        records: usize,
        /// The applied position afterwards.
        position: u64,
    },
    /// A running transaction outlived the pause budget; the segment stays due for the next tick.
    Busy {
        /// Transactions still active at the deadline.
        active: usize,
        /// How long new transactions were held.
        waited: Duration,
    },
    /// The archive holds an image past the applied position but no segment chain reaching it:
    /// the primary's history was rebased (a restore) or segments were pruned. Applying stops
    /// until the standby is seeded again from the archive.
    Behind {
        /// The applied position.
        position: u64,
        /// The newest archived image.
        image: u64,
    },
    /// An apply failed (a corrupt segment, a gap, an I/O error); logged and retried.
    Failed {
        /// The failure.
        error: nusadb_core::Error,
    },
}

/// One poll: apply, in order, every segment archived past the standby's position.
fn standby_tick(engine: &BtreeEngine, archive: &Path, max_pause: Duration) -> StandbyTick {
    let position = match engine.wal_last_lsn() {
        Ok(Some(position)) => position,
        Ok(None) => {
            return StandbyTick::Failed {
                error: nusadb_core::Error::Io(io::Error::other(
                    "the in-memory engine cannot follow a primary",
                )),
            };
        },
        Err(error) => return StandbyTick::Failed { error },
    };
    // A database the primary has not checkpointed yet has no archive directory: nothing to
    // follow until one appears.
    if !archive.is_dir() {
        return StandbyTick::Idle { position };
    }
    let segments = match nusadb_btree::shipped_segments_after(archive, position) {
        Ok(segments) => segments,
        Err(error) => return StandbyTick::Failed { error },
    };
    if segments.is_empty() {
        return match nusadb_btree::newest_archived_image(archive) {
            Ok(Some(image)) if image > position => StandbyTick::Behind { position, image },
            Ok(_) => StandbyTick::Idle { position },
            Err(error) => StandbyTick::Failed { error },
        };
    }
    let mut applied_segments = 0;
    let mut applied_records = 0;
    let mut position = position;
    for (_, path) in segments {
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                return StandbyTick::Failed {
                    error: error.into(),
                };
            },
        };
        match engine.apply_shipped_segment(&bytes, max_pause) {
            Ok(nusadb_btree::ShipOutcome::Applied { records, last, .. }) => {
                applied_segments += 1;
                applied_records += records;
                position = last;
            },
            Ok(nusadb_btree::ShipOutcome::NothingNew) => {},
            Ok(nusadb_btree::ShipOutcome::StillBusy { active, waited }) => {
                return StandbyTick::Busy { active, waited };
            },
            Err(error) => return StandbyTick::Failed { error },
        }
    }
    if applied_segments == 0 {
        StandbyTick::Idle { position }
    } else {
        StandbyTick::Applied {
            segments: applied_segments,
            records: applied_records,
            position,
        }
    }
}

/// Follow the primary's archive for one database: a detached thread holding only a weak handle,
/// so it ends when the engine is dropped.
fn spawn_standby_scheduler(engine: &Arc<BtreeEngine>, db: &str, cfg: &StandbyConfig) {
    let weak = Arc::downgrade(engine);
    let db = db.to_owned();
    let archive = cfg.root.join(&db);
    let (poll, max_pause) = (cfg.poll, cfg.max_pause);
    let thread_name = format!("standby-{db}");
    let db_in_thread = db.clone();
    let spawned = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let db = db_in_thread;
            loop {
                std::thread::sleep(poll);
                let Some(engine) = weak.upgrade() else { break };
                match standby_tick(&engine, &archive, max_pause) {
                    StandbyTick::Idle { position } => {
                        tracing::trace!(db = %db, position, "standby tick: nothing new");
                    },
                    StandbyTick::Applied {
                        segments,
                        records,
                        position,
                    } => {
                        tracing::info!(db = %db, segments, records, position, "standby applied the primary's segments");
                    },
                    StandbyTick::Busy { active, waited } => {
                        tracing::warn!(
                            db = %db,
                            active_transactions = active,
                            paused_ms = waited.as_millis(),
                            "standby could not apply a segment: a transaction is being held open; \
                             retrying next tick"
                        );
                    },
                    StandbyTick::Behind { position, image } => {
                        tracing::error!(
                            db = %db,
                            position,
                            newest_image = image,
                            "the primary's archive has moved past this standby without a segment \
                             chain to follow (a restore on the primary, or pruned segments); seed \
                             the standby again from the archive"
                        );
                    },
                    StandbyTick::Failed { error } => {
                        tracing::warn!(db = %db, error = %error, "standby apply failed; retrying next tick");
                    },
                }
                drop(engine);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(db = %db, error = %e, "could not start the standby scheduler thread");
    }
}

/// The log-bounding, archiving and standby settings a cluster opens with.
#[derive(Debug, Clone, Default)]
pub(crate) struct DurabilityOptions {
    /// Runtime checkpoint policy; `None` never checkpoints in the background.
    pub(crate) checkpoint: Option<CheckpointConfig>,
    /// Root of the write-ahead-log archive; `None` keeps no archive.
    pub(crate) wal_archive: Option<PathBuf>,
    /// Follow a primary's archive as a read-only standby; `None` serves as a primary.
    pub(crate) standby: Option<StandbyConfig>,
}

/// Policy for the background runtime checkpoint: fold the log into a checkpoint image once it
/// has grown past `threshold_bytes`, checking every `interval`. It reuses the engine's own
/// stop-the-world `checkpoint()`, which refuses while any transaction is active, so a tick that
/// lands on a busy engine simply retries next time; the log is bounded whenever the workload
/// leaves quiesced instants (single-writer and bursty loads do; continuously overlapping
/// multi-connection saturation may not, and then only `CHECKPOINT` at a quiet moment helps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointConfig {
    /// Log length that arms a checkpoint attempt.
    pub(crate) threshold_bytes: u64,
    /// Wait between attempts.
    pub(crate) interval: Duration,
    /// After [`BUSY_TICKS_BEFORE_PAUSE`] consecutive busy refusals, hold new transactions for at
    /// most this long so the running ones drain and the checkpoint can run; `None` never pauses.
    pub(crate) max_pause: Option<Duration>,
}

impl CheckpointConfig {
    /// Build the policy from the server flags; `0` for the threshold or the interval disables it
    /// (no thread is spawned) and `0` for the pause keeps the worker opportunistic only, matching
    /// the other background workers' `0 = off` convention.
    pub(crate) const fn from_flags(
        threshold_bytes: u64,
        interval_secs: u64,
        max_pause_secs: u64,
    ) -> Option<Self> {
        match (threshold_bytes, interval_secs) {
            (0, _) | (_, 0) => None,
            (threshold_bytes, secs) => Some(Self {
                threshold_bytes,
                interval: Duration::from_secs(secs),
                // Capped: a pause is a stall every client feels, and one longer than a minute
                // would never be what an operator meant.
                max_pause: match max_pause_secs {
                    0 => None,
                    p => Some(Duration::from_secs(if p > MAX_PAUSE_SECS {
                        MAX_PAUSE_SECS
                    } else {
                        p
                    })),
                },
            }),
        }
    }
}

/// The longest admission pause the flag accepts, in seconds.
const MAX_PAUSE_SECS: u64 = 60;

/// Consecutive busy ticks after which the worker stops hoping for a quiet instant and pauses
/// admission for one. Three ticks at the default interval is fifteen seconds of a log past its
/// threshold on an engine that never goes quiet, which is the workload the pause exists for.
const BUSY_TICKS_BEFORE_PAUSE: u32 = 3;

/// A pause defeated by a transaction that outlives it doubles the busy ticks required before the
/// next one, up to this many doublings (192 ticks, sixteen minutes at the default interval), so a
/// client idle inside `BEGIN` costs the other clients a stall a few times an hour, not every tick.
const MAX_PAUSE_BACKOFF_DOUBLINGS: u32 = 6;

/// Whether the next attempt should pause admission: only once `consecutive_busy` refusals have
/// shown the engine will not go quiet on its own, backed off by the `defeated` pauses before it,
/// and only when a pause budget is configured.
const fn should_pause(
    consecutive_busy: u32,
    defeated: u32,
    max_pause: Option<Duration>,
) -> Option<Duration> {
    let doublings = if defeated > MAX_PAUSE_BACKOFF_DOUBLINGS {
        MAX_PAUSE_BACKOFF_DOUBLINGS
    } else {
        defeated
    };
    let required = BUSY_TICKS_BEFORE_PAUSE << doublings;
    match max_pause {
        Some(budget) if consecutive_busy >= required => Some(budget),
        _ => None,
    }
}

/// What one checkpoint tick observed and did. Separated from the thread loop so the policy is
/// testable synchronously against a real engine.
#[derive(Debug)]
enum CheckpointTick {
    /// The engine has no durable log (in-memory); nothing to bound, the scheduler can stop.
    NoLog,
    /// The log is still under the threshold; nothing done.
    BelowThreshold {
        /// Current log length.
        len: u64,
    },
    /// A checkpoint ran: the log went from `before` to `after` bytes, after holding new
    /// transactions for `paused` (zero when no pause was needed).
    Done {
        /// Log length before the checkpoint.
        before: u64,
        /// Log length after the truncation.
        after: u64,
        /// How long admission was held before the checkpoint began.
        paused: Duration,
    },
    /// The engine did not go quiet: either a plain refusal (transactions active, no pause
    /// attempted) or a pause whose budget ran out with transactions still running. Retried later.
    Busy {
        /// Log length at the refusal.
        len: u64,
        /// Transactions still active, when known.
        active: Option<usize>,
        /// The pause budget that was spent, when a pause was attempted.
        paused: Option<Duration>,
    },
    /// The checkpoint attempt failed for another reason (disk full, permissions); logged, retried.
    Failed {
        /// The failure.
        error: nusadb_core::Error,
    },
}

/// One policy evaluation: read the log length and checkpoint if it is past `threshold_bytes`.
/// It never invents a second durability path: the only write it can cause is the engine's own
/// gated `checkpoint()`, whose refusal is the safe outcome.
fn checkpoint_tick(
    engine: &BtreeEngine,
    threshold_bytes: u64,
    pause: Option<Duration>,
) -> CheckpointTick {
    let before = match engine.wal_len() {
        Ok(Some(len)) => len,
        Ok(None) => return CheckpointTick::NoLog,
        Err(error) => return CheckpointTick::Failed { error },
    };
    // Two reasons to checkpoint: the log has grown past the threshold, or the pages changed
    // since the last checkpoint fill half the page cache (they cannot leave it until an image
    // holds them, and writes stop when they fill it).
    if before < threshold_bytes && !engine.page_cache_needs_checkpoint() {
        return CheckpointTick::BelowThreshold { len: before };
    }
    pause.map_or_else(
        || checkpoint_plain(engine, before),
        |budget| checkpoint_paused(engine, before, budget),
    )
}

/// The log length after a checkpoint, for the log line; a failed read reports zero rather than
/// turning a completed checkpoint into an error.
fn log_len_after(engine: &BtreeEngine) -> u64 {
    engine.wal_len().ok().flatten().unwrap_or(0)
}

/// The opportunistic attempt: the engine's own checkpoint, refused while transactions are active.
fn checkpoint_plain(engine: &BtreeEngine, before: u64) -> CheckpointTick {
    match engine.checkpoint() {
        Ok(()) => CheckpointTick::Done {
            before,
            after: log_len_after(engine),
            paused: Duration::ZERO,
        },
        Err(error) if is_quiesce_refusal(&error) => CheckpointTick::Busy {
            len: before,
            active: None,
            paused: None,
        },
        Err(error) => CheckpointTick::Failed { error },
    }
}

/// The escalated attempt: hold new transactions for up to `budget` so the running ones drain.
fn checkpoint_paused(engine: &BtreeEngine, before: u64, budget: Duration) -> CheckpointTick {
    match engine.checkpoint_with_admission_pause(budget) {
        Ok(nusadb_btree::CheckpointOutcome::Done { waited }) => CheckpointTick::Done {
            before,
            after: log_len_after(engine),
            paused: waited,
        },
        Ok(nusadb_btree::CheckpointOutcome::StillBusy { active, waited }) => CheckpointTick::Busy {
            len: before,
            active: Some(active),
            paused: Some(waited),
        },
        Err(error) => CheckpointTick::Failed { error },
    }
}

/// The engine signals "not quiesced" as an I/O would-block error; every other error is a real
/// failure worth an operator's attention.
fn is_quiesce_refusal(error: &nusadb_core::Error) -> bool {
    matches!(error, nusadb_core::Error::Io(io) if io.kind() == io::ErrorKind::WouldBlock)
}

/// Run the runtime checkpoint policy for one btree database: a detached thread holding only a
/// weak reference (it exits when the engine drops), sleeping `interval` between ticks.
fn spawn_checkpoint_scheduler(engine: &Arc<BtreeEngine>, db: &str, policy: CheckpointConfig) {
    let weak = Arc::downgrade(engine);
    let db = db.to_owned();
    let thread_name = format!("checkpoint-{db}");
    let spawned = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let mut consecutive_busy: u32 = 0;
            let mut defeated_pauses: u32 = 0;
            loop {
                std::thread::sleep(policy.interval);
                let Some(engine) = weak.upgrade() else { break };
                let pause = should_pause(consecutive_busy, defeated_pauses, policy.max_pause);
                match checkpoint_tick(&engine, policy.threshold_bytes, pause) {
                    CheckpointTick::NoLog => break,
                    CheckpointTick::BelowThreshold { len } => {
                        consecutive_busy = 0;
                        tracing::trace!(db = %db, log_bytes = len, "checkpoint tick: below threshold");
                    },
                    CheckpointTick::Done {
                        before,
                        after,
                        paused,
                    } => {
                        consecutive_busy = 0;
                        defeated_pauses = 0;
                        tracing::info!(
                            db = %db,
                            before_bytes = before,
                            after_bytes = after,
                            paused_ms = paused.as_millis(),
                            "runtime checkpoint folded the log into a fresh image"
                        );
                    },
                    CheckpointTick::Busy {
                        len,
                        active,
                        paused: Some(waited),
                    } => {
                        // Start the count over and back off: the next pause waits for more busy
                        // ticks, so one long transaction does not stall every client every tick.
                        consecutive_busy = 0;
                        defeated_pauses = defeated_pauses.saturating_add(1);
                        tracing::warn!(
                            db = %db,
                            log_bytes = len,
                            active_transactions = active.unwrap_or(0),
                            paused_ms = waited.as_millis(),
                            "checkpoint could not drain the engine within its pause budget: a \
                             transaction is being held open (an idle client inside BEGIN?); the \
                             log keeps growing until it ends"
                        );
                    },
                    CheckpointTick::Busy { len, .. } => {
                        consecutive_busy = consecutive_busy.saturating_add(1);
                        tracing::debug!(
                            db = %db,
                            log_bytes = len,
                            consecutive_busy,
                            "checkpoint tick: engine busy, retrying next tick"
                        );
                    },
                    CheckpointTick::Failed { error } => {
                        tracing::warn!(db = %db, error = %error, "runtime checkpoint failed");
                    },
                }
                drop(engine);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "checkpoint scheduler did not start; issue CHECKPOINT manually");
    }
}

/// Configuration for the background auto-analyze scheduler (D-AUTO-ANALYZE): how often to sweep for
/// tables whose statistics have gone stale, and the scale-factor + threshold that decides staleness.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AutoAnalyzeConfig {
    /// Sweep cadence. `None` disables auto-analyze entirely (no thread is spawned).
    pub(crate) interval: Option<Duration>,
    /// Scale factor: the fraction of a table's rows of churn (on top of `base`) that marks its
    /// statistics stale.
    pub(crate) scale: f64,
    /// Threshold: the constant churn floor added to the scaled part.
    pub(crate) base: u64,
}

/// Keep one btree database's planner statistics fresh on a cadence: a detached thread that, holding
/// only a weak reference (so it exits when the engine drops), periodically runs
/// [`auto_analyze_stale_tables`](nusadb_sql::auto_analyze_stale_tables) — which re-`ANALYZE`s exactly
/// the tables whose churn has crossed the threshold, off any query's path. Does nothing (spawns no
/// thread) when auto-analyze is disabled.
fn spawn_analyze_scheduler(engine: &Arc<BtreeEngine>, db: &str, config: AutoAnalyzeConfig) {
    let Some(interval) = config.interval else {
        return; // auto-analyze disabled
    };
    let weak = Arc::downgrade(engine);
    let db = db.to_owned();
    let spawned = std::thread::Builder::new()
        .name(format!("analyze-{db}"))
        .spawn(move || {
            while let Some(engine) = weak.upgrade() {
                match nusadb_sql::auto_analyze_stale_tables(&*engine, config.scale, config.base) {
                    Ok(tables) if !tables.is_empty() => {
                        tracing::debug!(db = %db, ?tables, "auto-analyze refreshed statistics");
                    },
                    Ok(_) => {},
                    Err(e) => tracing::warn!(error = %e, db = %db, "auto-analyze pass failed"),
                }
                drop(engine);
                std::thread::sleep(interval);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "auto-analyze scheduler did not start");
    }
}

/// Enforce the per-directory engine marker: refuse to open a directory recorded (or inferred,
/// for pre-marker directories, from the presence of an `lsm`-era `nusadb.wal`) as any engine
/// other than `btree`, and stamp the marker on first open. An `lsm` directory is data written by
/// the removed engine — it must be migrated (dump from a pre-removal release, restore here), not
/// silently misread.
fn check_engine_marker(dir: &Path) -> io::Result<()> {
    let path = dir.join("engine");
    let recorded = match std::fs::read_to_string(&path) {
        Ok(text) => Some(text.trim().to_owned()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if dir.join("nusadb.wal").exists() {
                Some("lsm".to_owned())
            } else if dir.join("btree.wal").exists() {
                Some(EngineKind::Btree.as_str().to_owned())
            } else {
                None
            }
        },
        Err(e) => return Err(e),
    };
    if let Some(recorded) = &recorded
        && recorded != EngineKind::Btree.as_str()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "database directory {} was written by the {recorded} engine, which this release \
                 no longer ships; dump the data with a release that still reads it, then restore \
                 into a fresh data directory (a silent cross-engine read would corrupt it)",
                dir.display(),
            ),
        ));
    }
    if !path.exists() {
        std::fs::write(&path, EngineKind::Btree.as_str())?;
    }
    Ok(())
}

impl DatabaseCluster for DatabaseManager {
    fn open(&self, name: &str) -> Result<Option<Arc<dyn StorageEngine>>, ClusterError> {
        let mut state = self.state.lock().map_err(poisoned)?;
        if !state.databases.contains(name) {
            return Ok(None);
        }
        let engine = self
            .engine_for(&mut state, name)
            .map_err(|e| ClusterError::Io(e.to_string()))?;
        Ok(Some(engine))
    }

    fn create(&self, name: &str, if_not_exists: bool) -> Result<bool, ClusterError> {
        if !is_valid_database_name(name) {
            return Err(ClusterError::InvalidName(name.to_owned()));
        }
        if self.is_standby() {
            return Err(ClusterError::Protected(
                "this server is a standby; databases are created and dropped on the primary"
                    .to_owned(),
            ));
        }
        let mut state = self.state.lock().map_err(poisoned)?;
        if state.databases.contains(name) {
            return if if_not_exists {
                Ok(false)
            } else {
                Err(ClusterError::AlreadyExists(name.to_owned()))
            };
        }
        // Create the database's directory and register it; the engine is opened lazily on first use.
        // Clear any storage orphaned by a partial earlier drop (catalog removed, directory not yet)
        // so a fresh database always starts empty and never resurrects a dropped database's data.
        let dir = base_dir(&self.root, name);
        if dir.exists() {
            // Nobody may have the orphan open: another process holding it keeps it.
            let _lock = lock_database_dir(&dir, name)?;
            std::fs::remove_dir_all(&dir).map_err(|e| ClusterError::Io(e.to_string()))?;
        }
        // Likewise an archive a partial drop left under this name: it is another history's,
        // and the new engine would refuse to open against it.
        self.set_archive_aside(name)?;
        std::fs::create_dir_all(&dir).map_err(|e| ClusterError::Io(e.to_string()))?;
        state.databases.insert(name.to_owned());
        save_catalog(&self.root, &state.databases).map_err(|e| ClusterError::Io(e.to_string()))?;
        Ok(true)
    }

    fn drop_database(
        &self,
        name: &str,
        if_exists: bool,
        connected: &str,
    ) -> Result<bool, ClusterError> {
        if name == connected {
            return Err(ClusterError::InUse(name.to_owned()));
        }
        if self.is_standby() {
            return Err(ClusterError::Protected(
                "this server is a standby; databases are created and dropped on the primary"
                    .to_owned(),
            ));
        }
        if name == self.default_name {
            return Err(ClusterError::Protected(format!(
                "cannot drop the default database \"{name}\""
            )));
        }
        let mut state = self.state.lock().map_err(poisoned)?;
        if !state.databases.contains(name) {
            return if if_exists {
                Ok(false)
            } else {
                Err(ClusterError::NotFound(name.to_owned()))
            };
        }
        // Refuse if another connection still holds this database's engine (its directory is about
        // to be removed). The background workers (purge, analyze, checkpoint) briefly upgrade their
        // weak handle during a pass, so holders are counted through a weak handle after dropping
        // the cache's reference: a transient worker hold drains within the grace window, while a
        // connection's hold persists, and only the latter is `InUse`. A checkpoint of a large
        // database can outlast the window; the drop is then refused as `InUse` with the cache entry
        // restored and nothing deleted, and a retry after the checkpoint succeeds. Waiting until
        // the count reaches zero also guarantees the engine (and its open WAL handle) is fully
        // dropped before the directory is removed.
        if let Some(engine) = state.engines.get(name) {
            let weak = Arc::downgrade(engine);
            state.engines.remove(name);
            let mut spins = 0;
            while weak.strong_count() > 0 {
                spins += 1;
                if spins > 50 {
                    // A connection still holds it: restore the cache entry and refuse.
                    if let Some(engine) = weak.upgrade() {
                        state.engines.insert(name.to_owned(), engine);
                    }
                    return Err(ClusterError::InUse(name.to_owned()));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        // The database's lock before anything changes: a process outside this server that has it
        // open keeps it, and the drop is refused with the catalog and the directory untouched.
        let dir = base_dir(&self.root, name);
        let lock = if dir.exists() {
            Some(lock_database_dir(&dir, name)?)
        } else {
            None
        };
        state.databases.remove(name);
        save_catalog(&self.root, &state.databases).map_err(|e| ClusterError::Io(e.to_string()))?;
        // Remove the database's storage last: the catalog no longer lists it, so a crash here leaves
        // an orphan directory (harmless — re-`CREATE` reuses it) rather than a dangling catalog entry.
        if lock.is_some() {
            std::fs::remove_dir_all(&dir).map_err(|e| ClusterError::Io(e.to_string()))?;
        }
        drop(lock);
        // The archive is the dropped database's history, not the name's: a database created
        // again under this name starts its own, and an engine refuses to open against an
        // archive of another line. Keep the old one aside for restores. The database is gone
        // either way; a failure here is reported, and `CREATE` moves the leftover aside itself.
        if let Err(e) = self.set_archive_aside(name) {
            tracing::warn!(database = name, error = %e, "the dropped database's archive could not be moved aside");
        }
        Ok(true)
    }

    fn list(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|s| s.databases.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn default_database(&self) -> String {
        self.default_name.clone()
    }
}

/// A poisoned-lock failure as a cluster error (a prior panic while the lock was held).
fn poisoned<T>(_: std::sync::PoisonError<T>) -> ClusterError {
    ClusterError::Io("database manager lock poisoned".to_owned())
}

/// The on-disk directory for database `name`: `<root>/base/<name>`.
fn base_dir(root: &Path, name: &str) -> PathBuf {
    root.join("base").join(name)
}

/// Take the lock of the database stored in `dir` (named `name`), refusing it as in use when
/// another process has that database open.
fn lock_database_dir(dir: &Path, name: &str) -> Result<std::fs::File, ClusterError> {
    nusadb_btree::lock_database(&dir.join("btree.wal")).map_err(|e| match e {
        nusadb_core::Error::Io(io) if io.kind() == io::ErrorKind::WouldBlock => {
            ClusterError::InUse(name.to_owned())
        },
        other => ClusterError::Io(other.to_string()),
    })
}

/// Take the exclusive lock on `<root>/global/cluster.lock`, creating the file if needed.
/// Refused when another server holds it: two servers on one data directory would rewrite each
/// other's catalog and write the same databases.
fn lock_cluster(root: &Path) -> io::Result<std::fs::File> {
    let path = root.join("global").join("cluster.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            format!(
                "the data directory {} is already in use by another nusadb-server (it holds {}); \
                 stop that server before starting one here",
                root.display(),
                path.display()
            ),
        )),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// The cluster catalog file: `<root>/global/databases`.
fn catalog_path(root: &Path) -> PathBuf {
    root.join("global").join("databases")
}

/// Read the registered database names, or an empty set if the catalog does not exist yet. Entries
/// that are not valid database names (a hand-edited or corrupt catalog — e.g. a path-traversal
/// string) are dropped defensively, so a catalog name can never build a path outside `base/`.
fn load_catalog(root: &Path) -> io::Result<BTreeSet<String>> {
    match std::fs::read_to_string(catalog_path(root)) {
        Ok(text) => Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| is_valid_database_name(l))
            .map(ToOwned::to_owned)
            .collect()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(BTreeSet::new()),
        Err(e) => Err(e),
    }
}

/// Persist the database names, one per line. Writes a temp file and renames it over the catalog so a
/// crash mid-write cannot truncate the list.
fn save_catalog(root: &Path, databases: &BTreeSet<String>) -> io::Result<()> {
    let path = catalog_path(root);
    let tmp = path.with_extension("tmp");
    let mut body = String::new();
    for db in databases {
        body.push_str(db);
        body.push('\n');
    }
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Auto-analyze disabled — most tests do not want a background sweeper thread.
    const NO_AUTOANALYZE: AutoAnalyzeConfig = AutoAnalyzeConfig {
        interval: None,
        scale: 0.1,
        base: 50,
    };

    fn manager(dir: &Path) -> DatabaseManager {
        DatabaseManager::open(
            dir,
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .expect("open cluster")
    }

    /// A second server on a data directory in use is refused at startup, before it reads or
    /// writes anything there, and starts once the first one is gone.
    #[test]
    fn a_second_server_on_one_data_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let first = manager(dir.path());
        let err = DatabaseManager::open(
            dir.path(),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        assert!(
            err.to_string()
                .contains("already in use by another nusadb-server"),
            "{err}"
        );
        drop(first);
        let again = manager(dir.path());
        assert!(again.open("nusadb").unwrap().is_some());
    }

    /// `DROP DATABASE` leaves a database another process has open alone: refused as in use, its
    /// directory untouched, and dropped once that process lets it go.
    #[test]
    fn dropping_a_database_another_process_has_open_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        assert!(m.create("shop", false).unwrap());
        let wal = base_dir(dir.path(), "shop").join("btree.wal");
        let outside = nusadb_btree::BtreeEngine::open(&wal).unwrap();
        let err = m.drop_database("shop", false, "nusadb").unwrap_err();
        assert!(
            matches!(err, ClusterError::InUse(ref n) if n == "shop"),
            "{err:?}"
        );
        assert!(base_dir(dir.path(), "shop").exists());
        assert!(
            m.list().contains(&"shop".to_owned()),
            "a refused drop changes nothing"
        );
        drop(outside);
        assert!(m.drop_database("shop", false, "nusadb").unwrap());
        assert!(!base_dir(dir.path(), "shop").exists());
    }

    #[test]
    fn purge_cadence_speeds_up_under_backlog_but_never_slower_than_the_base() {
        // A pass that reclaimed a batch-plus of work keeps the fast cadence so churn cannot outrun
        // purge.
        assert_eq!(
            next_purge_delay(PURGE_BACKLOG_HIGH_WATER),
            PURGE_MIN_INTERVAL
        );
        assert_eq!(
            next_purge_delay(PURGE_BACKLOG_HIGH_WATER + 1_000_000),
            PURGE_MIN_INTERVAL
        );
        // A quiet or modest pass falls back to the base cadence.
        assert_eq!(next_purge_delay(0), PURGE_INTERVAL);
        assert_eq!(
            next_purge_delay(PURGE_BACKLOG_HIGH_WATER - 1),
            PURGE_INTERVAL
        );
        // Key safety property: the adaptive delay only ever shortens the wait — it stays within
        // [MIN, base], so it can never delay reclamation relative to the old fixed schedule.
        for reclaimed in [0usize, 1, 100, 4095, 4096, 4097, 1_000_000] {
            let delay = next_purge_delay(reclaimed);
            assert!(delay >= PURGE_MIN_INTERVAL && delay <= PURGE_INTERVAL);
        }
    }

    #[test]
    fn open_threads_the_txn_write_ceiling_to_each_database_engine() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType};

        // The --max-txn-write-bytes wiring: a manager opened with a ceiling must hand every
        // database's engine that same per-transaction write cap, rather than the limit staying
        // dormant. Under a tight ceiling the engine rejects an oversized transaction with a loud
        // OutOfMemory; a manager with no ceiling (the default) leaves the engine unbounded. (The
        // exact per-row charge — logical bytes plus the footprint overhead — is pinned in
        // nusadb-btree; here we only assert the ceiling reached the engine at all.)
        let def = TableDef {
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };

        // Bounded: the ceiling reaches the engine and rejects the oversized transaction.
        let tmp = tempfile::tempdir().unwrap();
        let bounded = DatabaseManager::open(
            tmp.path(),
            "nusadb",
            Some(40),
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .expect("open cluster");
        let engine = bounded.open("nusadb").unwrap().expect("default engine");
        let setup = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let table = engine.create_table(setup, &def).unwrap();
        engine.commit(setup).unwrap();
        let t = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let mut inserted = 0u64;
        let mut rejected = false;
        for i in 0..1000i64 {
            match engine.insert(t, table, &i.to_le_bytes()) {
                Ok(_) => inserted += 1,
                Err(nusadb_core::Error::OutOfMemory(_)) => {
                    rejected = true;
                    break;
                },
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        engine.rollback(t).unwrap();
        assert!(
            rejected,
            "the wired ceiling must reject the oversized transaction"
        );
        assert!(
            inserted < 6,
            "the ceiling bounds the transaction well short of the unbounded engine's six rows \
             (got {inserted})"
        );

        // Unbounded (default): the same sixth row inserts fine, confirming the flag is what bounds.
        let tmp2 = tempfile::tempdir().unwrap();
        let free = DatabaseManager::open(
            tmp2.path(),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .expect("open cluster");
        let e2 = free.open("nusadb").unwrap().expect("default engine");
        let s2 = e2.begin(IsolationLevel::ReadCommitted).unwrap();
        let tbl2 = e2.create_table(s2, &def).unwrap();
        e2.commit(s2).unwrap();
        let t2 = e2.begin(IsolationLevel::ReadCommitted).unwrap();
        for i in 0..6i64 {
            e2.insert(t2, tbl2, &i.to_le_bytes())
                .expect("an unbounded engine accepts every row");
        }
        e2.rollback(t2).unwrap();
    }

    #[test]
    fn open_threads_the_resident_ceiling_to_each_database_engine() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType};

        // The --max-resident-bytes wiring: a manager opened with a global resident ceiling must hand
        // every database's engine that same cap. With a ceiling below the store's post-create-table
        // footprint, the first insert is rejected with a loud OutOfMemory — proving the ceiling
        // reached the engine (the exact resident accounting is pinned in nusadb-btree). A manager
        // with no ceiling leaves the engine unbounded.
        let def = TableDef {
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };

        let tmp = tempfile::tempdir().unwrap();
        let bounded = DatabaseManager::open(
            tmp.path(),
            "nusadb",
            None,
            Some(1),
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .expect("open cluster");
        let engine = bounded.open("nusadb").unwrap().expect("default engine");
        let setup = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let table = engine.create_table(setup, &def).unwrap();
        engine.commit(setup).unwrap();
        let t = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        assert!(
            matches!(
                engine.insert(t, table, &1i64.to_le_bytes()),
                Err(nusadb_core::Error::OutOfMemory(_))
            ),
            "the wired resident ceiling must reject a write once the store is over it"
        );
        engine.rollback(t).unwrap();

        // Unbounded (default): the same insert succeeds, confirming the flag is what bounds.
        let tmp2 = tempfile::tempdir().unwrap();
        let free = DatabaseManager::open(
            tmp2.path(),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
        .expect("open cluster");
        let e2 = free.open("nusadb").unwrap().expect("default engine");
        let s2 = e2.begin(IsolationLevel::ReadCommitted).unwrap();
        let tbl2 = e2.create_table(s2, &def).unwrap();
        e2.commit(s2).unwrap();
        let t2 = e2.begin(IsolationLevel::ReadCommitted).unwrap();
        e2.insert(t2, tbl2, &1i64.to_le_bytes())
            .expect("an unbounded engine accepts the row");
        e2.rollback(t2).unwrap();
    }

    #[test]
    fn fresh_cluster_bootstraps_the_default_database() {
        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        assert_eq!(m.list(), vec!["nusadb".to_owned()]);
        assert!(base_dir(tmp.path(), "nusadb").is_dir());
        assert_eq!(m.default_database(), "nusadb");
        // The default resolves to an engine; an unknown name does not.
        assert!(m.open("nusadb").unwrap().is_some());
        assert!(m.open("ghost").unwrap().is_none());
    }

    #[test]
    fn create_registers_and_persists_a_database() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let m = manager(tmp.path());
            assert!(
                m.create("shop", false).unwrap(),
                "a new database is created"
            );
            assert_eq!(m.list(), vec!["nusadb".to_owned(), "shop".to_owned()]);
            assert!(base_dir(tmp.path(), "shop").is_dir());
            // Duplicate without IF NOT EXISTS errors; with it, it is a no-op.
            assert_eq!(
                m.create("shop", false),
                Err(ClusterError::AlreadyExists("shop".to_owned()))
            );
            assert!(!m.create("shop", true).unwrap());
            // Invalid names are rejected.
            assert_eq!(
                m.create("../escape", false),
                Err(ClusterError::InvalidName("../escape".to_owned()))
            );
        }
        // Persistence: a fresh manager over the same directory still lists `shop`.
        let m2 = manager(tmp.path());
        assert_eq!(m2.list(), vec!["nusadb".to_owned(), "shop".to_owned()]);
    }

    #[test]
    fn drop_removes_a_database_and_guards_current_and_default() {
        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        m.create("shop", false).unwrap();

        // Cannot drop the database the connection is in, nor the default.
        assert_eq!(
            m.drop_database("shop", false, "shop"),
            Err(ClusterError::InUse("shop".to_owned()))
        );
        assert!(matches!(
            m.drop_database("nusadb", false, "shop"),
            Err(ClusterError::Protected(_))
        ));

        // Dropping a non-current database removes its catalog entry and storage.
        assert!(m.drop_database("shop", false, "nusadb").unwrap());
        assert_eq!(m.list(), vec!["nusadb".to_owned()]);
        assert!(!base_dir(tmp.path(), "shop").exists());

        // Dropping again errors unless IF EXISTS.
        assert_eq!(
            m.drop_database("shop", false, "nusadb"),
            Err(ClusterError::NotFound("shop".to_owned()))
        );
        assert!(!m.drop_database("shop", true, "nusadb").unwrap());
    }

    #[test]
    fn a_standby_cluster_follows_the_primary_archive_and_refuses_ddl() {
        use nusadb_core::engine::{ColumnDef, TableDef};
        use nusadb_core::{ColumnType, IsolationLevel};
        let primary_dir = tempfile::tempdir().unwrap();
        let archive_root = primary_dir.path().join("archive");
        let primary = DatabaseManager::open(
            primary_dir.path().join("data"),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions {
                checkpoint: None,
                wal_archive: Some(archive_root.clone()),
                standby: None,
            },
        )
        .expect("open primary");
        primary.create("shop", false).unwrap();
        let engine = primary.open("shop").unwrap().expect("shop engine");
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let table = engine
            .create_table(
                txn,
                &TableDef {
                    schema: "public".to_owned(),
                    name: "t".to_owned(),
                    columns: vec![ColumnDef {
                        name: "v".to_owned(),
                        ty: ColumnType::Bytes,
                        nullable: false,
                    }],
                },
            )
            .unwrap();
        engine.insert(txn, table, b"one").unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();

        // The standby registers and seeds `shop` from the archive, opens it read-only.
        let standby_dir = tempfile::tempdir().unwrap();
        let cfg = StandbyConfig::from_flags(archive_root.to_str(), 3600, 2).unwrap();
        let standby = DatabaseManager::open(
            standby_dir.path(),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions {
                checkpoint: None,
                wal_archive: None,
                standby: Some(cfg.clone()),
            },
        )
        .expect("open standby");
        assert!(standby.list().contains(&"shop".to_owned()));
        assert!(matches!(
            standby.create("other", false),
            Err(ClusterError::Protected(_))
        ));
        assert!(matches!(
            standby.drop_database("shop", false, "nusadb"),
            Err(ClusterError::Protected(_))
        ));
        let rows = |e: &dyn StorageEngine| {
            let txn = e.begin(IsolationLevel::ReadCommitted).unwrap();
            let mut scan = e.scan(txn, table).unwrap();
            let mut out = Vec::new();
            while let Some((_, tuple)) = scan.try_next().unwrap() {
                out.push(tuple.to_vec());
            }
            e.commit(txn).unwrap();
            out.sort();
            out
        };
        // The seeded directory, driven tick by tick (the scheduler thread runs the same tick).
        let wal = base_dir(standby_dir.path(), "shop").join("btree.wal");
        let btree = BtreeEngine::open(&wal).unwrap();
        btree.set_standby(true);
        assert_eq!(rows(&btree), vec![b"one".to_vec()]);
        assert!(matches!(
            standby_tick(&btree, &archive_root.join("shop"), cfg.max_pause),
            StandbyTick::Idle { .. }
        ));
        // The primary writes on and checkpoints; one tick brings the standby level.
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        engine.insert(txn, table, b"two").unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
        assert!(matches!(
            standby_tick(&btree, &archive_root.join("shop"), cfg.max_pause),
            StandbyTick::Applied { segments: 1, .. }
        ));
        assert_eq!(rows(&btree), vec![b"one".to_vec(), b"two".to_vec()]);
        drop(btree);
        // Opened through the manager, the database is a standby: its writes are refused.
        let follower = standby.open("shop").unwrap().expect("standby shop engine");
        assert_eq!(rows(&*follower), vec![b"one".to_vec(), b"two".to_vec()]);
        let txn = follower.begin(IsolationLevel::ReadCommitted).unwrap();
        assert!(matches!(
            follower.insert(txn, table, b"local"),
            Err(nusadb_core::Error::ReadOnly(_))
        ));
        follower.rollback(txn).unwrap();
    }

    #[test]
    fn drop_moves_the_archive_aside_so_a_recreated_database_starts_its_own() {
        let tmp = tempfile::tempdir().unwrap();
        let archive_root = tmp.path().join("archive");
        let m = DatabaseManager::open(
            tmp.path().join("data"),
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions {
                checkpoint: None,
                wal_archive: Some(archive_root.clone()),
                standby: None,
            },
        )
        .expect("open cluster");
        m.create("shop", false).unwrap();
        let engine = m.open("shop").unwrap().expect("shop engine");
        engine.checkpoint().unwrap();
        drop(engine);
        assert!(archive_root.join("shop").is_dir());

        assert!(m.drop_database("shop", false, "nusadb").unwrap());
        assert!(!archive_root.join("shop").exists());
        let aside = std::fs::read_dir(&archive_root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .find(|n| n.starts_with("shop.dropped-"))
            .expect("the old archive was moved aside");
        assert!(
            std::fs::read_dir(archive_root.join(aside))
                .unwrap()
                .any(|e| e.unwrap().path().extension().is_some_and(|x| x == "ckpt")),
            "the old history's images travel with it"
        );

        // The name is free again and the new database archives into a fresh directory.
        m.create("shop", false).unwrap();
        let engine = m.open("shop").unwrap().expect("new shop engine");
        engine.checkpoint().unwrap();
        assert!(archive_root.join("shop").is_dir());
        drop(engine);

        // A drop that died after removing the storage but before moving the archive aside
        // leaves the archive under the name; `CREATE` moves it aside itself.
        assert!(m.drop_database("shop", false, "nusadb").unwrap());
        let aside: Vec<String> = std::fs::read_dir(&archive_root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("shop.dropped-"))
            .collect();
        std::fs::rename(
            archive_root.join(aside.iter().max().unwrap()),
            archive_root.join("shop"),
        )
        .unwrap();
        m.create("shop", false).unwrap();
        let moved: Vec<String> = std::fs::read_dir(&archive_root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("shop.dropped-"))
            .collect();
        assert_eq!(moved.len(), 2, "{moved:?}");
        let engine = m.open("shop").unwrap().expect("third shop engine");
        engine.checkpoint().unwrap();
    }

    /// A legacy single-database data directory written by the removed `lsm` engine (its
    /// `nusadb.wal` at the root) is refused loudly at open with a migration hint — never
    /// silently shadowed by a fresh btree database beside the old data.
    #[test]
    fn legacy_lsm_root_is_refused_with_a_migration_hint() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("nusadb.wal"), b"legacy lsm data").unwrap();

        let m = manager(tmp.path());
        assert_eq!(m.list(), vec!["nusadb".to_owned()]);
        assert!(
            !base_dir(tmp.path(), "nusadb").exists(),
            "the legacy root is not shadowed by a fresh base/ database"
        );
        let Err(err) = m.open("nusadb") else {
            panic!("opening an lsm-era data directory must fail");
        };
        let message = err.to_string();
        assert!(message.contains("lsm engine"), "{message}");
        assert!(message.contains("no longer ships"), "{message}");
    }

    #[test]
    fn recreating_a_dropped_database_starts_empty() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType};

        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        let def = TableDef {
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };
        m.create("shop", false).unwrap();
        {
            let engine = m.open("shop").unwrap().unwrap();
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            let id = engine.create_table(txn, &def).unwrap();
            engine.insert(txn, id, &[1]).unwrap();
            engine.commit(txn).unwrap();
        }
        m.drop_database("shop", false, "nusadb").unwrap();
        // Recreating the same name yields an empty database — no resurrected table/rows.
        m.create("shop", false).unwrap();
        let engine = m.open("shop").unwrap().unwrap();
        assert!(
            engine.lookup_table("t").unwrap().is_none(),
            "a recreated database does not inherit the dropped one's data"
        );
    }

    #[test]
    fn databases_are_physically_isolated() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType};

        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        m.create("shop", false).unwrap();

        let def = || TableDef {
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };
        // A same-named table in each database holds independent rows.
        for (db, val) in [("nusadb", 1_u8), ("shop", 2)] {
            let engine = m.open(db).unwrap().unwrap();
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            let id = engine.create_table(txn, &def()).unwrap();
            engine.insert(txn, id, &[val]).unwrap();
            engine.commit(txn).unwrap();
        }
        // Neither database sees the other's row: each `t` has exactly one, its own.
        for (db, val) in [("nusadb", 1_u8), ("shop", 2)] {
            let engine = m.open(db).unwrap().unwrap();
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            let id = engine.lookup_table("t").unwrap().unwrap().id;
            let mut scan = engine.scan(txn, id).unwrap();
            let mut rows = Vec::new();
            while let Some(item) = scan.try_next().unwrap() {
                rows.push(item.1.to_vec());
            }
            assert_eq!(rows, vec![vec![val]], "{db} sees only its own row");
            engine.commit(txn).unwrap();
        }
    }
    /// A btree-engine cluster round-trips through the same `DatabaseCluster` surface — the
    /// seam is engine-agnostic — and stamps the `engine` marker in the database directory.
    #[test]
    fn btree_cluster_round_trips_and_stamps_the_marker() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType};

        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        let engine = m.open("nusadb").unwrap().unwrap();
        let def = TableDef {
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let id = engine.create_table(txn, &def).unwrap();
        engine.insert(txn, id, &[7]).unwrap();
        engine.commit(txn).unwrap();

        let marker = base_dir(tmp.path(), "nusadb").join("engine");
        assert_eq!(std::fs::read_to_string(marker).unwrap().trim(), "btree");
        assert!(base_dir(tmp.path(), "nusadb").join("btree.wal").exists());
    }

    /// The marker refuses an `lsm`-engine directory — one with an explicit `lsm` marker and
    /// one inferred pre-marker from its `nusadb.wal` — because that engine no longer ships.
    #[test]
    fn lsm_marked_database_open_is_refused() {
        // Explicitly-marked lsm database directory: refused.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base_dir(tmp.path(), "nusadb")).unwrap();
        std::fs::write(base_dir(tmp.path(), "nusadb").join("engine"), "lsm").unwrap();
        let m = manager(tmp.path());
        let Err(err) = m.open("nusadb") else {
            panic!("an lsm-marked directory must fail to open");
        };
        assert!(err.to_string().contains("lsm engine"), "{err}");

        // Pre-marker legacy lsm directory (WAL present, no marker): inferred and refused.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(base_dir(tmp.path(), "nusadb")).unwrap();
        std::fs::write(base_dir(tmp.path(), "nusadb").join("nusadb.wal"), b"legacy").unwrap();
        let m = manager(tmp.path());
        let Err(err) = m.open("nusadb") else {
            panic!("legacy lsm dir must fail to open");
        };
        assert!(err.to_string().contains("lsm engine"), "{err}");
    }

    #[test]
    fn analyze_scheduler_refreshes_stale_statistics_in_the_background() {
        use std::fmt::Write as _;
        use std::time::Instant;

        use nusadb_core::TableSchema;
        use nusadb_sql::{Catalog, Error, IndexInfo, Session, analyze, parse, plan};

        // A minimal catalog — planning `ANALYZE`/`INSERT` only needs to resolve the table.
        struct Cat<'a>(&'a dyn StorageEngine);
        impl Catalog for Cat<'_> {
            fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
                self.0.lookup_table(name).map_err(Into::into)
            }
            fn list_indexes(&self, _: &str) -> Result<Vec<IndexInfo>, Error> {
                Ok(Vec::new())
            }
        }
        let run = |engine: &dyn StorageEngine, session: &mut Session, sql: &str| {
            let logical = analyze(parse(sql).unwrap(), &Cat(engine)).unwrap();
            session.execute(plan(logical)).unwrap();
        };

        // A manager with a tight auto-analyze cadence (25 ms) enabled.
        let tmp = tempfile::tempdir().unwrap();
        let config = AutoAnalyzeConfig {
            interval: Some(Duration::from_millis(25)),
            scale: 0.1,
            base: 50,
        };
        let m = DatabaseManager::open(
            tmp.path(),
            "nusadb",
            None,
            None,
            config,
            DurabilityOptions::default(),
        )
        .expect("open cluster");
        let engine = m.open("nusadb").unwrap().expect("default engine");

        // Load a table past the threshold (100 rows > 50 + 0.1*100 = 60) WITHOUT a manual ANALYZE.
        let mut session = Session::new(&*engine);
        run(&*engine, &mut session, "CREATE TABLE t (id INT, v INT)");
        let mut insert = String::from("INSERT INTO t VALUES ");
        for i in 0..100 {
            if i > 0 {
                insert.push(',');
            }
            write!(insert, "({i},{})", i * 2).unwrap();
        }
        run(&*engine, &mut session, &insert);

        let table = engine.lookup_table("t").unwrap().unwrap();
        assert!(
            engine.table_stats(table.id).unwrap().is_none(),
            "no statistics exist before the background sweep"
        );

        // The background scheduler must analyse the churned table on its own within a few sweeps.
        let deadline = Instant::now() + Duration::from_secs(5);
        while engine.table_stats(table.id).unwrap().is_none() {
            assert!(
                Instant::now() < deadline,
                "the auto-analyze scheduler did not refresh the statistics in time"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use nusadb_core::engine::{ColumnDef, TableDef};
    use nusadb_core::{ColumnType, IsolationLevel};

    use super::*;

    const RC: IsolationLevel = IsolationLevel::ReadCommitted;
    const THRESHOLD: u64 = 64 * 1024;

    fn open_engine(dir: &tempfile::TempDir) -> (BtreeEngine, nusadb_core::TableId) {
        let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine
            .create_table(
                txn,
                &TableDef {
                    schema: "public".to_owned(),
                    name: "t".to_owned(),
                    columns: vec![ColumnDef {
                        name: "v".to_owned(),
                        ty: ColumnType::Bytes,
                        nullable: false,
                    }],
                },
            )
            .unwrap();
        engine.commit(txn).unwrap();
        (engine, table)
    }

    /// Autocommit-style writes until the on-disk log is past `THRESHOLD`; returns the log length
    /// reached and the number of rows committed.
    fn write_past_threshold(engine: &BtreeEngine, table: nusadb_core::TableId) -> (u64, usize) {
        let payload = vec![0xAB_u8; 1024];
        let mut rows = 0;
        loop {
            let txn = engine.begin(RC).unwrap();
            engine.insert(txn, table, &payload).unwrap();
            engine.commit(txn).unwrap();
            rows += 1;
            let len = engine.wal_len().unwrap().unwrap();
            if len >= THRESHOLD {
                return (len, rows);
            }
        }
    }

    fn count_rows(engine: &BtreeEngine, table: nusadb_core::TableId) -> usize {
        let txn = engine.begin(RC).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut rows = 0;
        while scan.try_next().unwrap().is_some() {
            rows += 1;
        }
        engine.commit(txn).unwrap();
        rows
    }

    #[test]
    fn zero_on_either_flag_disables_the_policy() {
        assert_eq!(CheckpointConfig::from_flags(0, 5, 2), None);
        assert_eq!(CheckpointConfig::from_flags(1024, 0, 2), None);
        assert_eq!(
            CheckpointConfig::from_flags(1024, 5, 2),
            Some(CheckpointConfig {
                threshold_bytes: 1024,
                interval: Duration::from_secs(5),
                max_pause: Some(Duration::from_secs(2)),
            })
        );
        assert_eq!(
            CheckpointConfig::from_flags(1024, 5, 0).map(|c| c.max_pause),
            Some(None)
        );
    }

    #[test]
    fn tick_below_threshold_leaves_the_log_alone() {
        let dir = tempfile::tempdir().unwrap();
        let (engine, _) = open_engine(&dir);
        let before = engine.wal_len().unwrap().unwrap();
        let tick = checkpoint_tick(&engine, THRESHOLD, None);
        assert!(
            matches!(tick, CheckpointTick::BelowThreshold { len } if len == before),
            "{tick:?}"
        );
        assert_eq!(engine.wal_len().unwrap().unwrap(), before);
    }

    #[test]
    fn tick_past_threshold_shrinks_the_log_file_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let (engine, table) = open_engine(&dir);
        let (grown, rows_written) = write_past_threshold(&engine, table);
        let wal_path = dir.path().join("btree.wal");
        assert!(std::fs::metadata(&wal_path).unwrap().len() >= THRESHOLD);

        let tick = checkpoint_tick(&engine, THRESHOLD, None);
        let CheckpointTick::Done { before, after, .. } = tick else {
            panic!("expected a checkpoint, got {tick:?}");
        };
        assert_eq!(before, grown);
        assert!(after < before, "log did not shrink: {before} -> {after}");
        // The file itself, not a proxy: the truncated log is what a restart would replay.
        assert_eq!(std::fs::metadata(&wal_path).unwrap().len(), after);
        assert!(after < THRESHOLD);
        // The image now carries the rows: a reopen replays image + empty tail and sees every
        // committed write.
        drop(engine);
        let reopened = BtreeEngine::open(&wal_path).unwrap();
        assert_eq!(count_rows(&reopened, table), rows_written);
    }

    #[test]
    fn tick_on_a_busy_engine_is_refused_and_retries_after_commit() {
        let dir = tempfile::tempdir().unwrap();
        let (engine, table) = open_engine(&dir);
        let _ = write_past_threshold(&engine, table);
        let open_txn = engine.begin(RC).unwrap();
        let len_before = engine.wal_len().unwrap().unwrap();

        let tick = checkpoint_tick(&engine, THRESHOLD, None);
        assert!(
            matches!(tick, CheckpointTick::Busy { len, .. } if len == len_before),
            "{tick:?}"
        );
        assert_eq!(
            engine.wal_len().unwrap().unwrap(),
            len_before,
            "busy tick must not touch the log"
        );

        engine.commit(open_txn).unwrap();
        assert!(matches!(
            checkpoint_tick(&engine, THRESHOLD, None),
            CheckpointTick::Done { .. }
        ));
    }

    /// The documented restore: a checkpoint image copied into a registered database's directory
    /// opens as that point in time through the manager, so the server serves the restored data.
    #[test]
    fn a_checkpoint_image_copied_into_a_registered_database_restores_it() {
        let live = tempfile::tempdir().unwrap();
        let (engine, table) = open_engine(&live);
        let _ = write_past_threshold(&engine, table);
        engine.checkpoint().unwrap();
        let rows_at_backup = count_rows(&engine, table);
        // Writes after the backup point stay in the live database only.
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, b"after-backup").unwrap();
        engine.commit(txn).unwrap();

        let restored_root = tempfile::tempdir().unwrap();
        let manager = DatabaseManager::open(
            restored_root.path(),
            "nusadb",
            None,
            None,
            AutoAnalyzeConfig {
                interval: None,
                scale: 0.0,
                base: 0,
            },
            DurabilityOptions::default(),
        )
        .unwrap();
        assert!(manager.create("shop", false).unwrap());
        // A backup is the image plus the pages directory it reads from.
        let target = base_dir(restored_root.path(), "shop");
        std::fs::copy(
            live.path().join("btree.wal.ckpt"),
            target.join("btree.wal.ckpt"),
        )
        .unwrap();
        std::fs::create_dir_all(target.join("btree.wal.pages")).unwrap();
        for entry in std::fs::read_dir(live.path().join("btree.wal.pages")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(
                entry.path(),
                target.join("btree.wal.pages").join(entry.file_name()),
            )
            .unwrap();
        }
        let restored = manager.open("shop").unwrap().expect("registered database");
        let restored_table = restored.lookup_table("t").unwrap().unwrap();
        let txn = restored.begin(RC).unwrap();
        let mut scan = restored.scan(txn, restored_table.id).unwrap();
        let mut rows = 0;
        while scan.try_next().unwrap().is_some() {
            rows += 1;
        }
        drop(scan);
        restored.commit(txn).unwrap();
        assert_eq!(rows, rows_at_backup);
    }

    /// With a pause budget, a transaction that ends within it lets the checkpoint run; one that
    /// outlives it is reported as busy with the pause spent, and the log is untouched.
    #[test]
    fn a_paused_tick_drains_a_short_transaction_and_reports_a_long_one() {
        let dir = tempfile::tempdir().unwrap();
        let (engine, table) = open_engine(&dir);
        let _ = write_past_threshold(&engine, table);
        let engine = std::sync::Arc::new(engine);

        let open_txn = engine.begin(RC).unwrap();
        let ender = {
            let engine = std::sync::Arc::clone(&engine);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(150));
                engine.commit(open_txn).unwrap();
            })
        };
        let tick = checkpoint_tick(&engine, THRESHOLD, Some(Duration::from_secs(5)));
        assert!(
            matches!(tick, CheckpointTick::Done { paused, .. } if paused >= Duration::from_millis(100)),
            "{tick:?}"
        );
        ender.join().unwrap();

        let _ = write_past_threshold(&engine, table);
        let len_before = engine.wal_len().unwrap().unwrap();
        let held = engine.begin(RC).unwrap();
        let tick = checkpoint_tick(&engine, THRESHOLD, Some(Duration::from_millis(100)));
        assert!(
            matches!(
                tick,
                CheckpointTick::Busy {
                    active: Some(1),
                    paused: Some(_),
                    ..
                }
            ),
            "{tick:?}"
        );
        assert_eq!(engine.wal_len().unwrap().unwrap(), len_before);
        engine.commit(held).unwrap();
    }

    #[test]
    fn the_pause_is_used_only_after_repeated_busy_ticks_and_only_when_configured() {
        let budget = Some(Duration::from_secs(2));
        assert_eq!(should_pause(0, 0, budget), None);
        assert_eq!(should_pause(BUSY_TICKS_BEFORE_PAUSE - 1, 0, budget), None);
        assert_eq!(should_pause(BUSY_TICKS_BEFORE_PAUSE, 0, budget), budget);
        assert_eq!(should_pause(u32::MAX, 0, None), None);
    }

    #[test]
    fn a_defeated_pause_backs_off_exponentially_up_to_a_cap() {
        let budget = Some(Duration::from_secs(2));
        // One defeat doubles the busy ticks required; six defeats cap the doubling.
        assert_eq!(should_pause(BUSY_TICKS_BEFORE_PAUSE, 1, budget), None);
        assert_eq!(should_pause(BUSY_TICKS_BEFORE_PAUSE * 2, 1, budget), budget);
        assert_eq!(should_pause(BUSY_TICKS_BEFORE_PAUSE * 4, 2, budget), budget);
        let capped = BUSY_TICKS_BEFORE_PAUSE << MAX_PAUSE_BACKOFF_DOUBLINGS;
        assert_eq!(should_pause(capped - 1, 50, budget), None);
        assert_eq!(should_pause(capped, 50, budget), budget);
    }

    #[test]
    fn the_pause_flag_is_capped_at_a_minute() {
        assert_eq!(
            CheckpointConfig::from_flags(1024, 5, 3600).and_then(|c| c.max_pause),
            Some(Duration::from_mins(1))
        );
    }

    #[test]
    fn in_memory_engine_has_no_log_to_bound() {
        let engine = BtreeEngine::new();
        assert!(matches!(
            checkpoint_tick(&engine, THRESHOLD, None),
            CheckpointTick::NoLog
        ));
    }
}
