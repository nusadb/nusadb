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
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use nusadb_btree::crypt::MasterKey;
use nusadb_btree::{BtreeEngine, OpenOptions};
use nusadb_core::StorageEngine;
use nusadb_wire::{ClusterError, DatabaseCluster, is_valid_database_name};

/// Which storage engine the cluster runs. The clustered B-link/B+tree engine (ADR 008) is the
/// sole engine: the former `lsm` value is no longer accepted, so a
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
    /// The same engines, concretely typed and weakly held, so the metrics endpoint can ask each
    /// whether it stopped after a storage error without keeping a dropped database alive.
    health: HashMap<String, Weak<BtreeEngine>>,
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
    /// The prepared-transaction ceiling applied to every database's engine as it opens; `None`
    /// keeps the engine's default.
    max_prepared_transactions: Option<usize>,
    /// The master key every database is encrypted under (`--encryption-key-file`); `None` for
    /// an unencrypted cluster.
    master_key: Option<Arc<MasterKey>>,
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
    /// The refusal for dropping `name` while prepared transactions wait in it, from a private open
    /// of its log; `None` when there are none or the database cannot be opened here. The caller
    /// holds the cluster state with the database's engine out of the cache.
    fn prepared_in(&self, name: &str) -> Result<Option<ClusterError>, ClusterError> {
        let wal = database_wal(&self.root, self.legacy_root, &self.default_name, name);
        if !wal.exists() && !wal.with_extension("wal.ckpt").exists() {
            return Ok(None);
        }
        let Ok(engine) = BtreeEngine::open_with(&wal, self.open_options(name)) else {
            return Ok(None);
        };
        let waiting = engine
            .list_prepared()
            .map_err(|e| ClusterError::Io(e.to_string()))?;
        Ok(waiting.first().map(|first| {
            ClusterError::Protected(format!(
                "cannot drop database \"{name}\": {} prepared transaction(s) are waiting in it \
                 (\"{}\" first); finish them with COMMIT PREPARED or ROLLBACK PREPARED",
                waiting.len(),
                first.gid
            ))
        }))
    }

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
            max_prepared_transactions,
            master_key,
        } = durability;
        let root = data_dir.as_ref().to_path_buf();
        let default_name = default_name.into();
        // Detect the legacy single-database layout (a WAL at the root) before creating `base/`.
        let legacy_root = root.join("nusadb.wal").exists();
        std::fs::create_dir_all(root.join("global"))?;
        // Before anything under the data directory is read or written: no other server may be
        // running on it.
        let cluster_lock = lock_cluster(&root)?;
        check_cluster_format(&root)?;
        // A database an interrupted `--encrypt-from` left unfinished, registered or not, stops
        // the start (only the copy itself may go on, and it does not open the target this way):
        // otherwise a fresh catalog could register it as the default database and serve it empty.
        refuse_unfinished_copies(&root)?;
        let mut databases = load_catalog(&root)?;
        // Every database's data format is checked now, so a release older than the data is refused
        // at start-up rather than at the first connection to each database, and before this
        // release records itself anywhere.
        for name in &databases {
            check_database_format(&database_wal(&root, legacy_root, &default_name, name))?;
        }
        // Likewise every database's encryption, against the key given or its absence: a server
        // started with the wrong key, without the key of encrypted data, or with a key over
        // unencrypted data refuses to start, naming the database, with nothing changed.
        for name in &databases {
            let wal = database_wal(&root, legacy_root, &default_name, name);
            nusadb_btree::crypt::check_database(&wal, master_key.as_deref())
                .map_err(|e| io::Error::other(format!("database {name}: {e}")))?;
        }
        stamp_cluster_format(&root)?;
        std::fs::create_dir_all(root.join("base"))?;
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
                    let covered = nusadb_btree::seed_standby_with_key(
                        &cfg.root.join(&name),
                        &wal,
                        master_key.as_deref(),
                    )
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
            max_prepared_transactions,
            master_key,
            state: Mutex::new(ManagerState {
                databases,
                engines: HashMap::new(),
                health: HashMap::new(),
            }),
        })
    }

    /// How database `name`'s engine opens: its archive, standby mode and the master key.
    fn open_options(&self, name: &str) -> OpenOptions {
        OpenOptions {
            archive: if self.is_standby() {
                None
            } else {
                self.archive_dir(name)
            },
            standby: self.is_standby(),
            master_key: self.master_key.clone(),
        }
    }

    /// The master key the cluster is encrypted under, if it is.
    pub(crate) fn master_key(&self) -> Option<&MasterKey> {
        self.master_key.as_deref()
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

    /// The root of the checkpoint archive (`--wal-archive-dir`), one subdirectory per database.
    pub(crate) fn archive_root(&self) -> Option<&Path> {
        self.wal_archive.as_deref()
    }

    /// The data directory this cluster lives in.
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Every registered database name, sorted.
    pub(crate) fn database_names(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|s| s.databases.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// The engine of database `name`, opened if it is not yet, with its log's path relative to the
    /// data directory; `None` when no such database is registered.
    pub(crate) fn engine_with_path(
        &self,
        name: &str,
    ) -> io::Result<Option<(Arc<BtreeEngine>, PathBuf)>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("database manager state poisoned"))?;
        if !state.databases.contains(name) {
            return Ok(None);
        }
        self.engine_for(&mut state, name)?;
        let engine = state
            .health
            .get(name)
            .and_then(Weak::upgrade)
            .ok_or_else(|| io::Error::other(format!("database {name} closed while opening")))?;
        drop(state);
        let wal = self.db_wal_path(name);
        let relative = wal
            .strip_prefix(&self.root)
            .map_or_else(|_| wal.clone(), Path::to_path_buf);
        Ok(Some((engine, relative)))
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
        database_wal(&self.root, self.legacy_root, &self.default_name, name)
    }

    /// Every database opened since the server started, sorted by name, with whether it stopped
    /// after a storage error interrupted a change (it then refuses all work until a restart).
    pub(crate) fn database_health(&self) -> Vec<(String, bool)> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let mut health: Vec<(String, bool)> = state
            .health
            .iter()
            .filter_map(|(name, engine)| {
                engine
                    .upgrade()
                    .map(|engine| (name.clone(), engine.fault().is_some()))
            })
            .collect();
        health.sort_unstable();
        health
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
            BtreeEngine::open_with(wal, self.open_options(name))
                // Apply the per-transaction and global resident write ceilings as the engine opens;
                // `None` leaves each unbounded (the pre-flag behavior). Both checks short-circuit
                // before any locking when unset, so an unconfigured server pays nothing. Applying the
                // ceilings AFTER open() means recovery (which does not go through `insert`) always
                // replays the full committed log even when the data exceeds the resident ceiling.
                .map(|e| {
                    let e = e
                        .with_max_txn_write_bytes(self.max_txn_write_bytes)
                        .with_max_total_resident_bytes(self.max_resident_bytes);
                    if let Some(limit) = self.max_prepared_transactions {
                        e.set_max_prepared_transactions(limit);
                    }
                    e
                })
                .map_err(|e| {
                    // Surface a refused open (e.g. a corrupt WAL mid-log hole) in the SERVER log,
                    // not only to the connecting client — the lazy per-database open means this is
                    // the first point an operator watching the logs learns the database is unhealthy.
                    tracing::error!(database = name, error = %e, "failed to open database engine");
                    io::Error::other(e.to_string())
                })?,
        );
        // Before any session reaches it: file column defaults an older build stored under a bare
        // table name. A standby gets the primary's repair through the log.
        if !self.is_standby() {
            repair_column_defaults(&engine, name);
        }
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
            spawn_standby_scheduler(&engine, name, cfg, self.master_key.clone());
        }
        state
            .health
            .insert(name.to_owned(), Arc::downgrade(&engine));
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
fn standby_tick(
    engine: &BtreeEngine,
    archive: &Path,
    max_pause: Duration,
    master_key: Option<&MasterKey>,
) -> StandbyTick {
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
    // The segments are sealed under the primary's data key, named by the archive's envelope.
    let archive_key = match nusadb_btree::crypt::archive_key(archive, master_key) {
        Ok(key) => key,
        Err(error) => return StandbyTick::Failed { error },
    };
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
        match engine.apply_shipped_segment_with_key(&bytes, max_pause, archive_key.as_ref()) {
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
fn spawn_standby_scheduler(
    engine: &Arc<BtreeEngine>,
    db: &str,
    cfg: &StandbyConfig,
    master_key: Option<Arc<MasterKey>>,
) {
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
                match standby_tick(&engine, &archive, max_pause, master_key.as_deref()) {
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
    /// How many prepared transactions may wait at once in each database
    /// (`--max-prepared-transactions`; `0` refuses `PREPARE TRANSACTION`). `None` keeps the
    /// engine's default.
    pub(crate) max_prepared_transactions: Option<usize>,
    /// The master key every database is encrypted under; `None` for an unencrypted cluster.
    pub(crate) master_key: Option<Arc<MasterKey>>,
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
pub(crate) enum CheckpointTick {
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
pub(crate) fn checkpoint_tick(
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

/// Configuration for the background auto-analyze scheduler: how often to sweep for
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
    if std::fs::symlink_metadata(&path).is_err() {
        write_fresh(&path, EngineKind::Btree.as_str().as_bytes())?;
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
            let lock = lock_database_dir(&dir, name)?;
            remove_locked_database_dir(&dir, lock).map_err(|e| ClusterError::Io(e.to_string()))?;
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
        // A prepared transaction waits in the database's own log for a COMMIT PREPARED or
        // ROLLBACK PREPARED; dropping the database would silently decide it. Asked here, with the
        // cluster state held and no engine of it left in memory, so nothing can prepare between
        // the answer and the drop: its log is recovered by a private open and let go again. One
        // that cannot be opened (another process holds it, or it is damaged) is left to the lock
        // below, which refuses the first and lets the second be dropped.
        if let Some(first) = self.prepared_in(name)? {
            return Err(first);
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
        state.health.remove(name);
        save_catalog(&self.root, &state.databases).map_err(|e| ClusterError::Io(e.to_string()))?;
        // Remove the database's storage last: the catalog no longer lists it, so a crash here leaves
        // an orphan directory (harmless — re-`CREATE` reuses it) rather than a dangling catalog entry.
        if let Some(lock) = lock {
            remove_locked_database_dir(&dir, lock).map_err(|e| ClusterError::Io(e.to_string()))?;
        }
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

/// Remove the directory of a database whose lock `lock` is held. Everything but the lock file goes
/// first, with the lock still held so no other process can open the database meanwhile; then the
/// lock is released and the lock file and the directory go. Some systems refuse to delete a file
/// while a handle to it is open, so the lock file is removed only after its handle is closed.
fn remove_locked_database_dir(dir: &Path, lock: std::fs::File) -> io::Result<()> {
    let mut lock_file = dir.join("btree.wal").into_os_string();
    lock_file.push(".lock");
    let lock_file = PathBuf::from(lock_file);
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path == lock_file {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else if kind.is_symlink() {
            // A link is removed, never followed; a link to a directory on Windows needs
            // `remove_dir`.
            std::fs::remove_file(&path).or_else(|_| std::fs::remove_dir(&path))?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    drop(lock);
    std::fs::remove_dir_all(dir)
}

/// A point of the copy at which a test can stop it, as a crash would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyStep {
    /// The journal of a new copy is written; nothing else of the copy is.
    Journaled,
    /// The target of the named database is marked incomplete; nothing is registered yet.
    Marked,
    /// Every target is marked and the catalog lists exactly the source's databases.
    Registered,
    /// The named database left over from an earlier run (the source no longer has it) is
    /// removed from the target.
    Removed,
    /// The named database is copied and its marker removed.
    Copied,
}

/// The journal of an `--encrypt-from` copy in progress: `<target>/global/encrypt-copy`. Written
/// before anything else of the copy, removed only once every database is finished and checked.
fn copy_journal_path(target: &Path) -> PathBuf {
    target.join("global").join("encrypt-copy")
}

/// The first line of a copy journal.
const COPY_JOURNAL_HEADER: &str = "nusadb encrypted copy 3";

/// A copy journal: the source it copies, the check value of the master key it writes under
/// ([`MasterKey::copy_check`]), and every database the copy has created in the target. Only
/// those may be written again or removed by a later run.
#[derive(Debug, Default, PartialEq, Eq)]
struct CopyJournal {
    source: String,
    key: String,
    created: BTreeSet<String>,
}

impl CopyJournal {
    /// The journal in `target`, `None` when no copy is in progress there.
    fn read(target: &Path) -> io::Result<Option<Self>> {
        let path = copy_journal_path(target);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let damaged = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} is not a readable copy journal; remove the target data directory and \
                     copy again",
                    path.display()
                ),
            )
        };
        let mut lines = text.lines();
        if lines.next() != Some(COPY_JOURNAL_HEADER) {
            return Err(damaged());
        }
        let source = lines
            .next()
            .and_then(|l| l.strip_prefix("source "))
            .ok_or_else(damaged)?
            .to_owned();
        let key = lines
            .next()
            .and_then(|l| l.strip_prefix("key "))
            .ok_or_else(damaged)?
            .to_owned();
        let mut created = BTreeSet::new();
        for line in lines {
            let name = line.strip_prefix("created ").ok_or_else(damaged)?;
            if !is_valid_database_name(name) {
                return Err(damaged());
            }
            created.insert(name.to_owned());
        }
        Ok(Some(Self {
            source,
            key,
            created,
        }))
    }

    /// Write the journal to `target`, atomically and durably.
    fn write(&self, target: &Path) -> io::Result<()> {
        use std::fmt::Write as _;
        let mut text = format!(
            "{COPY_JOURNAL_HEADER}\nsource {}\nkey {}\n",
            self.source, self.key
        );
        for name in &self.created {
            let _ = writeln!(text, "created {name}");
        }
        let path = copy_journal_path(target);
        let scratch = path.with_extension("tmp");
        write_fresh(&scratch, text.as_bytes())?;
        std::fs::rename(&scratch, &path)?;
        sync_dir(&target.join("global"))
    }
}

/// Write `bytes` to `path` as a new file of its own, durably: a link there is replaced, never
/// written through, and a hard link keeps its contents ([`nusadb_btree::crypt::write_fresh`]).
fn write_fresh(path: &Path, bytes: &[u8]) -> io::Result<()> {
    nusadb_btree::crypt::write_fresh(path, bytes).map_err(|e| io::Error::other(e.to_string()))
}

/// Make a directory's entries durable, where the platform lets a directory be synced.
fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Copy every database of the data directory `source` (unencrypted, or encrypted under
/// `source_key`) into the data directory `target`, encrypted under `key`. Returns each database
/// with the log position its copy covers.
///
/// A new copy needs a fresh target: its catalog names no database the source does not have, and
/// each directory under `base/` is empty but for an empty lock file. The copy first writes its
/// journal ([`copy_journal_path`]) naming the source and the key, records in it every database
/// before creating it, marks every target database incomplete, and only then registers them, so
/// at no moment is a database registered that holds neither its mark nor a finished copy; a
/// server started on the target refuses while the journal or a mark is there.
///
/// A run after an interruption must name the same source and key, and copies every database
/// again from scratch: what an earlier run finished may no longer match the source, whose log
/// position says nothing of its contents (a database dropped and created again, or another data
/// directory put at the same path). A database the journal records as created by the copy that
/// the source no longer has is removed, provided its directory holds only database files;
/// anything else the target holds that the source does not have stops the run with nothing
/// removed. The journal goes only once the target lists exactly the source's databases, each a
/// finished copy under `key`, and no mark is left. A finished target (no journal) is refused.
///
/// # Errors
/// Refusals as above, the copy's errors, and I/O errors.
pub(crate) fn encrypt_cluster(
    target: &Path,
    source: &Path,
    key: &Arc<MasterKey>,
    source_key: Option<&Arc<MasterKey>>,
    autoanalyze: AutoAnalyzeConfig,
) -> io::Result<Vec<(String, u64)>> {
    encrypt_cluster_stepped(target, source, key, source_key, autoanalyze, &mut |_, _| {
        Ok(())
    })
}

/// Refuse a `--encrypt-from` source that is not a data directory, before anything is opened
/// or created there.
fn check_copy_source(source: &Path) -> io::Result<()> {
    if !source.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "the source data directory {} does not exist; nothing was changed",
                source.display()
            ),
        ));
    }
    let global = source.join("global");
    if !global.join("format").is_file() && !global.join("databases").is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} is not a nusadb data directory (it has no global/format); name the data \
                 directory of the cluster to copy. Nothing was changed",
                source.display()
            ),
        ));
    }
    Ok(())
}

/// [`encrypt_cluster`], telling `step` each point reached; an error from it stops the copy
/// there, as a crash would.
fn encrypt_cluster_stepped(
    target: &Path,
    source: &Path,
    key: &Arc<MasterKey>,
    source_key: Option<&Arc<MasterKey>>,
    autoanalyze: AutoAnalyzeConfig,
    step: &mut dyn FnMut(CopyStep, &str) -> io::Result<()>,
) -> io::Result<Vec<(String, u64)>> {
    check_copy_source(source)?;
    let source_id = std::fs::canonicalize(source)?
        .to_string_lossy()
        .into_owned();
    if source_id.contains(['\n', '\r']) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "the source path {source_id:?} contains a line break; move or link the source to \
                 a path without one. Nothing was changed"
            ),
        ));
    }
    let source_manager = DatabaseManager::open(
        source,
        nusadb_wire::cluster::DEFAULT_DATABASE,
        None,
        None,
        autoanalyze,
        DurabilityOptions {
            master_key: source_key.cloned(),
            ..DurabilityOptions::default()
        },
    )?;
    let names = source_manager.database_names();
    let mut sources = Vec::with_capacity(names.len());
    for name in &names {
        let from = source_manager
            .registered_wal_path(name)
            .ok_or_else(|| io::Error::other(format!("database {name} vanished from the source")))?;
        sources.push((name.clone(), from));
    }
    // A new copy into a target that is not fresh is refused before anything is created there;
    // the check is made again under the target's lock.
    if !copy_journal_path(target).exists() {
        let wanted: BTreeSet<String> = names.iter().cloned().collect();
        check_fresh_target(target, &wanted)?;
    }
    check_target_layout(target)?;
    std::fs::create_dir_all(target.join("global"))?;
    // Held for the whole copy: no server may start on the target meanwhile.
    let _lock = lock_cluster(target)?;
    prepare_encrypted_targets(target, &source_id, &key.copy_check(), &sources, step)?;
    let mut done = Vec::with_capacity(sources.len());
    for (name, from) in sources {
        let to = base_dir(target, &name).join("btree.wal");
        let covered = BtreeEngine::copy_encrypted(&from, source_key.map(AsRef::as_ref), &to, key)
            .map_err(|e| io::Error::other(format!("database {name}: {e}")))?;
        tracing::info!(database = %name, position = covered, "database copied, encrypted");
        step(CopyStep::Copied, &name)?;
        done.push((name, covered));
    }
    check_finished_copy(target, &names, key)?;
    std::fs::remove_file(copy_journal_path(target))?;
    sync_dir(&target.join("global"))?;
    Ok(done)
}

/// Refuse a data directory where an `--encrypt-from` copy has not finished: its journal, or a
/// database marked incomplete, is still there.
fn refuse_unfinished_copies(root: &Path) -> io::Result<()> {
    let mut found: Vec<String> = Vec::new();
    let journal = copy_journal_path(root);
    if journal.exists() {
        found.push(journal.display().to_string());
    }
    found.extend(
        incomplete_marks(root)
            .into_iter()
            .map(|m| m.display().to_string()),
    );
    if found.is_empty() {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "an encrypted copy (--encrypt-from) into {} has not finished: {} is still there. Run the \
         same --encrypt-from command again to finish it. If this directory was never the target \
         of --encrypt-from, the file is stray and can be removed. Nothing was changed",
        root.display(),
        found.join(", ")
    )))
}

/// Every `btree.wal.incomplete` marker under `<root>/base/`, sorted.
fn incomplete_marks(root: &Path) -> Vec<PathBuf> {
    let mut marks: Vec<PathBuf> = std::fs::read_dir(root.join("base"))
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| nusadb_btree::crypt::incomplete_path(&e.path().join("btree.wal")))
                .filter(|m| m.exists())
                .collect()
        })
        .unwrap_or_default();
    marks.sort();
    marks
}

/// The names of the entries under `<root>/base/` (directories, and links or files in their
/// place), sorted. An entry whose name is no valid database name is not a database's, and is
/// left out.
fn base_dir_names(root: &Path) -> io::Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let entries = match std::fs::read_dir(root.join("base")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_valid_database_name(&name) {
            out.insert(name);
        }
    }
    Ok(out)
}

/// Refuse a target a new copy may not write into: one whose catalog names a database the source
/// does not have, or whose `base/` holds anything but directories that are empty but for an
/// empty lock file (what a server started once on a new data directory leaves).
fn check_fresh_target(target: &Path, wanted: &BTreeSet<String>) -> io::Result<()> {
    let refuse = |what: String| {
        io::Error::other(format!(
            "{} already holds {what}; copy into an empty data directory, or remove this one \
             first. Nothing was changed",
            target.display()
        ))
    };
    let foreign: Vec<String> = load_catalog(target)?
        .into_iter()
        .filter(|name| !wanted.contains(name))
        .collect();
    if !foreign.is_empty() {
        return Err(refuse(format!(
            "databases the source does not have ({})",
            foreign.join(", ")
        )));
    }
    let entries = match std::fs::read_dir(target.join("base")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_dir() {
            return Err(refuse(format!("{} under base/", path.display())));
        }
        for inner in std::fs::read_dir(&path)? {
            let inner = inner?;
            let empty_lock = inner.file_name() == "btree.wal.lock"
                && inner.file_type()?.is_file()
                && inner.metadata()?.len() == 0;
            if !empty_lock {
                return Err(refuse(format!(
                    "databases or other files ({} among them)",
                    inner.path().display()
                )));
            }
        }
    }
    Ok(())
}

/// The files a database of this engine keeps in its directory beside the page segments:
/// its log, the files named after it, and the `engine` marker.
const DATABASE_FILES: [&str; 11] = [
    "engine",
    "btree.wal",
    "btree.wal.lock",
    "btree.wal.ckpt",
    "btree.wal.ckpt.tmp",
    "btree.wal.key",
    "btree.wal.key.tmp",
    "btree.wal.incomplete",
    "btree.wal.spill",
    "btree.wal.format",
    "btree.wal.format.tmp",
];

/// The directory of a database's page segments, inside its directory.
const DATABASE_PAGES_DIR: &str = "btree.wal.pages";

/// Whether `name` is the file name of a page segment: `<20 digits>-<hex stamp>[-full].seg`.
fn is_segment_file_name(name: &str) -> bool {
    let Some(base) = name.strip_suffix(".seg") else {
        return false;
    };
    let base = base.strip_suffix("-full").unwrap_or(base);
    let Some((position, stamp)) = base.split_once('-') else {
        return false;
    };
    position.len() == 20
        && position.bytes().all(|b| b.is_ascii_digit())
        && !stamp.is_empty()
        && stamp.len() <= 16
        && stamp.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether `<target>/base/<name>` exists, refusing it unless it is a real directory: a link is
/// never followed, so a copy can neither write nor remove anything outside the target.
fn is_real_database_dir(target: &Path, name: &str) -> io::Result<bool> {
    let dir = base_dir(target, name);
    match std::fs::symlink_metadata(&dir) {
        Ok(meta) if meta.file_type().is_dir() => Ok(true),
        Ok(_) => Err(io::Error::other(format!(
            "{} is a link or a file, not a database directory; nothing was changed. Move it out \
             of the data directory and run the same --encrypt-from command again",
            dir.display()
        ))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Refuse a target whose `global/` or `base/` is a link or a file: everything the copy writes
/// stays inside the target.
fn check_target_layout(target: &Path) -> io::Result<()> {
    for part in ["global", "base"] {
        let path = target.join(part);
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_dir() => {},
            Ok(_) => {
                return Err(io::Error::other(format!(
                    "{} is a link or a file, not a directory; copy into a data directory that \
                     holds its own. Nothing was changed",
                    path.display()
                )));
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => {},
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Refuse a database directory `dir` about to be written where a database file
/// ([`DATABASE_FILES`]) or the page segments directory is a link: the copy never writes through
/// one.
fn refuse_linked_database_files(dir: &Path) -> io::Result<()> {
    for name in DATABASE_FILES.iter().chain([&DATABASE_PAGES_DIR]) {
        let path = dir.join(name);
        if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(io::Error::other(format!(
                "{} is a link; the copy never writes through one. Nothing was changed. Remove \
                 it and run the same --encrypt-from command again",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Refuse to remove the database directory `dir` unless it holds only what a database of this
/// engine keeps there ([`DATABASE_FILES`], each a plain file, and a [`DATABASE_PAGES_DIR`] of
/// plain segment files). Nothing is followed through a link.
fn check_only_database_files(dir: &Path) -> io::Result<()> {
    let foreign = |path: &Path| {
        io::Error::other(format!(
            "{} is not a file of a database; {} was left in place. Move it out and run the same \
             --encrypt-from command again",
            path.display(),
            dir.display()
        ))
    };
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let kind = entry.file_type()?;
        if name == DATABASE_PAGES_DIR && kind.is_dir() {
            for segment in std::fs::read_dir(&path)? {
                let segment = segment?;
                let plain = segment.file_type()?.is_file();
                if !plain
                    || !segment
                        .file_name()
                        .to_str()
                        .is_some_and(is_segment_file_name)
                {
                    return Err(foreign(&segment.path()));
                }
            }
        } else if !(kind.is_file() && DATABASE_FILES.iter().any(|f| name == *f)) {
            return Err(foreign(&path));
        }
    }
    Ok(())
}

/// Remove the database directory `<target>/base/<name>`, which the copy created: its lock is
/// taken first, then its entries are checked ([`check_only_database_files`]), and only those
/// named there are removed, the directories last with `remove_dir`, so a file that appears
/// meanwhile makes the removal fail instead of being deleted.
fn remove_copied_database_dir(target: &Path, name: &str) -> io::Result<()> {
    if !is_real_database_dir(target, name)? {
        return Ok(());
    }
    let dir = base_dir(target, name);
    let lock = lock_database_dir(&dir, name)
        .map_err(|e| io::Error::other(format!("database {name}: {e}")))?;
    check_only_database_files(&dir)?;
    let pages = dir.join(DATABASE_PAGES_DIR);
    if std::fs::symlink_metadata(&pages).is_ok_and(|m| m.file_type().is_dir()) {
        for segment in std::fs::read_dir(&pages)? {
            let segment = segment?;
            if segment.file_type()?.is_file()
                && segment
                    .file_name()
                    .to_str()
                    .is_some_and(is_segment_file_name)
            {
                std::fs::remove_file(segment.path())?;
            }
        }
        std::fs::remove_dir(&pages)?;
    }
    for file in DATABASE_FILES {
        if file == "btree.wal.lock" {
            continue;
        }
        let path = dir.join(file);
        if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_file()) {
            std::fs::remove_file(&path)?;
        }
    }
    // The lock file goes once its handle is closed; some systems refuse to delete an open file.
    drop(lock);
    let _ = std::fs::remove_file(dir.join("btree.wal.lock"));
    std::fs::remove_dir(&dir)
}

/// Prepare `target` (its cluster lock held) for a copy of `sources` (each database and its log)
/// from the source `source_id` under the key whose [`MasterKey::copy_check`] is `key_check`:
/// check it may be written, write or check the journal, record and mark every database to copy,
/// register exactly them, and remove what an earlier run created of databases the source no
/// longer has. See [`encrypt_cluster`].
fn prepare_encrypted_targets(
    target: &Path,
    source_id: &str,
    key_check: &str,
    sources: &[(String, PathBuf)],
    step: &mut dyn FnMut(CopyStep, &str) -> io::Result<()>,
) -> io::Result<()> {
    check_cluster_format(target)?;
    if target.join("nusadb.wal").exists() {
        return Err(io::Error::other(
            "the target data directory has the legacy single-database layout; copy into a new \
             data directory",
        ));
    }
    check_target_layout(target)?;
    let wanted: BTreeSet<String> = sources.iter().map(|(name, _)| name.clone()).collect();
    let mut journal = match CopyJournal::read(target)? {
        Some(journal) if journal.source != source_id => {
            return Err(io::Error::other(format!(
                "{} holds an unfinished copy from {}, not from {source_id}; finish it with that \
                 source, or copy into an empty data directory. Nothing was changed",
                target.display(),
                journal.source
            )));
        },
        Some(journal) if journal.key != key_check => {
            return Err(io::Error::other(format!(
                "{} holds an unfinished copy under another encryption key than the one given; \
                 run the same command with the key it was started with, or copy into an empty \
                 data directory. Nothing was changed",
                target.display(),
            )));
        },
        Some(journal) => journal,
        None => {
            check_fresh_target(target, &wanted)?;
            let journal = CopyJournal {
                source: source_id.to_owned(),
                key: key_check.to_owned(),
                created: BTreeSet::new(),
            };
            journal.write(target)?;
            step(CopyStep::Journaled, "")?;
            journal
        },
    };
    // What the target holds that the source does not have. Only a database this copy created
    // may go, and only when its directory holds nothing but database files; anything else stops
    // the run before it changes anything.
    let leftovers: BTreeSet<String> = load_catalog(target)?
        .into_iter()
        .chain(base_dir_names(target)?)
        .filter(|name| !wanted.contains(name))
        .collect();
    let foreign: Vec<&str> = leftovers
        .iter()
        .filter(|name| !journal.created.contains(*name))
        .map(String::as_str)
        .collect();
    if !foreign.is_empty() {
        return Err(io::Error::other(format!(
            "{} holds databases this copy did not create and the source does not have ({}); \
             nothing was removed. Move them out of the data directory (and out of \
             global/databases), or remove the data directory and copy again",
            target.display(),
            foreign.join(", ")
        )));
    }
    // Before anything is marked or changed: every directory the copy will write or remove is a
    // real directory (never a link out of the target), and every one it will remove holds only
    // database files.
    for name in wanted.iter().chain(&leftovers) {
        if !is_real_database_dir(target, name)? {
            continue;
        }
        if leftovers.contains(name) {
            check_only_database_files(&base_dir(target, name))?;
        } else {
            refuse_linked_database_files(&base_dir(target, name))?;
        }
    }
    stamp_cluster_format(target)?;
    // Every database is recorded as the copy's own, durably, before anything of it is created.
    if !wanted.is_subset(&journal.created) {
        journal.created.extend(wanted.iter().cloned());
        journal.write(target)?;
    }
    // Every database is copied afresh, so each is marked before the catalog may name it.
    for (name, _) in sources {
        nusadb_btree::crypt::mark_incomplete(&base_dir(target, name).join("btree.wal"))
            .map_err(|e| io::Error::other(format!("database {name}: {e}")))?;
        step(CopyStep::Marked, name)?;
    }
    // Registered, and the leftovers unregistered, in one atomic catalog write.
    save_catalog(target, &wanted)?;
    step(CopyStep::Registered, "")?;
    // No longer registered: their storage goes. A crash part way leaves directories the catalog
    // does not name, under the journal, which the next run removes.
    for name in leftovers {
        remove_copied_database_dir(target, &name)?;
        tracing::info!(database = %name, "removed from the target: the source no longer has it");
        step(CopyStep::Removed, &name)?;
    }
    sync_dir(&target.join("base"))?;
    Ok(())
}

/// Check that the copy into `target` is complete before its journal goes: the catalog and the
/// directories under `base/` name exactly `names`, no database is marked incomplete, and each is
/// a finished copy under `key`.
fn check_finished_copy(target: &Path, names: &[String], key: &MasterKey) -> io::Result<()> {
    let wanted: BTreeSet<String> = names.iter().cloned().collect();
    let unfinished = |what: String| {
        io::Error::other(format!(
            "the encrypted copy into {} is not complete: {what}. Run the same --encrypt-from \
             command again",
            target.display()
        ))
    };
    let catalog = load_catalog(target)?;
    if catalog != wanted {
        return Err(unfinished(format!(
            "the catalog lists {catalog:?}, the source {wanted:?}"
        )));
    }
    let dirs = base_dir_names(target)?;
    if dirs != wanted {
        return Err(unfinished(format!(
            "the databases stored are {dirs:?}, the source's {wanted:?}"
        )));
    }
    let marks = incomplete_marks(target);
    if !marks.is_empty() {
        return Err(unfinished(format!("{marks:?} remain")));
    }
    for name in names {
        let wal = base_dir(target, name).join("btree.wal");
        if !BtreeEngine::is_finished_encrypted_copy(&wal, key)
            .map_err(|e| io::Error::other(format!("database {name}: {e}")))?
        {
            return Err(unfinished(format!(
                "database {name} is not a finished copy"
            )));
        }
    }
    Ok(())
}

/// What [`rotate_master_key`] did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rotation {
    /// Envelopes sealed again under the new key.
    pub(crate) rewrapped: usize,
    /// Envelopes already under the new key (a rotation run again after an interruption).
    pub(crate) already: usize,
}

/// Seal every data key under `new` instead of `old`: the envelope of each database under the
/// data directory `root`, and every envelope found under `extra` (the checkpoint archive root,
/// the backup directory). Holds the cluster lock, so no server may run on `root` meanwhile. Each
/// envelope is replaced atomically and the data is not rewritten; an interrupted rotation is
/// finished by running it again with the same keys.
///
/// # Errors
/// Another server holds the data directory, an envelope opens with neither key (named in the
/// error; the envelopes before it were rotated), or I/O errors.
pub(crate) fn rotate_master_key(
    root: &Path,
    extra: &[PathBuf],
    old: &MasterKey,
    new: &MasterKey,
) -> io::Result<Rotation> {
    std::fs::create_dir_all(root.join("global"))?;
    let _lock = lock_cluster(root)?;
    let mut rotation = Rotation::default();
    let mut envelopes = Vec::new();
    for dir in std::iter::once(root.to_path_buf()).chain(extra.iter().cloned()) {
        find_envelopes(&dir, &mut envelopes)?;
    }
    for path in envelopes {
        match nusadb_btree::crypt::rewrap(&path, old, new)
            .map_err(|e| io::Error::other(format!("{}: {e}", path.display())))?
        {
            nusadb_btree::crypt::Rewrapped::Done => rotation.rewrapped += 1,
            nusadb_btree::crypt::Rewrapped::Already => rotation.already += 1,
        }
        tracing::info!(envelope = %path.display(), "data key sealed under the new master key");
    }
    Ok(rotation)
}

/// Every key envelope under `dir`, recursively: a file named `key` or `*.key` that starts with
/// the envelope header. Directories that cannot be read are skipped. A link at an envelope's
/// name, or to a directory, is refused: the rotation would otherwise pass over what it hides.
fn find_envelopes(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(());
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let kind = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let envelope_name = name == "key" || name.ends_with(".key");
        if kind.is_symlink() && (envelope_name || path.is_dir()) {
            // Not followed, and not passed over either: an envelope behind it would stay sealed
            // under the old key, lost with it once the old key is destroyed.
            return Err(io::Error::other(format!(
                "{} is a link; a rotation does not follow links, and would leave any key \
                 envelope behind it under the old key. Replace it with the file or directory \
                 itself and rotate again. Nothing was changed",
                path.display()
            )));
        }
        if kind.is_dir() {
            find_envelopes(&path, out)?;
        } else if kind.is_file() && envelope_name && nusadb_btree::crypt::is_envelope(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// Take the exclusive lock on `<root>/global/cluster.lock`, creating the file if needed.
/// Refused when another server holds it: two servers on one data directory would rewrite each
/// other's catalog and write the same databases.
fn lock_cluster(root: &Path) -> io::Result<std::fs::File> {
    let path = root.join("global").join("cluster.lock");
    if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} is a link; a lock file is never followed. Remove it. Nothing was changed",
                path.display()
            ),
        ));
    }
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

/// The cluster layout format this release writes: the `global/` and `base/<db>/` layout and the
/// database catalog in `global/databases`.
pub(crate) const CLUSTER_FORMAT_VERSION: u32 = 1;

/// The text of `global/format` for this release.
fn cluster_format_text() -> String {
    format!(
        "nusadb cluster format {CLUSTER_FORMAT_VERSION}\nwritten by nusadb {}\n",
        env!("CARGO_PKG_VERSION")
    )
}

/// The log of database `name`: under `base/<name>/`, or at the root for the default database of a
/// legacy single-database layout.
fn database_wal(root: &Path, legacy_root: bool, default_name: &str, name: &str) -> PathBuf {
    if legacy_root && name == default_name {
        root.join("btree.wal")
    } else {
        base_dir(root, name).join("btree.wal")
    }
}

/// Refuse a cluster laid out in a newer format than this release writes. A cluster with no
/// `global/format` was laid out before the format was recorded, in format 1. Runs under the cluster
/// lock, before anything else is read.
fn check_cluster_format(root: &Path) -> io::Result<()> {
    let path = root.join("global").join("format");
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let mut lines = text.lines();
            let version = lines
                .next()
                .and_then(|line| line.strip_prefix("nusadb cluster format "))
                .and_then(|v| v.trim().parse::<u32>().ok())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{} is not a cluster format file", path.display()),
                    )
                })?;
            if version > CLUSTER_FORMAT_VERSION {
                let writer = lines
                    .next()
                    .and_then(|line| line.strip_prefix("written by "))
                    .unwrap_or("a newer release");
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "the data directory {} is in cluster format {version}, written by \
                         {writer}; this release (nusadb {}) reads cluster format \
                         {CLUSTER_FORMAT_VERSION} and older. Start it with the release that wrote \
                         it or a newer one; nothing was changed",
                        root.display(),
                        env!("CARGO_PKG_VERSION"),
                    ),
                ));
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => {},
        Err(e) => return Err(e),
    }
    Ok(())
}

/// Record this release's cluster format and version in `global/format`, once every check passed.
fn stamp_cluster_format(root: &Path) -> io::Result<()> {
    let path = root.join("global").join("format");
    let text = cluster_format_text();
    if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
        let scratch = path.with_extension("tmp");
        write_fresh(&scratch, text.as_bytes())?;
        std::fs::rename(&scratch, &path)?;
        #[cfg(unix)]
        std::fs::File::open(root.join("global"))?.sync_all()?;
    }
    Ok(())
}

/// Refuse at start-up a database whose data format is newer than this release reads; the engine
/// repeats the check (and upgrades an older format) when the database is opened.
fn check_database_format(wal: &Path) -> io::Result<()> {
    let found =
        nusadb_btree::format::read_format(wal).map_err(|e| io::Error::other(e.to_string()))?;
    if let Some(version) = found
        && version > nusadb_btree::format::FORMAT_VERSION
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "the database at {} is in data format {version}; this release (nusadb {}) reads \
                 data format {} and older. Start the server with the release that wrote it or a \
                 newer one; nothing was changed",
                wal.display(),
                env!("CARGO_PKG_VERSION"),
                nusadb_btree::format::FORMAT_VERSION,
            ),
        ));
    }
    Ok(())
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

/// Persist the database names, one per line. Writes a temp file, syncs it and renames it over the
/// catalog, then syncs the directory, so a crash cannot truncate the list or lose the change.
fn save_catalog(root: &Path, databases: &BTreeSet<String>) -> io::Result<()> {
    let path = catalog_path(root);
    let tmp = path.with_extension("tmp");
    let mut body = String::new();
    for db in databases {
        body.push_str(db);
        body.push('\n');
    }
    write_fresh(&tmp, body.as_bytes())?;
    std::fs::rename(&tmp, &path)?;
    sync_dir(&root.join("global"))
}

/// Run the one-time column-default repair on a database as it opens (see
/// [`repair_misfiled_column_defaults`](nusadb_sql::repair_misfiled_column_defaults)) and log what
/// it did. A default it cannot attribute is logged with the manual fix; a failure is logged and the
/// database still opens, and the repair runs again at the next open.
fn repair_column_defaults(engine: &BtreeEngine, db: &str) {
    let report = match nusadb_sql::repair_misfiled_column_defaults(engine) {
        Ok(report) => report,
        Err(e) => {
            tracing::warn!(db, error = %e, "column default repair failed; it runs again at the next open");
            return;
        },
    };
    // A move is logged as a warning: the entry names no schema, so one left behind when an
    // earlier {schema}.{table} was dropped would be attached to a table re-created under that name
    // just the same, and only the operator can tell.
    for (schema, table, column) in &report.moved {
        tracing::warn!(
            db,
            "column default of {schema}.{table}.{column} was stored under the bare name {table}; \
             moved to {schema}.{table}. If {schema}.{table} was dropped and re-created since that \
             default was set, check it with SELECT column_default FROM information_schema.columns \
             WHERE table_schema = '{schema}' AND table_name = '{table}' AND column_name = \
             '{column}' and remove it with ALTER TABLE {schema}.{table} ALTER COLUMN {column} DROP \
             DEFAULT if it does not belong there"
        );
    }
    for entry in &report.ambiguous {
        let (table, column) = (&entry.table, &entry.column);
        let (in_use, drop_hint) = if entry
            .schemas
            .iter()
            .any(|s| s == nusadb_core::PUBLIC_SCHEMA)
        {
            (
                format!(" (an insert into public.{table} uses it)"),
                format!(
                    "; if public.{table} should not have it, run ALTER TABLE public.{table} ALTER \
                     COLUMN {column} DROP DEFAULT"
                ),
            )
        } else {
            (String::new(), String::new())
        };
        tracing::warn!(
            db,
            "column default {sql:?} stored under the bare name {table} for column {column} could \
             belong to the table {table} in any of the schemas {schemas:?}, so it was left in \
             place{in_use}. On the table that owns it run ALTER TABLE <schema>.{table} ALTER \
             COLUMN {column} SET DEFAULT {sql}{drop_hint}",
            sql = entry.default_sql,
            schemas = entry.schemas,
        );
    }
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

    use nusadb_core::IsolationLevel;

    /// A source cluster of three databases, each with table `t` holding rows `0..rows`.
    fn three_databases(dir: &Path, rows: u32) {
        three_databases_from(dir, 0..rows);
    }

    /// A source cluster of three databases, each with table `t` holding rows `values`.
    fn three_databases_from(dir: &Path, values: std::ops::Range<u32>) {
        let m = manager(dir);
        for db in ["nusadb", "db2", "db3"] {
            if db != "nusadb" {
                m.create(db, false).unwrap();
            }
            fill(&m, db, values.clone());
        }
    }

    /// Create table `t` in database `db` and insert rows `values`, in one transaction.
    fn fill(m: &DatabaseManager, db: &str, values: std::ops::Range<u32>) {
        let engine = m.open(db).unwrap().unwrap();
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let table = engine
            .create_table(
                txn,
                &nusadb_core::engine::TableDef {
                    schema: "public".to_owned(),
                    name: "t".to_owned(),
                    columns: vec![nusadb_core::ColumnDef {
                        name: "v".to_owned(),
                        ty: nusadb_core::ColumnType::Bytes,
                        nullable: false,
                    }],
                },
            )
            .unwrap();
        for i in values {
            engine.insert(txn, table, &i.to_be_bytes()).unwrap();
        }
        engine.commit(txn).unwrap();
    }

    fn open_encrypted(dir: &Path, key: &Arc<MasterKey>) -> io::Result<DatabaseManager> {
        DatabaseManager::open(
            dir,
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions {
                master_key: Some(Arc::clone(key)),
                ..DurabilityOptions::default()
            },
        )
    }

    /// Every file under `dir` but the lock files, with its contents.
    fn files_of(dir: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
        let mut out = std::collections::BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_none_or(|x| x != "lock") {
                    let bytes = std::fs::read(&p).unwrap();
                    out.insert(p, bytes);
                }
            }
        }
        out
    }

    /// No database is registered in `dir` that has neither its marker nor a finished copy.
    fn every_registered_target_is_marked_or_finished(dir: &Path, key: &MasterKey) {
        for name in load_catalog(dir).unwrap() {
            let wal = base_dir(dir, &name).join("btree.wal");
            assert!(
                nusadb_btree::crypt::incomplete_path(&wal).exists()
                    || BtreeEngine::is_finished_encrypted_copy(&wal, key).unwrap(),
                "{name} is registered with neither a marker nor a finished copy"
            );
        }
    }

    /// The target at `dir` is a finished copy serving exactly `dbs`: no journal, no marker, a
    /// catalog of `dbs` and nothing else stored, and a normal start accepts it.
    fn assert_finished_target(dir: &Path, key: &Arc<MasterKey>, dbs: &[&str]) -> DatabaseManager {
        assert!(!copy_journal_path(dir).exists());
        assert_eq!(incomplete_marks(dir), Vec::<PathBuf>::new());
        let wanted: BTreeSet<String> = dbs.iter().map(|d| (*d).to_owned()).collect();
        assert_eq!(load_catalog(dir).unwrap(), wanted);
        assert_eq!(base_dir_names(dir).unwrap(), wanted);
        let m = open_encrypted(dir, key).unwrap();
        assert_eq!(
            m.list(),
            dbs.iter().map(|d| (*d).to_owned()).collect::<Vec<_>>()
        );
        m
    }

    #[test]
    fn an_interrupted_encrypted_copy_of_several_databases_is_refused_and_a_rerun_finishes_it() {
        let key = Arc::new(MasterKey::from_text(&"3c".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 150);
        // Where to stop, and an extra crash artifact to leave behind there.
        let points: [(CopyStep, &str, bool); 6] = [
            (CopyStep::Journaled, "", false),
            (CopyStep::Marked, "db2", false),
            (CopyStep::Marked, "db3", false),
            (CopyStep::Registered, "", false),
            // Mid-segment: db2's copy died after its data key and part of its segment.
            (CopyStep::Registered, "", true),
            // Databases are copied in name order: db2 is done, db3 and nusadb are not.
            (CopyStep::Copied, "db2", false),
        ];
        for (at, name, mid_segment) in points {
            let target = tempfile::tempdir().unwrap();
            let stopped = encrypt_cluster_stepped(
                target.path(),
                source.path(),
                &key,
                None,
                NO_AUTOANALYZE,
                &mut |step, db| {
                    if step == at && db == name {
                        Err(io::Error::other("stopped here"))
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(stopped.is_err(), "{at:?} {name}");
            if mid_segment {
                let dir = base_dir(target.path(), "db2");
                let other = tempfile::tempdir().unwrap();
                let envelope = other.path().join("k");
                nusadb_btree::crypt::DataKey::create(&envelope, &key).unwrap();
                std::fs::copy(&envelope, dir.join("btree.wal.key")).unwrap();
                std::fs::create_dir_all(dir.join("btree.wal.pages")).unwrap();
                std::fs::write(
                    dir.join("btree.wal.pages")
                        .join("00000000000000000001-1-full.seg"),
                    [7u8; 9000],
                )
                .unwrap();
            }
            every_registered_target_is_marked_or_finished(target.path(), &key);
            // A normal start refuses, and changes nothing.
            let before = files_of(target.path());
            let err = open_encrypted(target.path(), &key).unwrap_err().to_string();
            assert!(err.contains("has not finished"), "{at:?} {name}: {err}");
            assert_eq!(files_of(target.path()), before, "{at:?} {name}");
            // The run again copies every database afresh.
            let done = run_copy(target.path(), source.path(), &key).unwrap();
            assert_eq!(
                done.iter().map(|(db, _)| db.as_str()).collect::<Vec<_>>(),
                ["db2", "db3", "nusadb"]
            );
            let m = assert_finished_target(target.path(), &key, &["db2", "db3", "nusadb"]);
            for db in ["nusadb", "db2", "db3"] {
                assert_eq!(
                    rows_in(&m, db),
                    (0..150).collect::<Vec<_>>(),
                    "{at:?} {name}: {db}"
                );
            }
            drop(m);
            // The copy is complete: a further run is refused, whatever the source did since.
            let before = files_of(target.path());
            let err = run_copy(target.path(), source.path(), &key)
                .unwrap_err()
                .to_string();
            assert!(err.contains("already holds databases"), "{err}");
            assert_eq!(files_of(target.path()), before);
        }
    }

    fn run_copy(
        target: &Path,
        source: &Path,
        key: &Arc<MasterKey>,
    ) -> io::Result<Vec<(String, u64)>> {
        encrypt_cluster(target, source, key, None, NO_AUTOANALYZE)
    }

    /// Stop a copy at `at` for database `db`, as a crash there would.
    fn copy_until_step(target: &Path, source: &Path, key: &Arc<MasterKey>, at: CopyStep, db: &str) {
        let stopped =
            encrypt_cluster_stepped(target, source, key, None, NO_AUTOANALYZE, &mut |step, d| {
                if step == at && d == db {
                    Err(io::Error::other("stopped here"))
                } else {
                    Ok(())
                }
            });
        assert!(stopped.is_err());
    }

    /// Stop a copy just after database `db` is finished.
    fn copy_until(target: &Path, source: &Path, key: &Arc<MasterKey>, db: &str) {
        copy_until_step(target, source, key, CopyStep::Copied, db);
    }

    /// Rows in table `t` of database `db` of the cluster at `dir`.
    fn rows_in(m: &DatabaseManager, db: &str) -> Vec<u32> {
        let engine = m.open(db).unwrap().unwrap();
        let table = engine.lookup_table("t").unwrap().unwrap().id;
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut out = Vec::new();
        while let Some((_, tuple)) = scan.try_next().unwrap() {
            out.push(u32::from_be_bytes(tuple[..4].try_into().unwrap()));
        }
        drop(scan);
        engine.commit(txn).unwrap();
        out.sort_unstable();
        out
    }

    /// Add rows `from..to` to table `t` of `db` in the unencrypted cluster at `dir`, and delete
    /// row 0.
    fn change_source(dir: &Path, db: &str, from: u32, to: u32) {
        let m = manager(dir);
        let engine = m.open(db).unwrap().unwrap();
        let table = engine.lookup_table("t").unwrap().unwrap().id;
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        for i in from..to {
            engine.insert(txn, table, &i.to_be_bytes()).unwrap();
        }
        let mut scan = engine.scan(txn, table).unwrap();
        let mut zero = None;
        while let Some((tid, tuple)) = scan.try_next().unwrap() {
            if tuple[..4] == 0u32.to_be_bytes() {
                zero = Some(tid);
            }
        }
        drop(scan);
        engine.delete(txn, table, zero.unwrap()).unwrap();
        engine.commit(txn).unwrap();
    }

    /// The log position database `db` of the unencrypted cluster at `dir` stands at.
    fn position_of(dir: &Path, db: &str) -> u64 {
        let engine =
            BtreeEngine::open_with(base_dir(dir, db).join("btree.wal"), OpenOptions::default())
                .unwrap();
        engine.wal_last_lsn().unwrap().unwrap_or(0)
    }

    #[test]
    fn a_copy_resumes_only_from_the_same_source_and_copies_everything_again() {
        let key = Arc::new(MasterKey::from_text(&"4d".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 100);

        // Another source with the same database names, and the same key: an interrupted copy
        // from the first is not finished from the second, and nothing is changed.
        let other = tempfile::tempdir().unwrap();
        three_databases(other.path(), 7);
        let target = tempfile::tempdir().unwrap();
        copy_until(target.path(), source.path(), &key, "db2");
        let before = files_of(target.path());
        let err = run_copy(target.path(), other.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unfinished copy from"), "{err}");
        assert_eq!(files_of(target.path()), before);

        // The source moved on after db2 was copied: the rerun holds the source's latest rows.
        change_source(source.path(), "db2", 100, 130);
        run_copy(target.path(), source.path(), &key).unwrap();
        let m = assert_finished_target(target.path(), &key, &["db2", "db3", "nusadb"]);
        assert_eq!(rows_in(&m, "db2"), (1..130).collect::<Vec<_>>());
        assert_eq!(rows_in(&m, "db3"), (0..100).collect::<Vec<_>>());
        drop(m);

        // A finished copy of another source in the target: refused, never kept.
        let err = run_copy(target.path(), other.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("already holds databases"), "{err}");
    }

    #[test]
    fn a_rerun_copies_again_a_database_recreated_at_the_same_log_position() {
        let key = Arc::new(MasterKey::from_text(&"5e".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 100);
        let target = tempfile::tempdir().unwrap();
        copy_until(target.path(), source.path(), &key, "db2");
        // db2 dropped and created again with other rows, by the same statements: its log
        // stands where the finished copy's did, though its contents differ.
        let was = position_of(source.path(), "db2");
        let m = manager(source.path());
        m.drop_database("db2", false, "nusadb").unwrap();
        m.create("db2", false).unwrap();
        fill(&m, "db2", 500..600);
        drop(m);
        assert_eq!(position_of(source.path(), "db2"), was);

        run_copy(target.path(), source.path(), &key).unwrap();
        let m = assert_finished_target(target.path(), &key, &["db2", "db3", "nusadb"]);
        assert_eq!(rows_in(&m, "db2"), (500..600).collect::<Vec<_>>());
        assert_eq!(rows_in(&m, "db3"), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn a_rerun_from_another_cluster_put_at_the_source_path_copies_only_it() {
        let key = Arc::new(MasterKey::from_text(&"6f".repeat(32)).unwrap());
        let parent = tempfile::tempdir().unwrap();
        let source = parent.path().join("data");
        three_databases(&source, 100);
        let target = tempfile::tempdir().unwrap();
        copy_until(target.path(), &source, &key, "db2");
        // The directory is replaced by another cluster built the same way, with other rows.
        std::fs::rename(&source, parent.path().join("old")).unwrap();
        three_databases_from(&source, 1000..1100);

        run_copy(target.path(), &source, &key).unwrap();
        let m = assert_finished_target(target.path(), &key, &["db2", "db3", "nusadb"]);
        for db in ["nusadb", "db2", "db3"] {
            assert_eq!(rows_in(&m, db), (1000..1100).collect::<Vec<_>>(), "{db}");
        }
    }

    #[test]
    fn a_database_the_source_dropped_is_removed_from_the_target_on_the_rerun() {
        let key = Arc::new(MasterKey::from_text(&"7a".repeat(32)).unwrap());
        // db2 finished before the source dropped it; db3 was still marked when the source
        // dropped it. Either way the rerun leaves exactly the source's databases.
        for dropped in ["db2", "db3"] {
            let source = tempfile::tempdir().unwrap();
            three_databases(source.path(), 50);
            let target = tempfile::tempdir().unwrap();
            copy_until(target.path(), source.path(), &key, "db2");
            manager(source.path())
                .drop_database(dropped, false, "nusadb")
                .unwrap();
            let kept: Vec<&str> = ["db2", "db3", "nusadb"]
                .into_iter()
                .filter(|d| *d != dropped)
                .collect();

            // A rerun that stops after the catalog no longer names it, before its directory
            // goes: still refused by a normal start, and finished by the next run.
            copy_until_step(target.path(), source.path(), &key, CopyStep::Registered, "");
            assert!(base_dir(target.path(), dropped).exists());
            assert!(!load_catalog(target.path()).unwrap().contains(dropped));
            let err = open_encrypted(target.path(), &key).unwrap_err().to_string();
            assert!(err.contains("has not finished"), "{err}");

            let done = run_copy(target.path(), source.path(), &key).unwrap();
            assert_eq!(
                done.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>(),
                kept
            );
            let m = assert_finished_target(target.path(), &key, &kept);
            assert!(m.open(dropped).unwrap().is_none(), "{dropped} is served");
            assert!(!base_dir(target.path(), dropped).exists());
            for db in &kept {
                assert_eq!(rows_in(&m, db), (0..50).collect::<Vec<_>>(), "{db}");
            }
        }
    }

    #[test]
    fn a_copy_that_does_not_end_consistent_keeps_its_journal() {
        let key = Arc::new(MasterKey::from_text(&"8b".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);
        let target = tempfile::tempdir().unwrap();
        // A marker appears under the target while the copy runs: the run does not claim success,
        // the journal stays, and a normal start refuses.
        let stray = base_dir(target.path(), "zz").join("btree.wal");
        let err = encrypt_cluster_stepped(
            target.path(),
            source.path(),
            &key,
            None,
            NO_AUTOANALYZE,
            &mut |step, db| {
                if step == CopyStep::Copied && db == "nusadb" {
                    nusadb_btree::crypt::mark_incomplete(&stray)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                }
                Ok(())
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("not complete"), "{err}");
        assert!(copy_journal_path(target.path()).exists());
        assert!(open_encrypted(target.path(), &key).is_err());
        // The copy did not create it, so the rerun leaves it and stops; once it is moved out,
        // the rerun finishes.
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("did not create"), "{err}");
        assert!(nusadb_btree::crypt::incomplete_path(&stray).exists());
        std::fs::remove_dir_all(base_dir(target.path(), "zz")).unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db2", "db3", "nusadb"],
        ));
    }

    #[test]
    fn a_new_copy_refuses_a_target_holding_anything_but_empty_database_directories() {
        let key = Arc::new(MasterKey::from_text(&"91".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);

        // A file that is no database's, in a directory under base/: refused, kept.
        let target = tempfile::tempdir().unwrap();
        let photos = target.path().join("base").join("photos");
        std::fs::create_dir_all(&photos).unwrap();
        std::fs::write(photos.join("IMPORTANT.txt"), b"keep me").unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("already holds"), "{err}");
        assert_eq!(
            std::fs::read(photos.join("IMPORTANT.txt")).unwrap(),
            b"keep me"
        );
        assert!(!target.path().join("global").exists(), "nothing is created");

        // A plain file directly under base/: refused too.
        let target = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(target.path().join("base")).unwrap();
        std::fs::write(target.path().join("base").join("notes"), b"x").unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("already holds"), "{err}");
        assert!(target.path().join("base").join("notes").exists());

        // A stopped cluster with an empty database of its own: refused, the database kept.
        let target = tempfile::tempdir().unwrap();
        manager(target.path()).create("app", false).unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("the source does not have (app)"), "{err}");
        assert!(load_catalog(target.path()).unwrap().contains("app"));
        assert!(base_dir(target.path(), "app").is_dir());
        assert!(!copy_journal_path(target.path()).exists());

        // A data directory a server was started on once, with nothing written: a fresh target.
        let target = tempfile::tempdir().unwrap();
        drop(manager(target.path()));
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db2", "db3", "nusadb"],
        ));
    }

    #[test]
    fn a_rerun_removes_only_databases_the_copy_created_and_only_their_files() {
        let key = Arc::new(MasterKey::from_text(&"92".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);

        // Stopped right after the journal was first written; then a directory of files and an
        // empty database appear in the target. The rerun removes neither, and keeps its journal.
        let target = tempfile::tempdir().unwrap();
        copy_until_step(target.path(), source.path(), &key, CopyStep::Journaled, "");
        let photos = target.path().join("base").join("photos");
        std::fs::create_dir_all(&photos).unwrap();
        std::fs::write(photos.join("IMPORTANT.txt"), b"keep me").unwrap();
        let app = base_dir(target.path(), "app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("btree.wal.lock"), b"").unwrap();
        save_catalog(target.path(), &BTreeSet::from(["app".to_owned()])).unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("did not create"), "{err}");
        assert!(err.contains("app") && err.contains("photos"), "{err}");
        assert_eq!(
            std::fs::read(photos.join("IMPORTANT.txt")).unwrap(),
            b"keep me"
        );
        assert!(app.is_dir());
        assert!(load_catalog(target.path()).unwrap().contains("app"));
        assert!(copy_journal_path(target.path()).exists());
        // Moved out, the same command finishes.
        std::fs::remove_dir_all(&photos).unwrap();
        std::fs::remove_dir_all(&app).unwrap();
        save_catalog(target.path(), &BTreeSet::new()).unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db2", "db3", "nusadb"],
        ));

        // A database the copy created, dropped on the source, holding a file that is no
        // database's: left in place, with the journal.
        let target = tempfile::tempdir().unwrap();
        copy_until(target.path(), source.path(), &key, "db2");
        manager(source.path())
            .drop_database("db2", false, "nusadb")
            .unwrap();
        let notes = base_dir(target.path(), "db2").join("notes.txt");
        std::fs::write(&notes, b"mine").unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not a file of a database"), "{err}");
        assert_eq!(std::fs::read(&notes).unwrap(), b"mine");
        assert!(copy_journal_path(target.path()).exists());
        std::fs::remove_file(&notes).unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db3", "nusadb"],
        ));
    }

    #[test]
    fn a_missing_source_or_one_that_is_no_data_directory_is_refused_creating_nothing() {
        let key = Arc::new(MasterKey::from_text(&"93".repeat(32)).unwrap());
        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("target");
        let missing = parent.path().join("missing");
        let err = run_copy(&target, &missing, &key).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
        assert!(!missing.exists());
        assert!(!target.exists());

        let plain = parent.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("file"), b"x").unwrap();
        let err = run_copy(&target, &plain, &key).unwrap_err().to_string();
        assert!(err.contains("is not a nusadb data directory"), "{err}");
        assert_eq!(
            std::fs::read_dir(&plain).unwrap().count(),
            1,
            "nothing is created in the source"
        );
        assert!(!target.exists());
    }

    #[test]
    fn a_rerun_under_another_key_is_refused() {
        let key = Arc::new(MasterKey::from_text(&"94".repeat(32)).unwrap());
        let other = Arc::new(MasterKey::from_text(&"95".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);
        let target = tempfile::tempdir().unwrap();
        copy_until(target.path(), source.path(), &key, "db2");
        let journal = std::fs::read_to_string(copy_journal_path(target.path())).unwrap();
        assert!(
            !journal.contains(&"94".repeat(32)),
            "the key itself is never stored"
        );
        assert!(!journal.contains(&key.fingerprint()));
        let before = files_of(target.path());
        let err = run_copy(target.path(), source.path(), &other)
            .unwrap_err()
            .to_string();
        assert!(err.contains("another encryption key"), "{err}");
        assert_eq!(files_of(target.path()), before);
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db2", "db3", "nusadb"],
        ));
    }

    /// A directory outside any target holding what a database keeps: the link's victim.
    #[cfg(unix)]
    fn victim() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("btree.wal"), b"victim log").unwrap();
        std::fs::write(dir.path().join("btree.wal.ckpt"), b"victim image").unwrap();
        std::fs::create_dir_all(dir.path().join("btree.wal.pages")).unwrap();
        std::fs::write(
            dir.path()
                .join("btree.wal.pages")
                .join("00000000000000000007-1-full.seg"),
            b"victim pages",
        )
        .unwrap();
        dir
    }

    /// Put a link to `to` at `<target>/base/<name>`, in place of whatever is there.
    #[cfg(unix)]
    fn link_database_dir(target: &Path, name: &str, to: &Path) {
        let at = base_dir(target, name);
        if std::fs::symlink_metadata(&at).is_ok() {
            std::fs::remove_dir_all(&at).unwrap();
        }
        std::os::unix::fs::symlink(to, &at).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_rerun_never_writes_through_a_database_directory_replaced_by_a_link() {
        let key = Arc::new(MasterKey::from_text(&"96".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);
        let target = tempfile::tempdir().unwrap();
        // Stopped once the journal names every database as the copy's own; db3 is not marked yet.
        copy_until_step(target.path(), source.path(), &key, CopyStep::Marked, "db2");
        assert!(
            std::fs::read_to_string(copy_journal_path(target.path()))
                .unwrap()
                .contains("created db3")
        );
        let victim = victim();
        let intact = files_of(victim.path());
        link_database_dir(target.path(), "db3", victim.path());
        let before = files_of(target.path());
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a link or a file"), "{err}");
        assert_eq!(files_of(victim.path()), intact, "the victim is untouched");
        assert!(!victim.path().join("btree.wal.incomplete").exists());
        assert!(!victim.path().join("btree.wal.lock").exists());
        assert!(
            std::fs::symlink_metadata(base_dir(target.path(), "db3"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(files_of(target.path()).len(), before.len());
        // The link moved out, the same command finishes.
        std::fs::remove_file(base_dir(target.path(), "db3")).unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db2", "db3", "nusadb"],
        ));
        assert_eq!(files_of(victim.path()), intact);
    }

    #[cfg(unix)]
    #[test]
    fn a_rerun_never_removes_through_a_dropped_database_directory_replaced_by_a_link() {
        let key = Arc::new(MasterKey::from_text(&"97".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 20);
        let target = tempfile::tempdir().unwrap();
        copy_until_step(target.path(), source.path(), &key, CopyStep::Registered, "");
        assert!(load_catalog(target.path()).unwrap().contains("db2"));
        manager(source.path())
            .drop_database("db2", false, "nusadb")
            .unwrap();
        let victim = victim();
        let intact = files_of(victim.path());
        link_database_dir(target.path(), "db2", victim.path());
        let catalog = load_catalog(target.path()).unwrap();
        let err = run_copy(target.path(), source.path(), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a link or a file"), "{err}");
        assert_eq!(files_of(victim.path()), intact, "the victim is untouched");
        assert!(!victim.path().join("btree.wal.lock").exists());
        assert_eq!(
            load_catalog(target.path()).unwrap(),
            catalog,
            "nothing was changed"
        );
        assert!(copy_journal_path(target.path()).exists());
        std::fs::remove_file(base_dir(target.path(), "db2")).unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        drop(assert_finished_target(
            target.path(),
            &key,
            &["db3", "nusadb"],
        ));
        assert_eq!(files_of(victim.path()), intact);
    }

    #[cfg(unix)]
    #[test]
    fn a_copy_into_a_target_whose_base_or_global_is_a_link_is_refused() {
        let key = Arc::new(MasterKey::from_text(&"98".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 5);
        for part in ["base", "global"] {
            let target = tempfile::tempdir().unwrap();
            let elsewhere = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(elsewhere.path(), target.path().join(part)).unwrap();
            let err = run_copy(target.path(), source.path(), &key)
                .unwrap_err()
                .to_string();
            assert!(err.contains("is a link or a file"), "{part}: {err}");
            assert_eq!(
                std::fs::read_dir(elsewhere.path()).unwrap().count(),
                0,
                "{part}"
            );
        }
    }

    #[test]
    fn only_named_database_files_are_removed_from_a_created_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = base_dir(dir.path(), "db9");
        std::fs::create_dir_all(db.join("btree.wal.pages")).unwrap();
        for f in [
            "btree.wal",
            "btree.wal.ckpt",
            "btree.wal.key",
            "btree.wal.incomplete",
        ] {
            std::fs::write(db.join(f), b"x").unwrap();
        }
        std::fs::write(
            db.join("btree.wal.pages")
                .join("00000000000000000001-a-full.seg"),
            b"x",
        )
        .unwrap();
        // A file with a database-like name that is not one of the database's: kept, refused.
        std::fs::write(db.join("btree.wal.backup"), b"mine").unwrap();
        let err = remove_copied_database_dir(dir.path(), "db9")
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not a file of a database"), "{err}");
        assert_eq!(std::fs::read(db.join("btree.wal.backup")).unwrap(), b"mine");
        assert!(db.join("btree.wal").exists());
        // So is a page file that is not a segment.
        std::fs::remove_file(db.join("btree.wal.backup")).unwrap();
        std::fs::write(db.join("btree.wal.pages").join("notes"), b"mine").unwrap();
        assert!(remove_copied_database_dir(dir.path(), "db9").is_err());
        assert!(db.join("btree.wal.pages").join("notes").exists());
        std::fs::remove_file(db.join("btree.wal.pages").join("notes")).unwrap();
        remove_copied_database_dir(dir.path(), "db9").unwrap();
        assert!(!db.exists());
    }

    /// A file outside every data directory with known contents, and a path beside it that does
    /// not exist.
    #[cfg(unix)]
    fn victim_file(dir: &Path) -> (PathBuf, PathBuf) {
        let file = dir.join("victim");
        std::fs::write(&file, b"victim bytes").unwrap();
        (file, dir.join("nowhere"))
    }

    #[cfg(unix)]
    #[test]
    fn a_new_copy_never_writes_its_scratch_files_through_a_link() {
        let key = Arc::new(MasterKey::from_text(&"99".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 5);
        for scratch in ["encrypt-copy.tmp", "databases.tmp", "format.tmp"] {
            for dangling in [false, true] {
                let outside = tempfile::tempdir().unwrap();
                let (file, nowhere) = victim_file(outside.path());
                let target = tempfile::tempdir().unwrap();
                std::fs::create_dir_all(target.path().join("global")).unwrap();
                let to = if dangling { &nowhere } else { &file };
                std::os::unix::fs::symlink(to, target.path().join("global").join(scratch)).unwrap();
                run_copy(target.path(), source.path(), &key).unwrap();
                assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes", "{scratch}");
                assert!(!nowhere.exists(), "{scratch}");
                drop(assert_finished_target(
                    target.path(),
                    &key,
                    &["db2", "db3", "nusadb"],
                ));
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_rerun_refuses_a_link_at_a_database_file_name_before_changing_anything() {
        let key = Arc::new(MasterKey::from_text(&"9a".repeat(32)).unwrap());
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 5);
        for name in [
            "btree.wal.incomplete",
            "btree.wal.key.tmp",
            "btree.wal.lock",
            "btree.wal.pages",
        ] {
            for dangling in [false, true] {
                let outside = tempfile::tempdir().unwrap();
                let (file, nowhere) = victim_file(outside.path());
                let target = tempfile::tempdir().unwrap();
                copy_until(target.path(), source.path(), &key, "db2");
                let at = base_dir(target.path(), "db3").join(name);
                if std::fs::symlink_metadata(&at).is_ok() {
                    std::fs::remove_file(&at).unwrap();
                }
                let to = if dangling { &nowhere } else { &file };
                std::os::unix::fs::symlink(to, &at).unwrap();
                let catalog = load_catalog(target.path()).unwrap();
                let err = run_copy(target.path(), source.path(), &key)
                    .unwrap_err()
                    .to_string();
                assert!(err.contains("is a link"), "{name}: {err}");
                assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes", "{name}");
                assert!(!nowhere.exists(), "{name}");
                assert_eq!(load_catalog(target.path()).unwrap(), catalog, "{name}");
                std::fs::remove_file(&at).unwrap();
                run_copy(target.path(), source.path(), &key).unwrap();
                drop(assert_finished_target(
                    target.path(),
                    &key,
                    &["db2", "db3", "nusadb"],
                ));
                assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes", "{name}");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_rotation_never_writes_an_envelope_scratch_file_through_a_link() {
        let key = Arc::new(MasterKey::from_text(&"9b".repeat(32)).unwrap());
        let new = MasterKey::from_text(&"9c".repeat(32)).unwrap();
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 5);
        let target = tempfile::tempdir().unwrap();
        run_copy(target.path(), source.path(), &key).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let (file, nowhere) = victim_file(outside.path());
        std::os::unix::fs::symlink(
            &file,
            base_dir(target.path(), "db2").join("btree.wal.key.tmp"),
        )
        .unwrap();
        std::os::unix::fs::symlink(
            &nowhere,
            base_dir(target.path(), "db3").join("btree.wal.key.tmp"),
        )
        .unwrap();
        std::fs::hard_link(
            &file,
            base_dir(target.path(), "nusadb").join("btree.wal.key.tmp"),
        )
        .unwrap();
        let done = rotate_master_key(target.path(), &[], &key, &new).unwrap();
        assert_eq!(done.rewrapped, 3);
        assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes");
        assert!(!nowhere.exists());
        drop(assert_finished_target(
            target.path(),
            &Arc::new(new),
            &["db2", "db3", "nusadb"],
        ));
    }

    #[cfg(unix)]
    #[test]
    fn a_lock_file_that_is_a_link_is_refused_and_creates_nothing() {
        let outside = tempfile::tempdir().unwrap();
        let (_, nowhere) = victim_file(outside.path());
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("global")).unwrap();
        std::os::unix::fs::symlink(&nowhere, dir.path().join("global").join("cluster.lock"))
            .unwrap();
        let err = lock_cluster(dir.path()).unwrap_err().to_string();
        assert!(err.contains("is a link"), "{err}");
        std::os::unix::fs::symlink(&nowhere, dir.path().join("btree.wal.lock")).unwrap();
        let err = nusadb_btree::lock_database(&dir.path().join("btree.wal"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a link"), "{err}");
        assert!(!nowhere.exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_rotation_refuses_a_linked_envelope_before_rewrapping_anything() {
        let key = Arc::new(MasterKey::from_text(&"9d".repeat(32)).unwrap());
        let new = MasterKey::from_text(&"9e".repeat(32)).unwrap();
        let source = tempfile::tempdir().unwrap();
        three_databases(source.path(), 5);
        let outside = tempfile::tempdir().unwrap();
        for linked in ["envelope", "directory"] {
            let target = tempfile::tempdir().unwrap();
            run_copy(target.path(), source.path(), &key).unwrap();
            let (from, to) = if linked == "envelope" {
                (
                    base_dir(target.path(), "db2").join("btree.wal.key"),
                    outside.path().join("db2.key"),
                )
            } else {
                (base_dir(target.path(), "db2"), outside.path().join("db2"))
            };
            std::fs::rename(&from, &to).unwrap();
            std::os::unix::fs::symlink(&to, &from).unwrap();
            let before = files_of(target.path());
            let outside_before = files_of(outside.path());
            let err = rotate_master_key(target.path(), &[], &key, &new)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("is a link") && err.contains("db2"),
                "{linked}: {err}"
            );
            assert_eq!(
                files_of(target.path()),
                before,
                "{linked}: nothing rewrapped"
            );
            assert_eq!(files_of(outside.path()), outside_before, "{linked}");
            std::fs::remove_file(&from).unwrap();
            std::fs::rename(&to, &from).unwrap();
            // Put back, the rotation goes through every envelope.
            assert_eq!(
                rotate_master_key(target.path(), &[], &key, &new)
                    .unwrap()
                    .rewrapped,
                3
            );
        }
    }

    #[test]
    fn a_journal_with_a_line_break_in_its_source_is_refused_as_damaged() {
        let target = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(target.path().join("global")).unwrap();
        std::fs::write(
            copy_journal_path(target.path()),
            format!("{COPY_JOURNAL_HEADER}\nsource /a\n/b\nkey 00\n"),
        )
        .unwrap();
        let err = CopyJournal::read(target.path()).unwrap_err().to_string();
        assert!(err.contains("not a readable copy journal"), "{err}");
    }

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

    /// `DROP DATABASE` would decide a prepared transaction waiting in the database, so it is refused
    /// while one waits, also after the server restarts, and allowed once it is finished.
    #[test]
    fn dropping_a_database_with_a_prepared_transaction_is_refused() {
        use nusadb_core::engine::IsolationLevel;
        let dir = tempfile::tempdir().unwrap();
        {
            let m = manager(dir.path());
            assert!(m.create("shop", false).unwrap());
            let engine = m.open("shop").unwrap().expect("shop engine");
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            engine.prepare_transaction(txn, "g", "u", "shop").unwrap();
            drop(engine);
            let err = m.drop_database("shop", false, "nusadb").unwrap_err();
            assert!(matches!(err, ClusterError::Protected(_)), "{err:?}");
            assert_eq!(err.sqlstate(), "55006");
        }
        // A new server over the same directory finds it prepared again and refuses again.
        let m = manager(dir.path());
        let err = m.drop_database("shop", false, "nusadb").unwrap_err();
        assert!(matches!(err, ClusterError::Protected(_)), "{err:?}");
        assert!(base_dir(dir.path(), "shop").exists());
        let engine = m.open("shop").unwrap().expect("shop engine");
        engine.rollback_prepared("g", None).unwrap();
        drop(engine);
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
    fn open_threads_the_prepared_transaction_limit_to_each_database_engine() {
        use nusadb_core::engine::IsolationLevel;
        // The --max-prepared-transactions wiring: a limit of 0 reaches the engine, which then
        // refuses PREPARE TRANSACTION; no limit keeps the engine's default, which allows it.
        let prepare = |limit: Option<usize>| {
            let tmp = tempfile::tempdir().unwrap();
            let manager = DatabaseManager::open(
                tmp.path(),
                "nusadb",
                None,
                None,
                NO_AUTOANALYZE,
                DurabilityOptions {
                    max_prepared_transactions: limit,
                    ..DurabilityOptions::default()
                },
            )
            .expect("open cluster");
            let engine = manager.open("nusadb").unwrap().expect("default engine");
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            engine.prepare_transaction(txn, "g", "u", "nusadb")
        };
        let refused = prepare(Some(0)).unwrap_err();
        assert!(
            matches!(&refused, nusadb_core::Error::Io(e) if e.kind() == io::ErrorKind::InvalidInput),
            "{refused:?}"
        );
        prepare(None).unwrap();
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

    /// Opening a database written by a build that filed column defaults under bare table names
    /// repairs it before any session runs: the repair finds nothing left to do afterwards.
    #[test]
    fn opening_a_database_repairs_misfiled_column_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        drop(manager(tmp.path()));
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../nusadb-e2e/tests/fixtures/misfiled_column_defaults.wal"),
            base_dir(tmp.path(), "nusadb").join("btree.wal"),
        )
        .unwrap();
        let m = manager(tmp.path());
        let engine = m.open("nusadb").unwrap().unwrap();
        let report = nusadb_sql::repair_misfiled_column_defaults(&*engine).unwrap();
        assert!(!report.ran, "the open already repaired it: {report:?}");
        let app_e = engine.lookup_table_in("app", "e").unwrap().unwrap();
        assert!(app_e.columns.iter().any(|c| c.name == "flag"));
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
                wal_archive: Some(archive_root.clone()),
                ..DurabilityOptions::default()
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
                standby: Some(cfg.clone()),
                ..DurabilityOptions::default()
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
            standby_tick(&btree, &archive_root.join("shop"), cfg.max_pause, None),
            StandbyTick::Idle { .. }
        ));
        // The primary writes on and checkpoints; one tick brings the standby level.
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        engine.insert(txn, table, b"two").unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
        assert!(matches!(
            standby_tick(&btree, &archive_root.join("shop"), cfg.max_pause, None),
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
                wal_archive: Some(archive_root.clone()),
                ..DurabilityOptions::default()
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

    fn try_manager(dir: &Path) -> io::Result<DatabaseManager> {
        DatabaseManager::open(
            dir,
            "nusadb",
            None,
            None,
            NO_AUTOANALYZE,
            DurabilityOptions::default(),
        )
    }

    #[test]
    fn a_cluster_records_its_format_and_one_in_a_newer_format_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        drop(manager(tmp.path()));
        let stamp = tmp.path().join("global").join("format");
        assert_eq!(
            std::fs::read_to_string(&stamp).unwrap(),
            cluster_format_text()
        );

        std::fs::write(
            &stamp,
            format!(
                "nusadb cluster format {}\nwritten by nusadb 9.0.0\n",
                CLUSTER_FORMAT_VERSION + 1
            ),
        )
        .unwrap();
        let catalog = std::fs::read(tmp.path().join("global").join("databases")).unwrap();
        let Err(err) = try_manager(tmp.path()) else {
            panic!("a newer cluster format must be refused");
        };
        let message = err.to_string();
        assert!(message.contains("written by nusadb 9.0.0"), "{message}");
        assert!(message.contains("nothing was changed"), "{message}");
        assert_eq!(
            std::fs::read(tmp.path().join("global").join("databases")).unwrap(),
            catalog
        );
    }

    #[test]
    fn a_cluster_from_before_the_format_was_recorded_is_stamped() {
        let tmp = tempfile::tempdir().unwrap();
        drop(manager(tmp.path()));
        let stamp = tmp.path().join("global").join("format");
        std::fs::remove_file(&stamp).unwrap();
        drop(manager(tmp.path()));
        assert_eq!(
            std::fs::read_to_string(&stamp).unwrap(),
            cluster_format_text()
        );
    }

    #[test]
    fn a_database_in_a_newer_data_format_is_refused_at_start_up() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let m = manager(tmp.path());
            m.open("nusadb").unwrap();
        }
        let wal = base_dir(tmp.path(), "nusadb").join("btree.wal");
        let mut name = wal.as_os_str().to_owned();
        name.push(".format");
        std::fs::write(
            std::path::PathBuf::from(name),
            format!(
                "nusadb data format {}\nwritten by nusadb 9.0.0\n",
                nusadb_btree::format::FORMAT_VERSION + 1
            ),
        )
        .unwrap();
        let cluster_stamp = tmp.path().join("global").join("format");
        std::fs::write(
            &cluster_stamp,
            "nusadb cluster format 1\nwritten by nusadb 0.0.1\n",
        )
        .unwrap();
        let Err(err) = try_manager(tmp.path()) else {
            panic!("a database in a newer data format must be refused at start-up");
        };
        assert!(err.to_string().contains("data format"), "{err}");
        assert_eq!(
            std::fs::read_to_string(&cluster_stamp).unwrap(),
            "nusadb cluster format 1\nwritten by nusadb 0.0.1\n",
            "a refused start-up records nothing"
        );
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

    /// A locked database directory is removed whole: its files and subdirectories while the lock
    /// is held, then the lock file itself once released.
    #[test]
    fn a_locked_database_directory_is_removed_whole() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("db");
        std::fs::create_dir_all(dir.join("btree.wal.pages")).unwrap();
        std::fs::write(dir.join("btree.wal"), b"log").unwrap();
        std::fs::write(dir.join("btree.wal.ckpt"), b"image").unwrap();
        std::fs::write(dir.join("btree.wal.pages").join("0.seg"), b"pages").unwrap();
        let lock = lock_database_dir(&dir, "db").unwrap();
        assert!(dir.join("btree.wal.lock").exists());

        remove_locked_database_dir(&dir, lock).unwrap();
        assert!(!dir.exists());
    }

    /// The health list reports a database that stopped after a storage error, and only that one:
    /// another database on the same server keeps serving and is reported healthy.
    #[test]
    fn database_health_reports_a_stopped_database_and_no_other() {
        use nusadb_core::engine::{IsolationLevel, TableDef};
        use nusadb_core::{ColumnDef, ColumnType, PAGE_SIZE};

        let tmp = tempfile::tempdir().unwrap();
        let m = manager(tmp.path());
        m.create("shop", false).unwrap();
        assert!(m.database_health().is_empty(), "nothing is open yet");

        // Give `shop` a checkpointed table, written before the manager opens it, so every page
        // lives in the image's segments; then damage every page there.
        let wal = base_dir(tmp.path(), "shop").join("btree.wal");
        {
            let engine = BtreeEngine::open(&wal).unwrap();
            let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
            let def = TableDef {
                schema: "public".to_owned(),
                name: "t".to_owned(),
                columns: vec![ColumnDef {
                    name: "v".to_owned(),
                    ty: ColumnType::Bytes,
                    nullable: false,
                }],
            };
            let table = engine.create_table(txn, &def).unwrap();
            for i in 0..2000_u32 {
                engine.insert(txn, table, &i.to_le_bytes()).unwrap();
            }
            engine.commit(txn).unwrap();
            engine.checkpoint().unwrap();
        }
        let pages = PathBuf::from(format!("{}.pages", wal.display()));
        for entry in std::fs::read_dir(pages).unwrap() {
            let path = entry.unwrap().path();
            let mut bytes = std::fs::read(&path).unwrap();
            for page in 0..bytes.len() / PAGE_SIZE {
                bytes[page * PAGE_SIZE + PAGE_SIZE / 2] ^= 0x5A;
            }
            std::fs::write(&path, &bytes).unwrap();
        }

        let shop = m.open("shop").unwrap().unwrap();
        let main = m.open("nusadb").unwrap().unwrap();
        // Both are listed once open. `shop` may already be stopped here: its background purge
        // can reach a damaged page before the insert below does.
        let names: Vec<String> = m.database_health().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["nusadb", "shop"]);

        // A change to `shop` reaches a damaged page (unless the purge already did) and stops it.
        let table = shop.lookup_table("t").unwrap().unwrap().id;
        let err = shop
            .begin(IsolationLevel::ReadCommitted)
            .and_then(|txn| {
                let result = shop.insert(txn, table, b"new");
                let _ = shop.rollback(txn);
                result
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("stopped after a storage error"), "{err}");
        assert_eq!(
            m.database_health(),
            vec![("nusadb".to_owned(), false), ("shop".to_owned(), true)]
        );

        // The other database still serves.
        let txn = main.begin(IsolationLevel::ReadCommitted).unwrap();
        main.commit(txn).unwrap();

        // A dropped database leaves the list.
        drop(shop);
        assert!(m.drop_database("shop", false, "nusadb").unwrap());
        assert_eq!(m.database_health(), vec![("nusadb".to_owned(), false)]);
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
