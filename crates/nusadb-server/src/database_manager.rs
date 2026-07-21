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
    ) -> io::Result<Self> {
        let root = data_dir.as_ref().to_path_buf();
        let default_name = default_name.into();
        // Detect the legacy single-database layout (a WAL at the root) before creating `base/`.
        let legacy_root = root.join("nusadb.wal").exists();
        std::fs::create_dir_all(root.join("global"))?;
        std::fs::create_dir_all(root.join("base"))?;

        let mut databases = load_catalog(&root)?;
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
            default_name,
            legacy_root,
            max_txn_write_bytes,
            max_resident_bytes,
            autoanalyze,
            state: Mutex::new(ManagerState {
                databases,
                engines: HashMap::new(),
            }),
        })
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
            BtreeEngine::open(wal)
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
        spawn_analyze_scheduler(&engine, name, self.autoanalyze);
        let engine: Arc<dyn StorageEngine> = engine;
        state.engines.insert(name.to_owned(), Arc::clone(&engine));
        Ok(engine)
    }
}

/// How often the background scheduler purges a btree database.
const PURGE_INTERVAL: Duration = Duration::from_secs(10);

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
                match engine.purge() {
                    Ok(stats) => tracing::trace!(?stats, db = %db, "purge pass"),
                    Err(e) => tracing::warn!(error = %e, db = %db, "purge pass failed"),
                }
                drop(engine);
                std::thread::sleep(PURGE_INTERVAL);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "purge scheduler did not start; run purge manually");
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
            std::fs::remove_dir_all(&dir).map_err(|e| ClusterError::Io(e.to_string()))?;
        }
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
        if name == self.default_name {
            return Err(ClusterError::Unsupported(format!(
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
        // to be removed). The purge scheduler briefly upgrades its weak handle during a pass, so
        // holders are counted through a weak handle after dropping the cache's reference: a
        // transient purge hold drains within the grace window, while a connection's hold persists
        // — only the latter is `InUse`. Waiting until the count reaches zero also guarantees the
        // engine (and its open WAL handle) is fully dropped before the directory is removed.
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
        state.databases.remove(name);
        save_catalog(&self.root, &state.databases).map_err(|e| ClusterError::Io(e.to_string()))?;
        // Remove the database's storage last: the catalog no longer lists it, so a crash here leaves
        // an orphan directory (harmless — re-`CREATE` reuses it) rather than a dangling catalog entry.
        std::fs::remove_dir_all(base_dir(&self.root, name))
            .map_err(|e| ClusterError::Io(e.to_string()))?;
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
        DatabaseManager::open(dir, "nusadb", None, None, NO_AUTOANALYZE).expect("open cluster")
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
        let bounded = DatabaseManager::open(tmp.path(), "nusadb", Some(40), None, NO_AUTOANALYZE)
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
        let free = DatabaseManager::open(tmp2.path(), "nusadb", None, None, NO_AUTOANALYZE)
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
        let bounded = DatabaseManager::open(tmp.path(), "nusadb", None, Some(1), NO_AUTOANALYZE)
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
        let free = DatabaseManager::open(tmp2.path(), "nusadb", None, None, NO_AUTOANALYZE)
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
            Err(ClusterError::Unsupported(_))
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
        let m =
            DatabaseManager::open(tmp.path(), "nusadb", None, None, config).expect("open cluster");
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
