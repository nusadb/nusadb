//! Scheduled backups of the whole cluster, and pruning of the checkpoint archive.
//!
//! A backup is a directory `<backup-dir>/<UTC moment>/` laid out like a data directory: the
//! cluster's `global/databases` and `global/format`, and for every database its newest checkpoint
//! image with the page segments it reads from and its format file (see
//! [`nusadb_btree::BtreeEngine::backup_into`]). Each database is checkpointed first, holding new
//! transactions for at most the checkpoint pause, so the copy is current; when a long
//! transaction keeps it from quiescing, the copy is of its last image, and the log says how old.
//! Each database is consistent on its own; databases are copied one after another.
//!
//! A backup is built under `.partial-<moment>` and renamed to its final name only once complete,
//! with a `BACKUP` manifest written last inside it, so a directory with a plain moment name is
//! always a whole backup. The newest `keep` backups are kept and older ones removed, as are
//! partial ones an interrupted run left behind.

#![allow(
    clippy::redundant_pub_crate,
    reason = "this is a private module of the server binary; its items are `pub(crate)` so the \
              rest of the binary can reach them"
)]

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::database_manager::{CheckpointTick, DatabaseManager, checkpoint_tick};

/// The scheduled backup settings.
#[derive(Debug, Clone)]
pub(crate) struct BackupConfig {
    /// Where backups go.
    pub(crate) dir: PathBuf,
    /// Time between backups.
    pub(crate) interval: Duration,
    /// How many complete backups to keep.
    pub(crate) keep: usize,
    /// The longest a backup holds new transactions so a database can checkpoint first.
    pub(crate) max_pause: Duration,
}

/// What the metrics endpoint reports about backups and archive pruning.
#[derive(Debug, Default)]
pub(crate) struct BackupStatus {
    /// When the last backup completed, seconds since the Unix epoch; 0 before the first.
    last_success: AtomicU64,
    /// How long the last completed backup took, milliseconds.
    last_duration_ms: AtomicU64,
    /// Backups that failed since the server started.
    failures: AtomicU64,
    /// Archive prunes that failed since the server started.
    prune_failures: AtomicU64,
}

impl BackupStatus {
    /// The status in the Prometheus text format.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "# HELP nusadb_backup_last_success_timestamp_seconds When the last scheduled backup \
             completed (0 before the first).\n\
             # TYPE nusadb_backup_last_success_timestamp_seconds gauge\n\
             nusadb_backup_last_success_timestamp_seconds {}\n\
             # HELP nusadb_backup_last_duration_seconds How long the last completed backup took.\n\
             # TYPE nusadb_backup_last_duration_seconds gauge\n\
             nusadb_backup_last_duration_seconds {}\n\
             # HELP nusadb_backup_failures_total Scheduled backups that failed.\n\
             # TYPE nusadb_backup_failures_total counter\n\
             nusadb_backup_failures_total {}\n\
             # HELP nusadb_archive_prune_failures_total Checkpoint archive prunes that failed.\n\
             # TYPE nusadb_archive_prune_failures_total counter\n\
             nusadb_archive_prune_failures_total {}\n",
            self.last_success.load(Ordering::Relaxed),
            millis_as_seconds(self.last_duration_ms.load(Ordering::Relaxed)),
            self.failures.load(Ordering::Relaxed),
            self.prune_failures.load(Ordering::Relaxed),
        );
        out
    }
}

/// Milliseconds as decimal seconds, for a metric.
fn millis_as_seconds(ms: u64) -> String {
    format!("{}.{:03}", ms / 1000, ms % 1000)
}

/// Seconds since the Unix epoch now.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `secs` since the Unix epoch as a UTC moment `YYYYMMDDTHHMMSSZ`, which sorts as it reads.
pub(crate) fn moment_name(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX);
    let rem = secs % 86_400;
    // Days since 1970-01-01 to a civil date (proleptic Gregorian).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Whether `name` is a complete backup's directory name.
fn is_moment_name(name: &str) -> bool {
    name.len() == 16
        && name.as_bytes().get(8) == Some(&b'T')
        && name.ends_with('Z')
        && name
            .bytes()
            .enumerate()
            .all(|(i, b)| i == 8 || i == 15 || b.is_ascii_digit())
}

/// Take one backup of every database of `manager` into a new directory under `config.dir`, and
/// apply the retention. Returns the backup's directory.
///
/// # Errors
/// Any database that cannot be copied fails the whole backup, which then leaves nothing under a
/// final name; I/O errors.
pub(crate) fn run_backup(manager: &DatabaseManager, config: &BackupConfig) -> io::Result<PathBuf> {
    std::fs::create_dir_all(&config.dir)?;
    remove_partials(&config.dir);
    let mut moment = unix_now();
    // Two backups within one second take the next free second's name.
    while config.dir.join(moment_name(moment)).exists() {
        moment += 1;
    }
    let name = moment_name(moment);
    let partial = config.dir.join(format!(".partial-{name}"));
    let built = build(manager, config, &partial, &name);
    if let Err(e) = built {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(e);
    }
    let complete = config.dir.join(&name);
    std::fs::rename(&partial, &complete)?;
    sync_dir(&config.dir)?;
    apply_retention(&config.dir, config.keep)?;
    Ok(complete)
}

/// Build the backup in `partial`.
fn build(
    manager: &DatabaseManager,
    config: &BackupConfig,
    partial: &Path,
    name: &str,
) -> io::Result<()> {
    std::fs::create_dir_all(partial.join("global"))?;
    let mut manifest = format!(
        "nusadb backup\nwritten by nusadb {}\ntaken {name}\n",
        env!("CARGO_PKG_VERSION")
    );
    let mut synced_dirs: Vec<PathBuf> = Vec::new();
    for db in manager.database_names() {
        let Some((engine, relative)) = manager.engine_with_path(&db)? else {
            // Dropped since the list was read.
            continue;
        };
        // Checkpoint first so the copy is current; the copy is of the last image otherwise.
        match checkpoint_tick(&engine, 0, Some(config.max_pause)) {
            CheckpointTick::Done { .. }
            | CheckpointTick::NoLog
            | CheckpointTick::BelowThreshold { .. } => {},
            CheckpointTick::Busy { active, .. } => tracing::warn!(
                database = %db,
                active_transactions = active.unwrap_or(0),
                "backup: the database did not go quiet for a checkpoint; copying its last image"
            ),
            CheckpointTick::Failed { error } => tracing::warn!(
                database = %db,
                error = %error,
                "backup: the checkpoint before the copy failed; copying the last image"
            ),
        }
        let target = partial.join(&relative);
        let info = engine
            .backup_into(&target)
            .map_err(|e| io::Error::other(format!("database {db}: {e}")))?;
        // The directories between the backup and the database's files, made durable below.
        let mut dir = target.parent();
        while let Some(d) = dir.filter(|d| *d != partial) {
            if !synced_dirs.iter().any(|s| s == d) {
                synced_dirs.push(d.to_path_buf());
            }
            dir = d.parent();
        }
        let _ = writeln!(
            manifest,
            "database {db} position {} image {}",
            info.covered_lsn,
            info.image_unix_ms
                .map_or_else(|| "none".to_owned(), |ms| moment_name(ms / 1000)),
        );
        if let Some(ms) = info.image_unix_ms {
            let age = unix_now().saturating_sub(ms / 1000);
            if age > 60 {
                tracing::warn!(database = %db, age_seconds = age, "backup: copied an image older than a minute");
            }
        }
    }
    // The cluster catalog and format, read after the databases: a database created meanwhile is
    // listed without its files, which a restore reports, rather than the other way round.
    for file in ["databases", "format"] {
        let from = manager.root().join("global").join(file);
        if from.exists() {
            let to = partial.join("global").join(file);
            std::fs::copy(&from, &to)?;
            std::fs::File::open(&to)?.sync_all()?;
        }
    }
    // Every file is durable; now the directories that name them, deepest first.
    for dir in synced_dirs.iter().rev() {
        sync_dir(dir)?;
    }
    sync_dir(&partial.join("global"))?;
    let path = partial.join("BACKUP");
    std::fs::write(&path, manifest)?;
    std::fs::File::open(&path)?.sync_all()?;
    sync_dir(partial)
}

/// Remove what an interrupted backup left: every `.partial-*` directory.
fn remove_partials(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(".partial-") {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Keep the newest `keep` complete backups in `dir` and remove the older ones. Only directories
/// named as a backup and holding a manifest count; nothing else in `dir` is touched.
fn apply_retention(dir: &Path, keep: usize) -> io::Result<()> {
    let mut backups: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| is_moment_name(n) && dir.join(n).join("BACKUP").exists())
        .collect();
    backups.sort();
    let excess = backups.len().saturating_sub(keep.max(1));
    for name in backups.into_iter().take(excess) {
        std::fs::remove_dir_all(dir.join(&name))?;
        tracing::info!(backup = %name, "backup retention removed an old backup");
    }
    Ok(())
}

/// Prune every database's checkpoint archive under `root` to `retain`, and remove the archives
/// of dropped databases (`<name>.dropped-<ms>`) set aside before it. Returns how many archives
/// failed to prune.
pub(crate) fn prune_archives(root: &Path, retain: Duration) -> usize {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    let keep_from = now_ms.saturating_sub(u64::try_from(retain.as_millis()).unwrap_or(u64::MAX));
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut failed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !path.is_dir() {
            continue;
        }
        // Only a database's archive (it holds a format file or an image); anything else is left.
        let is_archive = path.join("format").exists()
            || std::fs::read_dir(&path).is_ok_and(|entries| {
                entries
                    .flatten()
                    .any(|e| e.file_name().to_string_lossy().ends_with(".ckpt"))
            });
        if !is_archive && !name.contains(".dropped-") {
            continue;
        }
        if let Some((_, ms)) = name.split_once(".dropped-") {
            if ms.parse::<u64>().is_ok_and(|ms| ms <= keep_from) {
                match std::fs::remove_dir_all(&path) {
                    Ok(()) => {
                        tracing::info!(archive = %name, "archive retention removed a dropped database's archive");
                    },
                    Err(e) => {
                        failed += 1;
                        tracing::warn!(archive = %name, error = %e, "could not remove a dropped database's archive");
                    },
                }
            }
            continue;
        }
        match nusadb_btree::prune_archive(&path, keep_from) {
            Ok(stats) if stats == nusadb_btree::PruneStats::default() => {},
            Ok(stats) => tracing::info!(
                database = %name,
                images = stats.images,
                log_segments = stats.logs,
                page_segments = stats.page_segments,
                superseded = stats.superseded,
                "archive retention pruned the checkpoint archive"
            ),
            Err(e) => {
                failed += 1;
                tracing::warn!(database = %name, error = %e, "archive retention could not prune the checkpoint archive");
            },
        }
    }
    failed
}

/// Make a directory's entries durable, where the platform lets a directory be synced.
fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Run scheduled backups and archive pruning in a background thread for as long as the server
/// runs. Each is skipped when not configured.
pub(crate) fn spawn(
    manager: Arc<DatabaseManager>,
    backup: Option<BackupConfig>,
    archive_retain: Option<Duration>,
    status: Arc<BackupStatus>,
) {
    if backup.is_none() && archive_retain.is_none() {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("backup".to_owned())
        .spawn(move || {
            const PRUNE_EVERY: Duration = Duration::from_hours(1);
            // Wake often enough for the interval, and at least every 30 seconds.
            let tick = backup
                .as_ref()
                .map_or(Duration::from_secs(30), |b| b.interval.min(Duration::from_secs(30)));
            let mut next_backup = backup.as_ref().map(|b| SystemTime::now() + b.interval);
            let mut next_prune = SystemTime::now();
            loop {
                let now = SystemTime::now();
                if let (Some(config), Some(due)) = (&backup, next_backup)
                    && now >= due
                {
                    let started = std::time::Instant::now();
                    match run_backup(&manager, config) {
                        Ok(dir) => {
                            let took = started.elapsed();
                            status.last_success.store(unix_now(), Ordering::Relaxed);
                            status.last_duration_ms.store(
                                u64::try_from(took.as_millis()).unwrap_or(u64::MAX),
                                Ordering::Relaxed,
                            );
                            tracing::info!(backup = %dir.display(), took_ms = took.as_millis(), "scheduled backup complete");
                        },
                        Err(e) => {
                            status.failures.fetch_add(1, Ordering::Relaxed);
                            tracing::error!(error = %e, "scheduled backup failed; it is retried at the next interval");
                        },
                    }
                    next_backup = Some(SystemTime::now() + config.interval);
                }
                if let (Some(retain), Some(root)) = (archive_retain, manager.archive_root())
                    && now >= next_prune
                {
                    let failed = prune_archives(root, retain);
                    status
                        .prune_failures
                        .fetch_add(u64::try_from(failed).unwrap_or(u64::MAX), Ordering::Relaxed);
                    next_prune = SystemTime::now() + PRUNE_EVERY;
                }
                std::thread::sleep(tick);
            }
        });
    if let Err(e) = spawned {
        tracing::error!(error = %e, "the backup thread did not start; no scheduled backups or archive pruning");
    }
}

#[cfg(test)]
mod tests {
    use nusadb_core::engine::{ColumnDef, TableDef};
    use nusadb_core::{ColumnType, IsolationLevel};
    use nusadb_wire::DatabaseCluster;

    use super::*;
    use crate::database_manager::{AutoAnalyzeConfig, DurabilityOptions};

    fn manager(dir: &Path, archive: Option<&Path>) -> DatabaseManager {
        DatabaseManager::open(
            dir,
            "nusadb",
            None,
            None,
            AutoAnalyzeConfig {
                interval: None,
                scale: 0.1,
                base: 50,
            },
            DurabilityOptions {
                wal_archive: archive.map(Path::to_path_buf),
                ..DurabilityOptions::default()
            },
        )
        .unwrap()
    }

    /// Database `db` with table `t` holding rows `0..n`.
    fn with_rows(m: &DatabaseManager, db: &str, n: u32) {
        if db != "nusadb" {
            m.create(db, false).unwrap();
        }
        let engine = m.open(db).unwrap().unwrap();
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
        for i in 0..n {
            engine.insert(txn, table, &i.to_be_bytes()).unwrap();
        }
        engine.commit(txn).unwrap();
    }

    fn row_count(m: &DatabaseManager, db: &str) -> usize {
        let engine = m.open(db).unwrap().unwrap();
        let table = engine.lookup_table("t").unwrap().unwrap().id;
        let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
        let mut scan = engine.scan(txn, table).unwrap();
        let mut n = 0;
        while scan.try_next().unwrap().is_some() {
            n += 1;
        }
        drop(scan);
        engine.commit(txn).unwrap();
        n
    }

    fn config(dir: &Path, keep: usize) -> BackupConfig {
        BackupConfig {
            dir: dir.to_path_buf(),
            interval: Duration::from_mins(1),
            keep,
            max_pause: Duration::from_secs(1),
        }
    }

    /// Copy a directory tree, the way an operator restores a backup into a fresh data directory.
    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.path().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn a_backup_restores_every_database_as_a_data_directory() {
        let data = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let m = manager(data.path(), None);
        with_rows(&m, "nusadb", 30);
        with_rows(&m, "shop", 70);
        let dir = run_backup(&m, &config(backups.path(), 7)).unwrap();
        // Rows committed after the backup are not in it.
        with_rows(&m, "late", 5);
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        assert!(is_moment_name(&name), "{name}");
        let manifest = std::fs::read_to_string(dir.join("BACKUP")).unwrap();
        assert!(manifest.starts_with("nusadb backup\n"), "{manifest}");
        assert!(manifest.contains("database nusadb position"), "{manifest}");
        assert!(manifest.contains("database shop position"), "{manifest}");
        assert!(dir.join("global").join("format").exists());

        let restored = tempfile::tempdir().unwrap();
        copy_tree(&dir, restored.path());
        std::fs::remove_file(restored.path().join("BACKUP")).unwrap();
        let r = manager(restored.path(), None);
        assert_eq!(r.list(), vec!["nusadb".to_owned(), "shop".to_owned()]);
        assert_eq!(row_count(&r, "nusadb"), 30);
        assert_eq!(row_count(&r, "shop"), 70);
    }

    #[test]
    fn retention_keeps_the_newest_backups_and_clears_partial_ones() {
        let data = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let m = manager(data.path(), None);
        with_rows(&m, "nusadb", 3);
        std::fs::create_dir_all(backups.path().join(".partial-20000101T000000Z")).unwrap();
        std::fs::create_dir_all(backups.path().join("not-a-backup")).unwrap();
        let mut taken = Vec::new();
        for _ in 0..4 {
            taken.push(run_backup(&m, &config(backups.path(), 2)).unwrap());
        }
        let mut left: Vec<String> = std::fs::read_dir(backups.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        let newest: Vec<String> = taken[2..]
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            left,
            [newest, vec!["not-a-backup".to_owned()]].concat(),
            "{left:?}"
        );
    }

    /// A backup that fails part way (a database's page segments cannot be read) leaves no
    /// backup and no partial directory, and the next one succeeds.
    #[cfg(unix)]
    #[test]
    fn a_failed_backup_leaves_nothing_behind() {
        use std::os::unix::fs::PermissionsExt;

        let data = tempfile::tempdir().unwrap();
        let backups = tempfile::tempdir().unwrap();
        let m = manager(data.path(), None);
        with_rows(&m, "nusadb", 3);
        with_rows(&m, "shop", 3);
        run_backup(&m, &config(backups.path(), 5)).unwrap();
        let pages = data
            .path()
            .join("base")
            .join("shop")
            .join("btree.wal.pages");
        let mode = std::fs::metadata(&pages).unwrap().permissions().mode();
        std::fs::set_permissions(&pages, std::fs::Permissions::from_mode(0o000)).unwrap();
        let failed = run_backup(&m, &config(backups.path(), 5));
        std::fs::set_permissions(&pages, std::fs::Permissions::from_mode(mode)).unwrap();
        let err = failed.unwrap_err().to_string();
        assert!(err.contains("database shop"), "{err}");
        let names: Vec<String> = std::fs::read_dir(backups.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 1, "only the first backup: {names:?}");
        assert!(!names[0].starts_with(".partial-"), "{names:?}");
        run_backup(&m, &config(backups.path(), 5)).unwrap();
    }

    #[test]
    fn pruning_removes_dropped_archives_older_than_the_window() {
        let archive = tempfile::tempdir().unwrap();
        let old_ms = (unix_now() - 3 * 86_400) * 1000;
        let new_ms = unix_now() * 1000;
        for name in [
            format!("gone.dropped-{old_ms}"),
            format!("recent.dropped-{new_ms}"),
        ] {
            std::fs::create_dir_all(archive.path().join(name)).unwrap();
        }
        assert_eq!(prune_archives(archive.path(), Duration::from_hours(24)), 0);
        let left: Vec<String> = std::fs::read_dir(archive.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, vec![format!("recent.dropped-{new_ms}")]);
    }

    #[test]
    fn moment_names_are_utc_and_sort_as_they_read() {
        assert_eq!(moment_name(0), "19700101T000000Z");
        assert_eq!(moment_name(951_782_400), "20000229T000000Z");
        assert_eq!(moment_name(1_791_331_199), "20261006T235959Z");
        assert!(is_moment_name(&moment_name(1_791_331_199)));
        assert!(!is_moment_name(".partial-20260924T235959Z"));
        assert!(!is_moment_name("20260924X235959Z"));
    }
}
