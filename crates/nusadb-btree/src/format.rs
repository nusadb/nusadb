//! The data format version of a database directory.
//!
//! Every database directory records, beside its log, the version of the data format its files
//! are written in and the release that last opened it: the file `<wal>.format`, holding
//!
//! ```text
//! nusadb data format 1
//! written by nusadb 0.1.0
//! ```
//!
//! The format version covers every file of the database: the log's frames and record kinds, the
//! checkpoint image, the page segments and the page and row layouts. A release that changes any of
//! them in a way an older release cannot read raises [`FORMAT_VERSION`]. An engine checks the file
//! before it reads anything else, so a directory written in a newer format is refused untouched; in
//! particular a log record an older release does not know is never mistaken for a torn tail and
//! cut off. A directory in an older format is brought up to date by the release's upgrade steps,
//! before it is opened. A directory with no format file was written before the version was
//! recorded, in format 1.
//!
//! A checkpoint archive records the same in `<archive>/format`: the newest format written into it.
//! A release refuses to restore, seed or ship from an archive in a newer format than it reads.

use std::path::{Path, PathBuf};

use nusadb_core::{Error, Result};

/// The data format this release writes.
pub const FORMAT_VERSION: u32 = 1;

/// One step bringing a directory from format `from` to `from + 1`, given its log path.
pub(crate) struct Upgrade {
    pub(crate) from: u32,
    pub(crate) run: fn(&Path) -> Result<()>,
}

/// The upgrade steps, in order. Format 1 is the first, so there are none yet.
pub(crate) const UPGRADES: &[Upgrade] = &[];

/// The format file beside the log: `<wal>.format`.
pub(crate) fn format_path(wal: &Path) -> PathBuf {
    beside(wal, ".format")
}

/// The path `<wal><suffix>`.
fn beside(wal: &Path, suffix: &str) -> PathBuf {
    let mut name = wal.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// The text of the format file for `version`, written by this release.
fn format_text(version: u32) -> String {
    format!(
        "nusadb data format {version}\nwritten by nusadb {}\n",
        env!("CARGO_PKG_VERSION")
    )
}

fn invalid(message: String) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}

/// The format and writer recorded in `text`, or `None` if it is not a format file.
fn parse(text: &str) -> Option<(u32, Option<&str>)> {
    let mut lines = text.lines();
    let version = lines
        .next()?
        .strip_prefix("nusadb data format ")?
        .trim()
        .parse()
        .ok()?;
    let writer = lines
        .next()
        .and_then(|line| line.strip_prefix("written by "))
        .map(str::trim);
    Some((version, writer))
}

/// The format the directory of the log at `wal` is in: the recorded one, format 1 for an existing
/// directory written before the format was recorded, or `None` for a database not yet created.
///
/// # Errors
/// A format file that cannot be read or is not a format file.
pub fn read_format(wal: &Path) -> Result<Option<u32>> {
    let path = format_path(wal);
    match std::fs::read_to_string(&path) {
        Ok(text) => parse(&text)
            .map(|(version, _)| Some(version))
            .ok_or_else(|| {
                invalid(format!(
                    "nusadb-btree: {} is not a data format file; restore it from a backup of the \
                 database (it names the format the database's files are written in)",
                    path.display()
                ))
            }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // The log, its checkpoint image `<wal>.ckpt` or its page segments `<wal>.pages/`.
            let existing =
                wal.exists() || beside(wal, ".ckpt").exists() || beside(wal, ".pages").exists();
            Ok(existing.then_some(1))
        },
        Err(e) => Err(e.into()),
    }
}

/// Check the directory of the log at `wal` before anything else of it is read: refuse a format
/// newer than this release writes, upgrade an older one step by step, and record the format and
/// this release as its writer. The caller holds the database lock.
///
/// # Errors
/// A newer format, an unreadable format file, a failed upgrade step, or an I/O error.
pub(crate) fn check_format(wal: &Path) -> Result<()> {
    check_with(wal, FORMAT_VERSION, UPGRADES)
}

/// [`check_format`] for a release that writes format `current` with upgrade `steps`.
fn check_with(wal: &Path, current: u32, steps: &[Upgrade]) -> Result<()> {
    let path = format_path(wal);
    let found = read_format(wal)?;
    let version = found.unwrap_or(current);
    if version > current {
        let writer = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| parse(&text).and_then(|(_, w)| w.map(str::to_owned)))
            .unwrap_or_else(|| "a newer release".to_owned());
        return Err(invalid(format!(
            "nusadb-btree: the database at {} is in data format {version}, written by {writer}; \
             this release (nusadb {}) reads data format {current} and older. Open it with \
             the release that wrote it or a newer one; its files were left untouched",
            wal.display(),
            env!("CARGO_PKG_VERSION"),
        )));
    }
    let mut at = version;
    while at < current {
        let step = steps.iter().find(|u| u.from == at).ok_or_else(|| {
            invalid(format!(
                "nusadb-btree: the database at {} is in data format {at}, which this release \
                 cannot upgrade; dump it with a release that reads it and restore it here",
                wal.display()
            ))
        })?;
        (step.run)(wal)?;
        at += 1;
        // Record each step as it completes, so an upgrade that stops resumes where it was.
        write_format(wal, at)?;
    }
    let text = format_text(current);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
        write_format(wal, current)?;
    }
    Ok(())
}

/// The format file of a checkpoint archive directory: `<archive>/format`.
pub(crate) fn archive_format_path(archive: &Path) -> PathBuf {
    archive.join("format")
}

/// Refuse a checkpoint archive written in a newer format than this release reads: restoring or
/// shipping from it would read log segments with records this release does not know. An archive
/// with no format file was written before the format was recorded.
///
/// # Errors
/// A newer format, a format file that is not one, or an I/O error.
pub(crate) fn check_archive(archive: &Path) -> Result<()> {
    let path = archive_format_path(archive);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let Some((version, writer)) = parse(&text) else {
        return Err(invalid(format!(
            "nusadb-btree: {} is not a data format file",
            path.display()
        )));
    };
    if version > FORMAT_VERSION {
        return Err(invalid(format!(
            "nusadb-btree: the archive at {} is in data format {version}, written by {}; this \
             release (nusadb {}) reads data format {FORMAT_VERSION} and older. Use the release \
             that wrote it or a newer one",
            archive.display(),
            writer.unwrap_or("a newer release"),
            env!("CARGO_PKG_VERSION"),
        )));
    }
    Ok(())
}

/// The format a checkpoint archive records: format 1 when it has no format file (written before
/// the format was recorded).
///
/// # Errors
/// As [`check_archive`].
pub(crate) fn archive_format(archive: &Path) -> Result<u32> {
    check_archive(archive)?;
    match std::fs::read_to_string(archive_format_path(archive)) {
        Ok(text) => Ok(parse(&text).map_or(1, |(version, _)| version)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(1),
        Err(e) => Err(e.into()),
    }
}

/// Check a checkpoint archive an engine is about to write to and record this release's format in
/// it.
///
/// # Errors
/// As [`check_archive`], or an I/O error writing the file.
pub(crate) fn stamp_archive(archive: &Path) -> Result<()> {
    check_archive(archive)?;
    std::fs::create_dir_all(archive)?;
    let path = archive_format_path(archive);
    let text = format_text(FORMAT_VERSION);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
        write_file(&path, &text)?;
    }
    Ok(())
}

/// Write the format file for `version`.
pub(crate) fn write_format(wal: &Path, version: u32) -> Result<()> {
    write_file(&format_path(wal), &format_text(version))
}

/// Write `text` to `path` atomically: a scratch file, synced, renamed over it.
fn write_file(path: &Path, text: &str) -> Result<()> {
    let path = path.to_path_buf();
    let mut scratch = path.clone().into_os_string();
    scratch.push(".tmp");
    let scratch = PathBuf::from(scratch);
    std::fs::write(&scratch, text)?;
    std::fs::File::open(&scratch)?.sync_all()?;
    std::fs::rename(&scratch, &path)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        #[cfg(unix)]
        std::fs::File::open(dir)?.sync_all()?;
        #[cfg(not(unix))]
        let _ = dir;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static RAN: Mutex<Vec<u32>> = Mutex::new(Vec::new());

    #[allow(
        clippy::unnecessary_wraps,
        reason = "an upgrade step's signature is fallible"
    )]
    fn step_1(_: &Path) -> Result<()> {
        RAN.lock().unwrap().push(1);
        Ok(())
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "an upgrade step's signature is fallible"
    )]
    fn step_2(_: &Path) -> Result<()> {
        RAN.lock().unwrap().push(2);
        Ok(())
    }

    fn failing_step_2(_: &Path) -> Result<()> {
        Err(invalid("step 2 failed".to_owned()))
    }

    #[test]
    fn an_older_format_is_upgraded_step_by_step_and_resumes_after_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        let wal = dir.path().join("btree.wal");
        write_format(&wal, 1).unwrap();

        // Step 2 fails: step 1 is recorded as done, the directory stays at format 2.
        let failing = [
            Upgrade {
                from: 1,
                run: step_1,
            },
            Upgrade {
                from: 2,
                run: failing_step_2,
            },
        ];
        assert!(check_with(&wal, 3, &failing).is_err());
        assert_eq!(read_format(&wal).unwrap(), Some(2));
        assert_eq!(*RAN.lock().unwrap(), vec![1]);

        // The next open resumes at step 2 and finishes at format 3.
        let steps = [
            Upgrade {
                from: 1,
                run: step_1,
            },
            Upgrade {
                from: 2,
                run: step_2,
            },
        ];
        check_with(&wal, 3, &steps).unwrap();
        assert_eq!(read_format(&wal).unwrap(), Some(3));
        assert_eq!(*RAN.lock().unwrap(), vec![1, 2]);

        // A format with no step to upgrade it is refused.
        write_format(&wal, 1).unwrap();
        let err = check_with(&wal, 3, &steps[1..]).unwrap_err().to_string();
        assert!(err.contains("cannot upgrade"), "{err}");
    }
}
