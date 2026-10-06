//! A database directory records its data format in `<wal>.format`. A new database records the
//! format this release writes; one written before the format was recorded is format 1 and is
//! stamped when opened; one in a newer format, or with a format file that is not one, is refused
//! before anything else of it is read, with its files untouched: an older release must never take
//! a log record it does not know for a torn tail and cut it off.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::path::{Path, PathBuf};

use nusadb_btree::BtreeEngine;
use nusadb_btree::format::{FORMAT_VERSION, read_format};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine};

fn format_file(wal: &Path) -> PathBuf {
    let mut name = wal.as_os_str().to_owned();
    name.push(".format");
    PathBuf::from(name)
}

/// Create the database at `wal` with one committed row.
fn with_a_row(wal: &Path) {
    let engine = BtreeEngine::open(wal).unwrap();
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
    engine.insert(txn, table, b"row").unwrap();
    engine.commit(txn).unwrap();
}

/// The rows of table `t`.
fn rows(engine: &BtreeEngine) -> Vec<Vec<u8>> {
    let txn = engine.begin(IsolationLevel::ReadCommitted).unwrap();
    let table = engine.lookup_table("t").unwrap().unwrap().id;
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out
}

fn refused(wal: &Path) -> String {
    match BtreeEngine::open(wal) {
        Ok(_) => panic!("the open must be refused"),
        Err(e) => e.to_string(),
    }
}

/// Every file of the database directory with its bytes, the lock file aside.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        for entry in std::fs::read_dir(&at).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_none_or(|e| e != "lock") {
                let bytes = std::fs::read(&path).unwrap();
                files.push((path, bytes));
            }
        }
    }
    files.sort();
    files
}

#[test]
fn a_new_database_records_the_format_and_the_release() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    assert_eq!(read_format(&wal).unwrap(), None);
    with_a_row(&wal);
    let text = std::fs::read_to_string(format_file(&wal)).unwrap();
    assert_eq!(
        text,
        format!(
            "nusadb data format {FORMAT_VERSION}\nwritten by nusadb {}\n",
            env!("CARGO_PKG_VERSION")
        )
    );
    assert_eq!(read_format(&wal).unwrap(), Some(FORMAT_VERSION));
}

#[test]
fn a_database_from_before_the_format_was_recorded_is_format_1_and_is_stamped() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        engine.checkpoint().unwrap();
    }
    std::fs::remove_file(format_file(&wal)).unwrap();
    assert_eq!(read_format(&wal).unwrap(), Some(1));
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&engine), vec![b"row".to_vec()]);
    assert_eq!(read_format(&wal).unwrap(), Some(FORMAT_VERSION));
}

#[test]
fn a_newer_format_is_refused_with_every_file_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    let newer = FORMAT_VERSION + 1;
    std::fs::write(
        format_file(&wal),
        format!("nusadb data format {newer}\nwritten by nusadb 9.0.0\n"),
    )
    .unwrap();
    // A record kind this release does not know at the end of the log: read as this release
    // reads a log, it would be a torn tail and be cut off.
    let mut log = std::fs::read(&wal).unwrap();
    log.extend_from_slice(&[0x5A; 64]);
    std::fs::write(&wal, &log).unwrap();
    let before = snapshot(dir.path());

    let err = refused(&wal);
    assert!(err.contains(&format!("data format {newer}")), "{err}");
    assert!(err.contains("written by nusadb 9.0.0"), "{err}");
    assert!(err.contains("untouched"), "{err}");
    assert_eq!(
        snapshot(dir.path()),
        before,
        "a refused open changed the directory"
    );
    for open in [
        BtreeEngine::open_standby(&wal),
        BtreeEngine::open_with_archive(&wal, None),
    ] {
        assert!(open.is_err());
    }
    assert_eq!(snapshot(dir.path()), before);
}

#[test]
fn a_format_file_that_is_not_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    std::fs::write(format_file(&wal), "garbage\n").unwrap();
    let before = snapshot(dir.path());
    let err = refused(&wal);
    assert!(err.contains("is not a data format file"), "{err}");
    assert_eq!(snapshot(dir.path()), before);
}

#[test]
fn a_checkpoint_image_in_a_newer_version_is_refused_as_newer_not_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    with_a_row(&wal);
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        engine.checkpoint().unwrap();
    }
    let mut image_path = wal.as_os_str().to_owned();
    image_path.push(".ckpt");
    let mut image = std::fs::read(&image_path).unwrap();
    image[4..8].copy_from_slice(&4_u32.to_le_bytes());
    std::fs::write(&image_path, &image).unwrap();
    let err = refused(&wal);
    assert!(err.contains("image version 4"), "{err}");
    assert!(err.contains("newer release"), "{err}");

    image[4..8].copy_from_slice(&0_u32.to_le_bytes());
    std::fs::write(&image_path, &image).unwrap();
    let err = refused(&wal);
    assert!(err.contains("unsupported format version"), "{err}");
}

#[test]
fn an_archive_records_its_format_and_one_in_a_newer_format_is_refused() {
    let live = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    {
        let engine =
            BtreeEngine::open_with_archive(&wal, Some(archive.path().to_path_buf())).unwrap();
        drop(engine);
    }
    with_a_row(&wal);
    {
        let engine =
            BtreeEngine::open_with_archive(&wal, Some(archive.path().to_path_buf())).unwrap();
        engine.checkpoint().unwrap();
    }
    let stamp = archive.path().join("format");
    assert_eq!(
        std::fs::read_to_string(&stamp).unwrap(),
        format!(
            "nusadb data format {FORMAT_VERSION}\nwritten by nusadb {}\n",
            env!("CARGO_PKG_VERSION")
        )
    );

    // A restore and a standby seed publish a database that records its format, and leave no
    // scratch files behind.
    let restored = tempfile::tempdir().unwrap();
    let restored_wal = restored.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        archive.path(),
        nusadb_btree::RecoveryTarget::Latest,
        &restored_wal,
        None,
    )
    .unwrap();
    let seeded = tempfile::tempdir().unwrap();
    let seeded_wal = seeded.path().join("btree.wal");
    nusadb_btree::seed_standby(archive.path(), &seeded_wal).unwrap();
    for (dir, wal) in [(&restored, &restored_wal), (&seeded, &seeded_wal)] {
        assert_eq!(read_format(wal).unwrap(), Some(FORMAT_VERSION));
        assert!(format_file(wal).exists());
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("restoring") || n.contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    std::fs::write(
        &stamp,
        format!(
            "nusadb data format {}\nwritten by nusadb 9.0.0\n",
            FORMAT_VERSION + 1
        ),
    )
    .unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = BtreeEngine::restore_from_archive(
        archive.path(),
        nusadb_btree::RecoveryTarget::Latest,
        &out.path().join("btree.wal"),
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("written by nusadb 9.0.0"), "{err}");
    let err = nusadb_btree::seed_standby(archive.path(), &out.path().join("standby.wal"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("data format"), "{err}");
    let err = nusadb_btree::shipped_segments_after(archive.path(), 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("data format"), "{err}");
    let err = match BtreeEngine::open_with_archive(&wal, Some(archive.path().to_path_buf())) {
        Ok(_) => panic!("an engine must not write into a newer archive"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("data format"), "{err}");
}
