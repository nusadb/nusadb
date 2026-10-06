//! Point-in-time recovery through the checkpoint archive: every checkpoint leaves its log segment
//! and image in the archive, and a database directory can be rebuilt from them as of a log
//! position or a moment, with the transactions after it gone and the ones before it intact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nusadb_btree::{BtreeEngine, RecoveryTarget};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def() -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: "t".to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

fn now_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

fn insert(engine: &BtreeEngine, table: TableId, payload: &[u8]) {
    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, table, payload).unwrap();
    engine.commit(txn).unwrap();
}

fn payloads(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    engine.commit(txn).unwrap();
    out.sort();
    out
}

/// Distinct millisecond stamps between phases, so a moment falls strictly between commits.
fn tick() -> u64 {
    std::thread::sleep(Duration::from_millis(3));
    let at = now_ms();
    std::thread::sleep(Duration::from_millis(3));
    at
}

struct History {
    dir: tempfile::TempDir,
    archive: tempfile::TempDir,
    table: TableId,
    after_phase1: u64,
    after_phase2: u64,
    lsn_after_phase1: u64,
    lsn_after_phase2: u64,
}

/// Three phases of writes with a checkpoint after each of the first two; the third phase stays
/// in the live log only.
fn history() -> History {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let engine = BtreeEngine::open_with_archive(
        dir.path().join("btree.wal"),
        Some(archive.path().to_path_buf()),
    )
    .unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    insert(&engine, table, b"phase1-a");
    insert(&engine, table, b"phase1-b");
    let lsn_after_phase1 = engine.wal_last_lsn().unwrap().unwrap();
    engine.checkpoint().unwrap();
    let after_phase1 = tick();
    insert(&engine, table, b"phase2-a");
    let lsn_after_phase2 = engine.wal_last_lsn().unwrap().unwrap();
    engine.checkpoint().unwrap();
    let after_phase2 = tick();
    insert(&engine, table, b"phase3-a");
    insert(&engine, table, b"phase3-b");
    assert_eq!(payloads(&engine, table).len(), 5);
    drop(engine);
    History {
        dir,
        archive,
        table,
        after_phase1,
        after_phase2,
        lsn_after_phase1,
        lsn_after_phase2,
    }
}

/// Restore from a private copy of the archive: a restore forks the archive it reads, and each
/// call here wants the whole history.
fn restored(h: &History, target: RecoveryTarget, live: bool) -> Vec<Vec<u8>> {
    let archive = tempfile::tempdir().unwrap();
    copy_tree(h.archive.path(), archive.path());
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    let live_log = h.dir.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        archive.path(),
        target,
        &out_wal,
        live.then_some(live_log.as_path()),
    )
    .unwrap();
    // The restored directory is one sealed image: it opens plainly, and stays that state.
    let engine = BtreeEngine::open(&out_wal).unwrap();
    let rows = payloads(&engine, h.table);
    drop(engine);
    let again = BtreeEngine::open(&out_wal).unwrap();
    assert_eq!(payloads(&again, h.table), rows);
    rows
}

#[test]
fn every_checkpoint_leaves_a_segment_and_an_image_in_the_archive() {
    let h = history();
    let mut names: Vec<String> = std::fs::read_dir(h.archive.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    // Two images with their segment lists, two log segments, the pages directory and the
    // archive's format file.
    assert_eq!(names.len(), 8, "{names:?}");
    assert!(names.contains(&"format".to_owned()), "{names:?}");
    let mut in_pages: Vec<String> = std::fs::read_dir(h.archive.path().join("pages"))
        .unwrap()
        .map(|e| {
            e.unwrap()
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    in_pages.sort();
    // Every segment an archived image lists is in the pages directory, and nothing else is.
    let mut listed: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".segments"))
        .flat_map(|n| {
            std::fs::read_to_string(h.archive.path().join(n))
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    listed.sort();
    listed.dedup();
    assert!(!listed.is_empty());
    assert_eq!(listed, in_pages);
    let with_ext = |ext: &str| {
        names
            .iter()
            .filter(|n| {
                std::path::Path::new(n)
                    .extension()
                    .is_some_and(|e| e == ext)
            })
            .count()
    };
    assert_eq!(with_ext("ckpt"), 2);
    assert_eq!(with_ext("log"), 2);
    assert_eq!(with_ext("segments"), 2);
}

#[test]
fn a_restore_to_a_moment_keeps_exactly_the_commits_before_it() {
    let h = history();
    assert_eq!(
        restored(
            &h,
            RecoveryTarget::Time {
                unix_ms: h.after_phase1
            },
            false
        ),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
    assert_eq!(
        restored(
            &h,
            RecoveryTarget::Time {
                unix_ms: h.after_phase2
            },
            false
        ),
        vec![
            b"phase1-a".to_vec(),
            b"phase1-b".to_vec(),
            b"phase2-a".to_vec()
        ]
    );
    // A moment after the last checkpoint needs the live log, and then reaches phase 3.
    let latest = restored(&h, RecoveryTarget::Time { unix_ms: now_ms() }, true);
    assert_eq!(latest.len(), 5);
}

#[test]
fn a_restore_to_a_log_position_and_to_the_newest_archived_point() {
    let h = history();
    assert_eq!(
        restored(&h, RecoveryTarget::Lsn(h.lsn_after_phase2), false),
        vec![
            b"phase1-a".to_vec(),
            b"phase1-b".to_vec(),
            b"phase2-a".to_vec()
        ]
    );
    assert_eq!(restored(&h, RecoveryTarget::Latest, false).len(), 3);
    assert_eq!(restored(&h, RecoveryTarget::Latest, true).len(), 5);
}

#[test]
fn a_target_before_every_archived_image_is_refused() {
    let h = history();
    let out = tempfile::tempdir().unwrap();
    let err = BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(0),
        &out.path().join("btree.wal"),
        None,
    )
    .expect_err("no image at or before position 0");
    assert!(err.to_string().contains("no archived image"), "{err}");
}

#[test]
fn a_bounded_open_of_the_live_directory_seals_the_target_state() {
    let h = history();
    // Opening the live directory itself up to a moment cuts phase 3 and seals what is left.
    let wal = h.dir.path().join("btree.wal");
    let engine = BtreeEngine::open_until(
        &wal,
        RecoveryTarget::Time {
            unix_ms: h.after_phase2,
        },
    )
    .unwrap();
    assert_eq!(payloads(&engine, h.table).len(), 3);
    drop(engine);
    assert_eq!(
        payloads(&BtreeEngine::open(&wal).unwrap(), h.table).len(),
        3
    );
}

/// Bytes that do not compress, so a log grows by what is written.
fn incompressible(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u8::try_from(state & 0xff).unwrap()
        })
        .collect()
}

fn is_log(name: &str) -> bool {
    std::path::Path::new(name)
        .extension()
        .is_some_and(|e| e == "log")
}

fn archive_names(archive: &tempfile::TempDir) -> Vec<String> {
    // The images and log segments; the pages directory, the images' segment lists and the
    // archive's format file are left out.
    let mut names: Vec<String> = std::fs::read_dir(archive.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n != "pages" && n != "format" && !n.ends_with(".segments"))
        .collect();
    names.sort();
    names
}

/// A restart whose recovery folds a large log at open archives that segment too: the chain
/// stays contiguous and a restore into the span the restart folded sees every row.
#[test]
fn a_restart_that_checkpoints_at_open_keeps_the_archive_contiguous() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let archive_dir = Some(archive.path().to_path_buf());
    let table = {
        let engine = BtreeEngine::open_with_archive(&wal, archive_dir.clone()).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def()).unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
        // Past the open-time auto-checkpoint threshold (8 MiB) in one committed run.
        let txn = engine.begin(RC).unwrap();
        for i in 0..1200_u64 {
            engine
                .insert(txn, table, &incompressible(i + 1, 8000))
                .unwrap();
        }
        engine.commit(txn).unwrap();
        table
    };
    let mid = tick();
    // The restart folds the 9 MiB log at open; that checkpoint must archive it.
    let engine = BtreeEngine::open_with_archive(&wal, archive_dir).unwrap();
    insert(&engine, table, b"late");
    engine.checkpoint().unwrap();
    drop(engine);

    let out = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        archive.path(),
        RecoveryTarget::Time { unix_ms: mid },
        &out.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let restored = BtreeEngine::open(out.path().join("btree.wal")).unwrap();
    assert_eq!(payloads(&restored, table).len(), 1200);
}

/// A crash between archiving a segment and truncating the log leaves the next checkpoint
/// archiving an overlapping segment; a restore replays each record once.
#[test]
fn overlapping_segments_after_a_crash_between_archive_and_truncate_replay_once() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let archive_dir = Some(archive.path().to_path_buf());
    let engine = BtreeEngine::open_with_archive(&wal, archive_dir.clone()).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    insert(&engine, table, b"one");
    insert(&engine, table, b"two");
    engine.checkpoint().unwrap();
    drop(engine);
    // Simulate the crash: put the archived segment's bytes back into the live log, which is the
    // state a crash after archiving but before truncation leaves behind.
    let segment = archive
        .path()
        .join(archive_names(&archive).iter().find(|n| is_log(n)).unwrap());
    std::fs::copy(&segment, &wal).unwrap();

    let engine = BtreeEngine::open_with_archive(&wal, archive_dir).unwrap();
    assert_eq!(
        payloads(&engine, table).len(),
        2,
        "a plain open replays each record once"
    );
    insert(&engine, table, b"three");
    engine.checkpoint().unwrap(); // archives a segment overlapping the previous one
    let at = tick();
    drop(engine);

    let out = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        archive.path(),
        RecoveryTarget::Time { unix_ms: at },
        &out.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let restored = BtreeEngine::open(out.path().join("btree.wal")).unwrap();
    assert_eq!(
        payloads(&restored, table),
        vec![b"one".to_vec(), b"three".to_vec(), b"two".to_vec()]
    );
}

/// A checkpoint with nothing new to fold archives nothing twice, and a pruned segment makes a
/// restore across it refuse.
#[test]
fn a_repeated_checkpoint_archives_once_and_a_pruned_segment_is_refused() {
    let h = history();
    let before = archive_names(&h.archive);
    let engine = BtreeEngine::open_with_archive(
        h.dir.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .unwrap();
    engine.checkpoint().unwrap(); // folds phase 3
    engine.checkpoint().unwrap(); // nothing new: same covered position, nothing archived again
    drop(engine);
    let after = archive_names(&h.archive);
    assert_eq!(after.len(), before.len() + 2, "{after:?}");

    // Prune the middle segment (phase 2): a restore into phase 2 needs it and is refused. So is
    // one to a moment in phase 1: whether the pruned commits fell before that moment can no
    // longer be known. A restore to a position the first image already covers needs no segment.
    let mut segments: Vec<String> = after.iter().filter(|n| is_log(n)).cloned().collect();
    segments.sort();
    std::fs::remove_file(h.archive.path().join(&segments[1])).unwrap();
    for target in [
        RecoveryTarget::Lsn(h.lsn_after_phase2 - 1),
        RecoveryTarget::Time {
            unix_ms: h.after_phase1,
        },
    ] {
        let out = tempfile::tempdir().unwrap();
        let err = BtreeEngine::restore_from_archive(
            h.archive.path(),
            target,
            &out.path().join("btree.wal"),
            None,
        )
        .expect_err("the chain has a gap");
        assert!(err.to_string().contains("gap"), "{err}");
        assert!(
            !out.path().join("btree.wal").exists(),
            "a failed restore leaves nothing behind"
        );
    }
    assert_eq!(
        restored(&h, RecoveryTarget::Lsn(h.lsn_after_phase1), false).len(),
        2
    );
}

/// Serving a restored database with the same archive continues one history: the segments the
/// restore cut are superseded, the sealed image is archived, and a later restore lands on the
/// new line rather than resurrecting what was cut.
#[test]
fn serving_after_a_restore_forks_the_archive_cleanly() {
    let h = history();
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time {
            unix_ms: h.after_phase1,
        },
        &out_wal,
        Some(&h.dir.path().join("btree.wal")),
    )
    .unwrap();
    let names = archive_names(&h.archive);
    assert!(
        names.iter().any(|n| n.starts_with("superseded-")),
        "{names:?}"
    );

    // Serve the restored database against the same archive and write a new line of history.
    let engine =
        BtreeEngine::open_with_archive(&out_wal, Some(h.archive.path().to_path_buf())).unwrap();
    assert_eq!(payloads(&engine, h.table).len(), 2);
    insert(&engine, h.table, b"new-line");
    engine.checkpoint().unwrap();
    let after_new = tick();
    drop(engine);

    let again = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time { unix_ms: after_new },
        &again.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let restored = BtreeEngine::open(again.path().join("btree.wal")).unwrap();
    assert_eq!(
        payloads(&restored, h.table),
        vec![
            b"new-line".to_vec(),
            b"phase1-a".to_vec(),
            b"phase1-b".to_vec()
        ]
    );
    // The cut part of the old line is gone from this archive: a restore to a moment in it lands
    // on the new line, which holds only what the fork kept, and never resurrects phase 2.
    let cut = tempfile::tempdir().unwrap();
    let cut_wal = cut.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time {
            unix_ms: h.after_phase2,
        },
        &cut_wal,
        None,
    )
    .unwrap();
    let engine = BtreeEngine::open(&cut_wal).unwrap();
    assert_eq!(
        payloads(&engine, h.table),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
}

/// A stale copy of the live log (taken before the newest checkpoint) re-appended to a restore
/// overlaps the archived segments; the restored database still numbers its new records past
/// everything it has seen, so a second restore after serving replays nothing twice.
#[test]
fn a_stale_live_log_never_makes_the_new_line_reuse_a_position() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let archive_dir = Some(archive.path().to_path_buf());
    let engine = BtreeEngine::open_with_archive(&wal, archive_dir.clone()).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def()).unwrap();
    engine.commit(txn).unwrap();
    insert(&engine, table, b"one");
    engine.checkpoint().unwrap();
    insert(&engine, table, b"two");
    let stale = dir.path().join("stale.wal");
    std::fs::copy(&wal, &stale).unwrap(); // a live-log copy taken before the next checkpoint
    insert(&engine, table, b"three");
    engine.checkpoint().unwrap();
    drop(engine);
    // The newest image is unreadable, so the restore falls back to the first one plus the
    // segments, then the stale copy on top, which overlaps the last segment entirely.
    let newest = archive_names(&archive)
        .into_iter()
        .rfind(|n| !is_log(n))
        .unwrap();
    std::fs::write(archive.path().join(newest), b"garbage").unwrap();

    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        archive.path(),
        RecoveryTarget::Latest,
        &out_wal,
        Some(&stale),
    )
    .unwrap();
    let engine = BtreeEngine::open_with_archive(&out_wal, archive_dir).unwrap();
    assert_eq!(payloads(&engine, table).len(), 3);
    insert(&engine, table, b"four");
    engine.checkpoint().unwrap();
    let at = tick();
    drop(engine);

    let again = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        archive.path(),
        RecoveryTarget::Time { unix_ms: at },
        &again.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let restored = BtreeEngine::open(again.path().join("btree.wal")).unwrap();
    assert_eq!(
        payloads(&restored, table),
        vec![
            b"four".to_vec(),
            b"one".to_vec(),
            b"three".to_vec(),
            b"two".to_vec()
        ]
    );
}

/// An old-line image that escapes the fork (a crash partway through moving files, undone here
/// by moving it back) cannot chain onto the new line: the new line starts past every archived
/// position, so a restore that would need the escaped image's successors is refused.
#[test]
fn an_old_line_image_that_escapes_the_fork_cannot_resurrect_the_cut() {
    let h = history();
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time {
            unix_ms: h.after_phase1,
        },
        &out_wal,
        None,
    )
    .unwrap();
    let superseded = std::fs::read_dir(h.archive.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.is_dir())
        .unwrap();
    for entry in std::fs::read_dir(&superseded).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "ckpt") {
            std::fs::rename(&path, h.archive.path().join(path.file_name().unwrap())).unwrap();
        }
    }
    // Serve the restored database and write on the new line.
    let engine =
        BtreeEngine::open_with_archive(&out_wal, Some(h.archive.path().to_path_buf())).unwrap();
    insert(&engine, h.table, b"new-line");
    engine.checkpoint().unwrap();
    drop(engine);
    // A restore to a moment after phase 2 lands on the sealed image, which outranks the escaped
    // phase-2 image, and continues along the new line: phase 2 stays gone.
    let again = tempfile::tempdir().unwrap();
    let again_wal = again.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time {
            unix_ms: h.after_phase2,
        },
        &again_wal,
        None,
    )
    .unwrap();
    let engine = BtreeEngine::open(&again_wal).unwrap();
    assert_eq!(
        payloads(&engine, h.table),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
}

/// A live log corrupt in its middle is refused, not silently cut at the corruption, and a
/// position the archive never reaches is refused rather than quietly stopped short of.
#[test]
fn a_corrupt_live_log_and_an_unreached_position_are_refused() {
    let h = history();
    let live = h.dir.path().join("btree.wal");
    let mut bytes = std::fs::read(&live).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    bytes[mid + 1] ^= 0xff;
    let corrupt = h.dir.path().join("corrupt.wal");
    std::fs::write(&corrupt, &bytes).unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Latest,
        &out.path().join("btree.wal"),
        Some(&corrupt),
    )
    .expect_err("corruption in the middle of the live log");
    assert!(err.to_string().contains("corrupt"), "{err}");
    assert!(!out.path().join("btree.wal").exists());
    assert!(!out.path().join("btree.wal.restoring").exists());

    let out = tempfile::tempdir().unwrap();
    let err = BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(h.lsn_after_phase2 + 1000),
        &out.path().join("btree.wal"),
        None,
    )
    .expect_err("the archive ends before that position");
    assert!(err.to_string().contains("ends at log position"), "{err}");
}

/// A restore that dies after forking the archive but before publishing its result, run again to
/// the same moment, yields exactly the rows the first attempt would have.
#[test]
fn an_interrupted_restore_run_again_yields_the_same_rows() {
    let h = history();
    let target = RecoveryTarget::Time {
        unix_ms: h.after_phase1,
    };
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(h.archive.path(), target, &out_wal, None).unwrap();
    // The crash: the archive is forked and the sealed image archived, but nothing published.
    std::fs::remove_file(&out_wal).unwrap();
    std::fs::remove_file(out.path().join("btree.wal.ckpt")).unwrap();
    BtreeEngine::restore_from_archive(h.archive.path(), target, &out_wal, None).unwrap();
    let engine = BtreeEngine::open(&out_wal).unwrap();
    assert_eq!(
        payloads(&engine, h.table),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
    drop(engine);

    // A crash in the middle of the moves: the marker of the first fork is still there and one
    // old-line image it moved is back in the archive root. The next restore finishes the fork
    // before reading anything.
    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(h.archive.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n != "pages"))
        .collect();
    dirs.sort();
    let superseded = dirs.first().unwrap().clone();
    let escaped = std::fs::read_dir(&superseded)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "ckpt"))
        .unwrap();
    std::fs::rename(
        &escaped,
        h.archive.path().join(escaped.file_name().unwrap()),
    )
    .unwrap();
    // The first fork, as the archive recorded it.
    let records = std::fs::read_to_string(h.archive.path().join("forks")).unwrap();
    let first_fork = records.lines().next().unwrap().replace(' ', "\n");
    std::fs::write(
        h.archive.path().join("fork.pending"),
        format!(
            "{first_fork}\nsuperseded={}\n",
            superseded.file_name().unwrap().to_str().unwrap()
        ),
    )
    .unwrap();
    let again = tempfile::tempdir().unwrap();
    let again_wal = again.path().join("btree.wal");
    BtreeEngine::restore_from_archive(h.archive.path(), target, &again_wal, None).unwrap();
    assert!(!h.archive.path().join("fork.pending").exists());
    assert!(
        !h.archive.path().join(escaped.file_name().unwrap()).exists(),
        "the escaped image was moved aside by the settled fork"
    );
    let engine = BtreeEngine::open(&again_wal).unwrap();
    assert_eq!(
        payloads(&engine, h.table),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
}

/// A fork marker whose sealed image never reached the archive marks a restore that died before
/// moving anything: it is dropped, and the archive serves its whole history.
#[test]
fn a_fork_that_never_started_is_forgotten() {
    let h = history();
    std::fs::write(
        h.archive.path().join("fork.pending"),
        format!(
            "cut={}\nsealed=999999\nsuperseded=superseded-never\n",
            h.lsn_after_phase1
        ),
    )
    .unwrap();
    assert_eq!(
        restored(
            &h,
            RecoveryTarget::Time {
                unix_ms: h.after_phase2
            },
            false
        )
        .len(),
        3
    );
    // The same settlement runs when a server opens against the archive.
    let engine = BtreeEngine::open_with_archive(
        h.dir.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .unwrap();
    assert!(!h.archive.path().join("fork.pending").exists());
    assert_eq!(payloads(&engine, h.table).len(), 5);
}

/// An image past the end of the chain proves the history went on: a restore whose chain stops
/// short of it is refused instead of quietly holding less.
#[test]
fn a_chain_that_ends_before_a_later_image_is_refused() {
    let h = history();
    let before_fold = tick();
    let engine = BtreeEngine::open_with_archive(
        h.dir.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .unwrap();
    engine.checkpoint().unwrap(); // archives phase 3 as a segment and an image
    drop(engine);
    let mut segments: Vec<String> = archive_names(&h.archive)
        .into_iter()
        .filter(|n| is_log(n))
        .collect();
    segments.sort();
    std::fs::remove_file(h.archive.path().join(segments.last().unwrap())).unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Time {
            unix_ms: before_fold,
        },
        &out.path().join("btree.wal"),
        None,
    )
    .expect_err("the phase-3 segment is gone but its image remains");
    assert!(err.to_string().contains("continues"), "{err}");
}

/// A fresh database numbers its log from the start; opened against an archive that already
/// holds another history, it is refused rather than allowed to checkpoint onto names the old
/// history used.
#[test]
fn a_fresh_database_refuses_an_archive_of_another_history() {
    let h = history();
    let fresh = tempfile::tempdir().unwrap();
    let err = BtreeEngine::open_with_archive(
        fresh.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .expect_err("the archive belongs to the history in `h`");
    assert!(err.to_string().contains("another line"), "{err}");
    // The history's own directory still opens against it, before and after a checkpoint.
    let engine = BtreeEngine::open_with_archive(
        h.dir.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .unwrap();
    engine.checkpoint().unwrap();
    drop(engine);
    BtreeEngine::open_with_archive(
        h.dir.path().join("btree.wal"),
        Some(h.archive.path().to_path_buf()),
    )
    .unwrap();
}

/// A restore to a log position, interrupted after its fork, lands on the sealed image when run
/// again: the fork record says which cut that image holds.
#[test]
fn an_interrupted_restore_by_position_run_again_yields_the_same_rows() {
    let h = history();
    // The position of phase 2's insert, before its commit: the state is phase 1 alone.
    let target = RecoveryTarget::Lsn(h.lsn_after_phase2 - 1);
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(h.archive.path(), target, &out_wal, None).unwrap();
    let engine = BtreeEngine::open(&out_wal).unwrap();
    let first = payloads(&engine, h.table);
    assert_eq!(first, vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]);
    drop(engine);
    std::fs::remove_file(&out_wal).unwrap();
    std::fs::remove_file(out.path().join("btree.wal.ckpt")).unwrap();
    BtreeEngine::restore_from_archive(h.archive.path(), target, &out_wal, None).unwrap();
    let engine = BtreeEngine::open(&out_wal).unwrap();
    assert_eq!(payloads(&engine, h.table), first);
}

/// A position past the sealed image is the new line's own history: the request stands, through
/// an archived segment and through the live log alike.
#[test]
fn a_position_past_the_sealed_image_restores_the_new_line() {
    let h = history();
    let out = tempfile::tempdir().unwrap();
    let out_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(h.lsn_after_phase1),
        &out_wal,
        None,
    )
    .unwrap();
    let engine =
        BtreeEngine::open_with_archive(&out_wal, Some(h.archive.path().to_path_buf())).unwrap();
    insert(&engine, h.table, b"new-a");
    let after_new_a = engine.wal_last_lsn().unwrap().unwrap();
    insert(&engine, h.table, b"new-b");
    engine.checkpoint().unwrap();
    drop(engine);
    let expected = vec![
        b"new-a".to_vec(),
        b"phase1-a".to_vec(),
        b"phase1-b".to_vec(),
    ];

    // Through the archived segment the checkpoint left.
    let via_segment = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(after_new_a),
        &via_segment.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let restored = BtreeEngine::open(via_segment.path().join("btree.wal")).unwrap();
    assert_eq!(payloads(&restored, h.table), expected);
    drop(restored);

    // Through a live log copied before that checkpoint, against an archive that has not seen
    // the checkpoint: the fork above moved the newer files aside, so copy the archive from
    // before the first restore of this block.
    let fresh = history();
    let out2 = tempfile::tempdir().unwrap();
    let out2_wal = out2.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        fresh.archive.path(),
        RecoveryTarget::Lsn(fresh.lsn_after_phase1),
        &out2_wal,
        None,
    )
    .unwrap();
    let engine =
        BtreeEngine::open_with_archive(&out2_wal, Some(fresh.archive.path().to_path_buf()))
            .unwrap();
    insert(&engine, fresh.table, b"new-a");
    let after_new_a = engine.wal_last_lsn().unwrap().unwrap();
    insert(&engine, fresh.table, b"new-b");
    drop(engine);
    let via_live = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        fresh.archive.path(),
        RecoveryTarget::Lsn(after_new_a),
        &via_live.path().join("btree.wal"),
        Some(&out2_wal),
    )
    .unwrap();
    let restored = BtreeEngine::open(via_live.path().join("btree.wal")).unwrap();
    assert_eq!(payloads(&restored, fresh.table), expected);
}

/// A fork record is written only once its sealed image is in the archive: a restore that died
/// with the marker down but the image missing leaves no record, so a later restore whose sealed
/// image takes the same name is vouched for by its own cut alone.
#[test]
fn a_fork_that_never_sealed_leaves_no_record_behind() {
    let h = history();
    // The state a crash between the marker and the image leaves: the marker names a sealed
    // image that does not exist, and no record.
    std::fs::write(
        h.archive.path().join("fork.pending"),
        format!(
            "cut={}\nsealed=999\nsuperseded=superseded-never\n",
            h.lsn_after_phase1
        ),
    )
    .unwrap();
    let out = tempfile::tempdir().unwrap();
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(h.lsn_after_phase2),
        &out.path().join("btree.wal"),
        None,
    )
    .unwrap();
    let records = std::fs::read_to_string(h.archive.path().join("forks")).unwrap();
    assert_eq!(records.lines().count(), 1, "{records}");
    assert!(
        records.contains(&format!("cut={} ", h.lsn_after_phase2)),
        "{records}"
    );
    // The image sealing phase 2 stands in only for positions from its own cut on: a position
    // inside phase 2, before its commit, still yields phase 1 alone.
    let again = tempfile::tempdir().unwrap();
    let again_wal = again.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Lsn(h.lsn_after_phase2 - 1),
        &again_wal,
        None,
    )
    .unwrap();
    let engine = BtreeEngine::open(&again_wal).unwrap();
    assert_eq!(
        payloads(&engine, h.table),
        vec![b"phase1-a".to_vec(), b"phase1-b".to_vec()]
    );
    // Records that contradict each other about one image are refused outright.
    std::fs::write(
        h.archive.path().join("forks"),
        "cut=1 sealed=50\ncut=2 sealed=50\n",
    )
    .unwrap();
    let err = BtreeEngine::restore_from_archive(
        h.archive.path(),
        RecoveryTarget::Latest,
        &tempfile::tempdir().unwrap().path().join("btree.wal"),
        None,
    )
    .expect_err("contradicting fork records");
    assert!(err.to_string().contains("more than one cut"), "{err}");
}

/// Copy the directory tree `from` into `to`: files and the archive's pages subdirectory.
fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}
