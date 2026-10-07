//! `backup_into` copies a database as of its newest checkpoint image (the image, the page
//! segments it reads from, and the format file) while the database stays open and busy, and the
//! copy opens as exactly a committed state. `prune_archive` removes from a checkpoint archive
//! what a restore inside its window does not need, and what stays still restores.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::duration_suboptimal_units,
    reason = "integration test harness: asserts via unwrap/panic, reads fixed row layouts and \
              archive names it wrote itself"
)]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use nusadb_btree::{BtreeEngine, RecoveryTarget, prune_archive};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn create(engine: &BtreeEngine) -> TableId {
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
    table
}

/// Insert rows `from..to`, one transaction each; the payload is the number, padded so the rows
/// fill several pages.
fn insert(engine: &BtreeEngine, table: TableId, from: u32, to: u32) {
    for n in from..to {
        let txn = engine.begin(RC).unwrap();
        let mut payload = n.to_be_bytes().to_vec();
        payload.resize(200, b'.');
        engine.insert(txn, table, &payload).unwrap();
        engine.commit(txn).unwrap();
    }
}

/// The row numbers in table `t`, sorted.
fn numbers(engine: &BtreeEngine) -> Vec<u32> {
    let table = engine.lookup_table("t").unwrap().unwrap().id;
    let txn = engine.begin(RC).unwrap();
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

#[test]
fn a_backup_opens_as_the_state_of_the_image_and_leaves_the_database_alone() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = create(&engine);
    insert(&engine, table, 0, 300);
    engine.checkpoint().unwrap();
    insert(&engine, table, 300, 400);

    let out = tempfile::tempdir().unwrap();
    let copy = out.path().join("shop").join("btree.wal");
    let info = engine.backup_into(&copy).unwrap();
    assert!(info.segments > 0, "{info:?}");
    assert!(info.image_unix_ms.is_some(), "{info:?}");
    // The live database keeps going and keeps every row.
    insert(&engine, table, 400, 410);
    assert_eq!(numbers(&engine), (0..410).collect::<Vec<_>>());
    drop(engine);

    let restored = BtreeEngine::open(&copy).unwrap();
    assert_eq!(numbers(&restored), (0..300).collect::<Vec<_>>());
    assert_eq!(
        nusadb_btree::format::read_format(&copy).unwrap(),
        Some(nusadb_btree::format::FORMAT_VERSION)
    );
}

#[test]
fn a_backup_is_refused_without_an_image_or_into_a_database() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = create(&engine);
    insert(&engine, table, 0, 5);
    let out = tempfile::tempdir().unwrap();
    let copy = out.path().join("btree.wal");
    let err = engine.backup_into(&copy).unwrap_err().to_string();
    assert!(err.contains("no checkpoint image yet"), "{err}");
    assert!(!copy.exists());

    engine.checkpoint().unwrap();
    engine.backup_into(&copy).unwrap();
    let err = engine.backup_into(&copy).unwrap_err().to_string();
    assert!(err.contains("already holds a database"), "{err}");
}

/// Backups taken over and over while another thread commits rows and checkpoints: each copy
/// opens, and holds exactly the rows `0..n` for some `n` (a committed state, never a mix).
#[test]
fn backups_during_writes_and_checkpoints_are_each_a_committed_state() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = Arc::new(BtreeEngine::open(&wal).unwrap());
    let table = create(&engine);
    insert(&engine, table, 0, 50);
    engine.checkpoint().unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut next = 50;
            while !stop.load(Ordering::Relaxed) {
                insert(&engine, table, next, next + 20);
                next += 20;
                // A checkpoint can find a transaction of this thread still active only if one
                // were left open; none is, so it always runs.
                engine.checkpoint().unwrap();
            }
            next
        })
    };
    let out = tempfile::tempdir().unwrap();
    let mut copies = Vec::new();
    for n in 0..25 {
        let copy = out.path().join(format!("b{n}")).join("btree.wal");
        engine.backup_into(&copy).unwrap();
        copies.push(copy);
    }
    stop.store(true, Ordering::Relaxed);
    let written = writer.join().unwrap();
    assert!(written > 100, "the writer made progress: {written}");
    for copy in copies {
        let restored = BtreeEngine::open(&copy).unwrap();
        let rows = numbers(&restored);
        let n = u32::try_from(rows.len()).unwrap();
        assert_eq!(rows, (0..n).collect::<Vec<_>>(), "{}", copy.display());
        assert!(n >= 50);
    }
}

fn age(path: &Path, by: Duration) {
    let file = std::fs::File::options().write(true).open(path).unwrap();
    file.set_modified(SystemTime::now() - by).unwrap();
}

fn names(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

#[test]
fn pruning_keeps_the_newest_image_before_the_window_and_everything_after() {
    let live = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open_with_archive(&wal, Some(archive.path().to_path_buf())).unwrap();
    let table = create(&engine);
    // Four checkpoints, each over new rows: four images, four log segments.
    for round in 0..4 {
        insert(&engine, table, round * 100, round * 100 + 100);
        engine.checkpoint().unwrap();
    }
    insert(&engine, table, 400, 420);
    let images: Vec<String> = names(archive.path())
        .into_iter()
        .filter(|n| n.ends_with(".ckpt"))
        .collect();
    assert_eq!(images.len(), 4, "{images:?}");
    // The first three images were archived two days ago, the last one now.
    for image in &images[..3] {
        age(&archive.path().join(image), Duration::from_secs(2 * 86_400));
    }
    let pages_before = names(&archive.path().join("pages")).len();

    let a_day_ago = SystemTime::now() - Duration::from_secs(86_400);
    let keep_from = u64::try_from(
        a_day_ago
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap();
    let stats = prune_archive(archive.path(), keep_from).unwrap();
    assert_eq!(stats.images, 2, "{stats:?}");
    assert_eq!(stats.logs, 3, "{stats:?}");
    // Checkpoints are incremental: the base image still reads from the earlier images'
    // segments, so they all stay; every segment a remaining image names is there.
    for list in names(archive.path())
        .into_iter()
        .filter(|n| n.ends_with(".segments"))
    {
        for segment in std::fs::read_to_string(archive.path().join(&list))
            .unwrap()
            .lines()
        {
            assert!(
                archive
                    .path()
                    .join("pages")
                    .join(format!("{segment}.seg"))
                    .exists(),
                "{list} names {segment}"
            );
        }
    }
    let left = names(archive.path());
    assert!(left.contains(&images[2]), "the base image stays: {left:?}");
    assert!(left.contains(&images[3]), "{left:?}");
    assert!(
        !left.contains(&images[0]) && !left.contains(&images[1]),
        "{left:?}"
    );
    assert!(names(&archive.path().join("pages")).len() <= pages_before);
    // Pruning again removes nothing more.
    assert_eq!(
        prune_archive(archive.path(), keep_from).unwrap(),
        nusadb_btree::PruneStats::default()
    );
    drop(engine);

    // What stays restores: the latest state, from the base image and the segment after it.
    let out = tempfile::tempdir().unwrap();
    let restored_wal = out.path().join("btree.wal");
    BtreeEngine::restore_from_archive(
        archive.path(),
        RecoveryTarget::Latest,
        &restored_wal,
        Some(&wal),
    )
    .unwrap();
    let restored = BtreeEngine::open(&restored_wal).unwrap();
    assert_eq!(numbers(&restored), (0..420).collect::<Vec<_>>());
}

/// A page segment that only removed images read from is removed; one a kept image reads from
/// stays.
#[test]
fn pruning_removes_page_segments_only_removed_images_read() {
    let archive = tempfile::tempdir().unwrap();
    let pages = archive.path().join("pages");
    std::fs::create_dir_all(&pages).unwrap();
    // Two images archived by hand: the old one reads segments a and b, the new one b and c.
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 10);
        engine.checkpoint().unwrap();
    }
    let image = std::fs::read(live.path().join("btree.wal.ckpt")).unwrap();
    let segment = |n: u8| format!("{:020}-{:016x}", u64::from(n), u64::from(n));
    for (lsn, listed) in [(1_u64, [1_u8, 2]), (2, [2, 3])] {
        std::fs::write(archive.path().join(format!("{lsn:020}.ckpt")), &image).unwrap();
        let list: String = listed.iter().map(|&n| segment(n) + "\n").collect();
        std::fs::write(archive.path().join(format!("{lsn:020}.segments")), list).unwrap();
    }
    for n in 1..=3 {
        std::fs::write(pages.join(format!("{}.seg", segment(n))), b"page").unwrap();
    }
    age(
        &archive.path().join(format!("{:020}.ckpt", 1)),
        Duration::from_secs(3 * 86_400),
    );
    age(
        &archive.path().join(format!("{:020}.ckpt", 2)),
        Duration::from_secs(2 * 86_400),
    );
    let keep_from = u64::try_from(
        (SystemTime::now() - Duration::from_secs(86_400))
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap();
    let stats = prune_archive(archive.path(), keep_from).unwrap();
    assert_eq!((stats.images, stats.page_segments), (1, 1), "{stats:?}");
    assert_eq!(
        names(&pages),
        vec![format!("{}.seg", segment(2)), format!("{}.seg", segment(3))]
    );
}

/// A checkpoint archiving meanwhile has written its image's segment list and linked the
/// segments, but not the image yet: what that list names stays. A list below the base with no
/// image (left by a checkpoint that stopped before archiving its image) is removed.
#[test]
fn pruning_keeps_what_an_image_being_archived_names() {
    let archive = tempfile::tempdir().unwrap();
    let pages = archive.path().join("pages");
    std::fs::create_dir_all(&pages).unwrap();
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 10);
        engine.checkpoint().unwrap();
    }
    let image = std::fs::read(live.path().join("btree.wal.ckpt")).unwrap();
    let segment = |n: u8| format!("{:020}-{:016x}", u64::from(n), u64::from(n));
    let list = |listed: &[u8]| -> String { listed.iter().map(|&n| segment(n) + "\n").collect() };
    // Image 2 is the base (old); lsn 1's list is an orphan below it; lsn 3 is being archived.
    std::fs::write(
        archive.path().join(format!("{:020}.segments", 1)),
        list(&[1]),
    )
    .unwrap();
    std::fs::write(archive.path().join(format!("{:020}.ckpt", 2)), &image).unwrap();
    std::fs::write(
        archive.path().join(format!("{:020}.segments", 2)),
        list(&[2]),
    )
    .unwrap();
    std::fs::write(
        archive.path().join(format!("{:020}.segments", 3)),
        list(&[2, 3]),
    )
    .unwrap();
    for n in 1..=3 {
        std::fs::write(pages.join(format!("{}.seg", segment(n))), b"page").unwrap();
    }
    age(
        &archive.path().join(format!("{:020}.ckpt", 2)),
        Duration::from_secs(2 * 86_400),
    );
    let keep_from = u64::try_from(
        (SystemTime::now() - Duration::from_secs(86_400))
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap();
    let stats = prune_archive(archive.path(), keep_from).unwrap();
    assert_eq!(stats.page_segments, 1, "{stats:?}");
    assert_eq!(
        names(&pages),
        vec![format!("{}.seg", segment(2)), format!("{}.seg", segment(3))],
        "the segment of the image being archived stays"
    );
    let left = names(archive.path());
    assert!(!left.contains(&format!("{:020}.segments", 1)), "{left:?}");
    assert!(left.contains(&format!("{:020}.segments", 3)), "{left:?}");
}

/// A backup's scratch image left by an earlier, interrupted backup may be a hard link to the live
/// image; a new backup into the same place never writes through it.
#[test]
fn a_leftover_scratch_link_to_the_live_image_is_never_written_through() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = create(&engine);
    insert(&engine, table, 0, 40);
    engine.checkpoint().unwrap();
    let image = live.path().join("btree.wal.ckpt");
    let before = std::fs::read(&image).unwrap();

    let out = tempfile::tempdir().unwrap();
    let copy = out.path().join("btree.wal");
    std::fs::hard_link(&image, out.path().join("btree.wal.ckpt.tmp")).unwrap();
    engine.backup_into(&copy).unwrap();
    assert_eq!(
        std::fs::read(&image).unwrap(),
        before,
        "the live image changed"
    );
    drop(engine);
    assert_eq!(
        numbers(&BtreeEngine::open(&wal).unwrap()),
        (0..40).collect::<Vec<_>>()
    );
    assert_eq!(
        numbers(&BtreeEngine::open(&copy).unwrap()),
        (0..40).collect::<Vec<_>>()
    );
}
