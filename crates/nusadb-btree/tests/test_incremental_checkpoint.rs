//! A checkpoint writes only the pages that changed since the image before it, into a new
//! immutable segment, and names the older segments for every other page. Segments stay bounded:
//! once the segments still named would hold more than twice the live pages, every page is
//! written afresh. Segments of the image a checkpoint replaces stay one checkpoint longer, so a
//! copy of that image still finds them; anything older is removed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic and reads image bytes by offset"
)]

use std::path::{Path, PathBuf};

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, PAGE_SIZE, StorageEngine, TableId, Tid};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

fn table_def(name: &str) -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: name.to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

fn row(i: u64, tag: u8) -> Vec<u8> {
    let mut v = i.to_le_bytes().to_vec();
    v.extend(std::iter::repeat_n(tag, 200));
    v
}

fn rows(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out.sort();
    out
}

fn tids(engine: &BtreeEngine, table: TableId) -> Vec<Tid> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((tid, _)) = scan.try_next().unwrap() {
        out.push(tid);
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out
}

/// The segment files in the pages directory beside `wal`, with their sizes in pages.
fn segments(wal: &Path) -> Vec<(PathBuf, u64)> {
    let dir = PathBuf::from(format!("{}.pages", wal.display()));
    let mut out: Vec<(PathBuf, u64)> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "seg"))
                .map(|p| {
                    let pages = std::fs::metadata(&p).unwrap().len() / PAGE_SIZE as u64;
                    (p, pages)
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// A table of `n` rows, committed and checkpointed.
fn loaded(wal: &Path, n: u64) -> (BtreeEngine, TableId) {
    let engine = BtreeEngine::open(wal).unwrap();
    let txn = engine.begin(RC).unwrap();
    let table = engine.create_table(txn, &table_def("t")).unwrap();
    for i in 0..n {
        engine.insert(txn, table, &row(i, 0)).unwrap();
    }
    engine.commit(txn).unwrap();
    engine.checkpoint().unwrap();
    (engine, table)
}

/// After a checkpoint of a large table, changing one row and checkpointing again writes a
/// segment of a handful of pages, not the table; both images' rows read back after a reopen.
#[test]
fn a_small_change_writes_a_small_segment() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (engine, table) = loaded(&wal, 5000);
    let first = segments(&wal);
    assert_eq!(first.len(), 1, "{first:?}");
    let full_pages = first[0].1;
    assert!(full_pages > 100, "the table spans {full_pages} pages");
    let tid = tids(&engine, table)[2500];
    let txn = engine.begin(RC).unwrap();
    engine.update(txn, table, tid, &row(2500, 9)).unwrap();
    engine.commit(txn).unwrap();
    engine.checkpoint().unwrap();
    let second = segments(&wal);
    assert_eq!(second.len(), 2, "{second:?}");
    let new = second.iter().find(|s| !first.contains(s)).unwrap();
    assert!(
        new.1 <= 8,
        "one changed row wrote {} pages (the table has {full_pages})",
        new.1
    );
    let mut want: Vec<Vec<u8>> = (0..5000)
        .map(|i| row(i, if i == 2500 { 9 } else { 0 }))
        .collect();
    want.sort();
    assert_eq!(rows(&engine, table), want);
    drop(engine);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&reopened, table), want);
}

/// The segment names an image lists, read from its header (44 bytes, segment count at 32..40)
/// and the table of length-prefixed names that follows.
fn image_segments(image: &Path) -> Vec<String> {
    let bytes = std::fs::read(image).unwrap();
    let count = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let mut at = 44;
    let mut names = Vec::new();
    for _ in 0..count {
        let len = usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        names.push(String::from_utf8(bytes[at + 2..at + 2 + len].to_vec()).unwrap());
        at += 2 + len;
    }
    names.sort();
    names
}

fn segment_names(wal: &Path) -> Vec<String> {
    let mut names: Vec<String> = segments(wal)
        .into_iter()
        .map(|(p, _)| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// After each checkpoint the pages directory holds exactly the segments the new image and the
/// one it replaced read from: the replaced image's stay one checkpoint longer, anything older
/// is removed. Segments no image names are removed at open.
#[test]
fn old_segments_live_one_checkpoint_longer_and_strays_go_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let image = dir.path().join("btree.wal.ckpt");
    let (engine, table) = loaded(&wal, 3000);
    let all = tids(&engine, table);
    let mut dropped_any = false;
    for round in 1..=6_u8 {
        let replaced = image_segments(&image);
        let txn = engine.begin(RC).unwrap();
        // A different third of the rows each round, so some segments stop being named.
        for (i, tid) in all.iter().enumerate() {
            if i % 3 == usize::from(round % 3) {
                engine
                    .update(txn, table, *tid, &row(i as u64, round))
                    .unwrap();
            }
        }
        engine.commit(txn).unwrap();
        engine.purge().unwrap();
        let on_disk_before = segment_names(&wal);
        engine.checkpoint().unwrap();
        let current = image_segments(&image);
        let mut expected: Vec<String> = current.iter().chain(&replaced).cloned().collect();
        expected.sort();
        expected.dedup();
        assert_eq!(segment_names(&wal), expected, "round {round}");
        dropped_any |= on_disk_before.iter().any(|n| !expected.contains(n));
    }
    assert!(
        dropped_any,
        "some segment stopped being named and was removed"
    );
    drop(engine);
    // A stray segment (a checkpoint that crashed before naming it) is removed at open, as are
    // the replaced image's segments, which only a running engine keeps.
    let stray = PathBuf::from(format!(
        "{}.pages/00000000000000000001-0000000000000001.seg",
        wal.display()
    ));
    std::fs::write(&stray, vec![0u8; PAGE_SIZE]).unwrap();
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert!(
        !stray.exists(),
        "a segment no image names is removed at open"
    );
    assert_eq!(segment_names(&wal), image_segments(&image));
    assert_eq!(rows(&reopened, table).len(), 3000);
}

/// The pages the segments `names` hold, read from the pages directory beside `wal`.
fn named_pages(wal: &Path, names: &[String]) -> u64 {
    segments(wal)
        .iter()
        .filter(|(p, _)| names.contains(&p.file_stem().unwrap().to_string_lossy().into_owned()))
        .map(|(_, pages)| pages)
        .sum()
}

/// Repeated partial rewrites never let what an image names grow past about twice the live
/// pages: once it would, a checkpoint writes every page afresh into a `-full` segment. (A
/// rewrite of one quarter at a time settles below that bound, about two segments' worth of
/// the table, and never needs one.)
#[test]
fn segments_stay_bounded_under_churn() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let image = dir.path().join("btree.wal.ckpt");
    // Large enough that the dead weight this pattern leaves (about half the live pages)
    // outgrows the fixed slack, so the size rule is what triggers a rewrite.
    let (engine, table) = loaded(&wal, 100_000);
    let all = tids(&engine, table);
    let live_pages = segments(&wal)[0].1;
    assert!(live_pages > 2 * 1024, "{live_pages}");
    let first_full = segment_names(&wal);
    let mut rewrote = false;
    for round in 1..=6_u8 {
        // Three contiguous quarters of the table each round, a different three each time: the
        // pages they lie on are written again, and their older copies are dead weight in the
        // segments still named for the rest.
        let quarter = all.len() / 4;
        let txn = engine.begin(RC).unwrap();
        for (i, tid) in all.iter().enumerate() {
            if i / quarter != usize::from(round % 4) {
                engine
                    .update(txn, table, *tid, &row(i as u64, round))
                    .unwrap();
            }
        }
        engine.commit(txn).unwrap();
        engine.purge().unwrap();
        engine.checkpoint().unwrap();
        let named = image_segments(&image);
        rewrote |=
            named.len() == 1 && named[0].ends_with("-full") && !first_full.contains(&named[0]);
        let pages = named_pages(&wal, &named);
        assert!(
            pages <= 2 * live_pages + 1024 + live_pages / 8,
            "round {round}: the image names {pages} segment pages for {live_pages} live pages"
        );
    }
    assert!(
        rewrote,
        "the size rule wrote every page afresh at least once"
    );
    drop(engine);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&reopened, table).len(), 100_000);
}

/// Each checkpoint that changes a little adds a segment; past the most one image may name,
/// the next checkpoint writes every page afresh, so open files stay bounded.
#[test]
fn the_number_of_segments_an_image_names_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let image = dir.path().join("btree.wal.ckpt");
    let (engine, table) = loaded(&wal, 3000);
    let all = tids(&engine, table);
    let mut most = 0;
    let mut rewrote = false;
    for round in 0..40_usize {
        let txn = engine.begin(RC).unwrap();
        let i = (round * 71) % all.len();
        engine
            .update(txn, table, all[i], &row(i as u64, 7))
            .unwrap();
        engine.commit(txn).unwrap();
        engine.checkpoint().unwrap();
        let named = image_segments(&image);
        most = most.max(named.len());
        rewrote |= round > 0 && named.len() == 1 && named[0].ends_with("-full");
    }
    assert!(most <= 32, "an image named {most} segments");
    assert!(rewrote, "the segment count rule wrote every page afresh");
    drop(engine);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&reopened, table).len(), 3000);
}

/// An image whose segment is missing is refused at open, naming the segment, instead of
/// serving a database with pages missing.
#[test]
fn a_missing_segment_is_refused_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    let (engine, _) = loaded(&wal, 1000);
    drop(engine);
    let (segment, _) = segments(&wal).pop().unwrap();
    std::fs::remove_file(&segment).unwrap();
    let err = BtreeEngine::open(&wal).unwrap_err().to_string();
    assert!(
        err.contains("missing") && err.contains(".seg"),
        "the refusal names the missing segment: {err}"
    );
}

/// An image of the earlier single-file format opens, and the next checkpoint writes the
/// segmented format with every page in one segment.
#[test]
fn a_single_file_image_opens_and_converts_at_the_next_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/single_file_image_v2.ckpt"),
        dir.path().join("btree.wal.ckpt"),
    )
    .unwrap();
    let want: Vec<Vec<u8>> = (0..300_u32)
        .map(|i| format!("row-{i:04}").into_bytes())
        .collect();
    let engine = BtreeEngine::open(&wal).unwrap();
    let table = engine.lookup_table("t").unwrap().unwrap().id;
    assert_eq!(rows(&engine, table), want);
    assert!(segments(&wal).is_empty());
    engine.checkpoint().unwrap();
    let converted = segments(&wal);
    assert_eq!(converted.len(), 1, "{converted:?}");
    assert!(
        converted[0].0.to_string_lossy().ends_with("-full.seg"),
        "{converted:?}"
    );
    drop(engine);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(rows(&reopened, table), want);
}
