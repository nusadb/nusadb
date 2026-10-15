//! Encryption at rest: an encrypted database writes no row value, table name or key in the clear
//! to any file (its directory, its checkpoint archive, a backup), recovers after a restart and a
//! crash with its key, refuses a missing or wrong key with every file unchanged, refuses altered
//! or moved pages and records, keeps working across incremental checkpoints, spills, backups,
//! restores and a standby, survives a master key rotation, and an existing database can be
//! copied into an encrypted one.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness: asserts via unwrap/panic and edits known file offsets"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use nusadb_btree::crypt::{self, MasterKey, Rewrapped};
use nusadb_btree::{BtreeEngine, OpenOptions, RecoveryTarget};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, PAGE_SIZE, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;
/// Every row carries it; it must never be found on disk.
const MARK: &[u8] = b"ZEBRA-SECRET-ROW";
/// The table's name; it must never be found on disk either.
const TABLE: &str = "zebra_secret_table";

fn master(byte: u8) -> Arc<MasterKey> {
    Arc::new(MasterKey::from_text(&format!("{byte:02x}").repeat(32)).unwrap())
}

fn options(key: &Arc<MasterKey>) -> OpenOptions {
    OpenOptions {
        master_key: Some(Arc::clone(key)),
        ..OpenOptions::default()
    }
}

fn open(wal: &Path, key: &Arc<MasterKey>) -> BtreeEngine {
    BtreeEngine::open_with(wal, options(key)).unwrap()
}

fn create(engine: &BtreeEngine) -> TableId {
    let txn = engine.begin(RC).unwrap();
    let table = engine
        .create_table(
            txn,
            &TableDef {
                schema: "public".to_owned(),
                name: TABLE.to_owned(),
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

fn row(n: u32, len: usize) -> Vec<u8> {
    let mut payload = n.to_be_bytes().to_vec();
    payload.extend_from_slice(MARK);
    payload.resize(len, b'.');
    payload
}

/// Insert rows `from..to`, `per_txn` to a transaction.
fn insert(engine: &BtreeEngine, table: TableId, from: u32, to: u32, len: usize) {
    let mut n = from;
    while n < to {
        let txn = engine.begin(RC).unwrap();
        for m in n..(n + 50).min(to) {
            engine.insert(txn, table, &row(m, len)).unwrap();
        }
        engine.commit(txn).unwrap();
        n += 50;
    }
}

fn numbers(engine: &BtreeEngine) -> Vec<u32> {
    let table = engine.lookup_table(TABLE).unwrap().unwrap().id;
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        assert_eq!(&tuple[4..4 + MARK.len()], MARK);
        out.push(u32::from_be_bytes(tuple[..4].try_into().unwrap()));
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out.sort_unstable();
    out
}

/// Every regular file under `dir`, recursively.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(files(&path));
        } else {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Fail if any file under `dirs` holds a row's mark or the table's name.
fn assert_nothing_readable(dirs: &[&Path]) {
    let mut checked = 0;
    for dir in dirs {
        for file in files(dir) {
            let bytes = std::fs::read(&file).unwrap();
            assert!(
                !contains(&bytes, MARK),
                "{} holds a row in the clear",
                file.display()
            );
            assert!(
                !contains(&bytes, TABLE.as_bytes()),
                "{} holds the table name in the clear",
                file.display()
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// Every file under `dir` with its contents.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    files(dir)
        .into_iter()
        .map(|f| {
            let bytes = std::fs::read(&f).unwrap();
            (f, bytes)
        })
        .collect()
}

#[test]
fn no_row_or_name_is_readable_in_any_file_and_the_key_reads_them_back() {
    let live = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let backups = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let key = master(1);
    let engine = BtreeEngine::open_with(
        &wal,
        OpenOptions {
            archive: Some(archive.path().join("db")),
            ..options(&key)
        },
    )
    .unwrap();
    assert!(engine.is_encrypted());
    let table = create(&engine);
    insert(&engine, table, 0, 400, 300);
    engine.checkpoint().unwrap();
    // Changes after the checkpoint live in the log only.
    insert(&engine, table, 400, 500, 300);
    engine.checkpoint().unwrap();
    insert(&engine, table, 500, 600, 300);
    let copy = backups.path().join("db").join("btree.wal");
    engine.backup_into(&copy).unwrap();
    drop(engine);

    assert!(
        std::fs::metadata(&wal).unwrap().len() > 0,
        "the log holds rows"
    );
    assert!(
        files(&live.path().join("btree.wal.pages"))
            .iter()
            .any(|f| f.extension().is_some_and(|e| e == "seg"))
    );
    assert!(crypt::envelope_path(&wal).exists());
    assert!(crypt::archive_envelope_path(&archive.path().join("db")).exists());
    assert_eq!(nusadb_btree::format::read_format(&wal).unwrap(), Some(3));
    assert_nothing_readable(&[live.path(), archive.path(), backups.path()]);

    let engine = open(&wal, &key);
    assert_eq!(numbers(&engine), (0..600).collect::<Vec<_>>());
    drop(engine);
    let restored = open(&copy, &key);
    assert_eq!(numbers(&restored), (0..500).collect::<Vec<_>>());
}

#[test]
fn an_unencrypted_database_is_unchanged_without_a_key() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(!engine.is_encrypted());
    let table = create(&engine);
    insert(&engine, table, 0, 100, 100);
    engine.checkpoint().unwrap();
    drop(engine);
    assert!(!crypt::envelope_path(&wal).exists());
    assert_eq!(nusadb_btree::format::read_format(&wal).unwrap(), Some(2));
    let found = files(live.path())
        .iter()
        .any(|f| contains(&std::fs::read(f).unwrap(), MARK));
    assert!(
        found,
        "an unencrypted database keeps its rows readable, as before"
    );
    let engine = BtreeEngine::open(&wal).unwrap();
    assert_eq!(numbers(&engine).len(), 100);
}

#[test]
fn restart_and_crash_recover_every_committed_row() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let key = master(2);
    {
        let engine = open(&wal, &key);
        let table = create(&engine);
        insert(&engine, table, 0, 200, 120);
        engine.checkpoint().unwrap();
        insert(&engine, table, 200, 300, 120);
        // An uncommitted transaction is cut off by the "crash": the engine goes away with it
        // open and nothing is flushed on the way out (the engine has no shutdown path; every
        // commit was fsynced when it returned). A real `kill -9` is exercised on the server.
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, table, &row(9999, 120)).unwrap();
        drop(engine);
    }
    // A torn tail: half a record's worth of garbage after the last commit.
    let mut log = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    std::io::Write::write_all(&mut log, &[0x5a; 40]).unwrap();
    drop(log);
    let engine = open(&wal, &key);
    assert_eq!(numbers(&engine), (0..300).collect::<Vec<_>>());
    let table = engine.lookup_table(TABLE).unwrap().unwrap().id;
    insert(&engine, table, 300, 310, 120);
    drop(engine);
    let engine = open(&wal, &key);
    assert_eq!(numbers(&engine), (0..310).collect::<Vec<_>>());
}

#[test]
fn a_missing_or_wrong_key_is_refused_with_every_file_unchanged() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let key = master(3);
    {
        let engine = open(&wal, &key);
        let table = create(&engine);
        insert(&engine, table, 0, 100, 100);
        engine.checkpoint().unwrap();
        insert(&engine, table, 100, 150, 100);
    }
    let before = snapshot(live.path());

    let err = BtreeEngine::open(&wal).unwrap_err().to_string();
    assert!(err.contains("is encrypted"), "{err}");
    assert_eq!(snapshot(live.path()), before);

    let err = BtreeEngine::open_with(&wal, options(&master(4)))
        .unwrap_err()
        .to_string();
    assert!(err.contains("is not the one"), "{err}");
    assert_eq!(snapshot(live.path()), before);

    // Without its envelope the data cannot be opened, with or without a key.
    let envelope = crypt::envelope_path(&wal);
    let saved = std::fs::read(&envelope).unwrap();
    std::fs::remove_file(&envelope).unwrap();
    let without = snapshot(live.path());
    assert!(BtreeEngine::open(&wal).is_err());
    assert!(BtreeEngine::open_with(&wal, options(&key)).is_err());
    assert_eq!(snapshot(live.path()), without);
    std::fs::write(&envelope, saved).unwrap();

    let engine = open(&wal, &key);
    assert_eq!(numbers(&engine), (0..150).collect::<Vec<_>>());
}

#[test]
fn an_unencrypted_database_with_data_is_refused_when_a_key_is_given() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 10, 50);
    }
    let before = snapshot(live.path());
    let err = BtreeEngine::open_with(&wal, options(&master(5)))
        .unwrap_err()
        .to_string();
    assert!(err.contains("is not encrypted"), "{err}");
    assert_eq!(snapshot(live.path()), before);
    assert_eq!(numbers(&BtreeEngine::open(&wal).unwrap()).len(), 10);
}

/// A database with rows in its image and its log, closed.
fn sealed_database(dir: &Path, key: &Arc<MasterKey>) -> PathBuf {
    let wal = dir.join("btree.wal");
    let engine = open(&wal, key);
    let table = create(&engine);
    insert(&engine, table, 0, 200, 200);
    engine.checkpoint().unwrap();
    insert(&engine, table, 200, 250, 200);
    wal
}

fn only_segment(wal: &Path) -> PathBuf {
    let mut segments: Vec<PathBuf> = files(&wal.with_extension("wal.pages"))
        .into_iter()
        .filter(|f| f.extension().is_some_and(|e| e == "seg"))
        .collect();
    assert_eq!(segments.len(), 1);
    segments.remove(0)
}

/// Read every row, returning the first error.
fn try_read_all(wal: &Path, key: &Arc<MasterKey>) -> Result<Vec<u32>, String> {
    let engine = BtreeEngine::open_with(wal, options(key)).map_err(|e| e.to_string())?;
    let table = engine.lookup_table(TABLE).unwrap().unwrap().id;
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().map_err(|e| e.to_string())? {
        out.push(u32::from_be_bytes(tuple[..4].try_into().unwrap()));
    }
    Ok(out)
}

#[test]
fn an_altered_page_is_refused() {
    let live = tempfile::tempdir().unwrap();
    let key = master(6);
    let wal = sealed_database(live.path(), &key);
    let segment = only_segment(&wal);
    let mut bytes = std::fs::read(&segment).unwrap();
    for slot in 0..bytes.len() / crypt::SEALED_PAGE_SIZE {
        bytes[slot * crypt::SEALED_PAGE_SIZE + 100] ^= 0x01;
    }
    std::fs::write(&segment, &bytes).unwrap();
    let err = try_read_all(&wal, &key).unwrap_err();
    assert!(err.contains("fails authentication"), "{err}");
}

#[test]
fn pages_swapped_between_slots_are_refused() {
    // Two valid sealed pages exchanged: each is intact, but bound to its own id and slot.
    let live = tempfile::tempdir().unwrap();
    let key = master(7);
    let wal = sealed_database(live.path(), &key);
    let segment = only_segment(&wal);
    let mut bytes = std::fs::read(&segment).unwrap();
    let n = crypt::SEALED_PAGE_SIZE;
    assert!(bytes.len() >= 2 * n);
    let (first, rest) = bytes.split_at_mut(n);
    first.swap_with_slice(&mut rest[..n]);
    std::fs::write(&segment, &bytes).unwrap();
    let err = try_read_all(&wal, &key).unwrap_err();
    assert!(err.contains("fails authentication"), "{err}");
}

/// The byte offsets of the log's frames.
fn frames(log: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 16 <= log.len() {
        out.push(off);
        let len = u32::from_le_bytes(log[off + 8..off + 12].try_into().unwrap()) as usize;
        off += 16 + len;
    }
    out
}

/// Flip a body byte of the frame at `off` and fix its CRC, as someone without the key could.
fn forge(log: &mut [u8], off: usize) {
    let len = u32::from_le_bytes(log[off + 8..off + 12].try_into().unwrap()) as usize;
    log[off + 16 + 20] ^= 0x01;
    let mut h = crc32fast::Hasher::new();
    h.update(&log[off..off + 12]);
    h.update(&log[off + 16..off + 16 + len]);
    let crc = h.finalize().to_le_bytes();
    log[off + 12..off + 16].copy_from_slice(&crc);
}

#[test]
fn an_altered_log_record_is_refused_even_at_the_tail() {
    let live = tempfile::tempdir().unwrap();
    let key = master(8);
    let wal = sealed_database(live.path(), &key);
    let log = std::fs::read(&wal).unwrap();
    let offsets = frames(&log);
    assert!(offsets.len() > 2);
    for at in [offsets[1], *offsets.last().unwrap()] {
        let mut forged = log.clone();
        forge(&mut forged, at);
        std::fs::write(&wal, &forged).unwrap();
        let err = BtreeEngine::open_with(&wal, options(&key))
            .unwrap_err()
            .to_string();
        assert!(err.contains("fails authentication"), "{err}");
        // Refused, not truncated.
        assert_eq!(std::fs::read(&wal).unwrap(), forged);
    }
    std::fs::write(&wal, &log).unwrap();
    assert_eq!(try_read_all(&wal, &key).unwrap().len(), 250);
}

#[test]
fn an_altered_image_is_refused() {
    let live = tempfile::tempdir().unwrap();
    let key = master(9);
    let wal = sealed_database(live.path(), &key);
    let image = wal.with_extension("wal.ckpt");
    let clean = std::fs::read(&image).unwrap();
    // A byte of the sealed page directory, and a byte near the end (the sealed records).
    for at in [200, clean.len() - 30] {
        let mut bytes = clean.clone();
        bytes[at] ^= 0x01;
        std::fs::write(&image, &bytes).unwrap();
        let err = BtreeEngine::open_with(&wal, options(&key))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid"), "byte {at}: {err}");
    }
    std::fs::write(&image, &clean).unwrap();
    assert_eq!(try_read_all(&wal, &key).unwrap().len(), 250);
}

#[test]
fn incremental_checkpoints_and_spilled_pages_stay_sealed_and_readable() {
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    let key = master(10);
    {
        let engine = open(&wal, &key).with_max_total_resident_bytes(Some(64 * PAGE_SIZE as u64));
        let table = create(&engine);
        insert(&engine, table, 0, 3000, 400);
        assert!(engine.spilled_page_bytes() > 0, "nothing spilled");
        // Every row as written, read back through spilled pages.
        assert_eq!(numbers(&engine), (0..3000).collect::<Vec<_>>());
        engine.checkpoint().unwrap();
        // A few more rows: the next image writes the changed pages to a second segment.
        insert(&engine, table, 3000, 3010, 400);
        engine.checkpoint().unwrap();
        let segments = files(&live.path().join("btree.wal.pages"));
        assert_eq!(segments.len(), 2, "{segments:?}");
        assert!(
            segments
                .iter()
                .any(|s| !s.to_string_lossy().ends_with("-full.seg"))
        );
    }
    assert_nothing_readable(&[live.path()]);
    let engine = open(&wal, &key).with_max_total_resident_bytes(Some(64 * PAGE_SIZE as u64));
    assert_eq!(numbers(&engine), (0..3010).collect::<Vec<_>>());
}

#[test]
fn archive_restore_and_a_standby_work_with_the_key() {
    let primary = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("db");
    let wal = primary.path().join("btree.wal");
    let key = master(11);
    let engine = BtreeEngine::open_with(
        &wal,
        OpenOptions {
            archive: Some(archive.clone()),
            ..options(&key)
        },
    )
    .unwrap();
    let table = create(&engine);
    insert(&engine, table, 0, 100, 150);
    engine.checkpoint().unwrap();

    // A standby seeded from the archive.
    let standby_dir = tempfile::tempdir().unwrap();
    let standby_wal = standby_dir.path().join("btree.wal");
    assert!(
        nusadb_btree::seed_standby(&archive, &standby_wal).is_err(),
        "needs the key"
    );
    nusadb_btree::seed_standby_with_key(&archive, &standby_wal, Some(&key)).unwrap();
    let standby = BtreeEngine::open_with(
        &standby_wal,
        OpenOptions {
            standby: true,
            ..options(&key)
        },
    )
    .unwrap();
    assert_eq!(numbers(&standby), (0..100).collect::<Vec<_>>());

    insert(&engine, table, 100, 200, 150);
    engine.checkpoint().unwrap();
    insert(&engine, table, 200, 250, 150);
    engine.checkpoint().unwrap();
    drop(engine);

    // The standby follows the encrypted segments.
    let archive_key = crypt::archive_key(&archive, Some(&key)).unwrap();
    let position = standby.wal_last_lsn().unwrap().unwrap();
    for (_, path) in nusadb_btree::shipped_segments_after(&archive, position).unwrap() {
        let bytes = std::fs::read(path).unwrap();
        standby
            .apply_shipped_segment_with_key(&bytes, Duration::from_secs(1), archive_key.as_ref())
            .unwrap();
    }
    assert_eq!(numbers(&standby), (0..250).collect::<Vec<_>>());
    drop(standby);
    let standby = BtreeEngine::open_with(
        &standby_wal,
        OpenOptions {
            standby: true,
            ..options(&key)
        },
    )
    .unwrap();
    assert_eq!(numbers(&standby), (0..250).collect::<Vec<_>>());
    drop(standby);

    // A restore from the archive, refused without the key and with another one.
    let restored = tempfile::tempdir().unwrap();
    let out = restored.path().join("btree.wal");
    assert!(
        BtreeEngine::restore_from_archive(&archive, RecoveryTarget::Latest, &out, None).is_err()
    );
    assert!(
        BtreeEngine::restore_from_archive_with_key(
            &archive,
            RecoveryTarget::Latest,
            &out,
            None,
            Some(&master(12)),
        )
        .is_err()
    );
    BtreeEngine::restore_from_archive_with_key(
        &archive,
        RecoveryTarget::Latest,
        &out,
        None,
        Some(&key),
    )
    .unwrap();
    let engine = open(&out, &key);
    assert_eq!(numbers(&engine), (0..250).collect::<Vec<_>>());
    drop(engine);
    assert_nothing_readable(&[
        primary.path(),
        root.path(),
        standby_dir.path(),
        restored.path(),
    ]);
}

#[test]
fn rotating_the_master_key_rewraps_every_envelope() {
    let live = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("db");
    let wal = live.path().join("btree.wal");
    let (old, new) = (master(13), master(14));
    {
        let engine = BtreeEngine::open_with(
            &wal,
            OpenOptions {
                archive: Some(archive.clone()),
                ..options(&old)
            },
        )
        .unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 80, 100);
        engine.checkpoint().unwrap();
    }
    let data = snapshot(live.path());
    for envelope in [
        crypt::envelope_path(&wal),
        crypt::archive_envelope_path(&archive),
    ] {
        assert_eq!(
            crypt::rewrap(&envelope, &old, &new).unwrap(),
            Rewrapped::Done
        );
        // Run again after an interruption: nothing more to do.
        assert_eq!(
            crypt::rewrap(&envelope, &old, &new).unwrap(),
            Rewrapped::Already
        );
    }
    // Only the envelope changed: the data is not rewritten.
    let after = snapshot(live.path());
    let changed: Vec<_> = data
        .iter()
        .filter(|(path, bytes)| after.get(*path) != Some(bytes))
        .map(|(path, _)| path.clone())
        .collect();
    assert_eq!(changed, vec![crypt::envelope_path(&wal)]);
    assert!(BtreeEngine::open_with(&wal, options(&old)).is_err());
    let engine = open(&wal, &new);
    assert_eq!(numbers(&engine), (0..80).collect::<Vec<_>>());
    drop(engine);
    let restored = tempfile::tempdir().unwrap();
    let out = restored.path().join("btree.wal");
    BtreeEngine::restore_from_archive_with_key(
        &archive,
        RecoveryTarget::Latest,
        &out,
        None,
        Some(&new),
    )
    .unwrap();
    assert_eq!(numbers(&open(&out, &new)), (0..80).collect::<Vec<_>>());
}

#[test]
fn an_existing_database_is_copied_into_an_encrypted_one() {
    let plain = tempfile::tempdir().unwrap();
    let source = plain.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&source).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 300, 200);
        engine.checkpoint().unwrap();
        insert(&engine, table, 300, 350, 200);
    }
    let target = tempfile::tempdir().unwrap();
    let out = target.path().join("base").join("db").join("btree.wal");
    let key = master(15);
    BtreeEngine::copy_encrypted(&source, None, &out, &key).unwrap();
    assert_nothing_readable(&[target.path()]);
    let engine = open(&out, &key);
    assert_eq!(numbers(&engine), (0..350).collect::<Vec<_>>());
    let table = engine.lookup_table(TABLE).unwrap().unwrap().id;
    insert(&engine, table, 350, 360, 200);
    engine.checkpoint().unwrap();
    drop(engine);
    assert_eq!(numbers(&open(&out, &key)), (0..360).collect::<Vec<_>>());
    // The source still opens unencrypted.
    assert_eq!(numbers(&BtreeEngine::open(&source).unwrap()).len(), 350);
    // A copy into a directory that holds a database is refused.
    assert!(BtreeEngine::copy_encrypted(&source, None, &out, &key).is_err());

    // Copying an encrypted database under a new key replaces its data key.
    let again = tempfile::tempdir().unwrap();
    let out2 = again.path().join("btree.wal");
    let new = master(16);
    BtreeEngine::copy_encrypted(&out, Some(&key), &out2, &new).unwrap();
    assert_ne!(
        crypt::DataKey::id_at(&crypt::envelope_path(&out)).unwrap(),
        crypt::DataKey::id_at(&crypt::envelope_path(&out2)).unwrap()
    );
    assert_eq!(numbers(&open(&out2, &new)), (0..360).collect::<Vec<_>>());
}

/// What encryption costs on a bulk load, a checkpoint, a cold scan and a restart: run with
/// `cargo test --release -p nusadb-btree --test test_encryption -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check; run it in release"]
fn overhead_of_encryption() {
    const ROWS: u32 = 200_000;
    const LEN: usize = 200;
    let run = |key: Option<Arc<MasterKey>>| {
        let dir = tempfile::tempdir().unwrap();
        let wal = dir.path().join("btree.wal");
        let opts = OpenOptions {
            master_key: key,
            ..OpenOptions::default()
        };
        let engine = BtreeEngine::open_with(&wal, opts.clone()).unwrap();
        let table = create(&engine);
        let start = std::time::Instant::now();
        let mut n = 0;
        while n < ROWS {
            let txn = engine.begin(RC).unwrap();
            for m in n..(n + 1000).min(ROWS) {
                engine.insert(txn, table, &row(m, LEN)).unwrap();
            }
            engine.commit(txn).unwrap();
            n += 1000;
        }
        let insert = start.elapsed();
        let start = std::time::Instant::now();
        engine.checkpoint().unwrap();
        let checkpoint = start.elapsed();
        drop(engine);
        let start = std::time::Instant::now();
        let engine = BtreeEngine::open_with(&wal, opts).unwrap();
        let open = start.elapsed();
        let start = std::time::Instant::now();
        assert_eq!(numbers(&engine).len(), ROWS as usize);
        let scan = start.elapsed();
        (insert, checkpoint, open, scan)
    };
    // One warm-up of each, then the median of three.
    let mut results = Vec::new();
    for key in [None, Some(master(42))] {
        let _ = run(key.clone());
        let mut runs: Vec<_> = (0..3).map(|_| run(key.clone())).collect();
        runs.sort_by_key(|r| r.0);
        results.push((key.is_some(), runs[1]));
    }
    for (encrypted, (insert, checkpoint, open, scan)) in results {
        println!(
            "encrypted={encrypted} rows={ROWS} insert={insert:?} checkpoint={checkpoint:?} \
             reopen={open:?} cold_scan={scan:?}"
        );
    }
}

#[test]
fn a_stray_envelope_beside_unencrypted_data_is_refused_and_changes_nothing() {
    let key = master(17);
    // An envelope from another database, sealed under the same master key.
    let other = tempfile::tempdir().unwrap();
    let other_wal = sealed_database(other.path(), &key);
    let live = tempfile::tempdir().unwrap();
    let wal = live.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&wal).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 120, 100);
        engine.checkpoint().unwrap();
        insert(&engine, table, 120, 130, 100);
    }
    std::fs::copy(crypt::envelope_path(&other_wal), crypt::envelope_path(&wal)).unwrap();
    let before = snapshot(live.path());
    for attempt in [Some(&key), None] {
        let err = attempt
            .map_or_else(
                || BtreeEngine::open(&wal),
                |key| BtreeEngine::open_with(&wal, options(key)),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not belong"), "{err}");
        assert_eq!(snapshot(live.path()), before);
    }
    assert_eq!(nusadb_btree::format::read_format(&wal).unwrap(), Some(2));
    // Without the stray envelope it opens as the unencrypted database it is.
    std::fs::remove_file(crypt::envelope_path(&wal)).unwrap();
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(!engine.is_encrypted());
    assert_eq!(numbers(&engine), (0..130).collect::<Vec<_>>());
}

#[test]
fn an_interrupted_encrypted_copy_never_opens_and_a_rerun_finishes_it() {
    let plain = tempfile::tempdir().unwrap();
    let source = plain.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&source).unwrap();
        let table = create(&engine);
        insert(&engine, table, 0, 200, 150);
    }
    let target = tempfile::tempdir().unwrap();
    let out = target.path().join("btree.wal");
    let key = master(18);
    BtreeEngine::copy_encrypted(&source, None, &out, &key).unwrap();
    let marker = crypt::incomplete_path(&out);
    assert!(!marker.exists());
    // Interrupted after the data key and the segment were written, before the image: the
    // directory must not open as an empty encrypted database.
    std::fs::write(&marker, b"").unwrap();
    std::fs::remove_file(out.with_extension("wal.ckpt")).unwrap();
    let err = BtreeEngine::open_with(&out, options(&key))
        .unwrap_err()
        .to_string();
    assert!(err.contains("never finished"), "{err}");
    // Interrupted after the image, before the marker went: still refused.
    BtreeEngine::copy_encrypted(&source, None, &out, &key).unwrap();
    std::fs::write(&marker, b"").unwrap();
    assert!(BtreeEngine::open_with(&out, options(&key)).is_err());
    // Running the copy again finishes it.
    BtreeEngine::copy_encrypted(&source, None, &out, &key).unwrap();
    assert!(!marker.exists());
    assert_eq!(numbers(&open(&out, &key)), (0..200).collect::<Vec<_>>());
}
