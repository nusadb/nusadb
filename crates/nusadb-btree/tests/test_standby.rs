//! A standby follows a primary through the primary's checkpoint archive: it is seeded from the
//! newest archived image, applies each archived log segment once and in order, serves reads,
//! refuses writes of its own, survives a restart, and becomes a primary when promoted.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::time::Duration;

use nusadb_btree::{
    BtreeEngine, ShipOutcome, newest_archived_image, seed_standby, shipped_segments_after,
};
use nusadb_core::engine::{ColumnDef, TableDef};
use nusadb_core::{ColumnType, Error, IsolationLevel, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;
const WAIT: Duration = Duration::from_secs(2);

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

/// A primary with an archive, a table, and two checkpointed phases of rows.
struct Primary {
    dir: tempfile::TempDir,
    archive: tempfile::TempDir,
    engine: BtreeEngine,
    table: TableId,
}

fn primary() -> Primary {
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
    insert(&engine, table, b"p1-a");
    engine.checkpoint().unwrap();
    insert(&engine, table, b"p2-a");
    insert(&engine, table, b"p2-b");
    engine.checkpoint().unwrap();
    Primary {
        dir,
        archive,
        engine,
        table,
    }
}

/// Seed a standby from the archive and apply every segment archived so far.
fn standby_of(p: &Primary) -> (tempfile::TempDir, BtreeEngine) {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("btree.wal");
    seed_standby(p.archive.path(), &wal).unwrap();
    let engine = BtreeEngine::open_standby(&wal).unwrap();
    catch_up(&engine, p);
    (dir, engine)
}

fn catch_up(standby: &BtreeEngine, p: &Primary) {
    let applied = standby.wal_last_lsn().unwrap().unwrap();
    for (_, path) in shipped_segments_after(p.archive.path(), applied).unwrap() {
        let bytes = std::fs::read(path).unwrap();
        assert!(matches!(
            standby.apply_shipped_segment(&bytes, WAIT).unwrap(),
            ShipOutcome::Applied { .. } | ShipOutcome::NothingNew
        ));
    }
}

#[test]
fn a_standby_is_seeded_from_the_newest_image_and_follows_each_segment() {
    let p = primary();
    // The seed is the newest image: everything checkpointed so far, no segment needed yet.
    let (_dir, standby) = standby_of(&p);
    assert_eq!(
        standby.wal_last_lsn().unwrap().unwrap(),
        newest_archived_image(p.archive.path()).unwrap().unwrap()
    );
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    // The primary writes on and checkpoints twice; the standby applies both segments in order.
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    insert(&p.engine, p.table, b"p4-a");
    insert(&p.engine, p.table, b"p4-b");
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    assert_eq!(
        standby.wal_last_lsn().unwrap().unwrap(),
        p.engine.wal_last_lsn().unwrap().unwrap()
    );
}

#[test]
fn a_standby_serves_reads_but_refuses_its_own_writes_and_sequences() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let txn = standby.begin(RC).unwrap();
    let err = standby
        .insert(txn, p.table, b"local")
        .expect_err("a standby refuses writes at the write itself");
    assert!(matches!(err, Error::ReadOnly(_)), "{err}");
    assert_eq!(err.sqlstate(), "25006");
    standby.rollback(txn).unwrap();
    let txn = standby.begin(RC).unwrap();
    let err = standby
        .create_table(
            txn,
            &TableDef {
                name: "u".to_owned(),
                ..table_def()
            },
        )
        .expect_err("DDL is a write too");
    assert!(matches!(err, Error::ReadOnly(_)), "{err}");
    standby.rollback(txn).unwrap();
    // The refused write left nothing behind, and reads keep working.
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    // A writable engine refuses to apply shipped segments.
    let writable = BtreeEngine::new();
    let err = writable
        .apply_shipped_segment(b"", WAIT)
        .expect_err("not a standby");
    assert!(matches!(err, Error::ReadOnly(_)), "{err}");
}

#[test]
fn an_overlapping_segment_applies_once_and_a_gap_is_refused() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    let applied = standby.wal_last_lsn().unwrap().unwrap();
    let segments = shipped_segments_after(p.archive.path(), applied).unwrap();
    assert_eq!(segments.len(), 1);
    let bytes = std::fs::read(&segments[0].1).unwrap();
    // A crash between archiving and truncation on the primary makes the next segment repeat
    // this one; the repeat applies nothing.
    assert!(matches!(
        standby.apply_shipped_segment(&bytes, WAIT).unwrap(),
        ShipOutcome::Applied { records, .. } if records > 0
    ));
    assert_eq!(
        standby.apply_shipped_segment(&bytes, WAIT).unwrap(),
        ShipOutcome::NothingNew
    );
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    // Two more checkpoints; skipping the first segment is a gap and is refused, and applying
    // them in order then succeeds.
    insert(&p.engine, p.table, b"p4-a");
    p.engine.checkpoint().unwrap();
    insert(&p.engine, p.table, b"p5-a");
    p.engine.checkpoint().unwrap();
    let applied = standby.wal_last_lsn().unwrap().unwrap();
    let segments = shipped_segments_after(p.archive.path(), applied).unwrap();
    assert_eq!(segments.len(), 2);
    let err = standby
        .apply_shipped_segment(&std::fs::read(&segments[1].1).unwrap(), WAIT)
        .expect_err("a gap");
    assert!(err.to_string().contains("missing"), "{err}");
    catch_up(&standby, &p);
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
}

#[test]
fn a_running_transaction_defers_the_apply_without_losing_it() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    let applied = standby.wal_last_lsn().unwrap().unwrap();
    let (_, path) = shipped_segments_after(p.archive.path(), applied)
        .unwrap()
        .remove(0);
    let bytes = std::fs::read(path).unwrap();
    let reader = standby.begin(RC).unwrap();
    let outcome = standby
        .apply_shipped_segment(&bytes, Duration::from_millis(50))
        .unwrap();
    assert!(
        matches!(outcome, ShipOutcome::StillBusy { active: 1, .. }),
        "{outcome:?}"
    );
    assert_eq!(standby.wal_last_lsn().unwrap().unwrap(), applied);
    standby.commit(reader).unwrap();
    assert!(matches!(
        standby.apply_shipped_segment(&bytes, WAIT).unwrap(),
        ShipOutcome::Applied { .. }
    ));
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
}

#[test]
fn applied_segments_survive_a_restart_and_promotion_makes_a_primary() {
    let p = primary();
    let (dir, standby) = standby_of(&p);
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    let expected = payloads(&p.engine, p.table);
    drop(standby);
    // Restart: the applied records are in the standby's own log under the primary's positions.
    let wal = dir.path().join("btree.wal");
    let standby = BtreeEngine::open_standby(&wal).unwrap();
    assert_eq!(payloads(&standby, p.table), expected);
    assert_eq!(
        standby.wal_last_lsn().unwrap().unwrap(),
        p.engine.wal_last_lsn().unwrap().unwrap()
    );
    // Promotion: writes are accepted and the history continues from the primary's position.
    standby.set_standby(false);
    insert(&standby, p.table, b"promoted");
    let mut rows = expected;
    rows.push(b"promoted".to_vec());
    rows.sort();
    assert_eq!(payloads(&standby, p.table), rows);
    drop(standby);
    let reopened = BtreeEngine::open(&wal).unwrap();
    assert_eq!(payloads(&reopened, p.table), rows);
}

#[test]
fn seeding_refuses_a_directory_that_already_holds_a_database() {
    let p = primary();
    let err = seed_standby(p.archive.path(), &p.dir.path().join("btree.wal"))
        .expect_err("the primary's own directory");
    assert!(err.to_string().contains("already holds"), "{err}");
}

/// A rollback or a refused write on the standby leaves its log untouched, so the next segment
/// is judged against the primary's position and nothing in it is lost.
#[test]
fn local_rollbacks_and_refused_writes_never_shift_the_standby_position() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let position = standby.wal_last_lsn().unwrap().unwrap();
    let txn = standby.begin(RC).unwrap();
    standby.rollback(txn).unwrap();
    let txn = standby.begin(RC).unwrap();
    // The write fails at its log step, before any change is kept.
    let err = standby
        .insert(txn, p.table, b"local")
        .expect_err("a standby refuses writes");
    assert!(matches!(err, Error::ReadOnly(_)), "{err}");
    standby.rollback(txn).unwrap();
    assert_eq!(standby.wal_last_lsn().unwrap().unwrap(), position);
    insert(&p.engine, p.table, b"p3-a");
    insert(&p.engine, p.table, b"p3-b");
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    assert_eq!(
        standby.wal_last_lsn().unwrap().unwrap(),
        p.engine.wal_last_lsn().unwrap().unwrap()
    );
}

/// The standby's own image is stamped with an id that ended on the primary, so the primary
/// rolling back the id the standby would otherwise have minted cannot erase the image at the
/// next open.
#[test]
fn a_standby_checkpoint_survives_the_primary_aborting_a_colliding_id() {
    let p = primary();
    let (dir, standby) = standby_of(&p);
    // Readers on the standby mint ids past what was applied, as the primary will.
    for _ in 0..3 {
        let txn = standby.begin(RC).unwrap();
        standby.commit(txn).unwrap();
    }
    standby.checkpoint().unwrap();
    // The primary aborts the transactions with those ids, writes on, and checkpoints.
    for _ in 0..4 {
        let txn = p.engine.begin(RC).unwrap();
        p.engine.insert(txn, p.table, b"discarded").unwrap();
        p.engine.rollback(txn).unwrap();
    }
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    let expected = payloads(&p.engine, p.table);
    assert_eq!(payloads(&standby, p.table), expected);
    drop(standby);
    let reopened = BtreeEngine::open_standby(dir.path().join("btree.wal")).unwrap();
    assert_eq!(reopened.list_tables().unwrap().len(), 1);
    assert_eq!(payloads(&reopened, p.table), expected);
}

/// A transaction the primary's crash cut off leaves puts without an ending in the next
/// archived segment; the standby skips them like the primary's own recovery did.
#[test]
fn a_transaction_cut_off_by_a_primary_crash_is_skipped_not_fatal() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let Primary {
        dir,
        archive,
        engine,
        table,
    } = p;
    let orphan = engine.begin(RC).unwrap();
    engine.insert(orphan, table, b"never-ended").unwrap();
    insert(&engine, table, b"p3-a");
    // The crash: the engine is never dropped, so no rollback marker is ever written.
    std::mem::forget(engine);
    let engine = BtreeEngine::open_with_archive(
        dir.path().join("btree.wal"),
        Some(archive.path().to_path_buf()),
    )
    .unwrap();
    engine.checkpoint().unwrap();
    let p = Primary {
        dir,
        archive,
        engine,
        table,
    };
    // The archived segment really carries the cut-off transaction's put: two puts (the orphan
    // and the committed row) but only one commit marker.
    let applied = standby.wal_last_lsn().unwrap().unwrap();
    let (_, path) = shipped_segments_after(p.archive.path(), applied)
        .unwrap()
        .remove(0);
    let prefix = nusadb_wal::recover_prefix(&std::fs::read(path).unwrap()).unwrap();
    let puts = prefix
        .records
        .iter()
        .filter(|(_, r)| matches!(r, nusadb_wal::WalRecord::Put { .. }))
        .count();
    let commits = prefix
        .records
        .iter()
        .filter(|(_, r)| matches!(r, nusadb_wal::WalRecord::CommitTxn { .. }))
        .count();
    assert!(puts >= 2 && commits == 1, "puts {puts}, commits {commits}");
    catch_up(&standby, &p);
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
    assert!(!payloads(&standby, p.table).contains(&b"never-ended".to_vec()));
}

/// A refused write is never visible, however its transaction ends: rolled back, "committed"
/// (the commit is refused and rolls it back), or simply abandoned.
#[test]
fn a_refused_write_is_never_visible_however_its_transaction_ends() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let expected = payloads(&p.engine, p.table);
    let txn = standby.begin(RC).unwrap();
    assert!(standby.insert(txn, p.table, b"rolled-back").is_err());
    standby.rollback(txn).unwrap();
    assert_eq!(payloads(&standby, p.table), expected, "after rollback");
    let txn = standby.begin(RC).unwrap();
    assert!(standby.insert(txn, p.table, b"committed").is_err());
    let _ = standby.commit(txn);
    assert_eq!(payloads(&standby, p.table), expected, "after commit");
    let txn = standby.begin(RC).unwrap();
    assert!(standby.insert(txn, p.table, b"abandoned").is_err());
    assert_eq!(payloads(&standby, p.table), expected, "while abandoned");
    standby.rollback(txn).unwrap();
    assert_eq!(
        payloads(&standby, p.table),
        expected,
        "after the abandoned one ends"
    );
}

/// The SQL layer wraps every statement in a savepoint and inserts VALUES lists as a batch; a
/// refused write through those paths is never visible either.
#[test]
fn a_refused_batch_or_savepoint_write_is_never_visible() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let expected = payloads(&p.engine, p.table);
    let txn = standby.begin(RC).unwrap();
    assert!(
        standby
            .insert_batch(txn, p.table, &[b"b1".to_vec(), b"b2".to_vec()])
            .is_err()
    );
    let _ = standby.commit(txn);
    assert_eq!(
        payloads(&standby, p.table),
        expected,
        "after a refused batch"
    );
    let txn = standby.begin(RC).unwrap();
    standby.savepoint(txn, "s").unwrap();
    assert!(standby.insert(txn, p.table, b"sp").is_err());
    let rolled = standby.rollback_to(txn, "s");
    assert!(rolled.is_ok(), "{rolled:?}");
    let _ = standby.commit(txn);
    assert_eq!(
        payloads(&standby, p.table),
        expected,
        "after a savepoint rollback"
    );
    let txn = standby.begin(RC).unwrap();
    standby.savepoint(txn, "s").unwrap();
    assert!(
        standby
            .insert_batch(txn, p.table, &[b"b3".to_vec()])
            .is_err()
    );
    let rolled = standby.rollback_to(txn, "s");
    assert!(rolled.is_ok(), "{rolled:?}");
    let _ = standby.commit(txn);
    assert_eq!(
        payloads(&standby, p.table),
        expected,
        "batch under a savepoint"
    );
}

/// A segment whose copy is still in progress (a torn tail) applies nothing, so a transaction
/// whose commit lies in the missing part is never split; the complete copy applies it whole.
#[test]
fn a_torn_segment_applies_nothing_until_it_is_complete() {
    let p = primary();
    let (_dir, standby) = standby_of(&p);
    let position = standby.wal_last_lsn().unwrap().unwrap();
    insert(&p.engine, p.table, b"p3-a");
    insert(&p.engine, p.table, b"p3-b");
    p.engine.checkpoint().unwrap();
    let (_, path) = shipped_segments_after(p.archive.path(), position)
        .unwrap()
        .remove(0);
    let bytes = std::fs::read(path).unwrap();
    // Cut inside the last record: its prefix would hold the first row's commit and the second
    // row's put without its commit.
    let torn = &bytes[..bytes.len() - 3];
    assert_eq!(
        standby.apply_shipped_segment(torn, WAIT).unwrap(),
        ShipOutcome::NothingNew
    );
    assert_eq!(standby.wal_last_lsn().unwrap().unwrap(), position);
    assert!(matches!(
        standby.apply_shipped_segment(&bytes, WAIT).unwrap(),
        ShipOutcome::Applied { .. }
    ));
    assert_eq!(payloads(&standby, p.table), payloads(&p.engine, p.table));
}

/// DROP SCHEMA on a standby is refused like any write and leaves the schema in place, live and
/// across the standby's own checkpoint and a reopen.
#[test]
fn drop_schema_on_a_standby_is_refused_and_the_schema_stays() {
    let p = primary();
    let txn = p.engine.begin(RC).unwrap();
    p.engine.create_schema(txn, "app").unwrap();
    p.engine.commit(txn).unwrap();
    p.engine.checkpoint().unwrap();
    let (dir, standby) = standby_of(&p);
    let id = standby
        .lookup_schema("app")
        .unwrap()
        .expect("shipped schema");
    let txn = standby.begin(RC).unwrap();
    let err = standby
        .drop_schema(txn, id, false)
        .expect_err("a standby refuses DDL");
    assert!(matches!(err, Error::ReadOnly(_)), "{err}");
    standby.rollback(txn).unwrap();
    assert_eq!(standby.lookup_schema("app").unwrap(), Some(id));
    standby.checkpoint().unwrap();
    drop(standby);
    let reopened = BtreeEngine::open_standby(dir.path().join("btree.wal")).unwrap();
    assert_eq!(reopened.lookup_schema("app").unwrap(), Some(id));
}

/// A standby that has applied nothing since its seed checkpoints with the seed image's own
/// commit id, which the primary has already consumed; the primary aborting its next ids
/// cannot touch that image at the standby's next open.
#[test]
fn a_freshly_seeded_standby_checkpoint_survives_primary_aborts() {
    let p = primary();
    let (dir, standby) = standby_of(&p);
    for _ in 0..2 {
        let txn = standby.begin(RC).unwrap();
        standby.commit(txn).unwrap();
    }
    standby.checkpoint().unwrap();
    for _ in 0..3 {
        let txn = p.engine.begin(RC).unwrap();
        p.engine.insert(txn, p.table, b"discarded").unwrap();
        p.engine.rollback(txn).unwrap();
    }
    insert(&p.engine, p.table, b"p3-a");
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    let expected = payloads(&p.engine, p.table);
    drop(standby);
    let reopened = BtreeEngine::open_standby(dir.path().join("btree.wal")).unwrap();
    assert_eq!(reopened.list_tables().unwrap().len(), 1);
    assert_eq!(payloads(&reopened, p.table), expected);
}

/// A segment whose last ended transaction is a rollback stamps the standby's next image with
/// that id; the image is intact at the next open, since the abort marker sits before the
/// image's position and is never replayed against it.
#[test]
fn an_abort_marker_as_the_highest_ended_id_leaves_the_standby_image_intact() {
    let p = primary();
    let (dir, standby) = standby_of(&p);
    insert(&p.engine, p.table, b"p3-a");
    let txn = p.engine.begin(RC).unwrap();
    p.engine.insert(txn, p.table, b"discarded").unwrap();
    p.engine.rollback(txn).unwrap();
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    standby.checkpoint().unwrap();
    let expected = payloads(&p.engine, p.table);
    assert_eq!(payloads(&standby, p.table), expected);
    drop(standby);
    let reopened = BtreeEngine::open_standby(dir.path().join("btree.wal")).unwrap();
    assert_eq!(reopened.list_tables().unwrap().len(), 1);
    assert_eq!(payloads(&reopened, p.table), expected);
    // And the line continues: another segment applies on top of the standby's own image.
    insert(&p.engine, p.table, b"p4-a");
    p.engine.checkpoint().unwrap();
    catch_up(&reopened, &p);
    assert_eq!(payloads(&reopened, p.table), payloads(&p.engine, p.table));
}

/// A standby killed mid-append can leave its own log ending between a transaction's puts and
/// its commit. After a restart the next apply still replays that whole transaction, so no row
/// is lost, and its image afterwards holds it.
#[test]
fn a_standby_log_cut_between_a_put_and_its_commit_loses_no_rows() {
    let p = primary();
    let (dir, standby) = standby_of(&p);
    let txn = p.engine.begin(RC).unwrap();
    p.engine
        .insert_batch(
            txn,
            p.table,
            &[b"m1".to_vec(), b"m2".to_vec(), b"m3".to_vec()],
        )
        .unwrap();
    p.engine.commit(txn).unwrap();
    p.engine.checkpoint().unwrap();
    catch_up(&standby, &p);
    let expected = payloads(&p.engine, p.table);
    assert_eq!(payloads(&standby, p.table), expected);
    drop(standby);
    // The crash: cut the standby's own log right before the last record (the commit), on a
    // record boundary, so its puts are durable but their ending is not.
    let wal = dir.path().join("btree.wal");
    let bytes = std::fs::read(&wal).unwrap();
    let full = nusadb_wal::recover_prefix(&bytes).unwrap();
    assert!(matches!(
        full.records.last().map(|(_, r)| r),
        Some(nusadb_wal::WalRecord::CommitTxn { .. })
    ));
    let cut = (1..bytes.len())
        .rev()
        .find(|&len| {
            nusadb_wal::recover_prefix(&bytes[..len]).is_ok_and(|prefix| {
                prefix.good_bytes as usize == len && prefix.records.len() == full.records.len() - 1
            })
        })
        .expect("a record boundary before the commit");
    std::fs::write(&wal, &bytes[..cut]).unwrap();
    let standby = BtreeEngine::open_standby(&wal).unwrap();
    assert!(
        !payloads(&standby, p.table).contains(&b"m1".to_vec()),
        "the cut transaction is not committed on the standby yet"
    );
    catch_up(&standby, &p);
    assert_eq!(payloads(&standby, p.table), expected);
    standby.checkpoint().unwrap();
    drop(standby);
    let reopened = BtreeEngine::open_standby(&wal).unwrap();
    assert_eq!(payloads(&reopened, p.table), expected);
}
