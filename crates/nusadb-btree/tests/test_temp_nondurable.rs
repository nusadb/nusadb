//! Durability pins for non-durable (temporary) schemas and tables.
//!
//! A table created inside a schema minted by `create_temp_schema` is *non-durable*: its DDL and
//! row writes are never appended to the WAL, and the object is excluded from the checkpoint image.
//! The consequence — and the property these tests nail down — is that such a table can never
//! survive a restart, by either recovery path:
//!
//!   * **WAL replay** — nothing was ever logged, so replay cannot resurrect it.
//!   * **Checkpoint image** — `emit_image_records` filters it out, so a reopen from the image
//!     cannot resurrect it either.
//!
//! Meanwhile a *durable* table created in the same session must be wholly unaffected — its rows
//! survive both restart paths byte-for-byte. We also pin the two negative-space properties that
//! make "non-durable" mean what it says: temp-only writes do not grow the WAL on disk, and a
//! rolled-back temp object leaves no trace (no leak in the non-durable bookkeeping, gone after
//! restart regardless).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "durability pins assert via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{ColumnDef, IndexDef, IndexKind, TableDef};
use nusadb_core::{ColumnType, IsolationLevel, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;
const TEMP_SCHEMA: &str = "nusadb_temp_1";

fn one_col(schema: &str, name: &str) -> TableDef {
    TableDef {
        schema: schema.to_owned(),
        name: name.to_owned(),
        columns: vec![ColumnDef {
            name: "v".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

/// Scan every committed row of `table` in row-id order — doubles as a corruption check.
fn scan_payloads(engine: &BtreeEngine, table: TableId) -> Vec<Vec<u8>> {
    let txn = engine.begin(RC).unwrap();
    let mut scan = engine.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.push(tuple.to_vec());
    }
    drop(scan);
    engine.commit(txn).unwrap();
    out
}

/// Seed a session with one durable table (`public.keep`) carrying two rows and one temp table
/// (`nusadb_temp_1.scratch`) carrying two rows. Returns the durable table id.
fn seed(engine: &BtreeEngine) -> TableId {
    let txn = engine.begin(RC).unwrap();
    engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
    let temp = engine
        .create_table(txn, &one_col(TEMP_SCHEMA, "scratch"))
        .unwrap();
    let durable = engine
        .create_table(txn, &one_col(nusadb_core::PUBLIC_SCHEMA, "keep"))
        .unwrap();
    engine.commit(txn).unwrap();

    let txn = engine.begin(RC).unwrap();
    engine.insert(txn, temp, b"temp-a").unwrap();
    engine.insert(txn, temp, b"temp-b").unwrap();
    engine.insert(txn, durable, b"keep-a").unwrap();
    engine.insert(txn, durable, b"keep-b").unwrap();
    engine.commit(txn).unwrap();

    // Within the live session the temp table is fully usable.
    assert_eq!(
        scan_payloads(engine, temp),
        vec![b"temp-a".to_vec(), b"temp-b".to_vec()]
    );
    durable
}

/// After a restart the temp schema/table are gone while the durable table + rows survive intact.
/// Parameterised over whether we checkpoint before closing, so both recovery paths are covered.
fn assert_gone_after_restart(checkpoint_before_close: bool) {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("temp.wal");

    {
        let engine = BtreeEngine::open(&wal).unwrap();
        seed(&engine);
        if checkpoint_before_close {
            engine.checkpoint().unwrap();
        }
    }

    let engine = BtreeEngine::open(&wal).unwrap();

    // Temp schema and its table did not survive.
    assert!(
        engine.lookup_schema(TEMP_SCHEMA).unwrap().is_none(),
        "temp schema must not survive restart (checkpoint={checkpoint_before_close})"
    );
    assert!(
        engine
            .lookup_table_in(TEMP_SCHEMA, "scratch")
            .unwrap()
            .is_none(),
        "temp table must not survive restart (checkpoint={checkpoint_before_close})"
    );

    // Durable table and its rows did survive, byte-for-byte.
    let keep = engine
        .lookup_table_in(nusadb_core::PUBLIC_SCHEMA, "keep")
        .unwrap()
        .expect("durable table must survive restart");
    assert_eq!(
        scan_payloads(&engine, keep.id),
        vec![b"keep-a".to_vec(), b"keep-b".to_vec()],
        "durable rows must survive restart intact (checkpoint={checkpoint_before_close})"
    );
}

#[test]
fn temp_table_gone_after_wal_replay_reopen() {
    // No checkpoint: reopen replays the WAL prefix. The temp table was never logged, so it cannot
    // reappear; the durable table's logged rows replay normally.
    assert_gone_after_restart(false);
}

#[test]
fn temp_table_gone_after_checkpoint_reopen() {
    // Checkpoint first: reopen loads the image. The temp table is excluded from the image, so it
    // cannot reappear; the durable table is imaged and restored.
    assert_gone_after_restart(true);
}

#[test]
fn temp_writes_log_no_row_data_only_commit_markers() {
    // Non-durability means the row *payload* is never appended to the WAL. It does NOT mean the WAL
    // stops growing altogether: every committed transaction still writes a small commit marker (the
    // commit protocol needs it, and on replay a marker with no data ops is a harmless no-op). So the
    // property we pin is: a temp-only transaction grows the WAL by only that fixed marker, strictly
    // less than an otherwise-identical transaction against a durable table, which must additionally
    // log its insert payload. Same payload size on both sides isolates the data term.
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("temp.wal");

    let engine = BtreeEngine::open(&wal).unwrap();

    // One temp table and one durable table, each created + committed up front (temp DDL unlogged).
    let txn = engine.begin(RC).unwrap();
    engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
    let temp = engine
        .create_table(txn, &one_col(TEMP_SCHEMA, "scratch"))
        .unwrap();
    let durable = engine
        .create_table(txn, &one_col(nusadb_core::PUBLIC_SCHEMA, "keep"))
        .unwrap();
    engine.commit(txn).unwrap();

    let wal_len = || std::fs::metadata(&wal).map_or(0, |m| m.len());
    // Fat, *incompressible* fixed-width payloads (the WAL lz4-compresses records, so a run of
    // identical bytes would shrink to nothing and hide the data term). A per-byte LCG defeats lz4
    // without needing an RNG, keeping the test deterministic.
    let payload = |seed: u32| -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..512)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    };

    let before_temp = wal_len();
    for round in 0..50u32 {
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, temp, &payload(round)).unwrap();
        engine.commit(txn).unwrap();
    }
    let temp_growth = wal_len() - before_temp;

    let before_durable = wal_len();
    for round in 0..50u32 {
        let txn = engine.begin(RC).unwrap();
        engine.insert(txn, durable, &payload(round)).unwrap();
        engine.commit(txn).unwrap();
    }
    let durable_growth = wal_len() - before_durable;

    // Same transaction count + payload on both sides. The durable side must be larger by roughly
    // the total (incompressible) payload volume; the temp side carries commit markers only. Assert
    // the gap is at least half the raw payload volume — comfortably above marker noise, proof the
    // row data was never logged for the temp table.
    assert!(
        durable_growth > temp_growth + 50 * 256,
        "durable WAL growth must exceed temp growth by ~payload volume \
         (temp={temp_growth}, durable={durable_growth})"
    );
}

#[test]
fn rolled_back_temp_objects_leave_no_trace() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("temp.wal");

    {
        let engine = BtreeEngine::open(&wal).unwrap();

        // Create a temp schema + table + rows, then roll the whole transaction back.
        let txn = engine.begin(RC).unwrap();
        engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
        let temp = engine
            .create_table(txn, &one_col(TEMP_SCHEMA, "scratch"))
            .unwrap();
        engine.insert(txn, temp, b"gone").unwrap();
        engine.rollback(txn).unwrap();

        // The schema/table are gone immediately, and the non-durable bookkeeping did not leak —
        // proven by being able to re-mint the very same schema name without an "already exists".
        assert!(engine.lookup_schema(TEMP_SCHEMA).unwrap().is_none());
        let txn = engine.begin(RC).unwrap();
        engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
        engine.commit(txn).unwrap();
    }

    // And nothing rolled-back or temp survives a restart.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(engine.lookup_schema(TEMP_SCHEMA).unwrap().is_none());
    assert!(
        engine
            .lookup_table_in(TEMP_SCHEMA, "scratch")
            .unwrap()
            .is_none()
    );
}

/// A temp table that is dropped and then restored by a rollback must stay non-durable — the drop
/// path must not clear its non-durability. This pins a gap where clearing the marker at drop time
/// (before commit) let a restored temp table be re-logged as durable and survive recovery.
///
/// Covers both restore paths: a full transaction abort (`rollback`) and a partial
/// `ROLLBACK TO SAVEPOINT` (which additionally routes logical inverses through the WAL gate).
fn assert_temp_survives_drop_rollback_still_nondurable(savepoint: bool) {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("temp.wal");

    {
        let engine = BtreeEngine::open(&wal).unwrap();

        // A committed temp table with a row.
        let txn = engine.begin(RC).unwrap();
        engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
        let temp = engine
            .create_table(txn, &one_col(TEMP_SCHEMA, "scratch"))
            .unwrap();
        engine.insert(txn, temp, b"before").unwrap();
        engine.commit(txn).unwrap();

        // Drop it, then undo the drop — the table (and its schema) come back.
        let txn = engine.begin(RC).unwrap();
        if savepoint {
            engine.savepoint(txn, "s").unwrap();
            engine.drop_table(txn, temp).unwrap();
            engine.rollback_to(txn, "s").unwrap();
            // The transaction lives on and commits; the restored table must still be non-durable.
            engine.insert(txn, temp, b"after").unwrap();
            engine.commit(txn).unwrap();
        } else {
            engine.drop_table(txn, temp).unwrap();
            engine.rollback(txn).unwrap();
            // Fresh transaction writes to the restored table; it must still be non-durable.
            let txn = engine.begin(RC).unwrap();
            engine.insert(txn, temp, b"after").unwrap();
            engine.commit(txn).unwrap();
        }

        // Still fully usable in-session.
        assert!(
            engine
                .lookup_table_in(TEMP_SCHEMA, "scratch")
                .unwrap()
                .is_some()
        );

        // Checkpoint bakes the image — the restored temp table must be excluded from it.
        engine.checkpoint().unwrap();
    }

    // After restart the restored-then-written temp table must be GONE. If the drop had cleared its
    // non-durable marker, its DDL + rows would have been logged/imaged and it would survive here.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(
        engine.lookup_schema(TEMP_SCHEMA).unwrap().is_none(),
        "temp schema restored by rollback must stay non-durable (savepoint={savepoint})"
    );
    assert!(
        engine
            .lookup_table_in(TEMP_SCHEMA, "scratch")
            .unwrap()
            .is_none(),
        "temp table restored by rollback must stay non-durable (savepoint={savepoint})"
    );
}

#[test]
fn temp_table_restored_by_full_abort_stays_nondurable() {
    assert_temp_survives_drop_rollback_still_nondurable(false);
}

#[test]
fn temp_table_restored_by_savepoint_rollback_stays_nondurable() {
    assert_temp_survives_drop_rollback_still_nondurable(true);
}

/// An index on a temp table is itself non-durable, and DROPPING that index must not leak a WAL
/// record either — `drop_index` resolves durability from the parent table before the index leaves
/// the catalog. A durable table's index is the control: it survives restart. This exercises the
/// full temp-index lifecycle (create → populate → drop) plus a surviving durable index, then pins
/// the outcome across a checkpoint reopen.
#[test]
fn temp_table_index_is_nondurable_and_drop_leaks_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let wal = dir.path().join("temp.wal");

    let keep_id;
    {
        let engine = BtreeEngine::open(&wal).unwrap();

        // Temp table with an index; populate table + index, then drop the index within the session.
        let txn = engine.begin(RC).unwrap();
        engine.create_temp_schema(txn, TEMP_SCHEMA).unwrap();
        let temp = engine
            .create_table(txn, &one_col(TEMP_SCHEMA, "scratch"))
            .unwrap();
        let temp_idx = engine
            .create_index(
                txn,
                &IndexDef {
                    name: "scratch_v".to_owned(),
                    table: temp,
                    columns: vec!["v".to_owned()],
                    key_exprs: Vec::new(),
                    predicate: None,
                    include: Vec::new(),
                    kind: IndexKind::BTree,
                    unique: false,
                },
            )
            .unwrap();
        let tid = engine.insert(txn, temp, b"temp-a").unwrap();
        engine.index_insert(txn, temp_idx, b"temp-a", tid).unwrap();
        engine.commit(txn).unwrap();

        // Drop the temp index — the path under test. Must not resurrect the temp table as durable.
        let txn = engine.begin(RC).unwrap();
        engine.drop_index(txn, temp_idx).unwrap();
        engine.commit(txn).unwrap();

        // A durable table with an index is the control that MUST survive.
        let txn = engine.begin(RC).unwrap();
        let keep = engine
            .create_table(txn, &one_col(nusadb_core::PUBLIC_SCHEMA, "keep"))
            .unwrap();
        let keep_idx = engine
            .create_index(
                txn,
                &IndexDef {
                    name: "keep_v".to_owned(),
                    table: keep,
                    columns: vec!["v".to_owned()],
                    key_exprs: Vec::new(),
                    predicate: None,
                    include: Vec::new(),
                    kind: IndexKind::BTree,
                    unique: false,
                },
            )
            .unwrap();
        let tid = engine.insert(txn, keep, b"keep-a").unwrap();
        engine.index_insert(txn, keep_idx, b"keep-a", tid).unwrap();
        engine.commit(txn).unwrap();
        keep_id = keep;

        engine.checkpoint().unwrap();
    }

    // After restart: the temp table (and thus any index it had) is gone; the durable table with its
    // index survives, rows intact.
    let engine = BtreeEngine::open(&wal).unwrap();
    assert!(engine.lookup_schema(TEMP_SCHEMA).unwrap().is_none());
    assert!(
        engine
            .lookup_table_in(TEMP_SCHEMA, "scratch")
            .unwrap()
            .is_none()
    );
    assert!(engine.lookup_index("scratch_v").unwrap().is_none());

    let keep = engine
        .lookup_table_in(nusadb_core::PUBLIC_SCHEMA, "keep")
        .unwrap()
        .expect("durable table must survive restart");
    assert_eq!(keep.id, keep_id);
    assert!(
        engine.lookup_index("keep_v").unwrap().is_some(),
        "durable index must survive restart"
    );
    assert_eq!(scan_payloads(&engine, keep.id), vec![b"keep-a".to_vec()]);
}
