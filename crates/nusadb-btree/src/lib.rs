//! Clustered **B-link / B+tree** storage engine — the sole implementation of the
//! [`StorageEngine`](nusadb_core::StorageEngine) treaty. Decision record: ADR 008.
//!
//! # Shape (per ADR 008 §D2)
//!
//! - **Clustered heap**: the table *is* a B+tree keyed by an internal monotonic **row-id**; rows
//!   live in the leaves (no separate heap, no indirection). `Tid` *is* the row-id — stable
//!   across splits/merges by construction, so every Tid-addressed treaty contract holds.
//!   Monotonic row-ids make the sequential-insert fast path (append at the rightmost leaf) the
//!   default insert path.
//! - **B-link concurrency** (Lehman–Yao): every node carries a right-link + high key; a split
//!   installs the new right sibling *before* the split node is rewritten and the parent
//!   separator published, and any descent finding its key at or beyond a node's high key chases
//!   the right-link — the ordering discipline is structural from the start (see [`tree`]'s module
//!   docs). The engine state is **sharded** (per-table writer latches, latch-free
//!   B-link readers, O(1) txn/lock-manager critical sections — see [`engine::BtreeEngine`]'s
//!   latching docs); per-page/OLC writer latching is a later refinement.
//! - **Node = one 8 KiB page** (`nusadb_core::PAGE_SIZE`); the node codec lives in [`node`].
//! - **MVCC**: undo log + read view. **Durability**: physiological redo WAL + double-write
//!   torn-page protection + ARIES recovery. **Secondary indexes**: payload = row-id.
//!   **Purge**: incremental undo reclaim, the no-stall vacuum equivalent.
//!
//! # Phase status
//!
//! **purge — the no-stall vacuum equivalent.** [`BtreeEngine::purge`]
//! reclaims, in one incremental pass, everything **no current or future view can reach**: undo
//! chains below a settled newest version (slots recycled through a free list and reused by the
//! next update), rows whose delete is settled (physically removed, their stale index entries
//! swept), and the pages of dropped tables once the dropping transaction settles (an aborted
//! `CREATE TABLE` frees its tree immediately). "Settled" = the transaction ended **and** every
//! active transaction's pinned view sees it — so a pinned `REPEATABLE READ` snapshot blocks
//! exactly the history it still needs, and nothing else. Purge is unlogged (it changes no
//! logical content; recovery collapses versions anyway) and explicitly invoked — background
//! cadence is wired at the composition root. Structural reclamation of empty/underfull
//! leaf *pages* (they stay in the chain today) arrives with page-store persistence.
//!
//! Below it, **secondary indexes, payload = row-id**, on top of the durable
//! engine. The full treaty index family is live — `create_index`/`drop_index` (rollback-aware
//! DDL), `index_insert`/`index_delete` (buffered with the transaction, undone on rollback,
//! WAL-logged for recovery), `index_scan` (ascending key order over `[lo, hi]`; every entry is
//! a **pointer** resolved against the caller's read view at the clustered heap — the 2-hop of
//! ADR 008 §D2 · so stale entries filter out naturally), plus `lookup_index`/`list_indexes`/
//! `index_is_complete` and `txn_isolation` (the SQL layer's guard against index scans under a
//! frozen snapshot), and — since the constraint/catalog family landed (//! unblocking CREATE TABLE / the QA SQL-verify) — the full constraint/catalog surface:
//! `add_unique_constraint` (PK/UNIQUE with backing unique index, at-most-one-PK),
//! `add_check_constraint`, `drop_constraint` (FK-RESTRICT guard), the **sequence family**
//! **DDL evolution** (`alter_table` — ADD/DROP/RENAME column, RENAME table, ALTER TYPE,
//! SET/DROP NOT NULL, schema-versioned + `schema_for_version`/`current_schema_version`; the SQL
//! layer eagerly rewrites rows on ALTER so no per-row lazy migration is needed — plus
//! `create_schema`/`drop_schema` [RESTRICT/CASCADE]/`lookup_schema`/`list_schemas` and
//! schema-qualified `lookup_table_in`), the **sequence family**
//! (`create`/`drop`/`lookup`/`next`/`current`/`set` — non-transactional, every advance fsynced
//! before the value escapes so a crash never repeats one; a rolled-back create is neutralized
//! by an equally-durable drop record), `list_constraints`/
//! `has_unique_constraint`, `add_foreign_key`/`list_foreign_keys`/`fk_check`/`fk_on_delete`
//! (cascade one level), `analyze_table`/`table_stats`/`row_count`, and version-0
//! `schema_for_version`/`current_schema_version` (no ALTER yet). All rollback-aware,
//! WAL-durable, savepoint-compensated. Unique indexes byte-check against **live** heap rows,
//! with constraint-**backing** indexes exempt (the SQL layer owns the semantics).
//! Entries live in a sorted
//! in-memory map rebuilt from the WAL on open — the page-native key-bytes B-link index tree
//! rides in with page-store persistence (phase 2), when *any* structure stops being
//! log-rebuilt. Durability semantics unchanged: commit-fsync durability point, committed-only
//! two-pass replay, torn-tail truncation, savepoint compensation (now covering index ops and
//! re-logging live rows/entries for dropped tables/indexes so replay never resurrects them
//! empty). Remaining limitations, each owned by a later phase: empty/underfull leaf pages stay
//! chained until page-store persistence brings structural deletion; a tuple must fit one leaf
//! after the 24-byte version header (overflow pages later); purge scheduling is the caller's
//! for now. Correctness gates: unit invariants + isolation + crash tests now,
//! differential byte-parity as the cluster layer lands.

mod engine;
pub mod mvcc;
pub mod node;
pub mod store;
pub mod tree;
pub mod wal;

pub use engine::{BtreeEngine, PurgeStats, VersionMetadata};

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use super::*;

    /// The audit-caught staged-window schedule (SSI narrowing, durable engine): a reader
    /// that BEGINS while a writer is staged-but-mid-fsync cannot see the writer rows, yet the
    /// writer commit outranks it — so if the reader read the OLD row it MUST abort. A single
    /// stage-instant version map let the reader inherit the writer bump into its baseline and
    /// skip that validation (write-skew committed); the staged/finished split closes it. The
    /// assertion branches on what the reader actually read, so an iteration that misses the
    /// fsync window (reader began after finish) stays sound rather than flaky.
    #[test]
    fn staged_window_reader_still_aborts_on_write_skew() {
        let dir = tempfile::tempdir().unwrap();
        let engine: &'static BtreeEngine = Box::leak(Box::new(
            BtreeEngine::open(dir.path().join("w.wal")).unwrap(),
        ));
        let setup = engine.begin(RC).unwrap();
        let tx = engine.create_table(setup, &table_def("tx")).unwrap();
        let ty = engine.create_table(setup, &table_def("ty")).unwrap();
        engine.insert(setup, tx, &[1]).unwrap();
        engine.insert(setup, ty, &[1]).unwrap();
        engine.commit(setup).unwrap();

        for _ in 0..100 {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
            let writer = engine.begin(IsolationLevel::Serializable).unwrap();
            // Writer reads y (its anti-dependency side) and writes x.
            assert_eq!(collect(engine, writer, ty).len(), 1);
            let x_now = engine
                .scan(writer, tx)
                .unwrap()
                .try_next()
                .unwrap()
                .unwrap();
            engine.update(writer, tx, x_now.0, &[9]).unwrap();

            let b = std::sync::Arc::clone(&barrier);
            let w_handle = std::thread::spawn(move || {
                b.wait();
                engine.commit(writer) // stages, fsyncs (the window), finishes
            });
            barrier.wait();
            // Reader begins racing the writer commit — often inside the fsync window.
            let reader = engine.begin(IsolationLevel::Serializable).unwrap();
            let read_x = collect(engine, reader, tx)
                .into_iter()
                .next()
                .map(|(_, bytes)| bytes);
            // Reader writes y (completing the write-skew shape), then commits last.
            let y_now = engine
                .scan(reader, ty)
                .unwrap()
                .try_next()
                .unwrap()
                .unwrap();
            engine.update(reader, ty, y_now.0, &[7]).unwrap();
            w_handle.join().unwrap().unwrap();
            let reader_result = engine.commit(reader);
            match read_x.as_deref() {
                // Reader saw the OLD x: the writer committed first with the conflicting write
                // — the reader must abort, staged window or not.
                Some([1]) => assert!(
                    matches!(
                        reader_result,
                        Err(nusadb_core::Error::SerializationConflict { .. })
                    ),
                    "old-x reader must abort (staged-window skip must not hide the writer)"
                ),
                // Reader saw the NEW x (began after finish): no antidependency, commit is fine.
                Some([9]) => {
                    reader_result.unwrap();
                    // Restore y for the next iteration (this reader's y write committed).
                    let fix = engine.begin(RC).unwrap();
                    let y_now = engine.scan(fix, ty).unwrap().try_next().unwrap().unwrap();
                    engine.update(fix, ty, y_now.0, &[1]).unwrap();
                    engine.commit(fix).unwrap();
                },
                other => panic!("unexpected x read: {other:?}"),
            }
            // Restore x for the next iteration.
            let fix = engine.begin(RC).unwrap();
            let x_now = engine.scan(fix, tx).unwrap().try_next().unwrap().unwrap();
            engine.update(fix, tx, x_now.0, &[1]).unwrap();
            engine.commit(fix).unwrap();
        }
    }

    /// SSI narrowing: a `SERIALIZABLE` reader whose read tables saw NO concurrent
    /// committed writer skips validation entirely — and, critically, a writer of an UNRELATED
    /// table must never make it abort, while a writer of the READ table still does (the
    /// write-skew pin elsewhere proves the full conflict path survives the skip).
    #[test]
    fn serializable_reader_unaffected_by_unrelated_table_writers() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let a = engine.create_table(setup, &table_def("a")).unwrap();
        let b = engine.create_table(setup, &table_def("b")).unwrap();
        engine.insert(setup, a, &[1]).unwrap();
        engine.insert(setup, b, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // Reader of `a` overlaps a committing writer of `b` only: must commit cleanly.
        let reader = engine.begin(SER).unwrap();
        assert_eq!(collect(&engine, reader, a).len(), 1);
        let writer = engine.begin(RC).unwrap();
        engine.insert(writer, b, &[2]).unwrap();
        engine.commit(writer).unwrap();
        engine.commit(reader).unwrap();

        // Reader of `a` overlaps a committed UPDATE of the row it read: still aborts (40001).
        let reader = engine.begin(SER).unwrap();
        let rows = collect(&engine, reader, a);
        let tid = nusadb_core::engine::Tid {
            page: nusadb_core::PageId(rows[0].0),
            slot: nusadb_core::SlotIdx(0),
        };
        let writer = engine.begin(RC).unwrap();
        engine.update(writer, a, tid, &[9]).unwrap();
        engine.commit(writer).unwrap();
        assert!(
            matches!(
                engine.commit(reader),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "a concurrent committed write of a READ row must still abort the reader"
        );
    }

    /// `get` after the zero-allocation rewrite (`descend_read` + `leaf_find`): every
    /// inserted key reads back its exact tuple and every absent probe — before the first key,
    /// between keys, past the last — reads `None`, across enough sparse keys to split leaves
    /// and grow interior levels (so routing and the early-exit walk both face real boundaries).
    #[test]
    fn tree_get_finds_every_key_and_only_those() {
        let store = store::MemPageStore::default();
        let mut tree = tree::ClusteredTree::create(&store).unwrap();
        // Sparse keys (i*3+1) with ~200-byte tuples: forces multi-level structure well past one
        // leaf; the gaps give absent keys inside every leaf's range.
        let tuple = |i: u64| vec![u8::try_from(i % 251).unwrap(); 200];
        for i in 0..20_000 {
            tree.insert(i * 3 + 1, &tuple(i)).unwrap();
        }
        for i in 0..20_000 {
            assert_eq!(
                tree.get(i * 3 + 1).unwrap().as_deref(),
                Some(tuple(i).as_slice()),
                "present key {}",
                i * 3 + 1
            );
            assert_eq!(tree.get(i * 3).unwrap(), None, "absent key {}", i * 3);
            assert_eq!(
                tree.get(i * 3 + 2).unwrap(),
                None,
                "absent key {}",
                i * 3 + 2
            );
        }
        assert_eq!(tree.get(0).unwrap(), None);
        assert_eq!(tree.get(u64::MAX).unwrap(), None);
    }
    use nusadb_core::engine::{IndexDef, IndexKind, IsolationLevel, TableDef};
    use nusadb_core::{ColumnDef, ColumnType, StorageEngine};

    const RC: IsolationLevel = IsolationLevel::ReadCommitted;
    const SER: IsolationLevel = IsolationLevel::Serializable;

    fn table_def(name: &str) -> TableDef {
        TableDef {
            schema: "public".to_owned(),
            name: name.to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        }
    }

    fn collect(
        engine: &BtreeEngine,
        txn: nusadb_core::TxnId,
        table: nusadb_core::TableId,
    ) -> Vec<(u64, Vec<u8>)> {
        let mut scan = engine.scan(txn, table).unwrap();
        let mut out = Vec::new();
        while let Some((tid, tuple)) = scan.try_next().unwrap() {
            out.push((tid.page.0, tuple.to_vec()));
        }
        out
    }

    /// The single-byte value of the row at `tid`, as `txn` sees it (a tiny counter read).
    fn current(
        engine: &BtreeEngine,
        txn: nusadb_core::TxnId,
        table: nusadb_core::TableId,
        tid: nusadb_core::engine::Tid,
    ) -> u8 {
        collect(engine, txn, table)
            .into_iter()
            .find(|(id, _)| *id == tid.page.0)
            .and_then(|(_, v)| v.first().copied())
            .expect("row visible")
    }

    /// CRUD round-trip through the treaty: inserts scan back in row-id order, updates replace
    /// in place (same Tid), deletes disappear.
    #[test]
    fn crud_round_trips_in_row_id_order() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();

        let mut tids = Vec::new();
        for i in 0..100u8 {
            tids.push(engine.insert(txn, table, &[i]).unwrap());
        }
        let rows = collect(&engine, txn, table);
        assert_eq!(rows.len(), 100);
        assert!(rows.windows(2).all(|w| w[0].0 < w[1].0), "row-id order");

        let same = engine.update(txn, table, tids[7], &[200]).unwrap();
        assert_eq!(same, tids[7], "single-version update keeps the address");
        engine.delete(txn, table, tids[8]).unwrap();

        let rows = collect(&engine, txn, table);
        assert_eq!(rows.len(), 99);
        assert!(
            rows.iter()
                .any(|(id, t)| *id == tids[7].page.0 && t == &vec![200])
        );
        assert!(rows.iter().all(|(id, _)| *id != tids[8].page.0));
        engine.commit(txn).unwrap();
    }

    /// Enough rows to force leaf AND interior splits (multi-level tree): everything scans back,
    /// in order, and point operations against pre-split Tids still land afterward (Tid stability
    /// across splits — the ADR 008 row-address decision).
    #[test]
    fn splits_preserve_order_and_addresses() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();

        // ~600-byte tuples: ~13 per leaf, 5000 rows => hundreds of leaves + interior levels.
        let payload = vec![0xAB_u8; 600];
        let mut tids = Vec::new();
        for i in 0..5000u64 {
            let mut tuple = payload.clone();
            tuple[..8].copy_from_slice(&i.to_le_bytes());
            tids.push(engine.insert(txn, table, &tuple).unwrap());
        }
        let rows = collect(&engine, txn, table);
        assert_eq!(rows.len(), 5000);
        assert!(
            rows.windows(2).all(|w| w[0].0 < w[1].0),
            "leaf chain is sorted"
        );
        for (i, (row_id, tuple)) in rows.iter().enumerate() {
            assert_eq!(*row_id, tids[i].page.0, "tid stable across splits");
            let expect = u64::try_from(i).unwrap().to_le_bytes();
            assert_eq!(&tuple[..8], &expect, "payload intact");
        }
        // Updates against pre-split Tids still land after the tree deepened.
        engine.update(txn, table, tids[0], &[1u8; 600]).unwrap();
        engine.update(txn, table, tids[4999], &[2u8; 600]).unwrap();
        let rows = collect(&engine, txn, table);
        assert_eq!(rows[0].1, vec![1u8; 600]);
        assert_eq!(rows[4999].1, vec![2u8; 600]);
        engine.commit(txn).unwrap();
    }

    /// ROLLBACK undoes inserts, updates, deletes, and DDL, newest-first; SAVEPOINT rolls back
    /// partially and stays reusable after ROLLBACK TO.
    #[test]
    fn rollback_and_savepoints_undo_writes() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let keep = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // Rollback: an insert + update + delete all revert.
        let txn = engine.begin(RC).unwrap();
        let added = engine.insert(txn, table, &[2]).unwrap();
        engine.update(txn, table, keep, &[9]).unwrap();
        engine.delete(txn, table, keep).unwrap();
        engine.rollback(txn).unwrap();
        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table);
        assert_eq!(rows, vec![(keep.page.0, vec![1])], "state restored exactly");
        assert!(rows.iter().all(|(id, _)| *id != added.page.0));
        engine.commit(check).unwrap();

        // Savepoint: writes after the mark revert; writes before it stay; the savepoint remains
        // usable after a ROLLBACK TO.
        let txn = engine.begin(RC).unwrap();
        let before = engine.insert(txn, table, &[3]).unwrap();
        engine.savepoint(txn, "sp").unwrap();
        engine.insert(txn, table, &[4]).unwrap();
        engine.rollback_to(txn, "sp").unwrap();
        engine.insert(txn, table, &[5]).unwrap();
        engine.rollback_to(txn, "sp").unwrap();
        engine.commit(txn).unwrap();
        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table);
        assert_eq!(rows.len(), 2, "keep + the pre-savepoint insert survive");
        assert!(rows.iter().any(|(id, _)| *id == before.page.0));
        engine.commit(check).unwrap();

        // DDL rollback: a created table vanishes; a dropped table returns with its rows.
        let txn = engine.begin(RC).unwrap();
        engine.create_table(txn, &table_def("ephemeral")).unwrap();
        engine.drop_table(txn, table).unwrap();
        assert!(engine.lookup_table("t").unwrap().is_none());
        engine.rollback(txn).unwrap();
        assert!(engine.lookup_table("ephemeral").unwrap().is_none());
        assert!(engine.lookup_table("t").unwrap().is_some());
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table).len(),
            2,
            "rows survive the drop-rollback"
        );
        engine.commit(check).unwrap();
    }

    /// The audit-caught split bug: variable-length tuples mean a COUNT-midpoint split can leave
    /// a half that still overflows. The split point is byte-sized (first-fit chunking), so a
    /// legal near-`MAX_TUPLE` insert next to small rows must succeed — including the
    /// `[small, huge, small]` shape where NO single two-way split exists (three leaves).
    #[test]
    fn variable_size_tuples_split_by_bytes_not_count() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();

        // Audit scenario: 10 B, 1000 B, then 7200 B — the count midpoint would pair the two
        // biggest in one half and fail.
        engine.insert(txn, table, &[1u8; 10]).unwrap();
        engine.insert(txn, table, &[2u8; 1000]).unwrap();
        engine.insert(txn, table, &[3u8; 7200]).unwrap();
        let rows = collect(&engine, txn, table);
        assert_eq!(
            rows.iter().map(|(_, t)| t.len()).collect::<Vec<_>>(),
            vec![10, 1000, 7200]
        );

        // Two near-capacity rows next to a small one: the chunker packs by bytes (two leaves
        // here; the true three-leaf shape is exercised by the mixed-size batch below).
        let table2 = engine.create_table(txn, &table_def("t2")).unwrap();
        engine.insert(txn, table2, &[1u8; 10]).unwrap();
        engine.insert(txn, table2, &[2u8; 8000]).unwrap();
        engine.insert(txn, table2, &[3u8; 8000]).unwrap();
        let rows = collect(&engine, txn, table2);
        assert_eq!(
            rows.iter().map(|(_, t)| t.len()).collect::<Vec<_>>(),
            vec![10, 8000, 8000]
        );

        // An UPDATE that grows a mid-leaf row near the cap splits correctly too, and mixed
        // random-ish sizes keep the scan complete and ordered.
        let table3 = engine.create_table(txn, &table_def("t3")).unwrap();
        let mut tids = Vec::new();
        for i in 0..200u64 {
            let len = usize::try_from((i * 37) % 900 + 1).unwrap();
            tids.push(engine.insert(txn, table3, &vec![7u8; len]).unwrap());
        }
        engine.update(txn, table3, tids[100], &[9u8; 8000]).unwrap();
        engine.update(txn, table3, tids[150], &[8u8; 7000]).unwrap();
        let rows = collect(&engine, txn, table3);
        assert_eq!(rows.len(), 200);
        assert!(
            rows.windows(2).all(|w| w[0].0 < w[1].0),
            "scan stays sorted"
        );
        assert!(
            rows.iter()
                .any(|(id, t)| *id == tids[100].page.0 && t.len() == 8000)
        );
        assert!(
            rows.iter()
                .any(|(id, t)| *id == tids[150].page.0 && t.len() == 7000)
        );
        engine.commit(txn).unwrap();
    }

    /// Snapshot semantics: `REPEATABLE READ` pins its BEGIN view (a commit after it stays
    /// invisible), `READ COMMITTED` sees each new commit at its next read, and uncommitted
    /// writes are invisible to everyone but their own transaction.
    #[test]
    fn mvcc_read_views_isolate_snapshots() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        let rr = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        let rc = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, rr, table).len(), 1);
        assert_eq!(collect(&engine, rc, table).len(), 1);

        // A concurrent insert, first uncommitted, then committed.
        let writer = engine.begin(RC).unwrap();
        engine.insert(writer, table, &[2]).unwrap();
        assert_eq!(
            collect(&engine, rc, table).len(),
            1,
            "uncommitted writes are invisible to other transactions"
        );
        assert_eq!(
            collect(&engine, writer, table).len(),
            2,
            "a transaction always sees its own writes"
        );
        engine.commit(writer).unwrap();

        assert_eq!(
            collect(&engine, rr, table).len(),
            1,
            "REPEATABLE READ keeps its BEGIN snapshot"
        );
        // READ COMMITTED holds one snapshot per statement: without a new statement it still
        // sees its pinned view, and only advances after `begin_statement`.
        assert_eq!(
            collect(&engine, rc, table).len(),
            1,
            "READ COMMITTED holds its statement snapshot until the next statement"
        );
        engine.begin_statement(rc).unwrap();
        assert_eq!(
            collect(&engine, rc, table).len(),
            2,
            "READ COMMITTED sees the new commit at a new statement"
        );
        engine.commit(rr).unwrap();
        engine.commit(rc).unwrap();
    }

    /// Version chains: an old snapshot keeps reading the OLD tuple through the undo chain
    /// after newer transactions update (and even delete) the row and commit.
    #[test]
    fn mvcc_undo_chain_serves_old_snapshots() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[10]).unwrap();
        engine.commit(setup).unwrap();

        let old = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(collect(&engine, old, table), vec![(tid.page.0, vec![10])]);

        // Update twice, then delete, each by a committed later transaction.
        for value in [20u8, 30u8] {
            let w = engine.begin(RC).unwrap();
            engine.update(w, table, tid, &[value]).unwrap();
            engine.commit(w).unwrap();
        }
        let deleter = engine.begin(RC).unwrap();
        engine.delete(deleter, table, tid).unwrap();
        engine.commit(deleter).unwrap();

        assert_eq!(
            collect(&engine, old, table),
            vec![(tid.page.0, vec![10])],
            "the pinned snapshot walks the chain back to its version"
        );
        engine.commit(old).unwrap();
        let fresh = engine.begin(RC).unwrap();
        assert!(
            collect(&engine, fresh, table).is_empty(),
            "a fresh view sees the committed delete"
        );
        engine.commit(fresh).unwrap();
    }

    /// Write-write conflicts are no-wait: touching a row with an uncommitted foreign version
    /// raises `SerializationConflict` (40001); REPEATABLE READ also conflicts on a commit after
    /// its snapshot, while READ COMMITTED proceeds.
    #[test]
    fn mvcc_write_conflicts_are_no_wait() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // Uncommitted foreign write: 40001 for a second writer, no blocking.
        let t1 = engine.begin(RC).unwrap();
        engine.update(t1, table, tid, &[2]).unwrap();
        let t2 = engine.begin(RC).unwrap();
        assert!(
            matches!(
                engine.update(t2, table, tid, &[3]),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "second writer must get 40001, not block"
        );
        assert!(matches!(
            engine.delete(t2, table, tid),
            Err(nusadb_core::Error::SerializationConflict { .. })
        ));
        engine.rollback(t1).unwrap();
        engine.rollback(t2).unwrap();

        // RR: a commit after the snapshot conflicts (first-updater-wins); RC proceeds.
        let rr = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(collect(&engine, rr, table).len(), 1);
        let w = engine.begin(RC).unwrap();
        engine.update(w, table, tid, &[7]).unwrap();
        engine.commit(w).unwrap();
        assert!(matches!(
            engine.update(rr, table, tid, &[8]),
            Err(nusadb_core::Error::SerializationConflict { .. })
        ));
        engine.rollback(rr).unwrap();
        let rc = engine.begin(RC).unwrap();
        engine.update(rc, table, tid, &[9]).unwrap();
        engine.commit(rc).unwrap();

        // A committed delete makes later writes not-found, not a conflict.
        let d = engine.begin(RC).unwrap();
        engine.delete(d, table, tid).unwrap();
        engine.commit(d).unwrap();
        let late = engine.begin(RC).unwrap();
        let err = engine.update(late, table, tid, &[9]).expect_err("row gone");
        assert!(err.to_string().contains("no row at tid"));
        engine.rollback(late).unwrap();
    }

    /// Rollback erases versions physically: after an abort, other transactions read exactly
    /// the pre-transaction state (chain included), and the aborted stamps never surface.
    #[test]
    fn mvcc_rollback_leaves_no_versions_behind() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        let t = engine.begin(RC).unwrap();
        engine.update(t, table, tid, &[2]).unwrap();
        engine.insert(t, table, &[3]).unwrap();
        engine.delete(t, table, tid).unwrap();
        engine.rollback(t).unwrap();

        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table),
            vec![(tid.page.0, vec![1])],
            "aborted versions are gone; the committed state is intact"
        );
        // And the row is writable again (no phantom conflict from the aborted txn).
        engine.update(check, table, tid, &[5]).unwrap();
        engine.commit(check).unwrap();
    }

    #[test]
    fn per_transaction_write_memory_limit_rejects_loudly_and_the_engine_survives() {
        // With a per-transaction write-memory ceiling, a transaction that writes past it fails
        // loudly (OutOfMemory) and aborts — no partial state leaks and the engine stays fully usable
        // — instead of the in-memory page store growing unbounded until the OS OOM-kills the whole
        // server. (The default, no limit, is exercised unchanged by every other test.)
        // Each row is an 8-byte tuple charged at its logical length plus the fixed per-row footprint
        // overhead; a ceiling of exactly five rows' charge admits five, then rejects the sixth.
        let row_charge = 8 + crate::engine::PER_ROW_WRITE_OVERHEAD;
        let engine = BtreeEngine::new().with_max_txn_write_bytes(Some(5 * row_charge));
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let t = engine.begin(RC).unwrap();
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
        assert!(
            rejected,
            "the write-memory ceiling must reject the oversized transaction"
        );
        assert_eq!(
            inserted, 5,
            "five rows' worth fits under the five-row ceiling, the sixth rejects"
        );
        // The rejected transaction aborts cleanly; no partial state survives.
        engine.rollback(t).unwrap();

        // The engine is still fully usable: a fresh, bounded transaction commits and is the only
        // committed state (the aborted writes left nothing behind).
        let ok = engine.begin(RC).unwrap();
        let tid = engine.insert(ok, table, &[7]).unwrap();
        engine.commit(ok).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(tid.page.0, vec![7])]);
    }

    #[test]
    fn global_resident_memory_ceiling_rejects_insert_gracefully_and_delete_stays_open() {
        // A global resident-memory ceiling bounds the in-memory page store's whole footprint: once
        // it is reached, an INSERT fails loudly (OutOfMemory) instead of the store growing until the
        // OS OOM-kills the server — the streamed-bulk-load case the *per-transaction* ceiling misses
        // (many small committed batches, each under the per-txn limit but accumulating resident).
        // DELETE is deliberately NOT gated, so an operator can still free space at the ceiling.
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        // Pin the ceiling a few pages above the current footprint, then load in small *committed*
        // batches so the per-transaction ceiling is irrelevant — this exercises only the global
        // guard. Rebuild the engine with the ceiling (the builder is set-once before sharing).
        let ceiling = engine.resident_bytes().unwrap() + 6 * nusadb_core::PAGE_SIZE as u64;
        let engine = engine.with_max_total_resident_bytes(Some(ceiling));

        let mut committed: Vec<nusadb_core::engine::Tid> = Vec::new();
        let mut rejected = false;
        'load: for batch in 0..10_000i64 {
            let t = engine.begin(RC).unwrap();
            let mut batch_tids = Vec::new();
            for i in 0..16i64 {
                match engine.insert(t, table, &(batch * 16 + i).to_le_bytes()) {
                    Ok(tid) => batch_tids.push(tid),
                    Err(nusadb_core::Error::OutOfMemory(_)) => {
                        // The rejected batch aborts cleanly; nothing partial survives.
                        engine.rollback(t).unwrap();
                        rejected = true;
                        break 'load;
                    },
                    Err(e) => panic!("unexpected error: {e:?}"),
                }
            }
            engine.commit(t).unwrap();
            committed.extend(batch_tids);
        }
        assert!(
            rejected,
            "the resident-memory ceiling must eventually reject a growing bulk load"
        );
        assert!(
            !committed.is_empty(),
            "rows below the ceiling committed before it was hit"
        );
        assert!(
            engine.resident_bytes().unwrap() >= ceiling,
            "the store grew up to the ceiling before rejecting"
        );

        // DELETE is not gated by the resident ceiling, so an operator can still free space: a delete
        // transaction commits even though the store is at its ceiling.
        let del = engine.begin(RC).unwrap();
        engine
            .delete(del, table, committed[0])
            .expect("DELETE must stay available at the resident ceiling");
        engine.commit(del).unwrap();
    }

    #[test]
    fn resident_memory_ceiling_does_not_block_crash_recovery() {
        // Recovery replays committed data through `replay_op`, NOT `insert`, so a resident ceiling
        // set below the recovered data size must never abort recovery — the durably committed rows
        // are restored in full, and only *new* writes past the ceiling are then rejected.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("guard.wal");

        let footprint;
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            engine.commit(setup).unwrap();
            for batch in 0..40i64 {
                let t = engine.begin(RC).unwrap();
                for i in 0..16i64 {
                    engine
                        .insert(t, table, &(batch * 16 + i).to_le_bytes())
                        .unwrap();
                }
                engine.commit(t).unwrap();
            }
            footprint = engine.resident_bytes().unwrap();
        } // clean drop; the WAL holds every committed row.

        // Reopen with a ceiling well BELOW the recovered footprint: recovery must still complete.
        let ceiling = footprint / 2;
        let engine = BtreeEngine::open(&path)
            .unwrap()
            .with_max_total_resident_bytes(Some(ceiling));
        let scan = engine.begin(RC).unwrap();
        let table = engine.lookup_table("t").unwrap().unwrap();
        assert_eq!(
            collect(&engine, scan, table.id).len(),
            640,
            "every committed row is recovered despite the ceiling being below the data size"
        );
        // A new insert past the ceiling is now (correctly) rejected.
        let t = engine.begin(RC).unwrap();
        assert!(matches!(
            engine.insert(t, table.id, &[9]),
            Err(nusadb_core::Error::OutOfMemory(_))
        ));
        engine.rollback(t).unwrap();
    }

    #[test]
    fn per_transaction_delete_is_charged_against_the_write_memory_ceiling() {
        // DELETE also grows the transaction's undo log (it retains the old row for rollback/MVCC), so
        // a mass DELETE in one transaction is bounded by the same ceiling. Load 100 rows in one txn
        // that exactly fills a 100-row ceiling, then commit (which resets the per-txn counter).
        // Deleting them in a second txn charges each retained old row — which carries an MVCC header
        // and so is strictly larger than an insert's charge — so the cumulative delete charge crosses
        // the ceiling before all 100 are gone: it rejects loudly and the engine stays usable (the
        // aborted deletes roll back, leaving all 100 rows intact).
        let insert_charge = 8 + crate::engine::PER_ROW_WRITE_OVERHEAD;
        let engine = BtreeEngine::new().with_max_txn_write_bytes(Some(100 * insert_charge));
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let load = engine.begin(RC).unwrap();
        let mut tids = Vec::new();
        for i in 0..100i64 {
            tids.push(engine.insert(load, table, &i.to_le_bytes()).unwrap());
        }
        engine.commit(load).unwrap(); // 100 inserts fill the ceiling exactly; commit resets the counter.

        let del = engine.begin(RC).unwrap();
        let mut deleted = 0u64;
        let mut rejected = false;
        for tid in &tids {
            match engine.delete(del, table, *tid) {
                Ok(()) => deleted += 1,
                Err(nusadb_core::Error::OutOfMemory(_)) => {
                    rejected = true;
                    break;
                },
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(
            rejected,
            "a mass DELETE must be charged and reject once it passes the ceiling"
        );
        assert!(
            deleted > 0 && deleted < 100,
            "some deletes fit under the ceiling before it was hit (got {deleted})"
        );
        engine.rollback(del).unwrap();
        // The engine survives: every row is intact (the aborted deletes rolled back).
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table).len(), 100);
    }

    #[test]
    fn write_charge_counts_per_row_footprint_not_just_logical_bytes() {
        // D0-footprint: the ceiling must account for each row's real retained footprint (MVCC header,
        // slot, undo record), not only its logical tuple bytes — otherwise a flood of tiny rows piles
        // up past the ceiling before Σtuple.len() reaches it. A ceiling set above one tuple's logical
        // size but below its charged footprint must reject the very first insert.
        let tuple = 1i64.to_le_bytes(); // 8 logical bytes
        let ceiling = tuple.len() as u64 + crate::engine::PER_ROW_WRITE_OVERHEAD / 2;
        let engine = BtreeEngine::new().with_max_txn_write_bytes(Some(ceiling));
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let t = engine.begin(RC).unwrap();
        assert!(
            matches!(
                engine.insert(t, table, &tuple),
                Err(nusadb_core::Error::OutOfMemory(_))
            ),
            "a row whose logical bytes fit the ceiling but whose real footprint does not must reject"
        );
        engine.rollback(t).unwrap();
    }

    #[test]
    fn approx_row_count_tracks_committed_net_row_changes() {
        // The O(1) approximate row count: initialized once, then maintained by each commit's net
        // (inserted − deleted). A rolled-back write never skews it; an insert+delete of the same
        // rows in one transaction nets to zero.
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        // A fresh table: the first read initializes the counter to the actual count (0).
        assert_eq!(engine.approx_row_count(table).unwrap(), 0);

        // Five committed inserts → the counter tracks the net delta.
        let load = engine.begin(RC).unwrap();
        let mut tids = Vec::new();
        for i in 0..5i64 {
            tids.push(engine.insert(load, table, &i.to_le_bytes()).unwrap());
        }
        engine.commit(load).unwrap();
        assert_eq!(engine.approx_row_count(table).unwrap(), 5);

        // Two committed deletes → 3.
        let del = engine.begin(RC).unwrap();
        engine.delete(del, table, tids[0]).unwrap();
        engine.delete(del, table, tids[1]).unwrap();
        engine.commit(del).unwrap();
        assert_eq!(engine.approx_row_count(table).unwrap(), 3);

        // An insert then delete of the same row in one transaction nets zero.
        let churn = engine.begin(RC).unwrap();
        let x = engine.insert(churn, table, &[9]).unwrap();
        engine.delete(churn, table, x).unwrap();
        engine.commit(churn).unwrap();
        assert_eq!(engine.approx_row_count(table).unwrap(), 3);

        // A rolled-back insert must NOT skew the counter (the delta is applied on commit only).
        let aborted = engine.begin(RC).unwrap();
        engine.insert(aborted, table, &[7]).unwrap();
        engine.rollback(aborted).unwrap();
        assert_eq!(engine.approx_row_count(table).unwrap(), 3);
    }

    #[test]
    fn churn_since_analyze_counts_absolute_write_ops_and_resets_on_analyze() {
        use nusadb_core::engine::TableStats;

        // The auto-analyze churn tally (D-AUTO-ANALYZE): every committed insert/update/delete counts
        // — absolute, not net, so an update or an insert+delete of the same row still ages the stats
        // — a rolled-back write never counts, and ANALYZE resets it to zero.
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();
        assert_eq!(
            engine.churn_since_analyze(table).unwrap(),
            0,
            "fresh table: no churn"
        );

        // Three committed inserts → churn 3.
        let load = engine.begin(RC).unwrap();
        let mut tids = Vec::new();
        for i in 0..3i64 {
            tids.push(engine.insert(load, table, &i.to_le_bytes()).unwrap());
        }
        engine.commit(load).unwrap();
        assert_eq!(engine.churn_since_analyze(table).unwrap(), 3);

        // An update (+1) and a delete (+1) → 5: updates count even though the row count is flat.
        let mods = engine.begin(RC).unwrap();
        engine.update(mods, table, tids[0], &[42]).unwrap();
        engine.delete(mods, table, tids[1]).unwrap();
        engine.commit(mods).unwrap();
        assert_eq!(engine.churn_since_analyze(table).unwrap(), 5);

        // An insert + delete of the same row in one txn nets zero rows but churns twice (both ops).
        let both = engine.begin(RC).unwrap();
        let x = engine.insert(both, table, &[9]).unwrap();
        engine.delete(both, table, x).unwrap();
        engine.commit(both).unwrap();
        assert_eq!(engine.churn_since_analyze(table).unwrap(), 7);

        // A rolled-back write never counts.
        let aborted = engine.begin(RC).unwrap();
        engine.insert(aborted, table, &[7]).unwrap();
        engine.rollback(aborted).unwrap();
        assert_eq!(engine.churn_since_analyze(table).unwrap(), 7);

        // ANALYZE refreshes the statistics → churn resets to zero, then accrues again afterwards.
        let analyze = engine.begin(RC).unwrap();
        engine
            .analyze_table(
                analyze,
                table,
                &TableStats {
                    row_count: 2,
                    page_count: 1,
                    columns: Vec::new(),
                },
            )
            .unwrap();
        engine.commit(analyze).unwrap();
        assert_eq!(
            engine.churn_since_analyze(table).unwrap(),
            0,
            "ANALYZE resets churn"
        );

        let post = engine.begin(RC).unwrap();
        engine.insert(post, table, &[1]).unwrap();
        engine.commit(post).unwrap();
        assert_eq!(engine.churn_since_analyze(table).unwrap(), 1);

        // The tally is reachable through the `StorageEngine` trait object the SQL layer plans against
        // — the seam a future auto-analyze policy reads (it works with `&dyn StorageEngine`).
        let via_dyn: &dyn nusadb_core::StorageEngine = &engine;
        assert_eq!(via_dyn.churn_since_analyze(table).unwrap(), 1);
    }

    #[test]
    fn approx_row_count_lazily_counts_a_table_written_before_its_first_read() {
        // When rows are committed before the counter is ever read (the post-restart shape, where the
        // in-memory counter is 0 but the tree holds rows), the first read does the one O(n) walk and
        // reflects every committed row — the pre-read commits were skipped while it was uninitialized.
        let engine = BtreeEngine::new();
        let t = engine.begin(RC).unwrap();
        let table = engine.create_table(t, &table_def("t")).unwrap();
        for i in 0..4i64 {
            engine.insert(t, table, &i.to_le_bytes()).unwrap();
        }
        engine.commit(t).unwrap();
        assert_eq!(engine.approx_row_count(table).unwrap(), 4);
    }

    /// (CRITICAL): first-updater-wins holds under `READ COMMITTED` too, not
    /// just `REPEATABLE READ`+. Two overlapping transactions each read a row, one updates+commits,
    /// then the other updates — the second MUST conflict (40001), never silently overwrite (which
    /// would lose the first's update). Concurrent OLTP loses money without this guard.
    #[test]
    fn mvcc_read_committed_prevents_lost_update() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[0]).unwrap();
        engine.commit(setup).unwrap();

        // Both transactions BEGIN while the counter is 0 (overlapping); both would compute 0+1.
        let a = engine.begin(RC).unwrap();
        let b = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, a, table), vec![(tid.page.0, vec![0])]);
        assert_eq!(collect(&engine, b, table), vec![(tid.page.0, vec![0])]);

        // A writes 1 and commits.
        engine.update(a, table, tid, &[1]).unwrap();
        engine.commit(a).unwrap();

        // B, which began before A committed, must NOT be allowed to overwrite with its stale 1 —
        // that is the lost update. It conflicts and retries.
        assert!(
            matches!(
                engine.update(b, table, tid, &[1]),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "a stale write over a concurrently-committed update must conflict, not win"
        );
        engine.rollback(b).unwrap();

        // The retry: a fresh transaction reads the committed 1 and correctly writes 2.
        let retry = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, retry, table), vec![(tid.page.0, vec![1])]);
        engine.update(retry, table, tid, &[2]).unwrap();
        engine.commit(retry).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(tid.page.0, vec![2])]);
        engine.commit(check).unwrap();
    }

    /// The full lost-update pattern under `READ COMMITTED` conserves a total across many
    /// overlapping increments: an OCC retry loop (conflict → re-read → recompute → commit) always
    /// reaches the correct count, exactly as the concurrent-OLTP harness expects.
    #[test]
    fn mvcc_read_committed_increment_loop_conserves_the_total() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[0]).unwrap();
        engine.commit(setup).unwrap();

        // Interleave two "threads" of increments by hand: each round opens both transactions on
        // the same snapshot, so exactly one wins and the other must retry — never a lost update.
        let rounds = 50u8;
        for _ in 0..rounds {
            let a = engine.begin(RC).unwrap();
            let b = engine.begin(RC).unwrap();
            let va = current(&engine, a, table, tid);
            let vb = current(&engine, b, table, tid);
            engine.update(a, table, tid, &[va + 1]).unwrap();
            engine.commit(a).unwrap();
            // b started on the pre-increment snapshot; its write must conflict.
            match engine.update(b, table, tid, &[vb + 1]) {
                Err(nusadb_core::Error::SerializationConflict { .. }) => {
                    engine.rollback(b).unwrap();
                    // OCC retry: fresh read, recompute, commit.
                    let r = engine.begin(RC).unwrap();
                    let vr = current(&engine, r, table, tid);
                    engine.update(r, table, tid, &[vr + 1]).unwrap();
                    engine.commit(r).unwrap();
                },
                Ok(_) => panic!("the stale second write must conflict, not succeed"),
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table),
            vec![(tid.page.0, vec![rounds * 2])],
            "every increment is conserved: {rounds} rounds x 2 = {}",
            rounds * 2
        );
        engine.commit(check).unwrap();
    }

    /// End-to-end: real threads running guarded money transfers with an OCC
    /// retry loop must conserve the total (the exact invariant the concurrent-OLTP harness checks
    /// — before the fix the engine lost updates and the sum drifted below the seeded total).
    #[test]
    fn concurrent_transfers_conserve_money() {
        use std::sync::Arc;

        const ACCOUNTS: u64 = 16;
        const SEED: i64 = 1000;
        const WORKERS: u64 = 6;
        const PER_WORKER: u64 = 400;

        let engine = Arc::new(BtreeEngine::new());
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("ledger")).unwrap();
        let mut tids = Vec::new();
        for _ in 0..ACCOUNTS {
            tids.push(engine.insert(setup, table, &SEED.to_le_bytes()).unwrap());
        }
        engine.commit(setup).unwrap();
        let tids = Arc::new(tids);

        let mut handles = Vec::new();
        for w in 0..WORKERS {
            let engine = Arc::clone(&engine);
            let tids = Arc::clone(&tids);
            handles.push(std::thread::spawn(move || {
                // A cheap deterministic per-worker PRNG (no external dep).
                let mut st = w.wrapping_mul(2_654_435_761).wrapping_add(1);
                let mut next = || {
                    st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    st >> 33
                };
                let read = |txn, tid: nusadb_core::engine::Tid| -> i64 {
                    let mut scan = engine.scan(txn, table).unwrap();
                    while let Some((got, tuple)) = scan.try_next().unwrap() {
                        if got.page.0 == tid.page.0 {
                            return i64::from_le_bytes(
                                tuple.as_ref().try_into().expect("8-byte balance"),
                            );
                        }
                    }
                    panic!("account row not found");
                };
                for _ in 0..PER_WORKER {
                    let a = tids[(next() % ACCOUNTS) as usize];
                    let mut bi = next() % ACCOUNTS;
                    if tids[bi as usize].page.0 == a.page.0 {
                        bi = (bi + 1) % ACCOUNTS;
                    }
                    let b = tids[bi as usize];
                    let amt = 1 + i64::try_from(next() % 20).unwrap_or(0);
                    // Guarded transfer with OCC retry (bounded so the test always terminates).
                    for _ in 0..1000 {
                        let txn = engine.begin(RC).unwrap();
                        let result = (|| -> nusadb_core::Result<bool> {
                            let bal_a = read(txn, a);
                            if bal_a < amt {
                                return Ok(false); // insufficient funds: a no-op commit
                            }
                            engine.update(txn, table, a, &(bal_a - amt).to_le_bytes())?;
                            let bal_b = read(txn, b);
                            engine.update(txn, table, b, &(bal_b + amt).to_le_bytes())?;
                            Ok(true)
                        })();
                        match result {
                            Ok(_) => {
                                engine.commit(txn).unwrap();
                                break;
                            },
                            Err(nusadb_core::Error::SerializationConflict { .. }) => {
                                engine.rollback(txn).unwrap();
                                // retry
                            },
                            Err(e) => {
                                engine.rollback(txn).ok();
                                panic!("unexpected error: {e}");
                            },
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Money is conserved: the sum of all balances still equals the seeded total, and no
        // account went negative.
        let check = engine.begin(RC).unwrap();
        let mut total = 0i64;
        let mut min = i64::MAX;
        for tid in tids.iter() {
            let bytes = collect(&engine, check, table)
                .into_iter()
                .find(|(id, _)| *id == tid.page.0)
                .map(|(_, v)| v)
                .unwrap();
            let bal = i64::from_le_bytes(bytes.try_into().unwrap());
            total += bal;
            min = min.min(bal);
        }
        engine.commit(check).unwrap();
        assert_eq!(
            total,
            SEED * i64::try_from(ACCOUNTS).unwrap_or(0),
            "money must be conserved under concurrent transfers (lost-update guard)"
        );
        assert!(min >= 0, "no account may go negative");
    }

    /// Two `SERIALIZABLE` transactions that each read both rows of
    /// an invariant and then zero their own row must NOT both commit — one aborts (40001), so the
    /// invariant `v1 + v2 >= 1` holds. This is the exact Hermitage `G2` write-skew repro; under
    /// `SERIALIZABLE` it is prevented, under `REPEATABLE READ` it is allowed (snapshot isolation).
    #[test]
    fn serializable_prevents_write_skew() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let r1 = engine.insert(setup, table, &[1]).unwrap();
        let r2 = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // Two SERIALIZABLE transactions, both reading the whole invariant set (sum = 2).
        let a = engine.begin(SER).unwrap();
        let b = engine.begin(SER).unwrap();
        assert_eq!(collect(&engine, a, table).len(), 2);
        assert_eq!(collect(&engine, b, table).len(), 2);

        // Each zeroes its own row (the write-skew shape).
        engine.update(a, table, r1, &[0]).unwrap();
        engine.update(b, table, r2, &[0]).unwrap();

        // A commits first (it wrote r1; b's write to r2 is still uncommitted, so a sees no
        // conflict on its read set).
        engine.commit(a).unwrap();

        // B read r1, which A has now committed a new version of → read-write antidependency →
        // B must abort, not silently commit the invariant violation.
        assert!(
            matches!(
                engine.commit(b),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "the second transaction must abort on the write-skew antidependency"
        );

        // The invariant holds: r1 = 0 (from a), r2 = 1 (b aborted, its zero undone). Sum = 1 >= 1.
        let check = engine.begin(RC).unwrap();
        let mut vals: Vec<u8> = collect(&engine, check, table)
            .into_iter()
            .map(|(_, v)| v[0])
            .collect();
        vals.sort_unstable();
        assert_eq!(
            vals,
            vec![0, 1],
            "the aborted transaction's write was rolled back"
        );
        engine.commit(check).unwrap();
    }

    /// REPEATABLE READ (snapshot isolation) deliberately ALLOWS write-skew — the snapshot-isolation
    /// engine — so the read-set antidependency check must NOT fire below SERIALIZABLE (no
    /// spurious aborts for the common snapshot-isolation workload).
    #[test]
    fn repeatable_read_allows_write_skew() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let r1 = engine.insert(setup, table, &[1]).unwrap();
        let r2 = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        let a = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        let b = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(collect(&engine, a, table).len(), 2);
        assert_eq!(collect(&engine, b, table).len(), 2);
        engine.update(a, table, r1, &[0]).unwrap();
        engine.update(b, table, r2, &[0]).unwrap();
        // Both commit under snapshot isolation — write-skew is permitted at RR.
        engine.commit(a).unwrap();
        engine.commit(b).unwrap();
        let check = engine.begin(RC).unwrap();
        let sum: u32 = collect(&engine, check, table)
            .into_iter()
            .map(|(_, v)| u32::from(v[0]))
            .sum();
        assert_eq!(sum, 0, "RR permits the write-skew (both zeroed) by design");
        engine.commit(check).unwrap();
    }

    /// Masked-writer case: an in-flight (still-active) concurrent
    /// write stacked ON TOP of a concurrent-committed write to a read row must NOT hide the
    /// committed one. The antidependency check walks the version chain, so the write-skew is still
    /// caught even when a third transaction's uncommitted update is the newest version.
    #[test]
    fn serializable_write_skew_not_masked_by_active_writer() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let r1 = engine.insert(setup, table, &[1]).unwrap();
        let r2 = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // A and B overlap on the invariant {r1, r2} (both read sum = 2).
        let a = engine.begin(SER).unwrap();
        let b = engine.begin(SER).unwrap();
        assert_eq!(collect(&engine, a, table).len(), 2);
        assert_eq!(collect(&engine, b, table).len(), 2);

        // A zeroes r1 and COMMITS (a's version of r1 is now committed).
        engine.update(a, table, r1, &[0]).unwrap();
        engine.commit(a).unwrap();

        // A third transaction Y writes r1 again and stays ACTIVE — so the NEWEST version of r1 is
        // Y's uncommitted write, stacked on top of A's committed write.
        let y = engine.begin(RC).unwrap();
        engine.update(y, table, r1, &[7]).unwrap();

        // B zeroes r2 and commits. B read r1; the newest version of r1 is Y's (active, not a
        // conflict yet) but A's committed version sits beneath it → the chain walk finds A's
        // antidependency → B must abort. A top-version-only check would miss it (the bug).
        engine.update(b, table, r2, &[0]).unwrap();
        assert!(
            matches!(
                engine.commit(b),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "an active writer on top of a committed one must not mask the write-skew"
        );
        // The failed commit already rolled `b` back (SSI abort-at-commit); `y` is a
        // separate still-active transaction.
        engine.rollback(y).unwrap();

        // Invariant intact: r1 = 0 (a), r2 = 1 (b aborted). Sum = 1 >= 1.
        let check = engine.begin(RC).unwrap();
        let mut vals: Vec<u8> = collect(&engine, check, table)
            .into_iter()
            .map(|(_, v)| v[0])
            .collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![0, 1]);
        engine.commit(check).unwrap();
    }

    /// No spurious aborts: a SERIALIZABLE transaction commits cleanly when nothing concurrent
    /// touched its read set — a lone read+write, and two transactions that do not overlap (the
    /// second begins after the first commits, so its snapshot already sees that write).
    #[test]
    fn serializable_commits_without_spurious_conflict() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let r1 = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // Lone SERIALIZABLE read+write, no concurrency → commits.
        let a = engine.begin(SER).unwrap();
        assert_eq!(collect(&engine, a, table).len(), 1);
        engine.update(a, table, r1, &[2]).unwrap();
        engine.commit(a).unwrap();

        // Non-overlapping: b begins AFTER a committed, so b's snapshot sees a's write → no
        // antidependency → b commits.
        let b = engine.begin(SER).unwrap();
        assert_eq!(collect(&engine, b, table), vec![(r1.page.0, vec![2])]);
        engine.update(b, table, r1, &[3]).unwrap();
        engine.commit(b).unwrap();

        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(r1.page.0, vec![3])]);
        engine.commit(check).unwrap();
    }

    /// Recovery corner: a SERIALIZABLE transaction that both
    /// creates a sequence AND is aborted by the commit-time antidependency check must neutralize
    /// that sequence exactly like an explicit ROLLBACK — the commit-abort and rollback paths share
    /// The read-only commit fast path: a transaction whose `undo` is empty at commit —
    /// genuinely read-only, or written-then-fully-unwound via `ROLLBACK TO SAVEPOINT` — skips
    /// the `CommitTxn` marker and its fsync. Its WAL records (if any) have no marker, so replay
    /// must discard them: reopen sees exactly the durably committed rows, nothing more or less,
    /// and the live engine agrees before and after.
    #[test]
    fn empty_undo_commit_skips_marker_and_replays_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            engine.insert(setup, table, &[1]).unwrap();
            engine.commit(setup).unwrap();

            // Genuinely read-only transaction: sees the row, commits via the fast path.
            let reader = engine.begin(RC).unwrap();
            assert_eq!(collect(&engine, reader, table).len(), 1);
            engine.commit(reader).unwrap();

            // Written-then-unwound: the insert's WAL record exists, but ROLLBACK TO SAVEPOINT
            // empties the undo (appending compensations), so the commit also takes the fast
            // path — and the unwound row must stay invisible.
            let unwound = engine.begin(RC).unwrap();
            engine.savepoint(unwound, "s").unwrap();
            engine.insert(unwound, table, &[2]).unwrap();
            engine.rollback_to(unwound, "s").unwrap();
            engine.commit(unwound).unwrap();

            let check = engine.begin(RC).unwrap();
            assert_eq!(collect(&engine, check, table).len(), 1);
            engine.commit(check).unwrap();
        };

        // Replay: the unmarked records (insert + compensation) are discarded; exactly the one
        // durably committed row survives.
        let engine = BtreeEngine::open(&path).unwrap();
        let txn = engine.begin(RC).unwrap();
        let table = engine.lookup_table("t").unwrap().unwrap().id;
        let rows = collect(&engine, txn, table);
        assert_eq!(rows.len(), 1, "replay must keep exactly the committed row");
        assert_eq!(rows[0].1, vec![1]);
        engine.commit(txn).unwrap();
    }

    /// `abort_locked`. Otherwise the non-transactional `SeqCreate` record would replay a phantom
    /// sequence on recovery despite the transaction aborting.
    #[test]
    fn serializable_abort_with_sequence_create_leaves_no_phantom() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            let r1 = engine.insert(setup, table, &[1]).unwrap();
            let r2 = engine.insert(setup, table, &[1]).unwrap();
            engine.commit(setup).unwrap();

            // Two SERIALIZABLE transactions overlapping on the invariant {r1, r2}.
            let a = engine.begin(SER).unwrap();
            let b = engine.begin(SER).unwrap();
            assert_eq!(collect(&engine, a, table).len(), 2);
            assert_eq!(collect(&engine, b, table).len(), 2);

            // A zeroes r1 and commits first.
            engine.update(a, table, r1, &[0]).unwrap();
            engine.commit(a).unwrap();

            // B creates a sequence, then zeroes r2 — the write-skew shape plus a non-transactional
            // side effect. B read r1 (which A has since committed) → its commit must abort.
            engine.create_sequence(b, &seq_def("ghost")).unwrap();
            engine.update(b, table, r2, &[0]).unwrap();
            assert!(
                matches!(
                    engine.commit(b),
                    Err(nusadb_core::Error::SerializationConflict { .. })
                ),
                "the write-skew antidependency must abort B"
            );
            // The aborted transaction's sequence is gone in the live engine.
            assert_eq!(engine.lookup_sequence("ghost").unwrap(), None);
        };

        // And stays gone after crash recovery — the abort neutralized the SeqCreate record.
        let engine = BtreeEngine::open(&path).unwrap();
        assert_eq!(
            engine.lookup_sequence("ghost").unwrap(),
            None,
            "an aborted SERIALIZABLE transaction must not resurrect its sequence on replay"
        );
    }

    /// (btree treaty completion, L1): two transactions requesting the same uniqueness-key
    /// lock — the second aborts immediately (no-wait 40001, never blocks); a different key does
    /// not conflict, and the key frees when the holder ends.
    #[test]
    fn lock_key_is_no_wait_and_released_at_txn_end() {
        use nusadb_core::engine::RowLockMode;
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let a = engine.begin(RC).unwrap();
        let b = engine.begin(RC).unwrap();
        engine
            .lock_key(a, table, 42, RowLockMode::Exclusive)
            .unwrap();
        assert!(
            matches!(
                engine.lock_key(b, table, 42, RowLockMode::Exclusive),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "the second same-key writer must abort at lock time, not block"
        );
        engine
            .lock_key(b, table, 43, RowLockMode::Exclusive)
            .unwrap();
        engine.commit(a).unwrap();
        engine
            .lock_key(b, table, 42, RowLockMode::Exclusive)
            .unwrap();
        engine.commit(b).unwrap();
    }

    /// A rolled-back transaction releases its locks and leaves the active set, so a conflicting
    /// transaction can proceed and purge is never pinned by it (`abort`'s
    /// in-memory teardown — remove from `active`, drop locks — must always run, so a transaction
    /// can never leak on the rollback path). `has_open_transaction` reports engine-visible liveness.
    #[test]
    fn rollback_releases_locks_and_ends_the_transaction() {
        use nusadb_core::engine::RowLockMode;
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let a = engine.begin(RC).unwrap();
        engine
            .lock_key(a, table, 42, RowLockMode::Exclusive)
            .unwrap();
        // Roll `a` back: its lock on key 42 must be released and the transaction must end.
        engine.rollback(a).unwrap();
        // `a` is fully gone: a further op under it is an unknown-transaction error (it would still
        // be known if it had leaked in `active`/`txns`).
        assert!(
            engine
                .lock_key(a, table, 99, RowLockMode::Exclusive)
                .is_err(),
            "a rolled-back transaction must no longer be usable (else it pins purge forever)"
        );

        // A fresh transaction can now take the exact lock `a` held — proof the lock was released.
        let b = engine.begin(RC).unwrap();
        engine
            .lock_key(b, table, 42, RowLockMode::Exclusive)
            .expect("the rolled-back transaction's lock must be free");
        engine.commit(b).unwrap();
    }

    /// Under READ COMMITTED every read within ONE statement observes a
    /// single consistent snapshot, so a concurrent transfer committing between two reads of the
    /// same statement cannot make money appear. Previously each read took a fresh snapshot, so a
    /// two-table statement could read one table before and the other after the transfer.
    #[test]
    fn read_committed_uses_one_snapshot_per_statement() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("acct")).unwrap();
        let a = engine.insert(setup, table, &[10]).unwrap(); // balance 10
        let b = engine.insert(setup, table, &[0]).unwrap(); // balance 0
        engine.commit(setup).unwrap();

        // Reader R's statement snapshot is fixed at BEGIN (no `begin_statement` called in between).
        let r = engine.begin(RC).unwrap();
        let a_seen = current(&engine, r, table, a);
        // A concurrent transfer of 10 from a to b commits BETWEEN R's two reads.
        let w = engine.begin(RC).unwrap();
        engine.update(w, table, a, &[0]).unwrap();
        engine.update(w, table, b, &[10]).unwrap();
        engine.commit(w).unwrap();
        // R's second read (same statement) must use the SAME snapshot — the transfer is invisible,
        // so the total is conserved (no money created from air).
        let b_seen = current(&engine, r, table, b);
        assert_eq!(a_seen, 10, "R's first read");
        assert_eq!(
            b_seen, 0,
            "R's second read must be from the same statement snapshot, not see the transfer"
        );
        assert_eq!(
            u32::from(a_seen) + u32::from(b_seen),
            10,
            "the total must be conserved within one statement"
        );
        engine.commit(r).unwrap();
    }

    /// (the freshness half): a NEW statement under READ COMMITTED sees
    /// transactions that committed since the previous statement — the snapshot is refreshed at
    /// `begin_statement`, so RC is not frozen at BEGIN like REPEATABLE READ.
    #[test]
    fn read_committed_sees_intervening_commits_at_a_new_statement() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let x = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        let r = engine.begin(RC).unwrap();
        assert_eq!(current(&engine, r, table, x), 1, "statement 1 sees 1");
        // A concurrent commit changes x to 2.
        let w = engine.begin(RC).unwrap();
        engine.update(w, table, x, &[2]).unwrap();
        engine.commit(w).unwrap();
        // Still statement 1 (no refresh): R keeps its snapshot value.
        assert_eq!(
            current(&engine, r, table, x),
            1,
            "same statement keeps its snapshot"
        );
        // A NEW statement refreshes the RC snapshot → R now sees the committed 2.
        engine.begin_statement(r).unwrap();
        assert_eq!(
            current(&engine, r, table, x),
            2,
            "a new statement sees the intervening commit (standard READ COMMITTED)"
        );
        engine.commit(r).unwrap();
    }

    /// REPEATABLE READ ignores `begin_statement`: its snapshot stays pinned at BEGIN, so an
    /// intervening commit is never visible even after a new statement.
    #[test]
    fn repeatable_read_snapshot_survives_begin_statement() {
        use nusadb_core::IsolationLevel::RepeatableRead;
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let x = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        let r = engine.begin(RepeatableRead).unwrap();
        assert_eq!(current(&engine, r, table, x), 1);
        let w = engine.begin(RC).unwrap();
        engine.update(w, table, x, &[2]).unwrap();
        engine.commit(w).unwrap();
        engine.begin_statement(r).unwrap(); // no-op for REPEATABLE READ
        assert_eq!(
            current(&engine, r, table, x),
            1,
            "REPEATABLE READ stays pinned at BEGIN across statements"
        );
        engine.commit(r).unwrap();
    }

    /// (btree treaty completion, L1): `LOCK TABLE ACCESS EXCLUSIVE` excludes concurrent
    /// row writes (their shared intention conflicts) and vice versa.
    #[test]
    fn lock_table_access_exclusive_excludes_writers() {
        use nusadb_core::engine::TableLockMode;
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        let a = engine.begin(RC).unwrap();
        engine
            .lock_table(a, table, TableLockMode::AccessExclusive)
            .unwrap();
        let b = engine.begin(RC).unwrap();
        assert!(
            matches!(
                engine.insert(b, table, &[1]),
                Err(nusadb_core::Error::SerializationConflict { .. })
            ),
            "a row write under a concurrent ACCESS EXCLUSIVE must conflict"
        );
        engine.commit(a).unwrap();
        engine.insert(b, table, &[1]).unwrap();
        // The write's intention now blocks a new ACCESS EXCLUSIVE until b ends.
        let c = engine.begin(RC).unwrap();
        assert!(matches!(
            engine.lock_table(c, table, TableLockMode::AccessExclusive),
            Err(nusadb_core::Error::SerializationConflict { .. })
        ));
        engine.rollback(c).unwrap();
        engine.commit(b).unwrap();
    }

    /// Index-entry MVCC (btree treaty completion, L1): an `UPDATE` that moves a row to a new key
    /// leaves exactly ONE visible entry per reader — the new key for fresh snapshots, the old key
    /// for a snapshot pinned before the move — never both. The row keeps its address across
    /// versions, so only the entry stamps can witness the key move.
    #[test]
    fn index_key_move_shows_one_entry_per_snapshot() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        let idx = engine
            .create_index(txn, &index_def("i", table, false))
            .unwrap();
        let tid = engine.insert(txn, table, &[30]).unwrap();
        engine.index_insert(txn, idx, &[30], tid).unwrap();
        engine.commit(txn).unwrap();

        // Pin an RR snapshot BEFORE the key move.
        let old_reader = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(
            collect_index(&engine, old_reader, idx, Bound::Unbounded, Bound::Unbounded).len(),
            1
        );

        let updater = engine.begin(RC).unwrap();
        engine.update(updater, table, tid, &[99]).unwrap();
        engine.index_insert(updater, idx, &[99], tid).unwrap();
        engine.commit(updater).unwrap();

        // A fresh reader sees exactly one entry, under the NEW key.
        let new_reader = engine.begin(RC).unwrap();
        let entries = collect_index(&engine, new_reader, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(
            entries.len(),
            1,
            "UPDATE must move, not duplicate, the entry"
        );
        assert_eq!(entries[0].1, vec![99]);
        assert!(
            collect_index(
                &engine,
                new_reader,
                idx,
                Bound::Included(vec![30]),
                Bound::Included(vec![30])
            )
            .is_empty(),
            "the old key must resolve to nothing for a fresh reader"
        );
        engine.commit(new_reader).unwrap();

        // The pinned snapshot still finds the row under its OLD key only, with its old version.
        let old_entries =
            collect_index(&engine, old_reader, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(old_entries.len(), 1);
        assert_eq!(old_entries[0].1, vec![30]);
        engine.commit(old_reader).unwrap();
    }

    /// Rolling back a key-moving transaction revives the old entry: the dead-stamp its insert
    /// placed is undone together with the new entry — live and after WAL replay.
    #[test]
    fn index_key_move_rollback_revives_the_old_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let table;
        let idx;
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            table = engine.create_table(setup, &table_def("t")).unwrap();
            idx = engine
                .create_index(setup, &index_def("i", table, false))
                .unwrap();
            let tid = engine.insert(setup, table, &[30]).unwrap();
            engine.index_insert(setup, idx, &[30], tid).unwrap();
            engine.commit(setup).unwrap();

            // Move the key under a savepoint, then roll the move back (exercises the logged
            // compensations, incl. the unstamp record), and roll back a second whole-txn move.
            let a = engine.begin(RC).unwrap();
            engine.savepoint(a, "sp").unwrap();
            engine.update(a, table, tid, &[99]).unwrap();
            engine.index_insert(a, idx, &[99], tid).unwrap();
            engine.rollback_to(a, "sp").unwrap();
            engine.commit(a).unwrap();

            let b = engine.begin(RC).unwrap();
            engine.update(b, table, tid, &[77]).unwrap();
            engine.index_insert(b, idx, &[77], tid).unwrap();
            engine.rollback(b).unwrap();

            let check = engine.begin(RC).unwrap();
            let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
            assert_eq!(entries.len(), 1, "both rolled-back moves left one entry");
            assert_eq!(entries[0].1, vec![30], "the original key is alive again");
            engine.commit(check).unwrap();
        }
        // The same holds after crash recovery (compensations replayed).
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, vec![30]);
        engine.commit(check).unwrap();
    }

    /// A committed key move survives reopen (replay re-derives the dead-stamp through the shared
    /// apply path), and purge reclaims the settled dead entry.
    #[test]
    fn index_key_move_survives_reopen_and_purges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let table;
        let idx;
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            table = engine.create_table(setup, &table_def("t")).unwrap();
            idx = engine
                .create_index(setup, &index_def("i", table, false))
                .unwrap();
            let tid = engine.insert(setup, table, &[30]).unwrap();
            engine.index_insert(setup, idx, &[30], tid).unwrap();
            engine.commit(setup).unwrap();
            let a = engine.begin(RC).unwrap();
            engine.update(a, table, tid, &[99]).unwrap();
            engine.index_insert(a, idx, &[99], tid).unwrap();
            engine.commit(a).unwrap();
        }
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(entries.len(), 1, "replay re-derives the dead-stamp");
        assert_eq!(entries[0].1, vec![99]);
        engine.commit(check).unwrap();
        // With every snapshot settled, purge physically reclaims the dead-stamped old entry.
        let stats = engine.purge().unwrap();
        assert!(
            stats.index_entries_removed >= 1,
            "the settled dead entry is reclaimed, got {stats:?}"
        );
        let check = engine.begin(RC).unwrap();
        let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(entries.len(), 1, "purge must not change visibility");
        engine.commit(check).unwrap();
    }

    /// (QA CRITICAL, 2026-07-09): an UPDATE that does NOT
    /// move the indexed key re-inserts the same key (the SQL layer re-inserts every index entry
    /// on UPDATE). That re-insert must be a no-op: the committed entry keeps serving concurrent
    /// readers while the update is uncommitted, and a ROLLBACK must leave it untouched — the
    /// first draft overwrote it and rollback then stripped it, making the committed row vanish
    /// from every index read path.
    #[test]
    fn same_key_update_rollback_keeps_the_index_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let table;
        let idx;
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            table = engine.create_table(setup, &table_def("t")).unwrap();
            idx = engine
                .create_index(setup, &index_def("i", table, false))
                .unwrap();
            let tid = engine.insert(setup, table, &[7]).unwrap();
            engine.index_insert(setup, idx, &[7], tid).unwrap();
            engine.commit(setup).unwrap();

            // The updater changes the VALUE but not the key: heap update + same-key re-insert.
            let updater = engine.begin(RC).unwrap();
            engine.update(updater, table, tid, &[7]).unwrap();
            engine.index_insert(updater, idx, &[7], tid).unwrap();

            // A concurrent reader must still find the row via the index while the update is
            // uncommitted (QA fact #1: this failed BEFORE the rollback too).
            let reader = engine.begin(RC).unwrap();
            assert_eq!(
                collect_index(&engine, reader, idx, Bound::Unbounded, Bound::Unbounded).len(),
                1,
                "a concurrent reader must see the committed row through the index"
            );
            engine.commit(reader).unwrap();

            engine.rollback(updater).unwrap();

            // After the rollback the committed row is still reachable via the index.
            let check = engine.begin(RC).unwrap();
            let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
            assert_eq!(entries.len(), 1, "ROLLBACK must not strip the entry");
            assert_eq!(entries[0].1, vec![7]);
            engine.commit(check).unwrap();

            // Same shape through ROLLBACK TO SAVEPOINT (the compensation path).
            let sp = engine.begin(RC).unwrap();
            engine.savepoint(sp, "s").unwrap();
            engine.update(sp, table, tid, &[7]).unwrap();
            engine.index_insert(sp, idx, &[7], tid).unwrap();
            engine.rollback_to(sp, "s").unwrap();
            engine.commit(sp).unwrap();
            let check = engine.begin(RC).unwrap();
            assert_eq!(
                collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded).len(),
                1,
                "ROLLBACK TO SAVEPOINT must not strip the entry either"
            );
            engine.commit(check).unwrap();
        }
        // And the entry survives crash recovery.
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        let entries = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, vec![7]);
        engine.commit(check).unwrap();
    }

    /// Group commit: concurrent committers on the DURABLE engine share fsyncs — nothing may
    /// be lost. Every committed row is visible live and survives reopen (the WAL replay is the
    /// witness that each commit marker really became durable before the commit returned).
    #[test]
    fn group_commit_concurrent_committers_all_durable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let table;
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            table = engine.create_table(setup, &table_def("t")).unwrap();
            engine.commit(setup).unwrap();
            std::thread::scope(|scope| {
                for w in 0..6u8 {
                    let engine = &engine;
                    scope.spawn(move || {
                        for i in 0..30u8 {
                            let txn = engine.begin(RC).unwrap();
                            engine.insert(txn, table, &[w, i]).unwrap();
                            engine.commit(txn).unwrap();
                        }
                    });
                }
            });
            let check = engine.begin(RC).unwrap();
            assert_eq!(collect(&engine, check, table).len(), 180);
            engine.commit(check).unwrap();
        }
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table).len(),
            180,
            "every group-committed row survives reopen"
        );
        engine.commit(check).unwrap();
    }

    /// Under SERIALIZABLE contention on the DURABLE engine: racing increments with the
    /// abort-and-retry discipline must conserve the total — the staged-commit window (marker
    /// appended, fsync pending) must never let a concurrent committer slip an antidependency
    /// through or observe a not-yet-durable write.
    #[test]
    fn group_commit_serializable_increments_conserve_the_total() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let engine = BtreeEngine::open(&path).unwrap();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[0]).unwrap();
        engine.commit(setup).unwrap();

        let workers = 4u8;
        let per_worker = 25u8;
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let engine = &engine;
                scope.spawn(move || {
                    for _ in 0..per_worker {
                        // Abort-and-retry until this increment lands (no-wait 40001s expected).
                        loop {
                            let Ok(txn) = engine.begin(SER) else {
                                continue;
                            };
                            let attempt = (|| -> nusadb_core::Result<()> {
                                let current =
                                    collect(engine, txn, table).first().map_or(0, |(_, v)| v[0]);
                                engine.update(txn, table, tid, &[current + 1])?;
                                engine.commit(txn)
                            })();
                            match attempt {
                                Ok(()) => break,
                                Err(_) => {
                                    // The failed commit already rolled the txn back on a
                                    // conflict; a pre-commit error still needs the rollback.
                                    let _ = engine.rollback(txn);
                                },
                            }
                        }
                    }
                });
            }
        });

        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table);
        assert_eq!(
            rows.first().map(|(_, v)| v[0]),
            Some(workers * per_worker),
            "every serializable increment must be conserved under group commit"
        );
        engine.commit(check).unwrap();
    }

    /// Blast-radius pin: a BULK same-key `UPDATE` that aborts must leave every touched row's
    /// index entry intact — the reported failure wiped the whole updated set from index reads
    /// (batch job aborted, migration rolled back).
    #[test]
    fn bulk_update_rollback_keeps_every_index_entry() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let idx = engine
            .create_index(setup, &index_def("i", table, false))
            .unwrap();
        let mut tids = Vec::new();
        for i in 0..50u8 {
            let tid = engine.insert(setup, table, &[i]).unwrap();
            engine.index_insert(setup, idx, &[i], tid).unwrap();
            tids.push((i, tid));
        }
        engine.commit(setup).unwrap();

        // One transaction updates EVERY row (values change, keys do not), then aborts.
        let bulk = engine.begin(RC).unwrap();
        for &(key, tid) in &tids {
            engine.update(bulk, table, tid, &[key]).unwrap();
            engine.index_insert(bulk, idx, &[key], tid).unwrap();
        }
        engine.rollback(bulk).unwrap();

        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded).len(),
            50,
            "an aborted bulk update must not strip any index entry"
        );
        engine.commit(check).unwrap();
    }

    /// A key that moves away and later moves BACK gets a second visibility range under its
    /// original key: a snapshot pinned before the first move and a fresh reader each resolve the
    /// row under the key their version carries — neither loses it, neither double-counts.
    #[test]
    fn index_key_reuse_serves_pinned_and_fresh_snapshots() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let idx = engine
            .create_index(setup, &index_def("i", table, false))
            .unwrap();
        let tid = engine.insert(setup, table, &[7]).unwrap();
        engine.index_insert(setup, idx, &[7], tid).unwrap();
        engine.commit(setup).unwrap();

        // Pin a snapshot while the key is still 7.
        let pinned = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(
            collect_index(&engine, pinned, idx, Bound::Unbounded, Bound::Unbounded).len(),
            1
        );

        // Move 7 → 9, commit; then 9 → 7 again, commit.
        let a = engine.begin(RC).unwrap();
        engine.update(a, table, tid, &[9]).unwrap();
        engine.index_insert(a, idx, &[9], tid).unwrap();
        engine.commit(a).unwrap();
        let b = engine.begin(RC).unwrap();
        engine.update(b, table, tid, &[7]).unwrap();
        engine.index_insert(b, idx, &[7], tid).unwrap();
        engine.commit(b).unwrap();

        // The pinned snapshot still finds its version under 7 (its original range), exactly once.
        let pinned_rows = collect_index(&engine, pinned, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(pinned_rows.len(), 1, "the pinned snapshot keeps its row");
        assert_eq!(pinned_rows[0].1, vec![7]);
        engine.commit(pinned).unwrap();

        // A fresh reader finds the row under 7 (the re-used key), exactly once — and nothing
        // under 9.
        let fresh = engine.begin(RC).unwrap();
        let rows = collect_index(&engine, fresh, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(rows.len(), 1, "one entry after the key returned home");
        assert_eq!(rows[0].1, vec![7]);
        assert!(
            collect_index(
                &engine,
                fresh,
                idx,
                Bound::Included(vec![9]),
                Bound::Included(vec![9])
            )
            .is_empty()
        );
        engine.commit(fresh).unwrap();
    }

    /// RELEASE SAVEPOINT forgets the marker (keeping all writes), so a later ROLLBACK TO that
    /// name fails; releasing an unknown name fails loudly.
    #[test]
    fn release_savepoint_forgets_the_marker() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        engine.savepoint(txn, "sp").unwrap();
        engine.insert(txn, table, &[1]).unwrap();
        assert!(engine.release_savepoint(txn, "ghost").is_err());
        engine.release_savepoint(txn, "sp").unwrap();
        assert!(
            engine.rollback_to(txn, "sp").is_err(),
            "a released savepoint is gone"
        );
        engine.commit(txn).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table).len(),
            1,
            "RELEASE keeps the writes"
        );
        engine.commit(check).unwrap();
    }

    /// Follow-up (pre-existing leak): an aborted `UPDATE` disconnects the arena slot
    /// it parked the superseded version in — the rollback restores the pre-update leaf, whose
    /// chain pointer predates the slot — and nothing ever freed it. Purge must reclaim the
    /// orphan once the abort settles, and must NOT touch it while a view concurrent with the
    /// aborting transaction is still pinned (that reader may be mid-walk through the slot).
    #[test]
    fn aborted_update_arena_slot_is_reclaimed_once_settled() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();

        // A snapshot pinned BEFORE the aborted update: the orphan must survive it.
        let pinned = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(collect(&engine, pinned, table).len(), 1);

        let aborted = engine.begin(RC).unwrap();
        engine.update(aborted, table, tid, &[2]).unwrap();
        engine.rollback(aborted).unwrap();

        // The abort is not settled while the concurrent snapshot lives: nothing reclaimed.
        assert_eq!(
            engine.purge().unwrap().versions_reclaimed,
            0,
            "the orphan slot must survive a view concurrent with the abort"
        );
        engine.commit(pinned).unwrap();

        // Settled: exactly the orphan slot is reclaimed, and a second pass finds nothing.
        let stats = engine.purge().unwrap();
        assert_eq!(
            stats.versions_reclaimed, 1,
            "the aborted update's parked slot is reclaimed, got {stats:?}"
        );
        assert_eq!(engine.purge().unwrap().versions_reclaimed, 0);

        // The row itself is untouched by the reclamation.
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(tid.page.0, vec![1])]);
        engine.commit(check).unwrap();
    }

    /// The treaty `vacuum` maps to a purge pass and reports reclaimed versions.
    #[test]
    fn vacuum_reports_reclaimed_versions() {
        use nusadb_core::StorageEngine as _;
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.commit(setup).unwrap();
        let a = engine.begin(RC).unwrap();
        engine.update(a, table, tid, &[2]).unwrap();
        engine.commit(a).unwrap();
        assert!(
            engine.vacuum().unwrap() >= 1,
            "the settled superseded version is reclaimed"
        );
    }

    /// `list_tables` mirrors the catalog: sorted names, drops disappear.
    #[test]
    fn list_tables_reports_the_catalog() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let beta = engine.create_table(txn, &table_def("beta")).unwrap();
        engine.create_table(txn, &table_def("alpha")).unwrap();
        engine.commit(txn).unwrap();
        assert_eq!(
            engine.list_tables().unwrap(),
            vec!["alpha".to_owned(), "beta".to_owned()]
        );
        let txn = engine.begin(RC).unwrap();
        engine.drop_table(txn, beta).unwrap();
        engine.commit(txn).unwrap();
        assert_eq!(engine.list_tables().unwrap(), vec!["alpha".to_owned()]);
    }

    /// An oversized tuple is refused loudly (single-leaf capacity; overflow pages are a later
    /// phase) — never truncated or silently dropped.
    #[test]
    fn oversized_tuple_is_refused() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        let err = engine
            .insert(txn, table, &vec![0u8; nusadb_core::PAGE_SIZE])
            .expect_err("a page-sized tuple cannot fit a leaf");
        assert!(err.to_string().contains("exceeds the single-leaf capacity"));
        engine.rollback(txn).unwrap();
    }

    /// Crash safety, the core gate: committed transactions survive a "crash" (dropping the
    /// engine without any shutdown) — rows, updates, deletes, DDL, and Tid identity all come
    /// back on reopen, and the reopened engine keeps working (new writes get fresh row-ids).
    #[test]
    fn e3_committed_transactions_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, tids) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let table = engine.create_table(txn, &table_def("t")).unwrap();
            let mut tids = Vec::new();
            for i in 0..50u8 {
                tids.push(engine.insert(txn, table, &[i]).unwrap());
            }
            engine.commit(txn).unwrap();

            // A second committed txn: update + delete must replay too.
            let txn = engine.begin(RC).unwrap();
            engine.update(txn, table, tids[7], &[200]).unwrap();
            engine.delete(txn, table, tids[8]).unwrap();
            engine.commit(txn).unwrap();
            (table, tids)
        }; // "crash": engine dropped, no shutdown hook.

        let engine = BtreeEngine::open(&path).unwrap();
        let table = engine
            .lookup_table("t")
            .unwrap()
            .expect("committed DDL survives");
        assert_eq!(table.id, table_id, "table id is stable across recovery");
        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table.id);
        assert_eq!(rows.len(), 49, "50 inserts minus 1 committed delete");
        assert!(
            rows.iter()
                .any(|(id, t)| *id == tids[7].page.0 && t == &vec![200]),
            "committed update replayed at the same Tid"
        );
        assert!(rows.iter().all(|(id, _)| *id != tids[8].page.0));
        // The reopened engine keeps minting fresh row-ids past the recovered ones.
        let new_tid = engine.insert(check, table.id, &[99]).unwrap();
        assert!(
            new_tid.page.0 > tids[49].page.0,
            "row-ids continue, not reused"
        );
        engine.commit(check).unwrap();
    }

    /// A transaction with no commit marker in the log — a crash mid-transaction — is fully
    /// invisible after recovery, even though its op records are in the durable prefix.
    #[test]
    fn e3_uncommitted_transactions_are_lost_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, keep) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            let keep = engine.insert(setup, table, &[1]).unwrap();
            engine.commit(setup).unwrap();

            // Crash with this txn in flight: its insert/update/delete must all vanish.
            let doomed = engine.begin(RC).unwrap();
            engine.insert(doomed, table, &[2]).unwrap();
            engine.update(doomed, table, keep, &[9]).unwrap();
            engine.delete(doomed, table, keep).unwrap();
            // Also an uncommitted CREATE TABLE.
            engine.create_table(doomed, &table_def("ghost")).unwrap();
            (table, keep)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        assert!(engine.lookup_table("ghost").unwrap().is_none());
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table_id),
            vec![(keep.page.0, vec![1])],
            "only the committed state survives"
        );
        engine.commit(check).unwrap();
    }

    /// Savepoint compensation: a partial ROLLBACK TO inside a later-COMMITTED transaction
    /// must replay to the post-rollback state — the undone ops' records stay in the log, so the
    /// logged compensations are what makes recovery converge.
    #[test]
    fn e3_savepoint_compensations_replay_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, kept, updated) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            let updated = engine.insert(setup, table, &[10]).unwrap();
            engine.commit(setup).unwrap();

            let txn = engine.begin(RC).unwrap();
            let kept = engine.insert(txn, table, &[3]).unwrap();
            engine.savepoint(txn, "sp").unwrap();
            engine.insert(txn, table, &[4]).unwrap(); // undone
            engine.update(txn, table, updated, &[20]).unwrap(); // undone
            engine.delete(txn, table, kept).unwrap(); // undone
            engine.rollback_to(txn, "sp").unwrap();
            engine.update(txn, table, updated, &[30]).unwrap(); // survives
            engine.commit(txn).unwrap();
            (table, kept, updated)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        let mut rows = collect(&engine, check, table_id);
        rows.sort_by_key(|(id, _)| *id);
        assert_eq!(
            rows,
            vec![(updated.page.0, vec![30]), (kept.page.0, vec![3])],
            "replay converges to the post-rollback, post-commit state"
        );
        engine.commit(check).unwrap();
    }

    /// Torn-tail: garbage appended after the durable prefix (a torn last write) is detected,
    /// the file is truncated to the good prefix on open, and — critically — writes committed
    /// AFTER the recovery are readable by the NEXT recovery (nothing stranded behind garbage).
    #[test]
    fn e3_torn_tail_is_truncated_and_log_stays_appendable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let table_id = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let table = engine.create_table(txn, &table_def("t")).unwrap();
            engine.insert(txn, table, &[1]).unwrap();
            engine.commit(txn).unwrap();
            table
        };

        // Tear the tail: half-written garbage after the last durable record.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&[0xFF; 137]).unwrap();
        }
        let torn_len = std::fs::metadata(&path).unwrap().len();

        // First reopen: recovers the prefix, truncates the garbage, stays writable.
        {
            let engine = BtreeEngine::open(&path).unwrap();
            assert!(
                std::fs::metadata(&path).unwrap().len() < torn_len,
                "the torn tail was truncated"
            );
            let check = engine.begin(RC).unwrap();
            assert_eq!(collect(&engine, check, table_id).len(), 1);
            engine.insert(check, table_id, &[2]).unwrap();
            engine.commit(check).unwrap();
        }

        // Second reopen: the post-truncation commit is durable (it was not stranded).
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table_id).len(),
            2,
            "writes after a torn-tail recovery survive the next recovery"
        );
        engine.commit(check).unwrap();
    }

    /// DDL replay: a committed DROP TABLE stays dropped after recovery, and a rolled-back
    /// (aborted) transaction's DDL leaves no trace — plus rollback works identically on the
    /// durable engine.
    #[test]
    fn e3_ddl_and_rollback_replay_on_durable_engine() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let t1 = engine.create_table(txn, &table_def("gone")).unwrap();
            engine.insert(txn, t1, &[1]).unwrap();
            engine.create_table(txn, &table_def("stays")).unwrap();
            engine.commit(txn).unwrap();

            let txn = engine.begin(RC).unwrap();
            engine.drop_table(txn, t1).unwrap();
            engine.commit(txn).unwrap();

            // An explicitly rolled-back txn on the durable engine: no residue in memory or log.
            let txn = engine.begin(RC).unwrap();
            engine.create_table(txn, &table_def("aborted")).unwrap();
            engine.rollback(txn).unwrap();
        }

        let engine = BtreeEngine::open(&path).unwrap();
        assert!(engine.lookup_table("gone").unwrap().is_none());
        assert!(engine.lookup_table("aborted").unwrap().is_none());
        assert!(engine.lookup_table("stays").unwrap().is_some());
    }

    /// Log codec: encode→decode is the identity for every `LoggedOp` shape (pins the on-disk
    /// format against accidental drift).
    #[test]
    fn e3_logged_op_codec_round_trips() {
        let ops = [
            wal::LoggedOp::Insert {
                txn: 7,
                table: 3,
                row_id: 42,
                tuple: vec![1, 2, 3],
            },
            wal::LoggedOp::Update {
                txn: 7,
                table: 3,
                row_id: 42,
                tuple: vec![],
            },
            wal::LoggedOp::Delete {
                txn: u64::MAX,
                table: 0,
                row_id: u64::MAX,
            },
            wal::LoggedOp::CreateTable {
                txn: 1,
                table: 9,
                def: TableDef {
                    schema: "s".to_owned(),
                    name: "t".to_owned(),
                    columns: vec![
                        ColumnDef {
                            name: "a".to_owned(),
                            ty: ColumnType::VarChar(255),
                            nullable: true,
                        },
                        ColumnDef {
                            name: "b".to_owned(),
                            ty: ColumnType::Numeric {
                                precision: 10,
                                scale: 2,
                            },
                            nullable: false,
                        },
                        ColumnDef {
                            name: "c".to_owned(),
                            ty: ColumnType::Array(nusadb_core::engine::ArrayElem::Uuid),
                            nullable: false,
                        },
                    ],
                },
            },
            wal::LoggedOp::DropTable { txn: 2, table: 9 },
        ];
        for op in &ops {
            assert!(wal::roundtrip_check(op), "codec identity for {op:?}");
        }
    }

    /// Audit corner (last marker wins): a commit whose fsync failed can leave a durable
    /// `CommitTxn` marker for a transaction the caller then rolled back — the trailing
    /// `AbortTxn` must override it, or recovery would resurrect a rolled-back transaction.
    /// Pinned with a hand-crafted log, since a real fsync failure can't be forced portably.
    #[test]
    fn e3_trailing_abort_overrides_durable_commit_marker() {
        use nusadb_core::TxnId;
        use nusadb_wal::{WalRecord, WalWriter};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut w = WalWriter::new(file);
            // Txn 1: ops + commit marker + abort marker (the failed-fsync-then-rollback shape).
            w.append(
                &wal::LoggedOp::CreateTable {
                    txn: 1,
                    table: 0,
                    def: table_def("phantom"),
                }
                .to_record(),
            )
            .unwrap();
            w.append(&WalRecord::CommitTxn { txn: TxnId(1) }).unwrap();
            w.append(&WalRecord::AbortTxn { txn: TxnId(1) }).unwrap();
            // Txn 2: a genuinely committed table, proving replay itself still works.
            w.append(
                &wal::LoggedOp::CreateTable {
                    txn: 2,
                    table: 1,
                    def: table_def("real"),
                }
                .to_record(),
            )
            .unwrap();
            w.append(&WalRecord::CommitTxn { txn: TxnId(2) }).unwrap();
            w.flush().unwrap();
        }

        let engine = BtreeEngine::open(&path).unwrap();
        assert!(
            engine.lookup_table("phantom").unwrap().is_none(),
            "the trailing abort marker wins over the earlier commit marker"
        );
        assert!(engine.lookup_table("real").unwrap().is_some());
    }

    fn index_def(name: &str, table: nusadb_core::TableId, unique: bool) -> IndexDef {
        IndexDef {
            name: name.to_owned(),
            table,
            columns: vec!["v".to_owned()],
            key_exprs: Vec::new(),
            predicate: None,
            include: Vec::new(),
            kind: IndexKind::BTree,
            unique,
        }
    }

    fn collect_index(
        engine: &BtreeEngine,
        txn: nusadb_core::TxnId,
        index: nusadb_core::IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Vec<(u64, Vec<u8>)> {
        let mut scan = engine.index_scan(txn, index, lo, hi).unwrap();
        let mut out = Vec::new();
        while let Some((tid, tuple)) = scan.try_next().unwrap() {
            out.push((tid.page.0, tuple.to_vec()));
        }
        out
    }

    /// Basics: entries scan back in ascending KEY order (not insert or row-id order), range
    /// bounds work, a unique index rejects a duplicate pointing at a live row but accepts one
    /// whose old row was deleted (stale entries don't count), and catalog methods answer.
    #[test]
    fn e4_index_insert_scan_bounds_and_unique() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        let idx = engine
            .create_index(txn, &index_def("i", table, false))
            .unwrap();
        let uniq = engine
            .create_index(txn, &index_def("u", table, true))
            .unwrap();

        // Rows inserted with DESCENDING keys: the index must return ASCENDING key order.
        let mut tids = Vec::new();
        for key in (0u8..10).rev() {
            let tid = engine.insert(txn, table, &[key]).unwrap();
            engine.index_insert(txn, idx, &[key], tid).unwrap();
            tids.push((key, tid));
        }
        let all = collect_index(&engine, txn, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(all.len(), 10);
        let keys: Vec<u8> = all.iter().map(|(_, t)| t[0]).collect();
        assert_eq!(keys, (0u8..10).collect::<Vec<_>>(), "ascending key order");
        // Range [3, 7): includes 3..=6.
        let ranged = collect_index(
            &engine,
            txn,
            idx,
            Bound::Included(vec![3]),
            Bound::Excluded(vec![7]),
        );
        assert_eq!(
            ranged.iter().map(|(_, t)| t[0]).collect::<Vec<_>>(),
            vec![3, 4, 5, 6]
        );

        // Unique: same key at a live row → violation; after deleting that row, accepted.
        let (_, tid7) = tids.iter().find(|(k, _)| *k == 7).copied().unwrap();
        engine.index_insert(txn, uniq, b"k", tid7).unwrap();
        let other = engine.insert(txn, table, &[99]).unwrap();
        let err = engine
            .index_insert(txn, uniq, b"k", other)
            .expect_err("duplicate live key");
        assert!(err.to_string().contains("unique index u"));
        engine.delete(txn, table, tid7).unwrap();
        engine
            .index_insert(txn, uniq, b"k", other)
            .expect("stale entry no longer blocks");

        // Catalog methods.
        assert_eq!(engine.lookup_index("i").unwrap(), Some(idx));
        assert_eq!(engine.lookup_index("nope").unwrap(), None);
        assert_eq!(engine.list_indexes(table).unwrap().len(), 2);
        assert!(engine.index_is_complete(idx).unwrap());
        assert_eq!(engine.txn_isolation(txn), Some(RC));
        engine.commit(txn).unwrap();
    }

    /// MVCC: an index entry is only a pointer — a scan resolves it at the heap under the
    /// caller's read view, so another transaction's uncommitted row stays invisible through the
    /// index, and a pinned snapshot keeps seeing its version through the chain.
    #[test]
    fn e4_index_scan_respects_read_views() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let idx = engine
            .create_index(setup, &index_def("i", table, false))
            .unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.index_insert(setup, idx, &[1], tid).unwrap();
        engine.commit(setup).unwrap();

        let reader = engine.begin(RC).unwrap();
        let writer = engine.begin(RC).unwrap();
        let new = engine.insert(writer, table, &[2]).unwrap();
        engine.index_insert(writer, idx, &[2], new).unwrap();
        assert_eq!(
            collect_index(&engine, reader, idx, Bound::Unbounded, Bound::Unbounded).len(),
            1,
            "the uncommitted row is invisible through the index"
        );
        assert_eq!(
            collect_index(&engine, writer, idx, Bound::Unbounded, Bound::Unbounded).len(),
            2,
            "the writer sees its own entry"
        );
        engine.commit(writer).unwrap();
        // Still the reader's pinned statement snapshot — the commit is only visible at a new
        // statement.
        assert_eq!(
            collect_index(&engine, reader, idx, Bound::Unbounded, Bound::Unbounded).len(),
            1,
            "READ COMMITTED holds its statement snapshot through the index until the next statement"
        );
        engine.begin_statement(reader).unwrap();
        assert_eq!(
            collect_index(&engine, reader, idx, Bound::Unbounded, Bound::Unbounded).len(),
            2,
            "READ COMMITTED sees the commit through the index at a new statement"
        );
        engine.commit(reader).unwrap();
    }

    /// Rollback: aborted index DDL and entries vanish; a savepoint-rolled-back `drop_index`
    /// returns with all its entries.
    #[test]
    fn e4_index_rollback_and_savepoint_restore() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let idx = engine
            .create_index(setup, &index_def("i", table, false))
            .unwrap();
        let tid = engine.insert(setup, table, &[1]).unwrap();
        engine.index_insert(setup, idx, &[1], tid).unwrap();
        engine.commit(setup).unwrap();

        // Abort: entry insert + delete + a whole created index all revert.
        let txn = engine.begin(RC).unwrap();
        let t2 = engine.insert(txn, table, &[2]).unwrap();
        engine.index_insert(txn, idx, &[2], t2).unwrap();
        engine.index_delete(txn, idx, &[1], tid).unwrap();
        engine
            .create_index(txn, &index_def("ghost", table, false))
            .unwrap();
        engine.rollback(txn).unwrap();
        assert_eq!(engine.lookup_index("ghost").unwrap(), None);
        let check = engine.begin(RC).unwrap();
        let rows = collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(
            rows,
            vec![(tid.page.0, vec![1])],
            "entries restored exactly"
        );
        engine.commit(check).unwrap();

        // Savepoint: DROP INDEX undone by ROLLBACK TO — the index and its entries come back.
        let txn = engine.begin(RC).unwrap();
        engine.savepoint(txn, "sp").unwrap();
        engine.drop_index(txn, idx).unwrap();
        assert_eq!(engine.lookup_index("i").unwrap(), None);
        engine.rollback_to(txn, "sp").unwrap();
        assert_eq!(engine.lookup_index("i").unwrap(), Some(idx));
        assert_eq!(
            collect_index(&engine, txn, idx, Bound::Unbounded, Bound::Unbounded).len(),
            1,
            "the dropped index returns with its entries"
        );
        engine.commit(txn).unwrap();
    }

    /// Durability: committed index DDL + entries survive reopen (and stay complete);
    /// uncommitted index work is lost; a savepoint-rolled-back `drop_index` replays to the
    /// restored index WITH entries (the compensation re-logs them).
    #[test]
    fn e4_indexes_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, idx, tid) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let table = engine.create_table(txn, &table_def("t")).unwrap();
            let idx = engine
                .create_index(txn, &index_def("i", table, true))
                .unwrap();
            let tid = engine.insert(txn, table, &[5]).unwrap();
            engine.index_insert(txn, idx, &[5], tid).unwrap();
            engine.commit(txn).unwrap();

            // Savepoint-compensated drop: must replay to "index alive with entries".
            let txn = engine.begin(RC).unwrap();
            engine.savepoint(txn, "sp").unwrap();
            engine.drop_index(txn, idx).unwrap();
            engine.rollback_to(txn, "sp").unwrap();
            engine.commit(txn).unwrap();

            // Uncommitted: a new index + an extra entry, lost at the crash.
            let doomed = engine.begin(RC).unwrap();
            engine
                .create_index(doomed, &index_def("lost", table, false))
                .unwrap();
            let t2 = engine.insert(doomed, table, &[6]).unwrap();
            engine.index_insert(doomed, idx, &[6], t2).unwrap();
            (table, idx, tid)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        assert_eq!(engine.lookup_index("i").unwrap(), Some(idx));
        assert_eq!(engine.lookup_index("lost").unwrap(), None);
        assert!(engine.index_is_complete(idx).unwrap());
        assert_eq!(engine.list_indexes(table_id).unwrap().len(), 1);
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded),
            vec![(tid.page.0, vec![5])],
            "committed entries survive; uncommitted are gone"
        );
        // The unique check still bites after recovery (entries + heap agree).
        let t3 = engine.insert(check, table_id, &[7]).unwrap();
        assert!(engine.index_insert(check, idx, &[5], t3).is_err());
        engine.commit(check).unwrap();
    }

    /// Fix pin (audit follow-up found while building compensations): a DROP TABLE undone
    /// by ROLLBACK TO SAVEPOINT inside a later-COMMITTED transaction must replay to the table
    /// WITH its rows — the compensating `CreateTable` alone would resurrect it empty, because
    /// replay's `DropTable` discarded the rows.
    #[test]
    fn e4_dropped_table_savepoint_rows_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, keep) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            let keep = engine.insert(setup, table, &[42]).unwrap();
            engine.commit(setup).unwrap();

            let txn = engine.begin(RC).unwrap();
            engine.savepoint(txn, "sp").unwrap();
            engine.drop_table(txn, table).unwrap();
            engine.rollback_to(txn, "sp").unwrap();
            engine.commit(txn).unwrap();
            (table, keep)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        assert!(engine.lookup_table("t").unwrap().is_some());
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table_id),
            vec![(keep.page.0, vec![42])],
            "the un-dropped table replays WITH its rows, not empty"
        );
        engine.commit(check).unwrap();
    }

    /// Log codec: identity for every index-op shape (extends the pin).
    #[test]
    fn e4_index_op_codec_round_trips() {
        let ops = [
            wal::LoggedOp::CreateIndex {
                txn: 3,
                index: 1,
                def: IndexDef {
                    name: "i".to_owned(),
                    table: nusadb_core::TableId(7),
                    columns: vec!["a".to_owned(), "b".to_owned()],
                    key_exprs: Vec::new(),
                    predicate: None,
                    include: vec!["c".to_owned()],
                    kind: IndexKind::Hash,
                    unique: true,
                },
            },
            // A functional/expression index with a partial predicate exercises the appended codec
            // fields (Q index-DDL): key expressions carried as SQL text + an optional predicate.
            wal::LoggedOp::CreateIndex {
                txn: 4,
                index: 2,
                def: IndexDef {
                    name: "fx".to_owned(),
                    table: nusadb_core::TableId(7),
                    columns: Vec::new(),
                    key_exprs: vec!["lower(s)".to_owned(), "a + 1".to_owned()],
                    predicate: Some("active".to_owned()),
                    include: Vec::new(),
                    kind: IndexKind::BTree,
                    unique: true,
                },
            },
            wal::LoggedOp::DropIndex { txn: 3, index: 1 },
            wal::LoggedOp::IndexInsert {
                txn: 3,
                index: 1,
                row_id: 9,
                key: vec![0, 255, 7],
            },
            wal::LoggedOp::IndexDelete {
                txn: u64::MAX,
                index: 0,
                row_id: 0,
                key: Vec::new(),
            },
        ];
        for op in &ops {
            assert!(wal::roundtrip_check(op), "codec identity for {op:?}");
        }
    }
    /// Settled history is reclaimed — after the updating transactions commit and no
    /// snapshot pins them, purge frees every superseded version, and the freed arena slots are
    /// reused by the next update instead of growing the arena.
    #[test]
    fn e5_purge_reclaims_settled_history_and_reuses_slots() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[0]).unwrap();
        engine.commit(setup).unwrap();

        for value in 1..=5u8 {
            let txn = engine.begin(RC).unwrap();
            engine.update(txn, table, tid, &[value]).unwrap();
            engine.commit(txn).unwrap();
        }
        let stats = engine.purge().unwrap();
        assert_eq!(stats.versions_reclaimed, 5, "the whole chain is settled");
        assert_eq!(stats.rows_removed, 0);

        // The row still reads its newest value, and a fresh update reuses a freed slot (a
        // second purge reclaims exactly the one version it parked).
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(tid.page.0, vec![5])]);
        engine.update(check, table, tid, &[6]).unwrap();
        engine.commit(check).unwrap();
        assert_eq!(engine.purge().unwrap().versions_reclaimed, 1);
    }

    /// Incremental purge processes the tree in row-id batches, so a table larger than one batch
    /// must resume across batches (via its cursor) and still reclaim every settled version and read
    /// back every row. Uses more than `PURGE_ROW_BATCH` rows to force the multi-batch path.
    #[test]
    fn e5_purge_reclaims_across_many_batches() {
        let engine = BtreeEngine::new();
        let n = (crate::engine::PURGE_ROW_BATCH + 900) as u64; // > 2 batches
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let mut tids = Vec::with_capacity(n as usize);
        for _ in 0..n {
            tids.push(engine.insert(setup, table, &[0]).unwrap());
        }
        engine.commit(setup).unwrap();

        // Update every row once → one parked (dead) version per row.
        let w = engine.begin(RC).unwrap();
        for tid in &tids {
            engine.update(w, table, *tid, &[9]).unwrap();
        }
        engine.commit(w).unwrap();

        // One pass reclaims exactly one version per row, spanning several batches.
        let stats = engine.purge().unwrap();
        assert_eq!(
            stats.versions_reclaimed, n as usize,
            "every settled version reclaimed across batches"
        );
        assert_eq!(stats.rows_removed, 0);

        // Every row survives and reads its newest value; nothing was lost at a batch boundary.
        let check = engine.begin(RC).unwrap();
        let got = collect(&engine, check, table);
        assert_eq!(got.len(), n as usize);
        assert!(got.iter().all(|(_, v)| v == &[9]));
    }

    /// Incremental purge releases its latches between batches, so a writer interleaves at every
    /// batch boundary. Over a table larger than one batch, concurrent updates + repeated purges must
    /// never corrupt a chain or lose a row: every row still reads a committed value at the end.
    #[test]
    fn purge_batches_interleave_with_concurrent_updates() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let n = (crate::engine::PURGE_ROW_BATCH + 500) as u64; // spans batches
        let rounds = 20u16;
        let engine = Arc::new(BtreeEngine::new());
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("churn")).unwrap();
        let mut tids = Vec::with_capacity(n as usize);
        for _ in 0..n {
            tids.push(engine.insert(setup, table, &[0, 0]).unwrap());
        }
        engine.commit(setup).unwrap();
        let tids = Arc::new(tids);
        let done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|scope| {
            {
                let engine = Arc::clone(&engine);
                let tids = Arc::clone(&tids);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    for round in 1..=rounds {
                        let [lo, hi] = round.to_le_bytes();
                        for &tid in tids.iter() {
                            loop {
                                let txn = engine.begin(RC).unwrap();
                                match engine.update(txn, table, tid, &[lo, hi]) {
                                    Ok(_) => {
                                        engine.commit(txn).unwrap();
                                        break;
                                    },
                                    Err(_) => {
                                        let _ = engine.rollback(txn);
                                    },
                                }
                            }
                        }
                    }
                    done.store(true, Ordering::Release);
                });
            }
            // Purger: hammers purge (each pass spans several latch-releasing batches) until the
            // updater finishes.
            let engine = Arc::clone(&engine);
            let done = Arc::clone(&done);
            scope.spawn(move || {
                while !done.load(Ordering::Acquire) {
                    engine.purge().unwrap();
                }
            });
        });

        // Final state is consistent: every row is present and reads the last committed round's value.
        let last = rounds.to_le_bytes().to_vec();
        engine.purge().unwrap();
        let check = engine.begin(RC).unwrap();
        let got = collect(&engine, check, table);
        assert_eq!(got.len(), n as usize, "no row lost across batch boundaries");
        assert!(
            got.iter().all(|(_, v)| *v == last),
            "every row reads the last committed value"
        );
    }

    /// A pinned REPEATABLE READ snapshot blocks exactly the history it still needs — purge
    /// reclaims nothing while it lives, the old value keeps reading through the chain, and the
    /// moment the snapshot ends the history is reclaimable.
    #[test]
    fn e5_purge_respects_pinned_snapshots() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let tid = engine.insert(setup, table, &[10]).unwrap();
        engine.commit(setup).unwrap();

        let old = engine.begin(IsolationLevel::RepeatableRead).unwrap();
        assert_eq!(collect(&engine, old, table), vec![(tid.page.0, vec![10])]);

        let w = engine.begin(RC).unwrap();
        engine.update(w, table, tid, &[20]).unwrap();
        engine.commit(w).unwrap();

        assert_eq!(
            engine.purge().unwrap().versions_reclaimed,
            0,
            "the pinned snapshot still needs the old version"
        );
        assert_eq!(
            collect(&engine, old, table),
            vec![(tid.page.0, vec![10])],
            "the snapshot keeps reading through the chain after the purge attempt"
        );
        engine.commit(old).unwrap();
        assert_eq!(engine.purge().unwrap().versions_reclaimed, 1);
    }

    /// A settled delete is removed physically — the leaf entry disappears, its chain is
    /// freed, and its stale index entries are swept; an unsettled delete stays.
    #[test]
    fn e5_purge_removes_dead_rows_and_stale_index_entries() {
        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        let idx = engine
            .create_index(setup, &index_def("i", table, false))
            .unwrap();
        let dead = engine.insert(setup, table, &[1]).unwrap();
        engine.index_insert(setup, idx, &[1], dead).unwrap();
        let live = engine.insert(setup, table, &[2]).unwrap();
        engine.index_insert(setup, idx, &[2], live).unwrap();
        engine.commit(setup).unwrap();

        let d = engine.begin(RC).unwrap();
        engine.delete(d, table, dead).unwrap();
        engine.commit(d).unwrap();

        let stats = engine.purge().unwrap();
        assert_eq!(stats.rows_removed, 1);
        assert_eq!(stats.index_entries_removed, 1);
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table), vec![(live.page.0, vec![2])]);
        assert_eq!(
            collect_index(&engine, check, idx, Bound::Unbounded, Bound::Unbounded),
            vec![(live.page.0, vec![2])],
            "the stale entry is gone; the live one scans"
        );
        engine.commit(check).unwrap();
    }

    /// A committed DROP TABLE's pages are reclaimed once the drop settles; an aborted
    /// CREATE TABLE frees its tree immediately; the store's live-page count proves both.
    #[test]
    fn e5_purge_reclaims_dropped_table_pages() {
        let engine = BtreeEngine::new();
        let baseline = engine.live_pages().unwrap();

        // Aborted CREATE TABLE: pages come back on rollback, no purge needed.
        let txn = engine.begin(RC).unwrap();
        engine.create_table(txn, &table_def("ephemeral")).unwrap();
        assert!(engine.live_pages().unwrap() > baseline);
        engine.rollback(txn).unwrap();
        assert_eq!(engine.live_pages().unwrap(), baseline);

        // Committed DROP of a multi-page table: purge reclaims the whole tree.
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        for i in 0..2000u64 {
            let mut tuple = vec![0u8; 600];
            tuple[..8].copy_from_slice(&i.to_le_bytes());
            engine.insert(txn, table, &tuple).unwrap();
        }
        engine.commit(txn).unwrap();
        let grown = engine.live_pages().unwrap();
        assert!(grown > baseline + 10, "the table spans many pages");

        let txn = engine.begin(RC).unwrap();
        engine.drop_table(txn, table).unwrap();
        assert_eq!(
            engine.purge().unwrap().tables_reclaimed,
            0,
            "an unsettled (uncommitted) drop must not reclaim"
        );
        engine.commit(txn).unwrap();
        let stats = engine.purge().unwrap();
        assert_eq!(stats.tables_reclaimed, 1);
        assert!(stats.pages_reclaimed > 10);
        assert_eq!(engine.live_pages().unwrap(), baseline);
    }

    /// X durability: purge is unlogged, so a reopen replays the full committed history and
    /// converges to the same visible state; purging again after recovery works too.
    #[test]
    fn e5_purge_is_recovery_transparent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table_id, tid) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let table = engine.create_table(txn, &table_def("t")).unwrap();
            let tid = engine.insert(txn, table, &[1]).unwrap();
            let gone = engine.insert(txn, table, &[9]).unwrap();
            engine.commit(txn).unwrap();

            let txn = engine.begin(RC).unwrap();
            engine.update(txn, table, tid, &[2]).unwrap();
            engine.delete(txn, table, gone).unwrap();
            engine.commit(txn).unwrap();

            let stats = engine.purge().unwrap();
            assert_eq!(stats.rows_removed, 1);

            // Post-purge writes still log and commit normally.
            let txn = engine.begin(RC).unwrap();
            engine.update(txn, table, tid, &[3]).unwrap();
            engine.commit(txn).unwrap();
            (table, tid)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(
            collect(&engine, check, table_id),
            vec![(tid.page.0, vec![3])],
            "replayed state matches the purged pre-crash state"
        );
        engine.commit(check).unwrap();
        let stats = engine.purge().unwrap();
        assert_eq!(
            stats.rows_removed, 0,
            "nothing left to purge: replay is already physical (a committed delete removes              the entry and versions collapse), so recovery leaves no garbage behind"
        );
        assert_eq!(stats.versions_reclaimed, 0);
    }
    /// Constraint family basics through the treaty: a PK creates its backing unique index, the
    /// single-PK rule holds, backing indexes are exempt from the byte-level unique check,
    /// checks/FKs catalog correctly, and `fk_check`/`fk_on_delete` enforce.
    #[test]
    fn constraint_family_catalogs_and_enforces() {
        use nusadb_core::{ConstraintKind, FkAction, ForeignKeyDef};

        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let users = engine.create_table(txn, &table_def("users")).unwrap();
        let orders = engine.create_table(txn, &table_def("orders")).unwrap();

        let pk = engine
            .add_unique_constraint(txn, users, "users_pkey", &["v".to_owned()], true)
            .unwrap();
        assert_eq!(engine.lookup_index("users_pkey").unwrap(), Some(pk));
        assert!(engine.has_unique_constraint(users).unwrap());
        assert!(!engine.has_unique_constraint(orders).unwrap());
        // At most one PRIMARY KEY.
        assert!(
            engine
                .add_unique_constraint(txn, users, "users_pkey2", &["v".to_owned()], true)
                .is_err()
        );
        // The backing index is EXEMPT from the byte-level unique check (the SQL layer's
        // scan-based checks own the semantics — NULLs, transient duplicates).
        let r1 = engine.insert(txn, users, &[1]).unwrap();
        let r2 = engine.insert(txn, users, &[1]).unwrap();
        engine.index_insert(txn, pk, b"same", r1).unwrap();
        engine
            .index_insert(txn, pk, b"same", r2)
            .expect("backing index must not byte-enforce");

        engine
            .add_check_constraint(txn, users, "age_ck", b"age >= 0")
            .unwrap();
        assert!(
            engine
                .add_check_constraint(txn, users, "age_ck", b"dup")
                .is_err()
        );

        let fk = engine
            .add_foreign_key(
                txn,
                &ForeignKeyDef {
                    name: "orders_fk".to_owned(),
                    child_table: orders,
                    child_columns: vec!["v".to_owned()],
                    parent_table: users,
                    parent_columns: Vec::new(), // resolve to the parent PK
                    on_delete: FkAction::Restrict,
                    on_update: FkAction::NoAction,
                },
            )
            .unwrap();

        let cs = engine.list_constraints(users).unwrap();
        assert_eq!(cs.len(), 2, "PK + CHECK on users");
        assert!(cs.iter().any(|c| c.kind == ConstraintKind::PrimaryKey));
        assert!(cs.iter().any(|c| c.kind == ConstraintKind::Check
            && c.expr.as_deref() == Some(b"age >= 0".as_slice())));
        let ocs = engine.list_constraints(orders).unwrap();
        assert!(
            ocs.iter()
                .any(|c| c.kind == ConstraintKind::ForeignKey && c.index == Some(fk))
        );
        let fks = engine.list_foreign_keys(orders).unwrap();
        assert_eq!(fks.len(), 1);
        assert_eq!(fks[0].parent_columns, vec!["v".to_owned()]);

        // fk_check: present parent key passes, absent fails (entries live in the PK index).
        engine.fk_check(txn, "orders_fk", b"same").unwrap();
        assert!(engine.fk_check(txn, "orders_fk", b"absent").is_err());

        // fk_on_delete: RESTRICT refuses while a child row holds the key.
        let child = engine.insert(txn, orders, &[1]).unwrap();
        engine.index_insert(txn, fk, b"same", child).unwrap();
        assert!(engine.fk_on_delete(txn, users, b"same").is_err());
        engine.index_delete(txn, fk, b"same", child).unwrap();
        assert_eq!(engine.fk_on_delete(txn, users, b"same").unwrap(), 0);

        // RESTRICT on drop_constraint: the referenced PK cannot be dropped under the FK.
        assert!(engine.drop_constraint(txn, users, "users_pkey").is_err());
        engine.drop_constraint(txn, orders, "orders_fk").unwrap();
        engine.drop_constraint(txn, users, "users_pkey").unwrap();
        engine.drop_constraint(txn, users, "age_ck").unwrap();
        assert!(engine.list_constraints(users).unwrap().is_empty());
        assert_eq!(engine.lookup_index("users_pkey").unwrap(), None);
        engine.commit(txn).unwrap();
    }

    /// Rollback and savepoint semantics for the catalog family: an aborted transaction leaves
    /// no constraint behind; a savepoint-rolled-back drop restores the records.
    #[test]
    fn constraint_rollback_and_savepoints() {
        use nusadb_core::{FkAction, ForeignKeyDef};

        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let users = engine.create_table(setup, &table_def("users")).unwrap();
        let orders = engine.create_table(setup, &table_def("orders")).unwrap();
        engine
            .add_unique_constraint(setup, users, "users_pkey", &["v".to_owned()], true)
            .unwrap();
        engine.commit(setup).unwrap();

        // Abort: constraint + check + FK (and their backing indexes) all revert.
        let txn = engine.begin(RC).unwrap();
        engine
            .add_unique_constraint(txn, users, "email_uq", &["v".to_owned()], false)
            .unwrap();
        engine.add_check_constraint(txn, users, "ck", b"x").unwrap();
        engine
            .add_foreign_key(
                txn,
                &ForeignKeyDef {
                    name: "ofk".to_owned(),
                    child_table: orders,
                    child_columns: vec!["v".to_owned()],
                    parent_table: users,
                    parent_columns: Vec::new(),
                    on_delete: FkAction::NoAction,
                    on_update: FkAction::NoAction,
                },
            )
            .unwrap();
        engine.rollback(txn).unwrap();
        assert_eq!(engine.list_constraints(users).unwrap().len(), 1, "PK only");
        assert!(engine.list_foreign_keys(orders).unwrap().is_empty());
        assert_eq!(engine.lookup_index("email_uq").unwrap(), None);
        assert_eq!(engine.lookup_index("ofk").unwrap(), None);

        // Savepoint: a rolled-back DROP CONSTRAINT restores the record + backing index.
        let txn = engine.begin(RC).unwrap();
        engine.savepoint(txn, "sp").unwrap();
        engine.drop_constraint(txn, users, "users_pkey").unwrap();
        assert!(!engine.has_unique_constraint(users).unwrap());
        engine.rollback_to(txn, "sp").unwrap();
        assert!(engine.has_unique_constraint(users).unwrap());
        assert!(engine.lookup_index("users_pkey").unwrap().is_some());
        engine.commit(txn).unwrap();
    }

    /// Durability: constraints, checks, FKs, and ANALYZE stats survive a crash-reopen — and a
    /// savepoint-compensated drop replays to the restored records.
    #[test]
    fn constraints_and_stats_survive_reopen() {
        use nusadb_core::engine::TableStats;
        use nusadb_core::{ConstraintKind, FkAction, ForeignKeyDef};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (users, orders) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let users = engine.create_table(txn, &table_def("users")).unwrap();
            let orders = engine.create_table(txn, &table_def("orders")).unwrap();
            engine
                .add_unique_constraint(txn, users, "users_pkey", &["v".to_owned()], true)
                .unwrap();
            engine
                .add_check_constraint(txn, users, "ck", b"age >= 0")
                .unwrap();
            engine
                .add_foreign_key(
                    txn,
                    &ForeignKeyDef {
                        name: "ofk".to_owned(),
                        child_table: orders,
                        child_columns: vec!["v".to_owned()],
                        parent_table: users,
                        parent_columns: Vec::new(),
                        on_delete: FkAction::Cascade,
                        on_update: FkAction::NoAction,
                    },
                )
                .unwrap();
            engine
                .analyze_table(
                    txn,
                    users,
                    &TableStats {
                        row_count: 7,
                        page_count: 1,
                        columns: Vec::new(),
                    },
                )
                .unwrap();
            // Savepoint-compensated drop of the CHECK: must replay to "still present".
            engine.savepoint(txn, "sp").unwrap();
            engine.drop_constraint(txn, users, "ck").unwrap();
            engine.rollback_to(txn, "sp").unwrap();
            engine.commit(txn).unwrap();

            // Uncommitted constraint: lost at the crash.
            let doomed = engine.begin(RC).unwrap();
            engine
                .add_unique_constraint(doomed, users, "ghost_uq", &["v".to_owned()], false)
                .unwrap();
            (users, orders)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        let cs = engine.list_constraints(users).unwrap();
        assert!(cs.iter().any(|c| c.kind == ConstraintKind::PrimaryKey));
        assert!(
            cs.iter().any(|c| c.kind == ConstraintKind::Check),
            "the savepoint-compensated drop must replay to present"
        );
        assert!(!cs.iter().any(|c| c.name == "ghost_uq"));
        assert_eq!(engine.lookup_index("ghost_uq").unwrap(), None);
        let fks = engine.list_foreign_keys(orders).unwrap();
        assert_eq!(fks.len(), 1);
        assert_eq!(fks[0].on_delete, FkAction::Cascade);
        assert_eq!(
            engine.table_stats(users).unwrap().map(|s| s.row_count),
            Some(7),
            "ANALYZE stats survive recovery"
        );
        assert_eq!(engine.current_schema_version(users).unwrap(), Some(0));
    }

    /// Log codec identity for every constraint/stats op shape (extends the pins).
    #[test]
    fn constraint_op_codec_round_trips() {
        use nusadb_core::FkAction;
        use nusadb_core::engine::{ColumnStats, TableStats};

        let ops = [
            wal::LoggedOp::AddUnique {
                txn: 1,
                table: 2,
                index: 3,
                name: "pk".to_owned(),
                columns: vec!["a".to_owned(), "b".to_owned()],
                primary: true,
            },
            wal::LoggedOp::AddCheck {
                txn: 1,
                table: 2,
                name: "ck".to_owned(),
                expr: vec![0, 1, 255],
            },
            wal::LoggedOp::AddFk {
                txn: 1,
                name: "fk".to_owned(),
                child_table: 5,
                child_columns: vec!["x".to_owned()],
                parent_table: 2,
                parent_index: 3,
                child_index: 9,
                on_delete: FkAction::Cascade,
                on_update: FkAction::SetDefault,
            },
            wal::LoggedOp::DropConstraint {
                txn: 1,
                table: 2,
                name: "pk".to_owned(),
            },
            wal::LoggedOp::SetStats {
                txn: 1,
                table: 2,
                stats: TableStats {
                    row_count: 100,
                    page_count: 4,
                    columns: vec![ColumnStats {
                        column: "a".to_owned(),
                        null_count: 1,
                        distinct_count: 42,
                        min: Some(vec![0]),
                        max: None,
                        most_common: vec![(vec![7], 12)],
                        histogram: vec![vec![1], vec![2, 3]],
                    }],
                },
            },
            wal::LoggedOp::ClearStats { txn: 1, table: 2 },
        ];
        for op in &ops {
            assert!(wal::roundtrip_check(op), "codec identity for {op:?}");
        }
    }
    fn seq_def(name: &str) -> nusadb_core::engine::SequenceDef {
        nusadb_core::engine::SequenceDef {
            name: name.to_owned(),
            start: 1,
            increment: 1,
            min_value: 1,
            max_value: i64::MAX,
            cycle: false,
        }
    }

    /// Sequence family basics through the treaty: nextval is monotonic from `start`, currval
    /// follows, setval jumps, duplicate names are rejected, drop removes, and cycle/limit
    /// semantics were carried over unchanged from the predecessor engine.
    #[test]
    fn sequence_family_basics() {
        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let id = engine.create_sequence(txn, &seq_def("s")).unwrap();
        assert_eq!(engine.lookup_sequence("s").unwrap(), Some(id));
        assert!(engine.create_sequence(txn, &seq_def("s")).is_err());

        assert!(
            engine.sequence_current(id).is_err(),
            "currval before nextval"
        );
        assert_eq!(engine.sequence_next(id).unwrap(), 1);
        assert_eq!(engine.sequence_next(id).unwrap(), 2);
        assert_eq!(engine.sequence_current(id).unwrap(), 2);
        engine.sequence_set(id, 100).unwrap();
        assert_eq!(engine.sequence_next(id).unwrap(), 101);

        // Bounded + cycling: wraps to min; bounded + non-cycling: errors at the limit.
        let cyc = engine
            .create_sequence(
                txn,
                &nusadb_core::engine::SequenceDef {
                    name: "cyc".to_owned(),
                    start: 1,
                    increment: 1,
                    min_value: 1,
                    max_value: 2,
                    cycle: true,
                },
            )
            .unwrap();
        assert_eq!(engine.sequence_next(cyc).unwrap(), 1);
        assert_eq!(engine.sequence_next(cyc).unwrap(), 2);
        assert_eq!(engine.sequence_next(cyc).unwrap(), 1, "cycle wraps to min");
        let cap = engine
            .create_sequence(
                txn,
                &nusadb_core::engine::SequenceDef {
                    name: "cap".to_owned(),
                    start: 1,
                    increment: 1,
                    min_value: 1,
                    max_value: 1,
                    cycle: false,
                },
            )
            .unwrap();
        assert_eq!(engine.sequence_next(cap).unwrap(), 1);
        assert!(
            engine.sequence_next(cap).is_err(),
            "exhausted without cycle"
        );

        engine.drop_sequence(txn, id).unwrap();
        assert_eq!(engine.lookup_sequence("s").unwrap(), None);
        assert!(engine.sequence_next(id).is_err());
        engine.commit(txn).unwrap();
    }

    /// THE critical sequence property: after a crash, `nextval` never hands out
    /// a value it already handed out — even when the advancing transactions never committed
    /// (gap semantics: values may be skipped, never repeated).
    #[test]
    fn sequence_values_never_repeat_after_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let last = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let id = engine.create_sequence(txn, &seq_def("s")).unwrap();
            engine.commit(txn).unwrap();
            // Advance inside a transaction that NEVER commits — the advances must still be
            // durable (fsynced per advance), because the values escaped to the caller.
            let doomed = engine.begin(RC).unwrap();
            let mut last = 0;
            for _ in 0..5 {
                last = engine.sequence_next(id).unwrap();
            }
            let _ = doomed;
            last
        }; // crash: engine dropped, doomed txn never committed, no shutdown.

        let engine = BtreeEngine::open(&path).unwrap();
        let id = engine
            .lookup_sequence("s")
            .unwrap()
            .expect("sequence survives");
        let next = engine.sequence_next(id).unwrap();
        assert!(
            next > last,
            "nextval after crash must exceed every value handed out before it \
             (got {next}, already handed out up to {last})"
        );

        // setval is equally durable.
        engine.sequence_set(id, 1000).unwrap();
        drop(engine);
        let engine = BtreeEngine::open(&path).unwrap();
        let id = engine.lookup_sequence("s").unwrap().unwrap();
        assert_eq!(engine.sequence_next(id).unwrap(), 1001);

        // And a durable drop stays dropped.
        let txn = engine.begin(RC).unwrap();
        engine.drop_sequence(txn, id).unwrap();
        engine.commit(txn).unwrap();
        drop(engine);
        let engine = BtreeEngine::open(&path).unwrap();
        assert_eq!(engine.lookup_sequence("s").unwrap(), None);
    }

    /// The sequence OBJECT rolls back with its transaction (a rolled-back `CREATE TABLE ...
    /// SERIAL` leaves no phantom sequence) — in memory AND across a reopen (the neutralizing
    /// drop must be as durable as the unconditional create record); a `ROLLBACK TO SAVEPOINT`
    /// behaves identically. Committed values still advance non-transactionally.
    #[test]
    fn rolled_back_sequence_create_leaves_no_phantom() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let engine = BtreeEngine::open(&path).unwrap();
            // Full rollback.
            let txn = engine.begin(RC).unwrap();
            engine.create_sequence(txn, &seq_def("ghost")).unwrap();
            engine.rollback(txn).unwrap();
            assert_eq!(engine.lookup_sequence("ghost").unwrap(), None);
            // Savepoint partial rollback.
            let txn = engine.begin(RC).unwrap();
            engine.savepoint(txn, "sp").unwrap();
            engine.create_sequence(txn, &seq_def("ghost2")).unwrap();
            engine.rollback_to(txn, "sp").unwrap();
            assert_eq!(engine.lookup_sequence("ghost2").unwrap(), None);
            engine.commit(txn).unwrap();
            // A committed create for contrast.
            let txn = engine.begin(RC).unwrap();
            engine.create_sequence(txn, &seq_def("real")).unwrap();
            engine.commit(txn).unwrap();
        };

        let engine = BtreeEngine::open(&path).unwrap();
        assert_eq!(engine.lookup_sequence("ghost").unwrap(), None);
        assert_eq!(engine.lookup_sequence("ghost2").unwrap(), None);
        assert!(engine.lookup_sequence("real").unwrap().is_some());
    }

    /// FLAG-2 hardening: crash in the middle of a heavy split-producing workload — every
    /// COMMITTED batch replays complete and ordered, the uncommitted tail vanishes, and the
    /// engine keeps working after recovery.
    #[test]
    fn crash_mid_split_workload_recovers_committed_batches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let table_id = {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            let table = engine.create_table(setup, &table_def("t")).unwrap();
            engine.commit(setup).unwrap();
            // 4 committed batches of 500 × 600B rows (dozens of leaf splits + interior growth),
            // then an uncommitted 5th mid-flight at the "crash".
            for batch in 0..5u64 {
                let txn = engine.begin(RC).unwrap();
                for i in 0..500u64 {
                    let mut tuple = vec![0u8; 600];
                    tuple[..8].copy_from_slice(&(batch * 500 + i).to_le_bytes());
                    engine.insert(txn, table, &tuple).unwrap();
                }
                if batch < 4 {
                    engine.commit(txn).unwrap();
                }
            }
            table
        };

        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table_id);
        assert_eq!(
            rows.len(),
            2000,
            "4 committed batches, uncommitted 5th gone"
        );
        for (i, (_, tuple)) in rows.iter().enumerate() {
            let expect = u64::try_from(i).unwrap().to_le_bytes();
            assert_eq!(
                &tuple[..8],
                &expect,
                "payload intact and ordered after replay"
            );
        }
        // The recovered tree keeps splitting correctly.
        for i in 2000..2500u64 {
            let mut tuple = vec![0u8; 600];
            tuple[..8].copy_from_slice(&i.to_le_bytes());
            engine.insert(check, table_id, &tuple).unwrap();
        }
        engine.commit(check).unwrap();
        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table_id).len(), 2500);
        engine.commit(check).unwrap();
    }

    /// Sequence log codec identity (extends the earlier pins).
    #[test]
    fn sequence_op_codec_round_trips() {
        let ops = [
            wal::LoggedOp::SeqCreate {
                id: 3,
                def: nusadb_core::engine::SequenceDef {
                    name: "s".to_owned(),
                    start: -5,
                    increment: -2,
                    min_value: i64::MIN,
                    max_value: i64::MAX,
                    cycle: true,
                },
            },
            wal::LoggedOp::SeqDrop { id: 3 },
            wal::LoggedOp::SeqSet {
                id: 3,
                value: i64::MIN,
            },
        ];
        for op in &ops {
            assert!(wal::roundtrip_check(op), "codec identity for {op:?}");
        }
    }
    /// ALTER TABLE through the treaty: ADD/DROP/RENAME column, RENAME table, TYPE, SET/DROP NOT
    /// NULL all mutate the catalog schema, bump the version, and keep the by-name index correct.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "exercises every AlterOp variant in one flow"
    )]
    fn alter_table_mutates_schema_and_bumps_version() {
        use nusadb_core::ColumnType;
        use nusadb_core::engine::AlterOp;

        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();
        let table = engine.create_table(txn, &table_def("t")).unwrap();
        assert_eq!(engine.current_schema_version(table).unwrap(), Some(0));

        engine
            .alter_table(
                txn,
                table,
                &AlterOp::AddColumn(ColumnDef {
                    name: "extra".to_owned(),
                    ty: ColumnType::Text,
                    nullable: true,
                }),
            )
            .unwrap();
        assert_eq!(engine.current_schema_version(table).unwrap(), Some(1));
        let schema = engine.lookup_table("t").unwrap().unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[1].name, "extra");
        // Old version's schema is still resolvable (schema_for_version).
        assert_eq!(
            engine
                .schema_for_version(table, 0)
                .unwrap()
                .unwrap()
                .columns
                .len(),
            1
        );
        assert_eq!(
            engine
                .schema_for_version(table, 1)
                .unwrap()
                .unwrap()
                .columns
                .len(),
            2
        );

        // RENAME COLUMN, TYPE, SET/DROP NOT NULL.
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::RenameColumn {
                    from: "extra".to_owned(),
                    to: "note".to_owned(),
                },
            )
            .unwrap();
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::AlterColumnType {
                    column: "note".to_owned(),
                    ty: ColumnType::Int,
                },
            )
            .unwrap();
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::SetNotNull {
                    column: "note".to_owned(),
                },
            )
            .unwrap();
        let schema = engine.lookup_table("t").unwrap().unwrap();
        assert_eq!(schema.columns[1].name, "note");
        assert_eq!(schema.columns[1].ty, ColumnType::Int);
        assert!(!schema.columns[1].nullable);

        // DROP COLUMN + duplicate/unknown-column errors.
        assert!(
            engine
                .alter_table(
                    txn,
                    table,
                    &AlterOp::DropColumn {
                        name: "nope".to_owned()
                    }
                )
                .is_err()
        );
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::DropColumn {
                    name: "note".to_owned(),
                },
            )
            .unwrap();
        assert_eq!(engine.lookup_table("t").unwrap().unwrap().columns.len(), 1);

        // RENAME TABLE updates the by-name index; the old name resolves nowhere.
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::RenameTable {
                    name: "renamed".to_owned(),
                },
            )
            .unwrap();
        assert!(engine.lookup_table("t").unwrap().is_none());
        assert_eq!(engine.lookup_table("renamed").unwrap().unwrap().id, table);
        engine.commit(txn).unwrap();
    }

    /// CREATE / DROP SCHEMA through the treaty: namespaces catalog, qualified table lookup
    /// resolves, a non-existent schema blocks CREATE TABLE, and DROP RESTRICT/CASCADE behave.
    #[test]
    fn schema_namespaces_and_qualified_tables() {
        use nusadb_core::StorageEngine as _;

        let engine = BtreeEngine::new();
        let txn = engine.begin(RC).unwrap();

        // A table in a missing schema is rejected.
        let bad = TableDef {
            schema: "sales".to_owned(),
            name: "orders".to_owned(),
            columns: vec![ColumnDef {
                name: "v".to_owned(),
                ty: ColumnType::Int,
                nullable: false,
            }],
        };
        assert!(engine.create_table(txn, &bad).is_err());

        let sc = engine.create_schema(txn, "sales").unwrap();
        assert_eq!(engine.lookup_schema("sales").unwrap(), Some(sc));
        assert!(
            engine.create_schema(txn, "sales").is_err(),
            "duplicate schema"
        );

        // Now the qualified create works and resolves by (schema, name).
        let orders = engine.create_table(txn, &bad).unwrap();
        assert_eq!(
            engine
                .lookup_table_in("sales", "orders")
                .unwrap()
                .unwrap()
                .id,
            orders
        );
        assert!(
            engine.lookup_table("orders").unwrap().is_none(),
            "not in public"
        );
        assert!(
            engine
                .list_schemas()
                .unwrap()
                .iter()
                .any(|(_, n)| n == "sales")
        );

        // DROP RESTRICT on a non-empty schema errors; CASCADE drops the member table.
        assert!(engine.drop_schema(txn, sc, false).is_err());
        engine.drop_schema(txn, sc, true).unwrap();
        assert_eq!(engine.lookup_schema("sales").unwrap(), None);
        assert!(engine.lookup_table_in("sales", "orders").unwrap().is_none());
        engine.commit(txn).unwrap();
    }

    /// Rollback + savepoint for the DDL-evolution family: an aborted ALTER/CREATE SCHEMA leaves
    /// no trace, and a savepoint-rolled-back ALTER restores the exact prior schema + version.
    #[test]
    fn alter_and_schema_rollback_and_savepoints() {
        use nusadb_core::ColumnType;
        use nusadb_core::engine::AlterOp;

        let engine = BtreeEngine::new();
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("t")).unwrap();
        engine.commit(setup).unwrap();

        // Abort: ALTER + CREATE SCHEMA both revert.
        let txn = engine.begin(RC).unwrap();
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::AddColumn(ColumnDef {
                    name: "c".to_owned(),
                    ty: ColumnType::Int,
                    nullable: true,
                }),
            )
            .unwrap();
        engine.create_schema(txn, "ghost").unwrap();
        engine.rollback(txn).unwrap();
        assert_eq!(engine.current_schema_version(table).unwrap(), Some(0));
        assert_eq!(engine.lookup_table("t").unwrap().unwrap().columns.len(), 1);
        assert_eq!(engine.lookup_schema("ghost").unwrap(), None);

        // Savepoint: a rolled-back RENAME TABLE restores the name + version.
        let txn = engine.begin(RC).unwrap();
        engine.savepoint(txn, "sp").unwrap();
        engine
            .alter_table(
                txn,
                table,
                &AlterOp::RenameTable {
                    name: "moved".to_owned(),
                },
            )
            .unwrap();
        assert!(engine.lookup_table("moved").unwrap().is_some());
        engine.rollback_to(txn, "sp").unwrap();
        assert!(engine.lookup_table("t").unwrap().is_some());
        assert!(engine.lookup_table("moved").unwrap().is_none());
        assert_eq!(engine.current_schema_version(table).unwrap(), Some(0));
        engine.commit(txn).unwrap();
    }

    /// Durability: ALTER TABLE history, schemas, and qualified tables survive a crash-reopen,
    /// including a savepoint-compensated ALTER that must replay to the reverted schema.
    #[test]
    fn alter_and_schemas_survive_reopen() {
        use nusadb_core::ColumnType;
        use nusadb_core::engine::AlterOp;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        let (table, orders) = {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(RC).unwrap();
            let table = engine.create_table(txn, &table_def("t")).unwrap();
            engine
                .alter_table(
                    txn,
                    table,
                    &AlterOp::AddColumn(ColumnDef {
                        name: "c".to_owned(),
                        ty: ColumnType::Int,
                        nullable: true,
                    }),
                )
                .unwrap();
            engine
                .alter_table(
                    txn,
                    table,
                    &AlterOp::RenameColumn {
                        from: "c".to_owned(),
                        to: "d".to_owned(),
                    },
                )
                .unwrap();
            let sc = engine.create_schema(txn, "sales").unwrap();
            let orders = engine
                .create_table(
                    txn,
                    &TableDef {
                        schema: "sales".to_owned(),
                        name: "orders".to_owned(),
                        columns: vec![ColumnDef {
                            name: "v".to_owned(),
                            ty: ColumnType::Int,
                            nullable: false,
                        }],
                    },
                )
                .unwrap();
            let _ = sc;
            // Savepoint-compensated ALTER: must replay to the reverted schema (version 2).
            engine.savepoint(txn, "sp").unwrap();
            engine
                .alter_table(
                    txn,
                    table,
                    &AlterOp::DropColumn {
                        name: "d".to_owned(),
                    },
                )
                .unwrap();
            engine.rollback_to(txn, "sp").unwrap();
            engine.commit(txn).unwrap();

            // Uncommitted schema: lost at the crash.
            let doomed = engine.begin(RC).unwrap();
            engine.create_schema(doomed, "lost").unwrap();
            (table, orders)
        };

        let engine = BtreeEngine::open(&path).unwrap();
        // The compensated ALTER reverted: the "d" column is back, version is 2.
        let schema = engine.lookup_table("t").unwrap().unwrap();
        assert_eq!(
            schema.columns.len(),
            2,
            "the dropped-then-restored column is back"
        );
        assert_eq!(schema.columns[1].name, "d");
        assert_eq!(engine.current_schema_version(table).unwrap(), Some(2));
        // The qualified table + schema survive; the uncommitted one is gone.
        assert_eq!(engine.lookup_schema("sales").unwrap().map(|_| ()), Some(()));
        assert_eq!(
            engine
                .lookup_table_in("sales", "orders")
                .unwrap()
                .unwrap()
                .id,
            orders
        );
        assert_eq!(engine.lookup_schema("lost").unwrap(), None);
    }

    /// DDL log codec identity (extends the earlier pins).
    #[test]
    fn ddl_op_codec_round_trips() {
        let ops = [
            wal::LoggedOp::AlterSchema {
                txn: 4,
                table: 2,
                version: 3,
                def: TableDef {
                    schema: "sales".to_owned(),
                    name: "orders".to_owned(),
                    columns: vec![ColumnDef {
                        name: "a".to_owned(),
                        ty: ColumnType::VarChar(64),
                        nullable: false,
                    }],
                },
            },
            wal::LoggedOp::SchemaCreate {
                txn: 4,
                id: 7,
                name: "sales".to_owned(),
            },
            wal::LoggedOp::SchemaDrop {
                txn: 4,
                id: 7,
                name: "sales".to_owned(),
            },
        ];
        for op in &ops {
            assert!(wal::roundtrip_check(op), "codec identity for {op:?}");
        }
    }

    /// (sharded latching): writers on DISTINCT tables run genuinely in parallel — every
    /// commit must land, every table must hold exactly its own workers' rows, and no
    /// cross-table interleaving may corrupt either tree or the shared WAL replay (reopen).
    #[test]
    fn concurrent_writers_on_distinct_tables_all_land_and_replay() {
        const TABLES: usize = 4;
        const ROWS: u8 = 40;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let mut tables = Vec::new();
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let setup = engine.begin(RC).unwrap();
            for i in 0..TABLES {
                tables.push(
                    engine
                        .create_table(setup, &table_def(&format!("t{i}")))
                        .unwrap(),
                );
            }
            engine.commit(setup).unwrap();

            std::thread::scope(|scope| {
                for &table in &tables {
                    let engine = &engine;
                    scope.spawn(move || {
                        for v in 0..ROWS {
                            let txn = engine.begin(RC).unwrap();
                            engine.insert(txn, table, &[v]).unwrap();
                            engine.commit(txn).unwrap();
                        }
                    });
                }
            });

            let check = engine.begin(RC).unwrap();
            for &table in &tables {
                let mut values: Vec<u8> = collect(&engine, check, table)
                    .into_iter()
                    .map(|(_, v)| v[0])
                    .collect();
                values.sort_unstable();
                assert_eq!(values, (0..ROWS).collect::<Vec<u8>>());
            }
            engine.commit(check).unwrap();
        }
        // Reopen: per-table WAL order (appends under each table's latch) must replay exactly.
        let engine = BtreeEngine::open(&path).unwrap();
        let check = engine.begin(RC).unwrap();
        for &table in &tables {
            assert_eq!(collect(&engine, check, table).len(), usize::from(ROWS));
        }
        engine.commit(check).unwrap();
    }

    /// Readers are latch-free — a scan storm racing a split-heavy committing writer must
    /// only ever observe committed batch prefixes (B-link keeps the structure safe mid-split,
    /// MVCC stamps hide the in-flight batch), never a torn batch or an uncommitted row.
    #[test]
    fn latch_free_scans_see_only_committed_batches_during_split_storm() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        const BATCH: usize = 10;
        const BATCHES: usize = 60;
        let engine = Arc::new(BtreeEngine::new());
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("storm")).unwrap();
        engine.commit(setup).unwrap();
        let done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|scope| {
            {
                // Writer: commit BATCH rows at a time; wide tuples force frequent leaf splits.
                let engine = Arc::clone(&engine);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    for _ in 0..BATCHES {
                        let txn = engine.begin(RC).unwrap();
                        for _ in 0..BATCH {
                            engine.insert(txn, table, &[7u8; 512]).unwrap();
                        }
                        engine.commit(txn).unwrap();
                    }
                    done.store(true, Ordering::Release);
                });
            }
            for _ in 0..3 {
                let engine = Arc::clone(&engine);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    let mut last = 0usize;
                    while !done.load(Ordering::Acquire) {
                        let txn = engine.begin(RC).unwrap();
                        let seen = collect(&engine, txn, table).len();
                        engine.commit(txn).unwrap();
                        assert_eq!(seen % BATCH, 0, "a scan observed a torn batch: {seen} rows");
                        assert!(seen >= last, "visible history went backwards");
                        last = seen;
                    }
                });
            }
        });

        let check = engine.begin(RC).unwrap();
        assert_eq!(collect(&engine, check, table).len(), BATCH * BATCHES);
        engine.commit(check).unwrap();
    }

    /// Purge (the reclamation gate held exclusively) racing update+scan storms must never
    /// let a reader chase a freed-and-recycled undo slot into another row's version — every
    /// read must return one of the values actually committed for that row.
    #[test]
    fn purge_races_scans_and_updates_without_corrupting_chains() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        const ROWS: u8 = 8;
        const ROUNDS: u16 = 300;
        let engine = Arc::new(BtreeEngine::new());
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("churn")).unwrap();
        let mut tids = Vec::new();
        for i in 0..ROWS {
            // Value layout: [row id, round lo, round hi] — self-identifying.
            tids.push(engine.insert(setup, table, &[i, 0, 0]).unwrap());
        }
        engine.commit(setup).unwrap();
        let tids = Arc::new(tids);
        let done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|scope| {
            {
                // Updater: rewrites every row each round (chains grow), with OCC retries.
                let engine = Arc::clone(&engine);
                let tids = Arc::clone(&tids);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    for round in 1..=ROUNDS {
                        let [lo, hi] = round.to_le_bytes();
                        for (i, &tid) in tids.iter().enumerate() {
                            loop {
                                let txn = engine.begin(RC).unwrap();
                                let value = [u8::try_from(i).unwrap(), lo, hi];
                                match engine.update(txn, table, tid, &value) {
                                    Ok(_) => {
                                        engine.commit(txn).unwrap();
                                        break;
                                    },
                                    Err(_) => {
                                        let _ = engine.rollback(txn);
                                    },
                                }
                            }
                        }
                    }
                    done.store(true, Ordering::Release);
                });
            }
            {
                // Purge loop: reclaims settled chains while scans and updates are in flight.
                let engine = Arc::clone(&engine);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    while !done.load(Ordering::Acquire) {
                        engine.purge().unwrap();
                    }
                });
            }
            for _ in 0..2 {
                let engine = Arc::clone(&engine);
                let done = Arc::clone(&done);
                scope.spawn(move || {
                    while !done.load(Ordering::Acquire) {
                        let txn = engine.begin(RC).unwrap();
                        let rows = collect(&engine, txn, table);
                        engine.commit(txn).unwrap();
                        assert_eq!(rows.len(), usize::from(ROWS));
                        for (i, (_, value)) in rows.iter().enumerate() {
                            // Chain integrity: the version a view resolves must be the row's
                            // OWN (a recycled slot would leak another row's bytes here).
                            assert_eq!(
                                value[0],
                                u8::try_from(i).unwrap(),
                                "row {i} resolved a foreign version"
                            );
                        }
                    }
                });
            }
        });

        // Post-storm: all chains settled, purge reclaims them, values are the final round's.
        let stats = engine.purge().unwrap();
        let check = engine.begin(RC).unwrap();
        let rows = collect(&engine, check, table);
        engine.commit(check).unwrap();
        let [lo, hi] = ROUNDS.to_le_bytes();
        for (i, (_, value)) in rows.iter().enumerate() {
            assert_eq!(value.as_slice(), &[u8::try_from(i).unwrap(), lo, hi]);
        }
        // At least the final unsettled chains got reclaimed by SOME pass (this one or a racer).
        let _ = stats;
    }

    /// The commit gate — two SERIALIZABLE transactions in a symmetric write-skew (each read
    /// what the other wrote) racing their COMMITs from two threads must never both land: the
    /// [SSI check → stage] step is atomic across committers, so at least one observes the other
    /// as staged/committed and aborts, every round.
    #[test]
    fn racing_serializable_commits_cannot_both_land_write_skew() {
        use std::sync::{Arc, Barrier};

        const ROUNDS: usize = 200;
        let engine = Arc::new(BtreeEngine::new());
        let setup = engine.begin(RC).unwrap();
        let table = engine.create_table(setup, &table_def("skew")).unwrap();
        let on_call = engine.insert(setup, table, &[1, 0]).unwrap();
        let backup = engine.insert(setup, table, &[1, 0]).unwrap();
        engine.commit(setup).unwrap();

        for round in 0..ROUNDS {
            // Both doctors are on call at the start of every round.
            let reset = engine.begin(RC).unwrap();
            engine.update(reset, table, on_call, &[1, 0]).unwrap();
            engine.update(reset, table, backup, &[1, 0]).unwrap();
            engine.commit(reset).unwrap();

            // Each transaction reads BOTH rows (the guard: someone stays on call), then takes
            // itself off — the classic write-skew pair. The barrier lines both commits up.
            let t1 = engine.begin(SER).unwrap();
            let t2 = engine.begin(SER).unwrap();
            let sum1: u8 = collect(&engine, t1, table).iter().map(|(_, v)| v[0]).sum();
            let sum2: u8 = collect(&engine, t2, table).iter().map(|(_, v)| v[0]).sum();
            assert_eq!((sum1, sum2), (2, 2));
            engine.update(t1, table, on_call, &[0, 0]).unwrap();
            engine.update(t2, table, backup, &[0, 0]).unwrap();

            let barrier = Arc::new(Barrier::new(2));
            let (r1, r2) = std::thread::scope(|scope| {
                let h1 = {
                    let engine = Arc::clone(&engine);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        engine.commit(t1)
                    })
                };
                let h2 = {
                    let engine = Arc::clone(&engine);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        engine.commit(t2)
                    })
                };
                (h1.join().unwrap(), h2.join().unwrap())
            });
            assert!(
                !(r1.is_ok() && r2.is_ok()),
                "round {round}: both write-skew commits landed — the commit gate leaked"
            );

            let check = engine.begin(RC).unwrap();
            let sum: u8 = collect(&engine, check, table)
                .iter()
                .map(|(_, v)| v[0])
                .sum();
            engine.commit(check).unwrap();
            assert!(sum >= 1, "round {round}: nobody left on call (sum {sum})");
        }
    }
}
