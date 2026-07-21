//! DST recovery-invariant oracles over the durable [`BtreeEngine`].
//!
//! A seeded, sequential workload of committed / aborted transactions runs against a durable
//! engine whose WAL is never truncated (no flush), so the WAL alone is the authoritative history.
//! The harness then re-opens the engine from **byte-prefixes of the WAL** — every cut models a
//! crash whose tail was lost — and checks three invariants the recovery oracle calls for:
//!
//! 1. **Any-prefix consistency** — recovery from *any* byte prefix must succeed (a torn tail is
//!    data loss, never a brick) and the visible state must equal some prefix of the committed
//!    history: exactly the transactions whose commit marker is durable in the cut, applied
//!    in order — never a partial transaction, never an aborted one. As the cut grows the matched
//!    history index never goes backwards (monotone prefix-closedness).
//! 2. **MVCC metadata sanity** — no recovered version has `xmax < xmin`. (Not a universal MVCC
//!    invariant — a long-lived READ COMMITTED transaction can delete a younger transaction's row —
//!    but in this harness's *sequential* history every deleter begins after the creator committed,
//!    so a backwards `xmax` in recovered state would mean recovery mis-stamped version metadata.)
//! 3. **Committed-visible** — at the full-length cut (clean restart) every committed transaction's
//!    surviving rows are visible and nothing else is: the recovered state equals the final model.
//!
//! The workload is deterministic per seed (xorshift64*), so any divergence reproduces from its
//! seed alone — the standing DST contract.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use std::collections::BTreeSet;
use std::path::Path;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{IsolationLevel, TableDef, Tid};
use nusadb_core::{ColumnDef, ColumnType, StorageEngine, TableId};

const RC: IsolationLevel = IsolationLevel::ReadCommitted;

/// One register table, opaque 8-byte payloads.
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

/// Every payload visible to a fresh reader, as a set (order-free equality with the model).
fn visible_payloads(e: &BtreeEngine) -> BTreeSet<Vec<u8>> {
    let mut out = BTreeSet::new();
    let Ok(Some(table)) = e.lookup_table("t") else {
        return out; // cut fell before the schema record — the empty pre-create state
    };
    let txn = e.begin(RC).unwrap();
    let mut scan = e.scan(txn, table.id).unwrap();
    while let Some((_, tuple)) = scan.try_next().unwrap() {
        out.insert(tuple.to_vec());
    }
    e.commit(txn).unwrap();
    out
}

/// Oracle 2: no recovered version may carry `xmax < xmin` (see the module doc for why this holds
/// in a sequential history).
fn assert_no_backward_xmax(e: &BtreeEngine, context: &str) {
    for (table, tid, xmin, xmax) in e.version_metadata().unwrap() {
        if let Some(xmax) = xmax {
            assert!(
                xmax.0 >= xmin.0,
                "{context}: version {table:?}/{tid:?} has xmax {} < xmin {} — recovery mis-stamped \
                 MVCC metadata",
                xmax.0,
                xmin.0
            );
        }
    }
}

/// Run the seeded sequential workload against a durable engine at `path`. Returns the committed
/// history as model snapshots: `snapshots[i]` is the visible payload set after the `i`-th commit
/// (index 0 = before any data commit). Aborted transactions never contribute.
fn run_history(seed: u64, path: &Path) -> Vec<BTreeSet<Vec<u8>>> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    if s == 0 {
        s = 1;
    }
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let e = BtreeEngine::open(path).unwrap();
    let txn = e.begin(RC).unwrap();
    let table = e.create_table(txn, &table_def()).unwrap();
    e.commit(txn).unwrap();

    let mut live: BTreeSet<Vec<u8>> = BTreeSet::new();
    // The empty state covers every cut up to (and including) the first data commit's marker.
    let mut snapshots = vec![live.clone()];
    let mut next_payload = 0u64;
    let mut payload = move || {
        next_payload += 1;
        next_payload.to_le_bytes().to_vec()
    };

    let data_txns = 24 + rng() % 9; // 24..=32 transactions per seed
    for _ in 0..data_txns {
        match rng() % 10 {
            // ~20%: delete the smallest live payload (skip when empty) — exercises xmax stamping
            // and the delete-only-if-deleter-committed recovery gate.
            7 | 8 if !live.is_empty() => {
                let victim = live.iter().next().unwrap().clone();
                let txn = e.begin(RC).unwrap();
                let tid = find_tid(&e, txn, table, &victim);
                e.delete(txn, table, tid).unwrap();
                e.commit(txn).unwrap();
                live.remove(&victim);
                snapshots.push(live.clone());
            },
            // ~10%: insert then ROLL BACK — these payloads must never become visible at any cut.
            9 => {
                let txn = e.begin(RC).unwrap();
                for _ in 0..=rng() % 2 {
                    e.insert(txn, table, &payload()).unwrap();
                }
                e.rollback(txn).unwrap();
            },
            // ~70%: insert 1..=3 fresh payloads and commit.
            _ => {
                let txn = e.begin(RC).unwrap();
                for _ in 0..=rng() % 3 {
                    let p = payload();
                    e.insert(txn, table, &p).unwrap();
                    live.insert(p);
                }
                e.commit(txn).unwrap();
                snapshots.push(live.clone());
            },
        }
    }
    snapshots
}

/// Resolve the `Tid` of the row whose payload equals `needle` (must be visible to `txn`).
fn find_tid(e: &BtreeEngine, txn: nusadb_core::TxnId, table: TableId, needle: &[u8]) -> Tid {
    let mut scan = e.scan(txn, table).unwrap();
    while let Some((tid, tuple)) = scan.try_next().unwrap() {
        if &*tuple == needle {
            return tid;
        }
    }
    panic!("payload {needle:?} not visible to its delete transaction");
}

/// Replay byte-prefixes of `wal` (every `stride` bytes, plus the empty and full cuts) and assert
/// the three invariants at each cut.
fn check_prefixes(wal: &[u8], snapshots: &[BTreeSet<Vec<u8>>], stride: usize, seed: u64) {
    let mut cuts: Vec<usize> = (0..wal.len()).step_by(stride.max(1)).collect();
    cuts.push(wal.len());

    let mut prev_index = 0usize;
    for cut in cuts {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wal");
        std::fs::write(&path, &wal[..cut]).unwrap();

        // Invariant 1a: any prefix must recover — a torn tail loses data, it never bricks.
        let e = BtreeEngine::open(&path)
            .unwrap_or_else(|err| panic!("seed {seed} cut {cut}: recovery bricked: {err}"));

        // Invariant 1b: the visible state is exactly some committed-history prefix, and the
        // matched index never moves backwards as the cut grows.
        let state = visible_payloads(&e);
        let index = (prev_index..snapshots.len())
            .find(|&i| snapshots[i] == state)
            .unwrap_or_else(|| {
                snapshots.iter().position(|m| *m == state).map_or_else(
                    || {
                        panic!(
                            "seed {seed} cut {cut}: recovered state ({} rows) matches NO committed \
                             history prefix — partial or phantom transaction applied",
                            state.len()
                        )
                    },
                    |i| {
                        panic!(
                            "seed {seed} cut {cut}: state matches history index {i} but index \
                             {prev_index} was already recovered at a shorter prefix — replay went \
                             backwards"
                        )
                    },
                )
            });
        prev_index = index;

        // Invariant 2: recovered MVCC metadata is sane.
        assert_no_backward_xmax(&e, &format!("seed {seed} cut {cut}"));

        // Invariant 3 (at the full cut): every committed transaction is visible after a clean
        // restart — the recovered state is the *final* model, not just any prefix.
        if cut == wal.len() {
            assert_eq!(
                state,
                *snapshots.last().unwrap(),
                "seed {seed}: clean full-WAL restart lost committed data"
            );
        }
    }
}

/// Run one seed end-to-end: workload → drop engine → prefix sweep.
fn run_seed(seed: u64, stride: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wal");
    let snapshots = run_history(seed, &path);
    let wal = std::fs::read(&path).unwrap();
    assert!(
        snapshots.len() > 8,
        "seed {seed}: degenerate history ({} commits)",
        snapshots.len()
    );
    check_prefixes(&wal, &snapshots, stride, seed);
}

#[test]
fn wal_prefix_replay_is_consistent_at_every_cut() {
    // Gate-sized sweep: a few seeds, cuts every ~1/64 of the WAL. Full density via
    // `cargo dst-invariants` (the #[ignore] test below).
    for seed in 1..=4u64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wal");
        let snapshots = run_history(seed, &path);
        let wal = std::fs::read(&path).unwrap();
        check_prefixes(&wal, &snapshots, (wal.len() / 64).max(1), seed);
    }
}

#[test]
#[ignore = "dense prefix sweep (16 seeds, fine-grained cuts); run via `cargo dst-invariants`"]
fn wal_prefix_replay_full_sweep() {
    for seed in 1..=16u64 {
        run_seed(seed, 7); // prime stride: hits header/body/CRC misalignments of every frame
    }
    eprintln!("dst-invariants: 16 seeds, every 7-byte WAL prefix cut consistent");
}
