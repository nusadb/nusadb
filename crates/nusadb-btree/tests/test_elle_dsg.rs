//! Elle-style **Direct Serialization Graph (DSG)** checker over a *multi-key* list-append store
//! (the real Elle, beyond the single-key oracle in `test_elle_history.rs`).
//!
//! `test_elle_history.rs` recovers one register's version order and checks each read against it —
//! an exact oracle, but single-key, so it cannot exhibit the cross-object dependency *cycles* that
//! define the standard isolation anomalies. This file builds the Adya dependency graph and looks
//! for cycles, classifying each by the edge types it contains:
//!
//! * **G0 (dirty write)** — a cycle of only **ww** edges.
//! * **G1c (circular information flow)** — a cycle of **ww/wr** edges (no anti-dependency).
//! * **G2 (anti-dependency / write skew)** — a cycle containing at least one **rw** edge.
//!
//! ## The model
//!
//! Each row is a list-append register keyed by an integer. A transaction reads some keys and appends
//! a globally-unique element to one of them. From the final per-key lists (the recovered ww-order)
//! plus the recorded reads we recover, per Adya:
//!
//! * **ww** `Ti→Tj`: `Tj` appended the element immediately after `Ti`'s on the same key.
//! * **wr** `Ti→Tj`: `Tj` read a prefix whose last element was appended by `Ti` (read-from).
//! * **rw** `Ti→Tj`: `Ti` read a prefix of length `L` on a key and `Tj` appended that key's
//!   `L`-th element — i.e. `Ti` read a version that `Tj` then overwrote.
//!
//! Self-edges are dropped (a transaction that appends to the key it read fills its own next slot).
//! Cross-key rw edges between *distinct* transactions arise only from keys a transaction reads but
//! does not write — the write-skew shape this workload deliberately generates.
//!
//! Under SERIALIZABLE the recovered DSG must be **acyclic** (an exact serializability oracle). The
//! pure checker is also exercised with three hand-built histories — one G0, one G1c, one G2 — which
//! it must detect and classify, so the positive test cannot pass vacuously.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "concurrency test harness asserts via unwrap/panic; register bytes are fixed-width u64 chunks"
)]

use std::collections::HashMap;
use std::sync::{Barrier, Mutex};
use std::thread;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{IsolationLevel, TableDef, Tid};
use nusadb_core::{ColumnDef, ColumnType, StorageEngine, TableId, TxnId};

const MAX_ATTEMPTS: u32 = 100_000;

/// A committed transaction's identity. We use the unique element it appended as its tag, so the
/// element→writer map is total and unambiguous.
type Txn = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dep {
    Ww,
    Wr,
    Rw,
}

/// What one committed transaction did: the prefix length it read on each key, and the single
/// `(key, element)` it appended.
#[derive(Clone, Debug)]
struct TxnOps {
    tag: Txn,
    reads: Vec<(u64, usize)>,
    write: (u64, u64),
}

#[derive(Debug, PartialEq, Eq)]
enum Anomaly {
    /// A dependency cycle was found; the bool-tagged edges let the caller classify it.
    G0,
    G1c,
    G2,
}

/// Build the Adya DSG and return the first cycle's class, or `Ok(())` if acyclic.
///
/// `recovered[k]` is key `k`'s final committed list (its total ww-order); `element_writer[e]` is the
/// transaction that appended element `e`.
fn check_dsg(
    recovered: &HashMap<u64, Vec<u64>>,
    element_writer: &HashMap<u64, Txn>,
    ops: &[TxnOps],
) -> Result<(), Anomaly> {
    // adjacency: txn -> list of (neighbour, edge type). Parallel edges are kept so a cycle's class
    // reflects every edge along it.
    let mut adj: HashMap<Txn, Vec<(Txn, Dep)>> = HashMap::new();
    let add = |from: Txn, to: Txn, d: Dep, adj: &mut HashMap<Txn, Vec<(Txn, Dep)>>| {
        if from != to {
            adj.entry(from).or_default().push((to, d));
        }
        adj.entry(to).or_default(); // ensure node exists
    };

    // ww: consecutive writers on each key.
    for order in recovered.values() {
        for pair in order.windows(2) {
            let (a, b) = (element_writer[&pair[0]], element_writer[&pair[1]]);
            add(a, b, Dep::Ww, &mut adj);
        }
    }

    // wr + rw from each recorded read.
    for op in ops {
        adj.entry(op.tag).or_default();
        for &(key, len) in &op.reads {
            let order = &recovered[&key];
            // wr: read-from the writer of the last element in the read prefix.
            if len > 0 {
                add(element_writer[&order[len - 1]], op.tag, Dep::Wr, &mut adj);
            }
            // rw: this txn read version `len`; whoever appended the `len`-th element overwrote it.
            if len < order.len() {
                add(op.tag, element_writer[&order[len]], Dep::Rw, &mut adj);
            }
        }
    }

    find_cycle(&adj)
}

/// DFS for a back-edge; on the first cycle, classify it by the edge types along the closing path.
fn find_cycle(adj: &HashMap<Txn, Vec<(Txn, Dep)>>) -> Result<(), Anomaly> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        White,
        Gray,
        Black,
    }
    let mut color: HashMap<Txn, Mark> = adj.keys().map(|&n| (n, Mark::White)).collect();
    // Explicit stack of (node, index into its edges, edge-type used to enter it).
    for &start in adj.keys() {
        if color[&start] != Mark::White {
            continue;
        }
        let mut stack: Vec<(Txn, usize)> = vec![(start, 0)];
        // path edges entering each stacked node (parallel to `stack[1..]`).
        let mut path_edges: Vec<Dep> = Vec::new();
        *color.get_mut(&start).unwrap() = Mark::Gray;
        while let Some(&(node, idx)) = stack.last() {
            let edges = &adj[&node];
            if idx >= edges.len() {
                *color.get_mut(&node).unwrap() = Mark::Black;
                stack.pop();
                path_edges.pop();
                continue;
            }
            stack.last_mut().unwrap().1 += 1;
            let (next, dep) = edges[idx];
            match color[&next] {
                Mark::White => {
                    *color.get_mut(&next).unwrap() = Mark::Gray;
                    stack.push((next, 0));
                    path_edges.push(dep);
                },
                Mark::Gray => {
                    // Found a cycle: `next` is on the current stack. Collect edge types from `next`
                    // around to `node`, plus the closing edge `dep`.
                    let pos = stack.iter().position(|&(n, _)| n == next).unwrap();
                    let mut deps: Vec<Dep> = path_edges[pos..].to_vec();
                    deps.push(dep);
                    return Err(classify(&deps));
                },
                Mark::Black => {},
            }
        }
    }
    Ok(())
}

fn classify(deps: &[Dep]) -> Anomaly {
    if deps.contains(&Dep::Rw) {
        Anomaly::G2
    } else if deps.contains(&Dep::Wr) {
        Anomaly::G1c
    } else {
        Anomaly::G0
    }
}

// ----- engine-driven multi-key workload -----------------------------------------------------

fn encode(key: u64, list: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + list.len() * 8);
    out.extend_from_slice(&key.to_le_bytes());
    for &x in list {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn decode(bytes: &[u8]) -> (u64, Vec<u64>) {
    let key = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let list = bytes[8..]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (key, list)
}

fn snapshot(e: &BtreeEngine, txn: TxnId, table: TableId) -> HashMap<u64, (Tid, Vec<u64>)> {
    let mut scan = e.scan(txn, table).unwrap();
    let mut out = HashMap::new();
    while let Some((tid, tuple)) = scan.try_next().unwrap() {
        let (key, list) = decode(&tuple);
        out.insert(key, (tid, list));
    }
    out
}

fn make_table(e: &BtreeEngine, keys: u64) -> TableId {
    let t = e.begin(IsolationLevel::ReadCommitted).unwrap();
    let id = e
        .create_table(
            t,
            &TableDef {
                schema: "public".to_owned(),
                name: "reg".to_owned(),
                columns: vec![
                    ColumnDef {
                        name: "k".to_owned(),
                        ty: ColumnType::Bytes,
                        nullable: false,
                    },
                    ColumnDef {
                        name: "list".to_owned(),
                        ty: ColumnType::Bytes,
                        nullable: false,
                    },
                ],
            },
        )
        .unwrap();
    for k in 0..keys {
        e.insert(t, id, &encode(k, &[])).unwrap();
    }
    e.commit(t).unwrap();
    id
}

/// Run one write-skew-shaped transaction: read keys `k1` and `k2`, append `element` to `k1` only.
/// Returns the recorded ops on commit, or `None` to retry.
fn skew_txn(
    e: &BtreeEngine,
    table: TableId,
    level: IsolationLevel,
    k1: u64,
    k2: u64,
    element: u64,
) -> Option<TxnOps> {
    let txn = e.begin(level).unwrap();
    let snap = snapshot(e, txn, table);
    let (tid1, list1) = snap[&k1].clone();
    let len2 = snap[&k2].1.len();
    let mut next = list1.clone();
    next.push(element);
    if e.update(txn, table, tid1, &encode(k1, &next)).is_err() || e.commit(txn).is_err() {
        let _ = e.rollback(txn);
        return None;
    }
    Some(TxnOps {
        tag: element,
        reads: vec![(k1, list1.len()), (k2, len2)],
        write: (k1, element),
    })
}

/// Recover per-key order and the element→writer map from the final state + recorded ops.
fn recover(
    e: &BtreeEngine,
    table: TableId,
    keys: u64,
    history: &[TxnOps],
) -> (HashMap<u64, Vec<u64>>, HashMap<u64, Txn>) {
    let verify = e.begin(IsolationLevel::ReadCommitted).unwrap();
    let snap = snapshot(e, verify, table);
    e.commit(verify).unwrap();
    let mut recovered = HashMap::new();
    for k in 0..keys {
        recovered.insert(k, snap[&k].1.clone());
    }
    let element_writer = history.iter().map(|o| (o.write.1, o.tag)).collect();
    (recovered, element_writer)
}

fn run_workload(level: IsolationLevel) {
    const KEYS: u64 = 4;
    const WORKERS: u64 = 4;
    const PER_WORKER: u64 = 30;

    let engine = BtreeEngine::new();
    let table = make_table(&engine, KEYS);
    let history: Mutex<Vec<TxnOps>> = Mutex::new(Vec::new());
    let barrier = Barrier::new(WORKERS as usize);

    thread::scope(|s| {
        for w in 0..WORKERS {
            let (barrier, engine, history) = (&barrier, &engine, &history);
            s.spawn(move || {
                barrier.wait();
                for seq in 0..PER_WORKER {
                    let element = (w << 40) | seq;
                    let k1 = (w + seq) % KEYS;
                    let k2 = (w + seq + 1) % KEYS; // read-only key -> cross rw edges
                    for _ in 0..MAX_ATTEMPTS {
                        if let Some(op) = skew_txn(engine, table, level, k1, k2, element) {
                            history.lock().unwrap().push(op);
                            break;
                        }
                    }
                }
            });
        }
    });

    let history = history.into_inner().unwrap();
    assert_eq!(history.len() as u64, WORKERS * PER_WORKER);
    let (recovered, writers) = recover(&engine, table, KEYS, &history);
    if let Err(a) = check_dsg(&recovered, &writers, &history) {
        panic!("DSG cycle ({a:?}) found under {level:?} — serializability violated");
    }
}

#[test]
fn multi_key_dsg_is_acyclic_under_serializable() {
    run_workload(IsolationLevel::Serializable);
}

// -----: seeded, reproducible mini-Jepsen sweep -----------------------------------------

/// Run one seeded concurrent write-skew history and return the DSG outcome. The seed deterministically
/// chooses the key count, worker count, per-worker transaction count, and each transaction's
/// `(read-pair, written-key)` shape — so a divergent seed reproduces the same workload shape. Thread
/// interleaving still varies run to run, but the serializability invariant (acyclic under
/// SERIALIZABLE) must hold under *every* interleaving. Returns `Ok(history_len)` if the recovered DSG
/// is acyclic, else `Err(anomaly)`.
fn run_seeded(level: IsolationLevel, seed: u64) -> Result<usize, Anomaly> {
    // xorshift64* over a non-zero state derived from the seed — a self-contained deterministic PRNG
    // (no test-only dependency) so the chosen parameters are reproducible from the seed alone.
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

    // Key count is seeded like every other parameter — the serializability oracle must hold for
    // *arbitrary* key counts. (The sweep was briefly pinned to keys=4 under a "checker false
    // positive" label; testing disproved that — the G2 cycles at keys=2/3 were real write-skew escapes
    // caused by premature committed-SSI pruning, fixed in `BtreeEngine::commit` by finalizing
    // `commit_ts` only after `TxnManager::commit` makes the commit visible.)
    let keys = 2 + rng() % 4; // 2..=5 keys
    let workers = 3 + rng() % 4; // 3..=6 workers
    let per_worker = 15 + rng() % 25; // 15..=39 transactions each
    let offset = rng(); // per-seed rotation of the key assignment

    // Pre-generate every transaction's shape up front (single-threaded, deterministic): the two
    // adjacent keys it reads and the unique element it appends to the first (the proven checker-sound
    // append-to-the-key-you-read shape from `run_workload`, here seeded and parameterized). The seed
    // varies the key count, worker/transaction counts, and the key rotation.
    let plans: Vec<Vec<(u64, u64, u64)>> = (0..workers)
        .map(|w| {
            (0..per_worker)
                .map(|seq| {
                    let k1 = (w + seq + offset) % keys;
                    let k2 = (k1 + 1) % keys; // adjacent read-only key → cross rw edges
                    let element = (w << 40) | seq;
                    (k1, k2, element)
                })
                .collect()
        })
        .collect();

    let engine = BtreeEngine::new();
    let table = make_table(&engine, keys);
    let history: Mutex<Vec<TxnOps>> = Mutex::new(Vec::new());
    let barrier = Barrier::new(workers as usize);

    thread::scope(|sc| {
        for w in 0..workers {
            let (barrier, engine, history, plan) =
                (&barrier, &engine, &history, &plans[w as usize]);
            sc.spawn(move || {
                barrier.wait();
                for &(k1, k2, element) in plan {
                    for _ in 0..MAX_ATTEMPTS {
                        if let Some(op) = skew_txn(engine, table, level, k1, k2, element) {
                            history.lock().unwrap().push(op);
                            break;
                        }
                    }
                }
            });
        }
    });

    let history = history.into_inner().unwrap();
    let (recovered, writers) = recover(&engine, table, keys, &history);
    check_dsg(&recovered, &writers, &history).map(|()| history.len())
}

/// Drive two concurrent write-skew transactions at `level`: each reads the whole table (full scan)
/// and appends to a *different* key (disjoint writes, overlapping reads — the canonical write skew).
/// Returns whether **both** committed.
fn both_commit_write_skew(level: IsolationLevel) -> bool {
    let e = BtreeEngine::new();
    let table = make_table(&e, 2);
    let ta = e.begin(level).unwrap();
    let tb = e.begin(level).unwrap();
    let snap_a = snapshot(&e, ta, table);
    let snap_b = snapshot(&e, tb, table);
    let (tid_a, mut na) = snap_a[&0].clone();
    let (tid_b, mut nb) = snap_b[&1].clone();
    na.push(100);
    nb.push(200);
    let ua = e.update(ta, table, tid_a, &encode(0, &na));
    let ub = e.update(tb, table, tid_b, &encode(1, &nb));
    let ca = e.commit(ta);
    let cb = e.commit(tb);
    ua.is_ok() && ub.is_ok() && ca.is_ok() && cb.is_ok()
}

#[test]
fn serializable_prevents_write_skew() {
    // The canonical anomaly SSI must stop: two concurrent transactions read overlapping data and
    // write disjoint keys. Under SERIALIZABLE at least one must abort (no checker — a direct,
    // deterministic engine assertion).
    assert!(
        !both_commit_write_skew(IsolationLevel::Serializable),
        "SERIALIZABLE allowed a write skew — both disjoint-write transactions committed"
    );
}

/// Drive a deterministic **multi-key** write-skew ring at `level`: three concurrent transactions
/// each read every key (overlapping reads via a full scan), then each appends to its own key
/// (pairwise-disjoint writes). Any two of them committing together is a write skew (their reads
/// overlap both written keys), so a serializable engine may commit at most one. Returns how many
/// committed. Single-threaded and fully scripted — no checker, no interleaving variance.
fn ring_commit_count(level: IsolationLevel) -> usize {
    let e = BtreeEngine::new();
    let table = make_table(&e, 3);
    let txns: Vec<TxnId> = (0..3).map(|_| e.begin(level).unwrap()).collect();
    let snaps: Vec<_> = txns.iter().map(|&t| snapshot(&e, t, table)).collect();
    for (i, (&txn, snap)) in txns.iter().zip(&snaps).enumerate() {
        let key = i as u64;
        let (tid, mut list) = snap[&key].clone();
        list.push(100 + key);
        e.update(txn, table, tid, &encode(key, &list)).unwrap();
    }
    txns.iter().filter(|&&t| e.commit(t).is_ok()).count()
}

#[test]
fn serializable_prevents_multi_key_write_skew_ring() {
    // Multi-key regression for: three transactions with overlapping reads and disjoint writes
    // across three keys. SERIALIZABLE must abort all but one — two committing would already be a
    // non-serializable write skew (rw–rw cycle between them).
    assert_eq!(
        ring_commit_count(IsolationLevel::Serializable),
        1,
        "SERIALIZABLE must commit exactly one of three mutually-skewing transactions"
    );
}

#[test]
fn read_committed_permits_multi_key_write_skew_ring() {
    // Contrast control: the same scripted ring under READ COMMITTED commits all three (write skew is
    // permitted there), confirming the SERIALIZABLE assertion above does real work.
    assert_eq!(
        ring_commit_count(IsolationLevel::ReadCommitted),
        3,
        "READ COMMITTED unexpectedly blocked a multi-key write skew"
    );
}

#[test]
fn read_committed_permits_write_skew() {
    // Contrast: the same shape under READ COMMITTED is *allowed* to write-skew (both commit),
    // confirming the SERIALIZABLE guard above is doing real work rather than aborting unconditionally.
    assert!(
        both_commit_write_skew(IsolationLevel::ReadCommitted),
        "READ COMMITTED unexpectedly blocked a write skew"
    );
}

#[test]
fn jepsen_sweep_serializable_is_always_acyclic() {
    // Seeded mini-Jepsen sweep: across many randomized, reproducible concurrent histories,
    // SERIALIZABLE must ALWAYS produce an acyclic DSG (no G0/G1c/G2). A cycle here would be a real
    // serializability (SSI) violation. Histories are fast (~ms each), so the gate runs a batch.
    for seed in 1..=64u64 {
        match run_seeded(IsolationLevel::Serializable, seed) {
            Ok(n) => assert!(n > 0, "seed {seed}: empty SERIALIZABLE history"),
            Err(a) => panic!(
                "seed {seed}: SERIALIZABLE produced a DSG cycle {a:?} — serializability violated"
            ),
        }
    }
}

#[test]
#[ignore = "full mini-Jepsen sweep (>=200 histories); run via `cargo jepsen-sweep`"]
fn jepsen_sweep_full() {
    // The full sweep the `cargo jepsen-sweep` alias runs: >=200 reproducible seeded histories, each
    // of which SERIALIZABLE must serialize into an acyclic DSG.
    const SEEDS: u64 = 200;
    for seed in 1..=SEEDS {
        if let Err(a) = run_seeded(IsolationLevel::Serializable, seed) {
            panic!("seed {seed}: SERIALIZABLE DSG cycle {a:?} — serializability violated");
        }
    }
    eprintln!("jepsen-sweep: {SEEDS} SERIALIZABLE histories all acyclic");
}

// ----- negative controls: the checker must detect each anomaly class -------------------------

#[test]
fn checker_detects_g0_pure_ww_cycle() {
    // Two keys; on key 0 the order is [a,b], on key 1 it is [b,a] -> ww cycle T_a<->T_b, no reads.
    let recovered = HashMap::from([(0u64, vec![10u64, 20]), (1u64, vec![20u64, 10])]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![],
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![],
            write: (0, 20),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Err(Anomaly::G0));
}

#[test]
fn checker_detects_g1c_ww_wr_cycle() {
    // T10 reads key1 prefix containing 20 (wr: T20->T10); ww on key0 gives T10->T20. Cycle uses
    // only ww+wr -> G1c.
    let recovered = HashMap::from([(0u64, vec![10u64, 20]), (1u64, vec![20u64])]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(1, 1)], // read [20] on key1 -> wr T20->T10
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![],
            write: (0, 20),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Err(Anomaly::G1c));
}

#[test]
fn checker_detects_g2_write_skew_rw_cycle() {
    // Classic write skew: T10 reads key1 before T20 appends there (rw T10->T20); T20 reads key0
    // before T10 appends there (rw T20->T10). Cycle of two rw edges -> G2.
    let recovered = HashMap::from([(0u64, vec![10u64]), (1u64, vec![20u64])]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(1, 0)], // read key1 empty, T20 then wrote index 0 -> rw T10->T20
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![(0, 0)], // read key0 empty, T10 then wrote index 0 -> rw T20->T10
            write: (1, 20),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Err(Anomaly::G2));
}

#[test]
fn checker_accepts_acyclic_history() {
    // T10 then T20 append to key0; T20 read T10's value (wr). No cycle.
    let recovered = HashMap::from([(0u64, vec![10u64, 20])]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(0, 0)],
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![(0, 1)], // read [10] -> wr T10->T20 (same direction as ww)
            write: (0, 20),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Ok(()));
}

// ----- multi-hop cycles + false-positive guard (Elle/Jepsen deepen) ------------------

#[test]
fn checker_detects_three_txn_g2_cycle() {
    // A 3-transaction anti-dependency ring: each reads a key (empty) that the next then writes, so
    // T10 -> T20 -> T30 -> T10 are all rw edges. A 2-cycle guard would miss this; the DFS must not.
    let recovered = HashMap::from([
        (0u64, vec![20u64]),
        (1u64, vec![30u64]),
        (2u64, vec![10u64]),
    ]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64), (30u64, 30u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(0, 0)], // read key0 before T20 wrote it -> rw T10->T20
            write: (2, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![(1, 0)], // -> rw T20->T30
            write: (0, 20),
        },
        TxnOps {
            tag: 30,
            reads: vec![(2, 0)], // -> rw T30->T10
            write: (1, 30),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Err(Anomaly::G2));
}

#[test]
fn checker_detects_three_txn_g1c_cycle() {
    // ww chain T10 -> T20 -> T30 on key0, closed by a wr back-edge T30 -> T10 (T10 reads key1, whose
    // sole value T30 wrote). The cycle spans three transactions with only ww + wr edges -> G1c.
    let recovered = HashMap::from([(0u64, vec![10u64, 20, 30]), (1u64, vec![31u64])]);
    let writers = HashMap::from([
        (10u64, 10u64),
        (20u64, 20u64),
        (30u64, 30u64),
        (31u64, 30u64),
    ]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(1, 1)], // read [31] (written by T30) -> wr T30->T10
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![],
            write: (0, 20),
        },
        TxnOps {
            tag: 30,
            reads: vec![],
            write: (0, 30),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Err(Anomaly::G1c));
}

#[test]
fn checker_no_false_positive_on_acyclic_graph_with_rw() {
    // T10 -> T20 via both a ww edge (key0 order [10,20]) and an rw edge (T10 read key1 before T20's
    // write). Both point the same way, so the graph is acyclic: an rw edge alone must not be flagged.
    let recovered = HashMap::from([(0u64, vec![10u64, 20]), (1u64, vec![20u64])]);
    let writers = HashMap::from([(10u64, 10u64), (20u64, 20u64)]);
    let ops = vec![
        TxnOps {
            tag: 10,
            reads: vec![(1, 0)], // read key1 before T20 wrote index 0 -> rw T10->T20
            write: (0, 10),
        },
        TxnOps {
            tag: 20,
            reads: vec![],
            write: (0, 20),
        },
    ];
    assert_eq!(check_dsg(&recovered, &writers, &ops), Ok(()));
}
