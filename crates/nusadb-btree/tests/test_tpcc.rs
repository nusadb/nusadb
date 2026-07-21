//! TPC-C correctness subset — a transactional-consistency stress of the storage spine.
//!
//! Implements a scaled-down TPC-C over the real [`BtreeEngine`], driven directly through the
//! public `StorageEngine` surface (the storage-engine layer), the way drives the engine rather than
//! the SQL/e2e harness. Tuples are opaque `Vec<u8>` to the engine, so this test owns a
//! small fixed-width row codec for the nine TPC-C tables (every field is an `i64`; money is integer
//! cents so all consistency checks are *exact* — no float tolerance).
//!
//! The four profitable transactions are implemented faithfully:
//! - **`NewOrder`** — claim `D_NEXT_O_ID`, insert the order + new-order + order-lines, adjust stock.
//! - **`Payment`** — bump `W_YTD`/`D_YTD`, debit the customer, append a history row.
//! - **`Delivery`** — for each district, deliver the oldest undelivered order (drop its new-order,
//!   stamp the carrier + delivery dates, credit the customer).
//! - **`OrderStatus`** — read-only; asserts the snapshot invariant `O_OL_CNT == #order-lines` for
//!   the customer's latest order (atomic `NewOrder` ⇒ an order and all its lines are visible).
//!
//! After a workload we verify the twelve TPC-C consistency conditions (§3.3.2). They hold *only* if
//! the engine delivered atomicity (all-or-nothing NewOrder/Delivery), snapshot isolation (no torn
//! reads), and no lost updates (write locks order concurrent RMW) — so a single broken invariant
//! pins a real MVCC/lock/atomicity bug. We run the workload single-threaded (scripted, exercising
//! every transaction) and concurrently under both READ COMMITTED and SERIALIZABLE.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::suspicious_operation_groupings,
    clippy::option_if_let_else,
    reason = "correctness test harness: TPC-C uses w/d/c key names + field-by-field key matches"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Barrier;
use std::thread;

use nusadb_btree::BtreeEngine;
use nusadb_core::engine::{IsolationLevel, TableDef, Tid, TupleScan};
use nusadb_core::{ColumnDef, ColumnType, Error, StorageEngine, TableId, TxnId};

// ---- scale (kept tiny so no-wait-lock retries always converge) ----------------------------------

const WAREHOUSES: i64 = 2;
const DISTRICTS: i64 = 2; // per warehouse
const CUSTOMERS: i64 = 3; // per district
const ITEMS: i64 = 5;

/// Retry bound per transaction. The engine's row locks are *no-wait* (contention → immediate
/// `SerializationConflict`), so a transaction simply re-runs; every commit makes progress, so a
/// healthy engine converges far below this. Tripping it means a livelock/bug.
const MAX_ATTEMPTS: u32 = 2_000_000;

// ---- tiny deterministic RNG (xorshift64) --------------------------------------------------------

struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }
    const fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `0..n` (n > 0).
    const fn below(&mut self, n: i64) -> i64 {
        (self.next() % n as u64) as i64
    }
    /// Uniform in `lo..=hi`.
    const fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below(hi - lo + 1)
    }
}

// ---- fixed-width row codec (every field an i64; Option<i64> = [flag, value]) ---------------------

fn enc(fields: &[i64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(fields.len() * 8);
    for f in fields {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b
}

fn rd(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const fn enc_opt(v: Option<i64>) -> [i64; 2] {
    match v {
        Some(x) => [1, x],
        None => [0, 0],
    }
}

const fn dec_opt(flag: i64, val: i64) -> Option<i64> {
    if flag == 0 { None } else { Some(val) }
}

#[derive(Clone)]
struct Warehouse {
    id: i64,
    ytd: i64,
}
impl Warehouse {
    fn enc(&self) -> Vec<u8> {
        enc(&[self.id, self.ytd])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            id: f[0],
            ytd: f[1],
        }
    }
}

#[derive(Clone)]
struct District {
    w: i64,
    id: i64,
    ytd: i64,
    next_o_id: i64,
}
impl District {
    fn enc(&self) -> Vec<u8> {
        enc(&[self.w, self.id, self.ytd, self.next_o_id])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            id: f[1],
            ytd: f[2],
            next_o_id: f[3],
        }
    }
}

#[derive(Clone)]
struct Customer {
    w: i64,
    d: i64,
    id: i64,
    balance: i64,
    ytd_payment: i64,
    payment_cnt: i64,
    delivery_cnt: i64,
}
impl Customer {
    fn enc(&self) -> Vec<u8> {
        enc(&[
            self.w,
            self.d,
            self.id,
            self.balance,
            self.ytd_payment,
            self.payment_cnt,
            self.delivery_cnt,
        ])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            d: f[1],
            id: f[2],
            balance: f[3],
            ytd_payment: f[4],
            payment_cnt: f[5],
            delivery_cnt: f[6],
        }
    }
}

#[derive(Clone)]
struct History {
    w: i64,
    d: i64,
    c_id: i64,
    amount: i64,
}
impl History {
    fn enc(&self) -> Vec<u8> {
        enc(&[self.w, self.d, self.c_id, self.amount])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            d: f[1],
            c_id: f[2],
            amount: f[3],
        }
    }
}

#[derive(Clone)]
struct NewOrder {
    w: i64,
    d: i64,
    o_id: i64,
}
impl NewOrder {
    fn enc(&self) -> Vec<u8> {
        enc(&[self.w, self.d, self.o_id])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            d: f[1],
            o_id: f[2],
        }
    }
}

#[derive(Clone)]
struct Order {
    w: i64,
    d: i64,
    id: i64,
    c_id: i64,
    carrier: Option<i64>,
    ol_cnt: i64,
}
impl Order {
    fn enc(&self) -> Vec<u8> {
        let c = enc_opt(self.carrier);
        enc(&[self.w, self.d, self.id, self.c_id, c[0], c[1], self.ol_cnt])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            d: f[1],
            id: f[2],
            c_id: f[3],
            carrier: dec_opt(f[4], f[5]),
            ol_cnt: f[6],
        }
    }
}

#[derive(Clone)]
struct OrderLine {
    w: i64,
    d: i64,
    o_id: i64,
    number: i64,
    i_id: i64,
    qty: i64,
    amount: i64,
    delivery: Option<i64>,
}
impl OrderLine {
    fn enc(&self) -> Vec<u8> {
        let dl = enc_opt(self.delivery);
        enc(&[
            self.w,
            self.d,
            self.o_id,
            self.number,
            self.i_id,
            self.qty,
            self.amount,
            dl[0],
            dl[1],
        ])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            d: f[1],
            o_id: f[2],
            number: f[3],
            i_id: f[4],
            qty: f[5],
            amount: f[6],
            delivery: dec_opt(f[7], f[8]),
        }
    }
}

#[derive(Clone)]
struct Stock {
    w: i64,
    i_id: i64,
    quantity: i64,
    ytd: i64,
    order_cnt: i64,
}
impl Stock {
    fn enc(&self) -> Vec<u8> {
        enc(&[self.w, self.i_id, self.quantity, self.ytd, self.order_cnt])
    }
    fn dec(b: &[u8]) -> Self {
        let f = rd(b);
        Self {
            w: f[0],
            i_id: f[1],
            quantity: f[2],
            ytd: f[3],
            order_cnt: f[4],
        }
    }
}

// ---- catalog handles ----------------------------------------------------------------------------

#[derive(Clone)]
struct Tables {
    warehouse: TableId,
    district: TableId,
    customer: TableId,
    history: TableId,
    new_order: TableId,
    order: TableId,
    order_line: TableId,
    stock: TableId,
    /// Item price (cents) per item id — items are read-only reference data, cached so transactions
    /// don't contend on reading them.
    prices: Vec<i64>,
    /// How many warehouses were loaded (the TPC-C contention dial: terminals scale with
    /// warehouses in the spec, so a perf cell that wants low row-collision loads more of them).
    warehouses: i64,
    /// A terminal's home warehouse (the spec binds each terminal to one): `Some` pins every
    /// transaction of this handle to that warehouse, `None` picks uniformly per transaction.
    home: Option<i64>,
}

impl Tables {
    /// The warehouse this transaction targets: the terminal's home, or a uniform pick.
    fn pick_warehouse(&self, rng: &mut Rng) -> i64 {
        self.home.unwrap_or_else(|| rng.below(self.warehouses))
    }
}

/// A one-column placeholder table def (the engine never interprets the columns — tuples are opaque
/// — but `create_table` wants a `TableDef`).
fn def(name: &str) -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: name.to_owned(),
        columns: vec![ColumnDef {
            name: "row".to_owned(),
            ty: ColumnType::Bytes,
            nullable: false,
        }],
    }
}

// ---- engine helpers -----------------------------------------------------------------------------

/// Scan every live row of `table`, decoding each with `f`.
fn all<T>(e: &BtreeEngine, txn: TxnId, table: TableId, f: impl Fn(&[u8]) -> T) -> Vec<(Tid, T)> {
    let mut scan: Box<dyn TupleScan> = e.scan(txn, table).unwrap();
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next().unwrap() {
        out.push((tid, f(&bytes)));
    }
    out
}

/// Attempts (begin..commit tries) across the whole process — perf cells report the abort ratio
/// (attempts / committed) so a throughput move can be attributed to retry-rate vs per-txn cost.
static ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Run `body` as a transaction, retrying on any engine conflict until it commits.
fn run_txn<F>(e: &BtreeEngine, level: IsolationLevel, mut body: F)
where
    F: FnMut(TxnId) -> Result<(), Error>,
{
    // Full-jitter exponential backoff between conflict retries: colliding no-wait
    // transactions that retry immediately re-collide in lockstep — the retry storm measured
    // at 1.43 attempts/commit on the extreme contended cell. Sleep uniform[0, cap] with the cap
    // doubling per failed attempt, bounded well under a transaction's own runtime so the
    // uncontended cells pay nothing. Jitter comes from the same deterministic xorshift the
    // harness uses, seeded per call from a thread-distinct value.
    let mut jitter = Rng::new({
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut h);
        h.finish() | 1
    });
    for attempt in 0..MAX_ATTEMPTS {
        ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let txn = e.begin(level).unwrap();
        match body(txn) {
            Ok(()) => match e.commit(txn) {
                Ok(()) => return,
                Err(_) => {
                    let _ = e.rollback(txn);
                },
            },
            Err(_) => {
                let _ = e.rollback(txn);
            },
        }
        let cap_us = 1u64 << attempt.min(7); // 1..128 microseconds
        let sleep_us = jitter.next() % (cap_us + 1);
        if sleep_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(sleep_us));
        }
    }
    panic!("transaction livelocked: exceeded {MAX_ATTEMPTS} attempts");
}

// ---- load ---------------------------------------------------------------------------------------

fn load(e: &BtreeEngine) -> Tables {
    load_warehouses(e, WAREHOUSES)
}

fn load_warehouses(e: &BtreeEngine, warehouses: i64) -> Tables {
    let t = e.begin(IsolationLevel::ReadCommitted).unwrap();
    let tables = Tables {
        warehouse: e.create_table(t, &def("warehouse")).unwrap(),
        district: e.create_table(t, &def("district")).unwrap(),
        customer: e.create_table(t, &def("customer")).unwrap(),
        history: e.create_table(t, &def("history")).unwrap(),
        new_order: e.create_table(t, &def("new_order")).unwrap(),
        order: e.create_table(t, &def("oorder")).unwrap(),
        order_line: e.create_table(t, &def("order_line")).unwrap(),
        stock: e.create_table(t, &def("stock")).unwrap(),
        prices: (0..ITEMS).map(|i| (i + 1) * 100).collect(),
        warehouses,
        home: None,
    };

    // Everything starts at zero YTD / empty order history so conditions 8/9 are W_YTD == ΣH cleanly.
    for w in 0..warehouses {
        e.insert(t, tables.warehouse, &Warehouse { id: w, ytd: 0 }.enc())
            .unwrap();
        for d in 0..DISTRICTS {
            e.insert(
                t,
                tables.district,
                &District {
                    w,
                    id: d,
                    ytd: 0,
                    next_o_id: 1,
                }
                .enc(),
            )
            .unwrap();
            for c in 0..CUSTOMERS {
                e.insert(
                    t,
                    tables.customer,
                    &Customer {
                        w,
                        d,
                        id: c,
                        balance: 0,
                        ytd_payment: 0,
                        payment_cnt: 0,
                        delivery_cnt: 0,
                    }
                    .enc(),
                )
                .unwrap();
            }
        }
        for i in 0..ITEMS {
            e.insert(
                t,
                tables.stock,
                &Stock {
                    w,
                    i_id: i,
                    quantity: 100,
                    ytd: 0,
                    order_cnt: 0,
                }
                .enc(),
            )
            .unwrap();
        }
    }
    e.commit(t).unwrap();
    tables
}

// ---- transactions -------------------------------------------------------------------------------

fn new_order(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    let w = tb.pick_warehouse(rng);
    let d = rng.below(DISTRICTS);
    let c = rng.below(CUSTOMERS);
    // Distinct item ids (so we never update the same stock row twice in one txn).
    let mut item_ids: Vec<i64> = (0..ITEMS).collect();
    for i in (1..item_ids.len()).rev() {
        let j = rng.below(i as i64 + 1) as usize;
        item_ids.swap(i, j);
    }
    let ol_cnt = rng.range(1, ITEMS);
    item_ids.truncate(ol_cnt as usize);
    let lines: Vec<(i64, i64)> = item_ids.iter().map(|&i| (i, rng.range(1, 10))).collect();

    run_txn(e, level, |txn| {
        let (d_tid, dist) = all(e, txn, tb.district, District::dec)
            .into_iter()
            .find(|(_, x)| x.w == w && x.id == d)
            .expect("district exists");
        let o_id = dist.next_o_id;
        let mut dist = dist;
        dist.next_o_id += 1;
        e.update(txn, tb.district, d_tid, &dist.enc())?;

        e.insert(
            txn,
            tb.order,
            &Order {
                w,
                d,
                id: o_id,
                c_id: c,
                carrier: None,
                ol_cnt: lines.len() as i64,
            }
            .enc(),
        )?;
        e.insert(txn, tb.new_order, &NewOrder { w, d, o_id }.enc())?;

        for (number, &(i_id, qty)) in lines.iter().enumerate() {
            let (s_tid, stock) = all(e, txn, tb.stock, Stock::dec)
                .into_iter()
                .find(|(_, s)| s.w == w && s.i_id == i_id)
                .expect("stock exists");
            let mut stock = stock;
            stock.quantity = if stock.quantity - qty >= 10 {
                stock.quantity - qty
            } else {
                stock.quantity - qty + 91
            };
            stock.ytd += qty;
            stock.order_cnt += 1;
            e.update(txn, tb.stock, s_tid, &stock.enc())?;

            e.insert(
                txn,
                tb.order_line,
                &OrderLine {
                    w,
                    d,
                    o_id,
                    number: number as i64,
                    i_id,
                    qty,
                    amount: qty * tb.prices[i_id as usize],
                    delivery: None,
                }
                .enc(),
            )?;
        }
        Ok(())
    });
}

fn payment(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    let w = tb.pick_warehouse(rng);
    let d = rng.below(DISTRICTS);
    let c = rng.below(CUSTOMERS);
    let amount = rng.range(100, 50_000);

    run_txn(e, level, |txn| {
        let (w_tid, wh) = all(e, txn, tb.warehouse, Warehouse::dec)
            .into_iter()
            .find(|(_, x)| x.id == w)
            .expect("warehouse exists");
        let mut wh = wh;
        wh.ytd += amount;
        e.update(txn, tb.warehouse, w_tid, &wh.enc())?;

        let (d_tid, dist) = all(e, txn, tb.district, District::dec)
            .into_iter()
            .find(|(_, x)| x.w == w && x.id == d)
            .expect("district exists");
        let mut dist = dist;
        dist.ytd += amount;
        e.update(txn, tb.district, d_tid, &dist.enc())?;

        let (c_tid, cust) = all(e, txn, tb.customer, Customer::dec)
            .into_iter()
            .find(|(_, x)| x.w == w && x.d == d && x.id == c)
            .expect("customer exists");
        let mut cust = cust;
        cust.balance -= amount;
        cust.ytd_payment += amount;
        cust.payment_cnt += 1;
        e.update(txn, tb.customer, c_tid, &cust.enc())?;

        e.insert(
            txn,
            tb.history,
            &History {
                w,
                d,
                c_id: c,
                amount,
            }
            .enc(),
        )?;
        Ok(())
    });
}

fn delivery(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    let w = tb.pick_warehouse(rng);
    let carrier = rng.range(1, 10);
    let delivery_date = rng.range(1, 1_000_000);

    run_txn(e, level, |txn| {
        for d in 0..DISTRICTS {
            // Oldest undelivered order for this district.
            let Some((no_tid, no)) = all(e, txn, tb.new_order, NewOrder::dec)
                .into_iter()
                .filter(|(_, n)| n.w == w && n.d == d)
                .min_by_key(|(_, n)| n.o_id)
            else {
                continue; // no outstanding order in this district
            };
            e.delete(txn, tb.new_order, no_tid)?;

            let (o_tid, order) = all(e, txn, tb.order, Order::dec)
                .into_iter()
                .find(|(_, o)| o.w == w && o.d == d && o.id == no.o_id)
                .expect("order for new_order exists");
            let mut order = order;
            order.carrier = Some(carrier);
            e.update(txn, tb.order, o_tid, &order.enc())?;

            let mut total = 0;
            let lines: Vec<(Tid, OrderLine)> = all(e, txn, tb.order_line, OrderLine::dec)
                .into_iter()
                .filter(|(_, ol)| ol.w == w && ol.d == d && ol.o_id == no.o_id)
                .collect();
            for (ol_tid, ol) in lines {
                total += ol.amount;
                let mut ol = ol;
                ol.delivery = Some(delivery_date);
                e.update(txn, tb.order_line, ol_tid, &ol.enc())?;
            }

            let (c_tid, cust) = all(e, txn, tb.customer, Customer::dec)
                .into_iter()
                .find(|(_, x)| x.w == w && x.d == d && x.id == order.c_id)
                .expect("order's customer exists");
            let mut cust = cust;
            cust.balance += total;
            cust.delivery_cnt += 1;
            e.update(txn, tb.customer, c_tid, &cust.enc())?;
        }
        Ok(())
    });
}

fn order_status(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    let w = tb.pick_warehouse(rng);
    let d = rng.below(DISTRICTS);
    let c = rng.below(CUSTOMERS);

    run_txn(e, level, |txn| {
        // The customer's latest order, if any.
        let latest = all(e, txn, tb.order, Order::dec)
            .into_iter()
            .filter(|(_, o)| o.w == w && o.d == d && o.c_id == c)
            .max_by_key(|(_, o)| o.id);
        if let Some((_, order)) = latest {
            let n_lines = all(e, txn, tb.order_line, OrderLine::dec)
                .into_iter()
                .filter(|(_, ol)| ol.w == w && ol.d == d && ol.o_id == order.id)
                .count() as i64;
            // Snapshot invariant: an order and all its lines were inserted by one atomic NewOrder,
            // so a consistent snapshot must see exactly `O_OL_CNT` of them — never a partial order.
            assert_eq!(
                n_lines, order.ol_cnt,
                "OrderStatus saw a torn order: {} lines vs O_OL_CNT {}",
                n_lines, order.ol_cnt
            );
        }
        Ok(())
    });
}

/// TPC-C `StockLevel` (§2.8): read-only. Examine the district's last 20 orders' order-lines and count
/// the distinct items whose stock is below a threshold. Under a per-transaction snapshot the count
/// is recomputed within the same transaction and must be identical — a repeatable-read / consistent-
/// snapshot assertion (no phantom order-line or concurrent stock change leaks in mid-transaction).
fn stock_level(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    let w = tb.pick_warehouse(rng);
    let d = rng.below(DISTRICTS);
    let threshold = rng.range(10, 20);
    run_txn(e, level, |txn| {
        let first = low_stock(e, txn, tb, w, d, threshold);
        // REPEATABLE READ / SERIALIZABLE pin one snapshot per transaction, so a second pass must
        // match; READ COMMITTED re-reads per statement, so equality is not required there.
        if matches!(
            level,
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable
        ) {
            let second = low_stock(e, txn, tb, w, d, threshold);
            assert_eq!(
                first, second,
                "StockLevel non-repeatable read: {first} vs {second}"
            );
        }
        Ok(())
    });
}

/// Distinct low-stock item count across the district's recent orders, on transaction `txn`'s
/// snapshot.
fn low_stock(e: &BtreeEngine, txn: TxnId, tb: &Tables, w: i64, d: i64, threshold: i64) -> i64 {
    let next_o_id = all(e, txn, tb.district, District::dec)
        .into_iter()
        .find(|(_, dist)| dist.w == w && dist.id == d)
        .map_or(1, |(_, dist)| dist.next_o_id);
    let lo = (next_o_id - 20).max(0);
    let recent_items: BTreeSet<i64> = all(e, txn, tb.order_line, OrderLine::dec)
        .into_iter()
        .filter(|(_, ol)| ol.w == w && ol.d == d && ol.o_id >= lo && ol.o_id < next_o_id)
        .map(|(_, ol)| ol.i_id)
        .collect();
    let stock = all(e, txn, tb.stock, Stock::dec);
    recent_items
        .iter()
        .filter(|&&i| {
            stock
                .iter()
                .any(|(_, s)| s.w == w && s.i_id == i && s.quantity < threshold)
        })
        .count() as i64
}

// ---- consistency conditions (§3.3.2) ------------------------------------------------------------

/// Verify TPC-C consistency conditions 1–12 over a single consistent snapshot.
fn check_consistency(e: &BtreeEngine, tb: &Tables) {
    let txn = e.begin(IsolationLevel::ReadCommitted).unwrap();
    let warehouses: Vec<Warehouse> = all(e, txn, tb.warehouse, Warehouse::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let districts: Vec<District> = all(e, txn, tb.district, District::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let customers: Vec<Customer> = all(e, txn, tb.customer, Customer::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let histories: Vec<History> = all(e, txn, tb.history, History::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let new_orders: Vec<NewOrder> = all(e, txn, tb.new_order, NewOrder::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let orders: Vec<Order> = all(e, txn, tb.order, Order::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    let order_lines: Vec<OrderLine> = all(e, txn, tb.order_line, OrderLine::dec)
        .into_iter()
        .map(|(_, x)| x)
        .collect();
    e.commit(txn).unwrap();

    let orders_by_key: BTreeMap<(i64, i64, i64), &Order> =
        orders.iter().map(|o| ((o.w, o.d, o.id), o)).collect();

    // W_YTD == Σ D_YTD over the warehouse's districts.
    for w in &warehouses {
        let sum: i64 = districts
            .iter()
            .filter(|d| d.w == w.id)
            .map(|d| d.ytd)
            .sum();
        assert_eq!(w.ytd, sum, "C1 warehouse {} W_YTD != ΣD_YTD", w.id);
    }

    for d in &districts {
        let dist_orders: Vec<&Order> = orders
            .iter()
            .filter(|o| o.w == d.w && o.d == d.id)
            .collect();
        let dist_nos: Vec<&NewOrder> = new_orders
            .iter()
            .filter(|n| n.w == d.w && n.d == d.id)
            .collect();

        // D_NEXT_O_ID - 1 == max(O_ID) == max(NO_O_ID).
        let max_o = dist_orders.iter().map(|o| o.id).max();
        assert_eq!(
            d.next_o_id - 1,
            max_o.unwrap_or(0),
            "C2 district ({},{}) NEXT_O_ID-1 != max(O_ID)",
            d.w,
            d.id
        );
        if let Some(max_no) = dist_nos.iter().map(|n| n.o_id).max() {
            assert_eq!(
                max_no,
                max_o.unwrap_or(-1),
                "C2 district ({},{}) max(NO_O_ID) != max(O_ID)",
                d.w,
                d.id
            );

            // Max(NO_O_ID) - min(NO_O_ID) + 1 == #new_order rows (contiguous).
            let min_no = dist_nos.iter().map(|n| n.o_id).min().unwrap();
            assert_eq!(
                max_no - min_no + 1,
                dist_nos.len() as i64,
                "C3 district ({},{}) new_order ids not contiguous",
                d.w,
                d.id
            );
        }

        // Σ O_OL_CNT == #order_line rows for the district.
        let sum_ol_cnt: i64 = dist_orders.iter().map(|o| o.ol_cnt).sum();
        let n_lines = order_lines
            .iter()
            .filter(|ol| ol.w == d.w && ol.d == d.id)
            .count() as i64;
        assert_eq!(
            sum_ol_cnt, n_lines,
            "C4 district ({},{}) ΣO_OL_CNT != #order_line",
            d.w, d.id
        );
    }

    for o in &orders {
        let in_new_order = new_orders
            .iter()
            .any(|n| n.w == o.w && n.d == o.d && n.o_id == o.id);
        // O_CARRIER_ID is null iff the order is still in new_order (undelivered).
        assert_eq!(
            o.carrier.is_none(),
            in_new_order,
            "C5 order ({},{},{}) carrier/new_order mismatch",
            o.w,
            o.d,
            o.id
        );

        // O_OL_CNT == #order_line for the order.
        let n = order_lines
            .iter()
            .filter(|ol| ol.w == o.w && ol.d == o.d && ol.o_id == o.id)
            .count() as i64;
        assert_eq!(
            o.ol_cnt, n,
            "C6 order ({},{},{}) O_OL_CNT != #lines",
            o.w, o.d, o.id
        );
    }

    // OL_DELIVERY_D is null iff the order's O_CARRIER_ID is null.
    for ol in &order_lines {
        let order = orders_by_key
            .get(&(ol.w, ol.d, ol.o_id))
            .expect("order_line's order exists");
        assert_eq!(
            ol.delivery.is_none(),
            order.carrier.is_none(),
            "C7 order_line ({},{},{},{}) delivery/carrier mismatch",
            ol.w,
            ol.d,
            ol.o_id,
            ol.number
        );
    }

    // /: W_YTD == ΣH_AMOUNT (warehouse) and D_YTD == ΣH_AMOUNT (district).
    for w in &warehouses {
        let sum: i64 = histories
            .iter()
            .filter(|h| h.w == w.id)
            .map(|h| h.amount)
            .sum();
        assert_eq!(w.ytd, sum, "C8 warehouse {} W_YTD != ΣH_AMOUNT", w.id);
    }
    for d in &districts {
        let sum: i64 = histories
            .iter()
            .filter(|h| h.w == d.w && h.d == d.id)
            .map(|h| h.amount)
            .sum();
        assert_eq!(
            d.ytd, sum,
            "C9 district ({},{}) D_YTD != ΣH_AMOUNT",
            d.w, d.id
        );
    }

    // //: customer balance / ytd_payment reconcile with delivered lines and payments.
    for c in &customers {
        let paid: i64 = histories
            .iter()
            .filter(|h| h.w == c.w && h.d == c.d && h.c_id == c.id)
            .map(|h| h.amount)
            .sum();
        let delivered: i64 = order_lines
            .iter()
            .filter(|ol| {
                ol.delivery.is_some()
                    && orders_by_key
                        .get(&(ol.w, ol.d, ol.o_id))
                        .is_some_and(|o| o.w == c.w && o.d == c.d && o.c_id == c.id)
            })
            .map(|ol| ol.amount)
            .sum();

        // C_BALANCE == Σ(delivered OL_AMOUNT) − Σ(H_AMOUNT).
        assert_eq!(
            c.balance,
            delivered - paid,
            "C10 customer ({},{},{}) balance != delivered - paid",
            c.w,
            c.d,
            c.id
        );
        // C_YTD_PAYMENT == Σ(H_AMOUNT) for the customer.
        assert_eq!(
            c.ytd_payment, paid,
            "C11 customer ({},{},{}) YTD_PAYMENT != ΣH_AMOUNT",
            c.w, c.d, c.id
        );
        // C_BALANCE + C_YTD_PAYMENT == Σ(delivered OL_AMOUNT).
        assert_eq!(
            c.balance + c.ytd_payment,
            delivered,
            "C12 customer ({},{},{}) balance + ytd != delivered",
            c.w,
            c.d,
            c.id
        );
    }
}

// ---- workload driver ----------------------------------------------------------------------------

/// Run one randomly-chosen transaction using the TPC-C spec mix (§5.2.3): ~45% `NewOrder`, ~43%
/// `Payment`, ~4% each of `Delivery` / `OrderStatus` / `StockLevel`.
fn one_transaction(e: &BtreeEngine, tb: &Tables, level: IsolationLevel, rng: &mut Rng) {
    match rng.below(100) {
        0..=44 => new_order(e, tb, level, rng),
        45..=87 => payment(e, tb, level, rng),
        88..=91 => delivery(e, tb, level, rng),
        92..=95 => order_status(e, tb, level, rng),
        _ => stock_level(e, tb, level, rng),
    }
}

// ---- tests --------------------------------------------------------------------------------------

#[test]
fn scripted_single_thread_runs_all_five_transactions_and_stays_consistent() {
    let e = BtreeEngine::new();
    let tb = load(&e);
    let level = IsolationLevel::Serializable;
    let mut rng = Rng::new(0xA5_06);

    // Force every transaction type at least once, in an order that exercises the delivery path
    // (NewOrders create outstanding orders; Delivery then drains them).
    for _ in 0..6 {
        new_order(&e, &tb, level, &mut rng);
    }
    for _ in 0..4 {
        payment(&e, &tb, level, &mut rng);
    }
    order_status(&e, &tb, level, &mut rng);
    stock_level(&e, &tb, level, &mut rng);
    delivery(&e, &tb, level, &mut rng);
    delivery(&e, &tb, level, &mut rng);
    payment(&e, &tb, level, &mut rng);
    order_status(&e, &tb, level, &mut rng);
    stock_level(&e, &tb, level, &mut rng);

    check_consistency(&e, &tb);
}

fn concurrent_workload(level: IsolationLevel, seed: u64, workers: usize, iters: usize) {
    let e = BtreeEngine::new();
    let tb = load(&e);

    let barrier = Barrier::new(workers);
    thread::scope(|s| {
        for wkr in 0..workers {
            let barrier = &barrier;
            let e = &e;
            let tb = &tb;
            s.spawn(move || {
                let mut rng = Rng::new(seed.wrapping_add(wkr as u64 * 0x9e37_79b9));
                barrier.wait();
                for _ in 0..iters {
                    one_transaction(e, tb, level, &mut rng);
                }
            });
        }
    });

    check_consistency(&e, &tb);
}

#[test]
fn concurrent_mix_stays_consistent_under_serializable() {
    concurrent_workload(IsolationLevel::Serializable, 0x5E_71A1, 4, 30);
}

#[test]
fn concurrent_mix_stays_consistent_under_read_committed() {
    concurrent_workload(IsolationLevel::ReadCommitted, 0x4C_0117, 4, 30);
}

// ---- perf baseline ------------------------------------------------------------

/// One timed cell of the perf matrix: `workers` × `iters` spec-mix transactions at `level` against
/// `e` (already loaded with `tb`). The clock starts after the barrier (load excluded) and stops when
/// every worker finishes; the twelve consistency conditions are re-checked *outside* the timed
/// window, so a perf run is still a correctness run. Returns (committed transactions, seconds).
fn perf_cell(
    e: &BtreeEngine,
    tb: &Tables,
    level: IsolationLevel,
    seed: u64,
    workers: usize,
    iters: usize,
) -> (usize, f64) {
    perf_cell_shaped(e, tb, level, seed, workers, iters, false)
}

/// `home_bound = true` runs each worker as a spec-shaped TPC-C terminal pinned to its own home
/// warehouse (requires `warehouses >= workers`), so cross-worker row collisions vanish and the
/// cell measures ENGINE concurrency; `false` keeps the historical uniform-random pick (the
/// high-contention shape).
fn perf_cell_shaped(
    e: &BtreeEngine,
    tb: &Tables,
    level: IsolationLevel,
    seed: u64,
    workers: usize,
    iters: usize,
    home_bound: bool,
) -> (usize, f64) {
    assert!(!home_bound || tb.warehouses >= workers as i64);
    let attempts_before = ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed);
    let barrier = Barrier::new(workers + 1);
    let elapsed = thread::scope(|s| {
        for wkr in 0..workers {
            let barrier = &barrier;
            let mut tb = tb.clone();
            if home_bound {
                tb.home = Some(wkr as i64 % tb.warehouses);
            }
            s.spawn(move || {
                let tb = tb;
                let mut rng = Rng::new(seed.wrapping_add(wkr as u64 * 0x9e37_79b9));
                barrier.wait();
                for _ in 0..iters {
                    one_transaction(e, &tb, level, &mut rng);
                }
            });
        }
        barrier.wait();
        let start = std::time::Instant::now();
        // Workers run to completion when the scope joins; measure from release to full join by
        // re-entering the scope end. (The spawned threads are joined by `thread::scope` itself.)
        start
    })
    .elapsed()
    .as_secs_f64();
    check_consistency(e, tb);
    let attempts = ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed) - attempts_before;
    #[allow(
        clippy::cast_precision_loss,
        reason = "reporting only; attempt/commit counts are far below 2^52"
    )]
    let retry_factor = attempts as f64 / (workers * iters) as f64;
    eprintln!(
        "  attempts={attempts} for {} commits (x{retry_factor:.2} retry factor)",
        workers * iters,
    );
    (workers * iters, elapsed)
}

/// Measured TPC-C throughput baseline: the spec-mix workload, timed, across
/// {in-memory, durable-WAL} × {READ COMMITTED, SERIALIZABLE} × {1, 4 workers}. Numbers are
/// machine-relative — they are recorded with the hardware
/// spec, and the public-claim floor remains the cheap-VM run the plan mandates. Run via
/// `cargo tpcc-bench` (release; the debug gate skips it).
#[test]
#[ignore = "timed perf baseline; run via `cargo tpcc-bench` (release) and record in output_testing"]
fn tpcc_perf_baseline() {
    for (label, durable) in [("in-memory", false), ("durable-wal", true)] {
        for level in [IsolationLevel::ReadCommitted, IsolationLevel::Serializable] {
            for (workers, iters) in [(1usize, 1_000usize), (4, 250)] {
                let dir = tempfile::tempdir().unwrap();
                let e = if durable {
                    BtreeEngine::open(dir.path().join("tpcc.wal")).unwrap()
                } else {
                    BtreeEngine::new()
                };
                let tb = load(&e);
                let (tx, secs) = perf_cell(&e, &tb, level, 0xA5_31, workers, iters);
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "reporting only; tx counts are far below 2^52"
                )]
                let tps = tx as f64 / secs;
                eprintln!(
                    "tpcc-perf {label} {level:?} workers={workers}: {tps:.0} tx/s \
                     ({tx} tx in {secs:.2}s)"
                );
            }
        }
    }
}

/// Scaling-shape cell: the same spec mix over 8 warehouses — TPC-C's own rule (terminals
/// scale with warehouses), so at 4 workers row collisions are rare and the measurement isolates
/// ENGINE concurrency (latching) from workload contention. The 2-warehouse baseline above stays
/// as the high-contention (retry-storm) reference; this cell is where `w4 > w1` must show once
/// the global latch is gone. Record next to the baseline.
#[test]
#[ignore = "timed perf cell; run via `cargo test -p nusadb-btree --release --test test_tpcc tpcc_perf_scaled -- --ignored --nocapture`"]
fn tpcc_perf_scaled() {
    const SCALED_WAREHOUSES: i64 = 8;
    for (label, durable) in [("in-memory", false), ("durable-wal", true)] {
        for level in [IsolationLevel::ReadCommitted, IsolationLevel::Serializable] {
            for (workers, iters) in [(1usize, 1_000usize), (4, 250)] {
                let dir = tempfile::tempdir().unwrap();
                let e = if durable {
                    BtreeEngine::open(dir.path().join("tpcc.wal")).unwrap()
                } else {
                    BtreeEngine::new()
                };
                let tb = load_warehouses(&e, SCALED_WAREHOUSES);
                let (tx, secs) = perf_cell_shaped(&e, &tb, level, 0xA532, workers, iters, true);
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "reporting only; tx counts are far below 2^52"
                )]
                let tps = tx as f64 / secs;
                eprintln!(
                    "tpcc-scaled({SCALED_WAREHOUSES}wh) {label} {level:?} workers={workers}: \
                     {tps:.0} tx/s ({tx} tx in {secs:.2}s)"
                );
            }
        }
    }
}

// High-contention scale: 8 workers x 100 transactions hammer the same small schema (2 warehouses x
// 2 districts), so NewOrder's per-district `D_NEXT_O_ID` claim and Payment's warehouse/district YTD
// updates serialize heavily — a stress test for the lock manager + MVCC retry path. All 12
// consistency conditions must still hold, and the engine must converge (no livelock/deadlock).
#[test]
fn concurrent_high_contention_stays_consistent_under_serializable() {
    concurrent_workload(IsolationLevel::Serializable, 0x5CA1_5E11, 8, 100);
}

#[test]
fn concurrent_high_contention_stays_consistent_under_read_committed() {
    concurrent_workload(IsolationLevel::ReadCommitted, 0x5CA1_4C01, 8, 100);
}
