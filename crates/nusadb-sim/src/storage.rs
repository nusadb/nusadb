//! In-memory [`PageStore`] + a deterministic fault-injection wrapper for DST.
//!
//! [`SimStorage`] is a faultless in-memory page store: the same engine code that runs against the
//! real `DiskManager` in production runs against this in tests. [`FaultingStorage`] wraps it and
//! injects torn writes, fsync failures, and power loss at rates set by [`FaultRates`], scheduled by
//! a seeded [`SimRng`] — so a fault scenario is a pure function of its seed and
//! reproduces exactly.

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};
use parking_lot::Mutex;

use crate::SimRng;

/// One fixed-size page; the `PageStore` treaty's `[u8; PAGE_SIZE]` (the `Page` alias is not
/// re-exported at the core crate root, so spell it out as the `DiskManager` adapter does).
type Page = [u8; PAGE_SIZE];

/// In-memory implementation of [`PageStore`].
///
/// Pages live in a `DashMap` keyed by page id; [`allocate_page`](PageStore::allocate_page) hands out
/// monotonically increasing ids and seeds each slot with a zero page (matching the `DiskManager`,
/// which zero-fills a freshly extended slot). It never injects faults — wrap it in
/// [`FaultingStorage`] for that. `fsync` is a no-op because memory is trivially durable.
#[derive(Debug, Default)]
pub struct SimStorage {
    pages: DashMap<u64, Page>,
    next_page: AtomicU64,
}

impl SimStorage {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently allocated pages (test introspection).
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether no pages are allocated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

fn missing_page(id: PageId) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("sim: read of unallocated page {id:?}"),
    ))
}

impl PageStore for SimStorage {
    fn read_page(&self, id: PageId) -> Result<Page> {
        self.pages
            .get(&id.0)
            .map_or_else(|| Err(missing_page(id)), |bytes| Ok(*bytes))
    }

    fn write_page(&self, id: PageId, page: &Page) -> Result<()> {
        self.pages.insert(id.0, *page);
        Ok(())
    }

    fn allocate_page(&self) -> Result<PageId> {
        let id = self.next_page.fetch_add(1, Ordering::SeqCst);
        self.pages.insert(id, [0u8; PAGE_SIZE]);
        Ok(PageId(id))
    }

    fn deallocate_page(&self, id: PageId) -> Result<()> {
        self.pages.remove(&id.0);
        Ok(())
    }

    fn fsync(&self) -> Result<()> {
        Ok(())
    }
}

/// Probabilities for injecting faults into a [`FaultingStorage`].
///
/// All rates default to `0.0` / `None`, so a default `FaultRates` injects nothing — a
/// `FaultingStorage` then behaves exactly like its inner [`SimStorage`].
#[derive(Debug, Clone, Default)]
pub struct FaultRates {
    /// Probability per write that the page is torn — the first half is written, the second half
    /// keeps its previous (pre-write) contents, modelling a write interrupted mid-page.
    pub torn_write: f64,
    /// Probability per write that the write *syscall itself* fails (ENOSPC / EIO): `write_page`
    /// returns `Err` and **nothing** persists. This is categorically different from `torn_write`
    /// (which returns `Ok` but leaves the page half-written): a disk-full write reports failure
    /// *before* acknowledging, so the caller learns the write never happened and takes its
    /// error-propagation / abort path rather than its crash-recovery path.
    pub write_fail: f64,
    /// Probability per fsync that it returns an error rather than confirming durability.
    pub fsync_fail: f64,
    /// If set, simulate power loss once this many total operations have run: every later write is
    /// silently discarded (the bytes never reach the store), as writes in flight are lost when
    /// power is cut. Reads still return whatever was durable before the loss.
    pub power_loss_after_ops: Option<u64>,
}

/// Wraps a [`SimStorage`] and injects faults per [`FaultRates`], scheduled by a seeded
/// [`SimRng`].
///
/// The whole fault schedule is a pure function of the seed and the rates, so a failing DST scenario
/// reproduces exactly from its seed. With a default [`FaultRates`] it is a transparent pass-through.
#[derive(Debug)]
pub struct FaultingStorage {
    inner: SimStorage,
    rng: Mutex<SimRng>,
    rates: FaultRates,
    ops: AtomicU64,
}

impl FaultingStorage {
    /// Wrap `inner`, injecting `rates` on a schedule seeded by `seed`.
    #[must_use]
    pub const fn new(inner: SimStorage, rates: FaultRates, seed: u64) -> Self {
        Self {
            inner,
            rng: Mutex::new(SimRng::new(seed)),
            rates,
            ops: AtomicU64::new(0),
        }
    }

    /// Borrow the underlying faultless store (e.g. to assert what actually persisted).
    #[must_use]
    pub const fn inner(&self) -> &SimStorage {
        &self.inner
    }

    /// Draw the next fault decision for probability `p`. Returns `false` without consuming entropy
    /// when `p <= 0.0`, so an unconfigured fault category never perturbs the schedule of the others.
    fn roll(&self, p: f64) -> bool {
        use nusadb_core::Rng;
        p > 0.0 && self.rng.lock().next_f64() < p
    }

    /// Count one operation and report whether simulated power has already been lost.
    fn powered_off(&self) -> bool {
        let count = self.ops.fetch_add(1, Ordering::SeqCst);
        self.rates
            .power_loss_after_ops
            .is_some_and(|limit| count >= limit)
    }
}

impl PageStore for FaultingStorage {
    fn read_page(&self, id: PageId) -> Result<Page> {
        let _ = self.powered_off(); // count the op; reads survive a power cut
        self.inner.read_page(id)
    }

    fn write_page(&self, id: PageId, page: &Page) -> Result<()> {
        if self.powered_off() {
            return Ok(()); // write lost to the power cut — never reaches the store
        }
        if self.roll(self.rates.write_fail) {
            // Disk-full/EIO: the write syscall fails before acknowledging, so nothing persists.
            // Modelled with the portable `StorageFull` kind, mapping to `Error::Io` exactly as a
            // real `DiskManager` write does (via `?` from `std::io`).
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "sim: injected ENOSPC write failure",
            )));
        }
        if self.roll(self.rates.torn_write) {
            // First half written; second half keeps its pre-write contents (zero if never written).
            // Iterator copy (not range-slice) so the fixed-size bound is obvious and panic-free.
            let mut torn = self.inner.read_page(id).unwrap_or([0u8; PAGE_SIZE]);
            let half = PAGE_SIZE / 2;
            torn.iter_mut()
                .zip(page.iter())
                .take(half)
                .for_each(|(dst, &src)| *dst = src);
            return self.inner.write_page(id, &torn);
        }
        self.inner.write_page(id, page)
    }

    fn allocate_page(&self) -> Result<PageId> {
        let _ = self.powered_off();
        self.inner.allocate_page()
    }

    fn deallocate_page(&self, id: PageId) -> Result<()> {
        let _ = self.powered_off();
        self.inner.deallocate_page(id)
    }

    fn fsync(&self) -> Result<()> {
        if self.powered_off() {
            return Ok(());
        }
        if self.roll(self.rates.fsync_fail) {
            return Err(Error::FsyncFailed("sim: injected fsync failure".to_owned()));
        }
        self.inner.fsync()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test assertions unwrap known-good results and slice fixed-size page halves"
    )]

    use super::*;

    #[test]
    fn sim_storage_allocates_reads_writes_round_trip() {
        let store = SimStorage::new();
        let id = store.allocate_page().unwrap();
        assert_eq!(
            store.read_page(id).unwrap(),
            [0u8; PAGE_SIZE],
            "fresh = zero"
        );
        let mut page = [0u8; PAGE_SIZE];
        page[0] = 0xAB;
        page[PAGE_SIZE - 1] = 0xCD;
        store.write_page(id, &page).unwrap();
        assert_eq!(store.read_page(id).unwrap(), page);
        assert_eq!(store.len(), 1);
        store.deallocate_page(id).unwrap();
        assert!(store.read_page(id).is_err(), "deallocated page is gone");
    }

    #[test]
    fn no_rates_is_a_transparent_passthrough() {
        let store = FaultingStorage::new(SimStorage::new(), FaultRates::default(), 1);
        let id = store.allocate_page().unwrap();
        let page = [0x42u8; PAGE_SIZE];
        store.write_page(id, &page).unwrap();
        store.fsync().unwrap();
        assert_eq!(store.read_page(id).unwrap(), page);
    }

    #[test]
    fn torn_write_keeps_first_half_and_loses_second() {
        let rates = FaultRates {
            torn_write: 1.0, // always tear
            ..FaultRates::default()
        };
        let store = FaultingStorage::new(SimStorage::new(), rates, 7);
        let id = store.allocate_page().unwrap();
        store.write_page(id, &[0xFFu8; PAGE_SIZE]).unwrap();
        let got = store.read_page(id).unwrap();
        let half = PAGE_SIZE / 2;
        assert!(got[..half].iter().all(|&b| b == 0xFF), "first half written");
        assert!(got[half..].iter().all(|&b| b == 0x00), "second half lost");
    }

    #[test]
    fn fsync_fail_rate_one_always_errors() {
        let rates = FaultRates {
            fsync_fail: 1.0,
            ..FaultRates::default()
        };
        let store = FaultingStorage::new(SimStorage::new(), rates, 3);
        assert!(matches!(store.fsync(), Err(Error::FsyncFailed(_))));
    }

    #[test]
    fn write_fail_rate_one_errors_and_persists_nothing() {
        let rates = FaultRates {
            write_fail: 1.0, // every write reports ENOSPC
            ..FaultRates::default()
        };
        let store = FaultingStorage::new(SimStorage::new(), rates, 5);
        let id = store.allocate_page().unwrap();
        // The write fails with an ENOSPC-shaped I/O error — unlike a torn write, it returns `Err`.
        let err = store.write_page(id, &[0x99u8; PAGE_SIZE]).unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::StorageFull),
            "write_fail surfaces a StorageFull I/O error, got {err:?}"
        );
        // Nothing persisted: the page still reads as freshly-allocated zeros, not a half-write.
        assert_eq!(
            store.inner().read_page(id).unwrap(),
            [0u8; PAGE_SIZE],
            "a failed write leaves the page untouched"
        );
    }

    #[test]
    fn power_loss_discards_writes_after_the_threshold() {
        let rates = FaultRates {
            power_loss_after_ops: Some(2),
            ..FaultRates::default()
        };
        let store = FaultingStorage::new(SimStorage::new(), rates, 1);
        let id = store.allocate_page().unwrap(); // op 0
        store.write_page(id, &[0x11u8; PAGE_SIZE]).unwrap(); // op 1 — persists
        // op 2 reaches the threshold → this write is lost.
        store.write_page(id, &[0x22u8; PAGE_SIZE]).unwrap();
        assert_eq!(
            store.read_page(id).unwrap(),
            [0x11u8; PAGE_SIZE],
            "the post-power-loss write never persisted"
        );
    }

    #[test]
    fn same_seed_reproduces_the_same_fault_schedule() {
        let make = || {
            FaultingStorage::new(
                SimStorage::new(),
                FaultRates {
                    torn_write: 0.5,
                    ..FaultRates::default()
                },
                99,
            )
        };
        let run = |store: &FaultingStorage| {
            let mut outcomes = Vec::new();
            for _ in 0..32 {
                let id = store.allocate_page().unwrap();
                store.write_page(id, &[0x77u8; PAGE_SIZE]).unwrap();
                // A torn write leaves the second half zero; record the outcome per page.
                outcomes.push(store.read_page(id).unwrap()[PAGE_SIZE - 1] == 0x00);
            }
            outcomes
        };
        assert_eq!(
            run(&make()),
            run(&make()),
            "same seed → identical fault schedule"
        );
    }
}
