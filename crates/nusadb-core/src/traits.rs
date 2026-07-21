//! Ports (in the Ports & Adapters / Hexagonal Architecture sense) consumed by higher
//! layers.
//!
//! Production adapters live in `nusadb-storage`. Simulator adapters for deterministic
//! simulation testing (DST) live in `nusadb-sim`. The same use case code drives both.

use std::time::Duration;

use crate::{PageId, Result};

/// 8 KiB page slice — the canonical unit of physical storage I/O.
///
/// The exact representation is implementation-defined; downstream code treats it as
/// opaque bytes plus its header. See `nusadb-storage::page::Page` for the production
/// layout.
pub type Page = [u8; crate::PAGE_SIZE];

/// Page-level persistent storage.
///
/// All disk I/O in NusaDB goes through this trait. Production uses the `DiskManager`
/// adapter (in `nusadb-storage`); deterministic simulation uses `SimStorage`
/// (in `nusadb-sim`).
///
/// # Invariants
///
/// - Every page is exactly [`PAGE_SIZE`](crate::PAGE_SIZE) bytes.
/// - `write_page` followed by `read_page` for the same `PageId` must round-trip
///   identical bytes (in the absence of injected faults).
/// - After [`PageStore::fsync`] returns `Ok(())`, all prior writes are durable.
pub trait PageStore: Send + Sync {
    /// Read the page identified by `id`.
    fn read_page(&self, id: PageId) -> Result<Page>;

    /// Write `page` to the slot identified by `id`.
    fn write_page(&self, id: PageId, page: &Page) -> Result<()>;

    /// Allocate a fresh page; the returned `PageId` is exclusive to the caller.
    fn allocate_page(&self) -> Result<PageId>;

    /// Mark `id` free so a future [`allocate_page`](PageStore::allocate_page) may reuse it.
    ///
    /// The caller is responsible for ensuring no live data still references `id`. The default is
    /// a no-op: a store that never reclaims (e.g. an in-memory test double) simply leaves the slot
    /// allocated, which is harmless there. Disk-backed stores override this to recycle the slot.
    fn deallocate_page(&self, _id: PageId) -> Result<()> {
        Ok(())
    }

    /// Force all prior writes to durable storage. Returns only after the OS reports
    /// the data is on disk.
    fn fsync(&self) -> Result<()>;
}

/// Monotonic clock abstraction — replaces `std::time::Instant` in engine code.
///
/// Use [`SimClock`](https://docs.rs/nusadb-sim) in tests so DST runs are deterministic.
pub trait Clock: Send + Sync {
    /// Number of monotonic ticks since the clock was created.
    fn now_ticks(&self) -> u64;

    /// Advance virtual time (no-op for real clocks; advances the counter in `SimClock`).
    fn sleep(&self, duration: Duration);
}

/// Random number generator abstraction — replaces `rand::thread_rng` in engine code.
///
/// Seeded variants used in DST produce reproducible runs.
pub trait Rng: Send {
    /// Next 64-bit value.
    fn next_u64(&mut self) -> u64;

    /// Next value in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64;
}
