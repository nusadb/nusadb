//! In-memory [`PageStore`] backing the engine.
//!
//! Single-version and not yet durable: the redo WAL + double-write land, at which point
//! the disk-backed store from `nusadb-storage` plugs in behind the same trait.
//!
//! Latching: the directory (`Vec` of slots + free list) sits behind an `RwLock` that page
//! reads/writes only ever take in `read` mode — each slot carries its own `RwLock`, so distinct
//! pages are read and written fully in parallel and a same-page read/write pair is atomic at
//! page granularity (the property the engine's latch-free B-link readers lean on). Only
//! allocate/deallocate take the directory exclusively, and both are rare and O(1).

use std::sync::{Arc, Mutex, RwLock};

use nusadb_core::traits::Page;
use nusadb_core::{PAGE_SIZE, PageId, PageStore, Result};

/// One page slot: its own latch, shared out of the directory by `Arc` so a page operation never
/// holds the directory lock across the 8 KiB copy.
type Slot = Arc<RwLock<Page>>;

/// A `Vec`-backed page store: allocate appends, reads/writes index the vector. Freed pages go to
/// a free list and are reused before the vector grows.
#[derive(Debug, Default)]
pub struct MemPageStore {
    pages: RwLock<Vec<Slot>>,
    free: Mutex<Vec<PageId>>,
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard IS the critical section of its one-shot directory operation"
)]
impl MemPageStore {
    /// Pages currently allocated and not on the free list — observability for purge tests
    /// and ops counters.
    ///
    /// # Errors
    /// Fails only on a poisoned store lock.
    pub fn live_pages(&self) -> Result<usize> {
        let pages = self.pages.read().map_err(|_| poisoned())?;
        let free = self.free.lock().map_err(|_| poisoned())?;
        Ok(pages.len().saturating_sub(free.len()))
    }

    /// Total bytes of page memory the store holds resident. Every slot the backing vector has ever
    /// grown to keeps its `PAGE_SIZE` buffer — a freed page is zeroed and recycled through the free
    /// list, not dropped — so `vector length × PAGE_SIZE` is the store's real, monotonic RAM
    /// footprint. This is the metric a global memory guard bounds to reject growth gracefully before
    /// the OS OOM-kills the process.
    ///
    /// # Errors
    /// Fails only on a poisoned store lock.
    pub fn resident_bytes(&self) -> Result<u64> {
        let pages = self.pages.read().map_err(|_| poisoned())?;
        Ok((pages.len() as u64).saturating_mul(PAGE_SIZE as u64))
    }

    /// The slot for `id`, cloned out so the directory lock is released before the page copy.
    fn slot(&self, id: PageId) -> Result<Slot> {
        let index = usize::try_from(id.0).map_err(|_| bad_page(id))?;
        let pages = self.pages.read().map_err(|_| poisoned())?;
        pages.get(index).cloned().ok_or_else(|| bad_page(id))
    }
}

/// The store-level error for an out-of-range page id (a corruption-class bug, never expected).
fn bad_page(id: PageId) -> nusadb_core::Error {
    nusadb_core::Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("nusadb-btree: page {id:?} does not exist in the in-memory store"),
    ))
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each slot guard IS the critical section of its one-shot page operation"
)]
impl PageStore for MemPageStore {
    fn read_page(&self, id: PageId) -> Result<Page> {
        let slot = self.slot(id)?;
        let page = slot.read().map_err(|_| poisoned())?;
        Ok(*page)
    }

    fn write_page(&self, id: PageId, page: &Page) -> Result<()> {
        let slot = self.slot(id)?;
        let mut target = slot.write().map_err(|_| poisoned())?;
        *target = *page;
        Ok(())
    }

    fn allocate_page(&self) -> Result<PageId> {
        let recycled = self.free.lock().map_err(|_| poisoned())?.pop();
        if let Some(id) = recycled {
            return Ok(id);
        }
        let mut pages = self.pages.write().map_err(|_| poisoned())?;
        let id = PageId(u64::try_from(pages.len()).unwrap_or(u64::MAX));
        pages.push(Arc::new(RwLock::new([0u8; PAGE_SIZE])));
        Ok(id)
    }

    fn deallocate_page(&self, id: PageId) -> Result<()> {
        // Zero the slot (defensive: a stale reader bug surfaces as a decode error, not stale
        // data) and recycle the id.
        let slot = self.slot(id)?;
        {
            let mut page = slot.write().map_err(|_| poisoned())?;
            *page = [0u8; PAGE_SIZE];
        }
        self.free.lock().map_err(|_| poisoned())?.push(id);
        Ok(())
    }

    fn fsync(&self) -> Result<()> {
        Ok(()) // In-memory: nothing to make durable yet.
    }
}

/// A poisoned store lock means a prior panic mid-write; surface it as an I/O error rather than
/// unwrapping (production code must not panic).
fn poisoned() -> nusadb_core::Error {
    nusadb_core::Error::Io(std::io::Error::other(
        "nusadb-btree: page store lock poisoned by a previous panic",
    ))
}
