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

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

    /// How many page slots sit on the free list, recycled for the next allocation. Observability
    /// for tests that check pages come back.
    ///
    /// # Errors
    /// Fails only on a poisoned free-list lock.
    pub fn free_pages(&self) -> Result<usize> {
        Ok(self.free.lock().map_err(|_| poisoned())?.len())
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

/// The page section of a published checkpoint image.
///
/// The image holds only live pages: a directory of their ids in ascending order, then the pages
/// in that order, the `n`-th page at `offset + n * PAGE_SIZE`. Ids below `page_count` that the
/// directory does not name were free when the image was taken. Read-only: the image is complete
/// when it is named and is only ever replaced by the next checkpoint's rename.
#[derive(Debug)]
pub struct PageFile {
    file: File,
    offset: u64,
    page_count: u64,
    directory: Vec<u64>,
    /// CRC32 of each page, in directory order: a page whose bytes no longer match is refused.
    checksums: Vec<u32>,
}

impl PageFile {
    /// Open the page section of the image at `path`: the pages named by `directory` (ascending
    /// ids) stored from `offset`, within an id space of `page_count`, each checked against its
    /// entry in `checksums`.
    ///
    /// # Errors
    /// Propagates the open error.
    pub fn open(
        path: &Path,
        offset: u64,
        page_count: u64,
        directory: Vec<u64>,
        checksums: Vec<u32>,
    ) -> Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            offset,
            page_count,
            directory,
            checksums,
        })
    }

    /// Close the file, keeping what is needed to open the same section again.
    pub fn reopen_spec(self) -> PageFileSpec {
        PageFileSpec {
            offset: self.offset,
            page_count: self.page_count,
            directory: self.directory,
            checksums: self.checksums,
        }
    }

    /// The id space the image covers: the first id never handed out when it was taken.
    pub const fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Whether the image holds page `id`.
    fn holds(&self, id: u64) -> bool {
        self.directory.binary_search(&id).is_ok()
    }

    fn read(&self, id: u64) -> Result<Page> {
        let slot = self
            .directory
            .binary_search(&id)
            .map_err(|_| bad_page(PageId(id)))?;
        let mut page = [0u8; PAGE_SIZE];
        let at = u64::try_from(slot)
            .ok()
            .and_then(|slot| slot.checked_mul(PAGE_SIZE as u64))
            .and_then(|rel| self.offset.checked_add(rel))
            .ok_or_else(|| bad_page(PageId(id)))?;
        read_exact_at(&self.file, &mut page, at)?;
        let expected = self.checksums.get(slot).copied();
        if expected != Some(crc32fast::hash(&page)) {
            return Err(nusadb_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "nusadb-btree: page {id} of the checkpoint image fails its checksum; the image \
                     is damaged. Restore it from a backup or the archive"
                ),
            )));
        }
        Ok(page)
    }
}

/// A page section's layout without its open file: see [`PageFile::reopen_spec`].
#[derive(Debug)]
pub struct PageFileSpec {
    offset: u64,
    page_count: u64,
    directory: Vec<u64>,
    checksums: Vec<u32>,
}

impl PageFileSpec {
    /// Open the section again from the image at `path`.
    ///
    /// # Errors
    /// Propagates the open error.
    pub fn open(self, path: &Path) -> Result<PageFile> {
        PageFile::open(
            path,
            self.offset,
            self.page_count,
            self.directory,
            self.checksums,
        )
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, at)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], at + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read from the page section",
            ));
        }
        done += n;
    }
    Ok(())
}

/// A resident page: its own latch, whether it differs from the image's copy, and the clock's
/// second-chance bit.
#[derive(Debug)]
struct Frame {
    data: RwLock<Page>,
    dirty: AtomicBool,
    referenced: AtomicBool,
}

/// A page store whose pages live in a bounded cache over the page section of the last
/// published checkpoint image.
///
/// A page not resident is loaded from the image on first use; a page that differs from the image
/// (dirty) stays resident until the next checkpoint publishes an image that holds it, after which
/// every page is clean again. Under a capacity, clean pages are evicted by the clock (second-
/// chance) rule to make room; dirty pages are never evicted. The store itself never refuses a page:
/// a tree operation (a split writes several pages) must never stop half way, so a cache full of
/// dirty pages simply grows past its capacity, and the engine refuses new writes at an operation
/// boundary instead, from what cannot be evicted. Without a capacity, nothing is evicted and the
/// store behaves like an in-memory one that merely loads lazily. Without an image (the in-memory
/// engine, or a database before its first checkpoint) every page is resident.
///
/// Latching mirrors [`MemPageStore`]: the frame directory is read-locked for the lookup, each
/// frame carries its own latch, so distinct pages proceed in parallel and a same-page read and
/// write are atomic at page granularity, the property the engine's B-link readers lean on.
///
/// Eviction against writes: a writer takes the frame latch and then confirms the frame is still
/// the one registered for its id (retrying from the lookup if not), and marks it dirty under
/// that latch; the evictor removes a frame only while holding its latch without waiting for it
/// (`try_write`) and only if it is still clean. So a write never lands in a frame that has left
/// the cache. Lock order: the frame latch may be held while the directory is read-locked, never
/// the other way round with a wait (the evictor's latch attempt does not block), and the clock
/// is never taken while the directory is write-locked.
#[derive(Debug, Default)]
pub struct PagedStore {
    frames: RwLock<HashMap<u64, Arc<Frame>>>,
    /// Resident page ids in clock order; an id may be stale (evicted or freed) and is skipped.
    clock: Mutex<VecDeque<u64>>,
    /// Freed ids: the recycle list, and the same ids as a set so a read of a freed page is an
    /// error rather than a stale image page.
    free: Mutex<(Vec<u64>, HashSet<u64>)>,
    /// The first id never handed out.
    next_id: AtomicU64,
    /// Frames the cache may hold before evicting; `0` is unbounded.
    capacity_frames: AtomicU64,
    /// Resident frames that differ from the image.
    dirty_frames: AtomicU64,
    /// The image's page section, when the store is backed by one.
    file: RwLock<Option<PageFile>>,
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard IS the critical section of its one-shot directory operation"
)]
impl PagedStore {
    /// Bound the cache to `bytes` of page frames (`None`: unbounded). Clean pages are evicted to
    /// stay under it; dirty pages never leave the cache, so it can exceed the bound while they
    /// fill it (the engine refuses new writes at an operation boundary then).
    pub fn set_capacity_bytes(&self, bytes: Option<u64>) {
        let frames = bytes.map_or(0, |b| b / PAGE_SIZE as u64);
        self.capacity_frames.store(frames, Ordering::Release);
        // An unbounded cache keeps no clock (nothing is ever evicted); a bound makes every
        // resident page a candidate.
        if let Ok(mut clock) = self.clock.lock()
            && let Ok(resident) = self.frames.read()
        {
            clock.clear();
            if frames > 0 {
                clock.extend(resident.keys().copied());
            }
        }
    }

    /// Pages currently allocated and not on the free list.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn live_pages(&self) -> Result<usize> {
        let next = self.next_id.load(Ordering::Acquire);
        let free = self.free.lock().map_err(|_| poisoned())?;
        Ok(usize::try_from(next)
            .unwrap_or(usize::MAX)
            .saturating_sub(free.0.len()))
    }

    /// Bytes of page frames resident in the cache, dirty and clean.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn resident_bytes(&self) -> Result<u64> {
        let frames = self.frames.read().map_err(|_| poisoned())?;
        Ok((frames.len() as u64).saturating_mul(PAGE_SIZE as u64))
    }

    /// Bytes of resident frames that differ from the image and so cannot be evicted.
    pub fn dirty_bytes(&self) -> u64 {
        self.dirty_frames
            .load(Ordering::Acquire)
            .saturating_mul(PAGE_SIZE as u64)
    }

    /// How many page ids sit on the free list, recycled for the next allocation.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn free_pages(&self) -> Result<usize> {
        Ok(self.free.lock().map_err(|_| poisoned())?.0.len())
    }

    /// The first id never handed out: the page count a physical image must carry.
    pub fn page_count(&self) -> u64 {
        self.next_id.load(Ordering::Acquire)
    }

    /// Back the store with the page section of a published image: pages not resident are read
    /// from it, and every resident page is clean (the image holds it). Runs under the
    /// checkpoint's quiesce, or at open before any use.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn attach(&self, pages: PageFile) -> Result<()> {
        {
            let frames = self.frames.read().map_err(|_| poisoned())?;
            for frame in frames.values() {
                frame.dirty.store(false, Ordering::Release);
            }
            self.dirty_frames.store(0, Ordering::Release);
            // Every id the image leaves out, and that is not resident, is free: at open that is
            // the image's free set; after a checkpoint it is exactly the free list already held.
            let mut free = self.free.lock().map_err(|_| poisoned())?;
            for id in 0..pages.page_count {
                if !pages.holds(id) && !frames.contains_key(&id) && free.1.insert(id) {
                    free.0.push(id);
                }
            }
        }
        self.next_id.fetch_max(pages.page_count, Ordering::AcqRel);
        *self.file.write().map_err(|_| poisoned())? = Some(pages);
        Ok(())
    }

    /// Drop the backing page section (before the image it belongs to is replaced). Resident
    /// pages stay as they are; a page not resident cannot be read until the next `attach`.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn detach(&self) -> Result<Option<PageFile>> {
        Ok(self.file.write().map_err(|_| poisoned())?.take())
    }

    /// Evict clean pages until the cache is back under its capacity (after a checkpoint made
    /// every page clean).
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn trim(&self) -> Result<()> {
        self.make_room()
    }

    /// Put back the page section a failed publish detached: the image it belongs to is still
    /// the published one, so nothing resident became clean; dirty flags, the dirty count and
    /// the free list are left exactly as they are.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn reattach(&self, pages: PageFile) -> Result<()> {
        *self.file.write().map_err(|_| poisoned())? = Some(pages);
        Ok(())
    }

    /// The ids of every live page (allocated and not free), ascending: the directory a
    /// physical image carries.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn live_ids(&self) -> Result<Vec<u64>> {
        let count = self.page_count();
        let free = self.free.lock().map_err(|_| poisoned())?;
        Ok((0..count).filter(|id| !free.1.contains(id)).collect())
    }

    /// Write the pages named by `ids` to `out`, in order: the resident copy when there is one,
    /// the image's copy otherwise, each passed through `fix` first (on a copy; the cache is not
    /// touched). One page in memory at a time. Returns the CRC32 of each page as written.
    ///
    /// # Errors
    /// Propagates I/O errors; a live page found neither resident nor in the image is an error.
    pub fn write_pages_to(
        &self,
        out: &mut impl std::io::Write,
        ids: &[u64],
        fix: &dyn Fn(&mut Page),
    ) -> Result<Vec<u32>> {
        let file = self.file.read().map_err(|_| poisoned())?;
        let mut checksums = Vec::with_capacity(ids.len());
        for &id in ids {
            let resident = self
                .frames
                .read()
                .map_err(|_| poisoned())?
                .get(&id)
                .cloned();
            let mut page = if let Some(frame) = resident {
                *frame.data.read().map_err(|_| poisoned())?
            } else {
                match file.as_ref() {
                    Some(pages) if pages.holds(id) => pages.read(id)?,
                    _ => return Err(bad_page(PageId(id))),
                }
            };
            fix(&mut page);
            checksums.push(crc32fast::hash(&page));
            out.write_all(&page)?;
        }
        Ok(checksums)
    }

    fn resident(&self, id: u64) -> Result<Option<Arc<Frame>>> {
        Ok(self
            .frames
            .read()
            .map_err(|_| poisoned())?
            .get(&id)
            .cloned())
    }

    /// Make `page` resident under `id` as a new frame, evicting clean pages first when the cache
    /// is at its capacity. `None` when a frame for `id` is already resident (a racing load or
    /// write got there first); the caller then goes through that frame. When nothing can be
    /// evicted the frame goes in anyway: refusing here could stop a tree operation half way.
    fn insert_frame(&self, id: u64, page: &Page, dirty: bool) -> Result<Option<Arc<Frame>>> {
        self.make_room()?;
        let frame = Arc::new(Frame {
            data: RwLock::new(*page),
            dirty: AtomicBool::new(dirty),
            referenced: AtomicBool::new(true),
        });
        {
            let mut frames = self.frames.write().map_err(|_| poisoned())?;
            if frames.contains_key(&id) {
                return Ok(None);
            }
            frames.insert(id, Arc::clone(&frame));
            if dirty {
                self.dirty_frames.fetch_add(1, Ordering::AcqRel);
            }
        }
        if self.capacity_frames.load(Ordering::Acquire) > 0 {
            let mut clock = self.clock.lock().map_err(|_| poisoned())?;
            clock.push_back(id);
            // Stale entries (evicted, freed, or pushed again after a reuse) accumulate while the
            // cache is below capacity and no sweep runs; drop them once they dominate.
            if clock.len() > 64 {
                let frames = self.frames.read().map_err(|_| poisoned())?;
                if clock.len() > frames.len().saturating_mul(2) {
                    let mut seen = HashSet::with_capacity(frames.len());
                    clock.retain(|id| frames.contains_key(id) && seen.insert(*id));
                }
            }
        }
        Ok(Some(frame))
    }

    /// Whether `frame` is still the frame registered for `id`.
    fn is_registered(&self, id: u64, frame: &Arc<Frame>) -> Result<bool> {
        let frames = self.frames.read().map_err(|_| poisoned())?;
        Ok(frames.get(&id).is_some_and(|f| Arc::ptr_eq(f, frame)))
    }

    /// Evict clean pages by the clock rule until one frame is free under the capacity, or until
    /// nothing evictable remains.
    fn make_room(&self) -> Result<()> {
        let capacity = self.capacity_frames.load(Ordering::Acquire);
        if capacity == 0 {
            return Ok(());
        }
        loop {
            let len = self.frames.read().map_err(|_| poisoned())?.len() as u64;
            if len < capacity {
                return Ok(());
            }
            // Nothing is evictable while every resident page is dirty: skip the sweep.
            if self.dirty_frames.load(Ordering::Acquire) >= len {
                return Ok(());
            }
            let mut clock = self.clock.lock().map_err(|_| poisoned())?;
            let mut sweeps = clock.len().saturating_mul(2).saturating_add(1);
            let mut evicted = false;
            let mut frames = self.frames.write().map_err(|_| poisoned())?;
            while sweeps > 0 {
                sweeps -= 1;
                let Some(id) = clock.pop_front() else { break };
                let Some(frame) = frames.get(&id).cloned() else {
                    continue; // evicted or freed already
                };
                if frame.dirty.load(Ordering::Acquire)
                    || frame.referenced.swap(false, Ordering::AcqRel)
                {
                    clock.push_back(id);
                    continue;
                }
                // Evict only a frame no one is reading or writing right now, and only if it is
                // still clean under its latch: a writer marks dirty while holding it.
                let Ok(_latch) = frame.data.try_write() else {
                    clock.push_back(id);
                    continue;
                };
                if frame.dirty.load(Ordering::Acquire) {
                    clock.push_back(id);
                    continue;
                }
                frames.remove(&id);
                evicted = true;
                break;
            }
            drop(frames);
            drop(clock);
            if !evicted {
                return Ok(());
            }
        }
    }
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard IS the critical section of its one-shot page operation"
)]
impl PageStore for PagedStore {
    fn read_page(&self, id: PageId) -> Result<Page> {
        loop {
            if let Some(frame) = self.resident(id.0)? {
                frame.referenced.store(true, Ordering::Release);
                let page = frame.data.read().map_err(|_| poisoned())?;
                return Ok(*page);
            }
            if self.free.lock().map_err(|_| poisoned())?.1.contains(&id.0) {
                return Err(bad_page(id));
            }
            let page = {
                let file = self.file.read().map_err(|_| poisoned())?;
                match file.as_ref() {
                    Some(pages) if pages.holds(id.0) => pages.read(id.0)?,
                    _ => return Err(bad_page(id)),
                }
            };
            // The image's copy is what a fresh clean frame holds; return it directly. A racing
            // load or write that made the page resident first is read on the next pass.
            if self.insert_frame(id.0, &page, false)?.is_some() {
                return Ok(page);
            }
        }
    }

    fn write_page(&self, id: PageId, page: &Page) -> Result<()> {
        if id.0 >= self.next_id.load(Ordering::Acquire) {
            return Err(bad_page(id));
        }
        loop {
            if let Some(frame) = self.resident(id.0)? {
                let mut target = frame.data.write().map_err(|_| poisoned())?;
                // The frame may have been evicted between the lookup and the latch: then the
                // write belongs in whatever frame the id has now, so look it up again.
                if !self.is_registered(id.0, &frame)? {
                    continue;
                }
                *target = *page;
                frame.referenced.store(true, Ordering::Release);
                if !frame.dirty.swap(true, Ordering::AcqRel) {
                    self.dirty_frames.fetch_add(1, Ordering::AcqRel);
                }
                return Ok(());
            }
            if self.insert_frame(id.0, page, true)?.is_some() {
                return Ok(());
            }
        }
    }

    fn allocate_page(&self) -> Result<PageId> {
        let recycled = {
            let mut free = self.free.lock().map_err(|_| poisoned())?;
            let id = free.0.pop();
            if let Some(id) = id {
                free.1.remove(&id);
            }
            id
        };
        let id = recycled.unwrap_or_else(|| self.next_id.fetch_add(1, Ordering::AcqRel));
        let zero = [0u8; PAGE_SIZE];
        let inserted = match self.insert_frame(id, &zero, true) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => self.write_page(PageId(id), &zero),
            Err(e) => Err(e),
        };
        if let Err(e) = inserted {
            // Hand the id back so it is not lost to the store.
            let mut free = self.free.lock().map_err(|_| poisoned())?;
            if free.1.insert(id) {
                free.0.push(id);
            }
            return Err(e);
        }
        Ok(PageId(id))
    }

    fn deallocate_page(&self, id: PageId) -> Result<()> {
        if id.0 >= self.next_id.load(Ordering::Acquire) {
            return Err(bad_page(id));
        }
        {
            let mut frames = self.frames.write().map_err(|_| poisoned())?;
            if let Some(frame) = frames.remove(&id.0)
                && frame.dirty.load(Ordering::Acquire)
            {
                self.dirty_frames.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let mut free = self.free.lock().map_err(|_| poisoned())?;
        if free.1.insert(id.0) {
            free.0.push(id.0);
        }
        Ok(())
    }

    fn fsync(&self) -> Result<()> {
        Ok(()) // Durability is the log's and the image's; the cache itself is volatile.
    }
}
