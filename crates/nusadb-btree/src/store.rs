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
use nusadb_wal::Sealer;

use crate::crypt::{DataKey, SEALED_PAGE_SIZE, spill_aad};

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

/// Where a published checkpoint image keeps its pages: one or more read-only segment files.
///
/// An image of the single-file format holds its own pages, the `n`-th live page at `offset + n *
/// PAGE_SIZE` of the image file: one segment. A segmented image names segment files in the pages
/// directory beside the log; each live page sits at a slot of one of them, and a checkpoint only
/// writes the pages that changed since the image before it into a new segment, naming the older
/// segments for the rest. Every segment is complete before an image names it and is never
/// rewritten. Ids below `page_count` that the directory does not name were free when the image
/// was taken.
#[derive(Debug)]
pub struct PageFile {
    segments: Vec<Segment>,
    page_count: u64,
    /// Live page ids, ascending.
    directory: Vec<u64>,
    /// Where each page of `directory` is, in the same order.
    locations: Vec<Location>,
    /// CRC32 of each page, in directory order: a page whose bytes no longer match is refused.
    checksums: Vec<u32>,
    /// The data key the pages are sealed under; `None` for an unencrypted database.
    key: Option<Arc<DataKey>>,
}

/// One file of pages.
#[derive(Debug)]
struct Segment {
    path: std::path::PathBuf,
    /// The segment's name in the pages directory; `None` for the pages of a single-file image.
    name: Option<String>,
    file: File,
    /// The byte offset of slot 0.
    base: u64,
    /// Slots the file holds.
    slots: u64,
    /// Bytes per slot: a page, or a sealed page.
    slot_bytes: u64,
}

/// Where one page of an image is: a segment and the slot within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    /// Index into the image's segment list.
    pub segment: u32,
    /// Page slot within the segment.
    pub slot: u32,
}

/// An unchanged page's place in the current image, to be named again by the next one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Carried {
    /// The segment's index in the current image's list (see [`PagedStore::image_segments`]).
    pub segment: u32,
    /// Page slot within the segment.
    pub slot: u32,
    /// The page's CRC32.
    pub checksum: u32,
}

impl PageFile {
    /// Open the page section of a single-file image at `path`: the pages named by `directory`
    /// (ascending ids) stored from `offset`, within an id space of `page_count`, each checked
    /// against its entry in `checksums`.
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
        let slots = directory.len() as u64;
        let locations =
            (0..directory.len())
                .map(|n| {
                    u32::try_from(n).map(|slot| Location { segment: 0, slot }).map_err(|_| {
                    nusadb_core::Error::Io(std::io::Error::other(
                        "nusadb-btree: the image holds more pages than one segment can address",
                    ))
                })
                })
                .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            segments: vec![Segment {
                path: path.to_path_buf(),
                name: None,
                file: File::open(path)?,
                base: offset,
                slots,
                slot_bytes: PAGE_SIZE as u64,
            }],
            page_count,
            directory,
            locations,
            checksums,
            key: None,
        })
    }

    /// Open the pages of a segmented image: `names` are segment files in `dir`, and the `n`-th
    /// page of `directory` sits at `locations[n]`. With `key`, every page is sealed under it.
    ///
    /// # Errors
    /// Fails when a named segment is missing (the image cannot be served without it) or on
    /// other open errors.
    pub fn open_segments(
        dir: &Path,
        names: &[String],
        page_count: u64,
        directory: Vec<u64>,
        locations: Vec<Location>,
        checksums: Vec<u32>,
        key: Option<Arc<DataKey>>,
    ) -> Result<Self> {
        let slot_bytes = if key.is_some() {
            SEALED_PAGE_SIZE as u64
        } else {
            PAGE_SIZE as u64
        };
        let mut segments = Vec::with_capacity(names.len());
        for name in names {
            let path = segment_path(dir, name);
            let file = File::open(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    nusadb_core::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!(
                            "nusadb-btree: the checkpoint image names the page segment {} but it \
                             is missing; copy the image together with its pages directory",
                            path.display()
                        ),
                    ))
                } else {
                    e.into()
                }
            })?;
            let slots = file.metadata()?.len() / slot_bytes;
            segments.push(Segment {
                path,
                name: Some(name.clone()),
                file,
                base: 0,
                slots,
                slot_bytes,
            });
        }
        Ok(Self {
            segments,
            page_count,
            directory,
            locations,
            checksums,
            key,
        })
    }

    /// Close the files, keeping what is needed to open the same section again.
    pub fn reopen_spec(self) -> PageFileSpec {
        PageFileSpec {
            segments: self
                .segments
                .into_iter()
                .map(|s| (s.path, s.name, s.base, s.slots, s.slot_bytes))
                .collect(),
            page_count: self.page_count,
            directory: self.directory,
            locations: self.locations,
            checksums: self.checksums,
            key: self.key,
        }
    }

    /// The id space the image covers: the first id never handed out when it was taken.
    pub const fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Each segment this image reads from, in order: its name (`None` for the pages of a
    /// single-file image) and the slots it holds.
    pub fn segment_list(&self) -> Vec<(Option<String>, u64)> {
        self.segments
            .iter()
            .map(|s| (s.name.clone(), s.slots))
            .collect()
    }

    /// Whether the image holds page `id`.
    fn holds(&self, id: u64) -> bool {
        self.directory.binary_search(&id).is_ok()
    }

    /// Where page `id` is, when it lies in a named segment.
    fn carried(&self, id: u64) -> Option<Carried> {
        let n = self.directory.binary_search(&id).ok()?;
        let location = self.locations.get(n)?;
        // Only a page in a named segment can be named again by the next image.
        self.segments
            .get(location.segment as usize)?
            .name
            .as_ref()?;
        Some(Carried {
            segment: location.segment,
            slot: location.slot,
            checksum: *self.checksums.get(n)?,
        })
    }

    pub(crate) fn read(&self, id: u64) -> Result<Page> {
        let n = self
            .directory
            .binary_search(&id)
            .map_err(|_| bad_page(PageId(id)))?;
        let location = self.locations.get(n).ok_or_else(|| bad_page(PageId(id)))?;
        let segment = self
            .segments
            .get(location.segment as usize)
            .ok_or_else(|| bad_page(PageId(id)))?;
        let mut page = [0u8; PAGE_SIZE];
        let at = u64::from(location.slot)
            .checked_mul(segment.slot_bytes)
            .and_then(|rel| segment.base.checked_add(rel))
            .ok_or_else(|| bad_page(PageId(id)))?;
        if let Some(key) = &self.key {
            let mut sealed = vec![0u8; SEALED_PAGE_SIZE];
            read_exact_at(&segment.file, &mut sealed, at)?;
            let name = segment.name.as_deref().unwrap_or_default();
            if !key.open_page(id, location.slot, name, &sealed, &mut page) {
                return Err(nusadb_core::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "nusadb-btree: page {id} in the segment {} fails authentication: it was \
                         altered, moved, or written under another key. Restore the database \
                         from a backup or the archive",
                        segment.path.display()
                    ),
                )));
            }
        } else {
            read_exact_at(&segment.file, &mut page, at)?;
        }
        let expected = self.checksums.get(n).copied();
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

/// The file of the segment `name` in the pages directory `dir`.
pub fn segment_path(dir: &Path, name: &str) -> std::path::PathBuf {
    dir.join(format!("{name}.seg"))
}

/// A page section's layout without its open files: see [`PageFile::reopen_spec`].
#[derive(Debug)]
pub struct PageFileSpec {
    segments: Vec<(std::path::PathBuf, Option<String>, u64, u64, u64)>,
    page_count: u64,
    directory: Vec<u64>,
    locations: Vec<Location>,
    checksums: Vec<u32>,
    key: Option<Arc<DataKey>>,
}

impl PageFileSpec {
    /// Open the section again from the same files.
    ///
    /// # Errors
    /// Propagates the open error.
    pub fn open(self) -> Result<PageFile> {
        let mut segments = Vec::with_capacity(self.segments.len());
        for (path, name, base, slots, slot_bytes) in self.segments {
            segments.push(Segment {
                file: File::open(&path)?,
                path,
                name,
                base,
                slots,
                slot_bytes,
            });
        }
        Ok(PageFile {
            segments,
            page_count: self.page_count,
            directory: self.directory,
            locations: self.locations,
            checksums: self.checksums,
            key: self.key,
        })
    }
}

#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, at)
}

#[cfg(windows)]
fn write_all_at(file: &File, buf: &[u8], at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_write(&buf[done..], at + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "short write to the spill file",
            ));
        }
        done += n;
    }
    Ok(())
}

/// Changed pages that left memory: a scratch file of page slots and where each page sits. Never
/// durable and never read by recovery (the log and the image are the durable copies); it only
/// keeps a changed page off the heap until the next checkpoint writes it into an image.
#[derive(Debug, Default)]
struct Spill {
    file: Option<File>,
    /// Page id to its slot and the generation of the copy written there.
    slots: HashMap<u64, (u64, u64)>,
    free_slots: Vec<u64>,
    next_slot: u64,
    /// Stamped on every copy written, so a reader can tell whether the copy it loaded is still
    /// the latest one.
    next_generation: u64,
    /// The last write failed: the warning is logged once per run of failures, not per page.
    failing: bool,
    /// Seals every page written, under a key that exists only in this process; `None` writes
    /// pages as they are.
    seal: Option<Sealer>,
}

/// Where the current copy of a page is, as `locate` found it.
enum Located {
    Resident(Arc<Frame>),
    Spilled(Box<Page>, u64),
    Image,
}

/// Where a copy of a page handed to `insert_frame` came from, which decides whether it may
/// become the resident copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// Written by the caller: it is the newest copy by definition.
    Fresh,
    /// Read from the image: valid only while no spilled copy exists.
    Image,
    /// Read from the spill with this generation: valid only while that copy is still the latest.
    Spill(u64),
}

impl Spill {
    /// Write `page` for `id` into a slot. Fails only on I/O.
    fn put(&mut self, id: u64, page: &Page) -> std::io::Result<()> {
        let Some(file) = &self.file else {
            return Err(std::io::Error::other("no spill file"));
        };
        let generation = self.next_generation + 1;
        let sealed;
        let bytes: &[u8] = match &self.seal {
            Some(sealer) => {
                sealed = sealer
                    .seal(&spill_aad(id, generation), page)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                &sealed
            },
            None => page,
        };
        let existing = self.slots.get(&id).map(|&(slot, _)| slot);
        let slot = match existing {
            Some(slot) => slot,
            None => self.free_slots.pop().unwrap_or_else(|| {
                let slot = self.next_slot;
                self.next_slot += 1;
                slot
            }),
        };
        if let Err(e) = write_all_at(file, bytes, slot * self.slot_bytes()) {
            // A slot taken for this write goes back, so a failed write leaks no file space.
            if existing.is_none() {
                self.free_slots.push(slot);
            }
            if !self.failing {
                self.failing = true;
                tracing::warn!(
                    error = %e,
                    "could not write a changed page to the spill file; changed pages stay in \
                     memory until writes succeed again"
                );
            }
            return Err(e);
        }
        self.failing = false;
        self.next_generation = generation;
        self.slots.insert(id, (slot, generation));
        Ok(())
    }

    /// Bytes per slot: a page, or a sealed page.
    const fn slot_bytes(&self) -> u64 {
        if self.seal.is_some() {
            SEALED_PAGE_SIZE as u64
        } else {
            PAGE_SIZE as u64
        }
    }

    /// The spilled copy of `id` and its generation, if there is one.
    fn get(&self, id: u64) -> std::io::Result<Option<(Page, u64)>> {
        let (Some(file), Some(&(slot, generation))) = (&self.file, self.slots.get(&id)) else {
            return Ok(None);
        };
        let mut page = [0u8; PAGE_SIZE];
        let at = slot * self.slot_bytes();
        match &self.seal {
            Some(sealer) => {
                let mut sealed = vec![0u8; SEALED_PAGE_SIZE];
                read_exact_at(file, &mut sealed, at)?;
                if !sealer.open_into(&spill_aad(id, generation), &sealed, &mut page) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("page {id} in the spill file fails authentication"),
                    ));
                }
            },
            None => read_exact_at(file, &mut page, at)?,
        }
        Ok(Some((page, generation)))
    }

    /// Whether a copy of `id` from `source` is still the latest one.
    fn admits(&self, id: u64, source: Source) -> bool {
        match source {
            Source::Fresh => true,
            Source::Image => !self.slots.contains_key(&id),
            Source::Spill(generation) => self.slots.get(&id).is_some_and(|&(_, g)| g == generation),
        }
    }

    /// Forget `id`'s spilled copy (it is resident again, freed, or in a new image).
    fn remove(&mut self, id: u64) {
        if let Some((slot, _)) = self.slots.remove(&id) {
            self.free_slots.push(slot);
        }
    }

    /// Forget every spilled copy and give the file's space back.
    fn clear(&mut self) {
        self.slots.clear();
        self.free_slots.clear();
        self.next_slot = 0;
        if let Some(file) = &self.file {
            let _ = file.set_len(0);
        }
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

/// How many clock entries one eviction looks at for a clean page before it spills a changed
/// one instead (when a spill is enabled).
const SWEEP_LIMIT: usize = 512;

/// How many times one eviction tries again after its victim was taken or changed meanwhile.
const EVICT_RETRIES: u32 = 8;

/// What a clock sweep does with an unreferenced changed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirtyPolicy {
    /// Pass it by: only a clean frame may leave.
    Skip,
    /// Keep the first one to spill if the sweep finds no clean frame.
    Fallback,
    /// Take it: every resident frame is changed.
    Take,
}

/// A frame a sweep picked to leave the cache, and whether it leaves by spilling.
#[derive(Debug, Clone, Copy)]
struct Victim {
    id: u64,
    spill: bool,
}

/// How one eviction attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eviction {
    /// The frame left (or was gone already).
    Done,
    /// The frame was latched or changed after it was picked; it stays.
    Busy,
    /// Writing the frame to the spill failed; it stays.
    SpillFailed,
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
/// (dirty) stays out of the image until the next checkpoint publishes one that holds it, after
/// which every page is clean again. Under a capacity, pages are evicted by the clock
/// (second-chance) rule to make room. A clean page simply leaves; when a short sweep finds none
/// and a spill file is enabled, a dirty page is written there and leaves memory; it comes back
/// from there as a dirty page. Without a spill file only clean pages leave. The spill is scratch
/// space, never durable and never read by recovery. The store itself never refuses a page: a
/// tree operation (a split writes several pages) must never stop half way, so without a spill a
/// cache full of dirty pages simply grows past its capacity, and the engine refuses new writes
/// at an operation boundary instead, from what cannot be evicted. Without a capacity, nothing is
/// evicted and the store behaves like an in-memory one that merely loads lazily. Without an
/// image (the in-memory engine, or a database before its first checkpoint) every page is
/// resident.
///
/// Latching mirrors [`MemPageStore`]: the frame directory is read-locked for the lookup, each
/// frame carries its own latch, so distinct pages proceed in parallel and a same-page read and
/// write are atomic at page granularity, the property the engine's B-link readers lean on.
///
/// Eviction against writes: a writer takes the frame latch and then confirms the frame is still
/// the one registered for its id (retrying from the lookup if not), and marks it dirty under
/// that latch; the evictor removes a frame only while holding its latch without waiting for it
/// (`try_write`) and only if it is still clean, or after writing it to the spill when it was
/// picked to spill. So a write never lands in a frame that has left the cache. Lock order: the frame latch may be held while the directory is read-locked, never
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
    /// Changed pages evicted to scratch space. Taken after the directory and a frame latch,
    /// never before them.
    spill: Mutex<Spill>,
    /// The database's exclusive lock file, when the store belongs to a database opened on a
    /// path: held as long as the store, so as long as anything (a scan outliving its engine,
    /// say) can still reach the database's files.
    lock: Mutex<Option<File>>,
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard IS the critical section of its one-shot directory operation"
)]
impl PagedStore {
    /// Bound the cache to `bytes` of page frames (`None`: unbounded). Clean pages are evicted to
    /// stay under it, and dirty pages spill when a spill file is enabled. Without one, dirty pages
    /// stay, so the cache can exceed the bound while they fill it (the engine refuses new writes
    /// at an operation boundary then).
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

    /// Run `f` on page `id` where it sits, under its frame's read latch, without copying it out.
    /// A page not resident is loaded first (and read from that copy). The caller must keep every
    /// writer of this page out while `f` runs (the engine's index latch does), since a frame
    /// evicted meanwhile is not looked up again; `f` may read other pages of the store but must
    /// not write this one.
    ///
    /// # Errors
    /// Propagates the load's errors.
    pub fn with_page<R>(&self, id: PageId, f: impl FnOnce(&Page) -> R) -> Result<R> {
        if let Some(frame) = self.resident(id.0)? {
            frame.referenced.store(true, Ordering::Release);
            let page = frame.data.read().map_err(|_| poisoned())?;
            return Ok(f(&page));
        }
        let page = self.read_page(id)?;
        Ok(f(&page))
    }

    /// Change page `id` where it sits, under its frame's write latch: `f` returns its result and
    /// whether it changed the page, which then counts as changed. `f` must not touch the store.
    ///
    /// # Errors
    /// Propagates the load's errors.
    pub fn modify_page<R>(&self, id: PageId, f: impl FnOnce(&mut Page) -> (R, bool)) -> Result<R> {
        if id.0 >= self.next_id.load(Ordering::Acquire) {
            return Err(bad_page(id));
        }
        let mut f = Some(f);
        loop {
            if let Some(frame) = self.resident(id.0)? {
                let mut target = frame.data.write().map_err(|_| poisoned())?;
                // Evicted between the lookup and the latch: find the frame the id has now.
                if !self.is_registered(id.0, &frame)? {
                    continue;
                }
                let Some(f) = f.take() else {
                    return Err(bad_page(id));
                };
                let (out, changed) = f(&mut target);
                frame.referenced.store(true, Ordering::Release);
                if changed && !frame.dirty.swap(true, Ordering::AcqRel) {
                    self.dirty_frames.fetch_add(1, Ordering::AcqRel);
                }
                return Ok(out);
            }
            // Make it resident, then change it in place on the next pass.
            self.read_page(id)?;
        }
    }

    /// Hold the database's lock `file` for as long as this store lives.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn hold_lock(&self, file: File) -> Result<()> {
        *self.lock.lock().map_err(|_| poisoned())? = Some(file);
        Ok(())
    }

    /// Let changed pages leave memory for `file` (a scratch file the store owns from now on)
    /// when no clean page can be evicted. Without it a changed page stays resident until the
    /// next checkpoint.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn enable_spill(&self, file: File) -> Result<()> {
        let mut spill = self.spill.lock().map_err(|_| poisoned())?;
        spill.file = Some(file);
        Ok(())
    }

    /// [`enable_spill`](Self::enable_spill), sealing every page written to `file` with
    /// `sealer` (a key that never leaves the process: the spill is never read after it ends).
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn enable_sealed_spill(&self, file: File, sealer: Sealer) -> Result<()> {
        let mut spill = self.spill.lock().map_err(|_| poisoned())?;
        if !spill.slots.is_empty() {
            return Err(nusadb_core::Error::Io(std::io::Error::other(
                "nusadb-btree: the spill already holds pages",
            )));
        }
        spill.file = Some(file);
        spill.seal = Some(sealer);
        Ok(())
    }

    /// Keep changed pages in memory from now on (they stay until the next checkpoint). Has no
    /// effect once a page has been spilled, since the spill then holds its only copy.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn disable_spill(&self) -> Result<()> {
        let mut spill = self.spill.lock().map_err(|_| poisoned())?;
        if spill.slots.is_empty() {
            spill.file = None;
        }
        Ok(())
    }

    /// Whether changed pages may leave memory.
    pub fn can_spill(&self) -> bool {
        self.spill.lock().is_ok_and(|spill| spill.file.is_some())
    }

    /// Bytes of changed pages held in the spill file.
    pub fn spilled_bytes(&self) -> u64 {
        self.spill.lock().map_or(0, |spill| {
            (spill.slots.len() as u64).saturating_mul(PAGE_SIZE as u64)
        })
    }

    /// Bytes of resident frames that differ from the image: they leave memory only by spilling.
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
        // Every spilled page was written into this image: nothing is spilled any more.
        self.spill.lock().map_err(|_| poisoned())?.clear();
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

    /// For each id of `ids`, where the current image already holds it unchanged: not changed
    /// since that image (neither dirty nor spilled) and stored in a named segment. `None` for a
    /// page the next image must write. Runs under the checkpoint's quiesce.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn carried_locations(&self, ids: &[u64]) -> Result<Vec<Option<Carried>>> {
        let file = self.file.read().map_err(|_| poisoned())?;
        let Some(pages) = file.as_ref() else {
            return Ok(vec![None; ids.len()]);
        };
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            let unchanged = match self.locate(id)? {
                Located::Resident(frame) => !frame.dirty.load(Ordering::Acquire),
                Located::Spilled(..) => false,
                Located::Image => true,
            };
            out.push(if unchanged { pages.carried(id) } else { None });
        }
        Ok(out)
    }

    /// The segments the current image reads from, in its own order (a [`Carried`] segment is an
    /// index into this list): each one's name (`None` for the pages of a single-file image) and
    /// the slots it holds.
    ///
    /// # Errors
    /// Fails only on a poisoned lock.
    pub fn image_segments(&self) -> Result<Vec<(Option<String>, u64)>> {
        let file = self.file.read().map_err(|_| poisoned())?;
        Ok(file.as_ref().map_or_else(Vec::new, PageFile::segment_list))
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
        self.each_page(ids, fix, &mut |_, _, page| {
            out.write_all(page)?;
            Ok(())
        })
    }

    /// [`write_pages_to`](Self::write_pages_to) for a new segment of an encrypted database:
    /// the `n`-th page is sealed under `key` for slot `n` of the segment `segment`. The CRC32s
    /// returned are of the plain pages.
    ///
    /// # Errors
    /// As [`write_pages_to`](Self::write_pages_to).
    pub fn write_sealed_pages_to(
        &self,
        out: &mut impl std::io::Write,
        ids: &[u64],
        fix: &dyn Fn(&mut Page),
        key: &DataKey,
        segment: &str,
    ) -> Result<Vec<u32>> {
        let mut sealed = Vec::with_capacity(SEALED_PAGE_SIZE);
        self.each_page(ids, fix, &mut |n, id, page| {
            let slot = u32::try_from(n).map_err(|_| {
                nusadb_core::Error::Io(std::io::Error::other(
                    "nusadb-btree: too many pages for one segment",
                ))
            })?;
            sealed.clear();
            key.seal_page(id, slot, segment, page, &mut sealed)?;
            out.write_all(&sealed)?;
            Ok(())
        })
    }

    /// Hand each page named by `ids` to `emit` in order, with its position and id, after `fix`
    /// (see [`write_pages_to`](Self::write_pages_to)); returns each page's CRC32.
    fn each_page(
        &self,
        ids: &[u64],
        fix: &dyn Fn(&mut Page),
        emit: &mut dyn FnMut(usize, u64, &Page) -> Result<()>,
    ) -> Result<Vec<u32>> {
        let file = self.file.read().map_err(|_| poisoned())?;
        let mut checksums = Vec::with_capacity(ids.len());
        for (n, &id) in ids.iter().enumerate() {
            let mut page = match self.locate(id)? {
                Located::Resident(frame) => *frame.data.read().map_err(|_| poisoned())?,
                Located::Spilled(page, _) => *page,
                Located::Image => match file.as_ref() {
                    Some(pages) if pages.holds(id) => pages.read(id)?,
                    _ => return Err(bad_page(PageId(id))),
                },
            };
            fix(&mut page);
            checksums.push(crc32fast::hash(&page));
            emit(n, id, &page)?;
        }
        Ok(checksums)
    }

    /// Where the current copy of `id` is: resident, spilled, or (neither) the image. Both are
    /// looked up under one hold of the directory lock, which every move between memory and the
    /// spill takes for writing, so a page in transit is never missed. When neither holds it,
    /// the image's copy is current: a changed page is always resident or spilled.
    fn locate(&self, id: u64) -> Result<Located> {
        let frames = self.frames.read().map_err(|_| poisoned())?;
        if let Some(frame) = frames.get(&id) {
            return Ok(Located::Resident(Arc::clone(frame)));
        }
        let spilled = self.spill.lock().map_err(|_| poisoned())?.get(id)?;
        drop(frames);
        Ok(spilled.map_or(Located::Image, |(page, generation)| {
            Located::Spilled(Box::new(page), generation)
        }))
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
    fn insert_frame(
        &self,
        id: u64,
        page: &Page,
        dirty: bool,
        source: Source,
    ) -> Result<Option<Arc<Frame>>> {
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
            // A copy loaded while a newer one was spilled (or re-spilled) is stale: refuse it,
            // and the caller loads again. Checked and settled under the directory lock, which
            // every spill write also holds.
            let mut spill = self.spill.lock().map_err(|_| poisoned())?;
            if !spill.admits(id, source) {
                return Ok(None);
            }
            // The page is resident again: its spilled copy, if any, is stale from here on.
            spill.remove(id);
            drop(spill);
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

    /// Evict pages by the clock rule until one frame is free under the capacity, or until
    /// nothing evictable remains.
    ///
    /// With a working spill a sweep looks at most [`SWEEP_LIMIT`] frames for a clean page and
    /// otherwise spills the first changed page it passed, so a cache full of changed pages costs
    /// a short sweep per miss rather than a walk of the whole clock. Without a spill (or while
    /// its writes fail) only a clean page can leave, so the sweep covers the whole clock, and a
    /// changed page is spilled only once every resident page is changed. The victim is picked
    /// with the directory read-locked; the write lock is taken only to remove it.
    fn make_room(&self) -> Result<()> {
        let capacity = self.capacity_frames.load(Ordering::Acquire);
        if capacity == 0 {
            return Ok(());
        }
        let mut retries = 0;
        let mut spill_failed = false;
        loop {
            let len = self.frames.read().map_err(|_| poisoned())?.len() as u64;
            if len < capacity {
                return Ok(());
            }
            let (spill_on, spill_failing) = {
                let spill = self.spill.lock().map_err(|_| poisoned())?;
                (spill.file.is_some(), spill.failing)
            };
            let mut clock = self.clock.lock().map_err(|_| poisoned())?;
            let bounded = spill_on && !spill_failing && !spill_failed;
            let (limit, policy) = if bounded {
                (SWEEP_LIMIT, DirtyPolicy::Fallback)
            } else {
                // While every resident page is dirty, only spilling can make room.
                let all_dirty = self.dirty_frames.load(Ordering::Acquire) >= len;
                if all_dirty && (!spill_on || spill_failed) {
                    return Ok(());
                }
                let whole = clock.len().saturating_mul(2).saturating_add(1);
                (
                    whole,
                    if all_dirty {
                        DirtyPolicy::Take
                    } else {
                        DirtyPolicy::Skip
                    },
                )
            };
            let outcome = match self.pick_victim(&mut clock, limit, policy)? {
                Some(victim) => self.evict(&mut clock, victim)?,
                // A short sweep can pass only referenced or latched frames; it cleared their
                // bits, so the next one finds a victim. A whole sweep that found none means
                // nothing is evictable.
                None if bounded => Eviction::Busy,
                None => return Ok(()),
            };
            drop(clock);
            match outcome {
                Eviction::Done => {},
                // Racing users of the cache: let the frame in over the bound rather than spin;
                // the next miss evicts down to the capacity again.
                Eviction::Busy => {
                    retries += 1;
                    if retries > EVICT_RETRIES {
                        return Ok(());
                    }
                },
                // The store never fails a caller because of the spill: look for a clean page
                // the way a store without one does.
                Eviction::SpillFailed if bounded => spill_failed = true,
                Eviction::SpillFailed => return Ok(()),
            }
        }
    }

    /// Sweep at most `limit` clock entries for a victim: an unreferenced frame no one holds,
    /// clean, or changed as `policy` allows. Referenced frames lose their bit and go to the back.
    fn pick_victim(
        &self,
        clock: &mut VecDeque<u64>,
        limit: usize,
        policy: DirtyPolicy,
    ) -> Result<Option<Victim>> {
        let frames = self.frames.read().map_err(|_| poisoned())?;
        let mut fallback = None;
        for _ in 0..limit {
            let Some(id) = clock.pop_front() else { break };
            let Some(frame) = frames.get(&id) else {
                continue; // evicted or freed already
            };
            let dirty = frame.dirty.load(Ordering::Acquire);
            if (dirty && policy == DirtyPolicy::Skip)
                || frame.referenced.swap(false, Ordering::AcqRel)
            {
                clock.push_back(id);
                continue;
            }
            // A frame someone is reading or writing right now cannot leave.
            if frame.data.try_write().is_err() {
                clock.push_back(id);
                continue;
            }
            if !dirty || policy == DirtyPolicy::Take {
                if let Some(passed) = fallback {
                    clock.push_front(passed);
                }
                return Ok(Some(Victim { id, spill: dirty }));
            }
            if fallback.is_none() {
                fallback = Some(id);
            } else {
                clock.push_back(id);
            }
        }
        Ok(fallback.map(|id| Victim { id, spill: true }))
    }

    /// Remove `victim` from the directory if it is still evictable: not latched, and clean
    /// unless it was picked to spill. A spilled page is written out and registered before its
    /// frame leaves the directory, both under the directory write lock, so `locate` never misses
    /// a page in transit.
    fn evict(&self, clock: &mut VecDeque<u64>, victim: Victim) -> Result<Eviction> {
        let mut frames = self.frames.write().map_err(|_| poisoned())?;
        let Some(frame) = frames.get(&victim.id).cloned() else {
            return Ok(Eviction::Done); // freed meanwhile: its frame is gone already
        };
        // Evict only a frame no one is reading or writing right now, and only if it is still
        // clean under its latch (a writer marks dirty while holding it) or picked to spill.
        let Ok(latch) = frame.data.try_write() else {
            clock.push_back(victim.id);
            return Ok(Eviction::Busy);
        };
        if frame.dirty.load(Ordering::Acquire) {
            if !victim.spill {
                clock.push_back(victim.id);
                return Ok(Eviction::Busy);
            }
            // A failed write keeps the page resident.
            let spilled = self
                .spill
                .lock()
                .map_err(|_| poisoned())?
                .put(victim.id, &latch);
            if spilled.is_err() {
                clock.push_back(victim.id);
                return Ok(Eviction::SpillFailed);
            }
            self.dirty_frames.fetch_sub(1, Ordering::AcqRel);
        }
        drop(latch);
        frames.remove(&victim.id);
        Ok(Eviction::Done)
    }
}

#[allow(
    clippy::significant_drop_tightening,
    reason = "each guard IS the critical section of its one-shot page operation"
)]
impl PageStore for PagedStore {
    fn read_page(&self, id: PageId) -> Result<Page> {
        loop {
            let located = self.locate(id.0)?;
            if let Located::Resident(frame) = located {
                frame.referenced.store(true, Ordering::Release);
                let page = frame.data.read().map_err(|_| poisoned())?;
                return Ok(*page);
            }
            if self.free.lock().map_err(|_| poisoned())?.1.contains(&id.0) {
                return Err(bad_page(id));
            }
            // A changed page that was spilled comes back as a changed page.
            let (page, dirty, source) = if let Located::Spilled(page, generation) = located {
                (*page, true, Source::Spill(generation))
            } else {
                let file = self.file.read().map_err(|_| poisoned())?;
                match file.as_ref() {
                    Some(pages) if pages.holds(id.0) => (pages.read(id.0)?, false, Source::Image),
                    _ => return Err(bad_page(id)),
                }
            };
            // A fresh frame holds exactly this copy; return it directly. A racing load or write
            // that made the page resident first is read on the next pass.
            if self.insert_frame(id.0, &page, dirty, source)?.is_some() {
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
            if self
                .insert_frame(id.0, page, true, Source::Fresh)?
                .is_some()
            {
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
        let inserted = match self.insert_frame(id, &zero, true, Source::Fresh) {
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
            self.spill.lock().map_err(|_| poisoned())?.remove(id.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn page_of(byte: u8) -> Page {
        [byte; PAGE_SIZE]
    }

    /// A store of `capacity` frames that spills to a scratch file.
    fn spilling(capacity: u64) -> PagedStore {
        let store = PagedStore::default();
        store.set_capacity_bytes(Some(capacity * PAGE_SIZE as u64));
        store.enable_spill(tempfile::tempfile().unwrap()).unwrap();
        store
    }

    /// Write `count` fresh pages, so the pages written before them spill.
    fn crowd_out(store: &PagedStore, count: u64) {
        for _ in 0..count {
            let id = store.allocate_page().unwrap();
            store.write_page(id, &page_of(0xEE)).unwrap();
        }
    }

    /// A sealed spill holds no page in the clear, reads every page back, and refuses a slot
    /// that was altered.
    #[test]
    fn a_sealed_spill_holds_nothing_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spill");
        let store = PagedStore::default();
        store.set_capacity_bytes(Some(4 * PAGE_SIZE as u64));
        store
            .enable_sealed_spill(
                std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .read(true)
                    .write(true)
                    .open(&path)
                    .unwrap(),
                Sealer::ephemeral().unwrap(),
            )
            .unwrap();
        let mut page = page_of(0);
        page[..16].copy_from_slice(b"ZEBRA-SPILL-ROW!");
        let mut ids = Vec::new();
        for _ in 0..16 {
            let id = store.allocate_page().unwrap();
            store.write_page(id, &page).unwrap();
            ids.push(id);
        }
        assert!(store.spilled_bytes() > 0);
        let bytes = std::fs::read(&path).unwrap();
        assert!(!bytes.is_empty());
        assert!(!bytes.windows(16).any(|w| w == b"ZEBRA-SPILL-ROW!"));
        assert_eq!(bytes.len() % SEALED_PAGE_SIZE, 0);
        for &id in &ids {
            assert_eq!(store.read_page(id).unwrap(), page);
        }
        // Spill some again, then alter every slot: a spilled page no longer reads.
        crowd_out(&store, 8);
        let mut bytes = std::fs::read(&path).unwrap();
        for slot in 0..bytes.len() / SEALED_PAGE_SIZE {
            bytes[slot * SEALED_PAGE_SIZE + 40] ^= 1;
        }
        std::fs::write(&path, &bytes).unwrap();
        assert!(ids.iter().any(|&id| store.read_page(id).is_err()));
    }

    /// A copy loaded before the page was changed and spilled again is refused, as is an image
    /// copy while a spilled one exists: only the newest copy can become resident.
    #[test]
    fn a_stale_copy_never_replaces_a_newer_spilled_one() {
        let store = spilling(4);
        let id = store.allocate_page().unwrap();
        store.write_page(id, &page_of(1)).unwrap();
        crowd_out(&store, 8);
        let Located::Spilled(stale, first) = store.locate(id.0).unwrap() else {
            panic!("the page was not spilled");
        };
        assert_eq!(*stale, page_of(1));
        // Load it back, change it, and push it out again: a newer spilled copy.
        store.read_page(id).unwrap();
        store.write_page(id, &page_of(2)).unwrap();
        crowd_out(&store, 8);
        let Located::Spilled(_, second) = store.locate(id.0).unwrap() else {
            panic!("the page was not spilled again");
        };
        assert_ne!(first, second);
        // The copy read before the change must not be installed, from the spill or the image.
        assert!(
            store
                .insert_frame(id.0, &stale, true, Source::Spill(first))
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .insert_frame(id.0, &stale, false, Source::Image)
                .unwrap()
                .is_none()
        );
        assert_eq!(store.read_page(id).unwrap(), page_of(2));
    }

    /// A spill write that fails keeps the page in memory, takes no slot, and the store keeps
    /// serving every page.
    #[test]
    fn a_failed_spill_write_keeps_the_page_and_leaks_no_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spill");
        std::fs::write(&path, b"").unwrap();
        let store = PagedStore::default();
        store.set_capacity_bytes(Some(4 * PAGE_SIZE as u64));
        // Opened read-only: every spill write fails.
        store.enable_spill(File::open(&path).unwrap()).unwrap();
        let ids: Vec<PageId> = (0..12)
            .map(|i| {
                let id = store.allocate_page().unwrap();
                store.write_page(id, &page_of(i)).unwrap();
                id
            })
            .collect();
        assert_eq!(store.spilled_bytes(), 0);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        let (next_slot, free_slots) = {
            let spill = store.spill.lock().unwrap();
            (spill.next_slot, spill.free_slots.clone())
        };
        assert_eq!(next_slot, 1, "only one slot was ever taken");
        assert_eq!(free_slots, vec![0], "and it went back");
    }

    /// A spilled page comes back as a changed page and leaves the spill; freeing a spilled page
    /// drops its copy.
    #[test]
    fn spilled_pages_come_back_changed_and_free_their_slot() {
        let store = spilling(4);
        let ids: Vec<PageId> = (0..12)
            .map(|i| {
                let id = store.allocate_page().unwrap();
                store.write_page(id, &page_of(i)).unwrap();
                id
            })
            .collect();
        assert!(store.spilled_bytes() > 0);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        let spilled_before = store.spilled_bytes();
        let victim = ids
            .iter()
            .copied()
            .find(|id| matches!(store.locate(id.0).unwrap(), Located::Spilled(..)))
            .unwrap();
        store.deallocate_page(victim).unwrap();
        assert_eq!(
            store.spilled_bytes(),
            spilled_before - PAGE_SIZE as u64,
            "a freed page keeps no spilled copy"
        );
    }

    /// A store over an image of `count` pages (page `id` filled with `id`), bounded to
    /// `capacity` frames.
    fn imaged(dir: &Path, count: u64, capacity: u64) -> PagedStore {
        let path = dir.join("image");
        let mut bytes = Vec::new();
        let mut checksums = Vec::new();
        for id in 0..count {
            let page = page_of(id as u8);
            checksums.push(crc32fast::hash(&page));
            bytes.extend_from_slice(&page);
        }
        std::fs::write(&path, &bytes).unwrap();
        let store = PagedStore::default();
        let pages = PageFile::open(&path, 0, count, (0..count).collect(), checksums).unwrap();
        store.attach(pages).unwrap();
        store.set_capacity_bytes(Some(capacity * PAGE_SIZE as u64));
        store
    }

    /// Clear every second-chance bit, so the next sweep picks by clock position alone.
    fn settle(store: &PagedStore) {
        for frame in store.frames.read().unwrap().values() {
            frame.referenced.store(false, Ordering::Release);
        }
    }

    fn is_resident(store: &PagedStore, id: u64) -> bool {
        store.frames.read().unwrap().contains_key(&id)
    }

    fn frame_count(store: &PagedStore) -> u64 {
        store.frames.read().unwrap().len() as u64
    }

    /// The dirty count agrees with the frames that are dirty.
    fn assert_dirty_count(store: &PagedStore) {
        let dirty = store
            .frames
            .read()
            .unwrap()
            .values()
            .filter(|f| f.dirty.load(Ordering::Acquire))
            .count() as u64;
        assert_eq!(store.dirty_frames.load(Ordering::Acquire), dirty);
    }

    /// Fill a bounded cache over a one-page image: `before` changed pages, then image page 0
    /// (clean), then changed pages up to the capacity, every second-chance bit cleared.
    fn one_clean_frame_at(store: &PagedStore, before: u64, capacity: u64) -> Vec<PageId> {
        let mut ids: Vec<PageId> = (0..before)
            .map(|i| {
                let id = store.allocate_page().unwrap();
                store.write_page(id, &page_of(i as u8)).unwrap();
                id
            })
            .collect();
        assert_eq!(store.read_page(PageId(0)).unwrap(), page_of(0));
        while frame_count(store) < capacity {
            let id = store.allocate_page().unwrap();
            store.write_page(id, &page_of(ids.len() as u8)).unwrap();
            ids.push(id);
        }
        settle(store);
        ids
    }

    /// A cache of changed pages only stays within its bound by spilling, and every page reads
    /// back as last written.
    #[test]
    fn a_cache_of_changed_pages_spills_within_its_bound() {
        let store = spilling(64);
        let mut ids = Vec::new();
        for i in 0..3000_u64 {
            let id = store.allocate_page().unwrap();
            store.write_page(id, &page_of(i as u8)).unwrap();
            ids.push(id);
            assert!(frame_count(&store) <= 64, "{} frames", frame_count(&store));
        }
        assert!(store.spilled_bytes() >= (3000 - 64) * PAGE_SIZE as u64);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        assert_dirty_count(&store);
    }

    /// A clean page in the sweep's reach leaves before any changed page is spilled, and reads
    /// back from the image.
    #[test]
    fn a_clean_page_leaves_before_a_changed_one_spills() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 1, 1000);
        store.enable_spill(tempfile::tempfile().unwrap()).unwrap();
        one_clean_frame_at(&store, 0, 1000);
        crowd_out(&store, 1);
        assert!(!is_resident(&store, 0));
        assert_eq!(store.spilled_bytes(), 0);
        assert_eq!(frame_count(&store), 1000);
        assert_eq!(store.read_page(PageId(0)).unwrap(), page_of(0));
    }

    /// With a spill, the one clean page far down the clock is not searched for: the first
    /// changed page swept is spilled instead, and the cache stays within its bound.
    #[test]
    fn a_short_sweep_spills_rather_than_search_the_whole_clock() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 1, 1000);
        store.enable_spill(tempfile::tempfile().unwrap()).unwrap();
        let ids = one_clean_frame_at(&store, SWEEP_LIMIT as u64 + 100, 1000);
        crowd_out(&store, 1);
        assert!(is_resident(&store, 0), "the clean page lies past the sweep");
        assert_eq!(store.spilled_bytes(), PAGE_SIZE as u64);
        assert!(
            !is_resident(&store, ids[0].0),
            "the first page swept was spilled"
        );
        assert_eq!(frame_count(&store), 1000);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        assert_dirty_count(&store);
    }

    /// Without a spill the whole clock is searched: the one clean page leaves wherever it is,
    /// and once only changed pages remain the cache grows past its bound instead of failing.
    #[test]
    fn without_a_spill_the_whole_clock_is_searched_for_a_clean_page() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 1, 1000);
        let ids = one_clean_frame_at(&store, SWEEP_LIMIT as u64 + 100, 1000);
        crowd_out(&store, 1);
        assert!(!is_resident(&store, 0));
        assert_eq!(frame_count(&store), 1000);
        crowd_out(&store, 5);
        assert_eq!(
            frame_count(&store),
            1005,
            "changed pages stay without a spill"
        );
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        assert_eq!(store.read_page(PageId(0)).unwrap(), page_of(0));
        assert_dirty_count(&store);
    }

    /// A spill write that fails sends the sweep after a clean page the way a store without a
    /// spill does, leaks no slot, and with only changed pages left the cache grows instead.
    #[test]
    fn a_failing_spill_falls_back_to_a_clean_page() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 1, 1000);
        let path = dir.path().join("spill");
        std::fs::write(&path, b"").unwrap();
        // Opened read-only: every spill write fails.
        store.enable_spill(File::open(&path).unwrap()).unwrap();
        let ids = one_clean_frame_at(&store, SWEEP_LIMIT as u64 + 100, 1000);
        crowd_out(&store, 1);
        assert!(!is_resident(&store, 0), "the clean page left instead");
        assert_eq!(frame_count(&store), 1000);
        crowd_out(&store, 5);
        assert_eq!(frame_count(&store), 1005);
        assert_eq!(store.spilled_bytes(), 0);
        let (next_slot, free_slots) = {
            let spill = store.spill.lock().unwrap();
            (spill.next_slot, spill.free_slots.clone())
        };
        assert_eq!(next_slot, 1, "only one slot was ever taken");
        assert_eq!(free_slots, vec![0], "and it went back");
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(store.read_page(*id).unwrap(), page_of(i as u8));
        }
        assert_dirty_count(&store);
    }

    /// Page `id` at `version`: both stamped in front, the rest filled from them, so a torn or
    /// misplaced copy is caught.
    fn stamped(id: u64, version: u64) -> Page {
        let mut page = [(id ^ version) as u8; PAGE_SIZE];
        page[..8].copy_from_slice(&id.to_le_bytes());
        page[8..16].copy_from_slice(&version.to_le_bytes());
        page
    }

    /// The version a page read for `id` carries, after checking it is whole.
    fn version_of(id: u64, page: &Page) -> u64 {
        let got = u64::from_le_bytes(page[..8].try_into().unwrap());
        assert_eq!(got, id, "page {id} came back holding page {got}");
        let version = u64::from_le_bytes(page[8..16].try_into().unwrap());
        assert_eq!(*page, stamped(id, version), "page {id} came back torn");
        version
    }

    /// Writers, readers and a thread that allocates and frees pages, all over a small cache
    /// that evicts and spills constantly: every read finds the page (never in transit between
    /// memory and the spill), whole and no older than its writer last wrote, and the counts
    /// stay consistent.
    fn eviction_races(store: &PagedStore, image_pages: u64, capacity: u64) {
        const WRITERS: u64 = 4;
        const OWNED: u64 = 48;
        let owned: Vec<Vec<PageId>> = (0..WRITERS)
            .map(|_| {
                (0..OWNED)
                    .map(|_| {
                        let id = store.allocate_page().unwrap();
                        store.write_page(id, &stamped(id.0, 0)).unwrap();
                        id
                    })
                    .collect()
            })
            .collect();
        let latest: Vec<Vec<std::sync::atomic::AtomicU64>> = (0..WRITERS)
            .map(|_| (0..OWNED).map(|_| AtomicU64::new(0)).collect())
            .collect();
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let mut writers = Vec::new();
            for (w, ids) in owned.iter().enumerate() {
                let latest = &latest[w];
                writers.push(scope.spawn(move || {
                    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ w as u64;
                    for version in 1..=20_000_u64 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let n = (state % OWNED) as usize;
                        let id = ids[n];
                        store.write_page(id, &stamped(id.0, version)).unwrap();
                        latest[n].store(version, Ordering::Release);
                        let page = store.read_page(id).unwrap();
                        assert_eq!(version_of(id.0, &page), version);
                    }
                }));
            }
            for r in 0..4_u64 {
                let owned = &owned;
                let latest = &latest;
                let done = &done;
                scope.spawn(move || {
                    let mut state = 0xD1B5_4A32_D192_ED03_u64 ^ r;
                    while !done.load(Ordering::Acquire) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if state.is_multiple_of(3) {
                            let id = state % image_pages;
                            assert_eq!(store.read_page(PageId(id)).unwrap(), page_of(id as u8));
                            continue;
                        }
                        let w = (state % WRITERS) as usize;
                        let n = ((state >> 8) % OWNED) as usize;
                        let id = owned[w][n];
                        let floor = latest[w][n].load(Ordering::Acquire);
                        let version = store.with_page(id, |page| version_of(id.0, page)).unwrap();
                        assert!(version >= floor, "page {} went back in time", id.0);
                    }
                });
            }
            let done = &done;
            scope.spawn(move || {
                let mut version = 0;
                while !done.load(Ordering::Acquire) {
                    version += 1;
                    let id = store.allocate_page().unwrap();
                    store.write_page(id, &stamped(id.0, version)).unwrap();
                    crowd_out(store, 2);
                    assert_eq!(version_of(id.0, &store.read_page(id).unwrap()), version);
                    store.deallocate_page(id).unwrap();
                }
            });
            for writer in writers {
                writer.join().unwrap();
            }
            done.store(true, Ordering::Release);
        });
        for (w, ids) in owned.iter().enumerate() {
            for (n, id) in ids.iter().enumerate() {
                let page = store.read_page(*id).unwrap();
                assert_eq!(
                    version_of(id.0, &page),
                    latest[w][n].load(Ordering::Acquire)
                );
            }
        }
        assert_dirty_count(store);
        // A racing miss may let its frame in over the bound; the next miss evicts down again.
        crowd_out(store, 4);
        assert!(
            frame_count(store) <= capacity + 4,
            "{} frames",
            frame_count(store)
        );
    }

    #[test]
    fn eviction_races_with_readers_and_writers() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 200, 64);
        store.enable_spill(tempfile::tempfile().unwrap()).unwrap();
        eviction_races(&store, 200, 64);
        assert!(store.spilled_bytes() > 0);
    }

    #[test]
    fn eviction_races_with_readers_and_writers_over_a_sealed_spill() {
        let dir = tempfile::tempdir().unwrap();
        let store = imaged(dir.path(), 200, 64);
        store
            .enable_sealed_spill(tempfile::tempfile().unwrap(), Sealer::ephemeral().unwrap())
            .unwrap();
        eviction_races(&store, 200, 64);
    }
}
