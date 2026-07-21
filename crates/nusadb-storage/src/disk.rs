//! Disk-backed [`PageStore`] adapter.
//!
//! Wraps a single file containing densely-packed 8 KiB pages and exposes the same
//! trait surface as the in-memory `SimStorage` so that engine code is portable between
//! production and DST.
//!
//! # Free-space management
//!
//! [`DiskManager::allocate_page`] prefers reusing previously
//! [`DiskManager::deallocate_page`]-d slots before extending the file. Free slots are
//! tracked in an in-memory list and **also** stamped on disk with [`FREE_PAGE_MAGIC`] so
//! the list can be rebuilt by scanning the file on the next [`DiskManager::open`]. The
//! magic is disjoint from [`PAGE_MAGIC`](crate::page::PAGE_MAGIC) and `CATALOG_MAGIC`,
//! so live pages are never mistaken for free.
//!
//! # Concurrency
//!
//! All I/O is **positioned** (`pread`/`pwrite`-style: `read_at`/`write_at` on Unix,
//! `seek_read`/`seek_write` on Windows) so it neither uses nor moves the file cursor. That
//! lets the file be shared by `&self` without a lock, so many threads can
//! [`read_page`](PageStore::read_page) concurrently. Writes to distinct pages are likewise
//! independent; same-page coordination belongs to the buffer pool above.
//!
//! # Direct I/O (`direct-io` feature)
//!
//! With the optional `direct-io` feature the data file is opened **unbuffered**
//! (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH` on Windows, `O_DIRECT` on common Linux
//! architectures), bypassing the OS page cache so the buffer pool is the single cache and writes
//! reach the device without a second copy. Unbuffered I/O constrains the *caller's* buffer to be
//! sector-aligned, the I/O length to be a multiple of the sector size, and the file offset to be
//! sector-aligned. Pages are 8 KiB and laid out at 8 KiB-aligned offsets, so length and offset
//! already satisfy that; the buffer requirement is met by routing every page transfer through a
//! 4 KiB-aligned bounce buffer (`AlignedPage`). The feature is off by default (buffered I/O,
//! 1-byte-aligned stack buffers) and the on-disk format is identical either way. On platforms
//! without a portable unbuffered flag the feature still compiles and stays correct (the aligned
//! path runs; the cache simply isn't bypassed).

use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};
use parking_lot::Mutex;

const PAGE_SIZE_U64: u64 = PAGE_SIZE as u64;

/// Magic stamped into the first 4 bytes of a deallocated page so the free list can be
/// rebuilt by scanning the file on [`DiskManager::open`]. ASCII `"FREE"`.
pub const FREE_PAGE_MAGIC: u32 = 0x4652_4545;

/// Magic at the head of the free-space-map sidecar (`<data>.fsm`). ASCII `"NFSM"`.
const FSM_MAGIC: u32 = 0x4E46_534D;
/// FSM sidecar format version.
const FSM_VERSION: u32 = 1;

/// A blank page, reused for both file extension and the body of the zero-fill on reuse.
const ZERO_PAGE: [u8; PAGE_SIZE] = [0u8; PAGE_SIZE];

/// A page-sized buffer aligned to 4 KiB so it satisfies the buffer-alignment requirement that
/// unbuffered (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`) I/O imposes on the caller. A plain
/// `[u8; PAGE_SIZE]` is only 1-byte aligned, which the OS rejects under direct I/O. Used as a
/// bounce buffer so the public page API stays a plain `[u8; PAGE_SIZE]`.
#[cfg(feature = "direct-io")]
#[repr(C, align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

/// Production [`PageStore`] backed by a single OS file.
///
/// Pages are stored contiguously: page `n` occupies bytes `[n * 8192, (n+1) * 8192)`.
/// Deallocated pages are reused before the file is extended.
#[derive(Debug)]
pub struct DiskManager {
    file: File,
    /// Path of the data file, used to derive the free-space-map sidecar path.
    path: PathBuf,
    next_page: AtomicU64,
    /// Page ids that have been deallocated and are eligible for reuse. LIFO so the most
    /// recently freed id is recycled first; ordering is a debuggability aid, not a
    /// correctness requirement.
    free_list: Mutex<Vec<u64>>,
    /// Set whenever the free list changes (allocate-reuse or deallocate); cleared when the FSM
    /// sidecar is rewritten at [`fsync`](PageStore::fsync), so an unchanged free list is not
    /// re-serialized.
    free_dirty: AtomicBool,
}

impl DiskManager {
    /// Open (creating if absent) the data file at `path`, rebuilding the free list.
    ///
    /// Fast path: the free-space-map sidecar (`<path>.fsm`, written at [`fsync`](PageStore::fsync))
    /// is read and each listed id is **verified** to still bear [`FREE_PAGE_MAGIC`] on disk before
    /// being trusted — `O(free_count)` page reads. If the sidecar is absent or corrupt, fall
    /// back to a full `scan_free_pages` of the file (`O(page_count)`), the original behaviour.
    ///
    /// Verification keeps this correct across a crash: a page the sidecar lists as free but that was
    /// since reused (and so no longer carries the marker) is dropped, so a live page is never handed
    /// back as free. The only effect of a stale sidecar is bounded: pages freed *after* the last
    /// `fsync` are not relisted until a future full scan — a temporary space underuse, never
    /// corruption.
    ///
    /// The next page id is derived from the file length, so reopening an existing file
    /// resumes allocation where it left off.
    ///
    /// # Errors
    /// Propagates any filesystem error from opening, stat-ing, or scanning the file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(false);
        apply_unbuffered(&mut opts);
        let file = opts.open(&path)?;
        let page_count = file.metadata()?.len() / PAGE_SIZE_U64;
        // A clean FSM load needs no rewrite; a fallback scan (sidecar absent/corrupt) re-arms the
        // dirty flag so the next fsync regenerates the sidecar — the FSM self-heals.
        let (free_list, from_scan) = match load_fsm(&fsm_path(&path)) {
            Some(ids) => (verify_free_pages(&file, page_count, &ids)?, false),
            None => (scan_free_pages(&file, page_count)?, true),
        };
        Ok(Self {
            file,
            path,
            next_page: AtomicU64::new(page_count),
            free_list: Mutex::new(free_list),
            free_dirty: AtomicBool::new(from_scan),
        })
    }

    /// Number of pages backing the file. Includes both live and deallocated slots —
    /// physical space is reclaimed on reuse, not on deallocation.
    #[must_use]
    pub fn page_count(&self) -> u64 {
        self.next_page.load(Ordering::Acquire)
    }

    /// Number of slots currently on the free list.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_list.lock().len()
    }

    /// Byte offset of a page that must already have been allocated. A page id at or beyond
    /// the high-water mark (`next_page`) was never handed out — a stale or forged pointer — so it
    /// is rejected rather than aliasing a live slot or writing a sparse hole past end-of-file. The
    /// multiply is also overflow-checked. Used by every `&self`-addressed read/write/dealloc.
    fn require_allocated(&self, id: PageId) -> std::io::Result<u64> {
        if id.0 >= self.next_page.load(Ordering::Acquire) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "page id out of range (never allocated)",
            ));
        }
        page_byte_offset(id.0)
    }

    /// Read-ahead: fill `buf` with the run of consecutive pages starting at `start`, in a single
    /// positioned read, and return how many pages were read.
    ///
    /// `buf.len()` must be a non-zero multiple of [`PAGE_SIZE`]; it holds `buf.len() / PAGE_SIZE`
    /// pages on return (`buf[i*PAGE_SIZE .. (i+1)*PAGE_SIZE]` is page `start + i`). Pages are densely
    /// packed, so the run is contiguous on disk and is pulled with **one** syscall instead of one
    /// per page — the amortization a sequential scan (heap scan, B-tree leaf chain over a
    /// freshly built tree) wants. **Zero-copy and zero-alloc**: a scanner reuses one window buffer
    /// across the whole scan. This is an inherent method, not part of the minimal `PageStore`
    /// treaty.
    ///
    /// Under the `direct-io` feature the bytes are read one page at a time through the aligned
    /// path (unbuffered I/O needs a sector-aligned buffer, which the caller's `buf` may not be);
    /// correctness is identical, only the single-syscall batching is skipped.
    ///
    /// # Errors
    /// Returns [`Error::Io`] if `buf` is empty or not a multiple of `PAGE_SIZE`, and propagates read
    /// errors (including a short read past end-of-file).
    pub fn read_pages_into(&self, start: PageId, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() || !buf.len().is_multiple_of(PAGE_SIZE) {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read_pages_into: buffer must be a non-zero multiple of PAGE_SIZE",
            )));
        }
        let count = buf.len() / PAGE_SIZE;
        // The whole run must lie within allocated space: reject a forged/stale `start` or a
        // window that overruns the high-water mark instead of reading aliased or past-EOF bytes.
        let end = start.0.checked_add(count as u64).ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read_pages_into: page range overflow",
            ))
        })?;
        if end > self.next_page.load(Ordering::Acquire) {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "read_pages_into: range past end of allocated pages",
            )));
        }
        let base = page_byte_offset(start.0)?;
        #[cfg(feature = "direct-io")]
        {
            for (i, chunk) in buf.chunks_exact_mut(PAGE_SIZE).enumerate() {
                let page = read_page_at(&self.file, base + (i as u64) * PAGE_SIZE_U64)?;
                chunk.copy_from_slice(&page);
            }
        }
        #[cfg(not(feature = "direct-io"))]
        {
            read_exact_at(&self.file, buf, base)?;
        }
        Ok(count)
    }
}

impl PageStore for DiskManager {
    fn read_page(&self, id: PageId) -> Result<[u8; PAGE_SIZE]> {
        let offset = self.require_allocated(id)?;
        Ok(read_page_at(&self.file, offset)?)
    }

    fn write_page(&self, id: PageId, page: &[u8; PAGE_SIZE]) -> Result<()> {
        let offset = self.require_allocated(id)?;
        write_page_at(&self.file, offset, page)?;
        Ok(())
    }

    fn allocate_page(&self) -> Result<PageId> {
        // Reuse a deallocated slot first; the returned page is zero-filled so the caller
        // sees the same blank state it would get from a freshly extended slot.
        let recycled = self.free_list.lock().pop();
        if let Some(id) = recycled {
            self.free_dirty.store(true, Ordering::Relaxed);
            write_page_at(&self.file, page_byte_offset(id)?, &ZERO_PAGE)?;
            return Ok(PageId(id));
        }

        // Extending the file: `id` is the freshly reserved high-water slot (now < next_page).
        let id = self.next_page.fetch_add(1, Ordering::AcqRel);
        write_page_at(&self.file, page_byte_offset(id)?, &ZERO_PAGE)?;
        Ok(PageId(id))
    }

    /// Mark `id` as free so it can be returned by a future [`allocate_page`](
    /// PageStore::allocate_page). Writes [`FREE_PAGE_MAGIC`] over the slot so the
    /// deallocation survives a restart.
    ///
    /// Caller is responsible for ensuring no live data references `id` — this is the
    /// page-store layer; ref-counting belongs above it.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the free-list lock must span the marker write: releasing it between the \
                  duplicate check and the I/O would let a concurrent allocate_page pop and \
                  reuse the slot while this call overwrites it with the FREE marker"
    )]
    fn deallocate_page(&self, id: PageId) -> Result<()> {
        let offset = self.require_allocated(id)?;
        // Idempotent: a slot already on the free list must not be pushed twice. A double
        // push would let two `allocate_page` calls hand back the same id — two logical pages over
        // one physical slot. The free list is rebuilt from on-disk FREE markers at open(), so this
        // in-memory guard also covers a re-deallocation after restart.
        let mut free = self.free_list.lock();
        if free.contains(&id.0) {
            return Ok(());
        }
        let mut marker = [0u8; PAGE_SIZE];
        marker[..4].copy_from_slice(&FREE_PAGE_MAGIC.to_le_bytes());
        write_page_at(&self.file, offset, &marker)?;
        free.push(id.0);
        self.free_dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn fsync(&self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| Error::FsyncFailed(e.to_string()))?;
        // Checkpoint the free list to the FSM sidecar so the next open() skips the full file scan.
        // Only when it changed since the last checkpoint. The FREE markers on disk remain the
        // source of truth (open() re-verifies against them), so a torn/stale sidecar is never
        // unsafe — hence no separate fsync ordering requirement between the two.
        if self.free_dirty.swap(false, Ordering::AcqRel) {
            let snapshot = self.free_list.lock().clone();
            if let Err(e) = write_fsm(&fsm_path(&self.path), &snapshot) {
                // The sidecar is an optimization; if it can't be written, re-arm the dirty flag so a
                // later fsync retries, and surface the error.
                self.free_dirty.store(true, Ordering::Relaxed);
                return Err(e);
            }
        }
        Ok(())
    }
}

/// Byte offset of page `id`, overflow-checked. Multiplying a u64 page id by the page size
/// can wrap on a corrupt/forged id, so the multiply is fallible rather than silently aliasing a
/// low offset.
fn page_byte_offset(id: u64) -> std::io::Result<u64> {
    id.checked_mul(PAGE_SIZE_U64).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "page byte offset overflow",
        )
    })
}

/// Walk every page slot in the file once and collect the ids whose first 4 bytes equal
/// [`FREE_PAGE_MAGIC`]. Linear in `page_count`; acceptable Stage-1 cost — a dedicated
/// FSM page would replace this if open latency becomes a concern.
///
/// `page_count` is `file_len / PAGE_SIZE` (floored), so every id in `0..page_count` has a full
/// page on disk: a read error here is a real I/O fault, not an expected short read, and is
/// propagated rather than silently truncating the free list — swallowing it would leak the
/// un-scanned free slots permanently.
fn scan_free_pages(file: &File, page_count: u64) -> Result<Vec<u64>> {
    let mut free = Vec::new();
    for id in 0..page_count {
        // Read a whole page rather than just its 4-byte header: under direct I/O the OS rejects
        // sub-sector reads, and the per-slot cost is negligible at open time. We only inspect the
        // leading magic.
        let page = read_page_at(file, page_byte_offset(id)?)?;
        let head = u32::from_le_bytes([page[0], page[1], page[2], page[3]]);
        if head == FREE_PAGE_MAGIC {
            free.push(id);
        }
    }
    Ok(free)
}

/// Whether `id`'s slot on disk still carries [`FREE_PAGE_MAGIC`] (i.e. it is genuinely free).
fn slot_is_free(file: &File, id: u64) -> Result<bool> {
    let page = read_page_at(file, page_byte_offset(id)?)?;
    let head = u32::from_le_bytes([page[0], page[1], page[2], page[3]]);
    Ok(head == FREE_PAGE_MAGIC)
}

/// Build the free list from the FSM sidecar's candidate ids, keeping only those that are in range
/// and **still** marked free on disk. This is the fast-path counterpart of [`scan_free_pages`]:
/// `O(candidates)` reads instead of `O(page_count)`. Verification makes trusting the sidecar safe —
/// any candidate that was reused since the sidecar was written no longer bears the marker and is
/// dropped, so a live page is never recycled.
fn verify_free_pages(file: &File, page_count: u64, candidates: &[u64]) -> Result<Vec<u64>> {
    let mut free = Vec::with_capacity(candidates.len());
    for &id in candidates {
        if id < page_count && slot_is_free(file, id)? {
            free.push(id);
        }
    }
    Ok(free)
}

/// Path of the free-space-map sidecar for the data file at `data`: the same path with `.fsm`
/// appended (e.g. `nusadb.db` → `nusadb.db.fsm`).
fn fsm_path(data: &Path) -> PathBuf {
    let mut name = data.as_os_str().to_os_string();
    name.push(".fsm");
    PathBuf::from(name)
}

/// Read the FSM sidecar at `path`, returning its candidate free-page ids, or `None` if the file is
/// absent, truncated, or fails its magic/version/CRC check. A `None` return means the caller falls
/// back to a full scan, so a corrupt sidecar is self-healing, never fatal.
///
/// Layout (little-endian): `magic u32 | version u32 | count u64 | ids[count] u64 | crc32 u32`,
/// where the CRC32 covers every preceding byte.
#[allow(
    clippy::indexing_slicing,
    reason = "fixed-offset byte codec; every slice is bounded by the length checks above it"
)]
fn load_fsm(path: &Path) -> Option<Vec<u64>> {
    let bytes = std::fs::read(path).ok()?;
    // header (16) + crc (4) is the minimum; reject anything shorter or mis-sized.
    if bytes.len() < 20 {
        return None;
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if magic != FSM_MAGIC || version != FSM_VERSION {
        return None;
    }
    let count = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let body_end = 16usize.checked_add((count as usize).checked_mul(8)?)?;
    if bytes.len() != body_end + 4 {
        return None;
    }
    let stored_crc = u32::from_le_bytes(bytes[body_end..body_end + 4].try_into().ok()?);
    if crc32fast::hash(&bytes[..body_end]) != stored_crc {
        return None;
    }
    let mut ids = Vec::with_capacity(count as usize);
    for chunk in bytes[16..body_end].chunks_exact(8) {
        ids.push(u64::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(ids)
}

/// Serialize `free` to the FSM sidecar at `path`, written atomically (write a temp sibling, fsync
/// it, then rename over `path`) so a crash mid-write leaves the previous sidecar intact rather than
/// a torn one. See [`load_fsm`] for the layout.
fn write_fsm(path: &Path, free: &[u64]) -> Result<()> {
    let mut buf = Vec::with_capacity(20 + free.len() * 8);
    buf.extend_from_slice(&FSM_MAGIC.to_le_bytes());
    buf.extend_from_slice(&FSM_VERSION.to_le_bytes());
    buf.extend_from_slice(&(free.len() as u64).to_le_bytes());
    for &id in free {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let tmp = {
        let mut name = path.as_os_str().to_os_string();
        name.push(".tmp");
        PathBuf::from(name)
    };
    {
        use std::io::Write;
        let mut f = File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Apply the platform's unbuffered-I/O open flag when the `direct-io` feature is on. No-op
/// otherwise, and on platforms lacking a portable flag (the aligned-buffer path keeps the
/// feature correct regardless).
#[cfg(feature = "direct-io")]
fn apply_unbuffered(opts: &mut OpenOptions) {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
        opts.custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH);
    }
    #[cfg(all(
        target_os = "linux",
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "arm"
        )
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_DIRECT is 0o0040000 on these architectures (it differs on a few exotic ones, which
        // fall through to the no-flag path below).
        const O_DIRECT: i32 = 0x4000;
        opts.custom_flags(O_DIRECT);
    }
    let _ = &opts; // keep `opts` "used" on platforms where neither block fires
}

#[cfg(not(feature = "direct-io"))]
const fn apply_unbuffered(_opts: &mut OpenOptions) {}

/// Read one page at `offset`. Under `direct-io` this bounces through a 4 KiB-aligned buffer to
/// satisfy the OS alignment requirement; otherwise it reads straight into the result.
fn read_page_at(file: &File, offset: u64) -> std::io::Result<[u8; PAGE_SIZE]> {
    #[cfg(feature = "direct-io")]
    {
        let mut aligned = AlignedPage([0u8; PAGE_SIZE]);
        read_exact_at(file, &mut aligned.0, offset)?;
        Ok(aligned.0)
    }
    #[cfg(not(feature = "direct-io"))]
    {
        let mut buf = [0u8; PAGE_SIZE];
        read_exact_at(file, &mut buf, offset)?;
        Ok(buf)
    }
}

/// Write one page at `offset`. Under `direct-io` this bounces through a 4 KiB-aligned buffer so
/// the source is suitably aligned; otherwise it writes `page` directly.
fn write_page_at(file: &File, offset: u64, page: &[u8; PAGE_SIZE]) -> std::io::Result<()> {
    #[cfg(feature = "direct-io")]
    {
        let mut aligned = AlignedPage([0u8; PAGE_SIZE]);
        aligned.0.copy_from_slice(page);
        write_all_at(file, &aligned.0, offset)
    }
    #[cfg(not(feature = "direct-io"))]
    {
        write_all_at(file, page, offset)
    }
}

/// Read exactly `buf.len()` bytes at `offset` using a positioned read (`pread`-style):
/// it does not use or move the file cursor, so concurrent reads on a shared `&File` are
/// safe. On Windows `seek_read` may return short, so loop until the buffer is filled.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        let mut filled = 0usize;
        while let Some(rest) = buf.get_mut(filled..).filter(|r| !r.is_empty()) {
            let n = file.seek_read(rest, offset + filled as u64)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
            }
            filled += n;
        }
        Ok(())
    }
}

/// Write all of `buf` at `offset` using a positioned write (`pwrite`-style): it does not
/// use or move the file cursor. Writing past the end extends the file. On Windows
/// `seek_write` may return short, so loop until the buffer is drained.
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        file.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        let mut written = 0usize;
        while let Some(rest) = buf.get(written..).filter(|r| !r.is_empty()) {
            let n = file.seek_write(rest, offset + written as u64)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::WriteZero));
            }
            written += n;
        }
        Ok(())
    }
}
