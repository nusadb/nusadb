//! In-memory page cache (buffer pool) over a [`PageStore`].
//!
//! Sits between higher layers and a [`PageStore`] adapter (the production
//! [`DiskManager`](crate::DiskManager) or the simulator's store): callers fetch
//! pages by [`PageId`] and receive a pinned [`PageGuard`]. The pool keeps a fixed
//! number of 8 KiB frames resident, writing dirty pages back to the store on
//! eviction or flush. Eviction uses the clock (second-chance) algorithm.
//!
//! # Concurrency: per-frame latches
//!
//! Two lock tiers keep cache misses from serializing on a single mutex:
//!
//! - A small **meta mutex** (`Meta`) guards only the page table, the clock hand, and each
//!   frame's bookkeeping (`id`, `pins`, `referenced`). Critical sections under it are O(1) and
//!   never perform I/O.
//! - Each frame's **bytes** live behind their own [`RwLock`] (`FrameData`), the *frame latch*.
//!   Page reads/writes take only this latch, and the load/flush I/O of a miss runs while holding
//!   it with the meta mutex released — so two misses on different frames proceed in parallel, and
//!   a miss never blocks an unrelated hit behind disk I/O.
//!
//! The lock order is always **meta → frame latch** (the loader and the flushers both acquire meta
//! first; `PageGuard::read`/`write` take only the frame latch), so there is no cycle. A victim is
//! only ever a frame with `pins == 0`, which means no live guard and therefore no in-flight
//! `read`/`write` on it — so the loader's frame-latch acquisition is uncontended, and a concurrent
//! fetch of the *same* page that pins the reserved frame will simply block on the frame latch until
//! the load completes (it cannot observe half-loaded bytes).

// Frame indices are bounded by the fixed pool capacity: the clock hand is always
// reduced modulo the frame count, and the page table only stores indices that name
// a live frame. Indexing into `frames`/`slots` is therefore in range by construction.
#![allow(clippy::indexing_slicing)]
// The meta mutex and the frame latch are deliberately held across multi-step critical sections —
// most importantly, `fetch_page` takes the frame latch *while still holding meta* and only then
// drops meta, which is the load-bearing ordering that prevents a racer from reading half-loaded
// bytes (see the module docs). Tightening these drop scopes (what this lint suggests) would
// reintroduce the race. Every drop here is deliberate.
#![allow(clippy::significant_drop_tightening)]

use std::collections::HashMap;

use nusadb_core::{Error, PAGE_SIZE, PageId, PageStore, Result};
use parking_lot::{Mutex, RwLock};

/// A fixed-capacity page cache layered over a [`PageStore`].
pub struct BufferPool<S: PageStore> {
    store: S,
    /// Per-frame bookkeeping + the page table; small, I/O-free critical sections.
    meta: Mutex<Meta>,
    /// Per-frame bytes, each behind its own latch. Same length as `Meta::slots`.
    frames: Vec<RwLock<FrameData>>,
}

/// Frame bookkeeping guarded by the meta mutex.
struct Meta {
    slots: Vec<SlotMeta>,
    /// Maps each resident page to its frame index.
    table: HashMap<PageId, usize>,
    /// Clock hand for second-chance eviction.
    hand: usize,
    /// Cache-hit / miss counters. Incremented under this mutex on every `fetch_page`,
    /// so they cost nothing extra and stay consistent.
    hits: u64,
    misses: u64,
}

/// A snapshot of a [`BufferPool`]'s cache statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    /// `fetch_page` calls served from a resident frame (no I/O).
    pub hits: u64,
    /// `fetch_page` calls that had to load the page from the store.
    pub misses: u64,
}

impl BufferPoolStats {
    /// Cache hit ratio in `[0, 1]`, or `0.0` before any fetch.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "observability ratio; precision loss only past 2^52 fetches is irrelevant"
    )]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Per-frame metadata (everything but the bytes).
struct SlotMeta {
    /// Resident page, or `None` for an empty frame.
    id: Option<PageId>,
    /// Outstanding [`PageGuard`] count; a pinned frame is never evicted.
    pins: u32,
    /// Clock reference bit: set on access, cleared on a clock sweep.
    referenced: bool,
}

impl SlotMeta {
    const fn empty() -> Self {
        Self {
            id: None,
            pins: 0,
            referenced: false,
        }
    }
}

/// The cached bytes of one frame plus its dirty bit, guarded by the per-frame latch.
struct FrameData {
    bytes: [u8; PAGE_SIZE],
    /// Whether `bytes` holds unwritten modifications.
    dirty: bool,
}

impl FrameData {
    const fn empty() -> Self {
        Self {
            bytes: [0u8; PAGE_SIZE],
            dirty: false,
        }
    }
}

impl<S: PageStore> BufferPool<S> {
    /// Create a pool of `capacity` frames over `store`. `capacity` is clamped to
    /// at least one frame.
    #[must_use]
    pub fn new(store: S, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, SlotMeta::empty);
        let mut frames = Vec::with_capacity(capacity);
        frames.resize_with(capacity, || RwLock::new(FrameData::empty()));
        Self {
            store,
            meta: Mutex::new(Meta {
                hits: 0,
                misses: 0,
                slots,
                table: HashMap::new(),
                hand: 0,
            }),
            frames,
        }
    }

    /// Number of frames in the pool.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.frames.len()
    }

    /// Cache hit/miss statistics since the pool was created.
    #[must_use]
    pub fn stats(&self) -> BufferPoolStats {
        let meta = self.meta.lock();
        BufferPoolStats {
            hits: meta.hits,
            misses: meta.misses,
        }
    }

    /// Borrow the underlying store.
    pub const fn store(&self) -> &S {
        &self.store
    }

    /// Fetch `id`, loading it from the store on a cache miss (evicting a frame
    /// when the pool is full). The returned guard pins the page until dropped.
    ///
    /// On a miss the load (and any write-back of the evicted page) runs with the meta mutex
    /// released, holding only the reserved frame's latch — so concurrent fetches of other pages
    /// are not blocked behind this I/O.
    ///
    /// # Errors
    /// Propagates store I/O errors, or fails if every frame is currently pinned.
    pub fn fetch_page(&self, id: PageId) -> Result<PageGuard<'_, S>> {
        let mut meta = self.meta.lock();
        // Fast path: a cache hit pins the frame and returns immediately (no I/O).
        if let Some(&idx) = meta.table.get(&id) {
            meta.hits += 1;
            let slot = &mut meta.slots[idx];
            slot.pins += 1;
            slot.referenced = true;
            return Ok(PageGuard {
                pool: self,
                frame: idx,
            });
        }
        meta.misses += 1;

        // Miss: pick and reserve a victim, then take its frame latch *before* releasing the meta
        // mutex. The victim had `pins == 0`, so no live guard (hence no in-flight read/write)
        // exists for it and the latch is uncontended. Holding it before releasing `meta` closes
        // the window in which a concurrent fetch of `id` could pin the reserved frame and read its
        // stale (evicted-page) bytes — that fetch will block on this latch until the load lands.
        let victim = meta.choose_victim().ok_or_else(|| {
            Error::Io(std::io::Error::other(
                "buffer pool exhausted: all frames pinned",
            ))
        })?;
        let evicted = meta.slots[victim].id;
        if let Some(old) = evicted {
            meta.table.remove(&old);
        }
        meta.table.insert(id, victim);
        meta.slots[victim] = SlotMeta {
            id: Some(id),
            pins: 1,
            referenced: true,
        };
        let mut data = self.frames[victim].write();
        drop(meta);

        // Flush the evicted page (if dirty) before reusing the frame, with the meta mutex released.
        // If the write-back fails, the dirty bytes are still the *only* copy — dropping them (by
        // emptying the slot) would silently lose the write. Instead keep the victim resident and
        // dirty so a later flush/checkpoint can retry it.
        if let Some(old) = evicted
            && data.dirty
        {
            if let Err(e) = self.store.write_page(old, &data.bytes) {
                drop(data);
                let mut meta = self.meta.lock();
                meta.table.remove(&id);
                meta.table.insert(old, victim);
                meta.slots[victim] = SlotMeta {
                    id: Some(old),
                    pins: 0,
                    referenced: true,
                };
                return Err(e);
            }
            data.dirty = false;
        }

        // Victim is clean (or there was none); load the requested page into the frame.
        match self.store.read_page(id) {
            Ok(bytes) => {
                data.bytes = bytes;
                data.dirty = false;
                drop(data);
                Ok(PageGuard {
                    pool: self,
                    frame: victim,
                })
            },
            Err(e) => {
                drop(data);
                // The victim was already flushed (or absent); the requested page never loaded, so
                // the frame holds no unpersisted data — return it to the free pool.
                let mut meta = self.meta.lock();
                meta.table.remove(&id);
                meta.slots[victim] = SlotMeta::empty();
                Err(e)
            },
        }
    }

    /// Allocate a fresh page in the store and fetch it (pinned).
    ///
    /// # Errors
    /// Propagates store I/O errors, or fails if every frame is currently pinned.
    pub fn new_page(&self) -> Result<(PageId, PageGuard<'_, S>)> {
        let id = self.store.allocate_page()?;
        let guard = self.fetch_page(id)?;
        Ok((id, guard))
    }

    /// Write `id` back to the store if it is resident and dirty, clearing its
    /// dirty bit. A no-op when `id` is not resident or already clean.
    ///
    /// The write-back I/O runs with the meta mutex released (the frame is pinned across it so it
    /// cannot be evicted), so it does not block concurrent fetches.
    ///
    /// # Errors
    /// Propagates store I/O errors.
    pub fn flush_page(&self, id: PageId) -> Result<()> {
        let Some(idx) = self.pin_resident(id) else {
            return Ok(());
        };
        let result = self.flush_frame(idx, id);
        self.unpin(idx);
        result
    }

    /// Write every resident dirty page back to the store, clearing dirty bits.
    ///
    /// Frames are flushed one at a time, each pinned only for its own write-back, so this never
    /// holds the meta mutex across I/O.
    ///
    /// # Errors
    /// Propagates store I/O errors, stopping at the first failure.
    pub fn flush_all(&self) -> Result<()> {
        for idx in 0..self.frames.len() {
            // Re-read the resident id under the mutex each iteration (it may have changed).
            let id = {
                let mut meta = self.meta.lock();
                match meta.slots[idx].id {
                    Some(id) => {
                        meta.slots[idx].pins += 1;
                        id
                    },
                    None => continue,
                }
            };
            let result = self.flush_frame(idx, id);
            self.unpin(idx);
            result?;
        }
        Ok(())
    }

    /// Pin `id`'s frame if resident, returning its index. Pinning prevents eviction while the
    /// caller does I/O with the mutex released.
    fn pin_resident(&self, id: PageId) -> Option<usize> {
        let mut meta = self.meta.lock();
        let &idx = meta.table.get(&id)?;
        meta.slots[idx].pins += 1;
        Some(idx)
    }

    /// Write frame `idx` back to `id` if dirty, clearing the dirty bit. Takes the frame latch.
    fn flush_frame(&self, idx: usize, id: PageId) -> Result<()> {
        let mut data = self.frames[idx].write();
        if data.dirty {
            self.store.write_page(id, &data.bytes)?;
            data.dirty = false;
        }
        Ok(())
    }

    /// Drop one pin from frame `idx`.
    fn unpin(&self, idx: usize) {
        let mut meta = self.meta.lock();
        let pins = &mut meta.slots[idx].pins;
        *pins = pins.saturating_sub(1);
    }
}

impl Meta {
    /// Pick a frame to (re)use via the clock algorithm: skip pinned frames, give a
    /// second chance to referenced ones (clearing the bit), and reuse the first
    /// unpinned, unreferenced frame. Returns `None` if every frame is pinned.
    fn choose_victim(&mut self) -> Option<usize> {
        let n = self.slots.len();
        // Two full sweeps guarantee termination: the first clears reference bits,
        // so by the second every unpinned frame is a candidate.
        for _ in 0..(2 * n) {
            let idx = self.hand;
            self.hand = (idx + 1) % n;
            let slot = &mut self.slots[idx];
            if slot.pins > 0 {
                continue;
            }
            if slot.referenced {
                slot.referenced = false;
                continue;
            }
            return Some(idx);
        }
        None
    }
}

impl<S: PageStore> std::fmt::Debug for BufferPool<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Skip the frame bytes and the (possibly non-Debug) store; report only the
        // shape of the cache.
        let meta = self.meta.lock();
        f.debug_struct("BufferPool")
            .field("capacity", &meta.slots.len())
            .field("resident", &meta.table.len())
            .finish_non_exhaustive()
    }
}

/// A pinned handle to a resident page. The page stays in the pool until the guard
/// is dropped; access the bytes through [`PageGuard::read`] / [`PageGuard::write`].
///
/// Do not call other [`BufferPool`] methods from inside a `read`/`write` closure — the frame
/// latch is held for the duration of the closure and is not reentrant.
#[must_use = "dropping the guard immediately unpins the page"]
pub struct PageGuard<'pool, S: PageStore> {
    pool: &'pool BufferPool<S>,
    frame: usize,
}

impl<S: PageStore> PageGuard<'_, S> {
    /// Read the page bytes under the frame's read latch.
    pub fn read<R>(&self, f: impl FnOnce(&[u8; PAGE_SIZE]) -> R) -> R {
        let data = self.pool.frames[self.frame].read();
        f(&data.bytes)
    }

    /// Mutate the page bytes under the frame's write latch, marking the frame dirty.
    pub fn write<R>(&self, f: impl FnOnce(&mut [u8; PAGE_SIZE]) -> R) -> R {
        let mut data = self.pool.frames[self.frame].write();
        data.dirty = true;
        f(&mut data.bytes)
    }
}

impl<S: PageStore> std::fmt::Debug for PageGuard<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageGuard")
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}

impl<S: PageStore> Drop for PageGuard<'_, S> {
    fn drop(&mut self) {
        self.pool.unpin(self.frame);
    }
}
