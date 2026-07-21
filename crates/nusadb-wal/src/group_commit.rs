//! Group commit: coalesce many transactions' `fsync`s into one.
//!
//! `fsync` is the dominant cost of a durable commit. When many transactions commit concurrently,
//! they do not each need their own `fsync` — a single `fsync` makes *every* WAL byte written so far
//! durable, so it can satisfy a whole batch of waiting committers at once. [`GroupCommit`]
//! coordinates that with a **leader/follower** protocol: the first committer to find no `fsync` in
//! flight becomes the *leader* and performs one `fsync` covering everything appended up to that
//! moment; every other committer whose required durability point is covered by that `fsync` is a
//! *follower* and returns without issuing its own.
//!
//! Durability is tracked by a monotonic **sequence number** — in NusaDB the WAL [`Lsn`](nusadb_core::Lsn).
//! A committer appends its records (assigning it some `seq`), then calls [`GroupCommit::commit`]
//! with that `seq` and a closure that flushes + `fsync`s the log and returns the highest seq it made
//! durable. The closure runs **outside** the coordinator lock (only the leader calls it), so other
//! committers keep appending and queueing while the `fsync` is in flight, and the next leader sweeps
//! them all up.
//!
//! ```text
//! t0  T1 appends seq=1, becomes leader, fsync begins ............ (covers up to seq=1)
//! t1  T2 appends seq=2 ─┐
//! t2  T3 appends seq=3 ─┴─ both wait (leader busy)
//! t3  leader's fsync returns durable=1; T1 done
//! t4  T2 wakes: durable(1) < 2 → becomes leader, fsync begins ... (covers up to seq=3)
//! t5  fsync returns durable=3; T2 and T3 both done   ← 2 fsyncs served 3 commits
//! ```
//!
//! This type only sequences the `fsync`; appending to the WAL is still the caller's responsibility
//! and must be serialized by the caller (the [`WalWriter`](crate::WalWriter) is `&mut`).

use std::io;

use parking_lot::{Condvar, Mutex};

/// Coordinator that coalesces concurrent durability requests into shared `fsync`s.
#[derive(Debug, Default)]
pub struct GroupCommit {
    inner: Mutex<Inner>,
    /// Signalled when a leader finishes an `fsync` (durability advanced) so followers re-check.
    progress: Condvar,
}

#[derive(Debug, Default)]
struct Inner {
    /// Highest sequence number known to be durable on disk.
    durable: u64,
    /// Whether a leader is currently performing an `fsync`.
    syncing: bool,
}

impl GroupCommit {
    /// Create a coordinator with nothing yet durable.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The highest sequence number currently known to be durable.
    #[must_use]
    pub fn durable_seq(&self) -> u64 {
        self.inner.lock().durable
    }

    /// Ensure the log is durable through at least `seq`, coalescing with concurrent callers.
    ///
    /// Returns once `seq` is durable. If a concurrent leader's `fsync` already covered `seq`, this
    /// returns without calling `flush` at all. Otherwise this call becomes the leader: it invokes
    /// `flush` exactly once (outside the internal lock) to flush + `fsync` the log, and `flush`
    /// must return the highest sequence number it made durable (typically the latest appended
    /// `Lsn`, which is `>= seq`).
    ///
    /// # Errors
    /// Propagates the error from `flush`. On error, durability is left unchanged and waiting
    /// followers are woken to retry (one of them becomes the next leader).
    pub fn commit<F>(&self, seq: u64, flush: F) -> io::Result<()>
    where
        F: FnOnce() -> io::Result<u64>,
    {
        {
            let mut inner = self.inner.lock();
            // Wait until either our seq is durable (a leader covered us) or no leader is running
            // (so we may become the leader ourselves). The loop body only waits — `flush` is
            // called below, exactly once, so it can stay `FnOnce`.
            loop {
                if inner.durable >= seq {
                    return Ok(());
                }
                if !inner.syncing {
                    break;
                }
                self.progress.wait(&mut inner);
            }
            inner.syncing = true;
        }

        // Leader: fsync outside the lock so concurrent committers keep appending + queueing.
        let result = flush();

        let mut inner = self.inner.lock();
        inner.syncing = false;
        let ret = match result {
            Ok(made_durable) => {
                inner.durable = inner.durable.max(made_durable);
                Ok(())
            },
            Err(e) => Err(e),
        };
        drop(inner);
        // Wake every follower: those now covered return, the rest elect the next leader.
        self.progress.notify_all();
        ret
    }
}
