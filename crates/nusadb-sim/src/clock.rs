//! Deterministic clock backed by an atomic counter.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use nusadb_core::Clock;

/// Deterministic monotonic clock. Time advances only when [`Self::sleep`] is called.
#[derive(Debug, Default)]
pub struct SimClock {
    ticks: AtomicU64,
}

impl Clock for SimClock {
    fn now_ticks(&self) -> u64 {
        self.ticks.load(Ordering::SeqCst)
    }

    fn sleep(&self, duration: Duration) {
        let ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.ticks.fetch_add(ns, Ordering::SeqCst);
    }
}
