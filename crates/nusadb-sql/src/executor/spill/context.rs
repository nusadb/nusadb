//! The per-process spill configuration knob.
//!
//! Mirrors `work_mem` (`executor/ops.rs`): the server sets it once at startup; the executor reads
//! it when dispatching a blocking operator. It is a process-wide `RwLock` rather than a thread-local
//! because the async server may run one query's stages across worker threads — a thread-local could
//! be read on a thread that never saw the `set`. Reads are rare (once per blocking operator) so the
//! lock is uncontended.

use std::path::PathBuf;
use std::sync::RwLock;

/// Enables spill-to-disk for the executor: where to put transient files and how much an operator
/// may hold in memory before spilling the overflow.
#[derive(Clone, Debug)]
pub struct SpillConfig {
    /// Directory transient spill files are created in (e.g. `<data-dir>/tmp`). Must exist and be
    /// writable; the operators clean up their own files via `SpillWriter`/`SpillReader` RAII.
    pub dir: PathBuf,
    /// Bytes a single operator may hold in memory before spilling the overflow to disk. Usually the
    /// query's `work_mem`, but kept separate so tests can force a low threshold on small inputs.
    pub threshold_bytes: usize,
}

static SPILL_CONFIG: RwLock<Option<SpillConfig>> = RwLock::new(None);

/// Enable or disable spill-to-disk for this process.
///
/// `Some` turns it on; `None` (the default) keeps blocking operators on their in-memory path,
/// failing via `work_mem` rather than spilling. The server wires this to `--spill-dir`. A poisoned
/// lock (a prior panic while held) is ignored — the lock guards only the trivial store/load here.
pub fn set_spill_config(config: Option<SpillConfig>) {
    if let Ok(mut guard) = SPILL_CONFIG.write() {
        *guard = config;
    }
}

/// The active spill configuration, or `None` when spill-to-disk is disabled.
pub(in crate::executor) fn spill_config() -> Option<SpillConfig> {
    SPILL_CONFIG.read().ok().and_then(|g| g.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_clear_round_trips() {
        // The only test that touches the process-wide config; restores `None` so it cannot leak
        // into other tests.
        assert!(spill_config().is_none(), "default is disabled");
        set_spill_config(Some(SpillConfig {
            dir: PathBuf::from("scratch"),
            threshold_bytes: 4096,
        }));
        let got = spill_config().expect("just set");
        assert_eq!(got.dir, PathBuf::from("scratch"));
        assert_eq!(got.threshold_bytes, 4096);
        set_spill_config(None);
        assert!(spill_config().is_none(), "cleared");
    }
}
