//! RAM-aware auto-tuning of the engine's memory knobs (model + derivation).
//!
//! NusaDB's defaults are permissive — `work_mem = 0` (unbounded) and spill off — which is fine on a
//! big host but makes a large analytic query OOM-prone on a small one (the cloud free-tier 1 vCPU /
//! 1 GB target). The spill-to-disk machinery is the cure, but only if it is *on by default*
//! and the per-query budget is *bounded*. This module turns one input — a total **memory budget** —
//! into safe knob values so graceful degradation is the out-of-the-box behaviour.
//!
//! # The allocation model
//!
//! The budget must cover everything resident at once. The worst case is every connection running a
//! blocking operator (sort / aggregate / join) at its `work_mem` cap simultaneously, on top of the
//! engine's resident pages and MVCC version metadata. We keep the part this module controls
//! comfortably under the budget and reserve the rest for the engine and process overhead:
//!
//! ```text
//!   work_mem × max_connections   ≤  40% budget   (all queries spilling at their cap at once)
//!   ── reserved (not derived here) ──
//!   engine pages + version store + overhead   ≈  remaining ≥ 60% budget
//! ```
//!
//! The split is deliberately conservative for a 1 GB box. Spill is enabled with the per-query
//! `work_mem` as its threshold, so a query exceeding its share streams to disk instead of failing.
//! (The removed predecessor engine also took a write-buffer flush threshold and a footprint cap from this
//! budget; the btree engine has no such knobs — its version-store reclamation is the purge
//! scheduler, wired at the composition root.)
//!
//! All arithmetic is integer (`budget / 100 * pct`) so there is no float rounding or cast in the
//! derivation.

#![allow(
    clippy::redundant_pub_crate,
    reason = "this is a private module of the server binary; its items are `pub(crate)` so the \
              crate root (main.rs) can use them, which `unreachable_pub` requires — the two lints \
              are mutually exclusive here"
)]

/// Knob values derived from a memory budget. Bytes throughout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DerivedKnobs {
    /// Per-query work-memory cap (also the spill threshold). A blocking operator over this spills.
    pub(crate) work_mem: usize,
    /// Default per-transaction uncommitted-write ceiling (`--max-txn-write-bytes`): a single
    /// transaction whose buffered writes exceed this is failed loudly instead of growing the
    /// process until the host OOM-kills every client. Derived as 25% of the budget, floored at
    /// [`TXN_WRITE_FLOOR`], so one runaway bulk transaction cannot take the server down. This is a
    /// *per-transaction* bound, not a global one; an explicit `--max-txn-write-bytes` overrides it.
    pub(crate) max_txn_write_bytes: usize,
    /// Default global resident-memory ceiling (`--max-resident-bytes`) for the in-memory page store:
    /// once its total footprint reaches this, a row insert is failed loudly instead of the store
    /// growing until the host OOM-kills the server. Unlike [`max_txn_write_bytes`](Self::max_txn_write_bytes)
    /// (one in-flight transaction) this bounds *committed-resident* data across the whole store —
    /// the streamed-bulk-load case that accumulates past the per-transaction ceiling. Derived from
    /// the engine's ~60% page share of the budget divided by [`RESIDENT_RSS_FACTOR`] (the meter
    /// counts logical page bytes, but the real footprint runs larger), floored at [`RESIDENT_FLOOR`],
    /// so the ceiling trips while the real footprint is still within budget rather than after the OS
    /// kills the process; an explicit `--max-resident-bytes` overrides it.
    pub(crate) max_resident_bytes: usize,
    /// Default cap on the bytes the wire layer buffers for one `COPY ... FROM STDIN` before loading
    /// it (`--copy-max-bytes` when left unset). That buffer is transient and, unlike the two write
    /// ceilings above, is not charged against the store, so a fixed cap plus a near-full resident
    /// store can together exceed RAM on a memory-constrained host and get the process OOM-killed
    /// before the load's own inserts are rejected. Derived as 20% of the budget so it scales with the
    /// host, capped at [`COPY_MAX_CEILING`] (the historical fixed default, kept for roomy hosts) and
    /// floored at [`COPY_MAX_FLOOR`] so a small host still accepts a routine load; an explicit
    /// `--copy-max-bytes` (including `0` for unbounded) overrides it.
    pub(crate) max_copy_bytes: usize,
}

/// Floor for the per-query work-memory cap: below this, sort/agg/join become impractically spill-
/// bound, so a very small budget × many connections clamps here (and over-commits the work pool —
/// the caller logs a warning when that happens).
const WORK_MEM_FLOOR: usize = 1 << 20; // 1 MiB

/// Floor for the derived per-transaction write ceiling: below this a legitimate medium transaction
/// (a moderate bulk load) would be rejected, so a small budget clamps the ceiling up to here rather
/// than throttling normal work. 128 MiB leaves comfortable room while still bounding the multi-GB
/// runaways the ceiling exists to stop.
const TXN_WRITE_FLOOR: usize = 128 << 20; // 128 MiB

/// Floor for the derived global resident-memory ceiling: below this a legitimate small dataset would
/// be rejected, so a small budget clamps the ceiling up to here rather than throttling normal work.
/// 256 MiB holds a comfortably large in-memory dataset while still bounding the multi-GB runaway a
/// bulk load can become on a memory-constrained host.
const RESIDENT_FLOOR: usize = 256 << 20; // 256 MiB

/// How much larger the real resident-memory footprint runs than the logical page bytes the engine
/// meters when it decides whether to accept a write: allocator overhead, MVCC version chains, index
/// nodes, and fragmentation the logical count does not see, plus the transient buffers a bulk load
/// holds alongside the store (its parse buffer and per-batch index entries). The derived resident
/// ceiling is divided by this so the logical ceiling trips while the real footprint is still within
/// the budget — a graceful reject beats an out-of-memory kill. Bulk loads of wide rows were measured
/// near 3x, so err toward that: over-reserving costs some headroom, under-reserving costs the process.
const RESIDENT_RSS_FACTOR: usize = 3;

/// Upper bound for the derived single-`COPY` buffer cap: the historical fixed default. A host with
/// ample RAM keeps this cap; only a smaller budget derives a lower one.
const COPY_MAX_CEILING: usize = 1 << 30; // 1 GiB

/// Floor for the derived single-`COPY` buffer cap: below this a routine bulk load would be split
/// needlessly, so a small budget clamps up to here rather than throttling normal loads.
const COPY_MAX_FLOOR: usize = 64 << 20; // 64 MiB

/// Derive the memory knobs from `budget` (bytes) and the configured `max_connections`.
///
/// `max_connections` is treated as at least 1. Uses the percentages in the module-level allocation
/// model; the per-query `work_mem` is `40% budget / max_connections`, floored at [`WORK_MEM_FLOOR`],
/// and the per-transaction write ceiling is `25% budget`, floored at [`TXN_WRITE_FLOOR`].
#[must_use]
pub(crate) fn derive(budget: usize, max_connections: usize) -> DerivedKnobs {
    let conns = max_connections.max(1);
    let work_pool = budget / 100 * 40;
    let work_mem = (work_pool / conns).max(WORK_MEM_FLOOR);
    // The per-transaction write ceiling is a flat 25% of the budget (not divided per connection —
    // it bounds any *one* transaction), floored so a small budget does not reject normal work.
    let max_txn_write_bytes = (budget / 100 * 25).max(TXN_WRITE_FLOOR);
    // The global resident ceiling bounds committed data across the whole store. The engine's page
    // share of the budget is ~60%, but it meters *logical* page bytes while the OS limits real
    // resident memory, which runs about [`RESIDENT_RSS_FACTOR`]x larger — so the logical ceiling is
    // that share divided by the factor. The ceiling then trips (a graceful reject) while the real
    // footprint is still within budget, instead of the OS killing the process first. Floored so a
    // small budget does not reject normal work.
    let max_resident_bytes = (budget / 100 * 60 / RESIDENT_RSS_FACTOR).max(RESIDENT_FLOOR);
    // The single-COPY buffer cap is 20% of the budget so it scales down on a small host, then bounded
    // to [`COPY_MAX_FLOOR`, `COPY_MAX_CEILING`]: the ceiling keeps the historical 1 GiB on a roomy
    // host, the floor keeps a routine load working on a small one.
    let max_copy_bytes = (budget / 100 * 20).clamp(COPY_MAX_FLOOR, COPY_MAX_CEILING);
    DerivedKnobs {
        work_mem,
        max_txn_write_bytes,
        max_resident_bytes,
        max_copy_bytes,
    }
}

/// Whether the derived per-query `work_mem` was forced up to the floor, so the worst-case work pool
/// (`work_mem × max_connections`) over-commits its 40% share of `budget`. The caller warns when this
/// holds: on a tiny budget with many connections, concurrent large queries can still exceed the
/// budget — the operator should lower `--max-connections` or raise `--mem-budget`.
#[must_use]
pub(crate) fn work_pool_overcommits(
    knobs: DerivedKnobs,
    budget: usize,
    max_connections: usize,
) -> bool {
    let conns = max_connections.max(1);
    knobs.work_mem.saturating_mul(conns) > budget / 100 * 40
}

/// Detect the memory budget available to this process, in bytes.
///
/// On Linux this is `min(host RAM, cgroup memory limit)` — the cgroup limit is what actually bounds a
/// container, and on a cloud free-tier it is usually far below host RAM, so ignoring it would
/// over-commit and invite the OOM killer. Returns `None` when no budget can be determined (a
/// non-Linux host, or `/proc`/`/sys` unreadable); the caller then leaves auto-tuning off and honours
/// only explicit flags.
#[must_use]
#[allow(
    clippy::missing_const_for_fn,
    reason = "reads /proc/meminfo + cgroup files on Linux (not const); the lint only fires on the \
              non-Linux stub, which trivially returns None"
)]
pub(crate) fn detect_budget() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let host = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| parse_meminfo_total(&s));
        let cgroup = detect_cgroup_limit();
        match (host, cgroup) {
            (Some(h), Some(c)) => Some(h.min(c)),
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The cgroup memory limit (v2 `memory.max`, then v1 `memory.limit_in_bytes`), or `None` if neither
/// is present or both are "unlimited".
#[cfg(target_os = "linux")]
fn detect_cgroup_limit() -> Option<usize> {
    const V2: &str = "/sys/fs/cgroup/memory.max";
    const V1: &str = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
    for path in [V2, V1] {
        if let Ok(contents) = std::fs::read_to_string(path)
            && let Some(limit) = parse_cgroup_limit(&contents)
        {
            return Some(limit);
        }
    }
    None
}

/// Parse `MemTotal:` (in kB) from `/proc/meminfo` contents into bytes.
#[cfg(any(target_os = "linux", test))]
fn parse_meminfo_total(contents: &str) -> Option<usize> {
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

/// Parse a cgroup memory-limit file. `"max"` (v2) or an implausibly huge sentinel (v1's
/// `PAGE_COUNTER_MAX`-derived value, ≥ 2^53) means "unlimited" → `None`; otherwise the byte value.
#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_limit(contents: &str) -> Option<usize> {
    let trimmed = contents.trim();
    if trimmed == "max" {
        return None;
    }
    let value: usize = trimmed.parse().ok()?;
    // v1 reports a near-`u64::MAX` value when unlimited; treat anything ≥ 2^53 as no real cap.
    if value >= 1 << 53 { None } else { Some(value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_meminfo_total() {
        let sample = "MemTotal:        1024000 kB\nMemFree:          512000 kB\n";
        assert_eq!(parse_meminfo_total(sample), Some(1_024_000 * 1024));
        assert_eq!(parse_meminfo_total("MemFree: 1 kB"), None);
        assert_eq!(parse_meminfo_total("garbage"), None);
    }

    #[test]
    fn parses_cgroup_limit() {
        assert_eq!(parse_cgroup_limit("1073741824\n"), Some(1_073_741_824));
        assert_eq!(parse_cgroup_limit("max\n"), None); // v2 unlimited
        assert_eq!(parse_cgroup_limit("9223372036854771712"), None); // v1 ~unlimited sentinel
        assert_eq!(parse_cgroup_limit("not-a-number"), None);
    }

    #[test]
    fn derivation_respects_the_allocation_model() {
        // At 1 GiB / 25 connections the controllable resident total stays within budget and work_mem
        // is small enough that large queries spill.
        let budget = 1 << 30; // 1 GiB
        let k = derive(budget, 25);
        assert!(k.work_mem >= WORK_MEM_FLOOR);
        // Worst case (all connections at their cap) must stay within the work pool's budget share.
        let worst = k.work_mem * 25;
        assert!(
            worst <= budget / 100 * 40,
            "worst-case work pool {worst} exceeds its share of budget {budget}"
        );
        assert!(!work_pool_overcommits(k, budget, 25));
    }

    #[test]
    fn large_budget_does_not_starve_work_mem() {
        // On a big host the per-query budget scales up (no regression: queries are not over-throttled).
        let k = derive(64usize << 30, 25); // 64 GiB
        assert!(
            k.work_mem > 100 << 20,
            "work_mem should be generous on a big host"
        );
    }

    #[test]
    fn tiny_budget_clamps_and_flags_overcommit() {
        // 128 MiB / 200 connections: 40% / 200 ≈ 256 KiB < floor, so work_mem clamps up to the floor
        // and the worst-case pool (floor × 200) over-commits its 40% share — which must be flagged.
        let budget = 128 << 20;
        let k = derive(budget, 200);
        assert_eq!(k.work_mem, WORK_MEM_FLOOR);
        assert!(work_pool_overcommits(k, budget, 200));
    }

    #[test]
    fn zero_connections_is_treated_as_one() {
        let k = derive(1 << 30, 0);
        assert!(k.work_mem >= WORK_MEM_FLOOR);
    }

    #[test]
    fn derives_the_per_transaction_write_ceiling_at_25_percent_with_a_floor() {
        let ceiling = |budget: usize| derive(budget, 25).max_txn_write_bytes;
        // ~25% of the budget once above the floor (integer arithmetic, like work_mem's 40%): the
        // directive's 8 GiB→2 GiB, 64 GiB→16 GiB shape, allowing for `budget / 100 * 25` rounding.
        for &gib in &[2usize, 8, 64] {
            let budget = gib << 30;
            assert_eq!(
                ceiling(budget),
                budget / 100 * 25,
                "25% of a {gib} GiB budget"
            );
            assert!(
                ceiling(budget) > TXN_WRITE_FLOOR,
                "a {gib} GiB budget derives above the floor"
            );
        }
        // A small budget clamps up to the 128 MiB floor rather than rejecting normal work.
        assert_eq!(ceiling(512 << 20), TXN_WRITE_FLOOR, "512 MiB → floor");
        assert_eq!(ceiling(128 << 20), TXN_WRITE_FLOOR, "tiny budget → floor");
        // The ceiling bounds one transaction, so it does not depend on the connection count.
        assert_eq!(ceiling(8 << 30), derive(8 << 30, 200).max_txn_write_bytes);
    }

    #[test]
    fn derives_the_global_resident_ceiling_from_the_rss_adjusted_page_share() {
        let ceiling = |budget: usize| derive(budget, 25).max_resident_bytes;
        // The ~60% page share divided by the RSS factor, once above the floor — the logical ceiling
        // trips before the real footprint (which runs larger) reaches the budget.
        for &gib in &[2usize, 8, 64] {
            let budget = gib << 30;
            assert_eq!(
                ceiling(budget),
                budget / 100 * 60 / RESIDENT_RSS_FACTOR,
                "RSS-adjusted page share of a {gib} GiB budget"
            );
            assert!(
                ceiling(budget) > RESIDENT_FLOOR,
                "a {gib} GiB budget derives above the floor"
            );
            // The real footprint (~factor x the logical ceiling) still leaves the budget headroom.
            assert!(
                ceiling(budget) * RESIDENT_RSS_FACTOR <= budget,
                "the RSS-adjusted ceiling keeps the real footprint within a {gib} GiB budget"
            );
        }
        // A small budget clamps up to the 256 MiB floor rather than rejecting normal work.
        assert_eq!(ceiling(256 << 20), RESIDENT_FLOOR, "small budget → floor");
        // The resident ceiling bounds the whole store, so it does not depend on the connection count.
        assert_eq!(ceiling(8 << 30), derive(8 << 30, 200).max_resident_bytes);
    }

    #[test]
    fn derives_the_copy_buffer_cap_scaled_to_the_budget() {
        let copy = |budget: usize| derive(budget, 25).max_copy_bytes;
        // A roomy host keeps the historical 1 GiB cap (20% of the budget would exceed it).
        assert_eq!(
            copy(64 << 30),
            COPY_MAX_CEILING,
            "big host keeps the 1 GiB cap"
        );
        assert_eq!(copy(8 << 30), COPY_MAX_CEILING, "8 GiB: 20% > 1 GiB → cap");
        // A mid host scales the cap down to ~20% of the budget, below the fixed 1 GiB.
        let mid = 3usize << 30; // 3 GiB
        assert_eq!(copy(mid), mid / 100 * 20, "3 GiB → ~20% of the budget");
        assert!(copy(mid) < COPY_MAX_CEILING);
        // A tiny budget clamps up to the floor rather than rejecting a routine load.
        assert_eq!(copy(128 << 20), COPY_MAX_FLOOR, "tiny budget → floor");
        // The reason for scaling: the transient COPY buffer plus the resident-store ceiling must
        // leave headroom under the budget, so a bulk COPY on a constrained host is rejected
        // gracefully instead of pushing the process past the limit and being OOM-killed. (Checked on
        // budgets large enough that neither derivation hits its floor.)
        for &gib in &[2usize, 3, 4, 8] {
            let budget = gib << 30;
            let k = derive(budget, 25);
            assert!(
                k.max_copy_bytes + k.max_resident_bytes <= budget,
                "copy {} + resident {} exceeds a {gib} GiB budget",
                k.max_copy_bytes,
                k.max_resident_bytes
            );
        }
    }
}
