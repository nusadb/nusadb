//! Statement-thread PRNG for `RANDOM()` / `SETSEED()`.
//!
//! `RANDOM()` is volatile — a fresh value on every call — so, unlike the statement-stable clock
//! ([`super::clock`]), the generator *advances* per call. Its state lives in a thread-local
//! (`nusadb-sql` is single-threaded per statement, like the clock), seeded on first use from the
//! statement wall clock so independent runs differ, or pinned deterministically by `SETSEED` for
//! reproducibility. The algorithm is `SplitMix64` — tiny, fast, and well-distributed enough for a SQL
//! `RANDOM()` (it is not, and need not be, cryptographic).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the 53-bit mantissa fill is exact; SETSEED's u64::MAX scaling is approximate by design \
              and its operand is clamped non-negative"
)]

use std::cell::Cell;

thread_local! {
    /// `SplitMix64` state. `None` until the first `RANDOM()` (seeded from the clock) or a `SETSEED`.
    static STATE: Cell<Option<u64>> = const { Cell::new(None) };
}

/// One `SplitMix64` step over the stored state, returning the next 64 random bits and advancing.
fn next_u64() -> u64 {
    STATE.with(|cell| {
        // First use without SETSEED: seed from the statement clock (forced odd, never zero).
        let state = cell
            .get()
            .unwrap_or_else(|| super::clock::statement_now_micros().unsigned_abs() | 1);
        let next = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        cell.set(Some(next));
        let mut z = next;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// A uniform `f64` in `[0, 1)` — the value of `RANDOM()`.
pub(super) fn next_f64() -> f64 {
    // The top 53 bits map exactly onto an `f64` mantissa, giving a uniform value in `[0, 1)`.
    (next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Pin the generator seed — `SETSEED(x)` with `x` clamped to `[-1, 1]` per SQL.
pub(super) fn set_seed(x: f64) {
    let unit = f64::midpoint(x.clamp(-1.0, 1.0), 1.0); // → [0, 1]
    let scaled = (unit * u64::MAX as f64) as u64 | 1; // forced odd, never zero
    STATE.with(|cell| cell.set(Some(scaled)));
}

#[cfg(test)]
mod tests {
    use super::{next_f64, set_seed};

    #[test]
    fn random_is_in_unit_interval() {
        set_seed(0.25);
        for _ in 0..1000 {
            let v = next_f64();
            assert!((0.0..1.0).contains(&v), "RANDOM() out of [0,1): {v}");
        }
    }

    #[test]
    fn setseed_makes_the_sequence_reproducible() {
        set_seed(0.5);
        let first: Vec<f64> = (0..5).map(|_| next_f64()).collect();
        set_seed(0.5);
        let second: Vec<f64> = (0..5).map(|_| next_f64()).collect();
        assert_eq!(first, second);
        // A different seed produces a different sequence.
        set_seed(-0.5);
        let third: Vec<f64> = (0..5).map(|_| next_f64()).collect();
        assert_ne!(first, third);
    }
}
