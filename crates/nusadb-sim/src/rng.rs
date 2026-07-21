//! Seeded RNG for deterministic test runs.
//!
//! Uses xorshift64\* — sufficient for fault scheduling, not for cryptography.

use nusadb_core::Rng;

/// Seeded RNG. Pure function of its initial seed.
#[derive(Debug, Clone)]
pub struct SimRng {
    state: u64,
}

impl SimRng {
    /// Create a new RNG with the given non-zero seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        // Avoid the all-zero state (xorshift64 fixed point).
        let state = if seed == 0 {
            0xdead_beef_cafe_babe
        } else {
            seed
        };
        Self { state }
    }
}

impl Rng for SimRng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[allow(clippy::cast_precision_loss)]
    fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits → uniform in [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0f64 / ((1u64 << 53) as f64))
    }
}
