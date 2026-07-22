//! Deterministic Simulation Testing (DST) infrastructure.
//!
//! Provides in-memory adapters that fulfill the [`PageStore`](nusadb_core::PageStore),
//! [`Clock`](nusadb_core::Clock), and [`Rng`](nusadb_core::Rng) traits with controllable
//! fault injection. The same engine code runs against real disk in production and
//! against these adapters in DST scenarios — driven entirely by a single seed.
//!
//! See the deterministic-simulation testing docs.

#![warn(missing_docs)]

pub mod clock;
pub mod rng;
pub mod storage;

pub use clock::SimClock;
pub use rng::SimRng;
pub use storage::{FaultRates, FaultingStorage, SimStorage};
