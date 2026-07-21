//! Shared test helpers used across NusaDB crates (not published).
//!
//! Keep this crate small: helpers that genuinely cross crate boundaries. Single-crate
//! helpers stay in that crate's own `#[cfg(test)]` module.

// This crate is test scaffolding (dev-dependency only); panicking on setup failure is the
// intended behavior, so the production-grade `expect`/panic-doc lints don't apply here.
#![allow(missing_docs, clippy::expect_used, clippy::missing_panics_doc)]

/// Create a temporary directory for tests that need on-disk state.
#[must_use]
pub fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("create temp dir")
}
