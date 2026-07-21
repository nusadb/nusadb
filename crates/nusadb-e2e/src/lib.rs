//! End-to-end integration tests for NusaDB.
//!
//! This crate intentionally has no library surface of its own. Its purpose is the integration
//! tests under `tests/`, which drive the full `nusadb-sql` pipeline
//! (`parse → analyze → plan → execute`) against the production
//! `nusadb-btree::BtreeEngine` — the sole shipping engine — proving the SQL surface and the
//! storage spine converge over the `StorageEngine` treaty.
