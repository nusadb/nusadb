//! Core types and traits shared across all NusaDB crates.
//!
//! This is the **innermost** crate in NusaDB's clean architecture: every other crate
//! depends on `nusadb-core`, and `nusadb-core` depends on nothing internal. This is where
//! cross-cutting value types ([`PageId`], [`Lsn`], [`TxnId`]) and the load-bearing trait
//! abstractions ([`PageStore`], [`Clock`], [`Rng`]) live.
//!
//! # Layer
//!
//! `nusadb-core` defines the **ports** consumed by higher layers; the concrete adapters
//! live in `nusadb-storage` (production) and `nusadb-sim` (deterministic simulation).
//!
//! # Stability
//!
//! Public items here are part of the most-frequently-imported surface. Breaking changes
//! cascade across the whole workspace — bump the workspace MAJOR version.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod error;
pub mod ids;
pub mod traits;

pub use engine::{
    AlterOp, ArrayElem, ColumnDef, ColumnStats, ColumnType, Constraint, ConstraintKind, FkAction,
    ForeignKeyDef, IndexDef, IndexKind, IsolationLevel, PUBLIC_SCHEMA, SharedTuple, StorageEngine,
    TableDef, TableSchema, TableStats, Tid, Tuple, TupleScan,
};
pub use error::{Error, Result};
pub use ids::{IndexId, Lsn, PageId, SchemaId, SequenceId, SlotIdx, TableId, TxnId};
pub use traits::{Clock, PageStore, Rng};

/// The fixed page size used everywhere in NusaDB. 8 KiB.
///
/// This number is load-bearing: B-tree node = 1 page; buffer pool frame = 1 page;
/// disk I/O is aligned to multiples of this.
pub const PAGE_SIZE: usize = 8192;
