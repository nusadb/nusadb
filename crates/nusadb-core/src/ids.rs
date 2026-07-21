//! Strongly-typed identifiers used across the engine.
//!
//! Domain-specific newtypes prevent accidental mixing of `PageId` with `Lsn` or `TxnId`.
//! All four are 64-bit, `Copy`, and zero-cost at runtime.

use bytemuck::{Pod, Zeroable};

/// Identifies a single 8 KiB page in physical storage.
///
/// Page IDs are dense, monotonically increasing, and **never reused** during Stage 1.
/// Free-list reuse is introduced later under the same `PageId` type.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct PageId(pub u64);

/// Log Sequence Number — monotonic position in the WAL.
///
/// LSNs are the canonical ordering reference for "which write happened first" across
/// the entire engine. Recovery replays from the last clean LSN forward.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct Lsn(pub u64);

/// Transaction identifier — assigned monotonically at transaction start.
///
/// Used as `xmin` / `xmax` for MVCC visibility and as the key for lock manager
/// entries.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct TxnId(pub u64);

/// Slot index inside a page's slot array.
///
/// Together with [`PageId`], `(PageId, SlotIdx)` forms a TID — the stable physical
/// address of a tuple even after page reorganization.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct SlotIdx(pub u16);

/// Identifies a table within the catalog.
///
/// Assigned monotonically by the catalog at `CREATE TABLE` and used as the stable
/// reference to a table in the [`StorageEngine`](crate::engine::StorageEngine) treaty.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct TableId(pub u64);

/// Identifies a sequence object within the catalog.
///
/// Assigned monotonically at `CREATE SEQUENCE` and used as the stable reference to a sequence in
/// the [`StorageEngine`](crate::engine::StorageEngine) treaty (`SERIAL`/`IDENTITY` desugar onto a
/// sequence). Unlike row writes, a sequence's value advances **non-transactionally**.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct SequenceId(pub u64);

/// Identifies a SQL **schema** (namespace) within the catalog.
///
/// Assigned monotonically at `CREATE SCHEMA`. This is the SQL *namespace* a table/sequence can live
/// in — not to be confused with [`TableSchema`](crate::engine::TableSchema), which is one table's
/// column layout. The session's `search_path` (which schemas an unqualified name resolves against)
/// is the SQL layer's state, not the engine's.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct SchemaId(pub u64);

/// Identifies a secondary index within the catalog.
///
/// Assigned monotonically at `CREATE INDEX`. The SQL layer encodes the indexed columns into the
/// opaque key bytes it hands the engine; the engine stores `key → Tid` under this id without ever
/// decoding a tuple.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct IndexId(pub u64);
