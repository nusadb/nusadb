//! Spill-to-disk infrastructure for external-memory execution.
//!
//! The row-path executor is materialize-based — `execute_op` returns a full `Vec<Row>` — so a
//! blocking operator (sort / hash-join build / group-aggregate) over an input larger than memory
//! would OOM. This module is the substrate the spilling operators build on; the consumers
//! (streaming `RowSource`, grace hash join, external merge sort, …) land in later commits per
//! the storage design. This first
//! commit is the foundation only:
//!
//! - [`SpillConfig`] / [`set_spill_config`] — the per-process knob (mirrors `set_work_mem`) naming
//!   a scratch directory and the in-memory threshold at which an operator spills.
//! - [`SpillWriter`] / [`SpillReader`] — append rows to a transient on-disk run/partition file and
//!   read them back. The file is removed when the handle drops (RAII), so a spill never outlives
//!   its query even on error or panic.
//! - [`MemBudget`] — tracks the bytes an operator currently holds so it knows when to spill.
//! - a self-describing row codec (every [`ast::Value`](crate::ast::Value) variant) so intermediate
//!   rows — whose types are not a catalog tuple schema — round-trip through disk exactly.

mod budget;
mod codec;
mod context;
mod file;

pub use context::{SpillConfig, set_spill_config};

// Consumed by the spilling operators (`join.rs` grace join, `spill_sort.rs` external sort);
// re-exported to the executor module so siblings can reach them without the longer
// `spill::<submodule>::` paths.
pub(in crate::executor) use budget::MemBudget;
pub(in crate::executor) use context::spill_config;
pub(in crate::executor) use file::{SpillReader, SpillWriter};
