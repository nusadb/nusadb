//! L6 — Write-ahead log.
//!
//! Provides an append-only sequence of WAL records with CRC32 framing and optional
//! lz4 compression. Recovery replays records from the last clean checkpoint until
//! the first CRC mismatch (treated as the end of the durable log).
//!
//! # Stage
//!
//! the write-ahead logging layer.

#![warn(missing_docs)]

pub mod crypto;
pub mod group_commit;
pub mod reader;
pub mod record;
pub mod writer;

pub use group_commit::GroupCommit;
pub use reader::{MidLogHole, ReadOutcome, RecoveredPrefix, WalReader, recover_prefix};
pub use record::WalRecord;
pub use writer::WalWriter;
