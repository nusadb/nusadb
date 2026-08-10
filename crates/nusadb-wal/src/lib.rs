//! L6 — Write-ahead log.
//!
//! Provides an append-only sequence of WAL records with CRC32 framing and optional
//! lz4 compression. Pages live in memory, and this log is their durable backing: recovery
//! replays it **from the beginning of the file** until the first CRC mismatch (treated as the
//! end of the durable prefix). This crate itself has no checkpoint concept — it only appends and
//! replays. The engine layer (`nusadb-btree`) implements checkpointing *on top* of this log by
//! persisting a page image beside it and then truncating this file, so in a checkpointed database
//! the file holds only the records written since the last checkpoint; LSNs keep counting across
//! that truncation, so records after a checkpoint still sort after the image they follow.
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
