//! L7 — Physical storage layer.
//!
//! Provides the 8 KiB page format, disk manager (production [`nusadb_core::PageStore`] adapter),
//! B-tree index pages, and the catalog. This crate is **sync** — it does not depend
//! on `tokio` or any async runtime.
//!
//! # Stage
//!
//! the physical storage layer (pages, disk manager, buffer pool, catalog).

#![warn(missing_docs)]

pub mod btree;
pub mod buffer_pool;
pub mod catalog;
pub mod disk;
pub mod page;
pub mod toast;

pub use btree::BTree;
pub use buffer_pool::{BufferPool, BufferPoolStats, PageGuard};
pub use catalog::Catalog;
pub use disk::DiskManager;
pub use page::{Page, PageHeader};
pub use toast::{Toast, ToastPointer};
