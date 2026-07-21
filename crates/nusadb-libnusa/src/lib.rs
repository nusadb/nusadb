//! libnusa — the reference L1 client for the Nusa Wire Protocol.
//!
//! This crate is the canonical client implementation of `docs/wire-protocol.md`
//! (`PROTOCOL_VERSION 1.0`): the behaviour every language driver mirrors, and the oracle
//! the conformance suite validates against. It speaks the protocol directly over
//! `nusadb-wire`'s frame/message codec — there is no SQL engine in the client.
//!
//! # What it provides
//! - [`Client`] — one authenticated connection: connect (plaintext or TLS), trust-on-startup or
//!   SCRAM-SHA-256 auth, simple query, extended/parameterised query, prepared statements,
//!   transaction helpers, and out-of-band cancellation.
//! - [`Pool`] — a bounded connection pool with connection reuse.
//! - [`Param`] / [`Row`] / [`QueryResult`] — parameter encoding and typed row decoding.
//!
//! # Example
//! ```no_run
//! use nusadb_libnusa::{Client, Config, Param};
//!
//! # async fn run() -> nusadb_libnusa::Result<()> {
//! let config = Config::new("127.0.0.1", 5678, "nusa-root", "nusadb");
//! let mut client = Client::connect(&config).await?;
//! client.simple_query("CREATE TABLE t (id INT NOT NULL, name TEXT)").await?;
//! client.execute("INSERT INTO t VALUES ($1, $2)", &[1_i64.into(), "alice".into()]).await?;
//! let result = client.query("SELECT id, name FROM t WHERE id = $1", &[Param::from(1_i64)]).await?;
//! for row in &result.rows {
//!     let id = row.get_i64(0)?;
//!     let name = row.get_string(1)?;
//!     println!("{id:?} {name:?}");
//! }
//! client.close().await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
// `clippy::redundant_pub_crate` (nursery) and rustc's `unreachable_pub` (workspace-warn) give
// opposite advice for crate-internal items in private modules: the former wants `pub`, the latter
// `pub(crate)`. We keep `pub(crate)` (it states the real visibility) and silence the nursery lint.
#![allow(clippy::redundant_pub_crate)]

mod client;
mod error;
mod pool;
mod transport;
mod value;

pub use client::{BackendKey, Client, Config, Statement};
pub use error::{Error, Result};
pub use pool::{Pool, PooledClient};
pub use transport::tls_client_config;
pub use value::{Param, QueryResult, Row};
