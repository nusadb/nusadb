//! NusaDB internal SQL AST.
//!
//! Owned by us; insulates the rest of the codebase from `sqlparser-rs` upgrades.
//! The [`parser`](crate::parser) module is the only place that converts
//! `sqlparser` types into these — the analyzer, planner, and executor speak this
//! AST exclusively.
//!
//! The AST covers exactly the statements NusaDB's parser accepts: DDL, DML, queries, access
//! control and transaction control. Anything outside that surface is rejected at the door with
//! [`Error::Unsupported`](crate::error::Error) rather than represented here.
//!
//! The types are grouped into per-concern submodules (ADR 007: `statement`, `ddl`, `dml`,
//! `query`, `expr`) and re-exported here so consumers keep using `crate::ast::*` unchanged.

mod dcl;
mod ddl;
mod dml;
mod expr;
mod query;
mod statement;
#[allow(clippy::wildcard_imports)]
pub use dcl::*;
#[allow(clippy::wildcard_imports)]
pub use ddl::*;
#[allow(clippy::wildcard_imports)]
pub use dml::*;
#[allow(clippy::wildcard_imports)]
pub use expr::*;
#[allow(clippy::wildcard_imports)]
pub use query::*;
#[allow(clippy::wildcard_imports)]
pub use statement::*;
