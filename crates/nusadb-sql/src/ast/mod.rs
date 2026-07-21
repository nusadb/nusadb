//! NusaDB internal SQL AST.
//!
//! Owned by us; insulates the rest of the codebase from `sqlparser-rs` upgrades.
//! The [`parser`](crate::parser) module is the only place that converts
//! `sqlparser` types into these — the analyzer, planner, and executor speak this
//! AST exclusively.
//!
//! The AST covers exactly the statements NusaDB's Stage 4 parser accepts today:
//! `CREATE TABLE` / `DROP TABLE` (`IF [NOT] EXISTS`), `INSERT` (multi-row),
//! `SELECT` (projection with aliases, `FROM` with `INNER`/`LEFT`/`RIGHT`/`FULL`
//! joins, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`), `UPDATE`,
//! `DELETE`, `EXPLAIN`, and transaction control (`BEGIN` / `COMMIT` /
//! `ROLLBACK`). Expressions cover the usual operators plus `IS [NOT] NULL`,
//! `[NOT] IN`, `[NOT] BETWEEN`, `[NOT] LIKE`, `CASE`, `COALESCE`, `CAST`, and the
//! aggregates `COUNT`/`SUM`/`AVG`/`MIN`/`MAX`. Anything outside that surface is
//! rejected at the door with [`Error::Unsupported`](crate::error::Error) rather
//! than represented here.
//!
//! The types are grouped into per-concern submodules (ADR 007: `statement`, `ddl`, `dml`,
//! `query`, `expr`) and re-exported here so consumers keep using `crate::ast::*` unchanged.

mod ddl;
mod dml;
mod expr;
mod query;
mod statement;
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
