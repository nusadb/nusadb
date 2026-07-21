//! Columnar batch model — the vectorized executor's unit of data.
//!
//! The Stage 4 executor processes data column-at-a-time in fixed-size batches of
//! [`BATCH_SIZE`](crate::BATCH_SIZE) rows rather than one [`Row`](crate::Row) at a
//! time. A [`RecordBatch`] is a [`Schema`] paired with one columnar [`Array`] per
//! field; every operator consumes and produces `RecordBatch`es.
//!
//! This module defines only the *containers* and the [`Array`] contract:
//!
//! - [`Field`] / [`Schema`] — the typed shape of a batch.
//! - [`Array`] / [`ArrayRef`] — the trait every concrete column array implements
//!   (length, null-tracking, dynamic type, downcast).
//! - [`PrimitiveArray`] — fixed-width columns ([`Int64Array`], [`Float64Array`],
//!   [`BooleanArray`]).
//! - [`StringArray`] / [`BinaryArray`] — variable-length columns (offsets + data).
//! - [`TemporalArray`] ([`DateArray`], [`TimeArray`], [`TimestampArray`],
//!   [`TimestampTzArray`]) and [`IntervalArray`] — date/time columns.
//! - [`DecimalArray`] ([`ColumnType::Numeric`](nusadb_core::ColumnType::Numeric), carries
//!   precision/scale) and [`UuidArray`] (fixed 16-byte).
//! - [`JsonArray`] ([`ColumnType::Json`](nusadb_core::ColumnType::Json)) and [`ListArray`]
//!   (nested [`ColumnType::Array`](nusadb_core::ColumnType::Array), offsets + child). The
//!   bit-packed validity buffer lands in
//! - [`RecordBatch`] — schema + columns, with construction-time invariant checks.
//!
//! Dynamic dispatch through [`ArrayRef`] is per-batch, not per-row, so its cost is
//! amortized over [`BATCH_SIZE`](crate::BATCH_SIZE) rows.

mod array;
mod bytes;
pub mod convert;
mod decimal;
mod from_scan;
mod json;
mod list;
mod primitive;
mod record_batch;
mod schema;
mod temporal;
mod uuid;
mod validity;

pub use array::{Array, ArrayRef};
pub use bytes::{BinaryArray, StringArray};
pub use decimal::DecimalArray;
pub use from_scan::{RecordBatchScan, schema_from_columns};
pub use json::JsonArray;
pub use list::ListArray;
pub use primitive::{BooleanArray, Float64Array, Int64Array, PrimitiveArray, PrimitiveType};
pub use record_batch::RecordBatch;
pub use schema::{Field, Schema};
pub use temporal::{
    DateArray, DateKind, IntervalArray, TemporalArray, TemporalKind, TimeArray, TimeKind,
    TimeTzArray, TimeTzKind, TimestampArray, TimestampKind, TimestampTzArray, TimestampTzKind,
};
pub use uuid::{Uuid, UuidArray};
