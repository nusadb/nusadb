//! Row ↔ columnar conversion shared by the batch adapters and vectorized operators.
//!
//! Two directions, inverses of each other:
//!
//! - `rows_to_batch` transposes decoded [`Row`]s (one [`ast::Value`] per column) into a
//!   typed [`RecordBatch`] (one [`Array`] per column). Used by the scan adapter and by
//!   operators that materialize a filtered/derived result.
//! - `batch_to_rows` reads a [`RecordBatch`] back into [`Row`]s so the row-at-a-time
//!   [`eval`](crate::executor::eval) can evaluate a predicate / projection per row.
//!
//! The element encoding matches the storage tuple codec ([`crate::executor::row`]), so a
//! value survives `decode → rows_to_batch → batch_to_rows` unchanged.

use std::sync::Arc;

use nusadb_core::ColumnType;
use nusadb_core::engine::ArrayElem;

use crate::Row;
use crate::ast;
use crate::batch::{
    Array, ArrayRef, BinaryArray, BooleanArray, DateArray, DecimalArray, Float64Array, Int64Array,
    IntervalArray, JsonArray, ListArray, RecordBatch, Schema, StringArray, TimeArray, TimeTzArray,
    TimestampArray, TimestampTzArray, UuidArray,
};
use crate::error::Error;
use crate::executor::row;

/// Transpose `rows` into a [`RecordBatch`] shaped by `schema`.
///
/// Each row must have one value per field, in field order; a `Null` becomes a null slot.
///
/// # Errors
///
/// [`Error::TypeMismatch`] if a present value's runtime type does not match its column
/// type, or [`Error::ArityMismatch`] / [`Error::TypeMismatch`] from
/// [`RecordBatch::try_new`].
pub(crate) fn rows_to_batch(schema: &Arc<Schema>, rows: Vec<Row>) -> Result<RecordBatch, Error> {
    let row_count = rows.len();
    let mut columns: Vec<Vec<ast::Value>> = schema
        .fields()
        .iter()
        .map(|_| Vec::with_capacity(row_count))
        .collect();
    for row in rows {
        for (idx, value) in row.into_iter().enumerate() {
            if let Some(column) = columns.get_mut(idx) {
                column.push(value);
            }
        }
    }
    columns_to_batch(schema, columns)
}

/// Build a [`RecordBatch`] from already-columnar value accumulators — the tail of
/// [`rows_to_batch`], and the direct entry for the AP3 batch-decode scan, which fills the
/// per-column accumulators straight from the tuple codec without ever materializing rows.
pub(crate) fn columns_to_batch(
    schema: &Arc<Schema>,
    columns: Vec<Vec<ast::Value>>,
) -> Result<RecordBatch, Error> {
    let arrays = schema
        .fields()
        .iter()
        .zip(columns)
        .map(|(field, values)| build_column(field.data_type(), values))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), arrays)
}

/// Build a new [`RecordBatch`] from `batch`'s rows at `indices`, in the given order — so it serves
/// both row selection and reordering, without a round trip through a fully-materialized
/// [`Vec<Row>`]. Only the addressed rows are read back. An index `>= batch.num_rows()` contributes
/// a NULL row (defensive — callers pass in-range indices).
///
/// # Errors
///
/// Propagates the defensive column-build type checks of [`build_column`] (values come from
/// [`value_at`], so a mismatch is not expected in practice).
pub(crate) fn take_batch(batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch, Error> {
    let schema = batch.schema();
    let arrays = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, column)| {
            let values = indices
                .iter()
                .map(|&i| value_at(column.as_ref(), i))
                .collect();
            build_column(field.data_type(), values)
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), arrays)
}

/// Select the rows of `batch` whose `mask` bit is set into a new [`RecordBatch`] (a [`take_batch`]
/// over the surviving indices, preserving order). Only survivors are read back, so a selective
/// filter pays for them rather than the whole batch — the columnar counterpart of `batch_to_rows` +
/// filter + [`rows_to_batch`]. `mask` shorter than the batch treats missing entries as `false`.
///
/// # Errors
///
/// Propagates [`take_batch`]'s column-build type checks.
pub(crate) fn filter_batch(batch: &RecordBatch, mask: &[bool]) -> Result<RecordBatch, Error> {
    let kept: Vec<usize> = (0..batch.num_rows())
        .filter(|&i| mask.get(i).copied().unwrap_or(false))
        .collect();
    take_batch(batch, &kept)
}

/// Read every row of `batch` back into a [`Row`] (one [`ast::Value`] per column).
pub(crate) fn batch_to_rows(batch: &RecordBatch) -> Vec<Row> {
    let columns = batch.columns();
    (0..batch.num_rows())
        .map(|r| columns.iter().map(|c| value_at(c.as_ref(), r)).collect())
        .collect()
}

/// The [`ast::Value`] of `array` at `index` (the inverse of [`build_column`]). A null slot
/// — or a defensively-unexpected downcast failure — yields [`ast::Value::Null`]. Exposed so the
/// vectorized columnar aggregate fold (A-PERF.AGG5b) reads single elements with **exactly** the
/// conversion [`batch_to_rows`] would have used.
pub(crate) fn value_at(array: &dyn Array, index: usize) -> ast::Value {
    if array.is_null(index) {
        return ast::Value::Null;
    }
    let any = array.as_any();
    match array.data_type() {
        ColumnType::Bool => any
            .downcast_ref::<BooleanArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Bool),
        // SMALLINT/BIGINT are materialized as the same 64-bit integer as INT.
        ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => any
            .downcast_ref::<Int64Array>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Int),
        // REAL is materialized as the same 64-bit double as FLOAT.
        ColumnType::Float | ColumnType::Real => any
            .downcast_ref::<Float64Array>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Float),
        // VARCHAR/CHAR are stored and materialized identically to TEXT (`ColumnType::physical`).
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_) => any
            .downcast_ref::<StringArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, |s| ast::Value::Text(s.to_owned())),
        // JSONB is materialized identically to JSON (canonical text).
        ColumnType::Json | ColumnType::Jsonb => any
            .downcast_ref::<JsonArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, |s| ast::Value::Json(s.to_owned())),
        ColumnType::Date => any
            .downcast_ref::<DateArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Date),
        ColumnType::Time => any
            .downcast_ref::<TimeArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Time),
        ColumnType::Timestamp => any
            .downcast_ref::<TimestampArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Timestamp),
        ColumnType::TimestampTz => any
            .downcast_ref::<TimestampTzArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::TimestampTz),
        ColumnType::TimeTz => any
            .downcast_ref::<TimeTzArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::TimeTz),
        ColumnType::Uuid => any
            .downcast_ref::<UuidArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Uuid),
        ColumnType::Interval => any
            .downcast_ref::<IntervalArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Interval),
        ColumnType::Numeric { .. } => any
            .downcast_ref::<DecimalArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, ast::Value::Numeric),
        // BYTEA reads back from the Binary array, mirroring the row codec.
        ColumnType::Bytes => any
            .downcast_ref::<BinaryArray>()
            .and_then(|a| a.get(index))
            .map_or(ast::Value::Null, |b| ast::Value::Bytes(b.to_vec())),
        // VECTOR has no Arrow representation in this columnar path — `build_column` refuses to
        // materialize one — so it only ever yields the null case here.
        ColumnType::Vector(_) => ast::Value::Null,
        ColumnType::Array(_) => {
            let Some(list) = any.downcast_ref::<ListArray>() else {
                return ast::Value::Null;
            };
            let Some((start, end)) = list.value_range(index) else {
                return ast::Value::Null;
            };
            let child = list.child().as_ref();
            let items = (start..end).map(|i| value_at(child, i)).collect();
            ast::Value::Array(items)
        },
    }
}

/// Build the column array for `ty` from one batch's worth of values (one entry per row, in
/// row order). `Null` becomes a null slot; a present value whose runtime type does not
/// match `ty` is a defensive [`Error::TypeMismatch`].
pub(super) fn build_column(ty: ColumnType, values: Vec<ast::Value>) -> Result<ArrayRef, Error> {
    let array: ArrayRef = match ty {
        ColumnType::Bool => Arc::new(BooleanArray::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::Bool(b) => Ok(b),
                other => Err(other),
            },
        )?)),
        // SMALLINT/BIGINT build the same 64-bit integer array as INT.
        ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => Arc::new(
            Int64Array::from_options(collect(values, ty, |v| match v {
                ast::Value::Int(i) => Ok(i),
                other => Err(other),
            })?),
        ),
        // REAL builds the same 64-bit double array as FLOAT.
        ColumnType::Float | ColumnType::Real => Arc::new(Float64Array::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::Float(f) => Ok(f),
                // An INT or NUMERIC value widens into a FLOAT column/expression (a Float-typed
                // expression like `COALESCE(f, 0.5)` evaluates to a NUMERIC literal value). Mirrors
                // the `encode_value` FLOAT arm so the batch and row paths agree.
                #[allow(clippy::cast_precision_loss, reason = "INT→FLOAT column widening")]
                ast::Value::Int(i) => Ok(i as f64),
                ast::Value::Numeric(d) => Ok(d.to_f64()),
                other => Err(other),
            },
        )?)),
        // VARCHAR/CHAR build the same Arrow string array as TEXT (`ColumnType::physical`).
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_) => Arc::new(
            StringArray::from_options(collect(values, ty, |v| match v {
                ast::Value::Text(s) => Ok(s),
                other => Err(other),
            })?),
        ),
        // JSONB builds the same canonical-text array as JSON.
        ColumnType::Json | ColumnType::Jsonb => {
            Arc::new(JsonArray::from_options(collect(values, ty, |v| match v {
                ast::Value::Json(s) => Ok(s),
                other => Err(other),
            })?))
        },
        ColumnType::Date => Arc::new(DateArray::from_options(collect(values, ty, |v| match v {
            ast::Value::Date(d) => Ok(d),
            other => Err(other),
        })?)),
        ColumnType::Time => Arc::new(TimeArray::from_options(collect(values, ty, |v| match v {
            ast::Value::Time(t) => Ok(t),
            other => Err(other),
        })?)),
        ColumnType::Timestamp => Arc::new(TimestampArray::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::Timestamp(t) => Ok(t),
                other => Err(other),
            },
        )?)),
        ColumnType::TimestampTz => Arc::new(TimestampTzArray::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::TimestampTz(t) => Ok(t),
                other => Err(other),
            },
        )?)),
        ColumnType::TimeTz => Arc::new(TimeTzArray::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::TimeTz(t) => Ok(t),
                other => Err(other),
            },
        )?)),
        ColumnType::Uuid => Arc::new(UuidArray::from_options(collect(values, ty, |v| match v {
            ast::Value::Uuid(u) => Ok(u),
            other => Err(other),
        })?)),
        ColumnType::Interval => Arc::new(IntervalArray::from_options(collect(
            values,
            ty,
            |v| match v {
                ast::Value::Interval(iv) => Ok(iv),
                other => Err(other),
            },
        )?)),
        ColumnType::Numeric { precision, scale } => {
            let items = collect(values, ty, |v| match v {
                ast::Value::Numeric(d) => Ok(d),
                other => Err(other),
            })?;
            Arc::new(DecimalArray::from_options(items, precision, scale))
        },
        ColumnType::Bytes => {
            // BYTEA values materialize into a Binary array, mirroring the row codec.
            let items = collect::<Vec<u8>>(values, ty, |v| match v {
                ast::Value::Bytes(b) => Ok(b),
                other => Err(other),
            })?;
            Arc::new(BinaryArray::from_options(items))
        },
        ColumnType::Array(elem) => build_list_column(ty, elem, values)?,
        // VECTOR has no Arrow array yet; refuse loudly rather than silently dropping components to
        // NULL if the (dormant) columnar path is ever wired over a VECTOR column.
        ColumnType::Vector(_) => return Err(vector_not_in_batch()),
    };
    Ok(array)
}

/// The error returned when the columnar batch path meets a VECTOR column. The row path is
/// authoritative for vectors; the Arrow path has no vector array yet.
fn vector_not_in_batch() -> Error {
    Error::Unsupported("VECTOR columns are not supported in the columnar batch path".to_owned())
}

/// Build a nested [`ListArray`] column: flatten every row's elements into one child array
/// (recursively, over the scalar element type) and slice it with per-row offsets.
fn build_list_column(
    ty: ColumnType,
    elem: ArrayElem,
    values: Vec<ast::Value>,
) -> Result<ArrayRef, Error> {
    let mut child_values: Vec<ast::Value> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(values.len() + 1);
    offsets.push(0);
    let mut validity: Vec<bool> = Vec::with_capacity(values.len());
    let mut has_null = false;
    for value in values {
        match value {
            ast::Value::Null => {
                validity.push(false);
                has_null = true;
            },
            ast::Value::Array(items) => {
                child_values.extend(items);
                validity.push(true);
            },
            other => return Err(type_mismatch(ty, &other)),
        }
        offsets.push(child_values.len());
    }
    // The element type is always scalar (`ArrayElem` never nests), so this recursion is
    // one level deep.
    let child = build_column(elem.column_type(), child_values)?;
    let validity_ref = has_null.then_some(validity.as_slice());
    Ok(Arc::new(ListArray::try_new(
        elem,
        child,
        offsets,
        validity_ref,
    )?))
}

/// Collect one column's values into `Vec<Option<T>>`: `Null` → `None`, otherwise apply
/// `extract`, which returns the value back (`Err`) on a type mismatch so the caller can
/// name the offending runtime type.
fn collect<T>(
    values: Vec<ast::Value>,
    ty: ColumnType,
    extract: impl Fn(ast::Value) -> Result<T, ast::Value>,
) -> Result<Vec<Option<T>>, Error> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        match value {
            ast::Value::Null => out.push(None),
            present => match extract(present) {
                Ok(item) => out.push(Some(item)),
                Err(bad) => return Err(type_mismatch(ty, &bad)),
            },
        }
    }
    Ok(out)
}

/// A column value whose runtime type does not match its column type (defensive — values
/// from [`row::decode`] always match).
fn type_mismatch(ty: ColumnType, value: &ast::Value) -> Error {
    Error::TypeMismatch {
        context: "row → record batch column".to_owned(),
        expected: ty,
        found: row::runtime_type_of(value),
    }
}

#[cfg(test)]
mod tests {
    use super::{batch_to_rows, rows_to_batch};
    use crate::ast::Value;
    use crate::batch::{Field, Schema};
    use nusadb_core::ColumnType;
    use nusadb_core::engine::ArrayElem;
    use std::sync::Arc;

    fn schema(fields: Vec<(&str, ColumnType)>) -> Arc<Schema> {
        Arc::new(Schema::new(
            fields
                .into_iter()
                .map(|(n, t)| Field::new(n, t, true))
                .collect(),
        ))
    }

    #[test]
    fn rows_round_trip_through_a_batch() {
        let s = schema(vec![
            ("i", ColumnType::Int),
            ("t", ColumnType::Text),
            ("a", ColumnType::Array(ArrayElem::Int)),
        ]);
        let rows = vec![
            vec![
                Value::Int(1),
                Value::Text("x".to_owned()),
                Value::Array(vec![Value::Int(7), Value::Null]),
            ],
            vec![Value::Null, Value::Null, Value::Array(vec![])],
        ];
        let batch = rows_to_batch(&s, rows.clone()).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch_to_rows(&batch), rows);
    }

    #[test]
    fn filter_batch_keeps_masked_rows_columnarly() {
        use super::filter_batch;
        let s = schema(vec![("i", ColumnType::Int), ("t", ColumnType::Text)]);
        let rows = vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Null, Value::Text("b".to_owned())],
            vec![Value::Int(3), Value::Null],
        ];
        let batch = rows_to_batch(&s, rows).unwrap();
        // Keep rows 0 and 2; drop row 1.
        let filtered = filter_batch(&batch, &[true, false, true]).unwrap();
        assert_eq!(filtered.num_rows(), 2);
        assert_eq!(
            batch_to_rows(&filtered),
            vec![
                vec![Value::Int(1), Value::Text("a".to_owned())],
                vec![Value::Int(3), Value::Null],
            ],
        );
        // A short mask drops the unaddressed tail.
        let none = filter_batch(&batch, &[]).unwrap();
        assert_eq!(none.num_rows(), 0);
    }

    #[test]
    fn take_batch_reorders_and_repeats_rows() {
        use super::take_batch;
        let s = schema(vec![("i", ColumnType::Int)]);
        let batch = rows_to_batch(
            &s,
            vec![
                vec![Value::Int(10)],
                vec![Value::Int(20)],
                vec![Value::Int(30)],
            ],
        )
        .unwrap();
        // Reverse order, with a repeat and an out-of-range index (→ NULL).
        let taken = take_batch(&batch, &[2, 0, 0, 9]).unwrap();
        assert_eq!(
            batch_to_rows(&taken),
            vec![
                vec![Value::Int(30)],
                vec![Value::Int(10)],
                vec![Value::Int(10)],
                vec![Value::Null],
            ],
        );
    }

    #[test]
    fn type_mismatch_is_rejected() {
        let s = schema(vec![("i", ColumnType::Int)]);
        let err = rows_to_batch(&s, vec![vec![Value::Text("nope".to_owned())]]);
        assert!(matches!(err, Err(crate::error::Error::TypeMismatch { .. })));
    }
}
