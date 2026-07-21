//! Adapt a storage [`TupleScan`] into a stream of columnar [`RecordBatch`]es.
//!
//! The [`StorageEngine`](nusadb_core::StorageEngine) treaty serves rows one opaque
//! tuple at a time ([`TupleScan`]); the vectorized executor wants them column-at-a-time
//! in fixed-size batches of [`BATCH_SIZE`](crate::BATCH_SIZE) rows. [`RecordBatchScan`]
//! bridges the two: it pulls up to `batch_size` tuples, decodes each with the shared
//! tuple codec ([`row::decode`](crate::executor::row)), transposes the rows into one
//! typed [`Array`](crate::batch::Array) per column, and yields a [`RecordBatch`].
//!
//! Decoding reuses the single source-of-truth row codec rather than re-implementing it,
//! so the columnar path and the row-at-a-time path stay byte-for-byte consistent.

use std::sync::Arc;

use nusadb_core::ColumnType;
use nusadb_core::engine::{ColumnDef, TupleScan};

use crate::ast;
use crate::batch::bytes::StringBuilder;
use crate::batch::{BooleanArray, Field, Float64Array, Int64Array, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::row;

/// Build the columnar [`Schema`] for a table from its catalog column definitions.
///
/// The batch produced by a [`RecordBatchScan`] over that table has exactly these fields,
/// in declaration order.
#[must_use]
pub fn schema_from_columns(columns: &[ColumnDef]) -> Schema {
    Schema::new(
        columns
            .iter()
            .map(|c| Field::new(c.name.clone(), c.ty, c.nullable))
            .collect(),
    )
}

/// A forward-only adapter that turns a [`TupleScan`] into a [`RecordBatch`] iterator.
///
/// Each [`next`](Iterator::next) pulls up to `batch_size` tuples from the underlying scan,
/// decodes them, and assembles one [`RecordBatch`]. The final batch may be shorter than
/// `batch_size`; an exhausted scan yields no trailing empty batch. After the underlying
/// scan ends — or after any error — the iterator is fused (yields `None`).
pub struct RecordBatchScan {
    scan: Box<dyn TupleScan>,
    schema: Arc<Schema>,
    column_types: Vec<ColumnType>,
    /// Projection pushdown: the full source tuple's column types when the scan is
    /// narrowed — `keep` selects which of them feed the (narrowed) `schema`'s builders.
    /// An empty `keep` is the identity scan: every source column is a schema column.
    /// Shared slices so a caller building many short-lived scans (the parallel workers, one
    /// per chunk) clones a refcount, not the buffers.
    source_types: Arc<[ColumnType]>,
    keep: Arc<[usize]>,
    batch_size: usize,
    done: bool,
}

impl RecordBatchScan {
    /// Wrap `scan`, emitting batches of [`BATCH_SIZE`](crate::BATCH_SIZE) rows shaped by
    /// `schema`. The schema's field types drive how each column's tuple bytes are decoded.
    #[must_use]
    pub fn new(scan: Box<dyn TupleScan>, schema: Arc<Schema>) -> Self {
        Self::with_batch_size(scan, schema, crate::BATCH_SIZE)
    }

    /// Like [`new`](Self::new) but with an explicit batch size (clamped to at least 1).
    #[must_use]
    pub fn with_batch_size(
        scan: Box<dyn TupleScan>,
        schema: Arc<Schema>,
        batch_size: usize,
    ) -> Self {
        let column_types = schema.fields().iter().map(Field::data_type).collect();
        Self {
            scan,
            schema,
            column_types,
            source_types: Arc::from([]),
            keep: Arc::from([]),
            batch_size: batch_size.max(1),
            done: false,
        }
    }

    /// A **projected** scan: each source tuple is encoded under `source_types`, and
    /// only the ascending `keep` ordinals become batch columns — `schema` must be exactly
    /// those kept fields, in order. Unkept fields are decoded and dropped, mirroring the row
    /// path's `decode_projected` (same walk, same malformed-tuple and
    /// validation errors), so the two projected paths cannot disagree. An empty `keep` is the
    /// identity scan, exactly [`new`](Self::new).
    #[must_use]
    pub fn with_projection(
        scan: Box<dyn TupleScan>,
        schema: Arc<Schema>,
        source_types: Arc<[ColumnType]>,
        keep: Arc<[usize]>,
    ) -> Self {
        let mut this = Self::new(scan, schema);
        if !keep.is_empty() {
            this.source_types = source_types;
            this.keep = keep;
        }
        this
    }

    /// Pull and decode the next batch, or `Ok(None)` once the scan is exhausted.
    ///
    /// AP3 batch-decode (R2 stage 2): each tuple's fields append straight to per-column TYPED
    /// builders — the hot fixed-width types (integers, floats, booleans) parse through the same
    /// leaf readers `decode_value` uses and land as raw `Option<T>` without ever constructing an
    /// `ast::Value`; every other type falls back to the shared value codec and the ordinary
    /// column build. No per-row `Vec`, no transpose pass, no boxing on the hot types.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.done {
            return Ok(None);
        }
        let mut builders: Vec<ColumnBuilder> = self
            .column_types
            .iter()
            .map(|&ty| ColumnBuilder::new(ty, self.batch_size))
            .collect();
        let mut filled = 0usize;
        while filled < self.batch_size {
            let Some((_, tuple)) = self.scan.try_next()? else {
                self.done = true;
                break;
            };
            let mut pos = 0;
            if self.keep.is_empty() {
                for builder in &mut builders {
                    pos = builder.append_field(&tuple, pos)?;
                }
            } else {
                // Projected walk: every source field advances the cursor (an unkept field is
                // *skipped* — `decode_projected` parity, offsets and errors identical); the kept
                // fields feed the narrowed builders in ascending order.
                let mut want = self.keep.iter().peekable();
                let mut kept = builders.iter_mut();
                for (idx, &ty) in self.source_types.iter().enumerate() {
                    if want.peek().is_some_and(|&&w| w == idx) {
                        want.next();
                        let builder = kept.next().ok_or(Error::MalformedTuple { offset: idx })?;
                        pos = builder.append_field(&tuple, pos)?;
                    } else {
                        let (present, payload) = row::field_tag(&tuple, pos)?;
                        // Advance past a dropped field without materializing it — a dropped blob
                        // (TEXT/BYTEA/JSON) never allocates the String/Vec a full decode would build
                        // and discard, matching the row path's `decode_projected` skip byte-for-byte.
                        pos = if present {
                            row::skip_value(&tuple, payload, ty)?
                        } else {
                            payload
                        };
                    }
                }
            }
            filled += 1;
        }
        if filled == 0 {
            return Ok(None);
        }
        let arrays = builders
            .into_iter()
            .map(ColumnBuilder::finish)
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), arrays).map(Some)
    }
}

/// One column's batch accumulator (R2 stage 2). The fixed-width variants hold exactly the
/// `Vec<Option<T>>` their typed [`Array`](crate::batch::Array) constructor takes; `Values` is the
/// fallback through the shared value codec + [`build_column`](super::convert::columns_to_batch)
/// tail for every other type — so the fast path and the fallback cannot disagree on the format
/// (both parse through `executor::row`'s single set of readers).
enum ColumnBuilder {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    /// Text-family columns (R2 stage 2b): bytes append straight into the offsets+data buffers —
    /// no per-value `String`.
    Text(StringBuilder),
    Values {
        ty: ColumnType,
        values: Vec<ast::Value>,
    },
}

impl ColumnBuilder {
    fn new(ty: ColumnType, capacity: usize) -> Self {
        match ty {
            ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => {
                Self::Int(Vec::with_capacity(capacity))
            },
            ColumnType::Float | ColumnType::Real => Self::Float(Vec::with_capacity(capacity)),
            ColumnType::Bool => Self::Bool(Vec::with_capacity(capacity)),
            // VARCHAR/CHAR are stored and built identically to TEXT (`ColumnType::physical`),
            // mirroring `build_column`'s mapping exactly.
            ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_) => {
                Self::Text(StringBuilder::with_capacity(capacity))
            },
            _ => Self::Values {
                ty,
                values: Vec::with_capacity(capacity),
            },
        }
    }

    /// Append one field from `bytes` at `pos` (the tag byte), returning the next position.
    fn append_field(&mut self, bytes: &[u8], pos: usize) -> Result<usize, Error> {
        let (present, payload) = row::field_tag(bytes, pos)?;
        if !present {
            match self {
                Self::Int(v) => v.push(None),
                Self::Float(v) => v.push(None),
                Self::Bool(v) => v.push(None),
                Self::Text(b) => b.append_null(),
                Self::Values { values, .. } => values.push(ast::Value::Null),
            }
            return Ok(payload);
        }
        match self {
            Self::Int(v) => {
                let (value, next) = row::read_i64_field(bytes, payload)?;
                v.push(Some(value));
                Ok(next)
            },
            Self::Float(v) => {
                let (value, next) = row::read_f64_field(bytes, payload)?;
                v.push(Some(value));
                Ok(next)
            },
            Self::Bool(v) => {
                let (value, next) = row::read_bool_field(bytes, payload)?;
                v.push(Some(value));
                Ok(next)
            },
            Self::Text(b) => {
                let (text, next) = row::read_text_field(bytes, payload)?;
                b.append_value(text);
                Ok(next)
            },
            Self::Values { ty, values } => {
                let (value, next) = row::decode_present_value(bytes, payload, *ty)?;
                values.push(value);
                Ok(next)
            },
        }
    }

    /// Turn the accumulator into its typed column array.
    fn finish(self) -> Result<crate::batch::ArrayRef, Error> {
        Ok(match self {
            Self::Int(v) => Arc::new(Int64Array::from_options(v)),
            Self::Float(v) => Arc::new(Float64Array::from_options(v)),
            Self::Bool(v) => Arc::new(BooleanArray::from_options(v)),
            Self::Text(b) => Arc::new(b.finish()),
            Self::Values { ty, values } => super::convert::build_column(ty, values)?,
        })
    }
}

impl std::fmt::Debug for RecordBatchScan {
    // `Box<dyn TupleScan>` is not `Debug`, so the scan cursor is shown as opaque.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchScan")
            .field("schema", &self.schema)
            .field("column_types", &self.column_types)
            .field("batch_size", &self.batch_size)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl Iterator for RecordBatchScan {
    type Item = Result<RecordBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(err) => {
                self.done = true;
                Some(Err(err))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RecordBatchScan, schema_from_columns};
    use crate::ast::Value;
    use crate::batch::{Array, Int64Array, ListArray, StringArray};
    use crate::executor::row;
    use nusadb_core::engine::{ArrayElem, ColumnDef, SharedTuple, Tid, TupleScan};
    use nusadb_core::{ColumnType, PageId, Result as CoreResult, SlotIdx};
    use std::sync::Arc;

    /// A [`TupleScan`] over a fixed list of pre-encoded tuples.
    struct VecScan {
        tuples: Vec<SharedTuple>,
        pos: usize,
    }

    impl TupleScan for VecScan {
        fn try_next(&mut self) -> CoreResult<Option<(Tid, SharedTuple)>> {
            let item = self.tuples.get(self.pos).map(|t| {
                (
                    Tid {
                        page: PageId(0),
                        slot: SlotIdx(0),
                    },
                    Arc::clone(t),
                )
            });
            if item.is_some() {
                self.pos += 1;
            }
            Ok(item)
        }
    }

    fn col(name: &str, ty: ColumnType) -> ColumnDef {
        ColumnDef {
            name: name.to_owned(),
            ty,
            nullable: true,
        }
    }

    /// Encode `rows` against `types` into a scan over the resulting tuples.
    fn scan_of(rows: Vec<Vec<Value>>, types: &[ColumnType]) -> VecScan {
        let tuples = rows
            .into_iter()
            .map(|r| SharedTuple::from(row::encode(&r, types).unwrap().as_slice()))
            .collect();
        VecScan { tuples, pos: 0 }
    }

    /// R2 stage 2: the typed-builder batch must be cell-for-cell identical to what the
    /// row-decode + transpose path produces — across the fast fixed-width types (int family,
    /// float family, bool), the fallback types (text, numeric, date), and NULLs in every column.
    #[test]
    fn typed_builders_match_row_decode_transpose() {
        let columns = [
            col("i", ColumnType::Int),
            col("i2", ColumnType::Int),
            col("f", ColumnType::Float),
            col("f2", ColumnType::Float),
            col("b", ColumnType::Bool),
            col("t", ColumnType::Text),
            col(
                "n",
                ColumnType::Numeric {
                    precision: 10,
                    scale: 2,
                },
            ),
            col("d", ColumnType::Date),
        ];
        let types: Vec<ColumnType> = columns.iter().map(|c| c.ty).collect();
        let num = |m: i128, s: u8| {
            Value::Numeric(crate::numeric::Decimal {
                mantissa: m,
                scale: s,
            })
        };
        let rows = vec![
            vec![
                Value::Int(7),
                Value::Int(-2),
                Value::Float(1.5),
                Value::Float(-0.25),
                Value::Bool(true),
                Value::Text("alpha".to_owned()),
                num(12345, 2),
                Value::Date(19700),
            ],
            vec![Value::Null; 8],
            vec![
                Value::Int(i64::MAX),
                Value::Int(0),
                Value::Float(0.0),
                Value::Null,
                Value::Bool(false),
                Value::Text(String::new()),
                Value::Null,
                Value::Null,
            ],
        ];
        let schema = Arc::new(schema_from_columns(&columns));

        // Reference: the row path (decode each tuple, then transpose).
        let decoded: Vec<Vec<Value>> = rows
            .iter()
            .map(|r| row::decode(&row::encode(r, &types).unwrap(), &types).unwrap())
            .collect();
        let reference = crate::batch::convert::rows_to_batch(&schema, decoded).unwrap();

        // Under test: the typed-builder scan path.
        let scan = RecordBatchScan::new(Box::new(scan_of(rows, &types)), Arc::clone(&schema));
        let batches: Vec<_> = scan.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), reference.num_rows());
        assert_eq!(batch.num_columns(), reference.num_columns());
        for c in 0..batch.num_columns() {
            for r in 0..batch.num_rows() {
                let got = crate::batch::convert::value_at(batch.column(c).unwrap().as_ref(), r);
                let want =
                    crate::batch::convert::value_at(reference.column(c).unwrap().as_ref(), r);
                assert_eq!(got, want, "column {c} row {r}");
            }
        }
    }

    #[test]
    fn batches_scalar_columns_with_nulls() {
        let columns = [col("id", ColumnType::Int), col("name", ColumnType::Text)];
        let types: Vec<ColumnType> = columns.iter().map(|c| c.ty).collect();
        let rows = vec![
            vec![Value::Int(1), Value::Text("a".to_owned())],
            vec![Value::Int(2), Value::Null],
        ];
        let schema = Arc::new(schema_from_columns(&columns));
        let scan = RecordBatchScan::new(Box::new(scan_of(rows, &types)), Arc::clone(&schema));

        let batches: Vec<_> = scan.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let ids = batch
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.get(0), Some(1));
        assert_eq!(ids.get(1), Some(2));

        let names = batch
            .column(1)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.get(0), Some("a"));
        assert!(names.is_null(1));
    }

    #[test]
    fn splits_into_batches_at_batch_size() {
        let columns = [col("id", ColumnType::Int)];
        let types = [ColumnType::Int];
        // 2.5 batches worth of rows.
        let total = crate::BATCH_SIZE * 2 + 5;
        let rows: Vec<Vec<Value>> = (0..total)
            .map(|i| vec![Value::Int(i64::try_from(i).unwrap())])
            .collect();
        let schema = Arc::new(schema_from_columns(&columns));
        let scan = RecordBatchScan::new(Box::new(scan_of(rows, &types)), schema);

        let batches: Vec<_> = scan.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].num_rows(), crate::BATCH_SIZE);
        assert_eq!(batches[1].num_rows(), crate::BATCH_SIZE);
        assert_eq!(batches[2].num_rows(), 5);
        // Row order + values are preserved across the batch boundary.
        let last = batches[2]
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(last.get(4), Some(i64::try_from(total - 1).unwrap()));
    }

    #[test]
    fn empty_scan_yields_no_batch() {
        let columns = [col("id", ColumnType::Int)];
        let schema = Arc::new(schema_from_columns(&columns));
        let scan = RecordBatchScan::new(Box::new(scan_of(vec![], &[ColumnType::Int])), schema);
        assert_eq!(scan.count(), 0);
    }

    #[test]
    fn batches_nested_list_column() {
        let columns = [col("tags", ColumnType::Array(ArrayElem::Int))];
        let types = [ColumnType::Array(ArrayElem::Int)];
        let rows = vec![
            vec![Value::Array(vec![Value::Int(1), Value::Int(2)])],
            vec![Value::Null],
            vec![Value::Array(vec![])],
        ];
        let schema = Arc::new(schema_from_columns(&columns));
        let scan = RecordBatchScan::new(Box::new(scan_of(rows, &types)), schema);

        let batches: Vec<_> = scan.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let lists = batches[0]
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(lists.len(), 3);
        assert_eq!(lists.value_len(0), Some(2));
        assert!(lists.is_null(1));
        assert_eq!(lists.value_len(2), Some(0));
    }
}
