//! [`SeqScan`]: the leaf vectorized operator — a full table scan as a
//! [`RecordBatch`] stream.
//!
//! It opens a storage [`scan`](nusadb_core::StorageEngine::scan) over the table's visible
//! tuples and feeds it through the [`RecordBatchScan`] adapter, which decodes and
//! batches rows into columns. `SeqScan` is therefore a thin [`Operator`]: it owns the
//! batch stream and yields one batch per [`Operator::next_batch`] call.

use std::sync::Arc;

use nusadb_core::engine::{TableSchema, TupleScan};
use nusadb_core::{StorageEngine, TxnId};

use crate::batch::{RecordBatch, RecordBatchScan, Schema, schema_from_columns};
use crate::error::Error;
use crate::vectorized::Operator;

/// A sequential scan of every row of a table visible to a transaction, produced as a
/// stream of [`RecordBatch`]es.
#[derive(Debug)]
pub struct SeqScan {
    schema: Arc<Schema>,
    batches: RecordBatchScan,
}

impl SeqScan {
    /// Build a scan operator from an already-open tuple `scan` and the `schema` its rows
    /// are encoded under.
    #[must_use]
    pub fn new(scan: Box<dyn TupleScan>, schema: Arc<Schema>) -> Self {
        let batches = RecordBatchScan::new(scan, Arc::clone(&schema));
        Self { schema, batches }
    }

    /// Open a sequential scan of `table` within `txn`, deriving the batch schema from the
    /// table's catalog columns.
    ///
    /// # Errors
    ///
    /// Propagates any error from opening the storage [`scan`](StorageEngine::scan).
    pub fn open(
        engine: &dyn StorageEngine,
        txn: TxnId,
        table: &TableSchema,
    ) -> Result<Self, Error> {
        let schema = Arc::new(schema_from_columns(&table.columns));
        let scan = engine.scan(txn, table.id)?;
        Ok(Self::new(scan, schema))
    }
}

impl Operator for SeqScan {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        // Cooperative cancellation parity with the row path: a statement timeout / cancel
        // request aborts the scan at a batch boundary rather than running to completion.
        crate::cancel::check()?;
        self.batches.next().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::SeqScan;
    use crate::ast::Value;
    use crate::batch::{Array, Int64Array, Schema, StringArray, schema_from_columns};
    use crate::executor::row;
    use crate::vectorized::Operator;
    use nusadb_core::engine::{ColumnDef, SharedTuple, Tid, TupleScan};
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

    fn seq_scan(rows: Vec<Vec<Value>>, columns: &[ColumnDef]) -> (SeqScan, Arc<Schema>) {
        let types: Vec<ColumnType> = columns.iter().map(|c| c.ty).collect();
        let tuples = rows
            .into_iter()
            .map(|r| SharedTuple::from(row::encode(&r, &types).unwrap().as_slice()))
            .collect();
        let schema = Arc::new(schema_from_columns(columns));
        let scan = SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), Arc::clone(&schema));
        (scan, schema)
    }

    fn drain(mut op: SeqScan) -> Vec<crate::batch::RecordBatch> {
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            out.push(batch);
        }
        out
    }

    #[test]
    fn scans_rows_into_a_single_batch() {
        let columns = [col("id", ColumnType::Int), col("name", ColumnType::Text)];
        let rows = vec![
            vec![Value::Int(10), Value::Text("a".to_owned())],
            vec![Value::Int(20), Value::Null],
        ];
        let (op, schema) = seq_scan(rows, &columns);
        assert_eq!(op.schema(), &schema);

        let batches = drain(op);
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema(), &schema);

        let ids = batch
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.get(0), Some(10));
        assert_eq!(ids.get(1), Some(20));
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
    fn empty_table_yields_no_batch() {
        let columns = [col("id", ColumnType::Int)];
        let (op, _schema) = seq_scan(vec![], &columns);
        assert!(drain(op).is_empty());
    }

    #[test]
    fn scan_spans_multiple_batches() {
        let columns = [col("id", ColumnType::Int)];
        let total = crate::BATCH_SIZE + 3;
        let rows: Vec<Vec<Value>> = (0..total)
            .map(|i| vec![Value::Int(i64::try_from(i).unwrap())])
            .collect();
        let (op, _schema) = seq_scan(rows, &columns);

        let batches = drain(op);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), crate::BATCH_SIZE);
        assert_eq!(batches[1].num_rows(), 3);
    }
}
