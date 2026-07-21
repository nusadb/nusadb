//! [`Limit`]: skip a prefix of rows (`OFFSET`) then pass through at most a fixed
//! number (`LIMIT`) across its child's batch stream.
//!
//! It tracks a running offset and remaining-row budget rather than buffering: whole batches
//! that fall entirely inside the skipped prefix are dropped, a batch fully inside the kept
//! window is forwarded untouched (cheap [`Arc`] share — no re-materialization), and only a
//! batch straddling an offset/limit boundary is sliced columnarly (via
//! [`take_batch`](crate::batch::convert::take_batch), no row round-trip). Once the `LIMIT` budget
//! is exhausted the operator reports end-of-stream without pulling further batches from its child.
//! The output schema equals the input schema.

use std::sync::Arc;

use crate::batch::convert::take_batch;
use crate::batch::{RecordBatch, Schema};
use crate::error::Error;
use crate::vectorized::Operator;

/// An `OFFSET` + `LIMIT` slice over a child [`Operator`]'s batch stream.
#[derive(Debug)]
pub struct Limit {
    child: Box<dyn Operator>,
    schema: Arc<Schema>,
    /// Rows still to skip before any are emitted.
    remaining_offset: usize,
    /// Rows still allowed to be emitted; `None` means unbounded (offset-only).
    budget: Option<usize>,
    /// Set once the budget is spent or the child is drained.
    done: bool,
}

impl Limit {
    /// Build a limit/offset operator over `child`: skip the first `offset` rows, then emit
    /// at most `limit` rows (`None` = no upper bound, i.e. `OFFSET` with no `LIMIT`).
    #[must_use]
    pub fn new(child: Box<dyn Operator>, offset: usize, limit: Option<usize>) -> Self {
        let schema = Arc::clone(child.schema());
        Self {
            child,
            schema,
            remaining_offset: offset,
            budget: limit,
            done: false,
        }
    }
}

impl Operator for Limit {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.done || self.budget == Some(0) {
            self.done = true;
            return Ok(None);
        }
        loop {
            let Some(batch) = self.child.next_batch()? else {
                self.done = true;
                return Ok(None);
            };
            let rows = batch.num_rows();
            // Skip whole batches that lie entirely within the OFFSET prefix.
            if self.remaining_offset >= rows {
                self.remaining_offset -= rows;
                continue;
            }
            let start = self.remaining_offset;
            self.remaining_offset = 0;
            let available = rows - start;
            let take = self.budget.map_or(available, |l| l.min(available));
            // Charge the emitted rows against the budget; end the stream once it hits 0.
            if let Some(remaining) = self.budget.as_mut() {
                *remaining -= take;
                if *remaining == 0 {
                    self.done = true;
                }
            }
            // Fast path: the whole batch is inside the kept window — forward it untouched.
            if start == 0 && take == rows {
                return Ok(Some(batch));
            }
            let window: Vec<usize> = (start..start + take).collect();
            return take_batch(&batch, &window).map(Some);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Limit;
    use crate::Field;
    use crate::ast;
    use crate::batch::{Array, Int64Array, Schema};
    use crate::executor::row;
    use crate::vectorized::{Operator, SeqScan};
    use nusadb_core::engine::{SharedTuple, Tid, TupleScan};
    use nusadb_core::{ColumnType, PageId, Result as CoreResult, SlotIdx};
    use std::sync::Arc;

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

    /// A `SeqScan` over `id INT` rows `0..n`.
    fn id_scan(n: usize) -> SeqScan {
        let types = [ColumnType::Int];
        let tuples = (0..n)
            .map(|i| {
                let row = [ast::Value::Int(i64::try_from(i).unwrap())];
                SharedTuple::from(row::encode(&row, &types).unwrap().as_slice())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("id", ColumnType::Int, true)]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    fn ids(op: Limit) -> Vec<i64> {
        let mut op = op;
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            let col = batch
                .column(0)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..col.len() {
                out.push(col.get(i).unwrap());
            }
        }
        out
    }

    #[test]
    fn limit_only_takes_prefix() {
        assert_eq!(
            ids(Limit::new(Box::new(id_scan(10)), 0, Some(3))),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn offset_and_limit_slice_the_middle() {
        assert_eq!(
            ids(Limit::new(Box::new(id_scan(10)), 2, Some(3))),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn offset_only_drops_prefix() {
        assert_eq!(
            ids(Limit::new(Box::new(id_scan(8)), 5, None)),
            vec![5, 6, 7]
        );
    }

    #[test]
    fn zero_limit_is_empty() {
        assert!(ids(Limit::new(Box::new(id_scan(5)), 0, Some(0))).is_empty());
    }

    #[test]
    fn offset_past_end_is_empty() {
        assert!(ids(Limit::new(Box::new(id_scan(3)), 10, Some(5))).is_empty());
    }

    #[test]
    fn limit_caps_when_fewer_rows_available() {
        // Want 100 but only 4 exist → all 4.
        assert_eq!(
            ids(Limit::new(Box::new(id_scan(4)), 0, Some(100))),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn limit_spans_batch_boundary() {
        // OFFSET 0, LIMIT BATCH_SIZE + 2 over BATCH_SIZE + 5 rows → first BATCH_SIZE + 2.
        let total = crate::BATCH_SIZE + 5;
        let want = crate::BATCH_SIZE + 2;
        let got = ids(Limit::new(Box::new(id_scan(total)), 0, Some(want)));
        assert_eq!(got.len(), want);
        assert_eq!(got.first(), Some(&0));
        assert_eq!(got.last(), Some(&i64::try_from(want - 1).unwrap()));
    }

    #[test]
    fn offset_spans_batch_boundary() {
        // Skip past the first batch, then take 2 from the tail.
        let total = crate::BATCH_SIZE + 5;
        let offset = crate::BATCH_SIZE + 1;
        let got = ids(Limit::new(Box::new(id_scan(total)), offset, Some(2)));
        assert_eq!(
            got,
            vec![
                i64::try_from(offset).unwrap(),
                i64::try_from(offset + 1).unwrap()
            ]
        );
    }
}
