//! [`HashJoin`]: a vectorized INNER equi-join.
//!
//! The build (right) side is materialized once into a single [`RecordBatch`] plus the row path's own
//! [`JoinIndex`]; the probe (left) side is streamed a batch at a time. For each probe batch the
//! operator produces a **selection vector** — the `(probe_row, build_row)` index pairs that hash-match
//! — and gathers the output columns from the two sides by those indices ([`take_batch`]), so a matched
//! pair never materializes a `[left ++ right]` `Vec<Value>` the way the row path
//! ([`run_hash_join`](crate::executor::join::run_hash_join)) does. The optional non-equi residual is
//! then applied as a columnar predicate over the gathered batch.
//!
//! **Correctness = the row path's, by reuse.** Key extraction, NULL-key exclusion, duplicate-key
//! fan-out and multi-key equality all come from the shared [`JoinIndex`]; the residual is the shared
//! [`BatchPredicate`], whose mask is the row evaluator's 3VL. The emitted multiset therefore equals
//! `run_hash_join`'s for `Inner` on every CPU. Join order is unspecified (SQL imposes none), so the
//! emission order — probe-batch order, then build-match order — need not match the row path's.
//!
//! **Scope:** INNER equi-join, `ON` keys (no `USING`/`NATURAL` column merge), build materialized in
//! memory. The planner routes every other shape (outer joins, `USING`/`NATURAL`, a configured
//! spill budget, non-vectorizable residual/child) to the mature row path — this only adds a fast path.

use std::sync::Arc;

use crate::batch::convert::{batch_to_rows, filter_batch, rows_to_batch, take_batch};
use crate::batch::{ArrayRef, Field, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::join::JoinIndex;
use crate::executor::row::Row;
use crate::planner::{HashKey, TypedExpr};
use crate::vectorized::Operator;
use crate::vectorized::filter::BatchPredicate;

/// The materialized build side: the row path's hash index plus the build rows as one columnar batch
/// (gathered by the index's build-row ordinals).
struct BuildSide {
    index: JoinIndex,
    batch: RecordBatch,
}

/// A vectorized INNER equi-join over two child [`Operator`]s (`left` = probe, `right` = build).
pub struct HashJoin {
    /// Probe (left) input, streamed a batch at a time.
    left: Box<dyn Operator>,
    /// Build (right) input, fully materialized on first pull.
    right: Box<dyn Operator>,
    /// Equi-join keys (left references left ordinals `< left_width`, right references joined ordinals
    /// `>= left_width` — the [`JoinIndex`] contract).
    keys: Vec<HashKey>,
    /// Optional non-equi residual over the joined `[left ++ right]` row, applied to the gathered batch.
    residual: Option<BatchPredicate>,
    /// Number of columns the probe side produces (the right keys' ordinal offset).
    left_width: usize,
    /// Output schema: the probe fields followed by the build fields.
    schema: Arc<Schema>,
    /// Materialized build side (`None` until the first [`Operator::next_batch`]).
    build: Option<BuildSide>,
    /// Set once the probe side is exhausted.
    done: bool,
}

impl HashJoin {
    /// Build an INNER equi-join of `left` (probe) against `right` (build) on `keys`, with an optional
    /// non-equi `residual`. `left_width` is the probe side's column count (the right keys' offset).
    #[must_use]
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        keys: Vec<HashKey>,
        residual: Option<TypedExpr>,
        left_width: usize,
    ) -> Self {
        let mut fields: Vec<Field> = Vec::with_capacity(left.schema().len() + right.schema().len());
        fields.extend(left.schema().fields().iter().cloned());
        fields.extend(right.schema().fields().iter().cloned());
        let schema = Arc::new(Schema::new(fields));
        Self {
            left,
            right,
            keys,
            residual: residual.map(BatchPredicate::new),
            left_width,
            schema,
            build: None,
            done: false,
        }
    }

    /// Materialize the build side: pull every right batch, build the shared hash index over its rows,
    /// then keep the rows as one columnar [`RecordBatch`] for index-gather.
    fn materialize_build(&mut self) -> Result<(), Error> {
        let right_schema = Arc::clone(self.right.schema());
        let mut build_rows: Vec<Row> = Vec::new();
        while let Some(batch) = self.right.next_batch()? {
            build_rows.extend(batch_to_rows(&batch));
        }
        // Build the index first (it borrows the rows), then move the rows into the columnar batch.
        let index = JoinIndex::build_right(&build_rows, &self.keys, self.left_width)?;
        let batch = rows_to_batch(&right_schema, build_rows)?;
        self.build = Some(BuildSide { index, batch });
        Ok(())
    }
}

impl std::fmt::Debug for HashJoin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The hash index and materialized build batch are large internal state, not identity —
        // `finish_non_exhaustive` marks them intentionally omitted.
        f.debug_struct("HashJoin")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("keys", &self.keys.len())
            .field("has_residual", &self.residual.is_some())
            .field("left_width", &self.left_width)
            .field("built", &self.build.is_some())
            .finish_non_exhaustive()
    }
}

impl Operator for HashJoin {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.done {
            return Ok(None);
        }
        if self.build.is_none() {
            self.materialize_build()?;
        }
        loop {
            // Cooperative cancellation parity with the row path: abort at a probe-batch boundary.
            crate::cancel::check()?;
            let Some(left_batch) = self.left.next_batch()? else {
                self.done = true;
                return Ok(None);
            };
            let Some(build) = self.build.as_ref() else {
                return Err(Error::Unsupported(
                    "internal: hash-join build side missing".to_owned(),
                ));
            };
            // Probe: for each left row, emit an index pair per hash-matched build row. A NULL key (or
            // any non-matching key) contributes no pair — exactly the row path's unmatched treatment.
            let left_rows = batch_to_rows(&left_batch);
            let mut probe_idx: Vec<usize> = Vec::new();
            let mut build_idx: Vec<usize> = Vec::new();
            for (p, left_row) in left_rows.iter().enumerate() {
                if let Some(indices) = build.index.probe_left(&self.keys, left_row)? {
                    for &b in indices {
                        probe_idx.push(p);
                        build_idx.push(b);
                    }
                }
            }
            if probe_idx.is_empty() {
                continue; // no match in this probe batch — pull the next rather than emit empty
            }
            // Gather the output columns by selection vector: probe columns from this batch, build
            // columns from the materialized build batch — no per-pair `[left ++ right]` allocation.
            let left_gathered = take_batch(&left_batch, &probe_idx)?;
            let right_gathered = take_batch(&build.batch, &build_idx)?;
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.len());
            columns.extend(left_gathered.columns().iter().map(Arc::clone));
            columns.extend(right_gathered.columns().iter().map(Arc::clone));
            let out = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
            // Residual (non-equi ON conjuncts): keep only rows the shared predicate passes — the row
            // evaluator's 3VL, identical to `residual_passes` on the joined row.
            let out = match &self.residual {
                Some(pred) => {
                    let mask = pred.mask(&out)?;
                    filter_batch(&out, &mask)?
                },
                None => out,
            };
            if out.num_rows() == 0 {
                continue; // residual dropped every matched pair — pull the next probe batch
            }
            return Ok(Some(out));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HashJoin;
    use crate::Field;
    use crate::ast::{self, JoinKind};
    use crate::batch::Schema;
    use crate::executor::join::run_hash_join;
    use crate::executor::row::{self, Row};
    use crate::planner::{HashKey, TypedExpr, TypedExprKind};
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

    /// A vectorized `SeqScan` over `rows` encoded under `types`.
    fn scan(rows: &[Vec<ast::Value>], types: &[ColumnType]) -> Box<dyn Operator> {
        let tuples = rows
            .iter()
            .map(|r| SharedTuple::from(row::encode(r, types).unwrap().as_slice()))
            .collect();
        let fields = types
            .iter()
            .enumerate()
            .map(|(i, &ty)| Field::new(format!("c{i}"), ty, true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        Box::new(SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema))
    }

    fn col(i: usize, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Column(i),
            ty,
        }
    }

    /// `left.col(l) = right.col(r)` — the right key uses the joined ordinal `left_width + r`.
    fn key(l: usize, r_joined: usize, ty: ColumnType) -> HashKey {
        HashKey {
            left: col(l, ty),
            right: col(r_joined, ty),
        }
    }

    fn int(v: i64) -> ast::Value {
        ast::Value::Int(v)
    }

    fn run_vectorized(mut op: HashJoin) -> Vec<Row> {
        let mut rows = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            rows.extend(crate::batch::convert::batch_to_rows(&batch));
        }
        rows
    }

    /// Sort a multiset of rows by their `Debug` spelling so two join outputs compare regardless of
    /// emission order (SQL imposes none on a join).
    fn sorted(rows: Vec<Row>) -> Vec<String> {
        let mut keys: Vec<String> = rows.into_iter().map(|r| format!("{r:?}")).collect();
        keys.sort();
        keys
    }

    /// The vectorized join's output multiset must equal `run_hash_join`'s (`Inner`) over the same
    /// inputs — single/multi-key, duplicate keys, NULL keys, an empty build side, and a residual.
    #[test]
    fn matches_run_hash_join_over_many_shapes() {
        let two_int = [ColumnType::Int, ColumnType::Int];
        // (left rows, right rows, keys, residual, left_width, right_width, label)
        let l1 = vec![
            vec![int(1), int(10)],
            vec![int(2), int(20)],
            vec![int(2), int(21)],           // duplicate left key 2
            vec![ast::Value::Null, int(30)], // NULL key — matches nothing
            vec![int(5), int(50)],
        ];
        let r1 = vec![
            vec![int(2), int(200)],
            vec![int(2), int(201)], // duplicate right key 2 → fan-out
            vec![int(1), int(100)],
            vec![ast::Value::Null, int(300)],
            vec![int(9), int(900)],
        ];
        let single_key = vec![key(0, 2, ColumnType::Int)];
        // Multi-key: left (c0,c1) = right (c0,c1) → joined ordinals (2,3).
        let multi_key = vec![key(0, 2, ColumnType::Int), key(1, 3, ColumnType::Int)];
        // Residual: left.c1 < right.c1 (joined ordinals 1 < 3).
        let residual = TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(col(1, ColumnType::Int)),
                op: ast::BinaryOp::Lt,
                right: Box::new(col(3, ColumnType::Int)),
            },
            ty: ColumnType::Bool,
        };

        // Every case joins `l1` against `r1`; only the keys and residual vary.
        let cases = [
            (single_key, None, "single-key dup+NULL"),
            (
                vec![key(0, 2, ColumnType::Int)],
                Some(residual),
                "single-key + residual",
            ),
            (multi_key, None, "multi-key"),
        ];

        for (keys, residual, label) in cases {
            let expected =
                run_hash_join(&l1, &r1, &keys, residual.as_ref(), JoinKind::Inner, 2, 2).unwrap();
            let op = HashJoin::new(scan(&l1, &two_int), scan(&r1, &two_int), keys, residual, 2);
            let got = run_vectorized(op);
            assert_eq!(sorted(got), sorted(expected), "case: {label}");
        }
    }

    /// An empty build side yields no INNER rows.
    #[test]
    fn empty_build_yields_no_rows() {
        let two_int = [ColumnType::Int, ColumnType::Int];
        let left = vec![vec![int(1), int(10)], vec![int(2), int(20)]];
        let op = HashJoin::new(
            scan(&left, &two_int),
            scan(&[], &two_int),
            vec![key(0, 2, ColumnType::Int)],
            None,
            2,
        );
        assert!(run_vectorized(op).is_empty());
    }

    /// A probe side spanning multiple batches joins every batch against the one build side.
    #[test]
    fn probe_spans_multiple_batches() {
        let two_int = [ColumnType::Int, ColumnType::Int];
        let total = crate::BATCH_SIZE + 5;
        let left: Vec<Vec<ast::Value>> = (0..total)
            .map(|i| {
                vec![
                    int(i64::try_from(i).unwrap() % 3),
                    int(i64::try_from(i).unwrap()),
                ]
            })
            .collect();
        let right = vec![vec![int(0), int(1000)], vec![int(1), int(1001)]];
        let keys = vec![key(0, 2, ColumnType::Int)];
        let expected = run_hash_join(&left, &right, &keys, None, JoinKind::Inner, 2, 2).unwrap();
        let op = HashJoin::new(scan(&left, &two_int), scan(&right, &two_int), keys, None, 2);
        assert_eq!(sorted(run_vectorized(op)), sorted(expected));
    }
}
