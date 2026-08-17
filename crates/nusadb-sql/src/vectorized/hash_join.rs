//! [`HashJoin`]: a vectorized equi-join (`INNER`/`LEFT`/`RIGHT`/`FULL`).
//!
//! The build (right) side is materialized once into a single [`RecordBatch`] plus the row path's own
//! [`JoinIndex`]; the probe (left) side is streamed a batch at a time. For each probe batch the
//! operator produces a **selection vector** — the `(probe_row, build_row)` index pairs that hash-match
//! and pass the residual — and gathers the output columns from the two sides by those indices
//! ([`take_batch`]), so a matched pair never materializes a `[left ++ right]` `Vec<Value>` the way the
//! row path ([`run_hash_join`](crate::executor::join::run_hash_join)) does. An outer join's unmatched
//! rows are NULL-padded on the absent side: an unmatched probe row gathers the build side at an
//! out-of-range index (which [`take_batch`] yields as NULL); the unmatched build rows are emitted once,
//! after the probe is exhausted, with an all-NULL probe side.
//!
//! **Correctness = the row path's, by reuse.** Key extraction, NULL-key exclusion, duplicate-key
//! fan-out and multi-key equality all come from the shared [`JoinIndex`]; the residual is the shared
//! [`BatchPredicate`], whose mask is the row evaluator's 3VL; a hash-matched pair that fails the
//! residual does not count as matched (so it can still become an outer NULL-padded row), exactly as
//! `run_hash_join`. The emitted multiset therefore equals `run_hash_join`'s on every CPU. Join order
//! is unspecified (SQL imposes none), so the emission order need not match the row path's.
//!
//! **Scope:** equi-join on `ON` keys (no `USING`/`NATURAL` column merge), build materialized in
//! memory. The planner routes every other shape (`USING`/`NATURAL`, a configured spill budget, a
//! non-vectorizable residual/child) to the mature row path — this only adds a fast path.

use std::sync::Arc;

use crate::ast;
use crate::batch::convert::{batch_to_rows, build_column, rows_to_batch, take_batch};
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

/// A vectorized equi-join over two child [`Operator`]s (`left` = probe, `right` = build).
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
    /// Join kind — decides which side's unmatched rows are kept (NULL-padded).
    kind: ast::JoinKind,
    /// Output schema: the probe fields followed by the build fields.
    schema: Arc<Schema>,
    /// Materialized build side (`None` until the first [`Operator::next_batch`]).
    build: Option<BuildSide>,
    /// Per-build-row matched flag for `RIGHT`/`FULL` (empty otherwise); a build row is matched only by
    /// a residual-passing pair. Sized when the build side materializes.
    build_matched: Vec<bool>,
    /// Set once the unmatched build-row tail has been emitted (`RIGHT`/`FULL`).
    right_tail_done: bool,
    /// Set once the probe side is exhausted and any tail emitted.
    done: bool,
}

impl HashJoin {
    /// Build an equi-join of `left` (probe) against `right` (build) on `keys`, of kind `kind`, with an
    /// optional non-equi `residual`. `left_width` is the probe side's column count (the right keys'
    /// offset). Only `Inner`/`Left`/`Right`/`Full` are valid (the kinds a `HashJoin` carries).
    #[must_use]
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        keys: Vec<HashKey>,
        residual: Option<TypedExpr>,
        left_width: usize,
        kind: ast::JoinKind,
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
            kind,
            schema,
            build: None,
            build_matched: Vec::new(),
            right_tail_done: false,
            done: false,
        }
    }

    /// Keep an unmatched probe row, NULL-padded on the build side (`LEFT`/`FULL`).
    const fn keep_unmatched_left(&self) -> bool {
        matches!(self.kind, ast::JoinKind::Left | ast::JoinKind::Full)
    }

    /// Keep the unmatched build rows, NULL-padded on the probe side (`RIGHT`/`FULL`).
    const fn keep_unmatched_right(&self) -> bool {
        matches!(self.kind, ast::JoinKind::Right | ast::JoinKind::Full)
    }

    /// Materialize the build side: pull every right batch, build the shared hash index over its rows,
    /// then keep the rows as one columnar [`RecordBatch`] for index-gather. For `RIGHT`/`FULL`, also
    /// allocate the per-build-row matched flags.
    fn materialize_build(&mut self) -> Result<(), Error> {
        let right_schema = Arc::clone(self.right.schema());
        let mut build_rows: Vec<Row> = Vec::new();
        while let Some(batch) = self.right.next_batch()? {
            build_rows.extend(batch_to_rows(&batch));
        }
        if self.keep_unmatched_right() {
            self.build_matched = vec![false; build_rows.len()];
        }
        // Build the index first (it borrows the rows), then move the rows into the columnar batch.
        let index = JoinIndex::build_right(&build_rows, &self.keys, self.left_width)?;
        let batch = rows_to_batch(&right_schema, build_rows)?;
        self.build = Some(BuildSide { index, batch });
        Ok(())
    }
}

/// Gather a `[left ++ right]` output batch: probe columns from `left_batch` at `left_idx`, build
/// columns from `build_batch` at `right_idx`. An out-of-range `right_idx` entry yields a NULL build
/// side (the outer-join NULL pad); the two index slices have equal length (the output row count).
fn gather_join_output(
    schema: &Arc<Schema>,
    left_batch: &RecordBatch,
    build_batch: &RecordBatch,
    left_idx: &[usize],
    right_idx: &[usize],
) -> Result<RecordBatch, Error> {
    let left = take_batch(left_batch, left_idx)?;
    let right = take_batch(build_batch, right_idx)?;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.len());
    columns.extend(left.columns().iter().map(Arc::clone));
    columns.extend(right.columns().iter().map(Arc::clone));
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// The hash-matched pairs that pass the residual. With no residual every hash match survives; with a
/// residual, the joined batch for the matched pairs is evaluated (the shared [`BatchPredicate`], 3VL)
/// and only the passing `(probe, build)` pairs are kept — exactly `run_hash_join`'s `residual_passes`.
fn residual_survivors(
    residual: Option<&BatchPredicate>,
    schema: &Arc<Schema>,
    left_batch: &RecordBatch,
    build_batch: &RecordBatch,
    probe_idx: Vec<usize>,
    build_idx: Vec<usize>,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let Some(pred) = residual else {
        return Ok((probe_idx, build_idx));
    };
    if probe_idx.is_empty() {
        return Ok((probe_idx, build_idx));
    }
    let joined = gather_join_output(schema, left_batch, build_batch, &probe_idx, &build_idx)?;
    let mask = pred.mask(&joined)?;
    let mut sp = Vec::with_capacity(probe_idx.len());
    let mut sb = Vec::with_capacity(build_idx.len());
    for ((keep, p), b) in mask.iter().zip(&probe_idx).zip(&build_idx) {
        if *keep {
            sp.push(*p);
            sb.push(*b);
        }
    }
    Ok((sp, sb))
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
            .field("kind", &self.kind)
            .field("built", &self.build.is_some())
            .finish_non_exhaustive()
    }
}

impl HashJoin {
    /// The unmatched build rows, NULL-padded on the probe side (`RIGHT`/`FULL`), emitted once after the
    /// probe is exhausted. `Ok(None)` when there is no tail (inner/left join, or every build row matched).
    fn unmatched_right_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.right_tail_done || !self.keep_unmatched_right() {
            return Ok(None);
        }
        self.right_tail_done = true;
        let Some(build) = self.build.as_ref() else {
            return Ok(None);
        };
        let unmatched: Vec<usize> = self
            .build_matched
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| (!m).then_some(i))
            .collect();
        if unmatched.is_empty() {
            return Ok(None);
        }
        let right = take_batch(&build.batch, &unmatched)?;
        // The probe side is entirely NULL for an unmatched build row (the first `left_width` fields).
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.schema.len());
        for field in self.schema.fields().iter().take(self.left_width) {
            columns.push(build_column(
                field.data_type(),
                vec![ast::Value::Null; unmatched.len()],
            )?);
        }
        columns.extend(right.columns().iter().map(Arc::clone));
        Ok(Some(RecordBatch::try_new(
            Arc::clone(&self.schema),
            columns,
        )?))
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
                // Probe exhausted: emit the unmatched build-row tail (RIGHT/FULL) once, then finish.
                if let Some(tail) = self.unmatched_right_batch()? {
                    return Ok(Some(tail));
                }
                self.done = true;
                return Ok(None);
            };
            let Some(build) = self.build.as_ref() else {
                return Err(Error::Internal("hash-join build side missing".to_owned()));
            };
            // Probe: for each left row, collect an index pair per hash-matched build row. A NULL key
            // (or any non-matching key) contributes no pair — the row path's unmatched treatment.
            let left_rows = batch_to_rows(&left_batch);
            let n = left_rows.len();
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
            // Keep only the pairs that pass the residual — these are the true matches; a hash match
            // that fails the residual leaves its rows eligible to become outer NULL-padded rows.
            let (surv_probe, surv_build) = residual_survivors(
                self.residual.as_ref(),
                &self.schema,
                &left_batch,
                &build.batch,
                probe_idx,
                build_idx,
            )?;
            // Mark matched rows: build rows for RIGHT/FULL (persists), probe rows for this batch.
            let mut matched_probe = vec![false; n];
            for (&p, &b) in surv_probe.iter().zip(&surv_build) {
                if let Some(flag) = matched_probe.get_mut(p) {
                    *flag = true;
                }
                if self.keep_unmatched_right()
                    && let Some(flag) = self.build_matched.get_mut(b)
                {
                    *flag = true;
                }
            }
            // Output index vectors: the matched pairs, then (LEFT/FULL) each unmatched probe row with
            // an out-of-range build index so `take_batch` NULL-pads its build side.
            let mut left_idx = surv_probe;
            let mut right_idx = surv_build;
            if self.keep_unmatched_left() {
                let null_pad = build.batch.num_rows(); // out-of-range → NULL build columns
                for (p, &matched) in matched_probe.iter().enumerate() {
                    if !matched {
                        left_idx.push(p);
                        right_idx.push(null_pad);
                    }
                }
            }
            if left_idx.is_empty() {
                continue; // no output from this probe batch — pull the next rather than emit empty
            }
            let out = gather_join_output(
                &self.schema,
                &left_batch,
                &build.batch,
                &left_idx,
                &right_idx,
            )?;
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

    /// The vectorized join's output multiset must equal `run_hash_join`'s over the same inputs, for
    /// every join kind (INNER/LEFT/RIGHT/FULL) crossed with single/multi-key, duplicate keys, NULL
    /// keys, and a residual — the residual and outer NULL-padding interacting (a hash match that fails
    /// the residual must still surface as an unmatched outer row).
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
        let cases: [(Vec<HashKey>, Option<TypedExpr>, &str); 3] = [
            (single_key, None, "single-key dup+NULL"),
            (
                vec![key(0, 2, ColumnType::Int)],
                Some(residual),
                "single-key + residual",
            ),
            (multi_key, None, "multi-key"),
        ];
        let kinds = [
            JoinKind::Inner,
            JoinKind::Left,
            JoinKind::Right,
            JoinKind::Full,
        ];

        for (keys, residual, label) in &cases {
            for kind in kinds {
                let expected =
                    run_hash_join(&l1, &r1, keys, residual.as_ref(), kind, 2, 2).unwrap();
                let op = HashJoin::new(
                    scan(&l1, &two_int),
                    scan(&r1, &two_int),
                    keys.clone(),
                    residual.clone(),
                    2,
                    kind,
                );
                let got = run_vectorized(op);
                assert_eq!(sorted(got), sorted(expected), "kind={kind:?} case={label}");
            }
        }
    }

    /// Empty-side outer joins match the row path: an empty build side keeps every LEFT probe row
    /// (NULL-padded right); an empty probe side keeps every RIGHT/FULL build row (NULL-padded left).
    #[test]
    fn outer_join_empty_side_matches_row_path() {
        let two_int = [ColumnType::Int, ColumnType::Int];
        let rows = vec![vec![int(1), int(10)], vec![int(2), int(20)]];
        let keys = vec![key(0, 2, ColumnType::Int)];
        // Empty build (right): INNER → none; LEFT → both left rows NULL-padded.
        for kind in [JoinKind::Inner, JoinKind::Left] {
            let expected = run_hash_join(&rows, &[], &keys, None, kind, 2, 2).unwrap();
            let op = HashJoin::new(
                scan(&rows, &two_int),
                scan(&[], &two_int),
                keys.clone(),
                None,
                2,
                kind,
            );
            assert_eq!(
                sorted(run_vectorized(op)),
                sorted(expected),
                "kind={kind:?}"
            );
        }
        // Empty probe (left): RIGHT/FULL → every build row NULL-padded on the left.
        for kind in [JoinKind::Right, JoinKind::Full] {
            let expected = run_hash_join(&[], &rows, &keys, None, kind, 2, 2).unwrap();
            let op = HashJoin::new(
                scan(&[], &two_int),
                scan(&rows, &two_int),
                keys.clone(),
                None,
                2,
                kind,
            );
            assert_eq!(
                sorted(run_vectorized(op)),
                sorted(expected),
                "kind={kind:?}"
            );
        }
    }

    /// A probe side spanning multiple batches joins every batch against the one build side, for INNER
    /// and LEFT (the LEFT unmatched rows spread across batches).
    #[test]
    fn probe_spans_multiple_batches() {
        let two_int = [ColumnType::Int, ColumnType::Int];
        let total = crate::BATCH_SIZE + 5;
        let left: Vec<Vec<ast::Value>> = (0..total)
            .map(|i| {
                vec![
                    int(i64::try_from(i).unwrap() % 5), // keys 0..4; only 0,1 match → 2..4 unmatched
                    int(i64::try_from(i).unwrap()),
                ]
            })
            .collect();
        let right = vec![vec![int(0), int(1000)], vec![int(1), int(1001)]];
        let keys = vec![key(0, 2, ColumnType::Int)];
        for kind in [JoinKind::Inner, JoinKind::Left] {
            let expected = run_hash_join(&left, &right, &keys, None, kind, 2, 2).unwrap();
            let op = HashJoin::new(
                scan(&left, &two_int),
                scan(&right, &two_int),
                keys.clone(),
                None,
                2,
                kind,
            );
            assert_eq!(
                sorted(run_vectorized(op)),
                sorted(expected),
                "kind={kind:?}"
            );
        }
    }
}
