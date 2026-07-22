//! [`ParallelGroupedAggregate`] (morsel-parallel scan): hash `GROUP BY` over a bare table
//! scan, folded by worker threads — the first parallel operator, attacking the grouping/hash
//! residual QA measured once the scan itself went single-copy.
//!
//! The caller's thread is the **producer**: it walks the engine's single [`TupleScan`] cursor
//! (the treaty stays untouched — the design's "no treaty change" constraint) and deals fixed-size
//! chunks of [`SharedTuple`]s round-robin to N workers over bounded channels. Each **worker**
//! decodes its chunks through the same typed-builder batch decode the sequential path uses
//! ([`RecordBatchScan`]) and folds them into its own [`GroupIndex`] partial — the same
//! hash/equality contract and [`fold_value`] machinery as every other group-by path. The
//! producer then **merges** the partials ([`merge_acc`]) and finalizes.
//!
//! Determinism: the output must be bit-identical to the sequential fold, including the
//! first-seen group emission order. Every row has a global position (chunks are numbered, rows
//! within a chunk are dense), each worker records the position at which it first saw each
//! group, the merge takes the minimum across workers, and the merged groups are sorted by that
//! position — exactly the sequential first-seen order. The gates guarantee the values
//! themselves cannot leak merge order: every call passes
//! [`call_is_parallel_mergeable`] (associative, order-free folds only) and every group-key
//! column is [`parallel_safe_ty`] (compare-equal ⇒ byte-identical, so a key's stored spelling
//! cannot differ between workers).
//!
//! Shape gate (v1): `GroupAggregate` directly over a `SeqScan` (pushdown-narrowed or not) —
//! no intervening `Filter`, so workers evaluate **no expressions at all** (bare-column keys
//! and arguments only, enforced the same way [`GroupedAggregate`](super::GroupedAggregate)
//! does). The parallel fold engages only when the plan-time scanned-row estimate clears
//! [`PARALLEL_MIN_EST_ROWS`] (an un-`ANALYZE`d table stays sequential, mirroring the
//! vectorized routing) and at least two workers are available. Under a configured spill
//! budget it additionally demands ANALYZE statistics bounding the group count at
//! [`SPILL_GROUPS_CAP`] — the hash state then provably stays tiny — with a loud runtime valve
//! ([`RUNTIME_GROUP_CAP`]) against stale statistics; everything else keeps the bounded-memory
//! sort-based group-by.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::mpsc;

use nusadb_core::engine::{SharedTuple, TableSchema, Tid, TupleScan};
use nusadb_core::{PageId, SlotIdx, StorageEngine, TxnId};

use crate::Field;
use crate::ast;
use crate::batch::convert::{rows_to_batch, value_at};
use crate::batch::{RecordBatch, RecordBatchScan, Schema, schema_from_columns};
use crate::error::Error;
use crate::executor::agg::{
    Acc, GroupIndex, call_is_parallel_mergeable, finalize_aggregate, fold_count_star, fold_value,
    merge_acc, parallel_safe_ty,
};
use crate::executor::row::Row;
use crate::planner::{AggregateCall, TypedExpr, TypedExprKind};
use crate::vectorized::Operator;
use crate::vectorized::aggregate::{ColumnarShape, columnar_call_shape};
use crate::vectorized::filter::BatchPredicate;

/// Rows per chunk dealt to a worker — one decode batch.
const CHUNK_ROWS: usize = crate::BATCH_SIZE;

/// In-flight chunks per worker before the producer blocks (the backpressure bound: peak
/// undecoded state is `workers × (CHANNEL_DEPTH + 1)` chunks of `Arc` tuple handles).
const CHANNEL_DEPTH: usize = 2;

/// Plan-time scanned-row estimate below which the parallel fold is not worth its thread setup
/// and merge; the sequential vectorized fold keeps small inputs.
const PARALLEL_MIN_EST_ROWS: u64 = 100_000;

/// The worker-thread cap. The producer (caller) thread walks the scan cursor; workers decode
/// and fold. Conservative until the morsel scheduler generalizes past aggregation.
const MAX_WORKERS: usize = 4;

/// Under a configured spill budget, the parallel hash fold engages only when the ANALYZE
/// statistics bound the planned group count at or below this — the per-worker hash state then
/// provably stays far under any realistic budget. Above it (or with no statistics at all) the
/// bounded-memory sort-based group-by stays authoritative.
const SPILL_GROUPS_CAP: u64 = 4096;

/// Runtime valve for the spill-bounded engage: statistics can be stale, so a worker whose
/// group count blows this far (32×) past the planned cap aborts loudly with an `ANALYZE` hint
/// instead of accumulating unbounded hash state under a memory budget.
const RUNTIME_GROUP_CAP: usize = 131_072;

thread_local! {
    /// Test/bench override: `Some(true)` forces the parallel fold regardless of the estimate,
    /// `Some(false)` disables it, `None` (default) applies the estimate gate.
    static FORCE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };

    /// How many parallel folds actually ran on this thread — pins and probes assert on this
    /// so a silently-refused gate can never make an equivalence test vacuous.
    static FOLDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many parallel grouped folds have run on this thread (see [`parallel_scope`]).
#[must_use]
pub fn fold_count() -> u64 {
    FOLDS.with(std::cell::Cell::get)
}

/// Force the parallel grouped fold on or off for the returned guard's lifetime.
///
/// The counterpart of [`super::scope`] for the estimate gate — measurement and pin tests
/// compare the two strategies on inputs below the production threshold.
#[must_use]
pub fn parallel_scope(enabled: bool) -> ParallelGuard {
    let previous = FORCE.with(|c| c.replace(Some(enabled)));
    ParallelGuard { previous }
}

/// Restores the previous [`parallel_scope`] override on drop.
#[derive(Debug)]
pub struct ParallelGuard {
    previous: Option<bool>,
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        FORCE.with(|c| c.set(self.previous));
    }
}

/// Worker threads to spawn: every available core beyond the producer's, capped. Below 2 the
/// parallel fold is pointless and the builder refuses.
fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .saturating_sub(1)
        .min(MAX_WORKERS)
}

/// Whether `expr` can be evaluated on a worker thread: every thread-local the evaluator
/// consults lives behind the scalar-function family (`now()`, `random()`, `CURRENT_USER`,
/// session settings), subqueries (`OUTER_ROWS`), or a session-dependent `CAST` — so a strict
/// whitelist of pure value shapes over columns and literals is safe, and everything else
/// keeps the sequential path. Conservative by construction: growing it means re-proving
/// thread purity, not just vectorizability.
fn thread_pure_expr(expr: &TypedExpr) -> bool {
    use TypedExprKind as K;
    match &expr.kind {
        K::Column(_) | K::Literal(_) => true,
        K::Binary { left, right, .. } | K::IsDistinctFrom { left, right, .. } => {
            thread_pure_expr(left) && thread_pure_expr(right)
        },
        K::Unary { expr: inner, .. }
        | K::IsNull { expr: inner, .. }
        | K::IsBool { expr: inner, .. } => thread_pure_expr(inner),
        K::Between {
            expr, low, high, ..
        } => thread_pure_expr(expr) && thread_pure_expr(low) && thread_pure_expr(high),
        K::InList { expr, list, .. } => thread_pure_expr(expr) && list.iter().all(thread_pure_expr),
        K::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(thread_pure_expr)
                && branches
                    .iter()
                    .all(|b| thread_pure_expr(&b.when) && thread_pure_expr(&b.then))
                && default.as_deref().is_none_or(thread_pure_expr)
        },
        K::Coalesce(args) => args.iter().all(thread_pure_expr),
        _ => false,
    }
}

/// One worker's result: its groups in local first-seen order, each with the **global** row
/// position at which this worker first saw it.
struct WorkerPartial {
    states: Vec<(Vec<ast::Value>, Vec<Acc>)>,
    firsts: Vec<u64>,
}

/// Whether the ANALYZE statistics bound the grouped-by columns' combined distinct count at or
/// below [`SPILL_GROUPS_CAP`]. `key_indices` reference the (possibly narrowed) scan layout;
/// `scan_columns` maps them back to source columns. Missing statistics — for the table or any
/// key column — refuse (`false`): under a spill budget, no proof means no parallel hash state.
fn stats_bound_groups(
    engine: &dyn StorageEngine,
    table: &TableSchema,
    scan_columns: &[usize],
    key_indices: &[usize],
) -> Result<bool, Error> {
    let Some(stats) = engine.table_stats(table.id)? else {
        return Ok(false);
    };
    let mut est: u64 = 1;
    for &key in key_indices {
        let source = if scan_columns.is_empty() {
            key
        } else {
            match scan_columns.get(key) {
                Some(&s) => s,
                None => return Ok(false),
            }
        };
        let Some(column) = table.columns.get(source) else {
            return Ok(false);
        };
        let Some(cs) = stats.columns.iter().find(|c| c.column == column.name) else {
            return Ok(false);
        };
        // NDV counts non-NULL distinct values; a NULL group (if any NULLs exist) adds one.
        let groups = cs
            .distinct_count
            .max(1)
            .saturating_add(u64::from(cs.null_count > 0));
        est = est.saturating_mul(groups);
        if est > SPILL_GROUPS_CAP {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A [`TupleScan`] over one received chunk, feeding the worker's [`RecordBatchScan`] decode.
/// The synthetic `Tid` is never read (batch decode keys nothing on it).
struct ChunkScan {
    tuples: std::vec::IntoIter<SharedTuple>,
}

impl TupleScan for ChunkScan {
    fn try_next(&mut self) -> nusadb_core::Result<Option<(Tid, SharedTuple)>> {
        Ok(self.tuples.next().map(|t| {
            (
                Tid {
                    page: PageId(0),
                    slot: SlotIdx(0),
                },
                t,
            )
        }))
    }
}

/// The parallel hash `GROUP BY` over a table scan.
///
/// Built by its builder when the shape and estimate gates pass; folds on the first
/// [`Operator::next_batch`] pull and drains the finalized rows in batch-size chunks, exactly
/// like the sequential [`GroupedAggregate`](super::GroupedAggregate).
pub struct ParallelGroupedAggregate {
    /// The engine cursor, consumed by the fold on first pull.
    scan: Option<Box<dyn TupleScan>>,
    /// The scanned columns' batch schema (narrowed when the scan is projected) — drives the
    /// workers' decode.
    input_schema: Arc<Schema>,
    /// The full source tuple's column types when projected (empty for an unprojected scan).
    source_types: Arc<[nusadb_core::ColumnType]>,
    /// The kept source ordinals when projected, empty otherwise.
    keep: Arc<[usize]>,
    /// The batch column each group key reads (bare columns by construction).
    key_indices: Vec<usize>,
    calls: Vec<AggregateCall>,
    /// How each call folds: `COUNT(*)` or a single-value fold over its argument column.
    arg_shapes: Vec<ColumnarShape>,
    workers: usize,
    /// Whether this fold engaged under a spill budget (workers then enforce the runtime valve).
    spill_bounded: bool,
    /// The pushed-down `WHERE` predicate, evaluated on the workers (thread-pure by the gate),
    /// or `None` for a bare scan.
    predicate: Option<Arc<BatchPredicate>>,
    /// A scalar (no `GROUP BY`) aggregate: one output row, produced even over an empty input.
    scalar: bool,
    schema: Arc<Schema>,
    /// Finalized output rows, produced on the first pull; drained in `BATCH_SIZE` chunks.
    out: Option<std::vec::IntoIter<Row>>,
}

impl std::fmt::Debug for ParallelGroupedAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelGroupedAggregate")
            .field("key_indices", &self.key_indices)
            .field("calls", &self.calls.len())
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

impl ParallelGroupedAggregate {
    /// Build the parallel fold over `table`, or `Ok(None)` when any gate refuses — the caller
    /// then builds the sequential operator: estimate below [`PARALLEL_MIN_EST_ROWS`] (or absent,
    /// i.e. never `ANALYZE`d) unless forced, fewer than two workers, a computed or
    /// non-[`parallel_safe_ty`] group key, or a call outside
    /// [`call_is_parallel_mergeable`] / [`columnar_call_shape`].
    ///
    /// # Errors
    /// Propagates an engine error from opening the scan.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the aggregate plan shapes' own field set"
    )]
    pub(super) fn try_new(
        engine: &dyn StorageEngine,
        txn: TxnId,
        table: &TableSchema,
        scan_columns: &[usize],
        predicate: Option<&TypedExpr>,
        group_keys: &[TypedExpr],
        calls: Vec<AggregateCall>,
        scalar: bool,
        est_scan_rows: Option<u64>,
    ) -> Result<Option<Self>, Error> {
        let force = FORCE.with(std::cell::Cell::get);
        let allowed =
            force.unwrap_or_else(|| est_scan_rows.is_some_and(|est| est >= PARALLEL_MIN_EST_ROWS));
        if !allowed {
            return Ok(None);
        }
        // A forced run always gets two workers even on a starved host, so the pin tests
        // exercise the real dealing/merge machinery everywhere; production never forces.
        let workers = match force {
            Some(true) => worker_count().max(2),
            _ => worker_count(),
        };
        if workers < 2 {
            return Ok(None);
        }
        // A pushed-down WHERE runs on the workers, so it must be thread-pure (see
        // `thread_pure_expr`); anything else keeps the sequential path.
        if predicate.is_some_and(|p| !thread_pure_expr(p)) {
            return Ok(None);
        }
        let Some(key_indices) = group_keys
            .iter()
            .map(|k| match k.kind {
                TypedExprKind::Column(idx) if parallel_safe_ty(k.ty) => Some(idx),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        if !calls.iter().all(call_is_parallel_mergeable) {
            return Ok(None);
        }
        // Under a spill budget the sort-based group-by owns the unbounded-groups case; engage
        // only when the statistics bound the group count (see `SPILL_GROUPS_CAP`).
        let spill_bounded = crate::executor::spill_is_configured();
        if spill_bounded && !stats_bound_groups(engine, table, scan_columns, &key_indices)? {
            return Ok(None);
        }
        let Some(arg_shapes) = calls
            .iter()
            .map(columnar_call_shape)
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        // Output layout mirrors the sequential grouped aggregate: keys, then one value per call.
        let fields = group_keys
            .iter()
            .enumerate()
            .map(|(i, k)| Field::new(format!("k{i}"), k.ty, true))
            .chain(
                calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Field::new(format!("agg{i}"), c.result_ty, true)),
            )
            .collect();
        // A pushdown-narrowed scan yields the kept columns only; the planner's key and
        // argument ordinals already reference that narrowed layout. The workers decode through
        // the projected batch scan, whose narrowed schema is exactly the kept columns in order.
        let (input_schema, source_types): (_, Arc<[nusadb_core::ColumnType]>) =
            if scan_columns.is_empty() {
                (Arc::new(schema_from_columns(&table.columns)), Arc::from([]))
            } else {
                let kept: Vec<_> = scan_columns
                    .iter()
                    .filter_map(|&c| table.columns.get(c).cloned())
                    .collect();
                if kept.len() != scan_columns.len() {
                    return Ok(None);
                }
                (
                    Arc::new(schema_from_columns(&kept)),
                    table.columns.iter().map(|c| c.ty).collect(),
                )
            };
        let scan = engine.scan(txn, table.id)?;
        Ok(Some(Self {
            scan: Some(scan),
            input_schema,
            source_types,
            keep: Arc::from(scan_columns.to_vec()),
            key_indices,
            calls,
            arg_shapes,
            workers,
            spill_bounded,
            predicate: predicate.map(|p| Arc::new(BatchPredicate::new(p.clone()))),
            scalar,
            schema: Arc::new(Schema::new(fields)),
            out: None,
        }))
    }

    /// Run the parallel fold: deal the scan into chunks, fold partials on the workers, merge,
    /// and finalize in global first-seen order.
    fn fold(&mut self) -> Result<Vec<Row>, Error> {
        FOLDS.with(|c| c.set(c.get() + 1));
        // Observability for perf verification (RUST_LOG=debug): which aggregates ran parallel.
        tracing::debug!(
            workers = self.workers,
            keys = self.key_indices.len(),
            "grouped aggregate: parallel fold"
        );
        let mut scan = self.scan.take().ok_or_else(|| {
            Error::Unsupported("internal: parallel aggregate folded twice".to_owned())
        })?;
        let workers = self.workers;
        let mut partials: Vec<WorkerPartial> = Vec::with_capacity(workers);
        let (input_schema, source_types, keep, key_indices, calls, arg_shapes) = (
            &self.input_schema,
            &self.source_types,
            &self.keep,
            &self.key_indices,
            &self.calls,
            &self.arg_shapes,
        );
        let group_cap = self.spill_bounded.then_some(RUNTIME_GROUP_CAP);
        let predicate = self.predicate.as_ref();
        std::thread::scope(|scope| -> Result<(), Error> {
            let mut senders = Vec::with_capacity(workers);
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let (tx, rx) = mpsc::sync_channel::<(u64, Vec<SharedTuple>)>(CHANNEL_DEPTH);
                senders.push(tx);
                handles.push(scope.spawn(move || {
                    worker_fold(
                        &rx,
                        input_schema,
                        source_types,
                        keep,
                        predicate,
                        key_indices,
                        calls,
                        arg_shapes,
                        group_cap,
                    )
                }));
            }
            // Producer: walk the single engine cursor on this thread (cursor and cancellation
            // are both thread-local by design) and deal chunks round-robin.
            let mut seq: u64 = 0;
            let mut scan_result = Ok(());
            loop {
                // Cooperative cancellation at chunk granularity — parity with `SeqScan`'s
                // batch-boundary check; a cancel drops the senders, and the workers
                // drain out on the closed channels.
                if let Err(e) = crate::cancel::check() {
                    scan_result = Err(e);
                    break;
                }
                let mut chunk = Vec::with_capacity(CHUNK_ROWS);
                loop {
                    match scan.try_next() {
                        Ok(Some((_tid, tuple))) => {
                            chunk.push(tuple);
                            if chunk.len() == CHUNK_ROWS {
                                break;
                            }
                        },
                        Ok(None) => break,
                        Err(e) => {
                            scan_result = Err(e.into());
                            break;
                        },
                    }
                }
                if scan_result.is_err() || chunk.is_empty() {
                    break;
                }
                let exhausted = chunk.len() < CHUNK_ROWS;
                let worker = usize::try_from(seq).unwrap_or(usize::MAX) % workers;
                if senders
                    .get(worker)
                    .is_none_or(|tx| tx.send((seq, chunk)).is_err())
                {
                    // A worker hung up early — it hit an error; stop feeding and surface it
                    // from its join below.
                    break;
                }
                seq += 1;
                if exhausted {
                    break;
                }
            }
            drop(senders);
            let mut first_err: Option<Error> = scan_result.err();
            for handle in handles {
                match handle.join() {
                    Ok(Ok(partial)) => partials.push(partial),
                    Ok(Err(e)) => {
                        first_err.get_or_insert(e);
                    },
                    Err(_) => {
                        first_err.get_or_insert_with(|| {
                            Error::Unsupported(
                                "internal: parallel aggregate worker panicked".to_owned(),
                            )
                        });
                    },
                }
            }
            first_err.map_or(Ok(()), Err)
        })?;

        self.merge_and_finalize(partials)
    }

    /// Merge the workers' partials and finalize in **global first-seen order** — the
    /// sequential emission order, reconstructed from the per-group minimum row position.
    /// Group creation is detected by growth (`find_or_create` returns the old length exactly
    /// when it appends), keeping `firsts` aligned with the states.
    fn merge_and_finalize(&self, partials: Vec<WorkerPartial>) -> Result<Vec<Row>, Error> {
        let mut merged = GroupIndex::new();
        let mut firsts: Vec<u64> = Vec::new();
        for partial in partials {
            for ((key, accs), first) in partial.states.into_iter().zip(partial.firsts) {
                let at = merged.find_or_create(key, self.calls.len());
                let slot = merged.accs_at(at).ok_or_else(|| {
                    Error::Unsupported("internal: parallel merge lost a group".to_owned())
                })?;
                if at == firsts.len() {
                    // First worker to contribute this group: adopt its accumulators wholesale.
                    firsts.push(first);
                    *slot = accs;
                } else {
                    for ((into, from), call) in slot.iter_mut().zip(accs).zip(&self.calls) {
                        merge_acc(into, from, call)?;
                    }
                    if let Some(seen) = firsts.get_mut(at)
                        && first < *seen
                    {
                        *seen = first;
                    }
                }
            }
        }
        let mut ordered: Vec<_> = firsts.into_iter().zip(merged.into_states()).collect();
        ordered.sort_unstable_by_key(|&(first, _)| first);
        let mut out = Vec::with_capacity(ordered.len().max(1));
        for (_, (key, accs)) in ordered {
            let mut row = key;
            for (acc, call) in accs.into_iter().zip(&self.calls) {
                row.push(finalize_aggregate(acc, call)?);
            }
            out.push(row);
        }
        // A scalar aggregate yields one row even over an empty input (COUNT 0, NULL folds) —
        // unlike GROUP BY's zero rows.
        if self.scalar && out.is_empty() {
            let mut row = Vec::with_capacity(self.calls.len());
            for call in &self.calls {
                row.push(finalize_aggregate(Acc::default(), call)?);
            }
            out.push(row);
        }
        Ok(out)
    }
}

/// One worker's loop: decode each received chunk through the typed-builder batch decode and
/// fold it — the per-batch body of [`GroupedAggregate`](super::GroupedAggregate)'s
/// `fold_child`, plus the global first-seen position bookkeeping.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the operator's own field set; bundling would just rename them"
)]
fn worker_fold(
    rx: &mpsc::Receiver<(u64, Vec<SharedTuple>)>,
    input_schema: &Arc<Schema>,
    source_types: &Arc<[nusadb_core::ColumnType]>,
    keep: &Arc<[usize]>,
    predicate: Option<&Arc<BatchPredicate>>,
    key_indices: &[usize],
    calls: &[AggregateCall],
    arg_shapes: &[ColumnarShape],
    group_cap: Option<usize>,
) -> Result<WorkerPartial, Error> {
    let mut groups = GroupIndex::new();
    let mut firsts: Vec<u64> = Vec::new();
    // Only a `STRING_AGG … ORDER BY`'s sort keys would read the row, and the mergeable gate
    // excludes it, so an empty row satisfies `fold_value`'s contract (as in the sequential fold).
    let empty_row: Row = Vec::new();
    while let Ok((seq, tuples)) = rx.recv() {
        let base = seq * u64::try_from(CHUNK_ROWS).unwrap_or(u64::MAX);
        let scan = ChunkScan {
            tuples: tuples.into_iter(),
        };
        let mut batches = RecordBatchScan::with_projection(
            Box::new(scan),
            Arc::clone(input_schema),
            Arc::clone(source_types),
            Arc::clone(keep),
        );
        let mut offset: u64 = 0;
        while let Some(batch) = batches.next().transpose()? {
            // Pushed-down WHERE: the keep-mask aligns with the batch's rows, so a dropped row
            // simply never folds while the survivors keep their global scan positions — the
            // first-seen reconstruction stays exact.
            let mask = predicate.map(|p| p.mask(&batch)).transpose()?;
            let key_cols = key_indices
                .iter()
                .map(|&c| batch.column(c).ok_or(Error::MalformedTuple { offset: c }))
                .collect::<Result<Vec<_>, _>>()?;
            let arg_cols = arg_shapes
                .iter()
                .map(|shape| match shape {
                    ColumnarShape::CountStar => Ok(None),
                    ColumnarShape::Column(c) => batch
                        .column(*c)
                        .map(Some)
                        .ok_or(Error::MalformedTuple { offset: *c }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            for i in 0..batch.num_rows() {
                if mask
                    .as_ref()
                    .is_some_and(|m| !m.get(i).copied().unwrap_or(false))
                {
                    continue;
                }
                let key: Vec<ast::Value> = key_cols
                    .iter()
                    .map(|col| value_at(col.as_ref(), i))
                    .collect();
                let at = groups.find_or_create(key, calls.len());
                if at == firsts.len() {
                    // The runtime valve for a spill-bounded engage: statistics promised a small
                    // group count; if reality blows far past it, abort loudly instead of growing
                    // unbounded hash state under a memory budget.
                    if group_cap.is_some_and(|cap| firsts.len() >= cap) {
                        return Err(Error::Unsupported(
                            "parallel aggregate exceeded its statistics-bounded group estimate \
                             (stale statistics?); re-run ANALYZE on the scanned table"
                                .to_owned(),
                        ));
                    }
                    firsts.push(base + offset + u64::try_from(i).unwrap_or(u64::MAX));
                }
                let Some(accs) = groups.accs_at(at) else {
                    continue;
                };
                for ((acc, call), col) in accs.iter_mut().zip(calls).zip(&arg_cols) {
                    match col {
                        None => fold_count_star(acc),
                        Some(col) => fold_value(acc, call, value_at(col.as_ref(), i), &empty_row)?,
                    }
                }
            }
            offset += u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        }
    }
    Ok(WorkerPartial {
        states: groups.into_states(),
        firsts,
    })
}

impl Operator for ParallelGroupedAggregate {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        if self.out.is_none() {
            self.out = Some(self.fold()?.into_iter());
        }
        let Some(iter) = self.out.as_mut() else {
            return Ok(None);
        };
        let rows: Vec<Row> = iter.by_ref().take(crate::BATCH_SIZE).collect();
        if rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(rows_to_batch(&self.schema, rows)?))
    }
}
