//! Pull-based streaming execution for the linear pipeline (Phase 1).
//!
//! [`execute_op`] materializes every operator into a full `Vec<Row>`, so a subtree holds all its
//! intermediate rows at once. This module adds a [`RowSource`] — a forward-only pull cursor — for
//! the *linear* operators (scan → filter → project → limit), so a consumer (a spilling grace hash
//! join or external sort, landing next) can draw rows one at a time and bound its own memory rather
//! than receive a pre-materialized Vec.
//!
//! Inherently blocking operators (sort, aggregate, join, window, set-op, recursive CTE, …) are
//! materialized once via [`execute_op`] and streamed back through [`Materialized`]. So [`stream_op`]
//! is a faithful drop-in: it yields exactly the rows `execute_op` would, in the same order — it only
//! changes *when* memory is held. Converting the blocking operators to spill internally is later
//! work in this phase.

#![allow(clippy::wildcard_imports)]

use std::borrow::Cow;

use nusadb_core::TupleScan;

use super::*;

/// A forward-only pull cursor over an operator's output rows.
pub(super) trait RowSource {
    /// The next row, or `Ok(None)` once the output is exhausted.
    ///
    /// # Errors
    /// Propagates any evaluation or storage error from producing the row.
    fn try_next(&mut self) -> Result<Option<Row>, Error>;
}

/// Adapts a fully-materialized `Vec<Row>` (a blocking operator's result) to a [`RowSource`].
struct Materialized(std::vec::IntoIter<Row>);

impl RowSource for Materialized {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        Ok(self.0.next())
    }
}

/// Counts the rows a streaming source yields and adds the total to the live `EXPLAIN ANALYZE`
/// collection when the source is dropped — so an early stop (e.g. under `LIMIT`) reports the rows
/// actually pulled, not the rows the node could have produced.
struct CountingSource<'a> {
    inner: Box<dyn RowSource + 'a>,
    key: usize,
    count: u64,
}

impl RowSource for CountingSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let row = self.inner.try_next()?;
        if row.is_some() {
            self.count += 1;
        }
        Ok(row)
    }
}

impl Drop for CountingSource<'_> {
    fn drop(&mut self) {
        super::instrument::record(self.key, self.count);
    }
}

/// Wrap a truly-streaming source in the row counter while an `EXPLAIN ANALYZE` collection is live
/// (a no-op box-through otherwise). The materialized fallbacks in [`stream_op`] are deliberately
/// NOT wrapped — their rows are already recorded by `execute_op` itself, and wrapping them again
/// would double-count the node.
fn counted<'a>(op: &PhysicalOperator, src: Box<dyn RowSource + 'a>) -> Box<dyn RowSource + 'a> {
    if super::instrument::enabled() {
        Box::new(CountingSource {
            inner: src,
            key: super::instrument::key(op),
            count: 0,
        })
    } else {
        src
    }
}

/// The `(start, stop, step)` of a `ProjectSet` that is exactly the integer
/// `generate_series(<int literal>, <int literal> [, <int literal>])` over `OneRow` — the shape a
/// lazy counting source streams in O(1) memory (the SRF-source residual of P-INSERTSEL-OOM /
/// ). Anything else — temporal series, expression bounds, `WITH ORDINALITY`, extra
/// projection columns — returns `None` and takes the materializing path unchanged.
pub(super) fn lazy_int_series(op: &PhysicalOperator) -> Option<(i64, i64, i64)> {
    let PhysicalOperator::ProjectSet {
        input,
        columns,
        ordinality: false,
    } = op
    else {
        return None;
    };
    if !matches!(**input, PhysicalOperator::OneRow) {
        return None;
    }
    let [column] = columns.as_slice() else {
        return None;
    };
    let crate::planner::TypedExprKind::SetReturning {
        func: ast::SetReturningFunc::GenerateSeries,
        args,
    } = &column.expr.kind
    else {
        return None;
    };
    let lit = |i: usize| match args.get(i).map(|a| &a.kind) {
        Some(crate::planner::TypedExprKind::Literal(ast::Value::Int(v))) => Some(*v),
        _ => None,
    };
    let start = lit(0)?;
    let stop = lit(1)?;
    let step = match args.len() {
        2 => 1,
        3 => lit(2)?,
        _ => return None,
    };
    // A zero step errors at evaluation; keep that loud path on the materializing side.
    (step != 0).then_some((start, stop, step))
}

/// Streams the integer `generate_series` lazily: one `Vec[Int]` row at a time, O(1) state.
/// Mirrors the materializing evaluator exactly — inclusive `stop`, sign-aware direction, clean
/// stop on `i64` overflow, and the same loud error past the 10M-row guard (streaming removes the
/// memory, not the runaway-series guard).
struct SeriesSource {
    next: Option<i64>,
    stop: i64,
    step: i64,
    emitted: usize,
}

impl RowSource for SeriesSource {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        const MAX_ROWS: usize = 10_000_000;
        let Some(cur) = self.next else {
            return Ok(None);
        };
        let in_range = if self.step > 0 {
            cur <= self.stop
        } else {
            cur >= self.stop
        };
        if !in_range {
            self.next = None;
            return Ok(None);
        }
        if self.emitted >= MAX_ROWS {
            return Err(Error::Unsupported(format!(
                "generate_series: the series exceeds the {MAX_ROWS}-row limit"
            )));
        }
        crate::cancel::check()?;
        self.emitted += 1;
        self.next = cur.checked_add(self.step);
        Ok(Some(vec![ast::Value::Int(cur)]))
    }
}

/// Streams a table's MVCC-visible rows straight from the engine cursor, decoding one at a time.
struct ScanSource {
    scan: Box<dyn TupleScan>,
    table: nusadb_core::TableId,
    schema: Vec<ColumnType>,
    /// Projection pushdown: the ascending source ordinals to keep, or empty for the full
    /// row. A non-empty list yields a narrowed row holding just those columns.
    keep: Vec<usize>,
}

impl RowSource for ScanSource {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let (tid, tuple) = loop {
            let Some((tid, tuple)) = self.scan.try_next()? else {
                return Ok(None);
            };
            // `FOR UPDATE ... SKIP LOCKED` (see `scan_table`): a row another transaction holds
            // locked is invisible to this pipeline.
            if !super::lock_skip::skipped(self.table, tid) {
                break (tid, tuple);
            }
        };
        let _ = tid;
        // Cooperative cancellation, matching the materializing `scan_table`.
        crate::cancel::check()?;
        if self.keep.is_empty() {
            Ok(Some(row::decode(&tuple, &self.schema)?))
        } else {
            Ok(Some(row::decode_projected(
                &tuple,
                &self.schema,
                &self.keep,
            )?))
        }
    }
}

/// Streams the `child` rows for which `predicate` evaluates true.
struct FilterSource<'a> {
    child: Box<dyn RowSource + 'a>,
    predicate: TypedExpr,
    correlated: bool,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
}

impl RowSource for FilterSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        while let Some(row) = self.child.try_next()? {
            let verdict = if self.correlated {
                eval_correlated(&self.predicate, &row, self.engine, self.txn)?
            } else {
                eval::eval(&self.predicate, &row)?
            };
            if matches!(verdict, ast::Value::Bool(true)) {
                return Ok(Some(row));
            }
        }
        Ok(None)
    }
}

/// Streams the `child` rows with each projection column applied.
struct ProjectSource<'a> {
    child: Box<dyn RowSource + 'a>,
    columns: Vec<TypedExpr>,
    any_correlated: bool,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
}

impl RowSource for ProjectSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let Some(row) = self.child.try_next()? else {
            return Ok(None);
        };
        let projected = self
            .columns
            .iter()
            .map(|expr| {
                if self.any_correlated && contains_subquery(expr) {
                    eval_correlated(expr, &row, self.engine, self.txn)
                } else {
                    eval::eval(expr, &row)
                }
            })
            .collect::<Result<Row, _>>()?;
        Ok(Some(projected))
    }
}

/// Streams `child` rows after skipping `skip`, then yields at most `remaining`.
struct LimitSource<'a> {
    child: Box<dyn RowSource + 'a>,
    skip: usize,
    remaining: usize,
}

impl RowSource for LimitSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        while self.skip > 0 {
            match self.child.try_next()? {
                Some(_) => self.skip -= 1,
                None => return Ok(None),
            }
        }
        if self.remaining == 0 {
            return Ok(None);
        }
        if let Some(row) = self.child.try_next()? {
            self.remaining -= 1;
            Ok(Some(row))
        } else {
            self.remaining = 0;
            Ok(None)
        }
    }
}

/// Build a pull-based [`RowSource`] for `op`.
///
/// Linear operators (scan / filter / project / limit) stream truly; any other operator is
/// materialized once via [`execute_op`] and adapted, so the result set is identical to
/// `execute_op(op)` — only the memory profile differs.
///
/// # Errors
/// Propagates planning/storage/evaluation errors from building the source chain (a materialized
/// fallback runs its whole operator eagerly here).
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-operator dispatch; length tracks the streamable operator set"
)]
pub(super) fn stream_op<'a>(
    op: &'a PhysicalOperator,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
) -> Result<Box<dyn RowSource + 'a>, Error> {
    match op {
        PhysicalOperator::SeqScan { table, columns } => {
            // A recursive CTE's working set lives in memory; stream it from there, as `scan_rows`
            // does (including its setup cancel checkpoint). The pushdown pass never narrows a
            // recursive-CTE scan, so these rows stay full width.
            if let Some(rows) = super::recursive::working_set(table.id) {
                crate::cancel::check()?;
                return Ok(counted(op, Box::new(Materialized(rows.into_iter()))));
            }
            let schema = column_types(table);
            let scan = engine.scan(txn, table.id)?;
            Ok(counted(
                op,
                Box::new(ScanSource {
                    scan,
                    table: table.id,
                    schema,
                    keep: columns.clone(),
                }),
            ))
        },
        PhysicalOperator::Filter { input, predicate } => {
            // Pre-resolve any uncorrelated subquery once; a correlated one stays for
            // per-row resolution — exactly as the materializing Filter arm does.
            let predicate = resolved_expr(predicate, engine, txn)?.into_owned();
            let correlated = contains_subquery(&predicate);
            Ok(counted(
                op,
                Box::new(FilterSource {
                    child: stream_op(input, engine, txn)?,
                    predicate,
                    correlated,
                    engine,
                    txn,
                }),
            ))
        },
        PhysicalOperator::Project { input, columns } => {
            let mut columns = columns
                .iter()
                .map(|p| resolved_expr(&p.expr, engine, txn).map(Cow::into_owned))
                .collect::<Result<Vec<_>, _>>()?;
            // Sequence built-ins (nextval/currval/setval) resolve to a literal exactly once, here where
            // the input is structurally a single row — a no-FROM SELECT's `OneRow`. This mirrors the
            // materializing `execute_op` Project arm; without it a bare `SELECT nextval('s')` streamed
            // to a driver (the extended-query path streams) would reach the per-row evaluator and be
            // wrongly rejected. A per-row input leaves the calls unresolved so the evaluator still
            // rejects them loudly (resolving once would under-advance the sequence).
            if matches!(**input, PhysicalOperator::OneRow)
                && columns.iter().any(contains_sequence_call)
            {
                for col in &mut columns {
                    resolve_sequence_calls(col, engine)?;
                }
            }
            let any_correlated = columns.iter().any(contains_subquery);
            Ok(counted(
                op,
                Box::new(ProjectSource {
                    child: stream_op(input, engine, txn)?,
                    columns,
                    any_correlated,
                    engine,
                    txn,
                }),
            ))
        },
        PhysicalOperator::Limit {
            input,
            count,
            offset,
        } => {
            let skip = usize::try_from(*offset).unwrap_or(usize::MAX);
            let remaining = usize::try_from(*count).unwrap_or(usize::MAX);
            Ok(counted(
                op,
                Box::new(LimitSource {
                    child: stream_op(input, engine, txn)?,
                    skip,
                    remaining,
                }),
            ))
        },
        // A scalar aggregate is blocking but needs only its accumulators, not its input:
        // pull the streamed input row by row and fold, where the
        // materializing path held the entire scan and OOMed a full-table `count(*)` past ~1M rows.
        PhysicalOperator::ScalarAggregate { input, calls } => {
            // `COUNT(*)` over a plain scan counts visible tuples without decoding any row.
            let out =
                if let Some(out) = super::agg::scalar_count_star_fast(input, calls, engine, txn)? {
                    out
                } else {
                    let mut child = stream_op(input, engine, txn)?;
                    super::agg::fold_aggregates_streamed(calls, child.as_mut())?
                };
            Ok(counted(op, Box::new(Materialized(vec![out].into_iter()))))
        },
        // A group aggregate likewise folds its streamed input into per-group accumulators —
        // O(groups) memory instead of O(input). When spill is configured, defer to `execute_op`
        // (the sort-based spilling group-by) so the bounded-memory contract for huge group
        // counts is preserved.
        PhysicalOperator::GroupAggregate {
            input,
            group_keys,
            calls,
        } if super::spill::spill_config().is_none() => {
            let mut child = stream_op(input, engine, txn)?;
            let out = super::agg::run_group_aggregate_streamed(child.as_mut(), group_keys, calls)?;
            Ok(counted(op, Box::new(Materialized(out.into_iter()))))
        },
        // A literal integer `generate_series` in FROM streams lazily (O(1) state) instead of
        // materializing every element — the common ETL source shape.
        PhysicalOperator::ProjectSet { .. } => match lazy_int_series(op) {
            Some((start, stop, step)) => Ok(counted(
                op,
                Box::new(SeriesSource {
                    next: Some(start),
                    stop,
                    step,
                    emitted: 0,
                }),
            )),
            None => Ok(Box::new(Materialized(
                execute_op(op, engine, txn)?.into_iter(),
            ))),
        },
        // A hash join streams its OUTPUT: the build (right) side is
        // materialized + indexed exactly as the materializing arm does — that is the memory the
        // join inherently needs — but the probe (left) side is pulled row by row and every output
        // row is emitted as produced. A `LIMIT` / aggregate / filter above the join therefore
        // holds O(build side) instead of O(probe + join output).
        //
        // With spill configured (the server default), the build side is collected under the
        // spill budget FIRST (residual): if it fits, the join streams
        // exactly as above — QA measured `LIMIT 5` over `orders(1M) JOIN dim(100)` OOMing here
        // because this arm used to be disabled whenever spill was on, falling back to the
        // materializing path that buffered the whole 1M-row probe input. Only a build side that
        // genuinely overflows the budget takes the blocking grace partitioning (which streams
        // both inputs to disk, bounded).
        PhysicalOperator::HashJoin {
            left,
            right,
            keys,
            residual,
            kind,
            left_width,
            right_width,
            coalesce_pairs,
        } => {
            // Pre-resolve the residual (only it can carry a subquery), as every arm does.
            let resolved = residual
                .as_ref()
                .map(|r| resolved_expr(r, engine, txn).map(Cow::into_owned))
                .transpose()?;
            // (build-side selection, v1 gate): for an INNER equi-join with no
            // USING/NATURAL merge whose two inputs both resolve against ONE table's statistics
            // (the self-join shape), build on the LEFT when it is estimated decisively smaller
            // — instead of unconditionally materializing the right. The canonical win is a
            // pushed-down selective filter on the left of a self-join: a thousand-row build
            // replaces a millions-row one, and the big side streams as the probe. Everything
            // outside the gates keeps today's build-right path.
            if let Some(left_rows) =
                left_build_rows(left, right, *kind, coalesce_pairs, engine, txn)?
            {
                let table = super::join::JoinIndex::build_left(&left_rows, keys, *left_width)?;
                return Ok(counted(
                    op,
                    Box::new(HashJoinLeftBuildSource {
                        right: stream_op(right, engine, txn)?,
                        left_rows,
                        table,
                        keys,
                        residual: resolved,
                        left_width: *left_width,
                        pending: std::collections::VecDeque::new(),
                        padded: vec![ast::Value::Null; *left_width],
                        pulled: 0,
                    }),
                ));
            }
            let right_rows = match super::spill::spill_config() {
                // No spill: materialize the build side via the ordinary operator path.
                None => execute_op(right, engine, txn)?,
                Some(cfg) => {
                    match super::join::grace_build_or_partition(
                        left,
                        right,
                        keys,
                        resolved.as_ref(),
                        *kind,
                        *left_width,
                        *right_width,
                        &cfg,
                        engine,
                        txn,
                    )? {
                        super::join::GraceBuild::Fits(rows) => rows,
                        // Build overflowed: the whole join already ran through disk
                        // partitioning — stream the finished rows out.
                        super::join::GraceBuild::Joined(rows) => {
                            let rows = super::join::merge_join_using_columns(rows, coalesce_pairs);
                            return Ok(counted(op, Box::new(Materialized(rows.into_iter()))));
                        },
                    }
                },
            };
            let source = open_hash_join_source(
                left,
                right_rows,
                keys,
                resolved,
                *kind,
                *left_width,
                *right_width,
                coalesce_pairs,
                engine,
                txn,
            )?;
            Ok(counted(op, source))
        },
        // Inherently blocking or rarer operators: materialize once, then stream the buffer. Result is
        // identical to `execute_op`; converting these to spill internally is the next phase.
        _ => Ok(Box::new(Materialized(
            execute_op(op, engine, txn)?.into_iter(),
        ))),
    }
}

/// Build the streaming hash-join source over an already-collected build side (the caller
/// materialized it via `execute_op`, or budget-bounded it through the grace build phase), index
/// it, and stream the probe (left) side.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the HashJoin operator's own field set; bundling would just rename the arms"
)]
fn open_hash_join_source<'a>(
    left: &'a PhysicalOperator,
    right_rows: Vec<Row>,
    keys: &'a [crate::planner::HashKey],
    residual: Option<TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
    coalesce_pairs: &'a [(usize, usize)],
    engine: &'a dyn StorageEngine,
    txn: TxnId,
) -> Result<Box<dyn RowSource + 'a>, Error> {
    let table = super::join::JoinIndex::build_right(&right_rows, keys, left_width)?;
    let right_matched = vec![false; right_rows.len()];
    Ok(Box::new(HashJoinSource {
        left: stream_op(left, engine, txn)?,
        right_rows,
        table,
        keys,
        residual,
        kind,
        left_width,
        right_width,
        coalesce_pairs,
        right_matched,
        pending: std::collections::VecDeque::new(),
        drain_at: None,
    }))
}

/// The gate: materialize and return the LEFT input as the build side iff the flip is
/// safe and decisively profitable. Gates: INNER kind (no unmatched bookkeeping on either side);
/// no `USING`/`NATURAL` merge; both inputs scan exactly one table and it is the SAME analyzed
/// table (the self-join shape — the one case the single-table cost context resolves both sides
/// of); the left's post-filter estimate is finite, non-zero, at most a quarter of the right's,
/// and small in absolute terms (a stale estimate stays bounded). `None` = keep the default
/// build-right path.
fn left_build_rows(
    left: &PhysicalOperator,
    right: &PhysicalOperator,
    kind: ast::JoinKind,
    coalesce_pairs: &[(usize, usize)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<Vec<Row>>, Error> {
    /// The absolute build-size cap: past this, the flip's estimate risk outweighs its win.
    const MAX_LEFT_BUILD_EST: f64 = 100_000.0;

    if !matches!(kind, ast::JoinKind::Inner) || !coalesce_pairs.is_empty() {
        return Ok(None);
    }
    let (Some(lt), Some(rt)) = (single_subtree_table(left), single_subtree_table(right)) else {
        return Ok(None);
    };
    if lt.id != rt.id {
        return Ok(None);
    }
    let Some(stats) = engine.table_stats(lt.id)? else {
        return Ok(None);
    };
    let ctx = super::cost::ScanStats::new(lt, &stats);
    let est_left = super::cost::estimate_rows(left, Some(&ctx));
    let est_right = super::cost::estimate_rows(right, Some(&ctx));
    if !(est_left.is_finite()
        && est_right.is_finite()
        && est_left > 0.0
        && est_left * 4.0 <= est_right
        && est_left <= MAX_LEFT_BUILD_EST)
    {
        return Ok(None);
    }
    // Observability for perf verification (RUST_LOG=debug): which joins actually flipped.
    tracing::debug!(est_left, est_right, "hash join: left-build flip");
    Ok(Some(execute_op(left, engine, txn)?))
}

/// The lone table an operator subtree scans, or `None` for zero or several.
fn single_subtree_table(op: &PhysicalOperator) -> Option<&TableSchema> {
    let mut found = None;
    let mut count = 0usize;
    super::collect_scan_tables(op, &mut found, &mut count);
    if count == 1 { found } else { None }
}

/// The left-build streaming hash join (INNER only): the small LEFT side is materialized
/// and indexed; the big RIGHT side streams as the probe, each match emitting `[left ++ right]`
/// — the planned column order, so nothing downstream changes. Peak memory is O(left + output)
/// instead of O(right).
struct HashJoinLeftBuildSource<'a> {
    right: Box<dyn RowSource + 'a>,
    left_rows: Vec<Row>,
    table: super::join::JoinIndex,
    keys: &'a [crate::planner::HashKey],
    residual: Option<TypedExpr>,
    left_width: usize,
    /// Output rows produced by the current probe row, drained before the next pull.
    pending: std::collections::VecDeque<Row>,
    /// Reused probe scratch: the NULL prefix stays, only the right portion refreshes —
    /// this loop runs once per row of the BIG side, the whole reason the flip exists.
    padded: Row,
    pulled: u64,
}

impl RowSource for HashJoinLeftBuildSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        loop {
            if let Some(row) = self.pending.pop_front() {
                return Ok(Some(row));
            }
            let Some(right_row) = self.right.try_next()? else {
                return Ok(None);
            };
            // Cooperative cancellation at probe-row granularity, amortized.
            self.pulled += 1;
            if self.pulled.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            // Right-key expressions reference joined ordinals `>= left_width`; the index shifts
            // the bare probe row into the reused `padded` scratch when its path needs that.
            let Some(indices) =
                self.table
                    .probe_right(self.keys, &right_row, self.left_width, &mut self.padded)?
            else {
                continue;
            };
            for &index in indices {
                let Some(left_row) = self.left_rows.get(index) else {
                    continue;
                };
                let mut joined = left_row.clone();
                joined.extend(right_row.iter().cloned());
                if super::join::residual_passes(self.residual.as_ref(), &joined)? {
                    self.pending.push_back(joined);
                }
            }
        }
    }
}

/// Streams a hash join's output. Emission order is **identical** to
/// [`super::join::run_hash_join`]: for each probe (left) row, its matches in build-index order,
/// then the NULL-padded row for an unmatched LEFT/FULL probe; once the probe side is exhausted,
/// the unmatched build rows for RIGHT/FULL in build order. `USING` coalescing applies per row on
/// emission, exactly as the materializing arm's whole-result pass does.
struct HashJoinSource<'a> {
    left: Box<dyn RowSource + 'a>,
    right_rows: Vec<Row>,
    table: super::join::JoinIndex,
    keys: &'a [crate::planner::HashKey],
    residual: Option<TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
    coalesce_pairs: &'a [(usize, usize)],
    right_matched: Vec<bool>,
    /// Output rows produced by the current probe row, drained before the next pull.
    pending: std::collections::VecDeque<Row>,
    /// `Some(cursor)` once the probe side is exhausted — the RIGHT/FULL unmatched-build drain.
    drain_at: Option<usize>,
}

impl HashJoinSource<'_> {
    /// Probe one left row against the build index, queueing its output rows — the per-probe body
    /// of `run_hash_join`, verbatim in effect.
    fn probe(&mut self, left_row: &Row) -> Result<(), Error> {
        // Detach the match list from the table borrow so the matched-flags can be set below.
        let indices: Vec<usize> = self
            .table
            .probe_left(self.keys, left_row)?
            .cloned()
            .unwrap_or_default();
        let mut matched = false;
        for index in indices {
            let Some(right_row) = self.right_rows.get(index) else {
                continue;
            };
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if super::join::residual_passes(self.residual.as_ref(), &joined)? {
                self.pending.push_back(joined);
                matched = true;
                if let Some(flag) = self.right_matched.get_mut(index) {
                    *flag = true;
                }
            }
        }
        if matches!(self.kind, ast::JoinKind::Left | ast::JoinKind::Full) && !matched {
            let mut joined = left_row.clone();
            joined.extend(std::iter::repeat_n(ast::Value::Null, self.right_width));
            self.pending.push_back(joined);
        }
        Ok(())
    }

    /// Apply the `USING` coalesce pairs to one output row (the per-row form of
    /// [`super::join::merge_join_using_columns`]).
    fn coalesced(&self, row: Row) -> Row {
        if self.coalesce_pairs.is_empty() {
            return row;
        }
        super::join::merge_join_using_columns(vec![row], self.coalesce_pairs)
            .pop()
            .unwrap_or_default()
    }
}

impl RowSource for HashJoinSource<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        loop {
            if let Some(row) = self.pending.pop_front() {
                return Ok(Some(self.coalesced(row)));
            }
            if let Some(cursor) = self.drain_at {
                // Probe side exhausted: RIGHT/FULL emits each unmatched build row NULL-padded on
                // the left, in build order (same tail as the materializing join).
                let mut i = cursor;
                while let Some(&matched) = self.right_matched.get(i) {
                    if matched {
                        i += 1;
                        continue;
                    }
                    let Some(right_row) = self.right_rows.get(i) else {
                        break;
                    };
                    let mut joined: Row = vec![ast::Value::Null; self.left_width];
                    joined.extend(right_row.iter().cloned());
                    self.drain_at = Some(i + 1);
                    return Ok(Some(self.coalesced(joined)));
                }
                self.drain_at = Some(i);
                return Ok(None);
            }
            match self.left.try_next()? {
                Some(left_row) => self.probe(&left_row)?,
                None if matches!(self.kind, ast::JoinKind::Right | ast::JoinKind::Full) => {
                    self.drain_at = Some(0);
                },
                None => return Ok(None),
            }
        }
    }
}
