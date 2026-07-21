//! Join operators: `HashJoin`, `NestedLoopJoin`.
//!
//! Hash join: build side is the smaller table (estimated by planner).
//!
//! Hoisted verbatim out of `executor/ops.rs` (ADR 007 §4.6 deviation cleanup).
//! Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;
use crate::planner::TypedExprKind;

/// Nested-loop join over materialized inputs: emit `[left ++ right]` for each
/// pair satisfying `predicate`, plus the outer-join unmatched rows NULL-padded
/// on the absent side (see [`PhysicalOperator::NestedLoopJoin`]).
pub(super) fn run_nested_loop_join(
    left_rows: &[Row],
    right_rows: &[Row],
    predicate: &TypedExpr,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
) -> Result<Vec<Row>, Error> {
    let keep_unmatched_left = matches!(kind, ast::JoinKind::Left | ast::JoinKind::Full);
    let keep_unmatched_right = matches!(kind, ast::JoinKind::Right | ast::JoinKind::Full);
    let mut right_matched = vec![false; right_rows.len()];
    let mut out = Vec::new();
    // Bound the materialized output by work_mem, checked incrementally as it grows: a cross join of a
    // small outer against a large inner (e.g. 4000×20000 = 80M rows) would otherwise fill memory and
    // get the whole server OOM-killed before any post-stage guard runs. Failing loudly here keeps the
    // server alive. `budget == 0` (unset) means unlimited.
    // Use the session-effective budget so a per-connection `SET work_mem` bounds the join too, exactly
    // like every other stage guard.
    let budget = super::ops::effective_work_mem();
    let mut out_bytes: usize = 0;
    // Amortize the cancel poll over the inner loop (not just per outer row) so a statement timeout
    // interrupts a small-outer × large-inner scan between inner batches, not only between outer rows.
    let mut steps: u32 = 0;
    for left_row in left_rows {
        // Cooperative cancellation: break out of an O(n·m) nested-loop join at a left-row
        // boundary rather than running it to completion.
        crate::cancel::check()?;
        let mut left_matched = false;
        for (matched, right_row) in right_matched.iter_mut().zip(right_rows) {
            steps = steps.wrapping_add(1);
            if steps.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if matches!(eval::eval(predicate, &joined)?, ast::Value::Bool(true)) {
                if budget != 0 {
                    out_bytes += super::ops::row_bytes(&joined);
                    if out_bytes > budget {
                        return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                            "nested-loop join output exceeded work_mem of {budget} bytes \
                             ({out_bytes} bytes materialized) — add a more selective join \
                             predicate / WHERE, or raise --work-mem"
                        ))));
                    }
                }
                out.push(joined);
                left_matched = true;
                *matched = true;
            }
        }
        // Outer join keeps an unmatched left row, NULL-padded on the right.
        if keep_unmatched_left && !left_matched {
            let mut joined = left_row.clone();
            joined.extend(std::iter::repeat_n(ast::Value::Null, right_width));
            out.push(joined);
        }
    }
    // Outer join keeps unmatched right rows, NULL-padded on the left.
    if keep_unmatched_right {
        for (right_row, matched) in right_rows.iter().zip(&right_matched) {
            if !matched {
                let mut joined: Row = vec![ast::Value::Null; left_width];
                joined.extend(right_row.iter().cloned());
                out.push(joined);
            }
        }
    }
    Ok(out)
}

/// Apply a `USING`/`NATURAL` join's column merge: for each `(left, right)` ordinal pair, set the
/// kept-left slot to `coalesce(left, right)` — i.e. take the right side's value when the left is
/// NULL. The merged column then reads once, correctly, from the left slot for every join kind:
/// INNER/LEFT rows already carry the left value (this is a no-op there), while a RIGHT/FULL join's
/// unmatched row has a NULL-padded left and a present right, so the right value surfaces. Returns
/// the rows unchanged when there are no merged columns (an `ON`/`CROSS` join).
pub(super) fn merge_join_using_columns(mut rows: Vec<Row>, pairs: &[(usize, usize)]) -> Vec<Row> {
    if pairs.is_empty() {
        return rows;
    }
    for row in &mut rows {
        for &(left, right) in pairs {
            // Only an unmatched RIGHT/FULL row has a NULL left here; fill it from the right copy.
            if matches!(row.get(left), Some(ast::Value::Null))
                && let Some(value) = row.get(right).cloned()
                && let Some(slot) = row.get_mut(left)
            {
                *slot = value;
            }
        }
    }
    rows
}

/// Hash join over materialized inputs. Builds a hash table on `right_rows`
/// keyed by the right side of each [`HashKey`], then probes it once per left
/// row. Emits `[left ++ right]` for each hash-matched pair that also passes
/// `residual`, plus the outer-join unmatched rows NULL-padded on the absent
/// side — identical results to [`run_nested_loop_join`] for the equivalent
/// predicate. A `NULL` key value never matches (SQL `NULL = NULL` is unknown),
/// so such rows are treated as unmatched.
pub(super) fn run_hash_join(
    left_rows: &[Row],
    right_rows: &[Row],
    keys: &[HashKey],
    residual: Option<&TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
) -> Result<Vec<Row>, Error> {
    let table = JoinIndex::build_right(right_rows, keys, left_width)?;
    let keep_unmatched_left = matches!(kind, ast::JoinKind::Left | ast::JoinKind::Full);
    let keep_unmatched_right = matches!(kind, ast::JoinKind::Right | ast::JoinKind::Full);
    let mut right_matched = vec![false; right_rows.len()];
    let mut out = Vec::new();

    for left_row in left_rows {
        // Left-key expressions reference ordinals `< left_width`, which are exactly the columns of
        // the bare `left_row` — probe against it directly (no per-probe clone + NULL pad).
        let indices = table.probe_left(keys, left_row)?;
        let mut matched = false;
        for &index in indices.into_iter().flatten() {
            let Some(right_row) = right_rows.get(index) else {
                continue;
            };
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if residual_passes(residual, &joined)? {
                out.push(joined);
                matched = true;
                if let Some(flag) = right_matched.get_mut(index) {
                    *flag = true;
                }
            }
        }
        if keep_unmatched_left && !matched {
            let mut joined = left_row.clone();
            joined.extend(std::iter::repeat_n(ast::Value::Null, right_width));
            out.push(joined);
        }
    }

    if keep_unmatched_right {
        for (right_row, matched) in right_rows.iter().zip(&right_matched) {
            if !matched {
                let mut joined: Row = vec![ast::Value::Null; left_width];
                joined.extend(right_row.iter().cloned());
                out.push(joined);
            }
        }
    }
    Ok(out)
}

/// [`run_hash_join`] with a **streamed probe** (residual): identical
/// output, but the probe (left) side is pulled row by row instead of being materialized first —
/// QA measured `orders(1M) JOIN dim(100)` `OOMing` purely on the probe-input `Vec` even though the
/// build side was tiny. Peak memory is O(build + output), not O(probe + build + output).
pub(super) fn run_hash_join_streamed(
    left_src: &mut dyn super::stream::RowSource,
    right_rows: &[Row],
    keys: &[HashKey],
    residual: Option<&TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
) -> Result<Vec<Row>, Error> {
    let table = JoinIndex::build_right(right_rows, keys, left_width)?;
    let keep_unmatched_left = matches!(kind, ast::JoinKind::Left | ast::JoinKind::Full);
    let keep_unmatched_right = matches!(kind, ast::JoinKind::Right | ast::JoinKind::Full);
    let mut right_matched = vec![false; right_rows.len()];
    let mut out = Vec::new();
    let mut pulled: u64 = 0;

    while let Some(left_row) = left_src.try_next()? {
        // Cooperative cancellation at probe-row granularity, amortized.
        pulled += 1;
        if pulled.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        let indices = table.probe_left(keys, &left_row)?;
        let mut matched = false;
        for &index in indices.into_iter().flatten() {
            let Some(right_row) = right_rows.get(index) else {
                continue;
            };
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if residual_passes(residual, &joined)? {
                out.push(joined);
                matched = true;
                if let Some(flag) = right_matched.get_mut(index) {
                    *flag = true;
                }
            }
        }
        if keep_unmatched_left && !matched {
            let mut joined = left_row;
            joined.extend(std::iter::repeat_n(ast::Value::Null, right_width));
            out.push(joined);
        }
    }

    if keep_unmatched_right {
        for (right_row, matched) in right_rows.iter().zip(&right_matched) {
            if !matched {
                let mut joined: Row = vec![ast::Value::Null; left_width];
                joined.extend(right_row.iter().cloned());
                out.push(joined);
            }
        }
    }
    Ok(out)
}

/// Number of disk partitions a grace hash join fans its inputs into. Fixed for now; matching
/// keys always co-locate, so correctness is independent of the count — it only trades partition size
/// against file count.
const GRACE_PARTITIONS: usize = 16;

/// Extra bucket (index `GRACE_PARTITIONS`) holding the NULL-keyed rows of both sides — they match
/// nothing, so they cannot be hash-partitioned, but an outer join must still emit them unmatched.
const NULL_BUCKET: usize = GRACE_PARTITIONS;

/// Monotonic id so concurrent grace joins never collide on a partition file name. Not persisted, so
/// process-local ordering is all that matters.
static GRACE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Partition a key tuple deterministically (matching keys land together — that is the whole point).
fn partition_of(key: &[KeyAtom]) -> usize {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    (hasher.finish() % GRACE_PARTITIONS as u64) as usize
}

/// Equi-join that bounds the build side to `config.threshold_bytes` by spilling to disk (
/// grace hash join), for any equi-join `kind` (`Inner`/`Left`/`Right`/`Full`). It first streams the
/// build (right) side into memory under a [`MemBudget`](super::spill); if it fits, it falls through
/// to the in-memory [`run_hash_join`]. If the build overflows the budget, both inputs are
/// hash-partitioned to disk by their join key and each partition pair is joined in memory in turn,
/// so peak memory is ~one partition rather than the whole build side.
///
/// Matching keys always hash to the same partition, so per-partition `run_hash_join` produces the
/// same matches *and* the same within-partition unmatched (outer) rows as the in-memory join.
/// `NULL`-keyed rows match nothing in an equi-join, so they cannot be partitioned by key — both
/// sides' NULL-key rows go to a dedicated null bucket and are joined together: `run_hash_join` over
/// it emits exactly the unmatched-left / unmatched-right rows the outer `kind` requires (and nothing
/// for `Inner`).
///
/// # Errors
/// Propagates streaming, spill-file I/O, and predicate-evaluation errors.
/// The outcome of the grace build phase: either the build side fit the budget (the caller joins
/// against it — streaming the probe and, on the streaming path, the output too), or it
/// overflowed and the whole join ran through disk partitioning here.
pub(super) enum GraceBuild {
    /// The build (right) side fit `threshold_bytes`: join against these rows.
    Fits(Vec<Row>),
    /// The build overflowed: both sides were partitioned to disk and joined — the finished rows.
    Joined(Vec<Row>),
}

/// Phase 1 of the grace hash join: stream the build (right) side under the budget. If it
/// fits, hand the buffered build back ([`GraceBuild::Fits`]) so the caller can stream the probe
/// (and its output) against it. If it overflows, **continue the same right stream** into disk
/// partitioning — never re-open the operator, which under `READ COMMITTED` could observe a
/// different statement snapshot — partition the left side, join the partition pairs, and return
/// the finished rows ([`GraceBuild::Joined`]).
///
/// # Errors
/// Propagates streaming, spill-file I/O, and predicate-evaluation errors.
#[allow(
    clippy::too_many_arguments,
    reason = "join shape mirrors run_hash_join + the spill config"
)]
pub(super) fn grace_build_or_partition(
    left: &PhysicalOperator,
    right: &PhysicalOperator,
    keys: &[HashKey],
    residual: Option<&TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<GraceBuild, Error> {
    // Phase 1: buffer the build (right) side until it either ends or exceeds the budget.
    let mut right_buf: Vec<Row> = Vec::new();
    let mut budget = super::spill::MemBudget::new(config.threshold_bytes);
    let mut right_src = super::stream::stream_op(right, engine, txn)?;
    let mut overflow: Option<Row> = None;
    while let Some(row) = right_src.try_next()? {
        if budget.admit(&row) {
            right_buf.push(row);
        } else {
            overflow = Some(row);
            break;
        }
    }

    if overflow.is_none() {
        return Ok(GraceBuild::Fits(right_buf));
    }

    // Overflow → grace-partition both sides to disk, then join partition pairs one at a time.
    let seq = GRACE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut right_parts = PartitionSet::new(&config.dir, seq, "r");
    let mut left_parts = PartitionSet::new(&config.dir, seq, "l");

    // Partition the right side: the buffered rows, the overflow row, then the rest of the stream.
    // Right-key expressions reference ordinals `>= left_width`, so evaluate against a left-padded row
    // exactly as `build_right_index` does. A NULL key (matches nothing) goes to the null bucket so an
    // outer join can still surface it as an unmatched right row.
    let mut padded: Row = vec![ast::Value::Null; left_width];
    let mut partition_right = |row: Row, parts: &mut PartitionSet| -> Result<(), Error> {
        padded.truncate(left_width);
        padded.extend_from_slice(&row);
        let bucket =
            key_atoms(keys, &padded, KeySide::Right)?.map_or(NULL_BUCKET, |k| partition_of(&k));
        parts.write(bucket, &row)
    };
    for row in std::mem::take(&mut right_buf) {
        partition_right(row, &mut right_parts)?;
    }
    if let Some(row) = overflow {
        partition_right(row, &mut right_parts)?;
    }
    while let Some(row) = right_src.try_next()? {
        partition_right(row, &mut right_parts)?;
    }

    // Partition the left side by its own key (NULL key → null bucket for outer unmatched-left).
    let mut left_src = super::stream::stream_op(left, engine, txn)?;
    while let Some(row) = left_src.try_next()? {
        let bucket =
            key_atoms(keys, &row, KeySide::Left)?.map_or(NULL_BUCKET, |k| partition_of(&k));
        left_parts.write(bucket, &row)?;
    }

    // Join every bucket (incl. the null bucket) with the real `kind`. We process even a partition
    // whose right side is empty: an outer join must still emit its left rows NULL-padded.
    let mut out = Vec::new();
    for p in 0..=NULL_BUCKET {
        crate::cancel::check()?;
        let right_rows = right_parts.read(p)?;
        let left_rows = left_parts.read(p)?;
        if left_rows.is_empty() && right_rows.is_empty() {
            continue;
        }
        out.extend(run_hash_join(
            &left_rows,
            &right_rows,
            keys,
            residual,
            kind,
            left_width,
            right_width,
        )?);
    }
    Ok(GraceBuild::Joined(out))
}

/// Equi-join that bounds the build side to `config.threshold_bytes` by spilling to disk,
/// for any equi-join `kind`. When the build fits, the probe side is **streamed** against it
/// (residual — the probe input is never materialized); when it overflows,
/// both inputs hash-partition to disk and each partition pair joins in memory in turn.
///
/// # Errors
/// Propagates streaming, spill-file I/O, and predicate-evaluation errors.
#[allow(
    clippy::too_many_arguments,
    reason = "join shape mirrors run_hash_join + the spill config"
)]
pub(super) fn grace_join(
    left: &PhysicalOperator,
    right: &PhysicalOperator,
    keys: &[HashKey],
    residual: Option<&TypedExpr>,
    kind: ast::JoinKind,
    left_width: usize,
    right_width: usize,
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    match grace_build_or_partition(
        left,
        right,
        keys,
        residual,
        kind,
        left_width,
        right_width,
        config,
        engine,
        txn,
    )? {
        GraceBuild::Joined(out) => Ok(out),
        GraceBuild::Fits(right_buf) => {
            let mut left_src = super::stream::stream_op(left, engine, txn)?;
            run_hash_join_streamed(
                left_src.as_mut(),
                &right_buf,
                keys,
                residual,
                kind,
                left_width,
                right_width,
            )
        },
    }
}

/// A lazily-created set of on-disk partition files for one side of a grace join.
struct PartitionSet<'a> {
    dir: &'a std::path::Path,
    seq: u64,
    side: &'static str,
    writers: Vec<Option<super::spill::SpillWriter>>,
}

impl<'a> PartitionSet<'a> {
    fn new(dir: &'a std::path::Path, seq: u64, side: &'static str) -> Self {
        // One slot per hash partition plus the null bucket (`0..=NULL_BUCKET`).
        let mut writers = Vec::with_capacity(NULL_BUCKET + 1);
        writers.resize_with(NULL_BUCKET + 1, || None);
        Self {
            dir,
            seq,
            side,
            writers,
        }
    }

    /// Append `row` to partition `p`, creating its spill file on first use.
    fn write(&mut self, p: usize, row: &[ast::Value]) -> Result<(), Error> {
        // Include the process id: `seq` is a process-local counter, so two NusaDB processes (or two
        // test binaries) sharing one spill directory would otherwise collide on the same file name.
        let path = self.dir.join(format!(
            "nusadb-spill-grace-{}-{}-{}-{p}.tmp",
            std::process::id(),
            self.seq,
            self.side
        ));
        let Some(slot) = self.writers.get_mut(p) else {
            return Ok(()); // p <= NULL_BUCKET by construction; defensive
        };
        if slot.is_none() {
            *slot = Some(super::spill::SpillWriter::create(path)?);
        }
        if let Some(writer) = slot {
            writer.write_row(row)?;
        }
        Ok(())
    }

    /// Read back every row of partition `p` (empty if it was never written to).
    fn read(&mut self, p: usize) -> Result<Vec<Row>, Error> {
        let Some(writer) = self.writers.get_mut(p).and_then(Option::take) else {
            return Ok(Vec::new());
        };
        let mut reader = writer.into_reader()?;
        let mut rows = Vec::new();
        while let Some(row) = reader.read_row()? {
            rows.push(row);
        }
        Ok(rows)
    }
}

/// Build the probe index for a hash join: map each right row's key atoms to its
/// position. Right-key expressions reference ordinals `>= left_width`, so they
/// are evaluated against a left-NULL-padded full row. Rows with a `NULL` key are
/// omitted (they match nothing) but remain eligible as unmatched right rows.
/// A built hash-join index: join key → indices into the build-side rows.
///
/// A hash-join bucket map keyed by the build rows' join key, hashed with `ahash` — ~3-5× faster
/// than the standard `DefaultHasher` (SipHash-1-3) for the short key tuples a join probes once per
/// row (the residual join cost QA measured). The map only *buckets*: membership is always decided
/// by key equality (`HashMap`'s own `Eq`), so a collision costs one comparison and the choice of
/// non-cryptographic hash never changes a join result. A **random** seed is used (not a fixed one)
/// so it keeps exactly the hash-flooding resistance the std default provides over user-controlled
/// join keys.
type JoinMap<K> = HashMap<K, Vec<usize>, ahash::RandomState>;

/// A fresh, randomly-seeded [`JoinMap`] (see its docs for why the seed is random here).
fn join_map<K>() -> JoinMap<K> {
    HashMap::with_hasher(ahash::RandomState::new())
}

/// `Generic` is the universal path: [`key_atoms`] per row (expression evaluation, one
/// `Vec<KeyAtom>` allocation, enum-tagged hashing). `Int` is the fast path, taken when
/// the join key is a single bare `Int`-physical column on **both** sides: such a column holds
/// only `Value::Int` or `NULL`, and `NULL` never matches an equi-join, so hashing the raw
/// `i64` is match-for-match equivalent to the generic atoms — with no eval call, no per-row
/// allocation, and no enum dispatch. Measured at 1M rows (release): build 883→339 ns/row,
/// probe 674→193 ns/row.
pub(super) enum JoinIndex {
    /// The universal [`key_atoms`]-keyed index.
    Generic(JoinMap<Vec<KeyAtom>>),
    /// The single-int-column fast path.
    Int {
        /// Left-key ordinal into the bare left row.
        left_col: usize,
        /// Right-key ordinal into the **bare** right row (the planner's joined-space ordinal
        /// minus `left_width`).
        right_col: usize,
        /// Raw key → indices into the build rows.
        map: JoinMap<i64>,
    },
}

/// The `(left_col, bare right_col)` ordinals iff the fast path applies: exactly one key,
/// both sides bare `Int`-physical columns (`INT`/`SMALLINT`/`BIGINT`; **not** `NUMERIC`, whose
/// keys canonicalize through the exact decimal). Right-key ordinals are in joined space by the
/// planner contract; one below `left_width` would be a planner bug and falls back to the
/// generic path rather than misindexing.
fn int_key_cols(keys: &[HashKey], left_width: usize) -> Option<(usize, usize)> {
    let [key] = keys else {
        return None;
    };
    let (TypedExprKind::Column(left), TypedExprKind::Column(right)) =
        (&key.left.kind, &key.right.kind)
    else {
        return None;
    };
    if !matches!(key.left.ty.physical(), ColumnType::Int)
        || !matches!(key.right.ty.physical(), ColumnType::Int)
    {
        return None;
    }
    Some((*left, right.checked_sub(left_width)?))
}

/// The raw fast-path key of `row`, if present: an Int-physical column is `Value::Int` or
/// `NULL`, and anything else (impossible under the type system, including an out-of-range
/// ordinal) is defensively non-matching — the same outcomes [`key_atoms`] produces.
fn int_key(row: &[ast::Value], col: usize) -> Option<i64> {
    match row.get(col) {
        Some(ast::Value::Int(k)) => Some(*k),
        _ => None,
    }
}

impl JoinIndex {
    /// Build the index over the RIGHT rows (the default build side).
    pub(super) fn build_right(
        right_rows: &[Row],
        keys: &[HashKey],
        left_width: usize,
    ) -> Result<Self, Error> {
        if let Some((left_col, right_col)) = int_key_cols(keys, left_width) {
            let mut map: JoinMap<i64> = join_map();
            for (index, row) in right_rows.iter().enumerate() {
                if let Some(k) = int_key(row, right_col) {
                    map.entry(k).or_default().push(index);
                }
            }
            return Ok(Self::Int {
                left_col,
                right_col,
                map,
            });
        }
        Ok(Self::Generic(build_right_index(
            right_rows, keys, left_width,
        )?))
    }

    /// Build the index over the LEFT rows (the flipped build side).
    pub(super) fn build_left(
        left_rows: &[Row],
        keys: &[HashKey],
        left_width: usize,
    ) -> Result<Self, Error> {
        if let Some((left_col, right_col)) = int_key_cols(keys, left_width) {
            let mut map: JoinMap<i64> = join_map();
            for (index, row) in left_rows.iter().enumerate() {
                if let Some(k) = int_key(row, left_col) {
                    map.entry(k).or_default().push(index);
                }
            }
            return Ok(Self::Int {
                left_col,
                right_col,
                map,
            });
        }
        Ok(Self::Generic(build_left_index(left_rows, keys)?))
    }

    /// Probe with a bare LEFT row (the index was built over the right side).
    pub(super) fn probe_left(
        &self,
        keys: &[HashKey],
        left_row: &Row,
    ) -> Result<Option<&Vec<usize>>, Error> {
        match self {
            Self::Generic(table) => Ok(key_atoms(keys, left_row, KeySide::Left)?
                .as_ref()
                .and_then(|key| table.get(key))),
            Self::Int { left_col, map, .. } => {
                Ok(int_key(left_row, *left_col).and_then(|k| map.get(&k)))
            },
        }
    }

    /// Probe with a bare RIGHT row (the index was built over the left side). The generic arm
    /// evaluates right-key expressions against joined-space ordinals, so it shifts the row into
    /// `padded` — the caller's reusable scratch whose first `left_width` slots stay `NULL`;
    /// the Int arm indexes the bare row directly and leaves the scratch untouched.
    pub(super) fn probe_right(
        &self,
        keys: &[HashKey],
        right_row: &[ast::Value],
        left_width: usize,
        padded: &mut Row,
    ) -> Result<Option<&Vec<usize>>, Error> {
        match self {
            Self::Generic(table) => {
                padded.truncate(left_width);
                padded.extend_from_slice(right_row);
                Ok(key_atoms(keys, padded, KeySide::Right)?
                    .as_ref()
                    .and_then(|key| table.get(key)))
            },
            Self::Int { right_col, map, .. } => {
                Ok(int_key(right_row, *right_col).and_then(|k| map.get(&k)))
            },
        }
    }
}

pub(super) fn build_right_index(
    right_rows: &[Row],
    keys: &[HashKey],
    left_width: usize,
) -> Result<JoinMap<Vec<KeyAtom>>, Error> {
    let mut table: JoinMap<Vec<KeyAtom>> = join_map();
    // Right-key expressions reference joined ordinals `>= left_width`, so the right columns must sit
    // at that offset. Reuse one `left_width`-NULL-prefixed scratch row across all entries: keep
    // the NULL prefix and refresh only the right portion each iteration.
    let mut padded: Row = vec![ast::Value::Null; left_width];
    for (index, right_row) in right_rows.iter().enumerate() {
        padded.truncate(left_width);
        padded.extend_from_slice(right_row);
        if let Some(key) = key_atoms(keys, &padded, KeySide::Right)? {
            table.entry(key).or_default().push(index);
        }
    }
    Ok(table)
}

/// Build a hash index over the LEFT rows (build-side selection): key → indices into
/// `left_rows`. Left-key expressions reference ordinals `< left_width`, which are exactly the
/// bare left row's columns, so no padding is needed (the mirror of [`build_right_index`]).
pub(super) fn build_left_index(
    left_rows: &[Row],
    keys: &[HashKey],
) -> Result<JoinMap<Vec<KeyAtom>>, Error> {
    let mut table: JoinMap<Vec<KeyAtom>> = join_map();
    for (index, left_row) in left_rows.iter().enumerate() {
        if let Some(key) = key_atoms(keys, left_row, KeySide::Left)? {
            table.entry(key).or_default().push(index);
        }
    }
    Ok(table)
}

/// The left-build INNER hash join over materialized inputs: build the index on the
/// (smaller) LEFT side and probe with the right — identical output rows to [`run_hash_join`]
/// for `Inner` (emission order differs: probe order is right-side order; SQL imposes none).
/// INNER only: no unmatched bookkeeping exists on either side.
pub(super) fn run_hash_join_left_build(
    left_rows: &[Row],
    right_rows: &[Row],
    keys: &[HashKey],
    residual: Option<&TypedExpr>,
    left_width: usize,
) -> Result<Vec<Row>, Error> {
    let table = JoinIndex::build_left(left_rows, keys, left_width)?;
    let mut out = Vec::new();
    let mut padded: Row = vec![ast::Value::Null; left_width];
    for (probed, right_row) in right_rows.iter().enumerate() {
        // Cooperative cancellation, amortized per probe row.
        if probed % 1024 == 1023 {
            crate::cancel::check()?;
        }
        let Some(indices) = table.probe_right(keys, right_row, left_width, &mut padded)? else {
            continue;
        };
        for &index in indices {
            let Some(left_row) = left_rows.get(index) else {
                continue;
            };
            let mut joined = left_row.clone();
            joined.extend(right_row.iter().cloned());
            if residual_passes(residual, &joined)? {
                out.push(joined);
            }
        }
    }
    Ok(out)
}

/// Whether a hash-matched joined row passes the (optional) residual predicate.
/// No residual means the equi-keys fully decide the match.
pub(super) fn residual_passes(residual: Option<&TypedExpr>, joined: &Row) -> Result<bool, Error> {
    match residual {
        Some(pred) => Ok(matches!(eval::eval(pred, joined)?, ast::Value::Bool(true))),
        None => Ok(true),
    }
}

/// Which side of each [`HashKey`] to evaluate.
#[derive(Clone, Copy)]
pub(super) enum KeySide {
    Left,
    Right,
}

/// A hashable join-key component with **compare-compatible** equality: two values the evaluator
/// calls equal must map to equal atoms — the invariant that lets the hash table stand in for
/// per-pair predicate evaluation (widened this from `Int`/`Bool`/`Text`).
/// The planner (`is_hashable_key_type`) admits only types where that holds; anything else
/// (notably `NULL`) makes the key non-matching.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum KeyAtom {
    Bool(bool),
    Int(i64),
    Text(String),
    Date(i32),
    Time(i64),
    TimeTz(i64),
    Timestamp(i64),
    TimestampTz(i64),
    Uuid([u8; 16]),
    /// `(mantissa, scale)` of the **trimmed** exact decimal: `1.0` and `1.00` compare equal, so
    /// they must hash together — the same canonical-decimal rule `distinct_hash` uses. Also
    /// produced for an `Int` value under a NUMERIC-typed key expression (a mixed CASE/COALESCE
    /// branch), which compares equal to its decimal spelling.
    Numeric(i128, u8),
    Bytes(Vec<u8>),
}

/// Evaluate the chosen side of every [`HashKey`] against `row`, returning the
/// key tuple — or `None` if any component is `NULL` (or otherwise non-hashable),
/// so the row matches nothing.
pub(super) fn key_atoms(
    keys: &[HashKey],
    row: &Row,
    side: KeySide,
) -> Result<Option<Vec<KeyAtom>>, Error> {
    let decimal_atom = |d: crate::numeric::Decimal| {
        let d = d.trim_scale();
        KeyAtom::Numeric(d.mantissa, d.scale)
    };
    let mut atoms = Vec::with_capacity(keys.len());
    for key in keys {
        let expr = match side {
            KeySide::Left => &key.left,
            KeySide::Right => &key.right,
        };
        let value = eval::eval(expr, row)?;
        // A NUMERIC-typed key canonicalizes through the exact decimal whatever the runtime
        // spelling: a mixed CASE/COALESCE branch can yield `Int(5)` where another row yields
        // `Numeric(5.00)`, and the evaluator calls those equal — they must hash together.
        if matches!(expr.ty.physical(), ColumnType::Numeric { .. }) {
            match value {
                ast::Value::Int(i) => {
                    atoms.push(decimal_atom(crate::numeric::Decimal::from_i64(i)));
                },
                ast::Value::Numeric(d) => atoms.push(decimal_atom(d)),
                _ => return Ok(None),
            }
            continue;
        }
        match value {
            ast::Value::Bool(b) => atoms.push(KeyAtom::Bool(b)),
            ast::Value::Int(i) => atoms.push(KeyAtom::Int(i)),
            ast::Value::Text(s) => atoms.push(KeyAtom::Text(s)),
            ast::Value::Date(d) => atoms.push(KeyAtom::Date(d)),
            ast::Value::Time(t) => atoms.push(KeyAtom::Time(t)),
            // TIMETZ's packed i64 compares exactly (instant, then zone) — hash the packed form.
            ast::Value::TimeTz(t) => atoms.push(KeyAtom::TimeTz(t)),
            ast::Value::Timestamp(t) => atoms.push(KeyAtom::Timestamp(t)),
            ast::Value::TimestampTz(t) => atoms.push(KeyAtom::TimestampTz(t)),
            ast::Value::Uuid(u) => atoms.push(KeyAtom::Uuid(u)),
            // A Numeric value under a non-NUMERIC declared type (defensive — coercions normally
            // keep the families apart; canonicalizing keeps the hash compare-compatible anyway).
            ast::Value::Numeric(d) => atoms.push(decimal_atom(d)),
            ast::Value::Bytes(b) => atoms.push(KeyAtom::Bytes(b)),
            // NULL never matches in an equi-join. Float (NaN / `-0.0`), Interval (mixed-unit
            // equality: `1 mon` = `30 days`), Json, and the containers are not hash-join key
            // types (`is_hashable_key_type`); defensively, such a value makes the row
            // non-matching on the hash path (the planner never plans them here).
            ast::Value::Null
            | ast::Value::Float(_)
            | ast::Value::Json(_)
            | ast::Value::Interval(_)
            | ast::Value::Array(_)
            | ast::Value::Vector(_) => return Ok(None),
        }
    }
    Ok(Some(atoms))
}
