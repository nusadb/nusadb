//! External merge sort for the row-path `Sort` operator.
//!
//! Bounds the sort's working memory to ~`threshold_bytes` by streaming the input into sorted runs
//! on disk (generating a run whenever the in-memory buffer fills), then k-way merging the runs with
//! a min-heap. If the whole input fits the budget it sorts in memory and is identical to the
//! [`sort_rows`] fast path. The merged result equals the in-memory sort exactly — the same
//! [`compare_order_key`](crate::executor::eval::compare_order_key) ordering drives both run
//! generation and the merge.
//!
//! Like the grace hash join, this bounds the *working set* (one run + the merge heads), not
//! the final output `Vec<Row>`; end-to-end output bounding is the streaming-output phase (Fase 2).

#![allow(clippy::wildcard_imports)]

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::*;
use crate::planner::OrderByKey;

/// Monotonic id for sort-run file names (process-local uniqueness; not persisted).
static SORT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// External merge sort of `input` by `keys` into a materialized `Vec<Row>`. Collects the
/// streaming [`sorted_input`] cursor; used by the row-path `Sort` operator.
///
/// # Errors
/// Propagates streaming, spill-file I/O, and key-evaluation errors.
pub(super) fn external_sort(
    input: &PhysicalOperator,
    keys: &[OrderByKey],
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let mut sorted = sorted_input(input, keys, config, engine, txn)?;
    let mut out = Vec::new();
    while let Some(row) = sorted.try_next()? {
        out.push(row);
    }
    Ok(out)
}

/// Sort `input` by `keys` and return a **streaming** cursor over the sorted rows. Bounds
/// working memory to ~`threshold_bytes`: run generation streams the input, sorting+spilling a run
/// whenever the buffer fills; the cursor then yields rows lazily (in-memory `IntoIter` when the whole
/// input fit, else a k-way merge of the disk runs). A sort-based aggregate folds groups over
/// this without ever materializing the sorted input.
///
/// # Errors
/// Propagates streaming, spill-file I/O, and key-evaluation errors.
pub(super) fn sorted_input<'a>(
    input: &PhysicalOperator,
    keys: &'a [OrderByKey],
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<SortedInput<'a>, Error> {
    let seq = SORT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut runs: Vec<super::spill::SpillReader> = Vec::new();
    let mut buf: Vec<Row> = Vec::new();
    let mut budget = super::spill::MemBudget::new(config.threshold_bytes);
    let mut src = super::stream::stream_op(input, engine, txn)?;

    while let Some(row) = src.try_next()? {
        if !budget.admit(&row) {
            spill_run(&mut buf, keys, config, seq, runs.len(), &mut runs)?;
            budget.reset();
            budget.admit(&row); // account the overflow row against the fresh buffer
        }
        buf.push(row);
    }

    // The whole input fit the budget → an in-memory sort, streamed back via `IntoIter`.
    if runs.is_empty() {
        sort_rows(&mut buf, keys)?;
        return Ok(SortedInput::Memory(buf.into_iter()));
    }
    if !buf.is_empty() {
        spill_run(&mut buf, keys, config, seq, runs.len(), &mut runs)?;
    }
    Ok(SortedInput::Merge(MergeCursor::new(runs, keys)?))
}

/// A forward cursor over the fully-sorted rows: either an in-memory buffer (input fit the budget) or
/// a k-way merge of on-disk runs.
pub(super) enum SortedInput<'a> {
    Memory(std::vec::IntoIter<Row>),
    Merge(MergeCursor<'a>),
}

impl SortedInput<'_> {
    /// The next sorted row, or `Ok(None)` at end.
    ///
    /// # Errors
    /// Propagates spill-file read / key-evaluation errors from the merge.
    pub(super) fn try_next(&mut self) -> Result<Option<Row>, Error> {
        match self {
            Self::Memory(it) => Ok(it.next()),
            Self::Merge(cursor) => cursor.try_next(),
        }
    }
}

/// Lazy k-way merge of sorted on-disk runs via a min-heap over the run heads.
pub(super) struct MergeCursor<'a> {
    runs: Vec<super::spill::SpillReader>,
    heap: BinaryHeap<Reverse<Head>>,
    keys: &'a [OrderByKey],
}

impl<'a> MergeCursor<'a> {
    fn new(
        mut runs: Vec<super::spill::SpillReader>,
        keys: &'a [OrderByKey],
    ) -> Result<Self, Error> {
        let mut heap = BinaryHeap::new();
        for run in 0..runs.len() {
            if let Some(reader) = runs.get_mut(run)
                && let Some(row) = reader.read_row()?
            {
                heap.push(Reverse(Head {
                    keys: eval_keys(keys, &row)?,
                    run,
                    row,
                }));
            }
        }
        Ok(Self { runs, heap, keys })
    }

    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        crate::cancel::check()?;
        let Some(Reverse(head)) = self.heap.pop() else {
            return Ok(None);
        };
        let run = head.run;
        if let Some(reader) = self.runs.get_mut(run)
            && let Some(row) = reader.read_row()?
        {
            self.heap.push(Reverse(Head {
                keys: eval_keys(self.keys, &row)?,
                run,
                row,
            }));
        }
        Ok(Some(head.row))
    }
}

/// Sort `buf` in place and write it out as one on-disk run, then clear `buf`.
fn spill_run(
    buf: &mut Vec<Row>,
    keys: &[OrderByKey],
    config: &super::spill::SpillConfig,
    seq: u64,
    run: usize,
    runs: &mut Vec<super::spill::SpillReader>,
) -> Result<(), Error> {
    sort_rows(buf, keys)?;
    // Include the process id: `seq` is a process-local counter, so two NusaDB processes (or two test
    // binaries) sharing one spill directory would otherwise collide on the same file name.
    let path = config.dir.join(format!(
        "nusadb-spill-sort-{}-{seq}-{run}.tmp",
        std::process::id()
    ));
    let mut writer = super::spill::SpillWriter::create(path)?;
    for row in buf.drain(..) {
        writer.write_row(&row)?;
    }
    runs.push(writer.into_reader()?);
    Ok(())
}

fn eval_keys(keys: &[OrderByKey], row: &Row) -> Result<Vec<OrderedKey>, Error> {
    keys.iter()
        .map(|k| {
            Ok(OrderedKey {
                value: eval::eval(&k.expr, row)?,
                ascending: k.ascending,
                nulls: k.nulls,
            })
        })
        .collect()
}

/// A run's current head row, decorated with its evaluated sort keys for ordering.
struct Head {
    keys: Vec<OrderedKey>,
    run: usize,
    row: Row,
}

impl PartialEq for Head {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
    }
}
impl Eq for Head {}
impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Head {
    fn cmp(&self, other: &Self) -> Ordering {
        self.keys.cmp(&other.keys)
    }
}

/// One sort-key value carrying its column's `ASC`/`NULLS` directive, so a lexicographic compare of
/// `Vec<OrderedKey>` reproduces [`compare_order_key`](crate::executor::eval::compare_order_key).
struct OrderedKey {
    value: ast::Value,
    ascending: bool,
    nulls: ast::NullOrdering,
}

impl PartialEq for OrderedKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for OrderedKey {}
impl PartialOrd for OrderedKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedKey {
    fn cmp(&self, other: &Self) -> Ordering {
        eval::compare_order_key(&self.value, &other.value, self.ascending, self.nulls)
    }
}
