//! The joins `UPDATE ... FROM`, `DELETE ... USING` and `MERGE` run between their target table and
//! their source, under the work-memory budget.
//!
//! A source that fits the budget is held in memory and keyed by the condition's equalities, as it
//! always was. A larger one (given a spill directory) is written to disk. With equalities to key
//! on, the source and then the target are split into partitions by a hash of their key, and each
//! partition is joined on its own: its source rows are held and keyed when they fit, split again
//! (up to [`MAX_DEPTH`] levels) when they do not, and read from disk for every target row when one
//! key alone is larger than the budget. With no equality, every target row reads the whole source
//! from disk.
//!
//! Each partition keeps its rows in the order they arrived, and rows with equal keys always share a
//! partition, so the source rows a target row is offered are exactly the ones the in-memory path
//! offers, in the same order: the first match an `UPDATE` uses is the same. Only the order the
//! target rows are visited in changes, so the order of `RETURNING` rows and of row triggers, which
//! SQL leaves unspecified, may differ from an unspilled run.

#![allow(clippy::wildcard_imports)]

use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::join::{KeyAtom, KeySide, key_atoms};
use super::scan::TargetRows;
use super::spill::{MemBudget, SharedSpill, SpillConfig, SpillCursor, SpillWriter};
use super::stream::RowSource;
use super::*;
use crate::planner::HashKey;

/// Monotonic id for join spill file names (process-local uniqueness; not persisted).
static JOIN_SEQ: AtomicU64 = AtomicU64::new(0);

/// How many partitions one split makes.
const FAN_OUT: usize = 32;

/// How many times a partition still larger than the budget is split again.
const MAX_DEPTH: u32 = 3;

/// Rows indexed by their values for a predicate's equi-keys (see [`crate::planner::equi_keys`]):
/// the rows a predicate over `left ++ right` can match for one probe row, in their original
/// order, so a "first match" or "any match" loop over them sees exactly the rows a full scan would
/// find matching, in the same order.
pub(super) struct KeyedRows {
    keys: Vec<crate::planner::HashKey>,
    map: HashMap<Vec<super::join::KeyAtom>, Vec<usize>>,
}

impl KeyedRows {
    /// Index `rows` (the right side, `left_width` columns after the left one), or `None` when
    /// `predicate` has no usable equi-key.
    pub(super) fn right(
        predicate: Option<&TypedExpr>,
        rows: &[Row],
        left_width: usize,
    ) -> Result<Option<Self>, Error> {
        let keys = predicate.map_or_else(Vec::new, |p| crate::planner::equi_keys(p, left_width));
        if keys.is_empty() {
            return Ok(None);
        }
        Self::with_keys(keys, rows, left_width).map(Some)
    }

    /// Index `rows` (the right side) by `keys`.
    pub(super) fn with_keys(
        keys: Vec<HashKey>,
        rows: &[Row],
        left_width: usize,
    ) -> Result<Self, Error> {
        let mut map: HashMap<Vec<super::join::KeyAtom>, Vec<usize>> = HashMap::new();
        let mut padded: Row = vec![ast::Value::Null; left_width];
        for (index, row) in rows.iter().enumerate() {
            padded.truncate(left_width);
            padded.extend_from_slice(row);
            if let Some(key) = super::join::key_atoms(&keys, &padded, super::join::KeySide::Right)?
            {
                map.entry(key).or_default().push(index);
            }
        }
        Ok(Self { keys, map })
    }

    /// The indexed right rows a left `row` can match, in order.
    pub(super) fn for_left(&self, row: &Row) -> Result<&[usize], Error> {
        Ok(
            super::join::key_atoms(&self.keys, row, super::join::KeySide::Left)?
                .and_then(|key| self.map.get(&key))
                .map_or(&[][..], Vec::as_slice),
        )
    }
}

/// A join's source rows: in memory, or on disk once they outgrew the budget.
pub(super) enum JoinSource {
    Memory(Vec<Row>),
    Spilled(SharedSpill),
}

impl JoinSource {
    /// Read `rows`, keeping them in memory while they fit the spill threshold and writing them all
    /// to disk once they do not. Without a spill directory they are always kept in memory.
    ///
    /// # Errors
    /// Propagates source and spill-file errors.
    pub(super) fn load(rows: &mut dyn RowSource) -> Result<Self, Error> {
        let config = spill::spill_config();
        let mut budget = MemBudget::new(config.as_ref().map_or(0, |c| c.threshold_bytes));
        let mut held = Vec::new();
        let overflow = loop {
            match rows.try_next()? {
                Some(row) if budget.admit(&row) => held.push(row),
                Some(row) => break row,
                None => {
                    // Without a spill directory the source must fit `work_mem`, as any other
                    // stage must.
                    if config.is_none() {
                        super::ops::enforce_work_mem(&held)?;
                    }
                    return Ok(Self::Memory(held));
                },
            }
        };
        let Some(config) = config else {
            // A zero limit admits every row, so this is not reached; keep the rows regardless.
            held.push(overflow);
            while let Some(row) = rows.try_next()? {
                held.push(row);
            }
            super::ops::enforce_work_mem(&held)?;
            return Ok(Self::Memory(held));
        };
        let mut writer = spill_file(&config)?;
        for row in held.iter().chain(std::iter::once(&overflow)) {
            writer.write_row(row)?;
        }
        drop(held);
        let mut written = 0usize;
        while let Some(row) = rows.try_next()? {
            writer.write_row(&row)?;
            written += 1;
            if written.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
        }
        Ok(Self::Spilled(writer.into_shared()?))
    }
}

/// A new spill file for a join.
fn spill_file(config: &SpillConfig) -> Result<SpillWriter, Error> {
    let seq = JOIN_SEQ.fetch_add(1, Ordering::Relaxed);
    SpillWriter::create(config.dir.join(format!(
        "nusadb-spill-join-{}-{seq}.tmp",
        std::process::id()
    )))
}

/// The source rows offered to one target row, in source order.
pub(super) trait Candidates {
    fn next_row(&mut self) -> Result<Option<Cow<'_, Row>>, Error>;
}

/// Candidates held in memory: the rows at `picks`, or every row when there is no key.
struct Listed<'r> {
    rows: &'r [Row],
    picks: Option<&'r [usize]>,
    at: usize,
}

impl Candidates for Listed<'_> {
    fn next_row(&mut self) -> Result<Option<Cow<'_, Row>>, Error> {
        let index = match self.picks {
            Some(picks) => match picks.get(self.at) {
                Some(&i) => i,
                None => return Ok(None),
            },
            None => self.at,
        };
        self.at += 1;
        Ok(self.rows.get(index).map(Cow::Borrowed))
    }
}

/// Candidates read from a spill file, one pass per target row.
struct FromDisk {
    cursor: SpillCursor,
    read: usize,
}

impl FromDisk {
    const fn new(cursor: SpillCursor) -> Self {
        Self { cursor, read: 0 }
    }
}

impl Candidates for FromDisk {
    fn next_row(&mut self) -> Result<Option<Cow<'_, Row>>, Error> {
        self.read += 1;
        if self.read.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        Ok(self.cursor.read_row()?.map(Cow::Owned))
    }
}

/// No candidates (a target row whose key is `NULL` matches nothing).
struct NoCandidates;

impl Candidates for NoCandidates {
    fn next_row(&mut self) -> Result<Option<Cow<'_, Row>>, Error> {
        Ok(None)
    }
}

/// What [`join_each`] does with one target row and its candidates.
pub(super) type EachTarget<'e> = dyn FnMut(Tid, Row, &mut dyn Candidates) -> Result<(), Error> + 'e;

/// Call `each` once per target row with the source rows that `predicate` (over
/// `target ++ source`, the target `left_width` columns wide) can match, in source order.
///
/// # Errors
/// Propagates scan, spill-file and evaluation errors, and any error `each` returns.
pub(super) fn join_each(
    mut targets: TargetRows,
    source: JoinSource,
    predicate: Option<&TypedExpr>,
    left_width: usize,
    each: &mut EachTarget<'_>,
) -> Result<(), Error> {
    let file = match source {
        JoinSource::Memory(rows) => {
            let index = KeyedRows::right(predicate, &rows, left_width)?;
            while let Some((tid, row)) = targets.try_next()? {
                let picks = match &index {
                    Some(index) => Some(index.for_left(&row)?),
                    None => None,
                };
                each(
                    tid,
                    row,
                    &mut Listed {
                        rows: &rows,
                        picks,
                        at: 0,
                    },
                )?;
            }
            return Ok(());
        },
        JoinSource::Spilled(file) => file,
    };
    let keys = predicate.map_or_else(Vec::new, |p| crate::planner::equi_keys(p, left_width));
    let config = spill::spill_config();
    let (false, Some(config)) = (keys.is_empty(), config) else {
        while let Some((tid, row)) = targets.try_next()? {
            each(tid, row, &mut FromDisk::new(file.cursor()?))?;
        }
        return Ok(());
    };
    let split = Split {
        keys: &keys,
        left_width,
        config: &config,
    };
    let parts = split.split(
        &mut FileRows(file.cursor()?),
        &mut Tagged(targets),
        0,
        &mut |_| Ok(()),
        &mut |tagged| {
            let (tid, row) = untag(tagged)?;
            each(tid, row, &mut NoCandidates)
        },
    )?;
    drop(file);
    split.each_partition(parts, &mut |part| match part {
        PartitionRows::Held { source, mut target } => {
            let index = KeyedRows::with_keys(keys.clone(), &source, left_width)?;
            while let Some(tagged) = target.read_row()? {
                crate::cancel::check()?;
                let (tid, row) = untag(tagged)?;
                let picks = index.for_left(&row)?;
                each(
                    tid,
                    row,
                    &mut Listed {
                        rows: &source,
                        picks: Some(picks),
                        at: 0,
                    },
                )?;
            }
            Ok(())
        },
        PartitionRows::OnDisk { source, target } => {
            let mut target = target.cursor()?;
            while let Some(tagged) = target.read_row()? {
                crate::cancel::check()?;
                let (tid, row) = untag(tagged)?;
                each(tid, row, &mut FromDisk::new(source.cursor()?))?;
            }
            Ok(())
        },
    })
}

/// One partition of both sides, as spill files.
pub(super) struct Partition {
    source: SharedSpill,
    target: SharedSpill,
    depth: u32,
}

/// A partition handed to the caller: its source rows held in memory, or (one key larger than the
/// budget) left on disk.
pub(super) enum PartitionRows {
    Held {
        source: Vec<Row>,
        target: SpillCursor,
    },
    OnDisk {
        source: SharedSpill,
        target: SharedSpill,
    },
}

/// Splits a join's rows into partitions by a hash of their key.
pub(super) struct Split<'k> {
    pub(super) keys: &'k [HashKey],
    pub(super) left_width: usize,
    pub(super) config: &'k SpillConfig,
}

impl Split<'_> {
    /// Write `source` and `target` (tagged target rows, see [`Tagged`]) to [`FAN_OUT`] partitions
    /// at `depth`. A row whose key is `NULL` goes to `unkeyed_source` / `unkeyed_target` instead.
    pub(super) fn split(
        &self,
        source: &mut dyn RowSource,
        target: &mut dyn RowSource,
        depth: u32,
        unkeyed_source: &mut dyn FnMut(Row) -> Result<(), Error>,
        unkeyed_target: &mut dyn FnMut(Row) -> Result<(), Error>,
    ) -> Result<Vec<Partition>, Error> {
        let mut sources = (0..FAN_OUT)
            .map(|_| spill_file(self.config))
            .collect::<Result<Vec<_>, _>>()?;
        let mut padded: Row = Vec::new();
        let mut read = 0usize;
        while let Some(row) = source.try_next()? {
            padded.clear();
            padded.resize(self.left_width, ast::Value::Null);
            padded.extend_from_slice(&row);
            match key_atoms(self.keys, &padded, KeySide::Right)? {
                Some(key) => {
                    if let Some(file) = sources.get_mut(bucket(&key, depth)) {
                        file.write_row(&row)?;
                    }
                },
                None => unkeyed_source(row)?,
            }
            read += 1;
            if read.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
        }
        let mut targets = (0..FAN_OUT)
            .map(|_| spill_file(self.config))
            .collect::<Result<Vec<_>, _>>()?;
        while let Some(tagged) = target.try_next()? {
            let key = match tagged.get(TAG..) {
                Some(row) => key_atoms(self.keys, &row.to_vec(), KeySide::Left)?,
                None => None,
            };
            match key {
                Some(key) => {
                    if let Some(file) = targets.get_mut(bucket(&key, depth)) {
                        file.write_row(&tagged)?;
                    }
                },
                None => unkeyed_target(tagged)?,
            }
            read += 1;
            if read.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
        }
        sources
            .into_iter()
            .zip(targets)
            .map(|(source, target)| {
                Ok(Partition {
                    source: source.into_shared()?,
                    target: target.into_shared()?,
                    depth: depth + 1,
                })
            })
            .collect()
    }

    /// Hand every partition to `handle`, splitting again one whose source rows outgrow the budget.
    pub(super) fn each_partition(
        &self,
        mut work: Vec<Partition>,
        handle: &mut dyn FnMut(PartitionRows) -> Result<(), Error>,
    ) -> Result<(), Error> {
        while let Some(part) = work.pop() {
            crate::cancel::check()?;
            let mut budget = MemBudget::new(self.config.threshold_bytes);
            let mut cursor = part.source.cursor()?;
            let mut held = Vec::new();
            let mut fits = true;
            while let Some(row) = cursor.read_row()? {
                if !budget.admit(&row) {
                    fits = false;
                    break;
                }
                held.push(row);
            }
            if fits {
                handle(PartitionRows::Held {
                    source: held,
                    target: part.target.cursor()?,
                })?;
                continue;
            }
            drop(held);
            if part.depth < MAX_DEPTH {
                // Every row here has a key, so nothing reaches the unkeyed callbacks.
                work.extend(self.split(
                    &mut FileRows(part.source.cursor()?),
                    &mut FileRows(part.target.cursor()?),
                    part.depth,
                    &mut |_| Ok(()),
                    &mut |_| Ok(()),
                )?);
                continue;
            }
            handle(PartitionRows::OnDisk {
                source: part.source,
                target: part.target,
            })?;
        }
        Ok(())
    }
}

/// The partition of `key` at `depth` (each depth hashes differently, so a split partition spreads).
fn bucket(key: &[KeyAtom], depth: u32) -> usize {
    let mut hasher = DefaultHasher::new();
    depth.hash(&mut hasher);
    key.hash(&mut hasher);
    usize::try_from(hasher.finish() % FAN_OUT as u64).unwrap_or(0)
}

/// Columns a tagged target row carries before the row itself: its row address.
pub(super) const TAG: usize = 2;

/// Target rows tagged with their address, so they can be written to a spill file.
pub(super) struct Tagged(pub(super) TargetRows);

impl RowSource for Tagged {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let Some((tid, row)) = self.0.try_next()? else {
            return Ok(None);
        };
        let mut tagged = Vec::with_capacity(TAG + row.len());
        tagged.push(ast::Value::Int(i64::from_ne_bytes(
            tid.page.0.to_ne_bytes(),
        )));
        tagged.push(ast::Value::Int(i64::from(tid.slot.0)));
        tagged.extend(row);
        Ok(Some(tagged))
    }
}

/// A tagged target row's address and row.
pub(super) fn untag(mut tagged: Row) -> Result<(Tid, Row), Error> {
    let lost = || Error::Internal("a spilled target row lost its address".to_owned());
    let row = tagged.split_off(TAG);
    let (Some(ast::Value::Int(page)), Some(ast::Value::Int(slot))) =
        (tagged.first(), tagged.get(1))
    else {
        return Err(lost());
    };
    Ok((
        Tid {
            page: nusadb_core::PageId(u64::from_ne_bytes(page.to_ne_bytes())),
            slot: nusadb_core::SlotIdx(u16::try_from(*slot).map_err(|_| lost())?),
        },
        row,
    ))
}

/// Reads a spill file as a row source.
pub(super) struct FileRows(pub(super) SpillCursor);

impl RowSource for FileRows {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        self.0.read_row()
    }
}

/// A table's rows as a join source.
pub(super) struct TableRows(pub(super) TargetRows);

impl RowSource for TableRows {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        Ok(self.0.try_next()?.map(|(_, row)| row))
    }
}

/// What [`merge_each`] does with one source row and the first target row it matched, if any.
pub(super) type EachSource<'e> = dyn FnMut(Row, Option<(Tid, Row)>) -> Result<(), Error> + 'e;

/// What [`merge_each`] does with a target row no source row matched.
pub(super) type EachUnmatched<'e> = dyn FnMut(Tid, Row) -> Result<(), Error> + 'e;

/// The condition and callbacks of one `MERGE`.
pub(super) struct Merge<'m, 'e> {
    /// The `ON` condition over `target ++ source`.
    pub(super) on: &'m TypedExpr,
    /// The target's width.
    pub(super) left_width: usize,
    /// Whether a `WHEN NOT MATCHED BY SOURCE` clause needs the unmatched target rows.
    pub(super) by_source: bool,
    pub(super) each_source: &'m mut EachSource<'e>,
    pub(super) unmatched: &'m mut EachUnmatched<'e>,
}

impl Merge<'_, '_> {
    /// Whether `ON` holds for `target ++ source`, built in `scratch` (reused across pairs).
    fn matches(&self, scratch: &mut Row, target: &Row, source: &Row) -> Result<bool, Error> {
        scratch.clear();
        scratch.extend(target.iter().cloned());
        scratch.extend(source.iter().cloned());
        Ok(matches!(
            eval::eval(self.on, scratch)?,
            ast::Value::Bool(true)
        ))
    }
}

/// Run a `MERGE`'s join: every source row is handed to `each_source` with the first target row (in
/// scan order) its `ON` condition matches, and, when `by_source` is set, every target row no source
/// row matches is handed to `unmatched` as the target is read. Over a source held in memory the
/// source rows come in their order once every target row has been read; over a spilled source the
/// same happens one partition at a time.
///
/// # Errors
/// Propagates scan, spill-file and evaluation errors, and any error a callback returns.
pub(super) fn merge_each(
    mut targets: TargetRows,
    source: JoinSource,
    merge: &mut Merge<'_, '_>,
) -> Result<(), Error> {
    let file = match source {
        JoinSource::Memory(rows) => {
            let index = KeyedRows::right(Some(merge.on), &rows, merge.left_width)?;
            return merge_held(rows, index.as_ref(), &mut || targets.try_next(), merge);
        },
        JoinSource::Spilled(file) => file,
    };
    let Some(config) = spill::spill_config() else {
        return Err(Error::Internal(
            "a MERGE source was spilled without a spill directory".to_owned(),
        ));
    };
    let keys = crate::planner::equi_keys(merge.on, merge.left_width);
    if keys.is_empty() {
        // No equality to key on: the targets go to disk too and each side reads the other.
        let mut writer = spill_file(&config)?;
        let mut tagged = Tagged(targets);
        while let Some(row) = tagged.try_next()? {
            writer.write_row(&row)?;
        }
        return merge_on_disk(&file, &writer.into_shared()?, merge);
    }
    let split = Split {
        keys: &keys,
        left_width: merge.left_width,
        config: &config,
    };
    let by_source = merge.by_source;
    let parts = {
        let each_source = &mut *merge.each_source;
        let unmatched = &mut *merge.unmatched;
        // A row whose key is `NULL` matches nothing, so it is settled at once.
        split.split(
            &mut FileRows(file.cursor()?),
            &mut Tagged(targets),
            0,
            &mut |row| each_source(row, None),
            &mut |tagged| {
                if by_source {
                    let (tid, row) = untag(tagged)?;
                    unmatched(tid, row)?;
                }
                Ok(())
            },
        )?
    };
    drop(file);
    split.each_partition(parts, &mut |part| match part {
        PartitionRows::Held { source, mut target } => {
            let index = KeyedRows::with_keys(keys.clone(), &source, merge.left_width)?;
            merge_held(
                source,
                Some(&index),
                &mut || target.read_row()?.map(untag).transpose(),
                merge,
            )
        },
        PartitionRows::OnDisk { source, target } => merge_on_disk(&source, &target, merge),
    })
}

/// [`merge_each`] over source rows held in memory (keyed by `index` when the condition has
/// equalities) and target rows from `next_target`.
fn merge_held(
    source: Vec<Row>,
    index: Option<&KeyedRows>,
    next_target: &mut dyn FnMut() -> Result<Option<(Tid, Row)>, Error>,
    merge: &mut Merge<'_, '_>,
) -> Result<(), Error> {
    let mut hits: Vec<Option<(Tid, Row)>> = vec![None; source.len()];
    let mut scratch = Row::new();
    while let Some((tid, row)) = next_target()? {
        crate::cancel::check()?;
        let picks = match index {
            Some(index) => Some(index.for_left(&row)?),
            None => None,
        };
        let count = picks.map_or(source.len(), <[usize]>::len);
        let mut matched = false;
        for n in 0..count {
            if n > 0 && n.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            let i = match picks {
                Some(picks) => match picks.get(n) {
                    Some(&i) => i,
                    None => break,
                },
                None => n,
            };
            let Some(srow) = source.get(i) else {
                break;
            };
            // A source row that already has its first target needs this pair only to learn
            // whether the target matched anything, and only when that is asked.
            let settled = hits.get(i).is_some_and(Option::is_some);
            if settled && (matched || !merge.by_source) {
                continue;
            }
            if merge.matches(&mut scratch, &row, srow)? {
                matched = true;
                if let Some(hit @ None) = hits.get_mut(i) {
                    *hit = Some((tid, row.clone()));
                }
            }
        }
        if merge.by_source && !matched {
            (merge.unmatched)(tid, row)?;
        }
    }
    for (srow, hit) in source.into_iter().zip(hits) {
        (merge.each_source)(srow, hit)?;
    }
    Ok(())
}

/// [`merge_each`] with both sides on disk (`target` holds tagged rows): each source row reads the
/// targets up to its first match, and each target row reads the sources until one matches.
fn merge_on_disk(
    source: &SharedSpill,
    target: &SharedSpill,
    merge: &mut Merge<'_, '_>,
) -> Result<(), Error> {
    let mut scratch = Row::new();
    let mut sources = source.cursor()?;
    while let Some(srow) = sources.read_row()? {
        crate::cancel::check()?;
        let mut targets = target.cursor()?;
        let mut hit = None;
        let mut read = 0usize;
        while let Some(tagged) = targets.read_row()? {
            read += 1;
            if read.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            let (tid, row) = untag(tagged)?;
            if merge.matches(&mut scratch, &row, &srow)? {
                hit = Some((tid, row));
                break;
            }
        }
        (merge.each_source)(srow, hit)?;
    }
    if !merge.by_source {
        return Ok(());
    }
    let mut targets = target.cursor()?;
    while let Some(tagged) = targets.read_row()? {
        crate::cancel::check()?;
        let (tid, row) = untag(tagged)?;
        let mut sources = source.cursor()?;
        let mut matched = false;
        let mut read = 0usize;
        while let Some(srow) = sources.read_row()? {
            read += 1;
            if read.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            if merge.matches(&mut scratch, &row, &srow)? {
                matched = true;
                break;
            }
        }
        if !matched {
            (merge.unmatched)(tid, row)?;
        }
    }
    Ok(())
}
