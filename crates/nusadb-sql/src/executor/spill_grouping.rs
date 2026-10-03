//! `DISTINCT ON` and grouping sets (`ROLLUP` / `CUBE` / `GROUPING SETS`) under a memory budget.
//!
//! Both sort their input externally with each row's input position as the last key, so the rows
//! that tie on the grouping keys keep their input order: `DISTINCT ON` keeps exactly the row the
//! in-memory path keeps (the first of its key), and every aggregate folds a group's rows in the
//! order the in-memory path folds them. Only the order the groups leave in differs, which SQL
//! leaves unspecified unless an `ORDER BY` above sorts them. Each grouping set sorts the spilled
//! input once, so `CUBE` over `n` keys costs `2^n` external sorts.

#![allow(clippy::wildcard_imports)]

use std::borrow::Cow;

use super::spill::{MemBudget, SharedSpill, SpillConfig, SpillCursor, SpillWriter};
use super::spill_window::{Chain, Unnumbered, numbered, position_key};
use super::stream::{Materialized, RowSource, SortedSource};
use super::*;
use crate::planner::OrderByKey;

/// Monotonic id for grouping-set spill file names (process-local uniqueness; not persisted).
static GROUPING_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// An ascending, default-NULLs sort key on `expr`.
fn ascending(expr: &TypedExpr) -> OrderByKey {
    OrderByKey {
        expr: expr.clone(),
        ascending: true,
        nulls: ast::NullOrdering::Default,
    }
}

/// `DISTINCT ON (keys)` over `input` as a stream: the first row of each run of equal `keys`.
///
/// When `input` is a sort whose leading keys are exactly `keys` (the usual form, so the first row of
/// each key is the one the `ORDER BY` picks), the runs are already adjacent and the rows stream
/// through in the sort's order. Otherwise the input is sorted by `keys` and then its position, so
/// each run starts with the row the input held first, and the rows kept are sorted back into input
/// position: the result leaves in the order the in-memory path gives, which an `ORDER BY` below
/// (one that does not lead with the keys) relies on.
///
/// # Errors
/// Propagates source, spill-file and evaluation errors.
pub(super) fn distinct_on_source<'a>(
    input: &'a PhysicalOperator,
    keys: &'a [TypedExpr],
    config: &SpillConfig,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
) -> Result<Box<dyn RowSource + 'a>, Error> {
    if sorted_on(input, keys) {
        return Ok(Box::new(FirstOfRun {
            inner: super::stream::stream_op(input, engine, txn)?,
            keys,
            prev: None,
        }));
    }
    let mut source = numbered(super::stream::stream_op(input, engine, txn)?);
    let Some(first) = source.try_next()? else {
        return Ok(Box::new(Materialized(Vec::new().into_iter())));
    };
    let width = first.len();
    let mut order: Vec<OrderByKey> = keys.iter().map(ascending).collect();
    order.push(position_key(width));
    let mut all = Chain {
        first: Some(first),
        rest: source,
    };
    let by_key = super::spill_sort::sorted_rows(&mut all, &order, config)?;
    let mut kept = FirstOfRun {
        inner: Box::new(SortedSource(by_key)),
        keys,
        prev: None,
    };
    let in_input_order = super::spill_sort::sorted_rows(&mut kept, &[position_key(width)], config)?;
    Ok(Box::new(Unnumbered(Box::new(SortedSource(in_input_order)))))
}

/// Whether `op` is a sort whose leading keys are exactly the expressions in `keys`.
fn sorted_on(op: &PhysicalOperator, keys: &[TypedExpr]) -> bool {
    let PhysicalOperator::Sort { keys: order, .. } = op else {
        return false;
    };
    let Some(leading) = order.get(..keys.len()) else {
        return false;
    };
    !keys.is_empty()
        && leading.iter().all(|k| keys.contains(&k.expr))
        && keys
            .iter()
            .all(|key| leading.iter().any(|k| &k.expr == key))
}

/// The first row of each run of rows with equal `keys`.
struct FirstOfRun<'a> {
    inner: Box<dyn RowSource + 'a>,
    keys: &'a [TypedExpr],
    prev: Option<Vec<ast::Value>>,
}

impl RowSource for FirstOfRun<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        while let Some(row) = self.inner.try_next()? {
            let key = self
                .keys
                .iter()
                .map(|k| eval::eval(k, &row))
                .collect::<Result<Vec<_>, _>>()?;
            if self
                .prev
                .as_ref()
                .is_some_and(|prev| group_keys_equal(prev, &key))
            {
                continue;
            }
            self.prev = Some(key);
            return Ok(Some(row));
        }
        Ok(None)
    }
}

/// Grouping sets over `input`. An input that fits the budget is folded in memory exactly as
/// without spill; a larger one is written once to a spill file and each set is folded from its
/// own sorted pass over that file.
///
/// # Errors
/// Propagates source, spill-file and aggregate errors.
pub(super) fn grouping_sets_source<'a>(
    input: &'a PhysicalOperator,
    group_keys: &'a [TypedExpr],
    grouping_sets: &'a [Vec<usize>],
    calls: &'a [AggregateCall],
    config: &SpillConfig,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
) -> Result<Box<dyn RowSource + 'a>, Error> {
    let mut source = numbered(super::stream::stream_op(input, engine, txn)?);
    let mut budget = MemBudget::new(config.threshold_bytes);
    let mut held: Vec<Row> = Vec::new();
    let mut overflow = None;
    while let Some(row) = source.try_next()? {
        if !budget.admit(&row) {
            overflow = Some(row);
            break;
        }
        held.push(row);
    }
    let Some(overflow) = overflow else {
        // Everything fit: the in-memory fold, over the rows without their positions.
        for row in &mut held {
            row.pop();
        }
        let mut rows = Materialized(held.into_iter());
        let out = super::agg::run_grouping_sets_aggregate_streamed(
            &mut rows,
            group_keys,
            grouping_sets,
            calls,
        )?;
        return Ok(Box::new(Materialized(out.into_iter())));
    };
    let file = GROUPING_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut writer = SpillWriter::create(config.dir.join(format!(
        "nusadb-spill-grouping-{}-{file}.tmp",
        std::process::id()
    )))?;
    let width = overflow.len();
    for row in held.iter().chain(std::iter::once(&overflow)) {
        writer.write_row(row)?;
    }
    drop(held);
    let mut written = 0usize;
    while let Some(row) = source.try_next()? {
        writer.write_row(&row)?;
        written += 1;
        if written.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
    }
    Ok(Box::new(SpilledGroupingSets {
        file: writer.into_shared()?,
        width,
        group_keys,
        grouping_sets,
        calls,
        config: config.clone(),
        next_set: 0,
        current: None,
    }))
}

/// Reads a [`SharedSpill`] cursor as a row source.
struct CursorRows(SpillCursor);

impl RowSource for CursorRows {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        self.0.read_row()
    }
}

/// Grouping sets folded set by set from a spilled input.
struct SpilledGroupingSets<'a> {
    file: SharedSpill,
    /// Row width including the trailing position column.
    width: usize,
    group_keys: &'a [TypedExpr],
    grouping_sets: &'a [Vec<usize>],
    calls: &'a [AggregateCall],
    config: SpillConfig,
    next_set: usize,
    /// The set being folded, its groups, and whether it has produced a row yet.
    current: Option<(usize, super::agg::SortedGroups<'a>, bool)>,
}

impl SpilledGroupingSets<'_> {
    /// Lay a set's `[set keys ++ aggregates]` row out at full width: the set's key values in their
    /// slots, `NULL` for the keys it groups away, then the aggregates with each `GROUPING(...)`
    /// replaced by this set's bitmask.
    fn widen(&self, set: &[usize], folded: Row) -> Row {
        let mut values = folded.into_iter();
        let mut out = vec![ast::Value::Null; self.group_keys.len()];
        for (&slot, value) in set.iter().zip(values.by_ref()) {
            if let Some(cell) = out.get_mut(slot) {
                *cell = value;
            }
        }
        out.extend(values);
        for (ci, call) in self.calls.iter().enumerate() {
            if matches!(call.func, ast::AggregateFunc::Grouping)
                && let Some(cell) = out.get_mut(self.group_keys.len() + ci)
            {
                *cell = ast::Value::Int(super::agg::grouping_mask(&call.grouping_args, set));
            }
        }
        out
    }
}

impl RowSource for SpilledGroupingSets<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        loop {
            if self.current.is_none() {
                let Some(set) = self.grouping_sets.get(self.next_set) else {
                    return Ok(None);
                };
                let set_index = self.next_set;
                self.next_set += 1;
                let exprs: Vec<TypedExpr> = set
                    .iter()
                    .filter_map(|&i| self.group_keys.get(i).cloned())
                    .collect();
                let mut order: Vec<OrderByKey> = exprs.iter().map(ascending).collect();
                order.push(position_key(self.width));
                let sorted = super::spill_sort::sorted_rows(
                    &mut CursorRows(self.file.cursor()?),
                    &order,
                    &self.config,
                )?;
                let groups = super::agg::SortedGroups::over(sorted, Cow::Owned(exprs), self.calls);
                self.current = Some((set_index, groups, false));
            }
            let (set_index, groups, produced) = match self.current.as_mut() {
                Some((set_index, groups, produced)) => (*set_index, groups, produced),
                None => return Ok(None),
            };
            let set = self
                .grouping_sets
                .get(set_index)
                .map_or(&[][..], Vec::as_slice);
            if let Some(folded) = groups.try_next()? {
                *produced = true;
                return Ok(Some(self.widen(set, folded)));
            }
            let empty_total = set.is_empty() && !*produced;
            self.current = None;
            // A set that groups by nothing yields its grand total even over no rows. (Defensive:
            // only an input larger than the budget reaches this path, so its one group always
            // has rows.)
            if empty_total {
                let folded = self
                    .calls
                    .iter()
                    .map(|call| super::agg::finalize_aggregate(super::agg::Acc::default(), call))
                    .collect::<Result<Row, _>>()?;
                return Ok(Some(self.widen(set, folded)));
            }
        }
    }
}
