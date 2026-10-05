//! Window functions under a memory budget.
//!
//! Each pass handles the consecutive windows that share one `PARTITION BY` and `ORDER BY`. It sorts
//! its input externally by the partition keys, the window ordering and finally each row's position
//! in the original input, then walks the sorted rows one partition at a time. The last key makes
//! the order of rows that tie on the window ordering the same as the in-memory path's stable sort,
//! so every row gets exactly the value [`compute_window`] would give it.
//!
//! A partition that fits the budget is evaluated in memory by [`compute_window`] itself. A larger
//! one is written to a spill file and evaluated as it streams back, with independent cursors over
//! the file for the functions that look ahead or behind: the ranking and distribution functions,
//! `LAG`/`LEAD` with a constant offset, running aggregates from the partition start, any `ROWS`,
//! `RANGE` or `GROUPS` frame (holding just the frame's rows), and aggregates from the current row
//! to the partition end (computed backwards). `LAG` / `LEAD` with an offset that varies per row and
//! `FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE` over a `ROWS` or default frame read the row they need
//! from a copy of the partition readable by position, so they hold no rows at all. What still needs
//! memory past the budget is a `RANGE` / `GROUPS` frame with an offset, or a `ROWS` frame of an
//! aggregate without an exact sliding form (see [`SlidingAggregate`]), wider than the budget; those
//! fail with the budget error.
//!
//! Rows leave in partition order rather than input order; a query that orders its result has a
//! `Sort` above the window, which the planner always places there.

#![allow(clippy::wildcard_imports)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use super::spill::{
    MemBudget, RandomSpill, RandomSpillWriter, SharedSpill, SpillConfig, SpillCursor, SpillWriter,
};
use super::spill_sort::SortedInput;
use super::stream::RowSource;
use super::*;
use crate::ast::WindowFunc as W;
use crate::planner::{FrameBound, OrderByKey, WindowExpr};

/// Monotonic id for window spill file names (process-local uniqueness; not persisted).
static WINDOW_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Stream `input` with one column appended per window in `windows`, holding at most about the
/// budget of `config` in memory per partition (see the module docs).
///
/// # Errors
/// Propagates source, spill-file and evaluation errors, and fails with the budget error for a
/// frame that cannot be evaluated over a partition larger than the budget.
pub(super) fn window_source<'a>(
    input: &'a PhysicalOperator,
    windows: &'a [WindowExpr],
    config: &SpillConfig,
    engine: &'a dyn StorageEngine,
    txn: TxnId,
) -> Result<Box<dyn RowSource + 'a>, Error> {
    let mut source = numbered(super::stream::stream_op(input, engine, txn)?);
    let mut start = 0;
    while let Some(first) = windows.get(start) {
        let len = windows
            .get(start..)
            .unwrap_or_default()
            .iter()
            .take_while(|w| w.partition == first.partition && w.order == first.order)
            .count();
        let pass = windows.get(start..start + len).unwrap_or_default();
        source = Box::new(WindowPass::new(source, pass, config)?);
        start += len;
    }
    Ok(Box::new(Unnumbered(source)))
}

/// `source` with each row's position in it appended as a trailing `Int` column, the last sort key
/// that makes an external sort keep tied rows in input order.
pub(super) fn numbered<'a>(source: Box<dyn RowSource + 'a>) -> Box<dyn RowSource + 'a> {
    Box::new(Numbered {
        inner: source,
        next: 0,
    })
}

/// The sort key on the trailing position column of rows `width` wide (position included).
pub(super) const fn position_key(width: usize) -> OrderByKey {
    OrderByKey {
        expr: crate::planner::TypedExpr {
            kind: crate::planner::TypedExprKind::Column(width.saturating_sub(1)),
            ty: nusadb_core::ColumnType::BigInt,
        },
        ascending: true,
        nulls: ast::NullOrdering::Default,
    }
}

/// Appends each row's position in the input as a trailing `Int` column.
struct Numbered<'a> {
    inner: Box<dyn RowSource + 'a>,
    next: i64,
}

impl RowSource for Numbered<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let Some(mut row) = self.inner.try_next()? else {
            return Ok(None);
        };
        row.push(ast::Value::Int(self.next));
        self.next += 1;
        Ok(Some(row))
    }
}

/// Drops the trailing position column [`Numbered`] added.
pub(super) struct Unnumbered<'a>(pub(super) Box<dyn RowSource + 'a>);

impl RowSource for Unnumbered<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let Some(mut row) = self.0.try_next()? else {
            return Ok(None);
        };
        row.pop();
        Ok(Some(row))
    }
}

/// Yields `first`, then the rest of `rest`.
pub(super) struct Chain<'a> {
    pub(super) first: Option<Row>,
    pub(super) rest: Box<dyn RowSource + 'a>,
}

impl RowSource for Chain<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        match self.first.take() {
            Some(row) => Ok(Some(row)),
            None => self.rest.try_next(),
        }
    }
}

/// One pass: the windows sharing a partitioning and ordering, over input sorted for them.
struct WindowPass<'a> {
    sorted: Option<SortedInput>,
    /// The first row of the next partition, read while finding the end of the current one.
    peeked: Option<Row>,
    windows: &'a [WindowExpr],
    config: SpillConfig,
    /// Finished rows of a partition evaluated in memory.
    ready: VecDeque<Row>,
    /// A partition larger than the budget, being evaluated as it streams back from disk.
    large: Option<LargePartition<'a>>,
}

impl<'a> WindowPass<'a> {
    fn new(
        mut input: Box<dyn RowSource + 'a>,
        windows: &'a [WindowExpr],
        config: &SpillConfig,
    ) -> Result<Self, Error> {
        let mut pass = Self {
            sorted: None,
            peeked: None,
            windows,
            config: config.clone(),
            ready: VecDeque::new(),
            large: None,
        };
        let Some(first) = input.try_next()? else {
            return Ok(pass);
        };
        // Every window of the pass shares these; the last key is the input position, which the
        // previous stage left as the trailing column.
        let shared = windows.first();
        let position = first.len().saturating_sub(1);
        let mut keys: Vec<OrderByKey> = shared
            .map(|w| w.partition.as_slice())
            .unwrap_or_default()
            .iter()
            .map(|expr| OrderByKey {
                expr: expr.clone(),
                ascending: true,
                nulls: ast::NullOrdering::Default,
            })
            .collect();
        keys.extend(shared.map(|w| w.order.clone()).unwrap_or_default());
        keys.push(position_key(position + 1));
        let mut all = Chain {
            first: Some(first),
            rest: input,
        };
        pass.sorted = Some(super::spill_sort::sorted_rows(&mut all, &keys, config)?);
        Ok(pass)
    }

    fn partition_key(&self, row: &Row) -> Result<Vec<ast::Value>, Error> {
        self.windows
            .first()
            .map(|w| w.partition.as_slice())
            .unwrap_or_default()
            .iter()
            .map(|e| eval::eval(e, row))
            .collect()
    }

    fn next_sorted(&mut self) -> Result<Option<Row>, Error> {
        self.sorted.as_mut().map_or(Ok(None), SortedInput::try_next)
    }

    /// Read the next partition and either evaluate it in memory or start streaming it from disk.
    /// `false` once the input is exhausted.
    fn load_partition(&mut self) -> Result<bool, Error> {
        let first = match self.peeked.take() {
            Some(row) => row,
            None => match self.next_sorted()? {
                Some(row) => row,
                None => return Ok(false),
            },
        };
        crate::cancel::check()?;
        let key = self.partition_key(&first)?;
        let mut budget = MemBudget::new(self.config.threshold_bytes);
        budget.admit(&first);
        let mut rows = vec![first];
        while let Some(row) = self.next_sorted()? {
            if !group_keys_equal(&key, &self.partition_key(&row)?) {
                self.peeked = Some(row);
                break;
            }
            if !budget.admit(&row) {
                self.spill_partition(rows, &row, &key)?;
                return Ok(true);
            }
            rows.push(row);
        }
        // The partition fits: evaluate it exactly as the in-memory path does.
        let positions: Vec<ast::Value> = rows
            .iter_mut()
            .map(|row| row.pop().unwrap_or(ast::Value::Null))
            .collect();
        let columns = self
            .windows
            .iter()
            .map(|w| compute_window(&rows, w))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, (mut row, position)) in rows.into_iter().zip(positions).enumerate() {
            for column in &columns {
                row.push(column.get(i).cloned().unwrap_or(ast::Value::Null));
            }
            row.push(position);
            self.ready.push_back(row);
        }
        Ok(true)
    }

    /// Write the partition (`held`, then `next`, then the rest of the partition from the sorted
    /// input) to a spill file and set up its streaming evaluation.
    fn spill_partition(
        &mut self,
        held: Vec<Row>,
        next: &Row,
        key: &[ast::Value],
    ) -> Result<(), Error> {
        let file = WINDOW_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = self.config.dir.join(format!(
            "nusadb-spill-window-{}-{file}.tmp",
            std::process::id()
        ));
        let mut writer = SpillWriter::create(path)?;
        let mut len = 0usize;
        for row in held.iter().chain(std::iter::once(next)) {
            writer.write_row(row)?;
            len += 1;
        }
        drop(held);
        while let Some(row) = self.next_sorted()? {
            if !group_keys_equal(key, &self.partition_key(&row)?) {
                self.peeked = Some(row);
                break;
            }
            writer.write_row(&row)?;
            len += 1;
            if len.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
        }
        let shared = writer.into_shared()?;
        let mut random = RandomCopy {
            file: &shared,
            config: &self.config,
            copy: None,
        };
        let evaluators = self
            .windows
            .iter()
            .map(|w| evaluator(w, &mut random, len))
            .collect::<Result<Vec<_>, _>>()?;
        self.large = Some(LargePartition {
            rows: shared.cursor()?,
            next: 0,
            evaluators,
        });
        Ok(())
    }
}

impl RowSource for WindowPass<'_> {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        loop {
            if let Some(row) = self.ready.pop_front() {
                return Ok(Some(row));
            }
            if let Some(large) = self.large.as_mut() {
                if let Some(row) = large.next_row()? {
                    return Ok(Some(row));
                }
                self.large = None;
                continue;
            }
            if !self.load_partition()? {
                return Ok(None);
            }
        }
    }
}

/// A partition larger than the budget, streamed from its spill file.
struct LargePartition<'a> {
    rows: SpillCursor,
    next: usize,
    evaluators: Vec<Box<dyn Evaluator + 'a>>,
}

impl LargePartition<'_> {
    fn next_row(&mut self) -> Result<Option<Row>, Error> {
        let Some(mut row) = self.rows.read_row()? else {
            return Ok(None);
        };
        if self.next.is_multiple_of(1024) {
            crate::cancel::check()?;
        }
        let k = self.next;
        self.next += 1;
        let position = row.pop().unwrap_or(ast::Value::Null);
        let values = self
            .evaluators
            .iter_mut()
            .map(|e| e.value(k, &row))
            .collect::<Result<Vec<_>, _>>()?;
        row.extend(values);
        row.push(position);
        Ok(Some(row))
    }
}

/// Produces one window function's value for each row of a streamed partition, in order.
trait Evaluator {
    /// The value for the row at partition position `k` (called for `k = 0, 1, …`), given that row.
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error>;
}

/// The SQL name of a window function, for messages.
fn sql_name(func: &ast::WindowFunc) -> String {
    match func {
        W::RowNumber => "row_number".to_owned(),
        W::Rank => "rank".to_owned(),
        W::DenseRank => "dense_rank".to_owned(),
        W::Ntile => "ntile".to_owned(),
        W::CumeDist => "cume_dist".to_owned(),
        W::PercentRank => "percent_rank".to_owned(),
        W::Lag => "lag".to_owned(),
        W::Lead => "lead".to_owned(),
        W::FirstValue => "first_value".to_owned(),
        W::LastValue => "last_value".to_owned(),
        W::NthValue => "nth_value".to_owned(),
        W::Aggregate(func) => format!("{func:?}").to_lowercase(),
    }
}

/// The evaluator for `window` over a spilled partition of `len` rows, or the budget error when its
/// frame needs the whole partition in memory.
fn evaluator<'a>(
    window: &'a WindowExpr,
    random: &mut RandomCopy<'_>,
    len: usize,
) -> Result<Box<dyn Evaluator + 'a>, Error> {
    let file = random.file;
    let config = random.config;
    let budget = config.threshold_bytes;
    let unsupported = || {
        Error::Core(nusadb_core::Error::OutOfMemory(format!(
            "query work_mem of {budget} bytes exceeded: a window partition larger than the budget \
             cannot evaluate {} over this frame without holding the whole partition; split it \
             with PARTITION BY, use a ROWS frame, or raise work_mem (SET work_mem / --work-mem)",
            sql_name(&window.func)
        )))
    };
    match &window.func {
        W::RowNumber | W::Rank | W::DenseRank | W::PercentRank => Ok(Box::new(Ranking {
            window,
            len,
            prev: None,
            rank: 0,
            dense: 0,
        })),
        W::CumeDist => Ok(Box::new(CumeDist {
            peers: Peers::new(&window.order, file)?,
            len,
        })),
        W::Ntile => Ok(Box::new(Ntile {
            window,
            len,
            buckets: None,
        })),
        W::Lag | W::Lead => {
            let offset = match window.args.get(1) {
                None => 1,
                Some(expr) => match &expr.kind {
                    crate::planner::TypedExprKind::Literal(ast::Value::Int(n)) => *n,
                    crate::planner::TypedExprKind::Literal(ast::Value::Null) => 1,
                    // An offset that varies per row reads its target by position.
                    _ => return Ok(Box::new(Positional::shift(window, random.rows()?, len))),
                },
            };
            let delta = if matches!(window.func, W::Lag) {
                offset.checked_neg().unwrap_or(0)
            } else {
                offset
            };
            Ok(Box::new(Shift::new(window, file, delta)?))
        },
        W::FirstValue | W::LastValue | W::NthValue => {
            frame_navigation(window, random, len)?.ok_or_else(unsupported)
        },
        W::Aggregate(_) => {
            let exclusion = window
                .frame
                .as_ref()
                .is_some_and(|f| !matches!(f.exclude, ast::WindowExclude::NoOthers));
            match frame_shape(window) {
                Some((Lo::Start, hi)) if !exclusion => {
                    let call = window_aggregate_call(window).ok_or_else(|| {
                        Error::Internal(
                            "window aggregate reached without a prepared call".to_owned(),
                        )
                    })?;
                    Ok(Box::new(RunningAggregate::new(
                        window, call, file, len, hi,
                    )?))
                },
                _ => {
                    if let Some(call) = window_aggregate_call(window)
                        && let Some(reversed) = Reversed::try_new(window, &call, file, config)?
                    {
                        return Ok(Box::new(reversed));
                    }
                    if let Some(call) = window_aggregate_call(window)
                        && let Some(sliding) =
                            SlidingAggregate::new(window, call, file, len, budget)?
                    {
                        return Ok(Box::new(sliding));
                    }
                    if let Some(sliding) =
                        Sliding::new(window, window_aggregate_call(window), file, len, budget)?
                    {
                        return Ok(Box::new(sliding));
                    }
                    Ok(Box::new(
                        PeerSliding::new(window, window_aggregate_call(window), file, len, budget)?
                            .ok_or_else(unsupported)?,
                    ))
                },
            }
        },
    }
}

/// Monotonic id for the by-position copies of spilled partitions (process-local uniqueness).
static RANDOM_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A spilled partition, and a copy of it readable by position made the first time a window needs
/// one; every window of the partition shares that copy.
struct RandomCopy<'f> {
    file: &'f SharedSpill,
    config: &'f SpillConfig,
    copy: Option<Rc<RefCell<RandomSpill>>>,
}

impl RandomCopy<'_> {
    fn rows(&mut self) -> Result<Rc<RefCell<RandomSpill>>, Error> {
        if let Some(copy) = &self.copy {
            return Ok(Rc::clone(copy));
        }
        let seq = RANDOM_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut writer = RandomSpillWriter::create(
            &self.config.dir,
            &format!("nusadb-spill-window-{}-random-{seq}", std::process::id()),
        )?;
        let mut cursor = self.file.cursor()?;
        let mut written = 0usize;
        while let Some(row) = cursor.read_row()? {
            writer.write_row(&row)?;
            written += 1;
            if written.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
        }
        let copy = Rc::new(RefCell::new(writer.finish()?));
        self.copy = Some(Rc::clone(&copy));
        Ok(copy)
    }
}

/// Where a frame bound lies for the row at position `k`, as [`frame_bounds`] places it.
#[derive(Clone, Copy)]
enum Edge {
    First,
    Last,
    /// The current row (`ROWS`), or for a `RANGE` / `GROUPS` frame the current row's first peer
    /// (as a start) or last peer (as an end).
    Current,
    /// `n` rows before the current row.
    Before(u64),
    /// `n` rows after the current row.
    After(u64),
}

/// A frame whose bounds are positions computable row by row: any `ROWS` frame, the default frame,
/// and a `RANGE` / `GROUPS` frame bounded only by the partition ends and the current row.
#[derive(Clone, Copy)]
struct Span {
    start: Edge,
    end: Edge,
    peer_based: bool,
}

impl Span {
    fn of(window: &WindowExpr) -> Option<Self> {
        let Some(frame) = &window.frame else {
            let end = if window.order.is_empty() {
                Edge::Last
            } else {
                Edge::Current
            };
            return Some(Self {
                start: Edge::First,
                end,
                peer_based: true,
            });
        };
        let edge = |bound: &FrameBound| match bound {
            FrameBound::UnboundedPreceding => Some(Edge::First),
            FrameBound::UnboundedFollowing => Some(Edge::Last),
            FrameBound::CurrentRow => Some(Edge::Current),
            FrameBound::Preceding(n) if !frame.peer_based => Some(Edge::Before(*n)),
            FrameBound::Following(n) if !frame.peer_based => Some(Edge::After(*n)),
            _ => None,
        };
        Some(Self {
            start: edge(&frame.start)?,
            end: edge(&frame.end)?,
            peer_based: frame.peer_based,
        })
    }

    const fn needs_peers(self) -> bool {
        self.peer_based
            && (matches!(self.start, Edge::Current) || matches!(self.end, Edge::Current))
    }
}

/// Window functions that read one row chosen per current row, through a copy of the partition
/// readable by position, so they hold no rows whatever the offset or the frame width: `LAG` /
/// `LEAD` whose offset varies per row, and `FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE` over a
/// [`Span`] frame.
struct Positional<'a> {
    window: &'a WindowExpr,
    rows: Rc<RefCell<RandomSpill>>,
    len: usize,
    /// The frame, for the frame-reading functions; `None` for `LAG` / `LEAD`.
    span: Option<Span>,
    peers: Option<Peers<'a>>,
    /// The current row's peer group, `peer_lo..peer_end`.
    peer_lo: usize,
    peer_end: usize,
}

impl<'a> Positional<'a> {
    const fn shift(window: &'a WindowExpr, rows: Rc<RefCell<RandomSpill>>, len: usize) -> Self {
        Self {
            window,
            rows,
            len,
            span: None,
            peers: None,
            peer_lo: 0,
            peer_end: 0,
        }
    }

    fn in_frame(
        window: &'a WindowExpr,
        rows: Rc<RefCell<RandomSpill>>,
        len: usize,
        span: Span,
        peers: Option<Peers<'a>>,
    ) -> Self {
        Self {
            span: Some(span),
            peers,
            ..Self::shift(window, rows, len)
        }
    }

    /// The frame of row `k` as `(first, last)` positions, or `None` when it is empty.
    fn frame(&mut self, span: Span, k: usize) -> Result<Option<(usize, usize)>, Error> {
        if k >= self.peer_end {
            self.peer_lo = k;
            self.peer_end = match self.peers.as_mut() {
                Some(peers) => peers.group_of(k, |_| Ok(()))?,
                None => k + 1,
            };
        }
        let len = i64::try_from(self.len).unwrap_or(i64::MAX);
        let at = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
        let ki = at(k);
        let off = |n: u64| i64::try_from(n).unwrap_or(i64::MAX);
        let pos = |edge: Edge, start: bool| match edge {
            Edge::First => 0,
            Edge::Last => len - 1,
            Edge::Current if span.peer_based && start => at(self.peer_lo),
            Edge::Current if span.peer_based => at(self.peer_end) - 1,
            Edge::Current => ki,
            Edge::Before(n) => ki.saturating_sub(off(n)),
            Edge::After(n) => ki.saturating_add(off(n)),
        };
        let lo = pos(span.start, true).max(0);
        let hi = pos(span.end, false).min(len - 1);
        Ok((lo <= hi).then(|| {
            (
                usize::try_from(lo).unwrap_or(0),
                usize::try_from(hi).unwrap_or(0),
            )
        }))
    }

    /// The integer the expression gives for the current row, or `None` for anything else.
    fn int_arg(&self, index: usize, row: &Row) -> Result<Option<i64>, Error> {
        match self.window.args.get(index) {
            Some(expr) => match eval::eval(expr, row)? {
                ast::Value::Int(n) => Ok(Some(n)),
                _ => Ok(None),
            },
            None => Ok(None),
        }
    }
}

impl Evaluator for Positional<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let Some(value_expr) = self.window.args.first() else {
            return Ok(ast::Value::Null);
        };
        let target = match self.span {
            None => {
                let offset = self.int_arg(1, row)?.unwrap_or(1);
                let delta = if matches!(self.window.func, W::Lag) {
                    offset.checked_neg().unwrap_or(0)
                } else {
                    offset
                };
                i64::try_from(k)
                    .ok()
                    .and_then(|cur| cur.checked_add(delta))
                    .and_then(|p| usize::try_from(p).ok())
                    .filter(|&t| t < self.len)
            },
            Some(span) => {
                let frame = self.frame(span, k)?;
                match self.window.func {
                    W::FirstValue => frame.map(|(lo, _)| lo),
                    W::LastValue => frame.map(|(_, hi)| hi),
                    _ => match (self.int_arg(1, row)?, frame) {
                        (Some(n), Some((lo, hi))) if n >= 1 => usize::try_from(n - 1)
                            .ok()
                            .and_then(|i| lo.checked_add(i))
                            .filter(|&t| t <= hi),
                        _ => None,
                    },
                }
            },
        };
        let found = match target {
            Some(t) => self.rows.borrow_mut().get(t)?,
            None => None,
        };
        match found {
            Some(target) => eval::eval(value_expr, &target),
            // Out of range: the default argument evaluated at the current row, else NULL.
            None => self
                .window
                .args
                .get(2)
                .map_or(Ok(ast::Value::Null), |default| eval::eval(default, row)),
        }
    }
}

/// `FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE` over a spilled partition, or `None` for a frame no
/// streamed evaluator covers.
fn frame_navigation<'a>(
    window: &'a WindowExpr,
    random: &mut RandomCopy<'_>,
    len: usize,
) -> Result<Option<Box<dyn Evaluator + 'a>>, Error> {
    let file = random.file;
    if let Some((lo, hi)) = frame_shape(window)
        && let Some(value) = FrameValue::new(window, file, len, lo, hi)?
    {
        return Ok(Some(Box::new(value)));
    }
    // Every `ROWS` frame is a span, so what is left is a `RANGE` / `GROUPS` frame with an offset.
    if let Some(span) = Span::of(window) {
        let peers = span
            .needs_peers()
            .then(|| Peers::new(&window.order, file))
            .transpose()?;
        return Ok(Some(Box::new(Positional::in_frame(
            window,
            random.rows()?,
            len,
            span,
            peers,
        ))));
    }
    let Some(sliding) = PeerSliding::new(window, None, file, len, random.config.threshold_bytes)?
    else {
        return Ok(None);
    };
    Ok(Some(Box::new(sliding)))
}

/// The order-key values of `row` under `order`.
fn order_key(order: &[OrderByKey], row: &Row) -> Result<Vec<ast::Value>, Error> {
    order.iter().map(|k| eval::eval(&k.expr, row)).collect()
}

/// Where a frame starts, relative to the current row `k`.
#[derive(Clone, Copy)]
enum Lo {
    /// The partition's first row.
    Start,
    /// `n` rows back (`ROWS n PRECEDING`), clamped to the first row.
    Back(usize),
    /// The current row (`ROWS CURRENT ROW`).
    Current,
}

/// Where a frame ends, relative to the current row `k`.
#[derive(Clone, Copy)]
enum Hi {
    /// The partition's last row.
    End,
    /// The current row (`ROWS CURRENT ROW`).
    Current,
    /// `n` rows ahead (`ROWS n FOLLOWING`), clamped to the last row.
    Ahead(usize),
    /// The last of the current row's peers (the default frame under an `ORDER BY`, or a `RANGE` /
    /// `GROUPS` frame ending at `CURRENT ROW`).
    PeerEnd,
}

/// The frame of `window` in the shapes the streamed evaluation supports, or `None`. Every shape
/// here contains the current row, so the frame is never empty.
fn frame_shape(window: &WindowExpr) -> Option<(Lo, Hi)> {
    let Some(frame) = &window.frame else {
        let hi = if window.order.is_empty() {
            Hi::End
        } else {
            Hi::PeerEnd
        };
        return Some((Lo::Start, hi));
    };
    let lo = match (&frame.start, frame.peer_based) {
        (FrameBound::UnboundedPreceding, _) => Lo::Start,
        (FrameBound::CurrentRow, false) => Lo::Current,
        (FrameBound::Preceding(n), false) => Lo::Back(usize::try_from(*n).unwrap_or(usize::MAX)),
        _ => return None,
    };
    let hi = match (&frame.end, frame.peer_based) {
        (FrameBound::UnboundedFollowing, _) => Hi::End,
        (FrameBound::CurrentRow, false) => Hi::Current,
        (FrameBound::CurrentRow, true) => Hi::PeerEnd,
        (FrameBound::Following(n), false) => Hi::Ahead(usize::try_from(*n).unwrap_or(usize::MAX)),
        _ => return None,
    };
    Some((lo, hi))
}

/// Finds peer groups (rows with equal ordering keys) ahead of the current row with its own cursor.
struct Peers<'a> {
    order: &'a [OrderByKey],
    ahead: SpillCursor,
    /// The first row of the group after the current one.
    upcoming: Option<Row>,
    /// One past the last position of the current group.
    end: usize,
    /// The current group's last row.
    last: Option<Row>,
}

impl<'a> Peers<'a> {
    fn new(order: &'a [OrderByKey], file: &SharedSpill) -> Result<Self, Error> {
        let mut ahead = file.cursor()?;
        let upcoming = ahead.read_row()?;
        Ok(Self {
            order,
            ahead,
            upcoming,
            end: 0,
            last: None,
        })
    }

    /// Advance to the group holding position `k`, passing each newly read row to `each`; returns
    /// one past the group's last position.
    fn group_of(
        &mut self,
        k: usize,
        mut each: impl FnMut(&Row) -> Result<(), Error>,
    ) -> Result<usize, Error> {
        while k >= self.end {
            let Some(first) = self.upcoming.take() else {
                break;
            };
            let key = order_key(self.order, &first)?;
            each(&first)?;
            let mut last = first;
            let mut count = 1;
            loop {
                match self.ahead.read_row()? {
                    Some(row) if group_keys_equal(&key, &order_key(self.order, &row)?) => {
                        each(&row)?;
                        last = row;
                        count += 1;
                    },
                    other => {
                        self.upcoming = other;
                        break;
                    },
                }
            }
            self.end += count;
            self.last = Some(last);
        }
        Ok(self.end)
    }
}

/// `ROW_NUMBER`, `RANK`, `DENSE_RANK` and `PERCENT_RANK`, as [`assign_ranking`] and
/// [`assign_distribution`] compute them.
struct Ranking<'a> {
    window: &'a WindowExpr,
    len: usize,
    prev: Option<Vec<ast::Value>>,
    rank: usize,
    dense: usize,
}

impl Evaluator for Ranking<'_> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "row counts widen to f64 for the [0,1] PERCENT_RANK ratio, as in memory"
    )]
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let key = order_key(&self.window.order, row)?;
        if self
            .prev
            .as_ref()
            .is_none_or(|p| !group_keys_equal(p, &key))
        {
            self.dense += 1;
            self.rank = k + 1;
        }
        self.prev = Some(key);
        let int = |n: usize| ast::Value::Int(i64::try_from(n).unwrap_or(i64::MAX));
        Ok(match self.window.func {
            W::RowNumber => int(k + 1),
            W::Rank => int(self.rank),
            W::DenseRank => int(self.dense),
            _ if self.len > 1 => ast::Value::Float((self.rank - 1) as f64 / (self.len - 1) as f64),
            _ => ast::Value::Float(0.0),
        })
    }
}

/// `CUME_DIST`: the share of the partition up to the end of the current row's peer group.
struct CumeDist<'a> {
    peers: Peers<'a>,
    len: usize,
}

impl Evaluator for CumeDist<'_> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "row counts widen to f64 for the (0,1] CUME_DIST ratio, as in memory"
    )]
    fn value(&mut self, k: usize, _row: &Row) -> Result<ast::Value, Error> {
        let end = self.peers.group_of(k, |_| Ok(()))?;
        Ok(ast::Value::Float(end as f64 / self.len as f64))
    }
}

/// `NTILE(n)`: the bucket count is read at the partition's first row, as in memory.
struct Ntile<'a> {
    window: &'a WindowExpr,
    len: usize,
    buckets: Option<usize>,
}

impl Evaluator for Ntile<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let n = if let Some(n) = self.buckets {
            n
        } else {
            let raw = match self.window.args.first() {
                Some(e) => match eval::eval(e, row)? {
                    ast::Value::Int(n) => n,
                    _ => 0,
                },
                None => 0,
            };
            if raw < 1 {
                return Err(Error::InvalidParameterValue(
                    "NTILE requires a positive bucket count".to_owned(),
                ));
            }
            let n = usize::try_from(raw).unwrap_or(usize::MAX);
            self.buckets = Some(n);
            n
        };
        let base = self.len / n;
        let rem = self.len % n;
        let big = rem * (base + 1);
        let bucket = if base == 0 || k < big {
            k / (base + 1)
        } else {
            rem + (k - big) / base
        };
        Ok(ast::Value::Int(
            i64::try_from(bucket + 1).unwrap_or(i64::MAX),
        ))
    }
}

/// `LAG` / `LEAD` with a constant offset: the row `delta` positions away, else the default
/// argument evaluated at the current row (else `NULL`).
struct Shift<'a> {
    window: &'a WindowExpr,
    delta: i64,
    /// For a backward shift: a cursor `|delta|` rows behind the current row once it is that far
    /// in, so the shift holds no rows however large the offset.
    behind: Option<SpillCursor>,
    /// For a forward shift: a cursor `delta` rows ahead of the current row.
    ahead: Option<SpillCursor>,
}

impl<'a> Shift<'a> {
    fn new(window: &'a WindowExpr, file: &SharedSpill, delta: i64) -> Result<Self, Error> {
        let ahead = if delta > 0 {
            let mut cursor = file.cursor()?;
            for _ in 0..delta {
                if cursor.read_row()?.is_none() {
                    break;
                }
            }
            Some(cursor)
        } else {
            None
        };
        let behind = (delta < 0).then(|| file.cursor()).transpose()?;
        Ok(Self {
            window,
            delta,
            behind,
            ahead,
        })
    }

    fn fallback(&self, row: &Row) -> Result<ast::Value, Error> {
        self.window
            .args
            .get(2)
            .map_or(Ok(ast::Value::Null), |default| eval::eval(default, row))
    }
}

impl Evaluator for Shift<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let Some(value_expr) = self.window.args.first() else {
            return Ok(ast::Value::Null);
        };
        let target = if let Some(ahead) = self.ahead.as_mut() {
            ahead.read_row()?
        } else if let Some(behind) = self.behind.as_mut() {
            // Row `k - back` exists once `k >= back`; from then on the cursor yields one row per
            // call, in step with the current row.
            let back = usize::try_from(self.delta.unsigned_abs()).unwrap_or(usize::MAX);
            if k >= back { behind.read_row()? } else { None }
        } else {
            return eval::eval(value_expr, row);
        };
        target.map_or_else(
            || self.fallback(row),
            |target| eval::eval(value_expr, &target),
        )
    }
}

/// `FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE`: the value expression at a frame position.
struct FrameValue<'a> {
    window: &'a WindowExpr,
    len: usize,
    lo: Lo,
    hi: Hi,
    /// `NTH_VALUE`'s 1-based position from the frame start.
    nth: Option<i64>,
    /// The partition's first row (`Lo::Start` targets) or, for `NTH_VALUE`, its nth row.
    fixed: Option<Row>,
    /// For `Lo::Back(n)`: a cursor that starts moving one row per call once the frame start leaves
    /// the first row, and the row at the frame start it last produced.
    trailing: Option<SpillCursor>,
    trailing_row: Option<Row>,
    /// Reads ahead: to the partition's last row (`Hi::End`) or `n` rows ahead (`Hi::Ahead`).
    ahead: Option<SpillCursor>,
    /// The last row the forward cursor produced.
    ahead_last: Option<Row>,
    peers: Option<Peers<'a>>,
}

impl<'a> FrameValue<'a> {
    /// `None` for a combination the streamed evaluation does not cover.
    fn new(
        window: &'a WindowExpr,
        file: &SharedSpill,
        len: usize,
        lo: Lo,
        hi: Hi,
    ) -> Result<Option<Self>, Error> {
        let mut value = Self {
            window,
            len,
            lo,
            hi,
            nth: None,
            fixed: None,
            trailing: None,
            trailing_row: None,
            ahead: None,
            ahead_last: None,
            peers: None,
        };
        match window.func {
            W::FirstValue => match lo {
                Lo::Start => value.fixed = file.cursor()?.read_row()?,
                Lo::Back(_) => {
                    let mut cursor = file.cursor()?;
                    value.trailing_row = cursor.read_row()?;
                    value.trailing = Some(cursor);
                },
                Lo::Current => {},
            },
            W::LastValue => match hi {
                Hi::End => {
                    let mut cursor = file.cursor()?;
                    let mut read = 0usize;
                    while let Some(row) = cursor.read_row()? {
                        value.ahead_last = Some(row);
                        read += 1;
                        if read.is_multiple_of(1024) {
                            crate::cancel::check()?;
                        }
                    }
                },
                Hi::Ahead(n) => {
                    let mut cursor = file.cursor()?;
                    for _ in 0..n {
                        match cursor.read_row()? {
                            Some(row) => value.ahead_last = Some(row),
                            None => break,
                        }
                    }
                    value.ahead = Some(cursor);
                },
                Hi::PeerEnd => value.peers = Some(Peers::new(&window.order, file)?),
                Hi::Current => {},
            },
            _ => {
                // NTH_VALUE: a constant position counted from the partition start.
                if !matches!(lo, Lo::Start) {
                    return Ok(None);
                }
                value.nth = match window.args.get(1).map(|e| &e.kind) {
                    Some(crate::planner::TypedExprKind::Literal(ast::Value::Int(n))) => Some(*n),
                    Some(crate::planner::TypedExprKind::Literal(ast::Value::Null)) => Some(0),
                    _ => return Ok(None),
                };
                if let Some(n) = value.nth.filter(|&n| n >= 1)
                    && let Ok(target) = usize::try_from(n - 1)
                    && target < len
                {
                    let mut cursor = file.cursor()?;
                    for _ in 0..target {
                        cursor.read_row()?;
                    }
                    value.fixed = cursor.read_row()?;
                }
                if matches!(hi, Hi::PeerEnd) {
                    value.peers = Some(Peers::new(&window.order, file)?);
                }
            },
        }
        Ok(Some(value))
    }

    /// The last position of the frame of row `k`.
    fn frame_end(&mut self, k: usize) -> Result<usize, Error> {
        Ok(match self.hi {
            Hi::End => self.len.saturating_sub(1),
            Hi::Current => k,
            Hi::Ahead(n) => k.saturating_add(n).min(self.len.saturating_sub(1)),
            Hi::PeerEnd => match self.peers.as_mut() {
                Some(peers) => peers.group_of(k, |_| Ok(()))?.saturating_sub(1),
                None => k,
            },
        })
    }
}

impl Evaluator for FrameValue<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let Some(value_expr) = self.window.args.first() else {
            return Ok(ast::Value::Null);
        };
        match self.window.func {
            W::FirstValue => match self.lo {
                Lo::Start => self
                    .fixed
                    .as_ref()
                    .map_or(Ok(ast::Value::Null), |first| eval::eval(value_expr, first)),
                Lo::Current => eval::eval(value_expr, row),
                Lo::Back(n) => {
                    // The frame starts at row 0 until `k > n`, then at `k - n`, one row further
                    // per call.
                    if k > n
                        && let Some(trailing) = self.trailing.as_mut()
                    {
                        self.trailing_row = trailing.read_row()?;
                    }
                    self.trailing_row
                        .as_ref()
                        .map_or(Ok(ast::Value::Null), |first| eval::eval(value_expr, first))
                },
            },
            W::LastValue => match self.hi {
                Hi::End => self
                    .ahead_last
                    .as_ref()
                    .map_or(Ok(ast::Value::Null), |last| eval::eval(value_expr, last)),
                Hi::Current => eval::eval(value_expr, row),
                Hi::Ahead(_) => {
                    if let Some(ahead) = self.ahead.as_mut()
                        && let Some(next) = ahead.read_row()?
                    {
                        self.ahead_last = Some(next);
                    }
                    // `k + n` past the end leaves the cursor on the partition's last row.
                    eval::eval(value_expr, self.ahead_last.as_ref().unwrap_or(row))
                },
                Hi::PeerEnd => match self.peers.as_mut() {
                    Some(peers) => {
                        peers.group_of(k, |_| Ok(()))?;
                        eval::eval(value_expr, peers.last.as_ref().unwrap_or(row))
                    },
                    None => eval::eval(value_expr, row),
                },
            },
            _ => {
                let end = self.frame_end(k)?;
                match (self.nth, &self.fixed) {
                    (Some(n), Some(target))
                        if n >= 1 && usize::try_from(n - 1).is_ok_and(|t| t <= end) =>
                    {
                        eval::eval(value_expr, target)
                    },
                    _ => Ok(ast::Value::Null),
                }
            },
        }
    }
}

/// A window aggregate over a frame starting at the partition's first row: one accumulator that
/// takes each row as the frame's end passes it, finalized per row (or per peer group) from a
/// copy, the way the in-memory running aggregate does.
struct RunningAggregate<'a> {
    call: AggregateCall,
    len: usize,
    hi: Hi,
    acc: super::agg::Acc,
    /// Rows folded so far (the frame end of the last row evaluated, plus one).
    folded: usize,
    /// Reads the rows the frame end reaches next (`Hi::Ahead` / `Hi::End`).
    ahead: Option<SpillCursor>,
    peers: Option<Peers<'a>>,
    /// The last value, reused while the frame end has not moved (a peer group, or the whole
    /// partition).
    cached: Option<(usize, ast::Value)>,
}

impl<'a> RunningAggregate<'a> {
    fn new(
        window: &'a WindowExpr,
        call: AggregateCall,
        file: &SharedSpill,
        len: usize,
        hi: Hi,
    ) -> Result<Self, Error> {
        let ahead = matches!(hi, Hi::Ahead(_) | Hi::End)
            .then(|| file.cursor())
            .transpose()?;
        let peers = matches!(hi, Hi::PeerEnd)
            .then(|| Peers::new(&window.order, file))
            .transpose()?;
        Ok(Self {
            call,
            len,
            hi,
            acc: super::agg::Acc::default(),
            folded: 0,
            ahead,
            peers,
            cached: None,
        })
    }

    fn fold(acc: &mut super::agg::Acc, call: &AggregateCall, row: &Row) -> Result<(), Error> {
        super::agg::accumulate_row(std::slice::from_mut(acc), std::slice::from_ref(call), row)
    }
}

impl Evaluator for RunningAggregate<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let end = match self.hi {
            Hi::Current => {
                Self::fold(&mut self.acc, &self.call, row)?;
                k + 1
            },
            Hi::PeerEnd => {
                let (acc, call) = (&mut self.acc, &self.call);
                match self.peers.as_mut() {
                    Some(peers) => peers.group_of(k, |r| Self::fold(acc, call, r))?,
                    None => k + 1,
                }
            },
            Hi::Ahead(_) | Hi::End => {
                let end = match self.hi {
                    Hi::Ahead(n) => k.saturating_add(n).saturating_add(1).min(self.len),
                    _ => self.len,
                };
                while self.folded < end {
                    if self.folded.is_multiple_of(1024) {
                        crate::cancel::check()?;
                    }
                    let Some(next) = self
                        .ahead
                        .as_mut()
                        .map(SpillCursor::read_row)
                        .transpose()?
                        .flatten()
                    else {
                        break;
                    };
                    Self::fold(&mut self.acc, &self.call, &next)?;
                    self.folded += 1;
                }
                end
            },
        };
        if let Some((at, value)) = &self.cached
            && *at == end
        {
            return Ok(value.clone());
        }
        let value = super::agg::finalize_aggregate(self.acc.clone(), &self.call)?;
        self.cached = Some((end, value.clone()));
        Ok(value)
    }
}

/// The offsets of a `ROWS` frame's bounds from the current row (`None` for a partition end), or
/// `None` for a `RANGE` / `GROUPS` frame.
fn rows_offsets(frame: &crate::planner::WindowFrame) -> Option<(Option<i64>, Option<i64>)> {
    if frame.peer_based {
        return None;
    }
    let offset = |bound: &FrameBound| -> Option<Option<i64>> {
        let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        match bound {
            FrameBound::UnboundedPreceding | FrameBound::UnboundedFollowing => Some(None),
            FrameBound::CurrentRow => Some(Some(0)),
            FrameBound::Preceding(v) => Some(Some(-n(*v))),
            FrameBound::Following(v) => Some(Some(n(*v))),
            FrameBound::RangePreceding(_) | FrameBound::RangeFollowing(_) => None,
        }
    };
    Some((offset(&frame.start)?, offset(&frame.end)?))
}

/// The inclusive `[lo, hi]` positions of the `ROWS` frame of row `k` in a partition of `len` rows,
/// or `None` when it is empty, as [`frame_bounds`] places them.
fn rows_frame(
    start: Option<i64>,
    end: Option<i64>,
    k: usize,
    len: usize,
) -> Option<(usize, usize)> {
    let at = |base: usize, rel: i64| i64::try_from(base).unwrap_or(i64::MAX).saturating_add(rel);
    let last = i64::try_from(len).unwrap_or(i64::MAX) - 1;
    let lo = start.map_or(0, |rel| at(k, rel)).max(0);
    let hi = end.map_or(last, |rel| at(k, rel)).min(last);
    if lo > hi {
        return None;
    }
    Some((usize::try_from(lo).ok()?, usize::try_from(hi).ok()?))
}

/// An aggregate over a `ROWS` frame without `EXCLUDE` that has an exact sliding form (see
/// [`SlideState`]): each row entering the frame is added and each row leaving it removed, read by
/// two cursors that only move forward, so a row costs O(1) whatever the frame's width, exactly as
/// the in-memory path computes it. Only a `MIN` / `MAX` keeps values, at most one per frame row;
/// past the budget it fails with the budget error.
struct SlidingAggregate {
    call: AggregateCall,
    state: super::agg::SlideState,
    len: usize,
    start: Option<i64>,
    end: Option<i64>,
    budget: usize,
    ahead: SpillCursor,
    /// Reads the rows leaving the frame, for the aggregates that need their values.
    behind: Option<SpillCursor>,
    /// The live frame is the half-open `[lo, hi)`.
    lo: usize,
    hi: usize,
}

impl SlidingAggregate {
    /// `None` unless the window has a `ROWS` frame without `EXCLUDE` and `call` slides exactly.
    fn new(
        window: &WindowExpr,
        call: AggregateCall,
        file: &SharedSpill,
        len: usize,
        budget: usize,
    ) -> Result<Option<Self>, Error> {
        let Some(frame) = window
            .frame
            .as_ref()
            .filter(|f| matches!(f.exclude, ast::WindowExclude::NoOthers))
        else {
            return Ok(None);
        };
        let Some((start, end)) = rows_offsets(frame) else {
            return Ok(None);
        };
        let Some(state) = super::agg::SlideState::new(&call) else {
            return Ok(None);
        };
        let behind = state.reads_leaving().then(|| file.cursor()).transpose()?;
        Ok(Some(Self {
            call,
            state,
            len,
            start,
            end,
            budget,
            ahead: file.cursor()?,
            behind,
            lo: 0,
            hi: 0,
        }))
    }

    /// The aggregate's argument for `row` (`NULL` for `count(*)`, which never reads it).
    fn argument(call: &AggregateCall, row: &Row) -> Result<ast::Value, Error> {
        call.arg
            .as_ref()
            .map_or(Ok(ast::Value::Null), |arg| eval::eval(arg, row))
    }
}

impl Evaluator for SlidingAggregate {
    fn value(&mut self, k: usize, _row: &Row) -> Result<ast::Value, Error> {
        let Some((lo, hi)) = rows_frame(self.start, self.end, k, self.len) else {
            return super::agg::finalize_aggregate(super::agg::Acc::default(), &self.call);
        };
        while self.hi <= hi {
            let row = self.ahead.read_row()?.ok_or_else(|| {
                Error::Internal("a window frame ran past its partition".to_owned())
            })?;
            self.state.add(self.hi, Self::argument(&self.call, &row)?)?;
            self.hi += 1;
            self.check_budget()?;
        }
        while self.lo < lo {
            let leaving = match self.behind.as_mut() {
                Some(behind) => behind.read_row()?,
                None => None,
            };
            let call = &self.call;
            self.state.remove(self.lo, || {
                leaving
                    .as_ref()
                    .map_or(Ok(ast::Value::Null), |row| Self::argument(call, row))
            })?;
            self.lo += 1;
        }
        self.state.value(self.hi - self.lo, &self.call)
    }
}

impl SlidingAggregate {
    /// Fail with the budget error once a `MIN` / `MAX` holds more than the budget.
    fn check_budget(&self) -> Result<(), Error> {
        if self.budget != 0 && self.state.held_bytes() > self.budget {
            return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                "query work_mem of {} bytes exceeded: the window frame of {} holds {} bytes; use \
                     a narrower frame or raise work_mem (SET work_mem / --work-mem)",
                self.budget,
                sql_name(&ast::WindowFunc::Aggregate(self.call.func)),
                self.state.held_bytes()
            ))));
        }
        Ok(())
    }
}

/// Any other aggregate over a `ROWS` frame (one without an exact sliding form, or with `EXCLUDE`):
/// the rows of the current frame held in a window that slides with the current row (its bounds only
/// move forward), each value folded over the frame the way the in-memory path folds it. Memory is
/// the frame's width, so a frame wider than the budget fails with the budget error.
struct Sliding<'a> {
    window: &'a WindowExpr,
    call: AggregateCall,
    len: usize,
    /// The frame start relative to the current row, or `None` for the partition's first row.
    start: Option<i64>,
    /// The frame end relative to the current row, or `None` for the partition's last row.
    end: Option<i64>,
    exclude: ast::WindowExclude,
    budget: usize,
    /// The frame's rows with their positions and ordering keys, oldest first.
    held: VecDeque<(usize, Row, Vec<ast::Value>)>,
    held_bytes: usize,
    ahead: SpillCursor,
    /// Rows read from `ahead` so far.
    read: usize,
}

impl<'a> Sliding<'a> {
    /// `None` unless the window is an aggregate with an explicit `ROWS` frame.
    fn new(
        window: &'a WindowExpr,
        call: Option<AggregateCall>,
        file: &SharedSpill,
        len: usize,
        budget: usize,
    ) -> Result<Option<Self>, Error> {
        let (Some(call), Some(frame)) = (call, window.frame.as_ref()) else {
            return Ok(None);
        };
        let Some((start, end)) = rows_offsets(frame) else {
            return Ok(None);
        };
        Ok(Some(Self {
            window,
            call,
            len,
            start,
            end,
            exclude: frame.exclude,
            budget,
            held: VecDeque::new(),
            held_bytes: 0,
            ahead: file.cursor()?,
            read: 0,
        }))
    }
}

impl Evaluator for Sliding<'_> {
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        let Some((lo, hi)) = rows_frame(self.start, self.end, k, self.len) else {
            return super::agg::finalize_aggregate(super::agg::Acc::default(), &self.call);
        };
        // Read up to the frame end, keeping only rows at or past the frame start.
        while self.read <= hi {
            let Some(next) = self.ahead.read_row()? else {
                break;
            };
            let pos = self.read;
            self.read += 1;
            if pos >= lo {
                // Peers matter only to an exclusion; skip their keys otherwise.
                let key = if matches!(self.exclude, ast::WindowExclude::NoOthers) {
                    Vec::new()
                } else {
                    order_key(&self.window.order, &next)?
                };
                self.held_bytes += row_bytes(&next) + row_bytes(&key);
                self.held.push_back((pos, next, key));
            }
        }
        while self.held.front().is_some_and(|(pos, _, _)| *pos < lo) {
            if let Some((_, gone, key)) = self.held.pop_front() {
                self.held_bytes = self
                    .held_bytes
                    .saturating_sub(row_bytes(&gone) + row_bytes(&key));
            }
        }
        if self.budget != 0 && self.held.len() > 1 && self.held_bytes > self.budget {
            return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                "query work_mem of {} bytes exceeded: the window frame of {} holds {} bytes; use \
                 a narrower frame or raise work_mem (SET work_mem / --work-mem)",
                self.budget,
                sql_name(&self.window.func),
                self.held_bytes
            ))));
        }
        // EXCLUDE drops the current row, its peers, or its peers but itself; inside a ROWS frame
        // the peers are exactly the held rows with an equal ordering key.
        let current = if matches!(self.exclude, ast::WindowExclude::NoOthers) {
            Vec::new()
        } else {
            order_key(&self.window.order, row)?
        };
        let frame = self.held.iter().filter(|(pos, _, key)| {
            (lo..=hi).contains(pos)
                && match self.exclude {
                    ast::WindowExclude::NoOthers => true,
                    ast::WindowExclude::CurrentRow => *pos != k,
                    ast::WindowExclude::Group => !group_keys_equal(key, &current),
                    ast::WindowExclude::Ties => *pos == k || !group_keys_equal(key, &current),
                }
        });
        Ok(
            super::agg::fold_aggregates(
                std::slice::from_ref(&self.call),
                frame.map(|(_, r, _)| r),
            )?
            .into_iter()
            .next()
            .unwrap_or(ast::Value::Null),
        )
    }
}

/// One row held by [`PeerSliding`]: its partition position, the row, its ordering key and the
/// number of its peer group (rows with equal ordering keys share one, counted from 0).
struct PeerRow {
    pos: usize,
    row: Row,
    key: Vec<ast::Value>,
    group: usize,
}

/// A `RANGE` or `GROUPS` frame (any bounds, `EXCLUDE` included) over a partition read from disk.
/// The rows from the frame start (or the current row, if earlier) to the frame end are held with
/// their peer group numbers; each bound is found the way [`frame_bounds`] finds it in memory, and
/// since both bounds only move forward the held rows drop off the front. Memory is the frame's
/// width, so a frame wider than the budget fails with the budget error.
struct PeerSliding<'a> {
    window: &'a WindowExpr,
    frame: &'a crate::planner::WindowFrame,
    call: Option<AggregateCall>,
    len: usize,
    budget: usize,
    held: VecDeque<PeerRow>,
    held_bytes: usize,
    ahead: SpillCursor,
    /// Rows read from `ahead` so far.
    read: usize,
    /// The ordering key and group number of the last row read.
    last: Option<(Vec<ast::Value>, usize)>,
}

impl<'a> PeerSliding<'a> {
    /// `None` unless the window has an explicit peer-based (`RANGE` / `GROUPS`) frame.
    fn new(
        window: &'a WindowExpr,
        call: Option<AggregateCall>,
        file: &SharedSpill,
        len: usize,
        budget: usize,
    ) -> Result<Option<Self>, Error> {
        let Some(frame) = window.frame.as_ref().filter(|f| f.peer_based) else {
            return Ok(None);
        };
        Ok(Some(Self {
            window,
            frame,
            call,
            len,
            budget,
            held: VecDeque::new(),
            held_bytes: 0,
            ahead: file.cursor()?,
            read: 0,
            last: None,
        }))
    }

    /// Read one more row into the held window; `false` at the partition's end.
    fn read_one(&mut self) -> Result<bool, Error> {
        let Some(row) = self.ahead.read_row()? else {
            return Ok(false);
        };
        let key = order_key(&self.window.order, &row)?;
        let group = match &self.last {
            Some((prev, group)) if group_keys_equal(prev, &key) => *group,
            Some((_, group)) => group + 1,
            None => 0,
        };
        self.last = Some((key.clone(), group));
        self.held_bytes += row_bytes(&row) + row_bytes(&key);
        self.held.push_back(PeerRow {
            pos: self.read,
            row,
            key,
            group,
        });
        self.read += 1;
        Ok(true)
    }

    /// Read until the last row held satisfies `done` (the frame end lies behind it) or the
    /// partition ends.
    fn read_until(&mut self, done: impl Fn(&PeerRow) -> bool) -> Result<(), Error> {
        while self.held.back().is_none_or(|last| !done(last)) {
            if !self.read_one()? {
                break;
            }
        }
        Ok(())
    }

    /// The `RANGE` key of an ordering key: its first value, `None` when `NULL` or unsupported.
    fn range_of(key: &[ast::Value]) -> Option<super::ops::RangeKey> {
        key.first().and_then(super::ops::range_key)
    }

    /// Whether a row's `RANGE` key is on the inside of `boundary` for a start (`at_start`) or end
    /// bound, as the in-memory range scan decides it.
    fn reaches(
        descending: bool,
        key: &[ast::Value],
        boundary: super::ops::RangeKey,
        at_start: bool,
    ) -> bool {
        let ascending = !descending;
        Self::range_of(key).is_some_and(|k| {
            let ord = k.compare(boundary);
            if at_start == ascending {
                ord != std::cmp::Ordering::Less
            } else {
                ord != std::cmp::Ordering::Greater
            }
        })
    }
}

impl Evaluator for PeerSliding<'_> {
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per frame bound kind, mirroring frame_bounds"
    )]
    fn value(&mut self, k: usize, row: &Row) -> Result<ast::Value, Error> {
        // The current row and its peer group.
        while self.read <= k {
            if !self.read_one()? {
                break;
            }
        }
        let Some(current) = self.held.iter().find(|r| r.pos == k) else {
            return Ok(ast::Value::Null);
        };
        let group = current.group;
        let current_key = current.key.clone();
        let last_pos = self.len.saturating_sub(1);
        // The current peer group's last position: read until a row of a later group is held.
        self.read_until(|r| r.group > group)?;
        let peer_lo = self
            .held
            .iter()
            .find(|r| r.group == group)
            .map_or(k, |r| r.pos);
        let peer_hi = self
            .held
            .iter()
            .rev()
            .find(|r| r.group == group)
            .map_or(k, |r| r.pos);
        let descending = self.frame.range_descending;
        let ascending = !descending;
        let ranged = matches!(
            self.frame.start,
            FrameBound::RangePreceding(_) | FrameBound::RangeFollowing(_)
        ) || matches!(
            self.frame.end,
            FrameBound::RangePreceding(_) | FrameBound::RangeFollowing(_)
        );
        let current_range = Self::range_of(&current_key);
        // A NULL current ordering value frames only its peers.
        let (lo, hi) = if ranged && current_range.is_none() {
            (Some(peer_lo), Some(peer_hi))
        } else {
            let group_at = |this: &Self, target: usize, first: bool| -> Option<usize> {
                let mut rows = this.held.iter().filter(|r| r.group == target);
                if first { rows.next() } else { rows.next_back() }.map(|r| r.pos)
            };
            let boundary = |off: &ast::Value, preceding: bool| {
                current_range
                    .and_then(|cur| super::ops::range_boundary(cur, off, preceding, ascending))
            };
            let lo = match &self.frame.start {
                FrameBound::UnboundedPreceding => Some(0),
                FrameBound::CurrentRow => Some(peer_lo),
                FrameBound::Preceding(n) => {
                    let target = group.saturating_sub(usize::try_from(*n).unwrap_or(usize::MAX));
                    group_at(self, target, true)
                },
                FrameBound::Following(n) => {
                    let n = usize::try_from(*n).unwrap_or(usize::MAX);
                    let target = group.saturating_add(n);
                    self.read_until(|r| r.group > target)?;
                    // Past the last group the bound clamps to the last group.
                    let last_group = self.held.back().map_or(group, |r| r.group.min(target));
                    group_at(self, last_group, true)
                },
                FrameBound::RangePreceding(off) | FrameBound::RangeFollowing(off) => {
                    let preceding = matches!(self.frame.start, FrameBound::RangePreceding(_));
                    // An overflowing boundary lies before the partition (PRECEDING) or after it.
                    match boundary(off, preceding) {
                        // Before the partition: its first row with a value (NULLs are outside).
                        None if preceding => Some(
                            self.held
                                .iter()
                                .find(|r| Self::range_of(&r.key).is_some())
                                .map_or(self.len, |r| r.pos),
                        ),
                        None => Some(self.len),
                        Some(b) => {
                            self.read_until(|r| Self::reaches(descending, &r.key, b, true))?;
                            Some(
                                self.held
                                    .iter()
                                    .find(|r| Self::reaches(descending, &r.key, b, true))
                                    .map_or(self.len, |r| r.pos),
                            )
                        },
                    }
                },
                FrameBound::UnboundedFollowing => Some(last_pos),
            };
            let hi = match &self.frame.end {
                FrameBound::UnboundedFollowing => {
                    self.read_until(|_| false)?;
                    Some(last_pos)
                },
                FrameBound::CurrentRow => Some(peer_hi),
                FrameBound::Following(n) => {
                    let target = group.saturating_add(usize::try_from(*n).unwrap_or(usize::MAX));
                    self.read_until(|r| r.group > target)?;
                    let last_group = self.held.back().map_or(group, |r| r.group.min(target));
                    group_at(self, last_group, false)
                },
                FrameBound::Preceding(n) => {
                    let target = group.saturating_sub(usize::try_from(*n).unwrap_or(usize::MAX));
                    group_at(self, target, false)
                },
                FrameBound::RangePreceding(off) | FrameBound::RangeFollowing(off) => {
                    let preceding = matches!(self.frame.end, FrameBound::RangePreceding(_));
                    match boundary(off, preceding) {
                        None if preceding => None,
                        // After the partition: its last row with a value (NULLs are outside).
                        None => {
                            self.read_until(|_| false)?;
                            self.held
                                .iter()
                                .rev()
                                .find(|r| Self::range_of(&r.key).is_some())
                                .map(|r| r.pos)
                        },
                        Some(b) => {
                            // Read past the last row within the boundary: a later row outside it
                            // (a NULL ordering value included, since the NULLs sort together at
                            // one end and a non-NULL current row is not among them), or the end.
                            self.read_until(|r| !Self::reaches(descending, &r.key, b, false))?;
                            self.held
                                .iter()
                                .rev()
                                .find(|r| Self::reaches(descending, &r.key, b, false))
                                .map(|r| r.pos)
                        },
                    }
                },
                FrameBound::UnboundedPreceding => Some(0),
            };
            (lo, hi)
        };
        // Rows before both the frame start and the current row are never needed again.
        let keep_from = lo.unwrap_or(k).min(k);
        while self.held.front().is_some_and(|r| r.pos < keep_from) {
            if let Some(gone) = self.held.pop_front() {
                self.held_bytes = self
                    .held_bytes
                    .saturating_sub(row_bytes(&gone.row) + row_bytes(&gone.key));
            }
        }
        if self.budget != 0 && self.held.len() > 1 && self.held_bytes > self.budget {
            return Err(Error::Core(nusadb_core::Error::OutOfMemory(format!(
                "query work_mem of {} bytes exceeded: the window frame of {} holds {} bytes; use \
                 a narrower frame or raise work_mem (SET work_mem / --work-mem)",
                self.budget,
                sql_name(&self.window.func),
                self.held_bytes
            ))));
        }
        let in_frame = |pos: usize| match (lo, hi) {
            (Some(lo), Some(hi)) => lo <= pos && pos <= hi && lo <= last_pos,
            _ => false,
        };
        if let Some(call) = &self.call {
            let frame = self.held.iter().filter(|r| {
                in_frame(r.pos)
                    && match self.frame.exclude {
                        ast::WindowExclude::NoOthers => true,
                        ast::WindowExclude::CurrentRow => r.pos != k,
                        ast::WindowExclude::Group => r.group != group,
                        ast::WindowExclude::Ties => r.pos == k || r.group != group,
                    }
            });
            return Ok(super::agg::fold_aggregates(
                std::slice::from_ref(call),
                frame.map(|r| &r.row),
            )?
            .into_iter()
            .next()
            .unwrap_or(ast::Value::Null));
        }
        let Some(value_expr) = self.window.args.first() else {
            return Ok(ast::Value::Null);
        };
        let mut frame = self.held.iter().filter(|r| in_frame(r.pos));
        let target = match self.window.func {
            W::FirstValue => frame.next(),
            W::LastValue => frame.next_back(),
            _ => match self
                .window
                .args
                .get(1)
                .map(|e| eval::eval(e, row))
                .transpose()?
            {
                Some(ast::Value::Int(n)) if n >= 1 => {
                    usize::try_from(n - 1).ok().and_then(|i| frame.nth(i))
                },
                _ => None,
            },
        };
        target.map_or(Ok(ast::Value::Null), |r| eval::eval(value_expr, &r.row))
    }
}

/// An aggregate over a frame from the current row (or its first peer) to the partition's end,
/// computed backwards: read in reverse, that frame runs from the start through the current row
/// (or its last peer), which one accumulator folds as it goes. The values are written per
/// partition position and sorted back into partition order. Only for aggregates whose value does
/// not depend on the order rows are folded in (a floating-point sum can round differently).
struct Reversed {
    values: SortedInput,
}

impl Reversed {
    /// `None` unless the frame and aggregate qualify.
    fn try_new(
        window: &WindowExpr,
        call: &AggregateCall,
        file: &SharedSpill,
        config: &SpillConfig,
    ) -> Result<Option<Self>, Error> {
        let Some(frame) = &window.frame else {
            return Ok(None);
        };
        if !matches!(frame.start, FrameBound::CurrentRow)
            || !matches!(frame.end, FrameBound::UnboundedFollowing)
            || !matches!(frame.exclude, ast::WindowExclude::NoOthers)
        {
            return Ok(None);
        }
        let order_free = match call.func {
            ast::AggregateFunc::Count | ast::AggregateFunc::Min | ast::AggregateFunc::Max => true,
            ast::AggregateFunc::Sum | ast::AggregateFunc::Avg => {
                !matches!(call.result_ty, ColumnType::Float)
            },
            _ => false,
        };
        if !order_free {
            return Ok(None);
        }
        // The partition in reverse: each row tagged with its partition position, sorted on it
        // descending.
        let mut tagged = PositionTagged {
            rows: file.cursor()?,
            next: 0,
        };
        let descending = OrderByKey {
            expr: crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
                ty: ColumnType::BigInt,
            },
            ascending: false,
            nulls: ast::NullOrdering::Default,
        };
        let mut backwards = super::spill_sort::sorted_rows(&mut tagged, &[descending], config)?;
        let seq = WINDOW_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut out = SpillWriter::create(config.dir.join(format!(
            "nusadb-spill-window-{}-{seq}.tmp",
            std::process::id()
        )))?;
        let mut acc = super::agg::Acc::reversed();
        // The positions of the current peer group (for a RANGE / GROUPS frame) awaiting its value:
        // a peer group's rows are adjacent in the partition, so its lowest and highest position.
        let mut group: Option<(i64, i64)> = None;
        let mut group_key: Option<Vec<ast::Value>> = None;
        let flush = |out: &mut SpillWriter, span: Option<(i64, i64)>, value: &ast::Value| {
            if let Some((lo, hi)) = span {
                for pos in lo..=hi {
                    out.write_row(&[ast::Value::Int(pos), value.clone()])?;
                }
            }
            Ok::<_, Error>(())
        };
        let mut seen = 0usize;
        while let Some(mut tagged_row) = backwards.try_next()? {
            seen += 1;
            if seen.is_multiple_of(1024) {
                crate::cancel::check()?;
            }
            let pos = match tagged_row.first() {
                Some(ast::Value::Int(pos)) => *pos,
                _ => return Err(Error::Internal("window row lost its position".to_owned())),
            };
            let row: Row = tagged_row.drain(1..).collect();
            if frame.peer_based {
                let key = order_key(&window.order, &row)?;
                if group_key
                    .as_ref()
                    .is_some_and(|k| !group_keys_equal(k, &key))
                {
                    let value = super::agg::finalize_aggregate(acc.clone(), call)?;
                    flush(&mut out, group.take(), &value)?;
                }
                group_key = Some(key);
                RunningAggregate::fold(&mut acc, call, &row)?;
                group = Some(group.map_or((pos, pos), |(lo, hi)| (lo.min(pos), hi.max(pos))));
            } else {
                RunningAggregate::fold(&mut acc, call, &row)?;
                let value = super::agg::finalize_aggregate(acc.clone(), call)?;
                out.write_row(&[ast::Value::Int(pos), value])?;
            }
        }
        if group.is_some() {
            let value = super::agg::finalize_aggregate(acc, call)?;
            flush(&mut out, group.take(), &value)?;
        }
        let ascending = OrderByKey {
            expr: crate::planner::TypedExpr {
                kind: crate::planner::TypedExprKind::Column(0),
                ty: ColumnType::BigInt,
            },
            ascending: true,
            nulls: ast::NullOrdering::Default,
        };
        let mut written = ReaderRows(out.into_reader()?);
        let values = super::spill_sort::sorted_rows(&mut written, &[ascending], config)?;
        Ok(Some(Self { values }))
    }
}

impl Evaluator for Reversed {
    fn value(&mut self, _k: usize, _row: &Row) -> Result<ast::Value, Error> {
        Ok(self
            .values
            .try_next()?
            .and_then(|mut pair| pair.pop())
            .unwrap_or(ast::Value::Null))
    }
}

/// The rows of a partition file, each prefixed with its partition position.
struct PositionTagged {
    rows: SpillCursor,
    next: i64,
}

impl RowSource for PositionTagged {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        let Some(row) = self.rows.read_row()? else {
            return Ok(None);
        };
        let mut tagged = Vec::with_capacity(row.len() + 1);
        tagged.push(ast::Value::Int(self.next));
        tagged.extend(row);
        self.next += 1;
        Ok(Some(tagged))
    }
}

/// The rows of a finished spill file.
struct ReaderRows(super::spill::SpillReader);

impl RowSource for ReaderRows {
    fn try_next(&mut self) -> Result<Option<Row>, Error> {
        self.0.read_row()
    }
}
