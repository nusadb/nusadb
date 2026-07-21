//! Spilling set operations: `UNION`/`INTERSECT`/`EXCEPT [ALL]` over disk-backed sorted
//! streams.
//!
//! The materializing path ([`eval_set_tree`](super::ops::eval_set_tree) → `combine_set`) builds
//! every operand *and* the combined result fully in memory and de-duplicates via a hash/linear set —
//! so a large set-op holds all of its input and output rows at once, several times over for a nested
//! tree. This module bounds the *working set* instead: each leaf is drained into sorted runs on disk
//! (reusing the external merge sort, [`sorted_input`](super::spill_sort::sorted_input)), and
//! each node is a streaming sorted-merge of its two child cursors. The merge output is itself sorted
//! on all columns, so nodes compose up the tree without ever materializing an intermediate operand.
//! Like the rest of Phase 1, the *final* result is still collected into a `Vec<Row>` (end-to-end
//! output bounding is Fase 2); the win is that the operands and the de-dup set no longer live in
//! memory simultaneously.
//!
//! # Correctness
//! Two facts (the same ones spilling `DISTINCT` relies on) reduce set membership to comparing
//! and counting adjacent runs:
//! 1. The all-columns ascending order ([`compare_order_key`](crate::executor::eval::compare_order_key)
//!    with `NULLS Default`) is a **total order**, and
//! 2. rows that are [`group_keys_equal`] sort **adjacent** under it (a tie on every column ⇔
//!    `group_keys_equal`, because `compare_order_key` with default NULL handling falls through to
//!    `compare`, whose equality `group_keys_equal` reuses).
//!
//! So every distinct value occupies one contiguous run in each sorted child, and the operators are:
//! `UNION` = merge (de-dup adjacent for the non-`ALL` form); `INTERSECT [ALL]` = emit values present
//! in both, one copy (or `min(count)` copies for `ALL`); `EXCEPT [ALL]` = emit left values absent
//! from the right, one copy (or `max(0, left − right)` copies for `ALL`). Row equality is SQL "not
//! distinct" (`NULL` = `NULL`), matching `combine_set`, so the result is the same multiset.

#![allow(clippy::wildcard_imports)]

use std::cmp::Ordering;

use super::spill_sort::SortedInput;
use super::*;
use crate::planner::{OrderByKey, TypedExpr, TypedExprKind};

/// Evaluate a set-operation tree with spill-to-disk, returning the combined rows (sorted on all
/// columns; the caller applies any `ORDER BY`/`LIMIT`). `width` is the output column count — the sort
/// key is `Column(0..width)`.
///
/// # Errors
/// Propagates streaming, spill-file I/O, and key-evaluation errors.
pub(super) fn eval_set_tree_spilling(
    tree: &SetOpTree<PhysicalOperator>,
    width: usize,
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let keys = all_column_keys(width);
    let mut stream = build_stream(tree, &keys, config, engine, txn)?;
    let mut out = Vec::new();
    while let Some(row) = stream.take()? {
        out.push(row);
    }
    Ok(out)
}

/// Sort keys over all `width` output columns, ascending with default NULL ordering — the same order
/// spilling `DISTINCT` uses, so `group_keys_equal` rows sort adjacent. The `ty` is a sort-only
/// placeholder: a `Column` ref is resolved by row index, never by its declared type.
fn all_column_keys(width: usize) -> Vec<OrderByKey> {
    (0..width)
        .map(|i| OrderByKey {
            expr: TypedExpr {
                kind: TypedExprKind::Column(i),
                ty: nusadb_core::ColumnType::Int,
            },
            ascending: true,
            nulls: ast::NullOrdering::Default,
        })
        .collect()
}

/// Total order over two equal-width rows: lexicographic `compare_order_key` (ascending, default
/// NULLs) per column. A tie on every column is exactly `group_keys_equal`.
fn cmp_rows(a: &Row, b: &Row) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ord = eval::compare_order_key(x, y, true, ast::NullOrdering::Default);
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len()) // equal for set-op-compatible rows; keeps the order total regardless
}

/// A forward, sorted cursor over a set-op (sub)tree: a sorted leaf, or a merge node.
enum SetSource<'a> {
    Leaf(SortedInput<'a>),
    Node(Box<SetMerge<'a>>),
}

impl SetSource<'_> {
    fn next_row(&mut self) -> Result<Option<Row>, Error> {
        match self {
            Self::Leaf(src) => src.try_next(),
            Self::Node(merge) => merge.next_row(),
        }
    }
}

/// A [`SetSource`] with one row of look-ahead, so a merge can compare and count run boundaries on its
/// children's heads without consuming them.
struct Peek<'a> {
    src: SetSource<'a>,
    head: Option<Row>,
    primed: bool,
}

impl<'a> Peek<'a> {
    const fn new(src: SetSource<'a>) -> Self {
        Self {
            src,
            head: None,
            primed: false,
        }
    }

    /// The current head row without consuming it, or `None` at end.
    fn peek(&mut self) -> Result<Option<&Row>, Error> {
        if !self.primed {
            self.head = self.src.next_row()?;
            self.primed = true;
        }
        Ok(self.head.as_ref())
    }

    /// Consume and return the current head row, or `None` at end.
    fn take(&mut self) -> Result<Option<Row>, Error> {
        self.peek()?;
        self.primed = false;
        Ok(self.head.take())
    }

    /// Drop every leading row `group_keys_equal` to `value`, returning how many were dropped (≥1 when
    /// called on a head equal to `value`).
    fn consume_run(&mut self, value: &Row) -> Result<usize, Error> {
        let mut count = 0;
        while matches!(self.peek()?, Some(head) if group_keys_equal(head, value)) {
            self.take()?;
            count += 1;
        }
        Ok(count)
    }

    /// Take the head row and every following row `group_keys_equal` to it, returning the value and
    /// its run length (≥1), or `None` at end of stream.
    fn take_run(&mut self) -> Result<Option<(Row, usize)>, Error> {
        let Some(value) = self.take()? else {
            return Ok(None);
        };
        let count = 1 + self.consume_run(&value)?;
        Ok(Some((value, count)))
    }
}

/// Which child a merge step should advance.
#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// The relationship between the two children's heads at a merge step.
enum Heads {
    /// Both children are exhausted.
    Done,
    /// Only the left child has rows left.
    LeftOnly,
    /// Only the right child has rows left.
    RightOnly,
    /// Both have a head; ordering is the left head compared to the right.
    Both(Ordering),
}

/// A streaming sorted-merge of two sorted child cursors under one set operator.
struct SetMerge<'a> {
    op: ast::SetOp,
    all: bool,
    left: Peek<'a>,
    right: Peek<'a>,
    /// Last value emitted (non-`ALL` `UNION` dedup against the previous output row).
    last: Option<Row>,
    /// Remaining identical copies still to emit for the current value (`ALL` multiset counts).
    pending: Option<(Row, usize)>,
}

impl SetMerge<'_> {
    /// The next merged row, or `None` at end. Drains buffered `ALL` repeats first, then advances the
    /// per-operator merge.
    fn next_row(&mut self) -> Result<Option<Row>, Error> {
        if let Some((row, remaining)) = self.pending.as_mut() {
            let row = row.clone();
            *remaining -= 1;
            if *remaining == 0 {
                self.pending = None;
            }
            return Ok(Some(row));
        }
        match self.op {
            ast::SetOp::Union => self.next_union(),
            ast::SetOp::Intersect => self.next_intersect(),
            ast::SetOp::Except => self.next_except(),
        }
    }

    /// Classify the two children's heads (disjoint mutable borrows of the two child fields).
    fn heads(&mut self) -> Result<Heads, Error> {
        let left = self.left.peek()?;
        let right = self.right.peek()?;
        Ok(match (left, right) {
            (None, None) => Heads::Done,
            (Some(_), None) => Heads::LeftOnly,
            (None, Some(_)) => Heads::RightOnly,
            (Some(l), Some(r)) => Heads::Both(cmp_rows(l, r)),
        })
    }

    /// Take the head of `side`, which the caller has already established is present.
    fn take_side(&mut self, side: Side) -> Result<Option<Row>, Error> {
        match side {
            Side::Left => self.left.take(),
            Side::Right => self.right.take(),
        }
    }

    /// Emit `value` now and buffer `count - 1` further copies (for `ALL` multiset counts). `count`
    /// must be ≥1.
    fn start_run(&mut self, value: Row, count: usize) -> Row {
        debug_assert!(count >= 1, "start_run needs at least one copy");
        if count > 1 {
            self.pending = Some((value.clone(), count - 1));
        }
        value
    }

    /// `UNION` / `UNION ALL`: merge the two sorted streams; the non-`ALL` form drops a row equal to
    /// the previously emitted one (collapsing duplicates both within and across the children).
    fn next_union(&mut self) -> Result<Option<Row>, Error> {
        loop {
            crate::cancel::check()?;
            let side = match self.heads()? {
                Heads::Done => return Ok(None),
                Heads::RightOnly | Heads::Both(Ordering::Greater) => Side::Right,
                Heads::LeftOnly | Heads::Both(_) => Side::Left,
            };
            let Some(row) = self.take_side(side)? else {
                return Ok(None); // unreachable: `heads()` reported this side present
            };
            if self.all {
                return Ok(Some(row));
            }
            if self
                .last
                .as_ref()
                .is_some_and(|prev| group_keys_equal(prev, &row))
            {
                continue; // duplicate of the previously emitted distinct value
            }
            self.last = Some(row.clone());
            return Ok(Some(row));
        }
    }

    /// `INTERSECT` / `INTERSECT ALL`: emit values present in both children — one copy for the
    /// distinct form, `min(left_count, right_count)` copies for `ALL`. Once either child is
    /// exhausted no further match is possible.
    fn next_intersect(&mut self) -> Result<Option<Row>, Error> {
        loop {
            crate::cancel::check()?;
            match self.heads()? {
                Heads::Done | Heads::LeftOnly | Heads::RightOnly => return Ok(None),
                Heads::Both(Ordering::Less) => {
                    self.left.take()?; // left-only value
                },
                Heads::Both(Ordering::Greater) => {
                    self.right.take()?; // right-only value
                },
                Heads::Both(Ordering::Equal) => {
                    let Some((value, left_count)) = self.left.take_run()? else {
                        return Ok(None);
                    };
                    let right_count = self.right.consume_run(&value)?;
                    return Ok(Some(if self.all {
                        self.start_run(value, left_count.min(right_count))
                    } else {
                        value
                    }));
                },
            }
        }
    }

    /// `EXCEPT` / `EXCEPT ALL`: emit left values absent from the right — one copy for the distinct
    /// form, `max(0, left_count − right_count)` copies for `ALL`.
    fn next_except(&mut self) -> Result<Option<Row>, Error> {
        loop {
            crate::cancel::check()?;
            match self.heads()? {
                Heads::Done | Heads::RightOnly => return Ok(None),
                Heads::LeftOnly | Heads::Both(Ordering::Less) => {
                    // Left value strictly below the right head (or right exhausted) → kept.
                    let Some((value, left_count)) = self.left.take_run()? else {
                        return Ok(None);
                    };
                    return Ok(Some(if self.all {
                        self.start_run(value, left_count)
                    } else {
                        value
                    }));
                },
                Heads::Both(Ordering::Greater) => {
                    self.right.take()?; // right-only value, nothing to subtract from
                },
                Heads::Both(Ordering::Equal) => {
                    let Some((value, left_count)) = self.left.take_run()? else {
                        return Ok(None);
                    };
                    let right_count = self.right.consume_run(&value)?;
                    if self.all {
                        let keep = left_count.saturating_sub(right_count);
                        if keep > 0 {
                            return Ok(Some(self.start_run(value, keep)));
                        }
                    }
                    // distinct form, or `ALL` with the right covering the left → value excluded.
                },
            }
        }
    }
}

/// Build the disk-backed sorted cursor for a set-op (sub)tree. Leaves drain eagerly into sorted runs
/// (so only one leaf's sort buffer is resident at a time); nodes wrap their children in a streaming
/// [`SetMerge`]. Returns a [`Peek`] so a parent merge can look ahead one row.
///
/// # Errors
/// Propagates run-generation (streaming + spill I/O) and key-evaluation errors.
fn build_stream<'a>(
    tree: &SetOpTree<PhysicalOperator>,
    keys: &'a [OrderByKey],
    config: &super::spill::SpillConfig,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Peek<'a>, Error> {
    let src = match tree {
        SetOpTree::Leaf(op) => SetSource::Leaf(super::spill_sort::sorted_input(
            op, keys, config, engine, txn,
        )?),
        SetOpTree::Node {
            op,
            all,
            left,
            right,
        } => {
            let left = build_stream(left, keys, config, engine, txn)?;
            let right = build_stream(right, keys, config, engine, txn)?;
            SetSource::Node(Box::new(SetMerge {
                op: *op,
                all: *all,
                left,
                right,
                last: None,
                pending: None,
            }))
        },
    };
    Ok(Peek::new(src))
}
