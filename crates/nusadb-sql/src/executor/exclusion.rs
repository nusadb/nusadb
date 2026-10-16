//! Exclusion constraints (`EXCLUDE USING gist (k WITH =, r WITH &&)`): no two rows may have every
//! element's operator hold between them.
//!
//! # Storage
//!
//! The constraint catalog has no exclusion kind, so an exclusion constraint is cataloged as a
//! `CHECK` whose opaque expression bytes are its stored form behind a NUL-led tag
//! (`parser::exclusion`). The canonical text of a `CHECK` predicate never starts with a NUL byte (a
//! NUL can only appear quoted, inside a literal or identifier), so no predicate can carry the tag.
//! It is rollback-aware, survives restart and is read `_as_of` like every other constraint, and
//! every `CHECK` consumer tells the two apart with [`is_exclusion`]. When every `=` element is a
//! plain column of a type whose values all index (not a floating-point or `NUMERIC` column, which
//! can hold `NaN`), a plain b-tree index over those columns, named after the constraint, is created
//! with it so the check probes the rows sharing the `=` key instead of scanning the table. The
//! stored form records that index as the constraint's own: only an index it records is ever dropped
//! or skipped on its behalf, so a user's index of the same name is left alone (the constraint then
//! owns none, and probes whatever usable index exists).
//!
//! # Cost
//!
//! The rows a statement writes are grouped by their `=` values. Within one group, with a single
//! `&&` element, conflicts are found with a sweep over the ranges sorted by lower bound, so a
//! group of `n` rows costs `O(n log n)`; with several `&&` elements the group is compared
//! pairwise. Against the existing rows, each distinct `=` group is read once: one index probe per
//! group when an index covers the `=` columns, otherwise one read of the whole table per statement.
//! The index narrows the read to the `=` group only, not to the overlapping ranges: every row of
//! the group is fetched and decoded. A write into a very large group (tens of thousands of rows
//! under one `=` value) therefore costs time proportional to the group, and the index's per-row
//! fetches can then be slower than one sequential read of a table that is mostly that group.
//!
//! # Semantics
//!
//! A row takes part only when every element is non-NULL, no `&&` element is an empty range (an
//! empty range overlaps nothing), and the `WHERE` predicate (if any) is true. Two taking-part rows
//! conflict when every `=` element compares equal (the executor's total order, as `UNIQUE` uses)
//! and every `&&` element overlaps (bound inclusivity respected: `[1,5)` and `[5,9)` do not). A
//! conflict is `23P01` naming the constraint and both keys.
//!
//! # Concurrency
//!
//! The check reads the latest committed rows plus this transaction's own writes (the same
//! latest-committed path `UNIQUE` uses: `scan_committed` / `index_scan_committed`), never the
//! transaction's snapshot, so a row another transaction committed after a `REPEATABLE READ` /
//! `SERIALIZABLE` transaction began is still seen. That alone leaves the race where two
//! transactions each check before the other commits. It is closed with the engine's no-wait key
//! lock, as for `UNIQUE`: before checking, a writer takes an exclusive key lock on `(table,
//! constraint, values of the = elements)`, held to commit or rollback. Two concurrent writers of
//! rows that could conflict share their `=` values, so they need the same lock and the second fails
//! at once with the retryable `40001`; a writer that starts after the other committed takes the
//! lock and its check sees the committed row (`23P01`). A constraint with no `=` element locks one
//! key per constraint, which serializes all its writers. The lock is coarser than the conflict (two
//! writers of non-overlapping ranges under one `=` key also collide, and one retries), which costs
//! throughput, never correctness. A row an `UPDATE` rewrites without changing its element values
//! cannot create a new conflict, so it is neither locked nor re-checked; an `UPDATE` that changes a
//! row's `=` values locks only its new values. That is enough: the row's old version stays
//! committed (and visible to every other writer's check) until the update commits, so a concurrent
//! writer conflicting with the old values is refused by that row itself.
//!
//! # Timing
//!
//! The check is immediate: each writing statement checks its rows before writing them, against
//! each other and against the committed state. `DEFERRABLE` / `INITIALLY ...` are refused when the
//! constraint is declared. [`check_key_against_committed`] is the per-row primitive (lock, read,
//! compare) a later re-check, such as one at commit time, can call for each written row.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::*;
use crate::ast::ExclusionOp;
use crate::planner::TypedExpr;

/// Whether a `CHECK`-kind constraint is an exclusion constraint: not a synthetic type check (a
/// domain's or a declared type's, which only ever holds a predicate), and its stored text is
/// exactly one valid exclusion stored form. Any other `CHECK`, including one that merely starts
/// like the marker, is enforced as an ordinary predicate.
pub(super) fn is_exclusion(c: &nusadb_core::Constraint) -> bool {
    c.kind == nusadb_core::ConstraintKind::Check
        && !c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX)
        && c.expr
            .as_deref()
            .is_some_and(crate::parser::is_exclusion_spec)
}

/// Whether `table` declares any exclusion constraint (a catalog read, nothing compiled).
pub(super) fn table_has_exclusion(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    Ok(engine
        .list_constraints_as_of(txn, table.id)?
        .iter()
        .any(is_exclusion))
}

/// The stored form of an exclusion constraint (its catalog text without the tag), or `None` for
/// any other constraint.
pub(super) fn stored_spec(c: &nusadb_core::Constraint) -> Option<&str> {
    if !is_exclusion(c) {
        return None;
    }
    c.expr
        .as_deref()
        .and_then(crate::parser::untag_exclusion_spec)
}

/// The index an exclusion constraint created for itself, per its stored form.
pub(super) fn owned_index(c: &nusadb_core::Constraint) -> Option<String> {
    stored_spec(c)
        .and_then(|spec| crate::parser::parse_exclusion_spec(spec).ok())
        .and_then(|decl| decl.index)
}

/// One taking-part key: the element values of a row.
type Key = Vec<ast::Value>;

/// An exclusion constraint compiled against its table.
pub(super) struct Exclusion {
    /// The constraint name.
    name: String,
    /// The declaration, as stored.
    decl: ast::ExclusionDecl,
    /// The elements' SQL text as shown in messages.
    element_sql: Vec<String>,
    /// The elements, type-checked against the table, with their operators.
    elements: Vec<(TypedExpr, ExclusionOp)>,
    /// The partial-constraint predicate.
    predicate: Option<TypedExpr>,
    /// The column names of the `=` elements when every `=` element is a plain column and there is
    /// at least one: an index over exactly these columns can serve the check.
    eq_columns: Option<Vec<String>>,
    /// Whether the constraint may create its own index over [`Self::eq_columns`]: not when one of
    /// them is a floating-point or `NUMERIC` column, whose `NaN` the index cannot hold.
    auto_index: bool,
    /// The position of the only `&&` element, when there is exactly one (the sweep's dimension).
    sweep_element: Option<usize>,
    /// The number of `&&` elements.
    overlap_elements: usize,
    /// The table's column types, to bring a row about to be written into its stored form.
    types: Vec<ColumnType>,
}

impl Exclusion {
    /// Compile the stored form `spec` of constraint `name` against `table`.
    pub(super) fn compile(table: &TableSchema, name: &str, spec: &str) -> Result<Self, Error> {
        let decl = crate::parser::parse_exclusion_spec(spec)?;
        let mut elements = Vec::with_capacity(decl.elements.len());
        let mut eq_columns = Vec::new();
        let mut eq_all_columns = true;
        let mut auto_index = true;
        for element in &decl.elements {
            let typed =
                crate::analyzer::analyze_index_key_expr(&element.sql, table, &dml::EmptyCatalog)?;
            if element.op == ExclusionOp::Equal {
                match &typed.kind {
                    crate::planner::TypedExprKind::Column(ordinal) => {
                        if let Some(column) = table.columns.get(*ordinal) {
                            eq_columns.push(column.name.clone());
                            if matches!(
                                column.ty,
                                ColumnType::Float | ColumnType::Real | ColumnType::Numeric { .. }
                            ) {
                                auto_index = false;
                            }
                        }
                    },
                    _ => eq_all_columns = false,
                }
            }
            elements.push((typed, element.op));
        }
        let predicate = decl
            .predicate_sql
            .as_deref()
            .map(|sql| crate::analyzer::analyze_check_predicate(sql, table, &dml::EmptyCatalog))
            .transpose()?;
        let has_eq = elements.iter().any(|(_, op)| *op == ExclusionOp::Equal);
        let overlap_positions: Vec<usize> = elements
            .iter()
            .enumerate()
            .filter(|(_, (_, op))| *op == ExclusionOp::Overlap)
            .map(|(i, _)| i)
            .collect();
        let eq_columns = (has_eq && eq_all_columns).then_some(eq_columns);
        Ok(Self {
            name: name.to_owned(),
            element_sql: decl
                .elements
                .iter()
                .map(|e| crate::parser::display_identifiers(&e.sql))
                .collect(),
            decl,
            elements,
            predicate,
            auto_index: auto_index && eq_columns.is_some(),
            eq_columns,
            sweep_element: match overlap_positions.as_slice() {
                [only] => Some(*only),
                _ => None,
            },
            overlap_elements: overlap_positions.len(),
            types: column_types(table),
        })
    }

    /// [`Self::key`] of a row about to be written. Such a row may still hold a value in its input
    /// form (a range written as text, say), so it is first brought into the form a read of the
    /// stored row returns, the form the existing rows are compared in.
    fn new_key(&self, row: &Row) -> Result<Option<Key>, Error> {
        let stored = row::decode(&row::encode(row, &self.types)?, &self.types)?;
        self.key(&stored)
    }

    /// The element values of `row`, or `None` when the row does not take part: a `NULL` element,
    /// an empty `&&` range, or a `WHERE` predicate that is not true.
    fn key(&self, row: &Row) -> Result<Option<Key>, Error> {
        if let Some(predicate) = &self.predicate
            && !matches!(eval::eval(predicate, row)?, ast::Value::Bool(true))
        {
            return Ok(None);
        }
        let mut key = Vec::with_capacity(self.elements.len());
        for (expr, op) in &self.elements {
            let value = eval::eval(expr, row)?;
            match (&value, op) {
                (ast::Value::Null, _) => return Ok(None),
                (ast::Value::Range(r), ExclusionOp::Overlap) if r.empty => return Ok(None),
                (ast::Value::Range(_), ExclusionOp::Overlap) | (_, ExclusionOp::Equal) => {},
                (other, ExclusionOp::Overlap) => {
                    return Err(Error::Internal(format!(
                        "exclusion constraint \"{}\": element {} is not a range ({other:?})",
                        self.name,
                        self.element_sql.get(key.len()).map_or("?", String::as_str)
                    )));
                },
            }
            key.push(value);
        }
        Ok(Some(key))
    }

    /// Whether two taking-part keys conflict: every `=` element equal and every `&&` element
    /// overlapping.
    fn conflicts(&self, a: &[ast::Value], b: &[ast::Value]) -> bool {
        self.elements
            .iter()
            .zip(a.iter().zip(b))
            .all(|((_, op), (x, y))| match op {
                ExclusionOp::Equal => eval::compare(x, y) == Ordering::Equal,
                ExclusionOp::Overlap => match (x, y) {
                    (ast::Value::Range(x), ast::Value::Range(y)) => x.overlaps(y),
                    _ => false,
                },
            })
    }

    /// Order two keys by their `=` values alone (the executor's total order).
    fn cmp_eq(&self, a: &[ast::Value], b: &[ast::Value]) -> Ordering {
        self.elements
            .iter()
            .zip(a.iter().zip(b))
            .filter(|((_, op), _)| *op == ExclusionOp::Equal)
            .map(|(_, (x, y))| eval::compare(x, y))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }

    /// The values of the `=` elements of `key`, in element order.
    fn eq_part(&self, key: &[ast::Value]) -> Key {
        self.elements
            .iter()
            .zip(key)
            .filter(|((_, op), _)| *op == ExclusionOp::Equal)
            .map(|(_, v)| v.clone())
            .collect()
    }

    /// The range of the sweep element of `key`.
    fn sweep_range<'k>(&self, key: &'k [ast::Value]) -> Option<&'k crate::range::RangeVal> {
        match key.get(self.sweep_element?) {
            Some(ast::Value::Range(r)) => Some(r),
            _ => None,
        }
    }

    /// The `23P01` error for `new` conflicting with `existing`.
    fn violation(&self, new: &[ast::Value], existing: &[ast::Value]) -> Error {
        let columns = self.element_sql.join(", ");
        let render = |key: &[ast::Value]| {
            key.iter()
                .map(crate::display::value_text)
                .collect::<Vec<_>>()
                .join(", ")
        };
        Error::Coded {
            message: format!(
                "conflicting key value violates exclusion constraint \"{}\": key ({columns})=({}) \
                 conflicts with existing key ({columns})=({})",
                self.name,
                render(new),
                render(existing)
            ),
            sqlstate: "23P01",
        }
    }

    /// Take the exclusive key lock covering `key`'s `=` values (see the module docs).
    fn lock(
        &self,
        table: &TableSchema,
        key: &[ast::Value],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        let identity = [format!("exclude:{}", self.name)];
        let hash = dml::unique_key_hash(table, &identity, &self.eq_part(key));
        engine.lock_key(
            txn,
            table.id,
            hash,
            nusadb_core::engine::RowLockMode::Exclusive,
        )?;
        Ok(())
    }

    /// Split `items` (sorted by [`Self::cmp_eq`]) into its runs of equal `=` values.
    fn groups<'a, T>(
        &self,
        items: &'a [T],
        key: impl Fn(&T) -> &[ast::Value],
    ) -> impl Iterator<Item = &'a [T]> {
        let mut rest = items;
        std::iter::from_fn(move || {
            let first = rest.first()?;
            let len = rest
                .iter()
                .position(|item| self.cmp_eq(key(item), key(first)) != Ordering::Equal)
                .unwrap_or(rest.len());
            let (group, tail) = rest.split_at(len);
            rest = tail;
            Some(group)
        })
    }

    /// A conflicting pair within one `=` group of `(key, is_new)` items: `(new, other)`. Pairs of
    /// two existing rows are never reported, and pairs of two new rows only when `new_new`.
    ///
    /// With one `&&` element the items are swept in lower-bound order, keeping the new and the
    /// existing range with the greatest upper bound seen so far. Of the ranges starting at or
    /// before the current one, the one reaching furthest overlaps it whenever any of them does, so
    /// one comparison per kind suffices: `O(n log n)` for the group. Without a `&&` element every
    /// two members conflict; with several the group is compared pairwise.
    fn group_conflict<'k>(
        &self,
        items: &mut [(&'k [ast::Value], bool)],
        new_new: bool,
    ) -> Option<(&'k [ast::Value], &'k [ast::Value])> {
        let counts = |a: bool, b: bool| (a && b && new_new) || (a != b);
        if self.overlap_elements == 0 {
            let first_new = items.iter().position(|(_, new)| *new)?;
            let (new, _) = *items.get(first_new)?;
            return items
                .iter()
                .enumerate()
                .find(|(i, (_, is_new))| *i != first_new && counts(true, *is_new))
                .map(|(_, (other, _))| (new, *other));
        }
        if self.sweep_element.is_some() {
            items.sort_by(
                |(a, _), (b, _)| match (self.sweep_range(a), self.sweep_range(b)) {
                    (Some(x), Some(y)) => x.lower_cmp(y),
                    _ => Ordering::Equal,
                },
            );
            // The range reaching furthest so far, among the new and among the existing keys.
            let mut reach_new: Option<&'k [ast::Value]> = None;
            let mut reach_old: Option<&'k [ast::Value]> = None;
            for &(key, is_new) in items.iter() {
                for (held, held_new) in [(reach_new, true), (reach_old, false)] {
                    if let Some(other) = held
                        && counts(is_new, held_new)
                        && self.conflicts(key, other)
                    {
                        return Some(if is_new { (key, other) } else { (other, key) });
                    }
                }
                let slot = if is_new {
                    &mut reach_new
                } else {
                    &mut reach_old
                };
                let further =
                    slot.is_none_or(
                        |held| match (self.sweep_range(key), self.sweep_range(held)) {
                            (Some(x), Some(y)) => x.upper_cmp(y) == Ordering::Greater,
                            _ => false,
                        },
                    );
                if further {
                    *slot = Some(key);
                }
            }
            return None;
        }
        for (i, &(a, a_new)) in items.iter().enumerate() {
            for &(b, b_new) in items.iter().skip(i + 1) {
                if counts(a_new, b_new) && self.conflicts(a, b) {
                    return Some(if a_new { (a, b) } else { (b, a) });
                }
            }
        }
        None
    }

    /// The first conflicting pair among `keys`, if any (grouped by `=` values, then swept).
    fn conflict_among<'k>(
        &self,
        keys: &[&'k [ast::Value]],
    ) -> Option<(&'k [ast::Value], &'k [ast::Value])> {
        let mut sorted: Vec<(&'k [ast::Value], bool)> = keys.iter().map(|k| (*k, true)).collect();
        sorted.sort_by(|(a, _), (b, _)| self.cmp_eq(a, b));
        let mut start = 0;
        let mut found = None;
        let lens: Vec<usize> = self.groups(&sorted, |(k, _)| k).map(<[_]>::len).collect();
        for len in lens {
            if let Some(group) = sorted.get_mut(start..start + len)
                && let Some(pair) = self.group_conflict(group, true)
            {
                found = Some(pair);
                break;
            }
            start += len;
        }
        found
    }
}

/// Keys sorted under the executor's total order, for membership by binary search.
struct KeySet(Vec<Key>);

impl KeySet {
    fn new(mut keys: Vec<Key>) -> Self {
        keys.sort_by(|a, b| dml::unique_key_cmp(a, b));
        Self(keys)
    }

    fn contains(&self, key: &[ast::Value]) -> bool {
        self.0
            .binary_search_by(|probe| dml::unique_key_cmp(probe, key))
            .is_ok()
    }
}

/// The exclusion constraints on `table`, as `txn` sees the catalog.
pub(super) fn load(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Exclusion>, Error> {
    let mut out = Vec::new();
    for c in engine.list_constraints_as_of(txn, table.id)? {
        if !is_exclusion(&c) {
            continue;
        }
        let spec = stored_spec(&c).ok_or_else(|| {
            Error::Internal(format!(
                "exclusion constraint \"{}\" has a corrupt stored form",
                c.name
            ))
        })?;
        out.push(Exclusion::compile(table, &c.name, spec)?);
    }
    Ok(out)
}

/// Enforce every exclusion constraint of `table` on `new_rows`, the rows a statement is about to
/// write (before any of them is written). `rewritten` holds the row versions the statement replaces
/// or deletes and `old_rows` their images: those versions no longer count as existing rows.
///
/// The new rows must not conflict with each other, nor with any row committed now or written
/// earlier by this transaction (see the module docs for the read path and the locking).
pub(super) fn enforce_on_write(
    table: &TableSchema,
    new_rows: &[Row],
    rewritten: &HashSet<Tid>,
    old_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if new_rows.is_empty() {
        return Ok(());
    }
    enforce_compiled(
        &load(table, engine, txn)?,
        table,
        new_rows,
        rewritten,
        old_rows,
        engine,
        txn,
    )
}

/// [`enforce_on_write`] with the table's constraints already loaded.
pub(super) fn enforce_compiled(
    constraints: &[Exclusion],
    table: &TableSchema,
    new_rows: &[Row],
    rewritten: &HashSet<Tid>,
    old_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for c in constraints {
        let mut new_keys = Vec::with_capacity(new_rows.len());
        for row in new_rows {
            if let Some(key) = c.new_key(row)? {
                new_keys.push(key);
            }
        }
        if new_keys.is_empty() {
            continue;
        }
        let refs: Vec<&[ast::Value]> = new_keys.iter().map(Vec::as_slice).collect();
        if let Some((a, b)) = c.conflict_among(&refs) {
            return Err(c.violation(a, b));
        }
        let mut old_keys = Vec::with_capacity(old_rows.len());
        for row in old_rows {
            if let Some(key) = c.new_key(row)? {
                old_keys.push(key);
            }
        }
        let old_keys = KeySet::new(old_keys);
        // A rewritten row whose element values did not change cannot create a conflict.
        let pending: Vec<&[ast::Value]> = refs
            .iter()
            .copied()
            .filter(|k| !old_keys.contains(k))
            .collect();
        check_against_committed(c, table, &pending, rewritten, &old_keys, engine, txn)?;
    }
    Ok(())
}

/// Check one taking-part `key` of constraint `c` against the committed state (latest committed
/// rows plus this transaction's own writes), taking the key lock first. `rewritten` are row
/// versions that no longer count (the rows being replaced, or the row that holds `key` itself),
/// `old_keys` the keys those rows held.
///
/// This is the per-row form of the immediate check, and the hook for a re-check that runs later
/// than the writing statement: pass the written row's own `Tid` in `rewritten`.
#[allow(dead_code, reason = "the entry point a later, deferred re-check calls")]
pub(super) fn check_key_against_committed(
    c: &Exclusion,
    table: &TableSchema,
    key: &[ast::Value],
    rewritten: &HashSet<Tid>,
    old_keys: &[Key],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let old_keys = KeySet::new(old_keys.to_vec());
    check_against_committed(c, table, &[key], rewritten, &old_keys, engine, txn)
}

/// Check the `pending` keys of constraint `c` against the committed state: lock each distinct `=`
/// group, read the group's existing rows once (one index probe per group, or one table read for
/// all), and sweep each group's pending and existing keys together. Existing versions in
/// `rewritten`, and any whose key is one of `old_keys` (a newer committed version of a row being
/// rewritten: the stale write then fails with `40001`, which must surface rather than a bogus
/// conflict with the row's own image), do not count.
fn check_against_committed(
    c: &Exclusion,
    table: &TableSchema,
    pending: &[&[ast::Value]],
    rewritten: &HashSet<Tid>,
    old_keys: &KeySet,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut pending: Vec<&[ast::Value]> = pending.to_vec();
    pending.sort_by(|a, b| c.cmp_eq(a, b));
    let representatives: Vec<&[ast::Value]> = c
        .groups(&pending, |k| k)
        .filter_map(|group| group.first().copied())
        .collect();
    for key in &representatives {
        c.lock(table, key, engine, txn)?;
    }
    let mut existing: Vec<Key> = Vec::new();
    for (tid, row) in candidates(c, table, &representatives, engine, txn)? {
        if rewritten.contains(&tid) {
            continue;
        }
        if let Some(key) = c.key(&row)?
            && !old_keys.contains(&key)
        {
            existing.push(key);
        }
    }
    existing.sort_by(|a, b| c.cmp_eq(a, b));
    for group in c.groups(&pending, |k| k) {
        let Some(first) = group.first() else { continue };
        let lo = existing.partition_point(|e| c.cmp_eq(e, first) == Ordering::Less);
        let hi = existing.partition_point(|e| c.cmp_eq(e, first) != Ordering::Greater);
        let mut items: Vec<(&[ast::Value], bool)> = group.iter().map(|k| (*k, true)).collect();
        items.extend(
            existing
                .get(lo..hi)
                .unwrap_or_default()
                .iter()
                .map(|k| (k.as_slice(), false)),
        );
        if let Some((new, other)) = c.group_conflict(&mut items, false) {
            return Err(c.violation(new, other));
        }
    }
    Ok(())
}

/// The exclusion check of a statement that writes in batches (`COPY`, a streamed
/// `INSERT ... SELECT`): each batch takes its key locks as it is admitted, and the whole
/// statement's keys are checked once at the end of the stream, against each other and against the
/// committed state (the statement's own rows left out by `Tid`). Checking every batch against the
/// rows already written would read each `=` group once per batch, which grows quadratically with
/// the statement; this keeps it one read per group. Memory is one key per written row.
pub(super) struct Deferred {
    /// The table's exclusion constraints.
    constraints: Vec<Exclusion>,
    /// Per constraint (same order), the keys of the rows the statement wrote.
    keys: Vec<Vec<Key>>,
    /// The rows the statement wrote.
    written: HashSet<Tid>,
}

impl Deferred {
    /// Load `table`'s exclusion constraints; a table without any yields an empty (free) collector.
    pub(super) fn load(
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<Self, Error> {
        let constraints = load(table, engine, txn)?;
        Ok(Self {
            keys: constraints.iter().map(|_| Vec::new()).collect(),
            constraints,
            written: HashSet::new(),
        })
    }

    /// Lock the `=` groups of a batch about to be written and keep its keys for [`Self::finish`].
    pub(super) fn admit(
        &mut self,
        table: &TableSchema,
        rows: &[Row],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        for (c, keys) in self.constraints.iter().zip(&mut self.keys) {
            let mut batch = Vec::with_capacity(rows.len());
            for row in rows {
                if let Some(key) = c.new_key(row)? {
                    batch.push(key);
                }
            }
            let mut refs: Vec<&[ast::Value]> = batch.iter().map(Vec::as_slice).collect();
            refs.sort_by(|a, b| c.cmp_eq(a, b));
            for group in c.groups(&refs, |k| k) {
                if let Some(first) = group.first() {
                    c.lock(table, first, engine, txn)?;
                }
            }
            keys.extend(batch);
        }
        Ok(())
    }

    /// Record a row the statement wrote, so [`Self::finish`] does not compare it with itself.
    pub(super) fn record(&mut self, tid: Tid) {
        if !self.constraints.is_empty() {
            self.written.insert(tid);
        }
    }

    /// The end-of-stream check (every key lock already held).
    pub(super) fn finish(
        &self,
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        for (c, keys) in self.constraints.iter().zip(&self.keys) {
            if keys.is_empty() {
                continue;
            }
            let refs: Vec<&[ast::Value]> = keys.iter().map(Vec::as_slice).collect();
            if let Some((a, b)) = c.conflict_among(&refs) {
                return Err(c.violation(a, b));
            }
            let none = KeySet::new(Vec::new());
            check_against_committed(c, table, &refs, &self.written, &none, engine, txn)?;
        }
        Ok(())
    }
}

/// `ON CONFLICT DO NOTHING`: drop the rows that would violate an exclusion constraint, against the
/// existing rows or an earlier kept row of the batch, the way the statement drops a row that would
/// violate a `UNIQUE` constraint. The kept rows are then enforced as usual. Each row is compared
/// with the existing and kept rows sharing its `=` values.
pub(super) fn keep_non_conflicting(
    table: &TableSchema,
    rows: Vec<Row>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let constraints = load(table, engine, txn)?;
    if constraints.is_empty() {
        return Ok(rows);
    }
    // Per constraint, the keys of the existing rows sorted by `=` values (lock first, so the read
    // is not racing a concurrent writer of the same key).
    let mut seen: Vec<Vec<Key>> = Vec::with_capacity(constraints.len());
    for c in &constraints {
        let mut keys = Vec::new();
        for row in &rows {
            if let Some(key) = c.new_key(row)? {
                keys.push(key);
            }
        }
        let mut refs: Vec<&[ast::Value]> = keys.iter().map(Vec::as_slice).collect();
        refs.sort_by(|a, b| c.cmp_eq(a, b));
        let representatives: Vec<&[ast::Value]> = c
            .groups(&refs, |k| k)
            .filter_map(|g| g.first().copied())
            .collect();
        for key in &representatives {
            c.lock(table, key, engine, txn)?;
        }
        let mut existing = Vec::new();
        for (_, row) in candidates(c, table, &representatives, engine, txn)? {
            if let Some(key) = c.key(&row)? {
                existing.push(key);
            }
        }
        existing.sort_by(|a, b| c.cmp_eq(a, b));
        seen.push(existing);
    }
    let mut kept = Vec::with_capacity(rows.len());
    'rows: for row in rows {
        let mut keys = Vec::with_capacity(constraints.len());
        for (c, seen) in constraints.iter().zip(&seen) {
            let key = c.new_key(&row)?;
            if let Some(key) = &key {
                let lo = seen.partition_point(|e| c.cmp_eq(e, key) == Ordering::Less);
                let hi = seen.partition_point(|e| c.cmp_eq(e, key) != Ordering::Greater);
                if seen
                    .get(lo..hi)
                    .unwrap_or_default()
                    .iter()
                    .any(|s| c.conflicts(key, s))
                {
                    continue 'rows;
                }
            }
            keys.push(key);
        }
        for ((c, key), seen) in constraints.iter().zip(keys).zip(seen.iter_mut()) {
            if let Some(key) = key {
                let at = seen.partition_point(|e| c.cmp_eq(e, &key) != Ordering::Greater);
                seen.insert(at, key);
            }
        }
        kept.push(row);
    }
    Ok(kept)
}

/// Validate an exclusion constraint against the rows `table` holds (read fresh by the caller,
/// which holds the table exclusively): no two of them may conflict.
pub(super) fn validate_existing(
    c: &Exclusion,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut keys = Vec::new();
    for row in scan::scan_rows(table, engine, txn)? {
        if let Some(key) = c.key(&row)? {
            keys.push(key);
        }
    }
    let refs: Vec<&[ast::Value]> = keys.iter().map(Vec::as_slice).collect();
    match c.conflict_among(&refs) {
        Some((a, b)) => Err(c.violation(a, b)),
        None => Ok(()),
    }
}

/// Re-validate every exclusion constraint of `table` against its rows, after a change that
/// rewrote them (`ALTER COLUMN ... TYPE` can make distinct values equal, `1.4` and `1.2` as `int`).
pub(super) fn revalidate_table(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for c in load(table, engine, txn)? {
        validate_existing(&c, table, engine, txn)?;
    }
    Ok(())
}

/// The rows an exclusion check must compare keys against, one key per `=` group in
/// `representatives`: latest committed plus this transaction's own writes. With a usable index
/// over the `=` columns (a full, plain-column b-tree index on exactly those columns, typically the
/// one created with the constraint) only the rows sharing a group's `=` values are read, one probe
/// per group; otherwise the whole table is read once.
fn candidates(
    c: &Exclusion,
    table: &TableSchema,
    representatives: &[&[ast::Value]],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(Tid, Row)>, Error> {
    if let Some((index, order)) = usable_index(c, table, engine, txn)? {
        let mut probes: Vec<Key> = Vec::with_capacity(representatives.len());
        let mut hash_safe = true;
        for key in representatives {
            let eq = c.eq_part(key);
            let probe: Key = order.iter().filter_map(|&i| eq.get(i).cloned()).collect();
            if !probe.iter().all(super::ops::is_hash_safe_value) {
                hash_safe = false;
                break;
            }
            probes.push(probe);
        }
        if hash_safe {
            probes.sort_by(|a, b| dml::unique_key_cmp(a, b));
            probes.dedup_by(|a, b| dml::unique_key_cmp(a, b) == Ordering::Equal);
            let schema = column_types(table);
            let mut rows: HashMap<Tid, Row> = HashMap::new();
            for probe in probes {
                let encoded = index_key::encode_index_key(&probe)?;
                let mut scan = engine.index_scan_committed(
                    txn,
                    index,
                    std::ops::Bound::Included(encoded.clone()),
                    std::ops::Bound::Included(encoded),
                )?;
                while let Some((tid, tuple)) = scan.try_next()? {
                    crate::cancel::check()?;
                    if let std::collections::hash_map::Entry::Vacant(slot) = rows.entry(tid) {
                        slot.insert(row::decode(&tuple, &schema)?);
                    }
                }
            }
            return Ok(rows.into_iter().collect());
        }
    }
    scan::scan_table_committed(table, engine, txn)
}

/// An index whose key is exactly the constraint's `=` columns, and for each index key column the
/// position of its value among the `=` values.
fn usable_index(
    c: &Exclusion,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<(nusadb_core::IndexId, Vec<usize>)>, Error> {
    let Some(eq_columns) = &c.eq_columns else {
        return Ok(None);
    };
    for def in engine.list_indexes_as_of(txn, table.id)? {
        if def.kind != nusadb_core::engine::IndexKind::BTree
            || !def.key_exprs.is_empty()
            || def.predicate.is_some()
            || def.columns.len() != eq_columns.len()
        {
            continue;
        }
        let order: Option<Vec<usize>> = def
            .columns
            .iter()
            .map(|col| eq_columns.iter().position(|e| e == col))
            .collect();
        let Some(order) = order else { continue };
        let distinct: HashSet<usize> = order.iter().copied().collect();
        if distinct.len() != order.len() {
            continue;
        }
        if let Some(id) = engine.lookup_index(&def.name)? {
            return Ok(Some((id, order)));
        }
    }
    Ok(None)
}

/// Register exclusion constraint `name` (stored form `spec`) on `table`. When the constraint may
/// have its own index (see [`Exclusion`]) and its name is free in the index namespace, that index
/// is created over the `=` columns, filled from the table's rows, and recorded in the stored form
/// as the constraint's own. A taken name (a user's index, say) leaves the constraint owning none.
pub(super) fn register(
    table: &TableSchema,
    name: &str,
    spec: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let compiled = Exclusion::compile(table, name, spec)?;
    let owned = match &compiled.eq_columns {
        Some(columns) if compiled.auto_index && engine.lookup_index(name)?.is_none() => {
            let def = nusadb_core::IndexDef {
                name: name.to_owned(),
                table: table.id,
                columns: columns.clone(),
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: nusadb_core::engine::IndexKind::BTree,
                unique: false,
            };
            let id = engine.create_index(txn, &def)?;
            if let Some(target) = dml::build_index_target(id, table, &def) {
                dml::backfill_index(&target, table, engine, txn)?;
            }
            Some(name)
        },
        _ => None,
    };
    let stored = crate::parser::tag_exclusion_spec(&compiled.decl.stored_form(owned));
    engine.add_check_constraint(txn, table.id, name, stored.as_bytes())?;
    Ok(())
}

/// After an exclusion constraint of `table` with stored form `spec` is dropped, drop the index it
/// created for itself, if it recorded one and that index is still there on this table.
pub(super) fn drop_owned_index(
    table: &TableSchema,
    spec: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(index) = crate::parser::parse_exclusion_spec(spec)
        .ok()
        .and_then(|d| d.index)
    else {
        return Ok(());
    };
    let on_this_table = engine
        .list_indexes_as_of(txn, table.id)?
        .iter()
        .any(|def| def.name == index);
    if on_this_table && let Some(id) = engine.lookup_index(&index)? {
        engine.drop_index(txn, id)?;
    }
    Ok(())
}

/// Validate and register an `ALTER TABLE ... ADD CONSTRAINT ... EXCLUDE` (the caller holds the
/// table exclusively and reads fresh).
pub(super) fn add_to_existing(
    table: &TableSchema,
    name: &str,
    spec: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let compiled = Exclusion::compile(table, name, spec)?;
    validate_existing(&compiled, table, engine, txn)?;
    register(table, name, spec, engine, txn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusadb_core::engine::RangeKind;

    fn table() -> TableSchema {
        let col = |name: &str, ty, nullable| ColumnDef {
            name: name.to_owned(),
            ty,
            nullable,
        };
        TableSchema {
            id: nusadb_core::TableId(1),
            schema: "public".to_owned(),
            name: "t".to_owned(),
            columns: vec![
                col("k", ColumnType::Int, true),
                col("r", ColumnType::Range(RangeKind::Int), true),
                col("live", ColumnType::Bool, true),
            ],
        }
    }

    fn range(text: &str) -> ast::Value {
        ast::Value::Range(Box::new(crate::range::parse(text, RangeKind::Int).unwrap()))
    }

    fn row(k: Option<i64>, r: Option<&str>, live: bool) -> Row {
        vec![
            k.map_or(ast::Value::Null, ast::Value::Int),
            r.map_or(ast::Value::Null, range),
            ast::Value::Bool(live),
        ]
    }

    fn compiled() -> Exclusion {
        Exclusion::compile(
            &table(),
            "t_x",
            "__nusadb_exclude('gist', '', k, '=', r, '&&', (live))",
        )
        .unwrap()
    }

    fn conflict(a: &Row, b: &Row) -> bool {
        let c = compiled();
        match (c.key(a).unwrap(), c.key(b).unwrap()) {
            (Some(a), Some(b)) => c.conflicts(&a, &b),
            _ => false,
        }
    }

    #[test]
    fn overlap_respects_bounds_and_keys() {
        let a = row(Some(1), Some("[1,5)"), true);
        assert!(conflict(&a, &row(Some(1), Some("[4,9)"), true)));
        assert!(!conflict(&a, &row(Some(1), Some("[5,9)"), true)));
        assert!(conflict(&a, &row(Some(1), Some("[0,1]"), true)));
        assert!(!conflict(&a, &row(Some(2), Some("[1,5)"), true)));
        assert!(conflict(&a, &row(Some(1), Some("(,)"), true)));
    }

    #[test]
    fn nulls_empty_ranges_and_the_predicate_never_conflict() {
        let a = row(Some(1), Some("[1,5)"), true);
        assert!(!conflict(&a, &row(None, Some("[1,5)"), true)));
        assert!(!conflict(&a, &row(Some(1), None, true)));
        assert!(!conflict(&a, &row(Some(1), Some("empty"), true)));
        assert!(!conflict(&a, &row(Some(1), Some("[1,5)"), false)));
    }

    #[test]
    fn a_batch_conflict_is_found_within_its_key_group() {
        let c = compiled();
        let keys: Vec<Vec<ast::Value>> = [
            row(Some(1), Some("[1,5)"), true),
            row(Some(2), Some("[1,5)"), true),
            row(Some(1), Some("[5,9)"), true),
        ]
        .iter()
        .filter_map(|r| c.key(r).unwrap())
        .collect();
        let refs: Vec<&[ast::Value]> = keys.iter().map(Vec::as_slice).collect();
        assert!(c.conflict_among(&refs).is_none());
        let mut more = keys.clone();
        more.push(c.key(&row(Some(1), Some("[8,10)"), true)).unwrap().unwrap());
        let refs: Vec<&[ast::Value]> = more.iter().map(Vec::as_slice).collect();
        assert!(c.conflict_among(&refs).is_some());
        assert_eq!(c.eq_columns.as_deref(), Some(&["k".to_owned()][..]));
    }

    #[test]
    fn the_violation_names_the_constraint_and_both_keys() {
        let c = compiled();
        let a = c.key(&row(Some(1), Some("[1,5)"), true)).unwrap().unwrap();
        let b = c.key(&row(Some(1), Some("[3,4)"), true)).unwrap().unwrap();
        let Error::Coded { message, sqlstate } = c.violation(&b, &a) else {
            panic!("expected a coded error");
        };
        assert_eq!(sqlstate, "23P01");
        assert_eq!(
            message,
            "conflicting key value violates exclusion constraint \"t_x\": key (k, r)=(1, [3,4)) \
             conflicts with existing key (k, r)=(1, [1,5))"
        );
    }

    /// The sweep agrees with the pairwise definition on every small case.
    #[test]
    fn the_sweep_matches_pairwise_comparison() {
        let c = compiled();
        let ranges = [
            "[1,5)", "[5,9)", "[4,5]", "(8,10)", "[10,10]", "(,2)", "[20,)", "[3,4)", "[9,10)",
        ];
        let keys: Vec<Key> = ranges
            .iter()
            .map(|r| c.key(&row(Some(1), Some(r), true)).unwrap().unwrap())
            .collect();
        // Every subset of up to 4 ranges, split into new and existing in every way.
        let n = keys.len();
        for mask in 0_u32..(1 << n) {
            if mask.count_ones() > 4 {
                continue;
            }
            let chosen: Vec<&Key> = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| &keys[i])
                .collect();
            for sides in 0_u32..(1 << chosen.len()) {
                let mut items: Vec<(&[ast::Value], bool)> = chosen
                    .iter()
                    .enumerate()
                    .map(|(i, k)| (k.as_slice(), sides & (1 << i) != 0))
                    .collect();
                for new_new in [false, true] {
                    let expected = items.iter().enumerate().any(|(i, (a, an))| {
                        items.iter().skip(i + 1).any(|(b, bn)| {
                            ((*an && *bn && new_new) || an != bn) && c.conflicts(a, b)
                        })
                    });
                    let found = c.group_conflict(&mut items, new_new);
                    assert_eq!(found.is_some(), expected, "{chosen:?} sides {sides:b}");
                    if let Some((a, b)) = found {
                        assert!(c.conflicts(a, b));
                    }
                }
            }
        }
    }

    #[test]
    fn a_large_group_is_swept_not_compared_pairwise() {
        let c = compiled();
        let keys: Vec<Key> = (0..50_000)
            .map(|i| {
                c.key(&row(
                    Some(1),
                    Some(&format!("[{},{})", i * 2, i * 2 + 2)),
                    true,
                ))
                .unwrap()
                .unwrap()
            })
            .collect();
        let refs: Vec<&[ast::Value]> = keys.iter().map(Vec::as_slice).collect();
        let started = std::time::Instant::now();
        assert!(c.conflict_among(&refs).is_none());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let mut more = keys.clone();
        more.push(
            c.key(&row(Some(1), Some("[99,100)"), true))
                .unwrap()
                .unwrap(),
        );
        let refs: Vec<&[ast::Value]> = more.iter().map(Vec::as_slice).collect();
        assert!(c.conflict_among(&refs).is_some());
    }
}
