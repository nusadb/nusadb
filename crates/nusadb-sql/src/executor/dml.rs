//! DML execution: INSERT, UPDATE, DELETE (+ RETURNING projection).
//!
//! Split verbatim out of `executor/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === INSERT ===============================================================

pub(super) fn run_insert(
    plan: &InsertPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // `INSERT ... SELECT` streams its source in bounded batches when that is provably equivalent
    // to the materialized path (P-INSERTSEL-OOM): memory stays O(batch) instead of O(result), so
    // an ETL-sized source no longer trips the `work_mem` guard. Every non-qualifying shape falls
    // through to the materialized path below, which enforces `work_mem` loudly.
    if let InsertSource::Select(select) = &plan.source
        && plan.on_conflict.is_none()
        && plan.returning.is_empty()
    {
        let op = crate::planner::plan_select((**select).clone());
        if insert_select_can_stream(&op, &plan.table, engine, txn)? {
            return insert_select_streaming(&op, plan, engine, txn);
        }
    }
    // Each source produces one value tuple per row, in target-`columns` order.
    let value_rows = insert_value_rows(plan, engine, txn)?;
    // `ON CONFLICT DO UPDATE` upserts; everything else (plain INSERT, `DO NOTHING`) inserts.
    let full_rows = match &plan.on_conflict {
        Some(OnConflictPlan::DoUpdate {
            target,
            assignments,
            filter,
        }) => upsert_rows(
            &plan.table,
            &plan.columns,
            value_rows,
            plan.rls_check.as_ref(),
            target,
            assignments,
            filter.as_ref(),
            engine,
            txn,
        )?,
        other => insert_rows(
            &plan.table,
            &plan.columns,
            value_rows,
            plan.rls_check.as_ref(),
            matches!(other, Some(OnConflictPlan::DoNothing)),
            engine,
            txn,
        )?,
    };

    if plan.returning.is_empty() {
        Ok(ExecutionResult::Inserted(full_rows.len()))
    } else {
        // `RETURNING`: the projection's column ordinals index the table's columns, and each
        // `full` row is in that order. Re-encode + decode each row to its canonical *stored* form first
        // so a coerced type — e.g. a NUMERIC(p,s) rescaled to the column's scale — returns the value
        // actually persisted (what a later SELECT reads), not the un-coerced input. The extra
        // codec round-trip falls only on the RETURNING path, never on a plain INSERT / COPY.
        let schema = column_types(&plan.table);
        let returned = full_rows
            .iter()
            .map(|full| {
                let stored = row::decode(&row::encode(full, &schema)?, &schema)?;
                plan.returning
                    .iter()
                    .map(|p| eval::eval(&p.expr, &stored))
                    .collect::<Result<Row, _>>()
            })
            .collect::<Result<Vec<Row>, _>>()?;
        Ok(ExecutionResult::Rows {
            columns: plan.returning.iter().map(|p| p.name.clone()).collect(),
            rows: returned,
        })
    }
}

/// Fill each column omitted from the INSERT target list (`!covered`) of `full` from its `DEFAULT`
/// expression or `SERIAL` sequence, in place. A `DEFAULT` evaluates against an empty
/// row (it references no other column) and is re-evaluated per row so a volatile default
/// (`now()`/`random()`) is fresh each time; a `SERIAL` advances its sequence once per row. A fill that
/// produces `NULL` on a `NOT NULL` column is a violation, like an explicit `NULL`. Shared by `INSERT`
/// and `INSERT ... ON CONFLICT DO UPDATE`.
fn apply_column_fills(
    full: &mut Row,
    fills: &[Option<super::coldefault::ColumnFill>],
    covered: &HashSet<usize>,
    table: &TableSchema,
    engine: &dyn StorageEngine,
) -> Result<(), Error> {
    let empty_row: Row = Vec::new();
    // Pass 1: fill omitted DEFAULT / SERIAL columns. GENERATED columns are deferred to pass 2 — their
    // expression references the row's other columns, which must all be populated first.
    for (index, (fill, column)) in fills.iter().zip(&table.columns).enumerate() {
        if covered.contains(&index) {
            continue;
        }
        let value = match fill {
            // No default/serial: the column stays NULL. That is a violation on a NOT NULL column —
            // an omitted such column is already rejected by the caller's static pre-check, but a
            // per-cell `DEFAULT` reaches here uncovered (it is in the target list), so guard it too.
            None => {
                if !column.nullable {
                    return Err(Error::NotNullViolation {
                        column: column.name.clone(),
                    });
                }
                continue;
            },
            Some(super::coldefault::ColumnFill::Expr(expr)) => eval::eval(expr, &empty_row)?,
            Some(super::coldefault::ColumnFill::Serial { id, .. }) => {
                ast::Value::Int(engine.sequence_next(*id)?)
            },
            // Computed in pass 2 below, once the columns it references are populated.
            Some(super::coldefault::ColumnFill::Generated(_)) => continue,
        };
        if matches!(value, ast::Value::Null) && !column.nullable {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        set_at(full, index, value)?;
    }
    // Pass 2: compute every GENERATED column against the now-populated row. Always applied
    // (an explicit value was rejected earlier); the generation expression references only non-generated
    // columns, so the order among generated columns does not matter.
    for (index, fill) in fills.iter().enumerate() {
        let Some(super::coldefault::ColumnFill::Generated(expr)) = fill else {
            continue;
        };
        let value = eval::eval(expr, &*full)?;
        if matches!(value, ast::Value::Null)
            && let Some(column) = table.columns.get(index)
            && !column.nullable
        {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        set_at(full, index, value)?;
    }
    Ok(())
}

/// The set of target columns for which at least one row supplies a *concrete* value (a `Some` cell).
/// An explicit `DEFAULT` (`None`) cell does not make its column explicitly supplied — it is treated
/// like an omitted column and filled from the column's default/serial. Used to decide the
/// GENERATED-ALWAYS rejection and the post-insert sequence bump, which must react only to real values.
fn explicitly_supplied_columns(
    columns: &[usize],
    value_rows: &[Vec<Option<ast::Value>>],
) -> HashSet<usize> {
    let mut explicit = HashSet::new();
    for row in value_rows {
        for (value, &col_idx) in row.iter().zip(columns) {
            if value.is_some() {
                explicit.insert(col_idx);
            }
        }
    }
    explicit
}

/// Reject an explicit value for a `GENERATED ALWAYS AS IDENTITY` column (#9a): such a column is always
/// system-generated (NusaDB does not support `OVERRIDING SYSTEM VALUE`), so naming it in the target
/// list is an error. It still auto-fills from its sequence when omitted. `covered` is the set of
/// target-column ordinals the statement supplies a value for.
fn reject_explicit_identity_always(
    table: &TableSchema,
    fills: &[Option<super::coldefault::ColumnFill>],
    covered: &HashSet<usize>,
) -> Result<(), Error> {
    for (index, fill) in fills.iter().enumerate() {
        if covered.contains(&index)
            && matches!(
                fill,
                Some(super::coldefault::ColumnFill::Serial { always: true, .. })
            )
        {
            return Err(Error::Unsupported(format!(
                "column \"{}\" is GENERATED ALWAYS AS IDENTITY and cannot be given an explicit value",
                table.columns.get(index).map_or("", |c| c.name.as_str())
            )));
        }
    }
    Ok(())
}

/// Reject an explicit value for a `GENERATED ALWAYS AS (<expr>) STORED` column: its value is
/// always computed, so naming it in the INSERT target list (or an UPDATE `SET`) is an error. `covered`
/// is the set of column ordinals the statement supplies a concrete value for.
fn reject_explicit_generated(
    table: &TableSchema,
    fills: &[Option<super::coldefault::ColumnFill>],
    covered: &HashSet<usize>,
) -> Result<(), Error> {
    for (index, fill) in fills.iter().enumerate() {
        if covered.contains(&index)
            && matches!(fill, Some(super::coldefault::ColumnFill::Generated(_)))
        {
            return Err(Error::Unsupported(format!(
                "column \"{}\" is a generated column and cannot be given an explicit value",
                table.columns.get(index).map_or("", |c| c.name.as_str())
            )));
        }
    }
    Ok(())
}

/// Recompute every `GENERATED ALWAYS AS (<expr>) STORED` column of `row` against the row's current
/// values — used after an UPDATE's `SET` assignments, since a generated column must reflect the
/// new values of the columns it derives from. Generated columns reference only non-generated columns,
/// so a single pass is correct. A computed `NULL` on a `NOT NULL` column is a violation.
fn recompute_generated(
    mut row: Row,
    fills: &[Option<super::coldefault::ColumnFill>],
    table: &TableSchema,
) -> Result<Row, Error> {
    for (index, fill) in fills.iter().enumerate() {
        let Some(super::coldefault::ColumnFill::Generated(expr)) = fill else {
            continue;
        };
        let value = eval::eval(expr, &row)?;
        if matches!(value, ast::Value::Null)
            && let Some(column) = table.columns.get(index)
            && !column.nullable
        {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        set_at(&mut row, index, value)?;
    }
    Ok(row)
}

/// After inserting `full_rows`, advance each `SERIAL`/`IDENTITY` column's sequence past the largest
/// explicit value any row supplied for it (deep-gate #9b). A row that overrides a serial
/// column with an explicit value would otherwise leave the sequence behind, so the next auto-generated
/// value would collide with the override — fatal for a `SERIAL PRIMARY KEY`. The sequence only moves
/// forward (never below its current value); the advance is non-transactional, matching `nextval`.
fn advance_serials_past_explicit(
    fills: &[Option<super::coldefault::ColumnFill>],
    covered: &HashSet<usize>,
    full_rows: &[Row],
    engine: &dyn StorageEngine,
) -> Result<(), Error> {
    for (index, fill) in fills.iter().enumerate() {
        let Some(super::coldefault::ColumnFill::Serial { id: seq, .. }) = fill else {
            continue;
        };
        // A column filled from the sequence (not in the target list) already advanced it.
        if !covered.contains(&index) {
            continue;
        }
        let Some(max_explicit) = full_rows
            .iter()
            .filter_map(|row| match row.get(index) {
                Some(ast::Value::Int(n)) => Some(*n),
                _ => None,
            })
            .max()
        else {
            continue;
        };
        // `currval` is undefined until the first `nextval`; treat that as "behind any explicit value".
        let behind = engine
            .sequence_current(*seq)
            .map_or(true, |current| max_explicit > current);
        if behind {
            engine.sequence_set(*seq, max_explicit)?;
        }
    }
    Ok(())
}

/// Rows pulled from the source and inserted per batch on the streaming `INSERT ... SELECT` path —
/// the executor's vectorized batch size, bounding the statement's memory to O(batch).
const INSERT_SELECT_BATCH: usize = 1024;

/// Deferred `PRIMARY KEY`/`UNIQUE` enforcement for the streaming `INSERT ... SELECT` path (the
/// PK/UNIQUE-target residual). Auxiliary state is linear in the inserted row count, but holds
/// only each row's **key values + tid** — key-width instead of the full row width the removed
/// materialize-and-bail path buffered (and none of the source's intermediate rows).
///
/// Per batch, [`admit_batch`](Self::admit_batch) takes the no-wait key locks (serializing every
/// concurrent same-key writer, exactly like the immediate path) and checks the batch against the
/// keys this statement already inserted (an intra-statement duplicate fails loudly right there).
/// After the last batch, [`finish`](Self::finish) runs **one** latest-committed scan — with every
/// key lock still held — excluding the statement's own rows by tid: any other committed row
/// holding one of our keys is a duplicate.
///
/// Soundness of the single end-of-stream check: a concurrent writer of key `K` either
/// (a) committed *before* this statement locked `K` — then the final committed scan sees its row
/// and the statement fails with the honest duplicate error, or (b) tried to lock `K` *after* us —
/// then it aborted on our held lock (`40001`). There is no in-between: the lock is taken before
/// the key's row is written and held to transaction end.
struct DeferredUnique {
    /// Per PK/UNIQUE constraint: `(name, kind label, column names, ordinals)`.
    constraints: Vec<(String, &'static str, Vec<String>, Vec<usize>)>,
    /// Per constraint (same order), the statement's inserted keys, bucketed by the shared
    /// [`unique_key_hash`] (collisions only cost a comparison, never correctness).
    seen: Vec<HashMap<u64, Vec<Vec<ast::Value>>>>,
    /// Every tid this statement inserted — excluded from the final committed re-check.
    inserted: HashSet<Tid>,
}

impl DeferredUnique {
    /// Load the target's PK/UNIQUE constraints; a table without any yields an empty (free)
    /// collector.
    fn load(table: &TableSchema, engine: &dyn StorageEngine) -> Result<Self, Error> {
        let constraints: Vec<(String, &'static str, Vec<String>, Vec<usize>)> = engine
            .list_constraints(table.id)?
            .into_iter()
            .filter_map(|c| {
                let kind = match c.kind {
                    nusadb_core::ConstraintKind::PrimaryKey => "primary key",
                    nusadb_core::ConstraintKind::Unique => "unique",
                    _ => return None,
                };
                Some((c.name, kind, c.columns))
            })
            .map(|(name, kind, columns)| {
                constraint_ordinals(table, &columns).map(|ordinals| (name, kind, columns, ordinals))
            })
            .collect::<Result<_, _>>()?;
        let seen = constraints.iter().map(|_| HashMap::new()).collect();
        Ok(Self {
            constraints,
            seen,
            inserted: HashSet::new(),
        })
    }

    /// Lock the batch's keys and fold them into the statement's key state; an intra-statement
    /// duplicate fails here, before the batch is written.
    fn admit_batch(
        &mut self,
        table: &TableSchema,
        rows: &[Row],
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        if self.constraints.is_empty() {
            return Ok(());
        }
        lock_unique_keys(table, rows, engine, txn)?;
        for row in rows {
            for ((name, kind, columns, ordinals), buckets) in
                self.constraints.iter().zip(&mut self.seen)
            {
                let Some(key) = unique_key(row, ordinals) else {
                    continue;
                };
                let bucket = buckets
                    .entry(unique_key_hash(table, columns, &key))
                    .or_default();
                if bucket.iter().any(|prior| unique_key_eq(prior, &key)) {
                    return Err(duplicate_key_error(kind, name, columns));
                }
                bucket.push(key);
            }
        }
        Ok(())
    }

    /// Record a written row's tid so [`finish`](Self::finish) does not count the statement's own
    /// rows as duplicates.
    fn record_inserted(&mut self, tid: Tid) {
        if !self.constraints.is_empty() {
            self.inserted.insert(tid);
        }
    }

    /// The end-of-stream check: one latest-committed scan (every key lock already held); any row
    /// this statement did **not** write whose key is in the statement's set is a duplicate.
    fn finish(
        &self,
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        if self.constraints.is_empty() {
            return Ok(());
        }
        for (tid, row) in scan_table_committed(table, engine, txn)? {
            if self.inserted.contains(&tid) {
                continue;
            }
            for ((name, kind, columns, ordinals), buckets) in
                self.constraints.iter().zip(&self.seen)
            {
                let Some(key) = unique_key(&row, ordinals) else {
                    continue;
                };
                let clash = buckets
                    .get(&unique_key_hash(table, columns, &key))
                    .is_some_and(|bucket| bucket.iter().any(|k| unique_key_eq(k, &key)));
                if clash {
                    return Err(duplicate_key_error(kind, name, columns));
                }
            }
        }
        Ok(())
    }
}

/// The duplicate-key constraint violation, in the exact wording of the immediate path
/// (`enforce_unique_over_rows`).
fn duplicate_key_error(kind: &str, name: &str, columns: &[String]) -> Error {
    nusadb_core::Error::ConstraintViolation(format!(
        "duplicate key violates {} constraint \"{}\" on ({})",
        kind,
        name,
        columns.join(", "),
    ))
    .into()
}

/// Whether this `INSERT ... SELECT` may stream its source in batches instead of materializing it.
///
/// Streaming interleaves reads of the source with writes to the target, so it must be provably
/// equivalent to the materialized path. That holds only when **all** of:
///
/// - The source pipeline is a whitelist of truly streaming operators (seq-scan / filter /
///   project / limit — exactly what [`stream_op`](super::stream::stream_op) streams without
///   materializing; joins, aggregates, sorts, SRF projections, and set operations buffer
///   internally and are excluded) — **none of whose scans read the target table**, and none of
///   whose expressions carry a subquery (a subquery re-scans at eval time). A single
///   already-open scan of the target would be Halloween-safe (the version list is snapshotted at
///   scan open), but re-scans (join inner, subquery) would observe our own fresh inserts; ruling
///   the target out entirely keeps the proof local.
/// - The target is not an FK child (parent re-scan per batch), has no triggers (statement-level
///   triggers must fire exactly once), and feeds no IVM view (per-batch deltas unverified).
///
/// A `PRIMARY KEY`/`UNIQUE` target streams too (the PK/UNIQUE-target residual): enforcement is
/// **deferred** — per-batch key locks + O(inserted keys) state, one committed-visibility scan at
/// the end of the stream (see [`DeferredUnique`] for the soundness argument).
///
/// `CHECK`, `NOT NULL`, defaults/serials, generated columns, and RLS `WITH CHECK` are all
/// per-row and stay enforced per batch.
fn insert_select_can_stream(
    op: &PhysicalOperator,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    if !stream_safe_source(op, table.id) {
        return Ok(false);
    }
    if engine
        .list_foreign_keys(table.id)?
        .iter()
        .any(|fk| fk.child_table == table.id)
    {
        return Ok(false);
    }
    let triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Insert, engine, txn)?;
    if !triggers.is_empty() {
        return Ok(false);
    }
    if super::ivm::has_views_for_base(engine, txn, &table.name)? {
        return Ok(false);
    }
    Ok(true)
}

/// Whether `op` is a pipeline of truly streaming operators that never reads `target` and carries
/// no subquery (see [`insert_select_can_stream`]). The whitelist is exactly the set
/// [`stream_op`](super::stream::stream_op) streams without materializing — including the lazy
/// literal-integer `generate_series` source — anything else (joins, aggregates, sorts, general
/// SRF projections, …) it buffers via `execute_op`, so streaming them here would buy nothing and
/// dodge the `work_mem` guard. Fails closed on operators outside the list.
fn stream_safe_source(op: &PhysicalOperator, target: nusadb_core::TableId) -> bool {
    use super::ops::contains_subquery;
    match op {
        PhysicalOperator::SeqScan { table, .. } => table.id != target,
        // A literal integer `generate_series` source streams lazily (see
        // [`stream::lazy_int_series`]) — no table read at all.
        PhysicalOperator::ProjectSet { .. } => super::stream::lazy_int_series(op).is_some(),
        PhysicalOperator::Filter { input, predicate } => {
            !contains_subquery(predicate) && stream_safe_source(input, target)
        },
        PhysicalOperator::Project { input, columns } => {
            columns.iter().all(|p| !contains_subquery(&p.expr)) && stream_safe_source(input, target)
        },
        PhysicalOperator::Limit { input, .. } => stream_safe_source(input, target),
        _ => false,
    }
}

/// Stream an `INSERT ... SELECT` source into the target in [`INSERT_SELECT_BATCH`]-row batches
/// (P-INSERTSEL-OOM): pull a batch from the streaming source, push it through [`insert_rows`],
/// repeat. Memory is O(batch) instead of O(result), so a multi-million-row backfill no longer
/// trips the `work_mem` guard. Only reached when [`insert_select_can_stream`] proved per-batch
/// insertion equivalent to whole-statement insertion.
fn insert_select_streaming(
    op: &PhysicalOperator,
    plan: &InsertPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let mut source = super::stream::stream_op(op, engine, txn)?;
    let mut inserted = 0_usize;
    // PK/UNIQUE targets stream with deferred enforcement (free for a table without any).
    let mut unique = DeferredUnique::load(&plan.table, engine)?;
    let mut batch: Vec<Vec<Option<ast::Value>>> = Vec::with_capacity(INSERT_SELECT_BATCH);
    let mut flush = |batch: &mut Vec<Vec<Option<ast::Value>>>,
                     unique: &mut DeferredUnique|
     -> Result<(), Error> {
        let rows = std::mem::replace(batch, Vec::with_capacity(INSERT_SELECT_BATCH));
        inserted += insert_rows_with_unique(
            &plan.table,
            &plan.columns,
            rows,
            plan.rls_check.as_ref(),
            false,
            engine,
            txn,
            Some(unique),
        )?
        .len();
        Ok(())
    };
    while let Some(row) = source.try_next()? {
        // Cooperative cancellation: honor a statement timeout / cancel request at row
        // granularity, exactly as the buffered scan path does.
        crate::cancel::check()?;
        batch.push(row.into_iter().map(Some).collect());
        if batch.len() >= INSERT_SELECT_BATCH {
            flush(&mut batch, &mut unique)?;
        }
    }
    if !batch.is_empty() {
        flush(&mut batch, &mut unique)?;
    }
    // Every key lock is held; one committed-visibility pass closes the concurrent-committer
    // window (see `DeferredUnique`).
    unique.finish(&plan.table, engine, txn)?;
    Ok(ExecutionResult::Inserted(inserted))
}

/// Produce the rows to insert, one value tuple per row in target-`columns` order.
///
/// `VALUES` evaluates each typed expression against an empty input row; `INSERT ... SELECT`
/// lowers the analyzed subquery and runs it, taking its output rows. The subquery runs in the same
/// transaction `txn`, so it observes the table's pre-INSERT snapshot.
fn insert_value_rows(
    plan: &InsertPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Vec<Option<ast::Value>>>, Error> {
    match &plan.source {
        InsertSource::Values(rows) => {
            let empty_row: Row = Vec::new();
            // Resolve any *uncorrelated* subquery in a VALUES cell to a literal before evaluating it
            // (e.g. `INSERT INTO t VALUES (1, (SELECT max(v) FROM other))`). VALUES cells reference no
            // columns, so defer-correlated keeps a (meaningless) correlated subquery in place to be
            // rejected at eval rather than mis-resolved against the empty row.
            let _defer = super::ops::defer_correlated(true);
            rows.iter()
                .map(|row_exprs| {
                    row_exprs
                        .iter()
                        // A `None` cell is an explicit `DEFAULT`: keep it `None` so `insert_rows`
                        // fills it from the target column's default/serial/NULL, exactly as it
                        // does for a column omitted from the target list.
                        .map(|cell| match cell {
                            None => Ok(None),
                            // The common no-subquery / no-sequence case evaluates directly (no
                            // clone); only a cell carrying a subquery or a sequence built-in is
                            // cloned + resolved first. Each VALUES tuple is evaluated exactly once,
                            // so an advancing `nextval`/`setval` advances once per tuple — the
                            // correct N advances for N tuples.
                            Some(expr)
                                if super::ops::contains_subquery(expr)
                                    || super::ops::contains_sequence_call(expr) =>
                            {
                                let mut resolved = expr.clone();
                                super::ops::resolve_subqueries(&mut resolved, engine, txn)?;
                                super::ops::resolve_sequence_calls(&mut resolved, engine)?;
                                eval::eval(&resolved, &empty_row).map(Some)
                            },
                            Some(expr) => eval::eval(expr, &empty_row).map(Some),
                        })
                        .collect::<Result<Vec<Option<ast::Value>>, _>>()
                })
                .collect()
        },
        InsertSource::Select(select) => {
            // Mirror `lower.rs`'s top-level routing estimate: prefer the exact ANALYZE row count,
            // then fall back to the engine's O(1) approximate count, so `INSERT ... SELECT` from a
            // large un-analyzed source still routes to the vectorized path. Routing hint only —
            // the vectorized and row paths produce identical results, so a wrong estimate can only
            // mis-route, never corrupt.
            let est_scan_rows = select
                .table_stats
                .as_ref()
                .map(|s| s.row_count)
                .or(select.approx_scan_rows);
            let op = crate::planner::plan_select((**select).clone());
            match run_select(&op, est_scan_rows, engine, txn)? {
                // A subquery never produces `DEFAULT`; every output value is concrete.
                ExecutionResult::Rows { rows, .. } => Ok(rows
                    .into_iter()
                    .map(|r| r.into_iter().map(Some).collect())
                    .collect()),
                // `run_select` always yields `Rows`; anything else is an internal invariant break.
                _ => Err(Error::Unsupported(
                    "internal: INSERT ... SELECT subquery did not produce a row set".to_owned(),
                )),
            }
        },
    }
}

/// Insert `value_rows` (each in target-`columns` order) into `table`, returning the inserted
/// full-width rows. Shared by `INSERT` and `COPY FROM`: NOT-NULL coverage check, build the full
/// table-width rows, enforce UNIQUE/FK over the whole batch (against existing + the new rows), then
/// encode + insert each tuple and maintain secondary indexes.
pub(super) fn insert_rows(
    table: &TableSchema,
    columns: &[usize],
    value_rows: Vec<Vec<Option<ast::Value>>>,
    rls_check: Option<&crate::planner::TypedExpr>,
    conflict_do_nothing: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    insert_rows_with_unique(
        table,
        columns,
        value_rows,
        rls_check,
        conflict_do_nothing,
        engine,
        txn,
        None,
    )
}

/// [`insert_rows`] with the uniqueness mode explicit: `None` validates against the committed
/// table now (every whole-statement path); `Some(collector)` is the streaming `INSERT ... SELECT`
/// path — per-batch key locks + key collection now, one committed re-scan at end of stream
/// ([`DeferredUnique::finish`]).
#[allow(
    clippy::too_many_arguments,
    reason = "insert_rows' shared batch context plus the one uniqueness-mode knob"
)]
fn insert_rows_with_unique(
    table: &TableSchema,
    columns: &[usize],
    value_rows: Vec<Vec<Option<ast::Value>>>,
    rls_check: Option<&crate::planner::TypedExpr>,
    conflict_do_nothing: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
    mut deferred: Option<&mut DeferredUnique>,
) -> Result<Vec<Row>, Error> {
    // Column DEFAULTs / SERIAL: a column omitted from the target list — or written as
    // an explicit `DEFAULT` cell (`None`) — is filled by its default expression or its sequence (if
    // any) rather than NULL. Loaded once for the batch.
    let fills = super::coldefault::column_fills(table, engine, txn)?;
    // Columns the statement supplies a *concrete* value for in at least one row (an explicit `DEFAULT`
    // cell does not count): the basis for the GENERATED-ALWAYS reject and the post-insert serial bump.
    let any_explicit = explicitly_supplied_columns(columns, &value_rows);
    reject_explicit_identity_always(table, &fills, &any_explicit)?;
    reject_explicit_generated(table, &fills, &any_explicit)?;
    // Static NOT-NULL coverage: a non-nullable column missing from the target list AND without a
    // default/serial fill would be written as NULL. A `DEFAULT` cell on such a column is caught
    // per-row by `apply_column_fills` below.
    let covered: HashSet<usize> = columns.iter().copied().collect();
    for ((index, column), fill) in table.columns.iter().enumerate().zip(&fills) {
        if !column.nullable && !covered.contains(&index) && fill.is_none() {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
    }

    let schema = column_types(table);
    // Build every full row first so PRIMARY KEY / UNIQUE constraints validate the whole batch
    // before any tuple is written.
    let mut full_rows: Vec<Row> = Vec::with_capacity(value_rows.len());
    for values in value_rows {
        if values.len() != columns.len() {
            return Err(Error::ArityMismatch {
                context: "INSERT".to_owned(),
                expected: columns.len(),
                found: values.len(),
            });
        }
        let mut full = vec![ast::Value::Null; table.columns.len()];
        // A `DEFAULT` (`None`) cell leaves its column uncovered for this row so the per-row fill
        // below supplies the default/serial/NULL — exactly as for an omitted column.
        let mut row_covered: HashSet<usize> = HashSet::with_capacity(columns.len());
        for (value, &col_idx) in values.into_iter().zip(columns) {
            if let Some(value) = value {
                set_at(&mut full, col_idx, value)?;
                row_covered.insert(col_idx);
            }
        }
        apply_column_fills(&mut full, &fills, &row_covered, table, engine)?;
        full_rows.push(full);
    }
    // Row-level security WITH CHECK: every row a non-superuser writes must satisfy the
    // applicable policies. Checked before any tuple is written, so a violation aborts the whole
    // INSERT (the transaction rolls back). `None` for a superuser / RLS-free table.
    if let Some(check) = rls_check {
        for full in &full_rows {
            if !predicate_matches(Some(check), full)? {
                return Err(Error::RlsCheckViolation {
                    table: table.name.clone(),
                });
            }
        }
    }
    // `ON CONFLICT DO NOTHING`: drop the rows that would violate a UNIQUE/PRIMARY KEY
    // constraint (against existing rows or an earlier row in this batch) so the rest still insert.
    let full_rows = if conflict_do_nothing {
        keep_non_conflicting(table, full_rows, engine, txn)?
    } else {
        full_rows
    };
    // Triggers: load the INSERT triggers once, fire statement- and row-level BEFORE triggers,
    // then (after the writes) the AFTER triggers. Firing happens here so `COPY FROM` triggers too.
    let triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Insert, engine, txn)?;
    triggers.fire_stmt_before(table, engine, txn)?;
    if triggers.has_before_row() {
        for full in &full_rows {
            triggers.fire_row_before(table, None, Some(full), engine, txn)?;
        }
    }

    match deferred.as_deref_mut() {
        Some(collector) => collector.admit_batch(table, &full_rows, engine, txn)?,
        None => enforce_unique_on_insert(table, &full_rows, engine, txn)?,
    }
    enforce_fk_on_child_write(table, &full_rows, engine, txn)?;
    enforce_check_on_write(table, &full_rows, engine)?;

    let index_targets = secondary_index_targets(table, engine)?;
    // A bulk insert — COPY or a streamed INSERT ... SELECT, the callers that pass a cross-batch
    // uniqueness collector — builds each secondary index by handing all of the batch's entries to
    // `index_insert_batch`, which applies them in key order (sequential index writes) rather than a
    // random one per row. The flush runs below, before the serial bump / AFTER-row triggers / view
    // maintenance, so each of those sees the index exactly as the per-row path would leave it. A
    // plain insert keeps the immediate per-row maintenance.
    let defer_index_build = deferred.is_some();
    let mut deferred_tids: Vec<Tid> = Vec::new();
    for full in &full_rows {
        let bytes = row::encode(full, &schema)?;
        let tid = engine.insert(txn, table.id, &bytes)?;
        if let Some(collector) = deferred.as_deref_mut() {
            collector.record_inserted(tid);
        }
        if defer_index_build {
            deferred_tids.push(tid);
        } else {
            insert_into_indexes(&index_targets, full, tid, engine, txn)?;
        }
    }
    if defer_index_build {
        // Group by index, matching `insert_into_indexes`'s per-row filtering (partial-index
        // predicate, functional-key evaluation), then apply each index's entries in one sorted call.
        for target in &index_targets {
            let mut entries: Vec<(Vec<u8>, Tid)> = Vec::with_capacity(deferred_tids.len());
            for (full, &tid) in full_rows.iter().zip(&deferred_tids) {
                if row_is_indexed(target, full)? {
                    entries.push((index_key_for(full, &target.keys)?, tid));
                }
            }
            engine.index_insert_batch(txn, target.id, entries)?;
        }
    }
    // A row that supplied an explicit value for a SERIAL column must push its sequence forward so a
    // later auto-generated value cannot collide (deep-gate #9b). A `DEFAULT` cell is not explicit —
    // it already advanced the sequence via `apply_column_fills` — so `any_explicit`, not `covered`.
    advance_serials_past_explicit(&fills, &any_explicit, &full_rows, engine)?;

    if triggers.has_after_row() {
        for full in &full_rows {
            triggers.fire_row_after(table, None, Some(full), engine, txn)?;
        }
    }
    triggers.fire_stmt_after(table, engine, txn)?;
    // Incremental view maintenance: append the projected rows to any IVM view over this
    // table.
    super::ivm::maintain_on_change(&table.name, &full_rows, &[], engine, txn)?;
    Ok(full_rows)
}

/// Execute `INSERT ... ON CONFLICT (target) DO UPDATE SET ... [WHERE ...]` — the upsert.
///
/// Each proposed row that collides (on the arbiter's `PRIMARY KEY`/`UNIQUE` key) with an existing
/// row updates that row instead of inserting; a non-colliding row inserts normally. The `SET` values
/// and the optional `WHERE` are evaluated against the concatenated `existing ++ EXCLUDED` row (the
/// existing row's columns at `[0, n)`, the proposed row at `[n, 2n)`), so `excluded.col` reads the
/// proposed value and a bare column reads the existing one. Returns every affected row (final
/// values), in proposal order, for `RETURNING` and the affected-row count.
///
/// `UNIQUE`/`PRIMARY KEY` is enforced over the full post-statement state when any row is updated (a
/// `DO UPDATE` may move a unique key onto another row), matching plain `UPDATE`. A subquery
/// in the `SET`/`WHERE` is rejected at analysis time.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one cohesive upsert pass: build rows, probe conflicts, enforce, then apply"
)]
fn upsert_rows(
    table: &TableSchema,
    columns: &[usize],
    value_rows: Vec<Vec<Option<ast::Value>>>,
    rls_check: Option<&crate::planner::TypedExpr>,
    target: &ConflictArbiter,
    assignments: &[(usize, crate::planner::TypedExpr)],
    filter: Option<&crate::planner::TypedExpr>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    // Column DEFAULT / SERIAL fills for a column omitted from the proposed insert or
    // written as an explicit `DEFAULT` cell.
    let fills = super::coldefault::column_fills(table, engine, txn)?;
    // Columns supplied a concrete value by some row (an explicit `DEFAULT` cell does not count).
    let any_explicit = explicitly_supplied_columns(columns, &value_rows);
    reject_explicit_identity_always(table, &fills, &any_explicit)?;
    reject_explicit_generated(table, &fills, &any_explicit)?;
    // The DO UPDATE may not SET a generated column either; it is recomputed after the other SETs below.
    reject_explicit_generated(
        table,
        &fills,
        &assignments.iter().map(|(ordinal, _)| *ordinal).collect(),
    )?;
    // Any non-nullable column missing from the target list AND without a default/serial fill would be
    // written as NULL on insert; a `DEFAULT` cell on such a column is caught per-row below.
    let covered: HashSet<usize> = columns.iter().copied().collect();
    for ((index, column), fill) in table.columns.iter().enumerate().zip(&fills) {
        if !column.nullable && !covered.contains(&index) && fill.is_none() {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
    }
    // Materialize the proposed rows in full-table layout.
    let mut proposed: Vec<Row> = Vec::with_capacity(value_rows.len());
    for values in value_rows {
        if values.len() != columns.len() {
            return Err(Error::ArityMismatch {
                context: "INSERT".to_owned(),
                expected: columns.len(),
                found: values.len(),
            });
        }
        let mut full = vec![ast::Value::Null; table.columns.len()];
        // A `DEFAULT` (`None`) cell leaves its column uncovered for this row so the per-row fill
        // supplies the default/serial/NULL, as for an omitted column.
        let mut row_covered: HashSet<usize> = HashSet::with_capacity(columns.len());
        for (value, &col_idx) in values.into_iter().zip(columns) {
            if let Some(value) = value {
                set_at(&mut full, col_idx, value)?;
                row_covered.insert(col_idx);
            }
        }
        apply_column_fills(&mut full, &fills, &row_covered, table, engine)?;
        proposed.push(full);
    }

    let key_ordinals = resolve_arbiter(table, target, engine)?;
    let existing = scan_table(table, engine, txn)?;
    let schema = column_types(table);
    let index_targets = secondary_index_targets(table, engine)?;

    // Each conflicting row's `(tid, old image, new image)` — the old image feeds UPDATE triggers
    // (`OLD.col`) and the delete side of IVM.
    let mut updates: Vec<(Tid, Row, Row)> = Vec::new();
    let mut inserts: Vec<Row> = Vec::new();
    let mut affected: Vec<Row> = Vec::new();
    // Keys already updated/inserted this statement — a second proposed row on the same key is an
    // error (a single upsert may not affect one row twice), matching the standard upsert semantics.
    let mut affected_keys: Vec<Vec<ast::Value>> = Vec::new();

    for prow in proposed {
        let Some(key) = unique_key(&prow, &key_ordinals) else {
            // A NULL arbiter key never collides (NULLs are distinct) → plain insert.
            inserts.push(prow.clone());
            affected.push(prow);
            continue;
        };
        if affected_keys.iter().any(|seen| unique_key_eq(seen, &key)) {
            return Err(nusadb_core::Error::ConstraintViolation(format!(
                "ON CONFLICT DO UPDATE on \"{}\" cannot affect one row a second time",
                table.name
            ))
            .into());
        }
        let hit = existing.iter().find(|(_, erow)| {
            unique_key(erow, &key_ordinals).is_some_and(|ekey| unique_key_eq(&ekey, &key))
        });
        if let Some((tid, erow)) = hit {
            // Evaluate SET / WHERE against `existing ++ EXCLUDED(proposed)`.
            let mut combined = erow.clone();
            combined.extend(prow.iter().cloned());
            if let Some(predicate) = filter
                && !matches!(eval::eval(predicate, &combined)?, ast::Value::Bool(true))
            {
                continue; // WHERE not satisfied → leave the existing row untouched, insert nothing.
            }
            let mut new_row = erow.clone();
            for (ordinal, expr) in assignments {
                let value = eval::eval(expr, &combined)?;
                set_at(&mut new_row, *ordinal, value)?;
            }
            // Recompute generated columns against the updated row, like plain UPDATE.
            let new_row = recompute_generated(new_row, &fills, table)?;
            affected_keys.push(key);
            updates.push((*tid, erow.clone(), new_row.clone()));
            affected.push(new_row);
        } else {
            affected_keys.push(key);
            inserts.push(prow.clone());
            affected.push(prow);
        }
    }

    // Row-level security WITH CHECK: every affected (inserted or updated) row must satisfy the
    // applicable policies. Checked before any write, so a violation aborts the whole statement.
    if let Some(check) = rls_check {
        for row in &affected {
            if !predicate_matches(Some(check), row)? {
                return Err(Error::RlsCheckViolation {
                    table: table.name.clone(),
                });
            }
        }
    }
    // UNIQUE: when any row is updated, a `DO UPDATE` may move a (secondary or arbiter) unique key
    // onto another row, so validate the *whole* post-statement table state — exactly as plain
    // UPDATE does. For an insert-only upsert the cheaper inserts-vs-existing check suffices.
    if updates.is_empty() {
        enforce_unique_on_insert(table, &inserts, engine, txn)?;
    } else {
        let updated_tids: HashSet<Tid> = updates.iter().map(|(tid, _, _)| *tid).collect();
        // The rows actually written this statement: each updated row's new image + each inserted row.
        let written: Vec<Row> = updates
            .iter()
            .map(|(_, _, row)| row.clone())
            .chain(inserts.iter().cloned())
            .collect();
        // Serialize concurrent same-key writers before the snapshot scan (A-QA1d): a
        // `DO UPDATE` matched arm that moves a key, like INSERT/UPDATE/MERGE, needs the no-wait key
        // lock or two overlapping upserts could both commit a duplicate.
        lock_unique_keys(table, &written, engine, txn)?;
        let old_images: Vec<Row> = updates.iter().map(|(_, old, _)| old.clone()).collect();
        // Fast path: probe each backing index for the new keys (latest-committed, O(log n)); fall back
        // to the whole-table snapshot check + the latest-committed cross-transaction check when a
        // constraint is not probe-eligible.
        if !try_update_unique_by_index_probe(
            table,
            &updated_tids,
            &old_images,
            &written,
            engine,
            txn,
        )? {
            // Whole post-statement snapshot check — catches an in-statement key move.
            let mut full_result: Vec<Row> = existing
                .iter()
                .filter(|(tid, _)| !updated_tids.contains(tid))
                .map(|(_, row)| row.clone())
                .collect();
            full_result.extend(written.iter().cloned());
            enforce_unique_over_rows(table, &full_result, engine)?;
            enforce_new_keys_vs_committed(
                table,
                &updated_tids,
                &old_images,
                &written,
                engine,
                txn,
            )?;
        }
    }
    // Every affected row validates FOREIGN KEY and CHECK.
    enforce_fk_on_child_write(table, &affected, engine, txn)?;
    enforce_check_on_write(table, &affected, engine)?;

    // Triggers: an upsert fires INSERT triggers for the inserted rows and UPDATE triggers for
    // the conflicting rows it updates (`OLD`→`NEW`), so audit/maintenance triggers are not bypassed.
    // BEFORE triggers fire before the writes; AFTER triggers after.
    let insert_triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Insert, engine, txn)?;
    let update_triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Update, engine, txn)?;
    if !inserts.is_empty() {
        insert_triggers.fire_stmt_before(table, engine, txn)?;
    }
    if !updates.is_empty() {
        update_triggers.fire_stmt_before(table, engine, txn)?;
    }
    if insert_triggers.has_before_row() {
        for row in &inserts {
            insert_triggers.fire_row_before(table, None, Some(row), engine, txn)?;
        }
    }
    if update_triggers.has_before_row() {
        for (_, old, new) in &updates {
            update_triggers.fire_row_before(table, Some(old), Some(new), engine, txn)?;
        }
    }

    // Apply: update the conflicting rows in place, then insert the rest.
    for (tid, old_row, new_row) in &updates {
        let bytes = row::encode(new_row, &schema)?;
        let new_tid = engine.update(txn, table.id, *tid, &bytes)?;
        if !index_targets.is_empty() {
            remove_departed_index_entries(&index_targets, old_row, *tid, new_row, engine, txn)?;
            insert_into_indexes(&index_targets, new_row, new_tid, engine, txn)?;
        }
    }
    for row in &inserts {
        let bytes = row::encode(row, &schema)?;
        let tid = engine.insert(txn, table.id, &bytes)?;
        insert_into_indexes(&index_targets, row, tid, engine, txn)?;
    }
    // An inserted row that supplied an explicit SERIAL value advances its sequence too (deep-gate #9b),
    // matching the plain INSERT path.
    advance_serials_past_explicit(&fills, &covered, &inserts, engine)?;

    if insert_triggers.has_after_row() {
        for row in &inserts {
            insert_triggers.fire_row_after(table, None, Some(row), engine, txn)?;
        }
    }
    if update_triggers.has_after_row() {
        for (_, old, new) in &updates {
            update_triggers.fire_row_after(table, Some(old), Some(new), engine, txn)?;
        }
    }
    if !inserts.is_empty() {
        insert_triggers.fire_stmt_after(table, engine, txn)?;
    }
    if !updates.is_empty() {
        update_triggers.fire_stmt_after(table, engine, txn)?;
    }

    // Incremental view maintenance: the upsert's delta is the inserted rows plus the new
    // image of each updated row on the insert side, and the old image of each updated row on the
    // delete side — so a materialized view over this table stays consistent.
    let new_rows: Vec<Row> = updates
        .iter()
        .map(|(_, _, new)| new.clone())
        .chain(inserts.iter().cloned())
        .collect();
    let old_rows: Vec<Row> = updates.iter().map(|(_, old, _)| old.clone()).collect();
    super::ivm::maintain_on_change(&table.name, &new_rows, &old_rows, engine, txn)?;
    Ok(affected)
}

/// Resolve an `ON CONFLICT` arbiter to the key-column ordinals of the matching `PRIMARY KEY`/`UNIQUE`
/// constraint. Errors if no such constraint exists — an upsert needs a unique arbiter so that
/// at most one existing row collides.
fn resolve_arbiter(
    table: &TableSchema,
    target: &ConflictArbiter,
    engine: &dyn StorageEngine,
) -> Result<Vec<usize>, Error> {
    let constraints: Vec<_> = engine
        .list_constraints(table.id)?
        .into_iter()
        .filter(|c| {
            matches!(
                c.kind,
                nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
            )
        })
        .collect();
    match target {
        ConflictArbiter::Constraint(name) => {
            let constraint = constraints.iter().find(|c| &c.name == name).ok_or_else(|| {
                nusadb_core::Error::ConstraintViolation(format!(
                    "ON CONFLICT ON CONSTRAINT \"{name}\": no unique or primary key constraint with \
                     that name on \"{}\"",
                    table.name
                ))
            })?;
            constraint_ordinals(table, &constraint.columns)
        },
        ConflictArbiter::Columns(ordinals) => {
            let mut wanted = ordinals.clone();
            wanted.sort_unstable();
            for constraint in &constraints {
                let mut got = constraint_ordinals(table, &constraint.columns)?;
                got.sort_unstable();
                if got == wanted {
                    return constraint_ordinals(table, &constraint.columns);
                }
            }
            Err(nusadb_core::Error::ConstraintViolation(format!(
                "ON CONFLICT target columns do not match any unique or primary key constraint on \"{}\"",
                table.name
            ))
            .into())
        },
    }
}

/// Execute `COPY <table> FROM STDIN`: resolve the target columns, tokenize the text-format
/// `data` into rows, parse each field into a value of the column's type, and bulk-insert them all
/// under `txn`. Returns the number of rows inserted. The whole load is one transaction at the
/// caller's level — a single bad row aborts it (all-or-nothing), matching a non-streamed batch.
pub(super) fn run_copy_from(
    copy: &ast::Copy,
    data: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<usize, Error> {
    let table = engine
        .lookup_table_as_of(txn, &copy.table)?
        .ok_or_else(|| Error::TableNotFound {
            name: copy.table.clone(),
        })?;
    let columns = resolve_copy_columns(copy, &table)?;
    let target_types: Vec<ColumnType> = columns
        .iter()
        .filter_map(|&i| table.columns.get(i).map(|c| c.ty))
        .collect();

    // Stream the payload in INSERT_SELECT_BATCH-row batches rather than materializing every parsed
    // row up front: a multi-million-row COPY otherwise builds a Vec holding
    // every row's decoded `ast::Value`s — gigabytes — and OOM-kills the whole server before the
    // engine's per-transaction write ceiling can reject it. Per-batch insertion keeps executor
    // memory at O(batch), and each insert is charged against that ceiling, so an oversized COPY now
    // aborts loudly (like a giant INSERT/UPDATE/DELETE) instead of exhausting memory. Uniqueness is
    // enforced with the same deferred mechanism `INSERT ... SELECT` streaming uses — a naive
    // per-batch immediate check would not see prior batches' still-uncommitted keys.
    let mut inserted = 0_usize;
    let mut unique = DeferredUnique::load(&table, engine)?;
    let mut batch: Vec<Vec<Option<ast::Value>>> = Vec::with_capacity(INSERT_SELECT_BATCH);
    let mut flush = |batch: &mut Vec<Vec<Option<ast::Value>>>,
                     unique: &mut DeferredUnique|
     -> Result<(), Error> {
        let rows = std::mem::replace(batch, Vec::with_capacity(INSERT_SELECT_BATCH));
        // COPY runs only for a superuser on an RLS table (the wire layer refuses it otherwise), so
        // there is no per-row WITH CHECK to apply here.
        inserted += insert_rows_with_unique(
            &table,
            &columns,
            rows,
            None,
            false,
            engine,
            txn,
            Some(unique),
        )?
        .len();
        Ok(())
    };
    for (line_no, line) in copy_data_lines(data, copy.format.header).enumerate() {
        // Honor a statement timeout / cancel request at row granularity on a long load.
        crate::cancel::check()?;
        let fields = crate::copy::parse_text_row(line, copy.format.delimiter, &copy.format.null);
        if fields.len() != columns.len() {
            return Err(Error::ArityMismatch {
                context: format!("COPY data line {}", line_no + 1),
                expected: columns.len(),
                found: fields.len(),
            });
        }
        let mut row = Vec::with_capacity(fields.len());
        for (field, &ty) in fields.into_iter().zip(&target_types) {
            // COPY names every column explicitly; an empty/NULL field is a concrete NULL, never a
            // `DEFAULT` cell — so each value is `Some`.
            row.push(Some(match field {
                None => ast::Value::Null,
                Some(text) => parse_copy_field(&text, ty)?,
            }));
        }
        batch.push(row);
        if batch.len() >= INSERT_SELECT_BATCH {
            flush(&mut batch, &mut unique)?;
        }
    }
    if !batch.is_empty() {
        flush(&mut batch, &mut unique)?;
    }
    // Every inserted key lock is held; one committed-visibility pass closes the concurrent-committer
    // window (see `DeferredUnique`).
    unique.finish(&table, engine, txn)?;
    Ok(inserted)
}

/// Execute `COPY <table> TO STDOUT`: render the table's visible rows in the text format,
/// returning the row count and the rendered payload (newline-terminated lines). A leading header
/// line of column names is emitted when `WITH (HEADER)` is set.
pub(super) fn run_copy_to(
    copy: &ast::Copy,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(usize, String), Error> {
    let table = engine
        .lookup_table_as_of(txn, &copy.table)?
        .ok_or_else(|| Error::TableNotFound {
            name: copy.table.clone(),
        })?;
    let columns = resolve_copy_columns(copy, &table)?;
    let rows = scan_rows(&table, engine, txn)?;

    let delimiter = copy.format.delimiter;
    let null = &copy.format.null;
    let mut out = String::new();
    if copy.format.header {
        let names: Vec<Option<&str>> = columns
            .iter()
            .map(|&i| table.columns.get(i).map(|c| c.name.as_str()))
            .collect();
        out.push_str(&crate::copy::format_text_row(&names, delimiter, null));
        out.push('\n');
    }
    for row in &rows {
        let fields: Vec<Option<String>> = columns
            .iter()
            .map(|&i| match row.get(i) {
                Some(ast::Value::Null) | None => None,
                Some(value) => Some(crate::display::value_text(value)),
            })
            .collect();
        let refs: Vec<Option<&str>> = fields.iter().map(Option::as_deref).collect();
        out.push_str(&crate::copy::format_text_row(&refs, delimiter, null));
        out.push('\n');
    }
    Ok((rows.len(), out))
}

/// Resolve a `COPY`'s column list to table ordinals (empty = all columns, table order).
fn resolve_copy_columns(copy: &ast::Copy, table: &TableSchema) -> Result<Vec<usize>, Error> {
    if copy.columns.is_empty() {
        return Ok((0..table.columns.len()).collect());
    }
    let mut seen = HashSet::new();
    let mut indices = Vec::with_capacity(copy.columns.len());
    for name in &copy.columns {
        if !seen.insert(name.as_str()) {
            return Err(Error::DuplicateColumn { name: name.clone() });
        }
        let index = table
            .columns
            .iter()
            .position(|c| c.name == *name)
            .ok_or_else(|| Error::ColumnNotFound {
                column: name.clone(),
                table: table.name.clone(),
            })?;
        indices.push(index);
    }
    Ok(indices)
}

/// Split COPY data into non-empty data lines, optionally skipping a leading header line. A trailing
/// `\r` (CRLF) is trimmed; a final line without a newline is still yielded.
fn copy_data_lines(data: &str, header: bool) -> impl Iterator<Item = &str> {
    data.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.is_empty())
        .skip(usize::from(header))
}

/// Parse one COPY text field into a value of `ty`. `INT`/`FLOAT`/`BOOL` are parsed here; the other
/// types are carried as text and parsed at encode time (the same coercion `INSERT` relies on).
fn parse_copy_field(text: &str, ty: ColumnType) -> Result<ast::Value, Error> {
    let invalid = || Error::InvalidValue {
        ty,
        value: text.to_owned(),
    };
    // Match on the storage type so a declared alias parses like its base type:
    // BIGINT/SMALLINT are stored as Int and REAL as Float — leaving them to the Text fallback made
    // `COPY` into such a column fail at encode ("expected Int, found Text") while INSERT worked.
    match ty.physical() {
        ColumnType::Int => text
            .trim()
            .parse::<i64>()
            .map(ast::Value::Int)
            .map_err(|_| invalid()),
        ColumnType::Float => text
            .trim()
            .parse::<f64>()
            .map(ast::Value::Float)
            .map_err(|_| invalid()),
        ColumnType::Bool => match text.trim().to_ascii_lowercase().as_str() {
            "t" | "true" | "1" | "yes" | "on" => Ok(ast::Value::Bool(true)),
            "f" | "false" | "0" | "no" | "off" => Ok(ast::Value::Bool(false)),
            _ => Err(invalid()),
        },
        // Text, and every type whose `assignable` rule accepts a text value (numeric / temporal /
        // uuid / json / array / interval), are carried as text and coerced by `row::encode`.
        _ => Ok(ast::Value::Text(text.to_owned())),
    }
}

/// Enforce `PRIMARY KEY` / `UNIQUE` constraints for an INSERT batch: the new rows must not
/// collide with each other or with the table's existing visible rows.
fn enforce_unique_on_insert(
    table: &TableSchema,
    new_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if !table_has_unique_constraint(table, engine)? {
        return Ok(());
    }
    // Serialize concurrent writers of the same key *before* the snapshot scan: without this,
    // two READ COMMITTED inserts of the same key each scan a snapshot that does not see the other and
    // both commit a duplicate. The no-wait key lock makes the second writer abort here instead.
    lock_unique_keys(table, new_rows, engine, txn)?;
    // Fast path: probe each constraint's backing index for the new keys in O(log n), instead of
    // scanning + sorting the whole table. Falls back below when a constraint is not probe-eligible.
    if try_unique_by_index_probe(table, new_rows, engine, txn)? {
        return Ok(());
    }
    // The post-insert table = existing rows + the new rows; uniqueness must hold over it. The existing
    // rows are read with *latest-committed* visibility (not the txn's snapshot) so a row another
    // transaction committed after a REPEATABLE READ / SERIALIZABLE txn began is still seen — otherwise
    // a frozen snapshot lets a duplicate key commit. The key lock above already serialized any
    // concurrent writer of the same key.
    let mut all = scan_rows_committed(table, engine, txn)?;
    all.extend_from_slice(new_rows);
    enforce_unique_over_rows(table, &all, engine)
}

/// O(log n) uniqueness check: for each `PRIMARY KEY`/`UNIQUE` constraint, probe its backing index for
/// each new key with **latest-committed** visibility instead of scanning + sorting the whole table.
///
/// Returns `Ok(false)` — so [`enforce_unique_on_insert`] falls back to the full scan — unless EVERY
/// constraint is probe-eligible: a backing index exists and every participating key value is
/// index-equality-safe (`Float`/`NUMERIC` encode inconsistently, so a compare-equal duplicate could
/// be byte-unequal and slip past a byte-exact probe). Semantics then match [`enforce_unique_over_rows`]
/// exactly: a `NULL` key column means the row does not participate, and a collision — against a
/// committed row OR another new row in the batch — raises the same [`Error::ConstraintViolation`].
/// The caller already holds the per-key lock (`lock_unique_keys`) that serializes concurrent same-key
/// writers; the probe replaces the scan, not the lock, and its latest-committed view sees a key
/// committed after a `REPEATABLE READ`/`SERIALIZABLE` txn began (never a frozen snapshot).
fn try_unique_by_index_probe(
    table: &TableSchema,
    new_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    struct Probe {
        kind: nusadb_core::ConstraintKind,
        name: String,
        columns: Vec<String>,
        ordinals: Vec<usize>,
        index: nusadb_core::IndexId,
    }
    let mut probes: Vec<Probe> = Vec::new();
    for constraint in engine.list_constraints(table.id)? {
        if !matches!(
            constraint.kind,
            nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
        ) {
            continue;
        }
        let ordinals = constraint_ordinals(table, &constraint.columns)?;
        let Some(index) = engine.lookup_index(&constraint.name)? else {
            return Ok(false); // no backing index for some constraint → take the scan path
        };
        probes.push(Probe {
            kind: constraint.kind,
            name: constraint.name,
            columns: constraint.columns,
            ordinals,
            index,
        });
    }
    // Eligibility: every participating key value must be index-equality-safe. A single Float/NUMERIC
    // key column disqualifies the whole table (fall back to the scan, which uses `eval::compare`).
    for probe in &probes {
        for row in new_rows {
            if let Some(key) = unique_key(row, &probe.ordinals)
                && !key.iter().all(super::ops::is_hash_safe_value)
            {
                return Ok(false);
            }
        }
    }
    // Probe. New rows are not yet in any index (enforcement runs before `insert_into_indexes`), so an
    // index hit is always a distinct committed row.
    for probe in &probes {
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for row in new_rows {
            let Some(key) = unique_key(row, &probe.ordinals) else {
                continue; // a NULL key column: the row does not participate (NULL is distinct)
            };
            let encoded = index_key::encode_index_key(&key)?;
            if !seen.insert(encoded.clone()) {
                return Err(unique_violation_error(
                    probe.kind,
                    &probe.name,
                    &probe.columns,
                ));
            }
            let mut scan = engine.index_scan_committed(
                txn,
                probe.index,
                std::ops::Bound::Included(encoded.clone()),
                std::ops::Bound::Included(encoded),
            )?;
            if scan.try_next()?.is_some() {
                return Err(unique_violation_error(
                    probe.kind,
                    &probe.name,
                    &probe.columns,
                ));
            }
        }
    }
    Ok(true)
}

/// The `PRIMARY KEY`/`UNIQUE` duplicate-key error — shared by the scan path
/// ([`enforce_unique_over_rows`]) and the index-probe path so both raise a byte-identical message.
fn unique_violation_error(
    kind: nusadb_core::ConstraintKind,
    name: &str,
    columns: &[String],
) -> Error {
    nusadb_core::Error::ConstraintViolation(format!(
        "duplicate key violates {} constraint \"{}\" on ({})",
        match kind {
            nusadb_core::ConstraintKind::PrimaryKey => "primary key",
            _ => "unique",
        },
        name,
        columns.join(", "),
    ))
    .into()
}

/// Take an exclusive key-level lock for every `PRIMARY KEY`/`UNIQUE` key `rows` will write,
/// so concurrent transactions writing the same key serialize on it. Combined with the snapshot scan
/// this closes the duplicate-key race at every isolation level: overlapping writers conflict on the
/// lock (the second aborts no-wait), and a writer that starts after another commits sees that row in
/// its own snapshot. A `NULL` key column does not participate (SQL treats `NULL` as distinct).
fn lock_unique_keys(
    table: &TableSchema,
    rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let constraints: Vec<(Vec<String>, Vec<usize>)> = engine
        .list_constraints(table.id)?
        .into_iter()
        .filter(|c| {
            matches!(
                c.kind,
                nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
            )
        })
        .map(|c| constraint_ordinals(table, &c.columns).map(|ord| (c.columns, ord)))
        .collect::<Result<_, _>>()?;
    for row in rows {
        for (columns, ordinals) in &constraints {
            if let Some(key) = unique_key(row, ordinals) {
                let key_hash = unique_key_hash(table, columns, &key);
                engine.lock_key(
                    txn,
                    table.id,
                    key_hash,
                    nusadb_core::engine::RowLockMode::Exclusive,
                )?;
            }
        }
    }
    Ok(())
}

/// A stable hash of a unique key (constraint identity + key values) for [`StorageEngine::lock_key`].
///
/// Numeric values are hashed by a common `f64` form because the uniqueness check's equality
/// (`eval::compare`) treats `Int(1)`, `Float(1.0)`, and `Numeric(1)` as the *same* key — hashing
/// their debug strings would let two cross-representation writers of the same key skip the lock.
/// Any residual collision only *over*-serializes (the real uniqueness scan still runs); it never
/// mis-enforces, so a coarse-but-consistent hash is safe.
fn unique_key_hash(table: &TableSchema, columns: &[String], key: &[ast::Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    table.id.0.hash(&mut hasher);
    columns.hash(&mut hasher);
    for value in key {
        match value {
            ast::Value::Int(_) | ast::Value::Float(_) | ast::Value::Numeric(_) => {
                "num".hash(&mut hasher);
                crate::executor::agg::value_as_f64(value)
                    .to_bits()
                    .hash(&mut hasher);
            },
            other => format!("{other:?}").hash(&mut hasher),
        }
    }
    hasher.finish()
}

/// `ON CONFLICT DO NOTHING`: return the subset of `new_rows` that can be inserted without
/// violating any `PRIMARY KEY`/`UNIQUE` constraint — a row whose key (for some constraint) already
/// exists, in the table or in an earlier accepted row of this batch, is dropped. Rows are considered
/// in order, so the first of a set of colliding new rows is the one kept.
///
/// Key equality uses [`unique_key_cmp`] (the executor's total order) exactly as the enforcing path,
/// so a row this keeps would not have been rejected by [`enforce_unique_over_rows`]; a `NULL` key
/// column means the row does not participate in that constraint (NULL is distinct from everything).
fn keep_non_conflicting(
    table: &TableSchema,
    new_rows: Vec<Row>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    if !table_has_unique_constraint(table, engine)? {
        return Ok(new_rows);
    }
    // The key-column ordinals of each PK/UNIQUE constraint.
    let constraints: Vec<Vec<usize>> = engine
        .list_constraints(table.id)?
        .into_iter()
        .filter(|c| {
            matches!(
                c.kind,
                nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
            )
        })
        .map(|c| constraint_ordinals(table, &c.columns))
        .collect::<Result<_, _>>()?;
    // Per constraint, a sorted set of the keys already present (existing rows + accepted rows), so a
    // collision is a binary search under the total order.
    let existing = scan_rows(table, engine, txn)?;
    let mut seen: Vec<Vec<Vec<ast::Value>>> = constraints
        .iter()
        .map(|ordinals| {
            let mut keys: Vec<Vec<ast::Value>> = existing
                .iter()
                .filter_map(|row| unique_key(row, ordinals))
                .collect();
            keys.sort_by(|a, b| unique_key_cmp(a, b));
            keys
        })
        .collect();
    let mut kept = Vec::with_capacity(new_rows.len());
    for row in new_rows {
        let keys: Vec<Option<Vec<ast::Value>>> = constraints
            .iter()
            .map(|ordinals| unique_key(&row, ordinals))
            .collect();
        let conflicts = keys.iter().enumerate().any(|(i, key)| {
            key.as_ref().is_some_and(|k| {
                seen.get(i).is_some_and(|set| {
                    set.binary_search_by(|probe| unique_key_cmp(probe, k))
                        .is_ok()
                })
            })
        });
        if conflicts {
            continue;
        }
        // Accept: record its keys so a later row in the same batch collides with it too.
        for (i, key) in keys.into_iter().enumerate() {
            if let (Some(k), Some(set)) = (key, seen.get_mut(i)) {
                let pos = set
                    .binary_search_by(|probe| unique_key_cmp(probe, &k))
                    .unwrap_or_else(|insert_at| insert_at);
                set.insert(pos, k);
            }
        }
        kept.push(row);
    }
    Ok(kept)
}

/// Enforce `PRIMARY KEY` / `UNIQUE` constraints over a complete set of rows — no two rows may share
/// a non-`NULL` key for any unique constraint.
///
/// Validated by sorting each constraint's key tuples with the executor's total order (`eval::compare`)
/// and checking adjacent pairs — `O(rows·log rows·keys)`, down from the earlier `O(rows²)` linear
/// probe. Sorting (not hashing) is used deliberately so the equality matches `eval::compare`
/// exactly — e.g. `-0.0`/`+0.0` and `NUMERIC` scale that a byte hash would split. A row with a `NULL`
/// in any key column does not participate (SQL `UNIQUE` treats `NULL` as distinct from everything).
/// An index-backed fast path is a follow-up once the index subsystem reads constraints.
fn enforce_unique_over_rows(
    table: &TableSchema,
    rows: &[Row],
    engine: &dyn StorageEngine,
) -> Result<(), Error> {
    let constraints = engine.list_constraints(table.id)?;
    for constraint in &constraints {
        if !matches!(
            constraint.kind,
            nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
        ) {
            continue;
        }
        let ordinals = constraint_ordinals(table, &constraint.columns)?;
        // Gather the participating (non-NULL) key tuples, sort by the total order, then any two
        // equal keys are adjacent.
        let mut keys: Vec<Vec<ast::Value>> = rows
            .iter()
            .filter_map(|row| unique_key(row, &ordinals))
            .collect();
        keys.sort_by(|a, b| unique_key_cmp(a, b));
        if keys
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if unique_key_cmp(a, b) == std::cmp::Ordering::Equal))
        {
            return Err(unique_violation_error(
                constraint.kind,
                &constraint.name,
                &constraint.columns,
            ));
        }
    }
    Ok(())
}

/// Reject a rewritten row whose new key collides with a row another transaction committed after this
/// txn's snapshot (A-QA1b, the UPDATE / MERGE matched-update analogue of `enforce_unique_on_insert`).
/// The snapshot-based [`enforce_unique_over_rows`] cannot see such a row under REPEATABLE READ /
/// SERIALIZABLE. Checks the `new_rows` keys against the *latest-committed* state minus the rows this
/// statement itself rewrites, plus the new rows among themselves. The key lock the caller already
/// holds serializes any concurrent same-key writer.
///
/// "Rows this statement rewrites" are excluded two ways:
/// - by `rewritten` **tid** — the row version this statement saw; and
/// - by the **key its old image held** (`old_rows`, per PK/UNIQUE constraint) — a concurrent
///   transaction that committed an update to the same logical row gave it a *new* tid, so without
///   this the unchanged key "collides" with the newer committed version of the very row being
///   rewritten and a plain concurrent `UPDATE` misreports as a duplicate-key violation.
///
/// The key exclusion cannot hide a genuine duplicate: it fires only when the committed state of a
/// row this statement rewrites has changed, which makes the statement's later write to that stale
/// tid fail with the retryable serialization conflict (`40001`) before anything commits — the
/// exclusion exists purely so *that* honest error surfaces instead of a bogus dup-key.
fn enforce_new_keys_vs_committed(
    table: &TableSchema,
    rewritten: &HashSet<Tid>,
    old_rows: &[Row],
    new_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if !table_has_unique_constraint(table, engine)? {
        return Ok(());
    }
    // Per PK/UNIQUE constraint, the (sorted) keys the rewritten rows held before this statement.
    let constraints: Vec<Vec<usize>> = engine
        .list_constraints(table.id)?
        .into_iter()
        .filter(|c| {
            matches!(
                c.kind,
                nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
            )
        })
        .map(|c| constraint_ordinals(table, &c.columns))
        .collect::<Result<_, _>>()?;
    let old_keys: Vec<Vec<Vec<ast::Value>>> = constraints
        .iter()
        .map(|ordinals| {
            let mut keys: Vec<Vec<ast::Value>> = old_rows
                .iter()
                .filter_map(|row| unique_key(row, ordinals))
                .collect();
            keys.sort_by(|a, b| unique_key_cmp(a, b));
            keys
        })
        .collect();
    let holds_a_rewritten_key = |row: &Row| {
        constraints.iter().zip(&old_keys).any(|(ordinals, keys)| {
            unique_key(row, ordinals).is_some_and(|key| {
                keys.binary_search_by(|probe| unique_key_cmp(probe, &key))
                    .is_ok()
            })
        })
    };
    let mut universe: Vec<Row> = scan_table_committed(table, engine, txn)?
        .into_iter()
        .filter(|(tid, row)| !rewritten.contains(tid) && !holds_a_rewritten_key(row))
        .map(|(_, row)| row)
        .collect();
    universe.extend_from_slice(new_rows);
    enforce_unique_over_rows(table, &universe, engine)
}

/// O(log n) uniqueness re-check for an UPDATE/MERGE: probe each PK/UNIQUE constraint's backing index
/// for the rewritten rows' NEW keys with latest-committed visibility, instead of scanning + sorting
/// the whole table. The unified index probe subsumes BOTH the snapshot whole-table check
/// ([`enforce_unique_over_rows`] over the post-update image) and the latest-committed check
/// ([`enforce_new_keys_vs_committed`]): a duplicate is a duplicate against the state the statement
/// will actually commit into, and the latest-committed view (plus this txn's own writes) is exactly
/// that state.
///
/// Returns `Ok(false)` — so the caller runs the unchanged scan-based checks — unless every constraint
/// is probe-eligible (a backing index and index-equality-safe new keys). A hit is a duplicate UNLESS
/// it is a row this statement is itself rewriting, excluded exactly as [`enforce_new_keys_vs_committed`]
/// does: by rewritten `Tid` (the version this statement saw) OR by an old key one of the rewritten
/// rows held (a concurrent update gave that logical row a new `Tid`, so the unchanged key must not
/// read as a duplicate — the stale write then fails with the retryable `40001` instead). New rows are
/// checked against each other via a `HashSet` of encoded keys. The caller already holds the per-key
/// lock (`lock_unique_keys`).
fn try_update_unique_by_index_probe(
    table: &TableSchema,
    rewritten: &HashSet<Tid>,
    old_rows: &[Row],
    new_rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    struct Probe {
        kind: nusadb_core::ConstraintKind,
        name: String,
        columns: Vec<String>,
        ordinals: Vec<usize>,
        index: nusadb_core::IndexId,
    }
    let mut probes: Vec<Probe> = Vec::new();
    for constraint in engine.list_constraints(table.id)? {
        if !matches!(
            constraint.kind,
            nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
        ) {
            continue;
        }
        let ordinals = constraint_ordinals(table, &constraint.columns)?;
        let Some(index) = engine.lookup_index(&constraint.name)? else {
            return Ok(false);
        };
        probes.push(Probe {
            kind: constraint.kind,
            name: constraint.name,
            columns: constraint.columns,
            ordinals,
            index,
        });
    }
    for probe in &probes {
        for row in new_rows {
            if let Some(key) = unique_key(row, &probe.ordinals)
                && !key.iter().all(super::ops::is_hash_safe_value)
            {
                return Ok(false);
            }
        }
    }
    // The (sorted) keys the rewritten rows held before this statement, per constraint — a committed
    // row still holding one of these is the concurrently-reversioned version of a row we are
    // rewriting, so it must be excluded (mirrors `enforce_new_keys_vs_committed`).
    let old_keys: Vec<Vec<Vec<ast::Value>>> = probes
        .iter()
        .map(|probe| {
            let mut keys: Vec<Vec<ast::Value>> = old_rows
                .iter()
                .filter_map(|row| unique_key(row, &probe.ordinals))
                .collect();
            keys.sort_by(|a, b| unique_key_cmp(a, b));
            keys
        })
        .collect();
    let holds_a_rewritten_key = |row: &Row| {
        probes.iter().zip(&old_keys).any(|(probe, keys)| {
            unique_key(row, &probe.ordinals)
                .is_some_and(|key| keys.binary_search_by(|p| unique_key_cmp(p, &key)).is_ok())
        })
    };
    let schema = column_types(table);
    for probe in &probes {
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for row in new_rows {
            let Some(key) = unique_key(row, &probe.ordinals) else {
                continue;
            };
            let encoded = index_key::encode_index_key(&key)?;
            if !seen.insert(encoded.clone()) {
                return Err(unique_violation_error(
                    probe.kind,
                    &probe.name,
                    &probe.columns,
                ));
            }
            let mut scan = engine.index_scan_committed(
                txn,
                probe.index,
                std::ops::Bound::Included(encoded.clone()),
                std::ops::Bound::Included(encoded),
            )?;
            while let Some((tid, tuple)) = scan.try_next()? {
                if rewritten.contains(&tid) {
                    continue; // the version this statement is itself rewriting
                }
                let hit = row::decode(&tuple, &schema)?;
                if holds_a_rewritten_key(&hit) {
                    continue; // a concurrently-reversioned copy of a row being rewritten
                }
                return Err(unique_violation_error(
                    probe.kind,
                    &probe.name,
                    &probe.columns,
                ));
            }
        }
    }
    Ok(true)
}

/// Whether `table` has any `PRIMARY KEY` / `UNIQUE` constraint to enforce (avoids a needless scan).
/// Uses the treaty existence check so the engine answers from its catalog without cloning the whole
/// constraint list on every INSERT/UPDATE.
fn table_has_unique_constraint(
    table: &TableSchema,
    engine: &dyn StorageEngine,
) -> Result<bool, Error> {
    Ok(engine.has_unique_constraint(table.id)?)
}

/// Whether an `UPDATE` whose assigned columns are `set_cols` can change any `PRIMARY KEY` / `UNIQUE`
/// key of `table` — the gate for the whole-table uniqueness check on the update path. When no
/// assigned column participates in a unique/PK constraint, the SET cannot alter any key, so a table
/// that was unique before the update stays unique without materializing and re-sorting every row.
fn update_touches_unique_columns(
    table: &TableSchema,
    set_cols: &HashSet<usize>,
    engine: &dyn StorageEngine,
) -> Result<bool, Error> {
    for constraint in engine.list_constraints(table.id)? {
        if !matches!(
            constraint.kind,
            nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
        ) {
            continue;
        }
        if constraint_ordinals(table, &constraint.columns)?
            .iter()
            .any(|ordinal| set_cols.contains(ordinal))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The key tuple of `row` for a constraint's columns, or `None` if any key column is `NULL` (such
/// a row does not participate in `UNIQUE` — `NULL` is distinct from everything, including `NULL`).
pub(super) fn unique_key(row: &Row, ordinals: &[usize]) -> Option<Vec<ast::Value>> {
    let mut key = Vec::with_capacity(ordinals.len());
    for &ordinal in ordinals {
        match row.get(ordinal) {
            Some(ast::Value::Null) | None => return None,
            Some(value) => key.push(value.clone()),
        }
    }
    Some(key)
}

/// Equality of two non-NULL key tuples, using the executor's total-order comparison.
fn unique_key_eq(a: &[ast::Value], b: &[ast::Value]) -> bool {
    a.len() == b.len() && unique_key_cmp(a, b) == std::cmp::Ordering::Equal
}

/// Lexicographic total order over two equal-length key tuples, via the executor's `compare` (so the
/// uniqueness sort and the `eq` check share one definition of "same key").
pub(super) fn unique_key_cmp(a: &[ast::Value], b: &[ast::Value]) -> std::cmp::Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| eval::compare(x, y))
        .find(|ordering| *ordering != std::cmp::Ordering::Equal)
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Resolve a constraint's column names to their ordinals in `table` (declaration order). The names
/// come from the catalog, so they always resolve; a miss is an internal inconsistency.
pub(super) fn constraint_ordinals(
    table: &TableSchema,
    columns: &[String],
) -> Result<Vec<usize>, Error> {
    columns
        .iter()
        .map(|name| {
            table
                .columns
                .iter()
                .position(|c| &c.name == name)
                .ok_or_else(|| {
                    nusadb_core::Error::ConstraintViolation(format!(
                        "constraint references unknown column \"{name}\""
                    ))
                    .into()
                })
        })
        .collect()
}

// === FOREIGN KEY enforcement ======================================

/// The current schema of a table by id (FK enforcement scans the *other* table, which the caller
/// only knows by [`TableId`]). Composes the catalog's current-version + version→schema lookups.
pub(super) fn schema_by_id(
    engine: &dyn StorageEngine,
    table: nusadb_core::TableId,
) -> Result<Option<TableSchema>, Error> {
    match engine.current_schema_version(table)? {
        Some(version) => Ok(engine.schema_for_version(table, version)?),
        None => Ok(None),
    }
}

/// The ordinals of a table's `PRIMARY KEY` columns (a foreign key references the parent's PK), or
/// `None` if the table has no primary key.
fn primary_key_ordinals(
    table: &TableSchema,
    engine: &dyn StorageEngine,
) -> Result<Option<Vec<usize>>, Error> {
    let pk = engine
        .list_constraints(table.id)?
        .into_iter()
        .find(|c| matches!(c.kind, nusadb_core::ConstraintKind::PrimaryKey));
    match pk {
        Some(constraint) => Ok(Some(constraint_ordinals(table, &constraint.columns)?)),
        None => Ok(None),
    }
}

/// The ordinals (into `parent`) of the columns a foreign key references — the explicit referenced
/// `UNIQUE`/`PRIMARY KEY` columns the engine recorded, or the parent's `PRIMARY KEY` when none were
/// recorded. `None` only when the parent has no primary key and the FK named no columns —
/// nothing can reference such a parent.
fn fk_parent_ordinals(
    parent: &TableSchema,
    fk: &nusadb_core::ForeignKeyDef,
    engine: &dyn StorageEngine,
) -> Result<Option<Vec<usize>>, Error> {
    if fk.parent_columns.is_empty() {
        primary_key_ordinals(parent, engine)
    } else {
        Ok(Some(constraint_ordinals(parent, &fk.parent_columns)?))
    }
}

/// Enforce that every foreign key on `table` references an existing parent row, for the rows being
/// written by an INSERT/UPDATE. A row whose FK columns contain a `NULL` does not reference
/// anything (MATCH SIMPLE) and is skipped. Scan-based (the SQL layer owns row decoding); an
/// index-backed `fk_check` fast path is a follow-up once the index key encoder lands.
pub(super) fn enforce_fk_on_child_write(
    table: &TableSchema,
    rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let fks: Vec<_> = engine
        .list_foreign_keys(table.id)?
        .into_iter()
        .filter(|fk| fk.child_table == table.id)
        .collect();
    for fk in fks {
        let child_ordinals = constraint_ordinals(table, &fk.child_columns)?;
        let parent = schema_by_id(engine, fk.parent_table)?.ok_or_else(|| {
            nusadb_core::Error::ConstraintViolation(format!(
                "foreign key \"{}\": parent table is missing",
                fk.name
            ))
        })?;
        let Some(parent_key) = fk_parent_ordinals(&parent, &fk, engine)? else {
            return Err(nusadb_core::Error::ConstraintViolation(format!(
                "foreign key \"{}\": parent table \"{}\" has no referenced key",
                fk.name, parent.name
            ))
            .into());
        };
        let parent_rows = scan_rows(&parent, engine, txn)?;
        for row in rows {
            let Some(key) = unique_key(row, &child_ordinals) else {
                continue; // NULL foreign key — references nothing (MATCH SIMPLE).
            };
            let present = parent_rows
                .iter()
                .filter_map(|prow| unique_key(prow, &parent_key))
                .any(|pkey| unique_key_eq(&pkey, &key));
            if !present {
                return Err(nusadb_core::Error::ConstraintViolation(format!(
                    "insert or update on \"{}\" violates foreign key \"{}\": no matching row in \"{}\"",
                    table.name, fk.name, parent.name
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// A [`Catalog`](crate::analyzer::Catalog) that resolves nothing — used to re-analyze a stored
/// single-table CHECK predicate, which references only the target table's own columns (supplied via
/// the column scope), never another table, view, or index. Validated to be subquery-free at
/// CREATE/ALTER time, so it never needs to resolve a name.
pub(super) struct EmptyCatalog;

impl crate::analyzer::Catalog for EmptyCatalog {
    fn lookup_table(&self, _name: &str) -> Result<Option<TableSchema>, Error> {
        Ok(None)
    }
}

/// Enforce every `CHECK` constraint declared on `table` against the `rows` about to be written.
/// Each constraint persists its predicate as canonical SQL; we re-parse and type-check it
/// against the table's column scope, then evaluate it per row. A row passes when the predicate is
/// `TRUE` *or* `NULL` (SQL's three-valued CHECK semantics) — only an explicit `FALSE` is a
/// violation. Shared by INSERT, COPY, and UPDATE so every write path enforces it identically.
pub(super) fn enforce_check_on_write(
    table: &TableSchema,
    rows: &[Row],
    engine: &dyn StorageEngine,
) -> Result<(), Error> {
    let checks: Vec<(String, crate::planner::TypedExpr)> = engine
        .list_constraints(table.id)?
        .into_iter()
        .filter(|c| c.kind == nusadb_core::engine::ConstraintKind::Check)
        .map(|c| {
            let bytes = c.expr.ok_or_else(|| {
                nusadb_core::Error::ConstraintViolation(format!(
                    "check constraint \"{}\" has no stored predicate",
                    c.name
                ))
            })?;
            let sql = String::from_utf8(bytes).map_err(|_| {
                nusadb_core::Error::ConstraintViolation(format!(
                    "check constraint \"{}\" has a corrupt (non-UTF-8) predicate",
                    c.name
                ))
            })?;
            let typed = crate::analyzer::analyze_check_predicate(&sql, table, &EmptyCatalog)?;
            Ok::<_, Error>((c.name, typed))
        })
        .collect::<Result<_, _>>()?;
    if checks.is_empty() {
        return Ok(());
    }
    for row in rows {
        for (name, predicate) in &checks {
            if matches!(eval::eval(predicate, row)?, ast::Value::Bool(false)) {
                return Err(nusadb_core::Error::ConstraintViolation(format!(
                    "new row for \"{}\" violates check constraint \"{}\"",
                    table.name, name
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// Cascade-delete `rows` from `child`, firing the child table's row-level DELETE triggers around
/// each write (deep-gate #4). A referential action must not bypass the child's
/// audit/validation triggers the way a raw `engine.delete` would. Statement-level child triggers are
/// intentionally not fired: a cascade is a side effect of the parent statement, and firing them
/// inside the per-parent-row enforcement loop would fire them more than once.
fn cascade_delete_children(
    child: &TableSchema,
    rows: Vec<(Tid, Row)>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let triggers =
        super::trigger::load_table_triggers(&child.name, ast::TriggerEvent::Delete, engine, txn)?;
    let mut deleted: Vec<Row> = Vec::with_capacity(rows.len());
    for (tid, row) in rows {
        if triggers.has_before_row() {
            triggers.fire_row_before(child, Some(&row), None, engine, txn)?;
        }
        engine.delete(txn, child.id, tid)?;
        if triggers.has_after_row() {
            triggers.fire_row_after(child, Some(&row), None, engine, txn)?;
        }
        deleted.push(row);
    }
    // Incremental view maintenance (deep-gate #16): remove the cascade-deleted rows from any view over
    // the child. The child's secondary-index entries are left in place — the per-tid
    // visibility filter hides them, and VACUUM reclaims them — exactly as `run_delete` does.
    super::ivm::maintain_on_change(&child.name, &[], &deleted, engine, txn)?;
    Ok(())
}

/// Cascade-update `changes` (each `(tid, old, new)`) on `child`, firing the child table's row-level
/// UPDATE triggers around each write (deep-gate #4) — for `ON ... CASCADE` (key rewrite) and
/// `ON ... SET NULL`. See [`cascade_delete_children`] for why statement-level triggers are not fired.
fn cascade_update_children(
    child: &TableSchema,
    changes: Vec<(Tid, Row, Row)>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let triggers =
        super::trigger::load_table_triggers(&child.name, ast::TriggerEvent::Update, engine, txn)?;
    let child_types = column_types(child);
    let index_targets = secondary_index_targets(child, engine)?;
    let mut olds: Vec<Row> = Vec::with_capacity(changes.len());
    let mut news: Vec<Row> = Vec::with_capacity(changes.len());
    for (tid, old, new) in changes {
        if triggers.has_before_row() {
            triggers.fire_row_before(child, Some(&old), Some(&new), engine, txn)?;
        }
        let bytes = row::encode(&new, &child_types)?;
        let new_tid = engine.update(txn, child.id, tid, &bytes)?;
        // Maintain the child's secondary indexes for the rewritten row version, as `run_update`
        // does — the old version's entries stay until VACUUM, except a partial index the row is
        // leaving, whose stale entry is removed.
        if !index_targets.is_empty() {
            remove_departed_index_entries(&index_targets, &old, tid, &new, engine, txn)?;
            insert_into_indexes(&index_targets, &new, new_tid, engine, txn)?;
        }
        if triggers.has_after_row() {
            triggers.fire_row_after(child, Some(&old), Some(&new), engine, txn)?;
        }
        olds.push(old);
        news.push(new);
    }
    // Incremental view maintenance (deep-gate #16): apply the cascade rewrite's delta to any view over
    // the child, so a materialized view does not go stale after a cascade.
    super::ivm::maintain_on_change(&child.name, &news, &olds, engine, txn)?;
    Ok(())
}

/// Build the `(tid, old, new)` triples for a `SET NULL` cascade: each `new` image is the referencing
/// child row with its `fk_ordinals` columns set to `NULL`. Retaining `old` lets the child's UPDATE
/// triggers bind `OLD`.
fn null_fk_changes(
    referencing: Vec<(Tid, Row)>,
    fk_ordinals: &[usize],
) -> Result<Vec<(Tid, Row, Row)>, Error> {
    referencing
        .into_iter()
        .map(|(tid, old)| {
            let mut new = old.clone();
            for &ordinal in fk_ordinals {
                set_at(&mut new, ordinal, ast::Value::Null)?;
            }
            Ok((tid, old, new))
        })
        .collect()
}

/// Enforce referential actions when parent `rows` of `table` are about to be deleted: for
/// each foreign key pointing at `table`, find the child rows referencing a deleted key and apply
/// `NO ACTION`/`RESTRICT` (reject), `CASCADE` (delete the children, one level), or `SET NULL` (null
/// the child FK columns). `SET DEFAULT` needs a column DEFAULT clause and is rejected for honesty.
/// Cascade writes fire the child's row triggers (see [`cascade_delete_children`]).
fn enforce_fk_on_parent_delete(
    table: &TableSchema,
    rows: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let fks: Vec<_> = engine
        .list_foreign_keys(table.id)?
        .into_iter()
        .filter(|fk| fk.parent_table == table.id)
        .collect();
    if fks.is_empty() {
        return Ok(());
    }
    for fk in fks {
        // The parent key this FK references (a non-PK UNIQUE or the PRIMARY KEY) — resolved per FK,
        // since different FKs pointing at this table may reference different parent keys.
        let Some(parent_ordinals) = fk_parent_ordinals(table, &fk, engine)? else {
            continue; // This FK references the PK, but the parent has none — nothing references it.
        };
        let deleted_keys: Vec<Vec<ast::Value>> = rows
            .iter()
            .filter_map(|r| unique_key(r, &parent_ordinals))
            .collect();
        if deleted_keys.is_empty() {
            continue;
        }
        let child = schema_by_id(engine, fk.child_table)?.ok_or_else(|| {
            nusadb_core::Error::ConstraintViolation(format!(
                "foreign key \"{}\": child table is missing",
                fk.name
            ))
        })?;
        let child_ordinals = constraint_ordinals(&child, &fk.child_columns)?;
        let child_rows = scan_table(&child, engine, txn)?;
        let referencing: Vec<(Tid, Row)> = child_rows
            .into_iter()
            .filter(|(_, crow)| {
                unique_key(crow, &child_ordinals)
                    .is_some_and(|ckey| deleted_keys.iter().any(|dk| unique_key_eq(dk, &ckey)))
            })
            .collect();
        if referencing.is_empty() {
            continue;
        }
        match fk.on_delete {
            nusadb_core::FkAction::NoAction | nusadb_core::FkAction::Restrict => {
                return Err(nusadb_core::Error::ConstraintViolation(format!(
                    "delete on \"{}\" violates foreign key \"{}\": {} dependent row(s) in \"{}\"",
                    table.name,
                    fk.name,
                    referencing.len(),
                    child.name
                ))
                .into());
            },
            nusadb_core::FkAction::Cascade => {
                cascade_delete_children(&child, referencing, engine, txn)?;
            },
            // ON DELETE SET NULL: null the child's FK columns (a row rewrite the SQL layer owns).
            nusadb_core::FkAction::SetNull => {
                let changes = null_fk_changes(referencing, &child_ordinals)?;
                cascade_update_children(&child, changes, engine, txn)?;
            },
            // SET DEFAULT needs a column DEFAULT clause, which the CREATE TABLE surface does not
            // yet carry — reject honestly rather than silently nulling.
            nusadb_core::FkAction::SetDefault => {
                return Err(Error::Unsupported(format!(
                    "foreign key \"{}\" ON DELETE SET DEFAULT is not supported (no column DEFAULT)",
                    fk.name
                )));
            },
        }
    }
    Ok(())
}

/// Whether any foreign key in the catalog points at `table` (i.e. `table` is an FK parent). Lets
/// the UPDATE path skip per-row PK-change tracking when nothing can reference this table.
fn table_is_fk_parent(table: &TableSchema, engine: &dyn StorageEngine) -> Result<bool, Error> {
    Ok(engine
        .list_foreign_keys(table.id)?
        .iter()
        .any(|fk| fk.parent_table == table.id))
}

/// Enforce referential actions when a parent `table`'s rows are updated: for any row whose
/// PRIMARY KEY changed, apply each pointing foreign key's `ON UPDATE` action to the children that
/// referenced the *old* key — `NO ACTION`/`RESTRICT` (reject), `CASCADE` (rewrite the child FK to
/// the new key), or `SET NULL` (null the child FK). `SET DEFAULT` is rejected (no column DEFAULT).
/// `changes` are `(old_row, new_row)` pairs for the rows being updated.
fn enforce_fk_on_parent_update(
    table: &TableSchema,
    changes: &[(Row, Row)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let fks: Vec<_> = engine
        .list_foreign_keys(table.id)?
        .into_iter()
        .filter(|fk| fk.parent_table == table.id)
        .collect();
    if fks.is_empty() {
        return Ok(());
    }
    for fk in &fks {
        // The parent key this FK references (a non-PK UNIQUE or the PRIMARY KEY) — resolved per FK,
        // since different FKs pointing at this table may reference different parent keys.
        let Some(parent_ordinals) = fk_parent_ordinals(table, fk, engine)? else {
            continue;
        };
        let child = schema_by_id(engine, fk.child_table)?.ok_or_else(|| {
            nusadb_core::Error::ConstraintViolation(format!(
                "foreign key \"{}\": child table is missing",
                fk.name
            ))
        })?;
        let child_ordinals = constraint_ordinals(&child, &fk.child_columns)?;
        for (old, new) in changes {
            let (Some(old_key), new_key) = (
                unique_key(old, &parent_ordinals),
                unique_key(new, &parent_ordinals),
            ) else {
                continue; // Old key was NULL — nothing references it.
            };
            // Only a *changed* key needs propagation.
            if new_key
                .as_ref()
                .is_some_and(|nk| unique_key_eq(nk, &old_key))
            {
                continue;
            }
            let referencing: Vec<(Tid, Row)> = scan_table(&child, engine, txn)?
                .into_iter()
                .filter(|(_, crow)| {
                    unique_key(crow, &child_ordinals).is_some_and(|ck| unique_key_eq(&ck, &old_key))
                })
                .collect();
            if referencing.is_empty() {
                continue;
            }
            match fk.on_update {
                nusadb_core::FkAction::NoAction | nusadb_core::FkAction::Restrict => {
                    return Err(nusadb_core::Error::ConstraintViolation(format!(
                        "update on \"{}\" violates foreign key \"{}\": {} dependent row(s) in \"{}\"",
                        table.name,
                        fk.name,
                        referencing.len(),
                        child.name
                    ))
                    .into());
                },
                nusadb_core::FkAction::Cascade => {
                    // Rewrite each child's FK columns to the parent's new key.
                    let Some(new_key) = new_key.as_ref() else {
                        return Err(nusadb_core::Error::ConstraintViolation(format!(
                            "foreign key \"{}\" ON UPDATE CASCADE: new key is NULL",
                            fk.name
                        ))
                        .into());
                    };
                    let mut changes = Vec::with_capacity(referencing.len());
                    for (tid, crow) in referencing {
                        // FK columns map positionally onto the parent key columns (same arity).
                        let mut updated = crow.clone();
                        for (&ordinal, value) in child_ordinals.iter().zip(new_key) {
                            set_at(&mut updated, ordinal, value.clone())?;
                        }
                        changes.push((tid, crow, updated));
                    }
                    cascade_update_children(&child, changes, engine, txn)?;
                },
                nusadb_core::FkAction::SetNull => {
                    let changes = null_fk_changes(referencing, &child_ordinals)?;
                    cascade_update_children(&child, changes, engine, txn)?;
                },
                nusadb_core::FkAction::SetDefault => {
                    return Err(Error::Unsupported(format!(
                        "foreign key \"{}\" ON UPDATE SET DEFAULT is not supported (no column DEFAULT)",
                        fk.name
                    )));
                },
            }
        }
    }
    Ok(())
}

// === Secondary index maintenance =================================

/// Every index of `table` the SQL layer maintains on writes, as `(index id, key-column ordinals)`
/// — explicit `CREATE INDEX` indexes **and**, since the backing-index unification, the `PRIMARY KEY`/`UNIQUE`
/// constraint-backing indexes, which are now scannable access paths and so must cover every live
/// row. Backing indexes are maintained purely as lookup structures: the engine skips its
/// byte-level unique check for them (the scan-based constraint checks + key locks above own the
/// SQL semantics — NULLs never conflict, statements may pass through transient duplicates). Empty
/// for the common no-index table, so callers skip all per-row index work.
pub(super) fn secondary_index_targets(
    table: &TableSchema,
    engine: &dyn StorageEngine,
) -> Result<Vec<IndexTarget>, Error> {
    let mut targets = Vec::new();
    for def in engine.list_indexes(table.id)? {
        let Some(id) = engine.lookup_index(&def.name)? else {
            continue;
        };
        // An index that cannot be resolved against the current schema (its key columns dropped,
        // or a functional-key/partial-predicate that no longer analyzes) cannot be maintained —
        // skip it, mirroring the analyzer's plan-time skip, which also keeps such an index from
        // ever being offered as a scan candidate.
        if let Some(target) = build_index_target(id, table, &def) {
            targets.push(target);
        }
    }
    Ok(targets)
}

/// How an index's key is built from a row: plain column ordinals (the common case) or evaluated
/// key expressions (a functional/expression index).
pub(super) enum IndexKeys {
    /// Read these column ordinals from the row.
    Columns(Vec<usize>),
    /// Evaluate these key expressions against the row (functional/expression index).
    Exprs(Vec<TypedExpr>),
}

/// A resolved index the SQL layer maintains on writes: its id, how to build its key, and an
/// optional partial-index predicate (only rows for which it is true are indexed).
pub(super) struct IndexTarget {
    id: nusadb_core::IndexId,
    keys: IndexKeys,
    predicate: Option<TypedExpr>,
}

/// Whether an `UPDATE` whose assigned columns are `set_cols` can change `target`'s entries. A plain
/// column index is touched only when the SET assigns one of its key columns (the row keeps its
/// address across MVCC versions, so an unchanged key still maps to the same live row and its entry
/// needs no rewrite). A functional (expression) index or a partial index is treated as always
/// touched: its key or its membership can depend on columns the target list does not obviously name,
/// and keeping such an index maintained is only ever redundant work, never incorrect.
fn index_target_touched_by_set(target: &IndexTarget, set_cols: &HashSet<usize>) -> bool {
    if target.predicate.is_some() {
        return true;
    }
    match &target.keys {
        IndexKeys::Columns(cols) => cols.iter().any(|ordinal| set_cols.contains(ordinal)),
        IndexKeys::Exprs(_) => true,
    }
}

/// Resolve one index `def` (already mapped to `id`) into a maintainable [`IndexTarget`], or `None`
/// when it cannot be resolved against the current schema (key columns dropped, or a stored key
/// expression / partial predicate that no longer analyzes) — such an index is skipped on writes,
/// exactly as it is excluded from scan candidates.
pub(super) fn build_index_target(
    id: nusadb_core::IndexId,
    table: &TableSchema,
    def: &nusadb_core::IndexDef,
) -> Option<IndexTarget> {
    let predicate = match &def.predicate {
        Some(sql) => {
            Some(crate::analyzer::analyze_check_predicate(sql, table, &EmptyCatalog).ok()?)
        },
        None => None,
    };
    let keys = if def.key_exprs.is_empty() {
        IndexKeys::Columns(constraint_ordinals(table, &def.columns).ok()?)
    } else {
        let mut exprs = Vec::with_capacity(def.key_exprs.len());
        for sql in &def.key_exprs {
            exprs.push(crate::analyzer::analyze_index_key_expr(sql, table, &EmptyCatalog).ok()?);
        }
        IndexKeys::Exprs(exprs)
    };
    Some(IndexTarget {
        id,
        keys,
        predicate,
    })
}

/// The order-preserving key for `row` under an index's key definition (plain columns or evaluated
/// key expressions).
pub(super) fn index_key_for(row: &Row, keys: &IndexKeys) -> Result<Vec<u8>, Error> {
    let values: Vec<ast::Value> = match keys {
        IndexKeys::Columns(ordinals) => ordinals
            .iter()
            .map(|&o| row.get(o).cloned().unwrap_or(ast::Value::Null))
            .collect(),
        IndexKeys::Exprs(exprs) => exprs
            .iter()
            .map(|e| eval::eval(e, row))
            .collect::<Result<_, _>>()?,
    };
    index_key::encode_index_key(&values)
}

/// Whether `row` is indexed by `target` — a partial index skips rows whose predicate is not true
/// (NULL or false); a full index indexes every row.
fn row_is_indexed(target: &IndexTarget, row: &Row) -> Result<bool, Error> {
    match &target.predicate {
        Some(pred) => Ok(matches!(eval::eval(pred, row)?, ast::Value::Bool(true))),
        None => Ok(true),
    }
}

/// Add `row`@`tid` to every secondary index that indexes it (partial indexes skip non-matching
/// rows; functional/expression indexes evaluate their key).
pub(super) fn insert_into_indexes(
    targets: &[IndexTarget],
    row: &Row,
    tid: Tid,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for target in targets {
        if !row_is_indexed(target, row)? {
            continue;
        }
        engine.index_insert(txn, target.id, &index_key_for(row, &target.keys)?, tid)?;
    }
    Ok(())
}

/// Build one index over `table` (a `CREATE INDEX` backfill) by streaming its rows and applying their
/// entries in key-sorted chunks via `index_insert_batch`, so the build drives sequential index writes
/// without materializing the whole table or all of its entries at once. A chunk is flushed once its
/// buffered entries reach the maintenance-memory budget ([`maintenance_work_mem`](super::ops::maintenance_work_mem)),
/// so the footprint is bounded in bytes regardless of row width. A partial index's predicate skips
/// non-matching rows and a functional/expression key is evaluated, exactly as [`insert_into_indexes`]
/// does per row; a `unique` index still rejects a duplicate — each chunk is applied before the next
/// is read, so a later chunk sees an earlier one's keys.
pub(super) fn backfill_index_streaming(
    target: &IndexTarget,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let schema = column_types(table);
    let budget = super::ops::maintenance_work_mem();
    let mut scan = engine.scan(txn, table.id)?;
    let mut entries: Vec<(Vec<u8>, Tid)> = Vec::new();
    let mut buffered = 0usize;
    while let Some((tid, tuple)) = scan.try_next()? {
        // Honor a statement timeout / cancel at row granularity on a long build, and skip a row a
        // concurrent transaction holds locked — matching the scan the per-row backfill used.
        crate::cancel::check()?;
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        let row = row::decode(&tuple, &schema)?;
        if row_is_indexed(target, &row)? {
            let key = index_key_for(&row, &target.keys)?;
            buffered += key.len() + std::mem::size_of::<(Vec<u8>, Tid)>();
            entries.push((key, tid));
            // Flush once the buffered entries reach the budget, so the build stays bounded in bytes.
            if buffered >= budget {
                engine.index_insert_batch(txn, target.id, std::mem::take(&mut entries))?;
                buffered = 0;
            }
        }
    }
    if !entries.is_empty() {
        engine.index_insert_batch(txn, target.id, entries)?;
    }
    Ok(())
}

/// Build one index over `table` (a `CREATE INDEX` backfill), choosing the strategy by whether
/// spill-to-disk is configured. With spill, an external merge sort applies the entries in one global
/// key order for sequential index writes across the whole build; without it, the in-memory chunked
/// path ([`backfill_index_streaming`]) keeps the build bounded with per-chunk order. Both are bounded
/// by the maintenance-memory budget and produce the same complete, uniqueness-enforced index.
pub(super) fn backfill_index(
    target: &IndexTarget,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    super::spill::spill_config().map_or_else(
        || backfill_index_streaming(target, table, engine, txn),
        |config| backfill_index_external_sort(target, table, engine, txn, &config),
    )
}

/// Monotonic id for index-build spill-run file names (process-local; not persisted).
static INDEX_SORT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The fixed tid suffix of a spilled entry: `page` (`u64`) then `slot` (`u16`), little-endian.
const SPILLED_TID_BYTES: usize = 8 + 2;

/// A merged run's current head: the key drives the min-heap; the tid rides along.
struct EntryHead {
    key: Vec<u8>,
    tid: Tid,
    run: usize,
}

impl PartialEq for EntryHead {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.run == other.run
    }
}
impl Eq for EntryHead {}
impl Ord for EntryHead {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.run.cmp(&other.run))
    }
}
impl PartialOrd for EntryHead {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Encode one `(key, tid)` entry for a spill run: the key bytes followed by the fixed tid suffix.
fn encode_entry(key: &[u8], tid: Tid) -> Vec<u8> {
    let mut out = Vec::with_capacity(key.len() + SPILLED_TID_BYTES);
    out.extend_from_slice(key);
    out.extend_from_slice(&tid.page.0.to_le_bytes());
    out.extend_from_slice(&tid.slot.0.to_le_bytes());
    out
}

/// Decode a spilled entry record back into `(key, tid)` — the key is everything before the fixed
/// tid suffix.
fn decode_entry(record: &[u8]) -> Result<(Vec<u8>, Tid), Error> {
    let corrupt = || {
        Error::Core(nusadb_core::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "corrupt spilled index entry",
        )))
    };
    let split = record
        .len()
        .checked_sub(SPILLED_TID_BYTES)
        .ok_or_else(corrupt)?;
    let key = record.get(..split).ok_or_else(corrupt)?.to_vec();
    let page: [u8; 8] = record
        .get(split..split + 8)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(corrupt)?;
    let slot: [u8; 2] = record
        .get(split + 8..)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(corrupt)?;
    Ok((
        key,
        Tid {
            page: nusadb_core::PageId(u64::from_le_bytes(page)),
            slot: nusadb_core::SlotIdx(u16::from_le_bytes(slot)),
        },
    ))
}

/// Sort `buf` by key and write it as one on-disk run, then clear `buf`.
fn spill_entry_run(
    buf: &mut Vec<(Vec<u8>, Tid)>,
    config: &super::spill::SpillConfig,
    seq: u64,
    run: usize,
    runs: &mut Vec<super::spill::SpillReader>,
) -> Result<(), Error> {
    buf.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    let path = config.dir.join(format!(
        "nusadb-spill-index-{}-{seq}-{run}.tmp",
        std::process::id()
    ));
    let mut writer = super::spill::SpillWriter::create(path)?;
    for (key, tid) in buf.drain(..) {
        writer.write_bytes(&encode_entry(&key, tid))?;
    }
    runs.push(writer.into_reader()?);
    Ok(())
}

/// Build one index by external merge sort: stream the table into sorted runs on disk (a run whenever
/// the buffer fills the maintenance-memory budget), then k-way merge the runs and apply the entries
/// in one global key order via `index_insert_batch`. That drives sequential index writes across the
/// whole build while bounding memory to one run buffer plus the merge heads. Semantics match the
/// chunked path (partial-index predicate, functional key, uniqueness). The runs are transient
/// scratch, deleted on drop; a crash mid-build rolls back with the transaction, so the index ends up
/// complete or absent.
fn backfill_index_external_sort(
    target: &IndexTarget,
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
    config: &super::spill::SpillConfig,
) -> Result<(), Error> {
    let schema = column_types(table);
    let budget = super::ops::maintenance_work_mem();
    let seq = INDEX_SORT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut scan = engine.scan(txn, table.id)?;
    let mut buf: Vec<(Vec<u8>, Tid)> = Vec::new();
    let mut buffered = 0usize;
    let mut runs: Vec<super::spill::SpillReader> = Vec::new();

    // Phase 1: stream the table, generating a sorted run each time the buffer fills the budget.
    while let Some((tid, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        if super::lock_skip::skipped(table.id, tid) {
            continue;
        }
        let row = row::decode(&tuple, &schema)?;
        if row_is_indexed(target, &row)? {
            let key = index_key_for(&row, &target.keys)?;
            buffered += key.len() + std::mem::size_of::<(Vec<u8>, Tid)>();
            buf.push((key, tid));
            if buffered >= budget {
                spill_entry_run(&mut buf, config, seq, runs.len(), &mut runs)?;
                buffered = 0;
            }
        }
    }

    // The whole set fit the budget — apply it as one sorted batch (whole-table order, no spill).
    if runs.is_empty() {
        engine.index_insert_batch(txn, target.id, buf)?;
        return Ok(());
    }
    if !buf.is_empty() {
        spill_entry_run(&mut buf, config, seq, runs.len(), &mut runs)?;
    }

    // Phase 2: k-way merge the runs, applying entries in global key order in budget-sized batches.
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<EntryHead>> =
        std::collections::BinaryHeap::new();
    for (run, reader) in runs.iter_mut().enumerate() {
        if let Some(record) = reader.read_bytes()? {
            let (key, tid) = decode_entry(&record)?;
            heap.push(std::cmp::Reverse(EntryHead { key, tid, run }));
        }
    }
    let mut out: Vec<(Vec<u8>, Tid)> = Vec::new();
    let mut out_bytes = 0usize;
    while let Some(std::cmp::Reverse(head)) = heap.pop() {
        crate::cancel::check()?;
        let run = head.run;
        out_bytes += head.key.len() + std::mem::size_of::<(Vec<u8>, Tid)>();
        out.push((head.key, head.tid));
        if out_bytes >= budget {
            engine.index_insert_batch(txn, target.id, std::mem::take(&mut out))?;
            out_bytes = 0;
        }
        if let Some(reader) = runs.get_mut(run)
            && let Some(record) = reader.read_bytes()?
        {
            let (key, tid) = decode_entry(&record)?;
            heap.push(std::cmp::Reverse(EntryHead { key, tid, run }));
        }
    }
    if !out.is_empty() {
        engine.index_insert_batch(txn, target.id, out)?;
    }
    Ok(())
}

/// On UPDATE, remove a row's entry from every **partial** index it is leaving — it was indexed
/// (`old` satisfied the predicate) but its new version is not.
///
/// The append-only maintenance dead-stamps an old index entry only when a NEW entry is inserted for
/// the same row under a different key (inside `index_insert`); a row that departs a partial
/// predicate inserts no new entry, so without this its stale entry would linger forever and cause a
/// spurious uniqueness conflict against a genuinely new row. A full (non-partial) index needs
/// nothing here: a key change still inserts a new entry, which dead-stamps the old one. The partial
/// index is never a scan candidate, so the hard removal cannot make a concurrent reader miss a row.
pub(super) fn remove_departed_index_entries(
    targets: &[IndexTarget],
    old_row: &Row,
    old_tid: Tid,
    new_row: &Row,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for target in targets {
        // Only partial indexes can shed a row without a compensating insert.
        if target.predicate.is_none() {
            continue;
        }
        if row_is_indexed(target, old_row)? && !row_is_indexed(target, new_row)? {
            let key = index_key_for(old_row, &target.keys)?;
            engine.index_delete(txn, target.id, &key, old_tid)?;
        }
    }
    Ok(())
}

// === UPDATE / DELETE ======================================================

/// Run an inlined derived-relation plan — the source of `UPDATE ... FROM (VALUES/SELECT ...)` or
/// `DELETE ... USING (...)` — within the current transaction and collect its rows. Executed once per
/// statement (not per target row), against the same `txn` snapshot as the rest of the statement.
fn materialize_subplan(
    plan: &crate::planner::SelectPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let op = crate::planner::plan_select(plan.clone());
    match run_select(&op, None, engine, txn)? {
        ExecutionResult::Rows { rows, .. } => Ok(rows),
        // `run_select` always yields `Rows`; anything else is an internal invariant break.
        _ => Err(Error::Unsupported(
            "internal: a derived UPDATE/DELETE source did not produce a row set".to_owned(),
        )),
    }
}

/// Find an UPDATE/DELETE's target rows through a **unique point lookup** — an `O(log n)` index probe
/// in place of the `O(n)` full [`scan_table`].
///
/// Returns `None` (the caller then full-scans) unless the `WHERE` resolves to an equality covering
/// the whole key of a UNIQUE / PRIMARY KEY index (`WHERE pk = const`); a range, a non-unique bound,
/// or no matching predicate keeps the sequential scan.
///
/// Correctness: the returned set is a *superset* of the qualifying rows (an equality point lookup
/// matches at most one row, and the caller always re-applies the full `WHERE` per row), so no
/// qualifying row is missed and no extra row is affected. Restricting to a unique *point* lookup
/// also bounds the result to ≤1 row, so there is no scan-order difference to reason about versus a
/// sequential scan. Only the safe, complete, plain-column indexes ([`catalog_list_indexes`] already
/// drops functional / partial / still-building indexes) are offered as a path.
fn try_point_get_rows(
    table: &TableSchema,
    filter: Option<&TypedExpr>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<Vec<(Tid, Row)>>, Error> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    // Resolve the table's scannable indexes to planner metadata with key-column ordinals (mirrors
    // the analyzer's `resolve_table_indexes`, but from the executor's live catalog view).
    let mut metas: Vec<crate::planner::IndexMeta> = Vec::new();
    for info in super::catalog_list_indexes(engine, txn, &table.name)? {
        let mut columns = Vec::with_capacity(info.columns.len());
        let mut ok = true;
        for col in &info.columns {
            let Some(ord) = table.columns.iter().position(|c| &c.name == col) else {
                ok = false;
                break;
            };
            columns.push(ord);
        }
        if ok && !columns.is_empty() {
            metas.push(crate::planner::IndexMeta {
                name: info.name,
                columns,
                unique: info.unique,
            });
        }
    }
    // Reuse the SELECT index-selection (single source of truth for matching a predicate to an
    // index), but take only a *unique point lookup* — a `SeqScan`-superset guarantee plus the ≤1-row
    // bound. No stats are passed: an equality on a unique whole key is always cheaper than a full
    // scan, so the cost gate is unnecessary here.
    match crate::planner::try_point_get_index(table, &metas, filter) {
        Some(crate::planner::PhysicalOperator::IndexScan {
            index,
            lo,
            hi,
            unique_point: true,
            ..
        }) => Ok(Some(index_scan_table(
            table, &index, &lo, &hi, engine, txn,
        )?)),
        _ => Ok(None),
    }
}

/// Rebuild the whole-table **post-update image** for a plain `UPDATE` whose target rows were found
/// through the index (so the image was not materialized during the match pass). Scans every visible
/// row and substitutes each updated row's new value by `Tid`, yielding exactly what `result_rows`
/// would have held on the sequential-scan path — the set the whole-table uniqueness fallback checks.
/// Only reached when the `O(log n)` uniqueness probe cannot cover a constraint, so paying a full scan
/// here is unavoidable and no worse than the pre-index behavior.
fn rebuild_post_update_rows(
    table: &TableSchema,
    to_update: &[(Tid, Option<Row>, Row)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<Row>, Error> {
    let updated: HashMap<Tid, &Row> = to_update.iter().map(|(tid, _, new)| (*tid, new)).collect();
    Ok(scan_table(table, engine, txn)?
        .into_iter()
        .map(|(tid, row)| updated.get(&tid).map_or(row, |new| (*new).clone()))
        .collect())
}

#[allow(
    clippy::too_many_lines,
    reason = "one cohesive UPDATE pass: match (incl. FROM join), fire triggers, enforce constraints, write"
)]
pub(super) fn run_update(
    plan: &UpdatePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let schema = column_types(&plan.table);
    // Find the target rows through the backing index when the `WHERE` is a unique point lookup
    // (`UPDATE t ... WHERE pk = const`) — `O(log n)` instead of a full-table `scan_table`. Only for a
    // plain `UPDATE` (an `UPDATE ... FROM` join needs the whole target relation); the per-row `WHERE`
    // below is re-applied unchanged, so the index path only narrows the candidate set to a superset.
    // `via_index` records that the whole-table image (`result_rows`) was NOT materialized, so the
    // uniqueness fallback rebuilds it on demand (see `enforce_unique_over_rows` call below).
    let (rows, via_index) = match plan
        .from
        .is_none()
        .then(|| try_point_get_rows(&plan.table, plan.filter.as_ref(), engine, txn))
        .transpose()?
        .flatten()
    {
        Some(rows) => (rows, true),
        None => (scan_table(&plan.table, engine, txn)?, false),
    };
    // Compute the post-update state first: matched rows take their new values, unmatched rows stay.
    // PRIMARY KEY / UNIQUE must hold over the *whole* resulting table (a new key may collide with an
    // untouched row or with another updated row), so validate before writing anything.
    //
    // Heap-only-tuple fast path: gate the whole-table uniqueness materialize+sort and the per-row
    // index maintenance on the columns the SET actually assigns. A row keeps its address (row-id)
    // across MVCC versions (an update rewrites in place with a version chain), so an index whose key
    // columns the SET does not touch still maps its unchanged key to the same live row — its entry
    // needs no rewrite. Likewise a SET that touches no unique/PK column cannot create a duplicate.
    // This turns a large non-key range UPDATE from O(n log n) whole-table work into a pure in-place
    // update. `set_cols` is the assigned column ordinals, PLUS every STORED generated column: a
    // generated column's stored value is recomputed from its dependencies on any assignment, so it
    // can change without being assigned directly — folding all generated columns in (conservatively)
    // keeps a generated key/index column from ever being mis-classified as untouched.
    let mut set_cols: HashSet<usize> = plan.assignments.iter().map(|a| a.column).collect();
    if !set_cols.is_empty() {
        set_cols.extend(super::coldefault::generated_column_ordinals(
            &plan.table,
            engine,
            txn,
        )?);
    }
    let needs_unique = table_has_unique_constraint(&plan.table, engine)?
        && update_touches_unique_columns(&plan.table, &set_cols, engine)?;
    let is_fk_parent = table_is_fk_parent(&plan.table, engine)?;
    let mut index_targets = secondary_index_targets(&plan.table, engine)?;
    index_targets.retain(|target| index_target_touched_by_set(target, &set_cols));
    // Triggers: load UPDATE triggers once; a per-row trigger needs the old row image so
    // `OLD.col` can be bound, so force capture when one exists.
    let triggers = super::trigger::load_table_triggers(
        &plan.table.name,
        ast::TriggerEvent::Update,
        engine,
        txn,
    )?;
    // Track each matched row's pre-update value when a changed PRIMARY KEY must propagate to FK
    // children, its old secondary-index entries must be removed, a row trigger binds
    // `OLD`, or an IVM view over this table needs the delete side of the delta.
    let has_ivm = super::ivm::has_views_for_base(engine, txn, &plan.table.name)?;
    let track_old =
        is_fk_parent || !index_targets.is_empty() || triggers.needs_old_image() || has_ivm;
    let mut to_update: Vec<(Tid, Option<Row>, Row)> = Vec::new();
    let mut result_rows: Vec<Row> = if needs_unique {
        Vec::with_capacity(rows.len())
    } else {
        Vec::new()
    };
    // The matched rows' pre-update images, for the committed-state uniqueness re-check's
    // rewritten-key exclusion. Only kept when uniqueness is enforced.
    let mut old_for_unique: Vec<Row> = Vec::new();
    // `UPDATE ... FROM <src>`: the FROM rows the SET values and WHERE join against. Scanned
    // once; each target row uses the first FROM row the WHERE (over `target ++ from`) matches.
    // The FROM source rows: a derived source (`FROM (VALUES ...)` / `(SELECT ...)`) runs its inlined
    // plan; a named source is scanned; a plain UPDATE has none.
    let from_rows: Vec<Row> = if let Some(from_plan) = &plan.from_plan {
        materialize_subplan(from_plan, engine, txn)?
    } else if let Some(from_table) = &plan.from {
        scan_table(from_table, engine, txn)?
            .into_iter()
            .map(|(_, r)| r)
            .collect()
    } else {
        Vec::new()
    };
    // Resolve any *uncorrelated* subquery in the SET values / WHERE to a literal once, before the
    // per-row loop (for UPDATE) — e.g. `SET x = (SELECT max(v) FROM other)`. There is no per-row
    // pass here, so defer-correlated keeps a correlated subquery (one referencing the target row) in
    // place to be rejected at eval, rather than resolving it against an unbound row to a wrong NULL.
    let mut assignments = plan.assignments.clone();
    let mut filter = plan.filter.clone();
    {
        let _defer = super::ops::defer_correlated(true);
        for assignment in &mut assignments {
            super::ops::resolve_subqueries(&mut assignment.value, engine, txn)?;
        }
        if let Some(predicate) = filter.as_mut() {
            super::ops::resolve_subqueries(predicate, engine, txn)?;
        }
    }
    // Generated columns: SET-ting one is an error; any other SET recomputes them against the
    // new row (`recompute_generated` per updated row below). `fills` is a no-op when none are generated.
    let fills = super::coldefault::column_fills(&plan.table, engine, txn)?;
    {
        let set_cols: HashSet<usize> = assignments.iter().map(|a| a.column).collect();
        reject_explicit_generated(&plan.table, &fills, &set_cols)?;
    }
    for (tid, row) in rows {
        if plan.from.is_some() {
            // Join: find the first FROM row matching the WHERE over the concatenated row; apply the
            // SET against that combined row. No match → the target row is left unchanged.
            let mut applied = false;
            for frow in &from_rows {
                let mut combined = row.clone();
                combined.extend(frow.iter().cloned());
                if predicate_matches(filter.as_ref(), &combined)? {
                    let old = track_old.then(|| row.clone());
                    let new_row = recompute_generated(
                        apply_assignments_ctx(&assignments, &plan.table, row.clone(), &combined)?,
                        &fills,
                        &plan.table,
                    )?;
                    if needs_unique {
                        result_rows.push(new_row.clone());
                        old_for_unique.push(row.clone());
                    }
                    to_update.push((tid, old, new_row));
                    applied = true;
                    break;
                }
            }
            if !applied && needs_unique {
                result_rows.push(row);
            }
        } else if predicate_matches(filter.as_ref(), &row)? {
            let old = track_old.then(|| row.clone());
            if needs_unique {
                old_for_unique.push(row.clone());
            }
            let new_row = recompute_generated(
                apply_assignments(&assignments, &plan.table, row)?,
                &fills,
                &plan.table,
            )?;
            if needs_unique {
                result_rows.push(new_row.clone());
            }
            to_update.push((tid, old, new_row));
        } else if needs_unique {
            result_rows.push(row);
        }
    }
    // BEFORE triggers: fire statement-level once, then row-level for each matched row, before
    // any constraint check or write.
    triggers.fire_stmt_before(&plan.table, engine, txn)?;
    if triggers.has_before_row() {
        for (_, old, new_row) in &to_update {
            triggers.fire_row_before(&plan.table, old.as_deref(), Some(new_row), engine, txn)?;
        }
    }
    // Row-level security WITH CHECK: each post-update row must satisfy the applicable policies.
    // Checked before any write, so a violation aborts the whole UPDATE (the transaction rolls back).
    if let Some(check) = plan.rls_check.as_ref() {
        for (_, _, new_row) in &to_update {
            if !predicate_matches(Some(check), new_row)? {
                return Err(Error::RlsCheckViolation {
                    table: plan.table.name.clone(),
                });
            }
        }
    }
    // FOREIGN KEY: each updated row must still reference an existing parent…
    let updated_rows: Vec<Row> = to_update.iter().map(|(_, _, new)| new.clone()).collect();
    if needs_unique {
        // Serialize concurrent writers of the same key before the snapshot-based uniqueness check
        // (deep-gate #7): two UPDATEs that set different rows to the *same* new key each scan
        // a snapshot blind to the other and would both commit a duplicate. The no-wait key lock over
        // the rows actually written makes the second writer abort here, exactly as the INSERT path
        // (`enforce_unique_on_insert`) does. (The deeper frozen-snapshot case under RR/SER is the
        // engine-level A-QA1b.)
        lock_unique_keys(&plan.table, &updated_rows, engine, txn)?;
        let rewritten: HashSet<Tid> = to_update.iter().map(|(tid, _, _)| *tid).collect();
        // Fast path: probe each constraint's backing index for the new keys in O(log n) under the
        // latest-committed view, instead of the two whole-table scan+sort checks. Falls back to those
        // (snapshot whole-table check + the latest-committed cross-transaction check) when a
        // constraint is not probe-eligible (no backing index, or a Float/NUMERIC key that encodes
        // inconsistently).
        if !try_update_unique_by_index_probe(
            &plan.table,
            &rewritten,
            &old_for_unique,
            &updated_rows,
            engine,
            txn,
        )? {
            // The probe could not cover a constraint (no backing index, or a key type that encodes
            // inconsistently), so fall back to the whole-table uniqueness check. When the target rows
            // were found through the index (`via_index`), the whole-table post-update image was never
            // materialized — rebuild it now (a full scan, unavoidable for this check) so the fallback
            // sees exactly the rows the scan path would have.
            let whole_table;
            let result_rows = if via_index {
                whole_table = rebuild_post_update_rows(&plan.table, &to_update, engine, txn)?;
                &whole_table
            } else {
                &result_rows
            };
            enforce_unique_over_rows(&plan.table, result_rows, engine)?;
            enforce_new_keys_vs_committed(
                &plan.table,
                &rewritten,
                &old_for_unique,
                &updated_rows,
                engine,
                txn,
            )?;
        }
    }
    enforce_fk_on_child_write(&plan.table, &updated_rows, engine, txn)?;
    enforce_check_on_write(&plan.table, &updated_rows, engine)?;
    // …and a changed parent key must be propagated to its children (ON UPDATE) before any write.
    if is_fk_parent {
        let fk_changes: Vec<(Row, Row)> = to_update
            .iter()
            .filter_map(|(_, old, new)| old.as_ref().map(|o| (o.clone(), new.clone())))
            .collect();
        enforce_fk_on_parent_update(&plan.table, &fk_changes, engine, txn)?;
    }
    // RETURNING projects each updated row's *post-update* values.
    let mut returned: Vec<Row> = Vec::new();
    for (tid, old, new_row) in &to_update {
        let bytes = row::encode(new_row, &schema)?;
        let new_tid = engine.update(txn, plan.table.id, *tid, &bytes)?;
        // Add the new row version's secondary-index entries. The old tid's entries are
        // *not* removed here: they stay until VACUUM reclaims the old version, so a frozen
        // REPEATABLE READ / SERIALIZABLE snapshot can still find the pre-update version through the
        // index. The index scan filters every entry by per-tid MVCC visibility, so the superseded
        // old version (now `xmax`-stamped) is invisible to readers that should not see it.
        if !index_targets.is_empty() {
            // First shed any PARTIAL index the row is leaving (its predicate no longer holds), so
            // the departed entry does not linger and cause a false uniqueness conflict.
            if let Some(old) = old {
                remove_departed_index_entries(&index_targets, old, *tid, new_row, engine, txn)?;
            }
            insert_into_indexes(&index_targets, new_row, new_tid, engine, txn)?;
        }
        if !plan.returning.is_empty() {
            returned.push(project_row(&plan.returning, new_row)?);
        }
    }
    // AFTER triggers: row-level for each updated row, then statement-level once.
    if triggers.has_after_row() {
        for (_, old, new_row) in &to_update {
            triggers.fire_row_after(&plan.table, old.as_deref(), Some(new_row), engine, txn)?;
        }
    }
    triggers.fire_stmt_after(&plan.table, engine, txn)?;
    // Incremental view maintenance: an UPDATE is a delete of the old image + insert of the
    // new one on each IVM view (old images captured above via `has_ivm`).
    if has_ivm {
        let new_rows: Vec<Row> = to_update.iter().map(|(_, _, new)| new.clone()).collect();
        let old_rows: Vec<Row> = to_update
            .iter()
            .filter_map(|(_, old, _)| old.clone())
            .collect();
        super::ivm::maintain_on_change(&plan.table.name, &new_rows, &old_rows, engine, txn)?;
    }
    let updated = to_update.len();
    if plan.returning.is_empty() {
        Ok(ExecutionResult::Updated(updated))
    } else {
        Ok(ExecutionResult::Rows {
            columns: plan.returning.iter().map(|p| p.name.clone()).collect(),
            rows: returned,
        })
    }
}

pub(super) fn run_delete(
    plan: &DeletePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // Find the target rows through the backing index when the `WHERE` is a unique point lookup
    // (`DELETE FROM t WHERE pk = const`) — `O(log n)` instead of a full-table `scan_table`. Only for
    // a plain `DELETE` (a `USING` join needs the whole target relation); the per-row `WHERE` below is
    // re-applied unchanged, so the index path only narrows the candidate set to a correct superset.
    let rows = match plan
        .using
        .is_none()
        .then(|| try_point_get_rows(&plan.table, plan.filter.as_ref(), engine, txn))
        .transpose()?
        .flatten()
    {
        Some(rows) => rows,
        None => scan_table(&plan.table, engine, txn)?,
    };
    // `DELETE ... USING <src>`: the USING rows the WHERE joins against. A target row is
    // deleted if it matches the WHERE over `target ++ using` for *any* USING row.
    let using_rows: Vec<Row> = if let Some(using_plan) = &plan.using_plan {
        materialize_subplan(using_plan, engine, txn)?
    } else if let Some(using_table) = &plan.using {
        scan_table(using_table, engine, txn)?
            .into_iter()
            .map(|(_, r)| r)
            .collect()
    } else {
        Vec::new()
    };
    // Resolve any uncorrelated subquery in the WHERE to a literal once, before the per-row loop —
    // e.g. `DELETE FROM t WHERE id IN (SELECT id FROM stale)`. Defer-correlated keeps a correlated
    // subquery in place to be rejected at eval (no per-row pass here), not mis-resolved to NULL.
    let mut filter = plan.filter.clone();
    if let Some(predicate) = filter.as_mut() {
        let _defer = super::ops::defer_correlated(true);
        super::ops::resolve_subqueries(predicate, engine, txn)?;
    }
    // Collect the matching rows first, so foreign keys pointing at this table can be enforced
    // (RESTRICT) or propagated (CASCADE) before any parent row is removed.
    let mut to_delete: Vec<(Tid, Row)> = Vec::new();
    for (tid, row) in rows {
        let matched = if plan.using.is_some() {
            using_rows.iter().try_fold(false, |hit, urow| {
                if hit {
                    return Ok(true);
                }
                let mut combined = row.clone();
                combined.extend(urow.iter().cloned());
                predicate_matches(filter.as_ref(), &combined)
            })?
        } else {
            predicate_matches(filter.as_ref(), &row)?
        };
        if matched {
            to_delete.push((tid, row));
        }
    }
    // BEFORE triggers: statement-level once, then row-level for each matched row, before the
    // FK delete actions and the writes.
    let triggers = super::trigger::load_table_triggers(
        &plan.table.name,
        ast::TriggerEvent::Delete,
        engine,
        txn,
    )?;
    triggers.fire_stmt_before(&plan.table, engine, txn)?;
    if triggers.has_before_row() {
        for (_, row) in &to_delete {
            triggers.fire_row_before(&plan.table, Some(row), None, engine, txn)?;
        }
    }
    let deleted_rows: Vec<Row> = to_delete.iter().map(|(_, row)| row.clone()).collect();
    enforce_fk_on_parent_delete(&plan.table, &deleted_rows, engine, txn)?;

    // RETURNING projects each deleted row's *pre-delete* values.
    let mut returned: Vec<Row> = Vec::new();
    for (tid, row) in &to_delete {
        if !plan.returning.is_empty() {
            returned.push(project_row(&plan.returning, row)?);
        }
        // The row's secondary-index entries are left in place: the delete only stamps
        // `xmax` on the version, which the index scan's per-tid visibility filter already hides from
        // readers that should not see it. VACUUM removes the entries when it reclaims the version, so
        // a frozen snapshot can still reach the pre-delete row through the index until then.
        engine.delete(txn, plan.table.id, *tid)?;
    }
    // AFTER triggers: row-level for each deleted row, then statement-level once.
    if triggers.has_after_row() {
        for (_, row) in &to_delete {
            triggers.fire_row_after(&plan.table, Some(row), None, engine, txn)?;
        }
    }
    triggers.fire_stmt_after(&plan.table, engine, txn)?;
    // Incremental view maintenance: remove the projected rows from any IVM view over this
    // table.
    super::ivm::maintain_on_change(&plan.table.name, &[], &deleted_rows, engine, txn)?;
    // TRUNCATE ... RESTART IDENTITY: reset the backing sequence of each SERIAL/IDENTITY
    // column so the next insert restarts at the sequence's start value. Those sequences are created
    // with start = 1, increment = 1, so setting the current value to 0 makes the next `nextval`
    // return 1 (matching the reference engine, which restarts identity at the sequence start).
    if plan.restart_identity {
        let key = super::coldefault::catalog_key(&plan.table.schema, &plan.table.name);
        for (column, sql) in super::coldefault::load_defaults(&key, engine, txn)? {
            if let Some(seq) = super::coldefault::serial_sequence(&sql) {
                let id = engine.lookup_sequence(seq)?.ok_or_else(|| {
                    Error::Unsupported(format!(
                        "serial column \"{column}\" has no backing sequence"
                    ))
                })?;
                engine.sequence_set(id, 0)?;
            }
        }
    }
    let deleted = to_delete.len();
    if plan.returning.is_empty() {
        Ok(ExecutionResult::Deleted(deleted))
    } else {
        Ok(ExecutionResult::Rows {
            columns: plan.returning.iter().map(|p| p.name.clone()).collect(),
            rows: returned,
        })
    }
}

// === MERGE ========================================================

/// Run `MERGE INTO target USING source ON ... WHEN [NOT] MATCHED ...`. Each source row is
/// classified against the target by the `ON` condition: a matched row drives the first satisfied
/// `WHEN MATCHED` clause (UPDATE/DELETE), an unmatched row the first `WHEN NOT MATCHED` (INSERT). A
/// target row may be affected at most once (a second hit is a cardinality error). The three op sets
/// are then applied DELETE → UPDATE → INSERT, each with the same constraint/trigger/index/IVM
/// enforcement a plain `DELETE`/`UPDATE`/`INSERT` performs (inserts run through [`insert_rows`]).
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive MERGE pass: classify every source row, then apply the three op sets"
)]
pub(super) fn run_merge(
    plan: &crate::planner::MergePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    use crate::planner::{MergeMatchedAction, MergeWhen};
    let target_rows = scan_table(&plan.table, engine, txn)?;
    // A derived `USING (VALUES ...)` / `USING (SELECT ...)` source runs its inlined plan; a plain
    // named source is scanned (mirrors `UPDATE ... FROM` / `DELETE ... USING`).
    let source_rows: Vec<Row> = if let Some(source_plan) = &plan.source_plan {
        materialize_subplan(source_plan, engine, txn)?
    } else {
        scan_table(&plan.source, engine, txn)?
            .into_iter()
            .map(|(_, r)| r)
            .collect()
    };
    let null_target = vec![ast::Value::Null; plan.table.columns.len()];

    let mut updates: Vec<(Tid, Row, Row)> = Vec::new();
    let mut deletes: Vec<(Tid, Row)> = Vec::new();
    // NOT-MATCHED inserts grouped by their target-column list so each group is one `insert_rows` batch.
    let mut insert_groups: HashMap<Vec<usize>, Vec<Row>> = HashMap::new();
    let mut affected: HashSet<Tid> = HashSet::new();
    let mut count = 0usize;
    // Generated columns: a matched UPDATE recomputes them; `fills` is a no-op when none exist.
    let fills = super::coldefault::column_fills(&plan.table, engine, txn)?;

    for srow in &source_rows {
        // The first target row the ON condition matches (over `target ++ source`).
        let mut hit: Option<(Tid, Row)> = None;
        for (tid, trow) in &target_rows {
            let mut combined = trow.clone();
            combined.extend(srow.iter().cloned());
            if matches!(eval::eval(&plan.on, &combined)?, ast::Value::Bool(true)) {
                hit = Some((*tid, trow.clone()));
                break;
            }
        }
        if let Some((tid, trow)) = hit {
            let mut combined = trow.clone();
            combined.extend(srow.iter().cloned());
            for when in &plan.whens {
                let MergeWhen::Matched { pred, action } = when else {
                    continue;
                };
                if !predicate_matches(pred.as_ref(), &combined)? {
                    continue;
                }
                if !affected.insert(tid) {
                    return Err(nusadb_core::Error::ConstraintViolation(format!(
                        "MERGE on \"{}\" cannot affect one target row more than once",
                        plan.table.name
                    ))
                    .into());
                }
                match action {
                    MergeMatchedAction::Update { assignments } => {
                        reject_explicit_generated(
                            &plan.table,
                            &fills,
                            &assignments.iter().map(|a| a.column).collect(),
                        )?;
                        let new = recompute_generated(
                            apply_assignments_ctx(
                                assignments,
                                &plan.table,
                                trow.clone(),
                                &combined,
                            )?,
                            &fills,
                            &plan.table,
                        )?;
                        updates.push((tid, trow.clone(), new));
                    },
                    MergeMatchedAction::Delete => deletes.push((tid, trow.clone())),
                }
                count += 1;
                break;
            }
        } else {
            let mut combined = null_target.clone();
            combined.extend(srow.iter().cloned());
            for when in &plan.whens {
                let MergeWhen::NotMatched {
                    pred,
                    columns,
                    values,
                } = when
                else {
                    continue;
                };
                if !predicate_matches(pred.as_ref(), &combined)? {
                    continue;
                }
                let value_row: Row = values
                    .iter()
                    .map(|v| eval::eval(v, &combined))
                    .collect::<Result<_, _>>()?;
                insert_groups
                    .entry(columns.clone())
                    .or_default()
                    .push(value_row);
                count += 1;
                break;
            }
        }
    }

    // Apply DELETE, then UPDATE, then INSERT — so the inserts' constraint checks see the updated
    // state (a not-matched insert never collides with a row a matched clause just changed/removed).
    commit_merge_deletes(&plan.table, &deletes, engine, txn)?;
    commit_merge_updates(&plan.table, &updates, &deletes, &target_rows, engine, txn)?;
    for (columns, value_rows) in insert_groups {
        // A `MERGE ... WHEN NOT MATCHED THEN INSERT` row carries concrete evaluated values, never a
        // `DEFAULT` cell — wrap each as `Some` for `insert_rows`.
        let value_rows: Vec<Vec<Option<ast::Value>>> = value_rows
            .into_iter()
            .map(|r| r.into_iter().map(Some).collect())
            .collect();
        insert_rows(&plan.table, &columns, value_rows, None, false, engine, txn)?;
    }
    Ok(ExecutionResult::Merged(count))
}

/// Commit a `MERGE`'s matched-DELETE set with the same enforcement a plain `DELETE` performs:
/// triggers, parent-side foreign-key actions, and IVM.
fn commit_merge_deletes(
    table: &TableSchema,
    deletes: &[(Tid, Row)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if deletes.is_empty() {
        return Ok(());
    }
    let triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Delete, engine, txn)?;
    let deleted_rows: Vec<Row> = deletes.iter().map(|(_, r)| r.clone()).collect();
    triggers.fire_stmt_before(table, engine, txn)?;
    if triggers.has_before_row() {
        for (_, row) in deletes {
            triggers.fire_row_before(table, Some(row), None, engine, txn)?;
        }
    }
    enforce_fk_on_parent_delete(table, &deleted_rows, engine, txn)?;
    for (tid, _) in deletes {
        engine.delete(txn, table.id, *tid)?;
    }
    if triggers.has_after_row() {
        for (_, row) in deletes {
            triggers.fire_row_after(table, Some(row), None, engine, txn)?;
        }
    }
    triggers.fire_stmt_after(table, engine, txn)?;
    super::ivm::maintain_on_change(&table.name, &[], &deleted_rows, engine, txn)?;
    Ok(())
}

/// Commit a `MERGE`'s matched-UPDATE set with the same enforcement a plain `UPDATE` performs:
/// triggers, UNIQUE (over the resulting target state, with the deletes removed), CHECK, foreign keys
/// (child + parent-update), secondary indexes, and IVM. The `deletes`/`target_rows` are used
/// only to project the post-merge state for the uniqueness check.
fn commit_merge_updates(
    table: &TableSchema,
    updates: &[(Tid, Row, Row)],
    deletes: &[(Tid, Row)],
    target_rows: &[(Tid, Row)],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if updates.is_empty() {
        return Ok(());
    }
    let schema = column_types(table);
    let index_targets = secondary_index_targets(table, engine)?;
    let triggers =
        super::trigger::load_table_triggers(&table.name, ast::TriggerEvent::Update, engine, txn)?;
    triggers.fire_stmt_before(table, engine, txn)?;
    if triggers.has_before_row() {
        for (_, old, new) in updates {
            triggers.fire_row_before(table, Some(old), Some(new), engine, txn)?;
        }
    }
    let new_rows: Vec<Row> = updates.iter().map(|(_, _, n)| n.clone()).collect();
    // UNIQUE over the post-merge target state: deletes removed, updates applied (fresh inserts are
    // checked separately by `insert_rows` against the already-updated table).
    if table_has_unique_constraint(table, engine)? {
        // Serialize concurrent writers of the same key before the snapshot scan (/ deep-gate
        // #7): two MERGE statements that update different rows to the *same* new key each scan a
        // snapshot blind to the other and would both commit a duplicate. The no-wait key lock over
        // the rows actually written makes the second writer abort, exactly as the INSERT path does.
        lock_unique_keys(table, &new_rows, engine, txn)?;
        let deleted_tids: HashSet<Tid> = deletes.iter().map(|(t, _)| *t).collect();
        let updated: HashMap<Tid, &Row> = updates.iter().map(|(t, _, n)| (*t, n)).collect();
        let mut result_rows: Vec<Row> = Vec::with_capacity(target_rows.len());
        for (tid, row) in target_rows {
            if deleted_tids.contains(tid) {
                continue;
            }
            result_rows.push(
                updated
                    .get(tid)
                    .map_or_else(|| row.clone(), |new| (*new).clone()),
            );
        }
        enforce_unique_over_rows(table, &result_rows, engine)?;
        // A-QA1b: also reject a new key that collides with a row another txn committed after a frozen
        // RR/SER snapshot — the snapshot-based check above cannot see it. The rewritten set is the
        // matched UPDATE *and* DELETE tids (their old keys no longer occupy the space).
        let mut rewritten = deleted_tids;
        rewritten.extend(updates.iter().map(|(tid, _, _)| *tid));
        let old_images: Vec<Row> = updates
            .iter()
            .map(|(_, old, _)| old.clone())
            .chain(deletes.iter().map(|(_, old)| old.clone()))
            .collect();
        enforce_new_keys_vs_committed(table, &rewritten, &old_images, &new_rows, engine, txn)?;
    }
    enforce_fk_on_child_write(table, &new_rows, engine, txn)?;
    enforce_check_on_write(table, &new_rows, engine)?;
    if table_is_fk_parent(table, engine)? {
        let changes: Vec<(Row, Row)> = updates
            .iter()
            .map(|(_, o, n)| (o.clone(), n.clone()))
            .collect();
        enforce_fk_on_parent_update(table, &changes, engine, txn)?;
    }
    for (tid, old, new) in updates {
        let bytes = row::encode(new, &schema)?;
        let new_tid = engine.update(txn, table.id, *tid, &bytes)?;
        if !index_targets.is_empty() {
            remove_departed_index_entries(&index_targets, old, *tid, new, engine, txn)?;
            insert_into_indexes(&index_targets, new, new_tid, engine, txn)?;
        }
    }
    if triggers.has_after_row() {
        for (_, old, new) in updates {
            triggers.fire_row_after(table, Some(old), Some(new), engine, txn)?;
        }
    }
    triggers.fire_stmt_after(table, engine, txn)?;
    if super::ivm::has_views_for_base(engine, txn, &table.name)? {
        let olds: Vec<Row> = updates.iter().map(|(_, o, _)| o.clone()).collect();
        super::ivm::maintain_on_change(&table.name, &new_rows, &olds, engine, txn)?;
    }
    Ok(())
}

/// Evaluate a `RETURNING` projection list against one affected row. The projections'
/// column ordinals index the table's columns, and `row` is in that order.
pub(super) fn project_row(
    returning: &[crate::planner::Projection],
    row: &Row,
) -> Result<Row, Error> {
    returning.iter().map(|p| eval::eval(&p.expr, row)).collect()
}

pub(super) fn apply_assignments(
    assignments: &[Assignment],
    table: &TableSchema,
    mut row: Row,
) -> Result<Row, Error> {
    for assignment in assignments {
        let value = eval::eval(&assignment.value, &row)?;
        let column = column_at(table, assignment.column)?;
        if matches!(value, ast::Value::Null) && !column.nullable {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        set_at(&mut row, assignment.column, value)?;
    }
    Ok(row)
}

/// Like [`apply_assignments`], but each value is evaluated against a separate context row `ctx` — the
/// concatenated `target ++ from` row of `UPDATE ... FROM` — while the result is written into
/// `row` (the target row). Every RHS sees the same pre-update context, so the assignments are
/// simultaneous.
fn apply_assignments_ctx(
    assignments: &[Assignment],
    table: &TableSchema,
    mut row: Row,
    ctx: &Row,
) -> Result<Row, Error> {
    for assignment in assignments {
        let value = eval::eval(&assignment.value, ctx)?;
        let column = column_at(table, assignment.column)?;
        if matches!(value, ast::Value::Null) && !column.nullable {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        set_at(&mut row, assignment.column, value)?;
    }
    Ok(row)
}

pub(super) fn predicate_matches(
    filter: Option<&crate::planner::TypedExpr>,
    row: &Row,
) -> Result<bool, Error> {
    match filter {
        Some(expr) => Ok(matches!(eval::eval(expr, row)?, ast::Value::Bool(true))),
        None => Ok(true),
    }
}
