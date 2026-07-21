//! Trigger persistence + firing.
//!
//! A trigger is a triggered SQL statement attached to a table's `INSERT`/`UPDATE`/`DELETE`. Its
//! definition (timing, events, granularity, optional `WHEN` guard, and action SQL) is persisted in
//! an engine-scoped system catalog `nusadb_triggers`, mirroring the view/policy catalog pattern — no
//! storage-spine change. When the owning table is written, the DML executor ([`super::dml`]) loads
//! the relevant triggers once and fires them: per affected row (`FOR EACH ROW`, with `NEW`/`OLD`
//! bound) or once per statement (`FOR EACH STATEMENT`).
//!
//! `NEW.col` / `OLD.col` references in the action and `WHEN` bodies are bound by substituting them
//! with the affected row's literal values ([`substitute_row_refs`]) before the body is analyzed and
//! run re-entrantly in the same transaction. A thread-local depth guard ([`DepthGuard`]) bounds
//! cascading triggers so a (possibly mutual) self-firing trigger aborts rather than overflowing the
//! stack.
#![allow(clippy::wildcard_imports)]

use std::cell::Cell;

use super::*;
use crate::planner::{AlterTriggerPlan, CreateTriggerPlan, DropTriggerPlan};

/// Engine-scoped system catalog of trigger definitions. Eight text columns:
/// `(name, table, timing, events, for_each, when, action, enabled)`. `when` is empty when there is
/// no guard; `events` is a comma-separated list of canonical event keywords; `enabled` is `"t"` /
/// `"f"` (`ALTER TABLE ... {ENABLE|DISABLE} TRIGGER`). Created lazily — no treaty change. A catalog
/// created before the `enabled` column existed has seven columns; [`ensure_trigger_catalog`]
/// upgrades it in place on the next trigger DDL, and every reader tolerates the legacy width via
/// [`decode_catalog_row`] (a legacy row is enabled).
const TRIGGER_CATALOG: &str = "nusadb_triggers";

/// The eight-text-column schema of [`TRIGGER_CATALOG`].
const TRIGGER_CATALOG_SCHEMA: [ColumnType; 8] = [ColumnType::Text; 8];

/// The pre-`enabled` seven-column schema, kept for reading rows written before the upgrade.
const TRIGGER_CATALOG_SCHEMA_LEGACY: [ColumnType; 7] = [ColumnType::Text; 7];

/// Decode one trigger-catalog row, tolerating the legacy seven-column width: a row written before
/// the `enabled` column existed decodes against the legacy schema and is padded with `"t"`
/// (enabled), which was the only behavior back then. The eight-column decode is tried first, so a
/// current row can never be mistaken for a legacy one.
fn decode_catalog_row(bytes: &[u8]) -> Result<Vec<ast::Value>, Error> {
    row::decode(bytes, &TRIGGER_CATALOG_SCHEMA).or_else(|_| {
        let mut row = row::decode(bytes, &TRIGGER_CATALOG_SCHEMA_LEGACY)?;
        row.push(ast::Value::Text("t".to_owned()));
        Ok(row)
    })
}

/// Maximum nesting depth for cascading trigger actions.
const MAX_TRIGGER_DEPTH: usize = 64;

thread_local! {
    /// Current trigger nesting depth on this thread, used to bound cascades.
    static TRIGGER_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard that increments the trigger depth on entry and decrements on drop, refusing to enter
/// past [`MAX_TRIGGER_DEPTH`].
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Result<Self, Error> {
        TRIGGER_DEPTH.with(|depth| {
            let current = depth.get();
            if current >= MAX_TRIGGER_DEPTH {
                return Err(Error::TriggerRecursionLimit {
                    limit: MAX_TRIGGER_DEPTH,
                });
            }
            depth.set(current + 1);
            Ok(Self)
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        TRIGGER_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// A trigger definition decoded from the catalog (its owning table is implied by the lookup).
struct StoredTrigger {
    name: String,
    timing: ast::TriggerTiming,
    events: Vec<ast::TriggerEvent>,
    for_each: ast::TriggerForEach,
    when: Option<String>,
    action: String,
    /// Whether the trigger fires (`ALTER TABLE ... DISABLE TRIGGER` flips this off).
    enabled: bool,
}

// === DDL: CREATE / DROP TRIGGER ===========================================

/// `CREATE [OR REPLACE] TRIGGER ...`: persist the definition in the trigger catalog. Without
/// `OR REPLACE`, a same-named trigger on the table is an error.
pub(super) fn run_create_trigger(
    plan: &CreateTriggerPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if !plan.or_replace && trigger_exists(engine, txn, &plan.table, &plan.name)? {
        return Err(Error::TriggerExists {
            name: plan.name.clone(),
            table: plan.table.clone(),
        });
    }
    let cat = ensure_trigger_catalog(engine, txn)?;
    delete_trigger_row(engine, txn, &plan.table, &plan.name)?;
    let events = plan
        .events
        .iter()
        .map(|e| e.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let row = [
        ast::Value::Text(plan.name.clone()),
        ast::Value::Text(plan.table.clone()),
        ast::Value::Text(plan.timing.as_str().to_owned()),
        ast::Value::Text(events),
        ast::Value::Text(plan.for_each.as_str().to_owned()),
        ast::Value::Text(plan.when.clone().unwrap_or_default()),
        ast::Value::Text(plan.action.clone()),
        ast::Value::Text("t".to_owned()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &TRIGGER_CATALOG_SCHEMA)?)?;
    Ok(ExecutionResult::TriggerCreated)
}

/// `ALTER TRIGGER name ON table RENAME TO new_name`: rewrite the catalog row under the new name,
/// preserving every other field (including the enabled flag). The old name must exist and the new
/// name must be free on the same table.
pub(super) fn run_alter_trigger(
    plan: &AlterTriggerPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if plan.new_name != plan.name && trigger_exists(engine, txn, &plan.table, &plan.new_name)? {
        return Err(Error::TriggerExists {
            name: plan.new_name.clone(),
            table: plan.table.clone(),
        });
    }
    let cat = ensure_trigger_catalog(engine, txn)?;
    let mut renamed = false;
    let mut scan = engine.scan(txn, cat)?;
    let mut rewrites = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        let mut row = decode_catalog_row(&bytes)?;
        if row_matches(&row, &plan.table, &plan.name) {
            *row.first_mut().ok_or_else(|| internal_index(0))? =
                ast::Value::Text(plan.new_name.clone());
            rewrites.push((tid, row::encode(&row, &TRIGGER_CATALOG_SCHEMA)?));
            renamed = true;
        }
    }
    drop(scan);
    for (tid, bytes) in rewrites {
        engine.update(txn, cat, tid, &bytes)?;
    }
    if !renamed {
        return Err(Error::TriggerNotFound {
            name: plan.name.clone(),
            table: plan.table.clone(),
        });
    }
    Ok(ExecutionResult::TriggerAltered)
}

/// `ALTER TABLE table {ENABLE|DISABLE} TRIGGER {name|ALL}`: flip the enabled flag on the matching
/// catalog row(s). A named trigger must exist; `ALL` (`name == None`) succeeds even when the table
/// has no triggers, matching the reference behavior.
pub(super) fn set_triggers_enabled(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    name: Option<&str>,
    enabled: bool,
) -> Result<(), Error> {
    // `ALL` on a catalog that does not even exist yet is a no-op; a named trigger is not found.
    if engine.lookup_table_as_of(txn, TRIGGER_CATALOG)?.is_none() {
        return name.map_or(Ok(()), |name| {
            Err(Error::TriggerNotFound {
                name: name.to_owned(),
                table: table.to_owned(),
            })
        });
    }
    // Route through `ensure` so a legacy seven-column catalog is upgraded before rows are
    // rewritten at the eight-column width.
    let cat = ensure_trigger_catalog(engine, txn)?;
    let flag = ast::Value::Text(if enabled { "t" } else { "f" }.to_owned());
    let mut matched = false;
    let mut scan = engine.scan(txn, cat)?;
    let mut rewrites = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        let mut row = decode_catalog_row(&bytes)?;
        let is_target = name.map_or_else(
            || matches!(row.get(1), Some(ast::Value::Text(t)) if t == table),
            |name| row_matches(&row, table, name),
        );
        if is_target {
            *row.get_mut(7).ok_or_else(|| internal_index(7))? = flag.clone();
            rewrites.push((tid, row::encode(&row, &TRIGGER_CATALOG_SCHEMA)?));
            matched = true;
        }
    }
    drop(scan);
    for (tid, bytes) in rewrites {
        engine.update(txn, cat, tid, &bytes)?;
    }
    if !matched && let Some(name) = name {
        return Err(Error::TriggerNotFound {
            name: name.to_owned(),
            table: table.to_owned(),
        });
    }
    Ok(())
}

/// `DROP TRIGGER [IF EXISTS] name ON table`.
pub(super) fn run_drop_trigger(
    plan: &DropTriggerPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let removed = delete_trigger_row(engine, txn, &plan.table, &plan.name)?;
    if !removed && !plan.if_exists {
        return Err(Error::TriggerNotFound {
            name: plan.name.clone(),
            table: plan.table.clone(),
        });
    }
    Ok(ExecutionResult::TriggerDropped)
}

/// Look up the trigger catalog, creating it (lazily) if it does not exist yet. A legacy
/// seven-column catalog (created before the `enabled` column existed) is upgraded in place:
/// every row is rewritten at the eight-column width (enabled) and the declared schema gains the
/// `enabled` column, so a plain `SELECT * FROM nusadb_triggers` decodes cleanly afterwards.
fn ensure_trigger_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? {
        if schema.columns.len() < 8 {
            upgrade_legacy_catalog(engine, txn, schema.id)?;
        }
        return Ok(schema.id);
    }
    let columns = [
        "name", "table", "timing", "events", "for_each", "when", "action", "enabled",
    ]
    .into_iter()
    .map(|name| ColumnDef {
        name: name.to_owned(),
        ty: ColumnType::Text,
        nullable: false,
    })
    .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: TRIGGER_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Upgrade a legacy seven-column trigger catalog: rewrite every row at the eight-column width
/// (legacy rows decode padded with enabled = `"t"`) and add the `enabled` column to the declared
/// schema, mirroring what `ALTER TABLE ADD COLUMN` does for user tables (rows first, then the
/// schema, all inside this transaction).
fn upgrade_legacy_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
    cat: nusadb_core::TableId,
) -> Result<(), Error> {
    let mut rewrites = Vec::new();
    let mut scan = engine.scan(txn, cat)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        rewrites.push((tid, row::encode(&row, &TRIGGER_CATALOG_SCHEMA)?));
    }
    drop(scan);
    for (tid, bytes) in rewrites {
        engine.update(txn, cat, tid, &bytes)?;
    }
    engine.alter_table(
        txn,
        cat,
        &nusadb_core::AlterOp::AddColumn(ColumnDef {
            name: "enabled".to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        }),
    )?;
    Ok(())
}

/// Whether a trigger named `name` exists on `table`.
fn trigger_exists(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(false);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        if row_matches(&row, table, name) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Remove the `(table, name)` trigger row, returning whether one was deleted.
fn delete_trigger_row(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        if row_matches(&row, table, name) {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}

/// Whether a decoded catalog row is the trigger `(table, name)`.
fn row_matches(row: &[ast::Value], table: &str, name: &str) -> bool {
    matches!(
        (row.first(), row.get(1)),
        (Some(ast::Value::Text(n)), Some(ast::Value::Text(t))) if n == name && t == table
    )
}

// === Firing ===============================================================

/// The triggers relevant to one DML statement on a table, partitioned by timing × granularity, loaded
/// once per statement so the per-row firing loop carries no catalog cost.
pub(super) struct TriggerSet {
    before_row: Vec<StoredTrigger>,
    after_row: Vec<StoredTrigger>,
    before_stmt: Vec<StoredTrigger>,
    after_stmt: Vec<StoredTrigger>,
}

impl TriggerSet {
    /// Whether no trigger of any timing/granularity fires for this statement — the streaming
    /// `INSERT ... SELECT` precondition (statement-level triggers must fire exactly once, which a
    /// per-batch [`insert_rows`](super::dml) pass cannot guarantee).
    pub(super) const fn is_empty(&self) -> bool {
        self.before_row.is_empty()
            && self.after_row.is_empty()
            && self.before_stmt.is_empty()
            && self.after_stmt.is_empty()
    }

    /// Whether any per-row trigger fires before the write (gates the before-row loop).
    pub(super) const fn has_before_row(&self) -> bool {
        !self.before_row.is_empty()
    }

    /// Whether any per-row trigger fires after the write (gates the after-row loop).
    pub(super) const fn has_after_row(&self) -> bool {
        !self.after_row.is_empty()
    }

    /// Whether any per-row trigger fires at all — for `UPDATE`/`DELETE`, the signal that the old row
    /// image must be captured so `OLD.col` can be bound.
    pub(super) const fn needs_old_image(&self) -> bool {
        self.has_before_row() || self.has_after_row()
    }

    /// Fire the `BEFORE ... FOR EACH STATEMENT` triggers (once).
    pub(super) fn fire_stmt_before(
        &self,
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        fire_each(&self.before_stmt, table, None, None, engine, txn)
    }

    /// Fire the `AFTER ... FOR EACH STATEMENT` triggers (once).
    pub(super) fn fire_stmt_after(
        &self,
        table: &TableSchema,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        fire_each(&self.after_stmt, table, None, None, engine, txn)
    }

    /// Fire the `BEFORE ... FOR EACH ROW` triggers for one affected row.
    pub(super) fn fire_row_before(
        &self,
        table: &TableSchema,
        old: Option<&[ast::Value]>,
        new: Option<&[ast::Value]>,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        fire_each(&self.before_row, table, old, new, engine, txn)
    }

    /// Fire the `AFTER ... FOR EACH ROW` triggers for one affected row.
    pub(super) fn fire_row_after(
        &self,
        table: &TableSchema,
        old: Option<&[ast::Value]>,
        new: Option<&[ast::Value]>,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        fire_each(&self.after_row, table, old, new, engine, txn)
    }
}

/// Load the triggers on `table` that fire on `event`, partitioned by timing × granularity. The fast
/// path (no trigger catalog, or no matching trigger) costs a single catalog lookup.
pub(super) fn load_table_triggers(
    table: &str,
    event: ast::TriggerEvent,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<TriggerSet, Error> {
    let mut set = TriggerSet {
        before_row: Vec::new(),
        after_row: Vec::new(),
        before_stmt: Vec::new(),
        after_stmt: Vec::new(),
    };
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(set);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        let Some(trig) = decode_trigger(&row, table)? else {
            continue;
        };
        // A disabled trigger stays in the catalog but never fires
        // (`ALTER TABLE ... DISABLE TRIGGER`).
        if !trig.enabled {
            continue;
        }
        if !trig.events.contains(&event) {
            continue;
        }
        match (trig.timing, trig.for_each) {
            (ast::TriggerTiming::Before, ast::TriggerForEach::Row) => set.before_row.push(trig),
            (ast::TriggerTiming::After, ast::TriggerForEach::Row) => set.after_row.push(trig),
            (ast::TriggerTiming::Before, ast::TriggerForEach::Statement) => {
                set.before_stmt.push(trig);
            },
            (ast::TriggerTiming::After, ast::TriggerForEach::Statement) => {
                set.after_stmt.push(trig);
            },
        }
    }
    // Deterministic firing order: by trigger name within each bucket (SQL leaves it implementation-
    // defined; a stable order keeps results reproducible).
    for bucket in [
        &mut set.before_row,
        &mut set.after_row,
        &mut set.before_stmt,
        &mut set.after_stmt,
    ] {
        bucket.sort_by(|a, b| a.name.cmp(&b.name));
    }
    Ok(set)
}

/// Decode one catalog row into a [`StoredTrigger`] if it belongs to `table`; `None` otherwise.
fn decode_trigger(row: &[ast::Value], table: &str) -> Result<Option<StoredTrigger>, Error> {
    let text = |index: usize| -> Result<String, Error> {
        match row.get(index) {
            Some(ast::Value::Text(s)) => Ok(s.clone()),
            _ => Err(Error::MalformedTuple { offset: index }),
        }
    };
    let owner = text(1)?;
    if owner != table {
        return Ok(None);
    }
    let timing = match text(2)?.as_str() {
        "before" => ast::TriggerTiming::Before,
        _ => ast::TriggerTiming::After,
    };
    let events: Vec<ast::TriggerEvent> = text(3)?
        .split(',')
        .filter_map(ast::TriggerEvent::parse_keyword)
        .collect();
    let for_each = match text(4)?.as_str() {
        "statement" => ast::TriggerForEach::Statement,
        _ => ast::TriggerForEach::Row,
    };
    let when_text = text(5)?;
    let when = if when_text.is_empty() {
        None
    } else {
        Some(when_text)
    };
    Ok(Some(StoredTrigger {
        name: text(0)?,
        timing,
        events,
        for_each,
        when,
        action: text(6)?,
        // Column 7 exists on every row [`decode_catalog_row`] returns (legacy rows are padded
        // enabled); anything but the explicit "f" counts as enabled.
        enabled: text(7)? != "f",
    }))
}

/// Fire each trigger in `bucket` for the given `(old, new)` row binding.
fn fire_each(
    bucket: &[StoredTrigger],
    table: &TableSchema,
    old: Option<&[ast::Value]>,
    new: Option<&[ast::Value]>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for trig in bucket {
        fire_one(trig, table, old, new, engine, txn)?;
    }
    Ok(())
}

/// Fire a single trigger: evaluate its `WHEN` guard (if any) and, if it passes, run its action with
/// `NEW`/`OLD` bound. Runs re-entrantly in the same transaction, behind the recursion guard.
fn fire_one(
    trig: &StoredTrigger,
    table: &TableSchema,
    old: Option<&[ast::Value]>,
    new: Option<&[ast::Value]>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let _guard = DepthGuard::enter()?;
    let refs = RowRefs {
        schema: table,
        old,
        new,
    };
    if let Some(when) = &trig.when
        && !eval_when(when, &refs, engine, txn)?
    {
        return Ok(());
    }
    let mut stmt = crate::parse(&trig.action)?;
    substitute_row_refs(&mut stmt, &refs)?;
    let logical = crate::analyze(stmt, &ExecCatalog { engine, txn })?;
    super::dispatch(crate::plan(logical), engine, txn)?;
    Ok(())
}

/// Evaluate a `WHEN (cond)` guard against the bound row: `SELECT (cond)` after substitution. A `TRUE`
/// result fires the trigger; `FALSE`/`NULL` skip it (SQL three-valued semantics).
fn eval_when(
    when: &str,
    refs: &RowRefs<'_>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    let mut stmt = crate::parse(&format!("SELECT ({when}) AS w"))?;
    substitute_row_refs(&mut stmt, refs)?;
    let logical = crate::analyze(stmt, &ExecCatalog { engine, txn })?;
    match super::dispatch(crate::plan(logical), engine, txn)? {
        ExecutionResult::Rows { rows, .. } => Ok(matches!(
            rows.first().and_then(|r| r.first()),
            Some(ast::Value::Bool(true))
        )),
        _ => Ok(false),
    }
}

// === NEW/OLD substitution =================================================

/// The `NEW`/`OLD` row binding for one firing — resolves `new.col` / `old.col` to literal values.
struct RowRefs<'a> {
    schema: &'a TableSchema,
    old: Option<&'a [ast::Value]>,
    new: Option<&'a [ast::Value]>,
}

impl RowRefs<'_> {
    /// Resolve a qualified column to its bound value: `Some(value)` for a `new.`/`old.` reference,
    /// `None` for any other qualifier (left untouched — it refers to a real table/alias).
    fn resolve(&self, qualifier: &str, column: &str) -> Result<Option<ast::Value>, Error> {
        let (row, which) = match qualifier {
            "new" => (self.new, "NEW"),
            "old" => (self.old, "OLD"),
            _ => return Ok(None),
        };
        let row = row.ok_or_else(|| {
            Error::Unsupported(format!("{which} is not available in this trigger event"))
        })?;
        let index = self
            .schema
            .columns
            .iter()
            .position(|c| c.name == column)
            .ok_or_else(|| Error::ColumnNotFound {
                table: self.schema.name.clone(),
                column: column.to_owned(),
            })?;
        let value = row
            .get(index)
            .cloned()
            .ok_or_else(|| internal_index(index))?;
        Ok(Some(value))
    }
}

/// Replace every `NEW.col` / `OLD.col` reference in `stmt` with the bound row's literal value. Walks
/// the full statement (mirroring [`crate::params`]'s parameter substitution) so references nested in
/// subqueries, `CASE`, function arguments, etc. are all bound.
fn substitute_row_refs(stmt: &mut ast::Statement, refs: &RowRefs<'_>) -> Result<(), Error> {
    match stmt {
        ast::Statement::Select(select) => sub_select(select, refs),
        ast::Statement::SetOperation(set) => sub_set_body(&mut set.body, refs),
        ast::Statement::Insert(insert) => {
            match &mut insert.source {
                ast::InsertSource::Values(rows) => {
                    for row in rows.iter_mut() {
                        // A `None` cell is an explicit `DEFAULT` — nothing to substitute.
                        for expr in row.iter_mut().flatten() {
                            sub_expr(expr, refs)?;
                        }
                    }
                },
                ast::InsertSource::Select(select) => sub_select(select, refs)?,
                ast::InsertSource::DefaultValues => {},
            }
            sub_items(&mut insert.returning, refs)
        },
        ast::Statement::Update(update) => {
            for assignment in &mut update.assignments {
                sub_expr(&mut assignment.value, refs)?;
            }
            sub_from(update.from.as_mut(), refs)?;
            sub_opt(update.filter.as_mut(), refs)?;
            sub_items(&mut update.returning, refs)
        },
        ast::Statement::Delete(delete) => {
            sub_from(delete.using.as_mut(), refs)?;
            sub_opt(delete.filter.as_mut(), refs)?;
            sub_items(&mut delete.returning, refs)
        },
        // A trigger action is validated at CREATE time to be a data statement, so other statement
        // kinds never reach here.
        _ => Ok(()),
    }
}

fn sub_set_body(body: &mut ast::SelectBody, refs: &RowRefs<'_>) -> Result<(), Error> {
    match body {
        ast::SelectBody::Select(select) => sub_select(select, refs),
        ast::SelectBody::SetOp { left, right, .. } => {
            sub_set_body(left, refs)?;
            sub_set_body(right, refs)
        },
    }
}

fn sub_select(select: &mut ast::Select, refs: &RowRefs<'_>) -> Result<(), Error> {
    for cte in &mut select.with {
        match &mut cte.body {
            ast::CteBody::Query(q) => sub_set_body(q, refs)?,
            ast::CteBody::Modifying(stmt) => substitute_row_refs(stmt, refs)?,
        }
    }
    if let Some(ast::Distinct::On(exprs)) = &mut select.distinct {
        for expr in exprs {
            sub_expr(expr, refs)?;
        }
    }
    for item in &mut select.projection {
        if let ast::SelectItem::Expr { expr, .. } = item {
            sub_expr(expr, refs)?;
        }
    }
    sub_from(select.from.as_mut(), refs)?;
    sub_opt(select.filter.as_mut(), refs)?;
    sub_group_by(&mut select.group_by, refs)?;
    sub_opt(select.having.as_mut(), refs)?;
    for order in &mut select.order_by {
        sub_expr(&mut order.expr, refs)?;
    }
    Ok(())
}

fn sub_from(from: Option<&mut ast::FromClause>, refs: &RowRefs<'_>) -> Result<(), Error> {
    if let Some(from) = from {
        sub_table_ref(&mut from.base, refs)?;
        for join in &mut from.joins {
            sub_table_ref(&mut join.table, refs)?;
            if let ast::JoinCondition::On(expr) = &mut join.condition {
                sub_expr(expr, refs)?;
            }
        }
    }
    Ok(())
}

/// Substitute `NEW`/`OLD` row references inside a FROM item: a derived-table subquery, the cell
/// expressions of a `(VALUES ...)` derived table, or a `(SELECT ... UNION ...)` set-op body.
fn sub_table_ref(table: &mut ast::TableRef, refs: &RowRefs<'_>) -> Result<(), Error> {
    if let Some(subquery) = &mut table.subquery {
        sub_select(subquery, refs)?;
    }
    if let Some(values) = &mut table.values {
        for cell in values.iter_mut().flatten() {
            sub_expr(cell, refs)?;
        }
    }
    if let Some(set_op) = &mut table.set_op {
        sub_set_body(&mut set_op.body, refs)?;
    }
    Ok(())
}

fn sub_items(items: &mut [ast::SelectItem], refs: &RowRefs<'_>) -> Result<(), Error> {
    for item in items {
        if let ast::SelectItem::Expr { expr, .. } = item {
            sub_expr(expr, refs)?;
        }
    }
    Ok(())
}

fn sub_group_by(group_by: &mut ast::GroupBy, refs: &RowRefs<'_>) -> Result<(), Error> {
    match group_by {
        ast::GroupBy::Expressions(keys) => {
            for key in keys {
                sub_expr(key, refs)?;
            }
        },
        ast::GroupBy::Rollup(sets)
        | ast::GroupBy::Cube(sets)
        | ast::GroupBy::GroupingSets(sets) => {
            for group in sets {
                for expr in group {
                    sub_expr(expr, refs)?;
                }
            }
        },
    }
    Ok(())
}

fn sub_opt(expr: Option<&mut ast::Expr>, refs: &RowRefs<'_>) -> Result<(), Error> {
    expr.map_or(Ok(()), |e| sub_expr(e, refs))
}

#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive arm per Expr variant; mirrors crate::params substitution"
)]
fn sub_expr(expr: &mut ast::Expr, refs: &RowRefs<'_>) -> Result<(), Error> {
    match expr {
        ast::Expr::QualifiedColumn { table, column } => {
            if let Some(value) = refs.resolve(&table.to_ascii_lowercase(), column)? {
                *expr = ast::Expr::Literal(value);
            }
            Ok(())
        },
        ast::Expr::Literal(_) | ast::Expr::Column(_) | ast::Expr::Parameter(_) => Ok(()),
        ast::Expr::Binary { left, right, .. } | ast::Expr::IsDistinctFrom { left, right, .. } => {
            sub_expr(left, refs)?;
            sub_expr(right, refs)
        },
        ast::Expr::Unary { expr, .. }
        | ast::Expr::IsNull { expr, .. }
        | ast::Expr::IsBool { expr, .. }
        | ast::Expr::Cast { expr, .. } => sub_expr(expr, refs),
        ast::Expr::InList { expr, list, .. } => {
            sub_expr(expr, refs)?;
            for item in list {
                sub_expr(item, refs)?;
            }
            Ok(())
        },
        ast::Expr::Between {
            expr, low, high, ..
        } => {
            sub_expr(expr, refs)?;
            sub_expr(low, refs)?;
            sub_expr(high, refs)
        },
        ast::Expr::Like { expr, pattern, .. }
        | ast::Expr::SimilarTo { expr, pattern, .. }
        | ast::Expr::RegexMatch { expr, pattern, .. } => {
            sub_expr(expr, refs)?;
            sub_expr(pattern, refs)
        },
        ast::Expr::Case {
            operand,
            branches,
            default,
        } => {
            sub_opt(operand.as_deref_mut(), refs)?;
            for branch in branches {
                sub_expr(&mut branch.when, refs)?;
                sub_expr(&mut branch.then, refs)?;
            }
            sub_opt(default.as_deref_mut(), refs)
        },
        ast::Expr::Coalesce(args)
        | ast::Expr::ScalarFunction { args, .. }
        | ast::Expr::FunctionCall { args, .. }
        | ast::Expr::SetReturning { args, .. } => {
            for arg in args {
                sub_expr(arg, refs)?;
            }
            Ok(())
        },
        ast::Expr::Aggregate { arg, filter, .. } => {
            sub_opt(arg.as_deref_mut(), refs)?;
            sub_opt(filter.as_deref_mut(), refs)
        },
        ast::Expr::Encrypt { value, key } | ast::Expr::Decrypt { value, key } => {
            sub_expr(value, refs)?;
            sub_expr(key, refs)
        },
        ast::Expr::ScalarSubquery(select)
        | ast::Expr::Exists {
            subquery: select, ..
        } => sub_select(select, refs),
        ast::Expr::InSubquery { expr, subquery, .. }
        | ast::Expr::QuantifiedComparison { expr, subquery, .. } => {
            sub_expr(expr, refs)?;
            sub_select(subquery, refs)
        },
        ast::Expr::QuantifiedArray { expr, array, .. } => {
            sub_expr(expr, refs)?;
            sub_expr(array, refs)
        },
        ast::Expr::Row(items) | ast::Expr::ArrayLiteral(items) => {
            for item in items {
                sub_expr(item, refs)?;
            }
            Ok(())
        },
        ast::Expr::Subscript { base, index } => {
            sub_expr(base, refs)?;
            sub_expr(index, refs)
        },
        ast::Expr::ArraySlice { base, lower, upper } => {
            sub_expr(base, refs)?;
            for bound in [lower, upper].into_iter().flatten() {
                sub_expr(bound, refs)?;
            }
            Ok(())
        },
        ast::Expr::WindowFunction(wf) => {
            for arg in &mut wf.args {
                sub_expr(arg, refs)?;
            }
            for partition in &mut wf.partition {
                sub_expr(partition, refs)?;
            }
            for order in &mut wf.order {
                sub_expr(&mut order.expr, refs)?;
            }
            sub_frame(wf.frame.as_mut(), refs)
        },
        ast::Expr::WithinGroup(wg) => {
            for arg in &mut wg.args {
                sub_expr(arg, refs)?;
            }
            for order in &mut wg.order_by {
                sub_expr(&mut order.expr, refs)?;
            }
            Ok(())
        },
    }
}

fn sub_frame(frame: Option<&mut ast::WindowFrame>, refs: &RowRefs<'_>) -> Result<(), Error> {
    let Some(frame) = frame else { return Ok(()) };
    sub_frame_bound(&mut frame.start, refs)?;
    if let Some(end) = &mut frame.end {
        sub_frame_bound(end, refs)?;
    }
    Ok(())
}

fn sub_frame_bound(bound: &mut ast::WindowFrameBound, refs: &RowRefs<'_>) -> Result<(), Error> {
    match bound {
        ast::WindowFrameBound::Preceding(e) | ast::WindowFrameBound::Following(e) => {
            sub_expr(e, refs)
        },
        ast::WindowFrameBound::UnboundedPreceding
        | ast::WindowFrameBound::CurrentRow
        | ast::WindowFrameBound::UnboundedFollowing => Ok(()),
    }
}
