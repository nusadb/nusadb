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
use crate::parser::{ScriptBlock, ScriptStmt};
use crate::planner::{AlterTriggerPlan, CreateTriggerPlan, DropTriggerPlan};

/// Engine-scoped system catalog of trigger definitions. Eight text columns:
/// `(name, table, timing, events, for_each, when, action, enabled)`. `when` is empty when there is
/// no guard; `events` is a comma-separated list of canonical event keywords; `enabled` is `"t"` /
/// `"f"` (`ALTER TABLE ... {ENABLE|DISABLE} TRIGGER`). Created lazily — no treaty change. A catalog
/// created before the `enabled` column existed has seven columns; [`ensure_trigger_catalog`]
/// upgrades it in place on the next trigger DDL, and every reader tolerates the legacy width via
/// [`decode_catalog_row`] (a legacy row is enabled).
// `pub(super)` so the rename guard can scan for triggers whose WHEN/body name a column.
pub(super) const TRIGGER_CATALOG: &str = "nusadb_triggers";

/// The nine-text-column schema of [`TRIGGER_CATALOG`].
const TRIGGER_CATALOG_SCHEMA: [ColumnType; 9] = [ColumnType::Text; 9];

/// The pre-`schema` eight-column schema, kept for reading rows written before the upgrade.
const TRIGGER_CATALOG_SCHEMA_PRE_SCHEMA: [ColumnType; 8] = [ColumnType::Text; 8];

/// The pre-`enabled` seven-column schema, kept for reading rows written before that upgrade.
const TRIGGER_CATALOG_SCHEMA_LEGACY: [ColumnType; 7] = [ColumnType::Text; 7];

/// Decode one trigger-catalog row, tolerating both earlier widths. The widest decode is tried
/// first, so a current row can never be mistaken for an older one.
///
/// A pre-`enabled` row is padded with `"t"`, the only behaviour back then.
///
/// A pre-`schema` row is read as belonging to the default namespace. That is a migration decision
/// and not a recovered fact: such a row names a table but not the namespace, and `CREATE TRIGGER`
/// resolved through the session temp schema and search path, so it could have meant another one.
/// Reading it as any namespace is what let a trigger declared on one table fire on a same-named
/// table somewhere else — the behaviour this column exists to stop — so the ambiguity is resolved
/// towards the narrower reading. A trigger created on a table outside the default namespace before
/// this column existed must be recreated.
/// [`decode_catalog_row`] for the cross-catalog schema purge, which lives in the parent module.
pub(super) fn decode_catalog_row_for_purge(bytes: &[u8]) -> Result<Vec<ast::Value>, Error> {
    decode_catalog_row(bytes)
}

fn decode_catalog_row(bytes: &[u8]) -> Result<Vec<ast::Value>, Error> {
    if let Ok(row) = row::decode(bytes, &TRIGGER_CATALOG_SCHEMA) {
        return Ok(row);
    }
    if let Ok(mut row) = row::decode(bytes, &TRIGGER_CATALOG_SCHEMA_PRE_SCHEMA) {
        row.push(ast::Value::Text(
            nusadb_core::engine::PUBLIC_SCHEMA.to_owned(),
        ));
        return Ok(row);
    }
    let mut row = row::decode(bytes, &TRIGGER_CATALOG_SCHEMA_LEGACY)?;
    row.push(ast::Value::Text("t".to_owned()));
    row.push(ast::Value::Text(
        nusadb_core::engine::PUBLIC_SCHEMA.to_owned(),
    ));
    Ok(row)
}

/// Whether a decoded trigger row is attached to `schema.table`.
fn trigger_row_is_for(row: &[ast::Value], schema: &str, table: &str) -> bool {
    matches!((row.get(1), row.get(8)),
        (Some(ast::Value::Text(t)), Some(ast::Value::Text(s))) if t == table && s == schema)
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
    if !plan.or_replace && trigger_exists(engine, txn, &plan.schema, &plan.table, &plan.name)? {
        return Err(Error::TriggerExists {
            name: plan.name.clone(),
            table: plan.table.clone(),
        });
    }
    check_partition_trigger_name(
        &plan.schema,
        &plan.table,
        &plan.name,
        matches!(plan.for_each, ast::TriggerForEach::Row),
        engine,
        txn,
    )?;
    // `INSTEAD OF` attaches to a VIEW (it replaces the write), is row-level only, and takes no
    // `WHEN` guard; `BEFORE`/`AFTER` attach to a real table — the reference engine's rules.
    let view_key = crate::analyzer::qualified_display(&plan.schema, &plan.table);
    let is_view = super::lookup_view_definition(engine, txn, &view_key)?.is_some();
    if matches!(plan.timing, ast::TriggerTiming::InsteadOf) {
        if !is_view {
            return Err(Error::Coded {
                message: format!(
                    "\"{}\" is a table — INSTEAD OF triggers attach to views",
                    plan.table
                ),
                sqlstate: "42809", // wrong_object_type
            });
        }
        if !matches!(plan.for_each, ast::TriggerForEach::Row) {
            return Err(Error::Coded {
                message: "INSTEAD OF triggers must be FOR EACH ROW".to_owned(),
                sqlstate: "0A000",
            });
        }
        if plan.when.is_some() {
            return Err(Error::Coded {
                message: "INSTEAD OF triggers cannot have WHEN conditions".to_owned(),
                sqlstate: "0A000",
            });
        }
    } else if is_view {
        return Err(Error::Coded {
            message: format!(
                "\"{}\" is a view — views need INSTEAD OF triggers, not BEFORE/AFTER",
                plan.table
            ),
            sqlstate: "42809", // wrong_object_type
        });
    }
    // An `EXECUTE FUNCTION` action names a function that must already exist and be a callable
    // NusaScript routine — validate now (as the reference engine does at CREATE TRIGGER time)
    // rather than only when the trigger first fires.
    if let Some(func_name) = execute_function_target(&plan.action) {
        load_trigger_function(func_name, engine, txn)?;
    }
    let cat = ensure_trigger_catalog(engine, txn)?;
    delete_trigger_row(engine, txn, &plan.schema, &plan.table, &plan.name)?;
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
        ast::Value::Text(plan.schema.clone()),
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
    if plan.new_name != plan.name
        && trigger_exists(engine, txn, &plan.schema, &plan.table, &plan.new_name)?
    {
        return Err(Error::TriggerExists {
            name: plan.new_name.clone(),
            table: plan.table.clone(),
        });
    }
    refuse_inherited_trigger(&plan.schema, &plan.table, &plan.name, "rename", engine, txn)?;
    if plan.new_name != plan.name {
        let row = has_row_trigger(engine, txn, &plan.schema, &plan.table, &plan.name)?;
        check_partition_trigger_name(&plan.schema, &plan.table, &plan.new_name, row, engine, txn)?;
    }
    let cat = ensure_trigger_catalog(engine, txn)?;
    let mut renamed = false;
    let mut scan = engine.scan(txn, cat)?;
    let mut rewrites = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        let mut row = decode_catalog_row(&bytes)?;
        if row_matches(&row, &plan.schema, &plan.table, &plan.name) {
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
    schema: &str,
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
            || trigger_row_is_for(&row, schema, table),
            |name| row_matches(&row, schema, table, name),
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
    refuse_inherited_trigger(&plan.schema, &plan.table, &plan.name, "drop", engine, txn)?;
    let removed = delete_trigger_row(engine, txn, &plan.schema, &plan.table, &plan.name)?;
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
        if schema.columns.len() < 9 {
            upgrade_legacy_catalog(engine, txn, schema.id, schema.columns.len())?;
        }
        return Ok(schema.id);
    }
    let columns = [
        "name", "table", "timing", "events", "for_each", "when", "action", "enabled", "schema",
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
    present: usize,
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
    // Add whichever of the later columns this catalog is missing, oldest first, so a catalog at
    // either earlier width lands at the current one.
    for name in ["enabled", "schema"].iter().skip(present.saturating_sub(7)) {
        engine.alter_table(
            txn,
            cat,
            &nusadb_core::AlterOp::AddColumn(ColumnDef {
                name: (*name).to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            }),
        )?;
    }
    Ok(())
}

/// Remove every trigger declared on `schema.table`, for when the table is dropped.
///
/// Without this the rows outlive their table, and a later table of the same name inherits triggers
/// nobody declared on it — the same shape as the orphaned policies the drop already cleans up.
pub(super) fn delete_triggers_for_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        if trigger_row_is_for(&decode_catalog_row(&bytes)?, schema, table) {
            victims.push(tid);
        }
    }
    drop(scan);
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// `ALTER TABLE … RENAME TO`: re-key every trigger of `schema.old` to `schema.new`, so the renamed
/// table keeps firing them and the old name carries none.
pub(super) fn rename_triggers_for_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    old: &str,
    new: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(());
    };
    let mut moving = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        if trigger_row_is_for(&row, schema, old) {
            moving.push((tid, row));
        }
    }
    drop(scan);
    for (tid, mut row) in moving {
        engine.delete(txn, cat.id, tid)?;
        if let Some(table) = row.get_mut(1) {
            *table = ast::Value::Text(new.to_owned());
        }
        // `decode_catalog_row` pads a legacy row to the current width, so every row re-encodes at
        // the nine-column schema.
        let bytes = row::encode(&row, &TRIGGER_CATALOG_SCHEMA)?;
        engine.insert(txn, cat.id, &bytes)?;
    }
    Ok(())
}

/// Whether a trigger named `name` exists on `table`.
fn trigger_exists(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(false);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        if row_matches(&row, schema, table, name) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Remove the `(table, name)` trigger row, returning whether one was deleted.
fn delete_trigger_row(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
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
        if row_matches(&row, schema, table, name) {
            victims.push(tid);
        }
    }
    let deleted = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(deleted)
}

/// Whether a decoded catalog row is the trigger `(schema, table, name)`.
fn row_matches(row: &[ast::Value], schema: &str, table: &str, name: &str) -> bool {
    trigger_row_is_for(row, schema, table)
        && matches!(row.first(), Some(ast::Value::Text(n)) if n == name)
}

// === Firing ===============================================================

/// The triggers relevant to one DML statement on a table, partitioned by timing × granularity, loaded
/// once per statement so the per-row firing loop carries no catalog cost.
pub(super) struct TriggerSet {
    before_row: Vec<StoredTrigger>,
    after_row: Vec<StoredTrigger>,
    before_stmt: Vec<StoredTrigger>,
    after_stmt: Vec<StoredTrigger>,
    /// `INSTEAD OF ... FOR EACH ROW` (views only): fires in place of the write.
    instead_row: Vec<StoredTrigger>,
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
            && self.instead_row.is_empty()
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

    /// This set, keeping its statement-level triggers only when `keep` is set. A statement that
    /// names a partitioned or inheritance parent fires the parent's statement triggers only, not
    /// those of the descendants its rows land in: those parts load their triggers with `keep`
    /// false (their row-level triggers still run).
    #[must_use]
    pub(super) fn statement_triggers_if(mut self, keep: bool) -> Self {
        if !keep {
            self.before_stmt.clear();
            self.after_stmt.clear();
        }
        self
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

    /// Whether any `INSTEAD OF ... FOR EACH ROW` trigger replaces the write.
    pub(super) const fn has_instead_row(&self) -> bool {
        !self.instead_row.is_empty()
    }

    /// Fire the `INSTEAD OF ... FOR EACH ROW` triggers for one proposed row — the write itself.
    pub(super) fn fire_row_instead(
        &self,
        table: &TableSchema,
        old: Option<&[ast::Value]>,
        new: Option<&[ast::Value]>,
        engine: &dyn StorageEngine,
        txn: TxnId,
    ) -> Result<(), Error> {
        fire_each(&self.instead_row, table, old, new, engine, txn)
    }
}

/// Whether the view at `key` (schema-qualified; bare = `public`) has an enabled `INSTEAD OF`
/// trigger for `event` — the analyzer's gate for planning view DML as trigger firings.
pub fn view_has_instead_of_trigger(
    engine: &dyn StorageEngine,
    txn: TxnId,
    key: &str,
    event: ast::TriggerEvent,
) -> Result<bool, Error> {
    let (schema, name) = crate::analyzer::split_qualified(key);
    let set = load_table_triggers(
        schema.unwrap_or(nusadb_core::PUBLIC_SCHEMA),
        name,
        event,
        engine,
        txn,
    )?;
    Ok(set.has_instead_row())
}

/// Load the triggers on `table` that fire on `event`, partitioned by timing × granularity. The fast
/// path (no trigger catalog, or no matching trigger) costs a single catalog lookup.
pub(super) fn load_table_triggers(
    schema: &str,
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
        instead_row: Vec::new(),
    };
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(set);
    };
    // A partition also runs the row-level triggers of every partitioned table above it: a trigger
    // declared on a partitioned parent applies to each row whichever partition holds it, including
    // a partition attached after the trigger was created. Statement-level triggers are not
    // inherited, and neither is anything across a plain INHERITS edge.
    let ancestors = partition_ancestors(schema, table, engine, txn)?;
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        let trig = if let Some(trig) = decode_trigger(&row, schema, table)? {
            trig
        } else {
            let mut inherited = None;
            for (ancestor_schema, ancestor) in &ancestors {
                if let Some(trig) = decode_trigger(&row, ancestor_schema, ancestor)? {
                    inherited = Some(trig);
                    break;
                }
            }
            match inherited {
                Some(trig)
                    if trig.for_each == ast::TriggerForEach::Row
                        && trig.timing != ast::TriggerTiming::InsteadOf =>
                {
                    trig
                },
                _ => continue,
            }
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
            // INSTEAD OF is row-level only (enforced at CREATE); a statement-granularity row
            // in the catalog can only come from a hand-edited store — bucket it the same.
            (ast::TriggerTiming::InsteadOf, _) => set.instead_row.push(trig),
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

/// The partitioned tables above `schema.table`, nearest first, as `(schema, name)`; empty for a
/// table that is not a partition (one cheap probe when the database has no partitioning at all).
pub(super) fn partition_ancestors(
    schema: &str,
    table: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(String, String)>, Error> {
    const MAX_DEPTH: usize = 64;
    let mut out = Vec::new();
    if !super::partition::has_any(engine, txn)? {
        return Ok(out);
    }
    let edges = super::partition::partition_edges(engine, txn)?;
    let mut key = crate::analyzer::qualified_display(schema, table);
    // The depth cap guards a hand-edited catalog with a cycle; DDL cannot create one.
    while out.len() < MAX_DEPTH {
        let Some((_, parent)) = edges.iter().find(|(child, _)| *child == key) else {
            break;
        };
        let (parent_schema, parent_name) = crate::analyzer::split_qualified(parent);
        out.push((
            parent_schema
                .unwrap_or(nusadb_core::PUBLIC_SCHEMA)
                .to_owned(),
            parent_name.to_owned(),
        ));
        key.clone_from(parent);
    }
    Ok(out)
}

/// Every partition below `schema.table`, at any depth, as `(schema, name)`.
fn partition_descendants(
    schema: &str,
    table: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(String, String)>, Error> {
    let mut out = Vec::new();
    if !super::partition::has_any(engine, txn)? {
        return Ok(out);
    }
    let edges = super::partition::partition_edges(engine, txn)?;
    let mut keys = vec![crate::analyzer::qualified_display(schema, table)];
    while let Some(key) = keys.pop() {
        for (child, _) in edges.iter().filter(|(_, parent)| *parent == key) {
            // A guard against a hand-edited catalog with a cycle; DDL cannot create one.
            if out.len() > edges.len() {
                return Ok(out);
            }
            let (child_schema, child_name) = crate::analyzer::split_qualified(child);
            out.push((
                child_schema
                    .unwrap_or(nusadb_core::PUBLIC_SCHEMA)
                    .to_owned(),
                child_name.to_owned(),
            ));
            keys.push(child.clone());
        }
    }
    Ok(out)
}

/// Refuse a trigger named `name` on `schema.table` that would share its name with a trigger it runs
/// or that runs on a partition below it: a partition runs the row-level triggers of the tables above
/// it, so it cannot have one of their names, and a row-level trigger (`row`) on a partitioned table
/// runs on every partition below it, so the name must be free on each of them.
fn check_partition_trigger_name(
    schema: &str,
    table: &str,
    name: &str,
    row: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for (ancestor_schema, ancestor) in partition_ancestors(schema, table, engine, txn)? {
        if has_row_trigger(engine, txn, &ancestor_schema, &ancestor, name)? {
            return Err(Error::TriggerExists {
                name: name.to_owned(),
                table: table.to_owned(),
            });
        }
    }
    if row {
        for (sub_schema, sub) in partition_descendants(schema, table, engine, txn)? {
            if trigger_exists(engine, txn, &sub_schema, &sub, name)? {
                return Err(Error::TriggerExists {
                    name: name.to_owned(),
                    table: sub,
                });
            }
        }
    }
    Ok(())
}

/// Refuse changing (`action`: drop, rename) a trigger named `name` on the partition `schema.table`
/// when the partition only runs it because a table above it has it.
fn refuse_inherited_trigger(
    schema: &str,
    table: &str,
    name: &str,
    action: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if trigger_exists(engine, txn, schema, table, name)? {
        return Ok(());
    }
    for (ancestor_schema, ancestor) in partition_ancestors(schema, table, engine, txn)? {
        if has_row_trigger(engine, txn, &ancestor_schema, &ancestor, name)? {
            return Err(Error::Coded {
                message: format!(
                    "cannot {action} trigger \"{name}\" on table \"{table}\" because trigger \
                     \"{name}\" on table \"{ancestor}\" requires it; {action} it on \"{ancestor}\" \
                     instead"
                ),
                sqlstate: "2BP01", // dependent_objects_still_exist
            });
        }
    }
    Ok(())
}

/// Whether `schema.table` has a row-level trigger named `name`.
fn has_row_trigger(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(false);
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        if let Some(trig) = decode_trigger(&row, schema, table)?
            && trig.name == name
            && trig.for_each == ast::TriggerForEach::Row
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Refuse to make `child` (and the partitions under it) a partition of `parent` when one of them
/// has a trigger named like a row-level trigger of `parent` or a table above it: the partition
/// would run both under one name.
pub(super) fn check_attach_trigger_names(
    parent_schema: &str,
    parent: &str,
    child_schema: &str,
    child: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, TRIGGER_CATALOG)? else {
        return Ok(());
    };
    let mut above = vec![(parent_schema.to_owned(), parent.to_owned())];
    above.extend(partition_ancestors(parent_schema, parent, engine, txn)?);
    let mut below = vec![(child_schema.to_owned(), child.to_owned())];
    below.extend(partition_descendants(child_schema, child, engine, txn)?);
    let mut inherited: Vec<String> = Vec::new();
    let mut own: Vec<(String, String)> = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = decode_catalog_row(&bytes)?;
        for (s, t) in &above {
            if let Some(trig) = decode_trigger(&row, s, t)?
                && trig.for_each == ast::TriggerForEach::Row
            {
                inherited.push(trig.name);
            }
        }
        for (s, t) in &below {
            if let Some(trig) = decode_trigger(&row, s, t)? {
                own.push((trig.name, t.clone()));
            }
        }
    }
    if let Some((name, table)) = own.into_iter().find(|(name, _)| inherited.contains(name)) {
        return Err(Error::TriggerExists { name, table });
    }
    Ok(())
}

/// Decode one catalog row into a [`StoredTrigger`] if it belongs to `table`; `None` otherwise.
fn decode_trigger(
    row: &[ast::Value],
    schema: &str,
    table: &str,
) -> Result<Option<StoredTrigger>, Error> {
    let text = |index: usize| -> Result<String, Error> {
        match row.get(index) {
            Some(ast::Value::Text(s)) => Ok(s.clone()),
            _ => Err(Error::MalformedTuple { offset: index }),
        }
    };
    // The namespace as well as the name: a row filed against `app.t` must not fire on `public.t`.
    // Matching on the name alone is what let a trigger declared on one table run on another user's
    // write to a same-named one, with the writer's privileges.
    if !trigger_row_is_for(row, schema, table) {
        return Ok(None);
    }
    let timing = match text(2)?.as_str() {
        "before" => ast::TriggerTiming::Before,
        "instead of" => ast::TriggerTiming::InsteadOf,
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
    // `EXECUTE FUNCTION name()`: run the function's NusaScript body with `NEW`/`OLD` bound.
    if let Some(func_name) = execute_function_target(&trig.action) {
        return fire_trigger_function(func_name, &refs, engine, txn);
    }
    let mut stmt = crate::parse(&trig.action)?;
    substitute_row_refs(&mut stmt, &refs)?;
    let logical = crate::analyze(stmt, &ExecCatalog::new(engine, txn))?;
    super::dispatch(crate::plan(logical), engine, txn)?;
    Ok(())
}

/// If `action` is the canonical `EXECUTE FUNCTION <name>()` form, return the function name. The
/// action text is generated by the parser in exactly this shape, so an exact prefix/suffix match is
/// sufficient (and can never collide with a data-statement action, which never starts this way).
fn execute_function_target(action: &str) -> Option<&str> {
    let name = action
        .strip_prefix("EXECUTE FUNCTION ")?
        .strip_suffix("()")?;
    (!name.is_empty()).then_some(name)
}

/// Resolve an `EXECUTE FUNCTION` target to its definition, enforcing that it exists and is a callable
/// NusaScript routine that takes no parameters (a trigger passes none). Used both at CREATE TRIGGER
/// time (to fail fast) and at fire time (to run the body).
fn load_trigger_function(
    func_name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<crate::FunctionDef, Error> {
    let def = super::function::lookup_function_definition(engine, txn, func_name)?
        .ok_or_else(|| Error::UnknownFunction(func_name.to_owned()))?;
    if def.language != ast::FunctionLanguage::NusaScript || !crate::parser::is_script(&def.body) {
        return Err(Error::InvalidStatement(format!(
            "trigger function `{func_name}` must be a NusaScript function with a BEGIN … END body"
        )));
    }
    if def.param_count != 0 {
        return Err(Error::InvalidStatement(format!(
            "trigger function `{func_name}` must take no parameters"
        )));
    }
    Ok(def)
}

/// Fire an `EXECUTE FUNCTION` trigger: run the function's NusaScript body in the caller's transaction
/// with `NEW`/`OLD` bound. Side-effect semantics — the body's `RETURN` stops it but its value is
/// discarded, so (like the existing statement-action form) a `BEFORE` trigger cannot modify or skip
/// the row. Runs behind the shared depth guard, so cascades are still bounded.
fn fire_trigger_function(
    func_name: &str,
    refs: &RowRefs<'_>,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let def = load_trigger_function(func_name, engine, txn)?;
    let mut block = crate::parser::parse_script(&def.body)?;
    sub_script_block(&mut block, refs)?;
    super::script::run_block(&block, &[], &[], engine, txn)?;
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
    let logical = crate::analyze(stmt, &ExecCatalog::new(engine, txn))?;
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
            Error::InvalidStatement(format!("{which} is not available in this trigger event"))
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
        | ast::Expr::IsJson { operand: expr, .. }
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
        ast::Expr::Overlaps { s1, e1, s2, e2 } => {
            sub_expr(s1, refs)?;
            sub_expr(e1, refs)?;
            sub_expr(s2, refs)?;
            sub_expr(e2, refs)
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
        ast::Expr::FieldAccess { base: inner, .. } | ast::Expr::CastNamed { expr: inner, .. } => {
            sub_expr(inner, refs)
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
            if let Some(filter) = wf.filter.as_mut() {
                sub_expr(filter, refs)?;
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

// === NEW/OLD substitution in a trigger-function body ======================

/// Substitute `NEW`/`OLD` row references throughout a NusaScript block: every embedded SQL statement
/// and every expression (in `DECLARE`/`SET`/`IF`/`WHILE`/`FOR`/`RAISE`) is bound to the firing row's
/// literal values, and each nested block (and the `EXCEPTION` handler) is walked.
fn sub_script_block(block: &mut ScriptBlock, refs: &RowRefs<'_>) -> Result<(), Error> {
    sub_script_stmts(&mut block.body, refs)?;
    if let Some(handler) = &mut block.handler {
        sub_script_stmts(handler, refs)?;
    }
    Ok(())
}

fn sub_script_stmts(stmts: &mut [ScriptStmt], refs: &RowRefs<'_>) -> Result<(), Error> {
    for stmt in stmts {
        sub_script_stmt(stmt, refs)?;
    }
    Ok(())
}

fn sub_script_stmt(stmt: &mut ScriptStmt, refs: &RowRefs<'_>) -> Result<(), Error> {
    match stmt {
        ScriptStmt::Declare { default, .. } => sub_opt(default.as_mut(), refs),
        ScriptStmt::Assign { value, .. } => sub_expr(value, refs),
        ScriptStmt::If { arms, els } => {
            for (cond, body) in arms {
                sub_expr(cond, refs)?;
                sub_script_stmts(body, refs)?;
            }
            if let Some(els) = els {
                sub_script_stmts(els, refs)?;
            }
            Ok(())
        },
        ScriptStmt::While { cond, body } => {
            sub_expr(cond, refs)?;
            sub_script_stmts(body, refs)
        },
        ScriptStmt::For {
            low, high, body, ..
        } => {
            sub_expr(low, refs)?;
            sub_expr(high, refs)?;
            sub_script_stmts(body, refs)
        },
        ScriptStmt::Perform(expr) | ScriptStmt::Raise(expr) => sub_expr(expr, refs),
        // A trigger function's RETURN value is a row the side-effect firing model discards, so
        // neutralize it to a bare RETURN: it still stops the routine (preserving control flow)
        // without evaluating a `NEW`/`OLD` row reference the executor cannot produce as a value.
        ScriptStmt::Return(value) => {
            *value = None;
            Ok(())
        },
        ScriptStmt::Sql(stmt) => substitute_row_refs(stmt, refs),
        ScriptStmt::Block(block) => sub_script_block(block, refs),
    }
}
