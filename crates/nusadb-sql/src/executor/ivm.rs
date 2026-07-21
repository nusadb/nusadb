//! Incremental view maintenance for materialized views.
//!
//! For a materialized view whose body is a single-table projection + filter (the IVM-eligible shape
//! the analyzer marks via `ivm_base`), each base row contributes exactly one view row, so the view is
//! a *bag* of projected rows in bijection with the filtered base rows. That makes maintenance exact:
//! a base insert appends the projected row (when it passes the filter), and a base delete removes one
//! matching projected row — leaving the view byte-for-byte what a full `REFRESH` would produce, but
//! without rescanning the base table.
//!
//! Eligible views are recorded in an engine-scoped `nusadb_ivm_views` catalog `(name, base_table)`.
//! The DML executor calls [`maintain_on_change`] after writing base rows; views over other shapes
//! (joins, aggregates, …) are simply not registered and stay full-refresh-only. Maintenance writes
//! the view's backing table directly (not through the DML path), so it never re-fires triggers or
//! cascades into other views.
#![allow(clippy::wildcard_imports)]

use super::*;
use crate::planner::{LogicalPlan, SelectPlan};

/// Engine-scoped catalog of incrementally-maintained views: `(view name, base table)` text columns.
const IVM_CATALOG: &str = "nusadb_ivm_views";

/// The two-text-column schema of [`IVM_CATALOG`].
const IVM_CATALOG_SCHEMA: [ColumnType; 2] = [ColumnType::Text, ColumnType::Text];

/// Look up the IVM catalog, creating it (lazily) if it does not exist yet.
fn ensure_ivm_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, IVM_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["name", "base_table"]
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: IVM_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Record that materialized view `name` is incrementally maintainable over `base_table`.
pub(super) fn register_ivm_view(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
    base_table: &str,
) -> Result<(), Error> {
    let cat = ensure_ivm_catalog(engine, txn)?;
    unregister_ivm_view(engine, txn, name)?;
    let row = [
        ast::Value::Text(name.to_owned()),
        ast::Value::Text(base_table.to_owned()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &IVM_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Remove view `name` from the IVM catalog (on `DROP`), returning whether a row was removed.
pub(super) fn unregister_ivm_view(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, IVM_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &IVM_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == name) {
            victims.push(tid);
        }
    }
    let removed = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(removed)
}

/// Maintain every view registered over `base_table` for one statement's row delta. The fast
/// path (no IVM catalog, or no view over this table) costs a single catalog lookup.
pub(super) fn maintain_on_change(
    base_table: &str,
    inserted: &[Row],
    deleted: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if inserted.is_empty() && deleted.is_empty() {
        return Ok(());
    }
    let views = views_for_base(engine, txn, base_table)?;
    for view in views {
        maintain_one(&view, inserted, deleted, engine, txn)?;
    }
    Ok(())
}

/// Whether any view is registered over `base_table` — the signal that an `UPDATE` must capture the
/// old row image so the delete side of the delta can be computed.
pub(super) fn has_views_for_base(
    engine: &dyn StorageEngine,
    txn: TxnId,
    base_table: &str,
) -> Result<bool, Error> {
    Ok(!views_for_base(engine, txn, base_table)?.is_empty())
}

/// The names of the views registered over `base_table`.
fn views_for_base(
    engine: &dyn StorageEngine,
    txn: TxnId,
    base_table: &str,
) -> Result<Vec<String>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, IVM_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &IVM_CATALOG_SCHEMA)?;
        if let [ast::Value::Text(name), ast::Value::Text(base)] = row.as_slice()
            && base == base_table
        {
            out.push(name.clone());
        }
    }
    Ok(out)
}

/// Apply the base-row delta to one view: re-derive its filter + projection from the stored definition,
/// then append projected inserts and remove one matching projected row per delete.
fn maintain_one(
    view: &str,
    inserted: &[Row],
    deleted: &[Row],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(def_sql) = super::load_view_def(engine, txn, super::MATVIEW_CATALOG, view)? else {
        return Ok(()); // The view was dropped; nothing to maintain.
    };
    let LogicalPlan::Select(select) =
        crate::analyze(crate::parse(&def_sql)?, &ExecCatalog { engine, txn })?
    else {
        return Ok(()); // Not a SELECT — never IVM-registered, so unreachable in practice.
    };
    let Some(backing) = engine.lookup_table_as_of(txn, view)? else {
        return Ok(());
    };
    let schema = column_types(&backing);

    // Deletes first, so a delete + re-insert of an identical projected row in the same statement does
    // not remove the freshly-inserted copy.
    for base_row in deleted {
        if let Some(view_row) = project_if_match(&select, base_row)? {
            delete_one_matching(engine, txn, backing.id, &schema, &view_row)?;
        }
    }
    // Maintain the backing table's indexes for appended rows: a matview can carry explicit
    // indexes, and an index scan must not miss incrementally-added rows.
    let index_targets = super::dml::secondary_index_targets(&backing, engine)?;
    for base_row in inserted {
        if let Some(view_row) = project_if_match(&select, base_row)? {
            let tid = engine.insert(txn, backing.id, &row::encode(&view_row, &schema)?)?;
            super::dml::insert_into_indexes(&index_targets, &view_row, tid, engine, txn)?;
        }
    }
    Ok(())
}

/// Project a base row to its view row if it passes the view's filter; `None` if the filter rejects it.
fn project_if_match(select: &SelectPlan, base_row: &Row) -> Result<Option<Row>, Error> {
    let matches = match &select.filter {
        None => true,
        Some(filter) => eval::eval(filter, base_row)? == ast::Value::Bool(true),
    };
    if !matches {
        return Ok(None);
    }
    let view_row = select
        .projection
        .iter()
        .map(|p| eval::eval(&p.expr, base_row))
        .collect::<Result<Row, _>>()?;
    Ok(Some(view_row))
}

/// Delete one backing row equal to `view_row` (bag semantics: remove a single occurrence).
fn delete_one_matching(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table_id: nusadb_core::TableId,
    schema: &[ColumnType],
    view_row: &[ast::Value],
) -> Result<(), Error> {
    let mut scan = engine.scan(txn, table_id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        if row::decode(&bytes, schema)? == view_row {
            engine.delete(txn, table_id, tid)?;
            return Ok(());
        }
    }
    Ok(())
}
