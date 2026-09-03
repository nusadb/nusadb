//! Table inheritance (`INHERITS`) edge catalog + descendant lookup.
//!
//! `CREATE TABLE child (...) INHERITS (parent, ...)` records one `(child, parent, seq)` row per parent
//! in an engine-scoped system catalog `nusadb_inheritance`, mirroring the trigger/function catalog
//! pattern — no storage-spine change. Both endpoints are schema-qualified keys (`schema.name`, bare
//! for `public` — so keys recorded before namespaces existed read back unchanged). A query on a
//! parent then expands to the parent plus its transitive descendants (see the analyzer's scan
//! expansion), unless the reference says `ONLY`.
#![allow(clippy::wildcard_imports)]

use std::collections::HashSet;

use super::*;

/// Engine-scoped system catalog of inheritance edges: `(child, parent, seq)`.
pub(super) const INHERITANCE_CATALOG: &str = "nusadb_inheritance";

/// The three-column schema of [`INHERITANCE_CATALOG`]: `(child TEXT, parent TEXT, seq TEXT)`. `seq`
/// is a stringified index preserving multiple-parent order.
const INHERITANCE_CATALOG_SCHEMA: [ColumnType; 3] =
    [ColumnType::Text, ColumnType::Text, ColumnType::Text];

/// Record that `child` inherits from each of `parents`, in order. Creates the catalog lazily.
pub(super) fn record_inheritance(
    engine: &dyn StorageEngine,
    txn: TxnId,
    child: &str,
    parents: &[String],
) -> Result<(), Error> {
    if parents.is_empty() {
        return Ok(());
    }
    let cat = ensure_catalog(engine, txn)?;
    for (seq, parent) in parents.iter().enumerate() {
        let row = [
            ast::Value::Text(child.to_owned()),
            ast::Value::Text(parent.clone()),
            ast::Value::Text(seq.to_string()),
        ];
        engine.insert(txn, cat, &row::encode(&row, &INHERITANCE_CATALOG_SCHEMA)?)?;
    }
    Ok(())
}

/// The direct children of `parent` (tables that name it in an `INHERITS` clause), each paired with
/// the `seq` under which they recorded the edge — unused by callers today but kept so ordering is
/// available. Returns an empty list when the catalog does not exist yet or `parent` has no children.
pub(super) fn direct_children(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
) -> Result<Vec<String>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, INHERITANCE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut children = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &INHERITANCE_CATALOG_SCHEMA)?;
        if let (Some(ast::Value::Text(c)), Some(ast::Value::Text(p))) = (row.first(), row.get(1))
            && p == parent
        {
            children.push(c.clone());
        }
    }
    Ok(children)
}

/// Whether any inheritance edge exists at all — the cheap gate that lets a database with no
/// inheritance skip per-table descendant probes. Costs a single catalog lookup + first-row read.
pub(super) fn has_any(engine: &dyn StorageEngine, txn: TxnId) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, INHERITANCE_CATALOG)? else {
        return Ok(false);
    };
    Ok(engine.scan(txn, cat.id)?.try_next()?.is_some())
}

/// The transitive descendants of `parent` (children, grandchildren, …), in a stable breadth-first
/// order with each table listed once. `parent` itself is not included. A malformed edge cycle (which
/// DDL cannot create, but a hand-written catalog row could) is bounded by the `visited` set.
pub(super) fn descendants(
    engine: &dyn StorageEngine,
    txn: TxnId,
    parent: &str,
) -> Result<Vec<String>, Error> {
    let mut out = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(parent.to_owned());
    let mut frontier = vec![parent.to_owned()];
    while let Some(current) = frontier.pop() {
        for child in direct_children(engine, txn, &current)? {
            if visited.insert(child.clone()) {
                out.push(child.clone());
                frontier.push(child);
            }
        }
    }
    Ok(out)
}

/// Remove every inheritance edge naming `table` as child or parent — called when the table is dropped
/// so a later same-named table does not inherit stale edges.
pub(super) fn remove_edges_for(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, INHERITANCE_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &INHERITANCE_CATALOG_SCHEMA)?;
        let matches = matches!(row.first(), Some(ast::Value::Text(c)) if c == table)
            || matches!(row.get(1), Some(ast::Value::Text(p)) if p == table);
        if matches {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// Remove a single inheritance edge `child → parent`, leaving every other edge intact — including the
/// child's own edges to its sub-partitions. Used by `DETACH PARTITION`, which unlinks one partition
/// from its parent without disturbing the rest of the hierarchy.
pub(super) fn remove_edge(
    engine: &dyn StorageEngine,
    txn: TxnId,
    child: &str,
    parent: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, INHERITANCE_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &INHERITANCE_CATALOG_SCHEMA)?;
        let is_edge = matches!(row.first(), Some(ast::Value::Text(c)) if c == child)
            && matches!(row.get(1), Some(ast::Value::Text(p)) if p == parent);
        if is_edge {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// Look up the inheritance catalog, creating it lazily if absent.
fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, INHERITANCE_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = ["child", "parent", "seq"]
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: "public".to_owned(),
        name: INHERITANCE_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}
