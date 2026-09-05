//! Sequence definition shadow catalog, backing `information_schema.sequences`.
//!
//! Sequence definitions (start/increment/bounds/cycle) live inside the storage engine, which
//! exposes them only by name — there is no treaty surface to enumerate them. This engine-scoped
//! catalog mirrors each definition at `CREATE`/`ALTER SEQUENCE` time and removes it on `DROP`, the
//! same pattern the trigger/function/partition catalogs use — no storage-spine change. A sequence
//! created before this catalog existed is simply absent from the view until it is next altered
//! (best-effort, documented).
#![allow(clippy::wildcard_imports)]

use super::*;

/// Engine-scoped system catalog of sequence definitions.
pub(super) const SEQUENCE_CATALOG: &str = "nusadb_sequences";

/// Six-text-column schema: `(name, start, increment, min_value, max_value, cycle)` — the numeric
/// fields as decimal text, `cycle` as `"true"`/`"false"`.
const SEQUENCE_CATALOG_SCHEMA: [ColumnType; 6] = [ColumnType::Text; 6];

/// Record (or replace) the definition of `def.name`. Called after the engine-side create succeeds.
pub(super) fn record(
    engine: &dyn StorageEngine,
    txn: TxnId,
    def: &nusadb_core::engine::SequenceDef,
) -> Result<(), Error> {
    remove(engine, txn, &def.name)?;
    let cat = ensure_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(def.name.clone()),
        ast::Value::Text(def.start.to_string()),
        ast::Value::Text(def.increment.to_string()),
        ast::Value::Text(def.min_value.to_string()),
        ast::Value::Text(def.max_value.to_string()),
        ast::Value::Text(def.cycle.to_string()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &SEQUENCE_CATALOG_SCHEMA)?)?;
    Ok(())
}

/// Apply an `ALTER SEQUENCE` change to the recorded definition. A sequence with no recorded row
/// (created before this catalog existed) is skipped — the view stays best-effort rather than
/// inventing bounds the engine never confirmed.
pub(super) fn apply_change(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
    change: &nusadb_core::engine::SequenceChange,
) -> Result<(), Error> {
    let Some(mut def) = list(engine, txn)?.into_iter().find(|d| d.name == name) else {
        return Ok(());
    };
    if let Some(v) = change.start {
        def.start = v;
    }
    if let Some(v) = change.increment {
        def.increment = v;
    }
    if let Some(v) = change.min_value {
        def.min_value = v;
    }
    if let Some(v) = change.max_value {
        def.max_value = v;
    }
    if let Some(v) = change.cycle {
        def.cycle = v;
    }
    record(engine, txn, &def)
}

/// Remove the recorded definition of `name` (a missing row is a no-op).
pub(super) fn remove(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, SEQUENCE_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &SEQUENCE_CATALOG_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == name) {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// Every recorded sequence definition (empty when the catalog does not exist yet).
pub(super) fn list(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<nusadb_core::engine::SequenceDef>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, SEQUENCE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &SEQUENCE_CATALOG_SCHEMA)?;
        let text = |i: usize| match row.get(i) {
            Some(ast::Value::Text(s)) => s.as_str(),
            _ => "",
        };
        out.push(nusadb_core::engine::SequenceDef {
            name: text(0).to_owned(),
            start: text(1).parse().unwrap_or(1),
            increment: text(2).parse().unwrap_or(1),
            min_value: text(3).parse().unwrap_or(1),
            max_value: text(4).parse().unwrap_or(i64::MAX),
            cycle: text(5) == "true",
        });
    }
    Ok(out)
}

/// Look up the sequence catalog, creating it lazily if absent.
fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, SEQUENCE_CATALOG)? {
        return Ok(schema.id);
    }
    let columns = [
        "name",
        "start",
        "increment",
        "min_value",
        "max_value",
        "cycle",
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
        name: SEQUENCE_CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}
