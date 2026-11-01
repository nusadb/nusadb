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

/// The recorded definition of sequence `name`, or `None` when it has no catalog row (created before
/// this catalog existed) or no statement is executing on this thread to read it under.
fn recorded(
    engine: &dyn StorageEngine,
    name: &str,
) -> Result<Option<nusadb_core::engine::SequenceDef>, Error> {
    let Some(txn) = super::eval::exec_txn() else {
        return Ok(None);
    };
    Ok(list(engine, txn)?.into_iter().find(|d| d.name == name))
}

/// Whether `err` is the engine's report that a non-cycling sequence has handed out its last value.
/// The decision is the engine's: the catalog row is read only to word the message.
fn is_limit_error(err: &nusadb_core::Error) -> bool {
    matches!(
        err,
        nusadb_core::Error::Io(io)
            if io.kind() == std::io::ErrorKind::InvalidInput
                && io.to_string().contains("reached its limit")
    )
}

/// The `2200H` error for sequence `name`, which stopped at `bound` going up (`ascending`) or down.
fn exhausted(name: &str, ascending: bool, bound: i64) -> Error {
    let side = if ascending { "maximum" } else { "minimum" };
    Error::SequenceExhausted(format!(
        "nextval: reached {side} value of sequence \"{name}\" ({bound})"
    ))
}

/// `nextval` on sequence `id`, named `name`. A sequence that has handed out its last value and does
/// not cycle reports `2200H` instead of the storage error the engine raises for it. Whether it ran
/// out is the engine's verdict; the recorded definition only supplies the bound for the message,
/// and only when it agrees with the engine (the next step from the current value leaves its
/// bounds). Otherwise the message names the sequence without a bound.
pub(super) fn next_value(
    engine: &dyn StorageEngine,
    id: nusadb_core::SequenceId,
    name: &str,
) -> Result<i64, Error> {
    match engine.sequence_next(id) {
        Ok(value) => return Ok(value),
        Err(err) if !is_limit_error(&err) => return Err(err.into()),
        Err(_) => {},
    }
    let current = engine.sequence_current(id).ok();
    // The row agrees when the engine's last value lies inside its bounds and one more step would
    // leave them: then the engine stopped at the bound the row names.
    let agreeing = recorded(engine, name)?.filter(|def| {
        !def.cycle
            && current.is_some_and(|current| {
                (def.min_value..=def.max_value).contains(&current)
                    && current
                        .checked_add(def.increment)
                        .is_none_or(|next| next > def.max_value || next < def.min_value)
            })
    });
    Err(match agreeing {
        Some(def) if def.increment < 0 => exhausted(name, false, def.min_value),
        Some(def) => exhausted(name, true, def.max_value),
        None => {
            Error::SequenceExhausted(format!("nextval: reached the limit of sequence \"{name}\""))
        },
    })
}

/// What bounds the values an identity column draws: the column's declared integer range, resolved
/// once per statement, and the sequence's recorded definition (increment, bounds, `CYCLE`), read from
/// the catalog only when a draw reaches the edge of that range and at most once per statement. The
/// definition is the catalog's record, so it lags a rolled-back `ALTER SEQUENCE` the engine kept; a
/// draw the engine then hands out past the type's range is still refused after the fact.
pub(super) struct DrawLimits {
    /// The declared type's `(min, max)` when narrower than 64 bits, else `None`.
    type_range: Option<(i64, i64)>,
    /// The statement's transaction, to read the definition under.
    txn: TxnId,
    /// The recorded definition, once a draw at the edge of the range has needed it.
    def: std::sync::OnceLock<Option<nusadb_core::engine::SequenceDef>>,
}

impl DrawLimits {
    /// The limits of a column of declared range `type_range`, drawn in transaction `txn`. Reads
    /// nothing: the sequence's definition is fetched on demand.
    pub(super) const fn resolve(type_range: Option<(i64, i64)>, txn: TxnId) -> Self {
        Self {
            type_range,
            txn,
            def: std::sync::OnceLock::new(),
        }
    }

    /// The recorded definition of sequence `name`, read from the catalog on first use.
    fn def(
        &self,
        engine: &dyn StorageEngine,
        name: &str,
    ) -> Result<Option<&nusadb_core::engine::SequenceDef>, Error> {
        if self.def.get().is_none() {
            let fetched = list(engine, self.txn)?.into_iter().find(|d| d.name == name);
            // Another draw of the same statement may have filled it meanwhile; either value is the
            // same catalog row.
            let _ = self.def.set(fetched);
        }
        Ok(self.def.get().and_then(Option::as_ref))
    }
}

/// `nextval` for an identity column whose declared integer type is narrower than 64 bits: the
/// sequence's own bounds apply as in [`next_value`], and a value outside the type's range is
/// refused with `2200H` naming the type's bound, as a sequence of the column's type would.
///
/// While the sequence's last value lies strictly inside the type's range, the draw goes straight to
/// the engine: with no catalog read, the common path costs what a plain `nextval` does. Once the
/// last value sits at (or past) an edge of the range, the next draw is judged first from the
/// sequence's recorded definition, so a draw that would leave the range is refused before it is
/// taken and the sequence does not move, unless the sequence cycles at a bound inside the range,
/// where it wraps instead. A draw that still lands outside the range (an increment larger than one
/// that jumps from inside past the edge, a concurrent draw that crossed it, or a definition the
/// catalog no longer matches) is refused after the fact.
pub(super) fn next_value_within(
    engine: &dyn StorageEngine,
    id: nusadb_core::SequenceId,
    name: &str,
    limits: &DrawLimits,
) -> Result<i64, Error> {
    let Some((type_min, type_max)) = limits.type_range else {
        return next_value(engine, id, name);
    };
    if let Ok(current) = engine.sequence_current(id)
        && (current >= type_max || current <= type_min)
        && let Some(def) = limits.def(engine, name)?
    {
        let ascending = def.increment > 0;
        let wraps_inside = def.cycle
            && if ascending {
                def.max_value <= type_max
            } else {
                def.min_value >= type_min
            };
        let leaves = current
            .checked_add(def.increment)
            .is_none_or(|next| next > type_max || next < type_min);
        if leaves && !wraps_inside {
            return Err(exhausted(
                name,
                ascending,
                if ascending { type_max } else { type_min },
            ));
        }
    }
    let value = next_value(engine, id, name)?;
    if value > type_max {
        return Err(exhausted(name, true, type_max));
    }
    if value < type_min {
        return Err(exhausted(name, false, type_min));
    }
    Ok(value)
}

/// Refuse a `setval` target outside the recorded bounds of sequence `name` with `22003`. A sequence
/// with no recorded definition is not checked.
pub(super) fn check_set_value(
    engine: &dyn StorageEngine,
    name: &str,
    value: i64,
) -> Result<(), Error> {
    match recorded(engine, name)? {
        Some(def) if value < def.min_value || value > def.max_value => {
            Err(Error::NumericOutOfRange(format!(
                "setval: value {value} is out of bounds for sequence \"{name}\" ({}..{})",
                def.min_value, def.max_value
            )))
        },
        _ => Ok(()),
    }
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

#[cfg(test)]
mod tests {
    use super::is_limit_error;

    /// Only the engine's own report that a sequence ran out counts as exhaustion; any other
    /// failure (a standby, an I/O fault, a missing sequence) keeps its own error.
    #[test]
    fn only_the_engines_limit_report_is_exhaustion() {
        let limit = nusadb_core::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sequence reached its limit",
        ));
        assert!(is_limit_error(&limit));
        let other_input = nusadb_core::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sequence 7 not found",
        ));
        assert!(!is_limit_error(&other_input));
        let io_fault = nusadb_core::Error::Io(std::io::Error::other("reached its limit"));
        assert!(!is_limit_error(&io_fault));
        assert!(!is_limit_error(&nusadb_core::Error::ReadOnly(
            "this server is a standby".to_owned()
        )));
    }
}
