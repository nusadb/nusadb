//! Index entries of `CHAR(n)` values.
//!
//! Every index entry this build writes holds a `CHAR(n)` value without its trailing blanks, the
//! form its row is stored in, so one exact probe finds a key. Earlier releases wrote the value as
//! the statement gave it, so an index they built may hold `'x  '` for a row that reads back `'x'`.
//! Those indexes are not rebuilt behind the user's back: they stay correct, through a slower probe,
//! until `REINDEX` rebuilds them.
//!
//! # Which indexes predate this release
//!
//! The first time this build opens a database ([`note_legacy_char_indexes`], and again at the start
//! of every session as a safety net, see [`ensure_snapshot`]), it records, from the catalogs alone,
//! the id of every index whose key or predicate reads a `CHAR(n)` column. That is the legacy set,
//! kept in [`CANON_CATALOG`] with a marker row written in the same transaction. Every index created
//! afterwards (by `CREATE INDEX`, a key constraint, a foreign key, an exclusion constraint, a
//! partition copy, `TRUNCATE`, `ALTER ... TYPE` or `LIKE`) gets a fresh id, so it is canonical
//! without any record of its own, and a rollback has nothing to undo. Until the snapshot exists
//! (a database no session of this build has opened), every index reading a `CHAR(n)` column is
//! treated as legacy.
//!
//! # What a legacy index costs
//!
//! A key probe of a legacy index tries each spelling a `CHAR(n)` value may have been written in
//! (`super::global_unique::probe_ranges`); a unique index keyed on an expression or with a
//! predicate over a `CHAR(n)` column is matched against the table's keys instead, read once per
//! statement (`super::global_unique::legacy_key_held`). A query reads through a legacy column
//! index the same way, probing each spelling of an equality and widening a range to past every
//! spelling of its upper end (at most 256 spellings or blanks per value, however wide the column),
//! streaming the entries and rechecking each row; an ordered read with a
//! `LIMIT` keeps only its first rows in key order (`super::scan`). An expression index over such a
//! column is not offered to the planner.
//! `REINDEX` ([`run_reindex`]) rebuilds such an index from its rows and takes it out of the
//! legacy set.
#![allow(clippy::wildcard_imports)]

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};

use nusadb_core::{IndexId, TableId};

use super::*;

/// Engine-scoped catalog of the legacy set: `(kind, value)` text rows. The marker row is
/// `("\u{0}snapshot", "1")`; a legacy index is `("legacy", <index id>)`.
pub(super) const CANON_CATALOG: &str = "nusadb_char_index_canon";

const CANON_SCHEMA: [ColumnType; 2] = [ColumnType::Text; 2];

const MARKER: (&str, &str) = ("\u{0}snapshot", "1");

const LEGACY: &str = "legacy";

/// The catalog's table id and its `(kind, value)` rows, each with its tid.
type CatalogRows = (TableId, Vec<(Tid, String, String)>);

/// The catalog rows, or `None` when the catalog does not exist.
fn catalog_rows(engine: &dyn StorageEngine, txn: TxnId) -> Result<Option<CatalogRows>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, CANON_CATALOG)? else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &CANON_SCHEMA)?;
        if let (Some(ast::Value::Text(kind)), Some(ast::Value::Text(value))) =
            (row.first(), row.get(1))
        {
            out.push((tid, kind.clone(), value.clone()));
        }
    }
    Ok(Some((cat.id, out)))
}

fn insert_row(
    engine: &dyn StorageEngine,
    txn: TxnId,
    cat: TableId,
    kind: &str,
    value: &str,
) -> Result<(), Error> {
    let row = [
        ast::Value::Text(kind.to_owned()),
        ast::Value::Text(value.to_owned()),
    ];
    engine.insert(txn, cat, &row::encode(&row, &CANON_SCHEMA)?)?;
    Ok(())
}

/// Whether index `def` of `table` reads a `CHAR(n)` column: in its key (a column or an expression)
/// or in its predicate. Only such an index can hold an entry written with trailing blanks.
pub(super) fn reads_char(def: &nusadb_core::IndexDef, table: &TableSchema) -> bool {
    let is_char = |ordinal: usize| {
        table
            .columns
            .get(ordinal)
            .is_some_and(|c| matches!(c.ty, ColumnType::Char(_)))
    };
    let named_char = |name: &String| {
        table
            .columns
            .iter()
            .any(|c| &c.name == name && matches!(c.ty, ColumnType::Char(_)))
    };
    if def.columns.iter().any(named_char) {
        return true;
    }
    let mut referenced = Vec::new();
    for sql in &def.key_exprs {
        match crate::analyzer::analyze_index_key_expr(sql, table, &dml::EmptyCatalog) {
            Ok(typed) => crate::planner::collect_columns(&typed, &mut referenced),
            // An expression that no longer analyzes: assume the worst.
            Err(_) => return true,
        }
    }
    if let Some(sql) = super::unique_index::split_predicate(def).0 {
        match crate::analyzer::analyze_check_predicate(sql, table, &dml::EmptyCatalog) {
            Ok(typed) => crate::planner::collect_columns(&typed, &mut referenced),
            Err(_) => return true,
        }
    }
    referenced.into_iter().any(is_char)
}

/// Whether a predicate (typed against `table`) reads a `CHAR(n)` column.
pub(super) fn typed_reads_char(predicate: &crate::planner::TypedExpr, table: &TableSchema) -> bool {
    let mut referenced = Vec::new();
    crate::planner::collect_columns(predicate, &mut referenced);
    referenced.into_iter().any(|o| {
        table
            .columns
            .get(o)
            .is_some_and(|c| matches!(c.ty, ColumnType::Char(_)))
    })
}

/// Record the legacy set of `engine` if it is not recorded yet (see the module docs): one pass
/// over the catalogs, no row is read. Returns how many legacy indexes there are now.
///
/// # Errors
/// Storage errors; the transaction is rolled back and nothing is recorded.
pub fn note_legacy_char_indexes(engine: &dyn StorageEngine) -> Result<usize, Error> {
    let txn = engine.begin(IsolationLevel::default())?;
    match snapshot_in(engine, txn) {
        Ok(count) => {
            super::deferred::commit(engine, txn)?;
            forget_state(engine);
            Ok(count)
        },
        Err(e) => {
            let _ = super::deferred::rollback(engine, txn);
            Err(e)
        },
    }
}

/// The ids of every index `txn` sees (a catalog walk, no row read).
fn index_ids(engine: &dyn StorageEngine, txn: TxnId) -> Result<HashSet<IndexId>, Error> {
    let mut out = HashSet::new();
    for (schema, name) in engine.list_tables_qualified_as_of(txn)? {
        if let Some(table) = engine.lookup_table_as_of_in(txn, &schema, &name)? {
            for def in engine.list_indexes_as_of(txn, table.id)? {
                if let Some(id) = engine.lookup_index(&def.name)? {
                    out.insert(id);
                }
            }
        }
    }
    Ok(out)
}

/// Take the snapshot if no session of this build has opened `engine` yet. Cheap once it exists (a
/// catalog lookup). A failure (a read-only standby, say) leaves every index reading a `CHAR(n)`
/// column on the legacy path, which is slower but correct.
pub(super) fn ensure_snapshot(engine: &dyn StorageEngine) {
    if matches!(engine.lookup_table(CANON_CATALOG), Ok(Some(_))) {
        return;
    }
    if let Err(e) = note_legacy_char_indexes(engine) {
        tracing::debug!(error = %e, "could not record the indexes that predate CHAR canonical keys");
    }
}

/// The body of [`note_legacy_char_indexes`], inside `txn`.
fn snapshot_in(engine: &dyn StorageEngine, txn: TxnId) -> Result<usize, Error> {
    if let Some((_, rows)) = catalog_rows(engine, txn)?
        && rows
            .iter()
            .any(|(_, k, v)| (k.as_str(), v.as_str()) == MARKER)
    {
        // Only the ones still there (an index dropped since leaves its row behind).
        let live = index_ids(engine, txn)?;
        return Ok(rows
            .iter()
            .filter(|(_, k, v)| {
                k == LEGACY && v.parse::<u64>().is_ok_and(|id| live.contains(&IndexId(id)))
            })
            .count());
    }
    let mut legacy = Vec::new();
    for (schema, name) in engine.list_tables_qualified_as_of(txn)? {
        let Some(table) = engine.lookup_table_as_of_in(txn, &schema, &name)? else {
            continue;
        };
        if !table
            .columns
            .iter()
            .any(|c| matches!(c.ty, ColumnType::Char(_)))
        {
            continue;
        }
        for def in engine.list_indexes_as_of(txn, table.id)? {
            if reads_char(&def, &table)
                && let Some(id) = engine.lookup_index(&def.name)?
            {
                legacy.push(id);
            }
        }
    }
    let cat = if let Some(found) = engine.lookup_table_as_of(txn, CANON_CATALOG)? {
        found.id
    } else {
        {
            let columns = ["kind", "value"]
                .into_iter()
                .map(|name| ColumnDef {
                    name: name.to_owned(),
                    ty: ColumnType::Text,
                    nullable: false,
                })
                .collect();
            engine.create_table(
                txn,
                &TableDef {
                    schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
                    name: CANON_CATALOG.to_owned(),
                    columns,
                },
            )?
        }
    };
    for id in &legacy {
        insert_row(engine, txn, cat, LEGACY, &id.0.to_string())?;
    }
    insert_row(engine, txn, cat, MARKER.0, MARKER.1)?;
    Ok(legacy.len())
}

// === What a probe needs to know ==========================================================

/// Which indexes of an engine predate this release.
#[derive(Debug, Default)]
pub(super) struct CanonState {
    /// Whether the snapshot exists; without it every index reading a `CHAR(n)` column is legacy.
    taken: bool,
    legacy: HashSet<IndexId>,
}

impl CanonState {
    /// Whether `index` may hold entries written with trailing blanks (it predates this release
    /// and has not been rebuilt). Only meaningful for an index that reads a `CHAR(n)` column.
    pub(super) fn is_legacy(&self, index: IndexId) -> bool {
        !self.taken || self.legacy.contains(&index)
    }
}

/// The state, per engine address and epoch.
static STATE: Mutex<Option<(usize, u64, Arc<CanonState>)>> = Mutex::new(None);

fn engine_addr(engine: &dyn StorageEngine) -> usize {
    std::ptr::from_ref(engine).cast::<()>().addr()
}

fn forget_state(engine: &dyn StorageEngine) {
    let mut slot = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if slot
        .as_ref()
        .is_some_and(|(a, _, _)| *a == engine_addr(engine))
    {
        *slot = None;
    }
}

/// The legacy set of `engine`, as `txn` reads the catalog, cached until `epoch` moves.
///
/// # Errors
/// Storage errors.
pub(super) fn state(
    engine: &dyn StorageEngine,
    txn: TxnId,
    epoch: u64,
) -> Result<Arc<CanonState>, Error> {
    let addr = engine_addr(engine);
    if let Some((a, e, s)) = STATE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        && *a == addr
        && *e == epoch
    {
        return Ok(Arc::clone(s));
    }
    let mut out = CanonState::default();
    if let Some((_, rows)) = catalog_rows(engine, txn)? {
        for (_, kind, value) in rows {
            if (kind.as_str(), value.as_str()) == MARKER {
                out.taken = true;
            } else if kind == LEGACY
                && let Ok(id) = value.parse::<u64>()
            {
                out.legacy.insert(IndexId(id));
            }
        }
    }
    let out = Arc::new(out);
    *STATE.lock().unwrap_or_else(PoisonError::into_inner) = Some((addr, epoch, Arc::clone(&out)));
    Ok(out)
}

/// Whether index `id` (`def`, on `table`) needs the legacy probe: it reads a `CHAR(n)` column and
/// predates this release.
///
/// # Errors
/// Storage errors reading the catalog.
pub(super) fn needs_legacy_probe(
    engine: &dyn StorageEngine,
    txn: TxnId,
    id: IndexId,
    def: &nusadb_core::IndexDef,
    table: &TableSchema,
) -> Result<bool, Error> {
    if !reads_char(def, table) {
        return Ok(false);
    }
    Ok(super::global_unique::canon_state(engine, txn)?.is_legacy(id))
}

// === REINDEX =============================================================================

/// `REINDEX { INDEX | TABLE | SCHEMA | DATABASE | SYSTEM } name`: rebuild every legacy index
/// among the target's from its rows, so its entries are canonical, and take it out of the legacy
/// set. An index that is already canonical, or reads no `CHAR(n)` column, is left alone (there is
/// nothing it could hold out of form). A rebuilt index keeps its name, key, predicate,
/// `NULLS [NOT] DISTINCT`, uniqueness and what refers to it by name (comments, the deferral of its
/// constraint, an exclusion constraint owning it); a key constraint's referencing foreign keys are
/// dropped and added back around it. Rows are streamed, never held whole. The statement is one
/// transaction: a failure leaves every index as it was.
///
/// # Errors
/// `23505` naming the key when a unique index's rows hold one key twice once trailing blanks are
/// ignored (data an earlier release accepted); `42704` for an unknown target; storage errors.
pub(super) fn run_reindex(
    target: &ast::ReindexTarget,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let mut tables = reindex_tables(target, engine, txn)?;
    // `REINDEX INDEX` / `REINDEX TABLE` need the table's owner (or a superuser), as `CREATE INDEX`
    // does; `REINDEX SCHEMA` / `REINDEX DATABASE` rebuild the tables the role owns and skip the
    // others.
    match target {
        ast::ReindexTarget::Index { .. } | ast::ReindexTarget::Table { .. } => {
            for (table, _) in &tables {
                if !owns(engine, txn, table)? {
                    return Err(Error::PermissionDenied(format!(
                        "must be the owner of table `{}.{}` to reindex it",
                        table.schema, table.name
                    )));
                }
            }
        },
        ast::ReindexTarget::Schema(_) | ast::ReindexTarget::Database(_) => {
            let mut owned = Vec::with_capacity(tables.len());
            for entry in tables {
                if owns(engine, txn, &entry.0)? {
                    owned.push(entry);
                }
            }
            tables = owned;
        },
    }
    // A database no session of this build has opened yet: record what predates it first.
    if !super::global_unique::canon_state(engine, txn)?.taken {
        snapshot_in(engine, txn)?;
        forget_state(engine);
        super::global_unique::invalidate_canon();
    }
    let state = super::global_unique::canon_state(engine, txn)?;
    let mut rebuilt: Vec<IndexId> = Vec::new();
    let mut fks_done: HashSet<String> = HashSet::new();
    for (table, index_name) in tables {
        let wanted = |name: &str| index_name.as_deref().is_none_or(|n| n == name);
        for def in engine.list_indexes_as_of(txn, table.id)? {
            if !wanted(&def.name) {
                continue;
            }
            let Some(id) = engine.lookup_index(&def.name)? else {
                continue;
            };
            if !reads_char(&def, &table) || !state.is_legacy(id) {
                continue;
            }
            // A foreign key rebuilt with its referenced key already has a new child index.
            if fks_done.contains(&def.name) {
                continue;
            }
            rebuilt.push(id);
            rebuild_index(engine, txn, &table, &def, id, &mut fks_done, &mut rebuilt)?;
        }
    }
    forget_rebuilt(engine, txn, &rebuilt)?;
    Ok(ExecutionResult::Reindexed)
}

/// `REINDEX DATABASE` / `REINDEX SCHEMA` outside a transaction block, table by table.
///
/// Each table is rebuilt in a transaction of its own, committed before the next one starts, so
/// what one rebuild holds in memory is bounded by one table. A failure stops at that table (its
/// transaction rolls back); the tables before it stay rebuilt, and running the statement again
/// carries on. Only the tables the role owns are rebuilt (all of them for a superuser).
///
/// # Errors
/// `23505` naming the key when a unique index or constraint holds one key twice once trailing
/// blanks are ignored; `42704` for an unknown schema; storage errors.
#[allow(
    clippy::implicit_hasher,
    reason = "the wire server's per-connection settings are a default-hasher HashMap"
)]
pub fn reindex_each_table(
    engine: &dyn StorageEngine,
    target: &ast::ReindexTarget,
    session: Option<(&str, &std::collections::HashMap<String, String>)>,
) -> Result<ExecutionResult, Error> {
    /// How many times one table is tried when a concurrent writer holds it (`40001`).
    const ATTEMPTS: u32 = 5;
    // The connection's role and database, for the ownership check and the database name.
    if let Some((user, settings)) = session {
        session_ctx::set_session_user_with_settings(user, settings);
    }
    let txn = engine.begin(IsolationLevel::default())?;
    let listed = reindex_tables(target, engine, txn).and_then(|tables| {
        let mut owned = Vec::with_capacity(tables.len());
        for (table, _) in tables {
            if owns(engine, txn, &table)? {
                owned.push(table);
            }
        }
        Ok(owned)
    });
    let _ = super::deferred::rollback(engine, txn);
    for table in listed? {
        let one = ast::ReindexTarget::Table {
            schema: Some(table.schema.clone()),
            name: table.name.clone(),
        };
        let mut attempt = 0;
        loop {
            attempt += 1;
            let txn = engine.begin(IsolationLevel::default())?;
            let outcome = super::run_statement_atomically(engine, txn, || {
                super::global_unique::statement(engine, txn, || run_reindex(&one, engine, txn))
            });
            match outcome {
                Ok(_) => {
                    super::deferred::commit(engine, txn)?;
                    break;
                },
                Err(e) => {
                    let _ = super::deferred::rollback(engine, txn);
                    let busy = matches!(
                        e,
                        Error::Core(nusadb_core::Error::SerializationConflict { .. })
                    );
                    if !busy || attempt >= ATTEMPTS {
                        return Err(e);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50 * u64::from(attempt)));
                },
            }
        }
    }
    Ok(ExecutionResult::Reindexed)
}

/// The tables a `REINDEX` target covers, each with the one index it names (for `REINDEX INDEX`).
fn reindex_tables(
    target: &ast::ReindexTarget,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(TableSchema, Option<String>)>, Error> {
    let search_path = || {
        let mut schemas = Vec::new();
        if let Some(temp) = session_ctx::current_temp_schema() {
            schemas.push(temp);
        }
        schemas.extend(crate::search_path_schemas(
            session_ctx::setting("search_path").as_deref(),
        ));
        schemas
    };
    let user_tables = |schema: Option<&str>| -> Result<Vec<TableSchema>, Error> {
        let mut out = Vec::new();
        for (s, name) in engine.list_tables_qualified_as_of(txn)? {
            if name.starts_with(crate::SYSTEM_TABLE_PREFIX) || schema.is_some_and(|w| w != s) {
                continue;
            }
            if let Some(t) = engine.lookup_table_as_of_in(txn, &s, &name)? {
                out.push(t);
            }
        }
        Ok(out)
    };
    match target {
        ast::ReindexTarget::Index { schema, name } => {
            let candidates: Vec<String> = schema.as_ref().map_or_else(
                || {
                    let mut c = vec![name.clone()];
                    c.extend(
                        search_path()
                            .iter()
                            .map(|s| crate::analyzer::qualified_display(s, name)),
                    );
                    c
                },
                |s| vec![crate::analyzer::qualified_display(s, name)],
            );
            for table in user_tables(schema.as_deref())? {
                for def in engine.list_indexes_as_of(txn, table.id)? {
                    if candidates.contains(&def.name) {
                        return Ok(vec![(table, Some(def.name))]);
                    }
                }
            }
            Err(Error::ObjectNotFound(format!(
                "index \"{name}\" does not exist"
            )))
        },
        ast::ReindexTarget::Table { schema, name } => {
            let schemas = schema
                .as_ref()
                .map_or_else(search_path, |s| vec![s.clone()]);
            for s in schemas {
                if let Some(table) = engine.lookup_table_as_of_in(txn, &s, name)? {
                    // A partitioned table's rows live in its partitions: they are rebuilt too.
                    let mut out = vec![(table.clone(), None)];
                    let tree = super::partition::committed_tree(engine, txn)?;
                    let key = crate::analyzer::qualified_display(&table.schema, &table.name);
                    for below in tree.tables_below(&key) {
                        let (bs, bn) = crate::analyzer::split_qualified(&below);
                        if let Some(t) = engine.lookup_table_as_of_in(
                            txn,
                            bs.unwrap_or(nusadb_core::PUBLIC_SCHEMA),
                            bn,
                        )? {
                            out.push((t, None));
                        }
                    }
                    return Ok(out);
                }
            }
            Err(Error::TableNotFound { name: name.clone() })
        },
        ast::ReindexTarget::Schema(schema) => {
            if schema != nusadb_core::PUBLIC_SCHEMA && engine.lookup_schema(schema)?.is_none() {
                return Err(Error::ObjectNotFound(format!(
                    "schema \"{schema}\" does not exist"
                )));
            }
            Ok(user_tables(Some(schema))?
                .into_iter()
                .map(|t| (t, None))
                .collect())
        },
        ast::ReindexTarget::Database(name) => {
            if let Some(name) = name {
                let current = session_ctx::current_database();
                if *name != current {
                    return Err(Error::Coded {
                        message: format!(
                            "can only reindex the currently open database \"{current}\", not \
                             \"{name}\""
                        ),
                        sqlstate: "0A000",
                    });
                }
            }
            Ok(user_tables(None)?.into_iter().map(|t| (t, None)).collect())
        },
    }
}

/// Rebuild one legacy index `def` (id `id`) of `table`.
fn rebuild_index(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    def: &nusadb_core::IndexDef,
    id: IndexId,
    fks_done: &mut HashSet<String>,
    replaced: &mut Vec<IndexId>,
) -> Result<(), Error> {
    let constraint = engine
        .list_constraints_as_of(txn, table.id)?
        .into_iter()
        .find(|c| {
            c.index == Some(id)
                && matches!(
                    c.kind,
                    nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
                )
        });
    if let Some(c) = constraint {
        // A key constraint: the foreign keys that reference it go first, and come back after it.
        let referencing: Vec<nusadb_core::ForeignKeyDef> = engine
            .list_foreign_keys_as_of(txn, table.id)?
            .into_iter()
            .filter(|fk| {
                if fk.parent_table != table.id {
                    return false;
                }
                let mut referenced = fk.parent_columns.clone();
                let mut key = c.columns.clone();
                referenced.sort();
                key.sort();
                referenced == key
            })
            .collect();
        for fk in &referencing {
            // Its child index is rebuilt too.
            if let Some(child_index) = engine.lookup_index(&fk.name)? {
                replaced.push(child_index);
            }
            engine.drop_constraint(txn, fk.child_table, &fk.name)?;
        }
        engine.drop_constraint(txn, table.id, &c.name)?;
        let new = engine.add_unique_constraint(
            txn,
            table.id,
            &c.name,
            &c.columns,
            c.kind == nusadb_core::ConstraintKind::PrimaryKey,
            c.nulls_not_distinct,
        )?;
        backfill(engine, txn, table, new, &c.name)?;
        verify_unique(
            engine,
            txn,
            table,
            new,
            &c.columns,
            c.nulls_not_distinct,
            &c.name,
        )?;
        for fk in referencing {
            readd_foreign_key(engine, txn, &fk, fks_done)?;
        }
        return Ok(());
    }
    if let Some(fk) = engine
        .list_foreign_keys_as_of(txn, table.id)?
        .into_iter()
        .find(|fk| {
            let own = fk.child_table == table.id;
            own && fk.name == def.name
        })
    {
        // A foreign key's child-side index.
        engine.drop_constraint(txn, table.id, &fk.name)?;
        return readd_foreign_key(engine, txn, &fk, fks_done);
    }
    // Any other index: a plain, expression or partial one, a unique index (whose rebuild refuses a
    // duplicate through the engine's own check), or an exclusion constraint's own.
    engine.drop_index(txn, id)?;
    let new = engine.create_index(txn, def)?;
    backfill(engine, txn, table, new, &def.name).map_err(|e| match e {
        Error::Core(nusadb_core::Error::ConstraintViolation(message)) => Error::Coded {
            message: format!(
                "could not rebuild unique index \"{}\": {message} (its rows hold one key twice \
                 once trailing blanks of a CHAR value are ignored; remove the duplicate first)",
                def.name
            ),
            sqlstate: "23505",
        },
        other => other,
    })
}

/// Add back a foreign key dropped for the rebuild, and fill its new child index.
fn readd_foreign_key(
    engine: &dyn StorageEngine,
    txn: TxnId,
    fk: &nusadb_core::ForeignKeyDef,
    fks_done: &mut HashSet<String>,
) -> Result<(), Error> {
    let id = engine.add_foreign_key(txn, fk)?;
    if let Some(child) = engine.lookup_table_by_id_as_of(txn, fk.child_table)? {
        backfill(engine, txn, &child, id, &fk.name)?;
    }
    fks_done.insert(fk.name.clone());
    Ok(())
}

/// Fill the new index `id` (named `name`) of `table` from its stored rows, streamed.
fn backfill(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    id: IndexId,
    name: &str,
) -> Result<(), Error> {
    let Some(def) = engine
        .list_indexes_as_of(txn, table.id)?
        .into_iter()
        .find(|d| d.name == name)
    else {
        return Ok(());
    };
    if let Some(target) = dml::build_index_target(id, table, &def) {
        dml::backfill_index(&target, table, engine, txn)?;
    }
    Ok(())
}

/// After rebuilding key constraint `name`'s index `id`: no two live rows may share a key.
///
/// The table is streamed in its own order, the order the backfill read it in, so its pages are
/// read once each. Each key is remembered by a 64-bit hash only; a hash seen before is checked
/// against the rebuilt index (whose entries are now canonical, so one exact probe finds every
/// row holding the key). The hashes are bounded by the maintenance-memory budget (16 bytes a key);
/// a table with more keys than that is checked by [`verify_unique_walk`] instead. The engine's
/// catalog is held only for the length of one engine call, never across the table. The key value
/// goes into the error only for a role that may read the table.
fn verify_unique(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    id: IndexId,
    columns: &[String],
    nulls_not_distinct: bool,
    name: &str,
) -> Result<(), Error> {
    use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
    let ordinals = dml::constraint_ordinals(table, columns)?;
    let types = column_types(table);
    let cap = super::ops::maintenance_work_mem() / 16;
    #[cfg(test)]
    let cap = VERIFY_CAP.with(std::cell::Cell::get).unwrap_or(cap);
    let hasher = BuildHasherDefault::<DefaultHasher>::default();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut scan = engine.scan(txn, table.id)?;
    while let Some((_, tuple)) = scan.try_next()? {
        crate::cancel::check()?;
        let row = row::decode(&tuple, &types)?;
        let Some(key) = dml::unique_key(&row, &ordinals, nulls_not_distinct) else {
            continue;
        };
        let encoded = index_key::encode_index_key(&key)?;
        let hash = hasher.hash_one(&encoded);
        // A test narrows the hash so that distinct keys collide.
        #[cfg(test)]
        let hash = hash & VERIFY_HASH_MASK.with(std::cell::Cell::get);
        if seen.insert(hash) {
            if seen.len() > cap {
                drop(scan);
                return verify_unique_walk(
                    engine,
                    txn,
                    table,
                    id,
                    columns,
                    nulls_not_distinct,
                    name,
                );
            }
            continue;
        }
        // The hash was seen before: the key itself, or another key with the same hash.
        let mut probe = engine.index_scan_committed(
            txn,
            id,
            std::ops::Bound::Included(encoded.clone()),
            std::ops::Bound::Included(encoded),
        )?;
        let mut holders = 0usize;
        while let Some((_, held)) = probe.try_next()? {
            let held = row::decode(&held, &types)?;
            if dml::unique_key(&held, &ordinals, nulls_not_distinct)
                .is_some_and(|k| dml::unique_key_cmp(&k, &key) == std::cmp::Ordering::Equal)
            {
                holders += 1;
            }
        }
        if holders > 1 {
            return Err(duplicate_key_error(
                engine, txn, table, columns, &key, name,
            )?);
        }
    }
    Ok(())
}

/// [`verify_unique`] for a table with more keys than the maintenance-memory budget remembers.
///
/// The rebuilt index holds one canonical entry per row, so equal keys are adjacent in it. It is
/// read in key order a bounded chunk at a time ([`VERIFY_CHUNK`] rows per engine call, the next
/// chunk starting at the last key seen), each row's key compared with the one before it. What the
/// pass holds at once is one chunk. It reads the rows in key order, so it is slower than the
/// streamed pass when the table does not fit in memory.
fn verify_unique_walk(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    id: IndexId,
    columns: &[String],
    nulls_not_distinct: bool,
    name: &str,
) -> Result<(), Error> {
    use std::ops::Bound;
    let ordinals = dml::constraint_ordinals(table, columns)?;
    let types = column_types(table);
    // Where the walk stands: the encoded key of the last row read, that row, and its key when it
    // takes part in the constraint (a key with a `NULL` under `NULLS DISTINCT` does not, and the
    // next chunk starts past every row holding it).
    let mut at: Option<(Vec<u8>, Tid, Option<Vec<ast::Value>>)> = None;
    loop {
        let lo = match &at {
            None => Bound::Unbounded,
            Some((encoded, _, Some(_))) => Bound::Included(encoded.clone()),
            Some((encoded, _, None)) => Bound::Excluded(encoded.clone()),
        };
        let mut scan = engine.index_scan_directed_limited(
            txn,
            id,
            lo,
            Bound::Unbounded,
            nusadb_core::engine::ScanDirection::Forward,
            Some(VERIFY_CHUNK),
        )?;
        let mut read = 0usize;
        while let Some((tid, tuple)) = scan.try_next()? {
            crate::cancel::check()?;
            read += 1;
            let row = row::decode(&tuple, &types)?;
            let key = dml::unique_key(&row, &ordinals, nulls_not_distinct);
            if let (Some(k), Some((_, last_tid, Some(last)))) = (&key, &at)
                && dml::unique_key_cmp(last, k) == std::cmp::Ordering::Equal
            {
                if *last_tid == tid {
                    // The row the previous chunk ended on, read again.
                    continue;
                }
                return Err(duplicate_key_error(engine, txn, table, columns, k, name)?);
            }
            let values: Vec<ast::Value> = ordinals
                .iter()
                .map(|&o| row.get(o).cloned().unwrap_or(ast::Value::Null))
                .collect();
            at = Some((index_key::encode_index_key(&values)?, tid, key));
        }
        if read < VERIFY_CHUNK {
            return Ok(());
        }
    }
}

#[cfg(test)]
thread_local! {
    /// A smaller key budget for [`verify_unique`], so a test reaches [`verify_unique_walk`].
    static VERIFY_CAP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// A mask over [`verify_unique`]'s key hashes, so a test can make distinct keys collide.
    static VERIFY_HASH_MASK: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
}

/// How many rows [`verify_unique_walk`] reads per engine call.
const VERIFY_CHUNK: usize = 1024;

/// The `23505` a rebuild of key constraint `name` ends with when `key` is held twice.
fn duplicate_key_error(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    columns: &[String],
    key: &[ast::Value],
    name: &str,
) -> Result<Error, Error> {
    let shown = if may_read(engine, txn, table)? {
        let values: Vec<String> = key.iter().map(crate::display::value_text).collect();
        format!("key ({})=({})", columns.join(", "), values.join(", "))
    } else {
        format!("a key of ({})", columns.join(", "))
    };
    Ok(Error::Coded {
        message: format!(
            "could not rebuild the index of constraint \"{name}\": {shown} is held by more than \
             one row once trailing blanks of a CHAR value are ignored; remove the duplicate first"
        ),
        sqlstate: "23505",
    })
}

/// Whether the session may read `table` (so an error may show its values).
fn may_read(engine: &dyn StorageEngine, txn: TxnId, table: &TableSchema) -> Result<bool, Error> {
    crate::rbac::has_privilege(
        engine,
        txn,
        &session_ctx::current_user(),
        crate::ast::ObjectKind::Table,
        &format!("{}.{}", table.schema, table.name),
        crate::ast::Privilege::Select,
    )
}

/// Whether the session owns `table` (or is a superuser): the right `REINDEX` needs, as `CREATE
/// INDEX` does.
fn owns(engine: &dyn StorageEngine, txn: TxnId, table: &TableSchema) -> Result<bool, Error> {
    let user = session_ctx::current_user();
    let session = crate::rbac::ResolvedSession::resolve(engine, txn, &user)?;
    if session.is_superuser() {
        return Ok(true);
    }
    session.owns_object(
        engine,
        txn,
        crate::ast::ObjectKind::Table,
        &format!("{}.{}", table.schema, table.name),
    )
}

/// Take the rebuilt indexes out of the legacy set.
fn forget_rebuilt(
    engine: &dyn StorageEngine,
    txn: TxnId,
    rebuilt: &[IndexId],
) -> Result<(), Error> {
    if rebuilt.is_empty() {
        return Ok(());
    }
    let Some((cat, rows)) = catalog_rows(engine, txn)? else {
        return Ok(());
    };
    let gone: HashSet<String> = rebuilt.iter().map(|id| id.0.to_string()).collect();
    for (tid, kind, value) in rows {
        if kind == LEGACY && gone.contains(&value) {
            engine.delete(txn, cat, tid)?;
        }
    }
    super::global_unique::invalidate_canon();
    forget_state(engine);
    Ok(())
}

/// Forget the snapshot, so every index reading a `CHAR(n)` column counts as built by an earlier
/// release (tests that write entries the way such a release did).
#[cfg(test)]
pub(super) fn reset_to_legacy(engine: &dyn StorageEngine) {
    let txn = engine.begin(IsolationLevel::default()).expect("begin");
    if let Some((cat, rows)) = catalog_rows(engine, txn).expect("catalog") {
        for (tid, _, _) in rows {
            engine.delete(txn, cat, tid).expect("delete");
        }
    }
    super::deferred::commit(engine, txn).expect("commit");
    forget_state(engine);
    super::global_unique::invalidate_canon();
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::too_many_lines,
        reason = "unit test asserts via unwrap/panic"
    )]

    use super::*;
    use crate::{Catalog, IndexInfo, Session, analyze, parse, plan};
    use nusadb_btree::BtreeEngine;

    struct Cat<'a>(&'a dyn StorageEngine, Option<TxnId>);
    impl Catalog for Cat<'_> {
        fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
            self.1
                .map_or_else(
                    || self.0.lookup_table(name),
                    |txn| self.0.lookup_table_as_of(txn, name),
                )
                .map_err(Into::into)
        }
        fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, Error> {
            let txn = match self.1 {
                Some(txn) => txn,
                None => self.0.begin(IsolationLevel::default())?,
            };
            let out = crate::catalog_list_indexes(self.0, txn, name);
            if self.1.is_none() {
                let _ = super::super::deferred::rollback(self.0, txn);
            }
            out
        }
    }

    fn run(
        engine: &dyn StorageEngine,
        s: &mut Session,
        sql: &str,
    ) -> Result<ExecutionResult, Error> {
        let logical = analyze(parse(sql)?, &Cat(engine, s.current_txn()))?;
        s.execute(plan(logical))
    }

    fn ok(engine: &dyn StorageEngine, sql: &str) {
        let mut s = Session::new(engine);
        run(engine, &mut s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }

    fn fails(engine: &dyn StorageEngine, sql: &str, wanted: &str) -> String {
        let mut s = Session::new(engine);
        let e = run(engine, &mut s, sql).expect_err(sql).to_string();
        assert!(e.contains(wanted), "{sql}: {e}");
        e
    }

    fn refused(engine: &dyn StorageEngine, sql: &str) {
        fails(engine, sql, "duplicate key");
    }

    fn count(engine: &dyn StorageEngine, sql: &str) -> i64 {
        let mut s = Session::new(engine);
        match run(engine, &mut s, sql).unwrap() {
            ExecutionResult::Rows { rows, .. } => match rows.first().and_then(|r| r.first()) {
                Some(ast::Value::Int(n)) => *n,
                other => panic!("{sql}: {other:?}"),
            },
            other => panic!("{sql}: {other:?}"),
        }
    }

    fn text(s: &str) -> ast::Value {
        ast::Value::Text(s.to_owned())
    }

    fn int(n: i64) -> ast::Value {
        ast::Value::Int(n)
    }

    /// Write `values` into `table` the way an earlier release did: the row through the row encoding
    /// (which drops a CHAR value's trailing blanks), every index entry from the values as given.
    fn write_old(engine: &dyn StorageEngine, table: &str, values: &[Vec<ast::Value>]) {
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        let t = engine.lookup_table_as_of(txn, table).unwrap().unwrap();
        let targets = dml::secondary_index_targets(&t, engine).unwrap();
        for row in values {
            let tid = engine
                .insert(txn, t.id, &row::encode(row, &column_types(&t)).unwrap())
                .unwrap();
            dml::insert_into_indexes_unchecked(&targets, row, tid, engine, txn).unwrap();
        }
        super::super::deferred::commit(engine, txn).unwrap();
    }

    const SCHEMA: [&str; 9] = [
        "CREATE TABLE ou (k CHAR(4) UNIQUE, n INT)",
        "CREATE TABLE opk (k CHAR(4) PRIMARY KEY, n INT)",
        "CREATE TABLE ch (r CHAR(4) REFERENCES opk (k), n INT)",
        "CREATE TABLE ix (k CHAR(4), n INT)",
        "CREATE INDEX ix_k ON ix (k)",
        "CREATE TABLE ui (k CHAR(4), m INT, d INT)",
        "CREATE UNIQUE INDEX ui_k ON ui (k, m) NULLS NOT DISTINCT",
        "CREATE UNIQUE INDEX ui_part ON ui (k) WHERE d > 0",
        "CREATE TABLE dq (k CHAR(4), CONSTRAINT dq_k UNIQUE (k) DEFERRABLE INITIALLY DEFERRED)",
    ];

    /// Tables an earlier release built, holding rows whose index entries keep their blanks.
    fn old_data(engine: &dyn StorageEngine) {
        for sql in SCHEMA {
            ok(engine, sql);
        }
        ok(
            engine,
            "CREATE TABLE ex (k CHAR(4), a INT, b INT, EXCLUDE USING gist (k WITH =, int4range(a, b) WITH &&))",
        );
        reset_to_legacy(engine);
        write_old(
            engine,
            "ou",
            &[vec![text("a  "), int(1)], vec![text("b"), int(2)]],
        );
        write_old(engine, "opk", &[vec![text("p  "), int(1)]]);
        write_old(engine, "ch", &[vec![text("p "), int(1)]]);
        write_old(engine, "ix", &[vec![text("x  "), int(1)]]);
        write_old(
            engine,
            "ui",
            &[
                vec![text("u  "), ast::Value::Null, int(1)],
                vec![text("v "), int(1), int(0)],
            ],
        );
        write_old(engine, "dq", &[vec![text("d  ")]]);
        write_old(engine, "ex", &[vec![text("e  "), int(1), int(5)]]);
    }

    fn index_of(engine: &dyn StorageEngine, name: &str) -> IndexId {
        engine.lookup_index(name).unwrap().unwrap()
    }

    fn is_legacy(engine: &dyn StorageEngine, name: &str) -> bool {
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        forget_state(engine);
        super::super::global_unique::invalidate_canon();
        let legacy = super::super::global_unique::canon_state(engine, txn)
            .unwrap()
            .is_legacy(index_of(engine, name));
        super::super::deferred::rollback(engine, txn).unwrap();
        legacy
    }

    /// Every write path refuses a duplicate of an old key, and a lookup finds the old rows.
    fn still_correct(engine: &dyn StorageEngine) {
        refused(engine, "INSERT INTO ou VALUES ('a', 9)");
        refused(engine, "INSERT INTO ou VALUES ('a   ', 9)");
        refused(engine, "UPDATE ou SET k = 'a' WHERE n = 2");
        refused(engine, "INSERT INTO opk VALUES ('p', 9)");
        let crate::ast::Statement::Copy(copy) = parse("COPY ou FROM STDIN").unwrap() else {
            panic!("not a COPY");
        };
        let e = crate::copy_from(engine, &copy, "a \t30\n").expect_err("copy");
        assert!(e.to_string().contains("duplicate key"), "{e}");
        ok(
            engine,
            "INSERT INTO ou VALUES ('a', 40) ON CONFLICT (k) DO UPDATE SET n = n + 0",
        );
        ok(
            engine,
            "INSERT INTO ou VALUES ('a ', 50) ON CONFLICT DO NOTHING",
        );
        assert_eq!(count(engine, "SELECT count(*) FROM ou"), 2);
        // A unique index: NULLS NOT DISTINCT, and a partial one.
        refused(engine, "INSERT INTO ui VALUES ('u', NULL, 0)");
        refused(engine, "INSERT INTO ui VALUES ('v', 1, 0)");
        refused(engine, "INSERT INTO ui VALUES ('u', 5, 3)");
        ok(engine, "INSERT INTO ui VALUES ('v', 5, 0)");
        ok(engine, "DELETE FROM ui WHERE m = 5");
        // An exclusion constraint.
        fails(
            engine,
            "INSERT INTO ex VALUES ('e', 2, 3)",
            "conflicting key",
        );
        // The foreign key.
        let mut s = Session::new(engine);
        assert!(run(engine, &mut s, "DELETE FROM opk WHERE k = 'p'").is_err());
        assert!(run(engine, &mut s, "INSERT INTO ch VALUES ('q', 1)").is_err());
        drop(s);
        ok(engine, "INSERT INTO ch VALUES ('p   ', 2)");
        ok(engine, "DELETE FROM ch WHERE n = 2");
        // Lookups find the old rows.
        assert_eq!(count(engine, "SELECT count(*) FROM ix WHERE k = 'x'"), 1);
        assert_eq!(count(engine, "SELECT count(*) FROM ou WHERE k = 'a'"), 1);
        // The deferred key: a duplicate in a transaction is refused at its commit.
        let mut s = Session::new(engine);
        run(engine, &mut s, "BEGIN").unwrap();
        run(engine, &mut s, "INSERT INTO dq VALUES ('d')").unwrap();
        let e = run(engine, &mut s, "COMMIT").expect_err("commit");
        assert!(e.to_string().contains("duplicate key"), "{e}");
    }

    #[test]
    fn indexes_from_an_earlier_release_stay_correct_and_reindex_rebuilds_them() {
        let dir = nusadb_test_utils::temp_dir();
        let path = dir.path().join("data");
        {
            let engine = BtreeEngine::open(&path).unwrap();
            old_data(&engine);
        }
        let engine = BtreeEngine::open(&path).unwrap();
        // The open records what predates this release, from the catalogs alone.
        assert!(note_legacy_char_indexes(&engine).unwrap() >= 7);
        for name in ["ou_k_key", "opk_pkey", "ix_k", "ui_k", "ui_part", "dq_k"] {
            assert!(is_legacy(&engine, name), "{name}");
        }
        still_correct(&engine);
        // REINDEX rebuilds every one of them.
        for sql in [
            "REINDEX TABLE ou",
            "REINDEX TABLE opk",
            "REINDEX TABLE ch",
            "REINDEX INDEX ix_k",
            "REINDEX TABLE ui",
            "REINDEX TABLE dq",
            "REINDEX TABLE ex",
        ] {
            ok(&engine, sql);
        }
        for name in [
            "ou_k_key", "opk_pkey", "ix_k", "ui_k", "ui_part", "dq_k", "ex_excl",
        ] {
            assert!(!is_legacy(&engine, name), "{name}");
        }
        // One exact probe now.
        super::super::global_unique::RANGES_PROBED.with(|n| n.set(0));
        refused(&engine, "INSERT INTO ou VALUES ('a', 9)");
        assert_eq!(
            super::super::global_unique::RANGES_PROBED.with(std::cell::Cell::get),
            1
        );
        still_correct(&engine);
        // The index kept what it was declared with.
        let txn = engine.begin(IsolationLevel::default()).unwrap();
        let ui = engine.lookup_table_as_of(txn, "ui").unwrap().unwrap();
        let defs = engine.list_indexes_as_of(txn, ui.id).unwrap();
        let ui_k = defs.iter().find(|d| d.name == "ui_k").unwrap();
        assert!(
            super::super::unique_index::split_predicate(ui_k).1,
            "NULLS NOT DISTINCT kept"
        );
        let ui_part = defs.iter().find(|d| d.name == "ui_part").unwrap();
        assert_eq!(
            super::super::unique_index::split_predicate(ui_part).0,
            Some("d > 0")
        );
        super::super::deferred::rollback(&engine, txn).unwrap();
        // A restart keeps it.
        drop(engine);
        let engine = BtreeEngine::open(&path).unwrap();
        assert_eq!(note_legacy_char_indexes(&engine).unwrap(), 0);
        assert!(!is_legacy(&engine, "ou_k_key"));
        still_correct(&engine);
    }

    fn rows_of(engine: &dyn StorageEngine, sql: &str) -> Vec<Vec<ast::Value>> {
        let mut s = Session::new(engine);
        match run(engine, &mut s, sql).unwrap() {
            ExecutionResult::Rows { rows, .. } => rows,
            other => panic!("{sql}: {other:?}"),
        }
    }

    fn plan_of(engine: &dyn StorageEngine, sql: &str) -> String {
        rows_of(engine, &format!("EXPLAIN {sql}"))
            .into_iter()
            .flatten()
            .map(|v| crate::display::value_text(&v))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A legacy index is still read through, with every spelling of a key found, ranges and
    /// orderings right, and each row once.
    #[test]
    fn a_legacy_index_is_scanned_through_every_spelling() {
        let engine = BtreeEngine::new();
        ok(&engine, "CREATE TABLE sc (k CHAR(4) PRIMARY KEY, n INT)");
        ok(
            &engine,
            "CREATE TABLE sc2 (k CHAR(3), m INT, n INT, PRIMARY KEY (k, m))",
        );
        reset_to_legacy(&engine);
        let spellings = ["b  ", "a", "c ", "a b", "d   ", "ab "];
        let rows: Vec<Vec<ast::Value>> = spellings
            .iter()
            .enumerate()
            .map(|(i, k)| vec![text(k), int(i64::try_from(i).unwrap())])
            .collect();
        write_old(&engine, "sc", &rows);
        write_old(
            &engine,
            "sc2",
            &[
                vec![text("x  "), int(1), int(10)],
                vec![text("x"), int(2), int(20)],
                vec![text("y "), int(1), int(30)],
            ],
        );
        assert!(is_legacy(&engine, "sc_pkey"));
        assert!(plan_of(&engine, "SELECT n FROM sc WHERE k = 'b'").contains("IndexScan"));
        let one = |sql: &str| rows_of(&engine, sql);
        assert_eq!(one("SELECT n FROM sc WHERE k = 'b'"), vec![vec![int(0)]]);
        assert_eq!(one("SELECT n FROM sc WHERE k = 'd'"), vec![vec![int(4)]]);
        assert_eq!(
            one("SELECT n FROM sc WHERE k >= 'a' AND k <= 'b' ORDER BY k"),
            vec![vec![int(1)], vec![int(3)], vec![int(5)], vec![int(0)]]
        );
        assert_eq!(
            one("SELECT n FROM sc WHERE k > 'ab' ORDER BY k DESC LIMIT 2"),
            vec![vec![int(4)], vec![int(2)]]
        );
        assert_eq!(
            one("SELECT count(*) FROM sc WHERE k < 'c'"),
            vec![vec![int(4)]]
        );
        // A key prefix of a composite key, and the whole key.
        assert_eq!(
            one("SELECT n FROM sc2 WHERE k = 'x' ORDER BY m"),
            vec![vec![int(10)], vec![int(20)]]
        );
        assert_eq!(
            one("SELECT n FROM sc2 WHERE k = 'x' AND m = 1"),
            vec![vec![int(10)]]
        );
        assert_eq!(
            one("SELECT n FROM sc2 WHERE k = 'y' AND m = 1"),
            vec![vec![int(30)]]
        );
        // An IN list, and capped ascending reads.
        assert_eq!(
            one("SELECT n FROM sc WHERE k IN ('a', 'd') ORDER BY n"),
            vec![vec![int(1)], vec![int(4)]]
        );
        assert_eq!(
            one("SELECT n FROM sc ORDER BY k LIMIT 3"),
            vec![vec![int(1)], vec![int(3)], vec![int(5)]]
        );
        assert_eq!(
            one("SELECT n FROM sc WHERE k >= 'ab' ORDER BY k LIMIT 2"),
            vec![vec![int(5)], vec![int(0)]]
        );
        // UPDATE / DELETE find their rows through it too.
        ok(&engine, "UPDATE sc SET n = 99 WHERE k = 'c'");
        assert_eq!(one("SELECT n FROM sc WHERE k = 'c'"), vec![vec![int(99)]]);
        ok(&engine, "DELETE FROM sc2 WHERE k = 'x' AND m = 1");
        assert_eq!(one("SELECT count(*) FROM sc2"), vec![vec![int(2)]]);
    }

    /// Capped ordered reads through a legacy index, in both directions and across many chunks,
    /// match the same query over an unindexed copy of the rows; `LIMIT 0` reads nothing.
    #[test]
    fn capped_legacy_scans_match_a_sorted_copy() {
        let engine = BtreeEngine::new();
        ok(
            &engine,
            "CREATE TABLE lg (k CHAR(6), m INT, n INT, PRIMARY KEY (k, m))",
        );
        ok(&engine, "CREATE TABLE lg1 (k CHAR(8) PRIMARY KEY, n INT)");
        ok(&engine, "CREATE TABLE tw (k CHAR(6), m INT, n INT)");
        ok(&engine, "CREATE TABLE tw1 (k CHAR(8), n INT)");
        reset_to_legacy(&engine);
        let leads = ["a", "a b", "ab", "b", "b a", "c"];
        let mut rows = Vec::new();
        for i in 0..3000_i64 {
            let lead = leads[usize::try_from(i).unwrap() % leads.len()];
            let blanks = " ".repeat(usize::try_from(i).unwrap() % 4);
            rows.push(vec![text(&format!("{lead}{blanks}")), int(i), int(i % 7)]);
        }
        write_old(&engine, "lg", &rows);
        write_old(&engine, "tw", &rows);
        let singles: Vec<Vec<ast::Value>> = (0..2500_i64)
            .map(|i| {
                let blanks = " ".repeat(usize::try_from(i).unwrap() % 3);
                vec![text(&format!("k{:04}{blanks}", (i * 7919) % 2500)), int(i)]
            })
            .collect();
        write_old(&engine, "lg1", &singles);
        write_old(&engine, "tw1", &singles);
        assert!(
            plan_of(&engine, "SELECT k, m FROM lg ORDER BY k, m LIMIT 5").contains("IndexScan")
        );
        assert!(
            plan_of(&engine, "SELECT k FROM lg1 ORDER BY k DESC LIMIT 5").contains("IndexScan")
        );
        assert_eq!(
            rows_of(&engine, "SELECT k FROM lg1 ORDER BY k LIMIT 0"),
            Vec::<Vec<ast::Value>>::new()
        );
        assert_eq!(
            rows_of(&engine, "SELECT k FROM lg1 ORDER BY k DESC LIMIT 0"),
            Vec::<Vec<ast::Value>>::new()
        );
        for limit in [1, 7, 100, 999, 1500, 2999, 5000] {
            for (shape, twin) in [
                (
                    "SELECT k, m FROM lg ORDER BY k, m",
                    "SELECT k, m FROM tw ORDER BY k, m",
                ),
                (
                    "SELECT k, m FROM lg ORDER BY k DESC, m DESC",
                    "SELECT k, m FROM tw ORDER BY k DESC, m DESC",
                ),
                (
                    "SELECT k FROM lg1 ORDER BY k",
                    "SELECT k FROM tw1 ORDER BY k",
                ),
                (
                    "SELECT k FROM lg1 ORDER BY k DESC",
                    "SELECT k FROM tw1 ORDER BY k DESC",
                ),
                (
                    "SELECT k FROM lg1 WHERE k >= 'k1000' ORDER BY k",
                    "SELECT k FROM tw1 WHERE k >= 'k1000' ORDER BY k",
                ),
            ] {
                let got = rows_of(&engine, &format!("{shape} LIMIT {limit}"));
                let want = rows_of(&engine, &format!("{twin} LIMIT {limit}"));
                assert_eq!(got, want, "{shape} LIMIT {limit}");
            }
        }
        // Predicate scans stream the spellings and keep every row once.
        for (q, t) in [
            (
                "SELECT count(*) FROM lg WHERE k = 'a b'",
                "SELECT count(*) FROM tw WHERE k = 'a b'",
            ),
            (
                "SELECT count(*) FROM lg WHERE k > 'a' AND k < 'b a'",
                "SELECT count(*) FROM tw WHERE k > 'a' AND k < 'b a'",
            ),
            (
                "SELECT count(*) FROM lg1 WHERE k BETWEEN 'k0100' AND 'k0200'",
                "SELECT count(*) FROM tw1 WHERE k BETWEEN 'k0100' AND 'k0200'",
            ),
        ] {
            assert_eq!(rows_of(&engine, q), rows_of(&engine, t), "{q}");
        }
    }

    /// A long run of one key (its spellings mixed) read through a legacy index with `ORDER BY ...
    /// LIMIT`: the entries read stay bounded by the limit, not the run, and the rows match a sorted
    /// copy in both directions, including keys that sit between the run's spellings.
    #[test]
    fn a_long_equal_key_run_is_not_read_through() {
        let engine = BtreeEngine::new();
        for t in ["dx", "dxs"] {
            ok(
                &engine,
                &format!("CREATE TABLE {t} (id INT, c CHAR(6) NOT NULL, v TEXT)"),
            );
        }
        ok(&engine, "CREATE INDEX dx_c ON dx (c)");
        reset_to_legacy(&engine);
        let mut rows = Vec::new();
        for i in 0..100_000_i64 {
            let blanks = match i % 10 {
                0..=5 => 5,
                6 | 7 => 0,
                8 => 2,
                _ => 1,
            };
            rows.push(vec![
                int(i),
                text(&format!("a{}", " ".repeat(blanks))),
                text("v"),
            ]);
        }
        for (i, k) in ["b", "b ", "A", "A  ", "a\t", "a \t ", "ab", "Z", "b c"]
            .iter()
            .enumerate()
        {
            rows.push(vec![
                int(-1 - i64::try_from(i).unwrap()),
                text(k),
                text("w"),
            ]);
        }
        write_old(&engine, "dx", &rows);
        write_old(&engine, "dxs", &rows);
        let reads = |sql: &str| {
            super::super::scan::CAPPED_ENTRIES_READ.with(|n| n.set(0));
            let got = rows_of(&engine, sql);
            (
                got,
                super::super::scan::CAPPED_ENTRIES_READ.with(std::cell::Cell::get),
            )
        };
        for limit in [1_usize, 3, 10, 100, 2000] {
            for (shape, twin) in [
                (
                    "SELECT c FROM dx ORDER BY c",
                    "SELECT c FROM dxs ORDER BY c",
                ),
                (
                    "SELECT c FROM dx ORDER BY c DESC",
                    "SELECT c FROM dxs ORDER BY c DESC",
                ),
                (
                    "SELECT c FROM dx WHERE c >= 'a' ORDER BY c",
                    "SELECT c FROM dxs WHERE c >= 'a' ORDER BY c",
                ),
                (
                    "SELECT c FROM dx WHERE c <= 'a' ORDER BY c DESC",
                    "SELECT c FROM dxs WHERE c <= 'a' ORDER BY c DESC",
                ),
                (
                    "SELECT c FROM dx WHERE c = 'a' ORDER BY c",
                    "SELECT c FROM dxs WHERE c = 'a' ORDER BY c",
                ),
            ] {
                let sql = format!("{shape} LIMIT {limit}");
                assert!(plan_of(&engine, &sql).contains("IndexScan"), "{sql}");
                let (got, read) = reads(&sql);
                let want = rows_of(&engine, &format!("{twin} LIMIT {limit}"));
                assert_eq!(got, want, "{sql}");
                assert!(
                    read > 0 && read <= 2 * limit + 64,
                    "{sql}: read {read} entries"
                );
            }
        }
    }

    /// A legacy index over a very wide `CHAR(n)` column read with `ORDER BY ... LIMIT`: the work
    /// does not grow with the width (no bound is padded past a few hundred blanks, and the gaps
    /// between the spellings of a key are capped), and the rows match a sorted copy, including
    /// keys that sit in a gap past the cap. (The column is read as `c || ''`, which a result does
    /// not pad to the full width.)
    #[test]
    fn a_wide_char_legacy_index_reads_are_bounded() {
        let engine = BtreeEngine::new();
        for t in ["w", "ws"] {
            ok(
                &engine,
                &format!("CREATE TABLE {t} (id INT, c CHAR(100000) NOT NULL)"),
            );
        }
        ok(&engine, "CREATE INDEX w_c ON w (c)");
        reset_to_legacy(&engine);
        let mut rows = Vec::new();
        for i in 0..3000_i64 {
            rows.push(vec![int(i), text(&format!("a{}", i % 3))]);
        }
        rows.push(vec![int(-1), text(&format!("a1{}", " ".repeat(300)))]);
        rows.push(vec![int(-2), text(&format!("a2{}", " ".repeat(99_998)))]);
        rows.push(vec![int(-3), text(&format!("a1{}\t", " ".repeat(300)))]);
        rows.push(vec![int(-4), text("a1 \t")]);
        write_old(&engine, "w", &rows);
        write_old(&engine, "ws", &rows);
        let reads = |sql: &str| {
            super::super::scan::CAPPED_ENTRIES_READ.with(|n| n.set(0));
            let got = rows_of(&engine, sql);
            (
                got,
                super::super::scan::CAPPED_ENTRIES_READ.with(std::cell::Cell::get),
            )
        };
        for limit in [1_usize, 2, 4, 10, 500, 1004] {
            for (shape, twin) in [
                (
                    "SELECT c || '' FROM w ORDER BY c DESC",
                    "SELECT c || '' FROM ws ORDER BY c DESC",
                ),
                (
                    "SELECT c || '' FROM w WHERE c < 'a2' ORDER BY c DESC",
                    "SELECT c || '' FROM ws WHERE c < 'a2' ORDER BY c DESC",
                ),
                (
                    "SELECT c || '' FROM w ORDER BY c",
                    "SELECT c || '' FROM ws ORDER BY c",
                ),
                (
                    "SELECT c || '' FROM w WHERE c >= 'a1' ORDER BY c",
                    "SELECT c || '' FROM ws WHERE c >= 'a1' ORDER BY c",
                ),
                (
                    "SELECT c || '' FROM w WHERE c <= 'a2' ORDER BY c DESC",
                    "SELECT c || '' FROM ws WHERE c <= 'a2' ORDER BY c DESC",
                ),
                (
                    "SELECT c || '' FROM w WHERE c = 'a2' ORDER BY c",
                    "SELECT c || '' FROM ws WHERE c = 'a2' ORDER BY c",
                ),
            ] {
                let sql = format!("{shape} LIMIT {limit}");
                assert!(plan_of(&engine, &sql).contains("IndexScan"), "{sql}");
                let (got, read) = reads(&sql);
                let want = rows_of(&engine, &format!("{twin} LIMIT {limit}"));
                let trimmed = |rows: Vec<Vec<ast::Value>>| -> Vec<String> {
                    rows.into_iter()
                        .map(|r| {
                            crate::display::value_text(&r[0])
                                .trim_end_matches(' ')
                                .to_owned()
                        })
                        .collect()
                };
                let want = trimmed(want);
                // A cut at a key holding a character below the blank has no bound on how far the
                // read goes (a known limit); every other cut stops within the limit.
                let low_cut = want.last().is_some_and(|k| k.chars().any(|c| c < ' '));
                assert_eq!(trimmed(got), want, "{sql}");
                assert!(
                    read > 0 && (low_cut || read <= 2 * limit + 64),
                    "{sql}: read {read} entries"
                );
            }
        }
        // Equality and range lookups through it.
        assert_eq!(
            count(&engine, "SELECT count(*) FROM w WHERE c = 'a2'"),
            1001
        );
        assert_eq!(
            count(
                &engine,
                "SELECT count(*) FROM w WHERE c > 'a1' AND c < 'a3'"
            ),
            1003
        );
        // Bounds written with trailing blanks or a character below the blank match the copy.
        for filter in [
            "c < 'a1 '",
            "c <= 'a1 '",
            "c < 'a1'",
            "c > 'a1 '",
            "c >= 'a1 '",
            "c < 'a1 \t'",
            "c > 'a1' AND c < 'a2  '",
        ] {
            assert_eq!(
                count(&engine, &format!("SELECT count(*) FROM w WHERE {filter}")),
                count(&engine, &format!("SELECT count(*) FROM ws WHERE {filter}")),
                "{filter}"
            );
        }
    }

    /// A legacy index whose wide `CHAR(n)` column comes before another key column: ordered reads
    /// in both directions, with and without bounds, and lookups on the leading column match a
    /// sorted copy, every bound staying a few hundred blanks long.
    #[test]
    fn a_wide_leading_char_legacy_index_matches_a_sorted_copy() {
        let engine = BtreeEngine::new();
        for t in ["w", "ws"] {
            ok(
                &engine,
                &format!("CREATE TABLE {t} (id INT NOT NULL, c CHAR(100000) NOT NULL)"),
            );
        }
        ok(&engine, "CREATE INDEX w_c ON w (c, id)");
        reset_to_legacy(&engine);
        let mut rows = Vec::new();
        for i in 0..600_i64 {
            rows.push(vec![int(i), text(&format!("a{}", i % 3))]);
        }
        rows.push(vec![int(-1), text(&format!("a1{}", " ".repeat(300)))]);
        rows.push(vec![int(-2), text(&format!("a2{}", " ".repeat(99_998)))]);
        rows.push(vec![int(-3), text(&format!("a1{}\t", " ".repeat(300)))]);
        write_old(&engine, "w", &rows);
        write_old(&engine, "ws", &rows);
        let shown = |rows: Vec<Vec<ast::Value>>| -> Vec<String> {
            rows.into_iter()
                .map(|r| {
                    format!(
                        "{}|{}",
                        crate::display::value_text(&r[0]).trim_end_matches(' '),
                        crate::display::value_text(&r[1])
                    )
                })
                .collect()
        };
        for limit in [1_usize, 3, 50, 700] {
            for (filter, order) in [
                ("", "c DESC, id DESC"),
                ("", "c, id"),
                ("WHERE c < 'a2'", "c DESC, id DESC"),
                ("WHERE c <= 'a1'", "c DESC, id DESC"),
                ("WHERE c >= 'a1'", "c, id"),
                ("WHERE c = 'a1'", "c, id"),
                ("WHERE c = 'a2'", "c DESC, id DESC"),
            ] {
                let sql = |t: &str| {
                    format!("SELECT c || '', id FROM {t} {filter} ORDER BY {order} LIMIT {limit}")
                };
                assert!(
                    plan_of(&engine, &sql("w")).contains("IndexScan"),
                    "{}\n{}",
                    sql("w"),
                    plan_of(&engine, &sql("w"))
                );
                assert_eq!(
                    shown(rows_of(&engine, &sql("w"))),
                    shown(rows_of(&engine, &sql("ws"))),
                    "{}",
                    sql("w")
                );
            }
        }
        assert_eq!(count(&engine, "SELECT count(*) FROM w WHERE c = 'a2'"), 201);
    }

    /// A partial unique index whose predicate reads no CHAR column is probed through the key's
    /// spellings; one whose predicate reads a CHAR column reads the table once per statement.
    #[test]
    fn legacy_partial_unique_indexes_are_checked_without_a_scan_per_row() {
        let engine = BtreeEngine::new();
        ok(&engine, "CREATE TABLE pu (k CHAR(4), d INT, c CHAR(2))");
        ok(&engine, "CREATE UNIQUE INDEX pu_d ON pu (k) WHERE d > 0");
        ok(
            &engine,
            "CREATE UNIQUE INDEX pu_c ON pu (k, d) WHERE c = 'y'",
        );
        reset_to_legacy(&engine);
        write_old(
            &engine,
            "pu",
            &[
                vec![text("a  "), int(1), text("n ")],
                vec![text("b "), int(2), text("y ")],
            ],
        );
        refused(&engine, "INSERT INTO pu VALUES ('a', 5, 'n')");
        ok(&engine, "INSERT INTO pu VALUES ('a', 0, 'n')");
        refused(&engine, "INSERT INTO pu VALUES ('b', 2, 'y')");
        // Many rows in one statement (the table is read once, not once per row).
        let values: Vec<String> = (0..200)
            .map(|i| format!("('k{i}', {}, 'y')", i + 10))
            .collect();
        ok(
            &engine,
            &format!("INSERT INTO pu VALUES {}", values.join(", ")),
        );
        assert_eq!(count(&engine, "SELECT count(*) FROM pu"), 203);
        // Rewriting every row keeps its own key: no row collides with itself.
        ok(&engine, "UPDATE pu SET d = d + 1000 WHERE d > 0");
        ok(&engine, "UPDATE pu SET c = 'y' WHERE k = 'a' AND d = 1001");
        refused(&engine, "UPDATE pu SET k = 'b' WHERE k = 'a' AND d = 1001");
        refused(&engine, "INSERT INTO pu VALUES ('b', 1002, 'y')");
        assert_eq!(count(&engine, "SELECT count(*) FROM pu"), 203);
    }

    /// REINDEX needs the table's owner; SCHEMA / DATABASE skip the tables the role does not own.
    #[test]
    fn reindex_needs_the_owner() {
        let engine = BtreeEngine::new();
        for sql in SCHEMA {
            ok(&engine, sql);
        }
        ok(&engine, "CREATE ROLE bob LOGIN");
        reset_to_legacy(&engine);
        write_old(
            &engine,
            "ou",
            &[vec![text("a  "), int(1)], vec![text("a"), int(2)]],
        );
        let as_bob = |sql: &str| {
            let mut s = Session::new(&engine);
            s.set_current_user("bob");
            run(&engine, &mut s, sql)
        };
        for sql in ["REINDEX TABLE ou", "REINDEX INDEX ou_k_key"] {
            let e = as_bob(sql).expect_err(sql).to_string();
            assert!(e.contains("must be the owner"), "{sql}: {e}");
            assert!(!e.contains("(a)"), "{sql} leaked a key: {e}");
        }
        as_bob("REINDEX SCHEMA public").unwrap();
        assert!(is_legacy(&engine, "ou_k_key"));
        assert!(is_legacy(&engine, "opk_pkey"));
        // The owner sees the key it may read.
        fails(&engine, "REINDEX TABLE ou", "key (k)=(a)");
        fails(
            &engine,
            "REINDEX DATABASE elsewhere",
            "can only reindex the currently open database",
        );
    }

    /// A unique index whose rows hold one key under two spellings (data an earlier release
    /// accepted): REINDEX refuses with 23505 naming the key and leaves the index as it was.
    #[test]
    fn reindex_refuses_a_duplicate_and_leaves_the_index_enforced() {
        let engine = BtreeEngine::new();
        for sql in SCHEMA {
            ok(&engine, sql);
        }
        reset_to_legacy(&engine);
        write_old(
            &engine,
            "ou",
            &[vec![text("a  "), int(1)], vec![text("a"), int(2)]],
        );
        write_old(
            &engine,
            "ui",
            &[
                vec![text("w  "), int(1), int(0)],
                vec![text("w"), int(1), int(0)],
            ],
        );
        let e = fails(&engine, "REINDEX TABLE ou", "key (k)=(a)");
        assert!(e.contains("ou_k_key"), "{e}");
        fails(&engine, "REINDEX INDEX ui_k", "ui_k");
        assert!(is_legacy(&engine, "ou_k_key"));
        assert!(is_legacy(&engine, "ui_k"));
        refused(&engine, "INSERT INTO ou VALUES ('a ', 3)");
        refused(&engine, "INSERT INTO ui VALUES ('w', 1, 0)");
        assert_eq!(count(&engine, "SELECT count(*) FROM ou"), 2);
        // Other tables' indexes are not touched by the failure.
        ok(&engine, "REINDEX TABLE opk");
        assert!(!is_legacy(&engine, "opk_pkey"));
    }

    /// The duplicate check of a rebuilt key constraint finds a key held twice, by its hashed pass
    /// and by its key-order walk (a chunk at a time, stepping over more `NULL` keys than a chunk
    /// holds).
    #[test]
    fn reindex_checks_keys_across_chunks() {
        let engine = BtreeEngine::new();
        ok(&engine, "CREATE TABLE big (k CHAR(6) UNIQUE, n INT)");
        reset_to_legacy(&engine);
        let mut rows: Vec<Vec<ast::Value>> =
            (0..3000).map(|i| vec![ast::Value::Null, int(i)]).collect();
        rows.extend((0..3000).map(|i| vec![text(&format!("{i:05}")), int(i)]));
        write_old(&engine, "big", &rows);
        write_old(&engine, "big", &[vec![text("01500 "), int(-1)]]);
        // Both passes: the hashed one, and the key-order walk past a small key budget.
        for cap in [None, Some(100)] {
            VERIFY_CAP.with(|c| c.set(cap));
            fails(&engine, "REINDEX TABLE big", "key (k)=(01500)");
            assert!(is_legacy(&engine, "big_k_key"));
        }
        ok(&engine, "DELETE FROM big WHERE n = -1");
        ok(&engine, "REINDEX TABLE big");
        VERIFY_CAP.with(|c| c.set(None));
        assert!(!is_legacy(&engine, "big_k_key"));
        assert_eq!(
            count(&engine, "SELECT count(*) FROM big WHERE k = '01500'"),
            1
        );
        // Hashes narrowed to two values: nearly every key collides with an earlier one, and each
        // collision is settled by probing the index, finding the key held once.
        reset_to_legacy(&engine);
        VERIFY_HASH_MASK.with(|m| m.set(1));
        ok(&engine, "REINDEX TABLE big");
        assert!(!is_legacy(&engine, "big_k_key"));
        reset_to_legacy(&engine);
        write_old(&engine, "big", &[vec![text("02999  "), int(-2)]]);
        fails(&engine, "REINDEX TABLE big", "key (k)=(02999)");
        VERIFY_HASH_MASK.with(|m| m.set(u64::MAX));
    }

    /// Every index this build creates is canonical, with nothing to rebuild.
    #[test]
    fn new_indexes_are_canonical() {
        let dir = nusadb_test_utils::temp_dir();
        let engine = BtreeEngine::open(dir.path().join("data")).unwrap();
        for sql in SCHEMA {
            ok(&engine, sql);
        }
        ok(&engine, "CREATE TABLE lk (LIKE ou INCLUDING ALL)");
        ok(
            &engine,
            "CREATE TABLE pp (k CHAR(4), d INT NOT NULL, UNIQUE (k)) PARTITION BY LIST (d)",
        );
        ok(
            &engine,
            "CREATE TABLE pp_1 PARTITION OF pp FOR VALUES IN (1)",
        );
        ok(&engine, "TRUNCATE ou CASCADE");
        assert_eq!(note_legacy_char_indexes(&engine).unwrap(), 0);
        for name in [
            "ou_k_key",
            "opk_pkey",
            "ix_k",
            "ui_k",
            "ui_part",
            "dq_k",
            "pp_1_k_key",
        ] {
            assert!(!is_legacy(&engine, name), "{name}");
        }
        super::super::global_unique::RANGES_PROBED.with(|n| n.set(0));
        ok(&engine, "INSERT INTO ou VALUES ('a  ', 1)");
        refused(&engine, "INSERT INTO ou VALUES ('a', 2)");
        assert!(super::super::global_unique::RANGES_PROBED.with(std::cell::Cell::get) <= 2);
    }

    /// A crash during REINDEX (its transaction never commits) leaves the old index, still legacy
    /// and enforced; REINDEX can run again.
    #[test]
    fn a_crash_during_reindex_leaves_the_old_index() {
        let dir = nusadb_test_utils::temp_dir();
        let path = dir.path().join("data");
        {
            let engine = BtreeEngine::open(&path).unwrap();
            old_data(&engine);
            note_legacy_char_indexes(&engine).unwrap();
        }
        {
            let engine = BtreeEngine::open(&path).unwrap();
            let txn = engine.begin(IsolationLevel::default()).unwrap();
            run_reindex(&ast::ReindexTarget::Database(None), &engine, txn).unwrap();
            // The process dies here: the transaction is never committed.
        }
        let engine = BtreeEngine::open(&path).unwrap();
        assert!(is_legacy(&engine, "ou_k_key"));
        still_correct(&engine);
        ok(&engine, "REINDEX DATABASE nusadb");
        assert!(!is_legacy(&engine, "ou_k_key"));
        assert_eq!(note_legacy_char_indexes(&engine).unwrap(), 0);
        still_correct(&engine);
    }
}
