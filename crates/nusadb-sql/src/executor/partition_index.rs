//! Indexes on partitioned tables.
//!
//! A partitioned table stores no rows, so an index on it indexes nothing by itself: it stands for
//! one index on each partition. `CREATE INDEX` on a partitioned table creates the index on the
//! table and builds a matching index on every partition below it, at every level of
//! sub-partitioning; a partition created later (`PARTITION OF`) gets one when it is created, and a
//! table attached later (`ATTACH PARTITION`) adopts an equivalent index it already has (same
//! access method, key columns or expressions, predicate and `INCLUDE` columns) or gets one built.
//! Each partition's index is an ordinary index of that partition: a query that reads the
//! partition through the partitioned table scans it like any other index.
//!
//! # The link
//!
//! Which partition index belongs to which index of the partitioned table is recorded in an
//! engine-scoped SQL-layer catalog, `nusadb_partition_indexes`, one row per partition index:
//! `(index name, parent index name)`. Index names are unique in the engine's catalog and are kept
//! by every operation that rebuilds an index under the same name (`TRUNCATE`, `REINDEX`). A row
//! counts only while it describes the tree as it is: the child index lies on a direct partition of
//! the table the parent index lies on. A dropped index leaves no row behind ([`forget`]).
//!
//! The link decides what follows the partitioned table's index: `DROP INDEX` on it drops the
//! partitions' indexes too, and dropping one of those directly is refused (`2BP01`), as is
//! dropping the index of a sub-partitioned partition that is itself linked. `DETACH PARTITION`
//! unlinks the detached table's indexes, which it keeps as ordinary indexes of its own.
//!
//! # Names
//!
//! A partition's index is named after the partition and the key, the way the reference engine
//! names it: `<partition>_<column>[_<column>...]_idx`, with `expr` standing for each key
//! expression, and a number appended (`_idx1`, `_idx2`, ...) when the name is taken.
//!
//! # Earlier releases
//!
//! An index created on a partitioned table by an earlier release has no partition indexes and no
//! links. It is left as it is: queries stay correct (each partition is read through its own
//! indexes, so it is read by a scan), a partition created or attached from now on gets its index,
//! and `DROP INDEX` followed by `CREATE INDEX` builds it on the existing partitions.
#![allow(clippy::wildcard_imports)]

use std::collections::HashSet;

use nusadb_core::{IndexDef, Tid};

use super::*;

/// Engine-scoped catalog linking each partition index to its partitioned table's index.
const CATALOG: &str = "nusadb_partition_indexes";

/// The catalog's column names, in order; both `TEXT NOT NULL`.
const CATALOG_COLUMNS: [&str; 2] = ["index_name", "parent_index"];

/// `(index_name, parent_index)`, both text.
const CATALOG_SCHEMA: [ColumnType; 2] = [ColumnType::Text; 2];

/// The deepest partition tree walked; DDL cannot build a cycle, so this only bounds a corrupt one.
const MAX_DEPTH: usize = 64;

/// The catalog's table, when it exists with the catalog's shape. A table of that name with another
/// shape holds no links and is never written.
fn catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<Option<nusadb_core::TableId>, Error> {
    Ok(engine.lookup_table_as_of(txn, CATALOG)?.and_then(|t| {
        let shaped = t.columns.len() == CATALOG_COLUMNS.len()
            && t.columns
                .iter()
                .zip(CATALOG_COLUMNS)
                .all(|(c, name)| c.name == name && c.ty == ColumnType::Text && !c.nullable);
        shaped.then_some(t.id)
    }))
}

fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    if let Some(id) = catalog(engine, txn)? {
        return Ok(id);
    }
    if engine.lookup_table_as_of(txn, CATALOG)?.is_some() {
        return Err(Error::Coded {
            message: format!(
                "cannot link a partition index: a table named \"{CATALOG}\" exists and is not \
                 the catalog this engine keeps under that name; rename or drop it"
            ),
            sqlstate: "42P07",
        });
    }
    let def = TableDef {
        schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
        name: CATALOG.to_owned(),
        columns: CATALOG_COLUMNS
            .iter()
            .map(|name| ColumnDef {
                name: (*name).to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            })
            .collect(),
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Every link row as `(address, index, parent index)`.
fn links(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<(Tid, String, String)>, Error> {
    let Some(catalog) = catalog(engine, txn)? else {
        return Ok(Vec::new());
    };
    let mut scan = engine.scan(txn, catalog)?;
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &CATALOG_SCHEMA)?;
        if let (Some(ast::Value::Text(child)), Some(ast::Value::Text(parent))) =
            (row.first(), row.get(1))
        {
            out.push((tid, child.clone(), parent.clone()));
        }
    }
    Ok(out)
}

/// The link rows as one statement reads and writes them: the catalog is scanned once, and the
/// rows the statement adds or replaces are kept in step here, so building an index on many
/// partitions does not scan the whole catalog again for each of them.
struct Links {
    rows: Vec<(Tid, String, String)>,
}

impl Links {
    fn load(engine: &dyn StorageEngine, txn: TxnId) -> Result<Self, Error> {
        Ok(Self {
            rows: links(engine, txn)?,
        })
    }

    /// The index the link of `child` names as its parent, among the indexes in `standing`.
    fn standing_parent_of(&self, child: &str, standing: &HashSet<String>) -> Option<&str> {
        self.rows
            .iter()
            .find(|(_, c, p)| c == child && standing.contains(p))
            .map(|(_, _, p)| p.as_str())
    }

    /// Record `child` as the partition index of `parent`, replacing any earlier link of `child`.
    fn link(
        &mut self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        child: &str,
        parent: &str,
    ) -> Result<(), Error> {
        let catalog = ensure_catalog(engine, txn)?;
        for (tid, name, _) in &self.rows {
            if name == child {
                engine.delete(txn, catalog, *tid)?;
            }
        }
        self.rows.retain(|(_, name, _)| name != child);
        let row = [
            ast::Value::Text(child.to_owned()),
            ast::Value::Text(parent.to_owned()),
        ];
        let tid = engine.insert(txn, catalog, &row::encode(&row, &CATALOG_SCHEMA)?)?;
        self.rows.push((tid, child.to_owned(), parent.to_owned()));
        Ok(())
    }
}

/// Drop the links that name the index `name`, on either side: called when the index is dropped.
///
/// # Errors
/// Storage errors.
pub(super) fn forget(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<(), Error> {
    forget_all(engine, txn, &HashSet::from([name]))
}

/// Drop the links that name any of the indexes `names`, on either side, in one pass over the
/// catalog.
fn forget_all(engine: &dyn StorageEngine, txn: TxnId, names: &HashSet<&str>) -> Result<(), Error> {
    let Some(catalog) = catalog(engine, txn)? else {
        return Ok(());
    };
    for (tid, child, parent) in links(engine, txn)? {
        if names.contains(child.as_str()) || names.contains(parent.as_str()) {
            engine.delete(txn, catalog, tid)?;
        }
    }
    Ok(())
}

/// The schema-qualified key of `table`, as the partition catalog names it.
fn key_of(table: &TableSchema) -> String {
    crate::analyzer::qualified_display(&table.schema, &table.name)
}

/// Resolve a schema-qualified partition-catalog key.
fn table_of(
    key: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<TableSchema>, Error> {
    let (schema, name) = crate::analyzer::split_qualified(key);
    Ok(engine.lookup_table_as_of_in(txn, schema.unwrap_or(nusadb_core::PUBLIC_SCHEMA), name)?)
}

/// Whether `table` is a partitioned table.
fn is_partitioned(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<bool, Error> {
    Ok(super::partition::has_any(engine, txn)?
        && super::partition::parent_key_parts(engine, txn, &key_of(table))?.is_some())
}

/// The direct partitions of `table`.
fn direct_partitions(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<TableSchema>, Error> {
    let key = key_of(table);
    let mut out = Vec::new();
    for (child, parent) in super::partition::partition_edges(engine, txn)? {
        if parent == key
            && let Some(partition) = table_of(&child, engine, txn)?
        {
            out.push(partition);
        }
    }
    Ok(out)
}

/// The indexes of `table` that stand for an index on each of its partitions: every index that
/// does not back a constraint (a `PRIMARY KEY` / `UNIQUE` / `FOREIGN KEY` reaches the partitions
/// as a constraint of its own).
fn own_indexes(
    table: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<IndexDef>, Error> {
    let backing = super::unique_index::backing_index_ids(table, engine)?;
    let mut out = Vec::new();
    for def in engine.list_indexes_as_of(txn, table.id)? {
        if let Some(id) = engine.lookup_index(&def.name)?
            && !backing.contains(&id)
        {
            out.push(def);
        }
    }
    Ok(out)
}

/// Whether two index definitions index the same rows under the same key: the same access method,
/// uniqueness, key columns or expressions, predicate and `INCLUDE` columns.
fn equivalent(a: &IndexDef, b: &IndexDef) -> bool {
    a.kind == b.kind
        && a.unique == b.unique
        && a.columns == b.columns
        && a.key_exprs == b.key_exprs
        && a.predicate == b.predicate
        && a.include == b.include
}

/// The name a new index of `partition` copied from `def` takes: `<partition>_<key>_idx`, numbered
/// when that is taken by an index, a vector index or a table of the partition's schema.
fn child_index_name(
    partition: &TableSchema,
    def: &IndexDef,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<String, Error> {
    // One label per key part: the column's name, or `expr` for an expression; a repeated label is
    // numbered so the parts stay apart.
    let mut labels: Vec<String> = Vec::new();
    let parts = def
        .columns
        .iter()
        .map(String::as_str)
        .chain(def.key_exprs.iter().map(|_| "expr"));
    for part in parts {
        let mut label = part.to_owned();
        let mut n = 0;
        while labels.contains(&label) {
            n += 1;
            label = format!("{part}{n}");
        }
        labels.push(label);
    }
    let base = format!("{}_{}_idx", partition.name, labels.join("_"));
    let taken = |name: &str| -> Result<bool, Error> {
        Ok(engine.lookup_index(name)?.is_some()
            || super::vector_index_exists(engine, txn, name)?
            || engine
                .lookup_table_as_of_in(txn, &partition.schema, name)?
                .is_some())
    };
    let mut name = base.clone();
    let mut n = 0u32;
    while taken(&name)? {
        n += 1;
        name = format!("{base}{n}");
    }
    Ok(name)
}

/// Give `partition` its index for the index `parent` of the partitioned table it belongs to, and
/// recurse below it when it is partitioned itself: an index already linked to `parent` is kept, an
/// equivalent one not linked to anything is adopted, and otherwise one is built over the
/// partition's rows.
fn ensure_on_partition(
    parent: &IndexDef,
    partition: &TableSchema,
    depth: usize,
    links: &mut Links,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if depth >= MAX_DEPTH {
        return Err(Error::Internal(format!(
            "the partitions below \"{}\" are nested more than {MAX_DEPTH} levels deep",
            partition.name
        )));
    }
    let existing = own_indexes(partition, engine, txn)?;
    // A link counts while its parent index stands on the table `parent` lies on (the partition's
    // parent): one left behind by an index dropped some other way names nothing there.
    let standing: HashSet<String> = engine
        .list_indexes_as_of(txn, parent.table)?
        .into_iter()
        .map(|d| d.name)
        .collect();
    let linked_to_parent = existing.iter().find(|def| {
        equivalent(def, parent)
            && links.standing_parent_of(&def.name, &standing) == Some(parent.name.as_str())
    });
    let child = if let Some(def) = linked_to_parent {
        def.clone()
    } else if let Some(def) = existing.iter().find(|def| {
        equivalent(def, parent) && links.standing_parent_of(&def.name, &standing).is_none()
    }) {
        links.link(engine, txn, &def.name, &parent.name)?;
        def.clone()
    } else {
        let def = IndexDef {
            name: child_index_name(partition, parent, engine, txn)?,
            table: partition.id,
            ..parent.clone()
        };
        super::ddl::build_index(&def, engine, txn)?;
        links.link(engine, txn, &def.name, &parent.name)?;
        def
    };
    if is_partitioned(partition, engine, txn)? {
        for below in direct_partitions(partition, engine, txn)? {
            ensure_on_partition(&child, &below, depth + 1, links, engine, txn)?;
        }
    }
    Ok(())
}

/// `CREATE INDEX` on `table`: when it is a partitioned table, build the index `def` (just created
/// on it) on every partition below it.
///
/// # Errors
/// Storage and index-build errors.
pub(super) fn created_on(
    table: &TableSchema,
    def: &IndexDef,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if !is_partitioned(table, engine, txn)? {
        return Ok(());
    }
    let mut links = Links::load(engine, txn)?;
    for partition in direct_partitions(table, engine, txn)? {
        ensure_on_partition(def, &partition, 1, &mut links, engine, txn)?;
    }
    Ok(())
}

/// A partition joined `parent` (`CREATE TABLE ... PARTITION OF` or `ATTACH PARTITION`): give it,
/// and every partition below it, an index for each index of `parent`.
///
/// The transaction first takes a shared lock on `parent`, held until it ends. A `CREATE INDEX` on
/// `parent` holds `parent` exclusively until its transaction ends, so the two conflict: whichever
/// comes second fails with the no-wait `40001`. Without it, a partition joining while another
/// transaction's `CREATE INDEX` on `parent` is uncommitted would see no such index and never get
/// one. Writers, which take the same shared lock, are not held up.
///
/// # Errors
/// `40001` for a concurrent `CREATE INDEX` on `parent`; storage and index-build errors.
pub(super) fn joined(
    parent: &TableSchema,
    partition: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    engine.lock_table(
        txn,
        parent.id,
        nusadb_core::engine::TableLockMode::AccessShare,
    )?;
    let mut links = Links::load(engine, txn)?;
    for def in own_indexes(parent, engine, txn)? {
        ensure_on_partition(&def, partition, 1, &mut links, engine, txn)?;
    }
    Ok(())
}

/// `DETACH PARTITION`: the detached table keeps its indexes as its own, unlinked from `parent`'s.
///
/// # Errors
/// Storage errors.
pub(super) fn detached(
    parent: &TableSchema,
    partition: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(catalog) = catalog(engine, txn)? else {
        return Ok(());
    };
    let parent_indexes: HashSet<String> = engine
        .list_indexes_as_of(txn, parent.id)?
        .into_iter()
        .map(|d| d.name)
        .collect();
    let own: HashSet<String> = engine
        .list_indexes_as_of(txn, partition.id)?
        .into_iter()
        .map(|d| d.name)
        .collect();
    for (tid, child, linked_parent) in links(engine, txn)? {
        if own.contains(&child) && parent_indexes.contains(&linked_parent) {
            engine.delete(txn, catalog, tid)?;
        }
    }
    Ok(())
}

/// The partitioned table's index that the index `name` of `table` belongs to, if it is linked to
/// one that still stands over `table`'s partitioned parent.
fn live_parent(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<(String, TableSchema)>, Error> {
    if !super::partition::has_any(engine, txn)? {
        return Ok(None);
    }
    let Some(parent_key) = super::partition::partition_parent(engine, txn, &key_of(table))? else {
        return Ok(None);
    };
    let Some(parent_table) = table_of(&parent_key, engine, txn)? else {
        return Ok(None);
    };
    let Some(parent_index) = links(engine, txn)?
        .into_iter()
        .find(|(_, child, _)| child == name)
        .map(|(_, _, parent)| parent)
    else {
        return Ok(None);
    };
    let stands = engine
        .list_indexes_as_of(txn, parent_table.id)?
        .iter()
        .any(|d| d.name == parent_index);
    Ok(stands.then_some((parent_index, parent_table)))
}

/// The partition indexes linked to the index `name` of `table`, at every level below it, deepest
/// first, as `(partition, index name)`.
///
/// # Errors
/// Storage errors.
pub(super) fn children_of(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(TableSchema, String)>, Error> {
    let mut out = Vec::new();
    if !is_partitioned(table, engine, txn)? {
        return Ok(out);
    }
    let linked = links(engine, txn)?;
    let mut level = vec![(table.clone(), name.to_owned())];
    for _ in 0..MAX_DEPTH {
        let mut next = Vec::new();
        for (above, index) in &level {
            let Some(parent) = engine
                .list_indexes_as_of(txn, above.id)?
                .into_iter()
                .find(|d| d.name == *index)
            else {
                continue;
            };
            for partition in direct_partitions(above, engine, txn)? {
                for def in engine.list_indexes_as_of(txn, partition.id)? {
                    if equivalent(&def, &parent)
                        && linked
                            .iter()
                            .any(|(_, child, p)| *child == def.name && p == index)
                    {
                        next.push((partition.clone(), def.name));
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        out.extend(next.iter().cloned());
        level = next;
    }
    out.reverse();
    Ok(out)
}

/// The index `name` of `table` with the partition indexes linked to it, as `REINDEX INDEX` names
/// them: an index on a partitioned table stands for its partitions' indexes, which hold the
/// entries, so they are rebuilt with it.
///
/// # Errors
/// Storage errors.
pub(super) fn with_children(
    table: TableSchema,
    name: String,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(TableSchema, Option<String>)>, Error> {
    let mut out: Vec<(TableSchema, Option<String>)> = children_of(&table, &name, engine, txn)?
        .into_iter()
        .map(|(partition, child)| (partition, Some(child)))
        .collect();
    out.push((table, Some(name)));
    Ok(out)
}

/// `DROP INDEX name` on `table`: refuse dropping a partition's index that its partitioned
/// table's index requires (`2BP01`), and drop the partition indexes linked to it otherwise, before
/// the caller drops the index itself.
///
/// # Errors
/// `2BP01` for a linked partition index; storage errors.
pub(super) fn dropping(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    if let Some((parent_index, parent_table)) = live_parent(table, name, engine, txn)? {
        return Err(Error::Coded {
            message: format!(
                "cannot drop index \"{name}\" because index \"{parent_index}\" on partitioned \
                 table \"{}\" requires it; drop \"{parent_index}\" instead",
                parent_table.name
            ),
            sqlstate: "2BP01", // dependent_objects_still_exist
        });
    }
    let children = children_of(table, name, engine, txn)?;
    for (partition, child) in &children {
        if let Some(id) = engine.lookup_index(child)? {
            engine.drop_index(txn, id)?;
        }
        // A comment on the dropped index goes with it.
        super::comment::prune_table(engine, txn, &partition.schema, &partition.name)?;
    }
    let mut names: HashSet<&str> = children.iter().map(|(_, child)| child.as_str()).collect();
    names.insert(name);
    forget_all(engine, txn, &names)
}
