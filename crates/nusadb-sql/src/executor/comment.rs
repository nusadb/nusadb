//! `COMMENT ON`: persistence, lifecycle and read-back of object comments.
//!
//! Comments live in an engine-scoped SQL-layer catalog, [`COMMENT_CATALOG`], one row per commented
//! object, created lazily like the other `nusadb_*` catalogs. A row is five text columns
//! `(kind, schema, object, sub, description)`:
//!
//! | kind | schema | object | sub |
//! |---|---|---|---|
//! | `table` | table schema | table | (empty) |
//! | `column` | table schema | table | column |
//! | `constraint` | table schema | table | constraint |
//! | `index` | table schema | table | index |
//! | `trigger` | table schema | table | trigger |
//! | `view` | view schema | view | (empty) |
//! | `materialized view`, `function`, `procedure`, `type`, `domain` | `public` | name | (empty) |
//! | `schema` | schema | schema | (empty) |
//! | `sequence` | sequence schema | sequence | (empty) |
//!
//! An index on `(schema, object)` ([`OWNER_INDEX`]) finds the rows of one object, or of one table
//! and everything on it, without reading the whole catalog.
//!
//! # Concurrency
//!
//! Every write of a comment takes an exclusive key lock on the comment's identity, in a lock
//! namespace of its own ([`LOCK_SPACE`]), before it reads the existing row: two transactions
//! commenting on one object conflict (`40001`) instead of both inserting a row. A comment on a
//! table, or on something that belongs to a table, also takes a shared lock on the table, which
//! conflicts with the exclusive lock every `DROP TABLE` and `ALTER TABLE` holds until it ends. The
//! drop of any other object takes the same key lock as a comment on it, while it removes the
//! comment ([`forget`]). The existence check runs after the locks, against the latest committed
//! catalog (fresh reads), so a comment can never be written for an object another transaction has
//! dropped, or survive the drop of its object to attach to a later one of the same name.
//!
//! # Lifecycle
//!
//! Objects are keyed by name, the way every other SQL-layer catalog keys them:
//!
//! - a rename (`ALTER TABLE ... RENAME TO`, `RENAME COLUMN`, `ALTER TRIGGER ... RENAME TO`)
//!   re-keys the rows ([`rename_table`], [`rename_column`], [`rename_trigger`]);
//! - a drop removes them: each `DROP` removes the comment of what it dropped ([`forget`],
//!   [`forget_table`]), and an `ALTER TABLE` or `DROP INDEX` that removes something from a table
//!   re-checks that table's comments ([`prune_table`]).
//!
//! The catalog is ordinary MVCC data, so a comment written inside a transaction is invisible to
//! other sessions until it commits, rolls back with it, and survives a restart. It is read back
//! through the `information_schema.nusadb_comments` view ([`info_schema_rows`]).
#![allow(clippy::wildcard_imports)]

use std::hash::{Hash, Hasher};
use std::ops::Bound;

use super::*;
use crate::planner::{CommentObject, CommentPlan};

/// Engine-scoped catalog of object comments: `(kind, schema, object, sub, description)`.
pub(super) const COMMENT_CATALOG: &str = "nusadb_comments";

/// The index over the catalog's `(schema, object)` columns.
const OWNER_INDEX: &str = "nusadb_comments_owner";

/// The five-text-column schema of [`COMMENT_CATALOG`].
const COMMENT_CATALOG_SCHEMA: [ColumnType; 5] = [ColumnType::Text; 5];

/// The key-lock namespace comment writers and object drops meet in. No table is ever given this
/// id (ids are allocated upward from 1), so these locks share nothing with a real table's.
const LOCK_SPACE: nusadb_core::TableId = nusadb_core::TableId(u64::MAX - 101);

const TABLE: &str = "table";
const COLUMN: &str = "column";
const CONSTRAINT: &str = "constraint";
const INDEX: &str = "index";
const TRIGGER: &str = "trigger";
const VIEW: &str = "view";
const MATERIALIZED_VIEW: &str = "materialized view";
const FUNCTION: &str = "function";
const PROCEDURE: &str = "procedure";
const SCHEMA: &str = "schema";
const SEQUENCE: &str = "sequence";
const TYPE: &str = "type";
const DOMAIN: &str = "domain";

/// Prefix of a session's temporary schema (`nusadb_temp_<id>`).
const TEMP_SCHEMA_PREFIX: &str = "nusadb_temp_";

/// Whether a comment kind hangs off a table (its `object` column is the table's name).
fn table_scoped(kind: &str) -> bool {
    matches!(kind, TABLE | COLUMN | CONSTRAINT | INDEX | TRIGGER)
}

/// Whether `schema` is another session's temporary schema.
fn foreign_temp_schema(schema: &str) -> bool {
    schema.starts_with(TEMP_SCHEMA_PREFIX)
        && session_ctx::current_temp_schema().as_deref() != Some(schema)
}

/// The identity of one commented object, as stored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct CommentKey {
    kind: &'static str,
    schema: String,
    object: String,
    sub: String,
}

impl CommentKey {
    fn new(kind: &'static str, schema: &str, object: &str, sub: &str) -> Self {
        Self {
            kind,
            schema: schema.to_owned(),
            object: object.to_owned(),
            sub: sub.to_owned(),
        }
    }

    /// A key for an object with no schema of its own, filed under `public`.
    fn flat(kind: &'static str, name: &str) -> Self {
        Self::new(kind, nusadb_core::PUBLIC_SCHEMA, name, "")
    }

    /// A key for an object stored under a catalog key that folds a non-`public` schema in
    /// (`schema.name`, as views and sequences are).
    fn from_catalog_key(kind: &'static str, key: &str) -> Self {
        let (schema, name) = crate::analyzer::split_qualified(key);
        Self::new(kind, schema.unwrap_or(nusadb_core::PUBLIC_SCHEMA), name, "")
    }

    /// The key lock a writer of this comment, or the drop of its object, takes.
    fn lock(&self, engine: &dyn StorageEngine, txn: TxnId) -> Result<(), Error> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        engine.lock_key(
            txn,
            LOCK_SPACE,
            hasher.finish(),
            nusadb_core::engine::RowLockMode::Exclusive,
        )?;
        Ok(())
    }
}

/// The comment key of a view.
pub(super) fn view_key(catalog_key: &str) -> CommentKey {
    CommentKey::from_catalog_key(VIEW, catalog_key)
}

/// The comment key of a materialized view.
pub(super) fn matview_key(name: &str) -> CommentKey {
    CommentKey::flat(MATERIALIZED_VIEW, name)
}

/// The comment key of a function.
pub(super) fn function_key(name: &str) -> CommentKey {
    CommentKey::flat(FUNCTION, name)
}

/// The comment key of a procedure.
pub(super) fn procedure_key(name: &str) -> CommentKey {
    CommentKey::flat(PROCEDURE, name)
}

/// The comment key of an enum or composite type.
pub(super) fn type_key(name: &str) -> CommentKey {
    CommentKey::flat(TYPE, name)
}

/// The comment key of a domain.
pub(super) fn domain_key(name: &str) -> CommentKey {
    CommentKey::flat(DOMAIN, name)
}

/// The comment key of a sequence, from its catalog key.
pub(super) fn sequence_key(catalog_key: &str) -> CommentKey {
    CommentKey::from_catalog_key(SEQUENCE, catalog_key)
}

/// The comment key of a trigger.
pub(super) fn trigger_key(schema: &str, table: &str, name: &str) -> CommentKey {
    CommentKey::new(TRIGGER, schema, table, name)
}

/// The comment key of a constraint.
pub(super) fn constraint_key(schema: &str, table: &str, name: &str) -> CommentKey {
    CommentKey::new(CONSTRAINT, schema, table, name)
}

/// The comment key of a schema.
pub(super) fn schema_key(name: &str) -> CommentKey {
    CommentKey::new(SCHEMA, name, name, "")
}

/// One decoded catalog row.
struct StoredComment {
    tid: nusadb_core::engine::Tid,
    kind: String,
    schema: String,
    object: String,
    sub: String,
    description: String,
}

impl StoredComment {
    fn is(&self, key: &CommentKey) -> bool {
        self.kind == key.kind
            && self.schema == key.schema
            && self.object == key.object
            && self.sub == key.sub
    }

    fn decode(tid: nusadb_core::engine::Tid, bytes: &[u8]) -> Result<Option<Self>, Error> {
        let row = row::decode(bytes, &COMMENT_CATALOG_SCHEMA)?;
        Ok(match row.as_slice() {
            [
                ast::Value::Text(kind),
                ast::Value::Text(schema),
                ast::Value::Text(object),
                ast::Value::Text(sub),
                ast::Value::Text(description),
            ] => Some(Self {
                tid,
                kind: kind.clone(),
                schema: schema.clone(),
                object: object.clone(),
                sub: sub.clone(),
                description: description.clone(),
            }),
            _ => None,
        })
    }
}

/// The catalog as `txn` sees it: the table and, when present, its owner index.
struct Catalog {
    table: nusadb_core::TableId,
    index: Option<nusadb_core::IndexId>,
}

impl Catalog {
    /// The comment catalog, or `None` when it does not exist yet.
    fn find(engine: &dyn StorageEngine, txn: TxnId) -> Result<Option<Self>, Error> {
        let Some(table) = engine.lookup_table_as_of(txn, COMMENT_CATALOG)? else {
            return Ok(None);
        };
        let index = if engine
            .list_indexes_as_of(txn, table.id)?
            .iter()
            .any(|i| i.name == OWNER_INDEX)
        {
            engine.lookup_index(OWNER_INDEX)?
        } else {
            None
        };
        Ok(Some(Self {
            table: table.id,
            index,
        }))
    }

    /// The comment catalog, created (with its index) if it does not exist yet.
    fn ensure(engine: &dyn StorageEngine, txn: TxnId) -> Result<Self, Error> {
        if let Some(found) = Self::find(engine, txn)? {
            return Ok(found);
        }
        let columns = ["kind", "schema", "object", "sub", "description"]
            .into_iter()
            .map(|name| ColumnDef {
                name: name.to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            })
            .collect();
        let def = TableDef {
            schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
            name: COMMENT_CATALOG.to_owned(),
            columns,
        };
        let table = engine.create_table(txn, &def)?;
        let index_def = nusadb_core::engine::IndexDef {
            name: OWNER_INDEX.to_owned(),
            table,
            columns: vec!["schema".to_owned(), "object".to_owned()],
            key_exprs: Vec::new(),
            predicate: None,
            include: Vec::new(),
            kind: nusadb_core::engine::IndexKind::BTree,
            unique: false,
        };
        let index = match engine.create_index(txn, &index_def) {
            Ok(id) => Some(id),
            // An engine without secondary indexes (the in-memory test double) is read by a scan.
            Err(nusadb_core::Error::Io(e)) if e.kind() == std::io::ErrorKind::Unsupported => None,
            Err(e) => return Err(e.into()),
        };
        Ok(Self { table, index })
    }

    /// The rows filed under `(schema, object)`; every row when `owner` is `None`.
    fn rows(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        owner: Option<(&str, &str)>,
    ) -> Result<Vec<StoredComment>, Error> {
        let mut scan = match (self.index, owner) {
            (Some(index), Some((schema, object))) => {
                let key = owner_key(schema, object)?;
                engine.index_scan(
                    txn,
                    index,
                    Bound::Included(key.clone()),
                    Bound::Included(key),
                )?
            },
            _ => engine.scan(txn, self.table)?,
        };
        let mut out = Vec::new();
        while let Some((tid, bytes)) = scan.try_next()? {
            if let Some(row) = StoredComment::decode(tid, &bytes)?
                && owner.is_none_or(|(s, o)| row.schema == s && row.object == o)
            {
                out.push(row);
            }
        }
        Ok(out)
    }

    /// Insert one comment row, maintaining the owner index.
    fn insert(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        key: &CommentKey,
        description: &str,
    ) -> Result<(), Error> {
        let row = [key.kind, &key.schema, &key.object, &key.sub, description]
            .map(|s| ast::Value::Text(s.to_owned()));
        let tid = engine.insert(
            txn,
            self.table,
            &row::encode(&row, &COMMENT_CATALOG_SCHEMA)?,
        )?;
        if let Some(index) = self.index {
            engine.index_insert(txn, index, &owner_key(&key.schema, &key.object)?, tid)?;
        }
        Ok(())
    }
}

/// The owner-index key of `(schema, object)`.
fn owner_key(schema: &str, object: &str) -> Result<Vec<u8>, Error> {
    super::index_key::encode_index_key(&[
        ast::Value::Text(schema.to_owned()),
        ast::Value::Text(object.to_owned()),
    ])
}

/// Read the latest committed catalog (plus `txn`'s own writes) while alive, whatever the
/// isolation level: what a check made after taking a lock needs.
struct FreshReads<'e> {
    engine: &'e dyn StorageEngine,
    txn: TxnId,
}

impl<'e> FreshReads<'e> {
    fn begin(engine: &'e dyn StorageEngine, txn: TxnId) -> Result<Self, Error> {
        engine.begin_fresh_reads(txn)?;
        Ok(Self { engine, txn })
    }
}

impl Drop for FreshReads<'_> {
    fn drop(&mut self) {
        let _ = self.engine.end_fresh_reads(self.txn);
    }
}

/// `COMMENT ON <object> IS {'text' | NULL}`: lock the object, check it exists and that the
/// session may comment on it, then replace (or with `IS NULL` or `IS ''` remove) its comment.
pub(super) fn run_comment(
    plan: &CommentPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let key = resolve(&plan.target, engine, txn)?;
    let _fresh = FreshReads::begin(engine, txn)?;
    let text = plan.comment.as_deref().filter(|t| !t.is_empty());
    let catalog = match text {
        Some(_) => Some(Catalog::ensure(engine, txn)?),
        None => Catalog::find(engine, txn)?,
    };
    if let Some(catalog) = catalog {
        for row in catalog.rows(engine, txn, Some((&key.schema, &key.object)))? {
            if row.is(&key) {
                engine.delete(txn, catalog.table, row.tid)?;
            }
        }
        if let Some(text) = text {
            catalog.insert(engine, txn, &key, text)?;
        }
    }
    Ok(ExecutionResult::Commented)
}

/// Lock the comment's target, check that it exists for `txn` and that the session may comment on
/// it, and work out the key it is stored under.
#[allow(
    clippy::too_many_lines,
    reason = "one flat arm per comment target kind; length tracks the object taxonomy"
)]
fn resolve(
    target: &CommentObject,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<CommentKey, Error> {
    let session = crate::rbac::ResolvedSession::resolve(engine, txn, &session_ctx::current_user())?;
    // A table-scoped target: the table is held shared, so a concurrent DROP or ALTER of it
    // conflicts, then its existence is checked against the latest committed catalog.
    // The fresh-reads guard comes back with the table, so the caller's own existence checks also
    // read the latest committed catalog.
    let on_table = |schema: &str,
                    table: &str,
                    key: CommentKey|
     -> Result<(TableSchema, FreshReads<'_>), Error> {
        if foreign_temp_schema(schema) {
            return Err(Error::PermissionDenied(format!(
                "cannot comment on \"{table}\": it belongs to another session's temporary schema"
            )));
        }
        let resolved = require_table(engine, txn, schema, table)?;
        engine.lock_table(
            txn,
            resolved.id,
            nusadb_core::engine::TableLockMode::AccessShare,
        )?;
        key.lock(engine, txn)?;
        let fresh = FreshReads::begin(engine, txn)?;
        let resolved = require_table(engine, txn, schema, table)?;
        require_owner(
            engine,
            txn,
            &session,
            crate::ast::ObjectKind::Table,
            &format!("{schema}.{table}"),
        )?;
        Ok((resolved, fresh))
    };
    let key = match target {
        CommentObject::Table { schema, table } => {
            // A materialized view's backing table: its comment is the materialized view's, so the
            // object carries one comment whichever statement wrote it.
            let key = if schema == nusadb_core::PUBLIC_SCHEMA
                && load_view_def(engine, txn, MATVIEW_CATALOG, table)?.is_some()
            {
                matview_key(table)
            } else {
                CommentKey::new(TABLE, schema, table, "")
            };
            let _held = on_table(schema, table, key.clone())?;
            key
        },
        CommentObject::Column {
            schema,
            table,
            column,
        } => {
            let key = CommentKey::new(COLUMN, schema, table, column);
            let (resolved, _fresh) = on_table(schema, table, key.clone())?;
            if !resolved.columns.iter().any(|c| &c.name == column) {
                return Err(Error::ColumnNotFound {
                    table: table.clone(),
                    column: column.clone(),
                });
            }
            key
        },
        CommentObject::Constraint {
            schema,
            table,
            name,
        } => {
            let key = constraint_key(schema, table, name);
            let (resolved, _fresh) = on_table(schema, table, key.clone())?;
            if !constraint_names(engine, txn, resolved.id)?.contains(name) {
                return Err(Error::ObjectNotFound(format!(
                    "constraint \"{name}\" of relation \"{table}\" does not exist"
                )));
            }
            key
        },
        CommentObject::Trigger {
            schema,
            table,
            name,
        } => {
            let key = trigger_key(schema, table, name);
            let _held = on_table(schema, table, key.clone())?;
            if !super::trigger::trigger_exists(engine, txn, schema, table, name)? {
                return Err(Error::TriggerNotFound {
                    name: name.clone(),
                    table: table.clone(),
                });
            }
            key
        },
        CommentObject::Index { schema, name } => {
            let display = schema.as_deref().map_or_else(
                || name.clone(),
                |s| crate::analyzer::qualified_display(s, name),
            );
            let mut owners = index_owners(engine, txn, schema.as_deref(), name)?;
            if owners.len() > 1 {
                return Err(Error::InvalidStatement(format!(
                    "index name \"{name}\" is ambiguous; qualify it with its schema"
                )));
            }
            let Some((owner_schema, owner_table)) = owners.pop() else {
                return Err(Error::IndexNotFound { name: display });
            };
            let key = CommentKey::new(INDEX, &owner_schema, &owner_table, name);
            let (resolved, _fresh) = on_table(&owner_schema, &owner_table, key.clone())?;
            // Re-checked under the lock: a DROP INDEX that committed meanwhile took it away.
            let vector = list_vector_indexes(engine, txn)?;
            if !index_names(engine, txn, &resolved, &vector)?.contains(name) {
                return Err(Error::IndexNotFound { name: display });
            }
            key
        },
        CommentObject::Function { name, args } => {
            let key = function_key(name);
            key.lock(engine, txn)?;
            let _fresh = FreshReads::begin(engine, txn)?;
            let Some(def) = super::function::lookup_function_definition(engine, txn, name)? else {
                return Err(Error::FunctionNotFound {
                    name: signature(name, args.as_deref()),
                });
            };
            // A function name is unique, so the name alone identifies it; an argument list, when
            // given, has to describe that function, or the statement names one that does not exist.
            if args.as_ref().is_some_and(|a| a.len() != def.param_count) {
                return Err(Error::FunctionNotFound {
                    name: signature(name, args.as_deref()),
                });
            }
            require_superuser_owned(&session, "function", name)?;
            key
        },
        CommentObject::Procedure { name, args } => {
            let key = procedure_key(name);
            key.lock(engine, txn)?;
            let _fresh = FreshReads::begin(engine, txn)?;
            let Some((in_count, out_count)) = super::procedure::procedure_arity(engine, txn, name)?
            else {
                return Err(Error::ProcedureNotFound {
                    name: signature(name, args.as_deref()),
                });
            };
            if args
                .as_ref()
                .is_some_and(|a| a.len() != in_count && a.len() != in_count + out_count)
            {
                return Err(Error::ProcedureNotFound {
                    name: signature(name, args.as_deref()),
                });
            }
            require_superuser_owned(&session, "procedure", name)?;
            key
        },
        CommentObject::View { key: view } => {
            let key = view_key(view);
            key.lock(engine, txn)?;
            let _fresh = FreshReads::begin(engine, txn)?;
            if lookup_view_definition(engine, txn, view)?.is_none() {
                if load_view_def(engine, txn, MATVIEW_CATALOG, view)?.is_some() {
                    return Err(Error::WrongObjectType(format!(
                        "\"{view}\" is a materialized view; use COMMENT ON MATERIALIZED VIEW"
                    )));
                }
                if engine.lookup_table_as_of(txn, view)?.is_some() {
                    return Err(Error::WrongObjectType(format!("\"{view}\" is not a view")));
                }
                return Err(Error::TableNotFound { name: view.clone() });
            }
            // A plain view records no owner, so it reads as the bootstrap superuser's.
            require_owner(
                engine,
                txn,
                &session,
                crate::ast::ObjectKind::Table,
                &format!("{}.{}", key.schema, key.object),
            )?;
            key
        },
        CommentObject::MaterializedView { name } => {
            let key = matview_key(name);
            if load_view_def(engine, txn, MATVIEW_CATALOG, name)?.is_none() {
                if lookup_view_definition(engine, txn, name)?.is_some()
                    || engine.lookup_table_as_of(txn, name)?.is_some()
                {
                    return Err(Error::WrongObjectType(format!(
                        "\"{name}\" is not a materialized view"
                    )));
                }
                return Err(Error::TableNotFound { name: name.clone() });
            }
            let _held = on_table(nusadb_core::PUBLIC_SCHEMA, name, key.clone())?;
            key
        },
        CommentObject::Schema { name } => {
            let key = schema_key(name);
            key.lock(engine, txn)?;
            if foreign_temp_schema(name) {
                return Err(Error::PermissionDenied(format!(
                    "cannot comment on schema \"{name}\": it is another session's temporary schema"
                )));
            }
            // `public` always exists, whether or not the engine lists it as a namespace.
            if name != nusadb_core::PUBLIC_SCHEMA && engine.lookup_schema(name)?.is_none() {
                return Err(Error::SchemaNotFound { name: name.clone() });
            }
            require_owner(engine, txn, &session, crate::ast::ObjectKind::Schema, name)?;
            key
        },
        CommentObject::Sequence { key: seq } => {
            let key = sequence_key(seq);
            key.lock(engine, txn)?;
            if engine.lookup_sequence(seq)?.is_none() {
                return Err(Error::SequenceNotFound { name: seq.clone() });
            }
            // A sequence records no owner, so it reads as the bootstrap superuser's.
            require_owner(engine, txn, &session, crate::ast::ObjectKind::Sequence, seq)?;
            key
        },
        CommentObject::Type { name } => {
            // A domain is a type as well; its comment is kept under the domain, whichever
            // statement wrote it, so both spellings read and clear the same row.
            type_key(name).lock(engine, txn)?;
            domain_key(name).lock(engine, txn)?;
            let _fresh = FreshReads::begin(engine, txn)?;
            let key = if lookup_enum(engine, txn, name)?.is_some()
                || composite_defined(engine, txn, name)?
            {
                type_key(name)
            } else if lookup_domain(engine, txn, name)?.is_some() {
                domain_key(name)
            } else {
                return Err(Error::ObjectNotFound(format!(
                    "type {name:?} does not exist"
                )));
            };
            require_superuser_owned(&session, "type", name)?;
            key
        },
        CommentObject::Domain { name } => {
            let key = domain_key(name);
            key.lock(engine, txn)?;
            let _fresh = FreshReads::begin(engine, txn)?;
            if lookup_domain(engine, txn, name)?.is_none() {
                if lookup_enum(engine, txn, name)?.is_some()
                    || composite_defined(engine, txn, name)?
                {
                    return Err(Error::WrongObjectType(format!(
                        "\"{name}\" is not a domain"
                    )));
                }
                return Err(Error::ObjectNotFound(format!(
                    "domain {name:?} does not exist"
                )));
            }
            require_superuser_owned(&session, "domain", name)?;
            key
        },
    };
    Ok(key)
}

/// The session must own the object (`kind`, `object` as the ownership catalog keys it), or be a
/// superuser: the rule `DROP` applies to the objects that record an owner. An object with no
/// recorded owner reads as the bootstrap superuser's.
fn require_owner(
    engine: &dyn StorageEngine,
    txn: TxnId,
    session: &crate::rbac::ResolvedSession,
    kind: crate::ast::ObjectKind,
    object: &str,
) -> Result<(), Error> {
    if session.is_superuser() || session.owns_object(engine, txn, kind, object)? {
        return Ok(());
    }
    Err(Error::PermissionDenied(format!(
        "must be the owner of {} `{object}` to comment on it",
        kind.as_str().to_lowercase()
    )))
}

/// For the objects that record no owner at all (routines, types, domains): only a superuser.
fn require_superuser_owned(
    session: &crate::rbac::ResolvedSession,
    what: &str,
    name: &str,
) -> Result<(), Error> {
    if session.is_superuser() {
        return Ok(());
    }
    Err(Error::PermissionDenied(format!(
        "must be a superuser to comment on {what} `{name}` (it records no owner)"
    )))
}

/// `name(arg, ...)` for an error message, or the bare name when no argument list was given.
fn signature(name: &str, args: Option<&[String]>) -> String {
    args.map_or_else(
        || name.to_owned(),
        |args| format!("{name}({})", args.join(", ")),
    )
}

/// The table `schema.table` as `txn` sees it, or `42P01`.
fn require_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
) -> Result<TableSchema, Error> {
    engine
        .lookup_table_as_of_in(txn, schema, table)?
        .ok_or_else(|| Error::TableNotFound {
            name: crate::analyzer::qualified_display(schema, table),
        })
}

/// The user-visible constraint names of table `id` for `txn`: its keys, checks and foreign keys.
/// The synthetic checks that enforce a declared type's bounds are not user constraints.
fn constraint_names(
    engine: &dyn StorageEngine,
    txn: TxnId,
    id: nusadb_core::TableId,
) -> Result<HashSet<String>, Error> {
    let mut names: HashSet<String> = engine
        .list_constraints_as_of(txn, id)?
        .into_iter()
        .map(|c| c.name)
        .filter(|n| !n.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX))
        .collect();
    names.extend(
        engine
            .list_foreign_keys_as_of(txn, id)?
            .into_iter()
            .map(|fk| fk.name),
    );
    Ok(names)
}

/// The index names of `table` for `txn`, B-tree indexes and `USING hnsw` vector indexes both.
fn index_names(
    engine: &dyn StorageEngine,
    txn: TxnId,
    table: &TableSchema,
    vector: &[VectorIndexListing],
) -> Result<HashSet<String>, Error> {
    let mut names: HashSet<String> = engine
        .list_indexes_as_of(txn, table.id)?
        .into_iter()
        .map(|i| i.name)
        .collect();
    // The vector-index catalog records the bare table name of a table in the default namespace.
    if table.schema == nusadb_core::PUBLIC_SCHEMA {
        names.extend(
            vector
                .iter()
                .filter(|v| v.table == table.name)
                .map(|v| v.name.clone()),
        );
    }
    Ok(names)
}

/// Every `(schema, table)` with an index named `name` that `txn` can see, restricted to `schema`
/// when one is given.
pub(super) fn index_owners(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: Option<&str>,
    name: &str,
) -> Result<Vec<(String, String)>, Error> {
    let vector = list_vector_indexes(engine, txn)?;
    let mut owners = Vec::new();
    for (table_schema, table_name) in engine.list_tables_qualified_as_of(txn)? {
        if table_name.starts_with(crate::SYSTEM_TABLE_PREFIX)
            || schema.is_some_and(|s| s != table_schema)
        {
            continue;
        }
        let Some(table) = engine.lookup_table_as_of_in(txn, &table_schema, &table_name)? else {
            continue;
        };
        if index_names(engine, txn, &table, &vector)?.contains(name) {
            owners.push((table_schema, table_name));
        }
    }
    Ok(owners)
}

/// The drop of an object: remove its comment, under the same key lock a comment on it takes, so
/// a concurrent `COMMENT ON` of the object conflicts instead of outliving it. Called whether or not
/// the object had a comment.
///
/// # Errors
/// [`nusadb_core::Error::SerializationConflict`] when another transaction is commenting on the
/// object; propagates storage errors.
pub(super) fn forget(
    engine: &dyn StorageEngine,
    txn: TxnId,
    key: &CommentKey,
) -> Result<(), Error> {
    key.lock(engine, txn)?;
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(());
    };
    let _fresh = FreshReads::begin(engine, txn)?;
    for row in catalog.rows(engine, txn, Some((&key.schema, &key.object)))? {
        if row.is(key) {
            engine.delete(txn, catalog.table, row.tid)?;
        }
    }
    Ok(())
}

/// The drop of a table: remove its comment and those on its columns, constraints, indexes and
/// triggers. The table is held exclusively by the dropping transaction, which every comment on it
/// conflicts with.
pub(super) fn forget_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
) -> Result<(), Error> {
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(());
    };
    let _fresh = FreshReads::begin(engine, txn)?;
    for row in catalog.rows(engine, txn, Some((schema, table)))? {
        if table_scoped(&row.kind) || row.kind == MATERIALIZED_VIEW {
            engine.delete(txn, catalog.table, row.tid)?;
        }
    }
    Ok(())
}

/// After `schema.table` lost something (a column, a constraint, an index): remove the comments of
/// what is no longer there. The caller holds the table exclusively.
pub(super) fn prune_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
) -> Result<(), Error> {
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(());
    };
    let rows: Vec<StoredComment> = catalog
        .rows(engine, txn, Some((schema, table)))?
        .into_iter()
        .filter(|r| table_scoped(&r.kind))
        .collect();
    if rows.is_empty() {
        return Ok(());
    }
    let Some(resolved) = engine.lookup_table_as_of_in(txn, schema, table)? else {
        return forget_table(engine, txn, schema, table);
    };
    let columns: HashSet<&str> = resolved.columns.iter().map(|c| c.name.as_str()).collect();
    let constraints = constraint_names(engine, txn, resolved.id)?;
    let indexes = index_names(engine, txn, &resolved, &list_vector_indexes(engine, txn)?)?;
    for row in rows {
        let exists = match row.kind.as_str() {
            COLUMN => columns.contains(row.sub.as_str()),
            CONSTRAINT => constraints.contains(&row.sub),
            INDEX => indexes.contains(&row.sub),
            TRIGGER => super::trigger::trigger_exists(engine, txn, schema, table, &row.sub)?,
            _ => true,
        };
        if !exists {
            engine.delete(txn, catalog.table, row.tid)?;
        }
    }
    Ok(())
}

/// The teardown of a schema (`DROP SCHEMA`, a session's temporary schema): remove the comments of
/// the schema and of every table that was in it. A rare statement, so it reads the whole catalog.
pub(super) fn forget_schema(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
) -> Result<(), Error> {
    forget(engine, txn, &schema_key(schema))?;
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(());
    };
    for row in catalog.rows(engine, txn, None)? {
        if row.schema == schema && table_scoped(&row.kind) {
            engine.delete(txn, catalog.table, row.tid)?;
        }
    }
    Ok(())
}

/// Re-file the rows of `(schema, object)` that `selected` picks, changed by `change`. A rewritten
/// row is inserted afresh, so the owner index follows it.
fn refile(
    engine: &dyn StorageEngine,
    txn: TxnId,
    owner: (&str, &str),
    selected: impl Fn(&StoredComment) -> bool,
    change: impl Fn(&mut CommentKey),
) -> Result<(), Error> {
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(());
    };
    for row in catalog.rows(engine, txn, Some(owner))? {
        if !selected(&row) {
            continue;
        }
        let Some(kind) = known_kind(&row.kind) else {
            continue;
        };
        let mut key = CommentKey::new(kind, &row.schema, &row.object, &row.sub);
        change(&mut key);
        engine.delete(txn, catalog.table, row.tid)?;
        catalog.insert(engine, txn, &key, &row.description)?;
    }
    Ok(())
}

/// The stored kind string, mapped back onto the static constant.
fn known_kind(kind: &str) -> Option<&'static str> {
    [
        TABLE,
        COLUMN,
        CONSTRAINT,
        INDEX,
        TRIGGER,
        VIEW,
        MATERIALIZED_VIEW,
        FUNCTION,
        PROCEDURE,
        SCHEMA,
        SEQUENCE,
        TYPE,
        DOMAIN,
    ]
    .into_iter()
    .find(|k| *k == kind)
}

/// `ALTER TABLE schema.old RENAME TO new`: the table's comment and those on its columns,
/// constraints, indexes and triggers travel with it.
pub(super) fn rename_table(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    old: &str,
    new: &str,
) -> Result<(), Error> {
    refile(
        engine,
        txn,
        (schema, old),
        |row| table_scoped(&row.kind),
        |key| new.clone_into(&mut key.object),
    )
}

/// `ALTER TABLE ... RENAME COLUMN old TO new`: the column's comment follows the new name.
pub(super) fn rename_column(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
    old: &str,
    new: &str,
) -> Result<(), Error> {
    refile(
        engine,
        txn,
        (schema, table),
        |row| row.kind == COLUMN && row.sub == old,
        |key| new.clone_into(&mut key.sub),
    )
}

/// `ALTER TRIGGER old ON table RENAME TO new`: the trigger's comment follows the new name.
pub(super) fn rename_trigger(
    engine: &dyn StorageEngine,
    txn: TxnId,
    schema: &str,
    table: &str,
    old: &str,
    new: &str,
) -> Result<(), Error> {
    refile(
        engine,
        txn,
        (schema, table),
        |row| row.kind == TRIGGER && row.sub == old,
        |key| new.clone_into(&mut key.sub),
    )
}

/// `information_schema.nusadb_comments`: one row per comment the session may see,
/// `(object_type, object_schema, table_name, object_name, description)`. `table_name` is the table
/// a column, constraint, index or trigger belongs to (and the table itself for a table comment),
/// `NULL` for the other kinds; `object_name` is the commented object's own name (the column's, for a
/// column comment).
///
/// A comment on something in another session's temporary schema is never shown. A comment on a
/// table, or on anything that belongs to one, is shown only to a session with some relationship to
/// the table (a superuser, its owner, or a holder of any privilege on it), the rule that guards a
/// table's column list.
///
/// # Errors
/// Propagates storage and decode errors.
pub(super) fn info_schema_rows(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<Row>, Error> {
    let Some(catalog) = Catalog::find(engine, txn)? else {
        return Ok(Vec::new());
    };
    let mut rows = catalog.rows(engine, txn, None)?;
    rows.retain(|r| !foreign_temp_schema(&r.schema));
    let session = crate::rbac::ResolvedSession::resolve(engine, txn, &session_ctx::current_user())?;
    if !session.is_superuser() {
        let grants = crate::rbac::all_grants(engine, txn)?;
        let mut visible: HashMap<(String, String), bool> = HashMap::new();
        let mut keep = Vec::with_capacity(rows.len());
        for row in rows {
            if !(table_scoped(&row.kind) || row.kind == MATERIALIZED_VIEW) {
                keep.push(row);
                continue;
            }
            let owner = (row.schema.clone(), row.object.clone());
            let shown = if let Some(shown) = visible.get(&owner) {
                *shown
            } else {
                let object = format!("{}.{}", row.schema, row.object);
                let kind = crate::ast::ObjectKind::Table;
                let owner_name = crate::rbac::object_owner(engine, txn, kind, &object)?;
                let mut shown = false;
                for privilege in crate::ast::Privilege::all_for(kind) {
                    if session.decide(&owner_name, &grants, kind, &object, privilege, false)
                        || (crate::rbac::is_builtin_public_grant(kind, &object, privilege)
                            && !crate::rbac::builtin_grant_revoked(
                                engine, txn, kind, &object, privilege,
                            )?)
                    {
                        shown = true;
                        break;
                    }
                }
                visible.insert(owner, shown);
                shown
            };
            if shown {
                keep.push(row);
            }
        }
        rows = keep;
    }
    rows.sort_by(|a, b| {
        (&a.kind, &a.schema, &a.object, &a.sub).cmp(&(&b.kind, &b.schema, &b.object, &b.sub))
    });
    Ok(rows
        .into_iter()
        .map(|row| {
            let scoped = table_scoped(&row.kind);
            let object_name = if scoped && row.kind != TABLE {
                row.sub
            } else {
                row.object.clone()
            };
            vec![
                ast::Value::Text(row.kind.to_uppercase()),
                ast::Value::Text(row.schema),
                if scoped {
                    ast::Value::Text(row.object)
                } else {
                    ast::Value::Null
                },
                ast::Value::Text(object_name),
                ast::Value::Text(row.description),
            ]
        })
        .collect())
}
