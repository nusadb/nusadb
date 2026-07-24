//! DDL execution: CREATE/DROP TABLE, SCHEMA, SEQUENCE, INDEX, ALTER TABLE, ANALYZE.
//!
//! Split verbatim out of `executor/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === Catalog introspection =======================================

/// `SHOW TABLES` — one row per visible table, in a single `table` column (sorted by the engine).
///
/// Enumerates under the statement's transaction snapshot so a table created by an earlier statement
/// on the same connection is reliably listed (the non-transactional `list_tables` can lag a
/// just-committed write).
pub(super) fn run_show_tables(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let rows = engine
        .list_tables_as_of(txn)?
        .into_iter()
        .map(|name| vec![ast::Value::Text(name)])
        .collect();
    Ok(ExecutionResult::Rows {
        columns: vec!["table".to_owned()],
        rows,
    })
}

/// `SHOW COLUMNS FROM t` — one row per column: `(column, type, nullable)`.
pub(super) fn run_show_columns(schema: &TableSchema) -> ExecutionResult {
    let rows = schema
        .columns
        .iter()
        .map(|col| {
            vec![
                ast::Value::Text(col.name.clone()),
                ast::Value::Text(type_name(col.ty)),
                ast::Value::Bool(col.nullable),
            ]
        })
        .collect();
    ExecutionResult::Rows {
        columns: vec![
            "column".to_owned(),
            "type".to_owned(),
            "nullable".to_owned(),
        ],
        rows,
    }
}

/// Render a [`ColumnType`] as its SQL type name (for `SHOW COLUMNS` and `information_schema`).
pub(super) fn type_name(ty: ColumnType) -> String {
    match ty {
        ColumnType::Bool => "BOOLEAN".to_owned(),
        ColumnType::Int => "INT".to_owned(),
        ColumnType::SmallInt => "SMALLINT".to_owned(),
        ColumnType::BigInt => "BIGINT".to_owned(),
        ColumnType::Float => "FLOAT".to_owned(),
        ColumnType::Real => "REAL".to_owned(),
        ColumnType::Text => "TEXT".to_owned(),
        ColumnType::VarChar(n) => format!("VARCHAR({n})"),
        ColumnType::Char(n) => format!("CHAR({n})"),
        ColumnType::Bytes => "BYTES".to_owned(),
        ColumnType::Timestamp => "TIMESTAMP".to_owned(),
        ColumnType::Date => "DATE".to_owned(),
        ColumnType::Time => "TIME".to_owned(),
        ColumnType::TimestampTz => "TIMESTAMPTZ".to_owned(),
        ColumnType::TimeTz => "TIMETZ".to_owned(),
        ColumnType::Uuid => "UUID".to_owned(),
        ColumnType::Numeric { precision: 0, .. } => "NUMERIC".to_owned(),
        ColumnType::Numeric { precision, scale } => format!("NUMERIC({precision},{scale})"),
        ColumnType::Json => "JSON".to_owned(),
        ColumnType::Jsonb => "JSONB".to_owned(),
        ColumnType::Interval => "INTERVAL".to_owned(),
        ColumnType::Array(elem) => format!("{}[]", type_name(elem.column_type())),
        ColumnType::Vector(dim) => format!("VECTOR({dim})"),
    }
}

// === DDL ==================================================================

pub(super) fn run_create_table(
    plan: CreateTablePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if let Some(existing) = engine.lookup_table_as_of_in(txn, &plan.schema, &plan.table)? {
        if plan.if_not_exists {
            return Ok(ExecutionResult::Created(existing.id));
        }
        return Err(Error::TableExists {
            name: crate::analyzer::qualified_display(&plan.schema, &plan.table),
        });
    }
    // Resolve any deferred user-defined column type (B-ENUM): it must name a registered ENUM, which
    // is stored as its `TEXT` placeholder. An unresolved name is a loud error (caught here, not at
    // parse time, because only the executor can read the type catalog).
    let mut columns = Vec::with_capacity(plan.columns.len());
    for c in plan.columns {
        if let Some(udt) = &c.udt_name
            && super::lookup_enum(engine, txn, udt)?.is_none()
        {
            return Err(Error::Unsupported(format!("type \"{udt}\" does not exist")));
        }
        columns.push(ColumnDef {
            name: c.name,
            ty: c.ty,
            nullable: c.nullable,
        });
    }
    let def = TableDef {
        schema: plan.schema,
        name: plan.table,
        columns,
    };
    let id = engine.create_table(txn, &def)?;
    // Register each resolved PRIMARY KEY / UNIQUE constraint so INSERT/UPDATE enforce its
    // uniqueness (the analyzer collected them from column-level + table-level declarations).
    for constraint in &plan.unique_constraints {
        engine.add_unique_constraint(
            txn,
            id,
            &constraint.name,
            &constraint.columns,
            constraint.primary,
        )?;
    }
    // Register each FOREIGN KEY.
    for fk in &plan.foreign_keys {
        register_foreign_key(id, fk, engine, txn)?;
    }
    // Register each CHECK constraint: the canonical predicate SQL is persisted opaquely so
    // INSERT/UPDATE/COPY can re-parse and evaluate it per row.
    for chk in &plan.check_constraints {
        engine.add_check_constraint(txn, id, &chk.name, chk.predicate_sql.as_bytes())?;
    }
    // Create the backing sequence for each SERIAL column before persisting its sentinel
    // default, so INSERT's `lookup_sequence` resolves.
    for (_, sql) in &plan.defaults {
        if let Some(seq) = super::coldefault::serial_sequence(sql) {
            engine.create_sequence(
                txn,
                &nusadb_core::engine::SequenceDef {
                    name: seq.to_owned(),
                    start: 1,
                    increment: 1,
                    min_value: 1,
                    max_value: i64::MAX,
                    cycle: false,
                },
            )?;
        }
    }
    // Persist column DEFAULTs / SERIAL sentinels in the SQL-layer catalog so INSERT can fill
    // an omitted column.
    super::coldefault::store_defaults(
        &super::coldefault::catalog_key(&def.schema, &def.name),
        &plan.defaults,
        engine,
        txn,
    )?;
    Ok(ExecutionResult::Created(id))
}

/// Register one `FOREIGN KEY` on child table `child_id` (shared by CREATE TABLE and
/// ALTER TABLE ADD CONSTRAINT). Resolves the parent table against the live catalog and declares the
/// constraint — the engine validates the parent has a `PRIMARY KEY`. v1 references the parent's
/// `PRIMARY KEY` only: an explicit `REFERENCES parent(cols)` that is not exactly the parent's PK, or
/// any arity mismatch, is rejected (silently redirecting to the PK would mis-enforce). This does NOT
/// validate existing child rows — `ALTER TABLE ADD` does that separately.
pub(super) fn register_foreign_key(
    child_id: nusadb_core::TableId,
    fk: &crate::planner::ForeignKeySpec,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let parent = engine
        .lookup_table_as_of(txn, &fk.parent_table)?
        .ok_or_else(|| Error::TableNotFound {
            name: fk.parent_table.clone(),
        })?;
    let parent_constraints = engine.list_constraints(parent.id)?;
    // The referenced parent columns: an explicit `REFERENCES parent (cols)` list, else the parent's
    // PRIMARY KEY (the unqualified `REFERENCES parent` form).
    let parent_columns: Vec<String> = if fk.referred_columns.is_empty() {
        parent_constraints
            .iter()
            .find(|c| matches!(c.kind, nusadb_core::ConstraintKind::PrimaryKey))
            .map(|c| c.columns.clone())
            .unwrap_or_default()
    } else {
        fk.referred_columns.clone()
    };
    if parent_columns.is_empty() {
        return Err(Error::Unsupported(format!(
            "foreign key \"{}\" references \"{}\", which has no PRIMARY KEY — name the referenced \
             UNIQUE columns explicitly with REFERENCES \"{}\" (columns)",
            fk.name, fk.parent_table, fk.parent_table
        )));
    }
    // The referenced columns must form a PRIMARY KEY or UNIQUE constraint on the parent (a FK may
    // reference a non-PK UNIQUE key, not only the PRIMARY KEY).
    let references_unique_key = parent_constraints.iter().any(|c| {
        matches!(
            c.kind,
            nusadb_core::ConstraintKind::PrimaryKey | nusadb_core::ConstraintKind::Unique
        ) && c.columns == parent_columns
    });
    if !references_unique_key {
        return Err(Error::Unsupported(format!(
            "foreign key \"{}\" references columns of \"{}\" that are not a PRIMARY KEY or UNIQUE \
             constraint",
            fk.name, fk.parent_table
        )));
    }
    if fk.columns.len() != parent_columns.len() {
        return Err(Error::Unsupported(format!(
            "foreign key \"{}\" column count does not match the referenced key of \"{}\"",
            fk.name, fk.parent_table
        )));
    }
    engine.add_foreign_key(
        txn,
        &nusadb_core::ForeignKeyDef {
            name: fk.name.clone(),
            child_table: child_id,
            child_columns: fk.columns.clone(),
            parent_table: parent.id,
            parent_columns,
            on_delete: fk.on_delete,
            on_update: fk.on_update,
        },
    )?;
    Ok(())
}

pub(super) fn run_drop_table(
    plan: &DropTablePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    match engine.lookup_table_as_of_in(txn, &plan.schema, &plan.table)? {
        Some(schema) => {
            // RESTRICT (A-UR.01b): refuse to drop a table that another table's FOREIGN KEY references,
            // so the FK is not left silently dangling (standard SQL rejects this without CASCADE). A
            // self-referencing FK (child == parent) does not block — it drops with the table.
            // CASCADE drops those referencing constraints instead — the standard's
            // semantics: the dependent CONSTRAINT goes, never the child table or its rows.
            let table_fks = engine.list_foreign_keys(schema.id)?;
            for fk in &table_fks {
                if fk.parent_table == schema.id && fk.child_table != schema.id {
                    if plan.cascade {
                        engine.drop_constraint(txn, fk.child_table, &fk.name)?;
                        continue;
                    }
                    return Err(Error::Unsupported(format!(
                        "cannot drop table \"{}\": foreign key \"{}\" on another table depends on it",
                        plan.table, fk.name
                    )));
                }
            }
            // Drop the table's own foreign keys first (those it declares, including a self-referencing
            // one). This frees their child-side index and, crucially, removes the FK record before the
            // constraint loop below drops the PRIMARY KEY / UNIQUE it references — otherwise the
            // DROP CONSTRAINT FK-RESTRICT guard would (wrongly) block this table's own teardown.
            for fk in &table_fks {
                if fk.child_table == schema.id {
                    engine.drop_constraint(txn, schema.id, &fk.name)?;
                }
            }
            // Drop the table's indexes and constraints so the global index/constraint namespace is
            // freed (A-UR.01): otherwise a later same-named table fails to recreate its PRIMARY KEY
            // ("index `<t>_pkey` already exists"), breaking idempotent migrations / redeploys. A
            // PRIMARY KEY/UNIQUE/FK constraint's drop also drops its backing index, so only the
            // *secondary* (non-backing) indexes are dropped directly here — avoiding a double drop.
            let constraints = engine.list_constraints(schema.id)?;
            let backing: std::collections::HashSet<_> =
                constraints.iter().filter_map(|c| c.index).collect();
            for def in engine.list_indexes(schema.id)? {
                if let Some(id) = engine.lookup_index(&def.name)?
                    && !backing.contains(&id)
                {
                    engine.drop_index(txn, id)?;
                }
            }
            for constraint in &constraints {
                engine.drop_constraint(txn, schema.id, &constraint.name)?;
            }
            engine.drop_table(txn, schema.id)?;
            // Drop each SERIAL column's backing sequence, then the table's column DEFAULTs,
            // so a later same-named table starts clean. The default catalog is keyed by the
            // schema-qualified name so a non-public table's defaults are isolated.
            let default_key = super::coldefault::catalog_key(&plan.schema, &plan.table);
            for (_, sql) in super::coldefault::load_defaults(&default_key, engine, txn)? {
                if let Some(seq) = super::coldefault::serial_sequence(&sql)
                    && let Some(id) = engine.lookup_sequence(seq)?
                {
                    engine.drop_sequence(txn, id)?;
                }
            }
            super::coldefault::delete_defaults_for_table(&default_key, engine, txn)?;
            // Drop any `USING hnsw` vector index declared on the table (A-UR.01c), which
            // lives in the SQL-layer catalog rather than the engine's index namespace.
            super::delete_vector_indexes_for_table(engine, txn, &plan.table)?;
            // Cascade-drop the table's row-level-security policies and its RLS-enabled marker (
            // ): otherwise they orphan the catalog, and a later same-named table cannot
            // re-create a policy of the same name ("policy already exists").
            super::delete_policies_for_table(engine, txn, &plan.table)?;
            super::set_table_rls(engine, txn, &plan.table, false)?;
        },
        None => {
            if !plan.if_exists {
                return Err(Error::TableNotFound {
                    name: crate::analyzer::qualified_display(&plan.schema, &plan.table),
                });
            }
        },
    }
    Ok(ExecutionResult::Dropped)
}

// === CREATE / DROP SCHEMA =========================================

pub(super) fn run_create_schema(
    plan: &CreateSchemaPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // `IF NOT EXISTS` makes a pre-existing schema a no-op; otherwise the engine rejects a duplicate.
    if plan.if_not_exists && engine.lookup_schema(&plan.name)?.is_some() {
        return Ok(ExecutionResult::SchemaCreated);
    }
    engine.create_schema(txn, &plan.name)?;
    Ok(ExecutionResult::SchemaCreated)
}

pub(super) fn run_drop_schema(
    plan: &DropSchemaPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    match engine.lookup_schema(&plan.name)? {
        // RESTRICT (default) refuses a non-empty schema; CASCADE drops its member tables too.
        Some(id) => engine.drop_schema(txn, id, plan.cascade)?,
        None => {
            if !plan.if_exists {
                return Err(Error::SchemaNotFound {
                    name: plan.name.clone(),
                });
            }
        },
    }
    Ok(ExecutionResult::SchemaDropped)
}

// === CREATE / DROP SEQUENCE =======================================

pub(super) fn run_create_sequence(
    plan: &CreateSequencePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if plan.if_not_exists && engine.lookup_sequence(&plan.def.name)?.is_some() {
        return Ok(ExecutionResult::SequenceCreated);
    }
    engine.create_sequence(txn, &plan.def)?;
    Ok(ExecutionResult::SequenceCreated)
}

pub(super) fn run_drop_sequence(
    plan: &DropSequencePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    match engine.lookup_sequence(&plan.name)? {
        Some(id) => engine.drop_sequence(txn, id)?,
        None => {
            if !plan.if_exists {
                return Err(Error::SequenceNotFound {
                    name: plan.name.clone(),
                });
            }
        },
    }
    Ok(ExecutionResult::SequenceDropped)
}

// === CREATE / DROP INDEX ==================================

/// Register the index in the catalog, then **backfill** the rows already present so the index is
/// complete the moment it exists. The SQL layer owns the key encoding (shared with the
/// index-scan executor), so we scan the table and insert one entry per visible row; subsequent
/// `INSERT`/`UPDATE`/`DELETE` keep it in sync (see `executor::dml`).
pub(super) fn run_create_index(
    plan: &CreateIndexPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    // A `USING hnsw` vector index is recorded in the SQL-layer vector-index catalog rather
    // than created as an engine B-tree index; its graph is built on demand at query time.
    if let Some(spec) = &plan.vector {
        if plan.if_not_exists && super::vector_index_exists(engine, txn, &spec.name)? {
            return Ok(ExecutionResult::IndexCreated);
        }
        super::store_vector_index(engine, txn, spec)?;
        return Ok(ExecutionResult::IndexCreated);
    }
    if plan.if_not_exists && engine.lookup_index(&plan.def.name)?.is_some() {
        return Ok(ExecutionResult::IndexCreated);
    }
    engine.create_index(txn, &plan.def)?;
    // Backfill existing rows so an index created on a populated table is not missing them (which
    // would make a later index scan return wrong results).
    if let (Some(id), Some(table)) = (
        engine.lookup_index(&plan.def.name)?,
        dml::schema_by_id(engine, plan.def.table)?,
    ) && let Some(target) = dml::build_index_target(id, &table, &plan.def)
    {
        // Backfill through the same maintenance path DML writes take (a functional/expression key
        // is evaluated and a partial predicate skips non-matching rows exactly as on later inserts),
        // streaming the table and applying key-sorted chunks so building the index drives sequential
        // index writes without materializing the whole table's rows or entries at once.
        dml::backfill_index_streaming(&target, &table, engine, txn)?;
    }
    Ok(ExecutionResult::IndexCreated)
}

pub(super) fn run_drop_index(
    plan: &DropIndexPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if let Some(id) = engine.lookup_index(&plan.name)? {
        engine.drop_index(txn, id)?;
    } else {
        // Not an engine B-tree index — it may be a `USING hnsw` vector index, recorded in
        // the SQL-layer catalog. Drop that if present; otherwise it truly does not exist.
        let dropped = super::delete_vector_index(engine, txn, &plan.name)?;
        if !dropped && !plan.if_exists {
            return Err(Error::IndexNotFound {
                name: plan.name.clone(),
            });
        }
    }
    Ok(ExecutionResult::IndexDropped)
}

// === ALTER TABLE ==========================================================

/// Apply one `ALTER TABLE` action: rewrite the stored rows when the physical
/// layout changes, then flip the catalog schema via
/// [`StorageEngine::alter_table`].
///
/// Operations split into two kinds:
///
/// - **Layout-changing** (`ADD`/`DROP COLUMN`, `SET DATA TYPE`) — every visible
///   row is decoded under the old schema, transformed, and re-encoded under the
///   new layout before the catalog flips. Tuples are opaque to the engine, so
///   rewriting the bytes first and updating the catalog second is consistent.
/// - **Catalog-only** (`RENAME COLUMN`, `SET`/`DROP NOT NULL`) — no row bytes
///   change. `SET NOT NULL` still scans to reject a column that holds `NULL`s.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-action dispatch over the full ALTER TABLE surface"
)]
pub(super) fn run_alter_table(
    plan: AlterTablePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let (table, op) = match plan {
        AlterTablePlan::Noop => return Ok(ExecutionResult::Altered),
        // Row-level-security toggle: a SQL-layer catalog change, not a row rewrite.
        AlterTablePlan::SetRls { table, enabled } => {
            super::set_table_rls(engine, txn, &table, enabled)?;
            return Ok(ExecutionResult::Altered);
        },
        // Trigger toggle: a SQL-layer trigger-catalog change, not a row rewrite.
        AlterTablePlan::SetTriggerEnabled {
            table,
            name,
            enabled,
        } => {
            super::trigger::set_triggers_enabled(engine, txn, &table, name.as_deref(), enabled)?;
            return Ok(ExecutionResult::Altered);
        },
        // ADD PRIMARY KEY / UNIQUE: validate the existing rows satisfy it, then register it.
        AlterTablePlan::AddUniqueConstraint {
            table,
            name,
            columns,
            primary,
        } => {
            validate_add_unique_constraint(&table, &columns, primary, engine, txn)?;
            let index = engine.add_unique_constraint(txn, table.id, &name, &columns, primary)?;
            // Backfill the constraint's backing index with the existing rows: the backing
            // index is a scannable access path now, so on a populated table it must cover every
            // live row from the moment it exists — like `CREATE INDEX`'s backfill. Uniqueness was
            // already validated above (the backing index skips the engine's byte-level check).
            // A constraint-backing index is always plain-column, full, and ascending.
            let backing = nusadb_core::engine::IndexDef {
                name,
                table: table.id,
                columns,
                key_exprs: Vec::new(),
                predicate: None,
                include: Vec::new(),
                kind: nusadb_core::engine::IndexKind::BTree,
                unique: true,
            };
            if let Some(target) = dml::build_index_target(index, &table, &backing) {
                for (tid, row) in scan_table(&table, engine, txn)? {
                    dml::insert_into_indexes(
                        std::slice::from_ref(&target),
                        &row,
                        tid,
                        engine,
                        txn,
                    )?;
                }
            }
            return Ok(ExecutionResult::Altered);
        },
        // ADD FOREIGN KEY: register it, then validate the table's existing rows reference
        // live parent rows. A violation errors and the rollback-aware DDL unwinds the registration.
        AlterTablePlan::AddForeignKey { table, fk } => {
            register_foreign_key(table.id, &fk, engine, txn)?;
            let existing = scan_rows(&table, engine, txn)?;
            dml::enforce_fk_on_child_write(&table, &existing, engine, txn)?;
            return Ok(ExecutionResult::Altered);
        },
        // RENAME TO: a catalog-only rename, no row rewrite.
        AlterTablePlan::RenameTable { table, name } => {
            engine.alter_table(txn, table, &AlterOp::RenameTable { name })?;
            return Ok(ExecutionResult::Altered);
        },
        // DROP CONSTRAINT [IF EXISTS]. A missing constraint is a no-op only with IF EXISTS;
        // otherwise `drop_constraint` itself raises the engine's not-found error.
        AlterTablePlan::DropConstraint {
            table,
            name,
            if_exists,
        } => {
            let present = engine
                .list_constraints(table)?
                .iter()
                .any(|c| c.name == name);
            if !present && if_exists {
                return Ok(ExecutionResult::Altered);
            }
            engine.drop_constraint(txn, table, &name)?;
            return Ok(ExecutionResult::Altered);
        },
        // ADD CHECK: validate the existing rows satisfy the predicate (NULL/TRUE pass,
        // only FALSE fails), then persist the canonical predicate SQL so every later write enforces
        // it. The analyzer already type-checked `predicate` against this table's columns.
        AlterTablePlan::AddCheck {
            table,
            name,
            predicate_sql,
            predicate,
        } => {
            for row in &scan_rows(&table, engine, txn)? {
                if matches!(eval::eval(&predicate, row)?, ast::Value::Bool(false)) {
                    return Err(nusadb_core::Error::ConstraintViolation(format!(
                        "check constraint \"{}\" is violated by an existing row in \"{}\"",
                        name, table.name
                    ))
                    .into());
                }
            }
            engine.add_check_constraint(txn, table.id, &name, predicate_sql.as_bytes())?;
            return Ok(ExecutionResult::Altered);
        },
        AlterTablePlan::Apply { table, op } => (table, op),
    };

    // SET/DROP DEFAULT are SQL-layer column-default catalog edits — no engine layout change.
    match &op {
        AlterColumnOp::SetDefault {
            column,
            default_sql,
        } => {
            super::coldefault::set_default(&table.name, column, default_sql, engine, txn)?;
            return Ok(ExecutionResult::Altered);
        },
        AlterColumnOp::DropDefault { column } => {
            super::coldefault::drop_default(&table.name, column, engine, txn)?;
            return Ok(ExecutionResult::Altered);
        },
        _ => {},
    }

    let old_types = column_types(&table);
    let core_op = match &op {
        AlterColumnOp::AddColumn(column) => {
            rewrite_add_column(&table, column, &old_types, engine, txn)?;
            AlterOp::AddColumn(ColumnDef {
                name: column.name.clone(),
                ty: column.ty,
                nullable: column.nullable,
            })
        },
        AlterColumnOp::DropColumn { index } => {
            let name = column_name(&table, *index)?;
            rewrite_drop_column(&table, *index, &old_types, engine, txn)?;
            // Clear any persisted default for the dropped column, so a later re-add of the same
            // name does not inherit the stale default (the catalog keys by column name).
            super::coldefault::drop_default(&table.name, &name, engine, txn)?;
            AlterOp::DropColumn { name }
        },
        AlterColumnOp::SetType { index, ty } => {
            rewrite_set_type(&table, *index, *ty, &old_types, engine, txn)?;
            AlterOp::AlterColumnType {
                column: column_name(&table, *index)?,
                ty: *ty,
            }
        },
        AlterColumnOp::RenameColumn { index, to } => AlterOp::RenameColumn {
            from: column_name(&table, *index)?,
            to: to.clone(),
        },
        AlterColumnOp::SetNotNull { index } => {
            ensure_no_nulls(&table, *index, engine, txn)?;
            AlterOp::SetNotNull {
                column: column_name(&table, *index)?,
            }
        },
        AlterColumnOp::DropNotNull { index } => AlterOp::DropNotNull {
            column: column_name(&table, *index)?,
        },
        // Handled (and returned) above — they touch only the SQL-layer default catalog.
        AlterColumnOp::SetDefault { .. } | AlterColumnOp::DropDefault { .. } => {
            unreachable!("SET/DROP DEFAULT is handled before the engine-op match")
        },
    };

    engine.alter_table(txn, table.id, &core_op)?;
    Ok(ExecutionResult::Altered)
}

/// Validate that `table`'s existing rows satisfy a `PRIMARY KEY`/`UNIQUE` constraint about to be
/// added over `columns`. `PRIMARY KEY` additionally requires every key column to be
/// non-`NULL`. Uniqueness is checked with the same total order (`unique_key_cmp`) the runtime
/// enforcement uses, so a constraint added here behaves identically on later writes.
fn validate_add_unique_constraint(
    table: &TableSchema,
    columns: &[String],
    primary: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let ordinals = dml::constraint_ordinals(table, columns)?;
    let rows = scan_rows(table, engine, txn)?;
    if primary {
        for row in &rows {
            for (&ordinal, name) in ordinals.iter().zip(columns) {
                if matches!(row.get(ordinal), Some(ast::Value::Null) | None) {
                    return Err(nusadb_core::Error::ConstraintViolation(format!(
                        "column \"{name}\" contains NULL values; cannot add PRIMARY KEY"
                    ))
                    .into());
                }
            }
        }
    }
    let mut keys: Vec<Vec<ast::Value>> = rows
        .iter()
        .filter_map(|row| dml::unique_key(row, &ordinals))
        .collect();
    keys.sort_by(|a, b| dml::unique_key_cmp(a, b));
    if keys.windows(2).any(
        |pair| matches!(pair, [a, b] if dml::unique_key_cmp(a, b) == std::cmp::Ordering::Equal),
    ) {
        let kind = if primary { "primary key" } else { "unique" };
        return Err(nusadb_core::Error::ConstraintViolation(format!(
            "existing rows violate the {kind} constraint on ({})",
            columns.join(", ")
        ))
        .into());
    }
    Ok(())
}

/// Append the new column's slot to every stored row. A `DEFAULT
/// <expr>` backfills every existing row with the default's (constant) value and is persisted so
/// later inserts fill it too; a column with no default is backfilled with `NULL`. A `NOT NULL`
/// column is allowed only with a default (which fills the rows) or on an empty table — a
/// `NOT NULL` add with no default on a non-empty table is rejected (parity with the reference engine).
pub(super) fn rewrite_add_column(
    table: &TableSchema,
    column: &ast::ColumnDef,
    old_types: &[ColumnType],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut new_types = old_types.to_vec();
    new_types.push(column.ty);
    // The value existing rows get for the new column: its `DEFAULT <expr>` evaluated once (a
    // default references no other column, so it is a constant across rows), else `NULL`
    // (the reference engine backfills existing rows with the default, e.g.
    // `ADD COLUMN b INT DEFAULT 9` fills every old row with 9, not NULL). Generated / SERIAL
    // columns use their own fields, mutually exclusive with an explicit `DEFAULT`, and are not
    // reached here.
    let backfill = match &column.default_sql {
        Some(sql) => {
            let typed =
                crate::analyzer::analyze_default_expr(sql, column.ty, &super::dml::EmptyCatalog)?;
            eval::eval(&typed, &Vec::new())?
        },
        None => ast::Value::Null,
    };
    // A `NOT NULL` column with no default cannot backfill non-empty rows (the reference engine rejects it too); a
    // `NOT NULL DEFAULT <expr>` is fine — every row gets the default.
    if !column.nullable
        && matches!(backfill, ast::Value::Null)
        // Only an error if there actually is a row to leave null (short-circuit: don't
        // materialize the table just to test non-emptiness).
        && engine.scan(txn, table.id)?.try_next()?.is_some()
    {
        return Err(Error::NotNullViolation {
            column: column.name.clone(),
        });
    }
    // Persist the default so a later `INSERT` that omits the column also gets it (parity with the reference engine), the
    // same catalog the SET DEFAULT / CREATE TABLE paths use.
    if let Some(sql) = &column.default_sql {
        super::coldefault::set_default(&table.name, &column.name, sql, engine, txn)?;
    }
    // Re-index every rewritten row version: the rewrite supersedes each row under a new
    // tid, so without fresh entries every index of the table would lose all its rows to the
    // visibility filter. Ordinals are stable — the new column appends after the indexed ones.
    let index_targets = dml::secondary_index_targets(table, engine)?;
    for (tid, mut row) in scan_table(table, engine, txn)? {
        row.push(backfill.clone());
        let bytes = row::encode(&row, &new_types)?;
        let new_tid = engine.update(txn, table.id, tid, &bytes)?;
        dml::insert_into_indexes(&index_targets, &row, new_tid, engine, txn)?;
    }
    Ok(())
}

/// Drop the column at `index` from every stored row and re-encode under the
/// shortened layout.
pub(super) fn rewrite_drop_column(
    table: &TableSchema,
    index: usize,
    old_types: &[ColumnType],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut new_types = old_types.to_vec();
    new_types.remove(index);
    // Re-index against the POST-drop layout: ordinals shift past the removed column, and an
    // index on the dropped column itself no longer resolves — `secondary_index_targets` skips it
    // (the analyzer's plan-time skip keeps it from ever being scanned).
    let mut new_schema = table.clone();
    new_schema.columns.remove(index);
    let index_targets = dml::secondary_index_targets(&new_schema, engine)?;
    for (tid, mut row) in scan_table(table, engine, txn)? {
        row.remove(index);
        let bytes = row::encode(&row, &new_types)?;
        let new_tid = engine.update(txn, table.id, tid, &bytes)?;
        dml::insert_into_indexes(&index_targets, &row, new_tid, engine, txn)?;
    }
    Ok(())
}

/// Cast the value at `index` in every stored row to the new column type and
/// re-encode. A value that cannot be converted surfaces the cast's typed error.
pub(super) fn rewrite_set_type(
    table: &TableSchema,
    index: usize,
    ty: ColumnType,
    old_types: &[ColumnType],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let mut new_types = old_types.to_vec();
    *new_types
        .get_mut(index)
        .ok_or_else(|| internal_index(index))? = ty;
    // Re-index every rewritten row version: ordinals are unchanged, and an index over the
    // retyped column gets keys encoded from the cast values, matching what later query literals
    // of the new type probe with.
    let index_targets = dml::secondary_index_targets(table, engine)?;
    for (tid, mut row) in scan_table(table, engine, txn)? {
        let old = row.get(index).ok_or_else(|| internal_index(index))?.clone();
        set_at(&mut row, index, eval::cast_value(old, ty)?)?;
        let bytes = row::encode(&row, &new_types)?;
        let new_tid = engine.update(txn, table.id, tid, &bytes)?;
        dml::insert_into_indexes(&index_targets, &row, new_tid, engine, txn)?;
    }
    Ok(())
}

/// Reject `SET NOT NULL` when any visible row holds a `NULL` in the column.
pub(super) fn ensure_no_nulls(
    table: &TableSchema,
    index: usize,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    for (_, row) in scan_table(table, engine, txn)? {
        let value = row.get(index).ok_or_else(|| internal_index(index))?;
        if matches!(value, ast::Value::Null) {
            return Err(Error::NotNullViolation {
                column: column_name(table, index)?,
            });
        }
    }
    Ok(())
}

/// The catalog name of the column at `index`, or an internal error if the
/// analyzer produced an out-of-range ordinal.
pub(super) fn column_name(table: &TableSchema, index: usize) -> Result<String, Error> {
    Ok(column_at(table, index)?.name.clone())
}

/// One-line `EXPLAIN` summary of an `ALTER TABLE` plan.
pub(super) fn format_alter(plan: &AlterTablePlan) -> String {
    let AlterTablePlan::Apply { table, op } = plan else {
        return ": no-op".to_owned();
    };
    let detail = match op {
        AlterColumnOp::AddColumn(c) => format!("ADD COLUMN {}", c.name),
        AlterColumnOp::DropColumn { index } => {
            format!("DROP COLUMN {}", column_label(table, *index))
        },
        AlterColumnOp::RenameColumn { index, to } => {
            format!("RENAME COLUMN {} TO {to}", column_label(table, *index))
        },
        AlterColumnOp::SetType { index, ty } => {
            format!("ALTER COLUMN {} TYPE {ty:?}", column_label(table, *index))
        },
        AlterColumnOp::SetNotNull { index } => {
            format!("ALTER COLUMN {} SET NOT NULL", column_label(table, *index))
        },
        AlterColumnOp::DropNotNull { index } => {
            format!("ALTER COLUMN {} DROP NOT NULL", column_label(table, *index))
        },
        AlterColumnOp::SetDefault { column, .. } => format!("ALTER COLUMN {column} SET DEFAULT"),
        AlterColumnOp::DropDefault { column } => format!("ALTER COLUMN {column} DROP DEFAULT"),
    };
    format!(" {}: {detail}", table.name)
}

/// Best-effort column name for `EXPLAIN` output; falls back to the ordinal.
pub(super) fn column_label(table: &TableSchema, index: usize) -> String {
    table
        .columns
        .get(index)
        .map_or_else(|| format!("#{index}"), |c| c.name.clone())
}

// === ANALYZE ==============================================================

/// Recompute statistics for the planned columns and persist them via
/// [`StorageEngine::analyze_table`]. Scans the table once, computes per-column
/// sketch statistics ([`stats::column_stats`]), and pairs them with the
/// engine's authoritative [`row_count`](StorageEngine::row_count).
pub(super) fn run_analyze(
    plan: AnalyzePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let rows = scan_table(&plan.table, engine, txn)?;
    let mut columns = Vec::with_capacity(plan.columns.len());
    for &index in &plan.columns {
        let column = column_at(&plan.table, index)?;
        let values: Vec<ast::Value> = rows
            .iter()
            .map(|(_, row)| row.get(index).cloned().unwrap_or(ast::Value::Null))
            .collect();
        columns.push(stats::column_stats(&column.name, &values, column.ty)?);
    }
    let table_stats = TableStats {
        row_count: engine.row_count(plan.table.id)?,
        page_count: 0,
        columns,
    };
    engine.analyze_table(txn, plan.table.id, &table_stats)?;
    Ok(ExecutionResult::Analyzed {
        table: plan.table.name,
        columns: plan.columns.len(),
    })
}
