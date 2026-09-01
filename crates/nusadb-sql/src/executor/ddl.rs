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
        // Hide the engine's own catalogs, matching `information_schema.tables`. This listing had no
        // such filter, so any feature whose catalog had been created — a policy, a view, a trigger —
        // already leaked its `nusadb_*` table here; the ownership catalog, which exists as soon as
        // any table does, only made it unconditional.
        .filter(|name| !name.starts_with(crate::SYSTEM_TABLE_PREFIX))
        .map(|name| vec![ast::Value::Text(name)])
        .collect();
    Ok(ExecutionResult::Rows {
        columns: vec!["table".to_owned()],
        rows,
        command: RowsCommand::Select,
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
        command: RowsCommand::Select,
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
        ColumnType::Macaddr => "MACADDR".to_owned(),
        ColumnType::Macaddr8 => "MACADDR8".to_owned(),
        ColumnType::Inet => "INET".to_owned(),
        ColumnType::Cidr => "CIDR".to_owned(),
        ColumnType::Bit(n) => format!("BIT({n})"),
        ColumnType::VarBit(Some(n)) => format!("BIT VARYING({n})"),
        ColumnType::VarBit(None) => "BIT VARYING".to_owned(),
        ColumnType::Range(kind) => kind.name().to_owned(),
        ColumnType::Numeric { precision: 0, .. } => "NUMERIC".to_owned(),
        ColumnType::Numeric { precision, scale } => format!("NUMERIC({precision},{scale})"),
        ColumnType::Json => "JSON".to_owned(),
        ColumnType::Jsonb => "JSONB".to_owned(),
        ColumnType::Interval => "INTERVAL".to_owned(),
        ColumnType::Array(elem) => format!("{}[]", type_name(elem.column_type())),
        ColumnType::Vector(dim) => format!("VECTOR({dim})"),
        ColumnType::Geometry(kind) => kind.name().to_uppercase(),
        ColumnType::Tsvector => "TSVECTOR".to_owned(),
        ColumnType::Tsquery => "TSQUERY".to_owned(),
        ColumnType::Xml => "XML".to_owned(),
        // A bare enum type renders its generic keyword here; callers that know the column's enum type
        // name (SHOW COLUMNS / information_schema) render that name instead.
        ColumnType::Enum => "ENUM".to_owned(),
    }
}

// === DDL ==================================================================

#[allow(
    clippy::too_many_lines,
    reason = "flat CREATE TABLE: udt resolution, constraints, defaults, composite columns, ownership"
)]
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
    // A temporary table lives in the session's non-durable temp schema (`nusadb_temp_<id>`), which is
    // created lazily on the first temp CREATE and reused by later ones. Only mint it when absent — a
    // second `CREATE TEMP TABLE` in the same session must not fail on "schema already exists".
    if plan.temporary && engine.lookup_schema(&plan.schema)?.is_none() {
        engine.create_temp_schema(txn, &plan.schema)?;
    }
    // Resolve any deferred user-defined column type (B-ENUM): it must name a registered ENUM, which
    // is stored as its `TEXT` placeholder. An unresolved name is a loud error (caught here, not at
    // parse time, because only the executor can read the type catalog). The enum's label set is
    // captured so a membership CHECK can be registered below (the column stores as TEXT, so without
    // it any string would be accepted).
    let mut columns = Vec::with_capacity(plan.columns.len());
    let mut domain_checks: Vec<(String, String)> = Vec::new();
    // Composite-typed columns to register in the per-column catalog once the table exists:
    // `(column, type_name)`. The physical column type is `TEXT` (it holds the canonical form).
    let mut composite_columns: Vec<(String, String)> = Vec::new();
    // Enum-typed columns to register in the per-column enum catalog once the table exists:
    // `(column, enum_type_name)`. The physical column type is `Enum`; membership is enforced on
    // write by resolving the value to a declaration-order ordinal (an unknown label is rejected).
    let mut enum_columns: Vec<(String, String)> = Vec::new();
    for c in plan.columns {
        let mut ty = c.ty;
        let mut nullable = c.nullable;
        if let Some(udt) = &c.udt_name {
            if super::lookup_enum(engine, txn, udt)?.is_some() {
                // A native enum column: each value stores its own ordinal + label; the enum type name
                // is recorded per-column so writes resolve labels and DDL renders the declared type.
                ty = ColumnType::Enum;
                enum_columns.push((c.name.clone(), udt.clone()));
            } else if super::lookup_composite(engine, txn, udt)?.is_some() {
                // A composite column stores as its TEXT placeholder (the canonical `(f1,f2,…)` form);
                // its composite type name is recorded per-column so reads know it is composite.
                ty = ColumnType::Text;
                composite_columns.push((c.name.clone(), udt.clone()));
            } else if let Some(domain) = super::lookup_domain(engine, txn, udt)? {
                // A DOMAIN column takes the domain's base type; its NOT NULL and CHECKs are applied
                // to this column (the CHECK's `VALUE` placeholder rewritten to the column name).
                ty = crate::parser::parse_column_type(&domain.base_type_sql)?;
                nullable = nullable && !domain.not_null;
                for (i, check) in domain.checks.iter().enumerate() {
                    let predicate = crate::parser::substitute_check_value(check, &c.name)?;
                    let name = format!("{}{}_{i}", crate::SYNTHETIC_TYPE_CHECK_PREFIX, c.name);
                    domain_checks.push((name, predicate));
                }
            } else {
                return Err(Error::ObjectNotFound(format!(
                    "type \"{udt}\" does not exist"
                )));
            }
        }
        columns.push(ColumnDef {
            name: c.name,
            ty,
            nullable,
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
            constraint.nulls_not_distinct,
        )?;
    }
    // Register each FOREIGN KEY.
    for fk in &plan.foreign_keys {
        register_foreign_key(id, &def.schema, fk, engine, txn)?;
    }
    // Register each CHECK constraint: the canonical predicate SQL is persisted opaquely so
    // INSERT/UPDATE/COPY can re-parse and evaluate it per row.
    for chk in &plan.check_constraints {
        engine.add_check_constraint(txn, id, &chk.name, chk.predicate_sql.as_bytes())?;
    }
    // A DOMAIN column's CHECKs (already rewritten from `VALUE` to the column name) enforce on write
    // through the same machinery; the synthetic prefix hides them from introspection.
    for (name, predicate_sql) in &domain_checks {
        engine.add_check_constraint(txn, id, name, predicate_sql.as_bytes())?;
    }
    // Clear any per-column composite rows left by a prior table of this exact `(schema, name)` — a
    // dropped/recreated table would otherwise mis-tag a same-named non-composite column as composite.
    // Done unconditionally (even when this table has no composite column) so stale rows never linger.
    super::delete_composite_columns_for_table(engine, txn, &def.schema, &def.name)?;
    // Record each composite column's type name so reads (field access, comparison, output) know the
    // TEXT-stored column actually holds a composite value.
    for (column, type_name) in &composite_columns {
        super::store_composite_column(engine, txn, &def.schema, &def.name, column, type_name)?;
    }
    // Likewise clear then record per-column enum rows (same stale-row reasoning as composites), so a
    // write can resolve a label to its ordinal and DDL introspection can render the declared type.
    super::delete_enum_columns_for_table(engine, txn, &def.schema, &def.name)?;
    for (column, type_name) in &enum_columns {
        super::store_enum_column(engine, txn, &def.schema, &def.name, column, type_name)?;
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
    // Record who owns the new table. Without this the table would fall back to the "no recorded
    // owner" reading — owned by the bootstrap superuser — and the role that just created it would
    // be unable to read its own table.
    crate::rbac::set_owner(
        engine,
        txn,
        crate::ast::ObjectKind::Table,
        &format!("{}.{}", def.schema, def.name),
        &super::session_ctx::current_user(),
    )?;
    copy_like_width_checks(plan.like_source.as_deref(), id, engine, txn)?;
    // Record the child→parent inheritance edges so a later query on a parent expands to this table.
    // A prior same-named table's edges are cleared on DROP, so there is nothing stale to purge here.
    super::inheritance::record_inheritance(engine, txn, &def.name, &plan.inherits)?;
    // Partitioning: a `PARTITION BY` parent records its strategy + key; a `PARTITION OF` partition
    // validates + records its bound and joins the parent's inheritance set (so a query on the parent
    // expands over its partitions).
    if let Some(pb) = &plan.partition_by {
        super::partition::record_parent(engine, txn, &def.name, &pb.columns, pb.strategy)?;
    }
    if let Some(part) = &plan.partition_of {
        register_partition(&def, part, engine, txn)?;
    }
    Ok(ExecutionResult::Created(id))
}

/// Register a `PARTITION OF parent FOR VALUES ...` partition: the parent must be partitioned, the
/// bound's kind must match the parent's strategy, its values are coerced to the key column's type and
/// validated (range `lo < hi` + non-overlap; list values not already claimed; hash modulus/remainder
/// consistent), and the partition joins the parent's inheritance set so a query on the parent reads
/// its rows.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive per-strategy validation match (range/list/hash); splitting would scatter \
              the bound checks that belong together"
)]
fn register_partition(
    def: &TableDef,
    part: &crate::planner::PartitionOfPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    use std::cmp::Ordering;

    use crate::planner::PartitionBoundPlan;

    use super::partition::{self, PartitionBound};
    let key_columns =
        partition::parent_key_columns(engine, txn, &part.parent)?.ok_or_else(|| {
            Error::InvalidStatement(format!(
                "table \"{}\" is not partitioned, so \"{}\" cannot be a partition of it",
                part.parent, def.name
            ))
        })?;
    let strategy = partition::parent_strategy(engine, txn, &part.parent)?.unwrap_or_default();
    // Resolve each key column's type from the partition's (parent-derived) columns.
    let key_tys = key_columns
        .iter()
        .map(|kc| {
            def.columns
                .iter()
                .find(|c| &c.name == kc)
                .map(|c| c.ty)
                .ok_or_else(|| internal_index(0))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mismatch = |want: &str| {
        Error::InvalidStatement(format!(
            "partition \"{}\" bound does not match parent \"{}\"'s {want} strategy",
            def.name, part.parent
        ))
    };
    let existing = partition::partitions_of(engine, txn, &part.parent, &key_tys)?;
    let bound = match &part.bound {
        PartitionBoundPlan::Range { from, to } => {
            if strategy != "range" {
                return Err(mismatch("non-range"));
            }
            if from.len() != key_tys.len() || to.len() != key_tys.len() {
                return Err(Error::InvalidStatement(format!(
                    "partition \"{}\" range bound needs one value per key column ({})",
                    def.name,
                    key_tys.len()
                )));
            }
            let cast_tuple = |vals: &[ast::Value]| -> Result<Vec<ast::Value>, Error> {
                vals.iter()
                    .zip(&key_tys)
                    .map(|(v, ty)| super::eval::cast_value(v.clone(), *ty))
                    .collect()
            };
            let lo = cast_tuple(from)?;
            let hi = cast_tuple(to)?;
            if partition::compare_tuple(&lo, &hi) != Ordering::Less {
                return Err(Error::InvalidStatement(format!(
                    "partition \"{}\" lower bound must be strictly below the upper bound",
                    def.name
                )));
            }
            for e in &existing {
                if let PartitionBound::Range { lo: elo, hi: ehi } = &e.bound {
                    let disjoint = partition::compare_tuple(&hi, elo) != Ordering::Greater
                        || partition::compare_tuple(&lo, ehi) != Ordering::Less;
                    if !disjoint {
                        return Err(overlap_err(&def.name, &e.table));
                    }
                }
            }
            PartitionBound::Range { lo, hi }
        },
        PartitionBoundPlan::List(values) => {
            if strategy != "list" {
                return Err(mismatch("non-list"));
            }
            let one = key_tys.first().copied().ok_or_else(|| internal_index(0))?;
            let coerced = values
                .iter()
                .map(|v| super::eval::cast_value(v.clone(), one))
                .collect::<Result<Vec<_>, _>>()?;
            for e in &existing {
                if let PartitionBound::List(evals) = &e.bound {
                    for v in &coerced {
                        if evals
                            .iter()
                            .any(|x| super::eval::compare(v, x) == Ordering::Equal)
                        {
                            return Err(overlap_err(&def.name, &e.table));
                        }
                    }
                }
            }
            PartitionBound::List(coerced)
        },
        PartitionBoundPlan::Hash { modulus, remainder } => {
            if strategy != "hash" {
                return Err(mismatch("non-hash"));
            }
            if *modulus == 0 || *remainder >= *modulus {
                return Err(Error::InvalidStatement(format!(
                    "partition \"{}\": hash MODULUS must be > 0 and REMAINDER must be < MODULUS",
                    def.name
                )));
            }
            for e in &existing {
                if let PartitionBound::Hash {
                    modulus: em,
                    remainder: er,
                } = &e.bound
                {
                    if em != modulus {
                        return Err(Error::InvalidStatement(format!(
                            "partition \"{}\": every hash partition of \"{}\" must share MODULUS {em}",
                            def.name, part.parent
                        )));
                    }
                    if er == remainder {
                        return Err(overlap_err(&def.name, &e.table));
                    }
                }
            }
            PartitionBound::Hash {
                modulus: *modulus,
                remainder: *remainder,
            }
        },
        PartitionBoundPlan::Default => {
            // A hash-partitioned parent has no catch-all — every key hashes to some partition.
            if strategy == "hash" {
                return Err(Error::InvalidStatement(format!(
                    "a DEFAULT partition is not allowed under the hash-partitioned parent \"{}\"",
                    part.parent
                )));
            }
            // At most one catch-all per parent.
            if existing.iter().any(|e| partition::is_default(&e.bound)) {
                return Err(Error::InvalidStatement(format!(
                    "partition \"{}\" conflicts with the existing default partition of \"{}\"",
                    def.name, part.parent
                )));
            }
            PartitionBound::Default
        },
    };
    partition::record_partition(engine, txn, &def.name, &part.parent, &key_tys, &bound)?;
    // The partition reads through the parent via the shared inheritance-expansion machinery.
    super::inheritance::record_inheritance(
        engine,
        txn,
        &def.name,
        std::slice::from_ref(&part.parent),
    )?;
    Ok(())
}

/// Confirm every existing row of a freshly-attached `partition` falls within the bound just recorded
/// for it — the `ATTACH PARTITION` analogue of the reference engine's partition-constraint scan. A row
/// whose key lies outside the bound errors with `23514`, and the rollback-aware DDL unwinds the
/// attach. The bound is read back from the catalog so the exact recorded/coerced form is used.
fn validate_attach_rows(
    parent: &TableSchema,
    partition: &TableSchema,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(key_cols) = super::partition::parent_key_columns(engine, txn, &parent.name)? else {
        return Ok(());
    };
    // Locate each key column and its type in the partition (whose columns match the parent's).
    let mut key_pos = Vec::with_capacity(key_cols.len());
    let mut key_tys = Vec::with_capacity(key_cols.len());
    for kc in &key_cols {
        let Some((i, ty)) = partition
            .columns
            .iter()
            .enumerate()
            .find(|(_, c)| &c.name == kc)
            .map(|(i, c)| (i, c.ty))
        else {
            return Ok(());
        };
        key_pos.push(i);
        key_tys.push(ty);
    }
    let Some((_, bound)) =
        super::partition::partition_bound(engine, txn, &partition.name, &key_tys)?
    else {
        return Ok(());
    };
    // A catch-all accepts a row only if no sibling's bound does; a normal partition accepts a row
    // that falls within its own bound.
    let siblings = if super::partition::is_default(&bound) {
        super::partition::partitions_of(engine, txn, &parent.name, &key_tys)?
    } else {
        Vec::new()
    };
    for row in scan_rows(partition, engine, txn)? {
        let key: Vec<ast::Value> = key_pos
            .iter()
            .map(|&i| row.get(i).cloned().unwrap_or(ast::Value::Null))
            .collect();
        let ok = if super::partition::is_default(&bound) {
            !siblings
                .iter()
                .any(|s| super::partition::accepts(&key, &s.bound, &key_tys))
        } else {
            super::partition::accepts(&key, &bound, &key_tys)
        };
        if !ok {
            return Err(Error::Coded {
                message: format!(
                    "an existing row of \"{}\" does not fall within the bound for partition \"{}\"",
                    partition.name, partition.name
                ),
                sqlstate: "23514",
            });
        }
    }
    Ok(())
}

/// A "partition would overlap an existing one" error.
fn overlap_err(partition: &str, existing: &str) -> Error {
    Error::InvalidStatement(format!(
        "partition \"{partition}\" would overlap existing partition \"{existing}\""
    ))
}

/// For `CREATE TABLE ... (LIKE src)`, copy `src`'s synthetic width / length checks onto the new table
/// `id`. The analyzer already copied `src`'s columns, but the declared width of a narrow integer or
/// bounded string lives only in these generated checks (every integer stores as i64), not in the
/// runtime type — so without this the copy would silently accept values the source rejects. The
/// copied columns keep `src`'s names, so each predicate is valid as-is, and the constraint names are
/// per-table, so reusing them on the new table cannot collide.
fn copy_like_width_checks(
    like_source: Option<&str>,
    id: nusadb_core::TableId,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(source) = like_source else {
        return Ok(());
    };
    let Some(src) = engine.lookup_table_as_of(txn, source)? else {
        return Ok(());
    };
    for c in engine.list_constraints(src.id)? {
        if c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX)
            && let Some(bytes) = &c.expr
        {
            engine.add_check_constraint(txn, id, &c.name, bytes)?;
        }
    }
    Ok(())
}

/// Register one `FOREIGN KEY` on child table `child_id` (shared by CREATE TABLE and
/// ALTER TABLE ADD CONSTRAINT). Resolves the parent table against the live catalog and declares the
/// constraint — the engine validates the parent has a `PRIMARY KEY`. v1 references the parent's
/// `PRIMARY KEY` only: an explicit `REFERENCES parent(cols)` that is not exactly the parent's PK, or
/// Refuse to rename a column that something else records by name.
///
/// A column name is written down in more places than the table's own schema: a constraint keeps its
/// key columns and its `CHECK` body, a foreign key keeps the columns on both sides, an index keeps
/// its key and predicate, and a default is filed under the column. Renaming moves only the schema
/// entry, which leaves every one of those naming a column that no longer exists — the table stays
/// readable and stops accepting writes, silently.
///
/// Until those references are carried across, the rename is refused and the error names what is in
/// the way, so the operator can drop it, rename, and put it back. Refusing is worse than renaming
/// but far better than the table quietly going read-only.
///
/// # Errors
/// [`Error::Unsupported`] naming the first dependent found.
fn refuse_rename_with_dependents(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let refuse = |what: &str, name: &str| -> Error {
        // A generated check enforces the column's declared width and is not visible as a
        // constraint, so telling the operator to drop and redeclare it would be telling them to
        // drop the width enforcement they never wrote.
        let remedy = if name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX) {
            "the column's declared type is enforced by a check the engine generated, so renaming \
             is not yet supported for a column of this type"
        } else {
            "drop it, rename the column, then declare it again"
        };
        Error::DependentObjects(format!(
            "cannot rename column \"{column}\" of \"{}\": {what} \"{name}\" refers to it by name \
             and would stop resolving — {remedy}",
            table.name
        ))
    };
    for c in engine.list_constraints(table.id)? {
        // The engine's own synthetic type-check (declared width / length enforcement) references the
        // column by name, but the rename rewrites it for the new name rather than refusing — so it
        // never blocks a rename. Every non-synthetic dependent below still does.
        if c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX) {
            continue;
        }
        if c.columns.iter().any(|k| k == column) {
            return Err(refuse("constraint", &c.name));
        }
        // A CHECK body is stored as SQL text; look for the column as a whole word so a longer name
        // that merely contains it does not block the rename.
        if let Some(bytes) = &c.expr
            && let Ok(sql) = std::str::from_utf8(bytes)
            && sql_mentions_column(sql, column)
        {
            return Err(refuse("check constraint", &c.name));
        }
    }
    // The engine reports keys where this table is either side, so pin each column list to its own
    // side before matching — the child columns of a key we merely parent belong to another table,
    // and one of them sharing this column's name must not block the rename.
    for fk in engine.list_foreign_keys(table.id)? {
        let child_side = fk.child_table == table.id && fk.child_columns.iter().any(|k| k == column);
        let parent_side =
            fk.parent_table == table.id && fk.parent_columns.iter().any(|k| k == column);
        if child_side || parent_side {
            return Err(refuse("foreign key", &fk.name));
        }
    }
    for def in engine.list_indexes(table.id)? {
        let named = def.columns.iter().chain(&def.include).any(|k| k == column)
            || def.key_exprs.iter().any(|e| sql_mentions_column(e, column))
            || def
                .predicate
                .as_deref()
                .is_some_and(|p| sql_mentions_column(p, column));
        if named {
            return Err(refuse("index", &def.name));
        }
    }
    for (owner, sql) in super::coldefault::load_defaults(
        &super::coldefault::catalog_key(&table.schema, &table.name),
        engine,
        txn,
    )? {
        // The column's own default moves with it, and a generated column's expression may read
        // this column from elsewhere in the table — that expression is evaluated on every write.
        if owner == column {
            return Err(refuse("default expression on column", column));
        }
        if let Some(expr) = super::coldefault::generated_expr(&sql)
            && sql_mentions_column(expr, column)
        {
            return Err(refuse("generated column", &owner));
        }
    }
    if let Some((what, name)) = sql_dependent_naming(table, column, engine, txn)? {
        return Err(refuse(what, &name));
    }
    Ok(())
}

/// The engine's synthetic type-check constraints whose predicate names `column`, as
/// `(constraint name, predicate SQL)`. These are the width / length checks the engine generates for a
/// typed column (`INT` range, `VARCHAR`/`CHAR` length); their predicate references the column by name,
/// so `ALTER TABLE ... RENAME COLUMN` collects them here to drop and re-add rewritten for the new
/// name — they cannot be regenerated from the runtime type, since the declared width is stored only in
/// the predicate.
fn synthetic_type_checks_on_column(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
) -> Result<Vec<(String, String)>, Error> {
    let mut out = Vec::new();
    for c in engine.list_constraints(table.id)? {
        if !c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX) {
            continue;
        }
        if let Some(bytes) = &c.expr
            && let Ok(sql) = std::str::from_utf8(bytes)
            && sql_mentions_column(sql, column)
        {
            out.push((c.name.clone(), sql.to_owned()));
        }
    }
    Ok(out)
}

/// The first SQL-holding derived object — view, materialized view, policy, trigger, function,
/// procedure — whose stored definition names `column` of `table`, as `(kind, name)`.
///
/// Split from [`refuse_rename_with_dependents`] only for length; the policy is the same, erring
/// toward reporting a dependant.
fn sql_dependent_naming(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Option<(&'static str, String)>, Error> {
    // The derived objects below keep their defining SQL as text and re-analyze it later — a view
    // on every read, a materialized view on refresh (and, when incrementally maintained, on every
    // write to this table), a policy on every read it protects, a trigger on every firing, a
    // function or procedure body when called. Their catalogs key by name, so a definition naming
    // this table and this column is what breaks.
    //
    // The view and routine catalogs store the table bare; the policy and trigger catalogs now
    // carry a schema column, which the callers below check against this table's own. The qualified
    // spelling is still accepted for the catalogs that key by a single name.
    let qualified = format!("{}.{}", table.schema, table.name);
    let names_this_table = |t: &str| t == table.name || t == qualified;
    // For a catalog with its own schema column: the name must match AND, when the row records a
    // namespace, it must be this one. Without the second half, renaming a column on `public.t` was
    // refused because a policy on `app.t` mentioned the name.
    let owns =
        |t: &str, s: Option<&String>| names_this_table(t) && s.is_none_or(|s| *s == table.schema);
    let mut views = Vec::new();
    for (catalog, what) in [
        (VIEW_CATALOG, "view"),
        (MATVIEW_CATALOG, "materialized view"),
    ] {
        for row in scan_text_catalog(engine, txn, catalog, &[2])? {
            let mut row = row.into_iter();
            if let (Some(name), Some(def)) = (row.next(), row.next()) {
                views.push((what, name, def));
            }
        }
    }
    // A view can stand for the table without naming it: `v2` reads `v1` reads the table, and only
    // `v1`'s definition says so. Grow the set of names that reach the table until it stops
    // growing, then judge every definition against the whole set.
    let mut reaches: Vec<&str> = vec![&table.name];
    loop {
        let mut grew = false;
        for (_, name, def) in &views {
            if !reaches.iter().any(|n| n == name)
                && reaches.iter().any(|n| sql_mentions_column(def, n))
            {
                reaches.push(name);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    for (what, name, def) in &views {
        if sql_mentions_column(def, column) && reaches.iter().any(|n| sql_mentions_column(def, n)) {
            return Ok(Some((what, name.clone())));
        }
    }
    // A function or procedure body is re-analyzed when called, and may read the table through any
    // of the names that reach it. Both catalogs are `(name, …, body)` with the body last; the
    // function catalog has a legacy three-column layout from before parameter names were kept.
    for (catalog, what, widths) in [
        (
            super::function::FUNCTION_CATALOG,
            "function",
            &[4usize, 3][..],
        ),
        (super::procedure::PROCEDURE_CATALOG, "procedure", &[4][..]),
    ] {
        for row in scan_text_catalog(engine, txn, catalog, widths)? {
            if let (Some(name), Some(body)) = (row.first(), row.last())
                && sql_mentions_column(body, column)
                && reaches.iter().any(|n| sql_mentions_column(body, n))
            {
                return Ok(Some((what, name.clone())));
            }
        }
    }
    // (table, name, command, roles, using, check, permissive) — an orphaned policy fails closed,
    // which locks every non-superuser out of the table rather than leaking rows.
    for row in scan_text_catalog(engine, txn, POLICY_CATALOG, &[8, 7])? {
        if let [tbl, name, _, _, using, check, _, rest @ ..] = row.as_slice()
            && owns(tbl, rest.first())
            && (sql_mentions_column(using, column) || sql_mentions_column(check, column))
        {
            return Ok(Some(("policy", name.clone())));
        }
    }
    // (name, table, timing, events, for_each, when, action, enabled) — with a legacy seven-column
    // row missing the trailing flag. A disabled trigger blocks too: it can be re-enabled at any
    // time, and would come back broken.
    for row in scan_text_catalog(engine, txn, super::trigger::TRIGGER_CATALOG, &[9, 8, 7])? {
        if let [name, tbl, _, _, _, when, action, rest @ ..] = row.as_slice()
            && owns(tbl, rest.get(1))
            && (sql_mentions_column(when, column) || sql_mentions_column(action, column))
        {
            return Ok(Some(("trigger", name.clone())));
        }
    }
    Ok(None)
}

/// Scan an engine-scoped all-`TEXT` system catalog, yielding each row's columns as strings. A
/// catalog that was never created yields nothing. `widths` lists the accepted column counts,
/// current first, so a catalog with a legacy narrower layout still decodes; a row that decodes at
/// no accepted width is reported rather than skipped — skipping would silently disarm a guard.
fn scan_text_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
    catalog: &str,
    widths: &[usize],
) -> Result<Vec<Vec<String>>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, catalog)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let schemas: Vec<Vec<ColumnType>> = widths.iter().map(|&w| vec![ColumnType::Text; w]).collect();
    let malformed = || Error::Coded {
        message: format!(
            "system catalog {catalog} holds a row this build cannot decode; the statement is \
             refused rather than judged against a partial catalog"
        ),
        sqlstate: crate::error::INTERNAL_ERROR,
    };
    let mut scan = engine.scan(txn, cat.id)?;
    'rows: while let Some((_, bytes)) = scan.try_next()? {
        for schema in &schemas {
            if let Ok(row) = row::decode(&bytes, schema) {
                let mut cols = Vec::with_capacity(row.len());
                for value in row {
                    match value {
                        ast::Value::Text(s) => cols.push(s),
                        _ => return Err(malformed()),
                    }
                }
                out.push(cols);
                continue 'rows;
            }
        }
        return Err(malformed());
    }
    Ok(out)
}

/// Whether a stored SQL fragment refers to `column` as a whole identifier, rather than as part of a
/// longer name.
///
/// The comparison ignores case: a predicate is kept as the text it was written in, while a column
/// name is folded, so `CHECK (B > 0)` refers to column `b`. Missing that would let the rename
/// through and leave the table unwritable, which is the whole thing being prevented — so this errs
/// toward matching, and a name that merely appears inside a string literal blocks the rename too.
pub(super) fn sql_mentions_column(sql: &str, column: &str) -> bool {
    let is_part = |c: char| c.is_alphanumeric() || c == '_';
    // Fold the full character set, not just ASCII: the identifier boundary test is Unicode-aware,
    // so a non-ASCII name stored in another capitalisation would otherwise slip through — a miss
    // here is the unsafe direction.
    let (sql, column) = (sql.to_lowercase(), column.to_lowercase());
    let mut from = 0;
    while let Some(offset) = sql[from..].find(&column) {
        let at = from + offset;
        let end = at + column.len();
        let before = sql[..at].chars().next_back();
        let after = sql[end..].chars().next();
        if !before.is_some_and(is_part) && !after.is_some_and(is_part) {
            return true;
        }
        // Resume past the whole identifier this sat inside, so a longer name starting with the
        // column (`bb` for `b`) cannot match on its own tail.
        from = end + sql[end..].find(|c| !is_part(c)).unwrap_or(sql.len() - end);
    }
    false
}

/// Resolve the table a foreign key points at.
///
/// The name is tried whole first, so every target that resolved before resolves to the same table —
/// a key already pointing at the default schema keeps pointing there, and a table whose own name
/// contains a dot is still found. Only a name that would otherwise have been reported missing takes
/// the two further steps: read a qualifier out of it, then look in the referencing table's own
/// schema, so a table can reference a sibling without repeating the schema they share.
fn resolve_parent_table(
    child_schema: &str,
    parent: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<TableSchema, Error> {
    if let Some(t) = engine.lookup_table_as_of(txn, parent)? {
        return Ok(t);
    }
    if let Some((schema, name)) = parent.split_once('.')
        && let Some(t) = engine.lookup_table_as_of_in(txn, schema, name)?
    {
        return Ok(t);
    }
    engine
        .lookup_table_as_of_in(txn, child_schema, parent)?
        .ok_or_else(|| Error::TableNotFound {
            name: parent.to_owned(),
        })
}

/// Register one `FOREIGN KEY` on child table `child_id` (shared by CREATE TABLE and
/// ALTER TABLE ADD CONSTRAINT). Resolves the parent table against the live catalog and declares the
/// constraint — the engine validates the parent has a `PRIMARY KEY`. v1 references the parent's
/// `PRIMARY KEY` only: an explicit `REFERENCES parent(cols)` that is not exactly the parent's PK, or
/// any arity mismatch, is rejected (silently redirecting to the PK would mis-enforce). This does NOT
/// validate existing child rows — `ALTER TABLE ADD` does that separately.
pub(super) fn register_foreign_key(
    child_id: nusadb_core::TableId,
    child_schema: &str,
    fk: &crate::planner::ForeignKeySpec,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let parent = resolve_parent_table(child_schema, &fk.parent_table, engine, txn)?;
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
        return Err(Error::InvalidStatement(format!(
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
        return Err(Error::InvalidStatement(format!(
            "foreign key \"{}\" references columns of \"{}\" that are not a PRIMARY KEY or UNIQUE \
             constraint",
            fk.name, fk.parent_table
        )));
    }
    if fk.columns.len() != parent_columns.len() {
        return Err(Error::InvalidStatement(format!(
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
            // An inheritance parent cannot be dropped while children depend on it: the children would
            // be left with columns whose origin is gone. Refuse loudly (drop the inheriting tables
            // first) rather than orphan them — done before any teardown so a refusal is a no-op.
            let children = super::inheritance::direct_children(engine, txn, &plan.table)?;
            if !children.is_empty() {
                return Err(Error::DependentObjects(format!(
                    "cannot drop table \"{}\": {} table(s) inherit from it (drop them first)",
                    plan.table,
                    children.len()
                )));
            }
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
                    return Err(Error::DependentObjects(format!(
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
            // Scrub the table's per-column composite-type rows so a later same-named table cannot
            // inherit them (a non-composite column would otherwise be mis-tagged as composite).
            super::delete_composite_columns_for_table(engine, txn, &plan.schema, &plan.table)?;
            // Same for per-column enum rows, so a later same-named table's non-enum column of the
            // same name is not mis-tagged as an enum.
            super::delete_enum_columns_for_table(engine, txn, &plan.schema, &plan.table)?;
            // Cascade-drop the table's row-level-security policies and its RLS-enabled marker (
            // ): otherwise they orphan the catalog, and a later same-named table cannot
            // re-create a policy of the same name ("policy already exists").
            super::delete_policies_for_table(engine, txn, &plan.schema, &plan.table)?;
            super::set_table_rls(engine, txn, &plan.schema, &plan.table, false)?;
            // Triggers were not cascaded at all, so their rows outlived the table and a later
            // same-named one inherited them.
            super::trigger::delete_triggers_for_table(engine, txn, &plan.schema, &plan.table)?;
            // Ownership and grants go with the table. Leaving them behind would hand a later table
            // that reused the name the old one's permissions.
            let owned = format!("{}.{}", plan.schema, plan.table);
            crate::rbac::clear_owner(engine, txn, crate::ast::ObjectKind::Table, &owned)?;
            crate::rbac::delete_grants_on(engine, txn, crate::ast::ObjectKind::Table, &owned)?;
            // Remove this table's inheritance edges (as a child of its own parents) so a later
            // same-named table does not inherit stale parent links.
            super::inheritance::remove_edges_for(engine, txn, &plan.table)?;
            // Same for any partition metadata (parent key or partition bound).
            super::partition::remove_for(engine, txn, &plan.table)?;
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
    // Own the schema under its creator so its owner (not only a superuser) can later drop it.
    crate::rbac::set_owner(
        engine,
        txn,
        crate::ast::ObjectKind::Schema,
        &plan.name,
        &super::session_ctx::current_user(),
    )?;
    Ok(ExecutionResult::SchemaCreated)
}

pub(super) fn run_drop_schema(
    plan: &DropSchemaPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    match engine.lookup_schema(&plan.name)? {
        // RESTRICT (default) refuses a non-empty schema; CASCADE drops its member tables too.
        Some(id) => {
            // The engine's cascade drops member tables directly, bypassing `run_drop_table`'s
            // owner/grant cleanup — so clear each member's owner and grants first, then the
            // schema's own owner. Otherwise a later same-named table (or schema) silently
            // inherits the dropped one's permissions.
            if plan.cascade {
                for (schema, name) in engine.list_tables_qualified_as_of(txn)? {
                    if schema == plan.name {
                        let owned = format!("{schema}.{name}");
                        crate::rbac::clear_owner(
                            engine,
                            txn,
                            crate::ast::ObjectKind::Table,
                            &owned,
                        )?;
                        crate::rbac::delete_grants_on(
                            engine,
                            txn,
                            crate::ast::ObjectKind::Table,
                            &owned,
                        )?;
                    }
                }
            }
            engine.drop_schema(txn, id, plan.cascade)?;
            // The engine drops the tables; the SQL-layer catalogs live above it and kept their
            // rows, so recreating the schema inherited policies and triggers nobody declared.
            super::purge_schema_catalogs(engine, txn, &plan.name)?;
            crate::rbac::clear_owner(engine, txn, crate::ast::ObjectKind::Schema, &plan.name)?;
            crate::rbac::delete_grants_on(engine, txn, crate::ast::ObjectKind::Schema, &plan.name)?;
        },
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

pub(super) fn run_alter_sequence(
    plan: &AlterSequencePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    match engine.lookup_sequence(&plan.name)? {
        Some(id) => engine.alter_sequence(txn, id, &plan.change)?,
        None => {
            if !plan.if_exists {
                return Err(Error::SequenceNotFound {
                    name: plan.name.clone(),
                });
            }
        },
    }
    Ok(ExecutionResult::SequenceAltered)
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
        // A name already taken — by another vector index OR a B-tree index — must be refused, the
        // same as a B-tree `CREATE INDEX` does, rather than silently overwriting (and rebuilding
        // the expensive HNSW graph over) the existing one. `IF NOT EXISTS` makes it a no-op instead.
        let name_taken = super::vector_index_exists(engine, txn, &spec.name)?
            || engine.lookup_index(&spec.name)?.is_some();
        if name_taken {
            if plan.if_not_exists {
                return Ok(ExecutionResult::IndexCreated);
            }
            return Err(nusadb_core::Error::IndexExists {
                name: spec.name.clone(),
            }
            .into());
        }
        super::store_vector_index(engine, txn, spec)?;
        // Eager build: build the HNSW graph now, so the cost lands at `CREATE INDEX`
        // (like a B-tree backfill) rather than on the first query, and a failed/cancelled build
        // fails this statement instead of leaving a half-built graph for a query to hit. The graph
        // is cached in-process; a fresh process still rebuilds on first use (persistence is separate).
        if let Some(table) = engine.lookup_table_as_of(txn, &spec.table)? {
            ops::warm_vector_index(
                &spec.name,
                &table,
                spec.column_ordinal,
                spec.dim,
                spec.metric,
                engine,
                txn,
            )?;
        }
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
        // is evaluated and a partial predicate skips non-matching rows exactly as on later inserts).
        // The build streams the table and applies key-sorted entries — via an external merge sort
        // when spill is configured (one global key order), else in bounded in-memory chunks — without
        // materializing the whole table's rows or entries at once.
        dml::backfill_index(&target, &table, engine, txn)?;
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
        AlterTablePlan::SetRls {
            schema,
            table,
            enabled,
        } => {
            // Refused here rather than inside `set_table_rls`, which the `DROP TABLE` cascade also
            // calls: a marker recorded before namespaces covers every same-named table, so
            // toggling it for one of them cannot be answered — but dropping the table must still
            // be able to retire it, or the state has no way out.
            if super::covered_by_unrecorded_namespace(engine, txn, &table)? {
                return Err(Error::Unsupported(format!(
                    concat!(
                        "row-level security on `{0}` was recorded before namespaces were, so it ",
                        "covers every `{0}` in this engine and cannot be changed for one of them; ",
                        "drop or recreate `{0}` in the default namespace to retire it"
                    ),
                    table
                )));
            }
            super::set_table_rls(engine, txn, &schema, &table, enabled)?;
            return Ok(ExecutionResult::Altered);
        },
        // Trigger toggle: a SQL-layer trigger-catalog change, not a row rewrite.
        AlterTablePlan::SetTriggerEnabled {
            schema,
            table,
            name,
            enabled,
        } => {
            super::trigger::set_triggers_enabled(
                engine,
                txn,
                &schema,
                &table,
                name.as_deref(),
                enabled,
            )?;
            return Ok(ExecutionResult::Altered);
        },
        // ADD PRIMARY KEY / UNIQUE: validate the existing rows satisfy it, then register it.
        AlterTablePlan::AddUniqueConstraint {
            table,
            name,
            columns,
            primary,
            nulls_not_distinct,
        } => {
            validate_add_unique_constraint(
                &table,
                &columns,
                primary,
                nulls_not_distinct,
                engine,
                txn,
            )?;
            let index = engine.add_unique_constraint(
                txn,
                table.id,
                &name,
                &columns,
                primary,
                nulls_not_distinct,
            )?;
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
            register_foreign_key(table.id, &table.schema, &fk, engine, txn)?;
            let existing = scan_rows(&table, engine, txn)?;
            // Validating rows already in the table when a key is added: nothing is pending.
            dml::enforce_fk_on_child_write(&table, &existing, &[], engine, txn)?;
            return Ok(ExecutionResult::Altered);
        },
        // RENAME TO: a catalog-only rename, no row rewrite.
        AlterTablePlan::RenameTable {
            table,
            name,
            from,
            schema,
        } => {
            let to = format!("{schema}.{name}");
            engine.alter_table(txn, table, &AlterOp::RenameTable { name })?;
            // Carry ownership and grants to the new name. Without this the renamed table reads as
            // unowned — which resolves to the bootstrap superuser — locking its owner out, and the
            // stale rows under the old name would be inherited by whatever is created with it next.
            crate::rbac::rename_owned_object(
                engine,
                txn,
                crate::ast::ObjectKind::Table,
                &from,
                &to,
            )?;
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
        // ATTACH PARTITION: validate + record the bound and the child→parent edge (via the same
        // registration path CREATE ... PARTITION OF uses), then confirm every existing child row falls
        // within the bound. A stray row errors and the rollback-aware DDL unwinds the registration.
        AlterTablePlan::AttachPartition {
            parent,
            partition,
            bound,
        } => {
            if let Some(existing) =
                super::partition::partition_parent(engine, txn, &partition.name)?
            {
                return Err(Error::InvalidStatement(format!(
                    "table \"{}\" is already a partition of \"{existing}\"",
                    partition.name
                )));
            }
            if partition.name == parent.name {
                return Err(Error::InvalidStatement(
                    "a table cannot be attached as a partition of itself".to_owned(),
                ));
            }
            let def = TableDef {
                schema: partition.schema.clone(),
                name: partition.name.clone(),
                columns: partition.columns.clone(),
            };
            let part = crate::planner::PartitionOfPlan {
                parent: parent.name.clone(),
                bound,
            };
            register_partition(&def, &part, engine, txn)?;
            validate_attach_rows(&parent, &partition, engine, txn)?;
            return Ok(ExecutionResult::Altered);
        },
        // DETACH PARTITION: confirm the child really is a partition of this parent, then sever just
        // that link (its partition-catalog row and its child→parent edge). The table and its rows
        // survive as an independent table.
        AlterTablePlan::DetachPartition { parent, partition } => {
            match super::partition::partition_parent(engine, txn, &partition)? {
                Some(actual) if actual == parent => {},
                _ => {
                    return Err(Error::InvalidStatement(format!(
                        "table \"{partition}\" is not a partition of \"{parent}\""
                    )));
                },
            }
            super::partition::remove_partition_entry(engine, txn, &partition)?;
            super::inheritance::remove_edge(engine, txn, &partition, &parent)?;
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
        AlterColumnOp::RenameColumn { index, to } => {
            let from = column_name(&table, *index)?;
            refuse_rename_with_dependents(&table, &from, engine, txn)?;
            // The engine's synthetic type-check(s) on this column name it in their predicate, so they
            // must move with the rename. They cannot be regenerated from the column's runtime type
            // (the declared width lives only in the predicate — every integer stores as i64), so the
            // stored predicate is rewritten by re-quoting the new name. Drop before the rename, re-add
            // after, all in this txn: drop clears the old dependency, the rename lands, the rewritten
            // check re-enforces the same bound under the new name.
            let synthetic = synthetic_type_checks_on_column(&table, &from, engine)?;
            for c in &synthetic {
                engine.drop_constraint(txn, table.id, &c.0)?;
            }
            engine.alter_table(
                txn,
                table.id,
                &AlterOp::RenameColumn {
                    from: from.clone(),
                    to: to.clone(),
                },
            )?;
            for (old_name, predicate) in &synthetic {
                // Re-name by replacing only the column stem, so a per-check disambiguator suffix
                // (`…_0`, `…_1` on a multi-CHECK DOMAIN column) is preserved and two checks on one
                // renamed column cannot collide on the same new name.
                let old_stem = format!("{}{from}", crate::SYNTHETIC_TYPE_CHECK_PREFIX);
                let new_stem = format!("{}{to}", crate::SYNTHETIC_TYPE_CHECK_PREFIX);
                let new_name = old_name.replacen(&old_stem, &new_stem, 1);
                let new_predicate = predicate.replace(&format!("\"{from}\""), &format!("\"{to}\""));
                engine.add_check_constraint(txn, table.id, &new_name, new_predicate.as_bytes())?;
            }
            return Ok(ExecutionResult::Altered);
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
    nulls_not_distinct: bool,
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
        .filter_map(|row| dml::unique_key(row, &ordinals, nulls_not_distinct))
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
