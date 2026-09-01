//! DDL analyzers: CREATE TABLE, DROP TABLE, ALTER TABLE, ANALYZE.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === DDL ==================================================================

#[allow(
    clippy::too_many_lines,
    reason = "flat CREATE TABLE analysis: LIKE/INHERITS merge, partitioning, schema, constraints"
)]
pub(super) fn analyze_create_table(
    mut ct: ast::CreateTable,
    catalog: &dyn Catalog,
) -> Result<CreateTablePlan, Error> {
    // A non-superuser must not squat a system-catalog name (e.g. pre-create `nusadb_policies`
    // before the engine does, forging what later reads as the policy catalog).
    enforce_system_catalog(&ct.name, catalog)?;
    // `ON COMMIT {DELETE ROWS | DROP}` only means anything for a temporary table (its rows/lifetime
    // are tied to the transaction). On an ordinary table it is meaningless, so reject it loudly rather
    // than create a persistent table that silently ignores the clause.
    if !ct.temporary
        && matches!(
            ct.on_commit,
            ast::OnCommit::DeleteRows | ast::OnCommit::Drop
        )
    {
        return Err(Error::InvalidTableDefinition(
            "ON COMMIT can only be used on temporary tables".to_owned(),
        ));
    }
    // `CREATE TABLE ... (LIKE src)`: copy `src`'s columns (type + `NOT NULL`) ahead of any columns
    // written after the LIKE clause, as the reference engine does. Only name/type/nullable are copied
    // (the basic LIKE form); the executor additionally copies `src`'s synthetic width/length checks,
    // since the declared width is not recoverable from the runtime `ColumnType`.
    if let Some(source) = &ct.like_source {
        let src = catalog
            .lookup_table(source)?
            .ok_or_else(|| Error::TableNotFound {
                name: source.clone(),
            })?;
        let mut merged: Vec<ast::ColumnDef> = src
            .columns
            .iter()
            .map(|c| ast::ColumnDef {
                name: c.name.clone(),
                ty: c.ty,
                udt_name: None,
                nullable: c.nullable,
                primary_key: false,
                unique: false,
                default: None,
                default_sql: None,
                generated: None,
                serial: false,
                identity_always: false,
            })
            .collect();
        merged.append(&mut ct.columns);
        ct.columns = merged;
    }
    {
        let mut seen = HashSet::new();
        for column in &ct.columns {
            if !seen.insert(column.name.as_str()) {
                return Err(Error::DuplicateColumn {
                    name: column.name.clone(),
                });
            }
        }
    }
    // `INHERITS (parent, ...)`: prepend the parents' columns (in order, deduped across parents),
    // then merge the child's own columns by name. A query on a parent later expands to this table's
    // rows (see the inheritance catalog + the analyzer's scan expansion).
    let inherited_parents = merge_inherited_columns(&mut ct, catalog)?;
    // `PARTITION OF parent`: a range partition takes the parent's columns and a `[from, to)` bound.
    // `PARTITION BY RANGE (col)`: a partitioned parent, keyed on an existing column.
    let partition_of = resolve_partition_of(&mut ct, catalog)?;
    let partition_by = resolve_partition_by(&ct)?;
    if partition_by.is_some() && partition_of.is_some() {
        return Err(Error::Unsupported(
            "a partition that is itself partitioned (sub-partitioning) is not supported".to_owned(),
        ));
    }
    // An unqualified CREATE targets the session's current schema; an explicit qualifier wins. A
    // temporary table instead targets the session's non-durable temp schema — which does NOT change
    // `current_schema`, so a plain CREATE alongside it still lands in the search-path schema.
    let target_schema = if ct.temporary {
        if ct.schema.is_some() {
            // A temp table lives in the session's own temp schema; naming another schema is a
            // contradiction, not a silent override.
            return Err(Error::Unsupported(
                "CREATE TEMPORARY TABLE with an explicit schema qualifier".to_owned(),
            ));
        }
        catalog.temp_schema().ok_or_else(|| {
            Error::Unsupported(
                "temporary tables require a session (no temp schema available)".to_owned(),
            )
        })?
    } else {
        ct.schema
            .clone()
            .unwrap_or_else(|| catalog.current_schema())
    };
    if !ct.if_not_exists && catalog.lookup_table_in(&target_schema, &ct.name)?.is_some() {
        return Err(Error::TableExists {
            name: super::qualified_display(&target_schema, &ct.name),
        });
    }
    // The base for auto-generated constraint / index / sequence names. The engine's
    // constraint/index/sequence namespace is keyed by name, so for a non-public schema the base is
    // schema-qualified (`app.users`) — otherwise `app.users` and `public.users` would both want
    // `users_pkey`. A `public` table keeps the bare name, so existing names are byte-for-byte
    // unchanged. Names a user supplies (e.g. `CONSTRAINT pk PRIMARY KEY`) are honoured as-is.
    let name_base = super::qualified_display(&target_schema, &ct.name);
    let unique_constraints = resolve_unique_constraints(&ct, &name_base)?;
    let foreign_keys = resolve_foreign_keys(&ct, &name_base)?;
    let check_constraints = resolve_check_constraints(&ct, &name_base, catalog)?;
    let defaults = resolve_column_defaults(&ct, &name_base, catalog)?;
    // A constraint on a partitioned parent would need to be enforced across every partition, which
    // NusaDB does not propagate yet — so a `UNIQUE`/`PRIMARY KEY`/`CHECK`/`FOREIGN KEY` on the parent
    // would go silently unenforced on the partitions' rows. Refuse it loudly rather than mis-accept.
    // The synthetic type-range checks (a narrow integer's bound) are excluded: each partition
    // regenerates them from its copied columns, so they are enforced where the rows actually live.
    let explicit_check = check_constraints
        .iter()
        .any(|c| !c.name.starts_with(SYNTHETIC_TYPE_CHECK_PREFIX));
    if partition_by.is_some()
        && (!unique_constraints.is_empty() || !foreign_keys.is_empty() || explicit_check)
    {
        return Err(Error::Unsupported(
            "a UNIQUE / PRIMARY KEY / CHECK / FOREIGN KEY constraint on a partitioned table is not \
             supported yet (it would not be enforced across partitions)"
                .to_owned(),
        ));
    }
    Ok(CreateTablePlan {
        schema: target_schema,
        table: ct.name,
        columns: ct.columns,
        unique_constraints,
        foreign_keys,
        check_constraints,
        defaults,
        if_not_exists: ct.if_not_exists,
        temporary: ct.temporary,
        like_source: ct.like_source,
        on_commit: ct.on_commit,
        inherits: inherited_parents,
        partition_by,
        partition_of,
    })
}

/// Validate a `PARTITION BY {RANGE|LIST|HASH} (col)` key: the column must be one of the table's
/// declared columns. Returns the strategy + key column (or `None` for a non-partitioned table).
fn resolve_partition_by(ct: &ast::CreateTable) -> Result<Option<ast::PartitionBy>, Error> {
    let Some(pb) = &ct.partition_by else {
        return Ok(None);
    };
    // A list-partitioned table has a single key column (the reference engine's rule).
    if pb.strategy == ast::PartitionStrategy::List && pb.columns.len() != 1 {
        return Err(Error::Unsupported(
            "LIST partitioning supports a single key column".to_owned(),
        ));
    }
    for column in &pb.columns {
        if !ct.columns.iter().any(|c| &c.name == column) {
            return Err(Error::ColumnNotFound {
                table: ct.name.clone(),
                column: column.clone(),
            });
        }
    }
    Ok(Some(pb.clone()))
}

/// Resolve a `PARTITION OF parent FOR VALUES ...`: the parent must exist, the partition declares no
/// columns of its own (it takes the parent's), and range/list bound values must be constant literals.
/// Sets `ct.columns` to the parent's columns and returns the resolved bound (its kind is checked
/// against the parent's strategy by the executor, which reads the recorded strategy).
fn resolve_partition_of(
    ct: &mut ast::CreateTable,
    catalog: &dyn Catalog,
) -> Result<Option<crate::planner::PartitionOfPlan>, Error> {
    let Some(po) = ct.partition_of.clone() else {
        return Ok(None);
    };
    if !ct.columns.is_empty() {
        return Err(Error::Unsupported(
            "a partition takes its parent's columns and cannot declare its own".to_owned(),
        ));
    }
    let parent = catalog
        .lookup_table(&po.parent)?
        .ok_or_else(|| Error::TableNotFound {
            name: po.parent.clone(),
        })?;
    ct.columns = parent
        .columns
        .iter()
        .map(|c| ast::ColumnDef {
            name: c.name.clone(),
            ty: c.ty,
            udt_name: None,
            nullable: c.nullable,
            primary_key: false,
            unique: false,
            default: None,
            default_sql: None,
            generated: None,
            serial: false,
            identity_always: false,
        })
        .collect();
    let bound = convert_partition_bound(&po.bound)?;
    Ok(Some(crate::planner::PartitionOfPlan {
        parent: parent.name,
        bound,
    }))
}

/// Convert a partition bound's AST (whose operands are expressions) into a plan bound (constants),
/// evaluating each range/list operand to a literal. Shared by `PARTITION OF` and `ATTACH PARTITION`.
fn convert_partition_bound(
    bound: &ast::PartitionBound,
) -> Result<crate::planner::PartitionBoundPlan, Error> {
    use crate::planner::PartitionBoundPlan;
    Ok(match bound {
        ast::PartitionBound::Range { from, to } => PartitionBoundPlan::Range {
            from: from
                .iter()
                .map(const_bound_value)
                .collect::<Result<_, _>>()?,
            to: to.iter().map(const_bound_value).collect::<Result<_, _>>()?,
        },
        ast::PartitionBound::List(values) => PartitionBoundPlan::List(
            values
                .iter()
                .map(const_bound_value)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ast::PartitionBound::Hash { modulus, remainder } => PartitionBoundPlan::Hash {
            modulus: *modulus,
            remainder: *remainder,
        },
        ast::PartitionBound::Default => PartitionBoundPlan::Default,
    })
}

/// Extract the literal value of a partition bound expression. Only a literal (optionally negated for
/// a number) is supported; a non-constant bound is refused rather than silently mishandled.
fn const_bound_value(expr: &ast::Expr) -> Result<ast::Value, Error> {
    match expr {
        ast::Expr::Literal(v) => Ok(v.clone()),
        ast::Expr::Unary {
            op: ast::UnaryOp::Negate,
            expr,
        } => match const_bound_value(expr)? {
            ast::Value::Int(n) => Ok(ast::Value::Int(-n)),
            ast::Value::Float(f) => Ok(ast::Value::Float(-f)),
            ast::Value::Numeric(d) => Ok(ast::Value::Numeric(d.neg())),
            _ => Err(Error::Unsupported(
                "a partition bound must be a constant literal".to_owned(),
            )),
        },
        _ => Err(Error::Unsupported(
            "a partition bound must be a constant literal (an expression bound is not supported)"
                .to_owned(),
        )),
    }
}

/// Apply `INHERITS (parent, ...)`: prepend the parents' columns (in written order, deduped across
/// parents by name) ahead of the child's own, then merge the child's own columns by name — a
/// redeclared inherited column merges (its type must match) and the child's own definition wins,
/// while a new column is appended. Returns the resolved parent table names (empty for a
/// non-inheriting table) for the executor to record as inheritance edges. `NOT NULL` is the OR of the
/// merged columns (a column is nullable only if every merged copy is).
fn merge_inherited_columns(
    ct: &mut ast::CreateTable,
    catalog: &dyn Catalog,
) -> Result<Vec<String>, Error> {
    if ct.inherits.is_empty() {
        return Ok(Vec::new());
    }
    let inherited_col = |c: &ColumnDef| ast::ColumnDef {
        name: c.name.clone(),
        ty: c.ty,
        udt_name: None,
        nullable: c.nullable,
        primary_key: false,
        unique: false,
        default: None,
        default_sql: None,
        generated: None,
        serial: false,
        identity_always: false,
    };
    // Column counts are small, so a linear find-by-name (rather than a side index) keeps this simple
    // and avoids any fallible indexing.
    let mut merged: Vec<ast::ColumnDef> = Vec::new();
    let mut parents = Vec::with_capacity(ct.inherits.len());
    for parent_name in &ct.inherits {
        let parent = catalog
            .lookup_table(parent_name)?
            .ok_or_else(|| Error::TableNotFound {
                name: parent_name.clone(),
            })?;
        for col in &parent.columns {
            if let Some(slot) = merged.iter_mut().find(|c| c.name == col.name) {
                if slot.ty != col.ty {
                    return Err(Error::InvalidTableDefinition(format!(
                        "inherited column \"{}\" has a type conflict between parents",
                        col.name
                    )));
                }
                slot.nullable = slot.nullable && col.nullable;
            } else {
                merged.push(inherited_col(col));
            }
        }
        parents.push(parent.name.clone());
    }
    for col in std::mem::take(&mut ct.columns) {
        if let Some(slot) = merged.iter_mut().find(|c| c.name == col.name) {
            if slot.ty != col.ty {
                return Err(Error::InvalidTableDefinition(format!(
                    "column \"{}\" conflicts with the type of the inherited column",
                    col.name
                )));
            }
            // The child's own definition (its NOT NULL / DEFAULT / constraints) wins; the type is
            // already checked to match the inherited one.
            *slot = col;
        } else {
            merged.push(col);
        }
    }
    ct.columns = merged;
    Ok(parents)
}

/// Build the column scope for a not-yet-created table (its declared columns, qualified by the table
/// name) so a `CHECK` predicate can be type-checked at `CREATE TABLE` time.
fn create_table_scope(ct: &ast::CreateTable) -> Vec<ScopedColumn> {
    ct.columns
        .iter()
        .map(|c| ScopedColumn {
            qualifier: ct.name.clone(),
            def: ColumnDef {
                name: c.name.clone(),
                ty: c.ty,
                nullable: c.nullable,
            },
            qualified_only: false,
            // A `CHECK` predicate is type-checked at CREATE TABLE, before any composite catalog row
            // for this column exists, so field access is not resolved here.
            composite_type: None,
            enum_type: None,
            // Not a user `SELECT` read of a base table, so column-scoped SELECT never gates it.
            select_granted: true,
        })
        .collect()
}

/// Resolve the `CHECK` constraints of a `CREATE TABLE`, from column-level (lifted by the
/// parser) and table-level declarations. Each predicate is type-checked (boolean, columns exist)
/// against the new table's columns and must be subquery-free (a CHECK references only its own row);
/// the predicate's SQL text is carried for the executor to persist and re-enforce per row.
fn resolve_check_constraints(
    ct: &ast::CreateTable,
    name_base: &str,
    catalog: &dyn Catalog,
) -> Result<Vec<CheckSpec>, Error> {
    let scope = create_table_scope(ct);
    let mut specs = Vec::new();
    let mut seq = 0;
    for constraint in &ct.constraints {
        let ast::TableConstraint::Check {
            name,
            expr,
            predicate_sql,
        } = constraint
        else {
            continue;
        };
        validate_check_predicate(expr, &scope, catalog)?;
        // Only auto-named (unnamed) checks consume a sequence number, so a named check — including a
        // synthetic type-bound one — does not shift the `t_checkN` numbering of the user's checks.
        let name = name.clone().unwrap_or_else(|| {
            seq += 1;
            format!("{name_base}_check{seq}")
        });
        specs.push(CheckSpec {
            name,
            predicate_sql: predicate_sql.clone(),
        });
    }
    Ok(specs)
}

/// Type-check a `CHECK` predicate against `scope`: it must be boolean and subquery-free (the
/// executor re-checks it against a row-only scope on every write, where a subquery cannot resolve).
fn validate_check_predicate(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<(), Error> {
    let typed = analyze_expr(expr, scope, catalog, Some(ColumnType::Bool))?;
    if typed.ty != ColumnType::Bool {
        return Err(Error::TypeMismatch {
            context: "CHECK constraint".to_owned(),
            expected: ColumnType::Bool,
            found: typed.ty,
        });
    }
    if crate::executor::ops::contains_subquery(&typed) {
        return Err(Error::Unsupported(
            "a CHECK constraint may not contain a subquery".to_owned(),
        ));
    }
    Ok(())
}

/// Resolve the column `DEFAULT` expressions of a `CREATE TABLE`. Each default is type-checked
/// against an **empty** scope — a default references no other column — must be assignable to its
/// column's type, and must be subquery-free. Its canonical SQL text is carried `(column, sql)` for the
/// executor to persist in the column-default catalog and re-evaluate per write.
fn resolve_column_defaults(
    ct: &ast::CreateTable,
    name_base: &str,
    catalog: &dyn Catalog,
) -> Result<Vec<(String, String)>, Error> {
    // Scope of only the non-generated columns: a `GENERATED` expression is analyzed against this, so
    // one that references another generated column fails with column-not-found (the reference engine forbids a
    // generated column referencing another generated column). Defaults/serials use the empty scope.
    let non_generated_scope: Vec<ScopedColumn> = ct
        .columns
        .iter()
        .filter(|c| c.generated.is_none())
        .map(|c| ScopedColumn {
            qualifier: ct.name.clone(),
            def: ColumnDef {
                name: c.name.clone(),
                ty: c.ty,
                nullable: c.nullable,
            },
            qualified_only: false,
            // A `CHECK` predicate is type-checked at CREATE TABLE, before any composite catalog row
            // for this column exists, so field access is not resolved here.
            composite_type: None,
            enum_type: None,
            // Not a user `SELECT` read of a base table, so column-scoped SELECT never gates it.
            select_granted: true,
        })
        .collect();
    let mut defaults = Vec::new();
    for column in &ct.columns {
        // A GENERATED ALWAYS AS (<expr>) STORED column is a computed column: its expression
        // (referencing the row's other, non-generated columns) is stored as a sentinel "default" the
        // executor re-evaluates per row. VIRTUAL is not supported (it would need read-time evaluation).
        if let Some(generated) = &column.generated {
            if !generated.stored {
                return Err(Error::Unsupported(
                    "VIRTUAL generated columns are not supported; declare the column STORED"
                        .to_owned(),
                ));
            }
            if column.default.is_some() {
                // `42P16` rather than the plain syntax error `42601`: the statement parses and the
                // column definition contradicts itself, which is what invalid_table_definition
                // names. This is a deliberate divergence — the widely-deployed engine reports
                // `42601` here and keeps `42P16` for the `ON COMMIT` family — taken because the two
                // sit in the same class and no client branches between them, so matching the
                // standard's meaning costs nothing and reads truer.
                return Err(Error::InvalidTableDefinition(
                    "a GENERATED column may not also have a DEFAULT".to_owned(),
                ));
            }
            let typed = analyze_expr(
                &generated.expr,
                &non_generated_scope,
                catalog,
                Some(column.ty),
            )?;
            if crate::executor::ops::contains_subquery(&typed) {
                return Err(Error::Unsupported(
                    "a GENERATED column expression may not contain a subquery".to_owned(),
                ));
            }
            let col = ColumnDef {
                name: column.name.clone(),
                ty: column.ty,
                nullable: column.nullable,
            };
            super::typecheck::check_assignable(&col, &typed)?;
            defaults.push((
                column.name.clone(),
                crate::executor::coldefault::generated_default_sql(&generated.sql),
            ));
            continue;
        }
        // A SERIAL column is an auto-increment INT backed by a per-column sequence; it is
        // recorded as a sentinel "default" the executor resolves to `nextval`. It cannot also carry
        // an explicit DEFAULT.
        if column.serial {
            if column.default.is_some() {
                return Err(Error::InvalidTableDefinition(
                    "a SERIAL column may not also have a DEFAULT".to_owned(),
                ));
            }
            let seq = crate::executor::coldefault::sequence_name(name_base, &column.name);
            let sentinel = if column.identity_always {
                crate::executor::coldefault::identity_always_default_sql(&seq)
            } else {
                crate::executor::coldefault::serial_default_sql(&seq)
            };
            defaults.push((column.name.clone(), sentinel));
            continue;
        }
        let (Some(expr), Some(sql)) = (&column.default, &column.default_sql) else {
            continue;
        };
        let typed = analyze_expr(expr, &[], catalog, Some(column.ty))?;
        if crate::executor::ops::contains_subquery(&typed) {
            return Err(Error::Unsupported(
                "a column DEFAULT may not contain a subquery".to_owned(),
            ));
        }
        let col = ColumnDef {
            name: column.name.clone(),
            ty: column.ty,
            nullable: column.nullable,
        };
        super::typecheck::check_assignable(&col, &typed)?;
        defaults.push((column.name.clone(), sql.clone()));
    }
    Ok(defaults)
}

/// Resolve the `FOREIGN KEY` constraints of a `CREATE TABLE`. Child columns must exist on
/// the new table; the parent table/key is validated at registration time. An explicit
/// `REFERENCES parent (cols)` list references those columns (which must form a `PRIMARY KEY` or
/// `UNIQUE` constraint on the parent); an unqualified `REFERENCES parent` references the parent's
/// `PRIMARY KEY`. The `referred_columns` are carried through to the executor, which validates them.
fn resolve_foreign_keys(
    ct: &ast::CreateTable,
    name_base: &str,
) -> Result<Vec<ForeignKeySpec>, Error> {
    let mut specs: Vec<ForeignKeySpec> = Vec::new();
    let mut seq = 0;
    for constraint in &ct.constraints {
        let ast::TableConstraint::ForeignKey {
            name,
            columns,
            foreign_table,
            referred_columns,
            on_delete,
            on_update,
        } = constraint
        else {
            continue;
        };
        for column in columns {
            if !ct.columns.iter().any(|c| &c.name == column) {
                return Err(Error::ColumnNotFound {
                    table: ct.name.clone(),
                    column: column.clone(),
                });
            }
        }
        seq += 1;
        specs.push(ForeignKeySpec {
            name: name
                .clone()
                .unwrap_or_else(|| format!("{name_base}_fkey{seq}")),
            columns: columns.clone(),
            parent_table: foreign_table.clone(),
            referred_columns: referred_columns.clone(),
            on_delete: referential_action(*on_delete),
            on_update: referential_action(*on_update),
        });
    }
    Ok(specs)
}

/// Map a parsed [`ast::ReferentialAction`] (or its absence) to the engine's [`FkAction`]. An
/// unspecified action defaults to `NO ACTION` (the SQL default).
const fn referential_action(action: Option<ast::ReferentialAction>) -> nusadb_core::FkAction {
    use nusadb_core::FkAction as F;
    match action {
        None | Some(ast::ReferentialAction::NoAction) => F::NoAction,
        Some(ast::ReferentialAction::Restrict) => F::Restrict,
        Some(ast::ReferentialAction::Cascade) => F::Cascade,
        Some(ast::ReferentialAction::SetNull) => F::SetNull,
        Some(ast::ReferentialAction::SetDefault) => F::SetDefault,
    }
}

/// Resolve the `PRIMARY KEY` / `UNIQUE` constraints of a `CREATE TABLE`, from both
/// column-level (`id INT PRIMARY KEY`, `email TEXT UNIQUE`) and table-level (`PRIMARY KEY (a, b)`,
/// `UNIQUE (x)`) declarations. Every constraint column must exist; at most one `PRIMARY KEY` is
/// allowed. `FOREIGN KEY` / `CHECK` table constraints are out of scope here (FK enforcement is a
/// separate task; CHECK is not yet wired) and are rejected to keep the surface honest.
fn resolve_unique_constraints(
    ct: &ast::CreateTable,
    name_base: &str,
) -> Result<Vec<UniqueConstraintSpec>, Error> {
    let column_exists = |name: &str| ct.columns.iter().any(|c| c.name == name);
    let mut specs: Vec<UniqueConstraintSpec> = Vec::new();

    // Column-level PRIMARY KEY / UNIQUE.
    for column in &ct.columns {
        if column.primary_key {
            specs.push(UniqueConstraintSpec {
                name: format!("{name_base}_pkey"),
                columns: vec![column.name.clone()],
                primary: true,
                nulls_not_distinct: false,
            });
        }
        if column.unique {
            // A column-level `UNIQUE NULLS NOT DISTINCT` is not modelled at the surface (sqlparser has
            // no grammar for it on a column); the table-level `UNIQUE NULLS NOT DISTINCT (col)` form is.
            specs.push(UniqueConstraintSpec {
                name: format!("{name_base}_{}_key", column.name),
                columns: vec![column.name.clone()],
                primary: false,
                nulls_not_distinct: false,
            });
        }
    }

    // Table-level constraints.
    for constraint in &ct.constraints {
        match constraint {
            ast::TableConstraint::PrimaryKey { name, columns }
            | ast::TableConstraint::Unique { name, columns, .. } => {
                let (primary, nulls_not_distinct) = match constraint {
                    ast::TableConstraint::PrimaryKey { .. } => (true, false),
                    ast::TableConstraint::Unique {
                        nulls_not_distinct, ..
                    } => (false, *nulls_not_distinct),
                    _ => unreachable!("outer match binds only PrimaryKey / Unique"),
                };
                for column in columns {
                    if !column_exists(column) {
                        return Err(Error::ColumnNotFound {
                            table: ct.name.clone(),
                            column: column.clone(),
                        });
                    }
                }
                let default = if primary {
                    format!("{name_base}_pkey")
                } else {
                    format!("{name_base}_{}_key", columns.join("_"))
                };
                specs.push(UniqueConstraintSpec {
                    name: name.clone().unwrap_or(default),
                    columns: columns.clone(),
                    primary,
                    nulls_not_distinct,
                });
            },
            // Foreign keys / CHECK are resolved separately (`resolve_foreign_keys` /
            // `resolve_check_constraints`).
            ast::TableConstraint::ForeignKey { .. } | ast::TableConstraint::Check { .. } => {},
        }
    }

    if specs.iter().filter(|s| s.primary).count() > 1 {
        return Err(Error::InvalidTableDefinition(
            "a table may have at most one PRIMARY KEY".to_owned(),
        ));
    }
    Ok(specs)
}

pub(super) fn analyze_drop_table(
    dt: ast::DropTable,
    catalog: &dyn Catalog,
) -> Result<DropTablePlan, Error> {
    enforce_system_catalog(&dt.name, catalog)?;
    // Resolve through the search path (an explicit qualifier wins) so the plan drops the exact table
    // a bare name resolves to.
    let resolved = super::lookup_table_ref(dt.schema.as_deref(), &dt.name, catalog)?;
    if !dt.if_exists && resolved.is_none() {
        return Err(Error::TableNotFound {
            name: super::qualified_display_opt(dt.schema.as_deref(), &dt.name),
        });
    }
    // Destroying a table is the owner's right, not a grantable privilege — otherwise a role given
    // INSERT could drop the table out from under everyone else granted access to it.
    if let Some(table) = &resolved {
        super::dcl::require_table_ownership(catalog, table, "drop")?;
    }
    // Drop where it actually resolved; under IF EXISTS on a missing table fall back to the explicit
    // (or current) schema — the executor then finds nothing and no-ops.
    let schema = resolved
        .map(|t| t.schema)
        .or_else(|| dt.schema.clone())
        .unwrap_or_else(|| catalog.current_schema());
    Ok(DropTablePlan {
        cascade: dt.cascade,
        schema,
        table: dt.name,
        if_exists: dt.if_exists,
    })
}

/// Resolve and validate a single `ALTER TABLE` action against the catalog.
///
/// Column references become ordinals into the pre-alter schema so the executor
/// never re-consults the catalog. `IF [NOT] EXISTS` guards (missing table,
/// already-present added column, missing dropped column) collapse to
/// [`AlterTablePlan::Noop`]. Operations the [`AlterOp`](nusadb_core::AlterOp)
/// treaty does not model — column `DEFAULT` and `ADD`/`DROP CONSTRAINT` — are
/// rejected with [`Error::Unsupported`] so the surface stays honest.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-action dispatch over the full ALTER TABLE surface"
)]
pub(super) fn analyze_alter_table(
    at: ast::AlterTable,
    catalog: &dyn Catalog,
) -> Result<AlterTablePlan, Error> {
    enforce_system_catalog(&at.name, catalog)?;
    // Resolve the target through the search path (an explicit qualifier wins) so `ALTER TABLE` reaches
    // a non-public table exactly like the bare name would in a SELECT. The qualified form is
    // used in error messages and as the auto-name base for generated constraints.
    let display = super::qualified_display_opt(at.schema.as_deref(), &at.name);
    let Some(table) = super::lookup_table_ref(at.schema.as_deref(), &at.name, catalog)? else {
        if at.if_exists {
            return Ok(AlterTablePlan::Noop);
        }
        return Err(Error::TableNotFound { name: display });
    };
    // Restructuring a table is likewise the owner's right: an ALTER can drop a column, change a
    // type, or add a constraint that silently invalidates other roles' access.
    super::dcl::require_table_ownership(catalog, &table, "alter")?;
    let op = match at.action {
        // Row-level-security toggles are SQL-layer catalog changes, not column rewrites — they
        // produce a `SetRls` plan rather than an `AlterColumnOp`. Reserved to superusers, so a
        // non-superuser cannot lift its own RLS (e.g. `... DISABLE ROW LEVEL SECURITY`).
        ast::AlterTableAction::EnableRowLevelSecurity => {
            require_rls_admin(catalog, "enable row-level security on a table")?;
            return Ok(AlterTablePlan::SetRls {
                schema: table.schema,
                table: table.name,
                enabled: true,
            });
        },
        ast::AlterTableAction::DisableRowLevelSecurity => {
            require_rls_admin(catalog, "disable row-level security on a table")?;
            return Ok(AlterTablePlan::SetRls {
                schema: table.schema,
                table: table.name,
                enabled: false,
            });
        },
        // Trigger toggles are SQL-layer trigger-catalog changes, not column rewrites. The named
        // trigger's existence is checked by the executor against the trigger catalog (like
        // DROP TRIGGER); the table itself was resolved above.
        ast::AlterTableAction::EnableTrigger { name } => {
            return Ok(AlterTablePlan::SetTriggerEnabled {
                schema: table.schema.clone(),
                table: table.name,
                name,
                enabled: true,
            });
        },
        ast::AlterTableAction::DisableTrigger { name } => {
            return Ok(AlterTablePlan::SetTriggerEnabled {
                schema: table.schema.clone(),
                table: table.name,
                name,
                enabled: false,
            });
        },
        ast::AlterTableAction::AddColumn {
            column,
            if_not_exists,
        } => {
            if column.primary_key {
                return Err(Error::Unsupported(
                    "ALTER TABLE ADD COLUMN ... PRIMARY KEY is not supported \
                     (no analysis-time constraint catalog hook yet)"
                        .to_owned(),
                ));
            }
            if table.columns.iter().any(|c| c.name == column.name) {
                if if_not_exists {
                    return Ok(AlterTablePlan::Noop);
                }
                return Err(Error::DuplicateColumn { name: column.name });
            }
            AlterColumnOp::AddColumn(column)
        },
        ast::AlterTableAction::DropColumn { name, if_exists } => {
            let Some(index) = table.columns.iter().position(|c| c.name == name) else {
                if if_exists {
                    return Ok(AlterTablePlan::Noop);
                }
                return Err(Error::ColumnNotFound {
                    table: display,
                    column: name,
                });
            };
            if table.columns.len() == 1 {
                return Err(Error::InvalidTableDefinition(
                    "ALTER TABLE DROP COLUMN would leave the table with no columns".to_owned(),
                ));
            }
            AlterColumnOp::DropColumn { index }
        },
        ast::AlterTableAction::RenameColumn { from, to } => {
            let (index, _) = find_column(&table.columns, &from, &display)?;
            if table.columns.iter().any(|c| c.name == to) {
                return Err(Error::DuplicateColumn { name: to });
            }
            AlterColumnOp::RenameColumn { index, to }
        },
        ast::AlterTableAction::AlterColumn { column, change } => {
            let (index, _) = find_column(&table.columns, &column, &display)?;
            match change {
                ast::ColumnChange::SetType(ty) => AlterColumnOp::SetType { index, ty },
                ast::ColumnChange::SetNotNull => AlterColumnOp::SetNotNull { index },
                ast::ColumnChange::DropNotNull => AlterColumnOp::DropNotNull { index },
                // `SET DEFAULT <expr>`: type-check the default against an empty scope (it
                // references no column), require it assignable to the column type and subquery-free,
                // then persist it. The column ordinal is unused — defaults are keyed by name.
                ast::ColumnChange::SetDefault { expr, sql } => {
                    let col = find_column(&table.columns, &column, &display)?.1;
                    let typed = analyze_expr(&expr, &[], catalog, Some(col.ty.physical()))?;
                    if crate::executor::ops::contains_subquery(&typed) {
                        return Err(Error::Unsupported(
                            "a column DEFAULT may not contain a subquery".to_owned(),
                        ));
                    }
                    super::typecheck::check_assignable(col, &typed)?;
                    AlterColumnOp::SetDefault {
                        column,
                        default_sql: sql,
                    }
                },
                ast::ColumnChange::DropDefault => AlterColumnOp::DropDefault { column },
            }
        },
        ast::AlterTableAction::AddConstraint(constraint) => {
            return analyze_add_constraint(table, constraint, catalog);
        },
        ast::AlterTableAction::DropConstraint { name, if_exists } => {
            return Ok(AlterTablePlan::DropConstraint {
                table: table.id,
                name,
                if_exists,
            });
        },
        ast::AlterTableAction::RenameTable { name } => {
            return analyze_rename_table(table.id, &table.schema, &table.name, name, catalog);
        },
        ast::AlterTableAction::AttachPartition { partition, bound } => {
            return analyze_attach_partition(table, &partition, &bound, catalog);
        },
        ast::AlterTableAction::DetachPartition { partition } => {
            return analyze_detach_partition(table, &partition, catalog);
        },
    };
    Ok(AlterTablePlan::Apply { table, op })
}

/// Resolve `ALTER TABLE parent ATTACH PARTITION child FOR VALUES <bound>`. The parent must be a
/// partitioned parent and the child an existing table whose columns match the parent's (name + type +
/// order); the executor then validates the bound against the parent's strategy and that every existing
/// child row falls within it.
fn analyze_attach_partition(
    parent: TableSchema,
    partition: &str,
    bound: &ast::PartitionBound,
    catalog: &dyn Catalog,
) -> Result<AlterTablePlan, Error> {
    if catalog.partition_key_column(&parent.name)?.is_none() {
        return Err(Error::Unsupported(format!(
            "table \"{}\" is not partitioned, so a partition cannot be attached to it",
            parent.name
        )));
    }
    let child = catalog
        .lookup_table(partition)?
        .ok_or_else(|| Error::TableNotFound {
            name: partition.to_owned(),
        })?;
    // The child's columns must line up with the parent's (same names, types, and order).
    let columns_match = child.columns.len() == parent.columns.len()
        && child
            .columns
            .iter()
            .zip(&parent.columns)
            .all(|(c, p)| c.name == p.name && c.ty == p.ty);
    if !columns_match {
        return Err(Error::Unsupported(format!(
            "table \"{}\" cannot be attached to \"{}\": its columns must match the parent's",
            child.name, parent.name
        )));
    }
    let bound = convert_partition_bound(bound)?;
    Ok(AlterTablePlan::AttachPartition {
        parent,
        partition: child,
        bound,
    })
}

/// Resolve `ALTER TABLE parent DETACH PARTITION child`. The parent must be partitioned and the child
/// an existing table; the executor confirms the child is actually a partition of this parent before
/// severing the link.
fn analyze_detach_partition(
    parent: TableSchema,
    partition: &str,
    catalog: &dyn Catalog,
) -> Result<AlterTablePlan, Error> {
    if catalog.partition_key_column(&parent.name)?.is_none() {
        return Err(Error::Unsupported(format!(
            "table \"{}\" is not partitioned, so no partition can be detached from it",
            parent.name
        )));
    }
    let child = catalog
        .lookup_table(partition)?
        .ok_or_else(|| Error::TableNotFound {
            name: partition.to_owned(),
        })?;
    Ok(AlterTablePlan::DetachPartition {
        parent: parent.name,
        partition: child.name,
    })
}

/// Resolve `ALTER TABLE ... RENAME TO name`: the new name must be free and not collide with a system
/// catalog (which a rename would otherwise shadow), exactly as `CREATE TABLE` checks its name. The
/// rename stays within the table's own schema — `RENAME TO` never moves a table across schemas — so
/// the collision check looks in `schema`, not the search-path default.
fn analyze_rename_table(
    table_id: nusadb_core::TableId,
    schema: &str,
    current_name: &str,
    name: String,
    catalog: &dyn Catalog,
) -> Result<AlterTablePlan, Error> {
    enforce_system_catalog(&name, catalog)?;
    if catalog.lookup_table_in(schema, &name)?.is_some() {
        return Err(Error::TableExists {
            name: super::qualified_display(schema, &name),
        });
    }
    Ok(AlterTablePlan::RenameTable {
        table: table_id,
        from: format!("{schema}.{current_name}"),
        schema: schema.to_owned(),
        name,
    })
}

/// Resolve `ALTER TABLE ... ADD [CONSTRAINT name] <constraint>`. Only `PRIMARY KEY`/`UNIQUE`
/// are wired; `FOREIGN KEY` and `CHECK` (which need referential / predicate validation of the whole
/// table) are a follow-up. Every key column must exist; an unnamed constraint gets a generated name.
fn analyze_add_constraint(
    table: TableSchema,
    constraint: ast::TableConstraint,
    catalog: &dyn Catalog,
) -> Result<AlterTablePlan, Error> {
    // Auto-generated constraint names are keyed by the schema-qualified table name (bare for the
    // public schema, `schema.name` otherwise) so two same-named tables in different schemas do not
    // collide on `t_pkey`/`t_<col>_key`/… — exactly as `CREATE TABLE` qualifies its auto-names.
    let name_base = super::qualified_display(&table.schema, &table.name);
    let (name, columns, primary, nulls_not_distinct) = match constraint {
        ast::TableConstraint::PrimaryKey { name, columns } => (name, columns, true, false),
        ast::TableConstraint::Unique {
            name,
            columns,
            nulls_not_distinct,
        } => (name, columns, false, nulls_not_distinct),
        ast::TableConstraint::ForeignKey {
            name,
            columns,
            foreign_table,
            referred_columns,
            on_delete,
            on_update,
        } => {
            for column in &columns {
                find_column(&table.columns, column, &name_base)?;
            }
            let fk = ForeignKeySpec {
                name: name.unwrap_or_else(|| format!("{name_base}_fkey")),
                columns,
                parent_table: foreign_table,
                referred_columns,
                on_delete: referential_action(on_delete),
                on_update: referential_action(on_update),
            };
            return Ok(AlterTablePlan::AddForeignKey { table, fk });
        },
        ast::TableConstraint::Check {
            name,
            expr,
            predicate_sql,
        } => {
            let scope = single_table_scope(&table);
            validate_check_predicate(&expr, &scope, catalog)?;
            let predicate = analyze_expr(&expr, &scope, catalog, Some(ColumnType::Bool))?;
            return Ok(AlterTablePlan::AddCheck {
                name: name.unwrap_or_else(|| format!("{name_base}_check")),
                predicate_sql,
                predicate,
                table,
            });
        },
    };
    for column in &columns {
        find_column(&table.columns, column, &name_base)?;
    }
    let name = name.unwrap_or_else(|| {
        let suffix = if primary {
            "pkey".to_owned()
        } else {
            columns.join("_")
        };
        format!("{name_base}_{suffix}")
    });
    Ok(AlterTablePlan::AddUniqueConstraint {
        table,
        name,
        columns,
        primary,
        nulls_not_distinct,
    })
}

/// Resolve `ANALYZE [TABLE] name [(columns)]`: the table must exist; a column
/// list resolves to ordinals (rejecting duplicates and unknown names). A bare
/// `ANALYZE t` expands to every column.
pub(super) fn analyze_analyze(
    an: ast::Analyze,
    catalog: &dyn Catalog,
) -> Result<LogicalPlan, Error> {
    let ast::Analyze {
        schema,
        table: table_name,
        columns: requested,
    } = an;
    // Bare `ANALYZE` (no table) refreshes every user table, exactly like the `ANALYZE` clause of
    // `VACUUM ANALYZE` — same all-tables maintenance operation, so it carries no per-table resolution
    // or column list here (the executor enumerates the live tables under its snapshot).
    let Some(table_name) = table_name else {
        return Ok(LogicalPlan::AnalyzeAll);
    };
    let table = resolve_table(schema.as_deref(), &table_name, catalog)?;
    // ANALYZE reads every row and persists column values (MCV lists, histogram bounds), so it
    // needs the same SELECT privilege a plain read of the table would.
    super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Select)?;
    let columns = if requested.is_empty() {
        (0..table.columns.len()).collect()
    } else {
        let mut seen = HashSet::new();
        let mut indices = Vec::with_capacity(requested.len());
        for name in &requested {
            if !seen.insert(name.as_str()) {
                return Err(Error::DuplicateColumn { name: name.clone() });
            }
            let (index, _) = find_column(&table.columns, name, &table_name)?;
            indices.push(index);
        }
        indices
    };
    Ok(LogicalPlan::Analyze(AnalyzePlan { table, columns }))
}
