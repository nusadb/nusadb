//! DDL analyzers: CREATE TABLE, DROP TABLE, ALTER TABLE, ANALYZE.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === DDL ==================================================================

pub(super) fn analyze_create_table(
    ct: ast::CreateTable,
    catalog: &dyn Catalog,
) -> Result<CreateTablePlan, Error> {
    // A non-superuser must not squat a system-catalog name (e.g. pre-create `nusadb_policies`
    // before the engine does, forging what later reads as the policy catalog).
    enforce_system_catalog(&ct.name, catalog)?;
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
    // An unqualified CREATE targets the session's current schema; an explicit qualifier wins.
    let target_schema = ct
        .schema
        .clone()
        .unwrap_or_else(|| catalog.current_schema());
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
    Ok(CreateTablePlan {
        schema: target_schema,
        table: ct.name,
        columns: ct.columns,
        unique_constraints,
        foreign_keys,
        check_constraints,
        defaults,
        if_not_exists: ct.if_not_exists,
    })
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
                return Err(Error::Unsupported(
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
                return Err(Error::Unsupported(
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
            });
        }
        if column.unique {
            specs.push(UniqueConstraintSpec {
                name: format!("{name_base}_{}_key", column.name),
                columns: vec![column.name.clone()],
                primary: false,
            });
        }
    }

    // Table-level constraints.
    for constraint in &ct.constraints {
        match constraint {
            ast::TableConstraint::PrimaryKey { name, columns }
            | ast::TableConstraint::Unique { name, columns } => {
                let primary = matches!(constraint, ast::TableConstraint::PrimaryKey { .. });
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
                });
            },
            // Foreign keys / CHECK are resolved separately (`resolve_foreign_keys` /
            // `resolve_check_constraints`).
            ast::TableConstraint::ForeignKey { .. } | ast::TableConstraint::Check { .. } => {},
        }
    }

    if specs.iter().filter(|s| s.primary).count() > 1 {
        return Err(Error::Unsupported(
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
    let op = match at.action {
        // Row-level-security toggles are SQL-layer catalog changes, not column rewrites — they
        // produce a `SetRls` plan rather than an `AlterColumnOp`. Reserved to superusers, so a
        // non-superuser cannot lift its own RLS (e.g. `... DISABLE ROW LEVEL SECURITY`).
        ast::AlterTableAction::EnableRowLevelSecurity => {
            require_rls_admin(catalog, "enable row-level security on a table")?;
            return Ok(AlterTablePlan::SetRls {
                table: table.name,
                enabled: true,
            });
        },
        ast::AlterTableAction::DisableRowLevelSecurity => {
            require_rls_admin(catalog, "disable row-level security on a table")?;
            return Ok(AlterTablePlan::SetRls {
                table: table.name,
                enabled: false,
            });
        },
        // Trigger toggles are SQL-layer trigger-catalog changes, not column rewrites. The named
        // trigger's existence is checked by the executor against the trigger catalog (like
        // DROP TRIGGER); the table itself was resolved above.
        ast::AlterTableAction::EnableTrigger { name } => {
            return Ok(AlterTablePlan::SetTriggerEnabled {
                table: table.name,
                name,
                enabled: true,
            });
        },
        ast::AlterTableAction::DisableTrigger { name } => {
            return Ok(AlterTablePlan::SetTriggerEnabled {
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
                return Err(Error::Unsupported(
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
            return analyze_rename_table(table.id, &table.schema, name, catalog);
        },
    };
    Ok(AlterTablePlan::Apply { table, op })
}

/// Resolve `ALTER TABLE ... RENAME TO name`: the new name must be free and not collide with a system
/// catalog (which a rename would otherwise shadow), exactly as `CREATE TABLE` checks its name. The
/// rename stays within the table's own schema — `RENAME TO` never moves a table across schemas — so
/// the collision check looks in `schema`, not the search-path default.
fn analyze_rename_table(
    table_id: nusadb_core::TableId,
    schema: &str,
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
    let (name, columns, primary) = match constraint {
        ast::TableConstraint::PrimaryKey { name, columns } => (name, columns, true),
        ast::TableConstraint::Unique { name, columns } => (name, columns, false),
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
    })
}

/// Resolve `ANALYZE [TABLE] name [(columns)]`: the table must exist; a column
/// list resolves to ordinals (rejecting duplicates and unknown names). A bare
/// `ANALYZE t` expands to every column.
pub(super) fn analyze_analyze(
    an: ast::Analyze,
    catalog: &dyn Catalog,
) -> Result<AnalyzePlan, Error> {
    let ast::Analyze {
        table: table_name,
        columns: requested,
    } = an;
    // ANALYZE's name comes from `object_name` (public-only until NS3 opens it here).
    let table = resolve_table(None, &table_name, catalog)?;
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
    Ok(AnalyzePlan { table, columns })
}
