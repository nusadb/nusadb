//! DML analyzers: INSERT (+RETURNING, CREATE INDEX/SEQUENCE), UPDATE, DELETE.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === INSERT ===============================================================

#[allow(
    clippy::too_many_lines,
    reason = "flat INSERT analysis: column resolution, per-cell typing, RETURNING, ON CONFLICT, RLS"
)]
pub(super) fn analyze_insert(ins: ast::Insert, catalog: &dyn Catalog) -> Result<InsertPlan, Error> {
    let overriding = ins.overriding;
    // The system-catalog namespace is reserved: a user INSERT into e.g. `nusadb_policies`
    // would forge a policy and bypass RLS entirely.
    enforce_system_catalog(&ins.table, catalog)?;
    // Resolve without the RLS refusal `resolve_table` applies: a non-superuser may INSERT rows its
    // policies' WITH CHECK admit, so RLS is enforced by the `rls_check` predicate below.
    // Not a base table: an auto-updatable view rewrites onto its base table; anything else is an
    // unknown relation.
    let Some(table) = super::lookup_table_ref(ins.schema.as_deref(), &ins.table, catalog)? else {
        if let Some(view) = resolve_updatable_view(ins.schema.as_deref(), &ins.table, catalog)? {
            return insert_through_view(ins, view, catalog);
        }
        return Err(Error::TableNotFound {
            name: super::qualified_display_opt(ins.schema.as_deref(), &ins.table),
        });
    };
    // `DEFAULT VALUES` names no target columns — every column is omitted and takes its DEFAULT.
    let targets = if matches!(ins.source, ast::InsertSource::DefaultValues) {
        Vec::new()
    } else {
        resolve_insert_columns(&ins, &table)?
    };
    // INSERT needs the privilege on the table, or a column-scoped INSERT grant on each target column.
    // (`DEFAULT VALUES` names no column, so it can only be satisfied table-wide.)
    let insert_columns: Vec<&str> = targets
        .iter()
        .filter_map(|&i| table.columns.get(i).map(|c| c.name.as_str()))
        .collect();
    super::dcl::require_column_privilege(catalog, &table, &insert_columns, ast::Privilege::Insert)?;
    // RETURNING projects the inserted rows: resolve it against the table's columns.
    let returning = analyze_returning(&ins.returning, &table, catalog)?;
    // `ON CONFLICT`: `DO NOTHING` skips conflicting rows; `DO UPDATE` upserts the existing
    // row. Resolved against the target table (DO UPDATE also resolves the arbiter + EXCLUDED scope).
    let on_conflict = match ins.on_conflict {
        None => None,
        Some(conflict) => Some(analyze_on_conflict(conflict, &table, catalog)?),
    };
    let target_columns: Vec<ColumnDef> = targets
        .iter()
        .filter_map(|&index| table.columns.get(index).cloned())
        .collect();
    // For each target column, its user-defined composite type name (if it is a composite column).
    // A composite column stores as `TEXT`, so a `ROW(...)` / text value is coerced to its type here.
    let mut composite_targets: Vec<Option<String>> = Vec::with_capacity(target_columns.len());
    for column in &target_columns {
        composite_targets.push(catalog.lookup_composite_column(
            &table.schema,
            &table.name,
            &column.name,
        )?);
    }

    let source = match ins.source {
        ast::InsertSource::Values(rows_vec) => {
            let mut rows = Vec::with_capacity(rows_vec.len());
            for row in rows_vec {
                if row.len() != target_columns.len() {
                    return Err(Error::ArityMismatch {
                        context: "INSERT VALUES".to_owned(),
                        expected: target_columns.len(),
                        found: row.len(),
                    });
                }
                let mut typed_row = Vec::with_capacity(row.len());
                for ((value, column), composite) in
                    row.iter().zip(&target_columns).zip(&composite_targets)
                {
                    // A `None` cell is an explicit `DEFAULT`: leave it unresolved so the executor
                    // fills it from the column's default/serial/NULL, exactly like an omitted column.
                    let typed = match value {
                        // A composite column takes a `ROW(...)` / text value as its composite type;
                        // wrap it in a named cast so the composite construction/parse path types it.
                        // A bare `NULL` whole value is left alone so the ordinary NOT NULL check and
                        // NULL storage apply.
                        Some(expr) => match composite {
                            Some(type_name) if !super::typecheck::is_bare_null(expr) => {
                                let wrapped = ast::Expr::CastNamed {
                                    expr: Box::new(expr.clone()),
                                    type_name: type_name.clone(),
                                    try_cast: false,
                                };
                                Some(analyze_insert_value(&wrapped, column, catalog)?)
                            },
                            _ => Some(analyze_insert_value(expr, column, catalog)?),
                        },
                        None => None,
                    };
                    typed_row.push(typed);
                }
                rows.push(typed_row);
            }
            InsertSource::Values(rows)
        },
        // INSERT ... SELECT: analyze the subquery, then check that its output columns match
        // the target columns one-for-one (arity + per-column assignability).
        ast::InsertSource::Select(select) => {
            let plan = analyze_select(*select, catalog)?;
            if plan.projection.len() != target_columns.len() {
                return Err(Error::ArityMismatch {
                    context: "INSERT ... SELECT".to_owned(),
                    expected: target_columns.len(),
                    found: plan.projection.len(),
                });
            }
            for (proj, column) in plan.projection.iter().zip(&target_columns) {
                if !assignable(column.ty, proj.expr.ty) {
                    return Err(Error::TypeMismatch {
                        context: format!("INSERT ... SELECT into column `{}`", column.name),
                        expected: column.ty,
                        found: proj.expr.ty,
                    });
                }
            }
            InsertSource::Select(Box::new(plan))
        },
        // `DEFAULT VALUES` → a single empty row; with no target columns, the executor fills every
        // column from its DEFAULT (or NULL / NOT-NULL error when there is none).
        ast::InsertSource::DefaultValues => InsertSource::Values(vec![Vec::new()]),
    };

    // Row-level security: a non-superuser's inserted rows must satisfy the INSERT/ALL policies'
    // WITH CHECK (falling back to USING). Default-deny FALSE when no policy grants the insert, so an
    // RLS-enabled table with no INSERT policy rejects every non-superuser row.
    let rls_check = if !catalog.is_superuser() && catalog.rls_enabled(&table.schema, &table.name)? {
        Some(build_rls_check_predicate(
            &table.schema,
            &table.name,
            ast::PolicyCommand::Insert,
            &single_table_scope(&table),
            catalog,
        )?)
    } else {
        None
    };
    Ok(InsertPlan {
        table,
        columns: targets,
        source,
        returning,
        rls_check,
        // Set by `insert_through_view` when the target is a view WITH CHECK OPTION; a direct INSERT
        // has no view check.
        view_check: None,
        overriding,
        on_conflict,
    })
}

/// Resolve an `ON CONFLICT` clause against the target `table`. `DO NOTHING` needs no further
/// resolution (the executor checks every `PRIMARY KEY`/`UNIQUE` constraint); `DO UPDATE` resolves
/// the conflict arbiter and type-checks the `SET` assignments + optional `WHERE` against the combined
/// `[target ++ EXCLUDED]` scope.
fn analyze_on_conflict(
    conflict: ast::OnConflict,
    table: &TableSchema,
    catalog: &dyn Catalog,
) -> Result<OnConflictPlan, Error> {
    let ast::ConflictAction::DoUpdate {
        assignments,
        filter,
    } = conflict.action
    else {
        // DO NOTHING applies to any unique conflict, so a target is optional — but a stated one is
        // resolved and validated (a bad arbiter is rejected even when no row collides, as the
        // reference engine does).
        return Ok(OnConflictPlan::DoNothing {
            target: resolve_conflict_target(conflict.target, table)?,
        });
    };
    let Some(target) = resolve_conflict_target(conflict.target, table)? else {
        return Err(Error::InvalidStatement(
            "ON CONFLICT DO UPDATE requires a conflict target — \
             `ON CONFLICT (columns)` or `ON CONFLICT ON CONSTRAINT name`"
                .to_owned(),
        ));
    };
    // The combined scope: the existing row's columns (ordinals `[0, n)`) plus the proposed row as
    // `EXCLUDED` (ordinals `[n, 2n)`). `EXCLUDED` is reachable only via its qualifier, so a bare
    // column in the SET/WHERE refers to the existing (target) row.
    let scope = upsert_scope(table);
    let mut typed = Vec::with_capacity(assignments.len());
    let mut seen = HashSet::new();
    for assignment in assignments {
        let (index, column) = find_column(&table.columns, &assignment.column, &table.name)?;
        if !seen.insert(index) {
            return Err(Error::DuplicateColumn {
                name: assignment.column.clone(),
            });
        }
        let value = analyze_expr(&assignment.value, &scope, catalog, Some(column.ty))?;
        check_assignable(column, &value)?;
        reject_conflict_subquery(&value)?;
        typed.push((index, value));
    }
    let filter = match filter {
        None => None,
        Some(predicate) => {
            let typed = analyze_expr(&predicate, &scope, catalog, Some(ColumnType::Bool))?;
            if typed.ty != ColumnType::Bool {
                return Err(Error::TypeMismatch {
                    context: "ON CONFLICT ... WHERE".to_owned(),
                    expected: ColumnType::Bool,
                    found: typed.ty,
                });
            }
            reject_conflict_subquery(&typed)?;
            Some(typed)
        },
    };
    Ok(OnConflictPlan::DoUpdate {
        target,
        assignments: typed,
        filter,
    })
}

/// Resolve an `ON CONFLICT` conflict target to a [`ConflictArbiter`]. `ON CONFLICT (cols)` maps each
/// column name to its ordinal (the executor still checks those ordinals form a declared key); `ON
/// CONFLICT ON CONSTRAINT name` carries the name through (the executor looks it up). `None` (no
/// target) yields `None` — valid only for `DO NOTHING`.
fn resolve_conflict_target(
    target: Option<ast::ConflictTarget>,
    table: &TableSchema,
) -> Result<Option<ConflictArbiter>, Error> {
    match target {
        None => Ok(None),
        Some(ast::ConflictTarget::Columns(cols)) => {
            let mut ordinals = Vec::with_capacity(cols.len());
            for name in &cols {
                let (index, _) = find_column(&table.columns, name, &table.name)?;
                ordinals.push(index);
            }
            Ok(Some(ConflictArbiter::Columns(ordinals)))
        },
        Some(ast::ConflictTarget::Constraint(name)) => Ok(Some(ConflictArbiter::Constraint(name))),
    }
}

/// The combined `[target ++ EXCLUDED]` scope for an `ON CONFLICT DO UPDATE`: the target
/// table's columns at ordinals `[0, n)` and a second copy qualified `excluded` (reachable only as
/// `excluded.col`) at `[n, 2n)`, matching the row the executor evaluates assignments against.
fn upsert_scope(table: &TableSchema) -> Vec<ScopedColumn> {
    let mut scope = single_table_scope(table);
    scope.extend(table.columns.iter().map(|def| ScopedColumn {
        qualifier: "excluded".to_owned(),
        def: def.clone(),
        qualified_only: true,
        composite_type: None,
        enum_type: None,
        // The `EXCLUDED` pseudo-relation of `ON CONFLICT` is the proposed row, not a base-table read.
        select_granted: true,
    }));
    scope
}

/// Reject a subquery in a `DO UPDATE` assignment/predicate: the executor evaluates them
/// against an in-memory combined row, with no correlated-subquery machinery.
fn reject_conflict_subquery(expr: &TypedExpr) -> Result<(), Error> {
    if crate::executor::ops::contains_subquery(expr) {
        return Err(Error::Unsupported(
            "a subquery in ON CONFLICT DO UPDATE is not supported".to_owned(),
        ));
    }
    Ok(())
}

/// Resolve a `RETURNING` clause against the affected table's columns. The scope is the
/// affected row (every column of `table`, in table order), so `RETURNING *` and `RETURNING col`
/// resolve like a single-table projection. Aggregates are not meaningful over a per-row `RETURNING`
/// and are rejected. An empty clause yields an empty projection (the caller returns a row count).
pub(super) fn analyze_returning(
    returning: &[ast::SelectItem],
    table: &TableSchema,
    catalog: &dyn Catalog,
) -> Result<Vec<Projection>, Error> {
    if returning.is_empty() {
        return Ok(Vec::new());
    }
    let scope = scope_of(table);
    let mut aggregates = Vec::new();
    let source_len = scope.len();
    let projection = analyze_projection(
        returning.to_vec(),
        &scope,
        catalog,
        &mut aggregates,
        source_len,
    )?;
    if !aggregates.is_empty() {
        return Err(Error::InvalidGrouping(
            "aggregate functions are not allowed in RETURNING".to_owned(),
        ));
    }
    Ok(projection)
}

/// Resolve a `CREATE INDEX` against the catalog: the target table must exist, and every key
/// and `INCLUDE` column must be a column of it. The access method defaults to `BTree`; `USING hnsw`
/// builds a vector index over a single `VECTOR(n)` column instead. Builds an [`IndexDef`]
/// (and, for `hnsw`, a [`VectorIndexSpec`]) for the executor.
pub(super) fn analyze_create_index(
    ci: ast::CreateIndex,
    catalog: &dyn Catalog,
) -> Result<CreateIndexPlan, Error> {
    use nusadb_core::engine::{IndexDef, IndexKind};

    enforce_system_catalog(&ci.table, catalog)?;
    let table = super::lookup_table_ref(ci.table_schema.as_deref(), &ci.table, catalog)?
        .ok_or_else(|| Error::TableNotFound {
            name: super::qualified_display_opt(ci.table_schema.as_deref(), &ci.table),
        })?;
    // Adding an index restructures the table — the owner's right, like ALTER and DROP.
    super::dcl::require_table_ownership(catalog, &table, "create an index on")?;
    // Every plain key column and every INCLUDE column must exist (find_column reports
    // ColumnNotFound). Expression keys are validated below by type-checking them against the table.
    for column in ci.columns.iter().chain(&ci.include) {
        find_column(&table.columns, column, &table.name)?;
    }
    // Functional/expression keys and the partial predicate are re-parsed and evaluated per row on
    // the write path against a MINIMAL catalog (no function/table lookup) — so they are validated
    // here against that same empty catalog, not the real one. This keeps CREATE and maintenance
    // consistent: a key/predicate the write path cannot resolve (e.g. a SQL `CREATE FUNCTION` UDF,
    // whose lookup the write path lacks) is rejected LOUDLY now rather than silently producing an
    // index that maintenance can never populate. Built-in functions
    // and Rust UDFs resolve without a catalog, so they still pass.
    let index_catalog = IndexExprCatalog;
    for expr_sql in &ci.key_exprs {
        validate_index_key_expr(expr_sql, &table, &index_catalog)?;
    }
    // Partial-index predicate: must be a boolean, subquery-free expression over the table's columns
    // (same contract as a CHECK predicate).
    if let Some(pred) = &ci.predicate {
        let expr = crate::parser::parse_expression(pred)?;
        let typed = analyze_expr(
            &expr,
            &single_table_scope(&table),
            &index_catalog,
            Some(ColumnType::Bool),
        )?;
        if typed.ty != ColumnType::Bool {
            return Err(Error::TypeMismatch {
                context: "partial index predicate".to_owned(),
                expected: ColumnType::Bool,
                found: typed.ty,
            });
        }
        if crate::executor::ops::contains_subquery(&typed) {
            return Err(Error::Unsupported(
                "a partial index predicate may not contain a subquery".to_owned(),
            ));
        }
    }
    let vector = match ci.using.as_deref() {
        Some("hnsw") => Some(analyze_hnsw_index(&ci, &table)?),
        _ => None,
    };
    Ok(CreateIndexPlan {
        def: IndexDef {
            name: ci.name,
            table: table.id,
            columns: ci.columns,
            key_exprs: ci.key_exprs,
            predicate: ci.predicate,
            include: ci.include,
            kind: IndexKind::BTree,
            unique: ci.unique,
        },
        vector,
        if_not_exists: ci.if_not_exists,
    })
}

/// The minimal catalog an index key/predicate is validated against at `CREATE INDEX` — it exposes
/// no tables and no functions, exactly matching what the executor's write-path re-analysis sees
/// (a row-only scope with no catalog). A SQL `CREATE FUNCTION` UDF in a key/predicate therefore
/// fails to resolve here and is rejected loudly, rather than being accepted and then silently
/// unmaintainable. Built-in scalar functions and Rust UDFs resolve
/// from static registries without a catalog, so they still validate.
struct IndexExprCatalog;

impl Catalog for IndexExprCatalog {
    fn lookup_table(&self, _name: &str) -> Result<Option<TableSchema>, Error> {
        Ok(None)
    }
}

/// Validate a functional/expression index key: it must parse, type-check against `table`'s columns,
/// and contain no aggregate (rejected by the `None` aggregate sink) or subquery — the executor
/// re-parses and evaluates it per row against a row-only scope where neither resolves.
fn validate_index_key_expr(
    expr_sql: &str,
    table: &TableSchema,
    catalog: &dyn Catalog,
) -> Result<(), Error> {
    let expr = crate::parser::parse_expression(expr_sql)?;
    let typed = analyze_expr(&expr, &single_table_scope(table), catalog, None)?;
    if crate::executor::ops::contains_subquery(&typed) {
        return Err(Error::Unsupported(
            "a functional index key may not contain a subquery".to_owned(),
        ));
    }
    Ok(())
}

/// Validate a `USING hnsw` vector index: exactly one `VECTOR(n)` key column, no `UNIQUE`,
/// no `INCLUDE`. Returns the resolved [`VectorIndexSpec`] the executor records in the vector-index
/// catalog.
fn analyze_hnsw_index(
    ci: &ast::CreateIndex,
    table: &TableSchema,
) -> Result<VectorIndexSpec, Error> {
    if ci.unique {
        return Err(Error::Unsupported("a UNIQUE hnsw vector index".to_owned()));
    }
    if !ci.include.is_empty() {
        return Err(Error::Unsupported(
            "INCLUDE columns on an hnsw vector index".to_owned(),
        ));
    }
    let [column] = ci.columns.as_slice() else {
        return Err(Error::InvalidStatement(
            "an hnsw vector index must be over exactly one VECTOR column".to_owned(),
        ));
    };
    let (column_ordinal, def) = find_column(&table.columns, column, &table.name)?;
    let ColumnType::Vector(dim) = def.ty else {
        return Err(Error::InvalidStatement(
            "an hnsw index requires a VECTOR(n) column".to_owned(),
        ));
    };
    // The operator class picks the distance metric the graph is built under. Defaulting to cosine
    // keeps every index written before this was settable meaning what it already meant.
    let metric = match &ci.operator_class {
        None => crate::hnsw::Metric::Cosine,
        Some(name) => crate::hnsw::Metric::from_operator_class(name).ok_or_else(|| {
            Error::ObjectNotFound(format!(
                "operator class `{name}` on an hnsw index (expected one of `vector_l2_ops`, \
                 `vector_cosine_ops`, `vector_ip_ops`, `vector_l1_ops`)"
            ))
        })?,
    };
    Ok(VectorIndexSpec {
        name: ci.name.clone(),
        table: table.name.clone(),
        column: column.clone(),
        column_ordinal,
        dim: dim as usize,
        metric,
    })
}

/// Fold a `CREATE SEQUENCE` statement's options into a [`SequenceDef`].
///
/// Options must be integer constants (the realistic surface); a non-constant is rejected. Unspecified
/// bounds default to a standard ascending sequence: `MINVALUE 1`, `MAXVALUE` of `i64::MAX`,
/// `START` = the minimum, `INCREMENT 1`, no cycle. `CACHE` is accepted and ignored (the engine has
/// no cache concept). Descending sequences must give explicit bounds.
pub(super) fn analyze_create_sequence(
    cs: ast::CreateSequence,
) -> Result<CreateSequencePlan, Error> {
    use nusadb_core::engine::SequenceDef;

    let mut increment = 1i64;
    let mut min_value: Option<i64> = None;
    let mut max_value: Option<i64> = None;
    let mut start: Option<i64> = None;
    let mut cycle = false;
    for option in &cs.options {
        match option {
            ast::SequenceOption::Increment(e) => increment = const_i64(e)?,
            ast::SequenceOption::MinValue(Some(e)) => min_value = Some(const_i64(e)?),
            ast::SequenceOption::MaxValue(Some(e)) => max_value = Some(const_i64(e)?),
            ast::SequenceOption::Start(e) => start = Some(const_i64(e)?),
            ast::SequenceOption::Cycle(b) => cycle = *b,
            // NO MINVALUE / NO MAXVALUE → fall back to the default bound; CACHE is a no-op.
            ast::SequenceOption::MinValue(None)
            | ast::SequenceOption::MaxValue(None)
            | ast::SequenceOption::Cache(_) => {},
            // RESTART repositions an existing counter — it belongs to `ALTER SEQUENCE`, not `CREATE`.
            ast::SequenceOption::Restart(_) => {
                return Err(Error::InvalidStatement(
                    "RESTART is not valid in CREATE SEQUENCE".to_owned(),
                ));
            },
        }
    }
    let min_value = min_value.unwrap_or(1);
    let max_value = max_value.unwrap_or(i64::MAX);
    let start = start.unwrap_or(min_value);
    Ok(CreateSequencePlan {
        def: SequenceDef {
            name: cs.name,
            start,
            increment,
            min_value,
            max_value,
            cycle,
        },
        if_not_exists: cs.if_not_exists,
    })
}

/// Lower `ALTER SEQUENCE [IF EXISTS] name <options>` to a partial [`SequenceChange`]. `NO MINVALUE`
/// / `NO MAXVALUE` reset to the same defaults `CREATE` uses; `CACHE` is a no-op; `RESTART` becomes
/// the counter reposition the engine applies.
pub(super) fn analyze_alter_sequence(a: ast::AlterSequence) -> Result<AlterSequencePlan, Error> {
    use nusadb_core::engine::{SequenceChange, SequenceRestart};

    let mut change = SequenceChange::default();
    for option in &a.options {
        match option {
            ast::SequenceOption::Increment(e) => change.increment = Some(const_i64(e)?),
            ast::SequenceOption::MinValue(Some(e)) => change.min_value = Some(const_i64(e)?),
            ast::SequenceOption::MaxValue(Some(e)) => change.max_value = Some(const_i64(e)?),
            ast::SequenceOption::Start(e) => change.start = Some(const_i64(e)?),
            ast::SequenceOption::Cycle(b) => change.cycle = Some(*b),
            ast::SequenceOption::Restart(Some(e)) => {
                change.restart = Some(SequenceRestart::To(const_i64(e)?));
            },
            ast::SequenceOption::Restart(None) => change.restart = Some(SequenceRestart::ToStart),
            // `NO MINVALUE` / `NO MAXVALUE` reset to the engine's default bounds (as `CREATE` does);
            // `CACHE` has no effect.
            ast::SequenceOption::MinValue(None) => change.min_value = Some(1),
            ast::SequenceOption::MaxValue(None) => change.max_value = Some(i64::MAX),
            ast::SequenceOption::Cache(_) => {},
        }
    }
    Ok(AlterSequencePlan {
        name: a.name,
        if_exists: a.if_exists,
        change,
    })
}

/// Evaluate a sequence-option expression to a constant `i64` — an integer literal or its negation.
pub(super) fn const_i64(expr: &ast::Expr) -> Result<i64, Error> {
    match expr {
        ast::Expr::Literal(ast::Value::Int(n)) => Ok(*n),
        ast::Expr::Unary {
            op: ast::UnaryOp::Negate,
            expr,
        } => match expr.as_ref() {
            ast::Expr::Literal(ast::Value::Int(n)) => n.checked_neg().ok_or_else(|| {
                Error::InvalidParameterValue("sequence option value out of range".to_owned())
            }),
            _ => Err(Error::InvalidStatement(
                "sequence option must be an integer constant".to_owned(),
            )),
        },
        _ => Err(Error::InvalidStatement(
            "sequence option must be an integer constant".to_owned(),
        )),
    }
}

pub(super) fn resolve_insert_columns(
    ins: &ast::Insert,
    table: &TableSchema,
) -> Result<Vec<usize>, Error> {
    if ins.columns.is_empty() {
        return Ok((0..table.columns.len()).collect());
    }
    let mut seen = HashSet::new();
    let mut indices = Vec::with_capacity(ins.columns.len());
    for name in &ins.columns {
        if !seen.insert(name.as_str()) {
            return Err(Error::DuplicateColumn { name: name.clone() });
        }
        let (index, _) = find_column(&table.columns, name, &table.name)?;
        indices.push(index);
    }
    Ok(indices)
}

pub(super) fn analyze_insert_value(
    value: &ast::Expr,
    column: &ColumnDef,
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    // `INSERT ... VALUES` expressions cannot reference columns: empty scope.
    let typed = analyze_expr(value, &[], catalog, Some(column.ty))?;
    if !column.nullable && is_null_literal(&typed) {
        return Err(Error::NotNullViolation {
            column: column.name.clone(),
        });
    }
    // A bare string literal is "unknown"-typed and adopts an integer / float / boolean target, so
    // `INSERT INTO t(int_col) VALUES ('123')` stores `123` and `VALUES ('xyz')` loud-rejects at
    // evaluation. (A genuinely TEXT-typed value into a non-text column still stays a real mismatch;
    // bit / temporal / … columns already accept a text literal through their own length-checked
    // assignment path, so they are left untouched here.)
    let typed = super::expr::coerce_insert_literal(typed, column.ty);
    check_assignable(column, &typed)?;
    Ok(typed)
}

// === UPDATE / DELETE ======================================================

/// The transitive inheritance/partition descendants an `UPDATE`/`DELETE` on `table` must also touch,
/// or empty when it targets nothing extra: an `ONLY` write, a database with no inheritance, or a
/// non-parent table (including a view). Gated on the cheap `any_inheritance` probe so an ordinary
/// write pays almost nothing.
fn write_descendants(
    only: bool,
    schema: Option<&str>,
    table: &str,
    catalog: &dyn Catalog,
) -> Result<Vec<String>, Error> {
    if only || !catalog.any_inheritance()? {
        return Ok(Vec::new());
    }
    // Only a real base table can be an inheritance/partition parent; a view resolves to `None` here.
    let Some(resolved) = super::lookup_table_ref(schema, table, catalog)? else {
        return Ok(Vec::new());
    };
    catalog.inheritance_descendants(&resolved.name)
}

/// Defensive guard: `analyze_update`/`analyze_delete` must only ever be handed a single-table write —
/// either an `ONLY` write or one on a non-parent. A non-`ONLY` write on a parent-with-descendants is
/// routed through [`analyze_update_stmt`]/[`analyze_delete_stmt`], which build the per-descendant
/// sub-plans; if one reaches here directly it would silently touch only the parent, so refuse it.
fn reject_inheritance_write(
    table: &str,
    only: bool,
    verb: &str,
    catalog: &dyn Catalog,
) -> Result<(), Error> {
    if !only && catalog.any_inheritance()? && !catalog.inheritance_descendants(table)?.is_empty() {
        return Err(Error::Internal(format!(
            "{verb} on inheritance parent \"{table}\" reached single-table analysis without \
             propagation (should route through analyze_{}_stmt)",
            verb.to_ascii_lowercase()
        )));
    }
    Ok(())
}

/// Analyze an `UPDATE`, propagating to inheritance/partition descendants when the target is a parent
/// and `ONLY` was not given: the write becomes the parent's own (`ONLY`) update plus one
/// `UPDATE ONLY <descendant>` sub-plan per transitive descendant, each analyzed in its own right (so
/// each table's own row-level security and column resolution apply). An ordinary/`ONLY`/non-parent
/// update analyzes as a single table.
pub(super) fn analyze_update_stmt(
    upd: ast::Update,
    catalog: &dyn Catalog,
) -> Result<UpdatePlan, Error> {
    let descendants = write_descendants(upd.only, upd.schema.as_deref(), &upd.table, catalog)?;
    if descendants.is_empty() {
        return analyze_update(upd, catalog);
    }
    let mut parent = upd.clone();
    parent.only = true;
    let mut plan = analyze_update(parent, catalog)?;
    for descendant in descendants {
        let mut sub = upd.clone();
        sub.only = true;
        sub.schema = None;
        sub.table = descendant;
        plan.propagate.push(analyze_update(sub, catalog)?);
    }
    Ok(plan)
}

/// Analyze a `DELETE`, propagating to inheritance/partition descendants — the `DELETE` counterpart of
/// [`analyze_update_stmt`].
pub(super) fn analyze_delete_stmt(
    del: ast::Delete,
    catalog: &dyn Catalog,
) -> Result<DeletePlan, Error> {
    let descendants = write_descendants(del.only, del.schema.as_deref(), &del.table, catalog)?;
    if descendants.is_empty() {
        return analyze_delete(del, catalog);
    }
    let mut parent = del.clone();
    parent.only = true;
    let mut plan = analyze_delete(parent, catalog)?;
    for descendant in descendants {
        let mut sub = del.clone();
        sub.only = true;
        sub.schema = None;
        sub.table = descendant;
        plan.propagate.push(analyze_delete(sub, catalog)?);
    }
    Ok(plan)
}

pub(super) fn analyze_update(upd: ast::Update, catalog: &dyn Catalog) -> Result<UpdatePlan, Error> {
    // The system-catalog namespace is reserved: a user UPDATE of e.g. `nusadb_policies`
    // could widen a policy's USING predicate and bypass RLS.
    enforce_system_catalog(&upd.table, catalog)?;
    // Resolve without the RLS refusal `resolve_table` applies: a non-superuser may UPDATE the rows
    // its policies' USING grant (folded into `filter`) to values its WITH CHECK admit (`rls_check`).
    let Some(table) = super::lookup_table_ref(upd.schema.as_deref(), &upd.table, catalog)? else {
        if let Some(view) = resolve_updatable_view(upd.schema.as_deref(), &upd.table, catalog)? {
            let view_name = upd.table.clone();
            return update_through_view(upd, view, &view_name, catalog);
        }
        return Err(Error::TableNotFound {
            name: super::qualified_display_opt(upd.schema.as_deref(), &upd.table),
        });
    };
    reject_inheritance_write(&table.name, upd.only, "UPDATE", catalog)?;
    // UPDATE needs the privilege on the table, or a column-scoped UPDATE grant on each SET column.
    let update_columns: Vec<&str> = upd.assignments.iter().map(|a| a.column.as_str()).collect();
    super::dcl::require_column_privilege(catalog, &table, &update_columns, ast::Privilege::Update)?;
    // A `WHERE` or `RETURNING` reads the target's rows to decide what to change or to hand back, so
    // it needs SELECT too. An unconditional `UPDATE t SET c = 1` reads nothing and needs only
    // UPDATE — the standard distinction, and the one that keeps a write-only role write-only.
    // (A read here requires table-wide SELECT; a column-scoped SELECT does not yet satisfy an
    // UPDATE/DELETE predicate — a fail-closed limitation, not a leak.)
    if upd.filter.is_some() || !upd.returning.is_empty() {
        super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Select)?;
    }
    // RETURNING projects the updated rows, resolved against the table's (post-update) columns.
    let returning = analyze_returning(&upd.returning, &table, catalog)?;
    // UPDATE ... FROM: resolve a single named FROM table and extend the scope with it, so the
    // SET values and WHERE may reference its columns (at ordinals `target_width + j` of the
    // concatenated `target ++ from` row the executor evaluates against).
    let from = upd
        .from
        .map(|f| resolve_update_from(f, catalog))
        .transpose()?;
    // When the target is aliased (`UPDATE t AS x`), the SET values and WHERE reference it by the
    // alias (which shadows the table name), so build the scope under the alias qualifier.
    let mut scope = upd
        .alias
        .as_deref()
        .map_or_else(|| scope_of(&table), |alias| scope_of_aliased(&table, alias));
    let mut from_table: Option<TableSchema> = None;
    let mut from_plan: Option<Box<SelectPlan>> = None;
    if let Some((schema, qualifier, plan)) = from {
        // A FROM source is a de-facto join: like SELECT's RLS+JOIN refusal, a non-superuser must not
        // read an RLS-protected source table (its rows would otherwise leak through the SET values or
        // WHERE predicate). Fail closed (deep-gate security). A derived source's schema name is
        // its alias; in the unlikely case that alias collides with an RLS-protected table name the
        // guard merely over-rejects (fail closed) — never a leak.
        if !catalog.is_superuser() && catalog.rls_enabled(&schema.schema, &schema.name)? {
            return Err(Error::Unsupported(format!(
                "row-level security on `{}` combined with UPDATE ... FROM is not yet supported",
                schema.name
            )));
        }
        scope.extend(schema.columns.iter().map(|def| ScopedColumn {
            qualifier: qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
            // Composite field access on a secondary UPDATE/DELETE source is out of first-cut scope.
            composite_type: None,
            // A secondary source has already passed a table-wide SELECT check (`resolve_aux_relation`),
            // so its columns are readable; column-scoped SELECT does not gate a join source here.
            select_granted: true,
            enum_type: None,
        }));
        from_plan = plan.map(Box::new);
        from_table = Some(schema);
    }
    let mut assignments = Vec::with_capacity(upd.assignments.len());
    let mut seen = HashSet::new();
    for assignment in upd.assignments {
        let (index, column) = find_column(&table.columns, &assignment.column, &table.name)?;
        if !seen.insert(index) {
            return Err(Error::DuplicateColumn {
                name: assignment.column.clone(),
            });
        }
        let value = analyze_expr(&assignment.value, &scope, catalog, Some(column.ty))?;
        if !column.nullable && is_null_literal(&value) {
            return Err(Error::NotNullViolation {
                column: column.name.clone(),
            });
        }
        check_assignable(column, &value)?;
        assignments.push(Assignment {
            column: index,
            value,
        });
    }
    let mut filter = analyze_predicate(upd.filter, &scope, catalog)?;
    // Row-level security: a non-superuser may update only the rows the UPDATE/ALL policies'
    // USING grant (AND-injected into the filter, like DELETE), and only to values their WITH CHECK
    // admit (`rls_check`, evaluated against each post-update row by the executor). Default-deny
    // FALSE on both sides when no policy applies.
    let rls_check = if !catalog.is_superuser() && catalog.rls_enabled(&table.schema, &table.name)? {
        let using = build_rls_predicate(
            &table.schema,
            &table.name,
            ast::PolicyCommand::Update,
            &scope,
            catalog,
        )?;
        filter = Some(match filter {
            None => using,
            Some(existing) => and_exprs(existing, using),
        });
        Some(build_rls_check_predicate(
            &table.schema,
            &table.name,
            ast::PolicyCommand::Update,
            &scope,
            catalog,
        )?)
    } else {
        None
    };
    Ok(UpdatePlan {
        table,
        from: from_table,
        from_plan,
        assignments,
        filter,
        returning,
        rls_check,
        // Set by `update_through_view` when the target is a view WITH CHECK OPTION; a direct UPDATE
        // has no view check.
        view_check: None,
        // Populated by `analyze_update_stmt` for a non-`ONLY` update on an inheritance/partition
        // parent; a single-table analysis carries none.
        propagate: Vec::new(),
    })
}

/// Resolve an `UPDATE ... FROM` / `DELETE ... USING` clause to its single source relation: the
/// schema, the qualifier (alias, else table name) its columns are referenced by, and — for a derived
/// source (`(VALUES ...)` / `(SELECT ...)` / set operation) — the inlined plan that produces its rows
/// (`None` for a named table). A join (multiple comma sources) is rejected.
fn resolve_update_from(
    from: ast::FromClause,
    catalog: &dyn Catalog,
) -> Result<(TableSchema, String, Option<SelectPlan>), Error> {
    if !from.joins.is_empty() {
        return Err(Error::Unsupported(
            "UPDATE ... FROM / DELETE ... USING with a join is not yet supported (use a single \
             source)"
                .to_owned(),
        ));
    }
    let base = from.base;
    let qualifier = base.alias.clone().unwrap_or_else(|| base.name.clone());
    let (table, plan) = resolve_aux_relation(&base, catalog)?;
    Ok((table, qualifier, plan))
}

pub(super) fn analyze_delete(del: ast::Delete, catalog: &dyn Catalog) -> Result<DeletePlan, Error> {
    // The system-catalog namespace is reserved: a user DELETE on e.g. `nusadb_rls` would
    // switch row-level security off for the affected tables.
    enforce_system_catalog(&del.table, catalog)?;
    // Resolve the target without the RLS refusal `resolve_table` applies: a non-superuser may
    // DELETE the rows its policies grant, so RLS is enforced by injecting a predicate below rather
    // than refusing.
    let Some(table) = super::lookup_table_ref(del.schema.as_deref(), &del.table, catalog)? else {
        if let Some(view) = resolve_updatable_view(del.schema.as_deref(), &del.table, catalog)? {
            let view_name = del.table.clone();
            return delete_through_view(del, view, &view_name, catalog);
        }
        return Err(Error::TableNotFound {
            name: super::qualified_display_opt(del.schema.as_deref(), &del.table),
        });
    };
    reject_inheritance_write(&table.name, del.only, "DELETE", catalog)?;
    super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Delete)?;
    // As for UPDATE: a `WHERE` or `RETURNING` reads rows, so it additionally needs SELECT.
    if del.filter.is_some() || !del.returning.is_empty() {
        super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Select)?;
    }
    // RETURNING projects the deleted rows, resolved against the table's columns.
    let returning = analyze_returning(&del.returning, &table, catalog)?;
    // DELETE ... USING: resolve a single named source table and extend the scope, so the
    // WHERE may reference its columns (at ordinals `target_width + j` of `target ++ using`). Reuses
    // the same single-named-table resolution as UPDATE ... FROM.
    let using = del
        .using
        .map(|u| resolve_update_from(u, catalog))
        .transpose()?;
    // When the target is aliased (`DELETE FROM t AS x`), the WHERE (and USING join) reference it by
    // the alias, which shadows the table name — so build the target scope under the alias qualifier,
    // exactly as `UPDATE t AS x` does.
    let mut scope = del
        .alias
        .as_deref()
        .map_or_else(|| scope_of(&table), |alias| scope_of_aliased(&table, alias));
    let mut using_table: Option<TableSchema> = None;
    let mut using_plan: Option<Box<SelectPlan>> = None;
    if let Some((schema, qualifier, plan)) = using {
        // A USING source is a de-facto join: like SELECT's RLS+JOIN refusal, a non-superuser must not
        // read an RLS-protected source table (its rows would otherwise leak through the WHERE
        // predicate). Fail closed (deep-gate security). A derived source's schema name is its
        // alias; in the unlikely case that alias collides with an RLS-protected table name the guard
        // merely over-rejects (fail closed) — never a leak.
        if !catalog.is_superuser() && catalog.rls_enabled(&schema.schema, &schema.name)? {
            return Err(Error::Unsupported(format!(
                "row-level security on `{}` combined with DELETE ... USING is not yet supported",
                schema.name
            )));
        }
        scope.extend(schema.columns.iter().map(|def| ScopedColumn {
            qualifier: qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
            // Composite field access on a secondary UPDATE/DELETE source is out of first-cut scope.
            composite_type: None,
            // A secondary source has already passed a table-wide SELECT check (`resolve_aux_relation`),
            // so its columns are readable; column-scoped SELECT does not gate a join source here.
            select_granted: true,
            enum_type: None,
        }));
        using_plan = plan.map(Box::new);
        using_table = Some(schema);
    }
    let mut filter = analyze_predicate(del.filter, &scope, catalog)?;
    // Row-level security: a non-superuser may only delete rows the DELETE/ALL policies grant.
    // DELETE has no WITH CHECK, so injecting the USING predicate is complete (default-deny FALSE
    // when no policy applies, like SELECT).
    if !catalog.is_superuser() && catalog.rls_enabled(&table.schema, &table.name)? {
        let policy = build_rls_predicate(
            &table.schema,
            &table.name,
            ast::PolicyCommand::Delete,
            &scope,
            catalog,
        )?;
        filter = Some(match filter {
            None => policy,
            Some(existing) => and_exprs(existing, policy),
        });
    }
    Ok(DeletePlan {
        table,
        using: using_table,
        using_plan,
        filter,
        returning,
        // Populated by `analyze_delete_stmt` for a non-`ONLY` delete on an inheritance/partition
        // parent; a single-table analysis carries none.
        propagate: Vec::new(),
    })
}

/// Analyze `MERGE INTO target USING source ON ... WHEN [NOT] MATCHED ...`. The target must be a plain
/// named table; the source may be a plain table OR a derived relation (`VALUES` / subquery / set
/// operation), resolved like `UPDATE ... FROM` / `DELETE ... USING` (a `LATERAL` source is rejected).
/// Every clause expression is type-checked against the combined `target ++ source` scope; a `WHEN
/// MATCHED` UPDATE assigns target columns, a `WHEN NOT MATCHED` INSERT fills target columns from
/// source values.
pub(super) fn analyze_merge(m: ast::Merge, catalog: &dyn Catalog) -> Result<MergePlan, Error> {
    enforce_system_catalog(&m.target.name, catalog)?;
    if m.target.subquery.is_some()
        || m.target.values.is_some()
        || m.target.set_op.is_some()
        || m.target.lateral
    {
        return Err(Error::InvalidStatement(
            "MERGE target must be a plain table".to_owned(),
        ));
    }
    let table = super::lookup_table_ref(m.target.schema.as_deref(), &m.target.name, catalog)?
        .ok_or_else(|| Error::TableNotFound {
            name: super::qualified_display_opt(m.target.schema.as_deref(), &m.target.name),
        })?;
    // The USING source may be a plain table OR a derived relation (`VALUES` / subquery / set
    // operation) — resolved uniformly, exactly as `UPDATE ... FROM` / `DELETE ... USING` do. A
    // derived source carries an inlined plan the executor materializes; a plain table has `None` and
    // is scanned. `LATERAL` stays unsupported (rejected inside `resolve_aux_relation`).
    let (source, source_plan) = resolve_aux_relation(&m.source, catalog)?;
    // MERGE always reads the target — the `ON` condition matches against its rows — and then writes
    // it in whichever ways its WHEN arms name. Require exactly those: SELECT unconditionally, plus
    // INSERT / UPDATE / DELETE per the arms present. Without this a role with no privileges at all
    // on the target could read it through the match and rewrite it through the actions.
    super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Select)?;
    for when in &m.whens {
        match when {
            ast::MergeWhen::NotMatched { .. } => {
                super::dcl::require_table_privilege(catalog, &table, ast::Privilege::Insert)?;
            },
            ast::MergeWhen::Matched { action, .. }
            | ast::MergeWhen::NotMatchedBySource { action, .. } => {
                let privilege = match action {
                    ast::MatchedAction::Delete => ast::Privilege::Delete,
                    ast::MatchedAction::Update { .. } => ast::Privilege::Update,
                };
                super::dcl::require_table_privilege(catalog, &table, privilege)?;
            },
        }
    }
    // Row-level security on the MERGE target is not yet wired (the matched UPDATE/DELETE side would
    // not enforce the policies' USING / WITH CHECK that a plain UPDATE/DELETE does). Reject rather
    // than silently bypass RLS for a non-superuser; a superuser bypasses RLS anyway.
    if !catalog.is_superuser() && catalog.rls_enabled(&table.schema, &table.name)? {
        return Err(Error::Unsupported(
            "MERGE on a row-level-security protected table is not yet supported".to_owned(),
        ));
    }
    // The USING source is scanned in full to drive the match — it is not filtered by RLS, so a
    // non-superuser could read every row of a row-level-security protected source through a matched
    // action's SET / search condition (the same leak class as UPDATE ... FROM / DELETE ... USING).
    // Reject rather than silently leak; a superuser bypasses RLS anyway.
    if !catalog.is_superuser() && catalog.rls_enabled(&source.schema, &source.name)? {
        return Err(Error::Unsupported(
            "MERGE USING a row-level security protected source table is not yet supported"
                .to_owned(),
        ));
    }
    let target_qual = m.target.alias.clone().unwrap_or_else(|| table.name.clone());
    let source_qual = m
        .source
        .alias
        .clone()
        .unwrap_or_else(|| source.name.clone());
    let mut scope = scope_of_aliased(&table, &target_qual);
    scope.extend(scope_of_aliased(&source, &source_qual));

    let on = analyze_expr(&m.on, &scope, catalog, Some(ColumnType::Bool))?;
    if on.ty != ColumnType::Bool {
        return Err(Error::TypeMismatch {
            context: "MERGE ON condition".to_owned(),
            expected: ColumnType::Bool,
            found: on.ty,
        });
    }

    let mut whens = Vec::with_capacity(m.whens.len());
    for when in m.whens {
        whens.push(analyze_merge_when(when, &table, &scope, catalog)?);
    }
    Ok(MergePlan {
        table,
        source,
        source_plan: source_plan.map(Box::new),
        on,
        whens,
    })
}

/// Analyze one `WHEN [NOT] MATCHED` clause of a `MERGE` against the combined `target ++ source` scope.
fn analyze_merge_when(
    when: ast::MergeWhen,
    table: &TableSchema,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<MergeWhen, Error> {
    match when {
        ast::MergeWhen::Matched { pred, action } => Ok(MergeWhen::Matched {
            pred: analyze_predicate(pred, scope, catalog)?,
            action: analyze_merge_matched_action(action, table, scope, catalog)?,
        }),
        ast::MergeWhen::NotMatchedBySource { pred, action } => Ok(MergeWhen::NotMatchedBySource {
            pred: analyze_predicate(pred, scope, catalog)?,
            action: analyze_merge_matched_action(action, table, scope, catalog)?,
        }),
        ast::MergeWhen::NotMatched { pred, insert } => {
            let pred = analyze_predicate(pred, scope, catalog)?;
            let columns: Vec<usize> = if insert.columns.is_empty() {
                (0..table.columns.len()).collect()
            } else {
                insert
                    .columns
                    .iter()
                    .map(|name| find_column(&table.columns, name, &table.name).map(|(i, _)| i))
                    .collect::<Result<_, _>>()?
            };
            if insert.values.len() != columns.len() {
                return Err(Error::ArityMismatch {
                    context: "MERGE WHEN NOT MATCHED INSERT".to_owned(),
                    expected: columns.len(),
                    found: insert.values.len(),
                });
            }
            let mut values = Vec::with_capacity(insert.values.len());
            for (val, &col_idx) in insert.values.iter().zip(&columns) {
                let column = table
                    .columns
                    .get(col_idx)
                    .ok_or_else(|| Error::ColumnNotFound {
                        table: table.name.clone(),
                        column: col_idx.to_string(),
                    })?;
                let value = analyze_expr(val, scope, catalog, Some(column.ty))?;
                check_assignable(column, &value)?;
                values.push(value);
            }
            Ok(MergeWhen::NotMatched {
                pred,
                columns,
                values,
            })
        },
    }
}

/// Analyze the `THEN {UPDATE SET ... | DELETE}` action of a clause that acts on an existing target
/// row (`WHEN MATCHED` / `WHEN NOT MATCHED BY SOURCE`). Assignments name target columns and are
/// type-checked against the combined `target ++ source` scope; a column may be assigned once.
fn analyze_merge_matched_action(
    action: ast::MatchedAction,
    table: &TableSchema,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<MergeMatchedAction, Error> {
    match action {
        ast::MatchedAction::Update { assignments } => {
            let mut typed = Vec::with_capacity(assignments.len());
            let mut seen = HashSet::new();
            for a in assignments {
                let (index, column) = find_column(&table.columns, &a.column, &table.name)?;
                if !seen.insert(index) {
                    return Err(Error::DuplicateColumn { name: a.column });
                }
                let value = analyze_expr(&a.value, scope, catalog, Some(column.ty))?;
                check_assignable(column, &value)?;
                typed.push(Assignment {
                    column: index,
                    value,
                });
            }
            Ok(MergeMatchedAction::Update { assignments: typed })
        },
        ast::MatchedAction::Delete => Ok(MergeMatchedAction::Delete),
    }
}

// === Updatable views =====================================================
//
// An INSERT/UPDATE/DELETE whose target is an *auto-updatable* view rewrites onto the view's base
// table. A view is auto-updatable when its body is a single-table projection of plain (optionally
// aliased) columns, optionally row-filtered, with none of the machinery — DISTINCT, GROUP BY,
// HAVING, LIMIT/OFFSET, WITH, joins, set operations, derived tables — that makes a view read-only in
// the reference engine.

/// An auto-updatable view resolved to its base table.
pub(super) struct UpdatableView {
    /// The base table's explicit schema, when the view body qualified it; `None` otherwise.
    base_schema: Option<String>,
    /// The base table's name.
    base_table: String,
    /// The view's own row filter (its `WHERE`, written in terms of base columns), combined under
    /// `AND` into an UPDATE/DELETE so the statement only affects the rows the view exposes.
    filter: Option<ast::Expr>,
    /// `(view output column, base column)` pairs, in projection order.
    col_map: Vec<(String, String)>,
    /// Whether the view was created `WITH CHECK OPTION` — a row written through it must still satisfy
    /// [`filter`](Self::filter), enforced by a `view_check` predicate on the rewritten INSERT/UPDATE.
    check_option: bool,
}

/// If `name` (optionally schema-qualified) names a non-materialized view, decide whether it is
/// auto-updatable and, if so, return its base table + column mapping + filter. `Ok(None)` when `name`
/// is not a view at all (the caller then reports the ordinary "table not found"). `Err` when it *is* a
/// view but its shape is not auto-updatable.
pub(super) fn resolve_updatable_view(
    schema: Option<&str>,
    name: &str,
    catalog: &dyn Catalog,
) -> Result<Option<UpdatableView>, Error> {
    let Some(key) = super::view_lookup_key(schema, name, catalog)? else {
        return Ok(None); // not a view
    };
    let Some(sql) = catalog.lookup_view(&key)? else {
        return Ok(None);
    };
    let non_updatable = |why: &str| {
        Error::Unsupported(format!(
            "view `{name}` is not auto-updatable ({why}); modify its base table instead"
        ))
    };
    let Ok(ast::Statement::Select(select)) = crate::parse(&sql) else {
        return Err(non_updatable("its definition is not a plain SELECT"));
    };
    if select.distinct.is_some() {
        return Err(non_updatable("it uses DISTINCT"));
    }
    if !matches!(&select.group_by, ast::GroupBy::Expressions(keys) if keys.is_empty()) {
        return Err(non_updatable("it uses GROUP BY"));
    }
    if select.having.is_some() {
        return Err(non_updatable("it has a HAVING clause"));
    }
    if select.limit.is_some() || select.offset.is_some() {
        return Err(non_updatable("it uses LIMIT/OFFSET"));
    }
    if !select.with.is_empty() {
        return Err(non_updatable("it has a WITH clause"));
    }
    let Some(from) = &select.from else {
        return Err(non_updatable("it selects from no table"));
    };
    if !from.joins.is_empty() {
        return Err(non_updatable("it joins more than one table"));
    }
    let base = &from.base;
    if base.subquery.is_some() || base.values.is_some() || base.set_op.is_some() {
        return Err(non_updatable(
            "its FROM item is a derived table, not a base table",
        ));
    }
    // The base must be a real table — a view over another view is not auto-updatable in this version.
    let Some(base_table) = super::lookup_table_ref(base.schema.as_deref(), &base.name, catalog)?
    else {
        return Err(non_updatable("its base is not a plain table"));
    };
    // Every projection item must be a plain (optionally aliased) column reference; a wildcard expands
    // to the base table's columns in order. Anything computed makes the view read-only.
    let mut base_cols: Vec<String> = Vec::new();
    let mut inferred_names: Vec<String> = Vec::new();
    for item in &select.projection {
        match item {
            ast::SelectItem::Expr {
                expr: ast::Expr::Column(column),
                alias,
            } => {
                inferred_names.push(alias.clone().unwrap_or_else(|| column.clone()));
                base_cols.push(column.clone());
            },
            ast::SelectItem::Expr {
                expr: ast::Expr::QualifiedColumn { column, .. },
                alias,
            } => {
                inferred_names.push(alias.clone().unwrap_or_else(|| column.clone()));
                base_cols.push(column.clone());
            },
            ast::SelectItem::Wildcard | ast::SelectItem::QualifiedWildcard(_) => {
                for col in &base_table.columns {
                    inferred_names.push(col.name.clone());
                    base_cols.push(col.name.clone());
                }
            },
            ast::SelectItem::Expr { .. } => {
                return Err(non_updatable(
                    "a projected column is an expression, not a plain column",
                ));
            },
        }
    }
    // An explicit `CREATE VIEW v (a, b) AS ...` column list overrides the inferred output names.
    let declared = catalog.lookup_view_columns(&key)?;
    let out_names = if declared.is_empty() {
        inferred_names
    } else if declared.len() == base_cols.len() {
        declared
    } else {
        return Err(Error::Internal(format!(
            "view `{name}` declares {} output column(s) but its body projects {}",
            declared.len(),
            base_cols.len()
        )));
    };
    Ok(Some(UpdatableView {
        base_schema: base.schema.clone(),
        base_table: base.name.clone(),
        filter: select.filter.clone(),
        col_map: out_names.into_iter().zip(base_cols).collect(),
        check_option: catalog.lookup_view_check_option(&key)?,
    }))
}

/// Combine a statement filter with the view's own filter under `AND` (either may be absent).
fn and_filters(stmt: Option<ast::Expr>, view: Option<ast::Expr>) -> Option<ast::Expr> {
    match (stmt, view) {
        (Some(a), Some(b)) => Some(ast::Expr::Binary {
            left: Box::new(a),
            op: ast::BinaryOp::And,
            right: Box::new(b),
        }),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

/// INSERT through an auto-updatable view: map the (view) target columns to base columns and re-run
/// the INSERT against the base table. Column renames and column subsets are fine here — an omitted
/// base column simply takes its default, exactly as a partial-column INSERT into the base would.
fn insert_through_view(
    mut ins: ast::Insert,
    view: UpdatableView,
    catalog: &dyn Catalog,
) -> Result<InsertPlan, Error> {
    if ins.on_conflict.is_some() {
        return Err(Error::Unsupported(
            "ON CONFLICT through a view is not supported yet".to_owned(),
        ));
    }
    if !ins.returning.is_empty() {
        return Err(Error::Unsupported(
            "INSERT ... RETURNING through a view is not supported yet".to_owned(),
        ));
    }
    let base_columns = if ins.columns.is_empty() {
        // No column list: the target is every view column, in view order.
        view.col_map.iter().map(|(_, base)| base.clone()).collect()
    } else {
        let mut mapped = Vec::with_capacity(ins.columns.len());
        for column in &ins.columns {
            match view.col_map.iter().find(|(view_col, _)| view_col == column) {
                Some((_, base)) => mapped.push(base.clone()),
                None => {
                    return Err(Error::ColumnNotFound {
                        table: ins.table.clone(),
                        column: column.clone(),
                    });
                },
            }
        }
        mapped
    };
    let check_option = view.check_option;
    let view_filter = view.filter.clone();
    let view_name = ins.table.clone();
    ins.schema = view.base_schema;
    ins.table = view.base_table;
    ins.columns = base_columns;
    let mut plan = analyze_insert(ins, catalog)?;
    // WITH CHECK OPTION: every row inserted through the view must satisfy the view's `WHERE` (in base
    // columns), enforced against the fully-defaulted row at execution.
    if check_option {
        let scope = super::single_table_scope(&plan.table);
        plan.view_check =
            super::analyze_predicate(view_filter, &scope, catalog)?.map(|predicate| {
                crate::planner::ViewCheck {
                    predicate,
                    view: view_name,
                }
            });
    }
    Ok(plan)
}

/// A view usable as an UPDATE/DELETE target must expose *every* base column under its own name (no
/// rename, no subset). Then the statement's own column references — which this version does not
/// rewrite — resolve unchanged against the base table, and a reference to a column the view omits is
/// rejected exactly as the reference engine rejects it.
fn require_full_identity_view(
    view: &UpdatableView,
    view_name: &str,
    catalog: &dyn Catalog,
) -> Result<(), Error> {
    if view.col_map.iter().any(|(out, base)| out != base) {
        return Err(Error::Unsupported(format!(
            "UPDATE/DELETE through view `{view_name}` that renames a column is not supported yet"
        )));
    }
    let base = super::lookup_table_ref(view.base_schema.as_deref(), &view.base_table, catalog)?
        .ok_or_else(|| Error::TableNotFound {
            name: view.base_table.clone(),
        })?;
    let exposed: HashSet<&str> = view.col_map.iter().map(|(_, base)| base.as_str()).collect();
    if base
        .columns
        .iter()
        .any(|col| !exposed.contains(col.name.as_str()))
    {
        return Err(Error::Unsupported(format!(
            "UPDATE/DELETE through view `{view_name}` that exposes only some base columns is not \
             supported yet"
        )));
    }
    Ok(())
}

/// UPDATE through an auto-updatable view: retarget the base table and AND the view's filter into the
/// WHERE so only rows the view exposes are updated.
fn update_through_view(
    mut upd: ast::Update,
    view: UpdatableView,
    view_name: &str,
    catalog: &dyn Catalog,
) -> Result<UpdatePlan, Error> {
    if !upd.returning.is_empty() {
        return Err(Error::Unsupported(
            "UPDATE ... RETURNING through a view is not supported yet".to_owned(),
        ));
    }
    require_full_identity_view(&view, view_name, catalog)?;
    let check_option = view.check_option;
    let view_filter = view.filter.clone();
    upd.filter = and_filters(upd.filter.take(), view.filter);
    upd.schema = view.base_schema;
    upd.table = view.base_table;
    let mut plan = analyze_update(upd, catalog)?;
    // WITH CHECK OPTION: each post-update row must still be visible through the view (satisfy its
    // `WHERE`), enforced against the post-update row at execution.
    if check_option {
        let scope = super::single_table_scope(&plan.table);
        plan.view_check =
            super::analyze_predicate(view_filter, &scope, catalog)?.map(|predicate| {
                crate::planner::ViewCheck {
                    predicate,
                    view: view_name.to_owned(),
                }
            });
    }
    Ok(plan)
}

/// DELETE through an auto-updatable view: retarget the base table and AND the view's filter into the
/// WHERE so only rows the view exposes are deleted.
fn delete_through_view(
    mut del: ast::Delete,
    view: UpdatableView,
    view_name: &str,
    catalog: &dyn Catalog,
) -> Result<DeletePlan, Error> {
    if !del.returning.is_empty() {
        return Err(Error::Unsupported(
            "DELETE ... RETURNING through a view is not supported yet".to_owned(),
        ));
    }
    require_full_identity_view(&view, view_name, catalog)?;
    del.filter = and_filters(del.filter.take(), view.filter);
    del.schema = view.base_schema;
    del.table = view.base_table;
    analyze_delete(del, catalog)
}
