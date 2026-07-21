//! DML analyzers: INSERT (+RETURNING, CREATE INDEX/SEQUENCE), UPDATE, DELETE.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === INSERT ===============================================================

pub(super) fn analyze_insert(ins: ast::Insert, catalog: &dyn Catalog) -> Result<InsertPlan, Error> {
    // The system-catalog namespace is reserved: a user INSERT into e.g. `nusadb_policies`
    // would forge a policy and bypass RLS entirely.
    enforce_system_catalog(&ins.table, catalog)?;
    // Resolve without the RLS refusal `resolve_table` applies: a non-superuser may INSERT rows its
    // policies' WITH CHECK admit, so RLS is enforced by the `rls_check` predicate below.
    let table =
        super::lookup_table_ref(ins.schema.as_deref(), &ins.table, catalog)?.ok_or_else(|| {
            Error::TableNotFound {
                name: super::qualified_display_opt(ins.schema.as_deref(), &ins.table),
            }
        })?;
    // `DEFAULT VALUES` names no target columns — every column is omitted and takes its DEFAULT.
    let targets = if matches!(ins.source, ast::InsertSource::DefaultValues) {
        Vec::new()
    } else {
        resolve_insert_columns(&ins, &table)?
    };
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
                for (value, column) in row.iter().zip(&target_columns) {
                    // A `None` cell is an explicit `DEFAULT`: leave it unresolved so the executor
                    // fills it from the column's default/serial/NULL, exactly like an omitted column.
                    let typed = match value {
                        Some(expr) => Some(analyze_insert_value(expr, column, catalog)?),
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
    let rls_check = if !catalog.is_superuser() && catalog.rls_enabled(&table.name)? {
        Some(build_rls_check_predicate(
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
        // DO NOTHING applies to any unique conflict, so a stated target is not needed.
        return Ok(OnConflictPlan::DoNothing);
    };
    let target = match conflict.target {
        Some(ast::ConflictTarget::Columns(cols)) => {
            let mut ordinals = Vec::with_capacity(cols.len());
            for name in &cols {
                let (index, _) = find_column(&table.columns, name, &table.name)?;
                ordinals.push(index);
            }
            ConflictArbiter::Columns(ordinals)
        },
        Some(ast::ConflictTarget::Constraint(name)) => ConflictArbiter::Constraint(name),
        None => {
            return Err(Error::Unsupported(
                "ON CONFLICT DO UPDATE requires a conflict target — \
                 `ON CONFLICT (columns)` or `ON CONFLICT ON CONSTRAINT name`"
                    .to_owned(),
            ));
        },
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

/// The combined `[target ++ EXCLUDED]` scope for an `ON CONFLICT DO UPDATE`: the target
/// table's columns at ordinals `[0, n)` and a second copy qualified `excluded` (reachable only as
/// `excluded.col`) at `[n, 2n)`, matching the row the executor evaluates assignments against.
fn upsert_scope(table: &TableSchema) -> Vec<ScopedColumn> {
    let mut scope = single_table_scope(table);
    scope.extend(table.columns.iter().map(|def| ScopedColumn {
        qualifier: "excluded".to_owned(),
        def: def.clone(),
        qualified_only: true,
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
        return Err(Error::Unsupported(
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
        return Err(Error::Unsupported(
            "an hnsw vector index must be over exactly one VECTOR column".to_owned(),
        ));
    };
    let (column_ordinal, def) = find_column(&table.columns, column, &table.name)?;
    let ColumnType::Vector(dim) = def.ty else {
        return Err(Error::Unsupported(
            "an hnsw index requires a VECTOR(n) column".to_owned(),
        ));
    };
    Ok(VectorIndexSpec {
        name: ci.name.clone(),
        table: table.name.clone(),
        column: column.clone(),
        column_ordinal,
        dim: dim as usize,
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

/// Evaluate a sequence-option expression to a constant `i64` — an integer literal or its negation.
pub(super) fn const_i64(expr: &ast::Expr) -> Result<i64, Error> {
    match expr {
        ast::Expr::Literal(ast::Value::Int(n)) => Ok(*n),
        ast::Expr::Unary {
            op: ast::UnaryOp::Negate,
            expr,
        } => match expr.as_ref() {
            ast::Expr::Literal(ast::Value::Int(n)) => n
                .checked_neg()
                .ok_or_else(|| Error::Unsupported("sequence option value out of range".to_owned())),
            _ => Err(Error::Unsupported(
                "sequence option must be an integer constant".to_owned(),
            )),
        },
        _ => Err(Error::Unsupported(
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
    check_assignable(column, &typed)?;
    Ok(typed)
}

// === UPDATE / DELETE ======================================================

pub(super) fn analyze_update(upd: ast::Update, catalog: &dyn Catalog) -> Result<UpdatePlan, Error> {
    // The system-catalog namespace is reserved: a user UPDATE of e.g. `nusadb_policies`
    // could widen a policy's USING predicate and bypass RLS.
    enforce_system_catalog(&upd.table, catalog)?;
    // Resolve without the RLS refusal `resolve_table` applies: a non-superuser may UPDATE the rows
    // its policies' USING grant (folded into `filter`) to values its WITH CHECK admit (`rls_check`).
    let table =
        super::lookup_table_ref(upd.schema.as_deref(), &upd.table, catalog)?.ok_or_else(|| {
            Error::TableNotFound {
                name: super::qualified_display_opt(upd.schema.as_deref(), &upd.table),
            }
        })?;
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
        if !catalog.is_superuser() && catalog.rls_enabled(&schema.name)? {
            return Err(Error::Unsupported(format!(
                "row-level security on `{}` combined with UPDATE ... FROM is not yet supported",
                schema.name
            )));
        }
        scope.extend(schema.columns.iter().map(|def| ScopedColumn {
            qualifier: qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
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
    let rls_check = if !catalog.is_superuser() && catalog.rls_enabled(&table.name)? {
        let using = build_rls_predicate(&table.name, ast::PolicyCommand::Update, &scope, catalog)?;
        filter = Some(match filter {
            None => using,
            Some(existing) => and_exprs(existing, using),
        });
        Some(build_rls_check_predicate(
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
    let table =
        super::lookup_table_ref(del.schema.as_deref(), &del.table, catalog)?.ok_or_else(|| {
            Error::TableNotFound {
                name: super::qualified_display_opt(del.schema.as_deref(), &del.table),
            }
        })?;
    // RETURNING projects the deleted rows, resolved against the table's columns.
    let returning = analyze_returning(&del.returning, &table, catalog)?;
    // DELETE ... USING: resolve a single named source table and extend the scope, so the
    // WHERE may reference its columns (at ordinals `target_width + j` of `target ++ using`). Reuses
    // the same single-named-table resolution as UPDATE ... FROM.
    let using = del
        .using
        .map(|u| resolve_update_from(u, catalog))
        .transpose()?;
    let mut scope = scope_of(&table);
    let mut using_table: Option<TableSchema> = None;
    let mut using_plan: Option<Box<SelectPlan>> = None;
    if let Some((schema, qualifier, plan)) = using {
        // A USING source is a de-facto join: like SELECT's RLS+JOIN refusal, a non-superuser must not
        // read an RLS-protected source table (its rows would otherwise leak through the WHERE
        // predicate). Fail closed (deep-gate security). A derived source's schema name is its
        // alias; in the unlikely case that alias collides with an RLS-protected table name the guard
        // merely over-rejects (fail closed) — never a leak.
        if !catalog.is_superuser() && catalog.rls_enabled(&schema.name)? {
            return Err(Error::Unsupported(format!(
                "row-level security on `{}` combined with DELETE ... USING is not yet supported",
                schema.name
            )));
        }
        scope.extend(schema.columns.iter().map(|def| ScopedColumn {
            qualifier: qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
        }));
        using_plan = plan.map(Box::new);
        using_table = Some(schema);
    }
    let mut filter = analyze_predicate(del.filter, &scope, catalog)?;
    // Row-level security: a non-superuser may only delete rows the DELETE/ALL policies grant.
    // DELETE has no WITH CHECK, so injecting the USING predicate is complete (default-deny FALSE
    // when no policy applies, like SELECT).
    if !catalog.is_superuser() && catalog.rls_enabled(&table.name)? {
        let policy = build_rls_predicate(&table.name, ast::PolicyCommand::Delete, &scope, catalog)?;
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
        restart_identity: false,
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
        return Err(Error::Unsupported(
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
    // Row-level security on the MERGE target is not yet wired (the matched UPDATE/DELETE side would
    // not enforce the policies' USING / WITH CHECK that a plain UPDATE/DELETE does). Reject rather
    // than silently bypass RLS for a non-superuser; a superuser bypasses RLS anyway.
    if !catalog.is_superuser() && catalog.rls_enabled(&table.name)? {
        return Err(Error::Unsupported(
            "MERGE on a row-level-security protected table is not yet supported".to_owned(),
        ));
    }
    // The USING source is scanned in full to drive the match — it is not filtered by RLS, so a
    // non-superuser could read every row of a row-level-security protected source through a matched
    // action's SET / search condition (the same leak class as UPDATE ... FROM / DELETE ... USING).
    // Reject rather than silently leak; a superuser bypasses RLS anyway.
    if !catalog.is_superuser() && catalog.rls_enabled(&source.name)? {
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
        ast::MergeWhen::Matched { pred, action } => {
            let pred = analyze_predicate(pred, scope, catalog)?;
            let action = match action {
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
                    MergeMatchedAction::Update { assignments: typed }
                },
                ast::MatchedAction::Delete => MergeMatchedAction::Delete,
            };
            Ok(MergeWhen::Matched { pred, action })
        },
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
