//! SELECT analysis: FROM/scope resolution, grouping sets, set operations, projection.
//!
//! Split verbatim out of `analyzer/mod.rs` (ADR 007). Siblings resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === SELECT ===============================================================

/// A resolved `FROM` clause: base source, join chain, and the column scope.
pub(super) struct ResolvedFrom {
    /// Base source table; `None` for a `SELECT` without `FROM` or a CTE base (see `base_cte`).
    table: Option<TableSchema>,
    /// Inlined plan when the base source is a non-recursive CTE. Mutually exclusive with
    /// `table`; its output columns form the base scope.
    base_cte: Option<SelectPlan>,
    /// Resolved joins, in order.
    joins: Vec<JoinPlan>,
    /// Column scope `[base cols ++ join0 cols ++ ...]`, indexed by row ordinal.
    scope: Vec<ScopedColumn>,
}

/// Resolve a derived-table FROM base `(SELECT ...) AS x`: analyze the subquery standalone
/// and inline its plan exactly like a non-recursive CTE — its projection becomes the relation's
/// schema. A `LATERAL` derived table cannot be the first FROM item (nothing to its left to reference).
fn resolve_derived_base(
    base: &ast::TableRef,
    subquery: &ast::Select,
    catalog: &dyn Catalog,
) -> Result<(TableSchema, Option<SelectPlan>), Error> {
    if base.lateral {
        return Err(Error::Unsupported(
            "a LATERAL derived table cannot be the first item in FROM".to_owned(),
        ));
    }
    let plan = apply_ordinality(analyze_select(subquery.clone(), catalog)?, base)?;
    let schema = cte_schema(&base.name, &base.column_aliases, &plan)?;
    Ok((schema, Some(plan)))
}

/// Apply a `FROM ... WITH ORDINALITY` to a freshly-analyzed derived-table plan: require a
/// set-returning projection (the `ProjectSet` that appends the row number) and flag the plan so the
/// planner emits the 1-based `ordinality` column. A no-op when the table has no
/// `WITH ORDINALITY`.
fn apply_ordinality(mut plan: SelectPlan, table: &ast::TableRef) -> Result<SelectPlan, Error> {
    if table.with_ordinality {
        let has_srf = plan
            .projection
            .iter()
            .any(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }));
        if !has_srf {
            return Err(Error::Unsupported(
                "WITH ORDINALITY requires a set-returning function (e.g. unnest, generate_series)"
                    .to_owned(),
            ));
        }
        plan.ordinality = true;
    }
    Ok(plan)
}

/// Resolve a `(VALUES (row), ...) AS x` derived table into a [`SelectPlan`] whose source is the inline
/// rows. Every row must have the same arity; each column's type is unified across all rows (numeric
/// widening per `unify_result_ty`, and a bare `NULL` takes the type inferred from the column's other
/// rows). The plan's projection is the identity over the value columns, named `column1`, `column2`,
/// … — [`cte_schema`] then applies any `(a, b)` alias list positionally and qualifies by `name`.
fn analyze_values_table(
    rows: &[Vec<ast::Expr>],
    catalog: &dyn Catalog,
) -> Result<SelectPlan, Error> {
    // The parser guarantees at least one row.
    let ncols = rows.first().map_or(0, Vec::len);
    for row in rows {
        if row.len() != ncols {
            return Err(Error::ArityMismatch {
                context: "VALUES".to_owned(),
                expected: ncols,
                found: row.len(),
            });
        }
    }
    // VALUES cells reference no columns (LATERAL VALUES is not modelled), so they resolve against an
    // empty scope. Pass 1: infer each column's type, deferring a bare NULL (typed in pass 2 once the
    // column type is known from its other rows). Every row has `ncols` cells (checked above), so the
    // per-column accumulator is indexed by `enumerate` ordinal without bounds risk.
    let empty_scope: Vec<ScopedColumn> = Vec::new();
    let mut col_seen: Vec<Option<ColumnType>> = vec![None; ncols];
    for row in rows {
        for (slot, cell) in col_seen.iter_mut().zip(row) {
            match analyze_expr(cell, &empty_scope, catalog, None) {
                Ok(te) => *slot = Some(unify_result_ty(*slot, te.ty, "VALUES column")?),
                Err(Error::AmbiguousNull { .. }) => {},
                Err(e) => return Err(e),
            }
        }
    }
    let col_types: Vec<ColumnType> = col_seen
        .into_iter()
        .enumerate()
        .map(|(j, seen)| {
            seen.ok_or_else(|| Error::AmbiguousNull {
                context: format!("VALUES column {} is entirely NULL — add a cast", j + 1),
            })
        })
        .collect::<Result<_, _>>()?;
    // Pass 2: type every cell against its column type (resolving the deferred NULLs).
    let mut typed_rows: Vec<Vec<TypedExpr>> = Vec::with_capacity(rows.len());
    for row in rows {
        let typed = row
            .iter()
            .zip(&col_types)
            .map(|(cell, &ty)| analyze_expr(cell, &empty_scope, catalog, Some(ty)))
            .collect::<Result<_, _>>()?;
        typed_rows.push(typed);
    }
    let projection = col_types
        .iter()
        .enumerate()
        .map(|(j, &ty)| Projection {
            expr: TypedExpr {
                kind: TypedExprKind::Column(j),
                ty,
            },
            name: format!("column{}", j + 1),
        })
        .collect();
    Ok(SelectPlan {
        table: None,
        values: typed_rows,
        set_op_source: None,
        from_cte: None,
        joins: Vec::new(),
        distinct: false,
        distinct_on: Vec::new(),
        projection,
        filter: None,
        order_by: Vec::new(),
        limit: None,
        limit_with_ties: false,
        offset: None,
        group_keys: Vec::new(),
        grouping_sets: Vec::new(),
        windows: Vec::new(),
        having: None,
        aggregates: Vec::new(),
        indexes: Vec::new(),
        table_stats: None,
        approx_scan_rows: None,
        recursive_ctes: Vec::new(),
        modifying_ctes: Vec::new(),
        row_lock: None,
        ordinality: false,
    })
}

/// Resolve a `(SELECT ... UNION/INTERSECT/EXCEPT ...) AS x` derived table into a [`SelectPlan`] whose
/// source is the set operation. The set-op's output columns (names + types from the leftmost branch)
/// become the relation's columns; the identity projection over them flows through
/// [`cte_schema`], which applies any `(a, b)` alias list positionally and qualifies by `name`.
fn analyze_set_op_table(so: ast::SetOperation, catalog: &dyn Catalog) -> Result<SelectPlan, Error> {
    let set_op = analyze_set_operation(so, catalog)?;
    // The relation's typing is the branches' UNIFIED typing, not the
    // leftmost leaf's — `(SELECT 1 UNION SELECT 2.5) AS x` is a NUMERIC column.
    let types = set_op.column_types.clone();
    let projection = set_op
        .columns
        .iter()
        .zip(&types)
        .enumerate()
        .map(|(j, (name, &ty))| Projection {
            expr: TypedExpr {
                kind: TypedExprKind::Column(j),
                ty,
            },
            name: name.clone(),
        })
        .collect();
    Ok(SelectPlan {
        table: None,
        values: Vec::new(),
        set_op_source: Some(Box::new(set_op)),
        from_cte: None,
        joins: Vec::new(),
        distinct: false,
        distinct_on: Vec::new(),
        projection,
        filter: None,
        order_by: Vec::new(),
        limit: None,
        limit_with_ties: false,
        offset: None,
        group_keys: Vec::new(),
        grouping_sets: Vec::new(),
        windows: Vec::new(),
        having: None,
        aggregates: Vec::new(),
        indexes: Vec::new(),
        table_stats: None,
        approx_scan_rows: None,
        recursive_ctes: Vec::new(),
        modifying_ctes: Vec::new(),
        row_lock: None,
        ordinality: false,
    })
}

/// Resolve a join input (the right side of a `JOIN`): a derived table `JOIN (SELECT ...) AS x`
/// (increment 3b) — analyzed standalone and inlined via `input_cte` like a CTE — or a named
/// catalog table. A `LATERAL` join input is analyzed with `left_scope` pushed as the enclosing scope
/// (increment 3c) so its references to columns on its left resolve to `OuterColumn`s; a CTE
/// referenced in a `JOIN` is not yet supported.
fn resolve_join_input(
    table: &ast::TableRef,
    catalog: &dyn Catalog,
    ctes: &[ResolvedCte],
    left_scope: &[ScopedColumn],
) -> Result<(TableSchema, Option<Box<SelectPlan>>), Error> {
    if let Some(set_op) = &table.set_op {
        // `JOIN (SELECT ... UNION ...) AS x`: inline the set-op plan like a derived-table join input.
        let plan = analyze_set_op_table((**set_op).clone(), catalog)?;
        let schema = cte_schema(&table.name, &table.column_aliases, &plan)?;
        return Ok((schema, Some(Box::new(plan))));
    }
    if let Some(values) = &table.values {
        // `JOIN (VALUES ...) AS x`: inline the rows plan like a derived-table join input.
        let plan = analyze_values_table(values, catalog)?;
        let schema = cte_schema(&table.name, &table.column_aliases, &plan)?;
        return Ok((schema, Some(Box::new(plan))));
    }
    if let Some(subquery) = &table.subquery {
        // For `LATERAL`, push the columns to this join's left as the enclosing scope so the
        // subquery body can correlate to them (resolved to `OuterColumn`s the executor binds per
        // left row); the guard pops the scope when this resolution returns (increment 3c).
        let _outer = table.lateral.then(|| push_outer_scope(left_scope));
        let plan = apply_ordinality(analyze_select((**subquery).clone(), catalog)?, table)?;
        let schema = cte_schema(&table.name, &table.column_aliases, &plan)?;
        return Ok((schema, Some(Box::new(plan))));
    }
    // A CTE only shadows an unqualified reference; an explicit schema qualifier always denotes
    // a real base table, never a CTE. A non-recursive CTE referenced in a JOIN is inlined like a
    // derived-table join input (its planned body flows through `input_cte`); the join's alias / column
    // aliases re-qualify it via `cte_schema`, exactly as for `JOIN (SELECT ...) AS x`.
    if table.schema.is_none()
        && let Some(cte) = ctes.iter().find(|c| c.name == table.name)
    {
        return match &cte.source {
            CteSource::Inline(plan) => {
                let schema = cte_schema(&table.name, &table.column_aliases, plan)?;
                Ok((schema, Some(plan.clone())))
            },
            CteSource::Recursive | CteSource::Modifying => Err(Error::Unsupported(
                "a recursive or data-modifying CTE referenced in a JOIN is not yet supported \
                 — reference it as the FROM base"
                    .to_owned(),
            )),
        };
    }
    Ok((
        resolve_table(table.schema.as_deref(), &table.name, catalog)?,
        None,
    ))
}

/// Resolve a single auxiliary relation for `UPDATE ... FROM` / `DELETE ... USING`: a named
/// table, or a `(VALUES ...)` / `(SELECT ...)` / set-operation derived table. Returns the relation's
/// schema and, for a derived source, the inlined plan that produces its rows (`None` for a named
/// table, which the executor scans directly). A `LATERAL` source is rejected — there is no left-hand
/// scope here for it to correlate to. The named table is looked up without the `resolve_table` RLS
/// refusal; the caller applies the UPDATE/DELETE-FROM RLS guard against the returned schema.
pub(super) fn resolve_aux_relation(
    base: &ast::TableRef,
    catalog: &dyn Catalog,
) -> Result<(TableSchema, Option<SelectPlan>), Error> {
    if base.lateral {
        return Err(Error::Unsupported(
            "a LATERAL source in UPDATE ... FROM / DELETE ... USING is not supported".to_owned(),
        ));
    }
    if let Some(set_op) = &base.set_op {
        let plan = analyze_set_op_table((**set_op).clone(), catalog)?;
        let schema = cte_schema(&base.name, &base.column_aliases, &plan)?;
        Ok((schema, Some(plan)))
    } else if let Some(values) = &base.values {
        let plan = analyze_values_table(values, catalog)?;
        let schema = cte_schema(&base.name, &base.column_aliases, &plan)?;
        Ok((schema, Some(plan)))
    } else if let Some(subquery) = &base.subquery {
        let plan = apply_ordinality(analyze_select((**subquery).clone(), catalog)?, base)?;
        let schema = cte_schema(&base.name, &base.column_aliases, &plan)?;
        Ok((schema, Some(plan)))
    } else {
        // A named aux table resolves through the search path / its explicit qualifier.
        let schema =
            lookup_table_ref(base.schema.as_deref(), &base.name, catalog)?.ok_or_else(|| {
                Error::TableNotFound {
                    name: qualified_display_opt(base.schema.as_deref(), &base.name),
                }
            })?;
        Ok((schema, None))
    }
}

/// Resolve a `FROM` clause into its base source, join chain, and column scope
/// (against which the rest of the `SELECT` resolves). The base name resolves to a
/// `WITH` CTE first (shadowing a same-named table); otherwise to a catalog table.
/// Each join's `ON` predicate is type-checked against the scope built so far.
#[allow(
    clippy::too_many_lines,
    reason = "flat FROM-base dispatch: derived table / CTE / information_schema / catalog table, \
              then the join-chain walk; length scales with the source taxonomy, not nesting"
)]
pub(super) fn resolve_from(
    from: Option<&ast::FromClause>,
    catalog: &dyn Catalog,
    ctes: &[ResolvedCte],
) -> Result<ResolvedFrom, Error> {
    let Some(from) = from else {
        return Ok(ResolvedFrom {
            table: None,
            base_cte: None,
            joins: Vec::new(),
            scope: Vec::new(),
        });
    };
    // The base source: a CTE (shadows a same-named table) or a catalog table. A non-recursive CTE
    // is inlined via `base_cte`; a recursive one resolves to its synthetic table (scanned at
    // execution from the working-set registry), so it takes the plain-table path.
    let (base, base_cte) = if let Some(set_op) = &from.base.set_op {
        // A `(SELECT ... UNION ...) AS x` base: build the set-op plan and inline it like a derived
        // table.
        let plan = analyze_set_op_table((**set_op).clone(), catalog)?;
        let schema = cte_schema(&from.base.name, &from.base.column_aliases, &plan)?;
        (schema, Some(plan))
    } else if let Some(values) = &from.base.values {
        // A `(VALUES ...) AS x` base: build the inline-rows plan and inline it like a derived table.
        let plan = analyze_values_table(values, catalog)?;
        let schema = cte_schema(&from.base.name, &from.base.column_aliases, &plan)?;
        (schema, Some(plan))
    } else if let Some(subquery) = &from.base.subquery {
        resolve_derived_base(&from.base, subquery, catalog)?
    } else if from.base.schema.is_none()
        && let Some(cte) = ctes.iter().find(|c| c.name == from.base.name)
    {
        // A CTE only shadows an unqualified reference.
        match &cte.source {
            CteSource::Inline(plan) => (cte.schema.clone(), Some((**plan).clone())),
            // Recursive / data-modifying CTEs resolve to their synthetic table; the executor binds
            // the rows (the fixpoint working set / the RETURNING rows) before the body runs.
            CteSource::Recursive | CteSource::Modifying => (cte.schema.clone(), None),
        }
    } else if let Some(view) = crate::planner::InfoSchemaView::from_full_name(&from.base.name) {
        // An `information_schema.{tables,columns,...}` reference resolves to a synthetic table
        // whose rows are produced by the executor. The alias (if any) overrides the
        // view's canonical name as the column qualifier.
        (view.table_schema(), None)
    } else {
        // The system-catalog namespace is reserved: a non-superuser SELECT on e.g.
        // `nusadb_policies` would leak every policy definition. (A CTE may shadow a `nusadb_*`
        // name — that is the user's own data, so the guard sits on the catalog path only.)
        enforce_system_catalog(&from.base.name, catalog)?;
        if let Some(schema) =
            lookup_table_ref(from.base.schema.as_deref(), &from.base.name, catalog)?
        {
            // Row-level security for the base table is applied by `apply_rls` once the WHERE
            // filter is built (single-table queries inject policy predicates; joins are refused).
            (schema, None)
        } else if from.base.schema.is_none() {
            // Not a CTE or base table — a non-materialized view inlines its body like a CTE.
            // An explicit `CREATE VIEW name (cols)` list renames the body's columns positionally.
            // (Views live in the default namespace; an explicit schema qualifier denotes a table.)
            let plan = resolve_view(&from.base.name, catalog)?;
            let explicit = catalog.lookup_view_columns(&from.base.name)?;
            let schema = cte_schema(&from.base.name, &explicit, &plan)?;
            (schema, Some(plan))
        } else {
            return Err(Error::TableNotFound {
                name: qualified_display_opt(from.base.schema.as_deref(), &from.base.name),
            });
        }
    };
    let base_qualifier = from.base.alias.clone().unwrap_or_else(|| base.name.clone());
    let mut scope: Vec<ScopedColumn> = base
        .columns
        .iter()
        .map(|def| ScopedColumn {
            qualifier: base_qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
        })
        .collect();
    let mut joins = Vec::with_capacity(from.joins.len());
    for join in &from.joins {
        // A LATERAL join input correlates to the columns to its left, so resolve it against the
        // scope built so far. A right/full lateral join is meaningless (the right side depends on
        // the left), so only inner/left/cross lateral joins are allowed (increment 3c).
        if join.table.lateral && matches!(join.kind, ast::JoinKind::Right | ast::JoinKind::Full) {
            return Err(Error::Unsupported(
                "a RIGHT/FULL JOIN LATERAL is not supported".to_owned(),
            ));
        }
        let (joined, join_cte) = resolve_join_input(&join.table, catalog, ctes, &scope)?;
        let qualifier = join
            .table
            .alias
            .clone()
            .unwrap_or_else(|| joined.name.clone());
        // Columns to the LEFT of this join (the running scope) end here; the joined table's
        // columns follow. `USING`/`NATURAL` reference both sides by this boundary.
        let left_width = scope.len();
        scope.extend(joined.columns.iter().map(|def| ScopedColumn {
            qualifier: qualifier.clone(),
            def: def.clone(),
            qualified_only: false,
        }));
        // Resolve the join predicate. `ON` analyzes the explicit boolean; `CROSS` (no
        // condition) is a Cartesian product (predicate `true`); `USING (cols)` and `NATURAL` build
        // an equality conjunction over the named / common columns. All lower to the same join
        // operator — the executor treats `Cross` like an inner join with the synthesized predicate.
        // `USING`/`NATURAL` coalesce their join columns into ONE output column, so the names they
        // join on are tracked here and the right side's copy is hidden below.
        let mut coalesced: Vec<String> = Vec::new();
        let mut coalesce_pairs: Vec<(usize, usize)> = Vec::new();
        let on = match &join.condition {
            ast::JoinCondition::On(expr) => {
                let on = analyze_expr(expr, &scope, catalog, Some(ColumnType::Bool))?;
                if on.ty != ColumnType::Bool {
                    return Err(Error::TypeMismatch {
                        context: "JOIN ON clause".to_owned(),
                        expected: ColumnType::Bool,
                        found: on.ty,
                    });
                }
                on
            },
            ast::JoinCondition::None => bool_literal(true),
            ast::JoinCondition::Using(cols) => {
                let (predicate, pairs) =
                    join_equality_predicate(cols, &scope, left_width, &joined)?;
                coalesced.clone_from(cols);
                coalesce_pairs = pairs;
                predicate
            },
            ast::JoinCondition::Natural => {
                // NATURAL joins on every column name common to both sides; none in common → CROSS.
                let left_cols = scope.get(..left_width).unwrap_or(&[]);
                let common: Vec<String> = joined
                    .columns
                    .iter()
                    .filter(|d| left_cols.iter().any(|c| c.def.name == d.name))
                    .map(|d| d.name.clone())
                    .collect();
                let (predicate, pairs) =
                    join_equality_predicate(&common, &scope, left_width, &joined)?;
                coalesced = common;
                coalesce_pairs = pairs;
                predicate
            },
        };
        // A `USING`/`NATURAL` join's shared column is the merge `coalesce(left, right)`: the right
        // copy is hidden (it stays in the row — a qualified `right.col` still resolves — but is
        // invisible to a bare reference and `SELECT *`, so the column appears once) and the kept-left
        // slot carries `coalesce(left, right)`. For INNER/LEFT that already equals the left value
        // (the left row is present and, under the equi-join, equal where matched), so the merge is a
        // no-op there; for RIGHT/FULL an unmatched row has a NULL left but a present right, so the
        // executor fills the left slot from the right via `coalesce_pairs`.
        if !coalesced.is_empty() {
            for col in scope.iter_mut().skip(left_width) {
                if coalesced.iter().any(|name| name == &col.def.name) {
                    col.qualified_only = true;
                }
            }
        }
        joins.push(JoinPlan {
            table: joined,
            kind: join.kind,
            on,
            coalesce: coalesce_pairs,
            input_cte: join_cte,
            lateral: join.table.lateral,
        });
    }
    Ok(ResolvedFrom {
        // A CTE base is carried in `base_cte` and inlined by the planner, not scanned as a table.
        table: if base_cte.is_some() { None } else { Some(base) },
        base_cte,
        joins,
        scope,
    })
}

/// A `BOOL` literal typed expression (the predicate of a `CROSS JOIN`).
const fn bool_literal(value: bool) -> TypedExpr {
    TypedExpr {
        kind: TypedExprKind::Literal(ast::Value::Bool(value)),
        ty: ColumnType::Bool,
    }
}

/// The column scope of a single base table: every column qualified by the table name, at its
/// declaration ordinal. Used to type-check a policy predicate against the table it governs.
pub(super) fn single_table_scope(table: &TableSchema) -> Vec<ScopedColumn> {
    table
        .columns
        .iter()
        .map(|def| ScopedColumn {
            qualifier: table.name.clone(),
            def: def.clone(),
            qualified_only: false,
        })
        .collect()
}

/// `a AND b` over two boolean typed expressions.
pub(super) fn and_exprs(left: TypedExpr, right: TypedExpr) -> TypedExpr {
    TypedExpr {
        kind: TypedExprKind::Binary {
            left: Box::new(left),
            op: ast::BinaryOp::And,
            right: Box::new(right),
        },
        ty: ColumnType::Bool,
    }
}

/// Apply row-level security to a single base-table `SELECT`'s `WHERE` predicate.
///
/// For a superuser, an RLS-free table, or a non-base source (CTE / `SELECT` without `FROM`), the
/// filter is returned unchanged. For a non-superuser reading a single RLS-enabled base table, the
/// applicable permissive policies are folded into one predicate (default-deny `FALSE` when none
/// apply) and `AND`-combined with the existing filter. A `JOIN` over an RLS table is refused — the
/// per-relation predicate placement that joins need is a follow-up; refusing fails closed.
fn apply_rls(
    filter: Option<TypedExpr>,
    base_table: Option<&TableSchema>,
    has_joins: bool,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<Option<TypedExpr>, Error> {
    let Some(base) = base_table else {
        return Ok(filter);
    };
    if catalog.is_superuser() || !catalog.rls_enabled(&base.name)? {
        return Ok(filter);
    }
    if has_joins {
        return Err(Error::Unsupported(format!(
            "row-level security on `{}` combined with a JOIN is not yet supported",
            base.name
        )));
    }
    let policy = build_rls_predicate(&base.name, ast::PolicyCommand::Select, scope, catalog)?;
    Ok(Some(match filter {
        None => policy,
        Some(existing) => and_exprs(existing, policy),
    }))
}

/// Build the row predicate that selects which rows a non-superuser may touch on `table` for a
/// statement of `command`: the `OR` of every applicable permissive policy's `USING`
/// predicate, narrowed by the `AND` of every applicable restrictive policy's `USING`, or `FALSE`
/// (default-deny) when no permissive policy applies. See [`combine_rls_policies`].
pub(super) fn build_rls_predicate(
    table: &str,
    command: ast::PolicyCommand,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    // A policy grants row access through its USING predicate; a policy without one (e.g. an
    // INSERT-only WITH CHECK policy) grants no read/select access and is skipped.
    combine_rls_policies(table, command, scope, catalog, |p| p.using.clone())
}

/// Build the `WITH CHECK` predicate a non-superuser's written row must satisfy on `table` for a
/// statement of `command`: the `OR` of every applicable permissive policy's `WITH CHECK`
/// (falling back to its `USING`), narrowed by the `AND` of every applicable restrictive policy's
/// `WITH CHECK`, or `FALSE` (default-deny) when no permissive policy applies. The executor rejects a
/// row that fails this. See [`combine_rls_policies`].
pub(super) fn build_rls_check_predicate(
    table: &str,
    command: ast::PolicyCommand,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    // The write predicate is the policy's WITH CHECK, or its USING when WITH CHECK is omitted (so a
    // single `USING` policy also constrains the rows the user may write).
    combine_rls_policies(table, command, scope, catalog, |p| {
        p.check.clone().or_else(|| p.using.clone())
    })
}

/// Combine the row-level-security policies applicable to `table` for `command` into one predicate,
/// using `pick` to select each policy's relevant SQL text (`USING` for reads, `WITH CHECK`
/// for writes).
///
/// The result is `(OR of permissive predicates) AND (AND of restrictive predicates)`, following the
/// SQL-standard row-level-security model: permissive policies *grant* access (default-deny `FALSE` when
/// none apply, so a restrictive-only or policy-less table exposes no rows) and restrictive policies
/// *narrow* it (no further restriction when none apply). A policy applies when its command is `ALL`
/// or matches `command` and its role list is `PUBLIC` (empty) or names the session user; its picked
/// predicate is re-parsed and type-checked against `scope` so it filters exactly like a `WHERE`
/// clause. A policy whose `pick` yields no SQL is skipped.
fn combine_rls_policies(
    table: &str,
    command: ast::PolicyCommand,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    pick: impl Fn(&PolicyDef) -> Option<String>,
) -> Result<TypedExpr, Error> {
    let user = catalog.current_user();
    let mut permissive: Option<TypedExpr> = None;
    let mut restrictive: Option<TypedExpr> = None;
    for policy in catalog.lookup_policies(table)? {
        let applies_to_command =
            policy.command == ast::PolicyCommand::All || policy.command == command;
        let applies_to_role = policy.roles.is_empty() || policy.roles.iter().any(|r| r == &user);
        if !applies_to_command || !applies_to_role {
            continue;
        }
        let Some(sql) = pick(&policy) else {
            continue;
        };
        let expr = crate::parser::parse_expression(&sql)?;
        let Some(typed) = analyze_predicate(Some(expr), scope, catalog)? else {
            continue;
        };
        if policy.permissive {
            permissive = Some(fold_bool(permissive.take(), typed, ast::BinaryOp::Or));
        } else {
            restrictive = Some(fold_bool(restrictive.take(), typed, ast::BinaryOp::And));
        }
    }
    // No applicable permissive policy → default-deny; restrictive policies never grant access on
    // their own, they only narrow what a permissive policy already allows.
    let granted = permissive.unwrap_or_else(|| bool_literal(false));
    Ok(match restrictive {
        None => granted,
        Some(restr) => and_exprs(granted, restr),
    })
}

/// Fold `next` into the running boolean predicate `acc` with `op`, or return `next` when `acc` is
/// empty. Used to `OR`/`AND` row-level-security policy predicates together.
fn fold_bool(acc: Option<TypedExpr>, next: TypedExpr, op: ast::BinaryOp) -> TypedExpr {
    match acc {
        None => next,
        Some(acc) => TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(acc),
                op,
                right: Box::new(next),
            },
            ty: ColumnType::Bool,
        },
    }
}

/// Build the equality conjunction `left.c = right.c AND …` that a `USING (cols)` or `NATURAL` join
/// reduces to. `left_width` is the number of columns to the left of the joined table in
/// `scope`; each `c` must name a column on both sides. An empty `cols` (a `NATURAL` join with no
/// common column) yields `true` — a Cartesian product, per SQL.
fn join_equality_predicate(
    cols: &[String],
    scope: &[ScopedColumn],
    left_width: usize,
    joined: &TableSchema,
) -> Result<(TypedExpr, Vec<(usize, usize)>), Error> {
    let left_cols = scope.get(..left_width).unwrap_or(&[]);
    let mut predicate: Option<TypedExpr> = None;
    // The (kept-left, hidden-right) ordinal of each merged column, so the executor can surface the
    // merge as `coalesce(left, right)` (correct for RIGHT/FULL where the left side may be NULL).
    let mut coalesce_pairs: Vec<(usize, usize)> = Vec::with_capacity(cols.len());
    for name in cols {
        let (left_ord, left_col) = left_cols
            .iter()
            .enumerate()
            .find(|(_, c)| &c.def.name == name)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "join column `{name}` is not present on the left side"
                ))
            })?;
        let (right_pos, right_col) = joined
            .columns
            .iter()
            .enumerate()
            .find(|(_, d)| &d.name == name)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "join column `{name}` is not present on the right side"
                ))
            })?;
        // Compare physical types so a `VARCHAR(n)` USING/NATURAL-join column matches a `TEXT`
        // (or differently-sized character) column on the other side.
        let left_ty = left_col.def.ty.physical();
        let right_ty = right_col.ty.physical();
        if left_ty != right_ty {
            return Err(Error::TypeMismatch {
                context: format!("join column `{name}`"),
                expected: left_ty,
                found: right_ty,
            });
        }
        coalesce_pairs.push((left_ord, left_width + right_pos));
        let eq = TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(TypedExpr {
                    kind: TypedExprKind::Column(left_ord),
                    ty: left_ty,
                }),
                op: ast::BinaryOp::Eq,
                right: Box::new(TypedExpr {
                    kind: TypedExprKind::Column(left_width + right_pos),
                    ty: right_ty,
                }),
            },
            ty: ColumnType::Bool,
        };
        predicate = Some(match predicate {
            None => eq,
            Some(acc) => TypedExpr {
                kind: TypedExprKind::Binary {
                    left: Box::new(acc),
                    op: ast::BinaryOp::And,
                    right: Box::new(eq),
                },
                ty: ColumnType::Bool,
            },
        });
    }
    Ok((
        predicate.unwrap_or_else(|| bool_literal(true)),
        coalesce_pairs,
    ))
}

/// Reject parsed-but-unimplemented `SELECT`-level clauses before deeper analysis.
/// Resolve one `ORDER BY` key, supporting positional references and output-column aliases.
///
/// Resolution order follows SQL:
/// 1. A positional integer (`ORDER BY 2`) selects the Nth output column (1-based).
/// 2. A bare identifier naming an output column (`ORDER BY my_alias`) resolves to that output
///    column — output names take precedence over source columns of the same name.
/// 3. Anything else is analyzed as an ordinary expression against the row scope.
fn resolve_order_by_key(
    item_expr: &ast::Expr,
    projection: &[Projection],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregated: bool,
    aggregates: &mut Vec<AggregateCall>,
) -> Result<TypedExpr, Error> {
    // 1. Positional: ORDER BY <1-based integer literal> → the Nth output column.
    if let ast::Expr::Literal(ast::Value::Int(n)) = item_expr {
        return usize::try_from(*n)
            .ok()
            .filter(|&p| p >= 1)
            .and_then(|p| projection.get(p - 1))
            .map(|p| p.expr.clone())
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "ORDER BY position {n} is out of range (1..={})",
                    projection.len()
                ))
            });
    }
    // 2. Alias: a bare identifier naming an output column.
    if let ast::Expr::Column(name) = item_expr
        && let Some(proj) = projection.iter().find(|p| p.name == *name)
    {
        return Ok(proj.expr.clone());
    }
    // 3. Otherwise analyze as an ordinary expression against the row scope.
    if aggregated {
        analyze_projection_expr(item_expr, scope, catalog, aggregates)
    } else {
        analyze_expr(item_expr, scope, catalog, None)
    }
}

/// Validate a `SELECT ... FOR UPDATE` / `FOR SHARE` clause and map it to a [`RowLockMode`] plus
/// the `SKIP LOCKED` flag (the job-queue pattern). Only the simple shape is supported — a
/// single base table with no join / aggregate / GROUP BY / DISTINCT / window and a subquery-free
/// `WHERE` — so the executor can lock exactly the matched base rows by re-scanning. `OF <table>` /
/// `NOWAIT` and any richer shape are honest `Unsupported` (`NOWAIT` because the no-wait lock
/// manager reports a conflict as `40001`, not the reference engine's `55P03` — mapping it would mislead retry
/// logic). `None` lock → `None`.
fn analyze_row_lock(
    lock: Option<&ast::RowLock>,
    simple_shape: bool,
    filter: Option<&TypedExpr>,
) -> Result<Option<(nusadb_core::engine::RowLockMode, bool)>, Error> {
    let Some(lock) = lock else {
        return Ok(None);
    };
    if lock.of.is_some() {
        return Err(Error::Unsupported(
            "FOR UPDATE / FOR SHARE ... OF <table>".to_owned(),
        ));
    }
    if matches!(lock.wait, ast::LockWait::NoWait) {
        return Err(Error::Unsupported(
            "FOR UPDATE / FOR SHARE with NOWAIT".to_owned(),
        ));
    }
    if !simple_shape {
        return Err(Error::Unsupported(
            "FOR UPDATE / FOR SHARE is supported only on a single base table without \
             join / aggregate / GROUP BY / DISTINCT / window"
                .to_owned(),
        ));
    }
    if filter.is_some_and(crate::executor::ops::contains_subquery) {
        return Err(Error::Unsupported(
            "FOR UPDATE / FOR SHARE with a subquery in WHERE".to_owned(),
        ));
    }
    let mode = match lock.strength {
        ast::LockStrength::Update => nusadb_core::engine::RowLockMode::Exclusive,
        ast::LockStrength::Share => nusadb_core::engine::RowLockMode::Shared,
    };
    Ok(Some((mode, matches!(lock.wait, ast::LockWait::SkipLocked))))
}

/// Resolve a `UNION`/`INTERSECT`/`EXCEPT` statement: analyze the operand tree, enforce
/// column compatibility, and resolve the combined `ORDER BY`/`LIMIT` against the output columns
/// (named after the leftmost branch).
pub(super) fn analyze_set_operation(
    so: ast::SetOperation,
    catalog: &dyn Catalog,
) -> Result<SetOpPlan, Error> {
    // A `WITH` on a set operation scopes over every branch. Resolve it (against any enclosing visible
    // CTEs) and make it the visible frame while the body is analyzed, so each branch's `FROM` resolves
    // the CTE. A recursive / data-modifying CTE has no place to carry its def on the set-operation
    // envelope, so reject it loudly rather than drop it; a plain inline CTE inlines into each branch.
    let outer = visible_ctes();
    let (own_ctes, recursive, modifying) = analyze_ctes(&so.with, catalog, &outer)?;
    if !recursive.is_empty() || !modifying.is_empty() {
        return Err(Error::Unsupported(
            "a recursive or data-modifying WITH on a UNION / INTERSECT / EXCEPT is not yet supported"
                .to_owned(),
        ));
    }
    let ctes = combine_ctes(own_ctes, &outer);
    let _cte_guard = push_visible_ctes(&ctes);
    let (tree, cols) = analyze_set_body(so.body, catalog)?;
    // The combined result's scope: one (unqualified) column per output column, so ORDER BY can
    // reference output columns by name.
    let scope: Vec<ScopedColumn> = cols
        .iter()
        .map(|(name, ty)| ScopedColumn {
            qualifier: String::new(),
            def: ColumnDef {
                name: name.clone(),
                ty: *ty,
                nullable: true,
            },
            qualified_only: false,
        })
        .collect();
    let mut order_by = Vec::with_capacity(so.order_by.len());
    for item in &so.order_by {
        // Positional reference: ORDER BY <1-based integer literal> → the Nth output column.
        // Without this, an integer literal would sort by a constant (a silent no-op).
        let expr = if let ast::Expr::Literal(ast::Value::Int(n)) = &item.expr {
            let resolved = usize::try_from(*n)
                .ok()
                .filter(|&p| p >= 1)
                .and_then(|p| cols.get(p - 1).map(|(_, ty)| (p, *ty)));
            match resolved {
                Some((pos, ty)) => TypedExpr {
                    kind: TypedExprKind::Column(pos - 1),
                    ty,
                },
                None => {
                    return Err(Error::Unsupported(format!(
                        "ORDER BY position {n} is out of range (1..={})",
                        cols.len()
                    )));
                },
            }
        } else {
            // A bare alias resolves through `scope` (output columns are in scope by name).
            analyze_expr(&item.expr, &scope, catalog, None)?
        };
        order_by.push(OrderByKey {
            expr,
            ascending: item.ascending,
            nulls: item.nulls,
        });
    }
    let (columns, column_types) = cols.into_iter().unzip();
    Ok(SetOpPlan {
        tree,
        columns,
        column_types,
        order_by,
        limit: so.limit,
    })
}

/// A resolved set-operation tree plus its output columns `(name, type)`.
type SetBody = (SetOpTree<SelectPlan>, Vec<(String, ColumnType)>);

/// Recursively analyze a set-operation body into a resolved tree plus its output columns
/// `(name, type)`. A `Node` requires both operands to have the same column count and a unifiable
/// per-column type; the output column **names** come from the left operand and each **type** is the
/// two branches unified (numeric branches widen, like `CASE`).
pub(super) fn analyze_set_body(
    body: ast::SelectBody,
    catalog: &dyn Catalog,
) -> Result<SetBody, Error> {
    match body {
        ast::SelectBody::Select(select) => {
            let plan = analyze_select(*select, catalog)?;
            let cols = plan
                .projection
                .iter()
                .map(|p| (p.name.clone(), p.expr.ty))
                .collect();
            Ok((SetOpTree::Leaf(Box::new(plan)), cols))
        },
        ast::SelectBody::SetOp {
            op,
            all,
            left,
            right,
        } => {
            let (ltree, lcols) = analyze_set_body(*left, catalog)?;
            let (rtree, rcols) = analyze_set_body(*right, catalog)?;
            if lcols.len() != rcols.len() {
                return Err(Error::ArityMismatch {
                    context: "set operation".to_owned(),
                    expected: lcols.len(),
                    found: rcols.len(),
                });
            }
            // Unify each column's type across the two branches (names come from the left), so mixed
            // numeric branches widen — `SELECT 1 UNION SELECT 2.5` yields a NUMERIC column rather than
            // a spurious type mismatch — matching the standard. Incompatible types still error.
            let mut cols = lcols;
            for (l, r) in cols.iter_mut().zip(&rcols) {
                l.1 = unify_result_ty(Some(l.1), r.1, "set operation column")?;
            }
            Ok((
                SetOpTree::Node {
                    op,
                    all,
                    left: Box::new(ltree),
                    right: Box::new(rtree),
                },
                cols,
            ))
        },
    }
}

thread_local! {
    /// Recursion depth while inlining non-materialized views, to bound a pathological view cycle
    /// (e.g. one built via `CREATE OR REPLACE`) instead of overflowing the stack.
    static VIEW_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Resolve a non-materialized view name to its analyzed body by parsing + analyzing its stored SQL,
/// so it can be inlined in place of a `FROM` base. Errors with `TableNotFound` if `name` is
/// neither a base table nor a view, or `Unsupported` if view nesting is too deep (a cycle).
fn resolve_view(name: &str, catalog: &dyn Catalog) -> Result<SelectPlan, Error> {
    const MAX_VIEW_DEPTH: u32 = 64;
    let Some(sql) = catalog.lookup_view(name)? else {
        return Err(Error::TableNotFound {
            name: name.to_owned(),
        });
    };
    if VIEW_DEPTH.with(std::cell::Cell::get) >= MAX_VIEW_DEPTH {
        return Err(Error::Unsupported(format!(
            "view `{name}` nests deeper than {MAX_VIEW_DEPTH} levels (likely a recursive view)"
        )));
    }
    VIEW_DEPTH.with(|d| d.set(d.get() + 1));
    let result = match crate::parse(&sql) {
        Ok(ast::Statement::Select(sel)) => analyze_select(sel, catalog),
        Ok(_) => Err(Error::Unsupported(format!(
            "view `{name}` definition is not a SELECT"
        ))),
        Err(e) => Err(e),
    };
    VIEW_DEPTH.with(|d| d.set(d.get() - 1));
    result
}

thread_local! {
    /// The CTEs lexically visible at the current point, one frame per enclosing query block. A
    /// subquery (scalar / `IN` / `EXISTS` / quantified) or a set-operation branch has no `WITH` of its
    /// own, so it inherits the enclosing block's CTEs from the top frame here — making
    /// `WITH a AS (…) SELECT … WHERE x IN (SELECT … FROM a)` resolve `a`.
    static VISIBLE_CTES: std::cell::RefCell<Vec<Vec<ResolvedCte>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Make `ctes` the visible CTE frame for the duration of the returned guard, so any subquery /
/// set-operation branch analyzed within the current block inherits them. Popped on drop.
#[must_use]
fn push_visible_ctes(ctes: &[ResolvedCte]) -> VisibleCtesGuard {
    VISIBLE_CTES.with(|s| s.borrow_mut().push(ctes.to_vec()));
    VisibleCtesGuard
}

struct VisibleCtesGuard;

impl Drop for VisibleCtesGuard {
    fn drop(&mut self) {
        VISIBLE_CTES.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// The CTEs lexically visible at the current point (the top [`VISIBLE_CTES`] frame); empty at the top
/// level. A nested `analyze_select` (a subquery, or a set-operation branch) inherits these.
fn visible_ctes() -> Vec<ResolvedCte> {
    VISIBLE_CTES.with(|s| s.borrow().last().cloned().unwrap_or_default())
}

pub(super) fn analyze_select(sel: ast::Select, catalog: &dyn Catalog) -> Result<SelectPlan, Error> {
    // A subquery inherits the enclosing block's visible CTEs (pushed by `analyze_select_scoped`); the
    // top-level call sees an empty frame.
    analyze_select_scoped(sel, catalog, &visible_ctes())
}

/// Like [`analyze_select`] but with `outer_ctes` — the CTEs of any enclosing `WITH` clause already
/// resolved — in scope, so a CTE body can reference an earlier sibling CTE (`WITH a AS (…), b AS
/// (SELECT … FROM a) …`). The top-level call passes an empty slice.
#[allow(
    clippy::too_many_lines,
    reason = "flat clause-by-clause SELECT resolution (FROM/GROUP/projection/DISTINCT/HAVING/ORDER); \
              its length tracks the SQL surface, not branching complexity"
)]
fn analyze_select_scoped(
    mut sel: ast::Select,
    catalog: &dyn Catalog,
    outer_ctes: &[ResolvedCte],
) -> Result<SelectPlan, Error> {
    // `FOR UPDATE` / `FOR SHARE` is validated against the resolved shape just before the
    // plan is returned (below), once joins/aggregation/etc. are known.
    // Plain DISTINCT de-dups all output columns; DISTINCT ON (keys) keeps the first source row per
    // key tuple. The ON keys are resolved against the column scope below, once it is built.
    let (distinct, distinct_on_exprs) = match sel.distinct.take() {
        None => (false, Vec::new()),
        Some(ast::Distinct::All) => (true, Vec::new()),
        Some(ast::Distinct::On(exprs)) => (false, exprs),
    };
    // A data-modifying CTE's target may not be read elsewhere in this statement: forbid
    // every such target for the rest of this block's analysis (CTE siblings and the body, including
    // their subqueries, which all resolve base tables through `resolve_table`). Each modifying CTE
    // lifts its own target while it is analyzed.
    let dml_targets: Vec<String> = sel
        .with
        .iter()
        .filter_map(|c| match &c.body {
            ast::CteBody::Modifying(stmt) => dml_target_name(stmt).map(str::to_owned),
            ast::CteBody::Query(_) => None,
        })
        .collect();
    let _forbid = (!dml_targets.is_empty()).then(|| super::forbid_dml_targets(&dml_targets));
    // Resolve `WITH` CTEs so the FROM base can reference them. Recursive CTEs also
    // yield fixpoint defs threaded into the plan and materialized at execution; data-modifying CTEs
    // yield run-once defs whose RETURNING rows are bound to the synthetic table.
    let (own_ctes, recursive_ctes, modifying_ctes) = analyze_ctes(&sel.with, catalog, outer_ctes)?;
    // The CTEs visible to this block's FROM: its own (which shadow same-named enclosing ones), then
    // the enclosing `outer_ctes`.
    let ctes = combine_ctes(own_ctes, outer_ctes);
    // Make them the visible frame for any subquery in this block's projection / WHERE / HAVING /
    // ORDER BY (and FROM-derived tables), so a subquery with no `WITH` of its own still resolves them.
    let _cte_guard = push_visible_ctes(&ctes);
    // Resolve FROM (base + joins) and the column scope; ordinals throughout
    // index the concatenated joined row.
    let ResolvedFrom {
        table,
        base_cte,
        joins,
        scope: scope_vec,
    } = resolve_from(sel.from.as_ref(), catalog, &ctes)?;
    let scope: &[ScopedColumn] = &scope_vec;

    // Resolve `GROUP BY` (incl. ROLLUP/CUBE/GROUPING SETS). `group_keys` is the union of
    // every column mentioned by any grouping set, in first-seen declared order; each key also forms
    // a leading column of the synthesized post-aggregation row. `grouping_sets` records, for each
    // set, the indices into `group_keys` it activates (empty for a plain `GROUP BY`). v1: keys must
    // be column references.
    let mut group_keys: Vec<TypedExpr> = Vec::new();
    let mut grouping_sets: Vec<Vec<usize>> = Vec::new();
    match expand_grouping(&sel.group_by)? {
        GroupingSpec::Plain(keys) => {
            for key in &keys {
                intern_group_key(key, &sel.projection, scope, catalog, &mut group_keys)?;
            }
        },
        GroupingSpec::Sets(sets) => {
            for set in &sets {
                let mut indices: Vec<usize> = Vec::with_capacity(set.len());
                for col in set {
                    let slot =
                        intern_group_key(col, &sel.projection, scope, catalog, &mut group_keys)?;
                    if !indices.contains(&slot) {
                        indices.push(slot);
                    }
                }
                grouping_sets.push(indices);
            }
        },
    }

    // Aggregate sink — shared across the whole query block. Populated first by the window pass (a
    // grouping aggregate inside a window's PARTITION/ORDER, e.g. `rank() OVER (ORDER BY sum(x))`) and
    // then by the projection; both extract into the same sink so every aggregate has a stable slot.
    let mut aggregates: Vec<AggregateCall> = Vec::new();

    // Window functions: extract them from the projection, resolving each against the
    // source scope, and rewrite each occurrence to reference a synthetic appended column. The
    // executor's Window operator appends one column per window after the source (or, when the query
    // also aggregates, the post-aggregation) columns.
    let mut windows: Vec<WindowExpr> = Vec::new();
    let projection_items = rewrite_window_items(
        sel.projection,
        scope,
        catalog,
        &mut windows,
        Some(&mut aggregates),
    )?;
    // Project against the source scope extended with one synthetic column per window (so the
    // rewritten references resolve to the appended ordinals). `*` still expands source columns only.
    let source_len = scope.len();
    let projection_scope: Vec<ScopedColumn> = scope
        .iter()
        .cloned()
        .chain(windows.iter().enumerate().map(|(i, w)| ScopedColumn {
            qualifier: String::new(),
            def: ColumnDef {
                name: window_col_name(i),
                ty: w.result_ty,
                nullable: true,
            },
            qualified_only: false,
        }))
        .collect();

    // Projection. As a side effect, aggregate calls are extracted into
    // `aggregates` (already seeded by the window pass above) and replaced with `AggregateRef(idx)`.
    let mut projection = analyze_projection(
        projection_items,
        &projection_scope,
        catalog,
        &mut aggregates,
        source_len,
    )?;
    let filter = analyze_predicate(sel.filter, scope, catalog)?;
    // Row-level security: fold the base table's applicable policies into the filter for a
    // non-superuser (single-table only; joins over an RLS table are refused). Runs per query block,
    // so subqueries enforce their own base independently.
    let filter = apply_rls(filter, table.as_ref(), !joins.is_empty(), scope, catalog)?;

    // Set-returning functions expand rows, so v1 supports exactly one per projection and not
    // combined with aggregation/grouping/windows/DISTINCT (each would need a second evaluation phase).
    let srf_count = projection
        .iter()
        .filter(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }))
        .count();
    if srf_count > 1 {
        return Err(Error::Unsupported(
            "more than one set-returning function in a SELECT list is not supported yet".to_owned(),
        ));
    }
    if srf_count == 1
        && (!group_keys.is_empty()
            || !aggregates.is_empty()
            || !grouping_sets.is_empty()
            || !windows.is_empty()
            || distinct)
    {
        return Err(Error::Unsupported(
            "a set-returning function with GROUP BY / aggregation / window / DISTINCT is not \
             supported yet"
                .to_owned(),
        ));
    }

    // A `SELECT` aggregates when it has `GROUP BY` keys (incl. grouping sets) or aggregate calls
    // (the sink may already hold aggregates a window's PARTITION/ORDER pulled in above).
    let aggregated = !group_keys.is_empty() || !aggregates.is_empty() || !grouping_sets.is_empty();
    // A window function over aggregated rows runs after the GROUP BY / scalar-aggregate stage
    // (`rank() OVER (ORDER BY sum(x)) … GROUP BY …`): the planner places the Window operator above
    // the aggregate and its expressions are rebased onto the post-aggregation row below. Grouping
    // sets (ROLLUP/CUBE/GROUPING SETS) grow the aggregate sink during rebase (each GROUPING(...)
    // appends a slot), which would shift the window-column base — so that rarer combination stays
    // unsupported for now.
    if !windows.is_empty() && !grouping_sets.is_empty() {
        return Err(Error::Unsupported(
            "window functions together with GROUPING SETS / ROLLUP / CUBE are not supported yet"
                .to_owned(),
        ));
    }

    // Resolve DISTINCT ON keys against the source scope. They keep the first source row per
    // key tuple, so they reference source columns — only meaningful for a non-aggregated SELECT.
    let distinct_on = if distinct_on_exprs.is_empty() {
        Vec::new()
    } else if aggregated {
        return Err(Error::Unsupported(
            "SELECT DISTINCT ON together with GROUP BY / aggregation is not supported".to_owned(),
        ));
    } else {
        distinct_on_exprs
            .iter()
            .map(|expr| analyze_expr(expr, scope, catalog, None))
            .collect::<Result<Vec<_>, _>>()?
    };

    // HAVING — post-aggregation predicate; only valid when aggregating, and may
    // introduce new aggregates (e.g. `HAVING COUNT(*) > 1`).
    let mut having = match sel.having {
        Some(expr) => {
            if !aggregated {
                return Err(Error::Unsupported(
                    "HAVING requires GROUP BY or an aggregate function".to_owned(),
                ));
            }
            let typed = analyze_projection_expr(&expr, scope, catalog, &mut aggregates)?;
            if typed.ty != ColumnType::Bool {
                return Err(Error::TypeMismatch {
                    context: "HAVING clause".to_owned(),
                    expected: ColumnType::Bool,
                    found: typed.ty,
                });
            }
            Some(typed)
        },
        None => None,
    };

    // ORDER BY. For an aggregated SELECT, keys reference the post-aggregation
    // row (and may introduce aggregates); otherwise they reference source rows.
    let mut order_by = Vec::with_capacity(sel.order_by.len());
    for item in &sel.order_by {
        let expr = resolve_order_by_key(
            &item.expr,
            &projection,
            scope,
            catalog,
            aggregated,
            &mut aggregates,
        )?;
        order_by.push(OrderByKey {
            expr,
            ascending: item.ascending,
            nulls: item.nulls,
        });
    }

    // Rebase output expressions onto the synthesized post-aggregation row
    // `[group keys ++ aggregate results]`: group-key columns become
    // `AggregateRef(k)`, aggregate refs shift past the group keys, and any
    // other bare column is rejected (it is neither grouped nor aggregated).
    if aggregated {
        // The post-aggregation row width `[group keys ++ aggregates]`. A plain GROUP BY / scalar
        // aggregate never grows the sink during rebase (only grouping sets do, and those are rejected
        // with windows), so this is the final width — the base at which the Window operator appends
        // its output columns.
        let post_agg_width = group_keys.len() + aggregates.len();
        let mut ctx = AggRebase {
            group_keys: &group_keys,
            num_group_keys: group_keys.len(),
            grouping_sets: &grouping_sets,
            aggregates: &mut aggregates,
            source_len,
            post_agg_width,
        };
        for proj in &mut projection {
            rebase_onto_aggregation(&mut proj.expr, &mut ctx)?;
        }
        if let Some(having) = having.as_mut() {
            rebase_onto_aggregation(having, &mut ctx)?;
        }
        for key in &mut order_by {
            rebase_onto_aggregation(&mut key.expr, &mut ctx)?;
        }
        // A window over aggregated rows runs on the post-aggregation row, so its PARTITION / ORDER /
        // argument expressions rebase onto it too — a grouped column becomes its key slot and a
        // grouping aggregate its shifted `AggregateRef`, exactly as the projection's do.
        for window in &mut windows {
            for part in &mut window.partition {
                rebase_onto_aggregation(part, &mut ctx)?;
            }
            for key in &mut window.order {
                rebase_onto_aggregation(&mut key.expr, &mut ctx)?;
            }
            for arg in &mut window.args {
                rebase_onto_aggregation(arg, &mut ctx)?;
            }
        }
    }

    // Resolve the base table's indexes so the planner can consider an index scan. Done here,
    // while the catalog is in hand; the planner is pure and only sees the resolved metadata.
    let indexes = resolve_table_indexes(table.as_ref(), catalog)?;
    // Fetch the base table's ANALYZE stats for cost-based planning — single-table SELECTs
    // only (a join's per-table stats are not yet threaded). `None` leaves planning heuristic.
    let table_stats = match (table.as_ref(), joins.is_empty()) {
        (Some(schema), true) => catalog.table_stats(&schema.name)?,
        _ => None,
    };
    // The `O(1)` approximate row count of the same single base table — the vectorized-routing
    // cardinality fallback the planner uses when there are no `ANALYZE` stats (so a large un-analyzed
    // table still vectorizes). `0` (the default / no cheap estimate) leaves the fallback off.
    let approx_scan_rows = match (table.as_ref(), joins.is_empty()) {
        (Some(schema), true) => match catalog.approx_row_count(&schema.name)? {
            0 => None,
            n => Some(n),
        },
        _ => None,
    };

    // `FOR UPDATE` / `FOR SHARE`: allowed only on a single base table with no
    // join / aggregate / GROUP BY / DISTINCT / window and a subquery-free predicate, so the executor
    // can lock exactly the matched base rows.
    let simple_shape = table.is_some()
        && base_cte.is_none()
        && joins.is_empty()
        && !distinct
        && distinct_on.is_empty()
        && aggregates.is_empty()
        && group_keys.is_empty()
        && grouping_sets.is_empty()
        && windows.is_empty()
        && having.is_none()
        && recursive_ctes.is_empty()
        && modifying_ctes.is_empty();
    let row_lock = analyze_row_lock(sel.lock.as_ref(), simple_shape, filter.as_ref())?;

    // `FETCH FIRST n ROWS WITH TIES`: the tie set is defined by the ORDER BY, and
    // the tie trim runs on the sorted, pre-projection rows. It therefore requires an ORDER BY and is
    // not yet supported alongside DISTINCT / DISTINCT ON or a set-returning projection, which change
    // row identity above the sort. (`WITH TIES` without a row count is the same as fetching all rows.)
    let limit_with_ties = sel.limit_with_ties && sel.limit.is_some();
    if limit_with_ties {
        if order_by.is_empty() {
            return Err(Error::Unsupported(
                "FETCH FIRST ... WITH TIES requires an ORDER BY clause".to_owned(),
            ));
        }
        if distinct || !distinct_on.is_empty() {
            return Err(Error::Unsupported(
                "FETCH FIRST ... WITH TIES is not yet supported with DISTINCT / DISTINCT ON"
                    .to_owned(),
            ));
        }
        if projection
            .iter()
            .any(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }))
        {
            return Err(Error::Unsupported(
                "FETCH FIRST ... WITH TIES is not yet supported with a set-returning projection"
                    .to_owned(),
            ));
        }
    }

    Ok(SelectPlan {
        table,
        values: Vec::new(),
        set_op_source: None,
        from_cte: base_cte.map(Box::new),
        joins,
        distinct,
        distinct_on,
        projection,
        filter,
        order_by,
        limit: sel.limit,
        limit_with_ties,
        offset: sel.offset,
        group_keys,
        grouping_sets,
        windows,
        having,
        aggregates,
        indexes,
        table_stats,
        approx_scan_rows,
        recursive_ctes,
        modifying_ctes,
        row_lock,
        ordinality: false,
    })
}

/// Resolve a base table's catalog indexes to planner [`IndexMeta`] — mapping each index's key
/// column names to ordinals into `schema.columns`. An index naming a column that does not
/// resolve is skipped (it cannot be matched safely). Returns empty for a CTE / no-`FROM` base.
fn resolve_table_indexes(
    table: Option<&TableSchema>,
    catalog: &dyn Catalog,
) -> Result<Vec<IndexMeta>, Error> {
    let Some(schema) = table else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for info in catalog.list_indexes(&schema.name)? {
        let mut columns = Vec::with_capacity(info.columns.len());
        let mut ok = true;
        for col in &info.columns {
            let Some(ord) = schema.columns.iter().position(|c| &c.name == col) else {
                ok = false;
                break;
            };
            columns.push(ord);
        }
        if ok && !columns.is_empty() {
            out.push(IndexMeta {
                name: info.name,
                columns,
                unique: info.unique,
            });
        }
    }
    Ok(out)
}

/// A `WITH`-clause CTE resolved to its output schema + how the outer query consumes it.
#[derive(Clone)]
pub(super) struct ResolvedCte {
    /// CTE name; the `FROM` base matches against this.
    name: String,
    /// Synthetic output schema (column names + types) the CTE exposes to the outer query. For a
    /// recursive CTE the schema carries the synthetic table id it is bound to.
    schema: TableSchema,
    /// How the planner/executor sources the CTE's rows.
    source: CteSource,
}

/// How a [`ResolvedCte`]'s rows are produced.
#[derive(Clone)]
enum CteSource {
    /// Non-recursive: the planned body, inlined as the FROM base by the planner. Boxed
    /// because a `SelectPlan` is far larger than the unit `Recursive` variant.
    Inline(Box<SelectPlan>),
    /// Recursive: the CTE resolves to the synthetic table id in its `schema`, materialized
    /// to a fixpoint at execution; the def is carried separately in [`SelectPlan::recursive_ctes`].
    Recursive,
    /// Data-modifying: an `INSERT`/`UPDATE … RETURNING` run once at execution; the CTE
    /// resolves to the synthetic table id in its `schema`, bound to the statement's RETURNING rows.
    /// The def (the DML plan) is carried separately in [`SelectPlan::modifying_ctes`].
    Modifying,
}

/// Combine a block's own resolved CTEs (`own`, which take precedence) with the enclosing `outer`
/// CTEs, dropping any `outer` whose name an `own` CTE shadows.
fn combine_ctes(mut own: Vec<ResolvedCte>, outer: &[ResolvedCte]) -> Vec<ResolvedCte> {
    for o in outer {
        if !own.iter().any(|c| c.name == o.name) {
            own.push(o.clone());
        }
    }
    own
}

/// Resolve the `WITH` clause's CTEs. Non-recursive CTEs inline a plain-`SELECT` body;
/// recursive CTEs yield a fixpoint def materialized at execution. A CTE body sees the earlier
/// siblings in this clause plus any enclosing `outer_ctes`, so `WITH a …, b AS (SELECT … FROM a)`
/// resolves (a set-operation / recursive body does not yet see siblings — a follow-up).
#[allow(
    clippy::type_complexity,
    reason = "the three CTE kinds (inline/recursive/modifying) are returned together; a named tuple \
              struct would not read more clearly at the single call site"
)]
fn analyze_ctes(
    with: &[ast::Cte],
    catalog: &dyn Catalog,
    outer_ctes: &[ResolvedCte],
) -> Result<(Vec<ResolvedCte>, Vec<RecursiveCteDef>, Vec<ModifyingCteDef>), Error> {
    let mut resolved: Vec<ResolvedCte> = Vec::with_capacity(with.len());
    let mut recursive_defs: Vec<RecursiveCteDef> = Vec::new();
    let mut modifying_defs: Vec<ModifyingCteDef> = Vec::new();
    for cte in with {
        if resolved.iter().any(|c| c.name == cte.name) {
            return Err(Error::Unsupported(format!(
                "duplicate CTE name `{}` in the same WITH clause",
                cte.name
            )));
        }
        // A data-modifying CTE: analyze its statement, exposing the RETURNING rows as the
        // synthetic relation. The caller has already forbidden the target everywhere else.
        let query = match &cte.body {
            ast::CteBody::Modifying(stmt) => {
                let (cte_resolved, def) =
                    analyze_modifying_cte(cte, stmt, modifying_defs.len(), catalog)?;
                resolved.push(cte_resolved);
                modifying_defs.push(def);
                continue;
            },
            ast::CteBody::Query(q) => q,
        };
        // `WITH RECURSIVE` permits — but does not require — recursion. A recursive-flagged CTE whose
        // body is a set operation is the genuine recursive case (anchor UNION arm); one with a
        // plain `SELECT` body has no self-reference and is just an ordinary inline CTE.
        if cte.recursive && matches!(&**query, ast::SelectBody::SetOp { .. }) {
            // The synthetic table id is keyed by the CTE's position among the recursive ones.
            let (cte_resolved, def) = analyze_recursive_cte(cte, recursive_defs.len(), catalog)?;
            resolved.push(cte_resolved);
            recursive_defs.push(def);
            continue;
        }
        // A non-recursive CTE body is a plain SELECT or a set operation. The set-op
        // form is inlined exactly like a `(SELECT ... UNION ...) AS x` derived table — a CTE body
        // carries no envelope ORDER BY/LIMIT (rejected at the parser), so wrap it with empty ones.
        // This CTE body may reference any earlier sibling in this WITH plus any enclosing CTE.
        let visible = combine_ctes(resolved.clone(), outer_ctes);
        let plan = match &**query {
            ast::SelectBody::Select(select) => {
                analyze_select_scoped((**select).clone(), catalog, &visible)?
            },
            set_op @ ast::SelectBody::SetOp { .. } => {
                let so = ast::SetOperation {
                    with: Vec::new(),
                    body: set_op.clone(),
                    order_by: Vec::new(),
                    limit: None,
                };
                analyze_set_op_table(so, catalog)?
            },
        };
        let schema = cte_schema(&cte.name, &cte.columns, &plan)?;
        resolved.push(ResolvedCte {
            name: cte.name.clone(),
            schema,
            source: CteSource::Inline(Box::new(plan)),
        });
    }
    Ok((resolved, recursive_defs, modifying_defs))
}

/// The target table a data-modifying statement writes (guard / synthetic schema).
fn dml_target_name(stmt: &ast::Statement) -> Option<&str> {
    match stmt {
        ast::Statement::Insert(i) => Some(&i.table),
        ast::Statement::Update(u) => Some(&u.table),
        ast::Statement::Delete(d) => Some(&d.table),
        _ => None,
    }
}

/// Synthetic table id for the `index`-th data-modifying CTE. Reserved from the top of the
/// `u64` space, below the recursive-CTE band, so it collides with neither engine ids nor recursive ones.
const fn synthetic_modifying_table_id(index: usize) -> nusadb_core::TableId {
    nusadb_core::TableId(u64::MAX - 1_000_000 - index as u64)
}

/// Resolve a data-modifying CTE `WITH x AS (INSERT/UPDATE … RETURNING …)`: analyze the
/// statement (its own target temporarily un-forbidden, since it is the modification), expose the
/// `RETURNING` columns as the CTE's synthetic relation, and carry the plan for once-only execution.
fn analyze_modifying_cte(
    cte: &ast::Cte,
    stmt: &ast::Statement,
    index: usize,
    catalog: &dyn Catalog,
) -> Result<(ResolvedCte, ModifyingCteDef), Error> {
    let target = dml_target_name(stmt).ok_or_else(|| {
        Error::Unsupported("a data-modifying CTE must be INSERT/UPDATE/DELETE".to_owned())
    })?;
    // The statement modifies (and may read) its own target; lift the forbid for just this analysis.
    let logical = {
        let _allow = super::allow_dml_target(target);
        super::analyze(stmt.clone(), catalog)?
    };
    let returning = match &logical {
        crate::planner::LogicalPlan::Insert(p) => &p.returning,
        crate::planner::LogicalPlan::Update(p) => &p.returning,
        crate::planner::LogicalPlan::Delete(p) => &p.returning,
        _ => {
            return Err(Error::Unsupported(
                "a data-modifying CTE must be INSERT/UPDATE/DELETE".to_owned(),
            ));
        },
    };
    if returning.is_empty() {
        return Err(Error::Unsupported(format!(
            "data-modifying CTE `{}` must have a RETURNING clause",
            cte.name
        )));
    }
    let width = returning.len();
    if !cte.columns.is_empty() && cte.columns.len() != width {
        return Err(Error::Unsupported(format!(
            "CTE `{}` declares {} column name(s) but its statement returns {width}",
            cte.name,
            cte.columns.len()
        )));
    }
    let columns: Vec<ColumnDef> = returning
        .iter()
        .enumerate()
        .map(|(i, proj)| ColumnDef {
            name: cte
                .columns
                .get(i)
                .cloned()
                .unwrap_or_else(|| proj.name.clone()),
            ty: proj.expr.ty,
            nullable: true,
        })
        .collect();
    let schema = TableSchema {
        schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
        id: synthetic_modifying_table_id(index),
        name: cte.name.clone(),
        columns,
    };
    let def = ModifyingCteDef {
        id: schema.id,
        plan: Box::new(logical),
    };
    let resolved = ResolvedCte {
        name: cte.name.clone(),
        schema,
        source: CteSource::Modifying,
    };
    Ok((resolved, def))
}

/// Resolve a `WITH RECURSIVE` CTE of the form `<base> UNION [ALL] <recursive>`. The base
/// term fixes the CTE's column shape; the recursive term is resolved against a catalog that knows
/// the CTE's synthetic table (so its self-reference type-checks), and must produce the same shape.
fn analyze_recursive_cte(
    cte: &ast::Cte,
    recursive_index: usize,
    catalog: &dyn Catalog,
) -> Result<(ResolvedCte, RecursiveCteDef), Error> {
    let ast::CteBody::Query(query) = &cte.body else {
        return Err(Error::Unsupported(format!(
            "recursive CTE `{}` cannot be a data-modifying statement",
            cte.name
        )));
    };
    let ast::SelectBody::SetOp {
        op,
        all,
        left,
        right,
    } = &**query
    else {
        return Err(Error::Unsupported(format!(
            "recursive CTE `{}` must combine a base and a recursive term with UNION [ALL]",
            cte.name
        )));
    };
    if !matches!(op, ast::SetOp::Union) {
        return Err(Error::Unsupported(format!(
            "recursive CTE `{}` must use UNION [ALL], not INTERSECT/EXCEPT",
            cte.name
        )));
    }
    let (ast::SelectBody::Select(base_select), ast::SelectBody::Select(rec_select)) =
        (&**left, &**right)
    else {
        return Err(Error::Unsupported(format!(
            "recursive CTE `{}` supports a single base and single recursive SELECT term; nested set \
             operations in either term are not yet supported",
            cte.name
        )));
    };
    // The base term must not reference the CTE: resolve it against the plain catalog. Its projection
    // fixes the CTE's column names and types.
    let base_plan = analyze_select((**base_select).clone(), catalog)?;
    let mut schema = cte_schema(&cte.name, &cte.columns, &base_plan)?;
    schema.id = synthetic_recursive_table_id(recursive_index);
    // The recursive term references the CTE by name; expose its synthetic schema so the
    // self-reference resolves to a scan of the synthetic table.
    let cte_catalog = CteCatalog {
        inner: catalog,
        name: &cte.name,
        schema: &schema,
    };
    let recursive_plan = analyze_select((**rec_select).clone(), &cte_catalog)?;
    if recursive_plan.projection.len() != base_plan.projection.len() {
        return Err(Error::ArityMismatch {
            context: format!("recursive CTE `{}` term", cte.name),
            expected: base_plan.projection.len(),
            found: recursive_plan.projection.len(),
        });
    }
    for (base_col, rec_col) in base_plan.projection.iter().zip(&recursive_plan.projection) {
        if base_col.expr.ty != rec_col.expr.ty {
            return Err(Error::TypeMismatch {
                context: format!("recursive CTE `{}` column", cte.name),
                expected: base_col.expr.ty,
                found: rec_col.expr.ty,
            });
        }
    }
    let def = RecursiveCteDef {
        id: schema.id,
        base: Box::new(base_plan),
        recursive: Box::new(recursive_plan),
        union_all: *all,
    };
    let resolved = ResolvedCte {
        name: cte.name.clone(),
        schema,
        source: CteSource::Recursive,
    };
    Ok((resolved, def))
}

/// Synthetic [`nusadb_core::TableId`] for the `index`-th recursive CTE of a query. Reserved
/// from the top of the `u64` space so it never collides with an engine-assigned table id; the
/// executor's working-set registry binds the CTE's rows under this id and a scan reads them back.
const fn synthetic_recursive_table_id(index: usize) -> nusadb_core::TableId {
    nusadb_core::TableId(u64::MAX - index as u64)
}

/// A [`Catalog`] that overlays one recursive CTE's synthetic table over an inner catalog, so the
/// CTE's recursive term resolves its self-reference. All other lookups delegate to `inner`.
struct CteCatalog<'a> {
    inner: &'a dyn Catalog,
    name: &'a str,
    schema: &'a TableSchema,
}

impl Catalog for CteCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        if name == self.name {
            return Ok(Some(self.schema.clone()));
        }
        self.inner.lookup_table(name)
    }

    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        // The synthetic recursive-CTE table is unqualified; a schema qualifier always denotes
        // a real base table, so delegate qualified resolution straight to the inner catalog.
        if schema == nusadb_core::PUBLIC_SCHEMA && name == self.name {
            return Ok(Some(self.schema.clone()));
        }
        self.inner.lookup_table_in(schema, name)
    }

    fn search_path(&self) -> Vec<String> {
        // Forward the session search path: a recursive CTE term that references an unqualified
        // base table must resolve it under the real session path, not the default `public`.
        self.inner.search_path()
    }

    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        // The synthetic CTE table carries no indexes; everything else delegates.
        if table == self.name {
            return Ok(Vec::new());
        }
        self.inner.list_indexes(table)
    }

    // The security context must delegate to the inner catalog: otherwise a recursive CTE's term,
    // analyzed through this overlay, would fall back to the trait defaults (superuser, no RLS) and
    // a non-superuser could read an RLS-enabled base table by wrapping it in a CTE.
    fn is_superuser(&self) -> bool {
        self.inner.is_superuser()
    }

    fn rls_enabled(&self, name: &str) -> Result<bool, Error> {
        // The synthetic CTE table is derived, never a base table, so it carries no RLS flag.
        if name == self.name {
            return Ok(false);
        }
        self.inner.rls_enabled(name)
    }
}

/// Build a CTE's output [`TableSchema`] from its planned projection plus an optional explicit
/// column-name list. CTE columns are conservatively typed as nullable.
fn cte_schema(name: &str, explicit: &[String], plan: &SelectPlan) -> Result<TableSchema, Error> {
    let base_width = plan.projection.len();
    // `WITH ORDINALITY` adds one trailing column to the relation.
    let width = base_width + usize::from(plan.ordinality);
    if !explicit.is_empty() && explicit.len() != width {
        return Err(Error::Unsupported(format!(
            "CTE `{name}` declares {} column name(s) but its query returns {width}",
            explicit.len()
        )));
    }
    let mut columns: Vec<ColumnDef> = plan
        .projection
        .iter()
        .enumerate()
        .map(|(i, proj)| ColumnDef {
            name: explicit
                .get(i)
                .cloned()
                .unwrap_or_else(|| proj.name.clone()),
            ty: proj.expr.ty,
            nullable: true,
        })
        .collect();
    // The appended `WITH ORDINALITY` column is a 1-based row number (a non-null BIGINT, named
    // `ordinality` unless the alias list renames it).
    if plan.ordinality {
        columns.push(ColumnDef {
            name: explicit
                .get(base_width)
                .cloned()
                .unwrap_or_else(|| "ordinality".to_owned()),
            ty: ColumnType::BigInt,
            nullable: false,
        });
    }
    Ok(TableSchema {
        schema: "public".to_owned(),
        id: nusadb_core::TableId(0),
        name: name.to_owned(),
        columns,
    })
}

/// How a `GROUP BY` clause groups its input: a single implicit set over all keys,
/// or an explicit list of grouping sets expanded from ROLLUP/CUBE/GROUPING SETS.
pub(super) enum GroupingSpec {
    /// Plain `GROUP BY a, b` — one grouping set over every key.
    Plain(Vec<ast::Expr>),
    /// `ROLLUP`/`CUBE`/`GROUPING SETS` expanded to explicit grouping sets, each a
    /// (possibly empty) list of key columns.
    Sets(Vec<Vec<ast::Expr>>),
}

/// Largest `CUBE` width allowed. `CUBE(e₁ … eₙ)` expands to `2ⁿ` grouping sets,
/// so an unbounded `n` would overflow the shift and explode memory; cap it.
const MAX_CUBE_ELEMENTS: usize = 16;

/// Expand a `GROUP BY` clause into its grouping sets. `ROLLUP(a, b)`
/// yields the prefixes `(a, b), (a), ()`; `CUBE(a, b)` yields every subset
/// `(a, b), (b), (a), ()`; `GROUPING SETS(...)` is taken verbatim.
pub(super) fn expand_grouping(group_by: &ast::GroupBy) -> Result<GroupingSpec, Error> {
    Ok(match group_by {
        ast::GroupBy::Expressions(keys) => GroupingSpec::Plain(keys.clone()),
        ast::GroupBy::GroupingSets(sets) => GroupingSpec::Sets(sets.clone()),
        ast::GroupBy::Rollup(elements) => {
            // Prefixes of the element list, from the full set down to the empty (grand-total) set.
            let sets = (0..=elements.len())
                .rev()
                .map(|len| elements.iter().take(len).flatten().cloned().collect())
                .collect();
            GroupingSpec::Sets(sets)
        },
        ast::GroupBy::Cube(elements) => {
            // Every subset of the elements (2^n), highest bitmask first so the full set leads.
            let n = elements.len();
            if n > MAX_CUBE_ELEMENTS {
                return Err(Error::Unsupported(format!(
                    "CUBE with {n} elements is too large (max {MAX_CUBE_ELEMENTS}; it expands to 2^n grouping sets)"
                )));
            }
            let sets = (0..(1usize << n))
                .rev()
                .map(|mask| {
                    elements
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .flat_map(|(_, element)| element.iter().cloned())
                        .collect()
                })
                .collect();
            GroupingSpec::Sets(sets)
        },
    })
}

/// Resolve one `GROUP BY` key expression and return its slot in `group_keys`, appending it on first
/// sight. A grouped column is the common case, but any scalar expression (`GROUP BY a + b`,
/// `GROUP BY lower(name)`) is accepted — the aggregation operator evaluates each key per row, and
/// the projection's matching sub-expression collapses to the key via [`rebase_onto_aggregation`].
/// A bare integer literal is a **positional** reference to the Nth output column (`GROUP BY 1`, like
/// `ORDER BY 1`), resolved to that projection expression before analysis. Any other bare constant is
/// rejected: it is not a meaningful grouping.
pub(super) fn intern_group_key(
    expr: &ast::Expr,
    projection: &[ast::SelectItem],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    group_keys: &mut Vec<TypedExpr>,
) -> Result<usize, Error> {
    // Resolve a positional reference once, then analyze the underlying expression (never as another
    // position — so `SELECT 5, count(*) ... GROUP BY 1` still reports grouping by a constant).
    let effective = match expr {
        ast::Expr::Literal(ast::Value::Int(n)) => positional_group_key(*n, projection)?,
        other => other,
    };
    let typed = analyze_expr(effective, scope, catalog, None)?;
    if matches!(typed.kind, TypedExprKind::Literal(_)) {
        return Err(Error::Unsupported(
            "GROUP BY on a constant is not supported — group by a column or expression".to_owned(),
        ));
    }
    // De-duplicate by structural equality so `GROUP BY a, a` and a repeated expression share a slot.
    if let Some(slot) = group_keys.iter().position(|key| *key == typed) {
        return Ok(slot);
    }
    group_keys.push(typed);
    Ok(group_keys.len() - 1)
}

/// Resolve a positional `GROUP BY <n>` to the expression of the Nth (1-based) output column. A
/// position over a `*` / `table.*` wildcard, or out of range, is a loud error rather than a
/// silently-ignored grouping.
fn positional_group_key(n: i64, projection: &[ast::SelectItem]) -> Result<&ast::Expr, Error> {
    let item = usize::try_from(n)
        .ok()
        .filter(|&p| p >= 1)
        .and_then(|p| projection.get(p - 1));
    match item {
        Some(ast::SelectItem::Expr { expr, .. }) => Ok(expr),
        Some(_) => Err(Error::Unsupported(
            "positional GROUP BY over a `*` / `table.*` wildcard is not supported — name the column"
                .to_owned(),
        )),
        None => Err(Error::Unsupported(format!(
            "GROUP BY position {n} is out of range (1..={})",
            projection.len()
        ))),
    }
}

/// Rewrite a type-checked output expression so its column references point at
/// the aggregation operator's output row `[group keys ++ aggregate results]`:
/// a group-key column becomes `AggregateRef(k)`, an existing `AggregateRef`
/// shifts past the `num_group_keys` leading key columns, and any other bare
/// column is rejected (it must appear in `GROUP BY` or inside an aggregate).
/// Context threaded through [`rebase_onto_aggregation`]: the resolved `group_keys` (and their count,
/// the post-aggregation row's leading width), the query's `grouping_sets` (empty for a plain
/// `GROUP BY`), and the mutable aggregate sink — the last two only so a `GROUPING(...)` node can be
/// rewritten into a synthetic [`ast::AggregateFunc::Grouping`] slot (or a constant `0`).
pub(super) struct AggRebase<'a> {
    /// The resolved `GROUP BY` key expressions, in post-aggregation-row column order.
    pub group_keys: &'a [TypedExpr],
    /// `group_keys.len()` — how far aggregate-result slots shift to clear the leading key columns.
    pub num_group_keys: usize,
    /// The query's grouping sets (each a list of active `group_keys` indices); empty for a plain
    /// `GROUP BY`, where every `GROUPING(...)` is the constant `0`.
    pub grouping_sets: &'a [Vec<usize>],
    /// The aggregate sink; a `GROUPING(...)` call appends one synthetic [`AggregateCall`] here.
    pub aggregates: &'a mut Vec<AggregateCall>,
    /// The source-row width. A projection column reference at or above this ordinal is a synthetic
    /// window-output column (not a base column), remapped onto the post-aggregation row rather than
    /// rejected. `0` when there are no windows.
    pub source_len: usize,
    /// The post-aggregation row width `[group keys ++ aggregates]` — the base at which the Window
    /// operator appends its output columns, so window-column references remap to `post_agg_width + i`.
    pub post_agg_width: usize,
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per TypedExprKind variant; length tracks the expression grammar"
)]
pub(super) fn rebase_onto_aggregation(
    expr: &mut TypedExpr,
    ctx: &mut AggRebase<'_>,
) -> Result<(), Error> {
    // `GROUPING(key, ...)` (carried as a `ScalarFunction { Grouping, .. }` from analysis): resolve
    // each argument to its `group_keys` index and rewrite into the runtime bitmask reference, *before*
    // the whole-key match below (a `GROUPING` node is never itself a group key). A plain `GROUP BY`
    // (no grouping sets) folds to the constant `0` — every key is always present.
    if let TypedExprKind::ScalarFunction {
        func: ast::ScalarFunc::Grouping,
        args,
    } = &mut expr.kind
    {
        // Move the args out first so the mutable borrow on `expr.kind` ends before we hand `expr` to
        // the resolver (which overwrites `expr.kind`).
        let args = std::mem::take(args);
        return rebase_grouping_call(expr, &args, ctx);
    }

    // A whole sub-expression that *is* a GROUP BY key collapses to that key's slot in the
    // post-aggregation row. This covers a grouped column (`GROUP BY a`, then `SELECT a`) and a
    // grouped expression alike (`GROUP BY a + b`, then `SELECT a + b`). Checked at every level, so
    // it fires on the largest matching sub-expression before recursing into its children.
    if let Some(slot) = ctx.group_keys.iter().position(|key| key == &*expr) {
        expr.kind = TypedExprKind::AggregateRef(slot);
        return Ok(());
    }

    match &mut expr.kind {
        TypedExprKind::AggregateRef(idx) => *idx += ctx.num_group_keys,
        // A synthetic window-output column (ordinal at or above the source width): the Window operator
        // runs above the aggregate and appends its columns after `[group keys ++ aggregates]`, so
        // remap the reference from its source-relative ordinal onto the post-aggregation base.
        TypedExprKind::Column(ord) if *ord >= ctx.source_len => {
            *ord = ctx.post_agg_width + (*ord - ctx.source_len);
        },
        // A bare (base) column that is not (part of) a GROUP BY key is neither grouped nor aggregated.
        TypedExprKind::Column(_) => {
            return Err(Error::Unsupported(
                "column must appear in GROUP BY or inside an aggregate function".to_owned(),
            ));
        },
        // Literals carry no references. An `OuterColumn` resolves from an enclosing query's row, so
        // it is constant w.r.t. this query's aggregation and is not rebased. A scalar/EXISTS
        // subquery body has its own scope, so it carries no this-level column ref to rebase either.
        TypedExprKind::Literal(_)
        | TypedExprKind::OuterColumn { .. }
        | TypedExprKind::ScalarSubquery(_)
        // A set-returning function never coexists with aggregation (rejected in `analyze_select`), so
        // it is unreachable here; nothing to rebase.
        | TypedExprKind::SetReturning { .. }
        | TypedExprKind::Exists { .. } => {},
        TypedExprKind::Binary { left, right, .. }
        | TypedExprKind::IsDistinctFrom { left, right, .. } => {
            rebase_onto_aggregation(left, ctx)?;
            rebase_onto_aggregation(right, ctx)?;
        },
        // `IN (subquery)` rebases its (outer-referencing) probe; the single-child unary/cast
        // forms rebase their lone operand the same way.
        TypedExprKind::Unary { expr, .. }
        | TypedExprKind::IsNull { expr, .. }
        | TypedExprKind::IsBool { expr, .. }
        | TypedExprKind::Cast(expr, _)
        | TypedExprKind::InSubquery { expr, .. }
        | TypedExprKind::QuantifiedSubquery { expr, .. } => {
            rebase_onto_aggregation(expr, ctx)?;
        },
        TypedExprKind::QuantifiedArray { expr, array, .. } => {
            rebase_onto_aggregation(expr, ctx)?;
            rebase_onto_aggregation(array, ctx)?;
        },
        TypedExprKind::InList { expr, list, .. } => {
            rebase_onto_aggregation(expr, ctx)?;
            for item in list {
                rebase_onto_aggregation(item, ctx)?;
            }
        },
        TypedExprKind::Between {
            expr, low, high, ..
        } => {
            rebase_onto_aggregation(expr, ctx)?;
            rebase_onto_aggregation(low, ctx)?;
            rebase_onto_aggregation(high, ctx)?;
        },
        TypedExprKind::Like { expr, pattern, .. }
        | TypedExprKind::RegexMatch { expr, pattern, .. }
        | TypedExprKind::SimilarTo { expr, pattern, .. } => {
            rebase_onto_aggregation(expr, ctx)?;
            rebase_onto_aggregation(pattern, ctx)?;
        },
        TypedExprKind::Case {
            operand,
            branches,
            default,
        } => {
            if let Some(operand) = operand.as_mut() {
                rebase_onto_aggregation(operand, ctx)?;
            }
            for branch in branches.iter_mut() {
                rebase_onto_aggregation(&mut branch.when, ctx)?;
                rebase_onto_aggregation(&mut branch.then, ctx)?;
            }
            if let Some(default) = default.as_mut() {
                rebase_onto_aggregation(default, ctx)?;
            }
        },
        TypedExprKind::Coalesce(items) => {
            for item in items {
                rebase_onto_aggregation(item, ctx)?;
            }
        },
        TypedExprKind::Crypto { value, key, .. } => {
            rebase_onto_aggregation(value, ctx)?;
            rebase_onto_aggregation(key, ctx)?;
        },
        TypedExprKind::ScalarFunction { args, .. }
        | TypedExprKind::ScalarUdf { args, .. }
        | TypedExprKind::ArrayLiteral(args) => {
            for arg in args {
                rebase_onto_aggregation(arg, ctx)?;
            }
        },
        TypedExprKind::Subscript { base, index } => {
            rebase_onto_aggregation(base, ctx)?;
            rebase_onto_aggregation(index, ctx)?;
        },
        TypedExprKind::ArraySlice { base, lower, upper } => {
            rebase_onto_aggregation(base, ctx)?;
            for bound in [lower, upper].into_iter().flatten() {
                rebase_onto_aggregation(bound, ctx)?;
            }
        },
    }
    Ok(())
}

/// Rewrite a `GROUPING(arg, ...)` call into its runtime form. Each argument must name a
/// `GROUP BY` key (matched structurally against `group_keys`); otherwise it is an error. With no
/// grouping sets (plain `GROUP BY`) every key is always present, so the result is the constant `0`.
/// Otherwise a synthetic [`ast::AggregateFunc::Grouping`] call is appended to the sink carrying the
/// resolved key indices (leftmost argument = most-significant bit), and the node becomes an
/// [`TypedExprKind::AggregateRef`] to that slot — shifted past the leading key columns like any
/// aggregate ref.
fn rebase_grouping_call(
    expr: &mut TypedExpr,
    args: &[TypedExpr],
    ctx: &mut AggRebase<'_>,
) -> Result<(), Error> {
    let mut grouping_args = Vec::with_capacity(args.len());
    for arg in args {
        let slot = ctx
            .group_keys
            .iter()
            .position(|key| key == arg)
            .ok_or_else(|| {
                Error::Unsupported(
                    "GROUPING arguments must be expressions that appear in GROUP BY".to_owned(),
                )
            })?;
        grouping_args.push(slot);
    }
    if ctx.grouping_sets.is_empty() {
        // Plain GROUP BY: nothing is ever grouped away, so GROUPING(...) is a constant 0.
        expr.kind = TypedExprKind::Literal(ast::Value::Int(0));
        return Ok(());
    }
    let idx = ctx.aggregates.len();
    ctx.aggregates.push(AggregateCall {
        func: ast::AggregateFunc::Grouping,
        arg: None,
        result_ty: ColumnType::Int,
        distinct: false,
        fraction: None,
        ordered_set_descending: false,
        filter: None,
        separator: None,
        arg2: None,
        order_by: Vec::new(),
        grouping_args,
    });
    // The ref is relative to the aggregate sink; shift it past the leading key columns, exactly as
    // the `AggregateRef` arm does for ordinary aggregates.
    expr.kind = TypedExprKind::AggregateRef(idx + ctx.num_group_keys);
    Ok(())
}

pub(super) fn analyze_projection(
    items: Vec<ast::SelectItem>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: &mut Vec<AggregateCall>,
    source_len: usize,
) -> Result<Vec<Projection>, Error> {
    // `*` / `table.*` expand only the `source_len` real source columns, never the synthetic
    // window-result columns the caller may have appended to `scope`.
    let visible = scope.get(..source_len).unwrap_or(scope);
    let mut projection = Vec::with_capacity(items.len());
    for item in items {
        match item {
            ast::SelectItem::Wildcard => {
                if visible.is_empty() {
                    return Err(Error::Unsupported(
                        "SELECT * requires a FROM clause".to_owned(),
                    ));
                }
                // `*` expands to every column of every table in scope, in order, except a column
                // hidden by a `USING`/`NATURAL` merge (the right side's coalesced copy) — it appears
                // once, via the visible left column.
                for (index, col) in visible.iter().enumerate() {
                    if col.qualified_only {
                        continue;
                    }
                    projection.push(Projection {
                        expr: TypedExpr {
                            kind: TypedExprKind::Column(index),
                            // Physical type: VARCHAR/CHAR project as TEXT (declared length is
                            // catalog metadata only).
                            ty: col.def.ty.physical(),
                        },
                        name: col.def.name.clone(),
                    });
                }
            },
            ast::SelectItem::QualifiedWildcard(table) => {
                // `table.*` expands to every column owned by that table/alias, in order.
                let mut matched = false;
                for (index, col) in visible.iter().enumerate() {
                    if col.qualifier != table {
                        continue;
                    }
                    matched = true;
                    projection.push(Projection {
                        expr: TypedExpr {
                            kind: TypedExprKind::Column(index),
                            // Physical type: VARCHAR/CHAR project as TEXT (declared length is
                            // catalog metadata only).
                            ty: col.def.ty.physical(),
                        },
                        name: col.def.name.clone(),
                    });
                }
                if !matched {
                    return Err(Error::Unsupported(format!(
                        "`{table}.*` refers to a table not in the FROM clause"
                    )));
                }
            },
            // A set-returning function is valid only at the top of a SELECT-list item, so it
            // is resolved here rather than through the general expression path (which rejects it).
            ast::SelectItem::Expr {
                expr: ast::Expr::SetReturning { func, args },
                alias,
            } => {
                let typed = analyze_set_returning(func, &args, scope, catalog)?;
                projection.push(Projection {
                    expr: typed,
                    name: alias.unwrap_or_else(|| func.name().to_owned()),
                });
            },
            ast::SelectItem::Expr { expr, alias } => {
                let typed = analyze_projection_expr(&expr, scope, catalog, aggregates)?;
                projection.push(Projection {
                    expr: typed,
                    name: alias.unwrap_or_else(|| projection_name(&expr)),
                });
            },
        }
    }
    Ok(projection)
}

/// Analyze one projection-list expression with aggregate capture enabled.
/// An aggregate appearing anywhere inside — at the top level or nested within
/// any composite (`CAST(SUM(x) AS FLOAT)`, `COALESCE(MAX(x), 0)`,
/// `COUNT(*) > 0`, `SUM(x) IS NULL`, ...) — is registered in `aggregates` and
/// replaced with an [`TypedExprKind::AggregateRef`] so the surrounding
/// expression composes freely. Type-checking is shared with [`analyze_expr`]
/// via [`analyze_expr_agg`]; there is no separate projection-only type rule.
pub(super) fn analyze_projection_expr(
    expr: &ast::Expr,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
    aggregates: &mut Vec<AggregateCall>,
) -> Result<TypedExpr, Error> {
    // A bare `SELECT NULL` has no type context; default its column to `TEXT` (the standard "unknown"
    // resolution) rather than rejecting it. Any composite NULL (`NULL + NULL`, etc.) still gets no
    // hint and is rejected if genuinely untypable.
    let hint = matches!(expr, ast::Expr::Literal(ast::Value::Null)).then_some(ColumnType::Text);
    analyze_expr_agg(expr, scope, catalog, hint, Some(aggregates))
}

/// Resolve a set-returning function at the top of a `SELECT`-list item. `UNNEST(arr)` takes
/// one array argument and produces that array's element type per output row; the
/// [`PhysicalOperator::ProjectSet`] operator performs the row expansion at execution.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-set-returning-function dispatch; length tracks the function set"
)]
fn analyze_set_returning(
    func: ast::SetReturningFunc,
    args: &[ast::Expr],
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<TypedExpr, Error> {
    use ast::SetReturningFunc as Srf;
    // Each set-returning built-in has a fixed argument shape and per-row element type.
    let (expected, element_ty): (&[ColumnType], ColumnType) = match func {
        // UNNEST(arr): one array argument; element type comes from the array.
        Srf::Unnest => {
            let [arg_expr] = args else {
                return Err(srf_arity_error(func, 1, args.len()));
            };
            let arg = analyze_expr(arg_expr, scope, catalog, None)?;
            let ColumnType::Array(elem) = arg.ty else {
                return Err(Error::Unsupported(format!(
                    "unnest() expects an array argument, got {:?}",
                    arg.ty
                )));
            };
            return Ok(TypedExpr {
                kind: TypedExprKind::SetReturning {
                    func,
                    args: vec![arg],
                },
                ty: elem.column_type(),
            });
        },
        // GENERATE_SERIES(start, stop [, step]): the integer form takes two or three INT args → one
        // INT per row; the temporal form takes (start, stop, interval step) over DATE/TIMESTAMP[TZ] →
        // one timestamp per row. The first argument's type selects the form.
        Srf::GenerateSeries => {
            use ColumnType::{Date, Int, Interval, Timestamp, TimestampTz};
            let Some((first_arg, rest)) = args.split_first() else {
                return Err(srf_arity_error(func, 2, args.len()));
            };
            if rest.is_empty() || rest.len() > 2 {
                return Err(srf_arity_error(func, 2, args.len()));
            }
            let start = analyze_expr(first_arg, scope, catalog, None)?;
            if matches!(start.ty, Date | Timestamp | TimestampTz) {
                let [stop_e, step_e] = rest else {
                    return Err(Error::Unsupported(
                        "generate_series over a temporal range requires (start, stop, interval step)"
                            .to_owned(),
                    ));
                };
                let stop = analyze_expr(stop_e, scope, catalog, Some(start.ty))?;
                if !matches!(stop.ty, Date | Timestamp | TimestampTz) && !is_null_literal(&stop) {
                    return Err(Error::TypeMismatch {
                        context: "generate_series stop".to_owned(),
                        expected: start.ty,
                        found: stop.ty,
                    });
                }
                let step = analyze_expr(step_e, scope, catalog, Some(Interval))?;
                if step.ty != Interval && !is_null_literal(&step) {
                    return Err(Error::TypeMismatch {
                        context: "generate_series step".to_owned(),
                        expected: Interval,
                        found: step.ty,
                    });
                }
                // A timestamptz range yields timestamptz; a date or timestamp range yields timestamp.
                let element_ty = if start.ty == TimestampTz {
                    TimestampTz
                } else {
                    Timestamp
                };
                return Ok(TypedExpr {
                    kind: TypedExprKind::SetReturning {
                        func,
                        args: vec![start, stop, step],
                    },
                    ty: element_ty,
                });
            }
            // Integer form.
            if start.ty != Int && !is_null_literal(&start) {
                return Err(Error::TypeMismatch {
                    context: format!("{}() argument", func.name()),
                    expected: Int,
                    found: start.ty,
                });
            }
            let mut typed_args = vec![start];
            for arg in rest {
                let typed = analyze_expr(arg, scope, catalog, Some(Int))?;
                if typed.ty != Int && !is_null_literal(&typed) {
                    return Err(Error::TypeMismatch {
                        context: format!("{}() argument", func.name()),
                        expected: Int,
                        found: typed.ty,
                    });
                }
                typed_args.push(typed);
            }
            return Ok(TypedExpr {
                kind: TypedExprKind::SetReturning {
                    func,
                    args: typed_args,
                },
                ty: Int,
            });
        },
        // REGEXP_SPLIT_TO_TABLE(s, pattern [, flags]): two or three TEXT arguments → one TEXT per
        // split piece.
        Srf::RegexpSplitToTable => {
            if args.len() != 2 && args.len() != 3 {
                return Err(srf_arity_error(func, 2, args.len()));
            }
            let mut typed_args = Vec::with_capacity(args.len());
            for arg in args {
                let typed = analyze_expr(arg, scope, catalog, Some(ColumnType::Text))?;
                if typed.ty != ColumnType::Text && !is_null_literal(&typed) {
                    return Err(Error::TypeMismatch {
                        context: format!("{}() argument", func.name()),
                        expected: ColumnType::Text,
                        found: typed.ty,
                    });
                }
                typed_args.push(typed);
            }
            return Ok(TypedExpr {
                kind: TypedExprKind::SetReturning {
                    func,
                    args: typed_args,
                },
                ty: ColumnType::Text,
            });
        },
        // REGEXP_MATCHES(s, pattern [, flags]) → one TEXT[] row per match's capture groups. Same
        // (variadic) argument shape as REGEXP_SPLIT_TO_TABLE, but each output row is a TEXT[].
        Srf::RegexpMatches => {
            if args.len() != 2 && args.len() != 3 {
                return Err(srf_arity_error(func, 2, args.len()));
            }
            let mut typed_args = Vec::with_capacity(args.len());
            for arg in args {
                let typed = analyze_expr(arg, scope, catalog, Some(ColumnType::Text))?;
                if typed.ty != ColumnType::Text && !is_null_literal(&typed) {
                    return Err(Error::TypeMismatch {
                        context: format!("{}() argument", func.name()),
                        expected: ColumnType::Text,
                        found: typed.ty,
                    });
                }
                typed_args.push(typed);
            }
            return Ok(TypedExpr {
                kind: TypedExprKind::SetReturning {
                    func,
                    args: typed_args,
                },
                ty: ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
            });
        },
        // JSON_ARRAY_ELEMENTS(json) → JSON per element.
        Srf::JsonArrayElements => (&[ColumnType::Json], ColumnType::Json),
        // JSONB_PATH_QUERY(json, path) → JSON per match.
        Srf::JsonPathQuery => (&[ColumnType::Json, ColumnType::Text], ColumnType::Json),
        // JSONB_OBJECT_KEYS(json) → TEXT per top-level key; JSONB_ARRAY_ELEMENTS_TEXT(json) →
        // TEXT per array element.
        Srf::JsonObjectKeys | Srf::JsonArrayElementsText => (&[ColumnType::Json], ColumnType::Text),
        // STRING_TO_TABLE(s, sep) → TEXT per split piece.
        Srf::StringToTable => (&[ColumnType::Text, ColumnType::Text], ColumnType::Text),
    };
    if args.len() != expected.len() {
        return Err(srf_arity_error(func, expected.len(), args.len()));
    }
    let mut typed_args = Vec::with_capacity(args.len());
    for (arg, want) in args.iter().zip(expected) {
        let typed = analyze_expr(arg, scope, catalog, Some(*want))?;
        // A bare string literal for a JSON argument is coerced to that type (the unknown-literal
        // rule), so `jsonb_object_keys('{...}')` type-checks like `'...'::json`.
        let typed = coerce_text_literal_to(typed, *want);
        if typed.ty != *want && !is_null_literal(&typed) {
            return Err(Error::TypeMismatch {
                context: format!("{}() argument", func.name()),
                expected: *want,
                found: typed.ty,
            });
        }
        typed_args.push(typed);
    }
    Ok(TypedExpr {
        kind: TypedExprKind::SetReturning {
            func,
            args: typed_args,
        },
        ty: element_ty,
    })
}

/// Arity error for a set-returning function call.
fn srf_arity_error(func: ast::SetReturningFunc, expected: usize, got: usize) -> Error {
    Error::Unsupported(format!(
        "{}() expects {expected} argument(s), got {got}",
        func.name()
    ))
}

/// Type-check one aggregate call and return `(typed_arg, result_ty)`.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-aggregate dispatch; grows with the aggregate surface, not complexity"
)]
pub(super) fn analyze_aggregate(
    func: ast::AggregateFunc,
    arg: Option<&ast::Expr>,
    scope: &[ScopedColumn],
    catalog: &dyn Catalog,
) -> Result<(Option<TypedExpr>, ColumnType), Error> {
    match (func, arg) {
        // COUNT(*) — no arg; counts every input row.
        (ast::AggregateFunc::Count, None) => Ok((None, ColumnType::Int)),
        // COUNT(expr) — counts non-NULL `expr`.
        (ast::AggregateFunc::Count, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            Ok((Some(typed), ColumnType::Int))
        },
        // SUM(expr) / AVG(expr) — numeric only. SUM(Int)→Int, SUM(Float)→Float.
        (ast::AggregateFunc::Sum | ast::AggregateFunc::Avg, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            if !is_numeric(typed.ty) {
                return Err(Error::TypeMismatch {
                    context: format!("{func:?} requires a numeric argument"),
                    expected: ColumnType::Int,
                    found: typed.ty,
                });
            }
            // SUM keeps the argument type (Int->Int, Float->Float, Numeric->Numeric). AVG over an
            // exact type (Int / NUMERIC) is exact NUMERIC — `AVG(int)` must not lose precision in
            // f64 (Temuan-4); only AVG of FLOAT stays FLOAT.
            let unconstrained_numeric = ColumnType::Numeric {
                precision: 0,
                scale: 0,
            };
            let result_ty = if matches!(func, ast::AggregateFunc::Avg) {
                if typed.ty == ColumnType::Float {
                    ColumnType::Float
                } else {
                    unconstrained_numeric
                }
            } else {
                typed.ty
            };
            Ok((Some(typed), result_ty))
        },
        // MIN(expr) / MAX(expr) — any comparable type; result keeps the type.
        (ast::AggregateFunc::Min | ast::AggregateFunc::Max, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            Ok((Some(typed.clone()), typed.ty))
        },
        // ARRAY_AGG(expr) — collect values into an array; the element type is `expr`'s type, which
        // must be array-able (a scalar; not NUMERIC/JSON/nested-array).
        (ast::AggregateFunc::ArrayAgg, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            // A NUMERIC argument (e.g. `array_agg(0.5)` — a decimal literal now types as NUMERIC)
            // falls back to a FLOAT element, the same as `ARRAY[…]` literals: NUMERIC[] is
            // not a supported element type yet, so this preserves the earlier behavior.
            let elem = nusadb_core::engine::ArrayElem::from_column_type(typed.ty)
                .or_else(|| {
                    matches!(typed.ty, ColumnType::Numeric { .. })
                        .then(|| {
                            nusadb_core::engine::ArrayElem::from_column_type(ColumnType::Float)
                        })
                        .flatten()
                })
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "array_agg() does not support an argument of type {:?}",
                        typed.ty
                    ))
                })?;
            Ok((Some(typed), ColumnType::Array(elem)))
        },
        // JSONB_AGG(expr) — collect every value (NULLs become JSON null) into a JSON array. The
        // argument may be any type (the executor serializes each element to JSON), so no element-type
        // restriction applies; result is JSON.
        (ast::AggregateFunc::JsonAgg, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            Ok((Some(typed), ColumnType::Json))
        },
        // BOOL_AND(expr) / BOOL_OR(expr) — boolean argument; result BOOL. A bare NULL argument types
        // as BOOL via the hint, so the empty/NULL-only group still yields a BOOL NULL.
        (ast::AggregateFunc::BoolAnd | ast::AggregateFunc::BoolOr, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, Some(ColumnType::Bool))?;
            if typed.ty != ColumnType::Bool {
                return Err(Error::TypeMismatch {
                    context: format!("{func:?} requires a boolean argument"),
                    expected: ColumnType::Bool,
                    found: typed.ty,
                });
            }
            Ok((Some(typed), ColumnType::Bool))
        },
        // BIT_AND(expr) / BIT_OR(expr) / BIT_XOR(expr) — bitwise fold; INT argument, INT result.
        (
            ast::AggregateFunc::BitAnd | ast::AggregateFunc::BitOr | ast::AggregateFunc::BitXor,
            Some(arg),
        ) => {
            let typed = analyze_expr(arg, scope, catalog, Some(ColumnType::Int))?;
            if typed.ty != ColumnType::Int {
                return Err(Error::TypeMismatch {
                    context: format!("{func:?} requires an integer argument"),
                    expected: ColumnType::Int,
                    found: typed.ty,
                });
            }
            Ok((Some(typed), ColumnType::Int))
        },
        // STRING_AGG(expr, sep) — concatenate TEXT inputs; TEXT argument, TEXT result. The separator
        // is resolved in the aggregate analysis arm (it is a constant, not a per-row value).
        (ast::AggregateFunc::StringAgg, Some(arg)) => {
            let typed = analyze_expr(arg, scope, catalog, Some(ColumnType::Text))?;
            if typed.ty != ColumnType::Text {
                return Err(Error::TypeMismatch {
                    context: "STRING_AGG requires a text argument".to_owned(),
                    expected: ColumnType::Text,
                    found: typed.ty,
                });
            }
            Ok((Some(typed), ColumnType::Text))
        },
        // STDDEV / VARIANCE (sample), STDDEV_POP / VAR_POP (population), and the first argument of the
        // two-argument CORR / COVAR_* / REGR_* — numeric argument, FLOAT result, except REGR_COUNT
        // which is an INT pair count. (The second argument of the two-argument forms is resolved by
        // the caller in `analyze_expr`.)
        (
            ast::AggregateFunc::Stddev
            | ast::AggregateFunc::Variance
            | ast::AggregateFunc::StddevPop
            | ast::AggregateFunc::VarPop
            | ast::AggregateFunc::Corr
            | ast::AggregateFunc::CovarPop
            | ast::AggregateFunc::CovarSamp
            | ast::AggregateFunc::RegrCount
            | ast::AggregateFunc::RegrAvgx
            | ast::AggregateFunc::RegrAvgy
            | ast::AggregateFunc::RegrSxx
            | ast::AggregateFunc::RegrSyy
            | ast::AggregateFunc::RegrSxy
            | ast::AggregateFunc::RegrSlope
            | ast::AggregateFunc::RegrIntercept
            | ast::AggregateFunc::RegrR2,
            Some(arg),
        ) => {
            let typed = analyze_expr(arg, scope, catalog, None)?;
            if !is_numeric(typed.ty) {
                return Err(Error::TypeMismatch {
                    context: format!("{func:?} requires a numeric argument"),
                    expected: ColumnType::Float,
                    found: typed.ty,
                });
            }
            // REGR_COUNT yields the INT pair count; every other two-argument statistic is FLOAT.
            let result_ty = if func == ast::AggregateFunc::RegrCount {
                ColumnType::Int
            } else {
                ColumnType::Float
            };
            Ok((Some(typed), result_ty))
        },
        // Ordered-set aggregates are resolved by `analyze_within_group`, never here.
        (
            ast::AggregateFunc::PercentileCont
            | ast::AggregateFunc::PercentileDisc
            | ast::AggregateFunc::Mode,
            _,
        ) => Err(Error::Unsupported(
            "PERCENTILE_CONT / PERCENTILE_DISC / MODE require WITHIN GROUP syntax".to_owned(),
        )),
        // GROUPING is a synthetic aggregate created by `rebase_onto_aggregation` from a scalar
        // `GROUPING(...)` call, never type-checked through this path; reject defensively.
        (ast::AggregateFunc::Grouping, _) => Err(Error::Unsupported(
            "GROUPING is resolved against the query's grouping sets, not as an aggregate"
                .to_owned(),
        )),
        // SUM/AVG/MIN/MAX without an argument is a parser error in practice,
        // but defensively reject here.
        (_, None) => Err(Error::Unsupported(format!("{func:?} requires an argument"))),
    }
}

pub(super) fn projection_name(expr: &ast::Expr) -> String {
    match expr {
        ast::Expr::Column(name) => name.clone(),
        ast::Expr::QualifiedColumn { column, .. } => column.clone(),
        _ => "?column?".to_owned(),
    }
}
