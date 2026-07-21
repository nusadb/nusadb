//! SELECT body: projection, FROM/joins, GROUP BY (incl. grouping sets), ORDER BY, LIMIT.
//!
//! Split verbatim out of `parser/mod.rs` (ADR 007); see that module for the
//! anti-corruption-layer contract. Cross-submodule converters resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

/// Convert a single `SELECT` body (no `ORDER BY`/`LIMIT`, which live on the enclosing `Query`).
pub(super) fn convert_bare_select(select: sql::Select) -> Result<ast::Select, Error> {
    reject_unsupported_select(&select)?;
    // Capture this SELECT's `WINDOW w AS (...)` definitions so an `OVER w` reference in its
    // projection / DISTINCT / HAVING resolves to them; the guard restores the enclosing scope on
    // return so a subquery's WINDOW clause does not leak out.
    let named_windows = build_named_windows(&select.named_window)?;
    let _windows = super::expr::enter_named_windows(named_windows);
    let distinct = match select.distinct {
        // `SELECT ALL` is the explicit spelling of the default (keep every row) — same as no
        // quantifier. sqlparser 0.62 records it; 0.51 dropped it at parse time.
        None | Some(sql::Distinct::All) => None,
        Some(sql::Distinct::Distinct) => Some(ast::Distinct::All),
        Some(sql::Distinct::On(exprs)) => {
            if exprs.is_empty() {
                return unsupported("SELECT DISTINCT ON with no expressions");
            }
            let exprs = exprs
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Some(ast::Distinct::On(exprs))
        },
    };
    let from = convert_from(&select.from)?;
    let projection = select
        .projection
        .into_iter()
        .map(convert_select_item)
        .collect::<Result<Vec<_>, _>>()?;
    let filter = select.selection.map(convert_expr).transpose()?;
    let group_by = convert_group_by(select.group_by)?;
    let having = select.having.map(convert_expr).transpose()?;
    Ok(ast::Select {
        with: Vec::new(), // populated by convert_query for top-level WITH clauses
        distinct,
        projection,
        from,
        filter,
        group_by,
        having,
        order_by: Vec::new(),
        limit: None,
        limit_with_ties: false, // populated by convert_select for FETCH ... WITH TIES
        offset: None,           // populated by convert_select for OFFSET n ROWS
        lock: None,
    })
}

/// Convert a `GROUP BY` clause into an [`ast::GroupBy`].
///
/// - `ROLLUP(...)` / `CUBE(...)` / `GROUPING SETS(...)` appear in sqlparser as a single
///   `Expr::Rollup` / `Expr::Cube` / `Expr::GroupingSets` inside the expression list.
///   When the list is a single such expression it is lifted to the corresponding variant.
/// - `WITH ROLLUP` / `WITH CUBE` modifiers and `GROUP BY ALL` are rejected.
pub(super) fn convert_group_by(group_by: sql::GroupByExpr) -> Result<ast::GroupBy, Error> {
    match group_by {
        sql::GroupByExpr::All(_) => unsupported("GROUP BY ALL"),
        sql::GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return unsupported("GROUP BY WITH ROLLUP / CUBE / TOTALS modifier");
            }
            // Check if the expression list is a single ROLLUP / CUBE / GROUPING SETS wrapper.
            if let [single] = exprs.as_slice() {
                match single {
                    sql::Expr::Rollup(sets) => {
                        let sets = convert_grouping_sets(sets.clone())?;
                        return Ok(ast::GroupBy::Rollup(sets));
                    },
                    sql::Expr::Cube(sets) => {
                        let sets = convert_grouping_sets(sets.clone())?;
                        return Ok(ast::GroupBy::Cube(sets));
                    },
                    sql::Expr::GroupingSets(sets) => {
                        let sets = convert_grouping_sets(sets.clone())?;
                        return Ok(ast::GroupBy::GroupingSets(sets));
                    },
                    _ => {},
                }
            }
            // Plain expression list — convert each expr individually.
            let keys = exprs
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::GroupBy::Expressions(keys))
        },
    }
}

/// Convert the grouping-set matrix from sqlparser into `Vec<Vec<ast::Expr>>`.
pub(super) fn convert_grouping_sets(
    sets: Vec<Vec<sql::Expr>>,
) -> Result<Vec<Vec<ast::Expr>>, Error> {
    sets.into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(convert_expr)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

/// Collect a SELECT's `WINDOW name AS (spec)` definitions into a name → spec map for `OVER name`
/// resolution. A definition that *extends* another named window (`WINDOW w2 AS (w1 ...)`) is
/// rejected — only a direct spec is supported.
fn build_named_windows(
    defs: &[sql::NamedWindowDefinition],
) -> Result<std::collections::HashMap<String, sql::WindowSpec>, Error> {
    let mut map = std::collections::HashMap::with_capacity(defs.len());
    for def in defs {
        let spec = match &def.1 {
            sql::NamedWindowExpr::WindowSpec(spec) => spec.clone(),
            sql::NamedWindowExpr::NamedWindow(_) => {
                return unsupported("a WINDOW that references another named window");
            },
        };
        map.insert(fold_ident(&def.0), spec);
    }
    Ok(map)
}

/// Reject `SELECT` clauses outside the Stage 4 surface.
///
/// Exhaustively destructures the `Select` so a future sqlparser field cannot be silently ignored:
/// the clauses `convert_bare_select` actually consumes are bound with `_`; every other
/// clause is rejected explicitly rather than parsed-and-dropped.
pub(super) fn reject_unsupported_select(select: &sql::Select) -> Result<(), Error> {
    let sql::Select {
        select_token: _,
        distinct: _,
        projection: _,
        from: _,
        selection: _,
        group_by: _,
        having: _,
        window_before_qualify: _,
        // Whether `TOP` came before `DISTINCT` — cosmetic; `TOP` itself is rejected below.
        top_before_distinct: _,
        optimizer_hints,
        select_modifiers,
        exclude,
        flavor,
        top,
        into,
        lateral_views,
        prewhere,
        cluster_by,
        distribute_by,
        sort_by,
        named_window: _,
        qualify,
        value_table_mode,
        connect_by,
    } = select;
    if !optimizer_hints.is_empty() {
        return unsupported("optimizer hints");
    }
    if select_modifiers.is_some() {
        return unsupported("SELECT modifiers (HIGH_PRIORITY / STRAIGHT_JOIN / ...)");
    }
    if exclude.is_some() {
        return unsupported("SELECT ... EXCLUDE");
    }
    if !matches!(flavor, sql::SelectFlavor::Standard) {
        return unsupported("FROM-first SELECT");
    }
    if top.is_some() {
        return unsupported("SELECT TOP");
    }
    if into.is_some() {
        return unsupported("SELECT ... INTO");
    }
    if !lateral_views.is_empty() {
        return unsupported("LATERAL VIEW");
    }
    if prewhere.is_some() {
        return unsupported("PREWHERE");
    }
    if !cluster_by.is_empty() {
        return unsupported("CLUSTER BY");
    }
    if !distribute_by.is_empty() {
        return unsupported("DISTRIBUTE BY");
    }
    if !sort_by.is_empty() {
        return unsupported("SORT BY");
    }
    if qualify.is_some() {
        return unsupported("QUALIFY");
    }
    if value_table_mode.is_some() {
        return unsupported("SELECT AS VALUE / AS STRUCT");
    }
    if !connect_by.is_empty() {
        return unsupported("CONNECT BY");
    }
    Ok(())
}

pub(super) fn convert_from(from: &[sql::TableWithJoins]) -> Result<Option<ast::FromClause>, Error> {
    let Some((first, rest)) = from.split_first() else {
        return Ok(None);
    };
    let base = convert_from_item(&first.relation)?;
    let mut joins = first
        .joins
        .iter()
        .map(convert_join)
        .collect::<Result<Vec<_>, _>>()?;
    // Comma-separated FROM items are implicit CROSS JOINs: `FROM a, b` ≡ `FROM a CROSS JOIN b`,
    // with any `WHERE` filtering the product. Each later item must be a simple table / derived table
    // without its own JOINs — a comma mixed with an explicit JOIN (`FROM a, b JOIN c ON ...`) has subtle
    // precedence against outer joins (the JOIN binds tighter than the comma), so it stays rejected
    // rather than silently flattened to a different shape.
    for twj in rest {
        if !twj.joins.is_empty() {
            return unsupported("comma-separated FROM item mixed with an explicit JOIN");
        }
        joins.push(ast::Join {
            table: convert_from_item(&twj.relation)?,
            kind: ast::JoinKind::Cross,
            condition: ast::JoinCondition::None,
        });
    }
    Ok(Some(ast::FromClause { base, joins }))
}

/// Convert a *named* table reference only (`t [AS x]`). Used where a derived table is not allowed —
/// the MERGE target. A subquery/function FROM item is rejected here; a place that also accepts a
/// derived source (a `SELECT` FROM clause, `UPDATE ... FROM`, `DELETE ... USING`, the MERGE source)
/// uses [`convert_from_item`] instead.
pub(super) fn convert_table_ref(factor: &sql::TableFactor) -> Result<ast::TableRef, Error> {
    match factor {
        sql::TableFactor::Table {
            name,
            alias,
            args: None,
            ..
        } => {
            let alias = convert_table_alias(alias.as_ref())?;
            let (schema, name) = table_ref_name(name)?;
            Ok(ast::TableRef {
                schema,
                name,
                alias,
                subquery: None,
                values: None,
                set_op: None,
                lateral: false,
                column_aliases: Vec::new(),
                with_ordinality: false,
            })
        },
        _ => unsupported("FROM item that is not a plain table (subquery, function, ...)"),
    }
}

/// Fold an optional table alias, rejecting an alias column list (which NusaDB does not model).
fn convert_table_alias(alias: Option<&sql::TableAlias>) -> Result<Option<String>, Error> {
    match alias {
        Some(alias) => {
            if !alias.columns.is_empty() {
                return unsupported("table alias with a column list");
            }
            Ok(Some(fold_ident(&alias.name)))
        },
        None => Ok(None),
    }
}

/// Convert a `FROM`/join item: a named table (via [`convert_table_ref`]), a *derived table* —
/// `[LATERAL] (SELECT ...) AS x` — or a table function `func(args) [AS x]`, e.g.
/// `FROM generate_series(1, 10)`.
pub(super) fn convert_from_item(factor: &sql::TableFactor) -> Result<ast::TableRef, Error> {
    // `FROM func(args) [AS x[(cols)]]` — a (set-returning) function in FROM. Desugar to a derived
    // table `(SELECT func(args)) AS x` so it reuses the projection-`ProjectSet` + derived-table path.
    if let sql::TableFactor::Table {
        name,
        alias,
        args: Some(table_args),
        with_ordinality,
        ..
    } = factor
    {
        return desugar_table_function(name, table_args, alias.as_ref(), *with_ordinality);
    }
    // `FROM UNNEST(array) [AS x[(col)]]` — sqlparser models UNNEST as its own table factor, not the
    // generic function form, so route it through the dedicated desugaring.
    if let sql::TableFactor::UNNEST {
        alias,
        array_exprs,
        with_offset,
        with_offset_alias,
        with_ordinality,
    } = factor
    {
        return desugar_unnest(
            array_exprs,
            alias.as_ref(),
            *with_offset,
            with_offset_alias.as_ref(),
            *with_ordinality,
        );
    }
    if let sql::TableFactor::Derived {
        lateral,
        subquery,
        alias,
        sample,
    } = factor
    {
        if sample.is_some() {
            return unsupported("TABLESAMPLE on a derived table");
        }
        let Some(alias) = alias.as_ref() else {
            return unsupported("a subquery in FROM must have an alias");
        };
        // A derived table may rename its output columns: `(SELECT ...) AS x(a, b)`.
        let name = fold_ident(&alias.name);
        let column_aliases = fold_alias_columns(&alias.columns)?;
        // `(VALUES (row), ...) AS x` is a values derived table — carry the inline rows rather than a
        // SELECT body (which cannot represent a multi-row VALUES list).
        if let sql::SetExpr::Values(values) = &*subquery.body {
            let rows: Vec<Vec<ast::Expr>> = values
                .rows
                .iter()
                .map(|row| {
                    row.content
                        .iter()
                        .cloned()
                        .map(convert_expr)
                        .collect::<Result<_, _>>()
                })
                .collect::<Result<_, _>>()?;
            if rows.is_empty() {
                return unsupported("VALUES with no rows");
            }
            return Ok(ast::TableRef {
                // Derived table: the schema field is unused (it has a subquery/values body, no name).
                schema: None,
                name: name.clone(),
                alias: Some(name),
                subquery: None,
                values: Some(rows),
                set_op: None,
                lateral: *lateral,
                column_aliases,
                with_ordinality: false,
            });
        }
        // `(SELECT ... UNION/INTERSECT/EXCEPT ...) AS x` is a set-operation derived table — carry the
        // set-op body (a single SELECT body cannot represent it). `convert_query` lowers the inner
        // query, yielding a `SetOperation` for a set-op body and a `Select` for a plain one.
        if matches!(&*subquery.body, sql::SetExpr::SetOperation { .. }) {
            let ast::Statement::SetOperation(so) = convert_query((**subquery).clone())? else {
                return unsupported("a derived-table set operation could not be lowered");
            };
            return Ok(ast::TableRef {
                // Derived table: the schema field is unused (it has a subquery/values body, no name).
                schema: None,
                name: name.clone(),
                alias: Some(name),
                subquery: None,
                values: None,
                set_op: Some(Box::new(so)),
                lateral: *lateral,
                column_aliases,
                with_ordinality: false,
            });
        }
        let body = convert_select((**subquery).clone())?;
        return Ok(ast::TableRef {
            // Derived `(SELECT ...)` table: the schema field is unused.
            schema: None,
            name: name.clone(),
            alias: Some(name),
            subquery: Some(Box::new(body)),
            values: None,
            set_op: None,
            lateral: *lateral,
            column_aliases,
            with_ordinality: false,
        });
    }
    convert_table_ref(factor)
}

/// Desugar a `FROM func(args) [AS x[(cols)]]` table function into a derived table whose subquery is
/// `SELECT func(args)`. A set-returning function (e.g. `generate_series`) becomes a
/// `ProjectSet` producing N rows; a scalar function yields one row — both reuse the derived-table
/// path. The relation alias defaults to the function name; with no explicit column list the single
/// output column takes the relation's name (so `FROM gs(1,3) AS g` exposes column `g`).
fn desugar_table_function(
    name: &sql::ObjectName,
    table_args: &sql::TableFunctionArgs,
    alias: Option<&sql::TableAlias>,
    with_ordinality: bool,
) -> Result<ast::TableRef, Error> {
    // Rebuild the call as an ordinary function expression so the normal conversion maps a known
    // set-returning function to its `SetReturning` node (and an unknown name is rejected there).
    let call = sql::Function {
        name: name.clone(),
        uses_odbc_syntax: false,
        parameters: sql::FunctionArguments::None,
        args: sql::FunctionArguments::List(sql::FunctionArgumentList {
            duplicate_treatment: None,
            args: table_args.args.clone(),
            clauses: Vec::new(),
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: Vec::new(),
    };
    let func_expr = super::expr::convert_function_call(call)?;
    let table_alias = match alias {
        Some(a) => fold_ident(&a.name),
        None => object_name(name)?,
    };
    let column_aliases: Vec<String> = alias
        .map(|a| fold_alias_columns(&a.columns))
        .transpose()?
        .unwrap_or_default();
    Ok(srf_derived_table(
        func_expr,
        table_alias,
        column_aliases,
        with_ordinality,
    ))
}

/// Desugar `FROM UNNEST(array) [AS x[(col)]] [WITH ORDINALITY]` into a derived table
/// `(SELECT unnest(array)) AS x` — one row per element, reusing the same derived-table and
/// projection-`ProjectSet` path as `generate_series`. `WITH ORDINALITY` appends a 1-based row number.
/// Only the single-array form is supported: multiple arrays (the column-zip form) and `WITH OFFSET`
/// are rejected rather than silently dropped.
fn desugar_unnest(
    array_exprs: &[sql::Expr],
    alias: Option<&sql::TableAlias>,
    with_offset: bool,
    with_offset_alias: Option<&sql::Ident>,
    with_ordinality: bool,
) -> Result<ast::TableRef, Error> {
    if with_offset || with_offset_alias.is_some() {
        return unsupported("UNNEST ... WITH OFFSET");
    }
    let [array_expr] = array_exprs else {
        return unsupported("UNNEST of multiple arrays");
    };
    let func_expr = ast::Expr::SetReturning {
        func: ast::SetReturningFunc::Unnest,
        args: vec![convert_expr(array_expr.clone())?],
    };
    let table_alias = alias.map_or_else(|| "unnest".to_owned(), |a| fold_ident(&a.name));
    let column_aliases: Vec<String> = alias
        .map(|a| fold_alias_columns(&a.columns))
        .transpose()?
        .unwrap_or_default();
    Ok(srf_derived_table(
        func_expr,
        table_alias,
        column_aliases,
        with_ordinality,
    ))
}

/// Wrap a set-returning function expression as a one-column derived table `(SELECT expr) AS x`, the
/// shared tail of the `FROM func(args)` and `FROM UNNEST(array)` desugarings. With no explicit
/// `(cols)` list the lone output column takes the relation's name; otherwise `column_aliases` renames
/// it positionally.
fn srf_derived_table(
    func_expr: ast::Expr,
    table_alias: String,
    column_aliases: Vec<String>,
    with_ordinality: bool,
) -> ast::TableRef {
    let proj_alias = column_aliases.is_empty().then(|| table_alias.clone());
    let select = ast::Select {
        with: Vec::new(),
        distinct: None,
        projection: vec![ast::SelectItem::Expr {
            expr: func_expr,
            alias: proj_alias,
        }],
        from: None,
        filter: None,
        group_by: ast::GroupBy::Expressions(Vec::new()),
        having: None,
        order_by: Vec::new(),
        limit: None,
        limit_with_ties: false,
        offset: None,
        lock: None,
    };
    ast::TableRef {
        // Derived table (table-function rewrite): the schema field is unused.
        schema: None,
        name: table_alias.clone(),
        alias: Some(table_alias),
        subquery: Some(Box::new(select)),
        values: None,
        set_op: None,
        lateral: false,
        column_aliases,
        with_ordinality,
    }
}

pub(super) fn convert_join(join: &sql::Join) -> Result<ast::Join, Error> {
    let table = convert_from_item(&join.relation)?;
    // Cross join has no constraint (sqlparser 0.62 carries a constraint slot; a CROSS JOIN with an
    // actual constraint is out of surface).
    if matches!(
        join.join_operator,
        sql::JoinOperator::CrossJoin(sql::JoinConstraint::None)
    ) {
        return Ok(ast::Join {
            table,
            kind: ast::JoinKind::Cross,
            condition: ast::JoinCondition::None,
        });
    }
    let (kind, constraint) = match &join.join_operator {
        // A bare `JOIN` (0.62 models it separately) is the standard synonym for `INNER JOIN`.
        sql::JoinOperator::Join(c) | sql::JoinOperator::Inner(c) => (ast::JoinKind::Inner, c),
        sql::JoinOperator::Left(c) | sql::JoinOperator::LeftOuter(c) => (ast::JoinKind::Left, c),
        sql::JoinOperator::Right(c) | sql::JoinOperator::RightOuter(c) => (ast::JoinKind::Right, c),
        sql::JoinOperator::FullOuter(c) => (ast::JoinKind::Full, c),
        _ => return unsupported("unsupported JOIN kind (only INNER/LEFT/RIGHT/FULL/CROSS)"),
    };
    let condition = match constraint {
        sql::JoinConstraint::On(expr) => ast::JoinCondition::On(convert_expr(expr.clone())?),
        sql::JoinConstraint::Using(cols) => ast::JoinCondition::Using(
            cols.iter()
                .map(column_ident_name)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        sql::JoinConstraint::Natural => ast::JoinCondition::Natural,
        sql::JoinConstraint::None => ast::JoinCondition::None,
    };
    Ok(ast::Join {
        table,
        kind,
        condition,
    })
}

pub(super) fn convert_select_item(item: sql::SelectItem) -> Result<ast::SelectItem, Error> {
    match item {
        sql::SelectItem::Wildcard(opts) => {
            reject_wildcard_options(&opts)?;
            Ok(ast::SelectItem::Wildcard)
        },
        sql::SelectItem::UnnamedExpr(expr) => Ok(ast::SelectItem::Expr {
            expr: convert_expr(expr)?,
            alias: None,
        }),
        sql::SelectItem::ExprWithAlias { expr, alias } => Ok(ast::SelectItem::Expr {
            expr: convert_expr(expr)?,
            alias: Some(fold_ident(&alias)),
        }),
        // `expr AS (a, b)` — a multi-alias projection (a multi-alias form sqlparser 0.62 models).
        sql::SelectItem::ExprWithAliases { .. } => {
            unsupported("projection with multiple aliases `expr AS (a, b)`")
        },
        // `table.*`. The qualifier is the final name component (single namespace),
        // folded like any identifier; the wildcard decorations NusaDB does not model are rejected
        // (not silently dropped), as they are for plain `*`.
        sql::SelectItem::QualifiedWildcard(kind, opts) => {
            reject_wildcard_options(&opts)?;
            match kind {
                sql::SelectItemQualifiedWildcardKind::ObjectName(name) => {
                    Ok(ast::SelectItem::QualifiedWildcard(object_name(&name)?))
                },
                // `expr.*` (a struct/composite expansion) is out of surface.
                sql::SelectItemQualifiedWildcardKind::Expr(_) => {
                    unsupported("`<expression>.*` wildcard expansion")
                },
            }
        },
    }
}

pub(super) fn convert_order_by(
    order_by: Option<sql::OrderBy>,
) -> Result<Vec<ast::OrderByItem>, Error> {
    let Some(order_by) = order_by else {
        return Ok(Vec::new());
    };
    // `ORDER BY ALL` (a dialect extension newly parsed by sqlparser 0.62) is out of surface.
    let exprs = match order_by.kind {
        sql::OrderByKind::Expressions(exprs) => exprs,
        sql::OrderByKind::All(_) => return unsupported("ORDER BY ALL"),
    };
    exprs
        .into_iter()
        .map(|obe| {
            if obe.with_fill.is_some() {
                return unsupported("ORDER BY ... WITH FILL");
            }
            let nulls = match obe.options.nulls_first {
                None => ast::NullOrdering::Default,
                Some(true) => ast::NullOrdering::First,
                Some(false) => ast::NullOrdering::Last,
            };
            Ok(ast::OrderByItem {
                expr: convert_expr(obe.expr)?,
                ascending: obe.options.asc.unwrap_or(true),
                nulls,
            })
        })
        .collect()
}

pub(super) fn convert_limit(limit: Option<sql::Expr>) -> Result<Option<u64>, Error> {
    let Some(expr) = limit else {
        return Ok(None);
    };
    match expr {
        sql::Expr::Value(sql::ValueWithSpan {
            value: sql::Value::Number(n, _),
            ..
        }) => n.parse::<u64>().map_or_else(
            |_| unsupported(&format!("LIMIT value `{n}`")),
            |value| Ok(Some(value)),
        ),
        _ => unsupported("non-constant LIMIT"),
    }
}
