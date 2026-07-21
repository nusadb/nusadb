//! DML statement converters: INSERT (+ON CONFLICT/RETURNING), UPDATE, DELETE, MERGE.
//!
//! Split verbatim out of `parser/mod.rs` (ADR 007); see that module for the
//! anti-corruption-layer contract. Cross-submodule converters resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === INSERT ===============================================================

pub(super) fn convert_insert(insert: sql::Insert) -> Result<ast::Insert, Error> {
    // sqlparser 0.62 models the target as a `TableObject`; only a plain table name is in surface.
    let sql::TableObject::TableName(ref target) = insert.table else {
        return unsupported("INSERT into a table function / query target");
    };
    let (schema, table) = table_ref_name(target)?;
    let columns = insert
        .columns
        .iter()
        .map(column_ident_name)
        .collect::<Result<Vec<_>, _>>()?;
    // `INSERT INTO t DEFAULT VALUES`: sqlparser models it as no source + no column list.
    let Some(query) = insert.source else {
        return Ok(ast::Insert {
            schema,
            table,
            columns,
            source: ast::InsertSource::DefaultValues,
            on_conflict: convert_on_insert(insert.on)?,
            returning: convert_returning(insert.returning)?,
        });
    };
    let query = *query;
    let source = match *query.body {
        sql::SetExpr::Values(values) => {
            let rows: Vec<Vec<Option<ast::Expr>>> = values
                .rows
                .into_iter()
                .map(|row| {
                    row.content
                        .into_iter()
                        .map(convert_values_cell)
                        .collect::<Result<_, _>>()
                })
                .collect::<Result<_, _>>()?;
            if rows.is_empty() {
                return unsupported("INSERT with no rows");
            }
            ast::InsertSource::Values(rows)
        },
        sql::SetExpr::Select(_) | sql::SetExpr::Query(_) => {
            ast::InsertSource::Select(Box::new(convert_select(query)?))
        },
        _ => {
            return unsupported(
                "INSERT with an unsupported source (set operation, table function, etc.)",
            );
        },
    };
    let on_conflict = convert_on_insert(insert.on)?;
    let returning = convert_returning(insert.returning)?;
    Ok(ast::Insert {
        schema,
        table,
        columns,
        source,
        on_conflict,
        returning,
    })
}

/// Lower one cell of a `VALUES` row. A bare `DEFAULT` keyword (which sqlparser
/// tokenizes as an unquoted `DEFAULT` identifier) lowers to `None`, signalling
/// the executor to apply the target column's default/serial/NULL fill for that
/// position; every other cell lowers to `Some(expr)`.
fn convert_values_cell(cell: sql::Expr) -> Result<Option<ast::Expr>, Error> {
    if let sql::Expr::Identifier(ident) = &cell
        && ident.quote_style.is_none()
        && ident.value.eq_ignore_ascii_case("DEFAULT")
    {
        return Ok(None);
    }
    Ok(Some(convert_expr(cell)?))
}

/// Lower the optional `ON CONFLICT` clause. Only `DO NOTHING` is modelled
/// today (`DO UPDATE` lands later); the `ON DUPLICATE KEY UPDATE` form is
/// rejected.
pub(super) fn convert_on_insert(
    on: Option<sql::OnInsert>,
) -> Result<Option<ast::OnConflict>, Error> {
    let conflict = match on {
        None => return Ok(None),
        Some(sql::OnInsert::OnConflict(c)) => c,
        Some(sql::OnInsert::DuplicateKeyUpdate(_)) => {
            return unsupported("INSERT ... ON DUPLICATE KEY UPDATE");
        },
        Some(_) => return unsupported("INSERT with an unsupported ON clause"),
    };
    let target = match conflict.conflict_target {
        None => None,
        Some(sql::ConflictTarget::Columns(cols)) => Some(ast::ConflictTarget::Columns(
            cols.iter().map(fold_ident).collect(),
        )),
        Some(sql::ConflictTarget::OnConstraint(name)) => {
            Some(ast::ConflictTarget::Constraint(object_name(&name)?))
        },
    };
    let action = match conflict.action {
        sql::OnConflictAction::DoNothing => ast::ConflictAction::DoNothing,
        sql::OnConflictAction::DoUpdate(du) => convert_do_update(du)?,
    };
    Ok(Some(ast::OnConflict { target, action }))
}

/// Convert `DO UPDATE SET ... [WHERE ...]`. Assignment targets and value expressions reuse the
/// `UPDATE` conversion, so `EXCLUDED.col` arrives as an [`ast::Expr::QualifiedColumn`] with table
/// `excluded` (the analyzer interprets that pseudo-table when the upsert path lands).
pub(super) fn convert_do_update(du: sql::DoUpdate) -> Result<ast::ConflictAction, Error> {
    if du.assignments.is_empty() {
        return unsupported("ON CONFLICT DO UPDATE with no assignments");
    }
    let mut assignments = Vec::with_capacity(du.assignments.len());
    for assignment in du.assignments {
        assignments.push(convert_assignment(assignment)?);
    }
    let filter = du.selection.map(convert_expr).transpose()?;
    Ok(ast::ConflictAction::DoUpdate {
        assignments,
        filter,
    })
}

/// Convert an optional `RETURNING` projection list into [`ast::SelectItem`]s.
///
/// `None` (clause absent) → empty `Vec`. Each item reuses the same conversion as
/// a `SELECT` projection, so `*`, `col`, `col AS alias`, and `t.*` are all accepted.
pub(super) fn convert_returning(
    items: Option<Vec<sql::SelectItem>>,
) -> Result<Vec<ast::SelectItem>, Error> {
    items.map_or_else(
        || Ok(Vec::new()),
        |list| list.into_iter().map(convert_select_item).collect(),
    )
}

// === UPDATE ===============================================================

pub(super) fn convert_update(
    table: &sql::TableWithJoins,
    assignments: Vec<sql::Assignment>,
    from: Option<sql::UpdateTableFromKind>,
    selection: Option<sql::Expr>,
    returning: Option<Vec<sql::SelectItem>>,
) -> Result<ast::Update, Error> {
    if !table.joins.is_empty() {
        return unsupported("JOIN in UPDATE target");
    }
    if assignments.is_empty() {
        return unsupported("UPDATE with no assignments");
    }
    let target = convert_table_ref(&table.relation)?;
    let (schema, table_name, alias) = (target.schema, target.name, target.alias);
    // `UPDATE ... FROM` source: a single `TableWithJoins` (base + optional joins on that base).
    // 0.62 models the FROM as a list (and records whether it came before or after `SET` — a
    // dialect variation; both spellings carry the same source).
    let from = match from {
        None => None,
        Some(
            sql::UpdateTableFromKind::BeforeSet(twjs) | sql::UpdateTableFromKind::AfterSet(twjs),
        ) => {
            let [twj] = twjs.as_slice() else {
                return unsupported("UPDATE ... FROM with comma-separated items");
            };
            convert_from(std::slice::from_ref(twj))?
        },
    };
    let mut converted = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        converted.push(convert_assignment(assignment)?);
    }
    let filter = selection.map(convert_expr).transpose()?;
    let returning = convert_returning(returning)?;
    Ok(ast::Update {
        schema,
        table: table_name,
        alias,
        assignments: converted,
        from,
        filter,
        returning,
    })
}

pub(super) fn convert_assignment(assignment: sql::Assignment) -> Result<ast::Assignment, Error> {
    let column = match assignment.target {
        sql::AssignmentTarget::ColumnName(name) => object_name(&name)?,
        sql::AssignmentTarget::Tuple(_) => {
            return unsupported("multi-column assignment (`SET (a, b) = ...`)");
        },
    };
    Ok(ast::Assignment {
        column,
        value: convert_expr(assignment.value)?,
    })
}

// === DELETE ===============================================================

pub(super) fn convert_delete(delete: sql::Delete) -> Result<ast::Delete, Error> {
    if !delete.order_by.is_empty() || delete.limit.is_some() {
        return unsupported("DELETE with ORDER BY / LIMIT");
    }
    if !delete.tables.is_empty() {
        return unsupported("multi-table DELETE");
    }
    let tables = match delete.from {
        sql::FromTable::WithFromKeyword(tables) | sql::FromTable::WithoutKeyword(tables) => tables,
    };
    let (schema, table) = match tables.as_slice() {
        [twj] => {
            if !twj.joins.is_empty() {
                return unsupported("JOIN in DELETE");
            }
            let target = convert_table_ref(&twj.relation)?;
            (target.schema, target.name)
        },
        _ => return unsupported("DELETE must target exactly one table"),
    };
    // `DELETE ... USING` source: we model a single base table (with optional explicit JOINs) via
    // `FromClause`, mirroring the single-base `UPDATE ... FROM` surface. A comma-separated
    // USING list is rejected here — unlike a `SELECT`'s comma `FROM` (an implicit CROSS JOIN), the
    // DELETE-USING executor is single-base, so accepting a comma here would change its semantics.
    let using = match delete.using {
        None => None,
        Some(tables) => {
            if tables.len() > 1 {
                return unsupported(
                    "comma-separated tables in DELETE ... USING (use an explicit JOIN)",
                );
            }
            convert_from(&tables)?
        },
    };
    let filter = delete.selection.map(convert_expr).transpose()?;
    let returning = convert_returning(delete.returning)?;
    Ok(ast::Delete {
        schema,
        table,
        using,
        filter,
        returning,
    })
}

// === MERGE ================================================================

pub(super) fn convert_merge(
    into: bool,
    table: &sql::TableFactor,
    source: &sql::TableFactor,
    on: sql::Expr,
    clauses: Vec<sql::MergeClause>,
) -> Result<ast::Merge, Error> {
    if !into {
        return unsupported("MERGE without INTO");
    }
    let target = convert_table_ref(table)?;
    // The USING source may be a plain table or a derived relation (`VALUES` / subquery / set
    // operation), like `UPDATE ... FROM` / `DELETE ... USING`; the analyzer resolves both. The target
    // stays a plain named table.
    let source = convert_from_item(source)?;
    let on = convert_expr(on)?;
    if clauses.is_empty() {
        return unsupported("MERGE with no WHEN clauses");
    }
    let whens = clauses
        .into_iter()
        .map(convert_merge_clause)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ast::Merge {
        target,
        source,
        on,
        whens,
    })
}

pub(super) fn convert_merge_clause(clause: sql::MergeClause) -> Result<ast::MergeWhen, Error> {
    let sql::MergeClause {
        when_token: _,
        clause_kind,
        predicate,
        action,
    } = clause;
    let pred = predicate.map(convert_expr).transpose()?;
    match clause_kind {
        sql::MergeClauseKind::Matched => {
            let action = match action {
                sql::MergeAction::Update(update) => {
                    // The dialect-specific per-action WHERE / DELETE WHERE clauses (newly modelled
                    // by sqlparser 0.62) are out of surface.
                    if update.update_predicate.is_some() || update.delete_predicate.is_some() {
                        return unsupported("MERGE ... UPDATE with a WHERE / DELETE WHERE clause");
                    }
                    if update.assignments.is_empty() {
                        return unsupported(
                            "MERGE ... WHEN MATCHED THEN UPDATE with no assignments",
                        );
                    }
                    let assignments = update
                        .assignments
                        .into_iter()
                        .map(convert_assignment)
                        .collect::<Result<Vec<_>, _>>()?;
                    ast::MatchedAction::Update { assignments }
                },
                sql::MergeAction::Delete { .. } => ast::MatchedAction::Delete,
                sql::MergeAction::Insert(_) => {
                    return unsupported(
                        "MERGE ... WHEN MATCHED THEN INSERT (INSERT requires NOT MATCHED)",
                    );
                },
            };
            Ok(ast::MergeWhen::Matched { pred, action })
        },
        sql::MergeClauseKind::NotMatched => {
            let insert = match action {
                sql::MergeAction::Insert(insert) => convert_merge_insert(insert)?,
                sql::MergeAction::Update(_) | sql::MergeAction::Delete { .. } => {
                    return unsupported(
                        "MERGE ... WHEN NOT MATCHED THEN UPDATE/DELETE (NOT MATCHED requires INSERT)",
                    );
                },
            };
            Ok(ast::MergeWhen::NotMatched { pred, insert })
        },
        sql::MergeClauseKind::NotMatchedByTarget | sql::MergeClauseKind::NotMatchedBySource => {
            unsupported("MERGE ... WHEN NOT MATCHED BY TARGET/SOURCE")
        },
    }
}

pub(super) fn convert_merge_insert(
    insert: sql::MergeInsertExpr,
) -> Result<ast::MergeInsert, Error> {
    let sql::MergeInsertExpr {
        insert_token: _,
        columns,
        kind_token: _,
        kind,
        insert_predicate,
    } = insert;
    // A dialect extension (0.62): a per-INSERT `WHERE` predicate on the merge action.
    if insert_predicate.is_some() {
        return unsupported("MERGE ... INSERT with a WHERE predicate");
    }
    let columns = columns
        .iter()
        .map(column_ident_name)
        .collect::<Result<Vec<_>, _>>()?;
    let values = match kind {
        sql::MergeInsertKind::Values(values) => {
            let mut rows = values.rows;
            match rows.len() {
                1 => rows
                    .remove(0)
                    .content
                    .into_iter()
                    .map(convert_expr)
                    .collect::<Result<Vec<_>, _>>()?,
                _ => return unsupported("MERGE ... INSERT VALUES with multiple rows"),
            }
        },
        sql::MergeInsertKind::Row => {
            return unsupported("MERGE ... INSERT ROW (use an explicit VALUES list)");
        },
    };
    Ok(ast::MergeInsert { columns, values })
}
