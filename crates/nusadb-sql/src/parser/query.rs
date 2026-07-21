//! Query envelope: WITH/CTE, set-operation tree, top-level SELECT + row locks.
//!
//! Split verbatim out of `parser/mod.rs` (ADR 007); see that module for the
//! anti-corruption-layer contract. Cross-submodule converters resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === SELECT ===============================================================

/// Top-level entry for a `Query` statement: a plain `SELECT` or a set-operation tree.
///
/// A plain `SELECT` keeps the flat [`ast::Statement::Select`] shape (zero ripple to the
/// analyzer/planner). Anything with `UNION`/`INTERSECT`/`EXCEPT` at the root becomes a
/// [`ast::Statement::SetOperation`], whose `ORDER BY`/`LIMIT` bind to the combined result.
/// A `WITH` clause is extracted first and attached to the resulting `Select.with`.
pub(super) fn convert_query(query: sql::Query) -> Result<ast::Statement, Error> {
    // Extract and convert any WITH clause before handing off to sub-converters so that
    // reject_query_envelope (called by convert_select) does not see `with` and reject it.
    let with = convert_with(query.with)?;
    let query = sql::Query {
        with: None,
        ..query
    };

    if matches!(&*query.body, sql::SetExpr::Select(_)) {
        let mut select = convert_select(query)?;
        select.with = with;
        return Ok(ast::Statement::Select(select));
    }
    // A top-level `VALUES (row), ...` query returns the rows directly (standard SQL).
    if matches!(&*query.body, sql::SetExpr::Values(_)) {
        if !with.is_empty() {
            return unsupported("WITH before a top-level VALUES is not yet supported");
        }
        return convert_top_level_values(query);
    }
    if !with.is_empty() {
        return unsupported("WITH ... UNION/INTERSECT/EXCEPT is not yet supported");
    }
    reject_query_envelope(&query, false)?;
    let body = convert_set_expr(*query.body)?;
    let order_by = convert_order_by(query.order_by)?;
    // The envelope guard already rejected an OFFSET / `LIMIT ... BY` here, so only LIMIT remains.
    let (raw_limit, _, _) = split_limit_clause(query.limit_clause)?;
    let limit = convert_limit(raw_limit)?;
    Ok(ast::Statement::SetOperation(ast::SetOperation {
        body,
        order_by,
        limit,
    }))
}

/// Desugar a top-level `VALUES (row), ...` query into `SELECT * FROM (VALUES ...) AS values`, so the
/// row set flows through the VALUES-derived-table relation. Its `ORDER BY`/`LIMIT` bind to the
/// combined result; the output columns are named `column1`, `column2`, … (as for a VALUES relation).
fn convert_top_level_values(query: sql::Query) -> Result<ast::Statement, Error> {
    // Reject OFFSET / FETCH / locks (no silent drop, G20); ORDER BY + LIMIT are consumed below.
    reject_query_envelope(&query, false)?;
    let order_by = convert_order_by(query.order_by)?;
    let (raw_limit, _, _) = split_limit_clause(query.limit_clause)?;
    let limit = convert_limit(raw_limit)?;
    let sql::SetExpr::Values(values) = *query.body else {
        return unsupported("internal: expected a VALUES body");
    };
    let rows: Vec<Vec<ast::Expr>> = values
        .rows
        .into_iter()
        .map(|row| {
            row.content
                .into_iter()
                .map(convert_expr)
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        return unsupported("VALUES with no rows");
    }
    let base = ast::TableRef {
        // Derived `(VALUES ...)` table: the schema field is unused.
        schema: None,
        name: "values".to_owned(),
        alias: Some("values".to_owned()),
        subquery: None,
        values: Some(rows),
        set_op: None,
        lateral: false,
        column_aliases: Vec::new(),
        with_ordinality: false,
    };
    Ok(ast::Statement::Select(ast::Select {
        with: Vec::new(),
        distinct: None,
        projection: vec![ast::SelectItem::Wildcard],
        from: Some(ast::FromClause {
            base,
            joins: Vec::new(),
        }),
        filter: None,
        group_by: ast::GroupBy::Expressions(Vec::new()),
        having: None,
        order_by,
        limit,
        limit_with_ties: false,
        offset: None,
        lock: None,
    }))
}

/// Convert an optional `WITH` clause into a list of [`ast::Cte`]s.
///
/// `RECURSIVE` CTEs are supported; `AS [NOT] MATERIALIZED` never reaches here (the
/// mandatory `GenericDialect` tokenizerdoes not parse it; tracked with the other
/// upstream-grammar gaps) and would be ignored as a hint if it ever did; a `FROM` alias is
/// rejected with [`Error::Unsupported`].
pub(super) fn convert_with(with: Option<sql::With>) -> Result<Vec<ast::Cte>, Error> {
    let Some(with) = with else {
        return Ok(Vec::new());
    };
    let recursive = with.recursive;
    with.cte_tables
        .into_iter()
        .map(|cte| convert_cte(cte, recursive))
        .collect()
}

/// Convert one sqlparser `Cte` into the internal [`ast::Cte`] (non-recursive recursive).
pub(super) fn convert_cte(cte: sql::Cte, recursive: bool) -> Result<ast::Cte, Error> {
    // `AS [NOT] MATERIALIZED` is an inlining hint; NusaDB evaluates every CTE once (the
    // materialized semantic), so both spellings are accepted and the hint carries no effect.
    if cte.from.is_some() {
        return unsupported("CTE with a FROM alias");
    }
    let name = fold_ident(&cte.alias.name);
    let columns = fold_alias_columns(&cte.alias.columns)?;
    let body = convert_cte_body(*cte.query, recursive)?;
    Ok(ast::Cte {
        name,
        columns,
        body,
        recursive,
    })
}

/// Convert the body of a CTE entry into an [`ast::CteBody`].
///
/// Read body (`recursive = false`): a plain `SELECT` (→ `SelectBody::Select`) or any set operation
/// (UNION/INTERSECT/EXCEPT → `SelectBody::SetOp`), inlined like a derived table. Recursive
/// (`recursive = true`): `anchor UNION [ALL] recursive_arm`; any other operator/quantifier is
/// rejected. Data-modifying body: `INSERT`/`UPDATE … RETURNING` (→ `CteBody::Modifying`); the
/// `RETURNING` clause is required, since its rows form the CTE's relation. A nested `WITH` and
/// `ORDER BY`/`LIMIT`/`OFFSET`/`FETCH` on the body envelope are attached to the inlined SELECT for a
/// plain non-recursive `SELECT` body (the analyzer resolves the nested CTEs with the enclosing ones
/// in scope; the planner applies the Sort + Limit) and rejected on a set-op / recursive /
/// data-modifying body (no place to carry them).
pub(super) fn convert_cte_body(
    mut query: sql::Query,
    recursive: bool,
) -> Result<ast::CteBody, Error> {
    // A nested `WITH` and body-envelope pagination (`ORDER BY`/`LIMIT`/`OFFSET`/`FETCH`) are supported
    // on a plain, non-recursive `SELECT` CTE body: the nested CTEs and the pagination attach to the
    // inlined SELECT (the analyzer resolves the nested `WITH` with the enclosing CTEs in scope; the
    // planner applies the Sort + Limit). Capture them here, then let `reject_query_envelope` reject any
    // remaining envelope clause (locks / `LIMIT ... BY` / FOR / SETTINGS / FORMAT) rather than
    // silently dropping it. `query.with` is taken out first so the guard does not reject it.
    let raw_with = query.with.take();
    let raw_order = query.order_by.take();
    let (raw_limit, raw_offset, raw_limit_by) = split_limit_clause(query.limit_clause.take())?;
    if !raw_limit_by.is_empty() {
        return unsupported("LIMIT ... BY");
    }
    let raw_fetch = query.fetch.take();
    let has_pagination = raw_order.as_ref().is_some_and(order_by_is_effective)
        || raw_limit.is_some()
        || raw_offset.is_some()
        || raw_fetch.is_some();
    reject_query_envelope(&query, false)?;
    match *query.body {
        // A plain SELECT body: attach the nested `WITH` + pagination to the inlined SELECT.
        sql::SetExpr::Select(s) if !recursive => {
            let mut select = convert_bare_select(*s)?;
            select.with = convert_with(raw_with)?;
            select.order_by = convert_order_by(raw_order)?;
            let (limit, with_ties) = convert_limit_and_fetch(raw_limit, raw_fetch)?;
            select.limit = limit;
            select.limit_with_ties = with_ties;
            select.offset = convert_offset(raw_offset)?;
            Ok(ast::CteBody::Query(Box::new(ast::SelectBody::Select(
                Box::new(select),
            ))))
        },
        // A set-operation / recursive / data-modifying body has no place to carry a nested `WITH` or
        // the pagination, so reject them here rather than dropping them.
        _ if has_pagination || raw_with.is_some() => unsupported(
            "a nested WITH / ORDER BY / LIMIT / OFFSET / FETCH on a CTE body is only supported on a \
             plain (non-recursive, non-set-operation) SELECT",
        ),
        sql::SetExpr::Select(s) => {
            let select = convert_bare_select(*s)?;
            Ok(ast::CteBody::Query(Box::new(ast::SelectBody::Select(
                Box::new(select),
            ))))
        },
        sql::SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            if !recursive {
                // A non-recursive CTE body may be any set operation — inlined like a set-op
                // derived table. Delegate to the general set-expression converter, which handles
                // UNION/INTERSECT/EXCEPT trees.
                let body = convert_set_expr(sql::SetExpr::SetOperation {
                    op,
                    set_quantifier,
                    left,
                    right,
                })?;
                return Ok(ast::CteBody::Query(Box::new(body)));
            }
            if !matches!(op, sql::SetOperator::Union) {
                return unsupported(
                    "WITH RECURSIVE CTE body must use UNION [ALL] between anchor and recursive arm",
                );
            }
            // `UNION ALL` keeps every produced row; `UNION` (the default quantifier, or explicit
            // `DISTINCT`) keeps only rows not already in the result — both are valid recursion.
            let all = matches!(set_quantifier, sql::SetQuantifier::All);
            let anchor = convert_set_expr(*left)?;
            let arm = convert_set_expr(*right)?;
            Ok(ast::CteBody::Query(Box::new(ast::SelectBody::SetOp {
                op: ast::SetOp::Union,
                all,
                left: Box::new(anchor),
                right: Box::new(arm),
            })))
        },
        // A data-modifying CTE: `INSERT`/`UPDATE`/`DELETE … RETURNING` (+
        // The archive-then-remove pattern). sqlparser models all three as
        // `SetExpr` variants. The statement runs once and its RETURNING rows are the relation,
        // so RETURNING is required.
        sql::SetExpr::Insert(stmt) | sql::SetExpr::Update(stmt) | sql::SetExpr::Delete(stmt) => {
            if recursive {
                return unsupported(
                    "a WITH RECURSIVE CTE body cannot be a data-modifying statement",
                );
            }
            let converted = super::convert_statement(stmt)?;
            let has_returning = match &converted {
                ast::Statement::Insert(i) => !i.returning.is_empty(),
                ast::Statement::Update(u) => !u.returning.is_empty(),
                ast::Statement::Delete(d) => !d.returning.is_empty(),
                _ => false,
            };
            if !has_returning {
                return unsupported(
                    "a data-modifying CTE must have a RETURNING clause (its returned rows form the \
                     CTE's relation)",
                );
            }
            Ok(ast::CteBody::Modifying(Box::new(converted)))
        },
        _ => unsupported(
            "CTE body must be a SELECT, a set operation, or a data-modifying INSERT/UPDATE … RETURNING",
        ),
    }
}

/// Reject `Query`-envelope clauses outside the Stage-4 surface.
///
/// Exhaustively destructures the `Query` (by reference) so a future sqlparser field cannot be
/// silently dropped (G5, G20): `body`/`order_by`/`limit` are consumed by the caller; every other
/// clause is rejected here rather than parsed-and-ignored — historically `offset`/`fetch`/`locks`/
/// `limit_by`/`for_clause` were discarded, so `... OFFSET 5` returned from offset 0 and
/// `... FOR UPDATE` took no lock.
pub(super) fn reject_query_envelope(query: &sql::Query, allow_locks: bool) -> Result<(), Error> {
    let sql::Query {
        with,
        body: _,
        order_by: _,
        limit_clause,
        fetch,
        locks,
        for_clause,
        settings,
        format_clause,
        pipe_operators,
    } = query;
    if with.is_some() {
        return unsupported("WITH (common table expressions)");
    }
    // `convert_select` / the CTE + set-op operand paths take the limit clause out before calling
    // this function, so on those paths it is always `None`. A set-operation envelope keeps
    // its `LIMIT` (consumed by the caller) but cannot carry OFFSET / `LIMIT ... BY` — reject them
    // here (no silent drop, G20; prevents `UNION ... OFFSET n` from being silently ignored).
    match &limit_clause {
        None => {},
        Some(sql::LimitClause::LimitOffset {
            limit: _,
            offset,
            limit_by,
        }) => {
            if !limit_by.is_empty() {
                return unsupported("LIMIT ... BY");
            }
            if offset.is_some() {
                return unsupported(
                    "OFFSET in a set-operation envelope is not yet supported (applies OFFSET to \
                     the whole UNION/INTERSECT/EXCEPT result — use it on a plain SELECT instead)",
                );
            }
        },
        Some(sql::LimitClause::OffsetCommaLimit { .. }) => {
            return unsupported("comma-form `LIMIT <offset>, <limit>`");
        },
    }
    if !pipe_operators.is_empty() {
        return unsupported("pipe operator `|>`");
    }
    if fetch.is_some() {
        return unsupported(
            "FETCH FIRST/NEXT in a set-operation envelope is not yet supported (applies to the \
             whole UNION/INTERSECT/EXCEPT result — use LIMIT on a plain SELECT instead)",
        );
    }
    // `allow_locks` callers (plain SELECT) consume `locks` themselves via `convert_locks`; the
    // set-operation paths cannot model row locking, so they reject it here (no silent drop, G20).
    if !allow_locks && !locks.is_empty() {
        return unsupported("row locking (FOR UPDATE / FOR SHARE)");
    }
    if for_clause.is_some() {
        return unsupported("FOR XML / FOR JSON");
    }
    if settings.is_some() {
        return unsupported("SETTINGS clause");
    }
    if format_clause.is_some() {
        return unsupported("FORMAT clause");
    }
    Ok(())
}

/// Convert one node of a set-operation tree. A leaf `SELECT` becomes an [`ast::Select`]
/// with no `ORDER BY`/`LIMIT` (those bind to the whole set operation); a binary operator recurses
/// into both operands. A parenthesized operand `(SELECT ... [ORDER BY ...] [LIMIT ...])` carries
/// its pagination on the leaf `ast::Select` — the standard per-branch form, e.g. the top-K
/// traversal seed `(SELECT ... ORDER BY dist LIMIT 5) UNION ALL ...`; a parenthesized
/// operand that is itself a set operation has no place to carry them and stays rejected.
pub(super) fn convert_set_expr(body: sql::SetExpr) -> Result<ast::SelectBody, Error> {
    match body {
        sql::SetExpr::Select(select) => Ok(ast::SelectBody::Select(Box::new(convert_bare_select(
            *select,
        )?))),
        sql::SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            let op = match op {
                sql::SetOperator::Union => ast::SetOp::Union,
                sql::SetOperator::Intersect => ast::SetOp::Intersect,
                // `MINUS` is an alternate spelling of `EXCEPT` (newly parsed by sqlparser 0.62).
                sql::SetOperator::Except | sql::SetOperator::Minus => ast::SetOp::Except,
            };
            let all = match set_quantifier {
                sql::SetQuantifier::All => true,
                sql::SetQuantifier::Distinct | sql::SetQuantifier::None => false,
                sql::SetQuantifier::ByName
                | sql::SetQuantifier::AllByName
                | sql::SetQuantifier::DistinctByName => {
                    return unsupported("set operation with BY NAME");
                },
            };
            Ok(ast::SelectBody::SetOp {
                op,
                all,
                left: Box::new(convert_set_expr(*left)?),
                right: Box::new(convert_set_expr(*right)?),
            })
        },
        sql::SetExpr::Query(mut query) => {
            // Capture the operand's own pagination before the envelope guard sees it, mirroring
            // `convert_cte_body`: on a plain SELECT operand it attaches to the leaf `ast::Select`
            // (the planner applies the Sort + Limit before the set operation combines branches).
            let raw_order = query.order_by.take();
            let (raw_limit, raw_offset, raw_limit_by) =
                split_limit_clause(query.limit_clause.take())?;
            if !raw_limit_by.is_empty() {
                return unsupported("LIMIT ... BY");
            }
            let raw_fetch = query.fetch.take();
            let has_pagination = raw_order.as_ref().is_some_and(order_by_is_effective)
                || raw_limit.is_some()
                || raw_offset.is_some()
                || raw_fetch.is_some();
            reject_query_envelope(&query, false)?;
            match (*query.body, has_pagination) {
                (sql::SetExpr::Select(s), _) => {
                    let mut select = convert_bare_select(*s)?;
                    select.order_by = convert_order_by(raw_order)?;
                    let (limit, with_ties) = convert_limit_and_fetch(raw_limit, raw_fetch)?;
                    select.limit = limit;
                    select.limit_with_ties = with_ties;
                    select.offset = convert_offset(raw_offset)?;
                    Ok(ast::SelectBody::Select(Box::new(select)))
                },
                (body, false) => convert_set_expr(body),
                (_, true) => unsupported(
                    "ORDER BY / LIMIT / OFFSET / FETCH on a parenthesized operand that is itself \
                     a set operation (attach them to a plain SELECT operand instead)",
                ),
            }
        },
        sql::SetExpr::Values(_) => unsupported("VALUES as a set-operation operand"),
        sql::SetExpr::Insert(_)
        | sql::SetExpr::Update(_)
        | sql::SetExpr::Delete(_)
        | sql::SetExpr::Merge(_)
        | sql::SetExpr::Table(_) => unsupported("set-operation operand that is not a SELECT"),
    }
}

pub(super) fn convert_select(mut query: sql::Query) -> Result<ast::Select, Error> {
    // Extract LIMIT / OFFSET and FETCH FIRST before reject_query_envelope sees them, so
    // that the exhaustive-destructure guard does not reject them while they are legitimately
    // consumed.
    let (raw_limit, raw_offset, raw_limit_by) = split_limit_clause(query.limit_clause.take())?;
    if !raw_limit_by.is_empty() {
        return unsupported("LIMIT ... BY");
    }
    let raw_fetch = query.fetch.take();
    reject_query_envelope(&query, true)?;
    let select = match *query.body {
        sql::SetExpr::Select(select) => *select,
        _ => return unsupported("set operations (UNION / INTERSECT / EXCEPT)"),
    };
    let mut converted = convert_bare_select(select)?;
    converted.order_by = convert_order_by(query.order_by)?;
    let (limit, with_ties) = convert_limit_and_fetch(raw_limit, raw_fetch)?;
    converted.limit = limit;
    converted.limit_with_ties = with_ties;
    converted.offset = convert_offset(raw_offset)?;
    converted.lock = convert_locks(query.locks)?;
    Ok(converted)
}

/// Convert `LIMIT n` / `FETCH FIRST n ROWS [ONLY | WITH TIES]` into a `(row cap, with-ties)` pair.
///
/// `LIMIT` and `FETCH` are mutually exclusive. `WITH TIES` is carried through (the analyzer checks it
/// has an `ORDER BY` and a supported shape); `PERCENT` is rejected.
fn convert_limit_and_fetch(
    limit: Option<sql::Expr>,
    fetch: Option<sql::Fetch>,
) -> Result<(Option<u64>, bool), Error> {
    if limit.is_some() && fetch.is_some() {
        return unsupported("cannot combine LIMIT and FETCH FIRST in one query");
    }
    if let Some(fetch) = fetch {
        if fetch.percent {
            return unsupported("FETCH FIRST n PERCENT ROWS is not supported");
        }
        // FETCH FIRST with no quantity means "fetch all" — same as no row cap.
        return Ok((convert_limit(fetch.quantity)?, fetch.with_ties));
    }
    Ok((convert_limit(limit)?, false))
}

/// Convert `OFFSET n [ROW[S]]` into a row offset. Only constant integer offsets are accepted.
fn convert_offset(offset: Option<sql::Offset>) -> Result<Option<u64>, Error> {
    let Some(offset) = offset else {
        return Ok(None);
    };
    match offset.value {
        sql::Expr::Value(sql::ValueWithSpan {
            value: sql::Value::Number(n, _),
            ..
        }) => n.parse::<u64>().map_or_else(
            |_| unsupported(&format!("OFFSET value `{n}`")),
            |value| Ok(Some(value)),
        ),
        _ => unsupported("non-constant OFFSET"),
    }
}

/// Convert the `Query`-level `FOR UPDATE`/`FOR SHARE` lock clauses. At most one clause is
/// modelled; a SQL standard `OF <table>` and `NOWAIT`/`SKIP LOCKED` modifier are carried through.
pub(super) fn convert_locks(locks: Vec<sql::LockClause>) -> Result<Option<ast::RowLock>, Error> {
    let mut iter = locks.into_iter();
    let Some(lock) = iter.next() else {
        return Ok(None);
    };
    if iter.next().is_some() {
        return unsupported("multiple FOR UPDATE / FOR SHARE clauses");
    }
    let sql::LockClause {
        lock_type,
        of,
        nonblock,
    } = lock;
    let strength = match lock_type {
        sql::LockType::Update => ast::LockStrength::Update,
        sql::LockType::Share => ast::LockStrength::Share,
    };
    let of = of.as_ref().map(object_name).transpose()?;
    let wait = match nonblock {
        None => ast::LockWait::Default,
        Some(sql::NonBlock::Nowait) => ast::LockWait::NoWait,
        Some(sql::NonBlock::SkipLocked) => ast::LockWait::SkipLocked,
    };
    Ok(Some(ast::RowLock { strength, of, wait }))
}
