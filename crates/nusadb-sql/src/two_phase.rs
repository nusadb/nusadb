//! Two-phase commit at the SQL layer: `PREPARE TRANSACTION`, `COMMIT PREPARED`,
//! `ROLLBACK PREPARED`, and the `nusadb_prepared_xacts` system view.
//!
//! The engine keeps the prepared transactions (see the treaty's
//! [`prepare_transaction`](nusadb_core::StorageEngine::prepare_transaction)); this module holds what
//! every session holder shares: the identifier check, who may finish a prepared transaction, and
//! the SQLSTATE each refusal reports. The embedded [`Session`](crate::Session) and the wire server
//! both call it, so the two answer alike.
//!
//! | Refusal | SQLSTATE |
//! | --- | --- |
//! | identifier longer than [`MAX_GID_BYTES`] | `22023` |
//! | identifier already in use | `42710` |
//! | no prepared transaction of that name | `42704` |
//! | `PREPARE TRANSACTION` with no transaction open | `25P01` |
//! | `COMMIT PREPARED` / `ROLLBACK PREPARED` inside a transaction block | `25001` |
//! | the transaction changed the schema, used a temporary table, `LISTEN`/`UNLISTEN`/`NOTIFY`, or declared a cursor `WITH HOLD` | `0A000` |
//! | finishing another role's prepared transaction without being a superuser | `42501` |
//! | prepared transactions disabled (a limit of 0) | `55000` |
//! | the limit on waiting prepared transactions reached | `53400` |

use nusadb_core::{IsolationLevel, PreparedTransaction, StorageEngine, TxnId};

use crate::analyzer::Catalog;
use crate::ast;
use crate::error::Error;
use crate::executor::ExecutionResult;

/// The longest global transaction identifier accepted, in bytes.
pub const MAX_GID_BYTES: usize = 200;

/// The system view listing the current database's prepared transactions.
pub const PREPARED_XACTS_VIEW: &str = "nusadb_prepared_xacts";

/// The view's columns, in order.
pub const PREPARED_XACTS_COLUMNS: [&str; 5] =
    ["transaction", "gid", "prepared", "owner", "database"];

/// Refuse an identifier longer than [`MAX_GID_BYTES`] (`22023`).
///
/// # Errors
/// [`Error::InvalidParameterValue`] for a too-long identifier.
pub fn check_gid(gid: &str) -> Result<(), Error> {
    if gid.len() > MAX_GID_BYTES {
        return Err(Error::InvalidParameterValue(format!(
            "transaction identifier \"{gid}\" is too long (the limit is {MAX_GID_BYTES} bytes)"
        )));
    }
    Ok(())
}

/// The SQL error for an engine refusal of a two-phase statement, by the kind the engine reports
/// (see [`nusadb_core::StorageEngine::prepare_transaction`]).
#[must_use]
pub fn engine_error(error: nusadb_core::Error) -> Error {
    if let nusadb_core::Error::Io(io) = &error {
        let sqlstate = match io.kind() {
            std::io::ErrorKind::AlreadyExists => Some("42710"),
            std::io::ErrorKind::NotFound => Some("42704"),
            std::io::ErrorKind::Unsupported => Some("0A000"),
            std::io::ErrorKind::QuotaExceeded => Some("53400"),
            std::io::ErrorKind::InvalidInput => Some("55000"),
            std::io::ErrorKind::InvalidData => Some("22023"),
            std::io::ErrorKind::PermissionDenied => Some("42501"),
            _ => None,
        };
        if let Some(sqlstate) = sqlstate {
            return Error::Coded {
                message: io.to_string(),
                sqlstate,
            };
        }
    }
    error.into()
}

/// A refusal of `PREPARE TRANSACTION` the session knows of and the engine does not: what the
/// transaction did outside the engine.
#[must_use]
pub fn session_refusal(what: &str) -> Error {
    Error::Unsupported(format!("cannot PREPARE a transaction that has {what}"))
}

/// `PREPARE TRANSACTION 'gid'` of the open transaction `txn`, recorded as prepared by `owner` in
/// `database`. On an error the transaction has been rolled back.
///
/// # Errors
/// The refusals in the module table, or an engine failure.
pub fn prepare(
    engine: &dyn StorageEngine,
    txn: TxnId,
    gid: &str,
    owner: &str,
    database: &str,
) -> Result<ExecutionResult, Error> {
    if let Err(e) = check_gid(gid) {
        let _ = crate::executor::deferred::rollback(engine, txn);
        return Err(e);
    }
    // The deferred constraint checks run first: a prepared transaction owes none, so a restart
    // while it waits for COMMIT PREPARED has nothing to replay. A violation rolls it back.
    crate::executor::deferred::prepare(engine, txn)?;
    crate::executor::deferred::end_outside(engine, Some(txn), || {
        engine.prepare_transaction(txn, gid, owner, database)
    })
    .map(|()| ExecutionResult::TransactionPrepared)
    .map_err(engine_error)
}

/// `COMMIT PREPARED 'gid'` (`commit`) or `ROLLBACK PREPARED 'gid'`, run by `user` outside any
/// transaction block. Only the role that prepared it, or a superuser, may finish it.
///
/// # Errors
/// `42704` for an unknown identifier, `42501` for another role's transaction, or an engine
/// failure.
pub fn finish(
    engine: &dyn StorageEngine,
    gid: &str,
    commit: bool,
    user: &str,
) -> Result<ExecutionResult, Error> {
    // A superuser finishes anyone's; any other role only its own, which the engine checks in the
    // same step that finishes it.
    let owner = (!is_superuser(engine, user)?).then_some(user);
    let finished = crate::executor::deferred::end_outside(engine, None, || {
        if commit {
            engine
                .commit_prepared(gid, owner)
                .map(|()| ExecutionResult::PreparedCommitted)
        } else {
            engine
                .rollback_prepared(gid, owner)
                .map(|()| ExecutionResult::PreparedRolledBack)
        }
    });
    finished.map_err(engine_error)
}

/// Whether `user` is a superuser, read in a short transaction of its own.
fn is_superuser(engine: &dyn StorageEngine, user: &str) -> Result<bool, Error> {
    let txn = engine.begin(IsolationLevel::ReadCommitted)?;
    let resolved = crate::rbac::ResolvedSession::resolve(engine, txn, user);
    let _ = crate::executor::deferred::rollback(engine, txn);
    Ok(resolved?.is_superuser())
}

/// The rows of [`PREPARED_XACTS_VIEW`] a session may read: every prepared transaction for a
/// superuser, its own for anyone else.
fn visible_rows(catalog: &dyn Catalog) -> Result<Vec<PreparedTransaction>, Error> {
    let mut rows = catalog.prepared_transactions()?;
    if !catalog.is_superuser() {
        let user = catalog.current_user();
        rows.retain(|p| p.owner == user);
    }
    Ok(rows)
}

/// A string literal for `text`.
fn quoted(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// The view as a derived table: a `VALUES` list of the visible rows behind a filter that drops the
/// typed placeholder row, so an empty list still has its column types.
fn view_select(rows: &[PreparedTransaction]) -> Result<ast::Select, Error> {
    let mut values = vec![
        "(CAST(NULL AS BIGINT), CAST(NULL AS TEXT), CAST(NULL AS TIMESTAMPTZ), CAST(NULL AS TEXT), \
         CAST(NULL AS TEXT))"
            .to_owned(),
    ];
    for p in rows {
        let micros = i64::try_from(p.prepared_at_unix_ms)
            .unwrap_or(i64::MAX / 1000)
            .saturating_mul(1000);
        values.push(format!(
            "(CAST({} AS BIGINT), {}, CAST({} AS TIMESTAMPTZ), {}, {})",
            p.txn.0,
            quoted(&p.gid),
            quoted(&crate::temporal::format_timestamptz(micros)),
            quoted(&p.owner),
            quoted(&p.database),
        ));
    }
    let sql = format!(
        "SELECT * FROM (VALUES {}) AS prepared_xacts(\"{}\") WHERE gid IS NOT NULL",
        values.join(", "),
        PREPARED_XACTS_COLUMNS.join("\", \""),
    );
    match crate::parse(&sql)? {
        ast::Statement::Select(select) => Ok(select),
        _ => Err(Error::Internal(
            "the prepared-transaction view did not build as a query".to_owned(),
        )),
    }
}

/// Replace every `FROM` reference to [`PREPARED_XACTS_VIEW`] in `stmt` (the query's own `FROM` and
/// joins, and the derived tables inside them) with the view's rows as they stand now, as
/// `catalog` lets its session see them. A statement that names no such table is left alone.
///
/// # Errors
/// A catalog failure listing the prepared transactions.
pub(crate) fn rewrite_view(stmt: &mut ast::Statement, catalog: &dyn Catalog) -> Result<(), Error> {
    let ast::Statement::Select(select) = stmt else {
        return Ok(());
    };
    let mut rows = None;
    rewrite_select(select, catalog, &mut rows)
}

fn rewrite_select(
    select: &mut ast::Select,
    catalog: &dyn Catalog,
    rows: &mut Option<Vec<PreparedTransaction>>,
) -> Result<(), Error> {
    let Some(from) = select.from.as_mut() else {
        return Ok(());
    };
    rewrite_table(&mut from.base, catalog, rows)?;
    for join in &mut from.joins {
        rewrite_table(&mut join.table, catalog, rows)?;
    }
    Ok(())
}

fn rewrite_table(
    table: &mut ast::TableRef,
    catalog: &dyn Catalog,
    rows: &mut Option<Vec<PreparedTransaction>>,
) -> Result<(), Error> {
    if let Some(sub) = table.subquery.as_mut() {
        return rewrite_select(sub, catalog, rows);
    }
    let named = table.subquery.is_none()
        && table.values.is_none()
        && table.set_op.is_none()
        && table.schema.is_none()
        && table.name.eq_ignore_ascii_case(PREPARED_XACTS_VIEW);
    if !named {
        return Ok(());
    }
    if rows.is_none() {
        *rows = Some(visible_rows(catalog)?);
    }
    let select = view_select(rows.as_deref().unwrap_or_default())?;
    let alias = table
        .alias
        .clone()
        .unwrap_or_else(|| PREPARED_XACTS_VIEW.to_owned());
    table.name.clone_from(&alias);
    table.alias = Some(alias);
    table.subquery = Some(Box::new(select));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_past_the_limit_are_refused_with_22023() {
        assert!(check_gid(&"g".repeat(MAX_GID_BYTES)).is_ok());
        let e = check_gid(&"g".repeat(MAX_GID_BYTES + 1)).unwrap_err();
        assert_eq!(e.sqlstate(), "22023");
    }

    #[test]
    fn engine_refusals_map_to_their_sqlstates() {
        let io = |kind| nusadb_core::Error::Io(std::io::Error::new(kind, "x"));
        for (kind, code) in [
            (std::io::ErrorKind::AlreadyExists, "42710"),
            (std::io::ErrorKind::NotFound, "42704"),
            (std::io::ErrorKind::Unsupported, "0A000"),
            (std::io::ErrorKind::QuotaExceeded, "53400"),
            (std::io::ErrorKind::InvalidInput, "55000"),
            (std::io::ErrorKind::InvalidData, "22023"),
            (std::io::ErrorKind::PermissionDenied, "42501"),
        ] {
            assert_eq!(engine_error(io(kind)).sqlstate(), code, "{kind:?}");
        }
        let conflict = nusadb_core::Error::SerializationConflict { txn: TxnId(1) };
        assert_eq!(engine_error(conflict).sqlstate(), "40001");
    }

    #[test]
    fn the_view_builds_for_no_rows_and_for_quoted_values() {
        assert!(view_select(&[]).is_ok());
        let row = PreparedTransaction {
            gid: "it's".to_owned(),
            txn: TxnId(7),
            owner: "o'neil".to_owned(),
            database: "db".to_owned(),
            prepared_at_unix_ms: 1_760_000_000_123,
        };
        assert!(view_select(&[row]).is_ok());
    }
}
