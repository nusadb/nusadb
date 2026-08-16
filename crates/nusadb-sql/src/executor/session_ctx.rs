//! Per-statement session context for the niladic session functions
//! (`CURRENT_USER`, `SESSION_USER`) and `current_setting(name)`.
//!
//! Like the statement [`clock`](super::clock), the context is pinned once per top-level statement
//! in a thread-local and read by the evaluator for each row, so every occurrence in one statement
//! observes the same user and the same settings snapshot. (`nusadb-sql` is single-threaded per
//! statement — the wire layer dispatches each query on its own blocking task — so a thread-local is
//! the natural carrier and costs nothing on the hot path.)
//!
//! The context is pinned by [`Session::run_within_txn`](super::Session) from the session's current
//! user and its `SET`/`RESET` variable store. A unit test that evaluates an expression without
//! going through the session sees the defaults below ([`DEFAULT_USER`], no settings).

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::Error;

/// The database user reported by `CURRENT_USER`/`SESSION_USER` when no session has pinned one — the
/// implicit user a bare [`execute`](super::execute) call or a direct evaluator unit test runs as.
/// This is the bootstrap superuser; the wire server pins each connection's authenticated user via
/// [`execute_in_txn_as`](super::execute_in_txn_as).
pub(super) const DEFAULT_USER: &str = crate::BOOTSTRAP_SUPERUSER;

/// The default database name when no session has pinned one, reported by `CURRENT_DATABASE()`. The
/// physical cluster bootstraps this database on a fresh data dir; matches the drivers' default
/// `database` argument and [`nusadb_wire::cluster::DEFAULT_DATABASE`].
const DEFAULT_DATABASE: &str = "nusadb";

/// The default schema name when no session has pinned one, reported by `CURRENT_SCHEMA()`.
const DEFAULT_SCHEMA: &str = "public";

thread_local! {
    /// The current statement's session context, pinned by [`set_session_context`]. `None` before
    /// any statement has pinned one on this thread (e.g. a direct evaluator unit test).
    static SESSION_CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}

/// A snapshot of the session state the niladic session functions read.
struct Context {
    /// The session user (`CURRENT_USER` / `SESSION_USER`).
    user: String,
    /// The session's `SET`/`RESET` variables, read by `current_setting(name)`.
    settings: HashMap<String, String>,
    /// The current database name, read by `CURRENT_DATABASE()`.
    database: String,
    /// The current schema name, read by `CURRENT_SCHEMA()`.
    schema: String,
    /// The session's temporary schema (`nusadb_temp_<id>`), if this is a real session; `None` for a
    /// bare execution path with no session identity. An unqualified name resolves here first, but it
    /// does *not* change [`schema`](Self::schema) (a non-temp `CREATE` still targets the search path).
    temp_schema: Option<String>,
}

/// Pin the session context for the statement about to run, replacing any prior snapshot. Called
/// once per top-level statement so all of its session functions agree.
pub(super) fn set_session_context(
    user: &str,
    settings: &HashMap<String, String>,
    database: &str,
    schema: &str,
    temp_schema: Option<&str>,
) {
    SESSION_CONTEXT.with(|cell| {
        *cell.borrow_mut() = Some(Context {
            user: user.to_owned(),
            settings: settings.clone(),
            database: database.to_owned(),
            schema: schema.to_owned(),
            temp_schema: temp_schema.map(ToOwned::to_owned),
        });
    });
}

/// Pin just the session user (no session variables, defaults for database/schema), for an execution
/// path that has no `SET` store — a bare statement run as a user.
pub(super) fn set_session_user(user: &str) {
    set_session_context(
        user,
        &HashMap::new(),
        DEFAULT_DATABASE,
        DEFAULT_SCHEMA,
        None,
    );
}

/// Pin the session user **together with** an explicit `SET` store (defaults for database/schema), for
/// an execution path that carries its session variables across statements — the wire server, which
/// keeps a per-connection GUC map so `current_setting(name)` reflects an earlier `SET name = …`
/// The settings persist only for the duration of the statement, like every other pin.
pub(super) fn set_session_user_with_settings(user: &str, settings: &HashMap<String, String>) {
    // Derive the current schema from `search_path` so `CURRENT_SCHEMA()` over the wire tracks
    // `SET search_path = …` instead of always reporting the default.
    let schema =
        crate::current_schema_for_search_path(settings.get("search_path").map(String::as_str));
    // The connection's database: the wire stamps it under a reserved key so `CURRENT_DATABASE()`
    // names the physical database this connection is bound to (not the hard-coded default).
    let database = settings
        .get(crate::CONNECTION_DATABASE_SETTING)
        .map_or(DEFAULT_DATABASE, String::as_str);
    // The connection's temporary schema, stamped by the wire under a reserved key, so an unqualified
    // name in a statement resolving through the pinned context sees this session's temp tables.
    let temp_schema = settings
        .get(crate::CONNECTION_TEMP_SCHEMA_SETTING)
        .map(String::as_str);
    set_session_context(user, settings, database, &schema, temp_schema);
}

/// The pinned session user, or [`DEFAULT_USER`] if nothing has been pinned on this thread.
pub(super) fn current_user() -> String {
    SESSION_CONTEXT.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or_else(|| DEFAULT_USER.to_owned(), |ctx| ctx.user.clone())
    })
}

/// The pinned current database, or [`DEFAULT_DATABASE`] if nothing has been pinned.
pub(super) fn current_database() -> String {
    SESSION_CONTEXT.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or_else(|| DEFAULT_DATABASE.to_owned(), |ctx| ctx.database.clone())
    })
}

/// The pinned current schema, or [`DEFAULT_SCHEMA`] if nothing has been pinned.
pub(super) fn current_schema() -> String {
    SESSION_CONTEXT.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or_else(|| DEFAULT_SCHEMA.to_owned(), |ctx| ctx.schema.clone())
    })
}

/// The pinned session's temporary schema (`nusadb_temp_<id>`), or `None` if this execution path has
/// no session identity (a bare `execute` call or a direct evaluator unit test). Read by the default
/// [`Catalog::temp_schema`](crate::Catalog::temp_schema) so an unqualified name resolves temp-first.
pub(super) fn current_temp_schema() -> Option<String> {
    SESSION_CONTEXT.with(|cell| {
        cell.borrow()
            .as_ref()
            .and_then(|ctx| ctx.temp_schema.clone())
    })
}

/// The value of session setting `name`, or `None` if it is unset (or nothing has been pinned on
/// this thread). An explicit `SET` wins; otherwise a well-known read-only GUC falls back to its
/// honest built-in default (kept consistent with `SHOW name`). `current_setting(name)` maps `None`
/// to SQL `NULL`.
pub(super) fn setting(name: &str) -> Option<String> {
    // Parameter names are case-insensitive and stored folded, so fold the lookup too — only when
    // it would change anything, to keep the common all-lower-case call allocation-free.
    let folded;
    let name = if name.bytes().any(|b| b.is_ascii_uppercase()) {
        folded = name.to_ascii_lowercase();
        folded.as_str()
    } else {
        name
    };
    SESSION_CONTEXT
        .with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|ctx| ctx.settings.get(name).cloned())
        })
        .or_else(|| builtin_guc_static_default(name).map(ToOwned::to_owned))
}

/// Built-in default for a well-known read-only/session GUC whose value is a fixed constant (so it can
/// be shared by both `SHOW name` and `current_setting(name)` without a session handle). `server_version`
/// reports NusaDB's own version — the wire protocol is the Nusa protocol, so the honest engine version
/// is the right answer. `transaction_isolation` is *not* here: it depends on the session's current
/// level, so `SHOW` resolves it separately. Returns `None` for an unknown variable.
pub(super) fn builtin_guc_static_default(name: &str) -> Option<&'static str> {
    Some(match name {
        "server_version" => env!("CARGO_PKG_VERSION"),
        "server_encoding" | "client_encoding" => "UTF8",
        "standard_conforming_strings" | "integer_datetimes" => "on",
        "datestyle" => "ISO, MDY",
        "timezone" => "UTC",
        _ => return None,
    })
}

/// Check that `name` is a configuration parameter a session may `SET`.
///
/// A parameter this engine does not read cannot be honoured, so accepting one would store a value
/// that never takes effect — the trap that makes `SET ef_search = 100` look like it tuned the beam
/// width whose real name is `hnsw_ef_search`. A parameter carrying a dot is an application's own
/// (`myapp.tenant`); that class prefix is exactly what separates it from a misspelled built-in, so
/// it needs no fixed list. Names are matched case-insensitively, since a quoted `"TimeZone"` keeps
/// its case but denotes the same parameter.
///
/// Both the embedded session and the wire server call this, so the two cannot drift apart.
///
/// # Errors
/// [`Error::Coded`] `55P02` for a parameter that describes the server rather than the session, and
/// `42704` for one this engine does not recognize.
pub fn check_settable_parameter(name: &str) -> Result<(), Error> {
    let folded = name.to_ascii_lowercase();
    if matches!(
        folded.as_str(),
        "server_version" | "server_encoding" | "integer_datetimes"
    ) {
        return Err(Error::Coded {
            message: format!("parameter \"{name}\" cannot be changed"),
            sqlstate: "55P02", // cant_change_runtime_param
        });
    }
    let recognized = folded.contains('.')
        || matches!(
            folded.as_str(),
            // Parameters the engine reads.
            "search_path"
                | "work_mem"
                | "statement_timeout"
                | "hnsw_ef_search"
                | crate::retry::MAX_AUTOCOMMIT_RETRIES
                // Reported parameters a session may still set.
                | "client_encoding"
                | "standard_conforming_strings"
                | "datestyle"
                | "timezone"
                | "transaction_isolation"
                | "default_transaction_isolation"
        );
    if recognized {
        return Ok(());
    }
    Err(Error::Coded {
        message: format!(
            "unrecognized configuration parameter \"{name}\" — an application's own parameter \
             must carry a class prefix, as in \"myapp.{folded}\""
        ),
        sqlstate: "42704", // undefined_object
    })
}
