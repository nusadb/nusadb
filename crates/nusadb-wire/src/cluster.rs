//! Multiple databases — the **physical** model.
//!
//! Each database is a *physically isolated* storage engine over its own directory: separate
//! WAL/MVCC/recovery, so a heavy or corrupt database cannot affect another (the closest match to a
//! multi-database server). A connection resolves its database by name to exactly one engine and only
//! ever touches that engine, so isolation is automatic — there is no cross-database query and a
//! three-part `db.schema.table` name stays rejected, just like the reference servers.
//!
//! This module defines the [`DatabaseCluster`] port the wire server uses to resolve / create / drop
//! / list databases. The production manager (which opens one storage engine per database) lives in
//! `nusadb-server`, outside the wire crate, so the wire layer needs no storage-engine implementation
//! dependency. [`SingleDatabase`] is the trivial one-engine adapter for the embedded `serve(engine)`
//! entry point and tests.

use std::sync::Arc;

use nusadb_core::StorageEngine;

/// The default database a connection lands in when its startup message names none. Matches the
/// hard-coded `current_database()` value and the drivers' default `database` argument.
pub const DEFAULT_DATABASE: &str = "nusadb";

/// A user-facing failure of a cluster operation. The wire layer maps each to a `FATAL`/`ERROR`
/// response with the appropriate SQLSTATE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    /// `CREATE DATABASE` named a database that already exists (and `IF NOT EXISTS` was not given).
    AlreadyExists(String),
    /// `DROP DATABASE` named a database that does not exist (and `IF EXISTS` was not given), or a
    /// connection requested a database that does not exist.
    NotFound(String),
    /// `DROP DATABASE` named the database the calling connection is currently using.
    InUse(String),
    /// The database name is not a valid identifier for a physical database (empty, too long, or
    /// containing characters that are unsafe in a directory name).
    InvalidName(String),
    /// The operation is not supported by this cluster (e.g. creating a database in single-database
    /// embedded mode).
    Unsupported(String),
    /// An I/O error opening, creating, or removing a database's on-disk storage.
    Io(String),
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyExists(n) => write!(f, "database \"{n}\" already exists"),
            Self::NotFound(n) => write!(f, "database \"{n}\" does not exist"),
            Self::InUse(n) => write!(f, "cannot drop the currently open database \"{n}\""),
            Self::InvalidName(n) => write!(f, "invalid database name \"{n}\""),
            Self::Unsupported(m) => write!(f, "{m}"),
            Self::Io(m) => write!(f, "database storage error: {m}"),
        }
    }
}

impl std::error::Error for ClusterError {}

impl ClusterError {
    /// The SQLSTATE the wire reports for this error. `3D000` is *invalid catalog name* (unknown
    /// database); `42P04` is *duplicate database*; `55006` is *object in use*; the rest fall back to
    /// the generic internal-error code.
    #[must_use]
    pub const fn sqlstate(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "3D000",
            Self::AlreadyExists(_) => "42P04",
            Self::InUse(_) => "55006",
            Self::InvalidName(_) => "42602",
            Self::Unsupported(_) | Self::Io(_) => "XX000",
        }
    }
}

/// A cluster of physically-isolated databases. Each database resolves to its own
/// [`StorageEngine`]; a connection only ever touches the one its startup message selected.
pub trait DatabaseCluster: Send + Sync {
    /// Resolve (lazy-open) `name` to its engine, or `Ok(None)` if no such database exists. The name
    /// has already been identifier-folded by the SQL layer; the cluster matches it as stored.
    fn open(&self, name: &str) -> Result<Option<Arc<dyn StorageEngine>>, ClusterError>;

    /// Create database `name`. Returns `Ok(true)` when a new database was created, `Ok(false)` when
    /// it already existed and `if_not_exists` was set (otherwise [`ClusterError::AlreadyExists`]).
    fn create(&self, name: &str, if_not_exists: bool) -> Result<bool, ClusterError>;

    /// Drop database `name`. `connected` is the database the calling connection is using — a
    /// connection cannot drop the database it is in ([`ClusterError::InUse`]). Returns `Ok(true)`
    /// when a database was dropped, `Ok(false)` when it was absent and `if_exists` was set.
    fn drop_database(
        &self,
        name: &str,
        if_exists: bool,
        connected: &str,
    ) -> Result<bool, ClusterError>;

    /// All database names, sorted.
    fn list(&self) -> Vec<String>;

    /// The default database a connection lands in when it requests none. Defaults to
    /// [`DEFAULT_DATABASE`].
    fn default_database(&self) -> String {
        DEFAULT_DATABASE.to_owned()
    }
}

/// The trivial single-database cluster: one engine, no `CREATE`/`DROP DATABASE`.
///
/// Used by the embedded [`serve`](crate::server::serve) entry point (which takes a bare engine) and
/// by tests. Resolving any database name yields the one engine, preserving the historical "database
/// parameter ignored" behavior for embedded callers.
pub struct SingleDatabase {
    engine: Arc<dyn StorageEngine>,
}

impl std::fmt::Debug for SingleDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleDatabase").finish_non_exhaustive()
    }
}

impl SingleDatabase {
    /// Wrap a single engine as a one-database cluster.
    #[must_use]
    pub fn new(engine: Arc<dyn StorageEngine>) -> Self {
        Self { engine }
    }
}

impl DatabaseCluster for SingleDatabase {
    fn open(&self, _name: &str) -> Result<Option<Arc<dyn StorageEngine>>, ClusterError> {
        // Single-database mode: the one engine answers for every requested name (the historical
        // behavior, where the startup database parameter was ignored).
        Ok(Some(Arc::clone(&self.engine)))
    }

    fn create(&self, _name: &str, _if_not_exists: bool) -> Result<bool, ClusterError> {
        Err(ClusterError::Unsupported(
            "CREATE DATABASE requires the server's database manager (single-database mode)"
                .to_owned(),
        ))
    }

    fn drop_database(
        &self,
        _name: &str,
        _if_exists: bool,
        _connected: &str,
    ) -> Result<bool, ClusterError> {
        Err(ClusterError::Unsupported(
            "DROP DATABASE requires the server's database manager (single-database mode)"
                .to_owned(),
        ))
    }

    fn list(&self) -> Vec<String> {
        vec![DEFAULT_DATABASE.to_owned()]
    }
}

/// Whether `name` is a valid physical-database name (DB1 security).
///
/// A non-empty, ≤63-char identifier of lowercase ASCII letters, digits, and underscores starting
/// with a letter or underscore. This is deliberately strict — the name becomes a directory under
/// `base/`, so anything that could escape the cluster root (`/`, `\`, `.`, `..`, NUL) or surprise the
/// filesystem is rejected.
#[must_use]
pub fn is_valid_database_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 63 {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap_or('\0');
    if !(first.is_ascii_lowercase() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_database_names() {
        assert!(is_valid_database_name("nusadb"));
        assert!(is_valid_database_name("shop"));
        assert!(is_valid_database_name("_private"));
        assert!(is_valid_database_name("tenant_42"));
        assert!(is_valid_database_name(&"a".repeat(63)));
    }

    #[test]
    fn invalid_database_names_are_rejected() {
        // Empty / too long.
        assert!(!is_valid_database_name(""));
        assert!(!is_valid_database_name(&"a".repeat(64)));
        // Path-traversal / filesystem-unsafe.
        assert!(!is_valid_database_name("."));
        assert!(!is_valid_database_name(".."));
        assert!(!is_valid_database_name("a/b"));
        assert!(!is_valid_database_name("a\\b"));
        assert!(!is_valid_database_name("a.b"));
        assert!(!is_valid_database_name("a\0b"));
        // Must start with a letter or underscore; case-folded lowercase only (SQL folds identifiers).
        assert!(!is_valid_database_name("1db"));
        assert!(!is_valid_database_name("Shop"));
        assert!(!is_valid_database_name("a b"));
    }
}
