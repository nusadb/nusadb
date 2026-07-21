//! Shared error type for cross-crate boundaries.
//!
//! Each downstream crate defines its own error enum and converts to/from this one via
//! `#[from]`. This keeps crate-level error types focused while allowing seamless `?`
//! propagation across boundaries.

use crate::ids::{PageId, TxnId};

/// Result alias used throughout `nusadb-core` and re-exported as the canonical
/// cross-crate result type.
pub type Result<T> = core::result::Result<T, Error>;

/// Cross-cutting error variants produced by core types and trait contracts.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error from an underlying [`PageStore`](crate::PageStore) adapter.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// Page checksum did not match the value stored in its header.
    #[error("page {page_id:?} checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch {
        /// The page that failed verification.
        page_id: PageId,
        /// The checksum stored in the page header.
        expected: u32,
        /// The checksum computed by reading the page.
        actual: u32,
    },

    /// Page header magic bytes did not match the NusaDB signature.
    #[error("page {page_id:?} has invalid magic bytes")]
    InvalidMagic {
        /// The page that failed verification.
        page_id: PageId,
    },

    /// fsync returned an error from the OS; durability of recent writes is unknown.
    #[error("fsync failed: {0}")]
    FsyncFailed(String),

    /// A storage operation produced a torn write (partial page on disk).
    #[error("torn write detected on page {page_id:?}")]
    TornWrite {
        /// The page that was torn.
        page_id: PageId,
    },

    /// A referenced table does not exist in the catalog.
    #[error("table not found: {name}")]
    TableNotFound {
        /// The unresolved table name.
        name: String,
    },

    /// `CREATE TABLE` named a table that already exists.
    #[error("table already exists: {name}")]
    TableExists {
        /// The duplicate table name.
        name: String,
    },

    /// An engine operation referenced a transaction id that is not active (never
    /// started, already committed, or already aborted).
    #[error("unknown or inactive transaction {txn:?}")]
    UnknownTransaction {
        /// The offending transaction id.
        txn: TxnId,
    },

    /// The transaction could not commit without violating its isolation level and was
    /// aborted; the caller should retry it.
    #[error("serialization conflict; transaction {txn:?} must retry")]
    SerializationConflict {
        /// The transaction that lost the conflict.
        txn: TxnId,
    },

    /// A write violated a declared constraint (e.g. a duplicate key in a `UNIQUE` index).
    ///
    /// The message carries the offending constraint's kind and column(s); a structured
    /// `{kind, column}` form was kept as an opaque string to avoid churning the many
    /// existing call sites — promote it only if a consumer needs to pattern-match the fields.
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),

    /// An index could not be created because one with that name already exists.
    #[error("index already exists: {name}")]
    IndexExists {
        /// The duplicate index name.
        name: String,
    },

    /// A referenced index does not exist in the catalog.
    #[error("index not found: {name}")]
    IndexNotFound {
        /// The unresolved index name.
        name: String,
    },

    /// A non-cycling sequence ran past its bound and has no more values to hand out.
    #[error("sequence exhausted: {name}")]
    SequenceExhausted {
        /// The exhausted sequence's name.
        name: String,
    },

    /// A referenced schema (namespace) does not exist in the catalog.
    #[error("schema not found: {name}")]
    SchemaNotFound {
        /// The unresolved schema name.
        name: String,
    },

    /// The transaction was aborted to break a lock-wait cycle (deadlock victim).
    ///
    /// Distinct from [`Error::SerializationConflict`]: a deadlock is a *cyclic* wait the lock
    /// manager resolved by choosing a victim, not an optimistic write/serialization conflict.
    #[error("deadlock detected; transaction {txn:?} chosen as victim")]
    Deadlock {
        /// The transaction selected as the deadlock victim.
        txn: TxnId,
    },

    /// An operation could not allocate the memory it required (e.g. a spill budget exceeded).
    #[error("out of memory: {0}")]
    OutOfMemory(String),

    /// An object could not be dropped because other objects still depend on it — e.g. `DROP SCHEMA`
    /// without `CASCADE` on a schema that still has tables. Distinct from an I/O failure: it is
    /// a dependency conflict the caller resolves with `CASCADE` (SQLSTATE `2BP01`).
    #[error("dependent objects still exist: {0}")]
    DependentObjectsExist(String),
}

impl Error {
    /// The 5-character SQLSTATE class code the wire protocol reports for this error. A
    /// serialization conflict and a deadlock get the standard *retryable* codes (`40001` /
    /// `40P01`) so client retry middleware classifies them correctly; a dependency conflict gets
    /// `2BP01` (dependent objects still exist); a constraint violation gets its standard
    /// integrity-class code (`23505`/`23503`/`23514`/`23502`, else the `23000` class code,
    /// classified from the engine-owned message constants); everything else falls back to the
    /// generic internal-error code `XX000`.
    #[must_use]
    pub fn sqlstate(&self) -> &'static str {
        match self {
            Self::SerializationConflict { .. } => "40001",
            Self::Deadlock { .. } => "40P01",
            Self::DependentObjectsExist(_) => "2BP01",
            Self::ConstraintViolation(message) => constraint_sqlstate(message),
            _ => "XX000",
        }
    }
}

/// The standard integrity-constraint SQLSTATE for a [`Error::ConstraintViolation`] message.
///
/// The variant deliberately carries its kind inside the message string (see its doc), and every
/// message is an engine-owned constant, so classifying on the kind words is stable: a losing
/// concurrent duplicate-key writer must reach client middleware as `23505` (unique violation) —
/// the same code the reference behaviour produces — not the opaque `XX000` (QA minor).
fn constraint_sqlstate(message: &str) -> &'static str {
    if message.contains("already exists") || message.contains("cannot drop") {
        // A catalog-shape conflict (declaring a duplicate constraint, dropping a referenced
        // one), not a data violation: the generic integrity class.
        "23000"
    } else if message.contains("duplicate key") {
        "23505"
    } else if message.contains("foreign key") {
        "23503"
    } else if message.contains("check constraint") {
        "23514"
    } else if message.contains("NULL") {
        "23502"
    } else if message.contains("unique constraint") || message.contains("primary key constraint") {
        // e.g. "existing rows violate the unique constraint on (…)" (ALTER TABLE backfill).
        "23505"
    } else {
        // integrity_constraint_violation, the class code — still classifiable, never internal.
        "23000"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_errors_map_to_standard_sqlstates() {
        // A serialization conflict and a deadlock must carry the standard retryable SQLSTATE codes so
        // client retry middleware classifies them (B-QA SQLSTATE); everything else is XX000.
        assert_eq!(
            Error::SerializationConflict { txn: TxnId(1) }.sqlstate(),
            "40001"
        );
        assert_eq!(Error::Deadlock { txn: TxnId(2) }.sqlstate(), "40P01");
        assert_eq!(Error::OutOfMemory("spill".to_owned()).sqlstate(), "XX000");
        // A dependency conflict (e.g. DROP SCHEMA RESTRICT on a non-empty schema) is 2BP01, not the
        // generic internal-error code — and not an I/O error.
        assert_eq!(
            Error::DependentObjectsExist("schema not empty".to_owned()).sqlstate(),
            "2BP01"
        );
    }

    #[test]
    fn constraint_violations_map_to_their_integrity_sqlstates() {
        // A losing concurrent duplicate-key writer must reach client middleware as 23505 —
        // the standard unique-violation code — never the opaque XX000 (QA minor).
        let cv = |m: &str| Error::ConstraintViolation(m.to_owned());
        assert_eq!(
            cv("duplicate key violates unique index u").sqlstate(),
            "23505"
        );
        assert_eq!(
            cv("duplicate key violates PRIMARY KEY constraint \"pk\" on (id)").sqlstate(),
            "23505"
        );
        assert_eq!(
            cv("foreign key fk_x: referenced key not present in parent").sqlstate(),
            "23503"
        );
        assert_eq!(
            cv("check constraint \"c\" is violated by an existing row in \"t\"").sqlstate(),
            "23514"
        );
        assert_eq!(
            cv("column \"a\" contains NULL values; cannot add PRIMARY KEY").sqlstate(),
            "23502"
        );
        assert_eq!(
            cv("existing rows violate the unique constraint on (a)").sqlstate(),
            "23505"
        );
        // Catalog-shape conflicts are the class code, not a data-violation code.
        assert_eq!(cv("table t already has a primary key").sqlstate(), "23000");
        assert_eq!(cv("foreign key fk_x already exists").sqlstate(), "23000");
        assert_eq!(
            cv("check constraint c already exists on this table").sqlstate(),
            "23000"
        );
    }
}
