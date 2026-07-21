//! Crate-level error type for `nusadb-sql`.
//!
//! Per NusaDB convention (see [`nusadb_core::error`]), each crate exposes a
//! single error enum. [`Error`] spans the SQL layer's parsing and semantic
//! analysis stages; cross-crate failures from the storage spine enter through
//! the [`Error::Core`] variant via `#[from]`.

use nusadb_core::ColumnType;

/// An error produced anywhere in the `nusadb-sql` pipeline (parser → analyzer).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // --- Parser ---------------------------------------------------------
    /// `sqlparser-rs` rejected the input as syntactically invalid. The string
    /// is the underlying parser's message; the `sqlparser` type itself is not
    /// exposed, to keep it out of NusaDB's public API.
    #[error("syntax error: {0}")]
    Syntax(String),

    /// The input contained no statement (empty, or only whitespace/comments).
    #[error("expected exactly one SQL statement, found none")]
    Empty,

    /// The input contained more than one statement; the parser accepts exactly
    /// one per call.
    #[error("expected a single SQL statement, found {0}")]
    MultipleStatements(usize),

    /// The input is valid SQL but uses a construct the Stage 4 SQL engine does
    /// not support yet. The string names the offending construct.
    #[error("unsupported SQL construct: {0}")]
    Unsupported(String),

    // --- Analyzer -------------------------------------------------------
    /// A referenced table is not present in the catalog.
    #[error("table not found: {name}")]
    TableNotFound {
        /// The unresolved table name.
        name: String,
    },

    /// `CREATE TABLE` named a table that already exists (and no `IF NOT
    /// EXISTS` clause was given).
    #[error("table already exists: {name}")]
    TableExists {
        /// The duplicate table name.
        name: String,
    },

    /// `DROP SCHEMA` named a schema that does not exist (and no `IF EXISTS` clause was given).
    #[error("schema not found: {name}")]
    SchemaNotFound {
        /// The unresolved schema name.
        name: String,
    },

    /// `DROP SEQUENCE` named a sequence that does not exist (and no `IF EXISTS` clause was given).
    #[error("sequence not found: {name}")]
    SequenceNotFound {
        /// The unresolved sequence name.
        name: String,
    },

    /// `DROP INDEX` named an index that does not exist (and no `IF EXISTS` clause was given).
    #[error("index not found: {name}")]
    IndexNotFound {
        /// The unresolved index name.
        name: String,
    },

    /// A referenced column is not present in its table.
    #[error("column not found: {column} (in table {table})")]
    ColumnNotFound {
        /// The table that was searched.
        table: String,
        /// The unresolved column name.
        column: String,
    },

    /// The same column was named more than once — in a `CREATE TABLE` column
    /// list, an `INSERT` target list, or an `UPDATE` assignment list.
    #[error("column `{name}` specified more than once")]
    DuplicateColumn {
        /// The repeated column name.
        name: String,
    },

    /// An expression's type does not match what its context requires.
    #[error("type mismatch in {context}: expected {expected:?}, found {found:?}")]
    TypeMismatch {
        /// Human-readable description of where the mismatch occurred.
        context: String,
        /// The type the context requires.
        expected: ColumnType,
        /// The type the expression actually has.
        found: ColumnType,
    },

    /// A value list had the wrong number of elements (e.g. `INSERT` row width
    /// does not match the target column count).
    #[error("{context}: expected {expected} value(s), found {found}")]
    ArityMismatch {
        /// Human-readable description of where the mismatch occurred.
        context: String,
        /// The required number of values.
        expected: usize,
        /// The number of values supplied.
        found: usize,
    },

    /// A bare `NULL` literal appeared where its type cannot be inferred from
    /// context (e.g. `SELECT NULL`, `NULL = NULL`).
    #[error("cannot infer the type of NULL in {context}")]
    AmbiguousNull {
        /// Human-readable description of the offending position.
        context: String,
    },

    /// A `NULL` literal was assigned to a `NOT NULL` column.
    #[error("NULL assigned to NOT NULL column `{column}`")]
    NotNullViolation {
        /// The non-nullable column that received a `NULL`.
        column: String,
    },

    /// A row written by a non-superuser fails the `WITH CHECK` of every applicable row-level-security
    /// policy on `table` — the new/updated row would not be visible to the writer under the
    /// policy, so the write is rejected.
    #[error("new row violates row-level security policy for table `{table}`")]
    RlsCheckViolation {
        /// The table whose policies the row failed.
        table: String,
    },

    /// The session lacks the privilege to run a statement. Used to reserve security administration —
    /// for example, only a superuser may create/alter/drop a row-level-security policy or toggle a
    /// table's RLS, so the very session RLS constrains cannot lift its own restrictions. Full
    /// role-based access control is deferred, so this guards the security-critical cases.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// `CREATE TRIGGER` named a trigger that already exists on the table (and no `OR REPLACE` was
    /// given).
    #[error("trigger `{name}` already exists on table `{table}`")]
    TriggerExists {
        /// The duplicate trigger name.
        name: String,
        /// The table the trigger is attached to.
        table: String,
    },

    /// `DROP TRIGGER` named a trigger that does not exist on the table (and no `IF EXISTS` was given).
    #[error("trigger `{name}` does not exist on table `{table}`")]
    TriggerNotFound {
        /// The unresolved trigger name.
        name: String,
        /// The table the trigger was expected on.
        table: String,
    },

    /// Trigger actions cascaded deeper than the recursion limit — a (possibly mutual) trigger that
    /// keeps re-firing itself. Aborts the statement rather than overflowing the stack.
    #[error("trigger recursion limit ({limit}) exceeded")]
    TriggerRecursionLimit {
        /// The maximum allowed trigger nesting depth.
        limit: usize,
    },

    /// `CREATE PROCEDURE` named a procedure that already exists (and no `OR REPLACE` was given).
    #[error("procedure `{name}` already exists")]
    ProcedureExists {
        /// The duplicate procedure name.
        name: String,
    },

    /// `DROP PROCEDURE` / `CALL` named a procedure that does not exist (no `IF EXISTS` for DROP).
    #[error("procedure `{name}` does not exist")]
    ProcedureNotFound {
        /// The unresolved procedure name.
        name: String,
    },

    /// `CALL` supplied a different number of arguments than the procedure declares.
    #[error("procedure `{name}` expects {expected} argument(s), got {found}")]
    ProcedureArgCount {
        /// The procedure name.
        name: String,
        /// The declared parameter count.
        expected: usize,
        /// The number of arguments supplied.
        found: usize,
    },

    /// `CALL` cascaded deeper than the recursion limit — a (possibly mutual) procedure that keeps
    /// calling itself. Aborts rather than overflowing the stack.
    #[error("procedure call recursion limit ({limit}) exceeded")]
    ProcedureRecursionLimit {
        /// The maximum allowed call nesting depth.
        limit: usize,
    },

    /// A NusaScript `RAISE` raised a user error from a procedure body. The string is the
    /// raised message.
    #[error("raised exception: {0}")]
    Raised(String),

    /// `CREATE FUNCTION` named a function that already exists (and no `OR REPLACE` was given).
    #[error("function `{name}` already exists")]
    FunctionExists {
        /// The duplicate function name.
        name: String,
    },

    /// `DROP FUNCTION` named a function that does not exist (and no `IF EXISTS` was given).
    #[error("function `{name}` does not exist")]
    FunctionNotFound {
        /// The unresolved function name.
        name: String,
    },

    // --- Executor ------------------------------------------------------
    /// The statement was cancelled before it finished — a statement timeout or an
    /// out-of-band cancel request. The transaction is rolled back.
    #[error("statement cancelled")]
    Cancelled,

    /// An arithmetic operation divided by zero (integer or floating-point).
    #[error("division by zero")]
    DivisionByZero,

    /// An integer arithmetic operation overflowed the `BIGINT` (`i64`) range — e.g.
    /// `9223372036854775807 + 1`. NusaDB errors rather than silently wrapping, matching the
    /// standard `22003 numeric_value_out_of_range` behaviour.
    #[error("integer out of range")]
    IntegerOutOfRange,

    /// A math function received an argument outside its domain — e.g. `SQRT` of a negative number or
    /// `LN` of a non-positive one. NusaDB raises rather than returning a silent `NaN`/`±inf` that
    /// would propagate through later arithmetic, matching the standard error behaviour.
    #[error("{0}")]
    ArgumentOutOfDomain(String),

    /// A stored tuple's encoded form did not match its declared schema (CRC
    /// truncation, version skew, or an internal codec bug).
    #[error("malformed tuple for schema (offset {offset})")]
    MalformedTuple {
        /// Byte offset into the tuple where decoding failed.
        offset: usize,
    },

    /// `decrypt(...)` could not recover the plaintext: a wrong key, a tampered
    /// or truncated ciphertext (the AEAD tag failed), or non-UTF-8 plaintext.
    #[error("decryption failed: {0}")]
    Decryption(&'static str),

    /// A literal string could not be parsed as the target type (e.g. a malformed
    /// `DATE`/`TIME`/`TIMESTAMP`/`UUID` value)
    #[error("invalid {ty:?} value: {value:?}")]
    InvalidValue {
        /// The type the value was expected to parse as.
        ty: ColumnType,
        /// The offending input.
        value: String,
    },

    /// A regular-expression argument to a `REGEXP_*` function failed to compile —
    /// an invalid pattern or an unsupported flag character.
    #[error("invalid regular expression: {0}")]
    InvalidRegex(String),

    /// A function call named a function that is neither a built-in nor a registered scalar UDF.
    /// The string is the function name.
    #[error("unknown function: {0}")]
    UnknownFunction(String),

    /// A registered scalar UDF returned an error when invoked. The message is the UDF's own.
    #[error("function `{name}` failed: {message}")]
    UdfFailed {
        /// The UDF name.
        name: String,
        /// The error message the UDF returned.
        message: String,
    },

    /// A failure surfaced by the storage/transaction spine (e.g. a catalog
    /// read error) while resolving schema.
    #[error(transparent)]
    Core(#[from] nusadb_core::Error),

    /// An error that already carries its own SQLSTATE, surfaced by a layer above the SQL engine — the
    /// wire server's database-cluster operations (`CREATE`/`DROP DATABASE`), whose codes (`42P04`,
    /// `3D000`, `55006`, …) the SQL layer does not itself produce. The message is shown verbatim.
    #[error("{message}")]
    Coded {
        /// The user-facing error message.
        message: String,
        /// The 5-character SQLSTATE to report.
        sqlstate: &'static str,
    },
}

impl Error {
    /// The 5-character SQLSTATE the wire protocol reports for this error. Engine errors (a
    /// serialization conflict / deadlock / constraint violation) carry their standard codes via
    /// [`nusadb_core::Error::sqlstate`]; a `NOT NULL` assignment gets `23502`; a cancelled
    /// statement (timeout / cancel request) gets the standard `57014` so drivers that branch on
    /// `query_canceled` recognise it; every other SQL-layer error uses the generic `XX000`.
    #[must_use]
    pub fn sqlstate(&self) -> &'static str {
        match self {
            Self::Core(e) => e.sqlstate(),
            Self::Coded { sqlstate, .. } => sqlstate,
            Self::NotNullViolation { .. } => "23502",
            Self::Cancelled => "57014", // query_canceled
            _ => "XX000",
        }
    }
}
