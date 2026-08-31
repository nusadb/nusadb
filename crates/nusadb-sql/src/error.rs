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

    /// The input is valid SQL but uses a construct NusaDB has not built yet. The string names the
    /// offending construct.
    ///
    /// This reports `0A000` (`feature_not_supported`), and that is a promise about what it carries:
    /// a caller reading it is entitled to conclude the statement would work elsewhere and that
    /// nothing here is its fault. A migration tool acts on that by skipping the statement and
    /// carrying on. So an ordinary mistake — an unknown role, a mistyped argument count, a value
    /// out of range — must **not** arrive here; those have variants of their own below, and putting
    /// one back in this one tells the tool to sail past a statement it should have stopped at.
    #[error("unsupported SQL construct: {0}")]
    Unsupported(String),

    // --- Split out of the old catch-all refusal --------------------------
    // Every variant in this block was once `Unsupported`, back when that one variant answered for
    // engine bugs, ordinary user mistakes and genuinely missing features alike — and reported all
    // three as `internal_error`. They are separate now because a client acts on the class: these
    // say whose mistake it was, which is the whole point of sending a code rather than a sentence.
    /// The engine's own invariant broke: a plan node reached a stage that cannot serve it, a lookup
    /// that the planner guarantees came back empty, a cache that was built went missing. Nothing
    /// the caller wrote can cause this and nothing it can write will avoid it, which is exactly
    /// what `internal_error` means.
    #[error("internal error: {0}")]
    Internal(String),

    /// The statement parses but says something illegal: a constant where an expression was
    /// required, two rows of different width compared, `SELECT *` with nothing to select from.
    /// Reported as `42601`, the same class as a syntax error, because it is one — just caught a
    /// stage later.
    #[error("{0}")]
    InvalidStatement(String),

    /// A call gave a known function the wrong number of arguments, or arguments of a type it has no
    /// form for. Reported as `42883` (`undefined_function`): overload resolution found no function
    /// of that name and shape, which is what the caller needs to hear.
    #[error("{0}")]
    FunctionArgs(String),

    /// A named object the statement referred to is not there — a role, a policy, a type, a domain,
    /// an operator class. Distinct from [`TableNotFound`](Self::TableNotFound) and friends only in
    /// that those name a kind the standard gives a class of its own.
    #[error("{0}")]
    ObjectNotFound(String),

    /// A `CREATE` named an object, of a kind without a class of its own, that is already there.
    #[error("{0}")]
    ObjectExists(String),

    /// A column the query named that is not there, in a position where no structured
    /// [`ColumnNotFound`](Self::ColumnNotFound) is available — a `USING` join key missing from one
    /// side, where the "table" is a join input rather than a name the caller wrote.
    #[error("{0}")]
    UndefinedColumn(String),

    /// A `$n` placeholder reached execution without a value. Reported as `42P02`
    /// (`undefined_parameter`) rather than a syntax error: the statement is well formed, and a
    /// driver reads this to know it under-supplied `Bind` rather than that the SQL was wrong.
    #[error("{0}")]
    UndefinedParameter(String),

    /// A cast between two types with no conversion between them — `42846` (`cannot_coerce`), the
    /// class that says the pair is the problem rather than the value.
    #[error("{0}")]
    CannotCoerce(String),

    /// The name resolves to an object of the wrong kind for this statement — `REFRESH MATERIALIZED
    /// VIEW` naming a plain table, say. Distinct from "not there at all", which is `42P01`/`42704`:
    /// `42809` tells the caller the name is fine and the statement is not.
    #[error("{0}")]
    WrongObjectType(String),

    /// A column reference that cannot stand where it stands: an `ORDER BY`/`GROUP BY` ordinal past
    /// the end of the select list, or a column alias list whose width disagrees with the query it
    /// names. (An unqualified name matching two relations is [`Self::AmbiguousColumn`].)
    #[error("{0}")]
    InvalidColumnReference(String),

    /// An unqualified column name that matches more than one relation in scope — it must be qualified
    /// with a table name to disambiguate.
    #[error("{0}")]
    AmbiguousColumn(String),

    /// An aggregate used where the query's grouping cannot support it — a column neither grouped
    /// nor aggregated, an aggregate in a clause that has no grouping to speak of.
    #[error("{0}")]
    InvalidGrouping(String),

    /// A `CREATE`/`ALTER TABLE` that would leave the table itself ill-formed: a second `PRIMARY
    /// KEY`, `ON COMMIT` on a table that is not temporary, dropping the last column.
    #[error("{0}")]
    InvalidTableDefinition(String),

    /// A write supplied a value for a column the table computes itself — `GENERATED ALWAYS AS
    /// IDENTITY` or a generated column. `428C9` is the class a client branches on to retry the
    /// write without that column.
    #[error("{0}")]
    GeneratedAlways(String),

    /// `EXECUTE`/`DEALLOCATE` named a prepared statement this session does not hold. Reported as
    /// `26000` (`invalid_sql_statement_name`), the code a pool checks to decide whether its cached
    /// statement was discarded and needs re-preparing — advice `internal_error` cannot give.
    #[error("{0}")]
    PreparedStatementNotFound(String),

    /// `FETCH`/`CLOSE` named a cursor this session does not hold (or that was already closed).
    /// Reported as `34000` (`invalid_cursor_name`).
    #[error("{0}")]
    CursorNotFound(String),

    /// A statement or object hit a built-in limit: a series longer than the row cap, a view nested
    /// deeper than the engine walks, a value too large to encode. Class `54` tells a client the
    /// request must get smaller — not that it was wrong, and not that the engine broke.
    #[error("{0}")]
    LimitExceeded(String),

    /// The statement would orphan something that still points at it — a table another table's
    /// foreign key references, a role that still owns objects, a privilege it has granted onward.
    /// `2BP01` is what a client reads to know that `CASCADE`, or dropping the dependants first, is
    /// the way through.
    #[error("{0}")]
    DependentObjects(String),

    /// A subquery used where one row was required returned more than one — `21000`, the standard's
    /// cardinality violation.
    #[error("{0}")]
    CardinalityViolation(String),

    /// A statement arrived while the transaction is already in the failed state; nothing will run
    /// until it is ended. `25P02` is the single code a pool watches most closely, because it says
    /// "roll back and start again" rather than "retry this".
    #[error("{0}")]
    TransactionAborted(String),

    /// `COMMIT`, `ROLLBACK` or `SAVEPOINT` arrived with no transaction open.
    #[error("{0}")]
    NoActiveTransaction(String),

    /// A statement that must precede a transaction arrived inside one — a nested `BEGIN`, a `SET
    /// TRANSACTION` after the first statement.
    #[error("{0}")]
    ActiveTransaction(String),

    /// A write arrived in a `READ ONLY` transaction.
    #[error("{0}")]
    ReadOnlyTransaction(String),

    /// A runtime argument's *value* is outside what the function accepts, though its type is right:
    /// a zero step for `generate_series`, a bucket count of zero, a malformed escape in `decode`,
    /// a `NULL` where a JSON object key must go.
    #[error("{0}")]
    InvalidParameterValue(String),

    /// A date/time computation left the representable range — adding an interval past the end of
    /// time, an epoch too large to be a timestamp. `22008` rather than the numeric `22003` so a
    /// client can tell a calendar overflow from an integer one.
    #[error("{0}")]
    DatetimeOverflow(String),

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
    ///
    /// Not every arity error: a set operation whose branches disagree on column count carries
    /// [`SetOpArityMismatch`](Self::SetOpArityMismatch), and one the engine finds in its own
    /// bookkeeping carries [`MalformedBatch`](Self::MalformedBatch), which reports a different
    /// class. Match on all three when you mean "any arity error" — the compiler cannot warn you.
    #[error("{context}: expected {expected} value(s), found {found}")]
    ArityMismatch {
        /// Human-readable description of where the mismatch occurred.
        context: String,
        /// The required number of values.
        expected: usize,
        /// The number of values supplied.
        found: usize,
    },

    /// A shape mismatch the engine found in its own bookkeeping, not in anything the query said:
    /// a record batch, a list array, or an encoded tuple built with a column count that disagrees
    /// with the schema it was built from. Both sides come from the same plan node, so reaching this
    /// is a bug in the engine and it reports `internal_error`, which is what it is.
    ///
    /// Split from [`ArityMismatch`](Self::ArityMismatch) precisely so that one can report a
    /// malformed *query* without dressing an engine fault as the caller's mistake.
    #[error("{context}: expected {expected} value(s), found {found}")]
    MalformedBatch {
        /// Human-readable description of where the mismatch occurred.
        context: String,
        /// The required number of values.
        expected: usize,
        /// The number of values supplied.
        found: usize,
    },

    /// The branches of a set operation select a different number of columns — either the two sides
    /// of a `UNION`/`INTERSECT`/`EXCEPT`, or the anchor and recursive terms of a recursive CTE
    /// (which are the branches of its `UNION ALL`).
    ///
    /// Carried separately from [`ArityMismatch`](Self::ArityMismatch) so a caller can tell the two
    /// apart; both report `42601`, since either is a malformed query.
    #[error("{context}: expected {expected} value(s), found {found}")]
    SetOpArityMismatch {
        /// Human-readable description of which set operation disagreed.
        context: String,
        /// The number of columns the first branch selects.
        expected: usize,
        /// The number of columns the other branch selects.
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

    /// A row written through a view created `WITH CHECK OPTION` would not be visible through that
    /// view (it fails the view's own `WHERE`), so the write is rejected.
    #[error("new row violates check option for view `{view}`")]
    ViewCheckOptionViolation {
        /// The view whose check option the row failed.
        view: String,
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

    /// A character string is longer than its column's declared `VARCHAR(n)` / `CHAR(n)` length, and
    /// the overflow is not all trailing blanks (which would be truncated silently). SQLSTATE `22001`.
    #[error("value too long for type {ty}")]
    StringTooLong {
        /// The declared character type whose length was exceeded, already rendered (`VARCHAR(5)`).
        ty: String,
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
    /// The 5-character SQLSTATE the wire protocol reports for this error.
    ///
    /// The class is the part a client acts on: a driver, pool or migration tool reads it to decide
    /// whether to retry, report to the user, or stop. So a mistake in the query says so — class
    /// `42` for a malformed or unresolvable statement, `22` for a value the type cannot represent,
    /// `23` for a violated constraint — and `XX000` (`internal_error`) is reserved for the engine's
    /// own faults. Reporting an ordinary mistake as an engine fault is not a cosmetic error: it
    /// tells a caller there is nothing it can fix.
    ///
    /// `0A000` (`feature_not_supported`) is returned by exactly one variant,
    /// [`Unsupported`](Self::Unsupported), and it means what it says: NusaDB has not built this.
    /// It used to mean less than that — the same variant also carried unknown roles, miscounted
    /// arguments and engine bugs — which is why the block of variants split out of it exists.
    ///
    /// Engine errors carry their standard codes via [`nusadb_core::Error::sqlstate`], and a
    /// cancelled statement reports `57014` so a driver branching on `query_canceled` recognises it.
    ///
    /// The match below is exhaustive on purpose. A wildcard arm is what once let most of this
    /// enum report `internal_error` unnoticed; without one, a new variant does not compile until
    /// someone decides its class.
    #[must_use]
    pub fn sqlstate(&self) -> &'static str {
        match self {
            Self::Core(e) => e.sqlstate(),
            Self::Coded { sqlstate, .. } => sqlstate,
            Self::NotNullViolation { .. } => "23502",
            Self::Cancelled => "57014", // query_canceled
            // Class 42 — the query is malformed, not the engine. `XX000` here would tell a driver
            // the server had faulted on what is really a user typo. A driver, ORM or migration tool
            // reads the class to decide whether to retry, report or abort, so a whole category of
            // ordinary mistakes arriving as "internal error" is a defect that never shows up in
            // hand testing and behaves strangely in an integration layer.
            // syntax_error, including the arity mismatches: supplying the wrong number of values
            // is a malformed statement, not a fault.
            Self::Syntax(_)
            | Self::MultipleStatements(_)
            | Self::Empty
            | Self::ArityMismatch { .. }
            | Self::SetOpArityMismatch { .. }
            | Self::InvalidStatement(_) => "42601",
            Self::TableNotFound { .. } => "42P01", // undefined_table
            Self::TableExists { .. } => "42P07",   // duplicate_table
            Self::SchemaNotFound { .. } => "3F000", // invalid_schema_name
            // undefined_column — a name the query used that resolves to no column.
            Self::ColumnNotFound { .. } | Self::UndefinedColumn(_) => "42703",
            Self::UndefinedParameter(_) => "42P02", // undefined_parameter
            Self::CannotCoerce(_) => "42846",       // cannot_coerce
            Self::WrongObjectType(_) => "42809",    // wrong_object_type
            Self::DuplicateColumn { .. } => "42701", // duplicate_column
            Self::TypeMismatch { .. } => "42804",   // datatype_mismatch
            Self::AmbiguousNull { .. } => "42P18",  // indeterminate_datatype
            // Class 42501 — the role lacks the right, whether refused outright or by a row policy.
            Self::PermissionDenied(_) | Self::RlsCheckViolation { .. } => "42501",
            // Class 44000 — with_check_option_violation: a write through a view broke its CHECK OPTION.
            Self::ViewCheckOptionViolation { .. } => "44000",
            // undefined_function — a name the caller used that resolves to nothing callable, or a
            // call whose shape matches no form of a name that does exist. Overload resolution
            // cannot tell those apart and neither does the standard: both are `42883`.
            Self::UnknownFunction(_)
            | Self::FunctionNotFound { .. }
            | Self::ProcedureNotFound { .. }
            | Self::ProcedureArgCount { .. }
            | Self::FunctionArgs(_) => "42883",
            // duplicate_function / duplicate_object — creating something already there.
            Self::FunctionExists { .. } | Self::ProcedureExists { .. } => "42723",
            // duplicate_object
            Self::TriggerExists { .. } | Self::ObjectExists { .. } => "42710",
            // undefined_object — a named thing that is not there.
            Self::TriggerNotFound { .. }
            | Self::SequenceNotFound { .. }
            | Self::IndexNotFound { .. }
            | Self::ObjectNotFound { .. } => "42704",
            Self::InvalidColumnReference(_) => "42P10", // invalid_column_reference
            Self::AmbiguousColumn(_) => "42702",        // ambiguous_column
            Self::InvalidGrouping(_) => "42803",        // grouping_error
            Self::InvalidTableDefinition(_) => "42P16", // invalid_table_definition
            Self::GeneratedAlways(_) => "428C9",        // generated_always
            // invalid_sql_statement_name — the name is gone, so re-preparing is the way out.
            Self::PreparedStatementNotFound(_) => "26000",
            Self::CursorNotFound(_) => "34000",
            // Class 25 — the transaction is not in a state that admits this statement. A pool reads
            // these to decide between "roll back and retry" and "this connection is fine".
            Self::TransactionAborted(_) => "25P02", // in_failed_sql_transaction
            Self::NoActiveTransaction(_) => "25P01", // no_active_sql_transaction
            Self::ActiveTransaction(_) => "25001",  // active_sql_transaction
            Self::ReadOnlyTransaction(_) => "25006", // read_only_sql_transaction
            // dependent_objects_still_exist — CASCADE, or drop the dependants, is the way through.
            Self::DependentObjects(_) => "2BP01",
            Self::CardinalityViolation(_) => "21000", // cardinality_violation
            // program_limit_exceeded — the request must get smaller; it was not wrong.
            Self::LimitExceeded(_) => "54000",
            // The engine has not built this. The only variant that may say so; see its doc.
            Self::Unsupported(_) => "0A000", // feature_not_supported
            // statement_too_complex — the nesting limit, not a fault.
            Self::TriggerRecursionLimit { .. } | Self::ProcedureRecursionLimit { .. } => "54001",
            // raise_exception — the user's own RAISE from a procedure body. Reporting this as an
            // engine fault is the single most misleading code here: the statement did exactly what
            // it was told to.
            Self::Raised { .. } => "P0001",
            // Class 22 — data exception: a runtime value error, not an internal fault.
            Self::DivisionByZero => "22012", // division_by_zero
            // numeric_value_out_of_range — an overflow or a value outside a function's domain.
            Self::IntegerOutOfRange | Self::ArgumentOutOfDomain(_) => "22003",
            // A text value that does not parse as its target type: a bad date/time is
            // invalid_datetime_format (22007), everything else is invalid_text_representation (22P02).
            Self::InvalidValue { ty, .. } if is_datetime(*ty) => "22007",
            Self::InvalidValue { .. } => "22P02",
            Self::StringTooLong { .. } => "22001", // string_data_right_truncation
            Self::InvalidRegex(_) => "2201B",      // invalid_regular_expression
            // invalid_parameter_value — the argument's type is right, its value is not.
            Self::InvalidParameterValue(_) => "22023",
            Self::DatetimeOverflow(_) => "22008", // datetime_field_overflow
            // The residue, each for its own reason: `Internal`, `MalformedBatch` and
            // `MalformedTuple` are the engine's own bookkeeping; `Decryption` and `UdfFailed` come
            // from outside the engine but have no class that says more than "it went wrong" here.
            Self::Internal(_)
            | Self::MalformedBatch { .. }
            | Self::MalformedTuple { .. }
            | Self::Decryption(_)
            | Self::UdfFailed { .. } => INTERNAL_ERROR,
        }
    }
}

/// The code for a fault in the engine rather than in the query — `internal_error`.
///
/// Exposed so a caller reporting its own internal failure can name this directly instead of
/// reaching for whichever `Error` variant happens to map here today. One did, and it would have
/// started reporting "feature not supported" the moment that variant was given a code of its own.
pub const INTERNAL_ERROR: &str = "XX000";

/// Whether a column type is a date/time type, whose malformed text form is
/// `invalid_datetime_format` (`22007`) rather than the generic `invalid_text_representation`.
const fn is_datetime(ty: ColumnType) -> bool {
    matches!(
        ty,
        ColumnType::Date
            | ColumnType::Time
            | ColumnType::TimeTz
            | ColumnType::Timestamp
            | ColumnType::TimestampTz
    )
}

#[cfg(test)]
mod tests {
    use super::{Error, INTERNAL_ERROR};
    use nusadb_core::ColumnType;

    /// Every variant a query can provoke reports the class a client acts on. Asserted by code, not
    /// by message text: a driver switches on the code, and a test matching on wording would pass
    /// while the code beneath it was wrong.
    #[test]
    fn a_users_mistake_reports_its_own_class_not_internal_error() {
        let cases: Vec<(Error, &str)> = vec![
            (Error::Syntax("bad".to_owned()), "42601"),
            (Error::Empty, "42601"),
            (Error::MultipleStatements(2), "42601"),
            (
                Error::ArityMismatch {
                    context: "INSERT".to_owned(),
                    expected: 2,
                    found: 3,
                },
                "42601",
            ),
            (
                Error::TableNotFound {
                    name: "t".to_owned(),
                },
                "42P01",
            ),
            (
                Error::TableExists {
                    name: "t".to_owned(),
                },
                "42P07",
            ),
            (
                Error::SchemaNotFound {
                    name: "s".to_owned(),
                },
                "3F000",
            ),
            (
                Error::ColumnNotFound {
                    table: "t".to_owned(),
                    column: "c".to_owned(),
                },
                "42703",
            ),
            (
                Error::DuplicateColumn {
                    name: "c".to_owned(),
                },
                "42701",
            ),
            (
                Error::TypeMismatch {
                    context: "WHERE".to_owned(),
                    expected: ColumnType::Int,
                    found: ColumnType::Text,
                },
                "42804",
            ),
            (Error::PermissionDenied("no SELECT".to_owned()), "42501"),
            (
                Error::RlsCheckViolation {
                    table: "t".to_owned(),
                },
                "42501",
            ),
            (
                Error::AmbiguousNull {
                    context: "COALESCE".to_owned(),
                },
                "42P18",
            ),
            (
                Error::SetOpArityMismatch {
                    context: "UNION".to_owned(),
                    expected: 2,
                    found: 3,
                },
                "42601",
            ),
            (Error::UnknownFunction("nosuchfn".to_owned()), "42883"),
            (Error::Raised("custom".to_owned()), "P0001"),
        ];
        for (error, want) in cases {
            assert_eq!(
                error.sqlstate(),
                want,
                "wrong class for `{error}`; a client reads this to decide what to do"
            );
        }
    }

    /// The same contract for the variants split out of the old catch-all refusal. Kept apart from
    /// the case list above so the split's own coverage is visible as a block: every one of these
    /// reported `internal_error` before, and each now names whose mistake it was.
    #[test]
    fn a_refusal_split_from_the_catch_all_reports_its_own_class() {
        let cases: Vec<(Error, &str)> = vec![
            (Error::InvalidStatement("no FROM".to_owned()), "42601"),
            (Error::FunctionArgs("abs() takes 1".to_owned()), "42883"),
            (
                Error::ObjectNotFound("role `alice` does not exist".to_owned()),
                "42704",
            ),
            (
                Error::ObjectExists("role `alice` already exists".to_owned()),
                "42710",
            ),
            (
                Error::InvalidColumnReference("ORDER BY position 9".to_owned()),
                "42P10",
            ),
            (Error::InvalidGrouping("not grouped".to_owned()), "42803"),
            (
                Error::InvalidTableDefinition("two PRIMARY KEYs".to_owned()),
                "42P16",
            ),
            (
                Error::GeneratedAlways("identity column".to_owned()),
                "428C9",
            ),
            (
                Error::PreparedStatementNotFound("no statement \"p1\"".to_owned()),
                "26000",
            ),
            (Error::TransactionAborted("aborted".to_owned()), "25P02"),
            (Error::NoActiveTransaction("no txn".to_owned()), "25P01"),
            (Error::ActiveTransaction("nested BEGIN".to_owned()), "25001"),
            (Error::ReadOnlyTransaction("read only".to_owned()), "25006"),
            (
                Error::DependentObjects("a view uses it".to_owned()),
                "2BP01",
            ),
            (Error::CardinalityViolation("two rows".to_owned()), "21000"),
            (Error::LimitExceeded("too many rows".to_owned()), "54000"),
            (
                Error::InvalidParameterValue("step is 0".to_owned()),
                "22023",
            ),
            (Error::DatetimeOverflow("year 300000".to_owned()), "22008"),
            (
                Error::AmbiguousColumn("column `a` is ambiguous".to_owned()),
                "42702",
            ),
            (
                Error::StringTooLong {
                    ty: "VARCHAR(5)".to_owned(),
                },
                "22001",
            ),
        ];
        for (error, want) in cases {
            assert_eq!(
                error.sqlstate(),
                want,
                "wrong class for `{error}`; a client reads this to decide what to do"
            );
        }
    }

    /// A fault in the engine keeps reporting `internal_error`, which is what it is. Recoding the
    /// user-facing variants must not sweep these along: the batch-construction mismatch shares its
    /// wording with the query-level one and differs only in which variant carries it.
    #[test]
    fn an_engine_fault_still_reports_internal_error() {
        let internal: Vec<Error> = vec![
            Error::MalformedBatch {
                context: "column 0".to_owned(),
                expected: 2,
                found: 3,
            },
            Error::MalformedTuple { offset: 7 },
            Error::Internal("plan node reached the wrong stage".to_owned()),
            Error::UdfFailed {
                name: "f".to_owned(),
                message: "boom".to_owned(),
            },
        ];
        for error in internal {
            assert_eq!(
                error.sqlstate(),
                INTERNAL_ERROR,
                "`{error}` is the engine's own failure and must not be dressed as a user mistake"
            );
        }
    }

    /// `feature_not_supported` is a claim, not a shrug: it tells a migration tool the statement is
    /// fine and this server merely lacks the feature, which the tool answers by skipping it and
    /// carrying on. So exactly one variant may say it, and it is the one whose name promises it.
    ///
    /// The asymmetry is the point. A missing feature reported as `internal_error` costs a caller a
    /// needless stop; an ordinary mistake reported as `feature_not_supported` costs it the
    /// statement it should have stopped at. That is why the split had to happen before this
    /// mapping did, and why a variant that carries a user's mistake must never be added here.
    #[test]
    fn only_a_genuinely_missing_feature_reports_feature_not_supported() {
        assert_eq!(Error::Unsupported("LATERAL".to_owned()).sqlstate(), "0A000");
        let user_mistakes: Vec<Error> = vec![
            Error::ObjectNotFound("role `alice` does not exist".to_owned()),
            Error::FunctionArgs("abs() expects 1 argument, got 2".to_owned()),
            Error::InvalidParameterValue("step must not be zero".to_owned()),
            Error::InvalidStatement("SELECT * requires a FROM clause".to_owned()),
        ];
        for error in user_mistakes {
            assert_ne!(
                error.sqlstate(),
                "0A000",
                "`{error}` is the caller's mistake; telling a migration tool the feature is missing \
                 invites it to skip a statement it should have stopped at"
            );
        }
    }
}
