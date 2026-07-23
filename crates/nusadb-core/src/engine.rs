//! The **treaty**: the contract between NusaDB's storage/transaction *spine* and its
//! SQL *surface*.
//!
//! This module exists so the two halves of the engine can be built in parallel against
//! one agreed seam:
//!
//! - The **spine** (`nusadb-storage` + `nusadb-wal` + `nusadb-btree`)
//!   *implements* [`StorageEngine`] and [`TupleScan`].
//! - The **surface** (`nusadb-sql` and above) *consumes* `&dyn StorageEngine`, and uses
//!   an in-memory implementation as a test double until the spine is ready.
//!
//! # Design choices that make parallel work possible
//!
//! - **Opaque tuple bytes.** The contract speaks in [`Tuple`] blobs, not typed SQL
//!   values. Storage stores and returns bytes; the SQL layer owns all value
//!   encoding/decoding. This keeps the rich type system (`TypeTag`, `Value`, vectorized
//!   batches) entirely inside `nusadb-sql`. The only shared vocabulary is the minimal
//!   [`ColumnType`] the catalog needs to persist a schema.
//! - **Id-based transactions.** Every method takes a [`TxnId`] rather than borrowing a
//!   transaction handle. The engine owns the transaction table internally (keyed by
//!   `TxnId`), which matches how MVCC engines actually work and avoids borrow-checker
//!   conflicts when a statement must scan and write the same transaction (e.g.
//!   `INSERT ... SELECT`). The SQL layer can wrap a `TxnId` in its own RAII guard for
//!   auto-rollback ergonomics.
//!
//! # Stability
//!
//! This is a **negotiated contract**. Changing a signature here affects both teams —
//! treat any edit as an API change and coordinate before merging.

use std::ops::Bound;

use crate::{Error, IndexId, PageId, Result, SchemaId, SequenceId, SlotIdx, TableId, TxnId};

/// Stable physical address of a stored tuple: `(page, slot)`.
///
/// A `Tid` stays valid across page reorganization, so the SQL layer can hold one to
/// `UPDATE`/`DELETE` a specific row it previously read from a
/// [`scan`](StorageEngine::scan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tid {
    /// Page holding the tuple.
    pub page: PageId,
    /// Slot within the page.
    pub slot: SlotIdx,
}

/// A stored tuple as opaque bytes.
///
/// The storage spine never interprets these bytes; the SQL surface owns the row
/// encoding. Keeping the contract byte-oriented is what lets the type system live
/// entirely in `nusadb-sql`.
pub type Tuple = Vec<u8>;

/// A shared, read-only tuple handed out by a [`TupleScan`].
///
/// `Arc<[u8]>` so a scan yields a row's bytes with a refcount bump instead of copying them on every
/// read (§1.5): the version store owns each tuple once, and a scan shares it. Callers
/// decode from `&[u8]` (an `Arc<[u8]>` derefs), so the SQL layer still never owns or mutates the
/// bytes. The owned [`Tuple`] is still the *input* type (`insert`/`update` take `&[u8]`).
pub type SharedTuple = std::sync::Arc<[u8]>;

/// SQL transaction isolation level.
///
/// Defined in core because both the txn engine that *enforces* a level and the SQL
/// layer that *requests* one need to name it. Default is [`IsolationLevel::ReadCommitted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// `READ UNCOMMITTED`. Accepted for SQL compatibility but **promoted to
    /// [`ReadCommitted`](Self::ReadCommitted)** — a standard choice for an MVCC engine whose reads
    /// always exclude the active (uncommitted) set, so a `READ UNCOMMITTED` transaction observes
    /// only committed data, exactly like `READ COMMITTED`. No dirty reads occur at any level.
    ReadUncommitted,
    /// Reads observe only committed data, re-evaluated per statement.
    #[default]
    ReadCommitted,
    /// All reads in the transaction observe the snapshot taken at its start.
    RepeatableRead,
    /// Fully serializable execution; conflicting transactions are aborted.
    Serializable,
}

/// Strength of an explicit row lock requested via `SELECT ... FOR SHARE | FOR UPDATE`.
///
/// Named in core because the SQL layer *requests* a row lock and the txn engine *enforces* it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLockMode {
    /// `FOR SHARE`: a shared read lock — compatible with other shared lockers, but blocks any
    /// writer (or `FOR UPDATE`) of the row.
    Shared,
    /// `FOR UPDATE`: an exclusive lock — incompatible with any other locker or writer of the row.
    Exclusive,
}

/// Strength of an explicit table lock requested via `LOCK TABLE`.
///
/// The two extremes of the SQL table-lock hierarchy. `AccessExclusive` (taken by `DROP`/`ALTER`/
/// `TRUNCATE`) conflicts with every other table lock **and** with concurrent row activity on the
/// table; `AccessShare` (taken by a plain `SELECT`) conflicts only with `AccessExclusive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableLockMode {
    /// `ACCESS SHARE`: compatible with everything except `AccessExclusive`.
    AccessShare,
    /// `ACCESS EXCLUSIVE`: incompatible with any other table lock or row write/lock on the table.
    AccessExclusive,
}

/// The minimal column-type vocabulary needed to persist a schema in the catalog.
///
/// This is intentionally *not* the SQL layer's full type system: `nusadb-sql` defines
/// the rich `TypeTag`/`Value` types and maps them onto these for storage. Extend only
/// when the catalog must physically distinguish a new type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// Boolean.
    Bool,
    /// 64-bit signed integer (`INTEGER` / `INT` / `INT4`).
    Int,
    /// `SMALLINT` / `INT2` — a 16-bit-range integer. Stored identically to [`ColumnType::Int`] (the
    /// range is enforced separately by a synthetic check); the declared type is kept so it round-trips
    /// in DDL (`SHOW COLUMNS` / `information_schema`), mirroring [`ColumnType::VarChar`].
    SmallInt,
    /// `BIGINT` / `INT8` — a 64-bit integer. Stored identically to [`ColumnType::Int`]; the declared
    /// type is kept only for DDL round-tripping (the physical storage is the same 64-bit integer).
    BigInt,
    /// 64-bit IEEE-754 floating point (`DOUBLE PRECISION` / `FLOAT8`).
    Float,
    /// `REAL` / `FLOAT4` — single-precision float. Stored identically to [`ColumnType::Float`] (a 64-bit
    /// double); the declared type is kept only so it round-trips in DDL / `information_schema`,
    /// mirroring [`ColumnType::VarChar`].
    Real,
    /// UTF-8 text, variable length.
    Text,
    /// `VARCHAR(n)` / `CHARACTER VARYING(n)` — UTF-8 text with a declared maximum length `n`. Stored
    /// identically to [`ColumnType::Text`] (the length is enforced separately); the length is kept so
    /// the declared type round-trips in DDL (`SHOW COLUMNS` / `information_schema`).
    VarChar(u32),
    /// `CHAR(n)` / `CHARACTER(n)` — UTF-8 text with a declared length `n`. Like [`ColumnType::VarChar`]
    /// it is stored as text (NusaDB does not blank-pad); the length is kept for DDL round-tripping.
    Char(u32),
    /// Raw bytes, variable length.
    Bytes,
    /// Microseconds since the Unix epoch (local / no time zone).
    Timestamp,
    /// Calendar date — days since the Unix epoch (1970-01-01), proleptic Gregorian.
    Date,
    /// Time of day — microseconds since midnight, in `[0, 86_400_000_000)`.
    Time,
    /// Timestamp with time zone — microseconds since the Unix epoch, normalized to UTC.
    TimestampTz,
    /// Time of day with time zone — one packed `i64` carrying both the as-entered local time
    /// and its zone offset: `utc_equivalent_micros * 2^18 + (zone_west_secs + 2^17)`, chosen so
    /// plain `i64` ordering compares by the UTC-equivalent instant with the zone as tie-break
    /// (P-TIMETZ; see `nusadb-sql`'s `temporal` module for the pack/unpack helpers).
    TimeTz,
    /// 128-bit UUID, stored as 16 bytes.
    Uuid,
    /// Exact decimal (`NUMERIC` / `DECIMAL`) with a declared precision + scale. A
    /// `precision` of `0` means unconstrained (`NUMERIC` with no arguments); `scale` is the number
    /// of fractional digits.
    Numeric {
        /// Total significant digits allowed (`0` = unconstrained).
        precision: u8,
        /// Fractional digits.
        scale: u8,
    },
    /// `JSON` document, stored as canonical (sorted-key) text.
    Json,
    /// `JSONB` document. Stored identically to [`ColumnType::Json`] (canonical text); the declared
    /// type is kept only so it round-trips in DDL / `information_schema`, mirroring [`ColumnType::VarChar`].
    Jsonb,
    /// Calendar duration `INTERVAL` — months + days + microseconds.
    Interval,
    /// One-dimensional array of a scalar element type, e.g. `INT[]` / `TEXT[]`. The element
    /// type is a [`ArrayElem`] (a `Copy` scalar) so `ColumnType` stays `Copy`; nested arrays are
    /// not represented.
    Array(ArrayElem),
    /// Fixed-dimension `f32` vector `VECTOR(n)` for similarity search. The `u32` is the
    /// declared dimension `n`; every value of the column carries exactly `n` components. The element
    /// count keeps `ColumnType` `Copy`.
    Vector(u32),
}

impl ColumnType {
    /// The *physical* storage type — how values of this column are actually encoded. The declared
    /// character types `VARCHAR(n)`/`CHAR(n)` are stored identically to [`ColumnType::Text`], so they
    /// normalize to it here; every other type maps to itself.
    ///
    /// Use this anywhere the answer depends on storage/runtime behaviour (value encoding, coercion,
    /// type equality), so the declared length affects only DDL rendering — never semantics. The raw
    /// (un-normalized) type is what the catalog persists and what `SHOW COLUMNS` renders.
    #[must_use]
    pub const fn physical(self) -> Self {
        match self {
            Self::VarChar(_) | Self::Char(_) => Self::Text,
            // SMALLINT / BIGINT are stored as the same 64-bit integer as INT; the declared width is
            // metadata only (enforced by a synthetic range check, rendered in DDL).
            Self::SmallInt | Self::BigInt => Self::Int,
            // REAL is stored as the same 64-bit double as FLOAT; JSONB as the same canonical text as
            // JSON. Declared type is metadata only (rendered in DDL).
            Self::Real => Self::Float,
            Self::Jsonb => Self::Json,
            other => other,
        }
    }
}

/// The scalar element type of an [`ColumnType::Array`].
///
/// A dedicated `Copy` enum (rather than `Box<ColumnType>`) so [`ColumnType`] stays `Copy`. It
/// mirrors the scalar column types that may appear as array elements; map it to the corresponding
/// [`ColumnType`] with [`ArrayElem::column_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayElem {
    /// `BOOL[]`.
    Bool,
    /// `INT[]`.
    Int,
    /// `FLOAT[]`.
    Float,
    /// `NUMERIC[]` — exact-decimal elements (unconstrained precision/scale; an array element does not
    /// carry its own precision/scale).
    Numeric,
    /// `TEXT[]`.
    Text,
    /// `DATE[]`.
    Date,
    /// `TIME[]`.
    Time,
    /// `TIMESTAMP[]`.
    Timestamp,
    /// `TIMESTAMPTZ[]`.
    TimestampTz,
    /// `UUID[]`.
    Uuid,
}

impl ArrayElem {
    /// The scalar [`ColumnType`] of this element.
    #[must_use]
    pub const fn column_type(self) -> ColumnType {
        match self {
            Self::Bool => ColumnType::Bool,
            Self::Int => ColumnType::Int,
            Self::Float => ColumnType::Float,
            Self::Numeric => ColumnType::Numeric {
                precision: 0,
                scale: 0,
            },
            Self::Text => ColumnType::Text,
            Self::Date => ColumnType::Date,
            Self::Time => ColumnType::Time,
            Self::Timestamp => ColumnType::Timestamp,
            Self::TimestampTz => ColumnType::TimestampTz,
            Self::Uuid => ColumnType::Uuid,
        }
    }

    /// The element type for a scalar `ColumnType`, or `None` for types that cannot be array
    /// elements (nested arrays, `JSON`, `BYTES`, `INTERVAL`, `VECTOR`).
    ///
    /// `VARCHAR(n)`/`CHAR(n)` collapse to a `TEXT` element (the declared length is not tracked
    /// per array element), via [`ColumnType::physical`].
    #[must_use]
    pub const fn from_column_type(ty: ColumnType) -> Option<Self> {
        match ty.physical() {
            ColumnType::Bool => Some(Self::Bool),
            ColumnType::Int => Some(Self::Int),
            ColumnType::Float => Some(Self::Float),
            ColumnType::Numeric { .. } => Some(Self::Numeric),
            ColumnType::Text => Some(Self::Text),
            ColumnType::Date => Some(Self::Date),
            ColumnType::Time => Some(Self::Time),
            ColumnType::Timestamp => Some(Self::Timestamp),
            ColumnType::TimestampTz => Some(Self::TimestampTz),
            ColumnType::Uuid => Some(Self::Uuid),
            // `physical()` already mapped VarChar/Char → Text and SmallInt/BigInt → Int; these never
            // reach here.
            ColumnType::VarChar(_)
            | ColumnType::Char(_)
            | ColumnType::SmallInt
            | ColumnType::BigInt
            | ColumnType::Real
            | ColumnType::TimeTz
            | ColumnType::Bytes
            | ColumnType::Json
            | ColumnType::Jsonb
            | ColumnType::Interval
            | ColumnType::Array(_)
            | ColumnType::Vector(_) => None,
        }
    }
}

/// One column in a [`TableSchema`] or [`TableDef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    /// Column name (unique within its table).
    pub name: String,
    /// Physical column type.
    pub ty: ColumnType,
    /// Whether the column accepts `NULL`.
    pub nullable: bool,
}

/// The default schema (namespace) a table lives in when none is given — the single namespace that
/// existed before multi-schema support. `public.t` and a bare `t` denote the same table.
pub const PUBLIC_SCHEMA: &str = "public";

/// A table's persisted schema, as returned by the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSchema {
    /// Catalog id.
    pub id: TableId,
    /// Schema (namespace) the table lives in. [`PUBLIC_SCHEMA`] for the default namespace.
    pub schema: String,
    /// Table name.
    pub name: String,
    /// Columns in declaration order.
    pub columns: Vec<ColumnDef>,
}

/// The definition supplied to [`StorageEngine::create_table`] — a schema that does not
/// yet have a [`TableId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDef {
    /// Schema (namespace) to create the table in. [`PUBLIC_SCHEMA`] for the default namespace.
    pub schema: String,
    /// Table name.
    pub name: String,
    /// Columns in declaration order.
    pub columns: Vec<ColumnDef>,
}

/// The definition supplied to [`StorageEngine::create_sequence`].
///
/// `SERIAL`/`BIGSERIAL`/`IDENTITY` desugar onto a sequence with `start = 1`, `increment = 1`. A
/// sequence's value advances **non-transactionally** — a rolled-back transaction does not give the
/// value back (gap semantics), so two callers never receive the same number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceDef {
    /// Sequence name (unique within the catalog).
    pub name: String,
    /// First value handed out by [`sequence_next`](StorageEngine::sequence_next).
    pub start: i64,
    /// Amount added on each [`sequence_next`](StorageEngine::sequence_next); may be negative.
    pub increment: i64,
    /// Inclusive lower bound.
    pub min_value: i64,
    /// Inclusive upper bound.
    pub max_value: i64,
    /// Whether to wrap to the opposite bound on overflow instead of erroring.
    pub cycle: bool,
}

/// The access method of a secondary index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Ordered tree — supports equality and range scans (the default).
    BTree,
    /// Hash — equality only.
    Hash,
    /// Block range index — coarse min/max summaries for large, naturally-ordered tables.
    Brin,
}

/// A secondary index's catalog definition.
///
/// The engine treats indexes as opaque: the SQL layer encodes the indexed key into the bytes it
/// passes to [`index_insert`](StorageEngine::index_insert) / scans, so every field here is catalog
/// metadata for the SQL layer, not something the engine interprets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDef {
    /// Index name (unique within the catalog).
    pub name: String,
    /// The table this index covers.
    pub table: TableId,
    /// Key column names, in order — for a **plain-column** index. Informational for the planner
    /// (which can offer it as an equality/range access path); the engine indexes opaque bytes.
    /// Empty when the index keys on expressions instead (`key_exprs` is then set).
    pub columns: Vec<String>,
    /// Key **expressions**, in order, as SQL text — for a functional/expression index
    /// (`CREATE INDEX … (lower(s))`, `((a + b))`). The SQL layer re-parses and evaluates each
    /// against the row to build the key. Empty for a plain-column index (`columns` is then set).
    /// Exactly one of `columns` / `key_exprs` is non-empty.
    pub key_exprs: Vec<String>,
    /// Partial-index predicate as SQL text (`CREATE INDEX … WHERE <pred>`): only rows for which it
    /// is true are indexed. `None` for a full index. The SQL layer re-parses and evaluates it per
    /// row on the write path.
    pub predicate: Option<String>,
    /// Non-key columns stored alongside the key for index-only scans (`INCLUDE (...)`).
    pub include: Vec<String>,
    /// Access method.
    pub kind: IndexKind,
    /// Whether the index enforces uniqueness of its key.
    pub unique: bool,
}

/// The kind of a table constraint (the set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `PRIMARY KEY` — a unique key, at most one per table.
    PrimaryKey,
    /// `UNIQUE` — a unique key.
    Unique,
    /// `FOREIGN KEY` — references a parent table's key.
    ForeignKey,
    /// `CHECK` — a per-row boolean predicate.
    Check,
}

/// Referential action for a `FOREIGN KEY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FkAction {
    /// `NO ACTION` — reject if dependent rows remain (the default).
    NoAction,
    /// `RESTRICT` — reject if dependent rows remain.
    Restrict,
    /// `CASCADE` — delete/update the dependent rows too.
    Cascade,
    /// `SET NULL` — null the referencing columns (row rewrite → SQL-layer side).
    SetNull,
    /// `SET DEFAULT` — set the referencing columns to their default (row rewrite → SQL-layer side).
    SetDefault,
}

/// A `FOREIGN KEY` constraint definition.
///
/// The child's `child_columns` reference the parent table's `parent_columns`, which must be a
/// `PRIMARY KEY` or `UNIQUE` constraint. The SQL layer encodes the key bytes (for both the parent
/// lookup and the child-side index the engine maintains) consistently with the parent key's encoding
/// — the engine compares opaque bytes, never decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignKeyDef {
    /// Constraint name (unique within the catalog; names the backing child-side index).
    pub name: String,
    /// The referencing (child) table.
    pub child_table: TableId,
    /// The referencing columns, in order.
    pub child_columns: Vec<String>,
    /// The referenced (parent) table — must have a `PRIMARY KEY` / `UNIQUE` constraint.
    pub parent_table: TableId,
    /// The referenced parent columns, in the order they pair with `child_columns`. They must form a
    /// `PRIMARY KEY` or `UNIQUE` constraint on the parent. Empty on input to
    /// [`add_foreign_key`](StorageEngine::add_foreign_key) means "the parent's `PRIMARY KEY`"; the
    /// engine always reports the resolved columns from [`list_foreign_keys`](StorageEngine::list_foreign_keys).
    pub parent_columns: Vec<String>,
    /// Action when a referenced parent row is deleted.
    pub on_delete: FkAction,
    /// Action when a referenced parent key is updated.
    pub on_update: FkAction,
}

/// A declared table constraint, as returned by [`StorageEngine::list_constraints`].
///
/// `PRIMARY KEY`/`UNIQUE`/`FOREIGN KEY` carry a backing [`index`](Constraint::index); `CHECK`
/// carries its predicate [`expr`](Constraint::expr) instead (opaque bytes the SQL layer encodes and
/// evaluates per row). A backed constraint surfaces a violation as [`Error::ConstraintViolation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    /// Constraint name (unique within the catalog; also names the backing index, if any).
    pub name: String,
    /// The table the constraint is declared on.
    pub table: TableId,
    /// The constrained columns, in order (empty for `CHECK`).
    pub columns: Vec<String>,
    /// `PrimaryKey`, `Unique`, `ForeignKey`, or `Check`.
    pub kind: ConstraintKind,
    /// The backing index (PK/UNIQUE/FK), or `None` for `CHECK`.
    pub index: Option<IndexId>,
    /// The `CHECK` predicate as opaque bytes (the SQL layer's encoding), or `None` otherwise.
    pub expr: Option<Vec<u8>>,
}

/// Per-column statistics for the cost-based optimizer.
///
/// Values (`min`/`max`/`most_common`/`histogram`) are **opaque bytes** the SQL layer encoded from
/// the column's values — the engine stores and serves them without decoding (it cannot interpret a
/// tuple). The SQL layer computes these by scanning the table; the engine contributes the
/// authoritative [`row_count`](StorageEngine::row_count).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColumnStats {
    /// Column name.
    pub column: String,
    /// Number of `NULL` values observed.
    pub null_count: u64,
    /// Number of distinct values (NDV).
    pub distinct_count: u64,
    /// Smallest value (encoded), or `None` if no non-null value.
    pub min: Option<Vec<u8>>,
    /// Largest value (encoded), or `None` if no non-null value.
    pub max: Option<Vec<u8>>,
    /// Most-common values as `(value, frequency)` (MCV list).
    pub most_common: Vec<(Vec<u8>, u64)>,
    /// Equi-width histogram bucket boundaries (encoded values), ascending.
    pub histogram: Vec<Vec<u8>>,
}

/// Table-level statistics for the planner, produced by
/// [`analyze_table`](StorageEngine::analyze_table).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableStats {
    /// Live (committed, undeleted) row count at analyze time.
    pub row_count: u64,
    /// Approximate storage pages occupied (best-effort; `0` if unknown for the backend).
    pub page_count: u64,
    /// Per-column statistics.
    pub columns: Vec<ColumnStats>,
}

/// One `ALTER TABLE` schema mutation the engine applies to a table's catalog entry.
///
/// These mutate the *schema* the catalog stores. Row bytes are opaque to the engine, so the SQL
/// layer owns any row-format migration (e.g. defaulting a new column on read). Column **defaults**
/// are not represented here — the core schema ([`ColumnDef`]) has no default field, so they stay in
/// the SQL layer's catalog; **constraints** are altered via
/// [`add_unique_constraint`](StorageEngine::add_unique_constraint) / `drop_constraint`, not here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlterOp {
    /// `ADD COLUMN` — append `column` to the schema (rejected if the name already exists).
    AddColumn(ColumnDef),
    /// `DROP COLUMN` — remove the named column.
    DropColumn {
        /// Column to remove.
        name: String,
    },
    /// `RENAME COLUMN from TO to`.
    RenameColumn {
        /// Existing column name.
        from: String,
        /// New column name.
        to: String,
    },
    /// `RENAME TO name` — rename the table itself.
    RenameTable {
        /// New table name.
        name: String,
    },
    /// `ALTER COLUMN … TYPE ty` — change a column's physical type (the SQL layer checks data
    /// compatibility / rewrites rows; the engine only updates the catalog type).
    AlterColumnType {
        /// Column to retype.
        column: String,
        /// New physical type.
        ty: ColumnType,
    },
    /// `ALTER COLUMN … SET NOT NULL` (the SQL layer first checks no existing NULLs).
    SetNotNull {
        /// Column to mark `NOT NULL`.
        column: String,
    },
    /// `ALTER COLUMN … DROP NOT NULL`.
    DropNotNull {
        /// Column to make nullable.
        column: String,
    },
}

/// A forward-only cursor over the tuples of a table that are visible to a transaction.
///
/// The implementation applies the MVCC visibility filter, so the SQL layer sees only
/// rows visible to its transaction's snapshot.
pub trait TupleScan: Send {
    /// Advance to the next visible tuple, or `Ok(None)` at end of scan.
    ///
    /// The tuple is a [`SharedTuple`] (`Arc<[u8]>`): the scan shares the version store's bytes with a
    /// refcount bump rather than copying them per row (§1.5). Decode it from `&[u8]`.
    fn try_next(&mut self) -> Result<Option<(Tid, SharedTuple)>>;

    /// Advance, also reporting the schema version the tuple was encoded under.
    ///
    /// The default delegates to [`try_next`](TupleScan::try_next) and reports version `0` — the
    /// only version that exists before any `ALTER TABLE` — so existing implementations and callers
    /// are unaffected. The durable engine overrides this to report each row's real schema version,
    /// letting the SQL layer transform a row written under an older schema to the current one on
    /// read (lazy on-read), pairing with [`schema_for_version`](StorageEngine::schema_for_version).
    fn try_next_versioned(&mut self) -> Result<Option<(Tid, u32, SharedTuple)>> {
        Ok(self.try_next()?.map(|(tid, tuple)| (tid, 0, tuple)))
    }
}

/// The top-level port the SQL engine uses to drive the storage + transaction spine.
///
/// One implementation is the production clustered B-link/B+tree engine in `nusadb-btree`
/// (over `nusadb-wal`); another is an in-memory test double in `nusadb-sql`'s tests. Both
/// let each half of the engine be developed and tested independently of the other.
///
/// All mutating methods take a [`TxnId`] obtained from [`begin`](StorageEngine::begin).
/// Passing an id that is not active yields [`Error::UnknownTransaction`].
pub trait StorageEngine: Send + Sync {
    /// Begin a new transaction at `level`, returning its id.
    fn begin(&self, level: IsolationLevel) -> Result<TxnId>;

    /// Begin a new **statement** within `txn`: refresh the `READ COMMITTED` / `READ UNCOMMITTED`
    /// statement snapshot so that every table read in this statement observes ONE consistent view
    /// (standard RC — "one snapshot per statement"). Without it, a statement touching two tables (a
    /// join, a self-join, two scalar subqueries) can read each under a different snapshot and, with
    /// a concurrent commit landing between the reads, return an inconsistent result — and, worse, an
    /// `UPDATE` whose row-set was pinned to a stale snapshot silently skips rows a fresh snapshot
    /// would match. The SQL layer calls this once at the start of each statement executed within an
    /// existing transaction, at the execution choke-point every protocol (simple- and extended-query)
    /// passes through.
    ///
    /// This method is **required** — deliberately not defaulted — so an engine cannot silently opt
    /// out of statement snapshots. Engines with a frozen snapshot for `REPEATABLE READ` /
    /// `SERIALIZABLE`, and simple engines / test doubles with no MVCC snapshots, implement it as an
    /// explicit `Ok(())`.
    fn begin_statement(&self, txn: TxnId) -> Result<()>;

    /// The isolation level `txn` runs at, or `None` if the engine does not track it (or `txn` is
    /// unknown). The SQL layer consults this to avoid a secondary-index scan under a frozen
    /// snapshot (`REPEATABLE READ` / `SERIALIZABLE`): a single-version index can point at row
    /// versions that disagree with the transaction's MVCC-visible view, so an index scan would
    /// return wrong rows where a sequential scan is correct. The default returns `None`,
    /// suiting simple engines and test doubles that have no snapshot isolation.
    fn txn_isolation(&self, txn: TxnId) -> Option<IsolationLevel> {
        let _ = txn;
        None
    }

    /// Commit `txn`, making its writes durable and visible to later snapshots.
    ///
    /// # Contract on failure
    /// On a **recoverable** failure (e.g. a WAL append that hit `ENOSPC`, or a serializable conflict
    /// detected at commit time), the engine rolls `txn` back before returning `Err` — releasing its
    /// locks and discarding its writes — as defence in depth, so a forgotten caller rollback cannot
    /// strand the transaction in the active set with its locks held. An **unrecoverable durability**
    /// failure (a WAL write/`fsync` that may have partially reached disk) instead stops the process,
    /// so recovery replays the durable prefix honestly on restart rather than reporting a false
    /// failure for a commit that survives. Callers should still issue a best-effort
    /// [`rollback`](StorageEngine::rollback) on `Err` regardless — it resolves to a harmless no-op
    /// `UnknownTransaction` when the engine already cleaned up.
    fn commit(&self, txn: TxnId) -> Result<()>;

    /// Abort `txn`, discarding all of its writes.
    fn rollback(&self, txn: TxnId) -> Result<()>;

    /// Establish a named savepoint within `txn`.
    fn savepoint(&self, txn: TxnId, name: &str) -> Result<()>;

    /// Roll `txn` back to a previously established savepoint, discarding later writes.
    fn rollback_to(&self, txn: TxnId, name: &str) -> Result<()>;

    /// Release (discard) a previously established savepoint within `txn`, along with any
    /// savepoints established after it. Unlike [`rollback_to`](Self::rollback_to) this keeps all
    /// of the transaction's writes — it only forgets the savepoint markers, so a later
    /// `ROLLBACK TO` that name fails.
    ///
    /// The default is a no-op success, which suits engines (and test doubles) that do not track
    /// savepoint markers; the durable engine overrides it to validate the name and prune the
    /// marker stack.
    fn release_savepoint(&self, txn: TxnId, name: &str) -> Result<()> {
        let _ = (txn, name);
        Ok(())
    }

    /// Create a table from `def`, returning its assigned [`TableId`]. (DDL)
    fn create_table(&self, txn: TxnId, def: &TableDef) -> Result<TableId>;

    /// Drop a table and all of its data. (DDL)
    fn drop_table(&self, txn: TxnId, table: TableId) -> Result<()>;

    /// Resolve a table name to its schema, or `Ok(None)` if it does not exist. The
    /// non-transactional (latest-committed) view — prefer [`lookup_table_as_of`](
    /// Self::lookup_table_as_of) when a transaction context is available so resolution respects
    /// per-transaction schema visibility.
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>>;

    /// The names of all currently-visible (committed, not committed-dropped) tables, sorted. Used by
    /// catalog introspection (`SHOW TABLES`, the CLI's `\dt`). The default returns an empty list,
    /// suiting engines that expose no catalog listing; the durable engine overrides it.
    fn list_tables(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// The names of all tables visible **to `txn`'s snapshot**, sorted — the transactional
    /// counterpart of [`list_tables`](Self::list_tables). A table created by an uncommitted *other*
    /// transaction is hidden; one the calling transaction created (even before commit) is listed.
    ///
    /// The default ignores `txn` and falls back to [`list_tables`](Self::list_tables) — fine for
    /// in-memory test doubles. The MVCC engine overrides it: enumerating under the transaction's own
    /// snapshot avoids the visibility race in `list_tables`, whose latest-committed view can lag a
    /// just-committed prior statement on the same connection (so a transaction that began *after* that
    /// commit could otherwise see an empty catalog).
    fn list_tables_as_of(&self, txn: TxnId) -> Result<Vec<String>> {
        let _ = txn;
        self.list_tables()
    }

    /// A monotonic counter that increases whenever committed data changes, for result caching:
    /// a cached result computed at version `v` is still valid iff `data_version()` is still
    /// `v`. `None` means the engine does not track a version, so callers must not cache (the default,
    /// which keeps a result cache safely *disabled* rather than risk serving stale rows). The durable
    /// engine returns `Some(counter)` and bumps it on every committed write/DDL.
    fn data_version(&self) -> Option<u64> {
        None
    }

    /// Resolve a table name to its schema **as visible to `txn`'s snapshot**, or `Ok(None)`
    /// if no such table is visible to it. A table created by an uncommitted *other* transaction is
    /// not visible; one the calling transaction created (even before commit) is; one dropped by an
    /// uncommitted other transaction is still visible until that drop commits.
    ///
    /// The default implementation ignores `txn` and falls back to [`lookup_table`](
    /// Self::lookup_table) (latest-committed) — correct for single-transaction callers and
    /// in-memory test doubles; the MVCC-aware engine overrides it.
    fn lookup_table_as_of(&self, txn: TxnId, name: &str) -> Result<Option<TableSchema>> {
        let _ = txn;
        self.lookup_table(name)
    }

    /// Resolve `(schema, name)` to its table in the latest-committed view — the
    /// schema-qualified counterpart of [`lookup_table`](Self::lookup_table). A bare name resolves
    /// through the [`PUBLIC_SCHEMA`] namespace.
    ///
    /// The default resolves only the public namespace (delegating to `lookup_table`) and reports
    /// `Ok(None)` for any other schema — correct for in-memory test doubles that have a single
    /// namespace; the durable engine overrides it to resolve the real `(schema, name)` catalog key.
    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>> {
        if schema == PUBLIC_SCHEMA {
            self.lookup_table(name)
        } else {
            Ok(None)
        }
    }

    /// Resolve `(schema, name)` as visible to `txn`'s snapshot — the schema-qualified
    /// counterpart of [`lookup_table_as_of`](Self::lookup_table_as_of). A bare name resolves through
    /// the [`PUBLIC_SCHEMA`] namespace.
    ///
    /// The default resolves only the public namespace (delegating to `lookup_table_as_of`) and
    /// reports `Ok(None)` for any other schema; the MVCC engine overrides it.
    fn lookup_table_as_of_in(
        &self,
        txn: TxnId,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>> {
        if schema == PUBLIC_SCHEMA {
            self.lookup_table_as_of(txn, name)
        } else {
            let _ = txn;
            Ok(None)
        }
    }

    /// Insert `tuple` into `table`, returning the new row's [`Tid`].
    fn insert(&self, txn: TxnId, table: TableId, tuple: &[u8]) -> Result<Tid>;

    /// Replace the tuple at `tid`. Under MVCC this writes a new version and returns its
    /// new [`Tid`].
    fn update(&self, txn: TxnId, table: TableId, tid: Tid, tuple: &[u8]) -> Result<Tid>;

    /// Delete the tuple at `tid` (stamps it with `txn`'s `xmax`).
    fn delete(&self, txn: TxnId, table: TableId, tid: Tid) -> Result<()>;

    /// Open a sequential scan over `table`, yielding rows visible to `txn`'s snapshot.
    fn scan(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>>;

    /// Open a scan for a uniqueness / `PRIMARY KEY` constraint check: it yields every row committed as
    /// of now, plus `txn`'s own uncommitted writes — *not* `txn`'s frozen snapshot. Under REPEATABLE
    /// READ / SERIALIZABLE a snapshot scan would miss a row another transaction committed after `txn`
    /// began, letting a duplicate key commit; a constraint must instead see the latest committed state
    /// (uniqueness is not snapshot-isolated). The default delegates to [`scan`](Self::scan) — correct
    /// for engines that do not freeze a snapshot (e.g. the in-memory test double).
    fn scan_committed(&self, txn: TxnId, table: TableId) -> Result<Box<dyn TupleScan>> {
        self.scan(txn, table)
    }

    /// Acquire an explicit row lock on `tid` for `txn` — the engine side of `SELECT ... FOR SHARE`
    /// / `FOR UPDATE`. The lock is held until `txn` commits or rolls back. Under the
    /// engine's no-wait policy a conflict returns
    /// [`Error::SerializationConflict`] rather than blocking
    /// (the SQL layer maps `NOWAIT` onto this and may retry for the default blocking behaviour).
    ///
    /// The default is a no-op returning `Ok(())`, so an engine without locking — and the SQL
    /// layer's in-memory test double — need not implement it. The production engine
    /// overrides this to take a real row-level lock.
    fn lock_row(&self, txn: TxnId, table: TableId, tid: Tid, mode: RowLockMode) -> Result<()> {
        let _ = (txn, table, tid, mode);
        Ok(())
    }

    /// Acquire a table-level lock on `table` for `txn` — the engine side of `LOCK TABLE`.
    /// Held until commit/rollback. `AccessExclusive` conflicts with any concurrent row write/lock on
    /// the table (multi-granularity); under the no-wait policy a conflict returns
    /// [`Error::SerializationConflict`] rather than blocking.
    ///
    /// The default is a no-op returning `Ok(())`, so an engine without locking — and the SQL
    /// layer's in-memory test double — need not implement it. The production engine overrides it.
    fn lock_table(&self, txn: TxnId, table: TableId, mode: TableLockMode) -> Result<()> {
        let _ = (txn, table, mode);
        Ok(())
    }

    /// Acquire a **key-level** lock for `txn` — the seam that makes `UNIQUE`/`PRIMARY KEY`
    /// enforcement atomic under concurrency. `key_hash` is a stable hash of a logical key
    /// (the constraint's identity + the key's values); two transactions that touch the same key take
    /// the same lock, so a snapshot-based uniqueness check cannot admit two concurrent inserts of the
    /// same value. Held until commit/rollback. Under the engine's no-wait policy a conflict returns
    /// [`Error::SerializationConflict`] rather than blocking — so of two racing same-key writers the
    /// second aborts at lock time (before its scan), regardless of isolation level.
    ///
    /// The default is a no-op returning `Ok(())`, so an engine without locking — and the SQL layer's
    /// in-memory test double — need not serialize. The production engine overrides it
    /// to take a real lock in a key namespace distinct from row/table locks.
    fn lock_key(&self, txn: TxnId, table: TableId, key_hash: u64, mode: RowLockMode) -> Result<()> {
        let _ = (txn, table, key_hash, mode);
        Ok(())
    }

    /// Create a sequence from `def`, returning its [`SequenceId`].
    ///
    /// **Non-transactional DDL** (gap semantics): the create is durable immediately and is *not*
    /// undone by a rollback — consistent with a sequence's value, which also advances
    /// non-transactionally. The default returns an `Unsupported` error; the production engine
    /// overrides it. (A no-op default would silently swallow `CREATE SEQUENCE`.)
    fn create_sequence(&self, txn: TxnId, def: &SequenceDef) -> Result<SequenceId> {
        let _ = (txn, def);
        Err(unsupported("create_sequence"))
    }

    /// Drop a sequence. **Non-transactional DDL**: durable immediately, not undone by a
    /// rollback.
    fn drop_sequence(&self, txn: TxnId, id: SequenceId) -> Result<()> {
        let _ = (txn, id);
        Err(unsupported("drop_sequence"))
    }

    /// Resolve a sequence name to its id, or `Ok(None)` if it does not exist. Default
    /// `Ok(None)` so the in-memory test double need not implement sequences.
    fn lookup_sequence(&self, name: &str) -> Result<Option<SequenceId>> {
        let _ = name;
        Ok(None)
    }

    /// Advance the sequence and return the new value — `nextval`. **Non-transactional**:
    /// the advance is durable immediately and is not undone by a rollback.
    fn sequence_next(&self, id: SequenceId) -> Result<i64> {
        let _ = id;
        Err(unsupported("sequence_next"))
    }

    /// Return the sequence's most recently handed-out value — `currval`.
    fn sequence_current(&self, id: SequenceId) -> Result<i64> {
        let _ = id;
        Err(unsupported("sequence_current"))
    }

    /// Set the sequence's current value — `setval`. The next `sequence_next` returns
    /// `value + increment`.
    fn sequence_set(&self, id: SequenceId, value: i64) -> Result<()> {
        let _ = (id, value);
        Err(unsupported("sequence_set"))
    }

    /// Create a SQL schema (namespace), returning its [`SchemaId`]. Rollback-aware DDL.
    ///
    /// This registers a namespace; resolving unqualified names against a `search_path` is the SQL
    /// layer's session concern. The default returns an `Unsupported` error; the production engine
    /// overrides it.
    fn create_schema(&self, txn: TxnId, name: &str) -> Result<SchemaId> {
        let _ = (txn, name);
        Err(unsupported("create_schema"))
    }

    /// Drop a SQL schema (namespace). Rollback-aware DDL. With `cascade`, every table in the
    /// schema is dropped too; without it (`RESTRICT`, the default), dropping a non-empty schema
    /// is an error.
    fn drop_schema(&self, txn: TxnId, id: SchemaId, cascade: bool) -> Result<()> {
        let _ = (txn, id, cascade);
        Err(unsupported("drop_schema"))
    }

    /// Resolve a schema (namespace) name to its id, or `Ok(None)` if it does not exist.
    /// Default `Ok(None)` so the in-memory test double need not implement schemas.
    fn lookup_schema(&self, name: &str) -> Result<Option<SchemaId>> {
        let _ = name;
        Ok(None)
    }

    /// List every registered schema (namespace) as `(id, name)`.
    ///
    /// Backs `information_schema.schemata` / `\dn` on the SQL side. The order is unspecified; the
    /// caller sorts if it needs a stable presentation. Default `Ok(vec![])` so the in-memory test
    /// double need not implement schemas.
    fn list_schemas(&self) -> Result<Vec<(SchemaId, String)>> {
        Ok(Vec::new())
    }

    /// Register a secondary index from `def`, returning its [`IndexId`].
    ///
    /// The engine only records the index; **the SQL layer populates it** by computing each row's
    /// key bytes and calling [`index_insert`](StorageEngine::index_insert) (e.g. backfilling via a
    /// [`scan`](StorageEngine::scan) in the `CREATE INDEX` executor). Rollback-aware DDL. The default
    /// returns `Unsupported`; the production engine overrides it.
    fn create_index(&self, txn: TxnId, def: &IndexDef) -> Result<IndexId> {
        let _ = (txn, def);
        Err(unsupported("create_index"))
    }

    /// Drop a secondary index and discard its entries. Rollback-aware DDL.
    fn drop_index(&self, txn: TxnId, id: IndexId) -> Result<()> {
        let _ = (txn, id);
        Err(unsupported("drop_index"))
    }

    /// Resolve an index name to its id, or `Ok(None)` if it does not exist. Default `Ok(None)`.
    fn lookup_index(&self, name: &str) -> Result<Option<IndexId>> {
        let _ = name;
        Ok(None)
    }

    /// List the indexes defined on `table` (for the planner). Default empty.
    fn list_indexes(&self, table: TableId) -> Result<Vec<IndexDef>> {
        let _ = table;
        Ok(Vec::new())
    }

    /// Whether `index` **covers** its table — every live row has an entry — so an index scan is a
    /// safe superset of a sequential scan. The planner catalogs consult this and only offer
    /// complete indexes as scan candidates; an incomplete one (e.g. recovered from a data dir
    /// written by a build that did not maintain constraint-backing entries) stays catalog-visible
    /// but is never scanned, since its missing entries would silently drop rows. Default `true`:
    /// an engine that does not track coverage promises it by maintaining every index on every
    /// write.
    fn index_is_complete(&self, index: IndexId) -> Result<bool> {
        let _ = index;
        Ok(true)
    }

    /// Add an index entry mapping the opaque `key` bytes to the row `tid`.
    ///
    /// Called by the SQL layer's `INSERT`/`UPDATE` executors after it encodes the key. For a
    /// `unique` index, a duplicate key yields [`Error::ConstraintViolation`] — **except** on a
    /// constraint-backing index: there the SQL layer's scan-based checks + key locks own
    /// the constraint semantics (NULL keys never conflict; a statement may pass through a
    /// transient duplicate), so the entries are maintained purely as a lookup structure and the
    /// byte-level check is skipped. Buffered with the transaction: undone on rollback, durable on
    /// commit. Default returns `Unsupported`.
    fn index_insert(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        let _ = (txn, index, key, tid);
        Err(unsupported("index_insert"))
    }

    /// Add many `(key, tid)` entries to one index in a single call, for a bulk load (`COPY` or a
    /// large `INSERT ... SELECT`) that would otherwise call [`index_insert`](StorageEngine::index_insert)
    /// once per row per index.
    ///
    /// Semantically identical to calling `index_insert` for each entry in turn: every entry is
    /// added, a `unique` index rejects a duplicate key with [`Error::ConstraintViolation`], and the
    /// whole batch is buffered with the transaction — undone on rollback, durable on commit — so a
    /// crash mid-batch leaves the transaction fully indexed or not at all, exactly as the per-row
    /// path does. The engine MAY apply the entries in any order: a bulk load presents them in row
    /// order, but applying them in key order turns the random index writes into sequential ones,
    /// and neither the final index state nor the uniqueness outcome depends on order. The default
    /// applies them as given via `index_insert`; the production engine overrides it to sort first.
    fn index_insert_batch(
        &self,
        txn: TxnId,
        index: IndexId,
        entries: Vec<(Vec<u8>, Tid)>,
    ) -> Result<()> {
        for (key, tid) in entries {
            self.index_insert(txn, index, &key, tid)?;
        }
        Ok(())
    }

    /// Remove the index entry mapping `key` to `tid`. Buffered with the transaction.
    /// Default returns `Unsupported`.
    fn index_delete(&self, txn: TxnId, index: IndexId, key: &[u8], tid: Tid) -> Result<()> {
        let _ = (txn, index, key, tid);
        Err(unsupported("index_delete"))
    }

    /// Scan the index over the key range `[lo, hi]`, yielding the **visible** rows whose keys fall
    /// in range, in ascending key order.
    ///
    /// Bounds are over the opaque key bytes ([`Bound::Unbounded`] for an open end). The engine
    /// looks each entry's `tid` up in the row store and applies the usual MVCC visibility filter, so
    /// a caller sees only rows visible to its transaction's snapshot. Default returns `Unsupported`.
    fn index_scan(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        let _ = (txn, index, lo, hi);
        Err(unsupported("index_scan"))
    }

    /// Like [`index_scan`](Self::index_scan), but with **latest-committed** visibility (plus `txn`'s
    /// own writes) rather than `txn`'s frozen snapshot — the index-probe counterpart of
    /// [`scan_committed`](Self::scan_committed). It lets a uniqueness check probe a backing index in
    /// O(log n) while still seeing a key another transaction committed after a REPEATABLE READ /
    /// SERIALIZABLE `txn` began (uniqueness is not snapshot-isolated). No SERIALIZABLE read tracking
    /// (a system constraint probe, not a user observation). The default delegates to
    /// [`index_scan`](Self::index_scan) — correct for engines that do not freeze a snapshot (e.g. the
    /// in-memory test double).
    fn index_scan_committed(
        &self,
        txn: TxnId,
        index: IndexId,
        lo: Bound<Vec<u8>>,
        hi: Bound<Vec<u8>>,
    ) -> Result<Box<dyn TupleScan>> {
        self.index_scan(txn, index, lo, hi)
    }

    /// Declare a `UNIQUE` / `PRIMARY KEY` constraint on `table`, enforced by a backing unique index
    /// the engine creates (named after the constraint); returns that [`IndexId`].
    ///
    /// The SQL layer maintains the backing index with [`index_insert`](StorageEngine::index_insert)
    /// (encoding the key from `columns`), so a duplicate yields [`Error::ConstraintViolation`]. With
    /// `primary = true` the engine also enforces **at most one primary key per table**. Rollback-
    /// aware DDL. Default returns `Unsupported`; the production engine overrides it.
    fn add_unique_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        columns: &[String],
        primary: bool,
    ) -> Result<IndexId> {
        let _ = (txn, table, name, columns, primary);
        Err(unsupported("add_unique_constraint"))
    }

    /// Declare a `CHECK` constraint on `table` carrying the predicate `expr`.
    ///
    /// `expr` is **opaque bytes** the SQL layer encodes; the engine stores/recovers it but does not
    /// evaluate it (it never decodes a tuple). The SQL layer **evaluates the predicate per row** on
    /// `INSERT`/`UPDATE` (decoding `expr` + the row) and raises [`Error::ConstraintViolation`] on a
    /// false result. The engine catalogs it (rollback-aware DDL) so the predicate survives restart
    /// and is visible to [`list_constraints`](StorageEngine::list_constraints). Default returns
    /// `Unsupported`; the production engine overrides it.
    fn add_check_constraint(
        &self,
        txn: TxnId,
        table: TableId,
        name: &str,
        expr: &[u8],
    ) -> Result<()> {
        let _ = (txn, table, name, expr);
        Err(unsupported("add_check_constraint"))
    }

    /// Drop a named constraint and its backing index. Rollback-aware DDL.
    fn drop_constraint(&self, txn: TxnId, table: TableId, name: &str) -> Result<()> {
        let _ = (txn, table, name);
        Err(unsupported("drop_constraint"))
    }

    /// List the constraints declared on `table` (for the planner / introspection). Default empty.
    fn list_constraints(&self, table: TableId) -> Result<Vec<Constraint>> {
        let _ = table;
        Ok(Vec::new())
    }

    /// Whether `table` has any `PRIMARY KEY` / `UNIQUE` constraint — the cheap existence check the
    /// SQL layer makes before every INSERT/UPDATE (to decide whether a uniqueness scan is needed at
    /// all), without materialising and cloning the whole constraint list. The default derives the
    /// answer from [`list_constraints`](StorageEngine::list_constraints); the production engine
    /// overrides it to answer from the catalog directly, avoiding the per-write clone.
    fn has_unique_constraint(&self, table: TableId) -> Result<bool> {
        Ok(self
            .list_constraints(table)?
            .iter()
            .any(|c| matches!(c.kind, ConstraintKind::PrimaryKey | ConstraintKind::Unique)))
    }

    /// List every `FOREIGN KEY` in the catalog whose child *or* parent is `table`.
    ///
    /// The SQL layer needs the full [`ForeignKeyDef`] (parent table + actions, which
    /// [`list_constraints`](StorageEngine::list_constraints) does not carry) to enforce referential
    /// integrity end-to-end: a child `INSERT`/`UPDATE` checks its own FKs reference an existing
    /// parent row, and a parent `DELETE`/`UPDATE` checks/propagates the FKs that point at it.
    /// Default empty (the in-memory test double need not implement FKs).
    fn list_foreign_keys(&self, table: TableId) -> Result<Vec<ForeignKeyDef>> {
        let _ = table;
        Ok(Vec::new())
    }

    /// Declare a `FOREIGN KEY` on `def.child_table` referencing `def.parent_table`.
    ///
    /// The engine validates the parent has a `PRIMARY KEY` / `UNIQUE` to reference, then creates a
    /// (non-unique) child-side index over `def.child_columns` and returns its [`IndexId`]; the SQL
    /// layer maintains it with [`index_insert`](StorageEngine::index_insert)/`index_delete` on child
    /// writes (encoding the FK key the same way as the parent key). Rollback-aware DDL. Default
    /// returns `Unsupported`; the production engine overrides it.
    fn add_foreign_key(&self, txn: TxnId, def: &ForeignKeyDef) -> Result<IndexId> {
        let _ = (txn, def);
        Err(unsupported("add_foreign_key"))
    }

    /// Verify that a child row's foreign-key `key` references an existing parent row.
    ///
    /// The SQL layer calls this on a child `INSERT`/`UPDATE` (skipping a `NULL` key) with the
    /// encoded FK key; the engine looks it up in the parent's unique index and returns
    /// [`Error::ConstraintViolation`] if absent. Default `Ok(())` (no enforcement — the in-memory
    /// test double need not implement FKs).
    fn fk_check(&self, txn: TxnId, name: &str, key: &[u8]) -> Result<()> {
        let _ = (txn, name, key);
        Ok(())
    }

    /// Enforce foreign keys when a parent row with `parent_key` is about to be deleted.
    ///
    /// For every FK referencing `parent_table` with a matching child row: `CASCADE` deletes the
    /// dependent child rows (one level), `RESTRICT`/`NO ACTION` returns
    /// [`Error::ConstraintViolation`]. (`SET NULL`/`SET DEFAULT` need a row rewrite and are the SQL
    /// layer's responsibility.) Returns the number of child rows cascade-deleted. The SQL layer
    /// calls this *before* deleting the parent. Default `Ok(0)`.
    fn fk_on_delete(&self, txn: TxnId, parent_table: TableId, parent_key: &[u8]) -> Result<u64> {
        let _ = (txn, parent_table, parent_key);
        Ok(0)
    }

    /// Store the statistics computed for `table` by `ANALYZE`.
    ///
    /// The SQL layer computes the per-column stats (it owns row decoding) and calls this to persist
    /// them in the catalog for the planner; the engine fills/serves the
    /// [`row_count`](StorageEngine::row_count) authoritatively. Rollback-aware; durable on commit.
    /// Default returns `Unsupported`; the production engine overrides it.
    fn analyze_table(&self, txn: TxnId, table: TableId, stats: &TableStats) -> Result<()> {
        let _ = (txn, table, stats);
        Err(unsupported("analyze_table"))
    }

    /// The statistics most recently stored for `table`, or `Ok(None)` if never analyzed.
    /// Default `Ok(None)`.
    fn table_stats(&self, table: TableId) -> Result<Option<TableStats>> {
        let _ = table;
        Ok(None)
    }

    /// The live (committed, undeleted) row count of `table` — a cheap cardinality estimate the
    /// engine computes directly, even without a prior `ANALYZE`. Default `Ok(0)`.
    fn row_count(&self, table: TableId) -> Result<u64> {
        let _ = table;
        Ok(0)
    }

    /// An **`O(1)` approximate** row count of `table`, safe to call at plan time on every query —
    /// unlike [`row_count`](Self::row_count), which walks the table. It is a routing hint only (e.g.
    /// whether a scan is large enough to vectorize), never a correctness input, so an engine that
    /// maintains a running estimate may return a slightly stale value (a cached planner cardinality,
    /// not an exact count). Default `Ok(0)` — an engine without a cheap estimate declines to hint,
    /// and the caller falls back to its other cardinality sources.
    fn approx_row_count(&self, table: TableId) -> Result<u64> {
        let _ = table;
        Ok(0)
    }

    /// The write churn (`inserts + updates + deletes` committed) `table` has accumulated since its
    /// statistics were last refreshed by [`analyze_table`](Self::analyze_table) -- the staleness
    /// signal an auto-analyze policy consults to decide when the planner's histogram/MCV statistics
    /// have aged and the table is worth re-analysing. Absolute, not net: a heavily churned table that
    /// kept its row count still ages its statistics. A hint only, never a correctness input, so a
    /// slightly stale value is fine. Default `Ok(0)` -- an engine that keeps no such tally reports no
    /// churn, so the policy simply never fires for it.
    fn churn_since_analyze(&self, table: TableId) -> Result<u64> {
        let _ = table;
        Ok(0)
    }

    /// Apply one `ALTER TABLE` schema mutation to `table`.
    ///
    /// Mutates the table's catalog schema (columns / name) — rollback-aware DDL, durable on commit,
    /// recovered on restart. Row bytes stay opaque, so the SQL layer handles any row-format
    /// migration. Default returns `Unsupported`; the production engine overrides it.
    fn alter_table(&self, txn: TxnId, table: TableId, op: &AlterOp) -> Result<()> {
        let _ = (txn, table, op);
        Err(unsupported("alter_table"))
    }

    /// The schema `table` had at schema version `version`, or `None` if the table
    /// or version is unknown.
    ///
    /// `ALTER TABLE` advances a table's schema version; every row version remembers the version
    /// it was encoded under (see [`TupleScan::try_next_versioned`]). The SQL layer reads the
    /// historical schema for a row's version to decode it correctly and transform it to the
    /// current schema on read (lazy on-read). The default returns `Ok(None)` (engines
    /// without schema-version history); the production engine overrides it.
    fn schema_for_version(&self, table: TableId, version: u32) -> Result<Option<TableSchema>> {
        let _ = (table, version);
        Ok(None)
    }

    /// The current (latest) schema version of `table`, or `None` if the table is
    /// unknown or the engine does not track versions.
    ///
    /// A row whose version is below this was written under an older schema and needs transforming
    /// to the current one on read. The default returns `Ok(None)`; the production engine overrides
    /// it.
    fn current_schema_version(&self, table: TableId) -> Result<Option<u32>> {
        let _ = table;
        Ok(None)
    }

    /// Reclaim row versions no longer visible to any live or future transaction (MVCC
    /// garbage collection), returning how many versions were reclaimed.
    ///
    /// The default is a no-op returning `Ok(0)`, so an engine without reclamation — and
    /// the SQL layer's in-memory test double — need not implement it. The production
    /// engine overrides this to perform real MVCC vacuuming (the btree engine maps it to a purge pass).
    fn vacuum(&self) -> Result<usize> {
        Ok(0)
    }
}

/// The error returned by the default (unimplemented) treaty methods, e.g. an engine that does not
/// support sequences. Carries the operation name for diagnostics.
fn unsupported(op: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{op} is not implemented by this StorageEngine"),
    ))
}
