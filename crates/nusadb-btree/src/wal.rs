//! Durable redo WAL for the clustered engine (phase 1: logical redo + full replay).
//!
//! **Ordering (phase-1 honesty):** each mutating operation is applied to the in-memory structure
//! **first** and its log record appended **after** (the engine's per-object write latch keeps the
//! two adjacent and, per object, in the same order as replay). This is the *opposite* of
//! write-ahead ordering, and it is safe **only because the page store is volatile in phase 1** —
//! the WAL is the sole durable medium, so no unlogged change can ever reach durable storage ahead
//! of its record, and only the log's own commit ordering matters. **This must become true
//! write-ahead (log-before-apply) before the disk-backed page store lands** — once pages are
//! flushed, an apply-before-log window is a real durability hole. `COMMIT` appends its marker and
//! then **fsyncs** — that fsync is the durability point. Recovery replays the log in two passes:
//! pass 1 collects the committed transaction set (a `CommitTxn` marker in the durable prefix),
//! pass 2 re-applies, in log order, only the operations of committed transactions. Uncommitted
//! tails need no undo — their operations are simply never replayed — and a partial `ROLLBACK TO
//! SAVEPOINT` inside a later-committed transaction is logged as **compensation operations** (the
//! logical inverses), so replay converges to exactly the post-rollback state.
//!
//! Framing, CRC32, lz4, and torn-tail detection come from the shared [`nusadb_wal`] codec: a
//! torn or corrupt trailing record cleanly ends the durable prefix, and the file is truncated to
//! that prefix before new records are appended (so later records are never stranded behind
//! garbage). Operations ride in [`WalRecord::Put`] with a self-describing key
//! (`[tag][txn][table][row_id]`), keeping the shared record enum untouched.
//!
//! **Phase honesty:** this is *redo-only logical* WAL with full-log replay — recovery cost grows
//! with history until checkpoints land (a later phase: fuzzy checkpoint + page-store persistence +
//! double-write, which is also when replay stops being "from the beginning"). Crash safety —
//! the gate — is complete: `kill -9` at any point recovers every committed transaction and
//! none of an uncommitted one.

use nusadb_core::engine::{
    ArrayElem, ColumnStats, FkAction, IndexDef, IndexKind, SequenceDef, TableDef, TableStats,
};
use nusadb_core::{ColumnDef, ColumnType, TableId};
use nusadb_wal::WalRecord;

const TAG_INSERT: u8 = 0;
const TAG_UPDATE: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_CREATE_TABLE: u8 = 3;
const TAG_DROP_TABLE: u8 = 4;
const TAG_CREATE_INDEX: u8 = 5;
const TAG_DROP_INDEX: u8 = 6;
const TAG_INDEX_INSERT: u8 = 7;
const TAG_INDEX_DELETE: u8 = 8;
const TAG_ADD_UNIQUE: u8 = 9;
const TAG_ADD_CHECK: u8 = 10;
const TAG_ADD_FK: u8 = 11;
const TAG_DROP_CONSTRAINT: u8 = 12;
const TAG_SET_STATS: u8 = 13;
const TAG_CLEAR_STATS: u8 = 14;
const TAG_SEQ_CREATE: u8 = 15;
const TAG_SEQ_DROP: u8 = 16;
const TAG_SEQ_SET: u8 = 17;
const TAG_ALTER_SCHEMA: u8 = 18;
const TAG_SCHEMA_CREATE: u8 = 19;
const TAG_SCHEMA_DROP: u8 = 20;
const TAG_INDEX_UNSTAMP: u8 = 21;

/// One logical, replayable operation of a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoggedOp {
    /// Row `row_id` of `table` was created with `tuple`.
    Insert {
        /// Owning transaction.
        txn: u64,
        /// Target table id.
        table: u64,
        /// Engine-minted row id (the Tid).
        row_id: u64,
        /// The opaque tuple bytes.
        tuple: Vec<u8>,
    },
    /// Row `row_id` of `table` now holds `tuple`.
    Update {
        /// Owning transaction.
        txn: u64,
        /// Target table id.
        table: u64,
        /// Engine-minted row id (the Tid).
        row_id: u64,
        /// The new opaque tuple bytes.
        tuple: Vec<u8>,
    },
    /// Row `row_id` of `table` was deleted.
    Delete {
        /// Owning transaction.
        txn: u64,
        /// Target table id.
        table: u64,
        /// Engine-minted row id (the Tid).
        row_id: u64,
    },
    /// Table `table` was created with `def`.
    CreateTable {
        /// Owning transaction.
        txn: u64,
        /// The id the engine assigned.
        table: u64,
        /// The table definition.
        def: TableDef,
    },
    /// Table `table` was dropped.
    DropTable {
        /// Owning transaction.
        txn: u64,
        /// The dropped table id.
        table: u64,
    },
    /// Secondary index `index` was registered with `def`.
    CreateIndex {
        /// Owning transaction.
        txn: u64,
        /// The id the engine assigned.
        index: u64,
        /// The index definition.
        def: IndexDef,
    },
    /// Secondary index `index` was dropped.
    DropIndex {
        /// Owning transaction.
        txn: u64,
        /// The dropped index id.
        index: u64,
    },
    /// Entry `key → row_id` was added to `index`. The key bytes ride in the value slot
    /// (they are opaque and variable-length).
    IndexInsert {
        /// Owning transaction.
        txn: u64,
        /// Target index id.
        index: u64,
        /// The row the entry points at.
        row_id: u64,
        /// The opaque key bytes the SQL layer encoded.
        key: Vec<u8>,
    },
    /// Entry `key → row_id` was removed from `index`.
    IndexDelete {
        /// Owning transaction.
        txn: u64,
        /// Target index id.
        index: u64,
        /// The row the entry pointed at.
        row_id: u64,
        /// The opaque key bytes of the removed entry.
        key: Vec<u8>,
    },
    /// Entry `key → row_id` of `index` had its dead-stamp cleared (the compensation of the
    /// stamp an `IndexInsert` under a new key applied to the row's previous alive entry, emitted
    /// by a savepoint rollback so replay converges).
    IndexUnstamp {
        /// Owning transaction.
        txn: u64,
        /// Target index id.
        index: u64,
        /// The row the entry points at.
        row_id: u64,
        /// The opaque key bytes of the entry to revive.
        key: Vec<u8>,
    },
    /// A `PRIMARY KEY` / `UNIQUE` constraint was declared on `table`, backed by `index` (whose
    /// own `CreateIndex` record precedes this one in the log).
    AddUnique {
        /// Owning transaction.
        txn: u64,
        /// The constrained table.
        table: u64,
        /// The backing unique index.
        index: u64,
        /// Constraint name (also the backing index name).
        name: String,
        /// The constrained columns, in order.
        columns: Vec<String>,
        /// Whether this is the table's `PRIMARY KEY`.
        primary: bool,
    },
    /// A `CHECK` constraint was declared on `table` (predicate = opaque SQL-layer bytes).
    AddCheck {
        /// Owning transaction.
        txn: u64,
        /// The constrained table.
        table: u64,
        /// Constraint name.
        name: String,
        /// The opaque predicate bytes.
        expr: Vec<u8>,
    },
    /// A `FOREIGN KEY` was declared (child-side `CreateIndex` record precedes this one).
    AddFk {
        /// Owning transaction.
        txn: u64,
        /// Constraint name (global within the catalog).
        name: String,
        /// The referencing table.
        child_table: u64,
        /// The referencing columns, in order.
        child_columns: Vec<String>,
        /// The referenced table.
        parent_table: u64,
        /// The parent's backing unique index the FK resolves against.
        parent_index: u64,
        /// The child-side (non-unique) index the engine maintains.
        child_index: u64,
        /// Action on parent delete.
        on_delete: FkAction,
        /// Action on parent key update.
        on_update: FkAction,
    },
    /// The named constraint (unique / check / foreign key) was dropped from `table`'s catalog
    /// records; a backing index's own `DropIndex` record follows separately.
    DropConstraint {
        /// Owning transaction.
        txn: u64,
        /// The table the constraint was declared on (the child table for a FK).
        table: u64,
        /// Constraint name.
        name: String,
    },
    /// `ANALYZE` stored statistics for `table`.
    SetStats {
        /// Owning transaction.
        txn: u64,
        /// The analyzed table.
        table: u64,
        /// The stored statistics.
        stats: TableStats,
    },
    /// The statistics of `table` were cleared (the savepoint-compensation inverse of a first
    /// `ANALYZE`).
    ClearStats {
        /// Owning transaction.
        txn: u64,
        /// The table whose stats are cleared.
        table: u64,
    },
    /// Sequence `id` was created with `def`. **Non-transactional**: replayed
    /// unconditionally (no owning transaction — the reserved txn slot 0 is written); a
    /// rolled-back `CREATE` is neutralized by a follow-up [`LoggedOp::SeqDrop`] the rollback
    /// path appends.
    SeqCreate {
        /// The id the engine assigned.
        id: u64,
        /// The sequence definition.
        def: SequenceDef,
    },
    /// Sequence `id` was dropped (or a rolled-back create was neutralized). Non-transactional,
    /// replayed unconditionally.
    SeqDrop {
        /// The dropped sequence id.
        id: u64,
    },
    /// Sequence `id`'s counter now stands at `value` (`nextval` advance or `setval`).
    /// Non-transactional, replayed unconditionally, and **fsynced at append** — the record being
    /// durable before the value can escape is what makes a crash never hand a number out twice.
    SeqSet {
        /// The sequence id.
        id: u64,
        /// The counter value after the advance.
        value: i64,
    },
    /// `ALTER TABLE` advanced `table`'s schema to `version` with the new definition `def`.
    /// Transactional, committed-gated; the row rewrites the SQL layer drives ride
    /// as ordinary `Update` records.
    AlterSchema {
        /// Owning transaction.
        txn: u64,
        /// The altered table.
        table: u64,
        /// The new (post-alter) schema version.
        version: u32,
        /// The full new table definition (schema/name/columns).
        def: TableDef,
    },
    /// `CREATE SCHEMA` registered namespace `id` named `name`. Rollback-aware DDL.
    SchemaCreate {
        /// Owning transaction.
        txn: u64,
        /// The id the engine assigned.
        id: u64,
        /// The namespace name.
        name: String,
    },
    /// `DROP SCHEMA` removed namespace `id` (member tables drop as their own `DropTable`
    /// records). Rollback-aware DDL.
    SchemaDrop {
        /// Owning transaction.
        txn: u64,
        /// The dropped namespace id.
        id: u64,
        /// The namespace name (for the by-name index).
        name: String,
    },
}

impl LoggedOp {
    /// The transaction that performed this operation.
    #[must_use]
    pub const fn txn(&self) -> u64 {
        match self {
            Self::Insert { txn, .. }
            | Self::Update { txn, .. }
            | Self::Delete { txn, .. }
            | Self::CreateTable { txn, .. }
            | Self::DropTable { txn, .. }
            | Self::CreateIndex { txn, .. }
            | Self::DropIndex { txn, .. }
            | Self::IndexInsert { txn, .. }
            | Self::IndexDelete { txn, .. }
            | Self::IndexUnstamp { txn, .. }
            | Self::AddUnique { txn, .. }
            | Self::AddCheck { txn, .. }
            | Self::AddFk { txn, .. }
            | Self::DropConstraint { txn, .. }
            | Self::SetStats { txn, .. }
            | Self::ClearStats { txn, .. }
            | Self::AlterSchema { txn, .. }
            | Self::SchemaCreate { txn, .. }
            | Self::SchemaDrop { txn, .. } => *txn,
            // Non-transactional records own no transaction: the reserved id 0 (never begun,
            // never committed) — replay applies them unconditionally instead.
            Self::SeqCreate { .. } | Self::SeqDrop { .. } | Self::SeqSet { .. } => 0,
        }
    }

    /// Whether this record is **non-transactional** (sequence family): replayed unconditionally,
    /// not gated on a commit marker — gap semantics, a value/creation survives its transaction's
    /// rollback unless explicitly neutralized.
    #[must_use]
    pub const fn is_non_transactional(&self) -> bool {
        matches!(
            self,
            Self::SeqCreate { .. } | Self::SeqDrop { .. } | Self::SeqSet { .. }
        )
    }

    /// Encode into the shared WAL record shape (a `Put` with a self-describing key).
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-op encoder; splitting it would scatter the log format"
    )]
    pub fn to_record(&self) -> WalRecord {
        let mut key = Vec::with_capacity(25);
        let mut value = Vec::new();
        match self {
            Self::Insert {
                txn,
                table,
                row_id,
                tuple,
            } => {
                push_key(&mut key, TAG_INSERT, *txn, *table, Some(*row_id));
                value.extend_from_slice(tuple);
            },
            Self::Update {
                txn,
                table,
                row_id,
                tuple,
            } => {
                push_key(&mut key, TAG_UPDATE, *txn, *table, Some(*row_id));
                value.extend_from_slice(tuple);
            },
            Self::Delete { txn, table, row_id } => {
                push_key(&mut key, TAG_DELETE, *txn, *table, Some(*row_id));
            },
            Self::CreateTable { txn, table, def } => {
                push_key(&mut key, TAG_CREATE_TABLE, *txn, *table, None);
                encode_table_def(&mut value, def);
            },
            Self::DropTable { txn, table } => {
                push_key(&mut key, TAG_DROP_TABLE, *txn, *table, None);
            },
            Self::CreateIndex { txn, index, def } => {
                push_key(&mut key, TAG_CREATE_INDEX, *txn, *index, None);
                encode_index_def(&mut value, def);
            },
            Self::DropIndex { txn, index } => {
                push_key(&mut key, TAG_DROP_INDEX, *txn, *index, None);
            },
            Self::IndexInsert {
                txn,
                index,
                row_id,
                key: entry,
            } => {
                push_key(&mut key, TAG_INDEX_INSERT, *txn, *index, Some(*row_id));
                value.extend_from_slice(entry);
            },
            Self::IndexDelete {
                txn,
                index,
                row_id,
                key: entry,
            } => {
                push_key(&mut key, TAG_INDEX_DELETE, *txn, *index, Some(*row_id));
                value.extend_from_slice(entry);
            },
            Self::IndexUnstamp {
                txn,
                index,
                row_id,
                key: entry,
            } => {
                push_key(&mut key, TAG_INDEX_UNSTAMP, *txn, *index, Some(*row_id));
                value.extend_from_slice(entry);
            },
            Self::AddUnique {
                txn,
                table,
                index,
                name,
                columns,
                primary,
            } => {
                push_key(&mut key, TAG_ADD_UNIQUE, *txn, *table, None);
                value.extend_from_slice(&index.to_le_bytes());
                value.push(u8::from(*primary));
                push_str(&mut value, name);
                push_strs(&mut value, columns);
            },
            Self::AddCheck {
                txn,
                table,
                name,
                expr,
            } => {
                push_key(&mut key, TAG_ADD_CHECK, *txn, *table, None);
                push_str(&mut value, name);
                push_bytes(&mut value, expr);
            },
            Self::AddFk {
                txn,
                name,
                child_table,
                child_columns,
                parent_table,
                parent_index,
                child_index,
                on_delete,
                on_update,
            } => {
                push_key(&mut key, TAG_ADD_FK, *txn, *child_table, None);
                value.extend_from_slice(&parent_table.to_le_bytes());
                value.extend_from_slice(&parent_index.to_le_bytes());
                value.extend_from_slice(&child_index.to_le_bytes());
                value.push(encode_fk_action(*on_delete));
                value.push(encode_fk_action(*on_update));
                push_str(&mut value, name);
                push_strs(&mut value, child_columns);
            },
            Self::DropConstraint { txn, table, name } => {
                push_key(&mut key, TAG_DROP_CONSTRAINT, *txn, *table, None);
                push_str(&mut value, name);
            },
            Self::SetStats { txn, table, stats } => {
                push_key(&mut key, TAG_SET_STATS, *txn, *table, None);
                encode_table_stats(&mut value, stats);
            },
            Self::ClearStats { txn, table } => {
                push_key(&mut key, TAG_CLEAR_STATS, *txn, *table, None);
            },
            Self::SeqCreate { id, def } => {
                push_key(&mut key, TAG_SEQ_CREATE, 0, *id, None);
                encode_sequence_def(&mut value, def);
            },
            Self::SeqDrop { id } => {
                push_key(&mut key, TAG_SEQ_DROP, 0, *id, None);
            },
            Self::SeqSet { id, value: v } => {
                push_key(&mut key, TAG_SEQ_SET, 0, *id, None);
                value.extend_from_slice(&v.to_le_bytes());
            },
            Self::AlterSchema {
                txn,
                table,
                version,
                def,
            } => {
                push_key(&mut key, TAG_ALTER_SCHEMA, *txn, *table, None);
                value.extend_from_slice(&version.to_le_bytes());
                encode_table_def(&mut value, def);
            },
            Self::SchemaCreate { txn, id, name } => {
                push_key(&mut key, TAG_SCHEMA_CREATE, *txn, *id, None);
                push_str(&mut value, name);
            },
            Self::SchemaDrop { txn, id, name } => {
                push_key(&mut key, TAG_SCHEMA_DROP, *txn, *id, None);
                push_str(&mut value, name);
            },
        }
        WalRecord::Put { key, value }
    }

    /// Decode from a WAL record, or `None` for records this engine does not own (foreign shapes
    /// end replay loudly at the caller — a mixed log is a deployment error, not data).
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-op decoder; splitting it would scatter the log format"
    )]
    pub fn from_record(record: &WalRecord) -> Option<Self> {
        let WalRecord::Put { key, value } = record else {
            return None;
        };
        let (&tag, rest) = key.split_first()?;
        let txn = read_u64(rest, 0)?;
        let table = read_u64(rest, 8)?;
        Some(match tag {
            TAG_INSERT => Self::Insert {
                txn,
                table,
                row_id: read_u64(rest, 16)?,
                tuple: value.clone(),
            },
            TAG_UPDATE => Self::Update {
                txn,
                table,
                row_id: read_u64(rest, 16)?,
                tuple: value.clone(),
            },
            TAG_DELETE => Self::Delete {
                txn,
                table,
                row_id: read_u64(rest, 16)?,
            },
            TAG_CREATE_TABLE => Self::CreateTable {
                txn,
                table,
                def: decode_table_def(value)?,
            },
            TAG_DROP_TABLE => Self::DropTable { txn, table },
            TAG_CREATE_INDEX => Self::CreateIndex {
                txn,
                index: table,
                def: decode_index_def(value)?,
            },
            TAG_DROP_INDEX => Self::DropIndex { txn, index: table },
            TAG_INDEX_INSERT => Self::IndexInsert {
                txn,
                index: table,
                row_id: read_u64(rest, 16)?,
                key: value.clone(),
            },
            TAG_INDEX_DELETE => Self::IndexDelete {
                txn,
                index: table,
                row_id: read_u64(rest, 16)?,
                key: value.clone(),
            },
            TAG_INDEX_UNSTAMP => Self::IndexUnstamp {
                txn,
                index: table,
                row_id: read_u64(rest, 16)?,
                key: value.clone(),
            },
            TAG_ADD_UNIQUE => {
                let index = read_u64(value, 0)?;
                let primary = match *value.get(8)? {
                    0 => false,
                    1 => true,
                    _ => return None,
                };
                let mut at = 9;
                let name = read_str(value, &mut at)?;
                let columns = read_strs(value, &mut at)?;
                Self::AddUnique {
                    txn,
                    table,
                    index,
                    name,
                    columns,
                    primary,
                }
            },
            TAG_ADD_CHECK => {
                let mut at = 0;
                let name = read_str(value, &mut at)?;
                let expr = read_bytes(value, &mut at)?;
                Self::AddCheck {
                    txn,
                    table,
                    name,
                    expr,
                }
            },
            TAG_ADD_FK => {
                let parent_table = read_u64(value, 0)?;
                let parent_index = read_u64(value, 8)?;
                let child_index = read_u64(value, 16)?;
                let on_delete = decode_fk_action(*value.get(24)?)?;
                let on_update = decode_fk_action(*value.get(25)?)?;
                let mut at = 26;
                let name = read_str(value, &mut at)?;
                let child_columns = read_strs(value, &mut at)?;
                Self::AddFk {
                    txn,
                    name,
                    child_table: table,
                    child_columns,
                    parent_table,
                    parent_index,
                    child_index,
                    on_delete,
                    on_update,
                }
            },
            TAG_DROP_CONSTRAINT => {
                let mut at = 0;
                Self::DropConstraint {
                    txn,
                    table,
                    name: read_str(value, &mut at)?,
                }
            },
            TAG_SET_STATS => Self::SetStats {
                txn,
                table,
                stats: decode_table_stats(value)?,
            },
            TAG_CLEAR_STATS => Self::ClearStats { txn, table },
            TAG_SEQ_CREATE => Self::SeqCreate {
                id: table,
                def: decode_sequence_def(value)?,
            },
            TAG_SEQ_DROP => Self::SeqDrop { id: table },
            TAG_SEQ_SET => Self::SeqSet {
                id: table,
                value: read_i64(value, 0)?,
            },
            TAG_ALTER_SCHEMA => {
                let version = u32::from_le_bytes(value.get(0..4)?.try_into().ok()?);
                let mut at = 4;
                let def = decode_table_def_at(value, &mut at)?;
                Self::AlterSchema {
                    txn,
                    table,
                    version,
                    def,
                }
            },
            TAG_SCHEMA_CREATE => {
                let mut at = 0;
                Self::SchemaCreate {
                    txn,
                    id: table,
                    name: read_str(value, &mut at)?,
                }
            },
            TAG_SCHEMA_DROP => {
                let mut at = 0;
                Self::SchemaDrop {
                    txn,
                    id: table,
                    name: read_str(value, &mut at)?,
                }
            },
            _ => return None,
        })
    }
}

fn push_key(key: &mut Vec<u8>, tag: u8, txn: u64, table: u64, row_id: Option<u64>) {
    key.push(tag);
    key.extend_from_slice(&txn.to_le_bytes());
    key.extend_from_slice(&table.to_le_bytes());
    if let Some(row_id) = row_id {
        key.extend_from_slice(&row_id.to_le_bytes());
    }
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
}

// --- TableDef codec -------------------------------------------------------------------------
//
// Self-contained (this crate owns its log format); the exhaustive matches below make the
// compiler flag any new `ColumnType`/`ArrayElem` variant so the codec can never silently drop
// one.

fn push_str(out: &mut Vec<u8>, s: &str) {
    let len = u32::try_from(s.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn read_str(bytes: &[u8], at: &mut usize) -> Option<String> {
    let len = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
    *at += 4;
    let end = *at + usize::try_from(len).ok()?;
    let s = String::from_utf8(bytes.get(*at..end)?.to_vec()).ok()?;
    *at = end;
    Some(s)
}

fn encode_table_def(out: &mut Vec<u8>, def: &TableDef) {
    push_str(out, &def.schema);
    push_str(out, &def.name);
    let count = u32::try_from(def.columns.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&count.to_le_bytes());
    for column in &def.columns {
        push_str(out, &column.name);
        out.push(u8::from(column.nullable));
        encode_column_type(out, column.ty);
    }
}

fn decode_table_def(bytes: &[u8]) -> Option<TableDef> {
    let mut at = 0;
    decode_table_def_at(bytes, &mut at)
}

/// Decode a [`TableDef`] starting at `*at`, advancing the cursor past it — so it can follow a
/// header (e.g. the schema version in an `AlterSchema` record).
fn decode_table_def_at(bytes: &[u8], at: &mut usize) -> Option<TableDef> {
    let schema = read_str(bytes, at)?;
    let name = read_str(bytes, at)?;
    let count = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
    *at += 4;
    let mut columns = Vec::new();
    for _ in 0..count {
        let column_name = read_str(bytes, at)?;
        let nullable = *bytes.get(*at)? != 0;
        *at += 1;
        let ty = decode_column_type(bytes, at)?;
        columns.push(ColumnDef {
            name: column_name,
            ty,
            nullable,
        });
    }
    Some(TableDef {
        schema,
        name,
        columns,
    })
}

fn encode_column_type(out: &mut Vec<u8>, ty: ColumnType) {
    match ty {
        ColumnType::Bool => out.push(0),
        ColumnType::Int => out.push(1),
        ColumnType::SmallInt => out.push(2),
        ColumnType::BigInt => out.push(3),
        ColumnType::Float => out.push(4),
        ColumnType::Real => out.push(5),
        ColumnType::Text => out.push(6),
        ColumnType::VarChar(n) => {
            out.push(7);
            out.extend_from_slice(&n.to_le_bytes());
        },
        ColumnType::Char(n) => {
            out.push(8);
            out.extend_from_slice(&n.to_le_bytes());
        },
        ColumnType::Bytes => out.push(9),
        ColumnType::Timestamp => out.push(10),
        ColumnType::Date => out.push(11),
        ColumnType::Time => out.push(12),
        ColumnType::TimestampTz => out.push(13),
        ColumnType::TimeTz => out.push(14),
        ColumnType::Uuid => out.push(15),
        ColumnType::Numeric { precision, scale } => {
            out.push(16);
            out.push(precision);
            out.push(scale);
        },
        ColumnType::Json => out.push(17),
        ColumnType::Jsonb => out.push(18),
        ColumnType::Interval => out.push(19),
        ColumnType::Array(elem) => {
            out.push(20);
            out.push(encode_array_elem(elem));
        },
        ColumnType::Vector(n) => {
            out.push(21);
            out.extend_from_slice(&n.to_le_bytes());
        },
    }
}

fn decode_column_type(bytes: &[u8], at: &mut usize) -> Option<ColumnType> {
    let tag = *bytes.get(*at)?;
    *at += 1;
    let read_u32 = |at: &mut usize| -> Option<u32> {
        let v = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
        *at += 4;
        Some(v)
    };
    Some(match tag {
        0 => ColumnType::Bool,
        1 => ColumnType::Int,
        2 => ColumnType::SmallInt,
        3 => ColumnType::BigInt,
        4 => ColumnType::Float,
        5 => ColumnType::Real,
        6 => ColumnType::Text,
        7 => ColumnType::VarChar(read_u32(at)?),
        8 => ColumnType::Char(read_u32(at)?),
        9 => ColumnType::Bytes,
        10 => ColumnType::Timestamp,
        11 => ColumnType::Date,
        12 => ColumnType::Time,
        13 => ColumnType::TimestampTz,
        14 => ColumnType::TimeTz,
        15 => ColumnType::Uuid,
        16 => {
            let precision = *bytes.get(*at)?;
            let scale = *bytes.get(*at + 1)?;
            *at += 2;
            ColumnType::Numeric { precision, scale }
        },
        17 => ColumnType::Json,
        18 => ColumnType::Jsonb,
        19 => ColumnType::Interval,
        20 => {
            let elem = decode_array_elem(*bytes.get(*at)?)?;
            *at += 1;
            ColumnType::Array(elem)
        },
        21 => ColumnType::Vector(read_u32(at)?),
        _ => return None,
    })
}

const fn encode_array_elem(elem: ArrayElem) -> u8 {
    match elem {
        ArrayElem::Bool => 0,
        ArrayElem::Int => 1,
        ArrayElem::Float => 2,
        ArrayElem::Numeric => 3,
        ArrayElem::Text => 4,
        ArrayElem::Date => 5,
        ArrayElem::Time => 6,
        ArrayElem::Timestamp => 7,
        ArrayElem::TimestampTz => 8,
        ArrayElem::Uuid => 9,
    }
}

const fn decode_array_elem(tag: u8) -> Option<ArrayElem> {
    Some(match tag {
        0 => ArrayElem::Bool,
        1 => ArrayElem::Int,
        2 => ArrayElem::Float,
        3 => ArrayElem::Numeric,
        4 => ArrayElem::Text,
        5 => ArrayElem::Date,
        6 => ArrayElem::Time,
        7 => ArrayElem::Timestamp,
        8 => ArrayElem::TimestampTz,
        9 => ArrayElem::Uuid,
        _ => return None,
    })
}

/// Encode an [`IndexDef`] (same self-contained discipline as the `TableDef` codec above).
fn encode_index_def(out: &mut Vec<u8>, def: &IndexDef) {
    push_str(out, &def.name);
    out.extend_from_slice(&def.table.0.to_le_bytes());
    let columns = u32::try_from(def.columns.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&columns.to_le_bytes());
    for c in &def.columns {
        push_str(out, c);
    }
    let include = u32::try_from(def.include.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&include.to_le_bytes());
    for c in &def.include {
        push_str(out, c);
    }
    out.push(match def.kind {
        IndexKind::BTree => 0,
        IndexKind::Hash => 1,
        IndexKind::Brin => 2,
    });
    out.push(u8::from(def.unique));
    // Appended fields (functional/expression + partial indexes): key expressions and an optional
    // partial predicate, both as SQL text. Written after the original fields so a record laid down
    // by an older writer (which lacks them) decodes with these defaulted empty/None.
    let key_exprs = u32::try_from(def.key_exprs.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&key_exprs.to_le_bytes());
    for e in &def.key_exprs {
        push_str(out, e);
    }
    match &def.predicate {
        Some(pred) => {
            out.push(1);
            push_str(out, pred);
        },
        None => out.push(0),
    }
}

fn decode_index_def(bytes: &[u8]) -> Option<IndexDef> {
    let mut at = 0usize;
    let name = read_str(bytes, &mut at)?;
    let table = read_u64(bytes, at)?;
    at += 8;
    let read_u32 = |bytes: &[u8], at: &mut usize| -> Option<u32> {
        let v = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
        *at += 4;
        Some(v)
    };
    let n = read_u32(bytes, &mut at)?;
    // No preallocation from the untrusted length prefix (same discipline as the TableDef codec).
    let mut columns = Vec::new();
    for _ in 0..n {
        columns.push(read_str(bytes, &mut at)?);
    }
    let n = read_u32(bytes, &mut at)?;
    let mut include = Vec::new();
    for _ in 0..n {
        include.push(read_str(bytes, &mut at)?);
    }
    let kind = match *bytes.get(at)? {
        0 => IndexKind::BTree,
        1 => IndexKind::Hash,
        2 => IndexKind::Brin,
        _ => return None,
    };
    let unique = match *bytes.get(at + 1)? {
        0 => false,
        1 => true,
        _ => return None,
    };
    at += 2;
    // Appended fields (see `encode_index_def`). A record from an older writer ends here, so a
    // missing trailer decodes as an empty key-expression list and no partial predicate.
    let (key_exprs, predicate) = if at < bytes.len() {
        let n = read_u32(bytes, &mut at)?;
        let mut key_exprs = Vec::new();
        for _ in 0..n {
            key_exprs.push(read_str(bytes, &mut at)?);
        }
        let predicate = match *bytes.get(at)? {
            0 => None,
            1 => {
                at += 1;
                Some(read_str(bytes, &mut at)?)
            },
            _ => return None,
        };
        (key_exprs, predicate)
    } else {
        (Vec::new(), None)
    };
    Some(IndexDef {
        name,
        table: TableId(table),
        columns,
        key_exprs,
        predicate,
        include,
        kind,
        unique,
    })
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_bytes(bytes: &[u8], at: &mut usize) -> Option<Vec<u8>> {
    let len = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
    *at += 4;
    let end = *at + usize::try_from(len).ok()?;
    let out = bytes.get(*at..end)?.to_vec();
    *at = end;
    Some(out)
}

fn push_strs(out: &mut Vec<u8>, strs: &[String]) {
    let n = u32::try_from(strs.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n.to_le_bytes());
    for s in strs {
        push_str(out, s);
    }
}

fn read_strs(bytes: &[u8], at: &mut usize) -> Option<Vec<String>> {
    let n = u32::from_le_bytes(bytes.get(*at..*at + 4)?.try_into().ok()?);
    *at += 4;
    // No preallocation from the untrusted length prefix (TableDef-codec discipline).
    let mut out = Vec::new();
    for _ in 0..n {
        out.push(read_str(bytes, at)?);
    }
    Some(out)
}

fn push_opt_bytes(out: &mut Vec<u8>, opt: Option<&[u8]>) {
    match opt {
        Some(bytes) => {
            out.push(1);
            push_bytes(out, bytes);
        },
        None => out.push(0),
    }
}

#[allow(
    clippy::option_option,
    reason = "the outer Option is decode success, the inner is the encoded field's own               optionality (ColumnStats min/max) — a custom enum would just rename the two"
)]
fn read_opt_bytes(bytes: &[u8], at: &mut usize) -> Option<Option<Vec<u8>>> {
    let flag = *bytes.get(*at)?;
    *at += 1;
    Some(match flag {
        0 => None,
        1 => Some(read_bytes(bytes, at)?),
        _ => return None,
    })
}

const fn encode_fk_action(action: FkAction) -> u8 {
    match action {
        FkAction::NoAction => 0,
        FkAction::Restrict => 1,
        FkAction::Cascade => 2,
        FkAction::SetNull => 3,
        FkAction::SetDefault => 4,
    }
}

const fn decode_fk_action(tag: u8) -> Option<FkAction> {
    Some(match tag {
        0 => FkAction::NoAction,
        1 => FkAction::Restrict,
        2 => FkAction::Cascade,
        3 => FkAction::SetNull,
        4 => FkAction::SetDefault,
        _ => return None,
    })
}

/// Encode a [`TableStats`] (same self-contained discipline as the `TableDef` codec).
fn encode_table_stats(out: &mut Vec<u8>, stats: &TableStats) {
    out.extend_from_slice(&stats.row_count.to_le_bytes());
    out.extend_from_slice(&stats.page_count.to_le_bytes());
    let n = u32::try_from(stats.columns.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&n.to_le_bytes());
    for c in &stats.columns {
        push_str(out, &c.column);
        out.extend_from_slice(&c.null_count.to_le_bytes());
        out.extend_from_slice(&c.distinct_count.to_le_bytes());
        push_opt_bytes(out, c.min.as_deref());
        push_opt_bytes(out, c.max.as_deref());
        let mcv = u32::try_from(c.most_common.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&mcv.to_le_bytes());
        for (v, freq) in &c.most_common {
            push_bytes(out, v);
            out.extend_from_slice(&freq.to_le_bytes());
        }
        let hist = u32::try_from(c.histogram.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&hist.to_le_bytes());
        for b in &c.histogram {
            push_bytes(out, b);
        }
    }
}

fn decode_table_stats(bytes: &[u8]) -> Option<TableStats> {
    let row_count = read_u64(bytes, 0)?;
    let page_count = read_u64(bytes, 8)?;
    let mut at = 16;
    let n = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?);
    at += 4;
    let mut columns = Vec::new();
    for _ in 0..n {
        let column = read_str(bytes, &mut at)?;
        let null_count = read_u64(bytes, at)?;
        at += 8;
        let distinct_count = read_u64(bytes, at)?;
        at += 8;
        let min = read_opt_bytes(bytes, &mut at)?;
        let max = read_opt_bytes(bytes, &mut at)?;
        let mcv = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?);
        at += 4;
        let mut most_common = Vec::new();
        for _ in 0..mcv {
            let v = read_bytes(bytes, &mut at)?;
            let freq = read_u64(bytes, at)?;
            at += 8;
            most_common.push((v, freq));
        }
        let hist = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?);
        at += 4;
        let mut histogram = Vec::new();
        for _ in 0..hist {
            histogram.push(read_bytes(bytes, &mut at)?);
        }
        columns.push(ColumnStats {
            column,
            null_count,
            distinct_count,
            min,
            max,
            most_common,
            histogram,
        });
    }
    Some(TableStats {
        row_count,
        page_count,
        columns,
    })
}

fn read_i64(bytes: &[u8], at: usize) -> Option<i64> {
    Some(i64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
}

/// Encode a [`SequenceDef`] (same self-contained discipline as the other catalog codecs).
fn encode_sequence_def(out: &mut Vec<u8>, def: &SequenceDef) {
    push_str(out, &def.name);
    out.extend_from_slice(&def.start.to_le_bytes());
    out.extend_from_slice(&def.increment.to_le_bytes());
    out.extend_from_slice(&def.min_value.to_le_bytes());
    out.extend_from_slice(&def.max_value.to_le_bytes());
    out.push(u8::from(def.cycle));
}

fn decode_sequence_def(bytes: &[u8]) -> Option<SequenceDef> {
    let mut at = 0usize;
    let name = read_str(bytes, &mut at)?;
    let start = read_i64(bytes, at)?;
    let increment = read_i64(bytes, at + 8)?;
    let min_value = read_i64(bytes, at + 16)?;
    let max_value = read_i64(bytes, at + 24)?;
    let cycle = match *bytes.get(at + 32)? {
        0 => false,
        1 => true,
        _ => return None,
    };
    Some(SequenceDef {
        name,
        start,
        increment,
        min_value,
        max_value,
        cycle,
    })
}

/// A convenience: encode-then-decode must be the identity for every op shape (pinned by tests).
#[must_use]
pub fn roundtrip_check(op: &LoggedOp) -> bool {
    LoggedOp::from_record(&op.to_record()).as_ref() == Some(op)
}

#[cfg(test)]
mod index_def_codec_tests {
    use super::{decode_index_def, encode_index_def};
    use nusadb_core::engine::{IndexDef, IndexKind};

    /// A record laid down by an older writer ends right after the `unique` byte (it predates the
    /// appended `key_exprs` / `predicate` fields). Decoding such a truncated record must succeed
    /// with those fields defaulted — the upgrade-in-place path (recovering a data dir written
    /// before functional/partial indexes existed).
    #[test]
    fn decodes_pre_extension_record_with_defaults() {
        let def = IndexDef {
            name: "i".to_owned(),
            table: nusadb_core::TableId(1),
            columns: vec!["a".to_owned()],
            key_exprs: Vec::new(),
            predicate: None,
            include: Vec::new(),
            kind: IndexKind::BTree,
            unique: false,
        };
        let mut full = Vec::new();
        encode_index_def(&mut full, &def);
        // The appended trailer is: [key_exprs count u32 = 0][predicate flag u8 = 0] = 5 bytes.
        let old = &full[..full.len() - 5];
        let decoded = decode_index_def(old).expect("truncated (old-format) record must decode");
        assert_eq!(
            decoded, def,
            "old record decodes with empty key_exprs + no predicate"
        );
        // And the full record round-trips identically.
        assert_eq!(decode_index_def(&full).as_ref(), Some(&def));
    }
}
