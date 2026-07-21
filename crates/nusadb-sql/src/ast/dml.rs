//! DML statement types: INSERT/UPDATE/DELETE/MERGE.
//!
//! Pure AST types split verbatim out of `ast/mod.rs` (ADR 007). Sibling types resolve via
//! `use super::*` (re-exported by the parent).
#![allow(clippy::wildcard_imports)]

use super::*;

/// `INSERT INTO table (columns) <source> [ON CONFLICT ...] [RETURNING ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Insert {
    /// Explicit schema qualifier of the target: `Some(schema)` for `INSERT INTO schema.t`,
    /// `None` when unqualified (resolved through the search path).
    pub schema: Option<String>,
    /// Target table name.
    pub table: String,
    /// Explicit column list; empty means "all columns, in declaration order".
    pub columns: Vec<String>,
    /// The row source — either a `VALUES` list or a `SELECT` query.
    pub source: InsertSource,
    /// Optional `ON CONFLICT` clause; `None` when absent.
    pub on_conflict: Option<OnConflict>,
    /// `RETURNING` projection; empty when the clause is absent.
    pub returning: Vec<SelectItem>,
}

/// The row-producing source of an `INSERT` statement.
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// `VALUES (row1), (row2), ...` — always at least one row.
    ///
    /// A cell is `None` when the source text wrote the `DEFAULT` keyword for that
    /// position; the executor substitutes the target column's default/serial/NULL
    /// fill at insert time (per-cell `DEFAULT`).
    Values(Vec<Vec<Option<Expr>>>),
    /// `SELECT ...` — any supported select statement.
    Select(Box<Select>),
    /// `DEFAULT VALUES` — insert exactly one row whose every column takes its `DEFAULT`.
    DefaultValues,
}

/// A `COPY ... FROM STDIN` / `COPY ... TO STDOUT` statement in the text format.
///
/// Only the streaming STDIN/STDOUT forms are modeled; a file/program target or a query source is
/// rejected at the parser. The data itself rides the wire's COPY sub-protocol, so the AST carries
/// only the shape and the text-format options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Copy {
    /// Target (FROM) or source (TO) table.
    pub table: String,
    /// Explicit column list; empty means "all columns, in declaration order".
    pub columns: Vec<String>,
    /// Stream direction: `From` = `FROM STDIN` (bulk load), `To` = `TO STDOUT` (bulk export).
    pub direction: CopyDirection,
    /// Text-format options (delimiter, NULL marker, header).
    pub format: CopyFormat,
}

/// The direction of a [`Copy`](struct@Copy) stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyDirection {
    /// `COPY ... FROM STDIN` — read rows from the client and insert them.
    From,
    /// `COPY ... TO STDOUT` — write the table's rows to the client.
    To,
}

/// Text-format options for [`Copy`](struct@Copy) (the subset NusaDB honors today).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyFormat {
    /// Field separator (default tab).
    pub delimiter: char,
    /// The token that represents SQL `NULL` (default `\N`).
    pub null: String,
    /// Whether the first data line is a header to skip (FROM) / emit (TO). Default `false`.
    pub header: bool,
}

impl Default for CopyFormat {
    fn default() -> Self {
        Self {
            delimiter: '\t',
            null: "\\N".to_owned(),
            header: false,
        }
    }
}

/// `ON CONFLICT [target] <action>` on an `INSERT`.
#[derive(Debug, Clone, PartialEq)]
pub struct OnConflict {
    /// The conflict arbiter; `None` for a bare `ON CONFLICT` with no target.
    pub target: Option<ConflictTarget>,
    /// The action to take on a conflict.
    pub action: ConflictAction,
}

/// What `ON CONFLICT` arbitrates on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictTarget {
    /// `ON CONFLICT (col1, col2, ...)` — the conflicting columns.
    Columns(Vec<String>),
    /// `ON CONFLICT ON CONSTRAINT name`.
    Constraint(String),
}

/// The action an `ON CONFLICT` clause takes.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictAction {
    /// `DO NOTHING` — skip the conflicting row.
    DoNothing,
    /// `DO UPDATE SET col = value, ... [WHERE pred]` — update the existing (conflicting) row.
    ///
    /// A `value` (or the `WHERE` predicate) may reference the proposed insert row as
    /// `EXCLUDED.col`, which parses to an [`Expr::QualifiedColumn`] with table `excluded`.
    DoUpdate {
        /// `SET column = value` pairs; always non-empty.
        assignments: Vec<Assignment>,
        /// Optional `WHERE` predicate gating the update.
        filter: Option<Expr>,
    },
}

/// `UPDATE table SET assignments [FROM ...] [WHERE ...] [RETURNING ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Explicit schema qualifier of the target: `Some(schema)` for `UPDATE schema.t`, `None`
    /// when unqualified (resolved through the search path).
    pub schema: Option<String>,
    /// Target table name.
    pub table: String,
    /// Optional alias for the target table (`UPDATE t AS x`): when present, the `SET` values and
    /// `WHERE` reference the target by this alias (which shadows the table name), e.g. `x.id`.
    pub alias: Option<String>,
    /// `SET column = value` pairs; always non-empty.
    pub assignments: Vec<Assignment>,
    /// `FROM` join source — additional tables/joins for the `SET` expressions and `WHERE` predicate
    /// to reference. `None` when the clause is absent.
    pub from: Option<FromClause>,
    /// `WHERE` predicate.
    pub filter: Option<Expr>,
    /// `RETURNING` projection; empty when the clause is absent.
    pub returning: Vec<SelectItem>,
}

/// One `column = value` pair inside an [`Update`].
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Column being assigned.
    pub column: String,
    /// New value expression.
    pub value: Expr,
}

/// `DELETE FROM table [USING ...] [WHERE ...] [RETURNING ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Delete {
    /// Explicit schema qualifier of the target: `Some(schema)` for `DELETE FROM schema.t`,
    /// `None` when unqualified (resolved through the search path).
    pub schema: Option<String>,
    /// Target table name.
    pub table: String,
    /// `USING` join source — additional tables/joins the `WHERE` predicate can reference.
    /// `None` when the clause is absent.
    pub using: Option<FromClause>,
    /// `WHERE` predicate.
    pub filter: Option<Expr>,
    /// `RETURNING` projection; empty when the clause is absent.
    pub returning: Vec<SelectItem>,
}

/// `MERGE INTO target USING source ON cond { WHEN ... }+`.
#[derive(Debug, Clone, PartialEq)]
pub struct Merge {
    /// The target table being merged into.
    pub target: TableRef,
    /// The source table joined against the target. Only a plain table (with optional alias) is
    /// modelled today — a subquery source is rejected.
    pub source: TableRef,
    /// The `ON` join condition between target and source.
    pub on: Expr,
    /// The ordered `WHEN [NOT] MATCHED` clauses; always non-empty.
    pub whens: Vec<MergeWhen>,
}

/// One `WHEN` clause of a [`Merge`].
#[derive(Debug, Clone, PartialEq)]
pub enum MergeWhen {
    /// `WHEN MATCHED [AND pred] THEN {UPDATE SET ... | DELETE}`.
    Matched {
        /// Optional `AND <pred>` guard on the clause.
        pred: Option<Expr>,
        /// The action to apply to the matched target row.
        action: MatchedAction,
    },
    /// `WHEN NOT MATCHED [AND pred] THEN INSERT (...) VALUES (...)`.
    NotMatched {
        /// Optional `AND <pred>` guard on the clause.
        pred: Option<Expr>,
        /// The row to insert for an unmatched source row.
        insert: MergeInsert,
    },
}

/// The action of a `WHEN MATCHED` clause.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchedAction {
    /// `UPDATE SET column = value, ...`.
    Update {
        /// The `SET` assignments; always non-empty.
        assignments: Vec<Assignment>,
    },
    /// `DELETE`.
    Delete,
}

/// The `INSERT (columns) VALUES (values)` of a `WHEN NOT MATCHED` clause.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeInsert {
    /// Explicit target columns; empty when the `INSERT` omits the column list.
    pub columns: Vec<String>,
    /// The `VALUES (...)` row expressions.
    pub values: Vec<Expr>,
}
