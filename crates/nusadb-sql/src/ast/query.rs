//! Query types: WITH/CTE, SELECT, set operations, FROM/joins, ORDER BY.
//!
//! Pure AST types split verbatim out of `ast/mod.rs` (ADR 007). Sibling types resolve via
//! `use super::*` (re-exported by the parent).
#![allow(clippy::wildcard_imports)]

use super::*;

/// One entry in a `WITH` clause: `name [(cols)] AS (body)` (non-recursive recursive).
///
/// The body is a query for a read CTE, or a data-modifying statement for `WITH x AS (INSERT/UPDATE …
/// RETURNING …)`. For `WITH RECURSIVE` (`recursive = true`) the query body is `anchor
/// UNION ALL recursive_arm` (a `SelectBody::SetOp`). Nested CTEs, `AS MATERIALIZED`, and `FROM`
/// aliases are rejected at the parser door with `Error::Unsupported`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    /// CTE name, folded to lowercase (like any unquoted identifier).
    pub name: String,
    /// Explicit output column names; empty means "infer from the query".
    pub columns: Vec<String>,
    /// The CTE body — a query, or a data-modifying statement (its `RETURNING` rows form the relation).
    pub body: CteBody,
    /// `true` when this CTE was introduced by `WITH RECURSIVE`.
    pub recursive: bool,
}

/// A CTE's body.
#[derive(Debug, Clone, PartialEq)]
pub enum CteBody {
    /// A read query — a plain `SELECT` (non-recursive inline) or `anchor UNION ALL arm` (recursive).
    Query(Box<SelectBody>),
    /// A data-modifying `INSERT`/`UPDATE` with `RETURNING`. It runs once and its
    /// `RETURNING` rows form the CTE's relation.
    Modifying(Box<Statement>),
}

/// The `GROUP BY` clause of a `SELECT`.
///
/// Plain expression lists (`GROUP BY a, b`) are the common case. `ROLLUP`, `CUBE`, and
/// `GROUPING SETS` extend grouping to multiple grouping-set combinations.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupBy {
    /// `GROUP BY expr, ...` — flat key list; empty means no grouping.
    Expressions(Vec<Expr>),
    /// `GROUP BY ROLLUP (a, (b, c), ...)` — hierarchical subtotals.
    Rollup(Vec<Vec<Expr>>),
    /// `GROUP BY CUBE (a, b, ...)` — all-combinations power set.
    Cube(Vec<Vec<Expr>>),
    /// `GROUP BY GROUPING SETS ((a, b), (c), ())` — explicit grouping sets.
    GroupingSets(Vec<Vec<Expr>>),
}

/// `SELECT projection FROM table [WHERE ...] [GROUP BY ...] [HAVING ...]
/// [ORDER BY ...] [LIMIT ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Select {
    /// `WITH` CTEs prefixing this query. Non-empty only at the top-level
    /// statement; always empty inside subqueries and set-operation branches.
    pub with: Vec<Cte>,
    /// `DISTINCT` clause: `None` for no de-duplication, `Some(Distinct::All)` for plain
    /// `SELECT DISTINCT`, `Some(Distinct::On(..))` for `SELECT DISTINCT ON (..)`.
    pub distinct: Option<Distinct>,
    /// Projected items.
    pub projection: Vec<SelectItem>,
    /// `FROM` clause — a base table plus zero or more joins; `None` for a
    /// `SELECT` without `FROM` (e.g. `SELECT 1`).
    pub from: Option<FromClause>,
    /// `WHERE` predicate.
    pub filter: Option<Expr>,
    /// `GROUP BY` clause — a flat expression list or a ROLLUP/CUBE/GROUPING SETS form.
    pub group_by: GroupBy,
    /// `HAVING` predicate — a post-aggregation filter; requires `GROUP BY` or
    /// aggregates in the projection.
    pub having: Option<Expr>,
    /// `ORDER BY` keys, in priority order.
    pub order_by: Vec<OrderByItem>,
    /// `LIMIT` row cap (or `FETCH FIRST n ROWS ONLY`).
    pub limit: Option<u64>,
    /// `FETCH FIRST n ROWS WITH TIES`: when `true`, `limit` keeps the trailing
    /// rows that tie the last kept row on the `ORDER BY` keys. `false` for `LIMIT` / `FETCH ... ONLY`.
    pub limit_with_ties: bool,
    /// `OFFSET n [ROW[S]]` — skip the first `n` rows before returning results.
    /// `None` when absent.
    pub offset: Option<u64>,
    /// `FOR UPDATE`/`FOR SHARE` row-lock clause; `None` when absent.
    pub lock: Option<RowLock>,
}

/// A `FOR UPDATE`/`FOR SHARE [OF t] [NOWAIT | SKIP LOCKED]` row-lock clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowLock {
    /// The lock strength.
    pub strength: LockStrength,
    /// `OF <table>` — restrict the lock to one named table/alias; `None` locks all `FROM` tables.
    pub of: Option<String>,
    /// Blocking behaviour when a target row is already locked.
    pub wait: LockWait,
}

/// The strength of a [`RowLock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockStrength {
    /// `FOR UPDATE` — exclusive row lock.
    Update,
    /// `FOR SHARE` — shared row lock.
    Share,
}

/// The wait behaviour of a [`RowLock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockWait {
    /// Block until the conflicting lock is released (the default).
    Default,
    /// `NOWAIT` — error immediately if a target row is locked.
    NoWait,
    /// `SKIP LOCKED` — silently skip rows that are already locked.
    SkipLocked,
}

/// A set operation over `SELECT` branches: `<body> [ORDER BY ...] [LIMIT n]`.
///
/// `ORDER BY` and `LIMIT` bind to the whole set-operation result, not to any single branch.
#[derive(Debug, Clone, PartialEq)]
pub struct SetOperation {
    /// The set-operation tree (left-associative).
    pub body: SelectBody,
    /// `ORDER BY` applied to the combined result, in priority order.
    pub order_by: Vec<OrderByItem>,
    /// `LIMIT` row cap on the combined result.
    pub limit: Option<u64>,
}

/// A node in a set-operation tree: either a leaf `SELECT` or a binary set operator.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectBody {
    /// A leaf `SELECT` branch.
    Select(Box<Select>),
    /// `left {UNION | INTERSECT | EXCEPT} [ALL] right`. The tree is built
    /// left-associatively, so `a UNION b UNION c` nests as `(a UNION b) UNION c`.
    SetOp {
        /// The set operator.
        op: SetOp,
        /// `true` for `ALL` (keep duplicates); `false` for the default (distinct).
        all: bool,
        /// Left operand.
        left: Box<Self>,
        /// Right operand.
        right: Box<Self>,
    },
}

/// A SQL set operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOp {
    /// `UNION` — rows in either branch.
    Union,
    /// `INTERSECT` — rows in both branches.
    Intersect,
    /// `EXCEPT` — rows in the left branch but not the right.
    Except,
}

/// The `DISTINCT` clause of a [`Select`].
#[derive(Debug, Clone, PartialEq)]
pub enum Distinct {
    /// `SELECT DISTINCT` — drop duplicate output rows across all projected columns.
    All,
    /// `SELECT DISTINCT ON (exprs)` — keep the first row of each distinct `exprs` tuple.
    On(Vec<Expr>),
}

/// A `FROM` clause: a base table followed by a left-deep chain of joins.
#[derive(Debug, Clone, PartialEq)]
pub struct FromClause {
    /// The first table in the `FROM`.
    pub base: TableRef,
    /// Joins applied left-to-right onto `base`.
    pub joins: Vec<Join>,
}

/// A table reference in a `FROM`/join.
///
/// A named table or view carries its `name` (and optional `alias`) with `subquery: None`. A *derived
/// table* — `(SELECT ...) AS x` — carries the parenthesized `subquery`, with `name`/`alias` both set
/// to the (mandatory) alias used as its column qualifier. `lateral` marks a `LATERAL` derived table,
/// which may reference columns from FROM items to its left.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    /// Explicit schema qualifier for a *named* table/view: `Some(schema)` when written as
    /// `schema.table`, `None` when unqualified (the analyzer resolves it through the search path).
    /// Ignored for a derived table (`subquery`/`values`/`set_op`).
    pub schema: Option<String>,
    /// Table/view name, or — for a derived table — its mandatory alias (used as the qualifier).
    pub name: String,
    /// Optional `AS` alias; when present, qualified column refs use it.
    pub alias: Option<String>,
    /// For a derived table `(SELECT ...) AS x`, the parenthesized subquery; `None` for a named table
    /// or a `(VALUES ...)` derived table (which carries `values` instead).
    pub subquery: Option<Box<Select>>,
    /// For a `(VALUES (row), ...) AS x` derived table, the inline rows; `None` otherwise. Mutually
    /// exclusive with `subquery`. Each row is evaluated against an empty input (no column refs), and
    /// the rows form a relation whose column types are unified per column across all rows.
    pub values: Option<Vec<Vec<Expr>>>,
    /// For a `(SELECT ... UNION/INTERSECT/EXCEPT ...) AS x` derived table, the set-operation body
    /// (with its own `ORDER BY`/`LIMIT`); `None` otherwise. Mutually exclusive with `subquery` and
    /// `values`. Inlined like a derived table — its result forms the relation.
    pub set_op: Option<Box<SetOperation>>,
    /// `LATERAL` derived table — may reference left FROM items. Always `false` for a named
    /// table or a non-lateral derived table.
    pub lateral: bool,
    /// Explicit output column names from a derived table's alias, `(SELECT ...) AS x(a, b)` /
    /// `(VALUES ...) AS v(a, b)`; empty when none. Applied positionally to the derived relation's
    /// columns. Only meaningful for a derived table (`subquery: Some`).
    pub column_aliases: Vec<String>,
    /// `WITH ORDINALITY` on a `FROM` set-returning function (`unnest(arr) WITH ORDINALITY`):
    /// appends a 1-based `ordinality` column to the relation. Only set on a table-function /
    /// `UNNEST` derived table; `false` everywhere else.
    pub with_ordinality: bool,
}

/// One join in a [`FromClause`].
#[derive(Debug, Clone, PartialEq)]
pub struct Join {
    /// The joined-in table.
    pub table: TableRef,
    /// Join kind.
    pub kind: JoinKind,
    /// Join condition — how the two sides are matched.
    pub condition: JoinCondition,
}

/// Supported join kinds (adds `Cross`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    /// `INNER JOIN` — only matching row pairs.
    Inner,
    /// `LEFT [OUTER] JOIN` — every left row, NULL-padded when unmatched.
    Left,
    /// `RIGHT [OUTER] JOIN` — every right row, NULL-padded when unmatched.
    Right,
    /// `FULL [OUTER] JOIN` — every row from both sides, NULL-padded when
    /// unmatched on either side.
    Full,
    /// `CROSS JOIN` — Cartesian product; no condition.
    Cross,
}

/// How a join matches rows from the two sides.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinCondition {
    /// `ON <predicate>` — explicit boolean filter.
    On(Expr),
    /// `USING (col, ...)` — shorthand for `ON left.col = right.col` for each named column.
    Using(Vec<String>),
    /// `NATURAL` — implicit equality on all identically-named columns.
    Natural,
    /// No condition — used by `CROSS JOIN`.
    None,
}

/// One item in a `SELECT` projection list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// `*` — every column of the source table.
    Wildcard,
    /// `table.*` — every column of the named table/alias, in order.
    QualifiedWildcard(String),
    /// A scalar expression with an optional `AS` alias.
    Expr {
        /// The projected expression.
        expr: Expr,
        /// Optional output column alias.
        alias: Option<String>,
    },
}

/// One `ORDER BY` sort key.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem {
    /// Sort-key expression.
    pub expr: Expr,
    /// `true` for `ASC` (the default), `false` for `DESC`.
    pub ascending: bool,
    /// `NULLS FIRST` / `NULLS LAST` ordering for `NULL` values.
    pub nulls: NullOrdering,
}

/// How `NULL` values are sorted relative to non-`NULL` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOrdering {
    /// No explicit `NULLS` clause — database default (NULLs last for ASC, first for DESC).
    Default,
    /// `NULLS FIRST` — `NULL` values sort before all non-`NULL` values.
    First,
    /// `NULLS LAST` — `NULL` values sort after all non-`NULL` values.
    Last,
}
