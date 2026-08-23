//! Expression + literal/operator/window types.
//!
//! Pure AST types split verbatim out of `ast/mod.rs` (ADR 007). Sibling types resolve via
//! `use super::*` (re-exported by the parent).
#![allow(clippy::wildcard_imports)]

use super::*;
use nusadb_core::ColumnType;

/// A scalar SQL expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal constant.
    Literal(Value),
    /// A column reference by (unqualified) name.
    Column(String),
    /// A qualified column reference `table.column` (the qualifier is a table
    /// name or alias). Used to disambiguate columns across joined tables.
    QualifiedColumn {
        /// Table name or alias the column belongs to.
        table: String,
        /// Column name within that table.
        column: String,
    },
    /// A binary operation, e.g. `a = b` or `x + 1`.
    Binary {
        /// Left operand.
        left: Box<Self>,
        /// Operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Self>,
    },
    /// A unary operation, e.g. `NOT a` or `-x`.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        expr: Box<Self>,
    },
    /// `expr IS NULL` / `expr IS NOT NULL`.
    IsNull {
        /// Operand being tested.
        expr: Box<Self>,
        /// `true` for `IS NOT NULL`, `false` for `IS NULL`.
        negated: bool,
    },
    /// `left IS [NOT] DISTINCT FROM right` — a comparison that treats
    /// `NULL` as an ordinary comparable value, so `NULL IS DISTINCT FROM NULL`
    /// is false and `NULL IS DISTINCT FROM 1` is true. The result is always
    /// boolean, never `NULL`.
    IsDistinctFrom {
        /// Left operand.
        left: Box<Self>,
        /// Right operand.
        right: Box<Self>,
        /// `true` for `IS NOT DISTINCT FROM`, `false` for `IS DISTINCT FROM`.
        negated: bool,
    },
    /// `expr IS [NOT] {TRUE | FALSE | UNKNOWN}` — a three-valued-logic
    /// truth test on a boolean operand. The result is always boolean, never
    /// `NULL` (`UNKNOWN` is the `NULL` boolean).
    IsBool {
        /// Boolean operand being tested.
        expr: Box<Self>,
        /// Which truth value is being tested for.
        truth: TruthValue,
        /// `true` for the `IS NOT …` form.
        negated: bool,
    },
    /// `<expr> IS [NOT] JSON [VALUE | SCALAR | ARRAY | OBJECT] [WITH | WITHOUT UNIQUE KEYS]` —
    /// a JSON validity / shape predicate yielding a **nullable** boolean.
    ///
    /// A `NULL` operand yields `NULL` (three-valued: null in, null out; `IS NOT JSON` of a `NULL`
    /// operand is still `NULL`). Otherwise the operand's text is tested for validity (`IS JSON` /
    /// `IS JSON VALUE` = valid JSON of any type; `SCALAR`/`ARRAY`/`OBJECT` = valid JSON of that
    /// specific type); with `unique_keys`, every object's keys must additionally be unique,
    /// recursively. `negated` flips the non-null result.
    IsJson {
        /// Operand whose text is checked for JSON validity / shape (`TEXT` or `JSON`/`JSONB`).
        operand: Box<Self>,
        /// `true` for `IS NOT JSON`.
        negated: bool,
        /// Which JSON item type is required (`Value` = any type).
        item_type: JsonItemType,
        /// `true` for `WITH UNIQUE KEYS` — every object's keys must be unique, recursively
        /// (`WITHOUT UNIQUE KEYS`, the default, imposes no uniqueness requirement).
        unique_keys: bool,
    },
    /// `expr [NOT] IN (list...)` — membership test against a fixed list.
    InList {
        /// Value being tested.
        expr: Box<Self>,
        /// Constant list to test membership against.
        list: Vec<Self>,
        /// `true` for `NOT IN`, `false` for `IN`.
        negated: bool,
    },
    /// `expr [NOT] BETWEEN low AND high` — inclusive range test.
    Between {
        /// Value being tested.
        expr: Box<Self>,
        /// Lower bound (inclusive).
        low: Box<Self>,
        /// Upper bound (inclusive).
        high: Box<Self>,
        /// `true` for `NOT BETWEEN`, `false` for `BETWEEN`.
        negated: bool,
    },
    /// `(s1, e1) OVERLAPS (s2, e2)` — do two time periods overlap? Each side is a two-element
    /// row: a start endpoint and an end endpoint. The end may be a temporal value of the same
    /// type as the start, or an `INTERVAL` (then the real end is `start + interval`). The result
    /// is a nullable `BOOLEAN` under three-valued logic.
    Overlaps {
        /// Start of the first period.
        s1: Box<Self>,
        /// End of the first period (temporal value or `INTERVAL`).
        e1: Box<Self>,
        /// Start of the second period.
        s2: Box<Self>,
        /// End of the second period (temporal value or `INTERVAL`).
        e2: Box<Self>,
    },
    /// `expr [NOT] LIKE pattern [ESCAPE 'c']` — SQL pattern match where `%` matches zero
    /// or more characters and `_` matches exactly one character. The optional `ESCAPE`
    /// character overrides the default backslash escape.
    Like {
        /// Subject string.
        expr: Box<Self>,
        /// Pattern string.
        pattern: Box<Self>,
        /// `true` for `NOT LIKE`, `false` for `LIKE`.
        negated: bool,
        /// Escape character: a single character that escapes the next `%` or `_` in the pattern.
        /// Defaults to backslash when there is no `ESCAPE` clause; `None` means escaping is disabled
        /// (an explicit `ESCAPE ''`).
        escape: Option<char>,
        /// `true` for `ILIKE` — letters match case-insensitively. Done in the matcher (not by
        /// lower-casing both sides), so a single `_` still matches one source character even when a
        /// letter's lowercase form has a different length (e.g. `'İ'`), and an alphabetic `ESCAPE`
        /// character keeps its meaning (deep-gate #12).
        case_insensitive: bool,
    },
    /// `CASE` expression — both **simple** form
    /// (`CASE x WHEN a THEN ra ELSE re END`, `operand = Some(x)`) and
    /// **searched** form (`CASE WHEN cond THEN r ELSE re END`,
    /// `operand = None`).
    Case {
        /// Optional value to compare each branch's `when` against
        /// (simple form); `None` for the searched form.
        operand: Option<Box<Self>>,
        /// `WHEN … THEN …` branches in order. For the simple form, `when`
        /// is the value to match; for the searched form, `when` is a
        /// boolean predicate.
        branches: Vec<CaseBranch>,
        /// Optional `ELSE` value; if missing and no branch matches, the
        /// result is `NULL`.
        default: Option<Box<Self>>,
    },
    /// `COALESCE(a, b, c, ...)` — returns the first argument that is not
    /// `NULL`. Returns `NULL` only if every argument is `NULL`.
    Coalesce(Vec<Self>),
    /// `CAST(expr AS T)` — convert `expr` to the target column type. `TRY_CAST`/`SAFE_CAST` use the
    /// same node with `try_cast = true` (a failed conversion yields `NULL` instead of erroring).
    Cast {
        /// The expression being converted.
        expr: Box<Self>,
        /// Target type (mapped onto the catalog's [`ColumnType`] vocabulary).
        target: ColumnType,
        /// `true` for `TRY_CAST`/`SAFE_CAST`: a failed conversion becomes `NULL` rather than an error.
        try_cast: bool,
    },
    /// Aggregate function call (`COUNT(*)`, `COUNT(expr)`, `SUM(expr)`,
    /// `AVG(expr)`, `MIN(expr)`, `MAX(expr)`). Without `GROUP BY` the whole
    /// table is one group and the result is a single row.
    Aggregate {
        /// Which aggregate.
        func: AggregateFunc,
        /// Argument expression; `None` only for `COUNT(*)`.
        arg: Option<Box<Self>>,
        /// `DISTINCT` modifier — deduplicate the input before aggregating.
        distinct: bool,
        /// `FILTER (WHERE pred)` — restrict aggregation to rows where `pred` is true.
        /// `None` when the clause is absent.
        filter: Option<Box<Self>>,
        /// The separator of `STRING_AGG(expr, separator)` — a second argument carried only for that
        /// aggregate; `None` for every other. Must be a constant string at analysis time.
        separator: Option<Box<Self>>,
        /// The second per-row argument of the two-argument statistical aggregates
        /// (`CORR`/`COVAR_POP`/`COVAR_SAMP`); `None` for every other aggregate.
        arg2: Option<Box<Self>>,
        /// `ORDER BY` inside the aggregate call, e.g. `array_agg(x ORDER BY y DESC)`. Only meaningful
        /// for the order-sensitive aggregates `ARRAY_AGG` / `STRING_AGG` (the parser rejects it on
        /// any other aggregate); empty when absent.
        order_by: Vec<OrderByItem>,
    },
    /// `encrypt(value, key)` — column-level encryption. Returns the
    /// hex-encoded ciphertext as `TEXT`, opaque to the storage engine.
    Encrypt {
        /// Plaintext value (`TEXT`).
        value: Box<Self>,
        /// Encryption key string (`TEXT`).
        key: Box<Self>,
    },
    /// `decrypt(ciphertext, key)` — inverse of [`Encrypt`](Self::Encrypt);
    /// returns the recovered plaintext as `TEXT`.
    Decrypt {
        /// Hex ciphertext produced by `encrypt` (`TEXT`).
        value: Box<Self>,
        /// Encryption key string (`TEXT`).
        key: Box<Self>,
    },
    /// A query parameter placeholder `$1`, `$2`, … stored zero-based (`$1` → `0`).
    /// Replaced with its bound literal by
    /// [`bind_parameters`](crate::bind_parameters) before analysis; reaching the
    /// analyzer unbound is an error.
    Parameter(usize),
    /// `(SELECT ...)` used as a scalar value. Must yield one column / one row at run time.
    ScalarSubquery(Box<Select>),
    /// `[NOT] EXISTS (SELECT ...)`.
    Exists {
        /// `true` for `NOT EXISTS`.
        negated: bool,
        /// The subquery whose row presence is tested.
        subquery: Box<Select>,
    },
    /// `expr [NOT] IN (SELECT ...)`.
    InSubquery {
        /// The probed expression.
        expr: Box<Self>,
        /// `true` for `NOT IN`.
        negated: bool,
        /// The single-column subquery providing the membership set.
        subquery: Box<Select>,
    },
    /// `expr <op> ANY/ALL ((subquery))` — a quantified comparison against a single-column
    /// subquery, for operators other than the `= ANY` (IN) / `<> ALL` (NOT IN) forms.
    QuantifiedComparison {
        /// The probed expression (left operand).
        expr: Box<Self>,
        /// The comparison operator.
        op: BinaryOp,
        /// `true` for `ALL` (every row), `false` for `ANY`/`SOME` (some row).
        all: bool,
        /// The single-column subquery.
        subquery: Box<Select>,
    },
    /// `expr <op> ANY/ALL (array)` — a quantified comparison against every element of a **runtime**
    /// array value (a column or expression). The `ARRAY[...]` literal form desugars to a comparison
    /// chain in the parser, and a subquery operand uses [`Self::QuantifiedComparison`]; this variant
    /// covers the array-value operand.
    QuantifiedArray {
        /// The probed expression (left operand).
        expr: Box<Self>,
        /// The comparison operator.
        op: BinaryOp,
        /// `true` for `ALL` (every element), `false` for `ANY`/`SOME` (some element).
        all: bool,
        /// The array-valued right operand, evaluated per row.
        array: Box<Self>,
    },
    /// `expr [NOT] SIMILAR TO pattern` — SQL-standard regular-expression match.
    SimilarTo {
        /// The probed expression.
        expr: Box<Self>,
        /// The `SIMILAR TO` pattern.
        pattern: Box<Self>,
        /// `true` for `NOT SIMILAR TO`.
        negated: bool,
    },
    /// `ARRAY[a, b, c]` / `[a, b, c]` — array constructor from element expressions.
    /// Distinct from [`Value::Array`], which is a constant array literal; here the elements
    /// are arbitrary expressions evaluated per row.
    ArrayLiteral(Vec<Self>),
    /// `base[index]` — one-dimensional array element access. Slice forms (`a[i:j]`)
    /// use [`Self::ArraySlice`].
    Subscript {
        /// The array-valued expression being indexed.
        base: Box<Self>,
        /// The index expression (1-based per SQL array semantics).
        index: Box<Self>,
    },
    /// `base[lower:upper]` — a 1-based inclusive array slice (B-fn). Either bound may be omitted
    /// (`a[2:]`, `a[:3]`, `a[:]`), defaulting to the array's first / last element. The result is an
    /// array of the same element type.
    ArraySlice {
        /// The array-valued expression being sliced.
        base: Box<Self>,
        /// The lower bound (1-based, inclusive), or `None` for the array start.
        lower: Option<Box<Self>>,
        /// The upper bound (1-based, inclusive), or `None` for the array end.
        upper: Option<Box<Self>>,
    },
    /// `expr ~ pattern` / `~*` / `!~` / `!~*` — POSIX regular-expression match.
    RegexMatch {
        /// The probed expression.
        expr: Box<Self>,
        /// The POSIX regex pattern.
        pattern: Box<Self>,
        /// `true` for `~` / `!~` (case-sensitive); `false` for `~*` / `!~*` (case-insensitive).
        case_sensitive: bool,
        /// `true` for the negated forms `!~` / `!~*`.
        negated: bool,
    },
    /// `ROW(a, b, ...)` / `(a, b, ...)` row-value constructor.
    Row(Vec<Self>),
    /// `(expr).field` — composite field access. The operand must denote a value of a user-defined
    /// composite type (a composite column or a cast to a composite type); the analyzer resolves the
    /// field by name against the type's declared fields.
    FieldAccess {
        /// The composite-typed operand.
        base: Box<Self>,
        /// The field name being selected.
        field: String,
    },
    /// `expr::T` where `T` is a user-defined type name the parser could not resolve to a built-in
    /// [`ColumnType`] (e.g. a composite type). The name is resolved by the analyzer against the type
    /// registry; anything but a composite type is rejected there. Kept separate from
    /// [`Cast`](Self::Cast) because a composite type has no `ColumnType` to name a target with.
    CastNamed {
        /// The expression being cast.
        expr: Box<Self>,
        /// The user-defined type name (folded).
        type_name: String,
        /// `true` for `TRY_CAST`/`SAFE_CAST`.
        try_cast: bool,
    },
    /// `func([args]) OVER (PARTITION BY … ORDER BY …)` — a window function call.
    /// Boxed to keep `Expr` variants uniform in size.
    WindowFunction(Box<WindowFunction>),
    /// `func(args) WITHIN GROUP (ORDER BY ...)` — an ordered-set aggregate, e.g.
    /// `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`. Boxed to keep variant sizes uniform.
    WithinGroup(Box<WithinGroup>),
    /// A scalar built-in function call resolved to a known [`ScalarFunc`] (+), e.g.
    /// `UPPER(name)` or `SUBSTRING(s, 2, 3)`. Distinct from [`Aggregate`](Self::Aggregate) and
    /// [`WindowFunction`](Self::WindowFunction): it is evaluated independently per row. Special
    /// syntactic forms (`SUBSTRING(s FROM a FOR b)`, `TRIM(BOTH c FROM s)`, `POSITION(a IN b)`)
    /// are normalized by the parser into positional `args`.
    ScalarFunction {
        /// Which built-in.
        func: ScalarFunc,
        /// Argument expressions, in positional order.
        args: Vec<Self>,
    },
    /// A set-returning function call, e.g. `UNNEST(arr)`. Unlike a scalar function it yields
    /// *zero or more* rows per input row, so it is only valid at the top level of a `SELECT`-list
    /// item; the analyzer rejects it elsewhere.
    SetReturning {
        /// Which set-returning built-in.
        func: SetReturningFunc,
        /// Argument expressions, in positional order.
        args: Vec<Self>,
    },
    /// A call to a function the parser does not recognise as a built-in, kept by name for the
    /// analyzer to resolve against the user-defined-function registry. Unknown names that
    /// have no registered UDF are rejected there with [`Error::UnknownFunction`](crate::Error).
    FunctionCall {
        /// The function name, folded to lowercase.
        name: String,
        /// Argument expressions, in positional order.
        args: Vec<Self>,
    },
}

/// A set-returning built-in function — one that produces multiple output rows from a single input
/// row. Executed by the `ProjectSet` operator rather than the scalar evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetReturningFunc {
    /// `UNNEST(arr)` — expand an array into one row per element, in array order.
    Unnest,
    /// `JSON_ARRAY_ELEMENTS(json)` / `JSONB_ARRAY_ELEMENTS(json)` — one row per element of a JSON
    /// array, each as `JSON`.
    JsonArrayElements,
    /// `JSONB_PATH_QUERY(json, path)` / `JSON_PATH_QUERY(json, path)` — one row per `jsonpath` match
    /// in the document, each as `JSON`.
    JsonPathQuery,
    /// `GENERATE_SERIES(start, stop [, step])` — one `INT` row per value in the inclusive series from
    /// `start` to `stop` stepping by `step` (default `1`).
    GenerateSeries,
    /// `JSONB_OBJECT_KEYS(json)` / `JSON_OBJECT_KEYS(json)` — one `TEXT` row per top-level field name
    /// of a JSON object.
    JsonObjectKeys,
    /// `REGEXP_SPLIT_TO_TABLE(s, pattern [, flags])` — one `TEXT` row per piece of `s` split on each
    /// match of `pattern` (the set-returning form of `REGEXP_SPLIT_TO_ARRAY`).
    RegexpSplitToTable,
    /// `REGEXP_MATCHES(s, pattern [, flags])` — one `TEXT[]` row per match's capture groups (the whole
    /// match when the pattern has no groups); the `g` flag returns every match, else only the first
    /// (0 or 1 rows). The set-returning form of `REGEXP_MATCH`.
    RegexpMatches,
    /// `STRING_TO_TABLE(s, sep)` — one `TEXT` row per piece of `s` split on the literal separator
    /// `sep` (the set-returning form of `STRING_TO_ARRAY`).
    StringToTable,
    /// `JSON_ARRAY_ELEMENTS_TEXT(json)` / `JSONB_ARRAY_ELEMENTS_TEXT(json)` — one `TEXT` row per
    /// element of a JSON array (a string element's raw contents, a JSON `null` as SQL `NULL`).
    JsonArrayElementsText,
    /// `JSONB_EACH(json)` / `JSON_EACH(json)` — one row per top-level member of a JSON object, as
    /// `(key TEXT, value JSON)`. A *pair* function: see [`SetReturningFunc::pair_value_type`].
    JsonEach,
    /// `JSONB_EACH_TEXT(json)` / `JSON_EACH_TEXT(json)` — like [`Self::JsonEach`] but the value is
    /// `TEXT` (a string member's raw contents, a JSON `null` as SQL `NULL`).
    JsonEachText,
}

impl SetReturningFunc {
    /// The canonical (folded) name, used in diagnostics and as the default output column name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unnest => "unnest",
            Self::JsonArrayElements => "json_array_elements",
            Self::JsonPathQuery => "jsonb_path_query",
            Self::GenerateSeries => "generate_series",
            Self::JsonObjectKeys => "jsonb_object_keys",
            Self::RegexpSplitToTable => "regexp_split_to_table",
            Self::RegexpMatches => "regexp_matches",
            Self::StringToTable => "string_to_table",
            Self::JsonArrayElementsText => "jsonb_array_elements_text",
            Self::JsonEach => "jsonb_each",
            Self::JsonEachText => "jsonb_each_text",
        }
    }

    /// The type of the **second** column a *pair* set-returning function produces, or `None` for the
    /// ordinary single-column ones.
    ///
    /// `jsonb_each` yields `(key, value)` rather than one value per row. The second column rides
    /// along the way `WITH ORDINALITY`'s counter does — appended by the `ProjectSet` operator — so
    /// the single-column shape the rest of the pipeline assumes stays intact. The first column is
    /// the key, always `TEXT`, and carries the function's ordinary element type.
    #[must_use]
    pub const fn pair_value_type(self) -> Option<nusadb_core::ColumnType> {
        match self {
            Self::JsonEach => Some(nusadb_core::ColumnType::Json),
            Self::JsonEachText => Some(nusadb_core::ColumnType::Text),
            _ => None,
        }
    }

    /// The names a pair function's two columns take when the relation's alias list does not rename
    /// them, matching the reference engine's `jsonb_each(json) AS (key, value)`.
    pub(crate) const PAIR_KEY_COLUMN: &'static str = "key";
    /// See [`Self::PAIR_KEY_COLUMN`].
    pub(crate) const PAIR_COLUMN: &'static str = "value";

    /// The default name each column this function appends *after* its primary output column takes
    /// when the relation's alias list does not rename it: `value` for a pair function (`jsonb_each`),
    /// otherwise the function's own name. A multi-array `UNNEST(a, b, ...)` appends one column per
    /// further array, each taking this same `unnest` name — the reference engine names them alike.
    #[must_use]
    pub(crate) const fn appended_column_name(self) -> &'static str {
        if self.pair_value_type().is_some() {
            Self::PAIR_COLUMN
        } else {
            self.name()
        }
    }
}

/// A scalar (non-aggregate, non-window) built-in function (+). Result and argument types are
/// validated by the analyzer; the executor evaluates each one per row with SQL `NULL` propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFunc {
    /// `LENGTH(s)` / `CHAR_LENGTH(s)` — number of characters in `s`.
    Length,
    /// `OCTET_LENGTH(s)` — number of bytes in `s` (its UTF-8 encoding) as `INT`.
    OctetLength,
    /// `BIT_LENGTH(s)` — number of bits in `s` (`8 × OCTET_LENGTH`) as `INT`.
    BitLength,
    /// `GROUPING(col [, ...])` — super-aggregate grouping indicator. Valid only inside an
    /// aggregated `SELECT` whose arguments are `GROUP BY` keys: it reports, as a bitmask (one bit per
    /// argument, leftmost = most-significant), which arguments were *grouped away* in the current
    /// `ROLLUP`/`CUBE`/`GROUPING SETS` super-aggregate row (`1` = aggregated away, `0` = present).
    /// The analyzer resolves it against the grouping sets and rewrites it before evaluation, so it is
    /// never evaluated as an ordinary per-row scalar; a plain `GROUP BY` always yields `0`. Result `INT`.
    Grouping,
    /// `UPPER(s)` — `s` folded to upper case.
    Upper,
    /// `LOWER(s)` — `s` folded to lower case.
    Lower,
    /// `SUBSTRING(s, start [, length])` — 1-based substring (also `SUBSTRING(s FROM a FOR b)`).
    Substring,
    /// `REPLACE(s, from, to)` — every occurrence of `from` in `s` replaced by `to`.
    Replace,
    /// `POSITION(sub IN s)` — 1-based index of the first `sub` in `s`, or `0` if absent.
    Position,
    /// `OVERLAY(s PLACING r FROM start [FOR len])` — replace `len` characters of `s` (default: the
    /// length of `r`) starting at 1-based `start` with `r`.
    Overlay,
    /// `LPAD(s, len [, fill])` — left-pad (or truncate) `s` to `len` characters.
    Lpad,
    /// `RPAD(s, len [, fill])` — right-pad (or truncate) `s` to `len` characters.
    Rpad,
    /// `LTRIM(s [, chars])` / `TRIM(LEADING [chars] FROM s)` — strip leading characters.
    LTrim,
    /// `RTRIM(s [, chars])` / `TRIM(TRAILING [chars] FROM s)` — strip trailing characters.
    RTrim,
    /// `TRIM(s [, chars])` / `TRIM([BOTH] [chars] FROM s)` — strip leading and trailing characters.
    BTrim,
    /// `CONCAT(a, b, ...)` — concatenate all arguments, skipping `NULL`s (variadic).
    Concat,
    /// `CONCAT_WS(sep, a, b, ...)` — join non-`NULL` arguments with `sep`; `NULL` `sep` → `NULL`
    /// (variadic).
    ConcatWs,
    /// `LEFT(s, n)` — first `n` characters (`n < 0`: all but the last `|n|`).
    Left,
    /// `RIGHT(s, n)` — last `n` characters (`n < 0`: all but the first `|n|`).
    Right,
    /// `SPLIT_PART(s, delim, n)` — the 1-based `n`th field of `s` split on `delim`.
    SplitPart,
    /// `REVERSE(s)` — characters of `s` in reverse order.
    Reverse,
    /// `STARTS_WITH(s, prefix)` — whether `s` begins with `prefix`, as `BOOL`.
    StartsWith,
    /// `ASCII(s)` — the Unicode code point of the first character of `s` as `INT`; `0` for the empty
    /// string.
    Ascii,
    /// `CHR(n)` — the one-character `TEXT` whose Unicode code point is `n`.
    Chr,
    /// `INITCAP(s)` — `s` with the first letter of each word upper-cased and the rest lower-cased
    /// (word boundary = any non-alphanumeric character).
    Initcap,
    /// `REPEAT(s, n)` — `s` concatenated `n` times (`n ≤ 0` → empty string).
    Repeat,
    /// `STRPOS(s, sub)` — 1-based index of the first `sub` in `s`, or `0` if absent. Like
    /// [`POSITION`](Self::Position) but with the haystack-first argument order.
    Strpos,
    /// `TRANSLATE(s, from, to)` — each character of `s` that appears in `from` replaced by the
    /// character at the same position in `to` (or dropped if `to` is shorter).
    Translate,
    /// `REGEXP_REPLACE(s, pattern, replacement [, flags])` — replace matches of `pattern` in `s`
    /// (first match, or all with the `g` flag); `\1`..`\9`/`\&` backreferences.
    RegexpReplace,
    /// `REGEXP_MATCH(s, pattern [, flags])` — `TEXT[]` of the first match's capture groups (or the
    /// whole match when the pattern has no groups), or `NULL` if there is no match.
    RegexpMatch,
    /// `REGEXP_LIKE(s, pattern [, flags])` — whether `pattern` matches anywhere in `s`, as `BOOL`.
    RegexpLike,
    /// `REGEXP_COUNT(s, pattern [, flags])` — number of non-overlapping matches of `pattern` in `s`,
    /// as `INT`.
    RegexpCount,
    /// `REGEXP_INSTR(s, pattern [, flags])` — 1-based character position of the first match of
    /// `pattern` in `s` (`0` if none), as `INT`.
    RegexpInstr,
    /// `REGEXP_SUBSTR(s, pattern [, flags])` — the first substring of `s` matching `pattern`, as
    /// `TEXT` (`NULL` if none).
    RegexpSubstr,
    /// `REGEXP_SPLIT_TO_ARRAY(s, pattern [, flags])` — split `s` on each match of `pattern`, as
    /// `TEXT[]`.
    RegexpSplitToArray,
    /// `NOW()` — the statement's wall-clock instant as `TIMESTAMPTZ`. Niladic; stable for
    /// every row of one statement.
    Now,
    /// `STATEMENT_TIMESTAMP()` — the current statement's start instant as `TIMESTAMPTZ`. In this
    /// engine `NOW()`/`CURRENT_TIMESTAMP` are already statement-scoped, so it is the same instant.
    StatementTimestamp,
    /// `LOCALTIMESTAMP` — the statement instant as a `TIMESTAMP` (no time zone); the same instant as
    /// `CURRENT_TIMESTAMP` but without the zone (the session zone is UTC).
    LocalTimestamp,
    /// `CURRENT_TIMESTAMP` — synonym for [`NOW()`](Self::Now), `TIMESTAMPTZ`. Niladic.
    CurrentTimestamp,
    /// `CURRENT_DATE` — the statement instant's calendar date as `DATE`. Niladic.
    CurrentDate,
    /// `CURRENT_TIME` — the statement instant's time of day as `TIME`. Niladic.
    CurrentTime,
    /// `CURRENT_USER` / `USER` — the session user as `TEXT` (RLS). Niladic; read from the
    /// session, stable for every row of one statement.
    CurrentUser,
    /// `SESSION_USER` — the session user as `TEXT` (RLS). Niladic. NusaDB does not yet
    /// distinguish `SET ROLE` from the login user, so it is a synonym of [`CURRENT_USER`](Self::CurrentUser).
    SessionUser,
    /// `current_setting(name)` — the value of session setting `name` as `TEXT`, or `NULL` if it is
    /// unset. Reads the session's `SET`/`RESET` store.
    CurrentSetting,
    /// `EXTRACT(field FROM source)` — a date/time field of `source` as `FLOAT`. The field
    /// is carried as a lowercase text-literal first argument; `source` is the temporal value.
    Extract,
    /// `DATE_TRUNC(field, source)` — `source` truncated down to the precision named by the text
    /// `field`, returning the same temporal type. A `DATE` source widens to `TIMESTAMPTZ` at
    /// midnight first, so the call is always `TIMESTAMP`- or `TIMESTAMPTZ`-typed.
    DateTrunc,
    /// `AGE(end, start)` / `AGE(value)` — the calendar interval between two instants (or between
    /// the statement date and `value`), as `INTERVAL`.
    Age,
    /// `<value> AT TIME ZONE <zone>` — desugared from the SQL operator. Converts a `TIMESTAMP` to a
    /// `TIMESTAMPTZ` (interpreting the wall-clock value as being in `zone`) or a `TIMESTAMPTZ` to a
    /// `TIMESTAMP` (the UTC instant rendered as wall-clock in `zone`). `zone` is a text offset
    /// (`UTC` or `±HH[:MM]`); named zones with DST are not supported.
    AtTimeZone,
    /// `TO_CHAR(temporal, format)` — render a temporal value as `TEXT` per a format string.
    ToChar,
    /// `TO_DATE(text, format)` — parse `text` per `format` into a `DATE`.
    ToDate,
    /// `TO_TIMESTAMP(text, format)` — parse `text` per `format` into a `TIMESTAMP`.
    ToTimestamp,
    /// `TO_NUMBER(text, format)` — parse a formatted number `text` into a `NUMERIC`, reading the
    /// digits, sign, and decimal point and ignoring group separators / currency / padding (B-fn).
    ToNumber,
    /// `MAKE_DATE(year, month, day)` — build a `DATE` from integer fields; errors on a non-existent
    /// calendar day.
    MakeDate,
    /// `MAKE_TIME(hour, minute, second)` — build a `TIME` from integer fields; errors on an
    /// out-of-range field.
    MakeTime,
    /// `MAKE_TIMESTAMP(year, month, day, hour, minute, second)` — build a `TIMESTAMP` from integer
    /// fields; errors on an invalid date or time.
    MakeTimestamp,
    /// `MAKE_TIMESTAMPTZ(year, month, day, hour, minute, second)` — build a `TIMESTAMPTZ` from
    /// integer fields (interpreted in the session time zone, UTC); errors on an invalid date/time.
    MakeTimestamptz,
    /// `MAKE_INTERVAL([years [, months [, weeks [, days [, hours [, mins [, secs]]]]]]])` — build an
    /// `INTERVAL` from the (positional) field values, each defaulting to `0`.
    MakeInterval,
    /// `JUSTIFY_DAYS(interval)` — normalize an `INTERVAL` so every 30 days becomes one month (B-fn).
    JustifyDays,
    /// `JUSTIFY_HOURS(interval)` — normalize an `INTERVAL` so every 24 hours becomes one day (B-fn).
    JustifyHours,
    /// `JUSTIFY_INTERVAL(interval)` — apply both `JUSTIFY_DAYS` and `JUSTIFY_HOURS` (B-fn).
    JustifyInterval,
    /// `SCALE(numeric)` — the declared scale (count of fractional digits) of a `NUMERIC` (B-fn).
    Scale,
    /// `MIN_SCALE(numeric)` — the fewest fractional digits needed to represent the value (B-fn).
    MinScale,
    /// `TRIM_SCALE(numeric)` — the value with trailing fractional zeros removed (B-fn).
    TrimScale,
    /// `ISFINITE(value)` — whether a `NUMERIC` / temporal value is finite (always true here, since no
    /// infinite values are representable); `NULL` propagates (B-fn).
    IsFinite,
    /// `ENCODE(bytea, format)` — render a `BYTEA` as `TEXT` in `hex`/`escape`/`base64` (B-fn).
    Encode,
    /// `DECODE(text, format)` — parse a `TEXT` in `hex`/`escape`/`base64` into a `BYTEA` (B-fn).
    Decode,
    /// `CONVERT_TO(text, encoding)` — the bytes of `text` in `encoding` as `BYTEA`. Only `UTF8` (the
    /// engine's native encoding) is supported; any other encoding is rejected.
    ConvertTo,
    /// `CONVERT_FROM(bytea, encoding)` — decode `bytea` in `encoding` to `TEXT`. Only `UTF8` is
    /// supported; any other encoding, or bytes not valid in it, is rejected.
    ConvertFrom,
    /// `DATE_BIN(stride, source, origin)` — snap `TIMESTAMP source` down to the start of its
    /// `stride`-wide bin aligned to `origin`, as `TIMESTAMP`; the stride must be a positive,
    /// fixed-duration interval (no months/years).
    DateBin,
    /// `ABS(x)` — absolute value, preserving the numeric type.
    Abs,
    /// `ROUND(x [, d])` — round to `d` decimal places (default 0), preserving the numeric type.
    Round,
    /// `CEIL(x)` / `CEILING(x)` — smallest integer ≥ `x`, preserving the numeric type.
    Ceil,
    /// `FLOOR(x)` — largest integer ≤ `x`, preserving the numeric type.
    Floor,
    /// `SIGN(x)` — `-1`/`0`/`1`, preserving the numeric type.
    Sign,
    /// `MOD(x, y)` — remainder of `x / y`, preserving the unified numeric type.
    Mod,
    /// `POWER(x, y)` / `POW(x, y)` — `x` raised to `y`, as `FLOAT`.
    Power,
    /// `SQRT(x)` — square root, as `FLOAT`.
    Sqrt,
    /// `LN(x)` — natural logarithm, as `FLOAT`.
    Ln,
    /// `LOG(x)` / `LOG(b, x)` — base-10 (one arg) or base-`b` logarithm, as `FLOAT`.
    Log,
    /// `LOG10(x)` — base-10 logarithm, as `FLOAT` (one argument only).
    Log10,
    /// `EXP(x)` — `e` raised to `x`, as `FLOAT`.
    Exp,
    /// `SIN(x)` — sine (radians), as `FLOAT`.
    Sin,
    /// `COS(x)` — cosine (radians), as `FLOAT`.
    Cos,
    /// `TAN(x)` — tangent (radians), as `FLOAT`.
    Tan,
    /// `ASIN(x)` — arcsine, as `FLOAT`.
    Asin,
    /// `ACOS(x)` — arccosine, as `FLOAT`.
    Acos,
    /// `ATAN(x)` — arctangent, as `FLOAT`.
    Atan,
    /// `ATAN2(y, x)` — two-argument arctangent, as `FLOAT`.
    Atan2,
    /// `COT(x)` — cotangent, as `FLOAT`.
    Cot,
    /// `CBRT(x)` — cube root, as `FLOAT`.
    Cbrt,
    /// `SINH(x)` — hyperbolic sine, as `FLOAT`.
    Sinh,
    /// `COSH(x)` — hyperbolic cosine, as `FLOAT`.
    Cosh,
    /// `TANH(x)` — hyperbolic tangent, as `FLOAT`.
    Tanh,
    /// `ASINH(x)` — inverse hyperbolic sine, as `FLOAT`.
    Asinh,
    /// `ACOSH(x)` — inverse hyperbolic cosine, as `FLOAT`.
    Acosh,
    /// `ATANH(x)` — inverse hyperbolic tangent, as `FLOAT`.
    Atanh,
    /// `GCD(a, b)` — greatest common divisor of two integers, as `INT`.
    Gcd,
    /// `LCM(a, b)` — least common multiple of two integers, as `INT`.
    Lcm,
    /// `DIV(a, b)` — integer quotient of `a / b` truncated toward zero, as `INT`.
    Div,
    /// `FACTORIAL(n)` — `n!` as `INT`; errors on a negative argument or on overflow (`n > 20`).
    Factorial,
    /// `BIT_COUNT(n)` — the number of set bits in the two's-complement 64-bit representation of the
    /// integer `n` (population count), as `INT`.
    BitCount,
    /// `TO_HEX(n)` — integer rendered as a lowercase hexadecimal `TEXT` string.
    ToHex,
    /// `WIDTH_BUCKET(operand, low, high, count)` — the 1-based histogram bucket `operand` falls into
    /// across `count` equi-width buckets spanning `[low, high)`, as `INT` (`0`/`count+1` for
    /// out-of-range values; SQL:2003).
    WidthBucket,
    /// `SHA224(text)` — lowercase-hex SHA-224 digest, as `TEXT`.
    Sha224,
    /// `SHA256(text)` — lowercase-hex SHA-256 digest, as `TEXT`.
    Sha256,
    /// `SHA384(text)` — lowercase-hex SHA-384 digest, as `TEXT`.
    Sha384,
    /// `SHA512(text)` — lowercase-hex SHA-512 digest, as `TEXT`.
    Sha512,
    /// `MD5(text)` — 32-character lowercase-hex MD5 digest, as `TEXT`. A non-security
    /// fingerprint (ETL dedup, cache keys); MD5 is cryptographically
    /// broken — use [`Self::Sha256`] for security.
    Md5,
    /// `QUOTE_LITERAL(text)` — `text` wrapped as a SQL string literal (embedded `'` doubled), as
    /// `TEXT`.
    QuoteLiteral,
    /// `QUOTE_NULLABLE(text)` — like `QUOTE_LITERAL`, but a NULL argument yields the unquoted text
    /// `NULL` (for splicing into dynamic SQL) rather than propagating NULL, as `TEXT`.
    QuoteNullable,
    /// `QUOTE_IDENT(text)` — `text` quoted as a SQL identifier when needed (embedded `"` doubled), as
    /// `TEXT`.
    QuoteIdent,
    /// `PARSE_IDENT(text [, strict bool])` — split a qualified identifier into its parts as `TEXT[]`.
    /// Unquoted parts are folded to lowercase; quoted parts keep their case. In `strict` mode (the
    /// default) trailing text after the last identifier is an error; otherwise it is ignored.
    ParseIdent,
    /// `FORMAT(fmt, ...)` — a format string with `%s` (text), `%I` (identifier), `%L` (literal), and
    /// `%%` (a literal `%`) specifiers substituted by the trailing arguments in order, as `TEXT`
    /// (B-fn).
    Format,
    /// `NUM_NONNULLS(...)` — count of the non-`NULL` arguments, as `INT` (any argument types).
    NumNonNulls,
    /// `NUM_NULLS(...)` — count of the `NULL` arguments, as `INT` (any argument types).
    NumNulls,
    /// `DEGREES(x)` — radians converted to degrees, as `FLOAT`.
    Degrees,
    /// `RADIANS(x)` — degrees converted to radians, as `FLOAT`.
    Radians,
    /// `PI()` — the constant π, as `FLOAT`. Niladic.
    Pi,
    /// `TRUNC(x [, d])` — truncate toward zero to `d` decimal places (default 0), preserving the
    /// numeric type. Unlike `ROUND` it discards the remaining fraction rather than rounding.
    Trunc,
    /// `RANDOM()` — a uniform `FLOAT` in `[0, 1)`; volatile (a fresh value per call).
    Random,
    /// `SETSEED(x)` — pin the `RANDOM()` generator seed (`x` clamped to `[-1, 1]`); returns `BOOL`
    /// `true` once the seed is applied.
    Setseed,
    /// `NULLIF(a, b)` — `NULL` when `a = b`, otherwise `a`.
    Nullif,
    /// `GREATEST(a, b, ...)` — the largest non-`NULL` argument, or `NULL` if all are `NULL`.
    Greatest,
    /// `LEAST(a, b, ...)` — the smallest non-`NULL` argument, or `NULL` if all are `NULL`.
    Least,
    /// `CARDINALITY(arr)` — the number of elements in array `arr` as `INT`; `0` for an empty array,
    /// `NULL` for a `NULL` array.
    Cardinality,
    /// `ARRAY_LENGTH(arr, dim)` — the length of array `arr` along dimension `dim` as `INT`; `NULL`
    /// for an empty array or `dim ≠ 1` (only 1-D arrays are supported).
    ArrayLength,
    /// `ARRAY_FILL(value, dims)` — a one-dimensional array of `value` repeated `dims[1]` times (only
    /// 1-D `dims` supported); a NULL `value` fills with NULLs. The result element type is `value`'s.
    ArrayFill,
    /// `ARRAY_LOWER(arr, dim)` — the lower bound of array `arr` along dimension `dim` (always `1` for
    /// a non-empty 1-D array), as `INT`; `NULL` for an empty array or `dim ≠ 1`.
    ArrayLower,
    /// `ARRAY_UPPER(arr, dim)` — the upper bound of array `arr` along dimension `dim` (the length for
    /// a 1-D array), as `INT`; `NULL` for an empty array or `dim ≠ 1`.
    ArrayUpper,
    /// `ARRAY_DIMS(arr)` — a text description of the array's dimensions (`[1:n]` for a non-empty 1-D
    /// array), as `TEXT`; `NULL` for an empty array.
    ArrayDims,
    /// `ARRAY_TO_STRING(arr, sep)` — concatenate the non-`NULL` elements of `arr` as text, joined by
    /// `sep`, into `TEXT`.
    ArrayToString,
    /// `STRING_TO_ARRAY(s, sep)` — split `s` on `sep` into a `TEXT[]`; an empty `sep` yields a
    /// one-element array.
    StringToArray,
    /// `ARRAY_APPEND(arr, elem)` — `arr` with `elem` appended; result keeps `arr`'s type.
    ArrayAppend,
    /// `ARRAY_PREPEND(elem, arr)` — `arr` with `elem` prepended; result keeps `arr`'s type.
    ArrayPrepend,
    /// `ARRAY_CAT(a, b)` — the concatenation of two same-element-type arrays.
    ArrayCat,
    /// `ARRAY_POSITION(arr, elem)` — the 1-based index of the first `elem` in `arr` as `INT`, or
    /// `NULL` if absent (or `arr` is `NULL`).
    ArrayPosition,
    /// `ARRAY_REMOVE(arr, elem)` — `arr` with every element equal to `elem` removed; result keeps
    /// `arr`'s type.
    ArrayRemove,
    /// `ARRAY_REPLACE(arr, from, to)` — `arr` with every element equal to `from` replaced by `to`;
    /// result keeps `arr`'s type (B-fn).
    ArrayReplace,
    /// `TRIM_ARRAY(arr, n)` — `arr` with its last `n` elements removed; result keeps `arr`'s type.
    /// `n` must be between 0 and the array's length.
    TrimArray,
    /// `ARRAY_POSITIONS(arr, elem)` — an `INT[]` of every 1-based index where `elem` occurs in `arr`
    /// (empty array if none; `NULL` if `arr` is `NULL`) (B-fn).
    ArrayPositions,
    /// `ARRAY_NDIMS(arr)` — the number of array dimensions (always `1` here for a non-empty array;
    /// `NULL` for an empty array), as `INT` (B-fn).
    ArrayNdims,
    /// `L2_DISTANCE(a, b)` — Euclidean distance between two `VECTOR(n)`s, as `FLOAT`.
    L2Distance,
    /// `COSINE_DISTANCE(a, b)` — cosine distance `1 − cosθ` between two `VECTOR(n)`s, as `FLOAT`;
    /// the metric bound to the `<=>` operator.
    CosineDistance,
    /// `INNER_PRODUCT(a, b)` — the raw (positive) dot product `a · b` of two `VECTOR(n)`s, as `FLOAT`.
    InnerProduct,
    /// `L1_DISTANCE(a, b)` — Manhattan distance `Σ |aᵢ − bᵢ|` between two `VECTOR(n)`s, as `FLOAT`.
    L1Distance,
    /// `VECTOR_DIMS(a)` — the number of dimensions of a `VECTOR(n)`, as `INT`.
    VectorDims,
    /// `VECTOR_NORM(a)` — the Euclidean norm `‖a‖₂` of a `VECTOR(n)`, as `FLOAT`.
    VectorNorm,
    /// `HOST(inet)` — the host address without a mask, as `TEXT`.
    InetHost,
    /// `MASKLEN(inet)` — the network mask length in bits, as `INT`.
    InetMasklen,
    /// `FAMILY(inet)` — the address family (`4` or `6`), as `INT`.
    InetFamily,
    /// `NETWORK(inet)` — the network part (host bits zeroed), as `CIDR`.
    InetNetwork,
    /// `BROADCAST(inet)` — the broadcast address of the network, as `INET`.
    InetBroadcast,
    /// `SET_MASKLEN(inet, int)` — the address with a new mask length, as `INET`/`CIDR`.
    InetSetMasklen,
    /// `NETMASK(inet)` — the network mask (prefix bits set), as `INET`.
    InetNetmask,
    /// `HOSTMASK(inet)` — the host mask (suffix bits set), as `INET`.
    InetHostmask,
    /// `ABBREV(inet)` — the abbreviated text form (drops a `CIDR`'s trailing zero groups), as `TEXT`.
    InetAbbrev,
    /// `INET_MERGE(inet, inet)` — the smallest network containing both, as `CIDR`.
    InetMerge,
    /// `INET_SAME_FAMILY(inet, inet)` — whether both are the same address family, as `BOOLEAN`.
    InetSameFamily,
    /// `GET_BIT(bits, n)` — the `n`-th bit (0-based from the left) as `INT` (`0`/`1`).
    BitGetBit,
    /// `SET_BIT(bits, n, v)` — `bits` with its `n`-th bit set to `v` (`0`/`1`), as the same bit type.
    BitSetBit,
    /// `GET_BYTE(bytea, n)` — the `n`-th byte (0-based) of a binary string as `INT` (`0`–`255`).
    GetByte,
    /// `SET_BYTE(bytea, n, v)` — the binary string with its `n`-th byte set to `v` (low 8 bits), as
    /// `BYTEA`.
    SetByte,
    /// `LOWER(range)` — the lower bound, as the element type; `NULL` when empty or unbounded below.
    /// Spelled `lower`, like the text-folding [`Self::Lower`]; the analyzer picks between them by
    /// argument type.
    RangeLower,
    /// `UPPER(range)` — the upper bound, as the element type; `NULL` when empty or unbounded above.
    /// Spelled `upper`, like the text-folding [`Self::Upper`]; resolved the same way.
    RangeUpper,
    /// `ISEMPTY(range)` — whether the range contains no points, as `BOOL`.
    RangeIsEmpty,
    /// `LOWER_INC(range)` — whether the lower bound is inclusive, as `BOOL` (false when empty or
    /// unbounded below).
    RangeLowerInc,
    /// `UPPER_INC(range)` — whether the upper bound is inclusive, as `BOOL` (false when empty or
    /// unbounded above).
    RangeUpperInc,
    /// `LOWER_INF(range)` — whether the range is unbounded below, as `BOOL` (false when empty).
    RangeLowerInf,
    /// `UPPER_INF(range)` — whether the range is unbounded above, as `BOOL` (false when empty).
    RangeUpperInf,
    /// `INT4RANGE(lo, hi [, bounds])` — build an integer range. A `NULL` bound is unbounded on that
    /// side (not a `NULL` result); `bounds` is `'[)'` (the default), `'(]'`, `'[]'`, or `'()'`.
    Int4Range,
    /// `INT8RANGE(lo, hi [, bounds])` — as [`Self::Int4Range`]; the same integer element kind, kept
    /// apart only so the call renders back under the name it was written with.
    Int8Range,
    /// `NUMRANGE(lo, hi [, bounds])` — build an exact-decimal range.
    NumRange,
    /// `DATERANGE(lo, hi [, bounds])` — build a date range.
    DateRange,
    /// `TSRANGE(lo, hi [, bounds])` — build a timestamp range.
    TsRange,
    /// `TSTZRANGE(lo, hi [, bounds])` — build a timestamp-with-time-zone range.
    TsTzRange,
    /// `RANGE_MERGE(range, range)` — the smallest range covering both inputs, spanning any gap
    /// between them (unlike `+`, which errors on a gap). Both inputs share an element kind; the result
    /// is that range type. An empty input contributes nothing to the result.
    RangeMerge,
    /// `GEN_RANDOM_UUID()` / `UUID_GENERATE_V4()` — a random UUID v4 as `UUID`. Niladic;
    /// a fresh value per call. Both spellings parse to this variant.
    UuidGenerateV4,
    /// `VERSION()` — the NusaDB server version string as `TEXT`. Niladic.
    Version,
    /// `NUSADB_TYPEOF(expr)` — the SQL type name of `expr` as `TEXT` (e.g. `integer`, `text`,
    /// `numeric`). NusaDB's static type-introspection builtin. The type is known statically, so the
    /// analyzer folds this to a constant `TEXT` literal; it never reaches the executor.
    NusadbTypeof,
    /// `CURRENT_DATABASE()` — the name of the current database as `TEXT`. Niladic; read from
    /// the session, stable for every row of one statement.
    CurrentDatabase,
    /// `CURRENT_SCHEMA()` — the name of the current schema (namespace) as `TEXT`. Niladic;
    /// read from the session, stable for every row of one statement.
    CurrentSchema,
    /// `JSON_TYPEOF(json)` / `JSONB_TYPEOF(json)` — the JSON type name as `TEXT`
    /// (`null`/`boolean`/`number`/`string`/`array`/`object`).
    JsonTypeof,
    /// `JSON_ARRAY_LENGTH(json)` / `JSONB_ARRAY_LENGTH(json)` — the element count of a JSON array as
    /// `INT`; `NULL` if `json` is not an array.
    JsonArrayLength,
    /// `TO_JSON(value)` / `TO_JSONB(value)` — convert any value to its JSON representation.
    ToJson,
    /// `JSON_BUILD_OBJECT(k1, v1, ...)` / `JSONB_BUILD_OBJECT(...)` — build a JSON object from
    /// alternating key/value arguments (even arity; keys become text).
    JsonBuildObject,
    /// `JSON_OBJECT(pairs)` / `JSON_OBJECT(keys, values)` — build a JSON object from a flat key/value
    /// text array (even length) or from parallel key/value text arrays. Values are stored as JSON
    /// strings; a NULL value becomes JSON `null`; a NULL key errors. Result `JSON`.
    JsonObject,
    /// `JSON_BUILD_ARRAY(v1, v2, ...)` / `JSONB_BUILD_ARRAY(...)` — build a JSON array from the
    /// arguments in order (any arity, including none; each value at its natural type).
    JsonBuildArray,
    /// `ROW_TO_JSON(row(...))` — serialize a row-value constructor to a JSON object, naming the
    /// fields `f1`, `f2`, … in order, as `JSON`. Only the `ROW(...)` / `(a, b, …)` form is
    /// supported; whole-table-row references (`row_to_json(t)` with real column names) are not yet.
    RowToJson,
    /// `JSONB_EXTRACT_PATH(json, VARIADIC path text)` / `JSON_EXTRACT_PATH(...)` — the value at the
    /// key/index path, as `JSON` (the variadic form of `#>`); `NULL` if any step is missing.
    JsonExtractPath,
    /// `JSONB_EXTRACT_PATH_TEXT(json, VARIADIC path text)` / `JSON_EXTRACT_PATH_TEXT(...)` — the value
    /// at the path as `TEXT` (the variadic form of `#>>`; a JSON `null` becomes SQL `NULL`).
    JsonExtractPathText,
    /// `JSONB_SET(target, path, new_value [, create_missing])` — replace the value at `path` (a
    /// `TEXT[]` of object keys / array indices) with `new_value`.
    JsonbSet,
    /// `JSONB_SET_LAX(target, path, new_value [, create_missing [, null_value_treatment]])` — like
    /// `JSONB_SET`, but when `new_value` is SQL `NULL` the `null_value_treatment` decides the outcome:
    /// `use_json_null` (default), `delete_key`, `return_target`, or `raise_exception`.
    JsonbSetLax,
    /// `XML_IS_WELL_FORMED(text)` — whether `text` is well-formed XML content, as `BOOL` (the
    /// session default is CONTENT, so this matches [`Self::XmlIsWellFormedContent`]).
    XmlIsWellFormed,
    /// `XML_IS_WELL_FORMED_DOCUMENT(text)` — whether `text` is a well-formed XML document (single
    /// root), as `BOOL`.
    XmlIsWellFormedDocument,
    /// `XML_IS_WELL_FORMED_CONTENT(text)` — whether `text` is well-formed XML content (a fragment;
    /// bare text and multiple top-level nodes allowed), as `BOOL`.
    XmlIsWellFormedContent,
    /// `JSONB_STRIP_NULLS(json)` — recursively drop object members whose value is JSON `null`, as
    /// `JSON`.
    JsonbStripNulls,
    /// `JSONB_PRETTY(json)` — the JSON re-serialized with indentation, as `TEXT`.
    JsonbPretty,
    /// `JSONB_PATH_EXISTS(json, path)` / `JSON_PATH_EXISTS(json, path)` — whether the `jsonpath`
    /// matches anywhere in the document, as `BOOL`.
    JsonbPathExists,
    /// `JSONB_PATH_MATCH(json, path)` / `JSON_PATH_MATCH(json, path)` — the boolean result of a
    /// predicate-check `jsonpath` against the document, as `BOOL` (or `NULL` when the predicate is
    /// unknown or the path is not a single boolean value).
    JsonbPathMatch,
    /// `JSONB_INSERT(target, path, new_value [, insert_after])` — insert `new_value` at `path` without
    /// overwriting an existing value, as `JSON`.
    JsonbInsert,
    /// `JSONB_PATH_QUERY_FIRST(json, path)` / `JSON_PATH_QUERY_FIRST(json, path)` — the first
    /// `jsonpath` match in the document as `JSON`, or `NULL` if there is none.
    JsonbPathQueryFirst,
    /// `JSONB_PATH_QUERY_ARRAY(json, path)` / `JSON_PATH_QUERY_ARRAY(json, path)` — every `jsonpath`
    /// match collected into a JSON array (an empty array when there is no match).
    JsonbPathQueryArray,
    /// `JSONB_EXISTS(json, key)` — whether `key` is a top-level object key, an array string element,
    /// or equals a scalar string, as `BOOL`. The function form of the `?` operator, which the
    /// tokenizer cannot expose as an operator (it reserves `?` for parameters) (Q-jsonb-exists).
    JsonbExists,
    /// `TO_TSVECTOR([config,] text)` — tokenize `text` into the canonical `tsvector` text form
    /// (full-text search F1). Only the `simple` configuration is implemented.
    ToTsvector,
    /// `TO_TSQUERY([config,] text)` — parse a boolean lexeme query into the canonical `tsquery`
    /// text form (F1).
    ToTsquery,
    /// `PLAINTO_TSQUERY([config,] text)` — tokenize plain text into an AND-of-lexemes `tsquery`
    /// (F1).
    PlaintoTsquery,
    /// `TS_RANK(tsvector, tsquery [, normalization])` — the term-frequency relevance score as a
    /// `REAL`.
    TsRank,
    /// `TS_RANK_CD(tsvector, tsquery [, normalization])` — the cover-density relevance score as a
    /// `REAL`.
    TsRankCd,
    /// `NUMNODE(tsquery)` — the number of nodes (lexemes plus operators) in a `tsquery`, as `INT`
    /// (F1).
    Numnode,
    /// `STRIP(tsvector)` — the `tsvector` with positions and weights removed, keeping the distinct
    /// lexemes only (F1).
    Strip,
    /// `SETWEIGHT(tsvector, weight)` — the `tsvector` with every position's weight class set to
    /// `weight` (`A`/`B`/`C`/`D`) (F1).
    Setweight,
    /// `RRF_SCORE(rank [, k])` — the Reciprocal Rank Fusion contribution `1/(k + rank)` as a
    /// `FLOAT`, `k` defaulting to 60 (the standard constant). Summed across ranked lists (each
    /// rank from `RANK() OVER (...)`) to fuse FTS and vector rankings in hybrid search.
    RrfScore,
    /// `NEXTVAL(seq)` — advance sequence `seq` and return its new value as `INT` (`BIGINT`). Its sole
    /// argument is the sequence name as a text literal. Side-effecting: each call advances the
    /// sequence, so it is only valid where the executor evaluates it exactly once (a `SELECT` with no
    /// `FROM`, or a `VALUES` tuple); a per-row scan context rejects it rather than under-advancing.
    SequenceNext,
    /// `CURRVAL(seq)` — the current value of sequence `seq` in this session (the value the last
    /// [`NEXTVAL`](Self::SequenceNext) returned), as `INT`. Errors if `NEXTVAL` has not been called.
    /// Read-only, but — like the advancing sequence built-ins — resolved only where the executor
    /// evaluates it exactly once (a `SELECT` with no `FROM`, or a `VALUES` tuple); a per-row scan
    /// context rejects it too, keeping one uniform rule for every sequence function.
    SequenceCurrent,
    /// `SETVAL(seq, value [, is_called])` — set sequence `seq`'s current value to `value` and return
    /// `value` as `INT`. With `is_called = false` the *next* [`NEXTVAL`](Self::SequenceNext) returns
    /// `value` itself; the default `true` returns `value + increment`. Side-effecting like `NEXTVAL`.
    SequenceSet,
    /// `MACADDR8_SET7BIT(macaddr8)` — set the `0x02` (locally-administered) bit of the first byte,
    /// returning the modified `MACADDR8`.
    Macaddr8Set7bit,
    /// `POINT(x, y)` — construct a `point` from two `FLOAT` coordinates.
    PointCtor,
    /// `BOX(p1, p2)` — construct a `box` from two `point` corners.
    BoxCtor,
    /// `AREA(box)` / `AREA(circle)` — the area of a box (`width × height`) or a circle (`π · r²`),
    /// as `FLOAT`.
    GeomArea,
    /// `CENTER(box)` / `CENTER(circle)` — the center of a box (its midpoint) or a circle, as a
    /// `point`.
    GeomCenter,
    /// `HEIGHT(box)` — the height of a box, as `FLOAT`.
    GeomHeight,
    /// `WIDTH(box)` — the width of a box, as `FLOAT`.
    GeomWidth,
    /// `RADIUS(circle)` — the radius of a circle, as `FLOAT`.
    GeomRadius,
    /// `DIAMETER(circle)` — the diameter (`2 · r`) of a circle, as `FLOAT`.
    GeomDiameter,
    /// `NPOINTS(path)` — the number of vertices in a path, as `INT`.
    GeomNpoints,
    /// `ISOPEN(path)` — whether a path is an open polyline, as `BOOL`.
    GeomIsOpen,
    /// `ISCLOSED(path)` — whether a path is a closed loop, as `BOOL`.
    GeomIsClosed,
}

impl ScalarFunc {
    /// The element kind this range constructor builds, or `None` if it is not one.
    #[must_use]
    pub const fn range_kind(self) -> Option<nusadb_core::engine::RangeKind> {
        use nusadb_core::engine::RangeKind;
        Some(match self {
            // `int4range` and `int8range` build the same integer kind; only the spelling differs.
            Self::Int4Range | Self::Int8Range => RangeKind::Int,
            Self::NumRange => RangeKind::Num,
            Self::DateRange => RangeKind::Date,
            Self::TsRange => RangeKind::Ts,
            Self::TsTzRange => RangeKind::TsTz,
            _ => return None,
        })
    }

    /// The canonical (folded) name, used in diagnostics.
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "flat one-arm-per-variant name table; splitting it would scatter the mapping"
    )]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::OctetLength => "octet_length",
            Self::BitLength => "bit_length",
            Self::Grouping => "grouping",
            // The range accessors share their spelling with the text-folding pair; the analyzer
            // tells them apart by argument type, and both render back to the same name.
            Self::Upper | Self::RangeUpper => "upper",
            Self::Lower | Self::RangeLower => "lower",
            Self::Substring => "substring",
            Self::Replace => "replace",
            Self::Position => "position",
            Self::Overlay => "overlay",
            Self::Lpad => "lpad",
            Self::Rpad => "rpad",
            Self::LTrim => "ltrim",
            Self::RTrim => "rtrim",
            Self::BTrim => "btrim",
            Self::Concat => "concat",
            Self::ConcatWs => "concat_ws",
            Self::Left => "left",
            Self::Right => "right",
            Self::SplitPart => "split_part",
            Self::Reverse => "reverse",
            Self::Ascii => "ascii",
            Self::Chr => "chr",
            Self::Initcap => "initcap",
            Self::Repeat => "repeat",
            Self::Strpos => "strpos",
            Self::Translate => "translate",
            Self::RegexpReplace => "regexp_replace",
            Self::RegexpMatch => "regexp_match",
            Self::RegexpLike => "regexp_like",
            Self::RegexpCount => "regexp_count",
            Self::RegexpInstr => "regexp_instr",
            Self::RegexpSubstr => "regexp_substr",
            Self::RegexpSplitToArray => "regexp_split_to_array",
            Self::Now => "now",
            Self::StatementTimestamp => "statement_timestamp",
            Self::LocalTimestamp => "localtimestamp",
            Self::CurrentTimestamp => "current_timestamp",
            Self::CurrentDate => "current_date",
            Self::CurrentTime => "current_time",
            Self::CurrentUser => "current_user",
            Self::SessionUser => "session_user",
            Self::CurrentSetting => "current_setting",
            Self::Extract => "extract",
            Self::DateTrunc => "date_trunc",
            Self::Age => "age",
            Self::AtTimeZone => "at_time_zone",
            Self::ToChar => "to_char",
            Self::ToDate => "to_date",
            Self::ToTimestamp => "to_timestamp",
            Self::ToNumber => "to_number",
            Self::MakeDate => "make_date",
            Self::MakeTime => "make_time",
            Self::MakeTimestamp => "make_timestamp",
            Self::MakeTimestamptz => "make_timestamptz",
            Self::MakeInterval => "make_interval",
            Self::JustifyDays => "justify_days",
            Self::JustifyHours => "justify_hours",
            Self::JustifyInterval => "justify_interval",
            Self::Scale => "scale",
            Self::MinScale => "min_scale",
            Self::TrimScale => "trim_scale",
            Self::IsFinite => "isfinite",
            Self::Encode => "encode",
            Self::Decode => "decode",
            Self::ConvertTo => "convert_to",
            Self::ConvertFrom => "convert_from",
            Self::DateBin => "date_bin",
            Self::Abs => "abs",
            Self::Round => "round",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Sign => "sign",
            Self::Mod => "mod",
            Self::Power => "power",
            Self::Sqrt => "sqrt",
            Self::Ln => "ln",
            Self::Log => "log",
            Self::Log10 => "log10",
            Self::Exp => "exp",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Cot => "cot",
            Self::Cbrt => "cbrt",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Gcd => "gcd",
            Self::Lcm => "lcm",
            Self::Div => "div",
            Self::Factorial => "factorial",
            Self::BitCount => "bit_count",
            Self::ToHex => "to_hex",
            Self::WidthBucket => "width_bucket",
            Self::NumNonNulls => "num_nonnulls",
            Self::NumNulls => "num_nulls",
            Self::Sha224 => "sha224",
            Self::Sha256 => "sha256",
            Self::Sha384 => "sha384",
            Self::Sha512 => "sha512",
            Self::Md5 => "md5",
            Self::QuoteLiteral => "quote_literal",
            Self::QuoteNullable => "quote_nullable",
            Self::QuoteIdent => "quote_ident",
            Self::ParseIdent => "parse_ident",
            Self::Format => "format",
            Self::StartsWith => "starts_with",
            Self::Degrees => "degrees",
            Self::Radians => "radians",
            Self::Pi => "pi",
            Self::Trunc => "trunc",
            Self::Random => "random",
            Self::Setseed => "setseed",
            Self::Nullif => "nullif",
            Self::Greatest => "greatest",
            Self::Least => "least",
            Self::Cardinality => "cardinality",
            Self::ArrayLength => "array_length",
            Self::ArrayFill => "array_fill",
            Self::ArrayLower => "array_lower",
            Self::ArrayUpper => "array_upper",
            Self::ArrayDims => "array_dims",
            Self::ArrayToString => "array_to_string",
            Self::StringToArray => "string_to_array",
            Self::ArrayAppend => "array_append",
            Self::ArrayPrepend => "array_prepend",
            Self::ArrayCat => "array_cat",
            Self::ArrayPosition => "array_position",
            Self::ArrayRemove => "array_remove",
            Self::TrimArray => "trim_array",
            Self::ArrayReplace => "array_replace",
            Self::ArrayPositions => "array_positions",
            Self::ArrayNdims => "array_ndims",
            Self::L2Distance => "l2_distance",
            Self::CosineDistance => "cosine_distance",
            Self::InnerProduct => "inner_product",
            Self::L1Distance => "l1_distance",
            Self::VectorDims => "vector_dims",
            Self::VectorNorm => "vector_norm",
            Self::InetHost => "host",
            Self::InetMasklen => "masklen",
            Self::InetFamily => "family",
            Self::InetNetwork => "network",
            Self::InetBroadcast => "broadcast",
            Self::InetSetMasklen => "set_masklen",
            Self::InetNetmask => "netmask",
            Self::InetHostmask => "hostmask",
            Self::InetAbbrev => "abbrev",
            Self::InetMerge => "inet_merge",
            Self::InetSameFamily => "inet_same_family",
            Self::BitGetBit => "get_bit",
            Self::BitSetBit => "set_bit",
            Self::GetByte => "get_byte",
            Self::SetByte => "set_byte",
            Self::RangeIsEmpty => "isempty",
            Self::Int4Range => "int4range",
            Self::Int8Range => "int8range",
            Self::NumRange => "numrange",
            Self::DateRange => "daterange",
            Self::TsRange => "tsrange",
            Self::TsTzRange => "tstzrange",
            Self::RangeMerge => "range_merge",
            Self::RangeLowerInc => "lower_inc",
            Self::RangeUpperInc => "upper_inc",
            Self::RangeLowerInf => "lower_inf",
            Self::RangeUpperInf => "upper_inf",
            Self::UuidGenerateV4 => "uuid_generate_v4",
            Self::Version => "version",
            Self::NusadbTypeof => "nusadb_typeof",
            Self::CurrentDatabase => "current_database",
            Self::CurrentSchema => "current_schema",
            Self::JsonTypeof => "json_typeof",
            Self::JsonArrayLength => "json_array_length",
            Self::ToJson => "to_json",
            Self::RowToJson => "row_to_json",
            Self::JsonBuildObject => "json_build_object",
            Self::JsonObject => "json_object",
            Self::JsonBuildArray => "json_build_array",
            Self::JsonExtractPath => "jsonb_extract_path",
            Self::JsonExtractPathText => "jsonb_extract_path_text",
            Self::JsonbSet => "jsonb_set",
            Self::JsonbSetLax => "jsonb_set_lax",
            Self::XmlIsWellFormed => "xml_is_well_formed",
            Self::XmlIsWellFormedDocument => "xml_is_well_formed_document",
            Self::XmlIsWellFormedContent => "xml_is_well_formed_content",
            Self::JsonbStripNulls => "jsonb_strip_nulls",
            Self::JsonbPretty => "jsonb_pretty",
            Self::JsonbPathExists => "jsonb_path_exists",
            Self::JsonbPathMatch => "jsonb_path_match",
            Self::JsonbInsert => "jsonb_insert",
            Self::JsonbPathQueryFirst => "jsonb_path_query_first",
            Self::JsonbPathQueryArray => "jsonb_path_query_array",
            Self::JsonbExists => "jsonb_exists",
            Self::ToTsvector => "to_tsvector",
            Self::ToTsquery => "to_tsquery",
            Self::PlaintoTsquery => "plainto_tsquery",
            Self::TsRank => "ts_rank",
            Self::TsRankCd => "ts_rank_cd",
            Self::Numnode => "numnode",
            Self::Strip => "strip",
            Self::Setweight => "setweight",
            Self::RrfScore => "rrf_score",
            Self::SequenceNext => "nextval",
            Self::SequenceCurrent => "currval",
            Self::SequenceSet => "setval",
            Self::Macaddr8Set7bit => "macaddr8_set7bit",
            Self::PointCtor => "point",
            Self::BoxCtor => "box",
            Self::GeomArea => "area",
            Self::GeomCenter => "center",
            Self::GeomHeight => "height",
            Self::GeomWidth => "width",
            Self::GeomRadius => "radius",
            Self::GeomDiameter => "diameter",
            Self::GeomNpoints => "npoints",
            Self::GeomIsOpen => "isopen",
            Self::GeomIsClosed => "isclosed",
        }
    }

    /// Whether this is a sequence built-in (`NEXTVAL`, `CURRVAL`, `SETVAL`) — resolved against the
    /// engine's sequence store rather than by the pure per-row evaluator.
    #[must_use]
    pub const fn is_sequence(self) -> bool {
        matches!(
            self,
            Self::SequenceNext | Self::SequenceCurrent | Self::SequenceSet
        )
    }

    /// Whether this is a niladic clock function (`NOW`, `CURRENT_TIMESTAMP`, `CURRENT_DATE`,
    /// `CURRENT_TIME`) — resolved from the statement's wall clock rather than from arguments.
    #[must_use]
    pub const fn is_clock(self) -> bool {
        matches!(
            self,
            Self::Now
                | Self::CurrentTimestamp
                | Self::StatementTimestamp
                | Self::LocalTimestamp
                | Self::CurrentDate
                | Self::CurrentTime
        )
    }

    /// Whether this is a niladic session-user built-in (`CURRENT_USER`, `SESSION_USER`) — resolved
    /// from the session rather than from arguments. Like the clock built-ins, these arrive in
    /// their bare keyword form with no argument list.
    #[must_use]
    pub const fn is_session_user(self) -> bool {
        matches!(self, Self::CurrentUser | Self::SessionUser)
    }
}

/// An ordered-set aggregate call: `func(args) WITHIN GROUP (ORDER BY ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct WithinGroup {
    /// Aggregate function name, folded to lowercase (e.g. `percentile_cont`).
    pub func: String,
    /// Direct arguments (the fraction for `PERCENTILE_CONT`/`PERCENTILE_DISC`); empty for `MODE`.
    pub args: Vec<Expr>,
    /// The `WITHIN GROUP (ORDER BY ...)` sort keys.
    pub order_by: Vec<OrderByItem>,
}

/// A window function call: ranking / navigation / aggregate used `OVER` a window.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFunction {
    /// Which window function.
    pub func: WindowFunc,
    /// Argument expressions.
    pub args: Vec<Expr>,
    /// `PARTITION BY` key expressions; empty when absent.
    pub partition: Vec<Expr>,
    /// `ORDER BY` keys within the partition.
    pub order: Vec<OrderByItem>,
    /// `ROWS | RANGE | GROUPS` frame; `None` when absent.
    pub frame: Option<WindowFrame>,
    /// `FILTER (WHERE pred)` — restrict the aggregation to rows where `pred` is true. Only an
    /// aggregate used `OVER` a window may carry one; the parser rejects it on the ranking,
    /// navigation and distribution functions, which aggregate nothing. `None` when absent.
    ///
    /// A filtered row is excluded from the aggregate but still produces an output row, so this
    /// narrows what the window *reads*, never what the query *returns*.
    pub filter: Option<Box<Expr>>,
}

/// A `ROWS | RANGE | GROUPS BETWEEN start AND end` window frame.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    /// Frame unit.
    pub units: WindowFrameUnits,
    /// Frame start bound.
    pub start: WindowFrameBound,
    /// Frame end bound; `None` for the shorthand `ROWS n PRECEDING` form
    /// (semantically equivalent to `end = CURRENT ROW`).
    pub end: Option<WindowFrameBound>,
    /// `EXCLUDE {CURRENT ROW | GROUP | TIES | NO OTHERS}` — which rows to drop from the frame
    /// after the bounds have selected it. Defaults to [`WindowExclude::NoOthers`] (no exclusion).
    pub exclude: WindowExclude,
}

/// The `EXCLUDE` clause of a window frame: which rows to remove from the frame after the
/// frame bounds have selected it.
///
/// Peers are rows sharing the current row's `ORDER BY` value (with no `ORDER BY`, every row in the
/// partition is a peer of every other). The frame `EXCLUDE` clause is not parsed by the upstream
/// `sqlparser` crate (its `WindowFrame` records no exclusion), so the parser strips it before
/// handing SQL to `sqlparser` and threads the captured mode here — see the window-exclude
/// preprocessor in the parser module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowExclude {
    /// `EXCLUDE NO OTHERS` (the default) — no rows are excluded.
    #[default]
    NoOthers,
    /// `EXCLUDE CURRENT ROW` — drop the current row from its frame.
    CurrentRow,
    /// `EXCLUDE GROUP` — drop the current row and all its peers.
    Group,
    /// `EXCLUDE TIES` — drop the current row's peers but keep the current row itself.
    Ties,
}

/// Unit of a window frame: `ROWS`, `RANGE`, or `GROUPS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameUnits {
    /// Physical row offsets.
    Rows,
    /// Logical value-based range.
    Range,
    /// Peer-group count.
    Groups,
}

/// One bound of a window frame.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFrameBound {
    /// `UNBOUNDED PRECEDING`.
    UnboundedPreceding,
    /// `<n> PRECEDING` — a constant offset before the current row.
    Preceding(Box<Expr>),
    /// `CURRENT ROW`.
    CurrentRow,
    /// `<n> FOLLOWING` — a constant offset after the current row.
    Following(Box<Expr>),
    /// `UNBOUNDED FOLLOWING`.
    UnboundedFollowing,
}

/// A window-only or aggregate-as-window function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowFunc {
    /// `ROW_NUMBER()` — unique sequential number within each partition.
    RowNumber,
    /// `RANK()` — rank with gaps for ties.
    Rank,
    /// `DENSE_RANK()` — rank without gaps for ties.
    DenseRank,
    /// `NTILE(n)` — divides the partition into n equal buckets.
    Ntile,
    /// `CUME_DIST()` — cumulative distribution value in (0, 1].
    CumeDist,
    /// `PERCENT_RANK()` — relative rank in [0, 1].
    PercentRank,
    /// `LAG(expr [, offset [, default]])` — value from a preceding row.
    Lag,
    /// `LEAD(expr [, offset [, default]])` — value from a following row.
    Lead,
    /// `FIRST_VALUE(expr)` — first value in the window frame.
    FirstValue,
    /// `LAST_VALUE(expr)` — last value in the window frame.
    LastValue,
    /// `NTH_VALUE(expr, n)` — nth value in the window frame.
    NthValue,
    /// One of the five aggregate functions used as a window function.
    Aggregate(AggregateFunc),
}

/// The five baseline aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    /// `COUNT(*)` — every row, NULLs included.
    /// `COUNT(expr)` — rows where `expr` is not NULL.
    Count,
    /// `SUM(expr)` — total of non-NULL numeric values.
    Sum,
    /// `AVG(expr)` — arithmetic mean of non-NULL numeric values.
    Avg,
    /// `MIN(expr)` — minimum of non-NULL comparable values.
    Min,
    /// `MAX(expr)` — maximum of non-NULL comparable values.
    Max,
    /// `PERCENTILE_CONT(f) WITHIN GROUP (ORDER BY x)` — continuous percentile: the value at
    /// fraction `f` of the sorted non-NULL `x`, linearly interpolated. Numeric input only;
    /// result `FLOAT`.
    PercentileCont,
    /// `PERCENTILE_DISC(f) WITHIN GROUP (ORDER BY x)` — discrete percentile: the first sorted value
    /// whose cumulative position reaches `f`. Result is an element of `x`.
    PercentileDisc,
    /// `MODE() WITHIN GROUP (ORDER BY x)` — the most frequent non-NULL `x`, ties broken by the sort
    /// order. Result is an element of `x`.
    Mode,
    /// `RANK(a) WITHIN GROUP (ORDER BY x)` — hypothetical-set rank *with* gaps: the rank the direct
    /// argument `a` would take if inserted into the sorted non-NULL `x`, i.e. `1 + count(x ≺ a)`
    /// under the `ORDER BY` direction. Result `BIGINT`.
    Rank,
    /// `DENSE_RANK(a) WITHIN GROUP (ORDER BY x)` — hypothetical-set rank *without* gaps:
    /// `1 + count(distinct x ≺ a)`. Result `BIGINT`.
    DenseRank,
    /// `PERCENT_RANK(a) WITHIN GROUP (ORDER BY x)` — relative rank of the hypothetical `a`:
    /// `(rank(a) − 1) / n` over the `n` non-NULL `x` (`0` when `n = 0`). Result `FLOAT`.
    PercentRank,
    /// `CUME_DIST(a) WITHIN GROUP (ORDER BY x)` — cumulative distribution of the hypothetical `a`:
    /// `(count(x ≼ a) + 1) / (n + 1)` over the `n` non-NULL `x`. Result `FLOAT`.
    CumeDist,
    /// `ARRAY_AGG(expr)` — collect every input value (including `NULL`s) of `expr` into an array, in
    /// input order; an empty group yields `NULL`. Result type is an array of `expr`'s type.
    ArrayAgg,
    /// `JSONB_AGG(expr)` / `JSON_AGG(expr)` — collect every input value (a `NULL` becomes JSON `null`)
    /// into a JSON array, in input order; an empty group yields `NULL`. The argument may be any type
    /// (serialized to JSON per element). Result `JSON`.
    JsonAgg,
    /// `JSON_OBJECT_AGG(key, value)` / `JSONB_OBJECT_AGG` — aggregate key/value pairs into a JSON
    /// object. A NULL key errors; a NULL value becomes JSON `null`; duplicate keys keep the last value
    /// (our JSON is binary-backed, so keys dedup). Result `JSON`; an empty group yields `NULL`.
    JsonObjectAgg,
    /// `BOOL_AND(expr)` — `TRUE` iff every non-`NULL` boolean input is `TRUE`; `NULL` for an empty
    /// group. Result `BOOL`.
    BoolAnd,
    /// `BOOL_OR(expr)` — `TRUE` iff any non-`NULL` boolean input is `TRUE`; `NULL` for an empty group.
    /// Result `BOOL`.
    BoolOr,
    /// `STDDEV(expr)` / `STDDEV_SAMP(expr)` — sample standard deviation of the non-`NULL` numeric
    /// inputs; `NULL` for fewer than two values. Result `FLOAT`.
    Stddev,
    /// `VARIANCE(expr)` / `VAR_SAMP(expr)` — sample variance of the non-`NULL` numeric inputs; `NULL`
    /// for fewer than two values. Result `FLOAT`.
    Variance,
    /// `CORR(y, x)` — Pearson correlation coefficient of the non-`NULL` `(y, x)` pairs; `NULL` for an
    /// empty group or when either input has zero variance. Result `FLOAT`.
    Corr,
    /// `COVAR_POP(y, x)` — population covariance of the non-`NULL` `(y, x)` pairs (divisor `n`); `NULL`
    /// for an empty group. Result `FLOAT`.
    CovarPop,
    /// `COVAR_SAMP(y, x)` — sample covariance of the non-`NULL` `(y, x)` pairs (divisor `n − 1`);
    /// `NULL` for fewer than two pairs. Result `FLOAT`.
    CovarSamp,
    /// `REGR_COUNT(y, x)` — number of non-`NULL` `(y, x)` pairs. Result `INT` (`0` for an empty
    /// group, never `NULL`).
    RegrCount,
    /// `REGR_AVGX(y, x)` — mean of `x` over the non-`NULL` pairs; `NULL` for an empty group. Result
    /// `FLOAT`.
    RegrAvgx,
    /// `REGR_AVGY(y, x)` — mean of `y` over the non-`NULL` pairs; `NULL` for an empty group. Result
    /// `FLOAT`.
    RegrAvgy,
    /// `REGR_SXX(y, x)` — `Σ(x − avgx)²` over the non-`NULL` pairs; `NULL` for an empty group. Result
    /// `FLOAT`.
    RegrSxx,
    /// `REGR_SYY(y, x)` — `Σ(y − avgy)²` over the non-`NULL` pairs; `NULL` for an empty group. Result
    /// `FLOAT`.
    RegrSyy,
    /// `REGR_SXY(y, x)` — `Σ(x − avgx)(y − avgy)` over the non-`NULL` pairs; `NULL` for an empty
    /// group. Result `FLOAT`.
    RegrSxy,
    /// `REGR_SLOPE(y, x)` — slope of the least-squares-fit line `y = a + b·x` (`b = Sxy / Sxx`);
    /// `NULL` when `Sxx` is `0` or the group is empty. Result `FLOAT`.
    RegrSlope,
    /// `REGR_INTERCEPT(y, x)` — `y`-intercept of the least-squares-fit line (`avgy − slope·avgx`);
    /// `NULL` when `Sxx` is `0` or the group is empty. Result `FLOAT`.
    RegrIntercept,
    /// `REGR_R2(y, x)` — coefficient of determination (R²) of the least-squares fit; `NULL` when
    /// `Sxx` is `0` or the group is empty, and `1` when `Syy` is `0`. Result `FLOAT`.
    RegrR2,
    /// `BIT_AND(expr)` — bitwise AND of the non-`NULL` integer inputs; `NULL` for an empty group.
    /// Result `INT`.
    BitAnd,
    /// `BIT_OR(expr)` — bitwise OR of the non-`NULL` integer inputs; `NULL` for an empty group.
    /// Result `INT`.
    BitOr,
    /// `BIT_XOR(expr)` — bitwise XOR of the non-`NULL` integer inputs; `NULL` for an empty group.
    /// Result `INT` (B-fn).
    BitXor,
    /// `STDDEV_POP(expr)` — population standard deviation of the non-`NULL` numeric inputs (divisor
    /// `n`); `NULL` for an empty group. Result `FLOAT`.
    StddevPop,
    /// `VAR_POP(expr)` — population variance of the non-`NULL` numeric inputs (divisor `n`); `NULL`
    /// for an empty group. Result `FLOAT`.
    VarPop,
    /// `STRING_AGG(expr, separator)` — concatenate the non-`NULL` `TEXT` inputs in input order,
    /// joined by the constant `separator`; `NULL` for an empty group. Result `TEXT`.
    StringAgg,
    /// Synthetic `GROUPING(...)` super-aggregate indicator — *not* spelled by the user as an
    /// aggregate; the analyzer rewrites a `GROUPING(key, ...)` scalar call into this when the query
    /// has `GROUPING SETS`/`ROLLUP`/`CUBE`. It folds no row values: the executor emits a bitmask from
    /// the current grouping set and the call's `grouping_args` (in `planner::AggregateCall`). Result
    /// `INT`.
    Grouping,
}

impl AggregateFunc {
    /// The canonical (folded) name, used as the default name of an output column that the query
    /// did not alias — tools that read result-set column names expect the function's own name there.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Avg => "avg",
            Self::Min => "min",
            Self::Max => "max",
            Self::PercentileCont => "percentile_cont",
            Self::PercentileDisc => "percentile_disc",
            Self::Mode => "mode",
            Self::Rank => "rank",
            Self::DenseRank => "dense_rank",
            Self::PercentRank => "percent_rank",
            Self::CumeDist => "cume_dist",
            Self::ArrayAgg => "array_agg",
            Self::JsonAgg => "json_agg",
            Self::JsonObjectAgg => "json_object_agg",
            Self::BoolAnd => "bool_and",
            Self::BoolOr => "bool_or",
            Self::Stddev => "stddev",
            Self::Variance => "variance",
            Self::Corr => "corr",
            Self::CovarPop => "covar_pop",
            Self::CovarSamp => "covar_samp",
            Self::RegrCount => "regr_count",
            Self::RegrAvgx => "regr_avgx",
            Self::RegrAvgy => "regr_avgy",
            Self::RegrSxx => "regr_sxx",
            Self::RegrSyy => "regr_syy",
            Self::RegrSxy => "regr_sxy",
            Self::RegrSlope => "regr_slope",
            Self::RegrIntercept => "regr_intercept",
            Self::RegrR2 => "regr_r2",
            Self::BitAnd => "bit_and",
            Self::BitOr => "bit_or",
            Self::BitXor => "bit_xor",
            Self::StddevPop => "stddev_pop",
            Self::VarPop => "var_pop",
            Self::StringAgg => "string_agg",
            Self::Grouping => "grouping",
        }
    }

    /// Whether this is a two-argument statistical aggregate over `(y, x)` pairs — `CORR`, the
    /// `COVAR_*` covariances, and the `REGR_*` linear-regression family. These carry a second
    /// per-row argument and fold a pair only when both sides are non-`NULL`.
    #[must_use]
    pub const fn is_two_arg(self) -> bool {
        matches!(
            self,
            Self::Corr
                | Self::CovarPop
                | Self::CovarSamp
                | Self::RegrCount
                | Self::RegrAvgx
                | Self::RegrAvgy
                | Self::RegrSxx
                | Self::RegrSyy
                | Self::RegrSxy
                | Self::RegrSlope
                | Self::RegrIntercept
                | Self::RegrR2
        )
    }
}

/// One `WHEN ... THEN ...` clause inside an [`Expr::Case`].
#[derive(Debug, Clone, PartialEq)]
pub struct CaseBranch {
    /// Match value (simple `CASE`) or predicate (searched `CASE`).
    pub when: Expr,
    /// Result expression returned when this branch matches.
    pub then: Expr,
}

// `Coalesce(Vec<Expr>)` and `Cast { expr, target }` extend the `Expr` enum
// above; see those variants for documentation.

/// A literal SQL value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// SQL `NULL`.
    Null,
    /// Boolean literal.
    Bool(bool),
    /// 64-bit signed integer literal.
    Int(i64),
    /// 64-bit IEEE-754 floating-point literal.
    Float(f64),
    /// UTF-8 text literal.
    Text(String),
    /// Calendar date — days since the Unix epoch, proleptic Gregorian.
    Date(i32),
    /// Time of day — microseconds since midnight, in `[0, 86_400_000_000)`.
    Time(i64),
    /// Timestamp without time zone — microseconds since the Unix epoch.
    Timestamp(i64),
    /// Timestamp with time zone — microseconds since the Unix epoch, normalized to UTC.
    TimestampTz(i64),
    /// Time of day with time zone — one packed `i64` carrying both the as-entered local time and
    /// its zone offset (P-TIMETZ): `utc_equivalent_micros * 2^18 + (zone_west_secs + 2^17)`, so
    /// plain `i64` ordering compares by instant with the zone as tie-break. Build/inspect it with
    /// [`temporal::pack_timetz`](crate::temporal::pack_timetz) and friends — never interpret the
    /// raw value as microseconds.
    TimeTz(i64),
    /// 128-bit UUID.
    Uuid([u8; 16]),
    /// MAC address — six bytes in transmission order.
    Macaddr([u8; 6]),
    /// 8-byte MAC / EUI-64 — eight bytes in transmission order.
    Macaddr8([u8; 8]),
    /// IPv4/IPv6 address for `INET` or `CIDR` — the `is_cidr` flag inside distinguishes the two.
    Inet(crate::inet::InetAddr),
    /// A `BIT`/`BIT VARYING` bit string — index 0 is the leftmost bit.
    Bit(Vec<bool>),
    /// A range value (`int4range`, `numrange`, …); boxed since it holds its bound values.
    Range(Box<crate::range::RangeVal>),
    /// Exact decimal (`NUMERIC` / `DECIMAL`) value.
    Numeric(crate::numeric::Decimal),
    /// JSON / JSONB document as canonical text.
    Json(String),
    /// Calendar duration `INTERVAL`.
    Interval(crate::interval::Interval),
    /// One-dimensional array; elements share the column's declared element type.
    Array(Vec<Self>),
    /// Fixed-dimension `f32` vector `VECTOR(n)` for similarity search.
    Vector(Vec<f32>),
    /// Raw byte string `BYTEA` — the in-runtime value of a `ColumnType::Bytes` column. Cast
    /// to/from text via the `\x<hex>` form; rendered the same way.
    Bytes(Vec<u8>),
    /// A geometric value (`point` / `box`); its [`GeomKind`](nusadb_core::engine::GeomKind) is
    /// carried inside the [`crate::geometry::GeomVal`].
    Geometry(crate::geometry::GeomVal),
    /// A full-text search document (`tsvector`) in its canonical text form (lexemes sorted, quoted,
    /// with ascending positions).
    Tsvector(String),
    /// A full-text search query (`tsquery`) in its canonical text form (a boolean expression over
    /// quoted lexemes).
    Tsquery(String),
    /// An XML value (`xml`) as its original, well-formed text (not canonicalized).
    Xml(String),
}

/// Binary operators the parser accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `=`.
    Eq,
    /// `<>` / `!=`.
    NotEq,
    /// `<`.
    Lt,
    /// `<=`.
    LtEq,
    /// `>`.
    Gt,
    /// `>=`.
    GtEq,
    /// `AND`.
    And,
    /// `OR`.
    Or,
    /// `+`.
    Plus,
    /// `-`.
    Minus,
    /// `*`.
    Multiply,
    /// `/`.
    Divide,
    /// `%`.
    Modulo,
    /// `&` — bitwise AND of two integers, as `INT`.
    BitAnd,
    /// `|` — bitwise OR of two integers, as `INT`.
    BitOr,
    /// `#` — bitwise XOR of two integers, as `INT` (the reference engine's XOR spelling).
    BitXor,
    /// `<<` — left bit-shift of an integer by an integer count, as `INT` (B-fn).
    ShiftLeft,
    /// `>>` — right (arithmetic) bit-shift of an integer by an integer count, as `INT` (B-fn).
    ShiftRight,
    /// `&&` — array overlap: whether two arrays share any element, as `BOOL` (B-fn).
    ArrayOverlap,
    /// `||` — string concatenation. Both operands are text; the result is text.
    Concat,
    /// JSON `->` — get object field / array element as JSON.
    JsonGet,
    /// JSON `->>` — get object field / array element as text.
    JsonGetText,
    /// `@>` containment — does the left contain the right? For `JSON` documents and for
    /// arrays (every element of the right array is present in the left), yielding `BOOL`.
    JsonContains,
    /// `<@` contained-by — is the left contained by the right? The mirror of [`Self::JsonContains`]
    /// (`a <@ b` ≡ `b @> a`); covers both `JSON` and arrays, yielding `BOOL`.
    JsonContainedBy,
    /// JSON `?` — does a top-level key (object), string element (array) or scalar string equal the
    /// right `TEXT` operand, as `BOOL`?
    JsonExists,
    /// JSON `?|` — does **any** key in the right `text[]` exist per [`Self::JsonExists`]?
    JsonExistsAny,
    /// JSON `?&` — does **every** key in the right `text[]` exist per [`Self::JsonExists`]?
    JsonExistsAll,
    /// JSON `#>` — get the value at a `text[]` path as JSON.
    JsonGetPath,
    /// JSON `#>>` — get the value at a `text[]` path as text.
    JsonGetPathText,
    /// JSON `#-` — delete the element at a `text[]` path, yielding the trimmed `JSON` document.
    JsonDeletePath,
    /// Vector distance `<=>` — cosine distance between two `VECTOR(n)` operands, as `FLOAT`
    /// (the `<=>` operator). See [`crate::vector::cosine_distance`].
    VectorDistance,
    /// Vector distance `<->` — Euclidean (L2) distance between two `VECTOR(n)` operands, as
    /// `FLOAT`. See [`crate::vector::l2_distance`].
    VectorL2Distance,
    /// Vector distance `<#>` — negative inner product of two `VECTOR(n)` operands, as `FLOAT`
    /// (negated so that smaller is nearer, matching the distance convention). See
    /// [`crate::vector::neg_inner_product`].
    VectorNegInnerProduct,
    /// Vector distance `<+>` — Manhattan (L1, taxicab) distance between two `VECTOR(n)` operands,
    /// as `FLOAT`. See [`crate::vector::l1_distance`].
    VectorL1Distance,
    /// Full-text `@@` — does the left `tsvector` (canonical text form) match the right `tsquery`
    /// (text form), as `BOOL` (F1)? Either operand order is accepted, like the reference engine.
    TsMatch,
    /// JSON path exists `@?` — does the right `jsonpath` return any item for the left JSON value, as
    /// `BOOL`? An unsupported or invalid path is a loud runtime error, like `jsonb_path_exists`.
    JsonPathExists,
    /// Geometric same-as `~=` — are two geometric values (both `point` or both `box`) equal by
    /// their coordinates, as `BOOL`?
    GeomSameAs,
    /// Geometric parallel `?||` — are two `lseg`s or two `line`s parallel (their direction vectors,
    /// resp. normals, colinear — same or opposite direction), as `BOOL`?
    GeomParallel,
    /// Geometric perpendicular `?-|` — are two `lseg`s or two `line`s perpendicular (their direction
    /// vectors, resp. normals, at a right angle), as `BOOL`?
    GeomPerpendicular,
    /// Geometric intersects `?#` — do two `lseg`s cross/touch at a single point, or do two `line`s
    /// cross (i.e. are not parallel), as `BOOL`?
    GeomIntersects,
    /// `<<=` — subnet-or-equal: is the left network contained within **or equal to** the right, as
    /// `BOOL`? (`a <<= b` ≡ the right contains-or-equals the left.) A cross-family pair is `FALSE`.
    InetSubnetEq,
    /// `>>=` — supernet-or-equal: does the left network contain **or equal** the right, as `BOOL`?
    /// (`a >>= b` ≡ the left contains-or-equals the right.) A cross-family pair is `FALSE`.
    InetSupernetEq,
    /// `-|-` — range adjacency: do two ranges of the same element kind touch at exactly one boundary,
    /// with no overlap and no gap, as `BOOL`? An empty operand on either side is never adjacent.
    RangeAdjacent,
    /// `&<` — does-not-extend-to-the-right-of: is the left range's upper bound at or below the right's
    /// (both ranges of the same element kind), as `BOOL`? An empty operand on either side yields
    /// `FALSE`.
    RangeNotExtendRight,
    /// `&>` — does-not-extend-to-the-left-of: is the left range's lower bound at or above the right's
    /// (both ranges of the same element kind), as `BOOL`? An empty operand on either side yields
    /// `FALSE`.
    RangeNotExtendLeft,
}

/// Unary operators the parser accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `NOT`.
    Not,
    /// Numeric negation, `-`.
    Negate,
    /// Unary plus, `+` — a no-op on a numeric operand (returns it unchanged), rejected on others.
    Plus,
    /// Bitwise complement, `~` — flips every bit of an integer or `MACADDR8` operand.
    BitNot,
    /// Geometric center `@@` — the center point of a `box` (corner midpoint), `circle` (center),
    /// `lseg` (midpoint), or `polygon` (vertex mean), as a `point`.
    GeomCenter,
    /// Tsquery negation `!!` (prefix) — negate a `tsquery`, e.g. `!!'cat'` = `!'cat'`.
    TsqueryNot,
}

/// The JSON item type required by an [`Expr::IsJson`] predicate (`<expr> IS JSON <item_type>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonItemType {
    /// `IS JSON` / `IS JSON VALUE` — valid JSON of any type.
    Value,
    /// `IS JSON SCALAR` — a JSON scalar (number, string, boolean, or `null`).
    Scalar,
    /// `IS JSON ARRAY` — a JSON array.
    Array,
    /// `IS JSON OBJECT` — a JSON object.
    Object,
}

/// The truth value tested by an [`Expr::IsBool`] (`IS [NOT] TRUE/FALSE/UNKNOWN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthValue {
    /// `IS [NOT] TRUE`.
    True,
    /// `IS [NOT] FALSE`.
    False,
    /// `IS [NOT] UNKNOWN` (i.e. `NULL`).
    Unknown,
}
