//! Thin wrapper around the generic `sqlparser` crate (a Rust SQL tokenizer +
//! AST library, *not* tied to any specific database product), with conversion
//! to NusaDB's own internal AST.
//!
//! **Do not** import `sqlparser` types anywhere outside this module — they are
//! wrapped in our [`ast`] at the door. Downstream layers (analyzer,
//! planner, executor) must never see a `sqlparser` type; this is the
//! anti-corruption layer that lets the dependency be upgraded in one place.
//!
//! The parser accepts exactly one statement per [`parse`] call. Tokenising and
//! grammar use sqlparser's `GenericDialect`, which is the only way to enable the
//! grammar extensions NusaDB models — e.g. `UPDATE … FROM`, gated inside
//! sqlparser on a dialect **type** whitelist (`dialect_of!`), so a custom
//! `Dialect` cannot opt in. `GenericDialect`'s tokenizer is more permissive than
//! NusaDB's documented surface, so `reject_widened_lexicon` re-narrows the
//! identifier lexicon to exactly what [`NusaDialect`] defines (the
//! tokenize-time gate the surface promises), and the converters reject every
//! grammar construct outside the surface. Valid SQL outside NusaDB's current
//! surface is rejected with [`Error::Unsupported`].
//!
//! The converters are organised into per-concern submodules (ADR 007): `ddl`, `dml`,
//! `query`, `select`, `expr`. They form one tightly-connected web of free functions, so
//! each resolves its siblings through a glob re-export rather than a long explicit import list.
#![allow(clippy::wildcard_imports)]

use core::any::TypeId;
use sqlparser::ast as sql;
use sqlparser::dialect::{Dialect, GenericDialect};
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, Tokenizer};

use nusadb_core::ColumnType;

use crate::ast;
use crate::error::Error;

mod ddl;
mod dml;
mod expr;
mod query;
mod script;
mod select;
use ddl::*;
use dml::*;
use expr::*;
use query::*;
pub(crate) use script::{ScriptBlock, ScriptStmt, is_script, parse_script};
use select::*;

/// NusaDB's SQL dialect — the documented identifier surface.
///
/// Unquoted identifiers are case-insensitive (folded to lowercase) and consist of ASCII
/// letters/digits/`_` with `$` allowed as a non-leading character; `"..."` quotes an identifier and
/// `'...'` a string literal.
///
/// Tokenising actually runs on `GenericDialect` (the only way to reach the
/// grammar extensions NusaDB models — see the module docs), whose identifier
/// lexicon is wider (it admits `@`/`#`-led and non-ASCII identifiers). So
/// `NusaDialect` is **not** handed to the tokenizer; instead it is the single
/// source of truth for the identifier surface that `reject_widened_lexicon`
/// consults to reject any token outside it. Case-folding is applied separately
/// by `fold_ident`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NusaDialect;

impl Dialect for NusaDialect {
    fn is_identifier_start(&self, ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
    }
}

/// The dialect passed to the sqlparser tokenizer/parser at runtime.
///
/// Several grammar extensions (e.g. `UPDATE … FROM`, `AGG … FILTER (WHERE …)`) are
/// gated on a `dialect_of!` type-ID whitelist inside sqlparser — the only way to opt in
/// is to make the dialect's [`Dialect::dialect`] return a recognised [`TypeId`]. This
/// struct delegates `dialect()` to [`GenericDialect`]'s `TypeId` (which is in every
/// such whitelist), while adding the capabilities that `GenericDialect` alone does not
/// enable (notably `FILTER` during aggregation).
#[derive(Debug, Default, Clone, Copy)]
struct NusaParserDialect;

impl Dialect for NusaParserDialect {
    fn dialect(&self) -> TypeId {
        // Report as GenericDialect so sqlparser's dialect_of! whitelists pass.
        TypeId::of::<GenericDialect>()
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        // GenericDialect allows Unicode; NusaDB's documented surface is ASCII-only.
        // NusaDialect enforces that after the parse via `reject_widened_lexicon`.
        ch.is_alphabetic() || ch == '_'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        ch.is_alphabetic() || ch.is_ascii_digit() || ch == '_' || ch == '$'
    }

    fn supports_filter_during_aggregation(&self) -> bool {
        true
    }

    fn supports_group_by_expr(&self) -> bool {
        true // enables ROLLUP/CUBE/GROUPING SETS as GROUP BY expressions
    }

    fn supports_select_wildcard_except(&self) -> bool {
        // Must be true (same as GenericDialect) so that `SELECT * EXCEPT (col)`
        // is tokenised as a wildcard decoration that our converter can reject with
        // `Unsupported` rather than falling through to a misleading `Syntax` error
        // (sqlparser would parse `EXCEPT` as a set-operation keyword otherwise).
        true
    }

    fn supports_array_typedef_with_brackets(&self) -> bool {
        // `INT[]` / `TEXT[]` column + cast types (arrays). 0.51's GenericDialect parsed the
        // bracket suffix unconditionally; 0.62 gates it behind this hook.
        true
    }

    fn supports_bitwise_shift_operators(&self) -> bool {
        // `<<` / `>>` integer shifts. Same story: newly gated in 0.62.
        true
    }

    fn supports_string_escape_constant(&self) -> bool {
        // `E'…'` escape-string literals: the 0.51 tokenizer
        // recognized the prefix unconditionally; 0.62 gates it behind this hook, and the
        // migration dropped it — `E'a\tb'` tokenized as identifier `E` + a plain string. The
        // tokenizer interprets the backslash escapes, so the literal reaches the converter
        // already unescaped.
        true
    }

    fn supports_unicode_string_literal(&self) -> bool {
        // `U&'…'` unicode-string literals (`\XXXX` code points), same regression class.
        true
    }

    fn supports_select_wildcard_exclude(&self) -> bool {
        // Like `supports_select_wildcard_except` above: keep `SELECT * EXCLUDE (a)` parsing as a
        // wildcard decoration so our converter rejects it loudly with `Unsupported` instead of a
        // misleading `Syntax` error.
        true
    }

    fn supports_select_wildcard_replace(&self) -> bool {
        // Same rationale for `SELECT * REPLACE (expr AS col)`.
        true
    }

    fn parse_infix(
        &self,
        parser: &mut Parser,
        expr: &sql::Expr,
        precedence: u8,
    ) -> Option<Result<sql::Expr, sqlparser::parser::ParserError>> {
        // `#` — the reference engine's integer XOR. sqlparser's own infix table gates the Sharp
        // token behind a dialect TypeId this wrapper does not report (it must keep reporting
        // `GenericDialect`'s — see `dialect()`), so parse the operator here. The default precedence table
        // already assigns `#` the caret tier for every dialect, so the Pratt loop reaches this
        // hook; any other token falls through untouched.
        if parser.peek_token().token != Token::Sharp {
            return None;
        }
        let _sharp = parser.next_token();
        Some(
            parser
                .parse_subexpr(precedence)
                .map(|right| sql::Expr::BinaryOp {
                    left: Box::new(expr.clone()),
                    op: sql::BinaryOperator::PGBitwiseXor,
                    right: Box::new(right),
                }),
        )
    }
}

/// Parse a single SQL statement into the internal [`ast::Statement`].
///
/// Accepts the [`NusaDialect`] surface; the underlying tokenizer uses
/// `GenericDialect` to enable grammar extensions inside sqlparser. The input
/// must contain exactly one statement; empty input and multi-statement input
/// are both rejected.
pub fn parse(sql: &str) -> Result<ast::Statement, Error> {
    // Re-narrow the identifier lexicon to NusaDB's surface before anything else, so the gate
    // covers every path (the VACUUM / COMMENT recognizers below as well as the generic parser).
    reject_widened_lexicon(sql)?;
    // `VACUUM` is not in the generic tokenizer's statement grammar, so recognize
    // the bare, table-less form (and `EXPLAIN VACUUM`) ourselves first.
    if let Some(stmt) = recognize_vacuum(sql) {
        return Ok(stmt);
    }
    // `REINDEX ...` is likewise not modelled as a statement by the generic tokenizer; recognize it
    // here and accept it as a no-op (see `recognize_reindex`).
    if let Some(stmt) = recognize_reindex(sql) {
        return Ok(stmt);
    }
    // `COMMENT ON ...` is tokenized by `sqlparser` 0.51 but not modelled as a statement, so
    // (like `VACUUM`) recognize and parse it ourselves before the generic path.
    if let Some(result) = recognize_comment(sql) {
        return result;
    }
    // `RESET name` is not in the generic tokenizer's statement grammar; recognize it
    // ourselves and model it as a `SetVariable` with no value (reset to default).
    if let Some(result) = recognize_reset(sql) {
        return result;
    }
    // `COPY ... FROM STDIN` makes `sqlparser` 0.51 expect an inline data block (terminated by a
    // `\.` line); over the wire the rows arrive separately, so we drive the parser ourselves
    // rather than feed it a statement it would reject at EOF.
    if let Some(result) = recognize_copy(sql) {
        return result;
    }
    // `REFRESH MATERIALIZED VIEW name` — sqlparser has no REFRESH keyword, so recognize it here.
    if let Some(result) = recognize_refresh(sql) {
        return result;
    }
    // `DROP MATERIALIZED VIEW name` — sqlparser 0.51 rejects the MATERIALIZED keyword after DROP, so
    // recognize it ourselves (plain `DROP VIEW` still goes through the generic parser).
    if let Some(result) = recognize_drop_matview(sql) {
        return result;
    }
    // `DROP DATABASE [IF EXISTS] name` — sqlparser 0.51 has no DATABASE drop grammar; recognize the
    // single-database compatibility no-op ourselves (CREATE DATABASE goes through the generic parser).
    if let Some(result) = recognize_drop_database(sql) {
        return result;
    }
    // `ALTER DATABASE name ...` — single-database compatibility no-op (sqlparser's grammar is narrow).
    if let Some(result) = recognize_alter_database(sql) {
        return result;
    }
    // `CREATE POLICY ...` / `DROP POLICY ...` — sqlparser 0.51 has no RLS-policy grammar, so
    // drive its primitives ourselves.
    if let Some(result) = recognize_create_policy(sql) {
        return result;
    }
    if let Some(result) = recognize_drop_policy(sql) {
        return result;
    }
    if let Some(result) = recognize_alter_policy(sql) {
        return result;
    }
    // `CREATE [OR REPLACE] TRIGGER ...` / `DROP TRIGGER ...` — sqlparser 0.51 only
    // models the `EXECUTE FUNCTION` trigger form, but NusaDB triggers run a SQL statement body, so
    // drive the grammar ourselves.
    if let Some(result) = recognize_create_trigger(sql) {
        return result;
    }
    if let Some(result) = recognize_drop_trigger(sql) {
        return result;
    }
    if let Some(result) = recognize_alter_trigger(sql) {
        return result;
    }
    // `CREATE TYPE name AS ENUM (...)` / `DROP TYPE ...` (B-ENUM) — sqlparser 0.51 only models the
    // composite `CREATE TYPE name AS (...)` form and has no `DROP TYPE`, so drive these ourselves.
    if let Some(result) = recognize_create_type_enum(sql) {
        return result;
    }
    if let Some(result) = recognize_drop_type(sql) {
        return result;
    }
    // `CREATE [OR REPLACE] PROCEDURE ...` / `DROP PROCEDURE ...` / `CALL ...` —
    // drive these ourselves so the procedure body (a `$$…$$` / `'…'` SQL block) is captured verbatim.
    if let Some(result) = recognize_create_procedure(sql) {
        return result;
    }
    if let Some(result) = recognize_drop_procedure(sql) {
        return result;
    }
    if let Some(result) = recognize_call(sql) {
        return result;
    }
    // `CREATE [OR REPLACE] FUNCTION ...` / `DROP FUNCTION ...` — a SQL scalar function whose
    // `SELECT <expr>` body is captured verbatim and inlined at call sites.
    if let Some(result) = recognize_create_function(sql) {
        return result;
    }
    if let Some(result) = recognize_drop_function(sql) {
        return result;
    }
    // `LOCK TABLE t IN <mode> MODE` — the `IN <mode> MODE` form; sqlparser only models the
    // `LOCK TABLES … READ|WRITE` form, so recognize this syntax here.
    if let Some(result) = recognize_lock_table(sql) {
        return result;
    }
    // `LISTEN channel` / `UNLISTEN channel|*` / `NOTIFY channel [, 'payload']` (async pub/sub) —
    // sqlparser 0.51 has no grammar for any of these, so recognize them ourselves. The server owns
    // the cross-connection channel registry and intercepts these before the SQL engine.
    if let Some(result) = recognize_listen(sql) {
        return result;
    }
    if let Some(result) = recognize_unlisten(sql) {
        return result;
    }
    if let Some(result) = recognize_notify(sql) {
        return result;
    }
    let dialect = NusaParserDialect;
    let mut statements =
        Parser::parse_sql(&dialect, sql).map_err(|e| Error::Syntax(e.to_string()))?;
    match statements.len() {
        1 => convert_statement(statements.remove(0)),
        0 => Err(Error::Empty),
        n => Err(Error::MultipleStatements(n)),
    }
}

/// Recognize the bare `VACUUM` statement (and `EXPLAIN VACUUM`), which the
/// generic tokenizer does not model. Case-insensitive and tolerant of
/// surrounding/interior whitespace and a trailing semicolon. Argument forms like
/// `VACUUM <table>` are not recognized here and fall through to the generic
/// parser (which rejects them as unsupported).
fn recognize_vacuum(sql: &str) -> Option<ast::Statement> {
    // Cheap guard before any allocation: VACUUM is rare, so skip the normalize for every other
    // statement. Only `VACUUM …` / `EXPLAIN VACUUM` reach the allocating path.
    let trimmed = sql.trim();
    let starts_vacuum = trimmed
        .get(..6)
        .is_some_and(|p| p.eq_ignore_ascii_case("vacuum"));
    let starts_explain = trimmed
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("explain"));
    if !starts_vacuum && !starts_explain {
        return None;
    }
    let normalized = trimmed
        .trim_end_matches(';')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    // Strip an optional `explain ` prefix, then require `vacuum`, then parse its options
    // (`FULL`/`ANALYZE`, bare or parenthesized). Anything else falls through to the generic parser.
    let (is_explain, rest) = normalized
        .strip_prefix("explain ")
        .map_or((false, normalized.as_str()), |rest| (true, rest));
    let after = rest.strip_prefix("vacuum")?;
    // `vacuum` must be a whole word: the tail is empty or begins with a space (the normalizer
    // collapses runs to single spaces) or `(`, so a glued token like `vacuumfull` is not VACUUM.
    if !(after.is_empty() || after.starts_with(' ') || after.starts_with('(')) {
        return None;
    }
    let options = parse_vacuum_options(after)?;
    let vacuum = ast::Statement::Vacuum(options);
    Some(if is_explain {
        ast::Statement::Explain(Box::new(vacuum), ast::ExplainOptions::default())
    } else {
        vacuum
    })
}

/// Recognize `REINDEX { INDEX | TABLE | SCHEMA | DATABASE | SYSTEM } [CONCURRENTLY] name` (with an
/// optional leading `( ... )` option list) and accept it as a no-op. sqlparser does not model
/// `REINDEX` as a statement, so — like `VACUUM` — it is recognized here before the generic parser.
///
/// NusaDB's clustered B-tree indexes are always consistent (MVCC + background purge), so there is
/// nothing to rebuild; accepting the command rather than rejecting it keeps the migration tools and
/// ORM health-checks that emit `REINDEX` working. Requires a non-empty target after the keyword so a
/// bare `REINDEX` or a glued token like `reindexes` is not mistaken for it (those fall through to the
/// generic parser, which rejects them).
fn recognize_reindex(sql: &str) -> Option<ast::Statement> {
    let trimmed = sql.trim().trim_end_matches(';');
    let after = trimmed
        .get(..7)
        .filter(|p| p.eq_ignore_ascii_case("reindex"))
        .and_then(|_| trimmed.get(7..))?;
    // `reindex` must be a whole word (the next char is a space or `(`), followed by a target.
    if !(after.starts_with(' ') || after.starts_with('(')) || after.trim().is_empty() {
        return None;
    }
    Some(ast::Statement::Reindex)
}

/// Parse the text after the `VACUUM` keyword into [`ast::VacuumOptions`]. Accepts an empty tail, a
/// space-separated `full`/`analyze` list or a parenthesized `(full, analyze)` list, each optionally
/// followed by a comma-separated table list (`VACUUM ANALYZE t`, `VACUUM (FULL) a, b`). The table
/// list is accepted but not per-table applied: NusaDB's reclamation is cluster-wide, so `VACUUM t`
/// runs the same global vacuum (which includes `t`) — the point is to accept the maintenance command
/// that migration tools and ORMs emit rather than reject it and break their scripts. Returns `None`
/// for an unrecognized option or a non-identifier table token, so anything genuinely malformed still
/// falls through to the generic parser and is rejected (no silent accept of garbage).
fn parse_vacuum_options(after: &str) -> Option<ast::VacuumOptions> {
    let trimmed = after.trim();
    let mut options = ast::VacuumOptions::default();
    // Split the option list (parenthesized, or leading bare `full`/`analyze` keywords) from an
    // optional trailing table list, and validate each; a non-identifier token means this is not a
    // `VACUUM` we understand, so it falls through to the generic parser (which rejects it).
    if let Some(rest) = trimmed.strip_prefix('(') {
        let (inner, tail) = rest.split_once(')')?;
        for token in inner.replace(',', " ").split_whitespace() {
            match token {
                "full" => options.full = true,
                "analyze" => options.analyze = true,
                _ => return None,
            }
        }
        if !vacuum_table_list_is_valid(tail.trim()) {
            return None;
        }
    } else {
        // Bare form: leading `full`/`analyze` are options; the first other token starts the tables.
        let mut table_tokens: Vec<&str> = Vec::new();
        for token in trimmed.split_whitespace() {
            match token {
                "full" if table_tokens.is_empty() => options.full = true,
                "analyze" if table_tokens.is_empty() => options.analyze = true,
                _ => table_tokens.push(token),
            }
        }
        if !vacuum_table_list_is_valid(&table_tokens.join(" ")) {
            return None;
        }
    }
    Some(options)
}

/// Whether an (optional) trailing `VACUUM` table list is empty or a comma-separated list of plain,
/// optionally schema-qualified identifiers. NusaDB reclaims cluster-wide rather than per table, so
/// the list is only validated and then discarded — accepting `VACUUM t` (which the global vacuum
/// covers) instead of rejecting the maintenance command tools emit.
fn vacuum_table_list_is_valid(tables: &str) -> bool {
    tables.is_empty()
        || tables.split(',').all(|name| {
            let name = name.trim();
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        })
}

/// Recognize `LOCK [TABLE] name [, ...] [IN {ACCESS SHARE | ACCESS EXCLUSIVE} MODE]`.
///
/// sqlparser 0.51 only models the `LOCK TABLES t READ|WRITE` form, not the `IN <mode> MODE` form
/// NusaDB's lock manager speaks, so this is recognized here (like `VACUUM`/`COMMENT`).
/// Returns `Some(Err(..))` for a recognized-but-malformed statement, `None` for anything that is not
/// a `LOCK` statement (so the generic parser still sees the `LOCK TABLES` form).
fn recognize_lock_table(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim();
    if !trimmed
        .get(..4)
        .is_some_and(|p| p.eq_ignore_ascii_case("lock"))
    {
        return None;
    }
    // Keep commas as their own tokens; keywords are matched case-insensitively and table names are
    // case-folded (NusaDB's unquoted-identifier rule). Quoted identifiers are not handled here.
    let body = trimmed.trim_end_matches(';');
    let spaced = body.replace(',', " , ");
    let mut tokens = spaced.split_whitespace().peekable();

    if !tokens
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("lock"))
    {
        return None;
    }
    if tokens
        .peek()
        .is_some_and(|w| w.eq_ignore_ascii_case("table"))
    {
        tokens.next();
    }

    // Table list: name [, name]* until `IN` or end.
    let mut tables: Vec<String> = Vec::new();
    loop {
        let Some(name) = tokens.next() else {
            return Some(Err(Error::Syntax(
                "LOCK TABLE requires a table name".to_owned(),
            )));
        };
        let folded = fold_lock_table_name(name)?;
        tables.push(folded);
        match tokens.peek() {
            Some(&",") => {
                tokens.next();
            },
            Some(w) if w.eq_ignore_ascii_case("in") => break,
            None => break,
            _ => return None, // an unsupported trailer (e.g. NOWAIT) — leave it to the generic parser
        }
    }

    // Optional `IN <mode> MODE`; absent → ACCESS EXCLUSIVE (the conventional default).
    let mode = if tokens.peek().is_some_and(|w| w.eq_ignore_ascii_case("in")) {
        tokens.next();
        let mut words: Vec<String> = Vec::new();
        loop {
            let Some(w) = tokens.next() else {
                return Some(Err(Error::Syntax(
                    "LOCK TABLE ... IN <mode> MODE: missing MODE keyword".to_owned(),
                )));
            };
            if w.eq_ignore_ascii_case("mode") {
                break;
            }
            words.push(w.to_ascii_lowercase());
        }
        match words.join(" ").as_str() {
            "access share" => ast::LockMode::AccessShare,
            "access exclusive" => ast::LockMode::AccessExclusive,
            other => {
                return Some(Err(Error::Unsupported(format!(
                    "LOCK TABLE mode `{other}` is not supported; use ACCESS SHARE or ACCESS EXCLUSIVE"
                ))));
            },
        }
    } else {
        ast::LockMode::default()
    };

    if tokens.next().is_some() {
        return None; // trailing tokens after MODE (e.g. NOWAIT) — not supported here
    }
    Some(Ok(ast::Statement::LockTable { tables, mode }))
}

/// Validate a `LOCK TABLE` name token as a plain (optionally schema-qualified) identifier and fold it
/// to lowercase. Returns `None` for a quoted or otherwise non-plain token, so the caller defers to
/// the generic parser rather than mis-parsing it.
fn fold_lock_table_name(token: &str) -> Option<String> {
    if token == "," || token.is_empty() {
        return None;
    }
    for part in token.split('.') {
        let mut chars = part.chars();
        let first = chars.next()?;
        if !(first.is_ascii_alphabetic() || first == '_') {
            return None;
        }
        if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
            return None;
        }
    }
    Some(token.to_ascii_lowercase())
}

/// Fold a LISTEN/NOTIFY channel name: a bare identifier is lowercased (NusaDB's unquoted rule) and a
/// double-quoted identifier is taken verbatim with `""` unescaped (so `"MyChan"` keeps its case and
/// may contain spaces). Returns `None` for anything that is not a single valid channel name.
fn fold_channel_name(token: &str) -> Option<String> {
    if let Some(rest) = token.strip_prefix('"') {
        // A quoted identifier: the last char must be the closing quote, and any interior `"` must be
        // doubled (`""`). Reject an unbalanced or prematurely-closed quote.
        let inner = rest.strip_suffix('"')?;
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    out.push('"');
                } else {
                    return None; // a lone interior quote closes the identifier early
                }
            } else {
                out.push(c);
            }
        }
        if out.is_empty() {
            return None;
        }
        return Some(out);
    }
    let mut chars = token.chars();
    let first = chars.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
        return None;
    }
    Some(token.to_ascii_lowercase())
}

/// The channel text after a `LISTEN`/`UNLISTEN`/`NOTIFY` keyword: the input trimmed, with a trailing
/// `;` removed and re-trimmed. Returns `None` (via the caller's `?`) only for the empty tail.
fn stmt_tail<'a>(sql: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = sql.trim();
    // Whole-word guard: the first word must be exactly the keyword (case-insensitively), so an
    // identifier merely starting with it (e.g. `listener`) is not mistaken for the statement.
    let rest = trimmed.get(..keyword.len())?;
    if !rest.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let after = &trimmed[keyword.len()..];
    // The next char must end the word (space) or the statement (`;`/end) — else it is a longer token.
    if !after.is_empty() && !after.starts_with(|c: char| c.is_whitespace() || c == ';') {
        return None;
    }
    Some(after.trim().trim_end_matches(';').trim())
}

/// Recognize `LISTEN channel` (async pub/sub). The channel is a folded identifier; the server owns
/// the registry and intercepts this before the SQL engine.
fn recognize_listen(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let tail = stmt_tail(sql, "listen")?;
    if tail.is_empty() {
        return Some(unsupported("LISTEN requires a channel name"));
    }
    Some(
        fold_channel_name(tail)
            .map(ast::Statement::Listen)
            .ok_or_else(|| Error::Syntax(format!("invalid LISTEN channel: {tail}"))),
    )
}

/// Recognize `UNLISTEN channel` and `UNLISTEN *` (stop listening on one channel or all).
fn recognize_unlisten(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let tail = stmt_tail(sql, "unlisten")?;
    if tail.is_empty() {
        return Some(unsupported("UNLISTEN requires a channel name or *"));
    }
    if tail == "*" {
        return Some(Ok(ast::Statement::Unlisten(None)));
    }
    Some(
        fold_channel_name(tail)
            .map(|channel| ast::Statement::Unlisten(Some(channel)))
            .ok_or_else(|| Error::Syntax(format!("invalid UNLISTEN channel: {tail}"))),
    )
}

/// Recognize `NOTIFY channel [, 'payload']` (send a notification). The optional payload is a
/// single-quoted string literal with `''` unescaped to `'`.
fn recognize_notify(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let tail = stmt_tail(sql, "notify")?;
    if tail.is_empty() {
        return Some(unsupported("NOTIFY requires a channel name"));
    }
    // Split the channel from an optional `, 'payload'`. The channel is a bare identifier (no comma),
    // so the first comma separates it from the payload; a quoted channel with an interior comma is
    // not supported (a rare edge — the payload form covers the common case).
    let (chan_tok, payload) = match tail.split_once(',') {
        Some((chan, rest)) => {
            let literal = rest.trim();
            let Some(payload) = parse_notify_payload(literal) else {
                return Some(Err(Error::Syntax(format!(
                    "NOTIFY payload must be a string literal: {literal}"
                ))));
            };
            (chan.trim(), Some(payload))
        },
        None => (tail, None),
    };
    Some(
        fold_channel_name(chan_tok)
            .map(|channel| ast::Statement::Notify { channel, payload })
            .ok_or_else(|| Error::Syntax(format!("invalid NOTIFY channel: {chan_tok}"))),
    )
}

/// Parse a single-quoted SQL string literal into its value, unescaping doubled quotes (`''` -> `'`).
/// Returns `None` if the text is not a well-formed `'...'` literal.
fn parse_notify_payload(literal: &str) -> Option<String> {
    let inner = literal.strip_prefix('\'')?.strip_suffix('\'')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
                out.push('\'');
            } else {
                return None; // a lone interior quote closes the literal early
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Recognize `RESET <name>`, which the generic tokenizer does not model as a statement.
/// Modelled as a [`ast::SetVariable`] with `value: None` (reset to default). The first-word
/// guard allocates nothing; only a `RESET …` input reaches the splitting path.
fn recognize_reset(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim();
    if !trimmed
        .get(..5)
        .is_some_and(|p| p.eq_ignore_ascii_case("reset"))
    {
        return None;
    }
    let mut words = trimmed.trim_end_matches(';').split_whitespace();
    let first = words.next()?;
    if !first.eq_ignore_ascii_case("reset") {
        return None; // e.g. an identifier merely starting with "reset"
    }
    let Some(name) = words.next() else {
        return Some(unsupported("RESET requires a variable name"));
    };
    if words.next().is_some() {
        return Some(unsupported("RESET takes a single variable name"));
    }
    Some(Ok(ast::Statement::SetVariable(ast::SetVariable {
        name: name.to_ascii_lowercase(),
        value: None,
    })))
}

/// Recognize `REFRESH MATERIALIZED VIEW <name>`. sqlparser 0.51 has no `REFRESH` keyword, so
/// parse the fixed shape ourselves. Case-insensitive; tolerant of whitespace and a trailing `;`. The
/// view name is folded like any identifier (unquoted → lowercase, `"quoted"` preserved) so it
/// matches the backing table created by `CREATE MATERIALIZED VIEW`.
fn recognize_refresh(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim();
    if !trimmed
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("refresh"))
    {
        return None;
    }
    let mut words = trimmed.trim_end_matches(';').split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("refresh") {
        return None; // an identifier merely starting with "refresh"
    }
    match (words.next(), words.next()) {
        (Some(m), Some(v))
            if m.eq_ignore_ascii_case("materialized") && v.eq_ignore_ascii_case("view") => {},
        _ => {
            return Some(unsupported(
                "REFRESH must be `REFRESH MATERIALIZED VIEW <name>`",
            ));
        },
    }
    let Some(raw) = words.next() else {
        return Some(unsupported(
            "REFRESH MATERIALIZED VIEW requires a view name",
        ));
    };
    if words.next().is_some() {
        return Some(unsupported(
            "REFRESH MATERIALIZED VIEW takes a single view name",
        ));
    }
    let name = if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
        raw[1..raw.len() - 1].to_owned()
    } else {
        raw.to_ascii_lowercase()
    };
    Some(Ok(ast::Statement::RefreshMaterializedView(name)))
}

/// Recognize `DROP MATERIALIZED VIEW [IF EXISTS] <name>` — sqlparser 0.51 has no `MATERIALIZED`
/// object kind, so it rejects this spelling after `DROP`. A materialized view's backing store is an
/// ordinary table, so this routes to the same [`ast::DropView`] the generic `DROP VIEW` produces.
/// Returns `None` for any other statement (including plain `DROP VIEW`) so it falls through.
fn recognize_drop_matview(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim();
    let mut words = trimmed.trim_end_matches(';').split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("drop") {
        return None;
    }
    // Only intercept the MATERIALIZED form; `DROP VIEW`/`DROP TABLE`/... fall through.
    if !words
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("materialized"))
    {
        return None;
    }
    if !words.next().is_some_and(|w| w.eq_ignore_ascii_case("view")) {
        return Some(unsupported(
            "DROP MATERIALIZED must be `DROP MATERIALIZED VIEW <name>`",
        ));
    }
    let mut rest: Vec<&str> = words.collect();
    let if_exists = matches!(
        rest.as_slice(),
        [a, b, ..] if a.eq_ignore_ascii_case("if") && b.eq_ignore_ascii_case("exists")
    );
    if if_exists {
        rest.drain(..2);
    }
    let [raw] = rest.as_slice() else {
        return Some(unsupported(
            "DROP MATERIALIZED VIEW requires a single view name",
        ));
    };
    let name = if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
        raw[1..raw.len() - 1].to_owned()
    } else {
        raw.to_ascii_lowercase()
    };
    Some(Ok(ast::Statement::DropView(ast::DropView {
        name,
        if_exists,
    })))
}

/// True when `sql` begins with the two given keywords (case-insensitive), the cheap guard the policy
/// recognizers use before any allocation.
fn starts_with_two(sql: &str, first: &str, second: &str) -> bool {
    let mut words = sql.split_whitespace();
    words.next().is_some_and(|w| w.eq_ignore_ascii_case(first))
        && words.next().is_some_and(|w| w.eq_ignore_ascii_case(second))
}

/// Whether `sql`'s first three whitespace-delimited words match `first`/`second`/`third`
/// (case-insensitive) — for custom-recognized statements whose lead is three keywords (e.g.
/// `FIX DROP DATABASE`).
fn starts_with_three(sql: &str, first: &str, second: &str, third: &str) -> bool {
    let mut words = sql.split_whitespace();
    words.next().is_some_and(|w| w.eq_ignore_ascii_case(first))
        && words.next().is_some_and(|w| w.eq_ignore_ascii_case(second))
        && words.next().is_some_and(|w| w.eq_ignore_ascii_case(third))
}

/// Recognize `CREATE POLICY ...`. sqlparser 0.51 does not model RLS policies, so it would
/// reject the statement; drive its parser primitives ourselves instead.
fn recognize_create_policy(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "create", "policy").then(|| parse_create_policy(sql))
}

/// Recognize `DROP POLICY [IF EXISTS] name ON table`.
fn recognize_drop_policy(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "drop", "policy").then(|| parse_drop_policy(sql))
}

/// Recognize `ALTER POLICY ...` and hand it to [`parse_alter_policy`]; `None` otherwise.
fn recognize_alter_policy(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "alter", "policy").then(|| parse_alter_policy(sql))
}

/// Parse `( <expr> )` via sqlparser and render the predicate back to canonical SQL text, so a policy
/// can be re-analyzed against its table at query time.
fn parse_parenthesized_predicate(parser: &mut Parser) -> Result<String, Error> {
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;
    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    parser.expect_token(&Token::LParen).map_err(syntax)?;
    let expr = parser.parse_expr().map_err(syntax)?;
    parser.expect_token(&Token::RParen).map_err(syntax)?;
    Ok(expr.to_string())
}

/// Reject any trailing content after a manually-driven statement (matching the single-statement
/// contract of [`parse`]). A trailing `;` is allowed.
fn expect_statement_end(parser: &mut Parser) -> Result<(), Error> {
    use sqlparser::tokenizer::Token;
    loop {
        match parser.next_token().token {
            Token::EOF => return Ok(()),
            Token::SemiColon => {},
            other => return Err(Error::Syntax(format!("unexpected trailing token: {other}"))),
        }
    }
}

/// Drive `CREATE POLICY name ON table [AS { PERMISSIVE | RESTRICTIVE }] [FOR cmd] [TO role[, ...]]
/// [USING (expr)] [WITH CHECK (expr)]`. `AS` defaults to `PERMISSIVE`. The predicates are
/// captured as canonical SQL text and re-analyzed against the table by the analyzer/enforcer.
fn parse_create_policy(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::CREATE).map_err(syntax)?;
    parser.expect_keyword(Keyword::POLICY).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    // `AS PERMISSIVE | AS RESTRICTIVE` (PERMISSIVE/RESTRICTIVE are not sqlparser keywords, so they
    // arrive as identifiers); the default when `AS` is omitted is PERMISSIVE.
    let permissive = if parser.parse_keyword(Keyword::AS) {
        let kind = fold_ident(&parser.parse_identifier().map_err(syntax)?);
        match kind.as_str() {
            "permissive" => true,
            "restrictive" => false,
            _ => return unsupported("CREATE POLICY AS expects PERMISSIVE or RESTRICTIVE"),
        }
    } else {
        true
    };

    let command = if parser.parse_keyword(Keyword::FOR) {
        match parser.parse_one_of_keywords(&[
            Keyword::ALL,
            Keyword::SELECT,
            Keyword::INSERT,
            Keyword::UPDATE,
            Keyword::DELETE,
        ]) {
            Some(Keyword::ALL) => ast::PolicyCommand::All,
            Some(Keyword::SELECT) => ast::PolicyCommand::Select,
            Some(Keyword::INSERT) => ast::PolicyCommand::Insert,
            Some(Keyword::UPDATE) => ast::PolicyCommand::Update,
            Some(Keyword::DELETE) => ast::PolicyCommand::Delete,
            _ => {
                return unsupported(
                    "CREATE POLICY FOR expects ALL, SELECT, INSERT, UPDATE, or DELETE",
                );
            },
        }
    } else {
        ast::PolicyCommand::All
    };

    let roles = if parser.parse_keyword(Keyword::TO) {
        parser
            .parse_comma_separated(Parser::parse_identifier)
            .map_err(syntax)?
            .iter()
            .map(fold_ident)
            .collect()
    } else {
        Vec::new()
    };

    let using = if parser.parse_keyword(Keyword::USING) {
        Some(parse_parenthesized_predicate(&mut parser)?)
    } else {
        None
    };
    let check = if parser.parse_keywords(&[Keyword::WITH, Keyword::CHECK]) {
        Some(parse_parenthesized_predicate(&mut parser)?)
    } else {
        None
    };
    if using.is_none() && check.is_none() {
        return unsupported("CREATE POLICY requires a USING and/or WITH CHECK clause");
    }

    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::CreatePolicy(ast::CreatePolicy {
        name,
        table,
        permissive,
        command,
        roles,
        using,
        check,
    }))
}

/// Drive `DROP POLICY [IF EXISTS] name ON table`.
fn parse_drop_policy(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::POLICY).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::DropPolicy(ast::DropPolicy {
        name,
        table,
        if_exists,
    }))
}

/// Drive `ALTER POLICY name ON table [TO role[, ...]] [USING (expr)] [WITH CHECK (expr)]`. At
/// least one of `TO` / `USING` / `WITH CHECK` must be given; the `RENAME TO` form is not yet
/// supported. An omitted clause leaves that part of the policy unchanged (the analyzer merges).
fn parse_alter_policy(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::ALTER).map_err(syntax)?;
    parser.expect_keyword(Keyword::POLICY).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    if parser.parse_keyword(Keyword::RENAME) {
        return unsupported("ALTER POLICY ... RENAME TO is not yet supported");
    }

    let roles = if parser.parse_keyword(Keyword::TO) {
        Some(
            parser
                .parse_comma_separated(Parser::parse_identifier)
                .map_err(syntax)?
                .iter()
                .map(fold_ident)
                .collect(),
        )
    } else {
        None
    };
    let using = if parser.parse_keyword(Keyword::USING) {
        Some(parse_parenthesized_predicate(&mut parser)?)
    } else {
        None
    };
    let check = if parser.parse_keywords(&[Keyword::WITH, Keyword::CHECK]) {
        Some(parse_parenthesized_predicate(&mut parser)?)
    } else {
        None
    };
    if roles.is_none() && using.is_none() && check.is_none() {
        return unsupported("ALTER POLICY requires at least one of TO, USING, or WITH CHECK");
    }

    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::AlterPolicy(ast::AlterPolicy {
        name,
        table,
        roles,
        using,
        check,
    }))
}

/// Recognize `CREATE [OR REPLACE] TRIGGER ...`; `None` if not a `CREATE TRIGGER` statement
/// (so it falls through to the generic parser). Allocates nothing for the cheap first-words scan.
fn recognize_create_trigger(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let mut words = sql.split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("create") {
        return None;
    }
    let mut w = words.next()?;
    if w.eq_ignore_ascii_case("or") {
        if !words.next()?.eq_ignore_ascii_case("replace") {
            return None;
        }
        w = words.next()?;
    }
    w.eq_ignore_ascii_case("trigger")
        .then(|| parse_create_trigger(sql))
}

/// Recognize `DROP TRIGGER [IF EXISTS] name ON table`.
fn recognize_drop_trigger(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "drop", "trigger").then(|| parse_drop_trigger(sql))
}

/// Recognize `ALTER TRIGGER name ON table RENAME TO new_name`.
fn recognize_alter_trigger(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "alter", "trigger").then(|| parse_alter_trigger(sql))
}

/// Recognize `CREATE TYPE name AS ENUM (...)` (B-ENUM); `None` for the composite `AS (...)` form (the
/// generic parser handles that) or any other statement.
fn recognize_create_type_enum(sql: &str) -> Option<Result<ast::Statement, Error>> {
    if !starts_with_two(sql, "create", "type") {
        return None;
    }
    // Only the ENUM form — require an `AS ENUM` keyword pair so the composite form falls through.
    let lc = sql.to_ascii_lowercase();
    let tokens: Vec<&str> = lc.split_whitespace().collect();
    let is_enum = tokens
        .windows(2)
        .any(|w| matches!(w, [a, b] if *a == "as" && (*b == "enum" || b.starts_with("enum("))));
    is_enum.then(|| parse_create_type_enum(sql))
}

/// Recognize `DROP TYPE [IF EXISTS] name` (B-ENUM).
fn recognize_drop_type(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "drop", "type").then(|| parse_drop_type(sql))
}

/// Recognize `DROP DATABASE [IF EXISTS] name [FORCE]` and the `FIX DROP DATABASE [IF EXISTS] name`
/// alias (sqlparser 0.51 has no DATABASE drop grammar, and `FIX` is not SQL, so it would reject
/// them). Bare `DROP DATABASE` backs up every table before dropping; the trailing `FORCE` keyword
/// (canonical) or leading `FIX` (alias) drops permanently without a backup.
fn recognize_drop_database(sql: &str) -> Option<Result<ast::Statement, Error>> {
    if starts_with_three(sql, "fix", "drop", "database") {
        // Strip the leading `FIX` word and parse the rest as a forced DROP DATABASE.
        let rest = sql
            .trim_start()
            .split_once(char::is_whitespace)
            .map_or("", |(_fix, rest)| rest);
        Some(parse_drop_database(rest, true))
    } else if starts_with_two(sql, "drop", "database") {
        Some(parse_drop_database(sql, false))
    } else {
        None
    }
}

/// Drive `DROP DATABASE [IF EXISTS] name [FORCE]`. `force_prefix` is set for the `FIX DROP DATABASE`
/// alias; a trailing `FORCE` keyword (`DROP DATABASE name FORCE`) is the canonical no-backup form.
fn parse_drop_database(sql: &str, force_prefix: bool) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::DATABASE).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    // `DROP DATABASE name FORCE` (canonical) or the `FIX DROP DATABASE name` alias → drop without a
    // backup; bare `DROP DATABASE name` → backup-then-drop.
    let force = force_prefix || parser.parse_keyword(Keyword::FORCE);
    Ok(ast::Statement::DropDatabase(ast::DropDatabase {
        name,
        if_exists,
        force,
    }))
}

/// Recognize `ALTER DATABASE name ...` — a single-database compatibility no-op. sqlparser 0.51's
/// `ALTER DATABASE` grammar is narrow, so accept any tail ourselves and ignore it.
fn recognize_alter_database(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "alter", "database").then(|| parse_alter_database(sql))
}

/// Drive `ALTER DATABASE name ...`: extract the name, ignore the rest (no database-level options
/// exist to apply in a single-database engine).
fn parse_alter_database(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::ALTER).map_err(syntax)?;
    parser.expect_keyword(Keyword::DATABASE).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    Ok(ast::Statement::AlterDatabase(ast::AlterDatabase { name }))
}

/// Drive `CREATE TYPE name AS ENUM ('a', 'b', ...)` (B-ENUM). The labels are single-quoted string
/// literals in declaration order; a trailing comma before `)` is allowed.
fn parse_create_type_enum(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::CREATE).map_err(syntax)?;
    parser.expect_keyword(Keyword::TYPE).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::AS).map_err(syntax)?;
    parser.expect_keyword(Keyword::ENUM).map_err(syntax)?;
    parser.expect_token(&Token::LParen).map_err(syntax)?;
    let mut labels = Vec::new();
    loop {
        labels.push(parser.parse_literal_string().map_err(syntax)?);
        if parser.consume_token(&Token::Comma) {
            if parser.peek_token().token == Token::RParen {
                break; // trailing comma
            }
            continue;
        }
        break;
    }
    parser.expect_token(&Token::RParen).map_err(syntax)?;
    Ok(ast::Statement::CreateEnum(ast::CreateEnum { name, labels }))
}

/// Drive `DROP TYPE [IF EXISTS] name` (B-ENUM).
fn parse_drop_type(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::TYPE).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    Ok(ast::Statement::DropType(ast::DropType { name, if_exists }))
}

/// Drive `CREATE [OR REPLACE] TRIGGER name {BEFORE|AFTER} {INSERT|UPDATE|DELETE}[ OR ...] ON table
/// [FOR EACH {ROW|STATEMENT}] [WHEN (cond)] <triggered-statement>`. The `WHEN` predicate and
/// the action are captured as canonical SQL text (so they can be re-parsed, `NEW`/`OLD`-substituted,
/// and run when the trigger fires). The action must be a single data statement.
fn parse_create_trigger(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::CREATE).map_err(syntax)?;
    let or_replace = parser.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
    parser.expect_keyword(Keyword::TRIGGER).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);

    let timing = if parser.parse_keyword(Keyword::BEFORE) {
        ast::TriggerTiming::Before
    } else if parser.parse_keyword(Keyword::AFTER) {
        ast::TriggerTiming::After
    } else if parser.parse_keywords(&[Keyword::INSTEAD, Keyword::OF]) {
        return unsupported(
            "INSTEAD OF triggers require updatable views, which NusaDB does not support",
        );
    } else {
        return unsupported("trigger timing must be BEFORE or AFTER");
    };

    let mut events = Vec::new();
    loop {
        let event = match parser.parse_one_of_keywords(&[
            Keyword::INSERT,
            Keyword::UPDATE,
            Keyword::DELETE,
        ]) {
            Some(Keyword::INSERT) => ast::TriggerEvent::Insert,
            Some(Keyword::UPDATE) => ast::TriggerEvent::Update,
            Some(Keyword::DELETE) => ast::TriggerEvent::Delete,
            _ => return unsupported("trigger event must be INSERT, UPDATE, or DELETE"),
        };
        if !events.contains(&event) {
            events.push(event);
        }
        if !parser.parse_keyword(Keyword::OR) {
            break;
        }
    }

    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    // `FOR EACH ROW | STATEMENT` is optional; it defaults to ROW (the form that binds NEW/OLD).
    let for_each = if parser.parse_keyword(Keyword::FOR) {
        parser.expect_keyword(Keyword::EACH).map_err(syntax)?;
        if parser.parse_keyword(Keyword::ROW) {
            ast::TriggerForEach::Row
        } else if parser.parse_keyword(Keyword::STATEMENT) {
            ast::TriggerForEach::Statement
        } else {
            return unsupported("FOR EACH expects ROW or STATEMENT");
        }
    } else {
        ast::TriggerForEach::Row
    };

    let when = if parser.parse_keyword(Keyword::WHEN) {
        Some(parse_parenthesized_predicate(&mut parser)?)
    } else {
        None
    };

    // The triggered action is the trailing SQL statement, captured as canonical SQL.
    let action_stmt = parser.parse_statement().map_err(syntax)?;
    let action = action_stmt.to_string();
    let converted = convert_statement(action_stmt)?;
    if !matches!(
        converted,
        ast::Statement::Insert(_)
            | ast::Statement::Update(_)
            | ast::Statement::Delete(_)
            | ast::Statement::Select(_)
            | ast::Statement::SetOperation(_)
    ) {
        return unsupported(
            "a trigger action must be an INSERT, UPDATE, DELETE, or SELECT statement",
        );
    }

    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::CreateTrigger(ast::CreateTrigger {
        name,
        or_replace,
        timing,
        events,
        table,
        for_each,
        when,
        action,
    }))
}

/// Drive `DROP TRIGGER [IF EXISTS] name ON table`.
fn parse_drop_trigger(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::TRIGGER).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::DropTrigger(ast::DropTrigger {
        name,
        table,
        if_exists,
    }))
}

/// Drive `ALTER TRIGGER name ON table RENAME TO new_name` — the only `ALTER TRIGGER` form
/// (enable/disable ride `ALTER TABLE ... {ENABLE|DISABLE} TRIGGER`, matching the reference
/// surface).
fn parse_alter_trigger(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::ALTER).map_err(syntax)?;
    parser.expect_keyword(Keyword::TRIGGER).map_err(syntax)?;
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    if !parser.parse_keywords(&[Keyword::RENAME, Keyword::TO]) {
        return unsupported(
            "ALTER TRIGGER supports only RENAME TO; use ALTER TABLE ... {ENABLE|DISABLE} TRIGGER \
             to enable or disable a trigger",
        );
    }
    let new_name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::AlterTrigger(ast::AlterTrigger {
        name,
        table,
        new_name,
    }))
}

/// Recognize `CREATE [OR REPLACE] PROCEDURE ...`; `None` if not a `CREATE PROCEDURE`.
fn recognize_create_procedure(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let mut words = sql.split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("create") {
        return None;
    }
    let mut w = words.next()?;
    if w.eq_ignore_ascii_case("or") {
        if !words.next()?.eq_ignore_ascii_case("replace") {
            return None;
        }
        w = words.next()?;
    }
    w.eq_ignore_ascii_case("procedure")
        .then(|| parse_create_procedure(sql))
}

/// Recognize `DROP PROCEDURE [IF EXISTS] name`.
fn recognize_drop_procedure(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "drop", "procedure").then(|| parse_drop_procedure(sql))
}

/// Recognize `CALL name(args)`.
fn recognize_call(sql: &str) -> Option<Result<ast::Statement, Error>> {
    sql.split_whitespace()
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("call"))
        .then(|| parse_call(sql))
}

/// Parse a body of one or more `;`-separated statements into internal [`ast::Statement`]s — the
/// stored-procedure body. Uses the generic parser (the body holds plain data statements).
pub(crate) fn parse_statements(sql: &str) -> Result<Vec<ast::Statement>, Error> {
    let dialect = NusaParserDialect;
    Parser::parse_sql(&dialect, sql)
        .map_err(|e| Error::Syntax(e.to_string()))?
        .into_iter()
        .map(convert_statement)
        .collect()
}

/// Whether `stmt` is a statement a procedure body may contain: data statements, or a nested `CALL`
/// of another procedure (composition; bounded by the call-depth guard).
const fn is_procedure_body_statement(stmt: &ast::Statement) -> bool {
    matches!(
        stmt,
        ast::Statement::Insert(_)
            | ast::Statement::Update(_)
            | ast::Statement::Delete(_)
            | ast::Statement::Select(_)
            | ast::Statement::SetOperation(_)
            | ast::Statement::Call(_)
    )
}

/// Drive `CREATE [OR REPLACE] PROCEDURE name([p type, ...]) [LANGUAGE SQL] AS <body>`. The
/// body is a `$$…$$` or `'…'` block of `;`-separated data statements that reference call arguments
/// positionally as `$1`..`$n` (like a prepared statement); it is captured verbatim and validated.
fn parse_create_procedure(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::CREATE).map_err(syntax)?;
    let or_replace = parser.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
    parser.expect_keyword(Keyword::PROCEDURE).map_err(syntax)?;
    let name = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    let mut params = Vec::new();
    parser.expect_token(&Token::LParen).map_err(syntax)?;
    if !parser.consume_token(&Token::RParen) {
        loop {
            // Optional `IN` (default) / `OUT` direction before the parameter name.
            let out = if parser.parse_keyword(Keyword::OUT) {
                true
            } else {
                let _ = parser.parse_keyword(Keyword::IN); // `IN` is the optional default direction.
                false
            };
            let pname = fold_ident(&parser.parse_identifier().map_err(syntax)?);
            let ty = convert_data_type(&parser.parse_data_type().map_err(syntax)?)?;
            params.push(ast::ProcedureParam {
                name: pname,
                ty,
                out,
            });
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
        parser.expect_token(&Token::RParen).map_err(syntax)?;
    }
    // `$1..$n` in the body bind the IN parameters (in IN-only order).
    let in_count = params.iter().filter(|p| !p.out).count();

    if parser.parse_keyword(Keyword::LANGUAGE) {
        let lang = fold_ident(&parser.parse_identifier().map_err(syntax)?);
        if lang != "sql" {
            return unsupported("CREATE PROCEDURE supports only LANGUAGE SQL");
        }
    }
    parser.expect_keyword(Keyword::AS).map_err(syntax)?;
    let body = match parser.next_token().token {
        Token::SingleQuotedString(s) => s,
        Token::DollarQuotedString(dq) => dq.value,
        other => {
            return Err(Error::Syntax(format!(
                "expected a quoted procedure body after AS, found {other}"
            )));
        },
    };
    expect_statement_end(&mut parser)?;

    if is_script(&body) {
        // A NusaScript `BEGIN ... END` block: validate that it parses; parameter/variable
        // binding is checked when the procedure is called.
        parse_script(&body)?;
    } else {
        // A plain sequence of SQL data statements.
        let statements = parse_statements(&body)?;
        if statements.is_empty() {
            return unsupported("a procedure body must contain at least one statement");
        }
        for stmt in &statements {
            if !is_procedure_body_statement(stmt) {
                return unsupported(
                    "a procedure body may contain only INSERT, UPDATE, DELETE, CALL, or SELECT \
                     statements",
                );
            }
            if crate::params::parameter_count(stmt) > in_count {
                return unsupported(
                    "a procedure body references more parameters than are declared IN",
                );
            }
        }
    }
    Ok(ast::Statement::CreateProcedure(ast::CreateProcedure {
        name,
        or_replace,
        params,
        body,
    }))
}

/// Drive `DROP PROCEDURE [IF EXISTS] name`.
fn parse_drop_procedure(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::PROCEDURE).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::DropProcedure(ast::DropProcedure {
        name,
        if_exists,
    }))
}

/// Drive `CALL name(arg, ...)`. Arguments are arbitrary expressions; the analyzer reduces
/// them to constants (each must bind a `$n` placeholder before the body is analyzed).
fn parse_call(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::CALL).map_err(syntax)?;
    let name = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    let mut args = Vec::new();
    parser.expect_token(&Token::LParen).map_err(syntax)?;
    if !parser.consume_token(&Token::RParen) {
        loop {
            args.push(convert_expr(parser.parse_expr().map_err(syntax)?)?);
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
        parser.expect_token(&Token::RParen).map_err(syntax)?;
    }
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::Call(ast::Call { name, args }))
}

/// Recognize `CREATE [OR REPLACE] FUNCTION ...`; `None` if not a `CREATE FUNCTION`.
fn recognize_create_function(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let mut words = sql.split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("create") {
        return None;
    }
    let mut w = words.next()?;
    if w.eq_ignore_ascii_case("or") {
        if !words.next()?.eq_ignore_ascii_case("replace") {
            return None;
        }
        w = words.next()?;
    }
    w.eq_ignore_ascii_case("function")
        .then(|| parse_create_function(sql))
}

/// Recognize `DROP FUNCTION [IF EXISTS] name`.
fn recognize_drop_function(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_two(sql, "drop", "function").then(|| parse_drop_function(sql))
}

/// Drive `CREATE [OR REPLACE] FUNCTION name(p type, ...) RETURNS type [LANGUAGE SQL] AS <body>`.
/// The body must be a `SELECT <expr>` with a single scalar projection and no `FROM`; its
/// expression is inlined (with `$1..$n` replaced by the call arguments) at each call site.
fn parse_create_function(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::CREATE).map_err(syntax)?;
    let or_replace = parser.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
    parser.expect_keyword(Keyword::FUNCTION).map_err(syntax)?;
    let name = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    let mut params = Vec::new();
    parser.expect_token(&Token::LParen).map_err(syntax)?;
    if !parser.consume_token(&Token::RParen) {
        loop {
            let pname = fold_ident(&parser.parse_identifier().map_err(syntax)?);
            let ty = convert_data_type(&parser.parse_data_type().map_err(syntax)?)?;
            params.push(ast::ProcedureParam {
                name: pname,
                ty,
                out: false,
            });
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
        parser.expect_token(&Token::RParen).map_err(syntax)?;
    }

    parser.expect_keyword(Keyword::RETURNS).map_err(syntax)?;
    let return_type = convert_data_type(&parser.parse_data_type().map_err(syntax)?)?;

    if parser.parse_keyword(Keyword::LANGUAGE) {
        let lang = fold_ident(&parser.parse_identifier().map_err(syntax)?);
        if lang != "sql" {
            return unsupported("CREATE FUNCTION supports only LANGUAGE SQL");
        }
    }
    parser.expect_keyword(Keyword::AS).map_err(syntax)?;
    let body = match parser.next_token().token {
        Token::SingleQuotedString(s) => s,
        Token::DollarQuotedString(dq) => dq.value,
        other => {
            return Err(Error::Syntax(format!(
                "expected a quoted function body after AS, found {other}"
            )));
        },
    };
    expect_statement_end(&mut parser)?;

    // The body must be `SELECT <expr>` — one scalar projection, no FROM — and reference at most the
    // declared parameters.
    let ast::Statement::Select(select) = parse(&body)? else {
        return unsupported("a function body must be a `SELECT <expr>` statement");
    };
    if select.from.is_some()
        || select.projection.len() != 1
        || !matches!(
            select.projection.first(),
            Some(ast::SelectItem::Expr { .. })
        )
    {
        return unsupported(
            "a function body must be `SELECT <expr>` — a single scalar expression with no FROM",
        );
    }
    if crate::params::parameter_count(&ast::Statement::Select(select)) > params.len() {
        return unsupported("a function body references more parameters than are declared");
    }
    Ok(ast::Statement::CreateFunction(ast::CreateFunction {
        name,
        or_replace,
        params,
        return_type,
        body,
    }))
}

/// Drive `DROP FUNCTION [IF EXISTS] name`.
fn parse_drop_function(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::DROP).map_err(syntax)?;
    parser.expect_keyword(Keyword::FUNCTION).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let name = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::DropFunction(ast::DropFunction {
        name,
        if_exists,
    }))
}

/// Convert a `CALL name(args)` parsed by the generic parser (sqlparser models it as a `Function`)
/// into [`ast::Call`] — used when a `CALL` appears inside a procedure body.
fn convert_call_statement(func: sql::Function) -> Result<ast::Statement, Error> {
    let name = object_name(&func.name)?;
    let mut args = Vec::new();
    if let sql::FunctionArguments::List(list) = func.args {
        for arg in list.args {
            match arg {
                sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => {
                    args.push(convert_expr(e)?);
                },
                _ => return unsupported("CALL argument must be a plain expression"),
            }
        }
    }
    Ok(ast::Statement::Call(ast::Call { name, args }))
}

/// Parse a standalone boolean expression — a policy `USING`/`WITH CHECK` predicate stored as SQL —
/// into the internal [`ast::Expr`], so it can be re-analyzed against the target table.
pub(crate) fn parse_expression(sql: &str) -> Result<ast::Expr, Error> {
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    let expr = parser.parse_expr().map_err(syntax)?;
    if !matches!(parser.peek_token().token, Token::EOF | Token::SemiColon) {
        return unsupported("a policy predicate must be a single expression");
    }
    convert_expr(expr)
}

/// Recognize a `COMMENT ON ...` statement (or `EXPLAIN COMMENT ON ...`) and hand it to
/// [`parse_comment_on`]. Returns `None` for any other statement so it falls through to the
/// generic parser. The first-word scan allocates nothing.
fn recognize_comment(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim_start();
    let mut words = trimmed.split_whitespace();
    let first = words.next()?;
    if first.eq_ignore_ascii_case("comment") {
        return Some(parse_comment_on(trimmed));
    }
    // `EXPLAIN COMMENT ON ...` wraps the comment statement (mirrors `EXPLAIN VACUUM`).
    if first.eq_ignore_ascii_case("explain")
        && words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case("comment"))
    {
        let rest = trimmed[first.len()..].trim_start();
        return Some(
            parse_comment_on(rest).map(|stmt| {
                ast::Statement::Explain(Box::new(stmt), ast::ExplainOptions::default())
            }),
        );
    }
    None
}

/// Parse `COMMENT ON {TABLE | COLUMN} <object> IS {'text' | NULL}`.
///
/// `sqlparser` 0.51 tokenizes the `COMMENT` keyword but does not model the statement-level
/// grammar, so we drive a `sqlparser` [`Parser`] manually — reusing its tokenizer and
/// object-name handling (quoted identifiers, schema qualification) rather than splitting strings.
fn parse_comment_on(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::COMMENT).map_err(syntax)?;
    parser.expect_keyword(Keyword::ON).map_err(syntax)?;

    let target = if parser.parse_keyword(Keyword::TABLE) {
        let name = parser.parse_object_name(false).map_err(syntax)?;
        ast::CommentTarget::Table {
            table: object_name(&name)?,
        }
    } else if parser.parse_keyword(Keyword::COLUMN) {
        let name = parser.parse_object_name(false).map_err(syntax)?;
        let (table, column) = table_and_column(&name)?;
        ast::CommentTarget::Column { table, column }
    } else {
        return unsupported("COMMENT ON only supports TABLE and COLUMN targets");
    };

    parser.expect_keyword(Keyword::IS).map_err(syntax)?;
    let comment = if parser.parse_keyword(Keyword::NULL) {
        None // `IS NULL` clears any existing comment
    } else {
        match parser.next_token().token {
            Token::SingleQuotedString(s) => Some(s),
            other => {
                return Err(Error::Syntax(format!(
                    "expected a string literal or NULL after IS, found {other}"
                )));
            },
        }
    };

    // Reject trailing content so the surface stays honest and single-statement (matching `parse`).
    loop {
        match parser.next_token().token {
            Token::EOF => break,
            Token::SemiColon => {}, // a trailing `;` terminates the single statement
            other => return Err(Error::Syntax(format!("unexpected trailing token: {other}"))),
        }
    }
    Ok(ast::Statement::CommentOn(ast::CommentOn {
        target,
        comment,
    }))
}

/// Recognize a leading `COPY` so we can hand-parse it (see [`parse_copy_stmt`]).
fn recognize_copy(sql: &str) -> Option<Result<ast::Statement, Error>> {
    let trimmed = sql.trim_start();
    if trimmed
        .split_whitespace()
        .next()?
        .eq_ignore_ascii_case("copy")
    {
        return Some(parse_copy_stmt(trimmed));
    }
    None
}

/// Parse `COPY table [(cols)] {FROM STDIN | TO STDOUT} [WITH (opts)]`.
///
/// Driven through a `sqlparser` [`Parser`] (reusing its tokenizer + object-name handling) rather
/// than the statement grammar, which expects an inline data block for `FROM STDIN`. Only the
/// streaming STDIN/STDOUT text-format forms are accepted.
fn parse_copy_stmt(sql: &str) -> Result<ast::Statement, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;

    parser.expect_keyword(Keyword::COPY).map_err(syntax)?;
    let table = object_name(&parser.parse_object_name(false).map_err(syntax)?)?;

    let columns = if parser.consume_token(&Token::LParen) {
        let cols = parser
            .parse_comma_separated(Parser::parse_identifier)
            .map_err(syntax)?;
        parser.expect_token(&Token::RParen).map_err(syntax)?;
        cols.iter().map(fold_ident).collect()
    } else {
        Vec::new()
    };

    let direction = if parser.parse_keyword(Keyword::FROM) {
        ast::CopyDirection::From
    } else if parser.parse_keyword(Keyword::TO) {
        ast::CopyDirection::To
    } else {
        return unsupported("COPY requires FROM or TO");
    };

    // STDIN / STDOUT arrive as a bare word; match it against the direction.
    let stream = match parser.next_token().token {
        Token::Word(w) => w.value.to_ascii_uppercase(),
        // A quoted path is a file target — only the streaming forms are supported.
        Token::SingleQuotedString(_) => {
            return unsupported(
                "COPY only supports STDIN / STDOUT; file and program targets are not supported",
            );
        },
        other => {
            return Err(Error::Syntax(format!(
                "expected STDIN or STDOUT, found {other}"
            )));
        },
    };
    match (direction, stream.as_str()) {
        (ast::CopyDirection::From, "STDIN") | (ast::CopyDirection::To, "STDOUT") => {},
        (ast::CopyDirection::From, _) => return unsupported("COPY FROM only supports STDIN"),
        (ast::CopyDirection::To, _) => return unsupported("COPY TO only supports STDOUT"),
    }

    let format = parse_copy_options(&mut parser)?;

    // Reject trailing content so the surface stays single-statement (matching `parse`).
    loop {
        match parser.next_token().token {
            Token::EOF => break,
            Token::SemiColon => {},
            other => return Err(Error::Syntax(format!("unexpected trailing token: {other}"))),
        }
    }
    Ok(ast::Statement::Copy(ast::Copy {
        table,
        columns,
        direction,
        format,
    }))
}

/// Parse the optional `[WITH] ( FORMAT text, DELIMITER 'c', NULL 's', HEADER [bool] )` tail.
fn parse_copy_options(parser: &mut Parser) -> Result<ast::CopyFormat, Error> {
    use sqlparser::keywords::Keyword;
    use sqlparser::parser::ParserError;
    use sqlparser::tokenizer::Token;

    let syntax = |e: ParserError| Error::Syntax(e.to_string());
    let mut format = ast::CopyFormat::default();
    let _ = parser.parse_keyword(Keyword::WITH);
    if !parser.consume_token(&Token::LParen) {
        return Ok(format); // no options
    }
    loop {
        let key = match parser.next_token().token {
            Token::Word(w) => w.value.to_ascii_uppercase(),
            other => {
                return Err(Error::Syntax(format!(
                    "expected a COPY option, found {other}"
                )));
            },
        };
        match key.as_str() {
            "FORMAT" => {
                let name = match parser.next_token().token {
                    Token::Word(w) => w.value,
                    other => return Err(Error::Syntax(format!("expected a format name, {other}"))),
                };
                if !name.eq_ignore_ascii_case("text") {
                    return unsupported(
                        "only COPY WITH (FORMAT text) is supported (CSV/binary are follow-ups)",
                    );
                }
            },
            "DELIMITER" => format.delimiter = copy_single_char(parser)?,
            "NULL" => format.null = copy_string(parser)?,
            "HEADER" => format.header = copy_optional_bool(parser),
            other => {
                return Err(Error::Unsupported(format!(
                    "unsupported COPY option `{other}` (FORMAT text, DELIMITER, NULL, HEADER only)"
                )));
            },
        }
        if !parser.consume_token(&Token::Comma) {
            break;
        }
    }
    parser.expect_token(&Token::RParen).map_err(syntax)?;
    Ok(format)
}

/// Read a single-quoted string option value (e.g. `NULL '...'`).
fn copy_string(parser: &mut Parser) -> Result<String, Error> {
    use sqlparser::tokenizer::Token;
    match parser.next_token().token {
        Token::SingleQuotedString(s) => Ok(s),
        other => Err(Error::Syntax(format!(
            "expected a quoted string, found {other}"
        ))),
    }
}

/// Read a single-quoted, single-character option value (e.g. `DELIMITER ','`).
fn copy_single_char(parser: &mut Parser) -> Result<char, Error> {
    let s = copy_string(parser)?;
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(Error::Syntax(
            "DELIMITER must be a single character".to_owned(),
        )),
    }
}

/// Read an optional boolean after `HEADER`; a bare `HEADER` means `true`.
fn copy_optional_bool(parser: &mut Parser) -> bool {
    use sqlparser::tokenizer::Token;
    if let Token::Word(w) = parser.peek_token().token {
        let v = w.value.to_ascii_lowercase();
        if v == "true" || v == "on" {
            parser.next_token();
            return true;
        }
        if v == "false" || v == "off" {
            parser.next_token();
            return false;
        }
    }
    true
}

/// Split a `COMMENT ON COLUMN` object name into `(table, column)`. The column must be
/// table-qualified (`table.column`, optionally schema-qualified); a bare `column` has no table to
/// resolve against and is rejected.
fn table_and_column(name: &sql::ObjectName) -> Result<(String, String), Error> {
    match name.0.as_slice() {
        [table, column] => Ok((fold_part(table)?, fold_part(column)?)),
        // A schema-qualified `schema.table.column` must not silently collapse to `table.column`
        // (single namespace in Stage 4); a bare `column` has no table to resolve against.
        _ => unsupported("COMMENT ON COLUMN requires a table-qualified column name (table.column)"),
    }
}

/// Reject any SQL construct the Stage 4 parser does not model.
fn unsupported<T>(what: &str) -> Result<T, Error> {
    Err(Error::Unsupported(what.to_owned()))
}

/// Reject identifier tokens outside NusaDB's documented surface (follow-up).
///
/// Parsing uses `GenericDialect` (required for the grammar extensions NusaDB models — see the
/// module docs), but its tokenizer admits identifiers NusaDB does not: unquoted identifiers led by
/// `@`/`#` or containing non-ASCII letters, and delimited identifiers quoted with anything other
/// than `"` (e.g. backtick). This restores the tokenize-time rejection the documented surface
/// promises, by validating every word token against [`NusaDialect`]'s rules — the single source of
/// truth for the identifier surface. Keywords are ASCII and pass trivially; string/number/parameter
/// tokens are not `Word`s and are unaffected. Tokenising here is independent of the later parse
/// (one extra cheap tokenizer pass per statement; `parse` is not a per-row path).
fn reject_widened_lexicon(sql: &str) -> Result<(), Error> {
    let tokens = Tokenizer::new(&GenericDialect {}, sql)
        .tokenize()
        .map_err(|e| Error::Syntax(e.to_string()))?;
    for token in &tokens {
        let Token::Word(word) = token else { continue };
        match word.quote_style {
            None => {
                let mut chars = word.value.chars();
                let ok = chars
                    .next()
                    .is_some_and(|c| NusaDialect.is_identifier_start(c))
                    && chars.all(|c| NusaDialect.is_identifier_part(c));
                if !ok {
                    return unsupported(&format!(
                        "identifier `{}` outside NusaDB's surface \
                         (unquoted identifiers are ASCII letters/digits/`_`/`$`)",
                        word.value
                    ));
                }
            },
            // NusaDB documents only the standard double-quote delimiter for quoted identifiers.
            Some('"') => {},
            Some(quote) => {
                return unsupported(&format!(
                    "{quote}-quoted identifier `{}`; NusaDB quotes identifiers with double quotes",
                    word.value
                ));
            },
        }
    }
    Ok(())
}

/// Reject the non-standard wildcard decorations (`EXCEPT`/`EXCLUDE`/`REPLACE`/`RENAME`/`ILIKE`) that
/// `GenericDialect` parses but NusaDB does not model (follow-up). Previously they were
/// silently dropped — turning `SELECT * EXCEPT (a)` into a bare `SELECT *`, a silent semantic change.
fn reject_wildcard_options(opts: &sql::WildcardAdditionalOptions) -> Result<(), Error> {
    if opts.opt_ilike.is_some()
        || opts.opt_exclude.is_some()
        || opts.opt_except.is_some()
        || opts.opt_replace.is_some()
        || opts.opt_rename.is_some()
    {
        return unsupported("wildcard with EXCEPT / EXCLUDE / REPLACE / RENAME / ILIKE");
    }
    Ok(())
}

/// Normalize an identifier with NusaDB semantics: unquoted identifiers fold
/// to lowercase (so `Users` == `users` == `USERS`); quoted identifiers
/// (`"User"`) preserve their case verbatim.
fn fold_ident(ident: &sql::Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_lowercase()
    }
}

/// One component of a (possibly qualified) object name. sqlparser 0.62 models each part as
/// [`sql::ObjectNamePart`] — a plain identifier or a dialect-specific part-yielding function call
/// (dialect-specific); only plain identifiers are in the NusaDB surface, so a function part is
/// rejected loudly rather than folded.
fn part_ident(part: &sql::ObjectNamePart) -> Result<&sql::Ident, Error> {
    part.as_ident()
        .ok_or_else(|| Error::Unsupported(format!("object name part `{part}`")))
}

/// [`fold_ident`] over an [`sql::ObjectNamePart`] (see [`part_ident`]).
fn fold_part(part: &sql::ObjectNamePart) -> Result<String, Error> {
    part_ident(part).map(fold_ident)
}

/// Accept a `COLLATE <name>` clause only when it names the byte-order collation NusaDB already uses
/// (`C` / `POSIX`), as a no-op; otherwise reject it loudly (D-COLLATE).
///
/// NusaDB sorts text by byte value — deterministic, and all a single-node OLTP target needs — which
/// is exactly the SQL-standard `C` / `POSIX` collation. So `COLLATE "C"` / `"POSIX"` asks for what
/// NusaDB already does and changes nothing, while a locale collation (`"en_US"`, ICU, …) is refused
/// with an honest error rather than silently ignored: silently applying byte order to a query that
/// asked for locale order would be a wrong result with no signal. The final name component is used,
/// so a `pg_catalog."C"` qualifier is tolerated. Case-insensitive so `"c"` and `C` are also accepted.
fn require_byte_order_collation(name: &sql::ObjectName) -> Result<(), Error> {
    let ident = name
        .0
        .last()
        .map(part_ident)
        .transpose()?
        .map_or("", |i| i.value.as_str());
    if ident.eq_ignore_ascii_case("C") || ident.eq_ignore_ascii_case("POSIX") {
        Ok(())
    } else {
        unsupported(&format!(
            "COLLATE \"{ident}\" is not supported (NusaDB sorts text by byte value; only the \
             byte-order \"C\" / \"POSIX\" collation is accepted)"
        ))
    }
}

/// Fold a table-alias column list (`AS x(a, b)`). A typed alias column (`AS x(a INT)` — a
/// typed-alias form sqlparser 0.62 newly models) is out of surface, rejected rather than having its
/// type silently dropped.
fn fold_alias_columns(columns: &[sql::TableAliasColumnDef]) -> Result<Vec<String>, Error> {
    columns
        .iter()
        .map(|c| {
            if c.data_type.is_some() {
                return unsupported("table alias column with a declared type");
            }
            Ok(fold_ident(&c.name))
        })
        .collect()
}

/// Fold a column name that sqlparser 0.62 models as a full `ObjectName` (INSERT / MERGE INSERT
/// column lists). A column name is a single identifier — a qualified name there is out of surface.
fn column_ident_name(name: &sql::ObjectName) -> Result<String, Error> {
    match name.0.as_slice() {
        [part] => fold_part(part),
        _ => unsupported("qualified column name in a column list"),
    }
}

/// The 0.51-shaped `(LIMIT, OFFSET, LIMIT BY)` decomposition of a 0.62 `LimitClause`.
type SplitLimit = (Option<sql::Expr>, Option<sql::Offset>, Vec<sql::Expr>);

/// Split the sqlparser 0.62 `LimitClause` envelope into the `(LIMIT, OFFSET, LIMIT BY)` triple the
/// converters were written against. The comma-form `LIMIT <offset>, <limit>` spelling is out of
/// surface.
fn split_limit_clause(clause: Option<sql::LimitClause>) -> Result<SplitLimit, Error> {
    match clause {
        None => Ok((None, None, Vec::new())),
        Some(sql::LimitClause::LimitOffset {
            limit,
            offset,
            limit_by,
        }) => Ok((limit, offset, limit_by)),
        Some(sql::LimitClause::OffsetCommaLimit { .. }) => {
            unsupported("comma-form `LIMIT <offset>, <limit>`")
        },
    }
}

/// Whether an `ORDER BY` clause actually orders anything (an empty expression list does not;
/// `ORDER BY ALL` does — it is rejected downstream by `convert_order_by`).
const fn order_by_is_effective(order_by: &sql::OrderBy) -> bool {
    match &order_by.kind {
        sql::OrderByKind::Expressions(exprs) => !exprs.is_empty(),
        sql::OrderByKind::All(_) => true,
    }
}

/// `PREPARE name [(types)] AS <statement>`. The declared parameter types are dropped (the
/// count comes from the `$n` placeholders); the body is converted but not analyzed here. Only a
/// runnable query (`SELECT`/set-op/`INSERT`/`UPDATE`/`DELETE`) may be prepared.
fn convert_prepare(name: &sql::Ident, statement: sql::Statement) -> Result<ast::Statement, Error> {
    let inner = convert_statement(statement)?;
    if !is_prepareable(&inner) {
        return Err(Error::Unsupported(
            "PREPARE accepts only a SELECT, set operation, INSERT, UPDATE, or DELETE".to_owned(),
        ));
    }
    Ok(ast::Statement::Prepare {
        name: fold_ident(name),
        statement: Box::new(inner),
    })
}

/// Whether `stmt` is a statement that `PREPARE` may wrap (a runnable query, not transaction/session
/// control or another PREPARE/EXECUTE).
const fn is_prepareable(stmt: &ast::Statement) -> bool {
    matches!(
        stmt,
        ast::Statement::Select(_)
            | ast::Statement::SetOperation(_)
            | ast::Statement::Insert(_)
            | ast::Statement::Update(_)
            | ast::Statement::Delete(_)
    )
}

/// `EXECUTE name [(args)]`. `USING` is not supported. The argument expressions are kept as
/// expressions here and reduced to constant values by the analyzer.
fn convert_execute(
    name: &sql::ObjectName,
    parameters: Vec<sql::Expr>,
    using: &[sql::ExprWithAlias],
) -> Result<ast::Statement, Error> {
    if !using.is_empty() {
        return Err(Error::Unsupported(
            "EXECUTE ... USING is not supported".to_owned(),
        ));
    }
    // A prepared-statement name is a single identifier (0.62 models it as a full object name).
    let name = match name.0.as_slice() {
        [part] => fold_part(part)?,
        _ => return unsupported("a qualified EXECUTE statement name"),
    };
    let args = parameters
        .into_iter()
        .map(convert_expr)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ast::Statement::Execute { name, args })
}

/// `DEALLOCATE [PREPARE] {name | ALL}`. An unquoted `ALL` is the discard-everything form.
fn convert_deallocate(name: &sql::Ident) -> ast::Statement {
    if name.quote_style.is_none() && name.value.eq_ignore_ascii_case("all") {
        ast::Statement::Deallocate(ast::DeallocateTarget::All)
    } else {
        ast::Statement::Deallocate(ast::DeallocateTarget::Name(fold_ident(name)))
    }
}

// === Statement dispatch ===================================================

// A flat one-arm-per-statement-kind dispatch: its length scales with the
// breadth of the SQL surface, not with branching depth (each arm just
// destructures and delegates). The line cap is the wrong signal here.
#[allow(
    clippy::too_many_lines,
    reason = "flat per-statement dispatch; grows with the SQL surface, not with complexity"
)]
pub(super) fn convert_statement(stmt: sql::Statement) -> Result<ast::Statement, Error> {
    match stmt {
        sql::Statement::CreateTable(ct) => convert_create_table(ct),
        sql::Statement::CreateIndex(ci) => {
            convert_create_index(ci).map(ast::Statement::CreateIndex)
        },
        sql::Statement::Drop {
            object_type,
            if_exists,
            names,
            cascade,
            purge,
            temporary,
            ..
        } => {
            // PURGE / TEMPORARY request behavior NusaDB does not perform (no recycle bin, no
            // temporary tables), so honor them honestly by rejecting rather than silently running a
            // plain DROP. RESTRICT is the default no-cascade behavior, accepted as-is.
            // CASCADE drops a schema's member tables or, on a table, the FOREIGN KEYs on
            // other tables that reference it; other object kinds track no
            // dependencies, so CASCADE stays rejected there.
            if cascade
                && !matches!(
                    object_type,
                    sql::ObjectType::Schema | sql::ObjectType::Table
                )
            {
                return unsupported(
                    "DROP ... CASCADE is not supported for this object kind (NusaDB tracks no object \n                     dependencies; CASCADE applies to DROP SCHEMA and DROP TABLE)",
                );
            }
            if purge {
                return unsupported("DROP ... PURGE is not supported");
            }
            if temporary {
                return unsupported(
                    "DROP TEMPORARY is not supported (NusaDB has no temporary objects)",
                );
            }
            convert_drop(object_type, if_exists, cascade, &names)
        },
        sql::Statement::AlterTable(at) => {
            if at.on_cluster.is_some() {
                return unsupported("ALTER TABLE ... ON CLUSTER");
            }
            if at.table_type.is_some() {
                return unsupported("ALTER ICEBERG / DYNAMIC TABLE");
            }
            convert_alter_table(
                &at.name,
                at.if_exists,
                at.only,
                at.operations,
                at.location.is_some(),
            )
            .map(ast::Statement::AlterTable)
        },
        sql::Statement::CreateView(cv) => {
            if cv.or_alter {
                return unsupported("CREATE OR ALTER VIEW");
            }
            if cv.secure {
                return unsupported("CREATE SECURE VIEW");
            }
            if cv.copy_grants {
                return unsupported("CREATE VIEW ... COPY GRANTS");
            }
            if cv.params.is_some() {
                return unsupported("CREATE VIEW with dialect-specific parameters");
            }
            convert_create_view(
                &cv.name,
                cv.or_replace,
                cv.materialized,
                &cv.columns,
                *cv.query,
                cv.if_not_exists,
                cv.temporary,
                cv.with_no_schema_binding,
                &cv.cluster_by,
                cv.comment.as_ref(),
                cv.to.as_ref(),
                &cv.options,
            )
            .map(ast::Statement::CreateView)
        },
        sql::Statement::CreateSchema {
            schema_name,
            if_not_exists,
            with,
            options,
            default_collate_spec,
            clone,
        } => {
            if with.is_some() || options.is_some() {
                return unsupported("CREATE SCHEMA ... WITH / OPTIONS (...)");
            }
            if default_collate_spec.is_some() {
                return unsupported("CREATE SCHEMA ... DEFAULT COLLATE");
            }
            if clone.is_some() {
                return unsupported("CREATE SCHEMA ... CLONE");
            }
            convert_create_schema(schema_name, if_not_exists).map(ast::Statement::CreateSchema)
        },
        sql::Statement::CreateDatabase {
            db_name,
            if_not_exists,
            ..
        } => convert_create_database(&db_name, if_not_exists).map(ast::Statement::CreateDatabase),
        sql::Statement::CreateSequence {
            temporary,
            if_not_exists,
            name,
            data_type,
            sequence_options,
            owned_by,
        } => convert_create_sequence(
            temporary,
            if_not_exists,
            &name,
            data_type.as_ref(),
            sequence_options,
            owned_by.as_ref(),
        )
        .map(ast::Statement::CreateSequence),
        sql::Statement::Truncate(t) => {
            if t.if_exists {
                return unsupported("TRUNCATE ... IF EXISTS");
            }
            if t.on_cluster.is_some() {
                return unsupported("TRUNCATE ... ON CLUSTER");
            }
            // sqlparser 0.62 no longer models `ONLY` on TRUNCATE; `t.table` (the optional `TABLE`
            // keyword) is cosmetic.
            convert_truncate(
                &t.table_names,
                t.partitions.is_some(),
                false,
                t.identity.as_ref(),
                t.cascade.as_ref(),
            )
            .map(ast::Statement::Truncate)
        },
        sql::Statement::Insert(insert) => convert_insert(insert).map(ast::Statement::Insert),
        sql::Statement::Query(mut query) => {
            // A WITH prefix on a top-level `INSERT ... SELECT` pushes its CTEs
            // into the source select — the same scoping (the CTEs are visible to the source),
            // so the complete archive-then-remove pattern
            // `WITH m AS (DELETE ... RETURNING) INSERT INTO archive SELECT ... FROM m` runs
            // through the existing modifying-CTE machinery unchanged. UPDATE/DELETE with a
            // WITH prefix (CTE references in WHERE/USING) are a different wiring and stay
            // loudly rejected with a specific message.
            if query.with.is_some() {
                match query.body.as_ref() {
                    sql::SetExpr::Insert(_) => {
                        let with = query.with.take();
                        let sql::SetExpr::Insert(stmt) = *query.body else {
                            return unsupported("internal: INSERT body vanished");
                        };
                        let ast::Statement::Insert(mut insert) = convert_statement(stmt)? else {
                            return unsupported("WITH on a non-INSERT statement body");
                        };
                        let ast::InsertSource::Select(source) = &mut insert.source else {
                            return unsupported(
                                "WITH on INSERT requires a SELECT source (the CTEs are read by the source query)",
                            );
                        };
                        if !source.with.is_empty() {
                            return unsupported(
                                "combining a statement-level WITH and a source-level WITH on one INSERT",
                            );
                        }
                        source.with = query::convert_with(with)?;
                        return Ok(ast::Statement::Insert(insert));
                    },
                    sql::SetExpr::Update(_) | sql::SetExpr::Delete(_) => {
                        return unsupported(
                            "WITH on a top-level UPDATE/DELETE is not yet supported (WITH on INSERT ... SELECT is)",
                        );
                    },
                    _ => {},
                }
            }
            // `SELECT ... INTO t FROM ...` at the top level is the standard's CTAS spelling —
            // desugar to `CREATE TABLE t AS <select>` (same analyzer/executor path). Only the
            // plain form is modelled; TEMPORARY/UNLOGGED stay refused. The `into` field is
            // cleared before conversion so the deep per-SELECT reject (which still guards any
            // non-top position) does not fire.
            if let sql::SetExpr::Select(select) = query.body.as_mut()
                && let Some(into) = select.into.take()
            {
                if into.temporary || into.unlogged {
                    return unsupported("SELECT INTO TEMPORARY / UNLOGGED");
                }
                let name = object_name(&into.name)?;
                let ast::Statement::Select(body) = convert_query(*query)? else {
                    return unsupported("SELECT ... INTO over a non-SELECT query");
                };
                return Ok(ast::Statement::CreateTableAs(ast::CreateTableAs {
                    name,
                    query: Box::new(body),
                    if_not_exists: false,
                }));
            }
            convert_query(*query)
        },
        sql::Statement::Update(u) => {
            if !u.optimizer_hints.is_empty() {
                return unsupported("optimizer hints");
            }
            if u.output.is_some() {
                return unsupported("UPDATE ... OUTPUT");
            }
            if u.or.is_some() {
                return unsupported("UPDATE OR ROLLBACK/ABORT/... (conflict clause)");
            }
            if !u.order_by.is_empty() || u.limit.is_some() {
                return unsupported("UPDATE ... ORDER BY / LIMIT");
            }
            convert_update(&u.table, u.assignments, u.from, u.selection, u.returning)
                .map(ast::Statement::Update)
        },
        sql::Statement::Delete(delete) => convert_delete(delete).map(ast::Statement::Delete),
        sql::Statement::Merge(m) => {
            if !m.optimizer_hints.is_empty() {
                return unsupported("optimizer hints");
            }
            if m.output.is_some() {
                return unsupported("MERGE ... OUTPUT");
            }
            convert_merge(m.into, &m.table, &m.source, *m.on, m.clauses).map(ast::Statement::Merge)
        },
        sql::Statement::Explain {
            statement,
            analyze,
            verbose,
            format,
            ..
        } => {
            // FORMAT TEXT (default) and JSON are supported; other formats (e.g. GRAPHVIZ) are
            // rejected rather than silently ignored. (0.62 wraps the format in a keyword-vs-`=`
            // spelling kind — both spellings carry the same format.)
            let format = format.map(|kind| match kind {
                sql::AnalyzeFormatKind::Keyword(f) | sql::AnalyzeFormatKind::Assignment(f) => f,
            });
            let format = match format {
                None | Some(sql::AnalyzeFormat::TEXT) => ast::ExplainFormat::Text,
                Some(sql::AnalyzeFormat::JSON) => ast::ExplainFormat::Json,
                Some(other) => {
                    return Err(Error::Unsupported(format!(
                        "EXPLAIN (FORMAT {other}) is not supported; only TEXT and JSON"
                    )));
                },
            };
            let inner = convert_statement(*statement)?;
            Ok(ast::Statement::Explain(
                Box::new(inner),
                ast::ExplainOptions {
                    analyze,
                    verbose,
                    format,
                },
            ))
        },
        sql::Statement::Analyze(a) => {
            if a.compute_statistics {
                return unsupported("ANALYZE ... COMPUTE STATISTICS");
            }
            let Some(ref table_name) = a.table_name else {
                return unsupported("ANALYZE without a table name");
            };
            convert_analyze(
                table_name,
                a.partitions.as_deref(),
                a.columns,
                a.cache_metadata,
                a.noscan,
            )
        },
        sql::Statement::Prepare {
            name, statement, ..
        } => convert_prepare(&name, *statement),
        sql::Statement::Execute {
            name,
            parameters,
            has_parentheses: _,
            immediate,
            into,
            using,
            output,
            default,
        } => {
            if immediate {
                return unsupported("EXECUTE IMMEDIATE");
            }
            if !into.is_empty() {
                return unsupported("EXECUTE ... INTO");
            }
            if output {
                return unsupported("EXECUTE ... OUTPUT");
            }
            if default {
                return unsupported("EXECUTE ... DEFAULT");
            }
            let Some(name) = name else {
                return unsupported("EXECUTE without a statement name");
            };
            convert_execute(&name, parameters, &using)
        },
        sql::Statement::Deallocate { name, .. } => Ok(convert_deallocate(&name)),
        sql::Statement::StartTransaction { modes, .. } => {
            convert_transaction_modes(modes).map(ast::Statement::BeginTransaction)
        },
        sql::Statement::Commit { .. } => Ok(ast::Statement::Commit),
        sql::Statement::Rollback {
            savepoint: None, ..
        } => Ok(ast::Statement::Rollback),
        sql::Statement::Rollback {
            savepoint: Some(name),
            ..
        } => Ok(ast::Statement::RollbackToSavepoint(fold_ident(&name))),
        sql::Statement::Savepoint { name } => Ok(ast::Statement::Savepoint(fold_ident(&name))),
        sql::Statement::ReleaseSavepoint { name } => {
            Ok(ast::Statement::ReleaseSavepoint(fold_ident(&name)))
        },
        sql::Statement::Set(set) => convert_set_statement(set),
        sql::Statement::ShowVariable { variable } => convert_show_variable(&variable),
        // `SHOW DATABASES` is out of surface (dropped 2026-07-06: `SHOW` reads only a configuration
        // parameter; the cluster listing lives in the `nusadb_databases` catalog). sqlparser 0.62
        // parses it as a first-class statement — route it to the same session-side
        // unknown-parameter rejection (42704) it got in 0.51 via `ShowVariable(["DATABASES"])`.
        sql::Statement::ShowDatabases { .. } => Ok(ast::Statement::Show("databases".to_owned())),
        // `SHOW TABLES` / `SHOW COLUMNS FROM t` — catalog introspection. The optional
        // EXTENDED/FULL/db/filter decorations are not modeled; reject them rather than ignore.
        sql::Statement::ShowTables {
            terse: false,
            history: false,
            extended: false,
            full: false,
            external: false,
            show_options,
        } if show_options_are_empty(&show_options) => Ok(ast::Statement::ShowTables),
        sql::Statement::ShowTables { .. } => {
            unsupported("SHOW TABLES with EXTENDED/FULL/FROM/LIKE is not supported")
        },
        // `SHOW COLUMNS FROM t` — the table lives in the 0.62 `show_options.show_in` envelope.
        sql::Statement::ShowColumns {
            extended: false,
            full: false,
            show_options,
        } => {
            let table_name = show_columns_table(&show_options)?;
            Ok(ast::Statement::ShowColumns(object_name(&table_name)?))
        },
        // `DESCRIBE t` / `DESC t` / `EXPLAIN t` — aliases that list a table's columns.
        sql::Statement::ExplainTable {
            table_name,
            hive_format: None,
            ..
        } => Ok(ast::Statement::ShowColumns(object_name(&table_name)?)),
        sql::Statement::ShowColumns { .. } => {
            unsupported("SHOW COLUMNS with EXTENDED/FULL/LIKE is not supported")
        },
        sql::Statement::ExplainTable { .. } => {
            unsupported("DESCRIBE with a Hive format option is not supported")
        },
        // `CALL name(args)` reached through the generic parser (e.g. inside a procedure body);
        // the top-level form is handled earlier by `recognize_call`.
        sql::Statement::Call(func) => convert_call_statement(func),
        _ => unsupported(
            "only CREATE/DROP TABLE, INSERT, SELECT, UPDATE, DELETE, EXPLAIN, transaction \
             control (BEGIN/COMMIT/ROLLBACK/SAVEPOINT/SET/SHOW) are supported",
        ),
    }
}

// === Shared helpers =======================================================

/// Convert sqlparser transaction modes into [`ast::TransactionSettings`].
/// A repeated isolation level or access mode is rejected.
fn convert_transaction_modes(
    modes: Vec<sql::TransactionMode>,
) -> Result<ast::TransactionSettings, Error> {
    let mut settings = ast::TransactionSettings::default();
    for mode in modes {
        match mode {
            sql::TransactionMode::IsolationLevel(level) => {
                if settings.isolation.is_some() {
                    return unsupported("repeated ISOLATION LEVEL in transaction characteristics");
                }
                settings.isolation = Some(match level {
                    sql::TransactionIsolationLevel::ReadUncommitted => {
                        ast::IsolationLevel::ReadUncommitted
                    },
                    sql::TransactionIsolationLevel::ReadCommitted => {
                        ast::IsolationLevel::ReadCommitted
                    },
                    sql::TransactionIsolationLevel::RepeatableRead => {
                        ast::IsolationLevel::RepeatableRead
                    },
                    sql::TransactionIsolationLevel::Serializable => {
                        ast::IsolationLevel::Serializable
                    },
                    // A dialect-specific level (0.62): not one of the four standard levels.
                    sql::TransactionIsolationLevel::Snapshot => {
                        return unsupported("ISOLATION LEVEL SNAPSHOT");
                    },
                });
            },
            sql::TransactionMode::AccessMode(mode) => {
                if settings.access_mode.is_some() {
                    return unsupported("repeated READ ONLY/WRITE in transaction characteristics");
                }
                settings.access_mode = Some(match mode {
                    sql::TransactionAccessMode::ReadOnly => ast::AccessMode::ReadOnly,
                    sql::TransactionAccessMode::ReadWrite => ast::AccessMode::ReadWrite,
                });
            },
        }
    }
    Ok(settings)
}

/// Convert `SET name = value` into [`ast::Statement::SetVariable`]. `RESET name` arrives
/// here too — sqlparser models it as a `SetVariable` whose value is the `DEFAULT` keyword.
/// Dispatch a `SET ...` statement (0.62 groups every SET form under one `Set` enum). Only the
/// plain single-assignment (`SET name = value` / `SET name TO value`) and `SET TRANSACTION` forms
/// are in surface — everything else (SET ROLE / TIME ZONE / NAMES / session-param) stays rejected,
/// as it was in 0.51.
fn convert_set_statement(set: sql::Set) -> Result<ast::Statement, Error> {
    match set {
        sql::Set::SetTransaction {
            modes,
            snapshot,
            session: _,
        } => {
            if snapshot.is_some() {
                return unsupported("SET TRANSACTION SNAPSHOT");
            }
            convert_transaction_modes(modes).map(ast::Statement::SetTransaction)
        },
        sql::Set::SingleAssignment {
            scope,
            hivevar,
            variable,
            values,
        } => {
            if scope.is_some() || hivevar {
                return unsupported("SET LOCAL / SET HIVEVAR");
            }
            convert_set_variable(&variable, &values)
        },
        sql::Set::ParenthesizedAssignments { .. } | sql::Set::MultipleAssignments { .. } => {
            unsupported("SET with multiple variables")
        },
        other => unsupported(&format!("SET statement `{other}`")),
    }
}

fn convert_set_variable(
    name: &sql::ObjectName,
    value: &[sql::Expr],
) -> Result<ast::Statement, Error> {
    let name = object_name(name)?;
    // `search_path` is a list: `SET search_path TO a, public` carries several values — render
    // each and join with commas so the session stores the ordered list. Other GUCs take one value.
    if name == "search_path" && value.len() > 1 {
        let parts = value
            .iter()
            .map(set_value_text)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(ast::Statement::SetVariable(ast::SetVariable {
            name,
            value: Some(parts.join(", ")),
        }));
    }
    let [value] = value else {
        return unsupported("SET requires exactly one value");
    };
    // `RESET name` parses as `SET name = DEFAULT`; model it as a reset (value = None).
    let rendered = match value {
        sql::Expr::Identifier(id) if id.value.eq_ignore_ascii_case("default") => None,
        other => Some(set_value_text(other)?),
    };
    Ok(ast::Statement::SetVariable(ast::SetVariable {
        name,
        value: rendered,
    }))
}

/// Render a `SET` right-hand-side value as text. Only literals and bare identifiers are accepted.
fn set_value_text(expr: &sql::Expr) -> Result<String, Error> {
    if let sql::Expr::Value(v) = expr {
        return match &v.value {
            sql::Value::SingleQuotedString(s)
            | sql::Value::EscapedStringLiteral(s)
            | sql::Value::UnicodeStringLiteral(s) => Ok(s.clone()),
            sql::Value::Number(n, _) => Ok(n.clone()),
            sql::Value::Boolean(b) => Ok(b.to_string()),
            _ => unsupported("SET value must be a literal or identifier"),
        };
    }
    match expr {
        sql::Expr::Identifier(id) => Ok(id.value.clone()),
        _ => unsupported("SET value must be a literal or identifier"),
    }
}

/// Whether a 0.62 `SHOW ...` options envelope carries nothing (no IN/FROM clause, prefix filter,
/// LIMIT, or LIKE/WHERE filter) — the only `SHOW TABLES` shape in surface.
const fn show_options_are_empty(opts: &sql::ShowStatementOptions) -> bool {
    opts.show_in.is_none()
        && opts.starts_with.is_none()
        && opts.limit.is_none()
        && opts.limit_from.is_none()
        && opts.filter_position.is_none()
}

/// Extract the table name of a `SHOW COLUMNS FROM t` from the 0.62 options envelope. Any other
/// decoration (LIKE/WHERE filter, LIMIT, prefix) is rejected rather than ignored.
fn show_columns_table(opts: &sql::ShowStatementOptions) -> Result<sql::ObjectName, Error> {
    if opts.starts_with.is_some()
        || opts.limit.is_some()
        || opts.limit_from.is_some()
        || opts.filter_position.is_some()
    {
        return unsupported("SHOW COLUMNS with EXTENDED/FULL/LIKE is not supported");
    }
    let Some(show_in) = opts.show_in.as_ref() else {
        return unsupported("SHOW COLUMNS without a table (use SHOW COLUMNS FROM <table>)");
    };
    if show_in.parent_type.is_some() {
        return unsupported("SHOW COLUMNS FROM with a parent-object qualifier");
    }
    show_in.parent_name.clone().ok_or_else(|| {
        Error::Unsupported(
            "SHOW COLUMNS without a table (use SHOW COLUMNS FROM <table>)".to_owned(),
        )
    })
}

/// Convert `SHOW name` into [`ast::Statement::Show`].
fn convert_show_variable(variable: &[sql::Ident]) -> Result<ast::Statement, Error> {
    let [name] = variable else {
        return unsupported("SHOW with a multi-word or empty variable name");
    };
    Ok(ast::Statement::Show(fold_ident(name)))
}

/// Extract a bare object name.
///
/// A schema-qualified name (`schema.table`) is **rejected** rather than silently collapsed to its
/// last component: NusaDB has a single namespace in Stage 4, so quietly resolving `app.users`
/// and `users` — or `a.t` and `b.t` — to the same object would be a silent-wrong answer. (Consistent
/// with the 3-part `CompoundIdentifier` rejection in `convert_expr`.)
///
/// The exception is the `information_schema` schema: `information_schema.tables` is accepted and
/// joined with a `.`.
fn object_name(name: &sql::ObjectName) -> Result<String, Error> {
    match name.0.as_slice() {
        [part] => fold_part(part),
        [schema_part, name_part] => {
            let schema = fold_part(schema_part)?;
            let ident = fold_part(name_part)?;
            if schema == "information_schema" {
                Ok(format!("information_schema.{ident}"))
            } else if schema == PUBLIC_SCHEMA {
                // `public.<name>` resolves to the bare name: NusaDB has a single table
                // namespace, which is the default `public` schema — so `public.t` and `t` denote
                // the same table. Only `public` is recognised; any other schema qualifier is
                // rejected (no silent collapse).
                Ok(ident)
            } else {
                unsupported(
                    "schema-qualified object name (NusaDB has a single namespace; only the \
                     default `public` schema is recognised — use a bare name or `public.<name>`)",
                )
            }
        },
        [] => unsupported("empty object name"),
        _ => unsupported(
            "schema-qualified object name (NusaDB has a single namespace; only the default \
             `public` schema is recognised — use a bare name or `public.<name>`)",
        ),
    }
}

/// Resolve a **table-reference** object name into `(schema, name)`. An unqualified name yields
/// `None` — the analyzer resolves it through the session search path (current schema, then `public`).
/// An explicit qualifier yields `Some(schema)`: `public.<x>` selects the default namespace, any other
/// qualifier selects that schema. The synthetic `information_schema.<x>` catalog keeps its flat
/// encoding (`None` schema, `name = "information_schema.<x>"`) so the catalog special-case downstream
/// is untouched. Names with three or more parts (`db.schema.table`) are rejected.
fn table_ref_name(name: &sql::ObjectName) -> Result<(Option<String>, String), Error> {
    match name.0.as_slice() {
        [part] => Ok((None, fold_part(part)?)),
        [schema_part, name_part] => {
            let schema = fold_part(schema_part)?;
            let ident = fold_part(name_part)?;
            if schema == "information_schema" {
                Ok((None, format!("information_schema.{ident}")))
            } else {
                Ok((Some(schema), ident))
            }
        },
        [] => unsupported("empty object name"),
        _ => unsupported("schema-qualified object name with more than two parts (db.schema.table)"),
    }
}

/// The default (and only) schema name. NusaDB has a single flat table namespace, conventionally the
/// `public` schema, so a `public.` qualifier denotes that same namespace.
const PUBLIC_SCHEMA: &str = "public";

#[cfg(test)]
mod tests;
