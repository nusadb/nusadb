//! Exclusion constraints in their stored form.
//!
//! sqlparser has no grammar for `EXCLUDE`, so the pre-parse rewrite (`rewrite_exclude_constraint`)
//! turns a table constraint `EXCLUDE USING gist (k WITH =, tstzrange(a, b) WITH &&) WHERE (p)` into
//! a call of this parse's own marker inside a `CHECK` ([`rewrite_marker`]); conversion turns that,
//! and only that, back into an exclusion constraint ([`classify_check`]).
//!
//! The stored form is the call `__nusadb_exclude('gist', '', k, '=', tstzrange(a, b), '&&', (p))`.
//! The constraint catalog persists it behind [`EXCLUSION_TAG`], a NUL-led prefix. The canonical
//! text of a `CHECK` predicate never starts with a NUL byte (a NUL can only appear quoted, inside a
//! literal or identifier), so catalog text without the tag is always an ordinary `CHECK`, whatever
//! it says, and a predicate written by any build can never read as an exclusion constraint. Behind
//! the tag the stored form is an ordinary expression, so the column-reference tools that keep
//! `CHECK` predicates in step with `RENAME COLUMN` / `DROP COLUMN` work on it unchanged, and the
//! executor re-parses it here to enforce the constraint.
//!
//! Argument layout: the access-method name, the name of the index the constraint created for
//! itself (`''` when it owns none), then one `(expression, 'operator')` pair per element, then,
//! when present, the parenthesized `WHERE` predicate (so an odd count after the first two means a
//! predicate is present). Recording the owned index is what lets `DROP CONSTRAINT` drop only an
//! index the constraint made, never a user's index of the same name.

use sqlparser::ast as sql;
use sqlparser::parser::Parser;

use super::{NusaParserDialect, convert_expr};
use crate::ast;
use crate::error::Error;

/// The marker function name of an exclusion constraint's stored form.
pub(super) const EXCLUSION_MARKER: &str = "__nusadb_exclude";

thread_local! {
    /// The marker name the `EXCLUDE` rewrite uses within the current parse: the stored-form marker
    /// plus a random suffix minted for that parse ([`RewriteScope`]). Only a `CHECK` around a call
    /// of exactly this name becomes an exclusion constraint, and the name exists only in the token
    /// stream the rewrite builds, so no SQL text can produce one: a user-written call of the
    /// stored-form marker (or a guess at another parse's suffix) is refused or stays an ordinary
    /// call.
    static REWRITE_MARKER: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };

    /// Whether the current parse is a statement from a client (refuse a new `CHECK` calling the
    /// stored-form marker) rather than a routine, `DO` or trigger body (where such a `CHECK` stays
    /// the ordinary predicate an older build stored it as). See [`RewriteScope`].
    static TOP_LEVEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The span of one parse that may contain `EXCLUDE` rewrites: entering mints no marker yet (see
/// [`rewrite_marker`]); leaving restores the enclosing parse's marker, so a nested parse cannot
/// leak or reuse one.
pub(super) struct RewriteScope {
    /// The enclosing parse's marker.
    saved: Option<String>,
    /// The enclosing parse's [`TOP_LEVEL`].
    saved_top_level: bool,
}

impl RewriteScope {
    /// Begin a parse: `top_level` for a statement from a client, `false` for a routine, `DO` or
    /// trigger body.
    pub(super) fn enter(top_level: bool) -> Self {
        Self {
            saved: REWRITE_MARKER.with(|m| m.borrow_mut().take()),
            saved_top_level: TOP_LEVEL.with(|t| t.replace(top_level)),
        }
    }
}

impl Drop for RewriteScope {
    fn drop(&mut self) {
        let saved = self.saved.take();
        REWRITE_MARKER.with(|m| *m.borrow_mut() = saved);
        TOP_LEVEL.with(|t| t.set(self.saved_top_level));
    }
}

/// The current parse's rewrite marker name, minted on first use.
pub(super) fn rewrite_marker() -> String {
    REWRITE_MARKER.with(|m| {
        m.borrow_mut()
            .get_or_insert_with(|| {
                use std::hash::{BuildHasher, Hasher};
                let nonce = std::collections::hash_map::RandomState::new()
                    .build_hasher()
                    .finish();
                format!("{EXCLUSION_MARKER}_rw_{nonce:016x}")
            })
            .clone()
    })
}

/// The marker name of the current parse's rewrites, if it made any.
fn current_rewrite_marker() -> Option<String> {
    REWRITE_MARKER.with(|m| m.borrow().clone())
}

/// What a table `CHECK` expression is, as conversion sees it: an exclusion constraint the
/// `EXCLUDE` rewrite of this very parse produced, or an ordinary predicate. Nothing else can become
/// an exclusion: only the rewrite's per-parse marker does, and only the executor writes the tagged
/// form. A new `CHECK` calling the stored-form marker in a statement from a client is refused
/// (`42939`) for clarity; inside a routine, `DO` or trigger body it stays an ordinary `CHECK`, as
/// older builds stored it, so their routines keep running.
pub(super) fn classify_check(expr: &sql::Expr) -> Result<Option<ast::ExclusionDecl>, Error> {
    if let Some(name) = top_level_call_name(expr) {
        refuse_marker_call(name)?;
        if let Some(marker) = current_rewrite_marker()
            && name == marker
        {
            let mut decl = from_sql_expr_named(expr, &marker)?
                .ok_or_else(|| Error::Syntax("malformed EXCLUDE constraint".to_owned()))?;
            // Persist under the stored-form marker, never the per-parse name.
            decl.spec_sql = decl.stored_form(None);
            return Ok(Some(decl));
        }
    }
    Ok(None)
}

/// Refuse a column-level `CHECK` that calls the stored-form marker (`42939`), in a statement from
/// a client (see [`classify_check`]).
pub(super) fn refuse_marker_check(expr: &sql::Expr) -> Result<(), Error> {
    top_level_call_name(expr).map_or(Ok(()), refuse_marker_call)
}

/// `42939` for a `CHECK` calling `name` when `name` is the stored-form marker and the parse is a
/// statement from a client.
fn refuse_marker_call(name: &str) -> Result<(), Error> {
    if super::is_reserved_name(name) && TOP_LEVEL.with(std::cell::Cell::get) {
        return Err(super::reserved_name_error(name));
    }
    Ok(())
}

/// The last name part of a top-level function call (`f(...)`, `s.f(...)`).
fn top_level_call_name(expr: &sql::Expr) -> Option<&str> {
    let sql::Expr::Function(func) = expr else {
        return None;
    };
    match func.name.0.last()? {
        sql::ObjectNamePart::Identifier(ident) => Some(ident.value.as_str()),
        sql::ObjectNamePart::Function(_) => None,
    }
}

/// The tag that opens an exclusion constraint's catalog text: a NUL byte, which the canonical text
/// of a `CHECK` predicate (written by this build or an older one) never starts with.
pub(super) const EXCLUSION_TAG: &str = "\0exclude:";

/// The catalog text of an exclusion constraint whose stored form is `spec`.
pub(super) fn tag(spec: &str) -> String {
    format!("{EXCLUSION_TAG}{spec}")
}

/// The stored form inside tagged catalog text, or `None` for text without the tag (an ordinary
/// `CHECK` predicate, whatever it says).
pub(super) fn untag(stored: &[u8]) -> Option<&str> {
    std::str::from_utf8(stored.strip_prefix(EXCLUSION_TAG.as_bytes())?).ok()
}

/// Whether a constraint's catalog text is an exclusion constraint: it carries the tag and the rest
/// is exactly one valid stored form. Text without the tag is always an ordinary `CHECK`.
pub(super) fn is_exclusion_spec(stored: &[u8]) -> bool {
    untag(stored).is_some_and(|spec| parse_spec(spec).is_ok())
}

/// Parse a stored exclusion constraint (see the module docs) back into its declaration. The text
/// must be exactly one marker call; anything else is a syntax error, never an internal one.
pub(super) fn parse_spec(stored: &str) -> Result<ast::ExclusionDecl, Error> {
    use sqlparser::tokenizer::Token;
    let syntax = |e: sqlparser::parser::ParserError| Error::Syntax(e.to_string());
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(stored).map_err(syntax)?;
    let expr = parser.parse_expr().map_err(syntax)?;
    let malformed = || {
        Error::Syntax(format!(
            "not an exclusion constraint's stored form: {stored}"
        ))
    };
    if !matches!(parser.peek_token().token, Token::EOF) {
        return Err(malformed());
    }
    from_sql_expr_named(&expr, EXCLUSION_MARKER)?.ok_or_else(malformed)
}

/// If `expr` is a call of the marker named `marker`, its declaration; `None` for any other
/// expression. The declaration must obey the rules the `EXCLUDE` grammar enforces (a `btree` or
/// `gist` method, at least one element, `=` / `&&` operators), so a stored form can never carry
/// anything the grammar refuses, whichever path wrote it.
fn from_sql_expr_named(
    expr: &sql::Expr,
    marker: &str,
) -> Result<Option<ast::ExclusionDecl>, Error> {
    let sql::Expr::Function(func) = expr else {
        return Ok(None);
    };
    let [sql::ObjectNamePart::Identifier(name)] = func.name.0.as_slice() else {
        return Ok(None);
    };
    if name.value != marker {
        return Ok(None);
    }
    let malformed = || Error::Syntax("malformed EXCLUDE constraint".to_owned());
    let sql::FunctionArguments::List(list) = &func.args else {
        return Err(malformed());
    };
    let mut args = Vec::with_capacity(list.args.len());
    for arg in &list.args {
        match arg {
            sql::FunctionArg::Unnamed(sql::FunctionArgExpr::Expr(e)) => args.push(e),
            _ => return Err(malformed()),
        }
    }
    let [method, index, rest @ ..] = args.as_slice() else {
        return Err(malformed());
    };
    let method = string_literal(method).ok_or_else(malformed)?.to_lowercase();
    if method != "gist" && method != "btree" {
        return Err(Error::Unsupported(format!(
            "EXCLUDE USING {method} is not supported (the access method must be btree or gist)"
        )));
    }
    let index = string_literal(index).ok_or_else(malformed)?;
    let index = (!index.is_empty()).then(|| index.to_owned());
    let (pairs, predicate) = if rest.len() % 2 == 1 {
        let (last, pairs) = rest.split_last().ok_or_else(malformed)?;
        (pairs, Some(&**last))
    } else {
        (rest, None)
    };
    if pairs.is_empty() {
        return Err(malformed());
    }
    let mut elements = Vec::with_capacity(pairs.len() / 2);
    for pair in pairs.chunks(2) {
        let [element, op] = pair else {
            return Err(malformed());
        };
        let op = match string_literal(op).ok_or_else(malformed)? {
            "=" => ast::ExclusionOp::Equal,
            "&&" => ast::ExclusionOp::Overlap,
            other => {
                return Err(Error::Unsupported(format!(
                    "EXCLUDE operator {other} is not supported (only = and && are)"
                )));
            },
        };
        elements.push(ast::ExclusionElement {
            expr: convert_expr((*element).clone())?,
            sql: element.to_string(),
            op,
        });
    }
    let (predicate, predicate_sql) = match predicate {
        Some(p) => {
            let inner = match p {
                sql::Expr::Nested(inner) => inner.as_ref(),
                other => other,
            };
            (Some(convert_expr(inner.clone())?), Some(inner.to_string()))
        },
        None => (None, None),
    };
    Ok(Some(ast::ExclusionDecl {
        method,
        index,
        elements,
        predicate,
        predicate_sql,
        spec_sql: expr.to_string(),
    }))
}

/// The text of a single-quoted string literal argument.
fn string_literal(expr: &sql::Expr) -> Option<&str> {
    match expr {
        sql::Expr::Value(sql::ValueWithSpan {
            value: sql::Value::SingleQuotedString(s),
            ..
        }) => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_form_round_trips() {
        let stored = "__nusadb_exclude('gist', '', k, '=', tstzrange(valid_from, valid_to), '&&')";
        assert!(is_exclusion_spec(tag(stored).as_bytes()));
        // Untagged, the same text is an ordinary CHECK.
        assert!(!is_exclusion_spec(stored.as_bytes()));
        let decl = parse_spec(stored).unwrap();
        assert_eq!(decl.method, "gist");
        assert_eq!(decl.elements.len(), 2);
        assert_eq!(decl.elements[0].op, ast::ExclusionOp::Equal);
        assert_eq!(decl.elements[1].op, ast::ExclusionOp::Overlap);
        assert!(decl.predicate.is_none());
        assert_eq!(
            decl.definition(),
            "EXCLUDE USING gist (k WITH =, tstzrange(valid_from, valid_to) WITH &&)"
        );
        assert_eq!(decl.spec_sql, stored);
        assert_eq!(decl.index, None);
        assert_eq!(decl.stored_form(None), stored);
        let owned = decl.stored_form(Some("h_x"));
        assert_eq!(parse_spec(&owned).unwrap().index.as_deref(), Some("h_x"));
    }

    #[test]
    fn predicate_is_the_trailing_argument() {
        let decl = parse_spec("__nusadb_exclude('btree', '', r, '&&', (active = true))").unwrap();
        assert_eq!(decl.elements.len(), 1);
        assert_eq!(decl.predicate_sql.as_deref(), Some("active = true"));
        assert_eq!(
            decl.definition(),
            "EXCLUDE USING btree (r WITH &&) WHERE (active = true)"
        );
    }

    #[test]
    fn a_check_predicate_is_not_an_exclusion() {
        assert!(!is_exclusion_spec(b"a > 0"));
        assert!(!is_exclusion_spec(b"__nusadb_excludex(a)"));
        // Starts like the marker but is not exactly one valid marker call: an ordinary CHECK.
        assert!(!is_exclusion_spec(b"__nusadb_exclude(a) = 1"));
        assert!(!is_exclusion_spec(b"__nusadb_exclude(a)"));
        assert!(!is_exclusion_spec(
            b"__nusadb_exclude('gist', '', r, '&&') AND true"
        ));
        assert!(parse_spec("__nusadb_exclude(a) = 1").is_err());
        assert!(parse_spec("lower(a)").is_err());
    }

    #[test]
    fn a_stored_form_obeys_the_grammar_rules() {
        // Another access method, an unknown operator, or no element: never an exclusion.
        assert!(parse_spec("__nusadb_exclude('hash', '', r, '&&')").is_err());
        assert!(parse_spec("__nusadb_exclude('gist', '', r, '<>')").is_err());
        assert!(parse_spec("__nusadb_exclude('gist', '')").is_err());
        assert!(!is_exclusion_spec(
            tag("__nusadb_exclude('hash', '', r, '&&')").as_bytes()
        ));
        assert!(parse_spec("__nusadb_exclude('BTREE', '', r, '&&')").is_ok());
    }

    #[test]
    fn only_this_parses_rewrite_makes_an_exclusion() {
        use crate::ast;
        let constraint = |sql: &str| -> ast::TableConstraint {
            let ast::Statement::CreateTable(ct) = crate::parser::parse(sql).unwrap() else {
                panic!("expected CreateTable");
            };
            ct.constraints.into_iter().next().unwrap()
        };
        // The rewrite's own marker becomes an exclusion constraint stored under the stored-form
        // name, never the per-parse one.
        let ast::TableConstraint::Exclusion { decl, .. } =
            constraint("CREATE TABLE t (r int4range, EXCLUDE USING gist (r WITH &&))")
        else {
            panic!("expected an exclusion");
        };
        assert!(decl.spec_sql.starts_with("__nusadb_exclude('gist', ''"));
        assert!(is_exclusion_spec(tag(&decl.spec_sql).as_bytes()));
        // A guess at a rewrite marker is an ordinary call, so an ordinary CHECK.
        assert!(matches!(
            constraint(
                "CREATE TABLE t (r int4range, CHECK (__nusadb_exclude_rw_0123456789abcdef('gist', \
                 '', r, '&&')))"
            ),
            ast::TableConstraint::Check { .. }
        ));
        // Outside any parse no rewrite marker exists.
        assert!(current_rewrite_marker().is_none());
    }
}
