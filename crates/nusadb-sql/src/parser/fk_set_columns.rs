//! Surface support for the column list of a foreign key's `ON DELETE SET NULL (cols)` and
//! `ON DELETE SET DEFAULT (cols)` actions, which the generic parser does not model.
//!
//! The list names the referencing columns the action sets; the key's other columns keep their
//! values (a tenant column of a composite key, say). sqlparser has no grammar for it, so before
//! parsing [`rewrite_set_columns`] moves it into the key's referenced column list, behind a marker:
//! `REFERENCES p (a, b) ON DELETE SET NULL (b)` becomes `REFERENCES p (a, b, "<marker>", b) ON DELETE
//! SET NULL`, and `REFERENCES p ON DELETE SET NULL (b)` becomes `REFERENCES p ("<marker>", b) ON
//! DELETE SET NULL`. The foreign key conversion splits the list back at the marker
//! ([`split_referred_columns`]). The marker is a quoted identifier starting with a NUL, which no
//! real column name does, and a user-written one is refused ([`refuse_forged_marker`]).
//!
//! A column list belongs to `ON DELETE` only: on `ON UPDATE` it is refused with `0A000`, since an
//! update that sets only part of a changed key would leave the other part pointing at a key that
//! no longer exists.

use sqlparser::ast as sql;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, TokenWithSpan};

use super::{fold_ident, matching_paren, word_ci};
use crate::error::Error;

/// The marker that separates the referenced columns from the `ON DELETE SET` columns.
const SET_COLUMNS_MARKER: &str = "\0fk_set_columns\0";

/// The next non-whitespace token index after `from`.
fn nonws(tokens: &[TokenWithSpan], from: usize) -> Option<usize> {
    (from + 1..tokens.len())
        .find(|&j| !matches!(tokens.get(j).map(|t| &t.token), Some(Token::Whitespace(_))))
}

/// Whether the token at `at` is the unquoted keyword `keyword`.
fn is_keyword(tokens: &[TokenWithSpan], at: usize, keyword: Keyword) -> bool {
    matches!(tokens.get(at).map(|t| &t.token), Some(Token::Word(w)) if w.quote_style.is_none() && w.keyword == keyword)
}

/// Refuse a quoted identifier that starts like the marker (`42602`), so a user-written name can
/// never pass for one.
pub(super) fn refuse_forged_marker(tokens: &[TokenWithSpan]) -> Result<(), Error> {
    let forged = tokens.iter().any(|t| {
        matches!(&t.token, Token::Word(w) if w.quote_style.is_some() && w.value.starts_with(SET_COLUMNS_MARKER))
    });
    if forged {
        return Err(Error::Coded {
            message: "invalid name: an identifier may not start with a NUL character followed by \
                      fk_set_columns"
                .to_owned(),
            sqlstate: "42602",
        });
    }
    Ok(())
}

/// Rewrite every `ON DELETE SET {NULL | DEFAULT} (cols)` of a foreign key in `tokens` into the
/// marker form the generic parser accepts (see the module docs), and refuse a column list on
/// `ON UPDATE`.
///
/// Only the foreign key shape is touched: the action must follow a `REFERENCES <table>` of the
/// same constraint (same parenthesis level, no `,` or `;` between them). Anything else is left as
/// written, for the parser to judge.
///
/// # Errors
/// `0A000` for a column list on `ON UPDATE`; `42601` for an empty column list.
pub(super) fn rewrite_set_columns(tokens: &mut Vec<TokenWithSpan>) -> Result<(), Error> {
    if !tokens.iter().any(|t| word_ci(Some(t), "references")) {
        return Ok(());
    }
    let mut i = 0;
    while i < tokens.len() {
        i = set_columns_at(tokens, i)?.unwrap_or(i + 1);
    }
    Ok(())
}

/// Rewrite the `ON {DELETE | UPDATE} SET {NULL | DEFAULT} (cols)` whose `ON` is at `i`, returning
/// the index to resume from, or `None` (tokens untouched) when `i` does not start that shape.
fn set_columns_at(tokens: &mut Vec<TokenWithSpan>, i: usize) -> Result<Option<usize>, Error> {
    if !is_keyword(tokens, i, Keyword::ON) {
        return Ok(None);
    }
    let Some(event) = nonws(tokens, i) else {
        return Ok(None);
    };
    let on_delete = is_keyword(tokens, event, Keyword::DELETE);
    if !on_delete && !is_keyword(tokens, event, Keyword::UPDATE) {
        return Ok(None);
    }
    let Some(set) = nonws(tokens, event).filter(|&s| is_keyword(tokens, s, Keyword::SET)) else {
        return Ok(None);
    };
    let Some(action) = nonws(tokens, set).filter(|&a| {
        is_keyword(tokens, a, Keyword::NULL) || is_keyword(tokens, a, Keyword::DEFAULT)
    }) else {
        return Ok(None);
    };
    let Some(open) = nonws(tokens, action)
        .filter(|&o| matches!(tokens.get(o).map(|t| &t.token), Some(Token::LParen)))
    else {
        return Ok(None);
    };
    let Some(references) = references_before(tokens, i) else {
        return Ok(None);
    };
    let Some(close) = matching_paren(tokens, open) else {
        return Ok(None);
    };
    if !on_delete {
        return Err(Error::Coded {
            message: "a column list is supported only for ON DELETE actions, not for ON UPDATE \
                      SET NULL / SET DEFAULT"
                .to_owned(),
            sqlstate: "0A000",
        });
    }
    let inner: Vec<TokenWithSpan> = tokens.get(open + 1..close).unwrap_or_default().to_vec();
    if inner
        .iter()
        .all(|t| matches!(t.token, Token::Whitespace(_)))
    {
        return Err(Error::Syntax(
            "ON DELETE SET NULL / SET DEFAULT needs at least one column in its column list"
                .to_owned(),
        ));
    }
    // Where the list goes: before the `)` of the referenced column list, or as a new list right
    // after the referenced table's name.
    let Some(after_name) = table_name_end(tokens, references) else {
        return Ok(None);
    };
    let referred_open = nonws(tokens, after_name)
        .filter(|&o| matches!(tokens.get(o).map(|t| &t.token), Some(Token::LParen)));
    // The column list leaves the action first: it sits after the insertion point.
    tokens.drain(open..=close);
    let marker = TokenWithSpan::wrap(Token::Word(sqlparser::tokenizer::Word {
        value: SET_COLUMNS_MARKER.to_owned(),
        quote_style: Some('"'),
        keyword: Keyword::NoKeyword,
    }));
    let inserted = match referred_open.and_then(|o| matching_paren(tokens, o).map(|c| (o, c))) {
        Some((_, referred_close)) if referred_close < open => {
            let mut add = vec![TokenWithSpan::wrap(Token::Comma), marker];
            add.push(TokenWithSpan::wrap(Token::Comma));
            add.extend(inner);
            let n = add.len();
            tokens.splice(referred_close..referred_close, add);
            n
        },
        _ => {
            let mut add = vec![TokenWithSpan::wrap(Token::LParen), marker];
            add.push(TokenWithSpan::wrap(Token::Comma));
            add.extend(inner);
            add.push(TokenWithSpan::wrap(Token::RParen));
            let n = add.len();
            let at = after_name + 1;
            tokens.splice(at..at, add);
            n
        },
    };
    Ok(Some(action + inserted + 1))
}

/// The index of the `REFERENCES` keyword of the constraint whose action starts at `at`: walk back
/// at the same parenthesis level until the level opens, or a `,` or `;` ends the constraint.
fn references_before(tokens: &[TokenWithSpan], at: usize) -> Option<usize> {
    let mut nested = 0usize;
    for j in (0..at).rev() {
        match tokens.get(j).map(|t| &t.token) {
            Some(Token::RParen) => nested += 1,
            Some(Token::LParen | Token::Comma | Token::SemiColon) if nested == 0 => return None,
            Some(Token::LParen) => nested -= 1,
            Some(Token::Word(w))
                if nested == 0 && w.quote_style.is_none() && w.keyword == Keyword::REFERENCES =>
            {
                return Some(j);
            },
            _ => {},
        }
    }
    None
}

/// The index of the last token of the (possibly qualified) table name after `REFERENCES` at
/// `references`.
fn table_name_end(tokens: &[TokenWithSpan], references: usize) -> Option<usize> {
    let mut end = nonws(tokens, references)
        .filter(|&n| matches!(tokens.get(n).map(|t| &t.token), Some(Token::Word(_))))?;
    while let Some(dot) = nonws(tokens, end)
        .filter(|&d| matches!(tokens.get(d).map(|t| &t.token), Some(Token::Period)))
    {
        end = nonws(tokens, dot)
            .filter(|&n| matches!(tokens.get(n).map(|t| &t.token), Some(Token::Word(_))))?;
    }
    Some(end)
}

/// Split a parsed referenced column list at the marker: the referenced columns (folded) and the
/// `ON DELETE SET` columns (folded; empty when the list carries no marker).
pub(super) fn split_referred_columns(idents: &[sql::Ident]) -> (Vec<String>, Vec<String>) {
    let marker = idents
        .iter()
        .position(|i| i.quote_style.is_some() && i.value == SET_COLUMNS_MARKER);
    marker.map_or_else(
        || (idents.iter().map(fold_ident).collect(), Vec::new()),
        |at| {
            (
                idents.iter().take(at).map(fold_ident).collect(),
                idents.iter().skip(at + 1).map(fold_ident).collect(),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::ast;

    fn fk_of(sql: &str) -> ast::TableConstraint {
        let stmt = crate::parser::parse(sql).unwrap();
        match stmt {
            ast::Statement::CreateTable(ct) => ct
                .constraints
                .into_iter()
                .find(|c| matches!(c, ast::TableConstraint::ForeignKey { .. }))
                .unwrap(),
            ast::Statement::AlterTable(at) => match at.action {
                ast::AlterTableAction::AddConstraint(c) => c,
                other => panic!("unexpected action {other:?}"),
            },
            other => panic!("unexpected statement {other:?}"),
        }
    }

    fn parts(
        c: &ast::TableConstraint,
    ) -> (Vec<String>, Vec<String>, Option<ast::ReferentialAction>) {
        let ast::TableConstraint::ForeignKey {
            referred_columns,
            on_delete_set_columns,
            on_delete,
            ..
        } = c
        else {
            panic!("expected a foreign key");
        };
        (
            referred_columns.clone(),
            on_delete_set_columns.clone(),
            *on_delete,
        )
    }

    #[test]
    fn table_level_list_is_carried() {
        let fk = fk_of(
            "CREATE TABLE v (org INT, d_id INT, name TEXT, CONSTRAINT f FOREIGN KEY (org, d_id, \
             name) REFERENCES d (org, d_id, name) ON DELETE SET NULL (D_ID))",
        );
        assert_eq!(
            parts(&fk),
            (
                vec!["org".into(), "d_id".into(), "name".into()],
                vec!["d_id".into()],
                Some(ast::ReferentialAction::SetNull)
            )
        );
    }

    #[test]
    fn list_without_referred_columns_and_with_on_update() {
        let fk = fk_of(
            "CREATE TABLE v (a INT, b INT, FOREIGN KEY (a, b) REFERENCES s.d ON UPDATE CASCADE \
             ON DELETE SET DEFAULT (\"b\", a))",
        );
        assert_eq!(
            parts(&fk),
            (
                vec![],
                vec!["b".into(), "a".into()],
                Some(ast::ReferentialAction::SetDefault)
            )
        );
    }

    #[test]
    fn column_level_and_alter_forms() {
        let fk = fk_of("CREATE TABLE v (a INT REFERENCES d (x) ON DELETE SET NULL (a))");
        assert_eq!(parts(&fk).1, vec!["a".to_owned()]);
        let fk = fk_of(
            "ALTER TABLE v ADD CONSTRAINT f FOREIGN KEY (a, b) REFERENCES d (x, y) ON DELETE SET \
             NULL (b) DEFERRABLE",
        );
        assert_eq!(
            parts(&fk),
            (
                vec!["x".into(), "y".into()],
                vec!["b".into()],
                Some(ast::ReferentialAction::SetNull)
            )
        );
    }

    #[test]
    fn plain_actions_are_unchanged() {
        let fk = fk_of(
            "CREATE TABLE v (a INT, FOREIGN KEY (a) REFERENCES d (x) ON DELETE SET NULL ON UPDATE \
             SET NULL)",
        );
        assert_eq!(
            parts(&fk),
            (
                vec!["x".into()],
                vec![],
                Some(ast::ReferentialAction::SetNull)
            )
        );
    }

    #[test]
    fn update_list_and_empty_list_are_refused() {
        let err = crate::parser::parse(
            "CREATE TABLE v (a INT, FOREIGN KEY (a) REFERENCES d (x) ON UPDATE SET NULL (a))",
        )
        .unwrap_err();
        assert_eq!(err.sqlstate(), "0A000");
        let err = crate::parser::parse(
            "CREATE TABLE v (a INT, FOREIGN KEY (a) REFERENCES d (x) ON DELETE SET NULL ())",
        )
        .unwrap_err();
        assert_eq!(err.sqlstate(), "42601");
    }

    #[test]
    fn a_forged_marker_is_refused() {
        let err = crate::parser::parse(
            "CREATE TABLE v (a INT, FOREIGN KEY (a) REFERENCES d (x, \"\0fk_set_columns\0\", a))",
        )
        .unwrap_err();
        assert_eq!(err.sqlstate(), "42602");
    }
}
