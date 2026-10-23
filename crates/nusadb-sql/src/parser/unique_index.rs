//! Surface support for unique indexes that the generic parser does not model:
//!
//! - The stored form of `CREATE UNIQUE INDEX ... NULLS NOT DISTINCT`. The index catalog has no
//!   field for the setting, so it travels inside the index's predicate text behind a NUL-led tag
//!   ([`tag_index_predicate`] / [`untag_index_predicate`]). The canonical text of a predicate never
//!   starts with a NUL byte (a NUL can only appear quoted, inside a literal or identifier), so no
//!   real predicate can carry the tag. What follows the tag is the partial-index predicate, if any.
//!   The tag is written only for `NULLS NOT DISTINCT`: an index without it treats NULL keys as
//!   distinct, the default.
//! - The `ON CONFLICT (cols) WHERE pred` inference clause, which picks a partial unique index as the
//!   arbiter. sqlparser has no grammar for the `WHERE`, so it is moved into a marker at the end of
//!   the column list before parsing ([`rewrite_conflict_where`]) and taken back out when the
//!   `ON CONFLICT` is converted ([`take_conflict_where`]).

use sqlparser::keywords::Keyword;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, TokenWithSpan};

use super::{NusaParserDialect, matching_paren, word_ci};
use crate::ast;
use crate::error::Error;

/// The tag that marks a stored index predicate as `NULLS NOT DISTINCT`.
const NULLS_NOT_DISTINCT_TAG: &str = "\0nulls_not_distinct\0";

/// The predicate text an index is cataloged with: `predicate` unchanged for the default `NULLS
/// DISTINCT`, or the `NULLS NOT DISTINCT` tag followed by the predicate (empty for a full index).
pub(super) fn tag_index_predicate(
    predicate: Option<String>,
    nulls_not_distinct: bool,
) -> Option<String> {
    if !nulls_not_distinct {
        return predicate;
    }
    Some(format!(
        "{NULLS_NOT_DISTINCT_TAG}{}",
        predicate.unwrap_or_default()
    ))
}

/// Split a cataloged index predicate into the partial-index predicate (`None` for a full index)
/// and whether the index is `NULLS NOT DISTINCT`.
pub(super) fn untag_index_predicate(stored: Option<&str>) -> (Option<&str>, bool) {
    stored.map_or((None, false), |text| {
        match text.strip_prefix(NULLS_NOT_DISTINCT_TAG) {
            Some("") => (None, true),
            Some(rest) => (Some(rest), true),
            None => (Some(text), false),
        }
    })
}

/// The top-level `AND` conjuncts of the boolean expression `sql`, in the internal AST (identifiers
/// folded, parentheses dropped), so two spellings of one predicate compare equal.
pub(super) fn predicate_conjuncts(sql: &str) -> Result<Vec<ast::Expr>, Error> {
    fn flatten(expr: ast::Expr, out: &mut Vec<ast::Expr>) {
        match expr {
            ast::Expr::Binary {
                left,
                op: ast::BinaryOp::And,
                right,
            } => {
                flatten(*left, out);
                flatten(*right, out);
            },
            other => out.push(other),
        }
    }
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect)
        .try_with_sql(sql)
        .map_err(|e| Error::Syntax(e.to_string()))?;
    let expr = parser
        .parse_expr()
        .map_err(|e| Error::Syntax(e.to_string()))?;
    if !matches!(parser.peek_token().token, Token::EOF) {
        return Err(Error::Syntax(
            "an index or ON CONFLICT predicate must be a single expression".to_owned(),
        ));
    }
    let mut out = Vec::new();
    flatten(super::convert_expr(expr)?, &mut out);
    Ok(out)
}

/// The marker that carries an `ON CONFLICT (cols) WHERE pred` predicate through the generic
/// parser: a quoted identifier appended to the column list whose text is the marker followed by the
/// predicate. A quoted identifier can hold any text, and a real column name never starts with NUL.
const CONFLICT_WHERE_MARKER: &str = "\0conflict_where\0";

/// Rewrite every `ON CONFLICT (cols) WHERE <pred> DO ...` of an `INSERT` in `tokens` to `ON CONFLICT
/// (cols, "<marker><pred>") DO ...`, which sqlparser parses. The marker rides along wherever the
/// `INSERT` sits (a top-level statement, `EXPLAIN`, `PREPARE ... AS`, a data-modifying CTE, a
/// procedure body), and [`take_conflict_where`] turns it back into the predicate when the
/// `ON CONFLICT` is converted.
///
/// Only the upsert shape is touched: an `ON CONFLICT` preceded by `INSERT INTO` in the same
/// statement at the same parenthesis level, followed by a parenthesized column list, `WHERE`, one
/// complete expression and `DO`, all before the statement or the level ends. Anything else (a
/// table or function named `conflict`, a partial index on it, a join condition) is left as written,
/// for the parser to judge.
pub(super) fn rewrite_conflict_where(tokens: &mut Vec<TokenWithSpan>) {
    if !tokens.iter().any(|t| word_ci(Some(t), "conflict")) {
        return;
    }
    let mut i = 0;
    while i < tokens.len() {
        i = conflict_where_at(tokens, i).unwrap_or(i + 1);
    }
}

/// The next non-whitespace token index after `from`.
fn nonws(tokens: &[TokenWithSpan], from: usize) -> Option<usize> {
    (from + 1..tokens.len())
        .find(|&j| !matches!(tokens.get(j).map(|t| &t.token), Some(Token::Whitespace(_))))
}

/// Whether `INSERT INTO` appears before `at` in the same statement and at the same parenthesis
/// level: walk back, skipping nested parentheses, until the level opens or a `;` ends the previous
/// statement.
fn insert_precedes(tokens: &[TokenWithSpan], at: usize) -> bool {
    let mut nested = 0usize;
    let into_follows = |j: usize| {
        nonws(tokens, j).is_some_and(|n| {
            matches!(tokens.get(n).map(|t| &t.token), Some(Token::Word(w)) if w.keyword == Keyword::INTO)
        })
    };
    for j in (0..at).rev() {
        match tokens.get(j).map(|t| &t.token) {
            Some(Token::RParen) => nested += 1,
            Some(Token::LParen | Token::SemiColon) if nested == 0 => return false,
            Some(Token::LParen) => nested -= 1,
            Some(Token::Word(w))
                if nested == 0 && w.keyword == Keyword::INSERT && into_follows(j) =>
            {
                return true;
            },
            _ => {},
        }
    }
    false
}

/// Rewrite the upsert `ON CONFLICT (cols) WHERE <pred> DO` whose `ON` is at `i`, returning the index
/// to resume from, or `None` (tokens untouched) when `i` does not start that shape.
fn conflict_where_at(tokens: &mut Vec<TokenWithSpan>, i: usize) -> Option<usize> {
    if !matches!(tokens.get(i).map(|t| &t.token), Some(Token::Word(w)) if w.keyword == Keyword::ON)
    {
        return None;
    }
    let conflict = nonws(tokens, i).filter(|&c| word_ci(tokens.get(c), "conflict"))?;
    let open = nonws(tokens, conflict)
        .filter(|&o| matches!(tokens.get(o).map(|t| &t.token), Some(Token::LParen)))?;
    let close = matching_paren(tokens, open)?;
    let where_at = nonws(tokens, close).filter(|&w| word_ci(tokens.get(w), "where"))?;
    if !insert_precedes(tokens, i) {
        return None;
    }
    // The predicate runs to the `DO` of the conflict action at this level; a `;` or the level's
    // closing parenthesis first means this is not an upsert.
    let mut depth = 0usize;
    let mut do_at = None;
    for (k, tok) in tokens.iter().enumerate().skip(where_at + 1) {
        match &tok.token {
            Token::LParen => depth += 1,
            Token::RParen | Token::SemiColon if depth == 0 => break,
            Token::RParen => depth -= 1,
            Token::Word(w) if depth == 0 && w.keyword == Keyword::DO => {
                do_at = Some(k);
                break;
            },
            _ => {},
        }
    }
    let do_at = do_at?;
    let predicate: Vec<TokenWithSpan> = tokens.get(where_at + 1..do_at)?.to_vec();
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).with_tokens_with_locations(predicate);
    let expr = parser.parse_expr().ok()?;
    if !matches!(parser.peek_token().token, Token::EOF) {
        return None;
    }
    tokens.drain(where_at..do_at);
    let marker = Token::Word(sqlparser::tokenizer::Word {
        value: format!("{CONFLICT_WHERE_MARKER}{expr}"),
        quote_style: Some('"'),
        keyword: Keyword::NoKeyword,
    });
    tokens.splice(
        close..close,
        [
            TokenWithSpan::wrap(Token::Comma),
            TokenWithSpan::wrap(marker),
        ],
    );
    Some(close + 2)
}

/// Refuse a quoted identifier that starts like the `ON CONFLICT ... WHERE` marker (`42602`), so a
/// user-written name can never pass for one. Other names holding a NUL keep their usual handling
/// (the reserved setting names start with one and are refused where they are looked up).
pub(super) fn refuse_forged_markers(tokens: &[TokenWithSpan]) -> Result<(), Error> {
    let forged = tokens.iter().any(|t| {
        matches!(&t.token, Token::Word(w) if w.quote_style.is_some() && w.value.starts_with(CONFLICT_WHERE_MARKER))
    });
    if forged {
        return Err(Error::Coded {
            message: "invalid name: an identifier may not start with a NUL character followed by \
                      conflict_where"
                .to_owned(),
            sqlstate: "42602",
        });
    }
    Ok(())
}

/// Take the `WHERE` predicate marker [`rewrite_conflict_where`] left at the end of an `ON CONFLICT`
/// column list, returning the predicate text. `None` when the list carries no marker.
pub(super) fn take_conflict_where(columns: &mut Vec<sqlparser::ast::Ident>) -> Option<String> {
    let last = columns.last()?;
    let predicate = last.value.strip_prefix(CONFLICT_WHERE_MARKER)?.to_owned();
    columns.pop();
    Some(predicate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_round_trips() {
        assert_eq!(tag_index_predicate(None, false), None);
        assert_eq!(
            tag_index_predicate(Some("a > 1".to_owned()), false).as_deref(),
            Some("a > 1")
        );
        let full = tag_index_predicate(None, true);
        assert_eq!(untag_index_predicate(full.as_deref()), (None, true));
        let partial = tag_index_predicate(Some("s = 'x'".to_owned()), true);
        assert_eq!(
            untag_index_predicate(partial.as_deref()),
            (Some("s = 'x'"), true)
        );
        assert_eq!(untag_index_predicate(Some("a > 1")), (Some("a > 1"), false));
        assert_eq!(untag_index_predicate(None), (None, false));
    }

    #[test]
    fn conjuncts_ignore_parentheses_and_case() {
        let a = predicate_conjuncts("(S = 'active') AND (b > 1)").unwrap();
        let b = predicate_conjuncts("s = 'active' AND b > 1").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn conflict_where_is_parsed_and_attached() {
        let stmt = crate::parser::parse(
            "INSERT INTO t VALUES (1) ON CONFLICT (b) WHERE s = 'active' DO NOTHING",
        )
        .unwrap();
        let ast::Statement::Insert(ins) = stmt else {
            panic!("expected an INSERT");
        };
        let target = ins.on_conflict.and_then(|c| c.target);
        assert_eq!(
            target,
            Some(ast::ConflictTarget::ColumnsWhere {
                columns: vec!["b".to_owned()],
                predicate: "s = 'active'".to_owned(),
            })
        );
    }

    #[test]
    fn conflict_where_needs_a_column_list() {
        assert!(
            crate::parser::parse(
                "INSERT INTO t VALUES (1) ON CONFLICT ON CONSTRAINT c WHERE s = 'a' DO NOTHING"
            )
            .is_err()
        );
        assert!(
            crate::parser::parse("INSERT INTO t VALUES (1) ON CONFLICT (b) WHERE s = 'a'").is_err()
        );
    }

    #[test]
    fn other_on_conflict_where_shapes_are_left_alone() {
        for sql in [
            "CREATE INDEX ci ON conflict (a) WHERE a > 0",
            "CREATE UNIQUE INDEX cu ON conflict (a) WHERE s = 'x'",
            "CREATE INDEX ci ON \"conflict\" (a) WHERE a > 0",
            "SELECT * FROM t JOIN conflict ON conflict(t.c) WHERE t.id > 0",
            "INSERT INTO x SELECT t.id FROM t JOIN conflict ON conflict(t.c) WHERE t.id > 0",
        ] {
            assert!(crate::parser::parse(sql).is_ok(), "{sql}");
        }
        // A join condition before a real upsert: only the upsert's WHERE is taken.
        let stmt = crate::parser::parse(
            "INSERT INTO x SELECT t.id FROM t JOIN conflict ON conflict(t.c) WHERE t.id > 0 \
             ON CONFLICT (b) WHERE s = 'a' DO NOTHING",
        )
        .unwrap();
        let ast::Statement::Insert(ins) = stmt else {
            panic!("expected an INSERT");
        };
        assert!(matches!(
            ins.on_conflict.and_then(|c| c.target),
            Some(ast::ConflictTarget::ColumnsWhere { .. })
        ));
    }

    #[test]
    fn a_predicate_never_bleeds_into_the_next_statement() {
        let stmts = crate::parser::parse_statements(
            "CREATE INDEX ci ON conflict (a) WHERE a > 0; \
             INSERT INTO conflict VALUES (1, 'x') ON CONFLICT DO NOTHING",
        )
        .unwrap();
        assert_eq!(stmts.len(), 2);
        let mut tokens = sqlparser::tokenizer::Tokenizer::new(
            &NusaParserDialect,
            "CREATE INDEX ci ON conflict (a) WHERE a > 0; INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING",
        )
        .tokenize_with_location()
        .unwrap();
        let before = tokens.clone();
        rewrite_conflict_where(&mut tokens);
        assert_eq!(tokens, before);
    }

    #[test]
    fn a_forged_marker_is_refused() {
        let err = crate::parser::parse(
            "INSERT INTO t VALUES (1) ON CONFLICT (b, \"\0conflict_where\0true\") DO NOTHING",
        )
        .unwrap_err();
        assert_eq!(err.sqlstate(), "42602");
        assert!(predicate_conjuncts("s = 'a'; DROP TABLE t").is_err());
    }
}
