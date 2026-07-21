//! NusaScript — the procedural language for stored-procedure bodies.
//!
//! A procedure body that begins with `BEGIN` is a NusaScript block rather than a plain sequence of
//! SQL statements. The block supports variable declarations, assignment, `IF`/`WHILE` control flow,
//! `RAISE`, `RETURN`, and embedded SQL data statements. Conditions and assigned values are ordinary
//! SQL expressions (reusing the SQL expression parser); embedded SQL and expressions may reference
//! declared variables by name and the procedure's parameters as `$1`..`$n`.
//!
//! This module only *parses* the block into a [`ScriptStmt`] tree; the interpreter that runs it lives
//! in the executor (`executor/script.rs`).
#![allow(clippy::wildcard_imports)]
#![allow(
    clippy::redundant_pub_crate,
    reason = "these items are re-exported at crate scope by the parent (`pub(crate) use script::…`); \
              `pub(crate)` is the correct visibility — the lint's `pub` suggestion would instead trip \
              `unreachable_pub`"
)]

use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use super::*;
use crate::ast;
use crate::error::Error;

/// A `BEGIN <body> [EXCEPTION WHEN OTHERS THEN <handler>] END` block. When the body raises
/// an error and a handler is present, the body's writes are rolled back (to a savepoint) and the
/// handler runs in their place.
#[derive(Debug, Clone)]
pub(crate) struct ScriptBlock {
    /// The block body.
    pub(crate) body: Vec<ScriptStmt>,
    /// The `EXCEPTION WHEN OTHERS THEN` handler, if any.
    pub(crate) handler: Option<Vec<ScriptStmt>>,
}

/// One NusaScript statement.
#[derive(Debug, Clone)]
pub(crate) enum ScriptStmt {
    /// `DECLARE name TYPE [DEFAULT expr]` — introduce a variable (the declared type is accepted but
    /// not enforced; variables hold any value). Initialized to `DEFAULT` or `NULL`.
    Declare {
        /// Variable name (folded).
        name: String,
        /// Initial value expression, or `None` for `NULL`.
        default: Option<ast::Expr>,
    },
    /// `SET name = expr` — assign to a variable.
    Assign {
        /// Target variable name (folded).
        name: String,
        /// Value expression.
        value: ast::Expr,
    },
    /// `IF cond THEN ... [ELSIF cond THEN ...]* [ELSE ...] END IF`.
    If {
        /// `(condition, body)` for the `IF` and each `ELSIF`, in order; the first true one runs.
        arms: Vec<(ast::Expr, Vec<Self>)>,
        /// The `ELSE` body, if any.
        els: Option<Vec<Self>>,
    },
    /// `WHILE cond LOOP ... END LOOP`.
    While {
        /// Loop condition, re-evaluated before each iteration.
        cond: ast::Expr,
        /// Loop body.
        body: Vec<Self>,
    },
    /// `FOR var IN low TO high LOOP ... END LOOP` — iterate an integer variable inclusively from
    /// `low` to `high` (no iterations if `low > high`). `low`/`high` are evaluated once.
    For {
        /// Loop variable name (folded); bound to the current integer each iteration.
        var: String,
        /// Inclusive lower bound (evaluated once to an integer).
        low: ast::Expr,
        /// Inclusive upper bound (evaluated once to an integer).
        high: ast::Expr,
        /// Loop body.
        body: Vec<Self>,
    },
    /// `RAISE expr` — abort the procedure with the message `expr` evaluates to.
    Raise(ast::Expr),
    /// `RETURN` — stop the procedure (no value; procedures do not return values in v1).
    Return,
    /// An embedded SQL data statement (`INSERT`/`UPDATE`/`DELETE`/`SELECT`/`CALL`). Boxed because an
    /// [`ast::Statement`] is far larger than the other variants.
    Sql(Box<ast::Statement>),
    /// A nested `BEGIN ... [EXCEPTION ...] END` block.
    Block(ScriptBlock),
}

/// Whether a procedure body is a NusaScript block (begins with the `BEGIN` keyword) rather than a
/// plain sequence of SQL statements.
pub(crate) fn is_script(body: &str) -> bool {
    body.split_whitespace()
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("begin"))
}

/// Parse a NusaScript `BEGIN <statements> [EXCEPTION ...] END` block.
pub(crate) fn parse_script(sql: &str) -> Result<ScriptBlock, Error> {
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    let block = parse_block(&mut parser)?;
    eat_semicolons(&mut parser);
    if !matches!(parser.peek_token().token, Token::EOF) {
        return Err(Error::Syntax(
            "unexpected tokens after END of NusaScript block".to_owned(),
        ));
    }
    Ok(block)
}

/// Parse a `BEGIN <body> [EXCEPTION WHEN OTHERS THEN <handler>] END` block from the current position.
fn parse_block(parser: &mut Parser) -> Result<ScriptBlock, Error> {
    expect_word(parser, "begin")?;
    let body = parse_stmts(parser, &["exception", "end"])?;
    let handler = if eat_word(parser, "exception") {
        expect_word(parser, "when")?;
        expect_word(parser, "others")?;
        expect_word(parser, "then")?;
        Some(parse_stmts(parser, &["end"])?)
    } else {
        None
    };
    expect_word(parser, "end")?;
    Ok(ScriptBlock { body, handler })
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "used as a `map_err` function, which passes the error by value"
)]
fn syntax(error: ParserError) -> Error {
    Error::Syntax(error.to_string())
}

/// Peek the next token as a lowercased word, or `None` if it is not a word token.
fn peek_word(parser: &mut Parser) -> Option<String> {
    match parser.peek_token().token {
        Token::Word(w) => Some(w.value.to_ascii_lowercase()),
        _ => None,
    }
}

/// Consume the next token if it is the word `word`; return whether it was.
fn eat_word(parser: &mut Parser, word: &str) -> bool {
    if peek_word(parser).as_deref() == Some(word) {
        parser.next_token();
        true
    } else {
        false
    }
}

/// Consume the word `word`, or error.
fn expect_word(parser: &mut Parser, word: &str) -> Result<(), Error> {
    if eat_word(parser, word) {
        Ok(())
    } else {
        Err(Error::Syntax(format!(
            "expected `{}` in NusaScript block",
            word.to_uppercase()
        )))
    }
}

/// Skip any run of `;` separators.
fn eat_semicolons(parser: &mut Parser) {
    while matches!(parser.peek_token().token, Token::SemiColon) {
        parser.next_token();
    }
}

/// Parse an internal expression from the current position.
fn parse_expr(parser: &mut Parser) -> Result<ast::Expr, Error> {
    convert_expr(parser.parse_expr().map_err(syntax)?)
}

/// Parse statements until a terminator word (e.g. `end`, `else`, `elsif`) is next.
fn parse_stmts(parser: &mut Parser, terminators: &[&str]) -> Result<Vec<ScriptStmt>, Error> {
    let mut out = Vec::new();
    loop {
        eat_semicolons(parser);
        if matches!(parser.peek_token().token, Token::EOF) {
            break;
        }
        if peek_word(parser).is_some_and(|w| terminators.contains(&w.as_str())) {
            break;
        }
        out.push(parse_one(parser)?);
    }
    Ok(out)
}

/// Parse one NusaScript statement.
fn parse_one(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    match peek_word(parser).as_deref() {
        // A nested block (`parse_block` consumes the `BEGIN` itself).
        Some("begin") => parse_block(parser).map(ScriptStmt::Block),
        Some("declare") => {
            parser.next_token();
            parse_declare(parser)
        },
        Some("set") => {
            parser.next_token();
            parse_assign(parser)
        },
        Some("if") => {
            parser.next_token();
            parse_if(parser)
        },
        Some("while") => {
            parser.next_token();
            parse_while(parser)
        },
        Some("for") => {
            parser.next_token();
            parse_for(parser)
        },
        Some("raise") => {
            parser.next_token();
            Ok(ScriptStmt::Raise(parse_expr(parser)?))
        },
        Some("return") => {
            parser.next_token();
            Ok(ScriptStmt::Return)
        },
        // Anything else is an embedded SQL data statement.
        _ => {
            let stmt = parser.parse_statement().map_err(syntax)?;
            Ok(ScriptStmt::Sql(Box::new(convert_statement(stmt)?)))
        },
    }
}

/// `DECLARE name TYPE [DEFAULT expr]`.
fn parse_declare(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    // The declared type is accepted but not enforced (variables are dynamically typed in v1).
    parser.parse_data_type().map_err(syntax)?;
    let default = if eat_word(parser, "default") {
        Some(parse_expr(parser)?)
    } else {
        None
    };
    Ok(ScriptStmt::Declare { name, default })
}

/// `SET name = expr`.
fn parse_assign(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_token(&Token::Eq).map_err(syntax)?;
    let value = parse_expr(parser)?;
    Ok(ScriptStmt::Assign { name, value })
}

/// `IF cond THEN ... [ELSIF cond THEN ...]* [ELSE ...] END IF`.
fn parse_if(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let mut arms = Vec::new();
    let cond = parse_expr(parser)?;
    expect_word(parser, "then")?;
    arms.push((cond, parse_stmts(parser, &["elsif", "else", "end"])?));
    while eat_word(parser, "elsif") {
        let cond = parse_expr(parser)?;
        expect_word(parser, "then")?;
        arms.push((cond, parse_stmts(parser, &["elsif", "else", "end"])?));
    }
    let els = if eat_word(parser, "else") {
        Some(parse_stmts(parser, &["end"])?)
    } else {
        None
    };
    expect_word(parser, "end")?;
    expect_word(parser, "if")?;
    Ok(ScriptStmt::If { arms, els })
}

/// `WHILE cond LOOP ... END LOOP`.
fn parse_while(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let cond = parse_expr(parser)?;
    expect_word(parser, "loop")?;
    let body = parse_stmts(parser, &["end"])?;
    expect_word(parser, "end")?;
    expect_word(parser, "loop")?;
    Ok(ScriptStmt::While { cond, body })
}

/// `FOR var IN low TO high LOOP ... END LOOP`.
fn parse_for(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    use sqlparser::keywords::Keyword;
    let var = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    parser.expect_keyword(Keyword::IN).map_err(syntax)?;
    let low = parse_expr(parser)?;
    expect_word(parser, "to")?;
    let high = parse_expr(parser)?;
    expect_word(parser, "loop")?;
    let body = parse_stmts(parser, &["end"])?;
    expect_word(parser, "end")?;
    expect_word(parser, "loop")?;
    Ok(ScriptStmt::For {
        var,
        low,
        high,
        body,
    })
}
