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
    /// `SET name = expr` or `name := expr`: assign to a variable. In a trigger function the
    /// target may also be a field of the new row, `NEW.col` (named `new.col` here).
    Assign {
        /// Target variable name (folded; `record.field` for a qualified target).
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
    /// `PERFORM expr` — evaluate an expression and discard the result (the reference dialect's
    /// spelling for a call made for its side effects).
    Perform(ast::Expr),
    /// `RAISE expr` — abort the procedure with the message `expr` evaluates to.
    Raise(ast::Expr),
    /// `RAISE level 'format' [, arg ...]`: each `%` in `format` is replaced by the next argument's
    /// value (`%%` is a literal `%`). `EXCEPTION` aborts the routine like `RAISE expr`; `NOTICE`,
    /// `WARNING` and `INFO` send the message to the client as a notice and carry on; `LOG` and
    /// `DEBUG` carry on without sending anything.
    RaiseMessage {
        /// The level.
        level: RaiseLevel,
        /// The format string.
        format: String,
        /// The values substituted for its `%` placeholders, in order.
        args: Vec<ast::Expr>,
    },
    /// `RETURN [expr]` — stop the routine. A procedure uses the bare `RETURN` (`None`); a function
    /// uses `RETURN expr` to yield its result value.
    Return(Option<ast::Expr>),
    /// A trigger function's `RETURN NEW` / `RETURN OLD` / `RETURN NULL`: which row the firing
    /// write proceeds with. Never produced by the parser; trigger firing rewrites a `RETURN` into it.
    ReturnRow(TriggerReturn),
    /// An embedded SQL data statement (`INSERT`/`UPDATE`/`DELETE`/`SELECT`/`CALL`). Boxed because an
    /// [`ast::Statement`] is far larger than the other variants.
    Sql(Box<ast::Statement>),
    /// `SELECT ... INTO [STRICT] target [, target ...] ...`: run the query and assign the columns of
    /// its first row to the targets, in order. Inside a block, `INTO` names variables (or a trigger
    /// function's `NEW.col` fields), never a new table. No row sets every target to `NULL`; a
    /// target past the last column is set to `NULL` and a column past the last target is ignored.
    /// With `STRICT` the query must return exactly one row.
    SelectInto {
        /// The query, with its `INTO` clause removed.
        query: Box<ast::Statement>,
        /// The target names (folded; `record.field` for a qualified target), in order.
        targets: Vec<String>,
        /// Whether `STRICT` was given.
        strict: bool,
    },
    /// A nested `BEGIN ... [EXCEPTION ...] END` block.
    Block(ScriptBlock),
}

/// The level of a `RAISE level 'format'` statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RaiseLevel {
    /// `DEBUG`: not sent to the client.
    Debug,
    /// `LOG`: not sent to the client.
    Log,
    /// `INFO`: a notice of severity `INFO`.
    Info,
    /// `NOTICE`: a notice of severity `NOTICE`.
    Notice,
    /// `WARNING`: a notice of severity `WARNING`.
    Warning,
    /// `EXCEPTION`: an error (`P0001`) that aborts the routine.
    Exception,
}

/// The row a trigger function returns: the (possibly modified) new row, the old row, or none
/// (`NULL`), which skips the row in a `BEFORE ... FOR EACH ROW` trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TriggerReturn {
    /// `RETURN NEW`.
    New,
    /// `RETURN OLD`.
    Old,
    /// `RETURN NULL`.
    Null,
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
    // The embedded statements are user SQL: the same `EXCLUDE` rewrite as a top-level statement,
    // with this parse's own rewrite scope.
    let _exclusion_scope = super::exclusion::RewriteScope::enter(false);
    let tokens = super::embedded_tokens(sql)?;
    let mut parser = Parser::new(&dialect).with_tokens_with_locations(tokens);
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
        Some("perform") => {
            parser.next_token();
            Ok(ScriptStmt::Perform(parse_expr(parser)?))
        },
        Some("raise") => {
            parser.next_token();
            parse_raise(parser)
        },
        Some("return") => {
            parser.next_token();
            // A bare `RETURN` (at a `;`, block terminator, or end of input) yields no value — the
            // procedure form; `RETURN expr` yields a function's result.
            let value = if matches!(parser.peek_token().token, Token::SemiColon | Token::EOF)
                || peek_word(parser)
                    .is_some_and(|w| matches!(w.as_str(), "end" | "else" | "elsif" | "exception"))
            {
                None
            } else {
                Some(parse_expr(parser)?)
            };
            Ok(ScriptStmt::Return(value))
        },
        // `name := expr` / `record.field := expr`.
        Some(_) if is_walrus_assignment(parser) => parse_assign(parser),
        // A query whose `INTO` names variables to assign, else an ordinary query.
        Some("select" | "with") => parse_select_into(parser)?.map_or_else(|| parse_sql(parser), Ok),
        // Anything else is an embedded SQL data statement.
        _ => parse_sql(parser),
    }
}

/// An embedded SQL data statement.
fn parse_sql(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let stmt = parser.parse_statement().map_err(syntax)?;
    Ok(ScriptStmt::Sql(Box::new(convert_statement(stmt)?)))
}

/// A query with an `INTO` clause at the current position, as a [`ScriptStmt::SelectInto`]. Returns
/// `None`, consuming nothing, when the statement is not a query or its `INTO` is not at the top level
/// of the query.
///
/// The `INTO` clause may sit anywhere in the query (`SELECT a INTO x FROM t` or `SELECT a FROM t
/// INTO x`). Its tokens are cut out and the rest is parsed as an ordinary statement.
fn parse_select_into(parser: &mut Parser) -> Result<Option<ScriptStmt>, Error> {
    use sqlparser::keywords::Keyword;
    // The statement's tokens up to its `;` (or the end of input). The statement's verb is the first
    // top-level one: in `WITH c AS (...) INSERT INTO t ...` it is `INSERT`, and that `INTO` is its own.
    let mut tokens = Vec::new();
    let mut depth = 0usize;
    let mut verb = None;
    let mut into_at = None;
    loop {
        let token = parser.peek_nth_token_ref(tokens.len());
        match &token.token {
            Token::EOF => break,
            Token::SemiColon if depth == 0 => break,
            Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Word(w) if depth == 0 && w.quote_style.is_none() => match w.keyword {
                Keyword::SELECT
                | Keyword::INSERT
                | Keyword::UPDATE
                | Keyword::DELETE
                | Keyword::MERGE
                | Keyword::VALUES
                | Keyword::TABLE
                    if verb.is_none() =>
                {
                    verb = Some(w.keyword);
                },
                Keyword::INTO if verb == Some(Keyword::SELECT) && into_at.is_none() => {
                    into_at = Some(tokens.len());
                },
                _ => {},
            },
            _ => {},
        }
        tokens.push(token.clone());
    }
    let Some(into_at) = into_at else {
        return Ok(None);
    };

    // `[STRICT] target [, target ...]`.
    let after_into: Vec<_> = tokens.split_off(into_at).into_iter().skip(1).collect();
    let mut target_parser =
        Parser::new(&NusaParserDialect).with_tokens_with_locations(after_into.clone());
    let strict = eat_word(&mut target_parser, "strict");
    let mut targets = Vec::new();
    loop {
        let mut name = fold_ident(&target_parser.parse_identifier().map_err(syntax)?);
        if target_parser.consume_token(&Token::Period) {
            let field = fold_ident(&target_parser.parse_identifier().map_err(syntax)?);
            name = format!("{name}.{field}");
        }
        targets.push(name);
        if !target_parser.consume_token(&Token::Comma) {
            break;
        }
    }
    let targets_len = consumed(&target_parser, after_into.len());
    let into_len = 1 + targets_len;

    // The query without its `INTO` clause.
    tokens.extend(after_into.into_iter().skip(targets_len));
    let query_len = tokens.len();
    let mut query_parser = Parser::new(&NusaParserDialect).with_tokens_with_locations(tokens);
    let stmt = query_parser.parse_statement().map_err(syntax)?;
    let consumed = consumed(&query_parser, query_len);
    if consumed < into_at {
        return Err(Error::Syntax(
            "INTO must be part of the query in a NusaScript block".to_owned(),
        ));
    }
    // Move past the statement, its `INTO` clause included.
    for _ in 0..consumed + into_len {
        parser.next_token();
    }
    let query = convert_statement(stmt)?;
    if !matches!(
        query,
        ast::Statement::Select(_) | ast::Statement::SetOperation(_)
    ) {
        return Err(Error::Syntax(
            "INTO in a NusaScript block must follow a query".to_owned(),
        ));
    }
    Ok(Some(ScriptStmt::SelectInto {
        query: Box::new(query),
        targets,
        strict,
    }))
}

/// How many of a parser's `total` tokens (none of them whitespace) it has consumed.
fn consumed(parser: &Parser, total: usize) -> usize {
    let mut rest = 0;
    while !matches!(parser.peek_nth_token_ref(rest).token, Token::EOF) {
        rest += 1;
    }
    total - rest
}

/// The rest of a `RAISE`: `level 'format' [, arg ...]` when a level word is followed by a string
/// literal, else the expression form `RAISE expr`.
fn parse_raise(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let level = match peek_word(parser).as_deref() {
        Some("debug") => Some(RaiseLevel::Debug),
        Some("log") => Some(RaiseLevel::Log),
        Some("info") => Some(RaiseLevel::Info),
        Some("notice") => Some(RaiseLevel::Notice),
        Some("warning") => Some(RaiseLevel::Warning),
        Some("exception") => Some(RaiseLevel::Exception),
        _ => None,
    };
    let literal_follows = matches!(
        parser.peek_nth_token_ref(1).token,
        Token::SingleQuotedString(_)
    );
    let Some(level) = level.filter(|_| literal_follows) else {
        return Ok(ScriptStmt::Raise(parse_expr(parser)?));
    };
    parser.next_token();
    let Token::SingleQuotedString(format) = parser.next_token().token else {
        return Err(Error::Syntax(
            "RAISE expects a format string after its level".to_owned(),
        ));
    };
    let mut args = Vec::new();
    while parser.consume_token(&Token::Comma) {
        args.push(parse_expr(parser)?);
    }
    Ok(ScriptStmt::RaiseMessage {
        level,
        format,
        args,
    })
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

/// Whether the statement at the current position is `name := ...` or `record.field := ...`.
fn is_walrus_assignment(parser: &Parser) -> bool {
    match parser.peek_nth_token_ref(1).token {
        Token::Assignment => true,
        Token::Period => {
            matches!(parser.peek_nth_token_ref(2).token, Token::Word(_))
                && matches!(parser.peek_nth_token_ref(3).token, Token::Assignment)
        },
        _ => false,
    }
}

/// `name = expr` or `name := expr` (after `SET`, or the `:=` statement form); `name` may be a
/// `record.field` pair.
fn parse_assign(parser: &mut Parser) -> Result<ScriptStmt, Error> {
    let mut name = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    if parser.consume_token(&Token::Period) {
        let field = fold_ident(&parser.parse_identifier().map_err(syntax)?);
        name = format!("{name}.{field}");
    }
    if !parser.consume_token(&Token::Assignment) {
        parser.expect_token(&Token::Eq).map_err(syntax)?;
    }
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
