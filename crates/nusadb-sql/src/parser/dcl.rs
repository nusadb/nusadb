//! Parsing for the data-control statements: `GRANT`, `REVOKE`, role management, and `SET ROLE`.
//!
//! These are driven against `sqlparser`'s primitives rather than its own `parse_grant`, for three
//! reasons that all point the same way:
//!
//! * its grammar demands a privilege *keyword*, so the ordinary `GRANT reader TO app` spelling of a
//!   role grant does not parse at all;
//! * it has no `WITH ADMIN OPTION`, no `GRANT OPTION FOR`, and no `ADMIN OPTION FOR`; and
//! * its `Action` enum is a union of many vendors' privilege vocabularies, most of which NusaDB has
//!   no concept of — accepting them here would mean storing grants that nothing ever checks.
//!
//! Driving the tokens directly keeps the accepted surface exactly the surface the engine enforces.

use sqlparser::ast as sql;
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use super::{NusaParserDialect, expect_statement_end, fold_ident, table_ref_name, unsupported};
use crate::ast;
use crate::error::Error;

/// Map a `sqlparser` failure onto our syntax error.
fn syntax(e: &ParserError) -> Error {
    Error::Syntax(e.to_string())
}

/// Open a parser over `sql` on the NusaDB dialect.
fn parser_for(sql: &str) -> Result<Parser<'static>, Error> {
    // The dialect is a zero-sized unit type, so leaking one reference gives the parser the
    // `'static` borrow it wants without allocating per statement.
    const DIALECT: &NusaParserDialect = &NusaParserDialect;
    Parser::new(DIALECT)
        .try_with_sql(sql)
        .map_err(|e| syntax(&e))
}

/// Whether `sql` begins with `first` followed by any of `seconds` (case-insensitive).
fn starts_with_any(sql: &str, first: &str, seconds: &[&str]) -> bool {
    let mut words = sql.split_whitespace();
    words.next().is_some_and(|w| w.eq_ignore_ascii_case(first))
        && words
            .next()
            .is_some_and(|w| seconds.iter().any(|s| w.eq_ignore_ascii_case(s)))
}

/// The next token's word text, folded to lowercase, without consuming it.
///
/// Role attributes such as `NOSUPERUSER` and `CREATEDB` are not `sqlparser` keywords, so they
/// arrive as ordinary identifiers. Reading words uniformly — keyword or not — is what lets one
/// match arm cover both.
fn peek_word(parser: &mut Parser) -> Option<String> {
    match &parser.peek_token_ref().token {
        Token::Word(word) => Some(word.value.to_lowercase()),
        _ => None,
    }
}

/// The next token's word text, folded to lowercase, but **only if it was written unquoted** —
/// `None` for a quoted identifier. Used where an unquoted keyword and its quoted spelling must be
/// told apart: unquoted `PUBLIC` is the pseudo-role every session belongs to, but quoted
/// `"PUBLIC"` is an ordinary (case-sensitive) role identifier, not the pseudo-role.
fn peek_unquoted_word(parser: &mut Parser) -> Option<String> {
    match &parser.peek_token_ref().token {
        Token::Word(word) if word.quote_style.is_none() => Some(word.value.to_lowercase()),
        _ => None,
    }
}

/// Consume the next token, which the caller has already inspected with [`peek_word`].
fn eat_word(parser: &mut Parser) {
    let _ = parser.next_token();
}

/// An entry in the comma-separated list that follows `GRANT` / `REVOKE`.
///
/// Until the keyword after the list is read, `GRANT a, b TO c` is ambiguous: `a` could be a
/// privilege or a role name. Both readings are kept until `ON` (object grant) or `TO`/`FROM`
/// (role grant) settles it.
enum GrantItem {
    /// A recognized privilege keyword.
    Privilege(ast::Privilege),
    /// A bare identifier, which is a role name if this turns out to be a role grant.
    Name(String),
}

/// Parse one entry of the privilege/role list.
fn parse_grant_item(parser: &mut Parser) -> Result<GrantItem, Error> {
    let Some(word) = peek_word(parser) else {
        return unsupported("GRANT/REVOKE expects a privilege or role name");
    };
    let privilege = match word.as_str() {
        "select" => Some(ast::Privilege::Select),
        "insert" => Some(ast::Privilege::Insert),
        "update" => Some(ast::Privilege::Update),
        "delete" => Some(ast::Privilege::Delete),
        "truncate" => Some(ast::Privilege::Truncate),
        "references" => Some(ast::Privilege::References),
        "trigger" => Some(ast::Privilege::Trigger),
        "usage" => Some(ast::Privilege::Usage),
        "create" => Some(ast::Privilege::Create),
        "connect" => Some(ast::Privilege::Connect),
        "temporary" | "temp" => Some(ast::Privilege::Temporary),
        _ => None,
    };
    if let Some(privilege) = privilege {
        eat_word(parser);
        // A column list (`SELECT (a, b)`) is column-level access control, which NusaDB does not
        // enforce. Accepting and dropping the list would grant more than the statement asked for,
        // so it is refused outright.
        if parser.peek_token_ref().token == Token::LParen {
            return unsupported(
                "column-level privileges (`GRANT SELECT (col) ...`) are not supported; grant on \
                 the whole table instead",
            );
        }
        return Ok(GrantItem::Privilege(privilege));
    }
    let ident = parser.parse_identifier().map_err(|e| syntax(&e))?;
    Ok(GrantItem::Name(fold_ident(&ident)))
}

/// Parse the grantee list of a `GRANT ... TO` / `REVOKE ... FROM`.
///
/// A `GROUP` / `ROLE` / `USER` prefix on an entry is accepted and ignored: the three spell the same
/// thing here, since roles and users are one concept.
fn parse_grantees(parser: &mut Parser) -> Result<Vec<ast::Grantee>, Error> {
    let mut out = Vec::new();
    loop {
        if let Some(word) = peek_word(parser)
            && matches!(word.as_str(), "group" | "role" | "user")
        {
            eat_word(parser);
        }
        if let Some(word) = peek_unquoted_word(parser)
            && word == "public"
        {
            // Unquoted `PUBLIC` is the pseudo-role; quoted `"PUBLIC"` falls through to be read as an
            // ordinary role identifier (which `validate_role_name` then refuses as reserved).
            eat_word(parser);
            out.push(ast::Grantee::Public);
        } else {
            let ident = parser.parse_identifier().map_err(|e| syntax(&e))?;
            out.push(ast::Grantee::Role(fold_ident(&ident)));
        }
        if !parser.consume_token(&Token::Comma) {
            return Ok(out);
        }
    }
}

/// Render a possibly schema-qualified table or sequence name into the canonical form the privilege
/// catalog keys on. An unqualified name is left bare for the analyzer to resolve against the
/// `search_path` — the parser has no session to resolve it with.
fn qualified_name(name: &sql::ObjectName) -> Result<String, Error> {
    let (schema, base) = table_ref_name(name)?;
    Ok(match schema {
        Some(schema) => format!("{schema}.{base}"),
        None => base,
    })
}

/// Parse a comma-separated list of object names.
fn parse_name_list(parser: &mut Parser) -> Result<Vec<String>, Error> {
    let mut out = Vec::new();
    loop {
        let name = parser.parse_object_name(false).map_err(|e| syntax(&e))?;
        out.push(qualified_name(&name)?);
        if !parser.consume_token(&Token::Comma) {
            return Ok(out);
        }
    }
}

/// Parse the `ON <target>` clause shared by `GRANT` and `REVOKE`.
fn parse_targets(parser: &mut Parser) -> Result<ast::GrantTargets, Error> {
    // `ALL TABLES IN SCHEMA s` / `ALL SEQUENCES IN SCHEMA s`.
    if let Some(word) = peek_word(parser)
        && word == "all"
    {
        eat_word(parser);
        let kind = peek_word(parser);
        if !matches!(kind.as_deref(), Some("tables" | "sequences")) {
            return unsupported(
                "GRANT ... ON ALL expects TABLES or SEQUENCES (`ON ALL TABLES IN SCHEMA s`)",
            );
        }
        eat_word(parser);
        let sequences = kind.as_deref() == Some("sequences");
        if !parser.parse_keywords(&[Keyword::IN, Keyword::SCHEMA]) {
            return unsupported("GRANT ... ON ALL TABLES expects `IN SCHEMA <name>`");
        }
        let schemas = parse_name_list(parser)?;
        return Ok(if sequences {
            ast::GrantTargets::AllSequencesInSchemas(schemas)
        } else {
            ast::GrantTargets::AllTablesInSchemas(schemas)
        });
    }

    match peek_word(parser).as_deref() {
        Some("schema") => {
            eat_word(parser);
            Ok(ast::GrantTargets::Schemas(parse_name_list(parser)?))
        },
        Some("database") => {
            eat_word(parser);
            Ok(ast::GrantTargets::Databases(parse_name_list(parser)?))
        },
        Some("sequence") => {
            eat_word(parser);
            Ok(ast::GrantTargets::Sequences(parse_name_list(parser)?))
        },
        // `ON TABLE t` and the bare `ON t` are the same thing; `TABLE` is optional in the standard.
        Some("table") => {
            eat_word(parser);
            Ok(ast::GrantTargets::Tables(parse_name_list(parser)?))
        },
        _ => Ok(ast::GrantTargets::Tables(parse_name_list(parser)?)),
    }
}

/// Split a parsed item list into privileges, refusing a mix of privilege keywords and bare names.
fn items_as_privileges(items: Vec<GrantItem>) -> Result<Vec<ast::Privilege>, Error> {
    items
        .into_iter()
        .map(|item| match item {
            GrantItem::Privilege(p) => Ok(p),
            GrantItem::Name(name) => Err(Error::InvalidStatement(format!(
                "`{name}` is not a privilege; a privilege grant expects SELECT, INSERT, UPDATE, \
                 DELETE, TRUNCATE, REFERENCES, TRIGGER, USAGE, CREATE, CONNECT, or TEMPORARY"
            ))),
        })
        .collect()
}

/// Split a parsed item list into role names, refusing privilege keywords used as role names.
fn items_as_roles(items: Vec<GrantItem>) -> Result<Vec<String>, Error> {
    items
        .into_iter()
        .map(|item| match item {
            GrantItem::Name(name) => Ok(name),
            GrantItem::Privilege(p) => Err(Error::InvalidStatement(format!(
                "`{}` is a privilege keyword, not a role name — a privilege grant needs an `ON` \
                 clause naming what it applies to",
                p.as_str()
            ))),
        })
        .collect()
}

/// Recognize `GRANT ...`; `None` when the statement is something else.
pub(super) fn recognize_grant(sql: &str) -> Option<Result<ast::Statement, Error>> {
    sql.split_whitespace()
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("grant"))
        .then(|| parse_grant(sql))
}

/// Drive `GRANT <privileges> ON <targets> TO <grantees> [WITH GRANT OPTION]` and the role form
/// `GRANT <roles> TO <members> [WITH ADMIN OPTION]`.
fn parse_grant(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    parser
        .expect_keyword(Keyword::GRANT)
        .map_err(|e| syntax(&e))?;

    // `ALL [PRIVILEGES]` is only meaningful for an object grant, and is never a role name.
    let mut all_privileges = false;
    let mut items = Vec::new();
    if peek_word(&mut parser).as_deref() == Some("all") {
        eat_word(&mut parser);
        let _ = parser.parse_keyword(Keyword::PRIVILEGES);
        all_privileges = true;
    } else {
        loop {
            items.push(parse_grant_item(&mut parser)?);
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
    }

    if parser.parse_keyword(Keyword::ON) {
        let targets = parse_targets(&mut parser)?;
        parser.expect_keyword(Keyword::TO).map_err(|e| syntax(&e))?;
        let grantees = parse_grantees(&mut parser)?;
        let with_grant_option =
            parser.parse_keywords(&[Keyword::WITH, Keyword::GRANT, Keyword::OPTION]);
        reject_granted_by(&mut parser)?;
        expect_statement_end(&mut parser)?;
        return Ok(ast::Statement::Grant(ast::Grant {
            privileges: if all_privileges {
                None
            } else {
                Some(items_as_privileges(items)?)
            },
            targets,
            grantees,
            with_grant_option,
        }));
    }

    if all_privileges {
        return unsupported("GRANT ALL PRIVILEGES requires an `ON` clause naming the objects");
    }
    // No `ON`: this is a role grant.
    parser.expect_keyword(Keyword::TO).map_err(|e| syntax(&e))?;
    let roles = items_as_roles(items)?;
    let members = parse_grantees(&mut parser)?
        .into_iter()
        .map(|g| match g {
            ast::Grantee::Role(name) => Ok(name),
            ast::Grantee::Public => Err(Error::InvalidStatement(
                "membership in a role cannot be granted to PUBLIC".to_owned(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let with_admin_option =
        parser.parse_keywords(&[Keyword::WITH, Keyword::ADMIN, Keyword::OPTION]);
    reject_granted_by(&mut parser)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::GrantRole(ast::GrantRole {
        roles,
        members,
        with_admin_option,
    }))
}

/// Refuse `GRANTED BY <role>`. Honouring it means recording a grantor other than the session's own
/// role, which changes who `REVOKE ... CASCADE` can later reach; accepting it while recording the
/// session role instead would silently rewrite the audit trail.
fn reject_granted_by(parser: &mut Parser) -> Result<(), Error> {
    if parser.parse_keywords(&[Keyword::GRANTED, Keyword::BY]) {
        return unsupported(
            "GRANTED BY is not supported (the grantor is always the role running the statement)",
        );
    }
    Ok(())
}

/// Recognize `REVOKE ...`; `None` when the statement is something else.
pub(super) fn recognize_revoke(sql: &str) -> Option<Result<ast::Statement, Error>> {
    sql.split_whitespace()
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("revoke"))
        .then(|| parse_revoke(sql))
}

/// Drive `REVOKE [GRANT OPTION FOR] <privileges> ON <targets> FROM <grantees> [CASCADE|RESTRICT]`
/// and the role form `REVOKE [ADMIN OPTION FOR] <roles> FROM <members>`.
fn parse_revoke(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    parser
        .expect_keyword(Keyword::REVOKE)
        .map_err(|e| syntax(&e))?;

    let grant_option_for = parser.parse_keywords(&[Keyword::GRANT, Keyword::OPTION, Keyword::FOR]);
    let admin_option_for = !grant_option_for
        && parser.parse_keywords(&[Keyword::ADMIN, Keyword::OPTION, Keyword::FOR]);

    let mut all_privileges = false;
    let mut items = Vec::new();
    if peek_word(&mut parser).as_deref() == Some("all") {
        eat_word(&mut parser);
        let _ = parser.parse_keyword(Keyword::PRIVILEGES);
        all_privileges = true;
    } else {
        loop {
            items.push(parse_grant_item(&mut parser)?);
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
    }

    if parser.parse_keyword(Keyword::ON) {
        if admin_option_for {
            return unsupported(
                "ADMIN OPTION FOR applies to a role revoke, which has no `ON` clause",
            );
        }
        let targets = parse_targets(&mut parser)?;
        parser
            .expect_keyword(Keyword::FROM)
            .map_err(|e| syntax(&e))?;
        let grantees = parse_grantees(&mut parser)?;
        let cascade = parse_drop_behavior(&mut parser);
        reject_granted_by(&mut parser)?;
        expect_statement_end(&mut parser)?;
        return Ok(ast::Statement::Revoke(ast::Revoke {
            privileges: if all_privileges {
                None
            } else {
                Some(items_as_privileges(items)?)
            },
            targets,
            grantees,
            grant_option_for,
            cascade,
        }));
    }

    if all_privileges {
        return unsupported("REVOKE ALL PRIVILEGES requires an `ON` clause naming the objects");
    }
    if grant_option_for {
        return unsupported(
            "GRANT OPTION FOR applies to a privilege revoke, which needs an `ON` clause",
        );
    }
    parser
        .expect_keyword(Keyword::FROM)
        .map_err(|e| syntax(&e))?;
    let roles = items_as_roles(items)?;
    let members = parse_grantees(&mut parser)?
        .into_iter()
        .map(|g| match g {
            ast::Grantee::Role(name) => Ok(name),
            ast::Grantee::Public => Err(Error::InvalidStatement(
                "membership in a role cannot be revoked from PUBLIC".to_owned(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    // `CASCADE`/`RESTRICT` parse on a role revoke for grammar compatibility; membership has no
    // dependent grants to cascade to, so the choice does not change the outcome.
    let _ = parse_drop_behavior(&mut parser);
    reject_granted_by(&mut parser)?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::RevokeRole(ast::RevokeRole {
        roles,
        members,
        admin_option_for,
    }))
}

/// Parse the trailing `CASCADE` / `RESTRICT`, returning whether `CASCADE` was given. `RESTRICT` is
/// the standard default.
fn parse_drop_behavior(parser: &mut Parser) -> bool {
    if parser.parse_keyword(Keyword::CASCADE) {
        return true;
    }
    let _ = parser.parse_keyword(Keyword::RESTRICT);
    false
}

/// Parse the role attribute list shared by `CREATE ROLE` and `ALTER ROLE`.
///
/// `PASSWORD` is refused rather than accepted-and-ignored: connection credentials live in the
/// server's authentication store, so a password accepted here would never be checked at login —
/// an advertised control that does nothing is worse than one that is absent.
fn parse_role_options(parser: &mut Parser) -> Result<(ast::RoleOptions, Vec<String>), Error> {
    let mut options = ast::RoleOptions::default();
    let mut in_role = Vec::new();
    // `WITH` is noise in this position; the standard allows it before the attribute list.
    let _ = parser.parse_keyword(Keyword::WITH);
    while let Some(word) = peek_word(parser) {
        match word.as_str() {
            "superuser" => {
                eat_word(parser);
                options.superuser = Some(true);
            },
            "nosuperuser" => {
                eat_word(parser);
                options.superuser = Some(false);
            },
            "login" => {
                eat_word(parser);
                options.login = Some(true);
            },
            "nologin" => {
                eat_word(parser);
                options.login = Some(false);
            },
            "createdb" => {
                eat_word(parser);
                options.create_db = Some(true);
            },
            "nocreatedb" => {
                eat_word(parser);
                options.create_db = Some(false);
            },
            "createrole" => {
                eat_word(parser);
                options.create_role = Some(true);
            },
            "nocreaterole" => {
                eat_word(parser);
                options.create_role = Some(false);
            },
            "inherit" => {
                eat_word(parser);
                options.inherit = Some(true);
            },
            "noinherit" => {
                eat_word(parser);
                options.inherit = Some(false);
            },
            "in" => {
                eat_word(parser);
                if !parser.parse_keyword(Keyword::ROLE) && !parser.parse_keyword(Keyword::GROUP) {
                    return unsupported("CREATE ROLE ... IN expects ROLE or GROUP");
                }
                loop {
                    let ident = parser.parse_identifier().map_err(|e| syntax(&e))?;
                    in_role.push(fold_ident(&ident));
                    if !parser.consume_token(&Token::Comma) {
                        break;
                    }
                }
            },
            "password" | "encrypted" | "unencrypted" => {
                return unsupported(
                    "PASSWORD is not settable through SQL — connection credentials come from the \
                     server's authentication store (`--auth-user`), so a password set here would \
                     never be checked at login",
                );
            },
            "valid" | "connection" | "replication" | "noreplication" | "bypassrls"
            | "nobypassrls" | "sysid" | "admin" | "role" | "user" => {
                return unsupported(&format!(
                    "role attribute `{word}` is not supported; supported attributes are \
                     SUPERUSER/NOSUPERUSER, LOGIN/NOLOGIN, CREATEDB/NOCREATEDB, \
                     CREATEROLE/NOCREATEROLE, INHERIT/NOINHERIT, and IN ROLE"
                ));
            },
            _ => break,
        }
    }
    Ok((options, in_role))
}

/// Recognize `CREATE ROLE ...` / `CREATE USER ...`.
pub(super) fn recognize_create_role(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_any(sql, "create", &["role", "user"]).then(|| parse_create_role(sql))
}

/// Drive `CREATE {ROLE | USER} [IF NOT EXISTS] name [attributes]`.
///
/// `CREATE USER` differs from `CREATE ROLE` in exactly one way: `LOGIN` defaults on.
fn parse_create_role(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    parser
        .expect_keyword(Keyword::CREATE)
        .map_err(|e| syntax(&e))?;
    let is_user = matches!(peek_word(&mut parser).as_deref(), Some("user"));
    eat_word(&mut parser);
    let if_not_exists = parser.parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]);
    let name = fold_ident(&parser.parse_identifier().map_err(|e| syntax(&e))?);
    let (mut options, in_role) = parse_role_options(&mut parser)?;
    expect_statement_end(&mut parser)?;
    if is_user && options.login.is_none() {
        options.login = Some(true);
    }
    Ok(ast::Statement::CreateRole(ast::CreateRole {
        name,
        if_not_exists,
        options,
        in_role,
    }))
}

/// Recognize `DROP ROLE ...` / `DROP USER ...`.
pub(super) fn recognize_drop_role(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_any(sql, "drop", &["role", "user"]).then(|| parse_drop_role(sql))
}

/// Drive `DROP {ROLE | USER} [IF EXISTS] name[, ...]`.
fn parse_drop_role(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    parser
        .expect_keyword(Keyword::DROP)
        .map_err(|e| syntax(&e))?;
    eat_word(&mut parser);
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    let mut names = Vec::new();
    loop {
        names.push(fold_ident(
            &parser.parse_identifier().map_err(|e| syntax(&e))?,
        ));
        if !parser.consume_token(&Token::Comma) {
            break;
        }
    }
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::DropRole(ast::DropRole { names, if_exists }))
}

/// Recognize `ALTER ROLE ...` / `ALTER USER ...`.
pub(super) fn recognize_alter_role(sql: &str) -> Option<Result<ast::Statement, Error>> {
    starts_with_any(sql, "alter", &["role", "user"]).then(|| parse_alter_role(sql))
}

/// Drive `ALTER {ROLE | USER} name [attributes]`.
fn parse_alter_role(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    parser
        .expect_keyword(Keyword::ALTER)
        .map_err(|e| syntax(&e))?;
    eat_word(&mut parser);
    let name = fold_ident(&parser.parse_identifier().map_err(|e| syntax(&e))?);
    let (options, in_role) = parse_role_options(&mut parser)?;
    if !in_role.is_empty() {
        return unsupported(
            "ALTER ROLE ... IN ROLE is not supported; use `GRANT <role> TO <member>` to change \
             membership",
        );
    }
    if options == ast::RoleOptions::default() {
        return unsupported("ALTER ROLE requires at least one attribute to change");
    }
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::AlterRole(ast::AlterRole { name, options }))
}

/// Recognize `SET ROLE ...` and `RESET ROLE`.
///
/// This must run before the generic `SET` / `RESET` handling, which would otherwise treat `role`
/// as a session variable name and store it as one — leaving the session's actual privileges
/// unchanged while reporting success.
pub(super) fn recognize_set_role(sql: &str) -> Option<Result<ast::Statement, Error>> {
    (starts_with_any(sql, "set", &["role"]) || starts_with_any(sql, "reset", &["role"]))
        .then(|| parse_set_role(sql))
}

/// Drive `SET ROLE {name | NONE}` and `RESET ROLE`.
fn parse_set_role(sql: &str) -> Result<ast::Statement, Error> {
    let mut parser = parser_for(sql)?;
    let reset = matches!(peek_word(&mut parser).as_deref(), Some("reset"));
    eat_word(&mut parser);
    eat_word(&mut parser);
    if reset {
        expect_statement_end(&mut parser)?;
        return Ok(ast::Statement::SetRole(ast::SetRole { role: None }));
    }
    let role = match peek_word(&mut parser).as_deref() {
        Some("none") => {
            eat_word(&mut parser);
            None
        },
        _ => Some(fold_ident(
            &parser.parse_identifier().map_err(|e| syntax(&e))?,
        )),
    };
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::SetRole(ast::SetRole { role }))
}
