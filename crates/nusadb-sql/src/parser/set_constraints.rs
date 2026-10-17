//! The two statements about when a constraint is checked, neither of which the generic grammar
//! has:
//!
//! - `SET CONSTRAINTS {ALL | name [, ...]} {DEFERRED | IMMEDIATE}`, and
//! - `ALTER TABLE [IF EXISTS] [ONLY] t ALTER CONSTRAINT c [NOT] DEFERRABLE
//!   [INITIALLY {DEFERRED | IMMEDIATE}]`.
//!
//! Both are driven over sqlparser's own tokenizer and primitives, so quoting, qualified names and
//! comments behave exactly as in every other statement.

use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use super::{NusaParserDialect, expect_statement_end, fold_ident, fold_part, table_ref_name};
use crate::ast;
use crate::error::Error;

#[allow(
    clippy::needless_pass_by_value,
    reason = "the shape `map_err` hands over: the error by value"
)]
fn syntax(e: ParserError) -> Error {
    Error::Syntax(e.to_string())
}

/// Whether `sql` starts with `words` (case-insensitive, whitespace-separated).
fn leads_with(sql: &str, words: &[&str]) -> bool {
    let mut it = sql.split_whitespace();
    words
        .iter()
        .all(|w| it.next().is_some_and(|got| got.eq_ignore_ascii_case(w)))
}

/// Recognize `SET CONSTRAINTS ...`; `None` for any other statement.
pub(super) fn recognize_set_constraints(sql: &str) -> Option<Result<ast::Statement, Error>> {
    leads_with(sql, &["set", "constraints"]).then(|| parse_set_constraints(sql))
}

fn parse_set_constraints(sql: &str) -> Result<ast::Statement, Error> {
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::SET).map_err(syntax)?;
    // `CONSTRAINTS` is not a keyword of the generic grammar: match the word.
    match parser.next_token().token {
        Token::Word(w) if w.value.eq_ignore_ascii_case("constraints") => {},
        other => {
            return Err(Error::Syntax(format!(
                "expected CONSTRAINTS, found {other}"
            )));
        },
    }
    let names = if parser.parse_keyword(Keyword::ALL) {
        None
    } else {
        let mut names = Vec::new();
        loop {
            let name = parser.parse_object_name(false).map_err(syntax)?;
            let pair = match name.0.as_slice() {
                [only] => (None, fold_part(only)?),
                [schema, only] => (Some(fold_part(schema)?), fold_part(only)?),
                _ => {
                    return Err(Error::Syntax(
                        "a constraint name has at most two parts (schema.name)".to_owned(),
                    ));
                },
            };
            names.push(pair);
            if !parser.consume_token(&Token::Comma) {
                break;
            }
        }
        Some(names)
    };
    let deferred = if parser.parse_keyword(Keyword::DEFERRED) {
        true
    } else if parser.parse_keyword(Keyword::IMMEDIATE) {
        false
    } else {
        return Err(Error::Syntax(
            "SET CONSTRAINTS needs DEFERRED or IMMEDIATE".to_owned(),
        ));
    };
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::SetConstraints(ast::SetConstraints {
        names,
        deferred,
    }))
}

/// Recognize `ALTER TABLE ... ALTER CONSTRAINT ...`; `None` for any other statement, including
/// every other `ALTER TABLE` form (`ALTER COLUMN` among them).
pub(super) fn recognize_alter_constraint(sql: &str) -> Option<Result<ast::Statement, Error>> {
    fn is_word(t: &Token, w: &str) -> bool {
        matches!(t, Token::Word(x) if x.quote_style.is_none() && x.value.eq_ignore_ascii_case(w))
    }
    if !leads_with(sql, &["alter", "table"]) {
        return None;
    }
    // Only this form: `... ALTER CONSTRAINT <name> ...` somewhere after the table name. A cheap
    // token scan keeps every other ALTER TABLE on the generic path.
    let dialect = NusaParserDialect;
    let tokens = sqlparser::tokenizer::Tokenizer::new(&dialect, sql)
        .tokenize()
        .ok()?;
    let words: Vec<&Token> = tokens
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace(_)))
        .collect();
    let found = words
        .windows(2)
        .skip(2)
        .any(|pair| matches!(pair, [a, b] if is_word(a, "alter") && is_word(b, "constraint")));
    found.then(|| parse_alter_constraint(sql))
}

fn parse_alter_constraint(sql: &str) -> Result<ast::Statement, Error> {
    let dialect = NusaParserDialect;
    let mut parser = Parser::new(&dialect).try_with_sql(sql).map_err(syntax)?;
    parser.expect_keyword(Keyword::ALTER).map_err(syntax)?;
    parser.expect_keyword(Keyword::TABLE).map_err(syntax)?;
    let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
    // `ONLY` changes nothing here: the constraint is changed where it is declared, and a
    // partition's copy follows its parent's.
    let _only = parser.parse_keyword(Keyword::ONLY);
    let table = parser.parse_object_name(false).map_err(syntax)?;
    let (schema, name) = table_ref_name(&table)?;
    parser.expect_keyword(Keyword::ALTER).map_err(syntax)?;
    parser.expect_keyword(Keyword::CONSTRAINT).map_err(syntax)?;
    let constraint = fold_ident(&parser.parse_identifier().map_err(syntax)?);
    let characteristics = parser.parse_constraint_characteristics().map_err(syntax)?;
    if characteristics.is_none() {
        return Err(Error::Syntax(
            "ALTER CONSTRAINT needs [NOT] DEFERRABLE or INITIALLY DEFERRED / IMMEDIATE".to_owned(),
        ));
    }
    let deferral = super::ddl::convert_characteristics(characteristics.as_ref())?;
    expect_statement_end(&mut parser)?;
    Ok(ast::Statement::AlterTable(ast::AlterTable {
        schema,
        name,
        if_exists,
        action: ast::AlterTableAction::AlterConstraint {
            name: constraint,
            deferral,
        },
    }))
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use crate::parser::parse;

    fn set(sql: &str) -> ast::SetConstraints {
        match parse(sql).unwrap() {
            ast::Statement::SetConstraints(s) => s,
            other => panic!("expected SET CONSTRAINTS, got {other:?}"),
        }
    }

    #[test]
    fn set_constraints_forms() {
        assert_eq!(
            set("SET CONSTRAINTS ALL DEFERRED"),
            ast::SetConstraints {
                names: None,
                deferred: true
            }
        );
        assert_eq!(
            set("set constraints a, app.\"B\" immediate;"),
            ast::SetConstraints {
                names: Some(vec![
                    (None, "a".to_owned()),
                    (Some("app".to_owned()), "B".to_owned())
                ]),
                deferred: false
            }
        );
        assert!(parse("SET CONSTRAINTS ALL").is_err());
        assert!(parse("SET CONSTRAINTS ALL DEFERRED extra").is_err());
        assert!(parse("SET CONSTRAINTS a.b.c DEFERRED").is_err());
    }

    #[test]
    fn alter_constraint_forms() {
        let ast::Statement::AlterTable(at) =
            parse("ALTER TABLE app.t ALTER CONSTRAINT t_fk DEFERRABLE INITIALLY DEFERRED").unwrap()
        else {
            panic!("expected ALTER TABLE");
        };
        assert_eq!(at.schema.as_deref(), Some("app"));
        assert_eq!(at.name, "t");
        assert_eq!(
            at.action,
            ast::AlterTableAction::AlterConstraint {
                name: "t_fk".to_owned(),
                deferral: ast::Deferral {
                    deferrable: true,
                    initially_deferred: true
                }
            }
        );
        let ast::Statement::AlterTable(at) =
            parse("ALTER TABLE t ALTER CONSTRAINT c NOT DEFERRABLE").unwrap()
        else {
            panic!("expected ALTER TABLE");
        };
        assert_eq!(
            at.action,
            ast::AlterTableAction::AlterConstraint {
                name: "c".to_owned(),
                deferral: ast::Deferral::IMMEDIATE
            }
        );
        assert!(parse("ALTER TABLE t ALTER CONSTRAINT c").is_err());
        assert!(matches!(
            parse("ALTER TABLE t ALTER CONSTRAINT c NOT DEFERRABLE INITIALLY DEFERRED"),
            Err(crate::Error::Syntax(_))
        ));
        // Every other ALTER TABLE form is untouched.
        assert!(matches!(
            parse("ALTER TABLE t ALTER COLUMN constraint_x SET NOT NULL").unwrap(),
            ast::Statement::AlterTable(ast::AlterTable {
                action: ast::AlterTableAction::AlterColumn { .. },
                ..
            })
        ));
    }

    fn constraints(sql: &str) -> (Vec<ast::ColumnDef>, Vec<ast::TableConstraint>) {
        match parse(sql).unwrap() {
            // The synthetic type-range checks of the integer columns are left out.
            ast::Statement::CreateTable(ct) => (
                ct.columns,
                ct.constraints
                    .into_iter()
                    .filter(|c| !matches!(c, ast::TableConstraint::Check { .. }))
                    .collect(),
            ),
            other => panic!("expected CREATE TABLE, got {other:?}"),
        }
    }

    const DEFERRED: ast::Deferral = ast::Deferral {
        deferrable: true,
        initially_deferred: true,
    };
    const DEFERRABLE: ast::Deferral = ast::Deferral {
        deferrable: true,
        initially_deferred: false,
    };

    #[test]
    fn column_level_keys_keep_their_timing() {
        // An unnamed deferrable column key is lifted to a table constraint, keeping the default
        // name rule; an immediate one stays a column flag.
        let (columns, cs) = constraints(
            "CREATE TABLE t (a INT UNIQUE DEFERRABLE, b INT PRIMARY KEY INITIALLY DEFERRED, \
             c INT UNIQUE NOT DEFERRABLE, d INT REFERENCES p DEFERRABLE INITIALLY DEFERRED)",
        );
        let Some(ast::TableConstraint::Unique {
            name: None,
            deferral,
            ..
        }) = cs.first()
        else {
            panic!("expected a UNIQUE constraint: {cs:?}");
        };
        assert_eq!(*deferral, DEFERRABLE);
        let Some(ast::TableConstraint::PrimaryKey {
            name: None,
            deferral,
            ..
        }) = cs.get(1)
        else {
            panic!("expected a PRIMARY KEY constraint: {cs:?}");
        };
        assert_eq!(*deferral, DEFERRED);
        assert!(matches!(
            cs.get(2),
            Some(ast::TableConstraint::ForeignKey { deferral, .. }) if *deferral == DEFERRED
        ));
        assert_eq!(cs.len(), 3);
        // The primary key column is still NOT NULL; `c` stays a plain column flag.
        assert!(columns.iter().any(|c| c.name == "b" && !c.nullable));
        assert!(columns.iter().any(|c| c.name == "c" && c.unique));
        assert!(columns.iter().any(|c| c.name == "a" && !c.unique));
    }

    #[test]
    fn exclusion_constraints_carry_their_timing() {
        let (_, cs) = constraints(
            "CREATE TABLE t (k INT, r int4range, \
             CONSTRAINT x EXCLUDE USING gist (k WITH =, r WITH &&) DEFERRABLE INITIALLY DEFERRED, \
             EXCLUDE (k WITH =) INITIALLY DEFERRED, EXCLUDE (r WITH &&) NOT DEFERRABLE)",
        );
        let Some(ast::TableConstraint::Exclusion { decl, deferral, .. }) = cs.first() else {
            panic!("expected an exclusion constraint: {cs:?}");
        };
        assert_eq!(*deferral, DEFERRED);
        // The timing never reaches the stored form.
        assert_eq!(decl.method, "gist");
        assert!(!decl.spec_sql.contains("defer"), "{}", decl.spec_sql);
        // The plain `=` form is a UNIQUE constraint, with the timing.
        assert!(matches!(
            cs.get(1),
            Some(ast::TableConstraint::Unique { deferral, .. }) if *deferral == DEFERRED
        ));
        let Some(ast::TableConstraint::Exclusion { deferral, .. }) = cs.get(2) else {
            panic!("expected an exclusion constraint: {cs:?}");
        };
        assert_eq!(*deferral, ast::Deferral::IMMEDIATE);
    }

    #[test]
    fn check_and_not_null_are_never_deferrable() {
        for sql in [
            "CREATE TABLE t (a INT CHECK (a > 0) DEFERRABLE)",
            "CREATE TABLE t (a INT, CONSTRAINT c CHECK (a > 0) INITIALLY DEFERRED)",
            "CREATE TABLE t (a INT NOT NULL DEFERRABLE INITIALLY DEFERRED)",
            "ALTER TABLE t ADD CONSTRAINT c CHECK (a > 0) DEFERRABLE",
        ] {
            let err = parse(sql).unwrap_err();
            assert_eq!(err.sqlstate(), "0A000", "{sql}: {err}");
            assert!(
                err.to_string().contains("cannot be marked DEFERRABLE"),
                "{err}"
            );
        }
        // Saying what is true anyway is accepted.
        assert!(
            parse(
                "CREATE TABLE t (a INT NOT NULL NOT DEFERRABLE CHECK (a > 0) INITIALLY IMMEDIATE)"
            )
            .is_ok()
        );
        assert_eq!(
            parse("CREATE TABLE t (a INT UNIQUE NOT DEFERRABLE INITIALLY DEFERRED)")
                .unwrap_err()
                .sqlstate(),
            "42601"
        );
        assert_eq!(
            parse("CREATE TABLE t (a INT, UNIQUE (a) NOT ENFORCED)")
                .unwrap_err()
                .sqlstate(),
            "0A000"
        );
        assert!(parse("CREATE TABLE t (a INT, UNIQUE (a) ENFORCED)").is_ok());
    }
}
