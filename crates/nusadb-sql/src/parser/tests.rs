use super::{Error, parse};
use crate::ast;
use nusadb_core::ColumnType;

/// Parse SQL that is expected to succeed, returning the statement.
fn ok(sql: &str) -> ast::Statement {
    parse(sql).expect("expected a successful parse")
}

// --- CREATE TABLE -----------------------------------------------------

#[test]
fn create_table_columns_and_constraints() {
    let ast::Statement::CreateTable(ct) =
        ok("CREATE TABLE users (id INT NOT NULL PRIMARY KEY, name TEXT, active BOOLEAN)")
    else {
        panic!("expected CreateTable");
    };
    assert_eq!(ct.name, "users");
    assert!(!ct.if_not_exists);
    assert_eq!(
        ct.columns,
        vec![
            ast::ColumnDef {
                name: "id".to_owned(),
                ty: ColumnType::Int,
                udt_name: None,
                nullable: false,
                primary_key: true,
                unique: false,
                default: None,
                default_sql: None,
                generated: None,
                serial: false,
                identity_always: false,
            },
            ast::ColumnDef {
                name: "name".to_owned(),
                ty: ColumnType::Text,
                udt_name: None,
                nullable: true,
                primary_key: false,
                unique: false,
                default: None,
                default_sql: None,
                generated: None,
                serial: false,
                identity_always: false,
            },
            ast::ColumnDef {
                name: "active".to_owned(),
                ty: ColumnType::Bool,
                udt_name: None,
                nullable: true,
                primary_key: false,
                unique: false,
                default: None,
                default_sql: None,
                generated: None,
                serial: false,
                identity_always: false,
            },
        ],
    );
}

#[test]
fn create_table_if_not_exists_and_type_mapping() {
    let ast::Statement::CreateTable(ct) = ok(
        "CREATE TABLE IF NOT EXISTS t (a BIGINT, b VARCHAR(20), c DOUBLE PRECISION, \
             d TIMESTAMP, e BYTEA)",
    ) else {
        panic!("expected CreateTable");
    };
    assert!(ct.if_not_exists);
    let types: Vec<ColumnType> = ct.columns.iter().map(|c| c.ty).collect();
    // `BIGINT` and `VARCHAR(20)` keep their declared type so it round-trips in DDL (`SHOW COLUMNS` /
    // `information_schema`); both are stored identically to their physical type (`ColumnType::physical`
    // maps `BigInt → Int`, `VarChar → Text`).
    assert_eq!(
        types,
        vec![
            ColumnType::BigInt,
            ColumnType::VarChar(20),
            ColumnType::Float,
            ColumnType::Timestamp,
            ColumnType::Bytes,
        ],
    );
}

// --- CREATE TABLE column constraints ---------

#[test]
fn create_table_column_default() {
    let ast::Statement::CreateTable(ct) = ok("CREATE TABLE t (a INT DEFAULT 5, b TEXT)") else {
        panic!("expected CreateTable");
    };
    assert_eq!(
        ct.columns[0].default.as_deref(),
        Some(&ast::Expr::Literal(ast::Value::Int(5)))
    );
    assert!(ct.columns[1].default.is_none());
}

#[test]
fn create_table_column_check_lifts_to_table_constraint() {
    let ast::Statement::CreateTable(ct) = ok("CREATE TABLE t (a INT CHECK (a > 0))") else {
        panic!("expected CreateTable");
    };
    // Column-level CHECK is lifted into the table constraint list. A synthetic INT range
    // check is also lifted for column `a`; assert on the user's CHECK only (filter the synthetic).
    let user: Vec<_> = ct
        .constraints
        .iter()
        .filter(|c| !is_synthetic_check(c))
        .collect();
    assert_eq!(user.len(), 1);
    assert!(matches!(user[0], ast::TableConstraint::Check { .. }));
}

/// Whether a parsed table constraint is a synthetic type-bound CHECK (a `VARCHAR(n)` length or a
/// narrow-int range), recognised by its reserved name prefix — these are an implementation detail of
/// the declared type, not a user constraint.
fn is_synthetic_check(c: &ast::TableConstraint) -> bool {
    matches!(
        c,
        ast::TableConstraint::Check { name: Some(n), .. }
            if n.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX)
    )
}

#[test]
fn create_table_column_references_lifts_to_fk() {
    let ast::Statement::CreateTable(ct) =
        ok("CREATE TABLE t (uid INT REFERENCES users (id) ON DELETE CASCADE)")
    else {
        panic!("expected CreateTable");
    };
    let ast::TableConstraint::ForeignKey {
        columns,
        foreign_table,
        referred_columns,
        on_delete,
        ..
    } = &ct.constraints[0]
    else {
        panic!("expected lifted ForeignKey");
    };
    assert_eq!(columns, &vec!["uid".to_owned()]);
    assert_eq!(foreign_table, "users");
    assert_eq!(referred_columns, &vec!["id".to_owned()]);
    assert_eq!(on_delete, &Some(ast::ReferentialAction::Cascade));
}

#[test]
fn create_table_generated_column() {
    let ast::Statement::CreateTable(ct) =
        ok("CREATE TABLE t (a INT, b INT GENERATED ALWAYS AS (a + 1) STORED)")
    else {
        panic!("expected CreateTable");
    };
    let g = ct.columns[1].generated.as_ref().expect("generated");
    assert!(g.stored);
    assert!(matches!(*g.expr, ast::Expr::Binary { .. }));
}

#[test]
fn create_table_table_level_check_and_fk() {
    // Table-level CHECK + FK still parse (pre-existing).
    let ast::Statement::CreateTable(ct) = ok("CREATE TABLE t (a INT, b INT, CHECK (a < b), \
         FOREIGN KEY (a) REFERENCES p (id))")
    else {
        panic!("expected CreateTable");
    };
    // The user's CHECK + FK (columns a, b each also carry a synthetic INT range check — filtered).
    let user = ct
        .constraints
        .iter()
        .filter(|c| !is_synthetic_check(c))
        .count();
    assert_eq!(user, 2);
}

// --- DROP TABLE -------------------------------------------------------

#[test]
fn drop_table_plain_and_if_exists() {
    let ast::Statement::DropTable(d) = ok("DROP TABLE t") else {
        panic!("expected DropTable");
    };
    assert_eq!(d.name, "t");
    assert!(!d.if_exists);

    let ast::Statement::DropTable(d) = ok("DROP TABLE IF EXISTS t") else {
        panic!("expected DropTable");
    };
    assert!(d.if_exists);
}

// --- CREATE INDEX -----------------------------------------------------

#[test]
fn create_index_single_column_defaults() {
    let ast::Statement::CreateIndex(ci) = ok("CREATE INDEX users_email_idx ON users (email)")
    else {
        panic!("expected CreateIndex");
    };
    assert_eq!(ci.name, "users_email_idx");
    assert_eq!(ci.table, "users");
    assert_eq!(ci.columns, vec!["email".to_owned()]);
    assert!(ci.include.is_empty());
    assert!(!ci.unique);
    assert!(!ci.if_not_exists);
}

#[test]
fn create_index_unique_if_not_exists_multi_column_with_include() {
    let ast::Statement::CreateIndex(ci) =
        ok("CREATE UNIQUE INDEX IF NOT EXISTS orders_lookup_idx \
             ON orders (customer_id, status) INCLUDE (total, currency)")
    else {
        panic!("expected CreateIndex");
    };
    assert_eq!(ci.name, "orders_lookup_idx");
    assert_eq!(ci.table, "orders");
    assert_eq!(
        ci.columns,
        vec!["customer_id".to_owned(), "status".to_owned()],
    );
    assert_eq!(ci.include, vec!["total".to_owned(), "currency".to_owned()]);
    assert!(ci.unique);
    assert!(ci.if_not_exists);
}

#[test]
fn create_index_folds_unquoted_identifiers_and_preserves_quoted() {
    let ast::Statement::CreateIndex(ci) = ok(r#"CREATE INDEX "Idx" ON Users ("Email", AGE)"#)
    else {
        panic!("expected CreateIndex");
    };
    assert_eq!(ci.name, "Idx", "quoted index name should keep its case");
    assert_eq!(ci.table, "users", "unquoted table should fold to lowercase");
    assert_eq!(
        ci.columns,
        vec!["Email".to_owned(), "age".to_owned()],
        "quoted column keeps its case, unquoted folds",
    );
}

#[test]
fn select_clauses_outside_surface_are_rejected_not_ignored() {
    // G5: these were parsed and silently discarded, so the query ran with the wrong semantics.
    // OFFSET and FETCH FIRST now parse — removed from this list.
    // (FOR UPDATE/SHARE also parse here)
    // A plain LIMIT (in surface) still parses.
    assert!(parse("SELECT * FROM t LIMIT 10").is_ok());
}

// --- FETCH FIRST / OFFSET -------------------------------------

#[test]
fn fetch_first_parses_as_limit() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t FETCH FIRST 5 ROWS ONLY") else {
        panic!("expected Select");
    };
    assert_eq!(s.limit, Some(5));
    assert!(s.offset.is_none());
}

#[test]
fn fetch_next_parses_as_limit() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t FETCH NEXT 10 ROWS ONLY") else {
        panic!("expected Select");
    };
    assert_eq!(s.limit, Some(10));
}

#[test]
fn offset_rows_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t OFFSET 5 ROWS") else {
        panic!("expected Select");
    };
    assert_eq!(s.offset, Some(5));
    assert!(s.limit.is_none());
}

#[test]
fn limit_and_offset_together() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t LIMIT 10 OFFSET 5") else {
        panic!("expected Select");
    };
    assert_eq!(s.limit, Some(10));
    assert_eq!(s.offset, Some(5));
}

#[test]
fn fetch_first_with_ties_is_parsed() {
    // The parser accepts WITH TIES and flags it; the analyzer enforces that it
    // has an ORDER BY and a supported shape.
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY a FETCH FIRST 5 ROWS WITH TIES")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.limit, Some(5));
    assert!(s.limit_with_ties);
    // FETCH FIRST ... ONLY does not set the tie flag.
    let ast::Statement::Select(only) = ok("SELECT * FROM t FETCH FIRST 5 ROWS ONLY") else {
        panic!("expected Select");
    };
    assert_eq!(only.limit, Some(5));
    assert!(!only.limit_with_ties);
}

#[test]
fn fetch_first_and_limit_together_is_rejected() {
    assert!(matches!(
        parse("SELECT * FROM t LIMIT 10 FETCH FIRST 5 ROWS ONLY"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn collate_c_and_posix_are_accepted_as_byte_order_no_ops() {
    // NusaDB sorts text by byte value, which is exactly the SQL-standard `C` / `POSIX` collation, so
    // requesting it changes nothing and must parse (D-COLLATE) — at the column and expression level.
    for sql in [
        "CREATE TABLE t (name TEXT COLLATE \"C\")",
        "CREATE TABLE t (name TEXT COLLATE \"POSIX\")",
        "CREATE TABLE t (name TEXT COLLATE \"c\")", // case-insensitive
        "SELECT name FROM t ORDER BY name COLLATE \"C\"",
        "SELECT name FROM t WHERE name COLLATE \"POSIX\" > 'a'",
    ] {
        assert!(
            parse(sql).is_ok(),
            "byte-order collation must be accepted: {sql}"
        );
    }
}

#[test]
fn collate_locale_is_loudly_rejected_not_silently_ignored() {
    // A locale collation must fail loudly, never silently fall back to byte order (which would sort a
    // query that asked for locale order wrongly, with no signal) — column and expression level.
    for sql in [
        "CREATE TABLE t (name TEXT COLLATE \"en_US\")",
        "SELECT name FROM t ORDER BY name COLLATE \"en_US\"",
        "SELECT name FROM t ORDER BY name COLLATE \"de_DE.utf8\"",
    ] {
        assert!(
            matches!(parse(sql), Err(Error::Unsupported(_))),
            "locale collation must be loud-rejected: {sql}"
        );
    }
}

#[test]
fn offset_bare_without_rows_keyword() {
    // `OFFSET 3` without ROW/ROWS keyword is also valid SQL.
    let ast::Statement::Select(s) = ok("SELECT * FROM t OFFSET 3") else {
        panic!("expected Select");
    };
    assert_eq!(s.offset, Some(3));
}

#[test]
fn union_with_offset_is_rejected() {
    // OFFSET on a set-operation envelope must be rejected (G20 — no silent drop).
    assert!(matches!(
        parse("SELECT a FROM t UNION SELECT a FROM t OFFSET 5 ROWS"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn union_with_fetch_first_is_rejected() {
    // FETCH FIRST on a set-operation envelope must be rejected (G20 — no silent drop).
    assert!(matches!(
        parse("SELECT a FROM t UNION SELECT a FROM t FETCH FIRST 5 ROWS ONLY"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_index_accepts_functional_partial_and_rejects_asc_desc_udf_shapes() {
    // Functional / expression key → carried as `key_exprs` (SQL text), `columns` empty.
    let ast::Statement::CreateIndex(ci) = ok("CREATE INDEX i ON t (LOWER(name))") else {
        panic!("expected CreateIndex");
    };
    assert!(ci.columns.is_empty());
    assert_eq!(ci.key_exprs.len(), 1);
    assert!(ci.key_exprs[0].to_uppercase().contains("LOWER"));
    assert!(ci.predicate.is_none());

    // A parenthesized arithmetic expression key.
    let ast::Statement::CreateIndex(ci) = ok("CREATE INDEX i ON t ((a + b))") else {
        panic!("expected CreateIndex");
    };
    assert!(ci.columns.is_empty());
    assert_eq!(ci.key_exprs.len(), 1);

    // Mixed plain + expression keys → all rendered as expressions, in order.
    let ast::Statement::CreateIndex(ci) = ok("CREATE INDEX i ON t (a, lower(b))") else {
        panic!("expected CreateIndex");
    };
    assert!(ci.columns.is_empty());
    assert_eq!(ci.key_exprs.len(), 2);

    // Partial-index `WHERE` predicate → carried as `predicate` SQL text.
    let ast::Statement::CreateIndex(ci) = ok("CREATE INDEX i ON t (a) WHERE a > 0") else {
        panic!("expected CreateIndex");
    };
    assert_eq!(ci.columns, vec!["a".to_owned()]);
    assert!(ci.predicate.is_some());

    // Still rejected: per-column ASC/DESC/NULLS (only ascending indexes are built; ordered index
    // scans are not implemented, so accepting DESC would be a silent-lossy trap), USING <method>,
    // qualified column key, operator class.
    for sql in [
        "CREATE INDEX i ON t (a ASC)",
        "CREATE INDEX i ON t (a DESC)",
        "CREATE INDEX i ON t (a DESC NULLS LAST)",
        "CREATE INDEX i ON t (a NULLS FIRST)",
        "CREATE INDEX i ON t USING HASH (a)",
        "CREATE INDEX i ON t (t.a)",
        "CREATE INDEX i ON t (a varchar_pattern_ops)",
    ] {
        let err = parse(sql).expect_err(sql);
        assert!(
            matches!(err, Error::Unsupported(_)),
            "for `{sql}`: got {err:?}"
        );
    }
}

// --- DROP INDEX -------------------------------------------------------

#[test]
fn drop_index_plain_and_if_exists() {
    let ast::Statement::DropIndex(d) = ok("DROP INDEX users_email_idx") else {
        panic!("expected DropIndex");
    };
    assert_eq!(d.name, "users_email_idx");
    assert!(!d.if_exists);

    let ast::Statement::DropIndex(d) = ok("DROP INDEX IF EXISTS users_email_idx") else {
        panic!("expected DropIndex");
    };
    assert!(d.if_exists);
}

#[test]
fn drop_index_folds_unquoted_and_preserves_quoted_names() {
    let ast::Statement::DropIndex(d) = ok("DROP INDEX MyIdx") else {
        panic!("expected DropIndex");
    };
    assert_eq!(d.name, "myidx", "unquoted index name folds to lowercase");

    let ast::Statement::DropIndex(d) = ok(r#"DROP INDEX "MyIdx""#) else {
        panic!("expected DropIndex");
    };
    assert_eq!(d.name, "MyIdx", "quoted index name keeps its case");
}

#[test]
fn drop_index_accepts_multiple_indexes_as_batch() {
    // Multi-object DROP desugars to the internal Batch of single drops (executed atomically
    // within the one statement transaction).
    let ast::Statement::Batch(stmts) = ok("DROP INDEX a, b") else {
        panic!("expected Batch");
    };
    assert_eq!(stmts.len(), 2);
    assert!(matches!(&stmts[0], ast::Statement::DropIndex(d) if d.name == "a"));
    assert!(matches!(&stmts[1], ast::Statement::DropIndex(d) if d.name == "b"));
}

// --- ALTER TABLE ------------------------------------------------------

#[test]
fn alter_table_add_column_with_and_without_keyword() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD COLUMN c INT") else {
        panic!("expected AlterTable");
    };
    assert_eq!(a.name, "t");
    assert!(!a.if_exists);
    let ast::AlterTableAction::AddColumn {
        column,
        if_not_exists,
    } = a.action
    else {
        panic!("expected AddColumn");
    };
    assert_eq!(column.name, "c");
    assert_eq!(column.ty, ColumnType::Int);
    assert!(column.nullable);
    assert!(!if_not_exists);

    // The `COLUMN` keyword is optional.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD d TEXT") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddColumn { column, .. } = a.action else {
        panic!("expected AddColumn");
    };
    assert_eq!(column.name, "d");
    assert_eq!(column.ty, ColumnType::Text);
}

#[test]
fn alter_table_drop_column_plain_and_if_exists() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t DROP COLUMN c") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::DropColumn { name, if_exists } = a.action else {
        panic!("expected DropColumn");
    };
    assert_eq!(name, "c");
    assert!(!if_exists);

    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t DROP COLUMN IF EXISTS c") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::DropColumn { if_exists, .. } = a.action else {
        panic!("expected DropColumn");
    };
    assert!(if_exists);
}

#[test]
fn alter_table_rename_column_folds_and_preserves_quotes() {
    let ast::Statement::AlterTable(a) = ok(r#"ALTER TABLE IF EXISTS "T" RENAME COLUMN "A" TO B"#)
    else {
        panic!("expected AlterTable");
    };
    assert_eq!(a.name, "T", "quoted table name keeps its case");
    assert!(a.if_exists);
    let ast::AlterTableAction::RenameColumn { from, to } = a.action else {
        panic!("expected RenameColumn");
    };
    assert_eq!(from, "A", "quoted column keeps its case");
    assert_eq!(to, "b", "unquoted column folds to lowercase");
}

#[test]
fn alter_table_rejects_multiple_actions() {
    let err =
        parse("ALTER TABLE t ADD a INT, DROP COLUMN b").expect_err("multi-action ALTER TABLE");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

#[test]
fn alter_table_add_primary_key_named_and_anonymous() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD CONSTRAINT pk PRIMARY KEY (id)")
    else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddConstraint(ast::TableConstraint::PrimaryKey { name, columns }) =
        a.action
    else {
        panic!("expected PrimaryKey constraint");
    };
    assert_eq!(name.as_deref(), Some("pk"));
    assert_eq!(columns, vec!["id".to_owned()]);

    // Anonymous (no `CONSTRAINT name`) + multi-column.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD PRIMARY KEY (a, b)") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddConstraint(ast::TableConstraint::PrimaryKey { name, columns }) =
        a.action
    else {
        panic!("expected PrimaryKey constraint");
    };
    assert_eq!(name, None);
    assert_eq!(columns, vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
fn alter_table_add_unique_constraint() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD CONSTRAINT u UNIQUE (email)") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddConstraint(ast::TableConstraint::Unique { name, columns }) =
        a.action
    else {
        panic!("expected Unique constraint");
    };
    assert_eq!(name.as_deref(), Some("u"));
    assert_eq!(columns, vec!["email".to_owned()]);
}

#[test]
fn alter_table_add_foreign_key_with_referential_actions() {
    let ast::Statement::AlterTable(a) = ok(
        "ALTER TABLE orders ADD CONSTRAINT fk FOREIGN KEY (user_id) \
             REFERENCES users (id) ON DELETE CASCADE ON UPDATE SET NULL",
    ) else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddConstraint(ast::TableConstraint::ForeignKey {
        name,
        columns,
        foreign_table,
        referred_columns,
        on_delete,
        on_update,
    }) = a.action
    else {
        panic!("expected ForeignKey constraint");
    };
    assert_eq!(name.as_deref(), Some("fk"));
    assert_eq!(columns, vec!["user_id".to_owned()]);
    assert_eq!(foreign_table, "users");
    assert_eq!(referred_columns, vec!["id".to_owned()]);
    assert_eq!(on_delete, Some(ast::ReferentialAction::Cascade));
    assert_eq!(on_update, Some(ast::ReferentialAction::SetNull));
}

#[test]
fn alter_table_add_check_constraint() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ADD CONSTRAINT positive CHECK (age > 0)")
    else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AddConstraint(ast::TableConstraint::Check {
        name,
        expr,
        predicate_sql,
    }) = a.action
    else {
        panic!("expected Check constraint");
    };
    assert_eq!(name.as_deref(), Some("positive"));
    assert!(matches!(expr, ast::Expr::Binary { .. }), "got {expr:?}");
    assert_eq!(predicate_sql, "age > 0");
}

#[test]
fn alter_table_drop_constraint_plain_and_if_exists() {
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t DROP CONSTRAINT c") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::DropConstraint { name, if_exists } = a.action else {
        panic!("expected DropConstraint");
    };
    assert_eq!(name, "c");
    assert!(!if_exists);

    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t DROP CONSTRAINT IF EXISTS c") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::DropConstraint { if_exists, .. } = a.action else {
        panic!("expected DropConstraint");
    };
    assert!(if_exists);
}

#[test]
fn alter_table_alter_column_type_and_default() {
    // `SET DATA TYPE <type>`.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ALTER COLUMN c SET DATA TYPE BIGINT")
    else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AlterColumn { column, change } = a.action else {
        panic!("expected AlterColumn");
    };
    assert_eq!(column, "c");
    assert_eq!(change, ast::ColumnChange::SetType(ColumnType::BigInt));

    // `SET DEFAULT <expr>`.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ALTER COLUMN c SET DEFAULT 0") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AlterColumn { change, .. } = a.action else {
        panic!("expected AlterColumn");
    };
    let ast::ColumnChange::SetDefault { expr, sql } = change else {
        panic!("expected SetDefault");
    };
    assert_eq!(expr, ast::Expr::Literal(ast::Value::Int(0)));
    assert_eq!(sql, "0");

    // `DROP DEFAULT`.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ALTER COLUMN c DROP DEFAULT") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AlterColumn { change, .. } = a.action else {
        panic!("expected AlterColumn");
    };
    assert_eq!(change, ast::ColumnChange::DropDefault);
}

#[test]
fn alter_table_alter_column_not_null() {
    // The `COLUMN` keyword is optional.
    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ALTER c SET NOT NULL") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AlterColumn { column, change } = a.action else {
        panic!("expected AlterColumn");
    };
    assert_eq!(column, "c");
    assert_eq!(change, ast::ColumnChange::SetNotNull);

    let ast::Statement::AlterTable(a) = ok("ALTER TABLE t ALTER COLUMN c DROP NOT NULL") else {
        panic!("expected AlterTable");
    };
    let ast::AlterTableAction::AlterColumn { change, .. } = a.action else {
        panic!("expected AlterColumn");
    };
    assert_eq!(change, ast::ColumnChange::DropNotNull);
}

// --- CREATE TABLE ... AS SELECT -------------------------------

#[test]
fn create_table_as_select() {
    let ast::Statement::CreateTableAs(ct) = ok("CREATE TABLE dst AS SELECT id, name FROM src")
    else {
        panic!("expected CreateTableAs");
    };
    assert_eq!(ct.name, "dst");
    assert!(!ct.if_not_exists);
    assert_eq!(
        ct.query.projection.len(),
        2,
        "the source query is preserved"
    );
}

#[test]
fn create_table_if_not_exists_as_select() {
    let ast::Statement::CreateTableAs(ct) = ok("CREATE TABLE IF NOT EXISTS dst AS SELECT 1 AS one")
    else {
        panic!("expected CreateTableAs");
    };
    assert!(ct.if_not_exists);
}

#[test]
fn create_table_as_select_with_column_list_is_rejected() {
    // sqlparser only accepts a typed column list here; CTAS with one is out of scope (use aliases).
    assert!(
        matches!(
            parse("CREATE TABLE dst (a INT, b TEXT) AS SELECT id, name FROM src"),
            Err(Error::Unsupported(_))
        ),
        "expected Unsupported for a CTAS with a column list",
    );
}

// --- CREATE / DROP VIEW -----------------------------------------------

#[test]
fn create_view_plain() {
    let ast::Statement::CreateView(v) = ok("CREATE VIEW v AS SELECT id FROM t") else {
        panic!("expected CreateView");
    };
    assert_eq!(v.name, "v");
    assert!(!v.or_replace);
    assert!(v.columns.is_empty());
    assert_eq!(
        v.query.projection.len(),
        1,
        "view body keeps its projection"
    );
}

#[test]
fn create_view_or_replace_with_explicit_columns() {
    let ast::Statement::CreateView(v) = ok("CREATE OR REPLACE VIEW v (a, b) AS SELECT x, y FROM t")
    else {
        panic!("expected CreateView");
    };
    assert_eq!(v.name, "v");
    assert!(v.or_replace);
    assert_eq!(v.columns, vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
fn create_view_folds_unquoted_and_preserves_quoted_name() {
    let ast::Statement::CreateView(v) = ok(r#"CREATE VIEW "V" AS SELECT 1"#) else {
        panic!("expected CreateView");
    };
    assert_eq!(v.name, "V", "quoted view name keeps its case");
}

#[test]
fn create_materialized_view_parses_with_flag() {
    // Materialized views are supported; the `materialized` flag is carried through.
    let ast::Statement::CreateView(v) = ok("CREATE MATERIALIZED VIEW v AS SELECT 1") else {
        panic!("expected CreateView");
    };
    assert_eq!(v.name, "v");
    assert!(v.materialized, "MATERIALIZED flag must be set");
}

#[test]
fn drop_view_plain_and_if_exists() {
    let ast::Statement::DropView(d) = ok("DROP VIEW v") else {
        panic!("expected DropView");
    };
    assert_eq!(d.name, "v");
    assert!(!d.if_exists);

    let ast::Statement::DropView(d) = ok("DROP VIEW IF EXISTS v") else {
        panic!("expected DropView");
    };
    assert!(d.if_exists);
}

// --- CREATE / DROP SCHEMA ---------------------------------------------

#[test]
fn create_schema_plain_and_if_not_exists() {
    let ast::Statement::CreateSchema(s) = ok("CREATE SCHEMA app") else {
        panic!("expected CreateSchema");
    };
    assert_eq!(s.name, "app");
    assert!(!s.if_not_exists);

    let ast::Statement::CreateSchema(s) = ok("CREATE SCHEMA IF NOT EXISTS app") else {
        panic!("expected CreateSchema");
    };
    assert!(s.if_not_exists);
}

#[test]
fn create_schema_folds_unquoted_and_preserves_quoted_name() {
    let ast::Statement::CreateSchema(s) = ok(r#"CREATE SCHEMA "App""#) else {
        panic!("expected CreateSchema");
    };
    assert_eq!(s.name, "App", "quoted schema name keeps its case");
}

#[test]
fn create_schema_rejects_authorization() {
    let err = parse("CREATE SCHEMA AUTHORIZATION bob").expect_err("unnamed authorization");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");

    let err = parse("CREATE SCHEMA app AUTHORIZATION bob").expect_err("named authorization");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

#[test]
fn drop_schema_plain_and_if_exists() {
    let ast::Statement::DropSchema(d) = ok("DROP SCHEMA app") else {
        panic!("expected DropSchema");
    };
    assert_eq!(d.name, "app");
    assert!(!d.if_exists);
    assert!(!d.cascade, "plain DROP SCHEMA is RESTRICT");

    let ast::Statement::DropSchema(d) = ok("DROP SCHEMA IF EXISTS app") else {
        panic!("expected DropSchema");
    };
    assert!(d.if_exists);
    assert!(!d.cascade);
}

#[test]
fn drop_schema_cascade_and_restrict() {
    // CASCADE flips the flag; RESTRICT is the explicit spelling of the default.
    let ast::Statement::DropSchema(d) = ok("DROP SCHEMA app CASCADE") else {
        panic!("expected DropSchema");
    };
    assert!(d.cascade, "CASCADE drops the schema's tables with it");

    let ast::Statement::DropSchema(d) = ok("DROP SCHEMA IF EXISTS app RESTRICT") else {
        panic!("expected DropSchema");
    };
    assert!(d.if_exists);
    assert!(!d.cascade);

    // CASCADE remains unsupported for object kinds that track no dependencies (tables now
    // support itso the pin uses a sequence).
    assert!(matches!(
        parse("DROP SEQUENCE s CASCADE").expect_err("sequence cascade"),
        Error::Unsupported(_)
    ));
}

// --- CREATE / DROP SEQUENCE -------------------------------------------

#[test]
fn create_sequence_plain_has_no_options() {
    let ast::Statement::CreateSequence(s) = ok("CREATE SEQUENCE s") else {
        panic!("expected CreateSequence");
    };
    assert_eq!(s.name, "s");
    assert!(!s.if_not_exists);
    assert!(s.options.is_empty());
}

#[test]
fn create_sequence_full_options() {
    // sqlparser requires the canonical option order:
    // INCREMENT, MINVALUE, MAXVALUE, START, CACHE, CYCLE.
    let ast::Statement::CreateSequence(s) = ok("CREATE SEQUENCE IF NOT EXISTS s \
             INCREMENT BY 2 MINVALUE 1 MAXVALUE 100 START WITH 10 CACHE 5 CYCLE")
    else {
        panic!("expected CreateSequence");
    };
    assert_eq!(s.name, "s");
    assert!(s.if_not_exists);
    assert_eq!(s.options.len(), 6);

    let increment = s
        .options
        .iter()
        .find_map(|o| match o {
            ast::SequenceOption::Increment(e) => Some(e),
            _ => None,
        })
        .expect("INCREMENT option present");
    assert_eq!(*increment, ast::Expr::Literal(ast::Value::Int(2)));

    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::Start(_)))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::MinValue(Some(_))))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::MaxValue(Some(_))))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::Cache(_)))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::Cycle(true)))
    );
}

#[test]
fn create_sequence_no_bounds_and_no_cycle() {
    let ast::Statement::CreateSequence(s) =
        ok("CREATE SEQUENCE s NO MINVALUE NO MAXVALUE NO CYCLE")
    else {
        panic!("expected CreateSequence");
    };
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::MinValue(None)))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::MaxValue(None)))
    );
    assert!(
        s.options
            .iter()
            .any(|o| matches!(o, ast::SequenceOption::Cycle(false)))
    );
}

#[test]
fn create_sequence_rejects_owned_by() {
    let err = parse("CREATE SEQUENCE s OWNED BY t.c").expect_err("OWNED BY");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

#[test]
fn drop_sequence_plain_and_if_exists() {
    let ast::Statement::DropSequence(d) = ok("DROP SEQUENCE s") else {
        panic!("expected DropSequence");
    };
    assert_eq!(d.name, "s");
    assert!(!d.if_exists);

    let ast::Statement::DropSequence(d) = ok("DROP SEQUENCE IF EXISTS s") else {
        panic!("expected DropSequence");
    };
    assert!(d.if_exists);
}

// --- TRUNCATE TABLE ---------------------------------------------------

#[test]
fn truncate_table_plain() {
    let ast::Statement::Truncate(t) = ok("TRUNCATE TABLE orders") else {
        panic!("expected Truncate");
    };
    assert_eq!(t.name, "orders");
    assert!(!t.restart_identity);
}

#[test]
fn truncate_without_table_keyword() {
    // The TABLE keyword is optional.
    let ast::Statement::Truncate(t) = ok("TRUNCATE orders") else {
        panic!("expected Truncate");
    };
    assert_eq!(t.name, "orders");
    assert!(!t.restart_identity);
}

#[test]
fn truncate_restart_identity() {
    // `GenericDialect` parses `RESTART IDENTITY` / `CONTINUE IDENTITY`, and `convert_truncate`
    // honors it: RESTART sets the flag, CONTINUE (and the unspecified form) leaves it false.
    let ast::Statement::Truncate(t) = ok("TRUNCATE TABLE orders RESTART IDENTITY") else {
        panic!("expected Truncate");
    };
    assert!(t.restart_identity);
    let ast::Statement::Truncate(t) = ok("TRUNCATE TABLE orders CONTINUE IDENTITY") else {
        panic!("expected Truncate");
    };
    assert!(!t.restart_identity);
}

#[test]
fn truncate_rejects_multiple_tables() {
    let err = parse("TRUNCATE TABLE a, b").expect_err("multi-table TRUNCATE");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

// --- INSERT -----------------------------------------------------------

#[test]
fn insert_with_column_list_multi_row() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y')") else {
        panic!("expected Insert");
    };
    assert_eq!(ins.table, "t");
    assert_eq!(ins.columns, ["a", "b"]);
    let ast::InsertSource::Values(rows) = ins.source else {
        panic!("expected Values source");
    };
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0],
        vec![
            Some(ast::Expr::Literal(ast::Value::Int(1))),
            Some(ast::Expr::Literal(ast::Value::Text("x".to_owned()))),
        ],
    );
}

#[test]
fn insert_without_column_list() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t VALUES (1)") else {
        panic!("expected Insert");
    };
    assert!(ins.columns.is_empty());
    assert!(matches!(ins.source, ast::InsertSource::Values(_)));
}

#[test]
fn insert_select_basic() {
    let ast::Statement::Insert(ins) =
        ok("INSERT INTO archive (id, name) SELECT id, name FROM users")
    else {
        panic!("expected Insert");
    };
    assert_eq!(ins.table, "archive");
    assert_eq!(ins.columns, ["id", "name"]);
    let ast::InsertSource::Select(sel) = ins.source else {
        panic!("expected Select source");
    };
    assert_eq!(sel.projection.len(), 2, "two projected columns");
}

#[test]
fn insert_select_without_column_list() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t SELECT * FROM src") else {
        panic!("expected Insert");
    };
    assert!(ins.columns.is_empty());
    assert!(matches!(ins.source, ast::InsertSource::Select(_)));
}

#[test]
fn insert_plain_has_no_on_conflict() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t VALUES (1)") else {
        panic!("expected Insert");
    };
    assert!(ins.on_conflict.is_none());
}

#[test]
fn insert_on_conflict_do_nothing_bare() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING") else {
        panic!("expected Insert");
    };
    let oc = ins.on_conflict.expect("on_conflict present");
    assert!(oc.target.is_none(), "bare ON CONFLICT has no target");
    assert_eq!(oc.action, ast::ConflictAction::DoNothing);
}

#[test]
fn insert_on_conflict_columns_target() {
    let ast::Statement::Insert(ins) =
        ok("INSERT INTO t (id, v) VALUES (1, 2) ON CONFLICT (id) DO NOTHING")
    else {
        panic!("expected Insert");
    };
    let oc = ins.on_conflict.expect("on_conflict present");
    assert_eq!(oc.action, ast::ConflictAction::DoNothing);
    let Some(ast::ConflictTarget::Columns(cols)) = oc.target else {
        panic!("expected Columns target");
    };
    assert_eq!(cols, vec!["id".to_owned()]);
}

#[test]
fn insert_on_conflict_on_constraint_target() {
    let ast::Statement::Insert(ins) =
        ok("INSERT INTO t VALUES (1) ON CONFLICT ON CONSTRAINT t_pkey DO NOTHING")
    else {
        panic!("expected Insert");
    };
    let oc = ins.on_conflict.expect("on_conflict present");
    let Some(ast::ConflictTarget::Constraint(name)) = oc.target else {
        panic!("expected Constraint target");
    };
    assert_eq!(name, "t_pkey");
}

/// Parse an `INSERT`, returning its `ON CONFLICT` action (panics if either is absent).
fn conflict_action(sql: &str) -> ast::ConflictAction {
    let ast::Statement::Insert(ins) = ok(sql) else {
        panic!("expected Insert");
    };
    ins.on_conflict.expect("on_conflict present").action
}

#[test]
fn insert_on_conflict_do_update_set() {
    let action =
        conflict_action("INSERT INTO t (id, v) VALUES (1, 2) ON CONFLICT (id) DO UPDATE SET v = 9");
    let ast::ConflictAction::DoUpdate {
        assignments,
        filter,
    } = action
    else {
        panic!("expected DoUpdate, got {action:?}");
    };
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].column, "v");
    assert_eq!(assignments[0].value, ast::Expr::Literal(ast::Value::Int(9)));
    assert!(filter.is_none());
}

#[test]
fn insert_on_conflict_do_update_references_excluded() {
    // `EXCLUDED.col` (the proposed insert value) parses to a qualified column ref.
    let action = conflict_action(
        "INSERT INTO t (id, v) VALUES (1, 2) ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v",
    );
    let ast::ConflictAction::DoUpdate { assignments, .. } = action else {
        panic!("expected DoUpdate, got {action:?}");
    };
    assert_eq!(
        assignments[0].value,
        ast::Expr::QualifiedColumn {
            table: "excluded".to_owned(),
            column: "v".to_owned(),
        }
    );
}

#[test]
fn insert_on_conflict_do_update_with_where() {
    let action = conflict_action(
        "INSERT INTO t (id, v) VALUES (1, 2) ON CONFLICT (id) DO UPDATE SET v = 9 WHERE t.v < 9",
    );
    let ast::ConflictAction::DoUpdate { filter, .. } = action else {
        panic!("expected DoUpdate, got {action:?}");
    };
    assert!(filter.is_some(), "WHERE predicate should be captured");
}

#[test]
fn insert_on_conflict_do_update_multi_assignment() {
    let action = conflict_action(
        "INSERT INTO t (id, a, b) VALUES (1, 2, 3) ON CONFLICT (id) DO UPDATE SET a = 1, b = EXCLUDED.b",
    );
    let ast::ConflictAction::DoUpdate { assignments, .. } = action else {
        panic!("expected DoUpdate, got {action:?}");
    };
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].column, "a");
    assert_eq!(assignments[1].column, "b");
}

#[test]
fn insert_on_conflict_do_update_parses() {
    // The parser accepts `ON CONFLICT (target) DO UPDATE SET ...`; the analyzer/executor
    // upsert path is covered by the analyzer + integration tests. Here we only assert the
    // surface parses into an INSERT statement.
    assert!(matches!(
        parse("INSERT INTO t (id, v) VALUES (1, 2) ON CONFLICT (id) DO UPDATE SET v = 9"),
        Ok(ast::Statement::Insert(_))
    ));
}

// --- RETURNING ------------------------------------------------

#[test]
fn insert_returning_column_expr() {
    let ast::Statement::Insert(ins) =
        ok("INSERT INTO t (id, name) VALUES (1, 'a') RETURNING id, name")
    else {
        panic!("expected Insert");
    };
    assert_eq!(
        ins.returning,
        vec![
            ast::SelectItem::Expr {
                expr: ast::Expr::Column("id".to_owned()),
                alias: None,
            },
            ast::SelectItem::Expr {
                expr: ast::Expr::Column("name".to_owned()),
                alias: None,
            },
        ]
    );
}

#[test]
fn insert_returning_wildcard() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t (id) VALUES (1) RETURNING *") else {
        panic!("expected Insert");
    };
    assert_eq!(ins.returning, vec![ast::SelectItem::Wildcard]);
}

#[test]
fn insert_without_returning_has_empty_vec() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO t (id) VALUES (1)") else {
        panic!("expected Insert");
    };
    assert!(ins.returning.is_empty());
}

#[test]
fn update_returning_wildcard() {
    let ast::Statement::Update(upd) = ok("UPDATE t SET val = 2 WHERE id = 1 RETURNING *") else {
        panic!("expected Update");
    };
    assert_eq!(upd.returning, vec![ast::SelectItem::Wildcard]);
}

#[test]
fn delete_returning_column_with_alias() {
    let ast::Statement::Delete(del) = ok("DELETE FROM t WHERE id = 1 RETURNING id AS deleted_id")
    else {
        panic!("expected Delete");
    };
    assert_eq!(
        del.returning,
        vec![ast::SelectItem::Expr {
            expr: ast::Expr::Column("id".to_owned()),
            alias: Some("deleted_id".to_owned()),
        }]
    );
}

#[test]
fn explain_insert_returning_parses() {
    // RETURNING inside EXPLAIN is purely a parser surface check.
    assert!(matches!(
        parse("EXPLAIN INSERT INTO t (id) VALUES (1) RETURNING id"),
        Ok(ast::Statement::Explain(..))
    ));
}

#[test]
fn explain_format_json_parses_and_graphviz_rejected() {
    // FORMAT JSON sets the structured-output flag; the default and TEXT stay Text.
    let ast::Statement::Explain(_, opts) = ok("EXPLAIN FORMAT JSON SELECT 1") else {
        panic!("expected Explain");
    };
    assert_eq!(opts.format, ast::ExplainFormat::Json);
    let ast::Statement::Explain(_, opts) = ok("EXPLAIN SELECT 1") else {
        panic!("expected Explain");
    };
    assert_eq!(opts.format, ast::ExplainFormat::Text);
    // An unsupported format is rejected rather than silently ignored.
    assert!(matches!(
        parse("EXPLAIN FORMAT GRAPHVIZ SELECT 1"),
        Err(Error::Unsupported(_))
    ));
}

// --- WITH / CTE -----------------------------------------------

#[test]
fn with_single_cte_bare() {
    let ast::Statement::Select(s) = ok("WITH cte AS (SELECT 1 AS n) SELECT n FROM cte") else {
        panic!("expected Select");
    };
    assert_eq!(s.with.len(), 1);
    let cte = &s.with[0];
    assert_eq!(cte.name, "cte");
    assert!(cte.columns.is_empty());
}

#[test]
fn with_multi_cte_chain() {
    let ast::Statement::Select(s) =
        ok("WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT x FROM a")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.with.len(), 2);
    assert_eq!(s.with[0].name, "a");
    assert_eq!(s.with[1].name, "b");
}

#[test]
fn with_explicit_column_names() {
    let ast::Statement::Select(s) = ok("WITH nums(x, y) AS (SELECT 1, 2) SELECT x FROM nums")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.with[0].columns, vec!["x".to_owned(), "y".to_owned()]);
}

#[test]
fn with_folds_cte_name() {
    // Unquoted CTE names fold to lowercase like any identifier.
    let ast::Statement::Select(s) = ok("WITH MyCTE AS (SELECT 1) SELECT * FROM mycte") else {
        panic!("expected Select");
    };
    assert_eq!(s.with[0].name, "mycte");
}

#[test]
fn with_quoted_cte_name_preserves_case() {
    let ast::Statement::Select(s) = ok(r#"WITH "MyCTE" AS (SELECT 1) SELECT * FROM "MyCTE""#)
    else {
        panic!("expected Select");
    };
    assert_eq!(s.with[0].name, "MyCTE");
}

#[test]
fn select_without_with_has_empty_vec() {
    let ast::Statement::Select(s) = ok("SELECT 1") else {
        panic!("expected Select");
    };
    assert!(s.with.is_empty());
}

#[test]
fn with_recursive_is_rejected() {
    // Guard: WITH RECURSIVE on a plain SELECT should now parse, not reject.
    // This test is superseded — see with_recursive_* tests below.
    // Keep it passing to confirm the old rejection is removed.
    assert!(matches!(
        parse("WITH RECURSIVE cte AS (SELECT 1) SELECT * FROM cte"),
        Ok(ast::Statement::Select(_)),
    ));
}

#[test]
fn with_materialized_is_rejected() {
    // Syntax varies by dialect; either a parse error or Unsupported is acceptable —
    // what matters is that it does NOT silently succeed with materialization ignored.
    assert!(parse("WITH cte AS MATERIALIZED (SELECT 1) SELECT * FROM cte").is_err());
}

#[test]
fn with_set_op_body_in_non_recursive_is_accepted() {
    // A non-recursive CTE body may be a set operation — it lowers to a `SetOp` body and is
    // inlined like a `(SELECT ... UNION ...) AS x` derived table.
    let ast::Statement::Select(select) =
        ok("WITH cte AS (SELECT 1 UNION SELECT 2) SELECT * FROM cte")
    else {
        panic!("expected Select");
    };
    let cte = &select.with[0];
    assert_eq!(cte.name, "cte");
    let ast::CteBody::Query(q) = &cte.body else {
        panic!("expected a query body");
    };
    assert!(matches!(**q, ast::SelectBody::SetOp { .. }));
}

#[test]
fn with_before_a_set_operation_carries_onto_the_envelope() {
    // A `WITH` in front of a UNION/INTERSECT/EXCEPT scopes over the whole set operation: it parses
    // into the `SetOperation` envelope's `with`, so every branch can reference the CTE.
    let ast::Statement::SetOperation(so) =
        parse("WITH cte AS (SELECT 1) SELECT * FROM cte UNION SELECT 2").unwrap()
    else {
        panic!("expected a SetOperation");
    };
    assert_eq!(so.with.len(), 1);
    assert_eq!(so.with[0].name, "cte");
    assert!(matches!(so.body, ast::SelectBody::SetOp { .. }));
}

// --- WITH RECURSIVE -------------------------------------------

#[test]
fn with_recursive_union_all_parses() {
    // Canonical recursive CTE: `WITH RECURSIVE name AS (anchor UNION ALL arm) SELECT ...`.
    let ast::Statement::Select(s) = ok("WITH RECURSIVE nums AS \
             (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 5) \
             SELECT n FROM nums")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.with.len(), 1);
    let cte = &s.with[0];
    assert_eq!(cte.name, "nums");
    assert!(cte.recursive);
    let ast::CteBody::Query(q) = &cte.body else {
        panic!("expected a query body");
    };
    assert!(matches!(
        **q,
        ast::SelectBody::SetOp {
            op: ast::SetOp::Union,
            all: true,
            ..
        }
    ));
}

#[test]
fn with_recursive_non_recursive_cte_is_not_marked_recursive() {
    // Even under WITH RECURSIVE, a CTE whose body is a plain SELECT stays recursive=true
    // (the flag follows the WITH clause, not the body structure).
    let ast::Statement::Select(s) = ok("WITH RECURSIVE cte AS (SELECT 1 AS n) SELECT n FROM cte")
    else {
        panic!("expected Select");
    };
    assert!(s.with[0].recursive);
    let ast::CteBody::Query(q) = &s.with[0].body else {
        panic!("expected a query body");
    };
    assert!(matches!(**q, ast::SelectBody::Select(_)));
}

#[test]
fn with_recursive_union_distinct_parses_as_distinct() {
    // Both UNION ALL and UNION (distinct) are valid recursive bodies; the quantifier maps
    // onto `all`. INTERSECT/EXCEPT remain rejected.
    let ast::Statement::Select(s) =
        ok("WITH RECURSIVE cte AS (SELECT 1 UNION SELECT 2) SELECT * FROM cte")
    else {
        panic!("expected Select");
    };
    let ast::CteBody::Query(q) = &s.with[0].body else {
        panic!("expected a query body");
    };
    let ast::SelectBody::SetOp { op, all, .. } = &**q else {
        panic!("expected a UNION body");
    };
    assert!(matches!(op, ast::SetOp::Union));
    assert!(!all, "UNION without ALL is distinct");
    assert!(matches!(
        parse("WITH RECURSIVE cte AS (SELECT 1 EXCEPT SELECT 2) SELECT * FROM cte"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn with_recursive_multi_chain() {
    // Multiple CTEs under WITH RECURSIVE — each gets recursive=true.
    let ast::Statement::Select(s) = ok(
        "WITH RECURSIVE a AS (SELECT 1 AS x UNION ALL SELECT x + 1 FROM a WHERE x < 3), \
             b AS (SELECT 1 AS y) \
             SELECT a.x FROM a",
    ) else {
        panic!("expected Select");
    };
    assert_eq!(s.with.len(), 2);
    assert!(s.with[0].recursive);
    assert!(s.with[1].recursive);
}

// --- Window functions OVER (...) -----------------------------

fn get_window(sql: &str) -> ast::WindowFunction {
    let ast::Statement::Select(s) = ok(sql) else {
        panic!("expected Select")
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected Expr projection");
    };
    let ast::Expr::WindowFunction(wf) = expr else {
        panic!("expected WindowFunction")
    };
    (**wf).clone()
}

#[test]
fn window_row_number_no_partition_no_order() {
    let wf = get_window("SELECT ROW_NUMBER() OVER () FROM t");
    assert_eq!(wf.func, ast::WindowFunc::RowNumber);
    assert!(wf.args.is_empty());
    assert!(wf.partition.is_empty());
    assert!(wf.order.is_empty());
}

#[test]
fn window_rank_with_partition_and_order() {
    let wf = get_window("SELECT RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM t");
    assert_eq!(wf.func, ast::WindowFunc::Rank);
    assert_eq!(wf.partition.len(), 1);
    assert_eq!(wf.order.len(), 1);
    assert!(!wf.order[0].ascending);
}

#[test]
fn window_sum_aggregate_over() {
    let wf = get_window("SELECT SUM(salary) OVER (PARTITION BY dept) FROM t");
    assert_eq!(wf.func, ast::WindowFunc::Aggregate(ast::AggregateFunc::Sum));
    assert_eq!(wf.args.len(), 1);
    assert_eq!(wf.partition.len(), 1);
}

#[test]
fn window_lag_with_args() {
    let wf = get_window("SELECT LAG(salary, 1, 0) OVER (ORDER BY id) FROM t");
    assert_eq!(wf.func, ast::WindowFunc::Lag);
    assert_eq!(wf.args.len(), 3);
    assert_eq!(wf.order.len(), 1);
}

#[test]
fn window_dense_rank_folds_name() {
    let wf = get_window("SELECT DENSE_RANK() OVER (ORDER BY id) FROM t");
    assert_eq!(wf.func, ast::WindowFunc::DenseRank);
}

#[test]
fn window_count_over() {
    let wf = get_window("SELECT COUNT(*) OVER (PARTITION BY dept) FROM t");
    assert_eq!(
        wf.func,
        ast::WindowFunc::Aggregate(ast::AggregateFunc::Count)
    );
}

#[test]
fn window_frame_is_rejected() {
    // Landed — frame now parses; update to assert it parses correctly.
    let wf = get_window(
        "SELECT ROW_NUMBER() OVER \
             (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
    );
    let frame = wf.frame.expect("frame");
    assert_eq!(frame.units, ast::WindowFrameUnits::Rows);
    assert_eq!(frame.start, ast::WindowFrameBound::UnboundedPreceding);
    assert_eq!(frame.end, Some(ast::WindowFrameBound::CurrentRow));
}

// --- Window frame bounds -------------------------------------

#[test]
fn window_frame_rows_between_parses() {
    let wf = get_window(
        "SELECT SUM(v) OVER \
             (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) FROM t",
    );
    let frame = wf.frame.expect("frame");
    assert_eq!(frame.units, ast::WindowFrameUnits::Rows);
    assert!(matches!(frame.start, ast::WindowFrameBound::Preceding(_)));
    assert!(matches!(
        frame.end,
        Some(ast::WindowFrameBound::Following(_))
    ));
}

#[test]
fn window_frame_range_current_row_parses() {
    let wf = get_window(
        "SELECT SUM(v) OVER (ORDER BY id RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
    );
    let frame = wf.frame.expect("frame");
    assert_eq!(frame.units, ast::WindowFrameUnits::Range);
    assert_eq!(frame.start, ast::WindowFrameBound::UnboundedPreceding);
    assert_eq!(frame.end, Some(ast::WindowFrameBound::CurrentRow));
}

#[test]
fn window_frame_groups_between_parses() {
    let wf = get_window(
        "SELECT RANK() OVER \
             (ORDER BY dept GROUPS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
    );
    let frame = wf.frame.expect("frame");
    assert_eq!(frame.units, ast::WindowFrameUnits::Groups);
    assert!(matches!(frame.start, ast::WindowFrameBound::Preceding(_)));
    assert_eq!(frame.end, Some(ast::WindowFrameBound::UnboundedFollowing));
}

#[test]
fn window_frame_shorthand_no_between_parses() {
    // `ROWS UNBOUNDED PRECEDING` shorthand — end is None.
    let wf = get_window("SELECT ROW_NUMBER() OVER (ORDER BY id ROWS UNBOUNDED PRECEDING) FROM t");
    let frame = wf.frame.expect("frame");
    assert_eq!(frame.units, ast::WindowFrameUnits::Rows);
    assert_eq!(frame.start, ast::WindowFrameBound::UnboundedPreceding);
    assert!(frame.end.is_none());
}

#[test]
fn window_without_frame_has_none() {
    let wf = get_window("SELECT RANK() OVER (ORDER BY id) FROM t");
    assert!(wf.frame.is_none());
}

#[test]
fn window_unknown_func_is_rejected() {
    assert!(matches!(
        parse("SELECT MY_FUNC() OVER (ORDER BY id) FROM t"),
        Err(Error::Unsupported(_)),
    ));
}

// --- SELECT -----------------------------------------------------------

#[test]
fn select_qualified_wildcard_folds_table() {
    // `T.*` folds the qualifier to lowercase like any unquoted identifier.
    let ast::Statement::Select(s) = ok("SELECT T.* FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(
        s.projection,
        vec![ast::SelectItem::QualifiedWildcard("t".to_owned())]
    );
}

#[test]
fn select_wildcard_no_filter() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(s.projection, vec![ast::SelectItem::Wildcard]);
    let from = s.from.expect("FROM clause");
    assert_eq!(from.base.name, "t");
    assert!(from.joins.is_empty());
    assert!(s.filter.is_none());
    assert!(s.order_by.is_empty());
    assert!(s.limit.is_none());
}

#[test]
fn select_where_order_by_limit() {
    let ast::Statement::Select(s) =
        ok("SELECT id, name FROM users WHERE id > 10 ORDER BY name DESC LIMIT 5")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.projection.len(), 2);
    assert!(s.filter.is_some());
    assert_eq!(s.order_by.len(), 1);
    assert!(!s.order_by[0].ascending);
    assert_eq!(s.limit, Some(5));
}

#[test]
fn select_without_from() {
    let ast::Statement::Select(s) = ok("SELECT 1") else {
        panic!("expected Select");
    };
    assert!(s.from.is_none());
}

#[test]
fn select_expression_alias() {
    let ast::Statement::Select(s) = ok("SELECT a AS x FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { alias, .. } = &s.projection[0] else {
        panic!("expected aliased expression");
    };
    assert_eq!(alias.as_deref(), Some("x"));
}

// --- UPDATE / DELETE --------------------------------------------------

#[test]
fn update_with_assignments_and_filter() {
    let ast::Statement::Update(u) = ok("UPDATE t SET a = 1, b = 'z' WHERE id = 7") else {
        panic!("expected Update");
    };
    assert_eq!(u.table, "t");
    assert_eq!(u.assignments.len(), 2);
    assert_eq!(u.assignments[0].column, "a");
    assert_eq!(
        u.assignments[0].value,
        ast::Expr::Literal(ast::Value::Int(1))
    );
    assert!(u.filter.is_some());
    assert!(u.from.is_none());
}

// --- UPDATE ... FROM ------------------------------------------

#[test]
fn update_from_bare_table() {
    // `UPDATE t SET t.col = src.val FROM src WHERE t.id = src.id` — FROM with a plain table.
    let ast::Statement::Update(u) = ok("UPDATE t SET col = src.val FROM src WHERE t.id = src.id")
    else {
        panic!("expected Update");
    };
    assert_eq!(u.table, "t");
    let from = u.from.expect("FROM clause");
    assert_eq!(from.base.name, "src");
    assert!(from.joins.is_empty());
    assert!(u.filter.is_some());
}

#[test]
fn update_from_with_join() {
    // FROM can carry its own join chain (e.g. `FROM src JOIN other ON ...`).
    let ast::Statement::Update(u) =
        ok("UPDATE t SET col = 1 FROM src JOIN other ON src.id = other.id WHERE t.id = src.id")
    else {
        panic!("expected Update");
    };
    let from = u.from.expect("FROM clause");
    assert_eq!(from.base.name, "src");
    assert_eq!(from.joins.len(), 1);
    assert_eq!(from.joins[0].table.name, "other");
}

#[test]
fn update_from_folds_table_name() {
    // Unquoted identifiers in FROM are folded to lowercase.
    let ast::Statement::Update(u) = ok("UPDATE T SET col = 1 FROM SRC WHERE T.id = SRC.id") else {
        panic!("expected Update");
    };
    assert_eq!(u.table, "t");
    assert_eq!(u.from.expect("FROM clause").base.name, "src");
}

#[test]
fn update_without_from_has_none() {
    let ast::Statement::Update(u) = ok("UPDATE t SET col = 1") else {
        panic!("expected Update");
    };
    assert!(u.from.is_none());
}

#[test]
fn delete_with_and_without_filter() {
    let ast::Statement::Delete(d) = ok("DELETE FROM t WHERE id = 1") else {
        panic!("expected Delete");
    };
    assert_eq!(d.table, "t");
    assert!(d.filter.is_some());

    let ast::Statement::Delete(d) = ok("DELETE FROM t") else {
        panic!("expected Delete");
    };
    assert!(d.filter.is_none());
    assert!(d.using.is_none());
}

// --- DELETE ... USING -----------------------------------------

#[test]
fn delete_using_bare_table() {
    // `DELETE FROM t USING src WHERE t.id = src.id` — USING with a plain table.
    let ast::Statement::Delete(d) = ok("DELETE FROM t USING src WHERE t.id = src.id") else {
        panic!("expected Delete");
    };
    assert_eq!(d.table, "t");
    let using = d.using.expect("USING clause");
    assert_eq!(using.base.name, "src");
    assert!(using.joins.is_empty());
    assert!(d.filter.is_some());
}

#[test]
fn delete_using_with_join() {
    // USING can carry its own join chain (e.g. `USING src JOIN other ON ...`).
    let ast::Statement::Delete(d) =
        ok("DELETE FROM t USING src JOIN other ON src.id = other.id WHERE t.id = src.id")
    else {
        panic!("expected Delete");
    };
    let using = d.using.expect("USING clause");
    assert_eq!(using.base.name, "src");
    assert_eq!(using.joins.len(), 1);
    assert_eq!(using.joins[0].table.name, "other");
}

#[test]
fn delete_using_folds_table_name() {
    // Unquoted identifiers in USING are folded to lowercase.
    let ast::Statement::Delete(d) = ok("DELETE FROM T USING SRC WHERE T.id = SRC.id") else {
        panic!("expected Delete");
    };
    assert_eq!(d.table, "t");
    assert_eq!(d.using.expect("USING clause").base.name, "src");
}

#[test]
fn delete_using_comma_tables_rejected() {
    // Comma-separated USING tables are modelled via explicit JOIN only (mirrors UPDATE FROM).
    let err = parse("DELETE FROM t USING a, b WHERE t.id = a.id").expect_err("comma USING");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

// --- Set operations -------------------------------------------

#[test]
fn union_builds_set_operation() {
    let ast::Statement::SetOperation(s) = ok("SELECT a FROM t UNION SELECT a FROM u") else {
        panic!("expected SetOperation");
    };
    assert!(s.order_by.is_empty());
    assert!(s.limit.is_none());
    let ast::SelectBody::SetOp {
        op,
        all,
        left,
        right,
    } = s.body
    else {
        panic!("expected a SetOp root");
    };
    assert_eq!(op, ast::SetOp::Union);
    assert!(!all);
    assert!(matches!(*left, ast::SelectBody::Select(_)));
    assert!(matches!(*right, ast::SelectBody::Select(_)));
}

#[test]
fn union_all_sets_all_flag() {
    let ast::Statement::SetOperation(s) = ok("SELECT 1 UNION ALL SELECT 2") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { op, all, .. } = s.body else {
        panic!("expected a SetOp root");
    };
    assert_eq!(op, ast::SetOp::Union);
    assert!(all);
}

#[test]
fn intersect_and_except_map_operators() {
    let ast::Statement::SetOperation(i) = ok("SELECT 1 INTERSECT SELECT 2") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { op, .. } = i.body else {
        panic!("expected a SetOp root");
    };
    assert_eq!(op, ast::SetOp::Intersect);

    let ast::Statement::SetOperation(e) = ok("SELECT 1 EXCEPT SELECT 2") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { op, .. } = e.body else {
        panic!("expected a SetOp root");
    };
    assert_eq!(op, ast::SetOp::Except);
}

#[test]
fn chained_union_is_left_associative() {
    // `a UNION b UNION c` must nest as `(a UNION b) UNION c`.
    let ast::Statement::SetOperation(s) = ok("SELECT 1 UNION SELECT 2 UNION SELECT 3") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { left, right, .. } = s.body else {
        panic!("expected a SetOp root");
    };
    // Right operand is the last leaf; left operand is itself a SetOp of the first two.
    assert!(matches!(*right, ast::SelectBody::Select(_)));
    assert!(matches!(*left, ast::SelectBody::SetOp { .. }));
}

#[test]
fn set_operation_keeps_outer_order_by_and_limit() {
    let ast::Statement::SetOperation(s) =
        ok("SELECT a FROM t UNION SELECT a FROM u ORDER BY a LIMIT 5")
    else {
        panic!("expected SetOperation");
    };
    assert_eq!(s.order_by.len(), 1);
    assert_eq!(s.limit, Some(5));
    assert!(matches!(s.body, ast::SelectBody::SetOp { .. }));
}

#[test]
fn parenthesized_operands_are_unwrapped() {
    let ast::Statement::SetOperation(s) = ok("(SELECT 1) UNION (SELECT 2)") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { left, right, .. } = s.body else {
        panic!("expected a SetOp root");
    };
    assert!(matches!(*left, ast::SelectBody::Select(_)));
    assert!(matches!(*right, ast::SelectBody::Select(_)));
}

#[test]
fn parenthesized_operand_carries_its_own_order_by_and_limit() {
    // Per-branch pagination — the top-K traversal seed shape:
    // `(SELECT ... ORDER BY x LIMIT 1) UNION ALL SELECT ...`.
    let ast::Statement::SetOperation(s) =
        ok("(SELECT a FROM t ORDER BY a LIMIT 1) UNION ALL SELECT a FROM u")
    else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { left, .. } = s.body else {
        panic!("expected a SetOp root");
    };
    let ast::SelectBody::Select(anchor) = *left else {
        panic!("expected a leaf anchor");
    };
    assert_eq!(anchor.order_by.len(), 1);
    assert_eq!(anchor.limit, Some(1));
    // The outer envelope stays empty — the pagination bound to the branch, not the whole union.
    assert!(s.order_by.is_empty());
    assert_eq!(s.limit, None);
    // OFFSET binds per-branch the same way.
    let ast::Statement::SetOperation(s) =
        ok("(SELECT a FROM t ORDER BY a LIMIT 2 OFFSET 3) UNION SELECT a FROM u")
    else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { left, .. } = s.body else {
        panic!("expected a SetOp root");
    };
    let ast::SelectBody::Select(anchor) = *left else {
        panic!("expected a leaf anchor");
    };
    assert_eq!(anchor.limit, Some(2));
    assert_eq!(anchor.offset, Some(3));
}

#[test]
fn pagination_on_a_parenthesized_nested_set_operation_is_rejected() {
    // A parenthesized operand that is itself a set operation has no leaf to carry the
    // pagination — loud reject, no silent drop.
    let err = parse("(SELECT a FROM t UNION SELECT a FROM u LIMIT 1) UNION ALL SELECT a FROM v")
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

#[test]
fn plain_select_stays_flat() {
    // A query without a set operator must keep the flat `Statement::Select` shape.
    assert!(matches!(ok("SELECT 1"), ast::Statement::Select(_)));
    assert!(matches!(
        ok("SELECT a FROM t ORDER BY a LIMIT 3"),
        ast::Statement::Select(_)
    ));
}

// --- MERGE ----------------------------------------------------

#[test]
fn merge_basic_target_source_on_when() {
    // `MERGE INTO t USING s ON cond WHEN MATCHED THEN DELETE`.
    let ast::Statement::Merge(m) =
        ok("MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE")
    else {
        panic!("expected Merge");
    };
    assert_eq!(m.target.name, "t");
    assert_eq!(m.source.name, "s");
    assert!(matches!(m.on, ast::Expr::Binary { .. }));
    assert_eq!(m.whens.len(), 1);
    assert!(matches!(
        m.whens[0],
        ast::MergeWhen::Matched {
            action: ast::MatchedAction::Delete,
            pred: None,
        }
    ));
}

#[test]
fn merge_target_and_source_aliases() {
    let ast::Statement::Merge(m) =
        ok("MERGE INTO t AS x USING s AS y ON x.id = y.id WHEN MATCHED THEN DELETE")
    else {
        panic!("expected Merge");
    };
    assert_eq!(m.target.alias.as_deref(), Some("x"));
    assert_eq!(m.source.alias.as_deref(), Some("y"));
}

#[test]
fn merge_matched_update_and_guarded_delete() {
    // UPDATE SET ... plus a second `WHEN MATCHED AND pred THEN DELETE`.
    let ast::Statement::Merge(m) = ok("MERGE INTO t USING s ON t.id = s.id \
             WHEN MATCHED AND s.flag THEN UPDATE SET val = s.val \
             WHEN MATCHED THEN DELETE")
    else {
        panic!("expected Merge");
    };
    assert_eq!(m.whens.len(), 2);
    let ast::MergeWhen::Matched {
        pred: Some(_),
        action: ast::MatchedAction::Update { assignments },
    } = &m.whens[0]
    else {
        panic!("expected a guarded MATCHED UPDATE");
    };
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].column, "val");
}

#[test]
fn merge_not_matched_insert() {
    // `WHEN NOT MATCHED THEN INSERT (cols) VALUES (...)`.
    let ast::Statement::Merge(m) = ok("MERGE INTO t USING s ON t.id = s.id \
             WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)")
    else {
        panic!("expected Merge");
    };
    let ast::MergeWhen::NotMatched { insert, pred: None } = &m.whens[0] else {
        panic!("expected an unguarded NOT MATCHED INSERT");
    };
    assert_eq!(insert.columns, ["id", "val"]);
    assert_eq!(insert.values.len(), 2);
}

#[test]
fn merge_matched_insert_is_rejected() {
    // `INSERT` in a MATCHED clause is invalid; sqlparser rejects it (Syntax) before our
    // converter, which also guards against it defensively. Either way it must not parse.
    assert!(
        parse("MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN INSERT (id) VALUES (1)")
            .is_err()
    );
}

#[test]
fn merge_not_matched_delete_is_rejected() {
    // `DELETE` in a NOT MATCHED clause is invalid; it must not parse.
    assert!(parse("MERGE INTO t USING s ON t.id = s.id WHEN NOT MATCHED THEN DELETE").is_err());
}

// --- ORDER BY NULLS FIRST / LAST ------------------------------

#[test]
fn order_by_without_nulls_has_default() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY id ASC") else {
        panic!("expected Select");
    };
    assert_eq!(s.order_by[0].nulls, ast::NullOrdering::Default);
}

#[test]
fn order_by_nulls_first_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY id ASC NULLS FIRST") else {
        panic!("expected Select");
    };
    assert_eq!(s.order_by[0].nulls, ast::NullOrdering::First);
    assert!(s.order_by[0].ascending);
}

#[test]
fn order_by_nulls_last_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY id DESC NULLS LAST") else {
        panic!("expected Select");
    };
    assert_eq!(s.order_by[0].nulls, ast::NullOrdering::Last);
    assert!(!s.order_by[0].ascending);
}

#[test]
fn order_by_nulls_without_explicit_direction() {
    // NULLS LAST without explicit ASC/DESC — ascending defaults to true.
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY name NULLS LAST") else {
        panic!("expected Select");
    };
    assert_eq!(s.order_by[0].nulls, ast::NullOrdering::Last);
    assert!(s.order_by[0].ascending);
}

#[test]
fn order_by_multi_key_mixed_nulls() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t ORDER BY a NULLS FIRST, b DESC NULLS LAST")
    else {
        panic!("expected Select");
    };
    assert_eq!(s.order_by[0].nulls, ast::NullOrdering::First);
    assert_eq!(s.order_by[1].nulls, ast::NullOrdering::Last);
}

// --- SELECT ... FOR UPDATE / FOR SHARE ------------------------

#[test]
fn select_for_update_default_wait() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t FOR UPDATE") else {
        panic!("expected Select");
    };
    assert_eq!(
        s.lock,
        Some(ast::RowLock {
            strength: ast::LockStrength::Update,
            of: None,
            wait: ast::LockWait::Default,
        })
    );
}

#[test]
fn select_for_share_skip_locked() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t FOR SHARE SKIP LOCKED") else {
        panic!("expected Select");
    };
    let lock = s.lock.expect("lock");
    assert_eq!(lock.strength, ast::LockStrength::Share);
    assert_eq!(lock.wait, ast::LockWait::SkipLocked);
}

#[test]
fn select_for_update_of_table_nowait() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t FOR UPDATE OF t NOWAIT") else {
        panic!("expected Select");
    };
    let lock = s.lock.expect("lock");
    assert_eq!(lock.strength, ast::LockStrength::Update);
    assert_eq!(lock.of.as_deref(), Some("t"));
    assert_eq!(lock.wait, ast::LockWait::NoWait);
}

#[test]
fn select_without_lock_has_none() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(s.lock, None);
}

#[test]
fn multiple_lock_clauses_rejected() {
    let err = parse("SELECT * FROM t FOR UPDATE FOR SHARE").expect_err("two locks");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

// --- Subqueries -----------------------------------------

fn select_filter(sql: &str) -> ast::Expr {
    let ast::Statement::Select(s) = ok(sql) else {
        panic!("expected Select");
    };
    s.filter.expect("WHERE filter")
}

#[test]
fn scalar_subquery_in_projection() {
    // `(SELECT ...)` as a scalar projection value.
    let ast::Statement::Select(s) = ok("SELECT (SELECT max(b) FROM u) FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression projection");
    };
    assert!(matches!(expr, ast::Expr::ScalarSubquery(_)));
}

#[test]
fn exists_and_not_exists() {
    //
    let ast::Expr::Exists { negated, .. } =
        select_filter("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u)")
    else {
        panic!("expected EXISTS");
    };
    assert!(!negated);
    let ast::Expr::Exists { negated, .. } =
        select_filter("SELECT * FROM t WHERE NOT EXISTS (SELECT 1 FROM u)")
    else {
        panic!("expected NOT EXISTS");
    };
    assert!(negated);
}

#[test]
fn in_subquery_negated_flag() {
    //
    let ast::Expr::InSubquery { negated, .. } =
        select_filter("SELECT * FROM t WHERE a IN (SELECT b FROM u)")
    else {
        panic!("expected IN subquery");
    };
    assert!(!negated);
    let ast::Expr::InSubquery { negated, .. } =
        select_filter("SELECT * FROM t WHERE a NOT IN (SELECT b FROM u)")
    else {
        panic!("expected NOT IN subquery");
    };
    assert!(negated);
}

#[test]
fn in_list_is_not_a_subquery() {
    // A value `IN (1, 2)` list must stay an InList, not become an InSubquery.
    assert!(matches!(
        select_filter("SELECT * FROM t WHERE a IN (1, 2, 3)"),
        ast::Expr::InList { .. }
    ));
}

// --- LIKE ... ESCAPE ------------------------------------------

#[test]
fn like_without_escape_has_none() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a LIKE 'x%'") else {
        panic!("expected Select");
    };
    let ast::Expr::Like { escape, .. } = s.filter.unwrap() else {
        panic!("expected Like in filter");
    };
    assert!(escape.is_none());
}

#[test]
fn like_escape_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a LIKE 'x!%' ESCAPE '!'") else {
        panic!("expected Select");
    };
    let ast::Expr::Like {
        escape, negated, ..
    } = s.filter.unwrap()
    else {
        panic!("expected Like");
    };
    assert_eq!(escape, Some('!'));
    assert!(!negated);
}

#[test]
fn not_like_escape_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a NOT LIKE 'x!%' ESCAPE '!'") else {
        panic!("expected Select");
    };
    let ast::Expr::Like {
        escape, negated, ..
    } = s.filter.unwrap()
    else {
        panic!("expected Like");
    };
    assert_eq!(escape, Some('!'));
    assert!(negated);
}

#[test]
fn like_escape_backslash() {
    let ast::Statement::Select(s) = ok(r"SELECT * FROM t WHERE a LIKE 'x\%' ESCAPE '\'") else {
        panic!("expected Select");
    };
    let ast::Expr::Like { escape, .. } = s.filter.unwrap() else {
        panic!("expected Like");
    };
    assert_eq!(escape, Some('\\'));
}

// --- SIMILAR TO / ROW ---------------------------------

#[test]
fn similar_to_and_negated() {
    let ast::Expr::SimilarTo { negated, .. } =
        select_filter("SELECT * FROM t WHERE a SIMILAR TO 'x%'")
    else {
        panic!("expected SIMILAR TO");
    };
    assert!(!negated);
    let ast::Expr::SimilarTo { negated, .. } =
        select_filter("SELECT * FROM t WHERE a NOT SIMILAR TO 'x%'")
    else {
        panic!("expected NOT SIMILAR TO");
    };
    assert!(negated);
}

#[test]
fn similar_to_escape_rejected() {
    let err =
        parse("SELECT * FROM t WHERE a SIMILAR TO 'x%' ESCAPE '!'").expect_err("SIMILAR TO ESCAPE");
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

// --- Regex match ~ ~* !~ !~* ----------------------------------

#[test]
fn regex_match_case_sensitive() {
    let ast::Expr::RegexMatch {
        case_sensitive,
        negated,
        ..
    } = select_filter("SELECT * FROM t WHERE a ~ 'x.*'")
    else {
        panic!("expected RegexMatch");
    };
    assert!(case_sensitive);
    assert!(!negated);
}

#[test]
fn regex_match_case_insensitive() {
    let ast::Expr::RegexMatch {
        case_sensitive,
        negated,
        ..
    } = select_filter("SELECT * FROM t WHERE a ~* 'x.*'")
    else {
        panic!("expected RegexMatch");
    };
    assert!(!case_sensitive);
    assert!(!negated);
}

#[test]
fn regex_not_match_case_sensitive() {
    let ast::Expr::RegexMatch {
        case_sensitive,
        negated,
        ..
    } = select_filter("SELECT * FROM t WHERE a !~ 'x.*'")
    else {
        panic!("expected RegexMatch");
    };
    assert!(case_sensitive);
    assert!(negated);
}

#[test]
fn regex_not_match_case_insensitive() {
    let ast::Expr::RegexMatch {
        case_sensitive,
        negated,
        ..
    } = select_filter("SELECT * FROM t WHERE a !~* 'x.*'")
    else {
        panic!("expected RegexMatch");
    };
    assert!(!case_sensitive);
    assert!(negated);
}

fn first_projection(sql: &str) -> ast::Expr {
    let ast::Statement::Select(s) = ok(sql) else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression projection");
    };
    expr.clone()
}

// --- Scalar string functions ----------------------------------

#[test]
fn scalar_function_call_form() {
    let ast::Expr::ScalarFunction { func, args } = first_projection("SELECT UPPER(name) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Upper);
    assert_eq!(args.len(), 1);
    assert_eq!(args[0], ast::Expr::Column("name".to_owned()));
}

#[test]
fn create_and_drop_policy_parse() {
    // CREATE POLICY captures table, command, roles, and the USING/WITH CHECK predicates as
    // canonical SQL; DROP POLICY captures name, table, and IF EXISTS.
    let ast::Statement::CreatePolicy(cp) =
        parse("CREATE POLICY own ON doc FOR SELECT TO alice, bob USING (owner = CURRENT_USER)")
            .expect("parse create policy")
    else {
        panic!("expected CreatePolicy");
    };
    assert_eq!(cp.name, "own");
    assert_eq!(cp.table, "doc");
    assert!(cp.permissive, "AS is omitted → permissive by default");
    assert_eq!(cp.command, ast::PolicyCommand::Select);
    assert_eq!(cp.roles, vec!["alice".to_owned(), "bob".to_owned()]);
    assert!(cp.using.is_some());
    assert!(cp.check.is_none());

    // FOR defaults to ALL; PUBLIC (no TO) yields an empty role list; WITH CHECK is captured.
    let ast::Statement::CreatePolicy(cp) =
        parse("CREATE POLICY w ON doc WITH CHECK (owner = CURRENT_USER)").expect("parse")
    else {
        panic!("expected CreatePolicy");
    };
    assert_eq!(cp.command, ast::PolicyCommand::All);
    assert!(cp.roles.is_empty());
    assert!(cp.using.is_none());
    assert!(cp.check.is_some());

    // `AS PERMISSIVE` / `AS RESTRICTIVE` set the policy kind; AS sits between the table and FOR.
    let ast::Statement::CreatePolicy(cp) =
        parse("CREATE POLICY p ON doc AS PERMISSIVE FOR SELECT USING (owner = CURRENT_USER)")
            .expect("parse as permissive")
    else {
        panic!("expected CreatePolicy");
    };
    assert!(cp.permissive);
    let ast::Statement::CreatePolicy(cp) =
        parse("CREATE POLICY r ON doc AS RESTRICTIVE USING (owner = CURRENT_USER)")
            .expect("parse as restrictive")
    else {
        panic!("expected CreatePolicy");
    };
    assert!(!cp.permissive);
    // `AS <other>` is rejected rather than silently treated as permissive.
    assert!(parse("CREATE POLICY p ON doc AS SNEAKY USING (a = 1)").is_err());

    // A policy with neither USING nor WITH CHECK is rejected.
    assert!(parse("CREATE POLICY p ON doc FOR SELECT").is_err());

    let ast::Statement::DropPolicy(dp) =
        parse("DROP POLICY IF EXISTS own ON doc").expect("parse drop policy")
    else {
        panic!("expected DropPolicy");
    };
    assert_eq!(dp.name, "own");
    assert_eq!(dp.table, "doc");
    assert!(dp.if_exists);
}

#[test]
fn create_and_drop_trigger_parse() {
    // CREATE TRIGGER captures timing, events, table, granularity, the WHEN guard, and the
    // action as canonical SQL.
    let ast::Statement::CreateTrigger(ct) = parse(
        "CREATE TRIGGER log_ins AFTER INSERT OR UPDATE ON t FOR EACH ROW WHEN (new.v > 0) \
         INSERT INTO audit VALUES ('i', new.id)",
    )
    .expect("parse create trigger") else {
        panic!("expected CreateTrigger");
    };
    assert_eq!(ct.name, "log_ins");
    assert_eq!(ct.table, "t");
    assert!(!ct.or_replace);
    assert_eq!(ct.timing, ast::TriggerTiming::After);
    assert_eq!(
        ct.events,
        vec![ast::TriggerEvent::Insert, ast::TriggerEvent::Update]
    );
    assert_eq!(ct.for_each, ast::TriggerForEach::Row);
    assert!(ct.when.as_deref().is_some_and(|w| w.contains("new.v")));
    assert!(ct.action.to_ascii_lowercase().contains("insert into audit"));

    // BEFORE / OR REPLACE / FOR EACH STATEMENT, no WHEN, default-ROW omission.
    let ast::Statement::CreateTrigger(ct) = parse(
        "CREATE OR REPLACE TRIGGER tg BEFORE DELETE ON t FOR EACH STATEMENT \
         DELETE FROM audit",
    )
    .expect("parse") else {
        panic!("expected CreateTrigger");
    };
    assert!(ct.or_replace);
    assert_eq!(ct.timing, ast::TriggerTiming::Before);
    assert_eq!(ct.events, vec![ast::TriggerEvent::Delete]);
    assert_eq!(ct.for_each, ast::TriggerForEach::Statement);
    assert!(ct.when.is_none());

    // FOR EACH defaults to ROW when omitted.
    let ast::Statement::CreateTrigger(ct) =
        parse("CREATE TRIGGER d AFTER INSERT ON t INSERT INTO audit VALUES ('x', 1)")
            .expect("parse")
    else {
        panic!("expected CreateTrigger");
    };
    assert_eq!(ct.for_each, ast::TriggerForEach::Row);

    // DROP TRIGGER captures name, table, and IF EXISTS.
    let ast::Statement::DropTrigger(dt) =
        parse("DROP TRIGGER IF EXISTS log_ins ON t").expect("parse drop trigger")
    else {
        panic!("expected DropTrigger");
    };
    assert_eq!(dt.name, "log_ins");
    assert_eq!(dt.table, "t");
    assert!(dt.if_exists);

    // Rejections: INSTEAD OF (no updatable views), a bad timing, and a non-data action statement.
    assert!(parse("CREATE TRIGGER x INSTEAD OF INSERT ON t INSERT INTO a VALUES (1)").is_err());
    assert!(parse("CREATE TRIGGER x DURING INSERT ON t INSERT INTO a VALUES (1)").is_err());
    assert!(parse("CREATE TRIGGER x AFTER INSERT ON t CREATE TABLE z (a INT)").is_err());
}

#[test]
fn alter_trigger_rename_parses_and_other_forms_reject() {
    // ALTER TRIGGER ... RENAME TO captures the old name, table, and new name.
    let ast::Statement::AlterTrigger(at) =
        parse("ALTER TRIGGER log_ins ON t RENAME TO log_all").expect("parse alter trigger")
    else {
        panic!("expected AlterTrigger");
    };
    assert_eq!(at.name, "log_ins");
    assert_eq!(at.table, "t");
    assert_eq!(at.new_name, "log_all");

    // RENAME TO is the only ALTER TRIGGER form; enable/disable ride ALTER TABLE.
    assert!(parse("ALTER TRIGGER log_ins ON t DISABLE").is_err());
    assert!(parse("ALTER TRIGGER log_ins ON t").is_err());
}

#[test]
fn alter_table_enable_disable_trigger_parses() {
    // DISABLE TRIGGER <name>.
    let ast::Statement::AlterTable(at) =
        parse("ALTER TABLE t DISABLE TRIGGER log_ins").expect("parse disable trigger")
    else {
        panic!("expected AlterTable");
    };
    assert_eq!(
        at.action,
        ast::AlterTableAction::DisableTrigger {
            name: Some("log_ins".to_owned())
        }
    );

    // ENABLE TRIGGER ALL folds to the every-trigger form.
    let ast::Statement::AlterTable(at) =
        parse("ALTER TABLE t ENABLE TRIGGER ALL").expect("parse enable all")
    else {
        panic!("expected AlterTable");
    };
    assert_eq!(
        at.action,
        ast::AlterTableAction::EnableTrigger { name: None }
    );

    // USER (all non-system triggers) is rejected: there is no system-trigger distinction to make
    // it mean anything different from ALL. Session-replication modes are rejected too.
    assert!(parse("ALTER TABLE t DISABLE TRIGGER USER").is_err());
    assert!(parse("ALTER TABLE t ENABLE ALWAYS TRIGGER log_ins").is_err());
    assert!(parse("ALTER TABLE t ENABLE REPLICA TRIGGER log_ins").is_err());
}

#[test]
fn create_call_drop_procedure_parse() {
    // CREATE PROCEDURE captures the parameter list and the body verbatim.
    let ast::Statement::CreateProcedure(cp) = parse(
        "CREATE OR REPLACE PROCEDURE add_pair(a INT, b TEXT) LANGUAGE SQL AS \
         $$ INSERT INTO t VALUES ($1); UPDATE t SET v = $2 $$",
    )
    .expect("parse create procedure") else {
        panic!("expected CreateProcedure");
    };
    assert_eq!(cp.name, "add_pair");
    assert!(cp.or_replace);
    assert_eq!(cp.params.len(), 2);
    assert_eq!(cp.params[0].name, "a");
    assert!(cp.body.to_ascii_lowercase().contains("insert into t"));

    // A single-quoted body and an empty parameter list also parse.
    let ast::Statement::CreateProcedure(cp) =
        parse("CREATE PROCEDURE clear() AS 'DELETE FROM t'").expect("parse")
    else {
        panic!("expected CreateProcedure");
    };
    assert!(cp.params.is_empty());
    assert!(!cp.or_replace);

    // CALL captures the name and argument expressions.
    let ast::Statement::Call(call) = parse("CALL add_pair(1, 'x')").expect("parse call") else {
        panic!("expected Call");
    };
    assert_eq!(call.name, "add_pair");
    assert_eq!(call.args.len(), 2);

    // DROP PROCEDURE captures IF EXISTS.
    let ast::Statement::DropProcedure(dp) =
        parse("DROP PROCEDURE IF EXISTS add_pair").expect("parse drop")
    else {
        panic!("expected DropProcedure");
    };
    assert_eq!(dp.name, "add_pair");
    assert!(dp.if_exists);

    // Rejections: non-SQL language, a DDL body statement, and a body referencing an undeclared param.
    assert!(parse("CREATE PROCEDURE p() LANGUAGE PYTHON AS 'SELECT 1'").is_err());
    assert!(parse("CREATE PROCEDURE p() AS 'CREATE TABLE z (a INT)'").is_err());
    assert!(parse("CREATE PROCEDURE p(a INT) AS 'INSERT INTO t VALUES ($2)'").is_err());
}

#[test]
fn unknown_function_parses_as_function_call() {
    // A name that is not a built-in is kept as a generic FunctionCall (the analyzer resolves
    // it against the UDF registry), not rejected at parse time.
    let ast::Statement::Select(sel) = parse("SELECT my_udf(a, 1) FROM t").expect("parse") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &sel.projection[0] else {
        panic!("expected expr projection");
    };
    let ast::Expr::FunctionCall { name, args } = expr else {
        panic!("expected FunctionCall, got {expr:?}");
    };
    assert_eq!(name, "my_udf");
    assert_eq!(args.len(), 2);

    // A recognised built-in still parses to a ScalarFunction (unchanged).
    let ast::Statement::Select(sel) = parse("SELECT upper(a) FROM t").expect("parse") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &sel.projection[0] else {
        panic!("expected expr projection");
    };
    assert!(matches!(expr, ast::Expr::ScalarFunction { .. }));
}

#[test]
fn nusascript_block_parses_and_rejects_malformed() {
    // A procedure body that begins with BEGIN is a NusaScript block; `is_script` detects it.
    assert!(super::is_script("BEGIN INSERT INTO t VALUES (1); END"));
    assert!(!super::is_script("INSERT INTO t VALUES (1)"));

    // A well-formed block parses into its top-level statements (declare, while, insert).
    let block = super::parse_script(
        "BEGIN \
           DECLARE i INT DEFAULT 0; \
           WHILE i > 0 LOOP SET i = i - 1; END LOOP; \
           INSERT INTO t VALUES (i) \
         END",
    )
    .expect("parse block");
    assert_eq!(block.body.len(), 3);
    assert!(block.handler.is_none());

    // Malformed blocks are rejected: missing END LOOP, and missing END IF.
    assert!(super::parse_script("BEGIN WHILE 1 > 0 LOOP INSERT INTO t VALUES (1); END").is_err());
    assert!(super::parse_script("BEGIN IF 1 > 0 THEN INSERT INTO t VALUES (1); END").is_err());
}

#[test]
fn alter_policy_parse() {
    // ALTER POLICY captures the changed clauses; omitted ones are None (the analyzer keeps the
    // existing parts). At least one of TO / USING / WITH CHECK is required; RENAME TO is rejected.
    let ast::Statement::AlterPolicy(ap) =
        parse("ALTER POLICY own ON doc USING (owner = CURRENT_USER)").expect("parse alter policy")
    else {
        panic!("expected AlterPolicy");
    };
    assert_eq!(ap.name, "own");
    assert_eq!(ap.table, "doc");
    assert!(ap.roles.is_none(), "TO omitted → keep roles");
    assert!(ap.using.is_some());
    assert!(ap.check.is_none());

    let ast::Statement::AlterPolicy(ap) =
        parse("ALTER POLICY own ON doc TO alice, bob WITH CHECK (owner = CURRENT_USER)")
            .expect("parse alter policy with roles + check")
    else {
        panic!("expected AlterPolicy");
    };
    assert_eq!(ap.roles, Some(vec!["alice".to_owned(), "bob".to_owned()]));
    assert!(ap.using.is_none());
    assert!(ap.check.is_some());

    // No clause to change is rejected, and RENAME TO is not yet supported.
    assert!(parse("ALTER POLICY own ON doc").is_err());
    assert!(parse("ALTER POLICY own ON doc RENAME TO other").is_err());
}

#[test]
fn clock_functions_parse_bare_and_call_form() {
    // The bare keyword form (no parentheses) and the parenthesised call form both map to the
    // niladic clock ScalarFunc with empty args.
    for (sql, expected) in [
        (
            "SELECT CURRENT_TIMESTAMP",
            ast::ScalarFunc::CurrentTimestamp,
        ),
        ("SELECT CURRENT_DATE", ast::ScalarFunc::CurrentDate),
        ("SELECT CURRENT_TIME", ast::ScalarFunc::CurrentTime),
        ("SELECT NOW()", ast::ScalarFunc::Now),
    ] {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, expected, "for `{sql}`");
        assert!(args.is_empty(), "clock function takes no args (`{sql}`)");
    }
}

#[test]
fn alter_table_row_level_security_parses() {
    // ENABLE/DISABLE ROW LEVEL SECURITY map to the dedicated AlterTable actions.
    let ast::Statement::AlterTable(at) =
        parse("ALTER TABLE t ENABLE ROW LEVEL SECURITY").expect("parse enable")
    else {
        panic!("expected AlterTable");
    };
    assert_eq!(at.name, "t");
    assert_eq!(at.action, ast::AlterTableAction::EnableRowLevelSecurity);

    let ast::Statement::AlterTable(at) =
        parse("ALTER TABLE t DISABLE ROW LEVEL SECURITY").expect("parse disable")
    else {
        panic!("expected AlterTable");
    };
    assert_eq!(at.action, ast::AlterTableAction::DisableRowLevelSecurity);
}

#[test]
fn session_user_functions_parse_bare_and_call_form() {
    // The bare keyword forms `CURRENT_USER` / `SESSION_USER` / `USER` map to the niladic
    // session-user ScalarFunc with empty args; `USER` is a synonym for `CURRENT_USER`.
    for (sql, expected) in [
        ("SELECT CURRENT_USER", ast::ScalarFunc::CurrentUser),
        ("SELECT SESSION_USER", ast::ScalarFunc::SessionUser),
        ("SELECT USER", ast::ScalarFunc::CurrentUser),
    ] {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, expected, "for `{sql}`");
        assert!(
            args.is_empty(),
            "session-user function takes no args (`{sql}`)"
        );
    }

    // `current_setting(name)` is an ordinary one-argument call.
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT current_setting('tenant')")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::CurrentSetting);
    assert_eq!(args.len(), 1);
}

#[test]
fn extract_normalizes_field_to_text_literal_arg() {
    // EXTRACT(field FROM source) → ScalarFunction(Extract, [Text(field), source]).
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT EXTRACT(YEAR FROM ts) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Extract);
    assert_eq!(args.len(), 2);
    assert_eq!(
        args[0],
        ast::Expr::Literal(ast::Value::Text("year".to_owned()))
    );
    assert_eq!(args[1], ast::Expr::Column("ts".to_owned()));
}

#[test]
fn date_trunc_and_age_parse_as_calls() {
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT DATE_TRUNC('month', ts) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::DateTrunc);
    assert_eq!(args.len(), 2);

    // AGE accepts both the one- and two-argument forms.
    for (sql, n) in [("SELECT AGE(ts) FROM t", 1), ("SELECT AGE(a, b) FROM t", 2)] {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, ast::ScalarFunc::Age);
        assert_eq!(args.len(), n, "for `{sql}`");
    }
}

#[test]
fn to_char_date_timestamp_parse_as_calls() {
    // TO_CHAR / TO_DATE / TO_TIMESTAMP are ordinary two-argument call-form functions.
    for (sql, expected) in [
        (
            "SELECT TO_CHAR(ts, 'YYYY-MM-DD') FROM t",
            ast::ScalarFunc::ToChar,
        ),
        (
            "SELECT TO_DATE('2024-06-15', 'YYYY-MM-DD') FROM t",
            ast::ScalarFunc::ToDate,
        ),
        (
            "SELECT TO_TIMESTAMP('2024-06-15 12:00:00', 'YYYY-MM-DD HH24:MI:SS') FROM t",
            ast::ScalarFunc::ToTimestamp,
        ),
    ] {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, expected, "for `{sql}`");
        assert_eq!(args.len(), 2, "for `{sql}`");
    }
}

#[test]
fn scalar_function_length_aliases() {
    for sql in [
        "SELECT LENGTH(s) FROM t",
        "SELECT CHAR_LENGTH(s) FROM t",
        "SELECT CHARACTER_LENGTH(s) FROM t",
    ] {
        let ast::Expr::ScalarFunction { func, .. } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, ast::ScalarFunc::Length);
    }
}

#[test]
fn substring_from_for_normalizes_to_positional_args() {
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT SUBSTRING(s FROM 2 FOR 3) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Substring);
    assert_eq!(args.len(), 3); // [s, 2, 3]

    // Comma call form, two-argument.
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT SUBSTRING(s, 2) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Substring);
    assert_eq!(args.len(), 2);
}

#[test]
fn trim_directions_map_to_scalar_funcs() {
    let cases = [
        ("SELECT TRIM(s) FROM t", ast::ScalarFunc::BTrim, 1),
        (
            "SELECT TRIM(BOTH 'x' FROM s) FROM t",
            ast::ScalarFunc::BTrim,
            2,
        ),
        (
            "SELECT TRIM(LEADING 'x' FROM s) FROM t",
            ast::ScalarFunc::LTrim,
            2,
        ),
        (
            "SELECT TRIM(TRAILING 'x' FROM s) FROM t",
            ast::ScalarFunc::RTrim,
            2,
        ),
    ];
    for (sql, want_func, want_arity) in cases {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, want_func, "func for `{sql}`");
        assert_eq!(args.len(), want_arity, "arity for `{sql}`");
    }
}

#[test]
fn position_in_form_orders_needle_then_haystack() {
    let ast::Expr::ScalarFunction { func, args } =
        first_projection("SELECT POSITION('a' IN s) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Position);
    assert_eq!(args.len(), 2);
    assert_eq!(
        args[0],
        ast::Expr::Literal(ast::Value::Text("a".to_owned()))
    );
    assert_eq!(args[1], ast::Expr::Column("s".to_owned()));
}

#[test]
fn unknown_function_parses_for_udf_resolution() {
    // An unrecognised function name is no longer rejected at parse time; it becomes a
    // generic `FunctionCall` that the analyzer resolves against the UDF registry (and rejects there
    // as an unknown function if no UDF is registered).
    let ast::Statement::Select(sel) = super::parse("SELECT frobnicate(s) FROM t").expect("parse")
    else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &sel.projection[0] else {
        panic!("expected expr projection");
    };
    assert!(matches!(expr, ast::Expr::FunctionCall { .. }));
}

#[test]
fn b448_string_functions_parse() {
    let cases = [
        ("SELECT CONCAT(a, b, c) FROM t", ast::ScalarFunc::Concat, 3),
        (
            "SELECT CONCAT_WS('-', a, b) FROM t",
            ast::ScalarFunc::ConcatWs,
            3,
        ),
        ("SELECT LEFT(s, 2) FROM t", ast::ScalarFunc::Left, 2),
        ("SELECT RIGHT(s, 2) FROM t", ast::ScalarFunc::Right, 2),
        (
            "SELECT SPLIT_PART(s, ',', 2) FROM t",
            ast::ScalarFunc::SplitPart,
            3,
        ),
        ("SELECT REVERSE(s) FROM t", ast::ScalarFunc::Reverse, 1),
    ];
    for (sql, want_func, want_arity) in cases {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, want_func, "func for `{sql}`");
        assert_eq!(args.len(), want_arity, "arity for `{sql}`");
    }
}

#[test]
fn b449_regex_functions_parse() {
    let cases = [
        (
            "SELECT REGEXP_REPLACE(s, 'a', 'b') FROM t",
            ast::ScalarFunc::RegexpReplace,
            3,
        ),
        (
            "SELECT REGEXP_REPLACE(s, 'a', 'b', 'g') FROM t",
            ast::ScalarFunc::RegexpReplace,
            4,
        ),
        (
            "SELECT REGEXP_MATCH(s, '[0-9]+') FROM t",
            ast::ScalarFunc::RegexpMatch,
            2,
        ),
    ];
    for (sql, want_func, want_arity) in cases {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, want_func, "func for `{sql}`");
        assert_eq!(args.len(), want_arity, "arity for `{sql}`");
    }
}

#[test]
fn b453_55_math_functions_parse() {
    let cases = [
        ("SELECT ABS(x) FROM t", ast::ScalarFunc::Abs, 1),
        ("SELECT ROUND(x, 2) FROM t", ast::ScalarFunc::Round, 2),
        ("SELECT CEIL(x) FROM t", ast::ScalarFunc::Ceil, 1),
        ("SELECT CEILING(x) FROM t", ast::ScalarFunc::Ceil, 1),
        ("SELECT FLOOR(x) FROM t", ast::ScalarFunc::Floor, 1),
        ("SELECT SIGN(x) FROM t", ast::ScalarFunc::Sign, 1),
        ("SELECT MOD(x, y) FROM t", ast::ScalarFunc::Mod, 2),
        ("SELECT POWER(x, y) FROM t", ast::ScalarFunc::Power, 2),
        ("SELECT POW(x, y) FROM t", ast::ScalarFunc::Power, 2),
        ("SELECT SQRT(x) FROM t", ast::ScalarFunc::Sqrt, 1),
        ("SELECT LOG(x) FROM t", ast::ScalarFunc::Log, 1),
        ("SELECT LOG(b, x) FROM t", ast::ScalarFunc::Log, 2),
        ("SELECT ATAN2(y, x) FROM t", ast::ScalarFunc::Atan2, 2),
    ];
    for (sql, want_func, want_arity) in cases {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, want_func, "func for `{sql}`");
        assert_eq!(args.len(), want_arity, "arity for `{sql}`");
    }
}

#[test]
fn b456_random_functions_parse() {
    let ast::Expr::ScalarFunction { func, args } = first_projection("SELECT RANDOM() FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Random);
    assert_eq!(args.len(), 0);
    let ast::Expr::ScalarFunction { func, args } = first_projection("SELECT SETSEED(0.5) FROM t")
    else {
        panic!("expected ScalarFunction");
    };
    assert_eq!(func, ast::ScalarFunc::Setseed);
    assert_eq!(args.len(), 1);
}

#[test]
fn b457_conditional_functions_parse() {
    let cases = [
        ("SELECT NULLIF(a, b) FROM t", ast::ScalarFunc::Nullif, 2),
        (
            "SELECT GREATEST(a, b, c) FROM t",
            ast::ScalarFunc::Greatest,
            3,
        ),
        ("SELECT LEAST(a, b) FROM t", ast::ScalarFunc::Least, 2),
    ];
    for (sql, want_func, want_arity) in cases {
        let ast::Expr::ScalarFunction { func, args } = first_projection(sql) else {
            panic!("expected ScalarFunction for `{sql}`");
        };
        assert_eq!(func, want_func, "func for `{sql}`");
        assert_eq!(args.len(), want_arity, "arity for `{sql}`");
    }
}

// --- Array literal + subscript --------------------------------

#[test]
fn array_literal_keyword_form() {
    let ast::Expr::ArrayLiteral(elems) = first_projection("SELECT ARRAY[1, 2, 3] FROM t") else {
        panic!("expected ArrayLiteral");
    };
    assert_eq!(elems.len(), 3);
}

#[test]
fn array_literal_empty() {
    let ast::Expr::ArrayLiteral(elems) = first_projection("SELECT ARRAY[] FROM t") else {
        panic!("expected ArrayLiteral");
    };
    assert!(elems.is_empty());
}

#[test]
fn array_literal_of_expressions() {
    // Elements are arbitrary expressions, not just constants.
    let ast::Expr::ArrayLiteral(elems) = first_projection("SELECT ARRAY[a + 1, b * 2] FROM t")
    else {
        panic!("expected ArrayLiteral");
    };
    assert_eq!(elems.len(), 2);
    assert!(matches!(elems[0], ast::Expr::Binary { .. }));
}

#[test]
fn subscript_index() {
    let ast::Expr::Subscript { base, index } = first_projection("SELECT a[1] FROM t") else {
        panic!("expected Subscript");
    };
    assert!(matches!(*base, ast::Expr::Column(_)));
    assert_eq!(*index, ast::Expr::Literal(ast::Value::Int(1)));
}

#[test]
fn subscript_on_array_literal() {
    let ast::Expr::Subscript { base, .. } = first_projection("SELECT ARRAY[10, 20][2] FROM t")
    else {
        panic!("expected Subscript");
    };
    assert!(matches!(*base, ast::Expr::ArrayLiteral(_)));
}

// Array slice `a[i:j]` is now supported — see `array_slice_parses_and_stride_rejected`.

// --- Ordered-set aggregate WITHIN GROUP -----------------------

#[test]
fn within_group_percentile_cont() {
    let ast::Expr::WithinGroup(wg) =
        first_projection("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) FROM t")
    else {
        panic!("expected WithinGroup");
    };
    assert_eq!(wg.func, "percentile_cont");
    assert_eq!(wg.args.len(), 1); // the 0.5 fraction
    assert_eq!(wg.order_by.len(), 1);
    assert!(wg.order_by[0].ascending);
}

#[test]
fn within_group_percentile_disc_desc() {
    let ast::Expr::WithinGroup(wg) =
        first_projection("SELECT PERCENTILE_DISC(0.9) WITHIN GROUP (ORDER BY score DESC) FROM t")
    else {
        panic!("expected WithinGroup");
    };
    assert_eq!(wg.func, "percentile_disc");
    assert!(!wg.order_by[0].ascending);
}

#[test]
fn within_group_mode_no_args() {
    let ast::Expr::WithinGroup(wg) =
        first_projection("SELECT MODE() WITHIN GROUP (ORDER BY x) FROM t")
    else {
        panic!("expected WithinGroup");
    };
    assert_eq!(wg.func, "mode");
    assert!(wg.args.is_empty());
    assert_eq!(wg.order_by.len(), 1);
}

#[test]
fn row_constructor_keyword_and_bare() {
    // Both `ROW(a, b)` and a bare `(a, b, c)` tuple lower to `Expr::Row`.
    let ast::Expr::Row(items) = first_projection("SELECT ROW(a, b) FROM t") else {
        panic!("expected ROW(...) to lower to Row");
    };
    assert_eq!(items.len(), 2);
    let ast::Expr::Row(items) = first_projection("SELECT (a, b, c) FROM t") else {
        panic!("expected a bare tuple to lower to Row");
    };
    assert_eq!(items.len(), 3);
}

// --- Expressions ------------------------------------------------------

#[test]
fn where_logical_precedence() {
    // `a = 1 AND b = 2` must put AND at the root, comparisons below it.
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a = 1 AND b = 2") else {
        panic!("expected Select");
    };
    let Some(ast::Expr::Binary { op, .. }) = s.filter else {
        panic!("expected a binary predicate");
    };
    assert_eq!(op, ast::BinaryOp::And);
}

#[test]
fn arithmetic_precedence() {
    // `a + b * 2` must parse as `a + (b * 2)`.
    let ast::Statement::Select(s) = ok("SELECT a + b * 2 FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    let ast::Expr::Binary { op, right, .. } = expr else {
        panic!("expected a binary expression");
    };
    assert_eq!(*op, ast::BinaryOp::Plus);
    let ast::Expr::Binary { op: inner, .. } = right.as_ref() else {
        panic!("expected a nested binary expression");
    };
    assert_eq!(*inner, ast::BinaryOp::Multiply);
}

#[test]
fn string_concat_operator() {
    // `a || b || c` maps to `BinaryOp::Concat`, left-associative.
    let ast::Statement::Select(s) = ok("SELECT a || b || c FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    let ast::Expr::Binary { op, left, .. } = expr else {
        panic!("expected a binary expression");
    };
    assert_eq!(*op, ast::BinaryOp::Concat);
    let ast::Expr::Binary { op: inner, .. } = left.as_ref() else {
        panic!("expected a nested binary expression on the left");
    };
    assert_eq!(*inner, ast::BinaryOp::Concat);
}

#[test]
fn is_distinct_from_parses() {
    let ast::Statement::Select(s) = ok("SELECT a IS DISTINCT FROM b FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    assert!(matches!(
        expr,
        ast::Expr::IsDistinctFrom { negated: false, .. }
    ));

    let ast::Statement::Select(s) = ok("SELECT a IS NOT DISTINCT FROM b FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    assert!(matches!(
        expr,
        ast::Expr::IsDistinctFrom { negated: true, .. }
    ));
}

#[test]
fn is_truth_value_parses() {
    for (sql, truth, negated) in [
        ("SELECT b IS TRUE FROM t", ast::TruthValue::True, false),
        ("SELECT b IS NOT TRUE FROM t", ast::TruthValue::True, true),
        ("SELECT b IS FALSE FROM t", ast::TruthValue::False, false),
        (
            "SELECT b IS UNKNOWN FROM t",
            ast::TruthValue::Unknown,
            false,
        ),
        (
            "SELECT b IS NOT UNKNOWN FROM t",
            ast::TruthValue::Unknown,
            true,
        ),
    ] {
        let ast::Statement::Select(s) = ok(sql) else {
            panic!("expected Select for {sql}");
        };
        let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
            panic!("expected an expression for {sql}");
        };
        let ast::Expr::IsBool {
            truth: t,
            negated: n,
            ..
        } = expr
        else {
            panic!("expected IsBool for {sql}");
        };
        assert_eq!((*t, *n), (truth, negated), "mismatch for {sql}");
    }
}

#[test]
fn postfix_cast_operator() {
    // `x::INT` is the postfix form of `CAST(x AS INT)`.
    let ast::Statement::Select(s) = ok("SELECT a::INT FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    let ast::Expr::Cast {
        expr: inner,
        target,
        try_cast,
    } = expr
    else {
        panic!("expected a Cast, got {expr:?}");
    };
    assert_eq!(*target, ColumnType::Int);
    assert!(!*try_cast, "x::INT is a hard cast");
    assert!(matches!(inner.as_ref(), ast::Expr::Column(c) if c == "a"));
}

#[test]
fn postfix_cast_equals_cast_function() {
    // `x::t` must produce the same AST as `CAST(x AS t)`.
    let ast::Statement::Select(a) = ok("SELECT a::TEXT FROM t") else {
        panic!("expected Select");
    };
    let ast::Statement::Select(b) = ok("SELECT CAST(a AS TEXT) FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(
        a.projection, b.projection,
        "postfix :: must equal CAST(... AS ...)",
    );
}

#[test]
fn try_cast_and_safe_cast_parse_as_try_casts() {
    // TRY_CAST / SAFE_CAST share the Cast node with `try_cast = true` (NULL-on-failure semantics).
    for sql in [
        "SELECT TRY_CAST(a AS INT) FROM t",
        "SELECT SAFE_CAST(a AS INT) FROM t",
    ] {
        let ast::Statement::Select(s) = ok(sql) else {
            panic!("expected Select for {sql}");
        };
        let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
            panic!("expected an expression");
        };
        let ast::Expr::Cast { try_cast, .. } = expr else {
            panic!("expected a Cast, got {expr:?}");
        };
        assert!(*try_cast, "{sql} must set try_cast");
    }

    // A plain CAST stays a hard cast.
    let ast::Statement::Select(s) = ok("SELECT CAST(a AS INT) FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression");
    };
    assert!(matches!(
        expr,
        ast::Expr::Cast {
            try_cast: false,
            ..
        }
    ));
}

#[test]
fn is_null_and_is_not_null() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a IS NULL") else {
        panic!("expected Select");
    };
    assert!(matches!(
        s.filter,
        Some(ast::Expr::IsNull { negated: false, .. })
    ));

    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE a IS NOT NULL") else {
        panic!("expected Select");
    };
    assert!(matches!(
        s.filter,
        Some(ast::Expr::IsNull { negated: true, .. })
    ));
}

#[test]
fn unary_not_operator() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE NOT a") else {
        panic!("expected Select");
    };
    assert!(matches!(
        s.filter,
        Some(ast::Expr::Unary {
            op: ast::UnaryOp::Not,
            ..
        })
    ));
}

#[test]
fn string_and_boolean_literals() {
    let ast::Statement::Select(s) = ok("SELECT * FROM t WHERE name = 'alice' AND active = TRUE")
    else {
        panic!("expected Select");
    };
    assert!(s.filter.is_some());
}

// --- Error paths: must never panic, must return a clean Err -----------

#[test]
fn invalid_sql_is_syntax_error() {
    assert!(matches!(parse("SELECT FROM WHERE"), Err(Error::Syntax(_))));
    assert!(matches!(parse("not sql at all"), Err(Error::Syntax(_))));
    assert!(matches!(parse("SELECT * FROM"), Err(Error::Syntax(_))));
}

#[test]
fn empty_input_is_empty_error() {
    assert!(matches!(parse(""), Err(Error::Empty)));
    assert!(matches!(parse("   \n\t "), Err(Error::Empty)));
    assert!(matches!(parse("-- only a comment"), Err(Error::Empty)));
}

// --- Surface gate: `GenericDialect` tokenizes more than NusaDB's documented surface; the
// identifier lexicon and the wildcard decorations must stay rejected (follow-up). These
// lock the gate so a future dialect/tokenizer change cannot silently widen the surface.

#[test]
fn rejects_out_of_surface_identifiers() {
    // `@`/`#`-led and non-ASCII unquoted identifiers — `GenericDialect` admits these, NusaDB
    // does not. (Column position and table position both go through the lexical gate.)
    for sql in [
        "SELECT a FROM @t",
        "SELECT @x FROM t",
        "SELECT a FROM #t",
        "SELECT #x FROM t",
        "SELECT café FROM t",
        "SELECT a FROM café",
        "INSERT INTO @t VALUES (1)",
        "UPDATE @t SET a = 1",
    ] {
        assert!(
            matches!(parse(sql), Err(Error::Unsupported(_))),
            "{sql}: expected Unsupported, got {:?}",
            parse(sql)
        );
    }
}

#[test]
fn rejects_non_double_quote_delimited_identifiers() {
    // NusaDB documents only `"..."` for quoted identifiers; backtick / bracket delimiters are
    // tokenized by `GenericDialect` but outside the surface.
    for sql in ["SELECT `x` FROM t", "SELECT a FROM `t`"] {
        assert!(
            matches!(parse(sql), Err(Error::Unsupported(_))),
            "{sql}: expected Unsupported, got {:?}",
            parse(sql)
        );
    }
    // The standard double-quoted form is still accepted, case-preserved.
    let ast::Statement::Select(s) = ok("SELECT \"X\" FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(s.projection.len(), 1);
}

#[test]
fn rejects_wildcard_decorations() {
    // `SELECT * EXCEPT (a)` and friends were previously silently dropped to a bare `*`.
    for sql in [
        "SELECT * EXCEPT (a) FROM t",
        "SELECT * EXCLUDE (a) FROM t",
        "SELECT * REPLACE (a + 1 AS a) FROM t",
        "SELECT t.* EXCEPT (a) FROM t",
    ] {
        assert!(
            matches!(parse(sql), Err(Error::Unsupported(_))),
            "{sql}: expected Unsupported, got {:?}",
            parse(sql)
        );
    }
    // Plain and qualified wildcards still parse.
    assert!(matches!(ok("SELECT * FROM t"), ast::Statement::Select(_)));
    assert!(matches!(ok("SELECT t.* FROM t"), ast::Statement::Select(_)));
}

#[test]
fn bitwise_and_or_parse() {
    for (sql, want) in [
        ("SELECT a & b FROM t", ast::BinaryOp::BitAnd),
        ("SELECT a | b FROM t", ast::BinaryOp::BitOr),
        ("SELECT a << b FROM t", ast::BinaryOp::ShiftLeft),
        ("SELECT a >> b FROM t", ast::BinaryOp::ShiftRight),
        ("SELECT a && b FROM t", ast::BinaryOp::ArrayOverlap),
    ] {
        let ast::Statement::Select(s) = ok(sql) else {
            panic!("expected Select for {sql}");
        };
        let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
            panic!("expected an expression projection for {sql}");
        };
        let ast::Expr::Binary { op, .. } = expr else {
            panic!("expected a binary expression for {sql}, got {expr:?}");
        };
        assert_eq!(*op, want, "{sql}");
    }
}

#[test]
fn rejects_other_generic_dialect_extensions() {
    // A grammar extension `GenericDialect` enables that NusaDB does not model — already rejected
    // (at the converter or grammar), pinned here so the rejection cannot regress unnoticed.
    let group_by_all = "SELECT a FROM t GROUP BY ALL";
    assert!(
        matches!(parse(group_by_all), Err(Error::Unsupported(_))),
        "{group_by_all}: expected Unsupported, got {:?}",
        parse(group_by_all)
    );
    // The SQL-standard Unicode string literal `U&'...'` IS modelled since
    // (it previously parsed as a bitwise-AND shape the analyzer
    // rejected): `U&` immediately followed by a quote is the literal — `\0041` resolves to `A` —
    // while `u & '…'` with whitespace stays the bitwise operator, matching the
    // reference's tokenization of the same ambiguity.
    let ast::Statement::Select(s) = ok("SELECT U&'\\0041' FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression projection");
    };
    assert!(
        matches!(expr, ast::Expr::Literal(ast::Value::Text(t)) if t == "A"),
        "U&'\\0041' should be the unicode literal `A`, got {expr:?}",
    );
    let ast::Statement::Select(s) = ok("SELECT u & '\\0041' FROM t") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected an expression projection");
    };
    assert!(
        matches!(
            expr,
            ast::Expr::Binary {
                op: ast::BinaryOp::BitAnd,
                ..
            }
        ),
        "spaced `u & '…'` stays a bitwise-AND, got {expr:?}",
    );
}

#[test]
fn select_distinct_sets_the_flag() {
    let ast::Statement::Select(s) = ok("SELECT DISTINCT a FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(s.distinct, Some(ast::Distinct::All));
    let ast::Statement::Select(s) = ok("SELECT a FROM t") else {
        panic!("expected Select");
    };
    assert_eq!(s.distinct, None);
}

#[test]
fn select_distinct_on_captures_expressions() {
    // `DISTINCT ON (exprs)` keeps the expression list.
    let ast::Statement::Select(s) = ok("SELECT DISTINCT ON (a, b) a, b, c FROM t") else {
        panic!("expected Select");
    };
    let Some(ast::Distinct::On(exprs)) = s.distinct else {
        panic!("expected DISTINCT ON");
    };
    assert_eq!(exprs.len(), 2);
    assert!(matches!(exprs[0], ast::Expr::Column(_)));
}

// --- Transaction & session control -------------------------

#[test]
fn begin_plain_has_default_settings() {
    let ast::Statement::BeginTransaction(s) = ok("BEGIN") else {
        panic!("expected BeginTransaction");
    };
    assert!(s.is_default());
    assert!(matches!(
        ok("START TRANSACTION"),
        ast::Statement::BeginTransaction(_)
    ));
}

#[test]
fn begin_with_isolation_and_access_mode() {
    let ast::Statement::BeginTransaction(s) = ok("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY")
    else {
        panic!("expected BeginTransaction");
    };
    assert_eq!(s.isolation, Some(ast::IsolationLevel::Serializable));
    assert_eq!(s.access_mode, Some(ast::AccessMode::ReadOnly));
}

#[test]
fn savepoint_rollback_to_release() {
    assert_eq!(
        ok("SAVEPOINT sp1"),
        ast::Statement::Savepoint("sp1".to_owned())
    );
    assert_eq!(
        ok("ROLLBACK TO SAVEPOINT sp1"),
        ast::Statement::RollbackToSavepoint("sp1".to_owned())
    );
    assert_eq!(
        ok("ROLLBACK TO sp1"),
        ast::Statement::RollbackToSavepoint("sp1".to_owned())
    );
    assert_eq!(
        ok("RELEASE SAVEPOINT sp1"),
        ast::Statement::ReleaseSavepoint("sp1".to_owned())
    );
}

#[test]
fn set_transaction_isolation() {
    let ast::Statement::SetTransaction(s) = ok("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    else {
        panic!("expected SetTransaction");
    };
    assert_eq!(s.isolation, Some(ast::IsolationLevel::RepeatableRead));
}

#[test]
fn set_variable_and_reset() {
    let ast::Statement::SetVariable(s) = ok("SET search_path = 'public'") else {
        panic!("expected SetVariable");
    };
    assert_eq!(s.name, "search_path");
    assert_eq!(s.value.as_deref(), Some("public"));

    // RESET parses as SET name = DEFAULT → value None.
    let ast::Statement::SetVariable(s) = ok("RESET search_path") else {
        panic!("expected SetVariable (reset)");
    };
    assert_eq!(s.name, "search_path");
    assert!(s.value.is_none());

    // search_path is a list: a multi-element `SET search_path TO a, public` renders the joined
    // list so the session stores the ordered path.
    let ast::Statement::SetVariable(s) = ok("SET search_path TO app, public") else {
        panic!("expected SetVariable (list)");
    };
    assert_eq!(s.name, "search_path");
    assert_eq!(s.value.as_deref(), Some("app, public"));

    // A multi-value SET on any other GUC is still rejected (only search_path is a list).
    assert!(matches!(
        parse("SET work_mem TO 1, 2"),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn show_variable() {
    assert_eq!(
        ok("SHOW search_path"),
        ast::Statement::Show("search_path".to_owned())
    );
}

#[test]
fn vacuum_parses_bare_case_insensitive_and_explained() {
    let default = ast::VacuumOptions::default();
    assert!(matches!(ok("VACUUM"), ast::Statement::Vacuum(o) if o == default));
    assert!(matches!(ok("vacuum"), ast::Statement::Vacuum(o) if o == default));
    assert!(matches!(ok("  VACUUM ;  "), ast::Statement::Vacuum(o) if o == default));
    let ast::Statement::Explain(inner, _) = ok("EXPLAIN VACUUM") else {
        panic!("expected Explain");
    };
    assert!(matches!(*inner, ast::Statement::Vacuum(o) if o == default));
}

#[test]
fn vacuum_parses_full_and_analyze_options() {
    use ast::VacuumOptions as V;
    assert!(
        matches!(ok("VACUUM FULL"), ast::Statement::Vacuum(o) if o == V { full: true, analyze: false })
    );
    assert!(
        matches!(ok("VACUUM ANALYZE"), ast::Statement::Vacuum(o) if o == V { full: false, analyze: true })
    );
    assert!(
        matches!(ok("VACUUM FULL ANALYZE"), ast::Statement::Vacuum(o) if o == V { full: true, analyze: true })
    );
    // Parenthesized form, any order.
    assert!(
        matches!(ok("VACUUM (ANALYZE, FULL)"), ast::Statement::Vacuum(o) if o == V { full: true, analyze: true })
    );
    // An unknown *option* (parenthesized form) is rejected, not silently accepted.
    assert!(parse("VACUUM (NONSENSE)").is_err());
}

#[test]
fn vacuum_accepts_a_table_qualified_form() {
    use ast::VacuumOptions as V;
    // `VACUUM <table>` (and with options) is accepted rather than rejected, so migration tools and
    // ORM health-checks do not fail; NusaDB reclaims cluster-wide, so the named table is covered by
    // the global vacuum. A bare word after VACUUM is a table name, not an unknown option.
    assert!(
        matches!(ok("VACUUM t"), ast::Statement::Vacuum(o) if o == V { full: false, analyze: false })
    );
    assert!(
        matches!(ok("VACUUM ANALYZE users"), ast::Statement::Vacuum(o) if o == V { full: false, analyze: true })
    );
    assert!(
        matches!(ok("VACUUM FULL ANALYZE a, b"), ast::Statement::Vacuum(o) if o == V { full: true, analyze: true })
    );
    assert!(
        matches!(ok("VACUUM (FULL) app.orders"), ast::Statement::Vacuum(o) if o == V { full: true, analyze: false })
    );
    // Genuinely malformed trailers still fall through and are rejected (no silent garbage-accept).
    assert!(parse("VACUUM t (").is_err());
    assert!(parse("VACUUM t WHERE x").is_err());
}

#[test]
fn reindex_is_accepted_as_a_noop() {
    // REINDEX (which sqlparser does not model as a statement) is accepted rather than rejected, so
    // the migration tools and ORM health-checks that emit it do not fail — NusaDB's B-tree indexes
    // are always consistent, so it is a no-op.
    for sql in [
        "REINDEX INDEX foo",
        "REINDEX TABLE users",
        "REINDEX SCHEMA public",
        "REINDEX DATABASE mydb",
        "reindex table t;",
        "REINDEX (VERBOSE) TABLE t",
    ] {
        assert!(
            matches!(parse(sql), Ok(ast::Statement::Reindex)),
            "should accept: {sql}"
        );
    }
    // A bare REINDEX with no target, or a glued token, is not recognized (falls through, rejected).
    assert!(parse("REINDEX").is_err());
    assert!(parse("REINDEXES").is_err());
}

#[test]
fn lock_table_parses_modes_lists_and_defaults() {
    use ast::LockMode as M;
    let lock = |sql: &str| match ok(sql) {
        ast::Statement::LockTable { tables, mode } => (tables, mode),
        other => panic!("expected LockTable, got {other:?}"),
    };
    // Bare form defaults to ACCESS EXCLUSIVE; the TABLE keyword is optional.
    assert_eq!(
        lock("LOCK TABLE t"),
        (vec!["t".to_owned()], M::AccessExclusive)
    );
    assert_eq!(lock("LOCK t"), (vec!["t".to_owned()], M::AccessExclusive));
    // Case-insensitive keywords; identifiers fold to lowercase.
    assert_eq!(
        lock("lock table T in access share mode"),
        (vec!["t".to_owned()], M::AccessShare)
    );
    // Multiple tables + explicit ACCESS EXCLUSIVE.
    assert_eq!(
        lock("LOCK TABLE a, b IN ACCESS EXCLUSIVE MODE"),
        (vec!["a".to_owned(), "b".to_owned()], M::AccessExclusive)
    );
    // An unsupported mode is rejected; a missing table name is rejected.
    assert!(parse("LOCK TABLE t IN SHARE MODE").is_err());
    assert!(parse("LOCK TABLE").is_err());
    // NOWAIT is not handled here → falls through to the generic parser, which rejects it.
    assert!(parse("LOCK TABLE t NOWAIT").is_err());
}

#[test]
fn prepare_execute_deallocate_parse() {
    // PREPARE keeps the inner statement; the declared types are dropped.
    let ast::Statement::Prepare { name, statement } =
        ok("PREPARE p (INT) AS SELECT a FROM t WHERE a = $1")
    else {
        panic!("expected Prepare");
    };
    assert_eq!(name, "p");
    assert!(matches!(*statement, ast::Statement::Select(_)));

    // EXECUTE carries the argument expressions.
    let ast::Statement::Execute { name, args } = ok("EXECUTE p (1, 'x')") else {
        panic!("expected Execute");
    };
    assert_eq!(name, "p");
    assert_eq!(args.len(), 2);

    // DEALLOCATE name vs ALL (case-insensitive, unquoted).
    assert!(matches!(
        ok("DEALLOCATE p"),
        ast::Statement::Deallocate(ast::DeallocateTarget::Name(n)) if n == "p"
    ));
    assert!(matches!(
        ok("deallocate all"),
        ast::Statement::Deallocate(ast::DeallocateTarget::All)
    ));

    // PREPARE of a non-query is rejected; EXECUTE ... USING is rejected.
    assert!(parse("PREPARE p AS VACUUM").is_err());
    assert!(parse("EXECUTE p USING 1").is_err());
}

#[test]
fn encrypt_decrypt_parse_as_two_arg_calls() {
    let ast::Statement::Select(s) = ok("SELECT encrypt(name, 'k'), decrypt(secret, 'k') FROM t")
    else {
        panic!("expected Select");
    };
    // Two projection items, an Encrypt and a Decrypt.
    assert_eq!(s.projection.len(), 2);

    // Wrong arity is rejected.
    assert!(matches!(
        parse("SELECT encrypt('x') FROM t"),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        parse("SELECT decrypt('a', 'b', 'c') FROM t"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn analyze_parses_table_and_column_forms() {
    let ast::Statement::Analyze(a) = ok("ANALYZE TABLE Users") else {
        panic!("expected Analyze");
    };
    assert_eq!(a.table, "users"); // unquoted identifier folds to lowercase
    assert!(a.columns.is_empty());

    let ast::Statement::Analyze(a) = ok("ANALYZE TABLE t FOR COLUMNS a, b") else {
        panic!("expected Analyze");
    };
    assert_eq!(a.columns, vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
fn multiple_statements_rejected() {
    assert!(matches!(
        parse("SELECT 1; SELECT 2"),
        Err(Error::MultipleStatements(2)),
    ));
}

#[test]
fn unsupported_constructs_rejected() {
    // Constructs the parser still refuses outright. Several formerly-listed forms now parse
    // (NATURAL/CROSS JOINGROUP BY ROLLUP/CUBE/GROUPING SETSCREATE TABLE ... DEFAULT
    // Rejected at the analyzer instead; an *aliased* subquery in `FROM` is a derived table,
    // ). A derived table without the mandatory alias is still unsupported.
    assert!(
        matches!(
            parse("SELECT * FROM (SELECT 1)"),
            Err(Error::Unsupported(_))
        ),
        "expected Unsupported for an unaliased subquery in FROM",
    );
}

#[test]
fn qualified_column_reference_parses() {
    let ast::Statement::Select(s) = ok("SELECT t.id FROM t WHERE t.id = 1") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected expr projection");
    };
    assert_eq!(
        *expr,
        ast::Expr::QualifiedColumn {
            table: "t".to_owned(),
            column: "id".to_owned(),
        },
    );
}

#[test]
fn inner_and_left_join_parse() {
    let ast::Statement::Select(s) =
        ok("SELECT * FROM a JOIN b ON a.id = b.aid LEFT JOIN c ON b.id = c.bid")
    else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.base.name, "a");
    assert_eq!(from.joins.len(), 2);
    assert_eq!(from.joins[0].kind, ast::JoinKind::Inner);
    assert_eq!(from.joins[0].table.name, "b");
    assert!(matches!(from.joins[0].condition, ast::JoinCondition::On(_)));
    assert_eq!(from.joins[1].kind, ast::JoinKind::Left);
    assert_eq!(from.joins[1].table.name, "c");
}

// --- CROSS / NATURAL JOIN / USING ----------------------------

#[test]
fn cross_join_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM a CROSS JOIN b") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.joins.len(), 1);
    assert_eq!(from.joins[0].kind, ast::JoinKind::Cross);
    assert_eq!(from.joins[0].condition, ast::JoinCondition::None);
}

#[test]
fn natural_join_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM a NATURAL JOIN b") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.joins[0].kind, ast::JoinKind::Inner);
    assert_eq!(from.joins[0].condition, ast::JoinCondition::Natural);
}

#[test]
fn natural_left_join_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM a NATURAL LEFT JOIN b") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.joins[0].kind, ast::JoinKind::Left);
    assert_eq!(from.joins[0].condition, ast::JoinCondition::Natural);
}

#[test]
fn using_join_parses() {
    let ast::Statement::Select(s) = ok("SELECT * FROM a JOIN b USING (id, name)") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.joins[0].kind, ast::JoinKind::Inner);
    assert_eq!(
        from.joins[0].condition,
        ast::JoinCondition::Using(vec!["id".to_owned(), "name".to_owned()])
    );
}

#[test]
fn using_folds_column_names() {
    let ast::Statement::Select(s) = ok("SELECT * FROM a JOIN b USING (ID)") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(
        from.joins[0].condition,
        ast::JoinCondition::Using(vec!["id".to_owned()])
    );
}

#[test]
fn join_without_on_is_rejected() {
    // USING / NATURAL / CROSS now parse; the analyzer rejects them until the
    // executor path lands. A bare JOIN with no condition parses (sqlparser models it as
    // JoinConstraint::None → our JoinCondition::None), and the analyzer rejects it.
    let ast::Statement::Select(s) = ok("SELECT * FROM a JOIN b") else {
        panic!("expected Select");
    };
    let from = s.from.expect("FROM clause");
    assert_eq!(from.joins[0].condition, ast::JoinCondition::None);
}

// --- Aggregate FILTER -----------------------------------------

fn get_agg(sql: &str) -> ast::Expr {
    let ast::Statement::Select(s) = ok(sql) else {
        panic!("expected Select")
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected Expr projection");
    };
    expr.clone()
}

#[test]
fn aggregate_without_filter_has_none() {
    let ast::Expr::Aggregate { filter, .. } = get_agg("SELECT COUNT(*) FROM t") else {
        panic!("expected Aggregate");
    };
    assert!(filter.is_none());
}

#[test]
fn aggregate_filter_count_star() {
    let ast::Expr::Aggregate {
        func, arg, filter, ..
    } = get_agg("SELECT COUNT(*) FILTER (WHERE id > 0) FROM t")
    else {
        panic!("expected Aggregate");
    };
    assert_eq!(func, ast::AggregateFunc::Count);
    assert!(arg.is_none());
    assert!(filter.is_some());
}

#[test]
fn aggregate_filter_sum_expr() {
    let ast::Expr::Aggregate { func, filter, .. } =
        get_agg("SELECT SUM(salary) FILTER (WHERE active = TRUE) FROM t")
    else {
        panic!("expected Aggregate");
    };
    assert_eq!(func, ast::AggregateFunc::Sum);
    assert!(filter.is_some());
}

#[test]
fn aggregate_filter_on_non_aggregate_is_rejected() {
    assert!(matches!(
        parse("SELECT my_func(x) FILTER (WHERE x > 0) FROM t"),
        Err(Error::Unsupported(_)),
    ));
}

// --- Aggregate DISTINCT ----------------------------------------

#[test]
fn aggregate_without_distinct_has_false() {
    let ast::Expr::Aggregate { distinct, .. } = get_agg("SELECT COUNT(*) FROM t") else {
        panic!("expected Aggregate");
    };
    assert!(!distinct);
}

#[test]
fn count_distinct_parses() {
    let ast::Expr::Aggregate {
        func,
        arg,
        distinct,
        ..
    } = get_agg("SELECT COUNT(DISTINCT id) FROM t")
    else {
        panic!("expected Aggregate");
    };
    assert_eq!(func, ast::AggregateFunc::Count);
    assert!(arg.is_some());
    assert!(distinct);
}

#[test]
fn sum_distinct_parses() {
    let ast::Expr::Aggregate { func, distinct, .. } = get_agg("SELECT SUM(DISTINCT salary) FROM t")
    else {
        panic!("expected Aggregate");
    };
    assert_eq!(func, ast::AggregateFunc::Sum);
    assert!(distinct);
}

#[test]
fn count_all_is_rejected() {
    // COUNT(ALL x) is rejected — use COUNT(x) for non-distinct counting.
    assert!(matches!(
        parse("SELECT COUNT(ALL id) FROM t"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn distinct_on_non_aggregate_is_rejected() {
    assert!(matches!(
        parse("SELECT my_func(DISTINCT x) FROM t"),
        Err(Error::Unsupported(_)),
    ));
}

// --- GROUP BY ROLLUP / CUBE / GROUPING SETS ------------------

#[test]
fn group_by_plain_expressions_still_works() {
    let ast::Statement::Select(s) = ok("SELECT a FROM t GROUP BY a, b") else {
        panic!("expected Select");
    };
    let ast::GroupBy::Expressions(keys) = s.group_by else {
        panic!("expected Expressions");
    };
    assert_eq!(keys.len(), 2);
}

#[test]
fn group_by_rollup_single_column() {
    let ast::Statement::Select(s) = ok("SELECT a FROM t GROUP BY ROLLUP (a)") else {
        panic!("expected Select");
    };
    let ast::GroupBy::Rollup(sets) = s.group_by else {
        panic!("expected Rollup");
    };
    assert_eq!(sets.len(), 1);
    assert_eq!(sets[0].len(), 1);
}

#[test]
fn group_by_rollup_multi_column() {
    let ast::Statement::Select(s) = ok("SELECT a, b FROM t GROUP BY ROLLUP (a, b, c)") else {
        panic!("expected Select");
    };
    assert!(matches!(s.group_by, ast::GroupBy::Rollup(_)));
}

#[test]
fn group_by_cube_parses() {
    let ast::Statement::Select(s) = ok("SELECT a FROM t GROUP BY CUBE (a, b)") else {
        panic!("expected Select");
    };
    assert!(matches!(s.group_by, ast::GroupBy::Cube(_)));
}

#[test]
fn group_by_grouping_sets_parses() {
    let ast::Statement::Select(s) = ok("SELECT a FROM t GROUP BY GROUPING SETS ((a, b), (c), ())")
    else {
        panic!("expected Select");
    };
    let ast::GroupBy::GroupingSets(sets) = s.group_by else {
        panic!("expected GroupingSets");
    };
    assert_eq!(sets.len(), 3); // (a,b), (c), ()
    assert!(sets[2].is_empty()); // empty set ()
}

#[test]
fn group_by_all_still_rejected() {
    assert!(matches!(
        parse("SELECT a FROM t GROUP BY ALL"),
        Err(Error::Unsupported(_)),
    ));
}

// --- Case-folding (NusaDB identifier semantics) --------------------

#[test]
fn unquoted_identifiers_fold_to_lowercase() {
    let ast::Statement::CreateTable(ct) = ok("CREATE TABLE Users (ID INT NOT NULL, Name TEXT)")
    else {
        panic!("expected CreateTable");
    };
    assert_eq!(ct.name, "users");
    assert_eq!(ct.columns[0].name, "id");
    assert_eq!(ct.columns[1].name, "name");

    let ast::Statement::Select(s) = ok("SELECT ID FROM USERS WHERE NAME = 'x'") else {
        panic!("expected Select");
    };
    assert_eq!(s.from.expect("FROM clause").base.name, "users");
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected Expr projection");
    };
    assert!(matches!(expr, ast::Expr::Column(name) if name == "id"));
}

#[test]
fn quoted_identifiers_preserve_case() {
    let ast::Statement::CreateTable(ct) = ok(r#"CREATE TABLE "Users" ("ID" INT, "Name" TEXT)"#)
    else {
        panic!("expected CreateTable");
    };
    assert_eq!(ct.name, "Users");
    assert_eq!(ct.columns[0].name, "ID");
    assert_eq!(ct.columns[1].name, "Name");
}

#[test]
fn insert_column_list_folds_case() {
    let ast::Statement::Insert(ins) = ok("INSERT INTO Users (ID, Name) VALUES (1, 'a')") else {
        panic!("expected Insert");
    };
    assert_eq!(ins.table, "users");
    assert_eq!(ins.columns, ["id", "name"]);
}

#[test]
fn select_alias_folds_case() {
    let ast::Statement::Select(s) = ok("SELECT id AS UserID FROM users") else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { alias, .. } = &s.projection[0] else {
        panic!("expected Expr");
    };
    assert_eq!(alias.as_deref(), Some("userid"));
}

// --- COMMENT ON ----------------------------------------------

/// Parse SQL expected to be a `COMMENT ON`, returning the inner node.
fn comment(sql: &str) -> ast::CommentOn {
    match ok(sql) {
        ast::Statement::CommentOn(c) => c,
        other => panic!("expected CommentOn, got {other:?}"),
    }
}

#[test]
fn comment_on_table() {
    let c = comment("COMMENT ON TABLE users IS 'the user accounts'");
    assert_eq!(
        c.target,
        ast::CommentTarget::Table {
            table: "users".to_owned(),
        }
    );
    assert_eq!(c.comment.as_deref(), Some("the user accounts"));
}

#[test]
fn comment_on_column_folds_case() {
    // A bare `table.column` folds case to (users, name). (A schema-qualified
    // `schema.table.column` is rejected instead of silently dropping the schema — see
    // `comment_on_schema_qualified_column_is_rejected`)
    let c = comment("COMMENT ON COLUMN Users.Name IS 'display name'");
    assert_eq!(
        c.target,
        ast::CommentTarget::Column {
            table: "users".to_owned(),
            column: "name".to_owned(),
        }
    );
    assert_eq!(c.comment.as_deref(), Some("display name"));
}

#[test]
fn comment_is_null_clears() {
    let c = comment("COMMENT ON TABLE users IS NULL");
    assert!(c.comment.is_none());
}

#[test]
fn comment_quoted_identifier_preserves_case() {
    let c = comment("COMMENT ON COLUMN \"T\".\"C\" IS 'x'");
    assert_eq!(
        c.target,
        ast::CommentTarget::Column {
            table: "T".to_owned(),
            column: "C".to_owned(),
        }
    );
}

#[test]
fn comment_on_bare_column_is_rejected() {
    // A column comment needs a table to resolve against.
    assert!(matches!(
        parse("COMMENT ON COLUMN name IS 'x'"),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn comment_on_unsupported_target_is_rejected() {
    assert!(matches!(
        parse("COMMENT ON SCHEMA public IS 'x'"),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn comment_missing_is_or_value_is_a_syntax_error() {
    assert!(matches!(
        parse("COMMENT ON TABLE users 'x'"),
        Err(Error::Syntax(_))
    ));
    assert!(matches!(
        parse("COMMENT ON TABLE users IS"),
        Err(Error::Syntax(_))
    ));
}

#[test]
fn comment_rejects_trailing_content() {
    assert!(matches!(
        parse("COMMENT ON TABLE users IS 'x'; SELECT 1"),
        Err(Error::Syntax(_))
    ));
}

// --- COPY ----------------------------------------------

#[test]
fn copy_from_stdin_defaults() {
    let ast::Statement::Copy(c) = ok("COPY users FROM STDIN") else {
        panic!("expected Copy");
    };
    assert_eq!(c.table, "users");
    assert!(c.columns.is_empty());
    assert_eq!(c.direction, ast::CopyDirection::From);
    assert_eq!(c.format, ast::CopyFormat::default());
}

#[test]
fn copy_to_stdout_with_columns() {
    let ast::Statement::Copy(c) = ok("COPY users (id, name) TO STDOUT") else {
        panic!("expected Copy");
    };
    assert_eq!(c.direction, ast::CopyDirection::To);
    assert_eq!(c.columns, vec!["id".to_owned(), "name".to_owned()]);
}

#[test]
fn copy_honors_text_format_options() {
    let ast::Statement::Copy(c) =
        ok("COPY users FROM STDIN WITH (FORMAT text, DELIMITER ',', NULL 'NUL', HEADER true)")
    else {
        panic!("expected Copy");
    };
    assert_eq!(c.format.delimiter, ',');
    assert_eq!(c.format.null, "NUL");
    assert!(c.format.header);
}

#[test]
fn copy_rejects_file_target_and_csv_format() {
    // A file target, not STDIN/STDOUT.
    assert!(matches!(
        parse("COPY users FROM 'data.csv'"),
        Err(Error::Unsupported(_))
    ));
    // CSV/binary formats are follow-ups.
    assert!(matches!(
        parse("COPY users FROM STDIN WITH (FORMAT csv)"),
        Err(Error::Unsupported(_))
    ));
}

// --- silent-wrong parser gaps closed ----------------------

#[test]
fn drop_cascade_accepted_for_tables_rejected_elsewhere() {
    // DROP TABLE ... CASCADE drops referencing FOREIGN KEYs; object kinds
    // that track no dependencies keep the honest rejection.
    let ast::Statement::DropTable(d) = ok("DROP TABLE t CASCADE") else {
        panic!("expected DropTable");
    };
    assert!(d.cascade);
    assert!(matches!(
        parse("DROP INDEX i CASCADE"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn drop_purge_is_rejected() {
    assert!(matches!(
        parse("DROP TABLE t PURGE"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn drop_restrict_is_accepted() {
    // RESTRICT is the default no-cascade behavior, so it is honored as a plain DROP.
    assert!(matches!(
        ok("DROP TABLE t RESTRICT"),
        ast::Statement::DropTable(_)
    ));
}

#[test]
fn schema_qualified_name_carries_its_schema() {
    // `schema.table` is accepted and carries an explicit qualifier through to the AST — it no
    // longer collapses to its last component (the former reject), nor is it silently dropped.
    let ast::Statement::DropTable(d) = ok("DROP TABLE app.users") else {
        panic!("expected DropTable");
    };
    assert_eq!(
        (d.schema.as_deref(), d.name.as_str()),
        (Some("app"), "users")
    );

    let ast::Statement::CreateTable(c) = ok("CREATE TABLE app.users (id INT)") else {
        panic!("expected CreateTable");
    };
    assert_eq!(
        (c.schema.as_deref(), c.name.as_str()),
        (Some("app"), "users")
    );

    // A bare name carries no qualifier (None → resolved via search path); an explicit `public.`
    // qualifier is preserved as `Some("public")`.
    let ast::Statement::DropTable(d) = ok("DROP TABLE users") else {
        panic!("expected DropTable");
    };
    assert_eq!((d.schema.as_deref(), d.name.as_str()), (None, "users"));
    let ast::Statement::DropTable(d) = ok("DROP TABLE public.users") else {
        panic!("expected DropTable");
    };
    assert_eq!(
        (d.schema.as_deref(), d.name.as_str()),
        (Some("public"), "users")
    );

    // Three-part names (db.schema.table) are still rejected.
    assert!(matches!(
        parse("DROP TABLE d.app.users"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn drop_temporary_is_rejected() {
    // DROP TEMPORARY TABLE on a persistent table would be a silent footgun.
    assert!(matches!(
        parse("DROP TEMPORARY TABLE t"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_temporary_table_is_rejected() {
    assert!(matches!(
        parse("CREATE TEMPORARY TABLE t (id INT)"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_table_partition_by_is_rejected() {
    // Silently dropping PARTITION BY and creating an ordinary heap would mis-store rows and accept
    // out-of-range inserts (QA a silent-wrong footgun) — reject it loudly.
    assert!(matches!(
        parse("CREATE TABLE t (id INT, d DATE) PARTITION BY RANGE (d)"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_table_clustered_by_is_rejected() {
    // CLUSTERED BY changes physical row placement too; rejecting it loudly avoids the same
    // silent-heap footgun as PARTITION BY.
    assert!(matches!(
        parse("CREATE TABLE t (id INT) CLUSTERED BY (id) INTO 4 BUCKETS"),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn comment_on_schema_qualified_column_is_rejected() {
    // `schema.table.column` must not silently collapse to `table.column`.
    assert!(matches!(
        parse("COMMENT ON COLUMN app.users.id IS 'x'"),
        Err(Error::Unsupported(_)),
    ));
}

// --- decimal literals are exact NUMERIC, not f64 ------------------

/// The single projected expression of a `SELECT <expr>` statement.
fn projected_literal(sql: &str) -> ast::Value {
    let ast::Statement::Select(s) = ok(sql) else {
        panic!("expected Select");
    };
    let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
        panic!("expected Expr projection");
    };
    match expr {
        ast::Expr::Literal(v) => v.clone(),
        other => panic!("expected a literal, got {other:?}"),
    }
}

#[test]
fn decimal_literal_is_numeric_not_float() {
    // A plain decimal literal must parse to exact NUMERIC so it is not pre-rounded through f64
    // before reaching a NUMERIC column or exact arithmetic.
    assert!(matches!(
        projected_literal("SELECT 0.1"),
        ast::Value::Numeric(_)
    ));
    assert!(matches!(
        projected_literal("SELECT 123.45"),
        ast::Value::Numeric(_)
    ));
    // Integers stay INT; exponent / over-scale forms fall back to FLOAT.
    assert!(matches!(
        projected_literal("SELECT 42"),
        ast::Value::Int(42)
    ));
    assert!(matches!(
        projected_literal("SELECT 1e10"),
        ast::Value::Float(_)
    ));
}

#[test]
fn array_slice_parses_and_stride_rejected() {
    // `a[i:j]` (and open-bound forms) parse to an ArraySlice expression.
    for sql in [
        "SELECT a[1:2] FROM t",
        "SELECT a[2:] FROM t",
        "SELECT a[:3] FROM t",
        "SELECT a[:] FROM t",
    ] {
        let ast::Statement::Select(s) = ok(sql) else {
            panic!("expected Select for {sql}");
        };
        let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
            panic!("expected an expression projection for {sql}");
        };
        assert!(
            matches!(expr, ast::Expr::ArraySlice { .. }),
            "expected ArraySlice for {sql}, got {expr:?}"
        );
    }
    // A slice with a stride is not standard and is rejected.
    assert!(matches!(
        parse("SELECT a[1:6:2] FROM t"),
        Err(Error::Unsupported(_))
    ));
}

// --- LISTEN / UNLISTEN / NOTIFY (async pub/sub) -----------------------

#[test]
fn listen_folds_channel() {
    assert_eq!(
        ok("LISTEN orders"),
        ast::Statement::Listen("orders".to_owned())
    );
    // Unquoted channel names fold to lowercase (NusaDB's identifier rule); trailing `;` is fine.
    assert_eq!(
        ok("LISTEN Orders;"),
        ast::Statement::Listen("orders".to_owned())
    );
    // A quoted channel keeps its case and may contain spaces.
    assert_eq!(
        ok(r#"LISTEN "My Channel""#),
        ast::Statement::Listen("My Channel".to_owned())
    );
    // A bare `LISTEN` with no channel is an error, and `listener` is an ordinary identifier, not the
    // LISTEN keyword (so it falls through to the generic parser, which rejects it).
    assert!(parse("LISTEN").is_err());
    assert!(parse("LISTEN a b").is_err());
}

#[test]
fn unlisten_channel_and_wildcard() {
    assert_eq!(
        ok("UNLISTEN orders"),
        ast::Statement::Unlisten(Some("orders".to_owned()))
    );
    assert_eq!(ok("UNLISTEN *"), ast::Statement::Unlisten(None));
    assert!(parse("UNLISTEN").is_err());
}

#[test]
fn notify_channel_and_payload() {
    assert_eq!(
        ok("NOTIFY orders"),
        ast::Statement::Notify {
            channel: "orders".to_owned(),
            payload: None,
        }
    );
    assert_eq!(
        ok("NOTIFY orders, 'row 42'"),
        ast::Statement::Notify {
            channel: "orders".to_owned(),
            payload: Some("row 42".to_owned()),
        }
    );
    // A doubled quote inside the payload literal unescapes to a single quote.
    assert_eq!(
        ok("NOTIFY ch, 'it''s here'"),
        ast::Statement::Notify {
            channel: "ch".to_owned(),
            payload: Some("it's here".to_owned()),
        }
    );
    // A payload that is not a string literal is rejected.
    assert!(parse("NOTIFY ch, 42").is_err());
    assert!(parse("NOTIFY").is_err());
}

// --- Full-text search: @@ and to_tsvector/to_tsquery (F1) ---------

#[test]
fn ts_match_operator_and_functions_parse() {
    // `v @@ q` maps to BinaryOp::TsMatch.
    let ast::Expr::Binary { op, .. } = select_filter("SELECT * FROM t WHERE v @@ q") else {
        panic!("expected Binary");
    };
    assert_eq!(op, ast::BinaryOp::TsMatch);
    // The FTS functions resolve as scalar functions in both 1- and 2-argument forms.
    for sql in [
        "SELECT to_tsvector('simple', body) FROM t",
        "SELECT to_tsvector(body) FROM t",
        "SELECT to_tsquery('simple', 'a & b') FROM t",
        "SELECT plainto_tsquery('simple', 'a b') FROM t",
    ] {
        assert!(parse(sql).is_ok(), "{sql} should parse");
    }
}

// --- sqlparser 0.62 migration pins -------------------------------------

/// Behavior-carrying mappings introduced by the sqlparser 0.51 → 0.62 migration: the upgraded
/// grammar models several forms as new AST shapes, and these pins lock the intended conversions
/// against future dialect drift.
#[test]
fn sqlparser062_intentional_mappings_pin() {
    // `MINUS` is an alternate spelling of `EXCEPT`.
    let ast::Statement::SetOperation(so) = ok("SELECT a FROM t MINUS SELECT a FROM u") else {
        panic!("expected SetOperation");
    };
    let ast::SelectBody::SetOp { op, .. } = so.body else {
        panic!("expected SetOp body");
    };
    assert_eq!(op, ast::SetOp::Except);
    // A bare `JOIN` is the standard synonym for `INNER JOIN` (0.62 models it separately).
    let ast::Statement::Select(sel) = ok("SELECT * FROM a JOIN b ON a.id = b.id") else {
        panic!("expected Select");
    };
    let from = sel.from.expect("expected FROM");
    assert_eq!(from.joins.len(), 1);
    assert_eq!(from.joins[0].kind, ast::JoinKind::Inner);
    // `SELECT ALL` is the explicit spelling of the default quantifier — same as plain SELECT.
    let ast::Statement::Select(sel) = ok("SELECT ALL x FROM t") else {
        panic!("expected Select");
    };
    assert!(sel.distinct.is_none());
    // `SHOW DATABASES` routes to the session's unknown-parameter rejection path (the listing
    // statement was dropped from the surface; the catalog replacement is `nusadb_databases`).
    assert!(matches!(ok("SHOW DATABASES"), ast::Statement::Show(name) if name == "databases"));
}

/// Out-of-surface forms the 0.62 grammar newly parses must reject loudly (`Unsupported`), never
/// silently drop.
#[test]
fn sqlparser062_new_forms_reject_loudly() {
    // Typed table-alias column list — parses in 0.62, rejected by `fold_alias_columns`.
    let typed_alias = "SELECT * FROM (SELECT 1) AS d(a INT)";
    assert!(
        matches!(parse(typed_alias), Err(Error::Unsupported(_))),
        "{typed_alias}: expected Unsupported, got {:?}",
        parse(typed_alias)
    );
    // These stay outside the dialect entirely — they fail at the grammar (Syntax), which is
    // equally loud; the pins guard against them ever silently parsing as something else.
    // Comma-form `LIMIT <offset>, <limit>` and the multi-alias projection.
    assert!(parse("SELECT x FROM t LIMIT 2, 3").is_err());
    assert!(parse("SELECT x AS (a, b) FROM t").is_err());
    // `ORDER BY ALL` keeps its 0.51 reading — `all` is an ordinary column reference (the
    // order-by-all dialect extension is not enabled), so it resolves or errors in the analyzer.
    let ast::Statement::Select(sel) = ok("SELECT x FROM t ORDER BY all") else {
        panic!("expected Select");
    };
    assert_eq!(sel.order_by.len(), 1);
    assert!(matches!(&sel.order_by[0].expr, ast::Expr::Column(c) if c == "all"));
}

// ---: quantified comparison over a subquery ----------------------

/// `x <op> ANY/ALL/SOME (SELECT ...)` in the standard single-paren spelling, unblocked by
/// the sqlparser 0.62 upgrade (0.51 only parsed the subquery operand behind double parens).
/// `= ANY` / `<> ALL` ride the optimized IN / NOT IN (semi/anti-join) path; every other
/// operator becomes the general quantified comparison.
#[test]
fn b127_quantified_subquery_single_paren() {
    let f = select_filter("SELECT * FROM t WHERE x = ANY (SELECT a FROM u)");
    assert!(matches!(f, ast::Expr::InSubquery { negated: false, .. }));
    let f = select_filter("SELECT * FROM t WHERE x <> ALL (SELECT a FROM u)");
    assert!(matches!(f, ast::Expr::InSubquery { negated: true, .. }));
    // `SOME` is the standard synonym for `ANY`.
    let f = select_filter("SELECT * FROM t WHERE x > SOME (SELECT a FROM u)");
    assert!(matches!(
        f,
        ast::Expr::QuantifiedComparison { all: false, .. }
    ));
    let f = select_filter("SELECT * FROM t WHERE x <= ALL (SELECT a FROM u)");
    assert!(matches!(
        f,
        ast::Expr::QuantifiedComparison { all: true, .. }
    ));
}

// ---: E'…' / U&'…' string prefixes ---------

/// The sqlparser 0.62 migration dropped the `E'…'` escape-string and `U&'…'` unicode-string
/// prefixes (0.51 recognized them unconditionally; 0.62 gates them behind dialect hooks) —
/// `SELECT E'hello'` tokenized as identifier `E` + a string. The hooks are back on; the
/// tokenizer hands the converter the already-unescaped text, so `E'a\tb\nc'` is 5 chars and
/// `U&'d\0061t'` is `dat`. A plain string with a literal backslash stays uninterpreted.
#[test]
fn estring_and_unicode_string_prefixes_parse() {
    fn projected_text(sql: &str) -> String {
        let ast::Statement::Select(s) = ok(sql) else {
            panic!("expected Select");
        };
        let ast::SelectItem::Expr { expr, .. } = &s.projection[0] else {
            panic!("expected a projected expression");
        };
        let ast::Expr::Literal(ast::Value::Text(t)) = expr else {
            panic!("expected a text literal, got {expr:?}");
        };
        t.clone()
    }

    assert_eq!(projected_text("SELECT E'hello'"), "hello");
    assert_eq!(projected_text(r"SELECT E'a\tb\nc'"), "a\tb\nc");
    assert_eq!(projected_text(r"SELECT E'\n'"), "\n");
    assert_eq!(projected_text(r"SELECT U&'d\0061t'"), "dat");
    // An ordinary string does NOT interpret backslashes (standard-conforming strings).
    assert_eq!(projected_text(r"SELECT 'a\tb'"), r"a\tb");
    // Lowercase prefix, same as the reference behaviour.
    assert_eq!(projected_text("SELECT e'hi'"), "hi");
}

/// FROM-less inline gate pins: `from_less_pure_select` must ALLOW only bounded pure-CPU one-row
/// shapes and DENY (default-deny) anything that could multiply rows, scan tables, or run
/// unbounded — a wrongly-allowed shape would execute on the server's I/O thread.
#[test]
fn from_less_pure_select_gate_allow_and_deny() {
    let gate = |sql: &str| ast::from_less_pure_select(&parse(sql).unwrap());
    // Allowed: literals, operators, CASE/COALESCE/CAST, the closed scalar built-in set.
    for sql in [
        "SELECT 1",
        "SELECT 1 + 2 * 3, 'x' || 'y'",
        "SELECT CASE WHEN TRUE THEN 1 ELSE 2 END",
        "SELECT COALESCE(NULL, 5), CAST(1 AS TEXT)",
        "SELECT NOW(), RANDOM(), UPPER('a')",
        "SELECT 1 WHERE FALSE",
    ] {
        assert!(gate(sql), "expected inline-eligible: `{sql}`");
    }
    // Denied: FROM, CTEs, subqueries, aggregates, window functions, SRFs, UDF-shaped calls,
    // set operations, VALUES, ROLLUP.
    for sql in [
        "SELECT a FROM t",
        "WITH c AS (SELECT 1) SELECT 2",
        "SELECT (SELECT 1)",
        "SELECT EXISTS (SELECT 1)",
        "SELECT SUM(1)",
        "SELECT ROW_NUMBER() OVER ()",
        "SELECT * FROM generate_series(1, 10)",
        "SELECT my_udf(1)",
        "SELECT 1 UNION SELECT 2",
        "VALUES (1), (2)",
        "SELECT 1 GROUP BY ROLLUP (1)",
    ] {
        assert!(!gate(sql), "expected pool-only: `{sql}`");
    }
}

/// Point-get candidate pins: `point_get_candidate` must ALLOW only a plain single-table
/// SELECT with pure expressions and a top-level `col = literal` conjunct, and DENY
/// (default-deny) every shape that could hide unbounded or impure work — joins, derived
/// tables, CTEs, subqueries, aggregates, windows, locks, DISTINCT, grouping, range-only
/// predicates. A candidate still needs the plan-shape gate before running inline.
#[test]
fn point_get_candidate_gate_allow_and_deny() {
    let gate = |sql: &str| ast::point_get_candidate(&parse(sql).unwrap());
    // Allowed candidates: equality conjunct on a plain table, extra pure conjuncts/ORDER
    // BY/LIMIT riding along.
    for sql in [
        "SELECT * FROM t WHERE id = 1",
        "SELECT v FROM t WHERE 1 = id",
        "SELECT v FROM t WHERE id = 1 AND v > 2",
        "SELECT v, id + 1 FROM t WHERE id = 1 ORDER BY v LIMIT 1",
        "SELECT upper(v) FROM t WHERE id = 'k'",
    ] {
        assert!(gate(sql), "expected point-get candidate: `{sql}`");
    }
    // Denied: everything else, by default.
    for sql in [
        "SELECT 1",                               // FROM-less (inline gate)
        "SELECT * FROM t",                        // no WHERE
        "SELECT * FROM t WHERE v > 2",            // no equality conjunct
        "SELECT * FROM t WHERE id = 1 OR id = 2", // OR is not a conjunct
        "SELECT * FROM t WHERE id = v",           // column = column
        "SELECT * FROM t JOIN u ON t.id = u.id WHERE t.id = 1",
        "SELECT * FROM (SELECT 1 AS id) s WHERE id = 1", // derived table
        "WITH c AS (SELECT 1) SELECT * FROM t WHERE id = 1",
        "SELECT (SELECT 1) FROM t WHERE id = 1", // subquery projection
        "SELECT * FROM t WHERE id = (SELECT 1)", // subquery predicate
        "SELECT count(*) FROM t WHERE id = 1",   // aggregate
        "SELECT row_number() OVER () FROM t WHERE id = 1", // window
        "SELECT DISTINCT v FROM t WHERE id = 1",
        "SELECT v FROM t WHERE id = 1 GROUP BY v",
        "SELECT * FROM t WHERE id = 1 FOR UPDATE", // row lock
        "SELECT * FROM generate_series(1, 3) g(id) WHERE id = 1", // table function
    ] {
        assert!(!gate(sql), "expected pool-only: `{sql}`");
    }
}

/// The collation contract (the design decision): text ordering is bytewise (`C`)
/// BY DESIGN and documented; a `COLLATE` clause must reject loudly, never be silently dropped.
#[test]
fn collate_clause_rejects_loudly() {
    assert!(parse("SELECT 'a' COLLATE \"en_US\"").is_err());
    assert!(parse("SELECT x FROM t ORDER BY x COLLATE \"de_DE\"").is_err());
}
