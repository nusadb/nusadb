use super::{Catalog, analyze};
use crate::error::Error;
use crate::planner::{InsertSource, LogicalPlan, TypedExprKind};
use nusadb_core::{ColumnDef, ColumnType, TableId, TableSchema};

/// An in-memory [`Catalog`] for analyzer tests — no storage engine.
struct MockCatalog {
    tables: Vec<TableSchema>,
    superuser: bool,
}

impl MockCatalog {
    const fn new() -> Self {
        Self {
            tables: Vec::new(),
            superuser: true,
        }
    }

    fn with(mut self, schema: TableSchema) -> Self {
        self.tables.push(schema);
        self
    }

    /// Demote the session to a regular (non-superuser) user, like an authenticated wire session.
    const fn non_superuser(mut self) -> Self {
        self.superuser = false;
        self
    }
}

impl Catalog for MockCatalog {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        Ok(self.tables.iter().find(|t| t.name == name).cloned())
    }

    fn is_superuser(&self) -> bool {
        self.superuser
    }
}

fn col(name: &str, ty: ColumnType, nullable: bool) -> ColumnDef {
    ColumnDef {
        name: name.to_owned(),
        ty,
        nullable,
    }
}

/// `users(id INT NOT NULL, name TEXT, age INT, score FLOAT, active BOOL)`.
fn users() -> TableSchema {
    TableSchema {
        schema: "public".to_owned(),
        id: TableId(1),
        name: "users".to_owned(),
        columns: vec![
            col("id", ColumnType::Int, false),
            col("name", ColumnType::Text, true),
            col("age", ColumnType::Int, true),
            col("score", ColumnType::Float, true),
            col("active", ColumnType::Bool, true),
        ],
    }
}

fn catalog() -> MockCatalog {
    MockCatalog::new().with(users())
}

/// Parse then analyze, against the given catalog.
fn plan(sql: &str, catalog: &dyn Catalog) -> Result<LogicalPlan, Error> {
    analyze(crate::parser::parse(sql)?, catalog)
}

// --- CREATE TABLE -----------------------------------------------------

#[test]
fn create_table_ok() {
    let LogicalPlan::CreateTable(p) =
        plan("CREATE TABLE t (a INT, b TEXT)", &MockCatalog::new()).unwrap()
    else {
        panic!("expected CreateTable plan");
    };
    assert_eq!(p.table, "t");
    assert_eq!(p.columns.len(), 2);
}

#[test]
fn create_table_duplicate_column() {
    let result = plan("CREATE TABLE t (a INT, a TEXT)", &MockCatalog::new());
    assert!(matches!(result, Err(Error::DuplicateColumn { .. })));
}

#[test]
fn create_table_existing_is_rejected() {
    let result = plan("CREATE TABLE users (a INT)", &catalog());
    assert!(matches!(result, Err(Error::TableExists { .. })));
}

#[test]
fn create_table_if_not_exists_on_existing_is_ok() {
    let result = plan("CREATE TABLE IF NOT EXISTS users (a INT)", &catalog());
    assert!(matches!(result, Ok(LogicalPlan::CreateTable(_))));
}

// --- Column constraints in CREATE TABLE ------

#[test]
fn create_table_default_resolves_into_plan() {
    // DEFAULT resolves into the plan as `(column, SQL text)` for the executor to
    // persist; an expression default is type-checked and carried verbatim.
    let LogicalPlan::CreateTable(p) = plan(
        "CREATE TABLE t (a INT DEFAULT 5, b TEXT DEFAULT 'x', c INT)",
        &MockCatalog::new(),
    )
    .unwrap() else {
        panic!("expected CreateTable plan");
    };
    assert_eq!(
        p.defaults,
        vec![
            ("a".to_owned(), "5".to_owned()),
            ("b".to_owned(), "'x'".to_owned()),
        ],
    );

    // A type-incompatible default is rejected at analysis time.
    assert!(matches!(
        plan(
            "CREATE TABLE t (a INT DEFAULT 'notanint')",
            &MockCatalog::new()
        ),
        Err(Error::TypeMismatch { .. }),
    ));
    // A subquery default is rejected.
    assert!(matches!(
        plan(
            "CREATE TABLE t (a INT DEFAULT (SELECT 1))",
            &MockCatalog::new()
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_table_generated_stored_is_accepted() {
    // GENERATED ALWAYS AS (<expr>) STORED is now resolved + persisted; the executor computes
    // it per row. The analyzer accepts a valid STORED generated column.
    assert!(
        plan(
            "CREATE TABLE t (a INT, b INT GENERATED ALWAYS AS (a + 1) STORED)",
            &MockCatalog::new(),
        )
        .is_ok()
    );
}

#[test]
fn create_table_generated_virtual_is_rejected() {
    // VIRTUAL (the default when STORED is omitted) needs read-time evaluation — rejected honestly.
    assert!(matches!(
        plan(
            "CREATE TABLE t (a INT, b INT GENERATED ALWAYS AS (a + 1))",
            &MockCatalog::new(),
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_table_column_check_resolves_into_plan() {
    // Column-level CHECK lifts to a table CHECK, resolved into the plan for the executor to
    // persist and enforce per row.
    let LogicalPlan::CreateTable(p) =
        plan("CREATE TABLE t (a INT CHECK (a > 0))", &MockCatalog::new()).unwrap()
    else {
        panic!("expected CreateTable plan");
    };
    // The synthetic INT range check on column `a` is also present; assert on the user's CHECK only.
    let user: Vec<_> = p
        .check_constraints
        .iter()
        .filter(|c| !c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX))
        .collect();
    assert_eq!(user.len(), 1);
    assert_eq!(user[0].predicate_sql, "a > 0");

    // A non-boolean CHECK predicate is rejected at analysis time.
    assert!(matches!(
        plan("CREATE TABLE t (a INT CHECK (a + 1))", &MockCatalog::new()),
        Err(Error::TypeMismatch { .. }),
    ));
    // A subquery in a CHECK predicate is rejected.
    assert!(matches!(
        plan(
            "CREATE TABLE t (a INT CHECK (a > (SELECT 1)))",
            &MockCatalog::new()
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn create_table_column_references_resolves_as_fk() {
    // Column-level REFERENCES lifts to a FK and is resolved into the plan.
    let LogicalPlan::CreateTable(p) = plan(
        "CREATE TABLE orders (uid INT REFERENCES users (id))",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected CreateTable plan");
    };
    assert_eq!(p.foreign_keys.len(), 1);
    assert_eq!(p.foreign_keys[0].columns, vec!["uid".to_owned()]);
    assert_eq!(p.foreign_keys[0].parent_table, "users");
}

// --- DROP TABLE -------------------------------------------------------

#[test]
fn drop_table_ok() {
    assert!(matches!(
        plan("DROP TABLE users", &catalog()),
        Ok(LogicalPlan::DropTable(_)),
    ));
}

#[test]
fn drop_missing_table_is_rejected() {
    assert!(matches!(
        plan("DROP TABLE ghost", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn drop_missing_table_if_exists_is_ok() {
    assert!(matches!(
        plan("DROP TABLE IF EXISTS ghost", &catalog()),
        Ok(LogicalPlan::DropTable(_)),
    ));
}

// --- ALTER TABLE ------------------------------------------------------

use crate::planner::{AlterColumnOp, AlterTablePlan};

/// Resolve an `ALTER TABLE` against the `users` catalog, asserting success.
fn alter(sql: &str) -> AlterTablePlan {
    match plan(sql, &catalog()) {
        Ok(LogicalPlan::AlterTable(p)) => p,
        other => panic!("expected AlterTable plan, got {other:?}"),
    }
}

#[test]
fn alter_add_column_ok() {
    let AlterTablePlan::Apply { op, .. } = alter("ALTER TABLE users ADD COLUMN tag TEXT") else {
        panic!("expected Apply");
    };
    let AlterColumnOp::AddColumn(col) = op else {
        panic!("expected AddColumn");
    };
    assert_eq!(col.name, "tag");
    assert_eq!(col.ty, ColumnType::Text);
}

#[test]
fn alter_add_existing_column_is_rejected() {
    assert!(matches!(
        plan("ALTER TABLE users ADD COLUMN name TEXT", &catalog()),
        Err(Error::DuplicateColumn { .. }),
    ));
}

#[test]
fn alter_add_primary_key_column_is_unsupported() {
    assert!(matches!(
        plan("ALTER TABLE users ADD COLUMN k INT PRIMARY KEY", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn alter_drop_column_resolves_ordinal() {
    let AlterTablePlan::Apply { op, .. } = alter("ALTER TABLE users DROP COLUMN age") else {
        panic!("expected Apply");
    };
    assert_eq!(op, AlterColumnOp::DropColumn { index: 2 });
}

#[test]
fn alter_drop_missing_column_is_rejected() {
    assert!(matches!(
        plan("ALTER TABLE users DROP COLUMN ghost", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn alter_drop_missing_column_if_exists_is_noop() {
    assert_eq!(
        alter("ALTER TABLE users DROP COLUMN IF EXISTS ghost"),
        AlterTablePlan::Noop,
    );
}

#[test]
fn alter_rename_column_resolves_ordinal() {
    let AlterTablePlan::Apply { op, .. } = alter("ALTER TABLE users RENAME COLUMN name TO label")
    else {
        panic!("expected Apply");
    };
    assert_eq!(
        op,
        AlterColumnOp::RenameColumn {
            index: 1,
            to: "label".to_owned(),
        },
    );
}

#[test]
fn alter_rename_onto_existing_column_is_rejected() {
    assert!(matches!(
        plan("ALTER TABLE users RENAME COLUMN name TO age", &catalog()),
        Err(Error::DuplicateColumn { .. }),
    ));
}

#[test]
fn alter_column_set_type_resolves() {
    let AlterTablePlan::Apply { op, .. } =
        alter("ALTER TABLE users ALTER COLUMN age SET DATA TYPE TEXT")
    else {
        panic!("expected Apply");
    };
    assert_eq!(
        op,
        AlterColumnOp::SetType {
            index: 2,
            ty: ColumnType::Text,
        },
    );
}

#[test]
fn alter_column_set_and_drop_default_resolve() {
    // SET DEFAULT resolves to a column-default catalog op carrying the SQL text.
    let AlterTablePlan::Apply { op, .. } =
        alter("ALTER TABLE users ALTER COLUMN age SET DEFAULT 0")
    else {
        panic!("expected Apply");
    };
    assert_eq!(
        op,
        AlterColumnOp::SetDefault {
            column: "age".to_owned(),
            default_sql: "0".to_owned(),
        },
    );

    // DROP DEFAULT resolves to the matching removal op.
    let AlterTablePlan::Apply { op, .. } = alter("ALTER TABLE users ALTER COLUMN age DROP DEFAULT")
    else {
        panic!("expected Apply");
    };
    assert_eq!(
        op,
        AlterColumnOp::DropDefault {
            column: "age".to_owned(),
        },
    );

    // A type-incompatible default is rejected.
    assert!(matches!(
        plan(
            "ALTER TABLE users ALTER COLUMN age SET DEFAULT 'x'",
            &catalog()
        ),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn alter_missing_table_is_rejected() {
    assert!(matches!(
        plan("ALTER TABLE ghost ADD COLUMN x INT", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn alter_missing_table_if_exists_is_noop() {
    assert_eq!(
        alter("ALTER TABLE IF EXISTS ghost ADD COLUMN x INT"),
        AlterTablePlan::Noop,
    );
}

// --- ANALYZE ----------------------------------------------------------

#[test]
fn analyze_all_columns_expands() {
    let LogicalPlan::Analyze(p) = plan("ANALYZE TABLE users", &catalog()).unwrap() else {
        panic!("expected Analyze plan");
    };
    assert_eq!(p.columns, vec![0, 1, 2, 3, 4]);
}

#[test]
fn analyze_specific_columns_resolve() {
    let LogicalPlan::Analyze(p) =
        plan("ANALYZE TABLE users FOR COLUMNS id, age", &catalog()).unwrap()
    else {
        panic!("expected Analyze plan");
    };
    assert_eq!(p.columns, vec![0, 2]);
}

#[test]
fn analyze_missing_table_is_rejected() {
    assert!(matches!(
        plan("ANALYZE TABLE ghost", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn analyze_unknown_column_is_rejected() {
    assert!(matches!(
        plan("ANALYZE TABLE users FOR COLUMNS ghost", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn analyze_duplicate_column_is_rejected() {
    assert!(matches!(
        plan("ANALYZE TABLE users FOR COLUMNS id, id", &catalog()),
        Err(Error::DuplicateColumn { .. }),
    ));
}

// --- INSERT -----------------------------------------------------------

#[test]
fn insert_all_columns_ok() {
    let LogicalPlan::Insert(p) = plan(
        "INSERT INTO users VALUES (1, 'a', 30, 9.5, TRUE)",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Insert plan");
    };
    assert_eq!(p.columns, [0, 1, 2, 3, 4]);
    let InsertSource::Values(rows) = &p.source else {
        panic!("expected VALUES source");
    };
    assert_eq!(rows.len(), 1);
}

#[test]
fn insert_column_subset_ok() {
    let LogicalPlan::Insert(p) =
        plan("INSERT INTO users (id, name) VALUES (1, 'a')", &catalog()).unwrap()
    else {
        panic!("expected Insert plan");
    };
    assert_eq!(p.columns, [0, 1]);
}

#[test]
fn insert_unknown_table_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO ghost VALUES (1)", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn insert_unknown_column_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO users (ghost) VALUES (1)", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn insert_duplicate_target_column_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO users (id, id) VALUES (1, 2)", &catalog()),
        Err(Error::DuplicateColumn { .. }),
    ));
}

#[test]
fn insert_arity_mismatch_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO users (id, name) VALUES (1)", &catalog()),
        Err(Error::ArityMismatch { .. }),
    ));
}

#[test]
fn insert_type_mismatch_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO users (id) VALUES ('not an int')", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn insert_null_into_not_null_column_is_rejected() {
    assert!(matches!(
        plan("INSERT INTO users (id) VALUES (NULL)", &catalog()),
        Err(Error::NotNullViolation { .. }),
    ));
}

#[test]
fn insert_null_into_nullable_column_is_ok() {
    assert!(matches!(
        plan("INSERT INTO users (name) VALUES (NULL)", &catalog()),
        Ok(LogicalPlan::Insert(_)),
    ));
}

#[test]
fn insert_int_widens_into_float_column() {
    assert!(matches!(
        plan("INSERT INTO users (score) VALUES (3)", &catalog()),
        Ok(LogicalPlan::Insert(_)),
    ));
}

#[test]
fn insert_on_conflict_do_update_resolves_upsert() {
    // ON CONFLICT (cols) DO UPDATE resolves into an upsert plan: a bare column is the existing row,
    // `EXCLUDED.col` the proposed one.
    let LogicalPlan::Insert(p) = plan(
        "INSERT INTO users (id, name) VALUES (1, 'a') ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Insert plan");
    };
    assert!(matches!(
        p.on_conflict,
        Some(crate::planner::OnConflictPlan::DoUpdate { .. })
    ));

    // DO UPDATE requires a conflict target.
    assert!(matches!(
        plan(
            "INSERT INTO users (id) VALUES (1) ON CONFLICT DO UPDATE SET name = 'x'",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
    // A subquery in the SET value is rejected (the executor evaluates it against an in-memory row).
    assert!(matches!(
        plan(
            "INSERT INTO users (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET name = (SELECT 'x')",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn insert_from_select_analyzes_and_checks_columns() {
    // INSERT ... SELECT analyzes into an InsertSource::Select.
    assert!(matches!(
        plan(
            "INSERT INTO users (id, name) SELECT id, name FROM users",
            &catalog(),
        ),
        Ok(LogicalPlan::Insert(p)) if matches!(p.source, InsertSource::Select(_)),
    ));

    // Arity mismatch: the SELECT yields one column but two targets are listed.
    assert!(matches!(
        plan(
            "INSERT INTO users (id, name) SELECT id FROM users",
            &catalog(),
        ),
        Err(Error::ArityMismatch { .. }),
    ));

    // Type mismatch: a TEXT column cannot feed an INT target.
    assert!(matches!(
        plan("INSERT INTO users (id) SELECT name FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- RETURNING ------------------------------------------------

#[test]
fn insert_returning_resolves_against_the_inserted_row() {
    // INSERT ... RETURNING now resolves its projection against the table's columns.
    let LogicalPlan::Insert(p) = plan(
        "INSERT INTO users (id, name) VALUES (1, 'a') RETURNING id, name AS who",
        &catalog(),
    )
    .expect("INSERT RETURNING analyzes") else {
        panic!("expected Insert");
    };
    let names: Vec<&str> = p.returning.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["id", "who"]);
}

#[test]
fn insert_returning_star_expands_to_all_columns() {
    let LogicalPlan::Insert(p) = plan(
        "INSERT INTO users (id, name) VALUES (1, 'a') RETURNING *",
        &catalog(),
    )
    .expect("INSERT RETURNING * analyzes") else {
        panic!("expected Insert");
    };
    assert_eq!(p.returning.len(), 5, "RETURNING * expands to every column");
}

#[test]
fn insert_returning_unknown_column_is_rejected() {
    assert!(matches!(
        plan(
            "INSERT INTO users (id, name) VALUES (1, 'a') RETURNING nope",
            &catalog(),
        ),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn update_returning_resolves_against_the_table() {
    let LogicalPlan::Update(p) = plan(
        "UPDATE users SET name = 'b' WHERE id = 1 RETURNING id, name",
        &catalog(),
    )
    .expect("UPDATE RETURNING analyzes") else {
        panic!("expected Update");
    };
    let names: Vec<&str> = p.returning.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["id", "name"]);
}

#[test]
fn update_from_resolves_into_plan() {
    // UPDATE ... FROM a single named table resolves; the FROM table is carried for the executor and
    // the SET/WHERE may reference its (aliased) columns.
    let LogicalPlan::Update(p) = plan(
        "UPDATE users SET name = 'b' FROM users u WHERE users.id = u.id",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Update plan");
    };
    assert!(p.from.is_some());

    // A join in the FROM is still rejected (v1 supports a single FROM table).
    assert!(matches!(
        plan(
            "UPDATE users SET name = 'b' FROM users u JOIN users w ON u.id = w.id \
             WHERE users.id = u.id",
            &catalog()
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn delete_returning_resolves_against_the_table() {
    let LogicalPlan::Delete(p) = plan(
        "DELETE FROM users WHERE id = 1 RETURNING id AS gone",
        &catalog(),
    )
    .expect("DELETE RETURNING analyzes") else {
        panic!("expected Delete");
    };
    assert_eq!(p.returning.len(), 1);
    assert_eq!(p.returning[0].name, "gone");
}

#[test]
fn uncorrelated_subqueries_resolve() {
    for sql in [
        "SELECT (SELECT id FROM users) FROM users",
        "SELECT * FROM users WHERE EXISTS (SELECT 1 FROM users)",
        "SELECT * FROM users WHERE NOT EXISTS (SELECT 1 FROM users)",
        "SELECT * FROM users WHERE id IN (SELECT id FROM users)",
        "SELECT * FROM users WHERE id NOT IN (SELECT id FROM users)",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected to resolve: {sql}");
    }
}

#[test]
fn subquery_arity_and_type_are_checked() {
    // A scalar / IN subquery must project exactly one column.
    assert!(matches!(
        plan("SELECT (SELECT id, name FROM users) FROM users", &catalog(),),
        Err(Error::Unsupported(_)),
    ));
    // The IN probe type must match the subquery's single column.
    assert!(matches!(
        plan(
            "SELECT * FROM users WHERE name IN (SELECT id FROM users)",
            &catalog(),
        ),
        Err(Error::TypeMismatch { .. }),
    ));
    // A correlated reference (outer column inside the body) now resolves against the enclosing
    // scope — the outer `u.id` is visible to the subquery body.
    assert!(
        plan(
            "SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM users WHERE id = u.id)",
            &catalog(),
        )
        .is_ok(),
    );
    // A reference to a column in neither the inner nor any enclosing scope is still unknown.
    assert!(matches!(
        plan(
            "SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM users WHERE id = u.nope)",
            &catalog(),
        ),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn row_constructor_rejected_until_executor_path() {
    assert!(
        matches!(
            plan("SELECT ROW(id, name) FROM users", &catalog()),
            Err(Error::Unsupported(_))
        ),
        "ROW(...) should still be Unsupported",
    );
}

#[test]
fn similar_to_resolves_to_bool() {
    // Both forms analyze (TEXT SIMILAR TO TEXT -> BOOL), usable as a WHERE predicate.
    for sql in [
        "SELECT * FROM users WHERE name SIMILAR TO 'a%'",
        "SELECT * FROM users WHERE name NOT SIMILAR TO '(a|b)%'",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected {sql} to analyze");
    }
}

#[test]
fn regex_match_operator_resolves_to_bool() {
    // All four operator forms now analyze (TEXT ~ TEXT -> BOOL), usable as a WHERE predicate.
    for sql in [
        "SELECT * FROM users WHERE name ~ 'a.*'",
        "SELECT * FROM users WHERE name ~* 'a.*'",
        "SELECT * FROM users WHERE name !~ 'a.*'",
        "SELECT * FROM users WHERE name !~* 'a.*'",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for {sql}");
    }
    // A non-TEXT subject is rejected.
    assert!(matches!(
        plan("SELECT * FROM users WHERE age ~ 'a.*'", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn array_literal_resolves_to_array_type_and_subscript_to_element() {
    // ARRAY[...] resolves to ColumnType::Array(elem); a subscript to the element type.
    use nusadb_core::engine::ArrayElem;
    let LogicalPlan::Select(p) = plan("SELECT ARRAY[1, 2, 3]", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Array(ArrayElem::Int));

    // Int and a decimal literal unify to NUMERIC (a decimal literal is NUMERIC), so the array
    // is NUMERIC[] — matching the reference engine (`numeric[]`) rather than collapsing to float.
    let LogicalPlan::Select(p) = plan("SELECT ARRAY[1, 2.0]", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(
        p.projection[0].expr.ty,
        ColumnType::Array(ArrayElem::Numeric)
    );

    // Subscript yields the element's scalar type.
    let LogicalPlan::Select(p) = plan("SELECT ARRAY['a', 'b'][1]", &MockCatalog::new()).unwrap()
    else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Text);
}

#[test]
fn array_rejects_mixed_types_empty_and_bad_subscript() {
    // Incompatible element types.
    assert!(matches!(
        plan("SELECT ARRAY[1, 'a']", &MockCatalog::new()),
        Err(Error::TypeMismatch { .. }),
    ));
    // Empty array has no inferable element type.
    assert!(matches!(
        plan("SELECT ARRAY[]", &MockCatalog::new()),
        Err(Error::Unsupported(_)),
    ));
    // Subscripting a non-array is a type error.
    assert!(matches!(
        plan("SELECT age[1] FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn within_group_resolves_result_types() {
    // PERCENTILE_CONT → Float; PERCENTILE_DISC / MODE keep the ordering value's type.
    for (sql, ty) in [
        (
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) FROM users",
            ColumnType::Float,
        ),
        (
            "SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY age) FROM users",
            ColumnType::Int,
        ),
        (
            "SELECT MODE() WITHIN GROUP (ORDER BY name) FROM users",
            ColumnType::Text,
        ),
    ] {
        let LogicalPlan::Select(p) = plan(sql, &catalog()).unwrap() else {
            panic!("expected Select plan for `{sql}`");
        };
        assert_eq!(p.projection[0].expr.ty, ty, "for `{sql}`");
        assert_eq!(
            p.aggregates.len(),
            1,
            "registered as an aggregate (`{sql}`)"
        );
    }
}

#[test]
fn within_group_rejects_bad_fraction_type_and_order() {
    // PERCENTILE_CONT needs a numeric ordering value.
    assert!(matches!(
        plan(
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY name) FROM users",
            &catalog(),
        ),
        Err(Error::TypeMismatch { .. }),
    ));
    // Fraction outside [0, 1].
    assert!(matches!(
        plan(
            "SELECT PERCENTILE_CONT(2) WITHIN GROUP (ORDER BY age) FROM users",
            &catalog(),
        ),
        Err(Error::InvalidValue { .. }),
    ));
    // DESC ordering (and an explicit NULLS clause) are now accepted — the ordered set is reversed.
    assert!(
        plan(
            "SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY age DESC NULLS LAST) FROM users",
            &catalog(),
        )
        .is_ok()
    );
    // MODE takes no direct argument.
    assert!(matches!(
        plan(
            "SELECT MODE(1) WITHIN GROUP (ORDER BY age) FROM users",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
}

// --- Transaction & session control -------------------------

#[test]
fn plain_begin_still_executes() {
    // Plain BEGIN/COMMIT/ROLLBACK must keep planning (no regression from).
    assert!(matches!(
        plan("BEGIN", &catalog()),
        Ok(LogicalPlan::BeginTransaction(_)),
    ));
    assert!(matches!(
        plan("COMMIT", &catalog()),
        Ok(LogicalPlan::Commit)
    ));
    assert!(matches!(
        plan("ROLLBACK", &catalog()),
        Ok(LogicalPlan::Rollback),
    ));
}

#[test]
fn session_variable_statements_analyze() {
    // /SET/RESET/SHOW now analyze into session-variable plans.
    assert!(matches!(
        plan("SET search_path = 'public'", &catalog()),
        Ok(LogicalPlan::SetVariable { name, value: Some(v) }) if name == "search_path" && v == "public",
    ));
    assert!(matches!(
        plan("RESET search_path", &catalog()),
        Ok(LogicalPlan::SetVariable { name, value: None }) if name == "search_path",
    ));
    assert!(matches!(
        plan("SHOW search_path", &catalog()),
        Ok(LogicalPlan::ShowVariable(n)) if n == "search_path",
    ));
}

#[test]
fn savepoint_statements_analyze() {
    // SAVEPOINT / ROLLBACK TO / RELEASE now carry the name through to the executor.
    assert!(matches!(
        plan("SAVEPOINT sp1", &catalog()),
        Ok(LogicalPlan::Savepoint(n)) if n == "sp1",
    ));
    assert!(matches!(
        plan("ROLLBACK TO SAVEPOINT sp1", &catalog()),
        Ok(LogicalPlan::RollbackToSavepoint(n)) if n == "sp1",
    ));
    assert!(matches!(
        plan("RELEASE SAVEPOINT sp1", &catalog()),
        Ok(LogicalPlan::ReleaseSavepoint(n)) if n == "sp1",
    ));
}

#[test]
fn begin_and_set_transaction_carry_characteristics() {
    // BEGIN/SET TRANSACTION now analyze, threading the requested isolation + access mode
    // into the plan instead of being rejected.
    use nusadb_core::engine::IsolationLevel;

    let Ok(LogicalPlan::BeginTransaction(c)) =
        plan("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY", &catalog())
    else {
        panic!("BEGIN with characteristics should analyze");
    };
    assert_eq!(c.isolation, Some(IsolationLevel::Serializable));
    assert_eq!(c.read_only, Some(true));

    let Ok(LogicalPlan::SetTransaction(c)) = plan(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
        &catalog(),
    ) else {
        panic!("SET TRANSACTION should analyze");
    };
    assert_eq!(c.isolation, Some(IsolationLevel::RepeatableRead));
    assert_eq!(c.read_only, None);
}

#[test]
fn select_for_update_locks_single_table_else_unsupported() {
    // FOR UPDATE / FOR SHARE on a single base table is now accepted, including
    // SKIP LOCKED (the job-queue pattern, QA scale/production register).
    for sql in [
        "SELECT * FROM users FOR UPDATE",
        "SELECT * FROM users FOR SHARE",
        "SELECT * FROM users FOR UPDATE SKIP LOCKED",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for {sql}");
    }
    let LogicalPlan::Select(p) = plan("SELECT * FROM users FOR UPDATE SKIP LOCKED", &catalog())
        .expect("SKIP LOCKED should analyze")
    else {
        panic!("expected a SELECT plan");
    };
    assert_eq!(
        p.row_lock,
        Some((nusadb_core::engine::RowLockMode::Exclusive, true))
    );
    // Richer shapes and lock options remain honest `Unsupported` in v1 (NOWAIT because the
    // no-wait lock manager reports a conflict as 40001, not the reference engine's 55P03).
    for sql in [
        "SELECT COUNT(*) FROM users FOR UPDATE",  // aggregate
        "SELECT * FROM users FOR SHARE OF users", // OF <table>
        "SELECT * FROM users FOR UPDATE NOWAIT",  // NOWAIT
    ] {
        assert!(
            matches!(plan(sql, &catalog()), Err(Error::Unsupported(_))),
            "expected Unsupported for {sql}",
        );
    }
}

#[test]
fn distinct_on_resolves_keys() {
    // DISTINCT ON resolves its key expressions; plain DISTINCT still works.
    let LogicalPlan::Select(p) = plan(
        "SELECT DISTINCT ON (id) id, name FROM users ORDER BY id",
        &catalog(),
    )
    .expect("DISTINCT ON analyzes") else {
        panic!("expected Select");
    };
    assert_eq!(p.distinct_on.len(), 1);
    assert!(!p.distinct, "DISTINCT ON is not plain DISTINCT");
    assert!(matches!(
        plan("SELECT DISTINCT id FROM users", &catalog()),
        Ok(LogicalPlan::Select(_)),
    ));
    // DISTINCT ON with aggregation is rejected (out of scope for v1).
    assert!(matches!(
        plan(
            "SELECT DISTINCT ON (id) COUNT(*) FROM users GROUP BY id",
            &catalog()
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn merge_resolves_into_plan() {
    // MERGE over a named source resolves; the WHEN clauses are type-checked over target ++ source.
    let sql = "MERGE INTO users USING users s ON users.id = s.id \
                   WHEN MATCHED THEN DELETE \
                   WHEN NOT MATCHED THEN INSERT (id) VALUES (s.id)";
    let LogicalPlan::Merge(p) = plan(sql, &catalog()).unwrap() else {
        panic!("expected Merge plan");
    };
    assert_eq!(p.whens.len(), 2);

    // A derived subquery / VALUES source resolves too, carrying an inlined plan the executor runs.
    let LogicalPlan::Merge(p) = plan(
        "MERGE INTO users USING (SELECT 1 AS id) s ON users.id = s.id \
         WHEN MATCHED THEN DELETE",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Merge plan");
    };
    assert!(
        p.source_plan.is_some(),
        "a subquery source carries an inlined plan"
    );

    // A LATERAL source stays unsupported.
    assert!(matches!(
        plan(
            "MERGE INTO users USING LATERAL (SELECT 1 AS id) s ON users.id = s.id \
             WHEN MATCHED THEN DELETE",
            &catalog()
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn set_operations_resolve_and_check_compatibility() {
    // Compatible operands resolve to a SetOperation plan.
    for sql in [
        "SELECT id FROM users UNION SELECT id FROM users",
        "SELECT id FROM users INTERSECT SELECT id FROM users",
        "SELECT id FROM users EXCEPT ALL SELECT id FROM users",
        "SELECT id FROM users UNION SELECT age FROM users ORDER BY id",
    ] {
        assert!(
            matches!(plan(sql, &catalog()), Ok(LogicalPlan::SetOperation(_))),
            "expected a SetOperation plan for {sql}",
        );
    }
    // Column-count mismatch and per-column type mismatch are rejected.
    assert!(matches!(
        plan(
            "SELECT id FROM users UNION SELECT id, name FROM users",
            &catalog()
        ),
        Err(Error::ArityMismatch { .. }),
    ));
    assert!(matches!(
        plan(
            "SELECT id FROM users UNION SELECT name FROM users",
            &catalog()
        ),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn delete_using_resolves_into_plan() {
    // DELETE ... USING a single named table resolves; the USING table is carried for the executor.
    let LogicalPlan::Delete(p) = plan(
        "DELETE FROM users USING users u WHERE users.id = u.id",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Delete plan");
    };
    assert!(p.using.is_some());

    // A join in the USING is still rejected (v1 supports a single USING table).
    assert!(matches!(
        plan(
            "DELETE FROM users USING users u JOIN users w ON u.id = w.id WHERE users.id = u.id",
            &catalog()
        ),
        Err(Error::Unsupported(_)),
    ));
}

// --- WITH / CTE (non-recursive recursive) ----------------

#[test]
fn non_recursive_cte_resolves_against_its_columns() {
    // A non-recursive CTE referenced as the FROM base resolves; the outer query sees the CTE's
    // output columns.
    let LogicalPlan::Select(p) = plan(
        "WITH cte AS (SELECT id, name FROM users) SELECT name FROM cte WHERE id = 1",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    // The base source is the inlined CTE plan, not a catalog table.
    assert!(p.from_cte.is_some());
    assert!(p.table.is_none());
    assert_eq!(p.projection.len(), 1);
    // A CTE column not produced by its body is unknown to the outer query.
    assert!(matches!(
        plan(
            "WITH cte AS (SELECT id FROM users) SELECT name FROM cte",
            &catalog(),
        ),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn with_recursive_cte_resolves_base_and_recursive_terms() {
    // WITH RECURSIVE now resolves: the base term fixes the column shape and the recursive term's
    // self-reference type-checks against the CTE's synthetic table.
    let LogicalPlan::Select(p) = plan(
        "WITH RECURSIVE nums AS \
             (SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 5) \
             SELECT n FROM nums",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    // One recursive CTE def is threaded onto the plan; its body scans the synthetic table.
    assert_eq!(p.recursive_ctes.len(), 1);
    assert!(p.recursive_ctes[0].union_all);
    assert_eq!(p.projection.len(), 1);
    assert!(p.table.is_some());
    assert!(p.from_cte.is_none());
}

#[test]
fn with_recursive_flag_on_non_recursive_body_is_an_inline_cte() {
    // WITH RECURSIVE permits but does not require recursion: a recursive-flagged CTE with a plain
    // SELECT body (no UNION self-reference) resolves as an ordinary inline CTE.
    let LogicalPlan::Select(p) = plan(
        "WITH RECURSIVE cte AS (SELECT id FROM users) SELECT id FROM cte",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert!(p.recursive_ctes.is_empty());
    assert!(p.from_cte.is_some());
}

#[test]
fn with_recursive_cte_rejects_type_mismatch_between_terms() {
    // The recursive term must produce the same column types as the base term.
    assert!(matches!(
        plan(
            "WITH RECURSIVE t AS \
                 (SELECT 1 AS n UNION ALL SELECT name FROM users) \
                 SELECT n FROM t",
            &catalog(),
        ),
        Err(Error::TypeMismatch { .. } | Error::ArityMismatch { .. }),
    ));
}

// --- ORDER BY NULLS FIRST/LAST --------------------------------

#[test]
fn order_by_nulls_clause_resolves_and_carries_placement() {
    // NULLS FIRST/LAST now resolves and the placement is threaded onto the OrderByKey
    // (the comparator honours it; covered end-to-end in nusadb-e2e).
    use crate::ast::NullOrdering;
    let LogicalPlan::Select(p) = plan(
        "SELECT * FROM users ORDER BY id ASC NULLS FIRST",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.order_by[0].nulls, NullOrdering::First);

    let LogicalPlan::Select(p) = plan(
        "SELECT * FROM users ORDER BY id DESC NULLS LAST",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.order_by[0].nulls, NullOrdering::Last);
}

#[test]
fn order_by_without_nulls_clause_still_works() {
    // Plain ORDER BY (no explicit NULLS) must still plan fine — default placement is honoured.
    assert!(matches!(
        plan("SELECT * FROM users ORDER BY id ASC", &catalog()),
        Ok(LogicalPlan::Select(_)),
    ));
}

// --- ORDER BY position / alias --------------------------------

#[test]
fn order_by_position_resolves_to_output_column() {
    // `ORDER BY 2` sorts by the 2nd output column (`name`), not by the constant 2.
    let LogicalPlan::Select(p) = plan("SELECT id, name FROM users ORDER BY 2", &catalog()).unwrap()
    else {
        panic!("expected Select plan");
    };
    assert_eq!(p.order_by.len(), 1);
    // 2nd projection is `name`; the order key must equal that projection's expr.
    assert_eq!(p.order_by[0].expr, p.projection[1].expr);
}

#[test]
fn order_by_position_out_of_range_is_rejected() {
    assert!(matches!(
        plan("SELECT id, name FROM users ORDER BY 5", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn order_by_position_zero_is_rejected() {
    assert!(matches!(
        plan("SELECT id FROM users ORDER BY 0", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn order_by_alias_resolves_to_output_column() {
    // `ORDER BY total` resolves to the aliased output column, which has no source column.
    let LogicalPlan::Select(p) = plan(
        "SELECT age + 1 AS total FROM users ORDER BY total",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.order_by[0].expr, p.projection[0].expr);
}

#[test]
fn order_by_plain_column_still_resolves() {
    // A bare source column (also an output column) still resolves correctly.
    let LogicalPlan::Select(p) =
        plan("SELECT id, name FROM users ORDER BY age", &catalog()).unwrap()
    else {
        panic!("expected Select plan");
    };
    assert_eq!(p.order_by.len(), 1);
}

#[test]
fn order_by_unknown_name_still_rejected() {
    assert!(plan("SELECT id FROM users ORDER BY ghost", &catalog()).is_err());
}

// --- LIKE ... ESCAPE ------------------------------------------

#[test]
fn like_escape_resolves() {
    // LIKE ... ESCAPE now resolves end-to-end (the comparator honours the escape char).
    assert!(matches!(
        plan(
            "SELECT * FROM users WHERE name LIKE 'a!%' ESCAPE '!'",
            &catalog()
        ),
        Ok(LogicalPlan::Select(_)),
    ));
}

// --- FETCH FIRST / OFFSET -------------------------------------

#[test]
fn offset_analyzes() {
    // OFFSET (with or without LIMIT) now resolves end-to-end.
    assert!(plan("SELECT * FROM users OFFSET 5 ROWS", &catalog()).is_ok());
    assert!(
        plan(
            "SELECT * FROM users ORDER BY id LIMIT 2 OFFSET 1",
            &catalog()
        )
        .is_ok()
    );
}

// --- Aggregate FILTER -----------------------------------------

#[test]
fn aggregate_filter_resolves_with_carried_predicate() {
    // FILTER (WHERE pred) now resolves, carrying a boolean predicate onto the AggregateCall.
    let LogicalPlan::Select(p) = plan(
        "SELECT COUNT(*) FILTER (WHERE id > 0), SUM(age) FILTER (WHERE active) FROM users",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.aggregates.len(), 2);
    assert!(
        p.aggregates[0].filter.is_some(),
        "COUNT FILTER predicate carried"
    );
    assert!(
        p.aggregates[1].filter.is_some(),
        "SUM FILTER predicate carried"
    );
    // A non-boolean FILTER predicate is rejected.
    assert!(matches!(
        plan("SELECT COUNT(*) FILTER (WHERE name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- Aggregate DISTINCT ----------------------------------------

#[test]
fn aggregate_distinct_resolves_with_carried_flag() {
    // DISTINCT inside an aggregate now resolves, carrying the flag onto the AggregateCall.
    let LogicalPlan::Select(p) = plan(
        "SELECT COUNT(DISTINCT id), SUM(DISTINCT age) FROM users",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.aggregates.len(), 2);
    assert!(p.aggregates.iter().all(|c| c.distinct));
    // A plain (non-DISTINCT) aggregate keeps the flag false.
    let LogicalPlan::Select(p) = plan("SELECT COUNT(id) FROM users", &catalog()).unwrap() else {
        panic!("expected Select plan");
    };
    assert!(!p.aggregates[0].distinct);
}

#[test]
fn count_distinct_star_is_rejected() {
    // DISTINCT needs a concrete argument to dedupe — COUNT(DISTINCT *) is meaningless.
    assert!(matches!(
        plan("SELECT COUNT(DISTINCT *) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

// --- GROUP BY ROLLUP / CUBE / GROUPING SETS ------------------

#[test]
fn group_by_rollup_expands_to_prefix_sets() {
    // ROLLUP(id, name) → the prefixes {id, name}, {id}, {}.
    let LogicalPlan::Select(p) = plan(
        "SELECT id, name, COUNT(*) FROM users GROUP BY ROLLUP (id, name)",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.group_keys.len(), 2);
    assert_eq!(
        p.grouping_sets,
        vec![vec![0, 1], vec![0], Vec::<usize>::new()]
    );
}

#[test]
fn group_by_cube_expands_to_all_subsets() {
    // CUBE(id, name) → every subset; the full set and the empty set both appear.
    let LogicalPlan::Select(p) = plan(
        "SELECT id, name, COUNT(*) FROM users GROUP BY CUBE (id, name)",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.group_keys.len(), 2);
    assert_eq!(p.grouping_sets.len(), 4);
    assert!(p.grouping_sets.contains(&vec![0, 1]));
    assert!(p.grouping_sets.contains(&Vec::new()));
}

#[test]
fn group_by_grouping_sets_taken_verbatim() {
    // GROUPING SETS are used as listed; the union of columns forms `group_keys`.
    let LogicalPlan::Select(p) = plan(
        "SELECT id, name, COUNT(*) FROM users GROUP BY GROUPING SETS ((id, name), (id), ())",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(
        p.grouping_sets,
        vec![vec![0, 1], vec![0], Vec::<usize>::new()]
    );
}

#[test]
fn grouping_function_becomes_a_synthetic_aggregate_over_grouping_sets() {
    // GROUPING(id) with a real super-aggregate resolves into an appended synthetic
    // `Grouping` aggregate carrying the named key's index; the projection references it like any
    // other aggregate slot.
    let LogicalPlan::Select(p) = plan(
        "SELECT id, GROUPING(id), COUNT(*) FROM users GROUP BY ROLLUP (id)",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    let grouping = p
        .aggregates
        .iter()
        .find(|a| a.func == crate::ast::AggregateFunc::Grouping)
        .expect("a synthetic GROUPING aggregate call");
    assert_eq!(grouping.grouping_args, vec![0]);
}

#[test]
fn grouping_function_in_a_plain_group_by_folds_to_zero() {
    // Without grouping sets nothing is ever rolled up, so GROUPING(id) is the constant 0 — no
    // synthetic aggregate is created for it.
    let LogicalPlan::Select(p) =
        plan("SELECT id, GROUPING(id) FROM users GROUP BY id", &catalog()).unwrap()
    else {
        panic!("expected Select plan");
    };
    assert!(
        !p.aggregates
            .iter()
            .any(|a| a.func == crate::ast::AggregateFunc::Grouping),
        "plain GROUP BY must not synthesize a GROUPING aggregate"
    );
    // The second projection item is the folded constant 0.
    assert!(matches!(
        p.projection[1].expr.kind,
        TypedExprKind::Literal(crate::ast::Value::Int(0))
    ));
}

#[test]
fn grouping_of_a_non_group_key_is_rejected() {
    // GROUPING's argument must be a GROUP BY key.
    assert!(matches!(
        plan(
            "SELECT id, GROUPING(name) FROM users GROUP BY ROLLUP (id)",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn cube_with_too_many_elements_is_rejected() {
    // Regression (the design #11): a wide CUBE would overflow `1 << n` / explode memory — reject it
    // before expansion instead of panicking.
    let elements = (0..17)
        .map(|i| vec![crate::ast::Expr::Column(format!("c{i}"))])
        .collect();
    let group_by = crate::ast::GroupBy::Cube(elements);
    assert!(matches!(
        super::expand_grouping(&group_by),
        Err(Error::Unsupported(_)),
    ));
}

// --- Window functions --------------------------------------

#[test]
fn window_function_is_extracted_and_referenced_as_appended_column() {
    // `users` has 5 columns, so the window result lands at ordinal 5.
    let LogicalPlan::Select(p) = plan(
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY name ORDER BY id) FROM users",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.windows.len(), 1);
    assert_eq!(p.windows[0].partition.len(), 1);
    assert_eq!(p.windows[0].order.len(), 1);
    assert_eq!(p.windows[0].result_ty, ColumnType::Int);
    // The projection references the window result as a plain appended column.
    assert_eq!(p.projection.len(), 2);
    assert_eq!(p.projection[1].expr.kind, TypedExprKind::Column(5));
    assert_eq!(p.projection[1].name, "row_number");
}

#[test]
fn window_aggregate_keeps_argument_type() {
    let LogicalPlan::Select(p) = plan(
        "SELECT SUM(age) OVER (PARTITION BY name) AS s FROM users",
        &catalog(),
    )
    .unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.windows.len(), 1);
    assert_eq!(p.windows[0].result_ty, ColumnType::Int); // SUM(Int) -> Int
    assert_eq!(p.projection[0].name, "s");
}

#[test]
fn window_with_aggregation_is_rejected() {
    // A window function alongside GROUP BY / aggregation is out of scope for v1.
    assert!(matches!(
        plan(
            "SELECT id, COUNT(*), ROW_NUMBER() OVER () FROM users GROUP BY id",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn window_frames_resolve_incl_range_and_groups_offsets() {
    // ROWS / GROUPS / RANGE frames resolve, including a RANGE value offset over an integer ordering
    // column in either direction…
    for sql in [
        "SELECT SUM(age) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM users",
        "SELECT SUM(age) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM users",
        "SELECT SUM(age) OVER (ORDER BY id RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM users",
        "SELECT SUM(age) OVER (ORDER BY id RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM users",
        // A RANGE value offset over a DESC ordering resolves (the executor reverses the direction).
        "SELECT SUM(age) OVER (ORDER BY id DESC RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM users",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "should resolve: {sql}");
    }
    // …but a RANGE value offset requires exactly one ORDER BY column (multi-column is unsupported).
    assert!(matches!(
        plan(
            "SELECT SUM(age) OVER (ORDER BY id, age RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM users",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
    // A GROUPS frame *with* an offset resolves — the offset counts peer groups, not a value.
    assert!(plan(
        "SELECT SUM(age) OVER (ORDER BY id GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM users",
        &catalog(),
    )
    .is_ok());
    // A GROUPS frame requires ORDER BY (peer groups are undefined without an ordering)
    assert!(matches!(
        plan(
            "SELECT SUM(age) OVER (GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM users",
            &catalog(),
        ),
        Err(Error::Unsupported(_)),
    ));
    // …but a GROUPS frame *with* ORDER BY resolves.
    assert!(plan(
        "SELECT SUM(age) OVER (ORDER BY id GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM users",
        &catalog(),
    )
    .is_ok());
}

#[test]
fn window_function_nested_in_an_expression_is_supported() {
    // A window call nested inside a larger expression is extracted to a synthetic column and the
    // surrounding expression type-checks normally (e.g. `... OVER () + 1`, `CAST(... OVER () AS …)`).
    for sql in [
        "SELECT ROW_NUMBER() OVER () + 1 FROM users",
        "SELECT CAST(ROW_NUMBER() OVER () AS INT) FROM users",
        "SELECT ROW_NUMBER() OVER () + RANK() OVER (ORDER BY age) FROM users",
    ] {
        assert!(
            matches!(plan(sql, &catalog()), Ok(LogicalPlan::Select(_))),
            "expected a Select plan for `{sql}`",
        );
    }
}

// --- CROSS / NATURAL JOIN / USING -----------------------------

#[test]
fn cross_natural_using_joins_analyze() {
    // All three now resolve end-to-end (CROSS → predicate true; NATURAL/USING → equality conjunction).
    for sql in [
        "SELECT * FROM users CROSS JOIN users u2",
        "SELECT * FROM users NATURAL JOIN users u2",
        "SELECT * FROM users JOIN users u2 USING (id)",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected {sql} to analyze");
    }
}

// --- SELECT -----------------------------------------------------------

#[test]
fn select_wildcard_expands_to_all_columns() {
    let LogicalPlan::Select(p) = plan("SELECT * FROM users", &catalog()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection.len(), 5);
    assert_eq!(p.projection[0].name, "id");
}

#[test]
fn select_unknown_table_is_rejected() {
    assert!(matches!(
        plan("SELECT * FROM ghost", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn select_unknown_column_is_rejected() {
    assert!(matches!(
        plan("SELECT ghost FROM users", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn select_where_must_be_boolean() {
    assert!(matches!(
        plan("SELECT * FROM users WHERE name", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn select_where_boolean_is_ok() {
    assert!(matches!(
        plan("SELECT * FROM users WHERE age > 18", &catalog()),
        Ok(LogicalPlan::Select(_)),
    ));
}

#[test]
fn select_without_from_is_ok() {
    let LogicalPlan::Select(p) = plan("SELECT 1", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert!(p.table.is_none());
    assert_eq!(p.projection[0].expr.ty, ColumnType::Int);
}

#[test]
fn bare_null_defaults_to_text_but_composite_null_stays_ambiguous() {
    // A bare `SELECT NULL` defaults its column to TEXT (the "unknown" resolution) rather than erroring.
    let LogicalPlan::Select(p) = plan("SELECT NULL", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Text);
    // Arithmetic on two bare NULLs is genuinely ambiguous (no unique operator to resolve).
    assert!(matches!(
        plan("SELECT NULL + NULL", &MockCatalog::new()),
        Err(Error::AmbiguousNull { .. }),
    ));
    // But a comparison / logical / concatenation of two bare NULLs resolves (unknown -> default
    // type) and evaluates to NULL — `NULL = NULL` is BOOL, `NULL || NULL` is TEXT.
    let LogicalPlan::Select(p) = plan("SELECT NULL = NULL", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Bool);
    let LogicalPlan::Select(p) = plan("SELECT NULL AND NULL", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Bool);
    let LogicalPlan::Select(p) = plan("SELECT NULL || NULL", &MockCatalog::new()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Text);
    // GREATEST / LEAST of every-NULL arguments resolves to an untyped NULL (-> TEXT), like COALESCE.
    let LogicalPlan::Select(p) = plan("SELECT GREATEST(NULL, NULL)", &MockCatalog::new()).unwrap()
    else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Text);
}

// --- Expressions ------------------------------------------------------

#[test]
fn arithmetic_result_type_is_inferred() {
    let LogicalPlan::Select(p) = plan("SELECT age + 1 FROM users", &catalog()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Int);
}

#[test]
fn arithmetic_widens_to_float() {
    let LogicalPlan::Select(p) = plan("SELECT age + score FROM users", &catalog()).unwrap() else {
        panic!("expected Select plan");
    };
    assert_eq!(p.projection[0].expr.ty, ColumnType::Float);
}

#[test]
fn logical_operator_requires_boolean_operands() {
    assert!(matches!(
        plan("SELECT * FROM users WHERE name AND active", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn comparison_of_incompatible_types_is_rejected() {
    assert!(matches!(
        plan("SELECT * FROM users WHERE name = age", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn null_comparison_takes_sibling_type() {
    assert!(matches!(
        plan("SELECT * FROM users WHERE id = NULL", &catalog()),
        Ok(LogicalPlan::Select(_)),
    ));
}

// --- UPDATE / DELETE --------------------------------------------------

#[test]
fn update_ok() {
    let LogicalPlan::Update(p) =
        plan("UPDATE users SET age = 40 WHERE id = 1", &catalog()).unwrap()
    else {
        panic!("expected Update plan");
    };
    assert_eq!(p.assignments.len(), 1);
    assert_eq!(p.assignments[0].column, 2);
    assert!(p.filter.is_some());
}

#[test]
fn update_unknown_column_is_rejected() {
    assert!(matches!(
        plan("UPDATE users SET ghost = 1", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

#[test]
fn update_duplicate_assignment_is_rejected() {
    assert!(matches!(
        plan("UPDATE users SET age = 1, age = 2", &catalog()),
        Err(Error::DuplicateColumn { .. }),
    ));
}

#[test]
fn update_type_mismatch_is_rejected() {
    assert!(matches!(
        plan("UPDATE users SET id = 'x'", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn update_can_reference_columns_in_set() {
    let LogicalPlan::Update(p) = plan("UPDATE users SET age = age + 1", &catalog()).unwrap() else {
        panic!("expected Update plan");
    };
    assert!(matches!(
        p.assignments[0].value.kind,
        TypedExprKind::Binary { .. }
    ));
}

#[test]
fn delete_ok() {
    assert!(matches!(
        plan("DELETE FROM users WHERE id = 1", &catalog()),
        Ok(LogicalPlan::Delete(_)),
    ));
}

#[test]
fn delete_where_must_be_boolean() {
    assert!(matches!(
        plan("DELETE FROM users WHERE age", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- COMMENT ON ----------------------------------------------

#[test]
fn comment_on_table_resolves() {
    let LogicalPlan::Comment(p) = plan("COMMENT ON TABLE users IS 'accounts'", &catalog()).unwrap()
    else {
        panic!("expected Comment plan");
    };
    assert_eq!(p.table, "users");
    assert_eq!(p.column, None);
    assert_eq!(p.comment.as_deref(), Some("accounts"));
}

#[test]
fn comment_on_column_resolves() {
    let LogicalPlan::Comment(p) = plan("COMMENT ON COLUMN users.name IS NULL", &catalog()).unwrap()
    else {
        panic!("expected Comment plan");
    };
    assert_eq!(p.column.as_deref(), Some("name"));
    assert!(p.comment.is_none());
}

#[test]
fn comment_on_missing_table_is_rejected() {
    assert!(matches!(
        plan("COMMENT ON TABLE ghost IS 'x'", &catalog()),
        Err(Error::TableNotFound { .. }),
    ));
}

#[test]
fn comment_on_missing_column_is_rejected() {
    assert!(matches!(
        plan("COMMENT ON COLUMN users.ghost IS 'x'", &catalog()),
        Err(Error::ColumnNotFound { .. }),
    ));
}

// --- Scalar string functions ----------------------------------

#[test]
fn scalar_function_accepts_well_typed_calls() {
    // Each of these resolves cleanly against `users(name TEXT, age INT)`.
    for sql in [
        "SELECT UPPER(name) FROM users",
        "SELECT LOWER(name) FROM users",
        "SELECT LENGTH(name) FROM users",
        "SELECT SUBSTRING(name, 2, 3) FROM users",
        "SELECT SUBSTRING(name FROM age) FROM users",
        "SELECT REPLACE(name, 'a', 'b') FROM users",
        "SELECT POSITION('a' IN name) FROM users",
        "SELECT LPAD(name, age, '-') FROM users",
        "SELECT RPAD(name, 4) FROM users",
        "SELECT TRIM(LEADING 'x' FROM name) FROM users",
        "SELECT BTRIM(name) FROM users",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for `{sql}`");
    }
}

#[test]
fn scalar_function_result_types_propagate() {
    // LENGTH -> Int, usable in arithmetic; UPPER -> Text, usable in concatenation.
    assert!(plan("SELECT LENGTH(name) + 1 FROM users", &catalog()).is_ok());
    assert!(plan("SELECT UPPER(name) || 'x' FROM users", &catalog()).is_ok());
    // POSITION -> Int, comparable.
    assert!(
        plan(
            "SELECT name FROM users WHERE POSITION('a' IN name) > 0",
            &catalog()
        )
        .is_ok()
    );
}

/// `events(d DATE, t TIME, ts TIMESTAMP, tstz TIMESTAMPTZ, u UUID, iv INTERVAL)`.
fn events() -> TableSchema {
    TableSchema {
        schema: "public".to_owned(),
        id: TableId(2),
        name: "events".to_owned(),
        columns: vec![
            col("d", ColumnType::Date, true),
            col("t", ColumnType::Time, true),
            col("ts", ColumnType::Timestamp, true),
            col("tstz", ColumnType::TimestampTz, true),
            col("u", ColumnType::Uuid, true),
            col("iv", ColumnType::Interval, true),
        ],
    }
}

#[test]
fn temporal_and_uuid_columns_compare_in_between_in_and_case() {
    // The executor's `compare` orders every temporal type and UUID, and `=`/`<` already type-check
    // on them — so BETWEEN / IN / simple-CASE must accept them too rather than raising a spurious
    // TypeMismatch. Each query exercises `comparable` for a same-type temporal/UUID pair.
    let cat = MockCatalog::new().with(events());
    for sql in [
        "SELECT * FROM events WHERE d BETWEEN d AND d",
        "SELECT * FROM events WHERE t BETWEEN t AND t",
        "SELECT * FROM events WHERE ts BETWEEN ts AND ts",
        "SELECT * FROM events WHERE tstz BETWEEN tstz AND tstz",
        "SELECT * FROM events WHERE iv BETWEEN iv AND iv",
        "SELECT * FROM events WHERE u IN (u)",
        "SELECT * FROM events WHERE d IN (d)",
        "SELECT CASE u WHEN u THEN 1 ELSE 0 END FROM events",
        "SELECT CASE d WHEN d THEN 1 ELSE 0 END FROM events",
    ] {
        assert!(plan(sql, &cat).is_ok(), "should type-check: {sql}");
    }
}

#[test]
fn cross_type_temporal_comparison_is_still_rejected() {
    // Widening `comparable` must not let genuinely different types through: a DATE vs TIMESTAMP
    // BETWEEN/IN is still a TypeMismatch, exactly as `=`/`<` already reject it.
    let cat = MockCatalog::new().with(events());
    for sql in [
        "SELECT * FROM events WHERE d BETWEEN ts AND ts",
        "SELECT * FROM events WHERE u IN (d)",
        "SELECT CASE d WHEN u THEN 1 ELSE 0 END FROM events",
    ] {
        assert!(
            matches!(plan(sql, &cat), Err(Error::TypeMismatch { .. })),
            "should reject cross-type: {sql}"
        );
    }
}

#[test]
fn simple_case_untyped_null_operand_type_checks_from_when_values() {
    // A simple CASE with a bare untyped NULL operand types the operand from its WHEN values instead of
    // raising "cannot infer the type of NULL"; the result type still comes from the THEN/ELSE branches.
    let cat = catalog();
    for sql in [
        "SELECT CASE NULL WHEN NULL THEN 1 ELSE 2 END",
        "SELECT CASE NULL WHEN 5 THEN 1 ELSE 2 END",
        "SELECT CASE NULL WHEN 'a' THEN 1 ELSE 2 END",
        "SELECT CASE NULL WHEN age THEN 1 ELSE 2 END FROM users",
    ] {
        assert!(plan(sql, &cat).is_ok(), "should type-check: {sql}");
    }
    // The operand takes the first typed WHEN's type; a later WHEN of an incomparable type is still a
    // real TypeMismatch (the fix only supplies a type for the NULL, it does not relax comparability).
    assert!(
        matches!(
            plan(
                "SELECT CASE NULL WHEN 5 THEN 1 WHEN 'a' THEN 2 ELSE 3 END",
                &cat
            ),
            Err(Error::TypeMismatch { .. })
        ),
        "mixed-type WHEN values must still be rejected"
    );
}

#[test]
fn string_literal_compared_to_temporal_column_coerces() {
    // A bare string literal is an unknown type that adopts the temporal / UUID type it is compared
    // against (like the reference engine), so ORM-style date filters type-check without an explicit
    // cast. Once `bind_parameters` substitutes a `$1` date parameter (a driver sends it as text), the
    // bound statement is exactly one of these string-literal comparisons — so this is also the fix
    // for `WHERE created_at >= $1`.
    let cat = MockCatalog::new().with(events());
    for sql in [
        "SELECT * FROM events WHERE d >= '2026-01-01'",
        "SELECT * FROM events WHERE d = '2026-01-01'",
        "SELECT * FROM events WHERE '2026-01-01' <= d",
        "SELECT * FROM events WHERE ts < '2026-01-01 12:00:00'",
        "SELECT * FROM events WHERE tstz >= '2026-01-01T00:00:00Z'",
        "SELECT * FROM events WHERE t >= '12:00:00'",
        "SELECT * FROM events WHERE u = '00000000-0000-0000-0000-000000000000'",
        "SELECT * FROM events WHERE d BETWEEN '2026-01-01' AND '2026-12-31'",
        "SELECT * FROM events WHERE d IN ('2026-01-01', '2026-06-15')",
        "SELECT * FROM events WHERE d IS DISTINCT FROM '2026-01-01'",
    ] {
        assert!(plan(sql, &cat).is_ok(), "should coerce + type-check: {sql}");
    }
}

#[test]
fn bound_temporal_parameter_type_checks_end_to_end() {
    // The exact ORM/extended-query path: a `$1` date filter is parsed, the driver binds the value as
    // text (a `datetime.date` serializes to `'2026-01-01'`), and the *bound* statement must analyze
    // without an explicit `::date`. Previously this raised `TypeMismatch: expected Date, found Text`.
    let cat = MockCatalog::new().with(events());
    let cases: &[(&str, &[u8])] = &[
        ("SELECT * FROM events WHERE d >= $1", b"2026-01-01"),
        ("SELECT * FROM events WHERE ts < $1", b"2026-01-01 12:00:00"),
        (
            "SELECT * FROM events WHERE d BETWEEN $1 AND $1",
            b"2026-01-01",
        ),
        (
            "SELECT * FROM events WHERE u = $1",
            b"00000000-0000-0000-0000-000000000000",
        ),
    ];
    for (sql, param) in cases {
        let stmt = crate::parser::parse(sql).unwrap();
        let bound = crate::params::bind_parameters(stmt, &[Some(param.to_vec())]).unwrap();
        assert!(
            analyze(bound, &cat).is_ok(),
            "bound date/uuid parameter should type-check: {sql}"
        );
    }
}

#[test]
fn bound_array_parameter_in_any_type_checks() {
    // `WHERE id = ANY($1)` is the ORM bulk-lookup shape; a driver binds the array as its `{...}` text
    // form, so the bound statement is `id = ANY('{...}')`. The TEXT literal must coerce to an array of
    // the probe's type (like `$1::int[]`) instead of raising "type mismatch in ANY/ALL right operand".
    let cat = catalog();
    for (sql, param) in [
        (
            "SELECT id FROM users WHERE id = ANY($1)",
            b"{1,2,3}".as_slice(),
        ),
        (
            "SELECT id FROM users WHERE age <> ALL($1)",
            b"{4,5}".as_slice(),
        ),
        (
            "SELECT id FROM users WHERE name = ANY($1)",
            b"{a,b}".as_slice(),
        ),
    ] {
        let stmt = crate::parser::parse(sql).unwrap();
        let bound = crate::params::bind_parameters(stmt, &[Some(param.to_vec())]).unwrap();
        assert!(
            analyze(bound, &cat).is_ok(),
            "bound array parameter should type-check: {sql}"
        );
    }
    // A probe type that cannot be an array element (JSON) is still a real mismatch — the coercion only
    // supplies an array type for scalar-elementable probes, it does not force every TEXT into an array.
    assert!(
        matches!(
            plan("SELECT id FROM users WHERE age = ANY(42)", &cat),
            Err(Error::TypeMismatch { .. })
        ),
        "a non-array, non-coercible ANY operand must still be rejected"
    );
}

#[test]
fn only_a_bare_string_literal_coerces_not_a_text_expression() {
    // The unknown-literal rule applies to string *literals* only. A genuinely TEXT-typed expression
    // (here `UPPER(...)`, a function result) versus a temporal column stays a real TypeMismatch —
    // matching the reference engine, where only string literals are "unknown".
    let cat = MockCatalog::new().with(events());
    for sql in [
        "SELECT * FROM events WHERE d = UPPER('2026-01-01')",
        "SELECT * FROM events WHERE d BETWEEN UPPER('a') AND UPPER('b')",
    ] {
        assert!(
            matches!(plan(sql, &cat), Err(Error::TypeMismatch { .. })),
            "a TEXT expression (not a literal) must stay a mismatch: {sql}"
        );
    }
}

#[test]
fn scalar_function_rejects_wrong_argument_type() {
    // `age` is INT, but UPPER/LENGTH expect TEXT.
    assert!(matches!(
        plan("SELECT UPPER(age) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // A TEXT second argument is the regex form, no longer a mismatch;
    // a non-INT, non-TEXT start is still rejected.
    assert!(plan("SELECT SUBSTRING(name, name) FROM users", &catalog()).is_ok());
    assert!(matches!(
        plan("SELECT SUBSTRING(name, age = 1) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn scalar_function_rejects_wrong_arity() {
    // UPPER takes exactly one argument; REPLACE exactly three.
    assert!(matches!(
        plan("SELECT UPPER(name, name) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        plan("SELECT REPLACE(name, 'a') FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn clock_functions_resolve_to_temporal_types() {
    // Niladic clock built-ins resolve with no FROM and carry the right result type.
    for (sql, ty) in [
        ("SELECT NOW()", ColumnType::TimestampTz),
        ("SELECT CURRENT_TIMESTAMP", ColumnType::TimestampTz),
        ("SELECT CURRENT_DATE", ColumnType::Date),
        ("SELECT CURRENT_TIME", ColumnType::Time),
    ] {
        let LogicalPlan::Select(p) = plan(sql, &MockCatalog::new()).unwrap() else {
            panic!("expected Select plan for `{sql}`");
        };
        assert_eq!(p.projection[0].expr.ty, ty, "for `{sql}`");
    }
}

#[test]
fn clock_functions_reject_arguments() {
    // They are niladic (arity 0) — passing an argument is an arity error.
    assert!(matches!(
        plan("SELECT NOW(1)", &MockCatalog::new()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn temporal_functions_resolve_result_types() {
    // EXTRACT → Float, DATE_TRUNC → source temporal type, AGE → Interval.
    for (sql, ty) in [
        (
            "SELECT EXTRACT(YEAR FROM CURRENT_TIMESTAMP)",
            ColumnType::Float,
        ),
        ("SELECT DATE_TRUNC('month', NOW())", ColumnType::TimestampTz),
        ("SELECT AGE(NOW(), NOW())", ColumnType::Interval),
        ("SELECT AGE(NOW())", ColumnType::Interval),
    ] {
        let LogicalPlan::Select(p) = plan(sql, &MockCatalog::new()).unwrap() else {
            panic!("expected Select plan for `{sql}`");
        };
        assert_eq!(p.projection[0].expr.ty, ty, "for `{sql}`");
    }
}

#[test]
fn temporal_functions_reject_bad_field_and_type_and_arity() {
    // Unknown field.
    assert!(matches!(
        plan("SELECT EXTRACT(CENTURY FROM NOW())", &MockCatalog::new()),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        plan("SELECT DATE_TRUNC('fortnight', NOW())", &MockCatalog::new()),
        Err(Error::Unsupported(_)),
    ));
    // Non-temporal source.
    assert!(matches!(
        plan("SELECT EXTRACT(YEAR FROM name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // AGE arity (0 or >2).
    assert!(matches!(
        plan("SELECT AGE(NOW(), NOW(), NOW())", &MockCatalog::new()),
        Err(Error::Unsupported(_)),
    ));
}

#[test]
fn to_char_date_timestamp_resolve_result_types() {
    // TO_CHAR → Text, TO_DATE → Date, TO_TIMESTAMP → Timestamp.
    for (sql, ty) in [
        ("SELECT TO_CHAR(NOW(), 'YYYY')", ColumnType::Text),
        (
            "SELECT TO_DATE('2024-06-15', 'YYYY-MM-DD')",
            ColumnType::Date,
        ),
        (
            "SELECT TO_TIMESTAMP('2024-06-15', 'YYYY-MM-DD')",
            ColumnType::Timestamp,
        ),
    ] {
        let LogicalPlan::Select(p) = plan(sql, &MockCatalog::new()).unwrap() else {
            panic!("expected Select plan for `{sql}`");
        };
        assert_eq!(p.projection[0].expr.ty, ty, "for `{sql}`");
    }
}

#[test]
fn to_char_rejects_non_temporal_and_to_date_rejects_non_text() {
    // TO_CHAR needs a temporal first argument…
    assert!(matches!(
        plan("SELECT TO_CHAR(name, 'YYYY') FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // …and TO_DATE needs text, not a number.
    assert!(matches!(
        plan("SELECT TO_DATE(age, 'YYYY') FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- string functions -------------------------------------------

#[test]
fn b448_functions_accept_well_typed_calls() {
    for sql in [
        "SELECT CONCAT(name, name) FROM users",
        "SELECT CONCAT(name, name, name) FROM users",
        "SELECT CONCAT_WS('-', name, name) FROM users",
        "SELECT LEFT(name, 2) FROM users",
        "SELECT RIGHT(name, age) FROM users",
        "SELECT SPLIT_PART(name, ',', 2) FROM users",
        "SELECT REVERSE(name) FROM users",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for `{sql}`");
    }
}

#[test]
fn b448_functions_reject_wrong_types() {
    // CONCAT coerces textout-able scalars: `age` INT is now accepted...
    assert!(plan("SELECT CONCAT(name, age) FROM users", &catalog()).is_ok());
    // ...but a type with its own concatenation semantics (BYTEA) still is not.
    assert!(matches!(
        plan(
            "SELECT CONCAT(name, CAST(name AS BYTEA)) FROM users",
            &catalog()
        ),
        Err(Error::TypeMismatch { .. }),
    ));
    // LEFT length must be INT, not TEXT.
    assert!(matches!(
        plan("SELECT LEFT(name, name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // SPLIT_PART field index must be INT.
    assert!(matches!(
        plan("SELECT SPLIT_PART(name, ',', name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

#[test]
fn b448_functions_reject_wrong_arity() {
    // CONCAT / CONCAT_WS need at least one argument.
    assert!(matches!(
        plan("SELECT CONCAT() FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    // REVERSE takes exactly one; LEFT exactly two.
    assert!(matches!(
        plan("SELECT REVERSE(name, name) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        plan("SELECT LEFT(name) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

// --- regex functions --------------------------------------------

#[test]
fn b449_regex_functions_accept_and_typecheck() {
    // Well-typed calls resolve; REGEXP_MATCH yields TEXT[] (usable in a projection).
    for sql in [
        "SELECT REGEXP_REPLACE(name, 'a', 'b') FROM users",
        "SELECT REGEXP_REPLACE(name, 'a', 'b', 'gi') FROM users",
        "SELECT REGEXP_MATCH(name, '[0-9]+') FROM users",
        "SELECT REGEXP_MATCH(name, '([a-z])', 'i') FROM users",
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for `{sql}`");
    }
    // Wrong argument type (INT where TEXT expected).
    assert!(matches!(
        plan(
            "SELECT REGEXP_REPLACE(age, 'a', 'b') FROM users",
            &catalog()
        ),
        Err(Error::TypeMismatch { .. }),
    ));
    // Wrong arity.
    assert!(matches!(
        plan("SELECT REGEXP_MATCH(name) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

// --- math functions ------------------------------------------

#[test]
fn b453_55_math_accept_and_result_types() {
    // INT/FLOAT args accepted; result types propagate (ABS preserves, SQRT is FLOAT).
    for sql in [
        "SELECT ABS(age) FROM users",   // age INT -> INT
        "SELECT ABS(score) FROM users", // score FLOAT -> FLOAT
        "SELECT ROUND(score, 2) FROM users",
        "SELECT CEIL(score) FROM users",
        "SELECT MOD(age, 3) FROM users",
        "SELECT POWER(age, 2) FROM users",
        "SELECT SQRT(age) FROM users",
        "SELECT LOG(age) FROM users",
        "SELECT LOG(2, age) FROM users",
        "SELECT ABS(age) + 1 FROM users", // ABS(INT) usable as INT
        "SELECT score FROM users WHERE SQRT(age) > 2", // SQRT -> FLOAT, comparable
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for `{sql}`");
    }
}

#[test]
fn b453_55_math_reject_wrong_type_and_arity() {
    // Non-numeric argument.
    assert!(matches!(
        plan("SELECT ABS(name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // POWER needs two arguments; SQRT exactly one.
    assert!(matches!(
        plan("SELECT POWER(age) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        plan("SELECT SQRT(age, age) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    // ROUND decimal places must be an integer, not text.
    assert!(matches!(
        plan("SELECT ROUND(score, name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- conditional functions --------------------------------------

#[test]
fn b457_conditional_accept_and_reject() {
    // Same-type / unifiable arguments accept; result type usable.
    for sql in [
        "SELECT NULLIF(age, 0) FROM users",
        "SELECT GREATEST(age, 10) FROM users",
        "SELECT LEAST(age, 100) FROM users",
        "SELECT GREATEST(name, 'mid') FROM users", // text comparable
        "SELECT NULLIF(age, 0) + 1 FROM users",    // NULLIF -> INT usable
    ] {
        assert!(plan(sql, &catalog()).is_ok(), "expected Ok for `{sql}`");
    }
    // Incompatible argument types are rejected.
    assert!(matches!(
        plan("SELECT GREATEST(age, name) FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // NULLIF needs exactly two arguments.
    assert!(matches!(
        plan("SELECT NULLIF(age) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

// --- RANDOM / SETSEED -------------------------------------------

#[test]
fn b456_random_setseed_types_and_arity() {
    // RANDOM() -> FLOAT (usable in arithmetic); SETSEED(x) -> BOOL.
    assert!(plan("SELECT RANDOM() FROM users", &catalog()).is_ok());
    assert!(plan("SELECT RANDOM() * 100 FROM users", &catalog()).is_ok());
    assert!(plan("SELECT SETSEED(0.5) FROM users", &catalog()).is_ok());
    // RANDOM takes no argument; SETSEED takes exactly one.
    assert!(matches!(
        plan("SELECT RANDOM(1) FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
    assert!(matches!(
        plan("SELECT SETSEED() FROM users", &catalog()),
        Err(Error::Unsupported(_)),
    ));
}

// --- JSON path operators #> / #>> ------------------------------

#[test]
fn b458a_json_path_operators() {
    // JSON #> text[] -> JSON; #>> text[] -> TEXT (usable downstream).
    assert!(
        plan(
            "SELECT '{\"a\":1}'::json #> '{a}'::text[] FROM users",
            &catalog()
        )
        .is_ok()
    );
    assert!(
        plan(
            "SELECT '{\"a\":1}'::json #>> '{a}'::text[] FROM users",
            &catalog()
        )
        .is_ok()
    );
    // A bare text-literal path is accepted and coerced to text[] at eval time (SQL-standard).
    assert!(plan("SELECT '{\"a\":1}'::json #> '{a}' FROM users", &catalog()).is_ok());
    // A non-text/array path (e.g. an integer) is still rejected.
    assert!(matches!(
        plan("SELECT '{\"a\":1}'::json #> 5 FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
    // Left must be JSON.
    assert!(matches!(
        plan("SELECT name #> '{a}'::text[] FROM users", &catalog()),
        Err(Error::TypeMismatch { .. }),
    ));
}

// --- System-catalog namespace guard ------------------------------

/// `users` plus mock `nusadb_policies` / `nusadb_rls` system-catalog tables — the shape of the
/// the design adversarial probe (a non-superuser rewriting the RLS catalogs with plain SQL).
fn system_catalogs() -> MockCatalog {
    catalog()
        .with(TableSchema {
            schema: "public".to_owned(),
            id: TableId(8),
            name: "nusadb_policies".to_owned(),
            columns: vec![
                col("polname", ColumnType::Text, false),
                col("pred", ColumnType::Text, true),
            ],
        })
        .with(TableSchema {
            schema: "public".to_owned(),
            id: TableId(9),
            name: "nusadb_rls".to_owned(),
            columns: vec![col("tbl", ColumnType::Text, false)],
        })
}

#[test]
fn non_superuser_cannot_touch_system_catalogs() {
    // Every user-statement path that resolves a `nusadb_*` table by name must refuse a
    // non-superuser — DML would forge/disable RLS, SELECT leaks policy definitions, and DDL
    // (drop/alter/squat, including via the view path) destroys or impersonates the catalog.
    let cat = system_catalogs().non_superuser();
    for sql in [
        "INSERT INTO nusadb_policies VALUES ('evil', 'TRUE')", // forge a policy
        "DELETE FROM nusadb_rls",                              // switch RLS off
        "UPDATE nusadb_policies SET pred = 'TRUE'",            // widen a policy
        "SELECT * FROM nusadb_policies",                       // leak policy definitions
        "SELECT u.id FROM users u JOIN nusadb_rls r ON u.name = r.tbl", // join path
        "TRUNCATE nusadb_rls",
        "DROP TABLE nusadb_policies",
        "ALTER TABLE nusadb_policies ADD COLUMN evil INT",
        "CREATE TABLE nusadb_evil (a INT)", // squat the namespace
        "CREATE INDEX idx ON nusadb_policies (polname)",
        "CREATE MATERIALIZED VIEW nusadb_evil AS SELECT id FROM users",
        "DROP VIEW nusadb_policies",
        "COMMENT ON TABLE nusadb_policies IS 'x'",
    ] {
        assert!(
            matches!(plan(sql, &cat), Err(Error::PermissionDenied(_))),
            "expected PermissionDenied for non-superuser `{sql}`"
        );
    }
}

#[test]
fn superuser_keeps_system_catalog_access() {
    // The namespace is reserved *to superusers*: introspection and administration still work.
    let cat = system_catalogs();
    assert!(plan("SELECT * FROM nusadb_policies", &cat).is_ok());
    assert!(plan("INSERT INTO nusadb_policies VALUES ('p', 'TRUE')", &cat).is_ok());
    assert!(plan("DELETE FROM nusadb_rls", &cat).is_ok());
}

#[test]
fn non_superuser_keeps_normal_table_access() {
    // The guard is scoped to the reserved prefix — ordinary tables are untouched.
    let cat = system_catalogs().non_superuser();
    assert!(plan("SELECT id FROM users", &cat).is_ok());
    assert!(plan("INSERT INTO users VALUES (1, 'a', 2, 3.0, TRUE)", &cat).is_ok());
}

#[test]
fn cte_may_shadow_a_system_catalog_name() {
    // A CTE named `nusadb_*` is the user's own derived data (it shadows, never reaches, the
    // catalog table), so the guard must not fire on the CTE path.
    let cat = system_catalogs().non_superuser();
    assert!(
        plan(
            "WITH nusadb_policies AS (SELECT id FROM users) SELECT * FROM nusadb_policies",
            &cat
        )
        .is_ok()
    );
}

// --- CREATE VIEW column list ----------------------------------

#[test]
fn create_plain_view_carries_an_explicit_column_list() {
    let cat = catalog();
    let LogicalPlan::CreateView(p) = plan(
        "CREATE VIEW v (worker, age2) AS SELECT id, age FROM users",
        &cat,
    )
    .unwrap() else {
        panic!("expected a CreateView plan");
    };
    assert_eq!(p.name, "v");
    assert_eq!(p.columns, vec!["worker".to_owned(), "age2".to_owned()]);
}

#[test]
fn create_plain_view_without_a_column_list_records_none() {
    let cat = catalog();
    let LogicalPlan::CreateView(p) =
        plan("CREATE VIEW v AS SELECT id, age FROM users", &cat).unwrap()
    else {
        panic!("expected a CreateView plan");
    };
    assert!(p.columns.is_empty());
}

#[test]
fn create_plain_view_column_list_arity_must_match_the_body() {
    let cat = catalog();
    // The body projects two columns but the list names only one.
    assert!(matches!(
        plan(
            "CREATE VIEW v (only_one) AS SELECT id, age FROM users",
            &cat
        ),
        Err(Error::ArityMismatch { .. })
    ));
}
