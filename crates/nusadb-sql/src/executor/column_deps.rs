//! What a column change does to the objects that name the column.
//!
//! A column is named in more places than its table's schema: the table's own indexes and
//! constraints, foreign keys on either side, views, policies and triggers that keep SQL text, the
//! column grants, the enum and composite type tags and the vector indexes. `DROP COLUMN` takes
//! the table's own objects with the column and, unless `CASCADE` is given, refuses while an object
//! outside the table still needs it. `RENAME COLUMN` takes the table's own objects off the old name
//! before the rename and puts them back under the new one after it, in the same transaction, so a
//! rollback undoes all of it.
#![allow(clippy::wildcard_imports)]

use super::*;
use nusadb_core::engine::{Constraint, ConstraintKind, ForeignKeyDef, IndexDef};

/// An object outside the table whose stored SQL names a column of it.
struct SqlDependent {
    /// What it is, for the error message: `view`, `materialized view`, `policy`, `trigger`,
    /// `function` or `procedure`.
    kind: &'static str,
    name: String,
    /// For a policy or trigger, the `(schema, table)` it is attached to, which need not be the
    /// table whose column is dropped.
    on: Option<(String, String)>,
}

impl SqlDependent {
    /// A function or procedure body is not dropped by `CASCADE`: it is the user's code, and
    /// replacing it is theirs to decide. It refuses the drop either way.
    fn cascades(&self) -> bool {
        !matches!(self.kind, "function" | "procedure")
    }
}

/// The views, materialized views, policies, triggers, functions and procedures whose stored SQL
/// needs `column` of `table`.
///
/// A definition needs the column when it names it and reads the table, directly or through views
/// that do, or when it reads the table itself through a `*` / `t.*` or a `NATURAL` join, whose
/// columns are every column the table has. A view built on a view that needs the column needs it
/// too. A policy or trigger attached to the table needs it when its text names the column at all;
/// one attached to another table counts by the same rules as a view.
fn sql_dependents(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<SqlDependent>, Error> {
    let mut views = Vec::new();
    for (catalog, kind) in [
        (VIEW_CATALOG, "view"),
        (MATVIEW_CATALOG, "materialized view"),
    ] {
        for row in super::ddl::scan_text_catalog(engine, txn, catalog, &[2])? {
            let mut row = row.into_iter();
            if let (Some(name), Some(def)) = (row.next(), row.next()) {
                views.push((kind, name, def));
            }
        }
    }
    // The names that reach the table: the table itself and every view reading it, transitively.
    let qualified = format!("{}.{}", table.schema, table.name);
    let direct =
        |sql: &str| sql_mentions_column(sql, &table.name) || sql_mentions_column(sql, &qualified);
    let mut reaches: Vec<&str> = vec![&table.name, &qualified];
    loop {
        let mut grew = false;
        for (_, name, def) in &views {
            if !reaches.iter().any(|n| n == name)
                && reaches.iter().any(|n| sql_mentions_column(def, n))
            {
                reaches.push(name);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    let needs = |sql: &str| {
        (crate::parser::sql_references_column(sql, column)
            && reaches.iter().any(|n| sql_mentions_column(sql, n)))
            || (direct(sql) && crate::parser::sql_reads_all_columns(sql))
    };
    let mut out: Vec<SqlDependent> = Vec::new();
    for (kind, name, def) in &views {
        if needs(def) {
            out.push(SqlDependent {
                kind,
                name: name.clone(),
                on: None,
            });
        }
    }
    // A view over a dependent view stops resolving with it.
    loop {
        let mut grew = false;
        for (kind, name, def) in &views {
            if !out.iter().any(|d| &d.name == name)
                && out.iter().any(|d| sql_mentions_column(def, &d.name))
            {
                out.push(SqlDependent {
                    kind,
                    name: name.clone(),
                    on: None,
                });
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    routine_and_attached_dependents(table, column, &needs, engine, txn, &mut out)?;
    Ok(out)
}

/// The policies, triggers, functions and procedures of [`sql_dependents`], appended to `out`.
/// `needs` judges a definition the way a view is judged.
fn routine_and_attached_dependents(
    table: &TableSchema,
    column: &str,
    needs: &dyn Fn(&str) -> bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
    out: &mut Vec<SqlDependent>,
) -> Result<(), Error> {
    let qualified = format!("{}.{}", table.schema, table.name);
    let owns = |t: &str, s: &str| (t == table.name || t == qualified) && s == table.schema;
    let attached = |t: &str, s: Option<&String>| {
        let schema = s.map_or(nusadb_core::PUBLIC_SCHEMA, String::as_str);
        (schema.to_owned(), t.to_owned())
    };
    let needed_by = |on: &(String, String), texts: &[&String]| {
        texts.iter().any(|sql| {
            needs(sql) || (owns(&on.1, &on.0) && crate::parser::sql_references_column(sql, column))
        })
    };
    // (table, name, command, roles, using, check, permissive, schema)
    for row in super::ddl::scan_text_catalog(engine, txn, POLICY_CATALOG, &[8, 7])? {
        if let [tbl, name, _, _, using, check, _, rest @ ..] = row.as_slice() {
            let on = attached(tbl, rest.first());
            if needed_by(&on, &[using, check]) {
                out.push(SqlDependent {
                    kind: "policy",
                    name: name.clone(),
                    on: Some(on),
                });
            }
        }
    }
    // (name, table, timing, events, for_each, when, action, enabled, schema)
    for row in
        super::ddl::scan_text_catalog(engine, txn, super::trigger::TRIGGER_CATALOG, &[9, 8, 7])?
    {
        if let [name, tbl, _, _, _, when, action, rest @ ..] = row.as_slice() {
            let on = attached(tbl, rest.get(1));
            if needed_by(&on, &[when, action]) {
                out.push(SqlDependent {
                    kind: "trigger",
                    name: name.clone(),
                    on: Some(on),
                });
            }
        }
    }
    // A function or procedure body is re-analyzed when called.
    for (kind, name, body) in super::ddl::routine_bodies(engine, txn)? {
        if needs(&body) {
            out.push(SqlDependent {
                kind,
                name,
                on: None,
            });
        }
    }
    Ok(())
}

/// The refusal for a `DROP COLUMN` without `CASCADE` that `kind` `name` still depends on.
fn dependent_error(table: &TableSchema, column: &str, kind: &str, name: &str) -> Error {
    Error::DependentObjects(format!(
        "cannot drop column \"{column}\" of table \"{}\" because {kind} \"{name}\" depends on it; \
         use DROP COLUMN ... CASCADE to drop it as well",
        table.name
    ))
}

/// Whether the index `def` cannot outlive `column`: it keys on, carries, or filters by it.
fn index_names_column(def: &IndexDef, column: &str) -> bool {
    def.columns.iter().chain(&def.include).any(|c| c == column)
        || def
            .key_exprs
            .iter()
            .any(|e| crate::parser::sql_references_column(e, column))
        || def
            .predicate
            .as_deref()
            .is_some_and(|p| crate::parser::sql_references_column(p, column))
}

/// Whether the `CHECK` constraint `c` refers to `column` in its predicate.
fn check_names_column(c: &Constraint, column: &str) -> bool {
    c.expr
        .as_deref()
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .is_some_and(|sql| crate::parser::sql_references_column(sql, column))
}

/// The indexes of the table that back one of its constraints; they go with the constraint.
fn backing_indexes(constraints: &[Constraint]) -> std::collections::HashSet<nusadb_core::IndexId> {
    constraints.iter().filter_map(|c| c.index).collect()
}

/// The names of `table`'s generated columns whose expression reads `column`. They cannot be
/// computed without it, so dropping the column drops them too.
pub(super) fn generated_dependents(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<String>, Error> {
    let key = super::coldefault::catalog_key(&table.schema, &table.name);
    Ok(super::coldefault::load_defaults(&key, engine, txn)?
        .into_iter()
        .filter(|(owner, sql)| {
            owner != column
                && super::coldefault::generated_expr(sql)
                    .is_some_and(|expr| crate::parser::sql_references_column(expr, column))
        })
        .map(|(owner, _)| owner)
        .collect())
}

/// Before the column at `index` is dropped from `table`: refuse (without `cascade`) or drop (with
/// it) what depends on the column from outside the table, then drop the table's own objects that
/// involve the column, whether they involve it alone or with other columns: its indexes, its
/// `PRIMARY KEY`, `UNIQUE` and `CHECK` constraints, its foreign keys, its vector indexes, its
/// column grants and its type tags. The type-range check and the default are the caller's.
pub(super) fn drop_column_dependents(
    table: &TableSchema,
    index: usize,
    cascade: bool,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let column = column_name(table, index)?;
    let fks = engine.list_foreign_keys(table.id)?;
    // A key on this table whose own columns include this one goes with it below. Any other key
    // referencing the column, from another table or this one, is a dependent from outside.
    let own_fk =
        |fk: &ForeignKeyDef| fk.child_table == table.id && fk.child_columns.contains(&column);
    let referencing: Vec<&ForeignKeyDef> = fks
        .iter()
        .filter(|fk| {
            fk.parent_table == table.id && fk.parent_columns.contains(&column) && !own_fk(fk)
        })
        .collect();
    let text = sql_dependents(table, &column, engine, txn)?;
    if let Some(dep) = text.iter().find(|d| !d.cascades()) {
        return Err(Error::DependentObjects(format!(
            "cannot drop column \"{column}\" of table \"{}\" because the body of {} \"{}\" \
             reads it; replace or drop the {} first",
            table.name, dep.kind, dep.name, dep.kind
        )));
    }
    if !cascade {
        if let Some(fk) = referencing.first() {
            return Err(dependent_error(table, &column, "foreign key", &fk.name));
        }
        if let Some(dep) = text.first() {
            return Err(dependent_error(table, &column, dep.kind, &dep.name));
        }
    }
    for fk in &referencing {
        engine.drop_constraint(txn, fk.child_table, &fk.name)?;
    }
    for dep in &text {
        match (dep.kind, &dep.on) {
            ("policy", Some((schema, on))) => {
                super::delete_policy_row(engine, txn, schema, on, &dep.name)?;
            },
            ("trigger", Some((schema, on))) => {
                super::trigger::delete_trigger_row(engine, txn, schema, on, &dep.name)?;
            },
            _ => {
                super::run_drop_view(
                    &crate::planner::DropViewPlan {
                        name: dep.name.clone(),
                        if_exists: true,
                    },
                    engine,
                    txn,
                )?;
            },
        }
    }
    for fk in fks.iter().filter(|fk| own_fk(fk)) {
        engine.drop_constraint(txn, table.id, &fk.name)?;
    }
    // Secondary indexes before constraints: the set of backing indexes is read while every
    // remaining constraint still has its own.
    let constraints = engine.list_constraints(table.id)?;
    let backing = backing_indexes(&constraints);
    for def in engine.list_indexes(table.id)? {
        if let Some(id) = engine.lookup_index(&def.name)?
            && !backing.contains(&id)
            && index_names_column(&def, &column)
        {
            engine.drop_index(txn, id)?;
        }
    }
    for c in &constraints {
        let goes = match c.kind {
            ConstraintKind::PrimaryKey | ConstraintKind::Unique => c.columns.contains(&column),
            ConstraintKind::Check => {
                !c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX)
                    && check_names_column(c, &column)
            },
            ConstraintKind::ForeignKey => false,
        };
        if goes {
            engine.drop_constraint(txn, table.id, &c.name)?;
        }
    }
    super::shift_vector_indexes_for_drop(engine, txn, &table.name, index)?;
    crate::rbac::move_column_grants(
        engine,
        txn,
        &format!("{}.{}", table.schema, table.name),
        &column,
        None,
    )?;
    super::move_column_type_tags(engine, txn, &table.schema, &table.name, &column, None)?;
    Ok(())
}

/// The objects of a table taken off a column's old name for a rename, to be put back under the
/// new name once the column is renamed.
#[derive(Default)]
pub(super) struct Detached {
    /// `PRIMARY KEY` and `UNIQUE` constraints over the column.
    uniques: Vec<Constraint>,
    /// `CHECK` constraints naming the column, as `(name, predicate SQL)`.
    checks: Vec<(String, String)>,
    /// Secondary indexes keying on, carrying or filtering by the column.
    indexes: Vec<IndexDef>,
    /// Foreign keys with the column on either side.
    fks: Vec<ForeignKeyDef>,
}

/// Take the objects of `table` that name `column` off it ahead of a rename: foreign keys on either
/// side, `CHECK` constraints, secondary indexes, and `PRIMARY KEY` / `UNIQUE` constraints, in an
/// order that never drops a key another object still references. [`reattach_after_rename`] puts
/// them back.
pub(super) fn detach_for_rename(
    table: &TableSchema,
    column: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Detached, Error> {
    let mut detached = Detached::default();
    for fk in engine.list_foreign_keys(table.id)? {
        let child_side = fk.child_table == table.id && fk.child_columns.iter().any(|c| c == column);
        let parent_side =
            fk.parent_table == table.id && fk.parent_columns.iter().any(|c| c == column);
        if child_side || parent_side {
            engine.drop_constraint(txn, fk.child_table, &fk.name)?;
            detached.fks.push(fk);
        }
    }
    let constraints = engine.list_constraints(table.id)?;
    for c in &constraints {
        if matches!(c.kind, ConstraintKind::Check)
            && !c.name.starts_with(crate::SYNTHETIC_TYPE_CHECK_PREFIX)
            && check_names_column(c, column)
            && let Some(sql) = c.expr.as_deref().and_then(|b| std::str::from_utf8(b).ok())
        {
            detached.checks.push((c.name.clone(), sql.to_owned()));
            engine.drop_constraint(txn, table.id, &c.name)?;
        }
    }
    let backing = backing_indexes(&constraints);
    for def in engine.list_indexes(table.id)? {
        if let Some(id) = engine.lookup_index(&def.name)?
            && !backing.contains(&id)
            && index_names_column(&def, column)
        {
            engine.drop_index(txn, id)?;
            detached.indexes.push(def);
        }
    }
    for c in constraints {
        if matches!(c.kind, ConstraintKind::PrimaryKey | ConstraintKind::Unique)
            && c.columns.iter().any(|k| k == column)
        {
            engine.drop_constraint(txn, table.id, &c.name)?;
            detached.uniques.push(c);
        }
    }
    Ok(detached)
}

/// Fill the index `name` of `table` (just created over existing rows) with every live row.
fn backfill_named_index(
    table: &TableSchema,
    name: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let Some(id) = engine.lookup_index(name)? else {
        return Ok(());
    };
    let Some(def) = engine
        .list_indexes(table.id)?
        .into_iter()
        .find(|d| d.name == name)
    else {
        return Ok(());
    };
    if let Some(target) = dml::build_index_target(id, table, &def) {
        dml::backfill_index(&target, table, engine, txn)?;
    }
    Ok(())
}

/// Put back what [`detach_for_rename`] took off, now naming the column `to` instead of `from`:
/// `PRIMARY KEY` / `UNIQUE` first (a foreign key needs its referenced key), then the indexes, the
/// `CHECK` constraints and the foreign keys. Every recreated index is filled from the table's rows,
/// which the rename left as they were.
pub(super) fn reattach_after_rename(
    detached: Detached,
    table: &TableSchema,
    from: &str,
    to: &str,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<(), Error> {
    let rename = |cols: &[String]| -> Vec<String> {
        cols.iter()
            .map(|c| if c == from { to.to_owned() } else { c.clone() })
            .collect()
    };
    let renamed = super::ddl::reread_table(table, engine, txn)?;
    for c in detached.uniques {
        engine.add_unique_constraint(
            txn,
            table.id,
            &c.name,
            &rename(&c.columns),
            matches!(c.kind, ConstraintKind::PrimaryKey),
            c.nulls_not_distinct,
        )?;
        backfill_named_index(&renamed, &c.name, engine, txn)?;
    }
    for mut def in detached.indexes {
        def.columns = rename(&def.columns);
        def.include = rename(&def.include);
        def.key_exprs = def
            .key_exprs
            .iter()
            .map(|e| crate::parser::rename_column_in_sql(e, from, to))
            .collect::<Result<_, _>>()?;
        def.predicate = def
            .predicate
            .as_deref()
            .map(|p| crate::parser::rename_column_in_sql(p, from, to))
            .transpose()?;
        engine.create_index(txn, &def)?;
        backfill_named_index(&renamed, &def.name, engine, txn)?;
    }
    for (name, sql) in detached.checks {
        let sql = crate::parser::rename_column_in_sql(&sql, from, to)?;
        engine.add_check_constraint(txn, table.id, &name, sql.as_bytes())?;
    }
    for mut fk in detached.fks {
        if fk.child_table == table.id {
            fk.child_columns = rename(&fk.child_columns);
        }
        if fk.parent_table == table.id {
            fk.parent_columns = rename(&fk.parent_columns);
        }
        engine.add_foreign_key(txn, &fk)?;
        let child = if fk.child_table == table.id {
            Some(renamed.clone())
        } else {
            dml::schema_by_id(engine, fk.child_table)?
        };
        if let Some(child) = child {
            backfill_named_index(&child, &fk.name, engine, txn)?;
        }
    }
    // The rest of what files the column under its name: generated expressions reading it, its
    // column grants and its type tags.
    let key = super::coldefault::catalog_key(&table.schema, &table.name);
    for (owner, sql) in super::coldefault::load_defaults(&key, engine, txn)? {
        if let Some(expr) = super::coldefault::generated_expr(&sql)
            && crate::parser::sql_references_column(expr, from)
        {
            let expr = crate::parser::rename_column_in_sql(expr, from, to)?;
            super::coldefault::set_default(
                &key,
                &owner,
                &super::coldefault::generated_default_sql(&expr),
                engine,
                txn,
            )?;
        }
    }
    crate::rbac::move_column_grants(
        engine,
        txn,
        &format!("{}.{}", table.schema, table.name),
        from,
        Some(to),
    )?;
    super::move_column_type_tags(engine, txn, &table.schema, &table.name, from, Some(to))?;
    Ok(())
}
