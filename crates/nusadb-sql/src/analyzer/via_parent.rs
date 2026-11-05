//! Access checks for the tables a query reads through the table it names.
//!
//! A query on an inheritance or partitioned parent (without `ONLY`) also reads the rows of the
//! tables below it. Those tables are reached through the parent: the query's privileges were
//! checked on the parent, the table it names, and row-level security applies the parent's policies
//! to every row, wherever the row is stored. The read is expanded into one generated
//! `SELECT <columns> FROM ONLY <table>` branch per table, and [`ViaParent`] is the catalog those
//! branches are analyzed under: it answers the privilege questions about the branch tables as
//! granted (the parent's check already decided them) and reports no row-level security for them,
//! because the caller adds the parent's policies to every branch itself, built under the plain
//! catalog. Everything else is the wrapped catalog's answer.
//!
//! The answers are keyed by table name, so the wrapper is only ever handed a generated branch: one
//! that names its own table and nothing else (no subquery, function call or view), and a policy
//! predicate is never analyzed under it. Any other reference to a table, including the same
//! partition named elsewhere in the statement or in a policy, is analyzed under the plain catalog
//! and checks that table's own privileges and policies. An `UPDATE` or `DELETE` on a parent does not
//! use this wrapper at all: each of its per-table sub-plans takes the parent's privileges and
//! policies for its target alone (see `analyze_update_stmt`).

use std::collections::HashSet;

use nusadb_core::TableSchema;

use super::{Catalog, FunctionDef, IndexInfo, PolicyDef, PruneConstraint};
use crate::ast::{ObjectKind, Privilege, TriggerEvent};
use crate::error::Error;

/// The catalog the branches of a parent's expansion are analyzed under (see the module docs).
pub(super) struct ViaParent<'a> {
    inner: &'a dyn Catalog,
    /// The branch tables, as `(schema, name)`: the parent and the reached tables. Their row-level
    /// security is the parent's, which the caller applies to every branch.
    branches: HashSet<(String, String)>,
    /// The privilege-catalog keys (`schema.name`) whose privileges count as checked.
    covered: HashSet<String>,
}

impl<'a> ViaParent<'a> {
    /// The catalog for the branches over `parent` and the tables `reached` through it
    /// (schema-qualified edge keys, bare in `public`). With `cover_parent`, the parent's own
    /// privileges count as checked too: the caller generated a read of the parent's every column for
    /// an expansion whose real column reads the enclosing query checks against the parent.
    pub(super) fn new(
        inner: &'a dyn Catalog,
        parent: &TableSchema,
        reached: &[String],
        cover_parent: bool,
    ) -> Self {
        let mut branches: HashSet<(String, String)> = reached
            .iter()
            .map(|key| {
                let (schema, name) = super::split_qualified(key);
                (
                    schema.unwrap_or(nusadb_core::PUBLIC_SCHEMA).to_owned(),
                    name.to_owned(),
                )
            })
            .collect();
        let mut covered: HashSet<String> = branches
            .iter()
            .map(|(schema, name)| format!("{schema}.{name}"))
            .collect();
        if cover_parent {
            covered.insert(format!("{}.{}", parent.schema, parent.name));
        }
        branches.insert((parent.schema.clone(), parent.name.clone()));
        Self {
            inner,
            branches,
            covered,
        }
    }
}

impl Catalog for ViaParent<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
        self.inner.lookup_table(name)
    }
    fn lookup_table_in(&self, schema: &str, name: &str) -> Result<Option<TableSchema>, Error> {
        self.inner.lookup_table_in(schema, name)
    }
    fn search_path(&self) -> Vec<String> {
        self.inner.search_path()
    }
    fn current_schema(&self) -> String {
        self.inner.current_schema()
    }
    fn temp_schema(&self) -> Option<String> {
        self.inner.temp_schema()
    }
    fn session_timezone(&self) -> Option<String> {
        self.inner.session_timezone()
    }
    fn list_indexes(&self, table: &str) -> Result<Vec<IndexInfo>, Error> {
        self.inner.list_indexes(table)
    }
    fn table_stats(&self, table: &str) -> Result<Option<nusadb_core::TableStats>, Error> {
        self.inner.table_stats(table)
    }
    fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
        self.inner.approx_row_count(table)
    }
    fn list_indexes_in(&self, schema: &str, name: &str) -> Result<Vec<IndexInfo>, Error> {
        self.inner.list_indexes_in(schema, name)
    }
    fn table_stats_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, Error> {
        self.inner.table_stats_in(schema, name)
    }
    fn approx_row_count_in(&self, schema: &str, name: &str) -> Result<u64, Error> {
        self.inner.approx_row_count_in(schema, name)
    }
    fn has_instead_of_trigger(&self, name: &str, event: TriggerEvent) -> Result<bool, Error> {
        self.inner.has_instead_of_trigger(name, event)
    }
    fn lookup_view(&self, name: &str) -> Result<Option<String>, Error> {
        self.inner.lookup_view(name)
    }
    fn lookup_view_columns(&self, name: &str) -> Result<Vec<String>, Error> {
        self.inner.lookup_view_columns(name)
    }
    fn lookup_view_check_option(&self, name: &str) -> Result<bool, Error> {
        self.inner.lookup_view_check_option(name)
    }
    fn lookup_function(&self, name: &str) -> Result<Option<FunctionDef>, Error> {
        self.inner.lookup_function(name)
    }
    fn any_inheritance(&self) -> Result<bool, Error> {
        self.inner.any_inheritance()
    }
    fn inheritance_descendants(&self, table: &str) -> Result<Vec<String>, Error> {
        self.inner.inheritance_descendants(table)
    }
    fn partition_key_columns(&self, table: &str) -> Result<Option<Vec<String>>, Error> {
        self.inner.partition_key_columns(table)
    }
    fn inheritance_fingerprint(&self) -> Result<u64, Error> {
        self.inner.inheritance_fingerprint()
    }
    fn partitions_to_prune(
        &self,
        parent: &str,
        constraints: &[PruneConstraint],
    ) -> Result<Vec<String>, Error> {
        self.inner.partitions_to_prune(parent, constraints)
    }
    fn lookup_composite(
        &self,
        name: &str,
    ) -> Result<Option<Vec<(String, crate::composite::FieldType)>>, Error> {
        self.inner.lookup_composite(name)
    }
    fn lookup_composite_column(
        &self,
        schema: &str,
        table: &str,
        column: &str,
    ) -> Result<Option<String>, Error> {
        self.inner.lookup_composite_column(schema, table, column)
    }
    fn lookup_enum_column(
        &self,
        schema: &str,
        table: &str,
        column: &str,
    ) -> Result<Option<String>, Error> {
        self.inner.lookup_enum_column(schema, table, column)
    }
    fn enum_labels(&self, name: &str) -> Result<Option<Vec<String>>, Error> {
        self.inner.enum_labels(name)
    }
    fn is_superuser(&self) -> bool {
        self.inner.is_superuser()
    }
    fn rls_enabled(&self, schema: &str, name: &str) -> Result<bool, Error> {
        // The parent's policies are added to every branch by the caller, under the plain catalog.
        if self
            .branches
            .contains(&(schema.to_owned(), name.to_owned()))
        {
            return Ok(false);
        }
        self.inner.rls_enabled(schema, name)
    }
    fn current_user(&self) -> String {
        self.inner.current_user()
    }
    fn prepared_transactions(&self) -> Result<Vec<nusadb_core::PreparedTransaction>, Error> {
        self.inner.prepared_transactions()
    }
    fn lookup_policies(&self, schema: &str, name: &str) -> Result<Vec<PolicyDef>, Error> {
        self.inner.lookup_policies(schema, name)
    }
    fn has_privilege(
        &self,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if kind == ObjectKind::Table && self.covered.contains(object) {
            return Ok(true);
        }
        self.inner.has_privilege(kind, object, privilege)
    }
    fn has_column_privilege(
        &self,
        object: &str,
        column: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if self.covered.contains(object) {
            return Ok(true);
        }
        self.inner.has_column_privilege(object, column, privilege)
    }
    fn has_any_column_privilege(&self, object: &str, privilege: Privilege) -> Result<bool, Error> {
        if self.covered.contains(object) {
            return Ok(true);
        }
        self.inner.has_any_column_privilege(object, privilege)
    }
    fn may_grant_object(
        &self,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        self.inner.may_grant_object(kind, object, privilege)
    }
    fn may_create_role(&self) -> Result<bool, Error> {
        self.inner.may_create_role()
    }
    fn may_create_database(&self) -> Result<bool, Error> {
        self.inner.may_create_database()
    }
    fn may_administer_role(&self, role: &str) -> Result<bool, Error> {
        self.inner.may_administer_role(role)
    }
    fn may_assume_role(&self, role: &str) -> Result<bool, Error> {
        self.inner.may_assume_role(role)
    }
    fn role_exists(&self, name: &str) -> Result<bool, Error> {
        self.inner.role_exists(name)
    }
    fn owns_object(&self, kind: ObjectKind, object: &str) -> Result<bool, Error> {
        self.inner.owns_object(kind, object)
    }
}
