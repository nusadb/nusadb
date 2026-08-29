//! Analysis for the data-control statements, and the single funnel every privilege check in the
//! analyzer goes through.
//!
//! Two jobs live here. The first is turning a parsed `GRANT`/`REVOKE`/role statement into a plan:
//! resolving object names against the `search_path`, checking that each privilege is meaningful on
//! the object it names, and checking that the session is allowed to hand out what it is handing
//! out. The second is [`require_privilege`] — the one place a statement's access check is raised —
//! so an audit of "what is enforced" reads one function and its call sites rather than a scattering
//! of ad-hoc conditions.

use crate::ast::{self, Grantee, ObjectKind, Privilege};
use crate::error::Error;
use crate::planner::{
    AlterRolePlan, CreateRolePlan, DropRolePlan, GrantPlan, GrantRolePlan, GrantTargetsPlan,
    RevokePlan, RevokeRolePlan, SetRolePlan,
};

use super::Catalog;

/// Raise the access check for `privilege` on an object, returning `Ok(())` when the session holds
/// it and [`Error::PermissionDenied`] when it does not.
///
/// The denial names the privilege and the object but nothing else — in particular it does not
/// distinguish "you may not" from "there is no such object", which would turn a denied statement
/// into a way to enumerate the schema.
///
/// # Errors
/// [`Error::PermissionDenied`] when the check fails; propagates catalog errors.
pub(super) fn require_privilege(
    catalog: &dyn Catalog,
    kind: ObjectKind,
    object: &str,
    privilege: Privilege,
) -> Result<(), Error> {
    if catalog.has_privilege(kind, object, privilege)? {
        return Ok(());
    }
    Err(crate::rbac::denied(kind, object, privilege))
}

/// Raise the access check for `privilege` on a resolved table.
///
/// The privilege catalog keys tables by their canonical `schema.name`, which is exactly what a
/// resolved [`nusadb_core::TableSchema`] carries — so this is the form every check uses, and a
/// table reached by a bare name through the `search_path` is checked as the same object a
/// qualified reference would be.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session lacks the privilege.
pub(super) fn require_table_privilege(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
    privilege: Privilege,
) -> Result<(), Error> {
    // A superuser short-circuits before any catalog read, so the common administrative path pays
    // nothing for these checks.
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    require_privilege(catalog, ObjectKind::Table, &object, privilege)
}

/// Gate a table into a `SELECT` read scope: the session needs table-wide SELECT, ownership, superuser,
/// or a column-scoped SELECT grant on *some* column. Passing this gate only admits the table; the
/// per-column [`ScopedColumn`] flag then denies a reference to a column the session may not read (so
/// `SELECT count(*)` works with any column grant, but `SELECT secret` still fails). A session with no
/// SELECT of any kind gets the ordinary table-level denial.
pub(super) fn require_read_access(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    if catalog.has_privilege(ObjectKind::Table, &object, Privilege::Select)?
        || catalog.has_any_column_privilege(&object, Privilege::Select)?
    {
        return Ok(());
    }
    // Neither table-wide nor any column-scoped SELECT — the standard table-level denial.
    require_privilege(catalog, ObjectKind::Table, &object, Privilege::Select)
}

/// Require `privilege` on `columns` of `table`: a table-wide grant covers them all, otherwise every
/// named column must carry a column-scoped grant. Fail-closed — an empty column list (a statement
/// that acts on the table but names no column, like `INSERT ... DEFAULT VALUES`) can only be
/// satisfied table-wide, never by a column grant. Used for the column-scoped privileges (`SELECT`,
/// `INSERT`, `UPDATE`, `REFERENCES`).
pub(super) fn require_column_privilege(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
    columns: &[&str],
    privilege: Privilege,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    // A table-wide grant (or ownership, via `has_privilege`) covers every column.
    if catalog.has_privilege(ObjectKind::Table, &object, privilege)? {
        return Ok(());
    }
    if columns.is_empty() {
        return Err(Error::PermissionDenied(format!(
            "{} on table `{object}`",
            privilege.as_str()
        )));
    }
    for column in columns {
        if !catalog.has_column_privilege(&object, column, privilege)? {
            return Err(Error::PermissionDenied(format!(
                "{} on column `{column}` of table `{object}`",
                privilege.as_str()
            )));
        }
    }
    Ok(())
}

/// Raise the ownership check that `DROP` / `ALTER` on a table requires.
///
/// These are not grantable privileges in the SQL standard and are not modelled as ones here:
/// restructuring or destroying an object is the owner's right (or a superuser's). That is why this
/// asks a different question from [`require_table_privilege`] — no `GRANT` can ever answer it,
/// which is what stops a role holding, say, `INSERT` from dropping the table out from under
/// everyone else who was granted access to it.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session does not own the table.
pub(super) fn require_table_ownership(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
    action: &str,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    if catalog.owns_object(ObjectKind::Table, &object)? {
        return Ok(());
    }
    Err(Error::PermissionDenied(format!(
        "must be the owner of table `{object}` to {action} it"
    )))
}

/// Whether the session may see a table's metadata (its column list, via `SHOW COLUMNS`). Any
/// relationship to the table is enough — ownership, a superuser session, or any single privilege
/// on it — but a role with no relationship to the table cannot enumerate its columns, which would
/// otherwise leak the schema of tables it can neither read nor write.
pub(super) fn require_table_metadata_access(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    if catalog.owns_object(ObjectKind::Table, &object)? {
        return Ok(());
    }
    for privilege in Privilege::all_for(ObjectKind::Table) {
        if catalog.has_privilege(ObjectKind::Table, &object, privilege)? {
            return Ok(());
        }
    }
    Err(Error::PermissionDenied(format!(
        "must have some privilege on table `{object}` to see its columns"
    )))
}

/// Whether the session may `LOCK` a table: the table's owner, a superuser, or a role holding a
/// privilege that lets it change the table's rows (`UPDATE`/`DELETE`/`TRUNCATE`) — a lock exists
/// to guard a write, so read-only access does not grant it.
pub(super) fn require_table_lock(
    catalog: &dyn Catalog,
    table: &nusadb_core::TableSchema,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    let object = format!("{}.{}", table.schema, table.name);
    if catalog.owns_object(ObjectKind::Table, &object)? {
        return Ok(());
    }
    for privilege in [Privilege::Update, Privilege::Delete, Privilege::Truncate] {
        if catalog.has_privilege(ObjectKind::Table, &object, privilege)? {
            return Ok(());
        }
    }
    Err(Error::PermissionDenied(format!(
        "must own table `{object}` or hold UPDATE, DELETE or TRUNCATE on it to lock it"
    )))
}

/// Destroying a schema is its owner's right (or a superuser's), never a grantable privilege —
/// the same rule as a table. A schema with no recorded owner reads as bootstrap-superuser-owned,
/// so only a superuser reaches it, which is the safe default for the pre-RBAC `public` schema.
pub(super) fn require_schema_ownership(
    catalog: &dyn Catalog,
    schema: &str,
    action: &str,
) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    if catalog.owns_object(ObjectKind::Schema, schema)? {
        return Ok(());
    }
    Err(Error::PermissionDenied(format!(
        "must be the owner of schema `{schema}` to {action} it"
    )))
}

/// Emptying the whole database (`DROP DATABASE`) is a cluster-level operation with no finer
/// ownership model yet, so it is restricted to a superuser — fail-closed until a database-owner
/// model exists.
pub(super) fn require_superuser(catalog: &dyn Catalog, action: &str) -> Result<(), Error> {
    if catalog.is_superuser() {
        return Ok(());
    }
    Err(Error::PermissionDenied(format!(
        "must be a superuser to {action}"
    )))
}

/// Resolve a table or sequence name written in a `GRANT` into the canonical `schema.name` the
/// privilege catalog keys on.
///
/// An already-qualified name is taken as written. A bare name is resolved through the catalog so
/// it lands on the same object the session's other statements would reach; a name that resolves to
/// nothing keeps its bare spelling and is reported as a missing object by the caller.
fn canonical_object(catalog: &dyn Catalog, name: &str) -> Result<String, Error> {
    if name.contains('.') {
        return Ok(name.to_owned());
    }
    Ok(match catalog.lookup_table(name)? {
        Some(schema) => format!("{}.{}", schema.schema, schema.name),
        None => name.to_owned(),
    })
}

/// Resolve the parsed target list into the plan's form, canonicalizing what can be canonicalized.
fn resolve_targets(
    catalog: &dyn Catalog,
    targets: ast::GrantTargets,
) -> Result<GrantTargetsPlan, Error> {
    let objects = |kind: ObjectKind, names: Vec<String>| -> Vec<ast::GrantObject> {
        names
            .into_iter()
            .map(|name| ast::GrantObject { kind, name })
            .collect()
    };
    Ok(match targets {
        ast::GrantTargets::Tables(names) => {
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                out.push(ast::GrantObject {
                    kind: ObjectKind::Table,
                    name: canonical_object(catalog, &name)?,
                });
            }
            GrantTargetsPlan::Objects(out)
        },
        ast::GrantTargets::Sequences(names) => {
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                out.push(ast::GrantObject {
                    kind: ObjectKind::Sequence,
                    name: canonical_object(catalog, &name)?,
                });
            }
            GrantTargetsPlan::Objects(out)
        },
        // Schema and database names are already their own canonical form — there is no
        // `search_path` step to apply to a namespace name.
        ast::GrantTargets::Schemas(names) => {
            GrantTargetsPlan::Objects(objects(ObjectKind::Schema, names))
        },
        ast::GrantTargets::Databases(names) => {
            GrantTargetsPlan::Objects(objects(ObjectKind::Database, names))
        },
        ast::GrantTargets::AllTablesInSchemas(schemas) => {
            GrantTargetsPlan::AllTablesInSchemas(schemas)
        },
        ast::GrantTargets::AllSequencesInSchemas(schemas) => {
            GrantTargetsPlan::AllSequencesInSchemas(schemas)
        },
    })
}

/// Reject a privilege that is meaningless on the object it was named against — `GRANT CONNECT ON
/// TABLE t`, say. Storing such a grant would be harmless but dishonest: nothing would ever consult
/// it, so the operator would believe they had configured something they had not.
fn check_privileges_apply(
    privileges: Option<&Vec<Privilege>>,
    targets: &GrantTargetsPlan,
) -> Result<(), Error> {
    let Some(privileges) = privileges else {
        // `ALL PRIVILEGES` expands per object kind at execution, so it is meaningful everywhere.
        return Ok(());
    };
    let kinds: Vec<ObjectKind> = match targets {
        GrantTargetsPlan::Objects(objects) => objects.iter().map(|o| o.kind).collect(),
        GrantTargetsPlan::AllTablesInSchemas(_) => vec![ObjectKind::Table],
        GrantTargetsPlan::AllSequencesInSchemas(_) => vec![ObjectKind::Sequence],
    };
    for privilege in privileges {
        if let Some(kind) = kinds.iter().find(|kind| !privilege.applies_to(**kind)) {
            return Err(Error::InvalidStatement(format!(
                "{privilege} is not a privilege on a {noun}",
                privilege = privilege.as_str(),
                noun = kind.noun(),
            )));
        }
    }
    Ok(())
}

/// Check that the session may hand out each privilege it is granting.
///
/// Holding a privilege and being allowed to pass it on are different things — the second needs
/// ownership, superuser, or a grant that carries the grant option. Checking here means a `GRANT`
/// the session could not back is refused rather than half-applied across a multi-object list.
fn check_may_grant(
    catalog: &dyn Catalog,
    privileges: Option<&Vec<Privilege>>,
    targets: &GrantTargetsPlan,
) -> Result<(), Error> {
    let GrantTargetsPlan::Objects(objects) = targets else {
        // The `ALL ... IN SCHEMA` forms name objects that only exist at execution time; the
        // executor re-checks each one it expands to.
        return Ok(());
    };
    for object in objects {
        let wanted = privileges
            .cloned()
            .unwrap_or_else(|| Privilege::all_for(object.kind));
        for privilege in wanted {
            if !privilege.applies_to(object.kind) {
                continue;
            }
            if !catalog.may_grant_object(object.kind, &object.name, privilege)? {
                return Err(Error::PermissionDenied(format!(
                    "cannot grant {privilege} on {noun} `{name}` — the session neither owns it nor \
                     holds that privilege WITH GRANT OPTION",
                    privilege = privilege.as_str(),
                    noun = object.kind.noun(),
                    name = object.name,
                )));
            }
        }
    }
    Ok(())
}

/// Resolve a `GRANT`/`REVOKE` target's canonical (`schema.table` or bare) name to its table schema.
fn lookup_grant_table(
    catalog: &dyn Catalog,
    canonical: &str,
) -> Result<nusadb_core::TableSchema, Error> {
    let table = match canonical.split_once('.') {
        Some((schema, name)) => catalog.lookup_table_in(schema, name)?,
        None => catalog.lookup_table(canonical)?,
    };
    table.ok_or_else(|| Error::TableNotFound {
        name: canonical.to_owned(),
    })
}

/// Validate the column-scoped privileges of a `GRANT`/`REVOKE`: they apply only to named tables, each
/// named column must exist, and the session must have the authority to pass the privilege on (the
/// same `may_grant` authority a table-wide grant needs). Empty is a no-op.
fn check_column_privileges(
    catalog: &dyn Catalog,
    column_privileges: &[ast::ColumnPrivilege],
    targets: &GrantTargetsPlan,
) -> Result<(), Error> {
    if column_privileges.is_empty() {
        return Ok(());
    }
    let GrantTargetsPlan::Objects(objects) = targets else {
        return Err(Error::InvalidStatement(
            "column-scoped privileges apply only to named tables, not an `ALL ... IN SCHEMA` form"
                .to_owned(),
        ));
    };
    for object in objects {
        if object.kind != ObjectKind::Table {
            return Err(Error::InvalidStatement(format!(
                "column-scoped privileges apply only to tables, not a {}",
                object.kind.noun()
            )));
        }
        let table = lookup_grant_table(catalog, &object.name)?;
        for column_privilege in column_privileges {
            for column in &column_privilege.columns {
                if !table.columns.iter().any(|c| &c.name == column) {
                    return Err(Error::ColumnNotFound {
                        table: object.name.clone(),
                        column: column.clone(),
                    });
                }
            }
            if !catalog.may_grant_object(
                ObjectKind::Table,
                &object.name,
                column_privilege.privilege,
            )? {
                return Err(Error::PermissionDenied(format!(
                    "cannot grant {privilege} on columns of table `{name}` — the session neither \
                     owns it nor holds that privilege WITH GRANT OPTION",
                    privilege = column_privilege.privilege.as_str(),
                    name = object.name,
                )));
            }
        }
    }
    Ok(())
}

/// Analyze `GRANT privileges ON objects TO grantees`.
///
/// # Errors
/// Propagates catalog errors; [`Error::PermissionDenied`] when the session may not grant.
pub(super) fn analyze_grant(g: ast::Grant, catalog: &dyn Catalog) -> Result<GrantPlan, Error> {
    let targets = resolve_targets(catalog, g.targets)?;
    check_privileges_apply(g.privileges.as_ref(), &targets)?;
    check_may_grant(catalog, g.privileges.as_ref(), &targets)?;
    check_column_privileges(catalog, &g.column_privileges, &targets)?;
    require_known_grantees(catalog, &g.grantees)?;
    Ok(GrantPlan {
        privileges: g.privileges,
        column_privileges: g.column_privileges,
        targets,
        grantees: g.grantees,
        with_grant_option: g.with_grant_option,
    })
}

/// Analyze `REVOKE privileges ON objects FROM grantees`.
///
/// Revoking needs the same authority as granting: whoever could hand the privilege out is who may
/// take it back.
///
/// # Errors
/// Propagates catalog errors; [`Error::PermissionDenied`] when the session may not revoke.
pub(super) fn analyze_revoke(r: ast::Revoke, catalog: &dyn Catalog) -> Result<RevokePlan, Error> {
    let targets = resolve_targets(catalog, r.targets)?;
    check_privileges_apply(r.privileges.as_ref(), &targets)?;
    check_may_grant(catalog, r.privileges.as_ref(), &targets)?;
    check_column_privileges(catalog, &r.column_privileges, &targets)?;
    Ok(RevokePlan {
        privileges: r.privileges,
        column_privileges: r.column_privileges,
        targets,
        grantees: r.grantees,
        grant_option_for: r.grant_option_for,
        cascade: r.cascade,
    })
}

/// Refuse a grant to a role that does not exist, so a typo is reported rather than silently
/// stored as a grant that can never match.
fn require_known_grantees(catalog: &dyn Catalog, grantees: &[Grantee]) -> Result<(), Error> {
    for grantee in grantees {
        if let Grantee::Role(name) = grantee
            && !catalog.role_exists(name)?
        {
            return Err(Error::ObjectNotFound(format!(
                "role `{name}` does not exist"
            )));
        }
    }
    Ok(())
}

/// Analyze `GRANT role TO member`.
///
/// # Errors
/// Propagates catalog errors; [`Error::PermissionDenied`] when the session may not administer one
/// of the roles.
pub(super) fn analyze_grant_role(
    g: ast::GrantRole,
    catalog: &dyn Catalog,
) -> Result<GrantRolePlan, Error> {
    for role in g.roles.iter().chain(&g.members) {
        if !catalog.role_exists(role)? {
            return Err(Error::ObjectNotFound(format!(
                "role `{role}` does not exist"
            )));
        }
    }
    for role in &g.roles {
        if !catalog.may_administer_role(role)? {
            return Err(Error::PermissionDenied(format!(
                "cannot grant membership in role `{role}` — the session is neither a superuser, a \
                 CREATEROLE role, nor a member of it WITH ADMIN OPTION"
            )));
        }
    }
    Ok(GrantRolePlan {
        roles: g.roles,
        members: g.members,
        with_admin_option: g.with_admin_option,
    })
}

/// Analyze `REVOKE role FROM member`.
///
/// # Errors
/// Propagates catalog errors; [`Error::PermissionDenied`] when the session may not administer one
/// of the roles.
pub(super) fn analyze_revoke_role(
    r: ast::RevokeRole,
    catalog: &dyn Catalog,
) -> Result<RevokeRolePlan, Error> {
    // Mirror the GRANT twin: a typo'd role name is a loud error, not a silent no-op.
    for role in r.roles.iter().chain(&r.members) {
        if !catalog.role_exists(role)? {
            return Err(Error::ObjectNotFound(format!(
                "role `{role}` does not exist"
            )));
        }
    }
    for role in &r.roles {
        if !catalog.may_administer_role(role)? {
            return Err(Error::PermissionDenied(format!(
                "cannot revoke membership in role `{role}` — the session is neither a superuser, a \
                 CREATEROLE role, nor a member of it WITH ADMIN OPTION"
            )));
        }
    }
    Ok(RevokeRolePlan {
        roles: r.roles,
        members: r.members,
        admin_option_for: r.admin_option_for,
    })
}

/// Analyze `CREATE ROLE`.
///
/// Only a superuser may create a superuser: a `CREATEROLE` role that could mint one would hold
/// superuser authority by a two-step path, which defeats the point of not being one.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session may not create roles; [`Error::Unsupported`] for a
/// reserved name.
pub(super) fn analyze_create_role(
    c: ast::CreateRole,
    catalog: &dyn Catalog,
) -> Result<CreateRolePlan, Error> {
    crate::rbac::validate_role_name(&c.name)?;
    if !catalog.may_create_role()? {
        return Err(Error::PermissionDenied(
            "CREATE ROLE requires a superuser or a role with CREATEROLE".to_owned(),
        ));
    }
    if c.options.superuser == Some(true) && !catalog.is_superuser() {
        return Err(Error::PermissionDenied(
            "only a superuser may create a superuser role".to_owned(),
        ));
    }
    for role in &c.in_role {
        if !catalog.may_administer_role(role)? {
            return Err(Error::PermissionDenied(format!(
                "cannot add the new role to `{role}` — the session may not administer that role"
            )));
        }
    }
    Ok(CreateRolePlan {
        name: c.name,
        if_not_exists: c.if_not_exists,
        options: c.options,
        in_role: c.in_role,
    })
}

/// Analyze `DROP ROLE`.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session may not administer one of the roles.
pub(super) fn analyze_drop_role(
    d: ast::DropRole,
    catalog: &dyn Catalog,
) -> Result<DropRolePlan, Error> {
    for name in &d.names {
        crate::rbac::validate_role_name(name)?;
        if !catalog.may_administer_role(name)? {
            return Err(Error::PermissionDenied(format!(
                "cannot drop role `{name}` — requires a superuser or a role with CREATEROLE"
            )));
        }
    }
    Ok(DropRolePlan {
        names: d.names,
        if_exists: d.if_exists,
    })
}

/// Analyze `ALTER ROLE`.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session may not administer the role, or when a
/// non-superuser tries to grant or withdraw the superuser attribute.
pub(super) fn analyze_alter_role(
    a: ast::AlterRole,
    catalog: &dyn Catalog,
) -> Result<AlterRolePlan, Error> {
    crate::rbac::validate_role_name(&a.name)?;
    if !catalog.may_administer_role(&a.name)? {
        return Err(Error::PermissionDenied(format!(
            "cannot alter role `{name}` — requires a superuser or a role with CREATEROLE",
            name = a.name
        )));
    }
    if a.options.superuser.is_some() && !catalog.is_superuser() {
        return Err(Error::PermissionDenied(
            "only a superuser may change the SUPERUSER attribute".to_owned(),
        ));
    }
    Ok(AlterRolePlan {
        name: a.name,
        options: a.options,
    })
}

/// Analyze `SET ROLE` / `RESET ROLE`.
///
/// The membership check is the point of the statement: without it `SET ROLE` would be a way to
/// assume any role at all, which would make every other check here decorative. The executor
/// re-checks before switching, so a plan cached under one session cannot be replayed under
/// another to escalate.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session is not a member of the target role.
pub(super) fn analyze_set_role(
    s: ast::SetRole,
    catalog: &dyn Catalog,
) -> Result<SetRolePlan, Error> {
    if let Some(role) = &s.role {
        if !catalog.role_exists(role)? {
            return Err(Error::ObjectNotFound(format!(
                "role `{role}` does not exist"
            )));
        }
        if !catalog.may_assume_role(role)? {
            return Err(Error::PermissionDenied(format!(
                "cannot SET ROLE to `{role}` — the session is not a member of it"
            )));
        }
    }
    Ok(SetRolePlan { role: s.role })
}
