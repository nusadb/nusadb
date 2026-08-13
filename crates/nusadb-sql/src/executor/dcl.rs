//! Execution of the data-control statements.
//!
//! Everything here writes ordinary rows into the [`crate::rbac`] catalogs inside the statement's
//! transaction, so a `GRANT` in a transaction that later rolls back never took effect — the same
//! guarantee any other write gets, obtained by not doing anything special.
//!
//! Re-checking authority here rather than trusting the analyzer is deliberate. The analyzer runs
//! once and its plan can be cached, but the catalogs it consulted are mutable: a role's membership
//! can change between planning and execution. The checks that matter for escalation — expanding
//! `ALL TABLES IN SCHEMA`, and `SET ROLE` — are therefore made again against the transaction's own
//! snapshot.

use nusadb_core::TxnId;
use nusadb_core::engine::StorageEngine;

use crate::ast::{Grantee, ObjectKind, Privilege};
use crate::error::Error;
use crate::planner::{
    AlterRolePlan, CreateRolePlan, DropRolePlan, GrantPlan, GrantRolePlan, GrantTargetsPlan,
    RevokePlan, RevokeRolePlan, SetRolePlan,
};
use crate::rbac::{self, GrantRecord, RoleRecord};

use super::{ExecutionResult, session_ctx};

/// The role the running statement acts as.
fn actor() -> String {
    session_ctx::current_user()
}

/// Expand a plan's targets into concrete objects, resolving the `ALL ... IN SCHEMA` forms against
/// the tables that exist right now.
fn expand_targets(
    targets: &GrantTargetsPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<(ObjectKind, String)>, Error> {
    Ok(match targets {
        GrantTargetsPlan::Objects(objects) => {
            objects.iter().map(|o| (o.kind, o.name.clone())).collect()
        },
        GrantTargetsPlan::AllTablesInSchemas(schemas) => engine
            .list_tables_qualified_as_of(txn)?
            .into_iter()
            .filter(|(schema, _)| schemas.contains(schema))
            // The privilege catalogs are themselves tables. Handing them out under `ALL TABLES IN
            // SCHEMA public` would let a grantee rewrite its own grants, so they are never
            // included in a wildcard expansion.
            .filter(|(_, name)| !is_access_control_catalog(name))
            .map(|(schema, name)| (ObjectKind::Table, format!("{schema}.{name}")))
            .collect(),
        // The storage treaty exposes no way to enumerate a schema's sequences, and widening it is
        // a coordinated change rather than something to slip in here. Refusing is the honest
        // option: silently expanding to nothing would report success while granting nobody
        // anything. Naming sequences explicitly works.
        GrantTargetsPlan::AllSequencesInSchemas(_) => {
            return Err(Error::Unsupported(
                "GRANT/REVOKE ... ON ALL SEQUENCES IN SCHEMA is not supported (sequences cannot \
                 yet be enumerated per schema); name the sequences explicitly instead"
                    .to_owned(),
            ));
        },
    })
}

/// Whether `name` is one of the catalogs access control itself is stored in.
///
/// Write access to any of these is equivalent to superuser, so they are excluded from wildcard
/// grants. Naming one explicitly still works for a superuser, who could do it anyway.
fn is_access_control_catalog(name: &str) -> bool {
    matches!(
        name,
        rbac::ROLE_CATALOG
            | rbac::ROLE_MEMBER_CATALOG
            | rbac::PRIVILEGE_CATALOG
            | rbac::OWNER_CATALOG
    )
}

/// `GRANT privileges ON objects TO grantees`.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session may not pass on a privilege it named; propagates
/// storage errors.
pub(super) fn run_grant(
    plan: &GrantPlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let actor = actor();
    for (kind, object) in expand_targets(&plan.targets, engine, txn)? {
        let wanted = plan
            .privileges
            .clone()
            .unwrap_or_else(|| Privilege::all_for(kind));
        for privilege in wanted {
            if !privilege.applies_to(kind) {
                // Reached only through `ALL PRIVILEGES` over a mixed list, where the expansion is
                // per object kind; an explicitly named mismatch was already refused by the analyzer.
                continue;
            }
            if !rbac::may_grant(engine, txn, &actor, kind, &object, privilege)? {
                return Err(Error::PermissionDenied(format!(
                    "cannot grant {privilege} on {noun} `{object}` — the session neither owns it \
                     nor holds that privilege WITH GRANT OPTION",
                    privilege = privilege.as_str(),
                    noun = kind.noun(),
                )));
            }
            for grantee in &plan.grantees {
                rbac::insert_grant(
                    engine,
                    txn,
                    &GrantRecord {
                        grantee: grantee.clone(),
                        kind,
                        object: object.clone(),
                        privilege,
                        grantor: actor.clone(),
                        grantable: plan.with_grant_option,
                    },
                )?;
            }
        }
    }
    Ok(ExecutionResult::Granted)
}

/// `REVOKE privileges ON objects FROM grantees`.
///
/// `RESTRICT` (the default) refuses when the grantee has passed the privilege on; `CASCADE`
/// removes those dependent grants too, following the chain to its end.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session may not revoke; [`Error::Unsupported`] when
/// `RESTRICT` meets a dependent grant.
pub(super) fn run_revoke(
    plan: &RevokePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let actor = actor();
    for (kind, object) in expand_targets(&plan.targets, engine, txn)? {
        let wanted = plan
            .privileges
            .clone()
            .unwrap_or_else(|| Privilege::all_for(kind));
        for privilege in wanted {
            if !privilege.applies_to(kind) {
                continue;
            }
            if !rbac::may_grant(engine, txn, &actor, kind, &object, privilege)? {
                return Err(Error::PermissionDenied(format!(
                    "cannot revoke {privilege} on {noun} `{object}` — the session neither owns it \
                     nor holds that privilege WITH GRANT OPTION",
                    privilege = privilege.as_str(),
                    noun = kind.noun(),
                )));
            }
            let target = RevokeTarget {
                engine,
                txn,
                plan,
                kind,
                object: &object,
                privilege,
            };
            for grantee in &plan.grantees {
                let mut visited = std::collections::HashSet::new();
                revoke_one(&target, grantee, &mut visited)?;
            }
        }
    }
    Ok(ExecutionResult::Revoked)
}

/// The parts of a `REVOKE` that stay fixed while the cascade walks from grantee to grantee.
struct RevokeTarget<'a> {
    engine: &'a dyn StorageEngine,
    txn: TxnId,
    plan: &'a RevokePlan,
    kind: ObjectKind,
    object: &'a str,
    privilege: Privilege,
}

/// Revoke one (grantee, object, privilege) triple, honouring `GRANT OPTION FOR` and the
/// cascade/restrict behaviour.
///
/// `visited` guards the cascade against a cycle in the *grantor* graph. A grant is stored as one
/// row per `(grantee, kind, object, privilege)`, so a re-grant re-parents the existing row to the
/// new grantor — two ordinary roles passing a grant option back and forth end up with `a`'s grantor
/// being `b` while `b`'s is `a`. Without the guard the cascade walks that loop forever and
/// overflows the stack, which aborts the process rather than raising a catchable error.
fn revoke_one(
    target: &RevokeTarget,
    grantee: &Grantee,
    visited: &mut std::collections::HashSet<String>,
) -> Result<(), Error> {
    if !visited.insert(grantee.as_str().to_owned()) {
        return Ok(());
    }
    let RevokeTarget {
        engine,
        txn,
        plan,
        kind,
        object,
        privilege,
    } = *target;
    let dependents = rbac::dependent_grants(engine, txn, grantee, kind, object, privilege)?;
    if !dependents.is_empty() {
        if !plan.cascade {
            return Err(Error::Unsupported(format!(
                "cannot revoke {privilege} on {noun} `{object}` from `{who}`: it has granted that \
                 privilege to {n} other role(s) — use CASCADE to revoke those too",
                privilege = privilege.as_str(),
                noun = kind.noun(),
                who = grantee.as_str(),
                n = dependents.len(),
            )));
        }
        for dependent in dependents {
            // Recurse so a chain of grant-option handoffs is unwound to its end, not just one link.
            revoke_one(target, &dependent.grantee, visited)?;
        }
    }
    if plan.grant_option_for {
        rbac::downgrade_grant_option(engine, txn, grantee, kind, object, privilege)?;
    } else {
        rbac::delete_grant(engine, txn, grantee, kind, object, Some(privilege))?;
    }
    Ok(())
}

/// `GRANT role TO member [WITH ADMIN OPTION]`.
///
/// # Errors
/// [`Error::Unsupported`] when the grant would close a membership cycle.
pub(super) fn run_grant_role(
    plan: &GrantRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    for role in &plan.roles {
        for member in &plan.members {
            if rbac::would_create_cycle(engine, txn, role, member)? {
                return Err(Error::Unsupported(format!(
                    "granting `{role}` to `{member}` would create a membership cycle"
                )));
            }
            rbac::insert_membership(engine, txn, role, member, plan.with_admin_option)?;
        }
    }
    Ok(ExecutionResult::Granted)
}

/// `REVOKE role FROM member`.
///
/// # Errors
/// Propagates storage errors.
pub(super) fn run_revoke_role(
    plan: &RevokeRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    for role in &plan.roles {
        for member in &plan.members {
            if plan.admin_option_for {
                // Keep the membership, drop only the right to pass it on.
                if rbac::all_memberships(engine, txn)?
                    .iter()
                    .any(|e| &e.role == role && &e.member == member)
                {
                    rbac::insert_membership(engine, txn, role, member, false)?;
                }
            } else {
                rbac::delete_membership(engine, txn, role, member)?;
            }
        }
    }
    Ok(ExecutionResult::Revoked)
}

/// `CREATE ROLE | USER`.
///
/// # Errors
/// [`Error::Unsupported`] when the role already exists and `IF NOT EXISTS` was not given.
pub(super) fn run_create_role(
    plan: &CreateRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    if rbac::lookup_role(engine, txn, &plan.name)?.is_some() {
        if plan.if_not_exists {
            return Ok(ExecutionResult::RoleCreated);
        }
        return Err(Error::Unsupported(format!(
            "role `{}` already exists",
            plan.name
        )));
    }
    let options = &plan.options;
    rbac::insert_role(
        engine,
        txn,
        &RoleRecord {
            name: plan.name.clone(),
            // Every attribute defaults off except INHERIT, which defaults on: a role granted
            // membership is expected to be able to use it without a `SET ROLE` first.
            superuser: options.superuser.unwrap_or(false),
            login: options.login.unwrap_or(false),
            create_db: options.create_db.unwrap_or(false),
            create_role: options.create_role.unwrap_or(false),
            inherit: options.inherit.unwrap_or(true),
        },
    )?;
    for role in &plan.in_role {
        rbac::insert_membership(engine, txn, role, &plan.name, false)?;
    }
    Ok(ExecutionResult::RoleCreated)
}

/// `DROP ROLE | USER`.
///
/// A role that still owns objects is refused: dropping it would leave those objects owned by a
/// name nothing can resolve, and the safe reading of an unresolvable owner (superuser-only) would
/// silently lock everyone else out of data they could reach a moment earlier.
///
/// # Errors
/// [`Error::Unsupported`] when a named role does not exist (without `IF EXISTS`) or still owns
/// objects.
pub(super) fn run_drop_role(
    plan: &DropRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    for name in &plan.names {
        if rbac::lookup_role(engine, txn, name)?.is_none() {
            if plan.if_exists {
                continue;
            }
            return Err(Error::Unsupported(format!("role `{name}` does not exist")));
        }
        if let Some(object) = first_object_owned_by(engine, txn, name)? {
            return Err(Error::Unsupported(format!(
                "cannot drop role `{name}`: it still owns `{object}` — reassign or drop the \
                 objects it owns first"
            )));
        }
        rbac::delete_grants_of_role(engine, txn, name)?;
        rbac::delete_memberships_of(engine, txn, name)?;
        rbac::delete_role_row(engine, txn, name)?;
    }
    Ok(ExecutionResult::RoleDropped)
}

/// The first object `owner` owns, if any — the witness a `DROP ROLE` refusal names.
fn first_object_owned_by(
    engine: &dyn StorageEngine,
    txn: TxnId,
    owner: &str,
) -> Result<Option<String>, Error> {
    for (schema, table) in engine.list_tables_qualified_as_of(txn)? {
        let name = format!("{schema}.{table}");
        if rbac::object_owner(engine, txn, ObjectKind::Table, &name)? == owner {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// `ALTER ROLE`.
///
/// # Errors
/// [`Error::Unsupported`] when the role does not exist.
pub(super) fn run_alter_role(
    plan: &AlterRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let Some(mut record) = rbac::lookup_role(engine, txn, &plan.name)? else {
        return Err(Error::Unsupported(format!(
            "role `{}` does not exist",
            plan.name
        )));
    };
    let options = &plan.options;
    // An unmentioned attribute keeps its stored value — `ALTER ROLE r NOLOGIN` changes login and
    // nothing else.
    record.superuser = options.superuser.unwrap_or(record.superuser);
    record.login = options.login.unwrap_or(record.login);
    record.create_db = options.create_db.unwrap_or(record.create_db);
    record.create_role = options.create_role.unwrap_or(record.create_role);
    record.inherit = options.inherit.unwrap_or(record.inherit);
    rbac::update_role(engine, txn, &record)?;
    Ok(ExecutionResult::RoleAltered)
}

/// `SET ROLE` / `RESET ROLE`.
///
/// The membership check is made here, against this transaction's snapshot, and not merely in the
/// analyzer: a cached plan must not be able to carry an authorization decision made when the
/// session's memberships were different.
///
/// # Errors
/// [`Error::PermissionDenied`] when the session is not a member of the target role.
pub(super) fn run_set_role(
    plan: &SetRolePlan,
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<ExecutionResult, Error> {
    let Some(role) = &plan.role else {
        return Ok(ExecutionResult::RoleSet(None));
    };
    let actor = actor();
    let permitted = rbac::may_assume_role(engine, txn, &actor, role)?;
    if !permitted {
        return Err(Error::PermissionDenied(format!(
            "cannot SET ROLE to `{role}` — the session is not a member of it"
        )));
    }
    Ok(ExecutionResult::RoleSet(Some(role.clone())))
}
