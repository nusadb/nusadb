//! Role-based access control: the role catalog, privilege grants, ownership, and the decision
//! procedure that turns them into a yes or no.
//!
//! # Where the answer comes from
//!
//! A session asks one question — *may `user` do `privilege` to `object`?* — and three things can
//! answer yes:
//!
//! 1. the session is a **superuser**, which bypasses every check;
//! 2. the session **owns** the object, which confers every privilege on it implicitly; or
//! 3. a **grant** exists for the privilege, held by any role the session effectively has —
//!    itself, `PUBLIC`, or any role it inherits membership in, transitively.
//!
//! Nothing else grants. Absence of a grant is a denial, so a corrupt or unreadable catalog row
//! narrows access rather than widening it: every read path below skips rows it cannot decode
//! instead of guessing at their meaning.
//!
//! # Storage
//!
//! Four catalog tables, created lazily on first write, following the same all-text shape as the
//! row-level-security catalogs. No `StorageEngine` treaty change: everything here is ordinary rows
//! in ordinary tables, which also means grants are transactional and crash-safe for free —
//! a `GRANT` inside a rolled-back transaction never took effect.
//!
//! | Table | Columns |
//! |---|---|
//! | [`ROLE_CATALOG`] | `name`, `superuser`, `login`, `createdb`, `createrole`, `inherit` |
//! | [`ROLE_MEMBER_CATALOG`] | `role`, `member`, `admin` |
//! | [`PRIVILEGE_CATALOG`] | `grantee`, `kind`, `object`, `privilege`, `grantor`, `grantable` |
//! | [`OWNER_CATALOG`] | `kind`, `object`, `owner` |
//!
//! # Objects that predate this catalog
//!
//! A table created before access control existed has no owner row. Rather than treat "no owner"
//! as "anyone may", [`object_owner`] reports the bootstrap superuser, so such objects are
//! reachable only by a superuser or an explicit grant. That is the safe reading, and it is why
//! upgrading an installation that used a non-superuser login is a breaking change — documented,
//! deliberate, and preferable to leaving the data open.

use std::collections::{BTreeSet, HashMap, HashSet};

use nusadb_core::TxnId;
use nusadb_core::engine::{ColumnDef, ColumnType, StorageEngine, TableDef};

use crate::ast::{self, Grantee, ObjectKind, PUBLIC_GRANTEE, Privilege};
use crate::error::Error;
use crate::executor::row;

/// Roles and their attributes.
pub const ROLE_CATALOG: &str = "nusadb_roles";
/// Role membership edges (`member` is a member of `role`).
pub const ROLE_MEMBER_CATALOG: &str = "nusadb_role_members";
/// Object privilege grants.
pub const PRIVILEGE_CATALOG: &str = "nusadb_privileges";
/// Column-scoped privilege grants (`GRANT SELECT (col) ...`). Kept separate from
/// [`PRIVILEGE_CATALOG`] so the table-wide grant format is unchanged.
pub const COLUMN_PRIVILEGE_CATALOG: &str = "nusadb_column_privileges";
/// Object ownership.
pub const OWNER_CATALOG: &str = "nusadb_owners";

/// `(name, superuser, login, createdb, createrole, inherit)`.
const ROLE_SCHEMA: [ColumnType; 6] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];
/// `(role, member, admin)`.
const MEMBER_SCHEMA: [ColumnType; 3] = [ColumnType::Text, ColumnType::Text, ColumnType::Text];
/// `(grantee, kind, object, privilege, grantor, grantable)`.
const PRIVILEGE_SCHEMA: [ColumnType; 6] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];
/// `(grantee, object, column, privilege, grantor, grantable)` — object is always a table.
const COLUMN_PRIVILEGE_SCHEMA: [ColumnType; 6] = [
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
    ColumnType::Text,
];
/// `(kind, object, owner)`.
const OWNER_SCHEMA: [ColumnType; 3] = [ColumnType::Text, ColumnType::Text, ColumnType::Text];

/// The boolean spelling used in the all-text catalogs.
const TRUE: &str = "t";
/// See [`TRUE`]. Any value that is not [`TRUE`] reads as false, so an unreadable flag denies.
const FALSE: &str = "f";

/// Render a flag for storage.
fn flag(value: bool) -> String {
    if value { TRUE } else { FALSE }.to_owned()
}

/// A role as stored, with its attributes resolved.
///
/// The attributes are independent switches rather than a level or a bitmask, which is what the SQL
/// standard models and what `ALTER ROLE r NOCREATEDB` expects to change on its own; grouping them
/// into a flags type would only rename the same five booleans.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each attribute is an independent role switch the SQL surface sets by name"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleRecord {
    /// Role name.
    pub name: String,
    /// Bypasses every privilege check.
    pub superuser: bool,
    /// May open a connection.
    pub login: bool,
    /// May create databases.
    pub create_db: bool,
    /// May create, alter, and drop roles.
    pub create_role: bool,
    /// Whether membership automatically confers the granted role's privileges.
    pub inherit: bool,
}

/// A grant as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantRecord {
    /// Who holds it.
    pub grantee: Grantee,
    /// What kind of object.
    pub kind: ObjectKind,
    /// The object's canonical name.
    pub object: String,
    /// The privilege held.
    pub privilege: Privilege,
    /// The role that granted it — the audit trail `REVOKE ... CASCADE` walks.
    pub grantor: String,
    /// Whether the holder may pass it on.
    pub grantable: bool,
}

/// A column-scoped grant as stored. The object is always a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnGrantRecord {
    /// Who holds it.
    pub grantee: Grantee,
    /// The table's canonical (`schema.table`) name.
    pub object: String,
    /// The column the privilege is scoped to.
    pub column: String,
    /// The privilege held on that column (`SELECT`, `INSERT`, `UPDATE`, or `REFERENCES`).
    pub privilege: Privilege,
    /// The role that granted it.
    pub grantor: String,
    /// Whether the holder may pass it on.
    pub grantable: bool,
}

/// Whether `name` is usable as a role name.
///
/// The reserved grantee spelling `public` is refused so a real role can never collide with the
/// pseudo-role every session belongs to — otherwise `GRANT ... TO public` would be ambiguous
/// between "everyone" and "that one role", and the ambiguity would resolve toward more access.
///
/// # Errors
/// Returns [`Error::Unsupported`] for an empty or reserved name.
pub fn validate_role_name(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::InvalidStatement(
            "a role name may not be empty".to_owned(),
        ));
    }
    if name.eq_ignore_ascii_case(PUBLIC_GRANTEE) {
        return Err(Error::InvalidStatement(
            "`public` is reserved for the pseudo-role every session belongs to and may not be \
             created, altered, or dropped as a role"
                .to_owned(),
        ));
    }
    // The bootstrap superuser's name is what `object_owner` reports for every unowned object, so
    // a catalog role created under that name would inherit ownership of them all. Reserve it.
    if name.eq_ignore_ascii_case(crate::BOOTSTRAP_SUPERUSER) {
        return Err(Error::InvalidStatement(format!(
            "`{}` is reserved for the bootstrap superuser and may not be created, altered, or \
             dropped as a role",
            crate::BOOTSTRAP_SUPERUSER
        )));
    }
    Ok(())
}

// === Catalog creation =====================================================

/// Look up a catalog table, creating it with `columns` if absent.
fn ensure_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
    columns: &[&str],
) -> Result<nusadb_core::TableId, Error> {
    if let Some(schema) = engine.lookup_table_as_of(txn, name)? {
        return Ok(schema.id);
    }
    let def = TableDef {
        schema: "public".to_owned(),
        name: name.to_owned(),
        columns: columns
            .iter()
            .map(|c| ColumnDef {
                name: (*c).to_owned(),
                ty: ColumnType::Text,
                nullable: false,
            })
            .collect(),
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Create the role catalog if absent.
fn ensure_role_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    ensure_catalog(
        engine,
        txn,
        ROLE_CATALOG,
        &[
            "name",
            "superuser",
            "login",
            "createdb",
            "createrole",
            "inherit",
        ],
    )
}

/// Create the membership catalog if absent.
fn ensure_member_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    ensure_catalog(
        engine,
        txn,
        ROLE_MEMBER_CATALOG,
        &["role", "member", "admin"],
    )
}

/// Create the privilege catalog if absent.
fn ensure_privilege_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    ensure_catalog(
        engine,
        txn,
        PRIVILEGE_CATALOG,
        &[
            "grantee",
            "kind",
            "object",
            "privilege",
            "grantor",
            "grantable",
        ],
    )
}

/// Create the ownership catalog if absent.
fn ensure_owner_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    ensure_catalog(engine, txn, OWNER_CATALOG, &["kind", "object", "owner"])
}

// === Roles ================================================================

/// Every role in the catalog.
///
/// # Errors
/// Propagates storage errors. A row that does not decode to the expected shape is skipped.
pub fn all_roles(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<RoleRecord>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, ROLE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &ROLE_SCHEMA)?;
        if let [
            ast::Value::Text(name),
            ast::Value::Text(superuser),
            ast::Value::Text(login),
            ast::Value::Text(create_db),
            ast::Value::Text(create_role),
            ast::Value::Text(inherit),
        ] = row.as_slice()
        {
            out.push(RoleRecord {
                name: name.clone(),
                superuser: superuser == TRUE,
                login: login == TRUE,
                create_db: create_db == TRUE,
                create_role: create_role == TRUE,
                inherit: inherit == TRUE,
            });
        }
    }
    Ok(out)
}

/// Look up one role by name.
///
/// # Errors
/// Propagates storage errors.
pub fn lookup_role(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<Option<RoleRecord>, Error> {
    Ok(all_roles(engine, txn)?.into_iter().find(|r| r.name == name))
}

/// Insert a role. The caller has already checked that the name is free.
///
/// # Errors
/// Propagates storage errors.
pub fn insert_role(engine: &dyn StorageEngine, txn: TxnId, role: &RoleRecord) -> Result<(), Error> {
    let cat = ensure_role_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(role.name.clone()),
        ast::Value::Text(flag(role.superuser)),
        ast::Value::Text(flag(role.login)),
        ast::Value::Text(flag(role.create_db)),
        ast::Value::Text(flag(role.create_role)),
        ast::Value::Text(flag(role.inherit)),
    ];
    let bytes = row::encode(&row, &ROLE_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Replace a role's stored attributes (an `ALTER ROLE` rewrite).
///
/// # Errors
/// Propagates storage errors.
pub fn update_role(engine: &dyn StorageEngine, txn: TxnId, role: &RoleRecord) -> Result<(), Error> {
    delete_role_row(engine, txn, &role.name)?;
    insert_role(engine, txn, role)
}

/// Remove a role's catalog row, reporting whether one was present.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_role_row(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, ROLE_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &ROLE_SCHEMA)?;
        if matches!(row.first(), Some(ast::Value::Text(n)) if n == name) {
            victims.push(tid);
        }
    }
    let present = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(present)
}

// === Membership ===========================================================

/// One membership edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipRecord {
    /// The role whose privileges flow to `member`.
    pub role: String,
    /// The role receiving them.
    pub member: String,
    /// Whether `member` may grant this membership on.
    pub admin: bool,
}

/// Every membership edge.
///
/// # Errors
/// Propagates storage errors.
pub fn all_memberships(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<MembershipRecord>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, ROLE_MEMBER_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &MEMBER_SCHEMA)?;
        if let [
            ast::Value::Text(role),
            ast::Value::Text(member),
            ast::Value::Text(admin),
        ] = row.as_slice()
        {
            out.push(MembershipRecord {
                role: role.clone(),
                member: member.clone(),
                admin: admin == TRUE,
            });
        }
    }
    Ok(out)
}

/// Add or update a membership edge. Re-granting an existing membership with `WITH ADMIN OPTION`
/// upgrades it in place rather than duplicating the row.
///
/// # Errors
/// Propagates storage errors.
pub fn insert_membership(
    engine: &dyn StorageEngine,
    txn: TxnId,
    role: &str,
    member: &str,
    admin: bool,
) -> Result<(), Error> {
    delete_membership(engine, txn, role, member)?;
    let cat = ensure_member_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(role.to_owned()),
        ast::Value::Text(member.to_owned()),
        ast::Value::Text(flag(admin)),
    ];
    let bytes = row::encode(&row, &MEMBER_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Remove a membership edge, reporting whether one was present.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_membership(
    engine: &dyn StorageEngine,
    txn: TxnId,
    role: &str,
    member: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, ROLE_MEMBER_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &MEMBER_SCHEMA)?;
        if matches!((row.first(), row.get(1)),
            (Some(ast::Value::Text(r)), Some(ast::Value::Text(m))) if r == role && m == member)
        {
            victims.push(tid);
        }
    }
    let present = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(present)
}

/// Remove every membership edge mentioning `name`, on either side — used when a role is dropped so
/// no edge is left pointing at a role that no longer exists.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_memberships_of(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, ROLE_MEMBER_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &MEMBER_SCHEMA)?;
        if matches!((row.first(), row.get(1)),
            (Some(ast::Value::Text(r)), Some(ast::Value::Text(m))) if r == name || m == name)
        {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// The set of roles `user` effectively acts as: itself plus every role it inherits, transitively.
///
/// Only edges whose *member* inherits are followed — a `NOINHERIT` role holds its memberships but
/// must `SET ROLE` to use them, so they do not silently widen what its statements may do.
///
/// The walk is a breadth-first closure with a visited set, so a membership cycle (which
/// [`would_create_cycle`] prevents at grant time, but which a hand-edited catalog could still
/// contain) terminates instead of looping forever.
///
/// # Errors
/// Propagates storage errors.
pub fn effective_roles(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
) -> Result<BTreeSet<String>, Error> {
    let memberships = all_memberships(engine, txn)?;
    let roles = all_roles(engine, txn)?;
    Ok(effective_roles_from(&memberships, &roles, user))
}

/// The membership closure computed from already-fetched catalogs — no scan. Split out so a caller
/// that also needs `all_roles`/`all_memberships` for something else (the superuser attribute, a
/// grant scan) fetches each catalog once instead of paying for a fresh scan per question.
fn effective_roles_from(
    memberships: &[MembershipRecord],
    roles: &[RoleRecord],
    user: &str,
) -> BTreeSet<String> {
    let inherits: HashMap<&str, bool> =
        roles.iter().map(|r| (r.name.as_str(), r.inherit)).collect();

    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert(user.to_owned());
    let mut frontier = vec![user.to_owned()];
    while let Some(current) = frontier.pop() {
        // A role absent from the catalog (the bootstrap superuser, which is not a stored role)
        // inherits by default — the standard behaviour, and the conservative one only because the
        // bootstrap user bypasses checks anyway.
        if !inherits.get(current.as_str()).copied().unwrap_or(true) {
            continue;
        }
        for edge in memberships.iter().filter(|e| e.member == current) {
            if seen.insert(edge.role.clone()) {
                frontier.push(edge.role.clone());
            }
        }
    }
    seen
}

/// The full membership closure of `user`: every role it is a member of, directly or transitively,
/// **ignoring the `INHERIT` flag**. This is the set a session may `SET ROLE` into — a `NOINHERIT`
/// role does not automatically wield its memberships' privileges (that is [`effective_roles_from`],
/// which stops at the `NOINHERIT` edge), but it may still switch into them, which is precisely what
/// `NOINHERIT` is for. `user` itself is included (a session may always `SET ROLE` back to itself).
fn membership_closure_from(memberships: &[MembershipRecord], user: &str) -> BTreeSet<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert(user.to_owned());
    let mut frontier = vec![user.to_owned()];
    while let Some(current) = frontier.pop() {
        for edge in memberships.iter().filter(|e| e.member == current) {
            if seen.insert(edge.role.clone()) {
                frontier.push(edge.role.clone());
            }
        }
    }
    seen
}

/// The membership closure a session effectively holds, and whether it is a superuser.
///
/// Both facts come from the same two catalogs (roles and memberships), so resolving them together
/// in one scan pass avoids the scan-and-throw-away that separate [`effective_roles`] +
/// [`principal`] calls paid.
pub fn resolve_session(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
) -> Result<(BTreeSet<String>, bool), Error> {
    let resolved = ResolvedSession::resolve(engine, txn, user)?;
    Ok((resolved.effective, resolved.superuser))
}

/// A session resolved to the roles it effectively holds and whether it is a superuser — the two
/// facts every privilege question turns on, computed once from the role and membership catalogs.
///
/// Every RBAC question in one statement is about the same session, so resolving it once and asking
/// the resolved session (rather than re-deriving it per question) is what keeps a statement that
/// touches several tables from re-scanning the role catalogs per table. The decision procedure
/// lives here as pure methods over already-fetched catalog rows; the free functions below fetch
/// those rows and delegate, so a caller with nothing cached gets the exact same answer.
#[derive(Debug, Clone)]
pub struct ResolvedSession {
    /// Every role the session effectively holds — itself and everything it inherits, transitively.
    /// Used for privilege and ownership decisions.
    effective: BTreeSet<String>,
    /// Every role the session may `SET ROLE` into — the full membership closure, ignoring
    /// `INHERIT`. A superset of `effective` (a `NOINHERIT` edge is in `assumable` but not
    /// `effective`).
    assumable: BTreeSet<String>,
    /// Whether any of the effective roles carries the superuser attribute.
    superuser: bool,
}

impl ResolvedSession {
    /// Resolve `user` against the role and membership catalogs (one scan of each).
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn resolve(engine: &dyn StorageEngine, txn: TxnId, user: &str) -> Result<Self, Error> {
        if user == crate::BOOTSTRAP_SUPERUSER {
            let mut effective = BTreeSet::new();
            effective.insert(user.to_owned());
            return Ok(Self {
                assumable: effective.clone(),
                effective,
                superuser: true,
            });
        }
        let memberships = all_memberships(engine, txn)?;
        let roles = all_roles(engine, txn)?;
        let effective = effective_roles_from(&memberships, &roles, user);
        let assumable = membership_closure_from(&memberships, user);
        let superuser = roles
            .iter()
            .any(|r| r.superuser && effective.contains(&r.name));
        Ok(Self {
            effective,
            assumable,
            superuser,
        })
    }

    /// Whether the session is a superuser (bypasses every check).
    #[must_use]
    pub const fn is_superuser(&self) -> bool {
        self.superuser
    }

    /// Whether the session effectively holds the role named `owner` — pure ownership, given an
    /// already-resolved owner name. Deliberately does **not** fold in the superuser flag: ownership
    /// and superuser are distinct, and every caller that treats a superuser as able-to-do-anything
    /// checks [`is_superuser`](Self::is_superuser) separately first. Matches the historical
    /// `owns_object`, whose result was exactly `effective.contains(owner)`.
    #[must_use]
    pub fn owns(&self, owner: &str) -> bool {
        self.effective.contains(owner)
    }

    /// The privilege decision over already-fetched inputs: superuser, or owner, or a matching
    /// grant held by an effective role (or `PUBLIC`). `grantable_only` narrows the grant match to
    /// grantable grants (the `may_grant` rule). Pure — no scan — so the same logic serves the
    /// cached catalog path and the free-function path identically.
    #[must_use]
    pub fn decide(
        &self,
        owner: &str,
        grants: &[GrantRecord],
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
        grantable_only: bool,
    ) -> bool {
        if self.owns(owner) {
            return true;
        }
        grants.iter().any(|g| {
            g.kind == kind
                && g.object == object
                && g.privilege == privilege
                && (!grantable_only || g.grantable)
                && match &g.grantee {
                    Grantee::Public => true,
                    Grantee::Role(name) => self.effective.contains(name),
                }
        })
    }

    /// [`decide`](Self::decide), fetching the object's owner and the grant list itself.
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn has_privilege(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if self.superuser {
            return Ok(true);
        }
        let owner = object_owner(engine, txn, kind, object)?;
        let grants = all_grants(engine, txn)?;
        Ok(self.decide(&owner, &grants, kind, object, privilege, false))
    }

    /// Column-scoped decision over pre-fetched grants: a table-wide privilege on the object grants
    /// every column, else a column grant on this specific column to an effective role (or `PUBLIC`).
    fn decide_column(
        &self,
        owner: &str,
        table_grants: &[GrantRecord],
        column_grants: &[ColumnGrantRecord],
        object: &str,
        column: &str,
        privilege: Privilege,
    ) -> bool {
        // A table-wide grant (or ownership / superuser, via `decide`) covers every column.
        if self.decide(
            owner,
            table_grants,
            ObjectKind::Table,
            object,
            privilege,
            false,
        ) {
            return true;
        }
        column_grants.iter().any(|g| {
            g.object == object
                && g.column == column
                && g.privilege == privilege
                && match &g.grantee {
                    Grantee::Public => true,
                    Grantee::Role(name) => self.effective.contains(name),
                }
        })
    }

    /// Whether the session may access `column` of table `object` with `privilege` — a table-wide
    /// grant, ownership, superuser, or a column grant on exactly this column.
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn has_column_privilege(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        object: &str,
        column: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if self.superuser {
            return Ok(true);
        }
        let owner = object_owner(engine, txn, ObjectKind::Table, object)?;
        let table_grants = all_grants(engine, txn)?;
        let column_grants = all_column_grants(engine, txn)?;
        let decision = self.decide_column(
            &owner,
            &table_grants,
            &column_grants,
            object,
            column,
            privilege,
        );
        Ok(decision)
    }

    /// Whether the session may access *at least one* column of table `object` with `privilege` — the
    /// gate a query passes to bring a table into scope when it lacks the table-wide privilege (e.g.
    /// `SELECT count(*)` needs the privilege on the table or on some column).
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn has_any_column_privilege(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if self.superuser {
            return Ok(true);
        }
        let owner = object_owner(engine, txn, ObjectKind::Table, object)?;
        let table_grants = all_grants(engine, txn)?;
        if self.decide(
            &owner,
            &table_grants,
            ObjectKind::Table,
            object,
            privilege,
            false,
        ) {
            return Ok(true);
        }
        let column_grants = all_column_grants(engine, txn)?;
        Ok(column_grants.iter().any(|g| {
            g.object == object
                && g.privilege == privilege
                && match &g.grantee {
                    Grantee::Public => true,
                    Grantee::Role(name) => self.effective.contains(name),
                }
        }))
    }

    /// The grant-option variant of [`has_privilege`](Self::has_privilege) (`may_grant`).
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn may_grant(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        kind: ObjectKind,
        object: &str,
        privilege: Privilege,
    ) -> Result<bool, Error> {
        if self.superuser {
            return Ok(true);
        }
        let owner = object_owner(engine, txn, kind, object)?;
        let grants = all_grants(engine, txn)?;
        Ok(self.decide(&owner, &grants, kind, object, privilege, true))
    }

    /// Whether the session owns `object`, fetching its owner.
    ///
    /// # Errors
    /// Propagates storage errors.
    pub fn owns_object(
        &self,
        engine: &dyn StorageEngine,
        txn: TxnId,
        kind: ObjectKind,
        object: &str,
    ) -> Result<bool, Error> {
        Ok(self.owns(&object_owner(engine, txn, kind, object)?))
    }

    /// Whether the session may assume `role` — a superuser, or a role it effectively holds.
    #[must_use]
    pub fn may_assume_role(&self, role: &str) -> bool {
        // `SET ROLE` eligibility follows the full membership closure, not the inherit-gated
        // `effective` set: a `NOINHERIT` role is a member of its roles and may switch into them,
        // it just does not wield their privileges automatically.
        self.superuser || self.assumable.contains(role)
    }
}

/// Whether granting membership in `role` to `member` would close a cycle — that is, whether
/// `role` is already reachable from `member`'s side of the graph.
///
/// Cycles are refused because a cycle makes "which privileges does this role have" answer itself:
/// every role in the loop would hold every other's privileges, which is never what the operator
/// meant to write.
///
/// # Errors
/// Propagates storage errors.
pub fn would_create_cycle(
    engine: &dyn StorageEngine,
    txn: TxnId,
    role: &str,
    member: &str,
) -> Result<bool, Error> {
    if role == member {
        return Ok(true);
    }
    // The grant adds the edge `role -> member` ("member inherits from role"). That closes a loop
    // exactly when `role` is *already* reachable from `member` along existing edges — so the walk
    // starts at `member` and looks for `role`, following each edge from its role to its member.
    //
    // Walking the other way round would miss the two-role case entirely: with `a -> b` present,
    // `GRANT b TO a` has no edges leading out of `b` to find, yet plainly closes the cycle.
    //
    // The walk ignores the inherit flag: a cycle is structurally wrong whether or not privileges
    // would flow around it.
    let memberships = all_memberships(engine, txn)?;
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(member.to_owned());
    let mut frontier = vec![member.to_owned()];
    while let Some(current) = frontier.pop() {
        for edge in memberships.iter().filter(|e| e.role == current) {
            if edge.member == role {
                return Ok(true);
            }
            if seen.insert(edge.member.clone()) {
                frontier.push(edge.member.clone());
            }
        }
    }
    Ok(false)
}

// === Ownership ============================================================

/// Record `owner` as the owner of an object, replacing any previous record.
///
/// # Errors
/// Propagates storage errors.
pub fn set_owner(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    object: &str,
    owner: &str,
) -> Result<(), Error> {
    clear_owner(engine, txn, kind, object)?;
    let cat = ensure_owner_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(kind.as_str().to_owned()),
        ast::Value::Text(object.to_owned()),
        ast::Value::Text(owner.to_owned()),
    ];
    let bytes = row::encode(&row, &OWNER_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Forget an object's owner — used when the object is dropped.
///
/// # Errors
/// Propagates storage errors.
pub fn clear_owner(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    object: &str,
) -> Result<bool, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, OWNER_CATALOG)? else {
        return Ok(false);
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &OWNER_SCHEMA)?;
        if matches!((row.first(), row.get(1)),
            (Some(ast::Value::Text(k)), Some(ast::Value::Text(o)))
                if k == kind.as_str() && o == object)
        {
            victims.push(tid);
        }
    }
    let present = !victims.is_empty();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(present)
}

/// The owner of an object.
///
/// An object with no recorded owner — one created before this catalog existed — reports the
/// bootstrap superuser. See the module note: unowned must not mean unprotected.
///
/// # Errors
/// Propagates storage errors.
pub fn object_owner(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    object: &str,
) -> Result<String, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, OWNER_CATALOG)? else {
        return Ok(crate::BOOTSTRAP_SUPERUSER.to_owned());
    };
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &OWNER_SCHEMA)?;
        if let [
            ast::Value::Text(k),
            ast::Value::Text(o),
            ast::Value::Text(owner),
        ] = row.as_slice()
            && k == kind.as_str()
            && o == object
        {
            return Ok(owner.clone());
        }
    }
    Ok(crate::BOOTSTRAP_SUPERUSER.to_owned())
}

/// Rename an object's ownership record, keeping the owner. Called from `ALTER ... RENAME` so a
/// renamed object does not silently revert to the unowned (superuser-only) default.
///
/// # Errors
/// Propagates storage errors.
pub fn rename_owned_object(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    from: &str,
    to: &str,
) -> Result<(), Error> {
    let owner = object_owner(engine, txn, kind, from)?;
    if clear_owner(engine, txn, kind, from)? {
        set_owner(engine, txn, kind, to, &owner)?;
    }
    rename_object_grants(engine, txn, kind, from, to)
}

// === Grants ===============================================================

/// Every stored grant.
///
/// # Errors
/// Propagates storage errors. Rows naming an unknown privilege or object kind are skipped: an
/// entry that cannot be understood must not be treated as permission.
pub fn all_grants(engine: &dyn StorageEngine, txn: TxnId) -> Result<Vec<GrantRecord>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PRIVILEGE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PRIVILEGE_SCHEMA)?;
        if let [
            ast::Value::Text(grantee),
            ast::Value::Text(kind),
            ast::Value::Text(object),
            ast::Value::Text(privilege),
            ast::Value::Text(grantor),
            ast::Value::Text(grantable),
        ] = row.as_slice()
            && let Some(kind) = ObjectKind::from_catalog(kind)
            && let Some(privilege) = Privilege::from_catalog(privilege)
        {
            out.push(GrantRecord {
                grantee: Grantee::from_catalog(grantee),
                kind,
                object: object.clone(),
                privilege,
                grantor: grantor.clone(),
                grantable: grantable == TRUE,
            });
        }
    }
    Ok(out)
}

/// Record a grant. Re-granting an existing privilege replaces the row, so `WITH GRANT OPTION` on
/// a second grant upgrades the first rather than leaving two rows that disagree.
///
/// # Errors
/// Propagates storage errors.
pub fn insert_grant(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grant: &GrantRecord,
) -> Result<(), Error> {
    delete_grant(
        engine,
        txn,
        &grant.grantee,
        grant.kind,
        &grant.object,
        Some(grant.privilege),
    )?;
    let cat = ensure_privilege_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(grant.grantee.as_str().to_owned()),
        ast::Value::Text(grant.kind.as_str().to_owned()),
        ast::Value::Text(grant.object.clone()),
        ast::Value::Text(grant.privilege.as_str().to_owned()),
        ast::Value::Text(grant.grantor.clone()),
        ast::Value::Text(flag(grant.grantable)),
    ];
    let bytes = row::encode(&row, &PRIVILEGE_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Remove grants matching `grantee`/`kind`/`object`, and `privilege` when given (`None` removes
/// every privilege on the object). Reports how many rows went.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_grant(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grantee: &Grantee,
    kind: ObjectKind,
    object: &str,
    privilege: Option<Privilege>,
) -> Result<usize, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PRIVILEGE_CATALOG)? else {
        return Ok(0);
    };
    let wanted = grantee.as_str();
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PRIVILEGE_SCHEMA)?;
        if let [
            ast::Value::Text(g),
            ast::Value::Text(k),
            ast::Value::Text(o),
            ast::Value::Text(p),
            ..,
        ] = row.as_slice()
            && g == wanted
            && k == kind.as_str()
            && o == object
            && privilege.is_none_or(|want| p == want.as_str())
        {
            victims.push(tid);
        }
    }
    let count = victims.len();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(count)
}

/// Create the column-privilege catalog if absent.
fn ensure_column_privilege_catalog(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<nusadb_core::TableId, Error> {
    ensure_catalog(
        engine,
        txn,
        COLUMN_PRIVILEGE_CATALOG,
        &[
            "grantee",
            "object",
            "column",
            "privilege",
            "grantor",
            "grantable",
        ],
    )
}

/// Every column-scoped grant in the catalog.
///
/// # Errors
/// Propagates storage errors.
pub fn all_column_grants(
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<ColumnGrantRecord>, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, COLUMN_PRIVILEGE_CATALOG)? else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((_, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &COLUMN_PRIVILEGE_SCHEMA)?;
        if let [
            ast::Value::Text(grantee),
            ast::Value::Text(object),
            ast::Value::Text(column),
            ast::Value::Text(privilege),
            ast::Value::Text(grantor),
            ast::Value::Text(grantable),
        ] = row.as_slice()
            && let Some(privilege) = Privilege::from_catalog(privilege)
        {
            out.push(ColumnGrantRecord {
                grantee: Grantee::from_catalog(grantee),
                object: object.clone(),
                column: column.clone(),
                privilege,
                grantor: grantor.clone(),
                grantable: grantable == TRUE,
            });
        }
    }
    Ok(out)
}

/// Record a column-scoped grant, replacing any existing row for the same
/// (grantee, object, column, privilege) so a re-grant upgrades rather than duplicates.
///
/// # Errors
/// Propagates storage errors.
pub fn insert_column_grant(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grant: &ColumnGrantRecord,
) -> Result<(), Error> {
    delete_column_grant(
        engine,
        txn,
        &grant.grantee,
        &grant.object,
        Some(&grant.column),
        Some(grant.privilege),
    )?;
    let cat = ensure_column_privilege_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(grant.grantee.as_str().to_owned()),
        ast::Value::Text(grant.object.clone()),
        ast::Value::Text(grant.column.clone()),
        ast::Value::Text(grant.privilege.as_str().to_owned()),
        ast::Value::Text(grant.grantor.clone()),
        ast::Value::Text(flag(grant.grantable)),
    ];
    let bytes = row::encode(&row, &COLUMN_PRIVILEGE_SCHEMA)?;
    engine.insert(txn, cat, &bytes)?;
    Ok(())
}

/// Remove column grants matching `grantee`/`object`, and `column`/`privilege` when given (`None`
/// widens the match). Reports how many rows went.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_column_grant(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grantee: &Grantee,
    object: &str,
    column: Option<&str>,
    privilege: Option<Privilege>,
) -> Result<usize, Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, COLUMN_PRIVILEGE_CATALOG)? else {
        return Ok(0);
    };
    let wanted = grantee.as_str();
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &COLUMN_PRIVILEGE_SCHEMA)?;
        if let [
            ast::Value::Text(g),
            ast::Value::Text(o),
            ast::Value::Text(c),
            ast::Value::Text(p),
            ..,
        ] = row.as_slice()
            && g == wanted
            && o == object
            && column.is_none_or(|want| c == want)
            && privilege.is_none_or(|want| p == want.as_str())
        {
            victims.push(tid);
        }
    }
    let count = victims.len();
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(count)
}

/// Drop the grant option on a grant while keeping the privilege — `REVOKE GRANT OPTION FOR`.
///
/// # Errors
/// Propagates storage errors.
pub fn downgrade_grant_option(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grantee: &Grantee,
    kind: ObjectKind,
    object: &str,
    privilege: Privilege,
) -> Result<bool, Error> {
    let existing = all_grants(engine, txn)?.into_iter().find(|g| {
        &g.grantee == grantee && g.kind == kind && g.object == object && g.privilege == privilege
    });
    let Some(mut record) = existing else {
        return Ok(false);
    };
    if !record.grantable {
        return Ok(false);
    }
    record.grantable = false;
    insert_grant(engine, txn, &record)?;
    Ok(true)
}

/// Remove every grant on an object — used when the object is dropped, so a later object reusing
/// the name does not inherit the old one's permissions.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_grants_on(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    object: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PRIVILEGE_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PRIVILEGE_SCHEMA)?;
        if matches!((row.get(1), row.get(2)),
            (Some(ast::Value::Text(k)), Some(ast::Value::Text(o)))
                if k == kind.as_str() && o == object)
        {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// Move an object's grants to a new name, so `ALTER ... RENAME` does not quietly drop permissions.
///
/// # Errors
/// Propagates storage errors.
fn rename_object_grants(
    engine: &dyn StorageEngine,
    txn: TxnId,
    kind: ObjectKind,
    from: &str,
    to: &str,
) -> Result<(), Error> {
    let moving: Vec<GrantRecord> = all_grants(engine, txn)?
        .into_iter()
        .filter(|g| g.kind == kind && g.object == from)
        .collect();
    if moving.is_empty() {
        return Ok(());
    }
    delete_grants_on(engine, txn, kind, from)?;
    for mut record in moving {
        to.clone_into(&mut record.object);
        insert_grant(engine, txn, &record)?;
    }
    Ok(())
}

/// Remove every grant held by `name`, and every grant it made — used when a role is dropped.
///
/// Grants the dropped role *made* go too: leaving them would credit a privilege to a grantor that
/// no longer exists, and `REVOKE ... CASCADE` would then have no way to reach them.
///
/// # Errors
/// Propagates storage errors.
pub fn delete_grants_of_role(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
) -> Result<(), Error> {
    let Some(cat) = engine.lookup_table_as_of(txn, PRIVILEGE_CATALOG)? else {
        return Ok(());
    };
    let mut victims = Vec::new();
    let mut scan = engine.scan(txn, cat.id)?;
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &PRIVILEGE_SCHEMA)?;
        if matches!((row.first(), row.get(4)),
            (Some(ast::Value::Text(g)), Some(ast::Value::Text(grantor)))
                if g == name || grantor == name)
        {
            victims.push(tid);
        }
    }
    for tid in victims {
        engine.delete(txn, cat.id, tid)?;
    }
    Ok(())
}

/// The grants `grantee` made on an object using its grant option — what `REVOKE ... CASCADE`
/// removes and what `RESTRICT` refuses over.
///
/// # Errors
/// Propagates storage errors.
pub fn dependent_grants(
    engine: &dyn StorageEngine,
    txn: TxnId,
    grantor: &Grantee,
    kind: ObjectKind,
    object: &str,
    privilege: Privilege,
) -> Result<Vec<GrantRecord>, Error> {
    let grantor = grantor.as_str();
    Ok(all_grants(engine, txn)?
        .into_iter()
        .filter(|g| {
            g.grantor == grantor
                && g.kind == kind
                && g.object == object
                && g.privilege == privilege
                && g.grantee.as_str() != grantor
        })
        .collect())
}

// === The decision =========================================================

/// A session's identity, as far as access control is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    /// The role the session runs as — the login role, or whatever `SET ROLE` switched to.
    pub role: String,
    /// Whether that role bypasses every check.
    pub superuser: bool,
}

/// Resolve who a session is: its role and whether that role is a superuser.
///
/// The bootstrap superuser is a superuser even without a catalog row, so a fresh installation is
/// administrable before any role exists.
///
/// # Errors
/// Propagates storage errors.
pub fn principal(engine: &dyn StorageEngine, txn: TxnId, user: &str) -> Result<Principal, Error> {
    // Whether the session is a superuser is derived from the same role/membership catalogs as its
    // effective roles, so [`resolve_session`] computes both in one scan pass.
    let (_effective, superuser) = resolve_session(engine, txn, user)?;
    Ok(Principal {
        role: user.to_owned(),
        superuser,
    })
}

/// Whether `user` may exercise `privilege` on an object.
///
/// The three ways to yes are listed in the module documentation. This is the single funnel every
/// enforcement site goes through, so there is one place to audit rather than one per statement.
///
/// # Errors
/// Propagates storage errors.
pub fn has_privilege(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    kind: ObjectKind,
    object: &str,
    privilege: Privilege,
) -> Result<bool, Error> {
    ResolvedSession::resolve(engine, txn, user)?.has_privilege(engine, txn, kind, object, privilege)
}

/// Whether `user` may access `column` of table `object` with `privilege` (table-wide grant, column
/// grant, ownership, or superuser).
///
/// # Errors
/// Propagates storage errors.
pub fn has_column_privilege(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    object: &str,
    column: &str,
    privilege: Privilege,
) -> Result<bool, Error> {
    ResolvedSession::resolve(engine, txn, user)?
        .has_column_privilege(engine, txn, object, column, privilege)
}

/// Whether `user` may access at least one column of table `object` with `privilege` — the
/// bring-into-scope gate for a column-restricted table.
///
/// # Errors
/// Propagates storage errors.
pub fn has_any_column_privilege(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    object: &str,
    privilege: Privilege,
) -> Result<bool, Error> {
    ResolvedSession::resolve(engine, txn, user)?
        .has_any_column_privilege(engine, txn, object, privilege)
}

/// Whether `user` may *pass on* `privilege` — needed to run a `GRANT`.
///
/// An owner may always grant on what it owns; anyone else needs a grant that carries the grant
/// option. This is deliberately stricter than [`has_privilege`]: holding a privilege is not the
/// same as being allowed to hand it out.
///
/// # Errors
/// Propagates storage errors.
pub fn may_grant(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    kind: ObjectKind,
    object: &str,
    privilege: Privilege,
) -> Result<bool, Error> {
    ResolvedSession::resolve(engine, txn, user)?.may_grant(engine, txn, kind, object, privilege)
}

/// Whether `user` effectively owns an object — directly, or through a role it inherits.
///
/// # Errors
/// Propagates storage errors.
pub fn owns_object(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    kind: ObjectKind,
    object: &str,
) -> Result<bool, Error> {
    ResolvedSession::resolve(engine, txn, user)?.owns_object(engine, txn, kind, object)
}

/// Whether `user` may `SET ROLE` into `role`.
///
/// True for a superuser, or for a member of `role` (directly or transitively), **regardless of
/// `INHERIT`**. `SET ROLE` is exactly how a `NOINHERIT` role uses a membership it does not wield
/// automatically, so eligibility follows the full membership closure, not the inherit-gated
/// effective set.
///
/// # Errors
/// Propagates storage errors.
pub fn may_assume_role(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    role: &str,
) -> Result<bool, Error> {
    Ok(ResolvedSession::resolve(engine, txn, user)?.may_assume_role(role))
}

/// Whether `user` may create roles — a superuser, or a role holding `CREATEROLE`.
///
/// # Errors
/// Propagates storage errors.
pub fn may_create_role(engine: &dyn StorageEngine, txn: TxnId, user: &str) -> Result<bool, Error> {
    let (effective, superuser) = resolve_session(engine, txn, user)?;
    if superuser {
        return Ok(true);
    }
    Ok(all_roles(engine, txn)?
        .into_iter()
        .any(|r| r.create_role && effective.contains(&r.name)))
}

/// Whether `user` may administer `role` — grant membership in it, or drop it.
///
/// True for a superuser, for a role with the `CREATEROLE` attribute, and for a member holding the
/// admin option on that membership.
///
/// # Errors
/// Propagates storage errors.
pub fn may_administer_role(
    engine: &dyn StorageEngine,
    txn: TxnId,
    user: &str,
    role: &str,
) -> Result<bool, Error> {
    let (effective, superuser) = resolve_session(engine, txn, user)?;
    if superuser {
        return Ok(true);
    }
    let roles = all_roles(engine, txn)?;
    // Granting membership in a SUPERUSER role hands out superuser, exactly what minting one is
    // (`analyze_create_role` already blocks that for a non-superuser). Membership is the second
    // door to the same place, so a non-superuser may never administer a superuser role — even
    // holding CREATEROLE or ADMIN on it.
    if roles.iter().any(|r| r.name == role && r.superuser) {
        return Ok(false);
    }
    if roles
        .iter()
        .any(|r| r.create_role && effective.contains(&r.name))
    {
        return Ok(true);
    }
    Ok(all_memberships(engine, txn)?
        .into_iter()
        .any(|e| e.role == role && e.admin && effective.contains(&e.member)))
}

/// The error a denied check raises.
///
/// Centralised so every denial reads the same way and none of them leaks whether the object
/// exists — a message that distinguished "no such table" from "not allowed" would be a probe for
/// the schema.
#[must_use]
pub fn denied(kind: ObjectKind, object: &str, privilege: Privilege) -> Error {
    Error::PermissionDenied(format!(
        "{privilege} on {noun} `{object}`",
        privilege = privilege.as_str(),
        noun = kind.noun(),
    ))
}
