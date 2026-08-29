//! Access-control statements: roles, privilege grants, and the session's active role.
//!
//! These are the data-control half of the SQL surface. A privilege is always a
//! ([`Privilege`], [`GrantObject`]) pair — the verb and the thing it applies to — held by a
//! *grantee*, which is either a named role or `PUBLIC`. Roles double as users: a role with
//! `LOGIN` may open a connection, and a role granted membership in another inherits its
//! privileges. That is one concept where some systems use two, and it is what lets
//! `GRANT reader TO app` mean what it looks like it means.
//!
//! The parser converts `sqlparser`'s much wider vendor-flavoured grammar down to exactly these
//! shapes and rejects the rest, so everything represented here is something the analyzer and
//! executor genuinely enforce.

/// A single privilege verb.
///
/// Which verbs are meaningful depends on the object: a table takes `SELECT`/`INSERT`/… , a schema
/// takes `USAGE`/`CREATE`, and so on. [`Privilege::applies_to`] is the authority on the pairing,
/// and the analyzer rejects a mismatched pair rather than storing a grant that could never be
/// consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Privilege {
    /// Read rows (table, view) or read a sequence's current value.
    Select,
    /// Add rows.
    Insert,
    /// Modify existing rows, or advance a sequence.
    Update,
    /// Remove rows.
    Delete,
    /// Empty a table by the fast path rather than row-by-row.
    Truncate,
    /// Create a foreign key that points at this table.
    References,
    /// Attach a trigger to this table.
    Trigger,
    /// Look inside a schema, or draw values from a sequence.
    Usage,
    /// Create objects inside a schema or database.
    Create,
    /// Open a connection to a database.
    Connect,
    /// Create temporary objects in a database.
    Temporary,
}

impl Privilege {
    /// The canonical keyword, used for catalog persistence, `information_schema`, and diagnostics.
    ///
    /// This spelling is written to disk, so changing one is a data-format change, not a cosmetic
    /// one.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Select => "SELECT",
            Self::Insert => "INSERT",
            Self::Update => "UPDATE",
            Self::Delete => "DELETE",
            Self::Truncate => "TRUNCATE",
            Self::References => "REFERENCES",
            Self::Trigger => "TRIGGER",
            Self::Usage => "USAGE",
            Self::Create => "CREATE",
            Self::Connect => "CONNECT",
            Self::Temporary => "TEMPORARY",
        }
    }

    /// Parse a canonical keyword back into a privilege — the read side of [`Self::as_str`], used
    /// when loading the privilege catalog. Unknown text yields `None`; a caller reading persisted
    /// rows treats that as a corrupt entry and skips it rather than guessing (an unreadable grant
    /// must never widen access).
    #[must_use]
    pub fn from_catalog(text: &str) -> Option<Self> {
        Some(match text {
            "SELECT" => Self::Select,
            "INSERT" => Self::Insert,
            "UPDATE" => Self::Update,
            "DELETE" => Self::Delete,
            "TRUNCATE" => Self::Truncate,
            "REFERENCES" => Self::References,
            "TRIGGER" => Self::Trigger,
            "USAGE" => Self::Usage,
            "CREATE" => Self::Create,
            "CONNECT" => Self::Connect,
            "TEMPORARY" => Self::Temporary,
            _ => return None,
        })
    }

    /// Whether this privilege is meaningful on `kind` — the pairing `GRANT ALL` expands over and
    /// an explicit grant is checked against.
    #[must_use]
    pub const fn applies_to(self, kind: ObjectKind) -> bool {
        match kind {
            ObjectKind::Table => matches!(
                self,
                Self::Select
                    | Self::Insert
                    | Self::Update
                    | Self::Delete
                    | Self::Truncate
                    | Self::References
                    | Self::Trigger
            ),
            ObjectKind::Schema => matches!(self, Self::Usage | Self::Create),
            ObjectKind::Database => matches!(self, Self::Connect | Self::Create | Self::Temporary),
            ObjectKind::Sequence => matches!(self, Self::Usage | Self::Select | Self::Update),
        }
    }

    /// Every privilege meaningful on `kind`, in canonical order — what `GRANT ALL PRIVILEGES`
    /// expands to and what an owner implicitly holds.
    #[must_use]
    pub fn all_for(kind: ObjectKind) -> Vec<Self> {
        const EVERY: [Privilege; 11] = [
            Privilege::Select,
            Privilege::Insert,
            Privilege::Update,
            Privilege::Delete,
            Privilege::Truncate,
            Privilege::References,
            Privilege::Trigger,
            Privilege::Usage,
            Privilege::Create,
            Privilege::Connect,
            Privilege::Temporary,
        ];
        EVERY.into_iter().filter(|p| p.applies_to(kind)).collect()
    }
}

/// The kind of object a privilege is held on.
///
/// Views and materialized views are governed as [`ObjectKind::Table`]: they are read and written
/// through the same statements, so splitting them would only create a way to hold `SELECT` on a
/// view but not on the thing the check actually runs against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectKind {
    /// A base table, view, or materialized view.
    Table,
    /// A schema (namespace within a database).
    Schema,
    /// A database.
    Database,
    /// A sequence.
    Sequence,
}

impl ObjectKind {
    /// The canonical keyword, used for catalog persistence and diagnostics. Persisted — see the
    /// note on [`Privilege::as_str`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Table => "TABLE",
            Self::Schema => "SCHEMA",
            Self::Database => "DATABASE",
            Self::Sequence => "SEQUENCE",
        }
    }

    /// Parse a canonical keyword back into an object kind. Unknown text yields `None` and the
    /// caller skips the row — an unreadable grant must never widen access.
    #[must_use]
    pub fn from_catalog(text: &str) -> Option<Self> {
        Some(match text {
            "TABLE" => Self::Table,
            "SCHEMA" => Self::Schema,
            "DATABASE" => Self::Database,
            "SEQUENCE" => Self::Sequence,
            _ => return None,
        })
    }

    /// The word used in a `permission denied for <word> x` diagnostic.
    #[must_use]
    pub const fn noun(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Schema => "schema",
            Self::Database => "database",
            Self::Sequence => "sequence",
        }
    }
}

/// A concrete object a grant names, already resolved to its kind and canonical name.
///
/// The name is stored exactly as the privilege catalog keys it: schema-qualified for a table or
/// sequence (`app.orders`), bare for a schema or database. Resolution happens in the analyzer,
/// which has the `search_path`; by the time a [`GrantObject`] exists the name is unambiguous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantObject {
    /// What kind of thing this is.
    pub kind: ObjectKind,
    /// The canonical name, schema-qualified where the kind calls for it.
    pub name: String,
}

/// The set of objects a `GRANT`/`REVOKE` names, before the analyzer resolves it.
///
/// `ALL TABLES IN SCHEMA s` is expanded at execution time against the tables that exist *then* —
/// matching the SQL-standard reading that it is shorthand for naming them, not a standing rule for
/// tables created later. A `FUTURE` grant would be that standing rule, and it is deliberately not
/// in surface: honouring it needs a hook on every create path, and a half-honoured one grants less
/// than it promises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantTargets {
    /// `ON [TABLE] a, b` — named tables or views.
    Tables(Vec<String>),
    /// `ON ALL TABLES IN SCHEMA s [, ...]` — expanded when the statement runs.
    AllTablesInSchemas(Vec<String>),
    /// `ON SCHEMA a, b`.
    Schemas(Vec<String>),
    /// `ON DATABASE a, b`.
    Databases(Vec<String>),
    /// `ON SEQUENCE a, b`.
    Sequences(Vec<String>),
    /// `ON ALL SEQUENCES IN SCHEMA s [, ...]` — expanded when the statement runs.
    AllSequencesInSchemas(Vec<String>),
}

/// Who a privilege is granted to or revoked from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grantee {
    /// A named role (which may be a login user — roles and users are one concept here).
    Role(String),
    /// `PUBLIC` — every role, including ones created later.
    Public,
}

impl Grantee {
    /// The catalog spelling. `PUBLIC` is stored as the reserved name `public`, which
    /// [`crate::rbac::validate_role_name`] refuses for a real role, so the two can never collide.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Role(name) => name,
            Self::Public => PUBLIC_GRANTEE,
        }
    }

    /// Read a grantee back from its catalog spelling.
    #[must_use]
    pub fn from_catalog(text: &str) -> Self {
        if text == PUBLIC_GRANTEE {
            Self::Public
        } else {
            Self::Role(text.to_owned())
        }
    }
}

/// The reserved catalog name standing for "every role".
pub const PUBLIC_GRANTEE: &str = "public";

/// A privilege scoped to specific columns of a table: `SELECT (a, b)` / `UPDATE (c)`.
///
/// Only `SELECT`, `INSERT`, `UPDATE`, and `REFERENCES` may be column-scoped (the privileges that act
/// on individual columns); the parser rejects a column list on any other privilege.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnPrivilege {
    /// The privilege held on the listed columns.
    pub privilege: Privilege,
    /// The columns it applies to; always non-empty.
    pub columns: Vec<String>,
}

/// `GRANT privileges ON objects TO grantees [WITH GRANT OPTION]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grant {
    /// The table-wide privileges being granted; `None` means `ALL PRIVILEGES`, expanded per object
    /// kind. `Some(empty)` when the statement grants only column-scoped privileges.
    pub privileges: Option<Vec<Privilege>>,
    /// Column-scoped privileges (`GRANT SELECT (a, b) ...`); empty for a purely table-wide grant.
    pub column_privileges: Vec<ColumnPrivilege>,
    /// The objects the privileges apply to.
    pub targets: GrantTargets,
    /// Who receives them.
    pub grantees: Vec<Grantee>,
    /// Whether the grantees may pass the privilege on.
    pub with_grant_option: bool,
}

/// `REVOKE [GRANT OPTION FOR] privileges ON objects FROM grantees [CASCADE | RESTRICT]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Revoke {
    /// The table-wide privileges being revoked; `None` means `ALL PRIVILEGES`.
    pub privileges: Option<Vec<Privilege>>,
    /// Column-scoped privileges to revoke (`REVOKE SELECT (a) ...`); empty for a table-wide revoke.
    pub column_privileges: Vec<ColumnPrivilege>,
    /// The objects to revoke on.
    pub targets: GrantTargets,
    /// Who loses them.
    pub grantees: Vec<Grantee>,
    /// `GRANT OPTION FOR` — revoke only the ability to pass the privilege on, keeping the
    /// privilege itself.
    pub grant_option_for: bool,
    /// `CASCADE` also revokes the grants the grantee made using its grant option; `RESTRICT` (the
    /// default) refuses when such dependent grants exist.
    pub cascade: bool,
}

/// `GRANT role[, ...] TO member[, ...] [WITH ADMIN OPTION]` — role membership rather than an
/// object privilege. Distinguished from [`Grant`] at parse time by the absence of `ON`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantRole {
    /// The roles whose privileges the members gain.
    pub roles: Vec<String>,
    /// The roles receiving membership.
    pub members: Vec<String>,
    /// Whether the members may themselves grant this membership on.
    pub with_admin_option: bool,
}

/// `REVOKE [ADMIN OPTION FOR] role[, ...] FROM member[, ...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevokeRole {
    /// The roles to remove membership in.
    pub roles: Vec<String>,
    /// The roles losing membership.
    pub members: Vec<String>,
    /// Revoke only the admin option, keeping the membership.
    pub admin_option_for: bool,
}

/// The attributes a role carries. `None` on a field means "not mentioned": on `CREATE ROLE` it
/// takes the default, on `ALTER ROLE` it leaves the stored value alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoleOptions {
    /// May bypass every privilege check. Only a superuser may set this.
    pub superuser: Option<bool>,
    /// May open a connection.
    pub login: Option<bool>,
    /// May create databases.
    pub create_db: Option<bool>,
    /// May create, alter, and drop roles.
    pub create_role: Option<bool>,
    /// Whether membership in another role automatically confers its privileges. `false` means the
    /// role must `SET ROLE` to use them.
    pub inherit: Option<bool>,
}

/// `CREATE ROLE | USER name [IF NOT EXISTS] [options]`.
///
/// `CREATE USER` is the same statement with `LOGIN` defaulted on — the only difference the two
/// spellings carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateRole {
    /// The role name.
    pub name: String,
    /// Whether `IF NOT EXISTS` was specified.
    pub if_not_exists: bool,
    /// The attributes given.
    pub options: RoleOptions,
    /// `IN ROLE r` — memberships the new role is immediately given.
    pub in_role: Vec<String>,
}

/// `DROP ROLE | USER [IF EXISTS] name[, ...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropRole {
    /// The roles to drop.
    pub names: Vec<String>,
    /// Whether `IF EXISTS` was specified.
    pub if_exists: bool,
}

/// `ALTER ROLE name [options]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterRole {
    /// The role to alter.
    pub name: String,
    /// The attributes to change; unmentioned ones keep their stored value.
    pub options: RoleOptions,
}

/// `SET ROLE name` / `SET ROLE NONE` / `RESET ROLE`.
///
/// `None` restores the session's login role. Switching is only permitted to a role the session
/// user is a member of (or, for a superuser, any role) — otherwise `SET ROLE` would itself be a
/// privilege-escalation path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetRole {
    /// The role to assume, or `None` for `NONE`/`RESET`.
    pub role: Option<String>,
}
