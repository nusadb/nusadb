//! Shared test helpers used across NusaDB crates (not published).
//!
//! Keep this crate small: helpers that genuinely cross crate boundaries. Single-crate
//! helpers stay in that crate's own `#[cfg(test)]` module.

// This crate is test scaffolding (dev-dependency only); panicking on setup failure is the
// intended behavior, so the production-grade `expect`/panic-doc lints don't apply here.
#![allow(missing_docs, clippy::expect_used, clippy::missing_panics_doc)]

/// Create a temporary directory for tests that need on-disk state.
#[must_use]
pub fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("create temp dir")
}

/// Answer every access question of a catalog test double the way a superuser's session would.
///
/// The `Catalog` trait of `nusadb-sql` has no defaults for these questions, so that no catalog can
/// answer one by omission. A double with no notion of roles, which is most of them, expands this
/// in its `impl Catalog` block. Name the crate that holds the trait: `nusadb_sql` from outside it,
/// `crate` from inside. Living here keeps it out of reach of production code, since this crate is
/// only ever a dev-dependency.
#[macro_export]
macro_rules! catalog_answers_as_superuser {
    ($sql:tt) => {
        fn catalog_for_stored_sql(&self) -> &dyn $sql::Catalog {
            self
        }
        fn is_superuser(&self) -> bool {
            true
        }
        fn current_user(&self) -> ::std::string::String {
            "nusadb-root".to_owned()
        }
        fn rls_enabled(
            &self,
            _schema: &str,
            _name: &str,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(false)
        }
        fn lookup_policies(
            &self,
            _schema: &str,
            _name: &str,
        ) -> ::core::result::Result<::std::vec::Vec<$sql::PolicyDef>, $sql::Error> {
            ::core::result::Result::Ok(::std::vec::Vec::new())
        }
        fn has_privilege(
            &self,
            _kind: $sql::ast::ObjectKind,
            _object: &str,
            _privilege: $sql::ast::Privilege,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn has_column_privilege(
            &self,
            _object: &str,
            _column: &str,
            _privilege: $sql::ast::Privilege,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn has_any_column_privilege(
            &self,
            _object: &str,
            _privilege: $sql::ast::Privilege,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn may_grant_object(
            &self,
            _kind: $sql::ast::ObjectKind,
            _object: &str,
            _privilege: $sql::ast::Privilege,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn may_create_role(&self) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn may_create_database(&self) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn may_administer_role(&self, _role: &str) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn may_assume_role(&self, _role: &str) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn role_exists(&self, _name: &str) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
        fn owns_object(
            &self,
            _kind: $sql::ast::ObjectKind,
            _object: &str,
        ) -> ::core::result::Result<bool, $sql::Error> {
            ::core::result::Result::Ok(true)
        }
    };
}
