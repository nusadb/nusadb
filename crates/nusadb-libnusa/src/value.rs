//! Parameter encoding and typed row decoding.
//!
//! Parameters are sent in the wire **text** format (see `docs/wire-protocol.md` §10.4): each value
//! is its textual rendering, and SQL `NULL` is the absent marker. Result fields likewise arrive in
//! text format by default; the typed getters on [`Row`] parse them on demand.

use std::fmt::Display;
use std::sync::Arc;

use crate::error::{Error, Result};

/// A bound query parameter, carried as text (`docs/wire-protocol.md` §10.4).
///
/// Build one from a Rust value via [`From`] (`42_i64.into()`, `"alice".into()`,
/// `Option::<i64>::None.into()` for `NULL`) or the explicit constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param(Option<String>);

impl Param {
    /// A SQL `NULL` parameter.
    #[must_use]
    pub const fn null() -> Self {
        Self(None)
    }

    /// A parameter from any [`Display`] value (its text rendering is sent verbatim).
    pub fn text<T: Display>(value: T) -> Self {
        Self(Some(value.to_string()))
    }

    /// The wire form: `Some(text-bytes)` for a value, `None` for `NULL`.
    #[must_use]
    pub(crate) fn to_wire(&self) -> Option<Vec<u8>> {
        self.0.as_ref().map(|s| s.clone().into_bytes())
    }
}

impl From<&str> for Param {
    fn from(v: &str) -> Self {
        Self(Some(v.to_owned()))
    }
}

impl From<String> for Param {
    fn from(v: String) -> Self {
        Self(Some(v))
    }
}

macro_rules! param_from_display {
    ($($t:ty),*) => {$(
        impl From<$t> for Param {
            fn from(v: $t) -> Self {
                Self(Some(v.to_string()))
            }
        }
    )*};
}
param_from_display!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);

impl<T> From<Option<T>> for Param
where
    Self: From<T>,
{
    fn from(v: Option<T>) -> Self {
        v.map_or_else(Self::null, Self::from)
    }
}

/// One row of a result set: the shared column header plus this row's field bytes (text format, or
/// `None` for SQL `NULL`).
#[derive(Debug, Clone)]
pub struct Row {
    columns: Arc<[String]>,
    values: Vec<Option<Vec<u8>>>,
}

impl Row {
    pub(crate) const fn new(columns: Arc<[String]>, values: Vec<Option<Vec<u8>>>) -> Self {
        Self { columns, values }
    }

    /// The column names, in order.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// The number of fields in the row.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the row has no fields.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The zero-based index of column `name`, if present.
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Whether the field at `idx` is SQL `NULL` (also true when `idx` is out of range).
    #[must_use]
    pub fn is_null(&self, idx: usize) -> bool {
        self.values.get(idx).is_none_or(Option::is_none)
    }

    /// The raw field bytes at `idx` (`None` for `NULL` or an out-of-range index).
    #[must_use]
    pub fn get_bytes(&self, idx: usize) -> Option<&[u8]> {
        self.values.get(idx).and_then(|v| v.as_deref())
    }

    /// The field at `idx` as `&str` (`None` for `NULL`).
    ///
    /// # Errors
    /// [`Error::Decode`] if `idx` is out of range or the bytes are not valid UTF-8.
    pub fn get_str(&self, idx: usize) -> Result<Option<&str>> {
        match self.values.get(idx) {
            None => Err(Error::Decode(format!("column index {idx} out of range"))),
            Some(None) => Ok(None),
            Some(Some(bytes)) => std::str::from_utf8(bytes)
                .map(Some)
                .map_err(|_| Error::Decode(format!("column {idx} is not valid UTF-8"))),
        }
    }

    /// The field at `idx` as an owned `String` (`None` for `NULL`).
    ///
    /// # Errors
    /// [`Error::Decode`] as for [`get_str`](Self::get_str).
    pub fn get_string(&self, idx: usize) -> Result<Option<String>> {
        Ok(self.get_str(idx)?.map(ToOwned::to_owned))
    }

    /// The field at `idx` parsed as `i64` (`None` for `NULL`).
    ///
    /// # Errors
    /// [`Error::Decode`] if the index is out of range, the bytes are not UTF-8, or the text is not
    /// a valid integer.
    pub fn get_i64(&self, idx: usize) -> Result<Option<i64>> {
        self.parse_field(idx, "i64")
    }

    /// The field at `idx` parsed as `f64` (`None` for `NULL`).
    ///
    /// # Errors
    /// [`Error::Decode`] as for [`get_i64`](Self::get_i64) but for a float.
    pub fn get_f64(&self, idx: usize) -> Result<Option<f64>> {
        self.parse_field(idx, "f64")
    }

    /// The field at `idx` as `bool` (`None` for `NULL`). Accepts the text forms `true`/`t` and
    /// `false`/`f` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Decode`] if the index is out of range or the text is not a recognised boolean.
    pub fn get_bool(&self, idx: usize) -> Result<Option<bool>> {
        let Some(text) = self.get_str(idx)? else {
            return Ok(None);
        };
        match text.to_ascii_lowercase().as_str() {
            "true" | "t" => Ok(Some(true)),
            "false" | "f" => Ok(Some(false)),
            other => Err(Error::Decode(format!(
                "column {idx}: {other:?} is not a bool"
            ))),
        }
    }

    /// Parse the field at `idx` with `str::parse`, tagging errors with `ty` for the message.
    fn parse_field<T>(&self, idx: usize, ty: &str) -> Result<Option<T>>
    where
        T: std::str::FromStr,
    {
        let Some(text) = self.get_str(idx)? else {
            return Ok(None);
        };
        text.parse::<T>()
            .map(Some)
            .map_err(|_| Error::Decode(format!("column {idx}: {text:?} is not a valid {ty}")))
    }
}

/// The collected result of a statement: the column header, the rows, and the `CommandComplete` tag.
#[derive(Debug, Clone, Default)]
pub struct QueryResult {
    /// Output column names (empty for a non-row statement).
    pub columns: Arc<[String]>,
    /// Per-column type name, parallel to [`columns`](Self::columns) (protocol 1.1). Each
    /// entry is a canonical type name such as `"INT"` / `"TEXT"` / `"TIMESTAMP"`, or `None` when the
    /// server answered with the untyped (1.0) row description.
    pub column_types: Arc<[Option<String>]>,
    /// The result rows (empty for a non-row statement).
    pub rows: Vec<Row>,
    /// The `CommandComplete` tag (e.g. `SELECT 3`, `INSERT 1`), if the statement completed.
    pub tag: Option<String>,
}

impl QueryResult {
    /// The `CommandComplete` tag text, if any.
    #[must_use]
    pub fn command_tag(&self) -> Option<&str> {
        self.tag.as_deref()
    }

    /// The affected-row count parsed from the trailing number of the command tag
    /// (`INSERT 3` → `3`, `SELECT 2` → `2`), or `None` if the tag has no count.
    #[must_use]
    pub fn affected(&self) -> Option<u64> {
        self.tag
            .as_deref()?
            .rsplit(' ')
            .next()
            .and_then(|n| n.parse().ok())
    }
}
