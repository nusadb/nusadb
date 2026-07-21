//! JSON column array: [`JsonArray`] ([`ColumnType::Json`]).
//!
//! A JSON value is stored as canonical (sorted-key) text, exactly like
//! [`ast::Value::Json`](crate::ast::Value::Json). [`JsonArray`] therefore reuses the
//! variable-length [`StringArray`](super::StringArray) storage and only overrides the
//! reported [`ColumnType`]. Canonicalization is the caller's responsibility (see the
//! [`json`](crate::json) module); this array stores the text as given.

use std::any::Any;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::bytes::StringArray;

/// A contiguous, null-aware column of canonical JSON text ([`ColumnType::Json`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonArray(StringArray);

impl JsonArray {
    /// Build an array from canonical JSON strings with no nulls.
    #[must_use]
    pub fn from_values(values: Vec<String>) -> Self {
        Self(StringArray::from_values(values))
    }

    /// Build an array from optional JSON strings; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<String>>) -> Self {
        Self(StringArray::from_options(items))
    }

    /// The JSON text at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&str> {
        self.0.get(index)
    }
}

impl Array for JsonArray {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Json
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn is_null(&self, index: usize) -> bool {
        self.0.is_null(index)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::batch::ArrayRef;

    #[test]
    fn reports_json_and_roundtrips() {
        let a = JsonArray::from_values(vec![r#"{"a":1}"#.to_owned(), "[1,2]".to_owned()]);
        assert_eq!(a.len(), 2);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.data_type(), ColumnType::Json);
        assert_eq!(a.get(0), Some(r#"{"a":1}"#));
        assert_eq!(a.get(1), Some("[1,2]"));
        assert!(a.get(2).is_none());
    }

    #[test]
    fn tracks_nulls() {
        let a = JsonArray::from_options(vec![Some("null".to_owned()), None]);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.get(0), Some("null")); // JSON null literal is not a SQL NULL
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: ArrayRef = Arc::new(JsonArray::from_values(vec!["true".to_owned()]));
        assert_eq!(arr.data_type(), ColumnType::Json);
        let back = arr
            .as_any()
            .downcast_ref::<JsonArray>()
            .expect("downcast to JsonArray");
        assert_eq!(back.get(0), Some("true"));
    }
}
