//! UUID column array: [`UuidArray`] ([`ColumnType::Uuid`]).
//!
//! A UUID is a fixed 16-byte value ([`ast::Value::Uuid`](crate::ast::Value::Uuid)), so
//! the array stores `[u8; 16]` elements densely.
//!
//! Null tracking uses the shared [`validity`](crate::batch::validity) helpers; the
//! bit-packed validity buffer is a later refinement.

use std::any::Any;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::validity::{NullBuffer, is_null_at, null_count, split_options};

/// The raw 16-byte form of a UUID column element.
pub type Uuid = [u8; 16];

/// A contiguous, null-aware column of 16-byte UUID values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UuidArray {
    values: Vec<Uuid>,
    validity: Option<NullBuffer>,
}

impl UuidArray {
    /// Build an array from UUIDs with no nulls.
    #[must_use]
    pub const fn from_values(values: Vec<Uuid>) -> Self {
        Self {
            values,
            validity: None,
        }
    }

    /// Build an array from optional UUIDs; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<Uuid>>) -> Self {
        let (values, validity) = split_options(items, [0u8; 16]);
        Self { values, validity }
    }

    /// The UUID at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Uuid> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        self.values.get(index).copied()
    }
}

impl Array for UuidArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Uuid
    }

    fn null_count(&self) -> usize {
        null_count(self.validity.as_ref())
    }

    fn is_null(&self, index: usize) -> bool {
        is_null_at(self.validity.as_ref(), index)
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

    const A: Uuid = [1; 16];
    const B: Uuid = [2; 16];

    #[test]
    fn reports_uuid_and_roundtrips() {
        let arr = UuidArray::from_values(vec![A, B]);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.data_type(), ColumnType::Uuid);
        assert_eq!(arr.get(0), Some(A));
        assert_eq!(arr.get(1), Some(B));
        assert!(arr.get(2).is_none());
    }

    #[test]
    fn tracks_nulls() {
        let arr = UuidArray::from_options(vec![Some(A), None]);
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.get(0), Some(A));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: ArrayRef = Arc::new(UuidArray::from_values(vec![A]));
        assert_eq!(arr.data_type(), ColumnType::Uuid);
        let back = arr
            .as_any()
            .downcast_ref::<UuidArray>()
            .expect("downcast to UuidArray");
        assert_eq!(back.get(0), Some(A));
    }
}
