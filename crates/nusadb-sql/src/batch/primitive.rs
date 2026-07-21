//! Fixed-width primitive column arrays: [`Int64Array`], [`Float64Array`],
//! and [`BooleanArray`].
//!
//! All three are instances of one generic [`PrimitiveArray<T>`], where `T` is a
//! Rust native type bound to a [`ColumnType`] through [`PrimitiveType`]. NusaDB's
//! catalog [`ColumnType`] distinguishes only 64-bit integers, 64-bit floats, and
//! booleans, so those are the widths instantiated here; narrower widths
//! (`Int8`/`Int16`/`Int32`, `Float32`) are intentionally omitted until the catalog
//! grows a column type to represent them — implementing them now would force
//! [`Array::data_type`] to misreport their type.
//!
//! Null tracking uses a per-element `bool` validity vector (`true` = present); the
//! bit-packed validity buffer is a later refinement. A `None` validity means
//! every element is present.

use std::any::Any;
use std::fmt::Debug;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::validity::{NullBuffer, is_null_at, null_count, split_options};

/// A Rust native type that can back a [`PrimitiveArray`], bound to the catalog
/// [`ColumnType`] it represents.
pub trait PrimitiveType: Copy + Debug + Default + PartialEq + Send + Sync + 'static {
    /// The catalog column type a [`PrimitiveArray`] of this native type reports.
    const DATA_TYPE: ColumnType;
}

impl PrimitiveType for i64 {
    const DATA_TYPE: ColumnType = ColumnType::Int;
}

impl PrimitiveType for f64 {
    const DATA_TYPE: ColumnType = ColumnType::Float;
}

impl PrimitiveType for bool {
    const DATA_TYPE: ColumnType = ColumnType::Bool;
}

/// A contiguous, null-aware column of fixed-width [`PrimitiveType`] values.
///
/// `validity` is `None` when every element is present; otherwise its bits mark the present
/// elements. Null slots still occupy a (default-valued) entry in `values` so element `i`
/// is always at index `i`; read them null-aware with [`PrimitiveArray::get`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveArray<T: PrimitiveType> {
    values: Vec<T>,
    validity: Option<NullBuffer>,
}

impl<T: PrimitiveType> PrimitiveArray<T> {
    /// Build an array from values with no nulls.
    #[must_use]
    pub const fn from_values(values: Vec<T>) -> Self {
        Self {
            values,
            validity: None,
        }
    }

    /// Build an array from optional values; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<T>>) -> Self {
        let (values, validity) = split_options(items, T::default());
        Self { values, validity }
    }

    /// The raw value slice. Entries at null positions hold a default placeholder;
    /// use [`PrimitiveArray::get`] for null-aware reads.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The element at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        self.values.get(index).copied()
    }
}

impl<T: PrimitiveType> Array for PrimitiveArray<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> ColumnType {
        T::DATA_TYPE
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

/// A column of 64-bit signed integers ([`ColumnType::Int`]).
pub type Int64Array = PrimitiveArray<i64>;

/// A column of 64-bit IEEE-754 floats ([`ColumnType::Float`]).
pub type Float64Array = PrimitiveArray<f64>;

/// A column of booleans ([`ColumnType::Bool`]).
pub type BooleanArray = PrimitiveArray<bool>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_values_has_no_nulls() {
        let a = Int64Array::from_values(vec![1, 2, 3]);
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.data_type(), ColumnType::Int);
        assert_eq!(a.get(0), Some(1));
        assert_eq!(a.get(2), Some(3));
        assert!(a.is_valid(1));
        assert!(!a.is_null(1));
        assert_eq!(a.values(), &[1, 2, 3]);
    }

    #[test]
    fn from_options_tracks_nulls() {
        let a = Int64Array::from_options(vec![Some(10), None, Some(30)]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.get(0), Some(10));
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
        assert!(a.is_valid(0));
        assert_eq!(a.get(2), Some(30));
    }

    #[test]
    fn all_present_options_drop_validity() {
        let a = Int64Array::from_options(vec![Some(1), Some(2)]);
        assert_eq!(a.null_count(), 0);
        assert!(!a.is_null(0));
        // Out-of-range index is reported as non-null.
        assert!(!a.is_null(99));
        assert_eq!(a.get(99), None);
    }

    #[test]
    fn float_array_reports_float() {
        let a = Float64Array::from_options(vec![Some(1.5), None]);
        assert_eq!(a.data_type(), ColumnType::Float);
        assert_eq!(a.get(0), Some(1.5));
        assert_eq!(a.get(1), None);
        assert_eq!(a.null_count(), 1);
    }

    #[test]
    fn boolean_array_reports_bool() {
        let a = BooleanArray::from_values(vec![true, false, true]);
        assert_eq!(a.data_type(), ColumnType::Bool);
        assert_eq!(a.get(1), Some(false));
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: crate::batch::ArrayRef = std::sync::Arc::new(Int64Array::from_values(vec![7, 8]));
        assert_eq!(arr.data_type(), ColumnType::Int);
        assert_eq!(arr.len(), 2);
        let back = arr
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("downcast to Int64Array");
        assert_eq!(back.get(0), Some(7));
    }
}
