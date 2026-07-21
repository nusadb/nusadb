//! Nested list column array: [`ListArray`]
//! ([`ColumnType::Array`](nusadb_core::ColumnType::Array)).
//!
//! A list column holds, per row, a variable-length list of scalar elements that all share
//! the column's declared [`ArrayElem`] type ([`ast::Value::Array`](crate::ast::Value::Array)).
//! It uses the Arrow-style offsets + child layout: a single flat `child` [`ArrayRef`]
//! holds every element of every row concatenated, and an `offsets` vector of length
//! `rows + 1` slices it — row `i`'s elements are `child[offsets[i]..offsets[i + 1]]`.
//! A null row has equal offsets (zero elements) and is flagged in the validity vector.
//!
//! Null tracking uses the shared [`validity`](crate::batch::validity) helpers; the
//! bit-packed validity buffer is a later refinement.

use std::any::Any;

use nusadb_core::ColumnType;
use nusadb_core::engine::ArrayElem;

use crate::batch::array::{Array, ArrayRef};
use crate::batch::validity::{NullBuffer, is_null_at, null_count, pack_validity};
use crate::error::Error;

/// A contiguous, null-aware column of variable-length lists over a scalar [`ArrayElem`].
#[derive(Debug, Clone)]
pub struct ListArray {
    element: ArrayElem,
    child: ArrayRef,
    offsets: Vec<usize>,
    validity: Option<NullBuffer>,
}

impl ListArray {
    /// Assemble a list array from its element type, flat child column, row offsets, and
    /// optional validity (`true` = present row; `None` = no nulls).
    ///
    /// # Errors
    ///
    /// [`Error::ArityMismatch`] if `offsets` is empty, is not non-decreasing, does not end
    /// at `child.len()`, or if a supplied `validity` length does not match the row count.
    pub fn try_new(
        element: ArrayElem,
        child: ArrayRef,
        offsets: Vec<usize>,
        validity: Option<&[bool]>,
    ) -> Result<Self, Error> {
        let Some(&last) = offsets.last() else {
            return Err(Error::ArityMismatch {
                context: "list array offsets (must hold at least the leading 0)".to_owned(),
                expected: 1,
                found: 0,
            });
        };
        let monotonic = offsets
            .iter()
            .zip(offsets.iter().skip(1))
            .all(|(a, b)| a <= b);
        if !monotonic {
            return Err(Error::ArityMismatch {
                context: "list array offsets must be non-decreasing".to_owned(),
                expected: 0,
                found: 0,
            });
        }
        if last != child.len() {
            return Err(Error::ArityMismatch {
                context: "list array final offset vs child length".to_owned(),
                expected: child.len(),
                found: last,
            });
        }

        let rows = offsets.len() - 1;
        if let Some(v) = validity
            && v.len() != rows
        {
            return Err(Error::ArityMismatch {
                context: "list array validity length vs row count".to_owned(),
                expected: rows,
                found: v.len(),
            });
        }

        Ok(Self {
            element,
            child,
            offsets,
            validity: validity.and_then(pack_validity),
        })
    }

    /// The declared scalar element type of every list.
    #[must_use]
    pub const fn element_type(&self) -> ArrayElem {
        self.element
    }

    /// The flat child column holding every row's elements concatenated.
    #[must_use]
    pub const fn child(&self) -> &ArrayRef {
        &self.child
    }

    /// The `[start, end)` range of row `index`'s elements within [`ListArray::child`], or
    /// `None` if the row is null or out of range.
    #[must_use]
    pub fn value_range(&self, index: usize) -> Option<(usize, usize)> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        Some((start, end))
    }

    /// The number of elements in row `index`, or `None` if the row is null or out of range.
    #[must_use]
    pub fn value_len(&self, index: usize) -> Option<usize> {
        self.value_range(index).map(|(start, end)| end - start)
    }
}

impl Array for ListArray {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Array(self.element)
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
    use crate::batch::Int64Array;

    /// `[[1, 2], [], [3]]` over an `INT[]` column.
    fn sample() -> ListArray {
        let child: ArrayRef = Arc::new(Int64Array::from_values(vec![1, 2, 3]));
        ListArray::try_new(ArrayElem::Int, child, vec![0, 2, 2, 3], None).expect("valid list array")
    }

    #[test]
    fn reports_array_type_and_ranges() {
        let a = sample();
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.data_type(), ColumnType::Array(ArrayElem::Int));
        assert_eq!(a.element_type(), ArrayElem::Int);
        assert_eq!(a.value_range(0), Some((0, 2)));
        assert_eq!(a.value_len(0), Some(2));
        assert_eq!(a.value_len(1), Some(0)); // empty list, not null
        assert_eq!(a.value_range(2), Some((2, 3)));
        assert_eq!(a.value_range(3), None); // out of range
        assert_eq!(a.child().len(), 3);
    }

    #[test]
    fn tracks_null_rows() {
        let child: ArrayRef = Arc::new(Int64Array::from_values(vec![7]));
        let a = ListArray::try_new(ArrayElem::Int, child, vec![0, 1, 1], Some(&[true, false]))
            .expect("valid list array");
        assert_eq!(a.len(), 2);
        assert_eq!(a.null_count(), 1);
        assert!(a.is_null(1));
        assert_eq!(a.value_range(1), None); // null row
        assert_eq!(a.value_range(0), Some((0, 1)));
    }

    #[test]
    fn rejects_bad_offsets() {
        let child: ArrayRef = Arc::new(Int64Array::from_values(vec![1, 2]));
        // Final offset (1) does not match child length (2).
        let err = ListArray::try_new(ArrayElem::Int, child.clone(), vec![0, 1], None)
            .expect_err("final offset mismatch");
        assert!(matches!(err, Error::ArityMismatch { .. }));
        // Non-decreasing violated.
        let err = ListArray::try_new(ArrayElem::Int, child, vec![0, 2, 1], None)
            .expect_err("non-monotonic offsets");
        assert!(matches!(err, Error::ArityMismatch { .. }));
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let a = sample();
        let arr: ArrayRef = Arc::new(a);
        assert_eq!(arr.data_type(), ColumnType::Array(ArrayElem::Int));
        let back = arr
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("downcast to ListArray");
        assert_eq!(back.value_len(0), Some(2));
    }
}
