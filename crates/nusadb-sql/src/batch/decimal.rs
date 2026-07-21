//! Exact-decimal column array: [`DecimalArray`] ([`ColumnType::Numeric`]).
//!
//! Unlike the other scalar arrays, [`ColumnType::Numeric`] carries a `precision` and
//! `scale`, so a [`DecimalArray`] stores them at the column level and reports them from
//! [`Array::data_type`]. Element values are [`Decimal`]s (mantissa + per-value scale).
//!
//! Null tracking uses the shared [`validity`](crate::batch::validity) helpers; the
//! bit-packed validity buffer is a later refinement.

use std::any::Any;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::validity::{NullBuffer, is_null_at, null_count, split_options};
use crate::numeric::Decimal;

/// A contiguous, null-aware column of [`Decimal`] values with a declared precision and scale.
#[derive(Debug, Clone)]
pub struct DecimalArray {
    values: Vec<Decimal>,
    validity: Option<NullBuffer>,
    precision: u8,
    scale: u8,
}

impl DecimalArray {
    /// Build an array from decimals with no nulls, declaring the column `precision`/`scale`.
    #[must_use]
    pub const fn from_values(values: Vec<Decimal>, precision: u8, scale: u8) -> Self {
        Self {
            values,
            validity: None,
            precision,
            scale,
        }
    }

    /// Build an array from optional decimals (`None` → null), declaring `precision`/`scale`.
    #[must_use]
    pub fn from_options(items: Vec<Option<Decimal>>, precision: u8, scale: u8) -> Self {
        let (values, validity) = split_options(items, Decimal::ZERO);
        Self {
            values,
            validity,
            precision,
            scale,
        }
    }

    /// The column's declared total precision (`0` = unconstrained).
    #[must_use]
    pub const fn precision(&self) -> u8 {
        self.precision
    }

    /// The column's declared fractional scale.
    #[must_use]
    pub const fn scale(&self) -> u8 {
        self.scale
    }

    /// The decimal at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Decimal> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        self.values.get(index).copied()
    }
}

impl Array for DecimalArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Numeric {
            precision: self.precision,
            scale: self.scale,
        }
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

    fn dec(mantissa: i128, scale: u8) -> Decimal {
        Decimal { mantissa, scale }
    }

    #[test]
    fn reports_numeric_with_precision_scale() {
        let a = DecimalArray::from_values(vec![dec(1234, 2), dec(5, 0)], 10, 2);
        assert_eq!(a.len(), 2);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.precision(), 10);
        assert_eq!(a.scale(), 2);
        assert_eq!(
            a.data_type(),
            ColumnType::Numeric {
                precision: 10,
                scale: 2
            }
        );
        assert_eq!(a.get(0), Some(dec(1234, 2)));
        assert!(a.get(2).is_none());
    }

    #[test]
    fn tracks_nulls() {
        let a = DecimalArray::from_options(vec![Some(dec(9, 0)), None, Some(dec(7, 0))], 0, 0);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.get(0), Some(dec(9, 0)));
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
        assert!(a.is_valid(2));
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: ArrayRef = Arc::new(DecimalArray::from_values(vec![dec(42, 1)], 4, 1));
        assert_eq!(
            arr.data_type(),
            ColumnType::Numeric {
                precision: 4,
                scale: 1
            }
        );
        let back = arr
            .as_any()
            .downcast_ref::<DecimalArray>()
            .expect("downcast to DecimalArray");
        assert_eq!(back.get(0), Some(dec(42, 1)));
    }
}
