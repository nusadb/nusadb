//! Temporal column arrays: [`DateArray`], [`TimeArray`], [`TimestampArray`],
//! [`TimestampTzArray`], and [`IntervalArray`].
//!
//! The four date/time types are fixed-width integers ([`ColumnType::Date`] is `i32`
//! days; the others are `i64` microseconds), so they share one generic
//! [`TemporalArray<K>`], where the [`TemporalKind`] marker `K` supplies both the
//! native storage type and the reported [`ColumnType`]. (They cannot reuse
//! [`PrimitiveArray`](super::PrimitiveArray): its `data_type()` is derived from the
//! native type, so `i64` already means [`ColumnType::Int`].)
//!
//! [`IntervalArray`] is separate because an [`Interval`] is a composite
//! (months + days + microseconds), not a single integer.
//!
//! Null tracking uses a per-element `bool` validity vector (`true` = present); the
//! bit-packed validity buffer is a later refinement.

use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::validity::{NullBuffer, is_null_at, null_count, split_options};
use crate::interval::Interval;

/// A marker tying a date/time [`ColumnType`] to its native integer storage type.
pub trait TemporalKind: Copy + Debug + Send + Sync + 'static {
    /// The native storage type (`i32` days for `DATE`, `i64` microseconds otherwise).
    type Native: Copy + Debug + Default + Send + Sync + 'static;
    /// The column type a [`TemporalArray`] of this kind reports.
    const DATA_TYPE: ColumnType;
}

/// `DATE` — days since the Unix epoch, stored as `i32`.
#[derive(Debug, Clone, Copy)]
pub struct DateKind;
impl TemporalKind for DateKind {
    type Native = i32;
    const DATA_TYPE: ColumnType = ColumnType::Date;
}

/// `TIME` — microseconds since midnight, stored as `i64`.
#[derive(Debug, Clone, Copy)]
pub struct TimeKind;
impl TemporalKind for TimeKind {
    type Native = i64;
    const DATA_TYPE: ColumnType = ColumnType::Time;
}

/// `TIMESTAMP` — microseconds since the Unix epoch, stored as `i64`.
#[derive(Debug, Clone, Copy)]
pub struct TimestampKind;
impl TemporalKind for TimestampKind {
    type Native = i64;
    const DATA_TYPE: ColumnType = ColumnType::Timestamp;
}

/// `TIMESTAMP WITH TIME ZONE` — microseconds since the Unix epoch (UTC), stored as `i64`.
#[derive(Debug, Clone, Copy)]
pub struct TimestampTzKind;
impl TemporalKind for TimestampTzKind {
    type Native = i64;
    const DATA_TYPE: ColumnType = ColumnType::TimestampTz;
}

/// `TIME WITH TIME ZONE` — the packed local-time + zone `i64` (see [`crate::temporal`], P-TIMETZ).
#[derive(Debug, Clone, Copy)]
pub struct TimeTzKind;
impl TemporalKind for TimeTzKind {
    type Native = i64;
    const DATA_TYPE: ColumnType = ColumnType::TimeTz;
}

/// A contiguous, null-aware column of an integer-backed temporal type.
#[derive(Debug, Clone)]
pub struct TemporalArray<K: TemporalKind> {
    values: Vec<K::Native>,
    validity: Option<NullBuffer>,
    _marker: PhantomData<K>,
}

impl<K: TemporalKind> TemporalArray<K> {
    /// Build an array from native values with no nulls.
    #[must_use]
    pub const fn from_values(values: Vec<K::Native>) -> Self {
        Self {
            values,
            validity: None,
            _marker: PhantomData,
        }
    }

    /// Build an array from optional native values; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<K::Native>>) -> Self {
        let (values, validity) = split_options(items, K::Native::default());
        Self {
            values,
            validity,
            _marker: PhantomData,
        }
    }

    /// The native value at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<K::Native> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        self.values.get(index).copied()
    }
}

impl<K: TemporalKind> Array for TemporalArray<K> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> ColumnType {
        K::DATA_TYPE
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

/// A column of `DATE` values (`i32` days since the Unix epoch).
pub type DateArray = TemporalArray<DateKind>;

/// A column of `TIME` values (`i64` microseconds since midnight).
pub type TimeArray = TemporalArray<TimeKind>;

/// A column of `TIMESTAMP` values (`i64` microseconds since the Unix epoch).
pub type TimestampArray = TemporalArray<TimestampKind>;

/// A column of `TIMESTAMP WITH TIME ZONE` values (`i64` microseconds, UTC).
pub type TimestampTzArray = TemporalArray<TimestampTzKind>;

/// A column of `TIME WITH TIME ZONE` values (`i64` microseconds since midnight, UTC).
pub type TimeTzArray = TemporalArray<TimeTzKind>;

/// A contiguous, null-aware column of [`Interval`] values ([`ColumnType::Interval`]).
#[derive(Debug, Clone)]
pub struct IntervalArray {
    values: Vec<Interval>,
    validity: Option<NullBuffer>,
}

impl IntervalArray {
    /// Build an array from intervals with no nulls.
    #[must_use]
    pub const fn from_values(values: Vec<Interval>) -> Self {
        Self {
            values,
            validity: None,
        }
    }

    /// Build an array from optional intervals; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<Interval>>) -> Self {
        let zero = Interval {
            months: 0,
            days: 0,
            micros: 0,
        };
        let (values, validity) = split_options(items, zero);
        Self { values, validity }
    }

    /// The interval at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Interval> {
        if is_null_at(self.validity.as_ref(), index) {
            return None;
        }
        self.values.get(index).copied()
    }
}

impl Array for IntervalArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Interval
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

    #[test]
    fn date_array_reports_date_and_stores_i32() {
        let a = DateArray::from_values(vec![0, 19_000, -1]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.data_type(), ColumnType::Date);
        assert_eq!(a.get(0), Some(0_i32));
        assert_eq!(a.get(2), Some(-1_i32));
        assert!(a.get(3).is_none());
    }

    #[test]
    fn timestamp_array_tracks_nulls() {
        let a = TimestampArray::from_options(vec![Some(1_000_000), None, Some(2_000_000)]);
        assert_eq!(a.data_type(), ColumnType::Timestamp);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.get(0), Some(1_000_000_i64));
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
        assert!(a.is_valid(2));
    }

    #[test]
    fn time_and_timestamptz_report_their_types() {
        let t = TimeArray::from_values(vec![0, 86_399_000_000]);
        assert_eq!(t.data_type(), ColumnType::Time);
        let z = TimestampTzArray::from_values(vec![42]);
        assert_eq!(z.data_type(), ColumnType::TimestampTz);
        assert_eq!(z.get(0), Some(42_i64));
    }

    #[test]
    fn interval_array_roundtrips_and_tracks_nulls() {
        let one_day = Interval {
            months: 0,
            days: 1,
            micros: 0,
        };
        let a = IntervalArray::from_options(vec![Some(one_day), None]);
        assert_eq!(a.len(), 2);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.data_type(), ColumnType::Interval);
        assert_eq!(a.get(0), Some(one_day));
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: ArrayRef = Arc::new(TimestampArray::from_values(vec![7, 8, 9]));
        assert_eq!(arr.data_type(), ColumnType::Timestamp);
        assert_eq!(arr.len(), 3);
        let back = arr
            .as_any()
            .downcast_ref::<TimestampArray>()
            .expect("downcast to TimestampArray");
        assert_eq!(back.get(1), Some(8_i64));
    }
}
