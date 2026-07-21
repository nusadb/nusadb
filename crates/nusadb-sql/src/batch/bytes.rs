//! Variable-length column arrays: [`StringArray`] ([`ColumnType::Text`]) and
//! [`BinaryArray`] ([`ColumnType::Bytes`]).
//!
//! Both use the offsets + data layout: a single contiguous `data` byte buffer plus an
//! `offsets` vector of length `len + 1`, where element `i` occupies
//! `data[offsets[i]..offsets[i + 1]]`. A null element contributes zero bytes (its two
//! offsets are equal) and is marked in the validity vector. The shared storage lives in
//! the private [`VarBytes`]; the two public types differ only in element type
//! (`&str` vs `&[u8]`) and reported [`ColumnType`].
//!
//! Null tracking uses a per-element `bool` validity vector (`true` = present); the
//! bit-packed validity buffer is a later refinement.

use std::any::Any;

use nusadb_core::ColumnType;

use crate::batch::array::Array;
use crate::batch::validity::{NullBuffer, NullBufferBuilder, is_null_at, null_count};

/// Shared offsets + data storage for the variable-length arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VarBytes {
    /// Length `len + 1`; element `i` spans `data[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
    /// Concatenated bytes of every present element, in order.
    data: Vec<u8>,
    /// `None` when every element is present; else its bits mark the present elements.
    validity: Option<NullBuffer>,
}

impl VarBytes {
    /// Build storage from optional byte runs; `None` entries become nulls.
    fn build(items: impl IntoIterator<Item = Option<Vec<u8>>>) -> Self {
        let iter = items.into_iter();
        let (lower, _) = iter.size_hint();
        let mut offsets = Vec::with_capacity(lower + 1);
        let mut data = Vec::new();
        let mut builder = NullBufferBuilder::with_capacity(lower);
        offsets.push(0);
        for item in iter {
            if let Some(bytes) = item {
                data.extend_from_slice(&bytes);
                builder.push(true);
            } else {
                builder.push(false);
            }
            offsets.push(data.len());
        }
        Self {
            offsets,
            data,
            validity: builder.finish(),
        }
    }

    /// Number of elements (`offsets` has one more entry than there are elements).
    const fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of null elements.
    fn null_count(&self) -> usize {
        null_count(self.validity.as_ref())
    }

    fn is_null(&self, index: usize) -> bool {
        is_null_at(self.validity.as_ref(), index)
    }

    /// The bytes of element `index`, or `None` if it is null or out of range.
    fn value(&self, index: usize) -> Option<&[u8]> {
        if self.is_null(index) {
            return None;
        }
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        self.data.get(start..end)
    }
}

/// A column of UTF-8 strings ([`ColumnType::Text`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringArray(VarBytes);

impl StringArray {
    /// Build an array from strings with no nulls.
    #[must_use]
    pub fn from_values(values: Vec<String>) -> Self {
        Self(VarBytes::build(
            values.into_iter().map(|s| Some(s.into_bytes())),
        ))
    }

    /// Build an array from optional strings; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<String>>) -> Self {
        Self(VarBytes::build(
            items.into_iter().map(|o| o.map(String::into_bytes)),
        ))
    }

    /// The string at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&str> {
        // Always valid UTF-8: the bytes came from `String`s.
        self.0
            .value(index)
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
    }
}

/// Incremental [`StringArray`] builder (R2 stage 2b): values append as borrowed `&str` straight
/// into the shared offsets+data buffers — no per-value `String` allocation. The vectorized scan's
/// text columns fill through this; `from_options` remains for callers that already own `String`s.
pub(super) struct StringBuilder {
    offsets: Vec<usize>,
    data: Vec<u8>,
    validity: NullBufferBuilder,
}

impl StringBuilder {
    /// An empty builder sized for about `capacity` elements.
    pub(super) fn with_capacity(capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            offsets,
            data: Vec::new(),
            validity: NullBufferBuilder::with_capacity(capacity),
        }
    }

    /// Append one present string.
    pub(super) fn append_value(&mut self, value: &str) {
        self.data.extend_from_slice(value.as_bytes());
        self.offsets.push(self.data.len());
        self.validity.push(true);
    }

    /// Append one null slot.
    pub(super) fn append_null(&mut self) {
        self.offsets.push(self.data.len());
        self.validity.push(false);
    }

    /// Finish into the immutable array.
    pub(super) fn finish(self) -> StringArray {
        StringArray(VarBytes {
            offsets: self.offsets,
            data: self.data,
            validity: self.validity.finish(),
        })
    }
}

impl Array for StringArray {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Text
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

/// A column of raw byte strings ([`ColumnType::Bytes`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryArray(VarBytes);

impl BinaryArray {
    /// Build an array from byte runs with no nulls.
    #[must_use]
    pub fn from_values(values: Vec<Vec<u8>>) -> Self {
        Self(VarBytes::build(values.into_iter().map(Some)))
    }

    /// Build an array from optional byte runs; `None` entries become nulls.
    #[must_use]
    pub fn from_options(items: Vec<Option<Vec<u8>>>) -> Self {
        Self(VarBytes::build(items))
    }

    /// The bytes at `index`, or `None` if it is null or out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.0.value(index)
    }
}

impl Array for BinaryArray {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn data_type(&self) -> ColumnType {
        ColumnType::Bytes
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
    fn string_from_values_roundtrips() {
        let a = StringArray::from_values(vec!["a".to_owned(), String::new(), "héllo".to_owned()]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 0);
        assert_eq!(a.data_type(), ColumnType::Text);
        assert_eq!(a.get(0), Some("a"));
        assert_eq!(a.get(1), Some("")); // empty string is not null
        assert_eq!(a.get(2), Some("héllo"));
        assert!(a.get(3).is_none()); // out of range
    }

    #[test]
    fn string_tracks_nulls() {
        let a = StringArray::from_options(vec![Some("x".to_owned()), None, Some("z".to_owned())]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.get(0), Some("x"));
        assert_eq!(a.get(1), None);
        assert!(a.is_null(1));
        assert!(a.is_valid(2));
        assert_eq!(a.get(2), Some("z"));
    }

    #[test]
    fn binary_roundtrips_and_tracks_nulls() {
        let a = BinaryArray::from_options(vec![Some(vec![1, 2, 3]), None, Some(vec![])]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.data_type(), ColumnType::Bytes);
        assert_eq!(a.get(0), Some(&[1, 2, 3][..]));
        assert_eq!(a.get(1), None);
        assert_eq!(a.get(2), Some(&[][..])); // empty bytes is not null
        assert!(a.is_null(1));
    }

    #[test]
    fn empty_array() {
        let a = StringArray::from_values(vec![]);
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.null_count(), 0);
        assert!(a.get(0).is_none());
    }

    #[test]
    fn usable_as_dyn_array_and_downcasts() {
        let arr: ArrayRef = Arc::new(StringArray::from_values(vec!["hi".to_owned()]));
        assert_eq!(arr.data_type(), ColumnType::Text);
        assert_eq!(arr.len(), 1);
        let back = arr
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("downcast to StringArray");
        assert_eq!(back.get(0), Some("hi"));
    }
}
