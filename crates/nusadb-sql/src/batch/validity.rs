//! Shared null-tracking for the columnar arrays: the bit-packed [`NullBuffer`].
//!
//! A column's validity is one bit per element — **set = present (non-null)**, the Apache
//! Arrow convention. A `None` validity buffer means every element is present, so the
//! common all-valid column carries no per-row overhead. Trailing bits in the final byte
//! (beyond the element count) are always clear.

/// A bit-packed validity mask: one bit per element, set when the element is present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NullBuffer {
    /// Validity bits, LSB-first within each byte (bit `i` → byte `i / 8`, position `i % 8`).
    bits: Vec<u8>,
    /// Number of elements the mask describes.
    len: usize,
    /// Number of null (bit-clear) elements.
    null_count: usize,
}

impl NullBuffer {
    /// The number of null elements.
    pub(super) const fn null_count(&self) -> usize {
        self.null_count
    }

    /// Whether element `index` is null. An out-of-range index is reported as non-null.
    pub(super) fn is_null(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        self.bits
            .get(index / 8)
            .is_some_and(|byte| (*byte >> (index % 8)) & 1 == 0)
    }
}

/// Incrementally packs validity bits, yielding `None` when no nulls were seen.
pub(super) struct NullBufferBuilder {
    bits: Vec<u8>,
    len: usize,
    null_count: usize,
}

impl NullBufferBuilder {
    /// A builder pre-sized for `cap` elements.
    pub(super) fn with_capacity(cap: usize) -> Self {
        Self {
            bits: Vec::with_capacity(cap.div_ceil(8)),
            len: 0,
            null_count: 0,
        }
    }

    /// Append one element's validity (`true` = present).
    pub(super) fn push(&mut self, valid: bool) {
        if self.len.is_multiple_of(8) {
            self.bits.push(0);
        }
        if valid {
            if let Some(byte) = self.bits.get_mut(self.len / 8) {
                *byte |= 1u8 << (self.len % 8);
            }
        } else {
            self.null_count += 1;
        }
        self.len += 1;
    }

    /// Finish, returning `None` if every element was present (no per-row overhead needed).
    pub(super) fn finish(self) -> Option<NullBuffer> {
        if self.null_count == 0 {
            None
        } else {
            Some(NullBuffer {
                bits: self.bits,
                len: self.len,
                null_count: self.null_count,
            })
        }
    }
}

/// Split optional values into a dense value vector (nulls filled with `placeholder`) and a
/// packed validity buffer (`None` when every element is present).
pub(super) fn split_options<T: Copy>(
    items: Vec<Option<T>>,
    placeholder: T,
) -> (Vec<T>, Option<NullBuffer>) {
    let mut values = Vec::with_capacity(items.len());
    let mut builder = NullBufferBuilder::with_capacity(items.len());
    for item in items {
        if let Some(value) = item {
            values.push(value);
            builder.push(true);
        } else {
            values.push(placeholder);
            builder.push(false);
        }
    }
    (values, builder.finish())
}

/// Pack a `true = present` boolean validity slice into a [`NullBuffer`] (`None` if all present).
pub(super) fn pack_validity(validity: &[bool]) -> Option<NullBuffer> {
    let mut builder = NullBufferBuilder::with_capacity(validity.len());
    for &valid in validity {
        builder.push(valid);
    }
    builder.finish()
}

/// Whether element `index` is null, given an optional validity buffer. A `None` buffer
/// means every element is present.
pub(super) fn is_null_at(validity: Option<&NullBuffer>, index: usize) -> bool {
    validity.is_some_and(|v| v.is_null(index))
}

/// The null count of an optional validity buffer (`None` → 0).
pub(super) fn null_count(validity: Option<&NullBuffer>) -> usize {
    validity.map_or(0, NullBuffer::null_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_present_yields_none() {
        let (values, validity) = split_options(vec![Some(1_i64), Some(2)], 0);
        assert_eq!(values, vec![1, 2]);
        assert!(validity.is_none());
        assert_eq!(null_count(validity.as_ref()), 0);
    }

    #[test]
    fn packs_nulls_and_reads_back() {
        let (values, validity) = split_options(vec![Some(10_i64), None, Some(30)], -1);
        assert_eq!(values, vec![10, -1, 30]); // null slot holds the placeholder
        let buf = validity.expect("has nulls");
        assert_eq!(buf.null_count(), 1);
        assert!(!buf.is_null(0));
        assert!(buf.is_null(1));
        assert!(!buf.is_null(2));
        assert!(!buf.is_null(99)); // out of range → non-null
    }

    #[test]
    fn pack_spanning_multiple_bytes() {
        // 10 elements, nulls at 0, 8, 9 — exercises the byte boundary at bit 8.
        let mut flags = [true; 10];
        flags[0] = false;
        flags[8] = false;
        flags[9] = false;
        let buf = pack_validity(&flags).expect("has nulls");
        assert_eq!(buf.null_count(), 3);
        assert!(buf.is_null(0));
        assert!(!buf.is_null(1));
        assert!(!buf.is_null(7));
        assert!(buf.is_null(8));
        assert!(buf.is_null(9));
    }

    #[test]
    fn pack_all_present_is_none() {
        assert!(pack_validity(&[true, true, true]).is_none());
    }
}
