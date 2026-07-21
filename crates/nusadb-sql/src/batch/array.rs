//! The [`Array`] contract: a typed, null-aware column of a [`RecordBatch`](super::RecordBatch).

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use nusadb_core::ColumnType;

/// A single column of a [`RecordBatch`](super::RecordBatch): a contiguous, typed run
/// of values with per-element validity.
///
/// Concrete implementations (primitive, string, …) arrive in later tasks
/// This trait is the seam operators program against. It is
/// `Send + Sync` so batches can cross thread boundaries for parallel scans.
///
/// Downcast to a concrete array via [`Array::as_any`]:
///
/// ```ignore
/// let ints = array.as_any().downcast_ref::<Int64Array>().unwrap();
/// ```
pub trait Array: Debug + Send + Sync {
    /// The number of elements (logical rows) in the column.
    fn len(&self) -> usize;

    /// Whether the column has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The column's element type.
    fn data_type(&self) -> ColumnType;

    /// The number of `NULL` elements.
    fn null_count(&self) -> usize;

    /// Whether the element at `index` is `NULL`.
    ///
    /// Behavior for an out-of-range `index` is implementation-defined; callers
    /// must stay within `0..self.len()`.
    fn is_null(&self, index: usize) -> bool;

    /// Whether the element at `index` is non-`NULL` (the negation of [`Array::is_null`]).
    fn is_valid(&self, index: usize) -> bool {
        !self.is_null(index)
    }

    /// Upcast to [`Any`] for downcasting to a concrete array type.
    fn as_any(&self) -> &dyn Any;
}

/// A reference-counted, dynamically typed column shared between batches and operators.
pub type ArrayRef = Arc<dyn Array>;
