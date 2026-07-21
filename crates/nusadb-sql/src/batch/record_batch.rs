//! [`RecordBatch`]: a [`Schema`] paired with one [`ArrayRef`] column per field.

use std::sync::Arc;

use crate::batch::array::ArrayRef;
use crate::batch::schema::Schema;
use crate::error::Error;

/// A horizontal slice of a result: a [`Schema`] and one column [`ArrayRef`] per field,
/// all of equal length.
///
/// The vectorized executor's unit of data flow. Construct with [`RecordBatch::try_new`],
/// which enforces three invariants:
///
/// 1. the column count equals the schema's field count;
/// 2. each column's [`data_type`](super::Array::data_type) matches its field's;
/// 3. all columns have the same length (the batch's row count).
///
/// Cloning is cheap: columns are shared via [`Arc`].
#[derive(Debug, Clone)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
    row_count: usize,
}

impl RecordBatch {
    /// Assemble a batch from a schema and its columns, validating the batch invariants.
    ///
    /// A batch with no columns has a row count of zero.
    ///
    /// # Errors
    ///
    /// - [`Error::ArityMismatch`] if the number of columns differs from the number of
    ///   fields, or if the columns are not all the same length.
    /// - [`Error::TypeMismatch`] if a column's type differs from its field's type.
    pub fn try_new(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Self, Error> {
        if columns.len() != schema.len() {
            return Err(Error::ArityMismatch {
                context: "record batch columns vs schema fields".to_owned(),
                expected: schema.len(),
                found: columns.len(),
            });
        }

        let row_count = columns.first().map_or(0, |c| c.len());

        for (column, field) in columns.iter().zip(schema.fields()) {
            if column.data_type() != field.data_type() {
                return Err(Error::TypeMismatch {
                    context: format!("record batch column `{}`", field.name()),
                    expected: field.data_type(),
                    found: column.data_type(),
                });
            }
            if column.len() != row_count {
                return Err(Error::ArityMismatch {
                    context: format!("record batch column `{}` length", field.name()),
                    expected: row_count,
                    found: column.len(),
                });
            }
        }

        Ok(Self {
            schema,
            columns,
            row_count,
        })
    }

    /// The batch's schema.
    #[must_use]
    pub const fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// The number of columns.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// The number of rows (the shared length of every column).
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.row_count
    }

    /// All columns, in schema order.
    #[must_use]
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    /// The column at index `index`, or `None` if out of range.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&ArrayRef> {
        self.columns.get(index)
    }

    /// The column for the first field named `name`, or `None` if absent.
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.schema.index_of(name).and_then(|i| self.columns.get(i))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use nusadb_core::ColumnType;

    use super::*;
    use crate::batch::array::Array;
    use crate::batch::schema::Field;

    /// A minimal column stand-in for testing batch invariants before the concrete
    /// array types (+) exist.
    #[derive(Debug)]
    struct MockArray {
        data_type: ColumnType,
        len: usize,
    }

    impl Array for MockArray {
        fn len(&self) -> usize {
            self.len
        }
        fn data_type(&self) -> ColumnType {
            self.data_type
        }
        fn null_count(&self) -> usize {
            0
        }
        fn is_null(&self, _index: usize) -> bool {
            false
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn col(data_type: ColumnType, len: usize) -> ArrayRef {
        Arc::new(MockArray { data_type, len })
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", ColumnType::Int, false),
            Field::new("label", ColumnType::Text, true),
        ]))
    }

    #[test]
    fn try_new_accepts_well_formed_batch() {
        let batch = RecordBatch::try_new(
            schema(),
            vec![col(ColumnType::Int, 3), col(ColumnType::Text, 3)],
        )
        .expect("valid batch");
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch.column(0).map(|c| c.data_type()),
            Some(ColumnType::Int)
        );
        assert_eq!(
            batch.column_by_name("label").map(|c| c.data_type()),
            Some(ColumnType::Text),
        );
        assert!(batch.column_by_name("missing").is_none());
        assert!(batch.column(2).is_none());
    }

    #[test]
    fn try_new_rejects_column_count_mismatch() {
        let err = RecordBatch::try_new(schema(), vec![col(ColumnType::Int, 3)])
            .expect_err("too few columns");
        assert!(matches!(
            err,
            Error::ArityMismatch {
                expected: 2,
                found: 1,
                ..
            }
        ));
    }

    #[test]
    fn try_new_rejects_type_mismatch() {
        let err = RecordBatch::try_new(
            schema(),
            vec![col(ColumnType::Int, 3), col(ColumnType::Int, 3)],
        )
        .expect_err("label column is Int, not Text");
        assert!(matches!(
            err,
            Error::TypeMismatch {
                expected: ColumnType::Text,
                found: ColumnType::Int,
                ..
            }
        ));
    }

    #[test]
    fn try_new_rejects_ragged_columns() {
        let err = RecordBatch::try_new(
            schema(),
            vec![col(ColumnType::Int, 3), col(ColumnType::Text, 2)],
        )
        .expect_err("columns of unequal length");
        assert!(matches!(
            err,
            Error::ArityMismatch {
                expected: 3,
                found: 2,
                ..
            }
        ));
    }

    #[test]
    fn zero_column_batch_has_zero_rows() {
        let batch = RecordBatch::try_new(Arc::new(Schema::empty()), vec![]).expect("empty batch");
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 0);
    }
}
