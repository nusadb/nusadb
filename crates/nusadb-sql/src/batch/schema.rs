//! Typed shape of a [`RecordBatch`](super::RecordBatch): an ordered list of named,
//! typed [`Field`]s.

use nusadb_core::ColumnType;

/// One column of a [`Schema`]: a name, a [`ColumnType`], and whether it admits NULL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    name: String,
    data_type: ColumnType,
    nullable: bool,
}

impl Field {
    /// Create a field with the given name, type, and nullability.
    #[must_use]
    pub fn new(name: impl Into<String>, data_type: ColumnType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }

    /// The field's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The field's column type.
    #[must_use]
    pub const fn data_type(&self) -> ColumnType {
        self.data_type
    }

    /// Whether the field admits `NULL`.
    #[must_use]
    pub const fn is_nullable(&self) -> bool {
        self.nullable
    }
}

/// The ordered, typed columns of a [`RecordBatch`](super::RecordBatch).
///
/// Field order is significant: a batch's *i*-th column corresponds to the *i*-th field.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    /// Build a schema from an ordered list of fields.
    #[must_use]
    pub const fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    /// The empty schema (no columns).
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// The fields, in column order.
    #[must_use]
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// The number of fields (columns).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no fields.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field at column index `index`, or `None` if out of range.
    #[must_use]
    pub fn field(&self, index: usize) -> Option<&Field> {
        self.fields.get(index)
    }

    /// The column index of the first field named `name`, or `None` if absent.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name() == name)
    }

    /// The first field named `name`, or `None` if absent.
    #[must_use]
    pub fn field_with_name(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Schema {
        Schema::new(vec![
            Field::new("id", ColumnType::Int, false),
            Field::new("label", ColumnType::Text, true),
        ])
    }

    #[test]
    fn field_accessors() {
        let f = Field::new("id", ColumnType::Int, false);
        assert_eq!(f.name(), "id");
        assert_eq!(f.data_type(), ColumnType::Int);
        assert!(!f.is_nullable());
    }

    #[test]
    fn schema_len_and_lookup() {
        let s = sample();
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        assert_eq!(s.index_of("label"), Some(1));
        assert_eq!(s.index_of("missing"), None);
        assert_eq!(
            s.field_with_name("id").map(Field::data_type),
            Some(ColumnType::Int)
        );
        assert_eq!(s.field(1).map(Field::name), Some("label"));
        assert!(s.field(2).is_none());
    }

    #[test]
    fn empty_schema() {
        let s = Schema::empty();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(s.field(0).is_none());
    }
}
