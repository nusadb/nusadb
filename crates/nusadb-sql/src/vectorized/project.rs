//! [`Project`]: evaluate a list of expressions over its child's batches to form a
//! new set of output columns.
//!
//! It is the column-rewriting counterpart to [`Filter`](super::Filter)'s row-dropping: row
//! count is preserved 1:1, but the output schema is whatever the projection list defines
//! (a `SELECT a, b + 1 AS c, ...`). Each [`Projection`]'s expression is evaluated per row
//! with the shared row evaluator ([`eval`](crate::executor::eval)); the results are
//! re-assembled into a [`RecordBatch`] whose fields are the projections' names and types.
//!
//! A projection list of bare column references (the common `SELECT a, b` shape) takes a
//! passthrough fast path that just selects/reorders the input column arrays (cheap `Arc`
//! clones) with no per-row evaluation or row round-trip. A column-at-a-time kernel for
//! *computed* projections is a later SIMD refinement (+).

use std::sync::Arc;

use crate::batch::convert::{batch_to_rows, rows_to_batch};
use crate::batch::{Field, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::eval::eval;
use crate::planner::{Projection, TypedExprKind};
use crate::vectorized::Operator;

/// A projection over a child [`Operator`]'s batch stream: one output column per
/// [`Projection`], evaluated row by row.
#[derive(Debug)]
pub struct Project {
    child: Box<dyn Operator>,
    projections: Vec<Projection>,
    schema: Arc<Schema>,
    /// Pre-recognized passthrough: when every projection is a bare `Column(i)` reference, the
    /// child-column ordinals to select in output order. `None` means at least one projection is a
    /// computed expression, so the row evaluator runs.
    passthrough: Option<Vec<usize>>,
}

impl Project {
    /// Build a projection of `child` producing one column per entry of `projections`.
    ///
    /// Each projection's [`Column`](crate::planner::TypedExprKind::Column) ordinals index
    /// into `child`'s schema; the output schema's fields take their name from
    /// [`Projection::name`] and their type from the projection expression's type.
    #[must_use]
    pub fn new(child: Box<dyn Operator>, projections: Vec<Projection>) -> Self {
        let schema = Arc::new(Schema::new(
            projections
                .iter()
                // A projected expression can evaluate to NULL, so every output field is nullable.
                .map(|p| Field::new(p.name.clone(), p.expr.ty, true))
                .collect(),
        ));
        let passthrough = projections
            .iter()
            .map(|p| match p.expr.kind {
                TypedExprKind::Column(i) => Some(i),
                _ => None,
            })
            .collect::<Option<Vec<usize>>>();
        Self {
            child,
            projections,
            schema,
            passthrough,
        }
    }

    /// Select the child columns named by `indices` (output order) by cloning their `ArrayRef`s —
    /// no per-row work. `None` if an ordinal is out of range (defensive; the caller falls back).
    fn passthrough_batch(&self, batch: &RecordBatch, indices: &[usize]) -> Option<RecordBatch> {
        let columns = batch.columns();
        let selected = indices
            .iter()
            .map(|&i| columns.get(i).map(Arc::clone))
            .collect::<Option<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(&self.schema), selected).ok()
    }
}

impl Operator for Project {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, Error> {
        let Some(batch) = self.child.next_batch()? else {
            return Ok(None);
        };
        if let Some(indices) = &self.passthrough
            && let Some(out) = self.passthrough_batch(&batch, indices)
        {
            return Ok(Some(out));
        }
        let mut out_rows: Vec<crate::Row> = Vec::with_capacity(batch.num_rows());
        for row in batch_to_rows(&batch) {
            let mut projected = Vec::with_capacity(self.projections.len());
            for projection in &self.projections {
                projected.push(eval(&projection.expr, &row)?);
            }
            out_rows.push(projected);
        }
        rows_to_batch(&self.schema, out_rows).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::Project;
    use crate::Field;
    use crate::ast::{self, BinaryOp};
    use crate::batch::{Int64Array, RecordBatch, Schema, StringArray};
    use crate::executor::row;
    use crate::planner::{Projection, TypedExpr, TypedExprKind};
    use crate::vectorized::{Operator, SeqScan};
    use nusadb_core::engine::{SharedTuple, Tid, TupleScan};
    use nusadb_core::{ColumnType, PageId, Result as CoreResult, SlotIdx};
    use std::sync::Arc;

    struct VecScan {
        tuples: Vec<SharedTuple>,
        pos: usize,
    }

    impl TupleScan for VecScan {
        fn try_next(&mut self) -> CoreResult<Option<(Tid, SharedTuple)>> {
            let item = self.tuples.get(self.pos).map(|t| {
                (
                    Tid {
                        page: PageId(0),
                        slot: SlotIdx(0),
                    },
                    Arc::clone(t),
                )
            });
            if item.is_some() {
                self.pos += 1;
            }
            Ok(item)
        }
    }

    /// A `SeqScan` over `(id INT, name TEXT)` rows.
    fn scan(rows: &[(i64, &str)]) -> SeqScan {
        let types = [ColumnType::Int, ColumnType::Text];
        let tuples = rows
            .iter()
            .map(|&(id, name)| {
                let row = [ast::Value::Int(id), ast::Value::Text(name.to_owned())];
                SharedTuple::from(row::encode(&row, &types).unwrap().as_slice())
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ColumnType::Int, true),
            Field::new("name", ColumnType::Text, true),
        ]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    fn col(index: usize, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Column(index),
            ty,
        }
    }

    fn drain(mut op: Project) -> Vec<RecordBatch> {
        let mut out = Vec::new();
        while let Some(batch) = op.next_batch().unwrap() {
            out.push(batch);
        }
        out
    }

    #[test]
    fn reorders_and_renames_columns() {
        // SELECT name, id  → swap order, keep values.
        let projections = vec![
            Projection {
                expr: col(1, ColumnType::Text),
                name: "name".to_owned(),
            },
            Projection {
                expr: col(0, ColumnType::Int),
                name: "id".to_owned(),
            },
        ];
        let op = Project::new(Box::new(scan(&[(1, "a"), (2, "b")])), projections);
        assert_eq!(op.schema().field(0).unwrap().data_type(), ColumnType::Text);
        assert_eq!(op.schema().field(1).unwrap().data_type(), ColumnType::Int);

        let batches = drain(op);
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let names = batch
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ids = batch
            .column(1)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!((names.get(0), ids.get(0)), (Some("a"), Some(1)));
        assert_eq!((names.get(1), ids.get(1)), (Some("b"), Some(2)));
    }

    #[test]
    fn evaluates_a_computed_column() {
        // SELECT id + 10 AS bumped.
        let bumped = TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(col(0, ColumnType::Int)),
                op: BinaryOp::Plus,
                right: Box::new(TypedExpr {
                    kind: TypedExprKind::Literal(ast::Value::Int(10)),
                    ty: ColumnType::Int,
                }),
            },
            ty: ColumnType::Int,
        };
        let op = Project::new(
            Box::new(scan(&[(1, "a"), (5, "b")])),
            vec![Projection {
                expr: bumped,
                name: "bumped".to_owned(),
            }],
        );
        let batches = drain(op);
        let col = batches[0]
            .column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!((col.get(0), col.get(1)), (Some(11), Some(15)));
    }

    #[test]
    fn empty_input_yields_no_batch() {
        let op = Project::new(
            Box::new(scan(&[])),
            vec![Projection {
                expr: col(0, ColumnType::Int),
                name: "id".to_owned(),
            }],
        );
        assert!(drain(op).is_empty());
    }

    #[test]
    fn preserves_rows_across_batch_boundary() {
        // 1 full batch + 2 rows → Project keeps row count 1:1 across both batches.
        let total = crate::BATCH_SIZE + 2;
        let rows: Vec<(i64, &str)> = (0..total)
            .map(|i| (i64::try_from(i).unwrap(), "x"))
            .collect();
        let op = Project::new(
            Box::new(scan(&rows)),
            vec![Projection {
                expr: col(0, ColumnType::Int),
                name: "id".to_owned(),
            }],
        );
        let batches = drain(op);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), crate::BATCH_SIZE);
        assert_eq!(batches[1].num_rows(), 2);
    }
}
