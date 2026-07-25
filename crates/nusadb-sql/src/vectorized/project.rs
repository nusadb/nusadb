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

use crate::ast;
use crate::batch::convert::{batch_to_rows, rows_to_batch};
use crate::batch::{Array, ArrayRef, Field, Int64Array, RecordBatch, Schema};
use crate::error::Error;
use crate::executor::eval::{eval, int_op, int_value_bounds};
use crate::planner::{Projection, TypedExpr, TypedExprKind};
use crate::vectorized::Operator;
use nusadb_core::ColumnType;

/// A compiled integer-arithmetic expression evaluated **column-at-a-time** over a
/// [`RecordBatch`], skipping the row-boxing + `batch_to_rows`/`rows_to_batch` round-trip the scalar
/// evaluator pays. Only integer `+ - * / %` over integer columns and integer literals is compiled;
/// anything else stays `None` so the caller keeps the validated row path (this only *adds* a
/// fast-path). Reusing [`int_op`] (overflow-checked) and [`int_value_bounds`] makes the result
/// bit-identical to [`eval`] by construction.
#[derive(Debug)]
enum ColumnarInt {
    Column(usize),
    Literal(i64),
    Binary {
        op: ast::BinaryOp,
        left: Box<Self>,
        right: Box<Self>,
        /// This node's result type — for the same declared-width bounds check the row path applies.
        result_ty: ColumnType,
    },
}

impl ColumnarInt {
    /// Compile a `TypedExpr` into an integer-arithmetic plan, or `None` when any part is outside the
    /// supported shape (non-integer type, non-arithmetic op, or a non-`Int` literal).
    fn compile(expr: &TypedExpr) -> Option<Self> {
        use ast::BinaryOp as Op;
        if !matches!(expr.ty.physical(), ColumnType::Int) {
            return None;
        }
        match &expr.kind {
            TypedExprKind::Column(i) => Some(Self::Column(*i)),
            TypedExprKind::Literal(ast::Value::Int(v)) => Some(Self::Literal(*v)),
            TypedExprKind::Binary { left, op, right }
                if matches!(
                    op,
                    Op::Plus | Op::Minus | Op::Multiply | Op::Divide | Op::Modulo
                ) =>
            {
                Some(Self::Binary {
                    op: *op,
                    left: Box::new(Self::compile(left)?),
                    right: Box::new(Self::compile(right)?),
                    result_ty: expr.ty,
                })
            },
            _ => None,
        }
    }

    /// Evaluate over `batch`'s `n` rows into dense values with a per-row validity flag (`false` =
    /// NULL). An operand NULL propagates to a NULL result, exactly as `apply_arithmetic` does.
    #[allow(
        clippy::indexing_slicing,
        reason = "every operand/output vector is sized to n and r ranges over 0..n"
    )]
    fn eval(&self, batch: &RecordBatch, n: usize) -> Result<(Vec<i64>, Vec<bool>), Error> {
        match self {
            Self::Literal(v) => Ok((vec![*v; n], vec![true; n])),
            Self::Column(i) => {
                let arr = batch
                    .column(*i)
                    .and_then(|c| c.as_ref().as_any().downcast_ref::<Int64Array>())
                    .ok_or(Error::MalformedTuple { offset: *i })?;
                let valid = (0..n).map(|r| !arr.is_null(r)).collect();
                Ok((arr.values().to_vec(), valid))
            },
            Self::Binary {
                op,
                left,
                right,
                result_ty,
            } => {
                let (lv, lok) = left.eval(batch, n)?;
                let (rv, rok) = right.eval(batch, n)?;
                let bounds = int_value_bounds(*result_ty);
                let mut out = vec![0_i64; n];
                let mut valid = vec![false; n];
                for r in 0..n {
                    if !lok[r] || !rok[r] {
                        continue; // NULL propagates
                    }
                    let v = int_op(*op, lv[r], rv[r])?;
                    if bounds.is_some_and(|(lo, hi)| v < lo || v > hi) {
                        return Err(Error::IntegerOutOfRange);
                    }
                    out[r] = v;
                    valid[r] = true;
                }
                Ok((out, valid))
            },
        }
    }

    /// Evaluate over `batch` into an `Int64Array`, null where any operand is null.
    fn eval_array(&self, batch: &RecordBatch) -> Result<Int64Array, Error> {
        let n = batch.num_rows();
        let (vals, valid) = self.eval(batch, n)?;
        Ok(Int64Array::from_options(
            vals.into_iter()
                .zip(valid)
                .map(|(v, ok)| ok.then_some(v))
                .collect(),
        ))
    }
}

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
    /// Pre-compiled columnar plan: `Some` when every projection is an integer-arithmetic
    /// expression, so the batch is produced column-at-a-time without a row round-trip. `None` falls
    /// back to the scalar row evaluator. Never set alongside `passthrough`.
    columnar: Option<Vec<ColumnarInt>>,
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
        // A computed projection list that is entirely integer arithmetic evaluates columnar;
        // a bare-column list already takes the cheaper passthrough, so only compile when it does not.
        let columnar = if passthrough.is_some() {
            None
        } else {
            projections
                .iter()
                .map(|p| ColumnarInt::compile(&p.expr))
                .collect::<Option<Vec<_>>>()
        };
        Self {
            child,
            projections,
            schema,
            passthrough,
            columnar,
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
        // Columnar path: evaluate each integer-arithmetic projection over the batch's columns.
        // Genuine evaluation errors (overflow, out-of-range) propagate via `?` exactly as the row
        // path would raise them; only a `try_new` schema mismatch falls through to the row evaluator.
        if let Some(exprs) = &self.columnar {
            let columns = exprs
                .iter()
                .map(|e| -> Result<ArrayRef, Error> { Ok(Arc::new(e.eval_array(&batch)?)) })
                .collect::<Result<Vec<_>, _>>()?;
            if let Ok(out) = RecordBatch::try_new(Arc::clone(&self.schema), columns) {
                return Ok(Some(out));
            }
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

    /// A `SeqScan` over two nullable INT columns `(a, b)`.
    fn two_int_scan(rows: &[(ast::Value, ast::Value)]) -> SeqScan {
        let types = [ColumnType::Int, ColumnType::Int];
        let tuples = rows
            .iter()
            .map(|(a, b)| {
                SharedTuple::from(
                    row::encode(&[a.clone(), b.clone()], &types)
                        .unwrap()
                        .as_slice(),
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ColumnType::Int, true),
            Field::new("b", ColumnType::Int, true),
        ]));
        SeqScan::new(Box::new(VecScan { tuples, pos: 0 }), schema)
    }

    fn lit(v: i64) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Literal(ast::Value::Int(v)),
            ty: ColumnType::Int,
        }
    }
    fn bin(l: TypedExpr, op: BinaryOp, r: TypedExpr) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(l),
                op,
                right: Box::new(r),
            },
            ty: ColumnType::Int,
        }
    }

    /// The columnar integer-arithmetic path is bit-identical to the scalar reference over
    /// data with NULLs in both operands and a nested expression: a NULL operand propagates to a
    /// NULL result, and every non-NULL result matches the checked arithmetic (values stay in width).
    #[test]
    fn columnar_int_arithmetic_matches_scalar_with_nulls() {
        use ast::Value as V;
        let rows: Vec<(V, V)> = (0..37i64)
            .map(|i| {
                let a = if i % 4 == 0 { V::Null } else { V::Int(i - 18) };
                let b = if i % 7 == 0 {
                    V::Null
                } else {
                    V::Int((i % 9) + 1)
                };
                (a, b)
            })
            .collect();
        let a = || col(0, ColumnType::Int);
        let b = || col(1, ColumnType::Int);
        let projections = vec![
            Projection {
                expr: bin(a(), BinaryOp::Plus, b()),
                name: "s".into(),
            },
            Projection {
                expr: bin(a(), BinaryOp::Multiply, lit(2)),
                name: "d".into(),
            },
            Projection {
                expr: bin(bin(a(), BinaryOp::Plus, lit(1)), BinaryOp::Multiply, b()),
                name: "p".into(),
            },
            Projection {
                expr: bin(a(), BinaryOp::Minus, b()),
                name: "m".into(),
            },
        ];
        let op = Project::new(Box::new(two_int_scan(&rows)), projections);
        assert!(
            op.columnar.is_some(),
            "an all-integer-arithmetic projection list takes the columnar path"
        );
        let batches = drain(op);
        let column = |b: &RecordBatch, c: usize| -> Vec<Option<i64>> {
            let arr = b
                .column(c)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..b.num_rows()).map(|r| arr.get(r)).collect()
        };
        let mut got: Vec<Vec<Option<i64>>> = vec![Vec::new(); 4];
        for batch in &batches {
            for (c, slot) in got.iter_mut().enumerate() {
                slot.extend(column(batch, c));
            }
        }
        let val = |v: &V| match v {
            V::Int(x) => Some(*x),
            V::Null => None,
            _ => unreachable!(),
        };
        let mut want: Vec<Vec<Option<i64>>> = vec![Vec::new(); 4];
        for (av, bv) in &rows {
            let (a, b) = (val(av), val(bv));
            want[0].push(a.zip(b).map(|(a, b)| a + b));
            want[1].push(a.map(|a| a * 2));
            want[2].push(a.zip(b).map(|(a, b)| (a + 1) * b));
            want[3].push(a.zip(b).map(|(a, b)| a - b));
        }
        assert_eq!(got, want);
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
