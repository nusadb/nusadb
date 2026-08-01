//! Query plans.
//!
//! This module owns two intermediate representations:
//!
//! - [`LogicalPlan`] — the **analyzer's** output: a validated, type-checked
//!   tree where every table and column is resolved and every expression
//!   carries a concrete [`ColumnType`]. It is *what* to compute.
//! - [`PhysicalPlan`] — the **planner's** output (cost-based: index vs seq
//!   scan, join ordering). It is *how* to compute, and is consumed by the
//!   executor.
//!
//! [`plan`] performs the `LogicalPlan` → `PhysicalPlan` lowering. It is purely
//! structural today: with no index metadata, table statistics, or `JOIN`
//! support there is exactly one plan per query, so the cost-based choices
//! (index vs. sequential scan, join ordering) are deferred until that
//! infrastructure exists.

use nusadb_core::engine::{IndexDef, SequenceDef};
use nusadb_core::{ColumnType, TableSchema};

use crate::ast;

mod fold;
mod join;
mod lower;
mod plan_types;
mod pushdown;
#[allow(clippy::wildcard_imports)]
use join::*;
pub use lower::{plan, plan_select};

/// Match a single-table `WHERE` to an equality/range index access path, without cost statistics —
/// the crate-internal entry the executor's DML find-path uses to turn `WHERE pk = const` into an
/// [`PhysicalOperator::IndexScan`] point lookup. Returns `None` to keep a sequential scan. (A
/// `SELECT` lowers through the same [`lower::try_index_scan`] with stats for the cost gate.)
#[must_use]
pub(crate) fn try_point_get_index(
    table: &TableSchema,
    indexes: &[IndexMeta],
    filter: &TypedExpr,
) -> Option<PhysicalOperator> {
    lower::try_index_scan(table, indexes, filter, None)
}
#[allow(clippy::wildcard_imports)]
pub use plan_types::*;

/// Whether `plan` is a *bounded unique-key point lookup* the wire layer may run inline on the
/// reactor.
///
/// Admitted: a `SELECT` pipeline of only pass-through operators (`Project`/`Filter`/`Sort`/
/// `Limit`) over a single [`PhysicalOperator::IndexScan`] whose `unique_point` flag guarantees
/// **at most one row** — so the whole execution is a few B-tree pages and at most one output
/// row. Default-deny: every other operator (seq/values scans, joins, aggregation, windows,
/// set-returning projections, row locks, …) refuses.
///
/// This checks the PLAN's work bound only; the caller must pair it with the AST purity gate
/// ([`crate::ast::point_get_candidate`]) so per-row expressions are the same closed pure
/// built-in set the FROM-less inline gate admits (no subqueries/UDFs/SRFs).
#[must_use]
pub fn plan_is_inline_point_get(plan: &PhysicalPlan) -> bool {
    fn bounded(op: &PhysicalOperator) -> bool {
        match op {
            PhysicalOperator::Project { input, .. }
            | PhysicalOperator::Filter { input, .. }
            | PhysicalOperator::Sort { input, .. }
            | PhysicalOperator::Limit { input, .. } => bounded(input),
            PhysicalOperator::IndexScan { unique_point, .. } => *unique_point,
            _ => false,
        }
    }
    match plan {
        PhysicalPlan::Select(op, _) => bounded(op),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{LogicalPlan, PhysicalOperator, PhysicalPlan, plan};
    use crate::analyzer::{Catalog, analyze};
    use crate::error::Error;
    use nusadb_core::{ColumnDef, ColumnType, TableId, TableSchema};

    struct MockCatalog(Vec<TableSchema>);

    impl Catalog for MockCatalog {
        fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
            Ok(self.0.iter().find(|t| t.name == name).cloned())
        }

        // A large approximate count for the base table `t` (never analyzed, so `table_stats` stays
        // `None`) — exercises the vectorized-routing `est_scan_rows` fallback.
        fn approx_row_count(&self, table: &str) -> Result<u64, Error> {
            Ok(if table == "t" { 60_000 } else { 0 })
        }
    }

    /// Catalog holding `t(a INT NOT NULL, b TEXT)` and `s(a INT NOT NULL, c INT)`.
    fn catalog() -> MockCatalog {
        MockCatalog(vec![
            TableSchema {
                schema: "public".to_owned(),
                id: TableId(1),
                name: "t".to_owned(),
                columns: vec![
                    ColumnDef {
                        name: "a".to_owned(),
                        ty: ColumnType::Int,
                        nullable: false,
                    },
                    ColumnDef {
                        name: "b".to_owned(),
                        ty: ColumnType::Text,
                        nullable: true,
                    },
                ],
            },
            TableSchema {
                schema: "public".to_owned(),
                id: TableId(2),
                name: "s".to_owned(),
                columns: vec![
                    ColumnDef {
                        name: "a".to_owned(),
                        ty: ColumnType::Int,
                        nullable: false,
                    },
                    ColumnDef {
                        name: "c".to_owned(),
                        ty: ColumnType::Int,
                        nullable: true,
                    },
                ],
            },
        ])
    }

    /// Parse → analyze → plan.
    fn physical(sql: &str) -> PhysicalPlan {
        let stmt = crate::parser::parse(sql).expect("parse should succeed");
        let logical: LogicalPlan = analyze(stmt, &catalog()).expect("analysis should succeed");
        plan(logical)
    }

    /// The root operator of a planned `SELECT`.
    fn select_op(sql: &str) -> PhysicalOperator {
        match physical(sql) {
            PhysicalPlan::Select(op, _) => op,
            other => panic!("expected a Select plan, got {other:?}"),
        }
    }

    #[test]
    fn select_lowers_to_project_over_seqscan() {
        let PhysicalOperator::Project { input, columns } = select_op("SELECT * FROM t") else {
            panic!("expected Project at the root");
        };
        assert_eq!(columns.len(), 2);
        assert!(matches!(*input, PhysicalOperator::SeqScan { .. }));
    }

    #[test]
    fn select_without_from_scans_one_row() {
        let PhysicalOperator::Project { input, .. } = select_op("SELECT 1") else {
            panic!("expected Project at the root");
        };
        assert!(matches!(*input, PhysicalOperator::OneRow));
    }

    #[test]
    fn est_scan_rows_falls_back_to_the_approximate_row_count_when_unanalyzed() {
        // `t` has no ANALYZE stats but the catalog reports a 60k approximate count, so the
        // vectorized-routing estimate uses it (a large un-analyzed table still vectorizes).
        let PhysicalPlan::Select(_, est) = physical("SELECT * FROM t") else {
            panic!("expected a Select plan");
        };
        assert_eq!(est, Some(60_000));
        // A single-row source (no base table) has no estimate.
        let PhysicalPlan::Select(_, est) = physical("SELECT 1") else {
            panic!("expected a Select plan");
        };
        assert_eq!(est, None);
    }

    #[test]
    fn where_clause_inserts_filter() {
        let PhysicalOperator::Project { input, .. } = select_op("SELECT * FROM t WHERE a > 0")
        else {
            panic!("expected Project at the root");
        };
        assert!(matches!(*input, PhysicalOperator::Filter { .. }));
    }

    #[test]
    fn order_by_inserts_sort_below_project() {
        let PhysicalOperator::Project { input, .. } = select_op("SELECT * FROM t ORDER BY a")
        else {
            panic!("expected Project at the root");
        };
        assert!(matches!(*input, PhysicalOperator::Sort { .. }));
    }

    #[test]
    fn limit_inserts_limit_at_the_root() {
        let PhysicalOperator::Limit {
            input,
            count,
            offset,
        } = select_op("SELECT * FROM t LIMIT 3")
        else {
            panic!("expected Limit at the root");
        };
        assert_eq!(count, 3);
        assert_eq!(offset, 0);
        assert!(matches!(*input, PhysicalOperator::Project { .. }));
    }

    #[test]
    fn offset_without_limit_lowers_to_an_unbounded_limit() {
        let PhysicalOperator::Limit { count, offset, .. } = select_op("SELECT * FROM t OFFSET 5")
        else {
            panic!("expected a Limit at the root for a bare OFFSET");
        };
        assert_eq!(offset, 5);
        assert_eq!(
            count,
            u64::MAX,
            "no LIMIT → unbounded count, only the prefix is skipped"
        );
    }

    #[test]
    fn limit_with_offset_carries_both() {
        let PhysicalOperator::Limit { count, offset, .. } =
            select_op("SELECT * FROM t LIMIT 3 OFFSET 2")
        else {
            panic!("expected a Limit at the root");
        };
        assert_eq!(count, 3);
        assert_eq!(offset, 2);
    }

    #[test]
    fn full_pipeline_nesting_order() {
        // Limit → Project → Sort → Filter → SeqScan
        let PhysicalOperator::Limit { input, .. } =
            select_op("SELECT * FROM t WHERE a > 0 ORDER BY a LIMIT 5")
        else {
            panic!("expected Limit");
        };
        let PhysicalOperator::Project { input, .. } = *input else {
            panic!("expected Project");
        };
        let PhysicalOperator::Sort { input, .. } = *input else {
            panic!("expected Sort");
        };
        let PhysicalOperator::Filter { input, .. } = *input else {
            panic!("expected Filter");
        };
        assert!(matches!(*input, PhysicalOperator::SeqScan { .. }));
    }

    #[test]
    fn equi_join_lowers_to_hash_join() {
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a = s.a")
        else {
            panic!("expected Project at the root");
        };
        assert!(matches!(*input, PhysicalOperator::HashJoin { .. }));
    }

    #[test]
    fn non_equi_join_uses_nested_loop() {
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a > s.a")
        else {
            panic!("expected Project at the root");
        };
        assert!(matches!(*input, PhysicalOperator::NestedLoopJoin { .. }));
    }

    #[test]
    fn inner_join_pushes_single_side_on_conjunct_below_the_join() {
        // A right-only `ON` conjunct (`s.c > 0`) on an INNER join is pushed down
        // to a Filter on the right input, so it shrinks that side before the join instead of
        // staying in the residual and being checked on every emitted pair. The equi-key stays,
        // and with no cross-side residual left the join's residual is `None`.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a = s.a AND s.c > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin {
            keys,
            residual,
            right,
            ..
        } = *input
        else {
            panic!("expected HashJoin");
        };
        assert_eq!(keys.len(), 1);
        assert!(
            residual.is_none(),
            "the single-side `s.c > 0` is pushed down, leaving no residual"
        );
        assert!(
            matches!(*right, PhysicalOperator::Filter { .. }),
            "`s.c > 0` becomes a Filter on the right input"
        );
    }

    #[test]
    fn inner_join_keeps_cross_side_on_conjunct_as_residual() {
        // A conjunct referencing BOTH sides (`t.a < s.c`) cannot be pushed to either input; it
        // stays as the join residual. The single-side `t.a > 0` is pushed to the left instead.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a = s.a AND t.a < s.c AND t.a > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin { residual, left, .. } = *input else {
            panic!("expected HashJoin");
        };
        assert!(
            residual.is_some(),
            "the cross-side `t.a < s.c` stays residual"
        );
        assert!(
            matches!(*left, PhysicalOperator::Filter { .. }),
            "`t.a > 0` is pushed to the left input"
        );
    }

    #[test]
    fn left_join_does_not_push_on_conjuncts_down() {
        // An outer join must NOT push a single-side `ON` conjunct down: dropping a preserved-side
        // row (or a candidate on the NULL-extended side) would change which rows are emitted /
        // NULL-padded. The condition stays in the residual.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t LEFT JOIN s ON t.a = s.a AND s.c > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin {
            residual, right, ..
        } = *input
        else {
            panic!("expected HashJoin");
        };
        assert!(
            residual.is_some(),
            "a LEFT join keeps the single-side condition as residual"
        );
        assert!(
            matches!(*right, PhysicalOperator::SeqScan { .. }),
            "the right input is not wrapped in a pushed-down Filter"
        );
    }

    #[test]
    fn create_table_passes_through() {
        assert!(matches!(
            physical("CREATE TABLE u (x INT)"),
            PhysicalPlan::CreateTable(_),
        ));
    }

    #[test]
    fn drop_table_passes_through() {
        assert!(matches!(
            physical("DROP TABLE t"),
            PhysicalPlan::DropTable(_),
        ));
    }

    #[test]
    fn insert_passes_through() {
        assert!(matches!(
            physical("INSERT INTO t VALUES (1, 'x')"),
            PhysicalPlan::Insert(_),
        ));
    }

    #[test]
    fn update_passes_through() {
        assert!(matches!(
            physical("UPDATE t SET a = 1"),
            PhysicalPlan::Update(_),
        ));
    }

    #[test]
    fn delete_passes_through() {
        assert!(matches!(physical("DELETE FROM t"), PhysicalPlan::Delete(_),));
    }

    #[test]
    fn vacuum_lowers_to_vacuum_plan() {
        assert!(matches!(physical("VACUUM"), PhysicalPlan::Vacuum(_)));
    }

    #[test]
    fn select_distinct_inserts_distinct_above_project() {
        let PhysicalOperator::Distinct { input } = select_op("SELECT DISTINCT a FROM t") else {
            panic!("expected Distinct at the root");
        };
        assert!(matches!(*input, PhysicalOperator::Project { .. }));
    }

    #[test]
    fn select_distinct_with_limit_nests_distinct_below_limit() {
        let PhysicalOperator::Limit { input, .. } = select_op("SELECT DISTINCT a FROM t LIMIT 5")
        else {
            panic!("expected Limit at the root");
        };
        assert!(matches!(*input, PhysicalOperator::Distinct { .. }));
    }

    // === IndexScan selection =====================================

    use crate::analyzer::IndexInfo;
    use std::ops::Bound;

    /// Like [`catalog`], but reports a single-column index `t_a_idx` on `t(a)`; the second
    /// field is the reported index's `unique` flag.
    struct IndexedCatalog(MockCatalog, bool);

    impl Catalog for IndexedCatalog {
        fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, Error> {
            self.0.lookup_table(name)
        }

        fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, Error> {
            Ok(if name == "t" {
                vec![IndexInfo {
                    name: "t_a_idx".to_owned(),
                    columns: vec!["a".to_owned()],
                    unique: self.1,
                }]
            } else {
                Vec::new()
            })
        }
    }

    /// Root operator of a `SELECT` planned against the index-aware catalog; `unique` sets the
    /// reported index's uniqueness (the point-get gate's input).
    fn indexed_select_op_unique(sql: &str, unique: bool) -> PhysicalOperator {
        let stmt = crate::parser::parse(sql).expect("parse should succeed");
        let logical =
            analyze(stmt, &IndexedCatalog(catalog(), unique)).expect("analysis should succeed");
        match plan(logical) {
            PhysicalPlan::Select(op, _) => op,
            other => panic!("expected a Select plan, got {other:?}"),
        }
    }

    /// Root operator of a `SELECT` planned against the (non-unique) index-aware catalog.
    fn indexed_select_op(sql: &str) -> PhysicalOperator {
        indexed_select_op_unique(sql, false)
    }

    #[test]
    fn equality_on_indexed_column_uses_index_scan_under_filter() {
        // Project → Filter (kept) → IndexScan with a point bound.
        let PhysicalOperator::Project { input, .. } =
            indexed_select_op("SELECT * FROM t WHERE a = 5")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::Filter { input, .. } = *input else {
            panic!("expected the full WHERE filter to be kept above the index scan");
        };
        let PhysicalOperator::IndexScan { index, lo, hi, .. } = *input else {
            panic!("expected an IndexScan base");
        };
        assert_eq!(index, "t_a_idx");
        assert_eq!(lo, Bound::Included(vec![crate::ast::Value::Int(5)]));
        assert_eq!(hi, Bound::Included(vec![crate::ast::Value::Int(5)]));
    }

    #[test]
    fn range_on_indexed_column_uses_index_scan() {
        let PhysicalOperator::Project { input, .. } =
            indexed_select_op("SELECT * FROM t WHERE a > 5")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::Filter { input, .. } = *input else {
            panic!("expected Filter");
        };
        let PhysicalOperator::IndexScan { lo, hi, .. } = *input else {
            panic!("expected an IndexScan base");
        };
        assert_eq!(lo, Bound::Excluded(vec![crate::ast::Value::Int(5)]));
        assert_eq!(hi, Bound::Unbounded);
    }

    #[test]
    fn predicate_on_unindexed_column_keeps_seqscan() {
        // `b` has no index, so the base stays a SeqScan.
        let PhysicalOperator::Project { input, .. } =
            indexed_select_op("SELECT * FROM t WHERE b = 'x'")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::Filter { input, .. } = *input else {
            panic!("expected Filter");
        };
        assert!(matches!(*input, PhysicalOperator::SeqScan { .. }));
    }

    /// Inline point-get plan-shape pins: `plan_is_inline_point_get` admits exactly an equality bound
    /// on a UNIQUE single-column index (at most one row) under pass-through operators, and
    /// denies a non-unique index, a range bound, and a `SeqScan` — the bound that keeps inline
    /// reactor execution bounded.
    #[test]
    fn plan_shape_gate_admits_only_unique_point_lookup() {
        use super::plan_is_inline_point_get;
        let planned = |sql: &str, unique: bool| {
            let stmt = crate::parser::parse(sql).expect("parse should succeed");
            let logical =
                analyze(stmt, &IndexedCatalog(catalog(), unique)).expect("analysis should succeed");
            plan(logical)
        };
        // Equality on the UNIQUE index → admitted (also under ORDER BY / LIMIT wrappers).
        assert!(plan_is_inline_point_get(&planned(
            "SELECT * FROM t WHERE a = 5",
            true
        )));
        assert!(plan_is_inline_point_get(&planned(
            "SELECT b FROM t WHERE a = 5 AND b = 'x' ORDER BY b LIMIT 1",
            true
        )));
        // Same query, non-unique index → the scan may match many rows → denied.
        assert!(!plan_is_inline_point_get(&planned(
            "SELECT * FROM t WHERE a = 5",
            false
        )));
        // Range bound on the unique index → not a point → denied.
        assert!(!plan_is_inline_point_get(&planned(
            "SELECT * FROM t WHERE a > 5",
            true
        )));
        // Unindexed predicate → SeqScan → denied.
        assert!(!plan_is_inline_point_get(&planned(
            "SELECT * FROM t WHERE b = 'x'",
            true
        )));
        // And the flag itself is pinned on the operator: unique-eq sets it, non-unique does not.
        let PhysicalOperator::Project { input, .. } =
            indexed_select_op_unique("SELECT * FROM t WHERE a = 5", true)
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::Filter { input, .. } = *input else {
            panic!("expected Filter");
        };
        assert!(matches!(
            *input,
            PhysicalOperator::IndexScan {
                unique_point: true,
                ..
            }
        ));
    }

    #[test]
    fn join_query_pushes_base_predicate_and_keeps_seqscan() {
        // `WHERE t.a = 5` references only the base table `t`, so predicate pushdown moves it
        // below the join onto `t`'s scan — no Filter remains above the join. The base stays a
        // SeqScan; v1 never swaps a join base for an index scan, even though `a` is indexed.
        let PhysicalOperator::Project { input, .. } =
            indexed_select_op("SELECT * FROM t JOIN s ON t.a = s.a WHERE t.a = 5")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin { left, .. } = *input else {
            panic!(
                "expected HashJoin directly under Project (pushed-down predicate leaves no Filter)"
            );
        };
        // The pushed predicate wraps the join's left (base) input, which is still a SeqScan.
        let PhysicalOperator::Filter { input: scan, .. } = *left else {
            panic!("expected the pushed Filter on the join's left input");
        };
        assert!(
            matches!(*scan, PhysicalOperator::SeqScan { .. }),
            "join base must not be replaced by an index scan in v1"
        );
    }

    #[test]
    fn predicate_pushdown_pushes_each_single_table_conjunct_onto_its_own_input() {
        // INNER join: `t.b = 'x'` → left (base) input, `s.c > 0` → right input (two-sided pushdown).
        // Both single-table conjuncts push, so no residual Filter remains above the join.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a = s.a WHERE t.b = 'x' AND s.c > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin { left, right, .. } = *input else {
            panic!("expected HashJoin directly under Project — both conjuncts pushed, no Filter");
        };
        let PhysicalOperator::Filter { input: lscan, .. } = *left else {
            panic!("expected the base conjunct pushed onto the left input");
        };
        assert!(matches!(*lscan, PhysicalOperator::SeqScan { .. }));
        let PhysicalOperator::Filter { input: rscan, .. } = *right else {
            panic!("expected the right-only conjunct pushed onto the right input");
        };
        assert!(matches!(*rscan, PhysicalOperator::SeqScan { .. }));
    }

    #[test]
    fn predicate_pushdown_pushes_a_right_only_predicate_onto_an_inner_join_right_input() {
        // Even with no base-only conjunct, an INNER join's right-only `s.c > 0` is pushed onto the
        // right input (two-sided pushdown); the base stays a bare SeqScan and no residual Filter
        // remains above the join.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t JOIN s ON t.a = s.a WHERE s.c > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::HashJoin { left, right, .. } = *input else {
            panic!("expected HashJoin directly under Project");
        };
        assert!(
            matches!(*left, PhysicalOperator::SeqScan { .. }),
            "nothing is pushed onto the base",
        );
        assert!(
            matches!(*right, PhysicalOperator::Filter { .. }),
            "the right-only conjunct is pushed onto the right input",
        );
    }

    #[test]
    fn predicate_pushdown_keeps_a_right_predicate_above_a_left_join() {
        // SOUNDNESS: a LEFT join's right side is NULL-extended, so a WHERE predicate on it must NOT
        // be pushed below the join — that would change which rows survive. It stays in the residual
        // Filter above the join, and the right input remains a bare SeqScan.
        let PhysicalOperator::Project { input, .. } =
            select_op("SELECT * FROM t LEFT JOIN s ON t.a = s.a WHERE s.c > 0")
        else {
            panic!("expected Project at the root");
        };
        let PhysicalOperator::Filter { input: join, .. } = *input else {
            panic!("expected a residual Filter above the LEFT join");
        };
        let PhysicalOperator::HashJoin { left, right, .. } = *join else {
            panic!("expected HashJoin under the residual Filter");
        };
        assert!(matches!(*left, PhysicalOperator::SeqScan { .. }));
        assert!(
            matches!(*right, PhysicalOperator::SeqScan { .. }),
            "a LEFT join's right input must stay unfiltered (its rows are NULL-extended)",
        );
    }

    #[test]
    fn predicate_pushdown_keeps_right_predicate_above_a_downstream_full_join() {
        // SOUNDNESS (multi-join chain): `s.c > 0` references the right table of the INNER join, but a
        // DOWNSTREAM FULL join null-extends the whole accumulated left — which by then INCLUDES `s` —
        // so the predicate must NOT be pushed below the inner join; it stays in the residual Filter
        // above the FULL join. (Pushing it below would drop the FULL join's NULL-extended rows that
        // the predicate is supposed to filter, changing the result.)
        let PhysicalOperator::Project { input, .. } = select_op(
            "SELECT * FROM t JOIN s ON t.a = s.a FULL JOIN t t2 ON t.a = t2.a WHERE s.c > 0",
        ) else {
            panic!("expected Project at the root");
        };
        assert!(
            matches!(*input, PhysicalOperator::Filter { .. }),
            "a right-side predicate must remain above a downstream FULL join, not be pushed below it",
        );
    }

    /// The `top_n` cap of the first `Sort` in a planned `SELECT` (walking down the root operators).
    /// `None` = the sort is a full sort, or (for these tests, all of which have an `ORDER BY`)
    /// there is no `Sort`.
    fn sort_top_n(sql: &str) -> Option<u64> {
        fn find(op: &PhysicalOperator) -> Option<u64> {
            match op {
                PhysicalOperator::Sort { top_n, .. } => *top_n,
                PhysicalOperator::Project { input, .. }
                | PhysicalOperator::Distinct { input }
                | PhysicalOperator::DistinctOn { input, .. }
                | PhysicalOperator::Limit { input, .. }
                | PhysicalOperator::ProjectSet { input, .. } => find(input),
                _ => None,
            }
        }
        find(&select_op(sql))
    }

    /// A plain `LIMIT` (optionally with `OFFSET`) over an `ORDER BY` gives
    /// the `Sort` a top-N cap of `offset + limit`, so the executor selects the first rows without a
    /// full sort; a `DISTINCT`/set-returning projection between the sort and the limit, `WITH TIES`,
    /// an `OFFSET` with no `LIMIT`, or a cap beyond the bound leaves it a full sort (`None`).
    #[test]
    fn order_by_limit_gets_top_n_cap_when_row_count_is_preserved() {
        assert_eq!(sort_top_n("SELECT a FROM t ORDER BY a LIMIT 5"), Some(5));
        assert_eq!(
            sort_top_n("SELECT a FROM t ORDER BY a DESC LIMIT 5 OFFSET 10"),
            Some(15)
        );
        assert_eq!(sort_top_n("SELECT a FROM t ORDER BY a LIMIT 0"), Some(0));
        // No LIMIT bounding the sort → full sort.
        assert_eq!(sort_top_n("SELECT a FROM t ORDER BY a"), None);
        assert_eq!(sort_top_n("SELECT a FROM t ORDER BY a OFFSET 5"), None);
        // A DISTINCT dedup sits between the sort and the limit → the sort must emit every row.
        assert_eq!(
            sort_top_n("SELECT DISTINCT a FROM t ORDER BY a LIMIT 5"),
            None
        );
        // FETCH … WITH TIES is the sort's own `limit_ties`, not a top-N cap.
        assert_eq!(
            sort_top_n("SELECT a FROM t ORDER BY a FETCH FIRST 5 ROWS WITH TIES"),
            None
        );
        // A cap beyond the bound falls back to the ordinary (spill-aware) full sort.
        assert_eq!(sort_top_n("SELECT a FROM t ORDER BY a LIMIT 100000"), None);
    }

    /// The `top_n` cap of the `Window` in a planned `SELECT`, or `None` if there is no `Window`.
    fn window_top_n(sql: &str) -> Option<u64> {
        fn find(op: &PhysicalOperator) -> Option<u64> {
            match op {
                PhysicalOperator::Window { top_n, .. } => *top_n,
                PhysicalOperator::Project { input, .. }
                | PhysicalOperator::Distinct { input }
                | PhysicalOperator::DistinctOn { input, .. }
                | PhysicalOperator::Limit { input, .. }
                | PhysicalOperator::Sort { input, .. }
                | PhysicalOperator::ProjectSet { input, .. } => find(input),
                _ => None,
            }
        }
        find(&select_op(sql))
    }

    /// A ranking window over a single partition, whose order equals the
    /// outer `ORDER BY … LIMIT`, gets a top-N cap so it computes only the first rows (bounded
    /// memory) instead of materializing the whole input. A partition, a non-ranking function, an
    /// order that does not match the outer `ORDER BY`, or no bounding `LIMIT` leaves it `None`.
    #[test]
    fn ranking_window_gets_top_n_cap_only_when_safe() {
        assert_eq!(
            window_top_n("SELECT a, row_number() OVER (ORDER BY a) FROM t ORDER BY a LIMIT 5"),
            Some(5)
        );
        assert_eq!(
            window_top_n("SELECT a, rank() OVER (ORDER BY a) FROM t ORDER BY a LIMIT 3 OFFSET 2"),
            Some(5)
        );
        // A PARTITION BY splits the ordering — the first m overall are not the first m per partition.
        assert_eq!(
            window_top_n(
                "SELECT a, row_number() OVER (PARTITION BY b ORDER BY a) FROM t ORDER BY a LIMIT 5"
            ),
            None
        );
        // A navigation function (LAG) can look outside the first m rows.
        assert_eq!(
            window_top_n("SELECT a, lag(a) OVER (ORDER BY a) FROM t ORDER BY a LIMIT 5"),
            None
        );
        // The outer ORDER BY must match the window order.
        assert_eq!(
            window_top_n("SELECT a, row_number() OVER (ORDER BY a) FROM t ORDER BY b LIMIT 5"),
            None
        );
        // No outer ORDER BY at all → the LIMIT does not bound the window order.
        assert_eq!(
            window_top_n("SELECT a, row_number() OVER (ORDER BY a) FROM t LIMIT 5"),
            None
        );
        // No LIMIT.
        assert_eq!(
            window_top_n("SELECT a, row_number() OVER (ORDER BY a) FROM t ORDER BY a"),
            None
        );
    }
}
