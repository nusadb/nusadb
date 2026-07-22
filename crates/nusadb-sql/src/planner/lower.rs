//! Logical-to-physical lowering: the `plan()` entry point and the SELECT/set-op pipeline builder.
//!
//! Split verbatim out of `planner/mod.rs` (ADR 007). Resolves siblings via `use super::*`.
#![allow(clippy::wildcard_imports)]

use std::ops::Bound;

use super::*;

/// Lower a validated [`LogicalPlan`] into an executable [`PhysicalPlan`].
///
/// Infallible: the analyzer has already resolved and type-checked everything,
/// so the lowering is a pure structural transformation.
#[must_use]
pub fn plan(logical: LogicalPlan) -> PhysicalPlan {
    match logical {
        LogicalPlan::LockTable { tables, mode } => PhysicalPlan::LockTable { tables, mode },
        LogicalPlan::Prepare {
            name,
            statement,
            param_count,
        } => PhysicalPlan::Prepare {
            name,
            statement,
            param_count,
        },
        LogicalPlan::Execute { name, args } => PhysicalPlan::Execute { name, args },
        LogicalPlan::Deallocate(target) => PhysicalPlan::Deallocate(target),
        LogicalPlan::CreateTable(p) => PhysicalPlan::CreateTable(p),
        LogicalPlan::CreateTableAs(p) => PhysicalPlan::CreateTableAs(PhysicalCreateTableAs {
            name: p.name,
            columns: p.columns,
            body: Box::new(plan_select(*p.body)),
            if_not_exists: p.if_not_exists,
        }),
        LogicalPlan::DropTable(p) => PhysicalPlan::DropTable(p),
        LogicalPlan::Batch(children) => {
            PhysicalPlan::Batch(children.into_iter().map(plan).collect())
        },
        LogicalPlan::CreateMaterializedView(p) => {
            PhysicalPlan::CreateMaterializedView(PhysicalMaterializedView {
                name: p.name,
                or_replace: p.or_replace,
                if_not_exists: p.if_not_exists,
                columns: p.columns,
                body: Box::new(plan_select(*p.body)),
                definition_sql: p.definition_sql,
                ivm_base: p.ivm_base,
            })
        },
        LogicalPlan::CreateView(p) => PhysicalPlan::CreateView(p),
        LogicalPlan::DropView(p) => PhysicalPlan::DropView(p),
        LogicalPlan::CreateEnum(p) => PhysicalPlan::CreateEnum(p),
        LogicalPlan::DropType(p) => PhysicalPlan::DropType(p),
        LogicalPlan::CreateTrigger(p) => PhysicalPlan::CreateTrigger(p),
        LogicalPlan::DropTrigger(p) => PhysicalPlan::DropTrigger(p),
        LogicalPlan::AlterTrigger(p) => PhysicalPlan::AlterTrigger(p),
        LogicalPlan::CreateProcedure(p) => PhysicalPlan::CreateProcedure(p),
        LogicalPlan::DropProcedure(p) => PhysicalPlan::DropProcedure(p),
        LogicalPlan::Call(p) => PhysicalPlan::Call(p),
        LogicalPlan::CreateFunction(p) => PhysicalPlan::CreateFunction(p),
        LogicalPlan::DropFunction(p) => PhysicalPlan::DropFunction(p),
        LogicalPlan::RefreshMaterializedView(name) => PhysicalPlan::RefreshMaterializedView(name),
        LogicalPlan::CreatePolicy(p) => PhysicalPlan::CreatePolicy(p),
        LogicalPlan::DropPolicy(p) => PhysicalPlan::DropPolicy(p),
        LogicalPlan::AlterTable(p) => PhysicalPlan::AlterTable(p),
        LogicalPlan::Insert(p) => PhysicalPlan::Insert(p),
        LogicalPlan::Select(p) => {
            // Carry the single-table scanned-row estimate so the executor can route to the
            // vectorized path without re-fetching. Prefer the exact ANALYZE row count; fall back to
            // the engine's O(1) approximate count when the table was never analyzed, so a large
            // un-analyzed table still vectorizes.
            let est_scan_rows = p
                .table_stats
                .as_ref()
                .map(|s| s.row_count)
                .or(p.approx_scan_rows);
            PhysicalPlan::Select(plan_select(*p), est_scan_rows)
        },
        LogicalPlan::Update(p) => PhysicalPlan::Update(p),
        LogicalPlan::Delete(p) => PhysicalPlan::Delete(p),
        LogicalPlan::Merge(p) => PhysicalPlan::Merge(p),
        LogicalPlan::Explain(inner, options) => {
            PhysicalPlan::Explain(Box::new(plan(*inner)), options)
        },
        LogicalPlan::BeginTransaction(c) => PhysicalPlan::BeginTransaction(c),
        LogicalPlan::Commit => PhysicalPlan::Commit,
        LogicalPlan::Rollback => PhysicalPlan::Rollback,
        LogicalPlan::SetTransaction(c) => PhysicalPlan::SetTransaction(c),
        LogicalPlan::Savepoint(name) => PhysicalPlan::Savepoint(name),
        LogicalPlan::RollbackToSavepoint(name) => PhysicalPlan::RollbackToSavepoint(name),
        LogicalPlan::ReleaseSavepoint(name) => PhysicalPlan::ReleaseSavepoint(name),
        LogicalPlan::SetVariable { name, value } => PhysicalPlan::SetVariable { name, value },
        LogicalPlan::ShowVariable(name) => PhysicalPlan::ShowVariable(name),
        LogicalPlan::ShowTables => PhysicalPlan::ShowTables,
        LogicalPlan::ShowColumns(schema) => PhysicalPlan::ShowColumns(schema),
        LogicalPlan::Vacuum(options) => PhysicalPlan::Vacuum(options),
        LogicalPlan::Reindex => PhysicalPlan::Reindex,
        LogicalPlan::Analyze(p) => PhysicalPlan::Analyze(p),
        LogicalPlan::Comment(p) => PhysicalPlan::Comment(p),
        LogicalPlan::CreateSchema(p) => PhysicalPlan::CreateSchema(p),
        LogicalPlan::DropSchema(p) => PhysicalPlan::DropSchema(p),
        LogicalPlan::CreateDatabase(p) => PhysicalPlan::CreateDatabase(p),
        LogicalPlan::AlterDatabase(p) => PhysicalPlan::AlterDatabase(p),
        LogicalPlan::DropDatabase(p) => PhysicalPlan::DropDatabase(p),
        LogicalPlan::CreateSequence(p) => PhysicalPlan::CreateSequence(p),
        LogicalPlan::DropSequence(p) => PhysicalPlan::DropSequence(p),
        LogicalPlan::CreateIndex(p) => PhysicalPlan::CreateIndex(p),
        LogicalPlan::DropIndex(p) => PhysicalPlan::DropIndex(p),
        LogicalPlan::SetOperation(p) => PhysicalPlan::SetOperation(PhysicalSetOp {
            tree: lower_set_tree(p.tree),
            columns: p.columns,
            column_types: p.column_types,
            order_by: p.order_by,
            limit: p.limit,
        }),
    }
}

/// Lower a set-operation tree's `SELECT` leaves to operator pipelines.
pub(super) fn lower_set_tree(tree: SetOpTree<SelectPlan>) -> SetOpTree<PhysicalOperator> {
    match tree {
        SetOpTree::Leaf(select) => SetOpTree::Leaf(Box::new(plan_select(*select))),
        SetOpTree::Node {
            op,
            all,
            left,
            right,
        } => SetOpTree::Node {
            op,
            all,
            left: Box::new(lower_set_tree(*left)),
            right: Box::new(lower_set_tree(*right)),
        },
    }
}

/// Lower a `SELECT` into its operator pipeline:
/// `scan → [where] → [aggregate] → [having] → [sort] → project → [limit]`.
///
/// `Sort` is placed below `Project`. For a plain `SELECT` its keys reference
/// source-table columns; for an aggregated `SELECT` they reference the
/// synthesized post-aggregation row (so `Sort` sits above the aggregate).
#[allow(
    clippy::too_many_lines,
    reason = "flat clause-by-clause pipeline assembly; length tracks the operator set, not branching"
)]
pub fn plan_select(select: SelectPlan) -> PhysicalOperator {
    // Constant-fold the clause expressions first: `WHERE 1=1 AND p` → `WHERE p`, a constant
    // predicate collapses, etc. — so every later step (vector-knn detection, lowering) sees the
    // simplified form.
    let mut select = fold::fold_select(select);
    // `ORDER BY col <=> q LIMIT k` over a plain single-table scan lowers to a `VectorKnn`,
    // replacing the Sort+scan: the executor uses an HNSW index when one is declared, else an exact
    // scan. Detected up front, before the clause fields are consumed below.
    // `FETCH FIRST n ROWS WITH TIES` is applied by the `Sort` operator (it needs
    // the ORDER BY keys to find the trailing peers), so suppress the vector-knn rewrite — which would
    // replace the `Sort` — when a tie cap is present, and capture the cap before the fields are
    // consumed. The analyzer guarantees an ORDER BY exists, so the `Sort` branch below is taken.
    let ties_cap = if select.limit_with_ties {
        select.limit.map(|count| TiesLimit {
            offset: select.offset.unwrap_or(0),
            count,
        })
    } else {
        None
    };
    let has_ties = ties_cap.is_some();
    // Limit-aware top-N cap for the `Sort`, captured before any clause
    // field is consumed below — it inspects `limit`/`offset`/`distinct`/`distinct_on`/`projection`.
    let top_n_hint = top_n_cap(&select);
    // Limit-aware top-N cap for a ranking `Window`, likewise captured
    // before `windows`/`order_by` are consumed.
    let window_top_n_hint = window_top_n_cap(&select);
    let vector_knn = if has_ties {
        None
    } else {
        vector_knn_plan(&select)
    };
    // Recursive CTEs are lowered separately and wrapped around the finished body below.
    let recursive_ctes = select.recursive_ctes;
    // Data-modifying CTEs are likewise wrapped around the finished body below.
    let modifying_ctes = select.modifying_ctes;
    // `FOR UPDATE` / `FOR SHARE`: capture the lock target before the table/predicate are
    // consumed below, so the finished pipeline can be wrapped in a `LockRows` that locks each matched
    // base row. The analyzer only sets `row_lock` for the simple single-table shape.
    let row_lock = select.row_lock;
    let lock_table = select.table.clone();
    let lock_predicate = select.filter.clone();
    // Width of the row the running operator produces, tracked so each join can
    // NULL-pad the correct number of left columns for unmatched right rows.
    // The base source is a set operation (`(SELECT ... UNION ...) AS x`), inline `VALUES` rows, a
    // CTE's inlined pipeline, a real table's `SeqScan`, or `OneRow`.
    let values = std::mem::take(&mut select.values);
    let set_op_source = select.set_op_source.take();
    let (mut left_width, mut op) = if let Some(set_op) = set_op_source {
        // `(SELECT ... UNION ...) AS x` base: lower the set-op tree and run it as a row source,
        // exactly like a top-level set operation. Column count = the set-op's output column count.
        let set_op = *set_op;
        let width = set_op.columns.len();
        let physical = PhysicalSetOp {
            tree: lower_set_tree(set_op.tree),
            columns: set_op.columns,
            column_types: set_op.column_types,
            order_by: set_op.order_by,
            limit: set_op.limit,
        };
        (width, PhysicalOperator::SetOperation(Box::new(physical)))
    } else if !values.is_empty() {
        // `(VALUES ...) AS x` base: every row has the same width, so column count = the first row's
        // arity.
        let width = values.first().map_or(0, Vec::len);
        (width, PhysicalOperator::Values { rows: values })
    } else if let Some(cte) = select.from_cte {
        (cte.projection.len(), plan_select(*cte))
    } else {
        let width = select.table.as_ref().map_or(0, |t| t.columns.len());
        let op = match &select.table {
            // An `information_schema` view: emit an InfoSchemaScan that produces metadata
            // rows from the engine rather than going to storage. Detected by name — a real table or
            // CTE can never be named `information_schema.*` (the parser only mints that name for the
            // info-schema path), so this needs no reserved synthetic id.
            Some(table) if InfoSchemaView::from_full_name(&table.name).is_some() => {
                InfoSchemaView::from_full_name(&table.name).map_or_else(
                    || PhysicalOperator::SeqScan {
                        table: table.clone(),
                        columns: Vec::new(),
                    },
                    |view| PhysicalOperator::InfoSchemaScan { view },
                )
            },
            // Prefer an index scan when there are no joins (so the `WHERE` predicate references only
            // this table) and a predicate maps onto a single-column index. The full `WHERE`
            // filter is still applied above (it is never dropped), so the index only narrows the row
            // source — the result multiset is identical to a `SeqScan` + `Filter`.
            Some(table) => {
                // Cost-based plan choice: with ANALYZE stats, prefer the index only when it
                // is estimated cheaper than a sequential scan; without stats, keep the heuristic.
                let scan_stats = select
                    .table_stats
                    .as_ref()
                    .map(|st| crate::executor::cost::ScanStats::new(table, st));
                select
                    .joins
                    .is_empty()
                    .then(|| {
                        select.filter.as_ref().and_then(|f| {
                            try_index_scan(table, &select.indexes, f, scan_stats.as_ref())
                        })
                    })
                    .flatten()
                    .unwrap_or_else(|| PhysicalOperator::SeqScan {
                        table: table.clone(),
                        columns: Vec::new(),
                    })
            },
            None => PhysicalOperator::OneRow,
        };
        (width, op)
    };
    // Predicate pushdown: when joins follow, push each `WHERE` conjunct that references only
    // the base table's columns down onto the base scan, below the joins, so fewer rows enter the
    // join. Sound only when every join is base-preserving (Inner/Left/Cross): the base columns keep
    // ordinals `[0, base_width)` in the joined row and the base side is never NULL-extended, so a
    // base-only conjunct has the same truth value below or above the joins. A subquery-bearing
    // conjunct is left in place — it may correlate to a joined row, which exists only above the join.
    let base_width = left_width;
    let all_base_preserving = select.joins.iter().all(|j| {
        matches!(
            j.kind,
            ast::JoinKind::Inner | ast::JoinKind::Left | ast::JoinKind::Cross
        )
    });
    // Split `WHERE` into conjuncts up front so both the base scan AND each INNER/CROSS join's right
    // input can pull the single-input conjuncts that belong to them (two-sided predicate pushdown).
    // A subquery-bearing conjunct may correlate to a joined row — which exists only above the join —
    // so it is never pushed.
    let mut remaining: Vec<TypedExpr> = Vec::new();
    if let Some(filter) = select.filter.take() {
        split_conjuncts(filter, &mut remaining);
    }
    // Base side: a conjunct over only the base columns is sound to push below every base-preserving
    // join (the base keeps ordinals `[0, base_width)` and is never NULL-extended).
    if !select.joins.is_empty() && all_base_preserving {
        let (pushable, keep): (Vec<TypedExpr>, Vec<TypedExpr>) =
            std::mem::take(&mut remaining).into_iter().partition(|c| {
                !crate::executor::ops::contains_subquery(c)
                    && matches!(column_side(c, base_width), Side::Left)
            });
        if let Some(pushed) = rebuild_and(pushable) {
            op = PhysicalOperator::Filter {
                input: Box::new(op),
                predicate: pushed,
            };
        }
        remaining = keep;
    }
    // Left-deep join chain. Equi-joins lower to a hash join; anything else
    // falls back to a nested-loop join.
    // Join kinds captured up front so the right-side pushdown below can see whether a DOWNSTREAM
    // join null-extends the table it would push a filter onto.
    let join_kinds: Vec<ast::JoinKind> = select.joins.iter().map(|j| j.kind).collect();
    for (i, join) in select.joins.into_iter().enumerate() {
        let right_width = join.table.columns.len();
        // A LATERAL join input is re-executed per left row (its subquery correlates to the left), so
        // it lowers to a dependent `LateralJoin` rather than a materialized join (increment
        // 3c). A non-lateral derived input lowers to its inlined subquery pipeline; a named table to
        // a sequential scan.
        if join.lateral {
            // A lateral join input is always a derived table, so it always carries an inlined plan;
            // fall back defensively to a one-row source rather than panicking if that ever changes.
            let right = join
                .input_cte
                .map_or(PhysicalOperator::OneRow, |cte| plan_select(*cte));
            op = PhysicalOperator::LateralJoin {
                left: Box::new(op),
                right: Box::new(right),
                predicate: join.on,
                kind: join.kind,
                right_width,
            };
            left_width += right_width;
            continue;
        }
        let mut right = match join.input_cte {
            Some(cte) => plan_select(*cte),
            None => PhysicalOperator::SeqScan {
                table: join.table,
                columns: Vec::new(),
            },
        };
        // Right side: standard two-sided predicate pushdown. A `WHERE`
        // conjunct over only THIS join's right table is sound to push onto the right input when the
        // right side is never NULL-extended, so filtering it before the join cannot change the
        // result — and it shrinks the join's input (the un-pushed form kept `b.id<n` above the join
        // and blew the intermediate up to N×M). Two null-extension hazards forbid it: (1) THIS join
        // is LEFT/RIGHT/FULL — its own right side is null-extended; (2) a DOWNSTREAM RIGHT/FULL join
        // null-extends the whole accumulated left, which by then INCLUDES this right table, so the
        // conjunct must run above that outer join. Remap the conjunct's ordinals from the joined-row
        // space `[left_width, left_width+right_width)` down to the right input's own
        // `[0, right_width)`.
        let downstream_null_extends_left = join_kinds.get(i + 1..).is_some_and(|rest| {
            rest.iter()
                .any(|k| matches!(k, ast::JoinKind::Right | ast::JoinKind::Full))
        });
        if matches!(join.kind, ast::JoinKind::Inner | ast::JoinKind::Cross)
            && !downstream_null_extends_left
        {
            let (pushable, keep): (Vec<TypedExpr>, Vec<TypedExpr>) =
                std::mem::take(&mut remaining).into_iter().partition(|c| {
                    !crate::executor::ops::contains_subquery(c)
                        && matches!(column_side(c, left_width), Side::Right)
                        && matches!(column_side(c, left_width + right_width), Side::Left)
                });
            if let Some(mut pushed) = rebuild_and(pushable) {
                remap_columns(&mut pushed, left_width);
                right = PhysicalOperator::Filter {
                    input: Box::new(right),
                    predicate: pushed,
                };
            }
            remaining = keep;
        }
        op = build_join(
            op,
            right,
            join.on,
            join.kind,
            left_width,
            right_width,
            join.coalesce,
        );
        left_width += right_width;
    }
    // Whatever `WHERE` conjuncts could not be pushed to a single input (multi-input, subquery, or an
    // outer join's NULL-extended side) apply here, above the joins.
    if let Some(predicate) = rebuild_and(remaining) {
        op = PhysicalOperator::Filter {
            input: Box::new(op),
            predicate,
        };
    }
    // Aggregation folds the (WHERE-filtered) input stream. `GROUP BY` partitions
    // it into groups (one output row each); a bare aggregate is one global
    // group. The output row is `[group keys ++ aggregate results]`.
    if !select.grouping_sets.is_empty() {
        // Route every grouping-sets query here, even when the union of keys is empty
        // (e.g. `GROUPING SETS ((), ())`): the executor seeds a grand-total row per set,
        // so N empty sets correctly yield N rows rather than collapsing to one.
        op = PhysicalOperator::GroupingSetsAggregate {
            input: Box::new(op),
            group_keys: select.group_keys,
            grouping_sets: select.grouping_sets,
            calls: select.aggregates,
        };
    } else if !select.group_keys.is_empty() {
        op = PhysicalOperator::GroupAggregate {
            input: Box::new(op),
            group_keys: select.group_keys,
            calls: select.aggregates,
        };
    } else if !select.aggregates.is_empty() {
        op = PhysicalOperator::ScalarAggregate {
            input: Box::new(op),
            calls: select.aggregates,
        };
    }
    // `HAVING` filters the post-aggregation rows.
    if let Some(predicate) = select.having {
        op = PhysicalOperator::Filter {
            input: Box::new(op),
            predicate,
        };
    }
    // Window functions annotate the (WHERE-filtered, and — when the query aggregates — grouped and
    // HAVING-filtered) rows with appended columns before projection. The operator runs here, ABOVE
    // any aggregate: for a non-aggregated query its input is the filtered scan (columns at their
    // source ordinals); for an aggregated one its input is the post-aggregation row `[group keys ++
    // aggregates]`, which the analyzer rebased the windows' expressions and the projection's
    // window-column references onto. It appends one column per window after the input width.
    if !select.windows.is_empty() {
        op = PhysicalOperator::Window {
            input: Box::new(op),
            windows: select.windows,
            top_n: window_top_n_hint,
        };
    }
    if let Some((table, column_ordinal, query, k, filter)) = vector_knn {
        // Replace the `Sort` (and its scan, and the `WHERE` Filter if any) with a vector-index search
        // returning the k nearest matching rows in distance order; the outer `Limit`/`Project` still
        // apply unchanged. The filter is applied inside the search, so the separately-built
        // `Filter` operator above is discarded along with the scan.
        op = PhysicalOperator::VectorKnn {
            table,
            column_ordinal,
            query,
            k,
            filter,
        };
    } else if !select.order_by.is_empty() {
        // Limit-aware top-N: if a plain LIMIT bounds this sort and nothing
        // between the sort and that LIMIT changes the row count, only the first `offset + limit`
        // rows are ever consumed — let the executor select them without a full O(N log N) sort.
        op = PhysicalOperator::Sort {
            input: Box::new(op),
            keys: select.order_by,
            limit_ties: ties_cap,
            top_n: top_n_hint,
        };
    }
    // DISTINCT ON keeps the first source row per key tuple — it must run on the (sorted) source
    // rows, before projection, so its keys see source columns.
    if !select.distinct_on.is_empty() {
        op = PhysicalOperator::DistinctOn {
            input: Box::new(op),
            keys: select.distinct_on,
        };
    }
    // A projection holding a set-returning function expands rows, so it lowers to
    // `ProjectSet` rather than the row-preserving `Project`.
    let has_srf = select
        .projection
        .iter()
        .any(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }));
    op = if has_srf {
        PhysicalOperator::ProjectSet {
            input: Box::new(op),
            columns: select.projection,
            ordinality: select.ordinality,
        }
    } else {
        PhysicalOperator::Project {
            input: Box::new(op),
            columns: select.projection,
        }
    };
    // DISTINCT dedupes the projected output rows, so it sits above `Project`
    // (and below `Limit`, which caps the already-deduped stream).
    if select.distinct {
        op = PhysicalOperator::Distinct {
            input: Box::new(op),
        };
    }
    // `LIMIT` and/or `OFFSET` lower to one `Limit` operator. An `OFFSET` with no `LIMIT`
    // uses an unbounded count so it skips the prefix and emits the rest. When a `WITH TIES` cap is
    // present the `Sort` already applied the offset + count (plus the trailing ties), so no separate
    // `Limit` is emitted.
    if !has_ties && (select.limit.is_some() || select.offset.is_some()) {
        op = PhysicalOperator::Limit {
            input: Box::new(op),
            count: select.limit.unwrap_or(u64::MAX),
            offset: select.offset.unwrap_or(0),
        };
    }
    // `WITH RECURSIVE`: wrap the finished body so each CTE is materialized to a fixpoint and
    // bound to its synthetic table before the body (which scans those tables) runs.
    if !recursive_ctes.is_empty() {
        op = PhysicalOperator::WithRecursive {
            ctes: recursive_ctes
                .into_iter()
                .map(|def| PhysicalRecursiveCte {
                    id: def.id,
                    base: Box::new(plan_select(*def.base)),
                    recursive: Box::new(plan_select(*def.recursive)),
                    union_all: def.union_all,
                })
                .collect(),
            body: Box::new(op),
        };
    }
    // Data-modifying CTEs: wrap the body so each statement runs once and binds its
    // RETURNING rows to the synthetic table before the body (which reads them) runs.
    if !modifying_ctes.is_empty() {
        op = PhysicalOperator::WithModifying {
            ctes: modifying_ctes
                .into_iter()
                .map(|def| PhysicalModifyingCte {
                    id: def.id,
                    plan: plan(*def.plan),
                })
                .collect(),
            body: Box::new(op),
        };
    }
    // `FOR UPDATE` / `FOR SHARE`: wrap the finished pipeline so the executor locks every
    // matched base row before producing output. Only set for the validated single-table shape.
    if let (Some((mode, skip_locked)), Some(table)) = (row_lock, lock_table) {
        op = PhysicalOperator::LockRows {
            input: Box::new(op),
            table,
            predicate: lock_predicate,
            mode,
            skip_locked,
        };
    }
    // Projection pushdown: run last, on the finished tree, so a reference from any pipeline
    // layer is visible. A no-op unless the plan is the simple single-table shape — see `pushdown`.
    pushdown::pushdown_projection(&mut op);
    op
}

/// The largest `offset + limit` for which a `Sort` is given a top-N cap
/// Bounds the executor's retained set — beyond it, deep-pagination
/// queries fall back to the ordinary (spill-aware) full sort so the bounded pass never holds an
/// unreasonable number of rows in memory. Comfortably covers real pagination depths.
const TOP_N_CAP_LIMIT: u64 = 65_536;

/// The limit-aware top-N cap for a `Sort`: `Some(offset + limit)` when a
/// plain `LIMIT` bounds the sort's output and nothing between the sort and that `LIMIT` changes the
/// row count, else `None` (a full sort is required).
///
/// Excluded, because the sort must still emit every row: `FETCH … WITH TIES` (handled by the sort's
/// `limit_ties`), a `DISTINCT`/`DISTINCT ON` dedup or a set-returning projection that sits between
/// the sort and the `LIMIT`, an `OFFSET`-only query (no `LIMIT`, so nothing bounds the sort), and an
/// `offset + limit` beyond [`TOP_N_CAP_LIMIT`].
fn top_n_cap(select: &SelectPlan) -> Option<u64> {
    if select.limit_with_ties {
        return None;
    }
    let limit = select.limit?;
    if select.distinct || !select.distinct_on.is_empty() {
        return None;
    }
    let has_srf = select
        .projection
        .iter()
        .any(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }));
    if has_srf {
        return None;
    }
    let m = select.offset.unwrap_or(0).saturating_add(limit);
    (m <= TOP_N_CAP_LIMIT).then_some(m)
}

/// The limit-aware top-N cap for a ranking `Window`:
/// `Some(offset + limit)` when the outer `ORDER BY … LIMIT` provably needs only the first `m` rows
/// in the windows' shared order and the windows are safe to compute over just those rows, else
/// `None` (the full materializing computation).
///
/// Safe only when every window is a **ranking** function (`ROW_NUMBER`/`RANK`/`DENSE_RANK` — whose
/// value at position `k` depends only on rows at positions `≤ k`), over a **single partition** (no
/// `PARTITION BY`), with the **default frame**, all sharing one non-empty `ORDER BY`; and the outer
/// query has a plain bounding `LIMIT` (no `WITH TIES`/`DISTINCT`/set-returning projection) whose
/// `ORDER BY` **equals** that window order — so the first `m` rows by the window order are exactly
/// the first `m` the `LIMIT` selects. Any navigation/aggregate/distribution function, an explicit
/// frame, a partition, or a mismatched order falls back to `None`.
fn window_top_n_cap(select: &SelectPlan) -> Option<u64> {
    use crate::ast::WindowFunc as W;
    if select.limit_with_ties || select.distinct || !select.distinct_on.is_empty() {
        return None;
    }
    // A window over aggregated rows sits ABOVE the GROUP BY / scalar-aggregate stage, not a plain
    // ordered scan, so the top-N-into-window rewrite (which assumes the window reads the sorted input
    // directly) does not apply — compute the window fully.
    if !select.group_keys.is_empty()
        || !select.aggregates.is_empty()
        || !select.grouping_sets.is_empty()
    {
        return None;
    }
    let limit = select.limit?;
    // A set-returning projection expands rows above the window, so the LIMIT no longer bounds it.
    if select
        .projection
        .iter()
        .any(|p| matches!(p.expr.kind, TypedExprKind::SetReturning { .. }))
    {
        return None;
    }
    let (first, rest) = select.windows.split_first()?;
    // Every window: ranking function, no partition, default frame, and the SAME order as the first.
    let ranking_over_one_partition = |w: &crate::planner::WindowExpr| {
        matches!(w.func, W::RowNumber | W::Rank | W::DenseRank)
            && w.partition.is_empty()
            && w.frame.is_none()
    };
    if !ranking_over_one_partition(first) || first.order.is_empty() {
        return None;
    }
    if !rest
        .iter()
        .all(|w| ranking_over_one_partition(w) && w.order == first.order)
    {
        return None;
    }
    // The outer ORDER BY must equal the window order, so the first `m` rows by the window order are
    // exactly the first `m` the outer LIMIT keeps.
    if select.order_by != first.order {
        return None;
    }
    let m = select.offset.unwrap_or(0).saturating_add(limit);
    (m <= TOP_N_CAP_LIMIT).then_some(m)
}

/// Try to lower the base scan of a single (join-free) table to a [`PhysicalOperator::IndexScan`]
/// when `filter` constrains the leading column of a **single-column** index. Returns the
/// scan, or `None` to keep the `SeqScan`. The caller always re-applies the full `filter` above, so
/// the returned scan only has to be a *superset* of the qualifying rows — which it is, exactly, for
/// the value types handled here.
pub(super) fn try_index_scan(
    table: &TableSchema,
    indexes: &[IndexMeta],
    filter: &TypedExpr,
    stats: Option<&crate::executor::cost::ScanStats>,
) -> Option<PhysicalOperator> {
    // Flatten the top-level `AND` chain so each conjunct can be matched independently.
    let mut conjuncts = Vec::new();
    collect_and_conjuncts(filter, &mut conjuncts);

    for index in indexes {
        // v1: single-column indexes only. A multi-column index's key is the *concatenation* of all
        // its columns, so a leading-column-only bound would need prefix-range semantics (follow-up).
        let [col] = index.columns[..] else { continue };

        let mut eq: Option<&ast::Value> = None;
        let mut lo: Option<(&ast::Value, bool)> = None; // (value, inclusive)
        let mut hi: Option<(&ast::Value, bool)> = None;
        // Combined selectivity of the conjuncts that bound this index's column, for the cost-based
        // index-vs-seq decision. Each bounding conjunct multiplies in (they are AND-ed).
        let mut bound_selectivity = 1.0_f64;
        for conjunct in &conjuncts {
            // `col BETWEEN low AND high` is exactly `col >= low AND col <= high` — normalize it
            // into the same inclusive range bounds so the very common BETWEEN spelling (dates,
            // pagination) drives the index too (the BETWEEN form full-scanned
            // while the spelled-out form planned an IndexScan). `NOT BETWEEN` is not a contiguous
            // range and stays in the retained filter.
            if let TypedExprKind::Between {
                expr,
                low,
                high,
                negated: false,
            } = &conjunct.kind
                && let TypedExprKind::Column(ord) = expr.kind
                && ord == col
                && let (TypedExprKind::Literal(low_value), TypedExprKind::Literal(high_value)) =
                    (&low.kind, &high.kind)
                && is_index_safe_value(low_value)
                && is_index_safe_value(high_value)
            {
                if let Some(ctx) = stats {
                    bound_selectivity *= crate::executor::cost::selectivity(conjunct, ctx);
                }
                lo.get_or_insert((low_value, true));
                hi.get_or_insert((high_value, true));
                continue;
            }
            let Some((ord, op, value)) = col_op_literal(conjunct) else {
                continue;
            };
            if ord != col || !is_index_safe_value(value) {
                continue;
            }
            if let Some(ctx) = stats {
                bound_selectivity *= crate::executor::cost::selectivity(conjunct, ctx);
            }
            // Keep the first bound seen on each side; the retained `Filter` removes anything a
            // looser bound lets through, so this is correctness-safe (just less selective).
            match op {
                ast::BinaryOp::Eq => eq = Some(value),
                ast::BinaryOp::Gt => {
                    lo.get_or_insert((value, false));
                },
                ast::BinaryOp::GtEq => {
                    lo.get_or_insert((value, true));
                },
                ast::BinaryOp::Lt => {
                    hi.get_or_insert((value, false));
                },
                ast::BinaryOp::LtEq => {
                    hi.get_or_insert((value, true));
                },
                _ => {},
            }
        }

        // An equality bound on this (single-column, hence whole-key) index of a UNIQUE index
        // matches at most one row — the property the reactor-inline point-get gate requires.
        let unique_point = eq.is_some() && index.unique;
        // Equality is the tightest bound (`lo == hi`, both inclusive). Otherwise use whatever
        // open/closed range bounds the predicate supplied; skip the index if it gave neither.
        let (lo_bound, hi_bound) = if let Some(value) = eq {
            let key = vec![value.clone()];
            (Bound::Included(key.clone()), Bound::Included(key))
        } else if lo.is_some() || hi.is_some() {
            (range_bound(lo), range_bound(hi))
        } else {
            continue;
        };

        // Cost-based gate: when stats are available, only take the index if it is estimated
        // cheaper than scanning the whole table — a barely-selective bound is better served by a
        // sequential scan. Without stats the bound stays selective enough by assumption (heuristic).
        if let Some(ctx) = stats
            && !crate::executor::cost::prefers_index_scan(ctx.row_count(), bound_selectivity)
        {
            continue;
        }

        return Some(PhysicalOperator::IndexScan {
            table: table.clone(),
            index: index.name.clone(),
            lo: lo_bound,
            hi: hi_bound,
            unique_point,
        });
    }
    None
}

/// Flatten a predicate's top-level `AND` chain into its conjuncts (a non-`AND` node is one
/// conjunct). Only the top level is split — an `OR` or any other node stays whole.
fn collect_and_conjuncts<'a>(expr: &'a TypedExpr, out: &mut Vec<&'a TypedExpr>) {
    if let TypedExprKind::Binary {
        left,
        op: ast::BinaryOp::And,
        right,
    } = &expr.kind
    {
        collect_and_conjuncts(left, out);
        collect_and_conjuncts(right, out);
    } else {
        out.push(expr);
    }
}

/// Match a `column <cmp> literal` (or `literal <cmp> column`) comparison, returning the column
/// ordinal, the operator oriented as `column <op> value`, and the literal. `None` for any other
/// shape.
fn col_op_literal(expr: &TypedExpr) -> Option<(usize, ast::BinaryOp, &ast::Value)> {
    let TypedExprKind::Binary { left, op, right } = &expr.kind else {
        return None;
    };
    match (&left.kind, &right.kind) {
        (TypedExprKind::Column(ord), TypedExprKind::Literal(value)) => Some((*ord, *op, value)),
        // `5 = x` is `x = 5`; flipping the operator keeps the column on the left.
        (TypedExprKind::Literal(value), TypedExprKind::Column(ord)) => {
            Some((*ord, flip_comparison(*op)?, value))
        },
        _ => None,
    }
}

/// Mirror a comparison operator so `literal <op> column` becomes `column <flipped> literal`.
/// `None` for non-comparison operators (which cannot drive an index bound).
const fn flip_comparison(op: ast::BinaryOp) -> Option<ast::BinaryOp> {
    Some(match op {
        ast::BinaryOp::Eq => ast::BinaryOp::Eq,
        ast::BinaryOp::Lt => ast::BinaryOp::Gt,
        ast::BinaryOp::LtEq => ast::BinaryOp::GtEq,
        ast::BinaryOp::Gt => ast::BinaryOp::Lt,
        ast::BinaryOp::GtEq => ast::BinaryOp::LtEq,
        _ => return None,
    })
}

/// Build a key bound from an optional `(value, inclusive)`; `None` → `Unbounded`.
fn range_bound(bound: Option<(&ast::Value, bool)>) -> Bound<Vec<ast::Value>> {
    match bound {
        Some((value, true)) => Bound::Included(vec![value.clone()]),
        Some((value, false)) => Bound::Excluded(vec![value.clone()]),
        None => Bound::Unbounded,
    }
}

/// Whether `value`'s order-preserving index-key bytes compare *exactly* like the value itself, so an
/// index bound built from it neither misses nor mis-orders rows. This is the set whose
/// `encode_index_key` byte-equality matches value-equality and whose byte-order matches value-order:
/// `Float`/`Numeric` are excluded (e.g. `-0.0`/`+0.0` and unequal NUMERIC scales encode differently
/// yet compare equal — an equality index probe would miss rows), as are the composite/opaque types
/// the encoder rejects. Mirrors the executor's hash-safe set, minus `NULL` (never an equality match).
const fn is_index_safe_value(value: &ast::Value) -> bool {
    matches!(
        value,
        ast::Value::Bool(_)
            | ast::Value::Int(_)
            | ast::Value::Text(_)
            | ast::Value::Date(_)
            | ast::Value::Time(_)
            | ast::Value::Timestamp(_)
            | ast::Value::TimestampTz(_)
            | ast::Value::Uuid(_)
    )
}

/// Detect the `ORDER BY col <=> q LIMIT k` shape over a plain single-table scan that lowers to a
/// [`PhysicalOperator::VectorKnn`]. Returns the indexed column ordinal, the (constant)
/// query-vector expression, and `k`. Conservative: any clause that would change the row set or order
/// (joins, filter, grouping, aggregates, windows, distinct, offset, extra sort keys) disqualifies the
/// query, so the result is exactly the k nearest rows — the same as the Sort+Limit it replaces.
fn vector_knn_plan(
    select: &SelectPlan,
) -> Option<(TableSchema, usize, TypedExpr, u64, Option<TypedExpr>)> {
    let table = select.table.as_ref()?;
    // A `WHERE` filter is allowed — it is applied to the k-NN candidates. Everything else
    // that would change the row set or order disqualifies routing.
    if select.from_cte.is_some()
        || !select.joins.is_empty()
        || select.having.is_some()
        || !select.group_keys.is_empty()
        || !select.grouping_sets.is_empty()
        || !select.aggregates.is_empty()
        || !select.windows.is_empty()
        || select.distinct
        || !select.distinct_on.is_empty()
        || !select.recursive_ctes.is_empty()
    {
        return None;
    }
    let k = select.limit?;
    if select.offset.unwrap_or(0) != 0 {
        return None;
    }
    let [key] = select.order_by.as_slice() else {
        return None;
    };
    if !key.ascending {
        return None;
    }
    let TypedExprKind::Binary {
        left,
        op: ast::BinaryOp::VectorDistance,
        right,
    } = &key.expr.kind
    else {
        return None;
    };
    // One side is the indexed column, the other a constant query vector (cosine distance is symmetric).
    let (ordinal, query) = match (&left.kind, &right.kind) {
        (TypedExprKind::Column(o), _) if is_constant_expr(right) => (*o, (**right).clone()),
        (_, TypedExprKind::Column(o)) if is_constant_expr(left) => (*o, (**left).clone()),
        _ => return None,
    };
    if ordinal >= table.columns.len() {
        return None;
    }
    // A filter carrying a subquery may be correlated to this table's own rows; the exact Sort+Limit
    // pipeline resolves that per row (`eval_correlated`), the post-filter path does not. Leave such
    // queries on the exact path rather than routing them (it is correct, just not index-accelerated).
    if select
        .filter
        .as_ref()
        .is_some_and(crate::executor::ops::contains_subquery)
    {
        return None;
    }
    Some((table.clone(), ordinal, query, k, select.filter.clone()))
}

/// Whether `expr` is a constant — references no row column — so it can serve as a fixed query vector.
/// Conservative: only literal/cast/unary/binary nodes over constants qualify; anything else (a column,
/// outer column, function, subquery, …) is treated as non-constant, leaving the exact Sort path.
fn is_constant_expr(expr: &TypedExpr) -> bool {
    match &expr.kind {
        TypedExprKind::Literal(_) => true,
        TypedExprKind::Cast(inner, _) => is_constant_expr(inner),
        TypedExprKind::Unary { expr, .. } => is_constant_expr(expr),
        TypedExprKind::Binary { left, right, .. } => {
            is_constant_expr(left) && is_constant_expr(right)
        },
        _ => false,
    }
}
