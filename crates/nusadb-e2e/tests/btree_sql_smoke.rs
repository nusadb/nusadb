//! The `--storage-engine btree` SQL bridge (design requirement, user order 2026-07-05): real SQL
//! strings through `parse → analyze → plan → execute` against the clustered B-link/B+tree
//! engine — the `DoD` that unblocks the QA SQL-verify.
//!
//! Covers exactly what QA reported broken: `CREATE TABLE` without a PK (used to fail with
//! `add_check_constraint is not implemented`), with a PK (`add_unique_constraint`), the
//! INSERT/SELECT/`WHERE pk = ?` smoke path, constraint *enforcement* (unique + check + FK,
//! which the SQL layer drives through `list_constraints`), and durability of it all across a
//! crash-reopen.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test harness asserts by panicking on failure"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema};
use nusadb_sql::ast::Value;
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, analyze, execute, parse, plan};

/// Adapts the engine's schema lookup to the analyzer's narrower `Catalog` port (the same shape
/// `end_to_end.rs` was written against — engine-agnostic by construction).
struct EngineCatalog<'a>(&'a dyn StorageEngine);

impl Catalog for EngineCatalog<'_> {
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table(name).map_err(Into::into)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.0.lookup_table_in(schema, name).map_err(Into::into)
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(Vec::new());
        };
        let backing: std::collections::HashSet<_> = self
            .0
            .list_constraints(schema.id)?
            .into_iter()
            .filter_map(|c| c.index)
            .collect();
        let mut out = Vec::new();
        for def in self.0.list_indexes(schema.id)? {
            if self
                .0
                .lookup_index(&def.name)?
                .is_some_and(|id| backing.contains(&id))
            {
                continue;
            }
            // A functional/expression key or partial predicate is unsafe as a scan candidate —
            // mirror the production `catalog_list_indexes` exclusion.
            if !def.key_exprs.is_empty() || def.predicate.is_some() {
                continue;
            }
            out.push(IndexInfo {
                name: def.name,
                columns: def.columns,
                unique: def.unique,
            });
        }
        Ok(out)
    }

    fn table_stats(
        &self,
        name: &str,
    ) -> Result<Option<nusadb_core::TableStats>, nusadb_sql::Error> {
        let Some(schema) = self.0.lookup_table(name)? else {
            return Ok(None);
        };
        self.0.table_stats(schema.id).map_err(Into::into)
    }
}

fn run(engine: &dyn StorageEngine, sql: &str) -> ExecutionResult {
    run_try(engine, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn run_try(engine: &dyn StorageEngine, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
    let stmt = parse(sql)?;
    let logical = analyze(stmt, &EngineCatalog(engine))?;
    execute(plan(logical), engine)
}

fn rows(result: ExecutionResult) -> Vec<Vec<Value>> {
    match result {
        ExecutionResult::Rows { rows, .. } => rows,
        other => panic!("expected SELECT rows, got {other:?}"),
    }
}

/// The QA `DoD`, statement for statement: CREATE TABLE without a PK, with a PK, INSERT,
/// SELECT, and the point lookup `WHERE pk = ?`.
#[test]
fn create_table_insert_select_point_lookup() {
    let engine = BtreeEngine::new();

    // The exact statement QA reported failing (`add_check_constraint not implemented`).
    run(&engine, "CREATE TABLE plain (id INT, v INT)");
    run(&engine, "INSERT INTO plain VALUES (1, 10), (2, 20)");
    assert_eq!(rows(run(&engine, "SELECT v FROM plain")).len(), 2);

    // And the PK form QA reported failing (`add_unique_constraint not implemented`).
    run(
        &engine,
        "CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL, qty INT)",
    );
    run(
        &engine,
        "INSERT INTO t VALUES (1, 'satu', 100), (2, 'dua', 200), (3, 'tiga', 300)",
    );
    let got = rows(run(&engine, "SELECT name, qty FROM t WHERE id = 2"));
    assert_eq!(
        got,
        vec![vec![Value::Text("dua".to_owned()), Value::Int(200),]]
    );
    // Range + order over the same table for good measure.
    let got = rows(run(
        &engine,
        "SELECT id FROM t WHERE qty >= 200 ORDER BY id DESC",
    ));
    assert_eq!(got, vec![vec![Value::Int(3)], vec![Value::Int(2)]]);
    // UPDATE + DELETE round out the smoke.
    run(&engine, "UPDATE t SET qty = 250 WHERE id = 2");
    run(&engine, "DELETE FROM t WHERE id = 1");
    let got = rows(run(&engine, "SELECT qty FROM t ORDER BY id"));
    assert_eq!(got, vec![vec![Value::Int(250)], vec![Value::Int(300)]]);
}

/// Constraint ENFORCEMENT through SQL (driven by the SQL layer over `list_constraints`):
/// duplicate PK rejected, CHECK rejected, UNIQUE rejected, FK parent-existence enforced.
#[test]
fn constraints_enforce_through_sql() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, age INT CHECK (age >= 0))",
    );
    run(&engine, "INSERT INTO users VALUES (1, 'a@x', 30)");

    let err = run_try(&engine, "INSERT INTO users VALUES (1, 'b@x', 40)")
        .expect_err("duplicate primary key must be rejected");
    assert!(
        err.to_string().to_lowercase().contains("duplicate key"),
        "{err}"
    );

    let err = run_try(&engine, "INSERT INTO users VALUES (2, 'a@x', 40)")
        .expect_err("duplicate unique email must be rejected");
    assert!(err.to_string().to_lowercase().contains("uniq"), "{err}");

    let err = run_try(&engine, "INSERT INTO users VALUES (3, 'c@x', -1)")
        .expect_err("check violation must be rejected");
    assert!(err.to_string().to_lowercase().contains("check"), "{err}");

    // FOREIGN KEY: child insert must reference an existing parent.
    run(
        &engine,
        "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users)",
    );
    run(&engine, "INSERT INTO orders VALUES (10, 1)");
    let err = run_try(&engine, "INSERT INTO orders VALUES (11, 999)")
        .expect_err("fk to a missing parent must be rejected");
    assert!(err.to_string().to_lowercase().contains("foreign"), "{err}");
}

/// The whole SQL-visible catalog — tables, rows, PK/UNIQUE/CHECK/FK constraints — survives a
/// crash-reopen of the durable engine, and enforcement still bites afterward.
#[test]
fn sql_catalog_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE users (id INT PRIMARY KEY, email TEXT UNIQUE, age INT CHECK (age >= 0))",
        );
        run(
            &engine,
            "CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users)",
        );
        // A column type whose catalog entry carries a payload byte (the range's element kind), so
        // recovery has to read the payload back rather than the bare type tag.
        run(
            &engine,
            "CREATE TABLE spans (id INT PRIMARY KEY, r INT4RANGE)",
        );
        run(&engine, "INSERT INTO users VALUES (1, 'a@x', 30)");
        run(&engine, "INSERT INTO orders VALUES (10, 1)");
        run(&engine, "INSERT INTO spans VALUES (1, '[1,10)')");
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    let got = rows(run(&engine, "SELECT email FROM users WHERE id = 1"));
    assert_eq!(got, vec![vec![Value::Text("a@x".to_owned())]]);
    assert_eq!(rows(run(&engine, "SELECT id FROM orders")).len(), 1);
    // The range column decodes to its value, not to text or a differently-sized element.
    assert_eq!(
        rows(run(&engine, "SELECT r::TEXT FROM spans WHERE id = 1")),
        vec![vec![Value::Text("[1,10)".to_owned())]]
    );
    run(&engine, "INSERT INTO spans VALUES (2, '[20,30)')");
    assert_eq!(rows(run(&engine, "SELECT id FROM spans")).len(), 2);

    // Every constraint kind still enforces after recovery.
    assert!(run_try(&engine, "INSERT INTO users VALUES (1, 'z@x', 5)").is_err());
    assert!(run_try(&engine, "INSERT INTO users VALUES (2, 'a@x', 5)").is_err());
    assert!(run_try(&engine, "INSERT INTO users VALUES (3, 'c@x', -9)").is_err());
    assert!(run_try(&engine, "INSERT INTO orders VALUES (11, 999)").is_err());
    run(&engine, "INSERT INTO users VALUES (4, 'd@x', 44)");
    assert_eq!(rows(run(&engine, "SELECT id FROM users")).len(), 2);
}

/// A `MACADDR8` column is DURABLE on disk: the 8-byte value is written to the WAL and recovery
/// decodes it back to the same value after a crash — not just held in memory. The EUI-48 literal
/// also proves the parse-time EUI-48→64 expansion is what gets persisted (not the raw 6 bytes).
#[test]
fn macaddr8_value_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE nics (id INT PRIMARY KEY, mac MACADDR8)",
        );
        run(
            &engine,
            "INSERT INTO nics VALUES (1, '08:00:2b:01:02:03:04:05')",
        );
        // A 6-byte (EUI-48) literal expands to EUI-64 on the write path; the expanded value persists.
        run(&engine, "INSERT INTO nics VALUES (2, '08:00:2b:01:02:03')");
    } // crash: no shutdown.

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The macaddr8 column decodes to its 8-byte value (canonical text), not text/wrong length.
        assert_eq!(
            rows(run(&engine, "SELECT mac::TEXT FROM nics WHERE id = 1")),
            vec![vec![Value::Text("08:00:2b:01:02:03:04:05".to_owned())]]
        );
        assert_eq!(
            rows(run(&engine, "SELECT mac::TEXT FROM nics WHERE id = 2")),
            vec![vec![Value::Text("08:00:2b:ff:fe:01:02:03".to_owned())]]
        );
        // Ordering by the recovered value works, and a committed insert after recovery also persists.
        run(
            &engine,
            "INSERT INTO nics VALUES (3, '00:00:00:00:00:00:00:01')",
        );
        assert_eq!(
            rows(run(&engine, "SELECT id FROM nics ORDER BY mac")),
            vec![
                vec![Value::Int(3)],
                vec![Value::Int(1)],
                vec![Value::Int(2)],
            ]
        );
    } // second crash: no shutdown.

    // A SECOND crash-reopen: every committed transaction is durable across repeated recovery — the
    // pre-crash rows AND the row committed after the first recovery all survive, byte-for-byte. This
    // is the production durability guarantee: once a transaction commits, its data is on disk.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT id, mac::TEXT FROM nics ORDER BY mac")),
        vec![
            vec![
                Value::Int(3),
                Value::Text("00:00:00:00:00:00:00:01".to_owned()),
            ],
            vec![
                Value::Int(1),
                Value::Text("08:00:2b:01:02:03:04:05".to_owned()),
            ],
            vec![
                Value::Int(2),
                Value::Text("08:00:2b:ff:fe:01:02:03".to_owned()),
            ],
        ]
    );
}

/// A `POINT` and a `BOX` column are DURABLE on disk: each value is written to the WAL as its
/// canonical text and recovery decodes it back to the same typed geometry after a crash — not just
/// held in memory. The box literals also prove the parse-time normalization (upper-right,
/// lower-left) is what gets persisted, not the raw input order.
#[test]
fn geometry_value_survives_reopen() {
    use nusadb_sql::geometry::GeomVal;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE shapes (id INT PRIMARY KEY, p POINT, b BOX)",
        );
        run(
            &engine,
            "INSERT INTO shapes VALUES (1, '(1.5,-2.5)', '(1,1),(3,3)')",
        );
        // The box corners are given lower-left first; normalization to upper-right-first persists.
        run(
            &engine,
            "INSERT INTO shapes VALUES (2, '3,4', '(1,3),(3,1)')",
        );
    } // crash: no shutdown.

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The point/box columns decode back to typed geometry values (byte-correct), not text.
        assert_eq!(
            rows(run(&engine, "SELECT p, b FROM shapes WHERE id = 1")),
            vec![vec![
                Value::Geometry(GeomVal::point(1.5, -2.5)),
                Value::Geometry(GeomVal::make_box(1.0, 1.0, 3.0, 3.0)),
            ]]
        );
        // The canonical text of the recovered values matches (box normalized regardless of input).
        assert_eq!(
            rows(run(
                &engine,
                "SELECT p::TEXT, b::TEXT FROM shapes WHERE id = 2"
            )),
            vec![vec![
                Value::Text("(3,4)".to_owned()),
                Value::Text("(3,3),(1,1)".to_owned()),
            ]]
        );
        // A committed insert after recovery also persists.
        run(
            &engine,
            "INSERT INTO shapes VALUES (3, '(0,0)', '(0,0),(2,2)')",
        );
    } // second crash: no shutdown.

    // A SECOND crash-reopen: every committed row — the pre-crash rows AND the row committed after
    // the first recovery — survives byte-for-byte.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(
            &engine,
            "SELECT id, p::TEXT, b::TEXT FROM shapes ORDER BY id"
        )),
        vec![
            vec![
                Value::Int(1),
                Value::Text("(1.5,-2.5)".to_owned()),
                Value::Text("(3,3),(1,1)".to_owned()),
            ],
            vec![
                Value::Int(2),
                Value::Text("(3,4)".to_owned()),
                Value::Text("(3,3),(1,1)".to_owned()),
            ],
            vec![
                Value::Int(3),
                Value::Text("(0,0)".to_owned()),
                Value::Text("(2,2),(0,0)".to_owned()),
            ],
        ]
    );
}

/// A `CIRCLE` column is DURABLE on disk: each value is written to the WAL as its canonical text
/// `<(cx,cy),r>` and recovery decodes it back to the same typed geometry after a crash — not just
/// held in memory. The non-canonical input forms (paren and bare) prove the parse-and-canonicalize
/// happens before persistence, and a second crash-reopen proves a value committed after recovery
/// also survives.
#[test]
fn circle_value_survives_reopen() {
    use nusadb_sql::geometry::GeomVal;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE circles (id INT PRIMARY KEY, c CIRCLE)",
        );
        run(
            &engine,
            "INSERT INTO circles VALUES (1, '<(1.5,2.5),3.25>')",
        );
        // Non-canonical input forms canonicalize before they persist.
        run(&engine, "INSERT INTO circles VALUES (2, '((1,2),3)')");
        run(&engine, "INSERT INTO circles VALUES (3, '4,5,6')");
    } // crash: no shutdown.

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The circle column decodes back to a typed geometry value (byte-correct), not text.
        assert_eq!(
            rows(run(&engine, "SELECT c FROM circles WHERE id = 1")),
            vec![vec![Value::Geometry(GeomVal::circle(1.5, 2.5, 3.25))]]
        );
        // The canonical text of the recovered non-canonical inputs matches.
        assert_eq!(
            rows(run(
                &engine,
                "SELECT c::TEXT FROM circles WHERE id = 2 OR id = 3 ORDER BY id"
            )),
            vec![
                vec![Value::Text("<(1,2),3>".to_owned())],
                vec![Value::Text("<(4,5),6>".to_owned())],
            ]
        );
        // A committed insert after recovery also persists.
        run(&engine, "INSERT INTO circles VALUES (4, '<(0,0),0>')");
    } // second crash: no shutdown.

    // A SECOND crash-reopen: every committed row — pre-crash and post-recovery — survives.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT id, c::TEXT FROM circles ORDER BY id")),
        vec![
            vec![Value::Int(1), Value::Text("<(1.5,2.5),3.25>".to_owned())],
            vec![Value::Int(2), Value::Text("<(1,2),3>".to_owned())],
            vec![Value::Int(3), Value::Text("<(4,5),6>".to_owned())],
            vec![Value::Int(4), Value::Text("<(0,0),0>".to_owned())],
        ]
    );
}

/// An `LSEG` column is DURABLE on disk: each value is written to the WAL as its canonical text
/// `[(x1,y1),(x2,y2)]` and recovery decodes it back to the same typed geometry after a crash — not
/// just held in memory. The non-canonical input forms (paren and bare) prove the
/// parse-and-canonicalize happens before persistence, the reversed-endpoint row proves the segment
/// order is preserved (no box-style normalization), and a second crash-reopen proves a value
/// committed after recovery also survives.
#[test]
fn lseg_value_survives_reopen() {
    use nusadb_sql::geometry::GeomVal;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE segs (id INT PRIMARY KEY, s LSEG)");
        run(
            &engine,
            "INSERT INTO segs VALUES (1, '[(1.5,2.5),(3.5,4.5)]')",
        );
        // Non-canonical input forms canonicalize before they persist.
        run(&engine, "INSERT INTO segs VALUES (2, '((1,2),(3,4))')");
        run(&engine, "INSERT INTO segs VALUES (3, '5,6,7,8')");
        // Reversed endpoints stay reversed — the order is preserved, unlike a box.
        run(&engine, "INSERT INTO segs VALUES (4, '[(3,4),(1,2)]')");
    } // crash: no shutdown.

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The lseg column decodes back to a typed geometry value (byte-correct), not text.
        assert_eq!(
            rows(run(&engine, "SELECT s FROM segs WHERE id = 1")),
            vec![vec![Value::Geometry(GeomVal::lseg(1.5, 2.5, 3.5, 4.5))]]
        );
        // The canonical text of the recovered non-canonical inputs matches.
        assert_eq!(
            rows(run(
                &engine,
                "SELECT s::TEXT FROM segs WHERE id = 2 OR id = 3 ORDER BY id"
            )),
            vec![
                vec![Value::Text("[(1,2),(3,4)]".to_owned())],
                vec![Value::Text("[(5,6),(7,8)]".to_owned())],
            ]
        );
        // A committed insert after recovery also persists.
        run(&engine, "INSERT INTO segs VALUES (5, '[(0,0),(0,0)]')");
    } // second crash: no shutdown.

    // A SECOND crash-reopen: every committed row — pre-crash and post-recovery — survives, with the
    // reversed-endpoint row (id 4) still un-normalized.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT id, s::TEXT FROM segs ORDER BY id")),
        vec![
            vec![
                Value::Int(1),
                Value::Text("[(1.5,2.5),(3.5,4.5)]".to_owned())
            ],
            vec![Value::Int(2), Value::Text("[(1,2),(3,4)]".to_owned())],
            vec![Value::Int(3), Value::Text("[(5,6),(7,8)]".to_owned())],
            vec![Value::Int(4), Value::Text("[(3,4),(1,2)]".to_owned())],
            vec![Value::Int(5), Value::Text("[(0,0),(0,0)]".to_owned())],
        ]
    );
}

/// The the design sequence `DoD` via real SQL: `SERIAL PRIMARY KEY`, `GENERATED ALWAYS AS IDENTITY`, and
/// bare `CREATE SEQUENCE` all work on the btree engine (each used to fail with
/// `create_sequence is not implemented`), auto-assigned ids are monotonic, and — the critical
/// property — a crash never repeats an id: post-recovery inserts continue past every id handed
/// out before the crash, committed or not.
#[test]
fn serial_identity_and_sequences_work_and_never_repeat_after_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        // The exact statements QA reported failing.
        run(&engine, "CREATE TABLE d (id SERIAL PRIMARY KEY, v TEXT)");
        run(
            &engine,
            "CREATE TABLE g (id INT GENERATED ALWAYS AS IDENTITY, v TEXT)",
        );
        run(&engine, "CREATE SEQUENCE counter");

        run(&engine, "INSERT INTO d (v) VALUES ('a'), ('b'), ('c')");
        let got = rows(run(&engine, "SELECT id, v FROM d ORDER BY id"));
        assert_eq!(
            got.iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
            "SERIAL ids are monotonic from 1"
        );
        run(&engine, "INSERT INTO g (v) VALUES ('x'), ('y')");
        assert_eq!(rows(run(&engine, "SELECT id FROM g")).len(), 2);
        // A duplicate CREATE SEQUENCE is rejected (the catalog is live)...
        assert!(run_try(&engine, "CREATE SEQUENCE counter").is_err());
        // ...and IF NOT EXISTS tolerates it.
        run(&engine, "CREATE SEQUENCE IF NOT EXISTS counter");
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    // New inserts continue past the pre-crash ids — never a duplicate PK.
    run(&engine, "INSERT INTO d (v) VALUES ('after')");
    let got = rows(run(&engine, "SELECT id FROM d ORDER BY id"));
    let ids: Vec<&Value> = got.iter().map(|r| &r[0]).collect();
    assert_eq!(ids.len(), 4);
    let mut sorted = ids.clone();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        4,
        "no duplicate SERIAL id after recovery: {ids:?}"
    );
    assert!(
        matches!(got[3][0], Value::Int(n) if n >= 4),
        "the post-crash id continues past the pre-crash counter"
    );
    // The bare sequence survived the crash: recreating it without IF NOT EXISTS still errors.
    assert!(run_try(&engine, "CREATE SEQUENCE counter").is_err());
    run(&engine, "DROP SEQUENCE counter");
    run(&engine, "CREATE SEQUENCE counter");
}

/// The the design sequence-value `DoD` via real SQL: `nextval`/`currval`/`setval` advance, read, and set a
/// user `CREATE SEQUENCE`; an advance is durable across a crash (a value never repeats); and the
/// side-effecting calls are loud-rejected in a per-row context rather than silently under-advancing.
#[test]
fn nextval_currval_setval_advance_read_and_survive_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE SEQUENCE s");

        // nextval advances from the start (1); currval follows without advancing.
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(1)
        );
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(2)
        );
        assert_eq!(
            rows(run(&engine, "SELECT currval('s')"))[0][0],
            Value::Int(2)
        );
        assert_eq!(
            rows(run(&engine, "SELECT currval('s')"))[0][0],
            Value::Int(2)
        );

        // Two nextvals in one row advance twice.
        let two = rows(run(&engine, "SELECT nextval('s'), nextval('s')"));
        assert_eq!(two[0], vec![Value::Int(3), Value::Int(4)]);

        // setval jumps; the next nextval returns value + increment; setval returns the set value.
        assert_eq!(
            rows(run(&engine, "SELECT setval('s', 100)"))[0][0],
            Value::Int(100)
        );
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(101)
        );

        // currval before any nextval, in a fresh sequence, is an error.
        run(&engine, "CREATE SEQUENCE fresh");
        assert!(run_try(&engine, "SELECT currval('fresh')").is_err());
        // nextval on a missing sequence is an error, not a silent NULL.
        assert!(run_try(&engine, "SELECT nextval('nope')").is_err());

        // An advancing call over a multi-row scan is rejected (never silently under-advanced).
        run(&engine, "CREATE TABLE t (x INT)");
        run(&engine, "INSERT INTO t VALUES (1), (2), (3)");
        assert!(run_try(&engine, "SELECT nextval('s') FROM t").is_err());
        // The rejected query did not advance the sequence.
        assert_eq!(
            rows(run(&engine, "SELECT nextval('s')"))[0][0],
            Value::Int(102)
        );

        // nextval inside INSERT ... VALUES advances once per tuple.
        run(&engine, "CREATE SEQUENCE oseq");
        run(&engine, "CREATE TABLE o (id INT, label TEXT)");
        run(
            &engine,
            "INSERT INTO o VALUES (nextval('oseq'), 'a'), (nextval('oseq'), 'b')",
        );
        let got = rows(run(&engine, "SELECT id, label FROM o ORDER BY id"));
        assert_eq!(got[0], vec![Value::Int(1), Value::Text("a".to_owned())]);
        assert_eq!(got[1], vec![Value::Int(2), Value::Text("b".to_owned())]);
    } // crash: no clean shutdown.

    // After recovery, the next value continues past every value handed out before the crash.
    let engine = BtreeEngine::open(&path).unwrap();
    let after = rows(run(&engine, "SELECT nextval('s')"))[0][0].clone();
    assert!(
        matches!(after, Value::Int(n) if n >= 103),
        "post-crash nextval continues past the pre-crash value: {after:?}"
    );
}

/// The the design+QA DDL-evolution `DoD` via real SQL: `ALTER TABLE` (ADD/DROP/RENAME COLUMN, RENAME
/// TABLE) and `CREATE SCHEMA` (both reported by QA as `XX000: ... not implemented by this
/// StorageEngine`) now work on the btree engine, with existing rows migrated correctly and the
/// whole DDL-evolved catalog surviving a crash-reopen.
#[test]
fn alter_table_and_create_schema_work_and_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE t (id INT, name TEXT)");
        run(&engine, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");

        // ADD COLUMN with existing rows: the SQL layer migrates them; the new column is NULL.
        run(&engine, "ALTER TABLE t ADD COLUMN qty INT");
        let got = rows(run(&engine, "SELECT id, name, qty FROM t ORDER BY id"));
        assert_eq!(
            got,
            vec![
                vec![Value::Int(1), Value::Text("a".to_owned()), Value::Null],
                vec![Value::Int(2), Value::Text("b".to_owned()), Value::Null],
            ]
        );
        run(&engine, "UPDATE t SET qty = id * 10");
        // RENAME COLUMN + a query against the new name.
        run(&engine, "ALTER TABLE t RENAME COLUMN qty TO amount");
        let got = rows(run(&engine, "SELECT amount FROM t ORDER BY id"));
        assert_eq!(got, vec![vec![Value::Int(10)], vec![Value::Int(20)]]);
        // ADD COLUMN with a DEFAULT backfills existing rows — the
        // fix lives in the SQL layer, so it holds on the btree engine too.
        run(&engine, "ALTER TABLE t ADD COLUMN tag INT DEFAULT 5");
        assert_eq!(
            rows(run(&engine, "SELECT tag FROM t ORDER BY id")),
            vec![vec![Value::Int(5)], vec![Value::Int(5)]],
        );
        run(&engine, "ALTER TABLE t DROP COLUMN tag");
        // DROP COLUMN.
        run(&engine, "ALTER TABLE t DROP COLUMN name");
        assert_eq!(
            rows(run(&engine, "SELECT * FROM t ORDER BY id"))[0].len(),
            2
        );
        // RENAME TABLE.
        run(&engine, "ALTER TABLE t RENAME TO items");
        assert_eq!(rows(run(&engine, "SELECT id FROM items")).len(), 2);

        // CREATE SCHEMA + a qualified table.
        run(&engine, "CREATE SCHEMA sales");
        run(&engine, "CREATE TABLE sales.orders (oid INT, total INT)");
        run(&engine, "INSERT INTO sales.orders VALUES (100, 500)");
        let got = rows(run(
            &engine,
            "SELECT total FROM sales.orders WHERE oid = 100",
        ));
        assert_eq!(got, vec![vec![Value::Int(500)]]);
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    // The DDL-evolved catalog survived: renamed table + column, dropped column, qualified table.
    let got = rows(run(&engine, "SELECT id, amount FROM items ORDER BY id"));
    assert_eq!(
        got,
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
        ]
    );
    assert!(
        run_try(&engine, "SELECT * FROM t").is_err(),
        "old table name is gone"
    );
    let got = rows(run(&engine, "SELECT total FROM sales.orders"));
    assert_eq!(got, vec![vec![Value::Int(500)]]);
    // Further evolution still works after recovery.
    run(&engine, "ALTER TABLE items ADD COLUMN note TEXT");
    run(&engine, "INSERT INTO items VALUES (3, 30, 'new')");
    assert_eq!(rows(run(&engine, "SELECT id FROM items")).len(), 3);
}

/// `TRUNCATE`'s constant-time path (a drop-and-recreate through rollback-aware DDL, all inside
/// one transaction) is durable: after a crash-reopen the table is still empty, every constraint
/// kind still enforces, the secondary index answers over post-truncate rows, and `RESTART
/// IDENTITY`'s sequence reset held.
#[test]
fn truncate_fast_path_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");

    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(
            &engine,
            "CREATE TABLE items (id SERIAL PRIMARY KEY, tag TEXT UNIQUE, score INT CHECK (score >= 0))",
        );
        run(&engine, "CREATE INDEX items_score ON items (score)");
        run(
            &engine,
            "INSERT INTO items (tag, score) VALUES ('a', 10), ('b', 20), ('c', 30)",
        );
        run(&engine, "TRUNCATE items RESTART IDENTITY");
        run(&engine, "INSERT INTO items (tag, score) VALUES ('z', 99)");
    } // crash: no shutdown.

    let engine = BtreeEngine::open(&path).unwrap();
    // Only the post-truncate row survived, with the restarted id.
    assert_eq!(
        rows(run(&engine, "SELECT id, tag FROM items")),
        vec![vec![Value::Int(1), Value::Text("z".to_owned())]]
    );
    // The secondary index answers over the rebuilt (post-truncate) table.
    assert_eq!(
        rows(run(&engine, "SELECT id FROM items WHERE score = 99")),
        vec![vec![Value::Int(1)]]
    );
    assert!(rows(run(&engine, "SELECT id FROM items WHERE score = 10")).is_empty());
    // Every constraint kind still enforces after recovery.
    assert!(
        run_try(
            &engine,
            "INSERT INTO items (id, tag, score) VALUES (1, 'q', 5)"
        )
        .is_err()
    );
    assert!(run_try(&engine, "INSERT INTO items (tag, score) VALUES ('z', 5)").is_err());
    assert!(run_try(&engine, "INSERT INTO items (tag, score) VALUES ('w', -1)").is_err());
    run(&engine, "INSERT INTO items (tag, score) VALUES ('y', 44)");
    assert_eq!(rows(run(&engine, "SELECT id FROM items")).len(), 2);
}

/// Two `TRUNCATE` edges that only bite on the rebuild path. A CHECK constraint shares the
/// table's constraint namespace but not the index namespace — a secondary index bearing a
/// CHECK's name must survive the rebuild as a real index (not be mistaken for that
/// constraint's backing index and silently dropped, name leaked). And the non-CASCADE refusal
/// must name the referencing table in its message, not a placeholder.
#[test]
fn truncate_keeps_index_named_like_check_and_names_refusing_child() {
    let engine = BtreeEngine::new();
    run(
        &engine,
        "CREATE TABLE t (a INT, b INT, CONSTRAINT dup CHECK (a > 0))",
    );
    run(&engine, "CREATE INDEX dup ON t (b)");
    run(&engine, "INSERT INTO t VALUES (1, 10), (2, 20)");
    run(&engine, "TRUNCATE t");
    // The index survived as a real index on the rebuilt table — asserted directly, because the
    // failure mode is silent (a leaked name also collides, and a seq scan also answers).
    let table = engine.lookup_table("t").unwrap().unwrap();
    let index_names: Vec<String> = engine
        .list_indexes(table.id)
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect();
    assert_eq!(index_names, vec!["dup".to_owned()]);
    // And its name is owned by that index: a re-create collides, and the CHECK enforces.
    assert!(run_try(&engine, "CREATE INDEX dup ON t (b)").is_err());
    assert!(run_try(&engine, "INSERT INTO t VALUES (-5, 1)").is_err());
    run(&engine, "INSERT INTO t VALUES (3, 30)");
    assert_eq!(
        rows(run(&engine, "SELECT a FROM t WHERE b = 30")),
        vec![vec![Value::Int(3)]]
    );

    run(&engine, "CREATE TABLE p (id INT PRIMARY KEY)");
    run(
        &engine,
        "CREATE TABLE kid (id INT PRIMARY KEY, pid INT REFERENCES p (id))",
    );
    run(&engine, "INSERT INTO p VALUES (1)");
    let err = run_try(&engine, "TRUNCATE p").unwrap_err().to_string();
    assert!(
        err.contains("\"kid\""),
        "refusal must name the child: {err}"
    );
}

// ---- Checkpoint durability --------------------------------------------------------------
// The checkpoint folds the whole committed state into an on-disk image and truncates the log.
// Its durability order — image fsynced, atomic rename, then log truncate — must survive a crash
// at every phase boundary. A real SIGKILL can't be issued in-process, so each crash phase is
// reproduced by reconstructing the exact on-disk state that crash would leave, then reopening.

/// A committed dataset with an index and a sequence, so a checkpoint image carries every record
/// family (table, rows, secondary index, PK/UNIQUE/CHECK, sequence).
fn seed_checkpoint_dataset(engine: &BtreeEngine) {
    run(
        engine,
        "CREATE TABLE t (id SERIAL PRIMARY KEY, tag TEXT UNIQUE, n INT CHECK (n >= 0))",
    );
    run(engine, "CREATE INDEX t_n ON t (n)");
    for i in 0..500 {
        run(
            engine,
            &format!("INSERT INTO t (tag, n) VALUES ('tag{i}', {i})"),
        );
    }
    // A settled delete must NOT appear in the image.
    run(engine, "DELETE FROM t WHERE n = 0");
}

/// Assert the recovered engine sees exactly the seeded dataset: 499 live rows, the deleted row
/// gone, the secondary index answering, and every constraint kind still enforced.
fn assert_checkpoint_dataset(engine: &BtreeEngine) {
    assert_eq!(
        rows(run(engine, "SELECT count(*) FROM t")),
        vec![vec![Value::Int(499)]]
    );
    assert!(
        rows(run(engine, "SELECT id FROM t WHERE n = 0")).is_empty(),
        "the settled delete stayed out of the image"
    );
    assert_eq!(
        rows(run(engine, "SELECT tag FROM t WHERE n = 250")),
        vec![vec![Value::Text("tag250".to_owned())]],
        "the secondary index answers over recovered rows"
    );
    assert!(
        run_try(engine, "INSERT INTO t (tag, n) VALUES ('tag1', 7)").is_err(),
        "UNIQUE still enforced"
    );
    assert!(
        run_try(engine, "INSERT INTO t (tag, n) VALUES ('fresh', -1)").is_err(),
        "CHECK still enforced"
    );
    // The SERIAL sequence continues past every id handed out before the checkpoint.
    run(engine, "INSERT INTO t (tag, n) VALUES ('after', 1)");
    let ids = rows(run(engine, "SELECT id FROM t WHERE tag = 'after'"));
    let after_id = ids.first().and_then(|r| r.first());
    assert!(
        matches!(after_id, Some(Value::Int(n)) if *n >= 500),
        "the SERIAL sequence continued past the checkpointed ids: {after_id:?}"
    );
}

/// The happy path: checkpoint, then reopen. The image replaces the log, the log actually
/// shrinks, and the recovered state is exact.
#[test]
fn checkpoint_shrinks_the_log_and_recovers_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let ckpt = dir.path().join("btree.wal.ckpt");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
        let log_before = std::fs::metadata(&path).unwrap().len();
        engine.checkpoint().unwrap();
        let log_after = std::fs::metadata(&path).unwrap().len();
        assert!(
            log_after < log_before,
            "log must shrink after checkpoint: {log_before} -> {log_after}"
        );
        assert!(ckpt.exists(), "the image exists after checkpoint");
        assert!(
            !dir.path().join("btree.wal.ckpt.tmp").exists(),
            "no scratch image left behind"
        );
    } // crash: no clean shutdown.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_checkpoint_dataset(&engine);
}

/// Crash while writing the image (phase 1): a `.ckpt.tmp` exists, no named `.ckpt`, the full log
/// is intact. Open must discard the scratch file and replay the whole log — data intact.
#[test]
fn checkpoint_crash_while_writing_image_falls_back_to_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
    }
    // A half-written scratch image, exactly what a phase-1 crash leaves.
    std::fs::write(
        dir.path().join("btree.wal.ckpt.tmp"),
        b"NCKP\x01\x00\x00\x00garbage",
    )
    .unwrap();
    let engine = BtreeEngine::open(&path).unwrap();
    assert!(
        !dir.path().join("btree.wal.ckpt.tmp").exists(),
        "open removed the orphaned scratch image"
    );
    assert!(!dir.path().join("btree.wal.ckpt").exists());
    assert_checkpoint_dataset(&engine);
}

/// Crash between the rename and the log truncation (phase 2/3 boundary): the named image AND the
/// full log both exist. Recovery must load the image and skip every log record at or before its
/// watermark — no double-apply (which would violate PK uniqueness and inflate the row count).
#[test]
fn checkpoint_crash_between_rename_and_truncate_does_not_double_apply() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let saved_log = dir.path().join("log_before.bin");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
        // Snapshot the full log, then checkpoint (which truncates it).
        std::fs::copy(&path, &saved_log).unwrap();
        engine.checkpoint().unwrap();
    }
    // Reconstruct the phase-2/3 crash: image present, full log restored (truncation never ran).
    std::fs::copy(&saved_log, &path).unwrap();
    assert!(dir.path().join("btree.wal.ckpt").exists());
    let engine = BtreeEngine::open(&path).unwrap();
    // Exactly 499 — a double-apply would show 998 or fail on the PK.
    assert_checkpoint_dataset(&engine);
}

/// Two checkpoints back to back, with writes in between, recover correctly — the second image
/// supersedes the first and the suffix after it replays once.
#[test]
fn second_checkpoint_supersedes_the_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
        engine.checkpoint().unwrap();
        run(&engine, "INSERT INTO t (tag, n) VALUES ('extra', 999)");
        engine.checkpoint().unwrap();
        run(&engine, "INSERT INTO t (tag, n) VALUES ('suffix', 1000)");
    }
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT count(*) FROM t")),
        vec![vec![Value::Int(501)]],
        "499 seeded + 'extra' + 'suffix'"
    );
    assert_eq!(
        rows(run(&engine, "SELECT tag FROM t WHERE n = 999")),
        vec![vec![Value::Text("extra".to_owned())]]
    );
    assert_eq!(
        rows(run(&engine, "SELECT tag FROM t WHERE n = 1000")),
        vec![vec![Value::Text("suffix".to_owned())]]
    );
}

/// Post-checkpoint recovery reads far less of the log than a full-history recovery would. The
/// proxy for "faster" that a unit test can assert deterministically is bytes-read: the log after
/// a checkpoint is a tiny fraction of the pre-checkpoint history.
#[test]
fn checkpoint_recovery_reads_far_less_than_full_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let engine = BtreeEngine::open(&path).unwrap();
    seed_checkpoint_dataset(&engine);
    for i in 500..2000 {
        run(
            &engine,
            &format!("INSERT INTO t (tag, n) VALUES ('tag{i}', {i})"),
        );
    }
    let history = std::fs::metadata(&path).unwrap().len();
    engine.checkpoint().unwrap();
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after * 10 < history,
        "post-checkpoint log ({after}) should be a small fraction of the history ({history})"
    );
}

/// A corrupt NAMED image is refused, not silently ignored: falling back to the truncated log
/// would lose every row the image holds. A flipped byte in the checksummed header (here, the
/// covered-LSN watermark) must be caught by the header CRC.
#[test]
fn checkpoint_corrupt_named_image_refuses_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let ckpt = dir.path().join("btree.wal.ckpt");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
        engine.checkpoint().unwrap();
    }
    // Flip a bit inside the covered-LSN field (byte 8) of the named image.
    let mut bytes = std::fs::read(&ckpt).unwrap();
    bytes[8] ^= 0x01;
    std::fs::write(&ckpt, &bytes).unwrap();
    let err = BtreeEngine::open(&path).unwrap_err().to_string();
    assert!(
        err.contains("checkpoint image") && err.contains("invalid"),
        "a corrupt named image must be refused, got: {err}"
    );
}

/// A truncated NAMED image (torn record body) is refused too — a named image was fsynced whole
/// before its rename, so trailing damage is corruption, not a crash artifact.
#[test]
fn checkpoint_truncated_named_image_refuses_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    let ckpt = dir.path().join("btree.wal.ckpt");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        seed_checkpoint_dataset(&engine);
        engine.checkpoint().unwrap();
    }
    // Chop the last 32 bytes: header intact (CRC still valid), record body torn.
    let bytes = std::fs::read(&ckpt).unwrap();
    std::fs::write(&ckpt, &bytes[..bytes.len() - 32]).unwrap();
    let err = BtreeEngine::open(&path).unwrap_err().to_string();
    assert!(
        err.contains("checkpoint image") && err.contains("invalid"),
        "a truncated named image must be refused, got: {err}"
    );
}

/// `checkpoint()` refuses while a transaction is still open — the quiesce is what lets it truncate
/// the log safely.
#[test]
fn checkpoint_refuses_with_an_active_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let engine = BtreeEngine::open(dir.path().join("btree.wal")).unwrap();
    let txn = engine
        .begin(nusadb_core::IsolationLevel::default())
        .unwrap();
    let err = engine.checkpoint().unwrap_err().to_string();
    assert!(
        err.contains("quiesced") && err.contains("active"),
        "checkpoint with an open transaction must be refused, got: {err}"
    );
    engine.rollback(txn).unwrap();
    // With the transaction gone it succeeds.
    engine.checkpoint().unwrap();
}

/// A `TRUNCATE` drops and recreates the table, and a dropped table's ANALYZE stats used to stay in
/// the catalog under its retired id — and, since checkpointing, get baked into every image. So
/// `ANALYZE`-then-`TRUNCATE` in a loop (the ETL staging pattern the fast TRUNCATE serves) grew the
/// checkpoint image without bound. The stats now go with the drop, and the image emitter only
/// writes stats for tables that still exist, so the image no longer grows with the truncate count —
/// while a live analyzed table keeps its stats across a checkpoint and reopen.
#[test]
fn truncate_of_analyzed_table_does_not_grow_the_image_and_keeps_live_stats() {
    fn image_bytes_after_cycles(cycles: usize) -> u64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("btree.wal");
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v INT)");
        for _ in 0..cycles {
            run(
                &engine,
                "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
            );
            run(&engine, "ANALYZE t");
            run(&engine, "TRUNCATE t");
        }
        engine.checkpoint().unwrap();
        std::fs::metadata(dir.path().join("btree.wal.ckpt"))
            .unwrap()
            .len()
    }

    // One cycle versus forty: with the leak each truncate stranded one stats entry that the image
    // re-emitted, so forty would dwarf one. With the fix the image is the same shape regardless.
    let one = image_bytes_after_cycles(1);
    let many = image_bytes_after_cycles(40);
    assert!(
        many <= one + 512,
        "checkpoint image grew with the truncate count: {one} bytes -> {many} bytes"
    );

    // A live analyzed table keeps its stats across a checkpoint and reopen (the planner must not
    // lose its histogram just because an unrelated table was truncated).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE live (id INT PRIMARY KEY, v INT)");
        run(&engine, "CREATE TABLE staging (id INT PRIMARY KEY, v INT)");
        run(&engine, "INSERT INTO live VALUES (1, 10), (2, 20), (3, 30)");
        run(&engine, "ANALYZE live");
        // Exercise the orphan path alongside: analyze then truncate the staging table.
        run(&engine, "INSERT INTO staging VALUES (1, 1), (2, 2)");
        run(&engine, "ANALYZE staging");
        run(&engine, "TRUNCATE staging");
        engine.checkpoint().unwrap();
    } // crash: no shutdown.
    let engine = BtreeEngine::open(&path).unwrap();
    let live = engine.lookup_table("live").unwrap().unwrap();
    assert!(
        engine.table_stats(live.id).unwrap().is_some(),
        "a live table's ANALYZE stats must survive the checkpoint"
    );
}

/// `CHECKPOINT` as a SQL statement folds the log into an image and truncates it: the WAL shrinks
/// measurably, the data survives a reopen, and the statement refuses (rather than silently doing
/// nothing) while a transaction is active.
#[test]
fn checkpoint_statement_shrinks_the_log_and_refuses_when_busy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("btree.wal");
    {
        let engine = BtreeEngine::open(&path).unwrap();
        run(&engine, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)");
        for i in 0..2000 {
            run(&engine, &format!("INSERT INTO t VALUES ({i}, 'row{i}')"));
        }
        let before = std::fs::metadata(&path).unwrap().len();
        // The statement runs through the same parse/analyze/plan/execute path a client uses.
        assert!(matches!(
            run(&engine, "CHECKPOINT"),
            ExecutionResult::CheckpointDone
        ));
        let after = std::fs::metadata(&path).unwrap().len();
        assert!(
            after < before,
            "CHECKPOINT must shrink the log: {before} bytes -> {after} bytes"
        );
        assert!(dir.path().join("btree.wal.ckpt").exists());
    } // crash: no shutdown.
    let engine = BtreeEngine::open(&path).unwrap();
    assert_eq!(
        rows(run(&engine, "SELECT count(*) FROM t")),
        vec![vec![Value::Int(2000)]]
    );

    // With a transaction open the engine is not quiesced, so CHECKPOINT is refused — and the
    // message names the number of active transactions rather than silently succeeding.
    let txn = engine
        .begin(nusadb_core::IsolationLevel::default())
        .unwrap();
    let err = run_try(&engine, "CHECKPOINT").unwrap_err().to_string();
    assert!(
        err.contains("quiesced") && err.contains("active"),
        "CHECKPOINT during a transaction must be refused with a quiescence error, got: {err}"
    );
    engine.rollback(txn).unwrap();
    // Once the transaction ends it succeeds again.
    assert!(matches!(
        run(&engine, "CHECKPOINT"),
        ExecutionResult::CheckpointDone
    ));
}
