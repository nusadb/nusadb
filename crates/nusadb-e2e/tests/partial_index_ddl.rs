//! Partial indexes across column DDL, checked against a copy of the table with no index.
//!
//! The planner scans a partial index only for a query whose `WHERE` implies its predicate, and
//! trusts the index to hold exactly the rows the predicate accepts. These tests interleave DDL that
//! changes what a predicate means or which rows it accepts (`ALTER COLUMN ... TYPE`, `RENAME
//! COLUMN`, `ADD` / `DROP COLUMN`, `REINDEX`) with writes and queries, and require every query to
//! return what the same query returns on an index-free copy (NULL rows included). They also cover a
//! session that planned a query before another connection's DDL and plans it again after it, a
//! snapshot taken before a type change or a `REINDEX`, `CREATE INDEX` racing a writer, and `COPY`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic"
)]

use nusadb_btree::BtreeEngine;
use nusadb_core::{StorageEngine, TableSchema, TxnId};
use nusadb_sql::ast::{Statement, Value};
use nusadb_sql::{Catalog, ExecutionResult, IndexInfo, Session, analyze, copy_from, parse, plan};

/// The analyzer's view of the engine, under the session's open transaction when there is one.
struct TxnCatalog<'a> {
    engine: &'a dyn StorageEngine,
    txn: Option<TxnId>,
}

impl TxnCatalog<'_> {
    fn with_txn<T>(
        &self,
        f: impl FnOnce(TxnId) -> Result<T, nusadb_sql::Error>,
    ) -> Result<T, nusadb_sql::Error> {
        if let Some(txn) = self.txn {
            return f(txn);
        }
        let txn = self.engine.begin(nusadb_core::IsolationLevel::default())?;
        let out = f(txn);
        let _ = self.engine.commit(txn);
        out
    }
}

impl Catalog for TxnCatalog<'_> {
    nusadb_test_utils::catalog_answers_as_superuser!(nusadb_sql);
    fn lookup_table(&self, name: &str) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.lookup_table_in(nusadb_core::PUBLIC_SCHEMA, name)
    }

    fn lookup_table_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Option<TableSchema>, nusadb_sql::Error> {
        self.with_txn(|txn| {
            self.engine
                .lookup_table_as_of_in(txn, schema, name)
                .map_err(Into::into)
        })
    }

    fn list_indexes(&self, name: &str) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes(self.engine, txn, name))
    }

    fn list_indexes_in(
        &self,
        schema: &str,
        name: &str,
    ) -> Result<Vec<IndexInfo>, nusadb_sql::Error> {
        self.with_txn(|txn| nusadb_sql::catalog_list_indexes_in(self.engine, txn, schema, name))
    }
}

/// One client connection.
struct Conn<'a> {
    engine: &'a BtreeEngine,
    session: Session<'a>,
}

impl<'a> Conn<'a> {
    fn new(engine: &'a BtreeEngine) -> Self {
        Self {
            engine,
            session: Session::new(engine),
        }
    }

    fn try_run(&mut self, sql: &str) -> Result<ExecutionResult, nusadb_sql::Error> {
        let catalog = TxnCatalog {
            engine: self.engine,
            txn: self.session.current_txn(),
        };
        let logical = analyze(parse(sql)?, &catalog)?;
        self.session.execute(plan(logical))
    }

    fn run(&mut self, sql: &str) -> ExecutionResult {
        self.try_run(sql)
            .unwrap_or_else(|e| panic!("`{sql}` failed: {e} ({})", e.sqlstate()))
    }

    /// The first column of every row `sql` returns, in the order returned.
    fn ids(&mut self, sql: &str) -> Vec<Value> {
        match self.run(sql) {
            ExecutionResult::Rows { rows, .. } => rows
                .into_iter()
                .map(|row| row.into_iter().next().unwrap_or(Value::Null))
                .collect(),
            other => panic!("`{sql}`: expected rows, got {other:?}"),
        }
    }

    /// [`Self::ids`] sorted, for a query with no `ORDER BY`.
    fn id_set(&mut self, sql: &str) -> Vec<i64> {
        let mut out: Vec<i64> = self
            .ids(sql)
            .into_iter()
            .map(|v| match v {
                Value::Int(n) => n,
                other => panic!("`{sql}`: expected an integer id, got {other:?}"),
            })
            .collect();
        out.sort_unstable();
        out
    }

    fn explain(&mut self, sql: &str) -> String {
        self.ids(&format!("EXPLAIN {sql}"))
            .into_iter()
            .map(|v| match v {
                Value::Text(line) => line,
                other => panic!("unexpected EXPLAIN row {other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A small deterministic generator, so a failing interleaving can be replayed from its seed.
struct Lcg(u64);

impl Lcg {
    const fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// The type `v` has at one point of [`ddl_and_dml_interleaved_match_an_index_free_copy`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VType {
    Text,
    Int,
    Numeric,
}

impl VType {
    const fn sql(self) -> &'static str {
        match self {
            Self::Text => "TEXT",
            Self::Int => "INT",
            Self::Numeric => "NUMERIC(8,2)",
        }
    }
}

/// Interleave column DDL with writes and queries on `t` (partial indexes) and `t_plain` (none):
/// every query must return the same rows on both. `v` moves between TEXT, INT and NUMERIC (each
/// change moves rows into and out of `v > '5'`: `'10' > '5'` is false as text and true as a
/// number, and `5.4` rounds to `5` as an integer), is renamed back and forth, a column `x` comes
/// and goes with a partial index of its own, and `REINDEX` runs now and then.
#[test]
fn ddl_and_dml_interleaved_match_an_index_free_copy() {
    for seed in [1_u64, 7, 42, 99, 2026, 4242, 31_337, 65_521] {
        run_interleaving(seed, 300);
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one interleaving, its operations side by side"
)]
fn run_interleaving(seed: u64, steps: usize) {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    let mut rng = Lcg(seed);
    for t in ["t", "t_plain"] {
        c.run(&format!(
            "CREATE TABLE {t} (id INT PRIMARY KEY, v NUMERIC(8,2), k INT NOT NULL, x INT)"
        ));
    }
    let mut vt = VType::Numeric;
    let mut vname = "v";
    let mut has_x = true;
    let mut next_id = 1_i64;
    let indexes = |vname: &str| {
        vec![
            format!("CREATE INDEX t_v ON t (k) WHERE {vname} > '5'"),
            format!("CREATE INDEX t_v_nn ON t (id) WHERE {vname} IS NOT NULL AND k > 0"),
            format!("CREATE INDEX t_k ON t (k) WHERE {vname} > '5' AND k > 3"),
        ]
    };
    for sql in indexes(vname) {
        c.run(&sql);
    }
    c.run("CREATE INDEX t_x ON t (k) WHERE x > 2");
    let value = |rng: &mut Lcg, vt: VType| -> String {
        match rng.below(8) {
            0 => "NULL".to_owned(),
            n => {
                let tenths = [54, 46, 100, 70, 30, 55, 500]
                    .get(usize::try_from(n - 1).unwrap())
                    .copied()
                    .unwrap();
                match vt {
                    // A whole number written as text, so a later change to a number succeeds.
                    VType::Text => format!("'{}'", tenths / 10),
                    VType::Int => format!("{}", tenths / 10),
                    VType::Numeric => format!("{}.{}", tenths / 10, tenths % 10),
                }
            },
        }
    };
    let key = |rng: &mut Lcg| -> String { rng.below(7).to_string() };
    let mut log: Vec<String> = Vec::new();
    for step in 0..steps {
        let both: Vec<String> = match rng.below(20) {
            0..=5 => {
                let id = next_id;
                next_id += 1;
                let (v, k) = (value(&mut rng, vt), key(&mut rng));
                let cols = if has_x { ", x" } else { "" };
                let x = if has_x {
                    format!(", {}", rng.below(5))
                } else {
                    String::new()
                };
                vec![format!(
                    "INSERT INTO @t (id, {vname}, k{cols}) VALUES ({id}, {v}, {k}{x})"
                )]
            },
            6..=8 => {
                let id = rng.below(u64::try_from(next_id).unwrap()) + 1;
                let v = value(&mut rng, vt);
                vec![format!("UPDATE @t SET {vname} = {v} WHERE id = {id}")]
            },
            9 => {
                let id = rng.below(u64::try_from(next_id).unwrap()) + 1;
                let k = key(&mut rng);
                vec![format!("UPDATE @t SET k = {k} WHERE id % 3 = {}", id % 3)]
            },
            10 => {
                let id = rng.below(u64::try_from(next_id).unwrap()) + 1;
                vec![format!("DELETE FROM @t WHERE id = {id}")]
            },
            11 | 12 => {
                // Text values are whole numbers (written so, or cast from INT), so every change
                // succeeds: NUMERIC goes to INT, INT to TEXT or NUMERIC, TEXT back to INT.
                let to = match (vt, rng.below(2)) {
                    (VType::Numeric | VType::Text, _) => VType::Int,
                    (VType::Int, 0) => VType::Text,
                    (VType::Int, _) => VType::Numeric,
                };
                let sql = format!("ALTER TABLE @t ALTER COLUMN {vname} TYPE {}", to.sql());
                vt = to;
                vec![sql]
            },
            13 => {
                let to = if vname == "v" { "w" } else { "v" };
                let sql = format!("ALTER TABLE @t RENAME COLUMN {vname} TO {to}");
                vname = to;
                vec![sql]
            },
            14 => {
                if has_x {
                    has_x = false;
                    vec!["ALTER TABLE @t DROP COLUMN x".to_owned()]
                } else {
                    has_x = true;
                    let mut out = vec!["ALTER TABLE @t ADD COLUMN x INT DEFAULT 3".to_owned()];
                    out.push("CREATE INDEX t_x ON t (k) WHERE x > 2".to_owned());
                    out
                }
            },
            15 => {
                // A column added and dropped beside `v` shifts nothing `v`'s indexes read, but
                // rewrites every row.
                vec![
                    "ALTER TABLE @t ADD COLUMN y INT DEFAULT 1".to_owned(),
                    "ALTER TABLE @t DROP COLUMN y".to_owned(),
                ]
            },
            16 => vec![match rng.below(3) {
                0 => "REINDEX TABLE t".to_owned(),
                1 => "REINDEX INDEX t_v".to_owned(),
                _ => "REINDEX DATABASE".to_owned(),
            }],
            _ => Vec::new(),
        };
        for sql in both {
            if sql.contains("@t") {
                for t in ["t", "t_plain"] {
                    let one = sql.replace("@t", t);
                    log.push(one.clone());
                    c.run(&one);
                }
            } else {
                // Index DDL and REINDEX concern `t` alone.
                log.push(sql.clone());
                c.run(&sql);
            }
        }
        // The queries every index answers, each against the copy.
        let bound = rng.below(7);
        let mut queries = vec![
            format!("SELECT id FROM @t WHERE {vname} > '5' AND k >= 0"),
            format!("SELECT id FROM @t WHERE {vname} > '5' AND k = {bound}"),
            format!("SELECT id FROM @t WHERE {vname} > '5' AND k > 3 AND k <= {bound}"),
            format!("SELECT id FROM @t WHERE {vname} IS NOT NULL AND k > 0 AND id > 0"),
        ];
        if has_x {
            queries.push("SELECT id FROM @t WHERE x > 2 AND k >= 0".to_owned());
        }
        for q in &queries {
            let (got, want) = (
                c.id_set(&q.replace("@t", "t")),
                c.id_set(&q.replace("@t", "t_plain")),
            );
            assert_eq!(
                got,
                want,
                "seed {seed}, step {step}: `{q}` differs from the index-free copy\nplan:\n{}\nafter:\n{}",
                c.explain(&q.replace("@t", "t")),
                log.join(";\n")
            );
        }
        // A capped ordered read through the index stops at the LIMIT, so an entry for a row the
        // predicate no longer accepts, or a missing one, changes the answer. `k` and `id` order
        // the rows fully.
        let ordered = format!("SELECT id FROM @t WHERE {vname} > '5' ORDER BY k, id LIMIT 2");
        let (got, want) = (
            c.ids(&ordered.replace("@t", "t")),
            c.ids(&ordered.replace("@t", "t_plain")),
        );
        assert_eq!(
            got,
            want,
            "seed {seed}, step {step}: `{ordered}`\nafter:\n{}",
            log.join(";\n")
        );
        let capped = format!("SELECT id FROM @t WHERE {vname} > '5' ORDER BY k LIMIT 1");
        let (got, want) = (
            c.ids(&capped.replace("@t", "t")),
            c.ids(&capped.replace("@t", "t_plain")),
        );
        // Ties in `k` may return either row: compare the `k` each one has.
        let k_of = |c: &mut Conn<'_>, t: &str, ids: &[Value]| -> Vec<Value> {
            ids.iter()
                .map(|id| {
                    let Value::Int(id) = id else {
                        panic!("expected an integer id, got {id:?}")
                    };
                    c.ids(&format!("SELECT k FROM {t} WHERE id = {id}"))
                        .into_iter()
                        .next()
                        .unwrap()
                })
                .collect()
        };
        assert_eq!(
            k_of(&mut c, "t", &got),
            k_of(&mut c, "t_plain", &want),
            "seed {seed}, step {step}: `{capped}`\nafter:\n{}",
            log.join(";\n")
        );
    }
    // The indexes were in use throughout, not bypassed.
    let plan = c.explain(&format!(
        "SELECT id FROM t WHERE {vname} > '5' ORDER BY k LIMIT 1"
    ));
    assert!(plan.contains("using t_v"), "{plan}");
}

/// A thread that planned a query before another connection's type change and rename plans it again
/// after them: its resolved predicates follow the table, so the query answers by the new types and
/// names.
#[test]
fn a_session_planning_before_and_after_ddl_elsewhere() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE s (id INT PRIMARY KEY, v TEXT, k INT NOT NULL)");
    c.run("CREATE INDEX s_v ON s (k) WHERE v > '5'");
    c.run("INSERT INTO s VALUES (1, '10', 1), (2, '7', 2), (3, NULL, 3), (4, '3', 4)");
    let q = "SELECT id FROM s WHERE v > '5' AND k >= 0";
    assert_eq!(c.id_set(q), vec![2]);
    assert!(c.explain(q).contains("using s_v"));
    // The type change runs on another thread, as another connection would.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                Conn::new(&engine).run("ALTER TABLE s ALTER COLUMN v TYPE INT");
            })
            .join()
            .unwrap();
    });
    assert_eq!(c.id_set(q), vec![1, 2]);
    assert!(c.explain(q).contains("using s_v"));
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let mut other = Conn::new(&engine);
                other.run("ALTER TABLE s RENAME COLUMN v TO w");
                other.run("ALTER TABLE s ADD COLUMN v INT DEFAULT 0");
            })
            .join()
            .unwrap();
    });
    // `v` is now a different column, which no index filters on.
    assert_eq!(c.id_set(q), Vec::<i64>::new());
    assert!(!c.explain(q).contains("using s_v"));
    let renamed = "SELECT id FROM s WHERE w > '5' AND k >= 0";
    assert_eq!(c.id_set(renamed), vec![1, 2]);
    assert!(c.explain(renamed).contains("using s_v"));
    assert_eq!(
        c.ids("SELECT id FROM s WHERE w > '5' ORDER BY k LIMIT 1"),
        vec![Value::Int(1)]
    );
}

/// A REPEATABLE READ transaction that read through a partial index before another connection's
/// `REINDEX` reads the same rows through it afterwards: a row the reindex retires keeps its entry
/// for snapshots older than it.
#[test]
fn a_snapshot_older_than_a_reindex_keeps_its_rows() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE r (id INT PRIMARY KEY, v INT, k INT NOT NULL)");
    c.run("CREATE INDEX r_v ON r (k) WHERE v > 5");
    c.run("INSERT INTO r VALUES (1, 10, 1), (2, 1, 2), (3, NULL, 3), (4, 6, 4)");
    let q = "SELECT id FROM r WHERE v > 5 AND k >= 0";
    let mut reader = Conn::new(&engine);
    reader.run("BEGIN ISOLATION LEVEL REPEATABLE READ");
    assert_eq!(reader.id_set(q), vec![1, 4]);
    c.run("UPDATE r SET v = 0 WHERE id = 4");
    c.run("REINDEX TABLE r");
    assert_eq!(c.id_set(q), vec![1]);
    assert_eq!(reader.id_set(q), vec![1, 4]);
    assert!(reader.explain(q).contains("using r_v"));
    reader.run("COMMIT");
    assert_eq!(reader.id_set(q), vec![1]);
}

/// `CREATE INDEX` waits for no writer: a writer's open transaction refuses it, and once the writer
/// commits the index holds the writer's rows.
#[test]
fn create_index_beside_an_open_writer() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    c.run("CREATE TABLE w (id INT PRIMARY KEY, v INT, k INT NOT NULL)");
    c.run("INSERT INTO w VALUES (1, 10, 1), (2, 1, 2)");
    let mut writer = Conn::new(&engine);
    writer.run("BEGIN");
    writer.run("INSERT INTO w VALUES (3, 7, 0)");
    let err = c
        .try_run("CREATE INDEX w_v ON w (k) WHERE v > 5")
        .expect_err("an open writer refuses the build");
    assert_eq!(err.sqlstate(), "40001", "{err}");
    writer.run("COMMIT");
    c.run("CREATE INDEX w_v ON w (k) WHERE v > 5");
    let q = "SELECT id FROM w WHERE v > 5 ORDER BY k LIMIT 1";
    assert!(c.explain(q).contains("using w_v"));
    assert_eq!(c.ids(q), vec![Value::Int(3)]);
    assert_eq!(
        c.id_set("SELECT id FROM w WHERE v > 5 AND k >= 0"),
        vec![1, 3]
    );
}

/// Rows loaded by `COPY` enter a partial index exactly when its predicate accepts them.
#[test]
fn copy_fills_partial_indexes_by_their_predicates() {
    let engine = BtreeEngine::new();
    let mut c = Conn::new(&engine);
    for t in ["cp", "cp_plain"] {
        c.run(&format!(
            "CREATE TABLE {t} (id INT PRIMARY KEY, v NUMERIC(6,2), k INT NOT NULL)"
        ));
    }
    c.run("CREATE INDEX cp_v ON cp (k) WHERE v > 5");
    let data = "1\t5.4\t1\n2\t4.9\t0\n3\t\\N\t2\n4\t6\t4\n5\t70\t3\n";
    for t in ["cp", "cp_plain"] {
        let Statement::Copy(copy) = parse(&format!("COPY {t} FROM STDIN")).unwrap() else {
            panic!("expected COPY");
        };
        assert_eq!(copy_from(&engine, &copy, data).unwrap(), 5);
    }
    for q in [
        "SELECT id FROM @t WHERE v > 5 AND k >= 0",
        "SELECT id FROM @t WHERE v > 5 AND k > 2",
    ] {
        assert_eq!(
            c.id_set(&q.replace("@t", "cp")),
            c.id_set(&q.replace("@t", "cp_plain")),
            "{q}"
        );
    }
    let q = "SELECT id FROM @t WHERE v > 5 ORDER BY k LIMIT 1";
    assert!(c.explain(&q.replace("@t", "cp")).contains("using cp_v"));
    assert_eq!(c.ids(&q.replace("@t", "cp")), vec![Value::Int(1)]);
    assert_eq!(c.ids(&q.replace("@t", "cp_plain")), vec![Value::Int(1)]);
}
