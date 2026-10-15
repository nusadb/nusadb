//! Uncommitted DDL stays private to its transaction: a second connection resolves tables,
//! columns and indexes as they were last committed, through the simple and the extended protocol
//! and through its plan cache, and sees the change once it commits.

#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap/panic/indexing over linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, DescribeTarget, FrontendMessage, TxnStatus, handle_client,
};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn next(conn: &mut Conn) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Connect as `user` and consume the handshake up to the first `ReadyForQuery`.
async fn open(
    engine: &Arc<dyn StorageEngine>,
    user: &str,
) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: user.to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    while !matches!(next(&mut conn).await, BackendMessage::ReadyForQuery(_)) {}
    (conn, handle)
}

/// Render one backend message compactly: `C:<tag>`, `E:<sqlstate>`, `D:<fields>`, `T` for a row
/// description, `Z:<I|T|E>`, or the message's debug form.
fn render(msg: &BackendMessage) -> String {
    match msg {
        BackendMessage::CommandComplete { tag } => format!("C:{tag}"),
        BackendMessage::Error { code, .. } => format!("E:{code}"),
        BackendMessage::DataRow { values } => {
            let fields: Vec<String> = values
                .iter()
                .map(|v| {
                    v.as_ref().map_or_else(
                        || "NULL".to_owned(),
                        |b| String::from_utf8_lossy(b).into_owned(),
                    )
                })
                .collect();
            format!("D:{}", fields.join(","))
        },
        BackendMessage::RowDescription { .. } | BackendMessage::RowDescriptionTyped { .. } => {
            "T".to_owned()
        },
        BackendMessage::ReadyForQuery(status) => match status {
            TxnStatus::Idle => "Z:I".to_owned(),
            TxnStatus::InTransaction => "Z:T".to_owned(),
            TxnStatus::Failed => "Z:E".to_owned(),
        },
        other => format!("{other:?}"),
    }
}

/// Every message up to and including `ReadyForQuery`, rendered.
async fn until_ready(conn: &mut Conn) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(render(&msg));
        if done {
            return out;
        }
    }
}

/// Send a simple query and return its rendered response.
async fn q(conn: &mut Conn, sql: &str) -> Vec<String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    until_ready(conn).await
}

/// Run `sql` through Parse / Bind / Describe(portal) / Execute / Sync with text parameters.
async fn ext(conn: &mut Conn, sql: &str, params: &[&str]) -> Vec<String> {
    for msg in [
        FrontendMessage::Parse {
            name: String::new(),
            sql: sql.to_owned(),
            param_types: vec![],
        },
        FrontendMessage::Bind {
            portal: String::new(),
            statement: String::new(),
            params: params.iter().map(|p| Some(p.as_bytes().to_vec())).collect(),
            result_formats: vec![],
        },
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: String::new(),
        },
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 0,
        },
        FrontendMessage::Sync,
    ] {
        send(conn, msg).await;
    }
    until_ready(conn)
        .await
        .into_iter()
        .filter(|m| m != "ParseComplete" && m != "BindComplete" && m != "NoData")
        .collect()
}

/// A connection pair over one engine holding `t (id INT PRIMARY KEY, a INT)` with the row `(1, 10)`.
async fn pair() -> (Conn, Conn) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut a, _) = open(&engine, "nusadb-root").await;
    let (b, _) = open(&engine, "nusadb-root").await;
    q(&mut a, "CREATE TABLE t (id INT PRIMARY KEY, a INT)").await;
    q(&mut a, "INSERT INTO t VALUES (1, 10)").await;
    (a, b)
}

/// The data rows (and any error code) of a response, without the framing messages.
fn data(out: &[String]) -> Vec<&str> {
    out.iter()
        .filter(|m| m.starts_with("D:") || m.starts_with("E:"))
        .map(String::as_str)
        .collect()
}

#[tokio::test]
async fn an_added_column_is_invisible_until_it_commits() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE t ADD COLUMN c INT DEFAULT 5").await;
    assert_eq!(data(&q(&mut a, "SELECT * FROM t").await), ["D:1,10,5"]);

    // The other connection still sees two columns, over both protocols.
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10"]);
    assert_eq!(
        data(&ext(&mut b, "SELECT * FROM t WHERE id = $1", &["1"]).await),
        ["D:1,10"]
    );
    assert_eq!(
        data(
            &q(
                &mut b,
                "SELECT count(*) FROM information_schema.columns WHERE table_name = 't'"
            )
            .await
        ),
        ["D:2"]
    );
    // A write has to wait for the outcome: a conflict, not an arity error from a column that
    // does not exist for it.
    assert_eq!(
        data(&q(&mut b, "INSERT INTO t VALUES (2, 20)").await),
        ["E:40001"]
    );
    assert_eq!(
        data(&ext(&mut b, "INSERT INTO t VALUES ($1, $2)", &["2", "20"]).await),
        ["E:40001"]
    );
    let three = q(&mut b, "INSERT INTO t VALUES (2, 20, 2)").await;
    assert_eq!(data(&three), ["E:42601"], "{three:?}");

    q(&mut a, "COMMIT").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10,5"]);
    q(&mut b, "INSERT INTO t (id, a) VALUES (2, 20)").await;
    assert_eq!(
        data(&q(&mut b, "SELECT * FROM t ORDER BY id").await),
        ["D:1,10,5", "D:2,20,5"]
    );
}

#[tokio::test]
async fn dropped_and_renamed_columns_keep_their_committed_shape() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE t DROP COLUMN a").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10"]);
    assert_eq!(data(&q(&mut b, "SELECT a FROM t").await), ["D:10"]);
    q(&mut a, "ROLLBACK").await;

    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE t RENAME COLUMN a TO renamed").await;
    assert_eq!(data(&q(&mut b, "SELECT a FROM t").await), ["D:10"]);
    assert_eq!(data(&q(&mut b, "SELECT renamed FROM t").await), ["E:42703"]);
    q(&mut a, "COMMIT").await;
    assert_eq!(data(&q(&mut b, "SELECT renamed FROM t").await), ["D:10"]);
}

#[tokio::test]
async fn created_dropped_and_renamed_tables_resolve_as_committed() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "CREATE TABLE fresh (x INT)").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM fresh").await), ["E:42P01"]);
    // The name is taken only if the creator commits: retry, do not report it as existing.
    assert_eq!(
        data(&q(&mut b, "CREATE TABLE fresh (y INT)").await),
        ["E:40001"]
    );
    q(&mut a, "ROLLBACK").await;
    q(&mut b, "CREATE TABLE fresh (y INT)").await;
    assert_eq!(
        data(&q(&mut b, "SELECT count(*) FROM fresh").await),
        ["D:0"]
    );

    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE t RENAME TO moved").await;
    assert_eq!(data(&q(&mut b, "SELECT id FROM t").await), ["D:1"]);
    assert_eq!(data(&q(&mut b, "SELECT * FROM moved").await), ["E:42P01"]);
    q(&mut a, "ROLLBACK").await;

    q(&mut a, "BEGIN").await;
    q(&mut a, "DROP TABLE t").await;
    // The table still exists for the other connection, whose read waits for the outcome.
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["E:40001"]);
    q(&mut a, "ROLLBACK").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10"]);
}

#[tokio::test]
async fn an_uncommitted_index_is_not_planned_by_another_connection() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "CREATE INDEX t_a ON t (a)").await;
    let plan = q(&mut b, "EXPLAIN SELECT * FROM t WHERE a = 10")
        .await
        .join(" ");
    assert!(!plan.contains("IndexScan"), "{plan}");
    assert_eq!(
        data(&q(&mut b, "SELECT * FROM t WHERE a = 10").await),
        ["D:1,10"]
    );
    q(&mut a, "COMMIT").await;
    let plan = q(&mut b, "EXPLAIN SELECT * FROM t WHERE a = 10")
        .await
        .join(" ");
    assert!(plan.contains("IndexScan"), "{plan}");
    assert_eq!(
        data(&q(&mut b, "SELECT * FROM t WHERE a = 10").await),
        ["D:1,10"]
    );
}

#[tokio::test]
async fn the_plan_cache_never_serves_a_plan_from_another_schema() {
    let (mut a, mut b) = pair().await;
    // No index on it, so its plans are cacheable.
    q(&mut a, "CREATE TABLE p (id INT, a INT)").await;
    q(&mut a, "INSERT INTO p VALUES (1, 10)").await;
    // Warm the other connection's cache, then change the table under it.
    for _ in 0..2 {
        assert_eq!(data(&q(&mut b, "SELECT * FROM p").await), ["D:1,10"]);
    }
    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE p ADD COLUMN c INT DEFAULT 5").await;
    for _ in 0..2 {
        assert_eq!(data(&q(&mut b, "SELECT * FROM p").await), ["D:1,10"]);
    }
    q(&mut a, "COMMIT").await;
    for _ in 0..2 {
        assert_eq!(data(&q(&mut b, "SELECT * FROM p").await), ["D:1,10,5"]);
    }

    // A plan cached against this connection's own uncommitted column, then rolled back, is not
    // served once another connection commits a different column.
    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE p ADD COLUMN d INT DEFAULT 7").await;
    for _ in 0..2 {
        assert_eq!(data(&q(&mut a, "SELECT * FROM p").await), ["D:1,10,5,7"]);
    }
    q(&mut a, "ROLLBACK").await;
    q(&mut b, "ALTER TABLE p ADD COLUMN e TEXT DEFAULT 'x'").await;
    assert_eq!(data(&q(&mut a, "SELECT * FROM p").await), ["D:1,10,5,x"]);
}

/// Writes by another connection while one holds an uncommitted change to the table's columns or
/// constraints: each is refused with `40001` before it evaluates anything (no `42703` for a column
/// it cannot see, no `23514` for a check it cannot see), and goes through once the change ends.
#[tokio::test]
async fn writes_conflict_with_uncommitted_column_and_constraint_changes() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "ALTER TABLE t ADD COLUMN pad TEXT").await;
    q(&mut a, "INSERT INTO t VALUES (2, 20, 'x'), (3, 30, 'y')").await;
    let changes = [
        "ALTER TABLE t ADD COLUMN ci INT",
        "ALTER TABLE t ADD COLUMN cs SMALLINT DEFAULT 1",
        "ALTER TABLE t RENAME COLUMN a TO a2",
        "ALTER TABLE t ADD CONSTRAINT ck CHECK (a < 500)",
        "ALTER TABLE t ADD CONSTRAINT uq UNIQUE (id, pad)",
    ];
    let writes = [
        "UPDATE t SET pad = 'q' WHERE id = 1",
        "UPDATE t SET a = 900 WHERE id = 1",
        "INSERT INTO t (id, a) VALUES (9, 1)",
        "DELETE FROM t WHERE id = 99",
    ];
    for change in changes {
        for end in ["ROLLBACK", "COMMIT"] {
            q(&mut a, "BEGIN").await;
            assert_eq!(
                data(&q(&mut a, change).await),
                Vec::<&str>::new(),
                "{change}"
            );
            for write in writes {
                assert_eq!(
                    data(&q(&mut b, write).await),
                    ["E:40001"],
                    "{write} while {change}"
                );
            }
            assert_eq!(
                data(
                    &q(
                        &mut b,
                        "SELECT constraint_name FROM information_schema.table_constraints \
                         WHERE table_name = 't' ORDER BY 1"
                    )
                    .await
                ),
                ["D:t_pkey"],
                "an uncommitted constraint is not listed while {change}"
            );
            q(&mut a, end).await;
            // Settled: the write goes through (or fails on the committed constraint only).
            let after_out = q(&mut b, "UPDATE t SET pad = 'q' WHERE id = 1").await;
            let after = data(&after_out);
            assert!(after.is_empty(), "{change} {end}: {after:?}");
            if end == "COMMIT" {
                for undo in [
                    "ALTER TABLE t DROP COLUMN IF EXISTS ci",
                    "ALTER TABLE t DROP COLUMN IF EXISTS cs",
                    "ALTER TABLE t DROP CONSTRAINT IF EXISTS ck",
                    "ALTER TABLE t DROP CONSTRAINT IF EXISTS uq",
                ] {
                    q(&mut a, undo).await;
                }
                if change.contains("RENAME") {
                    q(&mut a, "ALTER TABLE t RENAME COLUMN a2 TO a").await;
                }
            }
        }
    }
}

/// A foreign key another connection is adding binds nobody else yet; dropping one takes the
/// referenced table too, so no write can break the key while the drop may still roll back.
#[tokio::test]
async fn uncommitted_foreign_keys_are_not_enforced_on_others() {
    let (mut a, mut b) = pair().await;
    q(&mut a, "CREATE TABLE ch (id INT PRIMARY KEY, pid INT)").await;
    for end in ["ROLLBACK", "COMMIT"] {
        q(&mut a, "BEGIN").await;
        q(
            &mut a,
            "ALTER TABLE ch ADD CONSTRAINT fk FOREIGN KEY (pid) REFERENCES t (id)",
        )
        .await;
        // The child is the key's own table, so writing it waits for the outcome.
        assert_eq!(
            data(&q(&mut b, "INSERT INTO ch VALUES (1, 77)").await),
            ["E:40001"]
        );
        // The parent was empty of references, so it was not taken; deleting from it is not
        // checked against a key that does not exist for this connection.
        q(&mut b, "INSERT INTO t VALUES (5, 50)").await;
        assert!(data(&q(&mut b, "DELETE FROM t WHERE id = 5").await).is_empty());
        q(&mut a, end).await;
    }
    // Committed: the key is enforced for everyone.
    assert_eq!(
        data(&q(&mut b, "INSERT INTO ch VALUES (1, 77)").await),
        ["E:23503"]
    );
    q(&mut b, "INSERT INTO ch VALUES (1, 1)").await;
    q(&mut a, "BEGIN").await;
    q(&mut a, "ALTER TABLE ch DROP CONSTRAINT fk").await;
    for write in [
        "DELETE FROM t WHERE id = 1",
        "INSERT INTO ch VALUES (2, 77)",
    ] {
        assert_eq!(data(&q(&mut b, write).await), ["E:40001"], "{write}");
    }
    q(&mut a, "ROLLBACK").await;
    assert_eq!(
        data(&q(&mut b, "DELETE FROM t WHERE id = 1").await),
        ["E:23503"],
        "the restored key still protects the referenced row"
    );
}

/// A snapshot older than a committed change that rewrote the rows or built an index gets
/// `40001` on its next read of the table, instead of rows of the wrong shape or an index result
/// that misses rows it should see. A rename rewrites nothing and is read normally.
#[tokio::test]
async fn an_older_snapshot_conflicts_with_a_committed_rewrite_or_index_build() {
    for iso in ["REPEATABLE READ", "SERIALIZABLE"] {
        for (change, conflicts) in [
            ("ALTER TABLE t ADD COLUMN c INT DEFAULT 5", true),
            ("ALTER TABLE t ALTER COLUMN a TYPE TEXT", true),
            ("ALTER TABLE t DROP COLUMN a", true),
            ("CREATE INDEX t_a ON t (a)", true),
            ("ALTER TABLE t RENAME COLUMN a TO aa", false),
        ] {
            let (mut a, mut b) = pair().await;
            q(
                &mut a,
                "INSERT INTO t SELECT g, g % 5 FROM generate_series(2, 50) g",
            )
            .await;
            q(&mut a, "CREATE TABLE other (x INT)").await;
            q(&mut b, &format!("BEGIN ISOLATION LEVEL {iso}")).await;
            q(&mut b, "SELECT count(*) FROM other").await;
            // Rows deleted before the index is built are still visible to the old snapshot.
            q(&mut a, "DELETE FROM t WHERE id > 40").await;
            assert!(data(&q(&mut a, change).await).is_empty(), "{change}");
            let read_out = q(&mut b, "SELECT count(*) FROM t WHERE id >= 1").await;
            let read = data(&read_out);
            if conflicts {
                assert_eq!(read, ["E:40001"], "{iso} after {change}");
            } else {
                assert_eq!(read, ["D:50"], "{iso} after {change}");
            }
            q(&mut b, "ROLLBACK").await;
            assert_eq!(
                data(&q(&mut b, "SELECT count(*) FROM t").await),
                ["D:40"],
                "a fresh snapshot reads the table"
            );
        }
    }
}

/// `READ COMMITTED` is not affected: an open transaction block reads the committed change in
/// its next statement.
#[tokio::test]
async fn read_committed_reads_a_committed_rewrite_in_the_next_statement() {
    let (mut a, mut b) = pair().await;
    q(&mut b, "BEGIN").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10"]);
    q(&mut a, "ALTER TABLE t ADD COLUMN c INT DEFAULT 5").await;
    q(&mut a, "CREATE INDEX t_a ON t (a)").await;
    assert_eq!(data(&q(&mut b, "SELECT * FROM t").await), ["D:1,10,5"]);
    assert_eq!(
        data(&q(&mut b, "SELECT id FROM t WHERE a = 10").await),
        ["D:1"]
    );
    q(&mut b, "COMMIT").await;
}

/// A session that read the table and went idle keeps an older snapshot alive; the schema changes
/// made meanwhile by others (several in a row, each one rewriting the table or building an index)
/// still go through, and the idle session itself sees the outcome its level promises.
#[tokio::test]
async fn an_idle_reader_does_not_make_later_ddl_fail() {
    for iso in ["REPEATABLE READ", "READ COMMITTED"] {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
        let (mut a, _) = open(&engine, "nusadb-root").await;
        let (mut idle, _) = open(&engine, "nusadb-root").await;
        let (mut other, _) = open(&engine, "nusadb-root").await;
        q(&mut a, "CREATE TABLE mz (id INT PRIMARY KEY, v INT)").await;
        q(&mut a, "INSERT INTO mz VALUES (1, 1)").await;
        q(&mut idle, &format!("BEGIN ISOLATION LEVEL {iso}")).await;
        assert_eq!(
            data(&q(&mut idle, "SELECT count(*) FROM mz").await),
            ["D:1"]
        );
        for n in 0..4 {
            for ddl in [
                format!("ALTER TABLE mz ADD COLUMN x{n} INT"),
                format!("CREATE INDEX mz_i{n} ON mz (v)"),
                format!("ALTER TABLE mz ADD COLUMN y{n} INT DEFAULT {n}"),
                format!("ALTER TABLE mz DROP COLUMN x{n}"),
            ] {
                let out = q(&mut a, &ddl).await;
                assert!(data(&out).is_empty(), "{iso}: {ddl}: {out:?}");
            }
        }
        // Writes and a block from a third session work too.
        for sql in [
            "UPDATE mz SET v = v + 1",
            "INSERT INTO mz (id, v) VALUES (2, 2)",
            "BEGIN",
            "SELECT count(*) FROM mz",
            "UPDATE mz SET v = v + 1",
            "COMMIT",
        ] {
            let out = q(&mut other, sql).await;
            assert!(
                !data(&out).iter().any(|m| m.starts_with("E:")),
                "{iso}: {sql}: {out:?}"
            );
        }
        let read = data(&q(&mut idle, "SELECT count(*) FROM mz").await)
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if iso == "READ COMMITTED" {
            assert_eq!(read, ["D:2"], "{iso}");
        } else {
            assert_eq!(
                read,
                ["E:40001"],
                "{iso}: its snapshot predates the rewrites"
            );
        }
        q(&mut idle, "ROLLBACK").await;
    }
}
