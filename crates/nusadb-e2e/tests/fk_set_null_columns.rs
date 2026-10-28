//! `ON DELETE SET NULL (cols)` over the wire: the consumer's composite tenant key, persistence of
//! the column list across a restart, and a parent delete racing a child write at every isolation
//! level.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "integration test harness asserts via unwrap/panic/indexing over linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, TxnStatus, handle_client};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn next(conn: &mut Conn) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

/// Connect and consume the handshake up to the first `ReadyForQuery`.
async fn open(engine: &Arc<dyn StorageEngine>) -> Conn {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "nusadb-root".to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    while !matches!(next(&mut conn).await, BackendMessage::ReadyForQuery(_)) {}
    conn
}

/// Render one backend message: `C:<tag>`, `E:<sqlstate>:<message>`, `D:<fields>`, `Z:<I|T|E>`.
fn render(msg: &BackendMessage) -> String {
    match msg {
        BackendMessage::CommandComplete { tag } => format!("C:{tag}"),
        BackendMessage::Error { code, message } => format!("E:{code}:{message}"),
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
        BackendMessage::ReadyForQuery(status) => match status {
            TxnStatus::Idle => "Z:I".to_owned(),
            TxnStatus::InTransaction => "Z:T".to_owned(),
            TxnStatus::Failed => "Z:E".to_owned(),
        },
        other => format!("{other:?}"),
    }
}

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

/// The SQLSTATE of the first error in a response, if any.
fn error_code(out: &[String]) -> Option<String> {
    out.iter()
        .find_map(|m| m.strip_prefix("E:"))
        .map(|rest| rest.chars().take(5).collect())
}

/// Run `sql` and require it to succeed.
async fn ok(conn: &mut Conn, sql: &str) {
    let out = q(conn, sql).await;
    assert_eq!(error_code(&out), None, "{sql}: {out:?}");
}

/// Run `sql` and return its single value.
async fn scalar(conn: &mut Conn, sql: &str) -> String {
    let out = q(conn, sql).await;
    out.iter()
        .find_map(|m| m.strip_prefix("D:"))
        .unwrap_or_else(|| panic!("{sql}: no row in {out:?}"))
        .to_owned()
}

fn memory_engine() -> Arc<dyn StorageEngine> {
    Arc::new(BtreeEngine::new())
}

/// The consumer's schema: a tenant-paired composite key whose delete clears only `d_id`.
const D: &str = "CREATE TABLE d (org int NOT NULL, d_id int NOT NULL, name text NOT NULL, \
    CONSTRAINT d_pkey PRIMARY KEY (d_id), CONSTRAINT d_key UNIQUE (org, d_id, name))";
const V: &str = "CREATE TABLE v (v_id int PRIMARY KEY, org int NOT NULL, d_id int, name text, \
    CONSTRAINT v_d_fkey FOREIGN KEY (org, d_id, name) REFERENCES d (org, d_id, name) \
    ON DELETE SET NULL (d_id))";

/// Child rows whose complete key has no parent: always zero.
const ORPHANS: &str = "SELECT count(*) FROM v WHERE d_id IS NOT NULL AND NOT EXISTS \
    (SELECT 1 FROM d WHERE d.org = v.org AND d.d_id = v.d_id AND d.name = v.name)";

async fn rows(c: &mut Conn, sql: &str) -> Vec<String> {
    q(c, sql)
        .await
        .into_iter()
        .filter_map(|m| m.strip_prefix("D:").map(str::to_owned))
        .collect()
}

#[tokio::test]
async fn the_consumer_statement_keeps_the_tenant_column() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, D).await;
    ok(&mut c, V).await;
    ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a'), (7, 2, 'b')").await;
    ok(
        &mut c,
        "INSERT INTO v VALUES (10, 7, 1, 'a'), (11, 7, 2, 'b')",
    )
    .await;
    ok(&mut c, "DELETE FROM d WHERE d_id = 1").await;
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v ORDER BY v_id").await,
        ["10,7,NULL,a", "11,7,2,b"]
    );
    assert_eq!(
        scalar(
            &mut c,
            "SELECT delete_rule FROM information_schema.referential_constraints \
             WHERE constraint_name = 'v_d_fkey'"
        )
        .await,
        "SET NULL"
    );
    // The refusals carry their codes.
    for (sql, code) in [
        (
            "CREATE TABLE x (id int, org int, d_id int, name text, FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON UPDATE SET NULL (d_id))",
            "0A000",
        ),
        (
            "CREATE TABLE x (id int, org int, d_id int, name text, FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON DELETE SET DEFAULT (d_id))",
            "0A000",
        ),
        (
            "CREATE TABLE x (id int, org int, d_id int, name text, FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON DELETE SET NULL (id))",
            "42P10",
        ),
        (
            "CREATE TABLE x (id int, org int, d_id int, name text, FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON DELETE SET NULL (nope))",
            "42703",
        ),
        (
            "CREATE TABLE x (id int, org int, d_id int, name text, FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON DELETE SET NULL ())",
            "42601",
        ),
    ] {
        let out = q(&mut c, sql).await;
        assert_eq!(error_code(&out).as_deref(), Some(code), "{sql}: {out:?}");
    }
    // NOT NULL on a listed column: the delete is refused and changes nothing.
    ok(
        &mut c,
        "CREATE TABLE nn (id int PRIMARY KEY, org int NOT NULL, d_id int, name text, \
         FOREIGN KEY (org, d_id, name) REFERENCES d (org, d_id, name) ON DELETE SET NULL (org))",
    )
    .await;
    ok(&mut c, "INSERT INTO nn VALUES (1, 7, 2, 'b')").await;
    let out = q(&mut c, "DELETE FROM d WHERE d_id = 2").await;
    assert_eq!(error_code(&out).as_deref(), Some("23502"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM d").await, "1");
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v ORDER BY v_id").await,
        ["10,7,NULL,a", "11,7,2,b"]
    );
}

#[tokio::test]
async fn the_column_list_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        ok(&mut c, D).await;
        ok(&mut c, V).await;
        ok(
            &mut c,
            "CREATE TABLE r (id int PRIMARY KEY, org int NOT NULL, d_id int, name text)",
        )
        .await;
        ok(
            &mut c,
            "ALTER TABLE r ADD CONSTRAINT r_fkey FOREIGN KEY (org, d_id, name) \
             REFERENCES d (org, d_id, name) ON DELETE SET NULL (d_id, name)",
        )
        .await;
        ok(&mut c, "ALTER TABLE r RENAME COLUMN name TO label").await;
        ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a'), (7, 2, 'b')").await;
        ok(&mut c, "INSERT INTO v VALUES (10, 7, 1, 'a')").await;
        ok(&mut c, "INSERT INTO r VALUES (1, 7, 2, 'b')").await;
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    ok(&mut c, "DELETE FROM d").await;
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v").await,
        ["10,7,NULL,a"]
    );
    assert_eq!(
        rows(&mut c, "SELECT id, org, d_id, label FROM r").await,
        ["1,7,NULL,NULL"]
    );
}

/// A parent delete and a child write of the same key, each in an open transaction: whichever
/// locks the key second fails, so no child ever commits pointing at a deleted parent, at every
/// isolation level.
#[tokio::test]
async fn a_parent_delete_racing_a_child_write_never_orphans() {
    for level in ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        ok(&mut a, D).await;
        ok(&mut a, V).await;
        ok(
            &mut a,
            "INSERT INTO d VALUES (7, 1, 'a'), (7, 2, 'b'), (7, 3, 'c'), (7, 4, 'd')",
        )
        .await;
        ok(&mut a, "INSERT INTO v VALUES (10, 7, 1, 'a')").await;

        // The delete first: a child inserted against the deleted key is refused.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(scalar(&mut b, "SELECT count(*) FROM d").await, "4");
        ok(&mut a, "DELETE FROM d WHERE d_id = 1").await;
        let out = q(&mut b, "INSERT INTO v VALUES (20, 7, 1, 'a')").await;
        assert!(
            matches!(error_code(&out).as_deref(), Some("40001" | "23503")),
            "{level}: {out:?}"
        );
        ok(&mut a, "COMMIT").await;
        ok(&mut b, "ROLLBACK").await;
        assert_eq!(
            rows(&mut a, "SELECT v_id, org, d_id, name FROM v ORDER BY v_id").await,
            ["10,7,NULL,a"],
            "{level}"
        );

        // The child insert first: the delete is refused while it is open.
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, "INSERT INTO v VALUES (21, 7, 2, 'b')").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        let out = q(&mut a, "DELETE FROM d WHERE d_id = 2").await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("40001"),
            "{level}: {out:?}"
        );
        ok(&mut a, "ROLLBACK").await;
        ok(&mut b, "COMMIT").await;
        // Once the insert committed, the delete sets only its d_id.
        ok(&mut a, "DELETE FROM d WHERE d_id = 2").await;
        assert_eq!(
            rows(
                &mut a,
                "SELECT v_id, org, d_id, name FROM v WHERE v_id = 21"
            )
            .await,
            ["21,7,NULL,b"],
            "{level}"
        );

        // A child moved onto the key while the delete is open.
        ok(&mut a, "INSERT INTO v VALUES (22, 7, NULL, 'c')").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "DELETE FROM d WHERE d_id = 3").await;
        let out = q(&mut b, "UPDATE v SET d_id = 3 WHERE v_id = 22").await;
        assert!(error_code(&out).is_some(), "{level}: {out:?}");
        ok(&mut a, "COMMIT").await;
        ok(&mut b, "ROLLBACK").await;

        // The delete rewrites a child another transaction is updating: one of them fails.
        ok(&mut a, "INSERT INTO v VALUES (23, 7, 4, 'd')").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, "UPDATE v SET v_id = 24 WHERE v_id = 23").await;
        let out = q(&mut a, "DELETE FROM d WHERE d_id = 4").await;
        assert!(error_code(&out).is_some(), "{level}: {out:?}");
        ok(&mut a, "ROLLBACK").await;
        ok(&mut b, "COMMIT").await;
        ok(&mut a, "DELETE FROM d WHERE d_id = 4").await;
        assert_eq!(
            rows(
                &mut a,
                "SELECT v_id, org, d_id, name FROM v WHERE v_id = 24"
            )
            .await,
            ["24,7,NULL,d"],
            "{level}"
        );

        assert_eq!(scalar(&mut a, ORPHANS).await, "0", "{level}");
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM v WHERE org IS NULL").await,
            "0",
            "{level}"
        );
    }
}

/// The catalog holding the lists is reserved: a role that is not a superuser can neither read nor
/// write it by name.
#[tokio::test]
async fn the_catalog_is_reserved_for_superusers() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, D).await;
    ok(&mut c, V).await;
    ok(&mut c, "CREATE ROLE app LOGIN").await;
    ok(&mut c, "GRANT ALL ON v TO app").await;
    ok(&mut c, "GRANT ALL ON d TO app").await;
    ok(&mut c, "SET ROLE app").await;
    for sql in [
        "SELECT * FROM nusadb_fk_set_columns",
        "INSERT INTO nusadb_fk_set_columns VALUES ('v_d_fkey', '3:org')",
        "UPDATE nusadb_fk_set_columns SET columns = '3:org'",
        "DELETE FROM nusadb_fk_set_columns",
        "DROP TABLE nusadb_fk_set_columns",
    ] {
        let out = q(&mut c, sql).await;
        assert_eq!(error_code(&out).as_deref(), Some("42501"), "{sql}: {out:?}");
    }
    // The role's own deletes still use the list.
    ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a')").await;
    ok(&mut c, "INSERT INTO v VALUES (10, 7, 1, 'a')").await;
    ok(&mut c, "DELETE FROM d").await;
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v").await,
        ["10,7,NULL,a"]
    );
}

/// What a superuser writes into the catalog is not trusted: a list naming a column outside the
/// key, a column the table lacks, a column twice, or a value that does not decode refuses the
/// delete with an error naming the key, and never nulls a column outside the key.
#[tokio::test]
async fn a_tampered_list_refuses_the_delete() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, D).await;
    ok(&mut c, V).await;
    ok(
        &mut c,
        "CREATE TABLE c (id int PRIMARY KEY, p int REFERENCES d ON DELETE SET NULL, secret text)",
    )
    .await;
    ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a')").await;
    ok(&mut c, "INSERT INTO v VALUES (10, 7, 1, 'a')").await;
    ok(&mut c, "INSERT INTO c VALUES (1, 1, 's')").await;
    for (target, value) in [
        ("c_fkey1", "6:secret"),
        ("c_fkey1", "7:missing"),
        ("c_fkey1", "1:p1:p"),
        ("c_fkey1", "9:garbage"),
        ("c_fkey1", ""),
    ] {
        ok(
            &mut c,
            "DELETE FROM nusadb_fk_set_columns WHERE constraint_name = 'c_fkey1'",
        )
        .await;
        ok(
            &mut c,
            &format!("INSERT INTO nusadb_fk_set_columns VALUES ('{target}', '{value}')"),
        )
        .await;
        let out = q(&mut c, "DELETE FROM d").await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("55000"),
            "{value}: {out:?}"
        );
        assert!(
            out.iter().any(|m| m.contains("c_fkey1")),
            "{value}: {out:?}"
        );
        assert_eq!(
            rows(&mut c, "SELECT id, p, secret FROM c").await,
            ["1,1,s"],
            "{value}"
        );
        assert_eq!(
            scalar(&mut c, "SELECT count(*) FROM d").await,
            "1",
            "{value}"
        );
    }
    // A second row for a key that has a list is refused the same way.
    ok(
        &mut c,
        "DELETE FROM nusadb_fk_set_columns WHERE constraint_name = 'c_fkey1'",
    )
    .await;
    ok(
        &mut c,
        "INSERT INTO nusadb_fk_set_columns VALUES ('v_d_fkey', '3:org')",
    )
    .await;
    let out = q(&mut c, "DELETE FROM d").await;
    assert_eq!(error_code(&out).as_deref(), Some("55000"), "{out:?}");
    // Re-creating the key repairs it.
    ok(&mut c, "ALTER TABLE v DROP CONSTRAINT v_d_fkey").await;
    ok(
        &mut c,
        "ALTER TABLE v ADD CONSTRAINT v_d_fkey FOREIGN KEY (org, d_id, name) \
         REFERENCES d (org, d_id, name) ON DELETE SET NULL (d_id)",
    )
    .await;
    ok(&mut c, "DELETE FROM d").await;
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v").await,
        ["10,7,NULL,a"]
    );
    assert_eq!(
        rows(&mut c, "SELECT id, p, secret FROM c").await,
        ["1,NULL,s"]
    );
    assert_eq!(scalar(&mut c, ORPHANS).await, "0");
}

/// A table that already holds the catalog's name but not its shape is never written: a key with
/// a list is refused while it is there, and the table stays readable.
#[tokio::test]
async fn a_foreign_table_under_the_catalog_name_is_never_written() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(
        &mut c,
        "CREATE TABLE nusadb_fk_set_columns (id int, note int)",
    )
    .await;
    ok(&mut c, "INSERT INTO nusadb_fk_set_columns VALUES (1, 2)").await;
    ok(&mut c, D).await;
    let out = q(&mut c, V).await;
    assert_eq!(error_code(&out).as_deref(), Some("42P07"), "{out:?}");
    assert!(
        out.iter().any(|m| m.contains("nusadb_fk_set_columns")),
        "{out:?}"
    );
    // Keys without a list, and drops, leave it alone.
    ok(
        &mut c,
        "CREATE TABLE w (id int PRIMARY KEY, d_id int REFERENCES d ON DELETE SET NULL)",
    )
    .await;
    ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a')").await;
    ok(&mut c, "INSERT INTO w VALUES (1, 1)").await;
    ok(&mut c, "DELETE FROM d").await;
    ok(&mut c, "DROP TABLE w").await;
    assert_eq!(
        rows(&mut c, "SELECT id, note FROM nusadb_fk_set_columns").await,
        ["1,2"]
    );
    // A same-shaped impostor: every list is still checked against its key.
    ok(&mut c, "DROP TABLE nusadb_fk_set_columns").await;
    ok(
        &mut c,
        "CREATE TABLE nusadb_fk_set_columns (constraint_name text NOT NULL, columns text NOT NULL)",
    )
    .await;
    ok(&mut c, V).await;
    ok(&mut c, "INSERT INTO d VALUES (7, 1, 'a')").await;
    ok(&mut c, "INSERT INTO v VALUES (10, 7, 1, 'a')").await;
    ok(&mut c, "DELETE FROM d").await;
    assert_eq!(
        rows(&mut c, "SELECT v_id, org, d_id, name FROM v").await,
        ["10,7,NULL,a"]
    );
}
