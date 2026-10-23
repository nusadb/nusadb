//! Unique indexes with `NULLS [NOT] DISTINCT` and a partial predicate over the wire: COPY, upsert,
//! restart persistence, isolation from uncommitted DDL, and two concurrent writers of one key at
//! every isolation level.

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

/// The consumer's shape: one active row per key, where the key may be NULL.
const P: &str = "CREATE TABLE p (a uuid NOT NULL, b text, s text NOT NULL, \
    CONSTRAINT p_pkey PRIMARY KEY (a))";
const P_INDEX: &str = "CREATE UNIQUE INDEX p_b_uidx ON p (b) NULLS NOT DISTINCT WHERE s = 'active'";

/// Run `COPY <table> FROM STDIN` with `data` and return the response.
async fn copy(c: &mut Conn, table: &str, data: &str) -> Vec<String> {
    send(
        c,
        FrontendMessage::Query {
            sql: format!("COPY {table} FROM STDIN"),
        },
    )
    .await;
    loop {
        let msg = next(c).await;
        if matches!(msg, BackendMessage::CopyInResponse { .. }) {
            break;
        }
        assert!(!matches!(msg, BackendMessage::Error { .. }), "{msg:?}");
    }
    send(
        c,
        FrontendMessage::CopyData {
            data: data.as_bytes().to_vec(),
        },
    )
    .await;
    send(c, FrontendMessage::CopyDone).await;
    until_ready(c).await
}

#[tokio::test]
async fn the_consumer_statement_is_accepted_and_enforced() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(&mut c, P).await;
    ok(&mut c, P_INDEX).await;
    ok(
        &mut c,
        "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'active'), \
         (gen_random_uuid(), NULL, 'done'), (gen_random_uuid(), NULL, 'done')",
    )
    .await;
    let out = q(
        &mut c,
        "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'active')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert!(
        out.iter().any(|m| m.contains("p_b_uidx")),
        "the error names the index: {out:?}"
    );
    // NULLS NOT DISTINCT on a non-unique index is refused clearly.
    let out = q(&mut c, "CREATE INDEX p_bad ON p (b) NULLS NOT DISTINCT").await;
    assert!(error_code(&out).is_some(), "{out:?}");
    assert!(
        out.iter()
            .any(|m| m.contains("only allowed on a UNIQUE index")),
        "{out:?}"
    );
}

#[tokio::test]
async fn copy_and_upsert_are_checked() {
    let engine = memory_engine();
    let mut c = open(&engine).await;
    ok(
        &mut c,
        "CREATE TABLE t (id int PRIMARY KEY, b text, s text)",
    )
    .await;
    ok(
        &mut c,
        "CREATE UNIQUE INDEX t_b ON t (b) NULLS NOT DISTINCT WHERE s = 'active'",
    )
    .await;
    // Two NULL keys in one COPY batch collide.
    let out = copy(&mut c, "t", "1\t\\N\tactive\n2\t\\N\tactive\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    let out = copy(&mut c, "t", "1\t\\N\tactive\n2\t\\N\tdone\n3\tk\tactive\n").await;
    assert_eq!(error_code(&out), None, "{out:?}");
    // A later COPY against a committed row.
    let out = copy(&mut c, "t", "4\t\\N\tactive\n").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(scalar(&mut c, "SELECT count(*) FROM t").await, "3");

    // Upsert inferring the partial index from its columns and predicate.
    ok(
        &mut c,
        "INSERT INTO t VALUES (5, NULL, 'active') ON CONFLICT (b) WHERE s = 'active' \
         DO UPDATE SET s = 'retired'",
    )
    .await;
    assert_eq!(
        scalar(&mut c, "SELECT s FROM t WHERE id = 1").await,
        "retired"
    );
    assert_eq!(
        scalar(&mut c, "SELECT count(*) FROM t WHERE id = 5").await,
        "0"
    );
    ok(
        &mut c,
        "INSERT INTO t VALUES (5, NULL, 'active') ON CONFLICT (b) WHERE s = 'active' DO NOTHING",
    )
    .await;
    ok(
        &mut c,
        "INSERT INTO t VALUES (6, NULL, 'active') ON CONFLICT (b) WHERE s = 'active' DO NOTHING",
    )
    .await;
    assert_eq!(
        scalar(
            &mut c,
            "SELECT string_agg(id::text, ',' ORDER BY id) FROM t WHERE s = 'active'"
        )
        .await,
        "3,5"
    );
    // Without the WHERE the partial index is not an arbiter.
    let out = q(
        &mut c,
        "INSERT INTO t VALUES (7, 'k', 'active') ON CONFLICT (b) DO NOTHING",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("42P10"), "{out:?}");
}

#[tokio::test]
async fn the_index_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("data");
    {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("open"));
        let mut c = open(&engine).await;
        ok(&mut c, P).await;
        ok(&mut c, P_INDEX).await;
        ok(&mut c, "CREATE TABLE d (b text)").await;
        ok(&mut c, "CREATE UNIQUE INDEX d_b ON d (b)").await;
        ok(
            &mut c,
            "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'active')",
        )
        .await;
        ok(&mut c, "INSERT INTO d VALUES (NULL), (NULL), ('x')").await;
        drop(c);
        while Arc::strong_count(&engine) > 1 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::open(&path).expect("reopen"));
    let mut c = open(&engine).await;
    let out = q(
        &mut c,
        "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'active')",
    )
    .await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    ok(
        &mut c,
        "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'done')",
    )
    .await;
    // The default index still treats NULLs as distinct after the restart.
    ok(&mut c, "INSERT INTO d VALUES (NULL)").await;
    let out = q(&mut c, "INSERT INTO d VALUES ('x')").await;
    assert_eq!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    assert_eq!(
        scalar(
            &mut c,
            "SELECT non_unique FROM information_schema.statistics WHERE index_name = 'p_b_uidx'"
        )
        .await,
        "0"
    );
    ok(&mut c, "DROP INDEX p_b_uidx").await;
    ok(
        &mut c,
        "INSERT INTO p VALUES (gen_random_uuid(), NULL, 'active')",
    )
    .await;
}

#[tokio::test]
async fn an_uncommitted_index_does_not_reject_other_sessions() {
    let engine = memory_engine();
    let mut a = open(&engine).await;
    let mut b = open(&engine).await;
    ok(&mut a, "CREATE TABLE u (b text)").await;
    ok(&mut a, "INSERT INTO u VALUES (NULL)").await;
    ok(&mut a, "BEGIN").await;
    ok(
        &mut a,
        "CREATE UNIQUE INDEX u_b ON u (b) NULLS NOT DISTINCT",
    )
    .await;
    // The other session never sees the uncommitted index: its write either waits out the DDL
    // (refused with the retryable 40001) or proceeds, but it is never a duplicate key.
    let out = q(&mut b, "INSERT INTO u VALUES (NULL)").await;
    assert_ne!(error_code(&out).as_deref(), Some("23505"), "{out:?}");
    let listed = scalar(
        &mut b,
        "SELECT count(*) FROM information_schema.statistics WHERE index_name = 'u_b'",
    )
    .await;
    assert_eq!(listed, "0");
    ok(&mut a, "ROLLBACK").await;
    ok(&mut b, "INSERT INTO u VALUES (NULL)").await;
}

/// Two connections write one NULL key of a partial NULLS NOT DISTINCT index inside open
/// transactions: never both commit, at every isolation level. Covered: two inserts, an insert
/// against a row moved into the predicate, and inserts against a row deleted or moved out of the
/// predicate by a transaction that then rolls back.
#[tokio::test]
async fn two_concurrent_writers_of_one_null_key_never_both_commit() {
    for level in ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        ok(
            &mut a,
            "CREATE TABLE c (id int PRIMARY KEY, b text, s text)",
        )
        .await;
        ok(
            &mut a,
            "CREATE UNIQUE INDEX c_b ON c (b) NULLS NOT DISTINCT WHERE s = 'active'",
        )
        .await;
        let active =
            async |c: &mut Conn| scalar(c, "SELECT count(*) FROM c WHERE s = 'active'").await;

        // Two inserts.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(active(&mut a).await, "0");
        assert_eq!(active(&mut b).await, "0");
        ok(&mut a, "INSERT INTO c VALUES (1, NULL, 'active')").await;
        let out_b = q(&mut b, "INSERT INTO c VALUES (2, NULL, 'active')").await;
        assert!(
            matches!(error_code(&out_b).as_deref(), Some("40001" | "23505")),
            "{level}: {out_b:?}"
        );
        let commit_a = q(&mut a, "COMMIT").await;
        assert_eq!(error_code(&commit_a), None, "{level}: {commit_a:?}");
        ok(&mut b, "ROLLBACK").await;
        assert_eq!(active(&mut a).await, "1", "{level}");

        // A snapshot taken before the other's commit still sees the committed row.
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        assert_eq!(active(&mut b).await, "1");
        ok(&mut a, "INSERT INTO c VALUES (3, 'k', 'active')").await;
        let out = q(&mut b, "INSERT INTO c VALUES (4, 'k', 'active')").await;
        assert_eq!(
            error_code(&out).as_deref(),
            Some("23505"),
            "{level}: {out:?}"
        );
        ok(&mut b, "ROLLBACK").await;

        // A row moved into the predicate against a concurrent insert.
        ok(&mut a, "INSERT INTO c VALUES (5, 'm', 'done')").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "UPDATE c SET s = 'active' WHERE id = 5").await;
        let out = q(&mut b, "INSERT INTO c VALUES (6, 'm', 'active')").await;
        assert!(error_code(&out).is_some(), "{level}: {out:?}");
        ok(&mut a, "COMMIT").await;
        ok(&mut b, "ROLLBACK").await;

        // The holder of the NULL key moves out of the predicate, the other inserts the key, the
        // first rolls back: the insert must not have committed beside the restored row.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "UPDATE c SET s = 'done' WHERE id = 1").await;
        let out = q(&mut b, "INSERT INTO c VALUES (7, NULL, 'active')").await;
        ok(&mut a, "ROLLBACK").await;
        assert!(
            error_code(&out).is_some(),
            "{level}: moved-out key taken: {out:?}"
        );

        // The same with a DELETE.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "DELETE FROM c WHERE id = 1").await;
        let out = q(&mut b, "INSERT INTO c VALUES (8, NULL, 'active')").await;
        ok(&mut a, "ROLLBACK").await;
        assert!(
            error_code(&out).is_some(),
            "{level}: deleted key taken: {out:?}"
        );

        // And with the key changed.
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "UPDATE c SET b = 'other' WHERE id = 1").await;
        let out = q(&mut b, "INSERT INTO c VALUES (9, NULL, 'active')").await;
        ok(&mut a, "ROLLBACK").await;
        assert!(
            error_code(&out).is_some(),
            "{level}: changed key taken: {out:?}"
        );

        assert_eq!(
            scalar(
                &mut a,
                "SELECT count(*) FROM c WHERE s = 'active' AND b IS NULL"
            )
            .await,
            "1",
            "{level}"
        );
        // Once the holder really leaves, the key is free.
        ok(&mut a, "UPDATE c SET s = 'done' WHERE id = 1").await;
        ok(&mut b, "INSERT INTO c VALUES (10, NULL, 'active')").await;
    }
}

/// Under the default NULLS DISTINCT, two concurrent writers of a NULL key both commit, while two
/// writers of one non-NULL key do not.
#[tokio::test]
async fn null_keys_of_a_default_index_never_collide_concurrently() {
    for level in ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"] {
        let engine = memory_engine();
        let mut a = open(&engine).await;
        let mut b = open(&engine).await;
        ok(&mut a, "CREATE TABLE n (id int PRIMARY KEY, b text)").await;
        ok(&mut a, "CREATE UNIQUE INDEX n_b ON n (b)").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "INSERT INTO n VALUES (1, NULL)").await;
        ok(&mut b, "INSERT INTO n VALUES (2, NULL)").await;
        ok(&mut a, "COMMIT").await;
        ok(&mut b, "COMMIT").await;
        ok(&mut a, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut b, &format!("BEGIN ISOLATION LEVEL {level}")).await;
        ok(&mut a, "INSERT INTO n VALUES (3, 'k')").await;
        let out = q(&mut b, "INSERT INTO n VALUES (4, 'k')").await;
        assert!(error_code(&out).is_some(), "{level}: {out:?}");
        ok(&mut a, "COMMIT").await;
        ok(&mut b, "ROLLBACK").await;
        assert_eq!(
            scalar(&mut a, "SELECT count(*) FROM n").await,
            "3",
            "{level}"
        );
    }
}
