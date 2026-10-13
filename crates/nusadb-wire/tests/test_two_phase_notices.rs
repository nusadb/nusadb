//! Protocol 1.3 notices and two-phase commit over the wire: a `NoticeResponse` reaches only a
//! connection that negotiated minor 3, ahead of its statement's completion on every path, while a
//! minor-2 connection keeps the exact byte stream it had; and a transaction prepared on one
//! connection is finished from another with every refusal carrying its SQLSTATE.

#![allow(
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::panic,
    reason = "integration test harness asserts via unwrap/panic; linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, Connection, DescribeTarget, Frame, FrontendMessage, TxnStatus, handle_client,
};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn next(conn: &mut Conn) -> BackendMessage {
    BackendMessage::decode(&next_frame(conn).await).unwrap()
}

async fn next_frame(conn: &mut Conn) -> Frame {
    conn.read_frame().await.unwrap().unwrap()
}

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

async fn query(conn: &mut Conn, sql: &str) {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
}

fn cc(tag: &str) -> BackendMessage {
    BackendMessage::CommandComplete {
        tag: tag.to_owned(),
    }
}

fn notice(severity: &str, code: &str, message: &str) -> BackendMessage {
    BackendMessage::NoticeResponse {
        severity: severity.to_owned(),
        code: code.to_owned(),
        message: message.to_owned(),
    }
}

/// Connect to `engine` as `user` at protocol `minor` and read through the startup handshake.
async fn connect(engine: &Arc<dyn StorageEngine>, user: &str, minor: u16) -> Conn {
    let (client, server) = tokio::io::duplex(256 * 1024);
    tokio::spawn(handle_client(server, Arc::clone(engine)));
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor,
            user: user.to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    loop {
        match next(&mut conn).await {
            BackendMessage::ReadyForQuery(TxnStatus::Idle) => return conn,
            BackendMessage::BackendKeyData { .. } | BackendMessage::ParameterStatus { .. } => {},
            other => panic!("unexpected handshake message: {other:?}"),
        }
    }
}

/// Every message of one simple query's response, up to and including its `ReadyForQuery`.
async fn run(conn: &mut Conn, sql: &str) -> Vec<BackendMessage> {
    query(conn, sql).await;
    let mut out = Vec::new();
    loop {
        let msg = next(conn).await;
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(msg);
        if done {
            return out;
        }
    }
}

/// The SQLSTATE of the error a simple query's response carries.
fn error_code(response: &[BackendMessage]) -> String {
    response
        .iter()
        .find_map(|m| match m {
            BackendMessage::Error { code, .. } => Some(code.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no error in {response:?}"))
}

const fn ready(status: TxnStatus) -> BackendMessage {
    BackendMessage::ReadyForQuery(status)
}

#[tokio::test]
async fn a_notice_reaches_only_a_minor_3_connection_ahead_of_its_completion() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut new = connect(&engine, "u", 3).await;
    let mut old = connect(&engine, "u", 2).await;
    assert_eq!(
        run(&mut new, "DROP TABLE IF EXISTS nope").await,
        vec![
            notice("NOTICE", "00000", "table \"nope\" does not exist, skipping"),
            cc("DROP TABLE"),
            ready(TxnStatus::Idle),
        ]
    );
    assert_eq!(
        run(&mut old, "DROP TABLE IF EXISTS nope").await,
        vec![cc("DROP TABLE"), ready(TxnStatus::Idle)]
    );
    // RAISE in a procedural block, including the notices sent before the error that ends it.
    assert_eq!(
        run(
            &mut new,
            "DO $$ BEGIN RAISE NOTICE 'one %', 1; RAISE WARNING 'two'; RAISE EXCEPTION 'three'; \
             END $$"
        )
        .await,
        vec![
            notice("NOTICE", "00000", "one 1"),
            notice("WARNING", "01000", "two"),
            BackendMessage::Error {
                code: "P0001".to_owned(),
                message: "raised exception: three".to_owned(),
            },
            ready(TxnStatus::Idle),
        ]
    );
    // A statement that raises nothing sends nothing extra, and nothing leaks into the next one.
    assert_eq!(
        run(&mut new, "CREATE TABLE t (n INT)").await,
        vec![cc("CREATE TABLE"), ready(TxnStatus::Idle)]
    );
}

/// The same script on two fresh servers, one connection at minor 2 and one at minor 3: the minor-3
/// stream with its `NoticeResponse` frames taken out is byte-for-byte the minor-2 stream, and the
/// minor-2 stream holds none.
#[tokio::test]
async fn a_minor_2_connection_sees_the_exact_old_byte_stream() {
    const SCRIPT: &[&str] = &[
        "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)",
        "CREATE TABLE IF NOT EXISTS t (id INT)",
        "INSERT INTO t VALUES (1, 'a')",
        "ALTER TABLE t DROP COLUMN IF EXISTS ghost",
        "SELECT id, v FROM t",
        "DO $$ BEGIN RAISE NOTICE 'n'; RAISE INFO 'i'; END $$",
        "DROP INDEX IF EXISTS nope",
        "BEGIN",
        "DROP SEQUENCE IF EXISTS s",
        "COMMIT",
    ];
    async fn frames(minor: u16) -> Vec<Frame> {
        let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
        let mut conn = connect(&engine, "u", minor).await;
        let mut out = Vec::new();
        for sql in SCRIPT {
            query(&mut conn, sql).await;
            loop {
                let frame = next_frame(&mut conn).await;
                let done = frame.message_type == b'Z';
                out.push(frame);
                if done {
                    break;
                }
            }
        }
        out
    }
    let old = frames(2).await;
    let new = frames(3).await;
    assert!(old.iter().all(|f| f.message_type != b'N'));
    let notices = new.iter().filter(|f| f.message_type == b'N').count();
    assert_eq!(notices, 6, "the script raises six notices");
    let without: Vec<&Frame> = new.iter().filter(|f| f.message_type != b'N').collect();
    assert_eq!(without.len(), old.len());
    // A frame is its type byte and payload; equal frames encode to equal bytes.
    for (a, b) in without.into_iter().zip(&old) {
        assert_eq!(a, b, "a minor-2 frame changed");
    }
}

/// Extended query: on the buffered path (a paginated `Execute`) and the streaming path (a
/// fetch-all text `Execute`) the notices come after `BindComplete` and before the statement's
/// completion, and an error mid-pipeline still skips to `Sync` as before.
#[tokio::test]
async fn notices_keep_their_place_in_the_extended_protocol() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut conn = connect(&engine, "u", 3).await;
    for max_rows in [5, 0] {
        send(
            &mut conn,
            FrontendMessage::Parse {
                name: String::new(),
                sql: "DROP VIEW IF EXISTS nv".to_owned(),
                param_types: vec![],
            },
        )
        .await;
        send(
            &mut conn,
            FrontendMessage::Bind {
                portal: String::new(),
                statement: String::new(),
                params: vec![],
                result_formats: vec![],
            },
        )
        .await;
        send(
            &mut conn,
            FrontendMessage::Describe {
                target: DescribeTarget::Portal,
                name: String::new(),
            },
        )
        .await;
        send(
            &mut conn,
            FrontendMessage::Execute {
                portal: String::new(),
                max_rows,
            },
        )
        .await;
        send(&mut conn, FrontendMessage::Sync).await;
        assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);
        assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);
        assert_eq!(next(&mut conn).await, BackendMessage::NoData);
        assert_eq!(
            next(&mut conn).await,
            notice("NOTICE", "00000", "view \"nv\" does not exist, skipping"),
            "max_rows {max_rows}"
        );
        assert_eq!(next(&mut conn).await, cc("DROP VIEW"));
        assert_eq!(next(&mut conn).await, ready(TxnStatus::Idle));
    }
    // A notice, then an error: the error still fails the pipeline until `Sync`.
    send(
        &mut conn,
        FrontendMessage::Parse {
            name: String::new(),
            sql: "DO $$ BEGIN RAISE NOTICE 'first'; RAISE EXCEPTION 'then'; END $$".to_owned(),
            param_types: vec![],
        },
    )
    .await;
    send(
        &mut conn,
        FrontendMessage::Bind {
            portal: String::new(),
            statement: String::new(),
            params: vec![],
            result_formats: vec![],
        },
    )
    .await;
    send(
        &mut conn,
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 1,
        },
    )
    .await;
    send(
        &mut conn,
        FrontendMessage::Execute {
            portal: String::new(),
            max_rows: 1,
        },
    )
    .await;
    send(&mut conn, FrontendMessage::Sync).await;
    assert_eq!(next(&mut conn).await, BackendMessage::ParseComplete);
    assert_eq!(next(&mut conn).await, BackendMessage::BindComplete);
    assert_eq!(next(&mut conn).await, notice("NOTICE", "00000", "first"));
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::Error { code, .. } if code == "P0001"
    ));
    // The second `Execute` was skipped; `Sync` ends the pipeline.
    assert_eq!(next(&mut conn).await, ready(TxnStatus::Idle));
}

/// `COPY` is untouched at minor 3: its sub-protocol frames arrive exactly as before, with no notice
/// among them.
#[tokio::test]
async fn copy_runs_unchanged_at_minor_3() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut conn = connect(&engine, "u", 3).await;
    run(&mut conn, "CREATE TABLE c (n INT)").await;
    query(&mut conn, "COPY c FROM STDIN").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CopyInResponse { .. }
    ));
    send(
        &mut conn,
        FrontendMessage::CopyData {
            data: b"1\n2\n".to_vec(),
        },
    )
    .await;
    send(&mut conn, FrontendMessage::CopyDone).await;
    assert_eq!(next(&mut conn).await, cc("COPY 2"));
    assert_eq!(next(&mut conn).await, ready(TxnStatus::Idle));
    query(&mut conn, "COPY c TO STDOUT").await;
    assert!(matches!(
        next(&mut conn).await,
        BackendMessage::CopyOutResponse { .. }
    ));
    let mut data = Vec::new();
    loop {
        match next(&mut conn).await {
            BackendMessage::CopyData { data: chunk } => data.extend(chunk),
            BackendMessage::CopyDone => break,
            other => panic!("unexpected message inside COPY: {other:?}"),
        }
    }
    assert_eq!(data, b"1\n2\n");
    assert_eq!(next(&mut conn).await, cc("COPY 2"));
    assert_eq!(next(&mut conn).await, ready(TxnStatus::Idle));
}

#[tokio::test]
async fn a_transaction_prepared_on_one_connection_is_finished_from_another() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut a = connect(&engine, "u", 3).await;
    let mut b = connect(&engine, "u", 3).await;
    run(&mut a, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)").await;
    run(&mut a, "INSERT INTO t VALUES (1, 'a')").await;
    assert_eq!(
        run(&mut a, "BEGIN").await,
        vec![cc("BEGIN"), ready(TxnStatus::InTransaction)]
    );
    run(&mut a, "UPDATE t SET v = 'A' WHERE id = 1").await;
    run(&mut a, "INSERT INTO t VALUES (2, 'b')").await;
    assert_eq!(
        run(&mut a, "PREPARE TRANSACTION 'w1'").await,
        vec![cc("PREPARE TRANSACTION"), ready(TxnStatus::Idle)]
    );
    // Invisible to the other connection, and its row is locked against it.
    let seen = run(&mut b, "SELECT v FROM t ORDER BY id").await;
    assert_eq!(
        seen.get(1),
        Some(&BackendMessage::DataRow {
            values: vec![Some(b"a".to_vec())]
        })
    );
    assert_eq!(seen.get(2), Some(&cc("SELECT 1")));
    assert_eq!(
        error_code(&run(&mut b, "UPDATE t SET v = 'x' WHERE id = 1").await),
        "40001"
    );
    // The connection that prepared it can carry on with other work.
    assert_eq!(
        run(&mut a, "SELECT gid FROM nusadb_prepared_xacts").await[1],
        BackendMessage::DataRow {
            values: vec![Some(b"w1".to_vec())]
        }
    );
    assert_eq!(
        run(&mut b, "COMMIT PREPARED 'w1'").await,
        vec![cc("COMMIT PREPARED"), ready(TxnStatus::Idle)]
    );
    let seen = run(&mut b, "SELECT v FROM t ORDER BY id").await;
    assert_eq!(seen.get(3), Some(&cc("SELECT 2")));
    assert_eq!(
        run(&mut b, "UPDATE t SET v = 'x' WHERE id = 1").await,
        vec![cc("UPDATE 1"), ready(TxnStatus::Idle)]
    );

    // ROLLBACK PREPARED discards.
    run(&mut a, "BEGIN").await;
    run(&mut a, "DELETE FROM t").await;
    run(&mut a, "PREPARE TRANSACTION 'w2'").await;
    assert_eq!(
        run(&mut b, "ROLLBACK PREPARED 'w2'").await,
        vec![cc("ROLLBACK PREPARED"), ready(TxnStatus::Idle)]
    );
    assert_eq!(run(&mut b, "SELECT v FROM t").await.len(), 5);
}

#[tokio::test]
async fn two_phase_refusals_over_the_wire() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = connect(&engine, "u", 3).await;
    run(&mut c, "CREATE TABLE t (n INT)").await;
    // Outside a transaction block.
    assert_eq!(
        error_code(&run(&mut c, "PREPARE TRANSACTION 'x'").await),
        "25P01"
    );
    // An unknown identifier.
    assert_eq!(
        error_code(&run(&mut c, "COMMIT PREPARED 'nope'").await),
        "42704"
    );
    // A duplicate, and an identifier too long: each rolls its transaction back.
    run(&mut c, "BEGIN").await;
    run(&mut c, "INSERT INTO t VALUES (1)").await;
    run(&mut c, "PREPARE TRANSACTION 'd'").await;
    run(&mut c, "BEGIN").await;
    let refused = run(&mut c, "PREPARE TRANSACTION 'd'").await;
    assert_eq!(error_code(&refused), "42710");
    assert_eq!(refused.last(), Some(&ready(TxnStatus::Idle)));
    run(&mut c, "BEGIN").await;
    let refused = run(
        &mut c,
        &format!("PREPARE TRANSACTION '{}'", "x".repeat(201)),
    )
    .await;
    assert_eq!(error_code(&refused), "22023");
    assert_eq!(refused.last(), Some(&ready(TxnStatus::Idle)));
    // Inside a transaction block COMMIT PREPARED is refused and fails the block.
    run(&mut c, "BEGIN").await;
    let refused = run(&mut c, "COMMIT PREPARED 'd'").await;
    assert_eq!(error_code(&refused), "25001");
    assert_eq!(refused.last(), Some(&ready(TxnStatus::Failed)));
    // PREPARE of a failed transaction rolls it back and says so.
    assert_eq!(
        run(&mut c, "PREPARE TRANSACTION 'f'").await,
        vec![cc("ROLLBACK"), ready(TxnStatus::Idle)]
    );
    // A transaction that ran LISTEN cannot be prepared.
    run(&mut c, "BEGIN").await;
    run(&mut c, "LISTEN ch").await;
    assert_eq!(
        error_code(&run(&mut c, "PREPARE TRANSACTION 'l'").await),
        "0A000"
    );
    // ... nor one that declared a cursor WITH HOLD.
    run(&mut c, "BEGIN").await;
    run(&mut c, "DECLARE k CURSOR WITH HOLD FOR SELECT n FROM t").await;
    assert_eq!(
        error_code(&run(&mut c, "PREPARE TRANSACTION 'k'").await),
        "0A000"
    );
    // A later transaction on the same connection prepares normally again.
    run(&mut c, "BEGIN").await;
    assert_eq!(
        run(&mut c, "PREPARE TRANSACTION 'ok'").await,
        vec![cc("PREPARE TRANSACTION"), ready(TxnStatus::Idle)]
    );
    // Another role may neither finish it nor see it; a superuser may do both.
    let mut other = connect(&engine, "v", 3).await;
    assert_eq!(
        error_code(&run(&mut other, "ROLLBACK PREPARED 'ok'").await),
        "42501"
    );
    assert_eq!(
        run(&mut other, "SELECT gid FROM nusadb_prepared_xacts").await,
        vec![
            BackendMessage::RowDescriptionTyped {
                columns: vec![(
                    "gid".to_owned(),
                    nusadb_wire::column_type_tag(nusadb_core::ColumnType::Text)
                )]
            },
            cc("SELECT 0"),
            ready(TxnStatus::Idle),
        ]
    );
    let mut root = connect(&engine, nusadb_sql::BOOTSTRAP_SUPERUSER, 3).await;
    assert_eq!(
        run(
            &mut root,
            "SELECT gid FROM nusadb_prepared_xacts ORDER BY gid"
        )
        .await
        .len(),
        5
    );
    run(&mut root, "ROLLBACK PREPARED 'ok'").await;
    run(&mut c, "COMMIT PREPARED 'd'").await;
    let seen = run(&mut c, "SELECT n FROM t").await;
    assert_eq!(seen.get(2), Some(&cc("SELECT 1")));
}

/// In a query of several statements each statement's notices come right before its own
/// completion, and a minor-2 connection gets the same batch without them.
#[tokio::test]
async fn notices_in_a_multi_statement_query_precede_each_completion() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut new = connect(&engine, "u", 3).await;
    let batch = "DROP TABLE IF EXISTS a; CREATE TABLE b (n INT); \
                 DO $$ BEGIN RAISE NOTICE 'hi %', 1; END $$; DROP VIEW IF EXISTS v";
    assert_eq!(
        run(&mut new, batch).await,
        vec![
            notice("NOTICE", "00000", "table \"a\" does not exist, skipping"),
            cc("DROP TABLE"),
            cc("CREATE TABLE"),
            notice("NOTICE", "00000", "hi 1"),
            cc("CALL"),
            notice("NOTICE", "00000", "view \"v\" does not exist, skipping"),
            cc("DROP VIEW"),
            ready(TxnStatus::Idle),
        ]
    );
    let mut old = connect(&engine, "u", 2).await;
    let quiet = "DROP TABLE IF EXISTS a; DROP VIEW IF EXISTS v";
    assert_eq!(
        run(&mut old, quiet).await,
        vec![cc("DROP TABLE"), cc("DROP VIEW"), ready(TxnStatus::Idle)]
    );
}

/// `PREPARE TRANSACTION` in a query of several statements: after a `BEGIN` in the same string it
/// prepares and ends the block, and later statements run on their own; in the implicit transaction
/// of a batch with no `BEGIN` it is refused with `25P01` and the batch rolls back.
/// `COMMIT PREPARED` inside the block of a batch is refused with `25001`.
#[tokio::test]
async fn prepare_transaction_inside_a_multi_statement_query() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = connect(&engine, "u", 3).await;
    run(&mut c, "CREATE TABLE t (n INT)").await;
    assert_eq!(
        run(
            &mut c,
            "BEGIN; INSERT INTO t VALUES (1); PREPARE TRANSACTION 'b1'; INSERT INTO t VALUES (2)"
        )
        .await,
        vec![
            cc("BEGIN"),
            cc("INSERT 1"),
            cc("PREPARE TRANSACTION"),
            cc("INSERT 1"),
            ready(TxnStatus::Idle),
        ]
    );
    // Only the statement after the prepare is visible yet.
    let seen = run(&mut c, "SELECT n FROM t").await;
    assert_eq!(seen.get(2), Some(&cc("SELECT 1")));
    // No BEGIN: an implicit transaction is not a block, so the batch fails and runs nothing.
    let refused = run(&mut c, "INSERT INTO t VALUES (3); PREPARE TRANSACTION 'b2'").await;
    assert_eq!(error_code(&refused), "25P01");
    assert_eq!(refused.last(), Some(&ready(TxnStatus::Idle)));
    let seen = run(&mut c, "SELECT n FROM t").await;
    assert_eq!(
        seen.get(2),
        Some(&cc("SELECT 1")),
        "the implicit insert rolled back"
    );
    // COMMIT PREPARED inside a block of the same string is refused and fails the block.
    let refused = run(&mut c, "BEGIN; COMMIT PREPARED 'b1'").await;
    assert_eq!(error_code(&refused), "25001");
    assert_eq!(refused.last(), Some(&ready(TxnStatus::Failed)));
    run(&mut c, "ROLLBACK").await;
    assert_eq!(
        run(&mut c, "COMMIT PREPARED 'b1'").await,
        vec![cc("COMMIT PREPARED"), ready(TxnStatus::Idle)]
    );
    let seen = run(&mut c, "SELECT n FROM t").await;
    assert_eq!(seen.get(3), Some(&cc("SELECT 2")));
}

/// On the extended protocol a row-returning statement's notices come after its last `DataRow` and
/// right before its `CommandComplete`: with a fetch-all `Execute` (the streaming path), and with a
/// paginated one, where they arrive with the last page.
#[tokio::test]
async fn extended_protocol_notices_come_right_before_the_completion() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = connect(&engine, "u", 3).await;
    run(&mut c, "CREATE TABLE t (n INT)").await;
    run(&mut c, "INSERT INTO t VALUES (1), (2), (3)").await;
    run(
        &mut c,
        "CREATE FUNCTION noisy(n INT) RETURNS INT LANGUAGE nusascript AS $$ BEGIN \
         RAISE NOTICE 'row %', $1; RETURN $1; END $$",
    )
    .await
    .first()
    .filter(|m| **m == cc("CREATE FUNCTION"))
    .expect("the function is created");
    let row = |n: &str| BackendMessage::DataRow {
        values: vec![Some(n.as_bytes().to_vec())],
    };
    let raised = || {
        ["1", "2", "3"]
            .iter()
            .map(|n| notice("NOTICE", "00000", &format!("row {n}")))
            .collect::<Vec<_>>()
    };
    for pages in [vec![0_u32], vec![2, 2]] {
        send(
            &mut c,
            FrontendMessage::Parse {
                name: String::new(),
                sql: "SELECT noisy(n) FROM t ORDER BY n".to_owned(),
                param_types: vec![],
            },
        )
        .await;
        send(
            &mut c,
            FrontendMessage::Bind {
                portal: String::new(),
                statement: String::new(),
                params: vec![],
                result_formats: vec![],
            },
        )
        .await;
        for &max_rows in &pages {
            send(
                &mut c,
                FrontendMessage::Execute {
                    portal: String::new(),
                    max_rows,
                },
            )
            .await;
        }
        send(&mut c, FrontendMessage::Sync).await;
        let mut got = Vec::new();
        loop {
            let msg = next(&mut c).await;
            let done = matches!(msg, BackendMessage::ReadyForQuery(_));
            got.push(msg);
            if done {
                break;
            }
        }
        let mut want = vec![BackendMessage::ParseComplete, BackendMessage::BindComplete];
        if pages.len() == 1 {
            want.extend([row("1"), row("2"), row("3")]);
        } else {
            want.extend([
                row("1"),
                row("2"),
                BackendMessage::PortalSuspended,
                row("3"),
            ]);
        }
        want.extend(raised());
        want.extend([cc("SELECT 3"), ready(TxnStatus::Idle)]);
        assert_eq!(got, want, "pages {pages:?}");
    }
}

/// `DETACH PARTITION` of a partition a prepared transaction wrote is refused like any other change
/// to its shape, and goes through once the transaction is finished.
#[tokio::test]
async fn detaching_a_partition_a_prepared_transaction_wrote_is_refused() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let mut c = connect(&engine, "u", 3).await;
    run(&mut c, "CREATE TABLE pt (v INT) PARTITION BY RANGE (v)").await;
    run(
        &mut c,
        "CREATE TABLE pt1 PARTITION OF pt FOR VALUES FROM (0) TO (100)",
    )
    .await;
    run(&mut c, "BEGIN").await;
    run(&mut c, "INSERT INTO pt VALUES (5)").await;
    run(&mut c, "PREPARE TRANSACTION 'd1'").await;
    assert_eq!(
        error_code(&run(&mut c, "ALTER TABLE pt DETACH PARTITION pt1").await),
        "55006"
    );
    run(&mut c, "COMMIT PREPARED 'd1'").await;
    assert_eq!(
        run(&mut c, "ALTER TABLE pt DETACH PARTITION pt1").await,
        vec![cc("ALTER TABLE"), ready(TxnStatus::Idle)]
    );
}
