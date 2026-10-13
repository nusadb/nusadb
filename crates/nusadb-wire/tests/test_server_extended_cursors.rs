//! Cursor statements, bit-string parameters, `COPY` column counts and composite type metadata as a
//! driver meets them over the wire.
//!
//! The cursor cases cover every path an extended-query `Execute` can take: the streamed fetch-all
//! of a text portal, a row limit that suspends the portal, and binary result formats. A `FETCH` must
//! be described with its cursor's columns and must find the connection's cursors on each of them.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "integration test harness asserts via unwrap/panic; linear protocol scripts"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::{ColumnType, StorageEngine};
use nusadb_wire::{
    BackendMessage, Connection, DescribeTarget, FrontendMessage, TxnStatus, column_type_tag,
    handle_client,
};
use tokio::io::DuplexStream;

type Conn = Connection<DuplexStream>;

async fn send(conn: &mut Conn, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

/// Every backend message up to and including the next `ReadyForQuery`.
async fn until_ready(conn: &mut Conn) -> Vec<BackendMessage> {
    let mut out = Vec::new();
    loop {
        let frame = conn.read_frame().await.unwrap().unwrap();
        let msg = BackendMessage::decode(&frame).unwrap();
        let done = matches!(msg, BackendMessage::ReadyForQuery(_));
        out.push(msg);
        if done {
            return out;
        }
    }
}

/// Open a trust-on-startup session as `user` at protocol `minor` on `engine`.
async fn connect(
    engine: &Arc<dyn StorageEngine>,
    user: &str,
    minor: u16,
) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let (client, server) = tokio::io::duplex(64 * 1024);
    let handle = tokio::spawn(handle_client(server, Arc::clone(engine)));
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
    let startup = until_ready(&mut conn).await;
    assert_eq!(startup.first(), Some(&BackendMessage::AuthOk));
    (conn, handle)
}

async fn finish(mut conn: Conn, handle: tokio::task::JoinHandle<std::io::Result<()>>) {
    send(&mut conn, FrontendMessage::Terminate).await;
    drop(conn);
    handle.await.unwrap().unwrap();
}

/// A simple query that must succeed; returns its messages.
async fn simple(conn: &mut Conn, sql: &str) -> Vec<BackendMessage> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let msgs = until_ready(conn).await;
    assert!(
        !msgs
            .iter()
            .any(|m| matches!(m, BackendMessage::Error { .. })),
        "`{sql}` failed: {msgs:?}"
    );
    msgs
}

/// One extended-query round trip: `Parse`, `Bind`, an optional `Describe(portal)`, one `Execute`
/// per entry of `executes` (each its `max_rows`), then `Sync`. Returns everything up to
/// `ReadyForQuery`.
struct Ext<'a> {
    sql: &'a str,
    params: Vec<Option<Vec<u8>>>,
    param_types: Vec<u8>,
    result_formats: Vec<u16>,
    describe: bool,
    executes: Vec<u32>,
}

impl<'a> Ext<'a> {
    fn new(sql: &'a str) -> Self {
        Self {
            sql,
            params: Vec::new(),
            param_types: Vec::new(),
            result_formats: Vec::new(),
            describe: false,
            executes: vec![0],
        }
    }

    fn param(mut self, tag: u8, value: &str) -> Self {
        self.params.push(Some(value.as_bytes().to_vec()));
        self.param_types.push(tag);
        self
    }

    fn binary(mut self) -> Self {
        self.result_formats = vec![1];
        self
    }

    const fn describe(mut self) -> Self {
        self.describe = true;
        self
    }

    fn executes(mut self, max_rows: &[u32]) -> Self {
        self.executes = max_rows.to_vec();
        self
    }

    async fn run(self, conn: &mut Conn) -> Vec<BackendMessage> {
        send(
            conn,
            FrontendMessage::Parse {
                name: String::new(),
                sql: self.sql.to_owned(),
                param_types: self.param_types,
            },
        )
        .await;
        send(
            conn,
            FrontendMessage::Bind {
                portal: String::new(),
                statement: String::new(),
                params: self.params,
                result_formats: self.result_formats,
            },
        )
        .await;
        if self.describe {
            send(
                conn,
                FrontendMessage::Describe {
                    target: DescribeTarget::Portal,
                    name: String::new(),
                },
            )
            .await;
        }
        for max_rows in self.executes {
            send(
                conn,
                FrontendMessage::Execute {
                    portal: String::new(),
                    max_rows,
                },
            )
            .await;
        }
        send(conn, FrontendMessage::Sync).await;
        let msgs = until_ready(conn).await;
        assert_eq!(msgs.first(), Some(&BackendMessage::ParseComplete));
        assert_eq!(msgs.get(1), Some(&BackendMessage::BindComplete));
        msgs.into_iter().skip(2).collect()
    }
}

fn cc(tag: &str) -> BackendMessage {
    BackendMessage::CommandComplete {
        tag: tag.to_owned(),
    }
}

fn text_row(cells: &[&str]) -> BackendMessage {
    BackendMessage::DataRow {
        values: cells.iter().map(|c| Some(c.as_bytes().to_vec())).collect(),
    }
}

fn names(cols: &[&str]) -> BackendMessage {
    BackendMessage::RowDescription {
        columns: cols.iter().map(|&c| c.to_owned()).collect(),
    }
}

const fn ready(status: TxnStatus) -> BackendMessage {
    BackendMessage::ReadyForQuery(status)
}

fn error_code(msgs: &[BackendMessage]) -> Option<String> {
    msgs.iter().find_map(|m| match m {
        BackendMessage::Error { code, .. } => Some(code.clone()),
        _ => None,
    })
}

async fn seeded(minor: u16) -> (Conn, tokio::task::JoinHandle<std::io::Result<()>>) {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut conn, handle) = connect(&engine, "u", minor).await;
    simple(&mut conn, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)").await;
    simple(
        &mut conn,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    .await;
    (conn, handle)
}

/// Inside a transaction, `Describe(portal)` of a `FETCH` reports the cursor's columns, and the
/// streamed `Execute` sends the rows after it.
#[tokio::test]
async fn fetch_in_a_transaction_is_described_with_the_cursor_columns() {
    let (mut conn, handle) = seeded(0).await;
    simple(&mut conn, "BEGIN").await;
    let declared = Ext::new("DECLARE c CURSOR FOR SELECT id, v FROM t ORDER BY id")
        .describe()
        .run(&mut conn)
        .await;
    assert_eq!(
        declared,
        vec![
            BackendMessage::NoData,
            cc("DECLARE CURSOR"),
            ready(TxnStatus::InTransaction)
        ]
    );
    let fetched = Ext::new("FETCH 2 FROM c").describe().run(&mut conn).await;
    assert_eq!(
        fetched,
        vec![
            names(&["id", "v"]),
            text_row(&["1", "a"]),
            text_row(&["2", "b"]),
            cc("FETCH 2"),
            ready(TxnStatus::InTransaction)
        ]
    );
    // `MOVE` returns no rows, so it describes as `NoData`.
    let moved = Ext::new("MOVE ABSOLUTE 0 IN c")
        .describe()
        .run(&mut conn)
        .await;
    assert_eq!(
        moved,
        vec![
            BackendMessage::NoData,
            cc("MOVE 0"),
            ready(TxnStatus::InTransaction)
        ]
    );
    simple(&mut conn, "COMMIT").await;
    finish(conn, handle).await;
}

/// At protocol minor 1 the `FETCH` description carries the cursor's column types.
#[tokio::test]
async fn fetch_description_is_typed_at_minor_one() {
    let (mut conn, handle) = seeded(1).await;
    simple(&mut conn, "BEGIN").await;
    Ext::new("DECLARE c CURSOR FOR SELECT id, v FROM t ORDER BY id")
        .run(&mut conn)
        .await;
    let fetched = Ext::new("FETCH 1 FROM c").describe().run(&mut conn).await;
    assert_eq!(
        fetched,
        vec![
            BackendMessage::RowDescriptionTyped {
                columns: vec![
                    ("id".to_owned(), column_type_tag(ColumnType::Int)),
                    ("v".to_owned(), column_type_tag(ColumnType::Text)),
                ],
            },
            text_row(&["1", "a"]),
            cc("FETCH 1"),
            ready(TxnStatus::InTransaction)
        ]
    );
    simple(&mut conn, "COMMIT").await;
    finish(conn, handle).await;
}

/// A `WITH HOLD` cursor outside any transaction serves `FETCH`, `MOVE` and `CLOSE` on the streamed
/// path, under a row limit (the portal suspends and resumes), and with binary results.
#[tokio::test]
async fn held_cursor_outside_a_transaction_works_on_every_execute_path() {
    let (mut conn, handle) = seeded(0).await;
    let declared = Ext::new("DECLARE h CURSOR WITH HOLD FOR SELECT id, v FROM t ORDER BY id")
        .run(&mut conn)
        .await;
    assert_eq!(declared, vec![cc("DECLARE CURSOR"), ready(TxnStatus::Idle)]);

    // A row limit of 1 over a two-row FETCH: one row, suspended, then the rest.
    let limited = Ext::new("FETCH 2 FROM h")
        .describe()
        .executes(&[1, 1])
        .run(&mut conn)
        .await;
    assert_eq!(
        limited,
        vec![
            names(&["id", "v"]),
            text_row(&["1", "a"]),
            BackendMessage::PortalSuspended,
            text_row(&["2", "b"]),
            cc("FETCH 2"),
            ready(TxnStatus::Idle)
        ]
    );

    // Binary results: the INT column arrives as an 8-byte big-endian integer.
    let binary = Ext::new("FETCH 1 FROM h").binary().run(&mut conn).await;
    assert_eq!(
        binary,
        vec![
            BackendMessage::DataRow {
                values: vec![Some(3_i64.to_be_bytes().to_vec()), Some(b"c".to_vec())],
            },
            cc("FETCH 1"),
            ready(TxnStatus::Idle)
        ]
    );

    let moved = Ext::new("MOVE FIRST IN h")
        .executes(&[5])
        .run(&mut conn)
        .await;
    assert_eq!(moved, vec![cc("MOVE 1"), ready(TxnStatus::Idle)]);
    let streamed = Ext::new("FETCH NEXT FROM h").run(&mut conn).await;
    assert_eq!(
        streamed,
        vec![text_row(&["2", "b"]), cc("FETCH 1"), ready(TxnStatus::Idle)]
    );

    let closed = Ext::new("CLOSE h").binary().run(&mut conn).await;
    assert_eq!(closed, vec![cc("CLOSE CURSOR"), ready(TxnStatus::Idle)]);
    let gone = Ext::new("FETCH 1 FROM h").describe().run(&mut conn).await;
    assert_eq!(error_code(&gone).as_deref(), Some("34000"));
    finish(conn, handle).await;
}

/// `DECLARE` under binary result formats succeeds (it returns no rows); the formats apply to the
/// `FETCH` that does. A cursor declared on the buffered path still closes at `COMMIT`, and one
/// without `WITH HOLD` is still refused outside a transaction.
#[tokio::test]
async fn declare_with_binary_result_formats() {
    let (mut conn, handle) = seeded(0).await;
    simple(&mut conn, "BEGIN").await;
    let declared = Ext::new("DECLARE c CURSOR FOR SELECT id FROM t ORDER BY id")
        .binary()
        .run(&mut conn)
        .await;
    assert_eq!(
        declared,
        vec![cc("DECLARE CURSOR"), ready(TxnStatus::InTransaction)]
    );
    let fetched = Ext::new("FETCH 2 FROM c")
        .binary()
        .describe()
        .run(&mut conn)
        .await;
    assert_eq!(
        fetched,
        vec![
            names(&["id"]),
            BackendMessage::DataRow {
                values: vec![Some(1_i64.to_be_bytes().to_vec())],
            },
            BackendMessage::DataRow {
                values: vec![Some(2_i64.to_be_bytes().to_vec())],
            },
            cc("FETCH 2"),
            ready(TxnStatus::InTransaction)
        ]
    );
    simple(&mut conn, "COMMIT").await;
    let closed_at_commit = Ext::new("FETCH 1 FROM c").binary().run(&mut conn).await;
    assert_eq!(error_code(&closed_at_commit).as_deref(), Some("34000"));

    let refused = Ext::new("DECLARE d CURSOR FOR SELECT id FROM t")
        .binary()
        .run(&mut conn)
        .await;
    assert_eq!(error_code(&refused).as_deref(), Some("25P01"));
    finish(conn, handle).await;
}

/// A parameter tagged BIT (0x14) or VARBIT (0x15) binds into a `BIT(n)` / `BIT VARYING(n)` column
/// when its length fits, is refused when it does not, and compares against a bit column.
#[tokio::test]
async fn bit_tagged_parameters_assign_into_bit_columns() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut conn, handle) = connect(&engine, "u", 0).await;
    simple(
        &mut conn,
        "CREATE TABLE b (id INT PRIMARY KEY, f BIT(4), v BIT VARYING(6))",
    )
    .await;
    for (sql, tag, value) in [
        ("INSERT INTO b (id, f) VALUES (1, $1)", 0x14, "1010"),
        ("INSERT INTO b (id, f) VALUES (2, $1)", 0x15, "0110"),
        ("INSERT INTO b (id, v) VALUES (3, $1)", 0x15, "101"),
        ("INSERT INTO b (id, v) VALUES (4, $1)", 0x14, "111111"),
        ("UPDATE b SET f = $1 WHERE id = 2", 0x14, "0001"),
    ] {
        let msgs = Ext::new(sql).param(tag, value).run(&mut conn).await;
        assert_eq!(
            error_code(&msgs),
            None,
            "{sql} with tag {tag:#04x}: {msgs:?}"
        );
    }
    // A BIT length mismatch is 22026 and a BIT VARYING overflow 22001, as for a literal.
    for (sql, tag, value, code) in [
        (
            "INSERT INTO b (id, f) VALUES (5, $1)",
            0x14,
            "10101",
            "22026",
        ),
        ("INSERT INTO b (id, f) VALUES (6, $1)", 0x15, "101", "22026"),
        (
            "INSERT INTO b (id, v) VALUES (7, $1)",
            0x15,
            "1010101",
            "22001",
        ),
    ] {
        let msgs = Ext::new(sql).param(tag, value).run(&mut conn).await;
        assert_eq!(
            error_code(&msgs).as_deref(),
            Some(code),
            "{sql} with {value}"
        );
    }
    let rows = simple(&mut conn, "SELECT id, f, v FROM b ORDER BY id").await;
    assert_eq!(
        rows.into_iter()
            .filter(|m| matches!(m, BackendMessage::DataRow { .. }))
            .collect::<Vec<_>>(),
        vec![
            BackendMessage::DataRow {
                values: vec![Some(b"1".to_vec()), Some(b"1010".to_vec()), None]
            },
            BackendMessage::DataRow {
                values: vec![Some(b"2".to_vec()), Some(b"0001".to_vec()), None]
            },
            BackendMessage::DataRow {
                values: vec![Some(b"3".to_vec()), None, Some(b"101".to_vec())]
            },
            BackendMessage::DataRow {
                values: vec![Some(b"4".to_vec()), None, Some(b"111111".to_vec())]
            },
        ]
    );
    let found = Ext::new("SELECT id FROM b WHERE f = $1")
        .param(0x14, "1010")
        .run(&mut conn)
        .await;
    assert_eq!(
        found,
        vec![text_row(&["1"]), cc("SELECT 1"), ready(TxnStatus::Idle)]
    );
    finish(conn, handle).await;
}

/// `CopyOutResponse` and `CopyInResponse` announce the number of columns copied: the listed ones,
/// the whole table's, or the query's output.
#[tokio::test]
async fn copy_responses_report_the_column_count() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut conn, handle) = connect(&engine, "u", 0).await;
    simple(&mut conn, "CREATE TABLE t (id INT, a TEXT, b TEXT)").await;
    simple(&mut conn, "INSERT INTO t VALUES (1, 'x', 'y')").await;
    for (sql, columns) in [
        ("COPY t TO STDOUT", 3),
        ("COPY t (b, id) TO STDOUT", 2),
        ("COPY (SELECT id, a FROM t) TO STDOUT", 2),
    ] {
        let msgs = simple(&mut conn, sql).await;
        assert_eq!(
            msgs.first(),
            Some(&BackendMessage::CopyOutResponse { format: 0, columns }),
            "{sql}"
        );
    }
    for (sql, columns) in [("COPY t FROM STDIN", 3), ("COPY t (a) FROM STDIN", 1)] {
        send(
            &mut conn,
            FrontendMessage::Query {
                sql: sql.to_owned(),
            },
        )
        .await;
        let frame = conn.read_frame().await.unwrap().unwrap();
        assert_eq!(
            BackendMessage::decode(&frame).unwrap(),
            BackendMessage::CopyInResponse { format: 0, columns },
            "{sql}"
        );
        send(&mut conn, FrontendMessage::CopyDone).await;
        assert_eq!(
            until_ready(&mut conn).await,
            vec![cc("COPY 0"), ready(TxnStatus::Idle)]
        );
    }
    finish(conn, handle).await;
}

/// A role without superuser rights reads composite type definitions from `information_schema`,
/// while the internal catalog stays closed to it.
#[tokio::test]
async fn composite_types_are_readable_through_information_schema() {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (mut root, root_handle) = connect(&engine, nusadb_sql::BOOTSTRAP_SUPERUSER, 0).await;
    simple(
        &mut root,
        "CREATE TYPE money_pair AS (amount NUMERIC(10,2), label VARCHAR(20))",
    )
    .await;
    simple(&mut root, "CREATE ROLE reader LOGIN").await;
    finish(root, root_handle).await;

    let (mut reader, reader_handle) = connect(&engine, "reader", 0).await;
    let attributes = simple(
        &mut reader,
        "SELECT udt_name, attribute_name, ordinal_position, data_type, numeric_precision, \
         numeric_scale, character_maximum_length FROM information_schema.attributes \
         ORDER BY ordinal_position",
    )
    .await;
    assert_eq!(
        attributes
            .into_iter()
            .filter(|m| matches!(m, BackendMessage::DataRow { .. }))
            .collect::<Vec<_>>(),
        vec![
            BackendMessage::DataRow {
                values: vec![
                    Some(b"money_pair".to_vec()),
                    Some(b"amount".to_vec()),
                    Some(b"1".to_vec()),
                    Some(b"numeric".to_vec()),
                    Some(b"10".to_vec()),
                    Some(b"2".to_vec()),
                    None,
                ]
            },
            BackendMessage::DataRow {
                values: vec![
                    Some(b"money_pair".to_vec()),
                    Some(b"label".to_vec()),
                    Some(b"2".to_vec()),
                    Some(b"character varying".to_vec()),
                    None,
                    None,
                    Some(b"20".to_vec()),
                ]
            },
        ]
    );
    let types = simple(
        &mut reader,
        "SELECT user_defined_type_name, user_defined_type_category \
         FROM information_schema.user_defined_types",
    )
    .await;
    assert!(types.contains(&text_row(&["money_pair", "STRUCTURED"])));

    send(
        &mut reader,
        FrontendMessage::Query {
            sql: "SELECT * FROM nusadb_composites".to_owned(),
        },
    )
    .await;
    assert_eq!(
        error_code(&until_ready(&mut reader).await).as_deref(),
        Some("42501")
    );
    finish(reader, reader_handle).await;
}

/// An error raised while describing a portal or a statement inside a transaction block fails the
/// block, as an error raised by `Execute` does; outside a block it leaves the session idle.
#[tokio::test]
async fn a_describe_error_fails_the_transaction_block() {
    let (mut conn, handle) = seeded(0).await;
    for sql in ["FETCH 1 FROM nosuch", "SELECT * FROM nosuchtable"] {
        simple(&mut conn, "BEGIN").await;
        let msgs = Ext::new(sql).describe().run(&mut conn).await;
        assert!(error_code(&msgs).is_some(), "{sql}: {msgs:?}");
        assert_eq!(msgs.last(), Some(&ready(TxnStatus::Failed)), "{sql}");
        send(
            &mut conn,
            FrontendMessage::Query {
                sql: "SELECT 1".to_owned(),
            },
        )
        .await;
        let next = until_ready(&mut conn).await;
        assert_eq!(
            error_code(&next).as_deref(),
            Some("25P02"),
            "{sql}: {next:?}"
        );
        let rolled = simple(&mut conn, "COMMIT").await;
        assert_eq!(rolled.first(), Some(&cc("ROLLBACK")));
    }

    // Describe of an unknown prepared statement fails the block too.
    simple(&mut conn, "BEGIN").await;
    send(
        &mut conn,
        FrontendMessage::Describe {
            target: DescribeTarget::Statement,
            name: "nosuch".to_owned(),
        },
    )
    .await;
    send(&mut conn, FrontendMessage::Sync).await;
    let msgs = until_ready(&mut conn).await;
    assert_eq!(msgs.last(), Some(&ready(TxnStatus::Failed)));
    simple(&mut conn, "ROLLBACK").await;

    // Outside a block the error is reported and the session stays idle.
    let msgs = Ext::new("FETCH 1 FROM nosuch")
        .describe()
        .run(&mut conn)
        .await;
    assert_eq!(error_code(&msgs).as_deref(), Some("34000"));
    assert_eq!(msgs.last(), Some(&ready(TxnStatus::Idle)));
    finish(conn, handle).await;
}

/// SQL-level `PREPARE` / `EXECUTE` / `DEALLOCATE` work on the buffered extended-query paths (a row
/// limit, binary results), and a `Describe` of an `EXECUTE` portal reports the prepared
/// statement's columns.
#[tokio::test]
async fn sql_prepared_statements_work_on_every_execute_path() {
    let (mut conn, handle) = seeded(0).await;
    let prepared = Ext::new("PREPARE all_rows AS SELECT id, v FROM t ORDER BY id")
        .binary()
        .run(&mut conn)
        .await;
    assert_eq!(prepared, vec![cc("PREPARE"), ready(TxnStatus::Idle)]);
    simple(
        &mut conn,
        "PREPARE one_row(INT) AS SELECT v FROM t WHERE id = $1",
    )
    .await;

    // Streamed fetch-all: described with the prepared statement's columns.
    let streamed = Ext::new("EXECUTE all_rows").describe().run(&mut conn).await;
    assert_eq!(
        streamed,
        vec![
            names(&["id", "v"]),
            text_row(&["1", "a"]),
            text_row(&["2", "b"]),
            text_row(&["3", "c"]),
            cc("SELECT 3"),
            ready(TxnStatus::Idle)
        ]
    );
    let with_arg = Ext::new("EXECUTE one_row(2)")
        .describe()
        .run(&mut conn)
        .await;
    assert_eq!(
        with_arg,
        vec![
            names(&["v"]),
            text_row(&["b"]),
            cc("SELECT 1"),
            ready(TxnStatus::Idle)
        ]
    );

    // A row limit suspends and resumes.
    let limited = Ext::new("EXECUTE all_rows")
        .describe()
        .executes(&[2, 2])
        .run(&mut conn)
        .await;
    assert_eq!(
        limited,
        vec![
            names(&["id", "v"]),
            text_row(&["1", "a"]),
            text_row(&["2", "b"]),
            BackendMessage::PortalSuspended,
            text_row(&["3", "c"]),
            cc("SELECT 3"),
            ready(TxnStatus::Idle)
        ]
    );

    // Binary results, inside a transaction block.
    simple(&mut conn, "BEGIN").await;
    let binary = Ext::new("EXECUTE one_row(3)").binary().run(&mut conn).await;
    assert_eq!(
        binary,
        vec![
            BackendMessage::DataRow {
                values: vec![Some(b"c".to_vec())],
            },
            cc("SELECT 1"),
            ready(TxnStatus::InTransaction)
        ]
    );
    simple(&mut conn, "COMMIT").await;

    let deallocated = Ext::new("DEALLOCATE all_rows")
        .executes(&[5])
        .run(&mut conn)
        .await;
    assert_eq!(deallocated, vec![cc("DEALLOCATE"), ready(TxnStatus::Idle)]);
    let gone = Ext::new("EXECUTE all_rows").describe().run(&mut conn).await;
    assert_eq!(error_code(&gone).as_deref(), Some("26000"));
    finish(conn, handle).await;
}
