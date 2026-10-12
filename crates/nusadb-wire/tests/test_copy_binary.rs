//! Binary `COPY` over the wire: `COPY ... FROM STDIN (FORMAT binary)` and
//! `COPY ... TO STDOUT (FORMAT binary)`, driven over an in-memory duplex stream against a real
//! `BtreeEngine`.
//!
//! The stream layout is built by hand here, from the protocol document, rather than with the
//! server's own writer, so a test that passes shows the server agrees with the document.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    reason = "integration test harness asserts via unwrap/panic; linear protocol scripts; field \
              builders return the row's optional field type"
)]

use std::sync::Arc;
use std::time::Duration;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{BackendMessage, Connection, FrontendMessage, handle_client, handle_client_with};
use tokio::io::{AsyncRead, AsyncWrite, DuplexStream};

/// The binary stream signature from the protocol document.
const SIGNATURE: [u8; 11] = [
    0x4E, 0x55, 0x53, 0x41, 0x43, 0x4F, 0x50, 0x59, 0xFF, 0x0D, 0x0A,
];

/// One field of a hand-built row: `None` is SQL NULL.
type Field = Option<Vec<u8>>;

/// The stream header: signature, zero flags, empty extension.
fn header() -> Vec<u8> {
    let mut out = SIGNATURE.to_vec();
    out.extend_from_slice(&0_u32.to_be_bytes());
    out.extend_from_slice(&0_u32.to_be_bytes());
    out
}

/// One encoded row.
fn row(fields: &[Field]) -> Vec<u8> {
    let mut out = i16::try_from(fields.len()).unwrap().to_be_bytes().to_vec();
    for field in fields {
        match field {
            None => out.extend_from_slice(&(-1_i32).to_be_bytes()),
            Some(bytes) => {
                out.extend_from_slice(&i32::try_from(bytes.len()).unwrap().to_be_bytes());
                out.extend_from_slice(bytes);
            },
        }
    }
    out
}

/// A complete stream: header, the rows, trailer.
fn stream(rows: &[Vec<Field>]) -> Vec<u8> {
    let mut out = header();
    for fields in rows {
        out.extend_from_slice(&row(fields));
    }
    out.extend_from_slice(&(-1_i16).to_be_bytes());
    out
}

fn int(i: i64) -> Field {
    Some(i.to_be_bytes().to_vec())
}

fn text(s: &str) -> Field {
    Some(s.as_bytes().to_vec())
}

async fn next<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) -> BackendMessage {
    let frame = conn.read_frame().await.unwrap().unwrap();
    BackendMessage::decode(&frame).unwrap()
}

async fn send<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>, msg: FrontendMessage) {
    conn.write_frame(&msg.encode().unwrap()).await.unwrap();
}

/// Read up to and including the next `ReadyForQuery`.
async fn until_ready<S: AsyncRead + AsyncWrite + Unpin>(conn: &mut Connection<S>) {
    while !matches!(next(conn).await, BackendMessage::ReadyForQuery(_)) {}
}

/// Open a session on a fresh engine, with an optional `COPY FROM` byte cap.
async fn session(copy_cap: Option<usize>) -> Connection<DuplexStream> {
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let (client, server) = tokio::io::duplex(1 << 20);
    match copy_cap {
        None => drop(tokio::spawn(handle_client(server, engine))),
        Some(cap) => drop(tokio::spawn(handle_client_with(
            server,
            engine,
            None,
            None,
            None,
            None,
            None,
            Some(Duration::from_mins(1)),
            Some(cap),
        ))),
    }
    let mut conn = Connection::new(client);
    send(
        &mut conn,
        FrontendMessage::Startup {
            major: 1,
            minor: 0,
            user: "u".to_owned(),
            database: "d".to_owned(),
        },
    )
    .await;
    assert_eq!(next(&mut conn).await, BackendMessage::AuthOk);
    until_ready(&mut conn).await;
    conn
}

/// Run one statement; its text rows, or the SQLSTATE of its error.
async fn run(conn: &mut Connection<DuplexStream>, sql: &str) -> Result<Vec<Vec<Field>>, String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let mut rows = Vec::new();
    let mut error = None;
    loop {
        match next(conn).await {
            BackendMessage::DataRow { values } => rows.push(values),
            BackendMessage::Error { code, message } => error = Some(format!("{code}: {message}")),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    error.map_or(Ok(rows), Err)
}

/// Run a statement that must succeed.
async fn ok(conn: &mut Connection<DuplexStream>, sql: &str) -> Vec<Vec<Field>> {
    run(conn, sql)
        .await
        .unwrap_or_else(|e| panic!("{sql} failed: {e}"))
}

/// Run `COPY ... TO STDOUT`; the response's format byte, the joined payload, and the tag.
async fn copy_out(conn: &mut Connection<DuplexStream>, sql: &str) -> (u8, Vec<u8>, String) {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    let format = match next(conn).await {
        BackendMessage::CopyOutResponse { format, .. } => format,
        other => panic!("{sql}: expected CopyOutResponse, got {other:?}"),
    };
    let mut payload = Vec::new();
    loop {
        match next(conn).await {
            BackendMessage::CopyData { data } => payload.extend_from_slice(&data),
            BackendMessage::CopyDone => break,
            other => panic!("{sql}: unexpected {other:?} in the copy-out stream"),
        }
    }
    let tag = match next(conn).await {
        BackendMessage::CommandComplete { tag } => tag,
        other => panic!("{sql}: expected CommandComplete, got {other:?}"),
    };
    until_ready(conn).await;
    (format, payload, tag)
}

/// Run `COPY ... FROM STDIN`, sending `chunks` as one `CopyData` each. The tag on success, or the
/// SQLSTATE on failure; the response's format byte must be `format`.
async fn copy_in(
    conn: &mut Connection<DuplexStream>,
    sql: &str,
    format: u8,
    chunks: &[&[u8]],
) -> Result<String, String> {
    send(
        conn,
        FrontendMessage::Query {
            sql: sql.to_owned(),
        },
    )
    .await;
    match next(conn).await {
        BackendMessage::CopyInResponse { format: got, .. } => {
            assert_eq!(got, format, "{sql}: CopyInResponse format byte");
        },
        other => panic!("{sql}: expected CopyInResponse, got {other:?}"),
    }
    for chunk in chunks {
        send(
            conn,
            FrontendMessage::CopyData {
                data: chunk.to_vec(),
            },
        )
        .await;
    }
    send(conn, FrontendMessage::CopyDone).await;
    let result = match next(conn).await {
        BackendMessage::CommandComplete { tag } => Ok(tag),
        BackendMessage::Error { code, .. } => Err(code),
        other => panic!("{sql}: expected the load's outcome, got {other:?}"),
    };
    until_ready(conn).await;
    result
}

/// A table with a column of every value type, and its three rows: ordinary values, all NULLs, and
/// edge values (empty strings, an empty BYTEA, extreme integers).
async fn every_type_table(conn: &mut Connection<DuplexStream>, name: &str) {
    ok(conn, &format!("CREATE TABLE {name} ({EVERY_TYPE_COLUMNS})")).await;
}

const EVERY_TYPE_COLUMNS: &str = "id INT PRIMARY KEY, b BOOL, i INT, si SMALLINT, bi BIGINT, \
     f DOUBLE PRECISION, r REAL, n NUMERIC, n2 NUMERIC(10,2), t TEXT, vc VARCHAR(10), ch CHAR(3), \
     raw BYTEA, d DATE, tm TIME, ttz TIMETZ, ts TIMESTAMP, tstz TIMESTAMPTZ, iv INTERVAL, u UUID, \
     j JSON, jb JSONB, ia INT[], ta TEXT[], v VECTOR(3), m MACADDR, m8 MACADDR8, ip INET, cn CIDR, \
     bt BIT(4), vb VARBIT(8), ir INT4RANGE, nr NUMRANGE, dr DATERANGE, tr TSRANGE, \
     tzr TSTZRANGE, p POINT, bx BOX, tv TSVECTOR, tq TSQUERY, x XML, e mood";

/// Every column of [`EVERY_TYPE_COLUMNS`] but `id`, in order.
const EVERY_TYPE_NAMES: &str = "b, i, si, bi, f, r, n, n2, t, vc, ch, raw, d, tm, ttz, ts, tstz, \
     iv, u, j, jb, ia, ta, v, m, m8, ip, cn, bt, vb, ir, nr, dr, tr, tzr, p, bx, tv, tq, x, e";

const EVERY_TYPE_ROWS: &str = "INSERT INTO src VALUES \
    (1, true, -42, 7, 9000000000, 1.5, 2.25, 123.456, 99.95, 'héllo', 'abc', 'xy', '\\xdeadbeef', \
     '2024-02-29', '13:45:30.5', '13:45:30+07', '2024-01-02 03:04:05.000006', \
     '2024-01-02 03:04:05+00', '1 year 2 days 03:04:05', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', \
     '{\"a\": [1, 2]}', '{\"k\": \"v\"}', ARRAY[1, 2, 3], ARRAY['a', 'b c'], '[1,2,3]', \
     '08:00:2b:01:02:03', '08:00:2b:01:02:03:04:05', '192.168.1.5/24', '10.0.0.0/8', B'1010', \
     '101', '[1,10)', '[1.5,2.5]', '[2024-01-01,2024-02-01)', \
     '[2024-01-01 00:00,2024-01-02 00:00)', '[2024-01-01 00:00+00,2024-01-02 00:00+00)', \
     '(1,2)', '(3,4),(1,2)', 'a:1 b:2', 'a & b', '<r>x</r>', 'ok'), \
    (2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL), \
    (3, false, -2147483648, -32768, -9223372036854775808, -0.25, 0, -0.001, 0, '', '', \
     '', '\\x', '1900-01-01', '00:00:00', '00:00:00-05:30', '1969-12-31 23:59:59', \
     '2000-01-01 00:00:00+00', '-3 mons -1 days', '00000000-0000-0000-0000-000000000000', 'null', \
     '[]', ARRAY[]::INT[], ARRAY['']::TEXT[], '[0,0,0]', '00:00:00:00:00:00', \
     'ff:ff:ff:ff:ff:ff:ff:ff', '::1', '2001:db8::/32', B'0000', '', 'empty', '(,)', \
     '[2024-01-01,)', '(,2024-01-01 00:00)', 'empty', '(0,0)', '(0,0),(0,0)', '', 'x', '', 'sad')";

#[tokio::test]
async fn every_type_round_trips_through_binary_copy() {
    let mut conn = session(None).await;
    ok(&mut conn, "CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')").await;
    every_type_table(&mut conn, "src").await;
    every_type_table(&mut conn, "dst").await;
    ok(&mut conn, EVERY_TYPE_ROWS).await;

    let (format, payload, tag) = copy_out(&mut conn, "COPY src TO STDOUT (FORMAT binary)").await;
    assert_eq!(format, 1, "a binary COPY TO reports overall format 1");
    assert_eq!(tag, "COPY 3");
    assert!(
        payload.starts_with(&header()),
        "the stream opens with the header"
    );
    assert!(
        payload.ends_with(&[0xFF, 0xFF]),
        "the stream ends with the trailer"
    );

    let loaded = copy_in(
        &mut conn,
        "COPY dst FROM STDIN (FORMAT binary)",
        1,
        &[&payload],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 3".to_owned()));

    // Binary out of the copy equals binary out of the original, byte for byte, and so does the text
    // export of every column.
    let (_, again, _) = copy_out(&mut conn, "COPY dst TO STDOUT (FORMAT binary)").await;
    assert_eq!(again, payload);
    let (_, src_text, _) = copy_out(&mut conn, "COPY src TO STDOUT").await;
    let (_, dst_text, _) = copy_out(&mut conn, "COPY dst TO STDOUT").await;
    assert_eq!(dst_text, src_text);
    // And every column compares equal, row by row, to its original.
    let qualified = |table: &str| {
        EVERY_TYPE_NAMES
            .split(", ")
            // Compared through text; the enum, which does not cast to text, is checked below.
            .filter(|&c| c != "e")
            .map(|c| format!("{table}.{c}::TEXT"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let differing = ok(
        &mut conn,
        &format!(
            "SELECT count(*) FROM src JOIN dst ON src.id = dst.id WHERE ({}) IS DISTINCT FROM ({})",
            qualified("src"),
            qualified("dst")
        ),
    )
    .await;
    assert_eq!(differing, vec![vec![text("0")]]);
    assert_eq!(
        ok(
            &mut conn,
            "SELECT count(*) FROM src JOIN dst ON src.id = dst.id"
        )
        .await,
        vec![vec![text("3")]]
    );
    assert_eq!(
        ok(&mut conn, "SELECT e FROM dst ORDER BY id").await,
        vec![vec![text("ok")], vec![None], vec![text("sad")]]
    );

    // The NULL row is all NULL markers, and the edge row keeps its empty strings and empty BYTEA
    // as values, not NULLs.
    let edge = ok(
        &mut conn,
        "SELECT t IS NULL, t = '', raw IS NULL, length(raw), vc = '', ch IS NULL FROM dst WHERE id = 3",
    )
    .await;
    assert_eq!(
        edge,
        vec![vec![
            text("false"),
            text("true"),
            text("false"),
            text("0"),
            text("true"),
            text("false")
        ]]
    );
    let nulls = ok(
        &mut conn,
        "SELECT count(*) FROM dst WHERE id = 2 AND b IS NULL AND raw IS NULL AND e IS NULL",
    )
    .await;
    assert_eq!(nulls, vec![vec![text("1")]]);
}

/// A load sent one byte per `CopyData` frame is reassembled into the same stream.
#[tokio::test]
async fn a_load_split_one_byte_per_frame_is_reassembled() {
    let mut conn = session(None).await;
    ok(&mut conn, "CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')").await;
    every_type_table(&mut conn, "src").await;
    every_type_table(&mut conn, "dst").await;
    ok(&mut conn, EVERY_TYPE_ROWS).await;
    let (_, payload, _) = copy_out(&mut conn, "COPY src TO STDOUT (FORMAT binary)").await;

    let bytes: Vec<&[u8]> = payload.chunks(1).collect();
    assert!(bytes.len() > 100, "the stream spans many frames");
    let loaded = copy_in(&mut conn, "COPY dst FROM STDIN (FORMAT binary)", 1, &bytes).await;
    assert_eq!(loaded, Ok("COPY 3".to_owned()));
    let (_, again, _) = copy_out(&mut conn, "COPY dst TO STDOUT (FORMAT binary)").await;
    assert_eq!(again, payload);
}

/// Each malformation is refused with its SQLSTATE, and the valid row before it is not kept: the
/// load is all or nothing.
#[tokio::test]
async fn malformed_streams_are_refused_and_nothing_is_loaded() {
    let mut conn = session(None).await;
    ok(&mut conn, "CREATE TABLE m (id INT, name TEXT)").await;
    let good = row(&[int(1), text("one")]);
    let trailer = (-1_i16).to_be_bytes();
    let with_good = |tail: &[u8]| {
        let mut out = header();
        out.extend_from_slice(&good);
        out.extend_from_slice(tail);
        out
    };

    let mut bad_signature = stream(&[vec![int(1), text("one")]]);
    bad_signature[0] = b'P';
    let mut flagged = stream(&[vec![int(1), text("one")]]);
    flagged[14] = 1;
    let mut past_end = Vec::new();
    past_end.extend_from_slice(&2_i16.to_be_bytes());
    past_end.extend_from_slice(&8_i32.to_be_bytes());
    past_end.extend_from_slice(&2_i64.to_be_bytes());
    past_end.extend_from_slice(&100_i32.to_be_bytes());
    past_end.extend_from_slice(b"abc");
    let mut negative_length = row(&[int(2)]);
    negative_length.truncate(2);
    negative_length.extend_from_slice(&(-2_i32).to_be_bytes());

    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("bad signature", bad_signature, "22P04"),
        ("a set flag bit", flagged, "22P04"),
        ("an empty stream", Vec::new(), "22P04"),
        ("a cut-off header", header()[..13].to_vec(), "22P04"),
        (
            "too few fields",
            with_good(&[row(&[int(2)]), trailer.to_vec()].concat()),
            "22P04",
        ),
        (
            "too many fields",
            with_good(&[row(&[int(2), text("b"), text("c")]), trailer.to_vec()].concat()),
            "22P04",
        ),
        ("a length past the end", with_good(&past_end), "22P04"),
        ("a negative length", with_good(&negative_length), "22P04"),
        (
            "a 4-byte INT",
            with_good(
                &[
                    row(&[Some(2_i32.to_be_bytes().to_vec()), text("b")]),
                    trailer.to_vec(),
                ]
                .concat(),
            ),
            "22P04",
        ),
        (
            "invalid UTF-8 in TEXT",
            with_good(&[row(&[int(2), Some(vec![0xC3, 0x28])]), trailer.to_vec()].concat()),
            "22021",
        ),
        ("a missing trailer", with_good(&[]), "22P04"),
        (
            "bytes after the trailer",
            with_good(&[&trailer[..], b"x"].concat()),
            "22P04",
        ),
    ];
    for (what, data, code) in cases {
        let result = copy_in(&mut conn, "COPY m FROM STDIN (FORMAT binary)", 1, &[&data]).await;
        assert_eq!(result, Err(code.to_owned()), "{what}");
        assert_eq!(
            ok(&mut conn, "SELECT count(*) FROM m").await,
            vec![vec![text("0")]],
            "{what}: nothing may be loaded"
        );
    }

    // The same stream, well formed, loads.
    let result = copy_in(
        &mut conn,
        "COPY m FROM STDIN (FORMAT binary)",
        1,
        &[&with_good(&trailer)],
    )
    .await;
    assert_eq!(result, Ok("COPY 1".to_owned()));
}

/// `COPY (query) TO STDOUT (FORMAT binary)` writes the query's rows in projection order, each field
/// in its binary layout.
#[tokio::test]
async fn a_query_exports_as_a_binary_stream() {
    let mut conn = session(None).await;
    ok(&mut conn, "CREATE TABLE q (id INT, name TEXT, flag BOOL)").await;
    ok(
        &mut conn,
        "INSERT INTO q VALUES (1, 'a', true), (2, NULL, false), (3, '', NULL)",
    )
    .await;
    let (format, payload, tag) = copy_out(
        &mut conn,
        "COPY (SELECT flag, id * 10, name FROM q WHERE id < 3 ORDER BY id DESC) TO STDOUT \
         (FORMAT binary)",
    )
    .await;
    assert_eq!(format, 1);
    assert_eq!(tag, "COPY 2");
    assert_eq!(
        payload,
        stream(&[
            vec![Some(vec![0]), int(20), None],
            vec![Some(vec![1]), int(10), text("a")],
        ])
    );

    // The empty string and the NULL stay apart.
    let (_, payload, _) = copy_out(&mut conn, "COPY q (name, id) TO STDOUT WITH BINARY").await;
    assert_eq!(
        payload,
        stream(&[
            vec![text("a"), int(1)],
            vec![None, int(2)],
            vec![text(""), int(3)],
        ])
    );
}

/// A column list loads the named columns in its order; the others take their defaults.
#[tokio::test]
async fn a_column_subset_loads_and_the_rest_take_defaults() {
    let mut conn = session(None).await;
    ok(
        &mut conn,
        "CREATE TABLE c (id INT, note TEXT DEFAULT 'none', n INT DEFAULT 7)",
    )
    .await;
    let loaded = copy_in(
        &mut conn,
        "COPY c (n, id) FROM STDIN BINARY",
        1,
        &[&stream(&[vec![int(5), int(1)], vec![None, int(2)]])],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 2".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT id, note, n FROM c ORDER BY id").await,
        vec![
            vec![text("1"), text("none"), text("5")],
            vec![text("2"), text("none"), None],
        ]
    );
}

/// A constraint violation anywhere in the stream rolls the whole load back, with the same SQLSTATE
/// an INSERT reports; triggers fire for each loaded row as they do for a text load.
#[tokio::test]
async fn constraints_and_triggers_apply_to_a_binary_load() {
    let mut conn = session(None).await;
    ok(
        &mut conn,
        "CREATE TABLE u (id INT PRIMARY KEY, v INT NOT NULL CHECK (v > 0))",
    )
    .await;
    ok(&mut conn, "CREATE TABLE audit (n INT)").await;
    ok(
        &mut conn,
        "CREATE TRIGGER tg AFTER INSERT ON u FOR EACH ROW INSERT INTO audit VALUES (1)",
    )
    .await;

    let cases = [
        (vec![vec![int(1), int(1)], vec![int(1), int(2)]], "23505"),
        (vec![vec![int(1), int(1)], vec![int(2), int(-1)]], "23514"),
        (vec![vec![int(1), int(1)], vec![int(2), None]], "23502"),
    ];
    for (rows, code) in cases {
        let result = copy_in(
            &mut conn,
            "COPY u FROM STDIN (FORMAT binary)",
            1,
            &[&stream(&rows)],
        )
        .await;
        assert_eq!(result, Err(code.to_owned()));
        assert_eq!(
            ok(&mut conn, "SELECT count(*) FROM u").await,
            vec![vec![text("0")]],
            "{code}: the first row must be rolled back too"
        );
    }
    assert_eq!(
        ok(&mut conn, "SELECT count(*) FROM audit").await,
        vec![vec![text("0")]],
        "a rolled-back load leaves no trigger effects"
    );

    let result = copy_in(
        &mut conn,
        "COPY u FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![int(1), int(1)], vec![int(2), int(2)]])],
    )
    .await;
    assert_eq!(result, Ok("COPY 2".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT count(*) FROM audit").await,
        vec![vec![text("2")]]
    );

    // A NULL for a NOT NULL column is refused in the text format as well.
    let result = copy_in(&mut conn, "COPY u FROM STDIN", 0, &[b"3\t3\n4\t\\N\n"]).await;
    assert_eq!(result, Err("23502".to_owned()));
}

/// The `COPY FROM` byte cap counts binary bytes the same way it counts text.
#[tokio::test]
async fn the_copy_byte_cap_applies_to_a_binary_load() {
    let mut conn = session(Some(24)).await;
    ok(&mut conn, "CREATE TABLE t (id INT)").await;
    let data = stream(&[vec![int(1)], vec![int(2)]]);
    assert!(data.len() > 24);
    let result = copy_in(
        &mut conn,
        "COPY t FROM STDIN (FORMAT binary)",
        1,
        &[&data[..10], &data[10..]],
    )
    .await;
    assert_eq!(result, Err("53400".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT count(*) FROM t").await,
        vec![vec![text("0")]]
    );
}

/// Text and CSV `COPY` keep their format byte `0` and their data.
#[tokio::test]
async fn text_and_csv_copy_are_unchanged() {
    let mut conn = session(None).await;
    ok(&mut conn, "CREATE TABLE t (id INT, name TEXT)").await;
    // A text payload that is not UTF-8 is refused before anything loads.
    let refused = copy_in(&mut conn, "COPY t FROM STDIN", 0, &[b"9\t\xC3\x28\n"]).await;
    assert_eq!(refused, Err("22021".to_owned()));
    let loaded = copy_in(&mut conn, "COPY t FROM STDIN", 0, &[b"1\talice\n2\t\\N\n"]).await;
    assert_eq!(loaded, Ok("COPY 2".to_owned()));
    let loaded = copy_in(
        &mut conn,
        "COPY t FROM STDIN (FORMAT csv)",
        0,
        &[b"3,\"b,c\"\n4,\n"],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 2".to_owned()));
    assert_eq!(
        copy_out(&mut conn, "COPY t TO STDOUT").await,
        (
            0,
            b"1\talice\n2\t\\N\n3\tb,c\n4\t\\N\n".to_vec(),
            "COPY 4".to_owned()
        )
    );
    assert_eq!(
        copy_out(&mut conn, "COPY t TO STDOUT (FORMAT csv, HEADER)").await,
        (
            0,
            b"id,name\n1,alice\n2,\n3,\"b,c\"\n4,\n".to_vec(),
            "COPY 4".to_owned()
        )
    );
}

/// The binary form of a range: kind tag, flags, then each finite bound little-endian.
fn range_field(kind: u8, flags: u8, bounds: &[&[u8]]) -> Field {
    let mut out = vec![kind, flags];
    for bound in bounds {
        out.extend_from_slice(bound);
    }
    Some(out)
}

/// Range fields are checked like text: bounds out of order are refused with the text format's
/// SQLSTATE, a numeric bound without a valid scale is malformed, and a discrete range that arrives
/// in non-canonical form is stored canonically.
#[tokio::test]
async fn range_fields_are_validated_and_canonicalized() {
    let mut conn = session(None).await;
    ok(
        &mut conn,
        "CREATE TABLE r (id INT, ir INT4RANGE, nr NUMRANGE)",
    )
    .await;
    let int_bounds = |lower: i64, upper: i64| [lower.to_le_bytes(), upper.to_le_bytes()];

    // `[10,1]`: both bounds inclusive (flag bits 3 and 4), lower above upper.
    let text_load = copy_in(
        &mut conn,
        "COPY r (id, ir) FROM STDIN",
        0,
        &[b"1\t[10,1]\n"],
    )
    .await;
    let text_code = text_load.expect_err("text refuses [10,1]");
    let [lo, hi] = int_bounds(10, 1);
    let binary = copy_in(
        &mut conn,
        "COPY r (id, ir) FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![int(1), range_field(0, 0x18, &[&lo, &hi])]])],
    )
    .await;
    assert_eq!(binary, Err(text_code), "binary refuses [10,1] as text does");

    // A numeric bound with a scale past the maximum, or the NaN marker.
    for scale in [39_u8, 255] {
        let mut bound = 15_i128.to_le_bytes().to_vec();
        bound.push(scale);
        let result = copy_in(
            &mut conn,
            "COPY r (id, nr) FROM STDIN (FORMAT binary)",
            1,
            &[&stream(&[vec![
                int(1),
                range_field(1, 0x04 | 0x08, &[&bound]),
            ]])],
        )
        .await;
        assert_eq!(result, Err("22P04".to_owned()), "scale {scale}");
    }
    assert_eq!(
        ok(&mut conn, "SELECT count(*) FROM r").await,
        vec![vec![text("0")]]
    );

    // `(1,5]`: exclusive lower, inclusive upper. An int range is stored as `[2,6)`.
    let [lo, hi] = int_bounds(1, 5);
    let loaded = copy_in(
        &mut conn,
        "COPY r (id, ir) FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![int(1), range_field(0, 0x10, &[&lo, &hi])]])],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 1".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT ir::TEXT, ir = '[2,6)'::INT4RANGE FROM r").await,
        vec![vec![text("[2,6)"), text("true")]]
    );
    let [lo, hi] = int_bounds(2, 6);
    let (_, payload, _) = copy_out(&mut conn, "COPY r (ir) TO STDOUT (FORMAT binary)").await;
    assert_eq!(payload, stream(&[vec![range_field(0, 0x08, &[&lo, &hi])]]));
}

/// A date or timestamp the text format would refuse as out of range is refused in binary too.
#[tokio::test]
async fn out_of_range_dates_and_timestamps_are_refused() {
    let mut conn = session(None).await;
    ok(
        &mut conn,
        "CREATE TABLE d (a DATE, b TIMESTAMP, c TIMESTAMPTZ)",
    )
    .await;
    let ok_date = Some(0_i32.to_be_bytes().to_vec());
    let ok_ts = int(0);
    let cases = [
        vec![
            Some((-719_529_i32).to_be_bytes().to_vec()),
            ok_ts.clone(),
            ok_ts.clone(),
        ],
        vec![ok_date.clone(), int(i64::MIN), ok_ts.clone()],
        vec![ok_date.clone(), ok_ts.clone(), int(i64::MIN)],
    ];
    for fields in cases {
        let result = copy_in(
            &mut conn,
            "COPY d FROM STDIN (FORMAT binary)",
            1,
            &[&stream(&[fields])],
        )
        .await;
        assert_eq!(result, Err("22008".to_owned()));
    }
    let loaded = copy_in(
        &mut conn,
        "COPY d FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![ok_date, ok_ts.clone(), ok_ts]])],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 1".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT a, b FROM d").await,
        vec![vec![text("1970-01-01"), text("1970-01-01 00:00:00")]]
    );
}

/// COPY into a partitioned table applies the NOT NULL of the partition each row lands in, in both
/// formats, even where the parent's column allows NULL.
#[tokio::test]
async fn copy_through_a_parent_applies_the_partition_not_null() {
    let mut conn = session(None).await;
    ok(
        &mut conn,
        "CREATE TABLE pt (k INT NOT NULL, v INT) PARTITION BY LIST (k)",
    )
    .await;
    ok(
        &mut conn,
        "CREATE TABLE pt_1 PARTITION OF pt FOR VALUES IN (1)",
    )
    .await;
    ok(
        &mut conn,
        "CREATE TABLE pt_2 PARTITION OF pt FOR VALUES IN (2)",
    )
    .await;
    ok(&mut conn, "ALTER TABLE pt_1 ALTER COLUMN v SET NOT NULL").await;

    let text_load = copy_in(&mut conn, "COPY pt FROM STDIN", 0, &[b"2\t\\N\n1\t\\N\n"]).await;
    assert_eq!(text_load, Err("23502".to_owned()));
    let binary = copy_in(
        &mut conn,
        "COPY pt FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![int(2), None], vec![int(1), None]])],
    )
    .await;
    assert_eq!(binary, Err("23502".to_owned()));
    assert_eq!(
        ok(&mut conn, "SELECT count(*) FROM pt").await,
        vec![vec![text("0")]]
    );
    let loaded = copy_in(
        &mut conn,
        "COPY pt FROM STDIN (FORMAT binary)",
        1,
        &[&stream(&[vec![int(2), None], vec![int(1), int(4)]])],
    )
    .await;
    assert_eq!(loaded, Ok("COPY 2".to_owned()));
}
