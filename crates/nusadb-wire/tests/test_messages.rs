//! Tests for `messages` (`src/messages.rs`) — the Nusa Wire Protocol message taxonomy.

#![allow(
    clippy::unwrap_used,
    reason = "integration test harness asserts via unwrap/panic"
)]

use bytes::{BufMut, Bytes, BytesMut};
use nusadb_wire::{BackendMessage, DescribeTarget, Frame, FrontendMessage, TxnStatus, WireError};

// Message type bytes are private to the crate; tests reference the same literals.
const T_STARTUP: u8 = b'S';
const T_QUERY: u8 = b'Q';
const T_BIND: u8 = b'B';

fn frontend_roundtrip(msg: &FrontendMessage) {
    let decoded = FrontendMessage::decode(&msg.encode().unwrap()).unwrap();
    assert_eq!(&decoded, msg);
}

fn backend_roundtrip(msg: &BackendMessage) {
    let decoded = BackendMessage::decode(&msg.encode().unwrap()).unwrap();
    assert_eq!(&decoded, msg);
}

#[test]
fn frontend_messages_roundtrip() {
    frontend_roundtrip(&FrontendMessage::Startup {
        major: 1,
        minor: 0,
        user: "rafi".to_owned(),
        database: "nusadb".to_owned(),
    });
    frontend_roundtrip(&FrontendMessage::Query {
        sql: "SELECT 1".to_owned(),
    });
    frontend_roundtrip(&FrontendMessage::Terminate);
}

#[test]
fn extended_query_frontend_messages_roundtrip() {
    frontend_roundtrip(&FrontendMessage::Parse {
        name: "s1".to_owned(),
        sql: "SELECT id FROM t".to_owned(),
        param_types: vec![],
    });
    frontend_roundtrip(&FrontendMessage::Bind {
        portal: "p1".to_owned(),
        statement: "s1".to_owned(),
        params: vec![Some(b"42".to_vec()), None],
        result_formats: vec![1, 0],
    });
    frontend_roundtrip(&FrontendMessage::Describe {
        target: DescribeTarget::Statement,
        name: "s1".to_owned(),
    });
    frontend_roundtrip(&FrontendMessage::Describe {
        target: DescribeTarget::Portal,
        name: "p1".to_owned(),
    });
    frontend_roundtrip(&FrontendMessage::Execute {
        portal: "p1".to_owned(),
        max_rows: 100,
    });
    frontend_roundtrip(&FrontendMessage::Sync);
    frontend_roundtrip(&FrontendMessage::Close {
        target: DescribeTarget::Portal,
        name: "p1".to_owned(),
    });
}

#[test]
fn extended_query_backend_messages_roundtrip() {
    backend_roundtrip(&BackendMessage::ParseComplete);
    backend_roundtrip(&BackendMessage::BindComplete);
    backend_roundtrip(&BackendMessage::CloseComplete);
    backend_roundtrip(&BackendMessage::ParameterDescription { count: 2 });
    backend_roundtrip(&BackendMessage::NoData);
    backend_roundtrip(&BackendMessage::PortalSuspended);
}

#[test]
fn sasl_messages_roundtrip() {
    frontend_roundtrip(&FrontendMessage::SaslInitialResponse {
        mechanism: "SCRAM-SHA-256".to_owned(),
        data: b"n,,n=alice,r=abc".to_vec(),
    });
    frontend_roundtrip(&FrontendMessage::SaslResponse {
        data: b"c=biws,r=abcXYZ,p=proof".to_vec(),
    });
    backend_roundtrip(&BackendMessage::AuthSasl {
        mechanisms: vec!["SCRAM-SHA-256".to_owned()],
    });
    backend_roundtrip(&BackendMessage::AuthSaslContinue {
        data: b"r=abcXYZ,s=c2FsdA==,i=4096".to_vec(),
    });
    backend_roundtrip(&BackendMessage::AuthSaslFinal {
        data: b"v=dmVyaWZpZXI=".to_vec(),
    });
}

#[test]
fn cancel_messages_roundtrip() {
    frontend_roundtrip(&FrontendMessage::CancelRequest {
        pid: 7,
        secret: 0xDEAD_BEEF,
    });
    backend_roundtrip(&BackendMessage::BackendKeyData {
        pid: 7,
        secret: 0xDEAD_BEEF,
    });
}

#[test]
fn copy_messages_roundtrip() {
    frontend_roundtrip(&FrontendMessage::CopyData {
        data: b"1\talice\n".to_vec(),
    });
    frontend_roundtrip(&FrontendMessage::CopyDone);
    frontend_roundtrip(&FrontendMessage::CopyFail {
        message: "client aborted".to_owned(),
    });
    backend_roundtrip(&BackendMessage::CopyInResponse { columns: 3 });
    backend_roundtrip(&BackendMessage::CopyOutResponse { columns: 2 });
    backend_roundtrip(&BackendMessage::CopyData {
        data: b"1\talice\n".to_vec(),
    });
    backend_roundtrip(&BackendMessage::CopyDone);
}

#[test]
fn backend_messages_roundtrip() {
    backend_roundtrip(&BackendMessage::AuthOk);
    backend_roundtrip(&BackendMessage::ReadyForQuery(TxnStatus::Idle));
    backend_roundtrip(&BackendMessage::ReadyForQuery(TxnStatus::Failed));
    backend_roundtrip(&BackendMessage::CommandComplete {
        tag: "SELECT 2".to_owned(),
    });
    backend_roundtrip(&BackendMessage::Error {
        code: "42P01".to_owned(),
        message: "relation does not exist".to_owned(),
    });
    backend_roundtrip(&BackendMessage::RowDescription {
        columns: vec!["id".to_owned(), "name".to_owned()],
    });
    // Typed row description (protocol 1.1): name + 1-byte type tag per column.
    backend_roundtrip(&BackendMessage::RowDescriptionTyped {
        columns: vec![("id".to_owned(), 0x02), ("name".to_owned(), 0x05)],
    });
    backend_roundtrip(&BackendMessage::DataRow {
        values: vec![Some(b"42".to_vec()), None, Some(b"hello".to_vec())],
    });
    // Asynchronous LISTEN/NOTIFY delivery: pid + channel + payload, including the empty-payload form.
    backend_roundtrip(&BackendMessage::NotificationResponse {
        pid: 4242,
        channel: "orders".to_owned(),
        payload: "row-42".to_owned(),
    });
    backend_roundtrip(&BackendMessage::NotificationResponse {
        pid: 1,
        channel: "cache_invalidate".to_owned(),
        payload: String::new(),
    });
}

#[test]
fn startup_requires_protocol_magic() {
    // A Query frame is not a valid Startup payload — its first u32 is not the magic.
    let bogus = FrontendMessage::Query {
        sql: "x".to_owned(),
    }
    .encode()
    .unwrap();
    let reframed = Frame::new(T_STARTUP, bogus.payload);
    assert!(matches!(
        FrontendMessage::decode(&reframed),
        Err(WireError::BadMagic(_))
    ));
}

#[test]
fn unknown_type_is_rejected() {
    let frame = Frame::new(b'?', Bytes::new());
    assert!(matches!(
        FrontendMessage::decode(&frame),
        Err(WireError::UnknownMessageType(b'?'))
    ));
}

#[test]
fn truncated_string_is_rejected() {
    // Claim a 100-byte string but provide none.
    let mut p = BytesMut::new();
    p.put_u32(100);
    let frame = Frame::new(T_QUERY, p.freeze());
    assert!(matches!(
        FrontendMessage::decode(&frame),
        Err(WireError::TruncatedPayload(T_QUERY))
    ));
}

#[test]
fn bind_with_lying_counts_errors_without_speculative_overalloc() {
    // A Bind that claims a huge u16 field/format count but provides no bytes must error
    // cleanly. The decoder bounds `Vec::with_capacity` by the remaining byte length, so it cannot
    // be tricked into reserving a large buffer from an unvalidated count before hitting the
    // truncated payload. (We can't observe capacity through the public API; the guarantee is that
    // decode returns a clean error rather than panicking or over-allocating.)

    // (a) the params field-list count lies: empty portal + statement, then 0xFFFF fields, no bytes.
    let mut p = BytesMut::new();
    p.put_u32(0); // portal = ""
    p.put_u32(0); // statement = ""
    p.put_u16(u16::MAX); // params: claims 65535 fields …
    let frame = Frame::new(T_BIND, p.freeze()); // … but provides none
    assert!(matches!(
        FrontendMessage::decode(&frame),
        Err(WireError::TruncatedPayload(T_BIND))
    ));

    // (b) the result-format count lies: valid (empty) params, then 0xFFFF formats, no bytes.
    let mut p = BytesMut::new();
    p.put_u32(0); // portal = ""
    p.put_u32(0); // statement = ""
    p.put_u16(0); // params: 0 fields (valid)
    p.put_u16(u16::MAX); // result formats: claims 65535 …
    let frame = Frame::new(T_BIND, p.freeze()); // … but provides none
    assert!(matches!(
        FrontendMessage::decode(&frame),
        Err(WireError::TruncatedPayload(T_BIND))
    ));
}

#[test]
fn row_description_with_too_many_columns_is_rejected_not_truncated() {
    // N1 / G21: a column count past u16::MAX must error, not truncate the count prefix (which
    // would make the decoder read too few columns and desync the stream).
    let columns: Vec<String> = (0..=u16::MAX as usize).map(|i| format!("c{i}")).collect();
    let msg = BackendMessage::RowDescription { columns };
    assert!(matches!(msg.encode(), Err(WireError::FieldTooLarge)));
}

#[test]
fn data_row_with_too_many_fields_is_rejected() {
    // N1 / G21: a field count past u16::MAX must error rather than truncate its count prefix.
    let values: Vec<Option<Vec<u8>>> = vec![Some(Vec::new()); u16::MAX as usize + 1];
    let msg = BackendMessage::DataRow { values };
    assert!(matches!(msg.encode(), Err(WireError::FieldTooLarge)));
}
