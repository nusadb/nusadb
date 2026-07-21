//! Tests for `frame` (`src/frame.rs`) — the `[type][len][payload]` frame codec.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; slicing simulates partial reads"
)]

use bytes::{BufMut, Bytes, BytesMut};
use nusadb_wire::{Frame, MAX_FRAME_LEN, WireError};

#[test]
fn encode_decode_roundtrip() {
    let frame = Frame::new(b'Q', Bytes::from_static(b"hello"));
    let mut buf = BytesMut::new();
    frame.encode(&mut buf);
    let decoded = Frame::decode(&mut buf).unwrap().unwrap();
    assert_eq!(decoded, frame);
    assert!(buf.is_empty()); // fully consumed
}

#[test]
fn incomplete_buffer_yields_none() {
    let frame = Frame::new(b'Q', Bytes::from_static(b"payload"));
    let mut full = BytesMut::new();
    frame.encode(&mut full);
    // Only the first few bytes have arrived.
    let mut partial = BytesMut::from(&full[..3]);
    assert_eq!(Frame::decode(&mut partial).unwrap(), None);
}

#[test]
fn two_frames_decode_in_sequence() {
    let mut buf = BytesMut::new();
    Frame::new(b'A', Bytes::from_static(b"1")).encode(&mut buf);
    Frame::new(b'B', Bytes::from_static(b"22")).encode(&mut buf);
    let first = Frame::decode(&mut buf).unwrap().unwrap();
    let second = Frame::decode(&mut buf).unwrap().unwrap();
    assert_eq!(first.message_type, b'A');
    assert_eq!(&first.payload[..], b"1");
    assert_eq!(second.message_type, b'B');
    assert_eq!(&second.payload[..], b"22");
    assert_eq!(Frame::decode(&mut buf).unwrap(), None);
}

#[test]
fn oversized_frame_is_rejected() {
    let mut buf = BytesMut::new();
    buf.put_u8(b'Q');
    buf.put_u32(MAX_FRAME_LEN + 1);
    assert!(matches!(
        Frame::decode(&mut buf),
        Err(WireError::FrameTooLarge(_))
    ));
}

#[test]
fn undersized_length_is_rejected() {
    let mut buf = BytesMut::new();
    buf.put_u8(b'Q');
    buf.put_u32(4); // below the 5-byte header
    assert!(matches!(
        Frame::decode(&mut buf),
        Err(WireError::MalformedFrame(4))
    ));
}

#[test]
fn empty_payload_roundtrips() {
    let frame = Frame::new(b'X', Bytes::new());
    let mut buf = BytesMut::new();
    frame.encode(&mut buf);
    assert_eq!(Frame::decode(&mut buf).unwrap().unwrap(), frame);
}
