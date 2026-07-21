//! Tests for `reader` (`src/reader.rs`) — sequential WAL replay, stopping at a torn/CRC-bad tail.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; byte indexing corrupts a known offset"
)]

use std::io::Cursor;

use nusadb_core::{Lsn, PageId, TxnId};
use nusadb_wal::{WalReader, WalRecord, WalWriter};

fn sample() -> Vec<WalRecord> {
    vec![
        WalRecord::BeginTxn { txn: TxnId(1) },
        WalRecord::PageUpdate {
            txn: TxnId(1),
            page: PageId(3),
            offset: 16,
            image: b"abc".to_vec(),
        },
        WalRecord::CommitTxn { txn: TxnId(1) },
    ]
}

fn write_log(records: &[WalRecord]) -> Vec<u8> {
    let mut w = WalWriter::new(Vec::new());
    for r in records {
        w.append(r).unwrap();
    }
    w.into_inner()
}

#[test]
fn roundtrip_preserves_order_and_lsns() {
    let records = sample();
    let bytes = write_log(&records);
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert_eq!(replayed.len(), 3);
    assert_eq!(replayed[0].0, Lsn(1));
    assert_eq!(replayed[2].0, Lsn(3));
    let got: Vec<WalRecord> = replayed.into_iter().map(|(_, r)| r).collect();
    assert_eq!(got, records);
}

#[test]
fn truncated_trailing_record_is_treated_as_end() {
    let mut bytes = write_log(&sample());
    bytes.truncate(bytes.len() - 3); // chop the last record's tail
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert_eq!(replayed.len(), 2); // first two survive, torn third dropped
}

#[test]
fn crc_mismatch_stops_recovery() {
    let mut bytes = write_log(&sample());
    // Corrupt a byte inside the first record's compressed payload (after the 16-byte header)
    // so its CRC fails — recovery should stop before any record.
    bytes[16] ^= 0xFF;
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert_eq!(replayed.len(), 0);
}

#[test]
fn empty_log_replays_nothing() {
    let replayed = WalReader::new(Cursor::new(Vec::new())).replay().unwrap();
    assert!(replayed.is_empty());
}

// The body begins with a kind byte at frame offset 16 (unencrypted) — 0 = stored, 1 =
// lz4-compressed. Tiny records are stored (lz4 would only add overhead); large ones are compressed.

#[test]
fn tiny_record_is_stored_uncompressed() {
    // BeginTxn encodes to 9 bytes (< the 64-byte threshold) → stored, kind byte 0.
    let bytes = write_log(&[WalRecord::BeginTxn { txn: TxnId(7) }]);
    assert_eq!(bytes[16], 0, "tiny record must be stored uncompressed");
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert_eq!(replayed[0].1, WalRecord::BeginTxn { txn: TxnId(7) });
}

#[test]
fn large_record_is_compressed_and_round_trips() {
    // A full 8 KiB page of repeated bytes is well above the threshold and highly compressible.
    let rec = WalRecord::FullPageWrite {
        txn: TxnId(2),
        page: PageId(5),
        image: vec![0xAB; 8192],
    };
    let bytes = write_log(std::slice::from_ref(&rec));
    assert_eq!(bytes[16], 1, "large record must be lz4-compressed");
    assert!(
        bytes.len() < 8192,
        "compressible page must frame smaller than its raw image ({} bytes)",
        bytes.len()
    );
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].1, rec);
}

#[test]
fn mixed_stored_and_compressed_log_round_trips() {
    let records = vec![
        WalRecord::BeginTxn { txn: TxnId(1) }, // stored
        WalRecord::FullPageWrite {
            txn: TxnId(1),
            page: PageId(9),
            image: vec![0xCD; 4096],
        }, // compressed
        WalRecord::CommitTxn { txn: TxnId(1) }, // stored
    ];
    let bytes = write_log(&records);
    let replayed: Vec<WalRecord> = WalReader::new(Cursor::new(bytes))
        .replay()
        .unwrap()
        .into_iter()
        .map(|(_, r)| r)
        .collect();
    assert_eq!(replayed, records);
}
