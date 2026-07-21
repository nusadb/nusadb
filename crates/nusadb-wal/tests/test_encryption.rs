//! Encrypted WAL round-trip: records sealed with AES-256-GCM-SIV replay correctly with
//! the right key, ciphertext on disk is not plaintext, and a wrong key or tampered byte stops
//! recovery rather than yielding garbage.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; byte indexing corrupts a known offset"
)]

use std::io::Cursor;

use nusadb_core::{Lsn, PageId, TxnId};
use nusadb_wal::{WalReader, WalRecord, WalWriter};

const KEY: [u8; 32] = [0x5a; 32];
const OTHER_KEY: [u8; 32] = [0xa5; 32];

fn sample() -> Vec<WalRecord> {
    vec![
        WalRecord::BeginTxn { txn: TxnId(1) },
        WalRecord::PageUpdate {
            txn: TxnId(1),
            page: PageId(3),
            offset: 16,
            image: b"sensitive-redo-image".to_vec(),
        },
        WalRecord::CommitTxn { txn: TxnId(1) },
    ]
}

fn write_encrypted(records: &[WalRecord], key: &[u8; 32]) -> Vec<u8> {
    let mut w = WalWriter::new_encrypted(Vec::new(), key);
    for r in records {
        w.append(r).unwrap();
    }
    w.into_inner()
}

#[test]
fn encrypted_log_round_trips_with_the_right_key() {
    let records = sample();
    let bytes = write_encrypted(&records, &KEY);
    let replayed = WalReader::new_encrypted(Cursor::new(bytes), &KEY)
        .replay()
        .unwrap();
    assert_eq!(replayed.len(), 3);
    assert_eq!(replayed[0].0, Lsn(1));
    assert_eq!(replayed[2].0, Lsn(3));
    let got: Vec<WalRecord> = replayed.into_iter().map(|(_, r)| r).collect();
    assert_eq!(got, records);
}

#[test]
fn ciphertext_does_not_contain_the_plaintext() {
    let bytes = write_encrypted(&sample(), &KEY);
    // The redo image must not appear verbatim on disk.
    assert!(
        !bytes
            .windows(b"sensitive-redo-image".len())
            .any(|w| w == b"sensitive-redo-image"),
        "plaintext leaked into the encrypted WAL"
    );
}

#[test]
fn a_wrong_key_recovers_nothing() {
    let bytes = write_encrypted(&sample(), &KEY);
    // Opening with the wrong key fails the AEAD tag on the first record → end of durable log.
    let replayed = WalReader::new_encrypted(Cursor::new(bytes), &OTHER_KEY)
        .replay()
        .unwrap();
    assert!(replayed.is_empty());
}

#[test]
fn reading_an_encrypted_log_without_the_key_recovers_nothing() {
    let bytes = write_encrypted(&sample(), &KEY);
    // A plaintext reader treats the ciphertext body as a (failed) compressed payload → stops.
    let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
    assert!(replayed.is_empty());
}

#[test]
fn a_tampered_record_stops_recovery_at_that_point() {
    let records = sample();
    let mut bytes = write_encrypted(&records, &KEY);
    // Flip a byte inside the first record's body (past the 16-byte frame header). This breaks the
    // CRC and the AEAD tag; either way recovery must stop before the tampered record.
    bytes[20] ^= 0x01;
    let replayed = WalReader::new_encrypted(Cursor::new(bytes), &KEY)
        .replay()
        .unwrap();
    assert!(replayed.is_empty());
}

#[test]
fn a_truncated_trailing_record_yields_the_valid_prefix() {
    let records = sample();
    let mut bytes = write_encrypted(&records, &KEY);
    // Drop the last 4 bytes — the trailing record is now torn and must not replay, while the
    // earlier records still do.
    bytes.truncate(bytes.len() - 4);
    let replayed = WalReader::new_encrypted(Cursor::new(bytes), &KEY)
        .replay()
        .unwrap();
    assert_eq!(replayed.len(), 2);
    assert_eq!(replayed[0].1, WalRecord::BeginTxn { txn: TxnId(1) });
}
