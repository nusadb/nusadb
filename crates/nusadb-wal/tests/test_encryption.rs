//! Encrypted WAL round-trip: records sealed with AES-256-GCM-SIV replay correctly with
//! the right key, ciphertext on disk is not plaintext, and a wrong key or tampered byte stops
//! recovery rather than yielding garbage.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; byte indexing corrupts a known offset"
)]

use std::io::Cursor;
use std::sync::Arc;

use nusadb_core::{Lsn, PageId, TxnId};
use nusadb_wal::crypto::DOMAIN_LOG;
use nusadb_wal::{RecordSeal, Sealer, WalReader, WalRecord, WalWriter, recover_sealed_prefix};

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
        WalRecord::CommitTxn {
            txn: TxnId(1),
            unix_ms: 0,
        },
    ]
}

fn seal(key: &[u8; 32]) -> RecordSeal {
    RecordSeal {
        sealer: Arc::new(Sealer::new(key).unwrap()),
        domain: DOMAIN_LOG,
        scope: 0,
    }
}

fn write_encrypted(records: &[WalRecord], key: &[u8; 32]) -> Vec<u8> {
    let mut w = WalWriter::new(Vec::new()).with_seal(seal(key));
    for r in records {
        w.append(r).unwrap();
    }
    w.into_inner()
}

#[test]
fn encrypted_log_round_trips_with_the_right_key() {
    let records = sample();
    let bytes = write_encrypted(&records, &KEY);
    let replayed = WalReader::new(Cursor::new(bytes))
        .with_seal(seal(&KEY))
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
    let replayed = WalReader::new(Cursor::new(bytes))
        .with_seal(seal(&OTHER_KEY))
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
    let replayed = WalReader::new(Cursor::new(bytes))
        .with_seal(seal(&KEY))
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
    let replayed = WalReader::new(Cursor::new(bytes))
        .with_seal(seal(&KEY))
        .replay()
        .unwrap();
    assert_eq!(replayed.len(), 2);
    assert_eq!(replayed[0].1, WalRecord::BeginTxn { txn: TxnId(1) });
}

#[test]
fn recovery_refuses_a_wrong_key_instead_of_truncating() {
    let bytes = write_encrypted(&sample(), &KEY);
    let hole = recover_sealed_prefix(&bytes, Some(&seal(&OTHER_KEY))).unwrap_err();
    assert!(hole.forged);
    assert_eq!(hole.at, 0);
}

#[test]
fn recovery_refuses_a_forged_last_record() {
    // A record rewritten with a valid checksum but without the key is not a torn tail: even as
    // the last record of the log it is refused rather than cut off.
    let mut bytes = write_encrypted(&sample(), &KEY);
    let full = recover_sealed_prefix(&bytes, Some(&seal(&KEY))).unwrap();
    assert_eq!(full.records.len(), 3);
    // Find the last frame's start by walking the frames.
    let mut off = 0usize;
    let mut last = 0usize;
    while off < bytes.len() {
        last = off;
        let len = u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()) as usize;
        off += 16 + len;
    }
    let len = u32::from_le_bytes(bytes[last + 8..last + 12].try_into().unwrap()) as usize;
    bytes[last + 16 + 14] ^= 0x40;
    let mut h = crc32fast::Hasher::new();
    h.update(&bytes[last..last + 12]);
    h.update(&bytes[last + 16..last + 16 + len]);
    let crc = h.finalize().to_le_bytes();
    bytes[last + 12..last + 16].copy_from_slice(&crc);
    let hole = recover_sealed_prefix(&bytes, Some(&seal(&KEY))).unwrap_err();
    assert!(hole.forged);
    assert_eq!(hole.at, last as u64);
}

#[test]
fn a_record_moved_to_another_position_is_refused() {
    // The position is bound into each record's tag: rewriting the frame header to another LSN
    // (with its checksum fixed) must not verify.
    let mut bytes = write_encrypted(&sample(), &KEY);
    bytes[0] = 9;
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut h = crc32fast::Hasher::new();
    h.update(&bytes[0..12]);
    h.update(&bytes[16..16 + len]);
    let crc = h.finalize().to_le_bytes();
    bytes[12..16].copy_from_slice(&crc);
    let hole = recover_sealed_prefix(&bytes, Some(&seal(&KEY))).unwrap_err();
    assert!(hole.forged);
}
