//! Tests for `record` (`src/record.rs`) — WAL record encode/decode.

#![allow(
    clippy::expect_used,
    reason = "integration test harness asserts via expect on known-good inputs"
)]

use nusadb_core::{Lsn, PageId, TxnId};
use nusadb_wal::WalRecord;

fn roundtrip(rec: &WalRecord) {
    let bytes = rec.encode().expect("encode");
    assert_eq!(WalRecord::decode(&bytes).as_ref(), Some(rec));
}

#[test]
fn encode_decode_all_kinds() {
    roundtrip(&WalRecord::BeginTxn { txn: TxnId(7) });
    roundtrip(&WalRecord::CommitTxn { txn: TxnId(7) });
    roundtrip(&WalRecord::AbortTxn { txn: TxnId(9) });
    roundtrip(&WalRecord::PageUpdate {
        txn: TxnId(1),
        page: PageId(42),
        offset: 64,
        image: b"hello".to_vec(),
    });
    roundtrip(&WalRecord::FullPageWrite {
        txn: TxnId(2),
        page: PageId(5),
        image: vec![0xAB; 8192],
    });
    roundtrip(&WalRecord::Checkpoint { lsn: Lsn(123) });
    roundtrip(&WalRecord::Put {
        key: b"k".to_vec(),
        value: b"v".to_vec(),
    });
    roundtrip(&WalRecord::Delete {
        key: b"gone".to_vec(),
    });
}

#[test]
fn decode_rejects_garbage() {
    assert!(WalRecord::decode(&[]).is_none()); // empty
    assert!(WalRecord::decode(&[99]).is_none()); // unknown tag
    assert!(WalRecord::decode(&[0, 1, 2]).is_none()); // tag 0 (BeginTxn), truncated u64
}
