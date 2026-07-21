//! WAL record kinds and their wire encoding.
//!
//! A record is the unit of recovery replay. The on-disk *frame* (length + CRC32 +
//! lz4-compressed payload) is handled by [`writer`](crate::writer) /
//! [`reader`](crate::reader); this module owns only the payload encoding of each kind.
//!
//! The log is **physical** (ARIES-style): it records page-level redo images plus
//! transaction boundaries, so recovery can reconstruct the page store after a crash.

use nusadb_core::{Lsn, PageId, TxnId};

/// A single WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// Marks the start of transaction `txn`.
    BeginTxn {
        /// Transaction being started.
        txn: TxnId,
    },
    /// Marks the durable commit of transaction `txn`.
    CommitTxn {
        /// Transaction being committed.
        txn: TxnId,
    },
    /// Marks the abort of transaction `txn`; its effects must be undone on recovery.
    AbortTxn {
        /// Transaction being aborted.
        txn: TxnId,
    },
    /// Redo image of a modified byte range within a page.
    PageUpdate {
        /// Owning transaction.
        txn: TxnId,
        /// Page that was modified.
        page: PageId,
        /// Byte offset of the change within the page.
        offset: u16,
        /// The new bytes written at `offset` (redo image).
        image: Vec<u8>,
    },
    /// A full 8 KiB page image — written the first time a page is dirtied after a
    /// checkpoint, to guard against torn-page writes during recovery.
    FullPageWrite {
        /// Owning transaction.
        txn: TxnId,
        /// Page whose full image follows.
        page: PageId,
        /// The complete page contents.
        image: Vec<u8>,
    },
    /// Checkpoint marker: all changes up to `lsn` are flushed to the page store.
    Checkpoint {
        /// Highest LSN guaranteed durable in the page store at checkpoint time.
        lsn: Lsn,
    },
    /// Logical key/value write: `key` now maps to `value` — the generic envelope an engine
    /// packs its logical operations into (the btree engine encodes each of its logged ops as
    /// a tagged `Put`), replayed on recovery.
    Put {
        /// Key written.
        key: Vec<u8>,
        /// Value written.
        value: Vec<u8>,
    },
    /// Logical key/value delete: `key` is tombstoned. Kept decodable for older logs; the
    /// btree engine expresses deletes inside its tagged `Put` envelope instead.
    Delete {
        /// Key deleted.
        key: Vec<u8>,
    },
}

const TAG_BEGIN: u8 = 0;
const TAG_COMMIT: u8 = 1;
const TAG_ABORT: u8 = 2;
const TAG_PAGE_UPDATE: u8 = 3;
const TAG_FULL_PAGE: u8 = 4;
const TAG_CHECKPOINT: u8 = 5;
const TAG_PUT: u8 = 6;
const TAG_DELETE: u8 = 7;

/// Append a `[u32 len][bytes]` field, erroring if `bytes` is longer than `u32::MAX`.
fn push_len_prefixed(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<(), nusadb_core::Error> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        nusadb_core::Error::Io(std::io::Error::other("WAL record field exceeds 4 GiB"))
    })?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

impl WalRecord {
    /// Serialize this record's payload (uncompressed, unframed).
    ///
    /// # Errors
    /// A field (key / value / page image) whose length does not fit a `u32` would truncate its
    /// length prefix and corrupt the record, so it is rejected rather than silently
    /// written. In practice no field reaches 4 GiB, so this never fires on a healthy engine.
    pub fn encode(&self) -> Result<Vec<u8>, nusadb_core::Error> {
        let mut buf = Vec::new();
        self.encode_into(&mut buf)?;
        Ok(buf)
    }

    /// Serialize this record's payload into `buf`, which is **cleared first**. Lets a caller reuse
    /// one scratch buffer across many records instead of allocating per call.
    ///
    /// # Errors
    /// Same as [`encode`](Self::encode): a field longer than `u32::MAX` is rejected.
    pub fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), nusadb_core::Error> {
        buf.clear();
        match self {
            Self::BeginTxn { txn } => {
                buf.push(TAG_BEGIN);
                buf.extend_from_slice(&txn.0.to_le_bytes());
            },
            Self::CommitTxn { txn } => {
                buf.push(TAG_COMMIT);
                buf.extend_from_slice(&txn.0.to_le_bytes());
            },
            Self::AbortTxn { txn } => {
                buf.push(TAG_ABORT);
                buf.extend_from_slice(&txn.0.to_le_bytes());
            },
            Self::PageUpdate {
                txn,
                page,
                offset,
                image,
            } => {
                buf.push(TAG_PAGE_UPDATE);
                buf.extend_from_slice(&txn.0.to_le_bytes());
                buf.extend_from_slice(&page.0.to_le_bytes());
                buf.extend_from_slice(&offset.to_le_bytes());
                push_len_prefixed(buf, image)?;
            },
            Self::FullPageWrite { txn, page, image } => {
                buf.push(TAG_FULL_PAGE);
                buf.extend_from_slice(&txn.0.to_le_bytes());
                buf.extend_from_slice(&page.0.to_le_bytes());
                push_len_prefixed(buf, image)?;
            },
            Self::Checkpoint { lsn } => {
                buf.push(TAG_CHECKPOINT);
                buf.extend_from_slice(&lsn.0.to_le_bytes());
            },
            Self::Put { key, value } => {
                buf.push(TAG_PUT);
                push_len_prefixed(buf, key)?;
                push_len_prefixed(buf, value)?;
            },
            Self::Delete { key } => {
                buf.push(TAG_DELETE);
                push_len_prefixed(buf, key)?;
            },
        }
        Ok(())
    }

    /// Parse a record payload. Returns `None` if the bytes are malformed (the reader
    /// treats this as the end of the durable log).
    #[must_use]
    pub fn decode(payload: &[u8]) -> Option<Self> {
        let mut cur = Cursor {
            bytes: payload,
            pos: 0,
        };
        let rec = match cur.u8()? {
            TAG_BEGIN => Self::BeginTxn {
                txn: TxnId(cur.u64()?),
            },
            TAG_COMMIT => Self::CommitTxn {
                txn: TxnId(cur.u64()?),
            },
            TAG_ABORT => Self::AbortTxn {
                txn: TxnId(cur.u64()?),
            },
            TAG_PAGE_UPDATE => {
                let txn = TxnId(cur.u64()?);
                let page = PageId(cur.u64()?);
                let offset = cur.u16()?;
                let len = cur.u32()? as usize;
                let image = cur.take(len)?;
                Self::PageUpdate {
                    txn,
                    page,
                    offset,
                    image,
                }
            },
            TAG_FULL_PAGE => {
                let txn = TxnId(cur.u64()?);
                let page = PageId(cur.u64()?);
                let len = cur.u32()? as usize;
                let image = cur.take(len)?;
                Self::FullPageWrite { txn, page, image }
            },
            TAG_CHECKPOINT => Self::Checkpoint {
                lsn: Lsn(cur.u64()?),
            },
            TAG_PUT => {
                let klen = cur.u32()? as usize;
                let key = cur.take(klen)?;
                let vlen = cur.u32()? as usize;
                let value = cur.take(vlen)?;
                Self::Put { key, value }
            },
            TAG_DELETE => {
                let klen = cur.u32()? as usize;
                let key = cur.take(klen)?;
                Self::Delete { key }
            },
            _ => return None,
        };
        Some(rec)
    }
}

/// Bounds-checked forward reader over a byte payload (never panics).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.bytes.get(self.pos..self.pos + 2)?;
        self.pos += 2;
        Some(bytemuck::pod_read_unaligned::<u16>(s))
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.bytes.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(bytemuck::pod_read_unaligned::<u32>(s))
    }
    fn u64(&mut self) -> Option<u64> {
        let s = self.bytes.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        Some(bytemuck::pod_read_unaligned::<u64>(s))
    }
    fn take(&mut self, n: usize) -> Option<Vec<u8>> {
        let s = self.bytes.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s.to_vec())
    }
}
