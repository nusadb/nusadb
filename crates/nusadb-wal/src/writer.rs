//! Append-only WAL writer.
//!
//! Frames each record as `[lsn:u64][len:u32][crc32:u32][lz4(payload)]` (all little-endian)
//! and appends it to the underlying sink. Generic over [`std::io::Write`] so production
//! writes to a file while tests write to an in-memory buffer.
//!
//! **Durability** is the caller's responsibility: after [`append`](WalWriter::append) the
//! bytes are in the writer's buffer; call [`flush`](WalWriter::flush) and then `fsync` the
//! underlying file before treating a record as durable (write-ahead protocol).

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use nusadb_core::{Lsn, Result};

use crate::crypto::WalCipher;
use crate::record::WalRecord;

/// Frame body kind (the first byte of every record body, inside the CRC/AEAD coverage): the
/// payload is stored **uncompressed**. Used for records too small for lz4 to help.
pub(crate) const BODY_STORED: u8 = 0;
/// Frame body kind: the payload is lz4-compressed (size-prepended, as `decompress_size_prepended`
/// expects).
pub(crate) const BODY_COMPRESSED: u8 = 1;
/// Encoded payloads at or below this many bytes are stored uncompressed: lz4 burns CPU and adds
/// header overhead on tiny records like `BeginTxn`/`CommitTxn`/`AbortTxn` (~9 B).
pub(crate) const COMPRESS_MIN_BYTES: usize = 64;

/// How many framed bytes an [`WalWriter`] coalesces in its append buffer before writing them to the
/// sink. A commit's handful of records (each ≤ a page) stays well under this, so the commit's whole
/// record set reaches the sink in **one** `write` instead of one per record — a single write per
/// commit, the mature-durable-store norm. A large single transaction (e.g. a bulk `COPY`) spills
/// every time the buffer fills, so its userspace footprint stays bounded rather than accumulating
/// every record.
const APPEND_BUF_CAP: usize = 64 * 1024;

/// Append-only writer that frames and persists [`WalRecord`]s.
#[derive(Debug)]
pub struct WalWriter<W: Write> {
    inner: W,
    next_lsn: u64,
    /// When set, each record's payload is sealed with AES-256-GCM-SIV before framing.
    cipher: Option<WalCipher>,
    /// Reused across appends to hold the encoded (uncompressed) record payload — avoids a fresh
    /// allocation per record on the hot durability path.
    scratch: Vec<u8>,
    /// Reused across appends to hold the pre-encryption body (`[kind:u8][stored|compressed]`).
    body: Vec<u8>,
    /// Reused across appends to hold the framed bytes handed to the sink.
    frame: Vec<u8>,
    /// Framed records accumulated but not yet written to the sink. Drained to one `write` by
    /// [`flush`](WalWriter::flush) (the per-commit durability point) or when it reaches
    /// [`APPEND_BUF_CAP`], so a commit's records coalesce into a single syscall while a large
    /// transaction's footprint stays bounded. Records are not durable until `flush` + an `fsync`.
    append_buf: Vec<u8>,
}

impl<W: Write> WalWriter<W> {
    /// Wrap `inner`, assigning LSNs starting at 1.
    #[must_use]
    pub const fn new(inner: W) -> Self {
        Self {
            inner,
            next_lsn: 1,
            cipher: None,
            scratch: Vec::new(),
            body: Vec::new(),
            frame: Vec::new(),
            append_buf: Vec::new(),
        }
    }

    /// Wrap `inner` for **encrypted** records, assigning LSNs starting at 1. Each
    /// record's lz4-compressed payload is sealed under `key` before framing; recovery needs the
    /// same key (see [`WalReader::new_encrypted`](crate::WalReader::new_encrypted)).
    #[must_use]
    pub fn new_encrypted(inner: W, key: &[u8; 32]) -> Self {
        Self {
            inner,
            next_lsn: 1,
            cipher: Some(WalCipher::new(key)),
            scratch: Vec::new(),
            body: Vec::new(),
            frame: Vec::new(),
            append_buf: Vec::new(),
        }
    }

    /// Wrap `inner`, resuming LSN assignment at `start` (e.g. after recovery).
    #[must_use]
    pub fn resume(inner: W, start: Lsn) -> Self {
        Self {
            inner,
            next_lsn: start.0.max(1),
            cipher: None,
            scratch: Vec::new(),
            body: Vec::new(),
            frame: Vec::new(),
            append_buf: Vec::new(),
        }
    }

    /// The LSN that the next [`append`](WalWriter::append) will assign.
    #[must_use]
    pub const fn next_lsn(&self) -> Lsn {
        Lsn(self.next_lsn)
    }

    /// Frame and append `record`, returning the LSN assigned to it.
    ///
    /// # Errors
    /// Propagates write errors from the underlying sink.
    pub fn append(&mut self, record: &WalRecord) -> Result<Lsn> {
        let lsn = Lsn(self.next_lsn);
        // Encode into the reused scratch buffer rather than a fresh Vec per record.
        record.encode_into(&mut self.scratch)?;

        // Build the pre-encryption body `[kind:u8][data]` in the reused buffer. Tiny payloads are
        // stored uncompressed — lz4 wastes CPU and adds header overhead on records like
        // Begin/Commit/Abort (~9 B), and skips the compression allocation entirely.
        self.body.clear();
        if self.scratch.len() <= COMPRESS_MIN_BYTES {
            self.body.push(BODY_STORED);
            self.body.extend_from_slice(&self.scratch);
        } else {
            self.body.push(BODY_COMPRESSED);
            self.body
                .extend_from_slice(&lz4_flex::compress_prepend_size(&self.scratch));
        }

        // Compress-then-encrypt: seal the body, then CRC + frame the ciphertext so a bit-flip is
        // still caught by the CRC and a wrong key / tamper by the AEAD tag. The body kind
        // byte is inside the sealed/CRC'd region, so it cannot be tampered undetected.
        let sealed = match &self.cipher {
            Some(cipher) => Some(cipher.seal(lsn.0, &self.body).map_err(|()| {
                nusadb_core::Error::Io(std::io::Error::other("WAL record encryption failed"))
            })?),
            None => None,
        };
        let body = sealed.as_deref().unwrap_or(&self.body);
        // The frame's length prefix is a `u32`; a body that does not fit would silently truncate it
        // and corrupt the log, so reject it rather than write a malformed frame.
        let body_len = u32::try_from(body.len()).map_err(|_| {
            nusadb_core::Error::Io(std::io::Error::other("WAL record body exceeds 4 GiB"))
        })?;
        // CRC the HEADER (lsn + len) as well as the body, so a corrupted length or lsn — not only a
        // body bit-flip — is caught. This closes two silent-data-loss holes: (1) a zeroed region (a
        // bad sector) whose all-zero header used to read as a valid `len=0` record because
        // `crc32(&[]) == 0` — now the CRC is taken over the 12 zero header bytes too, which is
        // non-zero, so the region fails validation; (2) a corrupted `len` that used to desync the
        // reader from every following frame boundary undetected.
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&lsn.0.to_le_bytes());
        hasher.update(&body_len.to_le_bytes());
        hasher.update(body);
        let crc = hasher.finalize();

        // Build the frame in the reused buffer — `[lsn:u64][len:u32][crc:u32][body]`.
        self.frame.clear();
        self.frame.extend_from_slice(&lsn.0.to_le_bytes());
        self.frame.extend_from_slice(&body_len.to_le_bytes());
        self.frame.extend_from_slice(&crc.to_le_bytes());
        self.frame.extend_from_slice(body);

        // Coalesce this record into the append buffer instead of a `write` per record. The buffer
        // is drained (one `write`) by `flush` at the commit's durability point, or here when it
        // reaches the cap so a large transaction's footprint stays bounded. A mid-transaction spill
        // reaches only the OS page cache — it is not durable until `flush` + `fsync`, so a crash
        // before the commit's fsync correctly loses it.
        self.append_buf.extend_from_slice(&self.frame);
        if self.append_buf.len() >= APPEND_BUF_CAP {
            self.inner.write_all(&self.append_buf)?;
            self.append_buf.clear();
        }
        self.next_lsn += 1;
        Ok(lsn)
    }

    /// Drain the append buffer to the underlying sink in one `write`, then flush the sink (does
    /// **not** fsync). This is the per-commit durability point: the caller flushes, then `fsync`s.
    ///
    /// # Errors
    /// Propagates write / flush errors.
    pub fn flush(&mut self) -> Result<()> {
        if !self.append_buf.is_empty() {
            self.inner.write_all(&self.append_buf)?;
            self.append_buf.clear();
        }
        self.inner.flush().map_err(Into::into)
    }

    /// Borrow the underlying sink — e.g. to `fsync` a `File` after [`flush`](WalWriter::flush).
    pub const fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Consume the writer, draining any buffered records into the sink first, then returning it.
    pub fn into_inner(mut self) -> W {
        if !self.append_buf.is_empty() {
            // The sinks this is called on (in-memory `Vec`, a `File`) do not fail a final write;
            // a best-effort drain keeps `into_inner` returning the full byte stream.
            let _ = self.inner.write_all(&self.append_buf);
        }
        self.inner
    }
}

impl WalWriter<File> {
    /// Truncate the log at `path` to empty, reset LSN assignment to 1, and swap in a fresh
    /// append handle for subsequent records.
    ///
    /// Call this only after a checkpoint has made every prior record redundant (their data
    /// is durable elsewhere, e.g. checkpointed into the page store). `path` must be the same file this
    /// writer appends to.
    ///
    /// Truncation goes through a short-lived write handle rather than `set_len` on the
    /// current handle: an append-mode handle is not granted `FILE_WRITE_DATA` on Windows,
    /// so `set_len` would fail with "Access is denied". The truncation is `fsync`-ed before
    /// the new append handle is installed.
    ///
    /// # Errors
    /// Propagates open / truncate / fsync errors.
    pub fn truncate(&mut self, path: &Path) -> Result<()> {
        let truncator = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        truncator.sync_all()?;
        drop(truncator);
        // Discard any not-yet-written records: the log is being reset, so buffered bytes (which
        // belong to the old file) must never reach the fresh handle.
        self.append_buf.clear();
        self.inner = OpenOptions::new().create(true).append(true).open(path)?;
        self.next_lsn = 1;
        Ok(())
    }

    /// Roll the log back to a known-durable byte length `len`, discarding every record appended
    /// past it, and `fsync` the truncation durable.
    ///
    /// Used to undo records that were appended but whose durability barrier (`fsync`) then failed:
    /// the bytes may sit in the page cache, so a *clean* restart would otherwise replay them. Rolling
    /// the file back to the last byte length that a successful `fsync` covered keeps the on-disk log
    /// equal to what was actually durable — the lynchpin of fsync-failure atomicity: a COMMIT
    /// marker that was never durable cannot resurrect.
    ///
    /// Like [`truncate`](Self::truncate), the resize goes through a short-lived `write` handle rather
    /// than `set_len` on the append handle (Windows denies `FILE_WRITE_DATA` to append handles), and is
    /// `fsync`-ed before a fresh append handle is installed. `next_lsn` is left untouched: the discarded
    /// records' LSNs are never reused on disk (recovery re-derives the resume LSN from the durable log),
    /// and the engine is fail-closed (poisoned) after such a failure, so no further append happens until
    /// it is reopened.
    ///
    /// # Errors
    /// Propagates flush / open / truncate / fsync errors.
    pub fn rollback_to(&mut self, path: &Path, len: u64) -> Result<()> {
        // The records still buffered are exactly the ones being rolled back (highest LSNs, past the
        // last durable length) — discard them rather than write them to the reopened handle.
        self.append_buf.clear();
        self.inner.flush()?;
        let handle = OpenOptions::new().write(true).open(path)?;
        handle.set_len(len)?;
        handle.sync_all()?;
        drop(handle);
        self.inner = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::WalRecord;
    use nusadb_core::TxnId;

    #[test]
    fn appends_coalesce_and_reach_the_sink_only_on_flush() {
        // A commit's records buffer in userspace and reach the sink together on `flush` (one write),
        // instead of one write per record. `into_inner` also drains, so no bytes are lost.
        let mut w = WalWriter::new(Vec::new());
        w.append(&WalRecord::BeginTxn { txn: TxnId(1) }).unwrap();
        w.append(&WalRecord::CommitTxn { txn: TxnId(1) }).unwrap();
        assert!(
            w.get_ref().is_empty(),
            "records buffer until flush, not written per append"
        );
        w.flush().unwrap();
        let after_flush = w.get_ref().len();
        assert!(after_flush > 0, "flush drains the buffer to the sink");

        // A further append re-buffers; the sink is unchanged until the next drain.
        w.append(&WalRecord::CommitTxn { txn: TxnId(2) }).unwrap();
        assert_eq!(
            w.get_ref().len(),
            after_flush,
            "the new record is still buffered"
        );
        let all = w.into_inner();
        assert!(
            all.len() > after_flush,
            "into_inner drains the remaining buffered record"
        );

        // The drained bytes decode back to exactly the three records, in order.
        let replayed = crate::WalReader::new(std::io::Cursor::new(all))
            .replay()
            .unwrap();
        assert!(matches!(
            replayed.as_slice(),
            [
                (_, WalRecord::BeginTxn { txn: TxnId(1) }),
                (_, WalRecord::CommitTxn { txn: TxnId(1) }),
                (_, WalRecord::CommitTxn { txn: TxnId(2) }),
            ]
        ));
    }

    #[test]
    fn a_long_run_of_appends_spills_at_the_cap_bounding_memory() {
        // Beyond the cap the buffer spills to the sink mid-run, so a large transaction's userspace
        // footprint stays bounded rather than holding every record; a final flush drains the rest.
        let mut w = WalWriter::new(Vec::new());
        for i in 0..100u64 {
            // Poorly-compressible value (a multiplicative-hash byte pattern) so the framed records
            // do not lz4 away to nothing — 100 × ~4 KiB clears the 64 KiB cap and forces spills.
            let value: Vec<u8> = (0..4096u32)
                .map(|j| (j.wrapping_mul(2_654_435_761).wrapping_add(i as u32) >> 13) as u8)
                .collect();
            w.append(&WalRecord::Put {
                key: i.to_le_bytes().to_vec(),
                value,
            })
            .unwrap();
        }
        assert!(
            !w.get_ref().is_empty(),
            "the run spilled to the sink at the cap (buffer stays bounded)"
        );
        w.flush().unwrap();
        assert_eq!(
            crate::WalReader::new(std::io::Cursor::new(w.into_inner()))
                .replay()
                .unwrap()
                .len(),
            100,
            "every record reached the sink across the spills + final flush"
        );
    }
}
