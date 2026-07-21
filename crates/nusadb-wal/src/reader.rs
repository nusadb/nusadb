//! Sequential WAL reader for crash recovery.
//!
//! Reads framed records in order. A truncated trailing record (partial write from an
//! unclean shutdown) and a CRC mismatch are both treated as the **end of the durable
//! log** — recovery replays the valid prefix and stops, rather than erroring. Genuine
//! I/O failures still propagate.

// The fixed 16-byte frame header is parsed with constant in-bounds ranges.
#![allow(clippy::indexing_slicing)]

use std::io::{ErrorKind, Read};

use nusadb_core::{Lsn, Result};

use crate::crypto::WalCipher;
use crate::record::WalRecord;

/// Refuse to allocate for a record claiming to be larger than this (corrupt length guard).
const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;

/// The result of [`WalReader::read_record_checked`].
///
/// A valid record, a clean/torn end-of-log, or a corrupt (but fully-consumed) record. The
/// distinction lets recovery tell a torn *tail* (safe to truncate) from a *hole in the middle* of
/// the log (must refuse to open).
#[derive(Debug)]
pub enum ReadOutcome {
    /// A valid record.
    Record(Lsn, WalRecord),
    /// A clean end of log, or a record whose header/body is incomplete at EOF (a crash mid-append).
    /// Nothing durable follows and the reader cannot advance past it.
    Eof,
    /// A complete record's worth of bytes was read but failed validation (CRC / AEAD / body-kind /
    /// decompress / decode). The reader has advanced past it, so scanning may continue — a valid
    /// record found afterwards proves a mid-log hole.
    Corrupt,
}

/// Sequential reader that yields valid `(Lsn, WalRecord)` pairs and stops at the first
/// truncated or CRC-invalid frame.
#[derive(Debug)]
pub struct WalReader<R: Read> {
    inner: R,
    /// When set, each record's framed body is decrypted before decompression.
    cipher: Option<WalCipher>,
}

impl<R: Read> WalReader<R> {
    /// Wrap a readable WAL source positioned at the first record.
    pub const fn new(inner: R) -> Self {
        Self {
            inner,
            cipher: None,
        }
    }

    /// Wrap an **encrypted** WAL source (written by
    /// [`WalWriter::new_encrypted`](crate::WalWriter::new_encrypted)) with the data-encryption
    /// `key`. Each record is decrypted after the CRC check and before decompression; a
    /// wrong key fails the AEAD tag and is treated as the end of the durable log.
    pub fn new_encrypted(inner: R, key: &[u8; 32]) -> Self {
        Self {
            inner,
            cipher: Some(WalCipher::new(key)),
        }
    }

    /// Read the next valid record, or `Ok(None)` at the end of the durable log
    /// (clean EOF, truncated trailing record, or a corrupt record).
    ///
    /// This collapses "clean/torn end" and "corruption" into `None`; recovery that must tell a
    /// torn *tail* (safe to truncate) from a *hole in the middle* of the log (must refuse to open,
    /// else committed data past the hole is silently lost) uses [`read_record_checked`] instead.
    ///
    /// [`read_record_checked`]: Self::read_record_checked
    ///
    /// # Errors
    /// Propagates genuine I/O errors (not EOF).
    pub fn read_record(&mut self) -> Result<Option<(Lsn, WalRecord)>> {
        match self.read_record_checked()? {
            ReadOutcome::Record(lsn, record) => Ok(Some((lsn, record))),
            ReadOutcome::Eof | ReadOutcome::Corrupt => Ok(None),
        }
    }

    /// Read the next record, distinguishing a clean/torn end-of-log from a corrupt record.
    ///
    /// - [`ReadOutcome::Record`] — a valid record.
    /// - [`ReadOutcome::Eof`] — a clean end, or a record whose header/body is **incomplete** at
    ///   EOF (a crash mid-append): nothing durable follows and the reader cannot advance past it,
    ///   so truncating the file to the last good record is safe.
    /// - [`ReadOutcome::Corrupt`] — a **complete** record's worth of bytes was read but did not
    ///   validate (CRC / AEAD / body-kind / decompress / decode). The reader HAS advanced past it,
    ///   so the caller can keep reading: a valid record found afterwards proves a mid-log hole
    ///   (recovery must then refuse to open, not truncate).
    ///
    /// # Errors
    /// Propagates genuine I/O errors (not EOF).
    pub fn read_record_checked(&mut self) -> Result<ReadOutcome> {
        let mut header = [0u8; 16]; // lsn(8) + len(4) + crc(4)
        if !self.fill(&mut header)? {
            return Ok(ReadOutcome::Eof); // clean or torn end-of-log (incomplete header)
        }
        let lsn = Lsn(bytemuck::pod_read_unaligned::<u64>(&header[0..8]));
        let len = bytemuck::pod_read_unaligned::<u32>(&header[8..12]) as usize;
        let crc = bytemuck::pod_read_unaligned::<u32>(&header[12..16]);

        if len > MAX_RECORD_BYTES {
            // An implausible length: the header is garbage (a torn/garbage region) and — crucially
            // — its real length is unknown, so the reader cannot skip to a following record. It is
            // treated as an unrecoverable end (`Eof`), like a torn tail; the definitive mid-log
            // hole detection covers the far more common case of body bit-rot with an intact header.
            tracing::warn!(
                ?lsn,
                len,
                "WAL record length implausible; stopping recovery"
            );
            return Ok(ReadOutcome::Eof);
        }

        let mut body = vec![0u8; len];
        if !self.fill(&mut body)? {
            return Ok(ReadOutcome::Eof); // torn trailing record (incomplete body)
        }
        // From here the full record's bytes are consumed, so the reader has advanced past it: any
        // validation failure below is `Corrupt`, and the caller may keep scanning. The CRC covers
        // the header (lsn + len) too — `header[0..12]` — so a corrupted length/lsn is caught, not
        // only a body bit-flip.
        if frame_crc(&header[0..12], &body) != crc {
            tracing::warn!(?lsn, "WAL CRC mismatch");
            return Ok(ReadOutcome::Corrupt);
        }
        // Decrypt before interpreting the body when the log is encrypted. A failed tag
        // check (wrong key, or a torn trailing record that still happened to match its CRC) ends
        // replay.
        let body = match &self.cipher {
            None => body,
            Some(cipher) => {
                let Ok(plain) = cipher.open(lsn.0, &body) else {
                    tracing::warn!(?lsn, "WAL record failed to decrypt");
                    return Ok(ReadOutcome::Corrupt);
                };
                plain
            },
        };
        decode_plain_body(&body).map_or_else(
            || {
                tracing::warn!(?lsn, "WAL record body empty, unknown-kind, or undecodable");
                Ok(ReadOutcome::Corrupt)
            },
            |record| Ok(ReadOutcome::Record(lsn, record)),
        )
    }

    /// Replay the whole durable log into a vector (valid prefix only).
    ///
    /// # Errors
    /// Propagates genuine I/O errors.
    pub fn replay(&mut self) -> Result<Vec<(Lsn, WalRecord)>> {
        let mut out = Vec::new();
        while let Some(entry) = self.read_record()? {
            out.push(entry);
        }
        Ok(out)
    }

    /// Fill `buf` completely. Returns `Ok(false)` if EOF is hit first (clean or partial),
    /// `Ok(true)` if fully read.
    fn fill(&mut self, buf: &mut [u8]) -> Result<bool> {
        let mut filled = 0;
        while filled < buf.len() {
            let Some(slice) = buf.get_mut(filled..) else {
                break;
            };
            match self.inner.read(slice) {
                Ok(0) => return Ok(false),
                Ok(n) => filled += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => {},
                Err(e) => return Err(e.into()),
            }
        }
        Ok(true)
    }
}

/// CRC32 over a frame's header bytes (`[lsn:u64][len:u32]`) followed by its body — the checksum the
/// writer stores. Covering the header means a corrupted length or lsn fails validation, not only a
/// body bit-flip.
fn frame_crc(header12: &[u8], body: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(header12);
    hasher.update(body);
    hasher.finalize()
}

/// Decode a plaintext framed body `[kind:u8][payload]` into a record, or `None` on any corruption
/// (empty body, unknown kind byte, bad lz4, or an undecodable payload).
fn decode_plain_body(body: &[u8]) -> Option<WalRecord> {
    let (&kind, data) = body.split_first()?;
    let payload = match kind {
        crate::writer::BODY_STORED => std::borrow::Cow::Borrowed(data),
        crate::writer::BODY_COMPRESSED => {
            std::borrow::Cow::Owned(lz4_flex::decompress_size_prepended(data).ok()?)
        },
        _ => return None,
    };
    WalRecord::decode(&payload)
}

/// One parsed frame at a byte offset in a fully-buffered WAL image — the building block of
/// [`recover_prefix`]'s scan and resync.
enum FrameParse {
    /// A valid record and the offset one past its frame.
    Record(Lsn, WalRecord, usize),
    /// Fewer than a full frame's bytes remain (a torn tail at EOF): the reader cannot advance.
    Incomplete,
    /// A full frame's worth of bytes was present but failed validation (CRC / decode), OR the length
    /// field was implausible (`> MAX_RECORD_BYTES`) so the true frame length is unknown. Either way
    /// the parser cannot be trusted to skip to the next frame — the caller must resync by scanning.
    Corrupt,
}

/// Parse the frame beginning at `off` in a buffered WAL image (unencrypted).
fn parse_frame(buf: &[u8], off: usize) -> FrameParse {
    let Some(header) = buf.get(off..off + 16) else {
        return FrameParse::Incomplete; // torn header at EOF
    };
    let lsn = Lsn(bytemuck::pod_read_unaligned::<u64>(&header[0..8]));
    let len = bytemuck::pod_read_unaligned::<u32>(&header[8..12]) as usize;
    let crc = bytemuck::pod_read_unaligned::<u32>(&header[12..16]);
    if len > MAX_RECORD_BYTES {
        // Implausible length: the header is corrupt and its real length is unknown — cannot advance.
        return FrameParse::Corrupt;
    }
    let body_start = off + 16;
    let Some(body) = buf.get(body_start..body_start + len) else {
        return FrameParse::Incomplete; // torn body at EOF
    };
    if frame_crc(&header[0..12], body) != crc {
        return FrameParse::Corrupt;
    }
    decode_plain_body(body).map_or(FrameParse::Corrupt, |record| {
        FrameParse::Record(lsn, record, body_start + len)
    })
}

/// The valid prefix recovered from a buffered WAL image, plus where to truncate a torn tail.
#[derive(Debug)]
pub struct RecoveredPrefix {
    /// The valid records, in log order.
    pub records: Vec<(Lsn, WalRecord)>,
    /// Byte length of the valid prefix: truncate a torn tail to here.
    pub good_bytes: u64,
    /// LSN of the last valid record (0 if none).
    pub last_lsn: u64,
}

/// A corruption that recovery must NOT truncate past, because committed data may follow it.
#[derive(Debug)]
pub struct MidLogHole {
    /// Byte offset where corruption was detected.
    pub at: u64,
    /// Byte offset of the first valid record found after the corruption (present for a genuine
    /// mid-log hole; `None` when the whole file has no valid prefix, e.g. an incompatible/older WAL
    /// format or corruption from byte 0).
    pub next_valid_at: Option<u64>,
}

/// Recover the valid prefix of a fully-buffered (unencrypted) WAL image, distinguishing a torn tail
/// (safe to truncate) from a mid-log hole (must refuse).
///
/// Scans frames from the start. On the first frame that fails to validate, it **resyncs** — scans
/// forward byte-by-byte for the next frame that both validates its CRC (header + body) and carries a
/// strictly greater LSN than the last good record. Finding one proves committed data survives past
/// the corruption, so this returns [`MidLogHole`] and the caller must refuse to open rather than
/// truncate (which would silently drop every transaction past the hole). If no valid frame follows,
/// the corruption reaches EOF — a torn/garbage tail — and the valid prefix is returned for
/// truncation. As an extra guard, a non-empty file with **no** valid prefix at all
/// (`good_bytes == 0` yet corruption was seen) is also refused, so an older-format or wholly
/// corrupt WAL is never silently truncated to nothing.
///
/// # Errors
/// Returns [`MidLogHole`] when the file must not be truncated.
pub fn recover_prefix(buf: &[u8]) -> core::result::Result<RecoveredPrefix, MidLogHole> {
    let mut off = 0usize;
    let mut good_bytes = 0u64;
    let mut last_lsn = 0u64;
    let mut records = Vec::new();
    loop {
        match parse_frame(buf, off) {
            FrameParse::Record(lsn, record, next) => {
                records.push((lsn, record));
                last_lsn = lsn.0;
                good_bytes = next as u64;
                off = next;
            },
            // A clean end, or a torn tail at EOF with no corruption in the middle: the valid prefix
            // is everything so far. Safe to truncate to `good_bytes`.
            FrameParse::Incomplete => break,
            // Corruption at `off`. Resync forward: is there any valid record after it?
            FrameParse::Corrupt => {
                let next_valid = find_valid_after(buf, off + 1, last_lsn);
                if next_valid.is_some() || good_bytes == 0 {
                    // Either a genuine mid-log hole (valid data survives past the corruption), or a
                    // non-empty file with no valid prefix at all (older format / corruption from the
                    // start). Refuse either way — never truncate past committed data or to nothing.
                    return Err(MidLogHole {
                        at: off as u64,
                        next_valid_at: next_valid,
                    });
                }
                // No valid record follows and we have a valid prefix: a torn/garbage tail. Truncate.
                break;
            },
        }
    }
    Ok(RecoveredPrefix {
        records,
        good_bytes,
        last_lsn,
    })
}

/// Scan forward from `start` for the first frame that validates AND carries an LSN strictly greater
/// than `last_lsn` (a real, later record — the resync anchor). `None` if none exists before EOF.
fn find_valid_after(buf: &[u8], start: usize, last_lsn: u64) -> Option<u64> {
    let mut cand = start;
    while cand < buf.len() {
        if let FrameParse::Record(lsn, _, _) = parse_frame(buf, cand)
            && lsn.0 > last_lsn
        {
            return Some(cand as u64);
        }
        cand += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use nusadb_core::{PageId, TxnId};

    use super::{WalReader, recover_prefix};
    use crate::WalWriter;
    use crate::record::WalRecord;

    /// Several distinct, decodable records for the recovery-scan tests.
    fn sample_records(n: u64) -> Vec<WalRecord> {
        (0..n)
            .flat_map(|i| {
                [
                    WalRecord::BeginTxn { txn: TxnId(i + 1) },
                    WalRecord::PageUpdate {
                        txn: TxnId(i + 1),
                        page: PageId(i + 10),
                        offset: 16,
                        image: format!("row-{i}-payload").into_bytes(),
                    },
                    WalRecord::CommitTxn { txn: TxnId(i + 1) },
                ]
            })
            .collect()
    }

    fn write_log(records: &[WalRecord]) -> Vec<u8> {
        let mut w = WalWriter::new(Vec::new());
        for r in records {
            w.append(r).unwrap();
        }
        w.into_inner()
    }

    /// Walk the frame start offsets of an intact log (`[lsn:8][len:4][crc:4][body:len]`).
    fn frame_offsets(bytes: &[u8]) -> Vec<usize> {
        let mut offs = Vec::new();
        let mut off = 0;
        while off + 16 <= bytes.len() {
            offs.push(off);
            let len = u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()) as usize;
            off += 16 + len;
        }
        offs
    }

    #[test]
    fn recover_prefix_refuses_corrupt_length_header_in_the_middle() {
        // Flipping a bit in a MIDDLE frame's `len` field is now caught (the CRC covers the header),
        // and the valid records after it prove a mid-log hole — refuse, do not truncate.
        let mut bytes = write_log(&sample_records(6));
        let offs = frame_offsets(&bytes);
        let target = offs[offs.len() / 2];
        bytes[target + 8] ^= 0x01; // low byte of `len`
        let err = recover_prefix(&bytes).expect_err("a corrupt mid-log length must refuse");
        assert!(
            err.next_valid_at.is_some(),
            "a valid record survives past the hole"
        );
    }

    #[test]
    fn recover_prefix_refuses_corrupt_lsn_header() {
        // An `lsn` bit-flip used to pass entirely undetected (the CRC did not cover it); it is now a
        // detected mid-log hole.
        let mut bytes = write_log(&sample_records(6));
        let offs = frame_offsets(&bytes);
        let target = offs[offs.len() / 2];
        bytes[target] ^= 0x01; // low byte of `lsn`
        let err = recover_prefix(&bytes).expect_err("a corrupt mid-log lsn must refuse");
        assert!(err.next_valid_at.is_some());
    }

    #[test]
    fn recover_prefix_resyncs_past_a_misaligned_length() {
        // A `len` corrupted to a plausible-but-wrong value desyncs the reader from every following
        // frame boundary. The byte-level resync still finds the next real frame and refuses, rather
        // than silently truncating the misaligned (but committed) records away.
        let mut bytes = write_log(&sample_records(8));
        let offs = frame_offsets(&bytes);
        let target = offs[2];
        bytes[target + 8] = bytes[target + 8].wrapping_add(3); // shift the boundary a few bytes
        let err = recover_prefix(&bytes).expect_err("misaligned length must still refuse");
        assert!(
            err.next_valid_at.is_some(),
            "resync must find the surviving records"
        );
    }

    #[test]
    fn recover_prefix_truncates_a_torn_tail() {
        // No corruption in the middle, just an interrupted final append: recover the prefix and
        // report where to truncate.
        let records = sample_records(5);
        let mut bytes = write_log(&records);
        let offs = frame_offsets(&bytes);
        let last_good_end = *offs.last().unwrap(); // start of the final frame
        bytes.truncate(bytes.len() - 3); // chop the final frame's tail
        let prefix = recover_prefix(&bytes).expect("a torn tail must recover, not refuse");
        assert_eq!(prefix.good_bytes as usize, last_good_end);
        assert_eq!(prefix.records.len(), records.len() - 1);
    }

    #[test]
    fn recover_prefix_accepts_a_clean_log() {
        let records = sample_records(4);
        let bytes = write_log(&records);
        let prefix = recover_prefix(&bytes).expect("a clean log recovers");
        assert_eq!(prefix.records.len(), records.len());
        assert_eq!(prefix.good_bytes as usize, bytes.len());
    }

    #[test]
    fn unknown_body_kind_stops_recovery() {
        // A body whose leading kind byte is neither STORED nor COMPRESSED is corruption. Forge
        // a frame with a valid CRC but an unknown kind, so it passes the CRC check yet is still
        // rejected (recovery ends rather than mis-decoding).
        let mut w = WalWriter::new(Vec::new());
        w.append(&WalRecord::BeginTxn { txn: TxnId(1) }).unwrap();
        let mut bytes = w.into_inner();

        bytes[16] = 0xFE; // kind byte (offset 16, unencrypted) → unknown
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        // Repair the CRC over the header (lsn + len) AND the forged body, matching the writer.
        let crc = super::frame_crc(&bytes[0..12], &bytes[16..16 + len]);
        bytes[12..16].copy_from_slice(&crc.to_le_bytes());

        let replayed = WalReader::new(Cursor::new(bytes)).replay().unwrap();
        assert!(
            replayed.is_empty(),
            "an unknown body kind must end recovery, not mis-decode"
        );
    }
}
