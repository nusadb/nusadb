//! Transient on-disk run/partition files for spilling.
//!
//! A [`SpillWriter`] appends rows (length-prefixed, via [`codec`](super::codec)) to a fresh file;
//! [`into_reader`](SpillWriter::into_reader) flushes it and hands back a [`SpillReader`] that streams
//! the rows back. Exactly one handle owns the path at a time and **deletes the file when it drops**
//! (RAII), so a spilled run never outlives its query — even if the operator errors or panics
//! mid-build. The server sweeps the scratch dir on startup to clear files orphaned by a crash.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use super::codec;
use crate::ast;
use crate::error::Error;
use crate::executor::row::Row;

const fn io_error(e: io::Error) -> Error {
    Error::Core(nusadb_core::Error::Io(e))
}

/// Appends rows to a transient spill file; deletes the file on drop unless converted to a
/// [`SpillReader`].
pub(in crate::executor) struct SpillWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    /// When `true`, the file lives on past this writer (a [`SpillReader`] took over its lifetime).
    handed_off: bool,
}

impl SpillWriter {
    /// Create a fresh spill file at `path` (truncating any stale file there).
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error if the file cannot be created.
    pub(in crate::executor) fn create(path: PathBuf) -> Result<Self, Error> {
        let file = File::create(&path).map_err(io_error)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            handed_off: false,
        })
    }

    /// Append one row to the file.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn write_row(&mut self, row: &[ast::Value]) -> Result<(), Error> {
        let bytes = codec::encode_row(row)?;
        let len = u32::try_from(bytes.len()).map_err(|_| {
            Error::Core(nusadb_core::Error::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "spilled row exceeds 4 GiB",
            )))
        })?;
        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(io_error)?;
        self.writer.write_all(&bytes).map_err(io_error)?;
        Ok(())
    }

    /// Append one opaque length-prefixed record (bytes the caller already encoded — e.g. a sorted
    /// index-build entry), bypassing the row codec.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error, or if the record exceeds 4 GiB.
    pub(in crate::executor) fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let len = u32::try_from(bytes.len()).map_err(|_| {
            Error::Core(nusadb_core::Error::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "spilled record exceeds 4 GiB",
            )))
        })?;
        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(io_error)?;
        self.writer.write_all(bytes).map_err(io_error)?;
        Ok(())
    }

    /// Flush the buffered writes and reopen the file for reading. The returned [`SpillReader`] takes
    /// over deletion of the file.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn into_reader(mut self) -> Result<SpillReader, Error> {
        self.writer.flush().map_err(io_error)?;
        let reader = SpillReader::open(self.path.clone())?;
        // Only once the reader exists does it own the file's lifetime; until then this writer's Drop
        // must still delete the file (so a failed reopen does not leak it).
        self.handed_off = true;
        Ok(reader)
    }
}

impl SpillWriter {
    /// Flush the buffered writes and share the file between any number of independent cursors
    /// ([`SharedSpill::cursor`]). The file is deleted once the [`SharedSpill`] and every cursor
    /// opened from it are gone.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn into_shared(mut self) -> Result<SharedSpill, Error> {
        self.writer.flush().map_err(io_error)?;
        self.handed_off = true;
        Ok(SharedSpill {
            path: std::rc::Rc::new(SharedPath(self.path.clone())),
        })
    }
}

/// Deletes a shared spill file once its last holder is gone.
struct SharedPath(PathBuf);

impl Drop for SharedPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A finished spill file that several cursors read independently, each from the start.
pub(in crate::executor) struct SharedSpill {
    path: std::rc::Rc<SharedPath>,
}

impl SharedSpill {
    /// A new cursor positioned at the first row.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error if the file cannot be opened.
    pub(in crate::executor) fn cursor(&self) -> Result<SpillCursor, Error> {
        let file = File::open(&self.path.0).map_err(io_error)?;
        Ok(SpillCursor {
            _path: std::rc::Rc::clone(&self.path),
            reader: BufReader::new(file),
        })
    }
}

/// One forward-only reader over a [`SharedSpill`]; keeps the file alive while it exists.
pub(in crate::executor) struct SpillCursor {
    _path: std::rc::Rc<SharedPath>,
    reader: BufReader<File>,
}

impl SpillCursor {
    /// Read the next row, or `Ok(None)` at end of file.
    ///
    /// # Errors
    /// [`Error::Core`] for an I/O error, or [`Error::MalformedTuple`] if the record is corrupt.
    pub(in crate::executor) fn read_row(&mut self) -> Result<Option<Row>, Error> {
        read_row_from(&mut self.reader)
    }
}

/// Read one length-prefixed row record from `reader`, or `Ok(None)` at a clean end of file.
fn read_row_from(reader: &mut BufReader<File>) -> Result<Option<Row>, Error> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {},
        // A clean EOF exactly at a record boundary is the normal end of the run.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(io_error(e)),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(Some(codec::decode_row(&bytes)?))
}

impl Drop for SpillWriter {
    fn drop(&mut self) {
        if !self.handed_off {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Streams rows back from a spill file; deletes the file on drop.
pub(in crate::executor) struct SpillReader {
    path: PathBuf,
    reader: BufReader<File>,
}

impl SpillReader {
    fn open(path: PathBuf) -> Result<Self, Error> {
        let file = File::open(&path).map_err(io_error)?;
        Ok(Self {
            path,
            reader: BufReader::new(file),
        })
    }

    /// Read the next row, or `Ok(None)` at end of file.
    ///
    /// # Errors
    /// [`Error::Core`] for an I/O error, or [`Error::MalformedTuple`] if the record is corrupt.
    pub(in crate::executor) fn read_row(&mut self) -> Result<Option<Row>, Error> {
        read_row_from(&mut self.reader)
    }

    /// Read the next opaque record written by [`SpillWriter::write_bytes`], or `Ok(None)` at end of
    /// file. The caller decodes the bytes; this does not go through the row codec.
    ///
    /// # Errors
    /// [`Error::Core`] for an I/O error.
    pub(in crate::executor) fn read_bytes(&mut self) -> Result<Option<Vec<u8>>, Error> {
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(io_error(e)),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut bytes = vec![0u8; len];
        self.reader.read_exact(&mut bytes).map_err(io_error)?;
        Ok(Some(bytes))
    }
}

impl Drop for SpillReader {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Builds a [`RandomSpill`]: rows appended to a data file, each row's byte offset to a second file
/// of fixed 8-byte records, so any row can later be found without holding anything per row.
pub(in crate::executor) struct RandomSpillWriter {
    data: SpillWriter,
    offsets: SpillWriter,
    at: u64,
    len: usize,
}

impl RandomSpillWriter {
    /// Create the two files as `<stem>.rows` and `<stem>.offsets` in `dir`.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn create(dir: &std::path::Path, stem: &str) -> Result<Self, Error> {
        Ok(Self {
            data: SpillWriter::create(dir.join(format!("{stem}.rows")))?,
            offsets: SpillWriter::create(dir.join(format!("{stem}.offsets")))?,
            at: 0,
            len: 0,
        })
    }

    /// Append one row.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn write_row(&mut self, row: &[ast::Value]) -> Result<(), Error> {
        let bytes = codec::encode_row(row)?;
        self.data.write_bytes(&bytes)?;
        self.offsets
            .writer
            .write_all(&self.at.to_le_bytes())
            .map_err(io_error)?;
        self.at += 4 + bytes.len() as u64;
        self.len += 1;
        Ok(())
    }

    /// Finish writing and open the files for reading by position.
    ///
    /// # Errors
    /// [`Error::Core`] wrapping the underlying I/O error.
    pub(in crate::executor) fn finish(mut self) -> Result<RandomSpill, Error> {
        self.data.writer.flush().map_err(io_error)?;
        self.offsets.writer.flush().map_err(io_error)?;
        let data = File::open(&self.data.path).map_err(io_error)?;
        let offsets = File::open(&self.offsets.path).map_err(io_error)?;
        self.data.handed_off = true;
        self.offsets.handed_off = true;
        Ok(RandomSpill {
            data_path: self.data.path.clone(),
            offsets_path: self.offsets.path.clone(),
            data: BufReader::new(data),
            offsets: BufReader::new(offsets),
            offsets_at: 0,
            next: Some(0),
            data_end: self.at,
            len: self.len,
        })
    }
}

/// Rows on disk read back by position; deletes its files on drop.
///
/// Reading rows in order costs no seek: the data reader stays on the row after the last one read.
/// A jump moves both readers relative to where they are, so a short step (a backward fetch) stays
/// inside their buffers.
#[derive(Debug)]
pub(in crate::executor) struct RandomSpill {
    data_path: PathBuf,
    offsets_path: PathBuf,
    data: BufReader<File>,
    offsets: BufReader<File>,
    /// Where the offsets reader is, in bytes.
    offsets_at: u64,
    /// The row the data reader is on, or `None` after a failed read left it somewhere unknown.
    next: Option<usize>,
    /// The data file's length: where the reader is after the last row.
    data_end: u64,
    len: usize,
}

impl RandomSpill {
    /// How many rows it holds.
    pub(in crate::executor) const fn len(&self) -> usize {
        self.len
    }

    /// The row at `index`, or `Ok(None)` past the end.
    ///
    /// # Errors
    /// [`Error::Core`] for an I/O error or a file that ends before row `index`, or
    /// [`Error::MalformedTuple`] if the record is corrupt.
    pub(in crate::executor) fn get(&mut self, index: usize) -> Result<Option<Row>, Error> {
        use std::io::{Seek, SeekFrom};
        if index >= self.len {
            return Ok(None);
        }
        // The reader's position is unknown until this read succeeds.
        let next = self.next.take();
        match next {
            Some(next) if next == index => {},
            Some(next) => {
                let here = self.offset_of(next)?;
                let target = self.offset_of(index)?;
                self.data
                    .seek_relative(relative(here, target))
                    .map_err(io_error)?;
            },
            None => {
                let target = self.offset_of(index)?;
                self.data.seek(SeekFrom::Start(target)).map_err(io_error)?;
            },
        }
        let row = read_row_from(&mut self.data)?.ok_or_else(|| {
            io_error(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("spill file ends before row {index} of {}", self.len),
            ))
        })?;
        self.next = Some(index + 1);
        Ok(Some(row))
    }

    /// The byte offset of row `index` in the data file (`data_end` for the row after the last).
    fn offset_of(&mut self, index: usize) -> Result<u64, Error> {
        if index >= self.len {
            return Ok(self.data_end);
        }
        let want = index as u64 * 8;
        let mut at = [0u8; 8];
        let read = self
            .offsets
            .seek_relative(relative(self.offsets_at, want))
            .and_then(|()| self.offsets.read_exact(&mut at));
        // After a failed seek or read the reader's position is unknown; re-anchor it absolutely.
        self.offsets_at = if read.is_ok() {
            want + 8
        } else {
            use std::io::Seek;
            self.offsets.rewind().map_err(io_error)?;
            0
        };
        read.map_err(io_error)?;
        Ok(u64::from_le_bytes(at))
    }
}

/// The signed distance from byte `from` to byte `to`.
fn relative(from: u64, to: u64) -> i64 {
    i64::try_from(i128::from(to) - i128::from(from)).unwrap_or(i64::MAX)
}

impl Drop for RandomSpill {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.data_path);
        let _ = std::fs::remove_file(&self.offsets_path);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A unique scratch path under the OS temp dir — no RNG (DST-safe), just pid + a counter.
    fn scratch_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("nusadb-spill-test-{}-{n}.tmp", std::process::id()))
    }

    fn sample_rows() -> Vec<Row> {
        vec![
            vec![ast::Value::Int(1), ast::Value::Text("a".to_owned())],
            vec![ast::Value::Null, ast::Value::Bool(true)],
            vec![
                ast::Value::Array(vec![ast::Value::Int(7)]),
                ast::Value::Float(2.5),
            ],
        ]
    }

    #[test]
    fn write_then_read_round_trips_and_cleans_up() {
        let path = scratch_path();
        let rows = sample_rows();

        let mut writer = SpillWriter::create(path.clone()).expect("create");
        for row in &rows {
            writer.write_row(row).expect("write");
        }
        let mut reader = writer.into_reader().expect("into_reader");
        assert!(path.exists(), "file lives while the reader holds it");

        let mut read_back = Vec::new();
        while let Some(row) = reader.read_row().expect("read") {
            read_back.push(row);
        }
        assert_eq!(read_back, rows);

        drop(reader);
        assert!(!path.exists(), "reader deletes the file on drop");
    }

    #[test]
    fn writer_dropped_without_handoff_deletes_the_file() {
        let path = scratch_path();
        {
            let mut writer = SpillWriter::create(path.clone()).expect("create");
            writer.write_row(&[ast::Value::Int(1)]).expect("write");
            assert!(path.exists(), "file exists while writing");
        } // writer dropped without into_reader
        assert!(!path.exists(), "an abandoned writer deletes its file");
    }

    #[test]
    fn empty_file_reads_as_no_rows() {
        let writer = SpillWriter::create(scratch_path()).expect("create");
        let mut reader = writer.into_reader().expect("into_reader");
        assert!(reader.read_row().expect("read").is_none());
    }

    #[test]
    fn random_spill_reads_any_row_and_cleans_up() {
        let dir = std::env::temp_dir();
        let scratch = scratch_path();
        let stem = scratch
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_owned();
        let rows: Vec<Row> = (0..500_i64)
            .map(|i| {
                vec![
                    ast::Value::Int(i),
                    ast::Value::Text("x".repeat(usize::try_from(i % 9).unwrap())),
                ]
            })
            .collect();
        let mut writer = RandomSpillWriter::create(&dir, &stem).unwrap();
        for row in &rows {
            writer.write_row(row).unwrap();
        }
        let mut spill = writer.finish().unwrap();
        assert_eq!(spill.len(), 500);
        // Jumps, then in-order runs forward and backward mixed with jumps.
        let forward = 0..500;
        let backward = (0..500).rev();
        let order = [499, 0, 250, 1, 498, 250]
            .into_iter()
            .chain(forward)
            .chain(backward)
            .chain([3, 4, 5, 400, 401, 2, 1, 0, 499]);
        for index in order {
            assert_eq!(spill.get(index).unwrap().as_ref(), rows.get(index));
        }
        assert!(spill.get(500).unwrap().is_none());
        let data = dir.join(format!("{stem}.rows"));
        assert!(data.exists());
        drop(spill);
        assert!(!data.exists() && !dir.join(format!("{stem}.offsets")).exists());
    }
}
