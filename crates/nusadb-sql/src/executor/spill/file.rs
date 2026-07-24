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
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {},
            // A clean EOF exactly at a record boundary is the normal end of the run.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(io_error(e)),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut bytes = vec![0u8; len];
        self.reader.read_exact(&mut bytes).map_err(io_error)?;
        Ok(Some(codec::decode_row(&bytes)?))
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
}
