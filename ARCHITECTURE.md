# Architecture

NusaDB is a single-node relational engine organized as a seven-layer stack. Each layer is a small set
of crates with a strict dependency direction: `nusadb-core` is depended on by everything, and no
inner (storage-side) crate imports an outer (SQL/network-side) one. Shared types live in
`nusadb-core`.

## Layers

| Layer | Responsibility |
| ----- | -------------- |
| L1 Client | `nusadb-cli` REPL, the C API, language drivers |
| L2 Protocol | binary wire protocol over TCP, TLS, SCRAM-SHA-256, connection pooling |
| L3 SQL engine | parser → analyzer → planner/optimizer → executor |
| L4 Transactions | MVCC, lock manager, rollback, isolation levels |
| L5 Storage engine | clustered B-link/B+tree, MVCC version store, index engine |
| L6 WAL | append-only log, CRC32, lz4, crash recovery |
| L7 Physical storage | data files, WAL segments, catalog/meta, index files |

## Crate map

| Crate | Layer | Responsibility |
| ----- | ----- | -------------- |
| `nusadb-core` | all | Shared types (`PageId`, `Lsn`, `TxnId`), port traits (`PageStore`, `Clock`, `Rng`), and the `StorageEngine` treaty |
| `nusadb-storage` | L7/L5 | `DiskManager`, `BufferPool`, page/B-tree primitives, catalog |
| `nusadb-wal` | L6 | Append-only WAL writer/reader, CRC32, lz4, group commit |
| `nusadb-btree` | L5/L4 | Clustered B-link/B+tree engine: MVCC, no-wait locks, SSI, WAL-replay recovery, background purge — implements `StorageEngine` |
| `nusadb-sql` | L3 | Parser wrapper, analyzer, planner, executor, session |
| `nusadb-wire` | L2 | Wire-protocol frames, TLS, SCRAM-SHA-256 auth |
| `nusadb-server` | L1 | Server binary that wires all layers together |
| `nusadb-cli` | L1 | Interactive SQL shell |
| `nusadb-libnusa` / `nusadb-capi` | L1 | Native client library and C ABI |
| `nusadb-sim` | test | Deterministic-simulation adapters (`SimStorage`, `SimClock`, `SimRng`) with fault injection |
| `nusadb-test-utils` | test | Cross-crate test helpers |
| `nusadb-e2e` | test | End-to-end SQL tests and the SQLLogicTest corpus |

Dependency direction: `nusadb-core` ← everything. No internal cycle is permitted.

## The `StorageEngine` treaty

`nusadb-core::engine` defines the `StorageEngine` and `TupleScan` traits. This is the seam between
the storage/transaction spine and the SQL surface, so the two halves can be developed independently.
The spine (`nusadb-btree`'s `BtreeEngine`) implements `StorageEngine`, and the surface (`nusadb-sql`)
consumes `&dyn StorageEngine`, running against an in-memory test double until the spine is ready.

Two choices are baked into the trait. Tuples are opaque `Vec<u8>`, so the SQL layer owns encoding.
Transactions are identified by `TxnId` rather than borrowed handles, which avoids borrow-checker
conflicts on statements like `INSERT ... SELECT`.

## Deterministic simulation testing

`nusadb-sim` provides in-memory implementations of `PageStore`, `Clock`, and `Rng` with configurable
fault injection (torn writes, `fsync` failures, power loss). The same engine code runs against real
disk in production and against these adapters in tests, driven by a single seed for reproducible
fault scenarios.

## Key technical decisions

- Pages are 8 KB, and each B-tree node is one page.
- The WAL is written before the data page changes. It is lz4-compressed with a CRC32 per record.
- Under MVCC, each row carries `xmin`/`xmax`. Readers see a snapshot as of transaction start and
  never block writers.
- Locks are row-level and no-wait: a conflict aborts rather than waits, so deadlocks cannot form.
  Serializability uses row-level SSI.
- The executor uses a vectorized batch model (columnar `RecordBatch`) with SIMD kernels on the hot
  paths, alongside a row-at-a-time path.
- TLS is `rustls` only.
- `NUMERIC` is exact base-10 fixed-point (an `i128` mantissa plus a scale), so there is no
  binary-float rounding.
- Data at rest can be encrypted with AES-256-GCM-SIV under a per-database data key sealed by an
  operator's master key. The decision record follows.

## Decision record: encryption at rest

### Status

Accepted, 2026-10-14.

### Context

Operators asked for encryption at rest, switched on by a server flag and usable from the
container image. The workspace already had record-level AES-256-GCM-SIV for log records in
`nusadb-wal`, but nothing used it: the engine wrote its log, checkpoint image, page segments and
page spill file in the clear, and so did every copy of them (the checkpoint archive, scheduled
backups, a standby's seed) and the executor's sort and join spill files. A row inserted on a live
server could be read back with `grep` from the log, the page segment, the archived log and the
archived segment. The old WAL cipher also derived its nonce from the log position, which repeats
after a log is truncated and appended again, and in every checkpoint image, whose records always
start at position 1.

### Decision

**Scope.** Every file that holds row values, names or keys is sealed: the log (`btree.wal`) and
therefore its archived and shipped segments, the checkpoint image's page directory and records,
the page segments, the page spill file, and the executor's spill files. Left in the clear, as
metadata: the format, lock and engine marker files, `global/databases` (database names are also
directory names), the archive's segment lists, fork records and format file, the image header
(magic, version, log position covered, page counts) and segment names.

**Cipher.** AES-256-GCM-SIV from the `aes-gcm-siv` crate already in the workspace; no new crypto
dependency (`getrandom`, `sha2` and `zeroize` were already workspace dependencies). A sealed string
is `nonce (12) | ciphertext | tag (16)`. Each sealer draws a random 96-bit nonce base from the
operating system and adds a 64-bit counter per seal, so one sealer never repeats a nonce and two
sealers under one key collide only if their random bases fall within each other's counter range.
The nonce is stored, never derived, so a log position written twice still gets a fresh nonce.
GCM-SIV is chosen for misuse resistance: should a nonce ever repeat, it reveals only whether two
plaintexts were equal.

**Binding (AAD).** Every sealed string carries what it belongs to: a log record its log position
(domain 1); an image record the position its image covers and its index in the image (domain 2);
a page its page id, slot and segment name (domain 3); the image's page directory the image header
and segment names (domain 4); a spilled page its id and write generation (domain 5). A record or
page moved to another place, another image or another file kind fails its tag. A database created
empty has a data key of its own, so nothing moves between two such databases. A database restored
from an archive and a standby seeded from one keep the archive's data key, because they continue
its history and write into it; between a database, its restores and its standbys a record or page
is bound to its place but not to which copy it came from. The executor's spill records carry a
fixed AAD under a key that exists only in one process, so they are bound to that process only.

**Checksums and tags.** The log keeps encrypt-then-CRC: the CRC32 covers the sealed frame, so a
torn tail is still recognised without the key. A frame whose CRC is intact but whose tag fails
cannot come from a crash, so recovery refuses the log at that point, the tail included, instead of
truncating it. Pages are opened first and then checked against the CRC32 of the plain page kept in
the (sealed) directory.

**Keys.** The operator supplies a master key: 32 bytes as 64 hex digits in a file
(`--encryption-key-file`, or `NUSADB_ENCRYPTION_KEY_FILE`; `NUSADB_ENCRYPTION_KEY` may carry the
hex itself, which is weaker: the key stays readable in the process environment). The file must
not be readable by others nor writable by group or others. Each database gets a random data key
when it is created, stored sealed under the master key in the envelope `<wal>.key` with a random
key id and a check value of the master key (8 bytes of an HMAC-SHA-256 under the key of the id,
so installations under one master key store unrelated values). Key material is wiped from memory
when it is dropped, AES round keys included. Archives and backups carry a copy of the envelope. The
spill files use keys drawn per process and never stored, since they are never read after the
process ends.

**Refusals.** Before anything is written, at server start for every database and again when an
engine opens: an encrypted database without a key or with another key is refused (naming the
given key's fingerprint), a database whose envelope is missing is refused, an unencrypted
database that holds data is refused when a key is given, an envelope found beside unencrypted
data is refused as not belonging to it (the directory is never recorded as encrypted), and a
database an interrupted `--encrypt-from` left behind is refused until the copy is run again. An archive must carry the same data key as its
database, and an unencrypted archive is refused for an encrypted database and the other way round.

**Turning it on.** A database is encrypted when it is created; there is no in-place conversion.
Converting in place would need a crash-safe swap of every file of a database while the old and
new forms share one directory, and a failure part way would leave data in the clear beside a flag
claiming encryption. Instead `--encrypt-from DIR` copies every database of a stopped data
directory into a new one: it opens and checkpoints the source, writes a new full page segment and
image sealed under a new data key, and renames the image into place last, so an interrupted copy
leaves a directory marked incomplete, which no open accepts and a second run writes again.
For a whole data directory, a new copy needs an empty target (no database the source lacks, no
file under `base/` but empty lock files) and a source that is a data directory. It writes a
journal in `global/` naming the source and a keyed check value of the master key before anything
else, records each database in it before creating it, marks every target before any is
registered in the new catalog (so no registered database ever lacks both its mark and a finished
copy), and removes the journal only once the catalog and `base/` name exactly the source's
databases, none is marked, and each opens as a finished copy under the key. A rerun continues
only from the same source path under the same key and copies every database again: a log
position is no proof of contents, since a database dropped and created again, or another data
directory put at the same path, can stand at the same position. A database the journal records
as created that the source no longer has is unregistered and its directory removed, but only
when that directory is a real directory (never a link) holding database files alone, removed
file by file with the directory going last; a link at a database's file names is refused, and
every scratch file, marker and journal is written as a new file, never through a link (the target
must not be changed by others while the copy runs); anything the copy did not create stops the run
with nothing removed. A target without a journal that holds databases is refused. The source
keeps working until
the operator removes it.

**Data format.** An encrypted directory is data format 3; an unencrypted one stays format 2, so a
release that reads format 2 keeps opening it and refuses an encrypted one by number. The sealed
checkpoint image is image version 4. No directory is upgraded from 2 to 3.

**Rotation.** `--rotate-encryption-key NEW` (offline) reseals every envelope found in the data
directory, the archive and the backup directory under the new master key; each envelope is
replaced atomically, and a second run finishes an interrupted one. Data keys are not rotated in
place; `--encrypt-from` with `--encrypt-from-key-file` copies a directory under new data keys.

**Interactions.** The page cache holds plain pages: they are opened when loaded from a segment
and sealed when written to a new segment or to the spill file. Incremental checkpoints keep a
carried page in its segment and slot, so its binding does not change. Backups hard-link sealed
files and copy the envelope. A standby needs the same master key; it opens shipped segments with
the archive's data key and seals what it appends under its own.

### Consequences

- Measured on an Apple M-series laptop, release build, 200,000 rows of 200 bytes in transactions
  of 1,000 rows, median of three runs: bulk insert 1.42 s unencrypted and 1.51 s encrypted (+7%),
  checkpoint 41 ms and 51 ms, reopen 3.5 ms and 3.6 ms, a scan of every row from a cold cache
  24 ms and 42 ms. Pages already in the cache cost nothing extra; every page read from or written
  to disk and every log record pays one AEAD operation.
- On 64-bit ARM the RustCrypto crates use the AES and carry-less multiply instructions only when
  built with `--cfg aes_armv8 --cfg polyval_armv8`, which `.cargo/config.toml` sets for those
  targets; without them the same cold scan took 219 ms. On x86-64 the instructions are detected
  at run time.
- A sealed page takes 8,220 bytes instead of 8,192 and a log record 28 more bytes.
- Not protected: memory (the cache, keys, core dumps), a key file kept on the data disk, metadata
  (file names and sizes, database names, log positions in file names), and the order and
  completeness of the log (its tail can be cut, or an older complete copy put back, by someone who
  can write the files). The wire is protected separately by TLS.
- Losing the master key loses the data.
