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
