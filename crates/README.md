# NusaDB

A relational database engine written from scratch in Rust — storage, WAL, MVCC transactions,
SQL engine, and wire protocol, with no third-party database core underneath.

> ## Status: `0.1.0.Beta` — pre-release
>
> **Do not run this on data you cannot afford to lose.** It is a Beta in the honest sense: the
> engine is feature-complete enough to be interesting, and *not yet* hardened enough to be trusted
> with production data. The known limits are written down in [Limitations](#limitations) rather
> than left for you to discover. Read that section before you deploy anything.

## Quick start

```bash
docker run -d --name nusadb -p 5678:5678 -v nusadb-data:/var/lib/nusadb \
  -e NUSADB_USER=nusa-root -e NUSADB_PASSWORD='choose-a-password' \
  nusadb/nusadb:0.1.0.Beta

docker exec -it nusadb nusa-cli --host 127.0.0.1:5678
```

Without `NUSADB_USER`/`NUSADB_PASSWORD` the server starts in **trust mode** — every client is
accepted with no password, and it says so loudly in the startup log. That is fine on a laptop and
unacceptable anywhere else.

```sql
CREATE TABLE akun (id INT NOT NULL, nama TEXT, bal INT);
INSERT INTO akun VALUES (1, 'satu', 100), (2, 'dua', 0);
SELECT id, nama, bal FROM akun ORDER BY id;
```

## Clients

Published and installable today:

| Language | Package | Install |
| --- | --- | --- |
| Rust | [`nusadb`](https://crates.io/crates/nusadb) | `cargo add nusadb` |
| Python | [`nusadb`](https://pypi.org/project/nusadb/) | `pip install nusadb` |
| Node.js | [`nusadb`](https://www.npmjs.com/package/nusadb) | `npm install nusadb` |
| Java (JDBC) | `com.nusadb:nusadb-jdbc:0.1.0` | Maven Central |
| Go | [`github.com/nusadb/go`](https://github.com/nusadb/go) | `go get github.com/nusadb/go` |
| PHP | [`nusadb/nusadb`](https://packagist.org/packages/nusadb/nusadb) | `composer require nusadb/nusadb` |
| Ruby | [`nusadb`](https://rubygems.org/gems/nusadb) | `gem install nusadb` |

A .NET driver exists in the repository but is **not published yet**.

## What is in this directory

Each crate is one layer of the stack. `nusadb-core` is depended on by everything else; nothing
depends back on it, and there are no cycles.

| Crate | Layer | Responsibility |
| --- | --- | --- |
| `nusadb-core` | all | Shared types (`PageId`, `Lsn`, `TxnId`), port traits, and the `StorageEngine` treaty — the seam between the storage spine and the SQL surface |
| `nusadb-storage` | L7/L5 | 8 KiB page format, disk manager, buffer pool, catalog |
| `nusadb-wal` | L6 | Append-only write-ahead log — CRC32 per record, lz4, group commit |
| `nusadb-btree` | L5/L4 | The storage engine: clustered B-link/B+tree, MVCC, no-wait locks, SSI, WAL recovery, background purge |
| `nusadb-sql` | L3 | Parser → analyzer → planner/optimizer → executor (vectorized, spill-to-disk) |
| `nusadb-wire` | L2 | Nusa Wire Protocol — framing, TLS, SCRAM-SHA-256 |
| `nusadb-server` | L1 | The server binary; composition root that wires every layer together |
| `nusadb-cli` | L1 | `nusa-cli` interactive shell |
| `nusadb-libnusa` | L1 | Native client library the drivers build on |
| `nusadb-capi` | L1 | C ABI over `libnusa`, for FFI-based drivers |
| `nusadb-sim` | test | Deterministic simulation adapters with fault injection |
| `nusadb-test-utils` | test | Cross-crate test helpers |
| `nusadb-e2e` | test | End-to-end SQL tests and the SQLLogicTest corpus |

## Design

```
L1 Client        nusa-cli · drivers (Rust/Python/Node/JDBC/Go)
L2 Protocol      Nusa Wire Protocol — TCP, TLS, SCRAM-SHA-256
L3 SQL Engine    Parser → Analyzer → Planner/Optimizer → Executor
L4 Transactions  MVCC · lock manager · savepoints · isolation levels
L5 Storage       Clustered B-link/B+tree · MVCC version store · indexes
L6 WAL           Append-only · CRC32 · lz4 · crash recovery
L7 Physical      8 KiB pages · WAL segments · catalog
```

- **Storage engine.** A clustered B-link/B+tree is the only engine. Rows live in the leaves, keyed
  by an engine-minted row-id; index entries and row versions carry MVCC visibility ranges. (An LSM
  engine existed earlier and was removed — an `lsm`-era data directory is refused at open, with a
  dump/restore hint rather than a silent misread.)
- **Transactions.** MVCC with `xmin`/`xmax` per version: readers take a snapshot and never block
  writers. Isolation levels are READ UNCOMMITTED / READ COMMITTED (default) / REPEATABLE READ /
  SERIALIZABLE, with SSI for the last. Locks are no-wait, so a conflict aborts rather than hangs.
- **Durability.** Every commit is fsynced before it is acknowledged, amortized across concurrent
  committers by group commit. Recovery replays the log. A WAL with a corrupt record *in the middle*
  makes the engine **refuse to open** rather than silently truncate the log to that point.
- **Garbage collection.** Dead versions are reclaimed by an incremental background purge, not by a
  periodic full-table rewrite.
- **Transaction ids are 64-bit**, so there is no wraparound to freeze against.

## Correctness

Correctness is treated as the product, not a phase. The suite that gates changes:

- **Deterministic simulation testing** — the engine runs against in-memory `PageStore`/`Clock`/`Rng`
  adapters with injectable faults, driven by a seed, so a failure replays exactly.
- **SQLLogicTest** corpus, plus a differential corpus that compares results against a reference
  engine byte-for-byte.
- **Jepsen-style history checking** (cycle detection) and **Hermitage** isolation tests.
- **Fuzzing** (`cargo-fuzz`) of the SQL parser, page decoder, WAL record codec, and wire frames.
- **Crash-consistency**: kill the process at arbitrary points and require that committed data
  survives and uncommitted data does not.

## Limitations

Honest list, current as of `0.1.0.Beta`:

- **The database is held in RAM.** The engine's page store is in-memory and durability comes from
  the write-ahead log, so **your dataset must fit in memory**. A disk-backed page store exists in
  `nusadb-storage` but is not yet on the server's path.
- **There is no checkpoint yet**, so the WAL is never truncated: it grows for the life of the
  database, and startup replays the whole log — meaning boot time and boot memory grow with total
  write history. Plan restarts accordingly.
- **No published performance claims.** Benchmarks exist internally, but until the two points above
  are fixed, any comparison against a disk-based engine would be measuring a different thing. We
  would rather publish nothing than publish a number we cannot defend.
- Some analytical operators still materialize more than they should; large `GROUP BY` with very
  high cardinality can exhaust the memory budget (it fails loudly and is tunable via `work_mem`,
  rather than returning a wrong answer).
- Text ordering is bytewise (C collation); `COLLATE` is rejected rather than silently ignored.

## Building from source

Requires the Rust toolchain pinned in `rust-toolchain.toml`.

```bash
cargo ck          # check the workspace
cargo test-all    # run every test
cargo lint        # clippy, warnings denied
cargo run -p nusadb-server -- --listen 127.0.0.1:5678 --data-dir ./data
cargo run -p nusadb-cli
```

## License

Apache-2.0.
