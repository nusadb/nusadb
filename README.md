# NusaDB

NusaDB is a relational database engine written from scratch in Rust. It runs on a single node and
handles both transactional and analytical queries.

What it has today:

- MVCC transactions with Serializable Snapshot Isolation, plus all four standard isolation levels,
  savepoints, and rollback
- a clustered B-link/B+tree storage engine: 8 KB pages, a clock-eviction buffer pool, rows in the
  leaves keyed by an engine-minted row id, and no-wait row locks
- write-ahead logging with crash recovery (CRC32 per record, lz4-compressed); on restart the engine
  replays the durable tail of the log
- a cost-based SQL engine (parser on top of `sqlparser-rs`, analyzer, planner with histogram/MCV
  statistics and predicate/projection pushdown, and an executor)
- window functions, recursive CTEs, subqueries, set operations, `MERGE`, views, sequences, and a
  numeric/temporal/JSON/array/UUID type system with exact base-10 `NUMERIC`
- a binary wire protocol over TCP with TLS (`rustls`) and SCRAM-SHA-256, both simple and
  extended (prepared-statement) queries, and `COPY`

One design choice worth calling out: the same engine code runs on real disk in production and
against an in-memory simulation with fault injection in the tests. That means a concurrency or
crash-recovery bug can be replayed from a single seed. Alongside it there's a SQLLogicTest corpus,
isolation and crash-recovery suites, and fuzz targets.

Status: pre-1.0 (`0.1.0`). The engine, transactions, SQL, and the wire protocol all work, but the
APIs and the on-disk format may still change before 1.0.

## Build

You need the Rust toolchain pinned in `rust-toolchain.toml` (Rust 1.95.0, edition 2024).

```bash
cargo build --workspace          # or: cargo ck   (alias: cargo check)
cargo test  --workspace          # or: cargo test-all
```

Project aliases are defined in `.cargo/config.toml`.

## Run

```bash
# Start the durable server (creates --data-dir if absent)
cargo run -p nusadb-server -- --listen 127.0.0.1:5678 --data-dir ./data

# Connect with the interactive shell
cargo run -p nusadb-cli -- --host 127.0.0.1:5678
```

## Docker

Prebuilt server images are published to Docker Hub as
[`nusadb/nusadb`](https://hub.docker.com/r/nusadb/nusadb) (linux/amd64). Use `latest` to track the
current release, or pin a version tag such as `0.1.0` for a reproducible deployment.

```bash
# Start the server; the volume keeps the database across container restarts
docker run -d --name nusadb \
  -p 5678:5678 \
  -v nusadb-data:/var/lib/nusadb \
  nusadb/nusadb:latest

# Connect with the interactive shell shipped in the same image
docker exec -it nusadb nusa-cli
```

The image keeps its durable state in `/var/lib/nusadb`, so mount a volume there or the database
dies with the container. Setting both `NUSADB_USER` and `NUSADB_PASSWORD` makes the server require
SCRAM-SHA-256 for that user; with neither set it runs trust-on-startup — any client accepted, no
password — which is fine for local work only. SQL files mounted into `/docker-entrypoint-initdb.d`
run once, on the first startup against an empty data directory.

```bash
docker run -d --name nusadb \
  -p 5678:5678 \
  -v nusadb-data:/var/lib/nusadb \
  -v ./init:/docker-entrypoint-initdb.d:ro \
  -e NUSADB_USER=nusa-root \
  -e NUSADB_PASSWORD=change-me \
  -e RUST_LOG=info \
  nusadb/nusadb:latest
```

Server flags go after the image name (`... nusadb/nusadb:latest --metrics-listen 0.0.0.0:9100`).
See [`docs/deployment.md`](docs/deployment.md) for the full flag reference, TLS, and metrics.

## Architecture

NusaDB is organized as seven layers: client, protocol, SQL engine, transactions, storage, WAL, and
physical storage. The crates form a strictly layered graph where `nusadb-core` is depended on by
everything and no inner crate imports an outer one. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
crate map and design decisions.

## Contributing

Correctness matters a lot for a database, so the contribution bar is higher than a typical
application. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the rules and the review checklist.

## License

See [`LICENSE`](LICENSE).
