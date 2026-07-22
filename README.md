# NusaDB

NusaDB is a relational database engine written from scratch in Rust. It's a single-node engine
that handles both transactional and analytical work: MVCC transactions, a clustered B-link/B+tree
storage engine, write-ahead logging with crash recovery, an SQL engine, and its own binary wire
protocol.

The same engine code runs against real disk in production and against an in-memory simulation
harness with fault injection in tests, so concurrency and crash-recovery bugs can be reproduced
from a single seed.

Status: pre-1.0 (`0.1.0.Beta`). The core engine, transactions, SQL, and wire protocol work today,
but APIs and on-disk formats may still change before 1.0.

## What's inside

Storage: 8 KB pages, a buffer pool with clock eviction, and a clustered B-link/B+tree engine. Rows
live in the leaves keyed by an engine-minted row id, carry MVCC stamps, and use no-wait row locks.

Transactions: MVCC with per-row `xmin`/`xmax` and lock-free reads, Serializable Snapshot Isolation,
the four standard isolation levels, savepoints and rollback, and a background worker that purges old
row versions.

Durability: an append-only WAL (CRC32 per record, lz4-compressed) written ahead of every data-page
change. Recovery replays the durable prefix of the log.

SQL: a parser built on `sqlparser-rs`, an analyzer that resolves scope and checks types, a
cost-based planner (histogram/MCV statistics, predicate and projection pushdown, join selection),
and an executor. It covers DDL and DML, joins, subqueries, recursive CTEs, window functions, set
operations, `MERGE`, sequences, views, a numeric/temporal/JSON/array/UUID type system, and exact
base-10 `NUMERIC` arithmetic.

Wire protocol: a binary protocol over TCP with TLS (`rustls`) and SCRAM-SHA-256 auth, both a simple
and an extended (prepared-statement) query path, and `COPY`.

Testing: deterministic simulation testing, a SQLLogicTest corpus, isolation and crash-recovery
suites, and fuzz targets.

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
current release, or pin a version tag such as `0.1.0.Beta` for a reproducible deployment.

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
