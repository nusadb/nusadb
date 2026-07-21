# Getting Started

> TODO: this page is a placeholder until Stage 4 makes NusaDB runnable end-to-end.

## Build

```bash
git clone https://github.com/nusadb/nusadb.git
cd nusadb
cargo build --release
```

## Connect

The stock defaults are host `127.0.0.1:5678`, database `nusadb`, and the bootstrap
superuser `nusa-root` (password `nusa-root`). A trust-on-startup server ignores the
password; a server started with `--auth-user nusa-root:nusa-root` (or the
`NUSADB_USER`/`NUSADB_PASSWORD` env pair) requires it.

```bash
./target/release/nusa-cli --host 127.0.0.1:5678 --user nusa-root --database nusa
```

## Multiple databases & schemas

NusaDB is multi-database (physical, one data dir per database under
`<data-dir>/base/<db>/`) and multi-schema. From any session:

```sql
CREATE DATABASE app;                 -- a new physical database
CREATE SCHEMA tenant;                -- a namespace within the current database
CREATE TABLE tenant.t (id INT);      -- schema-qualified
SELECT * FROM tenant.t;              -- resolved via search_path, falling back to public
```

Connect to a specific database by name (each connection targets one database):

```bash
./target/release/nusa-cli --host 127.0.0.1:5678 --user nusa-root --database app
```

## Breaking change: the `lsm` storage engine was removed (2026-07-09)

The clustered B-link/B+tree engine (`btree`) is now NusaDB's only storage engine.
`--storage-engine btree` remains accepted (it is the default); `--storage-engine lsm`
is rejected at startup.

A data directory written by the removed `lsm` engine cannot be opened by this release —
the server refuses it loudly rather than misreading the files. To migrate:

1. Start the **last release that still ships the lsm engine** over the old data dir.
2. Dump your data over the wire (e.g. `SELECT`/`COPY` per table, or your own export).
3. Start this release with a **fresh** `--data-dir` (it initializes as `btree`).
4. Restore the dump.

Each database directory records its engine in an `engine` marker file, so a mixed or
stale directory is always detected instead of silently corrupted.
