# Getting Started

From nothing to a queried table. Two installation routes are covered: the published container image,
and a build from source. Everything after that is the same either way.

- [Install](#install)
- [Connect](#connect)
- [A worked example](#a-worked-example)
- [Databases and schemas](#databases-and-schemas)
- [Loading and exporting data](#loading-and-exporting-data)
- [Running SQL from scripts](#running-sql-from-scripts)
- [Where to go next](#where-to-go-next)

---

## Install

### Container

The image carries both the server and the shell. Give it a volume: the data directory holds the
write-ahead log, which is the durable copy of your data.

```bash
docker run -d --name nusadb \
  -p 5678:5678 \
  -v nusadb-data:/var/lib/nusadb \
  -e NUSADB_USER=app \
  -e NUSADB_PASSWORD=change-me \
  nusadb/nusadb
```

Setting `NUSADB_USER` together with `NUSADB_PASSWORD` turns on SCRAM-SHA-256 for that user. Setting
only one of the pair is an error, so a half-configured container fails at start-up instead of quietly
accepting everyone.

> **Without credentials the server trusts every client.** A server started with no `--auth-user` and
> no environment pair accepts any client without a password and says so in its startup log. That is
> fine on a laptop and wrong for anything reachable by others.

### From source

A Rust toolchain is the only prerequisite; the pinned version lives in `rust-toolchain.toml` and
`rustup` installs it automatically.

```bash
git clone https://github.com/nusadb/nusadb.git
cd nusadb
cargo build --release

./target/release/nusadb-server \
  --listen 127.0.0.1:5678 \
  --data-dir ./data \
  --auth-user app:change-me
```

The data directory is created if it does not exist. `RUST_LOG=info` turns on informational logging.

---

## Connect

Defaults are host `127.0.0.1:5678`, database `nusadb`, and the bootstrap superuser `nusadb-root`.

```bash
nusadb-cli --host 127.0.0.1:5678 --user nusadb-root --database nusadb

# inside the container
docker exec -it nusadb nusadb-cli --user app --database nusadb
```

> **Clients for other databases cannot connect.** NusaDB implements its own wire protocol. A tool
> built for a different engine gets no answer to its handshake, which from its side looks like a dead
> socket. That is deliberate. Use `nusadb-cli`, one of the drivers, or write a client against the
> [protocol specification](wire-protocol.md). The SQL *dialect* is a separate matter: queries written
> for other engines largely run as-is once they arrive over a NusaDB connection.

---

## A worked example

Paste this in order. It creates two tables, loads a few rows, and answers a question with them.

```sql
CREATE TABLE customers (
  id      BIGSERIAL PRIMARY KEY,
  email   TEXT NOT NULL UNIQUE,
  country TEXT NOT NULL,
  joined  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE orders (
  id       BIGSERIAL PRIMARY KEY,
  customer BIGINT NOT NULL REFERENCES customers (id) ON DELETE CASCADE,
  total    NUMERIC(12,2) NOT NULL CHECK (total >= 0),
  placed   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX orders_customer ON orders (customer);
```

Insert, and let the server hand back what it generated:

```sql
INSERT INTO customers (email, country) VALUES
  ('ana@example.com',   'ID'),
  ('budi@example.com',  'ID'),
  ('carla@example.com', 'SG')
RETURNING id, email;
--  id |       email
-- ----+--------------------
--   1 | ana@example.com
--   2 | budi@example.com
--   3 | carla@example.com

INSERT INTO orders (customer, total) VALUES
  (1, 120.00), (1, 80.50), (2, 45.00), (3, 900.00);
```

Ask a question:

```sql
SELECT c.country,
       count(*)                                   AS orders,
       sum(o.total)                               AS revenue,
       round(avg(o.total), 2)                     AS average,
       rank() OVER (ORDER BY sum(o.total) DESC)   AS by_revenue
FROM   customers c
JOIN   orders    o ON o.customer = c.id
WHERE  o.placed >= now() - INTERVAL '30 days'
GROUP  BY c.country
ORDER  BY revenue DESC;
--  country | orders | revenue | average | by_revenue
-- ---------+--------+---------+---------+------------
--  SG      |      1 |  900.00 |  900.00 |          1
--  ID      |      3 |  245.50 |   81.83 |          2
```

Change rows safely, and keep part of the work:

```sql
BEGIN;
  UPDATE orders SET total = total * 1.1 WHERE customer = 1 RETURNING id, total;
  SAVEPOINT before_delete;
  DELETE FROM orders WHERE total < 50;
  ROLLBACK TO SAVEPOINT before_delete;    -- keep the update, undo the delete
COMMIT;
```

Constraints are enforced, and the error says which one:

```sql
INSERT INTO orders (customer, total) VALUES (999, 10);
-- ERROR 23503: foreign key violation

INSERT INTO customers (email, country) VALUES ('ana@example.com', 'ID');
-- ERROR 23505: unique constraint

INSERT INTO orders (customer, total) VALUES (1, -5);
-- ERROR 23514: check constraint
```

---

## Databases and schemas

NusaDB holds several databases, each with its own directory under `<data-dir>/base/`, and several
schemas within a database. One connection targets one database; a query cannot cross databases.

```sql
CREATE DATABASE app;              -- a separate physical database
CREATE SCHEMA tenant;             -- a namespace inside this database
CREATE TABLE tenant.t (id INT);
SELECT * FROM tenant.t;           -- resolved through search_path, then public

SET search_path = tenant, public;
```

```bash
nusadb-cli --host 127.0.0.1:5678 --user nusadb-root --database app
```

---

## Loading and exporting data

`COPY` moves rows in one streaming exchange rather than a round trip per row. In the shell forms the
data travels on the command's own standard input and output, so it composes with ordinary pipelines.

```bash
# Load: tab-delimited, \N for NULL (the server's text format)
nusadb-cli -c "COPY customers FROM STDIN" < rows.tsv

# CSV, with a header line
nusadb-cli -c "COPY customers FROM STDIN WITH (FORMAT csv, HEADER)" < rows.csv

# Export the same way
nusadb-cli -c "COPY customers TO STDOUT" > rows.tsv
```

The row count goes to standard error during an export, so it never lands in the exported file. One
redirect feeds one load: a batch holding two `COPY … FROM STDIN` statements refuses the second rather
than reporting that it loaded nothing. Typing `COPY … FROM STDIN` at the interactive prompt is
refused too, because there the keyboard is already the session's input.

After a large load, refresh the planner's statistics:

```sql
ANALYZE customers;
ANALYZE;              -- every table
```

---

## Running SQL from scripts

Both batch forms behave the same way: `-c` takes a statement and `-f` takes a file. A server error is
printed to standard error and the remaining statements still run, but the process then exits
non-zero, so a script that loads data in stages stops at the failed stage instead of continuing as
though it had worked.

```bash
set -eu
nusadb-cli -f schema.sql
nusadb-cli -c "COPY customers FROM STDIN" < customers.tsv
nusadb-cli -c "ANALYZE customers"
```

---

## Where to go next

- [SQL reference](sql-reference.md): types, statements and functions, with examples.
- [Transactions](transactions.md): isolation levels, conflicts and retries, savepoints.
- [Deployment](deployment.md): server flags, resource limits, TLS, metrics. **Read the capacity
  section before loading a large dataset:** table pages live in memory and are not evicted to disk,
  so a database's working set must fit inside the resident ceiling.
- [Wire protocol](wire-protocol.md): for writing a client.

---

## Upgrading from an older data directory

Each database directory records which storage engine wrote it. A directory written by the removed
`lsm` engine is refused at start-up rather than misread. To migrate: start the last release that
still shipped that engine, export your data (`COPY` per table), start this release with a fresh
`--data-dir`, and restore.
