# Getting Started

From nothing to a queried table. Two installation routes are covered: the published container
image, and a build from source. Everything after that is the same either way.

- [Install](#install)
- [Connect](#connect)
- [A worked example](#a-worked-example)
- [The shell, `nusadb-cli`](#the-shell-nusadb-cli)
- [Databases and schemas](#databases-and-schemas)
- [Loading and exporting data](#loading-and-exporting-data)
- [Running SQL from scripts](#running-sql-from-scripts)
- [Connecting from a program](#connecting-from-a-program)
- [Adding a user](#adding-a-user)
- [First errors and what they mean](#first-errors-and-what-they-mean)
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
`rustup` installs it automatically. The first build takes several minutes.

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
On Windows the binaries are `target\release\nusadb-server.exe` and `nusadb-cli.exe`.

---

## Connect

Defaults are host `127.0.0.1:5678`, database `nusadb`, and user `nusadb-root`, the bootstrap
superuser. With authentication on, the user must be one listed in `--auth-user`, and the password is
read from `NUSADB_PASSWORD` (preferred, so it stays out of the shell history) or `-W`.

```bash
nusadb-cli                                                  # trust-on-startup server, as nusadb-root
NUSADB_PASSWORD=change-me nusadb-cli --user app             # authenticated

# inside the container, which already has NUSADB_PASSWORD in its environment
docker exec -it nusadb nusadb-cli --user app
```

```text
nusadb-cli (NusaDB), type \q to quit
connected to 127.0.0.1:5678 as app
SELECT version();
version
------------
NusaDB 0.1.0
SELECT 1
```

> **Clients for other databases cannot connect.** NusaDB implements its own wire protocol. A tool
> built for a different engine gets no answer to its handshake, which from its side looks like a dead
> socket. That is deliberate. Use `nusadb-cli`, one of the [drivers](#connecting-from-a-program), or
> write a client against the [protocol specification](wire-protocol.md). The SQL *dialect* is a
> separate matter: queries written for other engines largely run as-is once they arrive over a
> NusaDB connection.

---

## A worked example

Paste this in order. It creates two tables, loads a few rows, and answers a question with them.

```sql
CREATE TABLE customers (
  id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email   TEXT NOT NULL UNIQUE,
  country TEXT NOT NULL,
  joined  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE orders (
  id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
-- ERROR 23503: constraint violation: insert or update on "orders" violates foreign key
--              "orders_fkey1": no matching row in "customers"

INSERT INTO customers (email, country) VALUES ('ana@example.com', 'ID');
-- ERROR 23505: constraint violation: duplicate key violates unique constraint
--              "customers_email_key" on (email)

INSERT INTO orders (customer, total) VALUES (1, -5);
-- ERROR 23514: constraint violation: new row for "orders" violates check constraint "orders_check1"
```

See how the planner will run a query:

```sql
EXPLAIN SELECT email FROM customers WHERE id = 2;
--  Project (1 column(s))
--    Filter
--      IndexScan: customers using customers_pkey
```

---

## The shell, `nusadb-cli`

```text
nusadb-cli [OPTIONS]
      --host <HOST>          Server to connect to            [default: 127.0.0.1:5678]
  -u, --user <USER>          User to connect as              [default: nusadb-root]
  -d, --database <DATABASE>  Database to open                [default: nusadb]
  -W, --password <PASSWORD>  Password; prefer NUSADB_PASSWORD in the environment
  -c, --command <COMMAND>    Run one batch of SQL (statements separated by ;) and exit
  -f, --file <FILE>          Run the SQL in a file and exit
  -F, --format <FORMAT>      aligned (default), expanded, csv, or json
      --tls                  Connect using TLS; requires --tls-ca
      --tls-ca <PATH>        PEM certificate to trust (a private CA or self-signed cert); implies --tls
      --tls-domain <NAME>    Server name to verify the certificate against (default: the host)
```

Interactively, a statement runs when a line ends with `;`, so a statement may span lines. Command
history is kept in `~/.nusadb_history`. Meta-commands, typed alone on a line:

| Command | Does |
| --- | --- |
| `\dt` or `\d` | list tables (`SHOW TABLES`) |
| `\d name` | describe a table's columns (`SHOW COLUMNS FROM name`) |
| `\l` | show the connected database |
| `\?` | help |
| `\q`, `\quit`, `quit` | exit |

Output formats, with the same query:

```bash
nusadb-cli -F csv  -c "SELECT id, email FROM customers ORDER BY id LIMIT 2"
# id,email
# 1,ana@example.com
# 2,budi@example.com

nusadb-cli -F json -c "SELECT id, email FROM customers ORDER BY id LIMIT 2"
# [{"id":"1","email":"ana@example.com"},{"id":"2","email":"budi@example.com"}]

nusadb-cli -F expanded -c "SELECT id, email FROM customers ORDER BY id LIMIT 1"
# -[ RECORD 1 ]-
# id    | 1
# email | ana@example.com
```

Values in the JSON format are strings, because the shell prints the protocol's text form. The
command tag (`SELECT 2`, `INSERT 1`) is printed after aligned and expanded results and omitted for
CSV and JSON so the output can be fed to another program.

To run the same statement with changing values, use `PREPARE` / `EXECUTE`; the shell has no other
parameter binding. Routine bodies in `$$ ... $$` pass through the shell intact, so procedures and
functions can be created from here as well as from a driver.

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
SELECT * FROM nusadb_databases;   -- what exists (superuser)
```

```bash
nusadb-cli --database app
```

---

## Loading and exporting data

`COPY` moves rows in one streaming exchange rather than a round trip per row. In the shell forms the
data travels on the command's own standard input and output, so it composes with ordinary pipelines.

```bash
# Load: tab-delimited, \N for NULL (the server's text format)
nusadb-cli -c "COPY customers FROM STDIN" < rows.tsv

# CSV with a header line, into named columns
nusadb-cli -c "COPY customers (email, country) FROM STDIN WITH (FORMAT csv, HEADER)" < rows.csv

# Export the same way
nusadb-cli -c "COPY customers TO STDOUT" > rows.tsv
nusadb-cli -c "COPY customers TO STDOUT WITH (FORMAT csv, HEADER)" > rows.csv
```

The row count goes to standard error during an export, so it never lands in the exported file. One
redirect feeds one load: a batch holding two `COPY ... FROM STDIN` statements refuses the second
rather than reporting that it loaded nothing. Typing `COPY ... FROM STDIN` at the interactive prompt
is refused too, because there the keyboard is already the session's input.

After a large load, refresh the planner's statistics (a background worker also does this, on a
timer):

```sql
ANALYZE customers;
ANALYZE;              -- every table
```

Before loading a large dataset, read the capacity section of [deployment](deployment.md): table
pages live in memory and a dataset larger than the resident ceiling is refused mid-load rather than
loaded slowly.

---

## Running SQL from scripts

Both batch forms behave the same way: `-c` takes a statement list and `-f` takes a file. A server
error is printed to standard error and the remaining statements still run, but the process then
exits non-zero, so a script that loads data in stages stops at the failed stage instead of continuing
as though it had worked.

```bash
set -eu
nusadb-cli -f schema.sql
nusadb-cli -c "COPY customers FROM STDIN" < customers.tsv
nusadb-cli -c "ANALYZE customers"
```

A file saved with a byte-order mark (an editor on Windows often adds one) is accepted; the mark is
dropped before parsing.

---

## Connecting from a program

Every driver speaks the wire protocol directly, has no native dependencies, and binds parameters
as `$1`, `$2`, ... The result comes back typed (an `INT` column is an integer in the host language,
`NUMERIC` a decimal, `TIMESTAMPTZ` a datetime).

| Language | Install | Package |
| --- | --- | --- |
| Python | `pip install nusadb` | DB-API 2.0 driver, with a SQLAlchemy dialect |
| Node.js | `npm install nusadb` | Promise-based client with TypeScript typings |
| Go | `go get github.com/nusadb/go` | `database/sql` driver, registered as `nusadb` |
| Rust | `cargo add nusadb` | sync and async (Tokio) client with a connection pool; `tls` feature |
| Ruby | `gem install nusadb` | native client |
| PHP | `composer require nusadb/nusadb` | PDO-style client, pure PHP |
| Java | `com.nusadb:nusadb-jdbc:0.1.0` on Maven Central | JDBC driver; Hibernate dialects ship as `nusadb-hibernate5` / `nusadb-hibernate6` |
| .NET | in the source tree | ADO.NET provider; not yet on NuGet |

Python:

```python
import nusadb

conn = nusadb.connect(host="127.0.0.1", port=5678, user="app", password="change-me",
                      database="nusadb")
cur = conn.cursor()
cur.execute("INSERT INTO customers (email, country) VALUES ($1, $2) RETURNING id",
            ["dewi@example.com", "ID"])
print(cur.fetchone())                       # (4,)
cur.execute("SELECT email FROM customers WHERE country = $1", ["ID"])
for (email,) in cur:
    print(email)
conn.close()
```

Node.js:

```js
const { connect } = require('nusadb');

const conn = await connect({ host: '127.0.0.1', port: 5678, user: 'app',
                             password: 'change-me', database: 'nusadb' });
const res = await conn.query('SELECT id, email FROM customers WHERE country = $1', ['ID']);
console.log(res.rows);          // [[1, 'ana@example.com'], [2, 'budi@example.com']]
await conn.close();
```

Go:

```go
import (
    "database/sql"
    _ "github.com/nusadb/go"
)

db, err := sql.Open("nusadb", "nusadb://app:change-me@127.0.0.1:5678/nusadb")
var email string
err = db.QueryRow("SELECT email FROM customers WHERE id = $1", 1).Scan(&email)
```

Every driver reports a failed statement with its SQLSTATE; see
[transactions](transactions.md) for the retry loop that every write path needs.

---

## Adding a user

Two lists are involved. The server's `--auth-user` list (or `NUSADB_USER` / `NUSADB_PASSWORD`)
decides who may connect and with which password; SQL roles inside the database decide what a
connected user may do. Add the user to both.

```bash
nusadb-server --data-dir ./data \
  --auth-user nusadb-root:root-secret \
  --auth-user app:change-me \
  --auth-user analyst:analyst-secret
```

```sql
-- connected as nusadb-root
CREATE ROLE analyst LOGIN;
GRANT SELECT ON customers, orders TO analyst;
```

```bash
NUSADB_PASSWORD=analyst-secret nusadb-cli --user analyst -c "SELECT count(*) FROM orders"
```

A user that authenticates but has no role can create its own tables and nothing else. `PASSWORD`
inside `CREATE ROLE` is refused, because a password set there would never be checked at login.
Once any `--auth-user` is set, `nusadb-root` must be listed too or it cannot connect.

---

## First errors and what they mean

| What you see | Why | What to do |
| --- | --- | --- |
| the client hangs, then times out | a client for another database is talking to NusaDB's port | use `nusadb-cli` or a NusaDB driver |
| `authentication failed` | wrong password, or a user not in `--auth-user` | check both; the message is the same on purpose |
| `ERROR 42501: permission denied: SELECT on table ...` | the role has no grant on the table | `GRANT ... TO role` as the owner or `nusadb-root` |
| `ERROR 40001: ...` | two transactions wrote the same row; locks never wait | run the whole transaction again ([transactions](transactions.md)) |
| `ERROR 25P02: current transaction is aborted` | an earlier statement in the transaction failed | `ROLLBACK` (or `ROLLBACK TO SAVEPOINT`) and start over |
| `ERROR 42804: type mismatch ...` | NusaDB does not coerce between unrelated types | cast explicitly: `value::BIGINT` |
| `ERROR 0A000: unsupported SQL construct: ...` | recognised but not built | the message names the construct; see the reference's "Not accepted" list |
| `ERROR XX000: out of memory: the in-memory store reached its resident-memory limit ...` | the data no longer fits the memory ceiling | raise `--max-resident-bytes`, use a larger host, or free rows |
| `COPY ... FROM STDIN` refused at the prompt | the keyboard is already the session's input | use `nusadb-cli -c "COPY ..." < file` |

---

## Where to go next

- [SQL reference](sql-reference.md): every type, statement and function, with examples, and the
  behaviours where NusaDB chooses differently.
- [Transactions](transactions.md): isolation levels, conflicts and retries, savepoints, locks.
- [Deployment](deployment.md): every server flag, memory limits, authentication, TLS, systemd,
  metrics, backup. **Read the capacity section before loading a large dataset.**
- [Wire protocol](wire-protocol.md): for writing a client.

---

## Upgrading from an older data directory

Each database directory records which storage engine wrote it. A directory written by the removed
`lsm` engine is refused at start-up rather than misread. To migrate: start the last release that
still shipped that engine, export your data (`COPY` per table), start this release with a fresh
`--data-dir`, and restore.
