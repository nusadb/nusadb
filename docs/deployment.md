# Deploying NusaDB

NusaDB ships as a single server binary (`nusadb-server`) plus an interactive client (`nusadb-cli`).
The server is configured entirely by command-line flags; there is no configuration file. The
durable state is the write-ahead log under `--data-dir`: back that path up and you have backed up
the database.

- [Server flags](#server-flags)
- [Authentication and users](#authentication-and-users)
- [Memory limits and capacity](#memory-limits-and-capacity)
- [Running the container image](#running-the-container-image)
- [Running on a Linux VM with systemd](#running-on-a-linux-vm-with-systemd)
- [TLS](#tls)
- [Metrics](#metrics)
- [Checkpoints, backup and restore](#checkpoints-backup-and-restore)
- [Upgrades](#upgrades)

---

## Server flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--listen` | `0.0.0.0:5678` | TCP listen address for the wire protocol. |
| `--data-dir` | `./data` | Durable data directory: the write-ahead log, checkpoint image and page segments of every database, under `base/<name>/`. Created if absent. One server at a time: a second server started on a directory in use exits at once, saying so. |
| `--auth-user USER:PASSWORD` | none | Require SCRAM-SHA-256 for this user. Repeatable. Once **any** is set, every connection must authenticate; with none, the server trusts every client and logs a warning. |
| `NUSADB_USER` + `NUSADB_PASSWORD` (environment) | none | The same as one `--auth-user`, for containers. Setting only one of the pair is a start-up error. |
| `--tls-cert` / `--tls-key` | none | PEM certificate chain and private key; TLS is on when both are set, on the same port. |
| `--tls-client-ca` | none | PEM CA for mutual TLS: every client must present a certificate signed by it. |
| `--max-connections` | `25` | Cap on concurrent connections; excess connections wait for a slot. `0` is unlimited. |
| `--reject-excess-connections` | off | Refuse a connection past the cap at once with SQLSTATE `53300` instead of queueing it, so a pool can back off. |
| `--idle-timeout` | `0` | Close a connection idle this many seconds. `0` is no limit. |
| `--handshake-timeout` | `60` | Drop a connection that has not finished the start-up and authentication exchange within this many seconds, so a stalled client cannot hold a slot. |
| `--statement-timeout` | `0` | Cancel any statement running longer than this many seconds (`57014`). Sessions can lower it with `SET statement_timeout`. |
| `--drain-timeout` | `30` | On Ctrl-C or SIGTERM (the signal `docker stop` and Kubernetes send), wait this long for in-flight connections to finish before aborting them, then exit cleanly. `0` waits indefinitely. |
| `--mem-budget` | `0` | Total memory budget in bytes. `0` auto-detects on Linux as the smaller of host RAM and the cgroup limit, so a container limit is honoured; on other systems `0` means no budget. The budget derives the limits below. |
| `--max-resident-bytes` | derived | Bound on each database's page cache: clean pages are evicted and changed pages spill past it, and an insert or update is refused only once what cannot leave memory reaches it (see [Table data](#table-data-a-page-cache-over-the-checkpoint-image)). Derived from the memory budget (floor 256 MiB); unlimited when no budget is known. |
| `--work-mem` | `0` | Per-query memory for one sort, aggregate or join stage. Past it a stage spills (with `--spill-dir`) or fails with an error naming the limit. `0` is unlimited unless a budget derives a value. |
| `--spill-dir` | none | Directory for transient spill files. Sorts, `DISTINCT`, `DISTINCT ON`, `GROUP BY` (grouping sets included), window functions, set operations and hash joins over `--work-mem` stream to it instead of failing. Stale files from a crash are removed at start-up. |
| `--maintenance-work-mem` | `0` | Bytes of index entries a `CREATE INDEX` buffers before flushing a sorted batch. `0` uses a built-in bound. |
| `--max-txn-write-bytes` | `0` | Ceiling on the uncommitted writes one transaction may buffer; past it the transaction fails with `XX000` instead of growing until the host kills the process. `0` derives 25% of the budget (floor 128 MiB). |
| `--copy-max-bytes` | derived | Ceiling on one `COPY ... FROM STDIN`. Derived as about 20% of the budget, capped at 1 GiB; `0` is unbounded. |
| `--autoanalyze-interval` | `60` | Seconds between sweeps of the background worker that re-analyzes tables whose statistics went stale. `0` disables it. |
| `--autoanalyze-scale` / `--autoanalyze-threshold` | `0.1` / `50` | A table is re-analyzed once its writes since the last analyze exceed `threshold + scale * rows`. |
| `--checkpoint-threshold-bytes` | `67108864` (64 MiB) | Log length past which the background checkpoint worker folds a database's log into a fresh image and truncates it. `0` disables the worker. |
| `--checkpoint-interval` | `5` | Seconds between the checkpoint worker's checks of each database's log. `0` disables the worker. |
| `--wal-archive-dir` | none | Archive every checkpoint's log segment and image under this directory, one subdirectory per database, for point-in-time recovery. |
| `--restore-database` / `--restore-to-time` / `--restore-to-lsn` / `--restore-live-log` | none | Offline: rebuild one database from its archive as of a moment or a log position, then exit. See [Point-in-time recovery](#point-in-time-recovery). |
| `--checkpoint-max-pause` | `2` | Once the log is past its threshold and three checks in a row found transactions active, hold new transactions for at most this many seconds (capped at 60) so the checkpoint can run. `0` never pauses. |
| `--metrics-listen` | none | Serve Prometheus metrics on this address, for example `127.0.0.1:9100`. |
| `--storage-engine` | `btree` | The only value. A data directory written by the removed `lsm` engine is refused with a migration hint. |

`RUST_LOG` sets log verbosity (`tracing` filter syntax, for example `RUST_LOG=info` or
`RUST_LOG=nusadb_wire=debug`). `NUSADB_DISABLE_SIMD=1` forces the portable executor path on a host
where the AVX2 path is suspect; results are identical either way. On a CPU without AVX2 the
fallback is automatic.

> **Production checklist.** Set at least one `--auth-user` (or the environment pair); terminate TLS
> with `--tls-cert` / `--tls-key`; keep `--data-dir` on a persistent, backed-up volume; keep the
> metrics port private; and read [Memory limits and capacity](#memory-limits-and-capacity) before
> loading a large dataset.

---

## Authentication and users

Authentication and authorization are two separate lists.

- **Who may connect** is the `--auth-user` list (or `NUSADB_USER` / `NUSADB_PASSWORD`). Each entry
  is a user name and a password verified with SCRAM-SHA-256; the password never crosses the wire.
  A wrong password and an unknown user return the same `authentication failed`, so the error does
  not reveal which names exist. Changing a password means restarting the server with the new
  `--auth-user` value.
- **What a connected user may do** is decided by SQL roles and grants inside each database. Create a
  role with the same name as the `--auth-user` entry and grant it what it needs; `PASSWORD` in
  `CREATE ROLE` is refused because it would never be checked. A user with no matching role can create
  its own tables and read nothing else.

```bash
nusadb-server --data-dir /var/lib/nusadb \
  --auth-user nusadb-root:ROOT_SECRET \
  --auth-user app:APP_SECRET
```

```sql
-- as nusadb-root
CREATE ROLE app LOGIN;
GRANT SELECT, INSERT, UPDATE, DELETE ON orders TO app;
```

```bash
NUSADB_PASSWORD=APP_SECRET nusadb-cli --user app
```

`nusadb-root` is the bootstrap superuser and bypasses every grant; list it in `--auth-user` with a
strong password, since with authentication on it cannot connect otherwise. Without any
`--auth-user` the server runs trust-on-startup: any name is accepted with no password. That is fine
on a laptop and wrong for anything reachable by others; the start-up log says so in capitals.

---

## Memory limits and capacity

NusaDB defaults small and scales up explicitly: a fresh install stays healthy on a host with about
2 GB of RAM and one or two cores, and a larger machine raises the limits on purpose.

### Table data: a page cache over the checkpoint image

Table pages live in a page cache backed by the last checkpoint image. The image
(`btree.wal.ckpt`) names every page of the database and where it lies in the page segments of
the pages directory beside it (`btree.wal.pages/`); a page is read from there on first use, so a
restart does not load the whole database before serving, and a page that has not changed since
the last checkpoint (a clean page) can leave the cache again when memory is needed. A page changed since the last checkpoint (a
dirty page) is not in the image yet. When the cache holds nothing clean to evict, a dirty page is
written to a scratch file beside the log (`btree.wal.spill`) and leaves memory; it is read back
from there when needed. The scratch file is not a durable copy: recovery never reads it, and it is
emptied at open (on Linux and macOS it is unlinked as soon as it is created, so it does not appear
in the directory). The write-ahead log and the image remain the only durable copies of the data,
and the next checkpoint writes every changed page into the new image and empties the scratch file.

`--max-resident-bytes` (derived from the memory budget when unset) bounds the cache. Clean pages
are evicted first, then dirty pages spill, so the pages changed between checkpoints are bounded
by disk rather than memory. The scratch file grows to at most the pages changed since the last
checkpoint, which the checkpoint threshold keeps in proportion to the log. What still counts
against the bound is the few index entries too large for an index page, and notes of index
entries awaiting purge (see below); once they reach it, the next insert or update is refused before it starts (a write already under way
always completes) with an error that names the limit and the bytes held:

```text
ERROR XX000: out of memory: the engine reached its resident-memory limit of 858993440 bytes
(859001088 bytes held by index entries); free rows (DELETE/TRUNCATE), drop indexes that are not
needed, raise the limit, or use a larger host
```

Reads keep working at the bound: a page loaded for a read may briefly overshoot it and is the first
to leave again. If the scratch file cannot be written (a full disk), the page stays in memory and
the cache grows past the bound until the next checkpoint instead of failing the write.

`DELETE`, `TRUNCATE`, `CREATE INDEX` and the background purge are not refused at the bound, so
space can always be freed; a large `DELETE` is still capped per transaction by
`--max-txn-write-bytes`.

A table scan reads the table a batch of rows at a time rather than loading it whole, so a
full scan of a table larger than memory stays within the cache. A scan still open when its
transaction commits or rolls back (a cursor left open) is read to its end at that moment.

B-tree index entries, the primary key's included, live in pages too: each index is a tree of
pages ordered by key, plus a map from row to key, both in the page cache and carried by the
image like table pages, so a restart does not rebuild them. An entry too large for an index page
(a key of roughly 2 KB or more) is kept in memory instead, carried by the image as a record, and
counts against the bound. So does a small note per index entry an `UPDATE` moved to a new key,
kept until the background purge removes the old entry; a long-running transaction delays that
purge, so heavy key-changing updates under one can grow it. Writing through an index costs a little more than it did when indexes
lived in memory: a bulk load into a table with a primary key and one more index runs about a
fifth slower, while lookups are as fast. Vector indexes (`USING hnsw`) are held in memory by the
SQL layer whatever the bound, and reloaded from their saved graphs at open.

A checkpoint writes only the pages changed since the one before it, into a new segment, and a
new image that names the older segments for every other page. Its cost grows with the changes,
plus the image's directory of pages (20 bytes per page). Once the segments
an image still names would hold more than twice the live pages (plus a fixed slack of 1024
pages), or more than 32 segments, the
checkpoint writes every page afresh into one segment instead, so dead pages on disk stay bounded.

Deleting rows frees pages for reuse and lowers the count the next image carries; page memory
within a running process is recycled through the cache rather than returned to the OS.

### Row size

A row is stored inline in its 8 KB B-tree leaf while it fits (about 8 KB encoded); a larger row
is spread over a chain of overflow pages and the leaf keeps a small stub, so a large `text`,
`json`, `bytea` or array value is stored whole. Reading such a row reassembles it from its chain,
which costs one page read per 8 KB of the value, and a version-only rewrite (a delete stamp, a
purge pass) touches the stub alone. A row's encoded size is capped at 32 MiB; a larger one is
refused with an error naming both sizes, never truncated:

```text
ERROR XX000: tuple of 40000000 bytes exceeds the maximum row size of 33554432 bytes
```

Overflow pages count toward the resident-memory ceiling like any other page.

### One query, one transaction, one load

The other three limits bound what a single client can do to the server:

| Limit | Flag | Past it |
| --- | --- | --- |
| one executor stage (sort, aggregate, join) | `--work-mem` | spills to `--spill-dir` for sorts, `DISTINCT`, `DISTINCT ON`, `GROUP BY` (grouping sets included), window functions, set operations and hash joins; otherwise fails with a message naming the limit and the flag |
| one transaction's uncommitted writes | `--max-txn-write-bytes` | the transaction fails with `XX000` |
| one `COPY ... FROM STDIN` | `--copy-max-bytes` | the load is aborted; split it or raise the flag |

A spilled sort, `DISTINCT` or `GROUP BY` streams its result to the client straight from the merge of
its spill files, so a result larger than `--work-mem` is fine. Window functions spill too: each
partition is evaluated in memory when it fits and from its spill file when it does not. What has to
fit is one frame: a frame that holds more rows than the budget (say
`ROWS BETWEEN 100000 PRECEDING AND CURRENT ROW`, or a floating-point `sum` or `avg` from
`CURRENT ROW` to `UNBOUNDED FOLLOWING`) fails at the budget, as do `lag`/`lead` with an offset that
changes from row to row and `nth_value` with such a position over a `ROWS` or default frame; split
such a partition with `PARTITION BY`. `DISTINCT ON` and `ROLLUP` / `CUBE` / `GROUPING SETS` spill as
well. `CREATE TABLE AS`, `CREATE MATERIALIZED VIEW` and `INSERT ... SELECT` stream the query's rows
into the table as they come, so a large result does not have to fit either (`INSERT ... SELECT`
falls back to holding the result when the query reads the target table, has a subquery, or the
target has triggers, foreign keys or incrementally maintained views). `REFRESH MATERIALIZED VIEW`
keeps the new rows in a spill file until the old ones are replaced. A cursor over a `SELECT` whose
result is larger than `--work-mem` keeps its rows in a spill file, which `FETCH` reads by position
in any direction and `CLOSE` removes. `UPDATE ... FROM` and `DELETE ... USING` over a subquery
still hold the subquery's whole result. A failed query leaves the server responsive, which is the point. A session's
`SET work_mem` moves both the budget and the point where spilling starts.

On Linux all of these derive from `--mem-budget`, which auto-detects the host or container limit, so
a container with a memory limit gets sensible ceilings with no flags. On other systems set
`--mem-budget` (or the individual flags) explicitly; without it the derived limits are unlimited.

### CPU compatibility

The executor uses AVX2 where the CPU has it and falls back to a portable path otherwise (older x86,
some budget VPS, ARM). The engine never emits an illegal instruction; only throughput differs.

---

## Running the container image

The image on Docker Hub, `nusadb/nusadb`, carries both `nusadb-server` and `nusadb-cli`. Use
`latest` or pin a version tag such as `0.1.0`. The durable state lives in `/var/lib/nusadb`; mount a
volume there or the database dies with the container.

```bash
docker run -d --name nusadb \
  -p 5678:5678 \
  -v nusadb-data:/var/lib/nusadb \
  -e NUSADB_USER=app \
  -e NUSADB_PASSWORD=change-me \
  -e RUST_LOG=info \
  nusadb/nusadb:latest
```

- Server flags go after the image name and replace the default
  `--listen 0.0.0.0:5678 --data-dir /var/lib/nusadb`, so repeat those two when you add others:
  `... nusadb/nusadb:latest --listen 0.0.0.0:5678 --data-dir /var/lib/nusadb --metrics-listen 0.0.0.0:9100`.
- `.sql` files mounted into `/docker-entrypoint-initdb.d` run once, in name order, on the first
  start against an empty data directory: the place for `CREATE ROLE`, `GRANT` and schema.
- The container honours its memory limit (`--memory=2g`) through the auto-detected budget.
- Connect from inside: `docker exec -it nusadb nusadb-cli --user app`. The password is read from
  `NUSADB_PASSWORD`, which the container already has.
- The image exposes `5678` and `9100`; publish `9100` only on a private interface.

```bash
docker run -d --name nusadb \
  -p 5678:5678 \
  -v nusadb-data:/var/lib/nusadb \
  -v ./init:/docker-entrypoint-initdb.d:ro \
  -e NUSADB_USER=nusadb-root \
  -e NUSADB_PASSWORD=change-me \
  --memory=2g \
  nusadb/nusadb:latest
```

---

## Running on a Linux VM with systemd

### 1. Get the binary

Build from source on the VM (the toolchain pinned in `rust-toolchain.toml` is installed by
`rustup` automatically), or download a release tarball for your platform when one is published for
the version you want.

```bash
cargo build --release --locked -p nusadb-server -p nusadb-cli
sudo install -m 0755 target/release/nusadb-server /usr/local/bin/
sudo install -m 0755 target/release/nusadb-cli    /usr/local/bin/
```

### 2. Create a service user and data directory

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin nusadb
sudo mkdir -p /var/lib/nusadb /etc/nusadb
sudo chown -R nusadb:nusadb /var/lib/nusadb
# Place server.crt / server.key under /etc/nusadb (root-owned, readable by nusadb).
sudo chown root:nusadb /etc/nusadb/server.* && sudo chmod 0640 /etc/nusadb/server.*
```

Put the credentials in an environment file readable only by root rather than in the unit:

```bash
sudo tee /etc/nusadb/nusadb.env >/dev/null <<'EOF'
NUSADB_USER=nusadb-root
NUSADB_PASSWORD=STRONG_PASSWORD
RUST_LOG=info
EOF
sudo chmod 0600 /etc/nusadb/nusadb.env
```

### 3. systemd unit

Create `/etc/systemd/system/nusadb.service`:

```ini
[Unit]
Description=NusaDB server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nusadb
Group=nusadb
EnvironmentFile=/etc/nusadb/nusadb.env
ExecStart=/usr/local/bin/nusadb-server \
  --listen 0.0.0.0:5678 \
  --data-dir /var/lib/nusadb \
  --tls-cert /etc/nusadb/server.crt \
  --tls-key /etc/nusadb/server.key \
  --spill-dir /var/lib/nusadb/spill \
  --metrics-listen 127.0.0.1:9100 \
  --max-connections 100
Restart=on-failure
RestartSec=2

# The server only needs its data directory writable.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/nusadb

[Install]
WantedBy=multi-user.target
```

### 4. Start and verify

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nusadb
sudo systemctl status nusadb
journalctl -u nusadb -f          # follow logs

# From a client host, after opening the VM firewall to 5678:
NUSADB_PASSWORD=STRONG_PASSWORD nusadb-cli --host VM_HOST:5678 --user nusadb-root \
  --tls --tls-ca /path/to/ca.crt
```

### 5. Firewall

Expose the wire port to your clients; keep metrics on localhost.

```bash
sudo ufw allow 5678/tcp
```

---

## TLS

TLS runs on the same port: when `--tls-cert` and `--tls-key` are set the server offers TLS and a
plaintext client is refused. Clients are told what to trust, because no system trust store is
consulted:

```bash
nusadb-cli --host db.internal:5678 --user app --tls --tls-ca /etc/nusadb/ca.crt
nusadb-cli --host 10.0.0.5:5678 --tls --tls-ca ca.crt --tls-domain db.internal   # name to verify
```

A certificate generated with the usual self-signed one-liner is marked as a certificate authority,
and presenting it as a server certificate is rejected. Either create a small private CA and sign a
server certificate with it, or generate a leaf certificate that is not marked as a CA. Host-name
verification is applied and a mismatch reports which names the certificate covers.

For mutual TLS add `--tls-client-ca`; every client must then present a certificate signed by that
CA, in addition to SCRAM authentication.

---

## Metrics

With `--metrics-listen` set, the server answers Prometheus scrapes in the text exposition format.
The endpoint is unauthenticated, so bind it to a private address.

| Metric | Type | Meaning |
| --- | --- | --- |
| `nusadb_connections_total` | counter | connections accepted since start |
| `nusadb_connections_active` | gauge | connections currently open |
| `nusadb_queries_total` | counter | statements executed |
| `nusadb_query_errors_total` | counter | statements that returned an error |
| `nusadb_database_stopped{database="..."}` | gauge | `1` once that database has stopped after a storage error, `0` while it serves; listed for each database opened since start |

Alert on `nusadb_database_stopped == 1`: a stopped database refuses every statement until the
server restarts (see "When a storage error interrupts a change" under
[Checkpoints, backup and restore](#checkpoints-backup-and-restore)).
The rest is enough to see whether the server is up and busy, and not enough for latency analysis:
there are no duration histograms, per-database query counters, or other storage metrics yet, and
no serialization-conflict counter, so track `40001` retries from the application side.

---

## Checkpoints, backup and restore

The write-ahead log is the durable copy of the data. A checkpoint folds the in-memory state into an
image file beside the log and truncates the log, so the data directory holds live data plus the
write history since the last checkpoint, and recovery replays only that tail. A checkpoint is taken
automatically when a database is opened with a log past a few megabytes.

While the server runs, a background worker per database checks the log every
`--checkpoint-interval` seconds and, once it is longer than `--checkpoint-threshold-bytes`
(64 MiB by default), takes the same checkpoint `CHECKPOINT` would. That checkpoint needs a
moment with no transaction active. With a single writer or a bursty load such moments are
frequent and the worker simply waits for one. Under continuously overlapping transactions from
many connections they can be rare, so after three consecutive refusals the worker makes one: it
holds new transactions for at most `--checkpoint-max-pause` seconds (2 by default), lets the
running ones end, checkpoints, and resumes. Clients see a short wait on their next `BEGIN` or
autocommit statement, never an error. A transaction held open longer than the pause budget, such
as an idle client inside `BEGIN`, defeats the pause: the worker logs a warning with the active
count, doubles the number of busy checks it waits before pausing again (up to about sixteen
minutes between attempts), and the log keeps growing until that transaction ends. The pause is
capped at 60 seconds. Each checkpoint writes the pages changed since the last one and the image's
page directory while the engine is paused, and now and then every page (see the page cache
section above); on a large database, raise the threshold so the pause is paid less often. Watch the server log at `info` for
`runtime checkpoint folded the log` and at `debug` for the busy retries. Set either flag to `0` to
turn the worker off and instead issue `CHECKPOINT` from a cron job over an otherwise idle
connection:

```bash
NUSADB_PASSWORD=... nusadb-cli --user nusadb-root -c "CHECKPOINT"
```

It requires a quiesced engine: it refuses, naming how many transactions are still active, while any
transaction is open, including one on the connection issuing it. Run it from a connection in
autocommit at a quiet moment; a load with continuously overlapping transactions may need a retry.

**When a storage error interrupts a change.** If reading or writing a page fails while a row,
an index entry, a rollback or the background purge is changing it (a damaged page on disk, an
unreadable page of the scratch file), or the log cannot take the records a `ROLLBACK TO
SAVEPOINT` must write (a full disk), the database stops: from then on every statement against it,
new transactions included, fails with

```text
ERROR XX000: i/o error: nusadb-btree: the database stopped after a storage error interrupted a
change (<the error>); restart it to recover from its log
```

and the server log holds the same error at `ERROR`; the metrics endpoint reports it as
`nusadb_database_stopped{database="<name>"} 1`. Memory may then disagree with the log, so
the stopped database never checkpoints: its image stays exactly as the last good checkpoint left
it. Other databases on the same server keep running. Fix the cause (check the disk, restore the
damaged files from a backup), then restart the server: the database is rebuilt from its last
image and its log, with every committed transaction and nothing of the interrupted one. An error
while a statement only reads a page fails that statement alone and stops nothing; the background
purge, though, walks every table and changes rows as it goes, so a damaged page it meets stops
the database within one purge interval even if no statement touches that page.

**One server per data directory.** A running server holds an exclusive lock on
`global/cluster.lock`, and each open database one on `base/<db>/btree.wal.lock` (on a data
directory of the older single-database layout, `nusadb.wal.lock` at its root). A second server
started on the same directory exits at once with a message naming the lock; a restore or a
standby seed into a database that is open is refused, and so is `DROP DATABASE` of a database
another process has open. The operating system releases the locks when the holder ends, even
when it is killed, so there is never a stale lock to remove by hand. The lock files themselves
stay; they are empty and harmless. Keep the data directory on a local file system: on some
network file systems these locks are not enforced across machines.

**Backup.** A checkpoint image together with the page segments it names is a complete copy of
one database as of its checkpoint. The engine only ever replaces the image by an atomic rename
and never rewrites a segment, so a copy of both is a consistent point-in-time backup even while
the server keeps writing. Take the image first, then the pages directory. The segments of an
image stay on disk until the checkpoint after the one that replaces it (a restart removes them
sooner: at open only the published image's segments are kept), so the safe way is a hard-link
snapshot on the same file system, which takes milliseconds, and then a copy of the snapshot
wherever it should go. Per database:

```bash
NUSADB_PASSWORD=... nusadb-cli --user nusadb-root -d shop -c "CHECKPOINT"
snap=/data/snapshots/shop-$(date +%F)       # on the same file system as $DATA_DIR
mkdir -p "$snap/btree.wal.pages"
ln "$DATA_DIR/base/shop/btree.wal.ckpt" "$snap/btree.wal.ckpt"
ln "$DATA_DIR/base/shop/btree.wal.pages/"*.seg "$snap/btree.wal.pages/"
cp -r "$snap" /backups/                     # then copy it anywhere
```

A copy that misses a segment its image names is refused when opened, naming the segment, never
read with pages missing.

The backup holds every transaction committed before the `CHECKPOINT`; what commits afterwards is
in the log tail only. The background checkpoint worker refreshes the image on its own as the log
grows, so a copy taken without an explicit `CHECKPOINT` is still consistent, just older.

**Restore.** The database must be registered in the cluster (it is, if it was created there; in a
fresh data directory run `CREATE DATABASE shop` first), and its directory must hold no log
(`btree.wal`): with the server stopped, remove that database's log if one exists, place the copy,
and start the server. It opens the image with an empty log tail.

```bash
rm -rf "$DATA_DIR/base/shop/btree.wal" "$DATA_DIR/base/shop/btree.wal.ckpt" \
  "$DATA_DIR/base/shop/btree.wal.pages"
cp -r /backups/shop-2026-09-26/btree.wal.pages "$DATA_DIR/base/shop/"
cp /backups/shop-2026-09-26/btree.wal.ckpt "$DATA_DIR/base/shop/btree.wal.ckpt"
nusadb-server --data-dir "$DATA_DIR"
```

A `btree.wal` left in place would be replayed on top of the image, which is not a restore. For a
whole-cluster copy with the server stopped, archive the `--data-dir` tree and extract it in place.
Logical export with `COPY table TO STDOUT` and reload with `COPY table FROM STDIN` remains
available. There is no built-in scheduled backup or replication.

### Point-in-time recovery

With `--wal-archive-dir DIR`, every checkpoint keeps two files under `DIR/<database>/` before it
truncates the log: `<lsn>.log`, the log segment it folded (every record up to log position
`<lsn>`), and `<lsn>.ckpt`, the image it published, with the page segments that image names
under `DIR/<database>/pages/` and their names, one per line, in `<lsn>.segments`; files are
linked rather than copied where the file system allows. The archive therefore holds a base image
plus an unbroken chain of log segments, and every commit record carries the moment it committed.
Prune old images (with their `.segments` lists) and the log segments before them once you no
longer need to restore that far back; keep the newest image and everything after it. Then remove
a page segment only when no `.segments` list left in `DIR/<database>/` names it:

```bash
set -e
cd /archive/shop
ls pages/*.seg > /tmp/segments            # list the segments first,
ls ./*.segments > /dev/null                # (stop if there is no list at all)
cat ./*.segments | sort -u > /tmp/keep     # then read the lists
while read -r f; do
  grep -qxF "$(basename "$f" .seg)" /tmp/keep || rm "$f"
done < /tmp/segments
```

The order matters while the server runs: an image's list is written before any segment it names
reaches `pages/`, so a segment present when the listing was taken is already named by a list
read after it.

A restore moves the images of the history it cuts away into a `superseded-*` directory together
with their lists; their page segments stay in `pages/`, where the rule above removes them once no
kept list names them.

To restore, stop the server if the target database is live, make sure its directory holds no
log or image (a fresh `CREATE DATABASE`, or remove `btree.wal`, `btree.wal.ckpt` and
`btree.wal.pages` from `base/<db>/`), and run the server in restore mode:

```bash
nusadb-server --data-dir "$DATA_DIR" --wal-archive-dir /archive \
  --restore-database shop --restore-to-time 2026-09-26T10:30:00Z
```

The moment is RFC 3339 in UTC or milliseconds since the Unix epoch; `--restore-to-lsn N`
restores to a log position instead, and neither restores to the newest archived point. Every
commit at or before the target is kept, in commit order: the first commit stamped later ends the
replay, and a transaction without its commit inside that prefix never happened. The result is
sealed into a fresh image in the database directory; start the server normally to serve it.

The archive ends at the last checkpoint. To restore to a moment after it, pass the database's
current log as well (`--restore-live-log "$DATA_DIR/base/shop/btree.wal"`, copied from the
source first); the tail of that file is replayed after the archived segments. Issuing
`CHECKPOINT` before taking a copy of the archive gives it the freshest point.

The segments must join up: if one in the middle was pruned, or one is corrupt in the middle,
the restore refuses rather than replay across it, and so does a chain that ends before an
image that proves the history went on. A target log position the archive never reaches is
refused too; a target moment past the end of the archive succeeds with everything the archive
holds, and the log says where the history actually ended.

A restore also forks the archive at its target. The segments and images after that point
(including the segment that spans it) move into a `superseded-<moment>/` directory under the
archive, and the sealed image, numbered past everything the archive names, takes their place.
Serving the restored database with the same archive then continues one consistent history, and a
later restore into the part that was cut is refused because those records no longer exist on
this line. Points between the last image before the cut and the cut itself are on that moved
segment too, so they are refused after a fork; issue `CHECKPOINT` before a restore whose target
lies far behind the last checkpoint if you may want the moments just before it later. The
restored directory is built under a scratch name and renamed into place only after the fork,
and the fork itself is journaled in a `fork.pending` marker that the next restore or server
start finishes or discards, so an interrupted restore leaves nothing that could be served and
no half-forked archive; run it again and it lands on the same state. Every fork is also
recorded in the archive's `forks` file.

An archive belongs to one history. An engine refuses to open against an archive that already
names a position past its own log, which is what a database dropped and created again under the
same name would meet; `DROP DATABASE` therefore moves the database's archive aside as
`<name>.dropped-<moment>/` under the archive root (and `CREATE DATABASE` does the same with one
a partial drop left behind). To restore a dropped database, copy that directory under a separate
root where it is named after the database again, for example `/tmp/archive/shop/`, create the
database, and restore into its empty directory with `--wal-archive-dir /tmp/archive`. Then keep
serving it with that root as its archive, or move the server root's `<root>/shop/` aside before
switching back, so the recreated database's short history never mixes with the restored one.

Restore from a copy of the archive, or from one no server is writing into: a restore tidies the
archive it reads (scratch copies, unfinished forks), which would trip a checkpoint running into
it at the same moment. That checkpoint fails safely, leaving the image and the full log, but the
restore should not race it.

The whole chain of segments after the base image is read into memory during a restore, so keep
images frequent enough (the checkpoint worker's threshold) that the chain between them stays a
few hundred megabytes at most.

---

### Standby (log shipping)

A second server can follow a primary through the primary's checkpoint archive:

```bash
# primary
nusadb-server --data-dir /var/lib/nusadb --wal-archive-dir /srv/nusadb-archive \
  --checkpoint-threshold-bytes 16777216 --checkpoint-interval 5
# standby (a different data directory; the archive root is shared or replicated to it)
nusadb-server --data-dir /var/lib/nusadb-standby --standby-from /srv/nusadb-archive --listen 0.0.0.0:5679
```

On start the standby registers every database the primary archives, seeds each empty database
directory from the newest archived image, and from then on applies every log segment the
primary's checkpoints archive, in order, once. It serves reads on every database. A write
statement is refused with SQLSTATE `25006`, sequences do not advance, and `CREATE DATABASE` and
`DROP DATABASE` are refused. Between polls (`--standby-poll`, default 5 seconds) new
transactions are held for at most `--standby-max-pause` seconds so the running ones end before
a segment is applied; a transaction held open longer defers the segment to the next poll.

The standby lags the primary by the primary's checkpoint cadence: a segment reaches the archive
only when the primary checkpoints, so set `--checkpoint-threshold-bytes` and
`--checkpoint-interval` on the primary to the lag you can accept. The standby keeps what it has
applied in its own log under the primary's positions, so it survives a restart and continues
from where it was. If the primary is restored to an earlier point (which forks its archive) or
segments are pruned, the standby logs that the archive has moved past it and stops applying;
empty its database directories and start it again to seed afresh.

A standby's own log holds the primary's records at the primary's positions and nothing else, so
`--standby-from` cannot be combined with `--wal-archive-dir`. When the archive root is replicated to
the standby's host rather than shared, each segment must land atomically (copied under a scratch
name and renamed into place): a segment whose copy is still in progress is left alone until it is
complete. A transaction the primary's crash cut off, whose records its recovery kept without an
ending, is skipped by the standby the way the primary's own recovery skipped it.

To promote the standby, stop it and start it again without `--standby-from` (add a
`--wal-archive-dir` of its own if it should archive): it opens every database writable and
continues the primary's history from the last applied position. Point the old primary's
clients at it, and do not start the old primary against the same archive root again without
moving that archive aside, or the two histories would meet in one archive.

## Upgrades

Replace the binary (or pull a newer image tag) and restart the service; recovery replays the log, so
a clean restart keeps every committed transaction. Read the release notes before restarting an
existing data directory on a new version: the on-disk format may still change before 1.0. Each
database directory records which engine wrote it, and one written by the removed `lsm` engine is
refused rather than misread; migrate it by exporting from the last release that shipped that engine
and reloading into a fresh `--data-dir`.
