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
| `--data-dir` | `./data` | Durable data directory: the write-ahead log and checkpoint image of every database, under `base/<name>/`. Created if absent. |
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
| `--max-resident-bytes` | derived | Ceiling on each database's in-memory page store; a row insert past it is refused with an error naming the limit. Derived from the memory budget (floor 256 MiB); unlimited when no budget is known. |
| `--work-mem` | `0` | Per-query memory for one sort, aggregate or join stage. Past it a stage spills (with `--spill-dir`) or fails with an error naming the limit. `0` is unlimited unless a budget derives a value. |
| `--spill-dir` | none | Directory for transient spill files. Sorts and hash joins over `--work-mem` stream to it instead of failing. Stale files from a crash are removed at start-up. |
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

Table pages live in a page cache backed by the last checkpoint image. The image holds every
page of the database; a page is read from it on first use, so a restart does not load the whole
database before serving, and a page that has not changed since the last checkpoint (a clean page)
can leave the cache again when memory is needed. A page changed since the last checkpoint (a
dirty page) stays in memory until the next checkpoint publishes an image that holds it, because
the write-ahead log and the image are the only durable copies of the data.

`--max-resident-bytes` (derived from the memory budget when unset) bounds the cache. Clean pages are
evicted first; once dirty pages and secondary index entries reach the bound, the next insert or
update is refused before it starts (a write already under way always completes) with an error that
names the limit and the bytes held:

```text
ERROR XX000: out of memory: the engine reached its resident-memory limit of 858993440 bytes
(859001088 bytes held: pages changed since the last checkpoint plus index entries); let a
checkpoint run, free rows (DELETE/TRUNCATE), raise the limit, or use a larger host
```

Reads keep working at the bound: a page loaded for a read may briefly overshoot it and is the first
to leave again. The remedy for a refused write is a checkpoint (the background worker runs one when
the log passes `--checkpoint-threshold-bytes` or when changed pages fill half the cache, or issue
`CHECKPOINT`), after which every page is clean and the cache can grow again. So the bound sizes the
working set of changes between checkpoints, not the database: on a host with a bound well below the
data, set the checkpoint threshold so a checkpoint runs before the changes fill the cache.

`DELETE`, `TRUNCATE`, `CREATE INDEX` and the background purge are not refused at the bound, so
space can always be freed; a large `DELETE` is still capped per transaction by
`--max-txn-write-bytes`, and the purge stops early to let a checkpoint run once changed pages
fill half the cache.

Two things still live in memory whatever the bound:

- Secondary index entries (B-tree indexes on columns, vector indexes). They are rebuilt from the
  image's index records at open and count against the bound.
- Every dirty page, as above. A checkpoint writes the whole image (every live page, changed or
  not), so its cost grows with the database, not with the changes.

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
| one executor stage (sort, aggregate, join) | `--work-mem` | spills to `--spill-dir` for sorts and hash joins; otherwise fails with a message naming the limit and the flag |
| one transaction's uncommitted writes | `--max-txn-write-bytes` | the transaction fails with `XX000` |
| one `COPY ... FROM STDIN` | `--copy-max-bytes` | the load is aborted; split it or raise the flag |

Aggregation, `DISTINCT` and window functions do not spill yet; at the budget they fail rather than
swap. A failed query leaves the server responsive, which is the point.

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

That is enough to see whether the server is up and busy, and not enough for latency analysis:
there are no duration histograms, per-database counters, or storage metrics yet, and no
serialization-conflict counter, so track `40001` retries from the application side.

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
capped at 60 seconds. Each checkpoint rewrites the whole image while the
engine is paused, so its cost grows with the database, not with the log: on a large database
raise the threshold so the pause is paid less often. Watch the server log at `info` for
`runtime checkpoint folded the log` and at `debug` for the busy retries. Set either flag to `0` to
turn the worker off and instead issue `CHECKPOINT` from a cron job over an otherwise idle
connection:

```bash
NUSADB_PASSWORD=... nusadb-cli --user nusadb-root -c "CHECKPOINT"
```

It requires a quiesced engine: it refuses, naming how many transactions are still active, while any
transaction is open, including one on the connection issuing it. Run it from a connection in
autocommit at a quiet moment; a load with continuously overlapping transactions may need a retry.

**Backup.** A checkpoint image is a complete copy of one database as of its checkpoint, and the
engine only ever replaces it by an atomic rename, so a copy of the image is a consistent
point-in-time backup even while the server keeps writing. Per database:

```bash
NUSADB_PASSWORD=... nusadb-cli --user nusadb-root -d shop -c "CHECKPOINT"
cp "$DATA_DIR/base/shop/btree.wal.ckpt" /backups/shop-$(date +%F).ckpt
```

The backup holds every transaction committed before the `CHECKPOINT`; what commits afterwards is
in the log tail only. The background checkpoint worker refreshes the image on its own as the log
grows, so a copy taken without an explicit `CHECKPOINT` is still consistent, just older.

**Restore.** The database must be registered in the cluster (it is, if it was created there; in a
fresh data directory run `CREATE DATABASE shop` first), and its directory must hold no log
(`btree.wal`): with the server stopped, remove that database's log if one exists, place the copy,
and start the server. It opens the image with an empty log tail.

```bash
rm -f "$DATA_DIR/base/shop/btree.wal" "$DATA_DIR/base/shop/btree.wal.ckpt"
cp /backups/shop-2026-09-26.ckpt "$DATA_DIR/base/shop/btree.wal.ckpt"
nusadb-server --data-dir "$DATA_DIR"
```

A `btree.wal` left in place would be replayed on top of the image, which is not a restore. For a
whole-cluster copy with the server stopped, archive the `--data-dir` tree and extract it in place.
Logical export with `COPY table TO STDOUT` and reload with `COPY table FROM STDIN` remains
available. There is no built-in scheduled backup or replication.

### Point-in-time recovery

With `--wal-archive-dir DIR`, every checkpoint keeps two files under `DIR/<database>/` before it
truncates the log: `<lsn>.log`, the log segment it folded (every record up to log position
`<lsn>`), and `<lsn>.ckpt`, the image it published, linked rather than copied where the file
system allows. The archive therefore holds a base image plus an unbroken chain of segments, and
every commit record carries the moment it committed. Prune old images and the segments before
them once you no longer need to restore that far back; keep the newest image and everything after
it.

To restore, stop the server if the target database is live, make sure its directory holds no
log or image (a fresh `CREATE DATABASE`, or remove `btree.wal` and `btree.wal.ckpt` from
`base/<db>/`), and run the server in restore mode:

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
