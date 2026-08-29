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
| `--drain-timeout` | `30` | On Ctrl-C, wait this long for in-flight connections to finish before aborting them. `0` waits indefinitely. |
| `--mem-budget` | `0` | Total memory budget in bytes. `0` auto-detects on Linux as the smaller of host RAM and the cgroup limit, so a container limit is honoured; on other systems `0` means no budget. The budget derives the limits below. |
| `--max-resident-bytes` | derived | Ceiling on each database's in-memory page store; a row insert past it is refused with an error naming the limit. Derived from the memory budget (floor 256 MiB); unlimited when no budget is known. |
| `--work-mem` | `0` | Per-query memory for one sort, aggregate or join stage. Past it a stage spills (with `--spill-dir`) or fails with an error naming the limit. `0` is unlimited unless a budget derives a value. |
| `--spill-dir` | none | Directory for transient spill files. Sorts and hash joins over `--work-mem` stream to it instead of failing. Stale files from a crash are removed at start-up. |
| `--maintenance-work-mem` | `0` | Bytes of index entries a `CREATE INDEX` buffers before flushing a sorted batch. `0` uses a built-in bound. |
| `--max-txn-write-bytes` | `0` | Ceiling on the uncommitted writes one transaction may buffer; past it the transaction fails with `XX000` instead of growing until the host kills the process. `0` derives 25% of the budget (floor 128 MiB). |
| `--copy-max-bytes` | derived | Ceiling on one `COPY ... FROM STDIN`. Derived as about 20% of the budget, capped at 1 GiB; `0` is unbounded. |
| `--autoanalyze-interval` | `60` | Seconds between sweeps of the background worker that re-analyzes tables whose statistics went stale. `0` disables it. |
| `--autoanalyze-scale` / `--autoanalyze-threshold` | `0.1` / `50` | A table is re-analyzed once its writes since the last analyze exceed `threshold + scale * rows`. |
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

### Table data is bounded by memory

Table pages live in memory and are made durable through the write-ahead log; pages are not evicted
to disk. Once a database's resident store reaches its ceiling (`--max-resident-bytes`, derived from
the memory budget when unset), further row inserts are refused with an error that names the limit
and the bytes resident:

```text
ERROR XX000: out of memory: the in-memory store reached its resident-memory limit of
858993440 bytes (859001088 bytes resident); free rows (DELETE/TRUNCATE), raise the limit,
or use a larger host
```

Size against the ceiling, not against total RAM. With the default derivation the ceiling is about a
fifth of the memory budget: measured, an 859 MB ceiling inside a 4 GB container, which held roughly
2 to 3 million rows of about 220 bytes. A dataset larger than the ceiling does not load slowly; it
does not load, and the first sign is the insert refusal itself, mid-load.

Three details worth knowing:

- Updates and index builds are not gated by the ceiling, so an update-heavy workload already at the
  limit can still grow past it.
- Deleting rows frees pages for reuse but does not lower the resident meter within a running
  process; page memory is recycled rather than returned. A restart after a checkpoint does lower
  it, because recovery rebuilds the store from the checkpoint image, which holds only live rows.
- Once the ceiling has been hit, the remedies are raising it, using a larger host, or reloading the
  live rows into a fresh data directory.

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

While the server runs, nothing checkpoints automatically, so a long-lived server keeps appending.
Bound the log and the restart time by issuing `CHECKPOINT` from a cron job over an otherwise idle
connection:

```bash
NUSADB_PASSWORD=... nusadb-cli --user nusadb-root -c "CHECKPOINT"
```

It requires a quiesced engine: it refuses, naming how many transactions are still active, while any
transaction is open, including one on the connection issuing it. Run it from a connection in
autocommit at a quiet moment; a load with continuously overlapping transactions may need a retry.

**Backup.** The whole database is the `--data-dir` tree. With the server stopped, or from a
filesystem snapshot for a consistent point-in-time copy, archive that directory; restore by
extracting it and starting the server against it. Alternatively export each table over a connection
with `COPY table TO STDOUT` and reload with `COPY table FROM STDIN`. There is no built-in scheduled
backup, point-in-time recovery, or replication.

---

## Upgrades

Replace the binary (or pull a newer image tag) and restart the service; recovery replays the log, so
a clean restart keeps every committed transaction. Read the release notes before restarting an
existing data directory on a new version: the on-disk format may still change before 1.0. Each
database directory records which engine wrote it, and one written by the removed `lsm` engine is
refused rather than misread; migrate it by exporting from the last release that shipped that engine
and reloading into a fresh `--data-dir`.
