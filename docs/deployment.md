# Deploying NusaDB

NusaDB ships as a single server binary (`nusadb-server`) plus an interactive
client (`nusa-cli`). This guide covers a bare-metal install on a Linux VM with systemd.

The server is configured entirely by command-line flags (no config file). The
durable state — WAL and page files — lives under `--data-dir`; back that path up
and you have backed up the database.

## Server flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--listen` | `0.0.0.0:5678` | TCP listen address for the wire protocol. |
| `--data-dir` | `./data` | Durable data directory (WAL + page files). |
| `--auth-user USER:PASSWORD` | — | Require SCRAM-SHA-256 for this user (repeatable). When **any** is set, every connection must authenticate; otherwise auth is trust-on-startup. |
| `NUSADB_USER` + `NUSADB_PASSWORD` (env) | — | Fallback auth when no `--auth-user` is given: both together require SCRAM for that user (lets a container require auth without baking a secret into the image). Setting only one is an error. |
| `--tls-cert` / `--tls-key` | — | PEM cert chain + key; enables TLS when both are set. |
| `--tls-client-ca` | — | PEM CA for **mutual** TLS: every client must present a cert signed by this CA. |
| `--metrics-listen` | — | Serve Prometheus metrics on this address (e.g. `0.0.0.0:9100`); disabled if unset. |
| `--idle-timeout` | `0` | Close a connection idle this many seconds (`0` = no limit). |
| `--max-connections` | `25` | Cap concurrent connections; excess queue (`0` = unlimited). Small-safe default — raise it on a larger host (see *Resource defaults* below). |
| `--mem-budget` | `0` | Engine memory budget in bytes; new transactions are refused with an honest error once the engine's logical footprint reaches it, instead of an OS OOM-kill (`0` = unlimited). |
| `--work-mem` | `0` | Per-query work-memory budget in bytes; a query that materializes more than this in one executor stage (a big sort / aggregate / join, or — since the executor is not yet streaming — a wide scan) is failed honestly instead of OOM-killing the server (`0` = unlimited). |
| `--drain-timeout` | `30` | On Ctrl-C, wait this long for in-flight connections to drain. |
| `--statement-timeout` | `0` | Cancel statements running longer than this many seconds (`0` = no limit). |

`RUST_LOG` controls log verbosity (`tracing` env-filter, e.g. `RUST_LOG=info`).

> **Production checklist:** always require authentication — set at least one
> `--auth-user` (or `NUSADB_USER`/`NUSADB_PASSWORD`) — terminate TLS with
> `--tls-cert`/`--tls-key`, and keep `--data-dir` on a persistent, backed-up volume.
> With no credentials the server runs **trust-on-startup** (any client accepted, no
> password) and logs a startup **WARNING** — acceptable for local/dev/testing only.

## Resource defaults (small-safe Tier-1)

NusaDB defaults **small and scales up explicitly**: a fresh install is tuned to stay healthy on a
2 GB RAM / 1–2 vCPU host (a $5–15/month VPS — "Tier-1") with no flags, rather than defaulting to
big-server values that OOM a small host. A larger machine raises the limits deliberately.

| Resource | Default (Tier-1) | Where it lives | Raise it on a bigger host |
| --- | ---: | --- | --- |
| Max concurrent connections | 25 | `--max-connections` | set a higher `--max-connections`, or front with a connection pooler |
| Engine memory budget | unlimited (`0`) | `--mem-budget` | set a byte cap on a constrained host; new transactions are then refused honestly past it rather than OOM-killing |
| Per-query work memory | unlimited (`0`) | `--work-mem` | set a byte cap so one big sort/aggregate/join (or wide scan) fails honestly rather than OOM-killing |
| Background purge worker | 1 | version-store purge scheduler (wired at the composition root) | (a single worker is sufficient at Tier-1 write rates) |

> The memory budget defaults to off (`0`) until its Tier-1 value is calibrated against measured RSS
> on a real small VM; the engine footprint it caps is a logical-bytes estimate, so the safe number is
> set from measurement, not guessed. Set `--mem-budget` explicitly to cap memory today.

Other knobs (`--idle-timeout`, `--drain-timeout`, `--statement-timeout`) default to off/30 s and are
not memory-bound. A declarative profile system (`--profile t0|t1|t2|t3|auto` over a `nusa.toml`) that
sets these in one step is planned; until then, raise the individual flags above.

### CPU compatibility (older CPUs / ARM)

The query executor uses AVX2 SIMD where available and **automatically falls back to a portable
scalar path** on CPUs without it (pre-2013 x86, some budget VPS, ARM) — the engine never emits an
illegal instruction. To force the scalar path even on an AVX2 host (to validate a deployment target,
or as an emergency switch if the SIMD path is ever suspect on specific hardware), set the
`NUSADB_DISABLE_SIMD` environment variable to any value before starting the server. Behaviour is
identical either way; only throughput differs.

## Running on a Linux VM (systemd)

### 1. Get the binary

Either download the release tarball for your platform from the GitHub Release
(`nusadb-<version>-x86_64-unknown-linux-gnu.tar.gz`), or build from source on the
VM (requires the Rust toolchain pinned in `rust-toolchain.toml`):

```bash
cargo build --release --locked -p nusadb-server -p nusadb-cli
sudo install -m 0755 target/release/nusadb-server /usr/local/bin/
sudo install -m 0755 target/release/nusa-cli      /usr/local/bin/
```

### 2. Create a service user and data directory

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin nusadb
sudo mkdir -p /var/lib/nusadb /etc/nusadb
sudo chown -R nusadb:nusadb /var/lib/nusadb
# Place server.crt / server.key under /etc/nusadb (root-owned, readable by nusadb).
sudo chown root:nusadb /etc/nusadb/server.* && sudo chmod 0640 /etc/nusadb/server.*
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
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/nusadb-server \
  --listen 0.0.0.0:5678 \
  --data-dir /var/lib/nusadb \
  --auth-user admin:STRONG_PASSWORD \
  --tls-cert /etc/nusadb/server.crt \
  --tls-key /etc/nusadb/server.key \
  --metrics-listen 127.0.0.1:9100
Restart=on-failure
RestartSec=2

# Hardening — the server only needs its data dir writable.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/nusadb

[Install]
WantedBy=multi-user.target
```

> Store secrets out of the unit file where you can: use
> `EnvironmentFile=/etc/nusadb/nusadb.env` (mode `0600`, root-owned) and reference
> the values, rather than inlining the password in `ExecStart`.

### 4. Start and verify

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nusadb
sudo systemctl status nusadb
journalctl -u nusadb -f          # follow logs

# From a client host (open the VM firewall to 5678 first):
nusa-cli --host VM_HOST:5678 --user admin
```

### 5. Firewall

Open the wire port to your clients; keep metrics private.

```bash
# ufw example — expose 5678, keep 9100 on localhost only.
sudo ufw allow 5678/tcp
```

## Backup & restore

The entire database is the `--data-dir` tree. With the server stopped (or using a
filesystem snapshot for a consistent point-in-time copy), archive that directory;
restore by extracting it back and starting the server against it.

## Upgrades

Replace the binary (or pull a newer image tag), then restart the service. The WAL
is replayed on startup, so a clean restart recovers all committed transactions.
