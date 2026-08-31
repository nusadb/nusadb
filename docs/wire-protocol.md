# Nusa Wire Protocol — Normative Specification

> Status: `PROTOCOL_VERSION 1.2` (`major.minor = 1.2`). The 1.0 surface is frozen; 1.1 and 1.2 are
> additive minor revisions. 1.1 adds one backend message, `RowDescriptionTyped` (§9/§16), sent only
> when a connection negotiates `minor >= 1`; a `minor = 0` connection sees the exact 1.0 byte stream
> (§2). 1.2 additionally encodes an `ARRAY` column's *element* type in its 1-byte type tag
> (`0x80 | element_tag`, §9.2) for a connection negotiating `minor >= 2`, so a client decodes the
> elements at their real type; a `minor < 2` connection keeps the plain `ARRAY` tag (`0x0F`). All
> earlier layouts are unchanged — old drivers do not break.
> This document is the normative contract between a NusaDB server (`nusadb-wire`) and any
> client/driver. It is the freeze point that lets driver development proceed in parallel
> with engine work: as long as the engine keeps emitting these frames in these layouts, drivers do
> not break — new SQL features ride over the same byte format via SQL text and `DataRow`.
>
> Source of truth = `crates/nusadb-wire/src/` (`frame.rs`, `messages.rs`, `value.rs`,
> `cancel.rs`, `auth/scram.rs`, `server.rs`). This spec is derived from, and kept in lockstep
> with, that code. If code and spec disagree, that is a bug in one of them — report it; do not
> silently diverge a driver. Any change to a wire-visible layout requires a `PROTOCOL_VERSION`
> bump (§2) and an update here.

## 1. Scope

This protocol is NusaDB's own. It is **not** compatible with any other database's wire format:
a client library or tool built for another engine cannot connect — the server does not answer a
foreign handshake, which from that client's side looks like a silent socket. This is a standing
design decision, not an omission; a client is either one of the NusaDB drivers, `nusadb-cli`, or an
implementation of this document.

This covers the on-wire byte format and message exchange:

- TCP framing and primitive encodings (§3–§4)
- Connection lifecycle / state machine (§5)
- Startup and version negotiation (§6)
- Authentication: trust-on-startup and SCRAM-SHA-256 / SASL (§7)
- Post-auth setup: `BackendKeyData`, `ReadyForQuery` (§8)
- Simple query (§9) and extended query / prepared statements (§10)
- Value encodings: text and binary `DataRow` fields (§11)
- COPY bulk sub-protocol (§12)
- Out-of-band statement cancellation (§13)
- Errors and SQLSTATE (§14), termination (§15)
- Full message reference (§16), limits & DoS guards (§17), conformance checklist (§18)

It does not specify SQL semantics (that is the engine's surface) nor TLS internals beyond the
handshake ordering (rustls / TLS 1.3, see §5.1).

This is a native protocol — NusaDB's own format, not a re-implementation of any other database's
wire protocol. The magic, version negotiation, startup payload, message framing, and every type
byte and sub-code are NusaDB's own. Do not assume a third-party database driver will interoperate.

## 2. Versioning

- Magic. Every `Startup` payload begins with the 4-byte magic `PROTOCOL_MAGIC = 0x4E55_5341`,
  ASCII `"NUSA"`. A server that reads any other value MUST reject the connection (`BadMagic`).
- Version. `PROTOCOL_VERSION = (major, minor) = (1, 2)`. The `Startup` message carries the
  client's requested `major:u16` and `minor:u16`.
  - Major identifies an incompatible change. The server rejects a `Startup` whose `major`
    differs from its own (it replies with an `Error` and closes — see §6).
  - Minor selects backward-compatible additions. The server uses the effective minor =
    `min(client_minor, server_minor)` for the connection and stores it (it is no longer
    ignored). A client requesting `minor >= 1` opts into the 1.1 additions (typed
    `RowDescriptionTyped`, §9); `minor >= 2` additionally opts into element-typed `ARRAY` tags (§9.2).
    A client requesting `minor = 0` — or any client talking to a 1.0
    server, which ignores the field — sees the exact 1.0 byte stream. A 1.1-aware client need not
    know the server's minor in advance: the 1.1 additions are self-describing (a distinct
    message type byte), so the client reacts to whichever form it actually receives.
- Compatibility rule. Within a frozen major version, the server MUST NOT change the layout of
  any message defined here, nor the meaning of any type byte, sub-code, or field. New messages or
  fields require a minor bump and MUST be additive (an old client that ignores them keeps
  working). Any breaking change requires a major bump.

## 3. Framing

Every message is a single frame:

```
[ message_type : u8 ] [ len : u32 ] [ payload : len - 5 bytes ]
```

- Byte order is big-endian (network order) everywhere: every multi-byte integer and every
  IEEE-754 float in this protocol.
- `message_type` is one ASCII byte identifying the message (§16).
- `len` is the total frame length including the 5-byte header (`type` + `len`). So the
  payload is `len - 5` bytes. A reader knows exactly how many bytes to consume.
- The header is `HEADER_LEN = 5` bytes.
- Limits: `len` MUST be `>= 5`. A frame with `len < 5` is malformed (`MalformedFrame`). A frame
  with `len > MAX_FRAME_LEN` (`256 MiB = 256 * 1024 * 1024`) MUST be rejected (`FrameTooLarge`)
  before allocating — this bounds memory against a crafted length prefix.
- A reader that has fewer than `len` bytes buffered waits for more; it never blocks the codec.

The same `message_type` byte may mean different messages in each direction (e.g. `K` is
`CancelRequest` client→server but `BackendKeyData` server→client; `C` is `Close` inbound but
`CommandComplete` outbound). This is unambiguous: a connection always knows which side it is
decoding. §16 lists both directions.

## 4. Primitive field encodings

Within a payload, fields are packed back-to-back using these primitives. They are the only
field shapes used; a driver that implements them can encode/decode every message.

### 4.1 Fixed-width integers
`u8`, `u16`, `u32` — big-endian, no padding. `i32`/`i64` (in value bodies) are big-endian two's
complement.

### 4.2 String — `Str`
```
[ len : u32 ] [ bytes : len ]     // UTF-8, NOT null-terminated
```
A length-prefixed UTF-8 byte run. There is no trailing `\0`; the length prefix delimits it.
A decoder MUST validate UTF-8 and reject invalid sequences (`InvalidString`). The encoder MUST
error rather than truncate if `len > u32::MAX` (`FieldTooLarge`).

### 4.3 Field list — `Fields` (used by `DataRow` and `Bind` params)
```
[ count : u16 ]
repeated count times:
    [ present : u8 ]              // 0x00 = SQL NULL (no further bytes); 0x01 = present
    if present == 0x01:
        [ len : u32 ] [ bytes : len ]
```
A column/parameter value list. `present = 0` is SQL `NULL` and carries no value bytes. `present = 1`
is followed by a `u32` length and that many raw bytes (text or binary format per §11). The `count`
is a `u16`, so a message carries at most 65 535 fields; an encoder MUST error (`FieldTooLarge`)
rather than truncate the count.

### 4.4 Count prefixes
Repeated sub-elements (e.g. column names in `RowDescription`, SASL mechanisms in `AuthSasl`,
result-format codes in `Bind`) use a leading `u16` count followed by that many elements. Lengths
of variable byte runs use `u32`. Encoders MUST error rather than truncate an over-wide count or
length.

### 4.5 Defensive decoding (normative for servers; recommended for clients)
A decoder MUST bound every allocation by the bytes actually remaining before trusting a count:
e.g. a `Fields` `count` cannot exceed the remaining byte length (each field is `>= 1` byte), and a
result-format `count` cannot exceed `remaining / 2`. A decoder MUST return a truncation error
(`TruncatedPayload`) when a declared length runs past the payload, never read out of bounds.

## 5. Connection lifecycle (state machine)

```
            (optional implicit TLS handshake — §5.1)
                          │
   ┌──────────────────────▼───────────────────────┐
   │  STARTUP                                      │
   │  client → Startup  (or CancelRequest, §13)    │
   └──────────────────────┬───────────────────────┘
                          │  version ok
            ┌─────────────┴──────────────┐
            │ AuthStore configured?      │
       no (trust)                  yes (SCRAM)
            │                            │
   server → AuthOk          ┌────────────▼─────────────┐
            │               │ AUTHENTICATION (SASL §7) │
            │               │ R(AuthSasl) →             │
            │               │ p(SASLInitial) →          │
            │               │ R(AuthSaslContinue) →     │
            │               │ r(SASLResponse) →         │
            │               │ R(AuthSaslFinal) →        │
            │               │ R(AuthOk)                 │
            │               └────────────┬─────────────┘
            └─────────────┬──────────────┘  auth ok
                          │
   server → BackendKeyData (K)   ── connection's (pid, secret) for cancel (§13)
   server → ReadyForQuery (Z, Idle)
                          │
   ┌──────────────────────▼───────────────────────┐
   │  QUERY LOOP  (repeat)                         │
   │   • simple query  (§9)                        │
   │   • extended query  (§10)                     │
   │   • COPY sub-protocol  (§12)                  │
   │  ends each unit with ReadyForQuery (Z)        │
   └──────────────────────┬───────────────────────┘
                          │  Terminate (X) or EOF
                          ▼
                       CLOSED
```

Any failed/abandoned handshake drops the connection without reaching the query loop. A
`handshake_timeout` (default 60 s, §17) bounds the whole Startup+auth phase independently of the
idle timeout, so a stalled unauthenticated client cannot hold a slot (slowloris defence).

### 5.1 TLS (§L2)
TLS is mandatory by default and implicit: when the server is configured with TLS, every
accepted TCP connection is wrapped in a rustls server session (TLS 1.3 only, forward-secret cipher
suites) before any frame is read. There is no in-band `SSLRequest` negotiation step — the
client opens TLS immediately, then sends `Startup` inside the TLS stream. mTLS is available for
service-to-service auth. A plaintext server (dev/local) skips this and reads frames directly.

## 6. Startup

Frontend → `Startup` (type `S`):
```
[ magic : u32 = 0x4E555341 ]
[ major : u16 ]
[ minor : u16 ]
[ user     : Str ]
[ database : Str ]
```

The server validates `magic == PROTOCOL_MAGIC` (else `BadMagic`) and `major == PROTOCOL_VERSION.0`.
On a major mismatch the server replies with an `Error` ("unsupported protocol major version") and
closes the connection. `user` is the authenticating role; `database` selects the target database.

The very first frame of a connection MUST be `Startup` — except that a connection opened solely
to cancel another's statement sends `CancelRequest` instead (§13). Any other first message gets an
`Error` ("expected Startup message") and the connection closes.

## 7. Authentication

After a valid `Startup`, one of two paths runs, decided by server configuration:

### 7.1 Trust-on-startup (no `AuthStore`)
The server immediately sends `AuthOk` and proceeds. The declared `user` is accepted without a
password. Suitable for local/dev or a trusted network only.

### 7.2 SCRAM-SHA-256 (RFC 7677 / RFC 5802) over SASL
Used when the server has an `AuthStore`. The only mechanism is `SCRAM-SHA-256` (no channel
binding in this build; the GS2 header is `n,,`).

All authentication messages share the backend type byte `R` (`AuthOk`). The first `u32` of an
`R` payload is the auth sub-code:

| sub-code | constant            | meaning                                  |
| -------- | ------------------- | ---------------------------------------- |
| `0`      | `AUTH_OK`           | authentication succeeded                 |
| `10`     | `AUTH_SASL`         | offer SASL mechanisms                     |
| `11`     | `AUTH_SASL_CONTINUE`| SASL challenge (server-first)            |
| `12`     | `AUTH_SASL_FINAL`   | SASL final (server verifier)             |

Exchange (each line is one frame):

1. server → `AuthSasl` (`R`, sub-code 10): `[10:u32][ count:u16 ][ mechanism:Str ]*` — offers
   `["SCRAM-SHA-256"]`.
2. client → `SASLInitialResponse` (`p`): `[ mechanism:Str ][ data_len:u32 ][ data:bytes ]`
   where `data` is the SCRAM client-first-message:
   `n,,n=<user>,r=<client-nonce>` (GS2 header `n,,` + `n=`username + `r=`client nonce, base64-safe
   printable). `mechanism` MUST be `SCRAM-SHA-256`.
3. server → `AuthSaslContinue` (`R`, sub-code 11): `[11:u32][ server-first-message bytes ]`.
   The server-first-message is `r=<combined-nonce>,s=<base64-salt>,i=<iterations>` where the
   combined nonce is `<client-nonce><server-nonce>` (server appends 18 random bytes' worth).
4. client → `SASLResponse` (`r`): `[ client-final-message bytes ]` (raw, no length prefix — the
   frame length delimits it). The client-final-message is
   `c=<base64(gs2-header)>,r=<combined-nonce>,p=<base64-client-proof>`. `c=` MUST be the base64 of
   the GS2 header (`biws` for `n,,`); `r=` MUST echo the combined nonce exactly.
5. server → `AuthSaslFinal` (`R`, sub-code 12): `[12:u32][ server-final-message bytes ]` =
   `v=<base64-server-signature>` (lets the client verify the server — mutual auth).
6. server → `AuthOk` (`R`, sub-code 0): `[0:u32]`. Authentication complete.

Rules. The SCRAM `n=` username MUST match the `Startup` `user` (no cross-user auth). A missing
user, wrong password, malformed message, mechanism mismatch, or nonce/channel-binding mismatch all
fail with a single generic `Error` ("authentication failed") to avoid user enumeration, after which
the connection closes. SCRAM crypto is RFC 5802 §3: `SaltedPassword = PBKDF2(password, salt, i)`,
`ClientKey = HMAC(SaltedPassword,"Client Key")`, `StoredKey = SHA-256(ClientKey)`,
`ClientSignature = HMAC(StoredKey, AuthMessage)`, `ClientProof = ClientKey XOR ClientSignature`,
`ServerSignature = HMAC(HMAC(SaltedPassword,"Server Key"), AuthMessage)`, with
`AuthMessage = client-first-bare + "," + server-first + "," + client-final-without-proof`. Clients
MUST verify the `v=` server signature in constant time.

## 8. Post-authentication setup

Immediately after `AuthOk` the server sends, in order:

1. `BackendKeyData` (type `K`): `[ pid : u32 ][ secret : u32 ]` — this connection's
   cancellation key. The client SHOULD store it to later cancel an in-flight statement (§13). The
   `secret` is a CSPRNG value.
2. `ReadyForQuery` (type `Z`): `[ status : u8 ]` — the server is now idle and ready. The status
   byte is the transaction state (§8.1).

### 8.1 `ReadyForQuery` transaction status
Every `ReadyForQuery` carries a 1-byte transaction status:

| byte | constant        | meaning                                              |
| ---- | --------------- | ---------------------------------------------------- |
| `I`  | `Idle`          | not in a transaction                                 |
| `T`  | `InTransaction` | inside an open transaction                            |
| `E`  | `Failed`        | inside a transaction that errored; must be rolled back |

> 1.0 note. The current server reports `Idle` (`I`) at every `ReadyForQuery`. Clients MUST still
> decode and tolerate `T` and `E` (they are part of the frozen surface and may be reported by a
> minor revision). Do not hard-code `I`.

## 9. Simple query

Frontend → `Query` (type `Q`): `[ sql : Str ]` — one SQL string.

The server responds with a sequence ending in `ReadyForQuery`:

- Row-returning statement:
  1. Column metadata, in order. At effective `minor = 0`: `RowDescription` (`T`):
     `[ count:u16 ][ column_name:Str ]*` — output column names. At effective `minor >= 1` (1.1):
     `RowDescriptionTyped` (`y`): `[ count:u16 ]( [ column_name:Str ][ type_tag:u8 ] )*` — each column
     also carries a 1-byte type tag (§9.2). The two are distinct message bytes, so a client tells
     them apart without knowing the negotiated minor; a 1.1 client MUST handle both (`y` when the
     server is 1.1, `T` when it is 1.0).
  2. zero or more `DataRow` (`D`): a `Fields` list (§4.3), one field per column, text format
     (§11) by default.
  3. `CommandComplete` (`C`): `[ tag:Str ]` — e.g. `SELECT 3`.
- Non-row statement: just `CommandComplete` with the appropriate tag (§9.1).
- Error: an `Error` message (§14) instead of the result, then `ReadyForQuery`.

After the result (success or error) the server always sends `ReadyForQuery` (`Z`).

If the SQL is a `COPY … FROM STDIN` / `COPY … TO STDOUT`, the COPY sub-protocol runs instead
(§12). A simple `Query` also abandons any half-built extended-query pipeline.

### 9.1 `CommandComplete` tags
The tag is a human-readable completion string. Known forms (1.0):

| statement | tag |
| --- | --- |
| `SELECT` (and row-returning) | `SELECT <rowcount>` |
| `INSERT` / `UPDATE` / `DELETE` / `MERGE` | `INSERT <n>` / `UPDATE <n>` / `DELETE <n>` / `MERGE <n>` |
| `COPY` | `COPY <count>` |
| `CREATE` / `DROP` / `ALTER TABLE` | `CREATE TABLE` / `DROP TABLE` / `ALTER TABLE` |
| `CREATE` / `DROP VIEW` | `CREATE VIEW` / `DROP VIEW` |
| `CREATE` / `DROP` / `REFRESH MATERIALIZED VIEW` | `CREATE MATERIALIZED VIEW` / `DROP MATERIALIZED VIEW` / `REFRESH MATERIALIZED VIEW` |
| `CREATE` / `DROP TYPE` (enum and composite) | `CREATE TYPE` / `DROP TYPE` |
| `CREATE` / `DROP DOMAIN` | `CREATE DOMAIN` / `DROP DOMAIN` |
| `CREATE` / `DROP POLICY` | `CREATE POLICY` / `DROP POLICY` |
| `CREATE` / `DROP` / `ALTER TRIGGER` | `CREATE TRIGGER` / `DROP TRIGGER` / `ALTER TRIGGER` |
| `CREATE` / `DROP PROCEDURE`, `CALL` | `CREATE PROCEDURE` / `DROP PROCEDURE` / `CALL` |
| `CREATE` / `DROP FUNCTION` | `CREATE FUNCTION` / `DROP FUNCTION` |
| `CREATE` / `DROP SCHEMA` | `CREATE SCHEMA` / `DROP SCHEMA` |
| `CREATE` / `DROP` / `ALTER SEQUENCE` | `CREATE SEQUENCE` / `DROP SEQUENCE` / `ALTER SEQUENCE` |
| `CREATE` / `DROP INDEX` | `CREATE INDEX` / `DROP INDEX` |
| `CREATE EXTENSION` (accepted, no-op) | `CREATE EXTENSION` |
| `CREATE` / `DROP` / `ALTER DATABASE` | `CREATE DATABASE` / `DROP DATABASE` / `ALTER DATABASE` |
| `CREATE` / `DROP` / `ALTER ROLE` | `CREATE ROLE` / `DROP ROLE` / `ALTER ROLE` |
| `GRANT` / `REVOKE` | `GRANT` / `REVOKE` |
| `BEGIN` / `COMMIT` / `ROLLBACK` | `BEGIN` / `COMMIT` / `ROLLBACK` |
| `SAVEPOINT` / `RELEASE` | `SAVEPOINT` / `RELEASE` |
| `SET` (vars, `SET TRANSACTION`, `SET ROLE`) | `SET` |
| `TRUNCATE` | `TRUNCATE TABLE` |
| `LOCK TABLE` | `LOCK TABLE` |
| `PREPARE` / `DEALLOCATE` | `PREPARE` / `DEALLOCATE` (embedded API only — see §14) |
| `DECLARE` / `CLOSE`, `FETCH` | `DECLARE CURSOR` / `CLOSE CURSOR` / `FETCH <rowcount>` (embedded API only — see §14) |
| `VACUUM` / `ANALYZE` / `COMMENT` | `VACUUM` / `ANALYZE` / `COMMENT` |
| `REINDEX` / `CHECKPOINT` | `REINDEX` / `CHECKPOINT` |
| `CLUSTER` (accepted, no-op) | `CLUSTER` |

A statement with `RETURNING` still reports its own action, not `SELECT`: an
`INSERT ... RETURNING` streams the returned rows and completes with `INSERT <n>` (and likewise
`UPDATE <n>` / `DELETE <n>`), so a client reading the affected-row count off the tag is not misled.

Each kind of object reports its own tag. A statement that creates a view says `CREATE VIEW`, not
`CREATE TABLE` — earlier builds reported the table tag for views, types, domains and policies alike,
so a client that branched on the tag was told the wrong thing.

A driver MUST treat the tag as informational text: parse the trailing count where it needs an
affected-row number, but tolerate tags it does not recognise.

### 9.2 Column `type_tag` (protocol 1.1, extended in 1.2)
Each column of a `RowDescriptionTyped` carries a 1-byte tag identifying its type — the §11 taxonomy
as a single byte. A client maps it to its own type system (e.g. a `columnTypes` array). `0x00` is
reserved for a type the server could not resolve; treat it as `TEXT`. `VECTOR` uses one tag (its
dimension is not carried). Values a client does not recognise SHOULD be treated as `TEXT`.

| tag    | type          | tag    | type          |
| ------ | ------------- | ------ | ------------- |
| `0x00` | `UNKNOWN`     | `0x09` | `TIMETZ`      |
| `0x01` | `BOOL`        | `0x0A` | `TIMESTAMP`   |
| `0x02` | `INT`         | `0x0B` | `TIMESTAMPTZ` |
| `0x03` | `FLOAT`       | `0x0C` | `INTERVAL`    |
| `0x04` | `NUMERIC`     | `0x0D` | `UUID`        |
| `0x05` | `TEXT`        | `0x0E` | `JSON`        |
| `0x06` | `BYTES`       | `0x0F` | `ARRAY`       |
| `0x07` | `DATE`        | `0x10` | `VECTOR`      |
| `0x08` | `TIME`        |        |               |

ARRAY element type (protocol 1.2, `minor >= 2`). For an `ARRAY` column the server sets the high
bit and carries the element type in the low 7 bits: `type_tag = 0x80 | element_tag`, where
`element_tag` is the element's own scalar tag from the table above. So `INT[]` is `0x82`, `TEXT[]` is
`0x85`, `TIMESTAMPTZ[]` is `0x8B`. A client recovers the element type as `tag & 0x7F` when `tag & 0x80`
is set, and decodes the (canonical-text) `{…}` array's elements at that type instead of as text. The
message layout is unchanged (still one byte per column); a `minor < 2` connection receives the plain
`0x0F` `ARRAY` tag (elements stay text), so this is backward-compatible. The element types that may
appear are the array-element scalars (`BOOL`, `INT`, `FLOAT`, `TEXT`, `DATE`, `TIME`, `TIMESTAMP`,
`TIMESTAMPTZ`, `UUID`); a client that does not recognise `0x80 | x` SHOULD treat the column as `ARRAY`.

The type metadata for an extended-query statement is reported by `Describe(Portal)` the same way
(§10): `RowDescriptionTyped` at `minor >= 1`, otherwise `RowDescription`.

## 10. Extended query (prepared statements)

The extended-query protocol separates parse, bind, and execute, enabling prepared statements,
positional parameters (`$1`, `$2`, …), and per-column result formats. Messages:

| msg        | type | payload                                                                          |
| ---------- | ---- | -------------------------------------------------------------------------------- |
| `Parse`    | `P`  | `[ name:Str ][ sql:Str ][ pt_count:u16 ][ param_type_tags:bytes ]`               |
| `Bind`     | `B`  | `[ portal:Str ][ statement:Str ][ params:Fields ][ fmt_count:u16 ][ result_format:u16 ]* ]` |
| `Describe` | `D`  | `[ target:u8 ('S'|'P') ][ name:Str ]`                                            |
| `Execute`  | `E`  | `[ portal:Str ][ max_rows:u32 ]`                                                 |
| `Sync`     | `Y`  | (empty)                                                                           |
| `Close`    | `C`  | `[ target:u8 ('S'|'P') ][ name:Str ]`                                            |

Backend replies: `ParseComplete` (`1`), `BindComplete` (`2`), `CloseComplete` (`3`),
`ParameterDescription` (`t`: `[ count:u16 ]`), `NoData` (`n`), `RowDescription` (`T`),
`DataRow` (`D`), `PortalSuspended` (`z`), `CommandComplete` (`C`), `ReadyForQuery` (`Z`).

### 10.1 Flow
1. `Parse` stores SQL under a statement name (empty `name` = the unnamed statement). The server
   replies `ParseComplete`. `param_types` are placeholder type-tag hints, in order; in 1.0 they
   are not required and MAY be empty (the server infers parameter types from value text, §10.4).
2. `Bind` creates a portal from a statement plus parameter values and result-format codes. The
   server replies `BindComplete`. `params` is a `Fields` list (§4.3); each present field is the
   parameter's value in text format (§10.4). An empty `portal`/`statement` name is the unnamed
   one.
3. `Describe` asks for metadata without running the statement (no side effects):
   - `target = 'S'` (Statement): server replies `ParameterDescription` (number of `$n`
     placeholders) then `NoData` (row shape is reported per-portal).
   - `target = 'P'` (Portal): server replies `RowDescription` (row-returning) or `NoData`.
4. `Execute` runs the portal (lazily, exactly once — caching the result) and streams up to
   `max_rows` `DataRow`s (`max_rows = 0` = all). If the cap is hit with rows remaining, the server
   sends `PortalSuspended`; a later `Execute` on the same portal resumes draining. When the
   portal drains fully the server sends `CommandComplete`. `Execute` does not repeat
   `RowDescription` — that comes from `Describe`.
5. `Sync` ends the pipeline; the server replies `ReadyForQuery`. `Sync` also clears the failed
   state (§10.2).
6. `Close` frees a statement (`'S'`) or portal (`'P'`); the server replies `CloseComplete`.

### 10.2 Error semantics — skip-until-Sync
If any extended-query message errors, the server sends an `Error`, then ignores all subsequent
extended-query messages until the next `Sync` (which replies `ReadyForQuery` and resets). This
lets a client pipeline a batch and recover cleanly at the next `Sync` boundary. A simple `Query`
also resets the failed state.

### 10.3 Result-column format codes (`Bind`)
The `result_formats` list (`0` = text, `1` = binary) selects per-column output encoding:
- empty → every column is text;
- one code → it applies to all columns;
- N codes → one per output column (a missing/extra entry defaults to text).
Any code other than `1` is treated as text. Binary fields use §11.2.

### 10.4 Parameter binding (1.0)
Parameter values in `Bind` arrive in text format (the same bytes a text `DataRow` carries).
The server substitutes each `$n` placeholder with the decoded literal before analysis. In 1.0 the
value's type is inferred from its text: integer, then float, then boolean (`true`/`t`/`TRUE`,
`false`/`f`/`FALSE`), else text. A numeric-looking value bound to a `TEXT` column would mis-infer —
bind such a value with an explicit `CAST` in the SQL. `NULL` is the `Fields` NULL marker. (Precise
type-directed binding from declared `param_types` is a forward-compatible follow-up; it will not
change this layout.)

## 11. Value formats

A `DataRow`/`Bind` field is either `NULL` (the `Fields` present-byte `0`) or a byte run in one of
two formats. The column's declared type (from `RowDescription` + the schema) tells the client how
to interpret the bytes.

### 11.1 Text format (default)
A human-readable UTF-8 rendering:

| type        | text rendering                                  |
| ----------- | ----------------------------------------------- |
| `BOOL`      | `true` / `false`                                |
| `INT`       | decimal, e.g. `-42`                             |
| `FLOAT`     | shortest round-trip decimal                     |
| `NUMERIC`   | canonical decimal text                          |
| `DATE`      | `YYYY-MM-DD`                                     |
| `TIME` / `TIMETZ` | canonical time text (`TIMETZ` keeps its entered zone: `13:45:30+07`) |
| `TIMESTAMP` / `TIMESTAMPTZ` | canonical timestamp text          |
| `INTERVAL`  | canonical interval text                         |
| `UUID`      | canonical `8-4-4-4-12` hex text                 |
| `TEXT`      | the string verbatim                             |
| `JSON`      | canonical JSON text                             |
| `ARRAY`     | canonical `{...}` braced array text             |
| `VECTOR`    | canonical `[..]` text                           |

### 11.2 Binary format (negotiated via `Bind` result-format `1`)
A compact fixed layout, big-endian throughout. `NULL` carries no bytes (only its `Fields` marker).

| type          | binary layout                                                             |
| ------------- | ------------------------------------------------------------------------- |
| `BOOL`        | 1 byte: `0x01` true, `0x00` false                                          |
| `INT`         | `i64`, 8 bytes big-endian                                                  |
| `FLOAT`       | IEEE-754 `f64` bit pattern, 8 bytes big-endian                            |
| `NUMERIC`     | canonical decimal text, UTF-8 (lossless at arbitrary precision)        |
| `DATE`        | `i32` days since 1970-01-01, 4 bytes big-endian                           |
| `TIME`        | `i64` microseconds since midnight, 8 bytes big-endian                     |
| `TIMETZ`      | `local micros:i64` ‖ `zone secs west of UTC:i32` (12 bytes, big-endian)  |
| `TIMESTAMP`   | `i64` microseconds since the epoch, 8 bytes big-endian                    |
| `TIMESTAMPTZ` | `i64` microseconds since the epoch (UTC), 8 bytes big-endian              |
| `INTERVAL`    | `months:i32` ‖ `days:i32` ‖ `micros:i64` (16 bytes, each big-endian)      |
| `UUID`        | the 16 bytes verbatim                                                      |
| `TEXT`        | UTF-8 bytes verbatim (text and binary coincide)                           |
| `JSON`        | canonical JSON text, UTF-8                                            |
| `ARRAY`       | canonical `{...}` array text, UTF-8                                   |
| `VECTOR`      | canonical `[..]` text, UTF-8                                          |

`NUMERIC`, `JSON`, `ARRAY`, and `VECTOR` use their canonical text even in binary mode because they
are arbitrary-precision / arbitrary-length: text is the lossless representation and avoids a
precision- or shape-dependent layout. Because `RowDescription` + schema disambiguate the column
type, the shared 8-byte layout of `TIME`/`TIMESTAMP`/`TIMESTAMPTZ` is unambiguous.

## 12. COPY sub-protocol

Triggered when a simple `Query` is `COPY … FROM STDIN` or `COPY … TO STDOUT`. All columns are
text format. (`COPY` over an RLS-enabled table by a non-superuser is refused with an `Error` —
fail-closed, no policy bypass.)

### 12.1 `COPY … FROM STDIN` (client uploads)
1. server → `CopyInResponse` (`G`): `[ overall_format:u8 = 0 ][ columns:u16 ]` (0 = text).
2. client → zero or more `CopyData` (`d`): `[ raw bytes ]` — each may carry any number of whole
   or partial data lines; the server reassembles.
3. client → `CopyDone` (`c`) to finish, or `CopyFail` (`f`): `[ message:Str ]` to abort.
4. server → `CommandComplete` (`COPY <count>`) on success, or `Error`, then `ReadyForQuery`.

The server caps cumulative buffered bytes at `copy_from_max_bytes` (default 1 GiB, §17). On
overflow it stops buffering, frees what it held, and keeps reading until `CopyDone`/`CopyFail` to
stay in protocol sync, then reports an `Error`. Split larger loads into multiple statements.

### 12.2 `COPY … TO STDOUT` (server downloads)
1. server → `CopyOutResponse` (`H`): `[ overall_format:u8 = 0 ][ columns:u16 ]`.
2. server → one or more `CopyData` (`d`): `[ raw bytes ]`.
3. server → `CopyDone` (`c`).
4. server → `CommandComplete` (`COPY <count>`), then `ReadyForQuery`.

### 12.3 Stray COPY/SASL messages
A `CopyData`/`CopyDone`/`CopyFail`/SASL message arriving outside its exchange is stray (a client
bug) and is harmlessly ignored — every frame is length-delimited, so dropping one cannot desync the
stream.

## 13. Cancellation (out-of-band)

To cancel an in-flight statement, a client opens a fresh connection and, in place of
`Startup`, sends:

Frontend → `CancelRequest` (type `K`): `[ pid : u32 ][ secret : u32 ]` — the `(pid, secret)`
from the target connection's `BackendKeyData` (§8).

The server looks up the target by `pid`, trips its cancel token iff `secret` matches, then
closes the cancel connection with no reply (the requester just disconnects). The targeted
statement aborts cooperatively at its next scan/loop check point, surfacing as an `Error` on the
original connection. A wrong/unknown key cancels nothing. The `secret` being a CSPRNG value is what
stops an attacker cancelling a statement it never observed.

A `CancelRequest` arriving mid-session (not as the first frame) is stray and ignored.

## 14. Errors

Backend → `Error` (type `E`): `[ code : Str ][ message : Str ]`.

- `code` is a 5-character SQLSTATE, and its first two characters — the class — are what a client
  should branch on. A fault in the query reports its own class; `XX000` (`internal_error`) is
  reserved for a fault in the server. Clients MUST read the code as 5 characters and MUST NOT
  assume a specific value within a class, since a code may be refined to a more precise one in the
  same class.

  | Class | Meaning | Examples |
  | --- | --- | --- |
  | `08` | connection exception | `08006` the connection failed while the statement was running |
  | `0A` | the server has not built this feature | `0A000` — see the note below on what this does and does not mean |
  | `21` | cardinality violation | `21000` a subquery returned more than one row where one was required |
  | `22` | data exception — a value the type cannot represent | `22012` division by zero, `22P02` unparseable text, `22007` bad date/time, `22008` date/time out of range, `22023` an argument value the function does not accept, `22P04` malformed COPY data |
  | `23` | integrity constraint violation | `23502` NOT NULL, `23505` unique, `23503` foreign key |
  | `25` | the transaction is in the wrong state for this statement | `25P02` transaction is aborted — send `ROLLBACK`; `25P01` no transaction is open; `25001` this statement cannot run here (inside a transaction block, or after a query has already run in it); `25006` the transaction is READ ONLY |
  | `26` | invalid SQL statement name | `26000` no such prepared statement — re-prepare it |
  | `2B` | dependent objects still exist | `2BP01` — resolve with `CASCADE` |
  | `2F` | SQL routine exception | `2F005` a function body finished without returning a value |
  | `34` | invalid cursor name | `34000` no such portal |
  | `3D` | invalid catalog (database) name | `3D000` |
  | `3F` | invalid schema name | `3F000` |
  | `40` | transaction rollback — **retryable** | `40001` serialization failure, `40P01` deadlock |
  | `42` | syntax error or access rule violation | `42601` syntax or a malformed statement, `42501` permission denied. **Not there:** `42P01` table, `42703` column, `42704` other named object (role, type, text-search configuration), `42883` function *or no form of it with these arguments*, `42P02` a `$n` never bound. **There but wrong:** `42809` right name, wrong kind of object, `42846` a cast between types with no conversion, `42P10` an invalid column reference, `42P16` an invalid table definition, `428C9` a value supplied for a `GENERATED ALWAYS` column. **Already there:** `42P07` table, `42701` column, `42710`/`42723` other object. **Wrong type or shape:** `42804` a type the context cannot take, `42P18` a type that cannot be inferred, `42803` an aggregate where the grouping does not allow one. Also `42602` an invalid name and `42P04` a database that already exists |
  | `44` | WITH CHECK OPTION violation | `44000` a row written through a view breaks its `CHECK OPTION` |
  | `53` | insufficient resources — **back off and retry** | `53300` too many connections; the server closes the connection after sending it |
  | `54` | program limit exceeded — the request must get smaller | `54000` a built-in limit (row cap, nesting depth, value size), `54001` recursion limit |
  | `55` | object not in a state the operation needs | `55006` object in use or reserved, `55P02` parameter cannot be changed now |
  | `57` | operator intervention | `57014` statement cancelled or timed out |
  | `58` | the environment failed, not the statement | `58030` an I/O error under the engine |
  | `P0` | raised by the user's own code | `P0001` `RAISE` from a procedure |
  | `XX` | internal error — the server's fault, not the caller's | `XX000` |

  The list is what the server emits today, not a closed set: a future version may add a class, and
  a client should treat an unrecognised one as it treats `XX`. Three notes worth having:

  - A client that retries should retry the whole class `40` — both `40001` and `40P01`, which is
    what the server's own retry helper does — and back off on `53300` rather than reconnecting at
    once. One that surfaces the error should distinguish class `42`/`22` (fix the query or the
    data) from `XX` (report it as a defect). On `25P02` there is nothing to retry until the
    transaction is ended: send `ROLLBACK` first.
  - `0A000` (`feature_not_supported`) means what it says, and a client may act on it: the statement
    is well formed and NusaDB has not built the construct, so a migration or compatibility tool can
    skip it and carry on. It is emitted **only** for that. An ordinary mistake — an unknown role, a
    call with the wrong number of arguments, a value out of range — reports its own class instead,
    so skipping on `0A000` will not skip past a statement that was simply wrong.
  - `26000` and `34000` are the two a pooler should watch: they say the *name* is gone, not that
    the statement was wrong. On `26000` re-prepare and retry; on `34000` re-issue `Bind` before
    `Execute`. `26000` now also answers the SQL-level `EXECUTE` / `DEALLOCATE` of a statement this
    session never prepared, so both spellings of the same mistake report the same code.

- `message` is human-readable.

An `Error` during a simple query is followed by `ReadyForQuery`. During extended query it triggers
skip-until-Sync (§10.2). During the handshake it precedes connection close.

## 15. Termination

Frontend → `Terminate` (type `X`): (empty) — politely closes the connection. The server also
treats a stray `Startup` mid-session as termination (rather than crashing). A clean EOF (socket
closed with no buffered partial frame) ends the connection normally; a partial frame at EOF is an
error.

## 16. Message reference

### 16.1 Frontend (client → server)

| type | name                   | payload                                                          |
| ---- | ---------------------- | --------------------------------------------------------------- |
| `S`  | Startup                | `magic:u32, major:u16, minor:u16, user:Str, database:Str`       |
| `Q`  | Query                  | `sql:Str`                                                        |
| `P`  | Parse                  | `name:Str, sql:Str, pt_count:u16, param_type_tags:bytes`        |
| `B`  | Bind                   | `portal:Str, statement:Str, params:Fields, fmt_count:u16, result_format:u16*` |
| `D`  | Describe               | `target:u8('S'/'P'), name:Str`                                  |
| `E`  | Execute                | `portal:Str, max_rows:u32`                                      |
| `Y`  | Sync                   | —                                                               |
| `C`  | Close                  | `target:u8('S'/'P'), name:Str`                                  |
| `p`  | SASLInitialResponse    | `mechanism:Str, data_len:u32, data:bytes`                       |
| `r`  | SASLResponse           | `data:bytes` (raw, frame-delimited)                             |
| `d`  | CopyData               | `data:bytes`                                                    |
| `c`  | CopyDone               | —                                                               |
| `f`  | CopyFail               | `message:Str`                                                   |
| `K`  | CancelRequest          | `pid:u32, secret:u32`                                           |
| `X`  | Terminate              | —                                                               |

### 16.2 Backend (server → client)

| type | name                  | payload                                                          |
| ---- | --------------------- | --------------------------------------------------------------- |
| `R`  | Authentication        | `sub_code:u32` + (sub-code-specific, §7.2): `AuthOk`=0; `AuthSasl`=10,`count:u16,mechanism:Str*`; `AuthSaslContinue`=11,`data:bytes`; `AuthSaslFinal`=12,`data:bytes` |
| `K`  | BackendKeyData        | `pid:u32, secret:u32`                                           |
| `Z`  | ReadyForQuery         | `status:u8('I'/'T'/'E')`                                        |
| `C`  | CommandComplete       | `tag:Str`                                                       |
| `E`  | Error                 | `code:Str(5), message:Str`                                      |
| `T`  | RowDescription        | `count:u16, column_name:Str*`                                  |
| `y`  | RowDescriptionTyped¹  | `count:u16, (column_name:Str, type_tag:u8)*`                   |
| `D`  | DataRow               | `Fields`                                                        |
| `1`  | ParseComplete         | —                                                               |
| `2`  | BindComplete          | —                                                               |
| `3`  | CloseComplete         | —                                                               |
| `t`  | ParameterDescription  | `count:u16`                                                     |
| `n`  | NoData                | —                                                               |
| `z`  | PortalSuspended       | —                                                               |
| `G`  | CopyInResponse        | `overall_format:u8(0), columns:u16`                            |
| `H`  | CopyOutResponse       | `overall_format:u8(0), columns:u16`                            |
| `d`  | CopyData              | `data:bytes`                                                    |
| `c`  | CopyDone              | —                                                               |

> Type bytes overlap across directions by design (`K`, `C`, `D`, `d`, `c`, `E`). A decoder always
> knows its direction, so there is no ambiguity. `p`/`r` are SASL inbound; `S` is Startup inbound
> only.
>
> ¹ `RowDescriptionTyped` (`y`) is protocol 1.1: sent in place of `RowDescription`
> only on a connection that negotiated effective `minor >= 1` (§2, §9.2). A `minor = 0` connection
> never receives it, so the 1.0 byte stream is unchanged.

## 17. Limits & DoS guards (server defaults)

| guard                  | default      | effect                                                                 |
| ---------------------- | ------------ | ---------------------------------------------------------------------- |
| `MAX_FRAME_LEN`        | 256 MiB      | frames over this are rejected before allocation (`FrameTooLarge`)      |
| field count (`u16`)    | 65 535       | `DataRow`/`Bind`/`RowDescription` element counts; over → `FieldTooLarge` |
| field/string length (`u32`) | 4 GiB−1 | length-prefix width; over → `FieldTooLarge`                            |
| `handshake_timeout`    | 60 s         | bounds Startup + TLS + SCRAM; a stalled handshake is dropped (anti-slowloris) |
| `copy_from_max_bytes`  | 1 GiB        | cumulative `COPY FROM` buffer cap; over → abort with `Error`          |
| `idle_timeout`         | none         | optional; closes a connection idle between requests                    |
| `statement_timeout`    | none         | optional; cooperatively cancels a long statement                       |
| `max_connections`      | none         | optional; excess connections queue in the kernel backlog               |

These are server-side; a driver does not configure them but MUST handle the resulting `Error`/close.

## 18. Conformance checklist

A conforming client/driver MUST exercise and pass:

1. Framing: round-trip every message: `encode → decode` byte-identical; reject `len < 5`,
   `len > MAX_FRAME_LEN`; handle a partial frame (read-more).
2. Startup + version: correct magic/version; observe the `Error` + close on a major mismatch.
3. Trust auth: Startup → `AuthOk` → `BackendKeyData` → `ReadyForQuery(I)`.
4. SCRAM auth: full SASL exchange against RFC 7677 vectors; verify the `v=` server signature;
   reject a forged signature; observe generic failure on wrong password.
5. Simple query: `SELECT` → `RowDescription`/`DataRow*`/`CommandComplete`/`ReadyForQuery`;
   non-row statement → `CommandComplete` only; parse the affected-row tag.
6. Extended query: `Parse`/`Bind`/`Describe`/`Execute`/`Sync`; `ParameterDescription` count;
   `PortalSuspended` on `max_rows`; resume on re-`Execute`; `Close`.
7. Skip-until-Sync: an error mid-pipeline is followed by ignored messages until `Sync`.
8. Parameters: positional `$n` binding in text format, including `NULL`.
9. Result formats: request binary (`1`) and decode every type per §11.2; mixed/all/empty
   format lists.
10. COPY: `FROM STDIN` (incl. `CopyFail` abort and the byte cap) and `TO STDOUT`.
11. Cancellation: open a second connection and `CancelRequest` an in-flight statement using the
    first connection's `(pid, secret)`; verify the target aborts and a wrong secret does nothing.
12. Errors & termination: decode `Error(code, message)`; `Terminate`; clean EOF.

A language-agnostic conformance runner drives these 12 categories, the reference client must pass
all of them, and every language driver must pass the same suite.

---

### Appendix A — implementation pointers (NusaDB tree)

| concern              | file                                            |
| -------------------- | ----------------------------------------------- |
| framing codec        | `crates/nusadb-wire/src/frame.rs`               |
| message taxonomy     | `crates/nusadb-wire/src/messages.rs`            |
| binary value codec   | `crates/nusadb-wire/src/value.rs`               |
| SCRAM / SASL         | `crates/nusadb-wire/src/auth/scram.rs`          |
| cancel registry      | `crates/nusadb-wire/src/cancel.rs`              |
| server state machine | `crates/nusadb-wire/src/server.rs`              |
| version / magic      | `crates/nusadb-wire/src/lib.rs`                 |
| parameter binding    | `crates/nusadb-sql/src/params.rs`               |
