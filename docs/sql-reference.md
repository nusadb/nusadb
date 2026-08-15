# SQL Reference

> TODO: populated as the SQL engine matures. See [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for the
> current design and scope.

## Text ordering and collation

NusaDB orders text **bytewise** (the `C` collation): `ORDER BY`, `MIN`/`MAX`, `DISTINCT`,
`GROUP BY`, and text-range comparisons all compare UTF-8 bytes directly. This is the same
semantics as running the reference engine with the `C` locale — a configuration its own
documentation recommends for performance, because locale-aware collation slows every text
sort and index comparison. Consequences to design for:

- Uppercase ASCII sorts before lowercase (`'B' < 'a'`), and multibyte characters order by
  their UTF-8 encoding, not language rules.
- Results are deterministic and locale-independent across hosts — a property linguistic
  collations do not have across library versions.
- For a case-insensitive ordering, sort an explicit expression: `ORDER BY lower(name)`.

Per-column `COLLATE` support (locale/ICU collations) is a tracked future unit; today a
`COLLATE` clause is rejected loudly rather than silently ignored.

## Truncating a DATE

`DATE_TRUNC` takes a timestamp, so a `DATE` argument is widened to midnight and the call
returns `TIMESTAMPTZ` — never `DATE`. This follows the reference engine, which picks the
time-zone-aware form whenever a conversion is needed and both forms would serve:

```sql
SELECT DATE_TRUNC('month', DATE '2024-06-15');   -- 2024-06-01 00:00:00+00
```

NusaDB compares and assigns temporal values only between matching types, so feeding that
result to a `TIMESTAMP` column or comparing it against a `TIMESTAMP` value needs an explicit
cast on either side:

```sql
INSERT INTO report (month_start)                 -- month_start TIMESTAMP
SELECT DATE_TRUNC('month', CAST(d AS TIMESTAMP)) FROM sales;
```

Comparing against a bare string literal needs no cast — `WHERE DATE_TRUNC('month', d) =
'2024-06-01'` reads the literal as the column's type.

## Fixed-width characters (`CHAR(n)`)

`CHAR(n)` stores and compares text exactly as entered — NusaDB does **not** blank-pad a
`CHAR` value out to its declared length. `'ab'` stored in a `CHAR(4)` stays `'ab'` (length 2),
so `CHAR(n)` behaves like `VARCHAR(n)` with a declared maximum, and comparisons never ignore
trailing spaces. This departs from the legacy blank-padding rule on purpose: padding is a
surprising, storage-wasting wart, and a single consistent text semantics across `TEXT`,
`VARCHAR`, and `CHAR` is easier to reason about. If an application needs a fixed-width,
space-filled rendering, pad explicitly with `rpad(col, 4)`.

## Summing 64-bit integers

`SUM` over a `BIGINT` column returns `NUMERIC`, not `BIGINT`. A total over 64-bit values can
exceed the 64-bit range, so the exact large sum is returned as `NUMERIC` rather than raised as
an overflow error. `SUM` over `INT`/`SMALLINT` still returns a 64-bit integer (which errors on
the far rarer overflow past that), and `SUM` over `NUMERIC` stays exact `NUMERIC`. The value is
exact in every case — the accumulator is a 128-bit integer for integer inputs.

## Inspecting a value's type

`nusadb_typeof(expr)` returns the static SQL type name of its argument as `TEXT` (`integer`,
`text`, `numeric`, …). The type is known at analysis time, so the argument is never evaluated.

## Array parameters to `ANY` / `ALL`

A driver commonly binds a list as a single array parameter — `WHERE id = ANY($1)` with `$1`
bound to `{1,2,3}`. The bound value arrives as its array text form (typed `TEXT`), and NusaDB
coerces it to an array of the probe's element type, exactly as an explicit `$1::INT[]` would.
The same works with an array text literal written inline:

```sql
SELECT id FROM t WHERE id = ANY('{1,3}');       -- id in (1, 3)
SELECT id FROM t WHERE id <> ALL('{1,3}');       -- id not in (1, 3)
```

An unparseable member (`'{1,x}'` against an integer probe) is rejected loudly rather than
silently dropping rows.

## `NUMERIC` division precision

`NUMERIC` division carries a fixed number of guard digits beyond its wider operand: the result
scale is `min(max(left_scale, right_scale) + 16, MAX_SCALE)`. So `10 / 3` keeps ~16 fractional
digits rather than truncating to the operands' scale. This is a fixed-scale rule (bounded,
deterministic digits) rather than a significant-digit rule; every digit it returns is correct,
and a computation needing a specific scale can `round(expr, n)` or cast to a declared
`NUMERIC(p, s)`.

## Materialized views are snapshots; maintenance is opt-in

A materialized view holds the rows computed when it was created, and they stay put until you
`REFRESH MATERIALIZED VIEW` it — the point of materializing is to hold an expensive result still.

NusaDB can also keep such a view current, adjusting it on every insert, update or delete to its base
table so it never needs a `REFRESH`. That is a per-view opt-in, because a view that quietly followed
its base table would not be doing the job you materialized it for:

```sql
CREATE MATERIALIZED VIEW big_paid WITH (incremental = true) AS
SELECT id, amount FROM orders WHERE status = 'paid' AND amount >= 100;
```

Incremental maintenance is only possible for a body that is a single base table with a projection
and an optional `WHERE` — no join, aggregate, `GROUP BY`, window, `DISTINCT`, `HAVING`,
`LIMIT`/`OFFSET`, CTE or subquery, and no volatile expression (`age()`, `now()`, …), whose stored
value would drift. Asking for it on any other body is an error rather than a silent downgrade to a
snapshot, so `WITH (incremental = true)` always means what it says. `WITH (incremental = false)`
spells out the default.

| View                                    | Freshness              | `REFRESH` needed? |
| --------------------------------------- | ---------------------- | ----------------- |
| `CREATE MATERIALIZED VIEW …`            | snapshot at last build | yes               |
| `… WITH (incremental = true)`           | always current         | no (automatic)    |

The incremental path writes only the rows that changed, so it avoids rescanning the base table; the
trade-off is a small maintenance cost on every base-table write while such a view exists.

## Assigning between `TIMESTAMP` and `TIMESTAMPTZ`

`CURRENT_TIMESTAMP` (and `now()`) is a `TIMESTAMPTZ`, so the near-universal column definition

```sql
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

needs a conversion. NusaDB applies it on assignment, in both directions: a `TIMESTAMPTZ` value
stores into a `TIMESTAMP` column and vice versa, and `CAST(expr AS TIMESTAMP)` /
`CAST(expr AS TIMESTAMPTZ)` spell the conversion out in a query. The session time zone is fixed at
UTC, so the instant is preserved exactly — only the rendering changes (a `TIMESTAMPTZ` prints its
`+00` offset, a `TIMESTAMP` does not).

*Comparison* is unchanged: the two types still do not compare without an explicit cast, so a mixed
`WHERE ts_col = tstz_col` is still rejected rather than answered from a guess.

## JSON operators

Beyond navigation (`->`, `->>`, `#>`, `#>>`) and containment (`@>`, `<@`), the `JSON` type supports:

| Operator          | Meaning                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `json \|\| json`  | merge two objects (the right side wins a shared key), concatenate two arrays, otherwise pair the operands into an array |
| `json - text`     | delete an object member by name, or every equal string element of an array |
| `json - int`      | delete an array element by position (negative counts from the end)        |
| `json - text[]`   | delete several keys at once                                              |
| `json ? text`     | does the key exist as a top-level object key, array string element, or scalar string? |
| `json ?\| text[]` | does **any** of the keys exist?                                          |
| `json ?& text[]`  | do **all** of the keys exist?                                            |

```sql
SELECT '{"a":1,"b":2}'::json || '{"b":3,"c":4}'::json;   -- {"a": 1, "b": 3, "c": 4}
SELECT '{"a":1,"b":2}'::json - 'a';                      -- {"b": 2}
SELECT '[10,20,30]'::json - (-1);                        -- [10, 20]
SELECT doc ?& ARRAY['a','b'] FROM t;                     -- both keys present?
```

The merge is shallow: `{"a":{"x":1}} || {"a":{"y":2}}` is `{"a": {"y": 2}}`, not a recursive merge.
Deleting from a scalar document, or by integer index from an object, is an error (SQLSTATE `22023`)
rather than a silently unchanged document. A text operand next to a `JSON` one is parsed as JSON, so
`doc || '{"c":3}'` needs no cast; text that is not valid JSON is rejected.

## Walking a JSON object with `jsonb_each`

`jsonb_each(json)` produces one row per top-level object member, as `(key, value)`;
`jsonb_each_text(json)` is the same with the value as `TEXT` (a string member's raw contents, a JSON
`null` as SQL `NULL`). Members come in canonical key order.

```sql
SELECT key, value FROM jsonb_each('{"b":2,"a":1}');    -- a|1 then b|2
SELECT * FROM jsonb_each(doc) AS e(k, v);              -- rename both columns
SELECT id, jsonb_each(doc) FROM t;                     -- id, key, value
```

Unlike every other set-returning function here it produces **two** columns, and the second is
appended after the projection. So in a `SELECT` list it must be the **last** item — anywhere earlier
the value column would be separated from its key by the items after it, and that shape is refused
rather than emitted. In a `FROM` item there is nothing to separate them, so the ordinary form reads
naturally. `WITH ORDINALITY` appends its counter after the value: `(key, value, ordinality)`.

A `NULL` document yields no rows. A document that is valid JSON but not an object — an array or a
scalar — is an error (SQLSTATE `22023`), not an empty result: the query asked for its members.

A `FROM` table function cannot yet reference a column of a table to its left
(`FROM t, jsonb_each(t.doc)`), so drive it from the `SELECT` list for that case.

## `TO_CHAR` numeric formats

The numeric picture accepts `9` (a digit position blanked when unused), `0` (a digit position that
forces zero-fill from itself rightward), `.` or `D` (the decimal point), `,` or `G` (a group
separator), and the `FM` prefix (fill mode, which drops the padding: the leading blanks including
the sign column, and the fraction's trailing zeros in `9` positions).

```sql
SELECT to_char(1234567.891, '9G999G999D99');   --  1,234,567.89
SELECT to_char(1234.5,     'FM9999.00');       -- 1234.50
SELECT to_char(12,         '9,999');           --     12   (no stray separator)
```

A group separator prints only when some position to its left prints something, so `9,999` renders
`12` as `    12` but `1234` as ` 1,234`. A number too wide for its integer positions renders as `#`
fill. Any other format character (`S`, `MI`, `PR`, `$`, …) is rejected rather than silently
mis-formatted.

## Session configuration variables

`SET name = value` accepts the settings the engine reads (`search_path`, `work_mem`,
`statement_timeout`, `hnsw_ef_search`, `max_autocommit_retries`), the reported connection parameters
a session may still set
(`client_encoding`, `datestyle`, `timezone`, …), and an application's own variables — which must
carry a class prefix, as in `SET myapp.request_id = '42'`. `SHOW name` and `current_setting('name')`
read one back as text, and an unset variable reads back as the empty string.

An unrecognized bare name is an error (`42704`), not a new custom variable: a misspelling such as
`SET word_mem` — or reaching for another engine's spelling of a knob, like `ef_search` instead of
`hnsw_ef_search` — fails immediately instead of reporting success and doing nothing. A read-only
parameter (`server_version`, `server_encoding`, `integer_datetimes`) reports `55P02`. A *value* a
setting cannot use is likewise rejected at `SET` time: `SET work_mem = 'huge'` fails with an
`invalid value for parameter` error rather than being stored and then ignored.

## Serialization conflicts and the auto-commit retry

Concurrency control is optimistic and locks never wait: when two sessions write the same row, one
of them is rejected with SQLSTATE `40001` immediately rather than queueing behind the other. The
rejected transaction changed nothing, and running it again against the now-committed value is the
correct response.

For a **single statement in auto-commit** — no `BEGIN` in effect — the server does that itself. Such
a statement has no intermediate results the application has seen and committed nothing, so
re-running it is exactly what a client retry loop would do, and the client never sees the conflict.
Attempts are separated by an exponential back-off with jitter, so sessions that collided once are
not released in lockstep to collide again.

Inside an explicit transaction the conflict is reported, not retried. A statement there may follow
others whose results the application already acted on, so silently re-running it would change what
those earlier results meant — only the application knows whether the whole transaction can be
replayed. Handle a class-`40` error — `40001` or `40P01` — by rolling back and retrying the
transaction; see [`transactions.md`](transactions.md) for the loop.

`SET max_autocommit_retries = N` bounds the attempts (default 50, maximum 100); `0` switches the
retry off and reports the first conflict. The budget is a bound, not a promise: a conflict that
outlives it is reported unchanged, so an application's own retry loop still works. A
`statement_timeout` or a cancel request applies to the retry sequence as a whole and ends it.

Consequences worth knowing:

- A retried statement re-runs its non-transactional side effects. `nextval` is the one to watch —
  a sequence can advance once per attempt, so gaps are possible under contention. Sequences never
  promise gapless values, but a retried statement makes gaps more likely. `RANDOM()` is the other:
  a re-run draws again, so under `SETSEED` a session's random sequence advances by an amount that
  depends on how many attempts the statement took.
- A statement whose rows have already started reaching the client is **not** retried. Results are
  streamed, and once a batch has been sent it cannot be unsaid — sending it twice would corrupt the
  result. So a large `SELECT` that conflicts part-way through reports `40001` like any other
  statement. Statements that return a count rather than rows, and results small enough not to have
  been sent yet, are unaffected.
- A statement that keeps losing occupies its connection for the length of the budget before
  reporting. `SET max_autocommit_retries = 0` restores the immediate report where that matters more
  than the automatic recovery.
- The retry budget bounds attempts, not time: the wall clock a losing statement can consume is
  each attempt's own run time plus its back-off, times the budget, and no statement deadline is set
  by default. Where connections are scarce, set the pair together — `max_autocommit_retries` for
  how hard to try, `statement_timeout` for how long the whole sequence may take — since the timeout
  covers all attempts as one.

## Access control: what is enforced

Objects are owned, and access defaults to closed: a role reaches only what it owns or has been
granted. The following are checked on every statement.

- **Reading and writing rows** — `SELECT`, `INSERT`, `UPDATE`, `DELETE` each need their own
  privilege on the table (or ownership). One does not imply another.
- **Emptying a table** — `TRUNCATE` needs the `TRUNCATE` privilege; it is not implied by `DELETE`.
- **`ANALYZE`** — needs `SELECT`, because it reads every row and stores column values.
- **Restructuring a table** — `CREATE INDEX`, `ALTER TABLE`, and `DROP TABLE` require ownership.
- **Attaching a trigger** — `CREATE TRIGGER` needs the `TRIGGER` privilege on the table.
- **Locking** — `LOCK TABLE` needs ownership or a write privilege (`UPDATE`/`DELETE`/`TRUNCATE`),
  since a lock exists to guard a write.
- **Seeing a table's columns** — `SHOW COLUMNS` needs some privilege on the table (any one), so a
  role cannot enumerate the shape of a table it has no relationship to.
- **Schemas and databases** — `DROP SCHEMA` requires ownership of the schema (its creator owns
  it); `DROP DATABASE` requires a superuser.
- **Roles** — a role may be administered only by a superuser, a `CREATEROLE` role, or a member
  holding it `WITH ADMIN OPTION`; a superuser role can be administered only by a superuser, so
  `CREATEROLE` cannot escalate itself to superuser.

### Not yet enforced (stored, and honest about it)

Some privileges parse, are stored, and appear in `information_schema`, but no statement consults
them yet. They are recorded so grants written today keep their meaning once enforcement lands;
until then, do not rely on them to restrict access:

- **`REFERENCES`** — creating a foreign key that points at another role's table is not yet gated by
  this privilege.
- **Schema `USAGE`/`CREATE`, database `CONNECT`/`CREATE`/`TEMPORARY`, and sequence
  `USAGE`/`SELECT`/`UPDATE`** — stored but not consulted at the corresponding call sites.

### Metadata enumeration

`SHOW TABLES` and the `information_schema` views list object *names* without a per-object
privilege filter (they do hide the engine's own internal catalogs). A role can therefore learn
which tables exist, though not their contents or (via `SHOW COLUMNS`) their shape without some
privilege. Per-privilege filtering of these listings is planned.
