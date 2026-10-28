# SQL Reference

Everything on this page is accepted by the current engine. Each section shows the syntax and a
runnable example; the examples are written to be pasted into `nusadb-cli` in order. Where NusaDB
deliberately behaves differently from what you may expect, that is called out in
[Behaviour worth knowing](#behaviour-worth-knowing) rather than hidden.

- [Statement index](#statement-index)
- [Data types](#data-types)
- [Creating and changing objects](#creating-and-changing-objects)
- [Inserting and changing rows](#inserting-and-changing-rows)
- [Querying](#querying)
- [Full-text search](#full-text-search)
- [Routines and triggers](#routines-and-triggers)
- [Prepared statements and cursors](#prepared-statements-and-cursors)
- [Transactions and locking](#transactions-and-locking)
- [Notifications](#notifications)
- [Maintenance](#maintenance)
- [Introspection](#introspection)
- [Functions](#functions)
- [Operators](#operators)
- [Access control](#access-control)
- [Session settings](#session-settings)
- [Error codes](#error-codes)
- [Behaviour worth knowing](#behaviour-worth-knowing)
- [Not accepted](#not-accepted)

---

## Statement index

A quick map from the statement you are looking for to the section that shows it.

| Statement | Section |
| --- | --- |
| `CREATE` / `ALTER` / `DROP TABLE`, `CREATE TEMP TABLE`, `CREATE TABLE AS`, `LIKE`, `INHERITS` | [Tables](#tables), [Altering](#altering) |
| `PARTITION BY`, `PARTITION OF`, `ATTACH` / `DETACH PARTITION` | [Partitioned tables](#partitioned-tables) |
| `CREATE` / `DROP INDEX` | [Indexes](#indexes) |
| `CREATE` / `DROP VIEW`, `CREATE MATERIALIZED VIEW`, `REFRESH` | [Views](#views-sequences-domains-schemas-databases) |
| `CREATE` / `DROP SEQUENCE`, `DOMAIN`, `TYPE`, `SCHEMA`, `DATABASE` | [Views, sequences, ...](#views-sequences-domains-schemas-databases) |
| `COMMENT ON` | [Comments](#comments) |
| `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `TRUNCATE`, `COPY` | [Inserting and changing rows](#inserting-and-changing-rows) |
| `SELECT` and all its clauses, `VALUES`, set operations | [Querying](#querying) |
| `EXPLAIN` | [Reading a plan](#reading-a-plan-explain) |
| `CREATE FUNCTION`, `CREATE PROCEDURE`, `CALL`, `DO` | [Routines and triggers](#routines-and-triggers) |
| `CREATE` / `ALTER` / `DROP TRIGGER` | [Triggers](#triggers) |
| `PREPARE`, `EXECUTE`, `DEALLOCATE`, parameters `$1..$n` | [Prepared statements](#prepared-statements) |
| `DECLARE`, `FETCH`, `MOVE`, `CLOSE` | [Cursors](#cursors) |
| `BEGIN`, `COMMIT`, `ROLLBACK`, `SAVEPOINT`, `SET TRANSACTION` | [Transactions and locking](#transactions-and-locking) |
| `DEFERRABLE`, `INITIALLY DEFERRED`, `SET CONSTRAINTS`, `ALTER CONSTRAINT` | [Deferrable constraints](#deferrable-constraints) |
| `PREPARE TRANSACTION`, `COMMIT PREPARED`, `ROLLBACK PREPARED` | [Transactions and locking](#transactions-and-locking) |
| `SELECT ... FOR UPDATE`, `LOCK TABLE` | [Row and table locks](#row-and-table-locks) |
| `LISTEN`, `UNLISTEN`, `NOTIFY` | [Notifications](#notifications) |
| `ANALYZE`, `VACUUM`, `REINDEX`, `CLUSTER`, `CHECKPOINT`, `CREATE EXTENSION` | [Maintenance](#maintenance) |
| `SHOW TABLES`, `SHOW COLUMNS`, `DESCRIBE`, `information_schema` | [Introspection](#introspection) |
| `CREATE ROLE`, `GRANT`, `REVOKE`, `SET ROLE`, `CREATE POLICY` | [Access control](#access-control) |
| `SET`, `SHOW`, `RESET` | [Session settings](#session-settings) |

---

## Data types

### Numbers

| Type | Notes |
| --- | --- |
| `SMALLINT` / `INT` / `BIGINT` | 16 / 32 / 64-bit signed integers (`INT2`, `INT4`, `INT8`, `INTEGER` are accepted spellings) |
| `REAL` / `DOUBLE PRECISION` | IEEE-754 binary floating point (`FLOAT4`, `FLOAT8`, `FLOAT`) |
| `NUMERIC(p,s)` / `DECIMAL(p,s)` | exact base-10 decimal; use this for money |

```sql
CREATE TABLE amounts (
  qty   INT,
  ratio DOUBLE PRECISION,
  price NUMERIC(12,2)
);
INSERT INTO amounts VALUES (3, 0.5, 19.99);
SELECT qty * price AS total FROM amounts;
--  total
-- -------
--  59.97
```

Integer arithmetic that overflows is an error (`22003`), not a silent wrap. Division by zero is
`22012`. Date and time arithmetic that leaves the representable range (a timestamp or date plus an
interval, an interval sum or product whose months, days or time no longer fit) is an error too
(`22008`), never a clamped or wrapped value.

### Auto-incrementing columns

`SERIAL`, `BIGSERIAL`, `SMALLSERIAL` and the standard `GENERATED { ALWAYS | BY DEFAULT } AS IDENTITY`
all create a column backed by its own sequence (`<table>_<column>_seq`). An omitted value takes the
next number; the column is implicitly `NOT NULL`.

```sql
CREATE TABLE tickets (
  id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  note TEXT
);
INSERT INTO tickets (note) VALUES ('first'), ('second') RETURNING id;
--  id
-- ----
--   1
--   2
```

### Text and binary

| Type | Notes |
| --- | --- |
| `TEXT` | variable length, no limit |
| `VARCHAR(n)` | variable length, rejected past `n` characters (`22001`) |
| `CHAR(n)` | fixed width; see [`CHAR(n)`](#fixed-width-characters-charn) |
| `BYTEA` | binary, written as `'\xDEADBEEF'` |

```sql
SELECT 'hello' || ' ' || 'world' AS greeting,
       length('héllo')           AS chars,
       octet_length('héllo')     AS bytes,
       '\xdeadbeef'::BYTEA       AS blob,
       encode('\xdeadbeef'::BYTEA, 'base64') AS b64;
```

A quote inside a literal is doubled: `'it''s'`. Dollar quoting — `$$ ... $$` or a tagged
`$tag$ ... $tag$` — is an ordinary string literal usable anywhere a quoted string is (routine
bodies included): no escape processing at all, so quotes and backslashes pass through verbatim,
and a tagged form can contain `$$` itself.

### Booleans and UUIDs

```sql
SELECT TRUE AS t,
       'f'::BOOLEAN                              AS parsed,
       gen_random_uuid()                         AS fresh,
       uuidv7()                                  AS time_ordered,
       '5b8f0b3e-2c4d-4f7a-9d1e-1a2b3c4d5e6f'::UUID AS fixed;
```

### Dates, times and intervals

| Type | Example literal |
| --- | --- |
| `DATE` | `DATE '2026-08-30'` |
| `TIME` / `TIME WITH TIME ZONE` | `TIME '14:30:00'` |
| `TIMESTAMP` | `TIMESTAMP '2026-08-30 14:30:00'` |
| `TIMESTAMPTZ` | `TIMESTAMPTZ '2026-08-30 14:30:00+07'` |
| `INTERVAL` | `INTERVAL '1 day 2 hours'`, `INTERVAL '90 minutes'`, `INTERVAL '250 ms'` |

```sql
SELECT now()                                   AS instant,
       date_trunc('month', DATE '2026-08-30')  AS month_start,
       EXTRACT(dow FROM DATE '2026-08-30')     AS day_of_week,
       DATE '2026-08-30' + INTERVAL '10 days'  AS later,
       age(TIMESTAMP '2026-08-30', TIMESTAMP '2020-01-01') AS elapsed,
       to_char(now(), 'YYYY-MM-DD HH24:MI')    AS formatted;
```

An interval literal counts any of `year`, `month`, `week`, `day`, `hour`, `minute`, `second`,
`millisecond` (`ms`, `msec`) and `microsecond` (`us`, `usec`), singular or plural, and an
`HH:MM:SS` clock. A time may give its hour with one digit (`TIME '3:16:17'`). A `DATE` read from
text that also carries a time of day (`'2024-01-01 10:00'`) keeps the date part.

Date and time arithmetic:

| Expression | Result |
| --- | --- |
| `date + integer`, `date - integer` | `DATE`, whole days later or earlier |
| `date - date` | `INTEGER`, the number of days between |
| `date + time` | `TIMESTAMP` at that time of day |
| `date + interval`, `timestamp + interval` | `TIMESTAMP` (a `TIMESTAMPTZ` stays one) |
| `timestamp - timestamp` | `INTERVAL` |
| `time + interval`, `time - interval` | `TIME`, wrapping around midnight |
| `interval * number`, `interval / number` | `INTERVAL` |

Multiplying or dividing an interval by a fractional number carries the fractions down: the months
truncate toward zero and the fractional month becomes days at 30 days per month, the days truncate
and the fractional day becomes time at 24 hours per day, and the time rounds to the microsecond.
Nothing carries upward. So `INTERVAL '1 mon' * 1.5` is `1 mon 15 days`, `INTERVAL '1 day' / 3` is
`08:00:00`, and `INTERVAL '23:59:59' * 2` is `47:59:58`. `sum` adds intervals field by field, and
`avg` divides that sum by the count with the same carry.

`DATE`, `TIMESTAMP` and `TIMESTAMPTZ` mix without a cast. A `DATE` becomes a `TIMESTAMP` at
midnight, and a `DATE` or `TIMESTAMP` becomes a `TIMESTAMPTZ` read in the session time zone, wherever
two of them meet: comparisons, `BETWEEN`, `IN`, `CASE`, `COALESCE`, `NULLIF`, `greatest`/`least`,
`ARRAY[...]`, `VALUES` and `UNION`/`INTERSECT`/`EXCEPT`. So `COALESCE(max(ts), CURRENT_TIMESTAMP)` is
a `TIMESTAMPTZ`, and `ts BETWEEN $1 AND $2` takes `DATE` bounds as midnight.

The session time zone defaults to UTC and can be changed with `SET TIME ZONE` (equivalently
`SET timezone = ...`). Supported zones are `UTC`/`GMT` and fixed offsets: a bare `'+07'` means
seven hours east, a colon form such as `'+05:30'` follows the POSIX convention (five and a half
hours *west*), and a number or `INTERVAL` value counts hours east. `SET TIME ZONE LOCAL`,
`SET TIME ZONE DEFAULT`, and `RESET timezone` restore UTC. The zone shifts how `timestamptz`
values render and how input without an explicit offset is read, along with `CURRENT_TIME`,
`CURRENT_DATE`, `LOCALTIMESTAMP`, wall-clock casts, `EXTRACT`, `date_trunc`, and `to_char`.
Region names such as `'Asia/Jakarta'` are refused: they need a time-zone database (daylight-saving
rules) that the engine does not carry. `AT TIME ZONE` accepts `UTC` and numeric offsets such as
`+07:00` under the same rule.

```sql
SET TIME ZONE '+07';
SELECT TIMESTAMPTZ '2026-08-30 12:00:00+00';  -- 2026-08-30 19:00:00+07
SELECT TIMESTAMPTZ '2026-08-30 12:00:00';     -- read as +07 wall time
SHOW timezone;                                -- <+07>-07
RESET timezone;
```

### JSON

`JSON` and `JSONB` both store a parsed value. See [How JSON is stored](#how-json-is-stored).

```sql
CREATE TABLE docs (id INT PRIMARY KEY, body JSONB);
INSERT INTO docs VALUES (1, '{"user":{"name":"ana","tags":["a","b"]},"active":true}');

SELECT body -> 'user' ->> 'name'                AS name,
       body #>> '{user,tags,0}'                 AS first_tag,
       body @> '{"active":true}'                AS is_active,
       jsonb_exists(body, 'active')             AS has_key,     -- the ? operator, as a function
       jsonb_path_query_first(body, '$.user.tags[*]') AS tag
FROM   docs;

-- Build and modify
SELECT jsonb_build_object('id', id, 'name', body -> 'user' ->> 'name') FROM docs;
UPDATE docs SET body = jsonb_set(body, '{user,name}', '"budi"') WHERE id = 1;

-- Expand: one row per element / per key (a set-returning function goes in the select list)
SELECT id, jsonb_array_elements_text(body #> '{user,tags}') AS tag FROM docs;
SELECT id, jsonb_object_keys(body) AS key FROM docs;
```

The `?` operator is not available because `?` is reserved for query parameters; use
`jsonb_exists(doc, key)` instead.

### Arrays

Arrays of any scalar type, one-dimensional or multidimensional. `INT[][]` names the same
type as `INT[]`: the declared number of dimensions is documentation, and a column of either
spelling stores an array of any depth. A multidimensional value is rectangular — every
sub-array at a level has the same length — and a ragged value such as `ARRAY[[1,2],[3]]` or
`'{{1,2},{3}}'` is refused.

```sql
CREATE TABLE posts (id INT, tags TEXT[]);
INSERT INTO posts VALUES (1, ARRAY['rust','db']), (2, '{"sql"}');

SELECT id, tags[1] AS first, cardinality(tags) AS n
FROM   posts
WHERE  tags @> ARRAY['rust'] OR 'sql' = ANY(tags);

SELECT id, unnest(tags) AS tag FROM posts;                       -- one row per element
SELECT * FROM unnest(ARRAY['a','b']) WITH ORDINALITY AS u(tag, n);

SELECT array_append(tags, 'new'), array_to_string(tags, ', '), string_to_array('a,b', ',')
FROM   posts;
SELECT ARRAY[1,2] || ARRAY[3] AS joined, ARRAY[1,2] && ARRAY[2,9] AS overlaps, ARRAY[1] <@ ARRAY[1,2] AS contained;
```

Multidimensional values nest in both the constructor and the text form, subscript one
dimension per `[i]`, and flatten through `unnest`. A subscript that stops short of a scalar
(`grid[1]` on a two-dimensional value) is `NULL`; a slice keeps the outer dimension.

```sql
CREATE TABLE boards (id INT, grid INT[][]);
INSERT INTO boards VALUES (1, ARRAY[[1,2],[3,4]]), (2, '{{5,6,7}}');

SELECT id, grid, grid[2][1], array_ndims(grid), array_dims(grid), array_length(grid, 2)
FROM   boards;                       -- {{1,2},{3,4}}  3  2  [1:2][1:2]  2

SELECT ARRAY[['1','2']]::INT[][];    -- a cast converts every leaf
```

Element search, append, overlap, containment and `= ANY` are defined over one dimension
and are refused on a multidimensional operand.

### Ranges

`INT4RANGE`, `INT8RANGE`, `NUMRANGE`, `DATERANGE`, `TSRANGE`, `TSTZRANGE`.

```sql
SELECT int4range(1, 10)                       AS r,          -- [1,10)
       int4range(1, 10, '[]')                 AS closed,
       int4range(1,10) @> 5                   AS contains,
       int4range(1,10) && int4range(5,20)     AS overlaps,
       lower(int4range(1,10))                 AS lo,
       int4range(1,5) + int4range(4,9)        AS merged,
       isempty(int4range(5,5))                AS empty;
```

### Network and MAC addresses

`INET`, `CIDR`, `MACADDR`, `MACADDR8`.

```sql
SELECT '192.168.1.5/24'::INET            AS addr,
       host('192.168.1.5/24'::INET)      AS just_host,
       masklen('192.168.1.5/24'::INET)   AS bits,
       network('192.168.1.5/24'::INET)   AS net,
       '192.168.1.0/24'::CIDR >> '192.168.1.5'::INET AS contains,
       '08:00:2b:01:02:03'::MACADDR      AS mac;
```

### Bit strings

`BIT(n)` and `BIT VARYING(n)`, with literals `B'1011'` and `X'0f'`.

```sql
SELECT B'1011' & B'1101'  AS and_,
       B'1011' | B'1101'  AS or_,
       ~B'1011'           AS not_,
       B'1011' << 1       AS shifted,
       length(B'1011')    AS len;
```

A bit string of either kind can be stored in a column of either kind. `BIT(n)` takes exactly `n`
bits (`22026` otherwise) and `BIT VARYING(n)` at most `n` (`22001` otherwise); storing a value never
pads or truncates it. An explicit cast does: `B'10101'::bit(3)` is `101`, `B'1'::bit(3)` is `100`.
The two kinds compare with each other whatever their lengths.

```sql
CREATE TABLE flags (v BIT VARYING(8), f BIT(3));
INSERT INTO flags (v) VALUES (B'101');       -- ok
INSERT INTO flags (f) VALUES (B'1010');      -- ERROR 22026
INSERT INTO flags (v) VALUES (B'101010101'); -- ERROR 22001
```

### Geometric types

`POINT`, `BOX`, `CIRCLE`, `LSEG`, `LINE`, `PATH`, `POLYGON`.

```sql
CREATE TABLE places (id INT, at POINT, area POLYGON);
INSERT INTO places VALUES (1, '(1,2)', '((0,0),(0,4),(4,4),(4,0))');

SELECT id FROM places WHERE area @> at;      -- point inside polygon
SELECT area('((0,0),(2,3))'::BOX) AS box_area, '((0,0),(3,4))'::LSEG AS segment;
```

### XML

Stored verbatim and validated as well-formed on cast or store; malformed input is a hard error.

```sql
SELECT '<a>1</a>'::XML                       AS doc,
       xml_is_well_formed('<a>1</a>')        AS ok,
       xmlconcat('<a/>'::XML, '<b/>'::XML)   AS joined;
```

### Vectors

`VECTOR(n)` with four distance operators and an HNSW index.

```sql
CREATE TABLE items (id INT PRIMARY KEY, emb VECTOR(3));
INSERT INTO items VALUES (1, '[1,0,0]'), (2, '[0,1,0]');

CREATE INDEX items_emb ON items USING hnsw (emb vector_cosine_ops);

SET hnsw_ef_search = 100;                    -- higher trades latency for recall
SELECT id, emb <=> '[1,0,0]' AS distance
FROM   items ORDER BY distance LIMIT 5;
```

Operators: `<->` L2, `<=>` cosine, `<#>` negative inner product, `<+>` L1. Operator classes:
`vector_l2_ops`, `vector_cosine_ops`, `vector_ip_ops`, `vector_l1_ops`. An index answers only the
metric its operator class declared, so build one index per metric you query; `EXPLAIN` says
whether a query used the index or fell back to an exact scan.

### Enumerated and composite types

```sql
CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');
CREATE TYPE point3 AS (x INT, y INT, z INT);

CREATE TABLE t (id INT, m mood, p point3);
INSERT INTO t VALUES (1, 'happy', '(1,2,3)');
SELECT m, p FROM t;
--    m    |    p
-- --------+---------
--  happy  | (1,2,3)

DROP TABLE t;
DROP TYPE point3;
```

An enum value outside its label list is rejected on insert (`22P02`). Enum comparison and
`MIN` / `MAX` follow the declaration order of the labels, not the label text, and mixing two
different enum types in a comparison, `CASE`, `COALESCE`, `GREATEST` / `LEAST` or `NULLIF` is
refused (`42883` / `42846`) rather than compared by position.

A composite value is written as `ROW(...)` or in its `'(a,b,c)'` text form, and a single field is
read back with `(value).field`:

```sql
CREATE TYPE pair AS (x INT, y TEXT);
CREATE TABLE cpx (id INT, p pair);
INSERT INTO cpx VALUES (1, ROW(7, 'seven')::pair), (2, '(8,eight)'::pair);

SELECT id, (p).x, (p).y FROM cpx ORDER BY id;
--  id | x | y
-- ----+---+-------
--   1 | 7 | seven
--   2 | 8 | eight

UPDATE cpx SET p = ROW(70, 'seventy')::pair WHERE (p).x = 7;
DELETE FROM cpx WHERE (p).x = 8;
```

`(value).*` expands a composite value into one column per field, named after the field. It is a
`SELECT`-list item only and takes no alias. A positional `ORDER BY n` counts the expanded columns;
a positional `GROUP BY n` counts the items as written, so it cannot name a `(value).*` item:

```sql
SELECT id, (p).* FROM cpx ORDER BY id;
--  id | x | y
-- ----+---+-------
--   1 | 7 | seven
```

The fields of every composite type are listed in `information_schema.attributes` (`udt_name`,
`attribute_name`, `ordinal_position`, `data_type`, and the length, precision and scale columns),
and the types themselves in `information_schema.user_defined_types`. Types carry no privileges, so
every role can read both.

A bare `ROW(a, b)` with no cast is an anonymous record: it renders in the canonical text form
(`(1,"a b")`), compares field by field, and its fields are read as `.f1 .. .fN`:

```sql
SELECT ROW(1, 'a b') AS r, (ROW(1, 'a')).f2 AS second;
--        r        | second
-- ---------------+-------
--  (1,"a b")     | a
SELECT (ROW(1, 'a')).f3;
-- ERROR 42703: could not identify column "f3" in record data type
```

A row value can be compared (`ROW(a, b) = ROW(1, 'x')`) and counted, but not aggregated by
`sum` / `min` / `max`.

A field can be of another composite type. In the text form the nested value is quoted like any
text, and field access chains through the levels:

```sql
CREATE TYPE inner_t AS (x INT, t TEXT);
CREATE TYPE outer_t AS (y inner_t, z INT);
CREATE TABLE nested (v outer_t);
INSERT INTO nested VALUES (ROW(ROW(1, 'a b')::inner_t, 2)::outer_t), ('("(10,x)",3)');

SELECT v, (v).y.x FROM nested;
--          v          | x
-- --------------------+----
--  ("(1,""a b"")",2)  |  1
--  ("(10,x)",3)       | 10
```

Text input is checked against every level (`'("(abc,x)",3)'` is refused), a comparison orders a
nested field by its own fields, and a type another composite type uses cannot be dropped (`2BP01`).
A field can also be any built-in type whose text form needs quoting, such as `BYTEA`, `INTERVAL` or
an array: `'("\\x01","1 day","{1,2}")'::cx` reads back as it was written. Arrays of a composite
type are not supported yet.

`ORDER BY` a composite column sorts by its text form, not field by field.

---

## Creating and changing objects

### Tables

```sql
CREATE TABLE customers (
  id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email    TEXT NOT NULL UNIQUE,
  country  TEXT DEFAULT 'ID',
  spend    NUMERIC(12,2) CHECK (spend >= 0),
  tax      NUMERIC(12,2) GENERATED ALWAYS AS (spend * 0.11) STORED,
  joined   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE orders (
  id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  customer BIGINT REFERENCES customers (id) ON DELETE CASCADE,
  total    NUMERIC(12,2),
  placed   TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT total_positive CHECK (total > 0)
);

CREATE TABLE IF NOT EXISTS archive (LIKE orders);   -- copy the shape, not the rows
CREATE TABLE archive2 (LIKE orders INCLUDING ALL);  -- also defaults, checks, keys, indexes
CREATE TABLE recent AS SELECT * FROM orders WHERE total > 100;
DROP TABLE IF EXISTS recent;
```

The bare `LIKE` copies columns with their types, widths and `NOT NULL`. The options add more:
`INCLUDING DEFAULTS` (a `SERIAL` column's default keeps drawing from the source's sequence),
`INCLUDING CONSTRAINTS` (user `CHECK`s), `INCLUDING INDEXES` (primary key, unique constraints and
secondary indexes, renamed onto the new table), `INCLUDING IDENTITY` (with a fresh sequence),
`INCLUDING GENERATED`, and `INCLUDING ALL` for everything. `COMMENTS`, `STATISTICS`, `STORAGE`
and `COMPRESSION` are accepted with nothing to copy. Foreign keys are never copied.

`CREATE UNLOGGED TABLE` is accepted; the table is stored like any other. The keyword elsewhere
trades crash durability for write speed, and this engine's write-ahead log is its only durable
copy, so the table simply keeps full durability — queries behave identically.

Constraints available: `PRIMARY KEY`, `UNIQUE`, `NOT NULL`, `CHECK`, `FOREIGN KEY ... REFERENCES`
with `ON DELETE` / `ON UPDATE` `CASCADE`, `SET NULL`, `SET DEFAULT`, `RESTRICT`, `NO ACTION`; each
can be written at column or table level. A `PRIMARY KEY` makes its columns `NOT NULL`, whether
declared with the column, as a table constraint, or added later with `ALTER TABLE ... ADD`
(which refuses the key while a row holds a NULL in one of them, on a partitioned table in any of
its partitions); the columns stay `NOT NULL` after the key is dropped, until `ALTER COLUMN ...
DROP NOT NULL`. A `CHAR(n)` value is stored without its trailing blanks, so as a key `'x'` and
`'x  '` are the same value on every write path (`INSERT`, `COPY`, `UPDATE`, `MERGE`, upserts and
the rows a foreign key action rewrites), including against keys written by earlier releases (see
[Indexes over CHAR columns from earlier releases](#indexes-over-char-columns-from-earlier-releases)).
`CHECK` constraints, generated columns, triggers and `RETURNING` see the stored value, without
the blanks.
Generated columns are `STORED`. A unique form that treats NULLs as equal:

```sql
CREATE TABLE nnd (a INT, b TEXT, UNIQUE NULLS NOT DISTINCT (a));
INSERT INTO nnd (a) VALUES (NULL);
INSERT INTO nnd (a) VALUES (NULL);
-- ERROR 23505: with NULLS NOT DISTINCT, NULLs count as equal; plain UNIQUE allows both
```

`UNIQUE NULLS NOT DISTINCT` treats NULL keys as equal, so at most one NULL row exists per key; it
also works through `ALTER TABLE ... ADD`.

### Exclusion constraints

An `EXCLUDE` constraint says that no two rows may have every listed operator hold between them.
Each element is a column or an expression with `WITH =` (equal) or `WITH &&` (range overlap,
for `int4range`, `int8range`, `numrange`, `daterange`, `tsrange` and `tstzrange` values):

```sql
CREATE TABLE hist (
  hist_id    UUID PRIMARY KEY,
  k          INT NOT NULL,
  valid_from TIMESTAMPTZ NOT NULL,
  valid_to   TIMESTAMPTZ,
  CONSTRAINT hist_no_overlap
    EXCLUDE USING gist (k WITH =, tstzrange(valid_from, valid_to) WITH &&)
);
INSERT INTO hist VALUES (gen_random_uuid(), 1, '2024-01-01', '2024-02-01');
INSERT INTO hist VALUES (gen_random_uuid(), 1, '2024-02-01', NULL);   -- ok: [) bounds only touch
INSERT INTO hist VALUES (gen_random_uuid(), 1, '2024-01-15', '2024-01-20');
-- ERROR 23P01: conflicting key value violates exclusion constraint "hist_no_overlap": ...

-- Partial: only rows matching the parenthesized WHERE take part.
CREATE TABLE booking (room INT, during INT4RANGE, cancelled BOOL NOT NULL DEFAULT false,
  EXCLUDE (room WITH =, during WITH &&) WHERE (NOT cancelled));

ALTER TABLE booking ADD CONSTRAINT one_per_room EXCLUDE (room WITH =, during WITH &&);
```

The rules:

- A row with a NULL element, or an empty range, never conflicts. Bounds are respected: `[1,5)`
  and `[5,9)` do not overlap.
- `USING gist` and `USING btree` are both accepted and behave the same; `gist` is the default.
- The check runs on `INSERT`, `UPDATE`, `MERGE`, `COPY` and both upsert arms. `ON CONFLICT DO
  NOTHING` skips a row that would conflict with an existing row or with an earlier row of the same
  statement, as it does for a `UNIQUE` key; `ON CONFLICT ... DO UPDATE` needs a `UNIQUE` arbiter,
  and a row that still violates the exclusion constraint fails with `23P01`. An `UPDATE` that moves
  a row's range is checked against every other row.
- `ALTER COLUMN ... TYPE` checks the constraint again over the converted rows, since the new type
  can make distinct values equal (`1.4` and `1.2` as `INT`); a conflict refuses the change with
  `23P01`.
- `ALTER TABLE ... ADD CONSTRAINT ... EXCLUDE` checks the rows already in the table and fails with
  `23P01` if two of them conflict. `DROP CONSTRAINT` removes it.
- The check sees every committed row, whatever the isolation level. Two transactions writing rows
  with the same `=` values at the same time cannot both succeed: the second fails at once with the
  retryable `40001` (also when their ranges would not have overlapped), or with `23P01` once the
  first has committed. A constraint with no `=` element lets one writer at a time through. An
  `UPDATE` that changes a row's `=` values locks only the new values: the row's old version stays
  visible to everyone else until the update commits, so a writer conflicting with it is still
  refused.
- Elements and the `WHERE` predicate may call only immutable functions (`42P17` otherwise).
- Cost: the rows a statement writes are grouped by their `=` values and, with one `&&` element,
  swept in range order, so a group of `n` rows costs `O(n log n)` (with several `&&` elements a
  group is compared pairwise). When every `=` element is a plain column, a b-tree index over those
  columns, named after the constraint, is created with it and dropped with it, and each distinct
  `=` group of a statement reads only its own rows (all of them: the index narrows by `=` value,
  not by range, so writing into one very large group costs time in proportion to the group). `COPY` and a streamed `INSERT ... SELECT`
  take their locks batch by batch and check the whole statement once at the end, holding one key
  per written row in memory. Without such an index (a `=` element that is an
  expression or a floating-point or `NUMERIC` column, which may hold `NaN`, or no `=` element at
  all) each writing statement reads the whole table once. If an index of the constraint's name
  already exists, the constraint leaves it alone and creates none (an index on exactly the `=`
  columns is still used).
- `information_schema.table_constraints` lists it with `constraint_type` `EXCLUDE`.
- When every element is a plain column `WITH =` and there is no `WHERE`, the constraint is exactly
  `UNIQUE` on those columns and is created as one:

```sql
CREATE TABLE bookings (room INT, day DATE, EXCLUDE (room WITH =, day WITH =));
```

Other operators, other access methods, `INCLUDE`, and an `EXCLUDE` on a partitioned table
(declare it on each partition) are refused with `0A000`. An exclusion constraint can be
`DEFERRABLE` (see [Deferrable constraints](#deferrable-constraints)).

### Deferrable constraints

A `PRIMARY KEY`, `UNIQUE`, `FOREIGN KEY` or `EXCLUDE` constraint may be declared `DEFERRABLE`, at
column or table level. Its check can then wait until the transaction commits, so a transaction may
pass through states that break the constraint as long as the state it commits keeps it:

```sql
CREATE TABLE slot (
  n    INT CONSTRAINT slot_n UNIQUE DEFERRABLE INITIALLY DEFERRED,
  name TEXT
);
INSERT INTO slot VALUES (1, 'a'), (2, 'b');
BEGIN;
UPDATE slot SET n = 2 WHERE name = 'a';   -- two rows hold 2 for now
UPDATE slot SET n = 1 WHERE name = 'b';
COMMIT;                                   -- checked here: fine

-- A key that points at its own table: the row it needs may arrive later in the transaction.
CREATE TABLE version (
  id   INT PRIMARY KEY,
  prev INT REFERENCES version (id) DEFERRABLE INITIALLY DEFERRED
);
```

- `NOT DEFERRABLE` (the default) checks the constraint in each statement that writes.
  `DEFERRABLE INITIALLY IMMEDIATE` does the same until the transaction defers it;
  `DEFERRABLE INITIALLY DEFERRED` (or just `INITIALLY DEFERRED`) starts each transaction deferred.
  `NOT DEFERRABLE INITIALLY DEFERRED` is a syntax error (`42601`). `CHECK` and `NOT NULL` are
  never deferrable. `ENFORCED` is accepted; `NOT ENFORCED` is refused with `0A000`.
- `SET CONSTRAINTS {ALL | name [, ...]} {DEFERRED | IMMEDIATE}` changes the mode for the rest of
  the transaction. A later name overrides an earlier `ALL`, and a later `ALL` replaces every
  earlier name. A name may be schema-qualified; without a qualifier it is looked up in the schemas
  of the `search_path`, and the name of a partitioned table's constraint reaches the copies on its
  partitions. An unknown name is `42704`, a constraint that is not deferrable `55000`. Outside a
  transaction block the statement only raises a warning.
- `SET CONSTRAINTS ... IMMEDIATE` checks what the named constraints deferred so far, at once. A
  violation fails the statement (and the transaction block, until `ROLLBACK` or `ROLLBACK TO
  SAVEPOINT`).
- A deferred check runs at `COMMIT` (and `END`), at `PREPARE TRANSACTION`, and at the end of an
  auto-committed statement. A violation then rolls the whole transaction back and reports the
  constraint's usual code: `23505` for a key, `23503` for a foreign key, `23P01` for an exclusion
  constraint. The check reads the state being committed, so a duplicate a later statement removed
  again, or a child whose parent arrived later, is fine.
- `ROLLBACK TO SAVEPOINT` forgets what was deferred after the savepoint and restores the
  `SET CONSTRAINTS` mode the savepoint had.
- For a foreign key, only the `NO ACTION` check is deferred: a child row may wait for its parent,
  and a parent `DELETE` or key `UPDATE` may leave children that a later statement fixes. `RESTRICT`
  is always checked at once, and the `CASCADE`, `SET NULL` and `SET DEFAULT` actions run when the
  parent row changes.
- `ON CONFLICT` cannot use a deferrable constraint as its arbiter (`0A000`). A bare
  `ON CONFLICT DO NOTHING` does not skip rows on account of a deferrable constraint: such a row is
  written, and the constraint checks it as usual.
- Concurrency is the same as for an immediate check: a deferred write takes its key locks when it
  writes, and keeps them to the end of the transaction. A second transaction writing the same key
  fails at once with `40001`; one that committed first is seen by the check at `COMMIT`. A child
  row holds its parent key, so a concurrent delete of that parent (or an insert of it) fails with
  `40001` until the child's transaction ends. `40001` can also be the answer to `COMMIT`.
- Cost: a deferred write keeps each key it wrote in memory until the transaction ends; past
  100,000 keys for one constraint it checks the whole table once at commit instead. At commit, each
  key is looked up through the constraint's index when there is one.
- `ALTER TABLE t ALTER CONSTRAINT name [NOT] DEFERRABLE [INITIALLY DEFERRED | INITIALLY IMMEDIATE]`
  changes the declared mode of a key, foreign key or exclusion constraint (and of its copies on a
  partitioned table's partitions). A `CHECK` is refused with `42809`.
- `information_schema.table_constraints` reports `is_deferrable` and `initially_deferred`
  (`YES` / `NO`) and `enforced` (always `YES`).

Temporary tables belong to the session that made them, are invisible to others, and disappear when
the session ends. `ON COMMIT DELETE ROWS` and `ON COMMIT DROP` are accepted.

```sql
CREATE TEMP TABLE scratch (id INT) ON COMMIT DROP;
```

### Partitioned tables

A table can be split by ranges, by value lists, or by hash. The parent stores no rows: an insert
into it is routed to the matching partition, a query on it reads the partitions the `WHERE` clause
cannot rule out, and a row that fits no partition is refused.

```sql
CREATE TABLE events (id INT, region INT) PARTITION BY RANGE (region);
CREATE TABLE events_lo PARTITION OF events FOR VALUES FROM (0)   TO (100);
CREATE TABLE events_hi PARTITION OF events FOR VALUES FROM (100) TO (200);

INSERT INTO events VALUES (1, 50), (2, 150);      -- routed to events_lo / events_hi
SELECT id, region FROM events ORDER BY id;        -- reads both partitions
SELECT id FROM events WHERE region = 42;          -- prunes: only events_lo is scanned

INSERT INTO events VALUES (3, 500);
-- ERROR 23514: no partition of relation "events" found for the inserted row
INSERT INTO events_lo VALUES (4, 150);
-- ERROR 23514: the inserted row does not belong to partition "events_lo"
```

The three strategies, plus a catch-all partition:

```sql
CREATE TABLE traffic (id INT, region TEXT) PARTITION BY LIST (region);
CREATE TABLE traffic_asia PARTITION OF traffic FOR VALUES IN ('ID', 'SG', 'JP');
CREATE TABLE traffic_rest PARTITION OF traffic DEFAULT;
INSERT INTO traffic VALUES (1, 'ID'), (2, 'US'), (3, NULL);   -- 2 and 3 land in traffic_rest

CREATE TABLE loads (id INT, v TEXT) PARTITION BY HASH (id);
CREATE TABLE loads_0 PARTITION OF loads FOR VALUES WITH (MODULUS 3, REMAINDER 0);
CREATE TABLE loads_1 PARTITION OF loads FOR VALUES WITH (MODULUS 3, REMAINDER 1);
CREATE TABLE loads_2 PARTITION OF loads FOR VALUES WITH (MODULUS 3, REMAINDER 2);
```

A range key may span several columns, compared as a tuple, and a partition may itself be
partitioned:

```sql
CREATE TABLE metrics (y INT, mo INT, v TEXT) PARTITION BY RANGE (y, mo);
CREATE TABLE metrics_q2 PARTITION OF metrics FOR VALUES FROM (2026, 4) TO (2026, 7);

CREATE TABLE logs (y INT, mo INT) PARTITION BY RANGE (y);
CREATE TABLE logs_2026 PARTITION OF logs
  FOR VALUES FROM (2026) TO (2027) PARTITION BY RANGE (mo);
CREATE TABLE logs_2026_h1 PARTITION OF logs_2026 FOR VALUES FROM (1) TO (7);
```

An existing table with matching columns can be attached, and a partition detached into an
ordinary table:

```sql
CREATE TABLE events_archive (id INT, region INT);
ALTER TABLE events ATTACH PARTITION events_archive FOR VALUES FROM (200) TO (300);
ALTER TABLE events DETACH PARTITION events_archive;
```

Worth knowing:

- Range bounds are constant literals, half-open `[lo, hi)`, compared per tuple, so a multi-column
  key exactly on a boundary belongs to the upper partition. Overlap, `lo >= hi`, wrong bound
  arity, and a bound kind that does not match the parent's strategy are refused at creation, and
  `ATTACH` scans the table first and refuses if an existing row falls outside the bound.
- The `DEFAULT` partition takes every row no sibling accepts, including a NULL key (a NULL key
  with no default partition is refused, since no range, list or hash bound matches NULL). A hash
  parent takes no default, and a new sibling whose bound overlaps rows already in the default is
  refused.
- `UPDATE` and `DELETE` on the parent reach every partition; `ONLY` restricts them to the parent
  itself. An `UPDATE` through the parent that changes a row's partition key MOVES the row to the
  accepting partition (performed as a delete plus a routed insert: the destination enforces
  uniqueness, and the source's `DELETE` plus the destination's `INSERT` row triggers fire — never
  `AFTER UPDATE`). A direct `UPDATE` on a partition refuses the move (`23514`): the named target
  must keep the row, exactly like the reference engine. `ON CONFLICT DO UPDATE` may not move a
  row either.
- The partition key must be written on every insert; the planner prunes partitions using `WHERE`
  conditions of the shape `key = / < / <= / > / >= constant` and `key BETWEEN a AND b`, which shows
  in `EXPLAIN` as fewer scanned partitions.
- `DROP TABLE` on the parent drops the whole partition tree with it.
- Parent and partitions may live in different schemas; qualified names work everywhere.

A partitioned parent may declare a `PRIMARY KEY` / `UNIQUE` constraint. Each partition inherits it
at `CREATE TABLE ... PARTITION OF` and at `ATTACH PARTITION` (existing rows are validated first,
and a nullable primary-key column in the attached table is refused). When the key includes every
partition-key column, equal keys always route to the same partition and each partition's copy
enforces it alone. `INSERT ... ON CONFLICT` works on the parent: rows route to their partition
first, then the upsert action runs against that leaf. A `DO UPDATE` that would move the row out of
its partition is refused; a plain `UPDATE` through the parent moves it (see the partitioning
section).

#### Keys across partitions

A key that leaves out a partition-key column (or any key of a table partitioned by an expression)
is accepted too, and enforced across every partition below the table: no two rows of any of its
partitions may share the key. Creating such a key raises a notice, because of what it costs: every
write of the key looks it up in every partition.

```sql
CREATE TABLE events (
  id         UUID PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  payload    TEXT
) PARTITION BY RANGE (created_at);
-- NOTICE: unique constraint "events_pkey" on partitioned table "events" does not include every
--         partitioning column, so it is enforced across all of its partitions: every write of
--         its key probes every partition
CREATE TABLE events_2026_01 PARTITION OF events FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE events_2026_02 PARTITION OF events FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

INSERT INTO events VALUES ('6f1c8a52-0d0e-4d33-9a51-1f6b2a0c9e11', '2026-01-05', 'a');
INSERT INTO events VALUES ('6f1c8a52-0d0e-4d33-9a51-1f6b2a0c9e11', '2026-02-05', 'b');
-- ERROR 23505: duplicate key violates primary key constraint "events_pkey" on (id)
```

- Every way of writing a row is checked: `INSERT` through the parent or into a partition,
  `INSERT ... SELECT`, `COPY`, `MERGE` into a partition, an `UPDATE` that changes the key or moves
  the row to another partition, a foreign key's `ON UPDATE CASCADE` or `ON DELETE SET NULL` that
  rewrites the key, and statements inside functions, procedures and triggers.
- The check runs when the statement ends, so one statement may move keys between partitions: an
  `UPDATE` that swaps two keys held in different partitions succeeds when the result is valid. A
  statement inside a function or procedure is checked when it ends, so an `EXCEPTION` block catches
  the error.
- `ON CONFLICT DO NOTHING` skips a row whose key another partition holds. `ON CONFLICT (key) DO
  UPDATE` on such a key updates the conflicting row in the partition that holds it, through the
  parent or when the statement names a partition. The key columns must be written by the statement
  for the conflict to be found there; a key filled by a default is checked like a plain insert.
  A conflicting row the statement's snapshot cannot see (one committed after the snapshot, which
  under `REPEATABLE READ` and `SERIALIZABLE` is the transaction's) is never updated: the statement
  fails, with `23505` when that row sits in the partition the new row routes to, exactly as on a
  table that is not partitioned, and with `40001` when it sits in another partition. Either way
  the statement can be retried.
- `ALTER TABLE ... ADD` such a key checks the rows of all partitions together. `ATTACH PARTITION`
  checks the attached table's keys against the other partitions (and fails with `40001` while
  another transaction is writing one of them). `DETACH PARTITION` needs no check: the detached
  table keeps its own copy of the key.
- Sub-partitioned tables are covered: a key that includes the top-level partition key but not the
  key of a partition that is itself partitioned is enforced across that partition's partitions.
- `DEFERRABLE` works as for any key (see [Deferrable constraints](#deferrable-constraints)): a
  deferred key is checked across the partitions at `COMMIT` or `SET CONSTRAINTS ... IMMEDIATE`.
- Refused with `0A000`: such a key over a column the index cannot hold exactly (`FLOAT`, `REAL`
  and `NUMERIC`, whose equal values can differ in the index, and `INTERVAL`, array, `VECTOR`,
  `INET`, `CIDR`, `BIT`, `BIT VARYING`, range, geometry, `TSVECTOR`, `TSQUERY` and `XML`, which
  have no index form), and `NULLS NOT DISTINCT`. Rows with a NULL key column never conflict. A
  `FOREIGN KEY` still cannot reference a partitioned table.

Concurrent writers are kept apart by a lock on the key, taken for the whole partitioned table and
held to the end of the writer's transaction. A second transaction writing the same key into any
partition fails at once with `40001`, and one that committed first is seen by the check at every
isolation level, including in a partition created or attached after a `REPEATABLE READ`
transaction began.

The price is one index lookup per partition for each key written, plus one read of the partition
list per statement; a statement that writes many keys into a small partition reads that partition
once instead. A key that includes the partition key costs nothing extra, and neither does a table
that is not partitioned.

A `CHECK` or `FOREIGN KEY` on a partitioned parent holds for every partition the same way,
declared in `CREATE TABLE` or added later with `ALTER TABLE ... ADD CONSTRAINT`, which also checks
the rows already in the partitions (and refuses the statement when one breaks it). Each partition
carries a copy named after it (`fact_v_check` on `fact` is `fact_01_v_check` on `fact_01`): the
copy cannot be dropped on its own (`42P16`), dropping the constraint on the parent drops every
copy, and a partition that is detached keeps its copies as its own constraints. A `UNIQUE` or
`PRIMARY KEY` added this way that leaves out a partition-key column is enforced across the
partitions, as above.

A range bound element may be `MINVALUE` or `MAXVALUE` to leave that side open —
`FROM (MINVALUE) TO (0)` holds every key below zero, `FROM (10) TO (MAXVALUE)` everything from
ten up, and a multi-column bound can pin the leading column and leave the rest open
(`FROM (1, MINVALUE) TO (1, MAXVALUE)`). Every element after a marker must repeat that marker
(`42804`), and a bound whose lower side is not strictly below its upper is an empty range
(`42P17`).

A partition key part may also be a parenthesized expression — `PARTITION BY RANGE ((k % 10))`,
`HASH ((a + b))`, or mixed with columns as in `RANGE (a, (b * 2))`. The expression must be
immutable (a key that can change would route the same row differently each time; `42P17`, see
[Volatility](#volatility)), every column it
reads must be provided in the `INSERT` (routing does not evaluate defaults), and a unique
constraint never covers an expression key, so one declared on such a parent is enforced across its
partitions.

Not accepted: `LIST` with more than one key column, non-literal bounds, a partition declaring
its own columns, and a `FOREIGN KEY` that references a partitioned table.

### Inherited tables

`INHERITS` gives a table its parents' columns in front of its own, and a query on the parent also
reads every descendant. `ONLY` reads just the named table.

```sql
CREATE TABLE cities   (name TEXT, pop INT);
CREATE TABLE capitals (state TEXT) INHERITS (cities);   -- columns: name, pop, state

INSERT INTO cities   VALUES ('Bandung', 2500000);
INSERT INTO capitals VALUES ('Jakarta', 10000000, 'JK');

SELECT name FROM cities ORDER BY name;        -- Bandung, Jakarta
SELECT name FROM ONLY cities;                 -- Bandung

UPDATE cities SET pop = pop + 1;              -- reaches capitals too
UPDATE ONLY cities SET pop = 0;               -- just the parent's own rows
DELETE FROM cities WHERE pop = 0;             -- likewise propagates; ONLY restricts
```

A child may redeclare an inherited column only with the same type; `NOT NULL` holds only if every
copy has it. Dropping a parent that still has children is refused (`2BP01`); `DROP TABLE ...
CASCADE` drops the descendants with it. Row-level security applies per table, so each descendant's
policies govern its own rows.

### Altering

One action per `ALTER TABLE` statement.

```sql
ALTER TABLE orders ADD COLUMN note TEXT;
ALTER TABLE orders ADD COLUMN channel TEXT DEFAULT 'web' NOT NULL;
ALTER TABLE orders ADD COLUMN qty INT NOT NULL DEFAULT 1 CHECK (qty > 0);
ALTER TABLE orders ADD COLUMN region INT REFERENCES regions (id);
ALTER TABLE orders ADD COLUMN gross NUMERIC(12,2) GENERATED ALWAYS AS (total * 1.11) STORED;
ALTER TABLE orders ALTER COLUMN note SET DEFAULT '';
ALTER TABLE orders ALTER COLUMN note DROP DEFAULT;
ALTER TABLE orders ALTER COLUMN note SET NOT NULL;
ALTER TABLE orders ALTER COLUMN note DROP NOT NULL;
ALTER TABLE orders ALTER COLUMN note TYPE VARCHAR(200);
ALTER TABLE orders RENAME COLUMN note TO memo;
ALTER TABLE orders DROP COLUMN memo;
ALTER TABLE orders DROP COLUMN channel RESTRICT;   -- RESTRICT is the default
ALTER TABLE orders DROP COLUMN region CASCADE;
ALTER TABLE orders RENAME TO sales_orders;
ALTER TABLE sales_orders RENAME TO orders;

ALTER TABLE orders DROP CONSTRAINT total_positive;
ALTER TABLE orders ADD CONSTRAINT total_positive CHECK (total > 0);
ALTER TABLE orders ALTER CONSTRAINT orders_customer_fkey DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer) REFERENCES customers (id);
ALTER TABLE orders DROP CONSTRAINT orders_customer_fk;
ALTER TABLE orders DROP CONSTRAINT IF EXISTS orders_customer_fk;

ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders DISABLE ROW LEVEL SECURITY;
```

Enabling or disabling row-level security is reserved to a superuser, even for the table's owner:
the toggle governs whose rows every role can see, so it is administered centrally rather than left
to whoever happens to own the table. A non-superuser owner is refused with `42501`.

A column added with inline constraints (`CHECK`, `REFERENCES`, `UNIQUE`) gets them
as if each had been added with `ADD CONSTRAINT` right after the column: the existing rows, with the
column's `DEFAULT` filled in, must satisfy them, or the statement leaves nothing behind. An added
column keeps the limits of its type, as a column declared in `CREATE TABLE` does: the range of a
`TINYINT`, `SMALLINT`, `MEDIUMINT` or `INTEGER`, and the length of a `VARCHAR(n)` or `CHAR(n)`.

On a partitioned table, or a table with inheritance children, a column change (`ADD`, `DROP`,
`RENAME`, `TYPE`, `SET`/`DROP NOT NULL`, `SET`/`DROP DEFAULT`) applies to every table below it, so
their rows stay readable through it. A partition takes all its columns from its parent: adding,
dropping, renaming or retyping one on the partition is refused (`42P16`). An inheritance child may
add columns of its own, but cannot drop, rename or retype one its parent has, and neither kind of
child can drop a `NOT NULL` its parent declares. A column in a partition key (at any level) cannot
be dropped or retyped; renaming it renames it in the key too, unless the key is an expression
that names it.

`ALTER TABLE t DISABLE TRIGGER name` and `ENABLE TRIGGER name` are shown under [Triggers](#triggers).

`DROP COLUMN` also drops what the table itself defines over the column, whether the column is
its only one or one of several: indexes (including expression and partial indexes that read it),
`PRIMARY KEY`, `UNIQUE` and `CHECK` constraints, foreign keys whose own columns include it, a
`USING hnsw` vector index on it, its column-level grants, and generated columns computed from it.
A `SERIAL` or identity column's sequence goes with it. Without `CASCADE` (or with `RESTRICT`),
the drop is refused with `2BP01` while something outside the table still needs the column: a
foreign key on another table (or a self-referencing one) that references it, or a view,
materialized view, row-level-security policy or trigger (on this table or any other) whose SQL
reads it. A view that reads the table through `*`, `t.*` or a `NATURAL` join counts as reading
every column of it. With `CASCADE` those are dropped too; the referencing table itself and its rows
stay. A function or procedure whose body reads the column refuses the drop even with `CASCADE`:
replace or drop it first. Dropping the last column of a table, counting the generated columns that
would go with it, is refused (`42P16`). A `USING hnsw` index on a later column of a table whose name
also exists in another schema refuses the drop (`0A000`); drop that index first.

`ALTER COLUMN ... DROP DEFAULT` on a `SERIAL` or identity column also drops the sequence it drew
from, unless another column's default still uses it (a table made with `LIKE ... INCLUDING
DEFAULTS` shares its source's sequence, and a `DEFAULT nextval('...')` names it).

Dropping a sequence (directly or with its column or table) is transactional. Until the dropping
transaction ends, another transaction creating a sequence of the same name is refused with
`40001`, since a rollback would bring the dropped one back; retry once the dropper is done.

`ALTER COLUMN ... TYPE` replaces the range or length the old type enforced with the new type's:
widening `SMALLINT` to `INTEGER` or `BIGINT`, or `VARCHAR(10)` to `VARCHAR(50)` or `TEXT`, lifts
the old bound. Narrowing checks the rows already there, and a value the new type cannot hold
refuses the change (`22003` for an integer out of range, `22001` for a string that is too long).

`RENAME COLUMN` carries everything the table records under the column's name to the new name: its
`DEFAULT` (including a `SERIAL` or identity sentinel; the backing sequence keeps its name), its
type-range check, its indexes (plain, multi-column, expression and partial), `PRIMARY KEY`,
`UNIQUE` and `CHECK` constraints, foreign keys on either side, generated expressions that read
it, column-level grants, and its enum or composite type. A view, materialized view, policy,
trigger, function or procedure keeps its SQL as text, which the rename does not rewrite: while one
of them names the column, the rename is refused (`2BP01`) and the error names it. Drop that
object, rename, and recreate it.

Every column change is transactional: a `ROLLBACK` (or a `ROLLBACK TO SAVEPOINT`) restores the
column, its data and everything the change dropped or moved, sequences included.

`DROP CONSTRAINT` of a constraint the table does not have is an error (`42704`); with `IF EXISTS`
it does nothing.

`RENAME TO` keeps the table's schema and takes everything filed under the table's name with it:
ownership and grants (table- and column-level), row-level security and its policies, triggers,
column defaults (`DEFAULT`, `SERIAL`, `IDENTITY`, `GENERATED`) and the enum or composite type of
its columns. A table later created under the old name starts with none of them. The rename is
refused, naming the object, while a view, materialized view, function, procedure, policy predicate
or trigger body spells the table's name, while an incrementally maintained view or a vector index
is built on it, or while it is a partition, a partitioned parent, or part of an inheritance chain:
those find the table by name and would stop resolving. Drop that object, rename, and recreate it.

`ADD COLUMN ... GENERATED ALWAYS AS (expr) STORED` computes the column for every existing row as
part of the statement, and every later insert or update recomputes it; an explicit value for it is
refused (`428C9`). On a partitioned or inherited table the column is added to every partition and
child the same way. The expression follows the same rules as in `CREATE TABLE`: it reads only the
table's ordinary columns and calls only immutable functions.

Not accepted: `ADD COLUMN ... FIRST | AFTER`, `ADD COLUMN ... PRIMARY KEY` (add the column, then
`ADD CONSTRAINT ... PRIMARY KEY`), `ADD COLUMN` with `SERIAL` or `GENERATED AS IDENTITY` (`0A000`),
dropping several columns in one statement, `ADD CONSTRAINT ... NOT VALID`, `ALTER COLUMN ... TYPE ... USING <expr>`,
`ALTER COLUMN ... ADD GENERATED AS IDENTITY`, and a comma-separated list of actions.

### Indexes

```sql
CREATE INDEX orders_customer ON orders (customer);
CREATE INDEX orders_big      ON orders (customer) WHERE total > 1000;   -- partial
CREATE INDEX orders_lower    ON customers ((lower(email)));             -- expression
CREATE UNIQUE INDEX customers_email ON customers (email);   -- an index needs a name
CREATE INDEX orders_desc     ON orders (total DESC);
CREATE INDEX IF NOT EXISTS orders_total ON orders (total) INCLUDE (customer);
CREATE INDEX orders_btree ON orders USING btree (customer);      -- btree is the default anyway
CREATE INDEX orders_nulls ON orders (total DESC NULLS FIRST);
CREATE UNIQUE INDEX orders_ref ON orders (ref) NULLS NOT DISTINCT;  -- at most one NULL ref
DROP INDEX IF EXISTS orders_desc;
```

Index methods: the scalar structure is always the B-tree. `USING btree` names it explicitly;
`USING hash` and `USING brin` record the declared method with the same tree answering the scans
(it covers at least what those methods promise); `USING gin`, `gist` and `spgist` back onto the
default tree; `USING hnsw` selects the vector index. An unknown method is an error (`42704`). A per-key `NULLS FIRST` / `NULLS LAST` is accepted; it does not change results, because a
query whose ordering column is nullable is served by an explicit sort that follows the query's own
`NULLS` clause. The planner picks an index from
collected statistics, so run `ANALYZE` after a large load (see [Maintenance](#maintenance)).

An expression index serves a `WHERE` comparison against a constant on exactly the same expression
(`lower(email) = 'a@x.org'` uses `orders_lower` above; `upper(email) = ...` does not). There are no
statistics on an expression, so an equality is assumed selective and takes the index, while a range
on it is assumed to keep a third of the rows and, once the table is analyzed, runs as a sequential
scan. A key expression or predicate that calls a function which is not immutable, such as `now()`,
`random()` or `uuidv7()`, is refused when the index is created (`42P17`, see
[Volatility](#volatility)). An expression involving a `TIMESTAMPTZ` or `TIMETZ` value is accepted
but never used to answer a query, because the session's `TimeZone` setting changes its value
(`CAST(ts AS DATE)`, `extract(hour FROM ts)`), so the key stored when a row was written need not be
the value a later query computes. Index such a value through a stored generated column instead. A partial index keeps only the rows that satisfy its predicate,
so it is maintained and enforces uniqueness but is never used to answer a query.

#### Unique indexes and NULL keys

A unique index treats NULL keys as distinct by default (`NULLS DISTINCT`): any number of rows may
have a NULL in a key column, and only rows whose whole key is non-NULL can collide. With
`NULLS NOT DISTINCT`, NULL counts as a value like any other, so two rows with the same key collide
even when it holds a NULL. Combined with a `WHERE` predicate, only the rows the predicate accepts
take part. That gives "one active row per key, where the key may be NULL":

```sql
CREATE TABLE subscriptions (id UUID PRIMARY KEY, account TEXT, state TEXT NOT NULL);
CREATE UNIQUE INDEX subscriptions_one_active ON subscriptions (account)
  NULLS NOT DISTINCT WHERE state = 'active';

INSERT INTO subscriptions VALUES (gen_random_uuid(), NULL, 'active');
INSERT INTO subscriptions VALUES (gen_random_uuid(), NULL, 'ended');   -- fine: not active
INSERT INTO subscriptions VALUES (gen_random_uuid(), NULL, 'active');
-- ERROR 23505: duplicate key violates unique index subscriptions_one_active

INSERT INTO subscriptions VALUES (gen_random_uuid(), NULL, 'active')
ON CONFLICT (account) WHERE state = 'active' DO UPDATE SET state = 'replaced';
```

- A unique index created by an earlier release allowed only one NULL per key. It now follows
  `NULLS DISTINCT` like any other, so it accepts several NULLs. If you rely on a single NULL,
  drop the index and recreate it with `CREATE UNIQUE INDEX ... NULLS NOT DISTINCT`.
- `NULLS [NOT] DISTINCT` is only accepted on a `UNIQUE` index (`42601` otherwise).
- A unique index cannot be created on a partitioned table (`0A000`), because it could not see a
  duplicate across partitions. Declare a `UNIQUE` constraint on the partitioned table instead (it
  must include the partition key), or create the unique index on each partition.
- The index is checked on every write: `INSERT`, `INSERT ... SELECT`, `COPY`, `UPDATE` (including a
  row moving into or out of the predicate), `MERGE`, both upsert arms and foreign-key cascades.
  `CREATE UNIQUE INDEX` over existing rows refuses with `23505` when they already hold a duplicate.
- Each writer of a key takes a key lock held to the end of its transaction, so two transactions
  writing the same key never both commit, at any isolation level: the second one fails at once with
  `40001` (retry it) or, when the first has already committed, with `23505`. A row that leaves a key
  (its key changes, it leaves the predicate) keeps the key locked until its transaction ends, and a
  row deleted by a transaction still in progress keeps its key taken, so a rollback can never
  restore a duplicate.
- `ON CONFLICT (cols)` infers a full unique index over exactly those columns when no `PRIMARY KEY`
  or `UNIQUE` constraint matches. A partial unique index is inferred only from
  `ON CONFLICT (cols) WHERE pred`, when each `AND`-ed condition of the index's predicate also appears
  in `pred`. This works wherever the `INSERT` appears, including `PREPARE ... AS` and a
  data-modifying `WITH`. A bare `ON CONFLICT DO NOTHING` skips a row that would collide on any
  unique index.
  `ON CONFLICT ON CONSTRAINT` names a constraint, not an index.
- The setting is kept across restarts and dropped with `DROP INDEX`. `information_schema.statistics`
  lists the index with `non_unique = 0`; the setting itself is not shown there, as for a constraint.

### Views, sequences, domains, schemas, databases

```sql
CREATE VIEW big_orders AS SELECT * FROM orders WHERE total > 1000;
CREATE OR REPLACE VIEW big_orders AS SELECT * FROM orders WHERE total > 500;
DROP VIEW big_orders;

CREATE MATERIALIZED VIEW daily AS
  SELECT date_trunc('day', placed) AS d, count(*) AS n FROM orders GROUP BY 1;
REFRESH MATERIALIZED VIEW daily;

CREATE SEQUENCE invoice_no START 1000 INCREMENT BY 5;
SELECT nextval('invoice_no'), currval('invoice_no');
SELECT setval('invoice_no', 5000);
ALTER SEQUENCE invoice_no INCREMENT BY 10 MAXVALUE 100000;
ALTER SEQUENCE invoice_no RESTART WITH 2000;    -- the next nextval is exactly 2000
ALTER SEQUENCE invoice_no RESTART;              -- back to its START
DROP SEQUENCE invoice_no;

CREATE DOMAIN positive_int AS INT CHECK (VALUE > 0);
DROP DOMAIN positive_int;

CREATE SCHEMA reporting;
CREATE TABLE reporting.facts (id INT);
DROP SCHEMA reporting CASCADE;          -- RESTRICT (the default) refuses a non-empty schema

CREATE DATABASE app;                    -- a separate physical database; connect to it by name
DROP DATABASE app;
```

A view over a single table with no aggregate is auto-updatable: `INSERT`, `UPDATE` and `DELETE`
through it reach the base table. With `WITH CHECK OPTION`, a row written through the view must stay
visible through it:

```sql
CREATE VIEW small_orders AS
  SELECT * FROM orders WHERE total < 100 WITH CHECK OPTION;
INSERT INTO small_orders (customer, total) VALUES (1, 500);
-- ERROR 44000: new row violates check option for view `small_orders`
```

The check applies to `INSERT` and `UPDATE` (a `DELETE` cannot make a row invisible-but-present).
`LOCAL` and `CASCADED` are both accepted and behave the same, since an updatable view sits on one
base table.

A materialized view holds the result of its last `REFRESH`. Incremental maintenance is opt-in with
`CREATE MATERIALIZED VIEW ... WITH (incremental = true) AS ...`; a body that cannot be maintained
incrementally is refused rather than silently downgraded.

`ALTER SEQUENCE` takes any mix of `INCREMENT [BY]`, `MINVALUE` / `NO MINVALUE`, `MAXVALUE` /
`NO MAXVALUE`, `START [WITH]`, `RESTART [[WITH] n]`, `CACHE n` (accepted, no effect) and
`CYCLE` / `NO CYCLE`, validates the merged definition, and `IF EXISTS` makes a missing sequence a
no-op. `information_schema.sequences` lists every sequence with its bounds (a sequence created on
an older release appears there after its first `ALTER SEQUENCE`). Sequence functions take the sequence name as a text literal and may only appear where they
are evaluated exactly once: a `SELECT` without `FROM`, a `VALUES` row, or a column default.

`CREATE DOMAIN` accepts a base type and a `CHECK`; a `DEFAULT` or `COLLATE` on a domain is not
accepted yet. A domain's `CHECK` follows the same [volatility](#volatility) rule as a table's.

### Comments

```sql
COMMENT ON TABLE orders IS 'One row per checkout';
COMMENT ON COLUMN orders.total IS 'Gross, including tax';
COMMENT ON COLUMN sales.orders.total IS 'Schema-qualified';
COMMENT ON CONSTRAINT orders_total_check ON orders IS 'No negative totals';
COMMENT ON INDEX orders_placed_idx IS 'Serves the daily report';
COMMENT ON TRIGGER orders_audit ON orders IS 'Writes the audit log';
COMMENT ON FUNCTION net_total(numeric) IS 'Total without tax';
COMMENT ON PROCEDURE close_day IS 'Run at midnight';
COMMENT ON VIEW sales.big_orders IS 'Orders over 1000';
COMMENT ON MATERIALIZED VIEW daily_totals IS 'Refreshed hourly';
COMMENT ON SCHEMA sales IS 'Order data';
COMMENT ON SEQUENCE sales.invoice_no IS 'Invoice numbers';
COMMENT ON TYPE order_state IS 'Lifecycle of an order';
COMMENT ON DOMAIN positive_amount IS 'Amounts above zero';
COMMENT ON TABLE orders IS NULL;        -- remove
```

Names resolve the way the object's own statements resolve them: a table (and the table of a
column, constraint or trigger) through the search path or an explicit schema, a view or sequence
in the current schema unless qualified, and functions, procedures, materialized views, types and
domains in `public`. An object that does not exist is refused with the SQLSTATE its `DROP` would
report: `42P01` for a table or view, `42703` for a column, `42704` for a constraint, index,
trigger, sequence, type or domain, `42883` for a function or procedure and `3F000` for a schema.
`IS NULL` or `IS ''` removes a comment, and setting one again replaces it. A materialized view
has one comment: `COMMENT ON TABLE` and `COMMENT ON MATERIALIZED VIEW` write the same one.

Commenting on an object needs its owner or a superuser. Tables (and their columns, constraints,
indexes and triggers), materialized views and schemas record an owner, and commenting on them
follows the same rule as dropping them. Plain views and sequences read as owned by the bootstrap
superuser, and functions, procedures, types and domains record no owner, so only a superuser
comments on those, even where another role may drop them. Nobody
comments on another session's temporary objects.

Function and procedure names are unique, so the name alone identifies one. An argument list, when
given, must have as many entries as the routine has parameters (a procedure may be named by its
`IN` parameters or by all of them); otherwise the statement names a routine that does not exist
and fails with `42883`. Parameter types are not recorded, so only the count is compared.
`COMMENT ON TYPE` also accepts a domain; the comment is the domain's under either statement.

A comment is part of the object. Renaming a table, a column or a trigger keeps it; dropping the
object removes it, including drops that happen by cascade (a dropped column takes the comments of
the indexes and constraints on it, `DROP TABLE` those of its columns, constraints, indexes and
triggers, `DROP SCHEMA ... CASCADE` those of its tables and the schema's own). Comments are
transactional: one written in an open transaction is invisible to other sessions until it
commits, and a rollback discards it. They are stored durably and survive a restart. Two
transactions writing the comment of one object, or one commenting on an object while another
drops or alters it, conflict: the second fails with `40001` and can retry.

Comments are read back from `information_schema.nusadb_comments`, one row per comment. A comment on
another session's temporary object is never listed, and a comment on a table, or on anything that
belongs to one, is listed only to a session that owns the table, holds a privilege on it, or is a
superuser:

| Column | Meaning |
|---|---|
| `object_type` | `TABLE`, `COLUMN`, `CONSTRAINT`, `INDEX`, `TRIGGER`, `FUNCTION`, `PROCEDURE`, `VIEW`, `MATERIALIZED VIEW`, `SCHEMA`, `SEQUENCE`, `TYPE` or `DOMAIN` |
| `object_schema` | the schema the object lives in (`public` for functions, procedures, materialized views, types and domains) |
| `table_name` | the table a column, constraint, index or trigger belongs to (the table itself for a table comment), otherwise `NULL` |
| `object_name` | the object's own name (the column's name for a column comment) |
| `description` | the comment text |

```sql
SELECT object_name, description
FROM   information_schema.nusadb_comments
WHERE  object_type = 'CONSTRAINT' AND table_name = 'orders';
```

---

## Inserting and changing rows

```sql
INSERT INTO customers (email, country) VALUES ('ana@example.com', 'ID')
RETURNING id, joined;

INSERT INTO archive SELECT * FROM orders WHERE total < 10;

INSERT INTO customers (email) VALUES ('ana@example.com')
ON CONFLICT DO NOTHING;

INSERT INTO customers (email, country) VALUES ('ana@example.com', 'SG')
ON CONFLICT (email) DO UPDATE SET country = EXCLUDED.country
WHERE customers.country <> EXCLUDED.country;

INSERT INTO tickets (id, note) OVERRIDING SYSTEM VALUE VALUES (99, 'imported');
-- without OVERRIDING SYSTEM VALUE, writing to a GENERATED ALWAYS identity column is ERROR 428C9.
-- OVERRIDING USER VALUE is the reverse: on a GENERATED BY DEFAULT column it ignores the supplied
-- value and takes the sequence's instead.

UPDATE orders o SET total = total * 1.1
FROM   customers c
WHERE  o.customer = c.id AND c.country = 'ID'
RETURNING id, total;

DELETE FROM orders USING customers c
WHERE  orders.customer = c.id AND c.country = 'XX';

CREATE TABLE stock    (sku TEXT PRIMARY KEY, qty INT);
CREATE TABLE shipment (sku TEXT, qty INT);
INSERT INTO stock    VALUES ('A', 1), ('B', 5), ('C', 2);
INSERT INTO shipment VALUES ('A', 3), ('B', 0), ('D', 7);

MERGE INTO stock t USING shipment s ON t.sku = s.sku
WHEN MATCHED AND s.qty = 0   THEN DELETE
WHEN MATCHED                 THEN UPDATE SET qty = t.qty + s.qty
WHEN NOT MATCHED             THEN INSERT (sku, qty) VALUES (s.sku, s.qty)
WHEN NOT MATCHED BY SOURCE   THEN UPDATE SET qty = 0;

SELECT * FROM stock ORDER BY sku;
--  sku | qty
-- -----+-----
--  A   |   4
--  C   |   0
--  D   |   7

TRUNCATE shipment;
TRUNCATE stock RESTART IDENTITY;
TRUNCATE customers CASCADE;      -- also empties tables that reference it
DROP TABLE shipment;
DROP TABLE stock;
```

`UPDATE ... FROM`, `DELETE ... USING` and `MERGE ... USING` take a single named table as the
extra source. `MERGE` into a partitioned table is refused (`0A000`); merge into its partitions.

### Bulk load and export

`COPY` streams rows in one exchange instead of a round trip per row. The only targets are
`STDIN` and `STDOUT`; in the shell the data travels on the command's own standard streams.

```bash
nusadb-cli -c "COPY customers FROM STDIN" < rows.tsv
nusadb-cli -c "COPY customers (email, country) FROM STDIN WITH (FORMAT csv, HEADER)" < rows.csv
nusadb-cli -c "COPY customers FROM STDIN WITH (DELIMITER '|', NULL '')" < rows.psv
nusadb-cli -c "COPY customers TO STDOUT" > out.tsv
nusadb-cli -c "COPY customers (email, country) TO STDOUT WITH (FORMAT csv, HEADER)" > out.csv
nusadb-cli -c "COPY (SELECT email FROM customers WHERE country = 'ID' ORDER BY email) TO STDOUT WITH (FORMAT csv)" > id.csv
nusadb-cli -c "COPY customers TO STDOUT (FORMAT binary)" > customers.bin
nusadb-cli -c "COPY customers FROM STDIN (FORMAT binary)" < customers.bin
```

Options: `FORMAT text | csv | binary`, `DELIMITER 'c'`, `NULL 'string'`, `HEADER`, and for CSV
only `QUOTE 'c'` and `ESCAPE 'c'`. The text format is tab-delimited with `\N` for NULL. The binary
format (also spelled `... FROM STDIN BINARY`) sends each value in its binary wire layout, framed by
length, and takes none of the other options; its stream layout is in the wire protocol
specification, section 12.4. Export takes a table (optionally with a column list) or a `(query)`;
the query form is `TO STDOUT` only, runs with the caller's privileges and row policies like any
`SELECT`, and resolves unqualified names in the default schema, so qualify names from other schemas.
Load takes a table only. A load larger than `--copy-max-bytes` is refused rather than buffered without
bound. Loading into a partitioned table sends each row to its partition, and a row no partition
accepts refuses the whole load; loading into a partition refuses a row outside its bound.

---

## Querying

### The SELECT statement

```text
SELECT [DISTINCT | DISTINCT ON (expr, ...)] projection
FROM   source [JOIN ...] [, lateral_source]
WHERE  predicate
GROUP  BY expr | ROLLUP (...) | CUBE (...) | GROUPING SETS (...)
HAVING predicate
WINDOW name AS (...)
ORDER  BY expr [ASC | DESC] [NULLS FIRST | LAST]
LIMIT  n OFFSET m           -- or: OFFSET m ROWS FETCH FIRST n ROWS ONLY
FOR    UPDATE | SHARE [OF table] [NOWAIT | SKIP LOCKED]
```

### Joins, subqueries, set operations

```sql
SELECT c.email, count(o.id) AS orders
FROM   customers c
LEFT JOIN orders o ON o.customer = c.id
WHERE  c.country = 'ID'
GROUP BY c.email
HAVING count(o.id) > 2
ORDER BY orders DESC NULLS LAST
LIMIT 10 OFFSET 20;

SELECT id FROM orders WHERE total BETWEEN SYMMETRIC 200 AND 50;  -- bounds in either order
SELECT id, total FROM orders ORDER BY total USING >, id USING <; -- USING < is ASC, USING > is DESC

SELECT DISTINCT ON (customer) customer, total
FROM   orders ORDER BY customer, total DESC;          -- the biggest order per customer

SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer = c.id AND o.total > 500);

SELECT email FROM customers WHERE id IN (SELECT customer FROM orders WHERE total > 100);
SELECT email FROM customers WHERE spend > ALL (SELECT total FROM orders WHERE customer = 2);
SELECT (id, email) IN ((1,'a@x'), (2,'b@x')) AS matched FROM customers;

SELECT email FROM customers WHERE country = 'ID'
UNION      SELECT email FROM customers WHERE spend > 100
EXCEPT     SELECT email FROM customers WHERE email LIKE '%@blocked.example'
INTERSECT  SELECT email FROM customers;

SELECT c.email, o.total
FROM   customers c
CROSS JOIN LATERAL (SELECT total FROM orders WHERE customer = c.id ORDER BY total DESC LIMIT 1) o;

SELECT c.email, n
FROM   customers c
CROSS JOIN LATERAL generate_series(1, c.id::INT) AS n;   -- a set-returning function may correlate

SELECT * FROM (VALUES (1, 'one'), (2, 'two')) AS v(n, word);

CREATE TABLE profiles (id BIGINT, nick TEXT);
SELECT email, nick FROM customers NATURAL JOIN profiles;
SELECT email, nick FROM customers JOIN profiles USING (id);
```

Join types: `INNER`, `LEFT`, `RIGHT`, `FULL`, `CROSS`, `NATURAL`, `LATERAL`, and `USING (...)`.
Subqueries may be scalar, `IN` / `NOT IN`, `EXISTS`, quantified (`ANY` / `SOME` / `ALL`), correlated,
and used in `FROM` as a derived table. `IN (subquery)` and a quantified comparison require the same
type on both sides (`BIGINT` against `BIGINT`, `NUMERIC(12,2)` against `NUMERIC(12,2)`); cast when
they differ.

### Pattern matching

```sql
SELECT email FROM customers
WHERE  email LIKE '%@example.com'
   OR  email ILIKE 'ANA%'                       -- case-insensitive
   OR  email SIMILAR TO '(ana|budi)@%'
   OR  email ~ '^[a-c]'                         -- regular expression, ~* for case-insensitive
   OR  email LIKE '50\%' ESCAPE '\';
```

### Window functions

```sql
SELECT country, email, spend,
       rank()       OVER w                         AS rk,
       row_number() OVER w                         AS n,
       sum(spend)   OVER (PARTITION BY country)    AS country_total,
       lag(spend)   OVER w                         AS previous,
       avg(spend)   OVER (ORDER BY joined ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving
FROM   customers
WINDOW w AS (PARTITION BY country ORDER BY spend DESC);

SELECT country, count(*) AS orders,
       sum(count(*)) OVER () AS total_orders          -- an aggregate may feed a window function
FROM   customers GROUP BY country;
```

Functions: `row_number`, `rank`, `dense_rank`, `ntile`, `cume_dist`, `percent_rank`, `lag`,
`lead`, `first_value`, `last_value`, `nth_value`, and `count` / `sum` / `avg` / `min` / `max`
over a window. Frames support `ROWS`, `RANGE`, `GROUPS` and the `EXCLUDE` clause.

### Common table expressions

```sql
CREATE TABLE staff (id INT, manager INT);
INSERT INTO staff VALUES (1, NULL), (2, 1), (3, 1), (4, 2);

WITH RECURSIVE chain AS (
  SELECT id, manager, 1 AS depth FROM staff WHERE manager IS NULL
  UNION ALL
  SELECT s.id, s.manager, c.depth + 1
  FROM staff s JOIN chain c ON s.manager = c.id
)
SELECT depth, count(*) FROM chain GROUP BY depth ORDER BY depth;
--  depth | count
-- -------+-------
--      1 |     1
--      2 |     2
--      3 |     1

WITH gone AS (
  DELETE FROM staff WHERE id = 4 RETURNING *
)
INSERT INTO profiles SELECT id, 'former staff' FROM gone;
```

`WITH x AS MATERIALIZED (...)` and `NOT MATERIALIZED` are accepted as hints. A data-modifying
statement with `RETURNING` can be used inside `WITH`.

A recursive walk over a graph that may loop uses `CYCLE` to stop instead of running forever:

```sql
CREATE TABLE edge (src INT, dst INT);
INSERT INTO edge VALUES (1,2), (2,3), (3,1);

WITH RECURSIVE walk(id, depth) AS (
  SELECT 1, 0
  UNION ALL
  SELECT e.dst, walk.depth + 1 FROM walk JOIN edge e ON walk.id = e.src
) CYCLE id SET is_cycle USING path
SELECT id, depth, is_cycle FROM walk ORDER BY depth;
--  id | depth | is_cycle
-- ----+-------+---------
--   1 |     0 | false
--   2 |     1 | false
--   3 |     2 | false
--   1 |     3 | true
```

The CTE needs an explicit column list; `CYCLE col SET mark USING path` appends two more columns to
it: `mark` (true on the row that revisits a value) and `path` (the array of visited values), and
recursion stops descending past a marked row. One cycle column only, and the boolean marker form
only.

### Grouping sets and aggregates with filters

```sql
SELECT country, date_trunc('month', joined) AS month, count(*)
FROM   customers
GROUP BY GROUPING SETS ((country), (date_trunc('month', joined)), ());   -- repeat the expression, not the alias

SELECT country, count(*) FROM customers GROUP BY ROLLUP (country);

SELECT count(*) FILTER (WHERE spend > 100)             AS big,
       count(DISTINCT country)                          AS countries,
       string_agg(email, ',' ORDER BY email)            AS list,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY spend) AS median,
       mode() WITHIN GROUP (ORDER BY country)           AS commonest,
       rank(500) WITHIN GROUP (ORDER BY spend)          AS where_500_would_rank
FROM   customers;
```

### Set-returning functions

A function that returns rows can be used in `FROM`, optionally with `WITH ORDINALITY`.

```sql
SELECT d::DATE FROM generate_series(DATE '2026-01-01', DATE '2026-01-07', INTERVAL '1 day') AS d;
SELECT n FROM generate_series(1, 5) AS n;
SELECT word FROM regexp_split_to_table('a b  c', '\s+') AS word;
SELECT part, i FROM string_to_table('x,y,z', ',') WITH ORDINALITY AS t(part, i);
SELECT unnest(ARRAY[30, 10, 20]) AS v ORDER BY v DESC;   -- the sort runs on the expanded rows
```

### Reading a plan (`EXPLAIN`)

```sql
EXPLAIN SELECT c.email, sum(o.total)
FROM   customers c JOIN orders o ON o.customer = c.id
WHERE  c.country = 'ID'
GROUP BY c.email;

EXPLAIN ANALYZE SELECT * FROM orders WHERE total > 100;   -- runs it, reports actual rows and time
EXPLAIN VERBOSE SELECT id FROM orders;                    -- adds each node's output columns
EXPLAIN (FORMAT JSON) SELECT id FROM orders;              -- one JSON document
```

The plan is an indented tree, one node per line:

```text
Project (2 column(s))
  GroupAggregate (1 key(s), 1 aggregate(s))
    HashJoin (Inner, 1 key(s))
      Filter
        SeqScan: customers
      SeqScan: orders
```

With `ANALYZE`, each node also reports what happened:

```text
Project (4 column(s)) (est. rows=4 cost=4.8) (actual rows=2)
  Filter (est. rows=4 cost=4.4) (actual rows=2)
    SeqScan: orders (est. rows=4 cost=4.0) (actual rows=4)
Execution: actual rows=2, total time=0.034 ms
```

`est. rows` and `cost` come from statistics, so they appear once the table has been analyzed (by
hand or by the background worker). Node names include `SeqScan`, `IndexScan`, `Filter`, `Sort`,
`HashJoin`, `NestedLoopJoin`, `GroupAggregate`, `Limit`, `Window` and `VectorKnn`; a vector query
that cannot use an HNSW index says so (`exact scan, no HNSW index for vector_l2_ops`).

---

## Full-text search

Text is turned into a `tsvector` (a sorted list of lexemes with positions) and matched against a
`tsquery` with `@@`. Three configurations exist: `simple` lower-cases words; `english` (the default)
also removes stop words and stems; and `indonesian` removes Indonesian stop words and applies a
dictionary-free, rule-based Indonesian stemmer, so inflected forms of one root match each other
(`membaca`, `dibaca`, `bacaan` all reduce to `baca`). Being dictionary-free, the Indonesian stemmer
cannot always recover a root that was formed by consonant elision (`memakan`) or one that merely
looks affixed (`sekolah`); it reduces such words consistently regardless, which keeps matching sound.

```sql
CREATE TABLE articles (id INT PRIMARY KEY, body TEXT);
INSERT INTO articles VALUES
  (1, 'The quick brown fox jumps over the lazy dog'),
  (2, 'A slow red fox');

SELECT to_tsvector('simple', 'The quick brown fox');
-- 'brown':3 'fox':4 'quick':2 'the':1

SELECT to_tsvector('english', 'The quick brown foxes');
-- 'brown':3 'fox':4 'quick':2

SELECT id FROM articles
WHERE  to_tsvector('english', body) @@ to_tsquery('english', 'quick & fox');
--  id
-- ----
--   1

SELECT id, ts_rank(to_tsvector(body), plainto_tsquery('red fox')) AS score
FROM   articles
WHERE  to_tsvector(body) @@ plainto_tsquery('red fox')
ORDER  BY score DESC;
```

Query operators: `&` and, `|` or, `!` not, `<->` / `<N>` phrase (followed-by at exactly N
positions), with parentheses. `to_tsquery` takes that syntax; `plainto_tsquery` takes plain words
and ANDs them; `phraseto_tsquery` takes plain words and chains them into a phrase whose distances
mirror the word positions (`phraseto_tsquery('the cat and the rat')` is `'cat' <3> 'rat'`).
`tsquery_phrase(a, b [, distance])` and `a <-> b` join two queries into a phrase. Phrase matching
is position-accurate: `to_tsvector('a b c') @@ 'a <2> c'` holds, `'a <-> c'` does not, and a
`strip()`-ed vector never matches a phrase. Ranking: `ts_rank`, `ts_rank_cd`;
`setweight(vector, 'A')` marks lexemes, `strip(vector)` drops positions. Other configurations
and prefix matching are refused with a clear error rather than approximated.
For hybrid search over a vector index and a text match, `rrf_score(rank [, k])` fuses two rankings.

---

## Routines and triggers

### Functions

A function is `LANGUAGE SQL` (the default) or NusaScript (`LANGUAGE nusascript`; `plpgsql` is
accepted as an alias). A SQL body is a single `SELECT <expression>` without `FROM`, inlined at
each call site. A NusaScript body is a `BEGIN ... END` block whose `RETURN <expr>` produces the
value, coerced to the declared `RETURNS` type; it may run several statements, branch, loop, and
recurse. Parameters are available by name and as `$1..$n`.

```sql
CREATE FUNCTION with_tax(amount NUMERIC) RETURNS NUMERIC
LANGUAGE SQL AS 'SELECT $1 * 1.11';

CREATE OR REPLACE FUNCTION greet(name TEXT) RETURNS TEXT AS $$ SELECT 'hello, ' || $1 $$;

CREATE FUNCTION fact(n INT) RETURNS INT LANGUAGE plpgsql AS $$
BEGIN
  IF n <= 1 THEN
    RETURN 1;
  END IF;
  RETURN n * fact(n - 1);
END
$$;

SELECT with_tax(100), greet('ana'), fact(5);
--  with_tax | greet      | fact
-- ----------+------------+------
--    111.00 | hello, ana |  120

DROP FUNCTION greet;          -- by name, without a parameter list
```

A NusaScript function whose body ends without reaching a `RETURN <value>` fails with `2F005`, and
runaway recursion is stopped at a nesting limit rather than overflowing the stack.

### Procedures and `CALL`

A procedure body is a `$$ ... $$` block of statements. A plain sequence of SQL statements is fine;
a body that starts with `BEGIN` is a NusaScript block with variables and control flow. Parameters
are `$1..$n`. `CALL` runs the procedure inside the caller's transaction. `nusadb-cli` keeps
`$$ ... $$` bodies intact when splitting a batch, so these work from the shell and from any driver.

```sql
CREATE TABLE audit (what TEXT, n INT);

CREATE PROCEDURE classify(v INT) AS $$
BEGIN
  DECLARE label TEXT;
  IF $1 > 100 THEN
    SET label = 'big';
  ELSIF $1 > 10 THEN
    SET label = 'medium';
  ELSE
    SET label = 'small';
  END IF;
  INSERT INTO audit VALUES (label, $1);
END
$$;

CREATE PROCEDURE countdown(start INT) AS $$
BEGIN
  DECLARE i INT DEFAULT $1;
  WHILE i > 0 LOOP
    INSERT INTO audit VALUES ('tick', i);
    SET i = i - 1;
  END LOOP;
END
$$;

CREATE PROCEDURE guarded(v INT) AS $$
BEGIN
  IF $1 < 0 THEN
    RAISE 'negative input';
  END IF;
  INSERT INTO audit VALUES ('ok', $1);
EXCEPTION WHEN OTHERS THEN
  INSERT INTO audit VALUES ('failed', $1);
END
$$;

CALL classify(500);
CALL countdown(3);
CALL guarded(-1);
SELECT * FROM audit ORDER BY what, n;
--   what  |  n
-- --------+-----
--  big    | 500
--  failed |  -1
--  tick   |   1
--  tick   |   2
--  tick   |   3

DROP PROCEDURE countdown;
```

`CALL loud(-1)` on a body without a handler reports `P0001: raised exception: negative input` and
the caller's statement fails.

NusaScript statements: `DECLARE name TYPE [DEFAULT expr]`, `SET name = expr` (or `name := expr`),
`IF ... THEN ... [ELSIF ...] [ELSE ...] END IF`, `WHILE cond LOOP ... END LOOP`,
`FOR i IN low TO high LOOP ... END LOOP`, `RAISE 'message'` (reported as `P0001`), `RETURN`,
`PERFORM expr` (evaluate and discard, for a call made for its side effects), and any
SQL data statement. An `EXCEPTION WHEN OTHERS THEN` handler rolls the body's writes back to a
savepoint and runs in their place. Variables may be referenced by name inside embedded SQL; a
column of the same name wins. `LANGUAGE SQL` is the only language.

`RAISE level 'format' [, value ...]` fills each `%` in the format with the next value (`%%` is a
literal `%`, `NULL` shows as `<NULL>`, and the counts must match, `42601` otherwise). `EXCEPTION`
aborts like `RAISE 'message'`; `NOTICE`, `INFO` and `WARNING` send the message to the client as a
notice and carry on; `LOG` and `DEBUG` carry on and send nothing.

```sql
DO $$
BEGIN
  RAISE NOTICE 'archived % rows into %', 12, 'audit_2026';
  RAISE WARNING 'quota at %%: %', 95;    -- WARNING: quota at %: 95
END
$$;
```

Notices reach a client that negotiated protocol 1.3 (see the
[wire protocol](wire-protocol.md#94-notices-protocol-13)); an embedded session returns them from
`Session::take_notices`. At most 1000 are kept per statement (and, in an embedded session, between
two calls to `take_notices`); the rest are dropped and counted by one closing `WARNING`. Besides
`RAISE`, the server sends one for every `IF EXISTS` that found nothing to drop
(`NOTICE: table "t" does not exist, skipping`) and every `IF NOT EXISTS` that found the object
already there (`NOTICE: relation "t" already exists, skipping`, code `42P07`).

### `DO` blocks

The same body grammar, run once without parameters:

```sql
DO $$
BEGIN
  DECLARE n INT DEFAULT 3;
  WHILE n > 0 LOOP
    INSERT INTO audit VALUES ('do', n);
    SET n = n - 1;
  END LOOP;
END
$$;
```

### Triggers

A trigger runs one SQL statement when a table is written. `NEW.col` and `OLD.col` refer to the
affected row. A `NULL` there is never stricter than the column's values. For a date/time,
interval, boolean, numeric, floating-point, `UUID`, enum, `JSON`/`JSONB` or MAC address
column, and a domain over one, the `NULL` has the type the values have, so `NEW.a + INTERVAL '1
day'` on a `TIMESTAMPTZ` column is a `NULL` timestamp when `a` is `NULL`, not a type error. An
integer column's `NULL` is an `INT` whatever its declared width, as its values are. Any other
column gives an untyped `NULL`, which takes the type it meets: a `TEXT`, `VARCHAR` or `CHAR` value
is a string literal that does the same (`NEW.code = 1`, or inserting it into a `BIGINT` column), and
so, in an `INSERT`, is a value written as a string for a column such as `INET`, a range or an
array.

```sql
CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INT);
CREATE TABLE stock_log (sku TEXT, before INT, after INT, at TIMESTAMPTZ DEFAULT now());

CREATE TRIGGER stock_changes
AFTER UPDATE ON stock
FOR EACH ROW
WHEN (NEW.qty <> OLD.qty)
INSERT INTO stock_log (sku, before, after) VALUES (OLD.sku, OLD.qty, NEW.qty);

INSERT INTO stock VALUES ('A-77', 10);
UPDATE stock SET qty = 8 WHERE sku = 'A-77';
SELECT sku, before, after FROM stock_log;
--  sku  | before | after
-- ------+--------+-------
--  A-77 |     10 |     8

ALTER TABLE stock DISABLE TRIGGER stock_changes;
ALTER TRIGGER stock_changes ON stock RENAME TO log_stock;
DROP TRIGGER log_stock ON stock;
```

Syntax: `CREATE [OR REPLACE] TRIGGER name {BEFORE | AFTER} {INSERT | UPDATE | DELETE} [OR ...] ON
table FOR EACH {ROW | STATEMENT} [WHEN (condition)] <action>`. The action is one `INSERT`,
`UPDATE`, `DELETE` or `SELECT` statement, or `EXECUTE FUNCTION name()` naming a NusaScript
function, so several triggers can share one body:

```sql
CREATE FUNCTION log_stock() RETURNS trigger LANGUAGE nusascript AS $$
BEGIN
  INSERT INTO stock_log (sku, before, after) VALUES (OLD.sku, OLD.qty, NEW.qty);
  RETURN NEW;
END
$$;

CREATE TRIGGER stock_fn AFTER UPDATE ON stock FOR EACH ROW
WHEN (NEW.qty <> OLD.qty) EXECUTE FUNCTION log_stock();
```

The function must take no parameters and have a `BEGIN ... END` body; that is checked when the
trigger is created, not on first fire. `EXECUTE PROCEDURE` is a synonym. Function arguments in
the trigger (`EXECUTE FUNCTION f('x')`) are not accepted.

Inside the body, `OLD.col` and `NEW.col` read the rows of the firing; the row an event does not
have (`OLD` in an `INSERT`, `NEW` in a `DELETE`, both in a statement trigger) reads as `NULL`,
typed by the same rule. So does a `NEW` field the body sets to `NULL`.
These variables describe the firing:

| Variable | Value |
| --- | --- |
| `TG_OP` | `INSERT`, `UPDATE` or `DELETE` |
| `TG_NAME` | the trigger's name |
| `TG_WHEN` | `BEFORE`, `AFTER` or `INSTEAD OF` |
| `TG_LEVEL` | `ROW` or `STATEMENT` |
| `TG_TABLE_NAME`, `TG_RELNAME` | the table the row is in (the partition, for a trigger on a partitioned table) |
| `TG_TABLE_SCHEMA` | that table's schema |
| `TG_NARGS` | `0` |

A column of a table the body queries shadows a variable of the same name.

In a `BEFORE ... FOR EACH ROW` trigger the body decides the row the write goes on with. It can
assign `NEW` fields (`NEW.col := expr` or `SET NEW.col = expr`; the value is converted to the
column's type) and then `RETURN NEW`; `RETURN OLD` writes the old row instead (for an `UPDATE`,
the row stays as it was); `RETURN NULL` skips the row: it is not written, not counted, and no
later trigger fires for it. In a `DELETE`, `NEW` is null, so `RETURN NEW` also skips the row. A
body that ends, or stops at a bare `RETURN`, without naming a row goes on with `NEW` as assigned
(for a `DELETE`, the row is deleted). A rewritten row is checked like any written row: generated
columns are recomputed, and `NOT NULL`, `CHECK`, unique and foreign keys apply to the new values.
Returning anything else is an error (`42804`). The return value of an `AFTER` or statement
trigger is ignored. A `BEFORE` trigger on a referencing table may change other columns of a row
an `ON DELETE` or `ON UPDATE` action writes (its `CHECK` constraints still apply), but skipping
that row or changing its foreign or unique key columns is refused (`27000`), since the action
computed them. So is a `BEFORE DELETE` trigger that keeps in place a row moving between partitions
when other rows reference it: they already follow its new key.

```sql
CREATE FUNCTION stamp() RETURNS trigger LANGUAGE nusascript AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE 'rows are never deleted';
  END IF;
  NEW.changed_at := now();
  RETURN NEW;
END
$$;
```

The action runs in the same transaction as the triggering statement; an error in it
aborts that statement. Cascading triggers are bounded by a depth limit (`54001`).

On a partitioned table, a row-level trigger runs for the rows of every partition below it,
whether the statement names the parent or the partition, and also for a partition created or
attached later. A partition that is detached stops running it. Disabling or dropping the trigger on
the parent applies to every partition; dropping it on a partition is refused (`2BP01`), and a
partition cannot have a trigger with the same name as one of its parent's row triggers (`42710`).
A statement-level trigger fires for the table the statement names: an `INSERT`, `UPDATE` or
`DELETE` through the parent fires the parent's statement triggers once, after every row in every
partition is written, and not the partitions' own. Moving a row between partitions with an
`UPDATE` runs the `DELETE` row triggers of the old partition and then the `INSERT` row triggers of
the new one. A plain `INHERITS` child does not run its parent's triggers.

`INSTEAD OF` triggers attach to a view and REPLACE the write: DML on the view fires the trigger
per row (with `NEW` as the proposed view row and `OLD` as the current one) and touches nothing
else — whatever the trigger's action does is the write. They win over the auto-updatable rewrite,
are row-level only, take no `WHEN` condition, and `RETURNING` projects `NEW` (INSERT/UPDATE) or
`OLD` (DELETE). `BEFORE`/`AFTER` on a view and `INSTEAD OF` on a table are refused (`42809`).

```sql
CREATE VIEW active AS SELECT id, v FROM items WHERE NOT hidden;
CREATE TRIGGER active_ins INSTEAD OF INSERT ON active
  FOR EACH ROW INSERT INTO items VALUES (NEW.id, NEW.v, false);
INSERT INTO active VALUES (1, 'a');            -- runs the trigger's INSERT instead
```

---

## Prepared statements and cursors

### Prepared statements

Drivers prepare through the wire protocol: a statement is parsed once and run with values for
`$1..$n`.

```python
cur.execute("SELECT email FROM customers WHERE country = $1 ORDER BY email", ["ID"])
```

The SQL spellings work too, on any connection, which is how the shell binds parameters:

```sql
PREPARE by_country AS SELECT email FROM customers WHERE country = $1 ORDER BY email;
EXECUTE by_country('ID');
EXECUTE by_country('SG');
DEALLOCATE by_country;       -- or DEALLOCATE ALL
```

A prepared statement lives for the connection and survives transaction ends. Only a runnable
query may be prepared (`SELECT`, set operation, `INSERT`, `UPDATE`, `DELETE`). Executing an
unknown or deallocated name is `26000`; the wrong number of arguments is `42883`;
`EXECUTE ... USING` is not accepted.

### Cursors

A cursor reads a result in pieces. Its rows are captured when the cursor is declared, so it pages
through a stable snapshot. With a spill directory configured, a `SELECT` result larger than
`work_mem` is kept on disk rather than in memory (a `UNION`, `INTERSECT` or `EXCEPT` result is still
held in memory).

```sql
BEGIN;
DECLARE c CURSOR FOR SELECT id, email FROM customers ORDER BY id;
FETCH 2 FROM c;
FETCH NEXT FROM c;
FETCH ALL FROM c;
CLOSE c;
COMMIT;
```

`MOVE` takes the same directions as `FETCH` and positions the cursor exactly as that `FETCH` would,
but returns no rows. Its command tag is `MOVE <count>`, where the count is the number of rows the
`FETCH` would have returned.

```sql
MOVE FORWARD 100 IN c;   -- skip ahead; the tag says how many rows were passed over
MOVE ABSOLUTE 0 IN c;    -- back to before the first row
```

The directions are `NEXT`, `PRIOR`, `FIRST`, `LAST`, `ABSOLUTE n`, `RELATIVE n`, a bare count `n`,
`ALL`, `FORWARD [n | ALL]` and `BACKWARD [n | ALL]`. With no direction the statement means `NEXT`,
and the `FROM` or `IN` before the cursor name may be left out. Every cursor can move in any
direction, whether or not it was declared `SCROLL`.

Counts are signed:

- `ABSOLUTE n` goes to the n-th row from the start; a negative `n` counts from the end, so
  `ABSOLUTE -1` is the last row and `ABSOLUTE -2` the one before it. `ABSOLUTE 0` goes to before
  the first row and returns nothing.
- `RELATIVE n` moves `n` rows from the current one, backward when `n` is negative.
- A negative count in `FETCH n` or `FORWARD n` reads backward, and a negative `BACKWARD n` reads
  forward. A count of `0` (and `RELATIVE 0`) re-reads the current row.
- Asking for a row before the first or after the last returns nothing and leaves the cursor
  before the first row or after the last one; the next `FETCH NEXT` or `FETCH PRIOR` starts from
  there. A count that runs off either end returns the rows it found and stops in the same place.

Backward reads return rows nearest first.

A cursor belongs to the transaction that declares it:

- Without `WITH HOLD`, the cursor closes when that transaction ends, by `COMMIT` or `ROLLBACK`,
  and fetching from it afterwards is `34000`. Such a cursor can only be declared inside
  `BEGIN ... COMMIT`; outside a transaction block `DECLARE` is refused with `25P01`.
- With `WITH HOLD`, the cursor stays open after its transaction commits, until `CLOSE` or the end of
  the session. It may be declared outside a transaction block. If its transaction rolls back, the
  cursor is closed with it.

```sql
DECLARE recent CURSOR WITH HOLD FOR SELECT id FROM orders ORDER BY id DESC;
FETCH 50 FROM recent;
CLOSE recent;
```

Declaring a name that is already open is refused with `42P03`, and the open cursor keeps its rows
and position. Fetching from, moving or closing an unknown cursor is `34000`.

A `DECLARE` sent through the extended query protocol may use `$n` parameters in its query. The bound
values are substituted when the cursor is declared, so the rows it returns are fixed then. Over that
protocol a `FETCH` is described with its cursor's columns, and it can be read in pieces with a row
limit or in binary result format like any other query.

---

## Transactions and locking

The full model, including what to do on a conflict, is in [transactions](transactions.md). The
statements:

```sql
BEGIN;                                             -- or START TRANSACTION
BEGIN ISOLATION LEVEL SERIALIZABLE;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;   -- before the transaction's first query
SAVEPOINT s1;
ROLLBACK TO SAVEPOINT s1;
RELEASE SAVEPOINT s1;
SET CONSTRAINTS ALL DEFERRED;                      -- see Deferrable constraints
COMMIT;                                            -- or END
ROLLBACK;

SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;   -- session default

PREPARE TRANSACTION 'gid';      -- two-phase commit: keep the transaction, prepared
COMMIT PREPARED 'gid';          -- from any connection, outside a transaction block
ROLLBACK PREPARED 'gid';
```

The default isolation level is `READ COMMITTED`. A statement outside `BEGIN` runs in its own
transaction. `SET TRANSACTION` configures the open transaction and must come before its first
statement (`25001` after it); outside a transaction it is refused (`25P01`).
`SET SESSION CHARACTERISTICS AS TRANSACTION` sets the default for transactions begun later.
After an error inside a transaction, every further statement is refused with `25P02` until
`ROLLBACK` (or `ROLLBACK TO SAVEPOINT`). Two-phase commit, its error codes and the
`nusadb_prepared_xacts` view are described in [transactions](transactions.md#two-phase-commit).

Several statements sent as one simple query (`INSERT ...; UPDATE ...`) run as one transaction
unless they are already inside one: an error rolls back all of them. A `BEGIN` among them opens an
ordinary transaction block instead. The full rules are in the
[wire protocol](wire-protocol.md) §9.3. `COPY` inside a transaction block is part of it and is
undone by its `ROLLBACK`.

### Row and table locks

Locks never wait. A row someone else has locked fails immediately with `40001`, so `NOWAIT` is
implied; `SKIP LOCKED` steps over such rows instead, which is the usual shape for a job queue.
A held row lock also blocks writers: an `UPDATE` or `DELETE` that touches the locked row fails
with the same immediate `40001` until the locking transaction ends. With `LIMIT n`, only the
first `n` rows in query order are locked — a `LIMIT 1 ... SKIP LOCKED` claim takes one row and
leaves the rest for other workers.

```sql
SELECT * FROM orders WHERE id = 1 FOR UPDATE;      -- also on an inheritance parent (locks
                                                   -- matched rows in parent and children)
SELECT * FROM orders ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED;   -- claim one job
SELECT * FROM orders FOR UPDATE OF orders NOWAIT;
SELECT * FROM orders FOR SHARE;

LOCK TABLE orders IN ACCESS EXCLUSIVE MODE;        -- inside a transaction
LOCK TABLE orders, customers IN ACCESS SHARE MODE;
```

`LOCK TABLE` accepts `ACCESS SHARE` and `ACCESS EXCLUSIVE`; the other modes are refused.

### Advisory locks

An advisory lock is a cooperative lock named by an integer key you choose; the engine attaches no
meaning to the key, so it is up to the application to agree on what a key stands for. A lock is held
by a session and is released automatically when the session ends. Acquiring never waits, matching
the engine's no-wait locking: it either takes the lock or reports that another session holds it.

```sql
SELECT nusadb_try_advisory_lock(42);        -- true if taken (or already held by this session)
SELECT nusadb_try_advisory_lock(1, 2);      -- two int keys combine into one bigint key
SELECT nusadb_advisory_unlock(42);          -- release one hold; true if this session held it
SELECT nusadb_advisory_unlock_all();        -- release every advisory lock this session holds
```

A session may take the same lock more than once; an equal number of unlocks releases it. A lock
call must be evaluated exactly once, so it is written as a `SELECT` without `FROM` and is refused in
a per-row position (a scan, `WHERE`, or an `UPDATE` assignment). The blocking `nusadb_advisory_lock`,
the shared-mode `..._shared`, and the transaction-scoped `..._xact_...` variants are not built and
are refused with a clear message.

---

## Notifications

`LISTEN` subscribes a connection to a channel; `NOTIFY` from any connection in the same database
delivers to every listener. Inside a transaction the notification is queued and sent at `COMMIT`
(and dropped on `ROLLBACK`); in autocommit it is sent at once. A listener receives notifications
between its own statements, which is how the drivers surface them.

```sql
-- session 1
LISTEN orders_changed;

-- session 2
NOTIFY orders_changed, 'order 42 shipped';

-- session 1
UNLISTEN orders_changed;
UNLISTEN *;
```

---

## Maintenance

```sql
ANALYZE customers;        -- refresh planner statistics for one table
ANALYZE;                  -- every table
VACUUM;                   -- reclaim row versions no snapshot can see
VACUUM (FULL, ANALYZE) orders;
REINDEX TABLE orders;     -- rebuild the indexes an earlier release built over CHAR columns (see below)
CLUSTER orders;           -- accepted for compatibility; rows are already clustered by row id, so it is a no-op
CREATE EXTENSION IF NOT EXISTS vector;   -- accepted for compatibility; installs nothing, every type is built in
CHECKPOINT;               -- fold the log into an image and truncate it; needs a quiet engine
```

### Indexes over CHAR columns from earlier releases

An index whose key or `WHERE` clause reads a `CHAR(n)` column holds each value without its
trailing blanks, as the row stores it. Earlier releases wrote the value as the statement gave it,
so an index they built may hold `'x  '` for a row that reads back as `'x'`. Such an index stays
correct with this release: every uniqueness, foreign key and exclusion check, and every query
that reads through the index, still finds a key however it was written. It is slower, though: a
lookup tries each way the value may have been padded, an ordered read through it with `LIMIT`
keeps only the rows it returns, and a unique index keyed on an expression over such a column, or
with a `WHERE` clause that reads one, is checked against the table's rows (read once per
statement). A query does not use an expression index over such a column at all until it is
rebuilt.

Rebuilding the index makes it as fast as one built by this release:

```sql
REINDEX INDEX orders_code_key;   -- one index
REINDEX TABLE orders;            -- every index of a table, and of the partitions below it
REINDEX SCHEMA app;              -- every table of a schema
REINDEX DATABASE mydb;           -- every table of the database
```

`REINDEX INDEX` and `REINDEX TABLE` need the table's owner (or a superuser), as `CREATE INDEX`
does; `REINDEX SCHEMA` and `REINDEX DATABASE` rebuild the tables the role owns and skip the others
(a superuser's covers every table). `REINDEX DATABASE name` must name the connection's database,
and `REINDEX SYSTEM` is the same as `REINDEX DATABASE`. Only those older indexes are rebuilt; any
other index is left alone. A rebuild streams the table's rows (the index of a `PRIMARY KEY` or
`UNIQUE` constraint reads them a second time, to check that no key is held twice), and keeps
everything the index was declared with: its key, `WHERE` clause, `NULLS [NOT] DISTINCT`, the
constraint it backs with its timing, the foreign keys that reference it, and comments.

`REINDEX INDEX` and `REINDEX TABLE` run in one transaction, and may run inside a transaction
block; `REINDEX SCHEMA` and `REINDEX DATABASE` commit each table on its own, so they are refused
inside a transaction block (`25001`), and stopping one part way keeps the tables already done.
Either way an interrupted rebuild leaves that index as it was. A rebuild needs its table to itself:
while another transaction is writing it, `REINDEX TABLE` and `REINDEX INDEX` fail at once with
`40001` (retry when the table is quiet), and `REINDEX SCHEMA` and `REINDEX DATABASE` retry each
table a few times before giving up with the same error. When a server opens such a database it
logs how many of these indexes there are.

An earlier release could let two rows hold one key of a unique index or constraint once trailing
blanks are ignored (`'a  '` and `'a'`). `REINDEX` then fails with `23505`, naming the key (its
value only for a role that may read the table), and leaves the index as it was, still enforced;
remove one of the rows and run it again.

A background worker re-analyzes tables whose statistics have gone stale, so `ANALYZE` by hand
matters mainly right after a large load. Version reclamation also runs in the background;
`VACUUM` triggers it now. `CHECKPOINT` refuses while any transaction is in flight, including one on
the connection that issues it; see [deployment](deployment.md#checkpoints-backup-and-restore).

---

## Introspection

```sql
SHOW TABLES;                          -- tables visible through search_path
SHOW COLUMNS FROM customers;          -- column, type
DESCRIBE customers;                   -- same as SHOW COLUMNS

SELECT table_schema, table_name FROM information_schema.tables ORDER BY 1, 2;
SELECT column_name, data_type, is_nullable
FROM   information_schema.columns WHERE table_name = 'orders';
SELECT * FROM information_schema.table_constraints WHERE table_name = 'orders';
SELECT * FROM information_schema.key_column_usage;
SELECT * FROM information_schema.statistics;          -- indexes
SELECT * FROM information_schema.views;
SELECT * FROM information_schema.schemata;
SELECT * FROM information_schema.table_privileges;
SELECT * FROM information_schema.applicable_roles;
SELECT * FROM information_schema.enabled_roles;

SELECT constraint_name, unique_constraint_name, update_rule, delete_rule
FROM   information_schema.referential_constraints;    -- foreign keys and their rules
SELECT constraint_name, check_clause FROM information_schema.check_constraints;
SELECT routine_name, routine_type, data_type, external_language
FROM   information_schema.routines;                   -- functions and procedures
SELECT sequence_name, start_value, increment, cycle_option
FROM   information_schema.sequences;
SELECT udt_name, attribute_name, ordinal_position, data_type
FROM   information_schema.attributes;                 -- fields of composite types
SELECT * FROM information_schema.user_defined_types;  -- composite types
SELECT * FROM information_schema.nusadb_comments;     -- COMMENT ON comments

SELECT * FROM nusadb_databases;       -- every database in this server
SELECT * FROM nusadb_prepared_xacts;  -- prepared transactions (transaction, gid, prepared, owner, database)
SELECT * FROM nusadb_triggers;        -- likewise nusadb_roles, nusadb_policies, nusadb_procedures,
                                      -- nusadb_functions, nusadb_matviews

SELECT version(), current_user, current_database(), current_schema();
SELECT nusadb_typeof(1.5), nusadb_typeof(now());
```

`SHOW COLUMNS` returns `column`, `type` and `nullable`. The `nusadb_*` catalogs are plain tables:
`nusadb_roles (name, superuser, login, createdb, createrole, inherit)`,
`nusadb_triggers (name, table, timing, events, for_each, when, action, enabled, schema)`,
`nusadb_policies (table, name, command, roles, using, check, permissive, schema)`,
`nusadb_procedures (name, param_count, out_params, body)`, `nusadb_functions (name, param_count,
param_names, body, language, return_type)`, `nusadb_matviews (name, def)`,
`nusadb_partitions (role, table, aux, lo, hi)`, `nusadb_inheritance (child, parent, seq)`,
`nusadb_sequences`, `nusadb_databases (name)`. Only a superuser may read a `nusadb_*` catalog; every other role uses
`information_schema`. Two are views every role may read: `nusadb_databases`, and
`nusadb_prepared_xacts`, which shows a superuser every prepared transaction of the database and
any other role its own. A catalog that has never been written to may not exist yet, in which case
selecting from it reports `42P01`.

In `nusadb-cli`, `\dt` runs `SHOW TABLES`, `\d name` runs `SHOW COLUMNS FROM name`, and `\l` lists
databases.

---

## Functions

Every name below is callable today. Signatures follow the usual conventions; where NusaDB's differs
it is noted.

### String

`length`, `char_length`, `character_length`, `octet_length`, `bit_length`, `upper`, `lower`,
`initcap`, `substr`, `substring`, `left`, `right`, `replace`, `translate`, `lpad`, `rpad`, `ltrim`,
`rtrim`, `btrim`, `trim`, `repeat`, `reverse`, `concat`, `concat_ws`, `format`, `split_part`,
`strpos`, `starts_with`, `ascii`, `chr`, `quote_literal`, `quote_nullable`, `quote_ident`,
`parse_ident`, `encode`, `decode`, `convert_to`, `convert_from`, `get_byte`, `set_byte`, plus the
syntax forms `POSITION(x IN s)`, `OVERLAY(s PLACING r FROM n [FOR l])`, `SUBSTRING(s FROM n FOR l)`,
`TRIM([LEADING | TRAILING | BOTH] c FROM s)`.

Regular expressions: `regexp_replace`, `regexp_match`, `regexp_matches`, `regexp_like`,
`regexp_count`, `regexp_instr`, `regexp_substr`, `regexp_split_to_array`, `regexp_split_to_table`.

`encode` / `decode` accept `base64`, `hex` and `escape`. `convert_to` / `convert_from` accept `UTF8`.

### Numeric

`abs`, `round`, `trunc`, `ceil`, `ceiling`, `floor`, `sign`, `mod`, `div`, `power`, `pow`, `sqrt`
(also the prefix `|/` operator; `||/` is `cbrt`),
`cbrt`, `exp`, `ln`, `log` (`log(x)` is base 10, `log(b, x)` is base `b`), `log10`, `sin`, `cos`,
`tan`, `cot`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
`degrees`, `radians`, `pi`, `gcd`, `lcm`, `factorial`, `bit_count`, `to_hex`, `to_number`,
`width_bucket`, `scale`, `min_scale`, `trim_scale`, `isfinite`, `random`, `setseed`.

### Conditional

`coalesce`, `nullif`, `greatest`, `least`, `nvl`, `ifnull`, `num_nulls`, `num_nonnulls`, `CASE`,
`CAST(x AS type)`, `x::type`, `TRY_CAST(x AS type)` (NULL instead of an error).

### Date and time

`now`, `transaction_timestamp`, `statement_timestamp`, `clock_timestamp`, `current_timestamp`,
`current_date`, `current_time`, `localtimestamp`, `localtime`, `date_trunc`, `date_part`, `EXTRACT(field FROM x)`, `date_bin`,
`age`, `to_char`, `to_date`, `to_timestamp`, `make_date`, `make_time`, `make_timestamp`,
`make_timestamptz`, `make_interval` (positional or named: `make_interval(days => 5)`),
`justify_days`, `justify_hours`, `justify_interval`,
`x AT TIME ZONE zone`.

`EXTRACT` fields: `year`, `quarter`, `month`, `week`, `day`, `hour`, `minute`, `second`, `dow`,
`isodow`, `doy`, `isoyear`, `epoch`, `decade`, `century`, `millennium`, `microseconds`,
`milliseconds`, `julian`, `timezone`, `timezone_hour`, `timezone_minute`. `date_trunc` accepts the
units from `microseconds` up to `millennium` and truncates a `TIMESTAMP`, `TIMESTAMPTZ`, `TIME` or
`INTERVAL`, keeping its type (`week` is refused for a `TIME` or an `INTERVAL`).

The clock functions read three different instants:

| Function | Instant |
| --- | --- |
| `now()`, `transaction_timestamp()`, `current_timestamp`, `current_date`, `current_time`, `localtimestamp`, `localtime` | the start of the current transaction, the same in every statement of it |
| `statement_timestamp()` | the start of the current statement, the same for every row it touches and every trigger, procedure body or function it runs |
| `clock_timestamp()` | the wall clock at the moment of the call |

Outside an explicit transaction every statement is its own transaction, so `now()` and
`statement_timestamp()` agree. Inside `BEGIN` ... `COMMIT`, or in one query string holding several
statements (which run as one implicit transaction), `now()` keeps the transaction's start while
`statement_timestamp()` moves on with each statement, so every row the transaction writes with
`DEFAULT now()` carries the same time.

`clock_timestamp()` changes within a statement and from row to row. It is never folded into a
constant or served from the result cache. The same holds for `uuidv7()`.

```sql
BEGIN;
SELECT now() AS txn_start, statement_timestamp() AS stmt_start, clock_timestamp() AS at_call;
SELECT now() AS txn_start, statement_timestamp() AS stmt_start;  -- same txn_start, later stmt_start
COMMIT;
SELECT g, clock_timestamp() FROM generate_series(1, 3) AS g;  -- a later value on each row
```

#### Volatility

Every built-in function belongs to one of three classes, which decide where it may be used:

- **Immutable**: the same arguments always give the same result (`lower`, `abs`, `date_trunc`,
  `age(end, start)` and most others).
- **Stable**: fixed within a statement or transaction but not across them or across sessions: the
  transaction clock (`now()`, `current_timestamp`, `current_date`, `current_time`, `localtimestamp`, `localtime`,
  `statement_timestamp()`), `age(value)`, `current_user`, `session_user`, `current_setting`,
  `current_database`, `current_schema`, `version()`.
- **Volatile**: may change on every call: `clock_timestamp()`, `random()`, `setseed()`,
  `gen_random_uuid()` and `uuid_generate_v4()`, `uuidv7()`, `nextval`, `currval`, `setval`, and the
  advisory-lock functions.

An index key expression, a partial-index `WHERE` predicate and a generated column may call only
immutable built-ins; anything else is refused with `42P17`. A `CHECK` constraint, on a table or a
domain, may not call a volatile built-in (`42P17`), but it may call a stable one: `CHECK (placed <= now())` is accepted and
tests each row against the time of the transaction that writes it. Such a check is not re-run
later, so rows already stored stay valid as time moves on. User-defined functions carry no
volatility marking and are not checked by these rules. A partition key is stricter: it accepts
immutable built-ins only, and no user-defined functions.

These rules apply when an object is created. An index, generated column, `CHECK` constraint or
domain that an earlier release accepted keeps loading and working as before. The error names a
function the way it is written: `current_user`, `current_date`, `localtime` and the other keyword
forms without parentheses, everything else as `random()`.

### JSON

Scalar: `json_typeof`, `json_array_length`, `to_json`, `to_jsonb`, `row_to_json`,
`json_build_object`, `json_build_array`, `json_object`, `json_set`, `jsonb_set`, `jsonb_set_lax`,
`json_insert`, `json_strip_nulls`, `json_pretty`, `json_extract_path`, `json_extract_path_text`,
`json_path_exists`, `json_path_match`, `json_path_query_first`, `json_path_query_array`,
`jsonb_exists`. Every `json_*` name has a `jsonb_*` twin.

A bare `NULL` value in `json_build_object` or `json_build_array` becomes JSON `null`; a `NULL` key
is refused. A number or a boolean casts to the JSON scalar it spells (`CAST(1 AS jsonb)` is `1`); a
text value casts by parsing it as a JSON document, so a JSON string needs its quotes
(`CAST('"x"' AS jsonb)`).

Set-returning: `json_array_elements`, `json_array_elements_text`, `json_each`, `json_each_text`,
`json_object_keys`, `json_path_query` (and their `jsonb_*` twins).

Aggregates: `json_agg`, `jsonb_agg`, `json_object_agg`, `jsonb_object_agg`.

### Arrays

`cardinality`, `array_length`, `array_lower`, `array_upper`, `array_dims`, `array_ndims`,
`array_fill`, `array_append`, `array_prepend`, `array_cat`, `array_position`, `array_positions`,
`array_remove`, `array_replace`, `array_to_string`, `string_to_array`, `string_to_table`,
`trim_array`, `unnest`, `generate_series`, and the aggregate `array_agg`.

### Ranges

`int4range`, `int8range`, `numrange`, `daterange`, `tsrange`, `tstzrange` (each takes
`(low, high [, bounds])` with bounds `'[)'`, `'(]'`, `'[]'`, `'()'`), `lower`, `upper`, `lower_inc`,
`upper_inc`, `lower_inf`, `upper_inf`, `isempty`, `range_merge`.

### Network

`host`, `masklen`, `family`, `network`, `broadcast`, `netmask`, `hostmask`, `set_masklen`,
`abbrev`, `inet_merge`, `inet_same_family`, `macaddr8_set7bit`.

### Bit strings and bytes

`get_bit`, `set_bit`, `get_byte`, `set_byte`, `bit_length`, `bit_count`.

### Geometry

`point`, `box`, `area`, `center`, `height`, `width`, `radius`, `diameter`, `npoints`, `isopen`,
`isclosed`.

### XML

`xmlcomment`, `xmlconcat`, `xml_is_well_formed`, `xml_is_well_formed_document`,
`xml_is_well_formed_content`.

### Vectors

`l1_distance`, `l2_distance`, `cosine_distance`, `inner_product`, `vector_dims`, `vector_norm`.

### Full-text search

`to_tsvector`, `to_tsquery`, `plainto_tsquery`, `phraseto_tsquery`, `tsquery_phrase`, `ts_rank`,
`ts_rank_cd`, `setweight`, `strip`,
`numnode`, `rrf_score`.

### Hashing, encryption, identifiers

`md5` (hex text), `sha224`, `sha256`, `sha384`, `sha512` (binary digests; wrap in `encode(..., 'hex')`
for text), `encrypt(value, key)` and `decrypt(value, key)` (deterministic AES-256-GCM-SIV, hex
ciphertext), `gen_random_uuid` (also `uuidv4` and `uuid_generate_v4`), `uuidv7`,
`uuid_extract_timestamp`, `uuid_extract_version`.

`uuidv7()` returns a version 7 UUID: a 48-bit Unix millisecond timestamp, a 12-bit fraction of the
millisecond, and 62 random bits from the operating system's secure random source. Values sort by
creation time, which keeps a `UUID PRIMARY KEY DEFAULT uuidv7()` index append-friendly. Each call
returns a value greater than every earlier call in the server process, so values grow within a
statement, a transaction and a session, even within one millisecond: when the clock has not moved
on, the 12-bit fraction is counted up instead, carrying into the millisecond field if it fills.
`uuidv7(shift)` moves the embedded timestamp by an interval (`uuidv7(INTERVAL '-1 hour')`); a
shift outside the range a UUID can hold is an error. `uuid_extract_timestamp(u)` returns the
timestamp of a version 1 or version 7 UUID as `TIMESTAMPTZ` (millisecond precision for version 7)
and NULL for other versions. `uuid_extract_version(u)` returns the version number as `SMALLINT`,
or NULL when the UUID is not of the standard variant.

```sql
CREATE TABLE events (id UUID PRIMARY KEY DEFAULT uuidv7(), body TEXT);
SELECT uuid_extract_timestamp(id), uuid_extract_version(id) FROM events ORDER BY id;
```

```sql
SELECT encode(sha256('secret'), 'hex') AS digest,
       decrypt(encrypt('card number', 'k3y'), 'k3y') AS round_trip;
```

### System

`version`, `current_user`, `session_user`, `user`, `current_database`, `current_catalog`,
`current_schema`, `current_setting(name)`, `nusadb_typeof(expr)`.

### Sequences

`nextval(name)`, `currval(name)`, `setval(name, value [, is_called])`.

### Aggregates

`count`, `sum`, `avg`, `min`, `max`, `string_agg`, `group_concat`, `array_agg`, `json_agg`,
`jsonb_agg`, `json_object_agg`, `jsonb_object_agg`, `bool_and`, `every`, `bool_or`, `bit_and`,
`bit_or`, `bit_xor`, `stddev`, `stddev_samp`, `stddev_pop`, `variance`, `var_samp`, `var_pop`,
`corr`, `covar_pop`, `covar_samp`, `regr_count`, `regr_avgx`, `regr_avgy`, `regr_sxx`, `regr_syy`,
`regr_sxy`, `regr_slope`, `regr_intercept`, `regr_r2`.

With `WITHIN GROUP (ORDER BY ...)`: `percentile_cont`, `percentile_disc`, `mode`, and the
hypothetical-set forms `rank`, `dense_rank`, `percent_rank`, `cume_dist`.

Every aggregate accepts `DISTINCT`, `FILTER (WHERE ...)` and an inner `ORDER BY`.

---

## Operators

| Kind | Operators |
| --- | --- |
| Comparison | `=` `<>` `!=` `<` `<=` `>` `>=` `BETWEEN` `IN` `IS [NOT] NULL` `IS [NOT] DISTINCT FROM` `IS [NOT] TRUE / FALSE / UNKNOWN` |
| Logic | `AND` `OR` `NOT` |
| Arithmetic | `+` `-` `*` `/` `%` `^` and integer `&` `\|` `#` `<<` `>>` `~` |
| Text | `\|\|` `LIKE` `ILIKE` `SIMILAR TO` `~` `~*` `!~` `!~*` |
| JSON | `->` `->>` `#>` `#>>` `@>` `<@` (`?` via `jsonb_exists`) |
| Array | `@>` `<@` `&&` `\|\|` `= ANY(...)` `= ALL(...)` subscript `a[1]` |
| Range | `@>` `<@` `&&` `+` `*` `-` `<<` `>>` `-\|-` |
| Network | `<<` `<<=` `>>` `>>=` `&&` `+` `-` |
| Vector | `<->` `<=>` `<#>` `<+>` |
| Full text | `@@` |
| Geometry | `@>` `<@` `&&` |

The integer bitwise operators work on `SMALLINT`, `INTEGER` and `BIGINT`. `&`, `|` and `#` (XOR)
over two widths give the wider type. A shift keeps the type of the value shifted: bits shifted past
its width are dropped, and the shift count is taken modulo the width (64 for `BIGINT`, 32
otherwise), so `CAST(1 AS bigint) << 63` is the smallest `BIGINT` and `CAST(1 AS bigint) << 64` is
`1`. On an `INTEGER` the shift is 32-bit: `1 << 40` is `256`, so cast to `BIGINT` for a 64-bit shift
(`CAST(1 AS bigint) << 40`). An operand that does not fit the result's width, such as a
`sum(integer)` past 2^31, is refused with `22003` rather than truncated.

A bare `NULL` operand takes the type of the other operand. When the operator has no form for two
values of that type, it takes the type that makes the operator resolve instead: `ts + NULL` is a
`NULL` timestamp (`timestamp + interval`), and `INTERVAL '1 day' * NULL` a `NULL` interval.

---

## Access control

Who may connect, and with which password, is decided by the server's `--auth-user` list (see
[deployment](deployment.md)); SQL roles decide what a connected user may do. Create a role with
the same name as each `--auth-user` entry, then grant it privileges. A user that authenticates but
has no role in the catalog can create its own tables and nothing else: reading or changing another
role's table needs a grant either way. `PASSWORD` in `CREATE ROLE` is refused, because a password
set there would never be checked at login.

```sql
CREATE ROLE analyst LOGIN;
CREATE USER app;                            -- USER is ROLE with LOGIN on
CREATE ROLE readers;                        -- a group: NOLOGIN by default
ALTER ROLE analyst NOLOGIN;
ALTER ROLE analyst LOGIN;
GRANT readers TO analyst;
REVOKE readers FROM analyst;
GRANT readers TO analyst;

GRANT SELECT ON customers TO readers;
GRANT SELECT (email, country) ON customers TO analyst;   -- column-level
GRANT INSERT, UPDATE ON orders TO analyst WITH GRANT OPTION;
GRANT ALL ON orders TO app;
REVOKE INSERT ON orders FROM analyst;

SET ROLE analyst;                           -- act as that role for the rest of the session
SELECT email FROM customers;                -- allowed through readers
INSERT INTO orders (customer, total) VALUES (1, 5);
-- ERROR 42501: permission denied: INSERT on column `customer` of table `public.orders`
RESET ROLE;
```

Privileges: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, `REFERENCES`, `TRIGGER`, `ALL`.
`SELECT`, `INSERT`, `UPDATE` and `REFERENCES` may name columns in parentheses. Role attributes:
`LOGIN` / `NOLOGIN` and `SUPERUSER` / `NOSUPERUSER`; the attribute is recorded in `nusadb_roles`, and
grants are what restrict a role. A role that lacks a privilege is refused with `42501`, and reading a
column outside its column grant is refused the same way even through `SELECT *`. The bootstrap
superuser `nusadb-root` bypasses every check.

Creating a table, view, materialized view, sequence, function, procedure, type or domain needs
`CREATE` on the schema it lands in. A schema's owner (the role that created it) always has it, and
other roles need a grant. Functions, procedures, types, domains and materialized views always live
in `public`. Every role may create in `public` until that is revoked:

```sql
REVOKE CREATE ON SCHEMA public FROM PUBLIC;   -- lock the default schema down
GRANT CREATE ON SCHEMA public TO app;         -- let one role back in
GRANT CREATE ON SCHEMA reporting TO analyst;
CREATE TABLE evil (id BIGINT);                -- as a role without the grant:
-- ERROR 42501: permission denied: CREATE on schema "public" is required to create objects in it
```

The revoke is stored in the catalog and survives a restart; `GRANT CREATE ON SCHEMA public TO
PUBLIC` restores the default. Temporary tables need no grant. `CREATE INDEX` needs ownership of the
table and `CREATE TRIGGER` the `TRIGGER` privilege on it. `CREATE SCHEMA` and `CREATE DATABASE` need
the `CREATEDB` role attribute. `USAGE` on a schema can be granted and revoked but is not checked
when a name is looked up.

Row-level security restricts which rows a role sees or may write:

```sql
GRANT SELECT, UPDATE ON orders TO analyst;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY own_rows ON orders
  FOR ALL TO analyst
  USING (customer = current_setting('app.customer')::BIGINT)
  WITH CHECK (customer = current_setting('app.customer')::BIGINT);

CREATE POLICY no_refunds ON orders AS RESTRICTIVE FOR UPDATE USING (total >= 0);

ALTER POLICY own_rows ON orders USING (customer = current_setting('app.customer')::BIGINT);
DROP POLICY no_refunds ON orders;

SET app.customer = '1';
SET ROLE analyst;
SELECT id, customer FROM orders;       -- only customer 1's rows
RESET ROLE;
```

`USING` filters rows read; `WITH CHECK` validates rows written (a write that fails it is `42501`).
Policies are `PERMISSIVE` by default (any one may allow a row) or `RESTRICTIVE` (all must allow).
Ownership, grants, role membership and policies are enforced inside the engine; a superuser
bypasses them.

---

## Session settings

```sql
SET search_path = reporting, public;
SET work_mem = '64MB';                 -- a bare number is kilobytes
SET statement_timeout = '30s';         -- a bare number is milliseconds; 0 disables
SET max_autocommit_retries = 50;       -- 0 turns the server-side retry off; capped at 100
SET hnsw_ef_search = 100;
SET default_transaction_isolation = 'serializable';
SET TIME ZONE '+07';                   -- session zone: UTC/GMT or a fixed offset
SET client_min_messages = warning;     -- client message threshold
SET myapp.tenant = '7';                -- application variables need a class prefix
SELECT current_setting('myapp.tenant');
SHOW search_path;
RESET search_path;
RESET ALL;
```

| Parameter | Effect |
| --- | --- |
| `search_path` | schema lookup order; `public` is always the fallback |
| `work_mem` | per-query memory budget for a sort, aggregate or join stage |
| `statement_timeout` | cancels a statement that runs longer (`57014`) |
| `max_autocommit_retries` | how many times a single auto-commit statement is retried on `40001` |
| `hnsw_ef_search` | candidate list size for vector index search; higher is more accurate and slower |
| `default_transaction_isolation` | isolation level for the next transactions in this session |
| `timezone` (also `SET TIME ZONE`) | the session time zone: `UTC`, `GMT`, or a fixed offset; see [Dates, times and intervals](#dates-times-and-intervals) |
| `client_min_messages` | client message threshold (`debug5`..`error`); the engine sends no sub-error messages, so any level is honoured |
| `client_encoding`, `datestyle`, `standard_conforming_strings` | accepted and echoed back; the engine is UTF-8, ISO dates |
| `server_version`, `server_encoding`, `integer_datetimes` | read-only (`55P02` on `SET`) |
| `<class>.<name>` | application-defined; any dotted name |

The server keeps some per-connection state in settings whose names start with a NUL character.
They are not parameters: `SET` refuses such a name (`55P02`), `SHOW` reports it as unknown
(`42704`), and `current_setting` returns `NULL` for it.

An unrecognised parameter name is an error (`42704`) rather than a silent no-op, and a bad value
for a known parameter is `22023`. `SET NAMES` is not accepted.

`SET LOCAL name = value` scopes the assignment to the current transaction: the value holds until
`COMMIT` or `ROLLBACK`, then reverts (outside a transaction block it has no effect). A plain
`SET` inside a transaction is itself transactional — it survives `COMMIT` but is undone by
`ROLLBACK`, and a later session-scoped `SET` beats an earlier `SET LOCAL` at commit. `SET LOCAL
TIME ZONE` works the same way. Parameter changes made after a `SAVEPOINT` are not undone by
`ROLLBACK TO SAVEPOINT` (only full transaction end settles them).

---

## Error codes

Every error carries a five-character SQLSTATE. A driver or application should branch on these, not
on the message text.

| Code | Meaning | What to do |
| --- | --- | --- |
| `0A000` | feature not supported | the statement is recognised but not built; see [Not accepted](#not-accepted) |
| `22001` | string too long for `VARCHAR(n)`, or a bit string too long for `BIT VARYING(n)` | shorten the value or widen the column |
| `22026` | a bit string whose length does not match `BIT(n)` | give exactly `n` bits, or cast explicitly |
| `22003` | numeric value out of range | integer overflow or a value outside a function's domain |
| `22007` | invalid date/time format | fix the literal |
| `22008` | date/time value out of range | a timestamp, date or interval result past the representable range |
| `22012` | division by zero | |
| `22023` | invalid parameter value | for example a bad `SET` value |
| `22P02` | invalid text representation | the text does not parse as the target type |
| `23502` | not-null violation | |
| `23503` | foreign-key violation | |
| `23505` | unique violation | |
| `23514` | check violation | also a row that fits no partition, or the wrong partition |
| `25001` | active transaction | `SET TRANSACTION` after the first query |
| `25P01` | no active transaction | `SAVEPOINT`, or `DECLARE` without `WITH HOLD`, outside `BEGIN` |
| `25P02` | transaction aborted | `ROLLBACK`, then start again |
| `26000` | unknown prepared statement | `PREPARE` it again |
| `2BP01` | dependent objects exist | drop them, or use `CASCADE` |
| `2F005` | function ended without `RETURN` | give every path through the body a `RETURN <value>` |
| `34000` | unknown cursor | also a cursor its transaction's end has closed |
| `3F000` | schema does not exist | |
| `40001` | serialization failure | retry the whole transaction |
| `40P01` | deadlock detected | retry the whole transaction |
| `42501` | permission denied | includes a row rejected by a policy |
| `42601` | syntax error | |
| `42701` / `42703` | duplicate / undefined column | |
| `42702` | ambiguous column | qualify it with the table name |
| `42704` | undefined object | unknown parameter, sequence, index, trigger, constraint |
| `42710` / `42723` | object / function already exists | |
| `42804` | datatype mismatch | cast explicitly |
| `42846` | cannot coerce | cast explicitly |
| `42883` | undefined function, or no matching signature | also an `EXECUTE` with the wrong argument count |
| `428C9` | writing a `GENERATED ALWAYS` column | use `OVERRIDING SYSTEM VALUE`, or omit the column |
| `42P17` | invalid object definition | a function that is not immutable in an index, generated column or partition key, or a volatile one in a `CHECK`; see [Volatility](#volatility) |
| `42P01` / `42P07` | undefined / duplicate table | |
| `42P03` | duplicate cursor | `CLOSE` the open cursor, or pick another name |
| `44000` | row violates a view's `WITH CHECK OPTION` | |
| `53300` | too many connections | only with `--reject-excess-connections`; back off and retry |
| `54000` | program limit exceeded | the request must get smaller |
| `54001` | statement too complex | trigger or procedure recursion limit |
| `55P02` | parameter cannot be changed | `server_version` and the other read-only settings |
| `57014` | query cancelled | `statement_timeout` or a client cancel |
| `P0001` | raised by a procedure's own `RAISE` | |
| `XX000` | internal error, or a memory ceiling was reached | the message names the limit and the flag that raises it |

---

## Behaviour worth knowing

These are deliberate choices. Each one is here because it can surprise you.

### Text ordering and collation

Comparison and ordering are bytewise, equivalent to a `C` collation. `COLLATE "C"` and `"POSIX"` are
accepted; locale collations are refused. Byte order is deterministic and does not change when a
system library is upgraded, which is what makes an index safe across upgrades. The visible effect is
that uppercase sorts before lowercase.

```sql
SELECT 'a' < 'B' AS byte_order;   -- false: 'B' (0x42) precedes 'a' (0x61)
```

### Fixed-width characters (`CHAR(n)`)

`CHAR(n)` is a fixed-width type. A value is stored with its trailing spaces trimmed, so comparison
and `||` treat `'ab'` and `'ab   '` as equal and `length` counts only the real characters. A `CHAR(n)`
column read out as a result on its own is blank-padded back to `n` characters, and `octet_length`
reports that full width.

### `NUMERIC` division precision

Division produces a fixed scale rather than deriving one from the operands. The value is the same;
only the number of digits kept differs. Round explicitly when a specific scale matters.

### How `JSON` is stored

`JSON` and `JSONB` both store a parsed value, so key order and insignificant whitespace from the
input text are not preserved. Keep the original text in a `TEXT` column when byte fidelity matters,
such as a signed payload.

### Floating-point text output is canonical

A `REAL` / `DOUBLE PRECISION` value renders in the shortest form that reads back to the same
number, switching to exponent notation outside `0.0001 .. 10^15` (`1e+20`, `2.5e-07`), and the
non-finite values render `Infinity`, `-Infinity` and `NaN`. The same rendering is used in query
results, casts to text and array elements, so the three always agree.

### Assigning between `TIMESTAMP` and `TIMESTAMPTZ`

A value of either type can be assigned to a column of the other and the instant passes through
unchanged. The two also compare and unify, the `TIMESTAMP` side read in the session time zone (see
[Dates, times and intervals](#dates-times-and-intervals)).

### Type coercion is explicit

The engine does not insert implicit coercions between unrelated types. Cast explicitly with
`::TYPE` when mixing them. The exceptions are the numeric widenings, `DATE` / `TIMESTAMP` /
`TIMESTAMPTZ` mixing, a quoted literal taking the type it is compared with, and a bare `NULL` taking
the type its operator needs.

### `SERIAL` columns are `integer`

`SERIAL`, `BIGSERIAL` and `SMALLSERIAL` all report their type as `integer`. Declare
`BIGINT GENERATED ALWAYS AS IDENTITY` when the key must be a `BIGINT`, for example because it is
compared with a `BIGINT` column in `IN (subquery)`, which requires both sides to have the same type.

### Identifiers fold to lower case

`Users` and `users` name the same table; `"Users"` (double-quoted) keeps its case and is a
different name. Identifiers are ASCII letters, digits and `_`.

### Comments may go anywhere

`-- …` runs to the end of the line and `/* … */` may span lines and nest
(`/* outer /* inner */ still a comment */`). A comment may come before, after or inside any
statement without changing it. Inside a string, a quoted identifier or a `$$ … $$` body the same
characters are plain text.

### Materialized views are snapshots

A materialized view holds the result from the last `REFRESH`. Incremental maintenance is opt-in with
`WITH (incremental = true)`, and asking for it on a body that cannot be maintained incrementally is
an error rather than a silent downgrade.

### Serialization conflicts and the auto-commit retry

Concurrency control is optimistic and locks do not wait: two sessions writing the same row make one
fail with SQLSTATE `40001` rather than queue.

A single statement outside a transaction is retried by the server, because it committed nothing
and showed the application no intermediate result. Inside `BEGIN ... COMMIT` it is not, because a
statement there may follow others whose results the application already acted on.

```sql
SET max_autocommit_retries = 50;   -- 0 turns the retry off
```

The budget is a bound, not a promise: a conflict that outlives it is reported unchanged, so an
application's own retry loop still sees its `40001`. A retry re-runs the statement, so `now()` and
`random()` produce fresh values and a sequence consumed by a failed attempt leaves a gap. Pair the
budget with `statement_timeout` if you want an upper bound on how long one statement may hold a
connection.

### Memory budgets: spill or fail, never silently swap

A sort, `DISTINCT`, `GROUP BY`, set operation or hash join whose input exceeds the work-memory
budget streams the overflow to the spill directory when one is configured (`--spill-dir`, or
automatically on a Linux host where the budget is detected), and a spilled sort, `DISTINCT` or
`GROUP BY` streams its result back from there, so the result may be larger than the budget. Without
a spill directory, a stage over `--work-mem` fails with an error naming the limit, the bytes
involved and how to raise it, and the server stays responsive. Window functions evaluate each
partition in memory when it fits and from disk when it does not; over a partition larger than the
budget only a `RANGE` / `GROUPS` frame with an offset, or an aggregate frame that has to hold its
rows (one with `EXCLUDE`, a floating-point `sum` or `avg`, or `array_agg` and similar), holding more
rows than the budget fails. `DISTINCT ON` and `ROLLUP` / `CUBE` / `GROUPING SETS` spill as well.
`SET work_mem` moves both the budget and the point where spilling starts.

### `TABLESAMPLE` samples per row

`TABLESAMPLE BERNOULLI (n)` and `SYSTEM (n)` keep each row with probability `n`%, drawn before
the `WHERE` filter runs. Both methods sample per row here. Elsewhere `SYSTEM` samples storage
pages, a physical grouping this engine does not reproduce; any subset is a valid
sample and the 0% / 100% endpoints agree exactly. `REPEATABLE (seed)` makes the draw
deterministic across repeated scans. A percentage outside `[0, 100]` is `2202H`; an unknown
method is `42704`. Supported on a base table (including an inheritance parent); not on a CTE,
view, derived table, or JOIN input.

### Capacity and memory

Table pages live in a page cache over the last checkpoint image: pages are loaded on first use,
clean pages are evicted and changed pages spill to a scratch file, so table data is bounded by
disk. See [deployment](deployment.md) for what still has to fit in memory.

---

## Not accepted

Recognised and refused with `0A000` and a clear message rather than half-implemented:

- `LIST` partitioning over several columns;
- `EXCLUDE` constraints with an operator other than `=` or `&&`, deferrable ones, and ones on a partitioned table;
- a multi-column `CYCLE` clause, or its `TO ... DEFAULT ...` marker form;
- `EXECUTE ... USING`, and arguments to a trigger function;
- aggregating a row value with anything but `count`;
- `LOCK TABLE` modes other than `ACCESS SHARE` and `ACCESS EXCLUSIVE`;
- `SET NAMES`, and an IANA region name as the session time zone;
- locale collations (only `"C"` / `"POSIX"`);
- `COPY` to or from a file or program (only `STDIN` / `STDOUT`);
- `BEGIN READ ONLY` / `SET TRANSACTION READ ONLY` over a connection;
- full-text configurations other than `simple`, `english`, and `indonesian`, and full-text prefix matching;
- `IGNORE NULLS` / `RESPECT NULLS` on window functions;
- a `DEFAULT` or `COLLATE` on a domain;
- `LANGUAGE` other than `SQL` for functions and procedures.

A statement may nest expressions and subqueries up to a fixed depth; one nested far past any
practical query (a hundred levels) is refused with `54001` before it is parsed, rather than risking
the stack. Ordinary queries are nowhere near this bound.
