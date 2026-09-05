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
| `DECLARE`, `FETCH`, `CLOSE` | [Cursors](#cursors) |
| `BEGIN`, `COMMIT`, `ROLLBACK`, `SAVEPOINT`, `SET TRANSACTION` | [Transactions and locking](#transactions-and-locking) |
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
`22012`.

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

A quote inside a literal is doubled: `'it''s'`. Dollar quoting (`$$ ... $$`) is accepted only for
routine bodies, not for ordinary string literals.

### Booleans and UUIDs

```sql
SELECT TRUE AS t,
       'f'::BOOLEAN                              AS parsed,
       gen_random_uuid()                         AS fresh,
       '5b8f0b3e-2c4d-4f7a-9d1e-1a2b3c4d5e6f'::UUID AS fixed;
```

### Dates, times and intervals

| Type | Example literal |
| --- | --- |
| `DATE` | `DATE '2026-08-30'` |
| `TIME` / `TIME WITH TIME ZONE` | `TIME '14:30:00'` |
| `TIMESTAMP` | `TIMESTAMP '2026-08-30 14:30:00'` |
| `TIMESTAMPTZ` | `TIMESTAMPTZ '2026-08-30 14:30:00+07'` |
| `INTERVAL` | `INTERVAL '1 day 2 hours'`, `INTERVAL '90 minutes'` |

```sql
SELECT now()                                   AS instant,
       date_trunc('month', DATE '2026-08-30')  AS month_start,
       EXTRACT(dow FROM DATE '2026-08-30')     AS day_of_week,
       DATE '2026-08-30' + INTERVAL '10 days'  AS later,
       age(TIMESTAMP '2026-08-30', TIMESTAMP '2020-01-01') AS elapsed,
       to_char(now(), 'YYYY-MM-DD HH24:MI')    AS formatted;
```

The session time zone is UTC and cannot be changed. `AT TIME ZONE` accepts `UTC` and numeric
offsets such as `+07:00`; named zones with daylight-saving rules are not available.

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

One-dimensional arrays of any scalar type.

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
CREATE TABLE recent AS SELECT * FROM orders WHERE total > 100;
DROP TABLE IF EXISTS recent;
```

Constraints available: `PRIMARY KEY`, `UNIQUE`, `NOT NULL`, `CHECK`, `FOREIGN KEY ... REFERENCES`
with `ON DELETE` / `ON UPDATE` `CASCADE`, `SET NULL`, `SET DEFAULT`, `RESTRICT`, `NO ACTION`; each
can be written at column or table level. Generated columns are `STORED`. Two more unique forms:

```sql
CREATE TABLE nnd (a INT, b TEXT, UNIQUE NULLS NOT DISTINCT (a));
INSERT INTO nnd (a) VALUES (NULL);
INSERT INTO nnd (a) VALUES (NULL);
-- ERROR 23505: with NULLS NOT DISTINCT, NULLs count as equal; plain UNIQUE allows both

CREATE TABLE bookings (room INT, day DATE, EXCLUDE (room WITH =, day WITH =));
```

`UNIQUE NULLS NOT DISTINCT` treats NULL keys as equal, so at most one NULL row exists per key; it
also works through `ALTER TABLE ... ADD`. An `EXCLUDE` constraint is accepted in its equality form
(every element `column WITH =`), where it is exactly `UNIQUE` on those columns; another operator,
`USING` a method other than btree, a `WHERE` clause, or an expression element is refused, because
those need index support the engine does not have.

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
  itself. An `UPDATE` that changes the partition key does not move the row between partitions, so
  change the key by delete-and-insert.
- The partition key must be written on every insert; the planner prunes partitions using `WHERE`
  conditions of the shape `key = / < / <= / > / >= constant` and `key BETWEEN a AND b`, which shows
  in `EXPLAIN` as fewer scanned partitions.
- `DROP TABLE` on the parent drops the whole partition tree with it.
- Parent and partitions may live in different schemas; qualified names work everywhere.

Not accepted: an expression as a partition key, `LIST` with more than one key column,
`MINVALUE` / `MAXVALUE` bounds, non-literal bounds, a partition declaring its own columns,
`UNIQUE` / `PRIMARY KEY` / `CHECK` / `FOREIGN KEY` on a partitioned parent (they would not span
partitions), and `INSERT ... ON CONFLICT` into a partitioned table.

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
ALTER TABLE orders ALTER COLUMN note SET DEFAULT '';
ALTER TABLE orders ALTER COLUMN note DROP DEFAULT;
ALTER TABLE orders ALTER COLUMN note SET NOT NULL;
ALTER TABLE orders ALTER COLUMN note DROP NOT NULL;
ALTER TABLE orders ALTER COLUMN note TYPE VARCHAR(200);
ALTER TABLE orders RENAME COLUMN note TO memo;
ALTER TABLE orders DROP COLUMN memo;
ALTER TABLE orders DROP COLUMN channel;
ALTER TABLE orders RENAME TO sales_orders;
ALTER TABLE sales_orders RENAME TO orders;

ALTER TABLE orders DROP CONSTRAINT total_positive;
ALTER TABLE orders ADD CONSTRAINT total_positive CHECK (total > 0);
ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer) REFERENCES customers (id);
ALTER TABLE orders DROP CONSTRAINT orders_customer_fk;

ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders DISABLE ROW LEVEL SECURITY;
```

`ALTER TABLE t DISABLE TRIGGER name` and `ENABLE TRIGGER name` are shown under [Triggers](#triggers).

`RENAME COLUMN` succeeds when nothing else records the column by name. If a user-defined check,
foreign key, index, default, view, trigger or policy refers to it, the rename is refused and the
error names what is in the way. Drop that object, rename, and recreate it.

Not accepted: `ADD COLUMN ... FIRST | AFTER`, an inline `CHECK` or `REFERENCES` on `ADD COLUMN`,
`DROP COLUMN ... CASCADE`, dropping several columns in one statement, `ADD CONSTRAINT ... NOT VALID`,
`ALTER COLUMN ... TYPE ... USING <expr>`, `ALTER COLUMN ... ADD GENERATED AS IDENTITY`, and a
comma-separated list of actions.

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
DROP INDEX IF EXISTS orders_desc;
```

Index methods are the default B-tree (`USING btree` is accepted and means the same) and `hnsw` for
vectors. A per-key `NULLS FIRST` / `NULLS LAST` is accepted; it does not change results, because a
query whose ordering column is nullable is served by an explicit sort that follows the query's own
`NULLS` clause. The planner picks an index from
collected statistics, so run `ANALYZE` after a large load (see [Maintenance](#maintenance)).

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
accepted yet.

### Comments

```sql
COMMENT ON TABLE orders IS 'One row per checkout';
COMMENT ON COLUMN orders.total IS 'Gross, including tax';
COMMENT ON TABLE orders IS NULL;        -- remove
```

Only table and column targets are accepted.

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
extra source.

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
```

Options: `FORMAT text | csv`, `DELIMITER 'c'`, `NULL 'string'`, `HEADER`, and for CSV only
`QUOTE 'c'` and `ESCAPE 'c'`. The text format is tab-delimited with `\N` for NULL. Export takes a
table (optionally with a column list) or a `(query)`; the query form is `TO STDOUT` only, runs with
the caller's privileges and row policies like any `SELECT`, and resolves unqualified names in the
default schema, so qualify names from other schemas. Load takes a table only, and binary format is
not available. A load larger than `--copy-max-bytes` is refused rather than buffered without
bound.

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
`tsquery` with `@@`. Two configurations exist: `simple` lower-cases words, and `english` (the
default) also removes stop words and stems.

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

Query operators: `&` and, `|` or, `!` not, with parentheses. `to_tsquery` takes that syntax;
`plainto_tsquery` takes plain words and ANDs them. Ranking: `ts_rank`, `ts_rank_cd`;
`setweight(vector, 'A')` marks lexemes, `strip(vector)` drops positions. Other configurations,
phrase search (`<->`), and prefix matching are refused with a clear error rather than approximated.
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

NusaScript statements: `DECLARE name TYPE [DEFAULT expr]`, `SET name = expr`,
`IF ... THEN ... [ELSIF ...] [ELSE ...] END IF`, `WHILE cond LOOP ... END LOOP`,
`FOR i IN low TO high LOOP ... END LOOP`, `RAISE 'message'` (reported as `P0001`), `RETURN`, and any
SQL data statement. An `EXCEPTION WHEN OTHERS THEN` handler rolls the body's writes back to a
savepoint and runs in their place. Variables may be referenced by name inside embedded SQL; a
column of the same name wins. `LANGUAGE SQL` is the only language.

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
affected row.

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
CREATE FUNCTION log_stock() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO stock_log (sku, before, after) VALUES (OLD.sku, OLD.qty, NEW.qty);
END
$$;

CREATE TRIGGER stock_fn AFTER UPDATE ON stock FOR EACH ROW
WHEN (NEW.qty <> OLD.qty) EXECUTE FUNCTION log_stock();
```

The function must take no parameters and have a `BEGIN ... END` body; that is checked when the
trigger is created, not on first fire. `EXECUTE PROCEDURE` is a synonym. `NEW` and `OLD` are
available throughout the body, a `RETURN` only ends it, and a `BEFORE` trigger cannot change or
skip the row either way. Function arguments in the trigger (`EXECUTE FUNCTION f('x')`) are not
accepted. The action runs in the same transaction as the triggering statement; an error in it
aborts that statement. Cascading triggers are bounded by a depth limit (`54001`). `INSTEAD OF`
triggers are not accepted.

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

A cursor reads a result in pieces inside a transaction.

```sql
BEGIN;
DECLARE c CURSOR FOR SELECT id, email FROM customers ORDER BY id;
FETCH 2 FROM c;
FETCH NEXT FROM c;
FETCH ALL FROM c;
CLOSE c;
COMMIT;
```

Cursors are forward-only and are closed when the transaction ends. Fetching from an unknown
cursor is `34000`.

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
COMMIT;                                            -- or END
ROLLBACK;

SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;   -- session default
```

The default isolation level is `READ COMMITTED`. A statement outside `BEGIN` runs in its own
transaction. After an error inside a transaction, every further statement is refused with `25P02`
until `ROLLBACK` (or `ROLLBACK TO SAVEPOINT`).

### Row and table locks

Locks never wait. A row someone else has locked fails immediately with `40001`, so `NOWAIT` is
implied; `SKIP LOCKED` steps over such rows instead, which is the usual shape for a job queue.

```sql
SELECT * FROM orders WHERE id = 1 FOR UPDATE;
SELECT * FROM orders ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED;   -- claim one job
SELECT * FROM orders FOR UPDATE OF orders NOWAIT;
SELECT * FROM orders FOR SHARE;

LOCK TABLE orders IN ACCESS EXCLUSIVE MODE;        -- inside a transaction
LOCK TABLE orders, customers IN ACCESS SHARE MODE;
```

`LOCK TABLE` accepts `ACCESS SHARE` and `ACCESS EXCLUSIVE`; the other modes are refused.

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
REINDEX TABLE orders;     -- accepted for compatibility; indexes are always consistent, so it is a no-op
CLUSTER orders;           -- accepted for compatibility; rows are already clustered by row id, so it is a no-op
CREATE EXTENSION IF NOT EXISTS vector;   -- accepted for compatibility; installs nothing, every type is built in
CHECKPOINT;               -- fold the log into an image and truncate it; needs a quiet engine
```

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

SELECT * FROM nusadb_databases;       -- every database in this server
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
`information_schema`. A catalog that has never been written to may not exist yet, in which case
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

`abs`, `round`, `trunc`, `ceil`, `ceiling`, `floor`, `sign`, `mod`, `div`, `power`, `pow`, `sqrt`,
`cbrt`, `exp`, `ln`, `log` (`log(x)` is base 10, `log(b, x)` is base `b`), `log10`, `sin`, `cos`,
`tan`, `cot`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
`degrees`, `radians`, `pi`, `gcd`, `lcm`, `factorial`, `bit_count`, `to_hex`, `to_number`,
`width_bucket`, `scale`, `min_scale`, `trim_scale`, `isfinite`, `random`, `setseed`.

### Conditional

`coalesce`, `nullif`, `greatest`, `least`, `nvl`, `ifnull`, `num_nulls`, `num_nonnulls`, `CASE`,
`CAST(x AS type)`, `x::type`, `TRY_CAST(x AS type)` (NULL instead of an error).

### Date and time

`now`, `transaction_timestamp`, `statement_timestamp`, `current_timestamp`, `current_date`,
`current_time`, `localtimestamp`, `date_trunc`, `date_part`, `EXTRACT(field FROM x)`, `date_bin`,
`age`, `to_char`, `to_date`, `to_timestamp`, `make_date`, `make_time`, `make_timestamp`,
`make_timestamptz`, `make_interval`, `justify_days`, `justify_hours`, `justify_interval`,
`x AT TIME ZONE zone`.

`EXTRACT` fields: `year`, `quarter`, `month`, `week`, `day`, `hour`, `minute`, `second`, `dow`,
`isodow`, `doy`, `isoyear`, `epoch`, `decade`, `century`, `millennium`, `microseconds`,
`milliseconds`, `julian`, `timezone`, `timezone_hour`, `timezone_minute`. `date_trunc` accepts the
units from `microseconds` up to `millennium`. `now()` is fixed for the whole statement.

### JSON

Scalar: `json_typeof`, `json_array_length`, `to_json`, `to_jsonb`, `row_to_json`,
`json_build_object`, `json_build_array`, `json_object`, `json_set`, `jsonb_set`, `jsonb_set_lax`,
`json_insert`, `json_strip_nulls`, `json_pretty`, `json_extract_path`, `json_extract_path_text`,
`json_path_exists`, `json_path_match`, `json_path_query_first`, `json_path_query_array`,
`jsonb_exists`. Every `json_*` name has a `jsonb_*` twin.

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

`to_tsvector`, `to_tsquery`, `plainto_tsquery`, `ts_rank`, `ts_rank_cd`, `setweight`, `strip`,
`numnode`, `rrf_score`.

### Hashing, encryption, identifiers

`md5` (hex text), `sha224`, `sha256`, `sha384`, `sha512` (binary digests; wrap in `encode(..., 'hex')`
for text), `encrypt(value, key)` and `decrypt(value, key)` (deterministic AES-256-GCM-SIV, hex
ciphertext), `gen_random_uuid`, `uuid_generate_v4`.

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
| Arithmetic | `+` `-` `*` `/` `%` `^` and integer `&` `\|` `<<` `>>` `~` |
| Text | `\|\|` `LIKE` `ILIKE` `SIMILAR TO` `~` `~*` `!~` `!~*` |
| JSON | `->` `->>` `#>` `#>>` `@>` `<@` (`?` via `jsonb_exists`) |
| Array | `@>` `<@` `&&` `\|\|` `= ANY(...)` `= ALL(...)` subscript `a[1]` |
| Range | `@>` `<@` `&&` `+` `*` `-` `<<` `>>` `-\|-` |
| Network | `<<` `<<=` `>>` `>>=` `&&` `+` `-` |
| Vector | `<->` `<=>` `<#>` `<+>` |
| Full text | `@@` |
| Geometry | `@>` `<@` `&&` |

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
| `client_encoding`, `datestyle`, `timezone`, `standard_conforming_strings` | accepted and echoed back; the engine is UTF-8, ISO dates, UTC |
| `server_version`, `server_encoding`, `integer_datetimes` | read-only (`55P02` on `SET`) |
| `<class>.<name>` | application-defined; any dotted name |

An unrecognised parameter name is an error (`42704`) rather than a silent no-op.
`SET LOCAL`, `SET TIME ZONE` and `SET NAMES` are not accepted.

---

## Error codes

Every error carries a five-character SQLSTATE. A driver or application should branch on these, not
on the message text.

| Code | Meaning | What to do |
| --- | --- | --- |
| `0A000` | feature not supported | the statement is recognised but not built; see [Not accepted](#not-accepted) |
| `22001` | string too long for `VARCHAR(n)` | shorten the value or widen the column |
| `22003` | numeric value out of range | integer overflow or a value outside a function's domain |
| `22007` | invalid date/time format | fix the literal |
| `22012` | division by zero | |
| `22023` | invalid parameter value | for example a bad `SET` value |
| `22P02` | invalid text representation | the text does not parse as the target type |
| `23502` | not-null violation | |
| `23503` | foreign-key violation | |
| `23505` | unique violation | |
| `23514` | check violation | also a row that fits no partition, or the wrong partition |
| `25001` | active transaction | `SET TRANSACTION` after the first query |
| `25P01` | no active transaction | `SAVEPOINT` outside `BEGIN` |
| `25P02` | transaction aborted | `ROLLBACK`, then start again |
| `26000` | unknown prepared statement | `PREPARE` it again |
| `2BP01` | dependent objects exist | drop them, or use `CASCADE` |
| `2F005` | function ended without `RETURN` | give every path through the body a `RETURN <value>` |
| `34000` | unknown cursor | |
| `3F000` | schema does not exist | |
| `40001` | serialization failure | retry the whole transaction |
| `40P01` | deadlock detected | retry the whole transaction |
| `42501` | permission denied | includes a row rejected by a policy |
| `42601` | syntax error | |
| `42701` / `42703` | duplicate / undefined column | |
| `42702` | ambiguous column | qualify it with the table name |
| `42704` | undefined object | unknown parameter, sequence, index, trigger |
| `42710` / `42723` | object / function already exists | |
| `42804` | datatype mismatch | cast explicitly |
| `42846` | cannot coerce | cast explicitly |
| `42883` | undefined function, or no matching signature | also an `EXECUTE` with the wrong argument count |
| `428C9` | writing a `GENERATED ALWAYS` column | use `OVERRIDING SYSTEM VALUE`, or omit the column |
| `42P01` / `42P07` | undefined / duplicate table | |
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
unchanged. Equality between the two types is still refused: this widened assignment, not comparison.

### Type coercion is explicit

The engine does not insert implicit coercions between unrelated types. Cast explicitly with
`::TYPE` when mixing them.

### `SERIAL` columns are `integer`

`SERIAL`, `BIGSERIAL` and `SMALLSERIAL` all report their type as `integer`. Declare
`BIGINT GENERATED ALWAYS AS IDENTITY` when the key must be a `BIGINT`, for example because it is
compared with a `BIGINT` column in `IN (subquery)`, which requires both sides to have the same type.

### Identifiers fold to lower case

`Users` and `users` name the same table; `"Users"` (double-quoted) keeps its case and is a
different name. Identifiers are ASCII letters, digits and `_`.

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

A sort or hash join whose input exceeds the work-memory budget streams the overflow to the spill
directory when one is configured (`--spill-dir`, or automatically on a Linux host where the budget
is detected). Without a spill directory, a stage over `--work-mem` fails with an error naming the
limit, the bytes involved and how to raise it, and the server stays responsive. Aggregation,
`DISTINCT` and window functions do not spill yet; they fail at the budget.

### Capacity is bounded by memory

Table pages live in memory and are not evicted to disk, so a database's working set must fit inside
the resident ceiling. See [deployment](deployment.md) before loading a large dataset.

---

## Not accepted

Recognised and refused with `0A000` and a clear message rather than half-implemented:

- an expression as a partition key, `LIST` partitioning over several columns, `MINVALUE` /
  `MAXVALUE` partition bounds, and `INSERT ... ON CONFLICT` into a partitioned table;
- `EXCLUDE` constraints with an operator other than `=`;
- a multi-column `CYCLE` clause, or its `TO ... DEFAULT ...` marker form;
- `EXECUTE ... USING`, and arguments to a trigger function;
- aggregating a row value with anything but `count`;
- dollar-quoted string literals outside a routine body;
- `LOCK TABLE` modes other than `ACCESS SHARE` and `ACCESS EXCLUSIVE`;
- `INSTEAD OF` triggers;
- `SET TIME ZONE`, `SET LOCAL`, `SET NAMES`;
- index methods other than B-tree and HNSW;
- locale collations (only `"C"` / `"POSIX"`);
- `COPY` to or from a file or program (only `STDIN` / `STDOUT`), and `COPY ... BINARY`;
- `BEGIN READ ONLY` / `SET TRANSACTION READ ONLY` over a connection;
- full-text configurations other than `simple` and `english`, phrase search, prefix matching;
- `IGNORE NULLS` / `RESPECT NULLS` on window functions;
- multi-dimensional arrays;
- a `DEFAULT` or `COLLATE` on a domain;
- `LANGUAGE` other than `SQL` for functions and procedures.
