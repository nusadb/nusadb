# SQL Reference

Everything on this page is accepted by the current engine. Each section shows the syntax and a
runnable example. Where NusaDB deliberately behaves differently from what you may expect, that is
called out in [Behaviour worth knowing](#behaviour-worth-knowing) rather than hidden.

Run the examples with `nusadb-cli`. They are written to be pasted in order.

- [Data types](#data-types)
- [Creating and changing objects](#creating-and-changing-objects)
- [Inserting and changing rows](#inserting-and-changing-rows)
- [Querying](#querying)
- [Functions](#functions)
- [Access control](#access-control)
- [Session settings](#session-settings)
- [Behaviour worth knowing](#behaviour-worth-knowing)

---

## Data types

### Numbers

| Type | Notes |
| --- | --- |
| `SMALLINT` / `INT` / `BIGINT` | 16 / 32 / 64-bit signed integers |
| `REAL` / `DOUBLE PRECISION` | IEEE-754 binary floating point |
| `NUMERIC(p,s)` | exact decimal; use this for money |

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

### Text and binary

| Type | Notes |
| --- | --- |
| `TEXT` | variable length, no limit |
| `VARCHAR(n)` | variable length, rejected past `n` |
| `CHAR(n)` | fixed width; see [`CHAR(n)`](#fixed-width-characters-charn) |
| `BYTEA` | binary, written as `'\xDEADBEEF'` |

```sql
SELECT 'hello' || ' ' || 'world' AS greeting,
       length('héllo')           AS chars,
       '\xdeadbeef'::BYTEA       AS blob;
```

### Booleans and UUIDs

```sql
SELECT TRUE AS t,
       'f'::BOOLEAN                              AS parsed,
       '5b8f...'::UUID IS NOT NULL               AS has_uuid;
```

### Dates, times and intervals

| Type | Example literal |
| --- | --- |
| `DATE` | `DATE '2026-08-30'` |
| `TIME` / `TIME WITH TIME ZONE` | `TIME '14:30:00'` |
| `TIMESTAMP` | `TIMESTAMP '2026-08-30 14:30:00'` |
| `TIMESTAMPTZ` | `TIMESTAMPTZ '2026-08-30 14:30:00+07'` |
| `INTERVAL` | `INTERVAL '1 day 2 hours'` |

```sql
SELECT now()                                   AS instant,
       date_trunc('month', DATE '2026-08-30')  AS month_start,
       EXTRACT(dow FROM DATE '2026-08-30')     AS day_of_week,
       DATE '2026-08-30' + INTERVAL '10 days'  AS later;
```

The session time zone is UTC and cannot currently be changed.

### JSON

`JSON` and `JSONB` both store a parsed value. See [How JSON is stored](#how-json-is-stored).

```sql
CREATE TABLE docs (id INT PRIMARY KEY, body JSONB);
INSERT INTO docs VALUES (1, '{"user":{"name":"ana","tags":["a","b"]},"active":true}');

SELECT body -> 'user' ->> 'name'          AS name,
       body #>> '{user,tags,0}'           AS first_tag,
       body @> '{"active":true}'          AS is_active,
       jsonb_path_query(body, '$.user.tags[*]') AS tag;
```

### Arrays

```sql
CREATE TABLE posts (id INT, tags TEXT[]);
INSERT INTO posts VALUES (1, ARRAY['rust','db']), (2, '{"sql"}');

SELECT id, array_length(tags, 1) AS n
FROM   posts
WHERE  tags @> ARRAY['rust'] OR 'sql' = ANY(tags);

SELECT id, tag FROM posts, unnest(tags) AS tag;
```

### Ranges

`INT4RANGE`, `INT8RANGE`, `NUMRANGE`, `DATERANGE`, `TSRANGE`.

```sql
SELECT int4range(1, 10)                       AS r,          -- [1,10)
       int4range(1, 10, '[]')                 AS closed,
       int4range(1,10) @> 5                   AS contains,
       int4range(1,10) && int4range(5,20)     AS overlaps,
       lower(int4range(1,10))                 AS lo,
       int4range(1,5) + int4range(4,9)        AS merged;
```

### Network and MAC addresses

`INET`, `CIDR`, `MACADDR`, `MACADDR8`.

```sql
SELECT '192.168.1.5/24'::INET            AS addr,
       host('192.168.1.5/24'::INET)      AS just_host,
       masklen('192.168.1.5/24'::INET)   AS bits,
       '192.168.1.0/24'::CIDR >> '192.168.1.5'::INET AS contains,
       '08:00:2b:01:02:03'::MACADDR      AS mac;
```

### Bit strings

`BIT(n)` and `BIT VARYING(n)`, with literals `B'1011'` and `X'0f'`.

```sql
SELECT B'1011' & B'1101'  AS and_,
       B'1011' | B'1101'  AS or_,
       ~B'1011'           AS not_,
       length(B'1011')    AS len;
```

### Geometric types

`POINT`, `BOX`, `CIRCLE`, `LSEG`, `LINE`, `PATH`, `POLYGON`.

```sql
CREATE TABLE places (id INT, at POINT, area POLYGON);
INSERT INTO places VALUES (1, '(1,2)', '((0,0),(0,4),(4,4),(4,0))');

SELECT id FROM places WHERE area @> at;      -- point inside polygon
SELECT '((0,0),(3,4))'::LSEG AS segment;
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

Operators: `<->` L2, `<=>` cosine, `<#>` negative inner product, `<+>` L1. An index answers only the
metric its operator class declared, so build one index per metric you query.

### Enumerated and composite types

```sql
CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');
CREATE TYPE point3 AS (x INT, y INT, z INT);

CREATE TABLE t (id INT, m mood, p point3);
INSERT INTO t VALUES (1, 'happy', ROW(1,2,3));
SELECT m, (p).y FROM t;
```

---

## Creating and changing objects

### Tables

```sql
CREATE TABLE customers (
  id       BIGSERIAL PRIMARY KEY,
  email    TEXT NOT NULL UNIQUE,
  country  TEXT DEFAULT 'ID',
  spend    NUMERIC(12,2) CHECK (spend >= 0),
  tax      NUMERIC(12,2) GENERATED ALWAYS AS (spend * 0.11) STORED,
  joined   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE orders (
  id       BIGSERIAL PRIMARY KEY,
  customer BIGINT REFERENCES customers (id) ON DELETE CASCADE,
  total    NUMERIC(12,2)
);

CREATE TABLE archive (LIKE orders);          -- copy the shape, not the rows
CREATE TABLE recent AS SELECT * FROM orders WHERE total > 100;
```

`CREATE TEMP TABLE` is supported. A temporary table belongs to the session that made it, is invisible
to others, and disappears when the session ends. `ON COMMIT DELETE ROWS` and `ON COMMIT DROP` are
accepted.

### Altering

```sql
ALTER TABLE orders ADD COLUMN note TEXT;
ALTER TABLE orders ALTER COLUMN note SET DEFAULT '';
ALTER TABLE orders ADD CONSTRAINT total_positive CHECK (total > 0);
ALTER TABLE orders RENAME TO sales_orders;
```

`RENAME COLUMN` succeeds when nothing else records the column by name. If a user-defined check,
foreign key, index, default, view, trigger or policy refers to it, the rename is refused and the
error names what is in the way. Drop that object, rename, and recreate it.

### Indexes

```sql
CREATE INDEX orders_customer ON orders (customer);
CREATE INDEX orders_big      ON orders (customer) WHERE total > 1000;   -- partial
CREATE INDEX orders_lower    ON customers ((lower(email)));             -- expression
CREATE UNIQUE INDEX ON customers (email);
CREATE INDEX orders_desc     ON orders (total DESC);
```

### Views, sequences, domains, schemas, databases

```sql
CREATE VIEW big_orders AS SELECT * FROM orders WHERE total > 1000;
CREATE MATERIALIZED VIEW daily AS SELECT date_trunc('day', created) d, count(*) FROM orders GROUP BY 1;
REFRESH MATERIALIZED VIEW daily;

CREATE SEQUENCE invoice_no START 1000 INCREMENT BY 5;
SELECT nextval('invoice_no'), currval('invoice_no');

CREATE DOMAIN positive_int AS INT CHECK (VALUE > 0);
CREATE SCHEMA reporting;
CREATE DATABASE app;
```

A view over a single table with no aggregate is **auto-updatable**: `INSERT`, `UPDATE` and `DELETE`
through it reach the base table.

A materialized view holds the result of its last `REFRESH`. Incremental maintenance is opt-in with
`WITH (incremental = true)`.

### Cursors

```sql
BEGIN;
DECLARE c CURSOR FOR SELECT id FROM orders ORDER BY id;
FETCH 10 FROM c;
FETCH NEXT FROM c;
CLOSE c;
COMMIT;
```

---

## Inserting and changing rows

```sql
INSERT INTO customers (email, country) VALUES ('ana@example.com', 'ID')
RETURNING id, joined;

INSERT INTO customers (email) VALUES ('ana@example.com')
ON CONFLICT DO NOTHING;

INSERT INTO customers (email, country) VALUES ('ana@example.com', 'SG')
ON CONFLICT (email) DO UPDATE SET country = EXCLUDED.country;

UPDATE orders o SET total = total * 1.1
FROM   customers c
WHERE  o.customer = c.id AND c.country = 'ID';

DELETE FROM orders USING customers c
WHERE  orders.customer = c.id AND c.country = 'XX';

MERGE INTO stock t USING shipment s ON t.sku = s.sku
WHEN MATCHED             THEN UPDATE SET qty = t.qty + s.qty
WHEN NOT MATCHED         THEN INSERT (sku, qty) VALUES (s.sku, s.qty)
WHEN NOT MATCHED BY SOURCE THEN UPDATE SET qty = 0;

TRUNCATE orders RESTART IDENTITY;
TRUNCATE customers CASCADE;      -- also empties tables that reference it
```

### Bulk load and export

`COPY` streams rows in one exchange instead of a round trip per row.

```bash
nusadb-cli -c "COPY customers FROM STDIN" < rows.tsv
nusadb-cli -c "COPY customers FROM STDIN WITH (FORMAT csv, HEADER)" < rows.csv
nusadb-cli -c "COPY customers TO STDOUT" > out.tsv
```

---

## Querying

### Joins, subqueries, set operations

```sql
SELECT c.email, count(o.id) AS orders
FROM   customers c
LEFT JOIN orders o ON o.customer = c.id
WHERE  c.country = 'ID'
GROUP BY c.email
HAVING count(o.id) > 2
ORDER BY orders DESC NULLS LAST
LIMIT 10;

SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer = c.id AND o.total > 500);

SELECT (id, email) IN ((1,'a@x'), (2,'b@x')) AS matched FROM customers;

SELECT email FROM customers
UNION      SELECT email FROM archive_customers
EXCEPT     SELECT email FROM blocked
INTERSECT  SELECT email FROM verified;

SELECT c.email, o.total
FROM   customers c
CROSS JOIN LATERAL (SELECT total FROM orders WHERE customer = c.id ORDER BY total DESC LIMIT 1) o;
```

### Window functions

```sql
SELECT country, email, spend,
       rank()       OVER w                         AS rk,
       sum(spend)   OVER (PARTITION BY country)    AS country_total,
       lag(spend)   OVER w                         AS previous,
       avg(spend)   OVER (ORDER BY joined ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving
FROM   customers
WINDOW w AS (PARTITION BY country ORDER BY spend DESC);
```

Frames support `ROWS`, `RANGE`, `GROUPS` and the `EXCLUDE` clause.

### Common table expressions

```sql
WITH RECURSIVE chain AS (
  SELECT id, manager, 1 AS depth FROM staff WHERE manager IS NULL
  UNION ALL
  SELECT s.id, s.manager, c.depth + 1
  FROM staff s JOIN chain c ON s.manager = c.id
)
SELECT depth, count(*) FROM chain GROUP BY depth ORDER BY depth;
```

`WITH x AS MATERIALIZED (...)` and `NOT MATERIALIZED` are accepted as hints.

### Grouping sets and aggregates with filters

```sql
SELECT country, date_trunc('month', joined) AS m, count(*)
FROM   customers
GROUP BY GROUPING SETS ((country), (m), ());

SELECT count(*) FILTER (WHERE spend > 100) AS big,
       string_agg(email, ',' ORDER BY email) AS list,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY spend) AS median,
       rank(500) WITHIN GROUP (ORDER BY spend) AS where_500_would_rank
FROM   customers;
```

### Row locking

```sql
SELECT * FROM orders WHERE id = 1 FOR UPDATE;
SELECT * FROM orders FOR UPDATE OF orders SKIP LOCKED;
SELECT * FROM orders FOR UPDATE NOWAIT;
SELECT * FROM orders FOR SHARE;
```

---

## Functions

A selection; the engine ships several hundred.

| Area | Examples |
| --- | --- |
| String | `length` `substring` `position` `overlay` `trim` `lpad` `split_part` `format` `regexp_replace` `regexp_substr` `regexp_count` `regexp_instr` `to_char` |
| Numeric | `abs` `round` `trunc` `ceil` `floor` `mod` `power` `sqrt` `ln` `log` `log10` `exp` `random` |
| Date/time | `now` `statement_timestamp` `localtimestamp` `date_trunc` `date_part` `EXTRACT` `age` `make_timestamptz` `to_timestamp` |
| JSON | `jsonb_set` `jsonb_set_lax` `jsonb_build_object` `json_object` `jsonb_agg` `json_object_agg` `jsonb_array_elements` `jsonb_each` `jsonb_path_query` `jsonb_path_match` `jsonb_extract_path` |
| Array | `array_agg` `array_length` `array_position` `array_remove` `array_fill` `unnest` |
| Aggregate | `count` `sum` `avg` `min` `max` `string_agg` `percentile_cont` `percentile_disc` `rank` `dense_rank` `percent_rank` `cume_dist` |
| Type | `nusadb_typeof` `quote_literal` `quote_nullable` `convert_to` `convert_from` |
| Vector | `l1_distance` `l2_distance` `cosine_distance` `inner_product` `vector_dims` `vector_norm` |

---

## Access control

```sql
CREATE ROLE analyst LOGIN PASSWORD 'secret';
CREATE ROLE readers;
GRANT readers TO analyst;

GRANT SELECT ON customers TO readers;
GRANT SELECT (email, country) ON customers TO analyst;   -- column-level
GRANT INSERT, UPDATE ON orders TO analyst WITH GRANT OPTION;
REVOKE INSERT ON orders FROM analyst;

SET ROLE analyst;
RESET ROLE;
```

Row-level security restricts which rows a role sees:

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY own_rows ON orders
  USING (customer = current_setting('app.customer')::BIGINT);
```

Ownership, grants, role membership and policies are enforced inside the engine. A superuser
bypasses these checks.

---

## Session settings

```sql
SET search_path = reporting, public;
SET work_mem = 67108864;
SET statement_timeout = '30s';
SET max_autocommit_retries = 50;
SET hnsw_ef_search = 100;
SET myapp.tenant = '7';        -- application variables need a class prefix
SHOW search_path;
RESET ALL;
```

An unrecognised parameter name is an error rather than a silent no-op, and a read-only parameter
reports that it cannot be changed.

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

`CHAR(n)` stores what you give it and does not pad to `n`.

### `NUMERIC` division precision

Division produces a fixed scale rather than deriving one from the operands. The value is the same;
only the number of digits kept differs. Round explicitly when a specific scale matters.

### How `JSON` is stored

`JSON` and `JSONB` both store a parsed value, so key order and insignificant whitespace from the
input text are not preserved. Keep the original text in a `TEXT` column when byte fidelity matters,
such as a signed payload.

### Assigning between `TIMESTAMP` and `TIMESTAMPTZ`

A value of either type can be assigned to a column of the other and the instant passes through
unchanged. Equality between the two types is still refused: this widened assignment, not comparison.

### Type coercion is explicit

The engine does not insert implicit coercions between unrelated types. Cast explicitly with
`::TYPE` when mixing them.

### Materialized views are snapshots

A materialized view holds the result from the last `REFRESH`. Incremental maintenance is opt-in with
`WITH (incremental = true)`, and asking for it on a body that cannot be maintained incrementally is
an error rather than a silent downgrade.

### Serialization conflicts and the auto-commit retry

Concurrency control is optimistic and locks do not wait: two sessions writing the same row make one
fail with SQLSTATE `40001` rather than queue.

A **single statement outside a transaction** is retried by the server, because it committed nothing
and showed the application no intermediate result. Inside `BEGIN … COMMIT` it is not, because a
statement there may follow others whose results the application already acted on.

```sql
SET max_autocommit_retries = 50;   -- 0 turns the retry off
```

The budget is a bound, not a promise: a conflict that outlives it is reported unchanged, so an
application's own retry loop still sees its `40001`. A retry re-runs the statement, so `now()` and
`random()` produce fresh values and a sequence consumed by a failed attempt leaves a gap. Pair the
budget with `statement_timeout` if you want an upper bound on how long one statement may hold a
connection.

### A query over its memory budget fails rather than spilling

With `--work-mem` set, a stage that materialises more than the budget fails with an error naming the
limit, the bytes involved and how to raise it. The server stays responsive. This keeps one large
query from slowing every other query, at the cost of not finishing it.

### Capacity is bounded by memory

Table pages live in memory and are not evicted to disk, so a database's working set must fit inside
the resident ceiling. See [deployment](deployment.md) before loading a large dataset.

---

## Not accepted

Recognised and refused with a clear message rather than half-implemented: advisory locks,
`PREPARE`/`EXECUTE` as statements over a connection (use the extended query protocol),
`SET TIME ZONE`, index methods other than B-tree and HNSW, and locale collations.
