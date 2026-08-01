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

`nusa_typeof(expr)` returns the static SQL type name of its argument as `TEXT` (`integer`,
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

## Materialized views stay fresh automatically

A materialized view whose body is a single-table projection with an optional `WHERE` filter is
kept up to date incrementally: every insert, update, or delete on its base table also adjusts the
view, so a query against the view always reflects the latest base rows without an explicit
`REFRESH`. This differs from the common convention where a materialized view holds a frozen
snapshot until refreshed — here that shape of view is never stale, and a `REFRESH` on it is a
no-op that recomputes the same contents.

Views whose body needs a join or an aggregate are **not** maintained incrementally: they hold the
result captured at `CREATE`/`REFRESH` time and change only when you `REFRESH` them again. So the
freshness of a materialized view depends on its shape:

| View body                            | Freshness              | `REFRESH` needed? |
| ------------------------------------ | ---------------------- | ----------------- |
| single-table projection (+ `WHERE`)  | always current         | no (auto)         |
| join / aggregate / grouped           | snapshot at last build | yes               |

The incremental path writes only the rows that changed, so it avoids rescanning the base table;
the trade-off is a small maintenance cost on each base-table write while such a view exists. If you
rely on a materialized view being a stable point-in-time snapshot, use a shape that is
refresh-only (for example, wrap the projection in a grouping or join), or query a regular `VIEW`
plus your own snapshot table.

## Session configuration variables

`SET name = value` records any variable name for the session — a built-in setting such as
`search_path` or `work_mem`, or an application's own, whether dotted (`SET myapp.request_id = '42'`)
or bare (`SET feature_flag = 'on'`). `SHOW name` and `current_setting('name')` read it back as text,
and an unset variable reads back as the empty string. This is broader than the reference engine,
which takes only dotted custom names and rejects an unqualified unknown one — NusaDB stores a bare
custom name too, so an application-defined session variable needs no special spelling.

A *value* that a built-in setting cannot use is still rejected loudly at `SET` time rather than
stored and then silently ignored: `SET work_mem = 'huge'` or an unparseable `statement_timeout`
fails immediately with an `invalid value for parameter` error. The trade-off of accepting any name
is that a misspelled built-in name (`word_mem` for `work_mem`) is taken as a new custom variable
rather than flagged, so the intended setting keeps its default — check `SHOW` if a setting does not
seem to take effect.
