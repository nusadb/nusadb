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
