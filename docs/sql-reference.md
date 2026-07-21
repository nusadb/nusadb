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
