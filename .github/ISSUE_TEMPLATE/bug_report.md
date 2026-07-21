---
name: Bug report
about: A correctness, crash, or data-integrity problem
title: "[bug] "
labels: bug
---

## Summary

<!-- What went wrong, in one or two sentences. Data-loss / corruption / wrong-result bugs
     are top priority — say so explicitly. -->

## Reproduction

- NusaDB version / commit:
- OS + arch:
- Steps (SQL, config, or a failing test/seed):

```sql
-- minimal SQL or commands that reproduce
```

## Expected vs actual

- Expected:
- Actual:

## Evidence

<!-- Logs, panic backtrace (RUST_BACKTRACE=1), DST seed, page/WAL dump, EXPLAIN output. -->

## Affected layer (if known)

<!-- e.g. nusadb-storage (L7), nusadb-lsm (L5), nusadb-txn (L4) ... -->
