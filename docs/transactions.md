# Transactions & Concurrency

NusaDB uses MVCC with optimistic concurrency control (OCC): readers never block writers and writers
never block readers. The trade-off is that under write contention the engine aborts one of the
conflicting transactions instead of blocking it, so your application must be prepared to retry.

## What your application will see

| Situation | NusaDB behaviour | SQLSTATE |
| --- | --- | --- |
| Two transactions write the same row/key | Second writer aborts immediately (no wait) | `40001` |
| Deadlock between transactions | Both abort (no hang) | `40001` |
| SERIALIZABLE rw-antidependency cycle | One transaction aborts at COMMIT | `40001` |

`40001` (`serialization_failure`) is not an application bug. It is the engine asking you to run the
transaction again. Integrity is preserved either way: exactly one writer wins, and there are no lost
updates.

## The retry loop (required discipline)

Wrap every write transaction in a bounded retry loop that re-runs the whole transaction on `40001`.
Never retry a single statement in isolation, because the aborted transaction's earlier reads may be
stale.

Python:

```python
import time, random

def with_retry(conn, work, attempts=5):
    for attempt in range(attempts):
        try:
            cur = conn.cursor()
            cur.execute("BEGIN")
            result = work(cur)
            cur.execute("COMMIT")
            return result
        except Exception as e:
            cur.execute("ROLLBACK")
            if getattr(e, "sqlstate", None) != "40001" or attempt == attempts - 1:
                raise
            time.sleep(random.uniform(0, 0.05 * 2**attempt))  # jittered backoff
```

The same shape applies through any driver or ORM: catch `40001`, roll back, back off with jitter,
re-run. ORMs with a "retry on serialization failure" option (e.g. SQLAlchemy retrying decorators)
should enable it.

## Isolation levels over the wire

The default is `READ COMMITTED`. All four standard levels are supported end-to-end, requested any
of these ways:

```sql
BEGIN ISOLATION LEVEL SERIALIZABLE;                                   -- this transaction
BEGIN; SET TRANSACTION ISOLATION LEVEL REPEATABLE READ; ...           -- before its first query
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;  -- session default
SET default_transaction_isolation = 'serializable';                   -- session default (GUC)
```

`SET TRANSACTION ISOLATION LEVEL` after the transaction has already run a query is refused with
`25001`: it must be set before the transaction runs any query. `BEGIN READ ONLY` /
`SET TRANSACTION READ ONLY` are refused loudly over the wire for now (the wire layer does not yet
enforce access modes; an error is safer than silently granting a writable "read-only"
transaction).

Higher isolation means more `40001`s under contention, and the retry loop above is what makes
SERIALIZABLE practical.

## Timeouts and cancellation

- `SET statement_timeout = <ms>` bounds any single statement.
- A connection cap (`--max-connections`) queues excess connections by default;
  `--reject-excess-connections` refuses them immediately with `53300` so a pool can back off.
