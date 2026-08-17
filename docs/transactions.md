# Transactions & Concurrency

NusaDB uses MVCC with optimistic concurrency control (OCC): readers never block writers and writers
never block readers. The trade-off is that under write contention the engine aborts one of the
conflicting transactions instead of blocking it, so your application must be prepared to retry.

## What your application will see

| Situation | NusaDB behaviour | SQLSTATE |
| --- | --- | --- |
| Two transactions write the same row/key | Second writer aborts immediately (no wait) | `40001` |
| Deadlock between transactions | Both abort (no hang) | `40P01` |
| SERIALIZABLE rw-antidependency cycle | One transaction aborts at COMMIT | `40001` |

Neither `40001` (`serialization_failure`) nor `40P01` (`deadlock_detected`) is an application bug.
Both are the engine asking you to run the transaction again, so retry on the class — any code
starting `40` — rather than on one value. The engine's own retry helper treats both as retryable.
Integrity is preserved either way: exactly one writer wins, and there are no lost updates.

## The retry loop (required discipline)

Wrap every write transaction in a bounded retry loop that re-runs the whole transaction on a
class-`40` error.
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
            # Retry the whole class: `40001` and `40P01` are both "run it again".
            sqlstate = getattr(e, "sqlstate", None) or ""
            if not sqlstate.startswith("40") or attempt == attempts - 1:
                raise
            time.sleep(random.uniform(0, 0.05 * 2**attempt))  # jittered backoff
```

The same shape applies through any driver or ORM: catch class `40`, roll back, back off with
jitter, re-run. ORMs with a "retry on serialization failure" option (e.g. SQLAlchemy retrying
decorators) should enable it.

### Class `25` — the transaction is in the wrong state

Class `40` says *run it again*. Class `25` says *the transaction you are in cannot take this
statement*, which is a different instruction:

| Code | What happened | What to do |
| --- | --- | --- |
| `25P02` | a statement already failed, so the transaction is aborted and every later statement is refused | `ROLLBACK` (or roll back to a `SAVEPOINT` taken before the failure), then start again |
| `25P01` | `SAVEPOINT` / `RELEASE` / `ROLLBACK TO` outside a transaction block | open a transaction first; there is no savepoint scope without one |
| `25001` | `SET TRANSACTION ISOLATION LEVEL` after the transaction has already run a query | issue it before the transaction's first statement |

Two things the server does **not** treat as errors, which a client should not code around:

- `COMMIT` and `ROLLBACK` outside a transaction block **succeed as no-ops**. A pool that ends a
  transaction defensively before returning a connection can do so unconditionally.
- A redundant `BEGIN` inside an open transaction is a **no-op**, not `25001`. Transactions do not
  nest; use `SAVEPOINT` for a nested scope.

`25P02` is the one worth handling explicitly. The loop above already does the right thing — its
handler rolls back before deciding whether to retry — but a handler that instead re-runs the
*statement* on a connection in this state will get `25P02` again for every attempt, because nothing
clears it but ending the transaction. Nothing in class `25` is retryable in place.

(`25006`, *read-only transaction*, is defined but not reachable over the wire: `BEGIN READ ONLY`
and `SET TRANSACTION READ ONLY` are refused up front with `0A000` — see the note below.)

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

Higher isolation means more class-`40` aborts under contention, and the retry loop above is what
makes SERIALIZABLE practical.

## Timeouts and cancellation

- `SET statement_timeout = <ms>` bounds any single statement.
- A connection cap (`--max-connections`) queues excess connections by default;
  `--reject-excess-connections` refuses them immediately with `53300` so a pool can back off.
