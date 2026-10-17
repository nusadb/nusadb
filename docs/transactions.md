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
| `25001` | `SET TRANSACTION ISOLATION LEVEL` after the transaction has already run a query, or `COMMIT PREPARED` / `ROLLBACK PREPARED` inside a transaction block | issue it before the transaction's first statement, or outside the block |

Two things the server does **not** treat as errors, which a client should not code around:

- `COMMIT` and `ROLLBACK` outside a transaction block **succeed as no-ops**. A pool that ends a
  transaction defensively before returning a connection can do so unconditionally.
- A redundant `BEGIN` inside an open transaction is a **no-op**, not `25001`. Transactions do not
  nest; use `SAVEPOINT` for a nested scope.

`25P02` is the one worth handling explicitly. The loop above already does the right thing — its
handler rolls back before deciding whether to retry — but a handler that instead re-runs the
*statement* on a connection in this state will get `25P02` again for every attempt, because nothing
clears it but ending the transaction. Nothing in class `25` is retryable in place.

A statement that fails inside a transaction leaves nothing of itself behind. Whatever it wrote
before the failure — rows, index entries, trigger side effects, maintained view rows — is undone
as part of the refusal, so the transaction holds exactly the effects of its successful statements.
Over the wire the transaction is then aborted (`25P02`) as described above; an embedded session
keeps it open, and a `ROLLBACK TO SAVEPOINT` in either case lands on a state the failed statement
never touched.

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

## Savepoints

A savepoint marks a point a transaction can roll back to without abandoning the whole transaction.
Transactions do not nest; a savepoint is the nested scope.

```sql
BEGIN;
INSERT INTO audit (event) VALUES ('import started');
SAVEPOINT before_rows;
INSERT INTO rows_in (payload) VALUES ('...');
ROLLBACK TO SAVEPOINT before_rows;   -- the audit row survives
RELEASE SAVEPOINT before_rows;       -- optional: forget the savepoint, keep the work
COMMIT;
```

`ROLLBACK TO SAVEPOINT` is also the way out of an aborted transaction (`25P02`) when the work before
the savepoint is worth keeping.

## Deferred constraint checks

A `DEFERRABLE` key, foreign key or exclusion constraint in deferred mode is checked when the
transaction commits instead of by each statement (declaring them and `SET CONSTRAINTS`:
[SQL reference](sql-reference.md#deferrable-constraints)). For an application that means:

- `COMMIT` (or `END`, or the end of an auto-committed statement) can fail with a constraint code:
  `23505`, `23503` or `23P01`. The transaction is then rolled back, exactly as for any failed
  `COMMIT`; nothing of it is kept. `PREPARE TRANSACTION` runs the same checks first, so a prepared
  transaction never fails one later, also not across a restart.
- `SET CONSTRAINTS ... IMMEDIATE` checks what is pending at that point. A violation there fails the
  transaction block like any failed statement: `ROLLBACK TO SAVEPOINT` to a savepoint taken before
  the offending writes recovers it.

```sql
BEGIN;
SAVEPOINT before_load;
INSERT INTO slot VALUES (1, 'clash');      -- deferred: accepted for now
SET CONSTRAINTS slot_n IMMEDIATE;          -- 23505: the block is failed
ROLLBACK TO SAVEPOINT before_load;         -- the insert, its pending check and the mode are undone
COMMIT;
```

- `ROLLBACK TO SAVEPOINT` drops the checks deferred after the savepoint, and restores the
  `SET CONSTRAINTS` mode it had.
- Deferring a check does not defer its locks. A deferred write takes the same key locks an
  immediate one does, when it writes, and holds them until the transaction ends, so the commit-time
  check cannot race another writer: a concurrent writer of the same key fails at once with
  `40001`, and one that committed earlier is seen by the check. A transaction holding a deferred
  child row also holds its parent key, so another transaction deleting that parent, or inserting
  it, fails with `40001` until the first one ends. As always, `40001` means retry (see the retry
  loop above); it can be the answer to `COMMIT` too.

## Row and table locks

Locks never wait, so `SELECT ... FOR UPDATE` on a row another transaction has locked fails at once
with `40001`; `NOWAIT` is accepted and changes nothing. `SKIP LOCKED` steps over such rows instead,
which is the usual shape for a job queue: each worker claims a different row and none of them
conflict.

```sql
BEGIN;
SELECT * FROM jobs WHERE state = 'queued' ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED;
UPDATE jobs SET state = 'running' WHERE id = 42;
COMMIT;
```

`LOCK TABLE name IN ACCESS SHARE MODE` and `IN ACCESS EXCLUSIVE MODE` are available inside a
transaction; the other modes are refused.

## Schema changes inside a transaction

DDL is transactional, and until it commits only its own transaction sees it. Every other session
keeps resolving tables, columns and indexes as they were last committed, whatever its isolation
level:

- a table another transaction created is not there yet (`42P01`), and creating one with the same
  name fails with `40001` until that transaction ends;
- a table another transaction altered keeps its committed columns, so `SELECT *` returns the
  committed shape and an `INSERT` without a column list counts the committed columns;
- a table another transaction renamed answers to its committed name only;
- a table another transaction dropped still exists, and reading or writing it fails with `40001`
  until the drop commits or rolls back;
- an index another transaction created is not used by the planner;
- constraints another transaction added or dropped (a `CHECK`, `UNIQUE`, primary key or foreign
  key) keep their committed state: `information_schema` lists the committed ones, and a foreign key
  still being added is not enforced against writes elsewhere.

`INSERT`, `UPDATE`, `DELETE`, `MERGE` and `COPY ... FROM` take the target table's shared lock before
they evaluate anything, so a write to a table another transaction is altering (a column, a
constraint, an index, a rename or a drop) fails at once with `40001`, even when no row matches,
rather than checking defaults or constraints against a definition that is not settled. The shared
lock is then held until the writing transaction ends, so DDL on that table from another session
fails with `40001` meanwhile (see [Two-phase commit](#two-phase-commit) for what the exclusive lock
covers). Dropping a foreign key also takes its referenced table exclusively.

Once the DDL commits, the next statement of every session sees it. Table definitions are not part
of a snapshot, so a transaction that began earlier still resolves the new definition. When the
change rewrote the table's rows (adding, dropping or retyping a column) or built an index on it, a
read of that table under a snapshot that does not see the change fails with `40001` instead of
returning rows of the wrong shape or an incomplete index result:

- a `REPEATABLE READ` or `SERIALIZABLE` transaction whose snapshot is older than such a change gets
  `40001` the next time it reads the table, and should be retried;
- a `READ COMMITTED` statement that resolved the table just before such a change committed and
  read it just after is retried automatically outside a transaction block, and reports `40001`
  inside one.

A rename, `SET NOT NULL`, `DROP NOT NULL` or a constraint change does not rewrite rows and does not
cause this.

## Cursors

A cursor reads a large result in pieces without holding it all on the client. Its rows are captured
when the cursor is declared. A cursor declared without `WITH HOLD` belongs to its transaction: it
can only be declared inside `BEGIN ... COMMIT` (`25P01` otherwise) and closes when the transaction
commits or rolls back. A `WITH HOLD` cursor stays open after its transaction commits, until `CLOSE`
or the end of the session, and is closed if that transaction rolls back. See
[the SQL reference](sql-reference.md#cursors) for `FETCH` and `MOVE`.

```sql
BEGIN;
DECLARE c CURSOR FOR SELECT id, email FROM customers ORDER BY id;
FETCH 100 FROM c;
FETCH NEXT FROM c;
CLOSE c;
COMMIT;
```

## Two-phase commit

`PREPARE TRANSACTION` is the first phase of a two-phase commit, for an external coordinator (an
XA transaction manager, a saga runner) that must make a transaction durable before it decides its
outcome. It ends the transaction on the connection without ending the transaction itself: the
writes are durable but invisible to everyone, the rows it changed or locked stay locked, and it
waits under its global identifier for the second phase, which any connection may run.

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 7;
PREPARE TRANSACTION 'transfer-42';   -- the connection is idle again

-- later, from this or any other connection, after the coordinator decided
COMMIT PREPARED 'transfer-42';       -- or ROLLBACK PREPARED 'transfer-42'
```

A prepared transaction survives a crash and a restart: after recovery it is still prepared, its
writes still invisible and its locks held again, until it is finished. Checkpoints go on while it
waits, and the log is truncated as usual; the checkpoint image carries the transaction. A standby
applies it from the primary's log and shows it prepared; it is finished on the primary (or on the
standby once promoted).

`SELECT * FROM nusadb_prepared_xacts` lists the database's prepared transactions: `transaction`,
`gid`, `prepared` (when it was prepared), `owner` (the role that prepared it) and `database`. A
superuser sees every row, any other role its own.

| Situation | SQLSTATE |
| --- | --- |
| `PREPARE TRANSACTION` with no transaction open (an error, not a warning, so a coordinator never believes it prepared a branch that does not exist) | `25P01` |
| `PREPARE TRANSACTION` in a failed transaction: it is rolled back and the tag is `ROLLBACK` | none |
| a deferred constraint is violated: it is rolled back (see [Deferred constraint checks](#deferred-constraint-checks)) | `23505` / `23503` / `23P01` |
| an identifier longer than 200 bytes | `22023` |
| an identifier already in use | `42710` |
| the transaction changed the schema, a sequence or statistics, wrote a temporary table, ran `LISTEN` / `UNLISTEN` / `NOTIFY`, or declared a cursor `WITH HOLD` | `0A000` |
| `--max-prepared-transactions` reached | `53400` |
| prepared transactions disabled (`--max-prepared-transactions 0`) | `55000` |
| `COMMIT PREPARED` / `ROLLBACK PREPARED` of an unknown identifier | `42704` |
| `COMMIT PREPARED` / `ROLLBACK PREPARED` inside a transaction block (the block is then failed) | `25001` |
| finishing another role's prepared transaction without being a superuser | `42501` |
| `DROP TABLE`, `TRUNCATE`, `ALTER TABLE` (adding a `CHECK`, `UNIQUE`, primary key or foreign key, or attaching or detaching a partition included), `CREATE INDEX`, `DROP INDEX`, `LOCK TABLE ... ACCESS EXCLUSIVE` or `DROP SCHEMA ... CASCADE` on a table a prepared transaction wrote; a foreign key added over existing rows also when it wrote the referenced table | `55006` |
| `DROP DATABASE` of a database with a prepared transaction waiting in it | `55006` |

A refused `PREPARE TRANSACTION` rolls the transaction back, whatever the reason. While a prepared
transaction waits, the tables it wrote cannot be dropped, truncated, altered or (re)indexed, and its
database cannot be dropped: such a statement fails at once with `55006` (object in use) rather than
waiting, and succeeds after `COMMIT PREPARED` or `ROLLBACK PREPARED`. The same DDL next to an
ordinary transaction that has written the table but not ended fails with the usual no-wait `40001`,
because a statement that drops, reshapes, re-indexes or validates a table takes it exclusively. Once
it holds the table, such a statement reads the latest committed rows (plus its own transaction's),
not the snapshot its transaction or statement began with, whatever the isolation level: a writer
that committed just before the lock was taken is therefore included when a constraint is validated
(`ADD CHECK`, `SET NOT NULL`, `ADD UNIQUE`, `ADD FOREIGN KEY`, `ATTACH PARTITION`), when rows are
rewritten (`ADD COLUMN ... DEFAULT`, a type change), and when an index is built. A
DDL statement run outside a transaction block is retried like any other autocommit conflict
(`max_autocommit_retries` attempts, 50 by default, with a growing back-off capped at 5 ms, so
about a quarter of a second in all) and goes through if the writer ends meanwhile; past that, or
inside a transaction block, the `40001` is reported for you to retry. A prepared
transaction is not a long transaction to leave lying around: it keeps its locks and holds back the
reclamation of old row versions until it is finished, so a coordinator should finish what it
prepared, and an operator can list the leftovers in `nusadb_prepared_xacts` and roll them back.

## Timeouts and cancellation

- `SET statement_timeout = '30s'` bounds any single statement on this session; the server flag
  `--statement-timeout` sets a default for every session. A cancelled statement reports `57014`.
- A connection cap (`--max-connections`) queues excess connections by default;
  `--reject-excess-connections` refuses them immediately with `53300` so a pool can back off.
- `CHECKPOINT` refuses while any transaction is open, including one on the connection issuing it;
  run it from a connection in autocommit.
