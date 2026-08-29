# NusaDB Documentation

User-facing documentation for the engine. The crate and layer map is in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); contribution rules are in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md). The same pages, with navigation, are published at
[nusadb.com/docs](https://nusadb.com/docs/).

| Page | Read it when |
| --- | --- |
| [`getting-started.md`](getting-started.md) | you have never run NusaDB: install (container or source), connect, a worked example, the shell, the drivers, common first errors |
| [`sql-reference.md`](sql-reference.md) | you are writing SQL: every type, statement and function the engine accepts, each with a runnable example, plus error codes and the behaviours where NusaDB chooses differently |
| [`transactions.md`](transactions.md) | you are writing an application: isolation levels, the class-`40` retry discipline, aborted transactions, savepoints, locks |
| [`deployment.md`](deployment.md) | you are running a server: every flag, memory limits and capacity, authentication and TLS, systemd, metrics, backup, upgrades |
| [`wire-protocol.md`](wire-protocol.md) | you are writing a client or driver: the Nusa Wire Protocol reference |

Every SQL example in these pages was run against the release it documents. Anything described as
planned is not in the release; features are listed only once they are implemented and tested.
