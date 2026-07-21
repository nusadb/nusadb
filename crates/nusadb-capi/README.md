# nusadb-capi — C ABI for NusaDB

A C ABI (`include/nusadb.h`) over [`nusadb-libnusa`](../nusadb-libnusa) — the native bridge any
FFI-capable language (C, C++, Zig, Swift, Dart-FFI, Python `ctypes`, …) can call without
re-implementing the [Nusa Wire Protocol](../../docs/wire-protocol.md).

libnusa is async (tokio); each connection owns a current-thread runtime and blocks on each call, so
the C ABI is fully **synchronous**.

## Build

```bash
cargo build -p nusadb-capi        # produces a cdylib (.dll/.so/.dylib) and a staticlib
```

The header is hand-maintained at `include/nusadb.h` (it can also be regenerated with
[cbindgen](https://github.com/mozilla/cbindgen)).

## API (see `include/nusadb.h`)

```c
NusaConnection *conn = nusadb_connect("127.0.0.1", 5678, "nusa-root", "nusadb", "nusa-root");
NusaResult *r = nusadb_query(conn, "SELECT id, name FROM t WHERE id = $1");

/* parameterised: params[i] is a C string, or NULL for SQL NULL */
const char *params[1] = {"1"};
NusaResult *r2 = nusadb_query_params(conn, "SELECT * FROM t WHERE id = $1", params, 1);

for (size_t row = 0; row < nusadb_result_rows(r); row++) {
    const char *id = nusadb_result_value(r, row, 0);   /* NULL => SQL NULL */
    printf("%s\n", id ? id : "(null)");
}
nusadb_result_free(r);
nusadb_close(conn);
```

### Batch (bulk insert/update)

`nusadb_execute_many` runs `sql` once per parameter set, reusing a single prepared statement.
`params` is a flat row-major array of `params_per_set * nsets` C strings; it returns the total
affected-row count and writes per-set counts into `out_counts` (if non-NULL). The wire protocol has
no batch pipeline, so this is N round-trips, not one.

```c
const char *batch[6] = {"1", "a", "2", "b", "3", "c"};
int64_t counts[3];
int64_t total = nusadb_execute_many(conn, "INSERT INTO t VALUES ($1, $2)", batch, 2, 3, counts);
```

### Ownership

- `nusadb_connect` → free with `nusadb_close`.
- `nusadb_query` / `nusadb_query_params` → free with `nusadb_result_free`; `NULL` means error,
  read it with `nusadb_error`.
- Every `const char*` returned points into its owning struct and is valid until that struct is
  freed; the caller must not free it.

## Transactions

Inherits libnusa's behaviour: the server runs each statement on its own implicit transaction and
does not yet accept explicit `BEGIN`/`COMMIT` over the wire, so there is no transaction API yet (a
server follow-up).

## Test

```bash
cargo build -p nusadb-server
bash crates/nusadb-capi/test/run.sh
```

`run.sh` builds the library, compiles `test/test.c` (which loads the library at runtime, so it
works with any C compiler regardless of the Rust target's import-library format), boots a real
`nusadb-server` on an ephemeral port, and runs the test: connect, DDL/DML, a parameterised insert
with a NULL, a SELECT with typed reads, and error recovery.

## License

Apache-2.0.
