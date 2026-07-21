/*
 * Integration test for the NusaDB C ABI. It loads the shared library at runtime (so it builds with
 * any C compiler, independent of the Rust target's import-library format) and runs assertions
 * against a server.
 *
 * Usage: test <library-path> <host> <port> [user] [password]
 * The accompanying harness boots a real nusadb-server and passes the port.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define LIB_HANDLE HMODULE
#define LIB_OPEN(path) LoadLibraryA(path)
#define LIB_SYM(h, name) ((void *)GetProcAddress((h), (name)))
#else
#include <dlfcn.h>
#define LIB_HANDLE void *
#define LIB_OPEN(path) dlopen((path), RTLD_NOW)
#define LIB_SYM(h, name) dlsym((h), (name))
#endif

typedef struct NusaConnection NusaConnection;
typedef struct NusaResult NusaResult;

typedef NusaConnection *(*fn_connect)(const char *, uint16_t, const char *, const char *, const char *);
typedef void (*fn_close)(NusaConnection *);
typedef const char *(*fn_error)(NusaConnection *);
typedef NusaResult *(*fn_query)(NusaConnection *, const char *);
typedef NusaResult *(*fn_query_params)(NusaConnection *, const char *, const char *const *, size_t);
typedef size_t (*fn_rows)(const NusaResult *);
typedef size_t (*fn_columns)(const NusaResult *);
typedef const char *(*fn_column_name)(const NusaResult *, size_t);
typedef const char *(*fn_value)(const NusaResult *, size_t, size_t);
typedef int (*fn_is_null)(const NusaResult *, size_t, size_t);
typedef const char *(*fn_command_tag)(const NusaResult *);
typedef void (*fn_result_free)(NusaResult *);
typedef int64_t (*fn_execute_many)(NusaConnection *, const char *, const char *const *, size_t, size_t, int64_t *);

static fn_connect connect_fn;
static fn_close close_fn;
static fn_error error_fn;
static fn_query query_fn;
static fn_query_params query_params_fn;
static fn_rows rows_fn;
static fn_columns columns_fn;
static fn_column_name column_name_fn;
static fn_value value_fn;
static fn_is_null is_null_fn;
static fn_command_tag command_tag_fn;
static fn_result_free result_free_fn;
static fn_execute_many execute_many_fn;

static int passed = 0;
static int failed = 0;

#define EXPECT(cond, what)                                          \
    do {                                                            \
        if (cond) {                                                 \
            passed++;                                               \
        } else {                                                    \
            failed++;                                               \
            printf("FAIL - %s (%s:%d)\n", (what), __FILE__, __LINE__); \
        }                                                           \
    } while (0)

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <library> <host> <port> [user] [password]\n", argv[0]);
        return 2;
    }
    const char *lib_path = argv[1];
    const char *host = argv[2];
    uint16_t port = (uint16_t)atoi(argv[3]);
    const char *user = argc > 4 ? argv[4] : "u";
    const char *password = argc > 5 ? argv[5] : NULL;

    LIB_HANDLE lib = LIB_OPEN(lib_path);
    if (!lib) {
        fprintf(stderr, "failed to load library: %s\n", lib_path);
        return 2;
    }

    connect_fn = (fn_connect)LIB_SYM(lib, "nusadb_connect");
    close_fn = (fn_close)LIB_SYM(lib, "nusadb_close");
    error_fn = (fn_error)LIB_SYM(lib, "nusadb_error");
    query_fn = (fn_query)LIB_SYM(lib, "nusadb_query");
    query_params_fn = (fn_query_params)LIB_SYM(lib, "nusadb_query_params");
    rows_fn = (fn_rows)LIB_SYM(lib, "nusadb_result_rows");
    columns_fn = (fn_columns)LIB_SYM(lib, "nusadb_result_columns");
    column_name_fn = (fn_column_name)LIB_SYM(lib, "nusadb_result_column_name");
    value_fn = (fn_value)LIB_SYM(lib, "nusadb_result_value");
    is_null_fn = (fn_is_null)LIB_SYM(lib, "nusadb_result_is_null");
    command_tag_fn = (fn_command_tag)LIB_SYM(lib, "nusadb_result_command_tag");
    result_free_fn = (fn_result_free)LIB_SYM(lib, "nusadb_result_free");
    execute_many_fn = (fn_execute_many)LIB_SYM(lib, "nusadb_execute_many");

    if (!connect_fn || !query_fn || !query_params_fn || !value_fn || !result_free_fn) {
        fprintf(stderr, "failed to resolve one or more symbols\n");
        return 2;
    }

    NusaConnection *conn = connect_fn(host, port, user, "nusadb", password);
    EXPECT(conn != NULL, "connect");
    if (!conn) {
        printf("\n%d passed, %d failed\n", passed, failed);
        return 1;
    }

    NusaResult *r = query_fn(conn, "CREATE TABLE capi_t (id INT NOT NULL, name TEXT)");
    EXPECT(r != NULL, "create table");
    if (r) result_free_fn(r);

    r = query_fn(conn, "INSERT INTO capi_t VALUES (5, 'alice')");
    EXPECT(r != NULL, "insert");
    if (r) {
        const char *tag = command_tag_fn(r);
        EXPECT(tag && strcmp(tag, "INSERT 1") == 0, "insert tag");
        result_free_fn(r);
    }

    /* Parameterised insert with a NULL. */
    const char *params2[2] = {"2", NULL};
    r = query_params_fn(conn, "INSERT INTO capi_t VALUES ($1, $2)", params2, 2);
    EXPECT(r != NULL, "param insert");
    if (r) result_free_fn(r);

    /* Rows come back ordered: (2, NULL) then (5, 'alice'). */
    r = query_fn(conn, "SELECT id, name FROM capi_t ORDER BY id");
    EXPECT(r != NULL, "select");
    if (r) {
        EXPECT(columns_fn(r) == 2, "two columns");
        EXPECT(rows_fn(r) == 2, "two rows");
        const char *c0 = column_name_fn(r, 0);
        EXPECT(c0 && strcmp(c0, "id") == 0, "column 0 name");
        const char *v00 = value_fn(r, 0, 0);
        EXPECT(v00 && strcmp(v00, "2") == 0, "row0 id == 2");
        EXPECT(is_null_fn(r, 0, 1) == 1, "row0 name is NULL");
        const char *v10 = value_fn(r, 1, 0);
        EXPECT(v10 && strcmp(v10, "5") == 0, "row1 id == 5");
        const char *v11 = value_fn(r, 1, 1);
        EXPECT(v11 && strcmp(v11, "alice") == 0, "row1 name == alice");
        result_free_fn(r);
    }

    /* A query error sets the error string and returns NULL; the connection survives. */
    r = query_fn(conn, "SELECT * FROM ghost");
    EXPECT(r == NULL, "missing table returns NULL");
    EXPECT(error_fn(conn) != NULL, "error message set");
    r = query_fn(conn, "SELECT 1");
    EXPECT(r != NULL && rows_fn(r) == 1, "connection usable after error");
    if (r) result_free_fn(r);

    /* Bulk insert via nusadb_execute_many: 3 parameter sets in one prepared statement. */
    EXPECT(execute_many_fn != NULL, "execute_many symbol resolved");
    if (execute_many_fn) {
        r = query_fn(conn, "CREATE TABLE capi_batch (id INT NOT NULL, name TEXT)");
        if (r) result_free_fn(r);
        const char *batch[6] = {"1", "a", "2", "b", "3", "c"};
        int64_t counts[3] = {0, 0, 0};
        int64_t total = execute_many_fn(conn, "INSERT INTO capi_batch VALUES ($1, $2)", batch, 2, 3, counts);
        EXPECT(total == 3, "execute_many total affected == 3");
        EXPECT(counts[0] == 1 && counts[1] == 1 && counts[2] == 1, "each set affected 1 row");
        r = query_fn(conn, "SELECT count(*) FROM capi_batch");
        EXPECT(r != NULL && rows_fn(r) == 1, "count query");
        if (r) {
            const char *v = value_fn(r, 0, 0);
            EXPECT(v && strcmp(v, "3") == 0, "three rows persisted");
            result_free_fn(r);
        }
    }

    close_fn(conn);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
