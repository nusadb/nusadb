/*
 * libnusa.h — C ABI for NusaDB (Nusa Wire Protocol, PROTOCOL_VERSION 1.0).
 *
 * The native bridge any FFI-capable language can call. Hand-maintained to match
 * crates/nusadb-capi/src/lib.rs (regenerate with cbindgen if preferred).
 *
 * Ownership:
 *   - nusadb_connect()         -> free with nusadb_close()
 *   - nusadb_query[_params]()  -> free with nusadb_result_free(); NULL means error
 *   - every const char* returned points into the owning struct and is valid until
 *     that struct is freed; do NOT free it yourself.
 */
#ifndef NUSADB_H
#define NUSADB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NusaConnection NusaConnection;
typedef struct NusaResult NusaResult;

/* Open a connection. password may be NULL (trust-on-startup). Returns NULL on failure. */
NusaConnection *nusadb_connect(const char *host, uint16_t port, const char *user,
                               const char *database, const char *password);

/* Close and free a connection (NULL-safe). */
void nusadb_close(NusaConnection *conn);

/* Last error message on conn, or NULL if the last call succeeded. Valid until the next call. */
const char *nusadb_error(NusaConnection *conn);

/* Run a simple query. Returns NULL on error (read nusadb_error). */
NusaResult *nusadb_query(NusaConnection *conn, const char *sql);

/* Run a parameterised query: params[i] is a C string or NULL (= SQL NULL). NULL on error. */
NusaResult *nusadb_query_params(NusaConnection *conn, const char *sql,
                                const char *const *params, size_t nparams);

/* Run sql once per parameter set (bulk insert/update), reusing one prepared statement. params is a
 * flat row-major array of (params_per_set * nsets) C strings (NULL = SQL NULL). Returns the total
 * affected-row count (>= 0) and, if out_counts is non-NULL, writes nsets per-set counts into it.
 * Returns -1 on error (read nusadb_error). */
int64_t nusadb_execute_many(NusaConnection *conn, const char *sql,
                            const char *const *params, size_t params_per_set,
                            size_t nsets, int64_t *out_counts);

/* Result accessors. */
size_t nusadb_result_rows(const NusaResult *result);
size_t nusadb_result_columns(const NusaResult *result);
const char *nusadb_result_column_name(const NusaResult *result, size_t col);
const char *nusadb_result_value(const NusaResult *result, size_t row, size_t col);
int nusadb_result_is_null(const NusaResult *result, size_t row, size_t col);
const char *nusadb_result_command_tag(const NusaResult *result);
void nusadb_result_free(NusaResult *result);

/* Cancellation. */
uint32_t nusadb_backend_pid(NusaConnection *conn);
int nusadb_cancel(NusaConnection *conn);

#ifdef __cplusplus
}
#endif

#endif /* NUSADB_H */
