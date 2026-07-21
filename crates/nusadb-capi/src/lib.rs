//! C ABI over [`nusadb-libnusa`](nusadb_libnusa) — the native bridge any FFI-capable language
//! (C, Zig, Swift, Dart-FFI, …) can call without re-implementing the Nusa Wire Protocol.
//!
//! The hand-maintained header is `include/nusadb.h` (it can also be regenerated with cbindgen).
//! libnusa is async (tokio); each connection owns a current-thread runtime and blocks on each call,
//! so the C ABI is fully synchronous.
//!
//! # Memory ownership
//! - [`nusadb_connect`] returns an owned `NusaConnection*`; free it with [`nusadb_close`].
//! - [`nusadb_query`] / [`nusadb_query_params`] return an owned `NusaResult*`; free it with
//!   [`nusadb_result_free`]. `NULL` signals an error — read it with [`nusadb_error`].
//! - Every `const char*` returned (column names, values, tags, errors) points into the owning
//!   struct and stays valid until that struct is freed. The caller must not free those pointers.

// FFI: raw-pointer handling is inherently unsafe; this crate is the trust boundary.
#![allow(
    clippy::missing_safety_doc,
    reason = "each unsafe fn documents its contract in prose"
)]
// extern "C" accessors deref a raw pointer (not const-evaluable) and gain nothing from `const` —
// no C caller can use a const fn at compile time.
#![allow(
    clippy::missing_const_for_fn,
    reason = "extern \"C\" FFI fns; const is inapplicable"
)]

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use nusadb_libnusa::{Client, Config, Param};

/// An open connection plus its runtime and last-error slot. Opaque to C.
#[derive(Debug)]
pub struct NusaConnection {
    runtime: tokio::runtime::Runtime,
    client: Client,
    config: Config,
    last_error: Option<CString>,
}

/// A collected query result, with every value pre-converted to a stable NUL-terminated string.
/// Opaque to C.
#[derive(Debug)]
pub struct NusaResult {
    columns: Vec<CString>,
    rows: Vec<Vec<Option<CString>>>,
    tag: Option<CString>,
}

/// Read a `*const c_char` into an owned `String`, or `None` for a null pointer.
///
/// # Safety
/// `ptr` must be null or a valid NUL-terminated C string.
unsafe fn opt_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller guarantees `ptr` is a valid NUL-terminated string when non-null.
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// A value to NUL-terminated bytes; interior NULs are dropped (lossless for text the server emits).
fn to_cstring(s: String) -> CString {
    CString::new(s).unwrap_or_else(|e| {
        let bytes = e.into_vec();
        let cleaned: Vec<u8> = bytes.into_iter().filter(|&b| b != 0).collect();
        CString::new(cleaned).unwrap_or_default()
    })
}

/// Open a connection. Returns `NULL` on failure (no connection handle to read the error from, so
/// the failure is only signalled by `NULL`).
///
/// # Safety
/// `host`, `user`, and `database` must be valid NUL-terminated C strings. `password` may be `NULL`
/// (trust-on-startup) or a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_connect(
    host: *const c_char,
    port: u16,
    user: *const c_char,
    database: *const c_char,
    password: *const c_char,
) -> *mut NusaConnection {
    // SAFETY: forwarded contract on the input pointers.
    let (Some(host), Some(user), Some(database)) =
        (unsafe { (opt_string(host), opt_string(user), opt_string(database)) })
    else {
        return ptr::null_mut();
    };
    let password = unsafe { opt_string(password) };

    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return ptr::null_mut();
    };

    let mut config = Config::new(host, port, user, database);
    if let Some(pw) = password {
        config = config.password(pw);
    }

    let outcome = runtime.block_on(Client::connect(&config));
    outcome.map_or_else(
        |_| ptr::null_mut(),
        |client| {
            Box::into_raw(Box::new(NusaConnection {
                runtime,
                client,
                config,
                last_error: None,
            }))
        },
    )
}

/// Close and free a connection. Safe to call with `NULL`.
///
/// # Safety
/// `conn` must be a pointer returned by [`nusadb_connect`] and not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_close(conn: *mut NusaConnection) {
    if !conn.is_null() {
        // SAFETY: `conn` came from `Box::into_raw` in `nusadb_connect`.
        drop(unsafe { Box::from_raw(conn) });
    }
}

/// The last error message on `conn`, or `NULL` if the last call succeeded. The pointer is valid
/// until the next call on `conn`.
///
/// # Safety
/// `conn` must be a valid connection pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_error(conn: *mut NusaConnection) -> *const c_char {
    if conn.is_null() {
        return ptr::null();
    }
    // SAFETY: `conn` is a valid connection pointer per the contract.
    let conn = unsafe { &*conn };
    conn.last_error.as_ref().map_or(ptr::null(), |e| e.as_ptr())
}

fn store_result(conn: &mut NusaConnection, result: nusadb_libnusa::QueryResult) -> *mut NusaResult {
    conn.last_error = None;
    let columns = result
        .columns
        .iter()
        .map(|c| to_cstring(c.clone()))
        .collect();
    let rows = result
        .rows
        .iter()
        .map(|row| {
            (0..result.columns.len())
                .map(|i| {
                    row.get_str(i)
                        .ok()
                        .flatten()
                        .map(|s| to_cstring(s.to_owned()))
                })
                .collect()
        })
        .collect();
    let tag = result.tag.map(to_cstring);
    Box::into_raw(Box::new(NusaResult { columns, rows, tag }))
}

/// Run a simple (parameterless) query. Returns `NULL` on error; read it with [`nusadb_error`].
///
/// # Safety
/// `conn` must be valid and `sql` a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_query(
    conn: *mut NusaConnection,
    sql: *const c_char,
) -> *mut NusaResult {
    if conn.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: contract on `conn`.
    let conn = unsafe { &mut *conn };
    // SAFETY: contract on `sql`.
    let Some(sql) = (unsafe { opt_string(sql) }) else {
        conn.last_error = Some(to_cstring("nusadb: sql must not be null".to_owned()));
        return ptr::null_mut();
    };

    let outcome = conn.runtime.block_on(conn.client.simple_query(&sql));
    match outcome {
        Ok(result) => store_result(conn, result),
        Err(e) => {
            conn.last_error = Some(to_cstring(e.to_string()));
            ptr::null_mut()
        },
    }
}

/// Run a parameterised query. `params` is an array of `nparams` C strings; a `NULL` entry is SQL
/// `NULL`. Returns `NULL` on error; read it with [`nusadb_error`].
///
/// # Safety
/// `conn` must be valid, `sql` a valid C string, and `params` an array of `nparams` pointers, each
/// either `NULL` or a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_query_params(
    conn: *mut NusaConnection,
    sql: *const c_char,
    params: *const *const c_char,
    nparams: usize,
) -> *mut NusaResult {
    if conn.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: contract on `conn`.
    let conn = unsafe { &mut *conn };
    // SAFETY: contract on `sql`.
    let Some(sql) = (unsafe { opt_string(sql) }) else {
        conn.last_error = Some(to_cstring("nusadb: sql must not be null".to_owned()));
        return ptr::null_mut();
    };

    let mut bound: Vec<Param> = Vec::with_capacity(nparams);
    if !params.is_null() {
        for i in 0..nparams {
            // SAFETY: `params` points to `nparams` valid pointer slots per the contract.
            let entry = unsafe { *params.add(i) };
            // SAFETY: each entry is null or a valid C string per the contract.
            match unsafe { opt_string(entry) } {
                Some(text) => bound.push(Param::text(text)),
                None => bound.push(Param::null()),
            }
        }
    }

    let outcome = conn.runtime.block_on(conn.client.query(&sql, &bound));
    match outcome {
        Ok(result) => store_result(conn, result),
        Err(e) => {
            conn.last_error = Some(to_cstring(e.to_string()));
            ptr::null_mut()
        },
    }
}

/// Run `sql` once per parameter set, reusing a single prepared statement — the bulk insert/update path.
///
/// `params` is a flat row-major array of `params_per_set * nsets` C strings (set 0's params, then
/// set 1's, …); a `NULL` entry is SQL `NULL`. On success returns the total affected-row count
/// (>= 0) and, when `out_counts` is non-null, writes the `nsets` per-set counts into it. Returns
/// `-1` on error; read it with [`nusadb_error`]. The wire protocol has no batch pipeline, so this
/// is `nsets` round-trips, not one.
///
/// # Safety
/// `conn` must be valid; `sql` a valid C string; `params` either `NULL` or an array of
/// `params_per_set * nsets` pointers, each `NULL` or a valid C string; `out_counts` either `NULL`
/// or writable for `nsets` `int64_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_execute_many(
    conn: *mut NusaConnection,
    sql: *const c_char,
    params: *const *const c_char,
    params_per_set: usize,
    nsets: usize,
    out_counts: *mut i64,
) -> i64 {
    if conn.is_null() {
        return -1;
    }
    // SAFETY: contract on `conn`.
    let conn = unsafe { &mut *conn };
    // SAFETY: contract on `sql`.
    let Some(sql) = (unsafe { opt_string(sql) }) else {
        conn.last_error = Some(to_cstring("nusadb: sql must not be null".to_owned()));
        return -1;
    };

    let mut sets: Vec<Vec<Param>> = Vec::with_capacity(nsets);
    for s in 0..nsets {
        let mut bound: Vec<Param> = Vec::with_capacity(params_per_set);
        if !params.is_null() {
            for p in 0..params_per_set {
                // SAFETY: `params` points to `params_per_set * nsets` valid slots per the contract.
                let entry = unsafe { *params.add(s * params_per_set + p) };
                // SAFETY: each entry is null or a valid C string per the contract.
                match unsafe { opt_string(entry) } {
                    Some(text) => bound.push(Param::text(text)),
                    None => bound.push(Param::null()),
                }
            }
        }
        sets.push(bound);
    }

    let outcome = conn.runtime.block_on(conn.client.execute_many(&sql, &sets));
    match outcome {
        Ok(counts) => {
            let mut total: i64 = 0;
            for (i, c) in counts.iter().enumerate() {
                let n = c.map_or(0, |v| i64::try_from(v).unwrap_or(i64::MAX));
                total = total.saturating_add(n);
                if !out_counts.is_null() {
                    // SAFETY: `out_counts` is writable for `nsets` i64 per the contract; i < nsets.
                    unsafe {
                        *out_counts.add(i) = n;
                    }
                }
            }
            total
        },
        Err(e) => {
            conn.last_error = Some(to_cstring(e.to_string()));
            -1
        },
    }
}

/// The number of result rows.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_rows(result: *const NusaResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: contract on `result`.
    unsafe { &*result }.rows.len()
}

/// The number of result columns.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_columns(result: *const NusaResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: contract on `result`.
    unsafe { &*result }.columns.len()
}

/// The name of column `col`, or `NULL` if out of range. Valid until the result is freed.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_column_name(
    result: *const NusaResult,
    col: usize,
) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: contract on `result`.
    unsafe { &*result }
        .columns
        .get(col)
        .map_or(ptr::null(), |c| c.as_ptr())
}

/// The value at (`row`, `col`) as a C string, or `NULL` for SQL `NULL` or an out-of-range index.
///
/// Use [`nusadb_result_is_null`] to distinguish SQL `NULL` from out-of-range. Valid until the result
/// is freed.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_value(
    result: *const NusaResult,
    row: usize,
    col: usize,
) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: contract on `result`.
    unsafe { &*result }
        .rows
        .get(row)
        .and_then(|r| r.get(col))
        .and_then(|v| v.as_ref())
        .map_or(ptr::null(), |s| s.as_ptr())
}

/// Whether the value at (`row`, `col`) is SQL `NULL`. Returns `1` for `NULL` (or out of range),
/// `0` otherwise.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_is_null(
    result: *const NusaResult,
    row: usize,
    col: usize,
) -> i32 {
    if result.is_null() {
        return 1;
    }
    // SAFETY: contract on `result`.
    let is_value = unsafe { &*result }
        .rows
        .get(row)
        .and_then(|r| r.get(col))
        .is_some_and(Option::is_some);
    i32::from(!is_value)
}

/// The command tag (e.g. `"INSERT 1"`), or `NULL`. Valid until the result is freed.
///
/// # Safety
/// `result` must be a valid result pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_command_tag(result: *const NusaResult) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: contract on `result`.
    unsafe { &*result }
        .tag
        .as_ref()
        .map_or(ptr::null(), |t| t.as_ptr())
}

/// Free a result. Safe to call with `NULL`.
///
/// # Safety
/// `result` must be a pointer returned by a query function and not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_result_free(result: *mut NusaResult) {
    if !result.is_null() {
        // SAFETY: `result` came from `Box::into_raw` in a query function.
        drop(unsafe { Box::from_raw(result) });
    }
}

/// The cancellation key pid of `conn`, or `0` if none.
///
/// # Safety
/// `conn` must be a valid connection pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_backend_pid(conn: *mut NusaConnection) -> u32 {
    if conn.is_null() {
        return 0;
    }
    // SAFETY: contract on `conn`.
    unsafe { &*conn }.client.backend_key().map_or(0, |k| k.pid)
}

/// Cancel `conn`'s in-flight statement out of band (opens a fresh connection and sends a cancel
/// request). Returns `1` if the cancel was dispatched, `0` otherwise. Best effort.
///
/// # Safety
/// `conn` must be a valid connection pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nusadb_cancel(conn: *mut NusaConnection) -> i32 {
    if conn.is_null() {
        return 0;
    }
    // SAFETY: contract on `conn`.
    let conn = unsafe { &mut *conn };
    let Some(key) = conn.client.backend_key() else {
        return 0;
    };
    match conn.runtime.block_on(Client::cancel(&conn.config, key)) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}
