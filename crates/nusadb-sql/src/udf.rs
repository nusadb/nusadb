//! User-defined scalar functions.
//!
//! A scalar UDF is a Rust function registered under a SQL name. Once registered it is callable as
//! `name(args)` in any expression, exactly like a built-in: the parser emits a generic
//! [`ast::Expr::FunctionCall`](crate::ast::Expr::FunctionCall) for any name it does not recognise as
//! a built-in, the analyzer resolves it against this registry (checking arity + argument types against
//! the declared signature), and the executor calls the registered function per row.
//!
//! Registration is process-global (the registry outlives any single connection), so UDFs are
//! installed once at start-up via [`register_scalar_udf`]. Aggregate and window UDFs (`AggregateUdf`
//! / `WindowUdf`) are honest follow-ups; this is the scalar core.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use nusadb_core::ColumnType;

use crate::ast::Value;
use crate::error::Error;

/// The implementation of a scalar UDF: maps its argument values (already coerced to the declared
/// argument types, in order) to a result value, or an error message describing why it failed.
pub type ScalarUdfFn = Arc<dyn Fn(&[Value]) -> Result<Value, String> + Send + Sync>;

/// A registered scalar UDF: its declared signature plus its implementation.
struct ScalarUdf {
    arg_types: Vec<ColumnType>,
    return_type: ColumnType,
    func: ScalarUdfFn,
}

/// The process-global scalar-UDF registry, keyed by folded (lowercase) function name.
static REGISTRY: LazyLock<RwLock<HashMap<String, ScalarUdf>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register (or replace) a scalar UDF callable as `name(args)` in SQL.
///
/// `arg_types` declares the function's parameters in order (the analyzer checks each call's argument
/// is assignable to the matching type, and the executor passes the coerced values in this order);
/// `return_type` is the result type. The name is folded to lowercase to match the parser's identifier
/// folding. Registering a name that is already a built-in does not shadow the built-in (the parser
/// resolves built-ins first); choose a distinct name.
pub fn register_scalar_udf(
    name: &str,
    arg_types: Vec<ColumnType>,
    return_type: ColumnType,
    func: ScalarUdfFn,
) {
    if let Ok(mut registry) = REGISTRY.write() {
        registry.insert(
            name.to_ascii_lowercase(),
            ScalarUdf {
                arg_types,
                return_type,
                func,
            },
        );
    }
}

/// Remove a registered scalar UDF, returning whether one was removed.
pub fn unregister_scalar_udf(name: &str) -> bool {
    REGISTRY
        .write()
        .is_ok_and(|mut registry| registry.remove(&name.to_ascii_lowercase()).is_some())
}

/// The `(argument types, return type)` of a registered scalar UDF, for the analyzer's type check.
/// `None` if no UDF is registered under `name` (the analyzer then reports an unknown function).
pub(crate) fn scalar_udf_signature(name: &str) -> Option<(Vec<ColumnType>, ColumnType)> {
    let registry = REGISTRY.read().ok()?;
    let signature = registry
        .get(name)
        .map(|udf| (udf.arg_types.clone(), udf.return_type));
    drop(registry);
    signature
}

/// Invoke a registered scalar UDF with the evaluated arguments (executor).
///
/// # Errors
/// [`Error::UnknownFunction`] if the UDF was unregistered since analysis, or [`Error::UdfFailed`] if
/// the function itself returns an error.
pub(crate) fn call_scalar_udf(name: &str, args: &[Value]) -> Result<Value, Error> {
    // Clone the function handle (a cheap `Arc`) and release the registry lock *before* invoking it,
    // so a UDF never runs while holding the registry read lock (which could otherwise deadlock if a
    // UDF registered another).
    let func = {
        let registry = REGISTRY
            .read()
            .map_err(|_| Error::Unsupported("UDF registry lock was poisoned".to_owned()))?;
        registry
            .get(name)
            .map(|udf| Arc::clone(&udf.func))
            .ok_or_else(|| Error::UnknownFunction(name.to_owned()))?
    };
    func(args).map_err(|message| Error::UdfFailed {
        name: name.to_owned(),
        message,
    })
}
