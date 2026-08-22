//! JSON / JSONB value support for the `JSON` column type (phase 3).
//!
//! A JSON value is stored as its **canonical text** (see [`ast::Value::Json`](crate::ast::Value::Json)):
//! parsed with `serde_json`, then re-serialized by [`to_text`]. Object keys are emitted in the
//! reference engine's `jsonb` order — shorter keys first, then bytewise — and insignificant
//! whitespace is dropped, giving JSONB semantics where `{"b":2,"aa":1}` and `{ "aa": 1, "b": 2 }`
//! normalize to the same text and so compare equal. Operators (`->`, `->>`, `@>`) parse the
//! canonical text on demand.

use serde_json::Value as J;

/// Parse + canonicalize `s`, or `None` if it is not valid JSON.
///
/// Goes through [`parse`], so numbers reach their canonical decimal form rather than being echoed as
/// written, and through [`to_text`], so object keys land in the reference engine's `jsonb` order.
#[must_use]
pub fn canonicalize(s: &str) -> Option<String> {
    Some(to_text(&parse(s)?))
}

/// Parse JSON text into a [`serde_json::Value`].
///
/// Numbers are normalized to canonical decimal on the way in, so `1e3` reads back as `1000` and a
/// long decimal keeps every digit. `serde_json` is built with `arbitrary_precision`, so a number
/// arrives as its source literal rather than a rounded `f64`; without the normalization it would
/// then *print* as that literal (`1e+3`), which is neither the input nor the canonical form.
#[must_use]
pub fn parse(s: &str) -> Option<J> {
    let mut value: J = serde_json::from_str(s).ok()?;
    normalize_numbers(&mut value);
    Some(value)
}

/// Rewrite every number in `value` to its canonical decimal text (see
/// [`crate::jsonb::normalize_number`]). A literal that does not fit is left exactly as written, so
/// the transformation never loses a digit.
fn normalize_numbers(value: &mut J) {
    match value {
        J::Number(n) => {
            if let Some(canonical) = crate::jsonb::normalize_number(&n.to_string())
                && let Ok(parsed) = serde_json::from_str::<serde_json::Number>(&canonical)
            {
                *n = parsed;
            }
        },
        J::Array(items) => {
            for item in items {
                normalize_numbers(item);
            }
        },
        J::Object(map) => {
            for val in map.values_mut() {
                normalize_numbers(val);
            }
        },
        _ => {},
    }
}

/// Serialize a [`serde_json::Value`] back to canonical (compact) text.
///
/// Object keys are emitted in the reference engine's `jsonb` order — shorter keys first, then
/// bytewise — rather than `serde_json`'s pure-bytewise map order, so a document's canonical form (and
/// therefore its text output, equality and `DISTINCT`) agrees with it. Arrays keep element order and
/// scalars use `serde_json`'s rendering (so number normalization and string escaping are unchanged).
#[must_use]
pub fn to_text(v: &J) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

/// Recursive worker for [`to_text`]: render `v` compactly with objects in `jsonb` key order.
fn write_canonical(v: &J, out: &mut String) {
    match v {
        J::Object(map) => {
            let mut entries: Vec<(&String, &J)> = map.iter().collect();
            entries.sort_by(|a, b| crate::jsonb::key_order(a.0, b.0));
            out.push('{');
            for (i, (key, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&string_literal(key));
                out.push(':');
                write_canonical(val, out);
            }
            out.push('}');
        },
        J::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        },
        scalar => {
            out.push_str(&serde_json::to_string(scalar).unwrap_or_else(|_| "null".to_owned()));
        },
    }
}

/// Render `s` as a JSON string literal — quoted, with the necessary escaping. Used when assembling
/// object text by hand (e.g. `row_to_json`), where a field name may contain characters that need
/// escaping.
#[must_use]
pub fn string_literal(s: &str) -> String {
    to_text(&J::String(s.to_owned()))
}

/// Render canonical (compact) JSON text in the spaced *display* form (`{"a": 1, "b": 2}`).
///
/// A space is inserted after each object-member colon and each comma, matching the standard `jsonb`
/// text output. Only the displayed / cast-to-text form is spaced; the stored canonical form (used for
/// storage and comparison) stays compact. Colons and commas inside string literals are untouched.
#[must_use]
pub fn display_form(canonical: &str) -> String {
    let mut out = String::with_capacity(canonical.len() + canonical.len() / 8 + 1);
    let mut in_string = false;
    let mut escaped = false;
    for c in canonical.chars() {
        out.push(c);
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            ':' | ',' if !in_string => out.push(' '),
            _ => {},
        }
    }
    out
}

/// `json -> field` — fetch an object member by name. `None` (SQL `NULL`) if `json` is not an
/// object or the key is absent. Returns the member as canonical JSON text.
#[must_use]
pub fn get_field(json: &str, key: &str) -> Option<String> {
    parse(json)?.get(key).map(to_text)
}

/// `json -> n` — fetch an array element by index (negative counts from the end).
/// `None` if `json` is not an array or the index is out of range.
#[must_use]
pub fn get_index(json: &str, index: i64) -> Option<String> {
    let v = parse(json)?;
    let arr = v.as_array()?;
    let idx = resolve_index(index, arr.len())?;
    arr.get(idx).map(to_text)
}

/// `json ->> field` — like [`get_field`] but returns the member as **text**.
///
/// A JSON string yields its raw contents (unquoted); anything else yields its canonical JSON text;
/// a JSON `null` member yields SQL `NULL`.
#[must_use]
pub fn get_field_text(json: &str, key: &str) -> Option<String> {
    scalar_text_opt(parse(json)?.get(key)?)
}

/// `json ->> n` — like [`get_index`] but returns the element as text (see [`get_field_text`]).
#[must_use]
pub fn get_index_text(json: &str, index: i64) -> Option<String> {
    let v = parse(json)?;
    let arr = v.as_array()?;
    let idx = resolve_index(index, arr.len())?;
    scalar_text_opt(arr.get(idx)?)
}

/// `a @> b` — does the JSON document `a` contain `b`? `None` if either side is invalid JSON.
#[must_use]
pub fn contains(a: &str, b: &str) -> Option<bool> {
    Some(value_contains(&parse(a)?, &parse(b)?))
}

/// `json ? key` (also the `jsonb_exists` function) — whether `key` is a top-level object key, a
/// string element of a top-level array, or equals a scalar string. Invalid JSON → `false`.
#[must_use]
pub fn has_key(json: &str, key: &str) -> bool {
    parse(json).is_some_and(|v| key_present(&v, key))
}

/// `json ?| keys` — whether **any** of `keys` is present per [`has_key`]. An empty list is `false`:
/// none of nothing is present.
#[must_use]
pub fn has_any_key(json: &str, keys: &[&str]) -> bool {
    parse(json).is_some_and(|v| keys.iter().any(|k| key_present(&v, k)))
}

/// `json ?& keys` — whether **every** key in `keys` is present per [`has_key`]. An empty list is
/// `true` (vacuously), matching the reference engine.
#[must_use]
pub fn has_all_keys(json: &str, keys: &[&str]) -> bool {
    parse(json).is_some_and(|v| keys.iter().all(|k| key_present(&v, k)))
}

/// The shared membership test behind `?` / `?|` / `?&`: an object matches by key name, an array by a
/// *string* element (a numeric element never matches the text `'1'`), and a scalar string by
/// equality. Any other document shape has no keys.
fn key_present(v: &J, key: &str) -> bool {
    match v {
        J::Object(map) => map.contains_key(key),
        J::Array(items) => items.iter().any(|e| e.as_str() == Some(key)),
        J::String(s) => s == key,
        _ => false,
    }
}

/// `a || b` — concatenate two JSON documents, as canonical text.
///
/// Two objects merge shallowly, with `b`'s members winning on a shared key (the merge is *not*
/// recursive: `{"a":{"x":1}} || {"a":{"y":2}}` is `{"a":{"y":2}}`). Two arrays concatenate.
/// Otherwise each non-array operand is treated as a one-element array, so `[1,2] || 3` is `[1,2,3]`,
/// `3 || [1,2]` is `[3,1,2]`, and `1 || 2` is `[1,2]`. `None` if either side is invalid JSON.
#[must_use]
pub fn concat(a: &str, b: &str) -> Option<String> {
    Some(to_text(&concat_values(parse(a)?, parse(b)?)))
}

fn concat_values(a: J, b: J) -> J {
    match (a, b) {
        (J::Object(mut left), J::Object(right)) => {
            left.extend(right);
            J::Object(left)
        },
        (J::Array(mut left), J::Array(right)) => {
            left.extend(right);
            J::Array(left)
        },
        (J::Array(mut left), other) => {
            left.push(other);
            J::Array(left)
        },
        (other, J::Array(right)) => {
            let mut out = Vec::with_capacity(right.len() + 1);
            out.push(other);
            out.extend(right);
            J::Array(out)
        },
        (left, right) => J::Array(vec![left, right]),
    }
}

/// Why a `json - key` / `json - index` delete does not fit the document's shape.
///
/// Both cases are errors in the reference engine rather than a silently unchanged document, so the
/// caller raises them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteRefusal {
    /// The document is a scalar (a number, string, boolean or `null`): it has nothing to delete.
    Scalar,
    /// An integer index was applied to an object, which is keyed by name and not by position.
    ObjectIndex,
}

/// `json - key` / `json - keys` — remove `keys` from a JSON document, as canonical text.
///
/// From an object, each key removes the member of that name; from an array, each key removes **every**
/// string element equal to it. A key that is not present is not an error. A scalar document is
/// [`DeleteRefusal::Scalar`]. `Ok(None)` means `json` is not valid JSON.
///
/// # Errors
/// [`DeleteRefusal::Scalar`] when the document is not an object or array.
pub fn delete_keys(json: &str, keys: &[&str]) -> Result<Option<String>, DeleteRefusal> {
    let Some(v) = parse(json) else {
        return Ok(None);
    };
    let out = match v {
        J::Object(mut map) => {
            for key in keys {
                map.remove(*key);
            }
            J::Object(map)
        },
        J::Array(mut items) => {
            items.retain(|e| !e.as_str().is_some_and(|s| keys.contains(&s)));
            J::Array(items)
        },
        _ => return Err(DeleteRefusal::Scalar),
    };
    Ok(Some(to_text(&out)))
}

/// `json - n` — remove the array element at index `n`, as canonical text.
///
/// A negative index counts from the end; an index outside the array leaves the document unchanged.
/// `Ok(None)` means `json` is not valid JSON.
///
/// # Errors
/// [`DeleteRefusal::ObjectIndex`] for an object document, [`DeleteRefusal::Scalar`] for a scalar
/// one — an object is keyed by name, and a scalar has nothing to delete.
pub fn delete_index(json: &str, index: i64) -> Result<Option<String>, DeleteRefusal> {
    let Some(v) = parse(json) else {
        return Ok(None);
    };
    let J::Array(mut items) = v else {
        return Err(if matches!(v, J::Object(_)) {
            DeleteRefusal::ObjectIndex
        } else {
            DeleteRefusal::Scalar
        });
    };
    if let Some(idx) = resolve_index(index, items.len()) {
        items.remove(idx);
    }
    Ok(Some(to_text(&J::Array(items))))
}

/// Why a `json #- path` delete could not follow the document. Both are raised by the reference
/// engine rather than silently leaving the document unchanged, so the caller reports them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeletePathError {
    /// The whole document is a scalar (number / string / boolean / `null`): a non-empty path has
    /// nothing to follow. (A scalar reached *mid-path* is different — navigation stops there and the
    /// document is left unchanged, not an error.)
    ScalarRoot,
    /// A path element addressing an array did not parse as an integer index. Carries the 1-based
    /// position of the element within the path and its text, to reproduce the reference message.
    NonIntegerArrayIndex {
        /// 1-based position of the offending element within the path.
        position: usize,
        /// The offending element text.
        element: String,
    },
}

/// `json #- path` — remove the element at `path` (object keys / array indices given as text), as
/// canonical text.
///
/// Each step follows the document like [`get_path`]: an object step selects a member by key, an
/// array step by the (possibly negative) integer it parses to. The final step then removes an object
/// member of that name, or an array element at that index. A missing object key or an out-of-range
/// array index leaves the document unchanged, as does an empty path. `Ok(None)` means `json` is not
/// valid JSON (the same "not a document" signal the sibling helpers give).
///
/// # Errors
/// [`DeletePathError::ScalarRoot`] when the whole document is a scalar and the path is non-empty, and
/// [`DeletePathError::NonIntegerArrayIndex`] when a step addressing an array is not an integer — both
/// raised by the reference engine.
pub fn delete_path(json: &str, path: &[&str]) -> Result<Option<String>, DeletePathError> {
    let Some(mut root) = parse(json) else {
        return Ok(None);
    };
    if path.is_empty() {
        return Ok(Some(to_text(&root)));
    }
    // A scalar top-level document has nothing to delete from — an error, not a no-op.
    if !matches!(root, J::Object(_) | J::Array(_)) {
        return Err(DeletePathError::ScalarRoot);
    }
    delete_in(&mut root, path, 1)?;
    Ok(Some(to_text(&root)))
}

/// Recursive worker for [`delete_path`]: remove `path` from `node`, where `position` is the 1-based
/// index of `path[0]` within the original path (used only for the not-an-integer error message).
fn delete_in(node: &mut J, path: &[&str], position: usize) -> Result<(), DeletePathError> {
    let Some((seg, rest)) = path.split_first() else {
        return Ok(());
    };
    let last = rest.is_empty();
    match node {
        J::Object(map) => {
            if last {
                map.remove(*seg);
            } else if let Some(child) = map.get_mut(*seg) {
                delete_in(child, rest, position + 1)?;
            }
            // A missing key mid-path leaves the document unchanged.
        },
        J::Array(arr) => {
            // An array step must be an integer, at any depth — the reference engine errors otherwise
            // (before any range check), so a non-integer is reported even when it would be out of range.
            let Ok(index) = seg.parse::<i64>() else {
                return Err(DeletePathError::NonIntegerArrayIndex {
                    position,
                    element: (*seg).to_owned(),
                });
            };
            if let Some(idx) = resolve_index(index, arr.len()) {
                if last {
                    arr.remove(idx);
                } else if let Some(child) = arr.get_mut(idx) {
                    delete_in(child, rest, position + 1)?;
                }
            }
            // An out-of-range index leaves the document unchanged.
        },
        // A scalar reached mid-path: nothing to descend into, so leave it unchanged.
        _ => {},
    }
    Ok(())
}

/// `json #> path` — follow `path` (object keys / array indices given as text) and return the value
/// as canonical JSON text. `None` (SQL `NULL`) if any step is missing or `json` is invalid.
#[must_use]
pub fn get_path(json: &str, path: &[&str]) -> Option<String> {
    navigate(parse(json)?, path).as_ref().map(to_text)
}

/// `json #>> path` — like [`get_path`] but returns the final value as text (see [`get_field_text`]).
/// A JSON `null` at the end of the path yields SQL `NULL`.
#[must_use]
pub fn get_path_text(json: &str, path: &[&str]) -> Option<String> {
    scalar_text_opt(navigate(parse(json)?, path).as_ref()?)
}

/// `json_array_elements(json)` — the elements of a JSON array, each as canonical JSON text.
/// `None` if `json` is not an array (the caller yields no rows in that case).
#[must_use]
pub fn array_elements(json: &str) -> Option<Vec<String>> {
    let v = parse(json)?;
    let arr = v.as_array()?;
    Some(arr.iter().map(to_text).collect())
}

/// `jsonb_array_elements_text(json)` — each element of a JSON array as SQL text.
///
/// A string element yields its raw contents, a JSON `null` yields SQL `NULL` (the inner `None`), and
/// everything else its JSON form. The outer `None` means `json` is not a JSON array.
#[must_use]
pub fn array_elements_text(json: &str) -> Option<Vec<Option<String>>> {
    let v = parse(json)?;
    let arr = v.as_array()?;
    Some(
        arr.iter()
            .map(|e| match e {
                J::Null => None,
                J::String(s) => Some(s.clone()),
                other => Some(to_text(other)),
            })
            .collect(),
    )
}

/// `jsonb_each(json)` — the members of a JSON object as `(key, value)` pairs.
///
/// The value is canonical JSON text, and keys come in the document's canonical (sorted) order, the
/// same order [`object_keys`] gives. `None` if `json` is not a JSON object (including a valid
/// non-object document): the caller reports that rather than yielding no rows.
#[must_use]
pub fn each(json: &str) -> Option<Vec<(String, String)>> {
    match parse(json)? {
        J::Object(map) => {
            let mut pairs: Vec<(String, String)> =
                map.into_iter().map(|(k, v)| (k, to_text(&v))).collect();
            pairs.sort_by(|a, b| crate::jsonb::key_order(&a.0, &b.0));
            Some(pairs)
        },
        _ => None,
    }
}

/// `jsonb_each_text(json)` — like [`each`] but the value is SQL text: a string member yields its raw
/// contents, a JSON `null` yields SQL `NULL` (the inner `None`), everything else its JSON form.
#[must_use]
pub fn each_text(json: &str) -> Option<Vec<(String, Option<String>)>> {
    match parse(json)? {
        J::Object(map) => {
            let mut pairs: Vec<(String, Option<String>)> = map
                .into_iter()
                .map(|(k, v)| {
                    let text = match v {
                        J::Null => None,
                        J::String(s) => Some(s),
                        other => Some(to_text(&other)),
                    };
                    (k, text)
                })
                .collect();
            pairs.sort_by(|a, b| crate::jsonb::key_order(&a.0, &b.0));
            Some(pairs)
        },
        _ => None,
    }
}

/// `json_typeof(json)` — the JSON type name of `json`: `null`/`boolean`/`number`/`string`/`array`/
/// `object`. `None` if `json` is not valid JSON.
#[must_use]
pub fn type_name(json: &str) -> Option<&'static str> {
    Some(match parse(json)? {
        J::Null => "null",
        J::Bool(_) => "boolean",
        J::Number(_) => "number",
        J::String(_) => "string",
        J::Array(_) => "array",
        J::Object(_) => "object",
    })
}

/// `json_array_length(json)` — the element count if `json` is a JSON array, else `None` (SQL `NULL`).
#[must_use]
pub fn array_length(json: &str) -> Option<i64> {
    match parse(json)? {
        J::Array(a) => i64::try_from(a.len()).ok(),
        _ => None,
    }
}

/// `jsonb_strip_nulls(json)` — recursively remove object members whose value is JSON `null`.
///
/// Returns canonical JSON text, or `None` if `json` is invalid. Null elements inside arrays are kept
/// (only object fields are stripped); an object that becomes empty after stripping is retained.
#[must_use]
pub fn strip_nulls(json: &str) -> Option<String> {
    let mut v = parse(json)?;
    strip_nulls_in_place(&mut v);
    Some(to_text(&v))
}

fn strip_nulls_in_place(v: &mut J) {
    match v {
        J::Object(map) => {
            map.retain(|_, val| !val.is_null());
            for val in map.values_mut() {
                strip_nulls_in_place(val);
            }
        },
        J::Array(arr) => {
            for val in arr {
                strip_nulls_in_place(val);
            }
        },
        _ => {},
    }
}

/// `jsonb_pretty(json)` — the JSON re-serialized with indentation for readability, as `TEXT`; `None`
/// if `json` is invalid.
///
/// Matches the reference engine's `jsonb_pretty`: 4-space indentation per level, a space after each
/// object-member colon, object keys in `jsonb` order, and empty containers spread over two lines
/// (`{\n}` / `[\n]`). A top-level scalar is rendered on one line.
#[must_use]
pub fn pretty(json: &str) -> Option<String> {
    let mut out = String::new();
    write_pretty(&parse(json)?, 0, &mut out);
    Some(out)
}

/// Recursive worker for [`pretty`]: render `v` at nesting `level` (4 spaces each).
fn write_pretty(v: &J, level: usize, out: &mut String) {
    let indent = |n: usize, out: &mut String| out.push_str(&"    ".repeat(n));
    match v {
        J::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &J)> = map.iter().collect();
            entries.sort_by(|a, b| crate::jsonb::key_order(a.0, b.0));
            for (i, (key, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                out.push_str(&string_literal(key));
                out.push_str(": ");
                write_pretty(val, level + 1, out);
            }
            // An empty object still closes on its own line: `{\n}` (with the parent's indent).
            out.push('\n');
            indent(level, out);
            out.push('}');
        },
        J::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                write_pretty(item, level + 1, out);
            }
            out.push('\n');
            indent(level, out);
            out.push(']');
        },
        scalar => out.push_str(&to_text(scalar)),
    }
}

/// `jsonb_object_keys(json)` — the top-level field names of a JSON object, in canonical (sorted)
/// order; `None` if `json` is invalid or not an object.
#[must_use]
pub fn object_keys(json: &str) -> Option<Vec<String>> {
    match parse(json)? {
        J::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort_by(|a, b| crate::jsonb::key_order(a, b));
            Some(keys)
        },
        _ => None,
    }
}

/// Convert a runtime [`crate::ast::Value`] to a JSON value (`to_json` / `to_jsonb`).
///
/// Primitives map directly; a `JSON` value embeds as-is; an `ARRAY` maps element-wise; every other
/// type (temporal, UUID, interval, vector) becomes its canonical text string.
#[must_use]
pub fn value_to_json(v: &crate::ast::Value) -> J {
    use crate::ast::Value as V;
    match v {
        V::Null => J::Null,
        V::Bool(b) => J::Bool(*b),
        V::Int(i) => J::Number((*i).into()),
        V::Float(f) => float_to_json(*f),
        V::Text(s) => J::String(s.clone()),
        V::Json(s) => parse(s).unwrap_or_else(|| J::String(s.clone())),
        V::Numeric(d) => {
            serde_json::from_str(&d.format()).unwrap_or_else(|_| J::String(d.format()))
        },
        // A temporal value uses ISO 8601 in JSON: a `T` between date and time, and `+00:00` (not a
        // bare `+00`) on a zoned timestamp. This is JSON-specific — the general text rendering, used
        // for display and casts, keeps its space separator and `+00` zone.
        V::Timestamp(micros) => {
            J::String(crate::temporal::format_timestamp(*micros).replacen(' ', "T", 1))
        },
        V::TimestampTz(micros) => {
            let iso = crate::temporal::format_timestamp(*micros).replacen(' ', "T", 1);
            J::String(format!("{iso}+00:00"))
        },
        V::Array(items) => J::Array(items.iter().map(value_to_json).collect()),
        other => J::String(crate::display::value_text(other)),
    }
}

/// Convert an `f64` (float8) to its JSON form. A finite value uses the shortest round-trip decimal
/// (so an integral float loses its `.0`, matching the reference: `1.0` renders as `1`); a non-finite
/// one becomes the JSON *string* `"Infinity"` / `"-Infinity"` / `"NaN"`, as the reference does (there
/// is no JSON number for them).
fn float_to_json(f: f64) -> J {
    if f.is_nan() {
        return J::String("NaN".to_owned());
    }
    if f.is_infinite() {
        return J::String(if f < 0.0 { "-Infinity" } else { "Infinity" }.to_owned());
    }
    // `f64`'s `Display` is the shortest round-tripping decimal without an exponent — the same form
    // the reference engine's float-to-json uses — so parsing it as a JSON number is exact.
    serde_json::from_str::<serde_json::Number>(&f.to_string()).map_or(J::Null, J::Number)
}

/// Build a JSON object from `(key, value)` pairs as canonical text (`json_build_object`). A
/// duplicate key keeps the last value, matching the catalog's JSONB semantics.
#[must_use]
pub fn build_object(pairs: Vec<(String, J)>) -> String {
    let map: serde_json::Map<String, J> = pairs.into_iter().collect();
    to_text(&J::Object(map))
}

/// Build a JSON array document from `items` in order (the `json_build_array` constructor).
/// Element order is preserved; an empty input yields `[]`.
#[must_use]
pub fn build_array(items: Vec<J>) -> String {
    to_text(&J::Array(items))
}

/// `jsonb_set(target, path, new_value[, create_missing])`: replace the value at `path`.
///
/// `path` segments are object keys or array indices given as text. Returns the updated document as
/// canonical text, or `None` if `target` is not valid JSON. A missing object key (final or
/// intermediate) is created only when `create_missing`. An array index out of range, or a step into a
/// scalar, leaves the document unchanged.
#[must_use]
pub fn set_path(
    target: &str,
    path: &[String],
    new_value: J,
    create_missing: bool,
) -> Option<String> {
    let mut root = parse(target)?;
    set_in(&mut root, path, new_value, create_missing);
    Some(to_text(&root))
}

/// `jsonb_insert(target, path, new_value [, insert_after])` — insert `new_value` at `path` *without*
/// overwriting.
///
/// At an object the key is added only if absent (an existing key is left untouched); at an array the
/// value is inserted before — or, with `insert_after`, after — the indexed element. Returns canonical
/// JSON, or `None` if `target` is invalid. An out-of-range array index leaves the document unchanged.
#[must_use]
pub fn insert_path(
    target: &str,
    path: &[String],
    new_value: J,
    insert_after: bool,
) -> Option<String> {
    let mut root = parse(target)?;
    insert_in(&mut root, path, new_value, insert_after);
    Some(to_text(&root))
}

/// Recursive worker for [`insert_path`].
fn insert_in(node: &mut J, path: &[String], new_value: J, insert_after: bool) {
    let Some((seg, rest)) = path.split_first() else {
        return;
    };
    let last = rest.is_empty();
    match node {
        J::Object(map) => {
            if last {
                // Add the key only if it is absent — never overwrite (this is the jsonb_set contrast).
                if !map.contains_key(seg) {
                    map.insert(seg.clone(), new_value);
                }
            } else if let Some(child) = map.get_mut(seg) {
                insert_in(child, rest, new_value, insert_after);
            }
        },
        J::Array(arr) => {
            if let Some(idx) = array_index(seg, arr.len()) {
                if last {
                    // `idx < len`, so `idx + 1 <= len` is a valid insertion point (len = push).
                    let at = if insert_after { idx + 1 } else { idx };
                    arr.insert(at, new_value);
                } else if let Some(child) = arr.get_mut(idx) {
                    insert_in(child, rest, new_value, insert_after);
                }
            }
        },
        _ => {},
    }
}

/// Recursive worker for [`set_path`]: mutate `node` at `path`.
fn set_in(node: &mut J, path: &[String], new_value: J, create_missing: bool) {
    let Some((seg, rest)) = path.split_first() else {
        return;
    };
    let last = rest.is_empty();
    match node {
        J::Object(map) => {
            if last {
                if create_missing || map.contains_key(seg) {
                    map.insert(seg.clone(), new_value);
                }
            } else if let Some(child) = map.get_mut(seg) {
                set_in(child, rest, new_value, create_missing);
            } else if create_missing {
                let mut child = J::Object(serde_json::Map::new());
                set_in(&mut child, rest, new_value, create_missing);
                map.insert(seg.clone(), child);
            }
        },
        J::Array(arr) => {
            if let Some(idx) = array_index(seg, arr.len()) {
                if last {
                    if let Some(slot) = arr.get_mut(idx) {
                        *slot = new_value;
                    }
                } else if let Some(child) = arr.get_mut(idx) {
                    set_in(child, rest, new_value, create_missing);
                }
            }
        },
        _ => {},
    }
}

/// Resolve a possibly-negative array-index path segment against a known length, or `None` if it is
/// not a numeric index in range.
fn array_index(seg: &str, len: usize) -> Option<usize> {
    let i: i64 = seg.trim().parse().ok()?;
    let resolved = if i < 0 {
        i64::try_from(len).ok()? + i
    } else {
        i
    };
    usize::try_from(resolved).ok().filter(|&u| u < len)
}

/// One step of the supported `jsonpath` subset.
enum PathStep {
    /// `.key` or `['key']`/`["key"]` — an object member.
    Key(String),
    /// `[n]` — an array element by (possibly negative) index.
    Index(i64),
    /// `[*]` (array elements) or `.*` (object values) — every child (the set-returning step).
    Wildcard,
    /// `.**` — recursive descent: the value itself and every descendant, in document (pre-order).
    RecursiveDescent,
    /// `? (predicate)` — keep only the current values for which `predicate` holds.
    Filter(Box<FilterExpr>),
}

/// A comparison operator inside a `jsonpath` filter predicate.
#[derive(Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// The root a filter accessor path starts from: `@` (the item under test) or `$` (the document).
#[derive(Clone, Copy)]
enum FilterRoot {
    Current,
    Root,
}

/// One side of a filter comparison: either an accessor path (relative to `@` or `$`) or a literal.
enum FilterOperand {
    Path {
        root: FilterRoot,
        steps: Vec<PathStep>,
    },
    Literal(J),
}

/// A `jsonpath` filter predicate — the expression inside `? (...)`, and (later) the whole argument to
/// a predicate-check (`@@`). Evaluated with three-valued logic (true / false / unknown); a `?` filter
/// keeps an item only when the predicate is definitely true.
enum FilterExpr {
    Or(Box<Self>, Box<Self>),
    And(Box<Self>, Box<Self>),
    Not(Box<Self>),
    /// `exists(path)` — true iff the path yields at least one value.
    Exists {
        root: FilterRoot,
        steps: Vec<PathStep>,
    },
    Cmp {
        left: FilterOperand,
        op: CmpOp,
        right: FilterOperand,
    },
}

/// Consume a `? (...)` filter body from `chars` (the `?` already consumed) and parse it: skip
/// whitespace, require the opening `(`, capture up to the matching `)` (tracking nesting and skipping
/// parentheses inside string literals), then parse the captured text as a predicate. `None` if the
/// parentheses are unbalanced or the body is not a valid predicate.
fn capture_filter(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<FilterExpr> {
    while matches!(chars.peek(), Some(' ' | '\t' | '\n' | '\r')) {
        chars.next();
    }
    if chars.next() != Some('(') {
        return None;
    }
    let mut inner = String::new();
    let mut depth = 1u32;
    let mut quote: Option<char> = None;
    for ch in chars.by_ref() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else {
            match ch {
                '"' | '\'' => quote = Some(ch),
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                },
                _ => {},
            }
        }
        inner.push(ch);
    }
    if depth != 0 {
        return None;
    }
    parse_filter(&inner)
}

/// Parse the supported `jsonpath` subset: `$` root, `.key`, `.*`, `.**` (recursive descent),
/// `['key']`/`["key"]`, `[n]`, `[*]`, and `? (predicate)` filters. Returns `None` for any syntax
/// outside the subset (e.g. the level-bounded `.**{n}` form).
fn parse_jsonpath(path: &str) -> Option<Vec<PathStep>> {
    let mut chars = path.trim().chars().peekable();
    if chars.next()? != '$' {
        return None;
    }
    let mut steps = Vec::new();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            },
            '?' => {
                chars.next();
                steps.push(PathStep::Filter(Box::new(capture_filter(&mut chars)?)));
            },
            '.' => {
                chars.next();
                if chars.peek() == Some(&'*') {
                    chars.next();
                    // `.**` is recursive descent; `.*` is a single-level wildcard.
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        // The level-bounded form `.**{n}` is not supported (reject it loudly rather
                        // than treat the `{...}` as a key).
                        if chars.peek() == Some(&'{') {
                            return None;
                        }
                        steps.push(PathStep::RecursiveDescent);
                    } else {
                        steps.push(PathStep::Wildcard);
                    }
                    continue;
                }
                let mut key = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '.' || c == '[' {
                        break;
                    }
                    key.push(c);
                    chars.next();
                }
                if key.is_empty() {
                    return None;
                }
                steps.push(PathStep::Key(key));
            },
            '[' => {
                chars.next();
                let mut inner = String::new();
                while let Some(&c) = chars.peek() {
                    if c == ']' {
                        break;
                    }
                    inner.push(c);
                    chars.next();
                }
                if chars.next() != Some(']') {
                    return None;
                }
                let inner = inner.trim();
                if inner == "*" {
                    steps.push(PathStep::Wildcard);
                } else if let Ok(n) = inner.parse::<i64>() {
                    steps.push(PathStep::Index(n));
                } else if (inner.starts_with('\'') && inner.ends_with('\'') && inner.len() >= 2)
                    || (inner.starts_with('"') && inner.ends_with('"') && inner.len() >= 2)
                {
                    steps.push(PathStep::Key(inner[1..inner.len() - 1].to_owned()));
                } else {
                    return None;
                }
            },
            _ => return None,
        }
    }
    Some(steps)
}

/// Recursive-descent parser for the filter predicate inside `? (...)`.
///
/// Grammar (loosest to tightest binding): `||`, then `&&`, then a leading `!`, then a primary — a
/// parenthesized group, an `exists(path)`, or a comparison `operand <cmp> operand`. An operand is an
/// accessor path (rooted at `@` or `$`) or a literal (number / double- or single-quoted string /
/// `true` / `false` / `null`). Returns `None` for anything outside this grammar.
struct FilterParser {
    chars: Vec<char>,
    pos: usize,
}

impl FilterParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }

    /// Consume the literal token `tok` if it appears next (after whitespace); else leave `pos` put.
    fn eat(&mut self, tok: &str) -> bool {
        self.skip_ws();
        let toks: Vec<char> = tok.chars().collect();
        if self
            .chars
            .get(self.pos..)
            .is_some_and(|rest| rest.starts_with(&toks))
        {
            self.pos += toks.len();
            true
        } else {
            false
        }
    }

    fn parse_or(&mut self) -> Option<FilterExpr> {
        let mut left = self.parse_and()?;
        while self.eat("||") {
            let right = self.parse_and()?;
            left = FilterExpr::Or(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    fn parse_and(&mut self) -> Option<FilterExpr> {
        let mut left = self.parse_not()?;
        while self.eat("&&") {
            let right = self.parse_not()?;
            left = FilterExpr::And(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    fn parse_not(&mut self) -> Option<FilterExpr> {
        self.skip_ws();
        // A leading `!` is negation only when it is not the `!=` comparison operator.
        if self.peek() == Some('!') && self.chars.get(self.pos + 1) != Some(&'=') {
            self.pos += 1;
            return Some(FilterExpr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Option<FilterExpr> {
        self.skip_ws();
        if self.eat("(") {
            let inner = self.parse_or()?;
            if !self.eat(")") {
                return None;
            }
            return Some(inner);
        }
        if self.eat("exists") {
            if !self.eat("(") {
                return None;
            }
            let (root, steps) = self.parse_path()?;
            if !self.eat(")") {
                return None;
            }
            return Some(FilterExpr::Exists { root, steps });
        }
        let left = self.parse_operand()?;
        let op = self.parse_cmp_op()?;
        let right = self.parse_operand()?;
        Some(FilterExpr::Cmp { left, op, right })
    }

    fn parse_cmp_op(&mut self) -> Option<CmpOp> {
        self.skip_ws();
        for (tok, op) in [
            ("==", CmpOp::Eq),
            ("!=", CmpOp::Ne),
            ("<>", CmpOp::Ne),
            ("<=", CmpOp::Le),
            (">=", CmpOp::Ge),
            ("<", CmpOp::Lt),
            (">", CmpOp::Gt),
        ] {
            if self.eat(tok) {
                return Some(op);
            }
        }
        None
    }

    fn parse_operand(&mut self) -> Option<FilterOperand> {
        self.skip_ws();
        match self.peek() {
            Some('@' | '$') => {
                let (root, steps) = self.parse_path()?;
                Some(FilterOperand::Path { root, steps })
            },
            _ => Some(FilterOperand::Literal(self.parse_literal()?)),
        }
    }

    /// Parse `@`/`$` followed by `.key`, `.*`, `.**`, `[n]`, `[*]`, or `['key']` accessors.
    fn parse_path(&mut self) -> Option<(FilterRoot, Vec<PathStep>)> {
        self.skip_ws();
        let root = match self.peek() {
            Some('@') => FilterRoot::Current,
            Some('$') => FilterRoot::Root,
            _ => return None,
        };
        self.pos += 1;
        let mut steps = Vec::new();
        while let Some(c) = self.peek() {
            match c {
                '.' => {
                    self.pos += 1;
                    if self.peek() == Some('*') {
                        self.pos += 1;
                        // `.**` is recursive descent; `.*` is a single-level wildcard.
                        if self.peek() == Some('*') {
                            self.pos += 1;
                            if self.peek() == Some('{') {
                                return None;
                            }
                            steps.push(PathStep::RecursiveDescent);
                        } else {
                            steps.push(PathStep::Wildcard);
                        }
                        continue;
                    }
                    let mut key = String::new();
                    while let Some(c) = self.peek() {
                        if matches!(c, '.' | '[' | ' ' | '\t' | '\n' | '\r')
                            || matches!(c, '=' | '!' | '<' | '>' | '&' | '|' | ')')
                        {
                            break;
                        }
                        key.push(c);
                        self.pos += 1;
                    }
                    if key.is_empty() {
                        return None;
                    }
                    steps.push(PathStep::Key(key));
                },
                '[' => {
                    self.pos += 1;
                    let mut inner = String::new();
                    while let Some(c) = self.peek() {
                        if c == ']' {
                            break;
                        }
                        inner.push(c);
                        self.pos += 1;
                    }
                    if self.peek() != Some(']') {
                        return None;
                    }
                    self.pos += 1;
                    let inner = inner.trim();
                    if inner == "*" {
                        steps.push(PathStep::Wildcard);
                    } else if let Ok(n) = inner.parse::<i64>() {
                        steps.push(PathStep::Index(n));
                    } else if (inner.starts_with('\'') && inner.ends_with('\'') && inner.len() >= 2)
                        || (inner.starts_with('"') && inner.ends_with('"') && inner.len() >= 2)
                    {
                        steps.push(PathStep::Key(inner[1..inner.len() - 1].to_owned()));
                    } else {
                        return None;
                    }
                },
                _ => break,
            }
        }
        Some((root, steps))
    }

    fn parse_literal(&mut self) -> Option<J> {
        self.skip_ws();
        if let Some(quote @ ('"' | '\'')) = self.peek() {
            self.pos += 1;
            let mut s = String::new();
            while let Some(c) = self.peek() {
                self.pos += 1;
                if c == '\\' {
                    if let Some(esc) = self.peek() {
                        self.pos += 1;
                        s.push(esc);
                    }
                } else if c == quote {
                    return Some(J::String(s));
                } else {
                    s.push(c);
                }
            }
            return None;
        }
        if self.eat("true") {
            return Some(J::Bool(true));
        }
        if self.eat("false") {
            return Some(J::Bool(false));
        }
        if self.eat("null") {
            return Some(J::Null);
        }
        let mut num = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                num.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        parse(&num)
    }
}

/// Parse a `jsonpath` filter predicate, or `None` if it falls outside the accepted grammar.
fn parse_filter(inner: &str) -> Option<FilterExpr> {
    let mut parser = FilterParser {
        chars: inner.chars().collect(),
        pos: 0,
    };
    let expr = parser.parse_or()?;
    parser.skip_ws();
    if parser.pos == parser.chars.len() {
        Some(expr)
    } else {
        None
    }
}

/// Why a [`path_query`] did not run. The two are different mistakes by different arguments, and a
/// caller that reports them as one thing names the wrong culprit half the time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathQueryError {
    /// The first argument is not JSON. Nothing to do with the path.
    MalformedDocument,
    /// The second argument is not a `jsonpath` this engine accepts — either malformed, or outside
    /// the supported subset. Both are the caller's to fix, and neither is a reason to skip the
    /// statement, so they share one variant.
    UnusablePath,
}

/// `jsonb_path_query(json, path)` — every value in `json` matching the `jsonpath` `path`.
///
/// Each match is canonical JSON text. A `[*]`/`.*` step fans out over children, so the result may
/// hold many matches (or none); a valid path matching nothing yields an empty `Vec`. A failure says
/// which argument was at fault — see [`PathQueryError`].
///
/// # Errors
///
/// [`PathQueryError::MalformedDocument`] if `json` does not parse, or
/// [`PathQueryError::UnusablePath`] if `path` is not a usable `jsonpath`.
pub fn path_query(json: &str, path: &str) -> Result<Vec<String>, PathQueryError> {
    let doc = parse(json).ok_or(PathQueryError::MalformedDocument)?;
    let steps = parse_jsonpath(path).ok_or(PathQueryError::UnusablePath)?;
    let matches = apply_steps(vec![doc.clone()], &steps, &doc);
    Ok(matches.iter().map(to_text).collect())
}

/// `jsonb_path_match(json, path)` / the `@@` operator — evaluate `path` as a predicate check against
/// `json` and return its boolean value.
///
/// `Ok(Some(bool))` is the predicate outcome. `Ok(None)` is SQL NULL, returned when the predicate is
/// unknown (a comparison against an incompatible type), or when `path` is a bare accessor expression
/// whose result is not a single boolean value. (A comparison against a path that resolves to nothing
/// is a definite `false`, not unknown.)
///
/// # Errors
///
/// [`PathQueryError::MalformedDocument`] if `json` does not parse, or
/// [`PathQueryError::UnusablePath`] if `path` is neither a usable predicate nor a usable accessor
/// path.
pub fn path_match(json: &str, path: &str) -> Result<Option<bool>, PathQueryError> {
    let doc = parse(json).ok_or(PathQueryError::MalformedDocument)?;
    let trimmed = path.trim();
    // A predicate-check expression (comparison / logical / exists) evaluates to a boolean directly.
    if let Some(expr) = parse_filter(trimmed) {
        return Ok(eval_filter(&expr, &doc, &doc));
    }
    // Otherwise the argument must be a plain accessor path, which is a boolean check only when it
    // resolves to exactly one boolean value; anything else is not a single boolean, hence NULL.
    let steps = parse_jsonpath(trimmed).ok_or(PathQueryError::UnusablePath)?;
    let values = apply_steps(vec![doc.clone()], &steps, &doc);
    Ok(match values.as_slice() {
        [J::Bool(b)] => Some(*b),
        _ => None,
    })
}

/// Walk `steps` over the `current` value set, returning the set of matched values. `root` is the
/// whole document, needed to resolve `$`-rooted operands inside a filter predicate.
fn apply_steps(mut current: Vec<J>, steps: &[PathStep], root: &J) -> Vec<J> {
    for step in steps {
        let mut next = Vec::new();
        for value in current {
            match step {
                PathStep::Key(key) => {
                    if let J::Object(mut map) = value
                        && let Some(child) = map.remove(key)
                    {
                        next.push(child);
                    }
                },
                PathStep::Index(index) => {
                    if let J::Array(arr) = value
                        && let Some(idx) = resolve_index(*index, arr.len())
                        && let Some(elem) = arr.into_iter().nth(idx)
                    {
                        next.push(elem);
                    }
                },
                PathStep::Wildcard => match value {
                    J::Array(arr) => next.extend(arr),
                    // Object values are emitted in `jsonb` key order (length-first, then bytewise),
                    // not the raw map's bytewise order, so `$.*` matches the reference engine even
                    // when the members have keys of differing length.
                    J::Object(map) => {
                        let mut entries: Vec<(String, J)> = map.into_iter().collect();
                        entries.sort_by(|a, b| crate::jsonb::key_order(&a.0, &b.0));
                        next.extend(entries.into_iter().map(|(_, value)| value));
                    },
                    _ => {},
                },
                PathStep::RecursiveDescent => collect_descendants(value, &mut next),
                PathStep::Filter(pred) => {
                    if eval_filter(pred, &value, root) == Some(true) {
                        next.push(value);
                    }
                },
            }
        }
        current = next;
    }
    current
}

/// Append `value` and every descendant to `out` in pre-order: the value itself, then each child
/// recursively. Array elements keep their order; object members are visited in `jsonb` key order, so
/// a bare `$.**` yields values in the same order as the reference engine.
fn collect_descendants(value: J, out: &mut Vec<J>) {
    out.push(value.clone());
    match value {
        J::Array(arr) => {
            for elem in arr {
                collect_descendants(elem, out);
            }
        },
        J::Object(map) => {
            let mut entries: Vec<(String, J)> = map.into_iter().collect();
            entries.sort_by(|a, b| crate::jsonb::key_order(&a.0, &b.0));
            for (_, child) in entries {
                collect_descendants(child, out);
            }
        },
        _ => {},
    }
}

/// Evaluate a filter predicate with three-valued logic: `Some(true)` / `Some(false)` / `None`
/// (unknown — e.g. a comparison against an incompatible type). `current` is the item under test (the
/// `@` in the predicate); `root` is the whole document (the `$`).
fn eval_filter(expr: &FilterExpr, current: &J, root: &J) -> Option<bool> {
    match expr {
        FilterExpr::Or(a, b) => {
            three_or(eval_filter(a, current, root), eval_filter(b, current, root))
        },
        FilterExpr::And(a, b) => {
            three_and(eval_filter(a, current, root), eval_filter(b, current, root))
        },
        FilterExpr::Not(a) => eval_filter(a, current, root).map(|v| !v),
        FilterExpr::Exists { root: r, steps } => {
            let start = filter_start(*r, current, root);
            Some(!apply_steps(vec![start], steps, root).is_empty())
        },
        FilterExpr::Cmp { left, op, right } => {
            let ls = operand_values(left, current, root);
            let rs = operand_values(right, current, root);
            existential_compare(&ls, *op, &rs)
        },
    }
}

/// The value a `@`/`$`-rooted filter path starts walking from.
fn filter_start(root: FilterRoot, current: &J, doc: &J) -> J {
    match root {
        FilterRoot::Current => current.clone(),
        FilterRoot::Root => doc.clone(),
    }
}

/// Resolve a filter operand to the set of JSON values it denotes.
fn operand_values(operand: &FilterOperand, current: &J, root: &J) -> Vec<J> {
    match operand {
        FilterOperand::Path { root: r, steps } => {
            apply_steps(vec![filter_start(*r, current, root)], steps, root)
        },
        FilterOperand::Literal(value) => vec![value.clone()],
    }
}

/// Three-valued AND: true only if both true, false if either is false, else unknown.
const fn three_and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Three-valued OR: true if either is true, false only if both false, else unknown.
const fn three_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Existential comparison over the cross product of both operand sets, with three-valued logic:
///
/// - **true** if any pair satisfies `op` (true wins outright);
/// - else **unknown** if any pair was incomparable — a type mismatch or a container operand raises an
///   error that the predicate suppresses to unknown, and unknown outranks a definite false;
/// - else **false** — every pair was comparable and none matched, *including* the empty case (a path
///   that resolved to nothing has no matching pair, so the comparison is definitely false).
fn existential_compare(left: &[J], op: CmpOp, right: &[J]) -> Option<bool> {
    let mut saw_error = false;
    for l in left {
        for r in right {
            match compare_pair(l, op, r) {
                Some(true) => return Some(true),
                Some(false) => {},
                None => saw_error = true,
            }
        }
    }
    if saw_error { None } else { Some(false) }
}

/// Compare one scalar pair under `op`. `None` when the pair is not comparable under that operator
/// (a container operand, or type-mismatched operands for an ordered comparison).
fn compare_pair(left: &J, op: CmpOp, right: &J) -> Option<bool> {
    match op {
        CmpOp::Eq => scalar_eq(left, right),
        CmpOp::Ne => scalar_eq(left, right).map(|equal| !equal),
        CmpOp::Lt => scalar_ord(left, right).map(|o| o == std::cmp::Ordering::Less),
        CmpOp::Le => scalar_ord(left, right).map(|o| o != std::cmp::Ordering::Greater),
        CmpOp::Gt => scalar_ord(left, right).map(|o| o == std::cmp::Ordering::Greater),
        CmpOp::Ge => scalar_ord(left, right).map(|o| o != std::cmp::Ordering::Less),
    }
}

/// Equality of two JSON scalars of a matching type: `Some(true/false)` only when both are numbers,
/// both strings, both booleans, or both null. Any other pairing — a container operand, or two scalars
/// of different types — is not comparable in a predicate and yields `None` (unknown), matching the
/// reference engine, where a cross-type comparison is a suppressed error rather than a plain false.
fn scalar_eq(left: &J, right: &J) -> Option<bool> {
    match (left, right) {
        (J::Number(a), J::Number(b)) => Some(match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x == y,
            _ => a.to_string() == b.to_string(),
        }),
        (J::String(a), J::String(b)) => Some(a == b),
        (J::Bool(a), J::Bool(b)) => Some(a == b),
        (J::Null, J::Null) => Some(true),
        _ => None,
    }
}

/// Ordering of two JSON scalars of the same comparable type: `None` unless both are numbers, both
/// strings, or both booleans.
fn scalar_ord(left: &J, right: &J) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (J::Number(a), J::Number(b)) => a.as_f64()?.partial_cmp(&b.as_f64()?),
        (J::String(a), J::String(b)) => Some(a.cmp(b)),
        (J::Bool(a), J::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Follow a `#>`/`#>>` path through `value`: each step indexes an object by key, or an array by the
/// (possibly negative) integer the step parses to. `None` if a step does not resolve.
fn navigate(value: J, path: &[&str]) -> Option<J> {
    let mut cur = value;
    for step in path {
        cur = match cur {
            J::Object(mut map) => map.remove(*step)?,
            J::Array(mut arr) => {
                let idx = resolve_index(step.parse::<i64>().ok()?, arr.len())?;
                arr.swap_remove(idx)
            },
            _ => return None,
        };
    }
    Some(cur)
}

/// Render a JSON value as SQL text: a string yields its raw contents, everything else its JSON form.
fn scalar_text(v: &J) -> String {
    match v {
        J::String(s) => s.clone(),
        other => to_text(other),
    }
}

/// Like [`scalar_text`], but a JSON `null` becomes SQL `NULL` (`None`) rather than the text
/// `"null"`. The `->>` / `#>>` text accessors return SQL NULL for a JSON null, so an `IS NULL`
/// test on an extracted-text value behaves as expected.
fn scalar_text_opt(v: &J) -> Option<String> {
    match v {
        J::Null => None,
        other => Some(scalar_text(other)),
    }
}

/// Resolve a possibly-negative index against `len`; `None` if out of range.
fn resolve_index(index: i64, len: usize) -> Option<usize> {
    if index >= 0 {
        usize::try_from(index).ok().filter(|&i| i < len)
    } else {
        // -1 is the last element. `unsigned_abs` avoids the overflow that `-index` hits at
        // `i64::MIN` (negation would panic in debug / wrap in release).
        let from_end = usize::try_from(index.unsigned_abs()).ok()?;
        len.checked_sub(from_end)
    }
}

/// Whether every JSON object in `text` has unique keys, checked **recursively** at every nesting
/// depth (the `WITH UNIQUE KEYS` check of the `IS JSON` predicate).
///
/// `Some(true)` when no object anywhere has a duplicate key, `Some(false)` when at least one does,
/// and `None` when `text` is not valid JSON. A non-object (array/scalar) is trivially unique.
///
/// `serde_json`'s object model is a map that COLLAPSES duplicate keys on the way in, so
/// [`parse`]/[`canonicalize`] cannot see them — this walks the **raw** text instead, tracking each
/// object's keys as it goes. It is intentionally permissive on scalar syntax (numbers, literals,
/// string escapes): validity is established separately by [`parse`], and this scanner only needs to
/// locate object boundaries and their keys.
#[must_use]
pub(crate) fn has_unique_keys_recursive(text: &str) -> Option<bool> {
    let mut scanner = RawScanner {
        bytes: text.as_bytes(),
        pos: 0,
    };
    scanner.skip_ws();
    let unique = scanner.scan_value()?;
    scanner.skip_ws();
    // Trailing non-whitespace means the text is not a single JSON value.
    if scanner.pos != scanner.bytes.len() {
        return None;
    }
    Some(unique)
}

/// A minimal cursor over raw JSON bytes for [`has_unique_keys_recursive`]. Structural characters
/// (`{}[]":,`) are ASCII, and UTF-8 continuation bytes never collide with them, so byte scanning is
/// safe for multi-byte string contents.
struct RawScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl RawScanner<'_> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Scan one JSON value, returning whether it (and every nested object) has unique keys, or
    /// `None` on any structural parse error.
    fn scan_value(&mut self) -> Option<bool> {
        self.skip_ws();
        match self.bytes.get(self.pos)? {
            b'{' => self.scan_object(),
            b'[' => self.scan_array(),
            b'"' => self.scan_string().map(|()| true),
            // Scalars (numbers, `true`/`false`/`null`) carry no keys; consume the run of characters
            // that can form one. Fine-grained validity is `parse`'s job, not this scanner's.
            _ => {
                let start = self.pos;
                while let Some(&b) = self.bytes.get(self.pos) {
                    if matches!(b, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                        break;
                    }
                    self.pos += 1;
                }
                (self.pos > start).then_some(true)
            },
        }
    }

    /// Scan an object (cursor on the opening `{`), returning whether it and its children have unique
    /// keys. A duplicate key anywhere makes the whole subtree non-unique, but scanning continues so
    /// the full document is still validated structurally.
    fn scan_object(&mut self) -> Option<bool> {
        self.pos += 1; // consume `{`
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Some(true);
        }
        let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut unique = true;
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return None;
            }
            let key = self.take_string()?;
            if !keys.insert(key) {
                unique = false;
            }
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return None;
            }
            self.pos += 1; // consume `:`
            unique &= self.scan_value()?;
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(&b',') => {
                    self.pos += 1;
                },
                Some(&b'}') => {
                    self.pos += 1;
                    return Some(unique);
                },
                _ => return None,
            }
        }
    }

    /// Scan an array (cursor on the opening `[`), returning whether every element subtree is unique.
    fn scan_array(&mut self) -> Option<bool> {
        self.pos += 1; // consume `[`
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Some(true);
        }
        let mut unique = true;
        loop {
            unique &= self.scan_value()?;
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(&b',') => {
                    self.pos += 1;
                },
                Some(&b']') => {
                    self.pos += 1;
                    return Some(unique);
                },
                _ => return None,
            }
        }
    }

    /// Consume a string (cursor on the opening `"`), discarding the value.
    fn scan_string(&mut self) -> Option<()> {
        self.take_string().map(|_| ())
    }

    /// Consume a string (cursor on the opening `"`) and return its decoded-enough key text. Only the
    /// two escapes that matter for delimiting — `\"` and `\\` — are unescaped; other escapes are kept
    /// verbatim, which is enough to compare object keys for equality (any two source keys that are
    /// byte-identical stay equal, and distinct ones stay distinct).
    fn take_string(&mut self) -> Option<String> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return None;
        }
        self.pos += 1; // consume opening `"`
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b'\\' => {
                    // Skip the escape introducer and the escaped byte (so a `\"` does not end the
                    // string and a `\\` is consumed as a pair).
                    self.pos += 2;
                },
                b'"' => {
                    let raw = self.bytes.get(start..self.pos)?;
                    self.pos += 1; // consume closing `"`
                    return Some(String::from_utf8_lossy(raw).into_owned());
                },
                _ => self.pos += 1,
            }
        }
        None // unterminated string
    }
}

/// Recursive JSONB containment (`@>`): objects must match key-by-key, arrays must have every
/// right element contained in some left element, scalars must be equal.
fn value_contains(a: &J, b: &J) -> bool {
    match (a, b) {
        (J::Object(am), J::Object(bm)) => bm
            .iter()
            .all(|(k, bv)| am.get(k).is_some_and(|av| value_contains(av, bv))),
        (J::Array(aa), J::Array(ba)) => ba
            .iter()
            .all(|bv| aa.iter().any(|av| value_contains(av, bv))),
        // An array contains a non-array scalar when some element matches it (containment semantics).
        (J::Array(aa), scalar) => aa.iter().any(|av| value_contains(av, scalar)),
        _ => a == b,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "unit-test assertions unwrap known-good inputs"
)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_sorts_keys_and_strips_whitespace() {
        assert_eq!(
            canonicalize(r#"{ "b": 2, "a": 1 }"#).unwrap(),
            r#"{"a":1,"b":2}"#
        );
        assert!(canonicalize("not json").is_none());
        assert_eq!(canonicalize("[1, 2,3]").unwrap(), "[1,2,3]");
    }

    #[test]
    fn display_form_spaces_after_colons_and_commas_outside_strings() {
        assert_eq!(display_form(r#"{"a":1,"b":2}"#), r#"{"a": 1, "b": 2}"#);
        assert_eq!(display_form("[1,2,3]"), "[1, 2, 3]");
        assert_eq!(
            display_form(r#"{"a":{"b":7},"c":[1,2]}"#),
            r#"{"a": {"b": 7}, "c": [1, 2]}"#
        );
        // A colon/comma inside a string value is left untouched.
        assert_eq!(display_form(r#"{"k":"a,b:c"}"#), r#"{"k": "a,b:c"}"#);
        // An escaped quote does not end the string early.
        assert_eq!(display_form(r#"{"k":"a\"b,c"}"#), r#"{"k": "a\"b,c"}"#);
    }

    #[test]
    fn arrow_get_field_and_index() {
        let j = r#"{"a":{"x":1},"b":[10,20,30]}"#;
        assert_eq!(get_field(j, "a").unwrap(), r#"{"x":1}"#);
        assert_eq!(get_field(j, "missing"), None);
        let arr = get_field(j, "b").unwrap();
        assert_eq!(get_index(&arr, 1).unwrap(), "20");
        assert_eq!(get_index(&arr, -1).unwrap(), "30");
        assert_eq!(get_index(&arr, 5), None);
    }

    #[test]
    fn arrow_text_unquotes_strings() {
        let j = r#"{"name":"alice","age":30}"#;
        assert_eq!(get_field_text(j, "name").unwrap(), "alice"); // unquoted
        assert_eq!(get_field_text(j, "age").unwrap(), "30");
    }

    #[test]
    fn containment() {
        assert_eq!(contains(r#"{"a":1,"b":2}"#, r#"{"a":1}"#), Some(true));
        assert_eq!(contains(r#"{"a":1}"#, r#"{"a":2}"#), Some(false));
        assert_eq!(contains("[1,2,3]", "[3,1]"), Some(true));
        assert_eq!(contains("[1,2,3]", "2"), Some(true));
        assert_eq!(contains("[1,2,3]", "[4]"), Some(false));
    }

    #[test]
    fn array_elements_yields_each_element() {
        assert_eq!(
            array_elements("[1,2,3]").unwrap(),
            vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]
        );
        assert_eq!(array_elements("[]").unwrap(), Vec::<String>::new());
        // Objects/scalars are not arrays → None (no rows).
        assert!(array_elements(r#"{"a":1}"#).is_none());
        assert!(array_elements("42").is_none());
    }

    #[test]
    fn path_query_supports_the_jsonpath_subset() {
        let doc = r#"{"items":[{"n":1},{"n":2},{"n":3}],"meta":{"k":"v"}}"#;
        // Root.
        assert_eq!(path_query(doc, "$").unwrap().len(), 1);
        // Object member chain.
        assert_eq!(
            path_query(doc, "$.meta.k").unwrap(),
            vec![r#""v""#.to_owned()]
        );
        // Array index.
        assert_eq!(
            path_query(doc, "$.items[1]").unwrap(),
            vec![r#"{"n":2}"#.to_owned()]
        );
        // Wildcard fans out, then a member step maps each.
        assert_eq!(
            path_query(doc, "$.items[*].n").unwrap(),
            vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]
        );
        // Quoted key.
        assert_eq!(
            path_query(doc, "$['meta']['k']").unwrap(),
            vec![r#""v""#.to_owned()]
        );
        // `.*` fans out over object values (two members here).
        assert_eq!(path_query(doc, "$.*").unwrap().len(), 2);
        // `.*` emits object values in jsonb key order (length-first, then bytewise) — here `b`, `z`,
        // `aa` — not the raw bytewise `aa`, `b`, `z`.
        assert_eq!(
            path_query(r#"{"aa":1,"z":2,"b":3}"#, "$.*").unwrap(),
            vec!["3".to_owned(), "2".to_owned(), "1".to_owned()]
        );
        // Valid path, no match: an empty result, not a failure.
        assert_eq!(path_query(doc, "$.nope").unwrap(), Vec::<String>::new());
        // Unsupported syntax and a malformed path are one outcome: the path is unusable.
        assert_eq!(path_query(doc, "items"), Err(PathQueryError::UnusablePath));
        assert_eq!(path_query(doc, "$..n"), Err(PathQueryError::UnusablePath));
        // A malformed *document* is a different mistake from a malformed *path*, and the caller
        // reports them in different classes, so the distinction is pinned here.
        assert_eq!(
            path_query("not json", "$"),
            Err(PathQueryError::MalformedDocument)
        );
    }

    #[test]
    fn path_query_filters_match_the_reference_engine() {
        // Numeric comparisons keep the qualifying elements.
        assert_eq!(
            path_query("[1,2,3,4]", "$[*] ? (@ > 2)").unwrap(),
            vec!["3".to_owned(), "4".to_owned()]
        );
        assert_eq!(
            path_query("[1,2,3]", "$[*] ? (@ != 2)").unwrap(),
            vec!["1".to_owned(), "3".to_owned()]
        );
        assert_eq!(
            path_query("[1,2,3]", "$[*] ? (@ <= 2)").unwrap(),
            vec!["1".to_owned(), "2".to_owned()]
        );
        // String equality against a double-quoted literal.
        assert_eq!(
            path_query(r#"[{"s":"x"},{"s":"y"}]"#, r#"$[*] ? (@.s == "x")"#).unwrap(),
            vec![r#"{"s":"x"}"#.to_owned()]
        );
        // Boolean OR / NOT / AND, and a nested-key accessor.
        assert_eq!(
            path_query("[1,2,3,4]", "$[*] ? (@ < 2 || @ > 3)").unwrap(),
            vec!["1".to_owned(), "4".to_owned()]
        );
        assert_eq!(
            path_query("[1,2,3]", "$[*] ? (!(@ == 2))").unwrap(),
            vec!["1".to_owned(), "3".to_owned()]
        );
        assert_eq!(
            path_query(r#"[{"a":{"b":5}}]"#, "$[*] ? (@.a.b == 5)").unwrap(),
            vec![r#"{"a":{"b":5}}"#.to_owned()]
        );
        assert_eq!(
            path_query(
                r#"[{"n":5,"ok":true},{"n":1,"ok":true}]"#,
                "$[*] ? (@.n > 2 && @.ok == true)"
            )
            .unwrap(),
            vec![r#"{"n":5,"ok":true}"#.to_owned()]
        );
        // A filter then a further step, and a filter that keeps nothing.
        assert_eq!(
            path_query(r#"{"a":[1,2,3]}"#, "$.a[*] ? (@ > 2)").unwrap(),
            vec!["3".to_owned()]
        );
        assert_eq!(
            path_query("[1,2,3]", "$[*] ? (@ > 9)").unwrap(),
            Vec::<String>::new()
        );
        // A malformed filter is an unusable path, not a document error.
        assert_eq!(
            path_query("[1,2,3]", "$[*] ? (@ >)"),
            Err(PathQueryError::UnusablePath)
        );
    }

    #[test]
    fn path_query_recursive_descent_matches_the_reference_engine() {
        // `.**` yields the value and every descendant in pre-order, then a further step maps each.
        assert_eq!(
            path_query(r#"{"a":{"x":1},"b":{"x":2}}"#, "$.**.x").unwrap(),
            vec!["1".to_owned(), "2".to_owned()]
        );
        assert_eq!(
            path_query(r#"{"a":{"b":{"c":9}}}"#, "$.**.c").unwrap(),
            vec!["9".to_owned()]
        );
        // A bare `$.**` over nested arrays: self first, then each element depth-first.
        assert_eq!(
            path_query("[1,[2,[3]]]", "$.**").unwrap(),
            vec![
                "[1,[2,[3]]]".to_owned(),
                "1".to_owned(),
                "[2,[3]]".to_owned(),
                "2".to_owned(),
                "[3]".to_owned(),
                "3".to_owned(),
            ]
        );
        // Recursive descent as a filter accessor, keeping the matching descendant.
        assert_eq!(
            path_query(r#"{"a":{"x":5},"b":{"x":9}}"#, "$.** ? (@.x == 5)").unwrap(),
            vec![r#"{"x":5}"#.to_owned()]
        );
        // Recursive descent inside an @@ predicate path.
        assert_eq!(
            path_match(r#"{"a":{"x":5}}"#, "$.**.x == 5"),
            Ok(Some(true))
        );
        // The level-bounded `.**{n}` form is not supported and is a loud unusable-path error.
        assert_eq!(
            path_query(r#"{"a":1}"#, "$.**{1}"),
            Err(PathQueryError::UnusablePath)
        );
    }

    #[test]
    fn path_match_predicate_checks_match_the_reference_engine() {
        // A comparison predicate evaluates to its boolean result.
        assert_eq!(path_match(r#"{"a":1}"#, "$.a == 1"), Ok(Some(true)));
        assert_eq!(path_match(r#"{"a":5}"#, "$.a > 3"), Ok(Some(true)));
        // exists(...) and an existential comparison over a wildcard.
        assert_eq!(path_match(r#"{"a":1}"#, "exists($.a)"), Ok(Some(true)));
        assert_eq!(path_match(r#"{"a":[1,2,3]}"#, "$.a[*] > 2"), Ok(Some(true)));
        // A missing path makes the comparison definitely false (not unknown), so NOT of it is true.
        assert_eq!(path_match(r#"{"a":1}"#, "$.x == 1"), Ok(Some(false)));
        assert_eq!(path_match(r#"{"a":1}"#, "!($.x == 1)"), Ok(Some(true)));
        // A type mismatch is a suppressed error → unknown (NULL), not false.
        assert_eq!(path_match(r#"{"a":1}"#, r#"$.a == "s""#), Ok(None));
        assert_eq!(path_match(r#"{"a":1}"#, r#"$.a < "s""#), Ok(None));
        // Existential precedence: a true wins; otherwise an error (unknown) outranks a false.
        assert_eq!(path_match(r#"[1,"s"]"#, "$[*] == 1"), Ok(Some(true)));
        assert_eq!(path_match(r#"[2,"s"]"#, "$[*] == 1"), Ok(None));
        assert_eq!(path_match("[2,3]", "$[*] == 1"), Ok(Some(false)));
        // Three-valued AND / OR with a missing operand.
        assert_eq!(
            path_match(r#"{"a":1}"#, "$.x == 1 && $.a == 1"),
            Ok(Some(false))
        );
        assert_eq!(
            path_match(r#"{"a":1}"#, "$.x == 1 || $.a == 1"),
            Ok(Some(true))
        );
        // A bare accessor path is a boolean check only when it is a single boolean value.
        assert_eq!(path_match(r#"{"a":true}"#, "$.a"), Ok(Some(true)));
        assert_eq!(path_match(r#"{"a":1}"#, "$.a"), Ok(None));
        assert_eq!(path_match("[true,false]", "$[*]"), Ok(None));
        // A malformed document is a document error, not a path error.
        assert_eq!(
            path_match("not json", "$.a == 1"),
            Err(PathQueryError::MalformedDocument)
        );
    }

    #[test]
    fn concat_merges_objects_and_joins_everything_else_as_arrays() {
        // Two objects merge shallowly; the right operand wins a shared key, and an object-valued
        // key is replaced rather than merged into.
        assert_eq!(
            concat(r#"{"a":1,"b":2}"#, r#"{"b":3,"c":4}"#).unwrap(),
            r#"{"a":1,"b":3,"c":4}"#
        );
        assert_eq!(
            concat(r#"{"a":{"x":1}}"#, r#"{"a":{"y":2}}"#).unwrap(),
            r#"{"a":{"y":2}}"#
        );
        // Two arrays concatenate; a non-array operand joins as one element, on either side.
        assert_eq!(concat("[1,2]", "[3,4]").unwrap(), "[1,2,3,4]");
        assert_eq!(concat("[1,2]", "3").unwrap(), "[1,2,3]");
        assert_eq!(concat("3", "[1,2]").unwrap(), "[3,1,2]");
        assert_eq!(concat("[1,2]", "null").unwrap(), "[1,2,null]");
        // Neither side an array (including an object beside a scalar): the pair becomes an array.
        assert_eq!(concat("1", "2").unwrap(), "[1,2]");
        assert_eq!(concat("null", "null").unwrap(), "[null,null]");
        assert_eq!(concat(r#"{"a":1}"#, r#""x""#).unwrap(), r#"[{"a":1},"x"]"#);
        // An object beside an array is wrapped, not merged.
        assert_eq!(concat(r#"{"a":1}"#, "[1,2]").unwrap(), r#"[{"a":1},1,2]"#);
        assert_eq!(concat(r#"{"a":1}"#, "[]").unwrap(), r#"[{"a":1}]"#);
        assert_eq!(concat("[]", r#"{"a":1}"#).unwrap(), r#"[{"a":1}]"#);
        // Empty operands of the same shape stay that shape.
        assert_eq!(concat("[]", "[]").unwrap(), "[]");
        assert_eq!(concat("{}", "{}").unwrap(), "{}");
        // Invalid JSON on either side → None.
        assert!(concat("oops", "[]").is_none());
        assert!(concat("[]", "oops").is_none());
    }

    #[test]
    fn delete_removes_keys_elements_and_indices() {
        // Object: by key name; an absent key is a no-op, not an error.
        assert_eq!(
            delete_keys(r#"{"a":1,"b":2}"#, &["a"]).unwrap().unwrap(),
            r#"{"b":2}"#
        );
        assert_eq!(
            delete_keys(r#"{"a":1,"b":2}"#, &["zz"]).unwrap().unwrap(),
            r#"{"a":1,"b":2}"#
        );
        // Several keys at once.
        assert_eq!(
            delete_keys(r#"{"a":1,"b":2,"c":3}"#, &["a", "c"])
                .unwrap()
                .unwrap(),
            r#"{"b":2}"#
        );
        // Array: a key removes *every* equal string element; non-string elements never match.
        assert_eq!(
            delete_keys(r#"["a","b","a"]"#, &["a"]).unwrap().unwrap(),
            r#"["b"]"#
        );
        assert_eq!(delete_keys("[1,2,3]", &["1"]).unwrap().unwrap(), "[1,2,3]");
        // Index: positive, negative (from the end), and out of range (a no-op).
        assert_eq!(delete_index("[1,2,3]", 1).unwrap().unwrap(), "[1,3]");
        assert_eq!(delete_index("[1,2,3]", -1).unwrap().unwrap(), "[1,2]");
        assert_eq!(delete_index("[1,2,3]", 9).unwrap().unwrap(), "[1,2,3]");
        assert_eq!(
            delete_index("[1,2,3]", i64::MIN).unwrap().unwrap(),
            "[1,2,3]"
        );
        // Shapes with nothing to delete are refused, not silently unchanged.
        assert_eq!(delete_keys("1", &["a"]), Err(DeleteRefusal::Scalar));
        assert_eq!(delete_keys(r#""s""#, &["a"]), Err(DeleteRefusal::Scalar));
        assert_eq!(delete_index("1", 0), Err(DeleteRefusal::Scalar));
        assert_eq!(
            delete_index(r#"{"a":1}"#, 0),
            Err(DeleteRefusal::ObjectIndex)
        );
        // Invalid JSON → Ok(None), the same "not a document" signal the other helpers give.
        assert_eq!(delete_keys("oops", &["a"]), Ok(None));
        assert_eq!(delete_index("oops", 0), Ok(None));
    }

    #[test]
    fn delete_path_removes_the_element_at_a_path() {
        let del = |j: &str, p: &[&str]| delete_path(j, p).unwrap().unwrap();
        // Object key, top-level and nested.
        assert_eq!(del(r#"{"a":1,"b":2}"#, &["a"]), r#"{"b":2}"#);
        assert_eq!(
            del(r#"{"a":{"b":1,"c":2}}"#, &["a", "b"]),
            r#"{"a":{"c":2}}"#
        );
        // Array index: positive, negative (from the end).
        assert_eq!(del("[10,20,30]", &["1"]), "[10,30]");
        assert_eq!(del("[10,20,30]", &["-1"]), "[10,20]");
        // Array nested inside an object.
        assert_eq!(del(r#"{"a":[1,2,3]}"#, &["a", "0"]), r#"{"a":[2,3]}"#);
        // Deep mix of object and array steps.
        assert_eq!(
            del(r#"{"a":{"b":{"c":[1,2,3]}}}"#, &["a", "b", "c", "1"]),
            r#"{"a":{"b":{"c":[1,3]}}}"#
        );
        // A missing object key leaves the document unchanged (not an error).
        assert_eq!(del(r#"{"a":1}"#, &["x"]), r#"{"a":1}"#);
        assert_eq!(del(r#"{"a":{"b":1}}"#, &["a", "x"]), r#"{"a":{"b":1}}"#);
        // An out-of-range array index (positive or negative) leaves it unchanged.
        assert_eq!(del("[10,20,30]", &["5"]), "[10,20,30]");
        assert_eq!(del("[10,20,30]", &["-5"]), "[10,20,30]");
        // A single-element path down to the last array element.
        assert_eq!(del("[100]", &["0"]), "[]");
        // An empty path leaves the document unchanged (still canonicalized).
        assert_eq!(del(r#"{ "a": 1 }"#, &[]), r#"{"a":1}"#);
        // An object key is matched as text even when it looks numeric (object, not array, parent).
        assert_eq!(del(r#"{"1":10,"2":20}"#, &["1"]), r#"{"2":20}"#);
        // Navigating *into* a scalar mid-path stops there, leaving the document unchanged.
        assert_eq!(del(r#"{"a":5}"#, &["a", "b"]), r#"{"a":5}"#);
        assert_eq!(
            del(r#"{"a":{"b":5}}"#, &["a", "b", "c"]),
            r#"{"a":{"b":5}}"#
        );
    }

    #[test]
    fn delete_path_reports_scalar_root_and_non_integer_index() {
        // A scalar top-level document with a non-empty path is an error, not a no-op.
        assert_eq!(delete_path("5", &["a"]), Err(DeletePathError::ScalarRoot));
        assert_eq!(
            delete_path(r#""hi""#, &["a"]),
            Err(DeletePathError::ScalarRoot)
        );
        assert_eq!(
            delete_path("null", &["a"]),
            Err(DeletePathError::ScalarRoot)
        );
        // But an *empty* path over a scalar is fine — nothing to follow — and returns it unchanged.
        assert_eq!(delete_path("5", &[]).unwrap().unwrap(), "5");
        // A non-integer element addressing an array is an error, carrying its 1-based position.
        assert_eq!(
            delete_path("[1,2,3]", &["a"]),
            Err(DeletePathError::NonIntegerArrayIndex {
                position: 1,
                element: "a".to_owned(),
            })
        );
        assert_eq!(
            delete_path(r#"{"a":[1,2,3]}"#, &["a", "x", "y"]),
            Err(DeletePathError::NonIntegerArrayIndex {
                position: 2,
                element: "x".to_owned(),
            })
        );
        // The integer check fires before any range check (a non-integer that would be out of range
        // still errors rather than being a silent no-op).
        assert_eq!(
            delete_path("[1,2,3]", &["1.5"]),
            Err(DeletePathError::NonIntegerArrayIndex {
                position: 1,
                element: "1.5".to_owned(),
            })
        );
        // Invalid JSON → Ok(None), the "not a document" signal the sibling helpers give.
        assert_eq!(delete_path("oops", &["a"]), Ok(None));
    }

    #[test]
    fn key_existence_covers_objects_arrays_and_scalar_strings() {
        let obj = r#"{"a":1,"b":2}"#;
        assert!(has_key(obj, "a"));
        assert!(!has_key(obj, "z"));
        // A JSON null value still counts as a present key.
        assert!(has_key(r#"{"a":null}"#, "a"));
        // Arrays match string elements only — a numeric element is not the text key `1`.
        assert!(has_key(r#"["a","b"]"#, "a"));
        assert!(!has_key("[1,2]", "1"));
        // A scalar string matches itself; other scalars have no keys.
        assert!(has_key(r#""a""#, "a"));
        assert!(!has_key("1", "1"));
        assert!(!has_key("oops", "a"));
        // Any / all, with the empty-list identities.
        assert!(has_any_key(obj, &["z", "b"]));
        assert!(!has_any_key(obj, &["z"]));
        assert!(!has_any_key(obj, &[]));
        assert!(has_all_keys(obj, &["a", "b"]));
        assert!(!has_all_keys(obj, &["a", "z"]));
        assert!(has_all_keys(obj, &[]));
    }

    #[test]
    fn numbers_keep_every_digit_and_normalize_the_exponent_form() {
        // The exponent form expands, and a scale written in the source is kept — both the canonical
        // decimal the reference engine prints. Neither goes through `f64`.
        assert_eq!(canonicalize(r#"{"b":1e3}"#).unwrap(), r#"{"b":1000}"#);
        assert_eq!(canonicalize(r#"{"b":1E3}"#).unwrap(), r#"{"b":1000}"#);
        assert_eq!(canonicalize(r#"{"b":1.5e-2}"#).unwrap(), r#"{"b":0.015}"#);
        assert_eq!(canonicalize(r#"{"a":1.0}"#).unwrap(), r#"{"a":1.0}"#);
        assert_eq!(canonicalize(r#"{"c":1}"#).unwrap(), r#"{"c":1}"#);
        assert_eq!(canonicalize(r#"{"n":-2.50}"#).unwrap(), r#"{"n":-2.50}"#);
        // A decimal well past `f64`'s 17 significant digits survives intact — this is the silent
        // truncation the old `f64` round trip caused.
        let long = "0.12345678901234567890123456789";
        assert_eq!(
            canonicalize(&format!(r#"{{"d":{long}}}"#)).unwrap(),
            format!(r#"{{"d":{long}}}"#)
        );
        // Nested positions are normalized too, not just top-level members.
        assert_eq!(
            canonicalize(r#"{"a":[1e2,{"b":2e1}]}"#).unwrap(),
            r#"{"a":[100,{"b":20}]}"#
        );
    }

    #[test]
    fn each_walks_object_members_in_canonical_order() {
        // Members come back in canonical (sorted) key order, values as canonical JSON text.
        assert_eq!(
            each(r#"{"b":2,"a":1}"#).unwrap(),
            vec![
                ("a".to_owned(), "1".to_owned()),
                ("b".to_owned(), "2".to_owned())
            ]
        );
        assert_eq!(each("{}").unwrap(), Vec::<(String, String)>::new());
        // The text form unwraps a string member and turns a JSON null into SQL NULL; anything else
        // keeps its JSON rendering.
        assert_eq!(
            each_text(r#"{"a":"x","b":1,"c":null,"d":[1,2],"e":{"k":1}}"#).unwrap(),
            vec![
                ("a".to_owned(), Some("x".to_owned())),
                ("b".to_owned(), Some("1".to_owned())),
                ("c".to_owned(), None),
                ("d".to_owned(), Some("[1,2]".to_owned())),
                ("e".to_owned(), Some(r#"{"k":1}"#.to_owned())),
            ]
        );
        // A document that is not an object has no members — the caller reports that, so `None`
        // rather than an empty list.
        assert!(each("[1,2]").is_none());
        assert!(each("1").is_none());
        assert!(each("oops").is_none());
        assert!(each_text("[1,2]").is_none());
    }

    #[test]
    fn has_unique_keys_recursive_detects_duplicates_at_every_depth() {
        // No duplicates anywhere → unique.
        assert_eq!(has_unique_keys_recursive(r#"{"a":1,"b":2}"#), Some(true));
        // A top-level duplicate key.
        assert_eq!(has_unique_keys_recursive(r#"{"a":1,"a":2}"#), Some(false));
        // A duplicate NESTED inside a value object — the check is recursive.
        assert_eq!(
            has_unique_keys_recursive(r#"{"a":{"b":1,"b":2}}"#),
            Some(false)
        );
        // A duplicate inside an object that is an array element.
        assert_eq!(
            has_unique_keys_recursive(r#"[{"a":1},{"c":1,"c":2}]"#),
            Some(false)
        );
        // A non-object (array / scalar) is trivially unique.
        assert_eq!(has_unique_keys_recursive("[1,2]"), Some(true));
        assert_eq!(has_unique_keys_recursive("5"), Some(true));
        assert_eq!(has_unique_keys_recursive(r#""s""#), Some(true));
        // Duplicate keys collapse under serde parsing, so a canonical form cannot be used here —
        // but the raw scanner still sees them.
        assert!(canonicalize(r#"{"a":1,"a":2}"#).is_some());
        // Keys that merely share a prefix are distinct.
        assert_eq!(has_unique_keys_recursive(r#"{"a":1,"ab":2}"#), Some(true));
        // A key containing an escaped quote does not end the key early or false-match.
        assert_eq!(
            has_unique_keys_recursive(r#"{"a\"b":1,"a\"b":2}"#),
            Some(false)
        );
        // Invalid JSON → None (the "not a document" signal).
        assert_eq!(has_unique_keys_recursive("not json"), None);
        assert_eq!(has_unique_keys_recursive(r#"{"a":1,}"#), None);
        assert_eq!(has_unique_keys_recursive("{"), None);
    }

    #[test]
    fn objects_render_in_reference_jsonb_key_order() {
        // Keys sort by (length, then bytes): the single-char keys first (bytewise b<c), then the
        // two-char (aa<bb), then the three-char — captured from the reference engine's jsonb.
        assert_eq!(
            canonicalize(r#"{"b":1,"aa":2,"zzz":3,"c":4,"bb":5}"#).unwrap(),
            r#"{"b":1,"c":4,"aa":2,"bb":5,"zzz":3}"#
        );
        // The doc's example: `b` (length 1) precedes `aa` (length 2), which pure bytewise reverses.
        assert_eq!(
            canonicalize(r#"{"b":1,"aa":2}"#).unwrap(),
            r#"{"b":1,"aa":2}"#
        );
        // The order reaches every object render, nested ones included.
        assert_eq!(
            canonicalize(r#"{"outer":{"bb":1,"a":2}}"#).unwrap(),
            r#"{"outer":{"a":2,"bb":1}}"#
        );
        // `object_keys` and `each` report the same order.
        assert_eq!(
            object_keys(r#"{"b":1,"aa":2,"zzz":3}"#).unwrap(),
            vec!["b".to_owned(), "aa".to_owned(), "zzz".to_owned()]
        );
        assert_eq!(
            each(r#"{"aa":1,"b":2}"#).unwrap(),
            vec![
                ("b".to_owned(), "2".to_owned()),
                ("aa".to_owned(), "1".to_owned())
            ]
        );
    }

    #[test]
    fn value_to_json_temporal_uses_iso_8601() {
        use crate::ast::Value as V;
        let ts = crate::temporal::parse_timestamp("2020-01-02 03:04:05").unwrap();
        // A timestamp gets a `T` separator and no zone; a timestamptz gets `T` and `+00:00`.
        assert_eq!(
            to_text(&value_to_json(&V::Timestamp(ts))),
            r#""2020-01-02T03:04:05""#
        );
        assert_eq!(
            to_text(&value_to_json(&V::TimestampTz(ts))),
            r#""2020-01-02T03:04:05+00:00""#
        );
        // Date and time keep their plain ISO forms.
        let d = crate::temporal::parse_date("2020-01-02").unwrap();
        assert_eq!(to_text(&value_to_json(&V::Date(d))), r#""2020-01-02""#);
    }

    #[test]
    fn value_to_json_float_matches_reference() {
        use crate::ast::Value as V;
        let j = |f: f64| to_text(&value_to_json(&V::Float(f)));
        // An integral float loses its `.0`; other finite floats use the shortest decimal.
        assert_eq!(j(1.0), "1");
        assert_eq!(j(100.0), "100");
        assert_eq!(j(1.5), "1.5");
        assert_eq!(j(0.1), "0.1");
        assert_eq!(j(-2.5), "-2.5");
        // Large / small magnitudes expand without an exponent, like the reference.
        assert_eq!(j(1e20), "100000000000000000000");
        assert_eq!(j(1e-7), "0.0000001");
        // Non-finite floats become JSON strings.
        assert_eq!(j(f64::INFINITY), r#""Infinity""#);
        assert_eq!(j(f64::NEG_INFINITY), r#""-Infinity""#);
        assert_eq!(j(f64::NAN), r#""NaN""#);
    }

    #[test]
    fn pretty_matches_reference_layout() {
        // 4-space indent, space after colon, jsonb key order, two-line empty containers.
        assert_eq!(
            pretty(r#"{"b":1,"aa":{"x":[1,2,3],"y":{}},"z":[]}"#).unwrap(),
            "{\n    \"b\": 1,\n    \"z\": [\n    ],\n    \"aa\": {\n        \"x\": [\n            1,\n            2,\n            3\n        ],\n        \"y\": {\n        }\n    }\n}"
        );
        // Empty containers spread over two lines.
        assert_eq!(pretty("{}").unwrap(), "{\n}");
        assert_eq!(pretty("[]").unwrap(), "[\n]");
        // A top-level scalar stays on one line.
        assert_eq!(pretty("42").unwrap(), "42");
        assert_eq!(pretty(r#""hi""#).unwrap(), r#""hi""#);
        assert_eq!(pretty("null").unwrap(), "null");
    }

    #[test]
    fn get_path_navigates_objects_and_arrays() {
        let doc = r#"{"a":{"b":42},"arr":[10,20,30]}"#;
        // Object path.
        assert_eq!(get_path(doc, &["a", "b"]).unwrap(), "42");
        // Array index (as text), then text form.
        assert_eq!(get_path(doc, &["arr", "1"]).unwrap(), "20");
        // Missing key -> None.
        assert!(get_path(doc, &["a", "z"]).is_none());
        // String leaf via #>> yields the raw (unquoted) text.
        let doc2 = r#"{"a":{"b":"hi"}}"#;
        assert_eq!(get_path_text(doc2, &["a", "b"]).unwrap(), "hi");
        // Negative array index counts from the end.
        assert_eq!(get_path(doc, &["arr", "-1"]).unwrap(), "30");
    }
}
