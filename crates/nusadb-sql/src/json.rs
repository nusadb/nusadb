//! JSON / JSONB value support for the `JSON` column type (phase 3).
//!
//! A JSON value is stored as its **canonical text** (see [`ast::Value::Json`](crate::ast::Value::Json)):
//! parsed with `serde_json`, then re-serialized. `serde_json`'s default map is a `BTreeMap`, so
//! object keys are emitted in sorted order and insignificant whitespace is dropped — giving JSONB
//! semantics where `{"b":2,"a":1}` and `{ "a": 1, "b": 2 }` normalize to the same text and so
//! compare equal. Operators (`->`, `->>`, `@>`) parse the canonical text on demand.

use serde_json::Value as J;

/// Parse + canonicalize `s`, or `None` if it is not valid JSON.
#[must_use]
pub fn canonicalize(s: &str) -> Option<String> {
    let v: J = serde_json::from_str(s).ok()?;
    serde_json::to_string(&v).ok()
}

/// Parse JSON text into a [`serde_json::Value`].
#[must_use]
pub fn parse(s: &str) -> Option<J> {
    serde_json::from_str(s).ok()
}

/// Serialize a [`serde_json::Value`] back to canonical text.
#[must_use]
pub fn to_text(v: &J) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "null".to_owned())
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
        J::Object(map) => Some(map.into_iter().map(|(k, v)| (k, to_text(&v))).collect()),
        _ => None,
    }
}

/// `jsonb_each_text(json)` — like [`each`] but the value is SQL text: a string member yields its raw
/// contents, a JSON `null` yields SQL `NULL` (the inner `None`), everything else its JSON form.
#[must_use]
pub fn each_text(json: &str) -> Option<Vec<(String, Option<String>)>> {
    match parse(json)? {
        J::Object(map) => Some(
            map.into_iter()
                .map(|(k, v)| {
                    let text = match v {
                        J::Null => None,
                        J::String(s) => Some(s),
                        other => Some(to_text(&other)),
                    };
                    (k, text)
                })
                .collect(),
        ),
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
#[must_use]
pub fn pretty(json: &str) -> Option<String> {
    let v = parse(json)?;
    serde_json::to_string_pretty(&v).ok()
}

/// `jsonb_object_keys(json)` — the top-level field names of a JSON object, in canonical (sorted)
/// order; `None` if `json` is invalid or not an object.
#[must_use]
pub fn object_keys(json: &str) -> Option<Vec<String>> {
    match parse(json)? {
        J::Object(map) => Some(map.keys().cloned().collect()),
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
        V::Float(f) => serde_json::Number::from_f64(*f).map_or(J::Null, J::Number),
        V::Text(s) => J::String(s.clone()),
        V::Json(s) => parse(s).unwrap_or_else(|| J::String(s.clone())),
        V::Numeric(d) => {
            serde_json::from_str(&d.format()).unwrap_or_else(|_| J::String(d.format()))
        },
        V::Array(items) => J::Array(items.iter().map(value_to_json).collect()),
        other => J::String(crate::display::value_text(other)),
    }
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
}

/// Parse the supported `jsonpath` subset: `$` root, `.key`, `.*`, `['key']`/`["key"]`, `[n]`, `[*]`.
/// Returns `None` for any syntax outside the subset (filters, `..` descent, etc.).
fn parse_jsonpath(path: &str) -> Option<Vec<PathStep>> {
    let mut chars = path.trim().chars().peekable();
    if chars.next()? != '$' {
        return None;
    }
    let mut steps = Vec::new();
    while let Some(&c) = chars.peek() {
        match c {
            '.' => {
                chars.next();
                if chars.peek() == Some(&'*') {
                    chars.next();
                    steps.push(PathStep::Wildcard);
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

/// `jsonb_path_query(json, path)` — every value in `json` matching the `jsonpath` `path`.
///
/// Each match is canonical JSON text. A `[*]`/`.*` step fans out over children, so the result may
/// hold many matches (or none). `None` if `json` is invalid or `path` is outside the supported
/// subset (the caller turns that into an error); a valid path matching nothing yields `Some(empty)`.
#[must_use]
pub fn path_query(json: &str, path: &str) -> Option<Vec<String>> {
    let doc = parse(json)?;
    let steps = parse_jsonpath(path)?;
    let mut current = vec![doc];
    for step in &steps {
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
                    J::Object(map) => next.extend(map.into_values()),
                    _ => {},
                },
            }
        }
        current = next;
    }
    Some(current.iter().map(to_text).collect())
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
        // Valid path, no match → empty (not None).
        assert_eq!(path_query(doc, "$.nope").unwrap(), Vec::<String>::new());
        // Unsupported syntax / bad path → None.
        assert!(path_query(doc, "items").is_none());
        assert!(path_query(doc, "$..n").is_none());
        // Invalid JSON → None.
        assert!(path_query("not json", "$").is_none());
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
