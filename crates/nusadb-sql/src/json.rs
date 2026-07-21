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

/// `json ->> field` — like [`get_field`] but returns the member as **text**: a JSON string yields
/// its raw contents (unquoted); anything else yields its canonical JSON text.
#[must_use]
pub fn get_field_text(json: &str, key: &str) -> Option<String> {
    parse(json)?.get(key).map(scalar_text)
}

/// `json ->> n` — like [`get_index`] but returns the element as text (see [`get_field_text`]).
#[must_use]
pub fn get_index_text(json: &str, index: i64) -> Option<String> {
    let v = parse(json)?;
    let arr = v.as_array()?;
    let idx = resolve_index(index, arr.len())?;
    arr.get(idx).map(scalar_text)
}

/// `a @> b` — does the JSON document `a` contain `b`? `None` if either side is invalid JSON.
#[must_use]
pub fn contains(a: &str, b: &str) -> Option<bool> {
    Some(value_contains(&parse(a)?, &parse(b)?))
}

/// `json ? key` (as the `jsonb_exists` function) — whether `key` is a top-level object key, a string
/// element of a top-level array, or equals a scalar string. Invalid JSON → `false`.
#[must_use]
pub fn has_key(json: &str, key: &str) -> bool {
    match parse(json) {
        Some(J::Object(map)) => map.contains_key(key),
        Some(J::Array(items)) => items.iter().any(|v| v.as_str() == Some(key)),
        Some(J::String(s)) => s == key,
        _ => false,
    }
}

/// `json #> path` — follow `path` (object keys / array indices given as text) and return the value
/// as canonical JSON text. `None` (SQL `NULL`) if any step is missing or `json` is invalid.
#[must_use]
pub fn get_path(json: &str, path: &[&str]) -> Option<String> {
    navigate(parse(json)?, path).as_ref().map(to_text)
}

/// `json #>> path` — like [`get_path`] but returns the final value as text (see [`get_field_text`]).
#[must_use]
pub fn get_path_text(json: &str, path: &[&str]) -> Option<String> {
    navigate(parse(json)?, path).as_ref().map(scalar_text)
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
