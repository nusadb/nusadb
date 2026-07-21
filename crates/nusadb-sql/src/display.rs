//! Canonical text rendering of a runtime [`Value`].
//!
//! Used to render array elements (`{a,b,c}`) and shared by the wire + e2e output paths so every
//! value type has one rendering. `NULL` renders as the bare token `NULL` (its array-element form);
//! callers that need a wire-NULL handle that before calling.

use crate::ast::Value;
use crate::temporal;

/// Render a value as its canonical SQL text.
#[must_use]
pub fn value_text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        // JSON renders in the spaced display form (`{"a": 1}`); the stored form stays compact.
        Value::Json(s) => crate::json::display_form(s),
        Value::Date(d) => temporal::format_date(*d),
        Value::Time(t) => temporal::format_time(*t),
        Value::Timestamp(t) => temporal::format_timestamp(*t),
        Value::TimestampTz(t) => temporal::format_timestamptz(*t),
        Value::TimeTz(t) => temporal::format_timetz(*t),
        Value::Uuid(u) => temporal::format_uuid(u),
        Value::Numeric(d) => d.format(),
        Value::Interval(iv) => iv.format(),
        Value::Array(items) => array_text(items),
        Value::Vector(vec) => crate::vector::format(vec),
        Value::Bytes(b) => bytea_hex(b),
    }
}

/// Render a `BYTEA` value in the standard `hex` output form: `\x` followed by lowercase hex digits.
/// The empty byte string renders as `\x`.
#[must_use]
pub fn bytea_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(2 + bytes.len() * 2);
    out.push_str("\\x");
    for byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    out
}

/// Render an array as the standard text form `{e1,e2,...}`. Elements are quoted + escaped
/// when needed so the form round-trips unambiguously (see `push_array_element`).
#[must_use]
pub fn array_text(items: &[Value]) -> String {
    // Rough reserve: braces + a few chars per element + separators.
    let mut out = String::with_capacity(items.len() * 8 + 2);
    out.push('{');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_array_element(&mut out, item);
    }
    out.push('}');
    out
}

/// Render one array element with the standard quoting rules so `{...}` round-trips through the parser
/// without ambiguity: a `NULL` element is the bare token `NULL`; a nested array keeps its bare
/// `{...}` form; every other element is double-quoted — with `"` and `\` backslash-escaped — when its
/// text is empty, spells `NULL` (case-insensitively), or contains a brace, comma, quote, backslash, or
/// whitespace. Unquoted otherwise (e.g. numbers, plain words).
fn push_array_element(out: &mut String, item: &Value) {
    match item {
        Value::Null => out.push_str("NULL"),
        // A nested array nests bare; NusaDB has no multidimensional arrays today, but keep the form
        // correct rather than quoting the inner braces.
        Value::Array(_) => out.push_str(&value_text(item)),
        _ => {
            let text = value_text(item);
            if array_element_needs_quoting(&text) {
                out.push('"');
                for ch in text.chars() {
                    if ch == '"' || ch == '\\' {
                        out.push('\\');
                    }
                    out.push(ch);
                }
                out.push('"');
            } else {
                out.push_str(&text);
            }
        },
    }
}

/// Whether an array element's rendered text must be double-quoted to round-trip: empty, an unquoted
/// `NULL` would be read as the null token, or it carries a structural character (`{} ,` `"` `\`) or
/// whitespace that the parser would otherwise mis-split or trim.
fn array_element_needs_quoting(text: &str) -> bool {
    text.is_empty()
        || text.eq_ignore_ascii_case("null")
        || text
            .chars()
            .any(|c| matches!(c, '{' | '}' | ',' | '"' | '\\') || c.is_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_renders_brace_style() {
        let a = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(value_text(&a), "{1,2,3}");
        let t = Value::Array(vec![
            Value::Text("a".to_owned()),
            Value::Null,
            Value::Text("c".to_owned()),
        ]);
        assert_eq!(value_text(&t), "{a,NULL,c}");
        assert_eq!(array_text(&[]), "{}");
    }
}
