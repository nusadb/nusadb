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
        Value::Float(f) => format_float(*f),
        // Full-text values render as their stored canonical text, like plain text.
        Value::Text(s) | Value::Tsvector(s) | Value::Tsquery(s) | Value::Xml(s) => s.clone(),
        // An enum renders as its label; the ordinal is an ordering detail, never shown.
        Value::Enum { label, .. } => label.clone(),
        // JSON renders in the spaced display form (`{"a": 1}`); the stored form stays compact.
        Value::Json(s) => crate::json::display_form(s),
        Value::Date(d) => temporal::format_date(*d),
        Value::Time(t) => temporal::format_time(*t),
        Value::Timestamp(t) => temporal::format_timestamp(*t),
        // An instant renders on the session-local wall clock with the session zone's offset
        // (`+00` under the default UTC).
        Value::TimestampTz(t) => {
            temporal::format_timestamptz_at(*t, crate::executor::statement_tz_offset_secs())
        },
        Value::TimeTz(t) => temporal::format_timetz(*t),
        Value::Uuid(u) => temporal::format_uuid(u),
        Value::Macaddr(m) => crate::macaddr::format(*m),
        Value::Geometry(g) => crate::geometry::format(g),
        Value::Macaddr8(m) => crate::macaddr8::format(*m),
        Value::Inet(a) => a.format(),
        Value::Bit(b) => crate::bit::format(b),
        Value::Range(r) => r.format(),
        Value::Numeric(d) => d.format(),
        Value::Interval(iv) => iv.format(),
        Value::Array(items) => array_text(items),
        Value::Vector(vec) => crate::vector::format(vec),
        Value::Bytes(b) => bytea_hex(b),
    }
}

/// Render an `f64` as `DOUBLE PRECISION` text, matching the reference engine.
///
/// Shortest round-trip digits (via Rust's own shortest-form `{:e}`), shown in fixed notation when the
/// leading digit's decimal exponent is in `-4..15` and in exponent notation (`1.5e+20`, `9.9e-05`)
/// outside it. Trailing zeros are dropped, the exponent is signed and at least two digits, and the
/// non-finite values render `Infinity` / `-Infinity` / `NaN` (`-0` keeps its sign).
#[must_use]
pub fn format_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_owned();
    }
    if f.is_infinite() {
        return if f < 0.0 { "-Infinity" } else { "Infinity" }.to_owned();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let neg = f < 0.0;
    // `{:e}` yields the shortest round-trip mantissa (`d` or `d.ddd`, no trailing zeros) and the
    // decimal exponent `x` of the leading digit.
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let x: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|&c| c != '.').collect();
    let body = if !(-4..15).contains(&x) {
        // Exponent form: leading digit, the rest as a fraction, then a signed ≥2-digit exponent.
        let mut s = digits[..1].to_owned();
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if x < 0 { '-' } else { '+' });
        let e = x.unsigned_abs();
        if e < 10 {
            s.push('0'); // pad to at least two exponent digits
        }
        s.push_str(&e.to_string());
        s
    } else if x >= 0 {
        // Fixed form ≥ 1: pad the integer part with zeros when it is wider than the digit run.
        let int_len = usize::try_from(x).unwrap_or(0) + 1;
        if digits.len() <= int_len {
            let mut s = digits;
            let pad = int_len - s.len();
            s.push_str(&"0".repeat(pad));
            s
        } else {
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        }
    } else {
        // Fixed form < 1: `0.` then `(-x - 1)` leading zeros then the digits.
        let zeros = usize::try_from(-x - 1).unwrap_or(0);
        format!("0.{}{}", "0".repeat(zeros), digits)
    };
    if neg { format!("-{body}") } else { body }
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
    #[allow(
        clippy::unreadable_literal,
        reason = "these floats mirror the reference engine's exact inputs; separators would obscure them"
    )]
    fn float_text_matches_reference_engine() {
        // Every expected string is the reference engine's `x::float8::text`.
        for (f, want) in [
            (0.0_f64, "0"),
            (1.0, "1"),
            (100.0, "100"),
            (1234.5, "1234.5"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000001, "1e-06"),
            (1e14, "100000000000000"),
            (1.5e14, "150000000000000"),
            (1e15, "1e+15"),
            (1.5e15, "1.5e+15"),
            (1e20, "1e+20"),
            (123456789012345.0, "123456789012345"),
            (1234567890123456.0, "1.234567890123456e+15"),
            (1e-4, "0.0001"),
            (9.9e-5, "9.9e-05"),
            (1e-10, "1e-10"),
            (1e100, "1e+100"),
            (1e-308, "1e-308"),
            (0.30000000000000004, "0.30000000000000004"),
            (123456.789, "123456.789"),
        ] {
            assert_eq!(format_float(f), want, "format_float({f})");
        }
        assert_eq!(format_float(f64::INFINITY), "Infinity");
        assert_eq!(format_float(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(format_float(f64::NAN), "NaN");
        assert_eq!(format_float(-0.0), "-0");
    }

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
