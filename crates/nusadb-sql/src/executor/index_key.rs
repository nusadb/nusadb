//! Order-preserving index-key encoding.
//!
//! The engine stores index entries as opaque bytes and [`index_scan`] returns the matching rows in
//! ascending key-*byte* order over its range bounds. So the SQL layer must encode indexed column
//! values such that **lexicographic byte order matches SQL value order** — for every supported
//! scalar type and for composite (multi-column) keys. Each field's encoding is self-delimiting
//! (scalars are fixed width; text is `0x00`-terminated with byte-stuffing), so concatenating the
//! per-column encodings preserves tuple order.
//!
//! Keys are **encode-only**: the engine maps a key to a `tid` and the SQL layer fetches the row by
//! `tid`, so an index key is never decoded back into values.
//!
//! `NULL` sorts before every non-`NULL` value (a `0x00` field tag vs `0x01`). Types without a clean
//! order-preserving byte form — `NUMERIC`, `JSON`, `INTERVAL`, `ARRAY`, and `NaN` floats — are
//! rejected as index keys for v1.
//!
//! [`index_scan`]: nusadb_core::StorageEngine::index_scan

use crate::ast;
use crate::error::Error;

/// Sign bit of a 64-bit integer, used to bias a signed value into an order-preserving unsigned one.
const SIGN64: u64 = 1 << 63;
/// Sign bit of a 32-bit integer (for `DATE`, stored as `i32` days).
const SIGN32: u32 = 1 << 31;

/// Encode a composite index key — one value per indexed column, in the index's column order — into
/// order-preserving bytes. Returns [`Error::Unsupported`] if any column holds a type that has no v1
/// order-preserving encoding.
pub(super) fn encode_index_key(values: &[ast::Value]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    for value in values {
        encode_field(value, &mut out)?;
    }
    Ok(out)
}

/// Encode one key field: a `0x00` tag for `NULL` (sorts first), else `0x01` + the value's
/// order-preserving payload.
fn encode_field(value: &ast::Value, out: &mut Vec<u8>) -> Result<(), Error> {
    match value {
        ast::Value::Null => {
            out.push(0x00);
            Ok(())
        },
        non_null => {
            out.push(0x01);
            encode_non_null(non_null, out)
        },
    }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "intentional bit-level reinterpretation for order-preserving integer encoding"
)]
fn encode_non_null(value: &ast::Value, out: &mut Vec<u8>) -> Result<(), Error> {
    match value {
        // Booleans: false (0) < true (1).
        ast::Value::Bool(b) => out.push(u8::from(*b)),
        // Signed integers: flip the sign bit so negatives sort before positives, big-endian.
        ast::Value::Int(v)
        | ast::Value::Time(v)
        | ast::Value::TimeTz(v)
        | ast::Value::Timestamp(v)
        | ast::Value::TimestampTz(v) => {
            out.extend_from_slice(&((*v as u64) ^ SIGN64).to_be_bytes());
        },
        ast::Value::Date(v) => out.extend_from_slice(&((*v as u32) ^ SIGN32).to_be_bytes()),
        // IEEE-754: for a non-negative value flip the sign bit; for a negative value flip all bits —
        // so the big-endian result is monotonic (−∞ < … < +∞). NaN has no place in an ordering.
        // Note: `-0.0` and `+0.0` encode to *distinct* keys (`-0.0` first); harmless for range order,
        // but a future equality-lookup wiring must normalize them if it wants `0.0` to match `-0.0`.
        ast::Value::Float(f) => {
            if f.is_nan() {
                return Err(Error::Unsupported(
                    "NaN cannot be used as an index key".to_owned(),
                ));
            }
            let bits = f.to_bits();
            let ordered = if bits & SIGN64 == 0 {
                bits | SIGN64
            } else {
                !bits
            };
            out.extend_from_slice(&ordered.to_be_bytes());
        },
        // UUID bytes are already big-endian, so lexicographic order is UUID order.
        ast::Value::Uuid(bytes) => out.extend_from_slice(bytes),
        // Text / BYTEA: byte-stuffed and 0x00-terminated so a prefix sorts before its extensions.
        ast::Value::Text(s) => encode_ordered_bytes(s.as_bytes(), out),
        ast::Value::Bytes(b) => encode_ordered_bytes(b, out),
        // No v1 order-preserving form.
        ast::Value::Numeric(_)
        | ast::Value::Json(_)
        | ast::Value::Interval(_)
        | ast::Value::Array(_)
        | ast::Value::Vector(_) => {
            return Err(Error::Unsupported(
                "NUMERIC / JSON / INTERVAL / ARRAY / VECTOR columns cannot yet be index keys"
                    .to_owned(),
            ));
        },
        ast::Value::Null => {
            unreachable!("encode_field handles NULL before calling encode_non_null")
        },
    }
    Ok(())
}

/// Encode a byte string order-preservingly: a real `0x00` becomes `0x00 0xFF`, and a terminating
/// `0x00` ends the field. A `0x00` (end) sorts before `0x00 0xFF` (an escaped interior NUL) and
/// before any `0x01..` continuation, so a prefix sorts before its extensions (`"a"` < `"ab"`).
fn encode_ordered_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        if b == 0x00 {
            out.push(0x00);
            out.push(0xFF);
        } else {
            out.push(b);
        }
    }
    out.push(0x00);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encoding `a` then `b` must order the same way as the SQL values do.
    fn assert_order(a: ast::Value, b: ast::Value) {
        let (da, db) = (format!("{a:?}"), format!("{b:?}"));
        let ea = encode_index_key(&[a]).unwrap();
        let eb = encode_index_key(&[b]).unwrap();
        assert!(ea < eb, "expected {da} < {db} but bytes {ea:?} !< {eb:?}");
    }

    #[test]
    fn integers_are_order_preserving_across_sign() {
        for w in [i64::MIN, -1_000_000, -1, 0, 1, 42, 1_000_000, i64::MAX].windows(2) {
            assert_order(ast::Value::Int(w[0]), ast::Value::Int(w[1]));
        }
    }

    #[test]
    fn floats_are_order_preserving_across_sign_and_specials() {
        for w in [
            f64::NEG_INFINITY,
            -1e9,
            -1.5,
            -0.0,
            0.0,
            1.5,
            1e9,
            f64::INFINITY,
        ]
        .windows(2)
        {
            // The window is ascending; assert only strictly-ordered pairs (skips -0.0 vs 0.0,
            // which compare equal — `<` avoids a float `==`/`!=` comparison too).
            if w[0] < w[1] {
                assert_order(ast::Value::Float(w[0]), ast::Value::Float(w[1]));
            }
        }
    }

    #[test]
    fn nan_float_is_rejected() {
        assert!(encode_index_key(&[ast::Value::Float(f64::NAN)]).is_err());
    }

    #[test]
    fn text_prefix_sorts_before_extension_and_handles_nul() {
        assert_order(
            ast::Value::Text("a".to_owned()),
            ast::Value::Text("ab".to_owned()),
        );
        assert_order(
            ast::Value::Text("ab".to_owned()),
            ast::Value::Text("b".to_owned()),
        );
        // An interior NUL must not break ordering: "a" < "a\0".
        assert_order(
            ast::Value::Text("a".to_owned()),
            ast::Value::Text("a\u{0}".to_owned()),
        );
    }

    #[test]
    fn null_sorts_before_every_value() {
        assert_order(ast::Value::Null, ast::Value::Int(i64::MIN));
        assert_order(ast::Value::Null, ast::Value::Text(String::new()));
        assert_order(ast::Value::Null, ast::Value::Bool(false));
    }

    #[test]
    fn composite_keys_order_lexicographically() {
        let k = |a: i64, b: &str| {
            encode_index_key(&[ast::Value::Int(a), ast::Value::Text(b.to_owned())]).unwrap()
        };
        // Primary component dominates; the second breaks ties.
        assert!(k(1, "z") < k(2, "a"));
        assert!(k(1, "a") < k(1, "b"));
        assert!(k(1, "a") < k(1, "aa"));
    }

    #[test]
    fn dates_and_timestamps_order_across_sign() {
        assert_order(ast::Value::Date(-1), ast::Value::Date(0));
        assert_order(ast::Value::Date(0), ast::Value::Date(20_000));
        assert_order(ast::Value::Timestamp(-5), ast::Value::Timestamp(5));
    }

    #[test]
    fn bool_and_uuid_order_preserving() {
        assert_order(ast::Value::Bool(false), ast::Value::Bool(true));
        assert_order(
            ast::Value::Uuid([0; 16]),
            ast::Value::Uuid([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        );
        let mut max = [0xFF; 16];
        max[0] = 0xFE;
        assert_order(ast::Value::Uuid(max), ast::Value::Uuid([0xFF; 16]));
    }

    #[test]
    fn null_sorts_first_in_a_composite_tie_break_position() {
        // Same leading column; the second column is NULL in one row and a value in the other —
        // NULL must sort first within the tie.
        let with_null = encode_index_key(&[ast::Value::Int(1), ast::Value::Null]).unwrap();
        let with_value = encode_index_key(&[ast::Value::Int(1), ast::Value::Int(0)]).unwrap();
        assert!(with_null < with_value);
    }

    #[test]
    fn unsupported_key_types_are_rejected() {
        assert!(encode_index_key(&[ast::Value::Json("{}".to_owned())]).is_err());
        assert!(encode_index_key(&[ast::Value::Array(vec![ast::Value::Int(1)])]).is_err());
    }
}
