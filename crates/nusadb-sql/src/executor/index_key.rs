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
                return Err(Error::InvalidParameterValue(
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
        // MACADDR is six fixed big-endian bytes, so lexicographic order is MAC-address order.
        ast::Value::Macaddr(bytes) => out.extend_from_slice(bytes),
        // MACADDR8 is eight fixed big-endian bytes, so lexicographic order is EUI-64 order.
        ast::Value::Macaddr8(bytes) => out.extend_from_slice(bytes),
        // Text / BYTEA: byte-stuffed and 0x00-terminated so a prefix sorts before its extensions.
        ast::Value::Text(s) => encode_ordered_bytes(s.as_bytes(), out),
        ast::Value::Bytes(b) => encode_ordered_bytes(b, out),
        // NUMERIC: a fixed-width canonical decimal layout (see `encode_numeric`).
        ast::Value::Numeric(d) => encode_numeric(d, out)?,
        // No v1 order-preserving form. INET/CIDR join here because their network order is not a
        // byte order (the masked prefix is compared before the mask), so a lexicographic key would
        // mis-order them; equality-only indexing is a follow-up.
        ast::Value::Json(_)
        | ast::Value::Interval(_)
        | ast::Value::Array(_)
        | ast::Value::Vector(_)
        | ast::Value::Inet(_)
        | ast::Value::Bit(_)
        | ast::Value::Range(_) => {
            return Err(Error::Unsupported(
                "JSON / INTERVAL / ARRAY / VECTOR / INET / CIDR / BIT / RANGE columns cannot yet be \
                 index keys"
                    .to_owned(),
            ));
        },
        ast::Value::Null => {
            unreachable!("encode_field handles NULL before calling encode_non_null")
        },
    }
    Ok(())
}

/// The largest `i128` magnitude has 39 decimal digits, and the fractional scale is capped at
/// [`numeric::MAX_SCALE`] (38), so every `NUMERIC` value fits in a fixed grid of 39 integer + 38
/// fractional digit columns.
const NUMERIC_INT_DIGITS: usize = 39;
const NUMERIC_FRAC_DIGITS: usize = crate::numeric::MAX_SCALE as usize;

/// Encode a `NUMERIC` order-preservingly as a **fixed-width canonical decimal**: a sign-class byte
/// (negative sorts before non-negative), then 39 integer digit columns (left-padded with `0`) and
/// 38 fractional digit columns (right-padded with `0`), each digit as its ASCII byte. Fixed-width
/// columns make leading/trailing zeros align magnitudes, so byte order equals numeric order, and two
/// spellings of the same value (`1.5` and `1.50`) produce identical bytes — the equality an index
/// needs. For a negative value every digit is inverted (`9 − d`) so a larger magnitude sorts lower.
fn encode_numeric(d: &crate::numeric::Decimal, out: &mut Vec<u8>) -> Result<(), Error> {
    let scale = d.scale as usize;
    if scale > NUMERIC_FRAC_DIGITS {
        return Err(Error::LimitExceeded(
            "NUMERIC scale beyond the index key limit".to_owned(),
        ));
    }
    let negative = d.mantissa < 0;
    // Sign class: negatives (0x00) sort before zero and positives (0x02); zero's all-`0` digits sort
    // below any positive within the non-negative branch.
    out.push(if negative { 0x00 } else { 0x02 });

    // Decimal digits of the magnitude (no leading zeros; "0" for zero). `unsigned_abs` handles
    // `i128::MIN`, whose magnitude does not fit in `i128`.
    let digits = d.mantissa.unsigned_abs().to_string();
    let dbytes = digits.as_bytes();
    let dlen = dbytes.len();
    // The last `scale` digits are fractional; the rest are the integer part. When the magnitude is
    // below 1 (`dlen <= scale`), the integer part is empty and the fraction gets leading zeros.
    let frac_padded;
    let (int_part, frac_part): (&[u8], &[u8]) = if dlen > scale {
        dbytes.split_at(dlen - scale)
    } else {
        frac_padded = {
            let mut f = vec![b'0'; scale - dlen];
            f.extend_from_slice(dbytes);
            f
        };
        (b"", frac_padded.as_slice())
    };

    // Emit a digit column, inverting for negatives so a larger magnitude sorts lower.
    let emit = |out: &mut Vec<u8>, c: u8| out.push(if negative { b'9' - (c - b'0') } else { c });
    // Integer part, left-padded to 39 columns.
    for _ in 0..NUMERIC_INT_DIGITS - int_part.len() {
        emit(out, b'0');
    }
    for &c in int_part {
        emit(out, c);
    }
    // Fractional part, right-padded to 38 columns.
    for &c in frac_part {
        emit(out, c);
    }
    for _ in 0..NUMERIC_FRAC_DIGITS - frac_part.len() {
        emit(out, b'0');
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
    fn numeric_is_order_preserving_and_scale_canonical() {
        use crate::numeric::Decimal;
        let dec = |m: i128, s: u8| Decimal {
            mantissa: m,
            scale: s,
        };
        let key = |d: Decimal| encode_index_key(&[ast::Value::Numeric(d)]).unwrap();

        // Explicit, hand-verified orderings — not relying on `Decimal::compare`.
        assert!(key(dec(-15, 1)) < key(dec(0, 0))); // -1.5 < 0
        assert!(key(dec(0, 0)) < key(dec(15, 1))); //  0  < 1.5
        assert_eq!(key(dec(15, 1)), key(dec(150, 2))); // 1.5 == 1.50 (scale-canonical)
        assert_eq!(key(dec(0, 0)), key(dec(0, 7))); //  0  == 0.0000000
        assert!(key(dec(5, 2)) < key(dec(5, 1))); // 0.05 < 0.5
        assert!(key(dec(-25, 1)) < key(dec(-15, 1))); // -2.5 < -1.5
        assert!(key(dec(149, 2)) < key(dec(15, 1))); // 1.49 < 1.5
        assert!(key(dec(15, 1)) < key(dec(151, 2))); // 1.5  < 1.51

        // A diverse set plus a deterministic pseudo-random sweep, cross-compared against the exact
        // `Decimal::compare`. Skip the rare extreme cross-scale pairs where `compare` itself takes a
        // rescale-overflow fallback (it is not the oracle there); the fixed-width encoding is exact.
        let mut values: Vec<Decimal> = vec![
            dec(0, 0),
            dec(1, 0),
            dec(15, 1),
            dec(150, 2),
            dec(149, 2),
            dec(151, 2),
            dec(-15, 1),
            dec(-150, 2),
            dec(5, 2),
            dec(5, 1),
            dec(-5, 2),
            dec(i128::MAX, 0),
            dec(i128::MIN, 0),
            dec(i128::MAX, 38),
            dec(i128::MIN, 38),
            dec(1, 38),
            dec(-1, 38),
        ];
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for _ in 0..400 {
            let m = (u128::from(next()) | (u128::from(next()) << 64)).cast_signed();
            let s = (next() % 39) as u8;
            values.push(dec(m, s));
        }

        for &a in &values {
            for &b in &values {
                let common = a.scale.max(b.scale);
                if a.rescale(common).is_none() || b.rescale(common).is_none() {
                    continue; // `compare` is not exact for this pair; encoding still is.
                }
                let by_bytes = key(a).cmp(&key(b));
                let by_value = a.compare(&b);
                assert_eq!(
                    by_bytes, by_value,
                    "order mismatch: {a:?} vs {b:?} — value {by_value:?}, bytes {by_bytes:?}"
                );
            }
        }
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
