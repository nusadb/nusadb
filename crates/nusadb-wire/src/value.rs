//! Binary value codec for `DataRow` fields.
//!
//! The Nusa Wire Protocol can transmit a column value in one of two formats, chosen per column by
//! the format code negotiated in `Bind`: the **text** format (a human-readable UTF-8
//! rendering — the server's default today) or the **binary** format implemented here, a compact
//! fixed-layout encoding a typed client can consume without re-parsing.
//!
//! Every multi-byte integer is big-endian (network order), matching the rest of the protocol;
//! floating point is the IEEE-754 bit pattern, also big-endian. The layouts are NusaDB's own and
//! are defined entirely by this module.
//!
//! Covers every [`Value`] type: bool, integer, float, numeric, the temporal family, interval,
//! uuid, text, JSON, and arrays (scalars + JSON/array). `NULL` carries no bytes.

use nusadb_sql::ast::Value;

/// Encode a [`Value`] in the **binary** wire format.
///
/// Returns `None` for SQL `NULL` (the field carries no bytes, only its NULL marker in the
/// surrounding `DataRow`) and `Some(bytes)` otherwise. The layouts are defined once, in
/// [`nusadb_sql::binary::encode_value`], so a binary result column and a binary `COPY` field carry
/// the same bytes; see that function for the per-type table.
#[must_use]
pub fn encode_binary(value: &Value) -> Option<Vec<u8>> {
    nusadb_sql::binary::encode_value(value)
}

#[cfg(test)]
mod tests {
    use nusadb_sql::interval::Interval;
    use nusadb_sql::numeric::Decimal;

    use super::*;

    /// Unwrap the `Some(..)` arm — every value in these tests is non-NULL.
    fn enc(value: &Value) -> Vec<u8> {
        encode_binary(value).unwrap()
    }

    #[test]
    fn null_has_no_bytes() {
        assert_eq!(encode_binary(&Value::Null), None);
    }

    #[test]
    fn bool_is_one_byte() {
        assert_eq!(enc(&Value::Bool(true)), [0x01]);
        assert_eq!(enc(&Value::Bool(false)), [0x00]);
    }

    #[test]
    fn int_is_8_bytes_big_endian() {
        assert_eq!(enc(&Value::Int(1)), [0, 0, 0, 0, 0, 0, 0, 1]);
        // -1 is all-ones in two's complement.
        assert_eq!(enc(&Value::Int(-1)), [0xFF; 8]);
        assert_eq!(
            enc(&Value::Int(i64::MAX)),
            [0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        );
    }

    #[test]
    fn float_is_ieee754_big_endian() {
        // 1.0 == 0x3FF0000000000000.
        assert_eq!(enc(&Value::Float(1.0)), 1.0_f64.to_bits().to_be_bytes());
        assert_eq!(
            enc(&Value::Float(1.0)),
            [0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn numeric_is_canonical_text() {
        let d = Decimal::parse("12.340").expect("parse decimal");
        assert_eq!(enc(&Value::Numeric(d)), d.format().into_bytes());
    }

    #[test]
    fn date_is_4_bytes_big_endian() {
        // 1970-01-02 is day 1.
        assert_eq!(enc(&Value::Date(1)), [0, 0, 0, 1]);
        assert_eq!(enc(&Value::Date(-1)), [0xFF; 4]);
    }

    #[test]
    fn temporal_micros_are_8_bytes_big_endian() {
        assert_eq!(enc(&Value::Time(1)), [0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(enc(&Value::Timestamp(1)), [0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(enc(&Value::TimestampTz(1)), [0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn timetz_is_local_micros_and_zone_west_12_bytes() {
        // '13:45:30+07' → local 49_530_000_000 µs, zone −25_200 s west (7h east), 12 bytes.
        let packed = nusadb_sql::temporal::parse_timetz("13:45:30+07").expect("parse timetz");
        let bytes = enc(&Value::TimeTz(packed));
        assert_eq!(bytes.len(), 12);
        assert_eq!(bytes[..8], 49_530_000_000_i64.to_be_bytes());
        assert_eq!(bytes[8..], (-25_200_i32).to_be_bytes());
    }

    #[test]
    fn uuid_is_16_bytes_verbatim() {
        let raw = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ];
        assert_eq!(enc(&Value::Uuid(raw)), raw);
    }

    #[test]
    fn interval_is_months_days_micros_big_endian() {
        let iv = Interval {
            months: 1,
            days: 2,
            micros: 3,
        };
        assert_eq!(
            enc(&Value::Interval(iv)),
            [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3]
        );
    }

    #[test]
    fn text_is_utf8_verbatim() {
        assert_eq!(enc(&Value::Text("héllo".to_owned())), "héllo".as_bytes());
    }

    #[test]
    fn json_is_sent_in_spaced_display_form() {
        // The stored value is compact; the wire sends the spaced display form (standard jsonb text).
        assert_eq!(enc(&Value::Json("{\"a\":1}".to_owned())), b"{\"a\": 1}");
    }

    #[test]
    fn array_is_canonical_braced_text() {
        let array = Value::Array(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(
            enc(&array),
            nusadb_sql::display::array_text(&[Value::Int(1), Value::Int(2)]).into_bytes(),
        );
    }
}
