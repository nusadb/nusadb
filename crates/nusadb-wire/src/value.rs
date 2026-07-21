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
/// surrounding `DataRow`) and `Some(bytes)` otherwise. Byte layout per type:
///
/// | type          | layout                                                                |
/// | ------------- | --------------------------------------------------------------------- |
/// | `BOOL`        | 1 byte: `0x01` true, `0x00` false                                     |
/// | `INT`         | `i64`, 8 bytes big-endian                                            |
/// | `FLOAT`       | IEEE-754 `f64` bit pattern, 8 bytes big-endian                       |
/// | `NUMERIC`     | canonical decimal text, UTF-8 (lossless across arbitrary precision)  |
/// | `DATE`        | `i32` days since 1970-01-01, 4 bytes big-endian                      |
/// | `TIME`        | `i64` microseconds since midnight, 8 bytes big-endian                |
/// | `TIMETZ`      | `local micros:i64` ‖ `zone secs west of UTC:i32`, big-endian (12 B)  |
/// | `TIMESTAMP`   | `i64` microseconds since the epoch, 8 bytes big-endian               |
/// | `TIMESTAMPTZ` | `i64` microseconds since the epoch (UTC), 8 bytes big-endian          |
/// | `INTERVAL`    | `months:i32` ‖ `days:i32` ‖ `micros:i64`, all big-endian (16 bytes)  |
/// | `UUID`        | the 16 bytes verbatim                                                 |
/// | `TEXT`        | UTF-8 bytes verbatim (text and binary coincide)                      |
/// | `JSON`        | canonical JSON text, UTF-8                                            |
/// | `ARRAY`       | canonical `{...}` array text, UTF-8                                  |
/// | `VECTOR`      | canonical `[..]` vector text, UTF-8                                  |
///
/// `NUMERIC`, `JSON`, `ARRAY`, and `VECTOR` use their canonical text rather than a packed layout
/// because the value is arbitrary-precision / arbitrary-length: text is the lossless representation
/// and avoids
/// a precision- or shape-dependent frame layout. The column's declared type (carried by
/// `RowDescription`) tells the client how to read each field, so `TIME` / `TIMESTAMP` /
/// `TIMESTAMPTZ` sharing an 8-byte big-endian layout is unambiguous.
#[must_use]
pub fn encode_binary(value: &Value) -> Option<Vec<u8>> {
    let bytes = match value {
        Value::Null => return None,
        Value::Bool(b) => vec![u8::from(*b)],
        Value::Int(i) => i.to_be_bytes().to_vec(),
        Value::Float(f) => f.to_bits().to_be_bytes().to_vec(),
        Value::Numeric(d) => d.format().into_bytes(),
        Value::Date(d) => d.to_be_bytes().to_vec(),
        // TIME, TIMESTAMP, and TIMESTAMPTZ are each a microsecond count in an `i64`.
        Value::Time(t) | Value::Timestamp(t) | Value::TimestampTz(t) => t.to_be_bytes().to_vec(),
        // TIMETZ carries its zone (P-TIMETZ): local time-of-day micros + zone seconds west of
        // UTC, mirroring the reference engine's binary timetz layout, so a typed client keeps the entered offset.
        Value::TimeTz(packed) => {
            let local = nusadb_sql::temporal::timetz_local_micros(*packed);
            let zone_west = i32::try_from(-nusadb_sql::temporal::timetz_offset_east_secs(*packed))
                .unwrap_or_default();
            let mut buf = Vec::with_capacity(12);
            buf.extend_from_slice(&local.to_be_bytes());
            buf.extend_from_slice(&zone_west.to_be_bytes());
            buf
        },
        Value::Uuid(u) => u.to_vec(),
        Value::Interval(iv) => {
            let mut buf = Vec::with_capacity(16);
            buf.extend_from_slice(&iv.months.to_be_bytes());
            buf.extend_from_slice(&iv.days.to_be_bytes());
            buf.extend_from_slice(&iv.micros.to_be_bytes());
            buf
        },
        Value::Text(s) => s.clone().into_bytes(),
        // JSON is sent in the spaced display form (`{"a": 1}`), matching standard jsonb text output.
        Value::Json(s) => nusadb_sql::json::display_form(s).into_bytes(),
        Value::Array(items) => nusadb_sql::display::array_text(items).into_bytes(),
        // VECTOR is sent as its canonical `[..]` text, like JSON/ARRAY (arbitrary length).
        Value::Vector(v) => nusadb_sql::vector::format(v).into_bytes(),
        // BYTEA is sent as its raw bytes in the binary format.
        Value::Bytes(b) => b.clone(),
    };
    Some(bytes)
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
