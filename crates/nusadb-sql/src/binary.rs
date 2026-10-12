//! Binary value layouts, shared by binary result columns and binary `COPY`.
//!
//! A value has one binary form wherever it travels: a `DataRow` field negotiated as binary and a
//! field of a binary `COPY` stream carry the same bytes, so a client needs one decoder for both.
//! [`encode_value`] writes that form and [`decode_value`] reads it back at a column's type.
//!
//! Every multi-byte integer the layouts define is big-endian (network order); floating point is the
//! IEEE-754 bit pattern, also big-endian. The arbitrary-precision and arbitrary-length types
//! (`NUMERIC`, `JSON`, arrays, `VECTOR`, the geometric and full-text types, `XML`, enum labels)
//! use their canonical text, which is lossless and needs no shape-dependent layout.

use nusadb_core::ColumnType;

use crate::ast::Value;
use crate::error::Error;

/// SQLSTATE for a binary field whose bytes do not fit the column's layout (`bad_copy_file_format`).
const BAD_FORMAT: &str = "22P04";

/// Microseconds in one day: the upper bound of a time of day (`24:00:00` included).
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Encode a [`Value`] in the binary format.
///
/// Returns `None` for SQL `NULL` (a NULL carries no bytes, only its marker in the surrounding
/// message) and `Some(bytes)` otherwise. Byte layout per type:
///
/// | type          | layout                                                                |
/// | ------------- | --------------------------------------------------------------------- |
/// | `BOOL`        | 1 byte: `0x01` true, `0x00` false                                     |
/// | `INT`         | `i64`, 8 bytes big-endian                                            |
/// | `FLOAT`       | IEEE-754 `f64` bit pattern, 8 bytes big-endian                       |
/// | `NUMERIC`     | canonical decimal text, UTF-8 (lossless across arbitrary precision)  |
/// | `DATE`        | `i32` days since 1970-01-01, 4 bytes big-endian                      |
/// | `TIME`        | `i64` microseconds since midnight, 8 bytes big-endian                |
/// | `TIMETZ`      | `local micros:i64` then `zone secs west of UTC:i32`, big-endian (12 B) |
/// | `TIMESTAMP`   | `i64` microseconds since the epoch, 8 bytes big-endian               |
/// | `TIMESTAMPTZ` | `i64` microseconds since the epoch (UTC), 8 bytes big-endian          |
/// | `INTERVAL`    | `months:i32`, `days:i32`, `micros:i64`, all big-endian (16 bytes)     |
/// | `UUID`        | the 16 bytes verbatim                                                 |
/// | `TEXT`        | UTF-8 bytes verbatim (text and binary coincide)                      |
/// | `BYTEA`       | the raw bytes                                                         |
/// | `JSON`        | canonical JSON text, UTF-8                                            |
/// | `ARRAY`       | canonical `{...}` array text, UTF-8                                  |
/// | `VECTOR`      | canonical `[..]` vector text, UTF-8                                  |
///
/// The column's declared type tells the reader how to interpret each field, so `TIME` /
/// `TIMESTAMP` / `TIMESTAMPTZ` sharing an 8-byte layout is unambiguous.
#[must_use]
pub fn encode_value(value: &Value) -> Option<Vec<u8>> {
    let bytes = match value {
        Value::Null => return None,
        Value::Bool(b) => vec![u8::from(*b)],
        Value::Int(i) => i.to_be_bytes().to_vec(),
        Value::Float(f) => f.to_bits().to_be_bytes().to_vec(),
        Value::Numeric(d) => d.format().into_bytes(),
        Value::Date(d) => d.to_be_bytes().to_vec(),
        // TIME, TIMESTAMP, and TIMESTAMPTZ are each a microsecond count in an `i64`.
        Value::Time(t) | Value::Timestamp(t) | Value::TimestampTz(t) => t.to_be_bytes().to_vec(),
        // TIMETZ carries its zone: local time-of-day micros + zone seconds west of UTC, so a typed
        // client keeps the entered offset.
        Value::TimeTz(packed) => {
            let local = crate::temporal::timetz_local_micros(*packed);
            let zone_west = i32::try_from(-crate::temporal::timetz_offset_east_secs(*packed))
                .unwrap_or_default();
            let mut buf = Vec::with_capacity(12);
            buf.extend_from_slice(&local.to_be_bytes());
            buf.extend_from_slice(&zone_west.to_be_bytes());
            buf
        },
        Value::Uuid(u) => u.to_vec(),
        // MACADDR is its six raw bytes in transmission order.
        Value::Macaddr(m) => m.to_vec(),
        // MACADDR8 is its eight raw bytes in transmission order.
        Value::Macaddr8(m) => m.to_vec(),
        // GEOMETRY is sent as its canonical text form (`(x,y)` / `(hx,hy),(lx,ly)`).
        Value::Geometry(g) => crate::geometry::format(g).into_bytes(),
        // INET/CIDR: the self-describing `[flags, masklen, octets]` encoding.
        Value::Inet(a) => a.encode(),
        // BIT/BIT VARYING: the self-describing `[len, packed bytes]` encoding.
        Value::Bit(b) => crate::bit::encode(b),
        // RANGE: the element-kind tag then the self-describing flags-and-bounds encoding.
        Value::Range(r) => {
            let mut out = vec![r.kind.tag()];
            out.extend_from_slice(&crate::range::encode(r));
            out
        },
        Value::Interval(iv) => {
            let mut buf = Vec::with_capacity(16);
            buf.extend_from_slice(&iv.months.to_be_bytes());
            buf.extend_from_slice(&iv.days.to_be_bytes());
            buf.extend_from_slice(&iv.micros.to_be_bytes());
            buf
        },
        // Full-text values are sent as their canonical text form, like plain text.
        Value::Text(s) | Value::Tsvector(s) | Value::Tsquery(s) | Value::Xml(s) => {
            s.clone().into_bytes()
        },
        // An enum is sent as its label text (the ordinal is a server-side ordering detail).
        Value::Enum { label, .. } => label.clone().into_bytes(),
        // JSON is sent in the spaced display form (`{"a": 1}`).
        Value::Json(s) => crate::json::display_form(s).into_bytes(),
        Value::Array(items) => crate::display::array_text(items).into_bytes(),
        // VECTOR is sent as its canonical `[..]` text, like JSON/ARRAY (arbitrary length).
        Value::Vector(v) => crate::vector::format(v).into_bytes(),
        // BYTEA is sent as its raw bytes.
        Value::Bytes(b) => b.clone(),
    };
    Some(bytes)
}

/// Decode one binary field written by [`encode_value`] as a value for a column of type `ty`.
///
/// Fixed-width types must carry exactly their layout's width. Types whose binary form is their
/// canonical text come back as [`Value::Text`] and are converted to the column type by the same
/// write-time coercion a text `COPY` field goes through, so the two formats accept the same values.
///
/// # Errors
/// [`Error::Coded`] with `22P04` (`bad_copy_file_format`) when the bytes do not fit the layout,
/// `22021` when a text-form field is not valid UTF-8, and `22008` for a date, timestamp, time of
/// day or zone offset the text form would refuse as out of range.
pub fn decode_value(bytes: &[u8], ty: ColumnType) -> Result<Value, Error> {
    let bad = |detail: &str| bad_format(ty, detail);
    Ok(match ty.physical() {
        ColumnType::Bool => match fixed::<1>(bytes, ty)? {
            [0] => Value::Bool(false),
            [1] => Value::Bool(true),
            [other] => return Err(bad(&format!("boolean byte 0x{other:02X} is not 0 or 1"))),
        },
        ColumnType::Int => Value::Int(i64::from_be_bytes(fixed(bytes, ty)?)),
        ColumnType::Float => Value::Float(f64::from_bits(u64::from_be_bytes(fixed(bytes, ty)?))),
        ColumnType::Date => Value::Date(within_text_range(
            i32::from_be_bytes(fixed(bytes, ty)?),
            crate::temporal::format_date,
            crate::temporal::parse_date,
            "date",
        )?),
        ColumnType::Time => Value::Time(time_of_day(i64::from_be_bytes(fixed(bytes, ty)?))?),
        ColumnType::Timestamp => Value::Timestamp(within_text_range(
            i64::from_be_bytes(fixed(bytes, ty)?),
            crate::temporal::format_timestamp,
            crate::temporal::parse_timestamp,
            "timestamp",
        )?),
        ColumnType::TimestampTz => Value::TimestampTz(within_text_range(
            i64::from_be_bytes(fixed(bytes, ty)?),
            crate::temporal::format_timestamptz,
            crate::temporal::parse_timestamptz,
            "timestamptz",
        )?),
        ColumnType::TimeTz => {
            let raw: [u8; 12] = fixed(bytes, ty)?;
            let (local, zone) = raw.split_at(8);
            let local = time_of_day(i64::from_be_bytes(fixed(local, ty)?))?;
            let zone_west = i64::from(i32::from_be_bytes(fixed(zone, ty)?));
            if zone_west.abs() >= MICROS_PER_DAY / 1_000_000 {
                return Err(Error::Coded {
                    message: format!("time zone offset {zone_west} s is out of range"),
                    sqlstate: "22008", // datetime_field_overflow
                });
            }
            Value::TimeTz(crate::temporal::pack_timetz(local, -zone_west))
        },
        ColumnType::Interval => {
            let raw: [u8; 16] = fixed(bytes, ty)?;
            let (months, rest) = raw.split_at(4);
            let (days, micros) = rest.split_at(4);
            Value::Interval(crate::interval::Interval {
                months: i32::from_be_bytes(fixed(months, ty)?),
                days: i32::from_be_bytes(fixed(days, ty)?),
                micros: i64::from_be_bytes(fixed(micros, ty)?),
            })
        },
        ColumnType::Uuid => Value::Uuid(fixed(bytes, ty)?),
        ColumnType::Macaddr => Value::Macaddr(fixed(bytes, ty)?),
        ColumnType::Macaddr8 => Value::Macaddr8(fixed(bytes, ty)?),
        ColumnType::Bytes => Value::Bytes(bytes.to_vec()),
        ColumnType::Inet | ColumnType::Cidr => {
            let addr = crate::inet::InetAddr::decode(bytes)
                .filter(|a| a.encode().len() == bytes.len())
                .ok_or_else(|| bad("not a [flags, masklen, address] network value"))?;
            // Handed on as text so the column's own parser applies its rule (a CIDR refuses set
            // host bits), exactly as for a text field.
            Value::Text(addr.format())
        },
        ColumnType::Bit(_) | ColumnType::VarBit(_) => match crate::bit::decode(bytes, 0) {
            Some((bits, end)) if end == bytes.len() => Value::Bit(bits),
            _ => return Err(bad("not a [length, packed bits] bit string")),
        },
        ColumnType::Range(kind) => {
            let decoded = match bytes.split_first() {
                Some((&tag, rest)) if tag == kind.tag() => crate::range::decode(rest, 0, kind)
                    .filter(|&(_, end)| end == rest.len())
                    .map(|(range, _)| range),
                _ => None,
            };
            let range = decoded.ok_or_else(|| bad("not a range of the column's element type"))?;
            // The storage decoder trusts its input; a client's bytes are checked here. A numeric
            // bound must carry a real scale (the NaN marker and anything past the maximum are not
            // values a range can hold) before the range can be rendered.
            let bad_scale = [&range.lower, &range.upper].into_iter().flatten().any(
                |bound| matches!(bound, Value::Numeric(d) if d.scale > crate::numeric::MAX_SCALE),
            );
            if bad_scale {
                return Err(bad("a numeric bound has no valid scale"));
            }
            // Handed on as text so the column's own parser checks the bounds' order and puts a
            // discrete range in canonical form, exactly as for a text field.
            Value::Text(range.format())
        },
        // Text, and every type whose binary form is its canonical text.
        _ => Value::Text(
            std::str::from_utf8(bytes)
                .map_err(|_| Error::Coded {
                    message: format!("binary value for a {ty:?} column is not valid UTF-8"),
                    sqlstate: "22021", // character_not_in_repertoire
                })?
                .to_owned(),
        ),
    })
}

/// A field whose bytes do not fit the layout of a `ty` column.
fn bad_format(ty: ColumnType, detail: &str) -> Error {
    Error::Coded {
        message: format!("binary value for a {ty:?} column: {detail}"),
        sqlstate: BAD_FORMAT,
    }
}

/// A fixed-width layout: exactly `N` bytes, or the field is malformed.
fn fixed<const N: usize>(bytes: &[u8], ty: ColumnType) -> Result<[u8; N], Error> {
    bytes
        .try_into()
        .map_err(|_| bad_format(ty, &format!("expected {N} byte(s), found {}", bytes.len())))
}

/// Accept a decoded date or timestamp only if the text form accepts it too: rendered and parsed
/// back, it must come out unchanged. The binary layout can name instants the text parser refuses
/// (before year 0, past the last representable day); those are refused with `22008`, as text is.
fn within_text_range<T: Copy + PartialEq + std::fmt::Display>(
    value: T,
    format: impl Fn(T) -> String,
    parse: impl Fn(&str) -> Option<T>,
    what: &str,
) -> Result<T, Error> {
    if parse(&format(value)) == Some(value) {
        Ok(value)
    } else {
        Err(Error::Coded {
            message: format!("{what} value {value} is out of range"),
            sqlstate: "22008", // datetime_field_overflow
        })
    }
}

/// Check a decoded time of day lies within one day.
fn time_of_day(micros: i64) -> Result<i64, Error> {
    if (0..=MICROS_PER_DAY).contains(&micros) {
        Ok(micros)
    } else {
        Err(Error::Coded {
            message: format!("time of day {micros} us is out of range"),
            sqlstate: "22008", // datetime_field_overflow
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(value: &Value, ty: ColumnType) -> Value {
        decode_value(&encode_value(value).unwrap(), ty).unwrap()
    }

    #[test]
    fn fixed_width_types_round_trip() {
        assert_eq!(round_trip(&Value::Int(-7), ColumnType::Int), Value::Int(-7));
        assert_eq!(
            round_trip(&Value::Int(9), ColumnType::BigInt),
            Value::Int(9)
        );
        assert_eq!(
            round_trip(&Value::Float(1.5), ColumnType::Real),
            Value::Float(1.5)
        );
        assert_eq!(
            round_trip(&Value::Bool(true), ColumnType::Bool),
            Value::Bool(true)
        );
        assert_eq!(
            round_trip(&Value::Date(-3), ColumnType::Date),
            Value::Date(-3)
        );
        assert_eq!(
            round_trip(&Value::Time(5), ColumnType::Time),
            Value::Time(5)
        );
        let tz = crate::temporal::parse_timetz("13:45:30+07").unwrap();
        assert_eq!(
            round_trip(&Value::TimeTz(tz), ColumnType::TimeTz),
            Value::TimeTz(tz)
        );
        assert_eq!(
            round_trip(&Value::Bytes(Vec::new()), ColumnType::Bytes),
            Value::Bytes(Vec::new())
        );
    }

    #[test]
    fn wrong_width_is_bad_format() {
        let err = decode_value(&[0, 0, 0, 1], ColumnType::Int).unwrap_err();
        assert_eq!(err.sqlstate(), "22P04");
        let err = decode_value(&[2], ColumnType::Bool).unwrap_err();
        assert_eq!(err.sqlstate(), "22P04");
    }

    #[test]
    fn text_form_must_be_utf8() {
        let err = decode_value(&[0xFF], ColumnType::Text).unwrap_err();
        assert_eq!(err.sqlstate(), "22021");
        assert_eq!(
            decode_value(
                b"1.50",
                ColumnType::Numeric {
                    precision: 0,
                    scale: 0
                }
            )
            .unwrap(),
            Value::Text("1.50".to_owned())
        );
    }

    #[test]
    fn dates_and_timestamps_the_text_form_refuses_are_refused() {
        // Day -719_529 is 31 December of year -1, before the earliest date the text parser takes.
        for (bytes, ty) in [
            ((-719_529_i32).to_be_bytes().to_vec(), ColumnType::Date),
            (i64::MIN.to_be_bytes().to_vec(), ColumnType::Timestamp),
            (i64::MIN.to_be_bytes().to_vec(), ColumnType::TimestampTz),
        ] {
            let err = decode_value(&bytes, ty).unwrap_err();
            assert_eq!(err.sqlstate(), "22008", "{ty:?}");
        }
        // The first day of year 0 is still accepted.
        assert_eq!(
            decode_value(&(-719_528_i32).to_be_bytes(), ColumnType::Date).unwrap(),
            Value::Date(-719_528)
        );
    }

    /// The binary form of an int range: kind tag, flags, little-endian bounds.
    fn int_range(flags: u8, lower: i64, upper: i64) -> Vec<u8> {
        let mut out = vec![nusadb_core::engine::RangeKind::Int.tag(), flags];
        out.extend_from_slice(&lower.to_le_bytes());
        out.extend_from_slice(&upper.to_le_bytes());
        out
    }

    #[test]
    fn a_range_is_handed_on_as_text_for_the_column_to_check() {
        let ty = ColumnType::Range(nusadb_core::engine::RangeKind::Int);
        // `[10,1]`: inclusive bounds (flag bits 3 and 4) in the wrong order. Decoding does not
        // build a range from it; the text goes to the column's parser, which refuses it.
        assert_eq!(
            decode_value(&int_range(0x18, 10, 1), ty).unwrap(),
            Value::Text("[10,1]".to_owned())
        );
        assert_eq!(
            decode_value(&int_range(0x10, 1, 5), ty).unwrap(),
            Value::Text("(1,5]".to_owned())
        );
    }

    #[test]
    fn a_numeric_range_bound_without_a_valid_scale_is_bad_format() {
        let ty = ColumnType::Range(nusadb_core::engine::RangeKind::Num);
        for scale in [crate::numeric::MAX_SCALE + 1, crate::numeric::NAN_SCALE] {
            // Only a lower bound (the upper-infinite flag, bit 2) with the given scale byte.
            let mut bytes = vec![nusadb_core::engine::RangeKind::Num.tag(), 0x04 | 0x08];
            bytes.extend_from_slice(&1_i128.to_le_bytes());
            bytes.push(scale);
            let err = decode_value(&bytes, ty).unwrap_err();
            assert_eq!(err.sqlstate(), "22P04", "scale {scale}");
        }
    }

    #[test]
    fn time_of_day_out_of_range_is_refused() {
        let err = decode_value(&(-1_i64).to_be_bytes(), ColumnType::Time).unwrap_err();
        assert_eq!(err.sqlstate(), "22008");
    }
}
