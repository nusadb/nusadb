//! Self-describing codec for spilling intermediate [`Row`](crate::executor::row::Row)s.
//!
//! Unlike [`row::encode`](crate::executor::row::encode) — which is schema-driven and only handles
//! the value variants a stored tuple can hold — an intermediate executor row can carry *any*
//! [`ast::Value`] (a `DATE` from a cast, a `NUMERIC` from arithmetic, a nested `ARRAY`, …). So each
//! value is tagged with its variant and decoded without an external schema.
//!
//! Layout: a row is `[count:u32]` then `count` values; each value is `[tag:u8]` then the payload
//! below (all integers little-endian):
//!
//! | tag | variant | payload |
//! |-----|---------|---------|
//! | 0 | `Null` | — |
//! | 1 | `Bool` | 1 byte (`0`/`1`) |
//! | 2 | `Int` | `i64` |
//! | 3 | `Float` | `f64` (`to_le_bytes`, bit-exact) |
//! | 4 | `Text` | `[len:u32]` + UTF-8 bytes |
//! | 5 | `Date` | `i32` |
//! | 6 | `Time` | `i64` |
//! | 7 | `Timestamp` | `i64` |
//! | 8 | `TimestampTz` | `i64` |
//! | 9 | `TimeTz` | `i64` |
//! | 10 | `Uuid` | 16 bytes |
//! | 11 | `Numeric` | `i128` mantissa + `u8` scale |
//! | 12 | `Json` | `[len:u32]` + UTF-8 bytes |
//! | 13 | `Interval` | `i32` months + `i32` days + `i64` micros |
//! | 14 | `Array` | `[count:u32]` + that many values (recursive) |
//! | 15 | `Vector` | `[count:u32]` + `count` × `f32` (`to_le_bytes`) |

use crate::ast;
use crate::error::Error;
use crate::interval::Interval;
use crate::numeric::Decimal;

const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT: u8 = 2;
const TAG_FLOAT: u8 = 3;
const TAG_TEXT: u8 = 4;
const TAG_DATE: u8 = 5;
const TAG_TIME: u8 = 6;
const TAG_TIMESTAMP: u8 = 7;
const TAG_TIMESTAMPTZ: u8 = 8;
const TAG_TIMETZ: u8 = 9;
const TAG_UUID: u8 = 10;
const TAG_NUMERIC: u8 = 11;
const TAG_JSON: u8 = 12;
const TAG_INTERVAL: u8 = 13;
const TAG_ARRAY: u8 = 14;
const TAG_VECTOR: u8 = 15;
const TAG_BYTES: u8 = 16;

/// Encode `row` into the self-describing byte form.
///
/// # Errors
/// [`Error::MalformedTuple`] if a length/count prefix (a column count, an array length, or a
/// text/JSON byte length) does not fit in the `u32` the format reserves — a clean error rather than
/// a silently truncating cast, matching [`row::encode`](crate::executor::row::encode).
pub(super) fn encode_row(row: &[ast::Value]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(row.len() * 9 + 4);
    write_len(&mut out, row.len())?;
    for v in row {
        write_value(&mut out, v)?;
    }
    Ok(out)
}

/// Decode a row written by [`encode_row`].
///
/// # Errors
/// [`Error::MalformedTuple`] if the bytes are truncated or carry an unknown variant tag.
pub(super) fn decode_row(bytes: &[u8]) -> Result<Vec<ast::Value>, Error> {
    let mut c = Cursor { bytes, pos: 0 };
    let n = c.u32()? as usize;
    // Bound the speculative reservation by the bytes left (each value is at least its 1-byte tag),
    // so a corrupt count cannot reserve a huge Vec (same guard as the wire decoder).
    let mut row = Vec::with_capacity(n.min(c.remaining()));
    for _ in 0..n {
        row.push(read_value(&mut c)?);
    }
    Ok(row)
}

fn write_value(out: &mut Vec<u8>, v: &ast::Value) -> Result<(), Error> {
    match v {
        ast::Value::Null => out.push(TAG_NULL),
        ast::Value::Bool(b) => {
            out.push(TAG_BOOL);
            out.push(u8::from(*b));
        },
        ast::Value::Int(i) => write_tagged_i64(out, TAG_INT, *i),
        ast::Value::Float(f) => {
            out.push(TAG_FLOAT);
            out.extend_from_slice(&f.to_le_bytes());
        },
        ast::Value::Text(s) => write_tagged_bytes(out, TAG_TEXT, s.as_bytes())?,
        ast::Value::Date(d) => {
            out.push(TAG_DATE);
            out.extend_from_slice(&d.to_le_bytes());
        },
        ast::Value::Time(t) => write_tagged_i64(out, TAG_TIME, *t),
        ast::Value::Timestamp(t) => write_tagged_i64(out, TAG_TIMESTAMP, *t),
        ast::Value::TimestampTz(t) => write_tagged_i64(out, TAG_TIMESTAMPTZ, *t),
        ast::Value::TimeTz(t) => write_tagged_i64(out, TAG_TIMETZ, *t),
        ast::Value::Uuid(bytes) => {
            out.push(TAG_UUID);
            out.extend_from_slice(bytes);
        },
        ast::Value::Numeric(d) => {
            out.push(TAG_NUMERIC);
            out.extend_from_slice(&d.mantissa.to_le_bytes());
            out.push(d.scale);
        },
        ast::Value::Json(s) => write_tagged_bytes(out, TAG_JSON, s.as_bytes())?,
        ast::Value::Interval(iv) => {
            out.push(TAG_INTERVAL);
            out.extend_from_slice(&iv.months.to_le_bytes());
            out.extend_from_slice(&iv.days.to_le_bytes());
            out.extend_from_slice(&iv.micros.to_le_bytes());
        },
        ast::Value::Array(items) => {
            out.push(TAG_ARRAY);
            write_len(out, items.len())?;
            for item in items {
                write_value(out, item)?;
            }
        },
        ast::Value::Vector(v) => {
            out.push(TAG_VECTOR);
            write_len(out, v.len())?;
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        },
        ast::Value::Bytes(b) => write_tagged_bytes(out, TAG_BYTES, b)?,
    }
    Ok(())
}

fn read_value(c: &mut Cursor<'_>) -> Result<ast::Value, Error> {
    let tag = c.u8()?;
    let value = match tag {
        TAG_NULL => ast::Value::Null,
        TAG_BOOL => ast::Value::Bool(c.u8()? != 0),
        TAG_INT => ast::Value::Int(c.i64()?),
        TAG_FLOAT => ast::Value::Float(f64::from_le_bytes(c.arr::<8>()?)),
        TAG_TEXT => ast::Value::Text(c.string()?),
        TAG_DATE => ast::Value::Date(i32::from_le_bytes(c.arr::<4>()?)),
        TAG_TIME => ast::Value::Time(c.i64()?),
        TAG_TIMESTAMP => ast::Value::Timestamp(c.i64()?),
        TAG_TIMESTAMPTZ => ast::Value::TimestampTz(c.i64()?),
        TAG_TIMETZ => ast::Value::TimeTz(c.i64()?),
        TAG_UUID => ast::Value::Uuid(c.arr::<16>()?),
        TAG_NUMERIC => ast::Value::Numeric(Decimal {
            mantissa: i128::from_le_bytes(c.arr::<16>()?),
            scale: c.u8()?,
        }),
        TAG_JSON => ast::Value::Json(c.string()?),
        TAG_INTERVAL => ast::Value::Interval(Interval {
            months: i32::from_le_bytes(c.arr::<4>()?),
            days: i32::from_le_bytes(c.arr::<4>()?),
            micros: c.i64()?,
        }),
        TAG_ARRAY => {
            let n = c.u32()? as usize;
            let mut items = Vec::with_capacity(n.min(c.remaining()));
            for _ in 0..n {
                items.push(read_value(c)?);
            }
            ast::Value::Array(items)
        },
        TAG_VECTOR => {
            let n = c.u32()? as usize;
            // Each component is 4 bytes; cap the reservation by the bytes left.
            let mut v = Vec::with_capacity(n.min(c.remaining() / 4));
            for _ in 0..n {
                v.push(f32::from_le_bytes(c.arr::<4>()?));
            }
            ast::Value::Vector(v)
        },
        TAG_BYTES => {
            let n = c.u32()? as usize;
            ast::Value::Bytes(c.take(n)?.to_vec())
        },
        _ => return Err(Error::MalformedTuple { offset: c.pos - 1 }),
    };
    Ok(value)
}

/// Write a length/count as a `u32` prefix, erroring rather than truncating if it does not fit.
fn write_len(out: &mut Vec<u8>, n: usize) -> Result<(), Error> {
    let n = u32::try_from(n).map_err(|_| Error::MalformedTuple { offset: out.len() })?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

fn write_tagged_i64(out: &mut Vec<u8>, tag: u8, v: i64) {
    out.push(tag);
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_tagged_bytes(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) -> Result<(), Error> {
    out.push(tag);
    write_len(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

/// A bounds-checked forward cursor over a spilled record's bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(Error::MalformedTuple { offset: self.pos })?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(Error::MalformedTuple { offset: self.pos })?;
        self.pos = end;
        Ok(slice)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?); // take(N) yields exactly N bytes → lengths match
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.arr::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.arr::<4>()?))
    }

    fn i64(&mut self) -> Result<i64, Error> {
        Ok(i64::from_le_bytes(self.arr::<8>()?))
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::MalformedTuple { offset: self.pos })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants_row() -> Vec<ast::Value> {
        vec![
            ast::Value::Null,
            ast::Value::Bool(true),
            ast::Value::Bool(false),
            ast::Value::Int(-9_223_372_036_854_775_808),
            ast::Value::Float(-0.5),
            ast::Value::Text("héllo 🦀".to_owned()),
            ast::Value::Text(String::new()),
            ast::Value::Date(-19_000),
            ast::Value::Time(86_399_999_999),
            ast::Value::Timestamp(1_700_000_000_000_000),
            ast::Value::TimestampTz(-42),
            ast::Value::TimeTz(0),
            ast::Value::Uuid([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 255]),
            ast::Value::Numeric(Decimal {
                mantissa: -123_456_789_012_345_678_901_234,
                scale: 12,
            }),
            ast::Value::Json("{\"a\":1}".to_owned()),
            ast::Value::Interval(Interval {
                months: -13,
                days: 40,
                micros: -1,
            }),
            ast::Value::Array(vec![
                ast::Value::Int(1),
                ast::Value::Null,
                ast::Value::Array(vec![ast::Value::Text("nested".to_owned())]),
            ]),
            // `Vector` (tag 15): empty + multi-component, incl. a non-finite f32 (bit-exact LE).
            ast::Value::Vector(vec![]),
            ast::Value::Vector(vec![1.5, -0.0, f32::INFINITY, 3.25]),
        ]
    }

    #[test]
    fn every_value_variant_round_trips() {
        let row = all_variants_row();
        let decoded = decode_row(&encode_row(&row).expect("encode")).expect("round-trips");
        assert_eq!(decoded, row);
    }

    #[test]
    fn empty_row_round_trips() {
        let decoded = decode_row(&encode_row(&[]).expect("encode")).expect("round-trips");
        assert!(decoded.is_empty());
    }

    #[test]
    fn truncated_bytes_error_instead_of_panicking() {
        let bytes = encode_row(&all_variants_row()).expect("encode");
        for cut in 0..bytes.len() {
            // Every prefix shorter than the whole record must error cleanly, never panic/index-OOB.
            let _ = decode_row(&bytes[..cut]);
        }
        assert!(matches!(
            decode_row(&[0xFF]), // a u32 count that runs off the end
            Err(Error::MalformedTuple { .. })
        ));
    }

    #[test]
    fn unknown_tag_is_rejected() {
        // count=1, then tag 200 (no such variant).
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.push(200);
        assert!(matches!(
            decode_row(&bytes),
            Err(Error::MalformedTuple { .. })
        ));
    }
}
