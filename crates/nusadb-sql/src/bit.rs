//! `BIT(n)` and `BIT VARYING(n)` — fixed- and variable-length bit strings.
//!
//! A bit string is a sequence of `0`/`1` bits, represented as a `Vec<bool>` (index 0 is the leftmost
//! bit). Both SQL types share this value; the column type carries the length rule — `BIT(n)` requires
//! exactly `n` bits, `BIT VARYING(n)` at most `n` (unbounded when `n` is omitted).
//!
//! Dependency-free and panic-free: [`parse`] rejects a non-binary character with `None`; the caller
//! raises the typed SQL error. Ordering compares the bits left to right, then the shorter string
//! first on a common prefix (the natural `Vec<bool>` order), matching the reference engine.

/// Parse a bit-string literal (`"1011"`) into its bits, or `None` if any character is not `0`/`1`.
#[must_use]
pub fn parse(s: &str) -> Option<Vec<bool>> {
    s.chars()
        .map(|c| match c {
            '0' => Some(false),
            '1' => Some(true),
            _ => None,
        })
        .collect()
}

/// The natural column type of a bit-string value — a fixed `BIT(len)`. (A value read from a
/// `BIT VARYING` column reports the same, since the two share a runtime representation.)
#[must_use]
pub fn column_type(bits: &[bool]) -> nusadb_core::ColumnType {
    nusadb_core::ColumnType::Bit(u32::try_from(bits.len()).unwrap_or(u32::MAX))
}

/// Fit `bits` to exactly `n` bits — the `::bit(n)` cast.
///
/// Right-truncates when longer, right-pads with zeros when shorter. An explicit cast adjusts the
/// length this way, whereas a column assignment of a wrong-length value is rejected instead.
#[must_use]
pub fn fit(bits: &[bool], n: usize) -> Vec<bool> {
    let mut out: Vec<bool> = bits.iter().take(n).copied().collect();
    out.resize(n, false);
    out
}

/// Truncate `bits` to at most `max` (right-truncation) — the `::bit varying(n)` cast; `None` = keep.
#[must_use]
pub fn truncate(bits: &[bool], max: Option<usize>) -> Vec<bool> {
    match max {
        Some(n) if bits.len() > n => bits.iter().take(n).copied().collect(),
        _ => bits.to_vec(),
    }
}

/// Shift `bits` by `amount` places, keeping the same length — the `<<` / `>>` operators.
///
/// A left shift moves bits toward index 0 (dropping the leading bits and zero-filling the trailing
/// ones); a right shift is the mirror. A negative `amount` reverses the direction, matching the
/// reference engine.
#[must_use]
pub fn shift(bits: &[bool], amount: i64, left: bool) -> Vec<bool> {
    let toward_front = if amount < 0 { !left } else { left };
    let k = usize::try_from(amount.unsigned_abs()).unwrap_or(usize::MAX);
    (0..bits.len())
        .map(|i| {
            let src = if toward_front {
                i.checked_add(k)
            } else {
                i.checked_sub(k)
            };
            src.and_then(|s| bits.get(s).copied()).unwrap_or(false)
        })
        .collect()
}

/// Complement every bit (the `~` operator), preserving the length: `~1011` is `0100`.
#[must_use]
pub fn complement(bits: &[bool]) -> Vec<bool> {
    bits.iter().map(|&b| !b).collect()
}

/// Build a `BIT(n)` from an integer — the `integer::bit(n)` cast.
///
/// Takes the low `n` bits of the value's 32-bit two's-complement representation, most-significant
/// first. Bit position `p` (counted from the right) is bit `p` of the 32-bit value; positions at or
/// beyond 32 repeat the sign bit, so a positive value zero-extends and a negative one sign-extends
/// (`5::bit(40)` is 40 bits ending `101`, `(-1)::bit(8)` is all ones).
#[must_use]
pub fn from_int(value: i64, n: usize) -> Vec<bool> {
    let v = value.cast_unsigned(); // reinterpret the two's-complement bits, no value change
    let sign = (v >> 31) & 1 == 1; // bit 31 is the 32-bit int's sign
    (0..n)
        .rev()
        .map(|p| if p < 32 { (v >> p) & 1 == 1 } else { sign })
        .collect()
}

/// Reinterpret a bit string as a signed 32-bit integer — the `bit::integer` cast.
///
/// The bits right-align in a 32-bit field and are read as a two's-complement `i32` (widened to
/// `i64`): a string of fewer than 32 bits is always non-negative, a 32-bit string with the top bit
/// set is negative. `None` when the string is longer than 32 bits (the value would not fit).
#[must_use]
pub fn to_int(bits: &[bool]) -> Option<i64> {
    if bits.len() > 32 {
        return None;
    }
    let mut acc: u32 = 0;
    for &b in bits {
        acc = (acc << 1) | u32::from(b);
    }
    Some(i64::from(acc.cast_signed()))
}

/// Read bit `i` of a `BYTEA` value — `get_bit(bytea, int)`.
///
/// Bits are numbered least-significant-first within each byte (bit 0 = the `0x01` bit of the first
/// byte, bit 7 = its `0x80` bit, bit 8 = the `0x01` bit of the second byte). `None` when `i` is at
/// or past the last bit.
#[must_use]
pub fn bytea_get_bit(bytes: &[u8], i: usize) -> Option<bool> {
    let byte = bytes.get(i / 8)?;
    Some((byte >> (i % 8)) & 1 == 1)
}

/// Set bit `i` (same numbering as [`bytea_get_bit`]) of a `BYTEA` value — `set_bit(bytea, int, int)`.
/// The caller bounds-checks `i`; an out-of-range index leaves the bytes unchanged.
#[must_use]
pub fn bytea_set_bit(bytes: &[u8], i: usize, one: bool) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if let Some(byte) = out.get_mut(i / 8) {
        let mask = 1u8 << (i % 8);
        if one {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }
    out
}

/// Parse a hex-string literal (`X'1a'`) into its bits — four bits per hex digit, most-significant
/// first (`X'1a'` is `00011010`). `None` if any character is not a hex digit.
#[must_use]
pub fn from_hex(s: &str) -> Option<Vec<bool>> {
    let mut bits = Vec::with_capacity(s.len() * 4);
    for c in s.chars() {
        let nibble = c.to_digit(16)?;
        for p in (0..4).rev() {
            bits.push((nibble >> p) & 1 == 1);
        }
    }
    Some(bits)
}

/// Render a bit string as its canonical text (`"1011"`).
#[must_use]
pub fn format(bits: &[bool]) -> String {
    bits.iter().map(|&b| if b { '1' } else { '0' }).collect()
}

/// Pack bits into bytes for storage: a `u32` length prefix (LE) then `ceil(len/8)` bytes, each bit
/// most-significant-first, the final byte zero-padded.
#[must_use]
pub fn encode(bits: &[bool]) -> Vec<u8> {
    let len = u32::try_from(bits.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(4 + bits.len().div_ceil(8));
    out.extend_from_slice(&len.to_le_bytes());
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << (7 - i);
            }
        }
        out.push(byte);
    }
    out
}

/// Unpack `len` bits from `data` (most-significant-first packing), or `None` if `data` is short.
#[must_use]
pub fn unpack(len: usize, data: &[u8]) -> Option<Vec<bool>> {
    let mut bits = Vec::with_capacity(len);
    for i in 0..len {
        let byte = data.get(i / 8)?;
        bits.push(byte & (1 << (7 - (i % 8))) != 0);
    }
    Some(bits)
}

/// The number of packed bytes a bit string of `len` bits occupies.
#[must_use]
pub const fn packed_len(len: usize) -> usize {
    len.div_ceil(8)
}

/// Decode the [`encode`] form starting at `pos`, returning the bits and the new position.
#[must_use]
pub fn decode(bytes: &[u8], pos: usize) -> Option<(Vec<bool>, usize)> {
    let len_bytes: [u8; 4] = bytes.get(pos..pos + 4)?.try_into().ok()?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let n_bytes = packed_len(len);
    let data = bytes.get(pos + 4..pos + 4 + n_bytes)?;
    let bits = unpack(len, data)?;
    Some((bits, pos + 4 + n_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format_round_trip() {
        assert_eq!(parse("1011"), Some(vec![true, false, true, true]));
        assert_eq!(parse(""), Some(vec![]));
        assert_eq!(parse("102"), None);
        assert_eq!(format(&[true, false, true, true]), "1011");
        assert_eq!(format(&[]), "");
    }

    #[test]
    fn from_hex_expands_each_digit_to_four_bits() {
        assert_eq!(
            from_hex("1a").map(|b| format(&b)),
            Some("00011010".to_owned())
        );
        assert_eq!(
            from_hex("ff").map(|b| format(&b)),
            Some("11111111".to_owned())
        );
        assert_eq!(from_hex("0").map(|b| format(&b)), Some("0000".to_owned()));
        assert_eq!(from_hex("").map(|b| format(&b)), Some(String::new()));
        assert_eq!(from_hex("1g"), None);
    }

    #[test]
    fn complement_flips_every_bit_preserving_length() {
        assert_eq!(format(&complement(&parse("10110").unwrap())), "01001");
        assert_eq!(format(&complement(&parse("0000").unwrap())), "1111");
        assert_eq!(format(&complement(&[])), "");
        // Double complement is the identity.
        let b = parse("1010011").unwrap();
        assert_eq!(complement(&complement(&b)), b);
    }

    #[test]
    fn encode_decode_round_trip() {
        for s in ["", "1", "1011", "10000001", "101100110", "1111111111111111"] {
            let bits = parse(s).unwrap();
            let bytes = encode(&bits);
            assert_eq!(
                decode(&bytes, 0),
                Some((bits.clone(), bytes.len())),
                "for {s}"
            );
        }
    }

    #[test]
    fn from_int_takes_low_n_bits_msb_first() {
        assert_eq!(format(&from_int(11, 8)), "00001011");
        assert_eq!(format(&from_int(259, 8)), "00000011"); // low 8 bits of 259 (0x103)
        assert_eq!(format(&from_int(-1, 8)), "11111111");
        assert_eq!(format(&from_int(5, 16)), "0000000000000101");
        // A width beyond 32 sign-extends: positive zero-fills, negative one-fills.
        assert_eq!(
            format(&from_int(5, 40)),
            "0000000000000000000000000000000000000101"
        );
        assert_eq!(
            format(&from_int(-1, 40)),
            "1111111111111111111111111111111111111111"
        );
        assert_eq!(
            format(&from_int(-2, 35)),
            "11111111111111111111111111111111110"
        );
        assert_eq!(format(&from_int(0, 0)), "");
    }

    #[test]
    fn to_int_reads_signed_32bit_two_complement() {
        assert_eq!(to_int(&parse("1011").unwrap()), Some(11));
        assert_eq!(to_int(&parse("101").unwrap()), Some(5));
        // 32 ones = -1; the 32-bit high-bit-set value is i32::MIN.
        assert_eq!(to_int(&parse(&"1".repeat(32)).unwrap()), Some(-1));
        let mut min = String::from("1");
        min.push_str(&"0".repeat(31));
        assert_eq!(to_int(&parse(&min).unwrap()), Some(-2_147_483_648));
        assert_eq!(to_int(&[]), Some(0));
        // A string longer than 32 bits cannot fit a signed 32-bit integer.
        assert_eq!(to_int(&parse(&"1".repeat(33)).unwrap()), None);
    }

    #[test]
    fn int_bit_round_trip_at_the_32bit_boundary() {
        for v in [0_i64, 1, -1, 5, 255, -128, 2_147_483_647, -2_147_483_648] {
            let bits = from_int(v, 32);
            assert_eq!(to_int(&bits), Some(v), "round trip for {v}");
        }
    }

    #[test]
    fn bytea_bit_accessors_are_lsb_first_within_a_byte() {
        assert_eq!(bytea_get_bit(&[0x80], 0), Some(false));
        assert_eq!(bytea_get_bit(&[0x80], 7), Some(true));
        assert_eq!(bytea_get_bit(&[0x01], 0), Some(true));
        assert_eq!(bytea_get_bit(&[0x12, 0x34], 3), Some(false));
        assert_eq!(bytea_get_bit(&[0x00, 0x80], 15), Some(true));
        assert_eq!(bytea_get_bit(&[0x80], 8), None); // one past the last bit
        assert_eq!(bytea_set_bit(&[0xff, 0xff], 0, false), vec![0xfe, 0xff]);
        assert_eq!(bytea_set_bit(&[0x12, 0x34], 3, true), vec![0x1a, 0x34]);
    }

    #[test]
    fn ordering_is_left_to_right_then_shorter_first() {
        // Natural Vec<bool> order: compare bit by bit, then a shorter prefix sorts first.
        assert!(parse("10").unwrap() < parse("11").unwrap());
        assert!(parse("10").unwrap() < parse("100").unwrap());
        assert!(parse("0").unwrap() < parse("1").unwrap());
    }
}
