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
    fn ordering_is_left_to_right_then_shorter_first() {
        // Natural Vec<bool> order: compare bit by bit, then a shorter prefix sorts first.
        assert!(parse("10").unwrap() < parse("11").unwrap());
        assert!(parse("10").unwrap() < parse("100").unwrap());
        assert!(parse("0").unwrap() < parse("1").unwrap());
    }
}
