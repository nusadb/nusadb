//! Minimal RFC 4648 base64 (standard alphabet, with `=` padding).
//!
//! SCRAM-SHA-256 (RFC 5802) base64-encodes the salt, the client/server proofs, and the
//! server signature. Rather than pull in a dependency for a ~70-line codec, the wire crate
//! carries its own. Standard alphabet only (`A–Z a–z 0–9 + /`); decoding is strict — it
//! rejects non-alphabet bytes, a length that is not a multiple of four, and padding outside
//! the final block.

use super::scram::ScramError;

/// Map a 6-bit value (`0..=63`) to its base64 alphabet byte. Callers always mask to 6 bits,
/// so the final arm only ever sees `63`.
const fn encode_sextet(v: u8) -> u8 {
    match v {
        0..=25 => b'A' + v,
        26..=51 => b'a' + (v - 26),
        52..=61 => b'0' + (v - 52),
        62 => b'+',
        _ => b'/',
    }
}

/// Map a base64 alphabet byte back to its 6-bit value, or `None` if it is not in the alphabet.
const fn decode_sextet(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Encode `input` as standard base64 with `=` padding.
pub(super) fn encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut it = input.iter().copied();
    while let Some(b0) = it.next() {
        let b1 = it.next();
        let b2 = it.next();
        out.push(encode_sextet(b0 >> 2) as char);
        out.push(encode_sextet(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as char);
        match b1 {
            Some(v) => out.push(encode_sextet(((v & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as char),
            None => out.push('='),
        }
        match b2 {
            Some(v) => out.push(encode_sextet(v & 0x3f) as char),
            None => out.push('='),
        }
    }
    out
}

/// Decode standard base64 with `=` padding. Strict: a non-alphabet byte, a length that is not
/// a multiple of four, or padding anywhere but the final block all yield [`ScramError::Malformed`].
pub(super) fn decode(input: &str) -> Result<Vec<u8>, ScramError> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(ScramError::Malformed);
    }
    let block_count = bytes.len() / 4;
    let mut out = Vec::with_capacity(block_count * 3);
    for (idx, chunk) in bytes.chunks_exact(4).enumerate() {
        let [c0, c1, c2, c3] = chunk else {
            return Err(ScramError::Malformed);
        };
        let is_last = idx + 1 == block_count;
        let v0 = decode_sextet(*c0).ok_or(ScramError::Malformed)?;
        let v1 = decode_sextet(*c1).ok_or(ScramError::Malformed)?;
        out.push((v0 << 2) | (v1 >> 4));
        match (*c2, *c3) {
            // Two pad bytes: one decoded output byte (e.g. "Zg==" -> 1 byte).
            (b'=', b'=') if is_last => {},
            // One pad byte: two decoded output bytes (e.g. "Zm8=" -> 2 bytes).
            (_, b'=') if is_last => {
                let v2 = decode_sextet(*c2).ok_or(ScramError::Malformed)?;
                out.push(((v1 & 0x0f) << 4) | (v2 >> 2));
            },
            // No padding: three decoded output bytes.
            (b'=', _) => return Err(ScramError::Malformed),
            (_, _) => {
                let v2 = decode_sextet(*c2).ok_or(ScramError::Malformed)?;
                let v3 = decode_sextet(*c3).ok_or(ScramError::Malformed)?;
                out.push(((v1 & 0x0f) << 4) | (v2 >> 2));
                out.push(((v2 & 0x03) << 6) | v3);
            },
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    // RFC 4648 §10 test vectors.
    const VECTORS: &[(&[u8], &str)] = &[
        (b"", ""),
        (b"f", "Zg=="),
        (b"fo", "Zm8="),
        (b"foo", "Zm9v"),
        (b"foob", "Zm9vYg=="),
        (b"fooba", "Zm9vYmE="),
        (b"foobar", "Zm9vYmFy"),
    ];

    #[test]
    fn encode_matches_rfc4648_vectors() {
        for (raw, b64) in VECTORS {
            assert_eq!(encode(raw), *b64, "encode({raw:?})");
        }
    }

    #[test]
    fn decode_matches_rfc4648_vectors() {
        for (raw, b64) in VECTORS {
            assert_eq!(decode(b64).unwrap(), *raw, "decode({b64:?})");
        }
    }

    #[test]
    fn round_trips_arbitrary_bytes() {
        for len in 0..=64usize {
            let raw: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(decode(&encode(&raw)).unwrap(), raw, "len {len}");
        }
    }

    #[test]
    fn rejects_bad_length() {
        for bad in ["a", "ab", "abc", "abcde"] {
            assert!(decode(bad).is_err(), "len {} should be rejected", bad.len());
        }
    }

    #[test]
    fn rejects_non_alphabet_byte() {
        assert!(decode("Zm9 v").is_err());
        assert!(decode("Zm9*").is_err());
        assert!(decode("....").is_err());
    }

    #[test]
    fn rejects_misplaced_padding() {
        // Padding only legal in the final block, and `=` in c2 requires `=` in c3.
        assert!(decode("Zg==Zg==").is_err());
        assert!(decode("Z=9v").is_err());
        assert!(decode("=m9v").is_err());
    }
}
