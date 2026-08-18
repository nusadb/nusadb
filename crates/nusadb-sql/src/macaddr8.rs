//! 8-byte MAC address (`MACADDR8`, EUI-64) — eight bytes in transmission order, with textual parsing
//! and canonical formatting.
//!
//! Dependency-free and panic-free: [`parse`] returns `None` for malformed input; [`format()`]
//! renders the canonical lowercase colon-separated form `08:00:2b:01:02:03:04:05`. Values are
//! compared and ordered by the eight bytes as an unsigned big-endian integer (the same order the
//! bytes already impose), so the storage bytes double as the order-preserving index key.
//!
//! A six-byte EUI-48 input is accepted and widened to EUI-64 by inserting `ff:fe` after the third
//! byte (`08:00:2b:01:02:03` → `08:00:2b:ff:fe:01:02:03`), so a `MACADDR` value or its text form can
//! be stored as a `MACADDR8`.

/// Parse a MAC-address text into its eight bytes, or `None` if it is not a valid EUI-48 (twelve hex
/// digits) or EUI-64 (sixteen hex digits) once the conventional group separators are removed.
///
/// Accepts the common forms — colon (`08:00:2b:01:02:03:04:05`), dash
/// (`08-00-2b-01-02-03-04-05`), and dot-quad (`0800.2b01.0203.0405`) — case-insensitively; the
/// separators `:`/`-`/`.` are stripped and exactly twelve or sixteen hex digits must remain. A
/// twelve-digit (EUI-48) value is widened to EUI-64 by inserting `ff:fe` after the third byte.
#[must_use]
pub fn parse(s: &str) -> Option<[u8; 8]> {
    let mut buf = [0u8; 8];
    let mut filled = 0usize; // full bytes written into `buf`
    let mut pending: Option<u8> = None; // a high nibble awaiting its low nibble
    for c in s.trim().chars() {
        match c {
            ':' | '-' | '.' => {},
            _ => {
                let nibble = u8::try_from(c.to_digit(16)?).ok()?;
                match pending.take() {
                    None => pending = Some(nibble),
                    Some(high) => {
                        // A completed byte — reject a ninth (too many hex digits).
                        let slot = buf.get_mut(filled)?;
                        *slot = (high << 4) | nibble;
                        filled += 1;
                    },
                }
            },
        }
    }
    if pending.is_some() {
        return None; // a dangling half-byte
    }
    match filled {
        // EUI-64 — already eight bytes.
        8 => Some(buf),
        // EUI-48 → EUI-64: insert `ff:fe` after the third byte.
        6 => Some([buf[0], buf[1], buf[2], 0xff, 0xfe, buf[3], buf[4], buf[5]]),
        _ => None,
    }
}

/// Render the canonical text form `08:00:2b:01:02:03:04:05` — lowercase, colon-separated.
#[must_use]
pub fn format(m: [u8; 8]) -> String {
    let mut out = String::with_capacity(23);
    for (i, b) in m.iter().enumerate() {
        if i > 0 {
            out.push(':');
        }
        // Two lowercase hex digits per byte.
        out.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    out
}

/// Widen a six-byte EUI-48 (`MACADDR`) to its eight-byte EUI-64 (`MACADDR8`) by inserting `ff:fe`
/// after the third byte — the same expansion [`parse`] applies to a twelve-digit input.
#[must_use]
pub const fn from_eui48(m: [u8; 6]) -> [u8; 8] {
    [m[0], m[1], m[2], 0xff, 0xfe, m[3], m[4], m[5]]
}

/// The bitwise complement of all eight bytes (`~` operator).
#[must_use]
pub const fn complement(m: [u8; 8]) -> [u8; 8] {
    [!m[0], !m[1], !m[2], !m[3], !m[4], !m[5], !m[6], !m[7]]
}

/// The byte-wise AND of two addresses (`&` operator).
#[must_use]
pub const fn and(a: [u8; 8], b: [u8; 8]) -> [u8; 8] {
    [
        a[0] & b[0],
        a[1] & b[1],
        a[2] & b[2],
        a[3] & b[3],
        a[4] & b[4],
        a[5] & b[5],
        a[6] & b[6],
        a[7] & b[7],
    ]
}

/// The byte-wise OR of two addresses (`|` operator).
#[must_use]
pub const fn or(a: [u8; 8], b: [u8; 8]) -> [u8; 8] {
    [
        a[0] | b[0],
        a[1] | b[1],
        a[2] | b[2],
        a[3] | b[3],
        a[4] | b[4],
        a[5] | b[5],
        a[6] | b[6],
        a[7] | b[7],
    ]
}

/// `trunc(macaddr8)` — keep the first three bytes (the OUI) and zero the last five.
#[must_use]
pub const fn trunc(m: [u8; 8]) -> [u8; 8] {
    [m[0], m[1], m[2], 0, 0, 0, 0, 0]
}

/// `macaddr8_set7bit(macaddr8)` — set the 7th bit (the `0x02` locally-administered bit) of the first
/// byte, returning the modified EUI-64.
#[must_use]
pub const fn set7bit(m: [u8; 8]) -> [u8; 8] {
    [m[0] | 0x02, m[1], m[2], m[3], m[4], m[5], m[6], m[7]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_common_eui64_forms() {
        let want = [0x08, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05];
        for s in [
            "08:00:2b:01:02:03:04:05",
            "08-00-2b-01-02-03-04-05",
            "0800.2b01.0203.0405",
            "08:00:2B:01:02:03:04:05", // upper-case
        ] {
            assert_eq!(parse(s), Some(want), "form {s}");
        }
    }

    #[test]
    fn parse_widens_eui48_to_eui64() {
        assert_eq!(
            parse("08:00:2b:01:02:03"),
            Some([0x08, 0x00, 0x2b, 0xff, 0xfe, 0x01, 0x02, 0x03])
        );
    }

    #[test]
    fn parse_rejects_malformed() {
        assert_eq!(parse("08:00:2b:01:02:03:04"), None); // fourteen digits
        assert_eq!(parse("08:00:2b:01:02:03:04:05:06"), None); // eighteen digits
        assert_eq!(parse("zz"), None); // non-hex
        assert_eq!(parse(""), None);
    }

    #[test]
    fn format_is_canonical_lowercase_colon() {
        assert_eq!(
            format([0x08, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05]),
            "08:00:2b:01:02:03:04:05"
        );
    }

    #[test]
    fn round_trip() {
        for m in [
            [0u8; 8],
            [0xff; 8],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef],
        ] {
            assert_eq!(parse(&format(m)), Some(m));
        }
    }

    #[test]
    fn bitwise_matches_reference() {
        let a = [0x08, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(
            complement(a),
            [0xf7, 0xff, 0xd4, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa]
        );
        let mask = [0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            and(a, mask),
            [0x08, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn trunc_and_set7bit() {
        let a = [0x08, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(trunc(a), [0x08, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let b = [0x00, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(set7bit(b), [0x02, 0x00, 0x2b, 0x01, 0x02, 0x03, 0x04, 0x05]);
    }
}
