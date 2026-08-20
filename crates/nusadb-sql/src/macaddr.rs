//! MAC address (`MACADDR`) — six bytes in transmission order, with textual parsing and canonical
//! formatting.
//!
//! Dependency-free and panic-free: [`parse`] returns `None` for malformed input; [`format()`]
//! renders the canonical lowercase colon-separated form `08:00:2b:01:02:03`. Values are compared and ordered
//! by the six bytes as an unsigned big-endian integer (the same order the bytes already impose), so
//! the storage bytes double as the order-preserving index key.

/// Parse a MAC-address text into its six bytes, or `None` if it is not twelve hex digits once the
/// conventional group separators are removed.
///
/// Accepts the common forms — `08:00:2b:01:02:03`, `08-00-2b-01-02-03`, `0800.2b01.0203` (Cisco),
/// `08002b:010203`, and bare `08002b010203` — case-insensitively; the separators `:`/`-`/`.` are
/// stripped and exactly twelve hex digits must remain.
#[must_use]
pub fn parse(s: &str) -> Option<[u8; 6]> {
    let mut out = [0u8; 6];
    let mut filled = 0usize; // full bytes written into `out`
    let mut pending: Option<u8> = None; // a high nibble awaiting its low nibble
    for c in s.trim().chars() {
        match c {
            ':' | '-' | '.' => {},
            _ => {
                let nibble = u8::try_from(c.to_digit(16)?).ok()?;
                match pending.take() {
                    None => pending = Some(nibble),
                    Some(high) => {
                        // A completed byte — reject a seventh (too many hex digits).
                        let slot = out.get_mut(filled)?;
                        *slot = (high << 4) | nibble;
                        filled += 1;
                    },
                }
            },
        }
    }
    // Exactly six bytes and no dangling half-byte.
    (filled == 6 && pending.is_none()).then_some(out)
}

/// Render the canonical text form `08:00:2b:01:02:03` — lowercase, colon-separated.
#[must_use]
pub fn format(m: [u8; 6]) -> String {
    let mut out = String::with_capacity(17);
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

/// The byte-wise complement of an address (the `~` operator).
#[must_use]
pub const fn complement(m: [u8; 6]) -> [u8; 6] {
    [!m[0], !m[1], !m[2], !m[3], !m[4], !m[5]]
}

/// The byte-wise AND of two addresses (the `&` operator).
#[must_use]
pub const fn and(a: [u8; 6], b: [u8; 6]) -> [u8; 6] {
    [
        a[0] & b[0],
        a[1] & b[1],
        a[2] & b[2],
        a[3] & b[3],
        a[4] & b[4],
        a[5] & b[5],
    ]
}

/// The byte-wise OR of two addresses (the `|` operator).
#[must_use]
pub const fn or(a: [u8; 6], b: [u8; 6]) -> [u8; 6] {
    [
        a[0] | b[0],
        a[1] | b[1],
        a[2] | b[2],
        a[3] | b[3],
        a[4] | b[4],
        a[5] | b[5],
    ]
}

/// `trunc(macaddr)` — the address with its last three bytes (the device identifier) zeroed, leaving
/// the first three (the manufacturer prefix).
#[must_use]
pub const fn trunc(m: [u8; 6]) -> [u8; 6] {
    [m[0], m[1], m[2], 0, 0, 0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_the_common_forms() {
        let want = [0x08, 0x00, 0x2b, 0x01, 0x02, 0x03];
        for s in [
            "08:00:2b:01:02:03",
            "08-00-2b-01-02-03",
            "0800.2b01.0203",
            "08002b:010203",
            "08002b-010203",
            "08002b010203",
            "08:00:2B:01:02:03", // upper-case
        ] {
            assert_eq!(parse(s), Some(want), "form {s}");
        }
    }

    #[test]
    fn parse_rejects_malformed() {
        assert_eq!(parse("08:00:2b:01:02"), None); // ten digits
        assert_eq!(parse("08:00:2b:01:02:03:04"), None); // fourteen digits
        assert_eq!(parse("gg:00:2b:01:02:03"), None); // non-hex
        assert_eq!(parse(""), None);
    }

    #[test]
    fn format_is_canonical_lowercase_colon() {
        assert_eq!(
            format([0x08, 0x00, 0x2b, 0x01, 0x02, 0x03]),
            "08:00:2b:01:02:03"
        );
        assert_eq!(
            format([0xff, 0x00, 0xab, 0xcd, 0xef, 0x10]),
            "ff:00:ab:cd:ef:10"
        );
    }

    #[test]
    fn round_trip() {
        for m in [[0u8; 6], [0xff; 6], [0x01, 0x23, 0x45, 0x67, 0x89, 0xab]] {
            assert_eq!(parse(&format(m)), Some(m));
        }
    }

    #[test]
    fn bitwise_and_trunc() {
        let m = parse("08:00:2b:01:02:03").unwrap();
        assert_eq!(format(complement(m)), "f7:ff:d4:fe:fd:fc");
        assert_eq!(
            format(and(m, parse("ff:ff:ff:00:00:00").unwrap())),
            "08:00:2b:00:00:00"
        );
        assert_eq!(
            format(or(
                parse("08:00:2b:00:00:00").unwrap(),
                parse("00:00:00:ff:ff:ff").unwrap()
            )),
            "08:00:2b:ff:ff:ff"
        );
        assert_eq!(format(trunc(m)), "08:00:2b:00:00:00");
        // Double complement is the identity.
        assert_eq!(complement(complement(m)), m);
    }
}
