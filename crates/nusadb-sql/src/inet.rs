//! `INET` and `CIDR` — IPv4/IPv6 host addresses and network specifications.
//!
//! Both share one representation, [`InetAddr`]: an [`IpAddr`] plus a network-mask length in bits and
//! a flag for which SQL type produced it. `INET` is a host address that also carries a subnet
//! (`192.168.1.5/24`); `CIDR` is a network whose host bits below the mask must be zero
//! (`192.168.1.0/24`). Dependency-free (only `std::net`) and panic-free: the parsers return `None`
//! on malformed input; the caller raises the typed SQL error.
//!
//! Ordering follows the reference engine's network comparison ([`InetAddr::network_cmp`]): by family
//! (IPv4 before IPv6), then the address bits up to the shorter mask, then the mask length, then the
//! full address. That is **not** a plain byte order, so `ORDER BY` uses `network_cmp` directly rather
//! than an order-preserving index key.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// A parsed `INET`/`CIDR` value: an IP address, a mask length in bits, and whether it came from a
/// `CIDR` column (which renders and validates as a network) or an `INET` column (a host address).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InetAddr {
    /// The host or network address.
    pub addr: IpAddr,
    /// Network mask length in bits: `0..=32` for IPv4, `0..=128` for IPv6.
    pub masklen: u8,
    /// `true` when this value is a `CIDR` (host bits below the mask are zero), `false` for `INET`.
    pub is_cidr: bool,
}

/// The maximum mask length for an address family.
#[must_use]
pub const fn max_masklen(addr: IpAddr) -> u8 {
    match addr {
        IpAddr::V4(_) => 32,
        IpAddr::V6(_) => 128,
    }
}

/// `true` if `addr` is IPv6.
const fn is_v6(addr: IpAddr) -> bool {
    matches!(addr, IpAddr::V6(_))
}

/// The address bytes, big-endian: 4 for IPv4, 16 for IPv6.
#[must_use]
pub fn addr_octets(addr: IpAddr) -> Vec<u8> {
    match addr {
        IpAddr::V4(v4) => v4.octets().to_vec(),
        IpAddr::V6(v6) => v6.octets().to_vec(),
    }
}

impl InetAddr {
    /// Whether every host bit below `masklen` is zero — the `CIDR` validity rule.
    #[must_use]
    pub fn host_bits_zero(&self) -> bool {
        let octets = addr_octets(self.addr);
        let bits = usize::from(self.masklen);
        for (i, &byte) in octets.iter().enumerate() {
            let bit_lo = i * 8;
            if bit_lo >= bits {
                // Whole byte is a host byte → must be zero.
                if byte != 0 {
                    return false;
                }
            } else if bit_lo + 8 > bits {
                // Partial byte: the low `(bit_lo+8 - bits)` bits are host bits.
                let host_bits = (bit_lo + 8) - bits;
                if byte & (0xff_u16 >> (8 - host_bits)) as u8 != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// A stable equality key `[family_tag, address octets…, masklen]` (fixed per family, 5 bytes for
    /// IPv4, 17 for IPv6). Distinct values get distinct keys, so it identifies a value for an equality
    /// probe — but it does **not** reproduce [`InetAddr::network_cmp`] (host bits are compared before
    /// the mask), so it is not an order-preserving key and must not drive a range/`ORDER BY` scan.
    #[must_use]
    pub fn index_key(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(18);
        out.push(u8::from(is_v6(self.addr)));
        out.extend_from_slice(&addr_octets(self.addr));
        out.push(self.masklen);
        out
    }

    /// Serialize for storage/wire: `[flags, masklen, octets…]` where `flags` bit0 = `is_cidr`,
    /// bit1 = IPv6. Self-describing (the family gives the octet count), so decode needs no external length.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let flags = u8::from(self.is_cidr) | (u8::from(is_v6(self.addr)) << 1);
        let mut out = Vec::with_capacity(18);
        out.push(flags);
        out.push(self.masklen);
        out.extend_from_slice(&addr_octets(self.addr));
        out
    }

    /// Inverse of [`InetAddr::encode`]; `None` if the bytes are truncated or the mask is out of range.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let &flags = bytes.first()?;
        let &masklen = bytes.get(1)?;
        let is_cidr = flags & 1 != 0;
        let v6 = flags & 2 != 0;
        let rest = bytes.get(2..)?;
        let addr = if v6 {
            let octets: [u8; 16] = rest.get(0..16)?.try_into().ok()?;
            IpAddr::V6(Ipv6Addr::from(octets))
        } else {
            let octets: [u8; 4] = rest.get(0..4)?.try_into().ok()?;
            IpAddr::V4(Ipv4Addr::from(octets))
        };
        if masklen > max_masklen(addr) {
            return None;
        }
        Some(Self {
            addr,
            masklen,
            is_cidr,
        })
    }

    /// Canonical display form (the reference engine's type-output function). `INET` shows `/masklen`
    /// only when it is not the family maximum (`192.168.1.5`, but `192.168.1.5/24`); `CIDR` always
    /// shows it (`192.168.1.0/24`). This is what a query result / wire text field renders.
    #[must_use]
    pub fn format(&self) -> String {
        if self.is_cidr || self.masklen != max_masklen(self.addr) {
            std::format!("{}/{}", self.addr, self.masklen)
        } else {
            self.addr.to_string()
        }
    }

    /// The `::text` **cast** form, which — matching the reference engine's `inet`→`text` cast, unlike
    /// its display output — always includes `/masklen`, even a family-maximum one (`10.0.0.1/32`).
    #[must_use]
    pub fn format_cast_text(&self) -> String {
        std::format!("{}/{}", self.addr, self.masklen)
    }

    /// Total order matching the reference engine's network comparison: different families order by
    /// family (IPv4 before IPv6); within a family, compare the address bits up to the shorter mask,
    /// then the mask length, then the full address. This is **not** a plain byte order, so it drives
    /// `ORDER BY`/`compare` directly rather than an order-preserving index key.
    #[must_use]
    pub fn network_cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let (fa, fb) = (is_v6(self.addr), is_v6(other.addr));
        if fa != fb {
            return fa.cmp(&fb); // false (IPv4) < true (IPv6)
        }
        let common = self.masklen.min(other.masklen);
        match bitncmp(&addr_octets(self.addr), &addr_octets(other.addr), common) {
            Ordering::Equal => {},
            ord => return ord,
        }
        match self.masklen.cmp(&other.masklen) {
            Ordering::Equal => {},
            ord => return ord,
        }
        addr_octets(self.addr).cmp(&addr_octets(other.addr))
    }

    /// Whether `self`'s network contains (or equals) `other`'s — the basis of `>>=`/`<<=`. Only
    /// meaningful within one family; a cross-family pair is never contained.
    #[must_use]
    pub fn contains_or_equal(&self, other: &Self) -> bool {
        if is_v6(self.addr) != is_v6(other.addr) || self.masklen > other.masklen {
            return false;
        }
        network_prefix_eq(self.addr, other.addr, self.masklen)
    }
}

/// Compare the first `bits` bits of two equal-family octet slices, most-significant first.
fn bitncmp(a: &[u8], b: &[u8], bits: u8) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let whole = usize::from(bits) / 8;
    for (x, y) in a.iter().take(whole).zip(b) {
        match x.cmp(y) {
            Ordering::Equal => {},
            ord => return ord,
        }
    }
    let leftover = bits % 8;
    if leftover != 0 {
        let mask = 0xff_u8 << (8 - leftover);
        let xa = a.get(whole).copied().unwrap_or(0) & mask;
        let yb = b.get(whole).copied().unwrap_or(0) & mask;
        return xa.cmp(&yb);
    }
    Ordering::Equal
}

/// Whether `a` and `b` agree on their first `bits` bits.
fn network_prefix_eq(a: IpAddr, b: IpAddr, bits: u8) -> bool {
    let (oa, ob) = (addr_octets(a), addr_octets(b));
    let mut remaining = usize::from(bits);
    for (x, y) in oa.iter().zip(&ob) {
        if remaining >= 8 {
            if x != y {
                return false;
            }
            remaining -= 8;
        } else if remaining == 0 {
            return true;
        } else {
            let mask = 0xff_u8 << (8 - remaining);
            return x & mask == y & mask;
        }
    }
    true
}

/// Parse an `INET` text: `addr` or `addr/masklen`. A bare address defaults to the family's full mask.
/// Host bits may be set (unlike `CIDR`).
#[must_use]
pub fn parse_inet(s: &str) -> Option<InetAddr> {
    parse_common(s.trim(), false)
}

/// Parse a `CIDR` text: `addr` or `addr/masklen`, rejecting a value whose host bits below the mask
/// are non-zero (a network address must be canonical). A bare address defaults to the family's full
/// mask.
#[must_use]
pub fn parse_cidr(s: &str) -> Option<InetAddr> {
    let v = parse_common(s.trim(), true)?;
    v.host_bits_zero().then_some(v)
}

fn parse_common(s: &str, is_cidr: bool) -> Option<InetAddr> {
    let (addr_str, mask_str) = match s.split_once('/') {
        Some((a, m)) => (a, Some(m)),
        None => (s, None),
    };
    let addr: IpAddr = addr_str.parse().ok()?;
    let masklen = match mask_str {
        Some(m) => {
            let bits: u8 = m.parse().ok()?;
            if bits > max_masklen(addr) {
                return None;
            }
            bits
        },
        None => max_masklen(addr),
    };
    Some(InetAddr {
        addr,
        masklen,
        is_cidr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format_inet() {
        let v = parse_inet("192.168.1.5/24").unwrap();
        assert_eq!(v.masklen, 24);
        assert!(!v.is_cidr);
        assert_eq!(v.format(), "192.168.1.5/24");
        // A bare host address renders without the /32.
        assert_eq!(parse_inet("10.0.0.1").unwrap().format(), "10.0.0.1");
        // IPv6.
        assert_eq!(parse_inet("::1").unwrap().format(), "::1");
        assert_eq!(
            parse_inet("2001:db8::/32").unwrap().format(),
            "2001:db8::/32"
        );
    }

    #[test]
    fn parse_and_format_cidr() {
        // CIDR always shows the mask.
        assert_eq!(
            parse_cidr("192.168.1.0/24").unwrap().format(),
            "192.168.1.0/24"
        );
        assert_eq!(parse_cidr("10.0.0.0/8").unwrap().format(), "10.0.0.0/8");
        // A CIDR with host bits set is rejected.
        assert_eq!(parse_cidr("192.168.1.5/24"), None);
        // INET accepts the same host bits.
        assert!(parse_inet("192.168.1.5/24").is_some());
    }

    #[test]
    fn parse_rejects_malformed() {
        assert_eq!(parse_inet("not-an-ip"), None);
        assert_eq!(parse_inet("192.168.1.5/33"), None); // mask too large for IPv4
        assert_eq!(parse_inet("::1/129"), None); // mask too large for IPv6
        assert_eq!(parse_inet("256.0.0.1"), None);
    }

    #[test]
    fn containment() {
        let net = parse_cidr("192.168.1.0/24").unwrap();
        let host = parse_inet("192.168.1.5/32").unwrap();
        assert!(net.contains_or_equal(&host));
        assert!(!host.contains_or_equal(&net));
        // Different networks do not contain each other.
        let other = parse_inet("10.0.0.1/32").unwrap();
        assert!(!net.contains_or_equal(&other));
        // Cross-family is never contained.
        let v6 = parse_inet("::1/128").unwrap();
        assert!(!net.contains_or_equal(&v6));
    }

    #[test]
    fn encode_decode_round_trip() {
        for s in ["192.168.1.5/24", "10.0.0.1", "2001:db8::1/64", "::1"] {
            let v = parse_inet(s).unwrap();
            assert_eq!(InetAddr::decode(&v.encode()), Some(v));
        }
    }

    #[test]
    fn network_cmp_matches_reference_order() {
        use std::cmp::Ordering;
        let c = |a: &str, b: &str| parse_inet(a).unwrap().network_cmp(&parse_inet(b).unwrap());
        // IPv4 sorts before IPv6.
        assert_eq!(c("255.255.255.255", "::"), Ordering::Less);
        // Same family, by address then mask.
        assert_eq!(c("10.0.0.0/8", "192.168.1.1"), Ordering::Less);
        assert_eq!(c("10.0.0.0/8", "255.255.255.255"), Ordering::Less);
        // Same leading bits: the shorter mask sorts first (network before its sub-range).
        assert_eq!(c("10.0.0.0/8", "10.0.0.0/16"), Ordering::Less);
        // Host bits are compared only after the mask: 10.1.0.0/8 shares the /8 prefix with
        // 10.0.0.0/16, but the shorter mask still wins.
        assert_eq!(c("10.1.0.0/8", "10.0.0.0/16"), Ordering::Less);
        assert_eq!(c("10.0.0.1", "10.0.0.1"), Ordering::Equal);
    }

    #[test]
    fn format_cast_text_always_shows_mask() {
        // Display omits the /32; the ::text cast keeps it (reference-engine quirk).
        assert_eq!(parse_inet("10.0.0.1").unwrap().format(), "10.0.0.1");
        assert_eq!(
            parse_inet("10.0.0.1").unwrap().format_cast_text(),
            "10.0.0.1/32"
        );
        assert_eq!(parse_inet("::1").unwrap().format_cast_text(), "::1/128");
    }

    #[test]
    fn index_key_orders_family_then_addr_then_mask() {
        let v4 = parse_inet("255.255.255.255/32").unwrap();
        let v6 = parse_inet("::/0").unwrap();
        // Any IPv4 key sorts before any IPv6 key (family tag 0 < 1).
        assert!(v4.index_key() < v6.index_key());
        let a = parse_cidr("10.0.0.0/8").unwrap();
        let b = parse_cidr("10.1.0.0/16").unwrap();
        assert!(a.index_key() < b.index_key());
    }
}
