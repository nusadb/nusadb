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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// The address as a big-endian unsigned integer (0..2^32 for IPv4, 0..2^128 for IPv6). IPv4 values
/// occupy only the low 32 bits, so the two families share one `u128` representation for arithmetic.
fn addr_to_u128(addr: IpAddr) -> u128 {
    addr_octets(addr)
        .iter()
        .fold(0u128, |acc, &byte| (acc << 8) | u128::from(byte))
}

impl InetAddr {
    /// The SQL column type this value belongs to: `CIDR` when `is_cidr`, else `INET`.
    #[must_use]
    pub const fn column_type(&self) -> nusadb_core::ColumnType {
        if self.is_cidr {
            nusadb_core::ColumnType::Cidr
        } else {
            nusadb_core::ColumnType::Inet
        }
    }

    /// This value as a `CIDR` network — host bits below the mask zeroed and `is_cidr` set. Matches
    /// the reference engine's `inet`→`cidr` cast (which masks rather than rejecting host bits).
    #[must_use]
    pub fn to_cidr(self) -> Self {
        Self {
            addr: self.masked_addr(false),
            masklen: self.masklen,
            is_cidr: true,
        }
    }

    /// The broadcast address of this value's network — every host bit set — as an `INET`. `host()`
    /// and `set_masklen` companions of the reference engine's `broadcast()` function.
    #[must_use]
    pub fn broadcast(self) -> Self {
        Self {
            addr: self.masked_addr(true),
            masklen: self.masklen,
            is_cidr: false,
        }
    }

    /// The same address with a new mask length, or `None` if `masklen` exceeds the family maximum.
    #[must_use]
    pub fn with_masklen(self, masklen: u8) -> Option<Self> {
        (masklen <= max_masklen(self.addr)).then_some(Self { masklen, ..self })
    }

    /// The host address alone, without any mask (`192.168.1.5`) — the reference engine's `host()`.
    #[must_use]
    pub fn host(self) -> String {
        self.addr.to_string()
    }

    /// The address family as the reference engine reports it: `4` for IPv4, `6` for IPv6.
    #[must_use]
    pub const fn family(self) -> i64 {
        if is_v6(self.addr) { 6 } else { 4 }
    }

    /// This address with the host bits below the mask either zeroed (`set` = false → network) or set
    /// (`set` = true → broadcast).
    fn masked_addr(self, set: bool) -> IpAddr {
        let mut octets = addr_octets(self.addr);
        let bits = usize::from(self.masklen);
        for (i, byte) in octets.iter_mut().enumerate() {
            let bit_lo = i * 8;
            let host_mask: u8 = if bit_lo >= bits {
                0xff
            } else if bit_lo + 8 > bits {
                ((1u16 << ((bit_lo + 8) - bits)) - 1) as u8
            } else {
                0
            };
            if set {
                *byte |= host_mask;
            } else {
                *byte &= !host_mask;
            }
        }
        octets_to_addr(&octets, is_v6(self.addr))
    }

    /// This value as a host address — `is_cidr` cleared, address and mask unchanged. The `cidr`→`inet` cast.
    #[must_use]
    pub const fn to_inet(self) -> Self {
        Self {
            is_cidr: false,
            addr: self.addr,
            masklen: self.masklen,
        }
    }

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

    /// This address offset by `delta` (added, or subtracted when negative), keeping the mask length,
    /// family and `is_cidr` flag. The address is treated as a big unsigned integer (32-bit for IPv4,
    /// 128-bit for IPv6) with carry/borrow across octets. Returns `None` when the result would leave
    /// the family's range — below the all-zero address or above the all-ones address — which the
    /// caller raises as the loud `inet ± int` out-of-range error. Total and panic-free.
    #[must_use]
    pub fn add_offset(self, delta: i64) -> Option<Self> {
        let cur = addr_to_u128(self.addr);
        let magnitude = u128::from(delta.unsigned_abs());
        let next = if delta >= 0 {
            cur.checked_add(magnitude)?
        } else {
            cur.checked_sub(magnitude)?
        };
        let addr = if is_v6(self.addr) {
            IpAddr::V6(Ipv6Addr::from(next))
        } else {
            IpAddr::V4(Ipv4Addr::from(u32::try_from(next).ok()?))
        };
        Some(Self { addr, ..self })
    }

    /// The signed difference `self - other` of the two addresses as integers (each a big unsigned
    /// integer per [`InetAddr::add_offset`]) — the basis of `inet - inet`. Returns `None` when the two
    /// are different families or the difference does not fit `i64`, both of which the caller raises as
    /// a loud error. Total and panic-free.
    #[must_use]
    pub fn diff(&self, other: &Self) -> Option<i64> {
        if is_v6(self.addr) != is_v6(other.addr) {
            return None;
        }
        let (a, b) = (addr_to_u128(self.addr), addr_to_u128(other.addr));
        let (magnitude, negative) = if a >= b {
            (a - b, false)
        } else {
            (b - a, true)
        };
        let signed = i64::try_from(magnitude).ok()?;
        Some(if negative { -signed } else { signed })
    }

    /// The network mask of this value's subnet: the first `masklen` bits set, the rest zero, as an
    /// `INET` whose own mask is the family maximum (so it renders as a bare address — `255.255.255.0`
    /// for `/24`). The `netmask()` function.
    #[must_use]
    pub fn netmask(self) -> Self {
        self.mask_value(true)
    }

    /// The host mask — the inverse of [`InetAddr::netmask`]: the first `masklen` bits zero, the rest
    /// set (`0.0.0.255` for `/24`). The `hostmask()` function.
    #[must_use]
    pub fn hostmask(self) -> Self {
        self.mask_value(false)
    }

    /// Build a mask value as a bare `INET`: `ones_in_prefix` sets the first `masklen` bits and clears
    /// the rest (netmask); `false` clears the first `masklen` bits and sets the rest (hostmask).
    fn mask_value(self, ones_in_prefix: bool) -> Self {
        let len = addr_octets(self.addr).len();
        let octets = mask_octets(len, usize::from(self.masklen), ones_in_prefix);
        Self {
            addr: octets_to_addr(&octets, is_v6(self.addr)),
            masklen: max_masklen(self.addr),
            is_cidr: false,
        }
    }

    /// The abbreviated text form. For an `INET` value it is the display form ([`InetAddr::format`] —
    /// the mask shown unless it is the family maximum). For a `CIDR` value it additionally drops the
    /// trailing all-zero octets/groups the mask does not need (`10.1.0.0/16` → `10.1/16`).
    #[must_use]
    pub fn abbrev(&self) -> String {
        if !self.is_cidr {
            return self.format();
        }
        let octets = addr_octets(self.addr);
        let body = if is_v6(self.addr) {
            abbrev_cidr_v6(&octets, self.masklen)
        } else {
            abbrev_cidr_v4(&octets, self.masklen)
        };
        std::format!("{body}/{}", self.masklen)
    }

    /// `true` when both values are the same address family (both IPv4 or both IPv6). Never errors on
    /// a mismatch — it reports `false`. The `inet_same_family()` function.
    #[must_use]
    pub const fn same_family(&self, other: &Self) -> bool {
        is_v6(self.addr) == is_v6(other.addr)
    }

    /// The smallest network (`CIDR`) that contains both `self` and `other`: the longest common bit
    /// prefix of the two addresses — examined only up to the shorter of the two masks — with the host
    /// bits below it zeroed. `None` when the two are different families (the caller raises the typed
    /// "cannot merge addresses from different families" error). The `inet_merge()` function.
    #[must_use]
    pub fn merge(self, other: Self) -> Option<Self> {
        if is_v6(self.addr) != is_v6(other.addr) {
            return None;
        }
        let oa = addr_octets(self.addr);
        let ob = addr_octets(other.addr);
        let limit = self.masklen.min(other.masklen);
        let common = common_prefix_len(&oa, &ob, limit);
        // Zero the host bits below the common prefix and stamp it as a CIDR network.
        Some(
            Self {
                addr: self.addr,
                masklen: common,
                is_cidr: false,
            }
            .to_cidr(),
        )
    }
}

/// Octets of a contiguous mask of `len` bytes: `ones_in_prefix` sets the first `bits` bits and clears
/// the rest; `false` clears the first `bits` bits and sets the rest.
fn mask_octets(len: usize, bits: usize, ones_in_prefix: bool) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let bit_lo = i * 8;
            let prefix: u8 = if bit_lo + 8 <= bits {
                0xff
            } else if bit_lo >= bits {
                0x00
            } else {
                // The high `bits - bit_lo` bits of this byte are in the prefix.
                let n = bits - bit_lo;
                (0xffu16 << (8 - n)) as u8
            };
            if ones_in_prefix { prefix } else { !prefix }
        })
        .collect()
}

/// The number of leading bits `a` and `b` share, examining at most `maxbits` bits.
fn common_prefix_len(a: &[u8], b: &[u8], maxbits: u8) -> u8 {
    let mut n = 0u8;
    while n < maxbits {
        let idx = usize::from(n / 8);
        let mask = 0x80u8 >> (n % 8);
        let xa = a.get(idx).copied().unwrap_or(0) & mask;
        let yb = b.get(idx).copied().unwrap_or(0) & mask;
        if xa != yb {
            break;
        }
        n += 1;
    }
    n
}

/// Abbreviate an IPv4 `CIDR` network body (without the `/mask` suffix): keep only the octets the mask
/// reaches (`10.1.0.0/16` → `10.1`), always at least one (`0.0.0.0/0` → `0`). Ports the reference
/// engine's network-notation output.
fn abbrev_cidr_v4(octets: &[u8], bits: u8) -> String {
    let mut out = String::new();
    if bits == 0 {
        out.push('0');
    }
    let whole = bits / 8;
    for i in 0..whole {
        if !out.is_empty() {
            out.push('.');
        }
        out.push_str(&octets.get(usize::from(i)).copied().unwrap_or(0).to_string());
    }
    let partial = bits % 8;
    if partial > 0 {
        if !out.is_empty() {
            out.push('.');
        }
        let m = (((1u16 << partial) - 1) << (8 - partial)) as u8;
        let val = octets.get(usize::from(whole)).copied().unwrap_or(0) & m;
        out.push_str(&val.to_string());
    }
    out
}

/// Abbreviate an IPv6 `CIDR` network body (without the `/mask` suffix), a faithful port of the
/// reference engine's network-notation output: emit only the 16-bit groups the mask reaches, with the
/// longest run of zero groups collapsed to `::`. A single-group mask is padded to two so a `/16`
/// still shows the `::` (`ffff::/16`).
fn abbrev_cidr_v6(src: &[u8], bits: u8) -> String {
    use std::fmt::Write as _;
    if bits == 0 {
        return "::".to_string();
    }
    // Copy the network bytes into a fixed buffer, zero the host part, and mask the partial last byte.
    let p = usize::from(bits).div_ceil(8);
    let mut buf = [0u8; 16];
    for (i, slot) in buf.iter_mut().enumerate().take(p.min(16)) {
        *slot = src.get(i).copied().unwrap_or(0);
    }
    let rem = bits % 8;
    if rem != 0 && (1..=16).contains(&p) {
        let m = 0xffu8 << (8 - rem);
        if let Some(byte) = buf.get_mut(p - 1) {
            *byte &= m;
        }
    }
    // How many 16-bit words the mask reaches; a lone word is padded to two.
    let mut words = usize::from(bits).div_ceil(16);
    if words == 1 {
        words = 2;
    }
    // Longest run of zero words among the first `words` — faithful to the reference (which does not
    // reset its running tally when a non-zero word is no longer than the best run so far).
    let (mut zero_s, mut zero_l) = (0usize, 0usize);
    let (mut tmp_s, mut tmp_l) = (0usize, 0usize);
    for i in 0..words {
        let hi = buf.get(2 * i).copied().unwrap_or(0);
        let lo = buf.get(2 * i + 1).copied().unwrap_or(0);
        if hi | lo == 0 {
            if tmp_l == 0 {
                tmp_s = i;
            }
            tmp_l += 1;
        } else if tmp_l != 0 && zero_l < tmp_l {
            zero_s = tmp_s;
            zero_l = tmp_l;
            tmp_l = 0;
        }
    }
    if tmp_l != 0 && zero_l < tmp_l {
        zero_s = tmp_s;
        zero_l = tmp_l;
    }
    // IPv4-in-IPv6 detection, matching the reference engine's dotted-quad tail rendering.
    let is_ipv4 = zero_l != words
        && zero_s == 0
        && (zero_l == 6
            || (zero_l == 5 && buf.get(10) == Some(&0xff) && buf.get(11) == Some(&0xff))
            || (zero_l == 7 && buf.get(14) != Some(&0) && buf.get(15) != Some(&1)));

    let mut out = String::new();
    let mut s = 0usize;
    for word in 0..words {
        if zero_l != 0 && word >= zero_s && word < zero_s + zero_l {
            if word == zero_s {
                out.push(':');
            }
            if word == words - 1 {
                out.push(':');
            }
            s += 2;
            continue;
        }
        if is_ipv4 && word > 5 {
            out.push(if word == 6 { ':' } else { '.' });
            out.push_str(&buf.get(s).copied().unwrap_or(0).to_string());
            s += 1;
            if word != 7 || bits > 120 {
                out.push('.');
                out.push_str(&buf.get(s).copied().unwrap_or(0).to_string());
                s += 1;
            }
        } else {
            if !out.is_empty() {
                out.push(':');
            }
            let hi = u16::from(buf.get(s).copied().unwrap_or(0));
            let lo = u16::from(buf.get(s + 1).copied().unwrap_or(0));
            let _ = write!(out, "{:x}", hi * 256 + lo);
            s += 2;
        }
    }
    out
}

/// Rebuild an [`IpAddr`] from big-endian octets (4 for IPv4 when `v6` is false, 16 for IPv6).
fn octets_to_addr(octets: &[u8], v6: bool) -> IpAddr {
    if v6 {
        let a: [u8; 16] = octets.try_into().unwrap_or([0; 16]);
        IpAddr::V6(Ipv6Addr::from(a))
    } else {
        let a: [u8; 4] = octets.try_into().unwrap_or([0; 4]);
        IpAddr::V4(Ipv4Addr::from(a))
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
    fn netmask_builds_prefix_ones() {
        let nm = |s: &str| parse_inet(s).unwrap().netmask().format();
        assert_eq!(nm("192.168.1.5/24"), "255.255.255.0");
        assert_eq!(nm("192.168.1.5"), "255.255.255.255");
        assert_eq!(nm("10.1.2.3/16"), "255.255.0.0");
        assert_eq!(nm("10.1.2.3/0"), "0.0.0.0");
        assert_eq!(nm("192.168.1.5/30"), "255.255.255.252");
        assert_eq!(nm("2001:db8::1/64"), "ffff:ffff:ffff:ffff::");
        assert_eq!(nm("2001:db8::1/48"), "ffff:ffff:ffff::");
        assert_eq!(
            nm("2001:db8::1/128"),
            "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"
        );
        assert_eq!(nm("2001:db8::1/0"), "::");
    }

    #[test]
    fn hostmask_builds_suffix_ones() {
        let hm = |s: &str| parse_inet(s).unwrap().hostmask().format();
        assert_eq!(hm("192.168.1.5/24"), "0.0.0.255");
        assert_eq!(hm("192.168.1.5/30"), "0.0.0.3");
        assert_eq!(hm("10.1.2.3/0"), "255.255.255.255");
        assert_eq!(hm("192.168.1.5"), "0.0.0.0");
        assert_eq!(hm("10.1.2.3/12"), "0.15.255.255");
        assert_eq!(hm("2001:db8::1/64"), "::ffff:ffff:ffff:ffff");
        assert_eq!(hm("2001:db8::1/128"), "::");
    }

    #[test]
    fn abbrev_inet_is_display_form() {
        // INET keeps every octet; the mask shows unless it is the family maximum.
        assert_eq!(
            parse_inet("192.168.1.5/24").unwrap().abbrev(),
            "192.168.1.5/24"
        );
        assert_eq!(parse_inet("192.168.1.5").unwrap().abbrev(), "192.168.1.5");
        assert_eq!(parse_inet("10.1.0.0/16").unwrap().abbrev(), "10.1.0.0/16");
        assert_eq!(
            parse_inet("2001:db8::1/64").unwrap().abbrev(),
            "2001:db8::1/64"
        );
    }

    #[test]
    fn abbrev_cidr_drops_trailing_zero_octets_v4() {
        let ab = |s: &str| parse_cidr(s).unwrap().abbrev();
        assert_eq!(ab("10.1.0.0/16"), "10.1/16");
        assert_eq!(ab("10.0.0.0/8"), "10/8");
        assert_eq!(ab("192.168.1.0/24"), "192.168.1/24");
        assert_eq!(ab("192.168.1.0/32"), "192.168.1.0/32");
        assert_eq!(ab("0.0.0.0/0"), "0/0");
        assert_eq!(ab("10.0.0.0/12"), "10.0/12");
    }

    #[test]
    fn abbrev_cidr_drops_trailing_zero_groups_v6() {
        let ab = |s: &str| parse_cidr(s).unwrap().abbrev();
        assert_eq!(ab("2001:db8::/32"), "2001:db8/32");
        assert_eq!(ab("2001:db8:abcd::/48"), "2001:db8:abcd/48");
        assert_eq!(ab("2001:db8::/64"), "2001:db8::/64");
        assert_eq!(ab("ffff::/16"), "ffff::/16");
        assert_eq!(ab("ff00::/8"), "ff00::/8");
        assert_eq!(ab("::/0"), "::/0");
        assert_eq!(ab("2001:db8::/128"), "2001:db8::/128");
        assert_eq!(ab("2001:db8:0:1::/64"), "2001:db8::1/64");
        assert_eq!(ab("1:2:3::/48"), "1:2:3/48");
        // IPv4-in-IPv6 networks keep the dotted-quad tail.
        assert_eq!(ab("::ffff:0:0/96"), "::ffff/96");
        assert_eq!(ab("::ffff:1.2.3.4/128"), "::ffff:1.2.3.4/128");
        assert_eq!(ab("::ffff:1.2.0.0/112"), "::ffff:1.2/112");
        assert_eq!(ab("::1.2.3.4/128"), "::1.2.3.4/128");
        assert_eq!(ab("::1/128"), "::1/128");
    }

    #[test]
    fn merge_finds_smallest_common_network() {
        let mg = |a: &str, b: &str| {
            parse_inet(a)
                .unwrap()
                .merge(parse_inet(b).unwrap())
                .unwrap()
                .format()
        };
        assert_eq!(mg("192.168.1.5/24", "192.168.2.5/24"), "192.168.0.0/22");
        assert_eq!(mg("10.0.0.0/24", "10.0.1.0/24"), "10.0.0.0/23");
        assert_eq!(mg("10.0.0.0/8", "10.5.3.2/32"), "10.0.0.0/8");
        assert_eq!(mg("192.168.1.1", "192.168.1.1"), "192.168.1.1/32");
        assert_eq!(mg("0.0.0.0/0", "255.255.255.255"), "0.0.0.0/0");
        assert_eq!(mg("2001:db8::1", "2001:db8::ffff"), "2001:db8::/112");
        assert_eq!(mg("2001:db8:1::/48", "2001:db8:2::/48"), "2001:db8::/46");
        // Cross-family merge has no answer.
        assert!(
            parse_inet("192.168.1.5")
                .unwrap()
                .merge(parse_inet("::1").unwrap())
                .is_none()
        );
    }

    #[test]
    fn same_family_never_errors() {
        let sf = |a: &str, b: &str| parse_inet(a).unwrap().same_family(&parse_inet(b).unwrap());
        assert!(sf("192.168.1.5", "10.0.0.1"));
        assert!(!sf("192.168.1.5", "::1"));
        assert!(sf("::1", "2001:db8::1"));
    }

    #[test]
    fn add_offset_carries_and_preserves_mask() {
        let off = |s: &str, d: i64| parse_inet(s).unwrap().add_offset(d).map(|v| v.format());
        // Plain add, and a carry across an octet boundary.
        assert_eq!(off("192.168.1.5", 10).as_deref(), Some("192.168.1.15"));
        assert_eq!(off("192.168.1.250", 10).as_deref(), Some("192.168.2.4"));
        // Subtract with a borrow across octets; the /24 mask is preserved.
        assert_eq!(off("192.168.1.5", -10).as_deref(), Some("192.168.0.251"));
        assert_eq!(off("192.168.1.5/24", 1).as_deref(), Some("192.168.1.6/24"));
        // IPv6.
        assert_eq!(off("2001:db8::1", 5).as_deref(), Some("2001:db8::6"));
        // Out of range: past the all-ones / all-zero address of the family → None.
        assert_eq!(off("255.255.255.255", 1), None);
        assert_eq!(off("0.0.0.0", -1), None);
        assert_eq!(off("::", -1), None);
        // A large IPv6 offset that stays in range.
        assert_eq!(
            off("2001:db8::", i64::MAX).as_deref(),
            Some("2001:db8::7fff:ffff:ffff:ffff")
        );
    }

    #[test]
    fn diff_is_signed_and_range_checked() {
        let d = |a: &str, b: &str| parse_inet(a).unwrap().diff(&parse_inet(b).unwrap());
        assert_eq!(d("192.168.1.20", "192.168.1.5"), Some(15));
        assert_eq!(d("192.168.1.5", "192.168.1.20"), Some(-15));
        assert_eq!(
            d("2001:db8::ffff:ffff:ffff", "2001:db8::"),
            Some(281_474_976_710_655)
        );
        // Difference that does not fit i64 → None.
        assert_eq!(d("2001:db8::", "2001::"), None);
        // Cross-family difference → None.
        assert_eq!(d("192.168.1.5", "::1"), None);
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
