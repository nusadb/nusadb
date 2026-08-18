//! Binary JSONB value codec — a decomposed form a lookup can search instead of re-parse.
//!
//! # Why
//!
//! A `JSON` value is stored today as canonical **text**, so every operator re-parses the whole
//! document to reach one field ([`crate::json::get_field`] and its siblings all begin with
//! `parse(json)?`). Measured by `benches/json_ops.rs`, fetching a single field costs 1.7 µs at 8
//! keys, 19 µs at 64, and **123 µs at 256** — and the first key costs the same as the last, which is
//! the signature of materializing the document before looking at any key.
//!
//! This module stores the document decomposed, with an offset table per container, so a member
//! lookup is a binary search over keys and a slice — `O(log k)` comparisons, no allocation, and
//! independent of how wide the document is.
//!
//! # Layout
//!
//! A stored value is one [`TAG`] byte followed by a node. The tag is what makes the encoding
//! *self-describing*: canonical JSON text can only ever begin with `{`, `[`, `"`, `-`, a digit, or
//! `t`/`f`/`n`, so a `0x01` first byte is unambiguous. A reader can therefore hold both forms in the
//! same column and decide per value — no data migration, and rows written by an older build keep
//! being read exactly as they were.
//!
//! ```text
//! value := TAG(0x01) node
//!
//! node  := type:u8, payload
//!
//!   0 NULL   payload: (empty)
//!   1 FALSE  payload: (empty)
//!   2 TRUE   payload: (empty)
//!   3 NUMBER payload: len:u32, ASCII digits  — the *exact* normalized decimal, never via f64
//!   4 STRING payload: len:u32, UTF-8 bytes
//!   5 ARRAY  payload: count:u32, (count+1)×end:u32, element nodes
//!   6 OBJECT payload: count:u32, (count+1)×key_end:u32, (count+1)×val_end:u32,
//!                     key bytes, value nodes
//! ```
//!
//! Offsets are cumulative ends relative to the start of their own blob, so entry `i` spans
//! `end[i]..end[i+1]` and any entry is reachable in `O(1)` — that is what makes the binary search
//! possible. Object keys are held **sorted by `(length, then bytes)`**, the order the reference
//! engine's `jsonb` uses, so a document's canonical form (and therefore equality, which compares
//! encoded bytes) agrees with it member for member.
//!
//! # Exactness
//!
//! Numbers are normalized through [`crate::numeric::Decimal`] and stored as text, so
//! `1e3` becomes `1000` and `1.0` stays `1.0` — matching the reference — while a value too long for
//! `Decimal` keeps its source digits verbatim. Nothing is routed through `f64`, which is how the
//! text path silently loses precision on a long decimal today.

use serde_json::Value as J;

use crate::numeric::Decimal;

/// Marks a stored value as this binary form. Chosen because canonical JSON text can never begin
/// with a control byte, so the two encodings coexist in one column without a version stamp.
pub const TAG: u8 = 0x01;

const T_NULL: u8 = 0;
const T_FALSE: u8 = 1;
const T_TRUE: u8 = 2;
const T_NUMBER: u8 = 3;
const T_STRING: u8 = 4;
const T_ARRAY: u8 = 5;
const T_OBJECT: u8 = 6;

/// Whether `bytes` is a binary document rather than canonical JSON text.
#[must_use]
pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.first() == Some(&TAG)
}

/// The order object keys are held in: shorter keys first, then bytewise. This is the reference
/// engine's `jsonb` key order; matching it keeps a document's canonical bytes — and so equality,
/// `DISTINCT` and text output — agreeing with it.
pub(crate) fn key_order(a: &str, b: &str) -> std::cmp::Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.as_bytes().cmp(b.as_bytes()))
}

/// Normalize a JSON number to its exact decimal text. `1e3` → `1000`, `1.0` → `1.0` — the same
/// canonical form the reference engine's `numeric`-backed `jsonb` prints. A literal beyond
/// [`Decimal`]'s range keeps its source digits, so no digit is ever lost.
///
/// The source text is what is normalized, never an `f64`: `serde_json` is built with
/// `arbitrary_precision` so a number arrives as its literal rather than a rounded double.
fn number_text(n: &serde_json::Number) -> String {
    let raw = n.to_string();
    normalize_number(&raw).unwrap_or(raw)
}

/// Parse a JSON number literal — including the exponent form, which [`Decimal::parse`] does not
/// accept — into canonical decimal text. `None` when it does not fit, and the caller then keeps the
/// literal verbatim.
pub(crate) fn normalize_number(raw: &str) -> Option<String> {
    let (mantissa_text, exponent) = match raw.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().ok()?),
        None => (raw, 0),
    };
    let base = Decimal::parse(mantissa_text)?;
    if exponent == 0 {
        return Some(base.format());
    }
    // Shifting by the exponent moves the decimal point: a positive exponent consumes scale, and
    // once the scale is spent it multiplies the mantissa instead.
    let shifted_scale = i32::from(base.scale) - exponent;
    if shifted_scale >= 0 {
        let scale = u8::try_from(shifted_scale).ok()?;
        return Some(
            Decimal {
                mantissa: base.mantissa,
                scale,
            }
            .format(),
        );
    }
    let factor = 10_i128.checked_pow(shifted_scale.unsigned_abs())?;
    Some(
        Decimal {
            mantissa: base.mantissa.checked_mul(factor)?,
            scale: 0,
        }
        .format(),
    )
}

/// Encode a parsed document into the binary form, including the leading [`TAG`].
#[must_use]
pub fn encode(value: &J) -> Vec<u8> {
    let mut out = vec![TAG];
    write_node(value, &mut out);
    out
}

/// Encode JSON *text* into the binary form, or `None` if it is not valid JSON.
#[must_use]
pub fn encode_text(text: &str) -> Option<Vec<u8>> {
    Some(encode(&crate::json::parse(text)?))
}

fn write_node(value: &J, out: &mut Vec<u8>) {
    match value {
        J::Null => out.push(T_NULL),
        J::Bool(false) => out.push(T_FALSE),
        J::Bool(true) => out.push(T_TRUE),
        J::Number(n) => {
            out.push(T_NUMBER);
            write_bytes(number_text(n).as_bytes(), out);
        },
        J::String(s) => {
            out.push(T_STRING);
            write_bytes(s.as_bytes(), out);
        },
        J::Array(items) => {
            out.push(T_ARRAY);
            write_u32(u32_of(items.len()), out);
            // Build the element blob first so the cumulative ends are known before they are written.
            let mut blob = Vec::new();
            let mut ends = Vec::with_capacity(items.len() + 1);
            ends.push(0_u32);
            for item in items {
                write_node(item, &mut blob);
                ends.push(u32_of(blob.len()));
            }
            for end in &ends {
                write_u32(*end, out);
            }
            out.extend_from_slice(&blob);
        },
        J::Object(map) => {
            out.push(T_OBJECT);
            // serde_json's map is bytewise-ordered; re-sort into the reference engine's key order.
            let mut entries: Vec<(&String, &J)> = map.iter().collect();
            entries.sort_by(|a, b| key_order(a.0, b.0));
            write_u32(u32_of(entries.len()), out);
            let mut keys = Vec::new();
            let mut vals = Vec::new();
            let mut key_ends = Vec::with_capacity(entries.len() + 1);
            let mut val_ends = Vec::with_capacity(entries.len() + 1);
            key_ends.push(0_u32);
            val_ends.push(0_u32);
            for (key, val) in entries {
                keys.extend_from_slice(key.as_bytes());
                key_ends.push(u32_of(keys.len()));
                write_node(val, &mut vals);
                val_ends.push(u32_of(vals.len()));
            }
            for end in &key_ends {
                write_u32(*end, out);
            }
            for end in &val_ends {
                write_u32(*end, out);
            }
            out.extend_from_slice(&keys);
            out.extend_from_slice(&vals);
        },
    }
}

fn write_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    write_u32(u32_of(bytes.len()), out);
    out.extend_from_slice(bytes);
}

fn write_u32(v: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Lengths are bounded by the 4 GiB tuple ceiling the text path already enforces, so a saturating
/// conversion cannot silently truncate a document that could have been stored.
fn u32_of(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

// === reading ===============================================================

/// A borrowed view of one encoded node. Every accessor slices; nothing is copied or parsed until a
/// caller asks for an owned value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node<'a> {
    bytes: &'a [u8],
}

/// The JSON type of a [`Node`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// JSON `null`.
    Null,
    /// `true` / `false`.
    Bool,
    /// A number.
    Number,
    /// A string.
    String,
    /// An array.
    Array,
    /// An object.
    Object,
}

impl<'a> Node<'a> {
    /// View a stored value, skipping the leading [`TAG`]. `None` if `bytes` is not this form or is
    /// truncated past its header.
    #[must_use]
    pub fn from_stored(bytes: &'a [u8]) -> Option<Self> {
        let rest = bytes.strip_prefix(&[TAG])?;
        let node = Self { bytes: rest };
        node.kind().map(|_| node)
    }

    /// The node's JSON type, or `None` if the type byte is unknown.
    #[must_use]
    pub fn kind(self) -> Option<Kind> {
        Some(match *self.bytes.first()? {
            T_NULL => Kind::Null,
            T_FALSE | T_TRUE => Kind::Bool,
            T_NUMBER => Kind::Number,
            T_STRING => Kind::String,
            T_ARRAY => Kind::Array,
            T_OBJECT => Kind::Object,
            _ => return None,
        })
    }

    fn payload(self) -> &'a [u8] {
        self.bytes.get(1..).unwrap_or(&[])
    }

    /// The member count of an array or object; `None` for a scalar.
    #[must_use]
    pub fn len(self) -> Option<usize> {
        match self.kind()? {
            Kind::Array | Kind::Object => read_u32(self.payload(), 0).map(|n| n as usize),
            _ => None,
        }
    }

    /// Whether an array or object has no members. `None` for a scalar.
    #[must_use]
    pub fn is_empty(self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    /// The object member named `key`, found by binary search over the sorted key table —
    /// `O(log k)` comparisons and no allocation. `None` if this is not an object or the key is
    /// absent.
    #[must_use]
    pub fn get(self, key: &str) -> Option<Self> {
        let obj = self.object()?;
        let mut lo = 0_usize;
        let mut hi = obj.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let found = obj.key(mid)?;
            match key_order(found, key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return obj.value(mid),
            }
        }
        None
    }

    /// The array element at `index`, negative counting from the end — `O(1)` through the offset
    /// table. `None` if this is not an array or the index is out of range.
    #[must_use]
    pub fn at(self, index: i64) -> Option<Self> {
        let arr = self.array()?;
        let idx = if index >= 0 {
            usize::try_from(index).ok().filter(|i| *i < arr.count)?
        } else {
            arr.count
                .checked_sub(usize::try_from(index.unsigned_abs()).ok()?)?
        };
        arr.get(idx)
    }

    /// The object's member names, in stored order.
    #[must_use]
    pub fn keys(self) -> Vec<&'a str> {
        self.object().map_or_else(Vec::new, |obj| {
            (0..obj.count).filter_map(|i| obj.key(i)).collect()
        })
    }

    /// The object's `(key, value)` members, in stored order.
    #[must_use]
    pub fn entries(self) -> Vec<(&'a str, Self)> {
        self.object().map_or_else(Vec::new, |obj| {
            (0..obj.count)
                .filter_map(|i| Some((obj.key(i)?, obj.value(i)?)))
                .collect()
        })
    }

    /// The array's elements, in order.
    #[must_use]
    pub fn elements(self) -> Vec<Self> {
        self.array().map_or_else(Vec::new, |arr| {
            (0..arr.count).filter_map(|i| arr.get(i)).collect()
        })
    }

    /// The string contents, or `None` if this node is not a string.
    #[must_use]
    pub fn as_str(self) -> Option<&'a str> {
        if self.kind()? != Kind::String {
            return None;
        }
        let len = read_u32(self.payload(), 0)? as usize;
        std::str::from_utf8(self.payload().get(4..4 + len)?).ok()
    }

    /// The number's exact decimal text, or `None` if this node is not a number.
    #[must_use]
    pub fn as_number(self) -> Option<&'a str> {
        if self.kind()? != Kind::Number {
            return None;
        }
        let len = read_u32(self.payload(), 0)? as usize;
        std::str::from_utf8(self.payload().get(4..4 + len)?).ok()
    }

    /// The boolean value, or `None` if this node is not a boolean.
    #[must_use]
    pub fn as_bool(self) -> Option<bool> {
        match *self.bytes.first()? {
            T_TRUE => Some(true),
            T_FALSE => Some(false),
            _ => None,
        }
    }

    /// Render this node as canonical JSON text — the same string the text path stores, so the two
    /// encodings agree on output byte for byte.
    #[must_use]
    pub fn to_text(self) -> String {
        let mut out = String::new();
        self.write_text(&mut out);
        out
    }

    fn write_text(self, out: &mut String) {
        match self.kind() {
            Some(Kind::Null) | None => out.push_str("null"),
            Some(Kind::Bool) => out.push_str(if self.as_bool() == Some(true) {
                "true"
            } else {
                "false"
            }),
            Some(Kind::Number) => out.push_str(self.as_number().unwrap_or("0")),
            Some(Kind::String) => {
                out.push_str(&crate::json::string_literal(self.as_str().unwrap_or("")));
            },
            Some(Kind::Array) => {
                out.push('[');
                for (i, item) in self.elements().into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_text(out);
                }
                out.push(']');
            },
            Some(Kind::Object) => {
                out.push('{');
                for (i, (key, val)) in self.entries().into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&crate::json::string_literal(key));
                    out.push(':');
                    val.write_text(out);
                }
                out.push('}');
            },
        }
    }

    fn object(self) -> Option<Obj<'a>> {
        if self.kind()? != Kind::Object {
            return None;
        }
        let p = self.payload();
        let count = read_u32(p, 0)? as usize;
        // count+1 key ends, then count+1 value ends, then the key blob, then the value blob.
        let key_ends = 4;
        let val_ends = key_ends + (count + 1) * 4;
        let keys = val_ends + (count + 1) * 4;
        let keys_len = read_u32(p, key_ends + count * 4)? as usize;
        Some(Obj {
            p,
            count,
            key_ends,
            val_ends,
            keys,
            vals: keys + keys_len,
        })
    }

    fn array(self) -> Option<Arr<'a>> {
        if self.kind()? != Kind::Array {
            return None;
        }
        let p = self.payload();
        let count = read_u32(p, 0)? as usize;
        Some(Arr {
            p,
            count,
            ends: 4,
            blob: 4 + (count + 1) * 4,
        })
    }
}

/// Cursor over an encoded object's offset tables.
struct Obj<'a> {
    p: &'a [u8],
    count: usize,
    key_ends: usize,
    val_ends: usize,
    keys: usize,
    vals: usize,
}

impl<'a> Obj<'a> {
    fn key(&self, i: usize) -> Option<&'a str> {
        let start = read_u32(self.p, self.key_ends + i * 4)? as usize;
        let end = read_u32(self.p, self.key_ends + (i + 1) * 4)? as usize;
        std::str::from_utf8(self.p.get(self.keys + start..self.keys + end)?).ok()
    }

    fn value(&self, i: usize) -> Option<Node<'a>> {
        let start = read_u32(self.p, self.val_ends + i * 4)? as usize;
        let end = read_u32(self.p, self.val_ends + (i + 1) * 4)? as usize;
        Some(Node {
            bytes: self.p.get(self.vals + start..self.vals + end)?,
        })
    }
}

/// Cursor over an encoded array's offset table.
struct Arr<'a> {
    p: &'a [u8],
    count: usize,
    ends: usize,
    blob: usize,
}

impl<'a> Arr<'a> {
    fn get(&self, i: usize) -> Option<Node<'a>> {
        let start = read_u32(self.p, self.ends + i * 4)? as usize;
        let end = read_u32(self.p, self.ends + (i + 1) * 4)? as usize;
        Some(Node {
            bytes: self.p.get(self.blob + start..self.blob + end)?,
        })
    }
}

/// Read a little-endian `u32` at `off`, or `None` when the slice is short — every structural read
/// goes through this, so a truncated or corrupt payload yields `None` instead of panicking.
fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(off..off + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(text: &str) -> Vec<u8> {
        encode_text(text).expect("valid JSON")
    }

    fn node(bytes: &[u8]) -> Node<'_> {
        Node::from_stored(bytes).expect("decodable")
    }

    #[test]
    fn round_trips_every_json_shape_to_canonical_text() {
        for (input, canonical) in [
            ("null", "null"),
            ("true", "true"),
            ("false", "false"),
            ("42", "42"),
            (r#""hi""#, r#""hi""#),
            ("[]", "[]"),
            ("{}", "{}"),
            ("[1,2,3]", "[1,2,3]"),
            (r#"{"a":1}"#, r#"{"a":1}"#),
            (
                r#"{"a":{"b":[1,null,true]}}"#,
                r#"{"a":{"b":[1,null,true]}}"#,
            ),
            // A string needing escapes survives the round trip.
            (r#"{"a":"x\"y"}"#, r#"{"a":"x\"y"}"#),
        ] {
            let bytes = enc(input);
            assert_eq!(node(&bytes).to_text(), canonical, "input {input}");
        }
    }

    #[test]
    fn keys_are_held_in_the_reference_engines_order() {
        // Shorter keys first, then bytewise — so `b` precedes `aa`, which plain bytewise ordering
        // would reverse. This is what makes the canonical bytes agree with the reference.
        let bytes = enc(r#"{"aa":1,"b":2,"ccc":3}"#);
        assert_eq!(node(&bytes).keys(), vec!["b", "aa", "ccc"]);
        assert_eq!(node(&bytes).to_text(), r#"{"b":2,"aa":1,"ccc":3}"#);
    }

    #[test]
    fn numbers_are_exact_and_normalized() {
        // `1e3` expands and `1.0` keeps its scale, matching the reference; neither goes via f64.
        let bytes = enc(r#"{"a":1.0,"b":1e3}"#);
        assert_eq!(node(&bytes).to_text(), r#"{"a":1.0,"b":1000}"#);
        // A decimal far beyond f64's 17 significant digits keeps every digit.
        let long = "0.12345678901234567890123456789";
        let bytes = enc(&format!(r#"{{"n":{long}}}"#));
        assert_eq!(node(&bytes).get("n").and_then(Node::as_number), Some(long));
    }

    #[test]
    fn member_lookup_finds_every_key_and_no_other() {
        // Widths spanning the binary search's branches, including keys of differing length.
        for count in [0_usize, 1, 2, 3, 8, 64, 257] {
            let doc = format!(
                "{{{}}}",
                (0..count)
                    .map(|i| format!(r#""k{i}":{i}"#))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let bytes = enc(&doc);
            let n = node(&bytes);
            assert_eq!(n.len(), Some(count));
            for i in 0..count {
                assert_eq!(
                    n.get(&format!("k{i}")).and_then(Node::as_number),
                    Some(i.to_string().as_str()),
                    "count {count}, key k{i}"
                );
            }
            assert!(n.get("absent").is_none());
            assert!(n.get("").is_none());
        }
    }

    #[test]
    fn array_indexing_counts_from_either_end() {
        let bytes = enc("[10,20,30]");
        let n = node(&bytes);
        assert_eq!(n.len(), Some(3));
        assert_eq!(n.at(0).and_then(Node::as_number), Some("10"));
        assert_eq!(n.at(2).and_then(Node::as_number), Some("30"));
        assert_eq!(n.at(-1).and_then(Node::as_number), Some("30"));
        assert_eq!(n.at(-3).and_then(Node::as_number), Some("10"));
        assert!(n.at(3).is_none());
        assert!(n.at(-4).is_none());
        // `i64::MIN` must not overflow the negation.
        assert!(n.at(i64::MIN).is_none());
    }

    #[test]
    fn entries_pair_each_key_with_its_own_value() {
        let bytes = enc(r#"{"b":"two","aa":[1],"c":{"d":null}}"#);
        let n = node(&bytes);
        let got: Vec<(&str, String)> = n
            .entries()
            .into_iter()
            .map(|(k, v)| (k, v.to_text()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("b", r#""two""#.to_owned()),
                ("c", r#"{"d":null}"#.to_owned()),
                ("aa", "[1]".to_owned()),
            ]
        );
    }

    #[test]
    fn the_tag_distinguishes_the_two_stored_forms() {
        let bytes = enc(r#"{"a":1}"#);
        assert!(is_binary(&bytes));
        // Every first byte canonical JSON text can start with must read as *not* binary, so a
        // column may hold both forms and decide per value.
        for text in [
            r#"{"a":1}"#,
            "[1]",
            r#""s""#,
            "-1",
            "0",
            "9",
            "true",
            "false",
            "null",
        ] {
            assert!(!is_binary(text.as_bytes()), "text {text} misread as binary");
        }
        assert!(!is_binary(&[]));
    }

    #[test]
    fn a_truncated_or_corrupt_payload_yields_none_never_a_panic() {
        let bytes = enc(r#"{"a":1,"bb":[1,2,3]}"#);
        // Every prefix of a valid document, and every single-byte corruption of its header.
        for cut in 0..bytes.len() {
            let node = Node::from_stored(&bytes[..cut]);
            if let Some(n) = node {
                let _ = n.get("a");
                let _ = n.keys();
                let _ = n.entries();
                let _ = n.to_text();
            }
        }
        let mut bad = bytes;
        if let Some(b) = bad.get_mut(1) {
            *b = 0xFF; // unknown type byte
        }
        assert!(Node::from_stored(&bad).is_none());
    }

    #[test]
    fn encoded_bytes_are_canonical_so_equality_can_compare_them() {
        // Two spellings of the same document — different key order, different whitespace, an
        // exponent — must encode to identical bytes, because equality compares the encoding.
        let a = enc(r#"{"b":1e3,"a":[1,2]}"#);
        let b = enc("{ \"a\" : [1, 2] , \"b\" : 1000 }");
        assert_eq!(a, b);
    }
}
