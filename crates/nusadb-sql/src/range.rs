//! Range types (`int4range`/`int8range`/`numrange`/`daterange`/`tsrange`/`tstzrange`).
//!
//! A range is either **empty** or a lower/upper bound pair, each bound either a value with an
//! inclusive/exclusive flag or infinite (unbounded). Discrete element kinds (integer, date) are
//! canonicalized to the half-open `[lower, upper)` form; continuous kinds (numeric, timestamp) keep
//! their bounds as given — matching the reference engine.
//!
//! This module is groundwork: [`RangeVal`] and its parsing, canonical formatting, containment, and
//! overlap are defined here; the type is wired into the SQL surface (column type, operators, and
//! constructor/accessor functions) separately.

use nusadb_core::engine::RangeKind;

use crate::ast;
use crate::executor::eval::compare;

/// A parsed range value: an element kind, an empty flag, and the two bounds. A `None` bound value is
/// infinite; the `_inc` flag is meaningful only for a finite bound.
#[derive(Clone, Debug, PartialEq)]
pub struct RangeVal {
    /// Which range type this is (its element kind).
    pub kind: RangeKind,
    /// The empty range contains no points; its bounds are ignored.
    pub empty: bool,
    /// Lower bound value, or `None` for unbounded below.
    pub lower: Option<ast::Value>,
    /// Upper bound value, or `None` for unbounded above.
    pub upper: Option<ast::Value>,
    /// Whether a finite lower bound is inclusive.
    pub lower_inc: bool,
    /// Whether a finite upper bound is inclusive.
    pub upper_inc: bool,
}

impl RangeVal {
    /// The canonical empty range of `kind`.
    #[must_use]
    pub const fn empty(kind: RangeKind) -> Self {
        Self {
            kind,
            empty: true,
            lower: None,
            upper: None,
            lower_inc: false,
            upper_inc: false,
        }
    }

    /// Build a range from bounds and inclusivity, canonicalizing it. Returns `None` when the lower
    /// bound strictly exceeds the upper (an invalid range, which the reference engine rejects), when
    /// a bound's value variant is not the one `kind` encodes, or when canonicalizing a discrete
    /// inclusive upper bound would run past the element type's maximum. Equal bounds are allowed and
    /// may canonicalize to a single point or to empty.
    #[must_use]
    pub fn new(
        kind: RangeKind,
        lower: Option<ast::Value>,
        upper: Option<ast::Value>,
        lower_inc: bool,
        upper_inc: bool,
    ) -> Option<Self> {
        // A bound whose value variant does not match the element kind has no encoding, so reject it
        // here rather than let it reach [`encode`].
        for bound in [lower.as_ref(), upper.as_ref()].into_iter().flatten() {
            if !bound_matches(bound, kind) {
                return None;
            }
        }
        if let (Some(lo), Some(hi)) = (&lower, &upper)
            && compare(lo, hi) == std::cmp::Ordering::Greater
        {
            return None;
        }
        let mut r = Self {
            kind,
            empty: false,
            // An infinite bound is never inclusive.
            lower_inc: lower_inc && lower.is_some(),
            upper_inc: upper_inc && upper.is_some(),
            lower,
            upper,
        };
        r.canonicalize().then_some(r)
    }

    /// Normalize a discrete range to `[)` and detect emptiness. Returns `false` when the range has
    /// no canonical form (see [`RangeVal::new`]).
    fn canonicalize(&mut self) -> bool {
        if self.empty {
            return true;
        }
        if self.kind.is_discrete() {
            // Exclusive lower `(x` → inclusive `[x+1`; inclusive upper `y]` → exclusive `y+1)`.
            if let (Some(v), false) = (&self.lower, self.lower_inc) {
                let Some(next) = increment(v) else {
                    // No element sits above the maximum, so `(max,…` holds nothing.
                    *self = Self::empty(self.kind);
                    return true;
                };
                self.lower = Some(next);
                self.lower_inc = true;
            }
            if let (Some(v), true) = (&self.upper, self.upper_inc) {
                // `…,max]` has no half-open form: its exclusive upper would be one past the maximum.
                // Reject loudly rather than silently widen the range to unbounded.
                let Some(next) = increment(v) else {
                    return false;
                };
                self.upper = Some(next);
                self.upper_inc = false;
            }
        }
        // If the bounds cross (or touch with an exclusive side), the range is empty.
        if let (Some(lo), Some(hi)) = (&self.lower, &self.upper) {
            let ord = compare(lo, hi);
            let touches_empty =
                ord == std::cmp::Ordering::Equal && !(self.lower_inc && self.upper_inc);
            if ord == std::cmp::Ordering::Greater || touches_empty {
                *self = Self::empty(self.kind);
            }
        }
        true
    }

    /// Whether `elem` (a value of the element type) falls inside the range.
    #[must_use]
    pub fn contains_elem(&self, elem: &ast::Value) -> bool {
        if self.empty {
            return false;
        }
        if let Some(lo) = &self.lower {
            match compare(elem, lo) {
                std::cmp::Ordering::Less => return false,
                std::cmp::Ordering::Equal if !self.lower_inc => return false,
                _ => {},
            }
        }
        if let Some(hi) = &self.upper {
            match compare(elem, hi) {
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal if !self.upper_inc => return false,
                _ => {},
            }
        }
        true
    }

    /// Total order matching the reference engine: empty ranges sort first, then by lower bound
    /// (−infinity first; equal values order an inclusive `[` before an exclusive `(`), then by upper
    /// bound (equal values order an exclusive `)` before an inclusive `]`, and +infinity last).
    #[must_use]
    pub fn range_cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // A column holds one kind, so a cross-kind comparison never arises from SQL; order by kind
        // first anyway, so that two empty ranges of different kinds do not compare equal while
        // [`hash_bytes`] keeps them in separate buckets.
        match self.kind.tag().cmp(&other.kind.tag()) {
            Ordering::Equal => {},
            ord => return ord,
        }
        match (self.empty, other.empty) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {},
        }
        // Lower bound: None is −infinity (smallest).
        match (&self.lower, &other.lower) {
            (None, None) => {},
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(a), Some(b)) => match compare(a, b) {
                Ordering::Equal => match other.lower_inc.cmp(&self.lower_inc) {
                    Ordering::Equal => {},
                    ord => return ord, // inclusive `[` sorts before exclusive `(`
                },
                ord => return ord,
            },
        }
        // Upper bound: None is +infinity (largest).
        match (&self.upper, &other.upper) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => match compare(a, b) {
                Ordering::Equal => self.upper_inc.cmp(&other.upper_inc), // `)` before `]`
                ord => ord,
            },
        }
    }

    /// Whether the two ranges share at least one point (`&&`).
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.empty || other.empty {
            return false;
        }
        // They overlap iff each one's lower bound is at-or-below the other's upper bound.
        lower_below_upper(self, other) && lower_below_upper(other, self)
    }

    /// Canonical text form: `empty`, or `[lo,hi)` with the actual inclusivity and blank infinite ends.
    #[must_use]
    pub fn format(&self) -> String {
        if self.empty {
            return "empty".to_owned();
        }
        let open = if self.lower_inc { '[' } else { '(' };
        let close = if self.upper_inc { ']' } else { ')' };
        let lo = self.lower.as_ref().map(fmt_bound).unwrap_or_default();
        let hi = self.upper.as_ref().map(fmt_bound).unwrap_or_default();
        std::format!("{open}{lo},{hi}{close}")
    }
}

/// Flags-byte bit: the range is empty (no bound bytes follow).
const FLAG_EMPTY: u8 = 1;
/// Flags-byte bit: the lower bound is infinite (its bytes are absent).
const FLAG_LOWER_INF: u8 = 2;
/// Flags-byte bit: the upper bound is infinite (its bytes are absent).
const FLAG_UPPER_INF: u8 = 4;
/// Flags-byte bit: a finite lower bound is inclusive.
const FLAG_LOWER_INC: u8 = 8;
/// Flags-byte bit: a finite upper bound is inclusive.
const FLAG_UPPER_INC: u8 = 16;

/// Serialize a range for storage/wire.
///
/// The layout is a flags byte (bit0 empty, bit1 lower-infinite, bit2 upper-infinite, bit3
/// lower-inclusive, bit4 upper-inclusive) then each finite bound's bytes. The element kind is not
/// stored — the column type supplies it at [`decode`] time. Numeric bounds keep their original scale
/// here (so a stored/displayed `numrange` shows `1.50`); [`hash_bytes`] is the scale-normalized form
/// for equality keys.
#[must_use]
pub fn encode(r: &RangeVal) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 2 * bound_size(r.kind));
    let flags = u8::from(r.empty)
        | (u8::from(r.lower.is_none()) << 1)
        | (u8::from(r.upper.is_none()) << 2)
        | (u8::from(r.lower_inc) << 3)
        | (u8::from(r.upper_inc) << 4);
    out.push(flags);
    if !r.empty {
        for bound in [r.lower.as_ref(), r.upper.as_ref()].into_iter().flatten() {
            encode_bound(bound, r.kind, &mut out);
        }
    }
    out
}

/// How many bytes follow the flags byte in the [`encode`] form of a range of `kind`.
///
/// A reader that frames the record itself — the spill codec, which has no length prefix to lean on —
/// asks here instead of re-deriving the flag layout, so the two stay in step by construction.
#[must_use]
pub const fn payload_len(flags: u8, kind: RangeKind) -> usize {
    if flags & FLAG_EMPTY != 0 {
        return 0;
    }
    let bounds = (flags & FLAG_LOWER_INF == 0) as usize + (flags & FLAG_UPPER_INF == 0) as usize;
    bounds * bound_size(kind)
}

/// Canonical bytes for equality/hash keys (`DISTINCT`, `GROUP BY`, hash-join).
///
/// Like [`encode`], but a `numrange` bound is scale-normalized first so that scale-equal values —
/// `1.5` and `1.50` — produce identical bytes and land in the same bucket, matching the
/// scale-independent `=` and comparison. Other element kinds already encode canonically, so this
/// defers to [`encode`] for them.
#[must_use]
pub fn hash_bytes(r: &RangeVal) -> Vec<u8> {
    // The kind leads the bytes: two kinds can share a bound layout (`tsrange` / `tstzrange`), and an
    // empty range of any kind encodes identically, so without it distinct values would collide.
    if r.kind != RangeKind::Num {
        let mut out = Vec::with_capacity(1 + 1 + 2 * bound_size(r.kind));
        out.push(r.kind.tag());
        out.extend_from_slice(&encode(r));
        return out;
    }
    let normalize = |bound: &Option<ast::Value>| -> Option<ast::Value> {
        bound.as_ref().map(|v| match v {
            ast::Value::Numeric(d) => ast::Value::Numeric(d.trim_scale()),
            other => other.clone(),
        })
    };
    let mut out = Vec::with_capacity(1 + 1 + 2 * bound_size(r.kind));
    out.push(r.kind.tag());
    out.extend_from_slice(&encode(&RangeVal {
        kind: r.kind,
        empty: r.empty,
        lower: normalize(&r.lower),
        upper: normalize(&r.upper),
        lower_inc: r.lower_inc,
        upper_inc: r.upper_inc,
    }));
    out
}

/// Decode the [`encode`] form of a range of `kind` starting at `pos`; returns it and the new position.
#[must_use]
pub fn decode(bytes: &[u8], pos: usize, kind: RangeKind) -> Option<(RangeVal, usize)> {
    let flags = *bytes.get(pos)?;
    let mut cur = pos + 1;
    if flags & FLAG_EMPTY != 0 {
        return Some((RangeVal::empty(kind), cur));
    }
    let lower = if flags & FLAG_LOWER_INF == 0 {
        let (v, next) = decode_bound(bytes, cur, kind)?;
        cur = next;
        Some(v)
    } else {
        None
    };
    let upper = if flags & FLAG_UPPER_INF == 0 {
        let (v, next) = decode_bound(bytes, cur, kind)?;
        cur = next;
        Some(v)
    } else {
        None
    };
    // Already canonical on disk, so reconstruct directly rather than re-canonicalizing. An
    // inclusive flag set on an infinite bound is not a shape [`encode`] ever writes; drop it so a
    // damaged page cannot produce a range that formats as `[,5)`.
    Some((
        RangeVal {
            kind,
            empty: false,
            lower_inc: flags & FLAG_LOWER_INC != 0 && lower.is_some(),
            upper_inc: flags & FLAG_UPPER_INC != 0 && upper.is_some(),
            lower,
            upper,
        },
        cur,
    ))
}

/// The number of bytes a single finite bound of `kind` occupies in the [`encode`] form.
#[must_use]
pub const fn bound_size(kind: RangeKind) -> usize {
    match kind {
        RangeKind::Int | RangeKind::Ts | RangeKind::TsTz => 8,
        RangeKind::Date => 4,
        RangeKind::Num => 17,
    }
}

/// Whether a bound value's variant is the one `kind` encodes.
const fn bound_matches(v: &ast::Value, kind: RangeKind) -> bool {
    matches!(
        (v, kind),
        (ast::Value::Int(_), RangeKind::Int)
            | (ast::Value::Numeric(_), RangeKind::Num)
            | (ast::Value::Date(_), RangeKind::Date)
            | (ast::Value::Timestamp(_), RangeKind::Ts)
            | (ast::Value::TimestampTz(_), RangeKind::TsTz)
    )
}

/// Encode a single bound value by its element type.
///
/// [`RangeVal::new`] rejects a bound whose variant does not match `kind` and [`decode`] builds its
/// bounds from `kind`, so the mismatch arm is unreachable; it still writes a full-width bound so a
/// value reaching here some other way cannot truncate the record and desync the columns after it.
fn encode_bound(v: &ast::Value, kind: RangeKind, out: &mut Vec<u8>) {
    debug_assert!(
        bound_matches(v, kind),
        "range bound does not match its kind"
    );
    match v {
        ast::Value::Int(i) if kind == RangeKind::Int => out.extend_from_slice(&i.to_le_bytes()),
        ast::Value::Date(d) if kind == RangeKind::Date => out.extend_from_slice(&d.to_le_bytes()),
        ast::Value::Timestamp(t) if kind == RangeKind::Ts => {
            out.extend_from_slice(&t.to_le_bytes());
        },
        ast::Value::TimestampTz(t) if kind == RangeKind::TsTz => {
            out.extend_from_slice(&t.to_le_bytes());
        },
        ast::Value::Numeric(d) if kind == RangeKind::Num => {
            out.extend_from_slice(&d.mantissa.to_le_bytes());
            out.push(d.scale);
        },
        _ => out.resize(out.len() + bound_size(kind), 0),
    }
}

/// Decode a single bound value of `kind` at `pos`; returns it and the new position.
fn decode_bound(bytes: &[u8], pos: usize, kind: RangeKind) -> Option<(ast::Value, usize)> {
    Some(match kind {
        RangeKind::Int => {
            let arr: [u8; 8] = bytes.get(pos..pos + 8)?.try_into().ok()?;
            (ast::Value::Int(i64::from_le_bytes(arr)), pos + 8)
        },
        RangeKind::Date => {
            let arr: [u8; 4] = bytes.get(pos..pos + 4)?.try_into().ok()?;
            (ast::Value::Date(i32::from_le_bytes(arr)), pos + 4)
        },
        RangeKind::Ts | RangeKind::TsTz => {
            let arr: [u8; 8] = bytes.get(pos..pos + 8)?.try_into().ok()?;
            let t = i64::from_le_bytes(arr);
            let v = if matches!(kind, RangeKind::Ts) {
                ast::Value::Timestamp(t)
            } else {
                ast::Value::TimestampTz(t)
            };
            (v, pos + 8)
        },
        RangeKind::Num => {
            let arr: [u8; 16] = bytes.get(pos..pos + 16)?.try_into().ok()?;
            let scale = *bytes.get(pos + 16)?;
            let d = crate::numeric::Decimal {
                mantissa: i128::from_le_bytes(arr),
                scale,
            };
            (ast::Value::Numeric(d), pos + 17)
        },
    })
}

/// Whether `a`'s lower bound lies at or below `b`'s upper bound (a half of the overlap test).
fn lower_below_upper(a: &RangeVal, b: &RangeVal) -> bool {
    let (Some(lo), Some(hi)) = (&a.lower, &b.upper) else {
        return true; // an infinite side never blocks the overlap
    };
    match compare(lo, hi) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Equal => a.lower_inc && b.upper_inc,
        std::cmp::Ordering::Greater => false,
    }
}

/// The next value after `v` for a discrete element (integer / date), used to canonicalize `[)`.
fn increment(v: &ast::Value) -> Option<ast::Value> {
    match v {
        ast::Value::Int(i) => Some(ast::Value::Int(i.checked_add(1)?)),
        ast::Value::Date(d) => Some(ast::Value::Date(d.checked_add(1)?)),
        other => Some(other.clone()),
    }
}

/// Format a single bound value as its canonical text.
fn fmt_bound(v: &ast::Value) -> String {
    crate::display::value_text(v)
}

/// Parse a range literal of the given `kind`: `empty`, or `[lo,hi)` / `(lo,hi]` etc. with either end
/// blank for infinite. Bound values are parsed as the element type; returns `None` on malformed input.
#[must_use]
pub fn parse(s: &str, kind: RangeKind) -> Option<RangeVal> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("empty") {
        return Some(RangeVal::empty(kind));
    }
    let mut chars = t.chars();
    let open = chars.next()?;
    let lower_inc = match open {
        '[' => true,
        '(' => false,
        _ => return None,
    };
    let inner = chars.as_str();
    let close = inner.chars().last()?;
    let upper_inc = match close {
        ']' => true,
        ')' => false,
        _ => return None,
    };
    let body = inner.get(..inner.len() - close.len_utf8())?;
    let (lo_str, hi_str) = body.split_once(',')?;
    let lower = parse_bound(lo_str.trim(), kind).ok()?;
    let upper = parse_bound(hi_str.trim(), kind).ok()?;
    RangeVal::new(kind, lower, upper, lower_inc, upper_inc)
}

/// Parse a single bound: an empty string is infinite (`Ok(None)`); a value parses to `Ok(Some(v))`;
/// a malformed value is `Err(())`. (The `Result` keeps the infinite case distinct from a failure.)
fn parse_bound(s: &str, kind: RangeKind) -> Result<Option<ast::Value>, ()> {
    // Strip optional surrounding quotes (the reference engine quotes bounds with special chars).
    let s = s
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(s);
    if s.is_empty() {
        return Ok(None);
    }
    let v = match kind {
        RangeKind::Int => ast::Value::Int(s.parse().map_err(|_| ())?),
        RangeKind::Num => ast::Value::Numeric(crate::numeric::Decimal::parse(s).ok_or(())?),
        RangeKind::Date => ast::Value::Date(crate::temporal::parse_date(s).ok_or(())?),
        RangeKind::Ts => ast::Value::Timestamp(crate::temporal::parse_timestamp(s).ok_or(())?),
        RangeKind::TsTz => {
            ast::Value::TimestampTz(crate::temporal::parse_timestamptz(s).ok_or(())?)
        },
    };
    Ok(Some(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> ast::Value {
        ast::Value::Int(n)
    }

    #[test]
    fn parse_and_canonicalize_discrete() {
        // `[1,10)` stays; `(1,10]` canonicalizes to `[2,11)`; `[1,10]` to `[1,11)`.
        assert_eq!(parse("[1,10)", RangeKind::Int).unwrap().format(), "[1,10)");
        assert_eq!(parse("(1,10]", RangeKind::Int).unwrap().format(), "[2,11)");
        assert_eq!(parse("[1,10]", RangeKind::Int).unwrap().format(), "[1,11)");
        // Empty and infinite ends.
        assert_eq!(parse("empty", RangeKind::Int).unwrap().format(), "empty");
        assert_eq!(parse("[5,)", RangeKind::Int).unwrap().format(), "[5,)");
        assert_eq!(parse("(,10)", RangeKind::Int).unwrap().format(), "(,10)");
        // A touching-exclusive range collapses to empty; a lower bound above the upper is rejected.
        assert_eq!(parse("[5,5)", RangeKind::Int).unwrap().format(), "empty");
        assert_eq!(parse("[5,1)", RangeKind::Int), None);
        // A single-point inclusive range is valid.
        assert_eq!(parse("[5,5]", RangeKind::Int).unwrap().format(), "[5,6)");
    }

    #[test]
    fn numeric_range_keeps_bounds() {
        // Continuous kind: no canonicalization.
        let r = parse("(1.5,3.5]", RangeKind::Num).unwrap();
        assert_eq!(r.format(), "(1.5,3.5]");
    }

    /// One range per bound shape, for a kind whose bounds parse from these literals.
    fn shapes(kind: RangeKind, lo: &str, hi: &str) -> Vec<RangeVal> {
        vec![
            RangeVal::empty(kind),
            parse(&std::format!("[{lo},{hi})"), kind).expect("two bounds"),
            parse(&std::format!("({lo},{hi}]"), kind).expect("two bounds, other inclusivity"),
            parse(&std::format!("[{lo},)"), kind).expect("infinite upper"),
            parse(&std::format!("(,{hi})"), kind).expect("infinite lower"),
            parse("(,)", kind).expect("both infinite"),
        ]
    }

    fn all_kind_shapes() -> Vec<RangeVal> {
        let mut out = shapes(RangeKind::Int, "1", "10");
        out.extend(shapes(RangeKind::Num, "1.50", "3.5"));
        out.extend(shapes(RangeKind::Date, "2024-01-01", "2024-02-01"));
        out.extend(shapes(
            RangeKind::Ts,
            "2024-01-01 00:00:00",
            "2024-01-02 12:30:00",
        ));
        out.extend(shapes(
            RangeKind::TsTz,
            "2024-01-01 00:00:00+00",
            "2024-01-02 12:30:00+00",
        ));
        out
    }

    #[test]
    fn encode_decode_round_trips_every_kind_and_shape() {
        for r in all_kind_shapes() {
            let bytes = encode(&r);
            let (back, pos) = decode(&bytes, 0, r.kind).expect("decodes");
            assert_eq!(back, r, "round-trip changed {}", r.format());
            assert_eq!(pos, bytes.len(), "decode consumed the wrong length");
            // Decoding at an offset must report the position after this value, not from zero.
            let mut padded = vec![0xAA, 0xBB];
            padded.extend_from_slice(&bytes);
            let (back, pos) = decode(&padded, 2, r.kind).expect("decodes at an offset");
            assert_eq!(back, r);
            assert_eq!(pos, padded.len());
        }
    }

    #[test]
    fn bound_size_matches_the_bytes_encode_writes() {
        for r in all_kind_shapes() {
            let n_bounds = usize::from(r.lower.is_some()) + usize::from(r.upper.is_some());
            let bytes = encode(&r);
            assert_eq!(
                bytes.len(),
                1 + n_bounds * bound_size(r.kind),
                "bound_size disagrees with encode for {}",
                r.format()
            );
            // The spill codec sizes its read from the flags byte alone; it must agree too.
            assert_eq!(payload_len(bytes[0], r.kind), bytes.len() - 1);
        }
    }

    #[test]
    fn truncated_or_inconsistent_bytes_decode_without_panicking() {
        for r in all_kind_shapes() {
            let bytes = encode(&r);
            for cut in 0..bytes.len() {
                // Every short prefix must report absence, never panic or index out of bounds.
                assert!(decode(&bytes[..cut], 0, r.kind).is_none());
            }
        }
        // An inclusive flag on an infinite bound is not a shape `encode` writes; a damaged page
        // must not yield a range that formats as `[,5)`.
        let r = parse("(,10)", RangeKind::Int).expect("infinite lower");
        let mut bytes = encode(&r);
        bytes[0] |= FLAG_LOWER_INC | FLAG_UPPER_INC;
        let (back, _) = decode(&bytes, 0, RangeKind::Int).expect("decodes");
        assert!(!back.lower_inc);
        assert!(back.upper_inc); // the upper bound is finite, so its flag is honoured
        assert_eq!(back.format(), "(,10]");
    }

    #[test]
    fn hash_bytes_separate_kinds_and_unify_numeric_scale() {
        // Equal micros under two kinds share a bound layout — the kind must keep them apart.
        let ts =
            parse("[2024-01-01 00:00:00,2024-01-02 00:00:00)", RangeKind::Ts).expect("tsrange");
        let tstz = parse(
            "[2024-01-01 00:00:00+00,2024-01-02 00:00:00+00)",
            RangeKind::TsTz,
        )
        .expect("tstzrange");
        assert_ne!(hash_bytes(&ts), hash_bytes(&tstz));
        assert_ne!(
            hash_bytes(&RangeVal::empty(RangeKind::Int)),
            hash_bytes(&RangeVal::empty(RangeKind::Date))
        );
        // Scale-equal numeric bounds must land in the same bucket, matching `=`.
        let a = parse("[1.5,3.5)", RangeKind::Num).expect("numrange");
        let b = parse("[1.50,3.500)", RangeKind::Num).expect("numrange, trailing zeros");
        assert_eq!(hash_bytes(&a), hash_bytes(&b));
        assert_ne!(encode(&a), encode(&b)); // storage keeps the original scale
    }

    #[test]
    fn canonicalization_at_the_element_maximum_never_widens_the_range() {
        // No element sits above the maximum, so an exclusive lower bound there holds nothing.
        let r = parse(&std::format!("({},)", i64::MAX), RangeKind::Int).expect("parses");
        assert!(r.empty, "got {}", r.format());
        // A date bound is spelled as text, so build the extreme one directly.
        let d = RangeVal::new(
            RangeKind::Date,
            Some(ast::Value::Date(i32::MAX)),
            None,
            false,
            false,
        )
        .expect("builds");
        assert!(d.empty, "got {}", d.format());
        assert_eq!(
            RangeVal::new(
                RangeKind::Date,
                Some(ast::Value::Date(0)),
                Some(ast::Value::Date(i32::MAX)),
                true,
                true
            ),
            None
        );
        // An inclusive upper bound at the maximum has no half-open form — reject, never widen to
        // unbounded above.
        assert_eq!(
            parse(&std::format!("[1,{}]", i64::MAX), RangeKind::Int),
            None
        );
        // One below the maximum still canonicalizes normally.
        let r = parse(&std::format!("[1,{}]", i64::MAX - 1), RangeKind::Int).expect("parses");
        assert_eq!(r.upper, Some(int(i64::MAX)));
        assert!(!r.upper_inc);
    }

    #[test]
    fn a_bound_of_the_wrong_element_type_is_rejected() {
        assert_eq!(
            RangeVal::new(
                RangeKind::Int,
                Some(ast::Value::Text("1".to_owned())),
                Some(int(10)),
                true,
                false
            ),
            None
        );
        assert_eq!(
            RangeVal::new(RangeKind::Date, Some(int(1)), None, true, false),
            None
        );
    }

    #[test]
    fn contains_and_overlaps() {
        let r = parse("[1,10)", RangeKind::Int).unwrap();
        assert!(r.contains_elem(&int(1)));
        assert!(r.contains_elem(&int(9)));
        assert!(!r.contains_elem(&int(10))); // exclusive upper
        assert!(!r.contains_elem(&int(0)));
        assert!(!RangeVal::empty(RangeKind::Int).contains_elem(&int(5)));

        let a = parse("[1,10)", RangeKind::Int).unwrap();
        let b = parse("[5,20)", RangeKind::Int).unwrap();
        let c = parse("[10,20)", RangeKind::Int).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c)); // [1,10) and [10,20) are adjacent, not overlapping
        assert!(!a.overlaps(&RangeVal::empty(RangeKind::Int)));
    }
}
