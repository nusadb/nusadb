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
    /// bound strictly exceeds the upper (an invalid range, which the reference engine rejects); equal
    /// bounds are allowed and may canonicalize to a single point or to empty.
    #[must_use]
    pub fn new(
        kind: RangeKind,
        lower: Option<ast::Value>,
        upper: Option<ast::Value>,
        lower_inc: bool,
        upper_inc: bool,
    ) -> Option<Self> {
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
        r.canonicalize();
        Some(r)
    }

    /// Normalize a discrete range to `[)` and detect emptiness.
    fn canonicalize(&mut self) {
        if self.empty {
            return;
        }
        if self.kind.is_discrete() {
            // Exclusive lower `(x` → inclusive `[x+1`; inclusive upper `y]` → exclusive `y+1)`.
            if let (Some(v), false) = (&self.lower, self.lower_inc) {
                self.lower = increment(v);
                self.lower_inc = true;
            }
            if let (Some(v), true) = (&self.upper, self.upper_inc) {
                self.upper = increment(v);
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
