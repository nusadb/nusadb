//! Geometric types `point` and `box`: parsing, canonical formatting, operators, and functions.
//!
//! A geometric value is a [`GeomVal`] — either a `point` (an `(x, y)` pair) or an axis-aligned
//! `box` (two opposite corners). Values persist as their canonical text form; the column's
//! [`GeomKind`] tells the reader which shape to parse back into. Every function here is total: parse
//! entry points return [`Option`] (a syntax error is `None`), and coordinate access is checked, so
//! no path indexes or unwraps.
//!
//! Coordinates are IEEE-754 doubles formatted by their shortest round-trip decimal, with a negative
//! zero normalized to `0`, matching the reference engine's canonical rendering for ordinary values.

use nusadb_core::engine::GeomKind;

/// A geometric value.
///
/// A `box` is kept normalized so its first corner is the upper-right (the per-axis maximum) and its
/// second corner is the lower-left (the per-axis minimum); [`GeomVal::make_box`] enforces this.
#[derive(Debug, Clone, PartialEq)]
pub enum GeomVal {
    /// A `point` — an `(x, y)` coordinate pair.
    Point {
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
    },
    /// A `box` — an axis-aligned rectangle. `high` is the upper-right corner (per-axis maximum),
    /// `low` is the lower-left corner (per-axis minimum).
    Box {
        /// Upper-right X (the larger of the two input X coordinates).
        high_x: f64,
        /// Upper-right Y (the larger of the two input Y coordinates).
        high_y: f64,
        /// Lower-left X (the smaller of the two input X coordinates).
        low_x: f64,
        /// Lower-left Y (the smaller of the two input Y coordinates).
        low_y: f64,
    },
}

impl GeomVal {
    /// The [`GeomKind`] of this value.
    #[must_use]
    pub const fn kind(&self) -> GeomKind {
        match self {
            Self::Point { .. } => GeomKind::Point,
            Self::Box { .. } => GeomKind::Box,
        }
    }

    /// Build a `point`.
    #[must_use]
    pub const fn point(x: f64, y: f64) -> Self {
        Self::Point { x, y }
    }

    /// Build a normalized `box` from two opposite corners, so the stored `high` corner is the
    /// per-axis maximum and `low` the per-axis minimum. Both `((1,1),(3,3))` and `((3,3),(1,1))`
    /// yield the same box.
    #[must_use]
    pub const fn make_box(ax: f64, ay: f64, bx: f64, by: f64) -> Self {
        Self::Box {
            high_x: ax.max(bx),
            high_y: ay.max(by),
            low_x: ax.min(bx),
            low_y: ay.min(by),
        }
    }
}

/// Format a coordinate by its shortest round-trip decimal, normalizing a negative zero to `0`.
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the intended test: it maps a negative zero (which equals +0.0) to a positive zero, leaving every other value untouched"
)]
fn fmt_num(v: f64) -> String {
    // `-0.0 == 0.0` is true, so this maps a negative zero to a positive zero while leaving every
    // other value untouched; the `{}` formatter then drops a trailing `.0` (e.g. `5`, `-2.5`).
    let v = if v == 0.0 { 0.0 } else { v };
    format!("{v}")
}

/// Render a value in its canonical text form: a `point` as `(x,y)`, a `box` as `(hx,hy),(lx,ly)`.
#[must_use]
pub fn format(v: &GeomVal) -> String {
    match v {
        GeomVal::Point { x, y } => format!("({},{})", fmt_num(*x), fmt_num(*y)),
        GeomVal::Box {
            high_x,
            high_y,
            low_x,
            low_y,
        } => format!(
            "({},{}),({},{})",
            fmt_num(*high_x),
            fmt_num(*high_y),
            fmt_num(*low_x),
            fmt_num(*low_y),
        ),
    }
}

/// Extract the finite floating-point tokens from a geometric literal, treating parentheses,
/// commas, and whitespace as separators. Returns `None` if any token fails to parse. A minus sign
/// and exponent stay attached to their number since only `(`, `)`, `,`, and whitespace separate
/// tokens.
fn floats(s: &str) -> Option<Vec<f64>> {
    let cleaned: String = s
        .chars()
        .map(|c| if c == '(' || c == ')' { ' ' } else { c })
        .collect();
    cleaned
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f64>().ok().filter(|f| f.is_finite()))
        .collect()
}

/// Parse a `point` literal: `(x,y)`, `( x , y )` (spaces ignored), or `x,y` (no parentheses).
/// Requires exactly two coordinates.
#[must_use]
pub fn parse_point(s: &str) -> Option<GeomVal> {
    match floats(s)?.as_slice() {
        &[x, y] => Some(GeomVal::point(x, y)),
        _ => None,
    }
}

/// Parse a `box` literal: `(x1,y1),(x2,y2)`, `((x1,y1),(x2,y2))`, or `x1,y1,x2,y2`. Requires exactly
/// four coordinates; the result is normalized (see [`GeomVal::make_box`]).
#[must_use]
pub fn parse_box(s: &str) -> Option<GeomVal> {
    match floats(s)?.as_slice() {
        &[ax, ay, bx, by] => Some(GeomVal::make_box(ax, ay, bx, by)),
        _ => None,
    }
}

/// Parse a geometric literal against the target [`GeomKind`].
#[must_use]
pub fn parse(s: &str, kind: GeomKind) -> Option<GeomVal> {
    match kind {
        GeomKind::Point => parse_point(s),
        GeomKind::Box => parse_box(s),
    }
}

// ── point operators ────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two points (`p1 <-> p2`).
#[must_use]
pub fn point_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (ax - bx).hypot(ay - by)
}

/// Vector addition `p1 + p2`.
#[must_use]
pub const fn point_add(ax: f64, ay: f64, bx: f64, by: f64) -> GeomVal {
    GeomVal::point(ax + bx, ay + by)
}

/// Vector subtraction `p1 - p2`.
#[must_use]
pub const fn point_sub(ax: f64, ay: f64, bx: f64, by: f64) -> GeomVal {
    GeomVal::point(ax - bx, ay - by)
}

/// Complex multiplication `p1 * p2`, treating each point as `x + y·i`:
/// `(a+bi)(c+di) = (ac − bd) + (ad + bc)i`.
#[must_use]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) multiply-then-add matches the reference engine's IEEE-754 result bit-for-bit; a fused mul_add would round differently"
)]
pub fn point_mul(ax: f64, ay: f64, bx: f64, by: f64) -> GeomVal {
    GeomVal::point(ax * bx - ay * by, ax * by + ay * bx)
}

/// Complex division `p1 / p2`, treating each point as `x + y·i`:
/// `(a+bi)/(c+di) = ((ac + bd) + (bc − ad)i) / (c² + d²)`. Division by the zero point (`c² + d² = 0`)
/// yields `None`.
#[must_use]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) arithmetic matches the reference engine's IEEE-754 result bit-for-bit; a fused mul_add would round differently"
)]
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the division-by-zero test the reference engine uses for the zero divisor point"
)]
pub fn point_div(ax: f64, ay: f64, bx: f64, by: f64) -> Option<GeomVal> {
    let denom = bx * bx + by * by;
    if denom == 0.0 {
        return None;
    }
    Some(GeomVal::point(
        (ax * bx + ay * by) / denom,
        (ay * bx - ax * by) / denom,
    ))
}

// ── box functions & operators ────────────────────────────────────────────────────────────────

/// Width of a box (`high_x − low_x`).
#[must_use]
pub const fn box_width(high_x: f64, low_x: f64) -> f64 {
    high_x - low_x
}

/// Height of a box (`high_y − low_y`).
#[must_use]
pub const fn box_height(high_y: f64, low_y: f64) -> f64 {
    high_y - low_y
}

/// Area of a box (`width × height`).
#[must_use]
pub const fn box_area(high_x: f64, high_y: f64, low_x: f64, low_y: f64) -> f64 {
    (high_x - low_x) * (high_y - low_y)
}

/// Center (midpoint) of a box.
#[must_use]
#[allow(
    clippy::manual_midpoint,
    reason = "plain `(a + b) / 2` matches the reference engine's center rounding; `f64::midpoint` rounds differently for large magnitudes"
)]
pub fn box_center(high_x: f64, high_y: f64, low_x: f64, low_y: f64) -> GeomVal {
    GeomVal::point((high_x + low_x) / 2.0, (high_y + low_y) / 2.0)
}

/// Whether a box contains a point (`box @> point`), inclusive of the boundary.
#[must_use]
pub fn box_contains_point(
    high_x: f64,
    high_y: f64,
    low_x: f64,
    low_y: f64,
    px: f64,
    py: f64,
) -> bool {
    px >= low_x && px <= high_x && py >= low_y && py <= high_y
}

/// Whether two boxes overlap (`box1 && box2`), inclusive of a shared edge or corner.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the eight coordinates are the two boxes' corners; grouping them into structs would only re-spell what the flat coordinate list already states"
)]
pub fn box_overlap(
    a_high_x: f64,
    a_high_y: f64,
    a_low_x: f64,
    a_low_y: f64,
    b_high_x: f64,
    b_high_y: f64,
    b_low_x: f64,
    b_low_y: f64,
) -> bool {
    a_low_x <= b_high_x && b_low_x <= a_high_x && a_low_y <= b_high_y && b_low_y <= a_high_y
}

/// Distance between two boxes (`box1 <-> box2`): the Euclidean distance between their centers.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the eight coordinates are the two boxes' corners; grouping them into structs would only re-spell what the flat coordinate list already states"
)]
#[allow(
    clippy::manual_midpoint,
    reason = "plain `(a + b) / 2` matches the reference engine's center rounding; `f64::midpoint` rounds differently for large magnitudes"
)]
pub fn box_distance(
    a_high_x: f64,
    a_high_y: f64,
    a_low_x: f64,
    a_low_y: f64,
    b_high_x: f64,
    b_high_y: f64,
    b_low_x: f64,
    b_low_y: f64,
) -> f64 {
    let acx = (a_high_x + a_low_x) / 2.0;
    let acy = (a_high_y + a_low_y) / 2.0;
    let bcx = (b_high_x + b_low_x) / 2.0;
    let bcy = (b_high_y + b_low_y) / 2.0;
    point_distance(acx, acy, bcx, bcy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_parse_forms() {
        assert_eq!(parse_point("(1,2)"), Some(GeomVal::point(1.0, 2.0)));
        assert_eq!(
            parse_point("( 1.5 , -2.5 )"),
            Some(GeomVal::point(1.5, -2.5))
        );
        assert_eq!(parse_point("3,4"), Some(GeomVal::point(3.0, 4.0)));
        assert_eq!(parse_point("abc"), None);
        assert_eq!(parse_point("(1,2,3)"), None);
        assert_eq!(parse_point("(1)"), None);
    }

    #[test]
    fn point_format_canonical() {
        assert_eq!(format(&GeomVal::point(1.5, -2.5)), "(1.5,-2.5)");
        assert_eq!(format(&GeomVal::point(-0.0, 0.0)), "(0,0)");
        assert_eq!(format(&GeomVal::point(3.0, 4.0)), "(3,4)");
    }

    #[test]
    fn box_normalizes() {
        let a = parse_box("(1,1),(3,3)").expect("valid");
        let b = parse_box("(3,3),(1,1)").expect("valid");
        assert_eq!(a, b);
        assert_eq!(format(&a), "(3,3),(1,1)");
        assert_eq!(
            format(&parse_box("(1,3),(3,1)").expect("valid")),
            "(3,3),(1,1)"
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued results compare exactly"
    )]
    fn point_arithmetic() {
        assert_eq!(point_add(1.0, 2.0, 3.0, 4.0), GeomVal::point(4.0, 6.0));
        assert_eq!(point_sub(1.0, 2.0, 3.0, 4.0), GeomVal::point(-2.0, -2.0));
        assert_eq!(point_mul(1.0, 2.0, 3.0, 4.0), GeomVal::point(-5.0, 10.0));
        assert_eq!(
            point_div(1.0, 2.0, 3.0, 4.0),
            Some(GeomVal::point(0.44, 0.08))
        );
        assert_eq!(point_div(1.0, 2.0, 0.0, 0.0), None);
        assert_eq!(point_distance(0.0, 0.0, 3.0, 4.0), 5.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued results compare exactly"
    )]
    fn box_ops() {
        assert_eq!(box_area(2.0, 3.0, 0.0, 0.0), 6.0);
        assert_eq!(box_center(2.0, 4.0, 0.0, 0.0), GeomVal::point(1.0, 2.0));
        assert!(box_contains_point(4.0, 4.0, 0.0, 0.0, 0.0, 0.0));
        assert!(box_contains_point(4.0, 4.0, 0.0, 0.0, 4.0, 2.0));
        assert!(!box_contains_point(4.0, 4.0, 0.0, 0.0, 5.0, 2.0));
        assert!(box_overlap(2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 1.0, 1.0));
        assert!(!box_overlap(2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 3.0, 3.0));
        assert_eq!(box_distance(2.0, 2.0, 0.0, 0.0, 12.0, 2.0, 10.0, 0.0), 10.0);
    }
}
