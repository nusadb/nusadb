//! Geometric types `point`, `box`, `circle`, `lseg`, and `line`: parsing, canonical formatting,
//! operators, and functions.
//!
//! A geometric value is a [`GeomVal`] — a `point` (an `(x, y)` pair), an axis-aligned `box` (two
//! opposite corners), a `circle` (a center `(x, y)` and a radius `r`), an `lseg` (a line segment
//! between two endpoints), or a `line` (an infinite line `A·x + B·y + C = 0`). Values persist as
//! their canonical text form; the column's [`GeomKind`] tells the reader which shape to parse back
//! into. Every function here is total: parse
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
    /// A `circle` — a center point and a non-negative radius.
    Circle {
        /// Center X coordinate.
        cx: f64,
        /// Center Y coordinate.
        cy: f64,
        /// Radius (never negative; a zero radius is a degenerate point-circle).
        r: f64,
    },
    /// An `lseg` — a line segment between two endpoints. Unlike a `box`, the endpoint order is
    /// preserved exactly as entered (no normalization).
    Lseg {
        /// First endpoint X coordinate.
        x1: f64,
        /// First endpoint Y coordinate.
        y1: f64,
        /// Second endpoint X coordinate.
        x2: f64,
        /// Second endpoint Y coordinate.
        y2: f64,
    },
    /// A `line` — an infinite line `A·x + B·y + C = 0`, carried as its three coefficients. The
    /// three-coefficient input form `{A,B,C}` is stored verbatim (no normalization).
    Line {
        /// Coefficient `A`.
        a: f64,
        /// Coefficient `B`.
        b: f64,
        /// Coefficient `C`.
        c: f64,
    },
}

impl GeomVal {
    /// The [`GeomKind`] of this value.
    #[must_use]
    pub const fn kind(&self) -> GeomKind {
        match self {
            Self::Point { .. } => GeomKind::Point,
            Self::Box { .. } => GeomKind::Box,
            Self::Circle { .. } => GeomKind::Circle,
            Self::Lseg { .. } => GeomKind::Lseg,
            Self::Line { .. } => GeomKind::Line,
        }
    }

    /// Build a `point`.
    #[must_use]
    pub const fn point(x: f64, y: f64) -> Self {
        Self::Point { x, y }
    }

    /// Build a `circle` from a center and a radius.
    #[must_use]
    pub const fn circle(cx: f64, cy: f64, r: f64) -> Self {
        Self::Circle { cx, cy, r }
    }

    /// Build an `lseg` from its two endpoints, preserving the given order (no normalization).
    #[must_use]
    pub const fn lseg(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self::Lseg { x1, y1, x2, y2 }
    }

    /// Build a `line` from its three coefficients `A·x + B·y + C = 0`, stored verbatim.
    #[must_use]
    pub const fn line(a: f64, b: f64, c: f64) -> Self {
        Self::Line { a, b, c }
    }

    /// Build a `line` through two distinct points, matching the reference engine's `line_construct`:
    /// a vertical line (`x1 == x2`) becomes `A=-1, B=0, C=x1`; a horizontal line (`y1 == y2`) becomes
    /// `A=0, B=-1, C=y1`; otherwise `A=(y2−y1)/(x2−x1)`, `B=-1`, `C=y1 − A·x1`.
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "an exact `== ` is the vertical/horizontal test the reference engine's line_construct uses for the two degenerate slopes"
    )]
    #[allow(
        clippy::suboptimal_flops,
        reason = "plain (non-fused) multiply-then-subtract matches the reference engine's IEEE-754 coefficient result bit-for-bit; a fused mul_add would round differently"
    )]
    pub fn line_construct(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        if x1 == x2 {
            Self::line(-1.0, 0.0, x1)
        } else if y1 == y2 {
            Self::line(0.0, -1.0, y1)
        } else {
            let a = (y2 - y1) / (x2 - x1);
            Self::line(a, -1.0, y1 - a * x1)
        }
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

/// Render a value in its canonical text form: a `point` as `(x,y)`, a `box` as `(hx,hy),(lx,ly)`, a
/// `circle` as `<(cx,cy),r>`, an `lseg` as `[(x1,y1),(x2,y2)]`, a `line` as `{A,B,C}`.
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
        GeomVal::Circle { cx, cy, r } => {
            format!("<({},{}),{}>", fmt_num(*cx), fmt_num(*cy), fmt_num(*r))
        },
        GeomVal::Lseg { x1, y1, x2, y2 } => format!(
            "[({},{}),({},{})]",
            fmt_num(*x1),
            fmt_num(*y1),
            fmt_num(*x2),
            fmt_num(*y2),
        ),
        GeomVal::Line { a, b, c } => {
            format!("{{{},{},{}}}", fmt_num(*a), fmt_num(*b), fmt_num(*c))
        },
    }
}

/// Extract the finite floating-point tokens from a geometric literal, treating parentheses, the
/// circle delimiters `<`/`>`, the lseg brackets `[`/`]`, the line braces `{`/`}`, commas, and
/// whitespace as separators. Returns `None` if any token fails to parse. A minus sign and exponent
/// stay attached to their number since only `(`, `)`, `<`, `>`, `[`, `]`, `{`, `}`, `,`, and
/// whitespace separate tokens.
fn floats(s: &str) -> Option<Vec<f64>> {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if matches!(c, '(' | ')' | '<' | '>' | '[' | ']' | '{' | '}') {
                ' '
            } else {
                c
            }
        })
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

/// Parse a `circle` literal: `<(x,y),r>`, `((x,y),r)`, or `x,y,r`.
///
/// Requires exactly three coordinates (center X, center Y, radius); a negative radius is rejected
/// (`None`), matching the reference engine's `invalid input syntax for type circle`. A zero radius
/// is valid.
#[must_use]
pub fn parse_circle(s: &str) -> Option<GeomVal> {
    match floats(s)?.as_slice() {
        &[cx, cy, r] if r >= 0.0 => Some(GeomVal::circle(cx, cy, r)),
        _ => None,
    }
}

/// Parse an `lseg` literal: `[(x1,y1),(x2,y2)]`, `((x1,y1),(x2,y2))`, or `x1,y1,x2,y2`.
///
/// Requires exactly four coordinates (the two endpoints). Unlike a `box`, the endpoint order is
/// preserved exactly (no normalization), so `[(3,4),(1,2)]` stays `[(3,4),(1,2)]`.
#[must_use]
pub fn parse_lseg(s: &str) -> Option<GeomVal> {
    match floats(s)?.as_slice() {
        &[x1, y1, x2, y2] => Some(GeomVal::lseg(x1, y1, x2, y2)),
        _ => None,
    }
}

/// Parse a `line` literal into the infinite line `A·x + B·y + C = 0`.
///
/// Two spellings are accepted, dispatched by coordinate count:
/// - three coordinates — the `{A,B,C}` coefficient form — are stored verbatim (no normalization,
///   matching the reference engine: `{2,-2,0}` stays `{2,-2,0}`);
/// - four coordinates — a two-point form `[(x1,y1),(x2,y2)]`, `((x1,y1),(x2,y2))`, or bare
///   `x1,y1,x2,y2` — construct the line through those points (see [`GeomVal::line_construct`]).
///
/// Any other count yields `None`.
///
/// A degenerate line is rejected (`None`), matching the reference engine: the `{A,B,C}` form with
/// `A` and `B` both zero is not a line, and the two-point form with coincident points has no unique
/// line through it.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "the exact `== 0.0` and point-equality tests are the degenerate-line rejections the reference engine makes: `{0,0,C}` (A and B both zero) and a two-point form whose points coincide are both invalid line specifications"
)]
pub fn parse_line(s: &str) -> Option<GeomVal> {
    match *floats(s)?.as_slice() {
        // `{A,B,C}` coefficient form — reject `A` and `B` both zero (not a line).
        [a, b, c] => (!(a == 0.0 && b == 0.0)).then(|| GeomVal::line(a, b, c)),
        // Two-point form — reject two coincident points (no unique line through a single point).
        [x1, y1, x2, y2] => {
            (!(x1 == x2 && y1 == y2)).then(|| GeomVal::line_construct(x1, y1, x2, y2))
        },
        _ => None,
    }
}

/// Parse a geometric literal against the target [`GeomKind`].
#[must_use]
pub fn parse(s: &str, kind: GeomKind) -> Option<GeomVal> {
    match kind {
        GeomKind::Point => parse_point(s),
        GeomKind::Box => parse_box(s),
        GeomKind::Circle => parse_circle(s),
        GeomKind::Lseg => parse_lseg(s),
        GeomKind::Line => parse_line(s),
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

// ── circle functions & operators ─────────────────────────────────────────────────────────────

/// Area of a circle (`π · r²`).
#[must_use]
pub fn circle_area(r: f64) -> f64 {
    std::f64::consts::PI * r * r
}

/// Diameter of a circle (`2 · r`).
#[must_use]
pub const fn circle_diameter(r: f64) -> f64 {
    2.0 * r
}

/// Distance from a circle to a point (`circle <-> point`): the gap between the point and the
/// circle's boundary, or `0` when the point is inside or on the circle.
#[must_use]
pub fn circle_distance_point(cx: f64, cy: f64, r: f64, px: f64, py: f64) -> f64 {
    (point_distance(cx, cy, px, py) - r).max(0.0)
}

/// Distance between two circles (`circle <-> circle`): the gap between their boundaries, or `0` when
/// they touch or overlap.
#[must_use]
pub fn circle_distance_circle(acx: f64, acy: f64, ar: f64, bcx: f64, bcy: f64, br: f64) -> f64 {
    (point_distance(acx, acy, bcx, bcy) - ar - br).max(0.0)
}

/// Whether two circles overlap (`circle && circle`), inclusive of a single touching point: the
/// distance between centers is at most the sum of the radii.
#[must_use]
pub fn circle_overlap(acx: f64, acy: f64, ar: f64, bcx: f64, bcy: f64, br: f64) -> bool {
    point_distance(acx, acy, bcx, bcy) <= ar + br
}

/// Whether a circle contains a point (`circle @> point`), inclusive of the boundary: the point is
/// within the radius of the center.
#[must_use]
pub fn circle_contains_point(cx: f64, cy: f64, r: f64, px: f64, py: f64) -> bool {
    point_distance(cx, cy, px, py) <= r
}

/// Whether the `outer` circle contains the `inner` circle (`outer @> inner`), inclusive of the
/// boundary: the distance between centers plus the inner radius does not exceed the outer radius.
#[must_use]
pub fn circle_contains_circle(ocx: f64, ocy: f64, or: f64, icx: f64, icy: f64, ir: f64) -> bool {
    point_distance(ocx, ocy, icx, icy) + ir <= or
}

// ── lseg functions & operators ─────────────────────────────────────────────────────────────────

/// Length of a line segment (`length(lseg)`): the Euclidean distance between its two endpoints.
#[must_use]
pub fn lseg_length(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    point_distance(x1, y1, x2, y2)
}

/// Distance from a line segment to a point (`lseg <-> point`): the distance to the nearest point on
/// the segment.
///
/// The point is projected onto the segment's line; if the foot lies within the segment the
/// perpendicular distance is used, otherwise the distance to the nearer endpoint. A degenerate
/// segment (both endpoints equal) collapses to the point-to-point distance.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the intended test for a degenerate (zero-length) segment, where no projection parameter exists"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) multiply-then-add matches the reference engine's IEEE-754 projection result bit-for-bit; a fused mul_add would round differently"
)]
pub fn lseg_distance_point(x1: f64, y1: f64, x2: f64, y2: f64, px: f64, py: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let denom = dx * dx + dy * dy;
    // Project the point onto the segment; `t` is the clamped parameter of the foot in `[0, 1]`.
    let t = if denom == 0.0 {
        0.0
    } else {
        (((px - x1) * dx + (py - y1) * dy) / denom).clamp(0.0, 1.0)
    };
    point_distance(x1 + t * dx, y1 + t * dy, px, py)
}

/// The intersection point of two line segments (`lseg # lseg`), or `None` when they do not cross at
/// a single point.
///
/// Solves the two-parameter line system `A1 + t·(A2−A1) = B1 + u·(B2−B1)`; the intersection is valid
/// only when both `t` and `u` lie in `[0, 1]`. A zero denominator (the segments are parallel or
/// collinear) yields `None` — collinear overlap is not reported as a point.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the eight coordinates are the two segments' endpoints; grouping them into structs would only re-spell what the flat coordinate list already states"
)]
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the parallel/collinear test the reference engine uses for the zero cross-product denominator"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) cross-product arithmetic matches the reference engine's IEEE-754 result bit-for-bit; a fused mul_add would round differently"
)]
pub fn lseg_intersection(
    ax1: f64,
    ay1: f64,
    ax2: f64,
    ay2: f64,
    bx1: f64,
    by1: f64,
    bx2: f64,
    by2: f64,
) -> Option<GeomVal> {
    let rx = ax2 - ax1;
    let ry = ay2 - ay1;
    let sx = bx2 - bx1;
    let sy = by2 - by1;
    let denom = rx * sy - ry * sx;
    if denom == 0.0 {
        // Parallel or collinear: no single crossing point.
        return None;
    }
    let qpx = bx1 - ax1;
    let qpy = by1 - ay1;
    let t = (qpx * sy - qpy * sx) / denom;
    let u = (qpx * ry - qpy * rx) / denom;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some(GeomVal::point(ax1 + t * rx, ay1 + t * ry))
    } else {
        None
    }
}

/// Distance between two line segments (`lseg <-> lseg`): `0` when they intersect, otherwise the
/// minimum endpoint-to-segment gap.
///
/// The minimum is taken over the four endpoint-to-segment distances (each endpoint of one segment to
/// the other segment). A collinear-overlapping pair falls out as `0` naturally, since an endpoint of
/// one lies on the other segment.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "the eight coordinates are the two segments' endpoints; grouping them into structs would only re-spell what the flat coordinate list already states"
)]
pub fn lseg_distance_lseg(
    ax1: f64,
    ay1: f64,
    ax2: f64,
    ay2: f64,
    bx1: f64,
    by1: f64,
    bx2: f64,
    by2: f64,
) -> f64 {
    if lseg_intersection(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2).is_some() {
        return 0.0;
    }
    let d1 = lseg_distance_point(bx1, by1, bx2, by2, ax1, ay1);
    let d2 = lseg_distance_point(bx1, by1, bx2, by2, ax2, ay2);
    let d3 = lseg_distance_point(ax1, ay1, ax2, ay2, bx1, by1);
    let d4 = lseg_distance_point(ax1, ay1, ax2, ay2, bx2, by2);
    d1.min(d2).min(d3).min(d4)
}

// ── line functions & operators ───────────────────────────────────────────────────────────────────

/// Perpendicular distance from an infinite line `A·x + B·y + C = 0` to a point (`line <-> point`).
///
/// The foot of the perpendicular (the closest point on the line) is found first, then the actual
/// distance to it is taken as the hypotenuse of the coordinate deltas. This matches the reference
/// engine's point↔line path bit-for-bit — which finds the foot then measures the deltas, and differs
/// in the last ULP from the algebraic `|A·px + B·py + C| / hypot(A, B)` closed form on some inputs.
#[must_use]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) arithmetic matches the reference engine's IEEE-754 foot-of-perpendicular result bit-for-bit; a fused mul_add would round differently"
)]
pub fn line_distance_point(a: f64, b: f64, c: f64, px: f64, py: f64) -> f64 {
    // `k` scales the line normal `(A, B)` to reach the point; subtracting it lands on the foot.
    let k = (a * px + b * py + c) / (a * a + b * b);
    let foot_x = px - a * k;
    let foot_y = py - b * k;
    (px - foot_x).hypot(py - foot_y)
}

/// The intersection point of two infinite lines (`line # line`), or `None` when they are parallel.
///
/// With `denom = A1·B2 − A2·B1`: a zero denominator means the lines are parallel (or identical) and
/// yields `None`; otherwise the crossing point is
/// `x = (B1·C2 − B2·C1)/denom`, `y = (A2·C1 − A1·C2)/denom`.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the parallel test the reference engine uses for the zero cross-coefficient denominator"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) cross-product arithmetic matches the reference engine's IEEE-754 result bit-for-bit; a fused mul_add would round differently"
)]
pub fn line_intersection(a1: f64, b1: f64, c1: f64, a2: f64, b2: f64, c2: f64) -> Option<GeomVal> {
    let denom = a1 * b2 - a2 * b1;
    if denom == 0.0 {
        // Parallel or identical lines: no single crossing point.
        return None;
    }
    Some(GeomVal::point(
        (b1 * c2 - b2 * c1) / denom,
        (a2 * c1 - a1 * c2) / denom,
    ))
}

/// Distance between two infinite lines (`line <-> line`): `0` when they cross (a non-zero
/// `A1·B2 − A2·B1` denominator), otherwise the constant perpendicular gap between the parallel lines.
///
/// For a parallel pair a point on the first line is taken (`(0, −C1/B1)` when `B1 ≠ 0`, else
/// `(−C1/A1, 0)`) and its perpendicular distance to the second line is returned by the algebraic
/// `|A2·px + B2·py + C2| / hypot(A2, B2)` closed form; this is scale-invariant, so two parallel lines
/// with differently scaled coefficients still report the same gap. (Unlike the point↔line operator,
/// the reference engine's line↔line path uses this closed form directly rather than the foot of the
/// perpendicular, so the two paths can differ in the last ULP — this reproduces the line↔line one.)
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "an exact `== 0.0` is the crossing/parallel test (denominator) and the vertical-line test (B1) the reference engine uses"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "plain (non-fused) cross-coefficient arithmetic matches the reference engine's IEEE-754 denominator and closed-form distance bit-for-bit; a fused mul_add would round differently"
)]
pub fn line_distance_line(a1: f64, b1: f64, c1: f64, a2: f64, b2: f64, c2: f64) -> f64 {
    let denom = a1 * b2 - a2 * b1;
    if denom != 0.0 {
        // The lines cross.
        return 0.0;
    }
    // Parallel: measure a point on line 1 against line 2. Take `(0, −C1/B1)` for a non-vertical line,
    // else (`B1 == 0`, a vertical line) the point `(−C1/A1, 0)`.
    let (px, py) = if b1 == 0.0 {
        (-c1 / a1, 0.0)
    } else {
        (0.0, -c1 / b1)
    };
    // The closed-form perpendicular distance — the reference engine's line↔line path (not the foot).
    (a2 * px + b2 * py + c2).abs() / a2.hypot(b2)
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

    #[test]
    fn circle_parse_forms() {
        assert_eq!(
            parse_circle("<(1,2),3>"),
            Some(GeomVal::circle(1.0, 2.0, 3.0))
        );
        assert_eq!(
            parse_circle("((1,2),3)"),
            Some(GeomVal::circle(1.0, 2.0, 3.0))
        );
        assert_eq!(parse_circle("4,5,6"), Some(GeomVal::circle(4.0, 5.0, 6.0)));
        // A zero radius is valid; a negative radius is rejected.
        assert_eq!(
            parse_circle("<(0,0),0>"),
            Some(GeomVal::circle(0.0, 0.0, 0.0))
        );
        assert_eq!(parse_circle("<(0,0),-1>"), None);
        assert_eq!(parse_circle("abc"), None);
        assert_eq!(parse_circle("<(1,2)>"), None);
        assert_eq!(parse_circle("<(1,2,3),4>"), None);
    }

    #[test]
    fn circle_format_canonical() {
        assert_eq!(format(&GeomVal::circle(1.5, 2.5, 3.25)), "<(1.5,2.5),3.25>");
        assert_eq!(format(&GeomVal::circle(0.0, 0.0, 0.0)), "<(0,0),0>");
        assert_eq!(format(&GeomVal::circle(-0.0, -0.0, 2.0)), "<(0,0),2>");
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued and power-of-two results compare exactly"
    )]
    fn circle_ops() {
        assert_eq!(circle_area(2.0), std::f64::consts::PI * 4.0);
        assert_eq!(circle_diameter(3.0), 6.0);
        // circle <-> point: gap to boundary, clamped to 0 when the point is inside.
        assert_eq!(circle_distance_point(0.0, 0.0, 1.0, 5.0, 0.0), 4.0);
        assert_eq!(circle_distance_point(0.0, 0.0, 5.0, 1.0, 1.0), 0.0);
        // circle <-> circle: gap between boundaries, clamped to 0 when overlapping.
        assert_eq!(circle_distance_circle(0.0, 0.0, 1.0, 5.0, 0.0, 1.0), 3.0);
        assert_eq!(circle_distance_circle(0.0, 0.0, 3.0, 4.0, 0.0, 3.0), 0.0);
        // Overlap.
        assert!(circle_overlap(0.0, 0.0, 2.0, 3.0, 0.0, 2.0));
        assert!(!circle_overlap(0.0, 0.0, 2.0, 5.0, 0.0, 2.0));
        // Contains point (inclusive) and contains circle.
        assert!(circle_contains_point(0.0, 0.0, 5.0, 3.0, 4.0));
        assert!(!circle_contains_point(0.0, 0.0, 5.0, 3.0, 5.0));
        assert!(circle_contains_circle(0.0, 0.0, 5.0, 0.0, 0.0, 2.0));
        assert!(!circle_contains_circle(0.0, 0.0, 5.0, 4.0, 0.0, 2.0));
    }

    #[test]
    fn lseg_parse_forms() {
        // All three input spellings parse identically.
        assert_eq!(
            parse_lseg("[(1,2),(3,4)]"),
            Some(GeomVal::lseg(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            parse_lseg("((1,2),(3,4))"),
            Some(GeomVal::lseg(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            parse_lseg("1,2,3,4"),
            Some(GeomVal::lseg(1.0, 2.0, 3.0, 4.0))
        );
        // Endpoint order is preserved (no normalization).
        assert_eq!(
            parse_lseg("[(3,4),(1,2)]"),
            Some(GeomVal::lseg(3.0, 4.0, 1.0, 2.0))
        );
        assert_eq!(parse_lseg("abc"), None);
        assert_eq!(parse_lseg("[(1,2),(3,4),(5,6)]"), None);
        assert_eq!(parse_lseg("[(1,2)]"), None);
    }

    #[test]
    fn lseg_format_canonical() {
        assert_eq!(
            format(&GeomVal::lseg(1.5, 2.5, 3.5, 4.5)),
            "[(1.5,2.5),(3.5,4.5)]"
        );
        // Order-preserving: the reversed endpoints render reversed, unlike a box.
        assert_eq!(format(&GeomVal::lseg(3.0, 4.0, 1.0, 2.0)), "[(3,4),(1,2)]");
        assert_eq!(
            format(&GeomVal::lseg(-0.0, -0.0, 0.0, 0.0)),
            "[(0,0),(0,0)]"
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued and √-of-perfect-square results compare exactly"
    )]
    fn lseg_ops() {
        // length = distance between endpoints.
        assert_eq!(lseg_length(0.0, 0.0, 3.0, 4.0), 5.0);
        assert_eq!(lseg_length(0.0, 0.0, 0.0, 7.0), 7.0);
        assert_eq!(lseg_length(0.0, 0.0, 0.0, 0.0), 0.0);
        // lseg <-> point: perpendicular when the foot is on the segment, else the nearer endpoint.
        assert_eq!(lseg_distance_point(0.0, 0.0, 4.0, 0.0, 6.0, 0.0), 2.0);
        assert_eq!(lseg_distance_point(0.0, 0.0, 4.0, 0.0, 2.0, 3.0), 3.0);
        assert_eq!(lseg_distance_point(0.0, 0.0, 0.0, 4.0, 3.0, 0.0), 3.0);
        // lseg <-> lseg: 0 when they cross, else the endpoint-gap minimum.
        assert_eq!(
            lseg_distance_lseg(0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0),
            0.0
        );
        assert_eq!(
            lseg_distance_lseg(0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 4.0, 3.0),
            3.0
        );
        assert_eq!(
            lseg_distance_lseg(0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0),
            4.0
        );
        // lseg # lseg: the crossing point, a shared endpoint, or None.
        assert_eq!(
            lseg_intersection(0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0),
            Some(GeomVal::point(1.0, 1.0))
        );
        assert_eq!(
            lseg_intersection(0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0),
            Some(GeomVal::point(2.0, 0.0))
        );
        // Collinear overlap and disjoint parallels return no single point.
        assert_eq!(
            lseg_intersection(0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0),
            None
        );
        assert_eq!(
            lseg_intersection(0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 2.0),
            None
        );
    }

    #[test]
    fn line_parse_forms() {
        // The three-coefficient `{A,B,C}` form is stored verbatim (no normalization).
        assert_eq!(parse_line("{1,-1,0}"), Some(GeomVal::line(1.0, -1.0, 0.0)));
        assert_eq!(parse_line("{2,-2,0}"), Some(GeomVal::line(2.0, -2.0, 0.0)));
        // The two-point forms construct via line_construct — bracket, paren, and bare spellings.
        assert_eq!(
            parse_line("[(1,2),(3,4)]"),
            Some(GeomVal::line(1.0, -1.0, 1.0))
        );
        assert_eq!(
            parse_line("((0,0),(1,1))"),
            Some(GeomVal::line(1.0, -1.0, 0.0))
        );
        assert_eq!(parse_line("2,0,2,5"), Some(GeomVal::line(-1.0, 0.0, 2.0)));
        assert_eq!(parse_line("0,3,5,3"), Some(GeomVal::line(0.0, -1.0, 3.0)));
        // A wrong coordinate count or garbage is rejected.
        assert_eq!(parse_line("abc"), None);
        assert_eq!(parse_line("{1,2}"), None);
        assert_eq!(parse_line("{1,2,3,4,5}"), None);
    }

    #[test]
    fn line_construct_oracle() {
        // Verified against the reference engine's line_construct.
        assert_eq!(
            GeomVal::line_construct(1.0, 2.0, 3.0, 4.0),
            GeomVal::line(1.0, -1.0, 1.0)
        );
        assert_eq!(
            GeomVal::line_construct(0.0, 0.0, 1.0, 1.0),
            GeomVal::line(1.0, -1.0, 0.0)
        );
        // Vertical and horizontal degenerate slopes.
        assert_eq!(
            GeomVal::line_construct(2.0, 0.0, 2.0, 5.0),
            GeomVal::line(-1.0, 0.0, 2.0)
        );
        assert_eq!(
            GeomVal::line_construct(0.0, 3.0, 5.0, 3.0),
            GeomVal::line(0.0, -1.0, 3.0)
        );
    }

    #[test]
    fn line_format_canonical() {
        assert_eq!(format(&GeomVal::line(1.0, -1.0, 0.0)), "{1,-1,0}");
        // The verbatim (non-normalized) coefficients render as-is.
        assert_eq!(format(&GeomVal::line(2.0, -2.0, 0.0)), "{2,-2,0}");
        assert_eq!(format(&GeomVal::line(-0.0, -1.0, 3.5)), "{0,-1,3.5}");
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact integer-valued and √-of-perfect-square results compare exactly"
    )]
    #[allow(
        clippy::approx_constant,
        clippy::unreadable_literal,
        reason = "the 1.414213562373095 literals are the line↔line closed-form gap — deliberately 1 ULP below SQRT_2, so the constant cannot be substituted"
    )]
    fn line_ops() {
        // line <-> point: the foot-of-perpendicular path reaches √2 exactly (`SQRT_2`), matching the
        // reference engine bit-for-bit — unlike the algebraic closed form, which is 1 ULP below here.
        assert_eq!(
            line_distance_point(1.0, -1.0, 0.0, 0.0, 2.0),
            std::f64::consts::SQRT_2
        );
        assert_eq!(line_distance_point(1.0, -1.0, 0.0, 3.0, 3.0), 0.0);
        assert_eq!(line_distance_point(-1.0, 0.0, 2.0, 5.0, 7.0), 3.0);
        // line # line: the crossing point, or None when parallel.
        assert_eq!(
            line_intersection(1.0, -1.0, 0.0, 1.0, 1.0, 0.0),
            Some(GeomVal::point(0.0, 0.0))
        );
        assert_eq!(
            line_intersection(1.0, -1.0, 1.0, 1.0, 1.0, -7.0),
            Some(GeomVal::point(3.0, 4.0))
        );
        assert_eq!(line_intersection(1.0, -1.0, 0.0, 1.0, -1.0, 5.0), None);
        // line <-> line: 0 when crossing, else the constant (scale-invariant) parallel gap. This path
        // uses the algebraic closed form, 1 ULP below `SQRT_2` — matching the reference engine, whose
        // line↔line path also uses the closed form. The differently scaled {2,-2,4} reports the same
        // gap as {1,-1,2}.
        assert_eq!(line_distance_line(1.0, -1.0, 0.0, 1.0, 1.0, 0.0), 0.0);
        assert_eq!(
            line_distance_line(1.0, -1.0, 0.0, 1.0, -1.0, 2.0),
            1.414213562373095
        );
        assert_eq!(
            line_distance_line(1.0, -1.0, 0.0, 2.0, -2.0, 4.0),
            1.414213562373095
        );
    }

    #[test]
    fn line_rejects_degenerate() {
        // `{A,B,C}` with A and B both zero is not a line.
        assert_eq!(parse_line("{0,0,5}"), None);
        // A two-point form whose points coincide has no unique line.
        assert_eq!(parse_line("[(2,2),(2,2)]"), None);
        assert_eq!(parse_line("3,3,3,3"), None);
        // A valid coefficient line and a valid two-point line still parse.
        assert_eq!(parse_line("{2,-2,0}"), Some(GeomVal::line(2.0, -2.0, 0.0)));
        assert!(parse_line("[(0,0),(1,1)]").is_some());
    }
}
