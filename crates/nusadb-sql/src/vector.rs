//! Fixed-dimension floating-point vectors (`VECTOR(n)`) and their distance metrics.
//!
//! A vector is a 1-D array of `f32` whose length (the *dimension*) is fixed by the column's declared
//! `VECTOR(n)`. The text form is a bracketed, comma-separated list — `[1,2,3]` — so
//! `'[1,2,3]'::VECTOR(3)` round-trips and matches the de-facto bracketed convention used across the
//! vector-search ecosystem.
//!
//! This module is dependency-free and panic-free: parsing rejects malformed input with `None`, and
//! the distance metrics return `None` on a dimension mismatch (the caller raises a typed SQL error).
//!
//! Distance metrics (all return *distances* — smaller means more similar):
//! - [`l2_distance`] — Euclidean distance `‖a − b‖₂`.
//! - [`cosine_distance`] — `1 − cosθ`; the metric bound to the `<=>` operator.
//! - [`inner_product`] — the *negative* dot product `−(a · b)`, so that, like the others, a smaller
//!   value is a closer match (bound to the `<#>` operator). [`dot`] exposes the raw dot product.

/// Parse a bracketed text literal `[x, y, z]` into its component `f32`s.
///
/// Whitespace around the brackets and between components is ignored. `[]` parses to an empty vector.
/// Returns `None` for anything that is not a bracketed, comma-separated list of finite-or-not floats
/// (each element is parsed with the standard `f32` grammar, which also accepts `inf`/`nan`).
#[must_use]
pub fn parse(s: &str) -> Option<Vec<f32>> {
    let inner = s.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| tok.trim().parse::<f32>().ok())
        .collect()
}

/// Render a vector as its canonical bracketed text form `[x,y,z]` (no interior spaces).
#[must_use]
pub fn format(v: &[f32]) -> String {
    let mut out = String::with_capacity(2 + v.len() * 4);
    out.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // `{x}` on an `f32` yields the shortest round-trippable decimal (e.g. `1` not `1.0`).
        out.push_str(&x.to_string());
    }
    out.push(']');
    out
}

/// The raw dot product `a · b`, computed in `f64`, or `None` if the dimensions differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum(),
    )
}

/// Euclidean distance `‖a − b‖₂`, or `None` if the dimensions differ.
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let sum_sq: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    Some(sum_sq.sqrt())
}

/// Cosine distance `1 − (a · b) / (‖a‖ ‖b‖)`, or `None` if the dimensions differ.
///
/// A zero-magnitude operand has no defined angle; we return `1.0` (maximally distant) rather than a
/// `NaN`, a conventional treatment of the zero vector.
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    let dot_ab = dot(a, b)?;
    let norm_a: f64 = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|&y| f64::from(y) * f64::from(y))
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Some(1.0);
    }
    Some(1.0 - dot_ab / (norm_a * norm_b))
}

/// The negative dot product `−(a · b)`, or `None` if the dimensions differ (the `<#>` operator).
///
/// Negated so that — as with [`l2_distance`] and [`cosine_distance`] — a *smaller* value means a
/// closer match, which is what a top-K nearest-neighbour `ORDER BY ... LIMIT k` wants.
#[must_use]
pub fn inner_product(a: &[f32], b: &[f32]) -> Option<f64> {
    dot(a, b).map(|d| -d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format_round_trip() {
        assert_eq!(parse("[1,2,3]"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(parse("  [ 1.5 , -2 , 0 ] "), Some(vec![1.5, -2.0, 0.0]));
        assert_eq!(parse("[]"), Some(Vec::new()));
        assert_eq!(format(&[1.0, 2.0, 3.0]), "[1,2,3]");
        assert_eq!(format(&[]), "[]");
    }

    #[test]
    fn parse_rejects_malformed() {
        assert_eq!(parse("1,2,3"), None); // no brackets
        assert_eq!(parse("[1,,3]"), None); // empty component
        assert_eq!(parse("[a,b]"), None); // non-numeric
        assert_eq!(parse("[1,2"), None); // unterminated
    }

    #[test]
    fn distances_match_hand_computed() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        // L2 of orthogonal unit vectors = sqrt(2).
        assert!((l2_distance(&a, &b).unwrap() - std::f64::consts::SQRT_2).abs() < 1e-9);
        // Orthogonal → cosine similarity 0 → cosine distance 1.
        assert!((cosine_distance(&a, &b).unwrap() - 1.0).abs() < 1e-9);
        // Dot is 0 → inner_product (negated) is 0.
        assert!(inner_product(&a, &b).unwrap().abs() < 1e-9);
        // Identical unit vectors → cosine distance 0, L2 0.
        assert!(cosine_distance(&a, &a).unwrap().abs() < 1e-9);
        assert!(l2_distance(&a, &a).unwrap().abs() < 1e-9);
        // inner_product of [1,2,3]·[4,5,6] = 32 → negated −32.
        assert!((inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() + 32.0).abs() < 1e-9);
    }

    #[test]
    fn dimension_mismatch_is_none() {
        assert_eq!(l2_distance(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine_distance(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(inner_product(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(dot(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn zero_vector_cosine_is_max_distance() {
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 1.0]), Some(1.0));
    }
}
