//! Fixed-dimension floating-point vectors (`VECTOR(n)`) and their distance metrics.
//!
//! A vector is a 1-D array of `f32` whose length (the *dimension*) is fixed by the column's declared
//! `VECTOR(n)`. The text form is a bracketed, comma-separated list — `[1,2,3]` — so
//! `'[1,2,3]'::VECTOR(3)` round-trips and matches the de-facto bracketed convention used across the
//! vector-search ecosystem.
//!
//! This module is dependency-free and panic-free: parsing rejects malformed input with `None`, and
//! the distance metrics return `None` on a dimension mismatch (the caller raises a typed SQL error).
//! The metrics run a blocked, `f64`-accumulating kernel that the compiler vectorizes to AVX2+FMA on
//! a capable x86-64 host (detected at runtime) and otherwise runs as a scalar loop — the two builds
//! produce bit-identical results, so a distance is independent of the host CPU (see the kernel note
//! below).
//!
//! Distance metrics (all return *distances* — smaller means more similar):
//! - [`l2_distance`] — Euclidean distance `‖a − b‖₂`.
//! - [`cosine_distance`] — `1 − cosθ`; the metric bound to the `<=>` operator.
//! - [`neg_inner_product`] — the *negative* dot product `−(a · b)`, so that, like the others, a
//!   smaller value is a closer match (the `<#>` distance form). [`dot`] exposes the raw dot product,
//!   which is what the `inner_product()` function returns.

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

// === Distance kernels =====================================================
//
// Each metric reduces two equal-length `f32` slices to an `f64`. The reduction is *blocked* into
// [`LANES`] independent accumulators combined with fused multiply-add, so on x86-64 with AVX2+FMA
// (detected at runtime) the compiler vectorizes each kernel to one 256-bit FMA per step, while every
// other target — and any host where `NUSADB_DISABLE_SIMD` is set — runs the identical source as a
// scalar loop. Because the accumulator layout and the order the lanes are combined are fixed in the
// source (LLVM does not reassociate floating-point without fast-math), the vectorized and scalar
// builds produce bit-identical results: a distance never depends on the host's CPU features. The
// scalar core is the correctness oracle the AVX2 build is tested against. Accumulation stays in
// `f64` (the inputs are `f32`), so precision matches the previous scalar implementation to within
// the last ULP of reassociation — no exact-value regression versus a per-element `f64` sum.

/// Independent `f64` accumulators each kernel carries — four fills one AVX2 `__m256d` register.
const LANES: usize = 4;

/// Whether the vectorized kernels may run, or the scalar path is forced everywhere. `false` when
/// `NUSADB_DISABLE_SIMD` is set (to any value), latched on first use — an emergency switch, and a
/// way for CI to exercise the scalar path on an AVX2 host. Behaviour is identical either way.
#[cfg(target_arch = "x86_64")]
fn simd_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NUSADB_DISABLE_SIMD").is_none())
}

/// Combine the four blocked accumulators in a fixed order (independent of build), then return.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "must inline into the #[target_feature] AVX2 wrapper so the reduction recompiles with \
              avx2+fma; the bench confirms it then vectorizes"
)]
fn horizontal(acc: [f64; LANES]) -> f64 {
    let [c0, c1, c2, c3] = acc;
    (c0 + c1) + (c2 + c3)
}

/// `Σ (aᵢ − bᵢ)²` in `f64` — the blocked, FMA-based core (see the module kernel note). `a` and `b`
/// must be the same length; the callers guarantee it.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "must inline into the #[target_feature] AVX2 wrapper so the body recompiles with avx2+fma"
)]
fn l2_sq_core(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let (mut a_blocks, mut b_blocks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    for (ca, cb) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
        for ((slot, &x), &y) in acc.iter_mut().zip(ca).zip(cb) {
            let d = f64::from(x) - f64::from(y);
            *slot = d.mul_add(d, *slot);
        }
    }
    let mut sum = horizontal(acc);
    for (&x, &y) in a_blocks.remainder().iter().zip(b_blocks.remainder()) {
        let d = f64::from(x) - f64::from(y);
        sum = d.mul_add(d, sum);
    }
    sum
}

/// `Σ aᵢ·bᵢ` in `f64` — the blocked, FMA-based core.
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "must inline into the #[target_feature] AVX2 wrapper so the body recompiles with avx2+fma"
)]
fn dot_core(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let (mut a_blocks, mut b_blocks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    for (ca, cb) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
        for ((slot, &x), &y) in acc.iter_mut().zip(ca).zip(cb) {
            *slot = f64::from(x).mul_add(f64::from(y), *slot);
        }
    }
    let mut sum = horizontal(acc);
    for (&x, &y) in a_blocks.remainder().iter().zip(b_blocks.remainder()) {
        sum = f64::from(x).mul_add(f64::from(y), sum);
    }
    sum
}

/// `Σ |aᵢ − bᵢ|` in `f64` — blocked core (a plain add; there is no product to fuse).
#[inline(always)]
#[allow(
    clippy::inline_always,
    reason = "must inline into the #[target_feature] AVX2 wrapper so the body recompiles with avx2+fma"
)]
fn l1_core(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; LANES];
    let (mut a_blocks, mut b_blocks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    for (ca, cb) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
        for ((slot, &x), &y) in acc.iter_mut().zip(ca).zip(cb) {
            *slot += (f64::from(x) - f64::from(y)).abs();
        }
    }
    let mut sum = horizontal(acc);
    for (&x, &y) in a_blocks.remainder().iter().zip(b_blocks.remainder()) {
        sum += (f64::from(x) - f64::from(y)).abs();
    }
    sum
}

/// Define the runtime dispatch (`$dispatch`) plus the AVX2+FMA build (`$avx2`) for a two-slice
/// `f64` reduction whose scalar core is `$core`. Both builds run the same source, so they agree
/// bit-for-bit; the AVX2 build just lets the compiler vectorize it.
macro_rules! simd_dispatch {
    ($dispatch:ident, $avx2:ident, $core:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx2(a: &[f32], b: &[f32]) -> f64 {
            $core(a, b)
        }

        /// Reduce equal-length `a` and `b` via the AVX2+FMA build when the host has those features
        /// (and SIMD is not disabled), else the identical scalar core.
        #[inline]
        fn $dispatch(a: &[f32], b: &[f32]) -> f64 {
            #[cfg(target_arch = "x86_64")]
            {
                if simd_enabled()
                    && std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                {
                    // SAFETY: dispatched only after confirming avx2 + fma at runtime, which is
                    // exactly what `#[target_feature(enable = "avx2,fma")]` on `$avx2` requires.
                    return unsafe { $avx2(a, b) };
                }
            }
            $core(a, b)
        }
    };
}

simd_dispatch!(l2_sq, l2_sq_avx2, l2_sq_core);
simd_dispatch!(dot_sum, dot_sum_avx2, dot_core);
simd_dispatch!(l1_sum, l1_sum_avx2, l1_core);

/// The raw dot product `a · b`, computed in `f64`, or `None` if the dimensions differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(dot_sum(a, b))
}

/// Manhattan (taxicab) distance `Σ |aᵢ − bᵢ|`, or `None` if the dimensions differ.
#[must_use]
pub fn l1_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(l1_sum(a, b))
}

/// The Euclidean norm (magnitude) `‖a‖₂ = √(Σ aᵢ²)`.
#[must_use]
pub fn norm(a: &[f32]) -> f64 {
    dot_sum(a, a).sqrt()
}

/// Euclidean distance `‖a − b‖₂`, or `None` if the dimensions differ.
#[must_use]
pub fn l2_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(l2_sq(a, b).sqrt())
}

/// Cosine distance `1 − (a · b) / (‖a‖ ‖b‖)`, or `None` if the dimensions differ.
///
/// A zero-magnitude operand has no defined angle; we return `1.0` (maximally distant) rather than a
/// `NaN`, a conventional treatment of the zero vector.
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let norm_a = dot_sum(a, a).sqrt();
    let norm_b = dot_sum(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Some(1.0);
    }
    Some(1.0 - dot_sum(a, b) / (norm_a * norm_b))
}

/// The negative dot product `−(a · b)`, or `None` if the dimensions differ (the `<#>` operator).
///
/// Negated so that — as with [`l2_distance`] and [`cosine_distance`] — a *smaller* value means a
/// closer match, which is what a top-K nearest-neighbour `ORDER BY ... LIMIT k` wants. The
/// `inner_product(a, b)` **function** returns the raw *positive* dot product ([`dot`]); only this
/// operator form negates.
#[must_use]
pub fn neg_inner_product(a: &[f32], b: &[f32]) -> Option<f64> {
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
        // Dot is 0 → neg_inner_product is 0. The positive dot product is `dot`.
        assert!(neg_inner_product(&a, &b).unwrap().abs() < 1e-9);
        assert!(dot(&a, &b).unwrap().abs() < 1e-9);
        // Identical unit vectors → cosine distance 0, L2 0.
        assert!(cosine_distance(&a, &a).unwrap().abs() < 1e-9);
        assert!(l2_distance(&a, &a).unwrap().abs() < 1e-9);
        // dot of [1,2,3]·[4,5,6] = 32 (the `inner_product()` function); the operator negates to −32.
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() - 32.0).abs() < 1e-9);
        assert!(
            (neg_inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() + 32.0).abs() < 1e-9
        );
    }

    #[test]
    fn dimension_mismatch_is_none() {
        assert_eq!(l2_distance(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine_distance(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(neg_inner_product(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(dot(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn zero_vector_cosine_is_max_distance() {
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 1.0]), Some(1.0));
    }

    #[test]
    fn l1_distance_and_norm_match_hand_computed() {
        // L1 of [1,2,3] and [4,6,8] = 3 + 4 + 5 = 12.
        assert!((l1_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0]).unwrap() - 12.0).abs() < 1e-9);
        assert_eq!(l1_distance(&[1.0], &[1.0, 2.0]), None); // dimension mismatch
        // Norm of [3,4] = 5; the empty vector has norm 0.
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-9);
        assert!(norm(&[]).abs() < 1e-9);
    }

    /// A deterministic, dependency-free pseudo-random `f32` in roughly `[-1, 1)`. A splitmix64-style
    /// hash of `(i, salt)` so the two operands differ and the sequence is reproducible per run.
    #[allow(
        clippy::cast_precision_loss,
        reason = "generating pseudo-random f32 test data; the integer-to-float rounding is intended"
    )]
    fn pseudo(i: usize, salt: u64) -> f32 {
        let mut z = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt.wrapping_mul(0xD1B5_4A32_D192_ED03));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map the top 24 bits to [0, 1), then to [-1, 1).
        ((z >> 40) as f32 / (1u32 << 24) as f32).mul_add(2.0, -1.0)
    }

    /// The AVX2+FMA build of every kernel must match its scalar oracle **bit-for-bit**, so a distance
    /// never depends on the host's CPU features. Checked across dims that exercise the four-wide tail
    /// and realistic ≥128 embedding widths. Skipped (not failed) on a host without AVX2+FMA, where
    /// only the scalar path exists.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_kernels_match_scalar_oracle_bit_for_bit() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        for dim in [
            0usize, 1, 3, 4, 7, 8, 15, 16, 31, 128, 129, 130, 255, 256, 384, 1536,
        ] {
            let a: Vec<f32> = (0..dim).map(|i| pseudo(i, 1)).collect();
            let b: Vec<f32> = (0..dim).map(|i| pseudo(i, 2)).collect();
            // SAFETY: the avx2 + fma runtime check above gates this whole block.
            unsafe {
                assert_eq!(
                    l2_sq_avx2(&a, &b).to_bits(),
                    l2_sq_core(&a, &b).to_bits(),
                    "l2 kernel diverged at dim {dim}"
                );
                assert_eq!(
                    dot_sum_avx2(&a, &b).to_bits(),
                    dot_core(&a, &b).to_bits(),
                    "dot kernel diverged at dim {dim}"
                );
                assert_eq!(
                    l1_sum_avx2(&a, &b).to_bits(),
                    l1_core(&a, &b).to_bits(),
                    "l1 kernel diverged at dim {dim}"
                );
            }
        }
    }

    /// Microbenchmark: the SIMD distance kernel vs the previous single-accumulator scalar `f64` sum,
    /// at realistic embedding widths and a row count comparable to scanning hundreds of thousands of
    /// rows. The perf gate for the vectorized kernel — run with
    /// `cargo test -p nusadb-sql bench_distance_kernel -- --ignored --nocapture`. Ignored so it never
    /// runs in the normal suite; skipped on a host without AVX2+FMA.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::items_after_statements,
        reason = "microbenchmark: integer-to-float timing arithmetic, and a local oracle fn kept next to its use"
    )]
    fn bench_distance_kernel_simd_vs_scalar() {
        use std::hint::black_box;
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            eprintln!("no AVX2+FMA on this host — skipping kernel benchmark");
            return;
        }
        // The previous implementation: promote each element to f64 and sum sequentially.
        fn scalar_seq_l2(a: &[f32], b: &[f32]) -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| {
                    let d = f64::from(x) - f64::from(y);
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        }
        let median3 =
            |f: &dyn Fn(&[f32], &[f32]) -> f64, a: &[f32], b: &[f32], iters: u64| -> u128 {
                let mut runs = [0u128; 3];
                for run in &mut runs {
                    let t = std::time::Instant::now();
                    let mut acc = 0.0f64;
                    for _ in 0..iters {
                        acc += f(black_box(a), black_box(b));
                    }
                    black_box(acc);
                    *run = t.elapsed().as_nanos();
                }
                runs.sort_unstable();
                runs[1]
            };
        for dim in [128usize, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| pseudo(i, 1)).collect();
            let b: Vec<f32> = (0..dim).map(|i| pseudo(i, 2)).collect();
            let iters = 1_000_000u64;
            let old = median3(&scalar_seq_l2, &a, &b, iters);
            let new = median3(&|a, b| l2_distance(a, b).unwrap(), &a, &b, iters);
            let speedup = old as f64 / new as f64;
            eprintln!(
                "l2 dim={dim} iters={iters}: scalar-seq {:.1} ns/op, simd {:.1} ns/op, speedup {speedup:.2}x",
                old as f64 / iters as f64,
                new as f64 / iters as f64,
            );
        }
    }

    /// Dump realistic 128-dim vectors and this kernel's four distances (full `f64` precision) so an
    /// external harness can differential them against pgvector. A no-op unless `VEC_DIFF_OUT` names an
    /// output file; ignored so it never runs in the normal suite. The printed vectors use the exact
    /// round-trippable `f32` text, so pgvector parses the identical values.
    #[test]
    #[ignore = "differential dump for the pgvector harness; set VEC_DIFF_OUT and run --ignored"]
    fn differential_dump() {
        use std::fmt::Write as _;
        let Ok(path) = std::env::var("VEC_DIFF_OUT") else {
            return;
        };
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| pseudo(i, 99)).collect();
        let mut out = String::new();
        writeln!(out, "QUERY|{}", format(&query)).unwrap();
        for id in 0..30u64 {
            let v: Vec<f32> = (0..dim).map(|i| pseudo(i, id + 1)).collect();
            writeln!(
                out,
                "{id}|{}|{:.17}|{:.17}|{:.17}|{:.17}",
                format(&v),
                l2_distance(&v, &query).unwrap(),
                neg_inner_product(&v, &query).unwrap(),
                cosine_distance(&v, &query).unwrap(),
                l1_distance(&v, &query).unwrap(),
            )
            .unwrap();
        }
        std::fs::write(&path, out).unwrap();
    }

    /// The blocked kernels must equal a straightforward per-element `f64` reference to within a tiny
    /// tolerance (they only reassociate the sum), over a realistic-width vector.
    #[test]
    fn kernels_match_naive_f64_reference() {
        let dim = 384;
        let a: Vec<f32> = (0..dim).map(|i| pseudo(i, 3)).collect();
        let b: Vec<f32> = (0..dim).map(|i| pseudo(i, 4)).collect();
        let ref_dot: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let ref_l2: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
            .sqrt();
        let ref_l1: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .sum();
        assert!((dot(&a, &b).unwrap() - ref_dot).abs() < 1e-9, "dot");
        assert!((l2_distance(&a, &b).unwrap() - ref_l2).abs() < 1e-9, "l2");
        assert!((l1_distance(&a, &b).unwrap() - ref_l1).abs() < 1e-9, "l1");
    }
}
