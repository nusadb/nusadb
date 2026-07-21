//! SIMD predicate kernels: compare an `i64` column against a scalar a vector lane at a time,
//! producing a per-row selection mask (`true` = keep). On x86-64 with AVX2 (detected at runtime) the
//! kernel processes four `i64` per 256-bit register; everywhere else it falls back to the scalar
//! loop. The scalar path is the correctness oracle — the AVX2 path must match it bit-for-bit, which
//! the tests assert over random data plus the `i64::MIN`/`MAX` and non-multiple-of-four edge cases.
//!
//! Setting the `NUSADB_DISABLE_SIMD` environment variable (to any value) forces the scalar path
//! everywhere, even on an AVX2 host. This lets CI prove the whole suite is green on the
//! scalar path without a non-AVX2 machine, and gives operators an emergency switch on hardware where
//! the SIMD path is suspect. The decision is read once and cached, so it costs nothing per call.

use crate::ast;

/// Whether the AVX2 kernels may run, or the scalar oracle is forced everywhere.
///
/// `false` when `NUSADB_DISABLE_SIMD` is set in the environment (any value), latched on first use.
/// Behaviour is unchanged either way — the scalar path is the AVX2 path's correctness oracle — so
/// forcing it only trades throughput for portability. Defined only for x86, where the AVX2
/// dispatch that consults it exists; on other targets (e.g. ARM) the scalar path is unconditional.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn simd_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NUSADB_DISABLE_SIMD").is_none())
}

/// A comparison a SIMD filter kernel can evaluate against a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// The kernel-supported comparison for a binary operator, or `None` for non-comparisons.
    pub(super) const fn from_binary_op(op: ast::BinaryOp) -> Option<Self> {
        Some(match op {
            ast::BinaryOp::Eq => Self::Eq,
            ast::BinaryOp::NotEq => Self::Ne,
            ast::BinaryOp::Lt => Self::Lt,
            ast::BinaryOp::LtEq => Self::Le,
            ast::BinaryOp::Gt => Self::Gt,
            ast::BinaryOp::GtEq => Self::Ge,
            _ => return None,
        })
    }

    /// The operator with operands swapped, so `scalar <op> column` becomes `column <swapped> scalar`
    /// (`<` ↔ `>`, `<=` ↔ `>=`; `=`/`<>` are symmetric).
    pub(super) const fn swapped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
        }
    }

    /// Evaluate `value <op> scalar`.
    const fn apply(self, value: i64, scalar: i64) -> bool {
        match self {
            Self::Eq => value == scalar,
            Self::Ne => value != scalar,
            Self::Lt => value < scalar,
            Self::Le => value <= scalar,
            Self::Gt => value > scalar,
            Self::Ge => value >= scalar,
        }
    }

    /// Evaluate `value <op> scalar` with IEEE-754 semantics (a `NaN` operand makes every comparison
    /// `false` except `<>`, which is `true`) — matching Rust's `f64` comparison operators.
    #[allow(
        clippy::float_cmp,
        reason = "SQL `=`/`<>` on floats is exact bit comparison by design; this is the oracle"
    )]
    fn apply_f64(self, value: f64, scalar: f64) -> bool {
        match self {
            Self::Eq => value == scalar,
            Self::Ne => value != scalar,
            Self::Lt => value < scalar,
            Self::Le => value <= scalar,
            Self::Gt => value > scalar,
            Self::Ge => value >= scalar,
        }
    }
}

/// Selection mask for `values[i] <op> scalar`. Uses AVX2 when available, else the scalar loop.
pub(super) fn filter_i64(values: &[i64], op: CmpOp, scalar: i64) -> Vec<bool> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd_enabled() && std::is_x86_feature_detected!("avx2") {
            // SAFETY: only called after the runtime AVX2 feature check above succeeds.
            return unsafe { filter_i64_avx2(values, op, scalar) };
        }
    }
    filter_i64_scalar(values, op, scalar)
}

/// The portable scalar kernel — also the oracle the SIMD path is tested against.
fn filter_i64_scalar(values: &[i64], op: CmpOp, scalar: i64) -> Vec<bool> {
    values.iter().map(|&v| op.apply(v, scalar)).collect()
}

/// AVX2 kernel: four `i64` per iteration. AVX2 only has signed `>` and `==` for 64-bit lanes, so the
/// other operators are derived by swapping operands (`<` is `scalar > value`) and/or inverting the
/// per-lane result bit (`<=` is `!(value > scalar)`, etc.).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "_mm256_loadu_si256 is an unaligned load; the source slice's alignment is irrelevant"
)]
unsafe fn filter_i64_avx2(values: &[i64], op: CmpOp, scalar: i64) -> Vec<bool> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_movemask_pd, _mm256_set1_epi64x,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_movemask_pd, _mm256_set1_epi64x,
    };

    // (base comparison, invert?) — derives every op from signed `cmpgt` / `cmpeq`.
    let (base_gt_lhs_is_value, use_eq, invert) = match op {
        CmpOp::Gt => (true, false, false),  // value > scalar
        CmpOp::Le => (true, false, true),   // !(value > scalar)
        CmpOp::Lt => (false, false, false), // scalar > value
        CmpOp::Ge => (false, false, true),  // !(scalar > value)
        CmpOp::Eq => (false, true, false),  // value == scalar
        CmpOp::Ne => (false, true, true),   // !(value == scalar)
    };

    let mut out = Vec::with_capacity(values.len());
    // SAFETY (whole block): every intrinsic is AVX2, enabled by `#[target_feature]`; the load reads
    // exactly 4 in-bounds `i64` (chunks_exact guarantees the length).
    unsafe {
        let s = _mm256_set1_epi64x(scalar);
        let mut chunks = values.chunks_exact(4);
        for c in &mut chunks {
            let v = _mm256_loadu_si256(c.as_ptr().cast::<__m256i>());
            let cmp = if use_eq {
                _mm256_cmpeq_epi64(v, s)
            } else if base_gt_lhs_is_value {
                _mm256_cmpgt_epi64(v, s)
            } else {
                _mm256_cmpgt_epi64(s, v)
            };
            // One sign bit per 64-bit lane -> a 4-bit mask, lane 0 in bit 0.
            let bits = _mm256_movemask_pd(_mm256_castsi256_pd(cmp));
            out.push(((bits & 0b0001) != 0) ^ invert);
            out.push(((bits & 0b0010) != 0) ^ invert);
            out.push(((bits & 0b0100) != 0) ^ invert);
            out.push(((bits & 0b1000) != 0) ^ invert);
        }
        for &v in chunks.remainder() {
            out.push(op.apply(v, scalar));
        }
    }
    out
}

/// Selection mask for `values[i] <op> scalar` over an `f64` column. AVX2 when available, else scalar.
pub(super) fn filter_f64(values: &[f64], op: CmpOp, scalar: f64) -> Vec<bool> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd_enabled() && std::is_x86_feature_detected!("avx2") {
            // SAFETY: only called after the runtime AVX2 feature check above succeeds.
            return unsafe { filter_f64_avx2(values, op, scalar) };
        }
    }
    filter_f64_scalar(values, op, scalar)
}

/// The portable scalar `f64` kernel — the oracle the SIMD path is tested against.
fn filter_f64_scalar(values: &[f64], op: CmpOp, scalar: f64) -> Vec<bool> {
    values.iter().map(|&v| op.apply_f64(v, scalar)).collect()
}

/// AVX2 `f64` kernel: four `f64` per iteration via `_mm256_cmp_pd`. The predicate immediates are
/// chosen so `NaN` semantics match Rust's operators — ordered-quiet (`_OQ`) for all but `<>`, which
/// uses unordered (`_UQ`) so a `NaN` operand yields `true` exactly as `f64::ne` does.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn filter_f64_avx2(values: &[f64], op: CmpOp, scalar: f64) -> Vec<bool> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ, _mm256_cmp_pd,
        _mm256_loadu_pd, _mm256_movemask_pd, _mm256_set1_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ, _mm256_cmp_pd,
        _mm256_loadu_pd, _mm256_movemask_pd, _mm256_set1_pd,
    };

    let mut out = Vec::with_capacity(values.len());
    // SAFETY (whole block): every intrinsic is AVX2, enabled by `#[target_feature]`; `_mm256_loadu_pd`
    // is an unaligned load of exactly 4 in-bounds `f64` (chunks_exact guarantees the length).
    unsafe {
        let s = _mm256_set1_pd(scalar);
        let mut chunks = values.chunks_exact(4);
        for c in &mut chunks {
            let v = _mm256_loadu_pd(c.as_ptr());
            let cmp = match op {
                CmpOp::Eq => _mm256_cmp_pd::<_CMP_EQ_OQ>(v, s),
                CmpOp::Ne => _mm256_cmp_pd::<_CMP_NEQ_UQ>(v, s),
                CmpOp::Lt => _mm256_cmp_pd::<_CMP_LT_OQ>(v, s),
                CmpOp::Le => _mm256_cmp_pd::<_CMP_LE_OQ>(v, s),
                CmpOp::Gt => _mm256_cmp_pd::<_CMP_GT_OQ>(v, s),
                CmpOp::Ge => _mm256_cmp_pd::<_CMP_GE_OQ>(v, s),
            };
            let bits = _mm256_movemask_pd(cmp);
            out.push((bits & 0b0001) != 0);
            out.push((bits & 0b0010) != 0);
            out.push((bits & 0b0100) != 0);
            out.push((bits & 0b1000) != 0);
        }
        for &v in chunks.remainder() {
            out.push(op.apply_f64(v, scalar));
        }
    }
    out
}

// --- Reduction kernels (A-PERF.AGG5a): MIN / MAX / SUM over a dense `i64` column ----------
//
// Integer MIN/MAX are order-independent and overflow-free, and integer SUM is associative with an
// exact `i128` result, so for all three the AVX2 and scalar paths agree bit-for-bit AND match the
// row path exactly. FLOAT SUM and FLOAT MIN/MAX deliberately stay on the row path: a SIMD `f64`
// SUM reorders the adds (FP is not associative) and `_mm256_min_pd`'s NaN handling differs from
// the row path's total-order compare, either of which would make a query's result depend on
// whether the CPU has AVX2 — and bit-exact batch=row results are a correctness / determinism
// invariant for the engine (the design, 2026-06-14).

/// Minimum / maximum of an `i64` column, or `None` for an empty column. Order-independent, so the
/// AVX2 and scalar paths agree exactly.
pub(super) fn min_i64(values: &[i64]) -> Option<i64> {
    reduce_i64(values, true)
}

/// See [`min_i64`].
pub(super) fn max_i64(values: &[i64]) -> Option<i64> {
    reduce_i64(values, false)
}

fn reduce_i64(values: &[i64], want_min: bool) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd_enabled() && std::is_x86_feature_detected!("avx2") {
            // SAFETY: only called after the runtime AVX2 feature check above succeeds.
            return Some(unsafe { reduce_i64_avx2(values, want_min) });
        }
    }
    Some(reduce_i64_scalar(values, want_min))
}

fn reduce_i64_scalar(values: &[i64], want_min: bool) -> i64 {
    values
        .iter()
        .copied()
        .reduce(|a, b| if (b < a) == want_min { b } else { a })
        .unwrap_or_default()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "_mm256_loadu_si256 is an unaligned load; the source slice's alignment is irrelevant"
)]
unsafe fn reduce_i64_avx2(values: &[i64], want_min: bool) -> i64 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };

    // AVX2 lacks `_mm256_min_epi64`; derive it from signed `cmpgt` + a byte blend that selects, per
    // 64-bit lane, the wanted operand. `chunks_exact(4)` may be empty (len < 4); then `seeded` stays
    // false and the running min/max comes entirely from the remainder, so a 1..=3-element column
    // still works.
    // SAFETY (whole block): AVX2 intrinsics enabled by `#[target_feature]`; every load reads exactly
    // 4 in-bounds `i64`.
    unsafe {
        let mut chunks = values.chunks_exact(4);
        let mut acc = _mm256_setzero_si256();
        let mut seeded = false;
        for c in &mut chunks {
            let v = _mm256_loadu_si256(c.as_ptr().cast::<__m256i>());
            if seeded {
                // `gt` lane = all-ones where acc > v.
                let gt = _mm256_cmpgt_epi64(acc, v);
                acc = if want_min {
                    _mm256_blendv_epi8(acc, v, gt) // pick v where acc > v
                } else {
                    _mm256_blendv_epi8(v, acc, gt) // pick acc where acc > v
                };
            } else {
                acc = v;
                seeded = true;
            }
        }
        // Reduce whatever lanes we accumulated, then fold the < 4-element remainder scalarly.
        let mut best: Option<i64> = if seeded {
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc);
            Some(reduce_i64_scalar(&lanes, want_min))
        } else {
            None
        };
        for &v in chunks.remainder() {
            best = Some(best.map_or(v, |b| if (v < b) == want_min { v } else { b }));
        }
        best.unwrap_or_default()
    }
}

/// Exact sum of an `i64` column, widened to `i128` (A-PERF.AGG5a / F2a). Integer addition is
/// associative, so the AVX2 block reduction returns the **same bits** as the sequential scalar
/// fold — this is what lets `SUM(INT)` take the SIMD path without violating the batch=row
/// determinism invariant, unlike float SUM. `0` for an empty column (the caller maps an empty
/// group to SQL NULL before the kernel is consulted). Never overflows: the result is exact in
/// `i128` for any in-memory column (`len · |i64| < 2^127`).
pub(super) fn sum_i128(values: &[i64]) -> i128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if simd_enabled() && std::is_x86_feature_detected!("avx2") {
            // SAFETY: only called after the runtime AVX2 feature check above succeeds.
            return unsafe { sum_i128_avx2(values) };
        }
    }
    sum_i128_scalar(values)
}

/// The portable scalar kernel — also the oracle the SIMD path is tested against, and the same
/// arithmetic as the row path's exact `i128` accumulator (`Acc::int_sum`).
fn sum_i128_scalar(values: &[i64]) -> i128 {
    values.iter().map(|&v| i128::from(v)).sum()
}

/// How many 4-lane chunks the AVX2 SUM accumulates in `i64` lanes before flushing to the `i128`
/// total. Each value contributes `< 2^32` to a low-half lane and `∈ [-2^31, 2^31)` to a high-half
/// lane, so after `2^20` chunks a lane holds at most `2^52` (low) / `2^51` (high) in magnitude —
/// comfortably inside `i64`, making the lane adds overflow-free by construction.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
const SUM_FLUSH_CHUNKS: usize = 1 << 20;

/// AVX2 kernel: four `i64` per iteration. AVX2 has no 64→128-bit widening add, so each value is
/// split exactly as `v = (v >> 32)·2^32 + (v & 0xFFFF_FFFF)` and the two halves are accumulated in
/// separate `i64` lane accumulators (both bounded — see [`SUM_FLUSH_CHUNKS`]), then recombined in
/// `i128` per block. Every step is exact integer arithmetic, so the total equals the scalar fold
/// bit-for-bit regardless of blocking.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "_mm256_loadu_si256 is an unaligned load; the source slice's alignment is irrelevant"
)]
unsafe fn sum_i128_avx2(values: &[i64]) -> i128 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_blend_epi32, _mm256_loadu_si256,
        _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_srli_epi64,
        _mm256_storeu_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_blend_epi32, _mm256_loadu_si256,
        _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_srli_epi64,
        _mm256_storeu_si256,
    };

    let mut total: i128 = 0;
    // SAFETY (whole block): every intrinsic is AVX2, enabled by `#[target_feature]`; each load reads
    // exactly 4 in-bounds `i64` (chunks_exact guarantees the length).
    unsafe {
        let low_mask = _mm256_set1_epi64x(0xFFFF_FFFF);
        for block in values.chunks(SUM_FLUSH_CHUNKS * 4) {
            let mut lo = _mm256_setzero_si256();
            let mut hi = _mm256_setzero_si256();
            let mut chunks = block.chunks_exact(4);
            for c in &mut chunks {
                let v = _mm256_loadu_si256(c.as_ptr().cast::<__m256i>());
                // Low half: `v & 0xFFFF_FFFF` — non-negative, `< 2^32`.
                lo = _mm256_add_epi64(lo, _mm256_and_si256(v, low_mask));
                // High half: arithmetic `v >> 32`. AVX2 lacks a 64-bit arithmetic shift; derive it
                // by blending the logical shift (even 32-bit words = the high words) with the
                // per-word sign fill (odd words = the 64-bit lane's sign extension).
                let shifted = _mm256_blend_epi32::<0b1010_1010>(
                    _mm256_srli_epi64::<32>(v),
                    _mm256_srai_epi32::<31>(v),
                );
                hi = _mm256_add_epi64(hi, shifted);
            }
            let mut lanes = [0i64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), lo);
            let lo_sum: i128 = lanes.iter().map(|&l| i128::from(l)).sum();
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), hi);
            let hi_sum: i128 = lanes.iter().map(|&l| i128::from(l)).sum();
            total += (hi_sum << 32) + lo_sum;
            // Only the final block can have a (< 4-element) remainder; fold it scalarly.
            for &v in chunks.remainder() {
                total += i128::from(v);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: [CmpOp; 6] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];

    // A deterministic spread including the signed-overflow-sensitive extremes and duplicates.
    fn sample() -> Vec<i64> {
        vec![
            0,
            1,
            -1,
            5,
            5,
            -5,
            42,
            -42,
            i64::MIN,
            i64::MAX,
            i64::MIN + 1,
            i64::MAX - 1,
            100,
            -100,
            7,
        ]
    }

    #[test]
    fn filter_dispatch_matches_scalar() {
        let values = sample();
        for op in OPS {
            for &scalar in &[0i64, 5, -5, 42, i64::MIN, i64::MAX] {
                assert_eq!(
                    filter_i64(&values, op, scalar),
                    filter_i64_scalar(&values, op, scalar),
                    "dispatch != scalar for {op:?} vs {scalar}"
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // no AVX2 on this host; the dispatch test already covers the scalar path
        }
        // Lengths spanning full registers + every remainder (0..=3), incl. the empty slice.
        for len in 0..=23 {
            let values: Vec<i64> = sample().into_iter().take(len).collect();
            for op in OPS {
                for &scalar in &[0i64, 5, -5, i64::MIN, i64::MAX, 7] {
                    // SAFETY: guarded by the AVX2 detection above.
                    let simd = unsafe { filter_i64_avx2(&values, op, scalar) };
                    assert_eq!(
                        simd,
                        filter_i64_scalar(&values, op, scalar),
                        "avx2 != scalar: len {len}, {op:?} vs {scalar}"
                    );
                }
            }
        }
    }

    // f64 spread including NaN, ±inf, and the duplicates the i64 sample has.
    fn sample_f64() -> Vec<f64> {
        vec![
            0.0,
            1.0,
            -1.0,
            5.0,
            5.0,
            -5.0,
            42.5,
            -42.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.5,
            -0.5,
            3.5,
            7.0,
        ]
    }

    #[test]
    fn filter_f64_dispatch_matches_scalar() {
        let values = sample_f64();
        for op in OPS {
            for &scalar in &[0.0f64, 5.0, -5.0, f64::NAN, f64::INFINITY] {
                assert_eq!(
                    filter_f64(&values, op, scalar),
                    filter_f64_scalar(&values, op, scalar),
                    "dispatch != scalar for {op:?} vs {scalar}"
                );
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn avx2_f64_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in 0..=23 {
            let values: Vec<f64> = sample_f64().into_iter().take(len).collect();
            for op in OPS {
                for &scalar in &[0.0f64, 5.0, -5.0, f64::NAN, f64::INFINITY, 7.0] {
                    // SAFETY: guarded by the AVX2 detection above.
                    let simd = unsafe { filter_f64_avx2(&values, op, scalar) };
                    assert_eq!(
                        simd,
                        filter_f64_scalar(&values, op, scalar),
                        "avx2 f64 != scalar: len {len}, {op:?} vs {scalar}"
                    );
                }
            }
        }
    }

    #[test]
    fn from_binary_op_maps_comparisons_only() {
        assert_eq!(CmpOp::from_binary_op(ast::BinaryOp::Gt), Some(CmpOp::Gt));
        assert_eq!(CmpOp::from_binary_op(ast::BinaryOp::LtEq), Some(CmpOp::Le));
        assert_eq!(CmpOp::from_binary_op(ast::BinaryOp::Plus), None);
    }

    // --- reduction kernels: i64 MIN/MAX only — SUM and float min/max stay on the row path ---

    #[test]
    fn min_max_i64_dispatch_matches_scalar() {
        for len in 0..=23 {
            let values: Vec<i64> = sample().into_iter().take(len).collect();
            assert_eq!(min_i64(&values), values.iter().copied().min());
            assert_eq!(max_i64(&values), values.iter().copied().max());
        }
        // An empty column reduces to `None` (the operator maps that to SQL NULL).
        assert_eq!(min_i64(&[]), None);
        assert_eq!(max_i64(&[]), None);
    }

    // --- SUM kernel (A-PERF.AGG5a / F2a): exact i128, bit-identical to the scalar fold ---

    #[test]
    fn sum_i128_dispatch_matches_scalar() {
        for len in 0..=23 {
            let values: Vec<i64> = sample().into_iter().take(len).collect();
            assert_eq!(
                sum_i128(&values),
                sum_i128_scalar(&values),
                "dispatch != scalar at len {len}"
            );
            assert_eq!(
                sum_i128_scalar(&values),
                values.iter().map(|&v| i128::from(v)).sum::<i128>(),
                "scalar oracle wrong at len {len}"
            );
        }
        // Past-i64 totals stay exact: nine i64::MAX sum to 9·(2^63−1), representable only in i128.
        let big = vec![i64::MAX; 9];
        assert_eq!(sum_i128(&big), 9 * i128::from(i64::MAX));
        let small = vec![i64::MIN; 9];
        assert_eq!(sum_i128(&small), 9 * i128::from(i64::MIN));
        assert_eq!(sum_i128(&[]), 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn avx2_sum_i128_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // Every remainder length (0..=3) over the extreme-heavy sample.
        for len in 0..=23 {
            let values: Vec<i64> = sample().into_iter().take(len).collect();
            // SAFETY: guarded by the AVX2 detection above.
            let simd = unsafe { sum_i128_avx2(&values) };
            assert_eq!(
                simd,
                sum_i128_scalar(&values),
                "avx2 != scalar at len {len}"
            );
        }
        // Deterministic pseudo-random spread (LCG — no dependency, reproducible), full i64 range.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let random: Vec<i64> = (0..1009)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                #[allow(
                    clippy::cast_possible_wrap,
                    reason = "reinterpreting LCG bits as i64 is the point — full-range test data"
                )]
                {
                    state as i64
                }
            })
            .collect();
        // SAFETY: guarded by the AVX2 detection above.
        let simd = unsafe { sum_i128_avx2(&random) };
        assert_eq!(
            simd,
            sum_i128_scalar(&random),
            "avx2 != scalar on random data"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn avx2_sum_i128_flush_boundary_is_exact() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // Cross the lane-accumulator flush with worst-case magnitudes: the per-lane bounds in
        // `SUM_FLUSH_CHUNKS`' contract are what keep these from overflowing i64 mid-block.
        let len = SUM_FLUSH_CHUNKS * 4 + 5;
        for fill in [i64::MAX, i64::MIN, -1] {
            let values = vec![fill; len];
            // SAFETY: guarded by the AVX2 detection above.
            let simd = unsafe { sum_i128_avx2(&values) };
            let expected = i128::from(fill) * i128::try_from(len).unwrap();
            assert_eq!(simd, expected, "flush-boundary sum wrong for fill {fill}");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn avx2_i64_minmax_matches_scalar_when_available() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // Lengths spanning full registers + every remainder (1..=3), incl. `i64::MIN`/`MAX`.
        for len in 1..=23 {
            let i: Vec<i64> = sample().into_iter().take(len).collect();
            // SAFETY: guarded by the AVX2 detection above.
            unsafe {
                assert_eq!(
                    reduce_i64_avx2(&i, true),
                    i.iter().copied().min().unwrap(),
                    "min len {len}"
                );
                assert_eq!(
                    reduce_i64_avx2(&i, false),
                    i.iter().copied().max().unwrap(),
                    "max len {len}"
                );
            }
        }
    }
}
