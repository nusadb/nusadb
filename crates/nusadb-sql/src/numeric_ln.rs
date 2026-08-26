//! Exact `NUMERIC` natural log and base-`b` log.
//!
//! The double-precision (`FLOAT8`) forms of `ln`/`log`/`log10` go through `f64`, which loses the last
//! digit or two — e.g. `log(10, 1000)` comes out `2.9999999999999996`. The reference engine's
//! `NUMERIC` forms are exact to ~16 significant digits, so this module computes them in `i128`
//! fixed-point at a generous working scale and rounds to the reference engine's display scale.
//!
//! The value is `ln(x) = e·ln 10 + k·ln 2 + ln(m)` where `x = f · 10^e`, `f = m · 2^k` is reduced to
//! `[1, 2)`, and `ln(m)` is the fast-converging area-tangent series `2·(t + t³/3 + t⁵/5 + …)` with
//! `t = (m−1)/(m+1)`. A scale-36 product needs a 256-bit intermediate, so the multiplications go
//! through `mul_div`.

use crate::numeric::{Decimal, MAX_SCALE};

/// Working scale (fractional digits) of the internal fixed-point ln computation.
const S: u32 = 36;
/// The same working scale as a signed integer (for weight arithmetic).
const S_I32: i32 = 36;
/// `1.0` at the working scale.
const ONE: u128 = 10u128.pow(S);
/// `ln 2` at the working scale (from the reference engine, rounded to 36 places).
const LN2: i128 = 693_147_180_559_945_309_417_232_121_458_176_568;
/// `ln 10` at the working scale.
const LN10: i128 = 2_302_585_092_994_045_684_017_991_454_684_364_208;

/// `(hi, lo)` = `a * b` — the full 256-bit product of two `u128`s.
const fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const M: u128 = u64::MAX as u128;
    let (a0, a1) = (a & M, a >> 64);
    let (b0, b1) = (b & M, b >> 64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & M) + (p10 & M);
    let lo = (p00 & M) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// Divide the 256-bit value `(hi, lo)` by `d`, returning `(quotient, remainder)`. `None` when the
/// quotient would not fit a `u128` (`hi >= d`) or `d == 0`. Restoring binary long division.
const fn div_wide(hi: u128, lo: u128, d: u128) -> Option<(u128, u128)> {
    if d == 0 || hi >= d {
        return None;
    }
    let mut rem = hi;
    let mut quo: u128 = 0;
    let mut i = 128;
    while i > 0 {
        i -= 1;
        let bit = (lo >> i) & 1;
        let overflow = rem >> 127;
        rem = (rem << 1) | bit;
        // After the shift the candidate value is `overflow·2^128 + rem`; it is `>= d` (so subtract
        // once) exactly when the overflow bit is set or the wrapped remainder already reached `d`.
        if overflow == 1 || rem >= d {
            rem = rem.wrapping_sub(d);
            quo |= 1u128 << i;
        }
    }
    Some((quo, rem))
}

/// `round(a * b / c)` (half away from zero) at full precision. `None` on `c == 0` or `u128` overflow
/// of the quotient.
#[allow(
    clippy::many_single_char_names,
    reason = "a/b/c and the q/r quotient-remainder are the conventional mul-div names"
)]
fn mul_div(a: u128, b: u128, c: u128) -> Option<u128> {
    let (hi, lo) = mul_wide(a, b);
    let (q, r) = div_wide(hi, lo, c)?;
    // Round half up: `2r >= c`, written to avoid overflowing `2r`.
    Some(if r >= c - r { q + 1 } else { q })
}

/// The number of base-10 digits of `m` (`m >= 1`).
const fn decimal_digits(mut m: u128) -> u32 {
    let mut n = 1;
    while m >= 10 {
        m /= 10;
        n += 1;
    }
    n
}

/// `ln(m)` for a working-scale `m` in `[1, 2)`, as a working-scale non-negative value. Uses the
/// area-tangent series, which converges by `t² < 1/9` per term.
fn ln_reduced(m: u128) -> Option<u128> {
    let num = m - ONE;
    let den = m + ONE;
    let t = mul_div(num, ONE, den)?; // (m-1)/(m+1)
    let t2 = mul_div(t, t, ONE)?;
    let mut term = t;
    let mut sum = t;
    let mut k: u128 = 3;
    loop {
        term = mul_div(term, t2, ONE)?; // t^k, k = 3, 5, 7, …
        let add = term / k;
        if add == 0 {
            break;
        }
        sum += add;
        k += 2;
    }
    Some(sum * 2)
}

/// `ln(x)` as a signed working-scale (`S` fractional digits) value, or `None` when `x <= 0`.
#[allow(
    clippy::many_single_char_names,
    reason = "m/s/e/k are the mantissa, scale, weight and halving-count of the reduction"
)]
fn ln_fixed(x: &Decimal) -> Option<i128> {
    if x.mantissa <= 0 {
        return None;
    }
    let m = x.mantissa.unsigned_abs();
    let s = i32::from(x.scale);
    // x = f · 10^e with f in [1, 10): e is x's decimal weight. `shift = S - s - e = 37 - digits`,
    // which stays in [-2, 36] for any representable numeric, so the power of ten never overflows.
    let digits = i32::try_from(decimal_digits(m)).ok()?;
    let e = digits - 1 - s;
    let shift = S_I32 - s - e;
    let fm: u128 = if shift >= 0 {
        m.checked_mul(10u128.checked_pow(u32::try_from(shift).ok()?)?)?
    } else {
        let p = 10u128.checked_pow(u32::try_from(-shift).ok()?)?;
        (m + p / 2) / p
    };
    // Reduce f in [1, 10) to m2 in [1, 2) by halving (at most three times).
    let mut m2 = fm;
    let mut k: i128 = 0;
    while m2 >= 2 * ONE {
        m2 /= 2;
        k += 1;
    }
    let ln_m2 = i128::try_from(ln_reduced(m2)?).ok()?;
    i128::from(e)
        .checked_mul(LN10)?
        .checked_add(k.checked_mul(LN2)?)?
        .checked_add(ln_m2)
}

/// The decimal weight of a signed working-scale value `v` (i.e. `floor(log10(|value|))`), or `None`
/// for zero.
fn fixed_weight(v: i128) -> Option<i32> {
    if v == 0 {
        return None;
    }
    Some(i32::try_from(decimal_digits(v.unsigned_abs())).ok()? - 1 - S_I32)
}

/// Round a signed working-scale value to `target` fractional digits.
fn round_to(v: i128, target: u8) -> Option<Decimal> {
    Decimal {
        mantissa: v,
        scale: u8::try_from(S).ok()?,
    }
    .rescale(target)
}

/// The decimal weight of a `Decimal` (`floor(log10(|value|))`), or `None` for zero.
fn decimal_weight(v: &Decimal) -> Option<i32> {
    if v.mantissa == 0 {
        return None;
    }
    Some(i32::try_from(decimal_digits(v.mantissa.unsigned_abs())).ok()? - 1 - i32::from(v.scale))
}

/// The reference engine's display scale for `ln(x)`. Mirrors its result-weight estimate: for `x` in
/// `[0.9, 1.1]` (where `ln(x) ≈ x − 1`) the scale is `16 − weight(x − 1)`; otherwise it is
/// `16 − max(0, weight(ln x))`. Floored by the input scale and capped at [`MAX_SCALE`].
fn ln_scale(x: &Decimal, result: i128) -> u8 {
    let one = Decimal {
        mantissa: 1,
        scale: 0,
    };
    let near_one = x.compare(&Decimal::parse("0.9").unwrap_or(one)).is_ge()
        && x.compare(&one).is_le()
        || x.compare(&one).is_gt() && x.compare(&Decimal::parse("1.1").unwrap_or(one)).is_le();
    let base = if near_one {
        // `w <= -1` in this range; `x == 1` (no weight) yields 16.
        x.checked_sub(&one)
            .and_then(|diff| decimal_weight(&diff))
            .map_or(16, |w| 16 - w)
    } else {
        16 - fixed_weight(result).map_or(0, |w| w.max(0))
    };
    let s = base.max(i32::from(x.scale)).clamp(0, i32::from(MAX_SCALE));
    u8::try_from(s).unwrap_or(MAX_SCALE)
}

/// `ln(x)` as an exact `NUMERIC`, or `None` when `x <= 0` (the caller raises the domain error).
#[must_use]
pub fn ln(x: &Decimal) -> Option<Decimal> {
    let lm = ln_fixed(x)?;
    round_to(lm, ln_scale(x, lm))
}

/// Display scale for `log_b(x)`: `16 - max(0, weight(ln x)) + max(0, weight(ln b))`, floored by both
/// inputs' scales.
fn log_scale(ln_x: i128, ln_b: i128, base_scale: u8, x_scale: u8) -> u8 {
    let dwx = fixed_weight(ln_x).map_or(0, |w| w.max(0));
    let dwb = fixed_weight(ln_b).map_or(0, |w| w.max(0));
    let s = (16 - dwx + dwb)
        .max(i32::from(base_scale))
        .max(i32::from(x_scale))
        .clamp(0, i32::from(MAX_SCALE));
    u8::try_from(s).unwrap_or(MAX_SCALE)
}

/// `ln_x / ln_b` (both signed working-scale) rounded to `target` fractional digits.
fn divide(ln_x: i128, ln_b: i128, target: u8) -> Option<Decimal> {
    let sign = ln_x.signum() * ln_b.signum();
    let num = ln_x.unsigned_abs();
    let den = ln_b.unsigned_abs();
    let scaled = mul_div(num, 10u128.checked_pow(u32::from(target))?, den)?;
    Some(Decimal {
        mantissa: sign * i128::try_from(scaled).ok()?,
        scale: target,
    })
}

/// `log_base(x)` (base-`b` logarithm) as an exact `NUMERIC`. `None` when `x <= 0`, `base <= 0`, or
/// `base == 1` (the caller raises the domain error).
#[must_use]
pub fn log_base(base: &Decimal, x: &Decimal) -> Option<Decimal> {
    let ln_b = ln_fixed(base)?;
    if ln_b == 0 {
        return None; // ln(1) = 0: base 1 has no logarithm
    }
    let ln_x = ln_fixed(x)?;
    divide(ln_x, ln_b, log_scale(ln_x, ln_b, base.scale, x.scale))
}

/// `log10(x)` as an exact `NUMERIC` (base 10, treated as an exact integer base). `None` when `x <= 0`.
#[must_use]
pub fn log10(x: &Decimal) -> Option<Decimal> {
    let ln_x = ln_fixed(x)?;
    // ln(10) is the LN10 constant; its weight is 0, so the base term of the scale contributes 0.
    divide(ln_x, LN10, log_scale(ln_x, LN10, 0, x.scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Decimal {
        Decimal::parse(s).unwrap()
    }

    #[test]
    fn ln_matches_reference_engine() {
        // Every expected string is the reference engine's `ln(x::numeric)::text`.
        assert_eq!(ln(&d("2")).unwrap().format(), "0.6931471805599453");
        assert_eq!(ln(&d("10")).unwrap().format(), "2.3025850929940457");
        assert_eq!(ln(&d("100")).unwrap().format(), "4.6051701859880914");
        assert_eq!(ln(&d("123456")).unwrap().format(), "11.723640096265401");
        assert_eq!(ln(&d("0.5")).unwrap().format(), "-0.6931471805599453");
        assert_eq!(ln(&d("0.001")).unwrap().format(), "-6.9077552789821371");
        assert_eq!(
            ln(&d("100000000000000000000")).unwrap().format(),
            "46.051701859880914"
        );
        assert_eq!(ln(&d("1")).unwrap().format(), "0.0000000000000000");
        assert_eq!(
            ln(&d("1.0000001")).unwrap().format(),
            "0.00000009999999500000033"
        );
        assert_eq!(
            ln(&d("22026.4657948")).unwrap().format(),
            "9.9999999999996951"
        );
        // Non-positive is a domain error (None).
        assert!(ln(&d("0")).is_none());
        assert!(ln(&d("-1")).is_none());
    }

    #[test]
    fn log10_and_log_base_match_reference_engine() {
        assert_eq!(log10(&d("1000")).unwrap().format(), "3.0000000000000000");
        assert_eq!(log10(&d("2")).unwrap().format(), "0.3010299956639812");
        assert_eq!(
            log10(&d("100000000000000000000")).unwrap().format(),
            "20.000000000000000"
        );
        assert_eq!(log10(&d("0.5")).unwrap().format(), "-0.3010299956639812");

        assert_eq!(
            log_base(&d("2"), &d("8")).unwrap().format(),
            "3.0000000000000000"
        );
        assert_eq!(
            log_base(&d("10"), &d("1000")).unwrap().format(),
            "3.0000000000000000"
        );
        assert_eq!(
            log_base(&d("2"), &d("10")).unwrap().format(),
            "3.3219280948873623"
        );
        assert_eq!(
            log_base(&d("2"), &d("1024")).unwrap().format(),
            "10.0000000000000000"
        );
        assert_eq!(
            log_base(&d("2"), &d("1.5")).unwrap().format(),
            "0.5849625007211562"
        );
        assert_eq!(
            log_base(&d("2"), &d("1267650600228229401496703205376"))
                .unwrap()
                .format(),
            "100.000000000000000"
        );
        // base 1 / non-positive are domain errors.
        assert!(log_base(&d("1"), &d("8")).is_none());
        assert!(log_base(&d("2"), &d("0")).is_none());
    }

    #[test]
    fn ln_log_stress_matches_reference_engine() {
        // A broad spread of magnitudes; every string is the reference engine's exact output.
        for (x, want) in [
            ("3", "1.0986122886681097"),
            ("7", "1.9459101490553133"),
            ("1.5", "0.4054651081081644"),
            ("0.1", "-2.3025850929940457"),
            ("0.01", "-4.6051701859880914"),
            ("99", "4.5951198501345899"),
            ("1000000", "13.815510557964274"),
            ("2.718281828", "0.9999999998311267"),
            ("50", "3.9120230054281461"),
            ("0.3", "-1.2039728043259360"),
            ("12345.678", "9.4210613212918320"),
            ("1.1", "0.09531017980432486"),
            ("999999999", "20.723265835946411"),
            ("0.999", "-0.0010005003335835335"),
        ] {
            assert_eq!(ln(&d(x)).unwrap().format(), want, "ln({x})");
        }
        for (b, x, want) in [
            ("3", "9", "2.0000000000000000"),
            ("3", "27", "3.0000000000000000"),
            ("5", "625", "4.0000000000000000"),
            ("10", "7", "0.8450980400142568"),
            ("2", "3", "1.5849625007211562"),
            ("7", "7", "1.0000000000000000"),
            ("10", "0.001", "-3.0000000000000000"),
            ("2", "0.25", "-2.0000000000000000"),
            ("4", "2", "0.5000000000000000"),
            ("100", "10", "0.5000000000000000"),
        ] {
            assert_eq!(
                log_base(&d(b), &d(x)).unwrap().format(),
                want,
                "log({b},{x})"
            );
        }
        for (x, want) in [
            ("5", "0.6989700043360188"),
            ("50", "1.6989700043360188"),
            ("0.2", "-0.6989700043360188"),
            ("7", "0.8450980400142568"),
            ("123", "2.0899051114393979"),
            ("1", "0.0000000000000000"),
        ] {
            assert_eq!(log10(&d(x)).unwrap().format(), want, "log10({x})");
        }
    }
}
