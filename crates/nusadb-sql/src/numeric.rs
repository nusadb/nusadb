//! Exact fixed-point decimal for the `NUMERIC` / `DECIMAL` types (phase 2).
//!
//! A [`Decimal`] is `mantissa * 10^(-scale)` with an `i128` mantissa, so values up to 38 significant
//! digits are represented and arithmetic is **exact** (addition/subtraction/multiplication never
//! lose precision; division rounds half-away-from-zero at a chosen result scale). This is the
//! in-memory form of [`ast::Value::Numeric`](crate::ast::Value::Numeric); the catalog column type
//! [`ColumnType::Numeric`](nusadb_core::ColumnType) carries the declared precision + scale.
//!
//! Parsing is exact from the decimal text (so `'19.99'` stores 1999 at scale 2, not an `f64`), and
//! formatting is canonical (the value's own scale), so a `text -> value -> text` round-trip is
//! stable.

use std::cmp::Ordering;

/// Extra fractional digits a division carries beyond its wider operand, so the quotient keeps at
/// least ~16 significant digits — rather than truncating to a handful and producing a
/// silently-wrong money result. Capped at [`MAX_SCALE`].
const DIV_GUARD: u8 = 16;
/// The largest scale (fractional digits) the codec / arithmetic will carry.
pub const MAX_SCALE: u8 = 38;

/// An exact base-10 fixed-point number: `mantissa * 10^(-scale)`.
#[derive(Debug, Clone, Copy)]
pub struct Decimal {
    /// Signed significand.
    pub mantissa: i128,
    /// Number of fractional digits (`value = mantissa / 10^scale`).
    pub scale: u8,
}

// Two decimals are equal when they denote the same number, regardless of trailing-zero scale
// (`19.9` == `19.90`). This is what `Value`'s derived `PartialEq` needs for row/DISTINCT equality.
impl PartialEq for Decimal {
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

/// `10^exp` as an `i128`, or `None` on overflow / `exp > 38`.
const fn pow10(exp: u32) -> Option<i128> {
    10i128.checked_pow(exp)
}

impl Decimal {
    /// The zero value (scale 0).
    pub const ZERO: Self = Self {
        mantissa: 0,
        scale: 0,
    };

    /// An integer as a scale-0 decimal.
    #[must_use]
    pub const fn from_i64(value: i64) -> Self {
        Self {
            mantissa: value as i128,
            scale: 0,
        }
    }

    /// A 128-bit integer as a scale-0 decimal — e.g. an exact `i128` sum accumulator (`AVG(int)`).
    #[must_use]
    pub const fn from_i128(value: i128) -> Self {
        Self {
            mantissa: value,
            scale: 0,
        }
    }

    /// `true` if the value is exactly zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Parse `[-|+]ddd[.ddd]` exactly. Rejects anything else (empty, multiple dots, non-digits,
    /// or more than [`MAX_SCALE`] fractional digits).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (neg, body) = match s.as_bytes().first()? {
            b'-' => (true, s.get(1..)?),
            b'+' => (false, s.get(1..)?),
            _ => (false, s),
        };
        if body.is_empty() {
            return None;
        }
        let (int_part, frac_part) = match body.split_once('.') {
            Some((i, f)) => (i, f),
            None => (body, ""),
        };
        if frac_part.len() > MAX_SCALE as usize {
            return None;
        }
        // Allow an empty integer part ("`.5`") but not an empty whole token ("`.`").
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        let digits: String = format!("{int_part}{frac_part}");
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut mantissa: i128 = digits.parse().ok()?;
        if neg {
            mantissa = -mantissa;
        }
        Some(Self {
            mantissa,
            scale: frac_part.len() as u8,
        })
    }

    /// Render in canonical form at the value's own scale (e.g. `1999`@2 -> `"19.99"`).
    #[must_use]
    pub fn format(&self) -> String {
        if self.scale == 0 {
            return self.mantissa.to_string();
        }
        let neg = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let scale = self.scale as usize;
        let sign = if neg { "-" } else { "" };
        if digits.len() > scale {
            let point = digits.len() - scale;
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        } else {
            // Pad with leading zeros: 5@2 -> "0.05".
            format!("{sign}0.{digits:0>scale$}")
        }
    }

    /// Re-scale to `target` fractional digits, rounding half-away-from-zero when shrinking.
    /// `None` on overflow.
    #[must_use]
    pub fn rescale(&self, target: u8) -> Option<Self> {
        match target.cmp(&self.scale) {
            Ordering::Equal => Some(*self),
            Ordering::Greater => {
                let factor = pow10(u32::from(target - self.scale))?;
                Some(Self {
                    mantissa: self.mantissa.checked_mul(factor)?,
                    scale: target,
                })
            },
            Ordering::Less => {
                let factor = pow10(u32::from(self.scale - target))?;
                let q = self.mantissa / factor;
                let r = self.mantissa % factor;
                let rounded = if r.unsigned_abs() * 2 >= factor.unsigned_abs() {
                    q + self.mantissa.signum()
                } else {
                    q
                };
                Some(Self {
                    mantissa: rounded,
                    scale: target,
                })
            },
        }
    }

    /// Total order by numeric value (scale-independent).
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        let scale = self.scale.max(other.scale);
        match (self.rescale(scale), other.rescale(scale)) {
            (Some(a), Some(b)) => a.mantissa.cmp(&b.mantissa),
            // Overflow on alignment is astronomically unlikely; fall back to a raw compare.
            _ => self.mantissa.cmp(&other.mantissa),
        }
    }

    /// Exact addition. `None` on overflow.
    #[must_use]
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let scale = self.scale.max(other.scale);
        let a = self.rescale(scale)?;
        let b = other.rescale(scale)?;
        Some(Self {
            mantissa: a.mantissa.checked_add(b.mantissa)?,
            scale,
        })
    }

    /// Exact subtraction. `None` on overflow.
    #[must_use]
    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.checked_add(&other.neg())
    }

    /// Exact multiplication (scales add). `None` on overflow / scale past [`MAX_SCALE`].
    #[must_use]
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        let scale = self.scale.checked_add(other.scale)?;
        if scale > MAX_SCALE {
            // Multiply then round back to MAX_SCALE so the scale stays representable.
            let product = self.mantissa.checked_mul(other.mantissa)?;
            return Self {
                mantissa: product,
                scale,
            }
            .rescale(MAX_SCALE);
        }
        Some(Self {
            mantissa: self.mantissa.checked_mul(other.mantissa)?,
            scale,
        })
    }

    /// Division at a result scale of `min(max(self.scale, other.scale) + DIV_GUARD, MAX_SCALE)`,
    /// rounding half-away-from-zero — enough fractional digits to keep ~16+ significant digits
    /// `None` on overflow (caller checks `other.is_zero()` for div-by-zero).
    #[must_use]
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        if other.mantissa == 0 {
            return None;
        }
        let result_scale = self
            .scale
            .max(other.scale)
            .saturating_add(DIV_GUARD)
            .min(MAX_SCALE);
        // value = (self.m / 10^self.s) / (other.m / 10^other.s)
        //       = self.m * 10^other.s / (other.m * 10^self.s)
        // want mantissa at result_scale: rm = round(self.m * 10^(result_scale + other.s - self.s) / other.m)
        let exp = i32::from(result_scale) + i32::from(other.scale) - i32::from(self.scale);
        let (num, den) = if exp >= 0 {
            (
                self.mantissa.checked_mul(pow10(exp.unsigned_abs())?)?,
                other.mantissa,
            )
        } else {
            (
                self.mantissa,
                other.mantissa.checked_mul(pow10(exp.unsigned_abs())?)?,
            )
        };
        let q = num / den;
        let r = num % den;
        let rounded = if r.unsigned_abs() * 2 >= den.unsigned_abs() {
            // Round away from zero; the quotient's sign is sign(num) * sign(den).
            q + num.signum() * den.signum()
        } else {
            q
        };
        Some(Self {
            mantissa: rounded,
            scale: result_scale,
        })
    }

    /// Modulo (remainder), aligning scales. `None` on overflow / div-by-zero.
    #[must_use]
    pub fn checked_rem(&self, other: &Self) -> Option<Self> {
        if other.mantissa == 0 {
            return None;
        }
        let scale = self.scale.max(other.scale);
        let a = self.rescale(scale)?;
        let b = other.rescale(scale)?;
        Some(Self {
            mantissa: a.mantissa % b.mantissa,
            scale,
        })
    }

    /// Numeric negation.
    #[must_use]
    pub const fn neg(&self) -> Self {
        Self {
            mantissa: self.mantissa.wrapping_neg(),
            scale: self.scale,
        }
    }

    /// Lossy conversion to `f64` (for mixed Numeric/Float arithmetic + stats).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "intentional lossy Numeric->f64 for mixed-type arithmetic + stats"
    )]
    pub fn to_f64(self) -> f64 {
        (self.mantissa as f64) / 10f64.powi(i32::from(self.scale))
    }

    /// Truncating conversion to `i64` (drops the fractional part). `None` if the integer part
    /// does not fit.
    #[must_use]
    pub fn to_i64(self) -> Option<i64> {
        let factor = pow10(u32::from(self.scale))?;
        i64::try_from(self.mantissa / factor).ok()
    }

    /// Rounding conversion to `i64`, rounding half-away-from-zero (`2.5 -> 3`, `-2.5 -> -3`). This is
    /// the SQL `CAST(numeric AS integer)` semantics — distinct from [`to_i64`](Self::to_i64), which
    /// truncates toward zero. `None` if the rounded integer does not fit.
    #[must_use]
    pub fn to_i64_rounded(self) -> Option<i64> {
        self.rescale(0)?.to_i64()
    }

    /// The `NUMERIC` precision this value occupies at its current scale: integer-part digits plus
    /// the scale.
    ///
    /// A value below 1 (e.g. `0.05`) still occupies `scale` fractional places, so counting mantissa
    /// digits alone undercounts it — `0.05` is precision 2, not 1. Used for `NUMERIC(p, s)`
    /// bound checks, where the value has already been rescaled to the column's scale.
    #[must_use]
    pub fn required_precision(&self) -> u32 {
        let scale = u32::from(self.scale);
        // `10^scale` as the divisor that strips the fractional part; saturate on an absurd scale
        // (which is itself an invalid declaration) rather than panic.
        let factor = 10u128.checked_pow(scale).unwrap_or(u128::MAX);
        let int_part = self.mantissa.unsigned_abs() / factor;
        let int_digits = if int_part == 0 {
            0
        } else {
            int_part.ilog10() + 1
        };
        int_digits + scale
    }

    /// The fewest fractional digits needed to represent this value exactly — i.e. the scale left
    /// after dropping trailing zeros (`12.340` → 2, `120` → 0, `0.000` → 0). `MIN_SCALE` (B-fn).
    #[must_use]
    pub const fn min_scale(&self) -> u8 {
        let mut mantissa = self.mantissa;
        let mut scale = self.scale;
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        scale
    }

    /// This value with trailing fractional zeros removed (`12.340` → `12.34`), denoting the same
    /// number at the smallest scale. `TRIM_SCALE` (B-fn).
    #[must_use]
    pub const fn trim_scale(&self) -> Self {
        let mut mantissa = self.mantissa;
        let mut scale = self.scale;
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        Self { mantissa, scale }
    }
}

/// Build a decimal from an `f64` via its shortest round-tripping decimal text, so a float literal
/// like `19.99` lands in a NUMERIC column as the exact decimal the user wrote. `None` for
/// non-finite values.
#[must_use]
pub fn from_f64_text(value: f64) -> Option<Decimal> {
    if !value.is_finite() {
        return None;
    }
    Decimal::parse(&value.to_string())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "unit-test assertions unwrap known-good inputs"
)]
mod tests {
    use super::*;

    fn d(s: &str) -> Decimal {
        Decimal::parse(s).unwrap()
    }

    #[test]
    fn parse_and_format_round_trip() {
        for s in [
            "0",
            "19.99",
            "-19.99",
            "100",
            "0.05",
            "-0.001",
            "12345.6789",
        ] {
            assert_eq!(d(s).format(), s);
        }
        assert_eq!(d(".5").format(), "0.5");
        assert!(Decimal::parse("abc").is_none());
        assert!(Decimal::parse("1.2.3").is_none());
        assert!(Decimal::parse("").is_none());
    }

    #[test]
    fn equality_is_scale_independent() {
        assert_eq!(d("19.9"), d("19.90"));
        assert_eq!(d("19.900"), d("19.9"));
        assert_ne!(d("19.9"), d("19.91"));
        assert_eq!(Decimal::from_i64(5), d("5"));
    }

    #[test]
    fn add_sub_are_exact() {
        assert_eq!(d("0.1").checked_add(&d("0.2")).unwrap(), d("0.3"));
        assert_eq!(d("19.99").checked_add(&d("0.01")).unwrap(), d("20.00"));
        assert_eq!(d("100").checked_sub(&d("0.01")).unwrap(), d("99.99"));
    }

    #[test]
    fn mul_adds_scales_exactly() {
        assert_eq!(d("1.5").checked_mul(&d("2")).unwrap(), d("3.0"));
        assert_eq!(d("0.1").checked_mul(&d("0.1")).unwrap(), d("0.01"));
    }

    #[test]
    fn div_keeps_enough_significant_digits() {
        // Division carries ~16+ significant digits, not a truncated 6.
        // 10 / 3 → scale max(0,0)+DIV_GUARD = 16.
        assert_eq!(
            d("10").checked_div(&d("3")).unwrap(),
            d("3.3333333333333333")
        );
        // The money case from the audit: 2.00 / 3 keeps full precision (scale max(2,0)+16 = 18),
        // not the old 0.666667.
        assert_eq!(
            d("2.00").checked_div(&d("3")).unwrap(),
            d("0.666666666666666667")
        );
        // 1 / 8 = 0.125 exactly (trailing zeros compare equal regardless of scale).
        assert_eq!(d("1").checked_div(&d("8")).unwrap(), d("0.125"));
        // Division by zero is rejected.
        assert!(d("1").checked_div(&Decimal::ZERO).is_none());
    }

    #[test]
    fn rescale_rounds() {
        assert_eq!(d("19.999").rescale(2).unwrap(), d("20.00"));
        assert_eq!(d("19.994").rescale(2).unwrap(), d("19.99"));
        assert_eq!(d("-19.995").rescale(2).unwrap(), d("-20.00"));
        assert_eq!(d("5").rescale(2).unwrap(), d("5.00"));
    }

    #[test]
    fn to_i64_rounded_rounds_half_away_from_zero() {
        // SQL CAST(numeric AS integer) rounds, unlike the truncating `to_i64`.
        assert_eq!(d("2.6").to_i64_rounded(), Some(3));
        assert_eq!(d("2.6").to_i64(), Some(2)); // contrast: truncation
        assert_eq!(d("3.5").to_i64_rounded(), Some(4));
        assert_eq!(d("2.5").to_i64_rounded(), Some(3));
        assert_eq!(d("2.4").to_i64_rounded(), Some(2));
        assert_eq!(d("-2.6").to_i64_rounded(), Some(-3));
        assert_eq!(d("-2.5").to_i64_rounded(), Some(-3));
        assert_eq!(d("-2.4").to_i64_rounded(), Some(-2));
        assert_eq!(d("5").to_i64_rounded(), Some(5));
        assert_eq!(Decimal::ZERO.to_i64_rounded(), Some(0));
    }

    #[test]
    fn min_scale_and_trim_scale_drop_trailing_zeros() {
        // `12.340` keeps scale 3 but only needs 2 fractional digits.
        assert_eq!(d("12.340").scale, 3);
        assert_eq!(d("12.340").min_scale(), 2);
        assert_eq!(d("12.340").trim_scale(), d("12.34"));
        // A value with no trailing zeros is unchanged.
        assert_eq!(d("12.34").min_scale(), 2);
        assert_eq!(d("12.34").trim_scale(), d("12.34"));
        // A whole number (with or without fractional zeros) needs scale 0.
        assert_eq!(d("120.00").min_scale(), 0);
        assert_eq!(d("120.00").trim_scale(), d("120"));
        assert_eq!(d("120.00").trim_scale().scale, 0);
        // Zero trims to scale 0.
        assert_eq!(d("0.000").min_scale(), 0);
        assert_eq!(d("0.000").trim_scale(), Decimal::ZERO);
    }

    #[test]
    fn conversions() {
        assert_eq!(Decimal::from_i64(42).to_i64(), Some(42));
        assert_eq!(d("19.99").to_i64(), Some(19));
        assert!((d("19.99").to_f64() - 19.99).abs() < 1e-9);
        assert_eq!(from_f64_text(19.99).unwrap(), d("19.99"));
        assert_eq!(d("123.45").required_precision(), 5);
        // A sub-1 value occupies `scale` places even though its mantissa has fewer digits:
        // 0.05 is precision 2 (mantissa 5 alone would mis-count as 1).
        assert_eq!(d("0.05").required_precision(), 2);
        assert_eq!(d("0.001").required_precision(), 3);
        assert_eq!(d("0").required_precision(), 0);
    }
}
