//! `INTERVAL` type — a calendar duration (phase 5).
//!
//! Represented as three independent components — **months**, **days**, and
//! **microseconds** — because a month and a day are not fixed multiples of microseconds (months
//! vary in length; days vary across DST, though NusaDB timestamps are UTC). This is the in-memory
//! form of [`ast::Value::Interval`](crate::ast::Value::Interval). Adding an interval to a
//! `TIMESTAMP`/`DATE` is calendar-aware (see [`crate::temporal::add_interval_to_micros`]).
//!
//! Two intervals compare/equate by a canonical microsecond estimate (`1 month = 30 days`,
//! `1 day = 24 h`) by calendar convention — so `INTERVAL '1 day' = INTERVAL '24:00:00'`.

use std::cmp::Ordering;

const MICROS_PER_SEC: i64 = 1_000_000;
const MICROS_PER_DAY: i64 = 86_400 * MICROS_PER_SEC;
const DAYS_PER_MONTH_EST: i64 = 30;

/// A calendar duration: months + days + microseconds (each may be negative).
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    /// Whole months (years fold in as `12 * years`).
    pub months: i32,
    /// Whole days (weeks fold in as `7 * weeks`).
    pub days: i32,
    /// Sub-day microseconds (hours/minutes/seconds fold in here).
    pub micros: i64,
}

// Equality + ordering by a canonical microsecond estimate (calendar semantics): `1 month` is
// estimated as 30 days and `1 day` as 24 h, so `'1 day' == '24:00:00'`.
impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        self.estimate_micros() == other.estimate_micros()
    }
}

impl Interval {
    /// The zero interval.
    pub const ZERO: Self = Self {
        months: 0,
        days: 0,
        micros: 0,
    };

    /// Build an interval from the individual `MAKE_INTERVAL(years, months, weeks, days, hours, mins,
    /// secs)` fields. Years fold into months and weeks into days; hours/minutes/seconds fold
    /// into microseconds. Every fold uses saturating arithmetic and clamps the month/day totals to
    /// `i32`, so even absurd field values stay panic-free.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "secs*1e6 is rounded then saturatingly cast to i64; the f64->i64 `as` saturates"
    )]
    pub fn make(
        years: i64,
        months: i64,
        weeks: i64,
        days: i64,
        hours: i64,
        mins: i64,
        secs: f64,
    ) -> Self {
        fn clamp_i32(v: i64) -> i32 {
            i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX })
        }
        let total_months = years.saturating_mul(12).saturating_add(months);
        let total_days = weeks.saturating_mul(7).saturating_add(days);
        let hm_micros = hours
            .saturating_mul(3_600_000_000)
            .saturating_add(mins.saturating_mul(60_000_000));
        let sec_micros = (secs * 1_000_000.0).round() as i64;
        Self {
            months: clamp_i32(total_months),
            days: clamp_i32(total_days),
            micros: hm_micros.saturating_add(sec_micros),
        }
    }

    /// A canonical microsecond estimate used for comparison/equality only (not for date math).
    #[must_use]
    pub fn estimate_micros(&self) -> i128 {
        i128::from(self.months) * i128::from(DAYS_PER_MONTH_EST) * i128::from(MICROS_PER_DAY)
            + i128::from(self.days) * i128::from(MICROS_PER_DAY)
            + i128::from(self.micros)
    }

    /// Total order by the canonical estimate.
    #[must_use]
    pub fn compare(&self, other: &Self) -> Ordering {
        self.estimate_micros().cmp(&other.estimate_micros())
    }

    /// Component-wise addition, or `None` on any component overflow. Previously
    /// `wrapping_add` silently wrapped a large/accumulated interval to a wrong duration as success.
    #[must_use]
    pub const fn checked_add(&self, other: &Self) -> Option<Self> {
        let (Some(months), Some(days), Some(micros)) = (
            self.months.checked_add(other.months),
            self.days.checked_add(other.days),
            self.micros.checked_add(other.micros),
        ) else {
            return None;
        };
        Some(Self {
            months,
            days,
            micros,
        })
    }

    /// Component-wise negation, or `None` on overflow (negating `i32::MIN` / `i64::MIN`).
    #[must_use]
    pub const fn checked_neg(&self) -> Option<Self> {
        let (Some(months), Some(days), Some(micros)) = (
            self.months.checked_neg(),
            self.days.checked_neg(),
            self.micros.checked_neg(),
        ) else {
            return None;
        };
        Some(Self {
            months,
            days,
            micros,
        })
    }

    /// Component-wise subtraction, or `None` on overflow.
    #[must_use]
    pub const fn checked_sub(&self, other: &Self) -> Option<Self> {
        let Some(negated) = other.checked_neg() else {
            return None;
        };
        self.checked_add(&negated)
    }

    /// Scale every component by an integer `factor` (`interval * n`), or `None` on overflow. Each
    /// field is multiplied independently — months stay months, days stay days — so no remainder is
    /// spilled across units (the calendar-aware spill only happens when an interval is applied to a
    /// timestamp).
    #[must_use]
    pub fn checked_mul(&self, factor: i64) -> Option<Self> {
        let months = i64::from(self.months)
            .checked_mul(factor)
            .and_then(|v| i32::try_from(v).ok())?;
        let days = i64::from(self.days)
            .checked_mul(factor)
            .and_then(|v| i32::try_from(v).ok())?;
        let micros = self.micros.checked_mul(factor)?;
        Some(Self {
            months,
            days,
            micros,
        })
    }

    /// Normalize so every 30 days rolls up into one month (`JUSTIFY_DAYS`, B-fn); the day field keeps
    /// its sign in `[-29, 29]`. Overflow of the month field saturates rather than wrapping.
    #[must_use]
    pub const fn justify_days(&self) -> Self {
        Self {
            months: self.months.saturating_add(self.days / 30),
            days: self.days % 30,
            micros: self.micros,
        }
    }

    /// Normalize so every 24 hours rolls up into one day (`JUSTIFY_HOURS`, B-fn); the micro field
    /// keeps its sign in `(-24h, 24h)`. Overflow of the day field saturates rather than wrapping.
    #[must_use]
    pub fn justify_hours(&self) -> Self {
        const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;
        let extra_days = self.micros / MICROS_PER_DAY;
        let days = i64::from(self.days)
            .saturating_add(extra_days)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        Self {
            months: self.months,
            // `days` is clamped to the `i32` range above, so the conversion never fails.
            days: i32::try_from(days).unwrap_or(self.days),
            micros: self.micros % MICROS_PER_DAY,
        }
    }

    /// Apply both [`Self::justify_hours`] and [`Self::justify_days`] so days land in `[-29, 29]` and
    /// the time part in `(-24h, 24h)` (`JUSTIFY_INTERVAL`, B-fn).
    #[must_use]
    pub fn justify_interval(&self) -> Self {
        self.justify_hours().justify_days()
    }

    /// Parse an interval literal: a sequence of `N unit` terms (`unit` is one of
    /// `year`, `month`, `week`, `day`, `hour`, `minute`, `second` — abbreviations + plurals
    /// accepted) and/or an `HH:MM:SS.ffffff` clock term. Examples: `1 day`, `2 hours`,
    /// `1 year 2 months`, `1 day 03:04:05`, `-1 mon`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let mut out = Self::ZERO;
        let mut tokens = s.split_whitespace();
        let mut saw_any = false;
        while let Some(tok) = tokens.next() {
            if tok.contains(':') {
                // Clock component HH:MM:SS[.ffffff], optionally signed.
                out.micros = out.micros.checked_add(parse_clock(tok)?)?;
                saw_any = true;
                continue;
            }
            // Otherwise a `N unit` pair.
            let n: i64 = tok.parse().ok()?;
            let unit = tokens.next()?;
            apply_unit(&mut out, n, unit)?;
            saw_any = true;
        }
        saw_any.then_some(out)
    }

    /// Render in a human-readable form, e.g. `1 year 2 mons 3 days 04:05:06`. Zero is `00:00:00`.
    #[must_use]
    pub fn format(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let years = self.months / 12;
        let mons = self.months % 12;
        if years != 0 {
            parts.push(format!("{years} year{}", plural(i64::from(years))));
        }
        if mons != 0 {
            parts.push(format!("{mons} mon{}", plural(i64::from(mons))));
        }
        if self.days != 0 {
            parts.push(format!("{} day{}", self.days, plural(i64::from(self.days))));
        }
        if self.micros != 0 || parts.is_empty() {
            parts.push(format_clock(self.micros));
        }
        parts.join(" ")
    }
}

const fn plural(n: i64) -> &'static str {
    // Only exactly +1 is singular; -1 (and every other count) is plural — e.g. `-1 days`, matching
    // the conventional interval rendering rather than treating -1 as singular.
    if n == 1 { "" } else { "s" }
}

/// Add `n` of `unit` to `out`. Years/months fold into months; weeks/days into days; hour/min/sec
/// into micros. Returns `None` on an unknown unit or arithmetic overflow.
fn apply_unit(out: &mut Interval, n: i64, unit: &str) -> Option<()> {
    let u = unit.trim_end_matches('s').to_ascii_lowercase();
    // (months delta, days delta, micros delta) for one of `n` units.
    let (months, days, micros): (i64, i64, i64) = match u.as_str() {
        "year" | "yr" => (12, 0, 0),
        "mon" | "month" => (1, 0, 0),
        "week" | "wk" => (0, 7, 0),
        "day" => (0, 1, 0),
        "hour" | "hr" => (0, 0, 3600 * MICROS_PER_SEC),
        "min" | "minute" => (0, 0, 60 * MICROS_PER_SEC),
        "sec" | "second" => (0, 0, MICROS_PER_SEC),
        _ => return None,
    };
    out.months = out
        .months
        .checked_add(i32::try_from(n.checked_mul(months)?).ok()?)?;
    out.days = out
        .days
        .checked_add(i32::try_from(n.checked_mul(days)?).ok()?)?;
    out.micros = out.micros.checked_add(n.checked_mul(micros)?)?;
    Some(())
}

/// Parse `[-]HH:MM:SS[.ffffff]` (or `HH:MM`) into signed microseconds.
fn parse_clock(s: &str) -> Option<i64> {
    let (sign, body) = s.strip_prefix('-').map_or((1i64, s), |rest| (-1, rest));
    let mut it = body.split(':');
    let h: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next()?.parse().ok()?;
    let (sec, frac) = match it.next() {
        Some(secs) => match secs.split_once('.') {
            Some((whole, f)) => (whole.parse::<i64>().ok()?, parse_frac(f)?),
            None => (secs.parse::<i64>().ok()?, 0),
        },
        None => (0, 0),
    };
    if it.next().is_some() {
        return None;
    }
    // The leading sign was already stripped, so each component must be non-negative, and minutes
    // and seconds must be 0..=59 — otherwise `'1:99:99'` or `'1:-30:00'` would parse to a nonsense
    // duration. Hours are unbounded (an interval may be `100:00:00`).
    if h < 0 || !(0..60).contains(&m) || !(0..60).contains(&sec) {
        return None;
    }
    Some(sign * ((h * 3600 + m * 60 + sec) * MICROS_PER_SEC + frac))
}

/// Fractional seconds digits → microseconds (truncating beyond 6).
fn parse_frac(f: &str) -> Option<i64> {
    if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut micros = 0i64;
    for i in 0..6 {
        micros *= 10;
        if let Some(c) = f.as_bytes().get(i) {
            micros += i64::from(c - b'0');
        }
    }
    Some(micros)
}

/// Render sub-day micros as `[-]HH:MM:SS[.ffffff]`.
fn format_clock(micros: i64) -> String {
    let neg = micros < 0;
    // `unsigned_abs` (not `abs`) so `i64::MIN` micros — reachable from interval arithmetic — formats
    // instead of panicking (debug) / wrapping to garbage (release).
    let micros = micros.unsigned_abs();
    let per_sec = MICROS_PER_SEC.unsigned_abs();
    let secs = micros / per_sec;
    let frac = micros % per_sec;
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let sign = if neg { "-" } else { "" };
    if frac == 0 {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    } else {
        format!("{sign}{h:02}:{m:02}:{s:02}.{frac:06}")
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "unit-test assertions unwrap known-good inputs"
)]
mod tests {
    use super::*;

    fn p(s: &str) -> Interval {
        Interval::parse(s).unwrap()
    }

    #[test]
    fn parse_and_format() {
        assert_eq!(p("1 day").format(), "1 day");
        assert_eq!(p("2 hours").format(), "02:00:00");
        assert_eq!(p("1 year 2 months").format(), "1 year 2 mons");
        assert_eq!(p("1 day 03:04:05").format(), "1 day 03:04:05");
        assert_eq!(p("90 minutes").format(), "01:30:00");
        assert!(Interval::parse("nonsense").is_none());
    }

    #[test]
    fn clock_components_out_of_range_are_rejected() {
        // G18: minutes/seconds must be 0..=59 and components non-negative — otherwise a nonsense
        // clock term parses to a bogus duration. Hours are unbounded.
        assert!(Interval::parse("1:99:00").is_none());
        assert!(Interval::parse("1:00:99").is_none());
        assert!(Interval::parse("1:-30:00").is_none());
        assert_eq!(p("100:00:00").format(), "100:00:00"); // large hours still valid
        assert_eq!(p("01:59:59").format(), "01:59:59"); // boundary values accepted
    }

    #[test]
    fn equality_uses_canonical_estimate() {
        assert_eq!(p("1 day"), p("24:00:00"));
        assert_eq!(p("1 mon"), p("30 days"));
        assert_ne!(p("1 day"), p("2 days"));
    }

    #[test]
    fn add_sub_neg() {
        let a = p("1 day");
        let b = p("2 hours");
        assert_eq!(a.checked_add(&b).unwrap().format(), "1 day 02:00:00");
        assert_eq!(a.checked_sub(&b).unwrap().format(), "1 day -02:00:00");
        // -1 is plural (only +1 is singular), matching the conventional rendering.
        assert_eq!(a.checked_neg().unwrap().format(), "-1 days");
    }

    #[test]
    fn only_positive_one_is_singular_in_format() {
        // Only exactly +1 is singular; -1 (and any other count) is plural.
        assert_eq!(p("1 day").format(), "1 day");
        assert_eq!(p("-1 day").format(), "-1 days");
        assert_eq!(p("1 year").format(), "1 year");
        assert_eq!(p("-1 year").format(), "-1 years");
        assert_eq!(p("-1 mon").format(), "-1 mons");
        assert_eq!(p("2 days").format(), "2 days");
    }

    #[test]
    fn mul_scales_each_component() {
        let scaled = p("1 month 2 days 3 hours").checked_mul(3).unwrap();
        assert_eq!(
            (scaled.months, scaled.days, scaled.micros),
            (3, 6, 9 * 3_600 * 1_000_000)
        );
        assert_eq!(p("5 days").checked_mul(0).unwrap(), p("0 days"));
        assert_eq!(p("2 days").checked_mul(-1).unwrap(), p("-2 days"));
        // Overflow (months exceeds i32) is reported as None, not wrapped.
        assert!(p("1 month").checked_mul(i64::MAX).is_none());
    }

    #[test]
    fn arithmetic_overflow_is_reported_not_wrapped() {
        // A component at its bound must not silently wrap to a wrong duration.
        let max = Interval {
            months: 0,
            days: 0,
            micros: i64::MAX,
        };
        let one = Interval {
            months: 0,
            days: 0,
            micros: 1,
        };
        assert!(max.checked_add(&one).is_none(), "micros overflow → None");
        let min = Interval {
            months: i32::MIN,
            days: 0,
            micros: 0,
        };
        assert!(min.checked_neg().is_none(), "negating i32::MIN → None");
        // `min - {months:1}` = i32::MIN - 1 → overflow.
        let one_month = Interval {
            months: 1,
            days: 0,
            micros: 0,
        };
        assert!(min.checked_sub(&one_month).is_none(), "sub overflow → None");
    }

    #[test]
    fn format_clock_handles_i64_min_micros_without_panicking() {
        // `abs()` on i64::MIN would panic/wrap; `unsigned_abs` formats it.
        let iv = Interval {
            months: 0,
            days: 0,
            micros: i64::MIN,
        };
        // Just assert it produces a negative-signed string rather than panicking.
        assert!(iv.format().starts_with('-'), "got {}", iv.format());
    }

    #[test]
    fn justify_rolls_days_and_hours_into_higher_units() {
        // JUSTIFY_DAYS: 35 days → 1 month 5 days.
        let jd = p("35 days").justify_days();
        assert_eq!((jd.months, jd.days, jd.micros), (1, 5, 0));
        // JUSTIFY_HOURS: 49 hours → 2 days 1 hour.
        let jh = p("49 hours").justify_hours();
        assert_eq!((jh.months, jh.days, jh.micros), (0, 2, 3_600 * 1_000_000));
        // JUSTIFY_INTERVAL applies both: 35 days + 49 h → 1 mon 7 days 1 h.
        let ji = p("35 days 49 hours").justify_interval();
        assert_eq!((ji.months, ji.days, ji.micros), (1, 7, 3_600 * 1_000_000));
        // Negative components keep their sign in range.
        let neg = p("-35 days").justify_days();
        assert_eq!((neg.months, neg.days, neg.micros), (-1, -5, 0));
    }
}
