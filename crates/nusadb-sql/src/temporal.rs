//! Dependency-free parse + format helpers for the temporal and UUID types.
//!
//! Backing representations (matching [`ast::Value`](crate::ast::Value) /
//! [`ColumnType`](nusadb_core::ColumnType)):
//! - **Date** — days since 1970-01-01, proleptic Gregorian (Howard Hinnant's day algorithms).
//! - **Time** — microseconds since midnight, `[0, 86_400_000_000)`.
//! - **Timestamp** / **TimestampTz** — microseconds since 1970-01-01T00:00:00 (UTC for the tz form).
//! - **Uuid** — 16 raw bytes.
//!
//! All parsing is strict ISO-8601-ish and all formatting is canonical, so a `text -> value -> text`
//! round-trip is stable (the property the SQL layer relies on for display + equality).

#![allow(
    clippy::doc_markdown,
    reason = "prose names like ISO-8601 / Howard Hinnant read better unbackticked"
)]

use std::fmt::Write as _;

const MICROS_PER_SEC: i64 = 1_000_000;
const SECS_PER_DAY: i64 = 86_400;
const MICROS_PER_DAY: i64 = SECS_PER_DAY * MICROS_PER_SEC;

// ---- Date <-> civil (year, month, day) -------------------------------------------------------

/// Days since 1970-01-01 for a proleptic-Gregorian `(y, m, d)`. Hinnant's algorithm.
const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + (d - 1); // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Format a microseconds-since-epoch timestamp as a compact `YYYYMMDDHHMMSS` stamp.
///
/// All digits, no separators — safe to embed in an SQL identifier. Used to name a `DROP DATABASE`
/// backup table (`{database}_{stamp}_{table}`, a NusaDB safety extension). Deterministic in the input.
#[must_use]
pub fn compact_stamp(ts_micros: i64) -> String {
    let days = ts_micros.div_euclid(MICROS_PER_DAY);
    let tod = ts_micros.rem_euclid(MICROS_PER_DAY);
    let (y, m, d) = civil_from_days(days);
    let secs = tod / MICROS_PER_SEC;
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{y:04}{m:02}{d:02}{hh:02}{mm:02}{ss:02}")
}

/// Inverse of [`days_from_civil`]: `(year, month, day)` from days since the epoch.
const fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Days in month `m` (1..=12) of year `y` (proleptic Gregorian).
const fn days_in_month(y: i64, m: i64) -> i64 {
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        _ => 28,
    }
}

/// Add an interval (`months` + `days` + `micros`) to a timestamp, calendar-aware.
///
/// `ts` is microseconds since the epoch. Months advance the year/month and clamp the day to the new
/// month's length; then whole days and the sub-day microseconds are added.
#[must_use]
pub fn add_interval_to_micros(ts: i64, months: i32, days: i32, micros: i64) -> i64 {
    let (day_count, tod) = (ts.div_euclid(MICROS_PER_DAY), ts.rem_euclid(MICROS_PER_DAY));
    let (y, m, d) = civil_from_days(day_count);
    // Advance months (0-based month arithmetic), then clamp the day.
    let total_months = y * 12 + (m - 1) + i64::from(months);
    let ny = total_months.div_euclid(12);
    let nm = total_months.rem_euclid(12) + 1;
    let nd = d.min(days_in_month(ny, nm));
    let new_days = days_from_civil(ny, nm, nd) + i64::from(days);
    new_days
        .saturating_mul(MICROS_PER_DAY)
        .saturating_add(tod)
        .saturating_add(micros)
}

/// `DATE_BIN(stride, source, origin)` — snap the timestamp `source` down to its `stride`-wide bin
/// aligned to `origin`.
///
/// Timestamps are microseconds since the epoch. The stride is a *fixed* duration of `stride_days`
/// whole days + `stride_micros` sub-day microseconds (days count as 24 h); a month/year component is
/// not allowed (the caller rejects it). Returns the bin start, or `None` when the stride is
/// non-positive or the result falls outside the `i64` microsecond range.
#[must_use]
pub fn date_bin(stride_days: i32, stride_micros: i64, source: i64, origin: i64) -> Option<i64> {
    let stride = i128::from(stride_days) * i128::from(MICROS_PER_DAY) + i128::from(stride_micros);
    if stride <= 0 {
        return None;
    }
    // Floor-divide the offset from `origin` into whole strides (`div_euclid` floors for a positive
    // divisor, so negative timestamps bin downward correctly), then step back out from `origin`.
    let n = (i128::from(source) - i128::from(origin)).div_euclid(stride);
    i64::try_from(i128::from(origin) + n * stride).ok()
}

fn is_valid_ymd(y: i64, m: i64, d: i64) -> bool {
    if !(1..=12).contains(&m) || d < 1 {
        return false;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    // `m` is in `1..=12`, so the index is in bounds; `try_from` + `.get` keep it panic-free.
    usize::try_from(m - 1)
        .ok()
        .and_then(|i| dim.get(i))
        .is_some_and(|&max| d <= max)
}

// ---- Date ------------------------------------------------------------------------------------

/// Parse `YYYY-MM-DD` into days since the epoch.
#[must_use]
pub fn parse_date(s: &str) -> Option<i32> {
    let (y, m, d) = parse_ymd(s.trim())?;
    i32::try_from(days_from_civil(y, m, d)).ok()
}

/// Build a `DATE` (days since the epoch) from a `(year, month, day)` triple, or `None` if it is not a
/// real calendar day or falls outside the representable range (`MAKE_DATE`).
#[must_use]
pub fn make_date(y: i64, m: i64, d: i64) -> Option<i32> {
    if !is_valid_ymd(y, m, d) {
        return None;
    }
    i32::try_from(days_from_civil(y, m, d)).ok()
}

/// Build a `TIME` (microseconds since midnight) from `(hour, minute, second)`, or `None` if any field
/// is out of range (`MAKE_TIME`). v1 takes whole seconds.
#[must_use]
pub fn make_time(h: i64, m: i64, s: i64) -> Option<i64> {
    if !(0..24).contains(&h) || !(0..60).contains(&m) || !(0..60).contains(&s) {
        return None;
    }
    Some((h * 3600 + m * 60 + s) * MICROS_PER_SEC)
}

/// Build a `TIMESTAMP` (microseconds since the epoch) from `(year, month, day, hour, minute, second)`,
/// or `None` if the date or time fields are invalid (`MAKE_TIMESTAMP`). v1 takes whole seconds.
#[must_use]
pub fn make_timestamp(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> Option<i64> {
    if !is_valid_ymd(y, mo, d) {
        return None;
    }
    let tod = make_time(h, mi, s)?;
    days_from_civil(y, mo, d)
        .checked_mul(MICROS_PER_DAY)
        .and_then(|day_micros| day_micros.checked_add(tod))
}

/// Format days-since-epoch as `YYYY-MM-DD`.
#[must_use]
pub fn format_date(days: i32) -> String {
    let (y, m, d) = civil_from_days(i64::from(days));
    format!("{y:04}-{m:02}-{d:02}")
}

fn parse_ymd(s: &str) -> Option<(i64, i64, i64)> {
    let mut it = s.splitn(3, '-');
    let y: i64 = it.next()?.parse().ok()?;
    let m = parse_fixed(it.next()?, 2)?;
    let d = parse_fixed(it.next()?, 2)?;
    if y < 0 || !is_valid_ymd(y, m, d) {
        return None;
    }
    Some((y, m, d))
}

/// Parse a zero-padded unsigned field of exactly `width` digits into an `i64`.
fn parse_fixed(s: &str, width: usize) -> Option<i64> {
    if s.len() != width || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

// ---- Time ------------------------------------------------------------------------------------

/// Parse `HH:MM:SS[.ffffff]` into microseconds since midnight.
#[must_use]
pub fn parse_time(s: &str) -> Option<i64> {
    parse_time_of_day(s.trim())
}

/// Format microseconds-since-midnight as `HH:MM:SS` (or `HH:MM:SS.ffffff` with a fraction).
#[must_use]
pub fn format_time(micros: i64) -> String {
    let secs = micros / MICROS_PER_SEC;
    let frac = micros % MICROS_PER_SEC;
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if frac == 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{h:02}:{m:02}:{s:02}.{frac:06}")
    }
}

fn parse_time_of_day(s: &str) -> Option<i64> {
    let mut it = s.splitn(3, ':');
    let h = parse_fixed(it.next()?, 2)?;
    let m = parse_fixed(it.next()?, 2)?;
    // Seconds are optional: `HH:MM` (and `HH:MM` inside a timestamp) defaults the seconds to 0, the
    // same as the reference engine. `HH:MM:SS[.ffffff]` keeps its explicit seconds and fractional part.
    let (sec, frac_micros) = match it.next() {
        Some(sec_field) => {
            let (sec_str, frac) = match sec_field.split_once('.') {
                Some((sec, frac)) => (sec, parse_fraction(frac)?),
                None => (sec_field, 0),
            };
            (parse_fixed(sec_str, 2)?, frac)
        },
        None => (0, 0),
    };
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    Some(((h * 3600 + m * 60 + sec) * MICROS_PER_SEC) + frac_micros)
}

/// Parse the fractional-seconds digits after the dot into microseconds (truncating beyond 6).
fn parse_fraction(frac: &str) -> Option<i64> {
    if frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut micros = 0i64;
    for i in 0..6 {
        micros *= 10;
        if let Some(c) = frac.as_bytes().get(i) {
            micros += i64::from(c - b'0');
        }
    }
    Some(micros)
}

// ---- TimeTz ------------------------------------------------------------------------------------
//
// A `TIMETZ` value is one packed `i64` carrying BOTH the as-entered local time-of-day and its
// zone offset (P-TIMETZ): `packed = utc_equivalent_micros * 2^18 + (zone_west_secs + 2^17)`.
// The packing is chosen so plain `i64` ordering implements the reference engine's `timetz_cmp` exactly — primary
// by the UTC-equivalent instant (deliberately NOT normalized into one day, like the reference engine), tie-broken
// by zone west-of-UTC — so the executor's compare / hash / order-preserving index-key / spill
// paths all stay untouched, and (like the reference engine) two equal instants with different zones are NOT equal.

/// Width of the zone field inside a packed `TIMETZ` (`2^18` — comfortably holds the parseable
/// ±23:59 offset range in seconds around the `2^17` bias).
const TIMETZ_ZONE_SPAN: i64 = 1 << 18;
/// Bias added to the zone-west seconds so the packed zone field is non-negative.
const TIMETZ_ZONE_BIAS: i64 = 1 << 17;

/// Pack a `timetz` from its local (as-entered) time-of-day `local_micros` and its zone offset
/// **east** of UTC in whole seconds (`+07` → `25_200`, `-05:30` → `-19_800`).
#[must_use]
pub const fn pack_timetz(local_micros: i64, offset_east_secs: i64) -> i64 {
    let zone_west = -offset_east_secs;
    let utc_equivalent = local_micros - offset_east_secs * MICROS_PER_SEC;
    utc_equivalent * TIMETZ_ZONE_SPAN + (zone_west + TIMETZ_ZONE_BIAS)
}

/// The zone offset east of UTC of a packed `timetz`, in seconds.
#[must_use]
pub const fn timetz_offset_east_secs(packed: i64) -> i64 {
    TIMETZ_ZONE_BIAS - packed.rem_euclid(TIMETZ_ZONE_SPAN)
}

/// The local (as-entered) time-of-day micros of a packed `timetz`.
#[must_use]
pub const fn timetz_local_micros(packed: i64) -> i64 {
    let offset_east = timetz_offset_east_secs(packed);
    packed.div_euclid(TIMETZ_ZONE_SPAN) + offset_east * MICROS_PER_SEC
}

/// Parse a `timetz` `HH:MM:SS[.ffffff][Z|±HH[:MM]]` into the packed local-time + zone form.
///
/// The offset is **kept** (P-TIMETZ), so the value renders back with the zone it was entered
/// with, exactly like the reference engine. A missing offset (or `Z`) is UTC (`+00`).
#[must_use]
pub fn parse_timetz(s: &str) -> Option<i64> {
    let mut time_part = s.trim();
    let mut offset_micros = 0i64;
    if let Some(stripped) = time_part.strip_suffix('Z') {
        time_part = stripped;
    } else if let Some(idx) = find_offset_sign(time_part) {
        let (t, off) = time_part.split_at(idx);
        offset_micros = parse_offset(off)?;
        time_part = t;
    }
    let tod = parse_time_of_day(time_part)?;
    Some(pack_timetz(tod, offset_micros / MICROS_PER_SEC))
}

/// Format a packed `timetz` as its local time with its zone offset — `HH:MM:SS[.ffffff]±HH`,
/// with `:MM` appended only when the offset has minutes (the reference engine's rendering: `+07`, `-05:30`).
#[must_use]
pub fn format_timetz(packed: i64) -> String {
    let time = format_time(timetz_local_micros(packed));
    let offset_east = timetz_offset_east_secs(packed);
    let sign = if offset_east < 0 { '-' } else { '+' };
    let abs = offset_east.abs();
    let (h, m, s) = (abs / 3600, (abs % 3600) / 60, abs % 60);
    if s != 0 {
        format!("{time}{sign}{h:02}:{m:02}:{s:02}")
    } else if m != 0 {
        format!("{time}{sign}{h:02}:{m:02}")
    } else {
        format!("{time}{sign}{h:02}")
    }
}

// ---- Timestamp / TimestampTz -----------------------------------------------------------------

/// Parse `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]` into microseconds since the epoch.
#[must_use]
pub fn parse_timestamp(s: &str) -> Option<i64> {
    parse_timestamp_inner(s.trim(), false)
}

/// Parse a timestamptz `YYYY-MM-DD[ T]HH:MM:SS[.ffffff][Z|±HH[:MM]]`, normalizing to UTC micros
/// since the epoch. A missing offset is treated as UTC.
#[must_use]
pub fn parse_timestamptz(s: &str) -> Option<i64> {
    parse_timestamp_inner(s.trim(), true)
}

/// Format micros-since-epoch as `YYYY-MM-DD HH:MM:SS[.ffffff]`.
#[must_use]
pub fn format_timestamp(micros: i64) -> String {
    let (days, tod) = div_floor_mod(micros, MICROS_PER_DAY);
    // `days` is `micros / 86_400_000_000`, so for any i64 micros `|days| ≤ ~1.07e8` — it always
    // fits i32. The clamp is therefore unreachable, but it preserves sign instead of falling back
    // to `0`, which would render an out-of-range timestamp as the epoch `1970-01-01`.
    let days = i32::try_from(days).unwrap_or(if days < 0 { i32::MIN } else { i32::MAX });
    format!("{} {}", format_date(days), format_time(tod))
}

/// Format micros-since-epoch (UTC) as `YYYY-MM-DD HH:MM:SS[.ffffff]+00`.
#[must_use]
pub fn format_timestamptz(micros: i64) -> String {
    format!("{}+00", format_timestamp(micros))
}

fn parse_timestamp_inner(s: &str, allow_offset: bool) -> Option<i64> {
    // Split date and time on the first 'T' or space. A date with no time part (`2024-03-15`) is a
    // timestamp at midnight, the same as the reference engine (`TIMESTAMP '2024-03-15'` → `2024-03-15 00:00:00`).
    let (date_part, mut time_part) = match s.find(['T', ' ']) {
        Some(sep) => (s.get(..sep)?, s.get(sep + 1..)?),
        None => (s, ""),
    };

    let mut offset_micros = 0i64;
    if allow_offset {
        if let Some(stripped) = time_part.strip_suffix('Z') {
            time_part = stripped;
        } else if let Some(idx) = find_offset_sign(time_part) {
            let (t, off) = time_part.split_at(idx);
            offset_micros = parse_offset(off)?;
            time_part = t;
        }
    }

    let (y, m, d) = parse_ymd(date_part)?;
    // An absent time part defaults to midnight; a present one must be valid `HH:MM[:SS[.ffffff]]`.
    let tod = if time_part.is_empty() {
        0
    } else {
        parse_time_of_day(time_part)?
    };
    days_from_civil(y, m, d)
        .checked_mul(MICROS_PER_DAY)?
        .checked_add(tod)?
        .checked_sub(offset_micros)
}

/// Find the index of a `+`/`-` that begins a trailing zone offset (not the leading hour).
fn find_offset_sign(time_part: &str) -> Option<usize> {
    time_part
        .bytes()
        .rposition(|b| b == b'+' || b == b'-')
        .filter(|&i| i > 0)
}

/// Parse `±HH`, `±HHMM`, or `±HH:MM` into a signed micro offset.
fn parse_offset(off: &str) -> Option<i64> {
    let (sign, rest) = match off.as_bytes().first()? {
        b'+' => (1i64, off.get(1..)?),
        b'-' => (-1i64, off.get(1..)?),
        _ => return None,
    };
    let rest = rest.replace(':', "");
    let (hh, mm) = match rest.len() {
        2 => (parse_fixed(&rest, 2)?, 0),
        4 => (
            parse_fixed(rest.get(..2)?, 2)?,
            parse_fixed(rest.get(2..)?, 2)?,
        ),
        _ => return None,
    };
    if hh > 23 || mm > 59 {
        return None;
    }
    Some(sign * (hh * 3600 + mm * 60) * MICROS_PER_SEC)
}

/// Parse a signed fixed time-zone offset `±HH`, `±HHMM`, or `±HH:MM` into microseconds, for the
/// `AT TIME ZONE` operator (a leading `+`/`-` is required). `None` for anything else.
#[must_use]
pub fn parse_zone_offset(off: &str) -> Option<i64> {
    parse_offset(off)
}

/// Floored division + modulo (handles negative `micros` so pre-epoch timestamps format correctly).
const fn div_floor_mod(a: i64, b: i64) -> (i64, i64) {
    (a.div_euclid(b), a.rem_euclid(b))
}

// ---- Field extraction / truncation / age ---------------------------------------------

/// Split an epoch-micros instant into calendar + clock components.
///
/// Returns `(year, month, day, hour, minute, second, microsecond)`. `month`/`day` are 1-based; the
/// clock fields are floored toward the previous midnight so pre-epoch instants decompose correctly.
#[allow(
    clippy::many_single_char_names,
    reason = "conventional y/m/d/h/s calendar component names"
)]
const fn decompose_micros(micros: i64) -> (i64, i64, i64, i64, i64, i64, i64) {
    let tod = micros.rem_euclid(MICROS_PER_DAY);
    let (y, m, d) = civil_from_days(micros.div_euclid(MICROS_PER_DAY));
    let h = tod / (3600 * MICROS_PER_SEC);
    let mi = (tod / (60 * MICROS_PER_SEC)) % 60;
    let s = (tod / MICROS_PER_SEC) % 60;
    let us = tod % MICROS_PER_SEC;
    (y, m, d, h, mi, s, us)
}

/// `EXTRACT(field FROM ts)` for a full timestamp (`field` already folded to lower case).
///
/// Returns `None` for an unrecognised field. SQL `EXTRACT` is double-precision, so the result is
/// `f64`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "EXTRACT yields a double-precision number; y/m/d/h/s are calendar components"
)]
pub fn extract_from_micros(field: &str, micros: i64) -> Option<f64> {
    let days = micros.div_euclid(MICROS_PER_DAY);
    let (y, m, d, h, mi, s, us) = decompose_micros(micros);
    let val = match field {
        "year" => y as f64,
        "month" => m as f64,
        "day" => d as f64,
        "hour" => h as f64,
        "minute" => mi as f64,
        "second" => s as f64 + us as f64 / MICROS_PER_SEC as f64,
        // `dow`: 0 = Sunday .. 6 = Saturday (1970-01-01 was a Thursday → day 0 maps to 4).
        "dow" => (days + 4).rem_euclid(7) as f64,
        // `isodow`: 1 = Monday .. 7 = Sunday.
        "isodow" => match (days + 4).rem_euclid(7) {
            0 => 7.0,
            w => w as f64,
        },
        "doy" => (days - days_from_civil(y, 1, 1) + 1) as f64,
        "quarter" => ((m - 1) / 3 + 1) as f64,
        "epoch" => micros as f64 / MICROS_PER_SEC as f64,
        "week" => iso_week(days, y) as f64,
        _ => return None,
    };
    Some(val)
}

/// ISO 8601 week number (1..53) of the day `days` (days since the epoch), where `y` is its Gregorian
/// year. Week 1 is the Monday-based week holding the year's first Thursday, so the first days of
/// January can fall in the last week of the previous year and the last days of December in week 1
/// of the next — matching `EXTRACT(WEEK …)`.
const fn iso_week(days: i64, y: i64) -> i64 {
    // ISO weekday: 1 = Monday .. 7 = Sunday (1970-01-01 was a Thursday → day 0 maps to 4).
    let iso_dow = match (days + 4).rem_euclid(7) {
        0 => 7,
        w => w,
    };
    let doy = days - days_from_civil(y, 1, 1) + 1;
    let week = (doy - iso_dow + 10).div_euclid(7);
    if week < 1 {
        iso_weeks_in_year(y - 1)
    } else if week > iso_weeks_in_year(y) {
        1
    } else {
        week
    }
}

/// The 7-day-cycle parameter used to decide whether an ISO year is long (53 weeks) or short (52):
/// `p(y) = (y + ⌊y/4⌋ − ⌊y/100⌋ + ⌊y/400⌋) mod 7`.
const fn iso_dominical_p(y: i64) -> i64 {
    (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7)
}

/// Number of ISO weeks (52 or 53) in ISO year `y`: 53 iff the year starts on a Thursday or is a leap
/// year starting on a Wednesday.
const fn iso_weeks_in_year(y: i64) -> i64 {
    if iso_dominical_p(y) == 4 || iso_dominical_p(y - 1) == 3 {
        53
    } else {
        52
    }
}

/// `EXTRACT(field FROM interval)` (QA category-D).
///
/// An INTERVAL carries independent `months`, `days`, and `micros` fields; `epoch` is the total
/// seconds using the calendar-agnostic convention of 30-day months and 365.25-day years (whole years
/// split out of the month count first), matching the standard. An inapplicable field returns `None`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "EXTRACT is defined to yield a double-precision number"
)]
#[allow(
    clippy::suboptimal_flops,
    reason = "the epoch sum is written term-by-term for clarity; mul_add would obscure it"
)]
pub fn extract_interval_field(field: &str, months: i64, days: i64, micros: i64) -> Option<f64> {
    const SECS_PER_DAY: i64 = 86_400;
    let val = match field {
        "year" => (months / 12) as f64,
        "month" => (months % 12) as f64,
        "day" => days as f64,
        "hour" => (micros / (3600 * MICROS_PER_SEC)) as f64,
        "minute" => ((micros / (60 * MICROS_PER_SEC)) % 60) as f64,
        "second" => {
            let s = (micros / MICROS_PER_SEC) % 60;
            let us = micros % MICROS_PER_SEC;
            s as f64 + us as f64 / MICROS_PER_SEC as f64
        },
        "epoch" => {
            let years = (months / 12) as f64;
            let rem_months = (months % 12) as f64;
            micros as f64 / MICROS_PER_SEC as f64
                + days as f64 * SECS_PER_DAY as f64
                + rem_months * 30.0 * SECS_PER_DAY as f64
                + years * 365.25 * SECS_PER_DAY as f64
        },
        _ => return None,
    };
    Some(val)
}

/// `EXTRACT(field FROM time)` for a `TIME` value (microseconds since midnight).
///
/// Only intraday fields are meaningful; a calendar field (`year`, `month`, …) returns `None` so the
/// caller can reject it for the `TIME` type.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "EXTRACT is defined to yield a double-precision number"
)]
pub fn extract_time_field(field: &str, tod_micros: i64) -> Option<f64> {
    let h = tod_micros / (3600 * MICROS_PER_SEC);
    let mi = (tod_micros / (60 * MICROS_PER_SEC)) % 60;
    let s = (tod_micros / MICROS_PER_SEC) % 60;
    let us = tod_micros % MICROS_PER_SEC;
    let val = match field {
        "hour" => h as f64,
        "minute" => mi as f64,
        "second" => s as f64 + us as f64 / MICROS_PER_SEC as f64,
        "epoch" => tod_micros as f64 / MICROS_PER_SEC as f64,
        _ => return None,
    };
    Some(val)
}

/// `DATE_TRUNC(field, ts)` — `ts` (epoch micros) floored to the start of the named precision.
/// Returns `None` for an unrecognised field. `week` truncates to the preceding Monday 00:00.
#[must_use]
pub fn date_trunc_micros(field: &str, micros: i64) -> Option<i64> {
    let unit = match field {
        "microsecond" | "microseconds" => return Some(micros),
        "second" => MICROS_PER_SEC,
        "minute" => 60 * MICROS_PER_SEC,
        "hour" => 3600 * MICROS_PER_SEC,
        "day" => MICROS_PER_DAY,
        _ => 0,
    };
    if unit != 0 {
        // `rem_euclid` is non-negative, so this floors toward the previous boundary for any sign.
        return Some(micros - micros.rem_euclid(unit));
    }
    let days = micros.div_euclid(MICROS_PER_DAY);
    let (y, m, _d) = civil_from_days(days);
    let start_day = match field {
        // 1970-01-01 was a Thursday → `(days + 3) mod 7` is 0 on Mondays.
        "week" => days - (days + 3).rem_euclid(7),
        "month" => days_from_civil(y, m, 1),
        "quarter" => days_from_civil(y, (m - 1) / 3 * 3 + 1, 1),
        "year" => days_from_civil(y, 1, 1),
        _ => return None,
    };
    Some(start_day.saturating_mul(MICROS_PER_DAY))
}

/// `AGE(end, start)` — the calendar interval `(months, days, micros)` such that
/// `start + interval == end`.
///
/// Computed field-by-field with borrowing (matching the conventional SQL `age` semantics).
/// Antisymmetric: swapping the arguments negates every component.
#[must_use]
pub fn calendar_age(end: i64, start: i64) -> (i32, i32, i64) {
    if end < start {
        let (months, days, micros) = calendar_age(start, end);
        return (-months, -days, -micros);
    }
    let (y1, mon1, d1, h1, mi1, s1, us1) = decompose_micros(end);
    let (y2, mon2, d2, h2, mi2, s2, us2) = decompose_micros(start);
    let (mut us, mut s, mut mi, mut h) = (us1 - us2, s1 - s2, mi1 - mi2, h1 - h2);
    let (mut d, mut mon, mut y) = (d1 - d2, mon1 - mon2, y1 - y2);
    if us < 0 {
        us += MICROS_PER_SEC;
        s -= 1;
    }
    if s < 0 {
        s += 60;
        mi -= 1;
    }
    if mi < 0 {
        mi += 60;
        h -= 1;
    }
    if h < 0 {
        h += 24;
        d -= 1;
    }
    if d < 0 {
        // Borrow the length of the month preceding the later instant's month.
        let (py, pm) = if mon1 == 1 {
            (y1 - 1, 12)
        } else {
            (y1, mon1 - 1)
        };
        d += days_in_month(py, pm);
        mon -= 1;
    }
    if mon < 0 {
        mon += 12;
        y -= 1;
    }
    let months = y * 12 + mon;
    let micros = (h * 3600 + mi * 60 + s) * MICROS_PER_SEC + us;
    (
        i32::try_from(months).unwrap_or(if months < 0 { i32::MIN } else { i32::MAX }),
        i32::try_from(d).unwrap_or(0),
        micros,
    )
}

// ---- TO_CHAR / TO_DATE / TO_TIMESTAMP format engine -----------------------------------

/// Abbreviated English month names, title case (index 0 = January).
const MONTH_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
/// Full English month names, title case (index 0 = January).
const MONTH_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
/// Abbreviated English weekday names, title case (index 0 = Sunday, matching `dow`).
const WEEKDAY_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
/// Full English weekday names, title case (index 0 = Sunday, matching `dow`).
const WEEKDAY_FULL: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// How a name pattern was capitalised, mirrored onto the rendered name.
#[derive(Clone, Copy)]
enum NameCase {
    /// `MONTH` → `JANUARY`.
    Upper,
    /// `Month` → `January`.
    Title,
    /// `month` → `january`.
    Lower,
}

/// A single field of a `TO_CHAR` / `TO_DATE` / `TO_TIMESTAMP` format string. Unrecognised input is
/// kept verbatim as a [`Literal`](FmtToken::Literal), matching the common convention.
#[derive(Clone)]
enum FmtToken {
    /// `YYYY` — 4-digit year.
    Year4,
    /// `YY` — 2-digit year (parsed into the 2000s).
    Year2,
    /// `MM` — 2-digit month number.
    MonthNum,
    /// `Mon` — abbreviated month name.
    MonthAbbr(NameCase),
    /// `Month` — full month name.
    MonthFull(NameCase),
    /// `DD` — 2-digit day of month.
    Day2,
    /// `DDD` — 3-digit day of year (`001..366`).
    DayOfYear,
    /// `IDDD` — 3-digit ISO day of year (`001..371`, counting from the Monday of ISO week 1).
    IsoDayOfYear,
    /// `D` — day of week, `1` (Sunday) .. `7` (Saturday).
    DayOfWeek,
    /// `ID` — ISO day of week, `1` (Monday) .. `7` (Sunday).
    IsoDayOfWeek,
    /// `Dy` — abbreviated weekday name.
    WeekdayAbbr(NameCase),
    /// `Day` — full weekday name.
    WeekdayFull(NameCase),
    /// `HH24` — hour `00..23`.
    Hour24,
    /// `HH` / `HH12` — hour `01..12` (needs a meridiem to disambiguate on parse).
    Hour12,
    /// `MI` — minute.
    Minute,
    /// `SS` — second.
    Second,
    /// `MS` — millisecond (3 digits).
    Milli,
    /// `US` — microsecond (6 digits).
    Micro,
    /// `AM`/`PM` (the bool is whether to render upper case).
    Meridiem(bool),
    /// `FM` — fill-mode modifier: suppress the leading zeros / trailing blanks of the *next* field.
    /// Emits nothing itself.
    FillMode,
    /// Verbatim text (a quoted `"..."` run, a separator, or any unrecognised character).
    Literal(String),
}

/// Detect the capitalisation style of a matched name pattern from its original characters.
fn detect_case(chars: &[char]) -> NameCase {
    let all_upper = chars.iter().all(char::is_ascii_uppercase);
    let first_upper = chars.first().is_some_and(char::is_ascii_uppercase);
    let rest_lower = chars.iter().skip(1).all(char::is_ascii_lowercase);
    if all_upper {
        NameCase::Upper
    } else if first_upper && rest_lower {
        NameCase::Title
    } else {
        NameCase::Lower
    }
}

/// True if `s` begins with `pat`, comparing ASCII case-insensitively.
fn ci_prefix(s: &[char], pat: &str) -> bool {
    let len = pat.chars().count();
    s.len() >= len
        && s.iter()
            .zip(pat.chars())
            .all(|(c, p)| c.eq_ignore_ascii_case(&p))
}

/// Match the longest known format token at the start of `s`, returning it and its char width.
fn match_pattern(s: &[char]) -> Option<(FmtToken, usize)> {
    use FmtToken as T;
    let name_case = |len: usize| s.get(..len).map_or(NameCase::Title, detect_case);
    // Longest patterns first so `HH24` wins over `HH`, `YYYY` over `YY`, `MONTH` over `MON`/`MM`.
    let candidates: &[(&str, usize)] = &[
        ("HH24", 0),
        ("HH12", 1),
        ("HH", 1),
        ("YYYY", 2),
        ("YY", 3),
        ("MONTH", 4),
        ("MON", 5),
        ("MM", 6),
        ("MI", 7),
        ("MS", 8),
        ("US", 9),
        ("DDD", 13),
        ("DAY", 17),
        ("DD", 10),
        ("DY", 16),
        ("IDDD", 18),
        ("ID", 15),
        ("SS", 11),
        ("AM", 12),
        ("PM", 12),
        ("FM", 19),
        ("D", 14),
    ];
    for (pat, id) in candidates {
        if ci_prefix(s, pat) {
            let len = pat.len();
            let tok = match id {
                0 => T::Hour24,
                1 => T::Hour12,
                2 => T::Year4,
                3 => T::Year2,
                4 => T::MonthFull(name_case(len)),
                5 => T::MonthAbbr(name_case(len)),
                6 => T::MonthNum,
                7 => T::Minute,
                8 => T::Milli,
                9 => T::Micro,
                10 => T::Day2,
                11 => T::Second,
                13 => T::DayOfYear,
                14 => T::DayOfWeek,
                15 => T::IsoDayOfWeek,
                16 => T::WeekdayAbbr(name_case(len)),
                17 => T::WeekdayFull(name_case(len)),
                18 => T::IsoDayOfYear,
                19 => T::FillMode,
                _ => T::Meridiem(s.first().is_some_and(char::is_ascii_uppercase)),
            };
            return Some((tok, len));
        }
    }
    None
}

/// Tokenise a format string into [`FmtToken`]s. A `"..."` run is a literal; any character that is
/// not part of a recognised pattern is kept as a one-character literal.
fn tokenize_format(fmt: &str) -> Vec<FmtToken> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if c == '"' {
            let mut lit = String::new();
            i += 1;
            while let Some(&q) = chars.get(i) {
                i += 1;
                if q == '"' {
                    break;
                }
                lit.push(q);
            }
            tokens.push(FmtToken::Literal(lit));
            continue;
        }
        if let Some((tok, len)) = match_pattern(chars.get(i..).unwrap_or_default()) {
            tokens.push(tok);
            i += len;
            continue;
        }
        tokens.push(FmtToken::Literal(c.to_string()));
        i += 1;
    }
    tokens
}

/// Render a name with the requested capitalisation (the source arrays are title case).
fn cased(name: &str, case: NameCase) -> String {
    match case {
        NameCase::Upper => name.to_uppercase(),
        NameCase::Title => name.to_owned(),
        NameCase::Lower => name.to_lowercase(),
    }
}

/// The month name (1-based `month`) from the given table, cased; empty for an out-of-range month.
fn month_name(month: i64, table: &[&str; 12], case: NameCase) -> String {
    usize::try_from(month - 1)
        .ok()
        .and_then(|i| table.get(i))
        .map_or_else(String::new, |name| cased(name, case))
}

/// Render an integer field zero-padded to `width`, or bare (no leading zeros) when the preceding `FM`
/// fill-mode modifier is active. Values here are non-negative calendar components.
fn num_field(value: i64, width: usize, fill: bool) -> String {
    if fill {
        value.to_string()
    } else {
        format!("{value:0width$}")
    }
}

/// Render a fixed-width name field blank-padded to the longest name's width (9), or bare (no trailing
/// blanks) when the preceding `FM` fill-mode modifier is active.
fn name_field(name: &str, fill: bool) -> String {
    if fill {
        name.to_owned()
    } else {
        format!("{name:<9}")
    }
}

/// The weekday name for `days` (days since the epoch) from the given table, cased. Index 0 = Sunday,
/// derived the same way as `EXTRACT(DOW …)` (1970-01-01 was a Thursday).
fn weekday_name(days: i64, table: &[&str; 7], case: NameCase) -> String {
    let dow = (days + 4).rem_euclid(7);
    usize::try_from(dow)
        .ok()
        .and_then(|i| table.get(i))
        .map_or_else(String::new, |name| cased(name, case))
}

/// `TO_CHAR(ts, fmt)` — render the epoch-micros instant `micros` per the format string `fmt`.
/// Patterns the engine does not recognise are emitted verbatim.
#[must_use]
#[allow(
    clippy::many_single_char_names,
    reason = "conventional y/m/d/h/s calendar component names"
)]
pub fn format_with_pattern(micros: i64, fmt: &str) -> String {
    let (y, m, d, h, mi, s, us) = decompose_micros(micros);
    let days = micros.div_euclid(MICROS_PER_DAY);
    let mut out = String::new();
    // `FM` sets fill mode for the next field only; it emits nothing and does not itself reset the
    // flag. A literal between `FM` and its field passes the flag through; every rendered field
    // consumes and clears it.
    let mut fill = false;
    for tok in tokenize_format(fmt) {
        if matches!(tok, FmtToken::FillMode) {
            fill = true;
            continue;
        }
        let is_literal = matches!(tok, FmtToken::Literal(_));
        match tok {
            FmtToken::Year4 => out.push_str(&num_field(y, 4, fill)),
            FmtToken::Year2 => out.push_str(&num_field(y.rem_euclid(100), 2, fill)),
            FmtToken::MonthNum => out.push_str(&num_field(m, 2, fill)),
            FmtToken::MonthAbbr(case) => out.push_str(&month_name(m, &MONTH_ABBR, case)),
            // The full month name is blank-padded to the longest month's width (9 = "September")
            // unless `FM` suppresses the padding.
            FmtToken::MonthFull(case) => {
                out.push_str(&name_field(&month_name(m, &MONTH_FULL, case), fill));
            },
            FmtToken::Day2 => out.push_str(&num_field(d, 2, fill)),
            FmtToken::DayOfYear => {
                out.push_str(&num_field(days - days_from_civil(y, 1, 1) + 1, 3, fill));
            },
            FmtToken::IsoDayOfYear => {
                // ISO day of year: within an ISO year, week W (ISO) day D (1=Mon..7=Sun) is the
                // `(W-1)*7 + D`-th day. `iso_week` already handles Jan/Dec dates that belong to the
                // adjacent ISO year, so this counts from the Monday of that ISO year's week 1.
                let iso_dow = match (days + 4).rem_euclid(7) {
                    0 => 7,
                    w => w,
                };
                out.push_str(&num_field((iso_week(days, y) - 1) * 7 + iso_dow, 3, fill));
            },
            // 1 = Sunday .. 7 = Saturday (already single-digit, so `FM` has no effect).
            FmtToken::DayOfWeek => out.push_str(&num_field((days + 4).rem_euclid(7) + 1, 1, fill)),
            FmtToken::IsoDayOfWeek => {
                // 1 = Monday .. 7 = Sunday.
                let iso = match (days + 4).rem_euclid(7) {
                    0 => 7,
                    w => w,
                };
                out.push_str(&num_field(iso, 1, fill));
            },
            FmtToken::WeekdayAbbr(case) => out.push_str(&weekday_name(days, &WEEKDAY_ABBR, case)),
            // The full weekday name is blank-padded to the longest name's width (9 = "Wednesday")
            // unless `FM` suppresses the padding.
            FmtToken::WeekdayFull(case) => {
                out.push_str(&name_field(&weekday_name(days, &WEEKDAY_FULL, case), fill));
            },
            FmtToken::Hour24 => out.push_str(&num_field(h, 2, fill)),
            FmtToken::Hour12 => out.push_str(&num_field((h + 11) % 12 + 1, 2, fill)),
            FmtToken::Minute => out.push_str(&num_field(mi, 2, fill)),
            FmtToken::Second => out.push_str(&num_field(s, 2, fill)),
            FmtToken::Milli => out.push_str(&num_field(us / 1000, 3, fill)),
            FmtToken::Micro => out.push_str(&num_field(us, 6, fill)),
            FmtToken::Meridiem(upper) => {
                out.push_str(match (h < 12, upper) {
                    (true, true) => "AM",
                    (true, false) => "am",
                    (false, true) => "PM",
                    (false, false) => "pm",
                });
            },
            FmtToken::Literal(lit) => out.push_str(&lit),
            // `FillMode` is handled before the match.
            FmtToken::FillMode => {},
        }
        // A field consumes the fill flag; a literal leaves it pending for the next field.
        if !is_literal {
            fill = false;
        }
    }
    out
}

/// Consume up to `max_digits` ASCII digits (at least one) as a non-negative integer.
fn take_int(input: &[char], pos: &mut usize, max_digits: usize) -> Option<i64> {
    let mut value: i64 = 0;
    let mut count = 0;
    while count < max_digits {
        match input.get(*pos).and_then(|c| c.to_digit(10)) {
            Some(digit) => {
                value = value * 10 + i64::from(digit);
                *pos += 1;
                count += 1;
            },
            None => break,
        }
    }
    (count > 0).then_some(value)
}

/// Match a month name (full preferred over abbreviated) case-insensitively, returning its 1-based
/// number and advancing `pos` past it.
fn take_month_name(input: &[char], pos: &mut usize) -> Option<i64> {
    let rest = input.get(*pos..).unwrap_or_default();
    for (i, name) in MONTH_FULL.iter().chain(MONTH_ABBR.iter()).enumerate() {
        if ci_prefix(rest, name) {
            *pos += name.chars().count();
            return i64::try_from(i % 12 + 1).ok();
        }
    }
    None
}

/// Match a weekday name (full preferred over abbreviated) case-insensitively, advancing `pos` past
/// it. The weekday does not constrain the calendar date, so the matched name is discarded — the
/// caller only needs the cursor advanced (mirrors how the field is ignored when parsing a date).
fn take_weekday_name(input: &[char], pos: &mut usize) -> Option<()> {
    let rest = input.get(*pos..).unwrap_or_default();
    for name in WEEKDAY_FULL.iter().chain(WEEKDAY_ABBR.iter()) {
        if ci_prefix(rest, name) {
            *pos += name.chars().count();
            return Some(());
        }
    }
    None
}

/// Match `AM`/`PM` (case-insensitive), returning whether it is PM and advancing `pos` by two.
fn take_meridiem(input: &[char], pos: &mut usize) -> Option<bool> {
    let rest = input.get(*pos..).unwrap_or_default();
    let is_pm = if ci_prefix(rest, "PM") {
        true
    } else if ci_prefix(rest, "AM") {
        false
    } else {
        return None;
    };
    *pos += 2;
    Some(is_pm)
}

/// Consume a literal token: whitespace in the pattern matches a run of input whitespace; any other
/// character must match the input case-insensitively.
fn consume_literal(input: &[char], pos: &mut usize, lit: &str) -> Option<()> {
    for pc in lit.chars() {
        if pc.is_whitespace() {
            while input.get(*pos).is_some_and(|c| c.is_whitespace()) {
                *pos += 1;
            }
        } else if input.get(*pos).is_some_and(|c| c.eq_ignore_ascii_case(&pc)) {
            *pos += 1;
        } else {
            return None;
        }
    }
    Some(())
}

/// Parse `input` per the format string `fmt` into epoch microseconds.
///
/// Fields absent from the format default to `1970-01-01 00:00:00`. Returns `None` if the input does
/// not match the format or yields an invalid date/time.
#[must_use]
pub fn parse_with_pattern(input: &str, fmt: &str) -> Option<i64> {
    let inp: Vec<char> = input.chars().collect();
    let mut pos = 0;
    let (mut year, mut month, mut day) = (1970_i64, 1_i64, 1_i64);
    let (mut hour, mut minute, mut second, mut micro) = (0_i64, 0_i64, 0_i64, 0_i64);
    let mut hour12 = false;
    let mut meridiem_pm: Option<bool> = None;
    let mut day_of_year: Option<i64> = None;
    for tok in tokenize_format(fmt) {
        match tok {
            FmtToken::Year4 => year = take_int(&inp, &mut pos, 4)?,
            FmtToken::Year2 => year = 2000 + take_int(&inp, &mut pos, 2)?,
            FmtToken::MonthNum => month = take_int(&inp, &mut pos, 2)?,
            FmtToken::MonthAbbr(_) | FmtToken::MonthFull(_) => {
                month = take_month_name(&inp, &mut pos)?;
            },
            FmtToken::Day2 => day = take_int(&inp, &mut pos, 2)?,
            FmtToken::DayOfYear => day_of_year = Some(take_int(&inp, &mut pos, 3)?),
            // Reconstructing a date from the ISO day of year needs the ISO year (`IYYY`), which the
            // format engine does not model, so `IDDD` is not accepted on the parse side (loud-fail
            // rather than silently ignore a field that does constrain the date).
            FmtToken::IsoDayOfYear => return None,
            // A weekday does not constrain the calendar date, so these are consumed and ignored.
            FmtToken::DayOfWeek | FmtToken::IsoDayOfWeek => {
                take_int(&inp, &mut pos, 1)?;
            },
            FmtToken::WeekdayAbbr(_) | FmtToken::WeekdayFull(_) => {
                take_weekday_name(&inp, &mut pos)?;
            },
            FmtToken::Hour24 => hour = take_int(&inp, &mut pos, 2)?,
            FmtToken::Hour12 => {
                hour = take_int(&inp, &mut pos, 2)?;
                hour12 = true;
            },
            FmtToken::Minute => minute = take_int(&inp, &mut pos, 2)?,
            FmtToken::Second => second = take_int(&inp, &mut pos, 2)?,
            FmtToken::Milli => micro = take_int(&inp, &mut pos, 3)? * 1000,
            FmtToken::Micro => micro = take_int(&inp, &mut pos, 6)?,
            FmtToken::Meridiem(_) => meridiem_pm = Some(take_meridiem(&inp, &mut pos)?),
            // `FM` is a modifier: on the parse side it relaxes fixed-width matching, and `take_int`
            // already accepts a variable number of digits, so it consumes no input here.
            FmtToken::FillMode => {},
            FmtToken::Literal(lit) => consume_literal(&inp, &mut pos, &lit)?,
        }
    }
    if let Some(doy) = day_of_year {
        // Day-of-year sets the calendar date relative to Jan 1 of the parsed year, overriding any
        // month/day fields. Reject a value outside the year's length rather than rolling silently.
        let year_len = days_from_civil(year + 1, 1, 1) - days_from_civil(year, 1, 1);
        if doy < 1 || doy > year_len {
            return None;
        }
        let (_, m, d) = civil_from_days(days_from_civil(year, 1, 1) + doy - 1);
        month = m;
        day = d;
    }
    if hour12 {
        match meridiem_pm {
            Some(true) if hour < 12 => hour += 12,
            Some(false) if hour == 12 => hour = 0,
            _ => {},
        }
    }
    if !is_valid_ymd(year, month, day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let tod = (hour * 3600 + minute * 60 + second) * MICROS_PER_SEC + micro;
    days_from_civil(year, month, day)
        .checked_mul(MICROS_PER_DAY)?
        .checked_add(tod)
}

// ---- UUID ------------------------------------------------------------------------------------

/// Parse a UUID — 32 hex digits, with or without the canonical hyphens.
#[must_use]
pub fn parse_uuid(s: &str) -> Option<[u8; 16]> {
    let hex: String = s.trim().chars().filter(|&c| c != '-').collect();
    if hex.len() != 32 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Format 16 bytes as the canonical hyphenated lowercase UUID.
#[must_use]
pub fn format_uuid(bytes: &[u8; 16]) -> String {
    let mut h = String::with_capacity(32);
    for b in bytes {
        // Writing to a `String` is infallible.
        let _ = write!(h, "{b:02x}");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "unit-test assertions unwrap on known-good inputs; UUID slices are fixed-width"
)]
mod tests {
    use super::*;

    #[test]
    fn date_round_trips() {
        for s in ["1970-01-01", "2024-02-29", "1999-12-31", "2000-01-01"] {
            assert_eq!(format_date(parse_date(s).unwrap()), s);
        }
        assert_eq!(parse_date("1970-01-01"), Some(0));
        assert_eq!(parse_date("1970-01-02"), Some(1));
    }

    #[test]
    fn date_rejects_invalid() {
        assert!(parse_date("2023-02-29").is_none()); // not a leap year
        assert!(parse_date("2024-13-01").is_none());
        assert!(parse_date("2024-00-01").is_none());
        assert!(parse_date("2024-1-1").is_none()); // not zero-padded
        assert!(parse_date("notadate").is_none());
    }

    #[test]
    fn time_round_trips() {
        for s in ["00:00:00", "23:59:59", "12:30:45.123456", "01:02:03.000500"] {
            assert_eq!(format_time(parse_time(s).unwrap()), s);
        }
        assert_eq!(parse_time("00:00:01"), Some(1_000_000));
        assert!(parse_time("24:00:00").is_none());
        assert!(parse_time("12:60:00").is_none());
    }

    #[test]
    fn timestamp_round_trips() {
        assert_eq!(parse_timestamp("1970-01-01 00:00:00"), Some(0));
        assert_eq!(
            format_timestamp(parse_timestamp("2024-06-15 13:45:30").unwrap()),
            "2024-06-15 13:45:30"
        );
        // 'T' separator accepted, formatted with a space.
        assert_eq!(
            format_timestamp(parse_timestamp("2024-06-15T13:45:30.500000").unwrap()),
            "2024-06-15 13:45:30.500000"
        );
    }

    #[test]
    fn time_without_seconds_defaults_to_zero() {
        // The reference engine accepts `HH:MM` for TIME / TIMESTAMP / TIMESTAMPTZ / TIMETZ, defaulting seconds to 0.
        // The seconds-bearing forms are unaffected.
        assert_eq!(parse_time("12:30"), parse_time("12:30:00"));
        assert_eq!(format_time(parse_time("12:30").unwrap()), "12:30:00");
        assert_eq!(
            parse_timestamp("2024-01-31 12:00"),
            parse_timestamp("2024-01-31 12:00:00")
        );
        assert_eq!(
            format_timestamp(parse_timestamp("2024-01-31 12:00").unwrap()),
            "2024-01-31 12:00:00"
        );
        // 'T' separator, timestamptz, and timetz also accept the seconds-less form.
        assert_eq!(
            parse_timestamp("2024-01-31T12:00"),
            parse_timestamp("2024-01-31 12:00:00")
        );
        assert_eq!(
            parse_timestamptz("2024-01-31 12:00+02:00"),
            parse_timestamptz("2024-01-31 12:00:00+02:00")
        );
        assert_eq!(parse_timetz("12:00Z"), parse_timetz("12:00:00Z"));
        // A bare hour (no minutes) is still rejected — at least `HH:MM` is required.
        assert!(parse_time("12").is_none());
        // Out-of-range minutes are still rejected in the seconds-less form.
        assert!(parse_time("12:60").is_none());
    }

    #[test]
    fn timetz_keeps_its_zone_and_orders_like_pg() {
        // P-TIMETZ: the entered offset is kept and rendered back (faithful to the reference engine) — whole hours
        // short (`+07`), minutes long (`+05:30`), UTC/`Z`/missing as `+00`.
        let fmt = |s: &str| format_timetz(parse_timetz(s).unwrap());
        assert_eq!(fmt("13:45:30+07"), "13:45:30+07");
        assert_eq!(fmt("09:15:00+05:30"), "09:15:00+05:30");
        assert_eq!(fmt("23:30:00-02:00"), "23:30:00-02");
        assert_eq!(fmt("06:45:30Z"), "06:45:30+00");
        assert_eq!(fmt("06:45:30"), "06:45:30+00");
        assert_eq!(fmt("12:00:00.250000+03"), "12:00:00.250000+03");

        // The packed accessors round-trip local time and offset.
        let packed = parse_timetz("13:45:30+07").unwrap();
        assert_eq!(timetz_local_micros(packed), parse_time("13:45:30").unwrap());
        assert_eq!(timetz_offset_east_secs(packed), 7 * 3600);

        // Plain i64 ordering of the packed form implements the reference engine's timetz_cmp: primary by the
        // UTC-equivalent instant, deliberately NOT wrapped into one day...
        let p = |s: &str| parse_timetz(s).unwrap();
        assert!(p("05:00:00+00") < p("13:45:30+07")); // 05:00 UTC < 06:45:30 UTC
        assert!(p("23:30:00-02") > p("12:00:00+00")); // 25:30 UTC-equivalent, not wrapped to 01:30
        // ...with the zone as tie-break, so the same instant at a different zone is NOT equal
        // (the reference engine equality quirk) and the more-westerly zone compares greater.
        assert_ne!(p("13:45:30+07"), p("06:45:30+00"));
        assert!(p("13:45:30+07") < p("06:45:30+00"));
    }

    #[test]
    fn date_only_timestamp_is_midnight() {
        // A timestamp literal with no time part is midnight, matching the reference engine (`TIMESTAMP '2024-03-15'`).
        assert_eq!(
            parse_timestamp("2024-03-15"),
            parse_timestamp("2024-03-15 00:00:00")
        );
        assert_eq!(
            format_timestamp(parse_timestamp("2024-03-15").unwrap()),
            "2024-03-15 00:00:00"
        );
        // timestamptz date-only is midnight UTC.
        assert_eq!(
            parse_timestamptz("2024-03-15"),
            parse_timestamptz("2024-03-15 00:00:00")
        );
        // A malformed date is still rejected.
        assert!(parse_timestamp("2024-13-99").is_none());
    }

    #[test]
    fn timestamptz_normalizes_offset_to_utc() {
        // 10:00:00+02:00 == 08:00:00 UTC
        let a = parse_timestamptz("2024-06-15 10:00:00+02:00").unwrap();
        let b = parse_timestamptz("2024-06-15 08:00:00Z").unwrap();
        assert_eq!(a, b);
        assert_eq!(format_timestamptz(b), "2024-06-15 08:00:00+00");
        // Bare (no offset) is treated as UTC.
        assert_eq!(parse_timestamptz("2024-06-15 08:00:00"), Some(b));
    }

    #[test]
    fn uuid_round_trips_hyphenated_and_bare() {
        let canonical = "550e8400-e29b-41d4-a716-446655440000";
        let bytes = parse_uuid(canonical).unwrap();
        assert_eq!(format_uuid(&bytes), canonical);
        // Bare (no hyphens) parses to the same bytes.
        assert_eq!(parse_uuid("550e8400e29b41d4a716446655440000"), Some(bytes));
        // Uppercase accepted, formatted lowercase.
        assert_eq!(
            format_uuid(&parse_uuid(&canonical.to_uppercase()).unwrap()),
            canonical
        );
        assert!(parse_uuid("xyz").is_none());
        assert!(parse_uuid("550e8400-e29b-41d4-a716-44665544000").is_none()); // 31 digits
    }

    #[test]
    fn pre_epoch_timestamp_formats() {
        let micros = parse_timestamp("1969-12-31 23:59:59").unwrap();
        assert!(micros < 0);
        assert_eq!(format_timestamp(micros), "1969-12-31 23:59:59");
    }

    // ----: extract / date_trunc / age ----

    #[test]
    fn extract_reads_timestamp_fields() {
        let ts = parse_timestamp("2024-06-15 13:45:30.500000").unwrap();
        assert_eq!(extract_from_micros("year", ts), Some(2024.0));
        assert_eq!(extract_from_micros("month", ts), Some(6.0));
        assert_eq!(extract_from_micros("day", ts), Some(15.0));
        assert_eq!(extract_from_micros("hour", ts), Some(13.0));
        assert_eq!(extract_from_micros("minute", ts), Some(45.0));
        assert_eq!(extract_from_micros("second", ts), Some(30.5));
        assert_eq!(extract_from_micros("quarter", ts), Some(2.0));
        // 2024-06-15 is a Saturday → dow 6 (Sun=0), isodow 6 (Mon=1).
        assert_eq!(extract_from_micros("dow", ts), Some(6.0));
        assert_eq!(extract_from_micros("isodow", ts), Some(6.0));
        // 2024 is a leap year: Jan31+Feb29+Mar31+Apr30+May31 = 152, +15 = 167.
        assert_eq!(extract_from_micros("doy", ts), Some(167.0));
        // ISO week 24 of 2024 (the Monday-based week holding 2024-06-15).
        assert_eq!(extract_from_micros("week", ts), Some(24.0));
        assert_eq!(extract_from_micros("nonsense", ts), None);
    }

    #[test]
    fn iso_week_handles_year_boundaries() {
        let week = |s: &str| extract_from_micros("week", parse_timestamp(s).unwrap());
        // 2021-01-01 (Friday) belongs to ISO week 53 of 2020 (a long ISO year).
        assert_eq!(week("2021-01-01 00:00:00"), Some(53.0));
        // 2023-01-01 (Sunday) belongs to ISO week 52 of 2022 (a short ISO year).
        assert_eq!(week("2023-01-01 00:00:00"), Some(52.0));
        // 2024-12-30 (Monday) is already week 1 of ISO-year 2025.
        assert_eq!(week("2024-12-30 00:00:00"), Some(1.0));
    }

    #[test]
    fn extract_reads_interval_fields() {
        // INTERVAL '1 year 2 mons 10 days 03:04:05.5' = 14 months, 10 days, time in micros.
        let micros = ((3 * 3600 + 4 * 60 + 5) * MICROS_PER_SEC) + 500_000;
        assert_eq!(extract_interval_field("year", 14, 10, micros), Some(1.0));
        assert_eq!(extract_interval_field("month", 14, 10, micros), Some(2.0));
        assert_eq!(extract_interval_field("day", 14, 10, micros), Some(10.0));
        assert_eq!(extract_interval_field("hour", 14, 10, micros), Some(3.0));
        assert_eq!(extract_interval_field("minute", 14, 10, micros), Some(4.0));
        assert_eq!(extract_interval_field("second", 14, 10, micros), Some(5.5));
        // epoch = 1 year (365.25 d = 31_557_600 s) + 2 mons (30 d each = 5_184_000 s)
        //       + 10 days (864_000 s) + 03:04:05.5 (11_045.5 s) = 37_616_645.5 s.
        assert_eq!(
            extract_interval_field("epoch", 14, 10, micros),
            Some(37_616_645.5)
        );
        assert_eq!(extract_interval_field("nonsense", 14, 10, micros), None);
    }

    #[test]
    fn extract_time_only_accepts_intraday_fields() {
        let tod = parse_time("13:45:30").unwrap();
        assert_eq!(extract_time_field("hour", tod), Some(13.0));
        assert_eq!(extract_time_field("minute", tod), Some(45.0));
        assert_eq!(extract_time_field("second", tod), Some(30.0));
        // A calendar field is meaningless for TIME.
        assert_eq!(extract_time_field("year", tod), None);
    }

    #[test]
    fn date_trunc_floors_to_precision() {
        let ts = parse_timestamp("2024-06-15 13:45:30.500000").unwrap();
        let truncs = |unit| format_timestamp(date_trunc_micros(unit, ts).unwrap());
        assert_eq!(truncs("hour"), "2024-06-15 13:00:00");
        assert_eq!(truncs("day"), "2024-06-15 00:00:00");
        assert_eq!(truncs("month"), "2024-06-01 00:00:00");
        assert_eq!(truncs("quarter"), "2024-04-01 00:00:00");
        assert_eq!(truncs("year"), "2024-01-01 00:00:00");
        // 2024-06-15 is a Saturday → the week truncates back to Monday 2024-06-10.
        assert_eq!(truncs("week"), "2024-06-10 00:00:00");
        assert_eq!(date_trunc_micros("nonsense", ts), None);
    }

    #[test]
    fn calendar_age_borrows_across_month_and_is_antisymmetric() {
        let later = parse_timestamp("2024-03-01 00:00:00").unwrap();
        let earlier = parse_timestamp("2024-01-15 00:00:00").unwrap();
        // 2024-01-15 + 1 month + 15 days = 2024-03-01 (Feb 2024 has 29 days).
        assert_eq!(calendar_age(later, earlier), (1, 15, 0));
        // Swapping negates every component.
        assert_eq!(calendar_age(earlier, later), (-1, -15, 0));
        // Sub-day difference lands entirely in the micros component.
        let a = parse_timestamp("2024-01-01 12:00:00").unwrap();
        let b = parse_timestamp("2024-01-01 10:30:00").unwrap();
        assert_eq!(calendar_age(a, b), (0, 0, 90 * 60 * MICROS_PER_SEC));
    }

    // ----: TO_CHAR / TO_DATE / TO_TIMESTAMP format engine ----

    #[test]
    fn format_with_pattern_renders_fields() {
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        assert_eq!(
            format_with_pattern(ts, "YYYY-MM-DD HH24:MI:SS"),
            "2024-06-15 13:45:30"
        );
        // Month names in three cases, and the 2-digit year.
        assert_eq!(format_with_pattern(ts, "DD Mon YYYY"), "15 Jun 2024");
        // The full month name is blank-padded to 9 chars (the width of "September"), like the reference engine.
        assert_eq!(format_with_pattern(ts, "Month"), "June     ");
        assert_eq!(format_with_pattern(ts, "MONTH"), "JUNE     ");
        assert_eq!(format_with_pattern(ts, "month"), "june     ");
        assert_eq!(format_with_pattern(ts, "YY"), "24");
        // 12-hour clock with meridiem.
        assert_eq!(format_with_pattern(ts, "HH12:MI PM"), "01:45 PM");
        let midnight = parse_timestamp("2024-06-15 00:30:00").unwrap();
        assert_eq!(format_with_pattern(midnight, "HH12:MI AM"), "12:30 AM");
        // A quoted run and unknown characters are emitted verbatim.
        assert_eq!(format_with_pattern(ts, "\"yr\" YYYY"), "yr 2024");
    }

    #[test]
    fn format_with_pattern_renders_weekday_and_day_of_year() {
        // 2024-06-15 is a Saturday, day-of-year 167 (a leap year).
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        // Full weekday name is blank-padded to 9 chars (width of "Wednesday"), like the reference engine.
        assert_eq!(format_with_pattern(ts, "Day"), "Saturday ");
        assert_eq!(format_with_pattern(ts, "DAY"), "SATURDAY ");
        assert_eq!(format_with_pattern(ts, "day"), "saturday ");
        // Abbreviated weekday name is fixed-width.
        assert_eq!(format_with_pattern(ts, "Dy"), "Sat");
        assert_eq!(format_with_pattern(ts, "DY"), "SAT");
        // `D` = day of week (1 = Sunday .. 7 = Saturday); `ID` = ISO (1 = Monday .. 7 = Sunday).
        assert_eq!(format_with_pattern(ts, "D"), "7");
        assert_eq!(format_with_pattern(ts, "ID"), "6");
        // `DDD` = zero-padded day of year; `DD` (day of month) still wins over the single `D`.
        assert_eq!(format_with_pattern(ts, "DDD"), "167");
        assert_eq!(format_with_pattern(ts, "DD"), "15");
        // `IDDD` = ISO day of year — matches `ID`+`DD` only by coincidence, so it must not tokenize
        // greedily as `ID` (6) followed by `DD` (15). 2024's ISO year aligns with the Gregorian one,
        // so IDDD == DDD == 167 here.
        assert_eq!(format_with_pattern(ts, "IDDD"), "167");
        // A Sunday maps `D` -> 1 and `ID` -> 7.
        let sunday = parse_timestamp("2024-06-16 00:00:00").unwrap();
        assert_eq!(format_with_pattern(sunday, "Day, D, ID"), "Sunday   , 1, 7");
        // Where the ISO year differs from the Gregorian one, IDDD diverges from DDD: 2023-01-01 is a
        // Sunday belonging to ISO week 52 of 2022, so IDDD = (52-1)*7 + 7 = 364 while DDD = 001.
        let ny = parse_timestamp("2023-01-01 00:00:00").unwrap();
        assert_eq!(format_with_pattern(ny, "DDD"), "001");
        assert_eq!(format_with_pattern(ny, "IDDD"), "364");
    }

    #[test]
    fn format_with_pattern_fill_mode_suppresses_padding() {
        // 2024-06-05 09:03:07 is a Wednesday with single-digit month/day/hour, so `FM` visibly strips
        // the leading zeros and trailing name padding (matching the reference engine); without `FM` they stay padded.
        let ts = parse_timestamp("2024-06-05 09:03:07").unwrap();
        // Name fields: no trailing blanks, and no literal "FM" echoed.
        assert_eq!(format_with_pattern(ts, "FMMonth"), "June");
        assert_eq!(format_with_pattern(ts, "Month"), "June     ");
        assert_eq!(format_with_pattern(ts, "FMDay"), "Wednesday");
        // Numeric fields: no leading zeros.
        assert_eq!(format_with_pattern(ts, "FMMM"), "6");
        assert_eq!(format_with_pattern(ts, "MM"), "06");
        assert_eq!(format_with_pattern(ts, "FMDD"), "5");
        assert_eq!(format_with_pattern(ts, "FMHH24"), "9");
        // `FM` modifies only the immediately following field: HH24 is stripped, MI stays padded.
        assert_eq!(format_with_pattern(ts, "FMHH24:MI"), "9:03");
        // A full date in fill mode, the reference engine's `FMDD FMMonth YYYY` shape.
        assert_eq!(format_with_pattern(ts, "FMDD FMMonth YYYY"), "5 June 2024");
    }

    #[test]
    fn parse_with_pattern_reads_fields_and_round_trips() {
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        assert_eq!(
            parse_with_pattern("2024-06-15 13:45:30", "YYYY-MM-DD HH24:MI:SS"),
            Some(ts)
        );
        // Month name + 12-hour clock with PM (no seconds in this format → 13:45:00).
        assert_eq!(
            parse_with_pattern("15 Jun 2024 01:45 PM", "DD Mon YYYY HH12:MI PM"),
            parse_timestamp("2024-06-15 13:45:00")
        );
        // 12 AM is midnight.
        assert_eq!(
            parse_with_pattern("2024-06-15 12:30 AM", "YYYY-MM-DD HH12:MI AM"),
            parse_timestamp("2024-06-15 00:30:00")
        );
        // Fields absent from the format default to 1970-01-01 00:00:00.
        assert_eq!(
            parse_with_pattern("2024", "YYYY"),
            parse_timestamp("2024-01-01 00:00:00")
        );
        // A non-matching input is rejected (the caller turns this into an error).
        assert!(parse_with_pattern("notadate", "YYYY-MM-DD").is_none());
        assert!(parse_with_pattern("2024-13-01", "YYYY-MM-DD").is_none());
    }

    #[test]
    fn parse_with_pattern_handles_day_of_year_and_ignores_weekday() {
        // Day-of-year sets the calendar date (167 -> 2024-06-15 in a leap year).
        assert_eq!(
            parse_with_pattern("2024-167", "YYYY-DDD"),
            parse_timestamp("2024-06-15 00:00:00")
        );
        // A weekday name and numeric weekday are consumed but do not constrain the date.
        assert_eq!(
            parse_with_pattern("Saturday 2024-167", "Day YYYY-DDD"),
            parse_timestamp("2024-06-15 00:00:00")
        );
        // Day-of-year out of range for the year is rejected rather than rolling into the next year.
        assert!(parse_with_pattern("2024-367", "YYYY-DDD").is_none());
        assert!(parse_with_pattern("2023-366", "YYYY-DDD").is_none());
    }
}
