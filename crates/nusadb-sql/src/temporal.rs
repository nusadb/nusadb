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

/// Whether `s` is a well-shaped `YYYY-MM-DD` that still won't parse — a field out of range.
///
/// A three-field date with a two-digit month and day that fails to parse (`2023-02-30`,
/// `99-99-99`) is a `datetime_field_overflow` (`22008`), whereas a mis-shaped literal (`abc`,
/// `2023-02`) is `invalid_datetime_format` (`22007`); this tells the two apart for the caller.
#[must_use]
pub fn is_date_field_out_of_range(s: &str) -> bool {
    let s = s.trim();
    let mut it = s.splitn(3, '-');
    let (Some(y), Some(m), Some(d)) = (it.next(), it.next(), it.next()) else {
        return false;
    };
    let well_shaped = !y.is_empty()
        && y.bytes().all(|b| b.is_ascii_digit())
        && parse_fixed(m, 2).is_some()
        && parse_fixed(d, 2).is_some();
    // A well-shaped date that still won't parse is out-of-range, not a format error.
    well_shaped && parse_date(s).is_none()
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

/// The trailing fractional-seconds suffix for `frac` sub-second microseconds: an empty string when
/// zero, otherwise a dot and the six-digit fraction with trailing zeros trimmed (`.5`, `.12`,
/// `.123456`) — matching the reference engine, which does not pad the fraction to a fixed width.
#[must_use]
pub(crate) fn subsecond_suffix(frac: u32) -> String {
    if frac == 0 {
        return String::new();
    }
    let digits = format!("{frac:06}");
    format!(".{}", digits.trim_end_matches('0'))
}

/// Format microseconds-since-midnight as `HH:MM:SS` (or `HH:MM:SS.f…` with a trimmed fraction).
#[must_use]
pub fn format_time(micros: i64) -> String {
    let secs = micros / MICROS_PER_SEC;
    let frac = micros % MICROS_PER_SEC;
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    // `frac` is `micros % 1_000_000`, always in `[0, 1_000_000)`, so the cast is lossless.
    format!(
        "{h:02}:{m:02}:{s:02}{}",
        subsecond_suffix(u32::try_from(frac).unwrap_or(0))
    )
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

/// Format a UTC instant as local wall time in a zone `offset_east_secs` east of UTC.
///
/// Renders `YYYY-MM-DD HH:MM:SS[.ffffff]±HH[:MM[:SS]]`, with minutes/seconds in the offset suffix
/// appended only when nonzero (the reference engine's rendering: `+07`, `-05:30`).
#[must_use]
pub fn format_timestamptz_at(micros: i64, offset_east_secs: i64) -> String {
    let local = micros.saturating_add(offset_east_secs.saturating_mul(MICROS_PER_SEC));
    let time = format_timestamp(local);
    let sign = if offset_east_secs < 0 { '-' } else { '+' };
    let abs = offset_east_secs.abs();
    let (h, m, s) = (abs / 3600, (abs % 3600) / 60, abs % 60);
    if s != 0 {
        format!("{time}{sign}{h:02}:{m:02}:{s:02}")
    } else if m != 0 {
        format!("{time}{sign}{h:02}:{m:02}")
    } else {
        format!("{time}{sign}{h:02}")
    }
}

/// Parse a timestamptz like [`parse_timestamptz`], but with the session zone as the default.
///
/// A **missing** offset is read as local wall time in a zone `offset_east_secs` east of UTC (the
/// session time zone) instead of UTC. An explicit offset (or `Z`) in the string still wins.
#[must_use]
pub fn parse_timestamptz_at(s: &str, offset_east_secs: i64) -> Option<i64> {
    let s = s.trim();
    if has_explicit_offset(s) {
        parse_timestamp_inner(s, true)
    } else {
        parse_timestamp_inner(s, true)?.checked_sub(offset_east_secs.checked_mul(MICROS_PER_SEC)?)
    }
}

/// Whether a timestamp string carries its own zone — a trailing `Z` or a `±HH[[:]MM]` offset after
/// the time part. Decides if the session time zone applies to the value.
#[must_use]
pub fn has_explicit_offset(s: &str) -> bool {
    let s = s.trim();
    if s.ends_with('Z') {
        return true;
    }
    let time_part = match s.find(['T', ' ']) {
        Some(sep) => &s[sep + 1..],
        None => return false,
    };
    find_offset_sign(time_part).is_some()
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

// ---- Session time zone setting ---------------------------------------------------------------

/// Why a `SET TIME ZONE` / `SET timezone` value was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionZoneError {
    /// The value names no time zone this engine can parse (bad syntax or offset out of range).
    Invalid,
    /// The value is an IANA-style zone name (`Region/City`), which needs a time-zone database this
    /// engine does not carry — only `UTC`/`GMT` and fixed offsets are supported.
    NeedsTzDatabase,
}

/// The largest fixed session-zone displacement accepted, in hours — the reference engine's bound.
const MAX_ZONE_DISP_HOURS: i64 = 15;

/// Validate a session time-zone value and canonicalize it to the form `SHOW timezone` reports.
///
/// Returns `(canonical, offset_east_secs)`. Accepted forms, matching the reference engine's
/// reading of each:
///
/// - `UTC` / `GMT` / `Etc/UTC` / `Etc/GMT` (case-insensitive) — offset `0`.
/// - A bare number (hours, from `SET TIME ZONE 7` / `SET TIME ZONE 5.5`) — **east** of UTC,
///   canonicalized to the POSIX display `<+05:30>-05:30`.
/// - `±HH` (no colon) — a zone-abbreviation spelling, **east** of UTC (`+07` is UTC+7),
///   canonicalized to the POSIX display `<+07>-07`.
/// - `±HH:MM[:SS]` (with colon) — a POSIX offset spec, whose sign is the **opposite** of ISO
///   (`+05:30` is 5½ hours *west*); kept verbatim as its own canonical form.
/// - An IANA-style name (`Asia/Jakarta`) is refused with [`SessionZoneError::NeedsTzDatabase`];
///   anything else with [`SessionZoneError::Invalid`].
///
/// # Errors
/// See the variants of [`SessionZoneError`].
pub fn parse_session_timezone(value: &str) -> Result<(String, i64), SessionZoneError> {
    let v = value.trim();
    if v.eq_ignore_ascii_case("utc") {
        return Ok(("UTC".to_owned(), 0));
    }
    if v.eq_ignore_ascii_case("gmt") {
        return Ok(("GMT".to_owned(), 0));
    }
    if v.eq_ignore_ascii_case("etc/utc") {
        return Ok(("Etc/UTC".to_owned(), 0));
    }
    if v.eq_ignore_ascii_case("etc/gmt") {
        return Ok(("Etc/GMT".to_owned(), 0));
    }
    // A bare number of hours east of UTC (`SET TIME ZONE 7`, `SET TIME ZONE 5.5`).
    if let Ok(hours) = v.parse::<f64>() {
        if !hours.is_finite() || hours.abs() > 24.0 {
            return Err(SessionZoneError::Invalid);
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "bounded to ±24 hours just above, so minutes fit far inside i64"
        )]
        let minutes = (hours * 60.0).round() as i64;
        if minutes.abs() > MAX_ZONE_DISP_HOURS * 60 + 59 {
            return Err(SessionZoneError::Invalid);
        }
        let offset = minutes * 60;
        return Ok((posix_zone_display(offset), offset));
    }
    // `±HH` with no colon is a zone-abbreviation spelling: ISO sign, east of UTC.
    if let Some((sign, digits)) = split_sign(v)
        && !digits.is_empty()
        && digits.len() <= 2
        && digits.bytes().all(|b| b.is_ascii_digit())
    {
        let hours: i64 = digits.parse().map_err(|_| SessionZoneError::Invalid)?;
        if hours > MAX_ZONE_DISP_HOURS {
            return Err(SessionZoneError::Invalid);
        }
        let offset = sign * hours * 3600;
        return Ok((posix_zone_display(offset), offset));
    }
    // `±HH:MM[:SS]` is a POSIX offset spec: its sign is the opposite of ISO (`+` is west of UTC).
    if let Some((sign, rest)) = split_sign(v)
        && rest.contains(':')
    {
        let mut it = rest.split(':');
        let (h, m, s) = (
            parse_zone_part(it.next(), 2)?,
            parse_zone_part(it.next(), 2)?,
            it.next().map_or(Ok(0), |p| parse_zone_part(Some(p), 2))?,
        );
        if it.next().is_some() || h > MAX_ZONE_DISP_HOURS || m > 59 || s > 59 {
            return Err(SessionZoneError::Invalid);
        }
        let offset = -sign * (h * 3600 + m * 60 + s);
        let mut canonical = format!("{}{h:02}:{m:02}", if sign < 0 { '-' } else { '+' });
        if s != 0 {
            let _ = std::fmt::Write::write_fmt(&mut canonical, format_args!(":{s:02}"));
        }
        return Ok((canonical, offset));
    }
    // An IANA-style `Region/City` name would need a time-zone database.
    if v.contains('/')
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'_' | b'+' | b'-'))
    {
        return Err(SessionZoneError::NeedsTzDatabase);
    }
    Err(SessionZoneError::Invalid)
}

/// The east-of-UTC offset a canonical [`parse_session_timezone`] form denotes, or `None` for a
/// string that is not one of its outputs (an unset or hand-poked variable falls back to UTC).
#[must_use]
pub fn session_zone_offset(canonical: &str) -> Option<i64> {
    let v = canonical.trim();
    if ["utc", "gmt", "etc/utc", "etc/gmt"]
        .iter()
        .any(|z| v.eq_ignore_ascii_case(z))
    {
        return Some(0);
    }
    // `<+05:30>-05:30` — the POSIX display; the bracketed abbreviation carries the ISO offset.
    if let Some(inner) = v.strip_prefix('<').and_then(|r| r.split('>').next()) {
        let (sign, rest) = split_sign(inner)?;
        let mut it = rest.split(':');
        let h: i64 = it.next()?.parse().ok()?;
        let m: i64 = it.next().map_or(Some(0), |p| p.parse().ok())?;
        return Some(sign * (h * 3600 + m * 60));
    }
    // `±HH:MM[:SS]` — the POSIX offset spec, sign flipped from ISO.
    let (sign, rest) = split_sign(v)?;
    let mut it = rest.split(':');
    let h: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next().map_or(Some(0), |p| p.parse().ok())?;
    let s: i64 = it.next().map_or(Some(0), |p| p.parse().ok())?;
    Some(-sign * (h * 3600 + m * 60 + s))
}

/// The canonical `timezone` setting string denoting a fixed east-of-UTC offset — what a path that
/// only holds the pinned offset (not the original setting) hands back to the analyzer's pin.
#[must_use]
pub fn zone_setting_for_offset(offset_east_secs: i64) -> String {
    posix_zone_display(offset_east_secs)
}

/// Render a fixed east-of-UTC offset as the reference engine's POSIX zone display:
/// `<+05:30>-05:30` (the bracketed abbreviation uses the ISO sign, the spec part the opposite).
fn posix_zone_display(offset_east_secs: i64) -> String {
    let iso = zone_hhmm(offset_east_secs, false);
    let posix = zone_hhmm(offset_east_secs, true);
    format!("<{iso}>{posix}")
}

/// `±HH[:MM]` for an offset, in ISO sign (east positive) or flipped (POSIX) form.
fn zone_hhmm(offset_east_secs: i64, flip_sign: bool) -> String {
    let shown = if flip_sign {
        -offset_east_secs
    } else {
        offset_east_secs
    };
    let sign = if shown < 0 { '-' } else { '+' };
    let abs = shown.abs();
    let (h, m) = (abs / 3600, (abs % 3600) / 60);
    if m != 0 {
        format!("{sign}{h:02}:{m:02}")
    } else {
        format!("{sign}{h:02}")
    }
}

/// Split a leading `+`/`-` sign off a zone string: `(±1, rest)`.
fn split_sign(v: &str) -> Option<(i64, &str)> {
    match v.as_bytes().first()? {
        b'+' => Some((1, v.get(1..)?)),
        b'-' => Some((-1, v.get(1..)?)),
        _ => None,
    }
}

/// One `HH`/`MM`/`SS` component of a POSIX offset spec: all digits, at most `max_len` long.
fn parse_zone_part(part: Option<&str>, max_len: usize) -> Result<i64, SessionZoneError> {
    let p = part.ok_or(SessionZoneError::Invalid)?;
    if p.is_empty() || p.len() > max_len || !p.bytes().all(|b| b.is_ascii_digit()) {
        return Err(SessionZoneError::Invalid);
    }
    p.parse().map_err(|_| SessionZoneError::Invalid)
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
        "decade" => y.div_euclid(10) as f64,
        "century" => century_of(y) as f64,
        "millennium" => millennium_of(y) as f64,
        "isoyear" => iso_year(days) as f64,
        // The seconds field carried down to microsecond resolution (`0..60_000_000`).
        "microseconds" => (s * MICROS_PER_SEC + us) as f64,
        // The seconds field to millisecond resolution, keeping the sub-millisecond fraction.
        "milliseconds" => (s * MICROS_PER_SEC + us) as f64 / 1_000.0,
        // The Julian Date: whole Julian-day count plus the fraction of the day since midnight.
        // 1970-01-01 is Julian day 2_440_588.
        "julian" => {
            let tod = micros.rem_euclid(MICROS_PER_DAY);
            2_440_588.0 + days as f64 + tod as f64 / MICROS_PER_DAY as f64
        },
        _ => return None,
    };
    Some(val)
}

/// The century of Gregorian year `y`: years 1..100 are century 1, 101..200 century 2, and so on, so
/// year 2026 is century 21. Matches the reference engine (which numbers centuries from year 1, not 0).
const fn century_of(y: i64) -> i64 {
    (y - 1).div_euclid(100) + 1
}

/// The millennium of Gregorian year `y`: years 1..1000 are millennium 1, so year 2026 is millennium 3.
const fn millennium_of(y: i64) -> i64 {
    (y - 1).div_euclid(1000) + 1
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
        // For an interval the decade/century/millennium are the whole-year count divided down (no
        // calendar-origin offset, unlike a timestamp): 25 years is decade 2, 250 years century 2.
        "decade" => (months / 12 / 10) as f64,
        "century" => (months / 12 / 100) as f64,
        "millennium" => (months / 12 / 1000) as f64,
        // The seconds field to microsecond / millisecond resolution.
        "microseconds" => (micros % (60 * MICROS_PER_SEC)) as f64,
        "milliseconds" => (micros % (60 * MICROS_PER_SEC)) as f64 / 1_000.0,
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
        // The sub-minute seconds carried down to micro-/milliseconds, matching the reference engine.
        "microseconds" => (s * MICROS_PER_SEC + us) as f64,
        "milliseconds" => (s * MICROS_PER_SEC + us) as f64 / 1_000.0,
        "epoch" => tod_micros as f64 / MICROS_PER_SEC as f64,
        _ => return None,
    };
    Some(val)
}

/// The three `EXTRACT` zone fields shared by `timestamptz` and `timetz`, from a UTC offset east of
/// UTC in whole seconds.
///
/// `timezone` is that offset in seconds, `timezone_hour` its whole hours, and `timezone_minute` its
/// leftover whole minutes (both keeping the offset's sign). `None` for any other field, so the caller
/// can fall through to its non-zone fields.
#[allow(
    clippy::cast_precision_loss,
    reason = "EXTRACT yields a double-precision number; the offset fits comfortably in an f64"
)]
fn extract_zone_field(field: &str, offset_east_secs: i64) -> Option<f64> {
    let val = match field {
        "timezone" => offset_east_secs as f64,
        "timezone_hour" => (offset_east_secs / 3600) as f64,
        "timezone_minute" => ((offset_east_secs % 3600) / 60) as f64,
        _ => return None,
    };
    Some(val)
}

/// `EXTRACT(field FROM timestamptz)` for the UTC instant `micros` under a session whose offset east
/// of UTC is `offset_east_secs`.
///
/// The zone fields read the offset; every other field reads the instant (already the session-local
/// wall clock, since the engine's session zone is UTC).
#[must_use]
pub fn extract_timestamptz_field(field: &str, micros: i64, offset_east_secs: i64) -> Option<f64> {
    extract_zone_field(field, offset_east_secs).or_else(|| {
        // Calendar fields read the zone-local wall clock; `epoch` alone stays the true UTC epoch.
        let shift = if field == "epoch" {
            0
        } else {
            offset_east_secs.saturating_mul(MICROS_PER_SEC)
        };
        extract_from_micros(field, micros.saturating_add(shift))
    })
}

/// `EXTRACT(field FROM timetz)` for a packed `timetz`.
///
/// The zone fields read the value's own offset, `epoch` is the UTC-equivalent seconds-of-day (local
/// minus offset), and every other field reads the local (as-entered) time-of-day.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "EXTRACT yields a double-precision number; a seconds-of-day count fits in an f64"
)]
pub fn extract_timetz_field(field: &str, packed: i64) -> Option<f64> {
    let offset_east_secs = timetz_offset_east_secs(packed);
    let local = timetz_local_micros(packed);
    if let Some(zone) = extract_zone_field(field, offset_east_secs) {
        return Some(zone);
    }
    if field == "epoch" {
        return Some((local - offset_east_secs * MICROS_PER_SEC) as f64 / MICROS_PER_SEC as f64);
    }
    extract_time_field(field, local)
}

// === EXTRACT as exact NUMERIC =============================================================
//
// `EXTRACT` (unlike `date_part`, which is double precision) yields a `numeric` whose scale is fixed
// per field: `second`/`epoch` carry six fractional digits, `milliseconds` three, `julian` twenty,
// and every other field is a whole number. Computing straight from the integer micro-counts keeps
// microsecond resolution exact even for far-future instants an `f64` could not represent.

/// A whole-number EXTRACT result (scale 0).
const fn extract_int(v: i64) -> crate::numeric::Decimal {
    crate::numeric::Decimal {
        mantissa: v as i128,
        scale: 0,
    }
}

/// The `julian` field as a scale-20 `numeric`: the whole Julian day at midnight plus the fraction of
/// the day elapsed, rounded half-up to twenty places to match the reference engine's rendering.
fn julian_decimal(days: i64, micros: i64) -> crate::numeric::Decimal {
    let scale_pow = 10i128.pow(20);
    let whole = (2_440_588i128 + i128::from(days)) * scale_pow;
    let tod = i128::from(micros.rem_euclid(MICROS_PER_DAY));
    let day = i128::from(MICROS_PER_DAY);
    let frac = (tod * scale_pow + day / 2) / day;
    crate::numeric::Decimal {
        mantissa: whole + frac,
        scale: 20,
    }
}

/// `EXTRACT(field FROM ts)` as an exact `numeric` (see [`extract_from_micros`] for the `date_part`
/// / double-precision form).
#[must_use]
#[allow(
    clippy::many_single_char_names,
    reason = "y/m/d/h/s are the calendar-component names, as in extract_from_micros"
)]
pub fn extract_decimal_from_micros(field: &str, micros: i64) -> Option<crate::numeric::Decimal> {
    use crate::numeric::Decimal;
    let days = micros.div_euclid(MICROS_PER_DAY);
    let (y, m, d, h, mi, s, us) = decompose_micros(micros);
    let sub_min_us = i128::from(s * MICROS_PER_SEC + us);
    let dec = match field {
        "year" => extract_int(y),
        "month" => extract_int(m),
        "day" => extract_int(d),
        "hour" => extract_int(h),
        "minute" => extract_int(mi),
        "second" => Decimal {
            mantissa: sub_min_us,
            scale: 6,
        },
        "dow" => extract_int((days + 4).rem_euclid(7)),
        "isodow" => extract_int(match (days + 4).rem_euclid(7) {
            0 => 7,
            w => w,
        }),
        "doy" => extract_int(days - days_from_civil(y, 1, 1) + 1),
        "quarter" => extract_int((m - 1) / 3 + 1),
        "epoch" => Decimal {
            mantissa: i128::from(micros),
            scale: 6,
        },
        "week" => extract_int(iso_week(days, y)),
        "decade" => extract_int(y.div_euclid(10)),
        "century" => extract_int(century_of(y)),
        "millennium" => extract_int(millennium_of(y)),
        "isoyear" => extract_int(iso_year(days)),
        "microseconds" => extract_int(s * MICROS_PER_SEC + us),
        "milliseconds" => Decimal {
            mantissa: sub_min_us,
            scale: 3,
        },
        "julian" => julian_decimal(days, micros),
        _ => return None,
    };
    Some(dec)
}

/// `EXTRACT(field FROM interval)` as an exact `numeric`.
#[must_use]
pub fn extract_decimal_interval_field(
    field: &str,
    months: i64,
    days: i64,
    micros: i64,
) -> Option<crate::numeric::Decimal> {
    use crate::numeric::Decimal;
    const SECS_PER_DAY: i64 = 86_400;
    // Sub-minute micros, using the truncated remainder (matching the `f64` form) so a negative
    // interval keeps the same sign convention.
    let sub_min_us = i128::from(micros % (60 * MICROS_PER_SEC));
    let dec = match field {
        "year" => extract_int(months / 12),
        "month" => extract_int(months % 12),
        "day" => extract_int(days),
        "hour" => extract_int(micros / (3600 * MICROS_PER_SEC)),
        "minute" => extract_int((micros / (60 * MICROS_PER_SEC)) % 60),
        "second" => Decimal {
            mantissa: i128::from((micros / MICROS_PER_SEC) % 60) * i128::from(MICROS_PER_SEC)
                + i128::from(micros % MICROS_PER_SEC),
            scale: 6,
        },
        "decade" => extract_int(months / 12 / 10),
        "century" => extract_int(months / 12 / 100),
        "millennium" => extract_int(months / 12 / 1000),
        "microseconds" => extract_int(micros % (60 * MICROS_PER_SEC)),
        "milliseconds" => Decimal {
            mantissa: sub_min_us,
            scale: 3,
        },
        // An interval's epoch is the total seconds under the standard 30-day-month, 365.25-day-year
        // convention; every day/month/year term is a whole number of seconds (365.25 * 86400 = 31 557
        // 600 s exactly) and the sub-day part contributes its raw microseconds, so the sum is exact at
        // scale six.
        "epoch" => {
            let years = i128::from(months / 12);
            let rem_months = i128::from(months % 12);
            let secs = years * 31_557_600
                + rem_months * 30 * i128::from(SECS_PER_DAY)
                + i128::from(days) * i128::from(SECS_PER_DAY);
            Decimal {
                mantissa: secs * i128::from(MICROS_PER_SEC) + i128::from(micros),
                scale: 6,
            }
        },
        _ => return None,
    };
    Some(dec)
}

/// `EXTRACT(field FROM time)` as an exact `numeric`.
#[must_use]
pub fn extract_decimal_time_field(field: &str, tod_micros: i64) -> Option<crate::numeric::Decimal> {
    use crate::numeric::Decimal;
    let s = (tod_micros / MICROS_PER_SEC) % 60;
    let us = tod_micros % MICROS_PER_SEC;
    let sub_min_us = i128::from(s * MICROS_PER_SEC + us);
    let dec = match field {
        "hour" => extract_int(tod_micros / (3600 * MICROS_PER_SEC)),
        "minute" => extract_int((tod_micros / (60 * MICROS_PER_SEC)) % 60),
        "second" => Decimal {
            mantissa: sub_min_us,
            scale: 6,
        },
        "microseconds" => extract_int(s * MICROS_PER_SEC + us),
        "milliseconds" => Decimal {
            mantissa: sub_min_us,
            scale: 3,
        },
        "epoch" => Decimal {
            mantissa: i128::from(tod_micros),
            scale: 6,
        },
        _ => return None,
    };
    Some(dec)
}

/// The three zone fields (`timezone`/`timezone_hour`/`timezone_minute`) as exact whole-number
/// `numeric`s, from a UTC offset east of UTC in whole seconds.
fn extract_decimal_zone_field(
    field: &str,
    offset_east_secs: i64,
) -> Option<crate::numeric::Decimal> {
    let v = match field {
        "timezone" => offset_east_secs,
        "timezone_hour" => offset_east_secs / 3600,
        "timezone_minute" => (offset_east_secs % 3600) / 60,
        _ => return None,
    };
    Some(extract_int(v))
}

/// `EXTRACT(field FROM timestamptz)` as an exact `numeric`.
#[must_use]
pub fn extract_decimal_timestamptz_field(
    field: &str,
    micros: i64,
    offset_east_secs: i64,
) -> Option<crate::numeric::Decimal> {
    extract_decimal_zone_field(field, offset_east_secs).or_else(|| {
        // Calendar fields read the zone-local wall clock; `epoch` alone stays the true UTC epoch.
        let shift = if field == "epoch" {
            0
        } else {
            offset_east_secs.saturating_mul(MICROS_PER_SEC)
        };
        extract_decimal_from_micros(field, micros.saturating_add(shift))
    })
}

/// `EXTRACT(field FROM timetz)` as an exact `numeric`.
#[must_use]
pub fn extract_decimal_timetz_field(field: &str, packed: i64) -> Option<crate::numeric::Decimal> {
    let offset_east_secs = timetz_offset_east_secs(packed);
    let local = timetz_local_micros(packed);
    if let Some(zone) = extract_decimal_zone_field(field, offset_east_secs) {
        return Some(zone);
    }
    if field == "epoch" {
        return Some(crate::numeric::Decimal {
            mantissa: i128::from(local - offset_east_secs * MICROS_PER_SEC),
            scale: 6,
        });
    }
    extract_decimal_time_field(field, local)
}

/// `DATE_TRUNC(field, ts)` — `ts` (epoch micros) floored to the start of the named precision.
/// Returns `None` for an unrecognised field. `week` truncates to the preceding Monday 00:00.
#[must_use]
pub fn date_trunc_micros(field: &str, micros: i64) -> Option<i64> {
    let unit = match field {
        "microsecond" | "microseconds" => return Some(micros),
        // Floor to the millisecond: drop the sub-millisecond microseconds.
        "millisecond" | "milliseconds" => return Some(micros - micros.rem_euclid(1_000)),
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
        // The decade floors the year to a multiple of ten; the century and millennium floor to the
        // year that starts them (year 2001 begins century 21 and millennium 3).
        "decade" => days_from_civil(y - y.rem_euclid(10), 1, 1),
        "century" => days_from_civil((century_of(y) - 1) * 100 + 1, 1, 1),
        "millennium" => days_from_civil((millennium_of(y) - 1) * 1000 + 1, 1, 1),
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
        // Borrow one month's worth of days, counted as the length of the *earlier* instant's own
        // month (matching the reference engine — not the month preceding the later instant).
        d += days_in_month(y2, mon2);
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
    /// `Q` — quarter of the year, `1..4`.
    Quarter,
    /// `WW` — week of the year, `1 + (doy - 1) / 7` (2 digits).
    WeekOfYear,
    /// `W` — week of the month, `1 + (day - 1) / 7`.
    WeekOfMonth,
    /// `CC` — century, `(year - 1) / 100 + 1` (2 digits).
    Century,
    /// `J` — Julian day number (days since 4714-11-24 BC, proleptic Gregorian).
    JulianDay,
    /// `IW` — ISO 8601 week of the year (2 digits).
    IsoWeek,
    /// `IYYY`/`IYY`/`IY`/`I` — the last 4/3/2/1 digits of the ISO 8601 year (the `usize` is the
    /// digit count).
    IsoYear(usize),
    /// `RM`/`rm` — the month as an uppercase / lowercase Roman numeral `I..XII` (the bool is upper
    /// case), blank-padded to 4 (the width of `VIII`).
    RomanMonth(bool),
    /// `SSSS`/`SSSSS` — seconds past midnight, `0..86399` (no zero padding).
    SecondsPastMidnight,
    /// `Y,YYY` — the 4-digit year with a comma before its last three digits (`2,024`).
    YearComma,
    /// `AD`/`BC`/`A.D.`/`B.C.` — the era of the value (`first`: upper case, `second`: with dots).
    /// The rendered era reflects the year, not the template letters (`BC` on an AD date is `AD`).
    Era(bool, bool),
    /// `AM`/`PM` (the bool is whether to render upper case).
    Meridiem(bool),
    /// `th`/`TH` — the English ordinal suffix (`st`/`nd`/`rd`/`th`) of the *preceding* numeric
    /// field (the bool is upper case). Emitted only immediately after a numeric field.
    OrdinalSuffix(bool),
    /// `FM` — fill-mode modifier: suppress the leading zeros / trailing blanks of the *next* field.
    /// Emits nothing itself.
    FillMode,
    /// Verbatim text (a quoted `"..."` run, a separator, or any unrecognised character).
    Literal(String),
}

/// Whether a token renders a bare number, so an immediately following `th`/`TH` becomes its ordinal
/// suffix. Name/era/roman/meridiem fields are not numeric.
const fn is_numeric_field(tok: &FmtToken) -> bool {
    use FmtToken as T;
    matches!(
        tok,
        T::Year4
            | T::Year2
            | T::MonthNum
            | T::Day2
            | T::DayOfYear
            | T::IsoDayOfYear
            | T::DayOfWeek
            | T::IsoDayOfWeek
            | T::Hour24
            | T::Hour12
            | T::Minute
            | T::Second
            | T::Milli
            | T::Micro
            | T::Quarter
            | T::WeekOfYear
            | T::WeekOfMonth
            | T::Century
            | T::JulianDay
            | T::IsoWeek
            | T::IsoYear(_)
            | T::SecondsPastMidnight
    )
}

/// The Roman-numeral form of a month `1..=12` (`I`..`XII`); empty for an out-of-range month.
const fn roman_month(month: i64) -> &'static str {
    match month {
        1 => "I",
        2 => "II",
        3 => "III",
        4 => "IV",
        5 => "V",
        6 => "VI",
        7 => "VII",
        8 => "VIII",
        9 => "IX",
        10 => "X",
        11 => "XI",
        12 => "XII",
        _ => "",
    }
}

/// The English ordinal suffix of `n` (`st`/`nd`/`rd`/`th`), in the requested case. Numbers whose last
/// two digits are 11–13 always take `th`.
fn ordinal_suffix(n: i64, upper: bool) -> &'static str {
    // 11–13 (by the last two digits) always take `th`; otherwise the last digit picks the suffix.
    let lower = if (11..=13).contains(&n.rem_euclid(100)) {
        "th"
    } else {
        match n.rem_euclid(10) {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    match (lower, upper) {
        ("st", true) => "ST",
        ("nd", true) => "ND",
        ("rd", true) => "RD",
        (_, true) => "TH",
        (other, false) => other,
    }
}

/// The calendar + clock components of the instant being formatted, so [`numeric_field`] can be a
/// free function rather than a giant closure inside [`format_with_pattern`].
struct TimeParts {
    y: i64,
    m: i64,
    d: i64,
    h: i64,
    mi: i64,
    s: i64,
    us: i64,
    days: i64,
}

/// The `(value, zero-pad width)` a numeric `TO_CHAR` field renders, or `None` for a non-numeric
/// token. Kept in lockstep with [`is_numeric_field`].
fn numeric_field(tok: &FmtToken, p: &TimeParts) -> Option<(i64, usize)> {
    // ISO weekday 1 = Monday .. 7 = Sunday.
    let iso_dow = match (p.days + 4).rem_euclid(7) {
        0 => 7,
        w => w,
    };
    Some(match tok {
        FmtToken::Year4 => (p.y, 4),
        FmtToken::Year2 => (p.y.rem_euclid(100), 2),
        FmtToken::MonthNum => (p.m, 2),
        FmtToken::Day2 => (p.d, 2),
        FmtToken::DayOfYear => (p.days - days_from_civil(p.y, 1, 1) + 1, 3),
        // ISO day of year: within an ISO year, week W day D (1=Mon..7=Sun) is the `(W-1)*7 + D`-th
        // day. `iso_week` handles Jan/Dec dates in the adjacent ISO year.
        FmtToken::IsoDayOfYear => ((iso_week(p.days, p.y) - 1) * 7 + iso_dow, 3),
        // 1 = Sunday .. 7 = Saturday.
        FmtToken::DayOfWeek => ((p.days + 4).rem_euclid(7) + 1, 1),
        FmtToken::IsoDayOfWeek => (iso_dow, 1),
        FmtToken::Hour24 => (p.h, 2),
        FmtToken::Hour12 => ((p.h + 11) % 12 + 1, 2),
        FmtToken::Minute => (p.mi, 2),
        FmtToken::Second => (p.s, 2),
        FmtToken::Milli => (p.us / 1000, 3),
        FmtToken::Micro => (p.us, 6),
        FmtToken::Quarter => ((p.m - 1) / 3 + 1, 1),
        FmtToken::WeekOfYear => ((p.days - days_from_civil(p.y, 1, 1)) / 7 + 1, 2),
        FmtToken::WeekOfMonth => ((p.d - 1) / 7 + 1, 1),
        // Century: 2001..2100 is century 21, matching `(year - 1) / 100 + 1`.
        FmtToken::Century => ((p.y - 1).div_euclid(100) + 1, 2),
        // Julian day number: day 0 (1970-01-01) is JDN 2_440_588.
        FmtToken::JulianDay => (p.days + 2_440_588, 1),
        FmtToken::IsoWeek => (iso_week(p.days, p.y), 2),
        FmtToken::IsoYear(digits) => {
            let modulus = 10_i64.pow(u32::try_from(*digits).unwrap_or(4));
            (iso_year(p.days).rem_euclid(modulus), *digits)
        },
        // Seconds past midnight, 0..86399 (never zero-padded, so a width of 1).
        FmtToken::SecondsPastMidnight => (p.h * 3600 + p.mi * 60 + p.s, 1),
        _ => return None,
    })
}

/// The ISO 8601 calendar year the day `days` belongs to (the Gregorian year of the Thursday of its
/// ISO week).
const fn iso_year(days: i64) -> i64 {
    let iso_dow = match (days + 4).rem_euclid(7) {
        0 => 7,
        w => w,
    };
    // The Thursday of the current ISO week fixes the ISO year (weeks 1..53 all contain their year's
    // Thursday). ISO weekday 1=Mon..7=Sun, so Thursday is `days + (4 - iso_dow)`.
    let (thursday_year, _, _) = civil_from_days(days + 4 - iso_dow);
    thursday_year
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
///
/// `th`/`TH` is deliberately *not* matched here: it is an ordinal-suffix modifier that
/// [`tokenize_format`] only recognises immediately after a numeric field, so a bare `th` stays a
/// literal (as the reference engine renders it).
fn match_pattern(s: &[char]) -> Option<(FmtToken, usize)> {
    use FmtToken as T;
    let name_case = |len: usize| s.get(..len).map_or(NameCase::Title, detect_case);
    let upper = || s.first().is_some_and(char::is_ascii_uppercase);
    // Multi-character / case-bearing patterns whose token depends on the matched text are handled
    // first, longest form before its prefix (`A.D.` before `AD`, `Y,YYY` before `YYYY`).
    if ci_prefix(s, "A.D.") {
        return Some((T::Era(upper(), true), 4));
    }
    if ci_prefix(s, "B.C.") {
        return Some((T::Era(upper(), true), 4));
    }
    if ci_prefix(s, "Y,YYY") {
        return Some((T::YearComma, 5));
    }
    if ci_prefix(s, "RM") {
        return Some((T::RomanMonth(upper()), 2));
    }
    if ci_prefix(s, "AD") || ci_prefix(s, "BC") {
        return Some((T::Era(upper(), false), 2));
    }
    // Longest patterns first so `HH24` wins over `HH`, `YYYY` over `YY`, `MONTH` over `MON`/`MM`,
    // `IYYY` over `IYY`/`IY`/`I`, `SSSSS` over `SSSS`/`SS`, `WW` over `W`.
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
        ("IYYY", 20),
        ("IYY", 21),
        ("IW", 24),
        ("IY", 22),
        ("ID", 15),
        ("I", 23),
        ("SSSSS", 30),
        ("SSSS", 30),
        ("SS", 11),
        ("WW", 25),
        ("W", 26),
        ("CC", 27),
        ("J", 28),
        ("Q", 29),
        ("AM", 12),
        ("PM", 12),
        ("FM", 19),
        ("D", 14),
    ];
    for (pat, id) in candidates {
        if ci_prefix(s, pat) {
            let len = pat.chars().count();
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
                20 => T::IsoYear(4),
                21 => T::IsoYear(3),
                22 => T::IsoYear(2),
                23 => T::IsoYear(1),
                24 => T::IsoWeek,
                25 => T::WeekOfYear,
                26 => T::WeekOfMonth,
                27 => T::Century,
                28 => T::JulianDay,
                29 => T::Quarter,
                30 => T::SecondsPastMidnight,
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
            let numeric = is_numeric_field(&tok);
            tokens.push(tok);
            i += len;
            // A `th`/`TH` immediately after a numeric field is its ordinal suffix; anywhere else it
            // is a plain literal (a separator or a bare `th` renders verbatim, like the reference).
            if numeric
                && let (Some(&c0), Some(&c1)) = (chars.get(i), chars.get(i + 1))
                && c0.eq_ignore_ascii_case(&'t')
                && c1.eq_ignore_ascii_case(&'h')
            {
                tokens.push(FmtToken::OrdinalSuffix(c0.is_ascii_uppercase()));
                i += 2;
            }
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
    format_parts(
        &TimeParts {
            y,
            m,
            d,
            h,
            mi,
            s,
            us,
            days,
        },
        fmt,
    )
}

/// `TO_CHAR(interval, fmt)` — render an interval's component fields through a datetime picture.
///
/// The interval is read field-wise as the reference engine does: `YYYY` is the whole years
/// (`months / 12`), `MM` the leftover months, `DD` the days, and `HH24`/`MI`/`SS` the time carried by
/// the micros part — so `HH24` of `26 hours` is `26` (the micros are not folded into a day). The
/// calendar-only codes (weekday, day-of-year, era, …) have no meaning for a duration and read from a
/// zeroed date.
#[must_use]
pub fn format_interval_with_pattern(months: i64, days: i64, micros: i64, fmt: &str) -> String {
    format_parts(
        &TimeParts {
            y: months / 12,
            m: months % 12,
            d: days,
            h: micros / (3600 * MICROS_PER_SEC),
            mi: (micros / (60 * MICROS_PER_SEC)) % 60,
            s: (micros / MICROS_PER_SEC) % 60,
            us: micros % MICROS_PER_SEC,
            days: 0,
        },
        fmt,
    )
}

/// Render a datetime picture `fmt` over already-decomposed [`TimeParts`]. Shared by the timestamp and
/// interval `TO_CHAR` entry points.
fn format_parts(parts: &TimeParts, fmt: &str) -> String {
    // `y`/`m`/`h`/`days` are read directly by the year, month-name, meridiem and weekday codes; the
    // rest reach the renderer through `numeric_field(&tok, parts)`.
    let TimeParts { y, m, h, days, .. } = *parts;
    let mut out = String::new();
    // `FM` sets fill mode for the next field only; it emits nothing and does not itself reset the
    // flag. A literal between `FM` and its field passes the flag through; every rendered field
    // consumes and clears it.
    let mut fill = false;
    // The numeric value of the most recently rendered numeric field, so an immediately following
    // `th`/`TH` can form its ordinal suffix. Cleared by any non-numeric field.
    let mut last_num: Option<i64> = None;
    for tok in tokenize_format(fmt) {
        if matches!(tok, FmtToken::FillMode) {
            fill = true;
            continue;
        }
        let is_literal = matches!(tok, FmtToken::Literal(_));
        let is_ordinal = matches!(tok, FmtToken::OrdinalSuffix(_));
        if let Some((value, width)) = numeric_field(&tok, parts) {
            out.push_str(&num_field(value, width, fill));
            last_num = Some(value);
            fill = false;
            continue;
        }
        match tok {
            FmtToken::OrdinalSuffix(upper) => {
                if let Some(n) = last_num {
                    out.push_str(ordinal_suffix(n, upper));
                }
            },
            FmtToken::YearComma => {
                // The 4-digit (min) year with a comma before its last three digits: 2024 -> "2,024".
                let digits = format!("{y:04}");
                let split = digits.len() - 3;
                out.push_str(digits.get(..split).unwrap_or_default());
                out.push(',');
                out.push_str(digits.get(split..).unwrap_or_default());
            },
            FmtToken::Era(upper, dotted) => {
                let ad = y >= 1;
                out.push_str(match (ad, upper, dotted) {
                    (true, true, false) => "AD",
                    (true, false, false) => "ad",
                    (true, true, true) => "A.D.",
                    (true, false, true) => "a.d.",
                    (false, true, false) => "BC",
                    (false, false, false) => "bc",
                    (false, true, true) => "B.C.",
                    (false, false, true) => "b.c.",
                });
            },
            FmtToken::RomanMonth(upper) => {
                let roman = roman_month(m);
                let cased = if upper {
                    roman.to_owned()
                } else {
                    roman.to_lowercase()
                };
                // Blank-padded to 4 (the width of "VIII") unless `FM` suppresses the padding.
                if fill {
                    out.push_str(&cased);
                } else {
                    let _ = write!(out, "{cased:<4}");
                }
            },
            FmtToken::MonthAbbr(case) => out.push_str(&month_name(m, &MONTH_ABBR, case)),
            // The full month name is blank-padded to the longest month's width (9 = "September")
            // unless `FM` suppresses the padding.
            FmtToken::MonthFull(case) => {
                out.push_str(&name_field(&month_name(m, &MONTH_FULL, case), fill));
            },
            FmtToken::WeekdayAbbr(case) => out.push_str(&weekday_name(days, &WEEKDAY_ABBR, case)),
            // The full weekday name is blank-padded to the longest name's width (9 = "Wednesday")
            // unless `FM` suppresses the padding.
            FmtToken::WeekdayFull(case) => {
                out.push_str(&name_field(&weekday_name(days, &WEEKDAY_FULL, case), fill));
            },
            FmtToken::Meridiem(upper) => {
                out.push_str(match (h < 12, upper) {
                    (true, true) => "AM",
                    (true, false) => "am",
                    (false, true) => "PM",
                    (false, false) => "pm",
                });
            },
            FmtToken::Literal(lit) => out.push_str(&lit),
            // `FillMode` is handled before the match and every numeric field is handled above, so
            // the remaining tokens (and the unreachable `FillMode`) render nothing here.
            _ => {},
        }
        // A non-numeric field ends any run a `th`/`TH` could suffix (a `th` after a name renders
        // nothing) and consumes a pending fill flag; a literal leaves both pending.
        if !is_literal {
            fill = false;
            if !is_ordinal {
                last_num = None;
            }
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
            // An ordinal suffix (`th`/`TH`) does not constrain the value; consume the two letters.
            FmtToken::OrdinalSuffix(_) => {
                pos += 2;
            },
            FmtToken::Literal(lit) => consume_literal(&inp, &mut pos, &lit)?,
            // Fields that either do not model back into a calendar date (`Q`/`WW`/`W`/`CC`/`IW`/
            // `SSSS`/era/Roman month/ISO day-of-year) or would need machinery the format engine does
            // not carry (`J`/`IYYY`…) are loud-rejected on the parse side rather than silently
            // ignored (`IDDD` would need the ISO year, which is not modelled).
            FmtToken::IsoDayOfYear
            | FmtToken::Quarter
            | FmtToken::WeekOfYear
            | FmtToken::WeekOfMonth
            | FmtToken::Century
            | FmtToken::JulianDay
            | FmtToken::IsoWeek
            | FmtToken::IsoYear(_)
            | FmtToken::RomanMonth(_)
            | FmtToken::SecondsPastMidnight
            | FmtToken::YearComma
            | FmtToken::Era(_, _) => return None,
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
    fn session_timezone_forms_match_the_reference_engine() {
        // Names: canonical case, offset 0.
        assert_eq!(parse_session_timezone("utc"), Ok(("UTC".to_owned(), 0)));
        assert_eq!(parse_session_timezone("GMT"), Ok(("GMT".to_owned(), 0)));
        assert_eq!(
            parse_session_timezone("etc/utc"),
            Ok(("Etc/UTC".to_owned(), 0))
        );
        // A bare `±HH` is a zone-abbreviation spelling: ISO sign (east), POSIX display.
        assert_eq!(
            parse_session_timezone("+07"),
            Ok(("<+07>-07".to_owned(), 7 * 3600))
        );
        assert_eq!(
            parse_session_timezone("-08"),
            Ok(("<-08>+08".to_owned(), -8 * 3600))
        );
        // `±HH:MM` is a POSIX offset spec: the sign is the OPPOSITE of ISO (`+` is west).
        assert_eq!(
            parse_session_timezone("+05:30"),
            Ok(("+05:30".to_owned(), -(5 * 3600 + 1800)))
        );
        assert_eq!(
            parse_session_timezone("-05:30"),
            Ok(("-05:30".to_owned(), 5 * 3600 + 1800))
        );
        // A number is an hour count east of UTC.
        assert_eq!(
            parse_session_timezone("5.5"),
            Ok(("<+05:30>-05:30".to_owned(), 5 * 3600 + 1800))
        );
        assert_eq!(
            parse_session_timezone("-8"),
            Ok(("<-08>+08".to_owned(), -8 * 3600))
        );
        // Errors: garbage and 4-digit offsets are invalid; an IANA name needs a tz database.
        assert_eq!(
            parse_session_timezone("zzz"),
            Err(SessionZoneError::Invalid)
        );
        assert_eq!(
            parse_session_timezone("+0530"),
            Err(SessionZoneError::Invalid)
        );
        assert_eq!(
            parse_session_timezone("+16"),
            Err(SessionZoneError::Invalid)
        );
        assert_eq!(
            parse_session_timezone("Asia/Jakarta"),
            Err(SessionZoneError::NeedsTzDatabase)
        );
        // Every canonical form parses back to its own offset.
        for v in ["utc", "GMT", "+07", "-08", "+05:30", "-05:30", "5.5", "-8"] {
            let (canonical, offset) = parse_session_timezone(v).unwrap();
            assert_eq!(
                session_zone_offset(&canonical),
                Some(offset),
                "round-trip for {v}"
            );
        }
    }

    #[test]
    fn timestamptz_at_renders_and_parses_on_the_session_wall_clock() {
        let noon = parse_timestamptz("2024-01-01 12:00:00+00").unwrap();
        assert_eq!(format_timestamptz_at(noon, 0), "2024-01-01 12:00:00+00");
        assert_eq!(
            format_timestamptz_at(noon, 7 * 3600),
            "2024-01-01 19:00:00+07"
        );
        assert_eq!(
            format_timestamptz_at(noon, -(5 * 3600 + 1800)),
            "2024-01-01 06:30:00-05:30"
        );
        // A missing offset is session-local wall time; an explicit offset (or Z) wins.
        assert_eq!(
            parse_timestamptz_at("2024-01-01 12:00:00", 7 * 3600),
            parse_timestamptz("2024-01-01 05:00:00+00")
        );
        assert_eq!(
            parse_timestamptz_at("2024-01-01 12:00:00+00", 7 * 3600),
            Some(noon)
        );
        assert_eq!(
            parse_timestamptz_at("2024-01-01 12:00:00Z", 7 * 3600),
            Some(noon)
        );
        assert!(has_explicit_offset("2024-01-01 12:00:00+07"));
        assert!(has_explicit_offset("2024-01-01 12:00:00Z"));
        assert!(!has_explicit_offset("2024-01-01 12:00:00"));
        assert!(!has_explicit_offset("2024-01-01"));
    }

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
        for s in ["00:00:00", "23:59:59", "12:30:45.123456", "01:02:03.0005"] {
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
            "2024-06-15 13:45:30.5"
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
        assert_eq!(fmt("12:00:00.250000+03"), "12:00:00.25+03");

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
    fn extract_reads_extended_calendar_fields() {
        // Reference-engine oracle values for 2026-08-20 13:45:30.123456.
        let ts = parse_timestamp("2026-08-20 13:45:30.123456").unwrap();
        assert_eq!(extract_from_micros("century", ts), Some(21.0));
        assert_eq!(extract_from_micros("decade", ts), Some(202.0));
        assert_eq!(extract_from_micros("millennium", ts), Some(3.0));
        assert_eq!(extract_from_micros("isoyear", ts), Some(2026.0));
        assert_eq!(extract_from_micros("microseconds", ts), Some(30_123_456.0));
        assert_eq!(extract_from_micros("milliseconds", ts), Some(30_123.456));
        // Julian Date: 2_461_273 whole days + the fraction of the day since midnight.
        let julian = extract_from_micros("julian", ts).unwrap();
        assert!(
            (julian - 2_461_273.573_265_3).abs() < 1e-6,
            "julian was {julian}"
        );
        // Century numbering starts at year 1: 2000 is still century 20, 2001 begins century 21.
        let c = |s: &str| extract_from_micros("century", parse_timestamp(s).unwrap());
        assert_eq!(c("2000-06-01 00:00:00"), Some(20.0));
        assert_eq!(c("2001-06-01 00:00:00"), Some(21.0));
        // A DATE's Julian Date is the whole day count (no time fraction).
        let jd = extract_from_micros("julian", parse_timestamp("2026-08-20 00:00:00").unwrap());
        assert_eq!(jd, Some(2_461_273.0));
    }

    #[test]
    fn extract_decimal_matches_reference_engine_scales() {
        // EXTRACT is an exact NUMERIC: whole fields at scale 0, second/epoch at scale 6,
        // milliseconds at scale 3, julian at scale 20. Every value below is the reference
        // engine's `EXTRACT(...)::text`.
        let d = |field: &str, ts_text: &str| {
            extract_decimal_from_micros(field, parse_timestamp(ts_text).unwrap())
                .unwrap()
                .format()
        };
        let ts = "2024-06-15 14:30:45.123456";
        assert_eq!(d("year", ts), "2024");
        assert_eq!(d("quarter", ts), "2");
        assert_eq!(d("dow", ts), "6");
        assert_eq!(d("doy", ts), "167");
        assert_eq!(d("week", ts), "24");
        assert_eq!(d("second", ts), "45.123456");
        assert_eq!(d("milliseconds", ts), "45123.456");
        assert_eq!(d("microseconds", ts), "45123456");
        assert_eq!(d("epoch", ts), "1718461845.123456");
        assert_eq!(d("julian", ts), "2460477.60468892888888888889");

        // Interval fields (30-day month, 365.25-day year for epoch — all whole micros, exact).
        let iv = |field: &str, months: i64, days: i64, micros: i64| {
            extract_decimal_interval_field(field, months, days, micros)
                .unwrap()
                .format()
        };
        // interval '1 year 2 mons 3 days 4:05:06.5'
        let hms = (4 * 3600 + 5 * 60 + 6) * MICROS_PER_SEC + 500_000;
        assert_eq!(iv("epoch", 14, 3, hms), "37015506.500000");
        assert_eq!(
            iv(
                "second",
                0,
                0,
                2 * 60 * MICROS_PER_SEC + 6 * MICROS_PER_SEC + 500_000
            ),
            "6.500000"
        );
        assert_eq!(iv("epoch", 12, 0, 0), "31557600.000000");

        // Time fields.
        let tod =
            13 * 3600 * MICROS_PER_SEC + 45 * 60 * MICROS_PER_SEC + 30 * MICROS_PER_SEC + 500_000;
        assert_eq!(
            extract_decimal_time_field("second", tod).unwrap().format(),
            "30.500000"
        );
        assert_eq!(
            extract_decimal_time_field("epoch", tod).unwrap().format(),
            "49530.500000"
        );

        // An inapplicable field is still rejected (None), like the f64 form.
        assert_eq!(extract_decimal_from_micros("nonsense", 0), None);
    }

    #[test]
    fn date_trunc_extended_precisions() {
        let ts = parse_timestamp("2026-08-20 13:45:30.123456").unwrap();
        let fmt = |field: &str| format_timestamp(date_trunc_micros(field, ts).unwrap());
        assert_eq!(fmt("decade"), "2020-01-01 00:00:00");
        assert_eq!(fmt("century"), "2001-01-01 00:00:00");
        assert_eq!(fmt("millennium"), "2001-01-01 00:00:00");
        // Millisecond floors the sub-millisecond microseconds (…123456 → …123000); microsecond is a
        // no-op. Checked on the raw micros so the assertion does not depend on fractional rendering.
        assert_eq!(date_trunc_micros("millisecond", ts), Some(ts - 456));
        assert_eq!(date_trunc_micros("microseconds", ts), Some(ts));
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
    fn out_of_range_date_is_distinguished_from_a_bad_format() {
        // Well-shaped `YYYY-MM-DD` but a field out of range — the reference engine's 22008 case.
        assert!(is_date_field_out_of_range("2023-02-30"));
        assert!(is_date_field_out_of_range("2023-13-01"));
        assert!(is_date_field_out_of_range("99-99-99"));
        // Mis-shaped input — the 22007 (invalid format) case, not out-of-range.
        assert!(!is_date_field_out_of_range("abc"));
        assert!(!is_date_field_out_of_range("2023-02"));
        assert!(!is_date_field_out_of_range("2023/02/28"));
        // A real calendar day is neither.
        assert!(!is_date_field_out_of_range("2024-02-29"));
    }

    #[test]
    fn extract_time_only_accepts_intraday_fields() {
        let tod = parse_time("13:45:30").unwrap();
        assert_eq!(extract_time_field("hour", tod), Some(13.0));
        assert_eq!(extract_time_field("minute", tod), Some(45.0));
        assert_eq!(extract_time_field("second", tod), Some(30.0));
        // Micro-/milliseconds carry the sub-minute seconds, matching the reference engine.
        let frac = parse_time("12:34:56.789012").unwrap();
        assert_eq!(extract_time_field("microseconds", frac), Some(56_789_012.0));
        assert_eq!(extract_time_field("milliseconds", frac), Some(56_789.012));
        // A calendar field is meaningless for TIME.
        assert_eq!(extract_time_field("year", tod), None);
    }

    #[test]
    fn extract_timetz_fields_match_the_reference_engine() {
        // 12:34:56.789 with a +05:30 zone.
        let tt = parse_timetz("12:34:56.789+05:30").unwrap();
        // Local time-of-day fields.
        assert_eq!(extract_timetz_field("hour", tt), Some(12.0));
        assert_eq!(extract_timetz_field("minute", tt), Some(34.0));
        assert_eq!(extract_timetz_field("second", tt), Some(56.789));
        // Zone fields read the offset (east positive).
        assert_eq!(extract_timetz_field("timezone", tt), Some(19800.0));
        assert_eq!(extract_timetz_field("timezone_hour", tt), Some(5.0));
        assert_eq!(extract_timetz_field("timezone_minute", tt), Some(30.0));
        // epoch is the UTC-equivalent seconds-of-day (local minus offset).
        assert_eq!(extract_timetz_field("epoch", tt), Some(25496.789));
        // A negative offset carries its sign into both hour and minute.
        let west = parse_timetz("12:00-08:00").unwrap();
        assert_eq!(extract_timetz_field("timezone_hour", west), Some(-8.0));
        let half = parse_timetz("12:00-05:30").unwrap();
        assert_eq!(extract_timetz_field("timezone_minute", half), Some(-30.0));
    }

    #[test]
    fn extract_timestamptz_zone_fields_are_zero_under_utc() {
        let ts = parse_timestamp("2024-06-15 12:00:00").unwrap();
        // The session zone is UTC, so a timestamptz instant carries a zero offset.
        assert_eq!(extract_timestamptz_field("timezone", ts, 0), Some(0.0));
        assert_eq!(extract_timestamptz_field("timezone_hour", ts, 0), Some(0.0));
        assert_eq!(
            extract_timestamptz_field("timezone_minute", ts, 0),
            Some(0.0)
        );
        // A non-zone field still reads the instant.
        assert_eq!(extract_timestamptz_field("hour", ts, 0), Some(12.0));
        // A plain timestamp errors on a zone field: that path never calls this helper, and here the
        // instant-based fallback has no zone field, so the caller surfaces the error.
        assert_eq!(extract_from_micros("timezone", ts), None);
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
        // A day borrow counts the *earlier* instant's own month: 1-15 borrows January's 31 days,
        // giving 17, not February's 29 (oracle-checked: `1 mon 17 days`).
        assert_eq!(calendar_age(later, earlier), (1, 17, 0));
        // Swapping negates every component.
        assert_eq!(calendar_age(earlier, later), (-1, -17, 0));
        // Sub-day difference lands entirely in the micros component.
        let a = parse_timestamp("2024-01-01 12:00:00").unwrap();
        let b = parse_timestamp("2024-01-01 10:30:00").unwrap();
        assert_eq!(calendar_age(a, b), (0, 0, 90 * 60 * MICROS_PER_SEC));

        // The borrow always uses the earlier date's month length, across leap and non-leap years and
        // every earlier-month length — each oracle-checked against the reference engine.
        let age = |e: &str, s: &str| {
            calendar_age(
                parse_timestamp(&format!("{e} 00:00:00")).unwrap(),
                parse_timestamp(&format!("{s} 00:00:00")).unwrap(),
            )
        };
        // Earlier month = January (31): the finding's own case.
        assert_eq!(age("2024-03-01", "2023-01-15"), (13, 17, 0));
        // Earlier month = November (30).
        assert_eq!(age("2024-01-01", "2023-11-30"), (1, 1, 0));
        // Earlier month = February, leap (29) vs non-leap (28) vs 1900 (28, not a leap year).
        assert_eq!(age("2024-03-10", "2024-02-20"), (0, 19, 0));
        assert_eq!(age("2023-03-10", "2023-02-20"), (0, 18, 0));
        assert_eq!(age("1900-03-01", "1900-02-15"), (0, 14, 0));
        // Earlier month = December (31), crossing the year with a month-underflow borrow too.
        assert_eq!(age("2023-03-01", "2022-12-15"), (2, 17, 0));
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
    fn format_with_pattern_new_codes_match_reference() {
        // All expected outputs captured from the reference engine for 2024-06-15 13:45:30.
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        let f = |fmt: &str| format_with_pattern(ts, fmt);
        assert_eq!(f("Q"), "2");
        assert_eq!(f("WW"), "24");
        assert_eq!(f("W"), "3");
        assert_eq!(f("CC"), "21");
        assert_eq!(f("J"), "2460477");
        assert_eq!(f("IW"), "24");
        assert_eq!(f("IYYY"), "2024");
        assert_eq!(f("IYY"), "024");
        assert_eq!(f("IY"), "24");
        assert_eq!(f("I"), "4");
        assert_eq!(f("RM"), "VI  ");
        assert_eq!(f("rm"), "vi  ");
        assert_eq!(f("SSSS"), "49530");
        assert_eq!(f("SSSSS"), "49530");
        assert_eq!(f("Y,YYY"), "2,024");
        // The era reflects the year, not the template letters, and keeps the template's case + dots.
        assert_eq!(f("BC"), "AD");
        assert_eq!(f("AD"), "AD");
        assert_eq!(f("B.C."), "A.D.");
        assert_eq!(f("A.D."), "A.D.");
        assert_eq!(f("ad"), "ad");
        assert_eq!(f("a.d."), "a.d.");
        // `RN`/`rn` are numeric-only in the reference engine; a datetime template emits them
        // verbatim (`R` and `N` are plain literals), which this already does.
        assert_eq!(f("RN"), "RN");
        assert_eq!(f("rn"), "rn");
        // Lower-case spellings of the value codes are accepted (case-insensitive), same values.
        assert_eq!(f("q ww cc j iw iyyy ssss"), "2 24 21 2460477 24 2024 49530");
        // An unknown letter run still passes through as a literal (matching the reference).
        assert_eq!(f("ZZZ"), "ZZZ");
        // `W` is a code even inside a word: `Week` -> week-of-month `3` then literal `eek`.
        assert_eq!(f("Week"), "3eek");
    }

    #[test]
    fn format_with_pattern_century_julian_and_iso_year_edges() {
        let cc = |s: &str| format_with_pattern(parse_timestamp(s).unwrap(), "CC");
        assert_eq!(cc("2000-01-01 00:00:00"), "20");
        assert_eq!(cc("2001-01-01 00:00:00"), "21");
        assert_eq!(cc("1999-12-31 00:00:00"), "20");
        let j = |s: &str| format_with_pattern(parse_timestamp(s).unwrap(), "J");
        assert_eq!(j("1970-01-01 00:00:00"), "2440588");
        assert_eq!(j("2000-01-01 00:00:00"), "2451545");
        // ISO year differs from the Gregorian year at the boundaries.
        let iso = |s: &str, fmt: &str| format_with_pattern(parse_timestamp(s).unwrap(), fmt);
        // 2023-01-01 (Sunday) belongs to ISO week 52 of ISO-year 2022.
        assert_eq!(iso("2023-01-01 00:00:00", "IYYY-IW"), "2022-52");
        assert_eq!(iso("2023-01-01 00:00:00", "IYY IY I"), "022 22 2");
        // 2021-01-01 (Friday) belongs to ISO week 53 of ISO-year 2020.
        assert_eq!(iso("2021-01-01 00:00:00", "IYYY IW"), "2020 53");
        // Week-of-year / week-of-month boundaries.
        let wy = |s: &str| format_with_pattern(parse_timestamp(s).unwrap(), "WW");
        assert_eq!(wy("2024-01-08 00:00:00"), "02");
        assert_eq!(wy("2024-12-31 00:00:00"), "53");
        // Roman month for August is the widest form (VIII); `FM` strips the blank padding.
        let aug = parse_timestamp("2024-08-15 00:00:00").unwrap();
        assert_eq!(format_with_pattern(aug, "RM"), "VIII");
        assert_eq!(format_with_pattern(aug, "FMRM"), "VIII");
        assert_eq!(format_with_pattern(aug, "FMrm"), "viii");
        // `Y,YYY` on a small year keeps its 4-digit-with-comma shape, even under `FM`.
        let yr7 = parse_timestamp("0007-01-01 00:00:00").unwrap();
        assert_eq!(format_with_pattern(yr7, "Y,YYY"), "0,007");
        assert_eq!(format_with_pattern(yr7, "FMY,YYY"), "0,007");
    }

    #[test]
    fn format_with_pattern_ordinal_suffix_matches_reference() {
        // The suffix is the English ordinal of the preceding number's value (11–13 always `th`).
        let dd = |day: i64| {
            let ts = parse_timestamp(&format!("2024-01-{day:02} 00:00:00")).unwrap();
            format_with_pattern(ts, "DDth")
        };
        assert_eq!(dd(1), "01st");
        assert_eq!(dd(2), "02nd");
        assert_eq!(dd(3), "03rd");
        assert_eq!(dd(4), "04th");
        assert_eq!(dd(11), "11th");
        assert_eq!(dd(12), "12th");
        assert_eq!(dd(13), "13th");
        assert_eq!(dd(21), "21st");
        assert_eq!(dd(22), "22nd");
        assert_eq!(dd(23), "23rd");
        assert_eq!(dd(24), "24th");
        assert_eq!(dd(31), "31st");
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        // `TH` upper-cases the suffix, and it attaches to any numeric field.
        assert_eq!(format_with_pattern(ts, "DDTH"), "15TH");
        assert_eq!(format_with_pattern(ts, "YYYYth"), "2024th");
        assert_eq!(format_with_pattern(ts, "MMth"), "06th");
        assert_eq!(format_with_pattern(ts, "HH24th"), "13th");
        assert_eq!(format_with_pattern(ts, "DDDth"), "167th");
        // `FM` strips the field's zero-padding but the suffix still comes from the value.
        let d5 = parse_timestamp("2024-01-05 00:00:00").unwrap();
        assert_eq!(format_with_pattern(d5, "FMDDth"), "5th");
        // A `th` not immediately after a number is a plain literal (matching the reference).
        assert_eq!(format_with_pattern(ts, "DD th"), "15 th");
        assert_eq!(format_with_pattern(ts, "th"), "th");
    }

    #[test]
    fn format_with_pattern_combined_formats_match_reference() {
        // Full realistic formats mixing old and new codes, captured from the reference engine.
        let ts = parse_timestamp("2024-06-15 13:45:30").unwrap();
        assert_eq!(
            format_with_pattern(
                ts,
                "FMDay, DDth \"of\" FMMonth YYYY \"(Q\"Q\", week \"WW\", J=\"J\")"
            ),
            "Saturday, 15th of June 2024 (Q2, week 24, J=2460477)"
        );
        let morning = parse_timestamp("2024-06-15 09:05:03").unwrap();
        assert_eq!(
            format_with_pattern(morning, "HH12:MI:SSam CCth \"cent\""),
            "09:05:03am 21st cent"
        );
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
