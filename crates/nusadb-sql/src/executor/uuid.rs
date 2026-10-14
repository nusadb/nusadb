//! Time-ordered UUIDs: the version 7 generator behind `UUIDV7()` and the field readers behind
//! `UUID_EXTRACT_TIMESTAMP()` and `UUID_EXTRACT_VERSION()`.
//!
//! A version 7 UUID (RFC 9562 §5.7) is laid out as
//!
//! ```text
//! unix_ts_ms (48 bits) | ver = 0111 (4) | rand_a (12) | var = 10 (2) | rand_b (62)
//! ```
//!
//! # Monotonicity
//!
//! The generator uses RFC 9562 §6.2 method 3, extra clock precision: `rand_a` carries the
//! sub-millisecond part of the clock scaled to 12 bits (steps of 1/4096 ms, about 244 ns). The 60 bits
//! in front of the variant then read as a single timestamp counted in those steps, called a tick here.
//!
//! The last tick handed out is kept in one process-wide atomic. If the clock reading is not later
//! than it (two calls inside the same step, or a clock that moved backwards), the generator hands out
//! the last tick plus one instead. In that case `rand_a` behaves as a counter, and a carry out of it
//! moves the millisecond field forward, which §6.2 permits. Every value is therefore strictly greater
//! than the one before it in the whole process, which covers successive calls in one statement, one
//! transaction and one session. If calls arrive faster than one per step for a sustained period, the
//! embedded time runs slightly ahead of the wall clock and falls back in line once the rate drops.
//!
//! `rand_b` comes from the OS CSPRNG (`getrandom`), never from the seedable `RANDOM()` generator, so
//! `SETSEED` cannot make the values predictable.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Error;
use crate::interval::Interval;

/// Bits of `rand_a`, the sub-millisecond fraction.
const FRACTION_BITS: u32 = 12;

/// Ticks per millisecond (`2^FRACTION_BITS`).
const TICKS_PER_MS: u128 = 1 << FRACTION_BITS;

/// One past the largest tick a UUID can hold: 48 bits of milliseconds plus the 12-bit fraction.
const TICK_LIMIT: u64 = 1 << 60;

/// Microseconds between the start of the Gregorian calendar (1582-10-15), the epoch of a version 1
/// timestamp, and the Unix epoch.
const GREGORIAN_TO_UNIX_MICROS: i64 = 12_219_292_800_000_000;

/// The last tick handed out by [`uuid_v7`] in this process (`0` before the first call).
static LAST_TICK: AtomicU64 = AtomicU64::new(0);

/// A fresh version 7 UUID. `shift`, when given, moves the embedded timestamp by that interval after
/// the monotonic tick is taken, so values generated with one fixed shift keep their order.
///
/// # Errors
/// [`Error::InvalidParameterValue`] when the shifted timestamp falls outside what 48 bits of Unix
/// milliseconds can hold, and an I/O error when the OS entropy source fails.
pub(super) fn uuid_v7(shift: Option<&Interval>) -> Result<[u8; 16], Error> {
    let mut tick = issue_tick();
    if let Some(shift) = shift {
        tick = shift_tick(tick, shift).ok_or_else(|| {
            Error::InvalidParameterValue(
                "uuidv7: the shifted timestamp is out of range for a UUID".to_owned(),
            )
        })?;
    }
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes[8..]).map_err(|e| {
        Error::Core(nusadb_core::Error::Io(std::io::Error::other(format!(
            "entropy failure generating UUID: {e}"
        ))))
    })?;
    Ok(encode_v7(tick, bytes))
}

/// Lay out a version 7 UUID from a tick and a buffer whose bytes 8..16 already hold random data.
fn encode_v7(tick: u64, mut bytes: [u8; 16]) -> [u8; 16] {
    let tick_bytes = (tick << 4).to_be_bytes();
    // The 48-bit millisecond count fills bytes 0..6.
    bytes[..6].copy_from_slice(&tick_bytes[..6]);
    // Version nibble, then the 12-bit fraction across the rest of byte 6 and byte 7.
    bytes[6] = 0x70 | (tick_bytes[6] >> 4);
    bytes[7] = (tick_bytes[6] << 4) | (tick_bytes[7] >> 4);
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
    bytes
}

/// Take the next tick: the clock reading, or one past the last tick handed out when the clock has
/// not moved beyond it.
fn issue_tick() -> u64 {
    let now = clock_tick();
    let mut last = LAST_TICK.load(Ordering::Acquire);
    loop {
        let next = next_tick(last, now);
        match LAST_TICK.compare_exchange_weak(last, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return next,
            Err(current) => last = current,
        }
    }
}

/// The tick to hand out after `last` when the clock reads `now`: always strictly greater than `last`.
const fn next_tick(last: u64, now: u64) -> u64 {
    if now > last {
        now
    } else {
        last.saturating_add(1)
    }
}

/// The wall clock as a tick: milliseconds since the Unix epoch shifted left by [`FRACTION_BITS`],
/// plus the sub-millisecond remainder scaled to that many bits. A clock before the epoch reads `0`.
fn clock_tick() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let ms = nanos / 1_000_000;
    let fraction = (nanos % 1_000_000) * TICKS_PER_MS / 1_000_000;
    u64::try_from((ms << FRACTION_BITS) | fraction).unwrap_or(u64::MAX)
}

/// Move `tick` by `shift`. Month and day parts follow the calendar from the tick's millisecond, and
/// the change is applied in whole ticks so the fraction (and with it the order between values) is
/// kept. `None` when the result does not fit in a UUID.
fn shift_tick(tick: u64, shift: &Interval) -> Option<u64> {
    let base_micros = i64::try_from(tick >> FRACTION_BITS)
        .ok()?
        .checked_mul(1000)?;
    let shifted_micros = crate::temporal::add_interval_to_micros(
        base_micros,
        shift.months,
        shift.days,
        shift.micros,
    );
    let delta_micros = i128::from(shifted_micros) - i128::from(base_micros);
    let delta_ticks = (delta_micros * i128::from(1u16 << FRACTION_BITS)).div_euclid(1000);
    u64::try_from(i128::from(tick) + delta_ticks)
        .ok()
        .filter(|t| *t < TICK_LIMIT)
}

/// Whether `uuid` has the standard variant (`10` in the top bits of byte 8).
const fn is_standard_variant(uuid: &[u8; 16]) -> bool {
    uuid[8] & 0xc0 == 0x80
}

/// The version field of a standard-variant UUID, or `None` for any other variant.
pub(super) fn extract_version(uuid: &[u8; 16]) -> Option<i64> {
    is_standard_variant(uuid).then(|| i64::from(uuid[6] >> 4))
}

/// The timestamp embedded in a standard-variant version 1 or version 7 UUID, as microseconds since
/// the Unix epoch. Version 7 carries whole milliseconds; version 1 carries 100 ns units since
/// 1582-10-15, truncated here to microseconds. `None` for any other version or variant.
pub(super) fn extract_timestamp_micros(uuid: &[u8; 16]) -> Option<i64> {
    if !is_standard_variant(uuid) {
        return None;
    }
    match uuid[6] >> 4 {
        7 => {
            let mut ms = [0u8; 8];
            ms[2..].copy_from_slice(&uuid[..6]);
            i64::from_be_bytes(ms).checked_mul(1000)
        },
        1 => {
            let low = u64::from(u32::from_be_bytes([uuid[0], uuid[1], uuid[2], uuid[3]]));
            let mid = u64::from(u16::from_be_bytes([uuid[4], uuid[5]]));
            let high = u64::from(u16::from_be_bytes([uuid[6], uuid[7]]) & 0x0fff);
            let hundred_nanos = (high << 48) | (mid << 32) | low;
            i64::try_from(hundred_nanos / 10)
                .ok()?
                .checked_sub(GREGORIAN_TO_UNIX_MICROS)
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick_of(uuid: &[u8; 16]) -> u64 {
        let mut ms = [0u8; 8];
        ms[2..].copy_from_slice(&uuid[..6]);
        let fraction = (u64::from(uuid[6] & 0x0f) << 8) | u64::from(uuid[7]);
        (u64::from_be_bytes(ms) << FRACTION_BITS) | fraction
    }

    #[test]
    fn next_tick_takes_the_clock_when_it_moved_forward() {
        assert_eq!(next_tick(100, 250), 250);
    }

    #[test]
    fn next_tick_counts_up_within_one_step() {
        assert_eq!(next_tick(100, 100), 101);
    }

    #[test]
    fn next_tick_counts_up_when_the_clock_moves_backwards() {
        assert_eq!(next_tick(100, 40), 101);
    }

    #[test]
    fn counter_carry_moves_into_the_millisecond_field() {
        let last = (5 << FRACTION_BITS) | 0x0fff;
        let next = next_tick(last, last);
        assert_eq!(next >> FRACTION_BITS, 6);
        assert_eq!(next & 0x0fff, 0);
    }

    #[test]
    fn encode_lays_out_version_variant_and_fields() {
        let tick = (0x0123_4567_89ab << FRACTION_BITS) | 0x0cde;
        let uuid = encode_v7(tick, [0xff; 16]);
        assert_eq!(&uuid[..6], &[0x01, 0x23, 0x45, 0x67, 0x89, 0xab]);
        assert_eq!(uuid[6], 0x7c);
        assert_eq!(uuid[7], 0xde);
        assert_eq!(uuid[8] & 0xc0, 0x80);
        assert_eq!(&uuid[9..], &[0xff; 7]);
        assert_eq!(tick_of(&uuid), tick);
    }

    #[test]
    fn generated_values_have_version_variant_and_current_time() {
        let before = clock_tick() >> FRACTION_BITS;
        let uuid = uuid_v7(None).unwrap();
        let after = clock_tick() >> FRACTION_BITS;
        assert_eq!(uuid[6] >> 4, 7);
        assert_eq!(uuid[8] & 0xc0, 0x80);
        let ms = tick_of(&uuid) >> FRACTION_BITS;
        // The counter may run a little ahead of the clock under load from parallel tests.
        assert!(
            ms >= before && ms <= after + 1000,
            "{before} <= {ms} <= {after}"
        );
    }

    #[test]
    fn ten_thousand_values_strictly_increase() {
        let mut previous = uuid_v7(None).unwrap();
        for _ in 0..10_000 {
            let next = uuid_v7(None).unwrap();
            assert!(next > previous, "{next:02x?} <= {previous:02x?}");
            assert!(tick_of(&next) > tick_of(&previous));
            previous = next;
        }
    }

    #[test]
    fn values_increase_across_threads() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let mut out = Vec::with_capacity(2000);
                    for _ in 0..2000 {
                        out.push(tick_of(&uuid_v7(None).unwrap()));
                    }
                    out
                })
            })
            .collect();
        let mut all = Vec::new();
        for handle in handles {
            let ticks = handle.join().unwrap();
            assert!(ticks.windows(2).all(|w| w[0] < w[1]));
            all.extend(ticks);
        }
        let count = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), count, "a tick was handed out twice");
    }

    #[test]
    fn shift_moves_the_timestamp_by_the_interval() {
        let tick = 1_700_000_000_000 << FRACTION_BITS | 7;
        let hour = Interval {
            months: 0,
            days: 0,
            micros: 3_600_000_000,
        };
        assert_eq!(
            shift_tick(tick, &hour),
            Some(tick + (3_600_000 << FRACTION_BITS))
        );
        let back = Interval {
            months: 0,
            days: -1,
            micros: 0,
        };
        assert_eq!(
            shift_tick(tick, &back),
            Some(tick - (86_400_000 << FRACTION_BITS))
        );
    }

    #[test]
    fn shift_out_of_range_is_refused() {
        let far_back = Interval {
            months: -12 * 100,
            days: 0,
            micros: 0,
        };
        assert_eq!(shift_tick(1 << FRACTION_BITS, &far_back), None);
        let far_ahead = Interval {
            months: 12 * 20_000,
            days: 0,
            micros: 0,
        };
        assert_eq!(shift_tick(1 << FRACTION_BITS, &far_ahead), None);
        assert!(uuid_v7(Some(&far_back)).is_err());
    }

    #[test]
    fn extract_reads_back_version_and_timestamp() {
        let uuid = encode_v7((1_700_000_000_123 << FRACTION_BITS) | 0x0abc, [0; 16]);
        assert_eq!(extract_version(&uuid), Some(7));
        assert_eq!(extract_timestamp_micros(&uuid), Some(1_700_000_000_123_000));
    }

    #[test]
    fn extract_reads_a_version_1_timestamp() {
        // The RFC 9562 version 1 example, C232AB00-9414-11EC-B3C8-9F6BDECED846: 2022-02-22 19:22:22 UTC.
        let uuid = [
            0xc2, 0x32, 0xab, 0x00, 0x94, 0x14, 0x11, 0xec, 0xb3, 0xc8, 0x9f, 0x6b, 0xde, 0xce,
            0xd8, 0x46,
        ];
        assert_eq!(extract_version(&uuid), Some(1));
        assert_eq!(extract_timestamp_micros(&uuid), Some(1_645_557_742_000_000));
    }

    #[test]
    fn extract_returns_none_outside_the_standard_variant_and_versions() {
        let mut v4 = [0u8; 16];
        v4[6] = 0x40;
        v4[8] = 0x80;
        assert_eq!(extract_version(&v4), Some(4));
        assert_eq!(extract_timestamp_micros(&v4), None);
        let mut other_variant = encode_v7(1 << FRACTION_BITS, [0; 16]);
        other_variant[8] = 0xc0;
        assert_eq!(extract_version(&other_variant), None);
        assert_eq!(extract_timestamp_micros(&other_variant), None);
    }
}
