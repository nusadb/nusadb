//! Sketch-based column statistics for `ANALYZE` (technique from
//! the design`Research_Result/nusadb_cardinality_estimation_sketches.md`).
//!
//! Per the stats treaty (ADR 003) the SQL layer *computes* per-column
//! statistics and hands the engine opaque bytes; the engine only stores them
//! and supplies the authoritative [`row_count`](nusadb_core::StorageEngine::row_count).
//! This module fills [`ColumnStats`] using three bounded-memory estimators so
//! the work scales to large columns without an `O(n)`-memory pass:
//!
//! - **NDV** (distinct count) — [`Hll`], a HyperLogLog with a deterministic
//!   hash (Flajolet 2007). ~1.6 % standard error at `2^12` registers.
//! - **MCV** (most-common values) — [`SpaceSaving`], a bounded top-k counter
//!   (Metwally 2005). Heavy hitters that fit in the counter table get exact
//!   counts.
//! - **Histogram** — equi-depth quantile boundaries over the sorted column.
//!
//! Every estimator is **deterministic**: the hash is fixed (no random seed) and
//! the histogram derives from a stable sort, so the same column values in the
//! same scan order always yield byte-identical [`ColumnStats`]. That keeps plans
//! reproducible across runs and under deterministic simulation (risk #1).
//!
//! `min`/`max`/`most_common`/`histogram` values are encoded with the same
//! single-column [`row`](super::row) codec the engine never interprets, so a
//! later cost model (``) decodes them with `row::decode(bytes, &[ty])`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "HLL estimation crosses integer<->float by design; values are bounded by register counts"
)]
#![allow(
    clippy::doc_markdown,
    reason = "algorithm names (HyperLogLog, SplitMix64) and 2^P notation read better unbackticked"
)]

use std::cmp::Ordering;
use std::collections::HashMap;

use nusadb_core::{ColumnStats, ColumnType};

use super::row;
use crate::ast;
use crate::error::Error;

/// Encoded `(min, max)` pair, each `None` when the column has no non-null value.
type MinMax = (Option<Vec<u8>>, Option<Vec<u8>>);

/// HyperLogLog register precision: `2^P` registers (4096 → ~1.6 % error).
const HLL_PRECISION: u32 = 12;
/// Number of HyperLogLog registers.
const HLL_REGISTERS: usize = 1 << HLL_PRECISION;
/// Bounded counter slots for the Space-Saving MCV sketch.
const MCV_CAPACITY: usize = 256;
/// Most-common values retained in the final [`ColumnStats`].
const MCV_KEEP: usize = 32;
/// Equi-depth histogram bucket target (boundaries = buckets + 1).
const HISTOGRAM_BUCKETS: usize = 64;

/// Compute statistics for one column from all of its values (including
/// `NULL`s). `ty` is the column's physical type, used to encode the opaque
/// `min`/`max`/`MCV`/`histogram` bytes.
pub(super) fn column_stats(
    name: &str,
    values: &[ast::Value],
    ty: ColumnType,
) -> Result<ColumnStats, Error> {
    let mut hll = Hll::new();
    let mut mcv = SpaceSaving::new(MCV_CAPACITY);
    let mut null_count: u64 = 0;
    let mut non_null: Vec<&ast::Value> = Vec::new();

    for value in values {
        if matches!(value, ast::Value::Null) {
            null_count += 1;
            continue;
        }
        let key = row::encode(std::slice::from_ref(value), std::slice::from_ref(&ty))?;
        hll.add(&key);
        mcv.offer(key);
        non_null.push(value);
    }

    let (min, max) = min_max(&non_null, ty)?;
    let histogram = equi_depth_histogram(&mut non_null, ty)?;
    let most_common = mcv.top_k(MCV_KEEP);

    Ok(ColumnStats {
        column: name.to_owned(),
        null_count,
        distinct_count: hll.estimate(),
        min,
        max,
        most_common,
        histogram,
    })
}

/// Encoded `(min, max)` of the non-null values, or `(None, None)` when the
/// column holds no non-null value.
fn min_max(values: &[&ast::Value], ty: ColumnType) -> Result<MinMax, Error> {
    let Some(first) = values.first() else {
        return Ok((None, None));
    };
    let mut lo: &ast::Value = first;
    let mut hi: &ast::Value = first;
    for &v in values.iter().skip(1) {
        if value_cmp(v, lo) == Ordering::Less {
            lo = v;
        }
        if value_cmp(v, hi) == Ordering::Greater {
            hi = v;
        }
    }
    Ok((Some(encode_one(lo, ty)?), Some(encode_one(hi, ty)?)))
}

/// Equi-depth histogram boundaries: sort the non-null values, then sample
/// `HISTOGRAM_BUCKETS + 1` ascending quantile boundaries (min … max). Returns an
/// empty vector when there are no non-null values.
fn equi_depth_histogram(values: &mut [&ast::Value], ty: ColumnType) -> Result<Vec<Vec<u8>>, Error> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    values.sort_by(|a, b| value_cmp(a, b));
    let n = values.len();
    let buckets = HISTOGRAM_BUCKETS.min(n);
    let mut boundaries = Vec::with_capacity(buckets + 1);
    let mut last: Option<usize> = None;
    for i in 0..=buckets {
        // Quantile index: i/buckets of the way through the sorted values.
        let pos = (i * (n - 1)) / buckets;
        if last == Some(pos) {
            continue; // Skip duplicate boundaries from a short/low-NDV column.
        }
        last = Some(pos);
        let value = values.get(pos).ok_or_else(|| internal("histogram index"))?;
        boundaries.push(encode_one(value, ty)?);
    }
    Ok(boundaries)
}

/// Encode a single value with the one-column tuple codec.
fn encode_one(value: &ast::Value, ty: ColumnType) -> Result<Vec<u8>, Error> {
    row::encode(std::slice::from_ref(value), std::slice::from_ref(&ty))
}

fn internal(what: &str) -> Error {
    Error::Unsupported(format!("internal: {what} out of bounds"))
}

/// Total order over non-null values of one column. All values in a column share
/// a type, so the cross-type arm is never reached in practice; it falls back to
/// a stable byte-independent ordering so a malformed mix cannot panic.
///
/// Shared with the cost estimator (`super::cost`) so selectivity comparisons use
/// the exact ordering the histogram was built with.
pub(super) fn value_cmp(a: &ast::Value, b: &ast::Value) -> Ordering {
    use ast::Value::{
        Bool, Date, Float, Int, Json, Numeric, Text, Time, Timestamp, TimestampTz, Uuid,
    };
    match (a, b) {
        (Bool(x), Bool(y)) => x.cmp(y),
        // `Int` + the i64-backed temporal types compare by their backing integer.
        (Int(x), Int(y))
        | (Time(x), Time(y))
        | (Timestamp(x), Timestamp(y))
        | (TimestampTz(x), TimestampTz(y)) => x.cmp(y),
        (Float(x), Float(y)) => x.total_cmp(y),
        (Text(x), Text(y)) | (Json(x), Json(y)) => x.cmp(y),
        (Date(x), Date(y)) => x.cmp(y),
        (Uuid(x), Uuid(y)) => x.cmp(y),
        (ast::Value::Bytes(x), ast::Value::Bytes(y)) => x.cmp(y),
        (Numeric(x), Numeric(y)) => x.compare(y),
        // Mixed/`NULL` (callers exclude `NULL`): order by a stable tag.
        _ => type_rank(a).cmp(&type_rank(b)),
    }
}

const fn type_rank(v: &ast::Value) -> u8 {
    match v {
        ast::Value::Null => 0,
        ast::Value::Bool(_) => 1,
        ast::Value::Int(_) => 2,
        ast::Value::Float(_) => 3,
        ast::Value::Text(_) => 4,
        ast::Value::Date(_) => 5,
        ast::Value::Time(_) => 6,
        ast::Value::Timestamp(_) => 7,
        ast::Value::TimestampTz(_) => 8,
        ast::Value::Uuid(_) => 9,
        ast::Value::Numeric(_) => 10,
        ast::Value::Json(_) => 11,
        ast::Value::Interval(_) => 12,
        ast::Value::Array(_) => 13,
        ast::Value::TimeTz(_) => 14,
        ast::Value::Vector(_) => 15,
        ast::Value::Bytes(_) => 16,
    }
}

// === HyperLogLog ==========================================================

/// HyperLogLog distinct-count estimator with a fixed (seedless) hash, so the
/// estimate is reproducible for a given multiset.
struct Hll {
    registers: Vec<u8>,
}

impl Hll {
    fn new() -> Self {
        Self {
            registers: vec![0; HLL_REGISTERS],
        }
    }

    fn add(&mut self, bytes: &[u8]) {
        let h = hash64(bytes);
        let idx = (h & (HLL_REGISTERS as u64 - 1)) as usize;
        // Remaining bits (top `P` bits are zero after the shift); the register
        // records the rank = position of the leftmost set bit + 1.
        let w = h >> HLL_PRECISION;
        let rank = (w.leading_zeros() - HLL_PRECISION + 1) as u8;
        if let Some(reg) = self.registers.get_mut(idx) {
            *reg = (*reg).max(rank);
        }
    }

    /// Standard HLL estimate with small-range linear counting.
    fn estimate(&self) -> u64 {
        let m = HLL_REGISTERS as f64;
        let mut sum = 0.0_f64;
        let mut zeros = 0_u32;
        for &reg in &self.registers {
            sum += 2.0_f64.powi(-i32::from(reg));
            if reg == 0 {
                zeros += 1;
            }
        }
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let raw = alpha * m * m / sum;
        let estimate = if raw <= 2.5 * m && zeros > 0 {
            // Linear counting is more accurate when many registers are empty.
            m * (m / f64::from(zeros)).ln()
        } else {
            raw
        };
        // Round to nearest; never report fewer than the registers touched imply.
        estimate.round().max(0.0) as u64
    }
}

/// Deterministic 64-bit hash: FNV-1a folded through a SplitMix64 finalizer for
/// good bit dispersion (HLL needs well-mixed high bits).
fn hash64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64; // FNV offset basis
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    // SplitMix64 finalizer.
    let mut z = h.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// === Space-Saving (MCV) ===================================================

/// Bounded top-k frequency counter. Heavy hitters that stay resident keep exact
/// counts; cold keys are evicted, so memory is `O(capacity)` regardless of NDV.
struct SpaceSaving {
    capacity: usize,
    counts: HashMap<Vec<u8>, u64>,
}

impl SpaceSaving {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            counts: HashMap::new(),
        }
    }

    fn offer(&mut self, key: Vec<u8>) {
        if let Some(count) = self.counts.get_mut(&key) {
            *count += 1;
            return;
        }
        if self.counts.len() < self.capacity {
            self.counts.insert(key, 1);
            return;
        }
        // Evict the current minimum and inherit its count + 1 (Space-Saving).
        if let Some((victim, min)) = self
            .counts
            .iter()
            .min_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
            .map(|kv| (kv.0.clone(), *kv.1))
        {
            self.counts.remove(&victim);
            self.counts.insert(key, min + 1);
        }
    }

    /// The `k` most frequent keys with count `> 1`, descending by count then key
    /// (the tie-break keeps the result deterministic).
    fn top_k(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        let mut items: Vec<(Vec<u8>, u64)> = self
            .counts
            .iter()
            .filter(|kv| *kv.1 > 1)
            .map(|kv| (kv.0.clone(), *kv.1))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        items.truncate(k);
        items
    }
}

#[cfg(test)]
mod tests {
    use super::{Hll, SpaceSaving, column_stats, value_cmp};
    use crate::ast::Value;
    use crate::executor::row;
    use nusadb_core::ColumnType;
    use std::cmp::Ordering;

    fn ints(xs: &[i64]) -> Vec<Value> {
        xs.iter().map(|&x| Value::Int(x)).collect()
    }

    #[test]
    fn hll_estimate_is_accurate_vs_exact() {
        let mut hll = Hll::new();
        let exact = 5000_u64;
        for i in 0..exact {
            hll.add(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let err = (est as f64 - exact as f64).abs() / exact as f64;
        assert!(
            err < 0.05,
            "HLL error {err} too high (est={est}, exact={exact})"
        );
    }

    #[test]
    fn hll_is_deterministic() {
        let build = || {
            let mut h = Hll::new();
            for i in 0..1000_u64 {
                h.add(&i.to_le_bytes());
            }
            h.estimate()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn hll_small_cardinality_is_near_exact() {
        let mut hll = Hll::new();
        for i in 0..10_u64 {
            hll.add(&i.to_le_bytes());
        }
        assert_eq!(hll.estimate(), 10);
    }

    #[test]
    fn space_saving_keeps_heavy_hitters_exact() {
        let mut ss = SpaceSaving::new(8);
        // Heavy hitter "a" x100, plus many singletons that churn the table.
        for _ in 0..100 {
            ss.offer(b"a".to_vec());
        }
        for i in 0..50_u32 {
            ss.offer(i.to_le_bytes().to_vec());
        }
        let top = ss.top_k(4);
        assert_eq!(
            top.first().map(|(k, c)| (k.clone(), *c)),
            Some((b"a".to_vec(), 100))
        );
    }

    #[test]
    fn column_stats_counts_nulls_and_ndv() {
        let mut values = ints(&[1, 2, 2, 3, 3, 3]);
        values.push(Value::Null);
        values.push(Value::Null);
        let stats = column_stats("c", &values, ColumnType::Int).unwrap();
        assert_eq!(stats.null_count, 2);
        assert_eq!(stats.distinct_count, 3); // {1,2,3}
    }

    #[test]
    fn column_stats_min_max_roundtrip() {
        let values = ints(&[7, 3, 9, 1, 5]);
        let stats = column_stats("c", &values, ColumnType::Int).unwrap();
        let min = row::decode(&stats.min.unwrap(), &[ColumnType::Int]).unwrap();
        let max = row::decode(&stats.max.unwrap(), &[ColumnType::Int]).unwrap();
        assert_eq!(min, vec![Value::Int(1)]);
        assert_eq!(max, vec![Value::Int(9)]);
    }

    #[test]
    fn column_stats_histogram_is_ascending_and_decodable() {
        let values = ints(&(0..1000).collect::<Vec<_>>());
        let stats = column_stats("c", &values, ColumnType::Int).unwrap();
        assert!(stats.histogram.len() >= 2);
        let decoded: Vec<i64> = stats
            .histogram
            .iter()
            .map(
                |b| match row::decode(b, &[ColumnType::Int]).unwrap().remove(0) {
                    Value::Int(i) => i,
                    other => panic!("expected Int, got {other:?}"),
                },
            )
            .collect();
        assert!(
            decoded.windows(2).all(|w| w[0] <= w[1]),
            "not ascending: {decoded:?}"
        );
        assert_eq!(decoded.first(), Some(&0));
        assert_eq!(decoded.last(), Some(&999));
    }

    #[test]
    fn column_stats_is_deterministic() {
        let values = ints(&[5, 1, 5, 9, 1, 5, 3]);
        let a = column_stats("c", &values, ColumnType::Int).unwrap();
        let b = column_stats("c", &values, ColumnType::Int).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn all_null_column_has_no_min_max() {
        let values = vec![Value::Null, Value::Null];
        let stats = column_stats("c", &values, ColumnType::Int).unwrap();
        assert_eq!(stats.null_count, 2);
        assert_eq!(stats.distinct_count, 0);
        assert!(stats.min.is_none() && stats.max.is_none());
        assert!(stats.histogram.is_empty());
    }

    #[test]
    fn value_cmp_orders_within_type() {
        assert_eq!(value_cmp(&Value::Int(1), &Value::Int(2)), Ordering::Less);
        assert_eq!(
            value_cmp(&Value::Text("a".into()), &Value::Text("b".into())),
            Ordering::Less,
        );
    }
}
