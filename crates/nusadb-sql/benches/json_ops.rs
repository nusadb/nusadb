//! JSON operator cost, measured — the baseline for deciding how JSON should be stored.
//!
//! Today a `JSON` value is stored as canonical **text** and every operator re-parses the whole
//! document ([`json::get_field`] and its 20 siblings all begin with `parse(json)?`). This bench
//! makes the consequence visible, and it is designed so the measurement *distinguishes* the two
//! candidate designs rather than merely reporting a number:
//!
//! - **Fetching the first key costs the same as fetching the last one.** A design that scans (or
//!   binary-searches) would show first-key ≪ last-key. Equal timings mean the whole document is
//!   materialized before either key is looked at.
//! - **Cost grows with the document, not with the query.** Widening a document while fetching a
//!   single field should be free in a decomposed format; here it is not.
//!
//! Both signatures are what a binary codec with a sorted key table would erase, so re-running this
//! after such a change is the evidence gate for it.
//!
//! Run: `cargo bench -p nusadb-sql --bench json_ops`.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    reason = "criterion_group! generates an undocumented `benches` fn; a bench harness panics on setup failure"
)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nusadb_sql::json;
use std::hint::black_box;

/// A flat object of `keys` members, values a mix of the JSON scalar types. Keys are `k000`, `k001`,
/// … so the first and last member are addressable by name and equally long — a length difference
/// would otherwise show up as a comparison-cost difference and muddy the reading.
fn object(keys: usize) -> String {
    use std::fmt::Write as _;
    let mut s = String::from("{");
    for i in 0..keys {
        if i > 0 {
            s.push(',');
        }
        // Rotate through string / number / bool / nested so the parse does representative work
        // instead of one trivially cheap type.
        let _ = match i % 4 {
            0 => write!(s, "\"k{i:03}\":\"value-{i}\""),
            1 => write!(s, "\"k{i:03}\":{i}"),
            2 => write!(s, "\"k{i:03}\":{}", i % 2 == 0),
            _ => write!(s, "\"k{i:03}\":{{\"n\":{i}}}"),
        };
    }
    s.push('}');
    s
}

/// The widths under test: a small config blob, a typical API payload, and a wide record.
const WIDTHS: [usize; 3] = [8, 64, 256];

fn bench_get_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("json/get_field_text");
    for keys in WIDTHS {
        let doc = object(keys);
        let first = "k000".to_owned();
        let last = format!("k{:03}", keys - 1);
        group.throughput(criterion::Throughput::Bytes(doc.len() as u64));
        // If these two diverge, lookup is doing work proportional to key position. If they match,
        // the document is fully materialized before the key is consulted.
        group.bench_with_input(BenchmarkId::new("first_key", keys), &keys, |b, _| {
            b.iter(|| black_box(json::get_field_text(black_box(&doc), black_box(&first))));
        });
        group.bench_with_input(BenchmarkId::new("last_key", keys), &keys, |b, _| {
            b.iter(|| black_box(json::get_field_text(black_box(&doc), black_box(&last))));
        });
        // A key that is not there: the whole document is parsed to conclude "absent".
        group.bench_with_input(BenchmarkId::new("missing_key", keys), &keys, |b, _| {
            b.iter(|| black_box(json::get_field_text(black_box(&doc), black_box("nope"))));
        });
    }
    group.finish();
}

fn bench_predicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("json/predicates");
    for keys in WIDTHS {
        let doc = object(keys);
        let probe = r#"{"k001":1}"#.to_owned();
        group.throughput(criterion::Throughput::Bytes(doc.len() as u64));
        group.bench_with_input(BenchmarkId::new("has_key", keys), &keys, |b, _| {
            b.iter(|| black_box(json::has_key(black_box(&doc), black_box("k001"))));
        });
        group.bench_with_input(BenchmarkId::new("contains", keys), &keys, |b, _| {
            b.iter(|| black_box(json::contains(black_box(&doc), black_box(&probe))));
        });
    }
    group.finish();
}

fn bench_write_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("json/canonicalize");
    for keys in WIDTHS {
        let doc = object(keys);
        group.throughput(criterion::Throughput::Bytes(doc.len() as u64));
        // What every INSERT / cast into a JSON column pays: parse, then re-serialize.
        group.bench_with_input(BenchmarkId::new("insert", keys), &keys, |b, _| {
            b.iter(|| black_box(json::canonicalize(black_box(&doc))));
        });
    }
    group.finish();
}

fn bench_chained(c: &mut Criterion) {
    // `doc->'a'->>'n'` is two operators, so the document is parsed twice and the intermediate is
    // re-serialized to text between them. This is the shape a nested lookup takes in real SQL.
    let mut group = c.benchmark_group("json/chained");
    for keys in WIDTHS {
        let doc = object(keys);
        let nested = format!("k{:03}", (keys - 1) / 4 * 4 + 3); // a member whose value is an object
        group.throughput(criterion::Throughput::Bytes(doc.len() as u64));
        group.bench_with_input(BenchmarkId::new("arrow_then_arrow", keys), &keys, |b, _| {
            b.iter(|| {
                let inner = json::get_field(black_box(&doc), black_box(&nested));
                black_box(inner.and_then(|v| json::get_field_text(&v, "n")))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_get_field,
    bench_predicates,
    bench_write_path,
    bench_chained
);
criterion_main!(benches);
