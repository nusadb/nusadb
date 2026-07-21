//! Wire-latency probe: per-query round-trip percentiles over the Nusa Wire Protocol.
//!
//! Measures the two reference shapes — `SELECT 1` (pure round-trip floor) and a
//! PK point-get — against a running `nusadb-server`, printing p50/p90/p99/min. Run the server
//! and this probe on the SAME host (Linux loopback / one container network namespace) so the
//! wire path itself is what is measured, not a NAT hop:
//!
//! ```text
//! nusadb-server --listen 127.0.0.1:5678 --data-dir /tmp/r1a &
//! cargo run --release -p nusadb-libnusa --example latency_probe -- 127.0.0.1 5678
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::print_stdout,
    reason = "manual measurement probe, not library code"
)]

use std::time::Instant;

use nusadb_libnusa::{Client, Config};

const WARMUP: usize = 500;
const SAMPLES: usize = 5_000;
const POINT_ROWS: i64 = 100_000;

fn percentile(sorted: &[u128], p: f64) -> u128 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted.get(idx).copied().unwrap_or(0)
}

async fn measure(
    client: &mut Client,
    label: &str,
    mut sql_for: impl FnMut(usize) -> String,
) -> Vec<u128> {
    for i in 0..WARMUP {
        client.simple_query(&sql_for(i)).await.unwrap();
    }
    let mut micros = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let sql = sql_for(i);
        let t = Instant::now();
        client.simple_query(&sql).await.unwrap();
        micros.push(t.elapsed().as_micros());
    }
    micros.sort_unstable();
    println!(
        "{label}: p50={}us p90={}us p99={}us min={}us ({SAMPLES} samples)",
        percentile(&micros, 0.50),
        percentile(&micros, 0.90),
        percentile(&micros, 0.99),
        micros.first().copied().unwrap_or(0),
    );
    micros
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut args = std::env::args().skip(1);
    let host = args.next().unwrap_or_else(|| "127.0.0.1".to_owned());
    let port: u16 = args.next().and_then(|p| p.parse().ok()).unwrap_or(5678);
    let config = Config::new(host, port, "nusa-root", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    // Round-trip floor.
    measure(&mut client, "SELECT 1", |_| "SELECT 1".to_owned()).await;

    // PK point-get over a real table (fresh per run; DROP tolerates a previous run's leftover).
    let _ = client.simple_query("DROP TABLE IF EXISTS r1a").await;
    client
        .simple_query("CREATE TABLE r1a (id INT PRIMARY KEY, v INT)")
        .await
        .unwrap();
    for start in (0..POINT_ROWS).step_by(1000) {
        let values: String = (start..start + 1000)
            .map(|i| format!("({i},{})", i % 97))
            .collect::<Vec<_>>()
            .join(",");
        client
            .simple_query(&format!("INSERT INTO r1a VALUES {values}"))
            .await
            .unwrap();
    }
    client.simple_query("ANALYZE r1a").await.unwrap();
    // Deterministic key spread across the table.
    measure(&mut client, "point-get PK", |i| {
        format!(
            "SELECT v FROM r1a WHERE id = {}",
            (i as i64 * 7919) % POINT_ROWS
        )
    })
    .await;
}
