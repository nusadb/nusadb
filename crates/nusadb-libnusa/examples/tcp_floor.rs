//! Companion probe: the raw TCP round-trip floor on this host — std (blocking) and tokio
//! (the stack the wire layer rides) — so the protocol's own share of a query round trip can be
//! separated from the kernel/runtime floor. 16-byte ping-pong, p50/p99 over 5000 samples.
//!
//! `cargo run --release -p nusadb-libnusa --example tcp_floor`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::print_stdout,
    reason = "manual measurement probe, not library code"
)]

use std::io::{Read, Write};
use std::time::Instant;

const SAMPLES: usize = 5_000;

fn percentile(sorted: &[u128], p: f64) -> u128 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted.get(idx).copied().unwrap_or(0)
}

fn report(label: &str, mut micros: Vec<u128>) {
    micros.sort_unstable();
    println!(
        "{label}: p50={}us p99={}us min={}us",
        percentile(&micros, 0.50),
        percentile(&micros, 0.99),
        micros.first().copied().unwrap_or(0),
    );
}

fn std_floor() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_nodelay(true).unwrap();
        let mut buf = [0u8; 16];
        while s.read_exact(&mut buf).is_ok() {
            if s.write_all(&buf).is_err() {
                break;
            }
        }
    });
    let mut c = std::net::TcpStream::connect(addr).unwrap();
    c.set_nodelay(true).unwrap();
    let buf = [7u8; 16];
    let mut back = [0u8; 16];
    let mut micros = Vec::with_capacity(SAMPLES);
    for _ in 0..500 {
        c.write_all(&buf).unwrap();
        c.read_exact(&mut back).unwrap();
    }
    for _ in 0..SAMPLES {
        let t = Instant::now();
        c.write_all(&buf).unwrap();
        c.read_exact(&mut back).unwrap();
        micros.push(t.elapsed().as_micros());
    }
    report("std blocking echo", micros);
}

async fn tokio_floor() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.unwrap();
        s.set_nodelay(true).unwrap();
        let mut buf = [0u8; 16];
        while s.read_exact(&mut buf).await.is_ok() {
            if s.write_all(&buf).await.is_err() {
                break;
            }
        }
    });
    let mut c = tokio::net::TcpStream::connect(addr).await.unwrap();
    c.set_nodelay(true).unwrap();
    let buf = [7u8; 16];
    let mut back = [0u8; 16];
    let mut micros = Vec::with_capacity(SAMPLES);
    for _ in 0..500 {
        c.write_all(&buf).await.unwrap();
        c.read_exact(&mut back).await.unwrap();
    }
    for _ in 0..SAMPLES {
        let t = Instant::now();
        c.write_all(&buf).await.unwrap();
        c.read_exact(&mut back).await.unwrap();
        micros.push(t.elapsed().as_micros());
    }
    report("tokio echo (multi-thread rt)", micros);
}

async fn spawn_blocking_floor() {
    let mut micros = Vec::with_capacity(SAMPLES);
    for _ in 0..500 {
        tokio::task::spawn_blocking(|| std::hint::black_box(1))
            .await
            .unwrap();
    }
    for _ in 0..SAMPLES {
        let t = Instant::now();
        tokio::task::spawn_blocking(|| std::hint::black_box(1))
            .await
            .unwrap();
        micros.push(t.elapsed().as_micros());
    }
    report("spawn_blocking round trip", micros);
}

fn main() {
    std_floor();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tokio_floor().await;
            spawn_blocking_floor().await;
        });
}
