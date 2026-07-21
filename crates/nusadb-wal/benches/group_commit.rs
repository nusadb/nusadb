//! Group-commit throughput benchmark.
//!
//! ```bash
//! cargo bench -p nusadb-wal --bench group_commit
//! ```
//!
//! Both workloads run `THREADS` threads each committing `PER_THREAD` times, where every commit
//! appends a few bytes to a shared file and must make them durable. `individual_fsync` issues one
//! real `fsync` per commit; `group_commit` routes the durability request through [`GroupCommit`],
//! which coalesces concurrent committers into shared `fsync`s. Because `fsync` dominates, the
//! coalesced version issues far fewer syscalls and should win by a wide margin under contention.

#![allow(
    clippy::unwrap_used,
    clippy::significant_drop_tightening,
    missing_docs,
    reason = "benchmark harness, not production code (criterion_group! macro lacks docs)"
)]

use std::fs::File;
use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use criterion::{Criterion, criterion_group, criterion_main};
use nusadb_wal::GroupCommit;

const THREADS: usize = 8;
const PER_THREAD: usize = 16;

/// Append 8 bytes to the file under `log` and fsync it; returns nothing.
fn append_and_fsync(log: &Mutex<File>) {
    let mut f = log.lock().unwrap();
    f.write_all(&[0u8; 8]).unwrap();
    f.sync_data().unwrap();
}

fn fresh_log(dir: &std::path::Path, name: &str) -> Mutex<File> {
    Mutex::new(File::create(dir.join(name)).unwrap())
}

fn bench_group_commit(c: &mut Criterion) {
    let dir = nusadb_test_utils::temp_dir();

    c.bench_function("individual_fsync_8t", |b| {
        let log = fresh_log(dir.path(), "individual.log");
        b.iter(|| {
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let log = &log;
                    s.spawn(move || {
                        for _ in 0..PER_THREAD {
                            append_and_fsync(log);
                        }
                    });
                }
            });
        });
    });

    c.bench_function("group_commit_8t", |b| {
        let log = fresh_log(dir.path(), "group.log");
        let gc = GroupCommit::new();
        let appended = AtomicU64::new(0);
        b.iter(|| {
            thread::scope(|s| {
                for _ in 0..THREADS {
                    let log = &log;
                    let gc = &gc;
                    let appended = &appended;
                    s.spawn(move || {
                        for _ in 0..PER_THREAD {
                            // Append under the log lock (assigns this record's seq).
                            let seq = {
                                let mut f = log.lock().unwrap();
                                f.write_all(&[0u8; 8]).unwrap();
                                appended.fetch_add(1, Ordering::SeqCst) + 1
                            };
                            // Durability is coalesced: only the leader fsyncs, for the whole batch.
                            gc.commit(seq, || {
                                log.lock().unwrap().sync_data()?;
                                Ok::<u64, io::Error>(appended.load(Ordering::SeqCst))
                            })
                            .unwrap();
                        }
                    });
                }
            });
        });
    });
}

criterion_group!(benches, bench_group_commit);
criterion_main!(benches);
