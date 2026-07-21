//! Group-commit coordination: concurrent committers share `fsync`s, durability is
//! monotonic, and a failing flush is propagated without wedging the coordinator.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test harness asserts via unwrap"
)]

use std::io;
use std::sync::Barrier;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use nusadb_wal::GroupCommit;

#[test]
fn a_single_commit_flushes_once_and_is_idempotent() {
    let gc = GroupCommit::new();
    let fsyncs = AtomicUsize::new(0);
    let flush = || {
        fsyncs.fetch_add(1, Ordering::SeqCst);
        Ok(1)
    };

    gc.commit(1, flush).unwrap();
    assert_eq!(fsyncs.load(Ordering::SeqCst), 1);
    assert_eq!(gc.durable_seq(), 1);

    // Already durable → no flush.
    gc.commit(1, || {
        fsyncs.fetch_add(1, Ordering::SeqCst);
        Ok(1)
    })
    .unwrap();
    assert_eq!(
        fsyncs.load(Ordering::SeqCst),
        1,
        "a covered commit must not flush again"
    );
}

#[test]
fn concurrent_commits_coalesce_into_fewer_fsyncs() {
    const THREADS: usize = 16;

    let gc = GroupCommit::new();
    let appended = AtomicU64::new(0); // monotonic "last appended LSN"
    let fsyncs = AtomicUsize::new(0);
    let barrier = Barrier::new(THREADS);

    thread::scope(|s| {
        for _ in 0..THREADS {
            let gc = &gc;
            let appended = &appended;
            let fsyncs = &fsyncs;
            let barrier = &barrier;
            s.spawn(move || {
                barrier.wait();
                // "Append" our record, taking the next LSN.
                let my_seq = appended.fetch_add(1, Ordering::SeqCst) + 1;
                gc.commit(my_seq, || {
                    fsyncs.fetch_add(1, Ordering::SeqCst);
                    // The leader's fsync makes *everything appended so far* durable. A small delay
                    // widens the batching window so followers reliably accumulate behind it.
                    thread::sleep(Duration::from_millis(2));
                    Ok::<u64, io::Error>(appended.load(Ordering::SeqCst))
                })
                .unwrap();
            });
        }
    });

    let count = fsyncs.load(Ordering::SeqCst);
    assert!(count >= 1, "at least one fsync must run");
    assert!(
        count < THREADS,
        "fsyncs ({count}) must coalesce below the {THREADS} commits"
    );
    assert_eq!(
        gc.durable_seq(),
        THREADS as u64,
        "every commit must end durable"
    );
}

#[test]
fn a_flush_error_is_propagated_and_the_coordinator_stays_usable() {
    let gc = GroupCommit::new();

    let err = gc.commit(1, || Err::<u64, io::Error>(io::Error::other("disk full")));
    assert!(
        err.is_err(),
        "the flush error must surface to the committer"
    );
    assert_eq!(gc.durable_seq(), 0, "a failed flush advances nothing");

    // The coordinator is not wedged: a later successful commit works and advances durability.
    gc.commit(1, || Ok(1)).unwrap();
    assert_eq!(gc.durable_seq(), 1);
}
