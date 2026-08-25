//! The SQLLogicTest corpus, run through a real connection.
//!
//! `sqllogictest.rs` drives the same files through `Session::execute` in-process. That path never
//! goes near the wire server, and the wire server never goes through `Session` — it calls
//! `execute_in_txn_as_*` directly. So everything `Session::execute` intercepts is structurally
//! invisible to that corpus: `PREPARE`/`EXECUTE`/`DEALLOCATE`, transaction control, savepoints,
//! `SET`/`SHOW`. Command tags are invisible too, because the embedded API returns a typed
//! `ExecutionResult` and never formats one.
//!
//! That is not a hypothesis. Two defects found in one week were wire-only, and one of them —
//! SQL-level `PREPARE` failing outright over a connection — was pinned as *working* in this very
//! corpus and reported as matching the reference engine, because both checks ran the embedded path.
//!
//! This file closes the hole by construction rather than by documenting it: the same `.slt` files,
//! a second `DB` implementation, a real server on a real socket. What it reports when a file
//! disagrees between the two paths is the thing nobody had — a list of what the corpus was hiding.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test: unwrap/panic-on-failure is the assertion mechanism"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::messages::{BackendMessage, FrontendMessage};
use nusadb_wire::server::{ServerConfig, serve_with_shutdown};
use nusadb_wire::{Connection, PROTOCOL_VERSION};
use sqllogictest::{DBOutput, DefaultColumnType, Runner};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;

/// A failure reported by the wire path — the SQLSTATE and message a client would see.
#[derive(Debug)]
struct SltWireError(String);

impl std::fmt::Display for SltWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SltWireError {}

/// A `.slt` connection that speaks the wire protocol to a server of its own.
///
/// One server and one connection per file, matching the in-process runner's "fresh engine per
/// file" rule, so a scenario cannot be perturbed by another.
///
/// That holds for engine state, and not for `LISTEN`/`NOTIFY`: the listener registry is global to
/// the process rather than owned by a server, and every file here starts up as the same database,
/// so files would deliver to each other's listeners. No file uses either statement today. Adding
/// one means giving it a channel or a database no other file uses.
struct WireConnection {
    /// Declared before `runtime` on purpose: a tokio IO resource must not outlive the runtime it
    /// was created on, and struct fields drop in declaration order.
    client: Connection<TcpStream>,
    runtime: Runtime,
}

impl WireConnection {
    fn new() -> Self {
        // `futures::executor::block_on` drives sqllogictest, so this is not a tokio context and a
        // current-thread runtime still drives the spawned server during `Runtime::block_on`.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let client = runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("addr");
            let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
            tokio::spawn(serve_with_shutdown(
                listener,
                engine,
                ServerConfig::default(),
                std::future::pending::<()>(),
            ));
            connect(addr).await
        });
        Self { client, runtime }
    }
}

async fn connect(addr: std::net::SocketAddr) -> Connection<TcpStream> {
    let mut client = Connection::new(TcpStream::connect(addr).await.expect("connect"));
    client
        .write_frame(
            &FrontendMessage::Startup {
                major: PROTOCOL_VERSION.0,
                minor: PROTOCOL_VERSION.1,
                user: "nusadb-root".to_owned(),
                database: "nusadb".to_owned(),
            }
            .encode()
            .expect("encode startup"),
        )
        .await
        .expect("send startup");
    loop {
        let frame = client
            .read_frame()
            .await
            .expect("read")
            .expect("server closed during startup");
        match BackendMessage::decode(&frame).expect("decode") {
            BackendMessage::ReadyForQuery(_) => return client,
            BackendMessage::Error { code, message } => {
                panic!("startup refused: {code}: {message}")
            },
            _ => {},
        }
    }
}

impl sqllogictest::DB for WireConnection {
    type Error = SltWireError;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        // Only a server `Error` frame may become `Self::Error`. Everything else — a dead socket, a
        // half-written frame, a decode failure — panics instead, because `statement error` in the
        // corpus is satisfied by *any* `Err`: laundering a transport failure into one would turn
        // "the connection broke" into a passing assertion, in the one harness whose entire purpose
        // is to prove the connection was used.
        let client = &mut self.client;
        self.runtime.block_on(async move {
            let frame = FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap_or_else(|e| panic!("encoding a query failed: {e}"));
            client
                .write_frame(&frame)
                .await
                .unwrap_or_else(|e| panic!("wire write failed: {e}"));

            // A wire-side hang is a live defect class for a harness built to catch wire-only
            // defects. The deadline has to wrap the read itself: a check before the await never
            // runs, because a pending read parks the runtime and control never returns to it.
            let deadline = std::time::Instant::now() + RESPONSE_DEADLINE;
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut columns = 0usize;
            let mut tags: Vec<u8> = Vec::new();
            let mut affected = 0u64;
            let mut error: Option<SltWireError> = None;
            loop {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                let frame = match tokio::time::timeout(remaining, client.read_frame()).await {
                    Err(tokio::time::error::Elapsed { .. }) => panic!(
                        "no frame within {RESPONSE_DEADLINE:?} — the wire hung rather than answered"
                    ),
                    Ok(Ok(Some(frame))) => frame,
                    Ok(Ok(None)) => {
                        panic!("wire closed mid-query — a transport failure, not a SQL error")
                    },
                    Ok(Err(e)) => panic!("wire read failed: {e}"),
                };
                let decoded = BackendMessage::decode(&frame)
                    .unwrap_or_else(|e| panic!("undecodable frame from the server: {e}"));
                match decoded {
                    BackendMessage::RowDescription { columns: c } => {
                        columns = c.len();
                        tags = vec![0; columns];
                    },
                    BackendMessage::RowDescriptionTyped { columns: c } => {
                        columns = c.len();
                        tags = c.iter().map(|(_, tag)| *tag).collect();
                    },
                    BackendMessage::DataRow { values } => rows.push(
                        values
                            .into_iter()
                            .enumerate()
                            .map(|(i, v)| {
                                // The corpus writes SQL NULL as `NULL`; the wire sends it absent.
                                v.map_or_else(
                                    || "NULL".to_owned(),
                                    |b| {
                                        canonicalize(
                                            &String::from_utf8_lossy(&b),
                                            tags.get(i).copied().unwrap_or(0),
                                        )
                                    },
                                )
                            })
                            .collect(),
                    ),
                    BackendMessage::Error { code, message } => {
                        error = Some(SltWireError(format!("{code}: {message}")));
                    },
                    BackendMessage::CommandComplete { tag } => {
                        // The corpus can assert an affected-row count, and the tag is the only
                        // place the wire carries one. Discarding it would report 0 for every
                        // write while the in-process path reports the real number — the module
                        // doc claims this harness covers command tags, so it has to read them.
                        affected = tag
                            .rsplit(' ')
                            .next()
                            .and_then(|n| n.parse::<u64>().ok())
                            .unwrap_or(0);
                    },
                    BackendMessage::ReadyForQuery(_) => break,
                    _ => {},
                }
            }
            if let Some(err) = error {
                return Err(err);
            }
            if rows.is_empty() && columns == 0 {
                return Ok(DBOutput::StatementComplete(affected));
            }
            Ok(DBOutput::Rows {
                types: vec![DefaultColumnType::Any; columns.max(rows.first().map_or(0, Vec::len))],
                rows,
            })
        })
    }

    fn engine_name(&self) -> &'static str {
        "nusadb-wire"
    }
}

/// Render a wire value the way the corpus writes it.
///
/// The corpus is written in sqllogictest's canonical text, which the in-process runner produces
/// directly from typed values. The wire sends the client-facing rendering instead — `true` where
/// the corpus writes `1`, `1.5` where it writes `1.500` — so the same row disagrees for reasons
/// that have nothing to do with the engine. Converting here keeps the *files* the single source of
/// truth for both paths, which is the point of running them twice.
///
/// The type tag comes from `RowDescriptionTyped`; an untyped description leaves the text alone.
///
/// The tags are derived from the taxonomy rather than transcribed: a hand-copied number that drifts
/// would let a TEXT column inherit the BOOL tag and silently rewrite a genuine `'true'` to `1` —
/// the corruption the `int_range.slt` note cites as the reason not to guess from text.
fn canonicalize(text: &str, type_tag: u8) -> String {
    match type_tag {
        BOOL_TAG => match text {
            "true" => "1".to_owned(),
            "false" => "0".to_owned(),
            other => other.to_owned(),
        },
        FLOAT_TAG => text
            .parse::<f64>()
            .map_or_else(|_| text.to_owned(), |f| format!("{f:.3}")),
        _ => text.to_owned(),
    }
}

/// How long one statement may take before the harness calls it a hang. Named so the panic can
/// report the actual budget: `tokio`'s `Elapsed` is a unit type whose `Display` is the fixed
/// string "deadline has elapsed", which names nothing.
const RESPONSE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Compile-time, from the taxonomy — see [`canonicalize`].
const BOOL_TAG: u8 = nusadb_wire::column_type_tag(nusadb_core::ColumnType::Bool);
/// Compile-time, from the taxonomy — see [`canonicalize`].
const FLOAT_TAG: u8 = nusadb_wire::column_type_tag(nusadb_core::ColumnType::Float);

// === Corpus over the wire ===================================================
//
// The corpus is globbed rather than hand-listed, because a hand-written list of what runs over the
// wire is a list that silently stops being true. A new `.slt` file is picked up here with no
// bookkeeping, and a file that cannot run yet has to say so in [`BLOCKED_LIST`] — where the entry
// is checked from both sides. That is the property worth the loss of one test name per file: the
// set of untested files stops being a comment and becomes something the suite enforces.

/// The corpus root, relative to this crate.
const CORPUS_ROOT: &str = "tests/slt";

/// Files that cannot run over the wire yet, one per line, each with the reason it was measured to
/// need. Checked in both directions: an entry that starts passing fails the suite, so the list can
/// only shrink.
const BLOCKED_LIST: &str = "tests/slt_over_wire_blocked.txt";

/// Every `.slt` file in the corpus, relative to [`CORPUS_ROOT`], in a stable order.
fn corpus_files() -> Vec<String> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("read a corpus directory") {
            let path = entry.expect("read a corpus entry").path();
            if path.is_dir() {
                walk(&path, root, out);
            } else if path.extension().is_some_and(|ext| ext == "slt") {
                let rel = path
                    .strip_prefix(root)
                    .expect("entry is under the corpus root");
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    let root = std::path::Path::new(CORPUS_ROOT);
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// One blocked-list entry: where the file first diverges, and why.
struct Blocked {
    /// The corpus line the divergence was measured at.
    ///
    /// Pinned because `run_file` is fail-fast: a blocked file runs only up to here, and everything
    /// after it has never been over the wire. Without the line, a *new* defect appearing earlier in
    /// the same file would be absorbed by the existing entry and reported as nothing at all — the
    /// list would go on excusing a failure that is no longer the one it describes.
    line: u32,
    reason: String,
}

/// The blocked list, parsed as path → entry. Entries are written `path:line  reason`.
fn blocked_files() -> std::collections::BTreeMap<String, Blocked> {
    let text = std::fs::read_to_string(BLOCKED_LIST).expect("read the blocked list");
    let mut entries = std::collections::BTreeMap::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (located, reason) = line
            .split_once(char::is_whitespace)
            .unwrap_or_else(|| panic!("blocked-list entry carries no reason: {line}"));
        let reason = reason.trim();
        assert!(
            !reason.is_empty(),
            "blocked-list entry carries no reason: {line}"
        );
        let (path, at) = located
            .rsplit_once(':')
            .unwrap_or_else(|| panic!("blocked-list entry names no line: {line}"));
        let at = at
            .parse::<u32>()
            .unwrap_or_else(|e| panic!("blocked-list entry has an unreadable line number: {e}"));
        let previous = entries.insert(
            path.to_owned(),
            Blocked {
                line: at,
                reason: reason.to_owned(),
            },
        );
        // Otherwise the later one wins in silence, and deleting the entry someone is reading does
        // not change what the suite enforces.
        assert!(previous.is_none(), "blocked-list names {path} twice");
    }
    entries
}

/// What running one file produced.
enum Outcome {
    Passed,
    /// The scenario ran and disagreed: an engine refusal, or a result the file did not expect.
    /// This is what the blocked list may excuse — at the line it names, and no other.
    Failed {
        line: Option<u32>,
        detail: String,
    },
    /// The harness itself broke — a transport error, a closed socket, the response deadline.
    ///
    /// Deliberately *not* excusable by the blocked list. Folding a broken connection into the same
    /// bucket as a genuine disagreement is how a harness starts reporting green for scenarios it
    /// never ran, and this file exists because that already happened once.
    Broke(String),
}

/// Run one file, surviving whatever it does. A panic in file 3 must not cost the report on the
/// other ninety.
fn attempt(path: &str) -> Outcome {
    let full = format!("{CORPUS_ROOT}/{path}");
    let ran = std::panic::catch_unwind(|| {
        let mut runner = Runner::new(|| async { Ok::<_, SltWireError>(WireConnection::new()) });
        runner.run_file(&full)
    });
    match ran {
        Ok(Ok(())) => Outcome::Passed,
        Ok(Err(err)) => {
            let detail = one_line(&err.to_string());
            Outcome::Failed {
                line: divergence_line(&detail, path),
                detail,
            }
        },
        Err(payload) => Outcome::Broke(one_line(&panic_message(payload.as_ref()))),
    }
}

/// The corpus line a failure points at, read back out of `sqllogictest`'s own `at <file>:<line>`.
///
/// Recovered from the message rather than tracked separately because the message is what the
/// library reports and what a reader compares against; a second count kept in parallel would be
/// free to disagree with it.
fn divergence_line(detail: &str, path: &str) -> Option<u32> {
    let marker = format!("{CORPUS_ROOT}/{path}:");
    let after = detail.rsplit_once(&marker)?.1;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panicked with a payload that is not a string".to_owned())
}

/// Collapse a failure onto one line so the report stays scannable when several files fail at once.
///
/// Length is left alone. An earlier version capped this at 240 characters on the grounds that the
/// reasons are read in a line-oriented file — but nothing writes this text into that file, and the
/// library puts the location, the SQL and the `-expected|+actual` header ahead of the diff, so the
/// cap threw away the diff and kept the preamble. Losing the per-file test names is only paid for
/// if what replaces them says more, not less.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Run the whole corpus over a real connection and hold it to [`BLOCKED_LIST`].
///
/// Reports every disagreement rather than the first, because the first tells you nothing about
/// whether the change you just made moved one file or forty.
///
/// Set `SLT_WIRE_ONLY` to a path fragment to run just the files matching it — the one thing the
/// per-file `#[test]`s this replaced were good for, in the harness whose whole purpose is
/// diagnosing a wire-only defect. It narrows what runs, never what is enforced: any file it skips
/// keeps its blocked-list entry untouched rather than being reported as stale.
#[test]
fn every_corpus_file_runs_over_the_wire() {
    let files = corpus_files();
    // This floor stays, where the one in `the_in_process_runner_names_every_corpus_file` was
    // removed as decoration. There, the set difference against an independently-maintained list
    // catches a short walk exactly; here there is no such list, and the blocked entries only touch
    // five of the fifteen corpus directories — so a walk that stopped reaching, say, `p8_cte`
    // would still be well above ninety and pass in silence.
    assert!(
        files.len() >= 90,
        "found only {} corpus files — the walk is not reaching them",
        files.len()
    );

    let only = std::env::var("SLT_WIRE_ONLY").ok();
    let selected: Vec<&String> = files
        .iter()
        .filter(|file| {
            only.as_ref()
                .is_none_or(|frag| file.contains(frag.as_str()))
        })
        .collect();
    assert!(
        !selected.is_empty(),
        "SLT_WIRE_ONLY={} matches no corpus file",
        only.unwrap_or_default()
    );

    let mut blocked = blocked_files();
    let mut complaints = Vec::new();

    for file in &selected {
        // Removed as we go, so whatever is left over is an entry naming a file that no longer
        // exists — a rename would otherwise leave the list quietly excusing nothing.
        let excuse = blocked.remove(*file);
        match (attempt(file), excuse) {
            (Outcome::Passed, Some(entry)) => complaints.push(format!(
                "{file}: passes over the wire now — delete its line from {BLOCKED_LIST} (it says: {})",
                entry.reason
            )),
            (Outcome::Failed { line, detail }, Some(entry)) if line != Some(entry.line) => {
                complaints.push(format!(
                    "{file}: blocked at line {}, but now diverges at {} — the entry no longer \
                     describes what happens, and everything between the two is unrun: {detail}",
                    entry.line,
                    line.map_or_else(|| "an unreported line".to_owned(), |at| at.to_string())
                ));
            },
            // The two silent outcomes: a file that passes and is not listed, and one that fails at
            // exactly the line its entry names. This has to sit below the guarded arm above, which
            // is the one that decides whether a listed failure is still the listed failure.
            (Outcome::Passed, None) | (Outcome::Failed { .. }, Some(_)) => {},
            (Outcome::Failed { detail, .. }, None) => {
                complaints.push(format!("{file}: fails over the wire: {detail}"));
            },
            (Outcome::Broke(err), _) => complaints.push(format!(
                "{file}: the harness broke, which no blocked-list entry excuses: {err}"
            )),
        }
    }

    // Only meaningful over a full run: a filtered run never visits the files it skipped, so their
    // entries are still there and are not stale.
    //
    // Gated on what actually ran, not on whether a filter was configured. `SLT_WIRE_ONLY=` — an
    // unset CI variable expanded into the command — parses as `Some("")`, which `contains` matches
    // against every file: the whole corpus runs, the report says "of 93", and the sweep would have
    // been skipped anyway. A run that looks complete while enforcing one of the two directions is
    // the failure this file exists to refuse.
    if selected.len() == files.len() {
        for (stale, entry) in blocked {
            complaints.push(format!(
                "{stale}: listed in {BLOCKED_LIST} but no such corpus file (it says: {})",
                entry.reason
            ));
        }
    }

    assert!(
        complaints.is_empty(),
        "{} of {} corpus files disagree with {BLOCKED_LIST}:\n{}",
        complaints.len(),
        selected.len(),
        complaints.join("\n")
    );
}

/// No corpus file may skip records conditionally, because the two runners answer to different
/// names.
///
/// `sqllogictest` dispatches `skipif`/`onlyif` on `DB::engine_name`, which is `nusadb` in-process
/// and `nusadb-wire` here, and `halt` stops a file early on both. Any of the three would make a
/// record run on one path and not the other — a corpus that silently stops covering what it looks
/// like it covers, which is the defect this whole file exists to remove, one layer further down.
///
/// `include` is here for a different reason: it makes the failure location print as the
/// *including* line, which is the one thing [`divergence_line`] reads, so a blocked entry would
/// pin a line that is not where the divergence is. Barring it keeps the pins meaning what the
/// blocked list says they mean.
///
/// None are used today; this keeps it that way rather than trusting that nobody reaches for them.
#[test]
fn no_corpus_file_hides_records_from_one_runner() {
    let offenders: Vec<String> = corpus_files()
        .into_iter()
        .filter_map(|file| {
            let text = std::fs::read_to_string(format!("{CORPUS_ROOT}/{file}"))
                .expect("read a corpus file");
            let found: Vec<&str> = ["skipif", "onlyif", "halt", "include"]
                .into_iter()
                .filter(|directive| {
                    text.lines()
                        .any(|line| line.split_whitespace().next() == Some(*directive))
                })
                .collect();
            (!found.is_empty()).then(|| format!("{file}: {}", found.join(", ")))
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "these files would run differently on the two paths — give the wire runner the same \
         `engine_name` first, or the corpus stops meaning the same thing on each:\n{}",
        offenders.join("\n")
    );
}

/// The in-process runner still names its files by hand, so hold that list to the corpus too.
///
/// It is complete today, and nothing was keeping it that way: a `.slt` file added without its
/// `#[test]` would sit in the corpus untested, and the only signal would be a test count nobody
/// reads. The wire runner cannot regress like that because it globs — this gives the other path
/// the same guarantee without restructuring it.
#[test]
fn the_in_process_runner_names_every_corpus_file() {
    const IN_PROCESS_RUNNER: &str = "tests/sqllogictest.rs";

    let source = std::fs::read_to_string(IN_PROCESS_RUNNER).expect("read the in-process runner");
    let listed: std::collections::BTreeSet<String> = source
        .match_indices("run_slt(\"")
        .filter_map(|(at, marker)| {
            let rest = &source[at + marker.len()..];
            rest.find('"').map(|end| rest[..end].to_owned())
        })
        .map(|path| path.trim_start_matches("tests/slt/").to_owned())
        .collect();

    // No count floor here on purpose: a scan that stopped matching would leave `listed` empty and
    // every corpus file would surface in `unlisted` below, which says the same thing exactly. A
    // floor would only look like it was guarding something.
    let corpus: std::collections::BTreeSet<String> = corpus_files().into_iter().collect();
    let unlisted: Vec<&String> = corpus.difference(&listed).collect();
    let missing: Vec<&String> = listed.difference(&corpus).collect();

    assert!(
        unlisted.is_empty(),
        "in the corpus but never run in-process — add a `#[test]` to {IN_PROCESS_RUNNER}: {unlisted:?}"
    );
    assert!(
        missing.is_empty(),
        "run in-process but no such corpus file — a rename left {IN_PROCESS_RUNNER} behind: {missing:?}"
    );
}

/// Pin the wire's own rendering, so canonicalisation cannot quietly absorb a change to it.
///
/// `canonicalize` bridges the wire's text to the corpus's convention, which is what keeps the
/// `.slt` files the single source of truth for both paths. The cost of that bridge is a blind spot
/// exactly one line of wire code wide: if `value_to_field` started formatting floats as `{:.2}`,
/// the value would round-trip through `parse::<f64>()` to a different number that still re-renders
/// as `1.500` whenever the third decimal is zero, and every file would stay green.
///
/// This asserts the raw bytes instead, closing that residual.
#[test]
fn the_wire_renders_values_the_way_the_canonicaliser_assumes() {
    let mut conn = WireConnection::new();
    // A real FLOAT column, not a decimal literal: `1.5` written bare is NUMERIC, which neither
    // renderer reformats. Writing this test against the literal asserted nothing about the float
    // path — it is the first thing this test caught, and the reason it exists.
    sqllogictest::DB::run(&mut conn, "CREATE TABLE r (b BOOL, f FLOAT, s TEXT)").expect("create");
    sqllogictest::DB::run(&mut conn, "INSERT INTO r VALUES (true, 1.5, '')").expect("insert");
    let out = sqllogictest::DB::run(&mut conn, "SELECT b, NOT b, f, s FROM r").expect("select");
    let DBOutput::Rows { rows, .. } = out else {
        panic!("expected rows from a constant projection");
    };

    // Canonicalised, because that is what the corpus compares against.
    // The first three are the corpus's canonical form. The fourth is not: the corpus writes an
    // empty TEXT value as `(empty)`, which is the in-process runner's own convention and the
    // reason `string_extra.slt` is on the blocked list above. Asserting `""` here records the
    // divergence rather than implying the corpus expects it.
    assert_eq!(
        rows,
        vec![vec!["1", "0", "1.500", ""]],
        "canonical form for bool and float; empty TEXT is the known divergence"
    );

    // And the raw form the canonicaliser is written against. Asserted separately so a change to
    // either side is a failure here rather than a silent agreement somewhere else.
    let raw = raw_row(&mut conn, "SELECT b, NOT b, f, s FROM r");
    assert_eq!(
        raw,
        vec![
            Some("true".to_owned()),
            Some("false".to_owned()),
            Some("1.5".to_owned()),
            Some(String::new())
        ],
        "the wire's own rendering changed; `canonicalize` is written against it"
    );

    // Last, because it adds a row and the pin above reads the last one. The command tag carries
    // the only affected-row count the wire sends, and nothing in the corpus consumes one today —
    // so without this the parse would run unchecked, which is the shape this file exists to
    // refuse. Demonstrated by output, not claimed in a comment.
    let inserted =
        sqllogictest::DB::run(&mut conn, "INSERT INTO r VALUES (false, 2.5, 'x')").expect("insert");
    assert!(
        matches!(inserted, DBOutput::StatementComplete(1)),
        "the affected-row count must come from the command tag"
    );
}

/// The *last* row of a result, exactly as the wire sends it, with no canonicalisation.
/// Used for single-row pins, where last and only coincide.
fn raw_row(conn: &mut WireConnection, sql: &str) -> Vec<Option<String>> {
    let client = &mut conn.client;
    conn.runtime.block_on(async move {
        let frame = FrontendMessage::Query {
            sql: sql.to_owned(),
        }
        .encode()
        .expect("encode");
        client.write_frame(&frame).await.expect("write");
        let mut out = Vec::new();
        loop {
            let frame = client
                .read_frame()
                .await
                .expect("read")
                .expect("server closed mid-query");
            match BackendMessage::decode(&frame).expect("decode") {
                BackendMessage::DataRow { values } => {
                    out = values
                        .into_iter()
                        .map(|v| v.map(|b| String::from_utf8_lossy(&b).into_owned()))
                        .collect();
                },
                BackendMessage::ReadyForQuery(_) => break,
                _ => {},
            }
        }
        out
    })
}
