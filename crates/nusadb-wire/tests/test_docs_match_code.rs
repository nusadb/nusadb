//! The protocol document must describe the protocol this build speaks.
//!
//! `docs/wire-protocol.md` is what a driver author reads to decide what to branch on, and both of
//! its enumerations are hand-maintained lists next to code that grows. They have drifted twice:
//! §14 was missing two SQLSTATE classes the server had been emitting since an earlier change, and
//! §9.1 was missing thirty-three command tags. Neither drift was visible to any test, because
//! nothing read the document.
//!
//! These checks are deliberately **two-way**. A one-way check (every documented item exists in the
//! code) is the one that feels natural to write and cannot report the direction that actually
//! misleads a reader: an item the code emits and the document omits.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test: unwrap/panic-on-failure is the assertion mechanism"
)]

use std::collections::BTreeSet;

/// The protocol document, read from the repository rather than embedded, so an edit to either side
/// is what these tests compare.
fn protocol_doc() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/wire-protocol.md");
    std::fs::read_to_string(path).expect("docs/wire-protocol.md is part of the repository")
}

fn wire_server_source() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/server.rs");
    std::fs::read_to_string(path).expect("the wire server's own source")
}

/// Every crate's production source, concatenated. SQLSTATE literals are not confined to one crate:
/// the SQL layer maps most of them, the wire server adds protocol-level ones, and the storage spine
/// raises the serialization and deadlock codes a client retries on. Scanning one file is how a
/// check like this quietly stops covering the thing it claims to cover.
fn all_crate_sources() -> String {
    fn walk(dir: &std::path::Path, out: &mut String) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name != "target" && name != "tests" {
                    walk(&path, out);
                }
            } else if name.ends_with(".rs")
                && name != "tests.rs"
                && let Ok(text) = std::fs::read_to_string(&path)
            {
                // Cut in-file test modules: a code named only by an assertion is not emitted.
                // Every module, not just the first: production code below one would otherwise be
                // invisible to this scan.
                for (index, part) in text.split("\n#[cfg(test)]").enumerate() {
                    if index == 0 {
                        out.push_str(part);
                    } else if let Some(after) = part.find("\n}\n") {
                        out.push_str(&part[after..]);
                    }
                }
                out.push('\n');
            }
        }
    }
    let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let mut out = String::new();
    walk(root, &mut out);
    assert!(
        out.len() > 100_000,
        "the crate walk read {} bytes, which is too little to be the workspace",
        out.len()
    );
    out
}

/// Every SQLSTATE class the engine can emit appears in the §14 table, and every class the table
/// lists is one something emits.
///
/// Compared by class rather than by full code: the table documents classes and cites examples, and
/// a client is told to branch on the class.
#[test]
fn every_sqlstate_class_the_engine_emits_is_documented() {
    let mut emitted: BTreeSet<String> = BTreeSet::new();
    for candidate in code_literals(&all_crate_sources()) {
        emitted.insert(candidate[..2].to_owned());
    }
    assert!(
        emitted.len() >= 20,
        "the scan found only {} SQLSTATE classes; the engine emits at least 20, so anything \
         below that means the scan stopped matching the source rather than that the engine got \
         simpler. Raise this floor when classes are added: a floor that only catches total \
         collapse lets a regex that halves its reach pass in silence",
        emitted.len()
    );

    let doc = protocol_doc();
    let table = section(
        &doc,
        "| Class | Meaning | Examples |",
        "The list is what the server emits",
    );
    let documented: BTreeSet<String> = table
        .lines()
        .filter_map(|line| {
            let cell = line.trim().strip_prefix("| `")?;
            let class = cell.split('`').next()?;
            (class.len() == 2).then(|| class.to_owned())
        })
        .collect();

    let undocumented: Vec<_> = emitted.difference(&documented).cloned().collect();
    let unemitted: Vec<_> = documented.difference(&emitted).cloned().collect();
    assert!(
        undocumented.is_empty() && unemitted.is_empty(),
        "docs/wire-protocol.md section 14 no longer matches the code.\n  \
         emitted but undocumented: {undocumented:?}\n  \
         documented but unemitted: {unemitted:?}"
    );
}

/// Every `CommandComplete` tag the wire can send appears in the §9.1 table, and every tag the
/// table lists is one the wire can send.
#[test]
fn every_command_tag_the_wire_sends_is_documented() {
    // One file, unlike the workspace-wide SQLSTATE scan above, and deliberately: `command_tag` is
    // the single funnel every non-COPY tag passes through, and the one COPY special case lives
    // beside it. Should a second tag source appear outside `server.rs`, this scan would stop
    // seeing it — the floor below is what would catch that.
    let source = wire_server_source();
    let body = section(
        &source,
        "fn command_tag(result: &ExecutionResult) -> String {",
        "\n}\n",
    );
    let mut emitted: BTreeSet<String> = BTreeSet::new();
    for (open, close) in [("\"", "\".to_owned()"), ("format!(\"", " {")] {
        for chunk in body.split(open).skip(1) {
            let Some(tag) = chunk.split(close).next() else {
                continue;
            };
            if is_tag(tag) {
                emitted.insert(tag.to_owned());
            }
        }
    }
    // The COPY sub-protocol writes its own tag without going through `command_tag`.
    for chunk in source.split("command_complete(&format!(\"").skip(1) {
        if let Some(tag) = chunk.split(' ').next()
            && is_tag(tag)
        {
            emitted.insert(tag.to_owned());
        }
    }
    assert!(
        emitted.len() >= 50,
        "the scan found only {} command tags; `command_tag` has at least 50 arms, so anything \
         below that means the scan stopped matching the source. Raise this floor when tags are \
         added",
        emitted.len()
    );

    let doc = protocol_doc();
    let table = section(
        &doc,
        "### 9.1 `CommandComplete` tags",
        "A driver MUST treat the tag as informational",
    );
    let mut documented: BTreeSet<String> = BTreeSet::new();
    for line in table.lines() {
        // Only the tag column: the statement column spells things that are not tags.
        let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
        if cells.len() != 2 {
            continue;
        }
        for piece in cells[1].split('`').skip(1).step_by(2) {
            let tag = piece.split(" <").next().unwrap_or(piece).trim();
            if is_tag(tag) {
                documented.insert(tag.to_owned());
            }
        }
    }

    let undocumented: Vec<_> = emitted.difference(&documented).cloned().collect();
    let unemitted: Vec<_> = documented.difference(&emitted).cloned().collect();
    assert!(
        undocumented.is_empty() && unemitted.is_empty(),
        "docs/wire-protocol.md section 9.1 no longer matches `command_tag`.\n  \
         sent but undocumented: {undocumented:?}\n  \
         documented but never sent: {unemitted:?}"
    );
}

/// Whether `s` looks like a command tag: upper-case words, nothing else.
fn is_tag(s: &str) -> bool {
    !s.is_empty()
        && s.len() < 32
        && s.chars().all(|c| c.is_ascii_uppercase() || c == ' ')
        && s.chars().any(|c| c.is_ascii_uppercase())
}

/// Every 5-character SQLSTATE literal in `source`.
fn code_literals(source: &str) -> Vec<String> {
    source
        .split('"')
        .filter(|piece| {
            piece.len() == 5
                && piece
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                && piece
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit() || c == 'P' || c == 'X')
                && piece.chars().filter(char::is_ascii_digit).count() >= 3
        })
        .map(str::to_owned)
        .collect()
}

/// The slice of `text` from `start` up to `end`, panicking with a clear message if either marker
/// has moved — a silently empty section would make both directions pass vacuously.
fn section<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    let from = text.find(start).unwrap_or_else(|| {
        panic!("marker not found, so this check was silently testing nothing: {start:?}")
    });
    let rest = &text[from..];
    let to = rest.find(end).unwrap_or_else(|| {
        panic!("end marker not found, so this check was silently testing nothing: {end:?}")
    });
    &rest[..to]
}
