# Contributing to NusaDB

Thanks for contributing. NusaDB is a database engine, so correctness matters more than in a typical
app and the contribution bar is a bit higher. Please read this before opening a PR.

## Getting started

```bash
# One-time setup
rustup show && rustup component add rustfmt clippy
# optional: cargo install cargo-deny

# Before every push — run the same checks CI runs:
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets --no-fail-fast
cargo deny check all          # if cargo-deny is installed
```

Toolchain is pinned to **Rust 1.95.0, edition 2024** (`rust-toolchain.toml`). Don't bump it in a
feature PR.

## What gets a PR rejected

A handful of hard rules, mostly there to keep the engine testable and the layering honest:

- Respect the dependency direction. The crates form a layered graph (`cli → … → storage → core`);
  an inner crate must never import an outer one. Shared types go in `nusadb-core`. The full map is in
  [`ARCHITECTURE.md`](ARCHITECTURE.md).
- No async below L2. `tokio` is allowed only in `nusadb-wire`, `nusadb-server`, and `nusadb-cli`. The
  storage, WAL, and transaction layers stay synchronous so they run under deterministic simulation.
- No hidden I/O, time, or randomness in engine code. Take `&dyn PageStore`, `&dyn Clock`, and
  `&impl Rng` (all from `nusadb-core`). A direct `std::time::now()`, `rand`, or filesystem call breaks
  DST and will be rejected.
- No `.unwrap()` or `unimplemented!()` in non-test code — they are `deny` clippy lints. Return a
  `Result` and use `?`.
- No third-party database protocol libraries. The wire protocol is NusaDB's own original format.

## Test placement convention

Tests go in a crate's `tests/` directory (a sibling of `src/`), one file per source module, named
`test_<module>.rs` — for example `src/page.rs` gets `tests/test_page.rs`. These are integration
tests: a separate crate that exercises only the public API, so they double as a check that the
public surface is actually usable. Put `#![allow(clippy::unwrap_used, ...)]` at the top, since the
in-test clippy carve-outs don't apply to a separate test crate.

The one exception is a true white-box unit test that has to reach a module's private internals —
`page.rs` checksum math, or a `btree` node split. Those stay inline as `#[cfg(test)] mod tests`. We
don't make internals `pub` just to move a test out, because loosening encapsulation on storage
internals is an integrity risk for a database. Anything that can be written against the public API
lives in `tests/`.

## Tests are part of the change, not a follow-up

- Logic change → tests in the crate's `tests/` directory (or inline only for private internals).
- New SQL behavior → a `.slt` file under `crates/nusadb-e2e/tests/slt/pN_*/`.
- Storage/WAL/txn change → it must survive deterministic simulation testing (DST); add a regression
  seed if DST found a bug.

Run a single test while iterating:

```bash
cargo test -p nusadb-storage btree::tests::insert_split
```

## Branching & pull requests

`master` is the stable, always-releasable trunk. Releases are tagged from it (`vX.Y.Z`), which
triggers the binary + GitHub Release build. Don't commit directly to `master`.

For an external contribution:

1. Fork the repo and branch off `master` with a short topic name: `feat/<topic>`, `fix/<topic>`,
   `docs/<topic>`, or `chore/<topic>`.
2. Make focused commits (one concern per PR). Keep the branch rebased on the latest `master`.
3. Open a pull request against `master`. CI runs `fmt` + `clippy -D warnings` + the test suite; all
   must be green.
4. A maintainer reviews and merges. Larger or architectural changes: open a discussion / issue first.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`,
`docs:`, `chore:`, `test:`, `perf:`, `refactor:` — a concise imperative summary, details in the body.
Don't put internal tracker IDs or issue codenames in messages or code; describe the change itself.

Green `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo test --workspace` are the minimum bar for review.

## Where things go

| Adding... | Put it in... |
|---|---|
| A type used by 3+ crates | `nusadb-core` |
| Layer-specific code | `nusadb-{that-layer}` |
| Cross-crate / end-to-end test | `crates/nusadb-e2e` |
| Architectural decision | `ARCHITECTURE.md` |

When in doubt about placement, see [`ARCHITECTURE.md`](ARCHITECTURE.md) or open a discussion.
