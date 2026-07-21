# Contributing to NusaDB

Thanks for contributing. NusaDB is a database engine — **correctness is non-negotiable**, so the
contribution bar is higher than a typical app. Read this before opening a PR.

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

## The rules that get PRs rejected

1. **Dependency direction.** Crates form a layered graph (`cli → … → storage → core`). An inner
   crate must never import an outer one. If two crates need a shared type, it goes in
   `nusadb-core`. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layer/crate map.
2. **No async below L2.** `tokio` is allowed only in `nusadb-wire`, `nusadb-server`, `nusadb-cli`.
   Storage/WAL/LSM/txn stay synchronous so they run under deterministic simulation.
3. **No hidden I/O, time, or randomness in engine code.** Take `&dyn PageStore` / `&dyn Clock` /
   `&impl Rng` (defined in `nusadb-core`). Direct `std::time::now()`, `rand`, or filesystem calls
   break DST and will be rejected.
4. **No `.unwrap()` / `unimplemented!()` in non-test code.** These are `deny` clippy lints. Return
   `Result` and use `?`.
5. **No third-party database protocol libraries.** The wire protocol is NusaDB's own original format.

## Test placement convention

Tests go in a crate's **`tests/` directory** (sibling of `src/`), one file per source module,
named **`test_<module>.rs`** (e.g. `src/memtable.rs` → `tests/test_memtable.rs`). These are
integration tests: a separate crate that exercises only the **public API** — so they double as a
check that the public surface is usable. Add `#![allow(clippy::unwrap_used, ...)]` at the top
since the in-tests clippy carve-outs do not apply to a separate test crate.

The **only** exception is a true white-box unit test that must reach a module's *private*
internals (e.g. `page.rs` checksum math, `btree` node split). Those stay inline as
`#[cfg(test)] mod tests` — we do **not** make internals `pub` just to relocate a test, because
loosening encapsulation on storage internals is an integrity risk for a database. Everything that
can be expressed against the public API lives in `tests/`.

## Tests are part of the change, not a follow-up

- Logic change → tests in the crate's `tests/` directory (or inline only for private internals).
- New SQL behavior → a `.slt` file under `crates/nusadb-e2e/tests/slt/pN_*/`.
- Storage/WAL/txn change → it must survive deterministic simulation testing (DST); add a regression
  seed if DST found a bug.

Run a single test while iterating:

```bash
cargo test -p nusadb-storage btree::tests::insert_split
```

## Commit & PR

- Keep PRs scoped to one concern.
- Describe what changed and why, and how it was tested, in the PR description.
- Green `cargo fmt --check`, `cargo clippy -D warnings`, and `cargo test` are the minimum bar for review.

## Where things go

| Adding... | Put it in... |
|---|---|
| A type used by 3+ crates | `nusadb-core` |
| Layer-specific code | `nusadb-{that-layer}` |
| Cross-crate integration test | `tests/integration/` |
| Architectural decision | `ARCHITECTURE.md` |

When in doubt about placement, see [`ARCHITECTURE.md`](ARCHITECTURE.md) or open a discussion.
