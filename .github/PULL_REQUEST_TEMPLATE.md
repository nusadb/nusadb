## What & why

<!-- One paragraph: what this changes and the motivation. -->

## Scope

- Layer:
- Crates touched:

## Checklist

- [ ] `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` are green
- [ ] No upward/circular crate dependency introduced (outer → inner only)
- [ ] No `tokio`/async added below L2 (wire/server/cli only)
- [ ] No direct `std::time` / `rand` / filesystem in engine code (use `Clock`/`Rng`/`PageStore`)
- [ ] No `.unwrap()` / `unimplemented!()` in non-test code
- [ ] Tests included with the change (unit/property, `.slt`, DST seed, or fuzz seed as applicable)
- [ ] Architectural decisions documented (e.g. in `ARCHITECTURE.md`) if a boundary or load-bearing choice changed

## Notes for reviewer

<!-- Anything non-obvious, trade-offs, follow-ups. -->
