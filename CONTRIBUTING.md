# Contributing to llmgate

Thanks for your interest! This document covers the development workflow,
conventions, and review process. It is the same workflow the maintainers use.

## Development setup

Requirements: Rust 1.85+ (edition 2024).

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

All four must pass before a PR is ready. The CI workflow
(`.github/workflows/ci.yml`) runs the same gates on every push.

## Branch model

```
main ────────────────────────────────────────────────► (releases)
   └── develop ──────────────────────────────► (integration)
         └── feat/<protocol-or-feature> ──► (feature branches)
```

- **`main`** — released state only. Receives changes via merges from `develop`.
- **`develop`** — integration branch. Feature branches are cut from and merged
  back into `develop`.
- **`feat/<name>`** — one feature or protocol adapter per branch, e.g.
  `feat/adapter/gemini`.

Worktrees are the recommended way to work on multiple branches at once:

```bash
git worktree add ../llmgate-develop develop
git worktree add ../llmgate-gemini -b feat/adapter/gemini develop
```

## Commit conventions

Conventional Commits: `type(scope): short description`.

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`, `perf`.

```bash
git commit -m "feat(adapters): add gemini generateContent conversion

- contents/parts, systemInstruction, generationConfig mapping
- functionCall/functionResponse tool blocks
- thought parts mapped to thinking blocks"
```

Keep the summary under ~72 characters; use the body for detail. A commit
should be one logical change.

## Adding a protocol adapter

The core is protocol-blind; adding a protocol is additive. See
[README.md#adding-a-protocol](README.md#adding-a-protocol) for the
`ProtocolAdapter` trait checklist.

- Create `src/adapters/<name>/` with `mod.rs`, `convert.rs`, `stream.rs`.
- Register the adapter in `src/adapters/mod.rs` and `src/main.rs`.
- Add conversion/stream unit tests alongside the code (`#[cfg(test)]` inline
  modules — there is no separate `tests/` directory).
- Update `config.example.toml` and the docs if the protocol changes
  configuration or endpoint surfaces.

## Testing

- **Unit tests** live inline as `#[cfg(test)] mod tests` in each module:
  conversion round-trips, stream decode/encode, error mapping, and core
  (SSE framing, auth, config, resolver).
- **E2E smoke tests** are run manually against local mock upstreams
  (stream + non-stream, tools, auth, models) before merge; they are not part
  of the automated suite.

## Pull requests

1. Cut `feat/<name>` from `develop`.
2. Implement + test + verify (build, test, clippy, fmt).
3. Open the PR against **`develop`**, not `main`.
4. The PR body should summarize what changed and how it was verified.
5. CI must be green before merge.

## Documentation

- `README.md` describes what the project does and how to use it. It is
  user-facing and must not contain project-status/milestone tracking — that
  lives in `CHANGELOG.md` and `PLAN.md`.
- `CHANGELOG.md` records user-visible changes,
  [Keep a Changelog](https://keepachangelog.com/) style.
- `PLAN.md` is the maintainers' roadmap/milestone tracker.
- `config.example.toml` documents the configuration surface; keep it in sync
  with `src/config.rs`.

## Code of conduct

Be respectful and constructive. Disagreement is fine; personal attacks are
not. Report unacceptable behavior to the maintainers.
