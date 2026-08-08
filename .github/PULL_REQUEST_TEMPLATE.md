## Summary

What this PR does and why. One logical change per PR.

## Verification

- [ ] `cargo test` passes
- [ ] `cargo clippy --all-targets -- -D warnings` is clean
- [ ] `cargo fmt --check` is clean
- [ ] E2E smoke tested against a local mock upstream (stream + non-stream) — describe what was exercised

## Checklist

- [ ] Target branch is `develop` (not `main`)
- [ ] Conventional Commits style: `type(scope): short description`
- [ ] Behavior changes carry inline unit tests (`#[cfg(test)]` modules, no `tests/` dir)
- [ ] `config.example.toml` updated if the change touches configuration
- [ ] README.md / CHANGELOG.md updated for user-visible changes

## Related issues

Closes #
