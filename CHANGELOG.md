# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OpenAI adapter** — `/v1/chat/completions` request/response conversion
  including tools, reasoning (`reasoning_content`/`reasoning`), and images
  (data-URI and URL sources).
- **Anthropic adapter** — `/v1/messages` request/response conversion including
  tools (`input_schema`), extended thinking blocks (with signatures), images,
  and protocol-native error mapping (`anthropic-version` header, 529 overloaded).
- **SSE streaming** — stateful stream decoders/encoders for both adapters;
  thinking/reasoning deltas, tool-call deltas, and usage relayed live in both
  directions. Anthropic event choreography (`message_start` →
  `content_block_*` → `signature_delta` → `message_delta` → `message_stop`)
  is emitted natively; OpenAI streams terminate with `data: [DONE]`.
- **Client auth** — optional API keys via `Authorization: Bearer <key>`,
  `api-key: <key>`, or `x-api-key: <key>`, with constant-time comparison.
- **Model listing** — `/v1/models` fetched from the upstream and re-serialized
  in each client protocol's native shape.
- **Token counting** — `/v1/messages/count_tokens` heuristic endpoint.
- **Model resolution** — prefix strip → `model_map` → default, protocol-agnostic.
- **Request correlation** — inbound `x-request-id` honored (or generated),
  threaded into logs, and echoed on responses.
- **SSE keep-alive** — gateway emits `: keepalive` comments on a 15s idle
  interval; `X-Accel-Buffering: no` and `Cache-Control: no-cache` set on
  streamed responses to defeat intermediary buffering.
- **JSON-as-stream fallback** — upstreams that ignore `stream: true` and reply
  with plain JSON are converted into a single-event stream instead of an empty
  stream.
- **Continuous integration** — `.github/workflows/ci.yml` runs `cargo fmt`,
  `cargo clippy -D warnings`, and `cargo test` on every push and PR.
- **Project scaffolding** — LICENSE, CONTRIBUTING.md, CHANGELOG.md, SECURITY.md,
  and Cargo.toml metadata (repository, readme, keywords, categories,
  rust-version).

### Changed

- **Timeout semantics** — `upstream.timeout_ms` now bounds non-streaming
  requests (conversations + model listings). Streaming requests have no total
  timeout (bounded by the SSE protocol) so long generations are not cut off;
  TCP connect timeout is a fixed 10s.
- **Spec-faithful thinking blocks** — Anthropic `content_block_start` for
  thinking blocks carries a `thinking` field (not `text`), per the Anthropic
  streaming spec.
- **Bounded SSE buffering** — the framing parser discards buffers past a
  64 MiB cap instead of growing unbounded on a broken or malicious upstream.
- **Anthropic stream indexing** — `input_json_delta` uses the delta's own
  `index` (authoritative per spec) instead of a tracked tool index.
- **OpenAI model listing** — `created` is a real timestamp rather than `0`.

### Fixed

- Empty client streams when an upstream ignores the `stream` flag
  (now handled by the JSON-as-stream fallback).
- Silent stream death behind proxies on long thinking/tool-call gaps
  (now handled by gateway keep-alive).

### Security

- API-key comparison is constant-time; timing no longer reveals key
  length/prefix structure.
- Inbound request bodies are capped at 16 MiB; SSE framing buffers are capped
  at 64 MiB.

## [0.1.0] - Unreleased

Initial development baseline: protocol-agnostic core (neutral model, adapter
trait, registry, pipeline), OpenAI and Anthropic adapters (non-streaming and
streaming), auth middleware, model listing, token counting, and configuration
via TOML with environment overrides.

[Unreleased]: https://github.com/devstroop/model-adapter/compare/main...develop
[0.1.0]: https://github.com/devstroop/model-adapter/releases/tag/v0.1.0
