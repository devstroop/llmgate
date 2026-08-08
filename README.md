# llmgate

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/devstroop/llmgate/actions/workflows/ci.yml/badge.svg)](https://github.com/devstroop/llmgate/actions/workflows/ci.yml)

The zero-trust gateway for LLMs. It sits between your applications and cloud AI providers, translating across LLM protocols and inspecting every token in real time. Redact sensitive data, cache responses, and route requests across models and providers—all before your data reaches the cloud.

Protocol-agnostic LLM API adapter gateway in Rust.

A reverse-proxy gateway that adapts between **LLM provider protocols** in both
directions — OpenAI-compatible clients can talk to Anthropic-compatible
upstreams and vice versa, with new protocols (Gemini, OpenAI Responses API,
Ollama, Bedrock, ...) added as pluggable adapters without touching the core.

## Design in one line

The core works on a protocol-neutral internal model (`NeutralRequest` /
`NeutralResponse` / `NeutralStreamEvent`); every protocol is a `ProtocolAdapter`
plugin implementing request/response serialization, SSE stream decode/encode,
and error mapping.

```
OpenAI client ──▶ /v1/chat/completions ──┐
                                         ├─▶ llmgate ──▶ upstream (any protocol, config-driven)
Anthropic client ──▶ /v1/messages ───────┘                  ◀── responses converted back to client format
```

## Features

- **Bidirectional protocol conversion** — OpenAI `chat.completions` ↔ neutral
  ↔ Anthropic `messages`, both client- and upstream-side. Gemini
  (`generateContent`) is supported upstream-side: OpenAI/Anthropic clients can
  be routed to a Gemini provider.
- **SSE streaming** — chunks/events converted live in both directions,
  including thinking/reasoning blocks, tool-call deltas, and usage. The gateway
  emits its own 15s keep-alive so long streams survive proxies, and sets
  `X-Accel-Buffering: no` to defeat intermediary buffering. Gemini streams
  (`streamGenerateContent?alt=sse`) relay with no `[DONE]` terminator, per the
  protocol.
- **Tool calling** — OpenAI `function` tools ↔ Anthropic `input_schema` tools,
  `tool_calls` ↔ `tool_use`/`tool_result` blocks.
- **Reasoning** — `reasoning_content` ↔ `thinking` blocks (with signatures)
  in bodies and streams, with spec-faithful thinking block shapes.
- **Images** — data-URI and URL image sources both ways.
- **Model resolution** — prefix strip → `model_map` → default, protocol-agnostic.
- **Client auth** — optional API keys via `Authorization: Bearer <key>`,
  `api-key: <key>`, or `x-api-key: <key>`; constant-time comparison.
- **Request correlation** — inbound `x-request-id` is honored (or generated),
  threaded into logs, and echoed on the response.
- **Model listing** — `/v1/models` fetched from the upstream and re-serialized
  in the client's native shape.
- **Token counting** — `/v1/messages/count_tokens` heuristic.
- **JSON-as-stream fallback** — if an upstream ignores `stream: true` and
  answers with plain JSON, the body is converted into a single-event stream
  instead of an empty stream.
- **Privacy Guard (zero-trust redaction)** — reversible, session-scoped
  pseudonymization at the gateway boundary: PII, IPs, secrets, and custom
  rule matches are replaced with tokens (`<EMAIL_1>`, `<IP_2>`, ...) before
  the request leaves the gateway and restored transparently in the
  response — streaming or not. The upstream provider never sees sensitive
  data; the client never sees tokens. See [Privacy Guard](#privacy-guard).

## Quick start

Requirements: Rust 1.85+ (edition 2024).

```bash
cargo build --release

# create config (see config.example.toml)
cp config.example.toml config.toml

CONFIG_PATH=config.toml ./target/release/llmgate
```

Default upstream is `http://localhost:11434` (an OpenAI-compatible endpoint) —
point `[upstream]` at any OpenAI- or Anthropic-compatible provider.

### Usage examples

```bash
# OpenAI format -> whatever [upstream] speaks
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}'

# Anthropic format
curl -X POST http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"hello"}],"max_tokens":1024}'

# Streaming
curl -N -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}],"stream":true}'

# Models
curl http://localhost:5000/v1/models

# Health
curl http://localhost:5000/health
```

## Configuration

See [config.example.toml](config.example.toml) for the full schema with
comments. Sections: `[client]` (protocols served), `[upstream]` (protocol,
URL, auth headers, timeout), `[models]` (default, map, prefixes), `[server]`
(bind host/port), `[auth]` (client API keys).

`CONFIG_PATH` env var overrides the config file location (default `./config.toml`).
When `CONFIG_PATH` is **unset** and no `config.toml` exists, built-in defaults
are used (loopback-only bind, no auth — see below). When `CONFIG_PATH` is set
but the file is missing, startup **aborts**: the gateway never silently runs
in the unauthenticated default configuration.

Timeout semantics: `upstream.timeout_ms` (default 60s) bounds **non-streaming**
requests (conversations and model listings). **Streaming** requests have no
total timeout — the stream is bounded by the SSE protocol itself
(`[DONE]` / `message_stop` / connection close) so long generations are not cut
off, with a 120s per-chunk idle timeout and a 15s bound on waiting for the
upstream's response headers. TCP connect timeout is a fixed 10s. Upstream
redirects are NEVER followed (credentials could be leaked to the redirect
target); a redirecting upstream is an error.

### Gemini upstream

Point `[upstream]` at a Gemini-compatible base URL (e.g.
`https://generativelanguage.googleapis.com`) with `protocol = "gemini"`, and
set the API key via `[[upstream.extra_headers]]` (`x-goog-api-key`) or
`authorization` (`Bearer <key>`). The resolved model name becomes the URL path
segment (`:generateContent` / `:streamGenerateContent?alt=sse`), so
`[models.map]` entries should map client model names to Gemini model ids (e.g.
`"gpt-4o" = "gemini-2.5-flash"`).

### Privacy Guard

The **Zero-Trust LLM Gateway**: route requests to any cloud LLM without ever
exposing PII, IPs, or secrets to the provider. When enabled, the gateway
replaces sensitive entities in the inbound request with synthetic tokens
(`<EMAIL_1>`, `<IP_2>`, `<SECRET_3>`, ...), forwards the sanitized request
upstream, and restores the original values as the response streams back —
the client experiences zero degradation, the provider learns nothing.

```toml
[privacy_guard]
enabled = true
vault = "memory"   # only "memory" (session-scoped) is implemented

# Custom rules (when omitted, a built-in conservative set is used: email,
# IPv4, US phone, API keys).
[[privacy_guard.rules]]
name = "INTERNAL_IP"
pattern = '\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
replacement = "<IP_{n}>"

[privacy_guard.allow_list]
domains = ["devstroop.com"]        # emails/URLs on these domains pass through
patterns = ['\bfoo@example\.com\b'] # regex-exempt matches
```

How it works:

- **Rules are regex-driven and order-sensitive** — every text-bearing content
  block (text, thinking, tool results, tool-call input JSON) is scanned per
  rule. Repeated values reuse one token so the model sees a consistent symbol
  per entity.
- **Restore is exact and streaming-safe** — tokens use the `<NAME_n>`
  grammar (`>` terminator), so `<IP_1>` can never match inside `<IP_10>`.
  Response streams are restored with a buffered Aho-Corasick matcher that
  reassembles tokens split across arbitrary SSE chunk boundaries, per channel
  (text / reasoning / tool arguments).
- **Allow-lists** — domains and regexes that must never be redacted
  (e.g. your own company domain).
- **Session-scoped vault** — token mappings live in memory exactly as long
  as the request; they are dropped when the response finishes. Nothing is
  persisted.
- **Fails closed** — an invalid rule pattern, replacement template (must
  contain `{n}`), or vault backend prevents startup rather than silently
  running unredacted.
- **Memory note** — redaction works on a request clone (so a cap-exhausted
  request is rejected atomically, never partially redacted): peak memory
  for privacy-enabled requests is ~2x the request size (up to ~32 MiB at
  the 16 MiB inbound cap). Account for that when enabling the guard on
  memory-constrained hosts.

Scope notes (v1): image blocks are not redacted (no OCR/vision pass), tool
*definitions* are not scanned, and there is a 4096-token cap per session —
a request that exhausts the cap is **rejected** (fail closed): the provider
never receives the unredacted tail. A persistent vault backend (e.g. sqlite)
is a future extension; the session API is the seam.

## Project layout

```
src/
├── main.rs            # bootstrap, route mounting per endpoint kind, auth + request-id layers
├── config.rs          # TOML config + env override
├── auth.rs            # client API-key middleware (constant-time)
├── resolver.rs        # model resolution pipeline (prefix strip -> map -> default)
├── proxy.rs           # upstream HTTP client + forward helpers (timeout split)
├── core/
│   ├── neutral.rs     # protocol-neutral request/response/stream model
│   ├── registry.rs    # ProtocolAdapter trait, StreamDecoder/Encoder, registry
│   ├── pipeline.rs    # generic handlers (conversation, models, count_tokens, streaming)
│   ├── error.rs       # protocol-independent error taxonomy
│   └── sse.rs         # SSE framing parser (bounded buffering)
└── adapters/
    ├── openai/        # convert.rs + stream.rs
    ├── anthropic/     # convert.rs + stream.rs
    └── gemini/        # convert.rs + stream.rs (upstream-side generateContent)
```

Unit tests are inline `#[cfg(test)]` modules; there is no separate `tests/`
directory.

## Adding a protocol

Implement `ProtocolAdapter` in a new `src/adapters/<name>/` module and register
it. The core does not change.

1. **`name()`** — stable protocol name used in config.
2. **`endpoints()`** — inbound paths served to clients with their
   `EndpointKind` (`Chat`, `Messages`, `Models`, `CountTokens`).
3. **`conversation_url()`** — upstream URL from the provider base; the `model`
   argument supports path-parameterized protocols (e.g. Gemini). Returns
   `Result` so path-injection attempts and invalid models can be rejected.
4. **`stream_conversation_url()`** — default `conversation_url()`, overridden
   by protocols whose streaming endpoint differs (e.g. Gemini's
   `:streamGenerateContent?alt=sse`).
5. **`request_headers()`** — protocol-required upstream headers
   (e.g. `anthropic-version`).
6. **`parse_request` / `serialize_request`** — inbound body ↔ `NeutralRequest`.
   The neutral model uses content blocks (text / image / thinking /
   redacted_thinking / tool_use / tool_result), so any block-based protocol
   maps directly.
7. **`parse_response` / `serialize_response`** — upstream body ↔
   `NeutralResponse` (content blocks + finish_reason + usage).
8. **`stream_decoder()` / `stream_encoder()`** — stateful SSE conversion.
   Decoder: upstream SSE payload → `NeutralStreamEvent`s; the pipeline feeds
   it with framing already applied. Encoder: event → client SSE lines.
9. **`serialize_error()`** — `AdapterError` → native error body + status.
10. **`parse_models` / `serialize_models`** — optional model listing support;
    the default `parse_models` handles the common `{"data":[{id,...}]}` shape.
11. **Register** in `src/adapters/mod.rs` and `src/main.rs`, and add the name
    to `[client]`/`[upstream]`.

`NeutralStreamEvent` is deliberately coarse (`MessageStart`, `TextDelta`,
`ReasoningDelta`, `ToolCallDelta`, `MessageStop`); adapters accumulate
protocol-specific fragments (e.g. tool-call arguments) themselves.

## Verification

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

E2E smoke tests run manually against local mock upstreams (both directions,
stream + non-stream, tools, auth, models).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch model, commit
conventions, and review process. Changes are tracked in
[CHANGELOG.md](CHANGELOG.md).

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability reporting policy and
supported-version scope. **Do not open public issues for security
vulnerabilities** — report them privately.

## Code of conduct

All contributors are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
