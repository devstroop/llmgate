# model-adapter

Protocol-agnostic LLM API adapter gateway in Rust.

A reverse-proxy gateway that adapts between **LLM provider protocols** in both
directions — OpenAI-compatible clients can talk to Anthropic-compatible
upstreams and vice versa, with new protocols (Gemini, OpenAI Responses API,
Ollama, Bedrock, ...) added as pluggable adapters without touching the core.

## Status

All planned milestones complete, including the M7 reliability-hardening pass.
See [PLAN.md](PLAN.md) for the full plan and
[the "adding a protocol" section](#adding-a-protocol) below for extending it.

| Milestone | Scope | Status |
|---|---|---|
| M1 | Core framework: neutral model, adapter trait, registry, pipeline, config | done |
| M2 | OpenAI adapter (non-stream) | done |
| M3 | Anthropic adapter (non-stream) | done |
| M4 | SSE streaming (both adapters) | done |
| M5 | Auth, /v1/models, count_tokens | done |
| M6 | Docs & hardening | done |
| M7 | Reliability hardening (timeouts, keep-alive, request-ids, bounded buffering) | done |

## Design in one line

The core works on a protocol-neutral internal model (`NeutralRequest` /
`NeutralResponse` / `NeutralStreamEvent`); every protocol is a `ProtocolAdapter`
plugin implementing request/response serialization, SSE stream decode/encode,
and error mapping.

```
OpenAI client ──▶ /v1/chat/completions ──┐
                                         ├─▶ model-adapter ──▶ upstream (any protocol, config-driven)
Anthropic client ──▶ /v1/messages ───────┘                  ◀── responses converted back to client format
```

## Features

- **Bidirectional protocol conversion** — OpenAI `chat.completions` ↔ neutral
  ↔ Anthropic `messages`, both client- and upstream-side.
- **SSE streaming** — chunks/events converted live in both directions,
  including thinking/reasoning blocks, tool-call deltas, and usage. The gateway
  emits its own 15s keep-alive so long streams survive proxies, and sets
  `X-Accel-Buffering: no` to defeat intermediary buffering.
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

## Quick start

Requirements: Rust 1.85+ (edition 2024).

```bash
cargo build --release

# create config (see config.example.toml)
cp config.example.toml config.toml

CONFIG_PATH=config.toml ./target/release/model-adapter
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

`CONFIG_PATH` env var overrides the config file location (default `./config.toml`);
if no file exists, defaults are used.

Timeout semantics: `upstream.timeout_ms` (default 60s) bounds **non-streaming**
requests (conversations and model listings). **Streaming** requests have no
total timeout — the stream is bounded by the SSE protocol itself
(`[DONE]` / `message_stop` / connection close) so long generations are not cut
off. TCP connect timeout is a fixed 10s.

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
    └── anthropic/     # convert.rs + stream.rs
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
   argument supports path-parameterized protocols (e.g. Gemini).
4. **`request_headers()`** — protocol-required upstream headers
   (e.g. `anthropic-version`).
5. **`parse_request` / `serialize_request`** — inbound body ↔ `NeutralRequest`.
   The neutral model uses content blocks (text / image / thinking /
   tool_use / tool_result), so any block-based protocol maps directly.
6. **`parse_response` / `serialize_response`** — upstream body ↔
   `NeutralResponse` (content blocks + finish_reason + usage).
7. **`stream_decoder()` / `stream_encoder()`** — stateful SSE conversion.
   Decoder: upstream SSE payload → `NeutralStreamEvent`s; the pipeline feeds
   it with framing already applied. Encoder: event → client SSE lines.
8. **`serialize_error()`** — `AdapterError` → native error body + status.
9. **`parse_models` / `serialize_models`** — optional model listing support;
   the default `parse_models` handles the common `{"data":[{id,...}]}` shape.
10. **Register** in `src/main.rs` and add the name to `[client]`/`[upstream]`.

`NeutralStreamEvent` is deliberately coarse (`MessageStart`, `TextDelta`,
`ReasoningDelta`, `ToolCallDelta`, `MessageStop`); adapters accumulate
protocol-specific fragments (e.g. tool-call arguments) themselves.

## Verification

```bash
cargo test        # 53 unit tests: converters, streams, sse, auth, config, resolver
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

E2E smoke tests run manually against local mock upstreams (both directions,
stream + non-stream, tools, auth, models — see commit messages for M3/M4 and
the working-tree diff for M7).

## Out of scope (for now)

- Embeddings / moderations / audio endpoints
- Rate limiting, per-user budgets, cost tracking
- Multi-upstream routing / load balancing / failover
- Egress proxy pools
