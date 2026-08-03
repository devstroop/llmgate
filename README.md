# model-adapter

Protocol-agnostic LLM API adapter gateway in Rust.

A reverse-proxy gateway that adapts between **LLM provider protocols** in both
directions — OpenAI-compatible clients can talk to Anthropic-compatible
upstreams and vice versa, with new protocols (Gemini, OpenAI Responses API,
Ollama, Bedrock, ...) added as pluggable adapters without touching the core.

## Status

All planned milestones complete. See [PLAN.md](PLAN.md) for the full plan and
[the "adding a protocol" section](#adding-a-protocol) below for extending it.

| Milestone | Scope | Status |
|---|---|---|
| M1 | Core framework: neutral model, adapter trait, registry, pipeline, config | done |
| M2 | OpenAI adapter (non-stream) | done |
| M3 | Anthropic adapter (non-stream) | done |
| M4 | SSE streaming (both adapters) | done |
| M5 | Auth, /v1/models, count_tokens | done |
| M6 | Docs & hardening | done |

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
  including thinking/reasoning blocks, tool-call deltas, and usage.
- **Tool calling** — OpenAI `function` tools ↔ Anthropic `input_schema` tools,
  `tool_calls` ↔ `tool_use`/`tool_result` blocks.
- **Reasoning** — `reasoning_content` ↔ `thinking` blocks (with signatures)
  in bodies and streams.
- **Images** — data-URI and URL image sources both ways.
- **Model resolution** — prefix strip → `model_map` → default, protocol-agnostic.
- **Client auth** — optional API keys via `Authorization: Bearer`,
  `api-key`, or `x-api-key` headers.
- **Model listing** — `/v1/models` fetched from the upstream and re-serialized
  in the client's native shape.
- **Token counting** — `/v1/messages/count_tokens` heuristic.

## Quick start

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
```

## Configuration

See [config.example.toml](config.example.toml) for the full schema with
comments. Sections: `[client]` (protocols served), `[upstream]` (protocol,
URL, auth headers, timeout), `[models]` (default, map, prefixes), `[server]`
(bind host/port), `[auth]` (client API keys).

`CONFIG_PATH` env var overrides the config file location (default `./config.toml`);
if no file exists, defaults are used.

## Project layout

```
src/
├── main.rs            # bootstrap, route mounting per endpoint kind, auth layer
├── config.rs          # TOML config + env override
├── auth.rs            # client API-key middleware
├── resolver.rs        # model resolution pipeline (prefix strip -> map -> default)
├── proxy.rs           # upstream HTTP client + forward helpers
├── core/
│   ├── neutral.rs     # protocol-neutral request/response/stream model
│   ├── registry.rs    # ProtocolAdapter trait, StreamDecoder/Encoder, registry
│   ├── pipeline.rs    # generic handlers (conversation, models, count_tokens)
│   ├── error.rs       # protocol-independent error taxonomy
│   └── sse.rs         # SSE framing parser
└── adapters/
    ├── openai/        # convert.rs + stream.rs
    └── anthropic/     # convert.rs + stream.rs
```

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
cargo test        # unit tests per adapter + core
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

E2E smoke tests run against local mock upstreams (see commit messages for M3/M4
for the exact flows verified: both directions, stream + non-stream, tools,
auth, models).

## Out of scope (for now)

- Embeddings / moderations / audio endpoints
- Rate limiting, per-user budgets, cost tracking
- Multi-upstream routing / load balancing / failover
- Egress proxy pools
