# model-adapter — Development Plan

Protocol-agnostic LLM API adapter gateway written in Rust. Translates between
provider protocols (OpenAI-compatible, Anthropic, and future ones) in both
directions, using a protocol-blind core and pluggable adapters.

## Vision

Most gateways pick one canonical protocol (usually OpenAI) and hardcode
conversion logic around it. `model-adapter` instead treats **every** protocol as
a plugin:

```
client request ─▶ [ProtocolParser]  ─▶ NeutralRequest ─▶ [ProtocolSerializer] ─▶ upstream
                  (client protocol)                    (upstream protocol)
upstream resp ──▶ [ProtocolParser] ─▶ NeutralResponse ─▶ [ProtocolSerializer] ─▶ client
stream        ──▶ [StreamDecoder] ──▶ NeutralEvents  ──▶ [StreamEncoder]  ──▶ client SSE
```

The core pipeline (`parse → resolve model → forward → convert back`) is blind to
any specific protocol. Adding a new protocol (Gemini, OpenAI Responses API,
Ollama, Bedrock, ...) means implementing one trait and registering it — zero
core changes.

## Architecture

### Neutral internal model

Protocol-independent types in `src/core/neutral.rs`:

- `NeutralRequest` — model, messages (content blocks: text / image /
  tool_use / tool_result / thinking), tools, max_tokens, temperature, top_p,
  top_k, stop, stream
- `NeutralResponse` — id, model, content blocks, tool_calls, finish_reason, usage
- `NeutralStreamEvent` — MessageStart / TextDelta / ReasoningDelta /
  ToolCallDelta / MessageStop / Error

### The extension point

```rust
trait ProtocolAdapter: Send + Sync {
    fn name(&self) -> &'static str;                          // "openai", "anthropic", ...
    fn endpoints(&self) -> Vec<(&'static str, EndpointKind)>; // inbound paths this protocol owns
    fn parse_request(&self, body: &str) -> Result<NeutralRequest>;
    fn serialize_request(&self, req: &NeutralRequest) -> Result<String>;
    fn parse_response(&self, body: &str) -> Result<NeutralResponse>;
    fn serialize_response(&self, resp: &NeutralResponse) -> Result<String>;
    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;       // upstream SSE -> NeutralStreamEvent
    fn stream_encoder(&self) -> Box<dyn StreamEncoder>;       // NeutralStreamEvent -> client SSE text
    fn serialize_error(&self, err: &AdapterError) -> String;  // protocol-native error shape
}
```

### Registry & wiring

- `ProtocolRegistry` holds named adapters (`openai`, `anthropic`), assembled at
  startup from config.
- Config selects which protocols to serve on the client side and which protocol
  + URL the upstream speaks.
- Model resolution (prefix strip → `model_map` → default) lives in the core and
  is protocol-agnostic.

## Project layout

```
model-adapter/
├── Cargo.toml              # axum, tokio, reqwest, serde, serde_json, anyhow, thiserror, tracing, toml
├── config.example.toml
├── PLAN.md
├── README.md
├── src/
│   ├── main.rs             # bootstrap, registry assembly, router, shutdown
│   ├── config.rs           # TOML + env overrides
│   ├── auth.rs             # per-protocol auth extraction (Bearer / x-api-key)
│   ├── resolver.rs         # model pipeline (agnostic)
│   ├── core/               # protocol-agnostic core
│   │   ├── mod.rs          # trait definitions, registry, pipeline orchestration
│   │   ├── neutral.rs      # NeutralRequest / NeutralResponse / NeutralStreamEvent
│   │   ├── pipeline.rs     # generic handler: parse -> resolve -> forward -> convert back
│   │   └── error.rs        # AdapterError (rate_limit, auth, overloaded, quota, api, ...)
│   ├── adapters/
│   │   ├── mod.rs          # registry impl, adapter wiring
│   │   ├── openai/         # request/response serde, stream decoder+encoder, error map
│   │   └── anthropic/      # request/response serde, stream decoder+encoder, error map
│   └── proxy.rs            # reqwest forwarding (streaming + non-streaming)
└── tests/
    ├── converters.rs       # per-adapter unit tests w/ fixture JSON
    └── integration.rs      # mock-upstream e2e: openai <-> anthropic, both directions
```

## Conversion mapping notes

Work in `src/adapters/*/` — the tables below live inside each adapter, not the core.

- **Anthropic ↔ neutral**: `system` field ↔ system message; tools
  `{name, description, input_schema}` ↔ neutral tool; `tool_use`/`tool_result`
  blocks ↔ neutral blocks; `thinking` ↔ reasoning content; stop_reason ↔
  finish_reason; `input_tokens`/`output_tokens` ↔ neutral usage.
- **OpenAI ↔ neutral**: `{"type":"function","function":{...}}` ↔ neutral tool;
  `role:"tool"` + `tool_call_id` ↔ tool_result block; `reasoning_content` ↔
  reasoning; finish_reason ↔ neutral; `prompt_tokens`/`completion_tokens` ↔
  neutral usage.
- **Streaming**: adapters provide a stateful `StreamDecoder` (upstream SSE →
  neutral events) and `StreamEncoder` (neutral events → client SSE lines).
  Anthropic event choreography (`message_start` → `content_block_start/delta` →
  `signature_delta` on block switch → `message_delta` → `message_stop` → `[DONE]`)
  is entirely inside the anthropic adapter.

## Milestones

| # | Scope | Definition of done |
|---|-------|--------------------|
| M1 | Core framework | `neutral.rs`, `ProtocolAdapter` trait, `ProtocolRegistry`, generic pipeline, config load, health endpoint. `cargo build` clean, core unit tests pass. |
| M2 | openai adapter | Request/response conversion (non-stream) incl. tools + reasoning fields. Unit tests w/ fixture JSON. |
| M3 | anthropic adapter | Request/response conversion (non-stream) incl. tools, thinking, images, error mapping. Unit tests w/ fixture JSON. |
| M4 | Streaming | `StreamDecoder`/`StreamEncoder` for both adapters; streaming e2e vs mock upstream (both directions). |
| M5 | Service polish | Auth (Bearer/x-api-key), `/v1/models` (list + per-protocol naming), proxy timeout/retry, edge cases. |
| M6 | Docs & hardening | README + "how to add a protocol" guide, config example, curl examples, clippy clean. |

### M1 details

- `Cargo.toml` deps: axum 0.8, tokio (full), reqwest (json/stream/rustls), serde,
  serde_json, anyhow, thiserror, tracing, tracing-subscriber, toml.
- `src/core/neutral.rs`: full neutral model types.
- `src/core/mod.rs`: `ProtocolAdapter`, `StreamDecoder`, `StreamEncoder`,
  `ProtocolRegistry` (register + get by name), `EndpointKind` (`Chat`,
  `Messages`, `Models`, `Health`).
- `src/core/pipeline.rs`: `handle_chat` generic over client adapter + upstream
  adapter — parse, resolve model, serialize for upstream, forward via
  `proxy.rs`, convert response back.
- `src/config.rs`: TOML schema with `[client]` protocols, `[upstream]`
  protocol/url/auth, `[models]` map; env overrides.
- `src/main.rs`: bootstrap registry from config, mount routes per client
  protocol's `endpoints()`, health at `/health`.

## Verification

- `cargo test` — per-adapter unit tests + e2e against a local mock upstream
  (both directions, stream + non-stream, with tools).
- `cargo clippy -- -D warnings`
- Manual curl for both protocols.

## Out of scope (for now)

- Embeddings / moderations / audio endpoints (sidecars etc.)
- Token-count heuristics, rate limiting, per-user budgets
- Multi-upstream routing / load balancing / failover
- Warpgate-style egress proxy pools
