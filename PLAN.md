# llmgate — Development Plan

Protocol-agnostic LLM API adapter gateway written in Rust. Translates between
provider protocols (OpenAI-compatible, Anthropic, and future ones) in both
directions, using a protocol-blind core and pluggable adapters.

## Vision

Most gateways pick one canonical protocol (usually OpenAI) and hardcode
conversion logic around it. `llmgate` instead treats **every** protocol as
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
    fn conversation_url(&self, base: &str, model: &str)
        -> Result<String, AdapterError>;      // upstream chat URL (Result: reject path-injection)
    fn stream_conversation_url(&self, base: &str, model: &str)
        -> Result<String, AdapterError> {     // default = conversation_url(); override when the
        self.conversation_url(base, model)    // streaming endpoint differs (e.g. Gemini ?alt=sse)
    }
    fn request_headers(&self) -> Vec<(String, String)>;
    fn parse_request(&self, body: &str) -> Result<NeutralRequest, AdapterError>;
    fn serialize_request(&self, req: &NeutralRequest) -> Result<String, AdapterError>;
    fn parse_response(&self, body: &str) -> Result<NeutralResponse, AdapterError>;
    fn serialize_response(&self, resp: &NeutralResponse) -> Result<String, AdapterError>;
    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;       // upstream SSE -> NeutralStreamEvent
    fn stream_encoder(&self) -> Box<dyn StreamEncoder>;       // NeutralStreamEvent -> client SSE text
    fn serialize_error(&self, err: &AdapterError) -> (u16, String); // status + native error body
    fn models_path(&self) -> Option<&'static str>;            // model listing path, if any
    fn parse_models(&self, body: &str) -> Result<Vec<ModelInfo>, AdapterError>; // default: {"data":[...]}
    fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError>;
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
llmgate/
├── Cargo.toml              # axum, tokio, reqwest, serde, serde_json, anyhow, thiserror, tracing, toml
├── config.example.toml
├── PLAN.md
├── README.md
└── src/
    ├── main.rs             # bootstrap, registry assembly, router, shutdown, request-id tracing
    ├── config.rs           # TOML + env overrides
    ├── auth.rs             # per-protocol auth extraction (Bearer / api-key / x-api-key), constant-time
    ├── resolver.rs         # model pipeline (agnostic)
    ├── proxy.rs            # reqwest forwarding (streaming + non-streaming, timeout split)
    ├── core/               # protocol-agnostic core
    │   ├── mod.rs          # module wiring
    │   ├── registry.rs     # ProtocolAdapter trait, StreamDecoder/Encoder, registry
    │   ├── neutral.rs      # NeutralRequest / NeutralResponse / NeutralStreamEvent
    │   ├── pipeline.rs     # generic handler: parse -> resolve -> forward -> convert back
    │   ├── error.rs        # AdapterError (rate_limit, auth, overloaded, quota, api, ...)
    │   └── sse.rs          # SSE framing parser with bounded buffering
    └── adapters/
        ├── mod.rs          # registry impl, adapter wiring
        ├── openai/         # request/response serde, stream decoder+encoder, error map
        ├── anthropic/      # request/response serde, stream decoder+encoder, error map
        └── gemini/         # generateContent serde, stream decoder+encoder, error map
```

Unit tests live inline as `#[cfg(test)] mod tests` in each module — there is no
separate `tests/` directory.

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
  `signature_delta` on block switch → `message_delta` → `message_stop`) is
  entirely inside the anthropic adapter.

## Milestones

| # | Scope | Definition of done |
|---|-------|--------------------|
| M1 | Core framework | Done. Neutral model, adapter trait/registry, generic pipeline, config, health. |
| M2 | openai adapter | Done. Request/response conversion incl. tools, reasoning, images. |
| M3 | anthropic adapter | Done. Request/response conversion incl. tools, thinking, images, error mapping. |
| M4 | Streaming | Done. Stateful decoders/encoders both adapters; e2e verified both directions. |
| M5 | Service polish | Done. Auth middleware, `/v1/models` (per-protocol shapes), `/v1/messages/count_tokens`, route dedup. |
| M6 | Docs & hardening | Done. README + "adding a protocol" guide, config.example.toml, curl examples, clippy clean. |
| M7 | Reliability hardening | Done. Spec-faithful thinking blocks, timeout split, SSE keep-alive + proxy headers, bounded SSE buffering, constant-time auth, request-id tracing, JSON-as-stream fallback. |
| M8 | gemini adapter | Done. Upstream-side `generateContent` adapter: request/response conversion (tools, thinking, images), SSE stream decoder/encoder, error mapping, model listing, `streamGenerateContent` URL switching. |
| M9 | Privacy Guard | Done. Reversible redaction (`[privacy_guard]`): regex rules + allow-list, session-scoped token vault, streaming restore across fragmented SSE chunks (buffered Aho-Corasick, per channel), fail-closed config validation. |
| M10 | Memory substrate & observability | Planned. `[memory]` config (fail-closed), embedded nqlite store behind a write-behind actor, per-request records (model, usage, latency, status, request-id) for stream + non-stream paths, TTL purge, `nql-cli` admin queries. |
| M11 | Persistent PII vault | Planned. Opt-in nqlite vault backend for Privacy Guard (cross-request restore + crash recovery); all M9 invariants must stay green. |
| M12 | Response cache | Planned. Exact + BM25 fuzzy tier first (no embeddings); vector tier behind an `Embedder` trait; TTL/eviction; non-stream first, stream replay later. |
| M13 | Session context engine | Planned. `x-session-id` contract (stripped before upstream), turn graph (`:next`/`:mentions`), salience recall injected as token-budgeted system context. |
| M14 | Multi-upstream failover | Planned. Pre-first-token re-route to a fallback upstream; token-exact mid-stream resume experimental. |

### M1 details

- `Cargo.toml` deps: axum 0.8, tokio (full), reqwest (json/stream/rustls), serde,
  serde_json, anyhow, thiserror, tracing, tracing-subscriber, toml.
- `src/core/neutral.rs`: full neutral model types.
- `src/core/mod.rs`: `ProtocolAdapter`, `StreamDecoder`, `StreamEncoder`,
  `ProtocolRegistry` (register + get by name), `EndpointKind` (`Chat`,
  `Messages`, `Models`, `CountTokens`).
- `src/core/pipeline.rs`: `handle_conversation` generic over client adapter +
  upstream adapter — parse, resolve model, serialize for upstream, forward via
  `proxy.rs`, convert response back.
- `src/config.rs`: TOML schema with `[client]` protocols, `[upstream]`
  protocol/url/auth, `[models]` map; env overrides.
- `src/main.rs`: bootstrap registry from config, mount routes per client
  protocol's `endpoints()`, health at `/health`.

### M7 details (reliability hardening)

Brings the runtime in line with community/upstream standards:

- **Thinking block shape** — Anthropic `content_block_start` for thinking
  blocks carries a `thinking` field (not `text`), per the Anthropic streaming
  spec; verified by a regression test.
- **Timeout split** — `timeout_ms` is now enforced as a total per-request
  timeout on non-streaming calls (conversation + model listing). Streaming
  requests have no total timeout (bounded by the SSE protocol) so long
  generations aren't cut off; connect timeout fixed at 10s.
- **SSE keep-alive** — the gateway emits `: keepalive` comment lines on a 15s
  idle interval so long streams (thinking, tool calls) survive proxies and
  idle-dropping clients. `X-Accel-Buffering: no` + `Cache-Control: no-cache`
  prevent intermediary buffering.
- **Bounded SSE buffering** — `SseFraming` discards buffers past a 64 MiB cap
  instead of growing unbounded on a broken/malicious upstream.
- **Constant-time auth** — API-key comparison no longer leaks prefix/length
  timing.
- **Request-id correlation** — inbound `x-request-id` is honored (or generated),
  threaded into the tracing span, and echoed on the response.
- **JSON-as-stream fallback** — if the upstream answers `application/json`
  despite `stream: true` (some OpenAI-compatible servers ignore the flag), the
  body is converted into a single-event stream instead of an empty stream.

### M8 details (gemini adapter)

Upstream-side Gemini (`generateContent`) support. OpenAI/Anthropic clients can
now be routed to a Gemini provider:

- **Conversion** — `contents`/`systemInstruction`/`generationConfig` ↔ neutral
  request; `candidates`/`finishReason`/`usageMetadata` ↔ neutral response.
  Parts cover text, images (`inlineData`), thinking (`thought: true`),
  `functionCall` ↔ tool use, and `functionResponse` ↔ tool results.
- **Streaming** — `:streamGenerateContent?alt=sse` chunks decoded to neutral
  events (text/reasoning/tool deltas, finish + usage) and encoded back;
  no `[DONE]` terminator (protocol ends on finish chunk / connection close).
- **URL switching** — new `stream_conversation_url()` default method on the
  adapter trait (falls back to `conversation_url`); the pipeline selects the
  streaming variant when the client requested a stream.
- **Errors** — `{"error": {code, message, status}}` shape; HTTP status mapped
  to the neutral taxonomy (400→invalid, 401→auth, 403→denied, 429→rate limit,
  503→overloaded).
- **Models** — `/v1beta/models` with the `models/`-prefixed id scheme and
  `displayName`; re-serialized per client protocol.
- Client-side Gemini inbound (`/v1beta/models/{model}:generateContent` needs
  path-parameterized routing) is a follow-up milestone; this adapter serves as
  an upstream only for now.

### M9 details (privacy guard)

- **Config** — `[privacy_guard]` (enabled, vault, rules, allow_list);
  disabled by default. With `enabled = true` and no rules, a built-in
  conservative set runs (email, IPv4, US phone, API keys). Fail closed:
  invalid patterns, templates without `{n}`, or a non-`"memory"` vault
  abort startup.
- **Redaction** — `RedactionEngine` (compiled rules + allow-list) shared
  per process; `RedactionSession` per request, applied to every
  text-bearing content block (text, thinking, tool results, tool-call
  input JSON). Repeated values reuse one token per rule; 4096-token
  session cap.
- **Vault** — session-scoped in-memory token ↔ original map, dropped with
  the request; automaton cache rebuilt on new tokens.
- **Streaming restore** — `RestoreStream` buffers fragments and matches
  tokens with Aho-Corasick across SSE chunk boundaries; per-channel
  isolation (text / reasoning / tool arguments) via `StreamRestorer`;
  bounded hold-back so plain text never lags more than one token length.
  Token grammar (`<NAME_n>` with `>` terminator) guarantees no prefix
  collisions (`<IP_1>` never matches inside `<IP_10>`).
- **E2E verified** — mock upstream + curl: redacted upstream bodies,
  restored client responses for non-streaming, streaming (fragmented
  chunks), and the JSON-as-stream fallback; disabled mode is byte-for-byte
  passthrough.

## Verification

- `cargo test` — 165 unit tests: per-adapter conversion/stream tests (openai,
  anthropic, gemini) + core (sse framing incl. buffer cap, auth incl.
  constant-time, config, resolver) + privacy guard (redaction, allow-list,
  round-trips, fragmented stream restore, prefix-collision safety, token
  cap, fail-closed validation).
- `cargo clippy --all-targets -- -D warnings`
- `cargo fmt --check`
- Manual curl smoke tests for both protocols (stream + non-stream, tools,
  auth, models).

## Backlog

- Client-side Gemini inbound (`/v1beta/models/{model}:generateContent` needs
  path-parameterized routing) — follow-up to M8 (Gemini is upstream-only today).

## Out of scope (for now)

- Embeddings for generation / moderations / audio endpoints (sidecars etc.)
- Exact (non-heuristic) token counting, rate limiting, per-user budgets
- Warpgate-style egress proxy pools
- (Multi-upstream routing / failover and response caching are NOT out of
  scope — they land via M10–M14, the nqlite memory initiative. See
  `/root/workspace/llmgate-nqlite-scope.md`.)
