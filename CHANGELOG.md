# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Observability store (M10)** — optional `[memory]` section: every request
  is recorded as a JSON document (model, protocols, mode, status, latency,
  token usage, request-id) in an embedded nqlite database behind a
  write-behind actor; WAL checkpoint cadence and TTL sweep are configurable;
  queryable offline with `nql-cli`. Disabled by default; fail-closed
  validation (`enabled` requires `path`; unopenable database aborts startup).
- **Community-standard docs** — `CODE_OF_CONDUCT.md` (Contributor Covenant
  2.1), GitHub issue templates (bug report, feature request), and a pull
  request template; README now links the security policy and code of conduct.
- **Gemini adapter (upstream)** — `generateContent` request/response conversion
  (`systemInstruction`, `contents`/`parts`, `generationConfig`), including
  `inlineData` images, `thought` thinking parts, and
  `functionCall`/`functionResponse` tool blocks. Streaming via
  `streamGenerateContent?alt=sse` with no `[DONE]` terminator; finish chunk
  carries stop reason + usage. Model listing via `/v1beta/models` with
  `models/`-prefixed ids. Protocol-native error mapping.
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
- **Privacy Guard** — reversible, session-scoped redaction
  (`[privacy_guard]`): regex rules + allow-list replace sensitive entities
  (PII, IPs, secrets) with `<NAME_n>` tokens before the request is
  forwarded; responses — streaming and non-streaming — restore the original
  values transparently. Streaming restore uses a buffered Aho-Corasick
  matcher per channel, safe across arbitrary SSE chunk boundaries. Fail
  closed on invalid config; session-scoped in-memory vault (persistent
  backends are a future extension).
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

- **Project renamed `model-adapter` → `llmgate`** — package/binary name,
  repository URL (`github.com/devstroop/llmgate`), and all docs
  (README/CHANGELOG/CONTRIBUTING/SECURITY/PLAN/config.example) standardized
  on the new name.
- **Adapter trait** — new `stream_conversation_url()` default method lets
  protocols with distinct streaming endpoints (e.g. Gemini
  `:streamGenerateContent?alt=sse`) switch URLs per request; defaults to
  `conversation_url()`.
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
- **Gemini upstream URL injection** — the client-supplied model name is now
  validated before it becomes the URL path segment (`conversation_url` /
  `stream_conversation_url` are fallible): query/fragment characters and
  `.`/`..` segments are rejected with a 400 instead of being forwarded to
  the provider. `conversation_url` on the adapter trait now returns
  `Result` so path-parameterized protocols must handle validation.
- **Gemini tool round-trip** — `functionResponse` now carries the function
  NAME resolved from the conversation history (correlation is by name on
  Gemini, not by call id), and id-less `functionCall` parts (the classic
  Gemini API sends none) get deterministic synthesized ids (`fc_1`, ...) in
  both body and stream paths so clients can correlate results.
- **JSON-as-stream fallback for real upstream shapes** — when an upstream
  ignores `stream: true` and replies with a standard non-stream body
  (`message`-shaped), the fallback previously produced an empty/content-less
  stream (stream decoders only read delta shapes). It now parses the body
  with `parse_response` and synthesizes a single-event stream; chunk-shaped
  JSON is still handled by the stream decoder first.
- **Streaming restore panic on multi-byte UTF-8** — the privacy restorer's
  hold-back window could land inside a CJK/emoji character and panic;
  `window_start` is now rounded down to a character boundary.
- **Privacy Guard config hardening** — replacement templates must match the
  documented `<NAME_{n}>` grammar (rejects `TOK{n}`-style templates whose
  tokens would collide or break streaming restore), duplicate templates
  across rules are rejected at startup, and a mint-time collision guard
  never overwrites an existing token mapping.
- **Streamed tool-argument restore is JSON-safe** — originals spliced into
  the raw-JSON tool-arguments channel are JSON-escaped, matching the
  non-streaming path; empty held-back deltas are no longer emitted.
- **Auth hardening** — API-key comparison no longer early-exits on length
  (length difference folded into the comparison); `Authorization` is parsed
  as a scheme + credentials (case-insensitive `Bearer` only), so other
  schemes can no longer shadow a valid `api-key`/`x-api-key` header.
- **Fail-closed config path** — a `CONFIG_PATH` pointing at a missing file
  is now a startup error instead of silently falling back to the
  unauthenticated default config.
- **Upstream error chunks surface in streams** — Gemini and OpenAI stream
  decoders emit `Error` events for `{"error": ...}` payloads instead of a
  spurious start + fake successful stop; Gemini's `MessageStop` is emitted
  exactly once, and Gemini's stream encoder buffers partial tool-call
  arguments until they form valid JSON instead of discarding them as null.
- **Anthropic empty responses** — `content` serializes as an empty array
  (the spec shape), never a bare string.
- **Stream error semantics** — an upstream transport failure or idle
  timeout now ends the client stream WITHOUT `[DONE]` (or a synthesized
  stop), so clients treat it as truncated rather than successful; decoders
  and restorers are flushed exactly once per stream. Streaming upstream
  reads are bounded by a 120s per-chunk idle timeout.
- **SSE framing** — oversized chunks are rejected before copying (the 64 MiB
  cap now bounds peak memory) and non-UTF-8 data lines log a rate-limited
  warning instead of vanishing silently.
- **Observability/ops** — `/health` is excluded from the auth middleware
  (load-balancer probes work with auth enabled); the request-id tracing span
  is active during handler execution; duplicate route detection keys on
  (method, path); `config.toml`/`config.local.toml` are gitignored.
- **Privacy Guard fails closed at the token cap** — a request that would
  exceed the 4096-token session cap is now REJECTED instead of forwarding
  the remaining matches unredacted; the core guarantee ("the provider never
  sees sensitive data") is unconditional again. Redaction works on a clone,
  so a failed request is never left partially redacted; a mint-time token
  collision (defense in depth on top of the build-time checks) is a request
  error, never a silent pass-through.
- **Strict token grammar** — replacement templates must be exactly
  `<NAME_{n}>` (single leading `<`, `[A-Za-z0-9_]` name, single `{n}`, no
  interior `<`/`>`), and template pairs whose token names differ only by a
  digit suffix (`<X_{n}>` vs `<X_1{n}>`) are rejected at startup.
- **Auth fail-closed hardening** — empty/whitespace-only API keys are
  rejected at startup (`api_keys = [""]` would have accepted keyless
  requests); requests with no credential at all are always unauthorized
  while auth is enabled; the constant-time comparison is bounded by the
  longest CONFIGURED key, so an attacker cannot drive CPU usage or a length
  oracle with a huge bearer token; `extra_headers` entries require both
  `name` and `value`.
- **Streaming connection hygiene** — the stream task aborts as soon as the
  client disconnects (`tx.closed()` in the read select) instead of holding
  the upstream connection until the idle timeout; the pre-headers phase of a
  streaming upstream request is bounded by a 15s timeout (504 to the
  client); `framing.finish()` payloads get the same `[DONE]` treatment as
  the main loop; `decoder.finish()` events are restored before encoding;
  `saw_error` is only set when the error event actually produced output, in
  both the streaming and fallback paths.
- **Startup fail-closed for wiring** — unknown CLIENT protocols are now a
  startup error (like upstream), and two distinct client protocols claiming
  the same method+path (which would serve one schema to the other's
  clients) abort startup.
- **Anthropic streamed reasoning fidelity** — streamed `message_start` now
  reports the real `input_tokens`; `signature_delta` is carried through a
  new neutral `ReasoningSignature` event and echoed verbatim when the
  thinking block closes (an extended-thinking conversation can continue);
  a tool block is opened on the first fragment even when the id is not yet
  known (no more deltas against an unopened block); an upstream `error`
  event no longer synthesizes a successful stop.
- **Anthropic mixed tool-result messages** — a user message mixing text and
  `tool_result` blocks is split into Tool + User messages so neither
  serializer drops the text (previously the whole message was relabeled
  Tool and the text vanished).
- **Image-bearing responses survive conversion** — OpenAI and Anthropic
  response serializers emit `image_url` parts / `image` blocks instead of
  silently dropping model-generated images; Gemini chunks always use
  candidate index 0 for tool-call parts (the tool index is not a candidate
  index).
- **Request-id hardening** — generated ids include a process-wide atomic
  counter (no collisions under concurrent load when the clock does not
  advance) and the response header never panics.
- **Credential headers are stripped before forwarding** — after a
  successful auth check the `Authorization`/`api-key`/`x-api-key` headers
  are removed from the request, so no downstream path (a header-copying
  adapter, a redirect, a logged request) can disclose the gateway key to
  the upstream provider.
- **Config validation extended (fail closed at startup)** — API keys must
  be presentable as HTTP header values and match their trimmed form
  (a key that could never authenticate is refused, not a 401 trap);
  `upstream.url` must be an absolute http(s) URL; `extra_headers` names
  must be valid HTTP header names; empty `models.prefixes` entries are
  rejected. Starting with `api_keys = []` on a non-loopback bind prints a
  prominent OPEN PROXY warning.
- **Model prefix resolution is longest-match** — overlapping prefixes
  (`vendor/` vs `vendor/gpt/`) resolve by longest prefix regardless of
  declaration order.
- **Streaming terminal-ordering fixed** — MessageStop events (mid-stream
  and `decoder.finish()`) are buffered and encoded only AFTER the privacy
  restorer's held tail is flushed, in both the streaming and JSON-as-stream
  paths: no content is delivered after the terminal event.
- **Privacy restore covers upstream errors** — upstream error bodies and
  SSE error chunks are run through the session restorer, so a provider
  echoing the redacted request cannot leak `<EMAIL_1>` tokens to the
  client.
- **Gemini tool results round-trip losslessly** — `functionResponse`
  parts inside user messages split into Tool messages (no more silent
  drops when routed to OpenAI/Anthropic); string responses are no longer
  JSON-quoted; the gateway's own `{"result", "is_error"}` wrapper unwraps
  on re-parse and the error flag survives routing.
- **Anthropic encoder fidelity** — a `signature_delta` is emitted only
  when a real signature was received (no more fabricated empty signatures
  that would break extended-thinking continuations), and closed tool
  blocks release their neutral index so a reused index opens a fresh
  block.
- **Gemini streamed errors carry real status codes** — 400/401/403/429/503
  variants map to their proper `code`/`status` instead of a blanket
  500/INTERNAL.
- **Upstream redirects disabled** — reqwest's default policy forwards
  provider-specific credential headers (`x-api-key`, `x-goog-api-key`) to
  redirect targets; `Policy::none()` closes that exfiltration path.
- **SSE framing hardened** — line processing is linear-time (no
  per-line buffer drain), and event data accumulates in a single byte
  buffer so a flood of tiny `data:` lines cannot defeat the byte cap with
  allocator overhead.
- **Route mounting uses one MethodRouter per path** — axum registers by
  path (methods merge inside a single `MethodRouter`); mounting the same
  path twice would replace the first registration instead of merging.
- **Sequential rule corruption** — tokens minted by an earlier rule are
  shielded from later rules (a broad `\d+`/`\w+` pattern can no longer
  re-match inside `<EMAIL_1>` and break restore); custom patterns that can
  match the empty string (e.g. `\b`, `x*`) are rejected at startup.
- **Anthropic tool results survive conversion** — `tool_result`-bearing
  `role: "user"` messages normalize to the Tool role so OpenAI/Gemini
  serializers emit them instead of dropping them (previously the tool loop
  broke); OpenAI serialization emits one `tool` message per result and
  preserves text/image interleaving order in multimodal messages.
- **Gemini prefixed model ids** — `models/…` and `tunedModels/…` names are
  treated as full resource paths (no more `models/models/…` URLs);
  name-keyed `functionResponse` parts (classic Gemini) correlate back to the
  call id via a reverse map.
- **OpenAI `max_completion_tokens`** — the field is parsed and re-emitted as
  the client sent it (reasoning models reject `max_tokens`); other protocols
  use it as the token limit when `max_tokens` is absent.
- **Redacted thinking round-trips verbatim** — Anthropic `redacted_thinking`
  blocks keep their opaque `data` payload (new neutral `RedactedThinking`
  block) instead of collapsing to an empty thinking block.
- **Bounded upstream reads** — upstream response bodies are read through a
  16 MiB cap on all paths (non-stream, model list, stream error, fallback)
  so a broken upstream cannot exhaust gateway memory.
- **SSE event-data cap** — the 64 MiB cap now also bounds events assembled
  from many newline-terminated `data:` lines without a blank line.
- **Auth/config fail-closed** — 401 responses carry `WWW-Authenticate:
  Bearer` (RFC 6750); key-list comparison no longer short-circuits (no match
  position timing leak); unknown config keys (e.g. `api_key` for `api_keys`)
  abort startup; the default bind is loopback (`127.0.0.1`); an unregistered
  upstream protocol is a startup error; a config-less start logs a prominent
  "authentication is DISABLED" warning.
- **Streaming edge cases** — the JSON-as-stream fallback assigns contiguous
  tool-call indices (not content-block positions); `saw_error` is set only
  after an error event is actually delivered; the first keep-alive waits for
  the idle interval (no spurious leading comment).

### Security

- API-key comparison is constant-time; timing no longer reveals key
  length/prefix structure.
- Inbound request bodies are capped at 16 MiB; upstream response bodies and
  SSE framing buffers are capped at 64 MiB.
- Privacy Guard exhausts its token cap fail-closed (request rejected), never
  forwarding unredacted data.

[Unreleased]: https://github.com/devstroop/llmgate/compare/main...develop
