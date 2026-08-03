# model-adapter

Protocol-agnostic LLM API adapter gateway in Rust.

A reverse-proxy gateway that adapts between **LLM provider protocols** in both
directions — OpenAI-compatible clients can talk to Anthropic-compatible
upstreams and vice versa, with new protocols (Gemini, OpenAI Responses API,
Ollama, Bedrock, ...) added as pluggable adapters without touching the core.

## Status

Early development. See [PLAN.md](PLAN.md) for the full plan and milestone
tracking. Currently at: **M1 — core framework** (in progress).

## Design in one line

The core works on a protocol-neutral internal model (`NeutralRequest` /
`NeutralResponse` / `NeutralStreamEvent`); every protocol is a `ProtocolAdapter`
plugin implementing request/response serialization, SSE stream
decode/encode, and error mapping.

```
OpenAI client ──▶ /v1/chat/completions ──┐
                                         ├─▶ model-adapter ──▶ upstream (any protocol, config-driven)
Anthropic client ──▶ /v1/messages ───────┘                  ◀── responses converted back to client format
```

## Getting started

(Coming with M5 — pre-alpha.)

```bash
cargo build
cargo test
```

## Milestones

- [x] Repo bootstrap (this repository)
- [ ] M1 — Core framework (neutral model, adapter trait, registry, pipeline, config)
- [ ] M2 — OpenAI adapter (non-stream)
- [ ] M3 — Anthropic adapter (non-stream)
- [ ] M4 — SSE streaming (both adapters)
- [ ] M5 — Auth, /v1/models, polish
- [ ] M6 — Docs & hardening

## License

TBD
