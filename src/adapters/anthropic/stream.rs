//! Anthropic SSE streaming: event decoder and event encoder.
//!
//! Upstream Anthropic events (`message_start`, `content_block_*`,
//! `message_delta`, `message_stop`, `ping`, `error`) are decoded into neutral
//! stream events; neutral events are encoded back into the Anthropic event
//! choreography.

use serde_json::{Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{FinishReason, NeutralStreamEvent, NeutralUsage};
use crate::core::registry::{StreamDecoder, StreamEncoder};

/// Stateful Anthropic event → neutral events decoder.
pub struct AnthropicStreamDecoder {
    started: bool,
    failed: bool,
    message_id: String,
    model: String,
    input_tokens: u64,
    stop_emitted: bool,
    pending_finish: Option<FinishReason>,
    pending_usage: Option<NeutralUsage>,
}

impl AnthropicStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            failed: false,
            message_id: String::new(),
            model: String::new(),
            input_tokens: 0,
            stop_emitted: false,
            pending_finish: None,
            pending_usage: None,
        }
    }
}

impl Default for AnthropicStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn feed(&mut self, data: &str) -> Vec<NeutralStreamEvent> {
        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let etype = match parsed.get("type").and_then(Value::as_str) {
            Some(t) => t,
            None => return Vec::new(),
        };

        match etype {
            "message_start" => {
                self.started = true;
                if let Some(message) = parsed.get("message") {
                    self.message_id = message
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.model = message
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.input_tokens = message
                        .get("usage")
                        .and_then(|u| u.get("input_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
                vec![NeutralStreamEvent::MessageStart {
                    id: self.message_id.clone(),
                    model: self.model.clone(),
                    // input_tokens is known at stream start; carry it so
                    // the encoder can report real accounting instead of 0.
                    usage: Some(NeutralUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: 0,
                    }),
                }]
            }
            "content_block_start" => {
                let index = parsed.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                let btype = parsed
                    .get("content_block")
                    .and_then(|b| b.get("type"))
                    .and_then(Value::as_str);
                match btype {
                    Some("tool_use") => {
                        let id = parsed
                            .get("content_block")
                            .and_then(|b| b.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let name = parsed
                            .get("content_block")
                            .and_then(|b| b.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        vec![NeutralStreamEvent::ToolCallDelta {
                            index,
                            id,
                            name,
                            arguments: String::new(),
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            "content_block_delta" => {
                let index = parsed.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                let delta = parsed.get("delta");
                let dtype = delta.and_then(|d| d.get("type")).and_then(Value::as_str);
                match dtype {
                    Some("text_delta") => delta
                        .and_then(|d| d.get("text"))
                        .and_then(Value::as_str)
                        .map(|t| vec![NeutralStreamEvent::TextDelta(t.to_string())])
                        .unwrap_or_default(),
                    Some("thinking_delta") => delta
                        .and_then(|d| d.get("thinking"))
                        .and_then(Value::as_str)
                        .map(|t| vec![NeutralStreamEvent::ReasoningDelta(t.to_string())])
                        .unwrap_or_default(),
                    Some("input_json_delta") => delta
                        .and_then(|d| d.get("partial_json"))
                        .and_then(Value::as_str)
                        .map(|j| {
                            // The delta's own index is authoritative: each
                            // content_block_delta targets a specific block.
                            vec![NeutralStreamEvent::ToolCallDelta {
                                index,
                                id: String::new(),
                                name: String::new(),
                                arguments: j.to_string(),
                            }]
                        })
                        .unwrap_or_default(),
                    // The signature is opaque but REQUIRED when an
                    // extended-thinking conversation continues: carry it
                    // through so the encoder can echo it verbatim.
                    Some("signature_delta") => delta
                        .and_then(|d| d.get("signature"))
                        .and_then(Value::as_str)
                        .map(|s| vec![NeutralStreamEvent::ReasoningSignature(s.to_string())])
                        .unwrap_or_default(),
                    _ => Vec::new(),
                }
            }
            "message_delta" => {
                if let Some(reason) = parsed
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.pending_finish = Some(parse_stop_reason(reason));
                }
                if let Some(usage) = parsed.get("usage") {
                    let output = usage
                        .get("output_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    self.pending_usage = Some(NeutralUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: output,
                    });
                }
                Vec::new()
            }
            "message_stop" => {
                if !self.stop_emitted {
                    self.stop_emitted = true;
                    vec![NeutralStreamEvent::MessageStop {
                        finish_reason: self.pending_finish.clone().unwrap_or(FinishReason::Stop),
                        usage: self.pending_usage,
                    }]
                } else {
                    Vec::new()
                }
            }
            "ping" => Vec::new(),
            "error" => {
                self.failed = true;
                let message = parsed
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("upstream stream error")
                    .to_string();
                vec![NeutralStreamEvent::Error(AdapterError::Api(message))]
            }
            _ => Vec::new(),
        }
    }

    fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        // After an error the stream must not synthesize a successful stop:
        // the client needs to see the stream as failed.
        if !self.started || self.failed {
            return Vec::new();
        }
        if !self.stop_emitted {
            self.stop_emitted = true;
            vec![NeutralStreamEvent::MessageStop {
                finish_reason: self.pending_finish.clone().unwrap_or(FinishReason::Stop),
                usage: self.pending_usage,
            }]
        } else {
            Vec::new()
        }
    }
}

fn parse_stop_reason(s: &str) -> FinishReason {
    match s {
        "end_turn" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "stop_sequence" => FinishReason::Stop,
        "refusal" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Stateful neutral events → Anthropic events encoder.
///
/// Maintains the open content block (text/thinking/tool_use) so it can emit
/// `content_block_start`/`content_block_stop` (and `signature_delta` when a
/// thinking block closes) at the right points.
pub struct AnthropicStreamEncoder {
    block_index: u32,
    current_block: Option<BlockKind>,
    message_started: bool,
    /// Opaque thinking signature received via `ReasoningSignature`; emitted
    /// verbatim when the thinking block closes (Anthropic requires it to
    /// continue an extended-thinking conversation).
    pending_signature: Option<String>,
    /// Neutral tool-call indices whose `content_block_start` was emitted,
    /// so an id-less first fragment still opens the block (a later
    /// `input_json_delta` must reference an opened block). Pruned when the
    /// block closes, so a stream that reuses an index opens a fresh block.
    tool_blocks_open: std::collections::HashSet<u32>,
    /// Neutral index of the currently open tool-use block (for pruning).
    open_tool_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
}

impl AnthropicStreamEncoder {
    pub fn new() -> Self {
        Self {
            block_index: 0,
            current_block: None,
            message_started: false,
            pending_signature: None,
            tool_blocks_open: std::collections::HashSet::new(),
            open_tool_index: None,
        }
    }
}

impl Default for AnthropicStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamEncoder for AnthropicStreamEncoder {
    fn encode(&mut self, event: NeutralStreamEvent) -> Vec<String> {
        let mut lines = Vec::new();
        match event {
            NeutralStreamEvent::MessageStart { id, model, usage } => {
                self.message_started = true;
                lines.push(format!(
                    "event: message_start\ndata: {}\n\n",
                    json!({
                        "type": "message_start",
                        "message": {
                            "id": id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {
                                "input_tokens": usage.map(|u| u.input_tokens).unwrap_or(0),
                                "output_tokens": 0,
                            },
                        },
                    })
                ));
            }
            NeutralStreamEvent::TextDelta(t) => {
                self.ensure_block(BlockKind::Text, &mut lines);
                lines.push(format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    json!({
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": { "type": "text_delta", "text": t },
                    })
                ));
            }
            NeutralStreamEvent::ReasoningDelta(t) => {
                self.ensure_block(BlockKind::Thinking, &mut lines);
                lines.push(format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    json!({
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": { "type": "thinking_delta", "thinking": t },
                    })
                ));
            }
            NeutralStreamEvent::ReasoningSignature(signature) => {
                // Stored until the thinking block closes; emitted verbatim
                // in the closing signature_delta.
                self.pending_signature = Some(signature);
            }
            NeutralStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                if !self.tool_blocks_open.contains(&index) {
                    // New tool-use block: close any open block, start the
                    // tool_use block at our own sequential index. The block
                    // opens even when this fragment carries no id yet — a
                    // later input_json_delta must never reference a block
                    // that was never started.
                    self.close_block(&mut lines);
                    let block_index = self.block_index;
                    lines.push(format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {},
                            },
                        })
                    ));
                    self.current_block = Some(BlockKind::ToolUse);
                    self.tool_blocks_open.insert(index);
                    self.open_tool_index = Some(index);
                }
                if !arguments.is_empty() {
                    lines.push(format!(
                        "event: content_block_delta\ndata: {}\n\n",
                        json!({
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": { "type": "input_json_delta", "partial_json": arguments },
                        })
                    ));
                }
            }
            NeutralStreamEvent::MessageStop {
                finish_reason,
                usage,
            } => {
                self.close_block(&mut lines);
                let stop_reason: String = match finish_reason {
                    FinishReason::Stop => "end_turn".to_string(),
                    FinishReason::Length => "max_tokens".to_string(),
                    FinishReason::ToolCalls => "tool_use".to_string(),
                    FinishReason::ContentFilter => "refusal".to_string(),
                    FinishReason::Other(s) => s,
                };
                lines.push(format!(
                    "event: message_delta\ndata: {}\n\n",
                    json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": &stop_reason },
                        "usage": {
                            "output_tokens": usage.map(|u| u.output_tokens).unwrap_or(0),
                        },
                    })
                ));
                lines
                    .push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string());
            }
            NeutralStreamEvent::Error(e) => {
                self.close_block(&mut lines);
                lines.push(format!(
                    "event: error\ndata: {}\n\n",
                    json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": e.to_string() },
                    })
                ));
            }
        }
        let _ = self.message_started;
        lines
    }

    fn done(&self) -> String {
        // Anthropic streams end after message_stop; no terminator line.
        String::new()
    }
}

impl AnthropicStreamEncoder {
    /// Open a text/thinking block if none is open; switch blocks otherwise.
    fn ensure_block(&mut self, kind: BlockKind, lines: &mut Vec<String>) {
        if self.current_block == Some(kind) {
            return;
        }
        self.close_block(lines);
        let btype = match kind {
            BlockKind::Text => "text",
            BlockKind::Thinking => "thinking",
            BlockKind::ToolUse => "tool_use",
        };
        // Thinking blocks carry a `thinking` field, not `text` (per the
        // Anthropic streaming spec).
        let content_block = match kind {
            BlockKind::Thinking => json!({ "type": btype, "thinking": "" }),
            _ => json!({ "type": btype, "text": "" }),
        };
        lines.push(format!(
            "event: content_block_start\ndata: {}\n\n",
            json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": content_block,
            })
        ));
        self.current_block = Some(kind);
    }

    fn close_block(&mut self, lines: &mut Vec<String>) {
        if let Some(kind) = self.current_block.take() {
            if kind == BlockKind::Thinking {
                // The signature is only emitted when one was actually
                // received: an EMPTY signature would be rejected by
                // Anthropic if the client later continues an extended-
                // thinking conversation, and thinking streamed from
                // providers without signatures must not fabricate one.
                if let Some(signature) = self.pending_signature.take() {
                    lines.push(format!(
                        "event: content_block_delta\ndata: {}\n\n",
                        json!({
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": { "type": "signature_delta", "signature": signature },
                        })
                    ));
                }
            }
            if kind == BlockKind::ToolUse {
                // A closed tool block releases its neutral index: a stream
                // that reuses an index must open a FRESH block, not stream
                // deltas against the closed one.
                if let Some(index) = self.open_tool_index.take() {
                    self.tool_blocks_open.remove(&index);
                }
            }
            lines.push(format!(
                "event: content_block_stop\ndata: {}\n\n",
                json!({ "type": "content_block_stop", "index": self.block_index })
            ));
            self.block_index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(_name: &str, payload: Value) -> String {
        // Mirrors the pipeline: SseFraming extracts the `data:` payload
        // before it reaches the decoder.
        payload.to_string()
    }

    #[test]
    fn error_chunk_suppresses_synthetic_stop() {
        // An upstream error must not be followed by a synthesized
        // successful stop (the client would see error + normal completion).
        let mut d = AnthropicStreamDecoder::new();
        d.feed(&event("message_start", json!({
            "type": "message_start",
            "message": { "id": "msg_1", "model": "claude", "usage": { "input_tokens": 5, "output_tokens": 0 } },
        })));
        let events = d.feed(&event(
            "error",
            json!({
                "type": "error",
                "error": { "type": "api_error", "message": "boom" },
            }),
        ));
        assert!(
            matches!(&events[0], NeutralStreamEvent::Error(e) if e.to_string().contains("boom"))
        );
        assert!(d.finish().is_empty(), "no stop after an error");
    }

    #[test]
    fn message_start_carries_input_tokens() {
        let mut d = AnthropicStreamDecoder::new();
        let mut events = d.feed(&event("message_start", json!({
            "type": "message_start",
            "message": { "id": "msg_1", "model": "claude", "usage": { "input_tokens": 42, "output_tokens": 0 } },
        })));
        match &events[0] {
            NeutralStreamEvent::MessageStart { usage, .. } => {
                assert_eq!(usage.as_ref().map(|u| u.input_tokens), Some(42));
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }
        // The encoder reports the real count instead of 0.
        let mut e = AnthropicStreamEncoder::new();
        let lines = e.encode(events.remove(0));
        assert!(lines[0].contains("\"input_tokens\":42"), "{}", lines[0]);
    }

    #[test]
    fn signature_delta_round_trips_through_encoder() {
        let mut d = AnthropicStreamDecoder::new();
        let events = d.feed(&event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "signature_delta", "signature": "sig-abc-123" },
            }),
        ));
        assert_eq!(
            events,
            vec![NeutralStreamEvent::ReasoningSignature("sig-abc-123".into())]
        );
        let mut e = AnthropicStreamEncoder::new();
        e.encode(NeutralStreamEvent::MessageStart {
            id: "m1".into(),
            model: "claude".into(),
            usage: None,
        });
        e.encode(NeutralStreamEvent::ReasoningDelta("hmm".into()));
        e.encode(NeutralStreamEvent::ReasoningSignature("sig-abc-123".into()));
        let lines = e.encode(NeutralStreamEvent::TextDelta("answer".into()));
        let joined = lines.join("");
        assert!(
            joined.contains("\"signature\":\"sig-abc-123\""),
            "signature must be echoed verbatim: {joined}"
        );
    }

    #[test]
    fn id_less_first_tool_fragment_still_opens_block() {
        // A first ToolCallDelta without an id must still open the block, so
        // the following input_json_delta references a started block.
        let mut e = AnthropicStreamEncoder::new();
        e.encode(NeutralStreamEvent::MessageStart {
            id: "m1".into(),
            model: "claude".into(),
            usage: None,
        });
        let lines = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(), // id-less first fragment
            name: String::new(),
            arguments: "{\"city\": \"paris\"}".into(),
        });
        let joined = lines.join("\n");
        assert!(
            joined.contains("content_block_start"),
            "block must open even without an id: {joined}"
        );
        assert!(
            joined.contains("input_json_delta"),
            "arguments must stream against the opened block: {joined}"
        );
    }

    #[test]
    fn reused_tool_index_reopens_a_fresh_block() {
        // A stream that reuses a neutral tool index after its block closed
        // must open a NEW content_block (the pruned register otherwise
        // swallows the second call's content_block_start).
        let mut e = AnthropicStreamEncoder::new();
        e.encode(NeutralStreamEvent::MessageStart {
            id: "m1".into(),
            model: "claude".into(),
            usage: None,
        });
        e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            arguments: "{\"a\":1}".into(),
        });
        e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::ToolCalls,
            usage: None,
        });
        let second = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0, // reused index
            id: "call_2".into(),
            name: "g".into(),
            arguments: "{\"b\":2}".into(),
        });
        let joined = second.join("\n");
        assert!(
            joined.contains("content_block_start"),
            "reused index must open a fresh block: {joined}"
        );
        assert!(joined.contains("\"id\":\"call_2\""), "{joined}");
    }

    #[test]
    fn decodes_anthropic_stream_sequence() {
        let mut d = AnthropicStreamDecoder::new();
        let events = d.feed(&event("message_start", json!({
            "type": "message_start",
            "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
                         "content": [], "stop_reason": null, "stop_sequence": null,
                         "usage": {"input_tokens": 5, "output_tokens": 0}},
        })));
        assert!(
            matches!(&events[0], NeutralStreamEvent::MessageStart { id, model, .. } if id == "msg_1" && model == "claude")
        );

        assert!(
            d.feed(&event(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            ))
            .is_empty()
        );

        let events = d.feed(&event(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "hmm"},
            }),
        ));
        assert_eq!(
            events,
            vec![NeutralStreamEvent::ReasoningDelta("hmm".into())]
        );

        let events = d.feed(&event(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "answer"},
            }),
        ));
        assert_eq!(events, vec![NeutralStreamEvent::TextDelta("answer".into())]);

        assert!(
            d.feed(&event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7},
                })
            ))
            .is_empty()
        );

        let events = d.feed(&event("message_stop", json!({"type": "message_stop"})));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStop { finish_reason: FinishReason::Stop, usage: Some(u) }
                if u.input_tokens == 5 && u.output_tokens == 7
        ));

        // finish() must not double-emit.
        assert!(d.finish().is_empty());
    }

    #[test]
    fn decodes_tool_use_blocks() {
        let mut d = AnthropicStreamDecoder::new();
        d.feed(&event("message_start", json!({
            "type": "message_start",
            "message": {"id": "m", "model": "claude", "usage": {"input_tokens": 1, "output_tokens": 0}},
        })));
        let events = d.feed(&event("content_block_start", json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {}},
        })));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, id, name, .. } if id == "tu_1" && name == "get_weather"
        ));
        let events = d.feed(&event(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"city\":\"paris\"}"},
            }),
        ));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, arguments, .. } if arguments == "{\"city\":\"paris\"}"
        ));
    }

    #[test]
    fn decodes_ping_and_error() {
        let mut d = AnthropicStreamDecoder::new();
        assert!(d.feed(&event("ping", json!({"type": "ping"}))).is_empty());
        let events = d.feed(&event(
            "error",
            json!({
                "type": "error", "error": {"type": "overloaded_error", "message": "busy"},
            }),
        ));
        assert!(
            matches!(&events[0], NeutralStreamEvent::Error(e) if e.to_string().contains("busy"))
        );
    }

    #[test]
    fn finish_emits_stop_without_message_stop() {
        let mut d = AnthropicStreamDecoder::new();
        d.feed(&event(
            "message_start",
            json!({
                "type": "message_start", "message": {"id": "m", "model": "claude"},
            }),
        ));
        d.feed(&event(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            }),
        ));
        let events = d.finish();
        assert!(matches!(&events[0], NeutralStreamEvent::MessageStop { .. }));
    }

    #[test]
    fn encodes_sequence_with_block_switching() {
        let mut e = AnthropicStreamEncoder::new();
        let mut all: Vec<String> = Vec::new();
        all.extend(e.encode(NeutralStreamEvent::MessageStart {
            id: "msg_1".into(),
            model: "claude".into(),
            usage: None,
        }));
        assert!(all[0].contains("event: message_start"));

        all.extend(e.encode(NeutralStreamEvent::ReasoningDelta("think".into())));
        all.extend(e.encode(NeutralStreamEvent::ReasoningDelta(" more".into())));
        all.extend(e.encode(NeutralStreamEvent::TextDelta("answer".into())));
        all.extend(e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::Stop,
            usage: None,
        }));

        let joined = all.join("");
        assert!(joined.contains("content_block_start"));
        assert!(joined.contains("\"type\":\"thinking\""));
        // Thinking blocks must carry a `thinking` field (not `text`), per the
        // Anthropic streaming spec. Keys are alphabetically sorted by serde.
        let thinking_block_start = joined.find("content_block_start").unwrap();
        let window = &joined[thinking_block_start..];
        // The thinking block's own section ends at its content_block_stop.
        let window_end = window.find("content_block_stop").unwrap_or(window.len());
        let thinking_block = &joined[thinking_block_start..thinking_block_start + window_end];
        assert!(
            thinking_block.contains("\"thinking\""),
            "thinking block should carry a thinking field, got: {thinking_block}"
        );
        assert!(
            !thinking_block.contains("\"text\""),
            "thinking block must NOT carry a text field, got: {thinking_block}"
        );
        // Block switch: thinking -> text closes the thinking block. No
        // signature_delta is emitted because no signature was received
        // (fabricating an empty one would break extended-thinking
        // continuations); the block still closes with content_block_stop.
        assert!(!joined.contains("signature_delta"));
        assert!(joined.contains("\"type\":\"text_delta\""));
        assert!(joined.contains("message_delta"));
        assert!(joined.contains("\"stop_reason\":\"end_turn\""));
        assert!(joined.contains("message_stop"));

        // Block indices are sequential: thinking=0, text=1.
        let first_stop_pos = joined.find("content_block_stop").unwrap();
        let text_start_pos = joined.rfind("content_block_start").unwrap();
        assert!(first_stop_pos < text_start_pos);
    }

    #[test]
    fn encodes_tool_use_and_arguments() {
        let mut e = AnthropicStreamEncoder::new();
        let mut all: Vec<String> = Vec::new();
        all.extend(e.encode(NeutralStreamEvent::MessageStart {
            id: "m".into(),
            model: "c".into(),
            usage: None,
        }));
        all.extend(e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "tu_1".into(),
            name: "f".into(),
            arguments: String::new(),
        }));
        all.extend(e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "{\"a\":1}".into(),
        }));
        all.extend(e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::ToolCalls,
            usage: None,
        }));
        let joined = all.join("");
        assert!(joined.contains("\"type\":\"tool_use\""));
        assert!(joined.contains("\"id\":\"tu_1\""));
        assert!(joined.contains("input_json_delta"));
        assert!(joined.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn done_is_empty() {
        let e = AnthropicStreamEncoder::new();
        assert_eq!(e.done(), "");
    }
}
