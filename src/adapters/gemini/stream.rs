//! Gemini SSE streaming: chunk decoder and chunk encoder.
//!
//! Gemini's `streamGenerateContent?alt=sse` returns a stream of
//! `GenerateContentResponse` chunks, one per SSE `data:` line. There is no
//! `[DONE]` marker — the stream ends when the connection closes or after a
//! chunk carrying `finishReason`/`usageMetadata`.

use serde_json::{Map, Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{FinishReason, NeutralStreamEvent, NeutralUsage};
use crate::core::registry::{StreamDecoder, StreamEncoder};

/// Stateful Gemini chunk → neutral events decoder.
pub struct GeminiStreamDecoder {
    started: bool,
    failed: bool,
    stop_emitted: bool,
    pending_finish: Option<FinishReason>,
    pending_usage: Option<NeutralUsage>,
    tool_index: u32,
}

impl GeminiStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            failed: false,
            stop_emitted: false,
            pending_finish: None,
            pending_usage: None,
            tool_index: 0,
        }
    }
}

impl Default for GeminiStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder for GeminiStreamDecoder {
    fn feed(&mut self, data: &str) -> Vec<NeutralStreamEvent> {
        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        // Upstream error chunks (`{"error": {...}}`) must surface as an
        // Error event, not be swallowed into a fake empty success.
        if let Some(error) = parsed.get("error") {
            self.failed = true;
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("upstream stream error")
                .to_string();
            return vec![NeutralStreamEvent::Error(AdapterError::Api(message))];
        }

        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(NeutralStreamEvent::MessageStart {
                id: String::new(), // Gemini chunks carry no message id.
                model: parsed
                    .get("modelVersion")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                usage: None,
            });
        }

        if let Some(candidate) = parsed
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        {
            if let Some(parts) = candidate
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        if part
                            .get("thought")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            events.push(NeutralStreamEvent::ReasoningDelta(text.to_string()));
                        } else if !text.is_empty() {
                            events.push(NeutralStreamEvent::TextDelta(text.to_string()));
                        }
                    }
                    if let Some(call) = part.get("functionCall") {
                        let index = self.tool_index;
                        self.tool_index += 1;
                        // Gemini functionCall parts carry no id in the
                        // classic API; mint a deterministic per-stream one
                        // so clients can correlate results.
                        let id = match call.get("id").and_then(Value::as_str) {
                            Some(id) if !id.is_empty() => id.to_string(),
                            // 1-based, matching the body parser's fc_<n>.
                            _ => format!("fc_{}", index + 1),
                        };
                        events.push(NeutralStreamEvent::ToolCallDelta {
                            index,
                            id,
                            name: call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: call
                                .get("args")
                                .map(|a| a.to_string())
                                .unwrap_or_else(|| "{}".to_string()),
                        });
                    }
                }
            }
            if let Some(fr) = candidate.get("finishReason").and_then(Value::as_str) {
                self.pending_finish = Some(parse_finish_reason(fr));
            }
        }

        if let Some(usage) = parsed.get("usageMetadata") {
            self.pending_usage = Some(NeutralUsage {
                input_tokens: usage
                    .get("promptTokenCount")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: usage
                    .get("candidatesTokenCount")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            });
        }

        // A Gemini stream has no [DONE]; a chunk carrying a finish reason
        // marks the end. Emit MessageStop exactly once (a later chunk may
        // still carry late usage), and suppress it in finish() via the
        // emitted flag.
        if self.pending_finish.is_some() && !self.stop_emitted {
            self.stop_emitted = true;
            let fr = self.pending_finish.clone().unwrap_or(FinishReason::Stop);
            events.push(NeutralStreamEvent::MessageStop {
                finish_reason: fr,
                usage: self.pending_usage,
            });
        }

        events
    }

    fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        if !self.started || self.failed || self.stop_emitted {
            return Vec::new();
        }
        // If the stream closed without a finish chunk (connection end),
        // emit the stop now — once (idempotent for repeated finish calls).
        self.stop_emitted = true;
        vec![NeutralStreamEvent::MessageStop {
            finish_reason: self.pending_finish.clone().unwrap_or(FinishReason::Stop),
            usage: self.pending_usage,
        }]
    }
}

fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Stateful neutral events → Gemini chunk encoder.
pub struct GeminiStreamEncoder {
    started: bool,
    emitted_stop: bool,
    /// Tool-call arguments arrive as incremental JSON fragments. Gemini
    /// `functionCall` parts must carry a complete JSON object, so fragments
    /// are buffered per call index until they parse as valid JSON.
    pending_calls: std::collections::HashMap<u32, PendingCall>,
}

struct PendingCall {
    id: String,
    name: String,
    args: String,
}

impl GeminiStreamEncoder {
    pub fn new() -> Self {
        Self {
            started: false,
            emitted_stop: false,
            pending_calls: std::collections::HashMap::new(),
        }
    }
}

impl Default for GeminiStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamEncoder for GeminiStreamEncoder {
    fn encode(&mut self, event: NeutralStreamEvent) -> Vec<String> {
        let mut lines = Vec::new();
        match event {
            NeutralStreamEvent::MessageStart { .. } => {
                self.started = true;
                // Gemini chunks carry no message id or model version at
                // the event level; the fields are intentionally not stored.
            }
            NeutralStreamEvent::TextDelta(t) => {
                lines.push(self.chunk(json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "text": t }] },
                        "index": 0,
                    }],
                })));
            }
            NeutralStreamEvent::ReasoningDelta(t) => {
                lines.push(self.chunk(json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "text": t, "thought": true }] },
                        "index": 0,
                    }],
                })));
            }
            NeutralStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                let entry = self
                    .pending_calls
                    .entry(index)
                    .or_insert_with(|| PendingCall {
                        id,
                        name,
                        args: String::new(),
                    });
                entry.args.push_str(&arguments);
                // Only a complete JSON value can be emitted as a Gemini
                // functionCall part; partial fragments stay buffered.
                let parsed_args = match serde_json::from_str::<Value>(&entry.args) {
                    Ok(v) => v,
                    Err(_) => return lines,
                };
                let mut call = Map::new();
                if !entry.id.is_empty() {
                    call.insert("id".to_string(), json!(entry.id));
                }
                call.insert("name".to_string(), json!(entry.name));
                call.insert("args".to_string(), parsed_args);
                lines.push(self.chunk(json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "functionCall": call }] },
                        // Each SSE chunk describes ONE candidate: the
                        // candidate index is always 0; the tool-call index
                        // is not a candidate index.
                        "index": 0,
                    }],
                })));
                self.pending_calls.remove(&index);
            }
            NeutralStreamEvent::ReasoningSignature(_) => {
                // No Gemini equivalent; signatures are Anthropic-specific.
            }
            NeutralStreamEvent::MessageStop {
                finish_reason,
                usage,
            } => {
                if self.emitted_stop {
                    return lines;
                }
                self.emitted_stop = true;
                // Surface any call whose arguments never completed (e.g.
                // truncated stream) with null args rather than dropping it.
                if !self.pending_calls.is_empty() {
                    tracing::warn!(
                        calls = self.pending_calls.len(),
                        "gemini stream: tool-call arguments never formed complete JSON; emitting with null args"
                    );
                    let mut indices: Vec<u32> = self.pending_calls.keys().copied().collect();
                    indices.sort_unstable();
                    for index in indices {
                        let Some(call) = self.pending_calls.remove(&index) else {
                            continue;
                        };
                        let mut part = Map::new();
                        if !call.id.is_empty() {
                            part.insert("id".to_string(), json!(call.id));
                        }
                        part.insert("name".to_string(), json!(call.name));
                        part.insert("args".to_string(), Value::Null);
                        lines.push(self.chunk(json!({
                            "candidates": [{
                                "content": { "role": "model", "parts": [{ "functionCall": part }] },
                                "index": index,
                            }],
                        })));
                    }
                }
                let mut chunk = Map::new();
                chunk.insert(
                    "candidates".to_string(),
                    json!([{
                        "content": {},
                        "index": 0,
                        "finishReason": serialize_finish_reason(&finish_reason),
                    }]),
                );
                if let Some(u) = usage {
                    chunk.insert(
                        "usageMetadata".to_string(),
                        json!({
                            "promptTokenCount": u.input_tokens,
                            "candidatesTokenCount": u.output_tokens,
                        }),
                    );
                }
                lines.push(self.chunk(Value::Object(chunk)));
            }
            NeutralStreamEvent::Error(e) => {
                // Same variant mapping as the non-streaming error path,
                // so streamed errors carry the correct code/status
                // (429/RESOURCE_EXHAUSTED, 503/UNAVAILABLE, ...).
                let (code, status) = match &e {
                    AdapterError::InvalidRequest(_) => (400, "INVALID_ARGUMENT"),
                    AdapterError::Authentication => (401, "UNAUTHENTICATED"),
                    AdapterError::PermissionDenied => (403, "PERMISSION_DENIED"),
                    AdapterError::RateLimit { .. } => (429, "RESOURCE_EXHAUSTED"),
                    AdapterError::InsufficientQuota => (429, "RESOURCE_EXHAUSTED"),
                    AdapterError::Overloaded => (503, "UNAVAILABLE"),
                    AdapterError::Upstream { status, .. } => (*status, "UPSTREAM_ERROR"),
                    _ => (500, "INTERNAL"),
                };
                lines.push(self.chunk(json!({
                    "error": { "code": code, "message": e.to_string(), "status": status },
                })));
            }
        }
        lines
    }

    fn done(&self) -> String {
        // Gemini streams have no [DONE] terminator; the finish chunk (or the
        // connection close) ends the stream.
        String::new()
    }
}

impl GeminiStreamEncoder {
    fn chunk(&self, payload: Value) -> String {
        format!("data: {payload}\n\n")
    }
}

fn serialize_finish_reason(fr: &FinishReason) -> &str {
    match fr {
        FinishReason::Stop => "STOP",
        FinishReason::Length => "MAX_TOKENS",
        FinishReason::ContentFilter => "SAFETY",
        FinishReason::ToolCalls => "STOP",
        FinishReason::Other(s) => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(parts: Value, finish: Option<&str>) -> String {
        let mut c = json!({
            "candidates": [{ "content": { "role": "model", "parts": parts }, "index": 0 }],
        });
        if let Some(f) = finish {
            c["candidates"][0]["finishReason"] = json!(f);
        }
        c.to_string()
    }

    #[test]
    fn decodes_text_and_reasoning_chunks() {
        let mut d = GeminiStreamDecoder::new();
        let events = d.feed(&chunk(json!([{ "text": "Hel" }]), None));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStart { .. }
        ));

        let events = d.feed(&chunk(json!([{ "text": "lo" }]), None));
        assert_eq!(events, vec![NeutralStreamEvent::TextDelta("lo".into())]);

        let events = d.feed(&chunk(json!([{ "text": "think", "thought": true }]), None));
        assert_eq!(
            events,
            vec![NeutralStreamEvent::ReasoningDelta("think".into())]
        );
    }

    #[test]
    fn decodes_tool_call_chunks() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([]), None));
        let events = d.feed(&chunk(
            json!([{ "functionCall": { "id": "c1", "name": "f", "args": {"a": 1} } }]),
            None,
        ));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, id, name, arguments }
                if id == "c1" && name == "f" && arguments.contains("a")
        ));
    }

    #[test]
    fn synthesizes_id_for_idless_tool_call_chunk() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([]), None));
        // Real Gemini functionCall parts carry no id; the decoder mints one
        // so clients can correlate results.
        let events = d.feed(&chunk(
            json!([{ "functionCall": { "name": "f", "args": {"a": 1} } }]),
            None,
        ));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, id, name, .. }
                if id == "fc_1" && name == "f"
        ));
    }

    #[test]
    fn decodes_finish_chunk_with_usage() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([{ "text": "x" }]), None));
        let data = json!({
            "candidates": [{ "content": {}, "index": 0, "finishReason": "MAX_TOKENS" }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 },
        })
        .to_string();
        let events = d.feed(&data);
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Length,
                usage: Some(u),
            } if u.input_tokens == 10 && u.output_tokens == 5
        ));
        // finish() must not double-emit after a finish chunk.
        assert!(d.finish().is_empty());
    }

    #[test]
    fn finish_emits_stop_when_stream_closed_without_finish() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([{ "text": "partial" }]), None));
        let events = d.finish();
        assert!(matches!(&events[0], NeutralStreamEvent::MessageStop { .. }));
    }

    #[test]
    fn empty_stream_no_finish_event() {
        let mut d = GeminiStreamDecoder::new();
        assert!(d.finish().is_empty());
    }

    #[test]
    fn encodes_neutral_events_to_chunks() {
        let mut e = GeminiStreamEncoder::new();
        assert!(
            e.encode(NeutralStreamEvent::MessageStart {
                id: String::new(),
                model: "m".into(),
                usage: None,
            })
            .is_empty()
        );

        let lines = e.encode(NeutralStreamEvent::TextDelta("hi".into()));
        assert!(lines[0].starts_with("data: "));
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["text"], "hi");

        let lines = e.encode(NeutralStreamEvent::ReasoningDelta("r".into()));
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["thought"], true);

        let lines = e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::Length,
            usage: Some(NeutralUsage {
                input_tokens: 1,
                output_tokens: 2,
            }),
        });
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["candidates"][0]["finishReason"], "MAX_TOKENS");
        assert_eq!(v["usageMetadata"]["candidatesTokenCount"], 2);
        // Stop must be emitted only once.
        assert!(
            e.encode(NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Stop,
                usage: None,
            })
            .is_empty()
        );

        assert_eq!(e.done(), "");
    }

    #[test]
    fn encodes_tool_call() {
        let mut e = GeminiStreamEncoder::new();
        let lines = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "c1".into(),
            name: "f".into(),
            arguments: "{\"a\":1}".into(),
        });
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        let call = &v["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(call["name"], "f");
        assert_eq!(call["args"]["a"], 1);
    }

    #[test]
    fn error_chunk_produces_error_event_and_no_synthetic_stop() {
        let mut d = GeminiStreamDecoder::new();
        let data = json!({
            "error": { "code": 400, "message": "bad request", "status": "INVALID_ARGUMENT" },
        })
        .to_string();
        let events = d.feed(&data);
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::Error(e) if e.to_string().contains("bad request")
        ));
        assert!(
            !matches!(&events[0], NeutralStreamEvent::MessageStart { .. }),
            "an error chunk must not be treated as a message start"
        );
        assert!(
            d.finish().is_empty(),
            "finish() must not synthesize a successful stop after an error"
        );
    }

    #[test]
    fn finish_chunk_emits_stop_only_once() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([{ "text": "x" }]), None));
        let finish = json!({
            "candidates": [{ "content": {}, "index": 0, "finishReason": "STOP" }],
        })
        .to_string();
        let events = d.feed(&finish);
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Stop,
                usage: None
            }
        ));
        // A later chunk carrying only usageMetadata must not emit a second
        // MessageStop.
        let usage = json!({
            "candidates": [{ "content": {}, "index": 0 }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 },
        })
        .to_string();
        let events = d.feed(&usage);
        assert!(
            events.is_empty(),
            "duplicate MessageStop emitted: {events:?}"
        );
        assert!(d.finish().is_empty());
    }

    #[test]
    fn finish_synthesized_stop_is_idempotent() {
        let mut d = GeminiStreamDecoder::new();
        d.feed(&chunk(json!([{ "text": "partial" }]), None));
        assert!(matches!(
            &d.finish()[0],
            NeutralStreamEvent::MessageStop { .. }
        ));
        // A second finish() (pipeline error path flushes once per call)
        // must not re-emit the stop.
        assert!(d.finish().is_empty());
    }

    #[test]
    fn encoder_buffers_partial_tool_args_until_valid_json() {
        let mut e = GeminiStreamEncoder::new();
        e.encode(NeutralStreamEvent::MessageStart {
            id: String::new(),
            model: "m".into(),
            usage: None,
        });
        // Individually invalid JSON fragments: nothing must be emitted
        // (previously each fragment became `args: null` and was lost).
        let first = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: "{\"city\":\"".into(),
        });
        assert!(
            first.is_empty(),
            "partial args must be buffered, not emitted as null"
        );
        let second = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "Paris\"}".into(),
        });
        assert_eq!(second.len(), 1);
        let v: Value = serde_json::from_str(&second[0][6..]).unwrap();
        let call = &v["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(call["name"], "get_weather");
        assert_eq!(call["args"]["city"], "Paris");
        // A second call with its own index is buffered independently.
        let third = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 1,
            id: "call_2".into(),
            name: "f2".into(),
            arguments: "{}".into(),
        });
        assert_eq!(third.len(), 1);
    }

    #[test]
    fn encoder_surfaces_incomplete_call_args_at_stop() {
        let mut e = GeminiStreamEncoder::new();
        e.encode(NeutralStreamEvent::MessageStart {
            id: String::new(),
            model: "m".into(),
            usage: None,
        });
        e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            arguments: "{\"city\":".into(),
        });
        let lines = e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::Stop,
            usage: None,
        });
        assert!(
            lines.iter().any(|l| l.contains("\"functionCall\"")),
            "an incomplete call must still surface at stop (args: null)"
        );
    }
}
