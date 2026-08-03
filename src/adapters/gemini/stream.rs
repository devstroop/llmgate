//! Gemini SSE streaming: chunk decoder and chunk encoder.
//!
//! Gemini's `streamGenerateContent?alt=sse` returns a stream of
//! `GenerateContentResponse` chunks, one per SSE `data:` line. There is no
//! `[DONE]` marker — the stream ends when the connection closes or after a
//! chunk carrying `finishReason`/`usageMetadata`.

use serde_json::{Map, Value, json};

use crate::core::neutral::{FinishReason, NeutralStreamEvent, NeutralUsage};
use crate::core::registry::{StreamDecoder, StreamEncoder};

/// Stateful Gemini chunk → neutral events decoder.
pub struct GeminiStreamDecoder {
    started: bool,
    pending_finish: Option<FinishReason>,
    pending_usage: Option<NeutralUsage>,
    tool_index: u32,
}

impl GeminiStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
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
                        events.push(NeutralStreamEvent::ToolCallDelta {
                            index,
                            id: call
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
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
        // marks the end. Emit MessageStop now so clients see termination
        // promptly, and suppress it in finish() via the pending state.
        if self.pending_finish.is_some() {
            let fr = self.pending_finish.clone().unwrap_or(FinishReason::Stop);
            events.push(NeutralStreamEvent::MessageStop {
                finish_reason: fr,
                usage: self.pending_usage,
            });
        }

        events
    }

    fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        if !self.started {
            return Vec::new();
        }
        // If the stream closed without a finish chunk (connection end), emit
        // the stop now.
        if self.pending_finish.is_none() {
            vec![NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Stop,
                usage: self.pending_usage,
            }]
        } else {
            Vec::new()
        }
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
    id: String,
    model: String,
    started: bool,
    emitted_stop: bool,
}

impl GeminiStreamEncoder {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            started: false,
            emitted_stop: false,
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
            NeutralStreamEvent::MessageStart { id, model } => {
                self.started = true;
                self.id = id;
                self.model = model;
                // Gemini streams have no message_start event; nothing to emit.
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
                let args: Value = serde_json::from_str(&arguments).unwrap_or(Value::Null);
                let mut call = Map::new();
                if !id.is_empty() {
                    call.insert("id".to_string(), json!(id));
                }
                call.insert("name".to_string(), json!(name));
                call.insert("args".to_string(), args);
                lines.push(self.chunk(json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "functionCall": call }] },
                        "index": index,
                    }],
                })));
            }
            NeutralStreamEvent::MessageStop {
                finish_reason,
                usage,
            } => {
                if self.emitted_stop {
                    return lines;
                }
                self.emitted_stop = true;
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
                lines.push(self.chunk(json!({
                    "error": { "code": 500, "message": e.to_string(), "status": "INTERNAL" },
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
}
