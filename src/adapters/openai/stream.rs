//! OpenAI SSE streaming: chunk decoder and chunk encoder.
//!
//! Upstream chunks (`chat.completion.chunk`) are decoded into neutral stream
//! events; neutral events are encoded back into client chunks.

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{FinishReason, NeutralStreamEvent, NeutralUsage};
use crate::core::registry::{StreamDecoder, StreamEncoder};

/// Stateful OpenAI chunk → neutral events decoder.
pub struct OpenAiStreamDecoder {
    started: bool,
    failed: bool,
    stop_emitted: bool,
    seen_tool_indices: HashSet<u32>,
    pending_finish: Option<FinishReason>,
    pending_usage: Option<NeutralUsage>,
}

impl OpenAiStreamDecoder {
    pub fn new() -> Self {
        Self {
            started: false,
            failed: false,
            stop_emitted: false,
            seen_tool_indices: HashSet::new(),
            pending_finish: None,
            pending_usage: None,
        }
    }
}

impl Default for OpenAiStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder for OpenAiStreamDecoder {
    fn feed(&mut self, data: &str) -> Vec<NeutralStreamEvent> {
        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        // Upstream error payloads must surface as an Error event, not be
        // swallowed into a spurious MessageStart + synthetic stop.
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
                id: parsed
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                model: parsed
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                usage: None,
            });
        }

        let Some(choice) = parsed
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            // Chunks without choices may still carry usage (final chunk).
            if let Some(usage) = parsed.get("usage") {
                self.pending_usage = parse_usage(usage);
            }
            return events;
        };

        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta
                .get("content")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
            {
                events.push(NeutralStreamEvent::TextDelta(content.to_string()));
            }
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
                .filter(|r| !r.is_empty())
            {
                events.push(NeutralStreamEvent::ReasoningDelta(reasoning.to_string()));
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    if let Some(event) = parse_tool_call(call, &mut self.seen_tool_indices) {
                        events.push(event);
                    }
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.pending_finish = Some(parse_finish_reason(fr));
        }

        if let Some(usage) = parsed.get("usage") {
            self.pending_usage = parse_usage(usage);
        }

        events
    }

    fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        if !self.started || self.failed || self.stop_emitted {
            return Vec::new();
        }
        // Emit the terminal event exactly once (idempotent for repeated
        // finish calls, e.g. the pipeline's post-loop flush).
        self.stop_emitted = true;
        vec![NeutralStreamEvent::MessageStop {
            finish_reason: self.pending_finish.clone().unwrap_or(FinishReason::Stop),
            usage: self.pending_usage,
        }]
    }
}

fn parse_tool_call(call: &Value, seen: &mut HashSet<u32>) -> Option<NeutralStreamEvent> {
    let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
    let arguments = call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let (id, name) = if seen.insert(index) {
        (
            call.get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            call.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )
    } else {
        (String::new(), String::new())
    };

    Some(NeutralStreamEvent::ToolCallDelta {
        index,
        id,
        name,
        arguments,
    })
}

fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

fn parse_usage(usage: &Value) -> Option<NeutralUsage> {
    Some(NeutralUsage {
        input_tokens: usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Stateful neutral events → OpenAI chunk encoder.
pub struct OpenAiStreamEncoder {
    id: String,
    model: String,
    created: u64,
}

impl OpenAiStreamEncoder {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

impl Default for OpenAiStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamEncoder for OpenAiStreamEncoder {
    fn encode(&mut self, event: NeutralStreamEvent) -> Vec<String> {
        match event {
            NeutralStreamEvent::MessageStart { id, model, .. } => {
                self.id = id;
                self.model = model;
                vec![self.chunk(json!({
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": "" },
                        "finish_reason": null,
                    }],
                }))]
            }
            NeutralStreamEvent::TextDelta(t) => vec![self.chunk(json!({
                "choices": [{
                    "index": 0,
                    "delta": { "content": t },
                    "finish_reason": null,
                }],
            }))],
            NeutralStreamEvent::ReasoningDelta(t) => vec![self.chunk(json!({
                "choices": [{
                    "index": 0,
                    "delta": { "reasoning_content": t },
                    "finish_reason": null,
                }],
            }))],
            NeutralStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                let mut call = json!({ "index": index });
                if !id.is_empty() {
                    call["id"] = json!(id);
                    call["type"] = json!("function");
                }
                let mut function = serde_json::Map::new();
                if !name.is_empty() {
                    function.insert("name".to_string(), json!(name));
                }
                if !arguments.is_empty() {
                    function.insert("arguments".to_string(), json!(arguments));
                }
                call["function"] = Value::Object(function);
                vec![self.chunk(json!({
                    "choices": [{
                        "index": 0,
                        "delta": { "tool_calls": [call] },
                        "finish_reason": null,
                    }],
                }))]
            }
            NeutralStreamEvent::ReasoningSignature(_) => {
                // No OpenAI equivalent; signatures are Anthropic-specific.
                Vec::new()
            }
            NeutralStreamEvent::MessageStop {
                finish_reason,
                usage,
            } => {
                let mut chunk = json!({
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason.as_str(),
                    }],
                });
                if let Some(u) = usage {
                    chunk["usage"] = json!({
                        "prompt_tokens": u.input_tokens,
                        "completion_tokens": u.output_tokens,
                        "total_tokens": u.input_tokens + u.output_tokens,
                    });
                }
                vec![self.chunk(chunk)]
            }
            NeutralStreamEvent::Error(e) => vec![self.chunk(json!({
                "error": { "message": e.to_string() },
            }))],
        }
    }

    fn done(&self) -> String {
        "data: [DONE]\n\n".to_string()
    }
}

impl OpenAiStreamEncoder {
    fn chunk(&self, payload: Value) -> String {
        let mut chunk = serde_json::Map::new();
        if !self.id.is_empty() {
            chunk.insert("id".to_string(), json!(self.id));
        }
        chunk.insert("object".to_string(), json!("chat.completion.chunk"));
        chunk.insert("created".to_string(), json!(self.created));
        chunk.insert("model".to_string(), json!(self.model));
        for (k, v) in payload.as_object().unwrap() {
            chunk.insert(k.clone(), v.clone());
        }
        format!(
            "data: {}\n\n",
            serde_json::to_string(&Value::Object(chunk)).unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, delta: Value, finish: Option<&str>) -> String {
        serde_json::to_string(&json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        }))
        .unwrap()
    }

    #[test]
    fn decodes_text_and_reasoning_chunks() {
        let mut d = OpenAiStreamDecoder::new();
        let events = d.feed(&chunk("c1", json!({"role": "assistant"}), None));
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], NeutralStreamEvent::MessageStart { id, .. } if id == "c1"));

        let events = d.feed(&chunk("c1", json!({"content": "hel"}), None));
        assert_eq!(events, vec![NeutralStreamEvent::TextDelta("hel".into())]);

        let events = d.feed(&chunk("c1", json!({"reasoning_content": "think"}), None));
        assert_eq!(
            events,
            vec![NeutralStreamEvent::ReasoningDelta("think".into())]
        );

        let events = d.feed(&chunk("c1", json!({}), Some("stop")));
        assert!(events.is_empty());

        let events = d.finish();
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Stop,
                ..
            }
        ));
    }

    #[test]
    fn decodes_tool_call_fragments() {
        let mut d = OpenAiStreamDecoder::new();
        d.feed(&chunk("c1", json!({"role": "assistant"}), None));

        let events = d.feed(&chunk("c1", json!({"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":"}}
        ]}), None));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, id, name, .. } if id == "call_1" && name == "f"
        ));

        let events = d.feed(&chunk(
            "c1",
            json!({"tool_calls": [
                {"index": 0, "type": "function", "function": {"arguments": " 1}"}}
            ]}),
            None,
        ));
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::ToolCallDelta { index: 0, id, name, arguments, .. }
                if id.is_empty() && name.is_empty() && arguments == " 1}"
        ));
    }

    #[test]
    fn decodes_final_chunk_with_usage() {
        let mut d = OpenAiStreamDecoder::new();
        d.feed(&chunk("c1", json!({"role": "assistant"}), None));
        let data = serde_json::to_string(&json!({
            "id": "c1", "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }))
        .unwrap();
        let events = d.feed(&data);
        assert!(events.is_empty());
        let events = d.finish();
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::MessageStop { finish_reason: FinishReason::Stop, usage: Some(u) }
                if u.input_tokens == 10 && u.output_tokens == 5
        ));
    }

    #[test]
    fn empty_stream_no_finish_event() {
        let mut d = OpenAiStreamDecoder::new();
        assert!(d.finish().is_empty());
    }

    #[test]
    fn error_chunk_produces_error_event_and_no_synthetic_stop() {
        let mut d = OpenAiStreamDecoder::new();
        let events =
            d.feed(&json!({"error": {"message": "boom", "type": "server_error"}}).to_string());
        assert!(matches!(
            &events[0],
            NeutralStreamEvent::Error(e) if e.to_string().contains("boom")
        ));
        assert!(
            !matches!(&events[0], NeutralStreamEvent::MessageStart { .. }),
            "an error payload must not be treated as a message start"
        );
        assert!(
            d.finish().is_empty(),
            "finish() must not report a successful stop after an error"
        );
    }

    #[test]
    fn finish_is_idempotent() {
        let mut d = OpenAiStreamDecoder::new();
        d.feed(&chunk("c1", json!({"role": "assistant"}), None));
        assert!(matches!(
            &d.finish()[0],
            NeutralStreamEvent::MessageStop { .. }
        ));
        assert!(d.finish().is_empty(), "second finish() must not re-emit");
    }

    #[test]
    fn encodes_neutral_events_to_chunks() {
        let mut e = OpenAiStreamEncoder::new();
        let lines = e.encode(NeutralStreamEvent::MessageStart {
            id: "c1".into(),
            model: "m".into(),
            usage: None,
        });
        assert!(lines[0].starts_with("data: "));
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");

        let lines = e.encode(NeutralStreamEvent::TextDelta("hi".into()));
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "hi");

        let lines = e.encode(NeutralStreamEvent::ReasoningDelta("r".into()));
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "r");

        let lines = e.encode(NeutralStreamEvent::MessageStop {
            finish_reason: FinishReason::Stop,
            usage: Some(NeutralUsage {
                input_tokens: 1,
                output_tokens: 2,
            }),
        });
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 2);

        assert_eq!(e.done(), "data: [DONE]\n\n");
    }

    #[test]
    fn encodes_tool_call_variants() {
        let mut e = OpenAiStreamEncoder::new();
        let lines = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            arguments: "{\"a\":".into(),
        });
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["id"], "call_1");

        let lines = e.encode(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "1}".into(),
        });
        let v: Value = serde_json::from_str(&lines[0][6..]).unwrap();
        let call = &v["choices"][0]["delta"]["tool_calls"][0];
        assert!(call.get("id").is_none());
        assert_eq!(call["function"]["arguments"], "1}");
    }
}
