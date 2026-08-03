//! Gemini ↔ neutral conversion (non-streaming).
//!
//! Converts between the Gemini `generateContent` wire format and the neutral
//! model, both request and response directions.
//!
//! Wire notes:
//! - Roles are `user` / `model` (Gemini has no `assistant`); the system
//!   instruction is a top-level `systemInstruction` object, not a message.
//! - Content parts: `text`, `inlineData` (images), `functionCall`,
//!   `functionResponse`, and `thought` parts for thinking.
//! - Tool calls: `functionCall {name, args}` parts in a `model` message;
//!   results are `functionResponse {name, response}` parts in a `user`
//!   message.
//! - Finish reasons are uppercase (`STOP`, `MAX_TOKENS`, `SAFETY`, ...).
//! - Usage is `usageMetadata {promptTokenCount, candidatesTokenCount}`.

use serde_json::{Map, Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{
    ContentBlock, FinishReason, NeutralMessage, NeutralRequest, NeutralResponse, NeutralRole,
    NeutralTool, NeutralUsage,
};

/// Parse a Gemini `generateContent` request body into the neutral model.
pub fn parse_request(body: &str) -> Result<NeutralRequest, AdapterError> {
    let root: Value = serde_json::from_str(body)
        .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| AdapterError::InvalidRequest("expected a JSON object".to_string()))?;

    let mut messages = Vec::new();

    if let Some(system) = root.get("systemInstruction") {
        let text = collect_text_parts(system);
        if !text.is_empty() {
            messages.push(NeutralMessage {
                role: NeutralRole::System,
                content: text.into_iter().map(ContentBlock::Text).collect(),
            });
        }
    }

    if let Some(contents) = root.get("contents").and_then(Value::as_array) {
        for content in contents {
            if let Some(m) = parse_content(content) {
                messages.push(m);
            }
        }
    }

    let mut tools = Vec::new();
    if let Some(tools_arr) = root.get("tools").and_then(Value::as_array) {
        for tool in tools_arr {
            if let Some(fns) = tool.get("functionDeclarations").and_then(Value::as_array) {
                for f in fns {
                    if let Some(t) = parse_function_declaration(f) {
                        tools.push(t);
                    }
                }
            }
        }
    }

    let generation = root.get("generationConfig").unwrap_or(&Value::Null);
    Ok(NeutralRequest {
        model: String::new(), // Gemini takes the model in the URL path.
        messages,
        tools,
        max_tokens: generation
            .get("maxOutputTokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        temperature: generation.get("temperature").and_then(Value::as_f64),
        top_p: generation.get("topP").and_then(Value::as_f64),
        top_k: generation
            .get("topK")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        stop: generation
            .get("stopSequences")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .filter(|s: &Vec<String>| !s.is_empty()),
        stream: false,
    })
}

fn parse_content(content: &Value) -> Option<NeutralMessage> {
    let role = match content.get("role").and_then(Value::as_str) {
        Some("user") => NeutralRole::User,
        Some("model") => NeutralRole::Assistant,
        _ => return None,
    };
    let blocks = parse_parts(content.get("parts").and_then(Value::as_array));
    Some(NeutralMessage {
        role,
        content: blocks,
    })
}

fn parse_parts(parts: Option<&Vec<Value>>) -> Vec<ContentBlock> {
    let Some(parts) = parts else {
        return Vec::new();
    };
    parts.iter().filter_map(parse_part).collect()
}

fn parse_part(part: &Value) -> Option<ContentBlock> {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        // Thinking parts carry `thought: true` and no visible rendering
        // requirement; treat them as reasoning blocks.
        if part
            .get("thought")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Some(ContentBlock::Thinking {
                thinking: text.to_string(),
                signature: None,
            });
        }
        return Some(ContentBlock::Text(text.to_string()));
    }
    if let Some(data) = part.get("inlineData") {
        return Some(ContentBlock::Image {
            media_type: data
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            base64: data
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });
    }
    if let Some(call) = part.get("functionCall") {
        return Some(ContentBlock::ToolUse {
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
            input: call
                .get("args")
                .cloned()
                .unwrap_or(Value::Object(Map::new())),
        });
    }
    if let Some(resp) = part.get("functionResponse") {
        return Some(ContentBlock::ToolResult {
            tool_use_id: resp
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content: resp
                .get("response")
                .map(|r| r.to_string())
                .unwrap_or_default(),
            is_error: false,
        });
    }
    None
}

fn parse_function_declaration(f: &Value) -> Option<NeutralTool> {
    Some(NeutralTool {
        name: f.get("name")?.as_str()?.to_string(),
        description: f
            .get("description")
            .and_then(Value::as_str)
            .map(String::from),
        parameters: f
            .get("parameters")
            .cloned()
            .unwrap_or(Value::Object(Map::new())),
    })
}

/// Collect the text of a `{parts: [{text, ...}]}` structure (used for both
/// `systemInstruction` and `functionResponse` content).
fn collect_text_parts(v: &Value) -> Vec<String> {
    v.get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Serialize a neutral request into a Gemini `generateContent` body.
pub fn serialize_request(req: &NeutralRequest) -> Result<String, AdapterError> {
    let mut root = Map::new();

    let mut contents: Vec<Value> = Vec::new();
    let mut system_text: Vec<String> = Vec::new();

    for msg in &req.messages {
        match msg.role {
            NeutralRole::System => {
                system_text.extend(
                    msg.content
                        .iter()
                        .filter_map(ContentBlock::text)
                        .map(String::from),
                );
            }
            NeutralRole::User | NeutralRole::Assistant => {
                let role = match msg.role {
                    NeutralRole::User => "user",
                    _ => "model",
                };
                let parts: Vec<Value> = msg.content.iter().filter_map(serialize_part).collect();
                if !parts.is_empty() {
                    contents.push(json!({ "role": role, "parts": parts }));
                }
            }
            // Tool results are `functionResponse` parts in a `user` message.
            NeutralRole::Tool => {
                let parts: Vec<Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => Some(json!({
                            "functionResponse": {
                                "id": tool_use_id,
                                "name": tool_use_id,
                                "response": {
                                    "result": content,
                                    "is_error": is_error,
                                },
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                if !parts.is_empty() {
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
        }
    }

    if !system_text.is_empty() {
        root.insert(
            "systemInstruction".to_string(),
            json!({ "parts": system_text.iter().map(|t| json!({ "text": t })).collect::<Vec<_>>() }),
        );
    }

    root.insert("contents".to_string(), Value::Array(contents));

    if !req.tools.is_empty() {
        root.insert(
            "tools".to_string(),
            json!([{
                "functionDeclarations": req.tools.iter().map(serialize_tool).collect::<Vec<_>>(),
            }]),
        );
    }

    let mut generation = Map::new();
    if let Some(mt) = req.max_tokens {
        generation.insert("maxOutputTokens".to_string(), json!(mt));
    }
    if let Some(t) = req.temperature {
        generation.insert("temperature".to_string(), json!(t));
    }
    if let Some(tp) = req.top_p {
        generation.insert("topP".to_string(), json!(tp));
    }
    if let Some(tk) = req.top_k {
        generation.insert("topK".to_string(), json!(tk));
    }
    if let Some(stop) = &req.stop {
        generation.insert(
            "stopSequences".to_string(),
            Value::Array(stop.iter().cloned().map(Value::String).collect()),
        );
    }
    if !generation.is_empty() {
        root.insert("generationConfig".to_string(), Value::Object(generation));
    }

    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
}

fn serialize_part(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Text(t) => Some(json!({ "text": t })),
        ContentBlock::Image { media_type, base64 } => Some(json!({
            "inlineData": { "mimeType": media_type, "data": base64 },
        })),
        ContentBlock::Thinking { thinking, .. } => Some(json!({
            "text": thinking,
            "thought": true,
        })),
        ContentBlock::ToolUse { id, name, input } => Some(json!({
            "functionCall": { "id": id, "name": name, "args": input },
        })),
        ContentBlock::ToolResult { .. } => None, // handled at message level
    }
}

fn serialize_tool(tool: &NeutralTool) -> Value {
    let mut f = Map::new();
    f.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(desc) = &tool.description {
        f.insert("description".to_string(), Value::String(desc.clone()));
    }
    f.insert("parameters".to_string(), tool.parameters.clone());
    json!(f)
}

/// Parse a Gemini `generateContent` response body into the neutral model.
pub fn parse_response(body: &str) -> Result<NeutralResponse, AdapterError> {
    let root: Value = serde_json::from_str(body)
        .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| AdapterError::InvalidRequest("expected a JSON object".to_string()))?;

    if let Some(error) = root.get("error") {
        let status = error.get("code").and_then(Value::as_u64).unwrap_or(500) as u16;
        return Err(AdapterError::Upstream {
            status,
            body: body.to_string(),
        });
    }

    let mut content = Vec::new();
    let mut finish_reason = FinishReason::Stop;
    if let Some(candidate) = root
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    {
        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
        {
            content = parts.iter().filter_map(parse_part).collect();
        }
        if let Some(fr) = candidate.get("finishReason").and_then(Value::as_str) {
            finish_reason = parse_finish_reason(fr);
        }
    }

    let usage = root.get("usageMetadata").and_then(parse_usage);

    Ok(NeutralResponse {
        id: String::new(), // Gemini responses carry no message id.
        model: root
            .get("modelVersion")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        content,
        finish_reason,
        usage,
    })
}

fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

fn parse_usage(usage: &Value) -> Option<NeutralUsage> {
    Some(NeutralUsage {
        input_tokens: usage
            .get("promptTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: usage
            .get("candidatesTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Serialize a neutral response into a Gemini `generateContent` body.
pub fn serialize_response(resp: &NeutralResponse) -> Result<String, AdapterError> {
    let parts: Vec<Value> = resp.content.iter().filter_map(serialize_part).collect();
    let mut root = Map::new();
    root.insert(
        "candidates".to_string(),
        json!([{
            "content": { "role": "model", "parts": parts },
            "finishReason": serialize_finish_reason(&resp.finish_reason),
        }]),
    );
    if let Some(u) = &resp.usage {
        root.insert(
            "usageMetadata".to_string(),
            json!({
                "promptTokenCount": u.input_tokens,
                "candidatesTokenCount": u.output_tokens,
            }),
        );
    }
    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
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

/// Render an error in the Gemini native shape.
pub fn serialize_error(err: &AdapterError) -> (u16, String) {
    let (status, etype) = match err {
        AdapterError::InvalidRequest(_) => (400, "INVALID_ARGUMENT"),
        AdapterError::Authentication => (401, "UNAUTHENTICATED"),
        AdapterError::PermissionDenied => (403, "PERMISSION_DENIED"),
        AdapterError::RateLimit { .. } => (429, "RESOURCE_EXHAUSTED"),
        AdapterError::InsufficientQuota => (429, "RESOURCE_EXHAUSTED"),
        AdapterError::Overloaded => (503, "UNAVAILABLE"),
        AdapterError::Api(_) => (500, "INTERNAL"),
        AdapterError::Upstream { status, .. } => (*status, "UPSTREAM_ERROR"),
        AdapterError::Internal(_) => (500, "INTERNAL"),
    };
    let body = json!({
        "error": {
            "code": status,
            "message": err.to_string(),
            "status": etype,
        }
    })
    .to_string();
    (status, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_request_with_system_and_generation_config() {
        let body = r#"{
            "systemInstruction": {"parts": [{"text": "be concise"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi"}]}
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 256,
                "stopSequences": ["END"]
            }
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, NeutralRole::System);
        assert_eq!(req.messages[1].role, NeutralRole::User);
        assert_eq!(req.messages[2].role, NeutralRole::Assistant);
        assert_eq!(
            req.messages[1].content[0],
            ContentBlock::Text("hello".into())
        );
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.stop, Some(vec!["END".into()]));
    }

    #[test]
    fn parses_parts_including_thought_images_and_function_calls() {
        let body = r#"{
            "contents": [{"role": "model", "parts": [
                {"text": "thinking...", "thought": true},
                {"text": "answer"},
                {"functionCall": {"id": "c1", "name": "get_weather", "args": {"city": "paris"}}}
            ]}]
        }"#;
        let req = parse_request(body).unwrap();
        let blocks = &req.messages[0].content;
        assert_eq!(blocks.len(), 3);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Thinking { thinking, .. } if thinking == "thinking..."
        ));
        assert_eq!(blocks[1], ContentBlock::Text("answer".into()));
        match &blocks[2] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "paris");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }

        let body = r#"{
            "contents": [{"role": "user", "parts": [
                {"text": "what is this?"},
                {"inlineData": {"mimeType": "image/png", "data": "QUFB"}}
            ]}]
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(
            req.messages[0].content[1],
            ContentBlock::Image {
                media_type: "image/png".into(),
                base64: "QUFB".into(),
            }
        );
    }

    #[test]
    fn parses_function_response_as_tool_result() {
        let body = r#"{
            "contents": [{"role": "user", "parts": [
                {"functionResponse": {"id": "c1", "name": "get_weather", "response": {"result": "22c"}}}
            ]}]
        }"#;
        let req = parse_request(body).unwrap();
        match &req.messages[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                assert_eq!(tool_use_id, "c1");
                assert!(content.contains("22c"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn parses_tools() {
        let body = r#"{
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "w", "parameters": {"type": "object"}}
            ]}]
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(req.tools[0].parameters["type"], "object");
    }

    #[test]
    fn serialize_request_round_trips() {
        let req = NeutralRequest {
            model: String::new(),
            messages: vec![
                NeutralMessage {
                    role: NeutralRole::System,
                    content: vec![ContentBlock::Text("be concise".into())],
                },
                NeutralMessage {
                    role: NeutralRole::User,
                    content: vec![
                        ContentBlock::Text("look".into()),
                        ContentBlock::Image {
                            media_type: "image/png".into(),
                            base64: "QUFB".into(),
                        },
                    ],
                },
                NeutralMessage {
                    role: NeutralRole::Assistant,
                    content: vec![
                        ContentBlock::Thinking {
                            thinking: "hmm".into(),
                            signature: None,
                        },
                        ContentBlock::ToolUse {
                            id: "c1".into(),
                            name: "f".into(),
                            input: json!({"a": 1}),
                        },
                    ],
                },
            ],
            tools: vec![NeutralTool {
                name: "f".into(),
                description: Some("d".into()),
                parameters: json!({"type": "object"}),
            }],
            max_tokens: Some(128),
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: false,
        };
        let out = serialize_request(&req).unwrap();
        let parsed = parse_request(&out).unwrap();
        assert_eq!(parsed.messages.len(), 3);
        assert_eq!(parsed.messages[0].role, NeutralRole::System);
        assert!(parsed.messages[1].content.contains(&ContentBlock::Image {
            media_type: "image/png".into(),
            base64: "QUFB".into(),
        }));
        assert!(matches!(
            &parsed.messages[2].content[0],
            ContentBlock::Thinking { thinking, .. } if thinking == "hmm"
        ));
        assert_eq!(parsed.tools[0].name, "f");
        assert_eq!(parsed.max_tokens, Some(128));
    }

    #[test]
    fn parses_response_with_finish_reason_and_usage() {
        let body = r#"{
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "answer"}]},
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
        }"#;
        let resp = parse_response(body).unwrap();
        assert_eq!(resp.content[0], ContentBlock::Text("answer".into()));
        assert_eq!(resp.finish_reason, FinishReason::Length);
        assert_eq!(
            resp.usage,
            Some(NeutralUsage {
                input_tokens: 10,
                output_tokens: 5
            })
        );
    }

    #[test]
    fn serialize_response_round_trips() {
        let resp = NeutralResponse {
            id: String::new(),
            model: String::new(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: None,
                },
                ContentBlock::Text("answer".into()),
            ],
            finish_reason: FinishReason::Stop,
            usage: Some(NeutralUsage {
                input_tokens: 3,
                output_tokens: 7,
            }),
        };
        let out = serialize_response(&resp).unwrap();
        let parsed = parse_response(&out).unwrap();
        assert!(
            parsed
                .content
                .contains(&ContentBlock::Text("answer".into()))
        );
        assert_eq!(parsed.finish_reason, FinishReason::Stop);
        assert_eq!(
            parsed.usage,
            Some(NeutralUsage {
                input_tokens: 3,
                output_tokens: 7
            })
        );
    }

    #[test]
    fn errors_map_to_gemini_shape() {
        let (status, body) = serialize_error(&AdapterError::RateLimit {
            retry_after_secs: None,
        });
        assert_eq!(status, 429);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["status"], "RESOURCE_EXHAUSTED");
        assert_eq!(v["error"]["code"], 429);
    }
}
