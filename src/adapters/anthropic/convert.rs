//! Anthropic ↔ neutral conversion (non-streaming).

use serde_json::{Map, Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{
    ContentBlock, FinishReason, NeutralMessage, NeutralRequest, NeutralResponse, NeutralRole,
    NeutralTool, NeutralUsage,
};

/// Anthropic requires `max_tokens`; OpenAI clients often omit it. The gateway
/// fills in this default when converting a neutral request to Anthropic.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Parse an Anthropic `/v1/messages` request body into the neutral model.
pub fn parse_request(body: &str) -> Result<NeutralRequest, AdapterError> {
    let root: Value = serde_json::from_str(body)
        .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| AdapterError::InvalidRequest("expected a JSON object".to_string()))?;

    let model = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let mut messages = Vec::new();

    // Top-level `system` field (string or array of text blocks) becomes a
    // system message.
    if let Some(system) = root.get("system") {
        let text: Vec<String> = match system {
            Value::String(s) => vec![s.clone()],
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .map(String::from)
                .collect(),
            _ => Vec::new(),
        };
        if !text.is_empty() {
            messages.push(NeutralMessage {
                role: NeutralRole::System,
                content: text.into_iter().map(ContentBlock::Text).collect(),
            });
        }
    }

    if let Some(msgs) = root.get("messages").and_then(Value::as_array) {
        for msg in msgs {
            messages.extend(parse_messages(msg));
        }
    }

    let mut tools = Vec::new();
    if let Some(tool_arr) = root.get("tools").and_then(Value::as_array) {
        for tool in tool_arr {
            if let Some(t) = parse_tool(tool) {
                tools.push(t);
            }
        }
    }

    Ok(NeutralRequest {
        model,
        messages,
        tools,
        max_tokens: root
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        max_completion_tokens: None,
        temperature: root.get("temperature").and_then(Value::as_f64),
        top_p: root.get("top_p").and_then(Value::as_f64),
        top_k: root.get("top_k").and_then(Value::as_u64).map(|v| v as u32),
        stop: root
            .get("stop_sequences")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .filter(|s: &Vec<String>| !s.is_empty()),
        stream: root.get("stream").and_then(Value::as_bool).unwrap_or(false),
    })
}

/// Parse one Anthropic message into neutral messages.
///
/// Anthropic sends tool results as `role: "user"` messages containing
/// tool_result blocks; those normalize to the Tool role so every serializer
/// emits them as tool results (OpenAI/Gemini key off the Tool role, and
/// dropping them would break the tool loop). A user message mixing TEXT with
/// tool_result blocks (which Anthropic permits) is split: the results become
/// a Tool message, the remaining blocks a User message — no content is lost.
fn parse_messages(msg: &Value) -> Vec<NeutralMessage> {
    let Some(role) = msg.get("role").and_then(Value::as_str) else {
        return Vec::new();
    };
    let neutral_role = match role {
        "user" => NeutralRole::User,
        "assistant" => NeutralRole::Assistant,
        _ => return Vec::new(),
    };
    let blocks = parse_content_blocks(msg.get("content"));
    if neutral_role == NeutralRole::User {
        let has_results = blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
        if has_results {
            let results: Vec<ContentBlock> = blocks
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
                .cloned()
                .collect();
            let rest: Vec<ContentBlock> = blocks
                .into_iter()
                .filter(|b| !matches!(b, ContentBlock::ToolResult { .. }))
                .collect();
            let mut out = Vec::new();
            if !results.is_empty() {
                out.push(NeutralMessage {
                    role: NeutralRole::Tool,
                    content: results,
                });
            }
            if !rest.is_empty() {
                out.push(NeutralMessage {
                    role: NeutralRole::User,
                    content: rest,
                });
            }
            return out;
        }
    }
    vec![NeutralMessage {
        role: neutral_role,
        content: blocks,
    }]
}

/// Anthropic content: a string, or an array of typed blocks.
fn parse_content_blocks(content: Option<&Value>) -> Vec<ContentBlock> {
    match content {
        Some(Value::String(s)) if !s.is_empty() => vec![ContentBlock::Text(s.clone())],
        Some(Value::Array(blocks)) => blocks.iter().filter_map(parse_block).collect(),
        _ => Vec::new(),
    }
}

fn parse_block(block: &Value) -> Option<ContentBlock> {
    let btype = block.get("type")?.as_str()?;
    match btype {
        "text" => Some(ContentBlock::Text(block.get("text")?.as_str()?.to_string())),
        "image" => {
            let source = block.get("source")?;
            let stype = source.get("type")?.as_str()?;
            match stype {
                "base64" => Some(ContentBlock::Image {
                    media_type: source.get("media_type")?.as_str()?.to_string(),
                    base64: source.get("data")?.as_str()?.to_string(),
                }),
                "url" => Some(ContentBlock::Image {
                    media_type: "url".to_string(),
                    base64: source.get("url")?.as_str()?.to_string(),
                }),
                _ => None,
            }
        }
        "thinking" => Some(ContentBlock::Thinking {
            thinking: block.get("thinking")?.as_str()?.to_string(),
            signature: block
                .get("signature")
                .and_then(Value::as_str)
                .map(String::from),
        }),
        // Redacted thinking carries an opaque payload that Anthropic
        // requires echoing back verbatim in continued conversations;
        // preserve it as-is.
        "redacted_thinking" => Some(ContentBlock::RedactedThinking {
            data: block.get("data")?.as_str()?.to_string(),
        }),
        "tool_use" => Some(ContentBlock::ToolUse {
            id: block.get("id")?.as_str()?.to_string(),
            name: block.get("name")?.as_str()?.to_string(),
            input: block
                .get("input")
                .cloned()
                .unwrap_or(Value::Object(Map::new())),
        }),
        "tool_result" => {
            let content = match block.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(blocks)) => blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };
            Some(ContentBlock::ToolResult {
                tool_use_id: block.get("tool_use_id")?.as_str()?.to_string(),
                content,
                is_error: block
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        }
        _ => None,
    }
}

fn parse_tool(tool: &Value) -> Option<NeutralTool> {
    Some(NeutralTool {
        name: tool.get("name")?.as_str()?.to_string(),
        description: tool
            .get("description")
            .and_then(Value::as_str)
            .map(String::from),
        parameters: tool
            .get("input_schema")
            .cloned()
            .unwrap_or(Value::Object(Map::new())),
    })
}

/// Serialize a neutral request into an Anthropic `/v1/messages` body.
pub fn serialize_request(req: &NeutralRequest) -> Result<String, AdapterError> {
    let mut root = Map::new();
    root.insert("model".to_string(), Value::String(req.model.clone()));

    let mut messages: Vec<Value> = Vec::new();
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
            NeutralRole::User => {
                messages.push(json!({
                    "role": "user",
                    "content": serialize_user_blocks(&msg.content),
                }));
            }
            NeutralRole::Assistant => {
                messages.push(json!({
                    "role": "assistant",
                    "content": serialize_assistant_blocks(&msg.content),
                }));
            }
            // Tool results are submitted as `role: "user"` messages containing
            // `tool_result` blocks (Anthropic spec).
            NeutralRole::Tool => {
                let results: Vec<Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => Some(json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content,
                            "is_error": is_error,
                        })),
                        _ => None,
                    })
                    .collect();
                if !results.is_empty() {
                    messages.push(json!({ "role": "user", "content": results }));
                }
            }
        }
    }

    if system_text.len() == 1 {
        root.insert(
            "system".to_string(),
            Value::String(system_text.pop().unwrap()),
        );
    } else if !system_text.is_empty() {
        root.insert(
            "system".to_string(),
            Value::Array(
                system_text
                    .into_iter()
                    .map(|t| json!({ "type": "text", "text": t }))
                    .collect(),
            ),
        );
    }

    root.insert("messages".to_string(), Value::Array(messages));

    if !req.tools.is_empty() {
        root.insert(
            "tools".to_string(),
            Value::Array(req.tools.iter().map(serialize_tool).collect()),
        );
    }

    root.insert(
        "max_tokens".to_string(),
        json!(
            req.max_tokens
                .or(req.max_completion_tokens)
                .unwrap_or(DEFAULT_MAX_TOKENS)
        ),
    );
    if let Some(t) = req.temperature {
        root.insert("temperature".to_string(), json!(t));
    }
    if let Some(tp) = req.top_p {
        root.insert("top_p".to_string(), json!(tp));
    }
    if let Some(tk) = req.top_k {
        root.insert("top_k".to_string(), json!(tk));
    }
    if let Some(stop) = &req.stop {
        root.insert(
            "stop_sequences".to_string(),
            Value::Array(stop.iter().cloned().map(Value::String).collect()),
        );
    }
    if req.stream {
        root.insert("stream".to_string(), Value::Bool(true));
    }

    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
}

fn serialize_user_blocks(content: &[ContentBlock]) -> Value {
    let blocks: Vec<Value> = content.iter().filter_map(serialize_user_block).collect();
    if blocks.len() == 1 && blocks[0].get("type") == Some(&Value::String("text".to_string())) {
        blocks[0].get("text").cloned().unwrap_or(Value::Null)
    } else if blocks.is_empty() {
        Value::String(String::new())
    } else {
        Value::Array(blocks)
    }
}

fn serialize_user_block(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Text(t) => Some(json!({ "type": "text", "text": t })),
        ContentBlock::Image { media_type, base64 } => {
            if media_type == "url" {
                Some(json!({
                    "type": "image",
                    "source": { "type": "url", "url": base64 },
                }))
            } else {
                Some(json!({
                    "type": "image",
                    "source": { "type": "base64", "media_type": media_type, "data": base64 },
                }))
            }
        }
        _ => None,
    }
}

fn serialize_assistant_blocks(content: &[ContentBlock]) -> Value {
    let blocks: Vec<Value> = content
        .iter()
        .filter_map(serialize_assistant_block)
        .collect();
    if blocks.is_empty() {
        Value::String(String::new())
    } else {
        Value::Array(blocks)
    }
}

fn serialize_assistant_block(block: &ContentBlock) -> Option<Value> {
    match block {
        ContentBlock::Text(t) => Some(json!({ "type": "text", "text": t })),
        ContentBlock::Image { media_type, base64 } => {
            // Model-generated images (e.g. Gemini inlineData routed to an
            // Anthropic client) must not vanish silently.
            if media_type == "url" {
                Some(json!({
                    "type": "image",
                    "source": { "type": "url", "url": base64 },
                }))
            } else {
                Some(json!({
                    "type": "image",
                    "source": { "type": "base64", "media_type": media_type, "data": base64 },
                }))
            }
        }
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut b = Map::new();
            b.insert("type".to_string(), Value::String("thinking".to_string()));
            b.insert("thinking".to_string(), Value::String(thinking.clone()));
            if let Some(sig) = signature {
                b.insert("signature".to_string(), Value::String(sig.clone()));
            }
            Some(Value::Object(b))
        }
        ContentBlock::ToolUse { id, name, input } => Some(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        })),
        // Redacted thinking must be echoed back verbatim (Anthropic
        // requirement for continued extended-thinking conversations).
        ContentBlock::RedactedThinking { data } => Some(json!({
            "type": "redacted_thinking",
            "data": data,
        })),
        _ => None,
    }
}

fn serialize_tool(tool: &NeutralTool) -> Value {
    let mut t = Map::new();
    t.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(desc) = &tool.description {
        t.insert("description".to_string(), Value::String(desc.clone()));
    }
    t.insert("input_schema".to_string(), tool.parameters.clone());
    Value::Object(t)
}

/// Parse an Anthropic `/v1/messages` response body into the neutral model.
pub fn parse_response(body: &str) -> Result<NeutralResponse, AdapterError> {
    let root: Value = serde_json::from_str(body)
        .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| AdapterError::InvalidRequest("expected a JSON object".to_string()))?;

    if root.contains_key("error") || root.contains_key("type") && root["type"] == "error" {
        let status = root.get("status").and_then(Value::as_u64).unwrap_or(500) as u16;
        return Err(AdapterError::Upstream {
            status,
            body: body.to_string(),
        });
    }

    let id = root
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let model = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let content = parse_content_blocks(root.get("content"));
    let finish_reason = root
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(parse_stop_reason)
        .unwrap_or(FinishReason::Stop);

    let usage = root.get("usage").and_then(parse_usage);

    Ok(NeutralResponse {
        id,
        model,
        content,
        finish_reason,
        usage,
    })
}

fn parse_stop_reason(s: &str) -> FinishReason {
    match s {
        "end_turn" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "stop_sequence" => FinishReason::Stop,
        "refusal" => FinishReason::ContentFilter,
        "ping" => FinishReason::Other("ping".to_string()),
        other => FinishReason::Other(other.to_string()),
    }
}

fn parse_usage(usage: &Value) -> Option<NeutralUsage> {
    Some(NeutralUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    })
}

/// Serialize a neutral response into an Anthropic `/v1/messages` body.
pub fn serialize_response(resp: &NeutralResponse) -> Result<String, AdapterError> {
    let mut root = Map::new();
    root.insert("id".to_string(), Value::String(resp.id.clone()));
    root.insert("type".to_string(), Value::String("message".to_string()));
    root.insert("role".to_string(), Value::String("assistant".to_string()));
    root.insert("model".to_string(), Value::String(resp.model.clone()));
    // Anthropic response `content` must be an array of blocks (never a
    // bare string), even when empty.
    root.insert(
        "content".to_string(),
        Value::Array(
            resp.content
                .iter()
                .filter_map(serialize_assistant_block)
                .collect(),
        ),
    );
    root.insert(
        "stop_reason".to_string(),
        Value::String(serialize_stop_reason(&resp.finish_reason).to_string()),
    );
    root.insert("stop_sequence".to_string(), Value::Null);
    if let Some(u) = &resp.usage {
        root.insert(
            "usage".to_string(),
            json!({
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
            }),
        );
    }

    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
}

fn serialize_stop_reason(fr: &FinishReason) -> &str {
    match fr {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCalls => "tool_use",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Other(s) => s,
    }
}

/// Render an error in the Anthropic native shape.
pub fn serialize_error(err: &AdapterError) -> (u16, String) {
    let (status, etype) = match err {
        AdapterError::InvalidRequest(_) => (400, "invalid_request_error"),
        AdapterError::Authentication => (401, "authentication_error"),
        AdapterError::PermissionDenied => (403, "permission_error"),
        AdapterError::RateLimit { .. } => (429, "rate_limit_error"),
        AdapterError::InsufficientQuota => (429, "rate_limit_error"),
        AdapterError::Overloaded => (529, "overloaded_error"),
        AdapterError::Api(_) => (500, "api_error"),
        AdapterError::Upstream { status, .. } => (*status, "api_error"),
        AdapterError::Internal(_) => (500, "api_error"),
    };
    let body = json!({
        "type": "error",
        "error": {
            "type": etype,
            "message": err.to_string(),
        }
    })
    .to_string();
    (status, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_response_serializes_content_as_array() {
        // Anthropic `/v1/messages` responses require `content` to be an
        // array; a string would be rejected by compliant clients.
        let resp = NeutralResponse {
            id: "r1".into(),
            model: "m".into(),
            content: vec![],
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        let out = serialize_response(&resp).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["content"], Value::Array(vec![]));
    }

    #[test]
    fn redacted_thinking_round_trips_verbatim() {
        // Anthropic requires redacted_thinking blocks to be echoed back
        // verbatim (opaque `data` payload) in continued conversations.
        let body = r#"{
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
            "content": [{"type": "redacted_thinking", "data": "Zm9vYmFy"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 5, "output_tokens": 3}
        }"#;
        let resp = parse_response(body).unwrap();
        match &resp.content[0] {
            ContentBlock::RedactedThinking { data } => assert_eq!(data, "Zm9vYmFy"),
            other => panic!("expected redacted thinking, got {other:?}"),
        }
        let out = serialize_response(&resp).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["content"][0]["type"], "redacted_thinking");
        assert_eq!(v["content"][0]["data"], "Zm9vYmFy");
    }

    #[test]
    fn mixed_user_message_with_tool_result_splits() {
        // Anthropic permits text + tool_result in one user message; the
        // split must keep BOTH (results as Tool, text as User).
        let body = r#"{
            "model": "claude",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "thanks!"},
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "22c"}
                ]}
            ]
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.messages.len(), 3, "mixed message splits into two");
        assert_eq!(req.messages[1].role, NeutralRole::Tool);
        assert!(matches!(
            &req.messages[1].content[0],
            ContentBlock::ToolResult { .. }
        ));
        assert_eq!(req.messages[2].role, NeutralRole::User);
        assert!(matches!(&req.messages[2].content[0], ContentBlock::Text(t) if t == "thanks!"));
        // Anthropic round trip: Tool messages serialize as user messages
        // with tool_result blocks (spec shape); the text stays a user
        // message.
        let out = serialize_request(&req).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[1]["content"][0]["content"], "22c");
        assert_eq!(msgs[2]["role"], "user");
        // Single-text messages collapse to a bare string (spec shape).
        assert_eq!(msgs[2]["content"], "thanks!");
        // The split guarantees the OpenAI serializer sees a Tool message
        // (results) + a User message (text) — nothing to drop.
    }

    #[test]
    fn response_with_image_serializes_as_image_block() {
        let resp = NeutralResponse {
            id: "r1".into(),
            model: "claude".into(),
            content: vec![
                ContentBlock::Text("here is the badge:".into()),
                ContentBlock::Image {
                    media_type: "image/png".into(),
                    base64: "AAAA".into(),
                },
            ],
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        let out = serialize_response(&resp).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][1]["type"], "image");
        assert_eq!(v["content"][1]["source"]["data"], "AAAA");
    }

    #[test]
    fn parses_basic_request() {
        let body = r#"{
            "model": "claude-3-5-sonnet",
            "system": "be concise",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
            ],
            "max_tokens": 256,
            "temperature": 0.5,
            "stop_sequences": ["END"],
            "stream": false
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet");
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, NeutralRole::System);
        assert_eq!(
            req.messages[0].content[0],
            ContentBlock::Text("be concise".into())
        );
        assert_eq!(req.messages[2].content[0], ContentBlock::Text("hi".into()));
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.stop, Some(vec!["END".into()]));
    }

    #[test]
    fn parses_system_array() {
        let body = r#"{
            "model": "m",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "x"}]
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.messages[0].role, NeutralRole::System);
        assert_eq!(req.messages[0].content.len(), 2);
    }

    #[test]
    fn parses_content_blocks() {
        let body = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUFB"}},
                {"type": "image", "source": {"type": "url", "url": "https://ex.com/i.png"}}
            ]}]
        }"#;
        let req = parse_request(body).unwrap();
        let blocks = &req.messages[0].content;
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[1],
            ContentBlock::Image {
                media_type: "image/png".into(),
                base64: "QUFB".into(),
            }
        );
        assert_eq!(
            blocks[2],
            ContentBlock::Image {
                media_type: "url".into(),
                base64: "https://ex.com/i.png".into(),
            }
        );
    }

    #[test]
    fn parses_thinking_and_tools() {
        let body = r#"{
            "model": "m",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig-1"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "paris"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "22c", "is_error": false}]}
            ],
            "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}]
        }"#;
        let req = parse_request(body).unwrap();
        let assistant = &req.messages[0];
        assert_eq!(
            assistant.content[0],
            ContentBlock::Thinking {
                thinking: "hmm".into(),
                signature: Some("sig-1".into()),
            }
        );
        match &assistant.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "paris");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        let tool = &req.messages[1];
        // Tool-result-bearing user messages normalize to the Tool role so
        // serializers for every upstream protocol emit them as results
        // (previously they were dropped, breaking the tool loop).
        assert_eq!(tool.role, NeutralRole::Tool);
        assert_eq!(
            tool.content[0],
            ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "22c".into(),
                is_error: false,
            }
        );
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(
            req.tools[0].parameters["type"],
            Value::String("object".into())
        );
    }

    #[test]
    fn serializes_tool_results_as_user_messages() {
        let req = NeutralRequest {
            model: "claude".into(),
            messages: vec![
                NeutralMessage {
                    role: NeutralRole::System,
                    content: vec![ContentBlock::Text("sys".into())],
                },
                NeutralMessage {
                    role: NeutralRole::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "tu_9".into(),
                        content: "ok".into(),
                        is_error: true,
                    }],
                },
            ],
            tools: vec![NeutralTool {
                name: "f".into(),
                description: None,
                parameters: json!({"type": "object"}),
            }],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: false,
        };
        let out = serialize_request(&req).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["system"], "sys");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(v["messages"][0]["content"][0]["tool_use_id"], "tu_9");
        assert_eq!(v["messages"][0]["content"][0]["is_error"], true);
        assert_eq!(v["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(v["max_tokens"], 1024);
    }

    #[test]
    fn serializes_user_message_with_single_text_as_string() {
        let req = NeutralRequest::new(
            "m",
            vec![NeutralMessage {
                role: NeutralRole::User,
                content: vec![ContentBlock::Text("hi".into())],
            }],
        );
        let out = serialize_request(&req).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["messages"][0]["content"], "hi");
    }

    #[test]
    fn parses_response() {
        let body = r#"{
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "thinking", "thinking": "think"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "tu_2", "name": "f", "input": {}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp = parse_response(body).unwrap();
        assert_eq!(resp.id, "msg_01");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
        assert!(matches!(resp.content[0], ContentBlock::Thinking { .. }));
        assert!(matches!(resp.content[1], ContentBlock::Text(_)));
        assert!(matches!(resp.content[2], ContentBlock::ToolUse { .. }));
        assert_eq!(
            resp.usage,
            Some(NeutralUsage {
                input_tokens: 10,
                output_tokens: 5
            })
        );
    }

    #[test]
    fn parse_response_stop_reason_mapping() {
        let body = r#"{"id":"m","type":"message","role":"assistant","model":"m","content":[],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let resp = parse_response(body).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    #[test]
    fn serialize_response_round_trips() {
        let resp = NeutralResponse {
            id: "msg_x".into(),
            model: "claude".into(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: Some("sig".into()),
                },
                ContentBlock::Text("answer".into()),
                ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "f".into(),
                    input: json!({"a": 1}),
                },
            ],
            finish_reason: FinishReason::Stop,
            usage: Some(NeutralUsage {
                input_tokens: 3,
                output_tokens: 7,
            }),
        };
        let out = serialize_response(&resp).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["content"][0]["type"], "thinking");
        assert_eq!(v["content"][0]["signature"], "sig");
        assert_eq!(v["content"][1]["text"], "answer");
        assert_eq!(v["content"][2]["type"], "tool_use");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"]["input_tokens"], 3);

        let parsed = parse_response(&out).unwrap();
        assert_eq!(parsed.content, resp.content);
        assert_eq!(parsed.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn errors_map_to_anthropic_shape() {
        let (status, body) = serialize_error(&AdapterError::Overloaded);
        assert_eq!(status, 529);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "overloaded_error");
        assert!(v["error"]["message"].as_str().is_some());
    }
}
