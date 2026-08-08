//! OpenAI ↔ neutral conversion (non-streaming).

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{
    ContentBlock, FinishReason, NeutralMessage, NeutralRequest, NeutralResponse, NeutralRole,
    NeutralTool, NeutralUsage,
};

/// Parse an OpenAI `/v1/chat/completions` request body into the neutral model.
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
    if let Some(msgs) = root.get("messages").and_then(Value::as_array) {
        for msg in msgs {
            if let Some(m) = parse_message(msg) {
                messages.push(m);
            }
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

    let stop = match root.get("stop") {
        Some(Value::String(s)) => Some(vec![s.clone()]),
        Some(Value::Array(arr)) => {
            let vals: Vec<String> = arr
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect();
            if vals.is_empty() { None } else { Some(vals) }
        }
        _ => None,
    };

    Ok(NeutralRequest {
        model,
        messages,
        tools,
        max_tokens: root
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        // Prefer max_completion_tokens (the field reasoning models accept);
        // the serializer re-emits whichever field the client used.
        max_completion_tokens: root
            .get("max_completion_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        temperature: root.get("temperature").and_then(Value::as_f64),
        top_p: root.get("top_p").and_then(Value::as_f64),
        top_k: None,
        stop,
        stream: root.get("stream").and_then(Value::as_bool).unwrap_or(false),
    })
}

fn parse_message(msg: &Value) -> Option<NeutralMessage> {
    let role = msg.get("role")?.as_str()?;
    let content = parse_content(msg);

    let neutral_role = match role {
        "system" => Some(NeutralRole::System),
        "user" => Some(NeutralRole::User),
        "assistant" => Some(NeutralRole::Assistant),
        "tool" | "function" => Some(NeutralRole::Tool),
        _ => None,
    };
    let neutral_role = neutral_role?;

    let mut blocks = content;
    if neutral_role == NeutralRole::Assistant {
        if let Some(rc) = msg
            .get("reasoning_content")
            .or_else(|| msg.get("reasoning"))
            .and_then(Value::as_str)
            .filter(|rc| !rc.is_empty())
        {
            blocks.push(ContentBlock::Thinking {
                thinking: rc.to_string(),
                signature: None,
            });
        }
        if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                if let Some(block) = parse_tool_call(call) {
                    blocks.push(block);
                }
            }
        }
    } else if neutral_role == NeutralRole::Tool {
        let tool_use_id = msg
            .get("tool_call_id")
            .or_else(|| msg.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let tool_text = blocks
            .first()
            .and_then(ContentBlock::text)
            .unwrap_or_default()
            .to_string();
        blocks = vec![ContentBlock::ToolResult {
            tool_use_id,
            content: tool_text,
            is_error: false,
        }];
    }

    Some(NeutralMessage {
        role: neutral_role,
        content: blocks,
    })
}

/// OpenAI message content: string, null, or an array of typed parts.
fn parse_content(msg: &Value) -> Vec<ContentBlock> {
    match msg.get("content") {
        Some(Value::String(s)) if !s.is_empty() => vec![ContentBlock::Text(s.clone())],
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                if let Some(b) = parse_content_part(part) {
                    blocks.push(b);
                }
            }
            blocks
        }
        _ => Vec::new(),
    }
}

fn parse_content_part(part: &Value) -> Option<ContentBlock> {
    let ptype = part.get("type")?.as_str()?;
    match ptype {
        "text" => {
            let text = part.get("text")?.as_str()?.to_string();
            Some(ContentBlock::Text(text))
        }
        "image_url" => {
            let url = part
                .get("image_url")
                .and_then(|i| i.get("url"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Some(image_from_url(&url))
        }
        _ => None,
    }
}

/// Image sources: `data:<media>;base64,<data>` data URIs or remote URLs.
/// Convention (see `core::neutral::ContentBlock::Image`): remote URLs are
/// stored with `media_type = "url"` and the URL in `base64`.
fn image_from_url(url: &str) -> ContentBlock {
    if let Some((meta, data)) = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
        .and_then(|(meta, data)| meta.strip_suffix(";base64").map(|m| (m, data)))
    {
        return ContentBlock::Image {
            media_type: meta.to_string(),
            base64: data.to_string(),
        };
    }
    ContentBlock::Image {
        media_type: "url".to_string(),
        base64: url.to_string(),
    }
}

fn parse_tool_call(call: &Value) -> Option<ContentBlock> {
    let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
    let f = call.get("function")?;
    let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
    let arguments = f.get("arguments").and_then(Value::as_str).unwrap_or("{}");
    let input = serde_json::from_str(arguments).unwrap_or(Value::Object(Map::new()));
    Some(ContentBlock::ToolUse {
        id: id.to_string(),
        name: name.to_string(),
        input,
    })
}

fn parse_tool(tool: &Value) -> Option<NeutralTool> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let f = tool.get("function")?;
    Some(NeutralTool {
        name: f
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
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

/// Serialize a neutral request into an OpenAI `/v1/chat/completions` body.
pub fn serialize_request(req: &NeutralRequest) -> Result<String, AdapterError> {
    let mut root = Map::new();
    root.insert("model".to_string(), Value::String(req.model.clone()));
    root.insert(
        "messages".to_string(),
        Value::Array(req.messages.iter().flat_map(serialize_message).collect()),
    );
    if !req.tools.is_empty() {
        root.insert(
            "tools".to_string(),
            Value::Array(req.tools.iter().map(serialize_tool).collect()),
        );
    }
    if let Some(mc) = req.max_completion_tokens {
        root.insert("max_completion_tokens".to_string(), json!(mc));
    } else if let Some(mt) = req.max_tokens {
        root.insert("max_tokens".to_string(), json!(mt));
    }
    if let Some(t) = req.temperature {
        root.insert("temperature".to_string(), json!(t));
    }
    if let Some(tp) = req.top_p {
        root.insert("top_p".to_string(), json!(tp));
    }
    if let Some(stop) = &req.stop {
        root.insert(
            "stop".to_string(),
            if stop.len() == 1 {
                Value::String(stop[0].clone())
            } else {
                Value::Array(stop.iter().cloned().map(Value::String).collect())
            },
        );
    }
    if req.stream {
        root.insert("stream".to_string(), Value::Bool(true));
    }
    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
}

fn serialize_message(msg: &NeutralMessage) -> Vec<Value> {
    let role = match msg.role {
        NeutralRole::System => "system",
        NeutralRole::User => "user",
        NeutralRole::Assistant => "assistant",
        NeutralRole::Tool => "tool",
    };

    match msg.role {
        NeutralRole::Tool => {
            // One OpenAI "tool" message per tool result (each carries a
            // single tool_call_id); a multi-result Anthropic message must
            // not collapse into one.
            msg.content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => Some(json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content,
                    })),
                    _ => None,
                })
                .collect()
        }
        NeutralRole::System | NeutralRole::User => {
            let has_images = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Image { .. }));
            if !has_images {
                let text: Vec<String> = msg
                    .content
                    .iter()
                    .filter_map(ContentBlock::text)
                    .map(String::from)
                    .collect();
                vec![json!({
                    "role": role,
                    "content": text.join("\n"),
                })]
            } else {
                // Preserve text/image interleaving order: emit one part
                // per block as encountered, not grouped by type.
                let mut parts: Vec<Value> = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            parts.push(json!({ "type": "text", "text": t }));
                        }
                        ContentBlock::Image { media_type, base64 } => {
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": image_to_url(media_type, base64) },
                            }));
                        }
                        _ => {}
                    }
                }
                vec![json!({ "role": role, "content": parts })]
            }
        }
        NeutralRole::Assistant => {
            let text: Vec<String> = msg
                .content
                .iter()
                .filter_map(ContentBlock::text)
                .map(String::from)
                .collect();
            let reasoning: Vec<String> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Thinking { thinking, .. } => Some(thinking.clone()),
                    _ => None,
                })
                .collect();
            let tool_calls: Vec<Value> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": input.to_string(),
                        },
                    })),
                    _ => None,
                })
                .collect();

            let mut m = Map::new();
            m.insert("role".to_string(), Value::String("assistant".to_string()));
            m.insert("content".to_string(), Value::String(text.join("\n")));
            if !reasoning.is_empty() {
                m.insert(
                    "reasoning_content".to_string(),
                    Value::String(reasoning.join("\n")),
                );
            }
            if !tool_calls.is_empty() {
                m.insert("tool_calls".to_string(), Value::Array(tool_calls));
            }
            vec![Value::Object(m)]
        }
    }
}

fn serialize_tool(tool: &NeutralTool) -> Value {
    let mut function = Map::new();
    function.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(desc) = &tool.description {
        function.insert("description".to_string(), Value::String(desc.clone()));
    }
    function.insert("parameters".to_string(), tool.parameters.clone());
    json!({ "type": "function", "function": function })
}

fn image_to_url(media_type: &str, base64: &str) -> String {
    if media_type == "url" {
        base64.to_string()
    } else {
        format!("data:{media_type};base64,{base64}")
    }
}

/// Parse an OpenAI `/v1/chat/completions` response body into the neutral model.
pub fn parse_response(body: &str) -> Result<NeutralResponse, AdapterError> {
    let root: Value = serde_json::from_str(body)
        .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| AdapterError::InvalidRequest("expected a JSON object".to_string()))?;

    if root.contains_key("error") {
        let status = root
            .get("error")
            .and_then(|e| e.get("status"))
            .and_then(Value::as_u64)
            .unwrap_or(500) as u16;
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

    let mut content = Vec::new();
    let mut finish_reason = FinishReason::Stop;
    if let Some(choice) = root
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    {
        content = choice
            .get("message")
            .map(parse_message_blocks)
            .unwrap_or_default();
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            finish_reason = parse_finish_reason(fr);
        }
    }

    let usage = root.get("usage").and_then(parse_usage);

    Ok(NeutralResponse {
        id,
        model,
        content,
        finish_reason,
        usage,
    })
}

fn parse_message_blocks(msg: &Value) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    match msg.get("content") {
        Some(Value::String(s)) if !s.is_empty() => blocks.push(ContentBlock::Text(s.clone())),
        Some(Value::Array(parts)) => {
            for part in parts {
                if let Some(b) = parse_content_part(part) {
                    blocks.push(b);
                }
            }
        }
        _ => {}
    }
    if let Some(rc) = msg
        .get("reasoning_content")
        .or_else(|| msg.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|rc| !rc.is_empty())
    {
        blocks.push(ContentBlock::Thinking {
            thinking: rc.to_string(),
            signature: None,
        });
    }
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            if let Some(b) = parse_tool_call(call) {
                blocks.push(b);
            }
        }
    }
    blocks
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

/// Serialize a neutral response into an OpenAI `/v1/chat/completions` body.
pub fn serialize_response(resp: &NeutralResponse) -> Result<String, AdapterError> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));
    let has_images = resp
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    let text: Vec<String> = resp
        .content
        .iter()
        .filter_map(ContentBlock::text)
        .map(String::from)
        .collect();
    let reasoning: Vec<String> = resp
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.clone()),
            _ => None,
        })
        .collect();
    let tool_calls: Vec<Value> = resp
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": input.to_string(),
                },
            })),
            _ => None,
        })
        .collect();
    if has_images {
        // Model-generated images (e.g. Gemini inlineData) survive as
        // multimodal content parts instead of vanishing silently.
        let mut parts: Vec<Value> = Vec::new();
        for block in &resp.content {
            match block {
                ContentBlock::Text(t) => parts.push(json!({ "type": "text", "text": t })),
                ContentBlock::Image { media_type, base64 } => parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": image_to_url(media_type, base64) },
                })),
                _ => {}
            }
        }
        message.insert("content".to_string(), Value::Array(parts));
    } else {
        message.insert("content".to_string(), Value::String(text.join("\n")));
    }
    if !reasoning.is_empty() {
        message.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning.join("\n")),
        );
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }

    let mut root = Map::new();
    root.insert("id".to_string(), Value::String(resp.id.clone()));
    root.insert(
        "object".to_string(),
        Value::String("chat.completion".to_string()),
    );
    root.insert("created".to_string(), json!(created));
    root.insert("model".to_string(), Value::String(resp.model.clone()));
    root.insert(
        "choices".to_string(),
        json!([{
            "index": 0,
            "message": message,
            "finish_reason": resp.finish_reason.as_str(),
        }]),
    );
    if let Some(u) = &resp.usage {
        root.insert(
            "usage".to_string(),
            json!({
                "prompt_tokens": u.input_tokens,
                "completion_tokens": u.output_tokens,
                "total_tokens": u.input_tokens + u.output_tokens,
            }),
        );
    }

    serde_json::to_string(&Value::Object(root)).map_err(|e| AdapterError::Internal(e.to_string()))
}

/// Render an error in the OpenAI native shape.
pub fn serialize_error(err: &AdapterError) -> (u16, String) {
    let (status, etype) = match err {
        AdapterError::InvalidRequest(_) => (400, "invalid_request_error"),
        AdapterError::Authentication => (401, "authentication_error"),
        AdapterError::PermissionDenied => (403, "permission_denied_error"),
        AdapterError::RateLimit { .. } => (429, "rate_limit_error"),
        AdapterError::InsufficientQuota => (429, "insufficient_quota"),
        AdapterError::Overloaded => (503, "overloaded_error"),
        AdapterError::Api(_) => (502, "api_error"),
        AdapterError::Upstream { status, .. } => (*status, "upstream_error"),
        AdapterError::Internal(_) => (500, "internal_error"),
    };
    let body = json!({
        "error": {
            "message": err.to_string(),
            "type": etype,
            "code": null,
        }
    })
    .to_string();
    (status, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_model(req: &NeutralRequest) -> &str {
        &req.model
    }

    #[test]
    fn parses_basic_request() {
        let body = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ],
            "temperature": 0.7,
            "stop": ["END", "STOP"],
            "stream": false
        }"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req_model(&req), "gpt-4o");
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, NeutralRole::System);
        assert_eq!(
            req.messages[1].content[0],
            ContentBlock::Text("hello".into())
        );
        assert_eq!(req.stop, Some(vec!["END".to_string(), "STOP".to_string()]));
        assert_eq!(req.temperature, Some(0.7));
        assert!(!req.stream);
    }

    #[test]
    fn parses_single_stop_as_array() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stop":"END"}"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.stop, Some(vec!["END".to_string()]));
    }

    #[test]
    fn parses_image_and_tools() {
        let body = r#"{
            "model": "m",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }]
        }"#;
        let req = parse_request(body).unwrap();
        let user = &req.messages[0];
        assert_eq!(user.content.len(), 2);
        assert_eq!(
            user.content[1],
            ContentBlock::Image {
                media_type: "image/png".into(),
                base64: "AAAA".into(),
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
    fn parses_tool_calls_and_results() {
        let body = r#"{
            "model": "m",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "22c"}
            ]
        }"#;
        let req = parse_request(body).unwrap();
        let assistant = &req.messages[0];
        assert!(!assistant.content.is_empty());
        match &assistant.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "paris");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        let tool = &req.messages[1];
        assert_eq!(tool.role, NeutralRole::Tool);
        match &tool.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert_eq!(content, "22c");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn parses_reasoning_content() {
        let body = r#"{
            "model": "m",
            "messages": [{"role": "assistant", "content": "done", "reasoning_content": "let me think"}]
        }"#;
        let req = parse_request(body).unwrap();
        let assistant = &req.messages[0];
        assert_eq!(assistant.content.len(), 2);
        assert!(matches!(
            &assistant.content[1],
            ContentBlock::Thinking { thinking, .. } if thinking == "let me think"
        ));
    }

    #[test]
    fn max_completion_tokens_round_trips() {
        // The client's token-limit field must be preserved: reasoning
        // models reject `max_tokens`, so serializing back the field the
        // client used matters.
        let body = r#"{"model":"o3-mini","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":500}"#;
        let req = parse_request(body).unwrap();
        assert_eq!(req.max_completion_tokens, Some(500));
        assert_eq!(req.max_tokens, None);
        let out = serialize_request(&req).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["max_completion_tokens"], 500);
        assert!(
            v.get("max_tokens").is_none(),
            "max_tokens must not be emitted when the client used max_completion_tokens"
        );
    }

    #[test]
    fn multimodal_blocks_keep_interleaved_order() {
        let req = NeutralRequest {
            model: "m".into(),
            messages: vec![NeutralMessage {
                role: NeutralRole::User,
                content: vec![
                    ContentBlock::Text("what is in this image?".into()),
                    ContentBlock::Image {
                        media_type: "image/png".into(),
                        base64: "AA==".into(),
                    },
                    ContentBlock::Text("and this one?".into()),
                    ContentBlock::Image {
                        media_type: "image/png".into(),
                        base64: "BB==".into(),
                    },
                ],
            }],
            tools: vec![],
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
        let parts = v["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 4, "text/image interleaving must be preserved");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[2]["type"], "text");
        assert_eq!(parts[3]["type"], "image_url");
        assert!(
            parts[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .ends_with("AA==")
        );
        assert!(
            parts[3]["image_url"]["url"]
                .as_str()
                .unwrap()
                .ends_with("BB==")
        );
    }

    #[test]
    fn tool_role_emits_one_message_per_result() {
        let req = NeutralRequest {
            model: "m".into(),
            messages: vec![NeutralMessage {
                role: NeutralRole::Tool,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: "22c".into(),
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "call_2".into(),
                        content: "paris".into(),
                        is_error: false,
                    },
                ],
            }],
            tools: vec![],
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
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "one OpenAI tool message per result");
        assert_eq!(msgs[0]["tool_call_id"], "call_1");
        assert_eq!(msgs[1]["tool_call_id"], "call_2");
        assert_eq!(msgs[1]["content"], "paris");
    }

    #[test]
    fn response_images_serialize_as_content_parts() {
        let resp = NeutralResponse {
            id: "r1".into(),
            model: "m".into(),
            content: vec![
                ContentBlock::Text("badge:".into()),
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
        let parts = v["choices"][0]["message"]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2, "images must not be dropped");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(
            parts[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .ends_with("AAAA")
        );
    }

    #[test]
    fn serialize_round_trips_tools_and_images() {
        let req = NeutralRequest {
            model: "m".into(),
            messages: vec![
                NeutralMessage {
                    role: NeutralRole::User,
                    content: vec![
                        ContentBlock::Text("look".into()),
                        ContentBlock::Image {
                            media_type: "image/png".into(),
                            base64: "AAAA".into(),
                        },
                    ],
                },
                NeutralMessage {
                    role: NeutralRole::Tool,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call_9".into(),
                        content: "ok".into(),
                        is_error: false,
                    }],
                },
            ],
            tools: vec![NeutralTool {
                name: "f".into(),
                description: Some("d".into()),
                parameters: json!({"type": "object"}),
            }],
            max_tokens: Some(100),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: Some(vec!["END".into()]),
            stream: false,
        };
        let out = serialize_request(&req).unwrap();
        let parsed = parse_request(&out).unwrap();
        assert_eq!(parsed.messages.len(), 2);
        assert!(parsed.messages[0].content.contains(&ContentBlock::Image {
            media_type: "image/png".into(),
            base64: "AAAA".into(),
        }));
        assert_eq!(parsed.tools[0].name, "f");
        assert_eq!(parsed.stop, Some(vec!["END".into()]));
        assert_eq!(parsed.max_tokens, Some(100));
    }

    #[test]
    fn parses_response_with_tools_and_usage() {
        let body = r#"{
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "sure",
                    "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let resp = parse_response(body).unwrap();
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
        assert!(matches!(resp.content[0], ContentBlock::Text(_)));
        assert!(matches!(resp.content[1], ContentBlock::ToolUse { .. }));
        assert_eq!(
            resp.usage,
            Some(NeutralUsage {
                input_tokens: 10,
                output_tokens: 5
            })
        );
    }

    #[test]
    fn parses_reasoning_response() {
        let body = r#"{
            "id": "r",
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "", "reasoning": "think"}, "finish_reason": "stop"}]
        }"#;
        let resp = parse_response(body).unwrap();
        assert!(
            matches!(&resp.content[0], ContentBlock::Thinking { thinking, .. } if thinking == "think")
        );
    }

    #[test]
    fn serialize_response_round_trips() {
        let resp = NeutralResponse {
            id: "chatcmpl-x".into(),
            model: "gpt-4o".into(),
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
        assert_eq!(parsed.content.len(), 2);
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
    fn errors_map_to_openai_shape() {
        let (status, body) = serialize_error(&AdapterError::RateLimit {
            retry_after_secs: None,
        });
        assert_eq!(status, 429);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_error");
        assert!(v["error"]["message"].as_str().is_some());
    }
}
