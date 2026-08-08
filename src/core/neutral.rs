use serde_json::Value;

use super::error::AdapterError;

/// A single message in a conversation, expressed with protocol-neutral
/// content blocks so that no provider shape leaks into the core.
#[derive(Debug, Clone, PartialEq)]
pub struct NeutralMessage {
    pub role: NeutralRole,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralRole {
    System,
    User,
    Assistant,
    Tool,
}

/// A protocol-neutral content block.
///
/// Mapping notes for adapters:
/// - OpenAI `image_url` content, Anthropic `image` blocks → `Image`
/// - OpenAI `reasoning_content` / `reasoning`, Anthropic `thinking` → `Thinking`
/// - OpenAI assistant `tool_calls`, Anthropic `tool_use` → `ToolUse`
/// - OpenAI `role: "tool"`, Anthropic `tool_result` → `ToolResult`
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text(String),
    Image {
        media_type: String,
        base64: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// Anthropic `redacted_thinking` block (extended thinking redaction):
    /// the opaque payload must be echoed back verbatim when a conversation
    /// with extended thinking is continued, so it is carried as-is.
    RedactedThinking {
        data: String,
    },
}

impl ContentBlock {
    pub fn text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(t) => Some(t),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeutralTool {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema describing the tool parameters.
    pub parameters: Value,
}

/// Protocol-neutral request, canonically representing any chat-completions
/// style call (OpenAI chat completions, Anthropic messages, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct NeutralRequest {
    pub model: String,
    pub messages: Vec<NeutralMessage>,
    pub tools: Vec<NeutralTool>,
    pub max_tokens: Option<u32>,
    /// OpenAI `max_completion_tokens` (the field several reasoning models
    /// accept instead of `max_tokens`). Serializers re-emit whichever
    /// field the client used.
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub stream: bool,
}

impl NeutralRequest {
    pub fn new(model: impl Into<String>, messages: Vec<NeutralMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: false,
        }
    }
}

/// Protocol-neutral response. The assistant's output is a list of content
/// blocks (text, thinking, tool use), which unifies OpenAI's `tool_calls`
/// array with Anthropic's `tool_use` blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct NeutralResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub finish_reason: FinishReason,
    pub usage: Option<NeutralUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

impl FinishReason {
    pub fn as_str(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other(s) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeutralUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A model as reported by the upstream provider's model list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub owned_by: String,
}

/// Neutral streaming events produced by a [`super::registry::StreamDecoder`]
/// and consumed by a [`super::registry::StreamEncoder`].
#[derive(Debug, Clone, PartialEq)]
pub enum NeutralStreamEvent {
    MessageStart {
        id: String,
        model: String,
        /// Initial usage if the upstream reported it at stream start
        /// (e.g. Anthropic `message_start` carries input_tokens).
        usage: Option<NeutralUsage>,
    },
    TextDelta(String),
    ReasoningDelta(String),
    /// Opaque reasoning signature (Anthropic `signature_delta`): must be
    /// echoed back verbatim when a thinking block closes so extended
    /// thinking conversations can continue.
    ReasoningSignature(String),
    /// Tool-call arguments arrive incrementally; adapters accumulate them and
    /// emit the completed call on stream end (or a zero-length delta).
    ToolCallDelta {
        index: u32,
        id: String,
        name: String,
        arguments: String,
    },
    MessageStop {
        finish_reason: FinishReason,
        usage: Option<NeutralUsage>,
    },
    Error(AdapterError),
}
