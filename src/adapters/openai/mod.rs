//! OpenAI chat-completions protocol adapter.
//!
//! Converts between the OpenAI `/v1/chat/completions` wire format and the
//! neutral model, both request and response directions.

mod convert;
mod stream;

use std::sync::Arc;

use serde_json::{Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{ModelInfo, NeutralRequest, NeutralResponse};
use crate::core::registry::{EndpointKind, ProtocolAdapter, StreamDecoder, StreamEncoder};

pub use convert::{
    parse_request, parse_response, serialize_error, serialize_request, serialize_response,
};
pub use stream::{OpenAiStreamDecoder, OpenAiStreamEncoder};

/// OpenAI chat-completions protocol adapter.
pub struct OpenAiAdapter;

impl Default for OpenAiAdapter {
    fn default() -> Self {
        Self
    }
}

impl ProtocolAdapter for OpenAiAdapter {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn endpoints(&self) -> Vec<(&'static str, EndpointKind)> {
        vec![
            ("/v1/chat/completions", EndpointKind::Chat),
            ("/v1/models", EndpointKind::Models),
        ]
    }

    fn conversation_url(&self, base: &str, _model: &str) -> String {
        format!("{}/v1/chat/completions", base.trim_end_matches('/'))
    }

    fn models_path(&self) -> Option<&'static str> {
        Some("/v1/models")
    }

    fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError> {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let data: Vec<Value> = models
            .iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "object": "model",
                    "created": created,
                    "owned_by": m.owned_by,
                })
            })
            .collect();
        serde_json::to_string(&json!({ "object": "list", "data": data }))
            .map_err(|e| AdapterError::Internal(e.to_string()))
    }

    fn parse_request(&self, body: &str) -> Result<NeutralRequest, AdapterError> {
        parse_request(body)
    }

    fn serialize_request(&self, req: &NeutralRequest) -> Result<String, AdapterError> {
        serialize_request(req)
    }

    fn parse_response(&self, body: &str) -> Result<NeutralResponse, AdapterError> {
        parse_response(body)
    }

    fn serialize_response(&self, resp: &NeutralResponse) -> Result<String, AdapterError> {
        serialize_response(resp)
    }

    fn serialize_error(&self, err: &AdapterError) -> (u16, String) {
        serialize_error(err)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(OpenAiStreamDecoder::new())
    }

    fn stream_encoder(&self) -> Box<dyn StreamEncoder> {
        Box::new(OpenAiStreamEncoder::new())
    }
}

/// Convenience for registering with the registry.
pub fn adapter() -> Arc<OpenAiAdapter> {
    Arc::new(OpenAiAdapter)
}
