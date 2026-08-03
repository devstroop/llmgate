//! Anthropic messages protocol adapter.
//!
//! Converts between the Anthropic `/v1/messages` wire format and the neutral
//! model, both request and response directions.

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
pub use stream::{AnthropicStreamDecoder, AnthropicStreamEncoder};

/// Anthropic messages protocol adapter.
pub struct AnthropicAdapter;

impl Default for AnthropicAdapter {
    fn default() -> Self {
        Self
    }
}

impl ProtocolAdapter for AnthropicAdapter {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn endpoints(&self) -> Vec<(&'static str, EndpointKind)> {
        vec![
            ("/v1/messages", EndpointKind::Messages),
            ("/v1/messages/count_tokens", EndpointKind::CountTokens),
            ("/v1/models", EndpointKind::Models),
        ]
    }

    fn conversation_url(&self, base: &str, _model: &str) -> String {
        format!("{}/v1/messages", base.trim_end_matches('/'))
    }

    fn models_path(&self) -> Option<&'static str> {
        Some("/v1/models")
    }

    fn parse_models(&self, body: &str) -> Result<Vec<ModelInfo>, AdapterError> {
        let root: Value = serde_json::from_str(body)
            .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
        let data = root
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| AdapterError::InvalidRequest("expected a data array".to_string()))?;
        let models = data
            .iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?.to_string();
                Some(ModelInfo {
                    owned_by: m
                        .get("display_name")
                        .or_else(|| m.get("owner"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    id,
                })
            })
            .collect();
        Ok(models)
    }

    fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError> {
        let data: Vec<Value> = models
            .iter()
            .map(|m| {
                json!({
                    "type": "model",
                    "id": m.id,
                    "display_name": m.owned_by,
                    "created_at": "1970-01-01T00:00:00Z",
                })
            })
            .collect();
        serde_json::to_string(&json!({ "data": data }))
            .map_err(|e| AdapterError::Internal(e.to_string()))
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![("anthropic-version".to_string(), "2023-06-01".to_string())]
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
        Box::new(AnthropicStreamDecoder::new())
    }

    fn stream_encoder(&self) -> Box<dyn StreamEncoder> {
        Box::new(AnthropicStreamEncoder::new())
    }
}

/// Convenience for registering with the registry.
pub fn adapter() -> Arc<AnthropicAdapter> {
    Arc::new(AnthropicAdapter)
}
