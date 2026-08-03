use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use super::error::AdapterError;
use super::neutral::{ModelInfo, NeutralRequest, NeutralResponse, NeutralStreamEvent};

/// What kind of conversation endpoint an inbound path represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    /// OpenAI-style chat completions.
    Chat,
    /// Anthropic-style messages.
    Messages,
    /// Model listing.
    Models,
    /// Token counting (e.g. Anthropic `/v1/messages/count_tokens`).
    CountTokens,
}

/// Stateful decoder: upstream SSE line (the JSON payload after `data: `) →
/// neutral stream events.
pub trait StreamDecoder: Send {
    fn feed(&mut self, data: &str) -> Vec<NeutralStreamEvent>;
    /// Called when the upstream stream ends (`[DONE]` or connection close) so
    /// pending events can be flushed.
    fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        Vec::new()
    }
}

/// Stateful encoder: neutral stream event → client SSE lines (excluding the
/// SSE `data: ` framing prefix decision, adapters emit the full text incl.
/// `data: ` prefix and blank line terminator).
pub trait StreamEncoder: Send {
    fn encode(&mut self, event: NeutralStreamEvent) -> Vec<String>;
    /// Terminal line(s) to send at the end of the stream (e.g. `data: [DONE]`).
    fn done(&self) -> String;
}

/// A pluggable protocol adapter. Each protocol implements the full set:
/// parse/serialize requests and responses, stream decode/encode, error
/// mapping. The core pipeline never sees protocol-specific shapes.
pub trait ProtocolAdapter: Send + Sync {
    /// Stable protocol name used in config (`openai`, `anthropic`, ...).
    fn name(&self) -> &'static str;

    /// Inbound endpoint paths this protocol serves on the client side.
    fn endpoints(&self) -> Vec<(&'static str, EndpointKind)>;

    /// Upstream conversation URL for this protocol, given the configured
    /// provider base URL. The `model` argument allows path-parameterized
    /// protocols (e.g. Gemini `/{model}:generateContent`).
    fn conversation_url(&self, base: &str, model: &str) -> String;

    /// Upstream conversation URL for a *streaming* request. Defaults to the
    /// non-streaming URL; protocols with a distinct streaming endpoint
    /// (e.g. Gemini `/{model}:streamGenerateContent?alt=sse`) override this.
    fn stream_conversation_url(&self, base: &str, model: &str) -> String {
        self.conversation_url(base, model)
    }

    /// Extra headers required by this protocol on upstream requests
    /// (e.g. `anthropic-version`).
    fn request_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Upstream path of this protocol's model listing, if it has one.
    fn models_path(&self) -> Option<&'static str> {
        None
    }

    /// Parse an upstream model listing into neutral model info. The default
    /// handles the common `{"data": [{"id": ..., "owned_by": ...}]}`
    /// shape.
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
                    id,
                    owned_by: m
                        .get("owned_by")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                })
            })
            .collect();
        Ok(models)
    }

    /// Serialize a neutral model list into this protocol's native shape.
    fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError>;

    fn parse_request(&self, body: &str) -> Result<NeutralRequest, AdapterError>;
    fn serialize_request(&self, req: &NeutralRequest) -> Result<String, AdapterError>;

    fn parse_response(&self, body: &str) -> Result<NeutralResponse, AdapterError>;
    fn serialize_response(&self, resp: &NeutralResponse) -> Result<String, AdapterError>;

    /// Map a non-2xx upstream response onto the error taxonomy. Adapters
    /// should parse the upstream's native error body for better fidelity.
    fn parse_upstream_error(&self, status: u16, body: &str) -> AdapterError {
        AdapterError::Upstream {
            status,
            body: body.to_string(),
        }
    }

    /// Render an error in this protocol's native shape. Returns (HTTP status,
    /// JSON body).
    fn serialize_error(&self, err: &AdapterError) -> (u16, String);

    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;
    fn stream_encoder(&self) -> Box<dyn StreamEncoder>;
}

/// Named collection of registered protocol adapters.
#[derive(Clone, Default)]
pub struct ProtocolRegistry {
    adapters: HashMap<String, Arc<dyn ProtocolAdapter>>,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, adapter: Arc<dyn ProtocolAdapter>) {
        self.adapters.insert(adapter.name().to_string(), adapter);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ProtocolAdapter>> {
        self.adapters.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.adapters.keys().cloned().collect();
        names.sort();
        names
    }

    /// Adapters for the configured client protocols, preserving config order.
    pub fn client_adapters(&self, protocols: &[String]) -> Vec<Arc<dyn ProtocolAdapter>> {
        protocols.iter().filter_map(|name| self.get(name)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;

    impl ProtocolAdapter for Dummy {
        fn name(&self) -> &'static str {
            "dummy"
        }
        fn endpoints(&self) -> Vec<(&'static str, EndpointKind)> {
            vec![("/v1/chat/completions", EndpointKind::Chat)]
        }
        fn conversation_url(&self, base: &str, _model: &str) -> String {
            format!("{}/chat/completions", base.trim_end_matches('/'))
        }
        fn parse_request(&self, body: &str) -> Result<NeutralRequest, AdapterError> {
            let v: serde_json::Value = serde_json::from_str(body)
                .map_err(|e| AdapterError::InvalidRequest(e.to_string()))?;
            Ok(NeutralRequest::new(
                v["model"].as_str().unwrap_or("").to_string(),
                Vec::new(),
            ))
        }
        fn serialize_request(&self, req: &NeutralRequest) -> Result<String, AdapterError> {
            Ok(serde_json::json!({
                "model": req.model,
                "messages": req.messages.len(),
            })
            .to_string())
        }
        fn parse_response(&self, body: &str) -> Result<NeutralResponse, AdapterError> {
            let v: serde_json::Value = serde_json::from_str(body)
                .map_err(|e| AdapterError::InvalidRequest(e.to_string()))?;
            Ok(NeutralResponse {
                id: v["id"].as_str().unwrap_or("").to_string(),
                model: v["model"].as_str().unwrap_or("").to_string(),
                content: Vec::new(),
                finish_reason: super::super::neutral::FinishReason::Stop,
                usage: None,
            })
        }
        fn serialize_response(&self, resp: &NeutralResponse) -> Result<String, AdapterError> {
            Ok(serde_json::json!({
                "id": resp.id,
                "model": resp.model,
            })
            .to_string())
        }
        fn serialize_error(&self, err: &AdapterError) -> (u16, String) {
            (500, format!("{{\"error\": \"{err}\"}}"))
        }
        fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError> {
            let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
            Ok(serde_json::json!({ "data": ids }).to_string())
        }
        fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
            Box::new(NoopDecoder)
        }
        fn stream_encoder(&self) -> Box<dyn StreamEncoder> {
            Box::new(NoopEncoder)
        }
    }

    struct NoopDecoder;
    impl StreamDecoder for NoopDecoder {
        fn feed(&mut self, _data: &str) -> Vec<NeutralStreamEvent> {
            Vec::new()
        }
    }

    struct NoopEncoder;
    impl StreamEncoder for NoopEncoder {
        fn encode(&mut self, _event: NeutralStreamEvent) -> Vec<String> {
            Vec::new()
        }
        fn done(&self) -> String {
            "data: [DONE]\n\n".to_string()
        }
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = ProtocolRegistry::new();
        reg.register(Arc::new(Dummy));
        assert!(reg.get("dummy").is_some());
        assert!(reg.get("missing").is_none());
        assert_eq!(reg.names(), vec!["dummy".to_string()]);
    }

    #[test]
    fn client_adapters_respects_order() {
        let mut reg = ProtocolRegistry::new();
        reg.register(Arc::new(Dummy));
        let protocols = vec!["dummy".to_string(), "nope".to_string()];
        let adapters = reg.client_adapters(&protocols);
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].name(), "dummy");
    }
}
