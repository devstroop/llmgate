//! Anthropic messages protocol adapter.
//!
//! Converts between the Anthropic `/v1/messages` wire format and the neutral
//! model, both request and response directions.

mod convert;

use std::sync::Arc;

use crate::core::error::AdapterError;
use crate::core::neutral::{NeutralRequest, NeutralResponse};
use crate::core::registry::{EndpointKind, ProtocolAdapter, StreamDecoder, StreamEncoder};

pub use convert::{
    parse_request, parse_response, serialize_error, serialize_request, serialize_response,
};

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
        vec![("/v1/messages", EndpointKind::Messages)]
    }

    fn conversation_url(&self, base: &str, _model: &str) -> String {
        format!("{}/v1/messages", base.trim_end_matches('/'))
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
        // Streaming lands in M4.
        Box::new(UnimplementedDecoder)
    }

    fn stream_encoder(&self) -> Box<dyn StreamEncoder> {
        Box::new(UnimplementedEncoder)
    }
}

pub struct UnimplementedDecoder;

impl StreamDecoder for UnimplementedDecoder {
    fn feed(&mut self, _data: &str) -> Vec<crate::core::neutral::NeutralStreamEvent> {
        Vec::new()
    }
}

pub struct UnimplementedEncoder;

impl StreamEncoder for UnimplementedEncoder {
    fn encode(&mut self, _event: crate::core::neutral::NeutralStreamEvent) -> Vec<String> {
        Vec::new()
    }
    fn done(&self) -> String {
        String::new()
    }
}

/// Convenience for registering with the registry.
pub fn adapter() -> Arc<AnthropicAdapter> {
    Arc::new(AnthropicAdapter)
}
