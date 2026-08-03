//! OpenAI chat-completions protocol adapter.
//!
//! Converts between the OpenAI `/v1/chat/completions` wire format and the
//! neutral model, both request and response directions.

mod convert;

use std::sync::Arc;

use crate::core::error::AdapterError;
use crate::core::neutral::{NeutralRequest, NeutralResponse};
use crate::core::registry::{EndpointKind, ProtocolAdapter, StreamDecoder, StreamEncoder};

pub use convert::{
    parse_request, parse_response, serialize_error, serialize_request, serialize_response,
};

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
        vec![("/v1/chat/completions", EndpointKind::Chat)]
    }

    fn conversation_url(&self, base: &str, _model: &str) -> String {
        format!("{}/v1/chat/completions", base.trim_end_matches('/'))
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
pub fn adapter() -> Arc<OpenAiAdapter> {
    Arc::new(OpenAiAdapter)
}
