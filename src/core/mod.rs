// Protocol-agnostic core: neutral model, adapter traits, registry, pipeline.

pub mod error;
pub mod neutral;
pub mod pipeline;
pub mod registry;
pub mod sse;

pub use error::AdapterError;
pub use neutral::{
    ContentBlock, FinishReason, ModelInfo, NeutralMessage, NeutralRequest, NeutralResponse,
    NeutralRole, NeutralStreamEvent, NeutralTool, NeutralUsage,
};
pub use pipeline::AppState;
pub use registry::{EndpointKind, ProtocolAdapter, ProtocolRegistry, StreamDecoder, StreamEncoder};
