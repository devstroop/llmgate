/// Protocol-independent error taxonomy.
///
/// Adapters map their protocol's native errors onto this taxonomy and are
/// responsible for serializing it back into a protocol-native error body.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdapterError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("authentication failed")]
    Authentication,
    #[error("permission denied")]
    PermissionDenied,
    #[error("rate limit exceeded")]
    RateLimit { retry_after_secs: Option<u64> },
    #[error("upstream overloaded")]
    Overloaded,
    #[error("insufficient quota")]
    InsufficientQuota,
    #[error("upstream api error: {0}")]
    Api(String),
    #[error("upstream http {status}: {body}")]
    Upstream { status: u16, body: String },
    #[error("internal error: {0}")]
    Internal(String),
}
