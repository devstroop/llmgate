use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, Request, StatusCode};
use axum::middleware;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tracing::Instrument;

use model_adapter::auth::require_auth;
use model_adapter::config::Config;
use model_adapter::core::pipeline::{handle_conversation, handle_count_tokens, handle_models};
use model_adapter::core::{AppState, EndpointKind};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Config::load()?;
    let mut state = AppState::new(config.clone())?;

    // Adapters are registered here as they land.
    let mut registry = model_adapter::core::ProtocolRegistry::new();
    registry.register(model_adapter::adapters::openai::adapter());
    registry.register(model_adapter::adapters::anthropic::adapter());
    registry.register(model_adapter::adapters::gemini::adapter());
    state.registry = registry;
    let state = Arc::new(state);

    // Fail closed on misconfiguration: an unknown client protocol means
    // its endpoints would silently 404 at runtime.
    let unknown: Vec<&String> = config
        .client
        .protocols
        .iter()
        .filter(|p| state.registry.get(p).is_none())
        .collect();
    if !unknown.is_empty() {
        anyhow::bail!(
            "configured client protocol(s) {:?} are not registered (available: {:?})",
            unknown,
            state.registry.names()
        );
    }
    if state.registry.get(&config.upstream.protocol).is_none() {
        // Fail closed: an unknown upstream protocol means every request
        // would fail at runtime; refuse to start.
        anyhow::bail!(
            "configured upstream protocol {:?} is not registered (available: {:?})",
            config.upstream.protocol,
            state.registry.names()
        );
    }
    if config.auth.api_keys.is_empty() {
        tracing::warn!(
            "authentication is DISABLED (auth.api_keys is empty): every request is allowed. \
             This is only safe on a trusted network."
        );
    }

    let mut app = Router::<Arc<AppState>>::new();

    // Group endpoints by path, then build ONE MethodRouter per path:
    // axum registers routes by path and combines methods inside a single
    // MethodRouter — calling route() twice for the same path replaces the
    // first registration instead of merging it.
    let adapters = state.registry.client_adapters(&config.client.protocols);
    let mut by_path: std::collections::BTreeMap<&str, Vec<(EndpointKind, &str)>> =
        std::collections::BTreeMap::new();
    for adapter in adapters {
        for (path, kind) in adapter.endpoints() {
            by_path
                .entry(path)
                .or_default()
                .push((kind, adapter.name()));
        }
    }
    let mut mounted: std::collections::HashMap<(String, String), &str> =
        std::collections::HashMap::new();
    for (path, routes) in by_path {
        let mut router = axum::routing::MethodRouter::new();
        for (kind, protocol) in routes {
            // Two DISTINCT protocols claiming the same method+path would
            // serve one protocol's shape to the other's clients — fail
            // startup; duplicates within one protocol are tolerated.
            let method = match kind {
                EndpointKind::Models => "GET",
                EndpointKind::Chat | EndpointKind::Messages | EndpointKind::CountTokens => "POST",
            };
            let key = (method.to_string(), path.to_string());
            if let Some(owner) = mounted.get(&key) {
                if *owner != protocol {
                    if kind == EndpointKind::Models {
                        // Both openai and anthropic claim GET /v1/models.
                        // The listing handler is per-client-protocol anyway
                        // and the shapes are close enough that the first
                        // protocol's handler serves both (round-2 behavior);
                        // failing startup here would break the DEFAULT
                        // config (["openai", "anthropic"]).
                        tracing::warn!(
                            "endpoint {method} {path}: keeping {owner:?}'s handler; \
                             {protocol:?} clients get the {owner:?}-shaped model listing"
                        );
                        continue;
                    }
                    anyhow::bail!(
                        "endpoint {method} {path} is claimed by both {owner:?} and {protocol:?}; \
                         client protocols must not share a method+path (wrong-schema responses)"
                    );
                }
                tracing::warn!(
                    "skipping duplicate endpoint {method} {path} (already mounted for the same protocol)"
                );
                continue;
            }
            mounted.insert(key, protocol);
            let client_name = protocol.to_string();
            let handler = match kind {
                EndpointKind::Models => {
                    let client_name = client_name.clone();
                    get(move |State(state): State<Arc<AppState>>| {
                        handle_models(state, client_name.clone())
                    })
                }
                EndpointKind::CountTokens => {
                    let client_name = client_name.clone();
                    post(
                        move |State(state): State<Arc<AppState>>, request: Request<Body>| {
                            handle_count_tokens(state, client_name.clone(), request)
                        },
                    )
                }
                EndpointKind::Chat | EndpointKind::Messages => {
                    let client_name = client_name.clone();
                    post(
                        move |State(state): State<Arc<AppState>>, request: Request<Body>| {
                            handle_conversation(state, client_name.clone(), request)
                        },
                    )
                }
            };
            tracing::info!(
                "serving {} endpoint {method} {path} (kind: {kind:?})",
                protocol
            );
            router = router.on(method_filter(kind), handler);
        }
        app = app.route(path, router);
    }
    let app = app
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth))
        // /health is registered AFTER route_layer so the liveness probe
        // stays unauthenticated (load balancers / orchestrators have no
        // gateway API key).
        .route("/health", get(health))
        .layer(middleware::from_fn(request_trace))
        .with_state(state);

    let addr = (config.server.host.as_str(), config.server.port);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("model-adapter listening on {}:{}", addr.0, addr.1);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn method_filter(kind: EndpointKind) -> axum::routing::MethodFilter {
    match kind {
        EndpointKind::Models => axum::routing::MethodFilter::GET,
        EndpointKind::Chat | EndpointKind::Messages | EndpointKind::CountTokens => {
            axum::routing::MethodFilter::POST
        }
    }
}

async fn health() -> (StatusCode, axum::Json<serde_json::Value>) {
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({ "status": "ok" })),
    )
}

/// Attach a request id (honour an inbound `x-request-id`, else generate one),
/// thread it into the tracing span for correlation, and echo it back on the
/// response.
async fn request_trace(request: Request<Body>, next: middleware::Next) -> axum::response::Response {
    const HEADER: &str = "x-request-id";
    let request_id = request
        .headers()
        .get(HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(generate_request_id);

    let span = tracing::info_span!("request", request_id = %request_id);
    // Instrument the future itself so the span is active while the handler
    // runs (in_scope would only cover future construction).
    let mut response = next.run(request).instrument(span).await;

    response.headers_mut().insert(
        HEADER,
        // Never panic in the request path: a future regression in the
        // generated id must degrade to a static fallback, not crash the
        // whole server on a single request.
        HeaderValue::from_str(&request_id).unwrap_or(HeaderValue::from_static("req-unknown")),
    );
    response
}

/// Generate a short unique request id for correlation.
///
/// A process-wide atomic counter guarantees uniqueness under concurrent
/// load (the system clock does not necessarily advance between two
/// in-flight requests, so a timestamp alone can collide).
fn generate_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Derive a deterministic-ish value from the timestamp plus a process
    // seed plus a per-process monotonic counter.
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u128(nanos);
    h.write_u64(std::process::id() as u64);
    h.write_u64(REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("req-{:016x}", h.finish())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl-c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received, exiting");
}
