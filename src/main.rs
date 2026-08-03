use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware;
use axum::routing::{get, post};
use tokio::net::TcpListener;

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
    let mut state = AppState::new(config.clone());

    // Adapters are registered here as they land.
    let mut registry = model_adapter::core::ProtocolRegistry::new();
    registry.register(model_adapter::adapters::openai::adapter());
    registry.register(model_adapter::adapters::anthropic::adapter());
    registry.register(model_adapter::adapters::gemini::adapter());
    state.registry = registry;
    let state = Arc::new(state);

    for protocol in &config.client.protocols {
        if state.registry.get(protocol).is_none() {
            tracing::warn!("configured client protocol not registered: {protocol}");
        }
    }
    if state.registry.get(&config.upstream.protocol).is_none() {
        tracing::warn!(
            "configured upstream protocol not registered: {}",
            config.upstream.protocol
        );
    }

    let mut app = Router::<Arc<AppState>>::new().route("/health", get(health));

    let adapters = state.registry.client_adapters(&config.client.protocols);
    let mut mounted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for adapter in adapters {
        for (path, kind) in adapter.endpoints() {
            if !mounted.insert(path.to_string()) {
                tracing::warn!(
                    "skipping duplicate endpoint {} (already mounted for another protocol)",
                    path
                );
                continue;
            }
            let client_name = adapter.name().to_string();
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
                "serving {} endpoint {} (kind: {kind:?})",
                adapter.name(),
                path
            );
            app = app.route(path, handler);
        }
    }
    let app = app
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth))
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
    let mut response = span.in_scope(|| next.run(request)).await;

    response.headers_mut().insert(
        HEADER,
        request_id
            .parse()
            .unwrap_or_else(|_| panic!("generated/inbound request id is not valid header value")),
    );
    response
}

/// Generate a short unique request id for correlation.
fn generate_request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Derive a deterministic-ish value from the timestamp plus a process seed.
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u128(nanos);
    h.write_u64(std::process::id() as u64);
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
