mod audio;
mod config;
mod error;
mod routes;
mod runtime;
mod state;

use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{fmt, EnvFilter};

use config::ServerConfig;
use runtime::{build_tts_runtime, RuntimeMetadata, RuntimeStatus};
use state::AppState;

use qwen3_tts::tensor::Device as TtsDevice;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (silently ignored if missing)
    dotenvy::dotenv().ok();

    // Initialize logging
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Parse configuration from environment variables
    let config = ServerConfig::from_env().map_err(|e| anyhow::anyhow!("{}", e))?;
    let transport_label = resolve_transport_label(&config);

    // Determine TTS device based on backend
    #[cfg(feature = "tch-backend")]
    let tts_device = {
        if tch::Cuda::is_available() {
            tracing::info!("TTS using CUDA GPU");
            TtsDevice::Gpu(0)
        } else {
            tracing::info!("TTS using CPU");
            TtsDevice::Cpu
        }
    };
    #[cfg(feature = "tch-backend")]
    let backend_label = if tch::Cuda::is_available() {
        "cuda"
    } else {
        "cpu"
    };

    #[cfg(feature = "mlx")]
    let tts_device = {
        qwen3_tts::backend::mlx::stream::init_mlx(true);
        tracing::info!("TTS using MLX Metal GPU");
        TtsDevice::Gpu(0)
    };
    #[cfg(feature = "mlx")]
    let backend_label = "metal-mlx";

    let (tts, runtime_metadata) =
        match build_tts_runtime(&config, tts_device, backend_label, &transport_label)? {
            Some((tts, runtime_metadata)) => (Some(tts), runtime_metadata),
            None => (
                None,
                RuntimeMetadata {
                    backend: backend_label.to_string(),
                    transport: transport_label.clone(),
                    loaded_at: std::time::SystemTime::now(),
                    status: Arc::new(Mutex::new(RuntimeStatus::ResidentHot)),
                    last_error: Arc::new(Mutex::new(None)),
                },
            ),
        };

    let state = AppState {
        tts,
        runtime: runtime_metadata,
    };

    // Build router
    let app = Router::new()
        .route("/v1/audio/speech", post(routes::speech::speech_handler))
        .route("/v1/models", get(routes::models::list_models))
        .route("/models/{id}/load", post(routes::models::load_model))
        .route("/models/{id}/offload", post(routes::models::offload_model))
        .route("/health", get(routes::health::health))
        .route("/health/details", get(routes::health::health_details))
        .route("/metrics", get(routes::health::metrics))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server on the configured local transport.
    if let Some(socket_path) = config.socket_path.clone() {
        #[cfg(unix)]
        {
            let socket_file = Path::new(&socket_path);
            if let Some(parent) = socket_file.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            let _ = tokio::fs::remove_file(socket_file).await;
            let listener = tokio::net::UnixListener::bind(socket_file)?;
            tracing::info!("Server listening on unix socket {}", socket_file.display());
            axum::serve(listener, app).await?;
            return Ok(());
        }

        #[cfg(not(unix))]
        {
            return Err(anyhow::anyhow!(
                "SOCKET_PATH is only supported on Unix platforms in this build."
            ));
        }
    }

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid bind address");
    tracing::info!("Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Build the transport label that is surfaced by health/details responses.
fn resolve_transport_label(config: &ServerConfig) -> String {
    if let Some(socket_path) = config.socket_path.as_ref() {
        return format!("unix:{socket_path}");
    }

    format!("tcp:{}:{}", config.host, config.port)
}
