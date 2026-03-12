use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::runtime::{
    LoadedModelInventory, RuntimeMetadata, RuntimeMetricsSnapshot, RuntimeStatus, TtsRuntimeHandle,
};

/// Shared application state used by all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    pub tts: Option<TtsRuntimeHandle>,
    pub runtime: RuntimeMetadata,
}

/// Transport details returned by health/details responses.
#[derive(Debug, Clone, Serialize)]
pub struct TransportInfo {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub socket_path: Option<String>,
}

/// Detailed runtime health payload for internal management and diagnostics.
#[derive(Debug, Serialize)]
pub struct RuntimeHealthDetails {
    pub status: RuntimeStatus,
    pub backend: String,
    pub transport: TransportInfo,
    pub loaded_models: LoadedModelInventory,
    pub metrics: RuntimeMetricsSnapshot,
    pub loaded_at_unix_ms: u128,
    pub uptime_ms: u128,
    pub last_error: Option<String>,
}

impl AppState {
    /// Build a detailed runtime snapshot suitable for `/health/details`.
    pub fn health_details(&self) -> RuntimeHealthDetails {
        let status = self.runtime_status();
        let loaded_models = self.loaded_models();
        let metrics = self.metrics_snapshot();
        let loaded_at_unix_ms = self
            .runtime
            .loaded_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let uptime_ms = SystemTime::now()
            .duration_since(self.runtime.loaded_at)
            .unwrap_or(Duration::ZERO)
            .as_millis();

        RuntimeHealthDetails {
            status,
            backend: self.runtime.backend.clone(),
            transport: transport_info(&self.runtime.transport),
            loaded_models,
            metrics,
            loaded_at_unix_ms,
            uptime_ms,
            last_error: self
                .runtime
                .last_error
                .lock()
                .ok()
                .and_then(|guard| (*guard).clone()),
        }
    }

    /// Return the loaded model inventory.
    pub fn loaded_models(&self) -> LoadedModelInventory {
        self.tts
            .as_ref()
            .map(|tts| tts.model_inventory())
            .unwrap_or_default()
    }

    /// Return a consistent metrics snapshot even when TTS is not loaded.
    pub fn metrics_snapshot(&self) -> RuntimeMetricsSnapshot {
        self.tts
            .as_ref()
            .map(|tts| tts.metrics())
            .unwrap_or(RuntimeMetricsSnapshot {
                queue_depth: 0,
                queue_capacity: 0,
                active_requests: 0,
                total_requests: 0,
                completed_requests: 0,
                failed_requests: 0,
                rejected_requests: 0,
                worker_failures: 0,
            })
    }

    /// Read the current runtime status.
    pub fn runtime_status(&self) -> RuntimeStatus {
        self.runtime
            .status
            .lock()
            .map(|guard| *guard)
            .unwrap_or(RuntimeStatus::Error)
    }
}

/// Build the serializable transport payload from the runtime transport string.
fn transport_info(transport: &str) -> TransportInfo {
    if let Some(socket_path) = transport.strip_prefix("unix:") {
        return TransportInfo {
            kind: "unix".to_string(),
            host: None,
            port: None,
            socket_path: Some(socket_path.to_string()),
        };
    }

    if let Some(address) = transport.strip_prefix("tcp:") {
        if let Some((host, port)) = address.rsplit_once(':') {
            return TransportInfo {
                kind: "tcp".to_string(),
                host: Some(host.to_string()),
                port: port.parse().ok(),
                socket_path: None,
            };
        }
    }

    TransportInfo {
        kind: "unknown".to_string(),
        host: None,
        port: None,
        socket_path: None,
    }
}
