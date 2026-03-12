use axum::extract::{Path, State};
use axum::Json;
use serde::Serialize;

use crate::error::ApiError;
use crate::runtime::ManagedModelId;
use crate::state::AppState;

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

/// List the concrete model families currently resident in memory.
pub async fn list_models(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    let inventory = state.loaded_models();
    let mut data = Vec::new();

    if let Some(id) = inventory.custom_voice_model_id {
        data.push(ModelInfo {
            id,
            object: "model",
            owned_by: "qwen",
        });
    }

    if let Some(id) = inventory.instruction_custom_voice_model_id {
        data.push(ModelInfo {
            id,
            object: "model",
            owned_by: "qwen",
        });
    }

    if let Some(id) = inventory.voice_design_model_id {
        data.push(ModelInfo {
            id,
            object: "model",
            owned_by: "qwen",
        });
    }

    if let Some(id) = inventory.base_model_id {
        data.push(ModelInfo {
            id,
            object: "model",
            owned_by: "qwen",
        });
    }

    if let Some(id) = inventory.asr_model_id {
        data.push(ModelInfo {
            id,
            object: "model",
            owned_by: "qwen",
        });
    }

    Json(ModelsResponse {
        object: "list",
        data,
    })
}

pub async fn load_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<crate::runtime::LoadedModelInventory>, ApiError> {
    let runtime = state
        .tts
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("TTS runtime is not loaded on this server."))?;
    let model_id = ManagedModelId::from_route_id(&model_id)
        .ok_or_else(|| ApiError::bad_request("Unknown model id."))?;
    let inventory = runtime.load_model(model_id).await?;
    Ok(Json(inventory))
}

pub async fn offload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<crate::runtime::LoadedModelInventory>, ApiError> {
    let runtime = state
        .tts
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("TTS runtime is not loaded on this server."))?;
    let model_id = ManagedModelId::from_route_id(&model_id)
        .ok_or_else(|| ApiError::bad_request("Unknown model id."))?;
    let inventory = runtime.offload_model(model_id).await?;
    Ok(Json(inventory))
}
