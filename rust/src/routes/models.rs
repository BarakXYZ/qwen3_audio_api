use axum::extract::State;
use axum::Json;
use serde::Serialize;

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
