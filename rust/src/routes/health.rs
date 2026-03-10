use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;

use crate::runtime::RuntimeStatus;
use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    let details = state.health_details();
    let status_code = match details.status {
        RuntimeStatus::ResidentHot => StatusCode::OK,
        RuntimeStatus::Starting => StatusCode::ACCEPTED,
        RuntimeStatus::Degraded => StatusCode::SERVICE_UNAVAILABLE,
        RuntimeStatus::Error => StatusCode::INTERNAL_SERVER_ERROR,
    };

    (status_code, Json(details))
}

pub async fn health_details(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    let details = state.health_details();
    let status_code = match details.status {
        RuntimeStatus::ResidentHot => StatusCode::OK,
        RuntimeStatus::Starting => StatusCode::ACCEPTED,
        RuntimeStatus::Degraded => StatusCode::SERVICE_UNAVAILABLE,
        RuntimeStatus::Error => StatusCode::INTERNAL_SERVER_ERROR,
    };

    (status_code, Json(details))
}

pub async fn metrics(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    Json(state.metrics_snapshot())
}
