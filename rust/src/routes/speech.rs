use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, State};
use axum::http::{header, HeaderValue, Request};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::Deserialize;

use crate::audio::{apply_speed, encode_audio};
use crate::config::ResponseFormat;
use crate::error::ApiError;
use crate::runtime::{TtsRequest, TtsRoute};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: ResponseFormat,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_language")]
    pub language: String,
    pub instructions: Option<String>,
    /// Base64-encoded reference audio for voice cloning
    pub audio_sample: Option<String>,
    /// Transcript of reference audio (enables ICL mode)
    pub audio_sample_text: Option<String>,
}

fn default_speed() -> f32 {
    1.0
}

fn default_language() -> String {
    "Auto".to_string()
}

/// Unified parameters parsed from either JSON or multipart.
struct SpeechParams {
    input: String,
    voice: Option<String>,
    response_format: ResponseFormat,
    speed: f32,
    language: String,
    instructions: Option<String>,
    audio_sample: Option<AudioSampleData>,
    audio_sample_text: Option<String>,
}

enum AudioSampleData {
    /// Base64-encoded audio string (from JSON body)
    Base64(String),
    /// Raw audio bytes (from multipart file upload)
    Bytes(Vec<u8>),
}

// ---------------------------------------------------------------------------
// Handler: inspects Content-Type and dispatches
// ---------------------------------------------------------------------------

/// Handle an OpenAI-compatible speech request and dispatch it to the resident worker.
pub async fn speech_handler(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
) -> Result<Response, ApiError> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.contains("multipart/form-data") {
        let multipart = Multipart::from_request(request, &()).await.map_err(|e| {
            ApiError::unprocessable(format!("Failed to parse multipart request: {e}"))
        })?;
        speech_multipart(state, multipart).await
    } else {
        let bytes = Bytes::from_request(request, &()).await.map_err(
            |e: axum::extract::rejection::BytesRejection| {
                ApiError::unprocessable(format!("Failed to read request body: {e}"))
            },
        )?;
        let req: SpeechRequest = serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::unprocessable(format!("Invalid JSON: {e}")))?;
        speech_json(state, req).await
    }
}

async fn speech_json(state: AppState, req: SpeechRequest) -> Result<Response, ApiError> {
    validate_input(&req.input, req.speed)?;
    validate_semantics(
        req.voice.as_deref(),
        req.instructions.as_deref(),
        req.audio_sample.is_some(),
    )?;

    let params = SpeechParams {
        input: req.input,
        voice: req.voice,
        response_format: req.response_format,
        speed: req.speed,
        language: req.language,
        instructions: req.instructions,
        audio_sample: req.audio_sample.map(AudioSampleData::Base64),
        audio_sample_text: req.audio_sample_text,
    };

    generate_speech(state, params).await
}

async fn speech_multipart(state: AppState, mut multipart: Multipart) -> Result<Response, ApiError> {
    let mut input: Option<String> = None;
    let mut voice: Option<String> = None;
    let mut response_format = ResponseFormat::Mp3;
    let mut speed: f32 = 1.0;
    let mut language = "Auto".to_string();
    let mut instructions: Option<String> = None;
    let mut audio_sample: Option<AudioSampleData> = None;
    let mut audio_sample_text: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::unprocessable(format!("Multipart parse error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "input" => {
                input = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            "model" => {
                let _ = field.text().await;
            }
            "voice" => {
                let parsed_voice = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                let trimmed = parsed_voice.trim();
                if !trimmed.is_empty() {
                    voice = Some(trimmed.to_string());
                }
            }
            "response_format" => {
                let fmt_str = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                response_format =
                    serde_json::from_value(serde_json::Value::String(fmt_str.clone())).map_err(
                        |_| ApiError::unprocessable(format!("Invalid response_format: {fmt_str}")),
                    )?;
            }
            "speed" => {
                let s = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                speed = s
                    .parse()
                    .map_err(|_| ApiError::unprocessable("Invalid speed value"))?;
            }
            "language" => {
                language = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
            }
            "instructions" => {
                instructions = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            "audio_sample" => {
                let ct = field.content_type().map(|s| s.to_string());
                if ct.is_some() {
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                    audio_sample = Some(AudioSampleData::Bytes(bytes.to_vec()));
                } else {
                    let text = field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                    audio_sample = Some(AudioSampleData::Base64(text));
                }
            }
            "audio_sample_text" => {
                audio_sample_text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let input = input.ok_or_else(|| ApiError::unprocessable("'input' field is required"))?;
    validate_input(&input, speed)?;
    validate_semantics(
        voice.as_deref(),
        instructions.as_deref(),
        audio_sample.is_some(),
    )?;

    let params = SpeechParams {
        input,
        voice,
        response_format,
        speed,
        language,
        instructions,
        audio_sample,
        audio_sample_text,
    };

    generate_speech(state, params).await
}

// ---------------------------------------------------------------------------
// Core generation logic
// ---------------------------------------------------------------------------

/// Perform generation through the resident worker and encode the resulting waveform.
async fn generate_speech(state: AppState, params: SpeechParams) -> Result<Response, ApiError> {
    let runtime = state
        .tts
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("TTS models are not loaded on this runtime."))?;

    let audio_sample = decode_audio_sample(params.audio_sample)?;
    let synthesis = runtime
        .synthesize(TtsRequest {
            input: params.input,
            voice: params.voice,
            language: params.language,
            instructions: params.instructions,
            audio_sample,
            audio_sample_text: params.audio_sample_text,
        })
        .await?;

    let waveform = if (params.speed - 1.0).abs() > f32::EPSILON {
        apply_speed(&synthesis.waveform, params.speed)
    } else {
        synthesis.waveform
    };

    let audio_bytes = encode_audio(&waveform, synthesis.sample_rate, params.response_format)?;
    let mut response = (
        [(header::CONTENT_TYPE, params.response_format.content_type())],
        audio_bytes,
    )
        .into_response();

    response.headers_mut().insert(
        "x-qwen-route",
        HeaderValue::from_str(tts_route_header_value(synthesis.route))
            .unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    response.headers_mut().insert(
        "x-qwen-queue-wait-ms",
        HeaderValue::from_str(&synthesis.queue_wait.as_millis().to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );
    response.headers_mut().insert(
        "x-qwen-processing-ms",
        HeaderValue::from_str(&synthesis.processing_time.as_millis().to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0")),
    );

    Ok(response)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate request size and playback parameters before queueing work.
fn validate_input(input: &str, speed: f32) -> Result<(), ApiError> {
    if input.is_empty() {
        return Err(ApiError::unprocessable("'input' field is required"));
    }
    if input.len() > 4096 {
        return Err(ApiError::unprocessable("'input' exceeds 4096 characters"));
    }
    if !(0.25..=4.0).contains(&speed) {
        return Err(ApiError::unprocessable(
            "'speed' must be between 0.25 and 4.0",
        ));
    }
    Ok(())
}

/// Reject semantically invalid combinations rather than silently falling back.
fn validate_semantics(
    voice: Option<&str>,
    instructions: Option<&str>,
    has_audio_sample: bool,
) -> Result<(), ApiError> {
    if has_audio_sample {
        return Ok(());
    }

    let has_voice = voice.map(|value| !value.trim().is_empty()).unwrap_or(false);
    let has_instructions = instructions
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);

    if has_instructions && !has_voice {
        return Err(ApiError::bad_request(
            "instructions without voice imply VoiceDesign, which is not implemented in the resident Rust runtime yet.",
        ));
    }

    if !has_voice {
        return Err(ApiError::bad_request(
            "voice is required when audio_sample is not provided.",
        ));
    }

    Ok(())
}

/// Decode the optional JSON/multipart audio sample into raw bytes.
fn decode_audio_sample(audio_sample: Option<AudioSampleData>) -> Result<Option<Vec<u8>>, ApiError> {
    let Some(audio_sample) = audio_sample else {
        return Ok(None);
    };

    let decoded = match audio_sample {
        AudioSampleData::Base64(encoded) => base64::engine::general_purpose::STANDARD
            .decode(encoded.trim())
            .map_err(|error| ApiError::bad_request(format!("Invalid base64: {error}")))?,
        AudioSampleData::Bytes(bytes) => bytes,
    };

    Ok(Some(decoded))
}

/// Map route enums to stable HTTP header values.
fn tts_route_header_value(route: TtsRoute) -> &'static str {
    match route {
        TtsRoute::CustomVoice => "custom_voice",
        TtsRoute::CustomVoiceInstruction => "custom_voice_instruction",
        TtsRoute::BaseXVectorClone => "base_xvector_clone",
        TtsRoute::BaseIclClone => "base_icl_clone",
    }
}

#[cfg(test)]
mod tests {
    use super::{tts_route_header_value, validate_semantics};
    use crate::runtime::TtsRoute;

    #[test]
    fn rejects_style_only_requests_until_voice_design_is_supported() {
        let error = validate_semantics(None, Some("warm and cinematic"), false).unwrap_err();
        assert!(error.message.contains("VoiceDesign"));
    }

    #[test]
    fn allows_reference_audio_without_voice() {
        validate_semantics(None, Some("ignored"), true).unwrap();
    }

    #[test]
    fn encodes_route_headers_with_stable_values() {
        assert_eq!(
            tts_route_header_value(TtsRoute::CustomVoice),
            "custom_voice"
        );
        assert_eq!(
            tts_route_header_value(TtsRoute::BaseIclClone),
            "base_icl_clone"
        );
    }
}
