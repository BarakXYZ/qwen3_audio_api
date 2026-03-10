use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use qwen3_tts::tensor::Device as TtsDevice;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

use crate::error::ApiError;

/// Stable runtime state values surfaced to health/details consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Starting,
    ResidentHot,
    Degraded,
    Error,
}

/// Enumerates the concrete synthesis path selected for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsRoute {
    CustomVoice,
    CustomVoiceInstruction,
    BaseXVectorClone,
    BaseIclClone,
}

/// Public model inventory summary for health/details and `/v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct LoadedModelInventory {
    pub custom_voice_model_id: Option<String>,
    pub instruction_custom_voice_model_id: Option<String>,
    pub base_model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub voice_design_supported: bool,
}

/// Current queue and request counters exposed for observability.
#[derive(Debug, Serialize)]
pub struct RuntimeMetricsSnapshot {
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub active_requests: usize,
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub rejected_requests: u64,
    pub worker_failures: u64,
}

/// Aggregate runtime metrics stored behind atomics.
#[derive(Debug)]
pub struct RuntimeMetrics {
    queue_depth: AtomicUsize,
    active_requests: AtomicUsize,
    total_requests: AtomicU64,
    completed_requests: AtomicU64,
    failed_requests: AtomicU64,
    rejected_requests: AtomicU64,
    worker_failures: AtomicU64,
    queue_capacity: usize,
}

impl RuntimeMetrics {
    /// Construct a fresh metrics collector for a bounded queue.
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            queue_depth: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            completed_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            worker_failures: AtomicU64::new(0),
            queue_capacity,
        }
    }

    /// Snapshot the current counters for JSON responses.
    pub fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            queue_capacity: self.queue_capacity,
            active_requests: self.active_requests.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            worker_failures: self.worker_failures.load(Ordering::Relaxed),
        }
    }
}

/// In-memory TTS models owned by the dedicated worker thread.
struct TtsModels {
    custom_voice: Option<TTSInference>,
    instruction_custom_voice: Option<TTSInference>,
    base_model: Option<TTSInference>,
    speaker_encoder: Option<SpeakerEncoder>,
    audio_encoder: Option<AudioEncoder>,
}

/// Normalized synthesis request sent to the worker thread.
pub struct TtsRequest {
    pub input: String,
    pub voice: Option<String>,
    pub language: String,
    pub instructions: Option<String>,
    pub audio_sample: Option<Vec<u8>>,
    pub audio_sample_text: Option<String>,
}

/// Audio generation result plus timing metadata used by the HTTP layer.
pub struct TtsResponse {
    pub waveform: Vec<f32>,
    pub sample_rate: u32,
    pub route: TtsRoute,
    pub queue_wait: Duration,
    pub processing_time: Duration,
}

struct TtsJob {
    request: TtsRequest,
    enqueued_at: Instant,
    response_tx: oneshot::Sender<Result<TtsResponse, ApiError>>,
}

/// Handle used by request handlers to submit work to the dedicated TTS worker.
#[derive(Clone)]
pub struct TtsRuntimeHandle {
    sender: mpsc::Sender<TtsJob>,
    metrics: Arc<RuntimeMetrics>,
    model_inventory: LoadedModelInventory,
}

impl TtsRuntimeHandle {
    /// Return the loaded model inventory.
    pub fn model_inventory(&self) -> &LoadedModelInventory {
        &self.model_inventory
    }

    /// Return a snapshot of current queue metrics.
    pub fn metrics(&self) -> RuntimeMetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Queue a synthesis request on the single active worker.
    pub async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, ApiError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);

        let send_result = self.sender.try_send(TtsJob {
            request,
            enqueued_at: Instant::now(),
            response_tx,
        });

        if let Err(error) = send_result {
            self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
            self.metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);

            let message = match error {
                mpsc::error::TrySendError::Full(_) => {
                    "TTS runtime queue is full. Retry after outstanding requests finish."
                }
                mpsc::error::TrySendError::Closed(_) => "TTS runtime worker is unavailable.",
            };

            return Err(ApiError::service_unavailable(message));
        }

        response_rx.await.map_err(|_| {
            self.metrics.worker_failures.fetch_add(1, Ordering::Relaxed);
            ApiError::service_unavailable("TTS runtime worker terminated unexpectedly.")
        })?
    }
}

/// Shared runtime metadata surfaced by the HTTP handlers.
#[derive(Clone)]
pub struct RuntimeMetadata {
    pub backend: String,
    pub transport: String,
    pub loaded_at: SystemTime,
    pub status: Arc<Mutex<RuntimeStatus>>,
    pub last_error: Arc<Mutex<Option<String>>>,
}

/// Build the resident TTS worker and return a handle if any TTS model is configured.
pub fn build_tts_runtime(
    config: &crate::config::ServerConfig,
    tts_device: TtsDevice,
    backend: &str,
    transport: &str,
) -> Result<Option<(TtsRuntimeHandle, RuntimeMetadata)>, ApiError> {
    let has_any_tts_model = config.tts_customvoice_model_path.is_some()
        || config.tts_instruction_model_path.is_some()
        || config.tts_base_model_path.is_some();

    if !has_any_tts_model {
        return Ok(None);
    }

    let status = Arc::new(Mutex::new(RuntimeStatus::Starting));
    let last_error = Arc::new(Mutex::new(None));

    let models = load_models(config, tts_device)?;
    let model_inventory = LoadedModelInventory {
        custom_voice_model_id: config
            .tts_customvoice_model_path
            .as_ref()
            .map(|_| "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string()),
        instruction_custom_voice_model_id: config
            .tts_instruction_model_path
            .as_ref()
            .map(|_| "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string()),
        base_model_id: config
            .tts_base_model_path
            .as_ref()
            .map(|_| "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string()),
        asr_model_id: None,
        voice_design_supported: false,
    };

    let queue_capacity = config.queue_capacity;
    let metrics = Arc::new(RuntimeMetrics::new(queue_capacity));
    let (sender, receiver) = mpsc::channel(queue_capacity);

    let worker_metrics = Arc::clone(&metrics);
    let worker_status = Arc::clone(&status);
    let worker_last_error = Arc::clone(&last_error);

    thread::Builder::new()
        .name("qwen3-audio-api-tts".to_string())
        .spawn(move || {
            run_tts_worker(
                models,
                receiver,
                worker_metrics,
                worker_status,
                worker_last_error,
            );
        })
        .map_err(|error| ApiError::internal(format!("Failed to spawn TTS worker: {error}")))?;

    *status
        .lock()
        .map_err(|_| ApiError::internal("Runtime status lock poisoned"))? =
        RuntimeStatus::ResidentHot;

    Ok(Some((
        TtsRuntimeHandle {
            sender,
            metrics,
            model_inventory,
        },
        RuntimeMetadata {
            backend: backend.to_string(),
            transport: transport.to_string(),
            loaded_at: SystemTime::now(),
            status,
            last_error,
        },
    )))
}

/// Load all configured resident TTS models before the server begins serving traffic.
fn load_models(
    config: &crate::config::ServerConfig,
    tts_device: TtsDevice,
) -> Result<TtsModels, ApiError> {
    let mut models = TtsModels {
        custom_voice: None,
        instruction_custom_voice: None,
        base_model: None,
        speaker_encoder: None,
        audio_encoder: None,
    };

    if let Some(ref path) = config.tts_customvoice_model_path {
        tracing::info!("Loading CustomVoice TTS model from {path}");
        models.custom_voice = Some(TTSInference::new(Path::new(path), tts_device).map_err(
            |error| ApiError::internal(format!("Failed to load custom voice model: {error}")),
        )?);
        tracing::info!("CustomVoice TTS model loaded successfully");
    }

    if let Some(ref path) = config.tts_instruction_model_path {
        tracing::info!("Loading instruction CustomVoice TTS model from {path}");
        models.instruction_custom_voice = Some(
            TTSInference::new(Path::new(path), tts_device).map_err(|error| {
                ApiError::internal(format!(
                    "Failed to load instruction custom voice model: {error}"
                ))
            })?,
        );
        tracing::info!("Instruction CustomVoice TTS model loaded successfully");
    }

    if let Some(ref path) = config.tts_base_model_path {
        tracing::info!("Loading Base TTS model from {path}");
        let inference = TTSInference::new(Path::new(path), tts_device)
            .map_err(|error| ApiError::internal(format!("Failed to load base model: {error}")))?;

        let se_config = inference.config().speaker_encoder_config.clone();
        let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, tts_device)
            .map_err(|error| {
                ApiError::internal(format!("Failed to load speaker encoder: {error}"))
            })?;
        tracing::info!("Speaker encoder loaded");

        let speech_tokenizer_path = Path::new(path)
            .join("speech_tokenizer")
            .join("model.safetensors");
        if speech_tokenizer_path.exists() {
            models.audio_encoder = Some(
                AudioEncoder::load(&speech_tokenizer_path, tts_device).map_err(|error| {
                    ApiError::internal(format!("Failed to load audio encoder: {error}"))
                })?,
            );
            tracing::info!("Audio encoder loaded for ICL mode");
        } else {
            tracing::warn!(
                "speech_tokenizer not found at {}; ICL voice cloning will not be available",
                speech_tokenizer_path.display()
            );
        }

        models.speaker_encoder = Some(speaker_encoder);
        models.base_model = Some(inference);
        tracing::info!("Base TTS model loaded successfully");
    }

    Ok(models)
}

/// Execute queued TTS requests on a single dedicated worker thread.
fn run_tts_worker(
    models: TtsModels,
    mut receiver: mpsc::Receiver<TtsJob>,
    metrics: Arc<RuntimeMetrics>,
    status: Arc<Mutex<RuntimeStatus>>,
    last_error: Arc<Mutex<Option<String>>>,
) {
    while let Some(job) = receiver.blocking_recv() {
        metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
        metrics.active_requests.fetch_add(1, Ordering::Relaxed);

        let queue_wait = job.enqueued_at.elapsed();
        let started_at = Instant::now();
        let result = handle_tts_request(&models, job.request).map(|response| TtsResponse {
            waveform: response.waveform,
            sample_rate: response.sample_rate,
            route: response.route,
            queue_wait,
            processing_time: started_at.elapsed(),
        });

        match &result {
            Ok(_) => {
                metrics.completed_requests.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                if error.status.is_server_error() {
                    if let Ok(mut last_error_guard) = last_error.lock() {
                        *last_error_guard = Some(error.message.clone());
                    }
                }
            }
        }

        metrics.active_requests.fetch_sub(1, Ordering::Relaxed);

        if job.response_tx.send(result).is_err() {
            tracing::warn!("Dropped TTS response because the requester disconnected");
        }
    }

    metrics.worker_failures.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut status_guard) = status.lock() {
        *status_guard = RuntimeStatus::Degraded;
    }
    if let Ok(mut last_error_guard) = last_error.lock() {
        *last_error_guard = Some("TTS worker channel closed unexpectedly.".to_string());
    }
}

struct GeneratedAudio {
    waveform: Vec<f32>,
    sample_rate: u32,
    route: TtsRoute,
}

/// Route and execute a synthesis request against the loaded resident models.
fn handle_tts_request(models: &TtsModels, request: TtsRequest) -> Result<GeneratedAudio, ApiError> {
    if let Some(audio_sample) = request.audio_sample {
        let base_model = models.base_model.as_ref().ok_or_else(|| {
            ApiError::bad_request(
                "audio_sample requires a base model. Set TTS_BASE_MODEL_PATH to enable voice cloning.",
            )
        })?;
        let speaker_encoder = models
            .speaker_encoder
            .as_ref()
            .ok_or_else(|| ApiError::internal("Speaker encoder not loaded"))?;

        let (ref_samples, ref_sr) = qwen3_tts::audio::load_wav_bytes(&audio_sample)
            .map_err(|error| ApiError::bad_request(format!("Invalid audio_sample: {error}")))?;

        let ref_samples_24k = if ref_sr != 24000 {
            qwen3_tts::audio::resample(&ref_samples, ref_sr, 24000).map_err(|error| {
                ApiError::internal(format!("Failed to resample reference audio: {error}"))
            })?
        } else {
            ref_samples
        };

        let speaker_embedding = speaker_encoder
            .extract_embedding(&ref_samples_24k)
            .map_err(|error| {
                ApiError::internal(format!("Failed to extract speaker embedding: {error}"))
            })?;

        if let Some(reference_text) = request.audio_sample_text {
            let audio_encoder = models.audio_encoder.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "audio_sample_text requires speech_tokenizer/model.safetensors in the base model directory.",
                )
            })?;
            let ref_codes = audio_encoder.encode(&ref_samples_24k).map_err(|error| {
                ApiError::internal(format!("Failed to encode reference audio: {error}"))
            })?;

            let (waveform, sample_rate) = base_model
                .generate_with_icl(
                    &request.input,
                    &reference_text,
                    &ref_codes,
                    &speaker_embedding,
                    &request.language,
                    0.9,
                    50,
                    2048,
                )
                .map_err(|error| ApiError::internal(error.to_string()))?;

            return Ok(GeneratedAudio {
                waveform,
                sample_rate,
                route: TtsRoute::BaseIclClone,
            });
        }

        let (waveform, sample_rate) = base_model
            .generate_with_xvector(
                &request.input,
                &speaker_embedding,
                &request.language,
                0.9,
                50,
                2048,
            )
            .map_err(|error| ApiError::internal(error.to_string()))?;

        return Ok(GeneratedAudio {
            waveform,
            sample_rate,
            route: TtsRoute::BaseXVectorClone,
        });
    }

    let voice = request.voice.as_deref().ok_or_else(|| {
        ApiError::bad_request(
            "voice is required when audio_sample is not provided. VoiceDesign is not available in the resident Rust runtime yet.",
        )
    })?;
    let speaker = crate::config::resolve_voice(voice).map_err(ApiError::bad_request)?;
    let instructions = request.instructions.unwrap_or_default();

    let model = if !instructions.trim().is_empty() {
        models
            .instruction_custom_voice
            .as_ref()
            .or(models.custom_voice.as_ref())
            .ok_or_else(|| {
                ApiError::bad_request(
                    "A CustomVoice model is required for preset speaker synthesis.",
                )
            })?
    } else {
        models.custom_voice.as_ref().ok_or_else(|| {
            ApiError::bad_request("A CustomVoice model is required for preset speaker synthesis.")
        })?
    };

    let (waveform, sample_rate) = model
        .generate_with_instruct(
            &request.input,
            &speaker,
            &request.language,
            &instructions,
            0.9,
            50,
            2048,
        )
        .map_err(|error| ApiError::internal(error.to_string()))?;

    Ok(GeneratedAudio {
        waveform,
        sample_rate,
        route: if instructions.trim().is_empty() {
            TtsRoute::CustomVoice
        } else {
            TtsRoute::CustomVoiceInstruction
        },
    })
}
