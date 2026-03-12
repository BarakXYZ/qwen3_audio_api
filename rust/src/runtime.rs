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
    VoiceDesign,
    BaseXVectorClone,
    BaseIclClone,
}

/// Public model inventory summary for health/details and `/v1/models`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadedModelInventory {
    pub custom_voice_model_id: Option<String>,
    pub instruction_custom_voice_model_id: Option<String>,
    pub voice_design_model_id: Option<String>,
    pub base_model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub voice_design_supported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagedModelId {
    CustomVoice,
    InstructionCustomVoice,
    VoiceDesign,
    BaseVoiceClone,
}

impl ManagedModelId {
    pub fn from_route_id(value: &str) -> Option<Self> {
        match value {
            "custom_voice" => Some(Self::CustomVoice),
            "instruction_custom_voice" => Some(Self::InstructionCustomVoice),
            "voice_design" => Some(Self::VoiceDesign),
            "base_voice_clone" => Some(Self::BaseVoiceClone),
            _ => None,
        }
    }

    fn inventory_id(self) -> &'static str {
        match self {
            Self::CustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::InstructionCustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::VoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::BaseVoiceClone => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }
    }
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
    voice_design: Option<TTSInference>,
    base_model: Option<TTSInference>,
    speaker_encoder: Option<SpeakerEncoder>,
    audio_encoder: Option<AudioEncoder>,
}

#[derive(Debug, Clone, Default)]
struct AvailableModelPaths {
    custom_voice: Option<String>,
    instruction_custom_voice: Option<String>,
    voice_design: Option<String>,
    base_voice_clone: Option<String>,
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

enum RuntimeCommand {
    Synthesize(TtsJob),
    LoadModel {
        model_id: ManagedModelId,
        response_tx: oneshot::Sender<Result<LoadedModelInventory, ApiError>>,
    },
    OffloadModel {
        model_id: ManagedModelId,
        response_tx: oneshot::Sender<Result<LoadedModelInventory, ApiError>>,
    },
}

/// Handle used by request handlers to submit work to the dedicated TTS worker.
#[derive(Clone)]
pub struct TtsRuntimeHandle {
    sender: mpsc::Sender<RuntimeCommand>,
    metrics: Arc<RuntimeMetrics>,
    model_inventory: Arc<Mutex<LoadedModelInventory>>,
}

impl TtsRuntimeHandle {
    /// Return the loaded model inventory.
    pub fn model_inventory(&self) -> LoadedModelInventory {
        self.model_inventory
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
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

        let send_result = self.sender.try_send(RuntimeCommand::Synthesize(TtsJob {
            request,
            enqueued_at: Instant::now(),
            response_tx,
        }));

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

    pub async fn load_model(
        &self,
        model_id: ManagedModelId,
    ) -> Result<LoadedModelInventory, ApiError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(RuntimeCommand::LoadModel {
                model_id,
                response_tx,
            })
            .await
            .map_err(|_| ApiError::service_unavailable("TTS runtime worker is unavailable."))?;

        response_rx.await.map_err(|_| {
            self.metrics.worker_failures.fetch_add(1, Ordering::Relaxed);
            ApiError::service_unavailable("TTS runtime worker terminated unexpectedly.")
        })?
    }

    pub async fn offload_model(
        &self,
        model_id: ManagedModelId,
    ) -> Result<LoadedModelInventory, ApiError> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(RuntimeCommand::OffloadModel {
                model_id,
                response_tx,
            })
            .await
            .map_err(|_| ApiError::service_unavailable("TTS runtime worker is unavailable."))?;

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
        || config.tts_voice_design_model_path.is_some()
        || config.tts_base_model_path.is_some();

    if !has_any_tts_model {
        return Ok(None);
    }

    let status = Arc::new(Mutex::new(RuntimeStatus::Starting));
    let last_error = Arc::new(Mutex::new(None));

    let available_model_paths = AvailableModelPaths {
        custom_voice: config.tts_customvoice_model_path.clone(),
        instruction_custom_voice: config.tts_instruction_model_path.clone(),
        voice_design: config.tts_voice_design_model_path.clone(),
        base_voice_clone: config.tts_base_model_path.clone(),
    };
    let mut models = load_models(config, tts_device)?;
    let model_inventory = Arc::new(Mutex::new(build_loaded_model_inventory(
        &available_model_paths,
        &models,
    )));

    let queue_capacity = config.queue_capacity;
    let metrics = Arc::new(RuntimeMetrics::new(queue_capacity));
    let (sender, receiver) = mpsc::channel(queue_capacity);

    let worker_metrics = Arc::clone(&metrics);
    let worker_status = Arc::clone(&status);
    let worker_last_error = Arc::clone(&last_error);
    let worker_inventory = Arc::clone(&model_inventory);

    thread::Builder::new()
        .name("qwen3-audio-api-tts".to_string())
        .spawn(move || {
            run_tts_worker(
                &mut models,
                available_model_paths,
                tts_device,
                receiver,
                worker_metrics,
                worker_status,
                worker_last_error,
                worker_inventory,
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
        voice_design: None,
        base_model: None,
        speaker_encoder: None,
        audio_encoder: None,
    };

    if config.tts_preload_model_ids.contains("custom_voice") {
        if let Some(ref path) = config.tts_customvoice_model_path {
            tracing::info!("Loading CustomVoice TTS model from {path}");
            models.custom_voice = Some(TTSInference::new(Path::new(path), tts_device).map_err(
                |error| ApiError::internal(format!("Failed to load custom voice model: {error}")),
            )?);
            tracing::info!("CustomVoice TTS model loaded successfully");
        }
    }

    if config
        .tts_preload_model_ids
        .contains("instruction_custom_voice")
    {
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
    }

    if config.tts_preload_model_ids.contains("voice_design") {
        if let Some(ref path) = config.tts_voice_design_model_path {
            tracing::info!("Loading VoiceDesign TTS model from {path}");
            models.voice_design = Some(TTSInference::new(Path::new(path), tts_device).map_err(
                |error| ApiError::internal(format!("Failed to load voice design model: {error}")),
            )?);
            tracing::info!("VoiceDesign TTS model loaded successfully");
        }
    }

    if config.tts_preload_model_ids.contains("base_voice_clone") {
        if let Some(ref path) = config.tts_base_model_path {
            tracing::info!("Loading Base TTS model from {path}");
            let inference = TTSInference::new(Path::new(path), tts_device).map_err(|error| {
                ApiError::internal(format!("Failed to load base model: {error}"))
            })?;

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
    }

    Ok(models)
}

fn build_loaded_model_inventory(
    available_model_paths: &AvailableModelPaths,
    models: &TtsModels,
) -> LoadedModelInventory {
    LoadedModelInventory {
        custom_voice_model_id: models
            .custom_voice
            .as_ref()
            .map(|_| ManagedModelId::CustomVoice.inventory_id().to_string()),
        instruction_custom_voice_model_id: models.instruction_custom_voice.as_ref().map(|_| {
            ManagedModelId::InstructionCustomVoice
                .inventory_id()
                .to_string()
        }),
        voice_design_model_id: models
            .voice_design
            .as_ref()
            .map(|_| ManagedModelId::VoiceDesign.inventory_id().to_string()),
        base_model_id: models
            .base_model
            .as_ref()
            .map(|_| ManagedModelId::BaseVoiceClone.inventory_id().to_string()),
        asr_model_id: None,
        voice_design_supported: available_model_paths.voice_design.is_some(),
    }
}

fn load_model_into_runtime(
    models: &mut TtsModels,
    available_model_paths: &AvailableModelPaths,
    model_id: ManagedModelId,
    tts_device: TtsDevice,
) -> Result<(), ApiError> {
    match model_id {
        ManagedModelId::CustomVoice => {
            if models.custom_voice.is_some() {
                return Ok(());
            }
            let path = available_model_paths.custom_voice.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "CustomVoice is not configured on this runtime. Set TTS_CUSTOMVOICE_MODEL_PATH.",
                )
            })?;
            tracing::info!("Hot-loading CustomVoice TTS model from {path}");
            models.custom_voice = Some(TTSInference::new(Path::new(path), tts_device).map_err(
                |error| ApiError::internal(format!("Failed to load custom voice model: {error}")),
            )?);
        }
        ManagedModelId::InstructionCustomVoice => {
            if models.instruction_custom_voice.is_some() {
                return Ok(());
            }
            let path = available_model_paths
                .instruction_custom_voice
                .as_ref()
                .ok_or_else(|| {
                    ApiError::bad_request(
                        "Instruction CustomVoice is not configured on this runtime. Set TTS_INSTRUCTION_MODEL_PATH.",
                    )
                })?;
            tracing::info!("Hot-loading Instruction CustomVoice TTS model from {path}");
            models.instruction_custom_voice = Some(
                TTSInference::new(Path::new(path), tts_device).map_err(|error| {
                    ApiError::internal(format!(
                        "Failed to load instruction custom voice model: {error}"
                    ))
                })?,
            );
        }
        ManagedModelId::VoiceDesign => {
            if models.voice_design.is_some() {
                return Ok(());
            }
            let path = available_model_paths.voice_design.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "VoiceDesign is not configured on this runtime. Set TTS_VOICEDESIGN_MODEL_PATH.",
                )
            })?;
            tracing::info!("Hot-loading VoiceDesign TTS model from {path}");
            models.voice_design = Some(TTSInference::new(Path::new(path), tts_device).map_err(
                |error| ApiError::internal(format!("Failed to load voice design model: {error}")),
            )?);
        }
        ManagedModelId::BaseVoiceClone => {
            if models.base_model.is_some() {
                return Ok(());
            }
            let path = available_model_paths
                .base_voice_clone
                .as_ref()
                .ok_or_else(|| {
                    ApiError::bad_request(
                    "Base voice clone is not configured on this runtime. Set TTS_BASE_MODEL_PATH.",
                )
                })?;
            tracing::info!("Hot-loading Base TTS model from {path}");
            let inference = TTSInference::new(Path::new(path), tts_device).map_err(|error| {
                ApiError::internal(format!("Failed to load base model: {error}"))
            })?;

            let se_config = inference.config().speaker_encoder_config.clone();
            let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, tts_device)
                .map_err(|error| {
                    ApiError::internal(format!("Failed to load speaker encoder: {error}"))
                })?;

            let speech_tokenizer_path = Path::new(path)
                .join("speech_tokenizer")
                .join("model.safetensors");
            let audio_encoder = if speech_tokenizer_path.exists() {
                Some(
                    AudioEncoder::load(&speech_tokenizer_path, tts_device).map_err(|error| {
                        ApiError::internal(format!("Failed to load audio encoder: {error}"))
                    })?,
                )
            } else {
                tracing::warn!(
                    "speech_tokenizer not found at {}; ICL voice cloning will not be available",
                    speech_tokenizer_path.display()
                );
                None
            };

            models.speaker_encoder = Some(speaker_encoder);
            models.audio_encoder = audio_encoder;
            models.base_model = Some(inference);
        }
    }

    Ok(())
}

fn offload_model_from_runtime(
    models: &mut TtsModels,
    _available_model_paths: &AvailableModelPaths,
    model_id: ManagedModelId,
) -> Result<(), ApiError> {
    match model_id {
        ManagedModelId::CustomVoice => {
            models.custom_voice = None;
        }
        ManagedModelId::InstructionCustomVoice => {
            models.instruction_custom_voice = None;
        }
        ManagedModelId::VoiceDesign => {
            models.voice_design = None;
        }
        ManagedModelId::BaseVoiceClone => {
            models.base_model = None;
            models.speaker_encoder = None;
            models.audio_encoder = None;
        }
    }

    Ok(())
}

/// Execute queued TTS requests on a single dedicated worker thread.
fn run_tts_worker(
    models: &mut TtsModels,
    available_model_paths: AvailableModelPaths,
    tts_device: TtsDevice,
    mut receiver: mpsc::Receiver<RuntimeCommand>,
    metrics: Arc<RuntimeMetrics>,
    status: Arc<Mutex<RuntimeStatus>>,
    last_error: Arc<Mutex<Option<String>>>,
    model_inventory: Arc<Mutex<LoadedModelInventory>>,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            RuntimeCommand::Synthesize(job) => {
                metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
                metrics.active_requests.fetch_add(1, Ordering::Relaxed);

                let queue_wait = job.enqueued_at.elapsed();
                let started_at = Instant::now();
                let result = handle_tts_request(models, job.request).map(|response| TtsResponse {
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
            RuntimeCommand::LoadModel {
                model_id,
                response_tx,
            } => {
                if let Ok(mut status_guard) = status.lock() {
                    *status_guard = RuntimeStatus::Starting;
                }
                let result =
                    load_model_into_runtime(models, &available_model_paths, model_id, tts_device)
                        .map(|_| build_loaded_model_inventory(&available_model_paths, models));
                if let Ok(mut inventory_guard) = model_inventory.lock() {
                    *inventory_guard = build_loaded_model_inventory(&available_model_paths, models);
                }
                if let Ok(mut status_guard) = status.lock() {
                    *status_guard = if result.is_ok() {
                        RuntimeStatus::ResidentHot
                    } else {
                        RuntimeStatus::Degraded
                    };
                }
                if let Ok(mut last_error_guard) = last_error.lock() {
                    *last_error_guard = result.as_ref().err().map(|error| error.message.clone());
                }
                if let Err(error) = &result {
                    tracing::error!("Failed to hot-load model {:?}: {}", model_id, error.message);
                }
                let _ = response_tx.send(result);
            }
            RuntimeCommand::OffloadModel {
                model_id,
                response_tx,
            } => {
                let result = offload_model_from_runtime(models, &available_model_paths, model_id)
                    .map(|_| build_loaded_model_inventory(&available_model_paths, models));
                if let Ok(mut inventory_guard) = model_inventory.lock() {
                    *inventory_guard = build_loaded_model_inventory(&available_model_paths, models);
                }
                if let Ok(mut status_guard) = status.lock() {
                    *status_guard = RuntimeStatus::ResidentHot;
                }
                if let Ok(mut last_error_guard) = last_error.lock() {
                    *last_error_guard = result.as_ref().err().map(|error| error.message.clone());
                }
                if let Err(error) = &result {
                    tracing::error!("Failed to offload model {:?}: {}", model_id, error.message);
                }
                let _ = response_tx.send(result);
            }
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

    let instructions = request.instructions.unwrap_or_default();
    let voice = request.voice.as_deref();

    if voice.is_none() {
        if instructions.trim().is_empty() {
            return Err(ApiError::bad_request(
                "voice is required when audio_sample is not provided unless VoiceDesign instructions are supplied.",
            ));
        }

        let model = models.voice_design.as_ref().ok_or_else(|| {
            ApiError::bad_request(
                "instructions without voice require the VoiceDesign model. Set TTS_VOICEDESIGN_MODEL_PATH to enable this route.",
            )
        })?;
        let (waveform, sample_rate) = model
            .generate_voice_design(
                &request.input,
                &request.language,
                &instructions,
                0.9,
                50,
                2048,
            )
            .map_err(|error| ApiError::internal(error.to_string()))?;

        return Ok(GeneratedAudio {
            waveform,
            sample_rate,
            route: TtsRoute::VoiceDesign,
        });
    }

    let speaker = crate::config::resolve_voice(voice.unwrap()).map_err(ApiError::bad_request)?;

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
