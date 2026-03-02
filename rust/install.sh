#!/bin/bash
# Installer for qwen3-audio-api — downloads binary, models, and tokenizers
# Usage: curl -sSf https://raw.githubusercontent.com/second-state/qwen3_audio_api/main/rust/install.sh | bash

set -e

REPO="second-state/qwen3_audio_api"
INSTALL_DIR="./qwen3_audio_api"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Detect platform
# ---------------------------------------------------------------------------
detect_platform() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="darwin" ;;
        *)
            err "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)    ARCH="x86_64" ;;
        aarch64|arm64)   ARCH="aarch64" ;;
        *)
            err "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac

    # CUDA detection (Linux x86_64 only)
    USE_CUDA=""
    if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ]; then
        if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null; then
            CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            info "NVIDIA GPU detected (driver ${CUDA_DRIVER})."
            echo "  1) CUDA (Recommended)"
            echo "  2) CPU only"
            printf "Select variant [1]: "
            read -r variant </dev/tty
            variant="${variant:-1}"
            if [ "$variant" = "1" ]; then
                USE_CUDA="1"
            fi
        fi
    fi

    info "Platform: ${OS} ${ARCH}${USE_CUDA:+ (CUDA)}"
}

# ---------------------------------------------------------------------------
# 2. Resolve release asset name
# ---------------------------------------------------------------------------
resolve_asset() {
    case "${OS}-${ARCH}" in
        darwin-aarch64)  ASSET_NAME="qwen3-audio-api-macos-arm64" ;;
        linux-x86_64)
            if [ -n "$USE_CUDA" ]; then
                ASSET_NAME="qwen3-audio-api-linux-x86_64-cuda"
            else
                ASSET_NAME="qwen3-audio-api-linux-x86_64"
            fi
            ;;
        linux-aarch64)   ASSET_NAME="qwen3-audio-api-linux-aarch64" ;;
        *)
            err "Unsupported platform: ${OS}-${ARCH}"
            exit 1
            ;;
    esac
    info "Release asset: ${ASSET_NAME}"
}

# ---------------------------------------------------------------------------
# 3. Download & extract release
# ---------------------------------------------------------------------------
download_release() {
    local tarball="${ASSET_NAME}.tar.gz"
    local url="https://github.com/${REPO}/releases/latest/download/${tarball}"

    info "Downloading release..."
    mkdir -p "${INSTALL_DIR}"

    local temp_dir
    temp_dir=$(mktemp -d)

    curl -fSL -o "${temp_dir}/${tarball}" "$url"
    info "Extracting release..."
    tar -xzf "${temp_dir}/${tarball}" -C "${temp_dir}"

    # Copy everything from the extracted asset directory into INSTALL_DIR
    cp -r "${temp_dir}/${ASSET_NAME}/"* "${INSTALL_DIR}/"
    chmod +x "${INSTALL_DIR}/qwen3-audio-api"

    rm -rf "$temp_dir"
    ok "Binary installed to ${INSTALL_DIR}/"
}

# ---------------------------------------------------------------------------
# 4. Download CUDA libtorch (Linux x86_64 CUDA only)
# ---------------------------------------------------------------------------
download_cuda_libtorch() {
    if [ -z "$USE_CUDA" ]; then
        return
    fi

    info "Downloading CUDA libtorch (this may take a while)..."
    local url="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu124.zip"
    local temp_dir
    temp_dir=$(mktemp -d)

    curl -fSL -o "${temp_dir}/libtorch.zip" "$url"
    info "Extracting libtorch..."
    unzip -q "${temp_dir}/libtorch.zip" -d "${temp_dir}"

    rm -rf "${INSTALL_DIR}/libtorch"
    mv "${temp_dir}/libtorch" "${INSTALL_DIR}/libtorch"

    rm -rf "$temp_dir"
    ok "CUDA libtorch installed to ${INSTALL_DIR}/libtorch/"
}

# ---------------------------------------------------------------------------
# 5. Model selection & download
# ---------------------------------------------------------------------------
download_models() {
    echo ""
    info "Select model size:"
    echo "  1) 0.6B (Recommended — smaller, faster)"
    echo "  2) 1.7B (Higher quality, needs more RAM/VRAM)"
    printf "Select model [1]: "
    read -r model_choice </dev/tty
    model_choice="${model_choice:-1}"

    if [ "$model_choice" = "2" ]; then
        MODEL_SIZE="1.7B"
    else
        MODEL_SIZE="0.6B"
    fi

    ASR_MODEL="Qwen3-ASR-${MODEL_SIZE}"
    TTS_BASE_MODEL="Qwen3-TTS-12Hz-${MODEL_SIZE}-Base"
    TTS_CV_MODEL="Qwen3-TTS-12Hz-${MODEL_SIZE}-CustomVoice"

    info "Selected ${MODEL_SIZE} models: ${ASR_MODEL}, ${TTS_BASE_MODEL}, ${TTS_CV_MODEL}"

    local models_dir="${INSTALL_DIR}/models"
    mkdir -p "$models_dir"

    for model in "$ASR_MODEL" "$TTS_BASE_MODEL" "$TTS_CV_MODEL"; do
        local model_dir="${models_dir}/${model}"
        if [ -d "$model_dir" ] && [ -f "$model_dir/model.safetensors" ]; then
            ok "${model} already downloaded, skipping."
        else
            info "Downloading ${model}..."
            mkdir -p "$model_dir"

            # List files via HuggingFace API and download each one
            local api_url="https://huggingface.co/api/models/Qwen/${model}"
            local hf_url="https://huggingface.co/Qwen/${model}/resolve/main"
            local files
            files=$(curl -fSL "$api_url" | grep -o '"rfilename":"[^"]*"' | sed 's/"rfilename":"//;s/"//')

            for file in $files; do
                # Skip metadata and documentation files
                case "$file" in
                    .gitattributes|README.md) continue ;;
                esac
                info "  ${file}..."
                mkdir -p "${model_dir}/$(dirname "$file")"
                curl -fSL -o "${model_dir}/${file}" "${hf_url}/${file}"
            done
            ok "${model} downloaded."
        fi
    done
}

# ---------------------------------------------------------------------------
# 6. Download tokenizers from release assets
# ---------------------------------------------------------------------------
download_tokenizers() {
    info "Downloading tokenizers..."
    for model in "$ASR_MODEL" "$TTS_BASE_MODEL" "$TTS_CV_MODEL"; do
        local tokenizer_url="https://github.com/${REPO}/releases/latest/download/tokenizer-${model}.json"
        local model_dir="${INSTALL_DIR}/models/${model}"
        info "  tokenizer for ${model}..."
        curl -fSL -o "${model_dir}/tokenizer.json" "$tokenizer_url"
    done
    ok "Tokenizers installed."
}

# ---------------------------------------------------------------------------
# 7. Done — print sample commands
# ---------------------------------------------------------------------------
print_usage() {
    local cv_path="models/${TTS_CV_MODEL}"
    local base_path="models/${TTS_BASE_MODEL}"
    local asr_path="models/${ASR_MODEL}"

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN} Installation complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""

    echo "Start the server:"
    echo ""
    echo "  cd ${INSTALL_DIR}"
    echo "  TTS_CUSTOMVOICE_MODEL_PATH=./${cv_path} \\"
    echo "    TTS_BASE_MODEL_PATH=./${base_path} \\"
    echo "    ASR_MODEL_PATH=./${asr_path} \\"
    echo "    ./qwen3-audio-api"
    echo ""

    echo "Text-to-Speech (after server starts):"
    echo ""
    echo "  curl -X POST http://localhost:8000/v1/audio/speech \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\": \"qwen3-tts\", \"input\": \"Hello world!\", \"voice\": \"alloy\"}' \\"
    echo "    --output speech.mp3"
    echo ""

    echo "Speech-to-Text:"
    echo ""
    echo "  curl -X POST http://localhost:8000/v1/audio/transcriptions \\"
    echo "    -F file=@audio.wav -F model=qwen3-asr"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    info "Qwen3 Audio API Installer"
    echo ""

    detect_platform
    resolve_asset
    download_release
    download_cuda_libtorch
    download_models
    download_tokenizers
    print_usage
}

main "$@"
