#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "\nERROR: setup failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

# Ascend 950DT environment bootstrap for the combined bernini_pixelprune repo.
#
# Expected remote layout:
#   /home/<user>/bernini_pixelprune/
#     bernini/
#     pixelprune/
#     setup.sh
#
# This script intentionally does NOT install or upgrade the NPU driver/CANN,
# and it does NOT download models or datasets. The machine provider must first
# install a CANN release that supports Ascend 950 (CANN 9.0+).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}
BASE_ROOT=${BASE_ROOT:-$(dirname -- "$PROJECT_ROOT")}
BERNINI_DIR=${BERNINI_DIR:-${PROJECT_ROOT}/bernini}
PIXELPRUNE_DIR=${PIXELPRUNE_DIR:-${PROJECT_ROOT}/pixelprune}
CONDA_ROOT=${CONDA_ROOT:-${BASE_ROOT}/miniforge3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-bernini}
RAINFUSION_ENV=${RAINFUSION_ENV:-}
MINDIESD_DIR=${MINDIESD_DIR:-${BASE_ROOT}/MindIE-SD}
VEOMNI_DIR=${VEOMNI_DIR:-${BASE_ROOT}/VeOmni}
DECORD_DIR=${DECORD_DIR:-${BASE_ROOT}/decord}
MODEL_DIR=${MODEL_DIR:-${BASE_ROOT}/Bernini-Diffusers}
EDITVERSE_DATA_ROOT=${EDITVERSE_DATA_ROOT:-${BASE_ROOT}/datasets/EditVerse/EditVerseBench}
OPENVE_ROOT=${OPENVE_ROOT:-${BASE_ROOT}/datasets/OpenVE}

# Known-compatible software pins for this Bernini branch. Override them only
# when the installed CANN/TorchNPU compatibility table requires it.
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
TORCH_VERSION=${TORCH_VERSION:-2.8.0}
TORCH_NPU_VERSION=${TORCH_NPU_VERSION:-2.8.0.post4}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.23.0}
TORCHDATA_VERSION=${TORCHDATA_VERSION:-0.11.0}
MINDIESD_REF=${MINDIESD_REF:-ff8eb69e5f5e323210e07362bb0e16759b8d1cad}
VEOMNI_TAG=${VEOMNI_TAG:-v0.1.10}
VEOMNI_REF=${VEOMNI_REF:-6ab293ecdfdd90ef3941fc81065d9f947b5b4e4f}
DECORD_REF=${DECORD_REF:-d2e56190286ae394032a8141885f76d5372bd44b}
BUILD_JOBS=${BUILD_JOBS:-8}
VERIFY_DEVICE=${VERIFY_DEVICE:-0}

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash setup.sh
  bash setup.sh --verify-only

Optional overrides:
  PROJECT_ROOT=/home/<user>/bernini_pixelprune
  BASE_ROOT=/home/<user>
  MODEL_DIR=/home/<user>/Bernini-Diffusers
  CANN_ENV_SCRIPT=/usr/local/Ascend/cann/set_env.sh
  CONDA_BIN=/path/to/conda
  CONDA_ENV_NAME=bernini
  VERIFY_DEVICE=0
  BUILD_JOBS=8
  FORCE_REBUILD_MINDIESD=1
  ALLOW_UNVERIFIED_CANN=1

The default target is CANN 9.0.x + Python 3.11 + PyTorch 2.8.0 +
TorchNPU 2.8.0.post4 on Ascend 950DT.
EOF
}

MODE=install
case "${1:-}" in
    "") ;;
    --verify-only) MODE=verify ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
esac

[[ -d "$BERNINI_DIR/bernini" ]] || die "Bernini source not found: $BERNINI_DIR"
[[ -f "$BERNINI_DIR/infer_multi_gpu.py" ]] || die "Missing: $BERNINI_DIR/infer_multi_gpu.py"
[[ -d "$PIXELPRUNE_DIR/pixelprune" ]] || die "PixelPrune source not found: $PIXELPRUNE_DIR"

detect_cann_env() {
    if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
        [[ -f "$CANN_ENV_SCRIPT" ]] || die "CANN_ENV_SCRIPT does not exist: $CANN_ENV_SCRIPT"
        return
    fi

    local candidate
    for candidate in \
        /usr/local/Ascend/cann/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
        if [[ -f "$candidate" ]]; then
            CANN_ENV_SCRIPT=$candidate
            return
        fi
    done
    die "CANN set_env.sh not found. Ask the machine provider to install the Ascend 950 CANN Toolkit and Kernels."
}

detect_cann_version() {
    local version_file=""
    local candidate
    for candidate in \
        "${ASCEND_HOME_PATH:-}/compiler/version.info" \
        /usr/local/Ascend/cann/compiler/version.info \
        /usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info \
        /usr/local/Ascend/cann/x86_64-linux/ascend_toolkit_install.info \
        /usr/local/Ascend/ascend-toolkit/latest/compiler/version.info \
        /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info \
        /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            version_file=$candidate
            break
        fi
    done

    if [[ -z "$version_file" ]]; then
        if [[ "${ALLOW_UNVERIFIED_CANN:-0}" == 1 ]]; then
            log "CANN version file was not found; continuing because ALLOW_UNVERIFIED_CANN=1"
            return
        fi
        die "Could not determine the CANN version. Set ALLOW_UNVERIFIED_CANN=1 only after manually confirming CANN 9.0+."
    fi

    local version
    version=$(sed -nE 's/^[Vv]ersion[[:space:]]*=[[:space:]]*([^[:space:]]+).*/\1/p' "$version_file" | head -n 1)
    [[ -n "$version" ]] || die "Could not parse CANN version from $version_file"
    log "Detected CANN $version from $version_file"

    local major=${version%%.*}
    [[ "$major" =~ ^[0-9]+$ ]] || die "Invalid CANN version: $version"
    (( major >= 9 )) || die "Ascend 950 needs CANN 9.0+; detected CANN $version"
}

detect_cann_env
# shellcheck disable=SC1090
source "$CANN_ENV_SCRIPT"
detect_cann_version

if command -v npu-smi >/dev/null 2>&1; then
    log "NPU inventory"
    npu-smi info || die "npu-smi failed; fix the driver/firmware before creating the Python environment"
else
    die "npu-smi is missing; the Ascend driver is not ready"
fi

resolve_conda() {
    if [[ -n "${CONDA_BIN:-}" ]]; then
        [[ -x "$CONDA_BIN" ]] || die "CONDA_BIN is not executable: $CONDA_BIN"
        return
    fi
    if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
        CONDA_BIN=$CONDA_ROOT/bin/conda
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN=$(command -v conda)
        return
    fi

    local machine_arch installer_arch installer=/tmp/miniforge3-installer.sh
    machine_arch=$(uname -m)
    case "$machine_arch" in
        aarch64|arm64) installer_arch=aarch64 ;;
        x86_64|amd64) installer_arch=x86_64 ;;
        *) die "Unsupported host architecture for Miniforge: $machine_arch" ;;
    esac

    log "Installing Miniforge into $CONDA_ROOT"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${installer_arch}.sh" \
            -o "$installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$installer" \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${installer_arch}.sh"
    else
        die "Neither curl nor wget is installed"
    fi
    bash "$installer" -b -p "$CONDA_ROOT"
    CONDA_BIN=$CONDA_ROOT/bin/conda
}

resolve_conda
CONDA_BASE=$("$CONDA_BIN" info --base)
[[ -n "$CONDA_BASE" ]] || die "Could not determine the Conda base directory"
if [[ -z "$RAINFUSION_ENV" ]]; then
    RAINFUSION_ENV=$CONDA_BASE/envs/$CONDA_ENV_NAME
fi
PYTHON_BIN=$RAINFUSION_ENV/bin/python

# Make `conda activate bernini` available in future interactive Bash shells.
if ! "$CONDA_BIN" init bash >/dev/null 2>&1; then
    log "Warning: conda init bash failed; use $CONDA_BASE/etc/profile.d/conda.sh manually"
fi

write_runtime_env() {
    local runtime_env=$PROJECT_ROOT/env_ascend950.sh
    cat >"$runtime_env" <<EOF
#!/usr/bin/env bash
# Generated by setup.sh. Source this file to activate Conda and configure the benchmarks.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
export PROJECT_ROOT="$PROJECT_ROOT"
export CONDA_BIN="$CONDA_BIN"
export RAINFUSION_ENV="$RAINFUSION_ENV"
export BERNINI_CONFIG="$MODEL_DIR"
export PIXELPRUNE_ROOT="$PIXELPRUNE_DIR"
export EDITVERSE_DATA_ROOT="$EDITVERSE_DATA_ROOT"
export EDITVERSE_TEST_JSON="$EDITVERSE_DATA_ROOT/EditVerseBench.json"
export OPENVE_ROOT="$OPENVE_ROOT"
export OPENVE_CSV="$OPENVE_ROOT/benchmark_videos.csv"
export ASCEND_RT_VISIBLE_DEVICES="\${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}"
source "$CANN_ENV_SCRIPT"
EOF
    chmod 600 "$runtime_env"
    log "Runtime variables written to $runtime_env"
}

verify_environment() {
    [[ -x "$PYTHON_BIN" ]] || die "Python environment not found: $PYTHON_BIN"

    log "Verifying Python imports and NPU availability"
    ASCEND_RT_VISIBLE_DEVICES=$VERIFY_DEVICE "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import pixelprune
import transformers
import diffusers
import accelerate
import decord
import veomni
from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("accelerate:", accelerate.__version__)
print("decord:", decord.__version__)
print("NPU available:", torch.npu.is_available())
assert torch.npu.is_available(), "torch.npu.is_available() is False"
assert hasattr(torch_npu, "npu_fusion_attention"), "torch_npu.npu_fusion_attention is unavailable"

x = torch.randn(2, 2, dtype=torch.float16, device="npu:0")
print("NPU matmul:", x @ x)

q = torch.randn(1, 128, 2, 128, dtype=torch.float16, device="npu:0")
fa_out = torch_npu.npu_fusion_attention(
    q,
    q,
    q,
    head_num=2,
    input_layout="BSND",
    scale=128 ** -0.5,
    pre_tockens=2147483647,
    next_tockens=2147483647,
)[0]
print("npu_fusion_attention:", tuple(fa_out.shape))
print("All required imports succeeded")
PY

    write_runtime_env

    log "Running the real RainFusion NPU smoke test on visible device $VERIFY_DEVICE"
    cd "$BERNINI_DIR"
    if ! ASCEND_RT_VISIBLE_DEVICES=$VERIFY_DEVICE "$PYTHON_BIN" -m pytest -q \
        tests/test_rainfusion.py::test_npu_single_block_v3_matches_dense_attention; then
        cat >&2 <<'EOF'

The Python/NPU environment was installed, but the Bernini public RF-v3
integration test failed. Do not run the four full benchmark scripts until the
error above is resolved and this test passes.
EOF
        return 2
    fi

    log "Environment verification passed"
}

if [[ "$MODE" == verify ]]; then
    verify_environment
    exit 0
fi

for command_name in git make g++; do
    command -v "$command_name" >/dev/null 2>&1 || \
        die "Missing build command '$command_name'. Install gcc/g++, make, git and FFmpeg development packages first."
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    log "Creating Conda environment '$CONDA_ENV_NAME' at $RAINFUSION_ENV"
    "$CONDA_BIN" create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip
else
    log "Reusing Conda environment '$CONDA_ENV_NAME' at $RAINFUSION_ENV"
fi

export PATH="$RAINFUSION_ENV/bin:$PATH"
# Do not let packages under /root/.local leak into the named Conda environment.
export PYTHONNOUSERSITE=1

log "Updating Python packaging and build tools"
"$PYTHON_BIN" -m pip install -U pip setuptools wheel cmake ninja

machine_arch=$(uname -m)
log "Installing PyTorch $TORCH_VERSION and TorchNPU $TORCH_NPU_VERSION for $machine_arch"
case "$machine_arch" in
    aarch64|arm64)
        "$PYTHON_BIN" -m pip install \
            "torch==$TORCH_VERSION" \
            "torchvision==$TORCHVISION_VERSION"
        ;;
    x86_64|amd64)
        "$PYTHON_BIN" -m pip install \
            "torch==${TORCH_VERSION}+cpu" \
            "torchvision==${TORCHVISION_VERSION}+cpu" \
            --index-url https://download.pytorch.org/whl/cpu
        ;;
    *) die "Unsupported host architecture: $machine_arch" ;;
esac

"$PYTHON_BIN" -m pip install \
    "torch-npu==$TORCH_NPU_VERSION" \
    "torchdata==$TORCHDATA_VERSION"

log "Installing Bernini runtime dependencies"
"$PYTHON_BIN" -m pip install \
    accelerate==1.9.0 \
    diffusers==0.35.1 \
    transformers==4.57.3 \
    numpy==1.26.4 \
    einops==0.8.2 \
    safetensors==0.7.0 \
    pillow \
    tqdm \
    ftfy \
    'scipy>=1.7.3' \
    imageio==2.37.3 \
    imageio-ffmpeg==0.6.0 \
    opencv-python==4.11.0.86 \
    blobfile==3.1.0 \
    datasets==2.21.0 \
    packaging==25.0 \
    'absl-py>=2.0.0' \
    'attrs>=23.0.0' \
    'decorator>=5.1.0' \
    pyzmq \
    strenum \
    tiktoken \
    'psutil>=5.9.0' \
    timm \
    wandb \
    pytest \
    huggingface_hub

log "Checking PyTorch and the built-in TorchNPU fused-attention API"
"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
assert hasattr(torch_npu, "npu_fusion_attention"), "torch_npu.npu_fusion_attention is unavailable"
PY

install_veomni() {
    command -v git >/dev/null 2>&1 || die "git is required to install VeOmni"

    if [[ ! -d "$VEOMNI_DIR/.git" ]]; then
        log "Preparing a local VeOmni checkout at $VEOMNI_DIR"
        mkdir -p "$VEOMNI_DIR"
        git -C "$VEOMNI_DIR" init
        git -C "$VEOMNI_DIR" remote add origin https://github.com/ByteDance-Seed/VeOmni.git
    else
        git -C "$VEOMNI_DIR" remote set-url origin https://github.com/ByteDance-Seed/VeOmni.git
    fi

    # Fetch the small pinned tag without Git's partial-clone/promisor mode.
    # VS Code sometimes leaves a GIT_ASKPASS path pointing at a deleted server
    # installation, so remove those variables for this public repository.
    local attempt fetched=0
    for attempt in 1 2 3; do
        log "Fetching VeOmni $VEOMNI_TAG (attempt $attempt/3)"
        if env \
            -u GIT_ASKPASS \
            -u SSH_ASKPASS \
            -u VSCODE_GIT_ASKPASS_NODE \
            -u VSCODE_GIT_ASKPASS_MAIN \
            -u VSCODE_GIT_IPC_HANDLE \
            GIT_TERMINAL_PROMPT=0 \
            git -C "$VEOMNI_DIR" -c http.version=HTTP/1.1 \
                fetch --force --depth=1 origin "refs/tags/$VEOMNI_TAG"; then
            fetched=1
            break
        fi
    done
    (( fetched == 1 )) || die "Failed to fetch VeOmni after 3 attempts. Check this machine's GitHub connection."

    git -C "$VEOMNI_DIR" checkout --detach FETCH_HEAD
    local actual_ref
    actual_ref=$(git -C "$VEOMNI_DIR" rev-parse HEAD)
    [[ "$actual_ref" == "$VEOMNI_REF" ]] || \
        die "VeOmni $VEOMNI_TAG resolved to $actual_ref, expected $VEOMNI_REF"

    log "Installing VeOmni from the verified local checkout"
    "$PYTHON_BIN" -m pip install --no-deps "$VEOMNI_DIR"
}

install_veomni

install_decord_from_source() {
    command -v pkg-config >/dev/null 2>&1 || \
        die "pkg-config is required to build Decord"
    if ! pkg-config --exists libavcodec libavfilter libavformat libavutil; then
        die "FFmpeg development libraries are missing. On openEuler install ffmpeg-devel; on Ubuntu install libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev."
    fi

    # A .pc file and the shared libraries can exist even when the development
    # headers are not installed. Decord specifically needs this header root.
    local ffmpeg_include_dir="" candidate flag
    candidate=$(pkg-config --variable=includedir libavcodec 2>/dev/null || true)
    if [[ -n "$candidate" && -f "$candidate/libavcodec/avcodec.h" ]]; then
        ffmpeg_include_dir=$candidate
    fi
    if [[ -z "$ffmpeg_include_dir" ]]; then
        for flag in $(pkg-config --cflags-only-I libavcodec 2>/dev/null || true); do
            candidate=${flag#-I}
            if [[ -f "$candidate/libavcodec/avcodec.h" ]]; then
                ffmpeg_include_dir=$candidate
                break
            fi
        done
    fi
    if [[ -z "$ffmpeg_include_dir" ]]; then
        for candidate in /usr/include /usr/include/ffmpeg /usr/local/include /usr/local/include/ffmpeg; do
            if [[ -f "$candidate/libavcodec/avcodec.h" ]]; then
                ffmpeg_include_dir=$candidate
                break
            fi
        done
    fi
    if [[ -z "$ffmpeg_include_dir" ]]; then
        cat >&2 <<'EOF'

FFmpeg runtime libraries were found, but the development header
libavcodec/avcodec.h is missing. Install the package that owns it, then rerun
this setup script. On openEuler/RHEL-compatible systems, run as root:

  dnf provides '*/libavcodec/avcodec.h'
  dnf install -y ffmpeg-devel

If the first command reports a differently named package, install that package
instead of ffmpeg-devel.
EOF
        return 1
    fi
    log "Using FFmpeg headers from $ffmpeg_include_dir"

    if [[ ! -d "$DECORD_DIR/.git" ]]; then
        log "Cloning Decord into $DECORD_DIR"
        git clone --recursive https://github.com/dmlc/decord.git "$DECORD_DIR"
    fi
    git -C "$DECORD_DIR" fetch origin "$DECORD_REF"
    git -C "$DECORD_DIR" checkout --detach "$DECORD_REF"

    local avcodec_major
    avcodec_major=$(pkg-config --modversion libavcodec | cut -d. -f1)
    if [[ "$avcodec_major" =~ ^[0-9]+$ ]] && (( avcodec_major >= 59 )); then
        log "Applying Decord compatibility fixes for libavcodec $avcodec_major"
        "$PYTHON_BIN" - "$DECORD_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

common = root / "src/video/ffmpeg/ffmpeg_common.h"
text = common.read_text()
bsf_include = "#include <libavcodec/bsf.h>"
if bsf_include not in text:
    anchor = "#include <libavcodec/avcodec.h>"
    if anchor not in text:
        raise SystemExit(f"could not patch {common}: include anchor is missing")
    common.write_text(text.replace(anchor, f"{anchor}\n{bsf_include}", 1))

reader = root / "src/video/video_reader.cc"
text = reader.read_text()
old = "AVCodec *dec;"
new = "const AVCodec *dec;"
if new not in text:
    if old not in text:
        raise SystemExit(f"could not patch {reader}: decoder declaration is missing")
    reader.write_text(text.replace(old, new, 1))
PY
    fi

    log "Building Decord without CUDA"
    cmake -S "$DECORD_DIR" -B "$DECORD_DIR/build" \
        -DUSE_CUDA=0 \
        -DFFMPEG_AVCODEC_INCLUDE_DIR="$ffmpeg_include_dir" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$DECORD_DIR/build" --parallel "$BUILD_JOBS"
    "$PYTHON_BIN" -m pip install --no-deps -e "$DECORD_DIR/python"
}

log "Installing Decord"
if ! "$PYTHON_BIN" -m pip install decord==0.6.0; then
    log "No usable Decord wheel was found; falling back to a source build"
    install_decord_from_source
fi

build_mindiesd() {
    if [[ ! -d "$MINDIESD_DIR/.git" ]]; then
        log "Cloning MindIE-SD into $MINDIESD_DIR"
        git clone https://gitcode.com/Ascend/MindIE-SD.git "$MINDIESD_DIR"
    fi

    git -C "$MINDIESD_DIR" fetch origin "$MINDIESD_REF"
    git -C "$MINDIESD_DIR" checkout --detach "$MINDIESD_REF"

    log "Building MindIE-SD for ascend950"
    export ASCEND_COMPUTE_UNIT=ascend950
    cd "$MINDIESD_DIR"
    "$PYTHON_BIN" setup.py bdist_wheel

    local wheel_path
    wheel_path=$(find "$MINDIESD_DIR/dist" -maxdepth 1 -type f -name 'mindiesd-*.whl' \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    [[ -n "$wheel_path" && -f "$wheel_path" ]] || die "MindIE-SD wheel was not produced"
    "$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$wheel_path"
    printf '%s\n' "$MINDIESD_REF" >"$MINDIESD_DIR/.bernini_ascend950_built"
}

if [[ "${FORCE_REBUILD_MINDIESD:-0}" == 1 ]] || \
   [[ ! -f "$MINDIESD_DIR/.bernini_ascend950_built" ]]; then
    build_mindiesd
else
    log "Reusing the existing Ascend 950 MindIE-SD build"
fi

log "Installing PixelPrune and Bernini from the combined repository"
"$PYTHON_BIN" -m pip install --no-deps -e "$PIXELPRUNE_DIR"
"$PYTHON_BIN" -m pip install --no-deps -e "$BERNINI_DIR"

verify_environment

cat <<EOF

Setup completed.

For each new shell, run:
  source "$PROJECT_ROOT/env_ascend950.sh"

The generated file runs `conda activate $CONDA_ENV_NAME` automatically.

Then enter Bernini:
  cd "$BERNINI_DIR"

Models and datasets are intentionally not downloaded by setup.sh.
EOF
