#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "\nERROR: setup_clone.sh failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

# Build a Bernini environment by cloning an existing, working Ascend/Wan
# environment.  Existing distributions in the clone are deliberately kept at
# their original versions.  Missing dependencies are installed one at a time
# with --no-deps so pip cannot replace torch, torch-npu, torchvision, or any
# other package inherited from the source environment.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

SOURCE_ENV=${SOURCE_ENV:-/home/qirui1547986/conda-envs/qirui_sparse}
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}
BASE_ROOT=${BASE_ROOT:-$(dirname -- "$PROJECT_ROOT")}
CLONE_ENV_NAME=${CLONE_ENV_NAME:-bernini_clone}
CLONE_ENV_ROOT=${CLONE_ENV_ROOT:-${BASE_ROOT}/conda-envs}
CLONE_ENV=${CLONE_ENV:-${CLONE_ENV_ROOT}/${CLONE_ENV_NAME}}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
BUILD_JOBS=${BUILD_JOBS:-8}
VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6,7}
VERIFY_PHYSICAL_DEVICE=${VERIFY_PHYSICAL_DEVICE:-${VISIBLE_DEVICES%%,*}}
CLONE_WORK_ROOT=${CLONE_WORK_ROOT:-${BASE_ROOT}/bernini_clone_work}
CLONE_TMPDIR=${CLONE_TMPDIR:-${CLONE_WORK_ROOT}/tmp}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${CLONE_WORK_ROOT}/pip-cache}
CLONE_CONDA_PKGS_DIR=${CLONE_CONDA_PKGS_DIR:-${CLONE_WORK_ROOT}/conda-pkgs}
# Optional colon-separated list of package caches belonging to the source
# environment's Conda installation.  They are searched after the writable
# clone cache and are never cleaned by this script.
SOURCE_CONDA_PKGS_DIRS=${SOURCE_CONDA_PKGS_DIRS:-}
REQUESTED_CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-}
REQUIRED_CANN_ROOT=${REQUIRED_CANN_ROOT:-/usr/local/Ascend/cann-9.1.0}
FORBIDDEN_CANN_ROOT=${FORBIDDEN_CANN_ROOT:-/home/qirui1547986/Ascend/cann-9.2.0}
CANN_ENV_SCRIPT=${CANN_ENV_SCRIPT:-${REQUIRED_CANN_ROOT}/set_env.sh}

MODEL_DIR=${MODEL_DIR:-${BASE_ROOT}/Bernini-Diffusers}
EDITVERSE_DATA_ROOT=${EDITVERSE_DATA_ROOT:-${BASE_ROOT}/datasets/EditVerse/EditVerseBench}
OPENVE_ROOT=${OPENVE_ROOT:-${BASE_ROOT}/datasets/OpenVE}
VEOMNI_DIR=${VEOMNI_DIR:-${BASE_ROOT}/VeOmni}
DECORD_DIR=${DECORD_DIR:-${BASE_ROOT}/decord}
MINDIESD_DIR=${MINDIESD_DIR:-${BASE_ROOT}/MindIE-SD}
FLASH_ATTN_NPU_DIR=${FLASH_ATTN_NPU_DIR:-${BASE_ROOT}/flash-attention-npu}
VEOMNI_TAG=${VEOMNI_TAG:-v0.1.10}
VEOMNI_REF=${VEOMNI_REF:-6ab293ecdfdd90ef3941fc81065d9f947b5b4e4f}

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
  bash setup_clone.sh
  bash setup_clone.sh --clone-only
  bash setup_clone.sh --verify-only

The default source and destination are:
  source: /home/qirui1547986/conda-envs/qirui_sparse
  target: <project-parent>/conda-envs/bernini_clone

Important optional overrides:
  SOURCE_ENV=/path/to/working/environment
  BASE_ROOT=/home/<user>
  CLONE_ENV=/home/<user>/conda-envs/bernini_clone
  PROJECT_ROOT=/home/<user>/bernini_pixelprune
  MODEL_DIR=/home/<user>/Bernini-Diffusers
  CONDA_BIN=/root/miniconda3/bin/conda
  CANN_ENV_SCRIPT=/usr/local/Ascend/cann-9.1.0/set_env.sh
  REQUIRED_CANN_ROOT=/usr/local/Ascend/cann-9.1.0
  ASCEND_RT_VISIBLE_DEVICES=6,7
  BUILD_JOBS=8
  CLONE_WORK_ROOT=/home/<user>/bernini_clone_work
  SOURCE_CONDA_PKGS_DIRS=/home/qirui1547986/miniconda3/pkgs

This script never runs setup.sh and never uses pip install -U.  It does not
modify SOURCE_ENV.  Re-running it reuses CLONE_ENV and only retries unfinished
steps.  To make a completely fresh clone, choose a new CLONE_ENV path.
EOF
}

configure_conda_package_caches() {
    local source_owner_root candidate cache_list=
    local -a candidates=() requested_caches=() source_caches=()

    # The first cache is on /home and is writable.  Conda can download or
    # extract here without filling the root filesystem.
    mkdir -p "$CLONE_CONDA_PKGS_DIR"
    candidates+=("$CLONE_CONDA_PKGS_DIR")

    if [[ -n "$REQUESTED_CONDA_PKGS_DIRS" ]]; then
        IFS=: read -r -a requested_caches <<<"$REQUESTED_CONDA_PKGS_DIRS"
        candidates+=("${requested_caches[@]}")
    elif [[ -n "$SOURCE_CONDA_PKGS_DIRS" ]]; then
        IFS=: read -r -a source_caches <<<"$SOURCE_CONDA_PKGS_DIRS"
        candidates+=("${source_caches[@]}")
    else
        # The source prefix lives outside our Conda base.  A Conda clone still
        # needs the original package cache records, so search the common cache
        # locations for both installations instead of hiding them behind a new
        # empty CONDA_PKGS_DIRS.
        source_owner_root=$(dirname -- "$(dirname -- "$SOURCE_ENV")")
        candidates+=(
            "$source_owner_root/.conda/pkgs"
            "$source_owner_root/miniconda3/pkgs"
            "$source_owner_root/anaconda3/pkgs"
            "$CONDA_BASE/pkgs"
            "/root/.conda/pkgs"
        )
    fi

    local -A seen=()
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" && -d "$candidate" ]] || continue
        candidate=$(cd -- "$candidate" && pwd -P)
        [[ -z "${seen[$candidate]:-}" ]] || continue
        seen[$candidate]=1
        if [[ -n "$cache_list" ]]; then
            cache_list+=":$candidate"
        else
            cache_list=$candidate
        fi
    done

    [[ -n "$cache_list" ]] || die "no usable Conda package cache was found"
    export CONDA_PKGS_DIRS=$cache_list
    log "Conda package caches: $CONDA_PKGS_DIRS"
}

MODE=install
case "${1:-}" in
    "") ;;
    --clone-only) MODE=clone ;;
    --verify-only) MODE=verify ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
esac

resolve_layout() {
    if [[ -z "${BERNINI_DIR:-}" ]]; then
        if [[ -f "$PROJECT_ROOT/bernini/infer_multi_gpu.py" && -d "$PROJECT_ROOT/bernini/bernini" ]]; then
            BERNINI_DIR=$PROJECT_ROOT/bernini
        elif [[ -f "$PROJECT_ROOT/infer_multi_gpu.py" && -d "$PROJECT_ROOT/bernini" ]]; then
            BERNINI_DIR=$PROJECT_ROOT
        else
            die "could not find the Bernini source under $PROJECT_ROOT"
        fi
    fi

    if [[ -z "${PIXELPRUNE_DIR:-}" ]]; then
        local candidate
        for candidate in \
            "$PROJECT_ROOT/pixelprune" \
            "$PROJECT_ROOT/PixelPrune" \
            "$(dirname -- "$BERNINI_DIR")/pixelprune" \
            "$(dirname -- "$BERNINI_DIR")/PixelPrune"; do
            if [[ -f "$candidate/setup.py" && -d "$candidate/pixelprune" ]]; then
                PIXELPRUNE_DIR=$candidate
                break
            fi
        done
    fi

    [[ -f "$BERNINI_DIR/infer_multi_gpu.py" ]] || die "missing Bernini source: $BERNINI_DIR"
    [[ -n "${PIXELPRUNE_DIR:-}" && -d "$PIXELPRUNE_DIR/pixelprune" ]] || \
        die "could not find PixelPrune; set PIXELPRUNE_DIR explicitly"

    local qwen_file=$BERNINI_DIR/bernini/models/modeling_qwen2_5_vl.py
    if grep -Eq '^[[:space:]]*from[[:space:]]+flash_attn_npu[[:space:]]+import' "$qwen_file"; then
        cat >&2 <<EOF

The Bernini Qwen attention code still imports the Ascend 910/v2 module:
  $qwen_file

Ascend 950 requires the already-tested flash_attn_npu_3 adaptation. Push or
copy that adapted file to this checkout before running setup_clone.sh.
EOF
        return 1
    fi
}

detect_cann_env() {
    if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
        [[ -f "$CANN_ENV_SCRIPT" ]] || die "CANN_ENV_SCRIPT does not exist: $CANN_ENV_SCRIPT"
        return
    fi

    local candidate
    for candidate in \
        /usr/local/Ascend/cann-9.1.0/set_env.sh \
        /usr/local/Ascend/cann/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
        if [[ -f "$candidate" ]]; then
            CANN_ENV_SCRIPT=$candidate
            return
        fi
    done
    die "CANN set_env.sh was not found"
}

load_clean_cann() {
    local python_prefix=$1 variable value

    # Do not inherit a different user's Toolkit/OPP/Python paths through the
    # source environment's activation hooks or the invoking login shell.
    export PYTHONPATH= LD_LIBRARY_PATH= LIBRARY_PATH= CPATH=
    export C_INCLUDE_PATH= CPLUS_INCLUDE_PATH= PKG_CONFIG_PATH=
    export ASCEND_HOME_PATH= ASCEND_TOOLKIT_HOME= ASCEND_OPP_PATH=
    export ASCEND_AICPU_PATH= ASCEND_CUSTOM_OPP_PATH= TBE_IMPL_PATH= TOOLCHAIN_HOME=

    export PATH="$python_prefix/bin:$CONDA_BASE/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    # shellcheck disable=SC1090
    source "$CANN_ENV_SCRIPT"

    [[ "${ASCEND_HOME_PATH:-}" == "$REQUIRED_CANN_ROOT" ]] || \
        die "CANN environment resolved to ${ASCEND_HOME_PATH:-UNSET}; expected $REQUIRED_CANN_ROOT"

    for variable in PATH PYTHONPATH LD_LIBRARY_PATH LIBRARY_PATH CPATH \
        C_INCLUDE_PATH CPLUS_INCLUDE_PATH PKG_CONFIG_PATH ASCEND_HOME_PATH \
        ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH \
        ASCEND_CUSTOM_OPP_PATH TBE_IMPL_PATH TOOLCHAIN_HOME; do
        value=${!variable-}
        if [[ "$value" == *"$FORBIDDEN_CANN_ROOT"* ]]; then
            die "$variable still contains forbidden CANN path $FORBIDDEN_CANN_ROOT"
        fi
    done

    log "Using clean CANN environment from $ASCEND_HOME_PATH"
}

python_imports() {
    local module=$1
    "$PYTHON_BIN" -c "import ${module}" >/dev/null 2>&1
}

snapshot_distributions() {
    local output_file=$1
    "$PYTHON_BIN" - "$output_file" <<'PY'
from importlib.metadata import distributions
from pathlib import Path
import sys

items = {}
for dist in distributions():
    name = dist.metadata.get("Name")
    if name:
        items[name.lower().replace("_", "-")] = dist.version

path = Path(sys.argv[1])
path.write_text("".join(f"{name}\t{items[name]}\n" for name in sorted(items)))
print(f"saved {len(items)} distributions to {path}")
PY
}

compare_existing_distributions() {
    local before_file=$1
    "$PYTHON_BIN" - "$before_file" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

changed = []
removed = []
for line in Path(sys.argv[1]).read_text().splitlines():
    name, old = line.split("\t", 1)
    try:
        new = version(name)
    except PackageNotFoundError:
        removed.append((name, old))
        continue
    if new != old:
        changed.append((name, old, new))

if changed or removed:
    print("Existing distributions unexpectedly changed:", file=sys.stderr)
    for name, old, new in changed:
        print(f"  {name}: {old} -> {new}", file=sys.stderr)
    for name, old in removed:
        print(f"  {name}: {old} -> MISSING", file=sys.stderr)
    raise SystemExit(1)

print("All distributions inherited from the source environment kept their versions.")
PY
}

install_missing_python_dependencies() {
    "$PYTHON_BIN" - <<'PY'
from collections import deque
import importlib
import importlib.metadata as md
import subprocess
import sys

# The second item is used only when the distribution is completely absent.
# An installed distribution is kept even when its version differs.
direct = [
    ("setuptools", "setuptools"),
    ("wheel", "wheel"),
    ("cmake", "cmake"),
    ("ninja", "ninja"),
    ("accelerate", "accelerate==1.9.0"),
    ("diffusers", "diffusers==0.35.1"),
    ("transformers", "transformers==4.57.3"),
    ("numpy", "numpy==1.26.4"),
    ("einops", "einops==0.8.2"),
    ("safetensors", "safetensors==0.7.0"),
    ("pillow", "pillow"),
    ("tqdm", "tqdm"),
    ("ftfy", "ftfy"),
    ("scipy", "scipy>=1.7.3"),
    ("imageio", "imageio==2.37.3"),
    ("imageio-ffmpeg", "imageio-ffmpeg==0.6.0"),
    ("opencv-python", "opencv-python==4.11.0.86"),
    ("blobfile", "blobfile==3.1.0"),
    ("datasets", "datasets==2.21.0"),
    ("packaging", "packaging==25.0"),
    ("absl-py", "absl-py>=2.0.0"),
    ("attrs", "attrs>=23.0.0"),
    ("decorator", "decorator>=5.1.0"),
    ("lxml", "lxml>=4.9.0"),
    ("psutil", "psutil>=5.9.0"),
    ("sympy", "sympy>=1.13.3"),
    ("pyzmq", "pyzmq"),
    ("strenum", "strenum"),
    ("tiktoken", "tiktoken"),
    ("timm", "timm>=1.0"),
    ("torchdata", "torchdata==0.11.0"),
    ("wandb", "wandb"),
    ("pytest", "pytest"),
    ("huggingface-hub", "huggingface_hub"),
]

def install(spec):
    print(f"INSTALL {spec}", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-deps", spec
    ])
    importlib.invalidate_caches()

for name, spec in direct:
    try:
        current = md.version(name)
        print(f"KEEP    {name}=={current}")
    except md.PackageNotFoundError:
        install(spec)

# Install missing transitive dependencies without changing anything already
# present.  Version conflicts are reported and deliberately left untouched.
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

queue = deque(canonicalize_name(name) for name, _ in direct)
seen = set()
conflicts = []

while queue:
    name = queue.popleft()
    if name in seen:
        continue
    seen.add(name)

    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        continue

    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue

        dep = canonicalize_name(req.name)
        try:
            current = md.version(dep)
        except md.PackageNotFoundError:
            spec = str(req).split(";", 1)[0].strip()
            install(spec)
            current = md.version(dep)

        if req.specifier and current not in req.specifier:
            conflicts.append((dep, current, str(req.specifier), name))
        queue.append(dep)

if conflicts:
    print("\nExisting version conflicts were preserved:")
    for dep, current, wanted, parent in sorted(set(conflicts)):
        print(f"  {dep}=={current}; {parent} declares {wanted}")
PY
}

prepare_veomni() {
    if python_imports veomni; then
        log "Keeping VeOmni inherited from the source environment"
        return
    fi

    if [[ -e "$VEOMNI_DIR" && ! -d "$VEOMNI_DIR/.git" ]]; then
        die "$VEOMNI_DIR exists but is not a Git checkout; set VEOMNI_DIR to a clean location"
    fi

    if [[ ! -d "$VEOMNI_DIR/.git" ]]; then
        mkdir -p "$VEOMNI_DIR"
        git -C "$VEOMNI_DIR" init
        git -C "$VEOMNI_DIR" remote add origin https://github.com/ByteDance-Seed/VeOmni.git
        local attempt fetched=0
        for attempt in 1 2 3; do
            log "Fetching VeOmni $VEOMNI_TAG (attempt $attempt/3)"
            if env -u GIT_ASKPASS -u SSH_ASKPASS \
                -u VSCODE_GIT_ASKPASS_NODE -u VSCODE_GIT_ASKPASS_MAIN \
                -u VSCODE_GIT_IPC_HANDLE GIT_TERMINAL_PROMPT=0 \
                git -C "$VEOMNI_DIR" -c http.version=HTTP/1.1 \
                    fetch --force --depth=1 origin "refs/tags/$VEOMNI_TAG"; then
                fetched=1
                break
            fi
        done
        (( fetched == 1 )) || die "failed to fetch VeOmni $VEOMNI_TAG"
        git -C "$VEOMNI_DIR" checkout --detach FETCH_HEAD
    fi

    local actual_ref
    actual_ref=$(git -C "$VEOMNI_DIR" rev-parse HEAD)
    [[ "$actual_ref" == "$VEOMNI_REF" ]] || \
        die "existing VeOmni checkout is $actual_ref, expected $VEOMNI_REF; set VEOMNI_DIR to a fresh path"

    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation "$VEOMNI_DIR"
    python_imports veomni || die "VeOMni still cannot be imported after installation"
}

prepare_decord() {
    if python_imports decord; then
        log "Keeping Decord inherited from the source environment"
        return
    fi

    [[ -f "$DECORD_DIR/python/setup.py" ]] || cat >&2 <<EOF

Decord is missing from the clone, and its prepared source tree was not found:
  $DECORD_DIR

Set DECORD_DIR to the Decord source tree previously built against the system
FFmpeg development libraries, then rerun this script.
EOF
    [[ -f "$DECORD_DIR/python/setup.py" ]] || return 1

    log "Installing Decord from the existing prepared source tree"
    "$PYTHON_BIN" -m pip install --no-deps -e "$DECORD_DIR/python"
    python_imports decord || \
        die "Decord was installed but cannot load; rebuild $DECORD_DIR against the system FFmpeg headers"
}

prepare_mindiesd() {
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention
PY
    then
        log "Keeping MindIE-SD inherited from the source environment"
        return
    fi

    [[ -f "$MINDIESD_DIR/setup.py" ]] || \
        die "MindIE-SD is missing; set MINDIESD_DIR to its source checkout"

    local build_dir wheel_path
    build_dir=$(mktemp -d "${BASE_ROOT}/mindiesd-bernini-clone.XXXXXX")
    log "Rebuilding MindIE-SD against the cloned Torch environment in $build_dir"
    cp -a "$MINDIESD_DIR/." "$build_dir/"

    [[ "$build_dir" == "${BASE_ROOT}/mindiesd-bernini-clone."* ]] || \
        die "refusing to clean unexpected MindIE-SD build directory: $build_dir"
    rm -rf -- "$build_dir/build/build" "$build_dir/dist" "$build_dir/wheel_build"
    mkdir -p "$build_dir/wheels"
    if [[ -d "$build_dir/mindiesd/plugin" ]]; then
        find "$build_dir/mindiesd/plugin" -maxdepth 1 -type f -name '*.so' -delete
    fi

    (
        cd "$build_dir"
        export ASCEND_COMPUTE_UNIT=ascend950
        export ASCEND_INSTALL_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}
        "$PYTHON_BIN" setup.py \
            build --build-base "$build_dir/wheel_build" --force \
            bdist_wheel --dist-dir "$build_dir/wheels"
    )

    wheel_path=$(find "$build_dir/wheels" -maxdepth 1 -type f -name 'mindiesd-*.whl' | head -n 1)
    [[ -n "$wheel_path" && -f "$wheel_path" ]] || die "MindIE-SD did not produce a wheel"
    "$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$wheel_path"

    "$PYTHON_BIN" - <<'PY' >/dev/null
from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention
PY
}

prepare_flash_attention_npu() {
    if python_imports flash_attn_npu_3; then
        log "Keeping flash_attn_npu_3 inherited from the source environment"
        return
    fi

    if [[ ! -d "$FLASH_ATTN_NPU_DIR/.git" ]]; then
        [[ ! -e "$FLASH_ATTN_NPU_DIR" ]] || \
            die "$FLASH_ATTN_NPU_DIR exists but is not a Git checkout"
        local clone_tmp
        clone_tmp=$(mktemp -d "${BASE_ROOT}/flash-attention-source.XXXXXX")
        log "Cloning MinghuasLab/flash-attention-npu"
        env -u GIT_ASKPASS -u SSH_ASKPASS \
            -u VSCODE_GIT_ASKPASS_NODE -u VSCODE_GIT_ASKPASS_MAIN \
            -u VSCODE_GIT_IPC_HANDLE GIT_TERMINAL_PROMPT=0 \
            git -c http.version=HTTP/1.1 clone --recursive \
                https://github.com/MinghuasLab/flash-attention-npu.git \
                "$clone_tmp/repo"
        mv "$clone_tmp/repo" "$FLASH_ATTN_NPU_DIR"
        rmdir "$clone_tmp"
    else
        git -C "$FLASH_ATTN_NPU_DIR" submodule update --init --recursive
    fi

    local source_ref build_dir wheel_path
    source_ref=$(git -C "$FLASH_ATTN_NPU_DIR" rev-parse HEAD)
    log "Building FlashAttention-NPU v3/950 from commit $source_ref"
    build_dir=$(mktemp -d "${BASE_ROOT}/flash-attention-bernini-clone.XXXXXX")
    cp -a "$FLASH_ATTN_NPU_DIR/." "$build_dir/"

    [[ "$build_dir" == "${BASE_ROOT}/flash-attention-bernini-clone."* ]] || \
        die "refusing to clean unexpected FlashAttention build directory: $build_dir"
    rm -rf -- "$build_dir/build" "$build_dir/dist" "$build_dir/wheel_build"
    mkdir -p "$build_dir/wheels"

    (
        cd "$build_dir"
        export MAX_JOBS=$BUILD_JOBS
        export FLASH_ATTENTION_FORCE_BUILD=TRUE
        export FLASH_ATTN_BUILD_VERSION=v3
        export FLASH_ATTN_BUILD_NPU=950
        "$PYTHON_BIN" setup.py \
            build --build-base "$build_dir/wheel_build" --force \
            bdist_wheel --dist-dir "$build_dir/wheels"
    )

    wheel_path=$(find "$build_dir/wheels" -maxdepth 1 -type f -name '*.whl' | head -n 1)
    [[ -n "$wheel_path" && -f "$wheel_path" ]] || \
        die "FlashAttention-NPU did not produce a wheel"
    "$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$wheel_path"
    python_imports flash_attn_npu_3 || die "flash_attn_npu_3 still cannot be imported"
}

install_project_sources() {
    log "Installing PixelPrune and Bernini without dependency resolution"
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$PIXELPRUNE_DIR"
    "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$BERNINI_DIR"
}

verify_environment() {
    log "Verifying imports and cloned core versions"
    "$PYTHON_BIN" - <<'PY'
import sys
import torch
import torch_npu
import torchvision
import transformers
import diffusers
import accelerate
import decord
import veomni
import mindiesd
import flash_attn_npu_3
import bernini
import pixelprune

print("python:", sys.version.split()[0], sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("torch_npu:", torch_npu.__version__, torch_npu.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("accelerate:", accelerate.__version__)
print("decord:", decord.__version__)
print("veomni:", veomni.__file__)
print("mindiesd:", mindiesd.__file__)
print("flash_attn_npu_3:", flash_attn_npu_3.__file__)
PY

    log "Running NPU matmul, native fused attention, and FlashAttention-NPU v3 smoke tests on physical device $VERIFY_PHYSICAL_DEVICE"
    ASCEND_RT_VISIBLE_DEVICES=$VERIFY_PHYSICAL_DEVICE "$PYTHON_BIN" - <<'PY'
import math
import torch
import torch_npu
from flash_attn_npu_3 import flash_attn_varlen_func

assert torch.npu.is_available(), "torch.npu.is_available() is False"
torch.npu.set_device(0)

x = torch.randn(256, 256, dtype=torch.float16, device="npu:0")
_ = x @ x

q = torch.randn(1, 128, 2, 128, dtype=torch.float16, device="npu:0")
native = torch_npu.npu_fusion_attention(
    q, q, q,
    head_num=2,
    input_layout="BSND",
    scale=128 ** -0.5,
    pre_tockens=2147483647,
    next_tockens=2147483647,
)[0]

packed = q.squeeze(0).contiguous()
cu = torch.tensor([0, 128], dtype=torch.int32, device="npu:0")
custom = flash_attn_varlen_func(
    packed, packed, packed,
    cu, cu, 128, 128,
    softmax_scale=1.0 / math.sqrt(128),
    causal=False,
)
custom = custom[0] if isinstance(custom, tuple) else custom
torch.npu.synchronize()

print("NPU matmul: PASS")
print("npu_fusion_attention:", tuple(native.shape))
print("flash_attn_npu_3:", tuple(custom.shape))
PY

    log "Running the Bernini RainFusion NPU smoke test"
    (
        cd "$BERNINI_DIR"
        ASCEND_RT_VISIBLE_DEVICES=$VERIFY_PHYSICAL_DEVICE \
            "$PYTHON_BIN" -m pytest -q \
            tests/test_rainfusion.py::test_npu_single_block_v3_matches_dense_attention
    )
}

write_runtime_env() {
    local runtime_env=$PROJECT_ROOT/env_bernini_clone.sh
    local log_dir=${ASCEND_PROCESS_LOG_PATH:-${BASE_ROOT}/ascend_logs_bernini_clone}
    mkdir -p "$log_dir"

    cat >"$runtime_env" <<EOF
#!/usr/bin/env bash
# Generated by setup_clone.sh. Source this file before running Bernini.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CLONE_ENV"

export PYTHONNOUSERSITE=1
export PYTHONPATH= LD_LIBRARY_PATH= LIBRARY_PATH= CPATH=
export C_INCLUDE_PATH= CPLUS_INCLUDE_PATH= PKG_CONFIG_PATH=
export ASCEND_HOME_PATH= ASCEND_TOOLKIT_HOME= ASCEND_OPP_PATH=
export ASCEND_AICPU_PATH= ASCEND_CUSTOM_OPP_PATH= TBE_IMPL_PATH= TOOLCHAIN_HOME=
export PATH="$CLONE_ENV/bin:$CONDA_BASE/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

source "$CANN_ENV_SCRIPT"

if [[ "\${ASCEND_HOME_PATH:-}" != "$REQUIRED_CANN_ROOT" ]]; then
    echo "ERROR: expected CANN $REQUIRED_CANN_ROOT, got \${ASCEND_HOME_PATH:-UNSET}" >&2
    return 1 2>/dev/null || exit 1
fi
if env | grep -F "$FORBIDDEN_CANN_ROOT"; then
    echo "ERROR: forbidden CANN 9.2 path is still present" >&2
    return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT="$PROJECT_ROOT"
export CONDA_BIN="$CONDA_BIN"
export RAINFUSION_ENV="$CLONE_ENV"
export PYTHON_BIN="$CLONE_ENV/bin/python"
export BERNINI_CONFIG="$MODEL_DIR"
export PIXELPRUNE_ROOT="$PIXELPRUNE_DIR"
export EDITVERSE_DATA_ROOT="$EDITVERSE_DATA_ROOT"
export EDITVERSE_TEST_JSON="$EDITVERSE_DATA_ROOT/EditVerseBench.json"
export OPENVE_ROOT="$OPENVE_ROOT"
export OPENVE_CSV="$OPENVE_ROOT/benchmark_videos.csv"
export ASCEND_RT_VISIBLE_DEVICES="\${ASCEND_RT_VISIBLE_DEVICES:-$VISIBLE_DEVICES}"
export ASCEND_PROCESS_LOG_PATH="\${ASCEND_PROCESS_LOG_PATH:-$log_dir}"
EOF
    chmod 644 "$runtime_env"
    log "Runtime environment written to $runtime_env"
}

resolve_layout
detect_cann_env
export PYTHONNOUSERSITE=1
mkdir -p "$CLONE_TMPDIR" "$PIP_CACHE_DIR"
export TMPDIR=$CLONE_TMPDIR
export PIP_CACHE_DIR

[[ -x "$CONDA_BIN" ]] || die "CONDA_BIN is not executable: $CONDA_BIN"
CONDA_BASE=$($CONDA_BIN info --base)
[[ -n "$CONDA_BASE" ]] || die "could not determine the Conda base directory"
configure_conda_package_caches
CONDA_SH=$CONDA_BASE/etc/profile.d/conda.sh
[[ -f "$CONDA_SH" ]] || die "Conda activation script is missing: $CONDA_SH"

if [[ "$MODE" != verify ]]; then
    [[ -x "$SOURCE_ENV/bin/python" ]] || die "source environment is missing: $SOURCE_ENV"

    log "Checking source metadata without importing its incomplete Torch installation"
    (
        # Activation hooks can be required by accelerator environments, so the
        # source must be tested the same way its owner normally runs it.
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$SOURCE_ENV"
        export PYTHONNOUSERSITE=1
        "$SOURCE_ENV/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys
assert sys.version_info >= (3, 11), f"Bernini requires Python >= 3.11; source has {sys.version}"
print("python:", sys.version.split()[0], sys.executable)
for name in ("torch", "torch-npu", "torchvision"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError as exc:
        raise SystemExit(f"required source distribution is missing: {name}") from exc
PY
    )

    if [[ ! -e "$CLONE_ENV" ]]; then
        log "Cloning $SOURCE_ENV to $CLONE_ENV"
        mkdir -p "$CLONE_ENV_ROOT"
        "$CONDA_BIN" create -y --copy --prefix "$CLONE_ENV" --clone "$SOURCE_ENV"
    else
        [[ -x "$CLONE_ENV/bin/python" ]] || \
            die "$CLONE_ENV exists but is not a complete Conda environment; choose another CLONE_ENV"
        log "Reusing the existing clone at $CLONE_ENV"
    fi
fi

# Run all subsequent build and verification steps with the clone's Conda
# activation hooks, not merely with its bin directory prepended to PATH.
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CLONE_ENV"
export PYTHONNOUSERSITE=1
load_clean_cann "$CLONE_ENV"

PYTHON_BIN=$CLONE_ENV/bin/python
[[ -x "$PYTHON_BIN" ]] || die "cloned Python is missing: $PYTHON_BIN"
export PATH="$CLONE_ENV/bin:$PATH"

"$PYTHON_BIN" - "$CLONE_ENV" <<'PY'
import os
import sys
expected = os.path.realpath(sys.argv[1])
actual = os.path.realpath(sys.prefix)
assert actual == expected, f"wrong Python prefix: {actual}; expected {expected}"
print("active clone prefix:", actual)
PY

BASELINE_FILE=$CLONE_ENV/.bernini_clone_original_distributions.tsv
if [[ "$MODE" != verify && ! -f "$BASELINE_FILE" ]]; then
    log "Recording the original cloned distributions"
    snapshot_distributions "$BASELINE_FILE"
    printf '%s\n' "$SOURCE_ENV" >"$CLONE_ENV/.bernini_clone_source"
fi

if [[ "$MODE" == clone ]]; then
    log "Clone-only stage completed; no package was installed or changed"
    "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys
print("python:", sys.version.split()[0], sys.executable)
for name in ("torch", "torch-npu", "torchvision", "transformers", "diffusers", "accelerate", "sympy"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")
PY
    cat <<EOF

Clone-only stage completed.
Source: $SOURCE_ENV
Clone:  $CLONE_ENV
No dependencies were installed. Send this command's output back before running
the full setup.
EOF
    exit 0
fi

if [[ "$MODE" == install ]]; then
    log "Installing only missing Python dependencies"
    install_missing_python_dependencies
    prepare_veomni
    prepare_decord
    prepare_mindiesd
    prepare_flash_attention_npu
    install_project_sources

    log "Checking that inherited package versions were not changed"
    compare_existing_distributions "$BASELINE_FILE"

    log "Running pip check for informational dependency metadata diagnostics"
    "$PYTHON_BIN" -m pip check || \
        log "pip check reported version metadata conflicts; existing versions were intentionally preserved"
fi

verify_environment
write_runtime_env

cat <<EOF

Setup completed successfully.

Environment name: $CLONE_ENV_NAME
Environment path: $CLONE_ENV

For each new shell:
  source "$PROJECT_ROOT/env_bernini_clone.sh"
  cd "$BERNINI_DIR"

Confirm the run scripts use two processes and the intended devices, then run:
  MAX_ITEMS=1 NPROC_PER_NODE=2 ULYSSES=2 bash scripts/pixelprune/run_editverse_baseline.sh

The original source environment was not modified:
  $SOURCE_ENV
EOF

log "Device assignments currently present in the four run scripts"
grep -n 'ASCEND_RT_VISIBLE_DEVICES=' "$BERNINI_DIR"/scripts/pixelprune/run_*.sh || true
