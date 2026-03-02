#!/bin/bash
# =============================================================================
# Gaussian Splatting Pipeline — Setup Script
# Installs everything from scratch into conda environment 'gsplat_env'
# Idempotent: safe to run multiple times
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gsplat_env"
PYTHON_VERSION="3.10"
DEPS_DIR="$SCRIPT_DIR/deps"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[setup]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[  OK  ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR ]${NC} $*"; }

log_info "========================================"
log_info " Gaussian Splatting Pipeline Setup"
log_info " Script dir: $SCRIPT_DIR"
log_info "========================================"

# ── Step 1: Install Miniforge if conda not found ─────────────────────────────
if ! command -v conda &>/dev/null; then
    log_info "conda not found. Installing Miniforge..."
    MINIFORGE_PATH="$HOME/miniforge3"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_PATH"
    eval "$("$MINIFORGE_PATH/bin/conda" shell.bash hook)"
    "$MINIFORGE_PATH/bin/conda" init bash
    rm /tmp/miniforge.sh
    log_ok "Miniforge installed at $MINIFORGE_PATH"
else
    log_ok "conda found: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ── Step 2: Detect system CUDA version ───────────────────────────────────────
log_info "Detecting CUDA version..."

CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
fi
if [ -z "$CUDA_VERSION" ] && [ -f /usr/local/cuda/version.json ]; then
    CUDA_VERSION=$(python3 -c "import json; d=json.load(open('/usr/local/cuda/version.json')); print(d['cuda']['version'])" 2>/dev/null | grep -oP '^[0-9]+\.[0-9]+' || true)
fi
if [ -z "$CUDA_VERSION" ] && command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs python3 -c "
import sys
drv = sys.argv[1]
major = int(drv.split('.')[0])
# Approximate mapping
if major >= 525: print('12.1')
elif major >= 520: print('11.8')
else: print('11.7')
" 2>/dev/null || true)
fi
if [ -z "$CUDA_VERSION" ]; then
    log_error "Cannot detect CUDA version. Ensure nvidia-smi or nvcc is in PATH."
    exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_CUDA="cu121"
    CUDA_TAG="12.1"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    TORCH_CUDA="cu118"
    CUDA_TAG="11.8"
else
    log_error "CUDA $CUDA_VERSION not supported. Need CUDA >= 11.8."
    exit 1
fi

log_ok "System CUDA: $CUDA_VERSION → Using PyTorch CUDA tag: $TORCH_CUDA"

# ── Step 3: Create conda environment ─────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    log_info "Conda environment '$ENV_NAME' already exists. Activating..."
else
    log_info "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    log_ok "Environment '$ENV_NAME' created."
fi

conda activate "$ENV_NAME"
log_ok "Activated: $(python --version)"

# ── Step 4: Install PyTorch (pinned) ─────────────────────────────────────────
TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
if [[ "$TORCH_INSTALLED" == "2.1.2"* ]]; then
    log_ok "PyTorch 2.1.2+${TORCH_CUDA} already installed."
else
    log_info "Installing PyTorch 2.1.2 for ${TORCH_CUDA}..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    log_ok "PyTorch installed."
fi

# ── Step 5: Install pinned pip dependencies ───────────────────────────────────
log_info "Installing pinned pip dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"
log_ok "Pip dependencies installed."

# ── Step 6: Install COLMAP ────────────────────────────────────────────────────
if command -v colmap &>/dev/null; then
    COLMAP_VER=$(colmap --version 2>&1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
    log_ok "COLMAP already available: $COLMAP_VER"
else
    log_info "Installing COLMAP from conda-forge..."
    conda install -c conda-forge colmap -y && log_ok "COLMAP installed via conda-forge." || {
        log_warn "conda-forge COLMAP failed. Building from source..."
        # Build from source
        conda install -c conda-forge cmake boost eigen freeimage glog gflags \
            glew qt-main cgal-cpp ceres-solver suitesparse metis -y
        mkdir -p "$DEPS_DIR"
        if [ ! -d "$DEPS_DIR/colmap-src" ]; then
            git clone https://github.com/colmap/colmap.git "$DEPS_DIR/colmap-src"
        fi
        mkdir -p "$DEPS_DIR/colmap-src/build"
        cd "$DEPS_DIR/colmap-src/build"
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
            -DCUDA_ENABLED=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DTESTS_ENABLED=OFF
        make -j"$(nproc)"
        make install
        cd "$SCRIPT_DIR"
        log_ok "COLMAP built from source."
    }
fi

# ── Step 7: Clone and build original 3DGS ────────────────────────────────────
mkdir -p "$DEPS_DIR"
GS_DIR="$DEPS_DIR/gaussian-splatting"
if [ ! -d "$GS_DIR" ]; then
    log_info "Cloning original 3DGS repo..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive "$GS_DIR"
    log_ok "3DGS repo cloned."
else
    log_info "3DGS repo exists, updating submodules..."
    cd "$GS_DIR" && git submodule update --init --recursive && cd "$SCRIPT_DIR"
fi

# Compile diff-gaussian-rasterization
log_info "Compiling diff-gaussian-rasterization CUDA kernels..."
cd "$GS_DIR"
if python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    log_ok "diff-gaussian-rasterization already compiled."
else
    pip install submodules/diff-gaussian-rasterization && log_ok "diff-gaussian-rasterization compiled." || {
        log_error "Failed to compile diff-gaussian-rasterization."
        log_error "Common fix: ensure CUDA toolkit version matches PyTorch CUDA ($TORCH_CUDA)."
        log_error "Check: nvcc --version matches torch.version.cuda"
        exit 1
    }
fi

# Compile simple-knn
log_info "Compiling simple-knn CUDA kernels..."
if python -c "import simple_knn" 2>/dev/null; then
    log_ok "simple-knn already compiled."
else
    pip install submodules/simple-knn && log_ok "simple-knn compiled." || {
        log_error "Failed to compile simple-knn. Check CUDA toolkit compatibility."
        exit 1
    }
fi
cd "$SCRIPT_DIR"

# ── Step 8: Install Nerfstudio + gsplat ──────────────────────────────────────
log_info "Installing nerfstudio and gsplat..."
if python -c "import nerfstudio" 2>/dev/null; then
    log_ok "nerfstudio already installed."
else
    pip install nerfstudio==1.1.4
fi
if python -c "import gsplat" 2>/dev/null; then
    log_ok "gsplat already installed."
else
    pip install gsplat==1.3.0
fi

# ── Step 9: Install hloc (SuperPoint + LightGlue) ────────────────────────────
HLOC_DIR="$DEPS_DIR/Hierarchical-Localization"
if [ ! -d "$HLOC_DIR" ]; then
    log_info "Cloning hloc (Hierarchical-Localization)..."
    git clone --recursive https://github.com/cvg/Hierarchical-Localization.git "$HLOC_DIR"
fi
if python -c "from hloc import extract_features" 2>/dev/null; then
    log_ok "hloc already installed."
else
    log_info "Installing hloc..."
    cd "$HLOC_DIR" && pip install -e . && cd "$SCRIPT_DIR"
    log_ok "hloc installed."
fi

# ── Step 10: Install Depth-Anything-V2 ───────────────────────────────────────
DA_DIR="$DEPS_DIR/Depth-Anything-V2"
if [ ! -d "$DA_DIR" ]; then
    log_info "Cloning Depth-Anything-V2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DA_DIR"
fi

# Install Depth-Anything-V2 dependencies
pip install transformers==4.40.2 2>/dev/null || true

# Download ViTb checkpoint
CKPT_DIR="$DA_DIR/checkpoints"
mkdir -p "$CKPT_DIR"
VITB_CKPT="$CKPT_DIR/depth_anything_v2_vitb.pth"
if [ -f "$VITB_CKPT" ]; then
    log_ok "Depth-Anything-V2 ViTb checkpoint already downloaded."
else
    log_info "Downloading Depth-Anything-V2 ViTb checkpoint (~376MB)..."
    wget -q --show-progress \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth" \
        -O "$VITB_CKPT" && log_ok "ViTb checkpoint downloaded." || {
        log_warn "Direct download failed. Trying alternative..."
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Base',
    filename='depth_anything_v2_vitb.pth',
    local_dir='$CKPT_DIR'
)
print('Downloaded via huggingface_hub.')
" && log_ok "ViTb checkpoint downloaded via huggingface_hub." || {
            log_error "Failed to download Depth-Anything-V2 checkpoint."
            log_error "Manually download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Base"
            log_error "Save to: $VITB_CKPT"
        }
    }
fi

# ── Step 11: Download COLMAP vocabulary tree ──────────────────────────────────
VOCAB_DIR="$DEPS_DIR/vocab_trees"
mkdir -p "$VOCAB_DIR"
VOCAB_FILE="$VOCAB_DIR/vocab_tree_flickr100K_words1M.bin"
if [ -f "$VOCAB_FILE" ]; then
    log_ok "COLMAP vocabulary tree already downloaded."
else
    log_info "Downloading COLMAP vocabulary tree 1M words (~430MB)..."
    wget -q --show-progress \
        "https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin" \
        -O "$VOCAB_FILE" && log_ok "Vocabulary tree downloaded." || {
        log_warn "Failed to download vocab tree. Loop detection will be disabled."
        log_warn "Manually download from: https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin"
    }
fi

# ── Step 12: Install rembg and Open3D ────────────────────────────────────────
log_info "Installing rembg (background removal) and Open3D..."
if python -c "import rembg" 2>/dev/null; then
    log_ok "rembg already installed."
else
    pip install rembg[gpu]==2.0.56 && log_ok "rembg installed." || {
        log_warn "rembg[gpu] failed, trying cpu-only..."
        pip install rembg==2.0.56
        log_ok "rembg (CPU) installed."
    }
fi

if python -c "import open3d" 2>/dev/null; then
    log_ok "Open3D already installed."
else
    pip install open3d==0.18.0 && log_ok "Open3D installed."
fi

# ── Step 13: Install ffmpeg and imagemagick via conda ─────────────────────────
log_info "Installing ffmpeg and imagemagick..."
if command -v ffmpeg &>/dev/null; then
    log_ok "ffmpeg already available: $(ffmpeg -version 2>&1 | head -1)"
else
    conda install -c conda-forge ffmpeg -y && log_ok "ffmpeg installed."
fi
if command -v convert &>/dev/null; then
    log_ok "imagemagick already available."
else
    conda install -c conda-forge imagemagick -y && log_ok "imagemagick installed."
fi

# ── Step 14: Install huggingface_hub for model downloads ─────────────────────
pip install huggingface_hub==0.22.2 2>/dev/null || true

# ── Step 15: Final verification ───────────────────────────────────────────────
log_info "========================================"
log_info " Running verification..."
log_info "========================================"
python "$SCRIPT_DIR/test_installation.py"

echo ""
log_ok "========================================"
log_ok " Setup complete!"
log_ok " Activate environment: conda activate $ENV_NAME"
log_ok " Run pipeline: ./run_pipeline.sh --help"
log_ok "========================================"
