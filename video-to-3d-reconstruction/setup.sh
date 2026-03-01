#!/usr/bin/env bash
# ============================================================
# setup.sh — Ubuntu 22.04 full environment setup
# Video → 3D/2D Reconstruction (Nerfstudio + Splatfacto)
#
# What this script does:
#   1. Installs all system packages (ffmpeg, colmap, build tools)
#   2. Creates a Python 3.10 virtual environment in ./venv/
#   3. Installs ALL Python dependencies into the venv (offline-capable after first run)
#   4. Installs Node.js 20 + npm deps for backend and frontend
#   5. Detects and reports NVIDIA GPUs
#   6. Prints simple run instructions
#
# USAGE:
#   chmod +x setup.sh
#   bash setup.sh
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[SETUP]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${BOLD}"
cat << 'BANNER'
  ____  ____    ____                      _                   _   _
 |___ \|  _ \  |  _ \ ___  ___ ___  _ __| |_ _ __ _   _  ___| |_(_) ___  _ __
   __) | | | | | |_) / _ \/ __/ _ \| '__| __| '__| | | |/ __| __| |/ _ \| '_ \
  / __/| |_| | |  _ <  __/ (_| (_) | |  | |_| |  | |_| | (__| |_| | (_) | | | |
 |_____|____/  |_| \_\___|\___\___/|_|   \__|_|   \__,_|\___|\__|_|\___/|_| |_|
BANNER
echo -e "${NC}"
echo -e "  Ubuntu 22.04 | Nerfstudio Splatfacto | React + Node.js | Multi-GPU"
echo -e "  ======================================================================="
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] && err "Run as your normal user (not root). sudo will be invoked when needed."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log "Project root: $SCRIPT_DIR"

# ── 1. OS check ───────────────────────────────────────────────────────────────
log "Checking operating system..."
. /etc/os-release 2>/dev/null || true
[[ "${ID:-}" != "ubuntu" ]] && warn "Not Ubuntu (detected '${ID:-unknown}'). Proceeding anyway."
ok "OS: ${PRETTY_NAME:-Linux}"

# ── 2. GPU detection ──────────────────────────────────────────────────────────
log "Detecting NVIDIA GPUs..."
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAMES=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    ok "Found ${GPU_COUNT} GPU(s):"
    while IFS= read -r line; do
        echo -e "     ${GREEN}•${NC} $line"
    done <<< "$GPU_NAMES"
    if [[ $GPU_COUNT -ge 4 ]]; then
        ok "4+ GPUs detected — multi-GPU training will be used automatically."
    fi
else
    warn "nvidia-smi not found. GPU training requires NVIDIA drivers."
    warn "Install drivers first: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
fi

# ── 3. System packages ────────────────────────────────────────────────────────
log "Installing system packages (requires sudo)..."
sudo apt-get update -qq
sudo apt-get install -y \
    ffmpeg \
    colmap \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libboost-all-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev 2>/dev/null || true

ok "System packages installed"

# Verify critical tools
for tool in ffmpeg colmap python3.10; do
    command -v $tool &>/dev/null && ok "$tool: $(command -v $tool)" || warn "$tool not found after install"
done

# ── 4. Python virtual environment ─────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
log "Setting up Python 3.10 virtual environment at: $VENV_DIR"

if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    warn "venv already exists — skipping creation (delete $VENV_DIR to recreate)"
else
    python3.10 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
ok "venv activated: $(which python) — $(python --version)"

# Upgrade pip/setuptools/wheel inside venv
pip install --quiet --upgrade pip setuptools wheel
ok "pip upgraded: $(pip --version)"

# ── 5. PyTorch with CUDA 11.8 ─────────────────────────────────────────────────
log "Installing PyTorch 2.1.2 + CUDA 11.8 (this may take a few minutes)..."
pip install --quiet \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python - <<'PYCHECK'
import torch
print(f"  PyTorch:    {torch.__version__}")
print(f"  CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"  GPU count:  {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
PYCHECK
ok "PyTorch installed"

# ── 6. All Python requirements ────────────────────────────────────────────────
log "Installing Python requirements (nerfstudio + all deps)..."
log "This installs everything into the venv — no internet needed after this step."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
ok "Python requirements installed"

# ── 7. Verify Nerfstudio ──────────────────────────────────────────────────────
log "Verifying Nerfstudio..."
if ! command -v ns-train &>/dev/null; then
    err "ns-train not found in venv. Check requirements.txt installation above."
fi
ok "Nerfstudio CLI: $(ns-train --version 2>/dev/null || echo 'installed')"

# ── 8. Node.js 20 ─────────────────────────────────────────────────────────────
log "Checking Node.js..."
NEED_NODE=false
if ! command -v node &>/dev/null; then
    NEED_NODE=true
else
    NODE_MAJOR=$(node -e "process.stdout.write(process.version.slice(1).split('.')[0])")
    [[ $NODE_MAJOR -lt 18 ]] && NEED_NODE=true
fi

if $NEED_NODE; then
    log "Installing Node.js 20 via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    ok "Node.js $(node --version) installed"
else
    ok "Node.js $(node --version) already installed"
fi

# ── 9. npm dependencies ────────────────────────────────────────────────────────
log "Installing backend npm packages..."
cd "$SCRIPT_DIR/backend" && npm install
ok "Backend npm packages installed"

log "Installing frontend npm packages..."
cd "$SCRIPT_DIR/frontend" && npm install
ok "Frontend npm packages installed"

cd "$SCRIPT_DIR"

# ── 10. Create required runtime directories ────────────────────────────────────
mkdir -p "$SCRIPT_DIR/uploads" "$SCRIPT_DIR/outputs"
ok "uploads/ and outputs/ directories ready"

# ── 11. Write activation helper script ────────────────────────────────────────
cat > "$SCRIPT_DIR/activate.sh" << 'EOF'
#!/usr/bin/env bash
# Quick helper: activates the venv in the current shell session.
# Usage: source activate.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo "venv activated. Python: $(which python)"
echo "ns-train: $(which ns-train 2>/dev/null || echo 'not found')"
EOF
chmod +x "$SCRIPT_DIR/activate.sh"
ok "activate.sh helper written"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}======================================================================="
echo -e " SETUP COMPLETE — Everything is installed in ./venv/"
echo -e "=======================================================================${NC}"
echo ""
echo -e " ${BOLD}To run the application:${NC}"
echo ""
echo -e "   ${YELLOW}# Step 1 — Activate the Python virtual environment${NC}"
echo -e "   source activate.sh"
echo ""
echo -e "   ${YELLOW}# Step 2 — Start the backend (new terminal)${NC}"
echo -e "   cd $SCRIPT_DIR/backend && node server.js"
echo ""
echo -e "   ${YELLOW}# Step 3 — Start the frontend (another terminal)${NC}"
echo -e "   cd $SCRIPT_DIR/frontend && npm run dev"
echo ""
echo -e "   ${YELLOW}# Step 4 — Open in browser${NC}"
echo -e "   ${CYAN}http://localhost:5173${NC}"
echo ""
echo -e " ${BOLD}GPU Info:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "   GPU %s: %s (%s)\n", $1, $2, $3}' || echo "   (no nvidia-smi)"
echo ""
