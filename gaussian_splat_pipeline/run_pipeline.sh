#!/bin/bash
# =============================================================================
# Gaussian Splatting Pipeline — Master Orchestrator
# Runs all 8 stages end-to-end with error checking and resume support
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="gsplat_env"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[pipeline]${NC} $*" | tee -a "$LOG_FILE" 2>/dev/null || echo -e "${BLUE}[pipeline]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[  OK  ]${NC} $*"  | tee -a "$LOG_FILE" 2>/dev/null || echo -e "${GREEN}[  OK  ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[ WARN ]${NC} $*" | tee -a "$LOG_FILE" 2>/dev/null || echo -e "${YELLOW}[ WARN ]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR ]${NC} $*"   | tee -a "$LOG_FILE" 2>/dev/null || echo -e "${RED}[ERROR ]${NC} $*"; }
log_stage() { echo -e "\n${CYAN}════════════════════════════════════════${NC}" | tee -a "$LOG_FILE" 2>/dev/null; echo -e "${CYAN} $*${NC}" | tee -a "$LOG_FILE" 2>/dev/null; echo -e "${CYAN}════════════════════════════════════════${NC}" | tee -a "$LOG_FILE" 2>/dev/null; }

usage() {
    cat <<EOF
Usage: $0 --input VIDEO --output OUTPUT_DIR [OPTIONS]

Required:
  --input PATH          Path to input video file
  --output PATH         Path to output directory

Optional:
  --fps N               Frames per second to extract (default: 3)
  --iterations N        Training iterations (default: 60000)
  --method METHOD       Training method: original|splatfacto|both (default: original)
  --matcher MATCHER     Feature matcher: superpoint|sift (default: superpoint)
  --depth-regularization BOOL   Enable depth regularization (default: true)
  --exposure-compensation BOOL  Enable exposure compensation (default: true)
  --antialiasing BOOL   Enable EWA antialiasing (default: true)
  --mask BOOL           Enable background masking (default: false)
  --num-gpus N          Number of GPUs to use (default: 4, auto-detect)
  --resume BOOL         Resume from last completed stage (default: false)
  --dry-run BOOL        Print commands without executing (default: false)
  --config PATH         Config file (default: configs/default.yaml)
  --skip-dense BOOL     Skip dense COLMAP reconstruction (default: false)

Examples:
  $0 --input /data/room.mp4 --output /data/room_reconstruction
  $0 --input /data/room.mp4 --output /data/out --config configs/quality_max.yaml
  $0 --input /data/room.mp4 --output /data/out --resume true
  $0 --input /data/room.mp4 --output /data/out --dry-run true
EOF
    exit 0
}

# ── Default values ────────────────────────────────────────────────────────────
INPUT_VIDEO=""
OUTPUT_DIR=""
FPS="3"
ITERATIONS="60000"
METHOD="original"
MATCHER="superpoint"
DEPTH_REG="true"
EXPOSURE_COMP="true"
ANTIALIASING="true"
MASK="false"
NUM_GPUS="4"
RESUME="false"
DRY_RUN="false"
SKIP_DENSE="false"
CONFIG_FILE="$SCRIPT_DIR/configs/default.yaml"
LOG_FILE="/dev/stderr"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)               INPUT_VIDEO="$2"; shift 2 ;;
        --output)              OUTPUT_DIR="$2"; shift 2 ;;
        --fps)                 FPS="$2"; shift 2 ;;
        --iterations)          ITERATIONS="$2"; shift 2 ;;
        --method)              METHOD="$2"; shift 2 ;;
        --matcher)             MATCHER="$2"; shift 2 ;;
        --depth-regularization) DEPTH_REG="$2"; shift 2 ;;
        --exposure-compensation) EXPOSURE_COMP="$2"; shift 2 ;;
        --antialiasing)        ANTIALIASING="$2"; shift 2 ;;
        --mask)                MASK="$2"; shift 2 ;;
        --num-gpus)            NUM_GPUS="$2"; shift 2 ;;
        --resume)              RESUME="$2"; shift 2 ;;
        --dry-run)             DRY_RUN="$2"; shift 2 ;;
        --skip-dense)          SKIP_DENSE="$2"; shift 2 ;;
        --config)              CONFIG_FILE="$2"; shift 2 ;;
        --help|-h)             usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[ -z "$INPUT_VIDEO" ] && { echo "ERROR: --input is required"; usage; }
[ -z "$OUTPUT_DIR"  ] && { echo "ERROR: --output is required"; usage; }
[ ! -f "$INPUT_VIDEO" ] && { echo "ERROR: Input video not found: $INPUT_VIDEO"; exit 1; }

# ── Setup output dir and log ──────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/pipeline.log"
STAGE_DIR="$OUTPUT_DIR"
PIPELINE_START=$(date +%s)

log_info "Gaussian Splatting Pipeline — $(date)"
log_info "Input:   $INPUT_VIDEO"
log_info "Output:  $OUTPUT_DIR"
log_info "Method:  $METHOD | Matcher: $MATCHER | GPUs: $NUM_GPUS"
log_info "Dry run: $DRY_RUN | Resume: $RESUME"

# ── Activate conda environment ────────────────────────────────────────────────
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || {
    log_error "Cannot activate conda env '$ENV_NAME'. Run ./setup.sh first."
    exit 1
}
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/scripts:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$(seq -s',' 0 $((NUM_GPUS - 1)))"
export PYTHONHASHSEED=0

PYTHON="python"
SCRIPTS="$SCRIPT_DIR/scripts"
DEPS="$SCRIPT_DIR/deps"

# ── Helper: run or dry-run a stage ───────────────────────────────────────────
run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    shift 2
    local marker="$STAGE_DIR/.stage${stage_num}_complete"

    log_stage "Stage $stage_num: $stage_name"

    if [ "$RESUME" = "true" ] && [ -f "$marker" ]; then
        log_ok "Stage $stage_num already complete (found $marker). Skipping."
        return 0
    fi

    local stage_start=$(date +%s)

    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would execute: $*"
        return 0
    fi

    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code="${PIPESTATUS[0]}"

    if [ "$exit_code" -ne 0 ]; then
        log_error "Stage $stage_num FAILED (exit code $exit_code)"
        log_error "Check logs: $LOG_FILE"
        local diag="$OUTPUT_DIR/logs/stage${stage_num}_diagnostic.txt"
        echo "Stage $stage_num failed at $(date)" > "$diag"
        echo "Command: $*" >> "$diag"
        echo "See pipeline.log for details." >> "$diag"
        exit "$exit_code"
    fi

    local stage_end=$(date +%s)
    local elapsed=$((stage_end - stage_start))
    echo "$stage_num:$elapsed" >> "$OUTPUT_DIR/logs/stage_timings.json" 2>/dev/null || true
    touch "$marker"
    log_ok "Stage $stage_num complete in ${elapsed}s"
}

# ── Build common Python args ──────────────────────────────────────────────────
COMMON_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --config "$CONFIG_FILE"
    --num-gpus "$NUM_GPUS"
)

# ── Stage 0: Pre-flight ───────────────────────────────────────────────────────
run_stage 0 "Pre-flight Checks" \
    "$PYTHON" "$SCRIPTS/stage0_preflight.py" \
    --input "$INPUT_VIDEO" \
    "${COMMON_ARGS[@]}"

# ── Stage 1: Frame Extraction & Quality Filtering ─────────────────────────────
run_stage 1 "Frame Extraction & Quality Filtering" \
    "$PYTHON" "$SCRIPTS/stage1_extract_filter.py" \
    --input "$INPUT_VIDEO" \
    --fps "$FPS" \
    "${COMMON_ARGS[@]}"

# ── Stage 2: Depth Map Generation ────────────────────────────────────────────
if [ "$DEPTH_REG" = "true" ]; then
    run_stage 2 "Depth Map Generation (Depth-Anything-V2)" \
        "$PYTHON" "$SCRIPTS/stage2_depth.py" \
        --deps-dir "$DEPS" \
        "${COMMON_ARGS[@]}"
else
    log_warn "Depth regularization disabled. Skipping Stage 2."
    touch "$STAGE_DIR/.stage2_complete"
fi

# ── Stage 3: Background Masking (optional) ────────────────────────────────────
if [ "$MASK" = "true" ]; then
    run_stage 3 "Background Masking (rembg)" \
        "$PYTHON" "$SCRIPTS/stage3_mask.py" \
        "${COMMON_ARGS[@]}"
else
    log_info "Masking disabled. Skipping Stage 3."
    touch "$STAGE_DIR/.stage3_complete"
fi

# ── Stage 4: COLMAP Pose Estimation ──────────────────────────────────────────
run_stage 4 "COLMAP Pose Estimation ($MATCHER)" \
    "$PYTHON" "$SCRIPTS/stage4_colmap.py" \
    --matcher "$MATCHER" \
    --deps-dir "$DEPS" \
    "${COMMON_ARGS[@]}"

# ── Stage 5: Gaussian Splatting Training ─────────────────────────────────────
TRAIN_EXTRA_ARGS=()
[ "$DEPTH_REG"    = "true" ] && TRAIN_EXTRA_ARGS+=(--depth-regularization true)
[ "$EXPOSURE_COMP" = "true" ] && TRAIN_EXTRA_ARGS+=(--exposure-compensation true)
[ "$ANTIALIASING" = "true" ] && TRAIN_EXTRA_ARGS+=(--antialiasing true)

run_stage 5 "Gaussian Splatting Training ($METHOD, $ITERATIONS iters)" \
    "$PYTHON" "$SCRIPTS/stage5_train.py" \
    --method "$METHOD" \
    --iterations "$ITERATIONS" \
    --deps-dir "$DEPS" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    "${COMMON_ARGS[@]}"

# ── Stage 6: Quality Evaluation ──────────────────────────────────────────────
run_stage 6 "Quality Evaluation (PSNR/SSIM/LPIPS)" \
    "$PYTHON" "$SCRIPTS/stage6_evaluate.py" \
    --method "$METHOD" \
    --deps-dir "$DEPS" \
    "${COMMON_ARGS[@]}"

# ── Stage 7: Export ───────────────────────────────────────────────────────────
EXPORT_EXTRA=()
[ "$SKIP_DENSE" = "true" ] && EXPORT_EXTRA+=(--skip-dense true)

run_stage 7 "Export (PLY, Dense Cloud, Mesh)" \
    "$PYTHON" "$SCRIPTS/stage7_export.py" \
    --method "$METHOD" \
    "${EXPORT_EXTRA[@]}" \
    "${COMMON_ARGS[@]}"

# ── Final Summary ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN} PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo "  Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "  Key outputs:"
echo "    Gaussian PLY:    $OUTPUT_DIR/exports/gaussians.ply"
echo "    Dense cloud:     $OUTPUT_DIR/exports/dense_pointcloud.ply"
echo "    Mesh:            $OUTPUT_DIR/exports/mesh.ply"
echo "    Quality report:  $OUTPUT_DIR/quality_report/metrics_summary.json"
echo "    Full report:     $OUTPUT_DIR/reconstruction_report.txt"
echo ""

# Print metrics if available
METRICS="$OUTPUT_DIR/quality_report/metrics_summary.json"
if [ -f "$METRICS" ]; then
    echo "  Quality Metrics:"
    python -c "
import json
m = json.load(open('$METRICS'))
print(f\"    PSNR:  {m.get('psnr_mean', 'N/A'):.2f} dB (std: {m.get('psnr_std', 'N/A'):.2f})\")
print(f\"    SSIM:  {m.get('ssim_mean', 'N/A'):.4f} (std: {m.get('ssim_std', 'N/A'):.4f})\")
print(f\"    LPIPS: {m.get('lpips_mean', 'N/A'):.4f} (std: {m.get('lpips_std', 'N/A'):.4f})\")
print(f\"    Gaussians: {m.get('num_gaussians', 'N/A'):,}\")
" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
