#!/usr/bin/env python3
"""
Stage 5: Gaussian Splatting Training
- Method A: Original 3DGS (Inria) — highest accuracy reference
- Method B: Splatfacto (Nerfstudio) — alternative
- Method C: Both — trains both and compares

Key accuracy features:
- Depth regularization (Depth-Anything-V2 maps via -d flag)
- EWA anti-aliasing (Mip-Splatting)
- Per-image exposure compensation
- All training on GPU 0 (single-GPU for stability)
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger        import get_pipeline_logger
from utils.seed_utils    import set_all_seeds
from utils.config_loader import load_config
from utils.gpu_utils     import detect_gpus, get_best_training_gpu


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5: Gaussian Splatting training")
    p.add_argument("--output-dir",            required=True)
    p.add_argument("--deps-dir",              required=True)
    p.add_argument("--method",                default="original", choices=["original","splatfacto","both"])
    p.add_argument("--iterations",            type=int, default=60000)
    p.add_argument("--depth-regularization",  default="true")
    p.add_argument("--exposure-compensation", default="true")
    p.add_argument("--antialiasing",          default="true")
    p.add_argument("--config",                default="configs/default.yaml")
    p.add_argument("--num-gpus",              type=int, default=4)
    return p.parse_args()


def bool_arg(val) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def run_cmd(cmd: list, logger, env: dict = None, cwd: str = None) -> int:
    """Run a command with live output streaming."""
    logger.info(f"  CMD: {' '.join(str(c) for c in cmd[:8])} ...")
    combined_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=combined_env, cwd=cwd
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"    {line}")
    proc.wait()
    return proc.returncode


def train_original_3dgs(
    data_dir: Path,
    model_dir: Path,
    depth_dir: Path,
    cfg: dict,
    deps_dir: Path,
    training_gpu: int,
    logger,
    iterations: int,
    depth_reg: bool,
    exposure_comp: bool,
    antialiasing: bool,
):
    """Train using the original Inria 3DGS repository."""
    gs_dir = deps_dir / "gaussian-splatting"
    train_script = gs_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Original 3DGS train.py not found at {train_script}. "
            f"Run ./setup.sh to clone the repo."
        )

    train_cfg = cfg.get("training", {})
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(train_script),
        "-s", str(data_dir),
        "-m", str(model_dir),
        "--iterations",              str(iterations),
        "--densify_until_iter",      str(train_cfg.get("densify_until_iter", 25000)),
        "--densification_interval",  str(train_cfg.get("densification_interval", 100)),
        "--opacity_reset_interval",  str(train_cfg.get("opacity_reset_interval", 5000)),
        "--densify_grad_threshold",  str(train_cfg.get("densify_grad_threshold", 0.0002)),
        "--sh_degree",               str(train_cfg.get("sh_degree", 3)),
        "--position_lr_init",        str(train_cfg.get("position_lr_init", 0.00016)),
        "--position_lr_final",       str(train_cfg.get("position_lr_final", 0.0000016)),
        "--scaling_lr",              str(train_cfg.get("scaling_lr", 0.005)),
        "--percent_dense",           str(train_cfg.get("percent_dense", 0.01)),
        "--lambda_dssim",            str(train_cfg.get("lambda_dssim", 0.2)),
        "--test_iterations",
        *[str(i) for i in train_cfg.get("test_iterations", [7000, 15000, 30000, 45000, 60000])],
        "--save_iterations",
        *[str(i) for i in train_cfg.get("save_iterations", [30000, 60000])],
        "--eval",
    ]

    # Depth regularization (original 3DGS -d flag, Oct 2024 update)
    if depth_reg and depth_dir and depth_dir.exists():
        depth_files = list(depth_dir.glob("*.png"))
        depth_files = [f for f in depth_files if not f.stem.endswith("_vis")]
        if depth_files:
            cmd += ["-d", str(depth_dir)]
            logger.info(f"  Depth regularization: ENABLED ({len(depth_files)} maps)")
        else:
            logger.warning("  Depth regularization: No depth maps found, skipping.")
    else:
        if depth_reg:
            logger.warning(f"  Depth maps dir not found: {depth_dir}. Skipping depth reg.")

    # EWA antialiasing (from Mip-Splatting)
    if antialiasing:
        cmd.append("--antialiasing")
        logger.info("  Antialiasing (EWA): ENABLED")

    # Per-image exposure compensation
    if exposure_comp:
        cmd += [
            "--exposure_lr_init",       str(train_cfg.get("exposure_lr_init", 0.001)),
            "--exposure_lr_final",      str(train_cfg.get("exposure_lr_final", 0.0001)),
            "--exposure_lr_delay_steps",str(train_cfg.get("exposure_lr_delay_steps", 5000)),
            "--exposure_lr_delay_mult", str(train_cfg.get("exposure_lr_delay_mult", 0.001)),
            "--train_test_exp",
        ]
        logger.info("  Exposure compensation: ENABLED")

    env = {
        "CUDA_VISIBLE_DEVICES": str(training_gpu),
        "PYTHONHASHSEED": "0",
    }

    logger.info(f"  Training on GPU {training_gpu}...")
    t_start = time.time()
    rc = run_cmd(cmd, logger, env=env, cwd=str(gs_dir))
    t_elapsed = time.time() - t_start

    if rc != 0:
        raise RuntimeError(f"Original 3DGS training failed (exit {rc})")

    logger.info(f"  Training complete in {t_elapsed/60:.1f} minutes.")

    # Count Gaussians from point_cloud.ply
    ply_candidates = sorted(model_dir.rglob("point_cloud.ply"))
    num_gaussians = 0
    if ply_candidates:
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(ply_candidates[-1]))
            num_gaussians = len(ply.elements[0].data)
        except Exception:
            pass

    return {
        "method": "original",
        "training_time_seconds": round(t_elapsed),
        "num_gaussians": num_gaussians,
        "iterations": iterations,
        "depth_regularization": depth_reg,
        "antialiasing": antialiasing,
        "exposure_compensation": exposure_comp,
    }


def train_splatfacto(
    data_dir: Path,
    model_dir: Path,
    cfg: dict,
    training_gpu: int,
    logger,
    iterations: int,
):
    """Train using Nerfstudio splatfacto."""
    train_cfg  = cfg.get("training", {})
    sf_cfg     = train_cfg
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert undistorted COLMAP data to nerfstudio format
    ns_data_dir = model_dir.parent / "splatfacto_data"
    ns_data_dir.mkdir(parents=True, exist_ok=True)

    # Use ns-process-data to convert
    logger.info("  Converting COLMAP data to Nerfstudio format...")
    rc = run_cmd([
        "ns-process-data", "images",
        "--data", str(data_dir / "images"),
        "--output-dir", str(ns_data_dir),
        "--matching-method", "sequential",
        "--skip-colmap",
        "--colmap-model-path", str(data_dir / "sparse" / "0"),
    ], logger, env={"CUDA_VISIBLE_DEVICES": str(training_gpu)})

    if rc != 0:
        # Fallback: point ns-train directly at undistorted dir
        logger.warning("  ns-process-data failed, using data_dir directly.")
        ns_data_dir = data_dir

    cmd = [
        "ns-train", "splatfacto",
        "--data",                   str(ns_data_dir),
        "--output-dir",             str(model_dir),
        "--max-num-iterations",     str(iterations),
        "--pipeline.model.cull-alpha-thresh",
            str(sf_cfg.get("splatfacto_cull_alpha_thresh", 0.005)),
        "--pipeline.model.continue-cull-post-densification",
            str(sf_cfg.get("splatfacto_continue_cull_post_densification", False)),
        "--pipeline.model.use-scale-regularization",
            str(sf_cfg.get("splatfacto_use_scale_regularization", True)),
        "--pipeline.model.sh-degree",
            str(sf_cfg.get("sh_degree", 3)),
        "--pipeline.model.densify-grad-thresh",
            str(sf_cfg.get("densify_grad_threshold", 0.0002)),
    ]

    env = {
        "CUDA_VISIBLE_DEVICES": str(training_gpu),
        "PYTHONHASHSEED": "0",
    }

    t_start = time.time()
    rc = run_cmd(cmd, logger, env=env)
    t_elapsed = time.time() - t_start

    if rc != 0:
        raise RuntimeError(f"Splatfacto training failed (exit {rc})")

    logger.info(f"  Splatfacto training complete in {t_elapsed/60:.1f} minutes.")
    return {
        "method": "splatfacto",
        "training_time_seconds": round(t_elapsed),
        "iterations": iterations,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    deps_dir   = Path(args.deps_dir)
    cfg        = load_config(args.config)
    seed       = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage5")
    logger.info("=" * 60)
    logger.info(f"Stage 5: Gaussian Splatting Training (method={args.method})")
    logger.info("=" * 60)

    # Detect best GPU for training
    gpus         = detect_gpus()
    training_gpu = cfg.get("gpu", {}).get("training_gpu", 0)
    if gpus:
        training_gpu = get_best_training_gpu(gpus)
    logger.info(f"  Training GPU: {training_gpu}")
    if gpus and training_gpu < len(gpus):
        logger.info(f"  GPU: {gpus[training_gpu]['name']} ({gpus[training_gpu]['vram_gb']} GB)")

    # Load COLMAP meta
    colmap_meta_path = output_dir / "logs" / "colmap_meta.json"
    if colmap_meta_path.exists():
        with open(colmap_meta_path) as f:
            colmap_meta = json.load(f)
        undist_dir = Path(colmap_meta.get("undistorted_dir", output_dir / "colmap" / "undistorted"))
    else:
        undist_dir = output_dir / "colmap" / "undistorted"
        logger.warning(f"COLMAP meta not found. Using default: {undist_dir}")

    depth_dir   = output_dir / "depth_maps"
    model_dir   = output_dir / "model"
    iterations  = args.iterations or cfg.get("training", {}).get("iterations", 60000)
    depth_reg   = bool_arg(args.depth_regularization)
    exp_comp    = bool_arg(args.exposure_compensation)
    antialiasing= bool_arg(args.antialiasing)
    method      = args.method

    all_results = []

    # ── Method A: Original 3DGS ────────────────────────────────────────────
    if method in ("original", "both"):
        logger.info("  → Training: Original 3DGS (Inria)")
        orig_model_dir = model_dir / "original"
        try:
            result = train_original_3dgs(
                data_dir     = undist_dir,
                model_dir    = orig_model_dir,
                depth_dir    = depth_dir,
                cfg          = cfg,
                deps_dir     = deps_dir,
                training_gpu = training_gpu,
                logger       = logger,
                iterations   = iterations,
                depth_reg    = depth_reg,
                exposure_comp= exp_comp,
                antialiasing = antialiasing,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Original 3DGS training failed: {e}")
            if method == "original":
                sys.exit(1)
            logger.warning("Continuing with splatfacto...")

    # ── Method B: Splatfacto ────────────────────────────────────────────────
    if method in ("splatfacto", "both"):
        logger.info("  → Training: Splatfacto (Nerfstudio)")
        splatfacto_dir = model_dir / "splatfacto"
        try:
            result = train_splatfacto(
                data_dir     = undist_dir,
                model_dir    = splatfacto_dir,
                cfg          = cfg,
                training_gpu = training_gpu,
                logger       = logger,
                iterations   = iterations,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Splatfacto training failed: {e}")
            if method == "splatfacto":
                sys.exit(1)

    # ── Save training results ──────────────────────────────────────────────
    results_path = output_dir / "logs" / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not all_results:
        logger.error("All training methods failed.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Stage 5 COMPLETE.")
    for r in all_results:
        logger.info(f"  Method: {r['method']}, "
                    f"Gaussians: {r.get('num_gaussians', 'N/A'):,}, "
                    f"Time: {r.get('training_time_seconds', 0)//60}m")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
