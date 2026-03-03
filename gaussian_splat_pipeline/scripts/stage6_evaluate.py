#!/usr/bin/env python3
"""
Stage 6: Quality Evaluation (PSNR / SSIM / LPIPS)
- Renders test views from trained model
- Computes PSNR, SSIM, LPIPS per test image and in aggregate
- Generates side-by-side comparison grid (GT | Rendered | Error)
- Tracks variance and flags high-variance reconstructions
- Saves metrics_summary.json
"""

import sys
import os
import json
import glob
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger        import get_pipeline_logger
from utils.seed_utils    import set_all_seeds
from utils.config_loader import load_config
from utils.metrics       import compute_metrics_batch, generate_comparison_grid
from utils.gpu_utils     import detect_gpus


def parse_args():
    p = argparse.ArgumentParser(description="Stage 6: Quality evaluation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deps-dir",   required=True)
    p.add_argument("--method",     default="original", choices=["original","splatfacto","both"])
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def render_original_3dgs(model_dir: Path, deps_dir: Path, training_gpu: int, logger) -> Path:
    """
    Run the original 3DGS render.py to generate test-view renders.
    Returns path to renders directory.
    """
    gs_dir = deps_dir / "gaussian-splatting"
    render_script = gs_dir / "render.py"
    render_dir = model_dir / "renders"

    if not render_script.exists():
        logger.warning(f"render.py not found at {render_script}")
        return render_dir

    env = {"CUDA_VISIBLE_DEVICES": str(training_gpu), "PYTHONHASHSEED": "0",
           "CC": "gcc-11", "CXX": "g++-11"}

    # Find latest checkpoint iteration
    point_clouds = sorted(model_dir.rglob("point_cloud.ply"))
    if not point_clouds:
        logger.warning("No checkpoint found. Skipping render.")
        return render_dir

    proc = subprocess.run([
        "python", str(render_script),
        "-m", str(model_dir),
        "--skip_train",
        "--quiet",
    ], capture_output=True, text=True,
    env={**os.environ, **env},
    cwd=str(gs_dir))

    if proc.returncode != 0:
        logger.warning(f"Render failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
    else:
        logger.info("  Renders generated successfully.")

    return render_dir


def find_test_images(model_dir: Path, method: str) -> tuple:
    """
    Find rendered test images and corresponding ground-truth images.
    Returns (pred_paths, gt_paths)
    """
    pred_paths = []
    gt_paths   = []

    if method == "original":
        # Original 3DGS writes to: model_dir/test/ours_{N}/renders/ and gt/
        test_dirs = sorted((model_dir / "original").rglob("ours_*"))
        for d in test_dirs:
            renders = sorted((d / "renders").glob("*.png")) + sorted((d / "renders").glob("*.jpg"))
            gts     = sorted((d / "gt").glob("*.png"))     + sorted((d / "gt").glob("*.jpg"))
            # Match by filename
            render_names = {p.stem: p for p in renders}
            for gt_p in gts:
                if gt_p.stem in render_names:
                    pred_paths.append(str(render_names[gt_p.stem]))
                    gt_paths.append(str(gt_p))

    elif method == "splatfacto":
        # Nerfstudio outputs to: model_dir/splatfacto/.../renders/
        render_dirs = sorted((model_dir / "splatfacto").rglob("renders"))
        for rd in render_dirs:
            renders = sorted(rd.glob("*.png")) + sorted(rd.glob("*.jpg"))
            pred_paths.extend([str(p) for p in renders])
            # Nerfstudio stores GT alongside renders as gt_*.png
            for p in renders:
                gt_p = p.parent / f"gt_{p.name}"
                if gt_p.exists():
                    gt_paths.append(str(gt_p))

    return pred_paths, gt_paths


def evaluate_method(
    model_dir: Path,
    method: str,
    deps_dir: Path,
    cfg: dict,
    output_dir: Path,
    training_gpu: int,
    logger,
) -> Optional[dict]:
    """Evaluate a single training method. Returns metrics dict."""
    eval_cfg = cfg.get("evaluation", {})
    compute_lpips = eval_cfg.get("compute_lpips", True)
    gen_grid      = eval_cfg.get("generate_comparison_grid", True)
    var_thresh    = eval_cfg.get("variance_psnr_threshold", 3.0)

    quality_dir = output_dir / "quality_report"
    quality_dir.mkdir(parents=True, exist_ok=True)

    method_model_dir = model_dir / method

    # ── Render test views ─────────────────────────────────────────────────────
    if method == "original":
        render_original_3dgs(method_model_dir, deps_dir, training_gpu, logger)
    elif method == "splatfacto":
        logger.info("  Nerfstudio renders test images during training.")

    # ── Find paired images ────────────────────────────────────────────────────
    pred_paths, gt_paths = find_test_images(model_dir, method)
    logger.info(f"  Found {len(pred_paths)} rendered / {len(gt_paths)} GT image pairs")

    if not pred_paths or not gt_paths:
        logger.warning(f"  No test images found for method='{method}'. "
                       f"Ensure --eval flag was set during training.")
        return None

    # ── Compute metrics ───────────────────────────────────────────────────────
    logger.info("  Computing PSNR / SSIM / LPIPS...")
    try:
        import torch
        device = f"cuda:{training_gpu}" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    per_image, agg = compute_metrics_batch(
        pred_paths, gt_paths,
        compute_lpips=compute_lpips,
        device=device,
    )

    if not agg:
        logger.warning("  Could not compute metrics.")
        return None

    # ── Log results ───────────────────────────────────────────────────────────
    psnr_mean  = agg.get("psnr_mean", 0)
    psnr_std   = agg.get("psnr_std", 0)
    ssim_mean  = agg.get("ssim_mean", 0)
    lpips_mean = agg.get("lpips_mean", -1)

    logger.info(f"  PSNR:  {psnr_mean:.2f} dB ± {psnr_std:.2f}")
    logger.info(f"  SSIM:  {ssim_mean:.4f}")
    logger.info(f"  LPIPS: {lpips_mean:.4f}" if lpips_mean >= 0 else "  LPIPS: N/A")

    # Quality assessment
    if psnr_mean >= 35:
        logger.info("  Quality: EXCELLENT (PSNR ≥ 35 dB)")
    elif psnr_mean >= 30:
        logger.info("  Quality: GOOD (PSNR ≥ 30 dB)")
    elif psnr_mean >= 25:
        logger.warning("  Quality: MODERATE (PSNR < 30 dB). Consider more training iterations.")
    else:
        logger.warning("  Quality: POOR (PSNR < 25 dB). Check COLMAP poses and depth maps.")

    # Variance check
    if psnr_std > var_thresh:
        logger.warning(
            f"  HIGH VARIANCE: PSNR std={psnr_std:.2f} > threshold={var_thresh}. "
            f"Some views are much worse than others. Possible causes: "
            f"few training frames for certain views, bad COLMAP poses."
        )

    # ── Save per-image CSV ────────────────────────────────────────────────────
    import csv
    csv_path = quality_dir / f"per_image_metrics_{method}.csv"
    if per_image:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename","psnr","ssim","lpips"])
            writer.writeheader()
            writer.writerows(per_image)
        logger.info(f"  Per-image metrics: {csv_path}")

    # ── Generate comparison grid ──────────────────────────────────────────────
    if gen_grid and pred_paths:
        grid_path = quality_dir / f"comparison_grid_{method}.png"
        try:
            generate_comparison_grid(
                pred_paths[:8], gt_paths[:8],
                str(grid_path),
                max_images=8, cols=4
            )
            logger.info(f"  Comparison grid: {grid_path}")
        except Exception as e:
            logger.warning(f"  Comparison grid failed: {e}")

    # ── Load training results for num_gaussians ───────────────────────────────
    num_gaussians = 0
    training_time = 0
    train_results_path = output_dir / "logs" / "training_results.json"
    if train_results_path.exists():
        with open(train_results_path) as f:
            tr = json.load(f)
        for r in tr:
            if r.get("method") == method:
                num_gaussians = r.get("num_gaussians", 0)
                training_time = r.get("training_time_seconds", 0)

    return {
        **agg,
        "method": method,
        "num_gaussians": num_gaussians,
        "training_time_seconds": training_time,
        "variance_status": "high" if psnr_std > var_thresh else "low",
        "psnr_quality": (
            "excellent" if psnr_mean >= 35 else
            "good"      if psnr_mean >= 30 else
            "moderate"  if psnr_mean >= 25 else "poor"
        ),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    deps_dir   = Path(args.deps_dir)
    cfg        = load_config(args.config)
    seed       = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage6")
    logger.info("=" * 60)
    logger.info("Stage 6: Quality Evaluation")
    logger.info("=" * 60)

    gpus = detect_gpus()
    training_gpu = cfg.get("gpu", {}).get("training_gpu", 0)

    model_dir    = output_dir / "model"
    quality_dir  = output_dir / "quality_report"
    quality_dir.mkdir(parents=True, exist_ok=True)

    methods = ["original", "splatfacto"] if args.method == "both" else [args.method]
    all_metrics = {}

    for method in methods:
        logger.info(f"\n  Evaluating method: {method}")
        metrics = evaluate_method(
            model_dir, method, deps_dir, cfg, output_dir, training_gpu, logger
        )
        if metrics:
            all_metrics[method] = metrics

    if not all_metrics:
        logger.warning("No metrics computed. Training may not have run with --eval flag.")
        # Create empty summary so downstream stages don't fail
        summary = {"status": "no_metrics", "methods": methods}
    else:
        # If "both", pick the better method
        best_method = None
        best_psnr   = -1
        for method, m in all_metrics.items():
            if m.get("psnr_mean", 0) > best_psnr:
                best_psnr   = m.get("psnr_mean", 0)
                best_method = method

        summary = {
            "best_method": best_method,
            **all_metrics.get(best_method or methods[0], {}),
            "all_methods": all_metrics,
        }
        if len(all_metrics) > 1:
            logger.info(f"\n  Best method: {best_method} (PSNR {best_psnr:.2f} dB)")

    # Save summary
    summary_path = quality_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Summary saved: {summary_path}")

    logger.info("=" * 60)
    logger.info("Stage 6 COMPLETE.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
