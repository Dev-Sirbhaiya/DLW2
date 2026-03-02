#!/usr/bin/env python3
"""
Stage 0: Pre-flight Checks
- Validates CUDA availability and prints GPU info
- Validates input video (ffprobe)
- Creates output directory structure
- Saves video/GPU metadata
"""

import sys
import os
import json
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger     import get_pipeline_logger
from utils.gpu_utils  import detect_gpus, print_gpu_table, save_gpu_info, get_driver_cuda_version
from utils.seed_utils import set_all_seeds, save_seeds
from utils.config_loader import load_config, save_config_snapshot


def parse_args():
    p = argparse.ArgumentParser(description="Stage 0: Pre-flight checks")
    p.add_argument("--input",      required=True, help="Path to input video")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def probe_video(video_path: str) -> dict:
    """Run ffprobe to get video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    probe = json.loads(result.stdout)
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise RuntimeError("No video stream found in file")
    return probe, video_stream


def create_output_structure(output_dir: Path):
    """Create all required output subdirectories."""
    subdirs = [
        "frames_raw",
        "frames_filtered",
        "depth_maps",
        "depth_maps/vis",
        "masks",
        "colmap/sparse",
        "colmap/dense",
        "colmap/undistorted",
        "colmap/vocab",
        "colmap/hloc",
        "model/original",
        "model/splatfacto",
        "exports",
        "logs",
        "quality_report",
    ]
    for d in subdirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = get_pipeline_logger(output_dir, "stage0")
    logger.info("=" * 60)
    logger.info("Stage 0: Pre-flight Checks")
    logger.info("=" * 60)

    # Load config
    cfg = load_config(args.config)
    seed = cfg.get("pipeline", {}).get("seed", 42)
    deterministic = cfg.get("pipeline", {}).get("deterministic", True)
    set_all_seeds(seed, deterministic)
    save_seeds(output_dir, seed)
    save_config_snapshot(cfg, output_dir)

    errors = []

    # ── 1. GPU Check ─────────────────────────────────────────────────────────
    logger.info("[1/4] GPU Check")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available in PyTorch. Check CUDA installation.")
        else:
            gpus = detect_gpus()
            driver_ver, cuda_ver = get_driver_cuda_version()
            logger.info(f"  PyTorch version:   {torch.__version__}")
            logger.info(f"  PyTorch CUDA:      {torch.version.cuda}")
            logger.info(f"  Driver version:    {driver_ver}")
            logger.info(f"  CUDA version:      {cuda_ver}")
            logger.info(f"  GPU count:         {len(gpus)}")
            print_gpu_table(gpus, logger)
            save_gpu_info(output_dir, gpus)
    except Exception as e:
        errors.append(f"GPU check failed: {e}")

    # ── 2. Dependency Check ───────────────────────────────────────────────────
    logger.info("[2/4] Dependency Check")
    for cmd, name in [("colmap", "COLMAP"), ("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")]:
        r = subprocess.run(["which", cmd], capture_output=True)
        if r.returncode != 0:
            errors.append(f"{name} not found in PATH. Run ./setup.sh first.")
        else:
            logger.info(f"  {name}: {r.stdout.decode().strip()}")

    # ── 3. Video Validation ───────────────────────────────────────────────────
    logger.info("[3/4] Video Validation")
    video_path = args.input
    if not Path(video_path).exists():
        errors.append(f"Input video not found: {video_path}")
    else:
        try:
            probe, vstream = probe_video(video_path)
            fmtinfo = probe["format"]
            w   = vstream.get("width", "?")
            h   = vstream.get("height", "?")
            fps_str = vstream.get("r_frame_rate", "?")
            dur = float(fmtinfo.get("duration", 0))
            codec = vstream.get("codec_name", "?")
            nb_frames = int(vstream.get("nb_frames", 0))

            try:
                num, den = fps_str.split("/")
                fps_val = float(num) / float(den)
            except Exception:
                fps_val = 0

            logger.info(f"  File:        {Path(video_path).name}")
            logger.info(f"  Resolution:  {w}x{h}")
            logger.info(f"  FPS:         {fps_str} ({fps_val:.2f})")
            logger.info(f"  Duration:    {dur:.1f}s ({dur/60:.1f} min)")
            logger.info(f"  Codec:       {codec}")
            logger.info(f"  Frames:      {nb_frames if nb_frames else 'unknown'}")

            target_fps = cfg.get("extraction", {}).get("fps", 3)
            est_frames = int(dur * target_fps)
            logger.info(f"  Est. frames at {target_fps} fps: ~{est_frames}")

            video_meta = {
                "path": str(video_path),
                "width": w, "height": h,
                "fps": fps_str, "fps_float": fps_val,
                "duration_s": dur, "codec": codec,
                "nb_frames": nb_frames,
                "estimated_extract_frames": est_frames,
            }
            with open(output_dir / "logs" / "video_info.json", "w") as f:
                json.dump(video_meta, f, indent=2)

        except Exception as e:
            errors.append(f"Video validation failed: {e}")

    # ── 4. Create Output Structure ────────────────────────────────────────────
    logger.info("[4/4] Creating output directory structure")
    try:
        create_output_structure(output_dir)
        logger.info(f"  Output dir: {output_dir}")
    except Exception as e:
        errors.append(f"Failed to create output structure: {e}")

    # ── Result ────────────────────────────────────────────────────────────────
    if errors:
        logger.error("=" * 60)
        logger.error("PRE-FLIGHT FAILED. Errors:")
        for e in errors:
            logger.error(f"  ✗ {e}")
        logger.error("=" * 60)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Stage 0 PASSED. All pre-flight checks OK.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
