#!/usr/bin/env python3
"""
Stage 1: Intelligent Frame Extraction & Quality Filtering
- Extract frames with ffmpeg at target fps
- Multi-pass quality filter: blur, motion blur, exposure, dedup
- Output filtered frames to frames_filtered/
"""

import sys
import os
import csv
import shutil
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger        import get_pipeline_logger
from utils.seed_utils    import set_all_seeds
from utils.config_loader import load_config
from utils.frame_quality import analyze_frame, apply_quality_filters


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Frame extraction and filtering")
    p.add_argument("--input",      required=True, help="Input video path")
    p.add_argument("--fps",        type=float, default=3.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def extract_frames(video_path: str, frames_dir: Path, fps: float, jpeg_quality: int = 1) -> int:
    """Extract frames with ffmpeg. Returns number of extracted frames."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(frames_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vsync", "cfr",
        "-vf", f"fps={fps}",
        "-q:v", str(jpeg_quality),
        output_pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed:\n{result.stderr}")

    extracted = sorted(frames_dir.glob("frame_*.jpg"))
    return len(extracted)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg = load_config(args.config)
    seed = cfg.get("pipeline", {}).get("seed", 42)
    det  = cfg.get("pipeline", {}).get("deterministic", True)
    set_all_seeds(seed, det)

    logger = get_pipeline_logger(output_dir, "stage1")
    logger.info("=" * 60)
    logger.info("Stage 1: Frame Extraction & Quality Filtering")
    logger.info("=" * 60)

    ext_cfg = cfg.get("extraction", {})
    fps          = args.fps or ext_cfg.get("fps", 3)
    jpeg_quality = ext_cfg.get("jpeg_quality", 1)
    min_frames   = ext_cfg.get("min_frames", 40)
    max_frames   = ext_cfg.get("max_frames", 500)
    ideal_min    = ext_cfg.get("ideal_min", 80)
    ideal_max    = ext_cfg.get("ideal_max", 300)

    frames_raw_dir      = output_dir / "frames_raw"
    frames_filtered_dir = output_dir / "frames_filtered"
    frames_filtered_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract frames ─────────────────────────────────────────────────────
    logger.info(f"Extracting frames at {fps} fps...")
    n_extracted = extract_frames(args.input, frames_raw_dir, fps, jpeg_quality)
    logger.info(f"  Extracted: {n_extracted} frames")

    all_frames = sorted(frames_raw_dir.glob("frame_*.jpg"))
    if n_extracted == 0:
        logger.error("No frames extracted. Check video file and ffmpeg installation.")
        sys.exit(1)

    # ── 2. Parallel quality analysis ─────────────────────────────────────────
    logger.info(f"Analyzing frame quality ({os.cpu_count()} CPU workers)...")
    frame_paths = [str(f) for f in all_frames]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        frame_data = list(executor.map(analyze_frame, frame_paths))

    logger.info(f"  Quality analysis complete for {len(frame_data)} frames.")

    # ── 3. Apply quality filters ──────────────────────────────────────────────
    logger.info("Applying quality filters...")
    kept, rejected, stats = apply_quality_filters(frame_data, cfg)

    logger.info(
        f"  Total:       {stats['total']}\n"
        f"  After blur:  {stats['after_blur']}\n"
        f"  After motion blur: {stats['after_motion_blur']}\n"
        f"  After exposure: {stats['after_exposure']}\n"
        f"  After dedup: {stats['after_dedup']}\n"
        f"  FINAL KEPT:  {stats['final']}"
    )
    logger.info(
        f"  Laplacian threshold: {stats['lap_threshold']:.1f} "
        f"(median: {stats['median_laplacian']:.1f})"
    )

    # ── 4. Guardrail checks ───────────────────────────────────────────────────
    final_count = stats["final"]
    if final_count < min_frames:
        logger.warning(
            f"Only {final_count} frames survived filtering (minimum: {min_frames}).\n"
            "  Reconstruction quality will likely be poor.\n"
            "  Recommendation: re-record with slower/longer capture, better lighting,\n"
            "  or lower --fps to sample fewer but higher-quality frames."
        )
    elif final_count < ideal_min:
        logger.warning(
            f"{final_count} frames is below ideal range ({ideal_min}-{ideal_max}).\n"
            "  Consider re-recording for better reconstruction quality."
        )
    elif final_count > max_frames:
        logger.warning(
            f"{final_count} frames is excessive (max recommended: {max_frames}).\n"
            "  Consider lowering --fps to reduce COLMAP processing time."
        )
    elif final_count > ideal_max:
        logger.warning(
            f"{final_count} frames is above ideal range ({ideal_min}-{ideal_max}).\n"
            "  COLMAP will be slower. Consider lowering --fps."
        )

    # ── 5. Copy kept frames to filtered dir ───────────────────────────────────
    logger.info(f"Copying {final_count} frames to frames_filtered/...")
    for frame_info in kept:
        src = frames_raw_dir  / frame_info["filename"]
        dst = frames_filtered_dir / frame_info["filename"]
        shutil.copy2(str(src), str(dst))

    # ── 6. Save quality report CSV ────────────────────────────────────────────
    report_path = output_dir / "logs" / "frame_quality_report.csv"
    all_data = [{**f, "status": "KEPT",      "reason": ""} for f in kept] + rejected
    all_data.sort(key=lambda x: x["filename"])

    fieldnames = [
        "filename", "laplacian", "tenengrad", "motion_blur",
        "mean_brightness", "std_brightness", "dark_ratio", "bright_ratio",
        "phash", "status", "reason"
    ]
    with open(report_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_data)

    logger.info(f"  Quality report saved: {report_path}")

    if final_count == 0:
        logger.error("Zero frames passed quality filters. Cannot continue.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Stage 1 COMPLETE. {final_count} frames ready in frames_filtered/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
