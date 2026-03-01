#!/usr/bin/env python3
"""
pipeline.py — Main orchestrator CLI for the 3D/2D reconstruction pipeline.
Emits JSON progress lines to stdout; captured by the Node.js backend.

Usage:
  python pipeline.py \\
    --video /path/to/video.mp4 \\
    --output-dir /path/to/outputs/job_id \\
    --mode both \\
    --max-iterations 30000 \\
    --num-frames 300
"""
import argparse
import json
import sys
import os
import traceback
from pathlib import Path

# Add parent dir to path so backend can call as subprocess
sys.path.insert(0, str(Path(__file__).parent))

import video_processor
import data_processor
import trainer
import exporter
import renderer


STAGES = {
    1: "Validate Video",
    2: "COLMAP Structure-from-Motion",
    3: "Train Splatfacto",
    4: "Export 3D Gaussian Splat",
    5: "Render 2D Novel Views",
    6: "Complete",
}


def emit(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    print(json.dumps(payload), flush=True)


def stage_start(n: int):
    emit("stage", stage=n, name=STAGES[n], status="running")


def stage_done(n: int):
    emit("stage", stage=n, name=STAGES[n], status="done")


def stage_skip(n: int):
    emit("stage", stage=n, name=STAGES[n], status="skipped")


def main():
    parser = argparse.ArgumentParser(description="Video → 3D/2D Reconstruction Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", required=True, help="Job output directory")
    parser.add_argument("--mode", choices=["3d", "2d", "both"], default="both",
                        help="Reconstruction mode")
    parser.add_argument("--max-iterations", type=int, default=30000,
                        help="Splatfacto training iterations")
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Target frames for COLMAP")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit("start", video=args.video, mode=args.mode,
         max_iterations=args.max_iterations, num_frames=args.num_frames,
         total_stages=6)

    try:
        # ── Stage 1: Validate Video ───────────────────────────────────────────
        stage_start(1)
        video_info = video_processor.validate(args.video)
        emit("log", line=f"Video: {video_info['width']}x{video_info['height']} "
                         f"@ {video_info['fps']} fps, "
                         f"{video_info['duration_sec']}s "
                         f"(~{video_info['estimated_frames']} frames)")
        stage_done(1)

        # Save video info
        with open(output_dir / "video_info.json", "w") as f:
            json.dump(video_info, f, indent=2)

        # ── Stage 2: COLMAP SfM ───────────────────────────────────────────────
        stage_start(2)
        processed_dir = data_processor.run(
            args.video,
            str(output_dir),
            num_frames=args.num_frames,
        )
        stage_done(2)

        # ── Stage 3: Train Splatfacto ─────────────────────────────────────────
        stage_start(3)
        config_yml = trainer.run(
            processed_dir,
            str(output_dir),
            max_iterations=args.max_iterations,
        )
        stage_done(3)

        # ── Stage 4: Export 3D ────────────────────────────────────────────────
        export_3d_dir = output_dir / "exports" / "3d"
        if args.mode in ("3d", "both"):
            stage_start(4)
            ply_path = exporter.run(config_yml, str(export_3d_dir))
            stage_done(4)
            emit("result_3d", ply_path=ply_path)
        else:
            stage_skip(4)

        # ── Stage 5: Render 2D ────────────────────────────────────────────────
        render_dir = output_dir / "exports" / "2d"
        if args.mode in ("2d", "both"):
            stage_start(5)
            render_result = renderer.run(
                config_yml,
                str(render_dir),
                interpolation_steps=120,
                fps=24,
            )
            stage_done(5)
            emit("result_2d", **render_result)
        else:
            stage_skip(5)

        # ── Stage 6: Complete ─────────────────────────────────────────────────
        stage_done(6)
        emit("done", output_dir=str(output_dir))

    except Exception as e:
        tb = traceback.format_exc()
        emit("error", message=str(e), traceback=tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
