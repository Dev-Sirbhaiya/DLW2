"""
data_processor.py — Runs ns-process-data to extract frames and run COLMAP SfM.
Emits JSON progress lines to stdout for the Node.js pipeline service.
"""
import subprocess
import sys
import os
import json
from pathlib import Path


def _emit(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    print(json.dumps(payload), flush=True)


def run(video_path: str, output_dir: str, num_frames: int = 300) -> str:
    """
    Run ns-process-data video to extract frames and run COLMAP.
    Returns the path to the processed data directory.
    """
    processed_dir = Path(output_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    _emit("log", line=f"Running ns-process-data on: {video_path}")
    _emit("log", line=f"Output directory: {processed_dir}")
    _emit("log", line=f"Target frame count: {num_frames}")

    cmd = [
        "ns-process-data", "video",
        "--data", str(video_path),
        "--output-dir", str(processed_dir),
        "--num-frames-target", str(num_frames),
    ]

    _emit("log", line=f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            _emit("log", line=line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"ns-process-data exited with code {process.returncode}. "
            "Check COLMAP is installed and the video has enough visual overlap."
        )

    # Validate COLMAP produced camera poses
    transforms_file = processed_dir / "transforms.json"
    if not transforms_file.exists():
        raise RuntimeError(
            "COLMAP did not produce transforms.json. "
            "Try a different video with more static scene content."
        )

    with open(transforms_file) as f:
        transforms = json.load(f)

    num_cameras = len(transforms.get("frames", []))
    if num_cameras < 10:
        raise RuntimeError(
            f"Only {num_cameras} camera poses recovered (need ≥10). "
            "Use a video with slower movement and more scene overlap."
        )

    _emit("log", line=f"COLMAP recovered {num_cameras} camera poses.")
    return str(processed_dir)
