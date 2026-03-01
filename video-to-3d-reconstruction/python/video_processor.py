"""
video_processor.py — Validates input video and extracts basic metadata.
"""
import subprocess
import json
import os
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    """Return duration, fps, resolution via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    num, den = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    duration = float(stream.get("duration", 0))

    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "estimated_frames": int(duration * fps),
    }


def validate(video_path: str) -> dict:
    """Validate video exists and is readable. Returns metadata dict."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ext = path.suffix.lower()
    supported = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if ext not in supported:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")

    info = get_video_info(video_path)
    if info["duration_sec"] < 2.0:
        raise ValueError("Video is too short (< 2 seconds). Need at least 2 seconds.")
    if info["width"] == 0 or info["height"] == 0:
        raise ValueError("Could not read video dimensions — file may be corrupted.")

    return info
