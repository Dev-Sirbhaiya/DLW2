"""
renderer.py — Renders 2D novel views from trained Splatfacto model.
Produces an MP4 video and a PNG contact sheet.
"""
import subprocess
import json
import math
import os
from pathlib import Path

try:
    import imageio
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def _emit(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    print(json.dumps(payload), flush=True)


def _make_contact_sheet(frames_dir: str, output_path: str, cols: int = 6) -> str:
    """Create a contact sheet PNG from rendered frames."""
    if not HAS_IMAGEIO:
        _emit("log", line="imageio not available — skipping contact sheet")
        return ""

    frame_files = sorted(Path(frames_dir).glob("*.png")) + \
                  sorted(Path(frames_dir).glob("*.jpg"))

    if not frame_files:
        _emit("log", line="No frame files found for contact sheet")
        return ""

    # Sample evenly — max 24 frames in sheet
    max_frames = 24
    step = max(1, len(frame_files) // max_frames)
    sampled = frame_files[::step][:max_frames]

    rows = math.ceil(len(sampled) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    fig.patch.set_facecolor("#050a14")

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in axes_flat:
        ax.axis("off")

    for i, fp in enumerate(sampled):
        img = imageio.v3.imread(str(fp))
        r, c = divmod(i, cols)
        if rows > 1:
            ax = axes[r][c]
        else:
            ax = axes[c] if cols > 1 else axes
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Frame {i + 1}", color="#00d4ff", fontsize=7)

    plt.suptitle("2D Novel View Synthesis — Rendered Frames",
                 color="#ffffff", fontsize=12, y=1.02)
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def run(config_yml: str, output_dir: str, interpolation_steps: int = 120, fps: int = 24) -> dict:
    """
    Render a spiral novel-view video and contact sheet.
    Returns dict with paths to render.mp4 and contact_sheet.png.
    """
    render_dir = Path(output_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = render_dir / "render.mp4"
    frames_dir = render_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    _emit("log", line=f"Rendering {interpolation_steps} novel-view frames...")
    _emit("log", line=f"Output: {mp4_path}")

    cmd = [
        "ns-render", "interpolate",
        "--load-config", str(config_yml),
        "--output-path", str(mp4_path),
        "--rendered-output-names", "rgb",
        "--interpolation-steps", str(interpolation_steps),
        "--fps", str(fps),
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
        raise RuntimeError(f"ns-render exited with code {process.returncode}.")

    if not mp4_path.exists():
        raise FileNotFoundError(f"Expected render output not found: {mp4_path}")

    # Extract frames from rendered video for contact sheet
    _emit("log", line="Extracting frames for contact sheet...")
    if HAS_IMAGEIO:
        try:
            reader = imageio.get_reader(str(mp4_path))
            n_frames = reader.count_frames()
            step = max(1, n_frames // 24)
            for i, frame in enumerate(reader):
                if i % step == 0:
                    out_frame = frames_dir / f"frame_{i:05d}.png"
                    imageio.v3.imwrite(str(out_frame), frame)
            reader.close()
        except Exception as e:
            _emit("log", line=f"Frame extraction warning: {e}")

    # Build contact sheet
    contact_sheet_path = str(render_dir / "contact_sheet.png")
    sheet = _make_contact_sheet(str(frames_dir), contact_sheet_path)

    result = {
        "mp4_path": str(mp4_path),
        "contact_sheet_path": sheet if sheet else None,
        "frames_dir": str(frames_dir),
    }

    with open(render_dir / "render_meta.json", "w") as f:
        json.dump(result, f, indent=2)

    _emit("log", line=f"2D render complete: {mp4_path}")
    if sheet:
        _emit("log", line=f"Contact sheet: {sheet}")

    return result
