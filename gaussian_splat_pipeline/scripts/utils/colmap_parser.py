"""
Parse COLMAP model output files to extract registration stats, reprojection errors,
and other quality metrics needed for pipeline validation.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ─── Binary model readers ──────────────────────────────────────────────────────

def read_cameras_binary(path: str) -> dict:
    """Read COLMAP cameras.bin and return {camera_id: camera_dict}."""
    cameras = {}
    path = str(path)
    try:
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                cam_id  = struct.unpack("<i", f.read(4))[0]
                model   = struct.unpack("<i", f.read(4))[0]
                width   = struct.unpack("<Q", f.read(8))[0]
                height  = struct.unpack("<Q", f.read(8))[0]
                # Read params (number depends on model)
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8, 7: 5}.get(model, 4)
                params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
                cameras[cam_id] = {
                    "id": cam_id, "model": model,
                    "width": width, "height": height,
                    "params": list(params),
                }
    except Exception as e:
        pass
    return cameras


def read_images_binary(path: str) -> dict:
    """Read COLMAP images.bin, return {image_id: image_dict} with 2D-3D correspondences."""
    images = {}
    path = str(path)
    try:
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                img_id  = struct.unpack("<i", f.read(4))[0]
                qvec    = struct.unpack("<4d", f.read(32))
                tvec    = struct.unpack("<3d", f.read(24))
                cam_id  = struct.unpack("<i", f.read(4))[0]
                name    = b""
                c = f.read(1)
                while c != b"\x00":
                    name += c
                    c = f.read(1)
                name = name.decode("utf-8")
                num_pts = struct.unpack("<Q", f.read(8))[0]
                xys     = struct.unpack(f"<{2*num_pts}d", f.read(16 * num_pts))
                pt3d_ids= struct.unpack(f"<{num_pts}q", f.read(8 * num_pts))
                images[img_id] = {
                    "id": img_id, "qvec": qvec, "tvec": tvec,
                    "camera_id": cam_id, "name": name,
                    "num_observations": num_pts,
                    "point3d_ids": pt3d_ids,
                }
    except Exception as e:
        pass
    return images


def read_points3d_binary(path: str) -> dict:
    """Read COLMAP points3D.bin, return {point3d_id: point_dict}."""
    points = {}
    path = str(path)
    try:
        with open(path, "rb") as f:
            num_pts = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_pts):
                pt_id   = struct.unpack("<q", f.read(8))[0]
                xyz     = struct.unpack("<3d", f.read(24))
                rgb     = struct.unpack("<3B", f.read(3))
                error   = struct.unpack("<d", f.read(8))[0]
                track_len = struct.unpack("<Q", f.read(8))[0]
                track   = struct.unpack(f"<{2*track_len}i", f.read(8 * track_len))
                points[pt_id] = {
                    "id": pt_id, "xyz": xyz, "rgb": rgb,
                    "error": error, "track_length": track_len,
                }
    except Exception as e:
        pass
    return points


# ─── High-level stats ──────────────────────────────────────────────────────────

def parse_reconstruction(sparse_path: str, total_images: int) -> dict:
    """
    Parse a COLMAP sparse reconstruction and compute quality stats.
    sparse_path: path to a sparse model directory (e.g., output/colmap/sparse/0)
    total_images: total number of input images (for registration rate)
    Returns a stats dict.
    """
    sparse_path = Path(sparse_path)

    cameras = read_cameras_binary(sparse_path / "cameras.bin")
    images  = read_images_binary(sparse_path / "images.bin")
    points  = read_points3d_binary(sparse_path / "points3D.bin")

    num_registered = len(images)
    num_points     = len(points)
    reg_rate       = num_registered / max(total_images, 1)

    errors = [p["error"] for p in points.values() if p["error"] > 0]
    mean_reproj = float(np.mean(errors)) if errors else 0.0

    track_lengths = [p["track_length"] for p in points.values()]
    mean_track = float(np.mean(track_lengths)) if track_lengths else 0.0

    cam_info = {}
    if cameras:
        c = next(iter(cameras.values()))
        cam_info = {"width": c["width"], "height": c["height"], "params": c["params"]}

    return {
        "num_registered": num_registered,
        "total_images": total_images,
        "registration_rate": round(reg_rate, 4),
        "num_3d_points": num_points,
        "mean_reprojection_error": round(mean_reproj, 4),
        "mean_track_length": round(mean_track, 2),
        "camera_info": cam_info,
    }


def validate_reconstruction(stats: dict, config: dict) -> Tuple[str, list]:
    """
    Validate reconstruction quality against config thresholds.
    Returns: ("ok" | "warning" | "abort", [list of messages])
    """
    colmap_cfg = config.get("colmap", {})
    min_rate   = colmap_cfg.get("min_registration_rate", 0.50)
    warn_rate  = colmap_cfg.get("warn_registration_rate", 0.80)
    abort_err  = colmap_cfg.get("max_reproj_error_abort", 3.0)
    warn_err   = colmap_cfg.get("max_reproj_error_warn", 1.5)

    reg_rate   = stats["registration_rate"]
    reproj_err = stats["mean_reprojection_error"]
    messages   = []
    status     = "ok"

    if reg_rate < min_rate:
        messages.append(
            f"Registration rate {reg_rate:.1%} < minimum {min_rate:.1%}. "
            f"Likely causes: blurry video, not enough overlap, challenging scene. "
            f"Try: slower capture speed, better lighting, or switch to --matcher sift."
        )
        status = "abort"
    elif reg_rate < warn_rate:
        messages.append(
            f"Registration rate {reg_rate:.1%} < {warn_rate:.1%}. "
            f"Results may have artifacts. Consider re-recording for best quality."
        )
        if status != "abort":
            status = "warning"

    if reproj_err > abort_err:
        messages.append(
            f"Mean reprojection error {reproj_err:.2f}px > {abort_err:.1f}px threshold. "
            f"Poses are unreliable. Causes: wrong camera model, noisy video, bad feature matching. "
            f"Try: switch to OPENCV_FISHEYE model if using fisheye lens, or lower matching threshold."
        )
        status = "abort"
    elif reproj_err > warn_err:
        messages.append(
            f"Mean reprojection error {reproj_err:.2f}px > {warn_err:.1f}px. "
            f"Poses may be slightly inaccurate. Reconstruction should still work."
        )
        if status != "abort":
            status = "warning"

    return status, messages
