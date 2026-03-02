"""
Depth-Anything-V2 inference wrapper for multi-GPU parallel depth generation.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List


# ─── Model Config ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

CHECKPOINT_NAMES = {
    "vits": "depth_anything_v2_vits.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vitl": "depth_anything_v2_vitl.pth",
}


def load_depth_model(encoder: str, checkpoint_path: str, device: str, deps_dir: str):
    """Load a Depth-Anything-V2 model on the specified device."""
    da_dir = Path(deps_dir) / "Depth-Anything-V2"
    if str(da_dir) not in sys.path:
        sys.path.insert(0, str(da_dir))

    from depth_anything_v2.dpt import DepthAnythingV2
    import torch

    cfg = MODEL_CONFIGS.get(encoder, MODEL_CONFIGS["vitb"])
    model = DepthAnythingV2(**cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def infer_depth(model, image_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
    """
    Run depth inference on a single BGR image.
    Returns float32 depth map (unnormalized, relative depth).
    """
    depth = model.infer_image(image_bgr, input_size)  # returns HxW float
    return depth.astype(np.float32)


def save_depth_16bit(depth: np.ndarray, output_path: str):
    """
    Save depth map as 16-bit PNG (format expected by original 3DGS -d flag).
    Depth is linearly normalized to [0, 65535].
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.uint16)
    else:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    cv2.imwrite(str(output_path), depth_norm)


def save_depth_colorized(depth: np.ndarray, output_path: str):
    """Save colorized depth visualization (for debugging)."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(output_path), colored)


def process_shard_worker(
    gpu_id: int,
    image_paths: List[str],
    output_dir: str,
    vis_dir: str,
    encoder: str,
    checkpoint_path: str,
    deps_dir: str,
):
    """
    Worker function for multi-GPU depth generation.
    Processes a shard of images on a specific GPU.
    Called via torch.multiprocessing.spawn or Process.
    """
    import torch
    import os

    # Bind to specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # after CUDA_VISIBLE_DEVICES, gpu_id maps to cuda:0

    # Set seeds for determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = load_depth_model(encoder, checkpoint_path, device, deps_dir)
    output_dir = Path(output_dir)
    vis_dir    = Path(vis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_path = Path(img_path)
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        with torch.no_grad():
            depth = infer_depth(model, img_bgr)

        out_path = output_dir / (img_path.stem + ".png")
        vis_path = vis_dir    / (img_path.stem + "_vis.png")
        save_depth_16bit(depth, str(out_path))
        save_depth_colorized(depth, str(vis_path))

    return gpu_id
