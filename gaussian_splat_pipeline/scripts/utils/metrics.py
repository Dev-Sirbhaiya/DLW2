"""
Image quality metrics: PSNR, SSIM, LPIPS.
Handles both single-image and batch computation.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def psnr(img_pred: np.ndarray, img_gt: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio. Higher is better. 30+ dB = good."""
    mse = np.mean((img_pred.astype(np.float64) - img_gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def ssim_score(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Structural Similarity Index. Closer to 1.0 = better."""
    try:
        from skimage.metrics import structural_similarity as ssim
        is_color = img_pred.ndim == 3 and img_pred.shape[-1] == 3
        return float(ssim(img_gt, img_pred, channel_axis=-1 if is_color else None,
                          data_range=1.0 if img_pred.max() <= 1.0 else 255.0))
    except Exception:
        return 0.0


def lpips_score(img_pred: np.ndarray, img_gt: np.ndarray, device: str = "cuda") -> float:
    """
    Learned Perceptual Image Patch Similarity. Lower is better (<0.1 = good).
    Images should be HxWxC uint8 or float [0,1].
    """
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net="vgg").to(device)

        def to_tensor(img):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            t = t * 2 - 1  # normalize to [-1, 1]
            return t.to(device)

        with torch.no_grad():
            val = loss_fn(to_tensor(img_pred), to_tensor(img_gt))
        return float(val.item())
    except Exception:
        return -1.0


def load_image_float(path: str) -> Optional[np.ndarray]:
    """Load image as float32 [0, 1] HxWxC (RGB)."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:
        return None


def compute_metrics_batch(
    pred_paths: List[str],
    gt_paths: List[str],
    compute_lpips: bool = True,
    device: str = "cuda",
) -> Tuple[list, dict]:
    """
    Compute PSNR/SSIM/LPIPS for pairs of predicted and GT images.
    Returns: (per_image_list, aggregate_dict)
    """
    try:
        import torch
        import lpips as lpips_lib
        loss_fn = lpips_lib.LPIPS(net="vgg").to(device) if compute_lpips else None
    except Exception:
        loss_fn = None
        compute_lpips = False

    per_image = []
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = load_image_float(pred_path)
        gt   = load_image_float(gt_path)
        if pred is None or gt is None:
            continue

        p = psnr(pred, gt)
        s = ssim_score(pred, gt)
        l = lpips_score(pred, gt, device) if compute_lpips and loss_fn else -1.0

        per_image.append({
            "filename": Path(pred_path).name,
            "psnr": round(p, 4),
            "ssim": round(s, 6),
            "lpips": round(l, 6),
        })

    if not per_image:
        return [], {}

    psnrs = [x["psnr"] for x in per_image if x["psnr"] != float("inf")]
    ssims = [x["ssim"] for x in per_image]
    lpipss = [x["lpips"] for x in per_image if x["lpips"] >= 0]

    agg = {
        "psnr_mean": round(float(np.mean(psnrs)), 4) if psnrs else 0,
        "psnr_std":  round(float(np.std(psnrs)),  4) if psnrs else 0,
        "ssim_mean": round(float(np.mean(ssims)), 6) if ssims else 0,
        "ssim_std":  round(float(np.std(ssims)),  6) if ssims else 0,
        "lpips_mean": round(float(np.mean(lpipss)), 6) if lpipss else -1,
        "lpips_std":  round(float(np.std(lpipss)),  6) if lpipss else -1,
        "num_images": len(per_image),
    }
    return per_image, agg


def generate_comparison_grid(
    pred_paths: List[str],
    gt_paths: List[str],
    output_path: str,
    max_images: int = 8,
    cols: int = 4,
):
    """
    Save a side-by-side comparison grid:
    [GT | Pred | Abs Error] repeated for multiple images.
    """
    import cv2
    import numpy as np

    pairs = list(zip(pred_paths, gt_paths))[:max_images]
    panels = []

    for pred_path, gt_path in pairs:
        pred = cv2.imread(str(pred_path))
        gt   = cv2.imread(str(gt_path))
        if pred is None or gt is None:
            continue
        h, w = gt.shape[:2]
        pred = cv2.resize(pred, (w, h))

        # Absolute error heatmap
        err = np.abs(gt.astype(np.float32) - pred.astype(np.float32)).mean(axis=-1)
        err_norm = (err / (err.max() + 1e-9) * 255).astype(np.uint8)
        err_color = cv2.applyColorMap(err_norm, cv2.COLORMAP_MAGMA)

        row = np.concatenate([gt, pred, err_color], axis=1)

        # Label
        for label, x_off in [("Ground Truth", 5), ("Rendered", w + 5), ("Abs Error", 2 * w + 5)]:
            cv2.putText(row, label, (x_off, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(row)

    if not panels:
        return

    # Stack panels vertically (3 cols each takes full width)
    grid = np.concatenate(panels, axis=0)
    cv2.imwrite(str(output_path), grid)
