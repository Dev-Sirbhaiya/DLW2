"""
Frame quality analysis utilities.
All functions operate on a single image path for parallelization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ─── Blur Detection ────────────────────────────────────────────────────────────

def laplacian_variance(image_path: str) -> float:
    """Compute Laplacian variance (higher = sharper)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def tenengrad_score(image_path: str) -> float:
    """Compute Tenengrad focus measure (Sobel-based, higher = sharper)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx**2 + gy**2).mean())


# ─── Motion Blur Detection (FFT) ──────────────────────────────────────────────

def motion_blur_score(image_path: str) -> float:
    """
    Compute FFT-based motion blur directionality score.
    High score (>0.7) indicates strong directional motion blur.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    band = max(2, min(h, w) // 50)
    horiz = magnitude[ch - band: ch + band, :].sum()
    vert  = magnitude[:, cw - band: cw + band].sum()
    total = magnitude.sum() + 1e-9
    return float(max(horiz, vert) / total)


# ─── Exposure Analysis ────────────────────────────────────────────────────────

def exposure_stats(image_path: str) -> dict:
    """
    Compute exposure statistics for a frame.
    Returns dict with: mean_brightness, std_brightness, dark_ratio, bright_ratio
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"mean_brightness": 0, "std_brightness": 0, "dark_ratio": 1.0, "bright_ratio": 0.0}
    total = img.size
    mean_b = float(img.mean()) / 255.0
    std_b  = float(img.std()) / 255.0
    dark_r  = float((img < 30).sum()) / total
    bright_r = float((img > 225).sum()) / total
    return {
        "mean_brightness": mean_b,
        "std_brightness": std_b,
        "dark_ratio": dark_r,
        "bright_ratio": bright_r,
    }


# ─── Perceptual Hashing (pHash) ───────────────────────────────────────────────

def compute_phash(image_path: str) -> str:
    """Compute perceptual hash string for deduplication."""
    try:
        import imagehash
        from PIL import Image
        return str(imagehash.phash(Image.open(str(image_path))))
    except Exception:
        return ""


def phash_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two pHash strings."""
    try:
        import imagehash
        return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
    except Exception:
        return 999


# ─── Batch Quality Analysis ───────────────────────────────────────────────────

def analyze_frame(image_path: str) -> dict:
    """
    Full quality analysis for a single frame.
    Returns all metrics in one dict. Designed for ProcessPoolExecutor.
    """
    path = str(image_path)
    lap   = laplacian_variance(path)
    teng  = tenengrad_score(path)
    mb    = motion_blur_score(path)
    exp   = exposure_stats(path)
    ph    = compute_phash(path)

    return {
        "filename": Path(path).name,
        "laplacian": lap,
        "tenengrad": teng,
        "motion_blur": mb,
        "mean_brightness": exp["mean_brightness"],
        "std_brightness": exp["std_brightness"],
        "dark_ratio": exp["dark_ratio"],
        "bright_ratio": exp["bright_ratio"],
        "phash": ph,
    }


def apply_quality_filters(
    frame_data: list,
    config: dict,
) -> Tuple[list, list, dict]:
    """
    Apply all quality filters to frame_data list (from analyze_frame).
    Returns (kept_frames, rejected_frames, stats_dict).
    """
    extraction_cfg = config.get("extraction", {})
    blur_ratio    = extraction_cfg.get("blur_threshold_ratio", 0.6)
    mb_thresh     = extraction_cfg.get("motion_blur_fft_threshold", 0.70)
    dark_max      = extraction_cfg.get("underexposed_ratio", 0.80)
    bright_max    = extraction_cfg.get("overexposed_ratio", 0.80)
    exp_sigma     = extraction_cfg.get("exposure_std_sigma", 2.0)
    phash_thresh  = extraction_cfg.get("phash_threshold", 5)

    # ── Step 1: Blur filter
    scores = [f["laplacian"] for f in frame_data]
    median_lap = float(np.median(scores)) if scores else 1.0
    lap_thresh = median_lap * blur_ratio

    after_blur = [f for f in frame_data if f["laplacian"] >= lap_thresh]
    blur_rejected = [
        {**f, "status": "DISCARDED", "reason": "blur_below_threshold"}
        for f in frame_data if f["laplacian"] < lap_thresh
    ]

    # ── Step 2: Motion blur filter
    after_mb = [f for f in after_blur if f["motion_blur"] <= mb_thresh]
    mb_rejected = [
        {**f, "status": "DISCARDED", "reason": "motion_blur"}
        for f in after_blur if f["motion_blur"] > mb_thresh
    ]

    # ── Step 3: Exposure filter
    mean_bright = np.mean([f["mean_brightness"] for f in after_mb]) if after_mb else 0.5
    std_bright  = np.std([f["mean_brightness"] for f in after_mb]) if after_mb else 0.1

    def exposure_ok(f):
        if f["dark_ratio"] > dark_max:
            return False, "underexposed"
        if f["bright_ratio"] > bright_max:
            return False, "overexposed"
        if abs(f["mean_brightness"] - mean_bright) > exp_sigma * std_bright:
            return False, "exposure_outlier"
        return True, ""

    after_exp = []
    exp_rejected = []
    for f in after_mb:
        ok, reason = exposure_ok(f)
        if ok:
            after_exp.append(f)
        else:
            exp_rejected.append({**f, "status": "DISCARDED", "reason": reason})

    # ── Step 4: Near-duplicate removal (pHash)
    after_dedup = []
    dedup_rejected = []
    seen_hashes = []

    for f in after_exp:
        ph = f["phash"]
        is_dup = False
        if ph:
            for prev_ph in seen_hashes:
                if phash_distance(ph, prev_ph) < phash_thresh:
                    is_dup = True
                    break
        if is_dup:
            dedup_rejected.append({**f, "status": "DISCARDED", "reason": "near_duplicate"})
        else:
            after_dedup.append(f)
            if ph:
                seen_hashes.append(ph)

    kept = [{**f, "status": "KEPT", "reason": ""} for f in after_dedup]
    rejected = blur_rejected + mb_rejected + exp_rejected + dedup_rejected

    stats = {
        "total": len(frame_data),
        "after_blur": len(after_blur),
        "after_motion_blur": len(after_mb),
        "after_exposure": len(after_exp),
        "after_dedup": len(after_dedup),
        "final": len(kept),
        "lap_threshold": lap_thresh,
        "median_laplacian": median_lap,
    }

    return kept, rejected, stats
