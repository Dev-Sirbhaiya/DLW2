#!/usr/bin/env python3
"""
Stage 3: Optional Background Masking (rembg)
- Generates foreground masks for all filtered frames
- Multi-GPU parallel using multiprocessing
- Saves masks as PNGs, and optionally white-background composite frames
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger        import get_pipeline_logger
from utils.seed_utils    import set_all_seeds
from utils.config_loader import load_config
from utils.gpu_utils     import detect_gpus, shard_list


def mask_shard_worker(args_tuple):
    """
    Process a shard of images on a single GPU using rembg.
    Returns (gpu_id, n_processed, error_or_None)
    """
    gpu_id, image_paths, masks_dir, masked_frames_dir, model_name = args_tuple

    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import cv2
        import numpy as np
        from rembg import remove, new_session
        from PIL import Image

        session = new_session(model_name)
        masks_dir        = Path(masks_dir)
        masked_frames_dir = Path(masked_frames_dir)

        for img_path in image_paths:
            img_path = Path(img_path)

            # Load image
            img = Image.open(str(img_path)).convert("RGBA")

            # Remove background
            result = remove(img, session=session)  # returns RGBA PIL Image

            # Save mask (alpha channel)
            mask = np.array(result)[:, :, 3]
            mask_path = masks_dir / (img_path.stem + ".png")
            cv2.imwrite(str(mask_path), mask)

            # Save white-background composite
            if masked_frames_dir:
                white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
                composite = Image.alpha_composite(white_bg, result)
                comp_rgb = composite.convert("RGB")
                comp_path = masked_frames_dir / img_path.name
                comp_rgb.save(str(comp_path), quality=95)

        return gpu_id, len(image_paths), None
    except Exception as e:
        return gpu_id, 0, str(e)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: Background masking")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg = load_config(args.config)
    seed = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage3")
    logger.info("=" * 60)
    logger.info("Stage 3: Background Masking (rembg)")
    logger.info("=" * 60)

    mask_cfg = cfg.get("masking", {})
    model_name     = mask_cfg.get("model", "u2net")
    white_bg       = mask_cfg.get("output_white_bg", True)

    frames_dir     = output_dir / "frames_filtered"
    masks_dir      = output_dir / "masks"
    masked_dir     = output_dir / "masks" / "masked_frames" if white_bg else None

    masks_dir.mkdir(parents=True, exist_ok=True)
    if masked_dir:
        masked_dir.mkdir(parents=True, exist_ok=True)

    # Gather frames
    image_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not image_paths:
        logger.error(f"No images in {frames_dir}. Run Stage 1 first.")
        sys.exit(1)

    # Skip already-processed
    existing = set(p.stem for p in masks_dir.glob("*.png"))
    pending  = [p for p in image_paths if p.stem not in existing]
    logger.info(f"  Model:   {model_name}")
    logger.info(f"  Total:   {len(image_paths)}, Pending: {len(pending)}")

    if not pending:
        logger.info("  All masks already exist. Skipping.")
    else:
        gpus = detect_gpus()
        num_gpus = min(args.num_gpus, max(len(gpus), 1))

        shards = shard_list([str(p) for p in pending], num_gpus)
        worker_args = [
            (
                gpu_id,
                shard,
                str(masks_dir),
                str(masked_dir) if masked_dir else "",
                model_name,
            )
            for gpu_id, shard in enumerate(shards) if shard
        ]

        logger.info(f"  Launching {len(worker_args)} GPU workers...")
        t_start = time.time()

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(mask_shard_worker, worker_args)

        t_elapsed = time.time() - t_start
        total = sum(r[1] for r in results)
        errors = [r for r in results if r[2]]
        for gpu_id, n, err in errors:
            logger.error(f"  GPU {gpu_id} error: {err}")

        logger.info(f"  Processed {total} frames in {t_elapsed:.1f}s")

    mask_count = len(list(masks_dir.glob("*.png")))
    logger.info("=" * 60)
    logger.info(f"Stage 3 COMPLETE. {mask_count} masks saved in masks/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
