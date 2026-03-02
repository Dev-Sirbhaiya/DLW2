#!/usr/bin/env python3
"""
Stage 2: Depth Map Generation (Depth-Anything-V2)
- Generates monocular depth maps for all filtered frames
- Multi-GPU parallel: shards frames across available GPUs
- Saves 16-bit PNGs (format expected by original 3DGS -d flag)
- Also saves colorized visualization for debugging
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
from utils.depth_utils   import process_shard_worker, MODEL_CONFIGS, CHECKPOINT_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Depth map generation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deps-dir",   required=True, help="Path to deps/ directory")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def worker_fn(args_tuple):
    """Wrapper for multiprocessing that unpacks arguments."""
    gpu_id, image_paths, output_dir, vis_dir, encoder, ckpt_path, deps_dir = args_tuple
    try:
        process_shard_worker(
            gpu_id=gpu_id,
            image_paths=image_paths,
            output_dir=output_dir,
            vis_dir=vis_dir,
            encoder=encoder,
            checkpoint_path=ckpt_path,
            deps_dir=deps_dir,
        )
        return gpu_id, len(image_paths), None
    except Exception as e:
        return gpu_id, 0, str(e)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    deps_dir   = Path(args.deps_dir)
    cfg = load_config(args.config)
    seed = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage2")
    logger.info("=" * 60)
    logger.info("Stage 2: Depth Map Generation (Depth-Anything-V2)")
    logger.info("=" * 60)

    depth_cfg = cfg.get("depth", {})
    encoder   = depth_cfg.get("model", "vitb")

    # Validate encoder
    if encoder not in MODEL_CONFIGS:
        logger.warning(f"Unknown encoder '{encoder}'. Using vitb.")
        encoder = "vitb"

    ckpt_name = CHECKPOINT_NAMES[encoder]
    ckpt_path = deps_dir / "Depth-Anything-V2" / "checkpoints" / ckpt_name

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run ./setup.sh to download depth model checkpoints.")
        sys.exit(1)

    frames_dir   = output_dir / "frames_filtered"
    depth_dir    = output_dir / "depth_maps"
    vis_dir      = output_dir / "depth_maps" / "vis"
    depth_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get all filtered frames
    image_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not image_paths:
        logger.error(f"No images found in {frames_dir}. Run Stage 1 first.")
        sys.exit(1)

    logger.info(f"  Encoder:       {encoder}")
    logger.info(f"  Checkpoint:    {ckpt_path}")
    logger.info(f"  Input frames:  {len(image_paths)}")

    # Detect GPUs
    gpus = detect_gpus()
    num_gpus = min(args.num_gpus, len(gpus)) if gpus else 1
    logger.info(f"  GPUs to use:   {num_gpus}")

    # Check for already-processed depth maps (resume support)
    existing = set(p.stem for p in depth_dir.glob("*.png") if not p.stem.endswith("_vis"))
    pending = [p for p in image_paths if p.stem not in existing]
    logger.info(f"  Already done:  {len(existing)}, Pending: {len(pending)}")

    if not pending:
        logger.info("  All depth maps already exist. Skipping generation.")
    else:
        # Shard across GPUs
        shards = shard_list([str(p) for p in pending], num_gpus)

        # Build args for each worker
        worker_args = []
        for gpu_id, shard in enumerate(shards):
            if shard:
                worker_args.append((
                    gpu_id,
                    shard,
                    str(depth_dir),
                    str(vis_dir),
                    encoder,
                    str(ckpt_path),
                    str(deps_dir),
                ))

        logger.info(f"  Launching {len(worker_args)} parallel GPU workers...")
        t_start = time.time()

        # Use spawn start method to avoid CUDA init issues
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_gpus) as pool:
            results = pool.map(worker_fn, worker_args)

        t_elapsed = time.time() - t_start
        total_processed = sum(r[1] for r in results)
        errors = [r for r in results if r[2] is not None]

        if errors:
            for gpu_id, n, err in errors:
                logger.error(f"  GPU {gpu_id} error: {err}")
            if total_processed == 0:
                logger.error("All GPU workers failed. Check Depth-Anything-V2 installation.")
                sys.exit(1)

        logger.info(f"  Processed {total_processed} frames in {t_elapsed:.1f}s "
                    f"({t_elapsed/max(total_processed,1):.1f}s/frame)")

    # Verify output
    depth_files = sorted(depth_dir.glob("*.png"))
    depth_files = [f for f in depth_files if not f.stem.endswith("_vis")]
    logger.info(f"  Total depth maps: {len(depth_files)}")

    if len(depth_files) == 0:
        logger.error("No depth maps generated. Check GPU workers.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Stage 2 COMPLETE. {len(depth_files)} depth maps in depth_maps/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
