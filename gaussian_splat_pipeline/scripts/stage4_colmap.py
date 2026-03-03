#!/usr/bin/env python3
"""
Stage 4: COLMAP Pose Estimation
- Path A: SIFT (traditional, --matcher sift)
- Path B: SuperPoint + LightGlue (via hloc, --matcher superpoint) — RECOMMENDED
- Full pipeline: feature extraction → matching → sparse reconstruction → undistortion
- Validates registration rate and reprojection error with abort thresholds
"""

import sys
import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger         import get_pipeline_logger
from utils.seed_utils     import set_all_seeds
from utils.config_loader  import load_config
from utils.colmap_parser  import parse_reconstruction, validate_reconstruction
from utils.gpu_utils      import detect_gpus


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4: COLMAP pose estimation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deps-dir",   required=True)
    p.add_argument("--matcher",    default="superpoint", choices=["sift", "superpoint"])
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def run_cmd(cmd: list, logger, check=True) -> subprocess.CompletedProcess:
    """Run a command, log it, and optionally raise on failure."""
    logger.info(f"  Running: {' '.join(str(c) for c in cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode}):")
        logger.error(result.stderr[-3000:])
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return result


def gpu_index_str(num_gpus: int) -> str:
    """Returns '0,1,2,3' style string for COLMAP GPU flags."""
    return ",".join(str(i) for i in range(num_gpus))


# ─── SIFT Pipeline ─────────────────────────────────────────────────────────────

def run_sift_pipeline(
    colmap_dir: Path,
    image_dir: Path,
    vocab_tree_path: Path,
    cfg: dict,
    num_gpus: int,
    logger,
):
    """Full COLMAP SIFT feature extraction and matching."""
    colmap_cfg = cfg.get("colmap", {})
    db_path    = colmap_dir / "database.db"
    gpu_idx    = gpu_index_str(num_gpus)

    # Feature Extraction
    logger.info("  [SIFT] Feature extraction...")
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path",              str(db_path),
        "--image_path",                 str(image_dir),
        "--ImageReader.single_camera",  "1" if colmap_cfg.get("single_camera", True) else "0",
        "--ImageReader.camera_model",   colmap_cfg.get("camera_model", "OPENCV"),
        "--FeatureExtraction.use_gpu",  "1",
        "--FeatureExtraction.gpu_index", gpu_idx,
        "--SiftExtraction.max_image_size", "4096",
        "--SiftExtraction.max_num_features", str(colmap_cfg.get("sift_max_features", 16384)),
        "--SiftExtraction.first_octave",     str(colmap_cfg.get("sift_first_octave", -1)),
        "--SiftExtraction.num_octaves",      str(colmap_cfg.get("sift_num_octaves", 8)),
        "--SiftExtraction.peak_threshold",   str(colmap_cfg.get("sift_peak_threshold", 0.004)),
    ], logger)

    # Sequential Matching
    logger.info("  [SIFT] Sequential feature matching...")
    match_cmd = [
        "colmap", "sequential_matcher",
        "--database_path",                    str(db_path),
        "--FeatureMatching.use_gpu",          "1",
        "--FeatureMatching.gpu_index",        gpu_idx,
        "--SequentialMatching.overlap",       str(colmap_cfg.get("sequential_overlap", 15)),
        "--SequentialMatching.loop_detection", "1" if colmap_cfg.get("loop_detection", True) else "0",
    ]
    if vocab_tree_path and vocab_tree_path.exists():
        match_cmd += ["--SequentialMatching.vocab_tree_path", str(vocab_tree_path)]
    run_cmd(match_cmd, logger)

    return db_path


# ─── SuperPoint+LightGlue Pipeline ────────────────────────────────────────────

def run_superpoint_pipeline(
    colmap_dir: Path,
    image_dir: Path,
    cfg: dict,
    num_gpus: int,
    logger,
) -> Path:
    """
    hloc-based SuperPoint + LightGlue pipeline.
    Returns the sparse model path (sfm_dir).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.hloc_wrapper import (
        generate_sequential_pairs,
        add_netvlad_pairs,
        run_superpoint_extraction,
        run_lightglue_matching,
        run_hloc_reconstruction,
    )

    colmap_cfg = cfg.get("colmap", {})
    sp_cfg     = colmap_cfg

    hloc_dir      = colmap_dir / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    feature_path  = hloc_dir / "features.h5"
    pairs_path    = hloc_dir / "pairs.txt"
    match_path    = hloc_dir / "matches.h5"
    sfm_dir       = colmap_dir / "sparse_hloc"

    overlap    = sp_cfg.get("sequential_overlap", 15)
    max_kp     = sp_cfg.get("superpoint_max_keypoints", 4096)
    depth_conf = sp_cfg.get("lightglue_depth_confidence", -1.0)
    width_conf = sp_cfg.get("lightglue_width_confidence", -1.0)
    loop_detect= sp_cfg.get("loop_detection", True)

    # Generate sequential pairs
    logger.info(f"  [hloc] Generating pairs (overlap={overlap})...")
    n_pairs = generate_sequential_pairs(image_dir, pairs_path, overlap=overlap)
    logger.info(f"  [hloc] Sequential pairs: {n_pairs}")

    # Add NetVLAD loop closure pairs
    if loop_detect:
        logger.info("  [hloc] Adding NetVLAD retrieval pairs for loop closure...")
        try:
            n_total = add_netvlad_pairs(image_dir, pairs_path, num_loc=20)
            logger.info(f"  [hloc] Total pairs after loop closure: {n_total}")
        except Exception as e:
            logger.warning(f"  NetVLAD failed (non-critical): {e}")

    # SuperPoint feature extraction
    logger.info(f"  [hloc] Extracting SuperPoint features (max_kp={max_kp})...")
    run_superpoint_extraction(image_dir, feature_path, max_keypoints=max_kp)

    # LightGlue matching
    logger.info(f"  [hloc] Matching with LightGlue (depth_conf={depth_conf})...")
    run_lightglue_matching(
        pairs_path, feature_path, match_path,
        depth_confidence=depth_conf, width_confidence=width_conf
    )

    # Reconstruction via hloc
    logger.info("  [hloc] Running COLMAP incremental mapper...")
    mapper_opts = {
        "ba_global_max_num_iterations": sp_cfg.get("mapper_ba_global_max_num_iterations", 100),
        "ba_global_max_refinements":    sp_cfg.get("mapper_ba_global_max_refinements", 5),
        "ba_local_max_num_iterations":  sp_cfg.get("mapper_ba_local_max_num_iterations", 40),
        "multiple_models": bool(sp_cfg.get("mapper_multiple_models", False)),
    }
    run_hloc_reconstruction(
        sfm_dir, image_dir, pairs_path, feature_path, match_path,
        camera_model=sp_cfg.get("camera_model", "OPENCV"),
        single_camera=sp_cfg.get("single_camera", True),
        mapper_options=mapper_opts,
    )
    return sfm_dir


# ─── COLMAP Mapper (SIFT path) ────────────────────────────────────────────────

def run_mapper(
    db_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    cfg: dict,
    logger,
):
    """Run COLMAP mapper for SIFT path."""
    colmap_cfg = cfg.get("colmap", {})
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(image_dir),
        "--output_path",   str(sparse_dir),
        "--Mapper.ba_global_max_num_iterations", str(colmap_cfg.get("mapper_ba_global_max_num_iterations", 100)),
        "--Mapper.ba_global_max_refinements",    str(colmap_cfg.get("mapper_ba_global_max_refinements", 5)),
        "--Mapper.ba_local_max_num_iterations",  str(colmap_cfg.get("mapper_ba_local_max_num_iterations", 40)),
        "--Mapper.multiple_models",  "0" if not colmap_cfg.get("mapper_multiple_models", False) else "1",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.tri_min_angle",    str(colmap_cfg.get("mapper_tri_min_angle", 1.5)),
        "--Mapper.filter_max_reproj_error", str(colmap_cfg.get("mapper_filter_max_reproj_error", 2.0)),
    ], logger)


# ─── Image Undistortion ────────────────────────────────────────────────────────

def run_undistortion(
    image_dir: Path,
    sparse_model: Path,
    output_path: Path,
    logger,
):
    """Undistort images using COLMAP's image_undistorter."""
    output_path.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path",   str(image_dir),
        "--input_path",   str(sparse_model),
        "--output_path",  str(output_path),
        "--output_type",  "COLMAP",
        "--max_image_size", "4096",
    ], logger)


# ─── Sparse model visualization ───────────────────────────────────────────────

def save_sparse_visualization(sparse_path: Path, output_dir: Path, logger):
    """Save a top-down visualization of the sparse reconstruction."""
    try:
        from utils.colmap_parser import read_images_binary, read_points3d_binary
        import numpy as np
        import cv2

        images = read_images_binary(str(sparse_path / "images.bin"))
        points = read_points3d_binary(str(sparse_path / "points3D.bin"))

        if not points or not images:
            return

        # Get camera positions (C = -R^T @ t)
        import math
        cam_pos = []
        for img in images.values():
            q = img["qvec"]
            t = img["tvec"]
            # Convert quaternion to rotation matrix
            w, x, y, z = q
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
            ])
            C = -R.T @ np.array(t)
            cam_pos.append(C)

        pt_xyz = np.array([p["xyz"] for p in points.values()])
        cam_arr = np.array(cam_pos)

        # Project to XZ plane (top-down)
        canvas_size = 800
        margin = 50

        all_pts = np.vstack([pt_xyz[:, [0, 2]], cam_arr[:, [0, 2]]])
        xmin, zmin = all_pts.min(axis=0) - 0.1
        xmax, zmax = all_pts.max(axis=0) + 0.1

        def to_px(x, z):
            px = int((x - xmin) / (xmax - xmin) * (canvas_size - 2*margin) + margin)
            pz = int((z - zmin) / (zmax - zmin) * (canvas_size - 2*margin) + margin)
            return px, canvas_size - pz

        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 30  # dark background
        for pt in pt_xyz[::max(1, len(pt_xyz)//5000)]:
            px, pz = to_px(pt[0], pt[2])
            if 0 <= px < canvas_size and 0 <= pz < canvas_size:
                canvas[pz, px] = (80, 200, 80)

        for pos in cam_arr:
            px, pz = to_px(pos[0], pos[2])
            if 0 <= px < canvas_size and 0 <= pz < canvas_size:
                cv2.circle(canvas, (px, pz), 5, (0, 100, 255), -1)

        cv2.putText(canvas, "Camera positions (orange) + Sparse points (green)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        vis_path = output_dir / "logs" / "sparse_topdown.png"
        cv2.imwrite(str(vis_path), canvas)
        logger.info(f"  Sparse visualization saved: {vis_path}")
    except Exception as e:
        logger.warning(f"  Could not save sparse visualization: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    deps_dir   = Path(args.deps_dir)
    cfg        = load_config(args.config)
    seed       = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage4")
    logger.info("=" * 60)
    logger.info(f"Stage 4: COLMAP Pose Estimation (matcher={args.matcher})")
    logger.info("=" * 60)

    image_dir    = output_dir / "frames_filtered"
    colmap_dir   = output_dir / "colmap"
    sparse_dir   = colmap_dir / "sparse"
    undist_dir   = colmap_dir / "undistorted"
    vocab_tree   = deps_dir / "vocab_trees" / "vocab_tree_flickr100K_words1M.bin"
    gpus         = detect_gpus()
    num_gpus     = min(args.num_gpus, max(len(gpus), 1))

    total_images = len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if total_images == 0:
        logger.error(f"No frames in {image_dir}. Run Stage 1 first.")
        sys.exit(1)
    logger.info(f"  Input frames: {total_images}")

    # ── Execute chosen pipeline ───────────────────────────────────────────────
    if args.matcher == "superpoint":
        logger.info("  Using SuperPoint + LightGlue via hloc (recommended)")
        try:
            sfm_dir = run_superpoint_pipeline(colmap_dir, image_dir, cfg, num_gpus, logger)
            # hloc writes the best model's .bin files directly into sfm_dir,
            # but also keeps all models in sfm_dir/models/<idx>/.
            # Pick the model directory with the most registered images.
            sparse_model = sfm_dir  # default: hloc's chosen model at sfm_dir root
            best_count = 0

            # Check sfm_dir root
            if (sfm_dir / "images.bin").exists():
                try:
                    s = parse_reconstruction(str(sfm_dir), total_images)
                    best_count = s.get("num_registered", 0)
                except Exception:
                    pass

            # Check all sub-models in models/ directory
            models_dir = sfm_dir / "models"
            if models_dir.is_dir():
                for sub in sorted(models_dir.iterdir()):
                    if sub.is_dir() and (sub / "images.bin").exists():
                        try:
                            s = parse_reconstruction(str(sub), total_images)
                            cnt = s.get("num_registered", 0)
                            if cnt > best_count:
                                best_count = cnt
                                sparse_model = sub
                        except Exception:
                            continue

            logger.info(f"  Best model has {best_count} registered images at {sparse_model}")
        except Exception as e:
            logger.error(f"SuperPoint+LightGlue failed: {e}")
            logger.warning("Falling back to SIFT...")
            args.matcher = "sift"

    if args.matcher == "sift":
        logger.info("  Using SIFT (traditional COLMAP pipeline)")
        db_path = run_sift_pipeline(colmap_dir, image_dir, vocab_tree, cfg, num_gpus, logger)
        logger.info("  Running COLMAP mapper...")
        run_mapper(db_path, image_dir, sparse_dir, cfg, logger)
        # Find model 0 (best model)
        sparse_model = sparse_dir / "0"
        if not sparse_model.exists():
            candidates = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
            if not candidates:
                logger.error("Mapper produced no models. Reconstruction failed.")
                sys.exit(1)
            sparse_model = candidates[0]

    logger.info(f"  Sparse model path: {sparse_model}")

    # ── Validate reconstruction ───────────────────────────────────────────────
    logger.info("  Validating reconstruction quality...")
    try:
        stats = parse_reconstruction(str(sparse_model), total_images)
        status, messages = validate_reconstruction(stats, cfg)

        logger.info(f"  Registered images: {stats['num_registered']} / {total_images} "
                    f"({stats['registration_rate']:.1%})")
        logger.info(f"  3D points:         {stats['num_3d_points']:,}")
        logger.info(f"  Mean reproj error: {stats['mean_reprojection_error']:.3f} px")
        logger.info(f"  Mean track length: {stats['mean_track_length']:.1f}")

        for msg in messages:
            if status == "abort":
                logger.error(f"  ABORT: {msg}")
            else:
                logger.warning(f"  WARN: {msg}")

        # Save stats
        stats_path = output_dir / "logs" / "colmap_stats.json"
        with open(stats_path, "w") as f:
            json.dump({**stats, "status": status, "messages": messages}, f, indent=2)

        if status == "abort":
            logger.error("Reconstruction quality below abort thresholds. Cannot continue.")
            logger.error("Troubleshooting:")
            logger.error("  1. Ensure video has good overlap between frames")
            logger.error("  2. Try: ./run_pipeline.sh --fps 5 (more frames)")
            logger.error("  3. Try: --matcher superpoint (better feature matching)")
            logger.error("  4. Check lighting: avoid pure white/black scenes")
            sys.exit(1)

    except Exception as e:
        logger.warning(f"  Could not parse COLMAP model for validation: {e}")
        logger.warning("  Continuing without validation...")

    # ── Sparse visualization ──────────────────────────────────────────────────
    save_sparse_visualization(sparse_model, output_dir, logger)

    # ── Image undistortion ────────────────────────────────────────────────────
    logger.info("  Running image undistortion...")
    run_undistortion(image_dir, sparse_model, undist_dir, logger)

    # ── Save undistorted model location for Stage 5 ───────────────────────────
    meta = {
        "sparse_model": str(sparse_model),
        "undistorted_dir": str(undist_dir),
        "matcher": args.matcher,
        "total_images": total_images,
        "stats": stats if "stats" in dir() else {},
    }
    with open(output_dir / "logs" / "colmap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Stage 4 COMPLETE. Poses estimated and images undistorted.")
    logger.info(f"  Undistorted data: {undist_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
