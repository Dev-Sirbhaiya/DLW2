"""
SuperPoint + LightGlue feature matching via hloc (Hierarchical-Localization).
Provides a clean interface for the COLMAP stage.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional


def generate_sequential_pairs(
    image_dir: Path,
    pairs_path: Path,
    overlap: int = 15,
):
    """Generate sequential image pairs with specified overlap for hloc."""
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    images = [p.name for p in images]

    pairs = []
    for i, img in enumerate(images):
        for j in range(i + 1, min(i + 1 + overlap, len(images))):
            pairs.append(f"{img} {images[j]}")

    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w") as f:
        f.write("\n".join(pairs))

    return len(pairs)


def run_superpoint_extraction(
    image_dir: Path,
    feature_path: Path,
    max_keypoints: int = 4096,
    resize: Optional[int] = None,
    device: str = "cuda",
):
    """
    Extract SuperPoint features using hloc.
    """
    from hloc import extract_features

    conf = {
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": max_keypoints,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": resize if resize else 1600,
        },
    }

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    extract_features.main(conf, image_dir, feature_path=feature_path)


def run_lightglue_matching(
    pairs_path: Path,
    feature_path: Path,
    match_path: Path,
    depth_confidence: float = -1.0,
    width_confidence: float = -1.0,
    device: str = "cuda",
):
    """
    Match features with LightGlue using hloc.
    depth_confidence=-1, width_confidence=-1 disables adaptive stopping for max accuracy.
    """
    from hloc import match_features

    conf = {
        "model": {
            "name": "lightglue",
            "features": "superpoint",
            "depth_confidence": depth_confidence,
            "width_confidence": width_confidence,
            "filter_threshold": 0.1,
        }
    }

    match_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.main(conf, pairs_path, feature_path, matches=match_path)


def run_hloc_reconstruction(
    sfm_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    feature_path: Path,
    match_path: Path,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    mapper_options: dict = None,
):
    """
    Run COLMAP incremental mapper via hloc reconstruction module.
    Returns the model path.
    """
    from hloc import reconstruction

    sfm_dir.mkdir(parents=True, exist_ok=True)

    mapper_opts = {
        "ba_global_max_num_iterations": 100,
        "ba_global_max_refinements": 5,
        "ba_local_max_num_iterations": 40,
        "multiple_models": False,
    }
    if mapper_options:
        mapper_opts.update(mapper_options)

    import pycolmap
    cam_mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO

    image_options = {
        "camera_model": camera_model,
    }

    model = reconstruction.main(
        sfm_dir,
        image_dir,
        pairs_path,
        feature_path,
        match_path,
        camera_mode=cam_mode,
        verbose=True,
        image_options=image_options,
        mapper_options=mapper_opts,
    )
    return model


def add_netvlad_pairs(
    image_dir: Path,
    pairs_path: Path,
    num_loc: int = 20,
    device: str = "cuda",
):
    """
    Add NetVLAD-based retrieval pairs for loop closure detection.
    Appends to existing pairs_path.
    """
    try:
        from hloc import extract_features, pairs_from_retrieval

        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = pairs_path.parent / "retrieval_features.h5"
        retrieval_pairs = pairs_path.parent / "retrieval_pairs.txt"

        extract_features.main(retrieval_conf, image_dir, feature_path=retrieval_path)
        pairs_from_retrieval.main(retrieval_path, retrieval_pairs, num_matched=num_loc)

        # Merge retrieval pairs into sequential pairs
        existing = set()
        if pairs_path.exists():
            existing = set(pairs_path.read_text().strip().split("\n"))

        new_pairs = retrieval_pairs.read_text().strip().split("\n")
        merged = existing | set(new_pairs)

        with open(pairs_path, "w") as f:
            f.write("\n".join(sorted(merged)))

        return len(merged)
    except Exception as e:
        print(f"[hloc_wrapper] NetVLAD retrieval failed (non-critical): {e}")
        return 0
