#!/usr/bin/env python3
"""
Stage 7: Export All Formats + Final Report
- Gaussian PLY (trained model)
- Dense point cloud (COLMAP patch_match_stereo + stereo_fusion) — multi-GPU
- Poisson mesh (COLMAP poisson_mesher + Open3D alternative)
- Camera poses JSON
- Final reconstruction_report.txt
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger        import get_pipeline_logger
from utils.seed_utils    import set_all_seeds
from utils.config_loader import load_config
from utils.gpu_utils     import detect_gpus, gpu_index_str


def parse_args():
    p = argparse.ArgumentParser(description="Stage 7: Export and final report")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method",     default="original", choices=["original","splatfacto","both"])
    p.add_argument("--skip-dense", default="false")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--num-gpus",   type=int, default=4)
    return p.parse_args()


def bool_arg(val) -> bool:
    return str(val).lower() in ("true","1","yes")


def run_cmd(cmd: list, logger, check=True, **kwargs) -> int:
    logger.info(f"  CMD: {' '.join(str(c) for c in cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return result.returncode


# ─── Gaussian PLY Export ──────────────────────────────────────────────────────

def export_gaussian_ply(model_dir: Path, exports_dir: Path, method: str, logger):
    """Copy or convert the trained Gaussian PLY to exports/."""
    exports_dir.mkdir(parents=True, exist_ok=True)

    if method in ("original", "both"):
        # Original 3DGS stores point_cloud.ply in model/original/point_cloud/iteration_N/
        candidates = sorted((model_dir / "original").rglob("point_cloud.ply"),
                            key=lambda p: int(p.parent.name.split("_")[-1])
                            if p.parent.name.startswith("iteration_") else 0)
        if candidates:
            best = candidates[-1]  # highest iteration
            dest = exports_dir / "gaussians.ply"
            shutil.copy2(str(best), str(dest))
            size_mb = dest.stat().st_size / 1e6
            logger.info(f"  Gaussian PLY: {dest} ({size_mb:.1f} MB)")
            return dest
        else:
            logger.warning("  No point_cloud.ply found in original 3DGS model.")

    if method in ("splatfacto", "both"):
        # Nerfstudio splatfacto stores .ply in model/splatfacto/.../splat.ply
        candidates = sorted((model_dir / "splatfacto").rglob("*.ply"))
        if candidates:
            dest = exports_dir / "gaussians_splatfacto.ply"
            shutil.copy2(str(candidates[-1]), str(dest))
            logger.info(f"  Gaussian PLY (splatfacto): {dest}")
            return dest

    return None


# ─── Dense Point Cloud (COLMAP MVS) ──────────────────────────────────────────

def run_dense_reconstruction(
    undist_dir: Path,
    exports_dir: Path,
    cfg: dict,
    num_gpus: int,
    logger,
):
    """
    Run COLMAP patch_match_stereo + stereo_fusion for dense point cloud.
    Multi-GPU: uses all available GPUs.
    """
    export_cfg = cfg.get("export", {})
    gpu_idx    = gpu_index_str(num_gpus)
    dense_ply  = exports_dir / "dense_pointcloud.ply"

    if not undist_dir.exists():
        logger.warning(f"Undistorted dir not found: {undist_dir}. Skipping dense reconstruction.")
        return

    logger.info(f"  patch_match_stereo (GPUs: {gpu_idx})...")
    t_start = time.time()

    try:
        run_cmd([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(undist_dir),
            "--PatchMatchStereo.gpu_index", gpu_idx,
            "--PatchMatchStereo.geom_consistency",
                "true" if export_cfg.get("patch_match_geom_consistency", True) else "false",
            "--PatchMatchStereo.max_image_size", "4096",
            "--PatchMatchStereo.window_radius",
                str(export_cfg.get("patch_match_window_radius", 7)),
            "--PatchMatchStereo.num_iterations",
                str(export_cfg.get("patch_match_num_iterations", 5)),
        ], logger)

        logger.info("  stereo_fusion...")
        run_cmd([
            "colmap", "stereo_fusion",
            "--workspace_path",     str(undist_dir),
            "--output_path",        str(dense_ply),
            "--StereoFusion.min_num_pixels",
                str(export_cfg.get("stereo_fusion_min_num_pixels", 5)),
            "--StereoFusion.max_reproj_error",
                str(export_cfg.get("stereo_fusion_max_reproj_error", 2.0)),
        ], logger)

        t_elapsed = time.time() - t_start
        if dense_ply.exists():
            size_mb = dense_ply.stat().st_size / 1e6
            logger.info(f"  Dense point cloud: {dense_ply} ({size_mb:.1f} MB) in {t_elapsed:.0f}s")
    except Exception as e:
        logger.error(f"  Dense reconstruction failed: {e}")
        logger.warning("  Skipping dense cloud. The Gaussian PLY is still the primary output.")


# ─── Poisson Mesh ─────────────────────────────────────────────────────────────

def run_poisson_mesh(undist_dir: Path, dense_ply: Path, exports_dir: Path, cfg: dict, logger):
    """Run COLMAP Poisson meshing on the dense point cloud."""
    export_cfg = cfg.get("export", {})
    mesh_ply   = exports_dir / "mesh_colmap.ply"

    if not dense_ply or not dense_ply.exists():
        logger.warning("  Dense point cloud not found. Skipping Poisson mesh.")
        return

    try:
        run_cmd([
            "colmap", "poisson_mesher",
            "--input_path",  str(dense_ply),
            "--output_path", str(mesh_ply),
            "--PoissonMeshing.trim", str(export_cfg.get("poisson_trim", 7)),
        ], logger)
        if mesh_ply.exists():
            size_mb = mesh_ply.stat().st_size / 1e6
            logger.info(f"  COLMAP Poisson mesh: {mesh_ply} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"  COLMAP Poisson mesh failed: {e}")


def run_open3d_mesh(gaussian_ply: Path, exports_dir: Path, cfg: dict, logger):
    """
    Alternative: Poisson mesh from Gaussian centers using Open3D.
    Useful when COLMAP dense reconstruction is skipped.
    """
    if not gaussian_ply or not gaussian_ply.exists():
        return

    export_cfg = cfg.get("export", {})
    mesh_path  = exports_dir / "mesh_open3d.ply"

    try:
        import open3d as o3d
        from plyfile import PlyData
        import numpy as np

        logger.info("  Building Open3D mesh from Gaussian centers...")
        ply = PlyData.read(str(gaussian_ply))
        verts = ply.elements[0].data
        xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        # Screened Poisson reconstruction
        depth = export_cfg.get("open3d_poisson_depth", 9)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        # Trim low-density vertices
        min_density = export_cfg.get("open3d_poisson_min_density", 0.01)
        import numpy as np
        densities_arr = np.asarray(densities)
        vert_mask = densities_arr > np.quantile(densities_arr, min_density)
        mesh = mesh.select_by_index(np.where(vert_mask)[0])

        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        if mesh_path.exists():
            size_mb = mesh_path.stat().st_size / 1e6
            logger.info(f"  Open3D Poisson mesh: {mesh_path} ({size_mb:.1f} MB, "
                        f"{len(mesh.triangles):,} triangles)")
    except Exception as e:
        logger.warning(f"  Open3D mesh failed: {e}")


# ─── Camera Poses JSON ────────────────────────────────────────────────────────

def export_camera_poses(colmap_dir: Path, exports_dir: Path, logger):
    """Export camera poses from COLMAP sparse model as JSON."""
    from utils.colmap_parser import read_images_binary, read_cameras_binary
    import numpy as np

    sparse_candidates = [
        colmap_dir / "sparse" / "0",
        colmap_dir / "sparse_hloc" / "0",
    ]
    sparse_model = next((p for p in sparse_candidates if p.exists()), None)
    if not sparse_model:
        logger.warning("  Sparse model not found for camera export.")
        return

    try:
        images  = read_images_binary(str(sparse_model / "images.bin"))
        cameras = read_cameras_binary(str(sparse_model / "cameras.bin"))

        cam_data = []
        for img in sorted(images.values(), key=lambda x: x["name"]):
            w, x, y, z = img["qvec"]
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
            ])
            t = np.array(img["tvec"])
            C = (-R.T @ t).tolist()
            cam_data.append({
                "image_name": img["name"],
                "camera_id":  img["camera_id"],
                "quaternion_wxyz": list(img["qvec"]),
                "translation": list(img["tvec"]),
                "camera_center_world": C,
                "rotation_matrix": R.tolist(),
            })

        out_path = exports_dir / "camera_poses.json"
        with open(out_path, "w") as f:
            json.dump({"cameras": cam_data, "num_cameras": len(cam_data)}, f, indent=2)
        logger.info(f"  Camera poses: {out_path} ({len(cam_data)} cameras)")
    except Exception as e:
        logger.warning(f"  Camera pose export failed: {e}")


# ─── Final Report ─────────────────────────────────────────────────────────────

def generate_final_report(output_dir: Path, logger):
    """Generate human-readable reconstruction_report.txt."""
    report_lines = [
        "=" * 70,
        " GAUSSIAN SPLATTING RECONSTRUCTION REPORT",
        "=" * 70,
        "",
    ]

    def add(key, val):
        report_lines.append(f"  {key:<35} {val}")

    # Video info
    video_meta_path = output_dir / "logs" / "video_info.json"
    if video_meta_path.exists():
        vm = json.load(open(video_meta_path))
        report_lines += ["── Input Video ──────────────────────────────────────────", ""]
        add("File:", Path(vm.get("path","")).name)
        add("Resolution:", f"{vm.get('width','?')}x{vm.get('height','?')}")
        add("FPS:", vm.get("fps","?"))
        add("Duration:", f"{float(vm.get('duration_s',0)):.1f}s")
        add("Codec:", vm.get("codec","?"))
        report_lines.append("")

    # Frame stats
    quality_report = output_dir / "logs" / "frame_quality_report.csv"
    if quality_report.exists():
        import csv
        rows = list(csv.DictReader(open(quality_report)))
        total = len(rows)
        kept  = sum(1 for r in rows if r.get("status") == "KEPT")
        report_lines += ["── Frame Extraction ─────────────────────────────────────", ""]
        add("Total extracted:", str(total))
        add("Passed filters:", str(kept))
        add("Rejection rate:", f"{(1-kept/max(total,1)):.1%}")
        report_lines.append("")

    # COLMAP stats
    colmap_stats_path = output_dir / "logs" / "colmap_stats.json"
    if colmap_stats_path.exists():
        cs = json.load(open(colmap_stats_path))
        report_lines += ["── COLMAP Reconstruction ────────────────────────────────", ""]
        add("Registered images:", f"{cs.get('num_registered','?')} / {cs.get('total_images','?')}")
        add("Registration rate:", f"{cs.get('registration_rate',0):.1%}")
        add("3D points:", f"{cs.get('num_3d_points',0):,}")
        add("Mean reprojection error:", f"{cs.get('mean_reprojection_error',0):.3f} px")
        add("Mean track length:", f"{cs.get('mean_track_length',0):.1f}")
        add("Status:", cs.get("status","?"))
        report_lines.append("")

    # Training
    train_path = output_dir / "logs" / "training_results.json"
    if train_path.exists():
        tr = json.load(open(train_path))
        report_lines += ["── Training ─────────────────────────────────────────────", ""]
        for r in tr:
            add("Method:", r.get("method","?"))
            add("Iterations:", str(r.get("iterations","?")))
            add("Gaussians:", f"{r.get('num_gaussians',0):,}")
            add("Training time:", f"{r.get('training_time_seconds',0)//60}m {r.get('training_time_seconds',0)%60}s")
            add("Depth regularization:", str(r.get("depth_regularization","?")))
            add("Antialiasing:", str(r.get("antialiasing","?")))
            add("Exposure comp:", str(r.get("exposure_compensation","?")))
        report_lines.append("")

    # Quality metrics
    metrics_path = output_dir / "quality_report" / "metrics_summary.json"
    if metrics_path.exists():
        m = json.load(open(metrics_path))
        report_lines += ["── Quality Metrics ──────────────────────────────────────", ""]
        add("Method:", m.get("method","?"))
        add("PSNR:", f"{m.get('psnr_mean',0):.2f} dB ± {m.get('psnr_std',0):.2f}")
        add("SSIM:", f"{m.get('ssim_mean',0):.4f} ± {m.get('ssim_std',0):.4f}")
        if m.get("lpips_mean",-1) >= 0:
            add("LPIPS:", f"{m.get('lpips_mean',0):.4f} ± {m.get('lpips_std',0):.4f}")
        add("Quality grade:", m.get("psnr_quality","?"))
        add("Variance status:", m.get("variance_status","?"))
        report_lines.append("")

    # Output files
    exports_dir = output_dir / "exports"
    report_lines += ["── Output Files ─────────────────────────────────────────", ""]
    for fname in sorted(exports_dir.glob("*")):
        size_mb = fname.stat().st_size / 1e6
        add(fname.name + ":", f"{size_mb:.1f} MB")
    report_lines.append("")

    # Seeds
    seeds_path = output_dir / "logs" / "seeds.json"
    if seeds_path.exists():
        seeds = json.load(open(seeds_path))
        report_lines += ["── Reproducibility ──────────────────────────────────────", ""]
        add("Master seed:", str(seeds.get("master_seed","?")))
        add("Config snapshot:", str(output_dir / "config_snapshot.yaml"))
    report_lines.append("")
    report_lines += ["=" * 70, ""]

    report_text = "\n".join(report_lines)
    report_path = output_dir / "reconstruction_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"  Final report: {report_path}")
    print(report_text)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg        = load_config(args.config)
    seed       = cfg.get("pipeline", {}).get("seed", 42)
    set_all_seeds(seed, True)

    logger = get_pipeline_logger(output_dir, "stage7")
    logger.info("=" * 60)
    logger.info("Stage 7: Export + Final Report")
    logger.info("=" * 60)

    export_cfg  = cfg.get("export", {})
    model_dir   = output_dir / "model"
    colmap_dir  = output_dir / "colmap"
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    skip_dense  = bool_arg(args.skip_dense) or not export_cfg.get("dense_pointcloud", True)

    gpus     = detect_gpus()
    num_gpus = min(args.num_gpus, max(len(gpus), 1))

    # ── 1. Export Gaussian PLY ────────────────────────────────────────────────
    logger.info("[1] Exporting Gaussian PLY...")
    gaussian_ply = export_gaussian_ply(model_dir, exports_dir, args.method, logger)

    # ── 2. Dense Point Cloud ──────────────────────────────────────────────────
    dense_ply = exports_dir / "dense_pointcloud.ply"
    if not skip_dense:
        logger.info(f"[2] Dense point cloud (COLMAP MVS, {num_gpus} GPUs)...")
        undist_dir = output_dir / "colmap" / "undistorted"
        run_dense_reconstruction(undist_dir, exports_dir, cfg, num_gpus, logger)
    else:
        logger.info("[2] Dense reconstruction skipped (--skip-dense or disabled in config).")

    # ── 3. Poisson Mesh ───────────────────────────────────────────────────────
    if export_cfg.get("poisson_mesh", True):
        logger.info("[3] Poisson mesh reconstruction...")
        if dense_ply.exists():
            undist_dir = output_dir / "colmap" / "undistorted"
            run_poisson_mesh(undist_dir, dense_ply, exports_dir, cfg, logger)
        # Always try Open3D mesh from Gaussian centers
        run_open3d_mesh(gaussian_ply, exports_dir, cfg, logger)
    else:
        logger.info("[3] Mesh skipped (disabled in config).")

    # ── 4. Camera Poses JSON ──────────────────────────────────────────────────
    if export_cfg.get("camera_poses_json", True):
        logger.info("[4] Exporting camera poses...")
        export_camera_poses(colmap_dir, exports_dir, logger)

    # ── 5. Final Report ───────────────────────────────────────────────────────
    logger.info("[5] Generating final reconstruction report...")
    generate_final_report(output_dir, logger)

    # Print output summary
    logger.info("\n  Export summary:")
    for f in sorted(exports_dir.glob("*")):
        logger.info(f"    {f.name}: {f.stat().st_size/1e6:.1f} MB")

    logger.info("=" * 60)
    logger.info("Stage 7 COMPLETE. Reconstruction finished!")
    logger.info(f"  All outputs: {exports_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
