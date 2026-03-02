#!/usr/bin/env python3
"""
Verify all pipeline components are installed and working.
Returns exit code 0 on full pass, 1 if any critical component fails.
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEPS_DIR   = SCRIPT_DIR / "deps"

GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE   = "\033[0;34m"
NC     = "\033[0m"

PASS = f"{GREEN}  OK  {NC}"
FAIL = f"{RED} FAIL {NC}"
WARN = f"{YELLOW} WARN {NC}"

results = []

def check(name: str, fn, critical=True):
    try:
        val = fn()
        print(f"[{PASS}] {name:<40} {val}")
        results.append(("PASS", name, str(val)))
        return True
    except Exception as e:
        tag = FAIL if critical else WARN
        label = "FAIL" if critical else "WARN"
        print(f"[{tag}] {name:<40} ERROR: {e}")
        results.append((label, name, str(e)))
        return not critical  # non-critical: continue


# ─── Python ──────────────────────────────────────────────────────────────────
check("Python version",
    lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# ─── PyTorch + CUDA ──────────────────────────────────────────────────────────
def check_torch():
    import torch
    ver = torch.__version__
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        raise RuntimeError("CUDA not available in PyTorch")
    return f"{ver} | CUDA {torch.version.cuda}"

check("PyTorch + CUDA", check_torch)

def check_gpus():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_properties(i).name for i in range(n)]
    vrams = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(n)]
    for i, (name, vram) in enumerate(zip(names, vrams)):
        print(f"  {' ':40}  GPU {i}: {name} ({vram} GB)")
    return f"{n} GPU(s) detected"

check("GPU count + info", check_gpus)

# ─── COLMAP ──────────────────────────────────────────────────────────────────
def check_colmap():
    r = subprocess.run(["colmap", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    ver = r.stdout.strip() or r.stderr.strip()
    return ver.split("\n")[0]

check("COLMAP", check_colmap)

# ─── ffmpeg ───────────────────────────────────────────────────────────────────
def check_ffmpeg():
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg not found")
    line = r.stdout.split("\n")[0]
    return line.split("Copyright")[0].strip()

check("ffmpeg", check_ffmpeg)

# ─── diff-gaussian-rasterization ─────────────────────────────────────────────
def check_diff_gauss():
    import diff_gaussian_rasterization
    return f"OK (path: {diff_gaussian_rasterization.__file__})"

check("diff-gaussian-rasterization", check_diff_gauss)

# ─── simple-knn ───────────────────────────────────────────────────────────────
def check_simple_knn():
    import simple_knn
    return f"OK (path: {simple_knn.__file__})"

check("simple-knn", check_simple_knn)

# ─── gsplat ───────────────────────────────────────────────────────────────────
def check_gsplat():
    import gsplat
    return getattr(gsplat, "__version__", "installed")

check("gsplat", check_gsplat)

# ─── Nerfstudio ───────────────────────────────────────────────────────────────
def check_nerfstudio():
    import nerfstudio
    return getattr(nerfstudio, "__version__", "installed")

check("nerfstudio", check_nerfstudio)

# ─── hloc ─────────────────────────────────────────────────────────────────────
def check_hloc():
    from hloc import extract_features, match_features
    return "OK (SuperPoint+LightGlue available)"

check("hloc (SuperPoint+LightGlue)", check_hloc)

# ─── Depth-Anything-V2 ────────────────────────────────────────────────────────
def check_depth_anything():
    vitb_ckpt = DEPS_DIR / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitb.pth"
    da_dir    = DEPS_DIR / "Depth-Anything-V2"
    if not da_dir.exists():
        raise RuntimeError(f"Depth-Anything-V2 not cloned at {da_dir}")
    if not vitb_ckpt.exists():
        raise RuntimeError(f"ViTb checkpoint missing: {vitb_ckpt}")
    sys.path.insert(0, str(da_dir))
    from depth_anything_v2.dpt import DepthAnythingV2
    return f"OK (ViTb checkpoint: {vitb_ckpt.stat().st_size / 1e6:.0f} MB)"

check("Depth-Anything-V2 (ViTb)", check_depth_anything)

# ─── rembg ────────────────────────────────────────────────────────────────────
def check_rembg():
    import rembg
    return getattr(rembg, "__version__", "installed")

check("rembg", check_rembg, critical=False)

# ─── Open3D ───────────────────────────────────────────────────────────────────
def check_open3d():
    import open3d as o3d
    return o3d.__version__

check("Open3D", check_open3d)

# ─── COLMAP Vocabulary Tree ───────────────────────────────────────────────────
def check_vocab_tree():
    vt = DEPS_DIR / "vocab_trees" / "vocab_tree_flickr100K_words1M.bin"
    if not vt.exists():
        raise RuntimeError(f"Not found: {vt}")
    return f"OK ({vt.stat().st_size / 1e6:.0f} MB)"

check("COLMAP vocab tree (1M words)", check_vocab_tree, critical=False)

# ─── Other pip packages ───────────────────────────────────────────────────────
def check_pkg(pkg_name, import_name=None):
    import importlib
    m = importlib.import_module(import_name or pkg_name)
    return getattr(m, "__version__", "installed")

for pkg, imp in [
    ("numpy",       None),
    ("cv2",         "cv2"),
    ("scipy",       None),
    ("PIL",         "PIL"),
    ("imagehash",   None),
    ("lpips",       None),
    ("plyfile",     None),
    ("trimesh",     None),
    ("torchmetrics",None),
    ("tqdm",        None),
    ("yaml",        "yaml"),
]:
    check(f"pip: {pkg}", lambda i=imp or pkg: check_pkg(i, i), critical=(pkg in ("numpy","cv2","scipy","PIL")))

# ─── Summary ──────────────────────────────────────────────────────────────────
print()
total = len(results)
n_pass = sum(1 for r in results if r[0] == "PASS")
n_fail = sum(1 for r in results if r[0] == "FAIL")
n_warn = sum(1 for r in results if r[0] == "WARN")

print(f"{BLUE}══════════════════════════════════════════════{NC}")
print(f"  Verification: {GREEN}{n_pass} passed{NC}, "
      f"{YELLOW}{n_warn} warnings{NC}, {RED}{n_fail} failed{NC}  (total: {total})")
print(f"{BLUE}══════════════════════════════════════════════{NC}")

if n_fail > 0:
    print(f"\n{RED}Some critical checks FAILED. Run ./setup.sh to fix.{NC}")
    print("\nFailed components:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  - {r[1]}: {r[2]}")
    print()
    print("Common fixes:")
    print("  diff-gaussian-rasterization: nvcc version must match torch.version.cuda")
    print("  COLMAP: Run 'conda install -c conda-forge colmap -y'")
    print("  hloc:   cd deps/Hierarchical-Localization && pip install -e .")
    print("  Depth-Anything-V2: Run setup.sh to re-download checkpoint")
    sys.exit(1)
else:
    print(f"\n{GREEN}All critical components OK. Pipeline is ready.{NC}")
    sys.exit(0)
