"""
GPU detection, VRAM queries, and multi-GPU dispatch utilities.
"""

import os
import json
import subprocess
from typing import List, Tuple


def detect_gpus() -> List[dict]:
    """
    Return list of GPU info dicts:
      [{"index": 0, "name": "RTX 4090", "vram_gb": 24.0}, ...]
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": p.name,
                "vram_gb": round(p.total_memory / 1e9, 1),
                "compute_capability": f"{p.major}.{p.minor}",
            })
        return gpus
    except Exception:
        return []


def get_best_training_gpu(gpus: List[dict]) -> int:
    """Return GPU index with most VRAM (for single-GPU training)."""
    if not gpus:
        return 0
    return max(gpus, key=lambda g: g["vram_gb"])["index"]


def shard_list(items: list, num_shards: int) -> List[list]:
    """Split a list into N roughly equal shards for multi-GPU dispatch."""
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return shards


def gpu_index_str(num_gpus: int) -> str:
    """Returns '0,1,2,3' style string for COLMAP GPU flags."""
    return ",".join(str(i) for i in range(num_gpus))


def set_cuda_visible(devices: str = "0,1,2,3"):
    """Set CUDA_VISIBLE_DEVICES environment variable."""
    os.environ["CUDA_VISIBLE_DEVICES"] = devices


def save_gpu_info(output_dir, gpus: List[dict]):
    """Save GPU info to logs/gpu_info.json."""
    import os
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    with open(f"{output_dir}/logs/gpu_info.json", "w") as f:
        json.dump(gpus, f, indent=2)


def print_gpu_table(gpus: List[dict], logger=None):
    """Pretty-print GPU table."""
    def pr(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    pr(f"  {'IDX':<5} {'NAME':<35} {'VRAM':>8}  {'COMPUTE':>10}")
    pr(f"  {'-'*5} {'-'*35} {'-'*8}  {'-'*10}")
    for g in gpus:
        pr(f"  {g['index']:<5} {g['name']:<35} {g['vram_gb']:>6.1f} GB  "
           f"  {g.get('compute_capability', 'N/A'):>8}")


def get_driver_cuda_version() -> Tuple[str, str]:
    """Return (driver_version, cuda_version) from nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        driver = r.stdout.strip().split("\n")[0].strip() if r.returncode == 0 else "unknown"
        r2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        # CUDA version from nvidia-smi top line
        r3 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        import re
        m = re.search(r"CUDA Version: (\S+)", r3.stdout)
        cuda = m.group(1) if m else "unknown"
        return driver, cuda
    except Exception:
        return "unknown", "unknown"
