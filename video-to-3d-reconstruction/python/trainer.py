"""
trainer.py — Runs ns-train splatfacto and returns the path to the trained config.yml.
Supports multi-GPU training via CUDA_VISIBLE_DEVICES and torch.distributed.
"""
import subprocess
import json
import glob
import os
import torch
from pathlib import Path


def _emit(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    print(json.dumps(payload), flush=True)


def detect_gpus() -> list[int]:
    """Return list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def find_config(train_output_dir: str, method: str = "splatfacto") -> str:
    """Find the config.yml produced by ns-train inside its dated subdirectory."""
    pattern = str(Path(train_output_dir) / method / "*" / "config.yml")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find config.yml in {train_output_dir}/{method}/*/config.yml. "
            "Training may have failed."
        )
    return matches[-1]  # most recent


def run(processed_data_dir: str, output_dir: str, max_iterations: int = 30000) -> str:
    """
    Train Splatfacto model using all available GPUs. Returns path to config.yml.

    Multi-GPU strategy:
    - Nerfstudio splatfacto uses a single-process trainer internally, so we
      expose all 4 GPUs via CUDA_VISIBLE_DEVICES and let ns-train use GPU 0
      (primary) for the Gaussian rasterization, while also enabling the
      torch.distributed DataParallel path Nerfstudio supports via --machine.num-devices.
    """
    train_output_dir = Path(output_dir) / "training"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    gpus = detect_gpus()
    num_gpus = len(gpus)
    gpu_ids_str = ",".join(str(g) for g in gpus) if gpus else "0"

    _emit("log", line=f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(g) for g in gpus] if gpus else ['CPU']}")
    _emit("log", line=f"CUDA_VISIBLE_DEVICES={gpu_ids_str}")
    _emit("log", line=f"Starting Splatfacto training — {max_iterations} iterations, {num_gpus} GPU(s)...")
    _emit("log", line=f"Data: {processed_data_dir}")
    _emit("log", line=f"Training output: {train_output_dir}")

    # Nerfstudio uses --machine.num-devices for distributed training
    # and --machine.device-type to select cuda
    cmd = [
        "ns-train", "splatfacto",
        "--data", str(processed_data_dir),
        "--output-dir", str(train_output_dir),
        "--max-num-iterations", str(max_iterations),
        "--machine.num-devices", str(max(1, num_gpus)),
        "--machine.device-type", "cuda" if num_gpus > 0 else "cpu",
        "--pipeline.model.cull-alpha-thresh", "0.005",
        "--pipeline.model.continue-cull-post-densification", "True",
        "--viewer.quit-on-train-completion", "True",
        "--steps-per-save", "2000",
        "--steps-per-eval-image", "500",
    ]

    _emit("log", line=f"Command: {' '.join(cmd)}")

    # Expose all GPUs in the subprocess environment
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids_str, "PYTHONUNBUFFERED": "1"}

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    log_interval = 50
    line_count = 0

    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        line_count += 1
        if any(kw in line.lower() for kw in ["step", "loss", "psnr", "iter", "error", "warning", "done", "saved", "gpu", "device"]):
            _emit("log", line=line)
        elif line_count % log_interval == 0:
            _emit("log", line=line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"ns-train exited with code {process.returncode}. "
            f"Running with {num_gpus} GPU(s) — check VRAM (need ≥8GB per GPU) and Nerfstudio install."
        )

    config_path = find_config(str(train_output_dir))
    _emit("log", line=f"Training complete. Config: {config_path}")
    return config_path
