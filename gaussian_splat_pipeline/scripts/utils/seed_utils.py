"""
Deterministic seed management for reproducible runs.
Saves seed state to logs/seeds.json.
"""

import os
import json
import random
from pathlib import Path


def set_all_seeds(seed: int = 42, deterministic: bool = True):
    """Set seeds for Python, NumPy, and PyTorch. Optionally force CUDNN determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_seeds(output_dir: Path, seed: int, extra: dict = None):
    """Save all seed values to logs/seeds.json."""
    seeds_file = Path(output_dir) / "logs" / "seeds.json"
    seeds_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "master_seed": seed,
        "PYTHONHASHSEED": seed,
        "numpy_seed": seed,
        "torch_manual_seed": seed,
        "torch_cuda_seed_all": seed,
    }
    if extra:
        data.update(extra)

    with open(seeds_file, "w") as f:
        json.dump(data, f, indent=2)
