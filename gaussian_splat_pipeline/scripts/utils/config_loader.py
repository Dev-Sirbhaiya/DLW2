"""
YAML config loading with CLI override support.
Merges nested config dicts: base config → CLI overrides.
"""

import yaml
import copy
from pathlib import Path


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, cli_overrides: dict = None) -> dict:
    """Load YAML config and apply any CLI overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if cli_overrides:
        cfg = deep_update(cfg, cli_overrides)

    return cfg


def save_config_snapshot(cfg: dict, output_dir: Path):
    """Save frozen copy of config used for this run."""
    out = Path(output_dir) / "config_snapshot.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def get(cfg: dict, *keys, default=None):
    """Safely get nested config value: get(cfg, 'training', 'iterations', default=30000)"""
    cur = cfg
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
        if cur is None:
            return default
    return cur
