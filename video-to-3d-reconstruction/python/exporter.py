"""
exporter.py — Exports trained Splatfacto model as a 3D Gaussian Splat PLY file.
"""
import subprocess
import shutil
import json
import glob
from pathlib import Path


def _emit(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    print(json.dumps(payload), flush=True)


def _ply_stats(ply_path: str) -> dict:
    """Read basic stats from PLY file header without Open3D."""
    stats = {"size_mb": round(Path(ply_path).stat().st_size / 1e6, 2)}
    try:
        with open(ply_path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                header += f.read(512)
        header_text = header.decode("ascii", errors="ignore")
        for line in header_text.splitlines():
            if line.startswith("element vertex"):
                stats["num_gaussians"] = int(line.split()[-1])
                break
    except Exception:
        pass
    return stats


def run(config_yml: str, output_dir: str) -> str:
    """
    Export the trained Splatfacto model as splat.ply.
    Returns path to the exported PLY file.
    """
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    _emit("log", line=f"Exporting 3D Gaussian Splat from: {config_yml}")
    _emit("log", line=f"Export directory: {export_dir}")

    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_yml),
        "--output-dir", str(export_dir),
    ]

    _emit("log", line=f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            _emit("log", line=line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"ns-export exited with code {process.returncode}.")

    # ns-export names output splat.ply inside export_dir
    ply_path = export_dir / "splat.ply"

    # Fallback: search for any .ply in export_dir
    if not ply_path.exists():
        ply_files = list(export_dir.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No PLY file found in {export_dir}")
        ply_path = ply_files[0]
        target = export_dir / "splat.ply"
        shutil.copy(str(ply_path), str(target))
        ply_path = target

    stats = _ply_stats(str(ply_path))

    # Save metadata
    meta = {
        "ply_path": str(ply_path),
        "size_mb": stats.get("size_mb"),
        "num_gaussians": stats.get("num_gaussians"),
    }
    with open(export_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _emit("log", line=f"3D export complete: {ply_path}")
    _emit("log", line=f"  Size: {stats.get('size_mb', '?')} MB | Gaussians: {stats.get('num_gaussians', '?')}")

    return str(ply_path)
