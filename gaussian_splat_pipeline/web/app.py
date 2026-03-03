#!/usr/bin/env python3
"""
Gaussian Splatting Pipeline — Web Frontend
Upload a video → run the full pipeline → view interactive 3D splat.
Real-time log streaming via Server-Sent Events (SSE).
"""

import os
import sys
import json
import uuid
import time
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response, stream_with_context,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent.resolve()
PIPELINE_DIR = APP_DIR.parent.resolve()
UPLOAD_DIR   = PIPELINE_DIR / "web" / "uploads"
JOBS_DIR     = PIPELINE_DIR / "web" / "jobs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB max upload

# ── In-memory job registry ────────────────────────────────────────────────────
jobs: dict[str, dict] = {}

STAGES = [
    {"id": 0, "name": "Pre-flight Checks"},
    {"id": 1, "name": "Frame Extraction & Filtering"},
    {"id": 2, "name": "Depth Map Generation"},
    {"id": 3, "name": "Background Masking"},
    {"id": 4, "name": "COLMAP Pose Estimation"},
    {"id": 5, "name": "Gaussian Splatting Training"},
    {"id": 6, "name": "Quality Evaluation"},
    {"id": 7, "name": "Export (PLY, Mesh)"},
]


# ═════════════════════════════════════════════════════════════════════════════
#  Routes — Pages
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", stages=STAGES)


@app.route("/viewer/<job_id>")
def viewer(job_id):
    job = jobs.get(job_id)
    if not job:
        return "Job not found", 404
    return render_template("viewer.html", job_id=job_id, job=job)


# ═════════════════════════════════════════════════════════════════════════════
#  Routes — API
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept video upload and start the pipeline."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Parse options
    config   = request.form.get("config", "default")
    fps      = request.form.get("fps", "3")
    method   = request.form.get("method", "original")
    mask     = request.form.get("mask", "false")
    num_gpus = request.form.get("num_gpus", "4")

    # Create job
    job_id    = uuid.uuid4().hex[:12]
    job_dir   = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    out_dir   = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    ext       = Path(video.filename).suffix or ".mp4"
    vid_path  = input_dir / f"video{ext}"
    video.save(str(vid_path))
    vid_size  = vid_path.stat().st_size / (1024 * 1024)

    config_path = PIPELINE_DIR / "configs" / f"{config}.yaml"
    if not config_path.exists():
        config_path = PIPELINE_DIR / "configs" / "default.yaml"

    job = {
        "id":          job_id,
        "status":      "queued",
        "stage":       -1,
        "stage_name":  "Queued",
        "video_name":  video.filename,
        "video_size":  f"{vid_size:.1f} MB",
        "config":      config,
        "fps":         fps,
        "method":      method,
        "mask":        mask,
        "num_gpus":    num_gpus,
        "created_at":  datetime.now().isoformat(),
        "log_file":    str(out_dir / "logs" / "pipeline.log"),
        "output_dir":  str(out_dir),
        "video_path":  str(vid_path),
        "config_path": str(config_path),
        "pid":         None,
        "error":       None,
        "metrics":     None,
    }
    jobs[job_id] = job

    # Start pipeline in background thread
    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/jobs")
def api_jobs():
    """List all jobs."""
    return jsonify([
        {k: v for k, v in j.items() if k != "pid"}
        for j in sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    ])


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    safe = {k: v for k, v in job.items() if k != "pid"}
    return jsonify(safe)


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    if job["pid"]:
        try:
            os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
        except Exception:
            pass
    job["status"] = "cancelled"
    job["stage_name"] = "Cancelled"
    return jsonify({"status": "cancelled"})


@app.route("/api/jobs/<job_id>/logs")
def api_logs_sse(job_id):
    """Stream pipeline logs as Server-Sent Events."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    def generate():
        log_path = Path(job["log_file"])
        # Wait for log file to appear
        waited = 0
        while not log_path.exists() and waited < 30:
            time.sleep(0.5)
            waited += 0.5
            yield f"data: {json.dumps({'type': 'waiting', 'message': 'Waiting for pipeline to start...'})}\n\n"

        if not log_path.exists():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Log file never appeared'})}\n\n"
            return

        with open(log_path, "r") as f:
            # Tail -f the log
            while True:
                line = f.readline()
                if line:
                    payload = {
                        "type":       "log",
                        "message":    line.rstrip(),
                        "stage":      job.get("stage", -1),
                        "stage_name": job.get("stage_name", ""),
                        "status":     job.get("status", ""),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    # If pipeline finished, send final status and close
                    if job["status"] in ("completed", "failed", "cancelled"):
                        payload = {
                            "type":    "done",
                            "status":  job["status"],
                            "metrics": job.get("metrics"),
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        return
                    time.sleep(0.3)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Serve output files (PLY, splat, etc.) ─────────────────────────────────────

@app.route("/api/jobs/<job_id>/files/<path:filename>")
def api_serve_file(job_id, filename):
    job = jobs.get(job_id)
    if not job:
        return "Not found", 404
    return send_from_directory(job["output_dir"], filename)


@app.route("/api/jobs/<job_id>/exports")
def api_exports(job_id):
    """List files in exports/ directory."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    exports = Path(job["output_dir"]) / "exports"
    if not exports.exists():
        return jsonify([])

    files = []
    for f in sorted(exports.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                "url": f"/api/jobs/{job_id}/files/exports/{f.name}",
            })
    return jsonify(files)


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline Runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str):
    """Run the bash pipeline in a subprocess, updating job state from log."""
    job = jobs[job_id]
    job["status"] = "running"
    job["stage"]  = 0
    job["stage_name"] = "Starting..."

    cmd = [
        "bash", str(PIPELINE_DIR / "run_pipeline.sh"),
        "--input",    job["video_path"],
        "--output",   job["output_dir"],
        "--config",   job["config_path"],
        "--fps",      job["fps"],
        "--method",   job["method"],
        "--mask",     job["mask"],
        "--num-gpus", job["num_gpus"],
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force GCC 11 for CUDA JIT compilation (gsplat, etc.)
    # GCC 13 is incompatible with PyTorch 2.1.2 pybind11 headers
    env["CC"]  = "gcc-11"
    env["CXX"] = "g++-11"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid,
            cwd=str(PIPELINE_DIR),
        )
        job["pid"] = proc.pid

        # Monitor output to track current stage
        for line in proc.stdout:
            line = line.rstrip()
            _update_stage_from_line(job, line)

        proc.wait()

        if proc.returncode == 0:
            job["status"]     = "completed"
            job["stage"]      = 7
            job["stage_name"] = "Complete!"
            _load_metrics(job)
        else:
            job["status"]     = "failed"
            job["error"]      = f"Pipeline exited with code {proc.returncode}"
            job["stage_name"] = f"Failed (stage {job['stage']})"

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)
        job["stage_name"] = f"Error: {e}"

    finally:
        job["pid"] = None


def _update_stage_from_line(job: dict, line: str):
    """Parse pipeline output to detect current stage."""
    lower = line.lower()
    for stage in STAGES:
        markers = [
            f"stage {stage['id']}:",
            f"stage {stage['id']} ",
        ]
        for marker in markers:
            if marker in lower:
                job["stage"]      = stage["id"]
                job["stage_name"] = stage["name"]
                break


def _load_metrics(job: dict):
    """Load quality metrics if available."""
    metrics_path = Path(job["output_dir"]) / "quality_report" / "metrics_summary.json"
    if metrics_path.exists():
        try:
            job["metrics"] = json.load(open(metrics_path))
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  🌐  Gaussian Splatting Pipeline UI")
    print(f"  📡  http://{args.host}:{args.port}")
    print(f"  📂  Pipeline: {PIPELINE_DIR}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
