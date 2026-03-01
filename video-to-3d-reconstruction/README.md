# 3D Reconstruction Studio

**Video → Interactive 3D Gaussian Splat + 2D Novel-View Video**

Powered by [Nerfstudio](https://docs.nerf.studio/) · Splatfacto · React · Node.js

---

## What This Does

Upload any video and the system automatically:

1. Extracts frames and runs **COLMAP** to recover camera positions
2. Trains a **3D Gaussian Splatting** model (Splatfacto) — supports **1–4 GPUs**
3. Exports an interactive **3D splat** (`.ply`) you can orbit in the browser
4. Renders a **2D novel-view video** (spiral camera path around the scene)

---

## Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS          | Ubuntu 22.04 | Other Debian/Ubuntu may work |
| GPU         | 1× NVIDIA GPU, 8 GB VRAM | 4× GPUs auto-detected |
| NVIDIA drivers | 520+ | Install before running setup |
| Disk space  | 20 GB free | For models, exports, venv |
| RAM         | 16 GB | 32 GB recommended |

---

## Quick Start (3 Steps)

### Step 1 — Install everything

```bash
git clone <this-repo>
cd video-to-3d-reconstruction
chmod +x setup.sh
bash setup.sh
```

This **one command** installs:
- System packages (`ffmpeg`, `colmap`, build tools)
- A Python 3.10 virtual environment (`./venv/`)
- PyTorch 2.1 with CUDA 11.8
- Nerfstudio and all Python dependencies
- Node.js 20 and all npm packages for both backend and frontend

**Takes 10–20 minutes** depending on internet speed. Only needed once.

---

### Step 2 — Start the application

You need **two terminal windows**.

**Terminal 1 — Backend:**
```bash
cd video-to-3d-reconstruction
source activate.sh          # activates the Python venv
cd backend
node server.js
```

**Terminal 2 — Frontend:**
```bash
cd video-to-3d-reconstruction/frontend
npm run dev
```

---

### Step 3 — Open the app

Open your browser and go to:

```
http://localhost:5173
```

---

## Using the App

### Upload
- Drag and drop your video file, or click to browse
- Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Maximum size: 2 GB

### Choose Mode
| Mode | What you get |
|------|-------------|
| **3D Reconstruction** | A `.ply` 3D Gaussian Splat file, viewable interactively in-browser |
| **2D Novel Views** | A `.mp4` video rendered along a spiral path around the scene |
| **Both (Recommended)** | Both outputs from a single training run |

### Advanced Settings (optional)
- **Training Iterations**: `30000` = full quality (15–30 min). `10000` = quick test (5–8 min).
- **COLMAP Frame Target**: How many frames to extract from the video for SfM (default 300).

### Progress
Watch all 6 pipeline stages with live log output:
1. Validate Video
2. COLMAP Structure-from-Motion
3. Train Splatfacto
4. Export 3D Gaussian Splat
5. Render 2D Novel Views
6. Complete

### Results
- **3D viewer**: Interactive orbit/zoom/pan using Three.js + Gaussian Splats 3D
- **2D viewer**: Video player + expandable frame contact sheet
- **Split view**: See both side-by-side
- **Download**: `.ply`, `.mp4`, `contact_sheet.png`

---

## Multi-GPU Support

The system **automatically detects and uses all available GPUs**.

With 4 GPUs, training is distributed across all of them via Nerfstudio's
`--machine.num-devices` flag + `CUDA_VISIBLE_DEVICES=0,1,2,3`.

GPU info is shown in the setup output:
```
GPU 0: NVIDIA RTX 4090 (24576 MB)
GPU 1: NVIDIA RTX 4090 (24576 MB)
GPU 2: NVIDIA RTX 4090 (24576 MB)
GPU 3: NVIDIA RTX 4090 (24576 MB)
```

---

## Tips for Good Results

| Tip | Why |
|-----|-----|
| Film slowly with smooth motion | COLMAP needs good feature matching |
| Orbit fully around the object | Gives complete 360° coverage |
| Keep scenes well-lit | Avoids dark/noisy frames |
| 10–60 second videos work best | Long videos slow down COLMAP |
| Avoid transparent/reflective objects | Hard for Gaussian splatting |

---

## File Outputs

After reconstruction, all files are saved in `outputs/<job-id>/`:

```
outputs/<job-id>/
├── video_info.json         # Video metadata
├── processed/              # COLMAP output (transforms.json, images/)
├── training/               # Nerfstudio training checkpoints
│   └── splatfacto/<date>/
│       └── config.yml
└── exports/
    ├── 3d/
    │   ├── splat.ply        # 3D Gaussian Splat file
    │   └── export_meta.json
    └── 2d/
        ├── render.mp4       # Novel-view render video
        ├── contact_sheet.png
        └── render_meta.json
```

---

## Troubleshooting

### `nvidia-smi not found`
Install NVIDIA drivers first:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### `ns-train not found` after setup
The Python venv must be active:
```bash
source activate.sh
which ns-train   # should print path inside ./venv/
```

### COLMAP fails with few camera poses
- Use a longer video with more overlap
- Film at a slower walking pace
- Ensure good lighting

### Out of GPU memory during training
Reduce iterations or use a smaller model config in `config.yaml`:
```yaml
nerfstudio:
  max_iterations: 10000    # reduce from 30000
```

### Backend not connecting
Make sure the backend is running on port 4000:
```bash
curl http://localhost:4000/api/health
# Expected: {"status":"ok"}
```

---

## Project Structure

```
video-to-3d-reconstruction/
├── setup.sh                 ← Run this first
├── activate.sh              ← Activate venv (auto-created by setup.sh)
├── requirements.txt         ← All Python packages
├── config.yaml              ← Pipeline configuration
│
├── venv/                    ← Python virtual environment (created by setup.sh)
│
├── python/                  ← Reconstruction pipeline
│   ├── pipeline.py          ← Main orchestrator
│   ├── video_processor.py   ← Video validation
│   ├── data_processor.py    ← COLMAP (ns-process-data)
│   ├── trainer.py           ← Splatfacto training (multi-GPU)
│   ├── exporter.py          ← 3D PLY export
│   └── renderer.py          ← 2D video render
│
├── backend/                 ← Node.js API server (port 4000)
│   ├── server.js
│   ├── routes/
│   └── services/
│
├── frontend/                ← React UI (port 5173)
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── UploadZone.jsx
│           ├── ModeSelector.jsx
│           ├── ProgressPanel.jsx
│           ├── Viewer3D.jsx     ← Interactive 3D Gaussian Splat viewer
│           ├── Viewer2D.jsx     ← Video player + contact sheet
│           └── ResultsPanel.jsx ← Side-by-side results
│
├── uploads/                 ← Uploaded videos (auto-created)
└── outputs/                 ← All reconstruction outputs (auto-created)
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| 3D Reconstruction | [Nerfstudio](https://docs.nerf.studio/) + Splatfacto (3DGS) |
| Camera Poses | COLMAP Structure-from-Motion |
| 3D Viewer | [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) + Three.js |
| Frontend | React 18 + Vite + framer-motion |
| Backend | Node.js + Express + WebSocket |
| Python env | Python 3.10 venv |
| GPU framework | PyTorch 2.1 + CUDA 11.8 |
