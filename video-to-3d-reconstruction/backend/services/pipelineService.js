/**
 * pipelineService.js — Spawns the Python pipeline subprocess and routes events
 * to the jobManager for WebSocket broadcasting.
 */
const { spawn } = require("child_process");
const path = require("path");
const { broadcast, updateJob, appendLog, getJob } = require("./jobManager");

const PYTHON_DIR = path.resolve(__dirname, "../../python");
const OUTPUT_BASE = path.resolve(__dirname, "../../outputs");

/**
 * Start the reconstruction pipeline for a job.
 * @param {string} jobId
 * @param {object} options - { videoPath, mode, maxIterations, numFrames }
 */
function start(jobId, options) {
  const { videoPath, mode = "both", maxIterations = 30000, numFrames = 300 } = options;
  const outputDir = path.join(OUTPUT_BASE, jobId);

  const args = [
    path.join(PYTHON_DIR, "pipeline.py"),
    "--video", videoPath,
    "--output-dir", outputDir,
    "--mode", mode,
    "--max-iterations", String(maxIterations),
    "--num-frames", String(numFrames),
  ];

  updateJob(jobId, { status: "running", outputDir });
  broadcast(jobId, { type: "status", status: "running" });

  // Prefer 'python3' on Ubuntu; fallback to 'python'
  const pythonBin = process.env.PYTHON_BIN || "python3";
  const proc = spawn(pythonBin, args, {
    cwd: PYTHON_DIR,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  let lineBuffer = "";

  proc.stdout.on("data", (chunk) => {
    lineBuffer += chunk.toString();
    const lines = lineBuffer.split("\n");
    lineBuffer = lines.pop(); // keep incomplete last line

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let parsed = null;
      try {
        parsed = JSON.parse(trimmed);
      } catch (_) {
        // Not JSON — treat as plain log
        appendLog(jobId, trimmed);
        broadcast(jobId, { type: "log", line: trimmed });
        continue;
      }

      handlePipelineEvent(jobId, parsed);
    }
  });

  proc.stderr.on("data", (chunk) => {
    const line = chunk.toString().trim();
    if (line) {
      appendLog(jobId, `[stderr] ${line}`);
      broadcast(jobId, { type: "log", line: `[stderr] ${line}` });
    }
  });

  proc.on("close", (code) => {
    if (code === 0) {
      updateJob(jobId, { status: "done" });
      broadcast(jobId, { type: "done", jobId });
    } else {
      const job = getJob(jobId);
      const errMsg = job?.error || `Pipeline exited with code ${code}`;
      updateJob(jobId, { status: "error", error: errMsg });
      broadcast(jobId, { type: "error", message: errMsg });
    }
  });

  proc.on("error", (err) => {
    const message = `Failed to start Python process: ${err.message}`;
    updateJob(jobId, { status: "error", error: message });
    broadcast(jobId, { type: "error", message });
  });
}

/**
 * Route a parsed pipeline JSON event to job state + WebSocket.
 */
function handlePipelineEvent(jobId, event) {
  switch (event.event) {
    case "start":
      broadcast(jobId, { type: "start", ...event });
      break;

    case "stage": {
      const job = getJob(jobId);
      if (job) {
        job.stages[event.stage] = { name: event.name, status: event.status };
        if (event.status === "running") job.currentStage = event.stage;
      }
      broadcast(jobId, { type: "stage", stage: event.stage, name: event.name, status: event.status });
      break;
    }

    case "log":
      appendLog(jobId, event.line);
      broadcast(jobId, { type: "log", line: event.line });
      break;

    case "result_3d":
      updateJob(jobId, { results: { ...getJob(jobId)?.results, ply_path: event.ply_path } });
      broadcast(jobId, { type: "result_3d", ply_path: event.ply_path });
      break;

    case "result_2d":
      updateJob(jobId, { results: { ...getJob(jobId)?.results, ...event } });
      broadcast(jobId, { type: "result_2d", ...event });
      break;

    case "done":
      updateJob(jobId, { status: "done" });
      broadcast(jobId, { type: "done", jobId });
      break;

    case "error":
      updateJob(jobId, { status: "error", error: event.message });
      broadcast(jobId, { type: "error", message: event.message });
      break;

    default:
      broadcast(jobId, { type: "log", line: JSON.stringify(event) });
  }
}

module.exports = { start };
