/**
 * routes/reconstruct.js — POST /api/reconstruct
 * Kicks off the Python pipeline for a previously uploaded video.
 */
const express = require("express");
const { getJob, updateJob } = require("../services/jobManager");
const { start } = require("../services/pipelineService");

const router = express.Router();

router.post("/", (req, res) => {
  const { jobId, mode = "both", maxIterations = 30000, numFrames = 300 } = req.body;

  if (!jobId) {
    return res.status(400).json({ error: "jobId is required." });
  }

  const job = getJob(jobId);
  if (!job) {
    return res.status(404).json({ error: `Job '${jobId}' not found. Upload a video first.` });
  }

  if (job.status === "running") {
    return res.status(409).json({ error: "Job is already running." });
  }

  const validModes = ["3d", "2d", "both"];
  if (!validModes.includes(mode)) {
    return res.status(400).json({ error: `Invalid mode '${mode}'. Must be one of: ${validModes.join(", ")}` });
  }

  updateJob(jobId, { mode, maxIterations, numFrames });

  start(jobId, {
    videoPath: job.uploadedPath,
    mode,
    maxIterations: parseInt(maxIterations, 10),
    numFrames: parseInt(numFrames, 10),
  });

  res.json({ jobId, status: "started", mode });
});

module.exports = router;
