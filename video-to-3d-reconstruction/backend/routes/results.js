/**
 * routes/results.js — Serve job metadata and output files.
 */
const express = require("express");
const path = require("path");
const fs = require("fs");
const { getJob } = require("../services/jobManager");

const router = express.Router();

const OUTPUT_BASE = path.resolve(__dirname, "../../outputs");

// GET /api/results/:jobId — job metadata
router.get("/:jobId", (req, res) => {
  const job = getJob(req.params.jobId);
  if (!job) return res.status(404).json({ error: "Job not found." });

  const { jobId, status, currentStage, stages, results, error, mode, createdAt, logs } = job;
  res.json({
    jobId, status, currentStage, stages, results, error, mode, createdAt,
    recentLogs: (logs || []).slice(-20),
  });
});

// GET /api/results/:jobId/splat.ply — 3D Gaussian splat file
router.get("/:jobId/splat.ply", (req, res) => {
  const plyPath = path.join(OUTPUT_BASE, req.params.jobId, "exports", "3d", "splat.ply");
  if (!fs.existsSync(plyPath)) {
    return res.status(404).json({ error: "splat.ply not found. Run 3D export first." });
  }
  res.setHeader("Content-Type", "application/octet-stream");
  res.setHeader("Content-Disposition", `attachment; filename="splat.ply"`);
  res.setHeader("Access-Control-Allow-Origin", "*");
  fs.createReadStream(plyPath).pipe(res);
});

// GET /api/results/:jobId/render.mp4 — 2D render video
router.get("/:jobId/render.mp4", (req, res) => {
  const mp4Path = path.join(OUTPUT_BASE, req.params.jobId, "exports", "2d", "render.mp4");
  if (!fs.existsSync(mp4Path)) {
    return res.status(404).json({ error: "render.mp4 not found. Run 2D render first." });
  }

  const stat = fs.statSync(mp4Path);
  const fileSize = stat.size;
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    const chunkSize = end - start + 1;
    const stream = fs.createReadStream(mp4Path, { start, end });
    res.writeHead(206, {
      "Content-Range": `bytes ${start}-${end}/${fileSize}`,
      "Accept-Ranges": "bytes",
      "Content-Length": chunkSize,
      "Content-Type": "video/mp4",
      "Access-Control-Allow-Origin": "*",
    });
    stream.pipe(res);
  } else {
    res.writeHead(200, {
      "Content-Length": fileSize,
      "Content-Type": "video/mp4",
      "Accept-Ranges": "bytes",
      "Access-Control-Allow-Origin": "*",
    });
    fs.createReadStream(mp4Path).pipe(res);
  }
});

// GET /api/results/:jobId/contact_sheet.png
router.get("/:jobId/contact_sheet.png", (req, res) => {
  const imgPath = path.join(OUTPUT_BASE, req.params.jobId, "exports", "2d", "contact_sheet.png");
  if (!fs.existsSync(imgPath)) {
    return res.status(404).json({ error: "contact_sheet.png not found." });
  }
  res.setHeader("Content-Type", "image/png");
  res.setHeader("Access-Control-Allow-Origin", "*");
  fs.createReadStream(imgPath).pipe(res);
});

module.exports = router;
