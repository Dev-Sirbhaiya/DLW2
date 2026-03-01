/**
 * routes/upload.js — POST /api/upload
 * Accepts a video file, saves it, and returns { jobId, filename }.
 */
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { v4: uuidv4 } = require("uuid");
const { createJob } = require("../services/jobManager");

const router = express.Router();

const UPLOAD_BASE = path.resolve(__dirname, "../../uploads");
const MAX_FILE_SIZE_MB = parseInt(process.env.MAX_UPLOAD_MB || "2000", 10);

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const jobId = uuidv4();
    req.jobId = jobId;
    const dir = path.join(UPLOAD_BASE, jobId);
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    // Preserve original extension
    const ext = path.extname(file.originalname).toLowerCase() || ".mp4";
    cb(null, `input_video${ext}`);
  },
});

const ALLOWED_TYPES = new Set([
  "video/mp4", "video/quicktime", "video/x-msvideo",
  "video/x-matroska", "video/webm", "video/x-m4v",
]);

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (ALLOWED_TYPES.has(file.mimetype)) return cb(null, true);
    const ext = path.extname(file.originalname).toLowerCase();
    const allowedExts = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];
    if (allowedExts.includes(ext)) return cb(null, true);
    cb(new Error(`Unsupported file type: ${file.mimetype}`));
  },
});

router.post("/", upload.single("video"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No video file provided." });
  }

  const jobId = req.jobId;

  // Create job record
  createJob(jobId, {
    originalName: req.file.originalname,
    uploadedPath: req.file.path,
    status: "uploaded",
  });

  res.json({
    jobId,
    filename: req.file.originalname,
    savedPath: req.file.path,
    sizeBytes: req.file.size,
    sizeMB: (req.file.size / 1e6).toFixed(1),
  });
});

// Error handler for multer
router.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({ error: `File too large (max ${MAX_FILE_SIZE_MB} MB)` });
    }
  }
  res.status(400).json({ error: err.message });
});

module.exports = router;
