/**
 * server.js — Express + WebSocket server for the reconstruction backend.
 * Port: 4000
 */
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const http = require("http");
const path = require("path");
const { WebSocketServer } = require("ws");
const { addWsClient } = require("./services/jobManager");

const uploadRouter = require("./routes/upload");
const reconstructRouter = require("./routes/reconstruct");
const resultsRouter = require("./routes/results");

const PORT = process.env.PORT || 4000;

const app = express();

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: "*" }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ── Static file serving ───────────────────────────────────────────────────────
// Serve the outputs directory for direct file access
app.use(
  "/outputs",
  express.static(path.resolve(__dirname, "../outputs"), {
    setHeaders: (res) => {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Accept-Ranges", "bytes");
    },
  })
);

// ── API Routes ────────────────────────────────────────────────────────────────
app.use("/api/upload", uploadRouter);
app.use("/api/reconstruct", reconstructRouter);
app.use("/api/results", resultsRouter);

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ── HTTP + WebSocket Server ───────────────────────────────────────────────────
const server = http.createServer(app);

const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", (ws, req) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const jobId = url.searchParams.get("jobId");

  if (!jobId) {
    ws.close(1008, "Missing jobId query param");
    return;
  }

  console.log(`[WS] Client connected for job: ${jobId}`);
  addWsClient(jobId, ws);

  // Send ping every 20s to keep connection alive
  const ping = setInterval(() => {
    if (ws.readyState === 1) ws.ping();
  }, 20000);

  ws.on("close", () => {
    clearInterval(ping);
    console.log(`[WS] Client disconnected for job: ${jobId}`);
  });

  ws.on("error", (err) => {
    console.error(`[WS] Error for job ${jobId}:`, err.message);
    clearInterval(ping);
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log("");
  console.log("  ╔══════════════════════════════════════════════════╗");
  console.log("  ║   3D/2D Reconstruction Backend  •  Ready         ║");
  console.log(`  ║   HTTP:  http://localhost:${PORT}                  ║`);
  console.log(`  ║   WS:    ws://localhost:${PORT}/ws?jobId=<id>      ║`);
  console.log("  ╚══════════════════════════════════════════════════╝");
  console.log("");
});

server.on("error", (err) => {
  console.error("[Server Error]", err);
  process.exit(1);
});
