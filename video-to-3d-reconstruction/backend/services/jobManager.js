/**
 * jobManager.js — In-memory job store and WebSocket broadcaster.
 */

const jobs = new Map();

/**
 * Create or reset a job entry.
 * @param {string} jobId
 * @param {object} meta - e.g. { video, mode, maxIterations, numFrames }
 */
function createJob(jobId, meta = {}) {
  jobs.set(jobId, {
    jobId,
    status: "created",
    currentStage: 0,
    stages: {},
    logs: [],
    results: {},
    error: null,
    createdAt: Date.now(),
    ...meta,
  });
}

/**
 * Get a job by ID. Returns null if not found.
 */
function getJob(jobId) {
  return jobs.get(jobId) || null;
}

/**
 * Update fields on a job.
 */
function updateJob(jobId, patch) {
  const job = jobs.get(jobId);
  if (!job) return;
  Object.assign(job, patch);
}

// WebSocket clients keyed by jobId
const wsClients = new Map(); // jobId → Set<WebSocket>

/**
 * Register a WebSocket client for a job.
 */
function addWsClient(jobId, ws) {
  if (!wsClients.has(jobId)) wsClients.set(jobId, new Set());
  wsClients.get(jobId).add(ws);

  ws.on("close", () => {
    const set = wsClients.get(jobId);
    if (set) set.delete(ws);
  });
}

/**
 * Send a JSON event to all WS clients for a job.
 */
function broadcast(jobId, event) {
  const set = wsClients.get(jobId);
  if (!set || set.size === 0) return;
  const msg = JSON.stringify(event);
  for (const ws of set) {
    try {
      if (ws.readyState === 1 /* OPEN */) ws.send(msg);
    } catch (_) {}
  }
}

/**
 * Append a log line to a job (max 500 lines).
 */
function appendLog(jobId, line) {
  const job = jobs.get(jobId);
  if (!job) return;
  job.logs.push(line);
  if (job.logs.length > 500) job.logs = job.logs.slice(-500);
}

module.exports = { createJob, getJob, updateJob, addWsClient, broadcast, appendLog };
