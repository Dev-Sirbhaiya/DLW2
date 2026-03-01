import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, XCircle, Loader, Clock, Terminal } from "lucide-react";

const STAGE_INFO = {
  1: { label: "Validate Video",               color: "#00d4ff" },
  2: { label: "COLMAP Structure-from-Motion", color: "#7b61ff" },
  3: { label: "Train Splatfacto",             color: "#00ff88" },
  4: { label: "Export 3D Gaussian Splat",     color: "#00d4ff" },
  5: { label: "Render 2D Novel Views",        color: "#7b61ff" },
  6: { label: "Complete",                      color: "#00ff88" },
};

const TOTAL_STAGES = 6;

function StageIcon({ status }) {
  if (status === "done") return <CheckCircle size={18} color="#00ff88" />;
  if (status === "error") return <XCircle size={18} color="#ff3b5c" />;
  if (status === "running") {
    return (
      <div style={{ animation: "spin 1s linear infinite", display: "flex" }}>
        <Loader size={18} color="#00d4ff" />
      </div>
    );
  }
  if (status === "skipped") return <div style={{ width: 18, height: 18, borderRadius: "50%", background: "rgba(255,255,255,0.12)" }} />;
  return <Clock size={18} color="rgba(255,255,255,0.2)" />;
}

export default function ProgressPanel({ jobId, mode, onDone }) {
  const [stages, setStages] = useState({});     // { [stageNum]: { name, status } }
  const [logs, setLogs] = useState([]);
  const [jobStatus, setJobStatus] = useState("running");
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const logEndRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:4000/ws?jobId=${jobId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }

      switch (msg.type) {
        case "stage":
          setStages((prev) => ({
            ...prev,
            [msg.stage]: { name: msg.name, status: msg.status },
          }));
          break;
        case "log":
          setLogs((prev) => [...prev.slice(-299), msg.line]);
          break;
        case "done":
          setJobStatus("done");
          setTimeout(() => onDone?.(), 1500);
          break;
        case "error":
          setJobStatus("error");
          setError(msg.message);
          break;
        default: break;
      }
    };

    ws.onerror = () => {
      setJobStatus("error");
      setError("WebSocket connection failed. Is the backend running?");
    };

    return () => ws.close();
  }, [jobId, onDone]);

  const doneCount = Object.values(stages).filter((s) => s.status === "done").length;
  const pct = Math.round((doneCount / TOTAL_STAGES) * 100);

  return (
    <div style={{ maxWidth: 760, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 36 }}>
        <motion.div
          animate={jobStatus === "running" ? { rotate: 360 } : {}}
          transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
          style={{ fontSize: 48, display: "inline-block", marginBottom: 12 }}
        >
          {jobStatus === "done" ? "✅" : jobStatus === "error" ? "❌" : "⚙️"}
        </motion.div>
        <h2 style={{
          fontFamily: "var(--font-heading)", fontSize: 26, fontWeight: 700,
          color: "var(--text-primary)", marginBottom: 6,
        }}>
          {jobStatus === "done" ? "Reconstruction Complete!" :
           jobStatus === "error" ? "Reconstruction Failed" : "Processing…"}
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: 14 }}>
          Mode: <strong style={{ color: "#00d4ff" }}>{mode === "both" ? "3D + 2D" : mode === "3d" ? "3D Only" : "2D Only"}</strong>
          {" · "}Job: <code style={{ color: "#7b61ff", fontSize: 12 }}>{jobId?.slice(0, 8)}…</code>
        </p>
      </div>

      {/* Overall progress bar */}
      <div className="glass" style={{ padding: 20, marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>Overall Progress</span>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#00d4ff" }}>{pct}%</span>
        </div>
        <div style={{
          height: 8, borderRadius: 4,
          background: "rgba(255,255,255,0.06)",
          overflow: "hidden",
        }}>
          <motion.div
            animate={{ width: `${pct}%` }}
            transition={{ type: "spring", stiffness: 60, damping: 15 }}
            style={{
              height: "100%", borderRadius: 4,
              background: jobStatus === "error"
                ? "#ff3b5c"
                : "linear-gradient(90deg, #00d4ff, #7b61ff)",
              boxShadow: jobStatus === "done" ? "0 0 12px rgba(0,212,255,0.5)" : "none",
            }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{doneCount} / {TOTAL_STAGES} stages</span>
          {jobStatus === "running" && (
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              This may take 15-30 min on a modern GPU…
            </span>
          )}
        </div>
      </div>

      {/* Stages list */}
      <div className="glass" style={{ padding: 20, marginBottom: 20 }}>
        {Array.from({ length: TOTAL_STAGES }, (_, i) => i + 1).map((n) => {
          const info = STAGE_INFO[n];
          const stage = stages[n] || {};
          const status = stage.status || "pending";
          const isActive = status === "running";

          return (
            <motion.div
              key={n}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: n * 0.05 }}
              style={{
                display: "flex", alignItems: "center", gap: 14,
                padding: "12px 14px",
                borderRadius: 10,
                background: isActive ? `${info.color}0a` : "transparent",
                border: isActive ? `1px solid ${info.color}22` : "1px solid transparent",
                marginBottom: n < TOTAL_STAGES ? 6 : 0,
                transition: "all 0.3s ease",
              }}
            >
              {/* Connector line */}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <StageIcon status={status} />
                {n < TOTAL_STAGES && (
                  <div style={{
                    width: 2, height: 18, marginTop: 4,
                    background: status === "done"
                      ? "rgba(0,255,136,0.3)"
                      : "rgba(255,255,255,0.06)",
                    borderRadius: 1,
                    transition: "background 0.4s",
                  }} />
                )}
              </div>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{
                    fontSize: 14, fontWeight: isActive ? 600 : 400,
                    color: isActive ? info.color :
                           status === "done" ? "var(--text-primary)" :
                           status === "error" ? "#ff3b5c" : "var(--text-muted)",
                    transition: "color 0.3s",
                  }}>
                    {info.label}
                  </span>
                  {status === "skipped" && (
                    <span style={{
                      fontSize: 11, color: "var(--text-muted)",
                      background: "rgba(255,255,255,0.05)",
                      borderRadius: 4, padding: "1px 6px",
                    }}>
                      skipped
                    </span>
                  )}
                </div>
              </div>

              <span style={{
                fontSize: 12, color: "var(--text-muted)",
                textTransform: "capitalize",
              }}>
                {status}
              </span>
            </motion.div>
          );
        })}
      </div>

      {/* Error message */}
      {error && (
        <div style={{
          background: "rgba(255,59,92,0.08)",
          border: "1px solid rgba(255,59,92,0.3)",
          borderRadius: 12, padding: "14px 18px",
          color: "#ff3b5c", fontSize: 14, marginBottom: 20,
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Live logs */}
      <div className="glass" style={{ overflow: "hidden" }}>
        <div style={{
          padding: "12px 16px",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          display: "flex", alignItems: "center", gap: 8,
          color: "var(--text-muted)", fontSize: 13,
        }}>
          <Terminal size={14} />
          Live Pipeline Log
          <span style={{
            marginLeft: "auto", fontSize: 11,
            background: "rgba(0,212,255,0.1)",
            border: "1px solid rgba(0,212,255,0.2)",
            color: "#00d4ff", borderRadius: 4, padding: "1px 6px",
          }}>
            {logs.length} lines
          </span>
        </div>
        <div style={{
          height: 220, overflowY: "auto", padding: "12px 16px",
          background: "rgba(0,0,0,0.3)",
          fontFamily: "'Courier New', monospace", fontSize: 12,
          lineHeight: 1.7, color: "#8b9ab5",
        }}>
          {logs.length === 0 ? (
            <span style={{ color: "rgba(255,255,255,0.15)" }}>Waiting for pipeline output…</span>
          ) : (
            logs.map((line, i) => (
              <div key={i} style={{
                color: line.includes("[stderr]") || line.toLowerCase().includes("error")
                  ? "#ff6b6b"
                  : line.toLowerCase().includes("done") || line.toLowerCase().includes("complete")
                  ? "#00ff88"
                  : "#8b9ab5",
              }}>
                <span style={{ color: "rgba(255,255,255,0.2)", marginRight: 8, userSelect: "none" }}>
                  {String(i + 1).padStart(3, "0")}
                </span>
                {line}
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  );
}
