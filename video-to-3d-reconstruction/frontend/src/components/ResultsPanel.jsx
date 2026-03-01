import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Box, Film, Layers, Download, RefreshCw, CheckCircle } from "lucide-react";
import Viewer3D from "./Viewer3D.jsx";
import Viewer2D from "./Viewer2D.jsx";

export default function ResultsPanel({ jobId, mode, onReset }) {
  const [activeTab, setActiveTab] = useState(mode === "2d" ? "2d" : "3d");
  const [layout, setLayout] = useState(mode === "both" ? "split" : "single");
  const [meta, setMeta] = useState(null);

  // Load job metadata
  useEffect(() => {
    fetch(`/api/results/${jobId}`)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => {});
  }, [jobId]);

  const show3d = mode === "3d" || mode === "both";
  const show2d = mode === "2d" || mode === "both";

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto" }}>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          display: "flex", alignItems: "center", flexWrap: "wrap", gap: 12,
          marginBottom: 28,
        }}
      >
        <div style={{
          width: 40, height: 40, borderRadius: 10,
          background: "linear-gradient(135deg, #00ff88, #00d4ff)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 20, boxShadow: "0 0 20px rgba(0,255,136,0.3)",
        }}>
          <CheckCircle size={22} color="#000" />
        </div>
        <div>
          <h2 style={{
            fontFamily: "var(--font-heading)", fontSize: 22, fontWeight: 700,
            color: "var(--text-primary)", lineHeight: 1.2,
          }}>
            Reconstruction Complete
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            Job <code style={{ color: "#7b61ff" }}>{jobId?.slice(0, 8)}…</code>
            {" · "}Mode: <strong style={{ color: "#00d4ff" }}>
              {mode === "both" ? "3D + 2D" : mode === "3d" ? "3D Only" : "2D Only"}
            </strong>
          </p>
        </div>

        {/* Controls */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, flexWrap: "wrap" }}>
          {/* Layout toggle (both mode only) */}
          {mode === "both" && (
            <div style={{
              display: "flex", gap: 4,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8, padding: 4,
            }}>
              {[
                { id: "split",  label: "Split",   icon: Layers },
                { id: "3d",     label: "3D",      icon: Box },
                { id: "2d",     label: "2D",      icon: Film },
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => { setLayout(id); if (id !== "split") setActiveTab(id); }}
                  style={{
                    padding: "5px 12px",
                    borderRadius: 6,
                    background: (layout === id) ? "rgba(0,212,255,0.15)" : "transparent",
                    border: (layout === id) ? "1px solid rgba(0,212,255,0.3)" : "1px solid transparent",
                    color: (layout === id) ? "#00d4ff" : "var(--text-muted)",
                    fontSize: 12, fontWeight: 500,
                    display: "flex", alignItems: "center", gap: 5,
                  }}
                >
                  <Icon size={13} />
                  {label}
                </button>
              ))}
            </div>
          )}

          {/* Download all */}
          <DownloadMenu jobId={jobId} mode={mode} />

          {/* New reconstruction */}
          <button
            onClick={onReset}
            style={{
              padding: "8px 14px",
              borderRadius: 8,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--text-secondary)",
              fontSize: 13, fontWeight: 500,
              display: "flex", alignItems: "center", gap: 6,
            }}
          >
            <RefreshCw size={14} /> New
          </button>
        </div>
      </motion.div>

      {/* ── Viewers ──────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">

        {/* SPLIT view — both 3d+2d side by side */}
        {layout === "split" && mode === "both" && (
          <motion.div
            key="split"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
              minHeight: 520,
            }}
          >
            <ViewerCard title="3D Gaussian Splat" icon={Box} color="#00d4ff" badge="Splatfacto">
              <Viewer3D jobId={jobId} />
            </ViewerCard>
            <ViewerCard title="2D Novel Views" icon={Film} color="#7b61ff" badge="ns-render">
              <Viewer2D jobId={jobId} />
            </ViewerCard>
          </motion.div>
        )}

        {/* Single 3D */}
        {(layout === "3d" || (layout === "single" && mode === "3d")) && (
          <motion.div
            key="single-3d"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ViewerCard title="3D Gaussian Splat" icon={Box} color="#00d4ff" badge="Splatfacto" fullHeight>
              <Viewer3D jobId={jobId} />
            </ViewerCard>
          </motion.div>
        )}

        {/* Single 2D */}
        {(layout === "2d" || (layout === "single" && mode === "2d")) && (
          <motion.div
            key="single-2d"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ViewerCard title="2D Novel Views" icon={Film} color="#7b61ff" badge="ns-render" fullHeight>
              <Viewer2D jobId={jobId} />
            </ViewerCard>
          </motion.div>
        )}

      </AnimatePresence>

      {/* ── Stats bar ────────────────────────────────────────────────── */}
      {meta && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass"
          style={{
            marginTop: 16, padding: "14px 20px",
            display: "flex", flexWrap: "wrap", gap: 24,
          }}
        >
          {[
            { label: "Mode", value: mode === "both" ? "3D + 2D" : mode === "3d" ? "3D Splat" : "2D Render" },
            meta.results?.num_gaussians && { label: "Gaussians", value: Number(meta.results.num_gaussians).toLocaleString() },
            meta.results?.size_mb && { label: "PLY Size", value: `${meta.results.size_mb} MB` },
          ].filter(Boolean).map(({ label, value }) => (
            <div key={label}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 2 }}>{label}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{value}</div>
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}

/* ── Sub-components ───────────────────────────────────────────────────────── */

function ViewerCard({ title, icon: Icon, color, badge, children, fullHeight }) {
  return (
    <div className="glass" style={{
      display: "flex", flexDirection: "column",
      minHeight: fullHeight ? 580 : 0,
      overflow: "hidden",
    }}>
      {/* Card header */}
      <div style={{
        padding: "14px 18px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        display: "flex", alignItems: "center", gap: 10,
        background: `linear-gradient(90deg, ${color}0d, transparent)`,
      }}>
        <div style={{
          width: 30, height: 30, borderRadius: 8,
          background: `${color}1a`,
          border: `1px solid ${color}30`,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Icon size={16} color={color} />
        </div>
        <span style={{ fontWeight: 600, fontSize: 15, color: "var(--text-primary)" }}>
          {title}
        </span>
        <span style={{
          marginLeft: "auto",
          fontSize: 11, fontWeight: 600,
          background: `${color}18`,
          border: `1px solid ${color}30`,
          color, borderRadius: 5, padding: "2px 8px",
          letterSpacing: "0.04em",
        }}>
          {badge}
        </span>
      </div>

      {/* Viewer */}
      <div style={{ flex: 1, padding: 16, minHeight: fullHeight ? 500 : 400 }}>
        {children}
      </div>
    </div>
  );
}

function DownloadMenu({ jobId, mode }) {
  const [open, setOpen] = useState(false);

  const items = [
    mode !== "2d" && { label: "splat.ply", url: `/api/results/${jobId}/splat.ply`, color: "#00d4ff" },
    mode !== "3d" && { label: "render.mp4", url: `/api/results/${jobId}/render.mp4`, color: "#7b61ff" },
    mode !== "3d" && { label: "contact_sheet.png", url: `/api/results/${jobId}/contact_sheet.png`, color: "#7b61ff" },
  ].filter(Boolean);

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          padding: "8px 14px",
          borderRadius: 8,
          background: "rgba(0,212,255,0.08)",
          border: "1px solid rgba(0,212,255,0.2)",
          color: "#00d4ff",
          fontSize: 13, fontWeight: 500,
          display: "flex", alignItems: "center", gap: 6,
        }}
      >
        <Download size={14} /> Download ▾
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            style={{
              position: "absolute", top: "calc(100% + 6px)", right: 0,
              background: "#0a1628",
              border: "1px solid rgba(0,212,255,0.2)",
              borderRadius: 10, overflow: "hidden",
              zIndex: 200, minWidth: 180,
              boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            }}
            onMouseLeave={() => setOpen(false)}
          >
            {items.map(({ label, url, color }) => (
              <a
                key={label}
                href={url}
                download={label}
                onClick={() => setOpen(false)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "10px 16px",
                  color: "var(--text-secondary)",
                  fontSize: 13,
                  textDecoration: "none",
                  borderBottom: "1px solid rgba(255,255,255,0.04)",
                  transition: "background 0.15s",
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = "rgba(0,212,255,0.06)"}
                onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
              >
                <span style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: color, flexShrink: 0,
                }} />
                {label}
              </a>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
