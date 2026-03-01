import React, { useState } from "react";
import { motion } from "framer-motion";
import { Box, Film, Layers, ChevronLeft, ChevronRight, Settings, Play } from "lucide-react";

const MODES = [
  {
    id: "3d",
    label: "3D Reconstruction",
    icon: Box,
    color: "#00d4ff",
    description: "Export a 3D Gaussian Splat (.ply) viewable in any 3DGS viewer. Full scene geometry with view-dependent colour.",
    badge: "Splatfacto",
  },
  {
    id: "2d",
    label: "2D Novel Views",
    icon: Film,
    color: "#7b61ff",
    description: "Synthesize a spiral camera path video around the scene. Photo-realistic novel-view renders as MP4.",
    badge: "ns-render",
  },
  {
    id: "both",
    label: "Both (Recommended)",
    icon: Layers,
    color: "#00ff88",
    description: "Generate the 3D Gaussian Splat AND the 2D render video in one pipeline run.",
    badge: "Full Pipeline",
  },
];

export default function ModeSelector({
  mode, setMode,
  maxIterations, setMaxIterations,
  numFrames, setNumFrames,
  onStart, onBack,
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div style={{ maxWidth: 800, margin: "0 auto" }}>
      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 36 }}>
        <h2 style={{
          fontFamily: "var(--font-heading)", fontSize: 28, fontWeight: 700,
          color: "var(--text-primary)", marginBottom: 8,
        }}>
          Choose Reconstruction Mode
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: 15 }}>
          Select what you want to generate from your video.
        </p>
      </div>

      {/* Mode Cards */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 16,
        marginBottom: 28,
      }}>
        {MODES.map((m) => {
          const Icon = m.icon;
          const selected = mode === m.id;
          return (
            <motion.div
              key={m.id}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setMode(m.id)}
              style={{
                padding: 24,
                borderRadius: "var(--radius-lg)",
                border: `2px solid ${selected ? m.color : "rgba(255,255,255,0.07)"}`,
                background: selected
                  ? `linear-gradient(135deg, ${m.color}14, ${m.color}06)`
                  : "rgba(10,22,40,0.6)",
                cursor: "pointer",
                backdropFilter: "blur(12px)",
                boxShadow: selected ? `0 0 24px ${m.color}25` : "none",
                transition: "all 0.25s ease",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Glow background on selected */}
              {selected && (
                <div style={{
                  position: "absolute", inset: 0, borderRadius: "var(--radius-lg)",
                  background: `radial-gradient(circle at 30% 30%, ${m.color}0f, transparent 70%)`,
                  pointerEvents: "none",
                }} />
              )}

              {/* Badge */}
              <div style={{
                display: "inline-block",
                background: selected ? `${m.color}22` : "rgba(255,255,255,0.06)",
                border: `1px solid ${selected ? m.color + "44" : "rgba(255,255,255,0.08)"}`,
                borderRadius: 6, padding: "2px 8px",
                fontSize: 11, fontWeight: 600,
                color: selected ? m.color : "var(--text-muted)",
                marginBottom: 14, letterSpacing: "0.05em",
              }}>
                {m.badge}
              </div>

              {/* Icon */}
              <div style={{
                width: 48, height: 48, borderRadius: 14,
                background: selected ? `${m.color}22` : "rgba(255,255,255,0.05)",
                border: `1px solid ${selected ? m.color + "33" : "rgba(255,255,255,0.06)"}`,
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: 14,
                transition: "all 0.25s",
              }}>
                <Icon size={24} color={selected ? m.color : "#8b9ab5"} />
              </div>

              <h3 style={{
                fontFamily: "var(--font-heading)", fontSize: 16, fontWeight: 600,
                color: selected ? m.color : "var(--text-primary)",
                marginBottom: 8, transition: "color 0.25s",
              }}>
                {m.label}
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: 13, lineHeight: 1.6 }}>
                {m.description}
              </p>

              {/* Selected check */}
              {selected && (
                <div style={{
                  position: "absolute", top: 14, right: 14,
                  width: 20, height: 20, borderRadius: "50%",
                  background: m.color,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, color: "#000", fontWeight: 700,
                }}>
                  ✓
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Advanced Settings */}
      <div className="glass" style={{ marginBottom: 28, overflow: "hidden" }}>
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          style={{
            width: "100%", padding: "16px 20px",
            background: "transparent", color: "var(--text-secondary)",
            display: "flex", alignItems: "center", gap: 8, fontSize: 14, fontWeight: 500,
          }}
        >
          <Settings size={16} />
          Advanced Settings
          <span style={{
            marginLeft: "auto", fontSize: 12,
            transform: showAdvanced ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            display: "inline-block",
          }}>
            ▶
          </span>
        </button>

        {showAdvanced && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            style={{ padding: "0 20px 20px", borderTop: "1px solid rgba(255,255,255,0.05)" }}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 16 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>
                  Training Iterations
                </span>
                <input
                  type="number"
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                  min={1000} max={100000} step={1000}
                  style={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 8, padding: "8px 12px",
                    fontSize: 14, color: "var(--text-primary)",
                  }}
                />
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  30000 = full quality · 10000 = quick test
                </span>
              </label>

              <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>
                  COLMAP Frame Target
                </span>
                <input
                  type="number"
                  value={numFrames}
                  onChange={(e) => setNumFrames(Number(e.target.value))}
                  min={50} max={1000} step={50}
                  style={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 8, padding: "8px 12px",
                    fontSize: 14, color: "var(--text-primary)",
                  }}
                />
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  Frames extracted from video for COLMAP SfM
                </span>
              </label>
            </div>
          </motion.div>
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: 12 }}>
        <button
          onClick={onBack}
          style={{
            padding: "13px 20px", borderRadius: 12, fontWeight: 500, fontSize: 14,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.08)",
            color: "var(--text-secondary)",
            display: "flex", alignItems: "center", gap: 6,
          }}
        >
          <ChevronLeft size={16} /> Back
        </button>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={onStart}
          style={{
            flex: 1, padding: "14px 24px",
            borderRadius: 12, fontWeight: 700, fontSize: 15,
            background: "linear-gradient(135deg, #00d4ff, #7b61ff)",
            color: "#fff",
            boxShadow: "0 4px 24px rgba(0,212,255,0.35)",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
          }}
        >
          <Play size={18} fill="#fff" />
          Start Reconstruction
        </motion.button>
      </div>
    </div>
  );
}
